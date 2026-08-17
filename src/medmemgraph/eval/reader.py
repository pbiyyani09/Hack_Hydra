"""eval/reader.py — the Chain-of-Note extract-then-reason reader.

Answers a question over **already retrieved** evidence (`RetrieveItem`s —
CONTRACT 2 in `contracts.py`, produced by `retrieve()`/`mock_retrieve()`).
No extra retrieval happens inside this module — `read()` never calls
`retrieve()` or the loader, by construction (see the module-level "no
retrieval inside read()" note below, enforced by
`tests/test_reader.py::test_read_never_calls_retrieve`).

**Real inference is the default path, dry_run is opt-in, never the silent
default.** Real calls route through `medmemgraph.llm.complete()` — the
single seam every LLM-dependent module in this project shares (see
`llm.py`'s own module docstring) — using `model=DEFAULT_READER_MODEL`
(`llm.ANSWER_MODEL`, OpenAI `gpt-4.1-mini` by default; never a hardcoded
model string here). This module previously imported `anthropic` directly
and only ever ran for real if `ANTHROPIC_API_KEY` happened to be set,
which it never was in this project's actual environment (`anthropic` was
removed from dependencies once the `llm.py` seam replaced it) — so every
prior run of this module silently fell back to the offline stub with no
error, and every number it produced meant nothing. That is no longer
possible: `read(..., dry_run=False)` (the default) always attempts a real
`llm.complete()` call, and if no API key is configured,
`llm.MissingAPIKeyError` raises immediately, naming exactly which env var
/ `.env` key to set — it does not quietly degrade. The deterministic
offline stub (`_stub_complete`, below) runs **only** when a caller
explicitly passes `dry_run=True` (what the harness's `--dry-run` flag
threads through `ReaderAnswerer.dry_run`) — every stub-produced answer
stays visibly labelled (`"[dry-run stub, ...]"` prefixes) so a results
table can never mistake a stub answer for a real one.

Two things live here at once, per the story's framing ("the single
best-value item in the whole survey set", `literature/15-query-
understanding-and-context-compression.md` §10/Recommendations table):

1. **Two reading modes**, `"direct"` (read the evidence and just answer —
   the ablation baseline) and `"chain_of_note"` (extract a one-sentence
   note per item, marking it relevant/IRRELEVANT, then reason from the
   notes only). Both are first-class so the harness can A/B them on this
   project's own data rather than assuming the literature's numbers
   transfer (`literature/15` closing paragraph: "never measured together,
   on this project's own MedLoCoMo-shaped data").

   Mechanism, quoted directly from the primary source via the survey:
   Chain-of-Note "instructs the LLM to first extract information from each
   memory item and then reason based on these notes" (LongMemEval, §5.5,
   cross-verified `literature/15` R-QCC-042 / `literature/02` R-MEM-012).
   Magnitude: "even with perfect retrieval, a suboptimal reading strategy
   results in up to a 10-point absolute performance drop ... for GPT-4o"
   (LongMemEval §5.5, Fig. 6, R-QCC-043); the original Chain-of-Note paper
   (Yu, Zhang, Pan, Ma, Wang, Yu — "Chain-of-Note: Enhancing Robustness in
   Retrieval-Augmented Language Models", EMNLP 2023, arXiv:2311.09210)
   separately reports "+7.9 in EM score given entirely noisy retrieved
   documents" and "+10.5 in rejection rates for real-time questions that
   fall outside the pre-training knowledge scope" (R-QCC-044) — the second
   number is why this module treats abstention as a first-class structured
   output (`Answer.abstained: bool`), not a side effect of parsing prose.

2. **Three swappable context-rendering strategies** (`json`, `prose`,
   `shuffled`), because `literature/15` §10 surfaces a genuine, unresolved
   tension between two well-evidenced findings on different task shapes:
   LongMemEval finds JSON-structured context helps reading accuracy
   (R-QCC-043 above), while Chroma Research's 18-frontier-model "Context
   Rot" study finds "models perform worse when the haystack preserves a
   logical flow of ideas. Shuffling the haystack and removing local
   coherence consistently improves performance ... across all 18 models
   tested" (R-QCC-037). Neither survey adjudicates the conflict for this
   project's bounded, pre-filtered, graph-traversal-shaped evidence sets —
   so this module does not pick a winner; it implements all three and lets
   the harness measure which one wins on MedLoCoMo data (see the return
   note / docs/algorithms/chain-of-note-reader.md for the A/B result).

Timestamp/provenance rendering: `RetrieveItem` (the frozen CONTRACT 2 shape
in `contracts.py`) carries `session_id`/`turn_ids` but no separate `time`
field. Every rendering strategy below surfaces `session_id`+`turn_ids`
explicitly regardless (the provenance the contract *does* carry), and
additionally detects and surfaces a per-item timestamp when the upstream
producer already serializes one as a leading bracket, e.g. EHR-RAG's
template `"[time] type - description (value: value)"`
(`literature/15` §11, Appendix A, R-QCC-045) — rather than inventing one
when it is not present. This follows directly from LLMs' documented
weakness at duration arithmetic and fact-order sensitivity (Test of Time,
`literature/05` R-TKG-10/11; Premise Order Matters, R-TKG-12): make the
dates unmissable where they exist, never fabricate one where they do not.

## Optional post-step: the cite-or-abstain guardrail (`eval/guardrail.py`)

`read()` and `ReaderAnswerer` accept `guardrail_enabled: bool = False` /
`guardrail_policy: str = "warn"`. **`guardrail_enabled=False` (the default)
is a genuine no-op** — when `False`, `guardrail.enforce()` is never even
called, so this module's behavior is byte-for-byte what it was before
`eval/guardrail.py` existed (`tests/test_reader.py`'s entire pre-existing
suite passes this way, unmodified). When `True`, after the completion above
produces `answer_text`/`abstained`, `guardrail.enforce(answer_text, items,
guardrail_policy, question=question, structural_absence=structural_absence,
dry_run=dry_run)` runs (one extra judge-model call at most — see
`guardrail.py`'s own docstring for the zero-call short-circuits, including
reusing `structural_absence` rather than re-deriving it), and its
(possibly-rewritten, under `"strip"`/`"abstain"`) `.text` becomes this
function's returned `Answer.text`; the full `GroundingReport` is always
attached at `Answer.grounding` regardless of policy, so a caller can see
what was found even under `"warn"`, which never rewrites `text` itself.
This is deliberately wired here, not in `eval/harness.py`, per the story
that added it — "an optional post-step... so the A/B measures its effect
rather than assuming it," i.e. run the harness twice (flag off, flag on)
and diff the resulting per-category tables, the same A/B pattern
`reader_direct`/`reader_con` already establishes for reading strategy. See
`docs/algorithms/guardrail.md` for the full algorithm writeup.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Callable

from medmemgraph import llm
from medmemgraph.contracts import RetrieveItem, RetrieveResult
from medmemgraph.eval import guardrail
from medmemgraph.eval.guardrail import GroundingReport
from medmemgraph.eval.types import AnswerResult, estimate_tokens
from medmemgraph.pipeline.loader import Conversation

__all__ = [
    "DEFAULT_READER_MODEL",
    "Note",
    "Answer",
    "read",
    "render_context",
    "ReaderAnswerer",
    "ReaderDirectAnswerer",
    "ReaderChainOfNoteAnswerer",
]

DEFAULT_READER_MODEL = llm.ANSWER_MODEL
"""OpenAI, gpt-4.1-mini by default (override via `MEDMEMGRAPH_ANSWER_MODEL`,
read once by `llm.py` at import time — never hardcode a model string here,
per project cost policy). Cheap enough to run across a whole patient's QA
set twice (once per reader mode for the A/B); deliberately a *different*
provider family from `judge.py`'s `DEFAULT_JUDGE_MODEL` (Google) — see
`judge.py`'s module docstring for the cross-family self-preference-bias
argument (`literature/17`), which applies to this constant too even though
the mitigation is documented on the judge side."""

_VALID_MODES = frozenset({"direct", "chain_of_note"})
_VALID_RENDERINGS = frozenset({"json", "prose", "shuffled"})
_READER_MAX_TOKENS = 700
"""Chain-of-Note needs room for one note per retrieved item plus the
reasoning/answer — bigger than judge.py's 300-token budget (a single
correct/reason verdict)."""
_DEFAULT_SHUFFLE_SEED = 0


# ---------------------------------------------------------------------------
# Timestamp/provenance splitting — see module docstring.
# ---------------------------------------------------------------------------

_LEADING_TIMESTAMP_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2}[^\]]*)\]\s*(.*)$", re.DOTALL)


def _split_timestamp(text: str) -> tuple[str | None, str]:
    """Best-effort split of a leading bracketed timestamp off an item's raw
    text (EHR-RAG's `[time] type - description (value: value)` template,
    `literature/15` R-QCC-045) into `(time, remaining_text)`. Returns
    `(None, text)` when no such prefix is present — an absent timestamp is
    rendered as "unknown", never invented."""
    match = _LEADING_TIMESTAMP_RE.match(text.strip())
    if not match:
        return None, text
    return match.group(1).strip(), match.group(2).strip()


# ---------------------------------------------------------------------------
# Context rendering — swappable strategy, story requirement.
# ---------------------------------------------------------------------------


def render_context(
    items: list[RetrieveItem], rendering: str, *, rng: random.Random | None = None
) -> str:
    """Render retrieved evidence into prompt text under one of three
    strategies (`json` | `prose` | `shuffled`). See module docstring for
    the literature tension this operationalizes rather than adjudicates."""
    if rendering not in _VALID_RENDERINGS:
        raise ValueError(
            f"unknown rendering strategy {rendering!r}; expected one of {sorted(_VALID_RENDERINGS)}"
        )
    if rendering == "json":
        return _render_json(items)
    if rendering == "prose":
        return _render_prose(items)
    return _render_prose(items, shuffle=True, rng=rng)


def _render_json(items: list[RetrieveItem]) -> str:
    """Structured, JSON-per-item rendering — the format LongMemEval's own
    ablation measured its +10-point gain from (R-QCC-043)."""
    payload = []
    for idx, item in enumerate(items, start=1):
        time_, body = _split_timestamp(item.text)
        payload.append(
            {
                "item": idx,
                "session_id": item.session_id,
                "turn_ids": list(item.turn_ids),
                "time": time_,
                "channel": item.channel,
                "score": item.score,
                "text": body,
            }
        )
    return json.dumps(payload, indent=2)


def _render_prose(
    items: list[RetrieveItem], *, shuffle: bool = False, rng: random.Random | None = None
) -> str:
    """One bracketed line per item, in natural-language prose, with
    `session_id`/`turn_ids`/timestamp (if known)/channel/score all rendered
    explicitly ahead of the item's text — deliberately *not* buried
    mid-sentence, per Test of Time's ordering-sensitivity findings
    (`literature/05` R-TKG-11/12, cited via `literature/15` §11).

    `shuffle=True` (the `"shuffled"` strategy) reorders *display* order only
    — the `Item N` ordinal is always the item's position in the original
    `items` list, so provenance stays unambiguous even though the reading
    order no longer follows the retrieval-score/session order. This
    operationalizes Chroma's "shuffling ... removes local coherence [and]
    consistently improves performance" finding (R-QCC-037) without losing
    the "do not invent/confuse a citation" guarantee.
    """
    ordered = list(enumerate(items, start=1))
    if shuffle:
        rng = rng or random.Random(_DEFAULT_SHUFFLE_SEED)
        rng.shuffle(ordered)

    lines = []
    for idx, item in ordered:
        time_, body = _split_timestamp(item.text)
        time_part = f"time {time_}" if time_ else "time unknown"
        lines.append(
            f"[Item {idx} · session {item.session_id} · turns {list(item.turn_ids)} · "
            f"{time_part} · channel {item.channel} · score {item.score:.3f}] {body}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompts + schemas
# ---------------------------------------------------------------------------

_DIRECT_SYSTEM = (
    "You are a clinical question-answering assistant. Read the RETRIEVED "
    "EVIDENCE below -- already selected for this question, from this "
    "patient's record -- and answer directly using ONLY that evidence and "
    "general medical knowledge. Do not invent patient-specific facts not "
    'present in the evidence. If the evidence does not contain enough '
    'information to answer, respond with the literal string "NOT_IN_RECORD" '
    "as the answer and set abstained=true. Respond with the required JSON "
    "only."
)
"""The 'just answer' ablation counterpart to Chain-of-Note — same evidence,
same schema shape, no extract-then-reason step. This is what the A/B
compares against (story requirement: "both modes must be available so we
can A/B them on our own data")."""

_CON_SYSTEM = (
    "You are a clinical question-answering assistant that reads previously "
    "retrieved evidence about a patient in two disciplined steps, per the "
    "Chain-of-Note extract-then-reason method (Yu et al., EMNLP 2023, "
    "arXiv:2311.09210; see literature/15-query-understanding-and-context-"
    "compression.md R-QCC-042/043/044).\n"
    "STEP 1 -- EXTRACT: for EVERY item in the RETRIEVED EVIDENCE below, in "
    "order, write exactly one short note stating what that item actually "
    "says that bears on the QUESTION, or the single word IRRELEVANT if it "
    "does not bear on the question at all. Do not skip any item. Do not "
    "merge two items into one note.\n"
    "STEP 2 -- REASON: using ONLY the notes you wrote in Step 1 -- never "
    "outside/general knowledge, and never an item you marked IRRELEVANT -- "
    "answer the QUESTION. If every note is IRRELEVANT, or STRUCTURAL_ABSENCE "
    'is true below (the retrieval system itself found no connecting '
    'evidence), respond with the literal string "NOT_IN_RECORD" as the '
    "answer and set abstained=true. Never state a session id or turn id "
    "that was not given to you in the evidence below.\n"
    "Respond with the required JSON only: one note object per evidence "
    "item, in the same order, then the answer, then abstained."
)

_DIRECT_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "abstained": {"type": "boolean"},
    },
    "required": ["answer", "abstained"],
    "additionalProperties": False,
}

_CON_SCHEMA = {
    "type": "object",
    "properties": {
        "notes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "session_id": {"type": "string"},
                    "turn_ids": {"type": "array", "items": {"type": "integer"}},
                    "note": {"type": "string"},
                    "relevant": {"type": "boolean"},
                },
                "required": ["session_id", "turn_ids", "note", "relevant"],
                "additionalProperties": False,
            },
        },
        "answer": {"type": "string"},
        "abstained": {"type": "boolean"},
    },
    "required": ["notes", "answer", "abstained"],
    "additionalProperties": False,
}


def _build_user_content(
    question: str, context_text: str, structural_absence: bool, rendering: str
) -> str:
    return (
        f"RETRIEVED EVIDENCE ({rendering} rendering):\n{context_text}\n\n"
        f"STRUCTURAL_ABSENCE: {structural_absence}\n"
        f"QUESTION: {question}"
    )


# ---------------------------------------------------------------------------
# Result shapes
# ---------------------------------------------------------------------------


@dataclass
class Note:
    """One Chain-of-Note extraction record for one retrieved item.

    `session_id`/`turn_ids` are always copied verbatim from the retrieved
    item itself (`_reconcile_notes`), never taken from the model's own
    output. That is what makes "never state a session id or turn id that
    was not given to you" a structural guarantee rather than a prompt
    request the model could ignore."""

    session_id: str
    turn_ids: list[int]
    text: str
    relevant: bool


@dataclass
class Answer:
    """What `read()` returns. `abstained` is a first-class structured field
    — populated from the model's own JSON output (`_DIRECT_SCHEMA` /
    `_CON_SCHEMA`), never inferred by regexing the answer prose — so a
    downstream judge or harness can trust it directly (story requirement:
    "Refusal must be a first-class output, not a parse failure")."""

    text: str
    abstained: bool
    notes: list[Note]
    mode: str
    rendering: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    grounding: GroundingReport | None = None
    """`None` when `guardrail_enabled=False` (the default — see module
    docstring's "Optional post-step" section) or when there were no items
    to check (the empty-pack early return below, which never reaches the
    guardrail either). Populated whenever `guardrail_enabled=True`, even
    under `"warn"` (which never changes `text`) or a zero-call shortcut
    (`GroundingReport.shortcut_reason` names which one) — always a real
    `GroundingReport`, never a stand-in, once the guardrail ran at all."""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def citations(self) -> list[dict]:
        """`session_id`/`turn_ids` of every note the reasoning step marked
        relevant. Always empty for `mode="direct"` — direct mode has no
        extraction step and therefore no provenance chain to cite, which is
        itself part of the A/B story: only Chain-of-Note can answer "why do
        you believe that" with a specific turn."""
        return [{"session_id": n.session_id, "turn_ids": list(n.turn_ids)} for n in self.notes if n.relevant]


# ---------------------------------------------------------------------------
# Note reconciliation — the "do not invent turns" guarantee
# ---------------------------------------------------------------------------


def _reconcile_notes(raw_notes: list[dict] | None, items: list[RetrieveItem]) -> list[Note]:
    """Build exactly one `Note` per retrieved item, in item order, with
    `session_id`/`turn_ids` grounded directly in the real item — never in
    whatever the model's JSON output claims. If the model's note for an
    item names a `(session_id, turn_ids)` pair that does not exactly match
    a real item (skipped it, or invented/garbled the citation), that raw
    note is silently unmatched and the item defaults to an IRRELEVANT,
    non-citable note instead of trusting an ungrounded claim."""
    raw_notes = raw_notes or []
    by_key: dict[tuple[str, tuple[int, ...]], dict] = {}
    for raw in raw_notes:
        try:
            key = (str(raw.get("session_id", "")), tuple(int(t) for t in (raw.get("turn_ids") or [])))
        except (TypeError, ValueError):
            continue
        by_key[key] = raw

    notes: list[Note] = []
    for item in items:
        key = (item.session_id, tuple(item.turn_ids))
        raw = by_key.get(key)
        if raw is None:
            notes.append(
                Note(session_id=item.session_id, turn_ids=list(item.turn_ids), text="IRRELEVANT", relevant=False)
            )
            continue
        notes.append(
            Note(
                session_id=item.session_id,
                turn_ids=list(item.turn_ids),
                text=str(raw.get("note", "")).strip() or "IRRELEVANT",
                relevant=bool(raw.get("relevant", False)),
            )
        )
    return notes


# ---------------------------------------------------------------------------
# Completion backends — real llm.py seam call, and a deterministic offline stub
# ---------------------------------------------------------------------------


def _real_complete(model: str, system: str, user: str, schema: dict) -> tuple[dict, int, int]:
    """Route through `medmemgraph.llm.complete()`. Schema-constrained (never
    a bare "respond in JSON" instruction) via `schema=...` — `llm.complete`
    normalizes it per provider and validates/retries against it, so
    `response.parsed` is always a dict matching `schema` by the time this
    function returns (or `llm.complete` itself raised). No `client` param:
    unlike the old direct-Anthropic-SDK code this replaces, `llm.py` owns
    provider-client construction/caching/keys itself — there is nothing
    left for this module to inject except `model`. Token counts are the
    provider's own reported usage (`response.prompt_tokens` /
    `.completion_tokens`), not `estimate_tokens()` — so cost reporting here
    matches what was actually billed, same guarantee the old code gave via
    `response.usage`."""
    response = llm.complete(
        user,
        model=model,
        schema=schema,
        system=system,
        max_tokens=_READER_MAX_TOKENS,
        temperature=0.0,
    )
    return response.parsed, response.prompt_tokens, response.completion_tokens


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _content_words(text: str) -> set[str]:
    """Coarse (no stopword list) tokenizer used only to decide the
    dry-run/offline stub's relevant/IRRELEVANT split — deliberately not the
    same machinery as `judge.py`'s scoring tokenizer, since this is a
    behavioral stand-in for a model's judgment, not a correctness score."""
    return {tok for tok in _TOKEN_RE.findall(text.lower()) if len(tok) > 2}


def _stub_complete(
    mode: str,
    items: list[RetrieveItem],
    question: str,
    structural_absence: bool,
    system: str,
    user: str,
) -> tuple[dict, int, int]:
    """Deterministic, API-free stand-in for `dry_run=True` / CI / no
    `ANTHROPIC_API_KEY` (story requirement 5: "Run without an API key via
    the deterministic stub so tests work offline").

    Deliberately encodes the *mechanistic* difference the literature
    attributes to Chain-of-Note, rather than faking a fixed answer: direct
    mode concatenates every item's text unfiltered (a "just answer, no
    filtering" reading strategy); chain_of_note mode filters to only the
    items that share a content word with the question before answering,
    and abstains outright if nothing survives that filter — the one thing
    direct mode structurally cannot do. This makes the offline stub useful
    for exercising both code paths honestly, not just returning canned
    text (see the fullctx/nomem baselines' `_stub_answer` for the same
    convention in this codebase)."""
    prompt_tokens = estimate_tokens(system) + estimate_tokens(user)

    if structural_absence:
        payload: dict = {"answer": "NOT_IN_RECORD", "abstained": True}
        if mode == "chain_of_note":
            payload["notes"] = [
                {"session_id": it.session_id, "turn_ids": list(it.turn_ids), "note": "IRRELEVANT", "relevant": False}
                for it in items
            ]
        completion_tokens = estimate_tokens(json.dumps(payload))
        return payload, prompt_tokens, completion_tokens

    if mode == "chain_of_note":
        q_words = _content_words(question)
        notes_raw = []
        relevant_snippets = []
        for it in items:
            _, body = _split_timestamp(it.text)
            is_relevant = bool(q_words & _content_words(it.text))
            if is_relevant:
                notes_raw.append(
                    {
                        "session_id": it.session_id,
                        "turn_ids": list(it.turn_ids),
                        "note": f"[dry-run stub note] {body[:160]}",
                        "relevant": True,
                    }
                )
                relevant_snippets.append(body)
            else:
                notes_raw.append(
                    {
                        "session_id": it.session_id,
                        "turn_ids": list(it.turn_ids),
                        "note": "IRRELEVANT",
                        "relevant": False,
                    }
                )
        if relevant_snippets:
            answer = "[dry-run stub, chain_of_note] " + " ".join(relevant_snippets)[:300]
            abstained = False
        else:
            answer = "NOT_IN_RECORD"
            abstained = True
        payload = {"notes": notes_raw, "answer": answer, "abstained": abstained}
    else:
        bodies = [_split_timestamp(it.text)[1] for it in items]
        combined = " ".join(bodies)[:300]
        payload = {"answer": f"[dry-run stub, direct] {combined}", "abstained": False}

    completion_tokens = estimate_tokens(json.dumps(payload))
    return payload, prompt_tokens, completion_tokens


# ---------------------------------------------------------------------------
# read() — the public entry point
# ---------------------------------------------------------------------------


def read(
    question: str,
    items: list[RetrieveItem],
    mode: str = "chain_of_note",
    *,
    rendering: str = "json",
    structural_absence: bool = False,
    model: str = DEFAULT_READER_MODEL,
    dry_run: bool = False,
    rng: random.Random | None = None,
    guardrail_enabled: bool = False,
    guardrail_policy: str = "warn",
) -> Answer:
    """Answer `question` over already-retrieved `items`. Never retrieves —
    this function never calls the graph/vector retriever or the corpus
    loader (verified by `tests/test_reader.py`'s source-grep test); the
    pack is an argument, per the story's "no second retrieval call inside
    the reader" hard constraint.

    `dry_run=False` (the default) always attempts a real
    `medmemgraph.llm.complete()` call — if no API key is configured, that
    raises `llm.MissingAPIKeyError` immediately rather than silently
    degrading to the offline stub (see module docstring). `dry_run=True`
    runs the deterministic, clearly-labelled offline stub instead
    (`_stub_complete`) — no network call, no key needed, no cost.

    `guardrail_enabled=False` (the default) is a genuine no-op — see module
    docstring's "Optional post-step" section for the full contract."""
    if mode not in _VALID_MODES:
        raise ValueError(f"unknown reader mode {mode!r}; expected one of {sorted(_VALID_MODES)}")
    if rendering not in _VALID_RENDERINGS:
        raise ValueError(
            f"unknown rendering strategy {rendering!r}; expected one of {sorted(_VALID_RENDERINGS)}"
        )

    start = time.monotonic()

    if not items:
        # No evidence at all -- nothing to extract or reason over, and no
        # signal worth spending a model call on. A boolean "no items"
        # precondition is a stronger, zero-cost abstention signal than
        # anything the generator itself could contribute here (consistent
        # with `literature/06-abstention-and-calibration.md`'s framing:
        # a pre-generation structural check beats relying on the
        # generator's own confidence, R-ABST-13/28).
        latency_ms = max((time.monotonic() - start) * 1000.0, 0.001)
        return Answer(
            text="NOT_IN_RECORD",
            abstained=True,
            notes=[],
            mode=mode,
            rendering=rendering,
            prompt_tokens=0,
            completion_tokens=0,
            latency_ms=latency_ms,
        )

    context_text = render_context(items, rendering, rng=rng)
    system_prompt = _CON_SYSTEM if mode == "chain_of_note" else _DIRECT_SYSTEM
    user_content = _build_user_content(question, context_text, structural_absence, rendering)
    schema = _CON_SCHEMA if mode == "chain_of_note" else _DIRECT_SCHEMA

    if dry_run:
        payload, prompt_tokens, completion_tokens = _stub_complete(
            mode, items, question, structural_absence, system_prompt, user_content
        )
    else:
        payload, prompt_tokens, completion_tokens = _real_complete(
            model, system_prompt, user_content, schema
        )

    latency_ms = (time.monotonic() - start) * 1000.0

    notes = _reconcile_notes(payload.get("notes"), items) if mode == "chain_of_note" else []
    answer_text = str(payload.get("answer", ""))
    abstained = bool(payload.get("abstained", False)) or answer_text.strip().upper() == "NOT_IN_RECORD"

    grounding_report: GroundingReport | None = None
    if guardrail_enabled:
        enforced = guardrail.enforce(
            answer_text, items, guardrail_policy,
            question=question, structural_absence=structural_absence, dry_run=dry_run,
        )
        answer_text = enforced.text
        grounding_report = enforced.report
        abstained = abstained or answer_text.strip().upper() == "NOT_IN_RECORD"

    return Answer(
        text=answer_text,
        abstained=abstained,
        notes=notes,
        mode=mode,
        rendering=rendering,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_ms=latency_ms,
        grounding=grounding_report,
    )


# ---------------------------------------------------------------------------
# Harness wiring — two selectable systems, reusing evaluate()'s machinery
# ---------------------------------------------------------------------------


def _default_retriever() -> Callable[[str, str, int], RetrieveResult]:
    """Prefer the real graph-backed retriever once Graph lands it (opt in
    via `$MEDMEMGRAPH_USE_REAL_RETRIEVE=1`); fall back to
    `contracts.mock_retrieve` otherwise. Per the project convention ("Do
    not wait for it -- code against contracts.mock_retrieve and import the
    real one behind a flag"), Evidence never blocks on Graph.

    **Always `retrieve_for_eval`, never bare `retrieve`.** This module's
    `answer()` feeds a results table (`eval/harness.py`), and
    `decisions/004`/`inbox/006-to-claude.md` both promise, in writing, that
    eval tables run at `epsilon=0` so the headline route is the frozen
    deterministic rule. `graph.retrieve.retrieve()` defaults to
    `DEFAULT_EPSILON=0.05` (E5-S1's own contract, correct for *live* use,
    where the exploration log is the point) -- calling it unmodified here
    would have silently carried ~5% epsilon-flipped routes into every
    reported number (reconciliation audit, `decisions/005`, Finding 1).
    `retrieve_for_eval` pins `epsilon=0` and raises if a caller tries to
    override it, so this call site cannot regress that quietly."""
    if os.environ.get("MEDMEMGRAPH_USE_REAL_RETRIEVE") == "1":
        try:
            from medmemgraph.graph.retrieve import retrieve_for_eval

            return retrieve_for_eval
        except ImportError:
            pass
    from medmemgraph.contracts import mock_retrieve

    return mock_retrieve


class ReaderAnswerer:
    """Answers by retrieving evidence (`_default_retriever()`) and reading
    it via `read()`. Two concrete subclasses below register two harness
    systems, `reader_direct` and `reader_con`, so the harness's existing
    per-category table / judge-routing machinery becomes the A/B tool for
    free — same `Answerer` protocol as `nomem`/`fullctx`, nothing else in
    `harness.py`'s scoring path has to change."""

    mode: str = "chain_of_note"
    name: str = "reader"

    def __init__(
        self,
        *,
        rendering: str = "json",
        k: int = 6,
        dry_run: bool = False,
        model: str = DEFAULT_READER_MODEL,
        client: object | None = None,
        retriever: Callable[[str, str, int], RetrieveResult] | None = None,
        guardrail_enabled: bool = False,
        guardrail_policy: str = "warn",
    ) -> None:
        self.rendering = rendering
        self.k = k
        self.dry_run = dry_run
        self.model = model
        self.guardrail_enabled = guardrail_enabled
        self.guardrail_policy = guardrail_policy
        # `client` is accepted (not removed) only for call-site compatibility
        # with `eval/baselines/dense.py` / `eval/baselines/lexical.py`, which
        # both construct a `ReaderAnswerer` (via subclassing) with a
        # `client=...` kwarg and are out of this story's file scope ("Both
        # files are yours; nothing else touches them"). It is otherwise
        # inert: `read()` no longer takes a `client` argument at all — real
        # calls always route through `medmemgraph.llm.complete()`, which
        # owns provider-client construction/caching/keys itself, so there is
        # nothing left for a caller-supplied client to inject. Both of the
        # only two call sites that pass this kwarg always pass `None`.
        self.client = client
        self.retriever = retriever or _default_retriever()

    def answer(
        self, question: str, conversation: Conversation | None, *, patient_id: str
    ) -> AnswerResult:
        del conversation  # this system answers over *retrieved* evidence, not raw turn history
        pack = self.retriever(question, patient_id, self.k)
        result = read(
            question,
            pack.items,
            self.mode,
            rendering=self.rendering,
            structural_absence=pack.structural_absence,
            model=self.model,
            dry_run=self.dry_run,
            guardrail_enabled=self.guardrail_enabled,
            guardrail_policy=self.guardrail_policy,
        )
        return AnswerResult(
            text=result.text,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            latency_ms=result.latency_ms,
        )


class ReaderDirectAnswerer(ReaderAnswerer):
    mode = "direct"
    name = "reader_direct"


class ReaderChainOfNoteAnswerer(ReaderAnswerer):
    mode = "chain_of_note"
    name = "reader_con"
