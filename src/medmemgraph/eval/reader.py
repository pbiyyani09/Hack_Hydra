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

import inspect
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
    "MedMemGraphAnswerer",
    "MedMemGraphP1Answerer",
    "MedMemGraphP2Answerer",
    "MedMemGraphP1P2Answerer",
    "MedMemGraphR1Answerer",
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
_READER_MAX_TOKENS = 4000
"""Chain-of-Note needs room for one note per retrieved item plus the
reasoning/answer — bigger than judge.py's 300-token budget (a single
correct/reason verdict).

Raised 700 -> 2500, then -> 4000 on 2026-08-19 when k rose to 60. One realistic note object measures ~44 tokens,
so 700 fit only ~14 notes before the answer text was squeezed out — which
silently capped the usable evidence budget at k~14 no matter what `k` was set
to. Past that, `llm.py` detects truncation and escalates max_tokens x4
(`_next_truncation_max_tokens`), so the run still completed but paid an extra
round trip per item. 4000 covers ~60 notes plus an answer without escalating; the cap must track
`k`, because Chain-of-Note emits one note per evidence item and an
under-sized cap silently converts extra evidence into extra retries."""
_DEFAULT_SHUFFLE_SEED = 0


@dataclass(frozen=True)
class EvidenceProvenance:
    """What the retrieved evidence was drawn FROM — the denominator the reader
    otherwise never sees. All fields optional; a caller that cannot cheaply
    compute one passes None rather than guessing."""

    total_units: int | None = None
    n_admissions: int | None = None
    first_time: str | None = None
    last_time: str | None = None


def _provenance_of(conversation: "Conversation | None") -> EvidenceProvenance | None:
    """Summarize a `Conversation` into the denominator for `describe_evidence`.

    Counts and date span ONLY — never turn text. This function is the one place
    `ReaderAnswerer` touches the raw history, and it must stay incapable of
    leaking content into the prompt, or the system stops being a retrieval
    system and becomes full-context wearing its name."""
    if conversation is None:
        return None
    try:
        admissions = conversation.admissions
        times = sorted(
            t.time for adm in admissions for t in adm.turns() if getattr(t, "time", None)
        )
        return EvidenceProvenance(
            total_units=sum(len(adm.conversation_lines) for adm in admissions),
            n_admissions=len(admissions),
            first_time=times[0] if times else None,
            last_time=times[-1] if times else None,
        )
    except Exception:  # noqa: BLE001 — provenance is a nicety; never break an answer for it
        return None


# ---------------------------------------------------------------------------
# Timestamp/provenance splitting — see module docstring.
# ---------------------------------------------------------------------------

_LEADING_TIMESTAMP_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2}[^\]]*)\]\s*(.*)$", re.DOTALL)
"""EHR-RAG's `[time] description` template (`literature/15` R-QCC-045) — a
timestamp as the FIRST thing inside the bracket. Kept because it is the shape
the literature describes, but note that NO producer in this project emits it;
see `_TIMESTAMP_PATTERNS` for the ones that actually occur."""

_TIMESTAMP_PATTERNS = (
    # vector_index turn units:  "[admission 20971116 turn 14 · 2120-08-06 20:15:00 · Doctor] ..."
    re.compile(r"^\[[^\]]*?·\s*(\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?)\s*·[^\]]*\]\s*(.*)$", re.DOTALL),
    # vector_index fact units:  "[fact <id> · admission <sid> · as of 2120-08-06T00:00:00] ..."
    re.compile(r"^\[[^\]]*?as of\s*(\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?)?)[^\]]*\]\s*(.*)$", re.DOTALL),
    # graph path units:  "... Claim[REPORTS_SYMPTOM asserted, active, 2160-08-14T00:00:00..ongoing]"
    re.compile(r"^(?P<head>.*?Claim\[[^\]]*?(?P<ts>\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}(?::\d{2})?)?)\.\.[^\]]*\].*)$", re.DOTALL),
)
"""The timestamp shapes this project's three item producers ACTUALLY emit.

Added 2026-08-19 after finding `_LEADING_TIMESTAMP_RE` matched none of them.
That regex requires the date to be the first characters inside the bracket;
every real item puts an admission id, a fact id, or a whole path expression
first. The consequence was that `time` was rendered `null` on EVERY evidence
item, and prose rendering printed "time unknown" — while the dates sat in plain
sight inside `text`, unlabelled.

This was not cosmetic. The three question categories this system loses to the
full-context baseline (`cross_admission_comparison`, `longitudinal_progression`,
`frequency_pattern`) are precisely the ones that need to order and compare
events in time, and the reader was being handed a field explicitly telling it
no timestamp was available. `fullctx`, by contrast, states "oldest first" in its
own prompt and gets the whole history in order.

Order matters: the turn pattern is tried first because it is the most common
unit, and the `as of` pattern before the graph one because a fact unit's bracket
can also contain a `..` interval."""

_TIMESTAMP_PATTERNS_R1 = _TIMESTAMP_PATTERNS + (
    # graph TIMELINE bodies:  "TIMELINE for <entity> [2160-08-14 .. present] <enumerated claims>"
    re.compile(r"^(?P<head>TIMELINE for .*?\[(?P<ts>\d{4}-\d{2}-\d{2}[^\]]*)\].*)$", re.DOTALL),
    # graph CONTEXT-for-admission bodies:  "CONTEXT for admission <sid> (~2160-08-14) <summary>"
    re.compile(r"^(?P<head>CONTEXT for admission [^(]*\(~\s*(?P<ts>\d{4}-\d{2}-\d{2}[^)]*)\).*)$", re.DOTALL),
)
"""`_TIMESTAMP_PATTERNS` plus the two graph TIMELINE/CONTEXT shapes, used
only when `aggregation_prompt=True` (arm R1) — gated, not merged into
`_TIMESTAMP_PATTERNS` itself, so the control's rendering stays byte-identical.

Added 2026-08-20. TIMELINE and CONTEXT items are the most aggregation-relevant
evidence this system retrieves (a TIMELINE item is a complete per-entity
enumeration; see `_CON_STEP3_AGGREGATION_R1`), and until this patch they fell
through every existing pattern to `(None, text)` — rendered `time: null` /
"time unknown" to a reader whose prompt explicitly says to order by time. The
date sits mid-expression in both shapes (inside a bracket/paren that also
carries other content), same as the graph path pattern above, so the `ts`
named group is used the same way: `_split_timestamp` returns the text
UNCHANGED alongside the extracted time rather than trying to strip a prefix."""


def _split_timestamp(text: str, *, extra_patterns: bool = False) -> tuple[str | None, str]:
    """Best-effort split of a timestamp off an item's raw text into
    `(time, remaining_text)`.

    Tries the EHR-RAG leading-bracket template first (`literature/15`
    R-QCC-045), then the shapes this project's own producers emit
    (`_TIMESTAMP_PATTERNS`, or `_TIMESTAMP_PATTERNS_R1` when
    `extra_patterns=True` — arm R1 only, see `_TIMESTAMP_PATTERNS_R1`'s
    docstring). Returns `(None, text)` when nothing matches — an absent
    timestamp is rendered as "unknown", never invented.

    For graph path/TIMELINE/CONTEXT items the text is returned UNCHANGED
    alongside the extracted time: the timestamp is embedded mid-expression
    (`Claim[... 2160-08-14..ongoing]`) rather than being a strippable prefix,
    and cutting it out would mangle the rendering the reader needs to read."""
    stripped = text.strip()
    match = _LEADING_TIMESTAMP_RE.match(stripped)
    if match:
        return match.group(1).strip(), match.group(2).strip()

    patterns = _TIMESTAMP_PATTERNS_R1 if extra_patterns else _TIMESTAMP_PATTERNS
    for pattern in patterns:
        match = pattern.match(stripped)
        if not match:
            continue
        if "ts" in (match.groupdict() or {}):
            # Graph path: keep the whole expression, just surface the time.
            return match.group("ts").strip(), stripped
        return match.group(1).strip(), match.group(2).strip()
    return None, text


# ---------------------------------------------------------------------------
# Context rendering — swappable strategy, story requirement.
# ---------------------------------------------------------------------------


def render_context(
    items: list[RetrieveItem],
    rendering: str,
    *,
    rng: random.Random | None = None,
    aggregation_prompt: bool = False,
) -> str:
    """Render retrieved evidence into prompt text under one of three
    strategies (`json` | `prose` | `shuffled`). See module docstring for
    the literature tension this operationalizes rather than adjudicates.

    `aggregation_prompt=False` (the default) renders exactly as before —
    control stays byte-identical. `aggregation_prompt=True` (arm R1) is
    the sole gate on the two rendering fixes below: it recognizes the graph
    TIMELINE/CONTEXT timestamp shapes (`_TIMESTAMP_PATTERNS_R1`) and drops
    the RRF-overwritten `score` field, both no-ops when False."""
    if rendering not in _VALID_RENDERINGS:
        raise ValueError(
            f"unknown rendering strategy {rendering!r}; expected one of {sorted(_VALID_RENDERINGS)}"
        )
    if rendering == "json":
        return _render_json(items, aggregation_prompt=aggregation_prompt)
    if rendering == "prose":
        return _render_prose(items, aggregation_prompt=aggregation_prompt)
    return _render_prose(items, shuffle=True, rng=rng, aggregation_prompt=aggregation_prompt)


def _render_json(items: list[RetrieveItem], *, aggregation_prompt: bool = False) -> str:
    """Structured, JSON-per-item rendering — the format LongMemEval's own
    ablation measured its +10-point gain from (R-QCC-043).

    `aggregation_prompt=True` (arm R1) drops the `score` key. RRF
    (`graph/fusion.py`) overwrites every channel's own score before this
    ever runs, so the rendered floats are near-tied and collapse to a
    handful of distinct values at `.3f` — information-free, and ~405 prompt
    tokens at k=40 spent stating it. Gated: the control branch below is the
    unmodified original code, so `aggregation_prompt=False` output is
    byte-identical to before this change."""
    payload = []
    for idx, item in enumerate(items, start=1):
        time_, body = _split_timestamp(item.text, extra_patterns=aggregation_prompt)
        if aggregation_prompt:
            payload.append(
                {
                    "item": idx,
                    "session_id": item.session_id,
                    "turn_ids": list(item.turn_ids),
                    "time": time_,
                    "channel": item.channel,
                    "text": body,
                }
            )
        else:
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
    items: list[RetrieveItem],
    *,
    shuffle: bool = False,
    rng: random.Random | None = None,
    aggregation_prompt: bool = False,
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

    `aggregation_prompt=True` (arm R1) drops the ` · score {:.3f}` fragment
    — see `_render_json`'s docstring for why. Gated the same way: the
    control branch is the unmodified original code.
    """
    ordered = list(enumerate(items, start=1))
    if shuffle:
        rng = rng or random.Random(_DEFAULT_SHUFFLE_SEED)
        rng.shuffle(ordered)

    lines = []
    for idx, item in ordered:
        time_, body = _split_timestamp(item.text, extra_patterns=aggregation_prompt)
        time_part = f"time {time_}" if time_ else "time unknown"
        if aggregation_prompt:
            lines.append(
                f"[Item {idx} · session {item.session_id} · turns {list(item.turn_ids)} · "
                f"{time_part} · channel {item.channel}] {body}"
            )
        else:
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

_CON_STEP2_REASON = (
    "STEP 2 -- REASON: using ONLY the notes you wrote in Step 1 -- never "
    "outside/general knowledge, and never an item you marked IRRELEVANT -- "
    "answer the QUESTION. If every note is IRRELEVANT, or STRUCTURAL_ABSENCE "
    'is true below (the retrieval system itself found no connecting '
    'evidence), respond with the literal string "NOT_IN_RECORD" as the '
    "answer and set abstained=true. Never state a session id or turn id "
    "that was not given to you in the evidence below.\n"
)
"""Control's STEP 2. Extracted into its own constant (rather than left
inline in `_CON_SYSTEM`) so it can serve as the anchor `read()` swaps out
for `_CON_STEP2_PREMISE_CHECK` when `premise_check_prompt=True` (arm P1,
below) — same splice-by-`.replace()` mechanism `commit_style` already uses
on `_RESPOND_LINE`. `_CON_SYSTEM` still concatenates this constant in
directly, so the control arm's prompt string is byte-identical to before
this constant was pulled out."""

_CON_STEP2_PREMISE_CHECK = (
    "STEP 2 -- CHECK THE PREMISE: a question can be about a topic your notes "
    "cover and still ask for a fact the record never states. Name the ONE "
    "specific thing the QUESTION asks for -- the confirming test or finding, "
    "the causal or temporal link, the trend across admissions -- then find "
    "the note(s) that ASSERT it. A note that mentions the same condition, "
    "drug, organ or admission is on-topic, not an answer. Having many "
    "relevant notes is not evidence that the question is answerable.\n"
    "STEP 3 -- REASON: using ONLY the notes you wrote in Step 1 -- never "
    "outside/general knowledge, and never an item you marked IRRELEVANT -- "
    "answer the QUESTION. Set abstained=true and answer with the literal "
    'string "NOT_IN_RECORD" if ANY of these holds: every note is IRRELEVANT; '
    "STRUCTURAL_ABSENCE is true below (the retrieval system itself found no "
    "connecting evidence); or no note asserts the specific thing you named "
    "in Step 2, so that answering would require you to supply the "
    "confirmation, the causal link or the trend yourself. Assembling a "
    "sequence or a comparison from several notes that each state a dated "
    "fact is NOT supplying it yourself -- that is allowed and expected. "
    "When you abstain, `answer` must be exactly NOT_IN_RECORD and nothing "
    "else -- do not append what the record does show. "
    "Never state a session id or turn id that was not given to you in the "
    "evidence below.\n"
)
"""Arm P1's STEP 2/3 (Change C). Diagnosed root cause: the control's ONLY
abstention trigger is a universal quantifier over topical relevance ("if
every note is IRRELEVANT"), which decays as (1-p)^k -- at k=40 a single
on-topic item out of forty disables abstention entirely. The adversarial
items this project cares about are false-premise questions where evidence
IS topical, so relevance is the wrong predicate. This version adds an
explicit premise-check step and makes "no note asserts the specific thing
asked for" its own abstention trigger, independent of how many notes are
merely on-topic.

The "Assembling a sequence or a comparison ... is NOT supplying it
yourself" sentence is LOAD-BEARING: without it this instruction is
indistinguishable from "abstain unless one note states the whole answer",
which would wrongly abstain on longitudinal/frequency/cross-admission
items whose correct answer is genuinely assembled from several dated
notes. Do not drop or reword it."""

_CON_STEP3_AGGREGATION = (
    "Two consequences you must respect. (a) TIME: each item carries a time "
    "field where one is known -- order events by that field, never by the order "
    "the items happen to appear in. (b) COMPLETENESS: if the question asks "
    "which thing was most frequent, most consistent, or most common, or how "
    "something was distributed across admissions, that is a question about the "
    "WHOLE record. Answer it only if your evidence plainly covers the whole "
    "span; otherwise say what the evidence does show and state that it is a "
    "selection. Do not generalise a count from a sample.\n"
)
"""Control's "consequences" paragraph. Extracted into its own constant
(rather than left inline in `_CON_SYSTEM`) so it can serve as the anchor
`read()` swaps out for `_CON_STEP3_AGGREGATION_R1` when
`aggregation_prompt=True` (arm R1, below) — same splice-by-`.replace()`
mechanism `premise_check_prompt`/`_CON_STEP2_PREMISE_CHECK` already uses.
`_CON_SYSTEM` still concatenates this constant in directly, so the control
arm's prompt string is byte-identical to before this constant was pulled
out."""

_CON_STEP3_AGGREGATION_R1 = (
    "Three consequences you must respect. (a) TIME: each item carries a time "
    "field where one is known -- order events by that field, never by the order "
    "the items happen to appear in.\n"
    "(b) AGGREGATION: some questions have no answer inside any single item -- "
    "'which X recurred most consistently', 'what persisted across admissions', "
    "'how did X evolve', 'compare X between two admissions'. Answer these by "
    "COUNTING AND COMPARING ACROSS your notes, not by quoting the best single "
    "item: group the relevant notes by admission id, count how many DISTINCT "
    "admissions each candidate appears in, and NAME the candidate with the "
    "widest coverage as a specific clinical entity -- a named diagnosis, "
    "medication, lab value, or symptom. An item whose text begins TIMELINE is "
    "already a complete enumeration of every claim on record for that entity, "
    "so a count taken over a TIMELINE item is a count over the whole record, "
    "not over a sample. 'Several problems recurred' is a wrong answer even "
    "when true; 'atrial fibrillation, in 4 of the 6 admissions represented "
    "here' is a right one. Say what your count is over, but still name the "
    "winner.\n"
    "(c) YES/NO QUESTIONS: if the question begins Was / Were / Did / Do / Does "
    "/ Is / Are / Has / Have / Had / Could / Would / Should -- asking whether "
    "something happened, was associated with something, or led to something -- "
    "the record must state that comparison or outcome DIRECTLY. If it does "
    'not, answer "NOT_IN_RECORD" and set abstained=true. Never infer a yes or '
    "a no from adjacent facts.\n"
)
"""Arm R1's consequences paragraph (aggregation_prompt). Diagnosed root
cause: the answerable-accuracy shortfall is concentrated entirely in the
three "roll-up" categories that require aggregating across multiple
retrieved items (longitudinal_progression 0.600, frequency_pattern 0.646,
cross_admission_comparison 0.646 — 98/156 — versus medical_reasoning /
care_plan_rationale at 118/120), and the control's own COMPLETENESS clause
tells the reader to hedge on exactly those questions rather than instructing
it how to aggregate. This version keeps (a) TIME verbatim, replaces (b)
COMPLETENESS's hedge-by-default framing with an explicit count-across-notes
procedure that still requires the model to say what its count is over, and
adds (c) a YES/NO discipline so implied yes/no questions are not answered
from adjacent facts.

Independent of `premise_check_prompt`/`premise_check_schema` (P1/P2) by
design — a later combined arm may want both; this flag never reads or sets
those, and they never read or set this one."""

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
    + _CON_STEP2_REASON +
    "The EVIDENCE PROVENANCE line above the evidence states how many items you "
    "were given, what they were selected from, and in what order. Read it. Your "
    "evidence is a SELECTION, not the patient's complete record, and it is "
    "ordered by retrieval relevance rather than by time.\n"
    + _CON_STEP3_AGGREGATION +
    "Respond with the required JSON only: one note object per evidence "
    "item, in the same order, then the answer, then abstained."
)
"""Chain-of-Note reading prompt.

STEP 3 added 2026-08-19. It exists because of a measured, specific failure, not
a style preference.

`eval/judge.py`'s rubric explicitly disclaims caring about verbosity -- "allowing
for different wording, phrasing, or level of detail", "not exact wording or
completeness of detail" -- and of 128 incorrect answerable records, 127 cite
FACTUAL MATCH and none cite NO CONTRADICTION. So long answers are not penalised
and shortening them buys nothing on its own.

What was failing is COMMITMENT. 44% of the temporal items full-context gets and
we do not are the judge saying we described where we should have asserted:
"described a detailed clinical timeline of reflux and ulceration rather than
[the gold]", "the system's detailed narrative of treatment changes deviates".
MedLoCoMo golds are overwhelmingly transition-shaped -- "topiramate to
valproate", "from unspecified to candidal esophagitis", "osteomyelitis in right
then left toe" -- and a survey of the clinical picture never asserts the one
before/after pair being asked for.

Brevity appears here only as a forcing function toward that commitment. The
instruction is to CHOOSE, not to be short."""

_RESPOND_LINE = (
    "Respond with the required JSON only: one note object per evidence "
    "item, in the same order, then the answer, then abstained."
)

COMMIT_INSTRUCTION = (
    "STEP 4 -- COMMIT: state the single most specific claim the evidence "
    "supports, and state it as a claim rather than as a description. If the "
    "question asks how something changed, progressed, or evolved, or asks you "
    "to compare two points in time, your answer must NAME THE BEFORE-STATE AND "
    "THE AFTER-STATE explicitly -- 'X to Y', 'X in <period>, Y in <period>'. "
    "Only then, if it helps, add one short sentence of support.\n"
    "Do NOT answer a 'how did this change' question by surveying everything "
    "that happened to the patient. A timeline of five findings that never says "
    "which one changed into which is a wrong answer even when every finding in "
    "it is true. Choose. If the evidence genuinely supports two competing "
    "readings, give the better-supported one first, in the required form.\n"
)
"""The "commit to a before/after claim" instruction, kept separate so it can
be switched off.

Renumbered STEP 3 -> STEP 4 on 2026-08-20, when arm P1 (`premise_check_prompt`,
`_CON_STEP2_PREMISE_CHECK`) inserted its own STEP 3 -- REASON into `_CON_SYSTEM`.
Unconditional: this text is a module-level constant, not a per-arm splice, so
if it kept saying "STEP 3" it would collide with P1's STEP 3 whenever both are
enabled together (the P1+P2 arm needs exactly that combination). Renumbering
once here, regardless of which arms happen to be on, is simpler and safer than
threading a second conditional numbering scheme through this text — the label
is inert prose either way; `commit_style` still gates whether this instruction
is spliced in at all.

On by default for evaluation, OFF for `demo/agent.py`. The benchmark rewards
asserting the one transition its gold names; a clinician at the terminal is
better served by the fuller picture with citations. Same retrieval, same
facts, different framing — and the README says so rather than letting the
demo quietly imply the eval answers look like that too.

Set `commit_style=False` to omit it."""


TRANSITION_TYPES: frozenset[str] = frozenset()
"""Question types answered with a forced before/after schema.

**Deliberately EMPTY — this was tried, measured, and reverted on 2026-08-19.**

The idea: MedLoCoMo golds for progression/comparison are transition-shaped
("topiramate to valproate"), our answers narrated instead, and 44% of temporal
losses were the judge saying we described where we should have asserted. A
prompt instruction did nothing (verified: identical answers with the
instruction present and absent, in both reader modes). Two REQUIRED schema
fields did change the shape, exactly as intended.

It made things worse, and the reason is specific and worth keeping:

    gold "ventricular tachycardia"
    ours "episodic ventricular tachycardia to sustained ventricular fibrillation"
    judge: "stated ventricular fibrillation instead of ventricular tachycardia"

    gold "C. difficile colitis"
    ours "No documented C. difficile ... to C. difficile infection and Klebsiella"
    judge: "explicitly denies documented C. difficile infection"

We SAID the gold term both times. Demanding a before-state for a question whose
gold is a single state manufactures a second term — often a negation — and the
judge reads it as contradicting the gold. Before this change, NO CONTRADICTION
had never once driven a verdict (0 of 128 failures); after it, it did.

The deeper lesson, from `eval/judge.py`'s own rubric: the judge explicitly does
not penalise length or extra detail, so a longer answer is PROTECTIVE — more
surface area, more chances to state the gold fact, no cost. Compressing to a
committed 5-word claim throws that away and adds contradiction risk. Verbosity
was not the bug.

Repopulate this set only with evidence that a category's golds are genuinely
transition-shaped AND that the change wins on measurement, not on argument."""
"""Question types whose gold answers are transition-shaped.

MedLoCoMo golds for these are almost always a before/after pair — "topiramate to
valproate", "from unspecified to candidal esophagitis", "meningeal signs vs
lupus flare" — with a median length of 4 words.

`frequency_pattern` is deliberately NOT here: its golds are single items
("respiratory failure", "in every admission"), not transitions, and forcing a
before/after on them would produce a wrong shape."""


def _transition_schema(base: dict) -> dict:
    """`base` plus required `before_state`/`after_state` string fields.

    Adding two REQUIRED schema fields, rather than asking for the transition in
    the prompt, because prompting did not work. Measured 2026-08-19: a STEP 3
    instruction telling the reader to "name the before-state and the after-state
    explicitly" produced an answer byte-comparable to the run without it, in
    both `chain_of_note` and `direct` modes. The model read the instruction
    (verified by capturing the system prompt) and narrated anyway.

    Schema-constrained decoding does what the instruction could not, which is
    the same conclusion this project reached for extraction (`pipeline/
    extract.py`: schema-guided beats "respond in JSON", literature/17
    R-GSJ-001/002). The model cannot emit the object without filling both
    fields, so it has to choose a start state and an end state."""
    out = json.loads(json.dumps(base))
    out["properties"]["before_state"] = {"type": "string"}
    out["properties"]["after_state"] = {"type": "string"}
    out["required"] = list(out["required"]) + ["before_state", "after_state"]
    return out


TRANSITION_INSTRUCTION = (
    "This question asks how something CHANGED between two points in time. In "
    "addition to the fields above you must fill `before_state` and "
    "`after_state`: the EARLIEST and the LATEST state of the specific thing "
    "asked about, each a short clinical noun phrase of about two to five words "
    "-- no dates, no hedging, no explanation. Set `answer` to exactly "
    '"<before_state> to <after_state>". Do not answer with a survey of '
    "everything that happened; name the two endpoints. If the evidence does "
    "not support a change, set abstained=true instead of inventing one.\n"
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


def _premise_check_schema(base: dict) -> dict:
    """Arm P2 (Change D): `base` (`_CON_SCHEMA`) with a descriptive
    `premise_check` field inserted and the property order changed to
    `notes -> premise_check -> abstained -> answer`.

    Structured-output decoding emits properties in schema order, so the
    control's `notes -> answer -> abstained` order lets the model write
    `answer` first and have `abstained` merely rationalise it after the
    fact. Reordering so `abstained` is decided BEFORE `answer` is written
    forces the decision to happen first; `premise_check` gives it a place
    to reason about that decision instead of jumping straight to a
    boolean.

    Deep-copies `base` (same `json` round-trip `_transition_schema` uses)
    so this never shares, and cannot accidentally mutate, `_CON_SCHEMA`'s
    own dict — the control's schema object must stay untouched.

    `premise_check` MUST stay a DESCRIPTIVE field ("name the thing asked
    for and which notes assert it"), never a yes/no verdict, and must never
    be concatenated into `answer`. `TRANSITION_TYPES`/`_transition_schema`
    (above) is the recorded cautionary tale: an extra REQUIRED schema field
    there manufactured a contradicting clinical term that the judge
    penalised, once its content started leaking toward the answer's
    meaning. `premise_check` here is read by the model as reasoning
    scaffolding, never read back by this module into `answer`."""
    out = json.loads(json.dumps(base))
    notes_schema = out["properties"]["notes"]
    out["properties"] = {
        "notes": notes_schema,
        "premise_check": {"type": "string"},
        "abstained": {"type": "boolean"},
        "answer": {"type": "string"},
    }
    out["required"] = ["notes", "premise_check", "abstained", "answer"]
    return out


_CON_SCHEMA_PREMISE_CHECK = _premise_check_schema(_CON_SCHEMA)
"""Arm P2's schema — built once at import time from `_CON_SCHEMA`, never
mutating it. Selected in `read()` only when `premise_check_schema=True`
and `mode == "chain_of_note"`; the control (`medmemgraph`) never
constructs or touches this object, so `_CON_SCHEMA` itself stays exactly
as it was before Change D."""


def describe_evidence(
    items: list[RetrieveItem], rendering: str, *, corpus: EvidenceProvenance | None = None
) -> str:
    """One line telling the reader what its evidence actually IS.

    Before this existed the reader was told the opposite of the truth. The
    system prompt says the evidence is "already selected for this question, from
    this patient's record", and nothing anywhere said it was a top-k SAMPLE.
    Meanwhile `eval/baselines/fullctx.py` tells its model it has "FULL access ...
    spanning every hospital admission on record" and that the history is "oldest
    first".

    That asymmetry is not cosmetic on this benchmark. "Which symptom was most
    consistently reported?" is unanswerable in principle from 6 of ~2,000 units,
    and a model that has not been told its evidence is a sample will answer it
    confidently anyway — which is exactly the judge feedback on our
    `frequency_pattern` losses ("stated fever was the most consistently reported
    symptom while the gold reference stated ...").

    Stating the sampling honestly does not make those items answerable. It lets
    the model distinguish "the evidence says X" from "the evidence I was given
    happens to contain X", which is the difference between a wrong answer and a
    correct abstention."""
    ordering = (
        "shuffled for reading (the Item N ordinal is still retrieval rank)"
        if rendering == "shuffled"
        else "ordered by retrieval relevance, NOT chronologically"
    )
    parts = [f"{len(items)} evidence item(s), {ordering}"]
    if corpus is not None:
        if corpus.total_units:
            parts.append(f"selected from ~{corpus.total_units} units in the record")
        if corpus.n_admissions:
            span = ""
            if corpus.first_time and corpus.last_time:
                span = f" spanning {corpus.first_time[:10]} to {corpus.last_time[:10]}"
            parts.append(f"across {corpus.n_admissions} admission(s){span}")
    tail = ""
    if corpus is not None and corpus.n_admissions:
        sessions = len({i.session_id for i in items if i.session_id})
        if sessions:
            tail = f"; these items touch {sessions} of those {corpus.n_admissions} admissions"
    return "EVIDENCE PROVENANCE: " + ", ".join(parts) + tail + "."


def _build_user_content(
    question: str,
    context_text: str,
    structural_absence: bool,
    rendering: str,
    *,
    items: list[RetrieveItem] | None = None,
    corpus: EvidenceProvenance | None = None,
) -> str:
    provenance = ""
    if items is not None:
        provenance = describe_evidence(items, rendering, corpus=corpus) + "\n\n"
    return (
        f"{provenance}"
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
    corpus: EvidenceProvenance | None = None,
    commit_style: bool = False,
    question_type: str | None = None,
    premise_check_prompt: bool = False,
    premise_check_schema: bool = False,
    aggregation_prompt: bool = False,
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
    docstring's "Optional post-step" section for the full contract.

    `premise_check_prompt` (arm P1) and `premise_check_schema` (arm P2) are
    independently toggleable, both default False, both no-ops outside
    `mode="chain_of_note"` (they touch `_CON_SYSTEM`/`_CON_SCHEMA`, not the
    `direct` mode's prompt/schema) — mirrors `commit_style`'s flag
    plumbing exactly, so `control` (both False), `P1`, `P2`, and `P1+P2`
    (both True) are four reachable combinations from the same code path.

    `aggregation_prompt` (arm R1) is a fifth, independent toggle: default
    False, deliberately uncoupled from `premise_check_prompt`/
    `premise_check_schema`. Its prompt splice (replacing
    `_CON_STEP3_AGGREGATION`, the "(a) TIME / (b) COMPLETENESS" paragraph —
    not `_CON_STEP2_REASON` or `_CON_SCHEMA`) is a no-op outside
    `mode="chain_of_note"`, same as P1/P2, so it composes with either or
    both without this function needing to know about that combination in
    advance. Unlike P1/P2, it ALSO gates two `render_context()` rendering
    fixes (graph TIMELINE/CONTEXT timestamp recognition; dropping the
    RRF-overwritten `score` field) that apply in EITHER mode, since
    rendering happens once before mode-specific prompt/schema selection —
    see `render_context`'s docstring."""
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

    context_text = render_context(items, rendering, rng=rng, aggregation_prompt=aggregation_prompt)
    system_prompt = _CON_SYSTEM if mode == "chain_of_note" else _DIRECT_SYSTEM
    if premise_check_prompt and mode == "chain_of_note":
        system_prompt = system_prompt.replace(
            _CON_STEP2_REASON, _CON_STEP2_PREMISE_CHECK
        )
    if aggregation_prompt and mode == "chain_of_note":
        system_prompt = system_prompt.replace(
            _CON_STEP3_AGGREGATION, _CON_STEP3_AGGREGATION_R1
        )
    if commit_style:
        system_prompt = system_prompt.replace(
            _RESPOND_LINE, COMMIT_INSTRUCTION + _RESPOND_LINE
        )
    user_content = _build_user_content(
        question, context_text, structural_absence, rendering, items=items, corpus=corpus
    )
    schema = _CON_SCHEMA if mode == "chain_of_note" else _DIRECT_SCHEMA
    if premise_check_schema and mode == "chain_of_note":
        schema = _CON_SCHEMA_PREMISE_CHECK
    transition = commit_style and question_type in TRANSITION_TYPES
    if transition:
        schema = _transition_schema(schema)
        system_prompt = system_prompt.replace(
            _RESPOND_LINE, TRANSITION_INSTRUCTION + _RESPOND_LINE
        )

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
    if transition and not payload.get("abstained"):
        before = str(payload.get("before_state") or "").strip()
        after = str(payload.get("after_state") or "").strip()
        if before and after:
            # Compose rather than trust the model's own `answer` field: asked
            # for both the endpoints AND a composed answer, the model reliably
            # fills the endpoints and then writes a paragraph into `answer`
            # anyway. The endpoints are the part the schema actually forces.
            answer_text = f"{before} to {after}"
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


def _accepts_gold_labels(retriever: Callable) -> bool:
    """Can this retriever take the benchmark's `scope`/`question_type` labels?

    `retrieve`/`retrieve_for_eval` can (keyword-only params / `**kwargs`);
    `contracts.mock_retrieve` cannot. Introspected rather than type-checked so a
    caller-injected retriever (tests, `dense.py`/`lexical.py`'s timing wrapper)
    participates without needing to know about this at all. Unintrospectable
    callables answer False — degrading to the old positional call, never a
    TypeError mid-run."""
    try:
        params = inspect.signature(retriever).parameters
    except (TypeError, ValueError):
        return False
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return True
    return "scope" in params and "question_type" in params


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
        commit_style: bool = False,
        premise_check_prompt: bool = False,
        premise_check_schema: bool = False,
        aggregation_prompt: bool = False,
    ) -> None:
        self.rendering = rendering
        self.k = k
        self.dry_run = dry_run
        self.model = model
        self.guardrail_enabled = guardrail_enabled
        self.guardrail_policy = guardrail_policy
        self.commit_style = commit_style
        self.premise_check_prompt = premise_check_prompt
        self.premise_check_schema = premise_check_schema
        self.aggregation_prompt = aggregation_prompt
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
        self._retriever_takes_labels = _accepts_gold_labels(self.retriever)

    def _retrieve(
        self,
        question: str,
        patient_id: str,
        *,
        scope: str | None,
        question_type: str | None,
    ) -> RetrieveResult:
        """Call the retriever, forwarding the benchmark's gold `scope` /
        `question_type` when it can accept them.

        This is load-bearing, not cosmetic. `graph/router.py` has two entry
        points: `route_eval`, which applies the FROZEN gold-label rule that is
        the actual thing under evaluation, and `route_live`, a keyword-regex
        heuristic that stands in when no labels exist (demo time). `retrieve()`
        picks `route_eval` only if `scope` or `question_type` is non-None.

        Until 2026-08-17 this method called `self.retriever(question, patient_id,
        self.k)` positionally, so every eval item hit `route_live` — meaning the
        reported numbers measured a keyword heuristic, not the routing rule the
        project's whole graph-vs-vector argument rests on, and
        `retrieve_for_eval`'s epsilon=0 pinning was guarding a decision that was
        never made from gold labels in the first place. Concretely, an
        `adversarial` item only reached the graph arm (the abstention signal) if
        its wording happened to match `_GRAPH_KEYWORD_RE`.

        `mock_retrieve` takes exactly `(question, patient_id, k)`, so the forward
        is conditional rather than unconditional."""
        if self._retriever_takes_labels:
            return self.retriever(
                question, patient_id, self.k, scope=scope, question_type=question_type
            )
        return self.retriever(question, patient_id, self.k)

    def answer(
        self,
        question: str,
        conversation: Conversation | None,
        *,
        patient_id: str,
        scope: str | None = None,
        question_type: str | None = None,
    ) -> AnswerResult:
        # `conversation` is NOT used as evidence — this system answers over
        # retrieved items, and reading raw turns here would quietly turn it into
        # a full-context baseline. It is used only to describe what the evidence
        # was drawn from (how many admissions, over what span), which is the
        # denominator the reader otherwise never sees. See `describe_evidence`.
        corpus = _provenance_of(conversation)
        pack = self._retrieve(question, patient_id, scope=scope, question_type=question_type)
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
            corpus=corpus,
            commit_style=self.commit_style,
            question_type=question_type,
            premise_check_prompt=self.premise_check_prompt,
            premise_check_schema=self.premise_check_schema,
            aggregation_prompt=self.aggregation_prompt,
        )
        return AnswerResult(
            text=result.text,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            latency_ms=result.latency_ms,
            abstained=result.abstained,
        )


class ReaderDirectAnswerer(ReaderAnswerer):
    mode = "direct"
    name = "reader_direct"


class ReaderChainOfNoteAnswerer(ReaderAnswerer):
    mode = "chain_of_note"
    name = "reader_con"


class MedMemGraphAnswerer(ReaderChainOfNoteAnswerer):
    """The headline system: the real graph/vector router + Chain-of-Note.

    Registered as its own system rather than relying on
    `$MEDMEMGRAPH_USE_REAL_RETRIEVE=1` flipping `reader_con`'s meaning, for two
    reasons:

    1. **It makes the results table honest.** `reader_con` is the Chain-of-Note
       READING ablation, scored on `mock_retrieve` evidence so that reading
       strategy is the only variable against `reader_direct`. This system varies
       retrieval instead. Collapsing both onto one row name, switched by an
       environment variable, would make a results file's meaning depend on the
       shell that produced it.

    2. **The env-var path fails silently in the worst direction.** If it is
       unset (or set to anything but the exact string `"1"`), the reader falls
       back to `contracts.mock_retrieve`; if that returns nothing, `read()`
       short-circuits to `NOT_IN_RECORD` at zero tokens and zero cost. The run
       completes, writes a full results file, spends nothing, and scores near
       zero on every answerable item — a real-looking table built on no
       retrieval at all.

    `retrieve_for_eval`, not `retrieve`: it pins `epsilon=0` and raises if a
    caller tries to override, so reported tables run the frozen deterministic
    routing rule rather than carrying ~5% exploration-flipped routes."""

    name = "medmemgraph"

    DEFAULT_K = 40
    """Evidence items handed to the reader.

    Measured on 10 patients / 336 paired items, 2026-08-19 — answerable accuracy
    against the full-context baseline's 0.757:

        k=6    0.674     k=40   0.783     k=60   0.775

    The curve peaks near 40 and turns DOWN by 60: more evidence past that point
    costs accuracy as well as tokens and latency. k=40 also beat k=60 on
    abstention (0.583 vs 0.550) at 28% fewer tokens and 37% lower p50 latency,
    so it wins on every axis measured rather than trading one for another.

    The previous default of 6 was not chosen — it was `ReaderAnswerer`'s
    undocumented default, and `--retrieve-k` could not override it because
    `_build_answerer` gated on a signature check this class's `**kwargs`
    constructor failed (see `eval/harness._accepts`)."""

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("k", self.DEFAULT_K)
        if kwargs.get("retriever") is None:
            from medmemgraph.graph.retrieve import retrieve_for_eval

            kwargs["retriever"] = retrieve_for_eval
        super().__init__(**kwargs)


# ---------------------------------------------------------------------------
# Toggleable arms — same retrieval/reading as `medmemgraph` (the control),
# differing only in the abstention-trigger mechanism under test, so a cheap
# adversarial-only screen (`harness.evaluate(..., only_question_types=
# ["adversarial"])`) can compare `control`, `P1`, `P2`, `P1+P2` paired on
# identical items. Each subclass only sets its own toggle's default via
# `kwargs.setdefault` (never overrides an explicit caller value), then
# defers to `MedMemGraphAnswerer.__init__` for the k=40 / retrieve_for_eval
# wiring every arm shares. Plain `medmemgraph` (above) sets neither flag and
# is therefore unchanged by their existence.
# ---------------------------------------------------------------------------


class MedMemGraphP1Answerer(MedMemGraphAnswerer):
    """Arm P1: `_CON_SYSTEM`'s STEP 2 replaced by an explicit premise-check
    step (Change C, `_CON_STEP2_PREMISE_CHECK`) — abstention no longer
    depends solely on "every note is IRRELEVANT"."""

    name = "medmemgraph_p1"

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("premise_check_prompt", True)
        super().__init__(**kwargs)


class MedMemGraphP2Answerer(MedMemGraphAnswerer):
    """Arm P2: `_CON_SCHEMA` reordered so `abstained` is decided (with a
    descriptive `premise_check` reasoning field) before `answer` is written
    (Change D, `_CON_SCHEMA_PREMISE_CHECK`)."""

    name = "medmemgraph_p2"

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("premise_check_schema", True)
        super().__init__(**kwargs)


class MedMemGraphP1P2Answerer(MedMemGraphAnswerer):
    """Arm P1+P2: both the prompt premise-check (Change C) and the schema
    decide-before-answer reordering (Change D) together."""

    name = "medmemgraph_p1p2"

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("premise_check_prompt", True)
        kwargs.setdefault("premise_check_schema", True)
        super().__init__(**kwargs)


class MedMemGraphR1Answerer(MedMemGraphAnswerer):
    """Arm R1: `_CON_SYSTEM`'s consequences paragraph replaced by an
    explicit count-across-notes aggregation procedure plus a yes/no-question
    discipline (`_CON_STEP3_AGGREGATION_R1`) — targets the
    longitudinal_progression / frequency_pattern / cross_admission_comparison
    shortfall, not the abstention mechanism P1/P2 target. Independent of
    `premise_check_prompt`/`premise_check_schema`: this class only sets
    `aggregation_prompt`, so it composes with P1/P2 in a later combined arm
    without either flag needing to know about the other."""

    name = "medmemgraph_r1"

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("aggregation_prompt", True)
        super().__init__(**kwargs)
