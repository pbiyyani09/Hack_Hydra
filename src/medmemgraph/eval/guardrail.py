"""eval/guardrail.py — the cite-or-abstain grounding guardrail.

`eval/retrieval_eval.py` measures whether retrieval *finds* the gold
evidence. `eval/ragas_metrics.py::faithfulness` measures, as a research
metric, whether an answer's claims are *inferable* from retrieved context.
Neither of those asks the operational question this module exists to
answer: **for one already-generated answer, right now, should it ship as
written, be trimmed, or be refused** — because in a clinical memory system
an uncited claim is indistinguishable from a confabulation, and a
confabulated medication history is a patient-safety failure. That framing
is this project's entire premise; this module is what makes "grounded"
something enforced at answer time, not just measured after the fact.

## The algorithm — decompose, then classify each claim in ONE call

`literature/06-abstention-and-calibration.md`'s own "Decision Questions"
recommendation names exactly this module's shape as its Stage 3 ("fires
only if [structural] Stage 1 and 2 both pass but the retrieved evidence is
thin/ambiguous... Decompose the drafted answer into claims and verify each
against the retrieved subgraph via an NLI-style entailment check
(RAGAS-faithfulness pattern, R-ABST-25) — one to two extra LLM calls, no
sampling"). RAGAS's own Faithfulness metric (R-ABST-25, restated in
`eval/ragas_metrics.py`'s own docstring) is "decompose the answer into a
set of statements S via an LLM, then verify each statement against the
context with a verification function"; `F = |V| / |S|`.

This module operationalizes that same decompose-then-verify shape with two
deliberate extensions beyond the bare RAGAS formula:

1. **Three-way classification, not a boolean.** RAGAS's verification step
   is binary (`attributable: bool`). A binary split conflates two very
   different situations: "the claim is general medical framing that never
   needed a citation" and "the claim asserts a specific patient fact the
   evidence never mentions." Collapsing them either over-flags routine
   clinical reasoning as ungrounded, or under-flags a genuine fabrication
   as merely "not yet cited." `check_grounding` classifies each claim as
   `"supported"` (a patient-record fact directly stated/entailed by a
   numbered evidence item — cite it), `"uncited"` (not a claimed
   patient-record fact at all — general knowledge, reasoning, or
   transitional language — needs no citation and does not fail
   groundedness), or `"unsupported"` (a claimed patient-record fact absent
   from, or contradicted by, every evidence item — the dangerous case:
   `GroundingReport.is_grounded` is `False` iff this list is non-empty).
2. **Explicit citation, verified, not merely a supported/unsupported bit.**
   Each `"supported"` claim also names *which* numbered evidence item it
   is grounded in (`cited_item`); this module cross-checks that index is
   real before trusting it (`_parse_claims` below) — the judge model
   claiming a citation to an item that does not exist is itself treated as
   `"unsupported"`, on the same principle `eval/reader.py`'s own
   `_reconcile_notes` already applies to the *reader's* per-item notes
   ("never trust a citation the model claims without verifying it's
   real").

**Honesty caveat, stated rather than hidden**: this is a decompose+verify
LLM-judge heuristic, not a validated ground-truth classifier.
`literature/06` R-ABST-49 measured RAGAS-Faithfulness itself at only 69.0%
binary hallucination-classification accuracy on HaluBench — well below
purpose-built judges like Lynx-70B (87.4%) — while RAGAS's own *separately*
reported 0.95 number is a pairwise-preference-agreement figure on a
different, curated task, not the same measurement (R-ABST-27 vs. R-ABST-49,
explicitly non-comparable per that survey). This module's classification
should be read the same way: a real, useful signal, not an oracle. That is
exactly why `enforce()`'s default policy is `"warn"` (surface the finding,
never silently rewrite a clinical answer on a heuristic's say-so) rather
than `"abstain"` — see `enforce()`'s own docstring.

## ONE judge call per answer — the story's own cost constraint

Unlike `eval/ragas_metrics.py::faithfulness` (one decomposition call + one
verification call *per claim*, a deliberate different cost-shape choice
documented in that module's own docstring), this module decomposes AND
classifies every claim in a single schema'd `medmemgraph.llm.complete()`
call (`_GROUNDING_SCHEMA` below) — the story's explicit cost-discipline
requirement ("this adds one judge-model call per answer"). The two modules
are independent estimates of the same underlying property, on purpose, not
merged into one function — `eval/ragas_metrics.py`'s own module docstring
says this explicitly ("guardrail.py does not exist yet ... the two are
meant to be independent estimates of the same property... not merged"),
and this module is that promised, now-landed counterpart.

## Cross-family judge — same argument as `judge.py` / `ragas_metrics.py`

`DEFAULT_GUARDRAIL_MODEL = llm.JUDGE_MODEL` (Google, `gemini-3.5-flash-lite`
by default) — a *different* provider family from `llm.ANSWER_MODEL`
(OpenAI, `gpt-4.1-mini`), which is what actually generated the answer being
checked. `literature/17-genai-structured-generation-and-judging.md`
(Part C.11-12) found self-preference bias large enough to skew a *medical*
rubric benchmark by up to 10 points; grading a model's own output with the
same model family is exactly the condition that bias is measured under.
`judge.py` and `ragas_metrics.py` already argue this at length; restated
here, in the third module that reuses the same constant, so nobody
"simplifies" this into a same-family setup later.

## `structural_absence` is reused, never rediscovered

`retrieve()`'s `structural_absence` flag (`contracts.RetrieveResult`,
ARCHITECTURE.md §7.6) means "the graph itself found no connecting evidence
for this patient at all" — a pre-generation, structural fact, always paired
with `items == []`. `literature/06` R-ABST-28 (the *Sufficient Context*
study) found capable models "often output incorrect answers instead of
abstaining" even when context insufficiency is, in principle, checkable
before generation — i.e. **a pre-generation structural check is more
reliable than asking a generator (or a second LLM judge) to rediscover the
same fact after the fact.** `check_grounding(..., structural_absence=True)`
therefore short-circuits immediately: zero evidence items exist to check
claims against regardless, so a judge call would either be a wasted no-op
or, worse, a second unreliable inference standing in for a structural fact
this project's retrieval layer already knows for certain. Structural
absence is *pre*-generation; this module's own check is *post*-generation
— complementary signals, not a chain where one re-derives the other (see
`tests/test_guardrail.py::TestStructuralAbsenceShortCircuits`, which
asserts zero `llm.complete` calls, not just a fast answer).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from medmemgraph import llm
from medmemgraph.contracts import RetrieveItem
from medmemgraph.eval.judge import _looks_like_abstention

__all__ = [
    "DEFAULT_GUARDRAIL_MODEL",
    "VALID_POLICIES",
    "ClaimVerdict",
    "GroundingReport",
    "EnforcedAnswer",
    "check_grounding",
    "enforce",
]

DEFAULT_GUARDRAIL_MODEL = llm.JUDGE_MODEL
"""Google, gemini-3.5-flash-lite by default (override via
`MEDMEMGRAPH_JUDGE_MODEL`, read once by `llm.py` at import time — never
hardcode a model string here, per project cost policy). Deliberately the
*same* cross-family judge constant `judge.py`/`ragas_metrics.py` already
use — see module docstring's "Cross-family judge" section."""

_ABSTAIN_TEXT = "NOT_IN_RECORD"
"""The exact sentinel `eval/reader.py`'s own Chain-of-Note/direct prompts
use for a declined answer (`_DIRECT_SYSTEM`/`_CON_SYSTEM`,
`read()`'s own `answer_text.strip().upper() == "NOT_IN_RECORD"` check).
`enforce()`'s `"abstain"` policy reuses this literal string, not an
invented refusal phrase, so downstream consumers that already recognize
this sentinel (the reader itself, `judge._looks_like_abstention` indirectly
via the same convention) treat a guardrail-triggered refusal identically to
one the reader produced on its own."""

VALID_POLICIES = frozenset({"warn", "strip", "abstain"})

_GUARDRAIL_MAX_TOKENS = 1500
"""Room for decomposition + a three-way status + a citation index + a
one-sentence reason, per claim, in a single completion — larger than
`ragas_metrics.py`'s per-call budgets (800/300) because this module does
decomposition and verification together in one call rather than splitting
them (see module docstring's "ONE judge call per answer")."""


# ---------------------------------------------------------------------------
# Schema + prompt
# ---------------------------------------------------------------------------

_GROUNDING_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "status": {
                        "type": "string",
                        "enum": ["supported", "uncited", "unsupported"],
                    },
                    "cited_item": {"type": "integer"},
                    "reason": {"type": "string"},
                },
                "required": ["text", "status", "cited_item", "reason"],
                "additionalProperties": False,
            },
        },
        "confidence": {"type": "number"},
    },
    "required": ["claims", "confidence"],
    "additionalProperties": False,
}

_GROUNDING_SYSTEM = (
    "You check whether a clinical system's answer is properly grounded in "
    "the RETRIEVED EVIDENCE it was given, per a cite-or-abstain policy: "
    "every patient-specific factual claim in the answer must be traceable "
    "to a specific numbered evidence item, or it is a hazard. Apply this "
    "procedure, in this exact order, every time:\n"
    "1. DECOMPOSE the answer into standalone atomic claims (RAGAS "
    "Faithfulness's own rule: one independently-verifiable fact per claim "
    "-- never combine two facts into one claim; resolve pronouns/vague "
    "references using the question's own wording; do not add a fact the "
    "answer does not state, and do not omit one it does). Skip pure "
    "transitional language that asserts no fact at all (e.g. 'Based on the "
    "record,'). If the answer asserts no factual content at all, return an "
    "empty claims list.\n"
    "2. For EACH claim, classify `status` as exactly one of:\n"
    "   - \"supported\": the claim states a specific patient-record fact "
    "(a medication, dose, diagnosis, symptom, date, event, etc.) that is "
    "directly stated or straightforwardly entailed by ONE of the numbered "
    "evidence items below. Set `cited_item` to that item's number. Never "
    "mark a claim supported because it merely sounds medically plausible "
    "-- only because an evidence item actually states it.\n"
    "   - \"unsupported\": the claim states a specific patient-record fact "
    "that is NOT found anywhere in the evidence items, or that contradicts "
    "what an evidence item states. This is the dangerous case -- a "
    "medication, dose, diagnosis, or event invented or misremembered by "
    "the system. Set `cited_item` to -1.\n"
    "   - \"uncited\": the claim is general medical knowledge, reasoning, "
    "or a qualifying/transitional statement -- NOT a claimed patient-record "
    "fact at all -- so it needs no evidence citation (e.g. \"this is "
    "commonly prescribed for type 2 diabetes\" said in general, not about "
    "this patient specifically). Set `cited_item` to -1.\n"
    "3. Report an overall `confidence` (0.0-1.0) in this assessment; lower "
    "it when the evidence is ambiguous or the answer's wording makes "
    "classification genuinely hard.\n"
    "Respond with the required JSON only: `claims` (one object per "
    "extracted claim, in the order they appear in the answer) and "
    "`confidence`."
)


def _grounding_user(question: str, answer: str, items: list[RetrieveItem]) -> str:
    evidence = "\n".join(f"[Item {i}] {it.text}" for i, it in enumerate(items, start=1))
    q_part = f"Question: {question}\n\n" if question.strip() else ""
    return f"{q_part}RETRIEVED EVIDENCE:\n{evidence}\n\nSYSTEM'S ANSWER TO CHECK:\n{answer}"


# ---------------------------------------------------------------------------
# Result shapes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClaimVerdict:
    """One atomic claim's classification. `cited_item` is a real
    `RetrieveItem` only for a `status == "supported"` claim whose citation
    survived `_parse_claims`'s index check; `None` otherwise (including a
    `"supported"` claim downgraded to `"unsupported"` for citing a
    nonexistent item — see module docstring)."""

    text: str
    status: str  # "supported" | "uncited" | "unsupported"
    cited_item: RetrieveItem | None
    reason: str


@dataclass(frozen=True)
class GroundingReport:
    """`check_grounding()`'s return shape.

    `is_grounded` is `True` iff `unsupported_claims` is empty --
    `uncited_claims` alone (general knowledge/reasoning that never claimed
    a patient fact) does not fail groundedness; only a claimed-but-absent
    patient fact does (module docstring's "Three-way classification"
    section). `shortcut_reason` is `""` for a real judge call, and one of
    `"structural_absence"` / `"abstained_answer"` / `"no_evidence"` /
    `"disabled"` when the check (or the whole guardrail) was skipped
    without spending a call -- see `check_grounding()`/`enforce()`. Every
    field is populated even in the shortcut cases (never `None` standing in
    for "didn't check"), so a caller never has to special-case a shortcut
    report before reading it."""

    is_grounded: bool
    citations: list[RetrieveItem]
    uncited_claims: list[str]
    unsupported_claims: list[str]
    confidence: float
    claims: list[ClaimVerdict] = field(default_factory=list)
    cost_usd: float = 0.0
    n_judge_calls: int = 0
    enabled: bool = True
    shortcut_reason: str = ""


@dataclass(frozen=True)
class EnforcedAnswer:
    """`enforce()`'s return shape. `modified` is `True` iff `text` differs
    from the input `answer` -- `"warn"` never sets it (see `enforce()`'s
    own docstring: annotation lives in `.report`, not spliced into the
    text)."""

    text: str
    report: GroundingReport
    policy: str
    modified: bool


# ---------------------------------------------------------------------------
# Shortcut reports — the zero-LLM-call paths
# ---------------------------------------------------------------------------


def _shortcut_report(*, reason: str, whole_answer: str = "", is_hazard: bool = False) -> GroundingReport:
    """Build a `GroundingReport` for one of the zero-`llm.complete`-call
    paths. `is_hazard=True` (only the `"no_evidence"` case) flags the
    *entire* answer as one unsupported claim -- deterministic, not a judge
    call, because with zero retrieved evidence there is nothing any
    decomposition could verify a patient-specific claim against (same "no
    context -> unsupported by construction" convention `ragas_metrics.py`'s
    `faithfulness()` already documents for its own empty-context case)."""
    if is_hazard:
        return GroundingReport(
            is_grounded=False,
            citations=[],
            uncited_claims=[],
            unsupported_claims=[whole_answer],
            confidence=1.0,
            claims=[],
            cost_usd=0.0,
            n_judge_calls=0,
            enabled=True,
            shortcut_reason=reason,
        )
    return GroundingReport(
        is_grounded=True,
        citations=[],
        uncited_claims=[],
        unsupported_claims=[],
        confidence=1.0,
        claims=[],
        cost_usd=0.0,
        n_judge_calls=0,
        enabled=True,
        shortcut_reason=reason,
    )


def _is_abstention(answer: str) -> bool:
    stripped = answer.strip()
    return not stripped or stripped.upper() == _ABSTAIN_TEXT or _looks_like_abstention(answer)


# ---------------------------------------------------------------------------
# check_grounding()
# ---------------------------------------------------------------------------


def check_grounding(
    answer: str,
    retrieved_items: list[RetrieveItem],
    *,
    question: str = "",
    structural_absence: bool = False,
    model: str = DEFAULT_GUARDRAIL_MODEL,
    dry_run: bool = False,
    use_cache: bool = True,
    temperature: float = 0.0,
) -> GroundingReport:
    """Decompose `answer` into atomic claims and classify each against
    `retrieved_items` in one judge-model call (module docstring's
    algorithm section). Three zero-call short-circuits run first, in this
    order, each returning a `GroundingReport` with `n_judge_calls=0` and a
    non-empty `shortcut_reason`:

    1. `structural_absence=True` -- reuse the retrieval layer's own signal
       (module docstring's "structural_absence is reused" section);
       `is_grounded=True` (nothing to hallucinate about, by construction).
    2. `answer` is empty or already looks like a decline (`_is_abstention`,
       reusing `judge._looks_like_abstention` rather than a second phrase
       list) -- an honest refusal makes no factual claims to check.
    3. `retrieved_items` is empty (defensive: covers a caller that passes
       `items=[]` without also setting `structural_absence`) -- zero
       context means any patient-specific claim in a non-abstaining answer
       is unsupported by construction; the whole answer is flagged as one
       hazard claim rather than spending a call a decomposition could not
       usefully verify anything against.

    Otherwise, exactly one `llm.complete(schema=_GROUNDING_SCHEMA)` call is
    made (`n_judge_calls=1`), and `GroundingReport.cost_usd` carries that
    call's real, measured cost (`llm.py`'s own ledger/cache/budget-cap
    machinery applies automatically, no extra code here)."""
    answer = answer or ""

    if structural_absence:
        return _shortcut_report(reason="structural_absence")

    if _is_abstention(answer):
        return _shortcut_report(reason="abstained_answer")

    if not retrieved_items:
        return _shortcut_report(reason="no_evidence", whole_answer=answer, is_hazard=True)

    response = llm.complete(
        _grounding_user(question, answer, retrieved_items),
        model=model,
        schema=_GROUNDING_SCHEMA,
        system=_GROUNDING_SYSTEM,
        max_tokens=_GUARDRAIL_MAX_TOKENS,
        temperature=temperature,
        dry_run=dry_run,
        use_cache=use_cache,
    )
    claims = _parse_claims(response.parsed or {}, retrieved_items)
    confidence = _parse_confidence(response.parsed or {})
    citations = _dedupe_citations(claims)
    uncited = [c.text for c in claims if c.status == "uncited"]
    unsupported = [c.text for c in claims if c.status == "unsupported"]

    return GroundingReport(
        is_grounded=not unsupported,
        citations=citations,
        uncited_claims=uncited,
        unsupported_claims=unsupported,
        confidence=confidence,
        claims=claims,
        cost_usd=response.cost_usd,
        n_judge_calls=1,
        enabled=True,
        shortcut_reason="",
    )


def _parse_claims(data: dict, items: list[RetrieveItem]) -> list[ClaimVerdict]:
    """Build `ClaimVerdict`s from the judge's raw parsed JSON, with two
    fail-safe rules -- both "fail toward the dangerous-looking case," the
    same direction `llm.py`'s own budget-estimate-overestimate and
    `ragas_metrics.py`'s malformed-length-array-padding already choose:

    1. An unrecognized `status` string (a schema-shape drift, not something
       the hand-rolled enum check in `llm.py` should ever actually let
       through, but never trusted positionally either way) is treated as
       `"unsupported"`, not silently dropped or defaulted to the safe-
       looking `"supported"`.
    2. A `"supported"` claim whose `cited_item` does not index a real
       retrieved item (out of `[1, len(items)]`, or unparseable) is
       downgraded to `"unsupported"` -- an ungrounded citation is itself
       indistinguishable from a confabulation dressed up as grounded (see
       module docstring; mirrors `eval/reader.py::_reconcile_notes`'s
       identical guarantee for the reader's own per-item notes)."""
    n = len(items)
    out: list[ClaimVerdict] = []
    for raw in data.get("claims") or []:
        if not isinstance(raw, dict):
            continue
        text = str(raw.get("text", "")).strip()
        if not text:
            continue
        status = str(raw.get("status", "")).strip().lower()
        reason = str(raw.get("reason", ""))
        cited_item: RetrieveItem | None = None
        if status == "supported":
            try:
                idx = int(raw.get("cited_item"))
            except (TypeError, ValueError):
                idx = -1
            if 1 <= idx <= n:
                cited_item = items[idx - 1]
            else:
                status = "unsupported"
        elif status not in ("uncited", "unsupported"):
            status = "unsupported"
        out.append(ClaimVerdict(text=text, status=status, cited_item=cited_item, reason=reason))
    return out


def _dedupe_citations(claims: list[ClaimVerdict]) -> list[RetrieveItem]:
    """The distinct `RetrieveItem`s backing at least one `"supported"`
    claim, in first-cited order -- `GroundingReport.citations`, "the
    retrieved items the answer actually draws on" (story requirement).
    Deduped by `(session_id, turn_ids)`, matching `contracts.RetrieveItem`'s
    own identity fields (it is not itself hashable/frozen)."""
    seen: set[tuple[str, tuple[int, ...]]] = set()
    out: list[RetrieveItem] = []
    for c in claims:
        if c.cited_item is None:
            continue
        key = (c.cited_item.session_id, tuple(c.cited_item.turn_ids))
        if key in seen:
            continue
        seen.add(key)
        out.append(c.cited_item)
    return out


def _parse_confidence(data: dict) -> float:
    try:
        value = float(data.get("confidence", 1.0))
    except (TypeError, ValueError):
        value = 1.0
    return max(0.0, min(1.0, value))


# ---------------------------------------------------------------------------
# enforce()
# ---------------------------------------------------------------------------


def enforce(
    answer: str,
    retrieved_items: list[RetrieveItem],
    policy: str = "warn",
    *,
    enabled: bool = True,
    question: str = "",
    structural_absence: bool = False,
    model: str = DEFAULT_GUARDRAIL_MODEL,
    dry_run: bool = False,
    use_cache: bool = True,
    temperature: float = 0.0,
    report: GroundingReport | None = None,
) -> EnforcedAnswer:
    """Apply one of three policies to `answer`, based on its grounding.

    - `"warn"` (the default): **never touches `text`** -- `text ==
      answer`, `modified=False`, always. The "annotation" is
      `EnforcedAnswer.report` itself (`uncited_claims`/`unsupported_claims`/
      `is_grounded`), which every caller already gets back regardless of
      policy. Silently rewriting a clinical answer is its own hazard
      (story requirement), so the default policy makes zero content
      changes -- a caller must explicitly opt into `"strip"`/`"abstain"` to
      have this module ever alter what a patient/clinician actually reads.
    - `"strip"`: if `unsupported_claims` is empty, a no-op (`text ==
      answer`). Otherwise, rebuild `text` from only the surviving
      (`"supported"`/`"uncited"`) claim texts, in original order -- this
      deliberately paraphrases rather than attempting a literal substring
      deletion, because a decomposed claim is a *resolved* restatement
      (pronouns/vague references resolved per the decomposition prompt),
      not guaranteed to be a verbatim substring of `answer` to begin with.
      If nothing survives (every claim was unsupported, or the shortcut
      no-evidence case flagged the whole answer), `text` becomes the same
      `_ABSTAIN_TEXT` sentinel `"abstain"` uses below -- there is no
      partial answer left to return.
    - `"abstain"`: if `report.is_grounded` is `False`, `text` becomes the
      literal `_ABSTAIN_TEXT` sentinel (`"NOT_IN_RECORD"`, matching
      `eval/reader.py`'s own convention exactly). If already grounded, a
      no-op.

    `enabled=False` is a genuine no-op: returns `text == answer`
    immediately, with a `GroundingReport(enabled=False,
    shortcut_reason="disabled")` and **without calling
    `check_grounding`/`llm.complete` at all** -- verified directly by
    `tests/test_guardrail.py::TestDisabledIsANoOp` via a trip-wire fake.
    This is the flag `eval/reader.py`'s optional post-step threads through
    so the harness A/B can measure the guardrail's effect against a run
    where it made zero difference by construction, not merely by policy
    choice (story requirement: "so the A/B measures its effect rather than
    assuming it").

    `report=...` lets a caller who already ran `check_grounding()` (e.g.
    `eval/reader.py`, which needs the report either way) reuse it instead
    of paying for a second judge call; when omitted, `enforce()` calls
    `check_grounding()` itself with the other keyword arguments."""
    if policy not in VALID_POLICIES:
        raise ValueError(f"unknown guardrail policy {policy!r}; expected one of {sorted(VALID_POLICIES)}")
    answer = answer or ""

    if not enabled:
        disabled_report = GroundingReport(
            is_grounded=True,
            citations=[],
            uncited_claims=[],
            unsupported_claims=[],
            confidence=1.0,
            claims=[],
            cost_usd=0.0,
            n_judge_calls=0,
            enabled=False,
            shortcut_reason="disabled",
        )
        return EnforcedAnswer(text=answer, report=disabled_report, policy=policy, modified=False)

    rep = report if report is not None else check_grounding(
        answer, retrieved_items, question=question, structural_absence=structural_absence,
        model=model, dry_run=dry_run, use_cache=use_cache, temperature=temperature,
    )

    if policy == "warn":
        return EnforcedAnswer(text=answer, report=rep, policy=policy, modified=False)

    if policy == "strip":
        if not rep.unsupported_claims:
            return EnforcedAnswer(text=answer, report=rep, policy=policy, modified=False)
        surviving = [c.text for c in rep.claims if c.status != "unsupported"]
        new_text = " ".join(surviving) if surviving else _ABSTAIN_TEXT
        return EnforcedAnswer(text=new_text, report=rep, policy=policy, modified=new_text != answer)

    # policy == "abstain"
    if rep.is_grounded:
        return EnforcedAnswer(text=answer, report=rep, policy=policy, modified=False)
    return EnforcedAnswer(text=_ABSTAIN_TEXT, report=rep, policy=policy, modified=_ABSTAIN_TEXT != answer)
