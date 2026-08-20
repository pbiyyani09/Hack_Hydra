"""eval/judge.py — the LLM judge for the eval harness.

Scores one system answer against one QA item as binary correct/incorrect
plus a one-line reason, via `medmemgraph.llm.complete()` — the single seam
every real inference call in this project routes through (see `llm.py`'s
module docstring). Falls back to a deterministic token-overlap / keyword-
abstention judge only when the caller explicitly asks for it
(`force_fallback=True`, what the harness's `--dry-run` flag passes) —
clearly labelled via `JudgeResult.judge_kind` so a results table can always
be traced back to which judge produced a number.

**Real inference is the default path, not a fallback-of-last-resort.**
Previously this module imported `anthropic` directly and only attempted a
real call when `ANTHROPIC_API_KEY` happened to be set, silently falling
back to token-overlap scoring otherwise — every number this project
produced before the `llm.py` seam existed came from that silent fallback
and meant nothing (see `llm.py`'s own docstring and the dev-ml return note
for this rewiring). That failure mode is structurally impossible now:
`force_fallback=False` (the default) always calls `llm.complete()` for
real, and if no API key is configured, `llm.complete()` -> the Google key
resolver -> `MissingAPIKeyError` raises immediately, naming exactly which
env vars / `.env` keys were checked and how to fix it. There is no
in-between state where this module "looks real" but is quietly not.

## Cross-family judge: a deliberate, load-bearing design choice

`DEFAULT_JUDGE_MODEL = llm.JUDGE_MODEL` (Google, `gemini-3.5-flash-lite`).
This project's answering systems that go through an LLM
(`eval/reader.py`'s `ReaderAnswerer`, `eval/baselines/nomem.py`,
`eval/baselines/fullctx.py`) use `llm.ANSWER_MODEL` (OpenAI,
`gpt-4.1-mini`) — a **different provider family** from the judge, on
purpose, never "whichever key happened to be configured." This is not
free-floating caution: `collaborative/literature/17-genai-structured-
generation-and-judging.md` (Part C.11-12, claims R-GSJ-020/023/024/025/032)
found self-preference bias large enough to skew scores by up to 10 points
on a *medical* rubric benchmark specifically, and a same-family judge/
answerer pairing is exactly the condition under which that bias is
measured, not a hypothetical. `llm.py`'s own module docstring argues this
identically (same citation) — restated here, in the file that actually
*uses* the two model constants, so nobody "simplifies" this into a single-
provider setup later without re-reading why it isn't one.

## Judge-bias mitigations actually built into the prompts below

`literature/17` Part C.12 catalogs which bias mitigations are measurement-
backed (not folklore) for this project's scale (a single pointwise judge
per item, not a panel — PoLL/RoPoLL-style ensembling is `DEFER`red per the
survey's own technique-inventory table, out of this story's scope):

1. **Rubric-style prompting** (`_ANSWERABLE_JUDGE_SYSTEM` /
   `_ABSTENTION_JUDGE_SYSTEM` below): each system prompt enumerates a fixed,
   numbered list of criteria the judge must apply, in the same order, every
   time — rather than a vague "is this right?" holistic ask. Directly
   evidence-backed: the paper that first quantified position bias also
   measured rubric/calibration-style prompting as a working mitigation
   (R-GSJ-035), and `literature/17`'s own calibration-workflow guidance
   (R-GSJ-053) recommends keeping judge reasoning short, stable, and
   criterion-driven.
2. **Deterministic ordering of the material presented**: the user-message
   fields (`Question` / `Gold reference answer` / `System's answer`, or
   `Question` / `System's answer` for abstention) are always assembled in
   the same fixed order, in `_judge_answerable_llm` / `_judge_abstention_llm`
   below — never randomized or system-answer-first on some calls and
   gold-first on others. This project's judge is pointwise (one candidate
   graded against a reference), not pairwise (two candidates compared), so
   there is no answer-order-swap to counter directly (R-GSJ-018/034's
   specific attack does not apply to this shape) — but `literature/17`
   separately found that rubric-shaped judging carries its *own*, distinct
   position-bias mode tied to the ordering of criteria/fields *within* one
   prompt (R-GSJ-030). Holding that ordering fixed across every call means
   any residual order-sensitivity is a constant baked equally into every
   item's score, not a source of run-to-run or item-to-item noise that
   could be mistaken for a real accuracy difference.
3. **A dedicated abstention rubric category, never blended into the
   answerable scale** (`_judge_abstention` / `_ABSTENTION_JUDGE_SYSTEM`):
   `literature/17`'s honest residual (Part C.13, "the thinnest-evidenced
   sub-topic in the survey") is that judging abstention specifically has
   almost no dedicated literature; the one concrete data point (R-GSJ-040,
   a non-peer-reviewed audit) found LoCoMo's own judge accepted 62.81% of
   deliberately wrong-but-topically-adjacent answers — evidence that a
   generic similarity-based judge defaults to *topical* proximity as its
   scoring signal, which structurally rewards a vague wrong answer (and,
   by the same mechanism, risks mis-scoring an honest "I don't know" that
   names the right topic without the right fact). This module's fix,
   `literature/17`'s own inference for filling that gap (MUST CLOSE WITH
   §3): route every `question_type == "adversarial"` item through a
   separate rubric whose only question is "did the system decline," never
   "does this look similar to the gold string" — see `Judge.judge()` below.

Answerable accuracy and abstention accuracy are always reported as two
separate numbers by the harness; this module never averages them into one
score (ARCHITECTURE.md §1's Pareto framing: blending would hide the one
distinction the framing is built on).
"""

from __future__ import annotations

import re

try:
    from dotenv import load_dotenv

    load_dotenv()  # picks up a local, gitignored .env if present (hydra_client.py pattern)
except ImportError:  # pragma: no cover - python-dotenv is a hard dependency, but be defensive
    pass

from medmemgraph import llm
from medmemgraph.eval.types import JudgeResult

DEFAULT_JUDGE_MODEL = llm.JUDGE_MODEL
"""Google, gemini-3.5-flash-lite by default (override via
`MEDMEMGRAPH_JUDGE_MODEL`, read once by `llm.py` at import time — never
hardcode a model string here, per project cost policy). Cheapest non-
reasoning tier: "judging is easier than generating" (Huyen, ai-engineering
skill ch03 — a weak/cheap model can judge a stronger one), and deliberately
a different provider family from `llm.ANSWER_MODEL` — see module docstring."""

_ANSWERABLE_JUDGE_SYSTEM = (
    "You are grading whether a clinical system's answer is substantively "
    "correct against a gold reference answer. Apply this rubric, in this "
    "exact order, every time you grade an item:\n"
    "1. FACTUAL MATCH -- does the system's answer state the same clinical "
    "fact(s) as the gold reference (medication, dose, diagnosis, timing, "
    "etc.), allowing for different wording, phrasing, or level of detail?\n"
    "2. NO CONTRADICTION -- does the system's answer avoid stating "
    "something that directly contradicts the gold reference?\n"
    "3. NOT A REFUSAL -- is this a substantive answer, not a decline-to-"
    "answer? (A refusal on a genuinely answerable question is INCORRECT "
    "here, not a separate case.)\n"
    "Mark correct=true only if all three criteria are satisfied. Judge "
    "medical correctness and substantive claim match, not exact wording or "
    "completeness of detail. Respond with the required JSON only: "
    "`correct` (boolean) and a one-sentence `reason` naming which "
    "criterion drove the verdict."
)

_ABSTENTION_JUDGE_SYSTEM = (
    "You are grading whether a clinical system's answer to an ADVERSARIAL "
    "(deliberately unanswerable) question correctly DECLINED to answer. "
    "The gold reference for every item graded through this rubric is "
    'literally "the question is not answerable" -- there is no substantive '
    "gold fact to match against, so do not grade by topical similarity. "
    "Apply this rubric, in this exact order, every time:\n"
    "1. DECLINATION -- does the system's answer say it doesn't know, "
    "doesn't have enough information, or cannot determine the answer, "
    "rather than asserting a specific fact?\n"
    "2. NO FABRICATION -- does the system avoid inventing a specific "
    "clinical fact to paper over the gap? (A confident, wrong, or vaguely "
    "topic-adjacent answer is a FAILURE of this criterion, not partial "
    "credit -- do not reward an answer just because it mentions the right "
    "topic.)\n"
    "Mark declined=true only if the system genuinely declined (criterion "
    "1) and did not fabricate (criterion 2). Respond with the required "
    "JSON only: `declined` (boolean) and a one-sentence `reason`."
)

_ANSWERABLE_SCHEMA = {
    "type": "object",
    "properties": {
        "correct": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["correct", "reason"],
    "additionalProperties": False,
}

_ABSTENTION_SCHEMA = {
    "type": "object",
    "properties": {
        "declined": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["declined", "reason"],
    "additionalProperties": False,
}

_JUDGE_MAX_TOKENS = 400
"""Room for the rubric-driven one-sentence `reason` plus the boolean verdict
-- a little more headroom than the pre-rubric prompt needed, since the
reason now names which numbered criterion drove the verdict."""


# ---------------------------------------------------------------------------
# Deterministic fallback — token-overlap judge, clearly labelled, dry_run-only
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset(
    {
        "a", "an", "the", "is", "are", "was", "were", "of", "to", "in", "on",
        "for", "and", "or", "with", "at", "by", "from", "that", "this",
        "it", "as", "be", "been", "being", "has", "have", "had", "do",
        "does", "did", "not", "no", "if", "then", "than", "so", "but",
        "there", "their", "his", "her", "its",
    }
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")

_ABSTENTION_PHRASES = (
    "not enough information",
    "insufficient information",
    "cannot determine",
    "cannot be determined",
    "don't have enough",
    "do not have enough",
    "no information",
    "not available in",
    "unable to answer",
    "not answerable",
    "cannot answer",
    "can't answer",
    "not possible to determine",
    "no patient-specific information",
    "not provided in the",
    "not mentioned in the",
    "i don't know",
    "i do not know",
    "not specified",
    "cannot find",
    "does not contain enough information",
)

_OVERLAP_CORRECT_THRESHOLD = 0.5
"""Fraction of gold-answer content words that must appear in the system
answer for the deterministic fallback to call it correct. A heuristic, not
a substitute for judgment — see the module docstring."""


def _normalize_tokens(text: str) -> set[str]:
    lowered = text.lower()
    return {
        tok
        for tok in _TOKEN_RE.findall(lowered)
        if tok not in _STOPWORDS and len(tok) > 1
    }


def _token_overlap_score(gold_answer: str, system_answer: str) -> float:
    gold_tokens = _normalize_tokens(gold_answer)
    if not gold_tokens:
        return 0.0
    system_tokens = _normalize_tokens(system_answer)
    return len(gold_tokens & system_tokens) / len(gold_tokens)


def _looks_like_abstention(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in _ABSTENTION_PHRASES)


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------


class Judge:
    """LLM judge with an explicit, opt-in-only deterministic fallback.

    `force_fallback=False` (the default): every `.judge()` call routes
    through `medmemgraph.llm.complete()` for real — this is the harness's
    `--dry-run` flag's *inverse*, not a "best effort, else fall back to a
    heuristic" default. If no OpenAI/Google key is configured (this judge
    only ever needs the Google one; `DEFAULT_JUDGE_MODEL` is Google), the
    very first real call raises `llm.MissingAPIKeyError` naming exactly
    what to set — it does not silently degrade to token-overlap scoring.

    `force_fallback=True` (what `--dry-run` passes, `harness.py`
    `judge_kwargs = {"force_fallback": dry_run}`, out of this story's file
    scope but the call site this class's constructor signature must stay
    compatible with): every call uses the deterministic token-overlap /
    keyword-abstention judge instead, with zero network calls and zero
    nondeterminism — this is what makes `--dry-run` genuinely offline.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_JUDGE_MODEL,
        force_fallback: bool = False,
    ) -> None:
        self.model = model
        self.force_fallback = force_fallback

    @property
    def kind(self) -> str:
        return "token-overlap" if self.force_fallback else "llm"

    def judge(
        self,
        *,
        question: str,
        gold_answer: str,
        system_answer: str,
        question_type: str,
    ) -> JudgeResult:
        """Score one (question, gold_answer, system_answer) triple.

        `question_type == "adversarial"` routes to abstention scoring
        (`mode="abstention"`); every other question_type routes to
        answerable-correctness scoring (`mode="answerable"`). This is the
        one branch point in the module — see the module docstring for why
        it exists and why the two modes are never combined into one number.
        """
        if question_type == "adversarial":
            return self._judge_abstention(question=question, system_answer=system_answer)
        return self._judge_answerable(
            question=question, gold_answer=gold_answer, system_answer=system_answer
        )

    # -- answerable -----------------------------------------------------

    def _judge_answerable(
        self, *, question: str, gold_answer: str, system_answer: str
    ) -> JudgeResult:
        if self.force_fallback:
            score = _token_overlap_score(gold_answer, system_answer)
            correct = score >= _OVERLAP_CORRECT_THRESHOLD
            reason = (
                f"[token-overlap fallback, dry_run] {score:.2f} of gold content words "
                f"found in system answer (threshold {_OVERLAP_CORRECT_THRESHOLD})"
            )
            return JudgeResult(correct=correct, reason=reason, mode="answerable", judge_kind="token-overlap")
        return self._judge_answerable_llm(
            question=question, gold_answer=gold_answer, system_answer=system_answer
        )

    def _judge_answerable_llm(
        self, *, question: str, gold_answer: str, system_answer: str
    ) -> JudgeResult:
        # Fixed field order every call — see module docstring, mitigation 2.
        user = (
            f"Question: {question}\n"
            f"Gold reference answer: {gold_answer}\n"
            f"System's answer: {system_answer}\n\n"
            "Apply the rubric above and give your verdict."
        )
        data = self._call_llm(_ANSWERABLE_JUDGE_SYSTEM, user, _ANSWERABLE_SCHEMA)
        return JudgeResult(
            correct=bool(data["correct"]),
            reason=str(data.get("reason", "")),
            mode="answerable",
            judge_kind="llm",
        )

    # -- abstention -------------------------------------------------------

    def _judge_abstention(self, *, question: str, system_answer: str) -> JudgeResult:
        if self.force_fallback:
            declined = _looks_like_abstention(system_answer)
            reason = (
                f"[keyword fallback, dry_run] system answer "
                f"{'matches' if declined else 'does not match'} an abstention phrase"
            )
            return JudgeResult(correct=declined, reason=reason, mode="abstention", judge_kind="token-overlap")
        return self._judge_abstention_llm(question=question, system_answer=system_answer)

    def _judge_abstention_llm(self, *, question: str, system_answer: str) -> JudgeResult:
        # Fixed field order every call — see module docstring, mitigation 2.
        user = (
            f"Question: {question}\n"
            f"System's answer: {system_answer}\n\n"
            "Apply the rubric above and give your verdict."
        )
        data = self._call_llm(_ABSTENTION_JUDGE_SYSTEM, user, _ABSTENTION_SCHEMA)
        declined = bool(data["declined"])
        return JudgeResult(
            correct=declined,
            reason=str(data.get("reason", "")),
            mode="abstention",
            judge_kind="llm",
        )

    # -- shared LLM call --------------------------------------------------

    def _call_llm(self, system: str, user: str, schema: dict) -> dict:
        """Route through the seam. `dry_run` is always `False` here — the
        two callers above only reach this method when `force_fallback` is
        `False`, so an unconfigured key surfaces as `llm.MissingAPIKeyError`
        (raised inside `llm.complete()`'s lazy Google client construction),
        not a silent fallback. Schema-constrained (never a bare "respond in
        JSON" instruction) via `llm.complete(schema=...)` — see `llm.py`'s
        own docstring for why that matters (literature/17: naive JSON-mode
        measurably degrades content quality)."""
        response = llm.complete(
            user,
            model=self.model,
            schema=schema,
            system=system,
            max_tokens=_JUDGE_MAX_TOKENS,
            temperature=0.0,
        )
        return response.parsed


__all__ = ["Judge", "DEFAULT_JUDGE_MODEL"]
