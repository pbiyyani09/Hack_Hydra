"""eval/ragas_metrics.py — RAGAS-defined LLM-judged answer-quality metrics
(Faithfulness, Answer Relevancy, Context Precision, Context Recall),
computed by our own implementation, not the `ragas` library.

## Why implemented rather than imported (verified this session, not assumed)

`ragas` was installed and removed. `ragas` 0.3.x does
`from langchain_community.chat_models.vertexai import ChatVertexAI`, and the
installed `langchain-community` 0.4.2 no longer ships that module — the
package itself emits a deprecation notice that it "is being sunset and is
no longer actively maintained." `ragas>=0.4` failed to resolve on Python
3.13 (this project's pinned interpreter, see `pyproject.toml`).

So: **these are RAGAS-defined metrics computed by our own implementation**,
not the RAGAS library's own numbers — results are not bit-comparable to a
published RAGAS leaderboard. Every formula below is quoted from RAGAS's own
published definition (docs.ragas.io, fetched live this session — URLs
inline per metric) and the original paper (Es, James, Espinosa-Anke,
Schockaert, "RAGAS: Automated Evaluation of Retrieval Augmented
Generation", arXiv:2309.15217, HTML edition fetched live this session), not
reconstructed from memory.

This is genuinely better for this project, for four concrete reasons:
1. It runs on the project's own cheap, cost-capped models
   (`llm.JUDGE_MODEL`) via the existing `medmemgraph.llm` seam, instead of
   wiring `ragas` to a non-default provider through `langchain` — which
   would have been more integration work than the four metrics themselves.
2. It inherits `llm.py`'s disk cache (re-runs are free) and its hard
   `MEDMEMGRAPH_MAX_USD` budget cap (`llm.BudgetExceeded`) for free — no
   separate cost-control layer to build or trust.
3. It inherits Arize/Phoenix tracing "for free" in the architectural sense:
   every call here routes through `llm.complete()`, and
   `medmemgraph.observability.init_tracing()` (landed concurrently during
   this session — re-checked directly before finalizing this docstring,
   not left as a stale claim) already auto-instruments exactly the two
   provider SDKs `llm.py` uses (OpenAI, Google GenAI). It is opt-in
   (`enabled` from env, degrades to a no-op on any failure) and, as of
   this session, no entry point (`harness.py`, `demo/agent.py`, ...) calls
   it yet — so tracing is not active project-wide right now — but the
   moment some entry point does call it, every call this module makes is
   traced automatically, with zero changes needed here.
4. It carries this project's own judge-bias mitigations (cross-family
   judge, rubric-style prompting, fixed field ordering — see "Judge model"
   below) rather than `ragas`'s own default judge-prompt conventions.

## The four metrics — definition followed, formula, and citation

Every metric below is computed via `medmemgraph.llm.complete()` with
`model=llm.JUDGE_MODEL` (Google, `gemini-3.5-flash-lite` by default — see
"Judge model" below) and a plain JSON Schema (never a bare "respond in
JSON" instruction — same discipline `judge.py`/`reader.py` already use).

1. **Faithfulness** — docs.ragas.io/en/stable/concepts/metrics/
   available_metrics/faithfulness/ (fetched live this session):
   "Faithfulness Score = Number of claims in the response supported by the
   retrieved context / Total number of claims in the response." Original
   paper (arXiv:2309.15217, HTML, fetched live this session): decompose the
   answer into a set of statements S via an LLM ("decompose longer
   sentences into shorter and more focused assertions"), then verify each
   statement against the context with a verification function v(s_i,
   c(q)); **F = |V| / |S|** where V is the subset the LLM judged
   inferable from the context. Implemented in `faithfulness()` below as:
   one LLM call to decompose the answer into atomic claims, then **one LLM
   call per claim** to verdict it against the joined context (this
   project's own cost-shape choice — see "Cost" below for why per-claim,
   not batched).

2. **Answer Relevancy** — docs.ragas.io/en/stable/concepts/metrics/
   available_metrics/answer_relevance/ (fetched live this session) and the
   original paper: generate N (default 3) candidate questions an LLM
   believes the answer addresses, embed each candidate question and the
   original question, and average the cosine similarities:
   **AR = (1/N) * sum_i cosine_similarity(E(q_i), E(q))**. Embeddings come
   from this project's **local, free** embedder, never a paid embedding
   API — see "Embedder" below. One documented extension beyond the bare
   published formula: a noncommittal/evasive answer (e.g. "I don't know")
   is scored 0.0 directly, skipping the LLM call — see "Noncommittal
   answers" below; the RAGAS paper's prose says the metric "penalises
   cases where the answer is incomplete or ... redundant" but neither the
   paper nor the current docs.ragas.io page spell out a noncommittal-flag
   mechanism in the formula itself (verified by fetching both this
   session), so this piece is explicitly this project's own addition, not
   claimed as RAGAS's.

3. **Context Precision** — docs.ragas.io/en/stable/concepts/metrics/
   available_metrics/context_precision/ (fetched live this session):
   "Precision@k = TP@k / (TP@k + FP@k)"; **Context Precision@K = (sum_{k=1
   to K} Precision@k * v_k) / (total number of relevant items in top K)**,
   where v_k in {0, 1} is a per-rank relevance indicator. Judges each
   retrieved chunk's relevance to the `(question, ground_truth)` pair (RAGAS's
   `LLMContextPrecisionWithReference` shape — the `answer` field is *not*
   used here, unlike Faithfulness/Answer Relevancy) in **one batched LLM
   call** returning a same-length list of booleans, then applies the exact
   formula above in pure Python.

4. **Context Recall** — docs.ragas.io/en/stable/concepts/metrics/
   available_metrics/context_recall/ (fetched live this session):
   "Context Recall = Number of claims in the reference supported by the
   retrieved context / Total number of claims in the reference." Same
   decompose-then-verify shape as Faithfulness, but decomposing
   `ground_truth` (the reference) instead of `answer`, and checking
   attribution to context instead of factual support — **one LLM call per
   claim**, same cost-shape rationale as Faithfulness.

Distinct from `eval/retrieval_eval.py`'s IR metrics (`recall_at_k`/
`ndcg_at_k` in `eval/metrics.py`, against MedLoCoMo's gold `evidence`
turn/admission field): those ask *did retrieval find the annotated gold
evidence*; Context Precision/Recall here ask *would an LLM judge the
retrieved text useful for reconstructing the reference answer*. The two
can disagree (a chunk can contain the literal gold turn without being
judged "useful," or vice versa) — that disagreement is itself informative,
not a bug in either metric. `eval/retrieval_eval.py` does not exist in
this checkout as of this session (verified) — no import is attempted.

## Judge model — reused, not reinvented

`DEFAULT_JUDGE_MODEL = llm.JUDGE_MODEL` (Google, `gemini-3.5-flash-lite` by
default), used for **every** LLM call in this module (claim decomposition,
claim verdicts, candidate-question generation, context-relevance
judgments) — deliberately the *same* cross-family judge `judge.py` already
uses and argues for at length (a different provider family from
`llm.ANSWER_MODEL`; `literature/17` self-preference-bias citation), not a
second judge policy invented here. Every call is schema-constrained
(`llm.complete(schema=...)`) and `temperature=0.0` by default, matching
`judge.py`/`reader.py`'s own reproducibility convention.

## Embedder — the local, free path (verified this session, not assumed)

The story that dispatched this module named `graph/embedders.py` /
`qwen3-0.6b` / "running free on the RTX 3090" as an existing seam to import
from. At the *start* of this session that exact file did not exist
(verified by `find` over `src/medmemgraph/graph/`) — but the underlying
claim was otherwise real and independently verified, not taken on faith:
`Qwen/Qwen3-Embedding-0.6B` was present in this machine's local Hugging
Face cache (native `sentence-transformers` config —
`config_sentence_transformers.json` / `modules.json` / `1_Pooling/`, zero
network calls needed to load it) and `nvidia-smi` confirmed a real RTX
3090 (24576 MiB) in this environment.

`graph/embedders.py` then **landed concurrently, during this same
session** (a sibling dev-ml story) — re-checked directly before finalizing
this module, not left as a stale assumption. It already exposes exactly
the seam the story named, and a better-built one than this module would
have written standalone: `get_backend("qwen3-0.6b") -> EmbedderBackend`
(`Qwen/Qwen3-Embedding-0.6B`, `sentence-transformers`, fp16-on-CUDA/fp32-
on-CPU) with `.encode(texts, *, is_query: bool) -> np.ndarray`, its own
`(model, role, content_hash)` disk cache, and dimension read live off the
loaded model. Per this module's own "import and reuse if the interface
fits" rule (already stated above for `eval/guardrail.py`): **this module
does not duplicate that loader.** `_get_embedders_backend()` calls
`graph.embedders.get_backend(DEFAULT_BACKEND_NAME)` directly and caches
the returned instance; `_real_embed()`'s fallback is a two-tier, honestly-
labelled chain (`MetricResult.detail["embedder"]` always names which tier
actually ran — never silent):

1. **`graph.embedders.get_backend("qwen3-0.6b").encode(texts,
   is_query=False)`** — the real "free on the RTX 3090" path. `is_query=
   False` on *both* the original question and every generated candidate
   question (a deliberate, stated choice): the registered `qwen3-0.6b`
   query instruction is "Given a web search query, retrieve relevant
   passages that answer the query" — a retrieval framing that fits
   neither side of Answer Relevancy's question-vs-question comparison.
   Using it asymmetrically (one side only) would inject a mismatched
   instruction and break the comparison's symmetry; `is_query=False` (an
   explicit *empty* prompt for this specific model, per `graph/
   embedders.py`'s own docstring) treats both sides identically, which is
   the property that actually matters for a fair cosine comparison.
2. **`graph.vector_index.embed_texts`** (the already-shipped, offline,
   deterministic hashing-trick embedding) as the final safety net, only
   reached if `graph.embedders` is not importable or its backend fails to
   load in whatever environment this runs in later.

`dry_run=True` never reaches any of the above — it uses
`llm.embed(texts, dry_run=True)` (an existing, already-tested, zero-
network, deterministic stub), which is what keeps this module's own test
suite fast and offline without needing a GPU or cached model weights to be
present in CI.

## Guardrail overlap — left independent, not merged (guardrail.py does not exist yet)

`eval/guardrail.py`, per the dispatching story, is being built concurrently
and its grounding check overlaps Faithfulness by design. As of this
session `eval/guardrail.py` does not exist in this checkout (verified) —
there is nothing to import or reuse yet, so nothing is imported here. Per
the story's own instruction ("import and reuse if the interface fits,
otherwise leave both and note... independent"): once `guardrail.py` lands,
whoever wires the two together should read this note first — the two are
meant to be **independent estimates of the same property** (a useful
cross-check), not merged into one function.

## Integration with `eval/harness.py` — `build_sample`, not a `harness.py` edit

The story asks this module to "consume what the harness already produced"
via `RetrieveResult` (`items[].text` are the contexts) rather than
re-running retrieval. Verified this session by reading `eval/harness.py`,
`eval/reader.py`, and `eval/types.py`: `harness.evaluate()`'s per-item
`Record` and `AnswerResult`/`Answer` currently persist neither the raw
system answer text nor the retrieved `RetrieveResult` — only
correct/tokens/latency/truncated. So there is nothing in `HarnessRun`
today to literally consume. `build_sample()` / `build_samples()` below are
the primitives a follow-on integration story would call at the point it
*does* have a `RetrieveResult` and an answer in hand (e.g. inside
`ReaderAnswerer.answer()`, right after `pack = self.retriever(...)` and
`result = read(...)`) — they take a `contracts.RetrieveResult` directly, per
the story's own contract citation, and turn it into a `RagasSample`
without re-running anything. Wiring that call site into `harness.py` /
`eval/report.py` is out of this story's file scope (`ragas_metrics.py` +
its own test file only) and is called out explicitly here rather than
silently worked around.

## Cost — bounded by construction, not by discipline

Every real call here already goes through `llm.complete()`, so
`MEDMEMGRAPH_MAX_USD` (`llm.BudgetExceeded`) and the disk cache apply with
zero extra code. Call-count shape, so a caller can project cost before
running: Faithfulness and Context Recall cost **1 (decomposition) + 1 per
extracted claim** (skipped entirely — 0 calls — when the answer/reference
is empty/abstention, or when the context is empty, since nothing can be
supported/attributed by construction in that case); Answer Relevancy costs
**1** (0 if the answer is noncommittal, per "Noncommittal answers" above);
Context Precision costs **1** (batched across every retrieved chunk, 0 if
there is no context or no reference to judge against). `evaluate(...,
sample_size=N)` caps the sample set *before* any call is made, and every
`MetricResult`/`SampleReport`/`RagasReport` carries its own measured
`cost_usd` and `n_judge_calls` — never a token-count estimate standing in
for a real number. `project_full_run_cost()` is a bare linear scale-up
from a real measured per-sample cost, deliberately not a token-based
model (a `dry_run` cost is always exactly $0.0 and must never be
projected from).

## Honesty — a metric that cannot fail is not evidence

The stated project goal is scores near 1; this module is built so that is
*earned*: `literature/02` (R-MEM-054) reports an independent, non-peer-
reviewed audit found a `gpt-4o-mini` LLM judge accepting **62.8%** of
deliberately wrong-but-topically-related answers on an adversarial test —
a high score from an LLM judge is evidence the judge is lenient at least
as often as it is evidence the answer is good. This module's response,
concretely:

- **A deliberately-wrong control belongs in every run** — this module does
  not special-case a "control" sample; it is just another `RagasSample`
  with a fabricated-wrong answer, scored and reported (`SampleReport`)
  alongside the real ones, so a low score on it (or a suspiciously high
  one) is directly visible, not averaged away. `tests/test_ragas_metrics.py`
  includes exactly this case and asserts it scores materially lower than a
  fully-faithful control — if it does not, the test fails loudly, by
  design (a metric that cannot fail is not evidence).
- **Variance across repeated judge runs** — `evaluate_with_variance()`
  runs the same sample set `n_runs` times (default 3, matching
  `report.py`'s own `n_judge_runs < 3` "needs a repeated-run note"
  convention) at a caller-supplied `temperature > 0` and reports mean/SD
  per metric, **forcing `use_cache=False`** — with `temperature` held
  constant across repeats (it must be, to isolate sampling variance, not a
  temperature change) and `llm.py`'s cache key being exactly `(model,
  system, prompt, schema, temperature)`, a cached repeated run would
  return the *same* response every time and report a fabricated zero
  variance. This is a real correctness trap, not a style choice, so it is
  not caller-overridable.
- **Wilson intervals, not bare point estimates**, for the two metrics that
  are literal pooled claim-level proportions (Faithfulness, Context
  Recall — `RagasReport.wilson_ci`, via `eval.metrics.wilson_interval`,
  imported optimistically exactly like `eval/report.py` already does for
  `eval/metrics.py`, so this module still degrades gracefully if that file
  is ever absent from a checkout). Context Precision is a *weighted*
  formula, not a bare proportion, so it is deliberately **not** Wilson-
  pooled (would misrepresent the metric's own weighting) — `pooled_k`/
  `pooled_n`/`wilson_ci` are `None` for it, not a silently-wrong number.
  Answer Relevancy is a continuous cosine-similarity mean, not a
  proportion at all, so Wilson does not apply; `RagasReport.bootstrap_ci`
  uses `eval.metrics.bootstrap_ci` (percentile bootstrap) instead, the
  statistically correct tool for a mean with no closed-form CI.
- **Never tune a prompt to raise a score without checking the control
  moves the other way** — restated here as the module's own maintenance
  rule, not just a one-time instruction: any future change to the four
  system prompts below must re-run `tests/test_ragas_metrics.py`'s control
  case and confirm the gap (real vs. control) has not shrunk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np

from medmemgraph import llm
from medmemgraph.contracts import RetrieveResult
from medmemgraph.eval.judge import _looks_like_abstention

try:
    from medmemgraph.eval import metrics as _metrics
except ImportError:  # eval/metrics.py absent from this checkout -- degrade
    _metrics = None  # every metrics-dependent number below becomes an
    # explicit None rather than a second implementation of
    # wilson_interval/bootstrap_ci (same optional-import convention
    # eval/report.py already established for this exact module).

__all__ = [
    "DEFAULT_JUDGE_MODEL",
    "DEFAULT_N_QUESTIONS",
    "DEFAULT_METRICS",
    "RagasSample",
    "build_sample",
    "build_samples",
    "MetricResult",
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "SampleReport",
    "RagasReport",
    "evaluate",
    "RepeatedMetricStat",
    "VarianceReport",
    "evaluate_with_variance",
    "project_full_run_cost",
    "render_report",
    "metrics_available",
]

DEFAULT_JUDGE_MODEL = llm.JUDGE_MODEL
"""Google, gemini-3.5-flash-lite by default (override via
`MEDMEMGRAPH_JUDGE_MODEL`, read once by `llm.py` at import time). Reused
from `judge.py`'s identical constant/rationale, not a second judge policy
— see module docstring's "Judge model" section."""

DEFAULT_N_QUESTIONS = 3
"""RAGAS's own stated default ("a set of artificial questions (default is
3)", docs.ragas.io/.../answer_relevance/, fetched live this session)."""


def metrics_available(*names: str) -> bool:
    """Whether `medmemgraph.eval.metrics` is importable and exposes every
    name in `names` — local copy of `eval/report.py`'s identical helper
    (not cross-imported from `report.py`, to avoid `ragas_metrics.py`
    depending on the module that is meant to *consume* it — see module
    docstring's INTEGRATION note)."""
    return _metrics is not None and all(hasattr(_metrics, n) for n in names)


# ---------------------------------------------------------------------------
# RagasSample — the (question, answer, contexts, ground_truth) unit every
# metric below scores.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RagasSample:
    question: str
    answer: str
    contexts: tuple[str, ...]
    ground_truth: str
    sample_id: str = ""


def build_sample(
    question: str,
    answer: str,
    retrieve_result: RetrieveResult,
    ground_truth: str,
    *,
    sample_id: str = "",
) -> RagasSample:
    """Build one `RagasSample` from a `contracts.RetrieveResult` directly —
    `items[].text` are the contexts, per the story's own contract citation.
    Never re-runs retrieval; `retrieve_result` is assumed already produced
    by the caller (see module docstring's INTEGRATION note)."""
    contexts = tuple(item.text for item in retrieve_result.items)
    return RagasSample(question=question, answer=answer, contexts=contexts, ground_truth=ground_truth, sample_id=sample_id)


def build_samples(
    qa_items: Sequence[Mapping],
    answers: Sequence[str],
    retrieve_results: Sequence[RetrieveResult],
    *,
    ground_truth_key: str = "answer",
) -> list[RagasSample]:
    """`build_sample` over three already-produced, index-paired sequences
    (one QA item dict, one system answer, one `RetrieveResult`, per index)
    — the shape a harness-integration call site would have in hand after
    running retrieval + answering once, per QA item. Raises rather than
    silently zipping mismatched lengths."""
    n = len(qa_items)
    if not (len(answers) == n and len(retrieve_results) == n):
        raise ValueError(
            "qa_items, answers, and retrieve_results must be the same length "
            f"(paired per QA item); got {n}, {len(answers)}, {len(retrieve_results)}"
        )
    return [
        build_sample(
            item["question"], answer, rr, item[ground_truth_key], sample_id=str(item.get("qa_id", ""))
        )
        for item, answer, rr in zip(qa_items, answers, retrieve_results)
    ]


# ---------------------------------------------------------------------------
# Shared LLM-call plumbing
# ---------------------------------------------------------------------------

_CLAIMS_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["claims"],
    "additionalProperties": False,
}

_CLAIM_VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "attributable": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["attributable", "reason"],
    "additionalProperties": False,
}

_QUESTIONS_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["questions"],
    "additionalProperties": False,
}

_CONTEXT_RELEVANCE_SCHEMA = {
    "type": "object",
    "properties": {
        "relevant": {"type": "array", "items": {"type": "boolean"}},
    },
    "required": ["relevant"],
    "additionalProperties": False,
}

_CLAIMS_MAX_TOKENS = 800
_VERDICT_MAX_TOKENS = 300
_QUESTIONS_MAX_TOKENS = 400
_CONTEXT_PRECISION_MAX_TOKENS = 800


def _call_llm(
    system: str,
    user: str,
    schema: dict,
    *,
    model: str,
    dry_run: bool,
    use_cache: bool,
    temperature: float,
    max_tokens: int,
) -> tuple[dict, "llm.LLMResponse"]:
    """Route through the seam — every real call in this module goes through
    exactly this function, so cost/cache/budget accounting is uniform (see
    `judge.py::Judge._call_llm` for the identical convention)."""
    response = llm.complete(
        user,
        model=model,
        schema=schema,
        system=system,
        max_tokens=max_tokens,
        temperature=temperature,
        dry_run=dry_run,
        use_cache=use_cache,
    )
    return response.parsed, response


# ---------------------------------------------------------------------------
# MetricResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricResult:
    """One metric's score for one sample.

    `k`/`n` are populated only when the metric is a literal claim-level
    proportion (Faithfulness, Context Recall: k=supported/attributed
    claims, n=total claims) or a count pair meaningful for pooling
    (Context Precision: k=n_relevant, n=n_contexts, kept for transparency
    in `detail` but never Wilson-pooled — see module docstring). `None`/
    `None` for Answer Relevancy (a continuous mean, not a proportion).
    `score=None` means undefined (e.g. 0 claims extracted, empty answer,
    empty context) — never silently 0.0, matching `eval/metrics.py`'s own
    `AbstentionPRF1` convention for undefined proportions."""

    name: str
    score: float | None
    k: int | None
    n: int | None
    cost_usd: float
    n_judge_calls: int
    detail: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 1. Faithfulness
# ---------------------------------------------------------------------------

_FAITHFULNESS_CLAIM_SYSTEM = (
    "You decompose a clinical system's answer into a list of standalone, "
    "atomic factual claims, per RAGAS's Faithfulness metric (docs.ragas.io "
    "/en/stable/concepts/metrics/available_metrics/faithfulness/): 'decompose "
    "longer sentences into shorter and more focused assertions.' Apply these "
    "rules, in order:\n"
    "1. Each claim must be a single, independently verifiable statement of "
    "fact (one medication, one dose, one diagnosis, one timing fact, etc. "
    "per claim -- never combine two facts into one claim).\n"
    "2. Each claim must be understandable on its own, without needing the "
    "other claims or the original question -- resolve pronouns and vague "
    "references using the question's own wording.\n"
    "3. Do not add any fact that is not stated or directly implied by the "
    "answer. Do not omit a stated fact.\n"
    "4. If the answer makes no factual claims at all (e.g. it only declines "
    "to answer), return an empty list.\n"
    "Respond with the required JSON only: `claims`, a list of strings."
)

_FAITHFULNESS_VERDICT_SYSTEM = (
    "You verify whether ONE claim can be inferred from a set of retrieved "
    "context passages, per RAGAS's Faithfulness metric (same source as "
    "above): a claim is 'supported' only if the context passages, read "
    "together, directly state or straightforwardly entail the claim -- "
    "never because the claim merely sounds medically plausible, and never "
    "using outside/general medical knowledge. Respond with the required "
    "JSON only: `attributable` (boolean, true iff supported by the given "
    "context) and a one-sentence `reason`."
)


def _claims_user(question: str, text: str) -> str:
    return f"Question: {question}\n\nText to decompose into claims:\n{text}"


def _claim_verdict_user(claim: str, context_text: str) -> str:
    return f"Claim: {claim}\n\nRetrieved context:\n{context_text}"


def _join_contexts(contexts: Sequence[str]) -> str:
    return "\n\n".join(f"[Context {i + 1}] {c}" for i, c in enumerate(contexts) if str(c).strip())


def _decompose_claims(
    question: str,
    text: str,
    system_prompt: str,
    *,
    model: str,
    dry_run: bool,
    use_cache: bool,
    temperature: float,
) -> tuple[list[str], float, int]:
    """Shared decomposition step for Faithfulness (`text=answer`) and
    Context Recall (`text=ground_truth`). Returns `(claims, cost_usd,
    n_calls)`; `n_calls` is always 0 or 1 (this function makes at most one
    LLM call). Fast-paths (0 calls) on an empty/abstention `text` -- see
    module docstring's Cost section."""
    if not text.strip() or _looks_like_abstention(text):
        return [], 0.0, 0
    data, response = _call_llm(
        system_prompt,
        _claims_user(question, text),
        _CLAIMS_SCHEMA,
        model=model,
        dry_run=dry_run,
        use_cache=use_cache,
        temperature=temperature,
        max_tokens=_CLAIMS_MAX_TOKENS,
    )
    claims = [str(c).strip() for c in (data.get("claims") or []) if str(c).strip()]
    return claims, response.cost_usd, 1


def _verify_claims(
    claims: list[str],
    context_text: str,
    system_prompt: str,
    *,
    model: str,
    dry_run: bool,
    use_cache: bool,
    temperature: float,
) -> tuple[int, list[dict], float, int]:
    """One LLM call per claim (module docstring's Cost section). Returns
    `(n_supported, verdicts, cost_usd, n_calls)`."""
    supported = 0
    verdicts: list[dict] = []
    cost = 0.0
    calls = 0
    for claim in claims:
        data, response = _call_llm(
            system_prompt,
            _claim_verdict_user(claim, context_text),
            _CLAIM_VERDICT_SCHEMA,
            model=model,
            dry_run=dry_run,
            use_cache=use_cache,
            temperature=temperature,
            max_tokens=_VERDICT_MAX_TOKENS,
        )
        calls += 1
        cost += response.cost_usd
        ok = bool(data.get("attributable"))
        supported += int(ok)
        verdicts.append({"claim": claim, "supported": ok, "reason": str(data.get("reason", ""))})
    return supported, verdicts, cost, calls


def faithfulness(
    sample: RagasSample,
    *,
    model: str = DEFAULT_JUDGE_MODEL,
    dry_run: bool = False,
    use_cache: bool = True,
    temperature: float = 0.0,
    **_ignored,
) -> MetricResult:
    """`F = |V| / |S|` — fraction of the answer's atomic claims inferable
    from the retrieved context (docs.ragas.io Faithfulness; see module
    docstring)."""
    claims, decomp_cost, decomp_calls = _decompose_claims(
        sample.question, sample.answer, _FAITHFULNESS_CLAIM_SYSTEM,
        model=model, dry_run=dry_run, use_cache=use_cache, temperature=temperature,
    )
    if not claims:
        reason = (
            "answer is empty or looks like an abstention; no factual claims to check"
            if not sample.answer.strip() or _looks_like_abstention(sample.answer)
            else "0 claims extracted from the answer"
        )
        return MetricResult(
            name="faithfulness", score=None, k=None, n=0, cost_usd=decomp_cost, n_judge_calls=decomp_calls,
            detail={"reason": reason},
        )

    context_text = _join_contexts(sample.contexts)
    if not context_text:
        return MetricResult(
            name="faithfulness", score=0.0, k=0, n=len(claims), cost_usd=decomp_cost, n_judge_calls=decomp_calls,
            detail={"claims": claims, "reason": "no retrieved context; every claim is unsupported by construction"},
        )

    supported, verdicts, verify_cost, verify_calls = _verify_claims(
        claims, context_text, _FAITHFULNESS_VERDICT_SYSTEM,
        model=model, dry_run=dry_run, use_cache=use_cache, temperature=temperature,
    )
    return MetricResult(
        name="faithfulness",
        score=supported / len(claims),
        k=supported,
        n=len(claims),
        cost_usd=decomp_cost + verify_cost,
        n_judge_calls=decomp_calls + verify_calls,
        detail={"claims": claims, "verdicts": verdicts},
    )


# ---------------------------------------------------------------------------
# 2. Context Recall
# ---------------------------------------------------------------------------

_CONTEXT_RECALL_CLAIM_SYSTEM = (
    "You decompose a GROUND-TRUTH REFERENCE ANSWER into a list of "
    "standalone, atomic factual claims, per RAGAS's Context Recall metric "
    "(docs.ragas.io/en/stable/concepts/metrics/available_metrics/"
    "context_recall/). Apply these rules, in order:\n"
    "1. Each claim must be a single, independently verifiable statement of "
    "fact.\n"
    "2. Each claim must be understandable on its own -- resolve pronouns "
    "and vague references using the question's own wording.\n"
    "3. Do not add any fact not stated or directly implied by the "
    "reference answer.\n"
    "4. If the reference asserts no factual content (e.g. it is itself "
    "'the question is not answerable'), return an empty list.\n"
    "Respond with the required JSON only: `claims`, a list of strings."
)

_CONTEXT_RECALL_VERDICT_SYSTEM = (
    "You verify whether ONE claim from a ground-truth reference answer can "
    "be attributed to a set of retrieved context passages, per RAGAS's "
    "Context Recall metric (same source as above): a claim is "
    "'attributable' only if the context passages, read together, directly "
    "state or straightforwardly support the claim -- never because it "
    "merely sounds plausible, and never using outside/general medical "
    "knowledge. Respond with the required JSON only: `attributable` "
    "(boolean) and a one-sentence `reason`."
)


def context_recall(
    sample: RagasSample,
    *,
    model: str = DEFAULT_JUDGE_MODEL,
    dry_run: bool = False,
    use_cache: bool = True,
    temperature: float = 0.0,
    **_ignored,
) -> MetricResult:
    """Fraction of the ground-truth reference's atomic claims attributable
    to the retrieved context (docs.ragas.io Context Recall; see module
    docstring). Note: uses `sample.ground_truth`, never `sample.answer`."""
    claims, decomp_cost, decomp_calls = _decompose_claims(
        sample.question, sample.ground_truth, _CONTEXT_RECALL_CLAIM_SYSTEM,
        model=model, dry_run=dry_run, use_cache=use_cache, temperature=temperature,
    )
    if not claims:
        return MetricResult(
            name="context_recall", score=None, k=None, n=0, cost_usd=decomp_cost, n_judge_calls=decomp_calls,
            detail={"reason": "no ground_truth reference, or 0 claims extracted from it"},
        )

    context_text = _join_contexts(sample.contexts)
    if not context_text:
        return MetricResult(
            name="context_recall", score=0.0, k=0, n=len(claims), cost_usd=decomp_cost, n_judge_calls=decomp_calls,
            detail={"claims": claims, "reason": "no retrieved context; every claim is unattributable by construction"},
        )

    supported, verdicts, verify_cost, verify_calls = _verify_claims(
        claims, context_text, _CONTEXT_RECALL_VERDICT_SYSTEM,
        model=model, dry_run=dry_run, use_cache=use_cache, temperature=temperature,
    )
    return MetricResult(
        name="context_recall",
        score=supported / len(claims),
        k=supported,
        n=len(claims),
        cost_usd=decomp_cost + verify_cost,
        n_judge_calls=decomp_calls + verify_calls,
        detail={"claims": claims, "verdicts": verdicts},
    )


# ---------------------------------------------------------------------------
# 3. Answer Relevancy — local embedder
# ---------------------------------------------------------------------------

_ANSWER_RELEVANCY_SYSTEM = (
    "You reverse-engineer questions from an answer, per RAGAS's Answer "
    "Relevancy metric (docs.ragas.io/en/stable/concepts/metrics/"
    "available_metrics/answer_relevance/): generate N distinct questions "
    "that this answer, taken alone, most directly and completely addresses "
    "-- as if you had to guess what the user originally asked, given only "
    "this answer. Each question must be answerable from the given answer "
    "text alone. Vary the phrasing across the N questions; do not just "
    "restate one question N times. Respond with the required JSON only: "
    "`questions`, a list of exactly N strings."
)


def _answer_relevancy_user(answer: str, n_questions: int) -> str:
    return f"N = {n_questions}\n\nAnswer:\n{answer}"


_embedders_backend_singleton = None  # type: ignore[var-annotated]
"""Cached `graph.embedders.EmbedderBackend` instance (module docstring's
"Embedder" section) — resolved once per process, reused across calls, same
lazy-singleton idiom as `llm.py`'s own provider clients."""


def _get_embedders_backend():
    """Resolve and cache `graph.embedders.get_backend(DEFAULT_BACKEND_NAME)`
    — `graph/embedders.py` landed in this checkout during this session
    (verified directly, not assumed): it already exposes exactly the
    seam the dispatching story named (`qwen3-0.6b` ->
    `Qwen/Qwen3-Embedding-0.6B`, local, cached, disk-cache-backed,
    dimension read live off the model), with its own registry, its own
    `(model, role, content_hash)` disk cache, and its own fp16/CUDA
    handling — reused here rather than re-implemented, per the "import and
    reuse if the interface fits" instruction this module's docstring
    already applies to `eval/guardrail.py`."""
    global _embedders_backend_singleton
    if _embedders_backend_singleton is None:
        from medmemgraph.graph.embedders import DEFAULT_BACKEND_NAME, get_backend

        _embedders_backend_singleton = get_backend(DEFAULT_BACKEND_NAME)
    return _embedders_backend_singleton


def reset_embedders_backend() -> None:
    """Test/CLI convenience: drop the cached backend singleton (mirrors
    `llm.reset_clients()`)."""
    global _embedders_backend_singleton
    _embedders_backend_singleton = None


def _real_embed(texts: list[str]) -> tuple[np.ndarray, str]:
    """The real (non-`dry_run`), non-paid embedding path. Returns
    `(vectors, embedder_label)`; the label always names which tier
    actually ran, never silently.

    `is_query=False` on **both** the original question and the generated
    candidate questions — a deliberate, stated choice, not an oversight:
    `graph/embedders.py`'s registered `qwen3-0.6b` query instruction is
    "Given a web search query, retrieve relevant passages that answer the
    query" (a retrieval framing), and Answer Relevancy compares two
    *questions* to each other, which is neither a "web search query" nor a
    "passage" — using `is_query=True` on one side only would inject a
    retrieval-flavored instruction that fits neither string's actual role
    and would break the comparison's symmetry. `is_query=False` (the
    model's plain, unprefixed "document" encoding — an explicit empty
    prompt for `qwen3-0.6b` specifically, per that module's own docstring)
    keeps both sides identically treated, which is the property that
    actually matters for a fair cosine comparison."""
    try:
        backend = _get_embedders_backend()
    except ImportError as exc:
        from medmemgraph.graph.vector_index import embed_texts

        return (
            np.asarray(embed_texts(texts), dtype=np.float32),
            f"hashing-fallback (ImportError: graph.embedders unavailable: {exc})",
        )
    try:
        vectors = backend.encode(texts, is_query=False)
        return np.asarray(vectors, dtype=np.float32), f"graph.embedders:{backend.name}"
    except Exception as exc:  # noqa: BLE001 -- torch/sentence_transformers missing, no CUDA, model load failure, ...
        from medmemgraph.graph.vector_index import embed_texts

        return (
            np.asarray(embed_texts(texts), dtype=np.float32),
            f"hashing-fallback ({exc.__class__.__name__}: graph.embedders backend failed)",
        )


def _embed_for_relevancy(
    texts: list[str], *, embed_fn: Callable[[list[str]], np.ndarray] | None, dry_run: bool
) -> tuple[np.ndarray, str]:
    if embed_fn is not None:
        return np.asarray(embed_fn(texts), dtype=np.float32), "custom"
    if dry_run:
        return np.asarray(llm.embed(texts, dry_run=True), dtype=np.float32), "dry-run-stub"
    return _real_embed(texts)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity, clipped to `[-1.0, 1.0]`. Mathematically cosine
    similarity is always in that range; float32 rounding on near-identical
    vectors can otherwise drift a hair past 1.0 (observed directly in this
    module's own test suite — not a hypothetical), which would silently
    push `answer_relevancy`'s mean outside its own documented `[0, 1]`-ish
    range for a reason that has nothing to do with the metric itself."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    cosine = float(np.dot(a, b) / (na * nb))
    return max(-1.0, min(1.0, cosine))


def answer_relevancy(
    sample: RagasSample,
    *,
    model: str = DEFAULT_JUDGE_MODEL,
    dry_run: bool = False,
    use_cache: bool = True,
    temperature: float = 0.0,
    n_questions: int = DEFAULT_N_QUESTIONS,
    embed_fn: Callable[[list[str]], np.ndarray] | None = None,
    **_ignored,
) -> MetricResult:
    """`AR = (1/N) * sum_i cosine_similarity(E(q_i), E(q))` (docs.ragas.io /
    original paper's Answer Relevancy; see module docstring). Scores 0.0,
    with 0 LLM calls, when the answer looks like a refusal/abstention —
    this project's own extension beyond the bare published formula (module
    docstring's "Noncommittal answers")."""
    answer = sample.answer or ""
    if not answer.strip() or _looks_like_abstention(answer):
        return MetricResult(
            name="answer_relevancy", score=0.0, k=None, n=None, cost_usd=0.0, n_judge_calls=0,
            detail={
                "noncommittal": True,
                "reason": (
                    "answer looks like a refusal/abstention (judge._looks_like_abstention); "
                    "scored 0.0 rather than letting generic reverse-engineered questions "
                    "inflate relevancy -- this module's own extension, not claimed as part "
                    "of the published RAGAS formula (see module docstring)"
                ),
            },
        )

    data, response = _call_llm(
        _ANSWER_RELEVANCY_SYSTEM, _answer_relevancy_user(answer, n_questions), _QUESTIONS_SCHEMA,
        model=model, dry_run=dry_run, use_cache=use_cache, temperature=temperature,
        max_tokens=_QUESTIONS_MAX_TOKENS,
    )
    questions = [str(q).strip() for q in (data.get("questions") or []) if str(q).strip()][:n_questions]
    if not questions:
        return MetricResult(
            name="answer_relevancy", score=None, k=None, n=0, cost_usd=response.cost_usd, n_judge_calls=1,
            detail={"reason": "0 candidate questions generated"},
        )

    vectors, embedder_label = _embed_for_relevancy([sample.question] + questions, embed_fn=embed_fn, dry_run=dry_run)
    q_vec = vectors[0]
    gen_vecs = vectors[1:]
    similarities = [_cosine(q_vec, v) for v in gen_vecs]
    score = float(sum(similarities) / len(similarities))
    return MetricResult(
        name="answer_relevancy",
        score=score,
        k=None,
        n=len(similarities),
        cost_usd=response.cost_usd,
        n_judge_calls=1,
        detail={"generated_questions": questions, "similarities": similarities, "embedder": embedder_label},
    )


# ---------------------------------------------------------------------------
# 4. Context Precision
# ---------------------------------------------------------------------------

_CONTEXT_PRECISION_SYSTEM = (
    "You judge, for EACH retrieved context passage, whether it would be "
    "useful for reconstructing the GROUND-TRUTH REFERENCE ANSWER to the "
    "QUESTION -- per RAGAS's Context Precision metric (docs.ragas.io/"
    "en/stable/concepts/metrics/available_metrics/context_precision/), "
    "judging relevance against the reference answer, never against the "
    "system's own answer (which is not shown to you). A passage is "
    "relevant if it states or directly supports a fact that appears in "
    "the reference answer; a passage about the same general topic that "
    "does not actually contain a fact used in the reference answer is NOT "
    "relevant. Judge every passage independently, in the order given. "
    "Respond with the required JSON only: `relevant`, a list of booleans, "
    "one per passage, same order and same length as the passages given."
)


def _context_precision_user(question: str, ground_truth: str, contexts: Sequence[str]) -> str:
    passages = "\n\n".join(f"[Passage {i + 1}] {c}" for i, c in enumerate(contexts))
    return f"Question: {question}\n\nReference answer: {ground_truth}\n\nPassages:\n{passages}"


def _context_precision_score(relevant: list[bool]) -> tuple[float | None, int, int]:
    """Context Precision@K = sum_{k=1..K}(Precision@k * v_k) / (total
    relevant items in top K) — the exact formula quoted from docs.ragas.io
    (module docstring). `K == 0` -> `(None, 0, 0)` (no contexts at all,
    genuinely undefined). `K > 0` but no relevant items -> `(0.0,
    0, K)` (RAGAS's own documented convention for the degenerate 0/0 case,
    avoiding NaN)."""
    k_total = len(relevant)
    if k_total == 0:
        return None, 0, 0
    total_relevant = sum(1 for r in relevant if r)
    if total_relevant == 0:
        return 0.0, 0, k_total
    numerator = 0.0
    true_positives = 0
    for rank, is_relevant in enumerate(relevant, start=1):
        if is_relevant:
            true_positives += 1
        precision_at_rank = true_positives / rank
        numerator += precision_at_rank * (1.0 if is_relevant else 0.0)
    return numerator / total_relevant, total_relevant, k_total


def context_precision(
    sample: RagasSample,
    *,
    model: str = DEFAULT_JUDGE_MODEL,
    dry_run: bool = False,
    use_cache: bool = True,
    temperature: float = 0.0,
    **_ignored,
) -> MetricResult:
    """Position-weighted precision of the retrieved chunks against the
    `(question, ground_truth)` reference pair (docs.ragas.io Context
    Precision; see module docstring). Note: uses `sample.ground_truth`,
    never `sample.answer` — `LLMContextPrecisionWithReference`'s own shape."""
    contexts = list(sample.contexts)
    k_total = len(contexts)
    if k_total == 0:
        return MetricResult(
            name="context_precision", score=None, k=None, n=0, cost_usd=0.0, n_judge_calls=0,
            detail={"reason": "no retrieved context to score"},
        )
    if not (sample.ground_truth or "").strip():
        return MetricResult(
            name="context_precision", score=None, k=None, n=k_total, cost_usd=0.0, n_judge_calls=0,
            detail={"reason": "no ground_truth reference to judge relevance against"},
        )

    data, response = _call_llm(
        _CONTEXT_PRECISION_SYSTEM, _context_precision_user(sample.question, sample.ground_truth, contexts),
        _CONTEXT_RELEVANCE_SCHEMA, model=model, dry_run=dry_run, use_cache=use_cache, temperature=temperature,
        max_tokens=_CONTEXT_PRECISION_MAX_TOKENS,
    )
    raw = [bool(x) for x in (data.get("relevant") or [])]
    # Conservative pad/truncate against a malformed-length response: pad
    # missing entries with False (assume not relevant -- same "an
    # overestimate/underestimate must fail safe" direction llm.py's own
    # budget check uses), truncate extras. Never silently trust a
    # mismatched-length array positionally beyond what was actually asked.
    relevant = (raw + [False] * k_total)[:k_total]
    score, n_relevant, k = _context_precision_score(relevant)
    return MetricResult(
        name="context_precision", score=score, k=n_relevant, n=k, cost_usd=response.cost_usd, n_judge_calls=1,
        detail={"relevant": relevant, "length_mismatch": len(raw) != k_total},
    )


# ---------------------------------------------------------------------------
# evaluate() — the sample-set-level entry point
# ---------------------------------------------------------------------------

DEFAULT_METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")

_METRIC_FUNCS: dict[str, Callable[..., MetricResult]] = {
    "faithfulness": faithfulness,
    "answer_relevancy": answer_relevancy,
    "context_precision": context_precision,
    "context_recall": context_recall,
}


@dataclass(frozen=True)
class SampleReport:
    sample_id: str
    question: str
    scores: dict[str, float | None]
    cost_usd: float
    n_judge_calls: int
    details: dict[str, dict] = field(default_factory=dict)


@dataclass(frozen=True)
class RagasReport:
    samples: tuple[SampleReport, ...]
    metric_names: tuple[str, ...]
    mean_scores: dict[str, float | None]
    n_scored: dict[str, int]
    pooled_k: dict[str, int | None]
    pooled_n: dict[str, int | None]
    wilson_ci: dict[str, tuple[float, float] | None]
    bootstrap_ci: dict[str, tuple[float, float] | None]
    n_requested: int
    n_samples: int
    sample_size: int | None
    total_cost_usd: float
    cost_per_sample_usd: float
    n_judge_calls: int
    dry_run: bool
    judge_model: str
    judge_temperature: float


def evaluate(
    samples: Sequence[RagasSample],
    *,
    metrics: Sequence[str] = DEFAULT_METRICS,
    sample_size: int | None = None,
    model: str = DEFAULT_JUDGE_MODEL,
    dry_run: bool = False,
    use_cache: bool = True,
    temperature: float = 0.0,
    embed_fn: Callable[[list[str]], np.ndarray] | None = None,
) -> RagasReport:
    """Score `samples` (capped to the first `sample_size`, before any call
    is made — module docstring's Cost section) on every metric named in
    `metrics`. Every real call routes through `medmemgraph.llm.complete()`,
    so the disk cache and `MEDMEMGRAPH_MAX_USD` budget cap apply
    automatically; `dry_run=True` makes zero network calls, in the LLM
    calls and in the embedding step alike (see module docstring's
    "Embedder" section)."""
    unknown = [m for m in metrics if m not in _METRIC_FUNCS]
    if unknown:
        raise ValueError(f"unknown ragas metric(s) {unknown!r}; choose from {sorted(_METRIC_FUNCS)}")

    all_samples = list(samples)
    used = all_samples if sample_size is None else all_samples[: max(0, sample_size)]

    sample_reports: list[SampleReport] = []
    for s in used:
        scores: dict[str, float | None] = {}
        details: dict[str, dict] = {}
        sample_cost = 0.0
        sample_calls = 0
        for metric_name in metrics:
            fn = _METRIC_FUNCS[metric_name]
            kwargs: dict = {
                "model": model, "dry_run": dry_run, "use_cache": use_cache, "temperature": temperature,
            }
            if metric_name == "answer_relevancy":
                kwargs["embed_fn"] = embed_fn
            result = fn(s, **kwargs)
            scores[metric_name] = result.score
            details[metric_name] = {
                "k": result.k, "n": result.n, "cost_usd": result.cost_usd,
                "n_judge_calls": result.n_judge_calls, **result.detail,
            }
            sample_cost += result.cost_usd
            sample_calls += result.n_judge_calls
        sample_reports.append(
            SampleReport(
                sample_id=s.sample_id or s.question[:40],
                question=s.question,
                scores=scores,
                cost_usd=sample_cost,
                n_judge_calls=sample_calls,
                details=details,
            )
        )

    mean_scores: dict[str, float | None] = {}
    n_scored: dict[str, int] = {}
    pooled_k: dict[str, int | None] = {}
    pooled_n: dict[str, int | None] = {}
    wilson_ci: dict[str, tuple[float, float] | None] = {}
    bootstrap_ci_out: dict[str, tuple[float, float] | None] = {}

    for metric_name in metrics:
        values = [sr.scores[metric_name] for sr in sample_reports if sr.scores[metric_name] is not None]
        n_scored[metric_name] = len(values)
        mean_scores[metric_name] = (sum(values) / len(values)) if values else None

        if metric_name in ("faithfulness", "context_recall"):
            k_total = sum(sr.details[metric_name]["k"] or 0 for sr in sample_reports if sr.details[metric_name]["n"])
            n_total = sum(sr.details[metric_name]["n"] or 0 for sr in sample_reports if sr.details[metric_name]["n"])
            if n_total and metrics_available("wilson_interval"):
                interval = _metrics.wilson_interval(k_total, n_total)
                pooled_k[metric_name] = k_total
                pooled_n[metric_name] = n_total
                wilson_ci[metric_name] = (interval.lo, interval.hi)
            else:
                pooled_k[metric_name] = k_total if n_total else None
                pooled_n[metric_name] = n_total if n_total else None
                wilson_ci[metric_name] = None
        else:
            pooled_k[metric_name] = None
            pooled_n[metric_name] = None
            wilson_ci[metric_name] = None

        if metric_name == "answer_relevancy" and len(values) >= 2 and metrics_available("bootstrap_ci"):
            boot = _metrics.bootstrap_ci(values)
            bootstrap_ci_out[metric_name] = (boot.lo, boot.hi)
        else:
            bootstrap_ci_out[metric_name] = None

    total_cost = sum(sr.cost_usd for sr in sample_reports)
    total_calls = sum(sr.n_judge_calls for sr in sample_reports)
    n = len(sample_reports)

    return RagasReport(
        samples=tuple(sample_reports),
        metric_names=tuple(metrics),
        mean_scores=mean_scores,
        n_scored=n_scored,
        pooled_k=pooled_k,
        pooled_n=pooled_n,
        wilson_ci=wilson_ci,
        bootstrap_ci=bootstrap_ci_out,
        n_requested=len(all_samples),
        n_samples=n,
        sample_size=sample_size,
        total_cost_usd=total_cost,
        cost_per_sample_usd=(total_cost / n) if n else 0.0,
        n_judge_calls=total_calls,
        dry_run=dry_run,
        judge_model=model,
        judge_temperature=temperature,
    )


# ---------------------------------------------------------------------------
# evaluate_with_variance() — repeated-run variance (HONESTY requirement)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RepeatedMetricStat:
    metric: str
    mean: float | None
    sd: float | None
    n_runs: int
    per_run_mean: tuple[float | None, ...]


@dataclass(frozen=True)
class VarianceReport:
    runs: tuple[RagasReport, ...]
    stats: dict[str, RepeatedMetricStat]
    n_runs: int
    temperature: float


def evaluate_with_variance(
    samples: Sequence[RagasSample],
    *,
    n_runs: int = 3,
    metrics: Sequence[str] = DEFAULT_METRICS,
    sample_size: int | None = None,
    model: str = DEFAULT_JUDGE_MODEL,
    dry_run: bool = False,
    temperature: float = 0.7,
    embed_fn: Callable[[list[str]], np.ndarray] | None = None,
) -> VarianceReport:
    """Run `evaluate()` `n_runs` times at `temperature` and report per-
    metric mean/SD across runs — module docstring's HONESTY section:
    "report variance across repeated judge runs." `use_cache` is **forced
    False** and not caller-overridable: `llm.py`'s cache key is exactly
    `(model, system, prompt, schema, temperature)`, so with `temperature`
    correctly held constant across repeats, a cached run would return the
    identical response every time and report a fabricated zero variance —
    a real correctness trap, not a style choice."""
    runs = tuple(
        evaluate(
            samples, metrics=metrics, sample_size=sample_size, model=model, dry_run=dry_run,
            use_cache=False, temperature=temperature, embed_fn=embed_fn,
        )
        for _ in range(n_runs)
    )
    stats: dict[str, RepeatedMetricStat] = {}
    for metric_name in metrics:
        per_run = tuple(r.mean_scores.get(metric_name) for r in runs)
        known = [v for v in per_run if v is not None]
        mean = (sum(known) / len(known)) if known else None
        sd = None
        if mean is not None and len(known) >= 2:
            sd = (sum((v - mean) ** 2 for v in known) / (len(known) - 1)) ** 0.5
        stats[metric_name] = RepeatedMetricStat(metric=metric_name, mean=mean, sd=sd, n_runs=len(runs), per_run_mean=per_run)
    return VarianceReport(runs=runs, stats=stats, n_runs=n_runs, temperature=temperature)


# ---------------------------------------------------------------------------
# Cost projection + rendering
# ---------------------------------------------------------------------------


def project_full_run_cost(cost_per_sample_usd: float, n_samples: int) -> float:
    """Linear projection from a real MEASURED per-sample cost — module
    docstring's Cost section: never a token-count model, and never fed a
    `dry_run` cost (always exactly $0.0, which would project a fabricated
    $0 for the real run)."""
    return max(0.0, cost_per_sample_usd) * max(0, n_samples)


def render_report(report: RagasReport) -> str:
    lines = [
        f"=== ragas_metrics report — n_samples={report.n_samples}/{report.n_requested} "
        f"(sample_size={report.sample_size}) judge={report.judge_model} dry_run={report.dry_run} ===",
    ]
    for metric_name in report.metric_names:
        mean = report.mean_scores.get(metric_name)
        mean_s = "n/a" if mean is None else f"{mean:.3f}"
        n_s = report.n_scored.get(metric_name, 0)
        ci = report.wilson_ci.get(metric_name) or report.bootstrap_ci.get(metric_name)
        ci_s = f" CI=[{ci[0]:.3f},{ci[1]:.3f}]" if ci else ""
        lines.append(f"  {metric_name:20s} mean={mean_s} (n_scored={n_s}){ci_s}")
    lines.append(
        f"  cost: total=${report.total_cost_usd:.4f} per_sample=${report.cost_per_sample_usd:.4f} "
        f"n_judge_calls={report.n_judge_calls}"
    )
    return "\n".join(lines)
