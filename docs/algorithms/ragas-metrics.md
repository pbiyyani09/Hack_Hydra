# RAGAS-defined answer-quality metrics — Faithfulness, Answer Relevancy, Context Precision, Context Recall

Owner: Evidence/eval. Code: `src/medmemgraph/eval/ragas_metrics.py`. Tests:
`tests/test_ragas_metrics.py`.

This document explains four pieces a reviewer would otherwise have to
reconstruct from the code: why this module reimplements RAGAS instead of
importing the `ragas` library, the exact algorithm each metric runs
(decompose → verify, not a black box), the embedder story (a genuine
mid-flight correction, kept visible rather than smoothed over), and the
honesty machinery that keeps a near-1.0 score from being a decoration.

## 1. Why reimplemented, not imported

`ragas` was installed and removed. `ragas` 0.3.x imports
`langchain_community.chat_models.vertexai.ChatVertexAI`, and this
project's pinned `langchain-community` 0.4.2 no longer ships that module
(the package's own deprecation notice says it "is being sunset"); `ragas
>=0.4` failed to resolve on this project's pinned Python 3.13. So: **these
are RAGAS-defined metrics computed by our own implementation**, not the
RAGAS library's own numbers — not bit-comparable to a published RAGAS
leaderboard.

This turned out better for the project on four independent axes: it runs
on `llm.JUDGE_MODEL` (cheap, cross-family, already cost-capped) through the
existing `medmemgraph.llm` seam instead of wiring `ragas` to a non-default
provider through `langchain`; it inherits `llm.py`'s disk cache and hard
`MEDMEMGRAPH_MAX_USD` budget cap for free; it will inherit Arize/Phoenix
tracing automatically the moment `medmemgraph.observability.init_tracing()`
is called from any entry point (it exists, opt-in, not yet wired into one
as of this module landing); and it carries this project's own judge-bias
mitigations (cross-family judge, rubric-style prompting, fixed field
ordering) rather than `ragas`'s own prompt conventions.

## 2. The four metrics — the actual algorithm, not just the formula

Every metric routes through `medmemgraph.llm.complete()` with
`model=llm.JUDGE_MODEL` (Google `gemini-3.5-flash-lite` by default — the
same cross-family judge `judge.py` already argues for at length: a
different provider family from `llm.ANSWER_MODEL`, mitigating the
self-preference bias `literature/17` documents) and a plain JSON Schema —
never a bare "respond in JSON" instruction.

### Faithfulness — decompose, then verify each claim independently

`F = |V| / |S|` (docs.ragas.io/en/stable/concepts/metrics/available_metrics/
faithfulness/, and the original paper, arXiv:2309.15217): decompose the
answer into a set of atomic statements `S`, then verify each against the
retrieved context; `V` is the subset an LLM judged inferable from it.

Two implementation choices worth naming explicitly:

1. **One LLM call per claim to verify it**, not one batched call for every
   claim at once. This is a deliberate cost-shape choice, not the only
   possible one — a single batched verdict call would be cheaper in call
   count (though not necessarily in tokens, since a batched prompt still
   carries every claim's text). Per-claim calls keep each verdict's
   context window small and its failure mode isolated (a malformed JSON
   response for one claim cannot corrupt every other claim's verdict in
   the same call), at the cost of `N` round trips instead of `1`. If this
   ever becomes the wrong tradeoff, `_verify_claims` is the one function to
   change — see the "Cost" section's own explicit warning against tuning
   this to make a control score move without re-checking it.
2. **Empty context short-circuits to `score=0.0` without spending a single
   verify call.** If `sample.contexts` joins to an empty string, every
   claim is unsupported by construction — asking an LLM to confirm that is
   a wasted call, not a real judgment. Similarly, an empty or
   abstention-shaped answer (`judge.py`'s own `_looks_like_abstention`
   heuristic, reused directly rather than re-implemented) skips the
   decomposition call entirely and returns `score=None` (undefined, not
   silently 0.0 — an abstention makes no factual claims to check
   faithfulness *of*).

### Context Recall — the same shape, the opposite input

`Context Recall = (claims in the reference supported by context) / (total
claims in the reference)` (docs.ragas.io/.../context_recall/). Structurally
identical to Faithfulness's decompose-then-verify machinery
(`_decompose_claims` / `_verify_claims` are shared functions), but
decomposing `sample.ground_truth` instead of `sample.answer`, and checking
attribution to context instead of factual support. `tests/
test_ragas_metrics.py::TestContextRecall::test_uses_ground_truth_not_answer`
asserts this directly — a coding mistake here (accidentally decomposing
the system's own answer) would silently turn Context Recall into a second
Faithfulness, which is exactly the kind of "looks like a metric, isn't
measuring what it claims" bug this project's HONESTY discipline exists to
catch.

### Answer Relevancy — reverse-engineer questions, then embed and average

`AR = (1/N) * sum_i cosine_similarity(E(q_i), E(q))` (same two sources):
ask the judge model to generate N (default 3) questions the answer, taken
alone, most directly addresses; embed each generated question and the
original question; average the cosine similarities.

One documented extension beyond the bare published formula:
**noncommittal answers score `0.0` directly, with zero LLM calls.** The
RAGAS paper's own prose says the metric "penalises cases where the answer
is incomplete or ... redundant," but neither the paper nor the current
docs.ragas.io page spell out a noncommittal-flag mechanism in the formula
itself (checked directly, both sources, this session) — so this is
explicitly this module's own addition, motivated by the project's own
HONESTY requirement: without it, an answer like "I don't know" can still
generate plausible, on-topic reverse-engineered questions and score highly
relevant despite committing to nothing. The check reuses `judge.py`'s
existing `_looks_like_abstention` phrase list rather than inventing a
second heuristic.

### Context Precision — position-weighted precision, judged against the reference, never the answer

`Context Precision@K = sum_{k=1..K}(Precision@k * v_k) / (total relevant
items in top K)`, `Precision@k = TP@k / k` (docs.ragas.io/.../
context_precision/). One batched LLM call judges every retrieved chunk's
relevance to the `(question, ground_truth)` pair — RAGAS's own
`LLMContextPrecisionWithReference` shape — and the formula above is then
applied in pure Python (`_context_precision_score`).

Two edge cases handled explicitly rather than left to arithmetic to decide
by accident: `K = 0` (no retrieved chunks at all) returns `score=None`
(genuinely undefined — there is nothing to rank); `K > 0` with zero
relevant chunks returns `score=0.0` rather than `0/0` (RAGAS's own
documented convention for the degenerate case, avoiding a NaN silently
propagating into a mean later). A malformed-length LLM response (the
`relevant` array shorter or longer than the number of chunks given) is
padded with `False` / truncated rather than trusted positionally beyond
what was actually asked — the same "fail toward assuming the conservative
answer" direction `llm.py`'s own budget-check overestimate uses.

**Context Precision/Recall here vs. `eval/retrieval_eval.py`'s IR
metrics** (`recall_at_k`/`ndcg_at_k` in `eval/metrics.py`, against
MedLoCoMo's gold `evidence` field): the two ask different questions and
can disagree. The IR metrics ask *did retrieval find the annotated gold
turn/admission*; RAGAS's Context Precision/Recall ask *would an LLM judge
this retrieved text useful for reconstructing the reference answer*. A
chunk can contain the literal gold turn without being judged "useful" by
an LLM reading it in isolation, or vice versa — that disagreement is
informative, not a bug in either metric, and neither substitutes for the
other. `eval/retrieval_eval.py` does not exist in this checkout as of this
module landing.

## 3. The embedder — a mid-session correction, kept visible

The dispatching story named `graph/embedders.py` / `qwen3-0.6b` /
"running free on the RTX 3090" as an existing seam. At the point this
module was started, that exact file did not exist. Rather than either (a)
assuming the story was simply wrong and silently defaulting to a paid
embedding API, or (b) blocking the whole story on a missing file, the
underlying claim was checked directly: `Qwen/Qwen3-Embedding-0.6B` was
already present in this machine's local Hugging Face cache (native
`sentence-transformers` config, zero network needed to load it), and
`nvidia-smi` confirmed a real RTX 3090 in the environment — the story's
factual claim was correct even though the specific file path wasn't there
yet.

`graph/embedders.py` then landed **during this same session**, from a
concurrent sibling story, with a better-built seam than a standalone
loader in this file would have been: `get_backend("qwen3-0.6b") ->
EmbedderBackend`, `.encode(texts, is_query=bool)`, its own
`(model, role, content_hash)` disk cache, dimension read live off the
model. This module was updated to call that seam directly
(`_get_embedders_backend()` / `_real_embed()`) rather than keep a
duplicate loader — "import and reuse if the interface fits," the same
rule this module's own docstring states for `eval/guardrail.py`, applied
the moment there was actually something to import.

**Why `is_query=False` on both sides, not the asymmetric "query"
instruction on the original question only:** `qwen3-0.6b`'s registered
query instruction is "Given a web search query, retrieve relevant passages
that answer the query" — a retrieval framing. Answer Relevancy compares
two *questions* to each other; neither is a "web search query" retrieving
a "passage." Applying the instruction to one side only would inject a
mismatched framing and break the comparison's symmetry, so both sides get
the model's plain, unprefixed encoding instead — the property that
actually matters for a fair cosine comparison is that both sides are
treated identically, not which specific role either technically resembles
more.

If `graph.embedders` is ever unimportable, or its backend fails to load,
`_real_embed` falls back to `graph.vector_index.embed_texts` (the
already-shipped, offline, deterministic hashing-trick embedding) — still
free, still local, honestly labelled in `MetricResult.detail["embedder"]`
either way, never a silent degrade to a paid API.

`dry_run=True` never reaches any of this — it uses `llm.embed(texts,
dry_run=True)`, an existing, zero-network, deterministic stub, which is
what keeps this module's own test suite fast and GPU-free.

## 4. Honesty — why a near-1.0 score here should be trusted, not assumed

The project's stated goal is scores near 1; a metric that can only ever
report high numbers is not evidence of anything.
`literature/02` (R-MEM-054) documents an independent audit finding a
`gpt-4o-mini` LLM judge accepting 62.8% of deliberately wrong-but-
topically-related answers — the concrete, cited reason this module does
not simply trust a high LLM-judge score at face value:

- **A deliberately-wrong control is not a special case in the API** — it
  is just another `RagasSample` with a fabricated-wrong answer, scored and
  reported alongside the real ones (`SampleReport`), so a suspiciously
  high score on it is directly visible rather than averaged into a mean.
  `tests/test_ragas_metrics.py::TestFaithfulness::
  test_deliberately_wrong_control_scores_materially_lower` asserts the gap
  directly and fails loudly if it closes.
- **`evaluate_with_variance()`** runs the same sample set `n_runs` times
  (default 3) at a caller-supplied `temperature > 0` and reports mean/SD
  per metric — **forcing `use_cache=False`**, not caller-overridable: with
  `temperature` held constant across repeats (as it must be, to isolate
  sampling variance rather than a temperature change) and `llm.py`'s cache
  key being exactly `(model, system, prompt, schema, temperature)`, a
  cached repeated run would return the identical response every time and
  report a fabricated zero variance. This is a correctness trap that was
  caught and closed while designing this function, not a hypothetical.
- **Wilson intervals, not bare point estimates**, for the two metrics that
  are literal pooled claim-level proportions (Faithfulness, Context
  Recall). Context Precision is a weighted formula, not a bare proportion
  — deliberately *not* Wilson-pooled, since doing so would misrepresent
  the metric's own position-weighting. Answer Relevancy is a continuous
  cosine-similarity mean, not a proportion at all — its interval comes
  from `eval.metrics.bootstrap_ci` (percentile bootstrap) instead, the
  statistically correct tool for a mean with no closed form.
- **Every real call's measured cost is carried through**, never a token-
  count estimate. Call-count scales predictably: Faithfulness / Context
  Recall cost `1 + (1 per extracted claim)` — 0 entirely on an empty
  answer/reference or empty context; Answer Relevancy costs `1` (`0` on a
  noncommittal answer); Context Precision costs `1` (`0` with no context
  or no reference). `evaluate(..., sample_size=N)` caps the sample set
  before any call is made.

## 5. Live-verified real run

Two samples, real `gemini-3.5-flash-lite` judge calls and a real,
GPU-loaded `graph.embedders` `qwen3-0.6b` backend (not `dry_run`, not a
fake): a faithful answer/context pair, and a **deliberately-wrong
control** — same question, same context, same reference, but a fabricated
wrong answer (a different drug, a different diagnosis).

| sample | faithfulness | answer_relevancy | context_precision | context_recall |
|---|---|---|---|---|
| faithful-control-A | **1.000** | 0.731 | 1.000 | 1.000 |
| deliberately-wrong-control-B | **0.000** | 0.619 | 1.000 | 1.000 |

**Reading this honestly, not just reporting it:**

- **Faithfulness discriminates cleanly** (1.0 vs. 0.0) — this is the
  metric this project needs to catch a fabricated clinical fact, and it
  did, on a real judge call, not a fake.
- **Answer Relevancy does *not* discriminate much (0.731 vs. 0.619,
  both plausible-looking)** — and per §2 above, this is *expected*
  behavior, not a bug: Answer Relevancy measures topical alignment to the
  *question* ("what medication for diabetes"), which a wrong-but-related
  drug name still satisfies. This is the concrete, live version of the
  exact failure mode `literature/02` (R-MEM-054) warns about — a
  wrong-but-topically-adjacent answer scoring deceptively well on a
  metric that was never designed to catch factual wrongness in the first
  place. Reading Answer Relevancy in isolation as "is this answer good"
  would be a mistake; reading it alongside Faithfulness (which *did*
  catch this exact case) is the intended use.
- **Context Precision/Recall both 1.000 for both samples** — correct and
  expected: both metrics are judged against `(question, ground_truth,
  contexts)` only, never the system's `answer` (§2), so a wrong answer
  cannot move either number. This is a feature, not a coincidence — it is
  exactly why RAGAS separates "did retrieval find good evidence" from
  "did the answer use it faithfully" into different metrics.

**Cost, measured, not modeled:** `total_cost_usd=$0.001136`,
`cost_per_sample_usd=$0.000568`, `18` real judge calls across both
samples (the two samples share an identical `context_precision`/
`context_recall` prompt — same question, context, and reference — so the
second sample's calls for those two metrics were free disk-cache hits,
observed live: control's per-sample cost, `$0.000417`, is lower than the
faithful sample's, `$0.000719`, for exactly this reason). Linear
projection (`project_full_run_cost`, never a token-count model) from this
measured per-sample cost:

| projected full-run size | projected cost |
|---|---|
| 30 samples | $0.017 |
| 100 samples | $0.057 |
| 964 samples (one real MedLoCoMo patient's full QA set, per `eval-harness.md`'s own corpus numbers) | $0.548 |

See the dev-ml return note (`.claude/logs/dev.log.md`) for this story for
the full pasted `pytest -v` output and the raw script output these
numbers are taken from.
