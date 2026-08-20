"""graph/reranker.py — trained cross-encoder reranking, the optional
second stage of a retrieve-then-rerank pipeline over the vector/lexical
"beside HydraDB" arm (`vector_index.py`/`lexical.py`).

This module is NOT wired into `retrieve.py` / `router.py` — the frozen
`retrieve()` contract (`contracts.py` CONTRACT 2) is untouched by this
file, and nothing here is imported by it. It exists so `eval/`'s baseline
ladder can run reranking as one more *arm of an ablation* (`NoopReranker`
vs. a trained cross-encoder), the same "isolate the retrieval variable,
measure it honestly" discipline `eval/baselines/dense.py` and
`eval/baselines/lexical.py` already establish for chunk size and k.

--------------------------------------------------------------------------
Read this first — the crux, and why this file is not a reversal of a
project ban
--------------------------------------------------------------------------
This project's own planning documents (`.grok/plans/phases-medmemgraph-
2026-08-16.md`, `research/GROK-INTAKE.md`) list "LLM reranker" as
something Graph "must not redesign" / "do not build", citing a sibling
survey's finding that a **prompted, untrained** reflection-style LLM
rerank step measured accuracy dropping 58.8% -> 31.0%, *below* the naive
baseline, recovering only once fine-tuned
(`research/01-agent-memory-architectures.md`, claim AM-063, inherited by
`collaborative/literature/12-vector-search-and-hybrid-fusion.md` R-VEC-080
context). Read literally, that ban is about the *prompted/untrained*
failure mode, not about reranking as a category — and
`collaborative/literature/12-vector-search-and-hybrid-fusion.md` (this
project's own dedicated retrieval-engineering survey) draws exactly that
line explicitly, in its own words:

    "A trained cross-encoder reranker (e.g. `bge-reranker-v2-m3`) is a
    defensible, low-risk *optional* second-stage add-on if time permits
    -- a prompted/untrained LLM or off-the-shelf-domain reranking step is
    not, and should not be added." (survey Q7)

    "reranking is safe when it is a *trained, task-appropriate* model,
    and risky when it is either an untrained/prompted LLM step or an
    off-the-shelf model borrowed from a different domain (general web
    search) without validation on this project's own conversational-
    clinical data." (survey summary)

So: this module implements the *safe* half of that distinction (a
fine-tuned relevance-scoring model, no prompt, no free-form generation),
never the banned half. Two honest caveats the survey itself is explicit
about, carried forward here rather than smoothed over:

1. The survey's own positive evidence is about trained cross-encoders
   *in general* (R-VEC-078: competitive with GPT-4 listwise reranking
   in-domain; R-VEC-082: `bge-reranker-v2-m3` used as a named production
   baseline elsewhere) — not a guarantee for *this* project's
   conversational-clinical data specifically. A sibling, independent
   ablation in the same survey (R-VEC-080) found an off-the-shelf,
   non-domain-tuned web-search cross-encoder *hurting* conversational-
   memory Hit@1 by 6.9 points in one clean test. Neither
   `qwen3-rerank-0.6b` nor `bge-rerank-v2-m3` has been validated on this
   project's own held-out data as of this module landing — that
   validation is exactly what `NoopReranker` as a first-class ablation
   arm is *for* (see below), not a thing this module can assert on its
   own.
2. `literature/12` names `bge-reranker-v2-m3` specifically as its
   evidenced recommendation (Q7, R-VEC-082); `Qwen/Qwen3-Reranker-0.6B`
   (this module's default) is not evaluated anywhere in that survey. It
   is registered here because the story that dispatched this module named
   it explicitly and it is independently well-documented as a *trained*
   (not prompted) cross-encoder-style scorer (see part 2 below) — the
   same safety argument applies to it structurally, but it carries the
   same "not yet validated on our data" caveat as `bge-rerank-v2-m3`.

Do not "simplify" this into a prompted-LLM rerank step later. That is
the one thing this module deliberately is not, and the one failure mode
every citation above measured as harmful.

--------------------------------------------------------------------------
Two registered backbones, two different scoring mechanisms
--------------------------------------------------------------------------
`bge-rerank-v2-m3` (`BAAI/bge-reranker-v2-m3`) is a standard cross-encoder:
an `AutoModelForSequenceClassification` model (XLM-RoBERTa backbone, a
genuine trained regression head) whose forward pass on a
`(query, document)` pair already IS a relevance logit
(huggingface.co/BAAI/bge-reranker-v2-m3, "Huggingface Transformers" usage
section, fetched 2026-08-16). `sentence_transformers.CrossEncoder` wraps
exactly this model shape, so it is used directly (`_SentenceTransformers
CrossEncoderBackend` below).

`qwen3-rerank-0.6b` (`Qwen/Qwen3-Reranker-0.6B`) is a **fine-tuned causal
LM** (`Qwen3-0.6B-Base`), not a regression head — it has no classification
layer at all. Its trained behavior is to answer a fixed yes/no judgment
prompt, and its relevance score is read off the **next-token logits for
the literal tokens "yes" and "no"** at the final sequence position, not a
pooled/projected scalar (huggingface.co/Qwen/Qwen3-Reranker-0.6B,
"Transformers Usage" section, fetched 2026-08-16 — quoted in full in
`docs/algorithms/reranker.md` §2). Loading it with
`AutoModelForSequenceClassification` (or any generic pooled-cross-encoder
wrapper that assumes a classification head) would attach a randomly
initialized head to a model that was never trained with one — the forward
pass would run without error and produce plausible-*looking* scores that
are pure noise, exactly the "scoring it as a generic cross-encoder
produces plausible-looking but wrong rankings" trap the dispatching story
warns about. `_Qwen3YesNoBackend` below therefore loads it with
`AutoModelForCausalLM` and reproduces the model card's own
`format_instruction`/`process_inputs`/`compute_logits` recipe verbatim
(fixed chat scaffold, left-padding so the final token position is always
the real last token, `log_softmax` over exactly the `{no, yes}` logit
pair, `exp` of the yes term).

--------------------------------------------------------------------------
Part 3 (this story): a CPU-optimized reranker registry extension — why,
and what "cheap to do" bought
--------------------------------------------------------------------------
Both backbones above are GPU-class: `bge-rerank-v2-m3` measured 13.65 s for
a 100-candidate rerank on a 2-thread CPU, `qwen3-rerank-0.6b` is a full
causal-LM forward pass and is slower still per candidate. Neither is a
product on this project's actual demo target — a CPU-only, 2 vCPU / 4 GB
EC2 box (no GPU line item in the budget) — and two GPU-class models loaded
at native precision already total ~4.8 GB, more than the box's entire RAM.
This story adds seven CPU-appropriate arms spanning the quality/latency/
memory frontier that gap leaves unmeasured, so the ablation the eval
harness runs can show the real trade-off instead of assuming a GPU-latency
number transfers to CPU (the same category of unvalidated-assumption risk
`REGISTERED_MODELS`' own docstring already warns about for domain
transfer).

Every new entry is a `kind == "seq_classification"` model — a real trained
regression head, verified per-model against its own `config.json`
(`architectures` + `id2label == {"0": "LABEL_0"}`, i.e. a genuine
single-label relevance score, not a repurposed multi-class head) before
registration, the same discipline `bge-rerank-v2-m3` was held to. None
needed a new backend `kind` — `_SentenceTransformersCrossEncoderBackend`
(already the correct class for any regression-head cross-encoder) handles
all seven, with two small, spec-driven extensions: `trust_remote_code`
(one model, `jina-rerank-v1-turbo-en`, ships a custom `modeling_bert.py`)
and `backend="onnx"` (one model, `ms-marco-minilm-l6-v2-onnx-int8`).

**Why the ONNX/int8 arm was "cheap to do" rather than a story-sinking PTQ
effort:** `cross-encoder/ms-marco-MiniLM-L-6-v2`'s own Hub repo already
ships several pre-exported, pre-quantized ONNX graphs
(`huggingface_hub.list_repo_files`, verified directly in this sandbox
before this entry was written) — including `onnx/model_qint8_avx512.onnx`,
an int8-quantized graph the model's own maintainers exported and hosted.
This story's registry entry simply points `sentence_transformers.
CrossEncoder`'s own native `backend="onnx"` support (shipped in this
project's pinned `sentence-transformers>=5.7.0`, backed by
`optimum[onnxruntime]`, added to `pyproject.toml` for this story) at that
file — no custom `optimum.onnxruntime` export script, no local
quantization pass. Measured directly in this sandbox on a 2-thread CPU (`
torch.set_num_threads(2)`, 100 synthetic candidate pairs, mean of 5 timed
calls after one untimed warm-up call): the native fp32 torch path scored
100 pairs in 0.423 s; the int8 ONNX path scored the same 100 pairs in
0.124 s — a ~3.4x wall-clock speedup on identical hardware, on top of the
~4x smaller weights-only
memory footprint (90.9 MB fp32 vs. 22.7 MB int8 — see `RerankerModelSpec.
approx_ram_mb`). See `docs/algorithms/reranker.md` §8 for the full table
and the reasoning for choosing `ms-marco-minilm-l6-v2` (not a smaller or
larger candidate) as the ONNX arm's base model.

Quantization is a **memory-fit lever, not only a speed one** on the 4 GB
box: the story's own framing (`bge-rerank-v2-m3` alone, fp32, already
~2.27 GB) means the int8 MiniLM-L6 arm (~23 MB) is the only registered
entry here that comfortably coexists with the rest of this project's
CPU-resident process (embedder + lexical index + Python/torch runtime
overhead) inside a 4 GB ceiling with real headroom to spare.

**The one disqualifying finding this session's own measurement caught,
that the params column alone would have hidden:** `jina-rerank-v1-turbo-en`
is a ~37.8M-parameter model — smaller than `bge-rerank-base`, comparable
to `ms-marco-minilm-l12-v2` — but measured directly in this sandbox, its
custom `modeling_bert.py` allocates a single `bert.encoder.alibi` buffer
of shape `(1, 12, 8192, 8192)` float32 (~3.22 GB) **at construction time**,
sized for the model's full 8192-token max context regardless of the
actual input length. That is 21x the checkpoint's own real trained-weight
footprint, and alone exceeds this story's entire 4 GB deployment target.
`RerankerModelSpec.approx_ram_mb` for this one entry is therefore
deliberately set to the measured real total, not the `params * 4 bytes`
convention every other entry uses (see that field's own docstring and the
registry entry's inline comment) — reporting the naive, misleadingly
small number here would be exactly the kind of unobserved, hand-waved
result this project's own Definition of Done forbids ("a model that fails
to load or OOMs is a finding — report it, do not quietly drop it"; this
model does not literally OOM in this session's environment, but would
almost certainly fail to fit the story's actual 4 GB target). See
`docs/algorithms/reranker.md` §8 for the full writeup.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Callable, Protocol, runtime_checkable

from medmemgraph.contracts import RetrieveItem

__all__ = [
    "RerankerBackend",
    "NoopReranker",
    "RerankerModelSpec",
    "REGISTERED_MODELS",
    "DEFAULT_MODEL_KEY",
    "CrossEncoderReranker",
    "TwoStageResult",
    "two_stage_retrieve",
]


# ---------------------------------------------------------------------------
# RerankerBackend — the structural contract every ablation arm implements.
# ---------------------------------------------------------------------------


@runtime_checkable
class RerankerBackend(Protocol):
    """`rerank(query, docs, top_k) -> [(original_index, score), ...]`,
    sorted by `score` descending, length `min(top_k, len(docs))` (or
    `len(docs)` if `top_k is None`). `original_index` indexes into the
    `docs` list the caller passed in, so a caller can always map a result
    back to the source `RetrieveItem` it came from
    (`two_stage_retrieve` below does exactly that). Every reranking arm of
    the ablation — including "no rerank at all" (`NoopReranker`) — is this
    one shape, so the eval harness can swap arms without a special case
    per arm."""

    def rerank(self, query: str, docs: list[str], top_k: int | None = None) -> list[tuple[int, float]]: ...


# ---------------------------------------------------------------------------
# NoopReranker — the "no rerank" arm, first-class, not a caller-side special
# case.
# ---------------------------------------------------------------------------


class NoopReranker:
    """Identity reranker: returns `docs` in its input order, unscored. This
    is deliberately a real `RerankerBackend` implementation, not `None`/a
    caller-side `if reranker is not None` branch — the whole point of the
    ablation this module exists for is to honestly show reranking helping
    *or hurting*, and that comparison only means something if "no rerank"
    goes through the exact same call shape and code path as every trained
    arm (`two_stage_retrieve` below never special-cases it)."""

    name = "noop"

    def rerank(self, query: str, docs: list[str], top_k: int | None = None) -> list[tuple[int, float]]:
        n = len(docs)
        k = n if top_k is None else max(0, min(top_k, n))
        # Descending synthetic scores (index 0 highest, strictly
        # decreasing) rather than a constant/zero score for every row: a
        # caller that trusts list order AND a caller that re-sorts by
        # score both reconstruct the exact input order this way — "no
        # rerank" must mean order-preserving under either reading of the
        # `RerankerBackend` contract, not just the one this module happens
        # to use internally.
        return [(i, float(n - i)) for i in range(k)]


# ---------------------------------------------------------------------------
# Registered trained cross-encoder backbones.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RerankerModelSpec:
    key: str
    hf_id: str
    kind: str  # "causal_yesno" (Qwen3-Reranker) | "seq_classification" (BGE-shape CE)
    instruction: str | None = None
    """Only meaningful for `kind == "causal_yesno"` — Qwen3-Reranker's own
    model card: "we recommend that developers create tailored instructions
    specific to their tasks and scenarios" (1-5% improvement over the
    generic web-search default it ships with)."""
    params: int = 0
    """Exact parameter count, read from the checkpoint's own safetensors
    header via `huggingface_hub.get_safetensors_metadata` (sums the
    `parameter_count` dict across dtypes) — NOT the model card's rounded
    prose figure, and not a download of the weights themselves (the
    safetensors header is a small JSON prefix; `get_safetensors_metadata`
    fetches only that). Verified directly in this sandbox for every entry
    below before this field was populated (see the dev-log return note for
    this story for the raw per-model output)."""
    approx_ram_mb: float = 0.0
    """Weights-only resident-memory estimate for the precision this model
    actually loads at on the CPU path (`CrossEncoderReranker._ensure_loaded`
    already forces `fp16=False` off CUDA — see that method's own comment —
    so every `backend="torch"` entry here is `params * 4 bytes / 1e6`,
    fp32). The one `backend="onnx"` entry below instead loads a
    pre-quantized int8 ONNX graph and is `params * 1 byte / 1e6`. This is
    a **weights-only** figure — no activation buffers, no framework
    overhead, no KV-cache-equivalent state (cross-encoders have none) — a
    deliberate simplification, called out here rather than silently
    presented as "total RAM the process will use," so a reader sizing the
    4 GB EC2 box against this table knows it is a floor, not a ceiling."""
    trust_remote_code: bool = False
    """`True` only for `jina-rerank-v1-turbo-en`: that checkpoint's
    `config.json` `auto_map` points `AutoModelForSequenceClassification` at
    a repo-hosted `modeling_bert.py` (custom `JinaBertForSequenceClassification`,
    ALiBi-based long-context BERT variant) rather than a library-built-in
    architecture — the model card's own `sentence-transformers` usage
    example passes `trust_remote_code=True` to `CrossEncoder(...)`
    explicitly (huggingface.co/jinaai/jina-reranker-v1-turbo-en, "Usage"
    §2, fetched 2026-08-17); every other entry below is a library-built-in
    architecture (`BertForSequenceClassification` / `XLMRobertaForSequenceClassification`
    / `DebertaV2ForSequenceClassification`) and never needs this."""
    backend: str = "torch"  # "torch" | "onnx"
    """`"onnx"` selects `sentence_transformers.CrossEncoder`'s own native
    `backend="onnx"` path (this project's pinned `sentence-transformers
    >=5.7.0` ships it — `optimum[onnxruntime]` added to `pyproject.toml`
    for this story supplies the ONNX Runtime session underneath). Every
    other entry stays `"torch"`, the pre-existing code path, unchanged."""
    onnx_file: str | None = None
    """Only meaningful for `backend == "onnx"` — passed through as
    `model_kwargs={"file_name": onnx_file}` to disambiguate which of a
    repo's several `onnx/*.onnx` files to load (see
    `_SentenceTransformersCrossEncoderBackend.__init__`)."""


REGISTERED_MODELS: dict[str, RerankerModelSpec] = {
    "qwen3-rerank-0.6b": RerankerModelSpec(
        key="qwen3-rerank-0.6b",
        hf_id="Qwen/Qwen3-Reranker-0.6B",
        kind="causal_yesno",
        instruction=(
            "Given a clinical conversation search query, retrieve the passage "
            "that most directly answers the query"
        ),
        params=595_776_512,
        approx_ram_mb=2383.1,
    ),
    "bge-rerank-v2-m3": RerankerModelSpec(
        key="bge-rerank-v2-m3",
        hf_id="BAAI/bge-reranker-v2-m3",
        kind="seq_classification",
        params=567_755_777,
        approx_ram_mb=2271.0,
    ),
    # -----------------------------------------------------------------
    # CPU-optimized ablation arms (this story). All six new entries below
    # were independently verified reachable on the Hub (`huggingface_hub.
    # model_info`, run directly in this sandbox before this block was
    # written) and their `config.json` architecture confirmed via
    # `huggingface_hub.hf_hub_download` — every one is a genuine trained
    # single-label (`id2label == {"0": "LABEL_0"}`) sequence-classification
    # regression head (`BertForSequenceClassification` /
    # `XLMRobertaForSequenceClassification` / `DebertaV2ForSequenceClassification`
    # / a `trust_remote_code` `JinaBertForSequenceClassification`), i.e. the
    # same "real cross-encoder head, no yes/no-token trick" shape as
    # `bge-rerank-v2-m3` — none of these needed a `causal_yesno` backend.
    # `params` for every entry below is the exact sum from
    # `huggingface_hub.get_safetensors_metadata(hf_id).parameter_count`
    # (safetensors-header-only, no weight download) — verified directly in
    # this sandbox, not the model card's rounded prose figure. See the
    # dev-log return note for this story for the raw per-model output.
    # -----------------------------------------------------------------
    "bge-rerank-base": RerankerModelSpec(
        key="bge-rerank-base",
        hf_id="BAAI/bge-reranker-base",
        kind="seq_classification",
        params=278_044_931,
        approx_ram_mb=1112.2,
    ),
    "mxbai-rerank-xsmall-v1": RerankerModelSpec(
        key="mxbai-rerank-xsmall-v1",
        hf_id="mixedbread-ai/mxbai-rerank-xsmall-v1",
        kind="seq_classification",
        params=70_830_337,
        approx_ram_mb=283.3,
    ),
    "jina-rerank-v1-turbo-en": RerankerModelSpec(
        key="jina-rerank-v1-turbo-en",
        hf_id="jinaai/jina-reranker-v1-turbo-en",
        kind="seq_classification",
        params=37_771_777,
        # NOT params*4/1e6 (that would be 151.1) — deliberately overriding
        # the field's own "weights * 4 bytes" convention (docstring) for
        # this one entry, because it would be dishonest here specifically.
        # Measured directly in this sandbox (`m.named_buffers()` on the
        # loaded `transformers` model): this checkpoint's custom JinaBERT
        # `modeling_bert.py` precomputes a SINGLE non-parameter buffer,
        # `bert.encoder.alibi`, shape `(1, 12, 8192, 8192)` float32 = 3.22
        # GB — the ALiBi positional-bias matrix, sized for the model's
        # full 8192-token max context and allocated eagerly at
        # construction time regardless of actual input length (confirmed:
        # RSS jumps ~6.7 GB during `CrossEncoder(...)` construction itself,
        # not during `.predict()`). This is 21x the checkpoint's own
        # ~151 MB of real trained weights, and alone exceeds this story's
        # entire 4 GB deployment target — see `docs/algorithms/reranker.md`
        # §8 "the one disqualifying finding" for the full measurement.
        # `params` above stays the honest weight-only count (37,771,777);
        # only `approx_ram_mb` is corrected to the real number
        # (151.1 weights + 3221.4 alibi buffer, rounded).
        approx_ram_mb=3372.5,
        trust_remote_code=True,
    ),
    "ms-marco-minilm-l12-v2": RerankerModelSpec(
        key="ms-marco-minilm-l12-v2",
        hf_id="cross-encoder/ms-marco-MiniLM-L-12-v2",
        kind="seq_classification",
        params=33_360_897,
        approx_ram_mb=133.4,
    ),
    "ms-marco-minilm-l6-v2": RerankerModelSpec(
        key="ms-marco-minilm-l6-v2",
        hf_id="cross-encoder/ms-marco-MiniLM-L-6-v2",
        kind="seq_classification",
        params=22_714_113,
        approx_ram_mb=90.9,
    ),
    "ms-marco-minilm-l6-v2-onnx-int8": RerankerModelSpec(
        key="ms-marco-minilm-l6-v2-onnx-int8",
        hf_id="cross-encoder/ms-marco-MiniLM-L-6-v2",
        kind="seq_classification",
        params=22_714_113,
        # int8 weights: ~1 byte/param vs. the torch fp32 sibling entry's 4
        # bytes/param above — same checkpoint, quantized graph.
        approx_ram_mb=22.7,
        backend="onnx",
        onnx_file="onnx/model_qint8_avx512.onnx",
    ),
    "ms-marco-tinybert-l2-v2": RerankerModelSpec(
        key="ms-marco-tinybert-l2-v2",
        hf_id="cross-encoder/ms-marco-TinyBERT-L-2-v2",
        kind="seq_classification",
        params=4_386_561,
        approx_ram_mb=17.5,
    ),
    # Finetuned MiniLM (FT-E2-S2). Additive keys. The original
    # ms-marco-minilm-l6-v2-onnx-int8 arm is untouched. Eval-only —
    # retrieve.py does not import this module.
    "ms-marco-minilm-l6-v2-ft-medlocomo": RerankerModelSpec(
        key="ms-marco-minilm-l6-v2-ft-medlocomo",
        hf_id="data/reranker_ft/ms-marco-minilm-l6-v2-ft-medlocomo",
        kind="seq_classification",
        params=22_714_113,
        approx_ram_mb=90.9,
        backend="torch",
    ),
    "ms-marco-minilm-l6-v2-ft-medlocomo-onnx-int8": RerankerModelSpec(
        key="ms-marco-minilm-l6-v2-ft-medlocomo-onnx-int8",
        hf_id="data/reranker_ft/ms-marco-minilm-l6-v2-ft-medlocomo-onnx",
        kind="seq_classification",
        params=22_714_113,
        approx_ram_mb=22.7,
        backend="onnx",
        onnx_file="onnx/model_qint8_avx512.onnx",
    ),
    "ms-marco-minilm-l6-v2-ft-orpo": RerankerModelSpec(
        key="ms-marco-minilm-l6-v2-ft-orpo",
        hf_id="data/reranker_ft/ms-marco-minilm-l6-v2-ft-orpo",
        kind="seq_classification",
        params=22_714_113,
        approx_ram_mb=90.9,
        backend="torch",
    ),
    "ms-marco-minilm-l6-v2-ft-orpo-onnx-int8": RerankerModelSpec(
        key="ms-marco-minilm-l6-v2-ft-orpo-onnx-int8",
        hf_id="data/reranker_ft/ms-marco-minilm-l6-v2-ft-orpo-onnx",
        kind="seq_classification",
        params=22_714_113,
        approx_ram_mb=22.7,
        backend="onnx",
        onnx_file="onnx/model_qint8_avx512.onnx",
    ),
    "ms-marco-minilm-l6-v2-ft-listwise": RerankerModelSpec(
        key="ms-marco-minilm-l6-v2-ft-listwise",
        hf_id="data/reranker_ft/ms-marco-minilm-l6-v2-ft-listwise",
        kind="seq_classification",
        params=22_714_113,
        approx_ram_mb=90.9,
        backend="torch",
    ),
    "ms-marco-minilm-l6-v2-ft-listwise-onnx-int8": RerankerModelSpec(
        key="ms-marco-minilm-l6-v2-ft-listwise-onnx-int8",
        hf_id="data/reranker_ft/ms-marco-minilm-l6-v2-ft-listwise-onnx",
        kind="seq_classification",
        params=22_714_113,
        approx_ram_mb=22.7,
        backend="onnx",
        onnx_file="onnx/model_qint8_avx512.onnx",
    ),
    "qwen3-rerank-0.6b-ft-listwise": RerankerModelSpec(
        key="qwen3-rerank-0.6b-ft-listwise",
        hf_id="data/reranker_ft/qwen3-rerank-0.6b-ft-listwise",
        kind="causal_yesno",
        instruction=(
            "Given a clinical conversation search query, retrieve the passage "
            "that most directly answers the query"
        ),
        params=595_776_512,
        approx_ram_mb=2383.1,
    ),
    "ms-marco-minilm-l6-v2-ft-orpo-turn": RerankerModelSpec(
        key="ms-marco-minilm-l6-v2-ft-orpo-turn",
        hf_id="data/reranker_ft/ms-marco-minilm-l6-v2-ft-orpo-turn",
        kind="seq_classification",
        params=22_714_113,
        approx_ram_mb=90.9,
        backend="torch",
    ),
    "ms-marco-minilm-l6-v2-ft-orpo-turn-onnx-int8": RerankerModelSpec(
        key="ms-marco-minilm-l6-v2-ft-orpo-turn-onnx-int8",
        hf_id="data/reranker_ft/ms-marco-minilm-l6-v2-ft-orpo-turn-onnx",
        kind="seq_classification",
        params=22_714_113,
        approx_ram_mb=22.7,
        backend="onnx",
        onnx_file="onnx/model_qint8_avx512.onnx",
    ),
}
"""Every `hf_id` above independently verified reachable on the Hub
(`huggingface_hub.model_info`, run directly in this sandbox before this
module was written — not assumed from training-data familiarity). The
CPU-optimized arms span `ms-marco-tinybert-l2-v2` (~4.4M params, the floor
of the ablation curve) up to `qwen3-rerank-0.6b` (~596M) — see this
module's own docstring §8 / `docs/algorithms/reranker.md` §8 for the real
measured CPU latency/quality trade-off this registry exists to make
measurable."""

DEFAULT_MODEL_KEY = "qwen3-rerank-0.6b"
"""The story's named default. `literature/12`'s own evidenced pick is
`bge-rerank-v2-m3` (module docstring caveat 2) — pass
`model_key="bge-rerank-v2-m3"` to use it instead; both are one-line
swaps, which is the point of registering rather than hardcoding a single
model id. Neither is CPU-appropriate for the 4 GB EC2 deployment target
this story's registry extension exists for — see §8."""


# ---------------------------------------------------------------------------
# CrossEncoderReranker — the trained-model RerankerBackend.
# ---------------------------------------------------------------------------


class CrossEncoderReranker:
    """A `RerankerBackend` backed by one of `REGISTERED_MODELS`. Lazy-
    loaded: the constructor never touches the network or a GPU — only the
    first call to `rerank()` does (`_ensure_loaded`) — so constructing one
    in a test that only checks, e.g., `isinstance(x, RerankerBackend)`
    costs nothing. fp16 on CUDA by default (this project's hardware is
    verified CUDA-capable — see the dev-log return note for this story for
    the measured VRAM/latency numbers); CPU fallback if CUDA is
    unavailable, with fp16 forced off on CPU (unsupported/slow on most CPU
    kernels — silently keeping fp16 there would be a wrong-looking-but-
    silently-degraded run, not an honest one)."""

    def __init__(
        self,
        model_key: str = DEFAULT_MODEL_KEY,
        *,
        device: str | None = None,
        batch_size: int = 32,
        fp16: bool | None = None,
        max_length: int = 4096,
    ) -> None:
        if model_key not in REGISTERED_MODELS:
            raise ValueError(
                f"unregistered reranker model_key={model_key!r}; choose one of {sorted(REGISTERED_MODELS)}"
            )
        self.spec = REGISTERED_MODELS[model_key]
        self.batch_size = batch_size
        self.max_length = max_length
        self._device = device
        self._fp16_requested = fp16
        self._backend: _Qwen3YesNoBackend | _SentenceTransformersCrossEncoderBackend | None = None
        self.last_load_time_s: float | None = None

    @property
    def name(self) -> str:
        return self.spec.key

    @property
    def device(self) -> str | None:
        """`None` until the backend has actually been loaded (the device
        is resolved lazily, at load time, not at construction time)."""
        return self._device if self._backend is not None else None

    def _resolve_device(self) -> str:
        if self._device is not None:
            return self._device
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    def _ensure_loaded(self) -> None:
        if self._backend is not None:
            return
        device = self._resolve_device()
        use_fp16 = self._fp16_requested if self._fp16_requested is not None else (device == "cuda")
        if device != "cuda":
            use_fp16 = False  # see class docstring

        t0 = time.monotonic()
        if self.spec.kind == "causal_yesno":
            self._backend = _Qwen3YesNoBackend(self.spec, device=device, fp16=use_fp16, max_length=self.max_length)
        elif self.spec.kind == "seq_classification":
            self._backend = _SentenceTransformersCrossEncoderBackend(
                self.spec, device=device, fp16=use_fp16, max_length=self.max_length
            )
        else:  # pragma: no cover - REGISTERED_MODELS only ever sets the two kinds above
            raise ValueError(f"unknown reranker kind {self.spec.kind!r}")
        self._device = device
        self.last_load_time_s = time.monotonic() - t0

    def rerank(self, query: str, docs: list[str], top_k: int | None = None) -> list[tuple[int, float]]:
        if not docs:
            return []
        self._ensure_loaded()
        assert self._backend is not None

        scores: list[float] = [0.0] * len(docs)
        for start in range(0, len(docs), self.batch_size):
            batch_docs = docs[start : start + self.batch_size]
            batch_scores = self._backend.score_batch(query, batch_docs)
            scores[start : start + len(batch_docs)] = batch_scores

        # Descending score, ties broken by original index ascending — a
        # deterministic order for exact-tie scores (e.g. `NoopReranker`-
        # adjacent stub backends in tests) rather than leaving it to
        # `sorted`'s incidental stability on a key that already encodes
        # the tie.
        order = sorted(range(len(docs)), key=lambda i: (-scores[i], i))
        k = len(docs) if top_k is None else max(0, min(top_k, len(docs)))
        return [(i, scores[i]) for i in order[:k]]


class _Qwen3YesNoBackend:
    """Qwen3-Reranker-0.6B's own yes/no-logit scoring recipe, reproduced
    verbatim from huggingface.co/Qwen/Qwen3-Reranker-0.6B's "Transformers
    Usage" section (fetched 2026-08-16; quoted in full in
    `docs/algorithms/reranker.md` §2) — NOT the generic
    `AutoModelForSequenceClassification` pattern (see module docstring,
    "Two registered backbones"). Loaded via `AutoModelForCausalLM`: the
    model is a fine-tuned causal LM, and its relevance score for
    `(query, doc)` is the softmax-normalized probability, over exactly the
    two tokens `{"no", "yes"}`, that the model's own next-token
    distribution assigns to `"yes"` after being shown a fixed judgment
    prompt ending in an empty `<think></think>` block."""

    def __init__(self, spec: RerankerModelSpec, *, device: str, fp16: bool, max_length: int) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        # left-padding is load-bearing, not cosmetic: `compute_logits`
        # below reads `logits[:, -1, :]` (the LAST sequence position) for
        # every row in the batch. With right-padding, the last position of
        # a shorter, padded sequence would be a pad token, not the real
        # final token, and the yes/no logits read off it would be
        # meaningless. Left-padding keeps the real final token in the
        # last position for every row regardless of that row's length.
        self.tokenizer = AutoTokenizer.from_pretrained(spec.hf_id, padding_side="left")
        dtype = torch.float16 if fp16 else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(spec.hf_id, dtype=dtype).to(device).eval()
        self.device = device
        self.max_length = max_length
        self.instruction = spec.instruction or (
            "Given a web search query, retrieve relevant passages that answer the query"
        )

        # Fixed chat scaffold, copied verbatim from the model card. The
        # system turn is what actually trains/constrains the model to a
        # binary yes/no answer; the assistant turn is pre-seeded with an
        # empty `<think></think>` block because Qwen3-Reranker is scored
        # on the logits at that exact next position, never on a generated
        # completion (no `model.generate()` call anywhere in this class).
        self._prefix = (
            "<|im_start|>system\nJudge whether the Document meets the "
            'requirements based on the Query and the Instruct provided. '
            'Note that the answer can only be "yes" or "no".<|im_end|>\n'
            "<|im_start|>user\n"
        )
        self._suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self._prefix_tokens = self.tokenizer.encode(self._prefix, add_special_tokens=False)
        self._suffix_tokens = self.tokenizer.encode(self._suffix, add_special_tokens=False)
        self._token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self._token_true_id = self.tokenizer.convert_tokens_to_ids("yes")

    def _format(self, query: str, doc: str) -> str:
        return f"<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: {doc}"

    def score_batch(self, query: str, docs: list[str]) -> list[float]:
        torch = self._torch
        pairs = [self._format(query, doc) for doc in docs]
        budget = self.max_length - len(self._prefix_tokens) - len(self._suffix_tokens)
        encoded = self.tokenizer(
            pairs, padding=False, truncation="longest_first", return_attention_mask=False, max_length=budget
        )
        for i, ids in enumerate(encoded["input_ids"]):
            encoded["input_ids"][i] = self._prefix_tokens + ids + self._suffix_tokens
        batch = self.tokenizer.pad(encoded, padding=True, return_tensors="pt", max_length=self.max_length)
        batch = {key: value.to(self.device) for key, value in batch.items()}

        with torch.no_grad():
            # logits[:, -1, :] — the next-token distribution at the final
            # (real, thanks to left-padding) position of every row.
            logits = self.model(**batch).logits[:, -1, :]
            true_scores = logits[:, self._token_true_id]
            false_scores = logits[:, self._token_false_id]
            # log_softmax over exactly {no, yes}, then exp the yes term:
            # the model card's own `compute_logits` — this is what makes
            # the score a genuine probability (sums to 1 with the "no"
            # probability), not a bare unnormalized logit.
            stacked = torch.stack([false_scores, true_scores], dim=1)
            log_probs = torch.nn.functional.log_softmax(stacked, dim=1)
            yes_prob = log_probs[:, 1].exp()
        return yes_prob.float().cpu().tolist()


class _SentenceTransformersCrossEncoderBackend:
    """Standard regression-head cross-encoder via
    `sentence_transformers.CrossEncoder` — correct for every `kind ==
    "seq_classification"` entry in `REGISTERED_MODELS` (`bge-rerank-v2-m3`
    and, as of this story, the seven CPU-optimized ablation arms) because
    every one of those checkpoints already IS an
    `AutoModelForSequenceClassification`-shaped model (a real trained
    relevance head, verified per-model against each `config.json` before
    registration — module docstring / `REGISTERED_MODELS` comment), which
    is exactly the model shape `CrossEncoder` wraps. No yes/no-token trick
    is needed or correct here — that is specifically a Qwen3-Reranker
    property (module docstring).

    Two `RerankerModelSpec` fields select non-default load paths, both
    resolved here rather than in `CrossEncoderReranker` so this class stays
    the single place that knows how to construct a `CrossEncoder`:

    - `trust_remote_code=True` (`jina-rerank-v1-turbo-en` only) — that
      checkpoint's `config.json` `auto_map` points at a repo-hosted
      `modeling_bert.py`; passed straight through to `CrossEncoder(...,
      trust_remote_code=...)`, which accepts it natively (this project's
      pinned `sentence-transformers>=5.7.0`).
    - `backend="onnx"` (`ms-marco-minilm-l6-v2-onnx-int8` only) — routes to
      `sentence_transformers.CrossEncoder`'s own native ONNX Runtime
      backend (via `optimum[onnxruntime]`, added to `pyproject.toml` for
      this story) instead of torch. `onnx_file` is passed as
      `model_kwargs={"file_name": ...}` to select one specific
      pre-quantized graph out of the several the source repo ships
      (verified via `huggingface_hub.list_repo_files` before this entry
      was registered: `cross-encoder/ms-marco-MiniLM-L-6-v2` ships
      `onnx/model_qint8_avx512.onnx` — an int8-quantized, AVX512-tuned
      graph already exported and hosted by the model's own maintainers, so
      this story does not need to run its own PTQ export pass at all — see
      `docs/algorithms/reranker.md` §8 for why that made this arm "cheap
      to do" rather than a story-sinking export effort). `fp16`/`dtype` are
      meaningless for an ONNX Runtime session (that is a torch-backend-only
      concept — `model_kwargs={"dtype": ...}` would be silently ignored or
      error depending on the `optimum` version, so this branch never
      passes it at all, rather than passing a value that only looks
      load-bearing)."""

    def __init__(self, spec: RerankerModelSpec, *, device: str, fp16: bool, max_length: int) -> None:
        import torch
        from sentence_transformers import CrossEncoder

        self._torch = torch
        common_kwargs: dict = dict(
            device=device,
            max_length=max_length,
            trust_remote_code=spec.trust_remote_code,
        )
        if spec.backend == "onnx":
            model_kwargs: dict = {}
            if spec.onnx_file:
                model_kwargs["file_name"] = spec.onnx_file
            self._model = CrossEncoder(
                spec.hf_id,
                backend="onnx",
                model_kwargs=model_kwargs,
                **common_kwargs,
            )
        else:
            self._model = CrossEncoder(
                spec.hf_id,
                model_kwargs={"dtype": torch.float16 if fp16 else torch.float32},
                **common_kwargs,
            )

    def score_batch(self, query: str, docs: list[str]) -> list[float]:
        pairs = [[query, doc] for doc in docs]
        # `activation_fn=Sigmoid()`: bge-reranker-v2-m3's raw transformers
        # usage returns unnormalized logits; FlagEmbedding's own
        # `compute_score(..., normalize=True)` documents applying a
        # sigmoid to map them onto [0, 1] as a relevance probability
        # (model card "Usage" section). Applied here for score
        # comparability with the Qwen3 backend's own already-[0, 1]
        # yes-probability — sigmoid is monotonic, so this never changes
        # the induced ranking, only the reported score's scale.
        raw = self._model.predict(pairs, activation_fn=self._torch.nn.Sigmoid(), convert_to_numpy=True)
        return [float(s) for s in raw]


# ---------------------------------------------------------------------------
# Two-stage retrieval helper — bi-encoder top-N, then rerank to top-k.
# ---------------------------------------------------------------------------


@dataclass
class TwoStageResult:
    """`items`: the final top-`top_k` `RetrieveItem`s, reordered by the
    reranker and with `.score` overwritten to the reranker's own score
    (provenance fields — `text`/`session_id`/`turn_ids`/`channel` — are
    untouched: `contracts.VALID_CHANNELS` is frozen and does not have a
    "reranked" value, so the bi-encoder's own channel label is preserved,
    same convention `fusion.rrf_fuse` already uses for its own re-scored
    output). `retrieve_latency_ms`/`rerank_latency_ms`: the two stages'
    wall-clock cost, recorded **separately** — story requirement: "record
    the latency of each stage separately — reranking is the expensive
    stage and the Pareto claim depends on knowing its cost.\""""

    items: list[RetrieveItem]
    retrieve_latency_ms: float
    rerank_latency_ms: float
    n_candidates: int
    """How many candidates the bi-encoder actually returned before
    reranking (may be `< n_candidates` requested, e.g. a small patient
    index) — needed to interpret `rerank_latency_ms` per-candidate."""


def two_stage_retrieve(
    query: str,
    *,
    retrieve_fn: Callable[[str, int], list[RetrieveItem]],
    reranker: RerankerBackend,
    n_candidates: int,
    top_k: int,
) -> TwoStageResult:
    """`retrieve_fn(query, n_candidates)` supplies the first stage (e.g.
    `PatientIndex.search`, `LexicalIndex.search`, or any callable with that
    shape — deliberately not hardcoded to one channel: this helper reranks
    whatever candidate pool it is handed, hybrid-fused or not). `reranker`
    is any `RerankerBackend`, including `NoopReranker` — passing
    `NoopReranker()` makes this function equivalent to plain top-`top_k`
    truncation of the bi-encoder's own ranking, with `rerank_latency_ms`
    correctly reporting that arm's near-zero cost (`ai-engineering` ch6's
    own hybrid-search framing: "cheap retriever fetches candidates,
    expensive one reranks" — this helper is exactly that two-stage
    sequence, made measurable)."""
    t0 = time.monotonic()
    candidates = retrieve_fn(query, n_candidates)
    retrieve_latency_ms = (time.monotonic() - t0) * 1000.0

    if not candidates:
        return TwoStageResult(items=[], retrieve_latency_ms=retrieve_latency_ms, rerank_latency_ms=0.0, n_candidates=0)

    t0 = time.monotonic()
    docs = [item.text for item in candidates]
    ranked = reranker.rerank(query, docs, top_k=top_k)
    rerank_latency_ms = (time.monotonic() - t0) * 1000.0

    items = [replace(candidates[idx], score=score) for idx, score in ranked]
    return TwoStageResult(
        items=items,
        retrieve_latency_ms=retrieve_latency_ms,
        rerank_latency_ms=rerank_latency_ms,
        n_candidates=len(candidates),
    )
