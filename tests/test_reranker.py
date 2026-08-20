"""tests/test_reranker.py — coverage for `graph/reranker.py` (trained
cross-encoder reranking, the "beside HydraDB" retrieve-then-rerank second
stage).

Two tiers, matching this repo's established convention
(`test_channels.py`'s own "synthetic-fixture tests always run / real-
artifact tests are the honest, real-numbers tier" split):

1. Offline, no model load: the `RerankerBackend` protocol holds for every
   arm; `NoopReranker` preserves input order exactly (under both a
   list-order reading and a score-resort reading); `top_k` truncation;
   `TwoStageResult` records the two stages' latency separately, using a
   fake, deterministically-timed backend (proves the *shape* of the
   two-stage accounting is right without needing a GPU for this part).

2. A real-model tier (`TestCrossEncoderRerankerReal` /
   `TestBgeRerankerReal`): the hand-built "ranked 5th by naive
   overlap -> promoted" case (the acceptance criterion's own words),
   `top_k` respected against real scores, and a real batching-invariance
   check (`batch_size=1` vs a full batch must produce the same scores).
   Both registered backbones (`Qwen/Qwen3-Reranker-0.6B`,
   `BAAI/bge-reranker-v2-m3`) are already present in this environment's
   local Hugging Face cache (verified via `find ~/.cache/huggingface/hub`
   before this file was written) — this project's own memory notes local
   inference is free (no API budget spent), so these are not
   `live`/`llm`-marker-guarded the way a real HydraDB or hosted-LLM call
   would be; they are marked with a generous `pytest.mark.timeout`
   instead, since model load (a few seconds, measured in the dev-log
   return note for this story) is the dominant cost relative to this
   repo's default 120s pytest-timeout.
"""

from __future__ import annotations

import re
import time

import pytest

from medmemgraph.contracts import RetrieveItem
from medmemgraph.graph.reranker import (
    REGISTERED_MODELS,
    CrossEncoderReranker,
    NoopReranker,
    RerankerBackend,
    TwoStageResult,
    two_stage_retrieve,
)

# ---------------------------------------------------------------------------
# Tier 1 — offline, no model load.
# ---------------------------------------------------------------------------


class _FakeBackend:
    """A `RerankerBackend` that scores by string length, deterministic and
    instant — used only to test `two_stage_retrieve`'s latency-accounting
    *shape* and score/provenance handling, never to test real reranking
    quality (that is the real-model tier's job below)."""

    name = "fake"

    def __init__(self, sleep_s: float = 0.0) -> None:
        self.sleep_s = sleep_s
        self.calls: list[tuple[str, list[str]]] = []

    def rerank(self, query: str, docs: list[str], top_k: int | None = None) -> list[tuple[int, float]]:
        self.calls.append((query, list(docs)))
        if self.sleep_s:
            time.sleep(self.sleep_s)
        order = sorted(range(len(docs)), key=lambda i: -len(docs[i]))
        k = len(docs) if top_k is None else min(top_k, len(docs))
        return [(i, float(len(docs[i]))) for i in order[:k]]


def test_rerankerbackend_protocol_holds_for_every_arm():
    """`NoopReranker`, `CrossEncoderReranker`, and a well-shaped fake all
    satisfy the `RerankerBackend` structural protocol — the eval harness
    can swap any arm in without an isinstance special case per arm. A
    bare object with no `rerank` method must NOT satisfy it — a genuine,
    discriminating check, not a tautology."""
    assert isinstance(NoopReranker(), RerankerBackend)
    assert isinstance(CrossEncoderReranker(), RerankerBackend)
    assert isinstance(_FakeBackend(), RerankerBackend)
    assert not isinstance(object(), RerankerBackend)


def test_noop_reranker_preserves_order_by_list_position():
    docs = ["alpha", "beta", "gamma", "delta"]
    ranked = NoopReranker().rerank("irrelevant query", docs)
    assert [idx for idx, _score in ranked] == [0, 1, 2, 3]


def test_noop_reranker_preserves_order_under_score_resort():
    """Even a caller that does not trust list order and re-sorts by score
    descending reconstructs the exact input order — `NoopReranker`'s
    scores are strictly decreasing by construction (never tied), so
    "preserves order" holds under either reading of the
    `RerankerBackend` contract."""
    docs = ["w", "x", "y", "z"]
    ranked = NoopReranker().rerank("q", docs)
    resorted = sorted(ranked, key=lambda pair: -pair[1])
    assert [idx for idx, _score in resorted] == [0, 1, 2, 3]


def test_noop_reranker_respects_top_k():
    docs = ["a", "b", "c", "d", "e"]
    ranked = NoopReranker().rerank("q", docs, top_k=2)
    assert [idx for idx, _score in ranked] == [0, 1]


def test_noop_reranker_empty_docs():
    assert NoopReranker().rerank("q", []) == []


def test_top_k_none_returns_every_doc_ranked():
    docs = ["a", "bb", "ccc"]
    ranked = _FakeBackend().rerank("q", docs, top_k=None)
    assert len(ranked) == 3


def test_top_k_clamped_to_available_docs():
    docs = ["a", "bb"]
    ranked = _FakeBackend().rerank("q", docs, top_k=10)
    assert len(ranked) == 2


def test_two_stage_retrieve_records_latency_per_stage_separately():
    """Story requirement: "record the latency of each stage separately."
    A slow-but-fake retrieve stage and a slow-but-fake rerank stage must
    show up as two *independently* nonzero numbers, not one merged
    total."""
    items = [
        RetrieveItem(text="doc zero", session_id="adm-1", turn_ids=[1], score=0.9, channel="vector"),
        RetrieveItem(
            text="doc one is longer than doc zero", session_id="adm-1", turn_ids=[2], score=0.5, channel="vector"
        ),
    ]

    def slow_retrieve(query: str, n: int) -> list[RetrieveItem]:
        time.sleep(0.03)
        return items[:n]

    backend = _FakeBackend(sleep_s=0.03)
    result = two_stage_retrieve(
        "irrelevant query", retrieve_fn=slow_retrieve, reranker=backend, n_candidates=2, top_k=2
    )
    assert isinstance(result, TwoStageResult)
    assert result.retrieve_latency_ms >= 20.0
    assert result.rerank_latency_ms >= 20.0
    assert result.n_candidates == 2
    # The reranker actually saw the retrieved docs' text, not something else.
    assert backend.calls[0][1] == [it.text for it in items]


def test_two_stage_retrieve_reorders_and_overwrites_score_not_provenance():
    items = [
        RetrieveItem(text="short", session_id="adm-1", turn_ids=[1], score=0.9, channel="vector"),
        RetrieveItem(
            text="a much longer document text here", session_id="adm-2", turn_ids=[7], score=0.1, channel="vector"
        ),
    ]

    def retrieve_fn(query: str, n: int) -> list[RetrieveItem]:
        return items[:n]

    result = two_stage_retrieve("q", retrieve_fn=retrieve_fn, reranker=_FakeBackend(), n_candidates=2, top_k=2)
    # _FakeBackend ranks by string length descending -> item index 1 first.
    assert [it.text for it in result.items] == ["a much longer document text here", "short"]
    assert result.items[0].session_id == "adm-2"  # provenance preserved
    assert result.items[0].turn_ids == [7]  # provenance preserved
    assert result.items[0].channel == "vector"  # frozen VALID_CHANNELS value, unchanged
    assert result.items[0].score == float(len("a much longer document text here"))  # score overwritten


def test_two_stage_retrieve_empty_candidates_short_circuits():
    result = two_stage_retrieve("q", retrieve_fn=lambda q, n: [], reranker=_FakeBackend(), n_candidates=5, top_k=3)
    assert result.items == []
    assert result.n_candidates == 0
    assert result.rerank_latency_ms == 0.0


def test_two_stage_retrieve_with_noop_is_plain_truncation():
    """`NoopReranker` as the `reranker=` argument makes `two_stage_retrieve`
    equivalent to plain top-`top_k` truncation of the bi-encoder's own
    order — the "no rerank" ablation arm, exercised through the exact
    same helper every trained arm goes through."""
    items = [
        RetrieveItem(text=f"doc {i}", session_id="adm-1", turn_ids=[i], score=1.0 - i * 0.1, channel="vector")
        for i in range(5)
    ]

    def retrieve_fn(query: str, n: int) -> list[RetrieveItem]:
        return items[:n]

    result = two_stage_retrieve("q", retrieve_fn=retrieve_fn, reranker=NoopReranker(), n_candidates=5, top_k=3)
    assert [it.text for it in result.items] == ["doc 0", "doc 1", "doc 2"]


def test_unregistered_model_key_raises():
    with pytest.raises(ValueError):
        CrossEncoderReranker(model_key="not-a-real-model")


CPU_ABLATION_MODEL_KEYS = [
    "bge-rerank-base",
    "mxbai-rerank-xsmall-v1",
    "jina-rerank-v1-turbo-en",
    "ms-marco-minilm-l12-v2",
    "ms-marco-minilm-l6-v2",
    "ms-marco-minilm-l6-v2-onnx-int8",
    "ms-marco-tinybert-l2-v2",
]
"""The seven CPU-optimized ablation arms added by this story (spanning
~4.4M to ~278M params) — every registered key except the two pre-existing
GPU-class backbones (`qwen3-rerank-0.6b`, `bge-rerank-v2-m3`, already
covered by their own `Test*Real` classes above) and `noop` (not in
`REGISTERED_MODELS` at all — `NoopReranker` is a standalone class)."""


FT_MEDLOCOMO_KEYS = (
    "ms-marco-minilm-l6-v2-ft-medlocomo",
    "ms-marco-minilm-l6-v2-ft-medlocomo-onnx-int8",
    "ms-marco-minilm-l6-v2-ft-orpo",
    "ms-marco-minilm-l6-v2-ft-orpo-onnx-int8",
    "ms-marco-minilm-l6-v2-ft-orpo-turn-onnx-int8",
    "ms-marco-minilm-l6-v2-ft-listwise",
    "ms-marco-minilm-l6-v2-ft-listwise-onnx-int8",
    "qwen3-rerank-0.6b-ft-listwise",
)


def test_registered_models_shape():
    assert set(REGISTERED_MODELS) == {
        "qwen3-rerank-0.6b",
        "bge-rerank-v2-m3",
        *CPU_ABLATION_MODEL_KEYS,
        *FT_MEDLOCOMO_KEYS,
    }
    assert REGISTERED_MODELS["qwen3-rerank-0.6b"].kind == "causal_yesno"
    assert REGISTERED_MODELS["bge-rerank-v2-m3"].kind == "seq_classification"
    # All seven CPU-optimized arms are genuine trained regression-head
    # cross-encoders (module docstring "Part 3"), never the yes/no-logit
    # causal-LM shape.
    for key in CPU_ABLATION_MODEL_KEYS:
        assert REGISTERED_MODELS[key].kind == "seq_classification"


def test_registered_models_carry_params_and_approx_ram_mb():
    """Story requirement: "Add a `params` and `approx_ram_mb` field to
    each registry entry so the frontier can plot memory too." Every
    entry — not just the seven new ones — must carry real, positive
    values (the two pre-existing entries were backfilled by this story,
    not left at the dataclass default of 0)."""
    for key, spec in REGISTERED_MODELS.items():
        assert spec.params > 0, key
        assert spec.approx_ram_mb > 0, key

    # The ablation's own stated span: ~4M (TinyBERT, the floor) to ~568M
    # (bge-rerank-v2-m3) / ~596M (qwen3-rerank-0.6b, the pre-existing
    # default) params.
    smallest = min(REGISTERED_MODELS.values(), key=lambda s: s.params)
    largest = max(REGISTERED_MODELS.values(), key=lambda s: s.params)
    assert smallest.key == "ms-marco-tinybert-l2-v2"
    assert 4_000_000 < smallest.params < 5_000_000
    assert largest.key == "qwen3-rerank-0.6b"
    assert 590_000_000 < largest.params < 600_000_000


def test_jina_ram_estimate_reflects_the_measured_alibi_buffer_not_just_weights():
    """`jina-rerank-v1-turbo-en`'s `approx_ram_mb` is a deliberate,
    documented exception to every other entry's `params * 4 bytes / 1e6`
    convention (module docstring "the one disqualifying finding") — this
    model precomputes a ~3.22 GB `bert.encoder.alibi` buffer at
    construction time. Guards against a future edit silently "fixing"
    this entry back to the naive weights-only formula, which would
    silently re-hide the exact finding this story's own measurement
    caught."""
    spec = REGISTERED_MODELS["jina-rerank-v1-turbo-en"]
    weights_only_estimate_mb = spec.params * 4 / 1e6
    assert spec.approx_ram_mb > weights_only_estimate_mb * 10


def test_jina_model_requires_trust_remote_code():
    """`jina-rerank-v1-turbo-en` ships a custom `modeling_bert.py`
    (`auto_map` in its `config.json`) — the one registered model that
    needs `trust_remote_code=True`; every other entry must NOT set it
    (a stray `True` on a library-built-in architecture would be an
    unreviewed, unnecessary trust grant)."""
    for key, spec in REGISTERED_MODELS.items():
        expected = key == "jina-rerank-v1-turbo-en"
        assert spec.trust_remote_code is expected, key


def test_onnx_int8_arm_is_registered_and_smaller_than_its_torch_sibling():
    """The ONNX/int8 arm (module docstring "Part 3") reuses the exact
    same checkpoint as `ms-marco-minilm-l6-v2` (same `hf_id`, same
    `params`) but a quantized graph — `approx_ram_mb` must reflect that:
    materially smaller than the fp32 torch sibling's, not just present."""
    onnx_spec = REGISTERED_MODELS["ms-marco-minilm-l6-v2-onnx-int8"]
    torch_spec = REGISTERED_MODELS["ms-marco-minilm-l6-v2"]
    assert onnx_spec.backend == "onnx"
    assert onnx_spec.onnx_file is not None
    assert torch_spec.backend == "torch"
    assert onnx_spec.hf_id == torch_spec.hf_id
    assert onnx_spec.params == torch_spec.params
    assert onnx_spec.approx_ram_mb < torch_spec.approx_ram_mb / 2  # int8 vs fp32


def test_old_onnx_int8_spec_untouched():
    spec = REGISTERED_MODELS["ms-marco-minilm-l6-v2-onnx-int8"]
    assert spec.hf_id == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert spec.backend == "onnx"
    assert spec.onnx_file == "onnx/model_qint8_avx512.onnx"
    assert spec.params == 22_714_113


def test_ft_medlocomo_torch_key_is_local_and_lazy():
    spec = REGISTERED_MODELS["ms-marco-minilm-l6-v2-ft-medlocomo"]
    assert spec.backend == "torch"
    assert spec.hf_id == "data/reranker_ft/ms-marco-minilm-l6-v2-ft-medlocomo"
    assert spec.params == 22_714_113
    rer = CrossEncoderReranker("ms-marco-minilm-l6-v2-ft-medlocomo")
    assert rer.spec.key == spec.key
    assert rer._backend is None  # lazy; construction must not load weights


def test_ft_medlocomo_onnx_key_points_at_local_export_not_hub():
    spec = REGISTERED_MODELS["ms-marco-minilm-l6-v2-ft-medlocomo-onnx-int8"]
    assert spec.backend == "onnx"
    assert spec.onnx_file == "onnx/model_qint8_avx512.onnx"
    assert spec.hf_id == "data/reranker_ft/ms-marco-minilm-l6-v2-ft-medlocomo-onnx"
    assert spec.hf_id != "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert spec.approx_ram_mb < 30
    rer = CrossEncoderReranker(spec.key)
    assert rer._backend is None


def test_cross_encoder_reranker_lazy_construction_costs_nothing():
    """Constructing a `CrossEncoderReranker` must not touch the network or
    a GPU — only `rerank()` does. `device` reads `None` until the backend
    has actually been loaded."""
    reranker = CrossEncoderReranker()
    assert reranker.device is None
    assert reranker.last_load_time_s is None


# ---------------------------------------------------------------------------
# Tier 2 — real models, each loaded once for the whole module.
# ---------------------------------------------------------------------------


def _naive_overlap_rank(query: str, docs: list[str]) -> list[int]:
    """A deliberately dumb bag-of-words overlap ranker — NOT
    `vector_index.embed_texts`/real cosine similarity, and not meant to
    be one: it stands in for "any bi-encoder/lexical first stage a
    trained cross-encoder can out-rank on a paraphrase case", the exact
    role a real bi-encoder plays ahead of `two_stage_retrieve`. Returns
    doc indices ordered best-first (most shared tokens with `query`)."""

    def overlap(doc: str) -> int:
        qw = set(re.findall(r"[a-z]+", query.lower()))
        dw = set(re.findall(r"[a-z]+", doc.lower()))
        return len(qw & dw)

    return sorted(range(len(docs)), key=lambda i: -overlap(docs[i]))


PROMOTION_QUERY = "What medication does the patient take for high blood pressure?"
PROMOTION_DOCS = [
    "Blood pressure medications include many different classes of drugs that lower blood pressure.",
    "Patient's blood pressure was measured at multiple visits: it was 130 over 85 once.",
    "The patient takes medication for diabetes management, specifically metformin twice daily.",
    "Blood pressure control also involves lifestyle modification such as reducing dietary sodium intake and exercise.",
    "The patient is prescribed lisinopril to help control hypertension.",
]
PROMOTION_CORRECT_IDX = 4
"""Semantically correct (hypertension IS high blood pressure; lisinopril
IS a blood-pressure medication) but shares only "patient" with the
query's literal tokens — deliberately worded so it ranks LAST (index 4 of
5, i.e. rank 5) under naive keyword overlap, verified directly below, so
a genuine promotion by the reranker is being tested, not a vacuous one."""


@pytest.fixture(scope="module")
def qwen_reranker() -> CrossEncoderReranker:
    """Loaded once per test module (a few seconds on this project's
    verified RTX 3090 hardware — see the dev-log return note for this
    story for the measured number), not once per test function."""
    reranker = CrossEncoderReranker()  # DEFAULT_MODEL_KEY == "qwen3-rerank-0.6b"
    reranker.rerank("warmup", ["warmup document"])  # force the lazy load here, once
    return reranker


@pytest.fixture(scope="module")
def bge_reranker() -> CrossEncoderReranker:
    reranker = CrossEncoderReranker(model_key="bge-rerank-v2-m3")
    reranker.rerank("warmup", ["warmup document"])
    return reranker


class TestCrossEncoderRerankerReal:
    """Default backbone: `qwen3-rerank-0.6b` (`Qwen/Qwen3-Reranker-0.6B`,
    the `causal_yesno` yes/no-logit backend)."""

    @pytest.mark.timeout(300)
    def test_naive_overlap_ranks_correct_doc_5th(self):
        """Precondition for the promotion test below: the correct doc
        really is last of 5 by naive overlap, so "the reranker promotes
        it" below is a genuine promotion, not a vacuous claim."""
        order = _naive_overlap_rank(PROMOTION_QUERY, PROMOTION_DOCS)
        assert order.index(PROMOTION_CORRECT_IDX) == 4  # last of 5 == rank 5

    @pytest.mark.timeout(300)
    def test_reranker_promotes_the_correct_doc_from_rank_5(self, qwen_reranker: CrossEncoderReranker):
        ranked = qwen_reranker.rerank(PROMOTION_QUERY, PROMOTION_DOCS)
        assert len(ranked) == 5
        top_idx, top_score = ranked[0]
        assert top_idx == PROMOTION_CORRECT_IDX
        assert top_score > 0.5  # a genuine "yes" verdict (yes-probability), not a coin flip

    @pytest.mark.timeout(300)
    def test_top_k_respected_real_model(self, qwen_reranker: CrossEncoderReranker):
        ranked = qwen_reranker.rerank(PROMOTION_QUERY, PROMOTION_DOCS, top_k=2)
        assert len(ranked) == 2
        assert ranked[0][1] >= ranked[1][1]  # sorted descending

    @pytest.mark.timeout(300)
    def test_batching_does_not_change_scores(self):
        """Scoring the same 6 docs in one batch (`batch_size=32`) vs
        one-at-a-time (`batch_size=1`, one forward pass per doc) must
        produce the same scores — a left-padding or batch-index bug would
        show up here as a score shift depending on what else shares a
        row's batch.

        Run in **fp32**, not this class's default fp16: measured directly
        while writing this test, fp16 batches of different composition
        (different per-batch padding width -> different accumulated
        rounding in the attention/matmul kernels) produce score deltas up
        to ~1.7e-3 even with fully correct batching logic — confirmed by
        re-running the identical comparison in fp32, where the delta
        drops to ~1e-6-1e-7 (pure floating-point noise). That is fp16
        *precision*, not a batching *bug*, and this test's job is to catch
        the bug (misaligned indices, wrong padding side, cross-row
        contamination), not to re-litigate fp16's own known precision
        trade-off — so it isolates that variable by using fp32 here. The
        real-numbers latency/VRAM report for this story uses fp16 (the
        realistic production setting), separately."""
        docs = PROMOTION_DOCS + [
            "An unrelated note about a scheduled follow-up appointment next month."
        ]
        reranker = CrossEncoderReranker(fp16=False)
        reranker.batch_size = 32
        batched_scores = dict(reranker.rerank(PROMOTION_QUERY, docs, top_k=None))
        reranker.batch_size = 1
        single_scores = dict(reranker.rerank(PROMOTION_QUERY, docs, top_k=None))

        assert set(batched_scores) == set(single_scores) == set(range(len(docs)))
        for idx in range(len(docs)):
            assert batched_scores[idx] == pytest.approx(single_scores[idx], abs=1e-4)


class TestBgeRerankerReal:
    """Second registered backbone: `bge-rerank-v2-m3`
    (`BAAI/bge-reranker-v2-m3`, the `seq_classification` regression-head
    backend) — proves the second registered model is not just a string in
    `REGISTERED_MODELS`, it actually loads and scores end-to-end."""

    @pytest.mark.timeout(300)
    def test_bge_backend_loads_and_scores_valid_probabilities(self, bge_reranker: CrossEncoderReranker):
        ranked = bge_reranker.rerank(PROMOTION_QUERY, PROMOTION_DOCS)
        assert len(ranked) == 5
        assert bge_reranker.spec.kind == "seq_classification"
        scores = [score for _idx, score in ranked]
        assert scores == sorted(scores, reverse=True)  # descending
        assert all(0.0 <= s <= 1.0 for s in scores)  # sigmoid-normalized, per module docstring

    @pytest.mark.timeout(300)
    def test_bge_also_promotes_the_correct_doc(self, bge_reranker: CrossEncoderReranker):
        """Not the story's literal acceptance criterion (that names one
        hand-built case, satisfied by `TestCrossEncoderRerankerReal`
        above on the default model) — an extra, honest cross-model check
        that the same promotion is not an artifact of one specific
        backbone."""
        ranked = bge_reranker.rerank(PROMOTION_QUERY, PROMOTION_DOCS)
        assert ranked[0][0] == PROMOTION_CORRECT_IDX


# ---------------------------------------------------------------------------
# Tier 3 — the seven CPU-optimized ablation arms added by this story.
#
# Explicitly `device="cpu"` (not the default cuda-if-available resolution
# `qwen_reranker`/`bge_reranker` above rely on) — this story's whole reason
# for existing is the CPU-only EC2 deployment target, so these tests must
# actually exercise the CPU load/inference path on whatever hardware runs
# them, not silently take the GPU path on a dev machine that happens to
# have one. `@pytest.mark.slow`: each model load is several seconds to
# ~20s (module-scoped fixture, so one load per model across every test
# function that uses it, not per test) — `-m "not slow"` skips this whole
# tier for a fast dev loop, per the story's "mark load-heavy tests
# appropriately so the suite stays fast."
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", params=CPU_ABLATION_MODEL_KEYS)
def cpu_ablation_reranker(request: pytest.FixtureRequest) -> CrossEncoderReranker:
    reranker = CrossEncoderReranker(model_key=request.param, device="cpu")
    reranker.rerank("warmup", ["warmup document"])  # force the lazy load here, once
    return reranker


@pytest.mark.slow
class TestCpuAblationModelsReal:
    """Story acceptance: "each registered model loads, scores, respects
    `top_k`" — exercised here for all seven CPU-optimized arms via one
    shared, parametrized, module-scoped fixture. `NoopReranker`
    order-preservation is registry-independent and already covered by
    Tier 1's `test_noop_reranker_preserves_order_*` tests above; it is not
    re-tested per model here (that would test `NoopReranker` again, not
    the registered model)."""

    @pytest.mark.timeout(300)
    def test_loads_on_cpu_and_scores_all_docs(self, cpu_ablation_reranker: CrossEncoderReranker):
        assert cpu_ablation_reranker.device == "cpu"
        ranked = cpu_ablation_reranker.rerank(PROMOTION_QUERY, PROMOTION_DOCS)
        assert len(ranked) == len(PROMOTION_DOCS)
        indices = {idx for idx, _score in ranked}
        assert indices == set(range(len(PROMOTION_DOCS)))
        scores = [score for _idx, score in ranked]
        assert scores == sorted(scores, reverse=True)  # descending, per RerankerBackend contract

    @pytest.mark.timeout(300)
    def test_respects_top_k(self, cpu_ablation_reranker: CrossEncoderReranker):
        ranked = cpu_ablation_reranker.rerank(PROMOTION_QUERY, PROMOTION_DOCS, top_k=2)
        assert len(ranked) == 2
        assert ranked[0][1] >= ranked[1][1]  # sorted descending

    # No per-model "promotes the correct doc" assertion here, unlike Tier
    # 2's two GPU-class backbones: measured directly while writing this
    # test (see the dev-log return note / `docs/algorithms/reranker.md`
    # §8 for the full table), only `bge-rerank-base` and
    # `jina-rerank-v1-turbo-en` actually promote `PROMOTION_CORRECT_IDX`
    # (idx 4, "lisinopril"/"hypertension") to rank 1 — the four
    # MS-MARCO-trained arms (`ms-marco-minilm-l12-v2`, `ms-marco-minilm-
    # l6-v2` and its onnx-int8 twin, `ms-marco-tinybert-l2-v2`) all rank
    # idx 0 ("blood pressure medications... classes of drugs," high
    # literal token overlap with the query) first instead, and
    # `mxbai-rerank-xsmall-v1` ranks idx 0 fractionally ahead of idx 4
    # (0.770 vs. 0.752). That is a genuine, measured quality signal this
    # ablation exists to surface — not a bug to assert away — so it would
    # be dishonest to bake "every CPU arm passes this one hand-built
    # semantic-promotion case" into the suite as a correctness
    # requirement; each arm's own honest capability is exactly what a
    # held-out-data ablation run (§1 caveat, module docstring) should
    # measure across many queries, not this one illustrative case.
