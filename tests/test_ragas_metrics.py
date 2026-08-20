"""tests/test_ragas_metrics.py — offline coverage for
`medmemgraph.eval.ragas_metrics`.

Every test in this file is offline: no test may make a real network call.
Two techniques, matching `tests/test_judge.py`'s established convention for
this exact seam:

1. Most tests replace `medmemgraph.llm.complete` with an in-process,
   queue-based fake (`_queued_complete`, below) that returns real
   `llm.LLMResponse` objects built from a caller-supplied list of already-
   parsed payloads, in call order — this exercises this module's own
   claim-counting / score-formula / call-orchestration logic against
   controlled, hand-computable inputs, without needing a real judge call
   (a live judge's actual *discrimination* quality — whether it really
   tells a faithful answer from a fabricated one — is verified separately,
   with a real, `dry_run=False` small-sample run; see the dev-ml return
   note for those numbers, not this file).
2. `TestDryRunNoNetwork` proves `dry_run=True` never reaches a real
   provider client or the real local embedder, via trip-wire monkeypatches
   that raise `AssertionError` if a network- or GPU-touching code path is
   ever invoked.

The autouse `isolate_llm_module` fixture points `llm.CACHE_DIR` at a
per-test `tmp_path` and resets the module's mutable singletons (same
convention `test_judge.py`/`test_llm.py` already establish) so no test
touches the real repo `data/llm_cache/` or reads the real `.env` — not that
this file relies on the cache (every test passes `use_cache=False`
explicitly), but the isolation is cheap insurance against an accidental
real cache write.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from medmemgraph import llm
from medmemgraph.eval import ragas_metrics as rm

# ---------------------------------------------------------------------------
# Isolation fixture — same pattern as test_judge.py / test_llm.py.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def isolate_llm_module(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path / "llm_cache")
    monkeypatch.setattr(llm, "_ledger", None)
    monkeypatch.setattr(llm, "_openai_client", None)
    monkeypatch.setattr(llm, "_google_client", None)
    monkeypatch.delenv("MEDMEMGRAPH_MAX_USD", raising=False)
    rm.reset_embedders_backend()
    yield
    rm.reset_embedders_backend()


def _no_network(*_a, **_kw):
    raise AssertionError("this code path should never touch a real network/GPU client")


# ---------------------------------------------------------------------------
# Fake `llm.complete` — queue of already-parsed payloads, returned in order.
# ---------------------------------------------------------------------------


def _queued_complete(payloads: list[dict]):
    """Returns `(fake_complete, calls)`. `fake_complete` matches
    `llm.complete`'s signature closely enough for every call site in
    `ragas_metrics.py`; `calls` records each invocation's `(system, prompt,
    schema)` so a test can assert call count and/or prompt content."""
    it = iter(payloads)
    calls: list[dict] = []

    def _fake(
        prompt,
        *,
        model=None,
        schema=None,
        system=None,
        max_tokens=1024,
        temperature=0.0,
        dry_run=False,
        use_cache=True,
        max_retries=5,
    ):
        calls.append({"prompt": prompt, "system": system, "schema": schema, "model": model})
        try:
            parsed = next(it)
        except StopIteration as exc:
            raise AssertionError(
                f"fake llm.complete called more times ({len(calls)}) than payloads were queued "
                f"({len(payloads)}) -- prompt was: {prompt!r}"
            ) from exc
        return llm.LLMResponse(
            text=json.dumps(parsed), parsed=parsed, model=model or "fake-model",
            prompt_tokens=10, completion_tokens=5, cost_usd=0.0001,
            latency_ms=1.0, cached=False, attempts=1,
        )

    return _fake, calls


def _sample(**overrides) -> rm.RagasSample:
    defaults = dict(
        question="What medication does the patient take for diabetes?",
        answer="The patient takes metformin 500mg twice daily for type 2 diabetes.",
        contexts=("2026-01-01: Patient reports taking metformin 500mg twice daily for type 2 diabetes.",),
        ground_truth="Metformin 500mg twice daily.",
        sample_id="s1",
    )
    defaults.update(overrides)
    return rm.RagasSample(**defaults)


# ---------------------------------------------------------------------------
# 1. Faithfulness — F = |V| / |S|
# ---------------------------------------------------------------------------


class TestFaithfulness:
    def test_fully_faithful_answer_scores_near_one(self, monkeypatch):
        fake, calls = _queued_complete(
            [
                {"claims": ["Patient takes metformin 500mg twice daily.", "Patient has type 2 diabetes."]},
                {"attributable": True, "reason": "context states metformin 500mg BID"},
                {"attributable": True, "reason": "context states T2DM diagnosis"},
            ]
        )
        monkeypatch.setattr(llm, "complete", fake)
        result = rm.faithfulness(_sample(), dry_run=False, use_cache=False)
        assert result.score == pytest.approx(1.0)
        assert result.k == 2 and result.n == 2
        assert len(calls) == 3  # 1 decomposition + 1 verdict per claim

    def test_deliberately_wrong_control_scores_materially_lower(self, monkeypatch):
        """HONESTY requirement: a fabricated-wrong answer must score
        materially lower than the fully-faithful case above -- if it does
        not, the metric is not discriminating and this test fails loudly,
        by design."""
        fake, calls = _queued_complete(
            [
                {"claims": ["Patient takes lisinopril 40mg daily.", "Patient has stage 4 kidney disease."]},
                {"attributable": False, "reason": "context does not mention lisinopril"},
                {"attributable": False, "reason": "context does not mention kidney disease"},
            ]
        )
        monkeypatch.setattr(llm, "complete", fake)
        control = _sample(answer="The patient takes lisinopril 40mg daily for stage 4 kidney disease.")
        result = rm.faithfulness(control, dry_run=False, use_cache=False)
        assert result.score == pytest.approx(0.0)
        real_score = 1.0  # from test_fully_faithful_answer_scores_near_one, same fixture context
        assert real_score - result.score >= 0.5, "control did not score materially lower than the real case"

    def test_multi_claim_decomposition(self, monkeypatch):
        """Claim decomposition handles a multi-claim answer -- 4 claims, a
        mixed 3/4 verdict split."""
        fake, calls = _queued_complete(
            [
                {"claims": ["Claim 1", "Claim 2", "Claim 3", "Claim 4"]},
                {"attributable": True, "reason": "r1"},
                {"attributable": True, "reason": "r2"},
                {"attributable": True, "reason": "r3"},
                {"attributable": False, "reason": "r4"},
            ]
        )
        monkeypatch.setattr(llm, "complete", fake)
        result = rm.faithfulness(_sample(), dry_run=False, use_cache=False)
        assert result.n == 4 and result.k == 3
        assert result.score == pytest.approx(0.75)
        assert len(calls) == 5
        assert [v["claim"] for v in result.detail["verdicts"]] == ["Claim 1", "Claim 2", "Claim 3", "Claim 4"]

    def test_empty_context_scores_zero_without_verify_calls(self, monkeypatch):
        fake, calls = _queued_complete([{"claims": ["Patient takes metformin."]}])
        monkeypatch.setattr(llm, "complete", fake)
        sample = _sample(contexts=())
        result = rm.faithfulness(sample, dry_run=False, use_cache=False)
        assert result.score == 0.0
        assert result.k == 0 and result.n == 1
        assert len(calls) == 1, "no per-claim verify call should fire against an empty context"

    def test_abstention_answer_is_undefined_with_zero_calls(self, monkeypatch):
        fake, calls = _queued_complete([])
        monkeypatch.setattr(llm, "complete", fake)
        sample = _sample(answer="I don't know")
        result = rm.faithfulness(sample, dry_run=False, use_cache=False)
        assert result.score is None
        assert len(calls) == 0


# ---------------------------------------------------------------------------
# 2. Context Recall
# ---------------------------------------------------------------------------


class TestContextRecall:
    def test_hand_computed_proportion(self, monkeypatch):
        fake, calls = _queued_complete(
            [
                {"claims": ["Claim A", "Claim B", "Claim C"]},
                {"attributable": True, "reason": "r1"},
                {"attributable": True, "reason": "r2"},
                {"attributable": False, "reason": "r3"},
            ]
        )
        monkeypatch.setattr(llm, "complete", fake)
        result = rm.context_recall(_sample(), dry_run=False, use_cache=False)
        assert result.score == pytest.approx(2 / 3)
        assert result.k == 2 and result.n == 3
        assert len(calls) == 4

    def test_uses_ground_truth_not_answer(self, monkeypatch):
        """Context Recall decomposes `ground_truth`, never `answer` --
        checked by asserting the first call's user content carries the
        reference text, not the answer text."""
        fake, calls = _queued_complete([{"claims": []}])
        monkeypatch.setattr(llm, "complete", fake)
        sample = _sample(answer="ANSWER_MARKER_TEXT", ground_truth="REFERENCE_MARKER_TEXT")
        rm.context_recall(sample, dry_run=False, use_cache=False)
        assert "REFERENCE_MARKER_TEXT" in calls[0]["prompt"]
        assert "ANSWER_MARKER_TEXT" not in calls[0]["prompt"]

    def test_no_ground_truth_is_undefined_with_zero_calls(self, monkeypatch):
        fake, calls = _queued_complete([])
        monkeypatch.setattr(llm, "complete", fake)
        result = rm.context_recall(_sample(ground_truth=""), dry_run=False, use_cache=False)
        assert result.score is None
        assert len(calls) == 0

    def test_empty_context_scores_zero_without_verify_calls(self, monkeypatch):
        fake, calls = _queued_complete([{"claims": ["Claim A"]}])
        monkeypatch.setattr(llm, "complete", fake)
        result = rm.context_recall(_sample(contexts=()), dry_run=False, use_cache=False)
        assert result.score == 0.0
        assert len(calls) == 1


# ---------------------------------------------------------------------------
# 3. Answer Relevancy — AR = (1/N) sum cosine_similarity(E(q_i), E(q))
# ---------------------------------------------------------------------------


class TestAnswerRelevancy:
    def test_hand_computed_with_injected_embed_fn(self, monkeypatch):
        fake, calls = _queued_complete([{"questions": ["Q1", "Q2", "Q3"]}])
        monkeypatch.setattr(llm, "complete", fake)

        vectors = {
            "What medication does the patient take for diabetes?": np.array([1.0, 0.0]),
            "Q1": np.array([1.0, 0.0]),  # cos = 1.0
            "Q2": np.array([0.0, 1.0]),  # cos = 0.0
            "Q3": np.array([1.0, 1.0]),  # cos = 1/sqrt(2)
        }

        def fake_embed(texts: list[str]) -> np.ndarray:
            return np.stack([vectors[t] for t in texts])

        result = rm.answer_relevancy(_sample(), dry_run=False, use_cache=False, embed_fn=fake_embed)
        expected = (1.0 + 0.0 + (1.0 / math.sqrt(2))) / 3.0
        assert result.score == pytest.approx(expected)
        assert result.n == 3
        assert len(calls) == 1
        assert result.detail["embedder"] == "custom"

    def test_noncommittal_answer_scores_zero_with_zero_calls(self, monkeypatch):
        fake, calls = _queued_complete([])
        monkeypatch.setattr(llm, "complete", fake)
        result = rm.answer_relevancy(_sample(answer="I don't know"), dry_run=False, use_cache=False)
        assert result.score == 0.0
        assert result.detail["noncommittal"] is True
        assert len(calls) == 0

    def test_zero_generated_questions_is_undefined(self, monkeypatch):
        fake, calls = _queued_complete([{"questions": []}])
        monkeypatch.setattr(llm, "complete", fake)
        result = rm.answer_relevancy(_sample(), dry_run=False, use_cache=False)
        assert result.score is None
        assert len(calls) == 1


class TestRealEmbedIntegration:
    """`_real_embed` reuses `graph.embedders.get_backend` (landed
    concurrently during this story) rather than a duplicate loader --
    these tests exercise the wiring with a fake backend object, never a
    real GPU/model load."""

    def test_uses_graph_embedders_backend_with_is_query_false(self, monkeypatch):
        encode_calls = []

        class _FakeBackend:
            name = "qwen3-0.6b"

            def encode(self, texts, *, is_query=False):
                encode_calls.append({"texts": list(texts), "is_query": is_query})
                return np.ones((len(texts), 4), dtype=np.float32)

        monkeypatch.setattr(rm, "_get_embedders_backend", lambda: _FakeBackend())
        vectors, label = rm._real_embed(["question", "generated q1"])
        assert label == "graph.embedders:qwen3-0.6b"
        assert vectors.shape == (2, 4)
        assert len(encode_calls) == 1
        assert encode_calls[0]["is_query"] is False

    def test_falls_back_to_hashing_when_graph_embedders_unimportable(self, monkeypatch):
        def _raise_import_error():
            raise ImportError("graph.embedders not available in this checkout")

        monkeypatch.setattr(rm, "_get_embedders_backend", _raise_import_error)
        vectors, label = rm._real_embed(["a", "b"])
        assert label.startswith("hashing-fallback (ImportError")
        assert vectors.shape[0] == 2

    def test_falls_back_to_hashing_when_backend_encode_fails(self, monkeypatch):
        class _BrokenBackend:
            name = "qwen3-0.6b"

            def encode(self, texts, *, is_query=False):
                raise RuntimeError("CUDA out of memory")

        monkeypatch.setattr(rm, "_get_embedders_backend", lambda: _BrokenBackend())
        vectors, label = rm._real_embed(["a", "b"])
        assert label.startswith("hashing-fallback (RuntimeError")
        assert vectors.shape[0] == 2


# ---------------------------------------------------------------------------
# 4. Context Precision — position-weighted precision@K
# ---------------------------------------------------------------------------


class TestContextPrecision:
    def test_hand_computed_against_ragas_formula(self, monkeypatch):
        fake, calls = _queued_complete([{"relevant": [True, False, True]}])
        monkeypatch.setattr(llm, "complete", fake)
        sample = _sample(contexts=("c1", "c2", "c3"))
        result = rm.context_precision(sample, dry_run=False, use_cache=False)
        # precision@1=1/1=1 (v=1) -> +1; precision@2=1/2 (v=0) -> +0;
        # precision@3=2/3 (v=1) -> +2/3; total_relevant=2
        expected = (1.0 + (2.0 / 3.0)) / 2.0
        assert result.score == pytest.approx(expected)
        assert result.k == 2 and result.n == 3
        assert len(calls) == 1

    def test_no_relevant_items_scores_zero(self, monkeypatch):
        fake, calls = _queued_complete([{"relevant": [False, False]}])
        monkeypatch.setattr(llm, "complete", fake)
        sample = _sample(contexts=("c1", "c2"))
        result = rm.context_precision(sample, dry_run=False, use_cache=False)
        assert result.score == 0.0
        assert result.k == 0 and result.n == 2

    def test_no_contexts_is_undefined_with_zero_calls(self, monkeypatch):
        fake, calls = _queued_complete([])
        monkeypatch.setattr(llm, "complete", fake)
        result = rm.context_precision(_sample(contexts=()), dry_run=False, use_cache=False)
        assert result.score is None
        assert len(calls) == 0

    def test_no_ground_truth_is_undefined_with_zero_calls(self, monkeypatch):
        fake, calls = _queued_complete([])
        monkeypatch.setattr(llm, "complete", fake)
        result = rm.context_precision(_sample(ground_truth=""), dry_run=False, use_cache=False)
        assert result.score is None
        assert len(calls) == 0

    def test_length_mismatch_pads_conservatively_with_false(self, monkeypatch):
        fake, calls = _queued_complete([{"relevant": [True]}])  # too short for 3 contexts
        monkeypatch.setattr(llm, "complete", fake)
        sample = _sample(contexts=("c1", "c2", "c3"))
        result = rm.context_precision(sample, dry_run=False, use_cache=False)
        assert result.detail["length_mismatch"] is True
        assert result.detail["relevant"] == [True, False, False]
        assert result.score == pytest.approx(1.0)  # precision@1=1(v=1)=1; total_relevant=1 -> 1/1


# ---------------------------------------------------------------------------
# 5. Every metric returns [0, 1] (or None when undefined)
# ---------------------------------------------------------------------------


class TestScoreRange:
    def test_every_constructed_case_is_in_unit_interval_or_none(self, monkeypatch):
        cases = [
            (rm.faithfulness, [{"claims": ["a", "b"]}, {"attributable": True, "reason": ""}, {"attributable": False, "reason": ""}]),
            (rm.context_recall, [{"claims": ["a"]}, {"attributable": True, "reason": ""}]),
            (rm.context_precision, [{"relevant": [True, True, False]}]),
        ]
        for fn, payloads in cases:
            fake, _ = _queued_complete(payloads)
            monkeypatch.setattr(llm, "complete", fake)
            result = fn(_sample(), dry_run=False, use_cache=False)
            assert result.score is None or 0.0 <= result.score <= 1.0 + 1e-9, (fn, result.score)

        fake, _ = _queued_complete([{"questions": ["Q1"]}])
        monkeypatch.setattr(llm, "complete", fake)
        result = rm.answer_relevancy(
            _sample(), dry_run=False, use_cache=False,
            embed_fn=lambda texts: np.ones((len(texts), 2)),
        )
        assert result.score is None or 0.0 <= result.score <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# 6. evaluate() — sample_size caps the number of judge calls
# ---------------------------------------------------------------------------


class TestEvaluateSampleSize:
    def test_sample_size_caps_judge_calls(self):
        """dry_run=True with metrics=['faithfulness']: every sample makes
        exactly 1 decomposition call (the schema stub's one-empty-string
        claim is filtered out by `_decompose_claims`, so 0 verify calls
        ever fire) -- a deterministic, fake-free way to assert the call
        count scales with `sample_size`."""
        samples = [_sample(sample_id=f"s{i}", question=f"q{i}") for i in range(3)]

        capped = rm.evaluate(samples, metrics=["faithfulness"], sample_size=1, dry_run=True)
        full = rm.evaluate(samples, metrics=["faithfulness"], sample_size=None, dry_run=True)

        assert capped.n_samples == 1
        assert full.n_samples == 3
        assert capped.n_judge_calls == 1
        assert full.n_judge_calls == 3
        assert capped.n_judge_calls < full.n_judge_calls

    def test_sample_size_none_uses_every_sample(self):
        samples = [_sample(sample_id=f"s{i}", question=f"q{i}") for i in range(5)]
        report = rm.evaluate(samples, metrics=["faithfulness"], dry_run=True)
        assert report.n_requested == 5
        assert report.n_samples == 5


# ---------------------------------------------------------------------------
# 7. dry_run makes no network call, anywhere -- LLM calls or the embedder
# ---------------------------------------------------------------------------


class TestDryRunNoNetwork:
    def test_embed_dry_run_never_touches_real_embed(self, monkeypatch):
        monkeypatch.setattr(rm, "_real_embed", _no_network)
        vectors, label = rm._embed_for_relevancy(["question", "generated q1"], embed_fn=None, dry_run=True)
        assert label == "dry-run-stub"
        assert vectors.shape[0] == 2

    def test_full_evaluate_dry_run_never_touches_any_real_client(self, monkeypatch):
        monkeypatch.setattr(llm, "_get_openai_client", _no_network)
        monkeypatch.setattr(llm, "_get_google_client", _no_network)
        monkeypatch.setattr(rm, "_real_embed", _no_network)

        samples = [_sample()]
        report = rm.evaluate(samples, metrics=rm.DEFAULT_METRICS, dry_run=True)

        assert report.dry_run is True
        assert report.total_cost_usd == 0.0
        assert report.n_samples == 1
        for metric_name in rm.DEFAULT_METRICS:
            score = report.mean_scores[metric_name]
            assert score is None or 0.0 <= score <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# 8. RagasReport aggregation — pooled Wilson CI, bootstrap CI
# ---------------------------------------------------------------------------


class TestRagasReportAggregation:
    def test_faithfulness_pools_k_and_n_with_wilson_ci(self, monkeypatch):
        fake, calls = _queued_complete(
            [
                # sample 1: 2 claims, 1 supported
                {"claims": ["c1", "c2"]},
                {"attributable": True, "reason": ""},
                {"attributable": False, "reason": ""},
                # sample 2: 2 claims, 2 supported
                {"claims": ["c3", "c4"]},
                {"attributable": True, "reason": ""},
                {"attributable": True, "reason": ""},
            ]
        )
        monkeypatch.setattr(llm, "complete", fake)
        samples = [_sample(sample_id="s1"), _sample(sample_id="s2")]
        report = rm.evaluate(samples, metrics=["faithfulness"], dry_run=False, use_cache=False)

        assert report.pooled_k["faithfulness"] == 3
        assert report.pooled_n["faithfulness"] == 4
        assert report.mean_scores["faithfulness"] == pytest.approx((0.5 + 1.0) / 2)

        from medmemgraph.eval import metrics as em

        expected_ci = em.wilson_interval(3, 4)
        ci = report.wilson_ci["faithfulness"]
        assert ci is not None
        assert ci[0] == pytest.approx(expected_ci.lo)
        assert ci[1] == pytest.approx(expected_ci.hi)

    def test_context_precision_is_never_wilson_pooled(self, monkeypatch):
        fake, _ = _queued_complete([{"relevant": [True, False]}])
        monkeypatch.setattr(llm, "complete", fake)
        report = rm.evaluate([_sample(contexts=("c1", "c2"))], metrics=["context_precision"], dry_run=False, use_cache=False)
        assert report.pooled_k["context_precision"] is None
        assert report.pooled_n["context_precision"] is None
        assert report.wilson_ci["context_precision"] is None

    def test_answer_relevancy_uses_bootstrap_ci_not_wilson(self, monkeypatch):
        fake, _ = _queued_complete([{"questions": ["Q1"]}, {"questions": ["Q1"]}, {"questions": ["Q1"]}])
        monkeypatch.setattr(llm, "complete", fake)
        samples = [_sample(sample_id=f"s{i}") for i in range(3)]
        report = rm.evaluate(
            samples, metrics=["answer_relevancy"], dry_run=False, use_cache=False,
            embed_fn=lambda texts: np.tile(np.array([1.0, 0.0]), (len(texts), 1)),
        )
        assert report.wilson_ci["answer_relevancy"] is None
        assert report.bootstrap_ci["answer_relevancy"] is not None


# ---------------------------------------------------------------------------
# 9. evaluate_with_variance() forces use_cache=False
# ---------------------------------------------------------------------------


class TestEvaluateWithVariance:
    def test_forces_use_cache_false_on_every_run(self, monkeypatch):
        seen_kwargs: list[dict] = []

        def fake_evaluate(samples, **kwargs):
            seen_kwargs.append(kwargs)
            return rm.RagasReport(
                samples=(), metric_names=tuple(kwargs.get("metrics", rm.DEFAULT_METRICS)),
                mean_scores={m: 0.5 for m in kwargs.get("metrics", rm.DEFAULT_METRICS)},
                n_scored={}, pooled_k={}, pooled_n={}, wilson_ci={}, bootstrap_ci={},
                n_requested=0, n_samples=0, sample_size=None, total_cost_usd=0.0,
                cost_per_sample_usd=0.0, n_judge_calls=0, dry_run=kwargs.get("dry_run", False),
                judge_model=kwargs.get("model", "m"), judge_temperature=kwargs.get("temperature", 0.0),
            )

        monkeypatch.setattr(rm, "evaluate", fake_evaluate)
        report = rm.evaluate_with_variance([_sample()], n_runs=3, temperature=0.7, dry_run=True)

        assert len(seen_kwargs) == 3
        assert all(kw["use_cache"] is False for kw in seen_kwargs)
        assert all(kw["temperature"] == 0.7 for kw in seen_kwargs)
        assert report.n_runs == 3

    def test_reports_mean_and_sd_across_runs(self, monkeypatch):
        values = iter([0.2, 0.6, 1.0])

        def fake_evaluate(samples, **kwargs):
            v = next(values)
            return rm.RagasReport(
                samples=(), metric_names=("faithfulness",), mean_scores={"faithfulness": v},
                n_scored={}, pooled_k={}, pooled_n={}, wilson_ci={}, bootstrap_ci={},
                n_requested=1, n_samples=1, sample_size=None, total_cost_usd=0.0,
                cost_per_sample_usd=0.0, n_judge_calls=1, dry_run=False, judge_model="m", judge_temperature=0.7,
            )

        monkeypatch.setattr(rm, "evaluate", fake_evaluate)
        report = rm.evaluate_with_variance([_sample()], n_runs=3, metrics=["faithfulness"], temperature=0.7)
        stat = report.stats["faithfulness"]
        assert stat.mean == pytest.approx(0.6)
        assert stat.sd == pytest.approx(0.4, abs=1e-9)
        assert stat.per_run_mean == (0.2, 0.6, 1.0)


# ---------------------------------------------------------------------------
# 10. build_sample / build_samples — the RetrieveResult contract
# ---------------------------------------------------------------------------


class TestBuildSample:
    def test_build_sample_uses_retrieve_result_items_text_as_contexts(self):
        from medmemgraph.contracts import RetrieveResult, mock_retrieve

        pack: RetrieveResult = mock_retrieve("q", "patient-0001", 3)
        sample = rm.build_sample("q", "answer text", pack, "ground truth text")
        assert sample.contexts == tuple(item.text for item in pack.items)
        assert sample.question == "q"
        assert sample.answer == "answer text"
        assert sample.ground_truth == "ground truth text"

    def test_build_samples_requires_matching_lengths(self):
        from medmemgraph.contracts import mock_retrieve

        qa_items = [{"question": "q1", "answer": "a1", "qa_id": "x1"}]
        answers = ["sys answer 1", "sys answer 2"]  # mismatched length
        retrieve_results = [mock_retrieve("q1", "patient-0001", 2)]
        with pytest.raises(ValueError):
            rm.build_samples(qa_items, answers, retrieve_results)

    def test_build_samples_happy_path(self):
        from medmemgraph.contracts import mock_retrieve

        qa_items = [
            {"question": "q1", "answer": "gt1", "qa_id": "x1"},
            {"question": "q2", "answer": "gt2", "qa_id": "x2"},
        ]
        answers = ["sys answer 1", "sys answer 2"]
        retrieve_results = [mock_retrieve("q1", "patient-0001", 2), mock_retrieve("q2", "patient-0001", 2)]
        samples = rm.build_samples(qa_items, answers, retrieve_results)
        assert len(samples) == 2
        assert samples[0].sample_id == "x1"
        assert samples[0].ground_truth == "gt1"
        assert samples[1].answer == "sys answer 2"


# ---------------------------------------------------------------------------
# 11. project_full_run_cost / render_report — plumbing sanity
# ---------------------------------------------------------------------------


class TestCostProjectionAndRendering:
    def test_project_full_run_cost_is_linear(self):
        assert rm.project_full_run_cost(0.002, 100) == pytest.approx(0.2)
        assert rm.project_full_run_cost(0.0, 100) == 0.0
        assert rm.project_full_run_cost(0.002, 0) == 0.0

    def test_render_report_is_a_nonempty_string(self, monkeypatch):
        fake, _ = _queued_complete([{"claims": ["c1"]}, {"attributable": True, "reason": ""}])
        monkeypatch.setattr(llm, "complete", fake)
        report = rm.evaluate([_sample()], metrics=["faithfulness"], dry_run=False, use_cache=False)
        text = rm.render_report(report)
        assert "faithfulness" in text
        assert "cost:" in text
