"""tests/test_retrieval_eval.py — `eval/retrieval_eval.py` (the IR metric
sweep + embedder/reranker ablation story).

Two tiers, matching this repo's established convention
(`test_embedders.py`/`test_reranker.py`'s own "offline pure-Python
assertions vs. real-model tier" split):

1. Offline, hand-computed: `admission_only_evidence`/`turn_only_evidence`,
   `evaluate_item` (full hand-worked Recall/Hit/nDCG at both grains, cross-
   checked against `test_metrics.py`'s own worked nDCG example), and
   `aggregate_config`'s mean/denominator arithmetic (built from hand-crafted
   `ItemMetrics`, no model, no corpus). `paired_reranker_vs_noop` is
   exercised with a fully-discordant synthetic pair (deterministic
   McNemar/MDE) and a no-effect pair (deterministic `underpowered=True`).

2. One lightweight real-model tier: `run_sweep` end-to-end against a tiny
   2-admission synthetic conversation (monkeypatched loader, `qwen3-0.6b` +
   `"noop"` only — no cross-encoder load), asserting only the grounding
   invariants that hold regardless of the real model's actual similarity
   ranking (both admissions always fit inside a 2-unit index, so Recall/Hit
   are rank-order-independent; nDCG is NOT asserted to an exact value here
   for that reason).
"""

from __future__ import annotations

import math

import pytest

from medmemgraph.contracts import RetrieveItem
from medmemgraph.eval import metrics
from medmemgraph.eval import retrieval_eval as re
from medmemgraph.pipeline.loader import Admission, Conversation

pytestmark = pytest.mark.timeout(180)


# ---------------------------------------------------------------------------
# admission_only_evidence / turn_only_evidence
# ---------------------------------------------------------------------------


def test_admission_only_evidence_strips_turn_ids():
    evidence = {"admissions": ["adm-1", "adm-2"], "turn_ids": [5, 9]}
    assert re.admission_only_evidence(evidence) == {"admissions": ["adm-1", "adm-2"]}


def test_admission_only_evidence_handles_missing_admissions_key():
    assert re.admission_only_evidence({}) == {"admissions": []}


def test_turn_only_evidence_none_when_no_turn_ids():
    assert re.turn_only_evidence({"admissions": ["adm-1"]}) is None
    assert re.turn_only_evidence({"admissions": ["adm-1"], "turn_ids": []}) is None


def test_turn_only_evidence_keeps_both_fields_when_present():
    evidence = {"admissions": ["adm-1", "adm-2"], "turn_ids": [5, 9]}
    assert re.turn_only_evidence(evidence) == {"admissions": ["adm-1", "adm-2"], "turn_ids": [5, 9]}


# ---------------------------------------------------------------------------
# evaluate_item — full hand-worked Recall/Hit/nDCG at both grains.
# See module docstring's worked example / this file's own inline comments
# for the by-hand derivation; cross-checked numerically against
# test_metrics.py::test_ndcg_at_k_hand_computed's identical dcg/idcg shape.
# ---------------------------------------------------------------------------


def _mk(session_id: str, turn_ids: list[int], text: str = "x") -> RetrieveItem:
    return RetrieveItem(text=text, session_id=session_id, turn_ids=turn_ids, score=0.0, channel="vector")


class TestEvaluateItemHandComputed:
    EVIDENCE = {"admissions": ["adm-1", "adm-2"], "turn_ids": [5, 9]}

    # Bi-encoder stage (pre-rerank): r0=adm-2/turn1 (admission hit, no turn
    # hit), r1=adm-1/turn5 (admission + turn hit), r2=adm-3/turn9 (wrong
    # admission entirely -- turn 9 is gold but attached to a non-gold
    # admission, so it must NOT count under either grain).
    BI = [_mk("adm-2", [1]), _mk("adm-1", [5]), _mk("adm-3", [9])]

    # Post-rerank order: deliberately reshuffled so the one turn-relevant
    # item (adm-1/turn5) drops to rank 3 -- this is what makes the Hit@2
    # turn-grain "not yet found" case below a genuine k-cutoff effect, not
    # a vacuous one.
    RERANKED = [_mk("adm-3", [9]), _mk("adm-2", [1]), _mk("adm-1", [5])]

    def test_recall_admission_is_full_regardless_of_k(self):
        result = re.evaluate_item(
            patient_id="p", qa_id="q", embedder="e", reranker="r",
            bi_encoder_retrieved=self.BI, reranked=self.RERANKED, evidence=self.EVIDENCE,
            retrieve_latency_ms=1.0, rerank_latency_ms=1.0,
        )
        # both gold admissions (adm-1 via r1, adm-2 via r0) found -> 2/2 = 1.0
        # for every RECALL_KS value (all >= 3 = len(BI)).
        for k in re.RECALL_KS:
            assert result.recall_admission[k] == pytest.approx(1.0, abs=1e-9)

    def test_recall_turn_is_half(self):
        result = re.evaluate_item(
            patient_id="p", qa_id="q", embedder="e", reranker="r",
            bi_encoder_retrieved=self.BI, reranked=self.RERANKED, evidence=self.EVIDENCE,
            retrieve_latency_ms=1.0, rerank_latency_ms=1.0,
        )
        # gold turns {5, 9}; only turn 5 (r1, admission adm-1 correct) is
        # found -- r2's turn 9 is dropped because its admission (adm-3) is
        # not gold. found={5} -> 1/2 = 0.5 for every RECALL_KS.
        for k in re.RECALL_KS:
            assert result.recall_turn is not None
            assert result.recall_turn[k] == pytest.approx(0.5, abs=1e-9)

    def test_hit_admission_true_at_every_k(self):
        result = re.evaluate_item(
            patient_id="p", qa_id="q", embedder="e", reranker="r",
            bi_encoder_retrieved=self.BI, reranked=self.RERANKED, evidence=self.EVIDENCE,
            retrieve_latency_ms=1.0, rerank_latency_ms=1.0,
        )
        # RERANKED[0]=adm-3 (not gold), RERANKED[1]=adm-2 (gold) -> already
        # a hit by k=2, and stays a hit at every larger k.
        for k in re.RERANK_KS:
            assert result.hit_admission[k] == pytest.approx(1.0, abs=1e-9)

    def test_hit_turn_is_false_at_k2_true_from_k5(self):
        result = re.evaluate_item(
            patient_id="p", qa_id="q", embedder="e", reranker="r",
            bi_encoder_retrieved=self.BI, reranked=self.RERANKED, evidence=self.EVIDENCE,
            retrieve_latency_ms=1.0, rerank_latency_ms=1.0,
        )
        assert result.hit_turn is not None
        # top-2 of RERANKED = [adm-3/turn9 (admission not gold -> excluded),
        # adm-2/turn1 (admission gold, turn 1 not in {5,9})] -> no turn hit yet.
        assert result.hit_turn[2] == pytest.approx(0.0, abs=1e-9)
        # top-5 (=full list) now includes adm-1/turn5 -> turn hit.
        for k in (5, 10, 20):
            assert result.hit_turn[k] == pytest.approx(1.0, abs=1e-9)

    def test_ndcg_admission_hand_computed(self):
        result = re.evaluate_item(
            patient_id="p", qa_id="q", embedder="e", reranker="r",
            bi_encoder_retrieved=self.BI, reranked=self.RERANKED, evidence=self.EVIDENCE,
            retrieve_latency_ms=1.0, rerank_latency_ms=1.0,
        )
        # rels (admission grain, n_gold=2) over RERANKED = [0, 1, 1].
        # k=2: dcg=0/log2(2)+1/log2(3); idcg(ideal_count=2)=1/log2(2)+1/log2(3)
        dcg2 = 0 / math.log2(2) + 1 / math.log2(3)
        idcg2 = 1 / math.log2(2) + 1 / math.log2(3)
        assert result.ndcg_admission[2] == pytest.approx(dcg2 / idcg2, abs=1e-9)
        # k=5 (=full list): dcg=0/log2(2)+1/log2(3)+1/log2(4); same idcg (n_gold=2 caps it)
        dcg5 = 0 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
        expected5 = dcg5 / idcg2
        assert expected5 == pytest.approx(0.6934264036172708, abs=1e-6)  # same shape as test_metrics.py's own worked case
        for k in (5, 10, 20):
            assert result.ndcg_admission[k] == pytest.approx(expected5, abs=1e-9)

    def test_ndcg_turn_hand_computed(self):
        result = re.evaluate_item(
            patient_id="p", qa_id="q", embedder="e", reranker="r",
            bi_encoder_retrieved=self.BI, reranked=self.RERANKED, evidence=self.EVIDENCE,
            retrieve_latency_ms=1.0, rerank_latency_ms=1.0,
        )
        assert result.ndcg_turn is not None
        # rels (turn grain, n_gold=2) over RERANKED = [0, 0, 1].
        idcg2 = 1 / math.log2(2) + 1 / math.log2(3)
        assert result.ndcg_turn[2] == pytest.approx(0.0, abs=1e-9)
        dcg5 = 0 / math.log2(2) + 0 / math.log2(3) + 1 / math.log2(4)
        expected5 = dcg5 / idcg2
        assert expected5 == pytest.approx(0.3065735963827292, abs=1e-9)
        for k in (5, 10, 20):
            assert result.ndcg_turn[k] == pytest.approx(expected5, abs=1e-9)

    def test_no_turn_evidence_item_leaves_turn_fields_none(self):
        evidence = {"admissions": ["adm-1"]}  # no turn_ids at all (~50% case)
        result = re.evaluate_item(
            patient_id="p", qa_id="q", embedder="e", reranker="r",
            bi_encoder_retrieved=[_mk("adm-1", [1])], reranked=[_mk("adm-1", [1])], evidence=evidence,
            retrieve_latency_ms=1.0, rerank_latency_ms=1.0,
        )
        assert result.recall_turn is None
        assert result.hit_turn is None
        assert result.ndcg_turn is None
        # admission grain is still fully computed.
        assert result.recall_admission[10] == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# aggregate_config — hand-computed means, correct n_admission/n_turn
# denominators (turn n must exclude the item with no turn evidence).
# ---------------------------------------------------------------------------


def _item(embedder="e", reranker="r", *, recall_admission, recall_turn, hit_admission, hit_turn, ndcg_admission, ndcg_turn) -> re.ItemMetrics:
    return re.ItemMetrics(
        patient_id="p", qa_id="q", embedder=embedder, reranker=reranker, n_candidates=10,
        retrieve_latency_ms=2.0, rerank_latency_ms=3.0,
        recall_admission=recall_admission, recall_turn=recall_turn,
        hit_admission=hit_admission, hit_turn=hit_turn,
        ndcg_admission=ndcg_admission, ndcg_turn=ndcg_turn,
    )


def test_aggregate_config_hand_computed_means_and_denominators():
    item1 = _item(
        recall_admission={10: 1.0, 20: 1.0, 50: 1.0, 100: 1.0, 500: 1.0},
        recall_turn={10: 0.5, 20: 0.5, 50: 0.5, 100: 0.5, 500: 0.5},
        hit_admission={2: 1.0, 5: 1.0, 10: 1.0, 20: 1.0},
        hit_turn={2: 0.0, 5: 1.0, 10: 1.0, 20: 1.0},
        ndcg_admission={2: 0.4, 5: 0.7, 10: 0.7, 20: 0.7},
        ndcg_turn={2: 0.0, 5: 0.3, 10: 0.3, 20: 0.3},
    )
    item2 = _item(  # no turn evidence
        recall_admission={10: 0.0, 20: 0.5, 50: 0.5, 100: 1.0, 500: 1.0},
        recall_turn=None,
        hit_admission={2: 0.0, 5: 0.0, 10: 1.0, 20: 1.0},
        hit_turn=None,
        ndcg_admission={2: 0.0, 5: 0.2, 10: 0.5, 20: 0.5},
        ndcg_turn=None,
    )
    item3 = _item(
        recall_admission={10: 1.0, 20: 1.0, 50: 1.0, 100: 1.0, 500: 1.0},
        recall_turn={10: 1.0, 20: 1.0, 50: 1.0, 100: 1.0, 500: 1.0},
        hit_admission={2: 1.0, 5: 1.0, 10: 1.0, 20: 1.0},
        hit_turn={2: 1.0, 5: 1.0, 10: 1.0, 20: 1.0},
        ndcg_admission={2: 1.0, 5: 1.0, 10: 1.0, 20: 1.0},
        ndcg_turn={2: 1.0, 5: 1.0, 10: 1.0, 20: 1.0},
    )
    config = re.ConfigResult(
        embedder="e", reranker="r", items=[item1, item2, item3],
        build_time_s_mean=1.5, embedder_vram_mb=100.0, reranker_vram_mb=50.0, n_patients=1,
    )
    row = re.aggregate_config(config)

    assert row.n_admission == 3
    assert row.n_turn == 2  # only item1 and item3 carry turn evidence

    assert row.recall_admission[10] == pytest.approx((1.0 + 0.0 + 1.0) / 3, abs=1e-9)
    assert row.recall_turn[10] == pytest.approx((0.5 + 1.0) / 2, abs=1e-9)

    assert row.hit_admission[2] == pytest.approx((1.0 + 0.0 + 1.0) / 3, abs=1e-9)
    assert row.hit_turn[2] == pytest.approx((0.0 + 1.0) / 2, abs=1e-9)

    assert row.ndcg_admission[5] == pytest.approx((0.7 + 0.2 + 1.0) / 3, abs=1e-9)
    assert row.ndcg_turn[5] == pytest.approx((0.3 + 1.0) / 2, abs=1e-9)

    assert row.total_vram_mb == pytest.approx(150.0, abs=1e-9)


def test_aggregate_config_no_turn_items_gives_nan_turn_row_and_zero_n():
    item = _item(
        recall_admission={k: 1.0 for k in re.RECALL_KS},
        recall_turn=None,
        hit_admission={k: 1.0 for k in re.RERANK_KS},
        hit_turn=None,
        ndcg_admission={k: 1.0 for k in re.RERANK_KS},
        ndcg_turn=None,
    )
    config = re.ConfigResult(
        embedder="e", reranker="r", items=[item],
        build_time_s_mean=0.0, embedder_vram_mb=None, reranker_vram_mb=None, n_patients=1,
    )
    row = re.aggregate_config(config)
    assert row.n_turn == 0
    assert math.isnan(row.recall_turn[10])
    assert row.total_vram_mb is None  # missing components never fabricated


# ---------------------------------------------------------------------------
# build_reranker
# ---------------------------------------------------------------------------


def test_build_reranker_noop_dispatches_to_noopreranker():
    from medmemgraph.graph.reranker import NoopReranker

    assert isinstance(re.build_reranker("noop"), NoopReranker)


def test_build_reranker_unregistered_name_raises_without_loading_anything():
    with pytest.raises(ValueError):
        re.build_reranker("not-a-real-reranker")


# ---------------------------------------------------------------------------
# mde_table — thin, calibrated wrapper over metrics.mde
# ---------------------------------------------------------------------------


def test_mde_table_matches_metrics_mde_directly():
    rows = re.mde_table(250, correlations=(0.0, 0.9))
    expected_uncorrelated = metrics.mde(250, 0.5, 0.0).mde_pp
    expected_correlated = metrics.mde(250, 0.5, 0.9).mde_pp
    by_corr = {r.correlation: r.mde_pp for r in rows}
    assert by_corr[0.0] == pytest.approx(expected_uncorrelated, abs=1e-12)
    assert by_corr[0.9] == pytest.approx(expected_correlated, abs=1e-12)
    # higher correlation -> smaller detectable effect at the same n (the
    # whole reason paired testing is more powerful on correlated systems).
    assert by_corr[0.9] < by_corr[0.0]


def test_mde_table_larger_n_detects_smaller_effects():
    small_n = {r.correlation: r.mde_pp for r in re.mde_table(250, correlations=(0.0,))}
    large_n = {r.correlation: r.mde_pp for r in re.mde_table(1000, correlations=(0.0,))}
    assert large_n[0.0] < small_n[0.0]


# ---------------------------------------------------------------------------
# paired_reranker_vs_noop — synthetic, fully deterministic SweepResults.
# ---------------------------------------------------------------------------


def _synthetic_config(embedder: str, reranker: str, hits: list[bool], ndcgs: list[float]) -> re.ConfigResult:
    items = []
    for i, (h, s) in enumerate(zip(hits, ndcgs)):
        items.append(
            re.ItemMetrics(
                patient_id="p", qa_id=f"q{i}", embedder=embedder, reranker=reranker, n_candidates=10,
                retrieve_latency_ms=1.0, rerank_latency_ms=1.0,
                recall_admission={k: 1.0 for k in re.RECALL_KS}, recall_turn=None,
                hit_admission={k: (1.0 if h else 0.0) for k in re.RERANK_KS}, hit_turn=None,
                ndcg_admission={k: s for k in re.RERANK_KS}, ndcg_turn=None,
            )
        )
    return re.ConfigResult(
        embedder=embedder, reranker=reranker, items=items,
        build_time_s_mean=0.0, embedder_vram_mb=None, reranker_vram_mb=None, n_patients=1,
    )


def test_paired_reranker_vs_noop_fully_discordant_case_is_significant_and_not_underpowered():
    n = 20
    noop = _synthetic_config("e", "noop", hits=[False] * n, ndcgs=[0.1] * n)
    helper = _synthetic_config("e", "helper", hits=[True] * n, ndcgs=[0.9] * n)
    sweep = re.SweepResult(configs=[noop, helper], patient_ids=["p"], n_candidates=10)

    out = re.paired_reranker_vs_noop(sweep, k=10)
    result = out["e"]["helper"]

    assert result["n_paired"] == n
    assert result["hit_delta"] == pytest.approx(1.0, abs=1e-9)
    # McNemar exact test recomputed independently for the identical 2x2
    # (a=0,b=0,c=20,d=0) -- fully discordant, all in "other" system's favor.
    expected = metrics.mcnemar_test([False] * n, [True] * n)
    assert result["hit_p_raw"] == pytest.approx(expected.p_value, abs=1e-12)
    assert result["reject_holm"] is True
    assert result["underpowered"] is False


def test_paired_reranker_vs_noop_no_effect_case_is_underpowered():
    n = 20
    noop = _synthetic_config("e", "noop", hits=[True, False] * (n // 2), ndcgs=[0.5] * n)
    same = _synthetic_config("e", "same", hits=[True, False] * (n // 2), ndcgs=[0.5] * n)
    sweep = re.SweepResult(configs=[noop, same], patient_ids=["p"], n_candidates=10)

    out = re.paired_reranker_vs_noop(sweep, k=10)
    result = out["e"]["same"]
    assert result["hit_delta"] == pytest.approx(0.0, abs=1e-9)
    assert result["underpowered"] is True  # any nonzero MDE exceeds a zero observed delta


def test_paired_reranker_vs_noop_no_noop_baseline_returns_empty():
    only_helper = _synthetic_config("e", "helper", hits=[True] * 5, ndcgs=[0.9] * 5)
    sweep = re.SweepResult(configs=[only_helper], patient_ids=["p"], n_candidates=10)
    assert re.paired_reranker_vs_noop(sweep, k=10) == {}


def _pair_item(pid: str, qid: str, hit_adm: float, hit_turn: float | None, ndcg_adm: float = 0.5, ndcg_turn: float | None = 0.5) -> re.ItemMetrics:
    turn = None if hit_turn is None else {k: hit_turn for k in re.RERANK_KS}
    ndcg_t = None if ndcg_turn is None else {k: ndcg_turn for k in re.RERANK_KS}
    return re.ItemMetrics(
        patient_id=pid, qa_id=qid, embedder="e", reranker="r", n_candidates=10,
        retrieve_latency_ms=1.0, rerank_latency_ms=1.0,
        recall_admission={k: 1.0 for k in re.RECALL_KS}, recall_turn=None,
        hit_admission={k: hit_adm for k in re.RERANK_KS}, hit_turn=turn,
        ndcg_admission={k: ndcg_adm for k in re.RERANK_KS}, ndcg_turn=ndcg_t,
    )


def test_paired_configs_turn_grain_drops_missing_hit_turn():
    a = [
        _pair_item("p1", "q1", 1.0, 1.0),
        _pair_item("p1", "q2", 1.0, None),
        _pair_item("p2", "q3", 0.0, 0.0),
    ]
    b = [
        _pair_item("p1", "q1", 1.0, 0.0),
        _pair_item("p1", "q2", 1.0, 1.0),
        _pair_item("p2", "q3", 0.0, 1.0),
    ]
    out = re.paired_configs(a, b, grain="turn", k=10)
    assert out["n_paired"] == 2
    assert out["hit_delta"] == pytest.approx(0.0, abs=1e-9)  # (0,1) vs (1,0)


def test_paired_configs_vs_non_noop_baseline_on_dicts():
    gpu = [
        {"patient_id": "p", "qa_id": "q0", "hit_admission": {"10": 1.0}, "hit_turn": {"10": 1.0},
         "ndcg_admission": {"10": 0.9}, "ndcg_turn": {"10": 0.9}},
        {"patient_id": "p", "qa_id": "q1", "hit_admission": {"10": 1.0}, "hit_turn": {"10": 1.0},
         "ndcg_admission": {"10": 0.9}, "ndcg_turn": {"10": 0.9}},
    ]
    cpu = [
        {"patient_id": "p", "qa_id": "q0", "hit_admission": {"10": 0.0}, "hit_turn": {"10": 0.0},
         "ndcg_admission": {"10": 0.1}, "ndcg_turn": {"10": 0.1}},
        {"patient_id": "p", "qa_id": "q1", "hit_admission": {"10": 0.0}, "hit_turn": {"10": 0.0},
         "ndcg_admission": {"10": 0.1}, "ndcg_turn": {"10": 0.1}},
    ]
    out = re.paired_configs(gpu, cpu, grain="turn", k=10)
    assert out["n_paired"] == 2
    assert out["hit_delta"] == pytest.approx(-1.0, abs=1e-9)
    assert out["a_hit"] == pytest.approx(1.0)
    assert out["b_hit"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Rendering — shape/smoke checks (real numbers come from the actual sweep,
# not this test file).
# ---------------------------------------------------------------------------


def _tiny_sweep() -> re.SweepResult:
    noop = _synthetic_config("qwen3-0.6b", "noop", hits=[True, False], ndcgs=[0.8, 0.2])
    helper = _synthetic_config("qwen3-0.6b", "qwen3-rerank-0.6b", hits=[True, True], ndcgs=[0.9, 0.7])
    return re.SweepResult(configs=[noop, helper], patient_ids=["patient-x"], n_candidates=10)


def test_render_markdown_admission_grain_includes_ceiling_note_and_all_columns():
    text = re.render_markdown(_tiny_sweep(), grain="admission")
    assert re.RECALL_CEILING_NOTE in text
    assert "Recall@500" in text
    assert "Hit@20" in text
    assert "nDCG@20" in text
    assert "qwen3-0.6b" in text and "qwen3-rerank-0.6b" in text


def test_render_markdown_turn_grain_omits_ceiling_note():
    text = re.render_markdown(_tiny_sweep(), grain="turn")
    assert re.RECALL_CEILING_NOTE not in text
    assert re.NDCG_ADMISSION_CAVEAT not in text


def test_render_markdown_admission_grain_includes_ndcg_caveat():
    text = re.render_markdown(_tiny_sweep(), grain="admission")
    assert re.NDCG_ADMISSION_CAVEAT in text
    assert "[0, 1]" in re.NDCG_ADMISSION_CAVEAT


def test_ndcg_at_k_admission_many_turns_is_at_most_one():
    """Four turns of one gold admission used to score nDCG@10 ≈ 2.56
    because IDCG used n_gold_admissions=1. That is a bug. Same labels
    on both sides → perfect ranking of those four items is 1.0."""
    class _I:
        def __init__(self, session_id, turn_ids):
            self.session_id = session_id
            self.turn_ids = turn_ids

    evidence = {"admissions": ["adm-X"]}
    retrieved = [_I("adm-X", [1]), _I("adm-X", [2]), _I("adm-X", [3]), _I("adm-X", [4])]
    score = metrics.ndcg_at_k(retrieved, evidence, 10)
    assert score == pytest.approx(1.0, abs=1e-9)
    assert 0.0 <= score <= 1.0


def test_render_markdown_rejects_unknown_grain():
    with pytest.raises(ValueError):
        re.render_markdown(_tiny_sweep(), grain="bogus")


def test_write_report_writes_three_files(tmp_path):
    paths = re.write_report(_tiny_sweep(), out_dir=tmp_path)
    assert set(paths) == {"admission", "turn", "summary"}
    for p in paths.values():
        assert p.exists()
        assert p.read_text(encoding="utf-8")  # non-empty


def test_render_terminal_summary_includes_mde_and_significance_sections():
    text = re.render_terminal_summary(_tiny_sweep())
    assert "Minimum detectable effect" in text
    assert "Paired significance" in text


# ---------------------------------------------------------------------------
# Real-model tier: run_sweep end-to-end, qwen3-0.6b + noop only (no
# cross-encoder load), grounding invariants only (see module docstring).
# ---------------------------------------------------------------------------


def _synthetic_two_admission_conversation(patient_id: str) -> Conversation:
    adm_a = Admission(
        hadm_id="adm-A",
        admission_start="2126-01-01",
        admission_end="2126-01-02",
        conversation_lines=(
            {"turn_number": 1, "time": "2126-01-01T09:00:00", "speaker": "Patient", "text": "I have been feeling nauseous."},
        ),
    )
    adm_b = Admission(
        hadm_id="adm-B",
        admission_start="2126-02-01",
        admission_end="2126-02-02",
        conversation_lines=(
            {"turn_number": 1, "time": "2126-02-01T09:00:00", "speaker": "Patient", "text": "I have been feeling fatigued."},
        ),
    )
    return Conversation(subject_id=patient_id, processed_hadm_ids=("adm-A", "adm-B"), admissions=(adm_a, adm_b))


_SYNTHETIC_QA = [
    {
        "qa_id": "q-turn",
        "question": "Did the patient report nausea?",
        "evidence": {"admissions": ["adm-A"], "turn_ids": [1]},
    },
    {
        "qa_id": "q-admission-only",
        "question": "Did the patient report fatigue?",
        "evidence": {"admissions": ["adm-B"]},
    },
]


@pytest.mark.timeout(180)
def test_run_sweep_end_to_end_real_qwen3_noop_grounding_invariants(monkeypatch):
    """Both QA items' gold admission is one of exactly 2 indexed units, so
    Recall/Hit are 1.0 regardless of the real model's actual similarity
    ranking -- this test is deterministic without needing to control (or
    know) the model's real scores, while still exercising the real
    embedder + PatientIndex + run_sweep orchestration end-to-end."""
    monkeypatch.setattr(re, "load_conversation", lambda pid, root: _synthetic_two_admission_conversation(pid))
    monkeypatch.setattr(re, "load_qa", lambda pid, root: _SYNTHETIC_QA)

    sweep = re.run_sweep(
        ["patient-retrieval-eval-smoke"],
        embedders_=("qwen3-0.6b",),
        rerankers_=("noop",),
        n_candidates=10,
        cache_path=None,
    )

    assert len(sweep.configs) == 1
    config = sweep.configs[0]
    assert config.embedder == "qwen3-0.6b"
    assert config.reranker == "noop"
    assert len(config.items) == 2  # both QA items groundable

    row = re.aggregate_config(config)
    assert row.n_admission == 2
    assert row.n_turn == 1  # only q-turn carries turn-level gold

    for k in re.RECALL_KS:
        assert row.recall_admission[k] == pytest.approx(1.0, abs=1e-9)
        assert row.recall_turn[k] == pytest.approx(1.0, abs=1e-9)
    for k in re.RERANK_KS:
        assert row.hit_admission[k] == pytest.approx(1.0, abs=1e-9)
        assert row.hit_turn[k] == pytest.approx(1.0, abs=1e-9)

    # VRAM/build-time are measured, not fabricated -- on this project's
    # verified-CUDA hardware they must be real (non-None) numbers; on a
    # CPU-only box they degrade to None rather than a fake 0.0.
    import torch

    if torch.cuda.is_available():
        assert row.embedder_vram_mb is not None and row.embedder_vram_mb >= 0.0
    assert row.build_time_s_mean >= 0.0
