"""tests/test_metrics.py — hand-computed expected values for eval/metrics.py.

Discipline (story requirement): every expected value below is computed
independently of the module under test — by closed-form arithmetic, exact
combinatorics, or cross-validation against a different algorithm for the
same quantity (e.g. exact bisection vs. the closed-form MDE approximation,
which the source survey itself reports should agree to within ~1%) — never
by calling the function under test and asserting it agrees with itself.

Where a hand-computed value matches a number independently published in
``collaborative/literature/13-context-selection-and-statistics.md``
(the MDE table, R-CTX-32; the StataCorp sample-size worked example,
R-CTX-29/30) or a standard statistical table constant (normal quantiles),
that is noted inline.

The last section covers the ``collaborative/design/stories/E7/`` E7-S2/
E7-S3 literal-contract adapter functions (``mcnemar_pvalue``,
``abstention_prf``, ``paired_bootstrap``, ``token_f1``) added alongside the
canonical functions above — see ``eval/metrics.py``'s module docstring
"Story-contract reconciliation" note for why both interfaces exist.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pytest

from medmemgraph.eval import metrics as m


@dataclass
class _Item:
    session_id: str
    turn_ids: list[int]


# ---------------------------------------------------------------------------
# Normal-quantile primitives — standard statistical table constants.
# ---------------------------------------------------------------------------


def test_norm_ppf_matches_standard_table_constants():
    assert m._norm_ppf(0.975) == pytest.approx(1.959963985, abs=1e-6)
    assert m._norm_ppf(0.95) == pytest.approx(1.644853627, abs=1e-6)
    assert m._norm_ppf(0.995) == pytest.approx(2.575829304, abs=1e-6)
    assert m._norm_ppf(0.80) == pytest.approx(0.841621234, abs=1e-6)


def test_norm_cdf_ppf_are_inverses():
    for p in (0.01, 0.1, 0.5, 0.8, 0.99):
        assert m._norm_cdf(m._norm_ppf(p)) == pytest.approx(p, abs=1e-9)


# ---------------------------------------------------------------------------
# Wilson score interval — hand-computed via the closed-form formula
# transcribed directly (not via the module), R-CTX-35.
# ---------------------------------------------------------------------------


def test_wilson_interval_symmetric_case_hand_computed():
    # n=100, x=50 (p_hat=0.5), z=1.959964 (95%).
    # By symmetry center = 0.5 exactly. Half-width computed by hand:
    #   sqrt(0.25/100 + z^2/(4*100^2)) = sqrt(0.0025 + 0.00009604) = 0.0509513
    #   half_width = z*0.0509513 / (1 + z^2/100) = 1.959964*0.0509513/1.0384146
    #              = 0.0998645/1.0384146 = 0.0961818
    result = m.wilson_interval(50, 100, confidence=0.95)
    assert result.point == 0.5
    assert result.lo == pytest.approx(0.5 - 0.0961818, abs=1e-4)
    assert result.hi == pytest.approx(0.5 + 0.0961818, abs=1e-4)


def test_wilson_interval_near_boundary_hand_computed():
    # n=20, x=18 (p_hat=0.9), z=1.959964. Hand-computed via the closed form:
    #   z^2 = 3.841459
    #   center = (0.9 + 3.841459/40) / (1 + 3.841459/20)
    #          = 0.99603648 / 1.19207295 = 0.835572
    #   half_width = z*sqrt(0.9*0.1/20 + 3.841459/(4*400)) / 1.19207295
    #              = 1.959964*sqrt(0.0045+0.00240091)/1.19207295
    #              = 1.959964*0.0830717/1.19207295 = 0.136586
    result = m.wilson_interval(18, 20, confidence=0.95)
    assert result.point == 0.9
    assert result.lo == pytest.approx(0.835572 - 0.136586, abs=1e-3)
    assert result.hi == pytest.approx(0.835572 + 0.136586, abs=1e-3)
    # Sanity: Wilson stays within [0,1] and is asymmetric near the boundary
    # (unlike Wald), the entire reason R-CTX-34 recommends it here.
    assert 0.0 <= result.lo < result.hi <= 1.0
    assert (result.point - result.lo) != pytest.approx(result.hi - result.point, abs=1e-3)


def test_wilson_interval_rejects_bad_inputs():
    with pytest.raises(ValueError):
        m.wilson_interval(5, 0)
    with pytest.raises(ValueError):
        m.wilson_interval(-1, 10)
    with pytest.raises(ValueError):
        m.wilson_interval(11, 10)


# ---------------------------------------------------------------------------
# McNemar's test — exact combinatorics + closed-form chi-square, hand-computed.
# ---------------------------------------------------------------------------


def test_mcnemar_exact_hand_computed_binomial():
    # 10 items, all discordant: b=2 (A right, B wrong), c=8 (A wrong, B right).
    correct_a = [True, True, False, False, False, False, False, False, False, False]
    correct_b = [False, False, True, True, True, True, True, True, True, True]
    result = m.mcnemar_test(correct_a, correct_b)
    assert result.b == 2 and result.c == 8 and result.a == 0 and result.d == 0
    assert result.exact is True  # disc=10 < 25, auto-selected
    # Hand-computed exact two-sided binomial (sign) test, p=0.5, n=10, k=min(2,8)=2:
    #   P(X<=2) = [C(10,0)+C(10,1)+C(10,2)] / 2^10 = (1+10+45)/1024 = 56/1024 = 0.0546875
    #   two-sided p = 2*0.0546875 = 0.109375
    assert result.p_value == pytest.approx(0.109375, abs=1e-9)
    assert result.statistic == 2.0
    # Cohen's g = |b/(b+c) - 0.5| = |2/10 - 0.5| = 0.3
    assert result.effect_size == pytest.approx(0.3, abs=1e-9)


def test_mcnemar_chi2_hand_computed():
    # b=10, c=20 (disc=30 >= 25 -> chi-square branch auto-selected).
    correct_a = [True] * 10 + [False] * 20
    correct_b = [False] * 10 + [True] * 20
    result = m.mcnemar_test(correct_a, correct_b)
    assert result.exact is False
    # Hand-computed continuity-corrected chi-square statistic:
    #   (|b-c|-1)^2 / (b+c) = (|10-20|-1)^2/30 = 81/30 = 2.7
    assert result.statistic == pytest.approx(2.7, abs=1e-9)
    # p-value cross-checked via the exact chi2(1)-CDF identity, computed
    # independently in this test (not by calling the module's internals):
    #   P(chi2_1 > x) = erfc(sqrt(x/2))
    expected_p = math.erfc(math.sqrt(2.7 / 2.0))
    assert result.p_value == pytest.approx(expected_p, abs=1e-12)
    assert result.p_value == pytest.approx(0.1003, abs=2e-4)


def test_mcnemar_explicit_exact_false_small_n_hand_computed():
    # b=6, c=16 (disc=22, would auto-select exact) -- force the chi-square
    # branch to hand-verify a second worked chi-square case.
    correct_a = [True] * 6 + [False] * 16
    correct_b = [False] * 6 + [True] * 16
    result = m.mcnemar_test(correct_a, correct_b, exact=False)
    # (|6-16|-1)^2/22 = 81/22 = 3.681818...
    assert result.statistic == pytest.approx(81 / 22, abs=1e-9)
    expected_p = math.erfc(math.sqrt((81 / 22) / 2.0))
    assert result.p_value == pytest.approx(expected_p, abs=1e-12)


def test_mcnemar_requires_paired_equal_length():
    with pytest.raises(ValueError):
        m.mcnemar_test([True, False], [True])


def test_mcnemar_exact_vs_chi2_agree_directionally_at_large_disc():
    # Cross-validation (different algorithms for the same underlying
    # question), not self-consistency: at a large discordant count the
    # asymptotic chi-square p-value and the exact binomial p-value must be
    # close (Central Limit Theorem), even though computed via unrelated
    # code paths (erfc-based chi2 vs. exact math.comb enumeration).
    b, c = 120, 160
    correct_a = [True] * b + [False] * c
    correct_b = [False] * b + [True] * c
    exact_result = m.mcnemar_test(correct_a, correct_b, exact=True)
    chi2_result = m.mcnemar_test(correct_a, correct_b, exact=False)
    assert exact_result.p_value == pytest.approx(chi2_result.p_value, abs=0.02)


# ---------------------------------------------------------------------------
# Sample size / MDE — verified against the StataCorp worked example
# (R-CTX-29/30) and the MDE table (R-CTX-32).
# ---------------------------------------------------------------------------


def test_sample_size_for_mde_matches_stata_worked_example():
    # R-CTX-29: p12=0.037, p21=0.125 (p_disc=0.162), p_diff=0.088,
    # documented required N=162 at 80% power, alpha=.05 two-sided.
    # R-CTX-30 independently reproduced this as n=161.8; this test
    # reproduces it a second, independent time via this module's own
    # sample_size_for_mde().
    n = m.sample_size_for_mde(p_disc=0.162, p_diff=0.088, power=0.8, alpha=0.05)
    assert n == pytest.approx(161.8, abs=0.3)
    assert math.ceil(n) == 162


def test_mde_closed_form_matches_published_table_r_ctx_32():
    # R-CTX-32's MDE table at n=250: p_disc=0.10->5.6pp, 0.20->7.9pp,
    # 0.30->9.7pp, 0.40->11.2pp. baseline_acc=0.5 makes
    # p_disc = 2*0.5*0.5*(1-rho) = 0.5*(1-rho), so rho = 1 - 2*p_disc
    # hits each grid point exactly.
    cases = [
        (0.10, 1 - 2 * 0.10, 5.6),
        (0.20, 1 - 2 * 0.20, 7.9),
        (0.30, 1 - 2 * 0.30, 9.7),
        (0.40, 1 - 2 * 0.40, 11.2),
    ]
    for p_disc, correlation, expected_pp in cases:
        result = m.mde(n=250, baseline_acc=0.5, correlation=correlation, power=0.8, alpha=0.05)
        assert result.p_disc == pytest.approx(p_disc, abs=1e-9)
        assert result.mde_pp * 100 == pytest.approx(expected_pp, abs=0.05)


def test_mde_zero_correlation_matches_independence_baseline():
    # R-CTX-31: at rho=0, p_disc = p_A(1-p_A) + p_B(1-p_B) = 2p(1-p) when
    # both systems share baseline_acc under H0 -- verified here directly
    # from the derivation, independent of the mde() call itself.
    baseline = 0.6
    expected_p_disc = 2 * baseline * (1 - baseline)
    result = m.mde(n=500, baseline_acc=baseline, correlation=0.0)
    assert result.p_disc == pytest.approx(expected_p_disc, abs=1e-9)


def test_mde_exact_matches_closed_form_within_one_percent():
    # R-CTX-32: "The closed-form approximation ... matches the exact
    # bisection solution to within 1% at every grid point tested."
    # Cross-validation of two different algorithms, not self-consistency.
    for n in (250, 500, 1000):
        for correlation in (0.2, 0.5, 0.8):
            closed = m.mde(n=n, baseline_acc=0.6, correlation=correlation, method="closed_form")
            exact = m.mde(n=n, baseline_acc=0.6, correlation=correlation, method="exact")
            rel_err = abs(closed.mde_pp - exact.mde_pp) / closed.mde_pp
            assert rel_err < 0.01, f"n={n} rho={correlation}: {closed.mde_pp} vs {exact.mde_pp}"


def test_mde_rejects_bad_inputs():
    with pytest.raises(ValueError):
        m.mde(n=0, baseline_acc=0.5, correlation=0.0)
    with pytest.raises(ValueError):
        m.mde(n=100, baseline_acc=1.5, correlation=0.0)
    with pytest.raises(ValueError):
        m.mde(n=100, baseline_acc=0.5, correlation=1.5)


# ---------------------------------------------------------------------------
# Bootstrap CI / paired bootstrap test.
# ---------------------------------------------------------------------------


def test_bootstrap_ci_degenerate_constant_array():
    # A constant array's bootstrap distribution is a point mass -- the one
    # bootstrap case with an exact, hand-derivable expected CI: [1, 1].
    result = m.bootstrap_ci([1.0] * 50, seed=0)
    assert result.point_estimate == 1.0
    assert result.lo == pytest.approx(1.0, abs=1e-9)
    assert result.hi == pytest.approx(1.0, abs=1e-9)


def test_bootstrap_ci_known_proportion_reasonable_bounds():
    # values = 50 zeros + 50 ones (mean=0.5). Analytic SE of a proportion
    # at n=100 is sqrt(0.5*0.5/100)=0.05, so a 95% CI is roughly
    # 0.5 +/- 1.96*0.05 ~= [0.402, 0.598] under the normal approximation.
    # The percentile bootstrap need not match this exactly (different
    # method), but must land in the same ballpark -- a property check, not
    # an exact hand-computation (bootstrap has no closed form by
    # construction, which is why it exists).
    values = [0.0] * 50 + [1.0] * 50
    result = m.bootstrap_ci(values, n_resamples=5000, seed=1)
    assert result.point_estimate == 0.5
    assert 0.30 < result.lo < 0.45
    assert 0.55 < result.hi < 0.70


def test_paired_bootstrap_identical_arrays_gives_null_result():
    # a and b identical element-for-element: every resample (same indices
    # applied to both arrays) yields a difference of exactly 0, always.
    # Hand-derivable: observed_diff=0, lo=hi=0, p_value=1.0.
    a = [0.1, 0.4, 0.9, 0.3, 0.7]
    result = m.paired_bootstrap_test(a, a, n_resamples=200, seed=2)
    assert result.observed_diff == 0.0
    assert result.lo == pytest.approx(0.0, abs=1e-9)
    assert result.hi == pytest.approx(0.0, abs=1e-9)
    assert result.p_value == 1.0


def test_paired_bootstrap_constant_shift_gives_floor_p_value():
    # a is constant 1s, b is constant 0s: every resample difference is
    # exactly 1 (constant arrays -> resampling changes nothing), so the
    # bootstrap distribution is a point mass at 1. Centered at 0, every
    # centered value is exactly 0, which is never >= |observed_diff|=1,
    # so the raw p-value is exactly 0 -- the module floors this at
    # 1/n_resamples (never report an exact-zero p-value from a finite
    # resample count). Hand-derivable exact expected p-value: 1/500.
    a = [1.0] * 20
    b = [0.0] * 20
    result = m.paired_bootstrap_test(a, b, n_resamples=500, seed=3)
    assert result.observed_diff == 1.0
    assert result.p_value == pytest.approx(1 / 500, abs=1e-12)


def test_bootstrap_requires_paired_equal_length():
    with pytest.raises(ValueError):
        m.paired_bootstrap_test([1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        m.bootstrap_ci([])


# ---------------------------------------------------------------------------
# Multiple comparisons — Holm-Bonferroni and Benjamini-Hochberg, hand-worked.
# ---------------------------------------------------------------------------


def test_holm_bonferroni_hand_worked_example():
    # p = [0.01, 0.04, 0.03, 0.005], alpha=0.05, m=4.
    # Sorted ascending: 0.005(idx3), 0.01(idx0), 0.03(idx2), 0.04(idx1).
    #  rank0 (idx3): factor=4, adj=min(1,4*0.005)=0.02;  thresh=.05/4=.0125;  0.005<=.0125 -> reject
    #  rank1 (idx0): factor=3, adj=min(1,3*0.01)=0.03,  running_max=0.03;   thresh=.05/3=.01667; 0.01<=.01667 -> reject
    #  rank2 (idx2): factor=2, adj=min(1,2*0.03)=0.06,  running_max=0.06;   thresh=.05/2=.025;    0.03<=.025?  NO -> stop rejecting
    #  rank3 (idx1): factor=1, adj=min(1,1*0.04)=0.04,  running_max=max(0.06,0.04)=0.06; not reject (already stopped)
    # Mapped back to original order [idx0,idx1,idx2,idx3]:
    #   adjusted = [0.03, 0.06, 0.06, 0.02]
    #   reject   = [True, False, False, True]
    p_values = [0.01, 0.04, 0.03, 0.005]
    result = m.holm_bonferroni(p_values, alpha=0.05)
    assert result.adjusted_p == pytest.approx([0.03, 0.06, 0.06, 0.02], abs=1e-9)
    assert result.reject == [True, False, False, True]


def test_benjamini_hochberg_hand_worked_example():
    # Same p-values, BH step-up:
    # sorted ascending (rank k=1..4): 0.005, 0.01, 0.03, 0.04
    #  k=1: 0.005 <= .05*1/4=.0125  -> True
    #  k=2: 0.01  <= .05*2/4=.025   -> True
    #  k=3: 0.03  <= .05*3/4=.0375  -> True
    #  k=4: 0.04  <= .05*4/4=.05    -> True
    # max_k=4 -> ALL rejected (unlike Holm above -- BH is more powerful).
    # q-values (running min from the top): q4=0.04*4/4=0.04;
    #   q3=min(0.04, 0.03*4/3=0.04)=0.04; q2=min(0.04,0.01*4/2=0.02)=0.02;
    #   q1=min(0.02,0.005*4/1=0.02)=0.02
    # sorted q = [0.02(idx3), 0.02(idx0), 0.04(idx2), 0.04(idx1)]
    # mapped back to original order [idx0,idx1,idx2,idx3]:
    #   adjusted = [0.02, 0.04, 0.04, 0.02]; reject = [True]*4
    p_values = [0.01, 0.04, 0.03, 0.005]
    result = m.benjamini_hochberg(p_values, alpha=0.05)
    assert result.adjusted_p == pytest.approx([0.02, 0.04, 0.04, 0.02], abs=1e-9)
    assert result.reject == [True, True, True, True]


def test_bh_rejects_at_least_as_much_as_holm_on_same_data():
    # Structural property (not a numeric coincidence): BH controls FDR,
    # Holm controls FWER, and Holm's rejection region is a subset of BH's
    # on the same p-values/alpha -- true by construction of both
    # procedures. A stronger check than "the code agrees with itself."
    rng = np.random.default_rng(7)
    p_values = list(rng.uniform(0.0, 0.2, size=15))
    holm = m.holm_bonferroni(p_values, alpha=0.05)
    bh = m.benjamini_hochberg(p_values, alpha=0.05)
    for h, b in zip(holm.reject, bh.reject):
        assert (not h) or b  # holm reject implies bh reject


# ---------------------------------------------------------------------------
# Retrieval metrics — recall@k / nDCG@k, hand-built tiny cases.
# ---------------------------------------------------------------------------


def test_recall_at_k_turn_level_hand_computed():
    evidence = {"admissions": ["adm-1"], "turn_ids": [5, 9, 12]}
    retrieved = [
        _Item(session_id="adm-1", turn_ids=[5, 20]),
        _Item(session_id="adm-1", turn_ids=[9]),
        _Item(session_id="adm-2", turn_ids=[12]),  # wrong admission -> doesn't count
    ]
    # found gold turns = {5, 9} out of {5, 9, 12} -> 2/3
    assert m.recall_at_k(retrieved, evidence, k=3) == pytest.approx(2 / 3, abs=1e-9)


def test_recall_at_k_admission_level_hand_computed():
    evidence = {"admissions": ["adm-1", "adm-2", "adm-3"]}  # no turn_ids -> ~50% case
    retrieved = [
        _Item(session_id="adm-1", turn_ids=[1]),
        _Item(session_id="adm-4", turn_ids=[2]),
    ]
    # found = {adm-1} out of {adm-1, adm-2, adm-3} -> 1/3
    assert m.recall_at_k(retrieved, evidence, k=2) == pytest.approx(1 / 3, abs=1e-9)


def test_recall_at_k_respects_k_cutoff():
    evidence = {"admissions": ["adm-1"], "turn_ids": [5, 9]}
    retrieved = [
        _Item(session_id="adm-1", turn_ids=[9]),  # rank 1
        _Item(session_id="adm-1", turn_ids=[5]),  # rank 2
    ]
    # k=1 only sees turn 9 -> 1/2; k=2 sees both -> 2/2
    assert m.recall_at_k(retrieved, evidence, k=1) == pytest.approx(0.5, abs=1e-9)
    assert m.recall_at_k(retrieved, evidence, k=2) == pytest.approx(1.0, abs=1e-9)


def test_recall_at_k_finds_gold_admission_in_top5_e7s2_ac1():
    # E7-S2 AC1: fixture item whose gold admission is H1, ranked list whose
    # top-5 session_ids include H1 -> recall=1.0, NDCG in (0, 1].
    evidence = {"admissions": ["H1"]}
    retrieved = [
        _Item(session_id="H9", turn_ids=[1]),
        _Item(session_id="H1", turn_ids=[2]),
        _Item(session_id="H2", turn_ids=[3]),
        _Item(session_id="H3", turn_ids=[4]),
        _Item(session_id="H4", turn_ids=[5]),
    ]
    assert m.recall_at_k(retrieved, evidence, k=5) == pytest.approx(1.0, abs=1e-9)
    ndcg = m.ndcg_at_k(retrieved, evidence, k=5)
    assert 0.0 < ndcg <= 1.0


def test_recall_at_k_raises_on_ungroundable_evidence():
    with pytest.raises(ValueError):
        m.recall_at_k([_Item(session_id="adm-1", turn_ids=[1])], {"admissions": []}, k=1)
    with pytest.raises(ValueError):
        m.recall_at_k([_Item(session_id="adm-1", turn_ids=[1])], {}, k=1)


def test_ndcg_at_k_hand_computed():
    # gold turn_ids={5,9} (n_gold=2), admissions=["adm-1"].
    # Retrieved (rank order): pos1 not relevant, pos2 relevant, pos3 relevant.
    evidence = {"admissions": ["adm-1"], "turn_ids": [5, 9]}
    retrieved = [
        _Item(session_id="adm-1", turn_ids=[1]),  # rel=0
        _Item(session_id="adm-1", turn_ids=[9]),  # rel=1
        _Item(session_id="adm-1", turn_ids=[5]),  # rel=1
    ]
    # DCG = 0/log2(2) + 1/log2(3) + 1/log2(4) = 0 + 0.6309298 + 0.5 = 1.1309298
    # IDCG (ideal_count=min(3,2)=2) = 1/log2(2) + 1/log2(3) = 1 + 0.6309298 = 1.6309298
    # nDCG = 1.1309298 / 1.6309298 = 0.6934264
    dcg = 0 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    expected = dcg / idcg
    assert expected == pytest.approx(0.6934264, abs=1e-6)
    assert m.ndcg_at_k(retrieved, evidence, k=3) == pytest.approx(expected, abs=1e-9)


def test_ndcg_at_k_perfect_ranking_is_one():
    evidence = {"admissions": ["adm-1"], "turn_ids": [5, 9]}
    retrieved = [
        _Item(session_id="adm-1", turn_ids=[5]),
        _Item(session_id="adm-1", turn_ids=[9]),
    ]
    assert m.ndcg_at_k(retrieved, evidence, k=2) == pytest.approx(1.0, abs=1e-9)


def test_ndcg_at_k_no_relevant_items_is_zero():
    evidence = {"admissions": ["adm-1"], "turn_ids": [5, 9]}
    retrieved = [_Item(session_id="adm-9", turn_ids=[1])]
    assert m.ndcg_at_k(retrieved, evidence, k=1) == pytest.approx(0.0, abs=1e-9)


def test_ndcg_at_k_never_exceeds_one_when_many_admission_turns_are_relevant():
    evidence = {"admissions": ["adm-1"]}
    retrieved = [_Item(session_id="adm-1", turn_ids=[i]) for i in range(1, 11)]
    score = m.ndcg_at_k(retrieved, evidence, k=10)
    assert score == pytest.approx(1.0, abs=1e-9)


def test_ndcg_at_k_admission_partial_list_stays_in_unit_interval():
    evidence = {"admissions": ["adm-1"]}
    retrieved = [
        _Item(session_id="adm-9", turn_ids=[1]),
        _Item(session_id="adm-1", turn_ids=[2]),
        _Item(session_id="adm-1", turn_ids=[3]),
    ]
    score = m.ndcg_at_k(retrieved, evidence, k=3)
    dcg = 0 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert score == pytest.approx(dcg / idcg, abs=1e-9)
    assert 0.0 < score <= 1.0


def test_recall_and_ndcg_no_turn_ids_key_e7s2_ac2():
    # E7-S2 AC2: fixture item with no turn_ids key -> admission-level
    # Recall/NDCG are computed (not zero-filled, not raising) using the
    # admission grain; there is no separate "turn-level column" object
    # returned by this module (a single evidence-grounded value per call),
    # so "turn-level columns are None" is verified at the GoldEvidence
    # parse layer directly: turn_ids parses to None, never to an empty/
    # zero-filled tuple that would be silently scored as "found nothing".
    evidence = {"admissions": ["adm-1", "adm-2"]}
    gold = m._parse_gold_evidence(evidence)
    assert gold.turn_ids is None
    retrieved = [_Item(session_id="adm-1", turn_ids=[1])]
    assert m.recall_at_k(retrieved, evidence, k=1) == pytest.approx(0.5, abs=1e-9)
    ndcg = m.ndcg_at_k(retrieved, evidence, k=1)
    assert 0.0 < ndcg <= 1.0


# ---------------------------------------------------------------------------
# hit_at_k -- distinct from recall_at_k (fraction found vs any-found),
# hand-computed cases (retrieval_eval.py's dispatching story).
# ---------------------------------------------------------------------------


def test_hit_at_k_one_of_several_gold_admissions_found_is_a_hit():
    # 4 gold admissions, only 1 found in top-k -> recall=0.25 but hit=1.0.
    # This is the exact conflation the story warns against: a system that
    # merely "found some evidence" must not be scored as "found most of it".
    evidence = {"admissions": ["adm-1", "adm-2", "adm-3", "adm-4"]}
    retrieved = [_Item(session_id="adm-1", turn_ids=[1]), _Item(session_id="adm-9", turn_ids=[2])]
    assert m.recall_at_k(retrieved, evidence, k=2) == pytest.approx(0.25, abs=1e-9)
    assert m.hit_at_k(retrieved, evidence, k=2) == pytest.approx(1.0, abs=1e-9)


def test_hit_at_k_no_gold_found_is_zero():
    evidence = {"admissions": ["adm-1"], "turn_ids": [5, 9]}
    retrieved = [_Item(session_id="adm-2", turn_ids=[5]), _Item(session_id="adm-1", turn_ids=[1])]
    # adm-2/turn5 -> wrong admission, doesn't count; adm-1/turn1 -> right
    # admission, wrong turn -> no gold turn matched anywhere in top-k.
    assert m.hit_at_k(retrieved, evidence, k=2) == pytest.approx(0.0, abs=1e-9)


def test_hit_at_k_respects_k_cutoff():
    evidence = {"admissions": ["adm-1"], "turn_ids": [9]}
    retrieved = [
        _Item(session_id="adm-2", turn_ids=[1]),  # rank 1: not gold
        _Item(session_id="adm-1", turn_ids=[9]),  # rank 2: gold turn
    ]
    assert m.hit_at_k(retrieved, evidence, k=1) == pytest.approx(0.0, abs=1e-9)
    assert m.hit_at_k(retrieved, evidence, k=2) == pytest.approx(1.0, abs=1e-9)


def test_hit_at_k_turn_level_requires_the_right_admission_too():
    # The gold turn id 9 appears in the retrieved list, but attached to the
    # wrong admission -- turn ids are only meaningful within their own
    # admission, so this must not count as a hit.
    evidence = {"admissions": ["adm-1"], "turn_ids": [9]}
    retrieved = [_Item(session_id="adm-9", turn_ids=[9])]
    assert m.hit_at_k(retrieved, evidence, k=1) == pytest.approx(0.0, abs=1e-9)


def test_hit_at_k_raises_on_ungroundable_evidence():
    with pytest.raises(ValueError):
        m.hit_at_k([_Item(session_id="adm-1", turn_ids=[1])], {"admissions": []}, k=1)


# ---------------------------------------------------------------------------
# Abstention precision/recall/F1 -- explicit confusion matrix, hand-worked.
# ---------------------------------------------------------------------------


def test_abstention_prf1_hand_worked_confusion_matrix():
    # 10 items. abstained = model's decision; unanswerable = gold label.
    #  i1: T,T -> TP   i2: T,F -> FP   i3: T,T -> TP   i4: F,T -> FN
    #  i5: F,F -> TN   i6: F,F -> TN   i7: F,F -> TN   i8: F,T -> FN
    #  i9: T,F -> FP   i10: F,F -> TN
    # TP=2, FP=2, FN=2, TN=4 (sums to 10).
    abstained = [True, True, True, False, False, False, False, False, True, False]
    unanswerable = [True, False, True, True, False, False, False, True, False, False]
    result = m.abstention_precision_recall_f1(abstained, unanswerable)
    assert (result.tp, result.fp, result.fn, result.tn) == (2, 2, 2, 4)
    assert result.precision == pytest.approx(0.5, abs=1e-9)  # 2/(2+2)
    assert result.recall == pytest.approx(0.5, abs=1e-9)  # 2/(2+2)
    assert result.f1 == pytest.approx(0.5, abs=1e-9)


def test_abstention_prf1_precision_undefined_with_no_abstentions():
    abstained = [False, False, False]
    unanswerable = [True, False, True]
    result = m.abstention_precision_recall_f1(abstained, unanswerable)
    assert result.tp == 0 and result.fn == 2
    assert result.precision is None  # no abstentions -> undefined, not 0.0
    assert result.recall == pytest.approx(0.0, abs=1e-9)  # 0/(0+2)
    assert result.f1 is None


def test_abstention_prf1_perfect_score():
    abstained = [True, False, True, False]
    unanswerable = [True, False, True, False]
    result = m.abstention_precision_recall_f1(abstained, unanswerable)
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.f1 == 1.0


# ---------------------------------------------------------------------------
# Risk-coverage curve + AURC -- hand-computed trapezoid integration.
# ---------------------------------------------------------------------------


def test_risk_coverage_curve_and_aurc_hand_computed():
    # 4 items, already confidence-sorted: [.9,.7,.5,.3], correct=[T,T,F,T]
    #  rank1: correct_so_far=1, coverage=.25, risk=1-1/1=0
    #  rank2: correct_so_far=2, coverage=.50, risk=1-2/2=0
    #  rank3: correct_so_far=2, coverage=.75, risk=1-2/3=1/3
    #  rank4: correct_so_far=3, coverage=1.0, risk=1-3/4=.25
    confidence = [0.9, 0.7, 0.5, 0.3]
    correct = [True, True, False, True]
    points = m.risk_coverage_curve(confidence, correct)
    expected = [(0.25, 0.0), (0.5, 0.0), (0.75, 1 / 3), (1.0, 0.25)]
    for p, (cov, risk) in zip(points, expected):
        assert p.coverage == pytest.approx(cov, abs=1e-9)
        assert p.risk == pytest.approx(risk, abs=1e-9)

    # AURC via trapezoid from (0, risk[0]) through all points, hand-computed:
    #  seg1: (.25-0)*(0+0)/2 = 0
    #  seg2: (.5-.25)*(0+0)/2 = 0
    #  seg3: (.75-.5)*(0+1/3)/2 = .25*(1/3)/2 = 0.0416667
    #  seg4: (1-.75)*(1/3+.25)/2 = .25*.5833333/2 = 0.0729167
    #  total = 0.1145833
    assert m.aurc(points) == pytest.approx(0.1145833, abs=1e-6)


def test_risk_coverage_curve_all_correct_gives_zero_aurc():
    points = m.risk_coverage_curve([0.9, 0.5, 0.1], [True, True, True])
    assert m.aurc(points) == pytest.approx(0.0, abs=1e-9)


def test_risk_coverage_requires_equal_length():
    with pytest.raises(ValueError):
        m.risk_coverage_curve([0.5, 0.5], [True])


# ---------------------------------------------------------------------------
# Retrieval-vs-generation cross-tab -- hand-counted.
# ---------------------------------------------------------------------------


def test_retrieval_vs_generation_hand_computed():
    # 10 items. retrieved (gold evidence found in top-k) vs correct.
    #  positions 1-5 retrieved=True, 6-10 retrieved=False
    #  correct = [T,T,F,F,F, T,F,F,F,F]
    #  rc (retrieved&correct): pos1,2 -> 2
    #  ri (retrieved&incorrect): pos3,4,5 -> 3
    #  nc (not-retrieved&correct): pos6 -> 1
    #  ni (not-retrieved&incorrect): pos7,8,9,10 -> 4
    retrieved = [True] * 5 + [False] * 5
    correct = [True, True, False, False, False, True, False, False, False, False]
    result = m.retrieval_vs_generation(retrieved, correct)
    assert result.retrieved_and_correct == 2
    assert result.retrieved_and_incorrect == 3
    assert result.not_retrieved_and_correct == 1
    assert result.not_retrieved_and_incorrect == 4
    assert result.retrieval_coverage == pytest.approx(0.5, abs=1e-9)
    # generation_failure_rate = ri/(rc+ri) = 3/5 = 0.6
    assert result.generation_failure_rate == pytest.approx(0.6, abs=1e-9)
    # retrieval_failure_share = ni/(ri+ni) = 4/7
    assert result.retrieval_failure_share == pytest.approx(4 / 7, abs=1e-9)


def test_retrieval_vs_generation_all_retrieved_generation_failure_undefined_retrieval_share():
    # If nothing is ever incorrect, retrieval_failure_share is undefined
    # (0/0), not silently 0.0.
    retrieved = [True, True]
    correct = [True, True]
    result = m.retrieval_vs_generation(retrieved, correct)
    assert result.retrieval_failure_share is None


def test_retrieval_vs_generation_requires_equal_length():
    with pytest.raises(ValueError):
        m.retrieval_vs_generation([True], [True, False])


# ---------------------------------------------------------------------------
# E7-S2/E7-S3 literal-contract adapters -- hand-worked, cross-checked
# against the canonical functions above (see eval/metrics.py's module
# docstring "Story-contract reconciliation" note).
# ---------------------------------------------------------------------------


def test_mcnemar_pvalue_ac1_float_in_unit_interval_matches_mcnemar_test():
    # E7-S3 AC1 verbatim: two paired boolean lists differing on 10 of 100
    # items -> mcnemar_pvalue returns a float in [0, 1], and (per the
    # module's own docstring) is never an unpaired two-sample test --
    # verified here by cross-checking against mcnemar_test's own p_value,
    # not by asserting the wrapper agrees with itself.
    correct_a = [True] * 100
    correct_b = [True] * 90 + [False] * 10  # disagree on exactly 10 items
    p = m.mcnemar_pvalue(correct_a, correct_b)
    assert isinstance(p, float)
    assert 0.0 <= p <= 1.0
    assert p == m.mcnemar_test(correct_a, correct_b).p_value


def test_wilson_interval_e7s3_ac2_k60_n100_contains_point_six():
    # E7-S3 AC2 verbatim: k=60, n=100 -> the interval contains 0.6 and is
    # not a Wald p+/-1.96*sigma interval that goes negative near 0 (the
    # near-0/near-1 non-negativity property is covered exhaustively by
    # test_wilson_interval_near_boundary_hand_computed above; this is the
    # story's own exact k/n pair, called positionally per its contract).
    result = m.wilson_interval(60, 100)
    assert result.lo <= 0.6 <= result.hi
    assert result.lo >= 0.0


def test_abstention_prf_ac3_hand_worked_confusion_matrix():
    # E7-S3 AC3 verbatim: pred abstain [T,T,F,F], gold [T,F,T,F] -> P=0.5,
    # R=0.5, F1=0.5.
    #  i1: T,T -> TP   i2: T,F -> FP   i3: F,T -> FN   i4: F,F -> TN
    # TP=1, FP=1, FN=1, TN=1 -> P=1/2=0.5, R=1/2=0.5, F1=0.5.
    result = m.abstention_prf([True, True, False, False], [True, False, True, False])
    assert result == {"p": 0.5, "r": 0.5, "f1": 0.5}


def test_abstention_prf_propagates_none_not_zero():
    result = m.abstention_prf([False, False], [True, False])
    assert result["p"] is None  # no abstentions -> undefined, not 0.0
    assert result["f1"] is None


def test_paired_bootstrap_constant_delta_fn_hand_derived():
    # delta_fn ignores its argument and always returns 3.0 -> every
    # resample delta is exactly 3.0 regardless of which items are drawn,
    # so lo == hi == 3.0 exactly (hand-derivable point-mass case, same
    # logic as bootstrap_ci's degenerate-constant-array test above).
    paired_items = [(1, 4), (2, 5), (3, 6)]
    lo, hi = m.paired_bootstrap(lambda items: 3.0, paired_items, n_resample=200, seed=5)
    assert lo == pytest.approx(3.0, abs=1e-9)
    assert hi == pytest.approx(3.0, abs=1e-9)


def test_paired_bootstrap_constant_pairwise_difference_hand_derived():
    # Every (a, b) pair has a - b == 2 exactly, so the mean-difference
    # delta_fn evaluates to 2.0 on every possible resample (with or
    # without replacement, whichever items are drawn) -- hand-derivable
    # exact CI: [2.0, 2.0].
    paired_items = [(10, 8), (20, 18), (30, 28), (40, 38)]

    def delta_fn(items):
        return sum(a - b for a, b in items) / len(items)

    lo, hi = m.paired_bootstrap(delta_fn, paired_items, n_resample=300, seed=6)
    assert lo == pytest.approx(2.0, abs=1e-9)
    assert hi == pytest.approx(2.0, abs=1e-9)


def test_paired_bootstrap_requires_nonempty():
    with pytest.raises(ValueError):
        m.paired_bootstrap(lambda items: 0.0, [])


def test_token_f1_hand_computed_partial_overlap():
    # pred="the cat sat on the mat" (6 tokens, "the" repeated twice)
    # gold="the cat sat on a mat"   (6 tokens)
    # multiset intersection: the=min(2,1)=1, cat=1, sat=1, on=1, mat=1 (a
    # is not in pred) -> n_common=5.
    # precision = 5/6, recall = 5/6, f1 = 2*(5/6)*(5/6)/((5/6)+(5/6)) = 5/6
    pred = "the cat sat on the mat"
    gold = "the cat sat on a mat"
    assert m.token_f1(pred, gold) == pytest.approx(5 / 6, abs=1e-9)


def test_token_f1_identical_strings_is_one():
    text = "metformin 500mg twice daily"
    assert m.token_f1(text, text) == pytest.approx(1.0, abs=1e-9)


def test_token_f1_no_overlap_is_zero():
    assert m.token_f1("penicillin allergy", "no known allergies") == pytest.approx(0.0, abs=1e-9)


def test_token_f1_both_empty_is_one_squad_convention():
    assert m.token_f1("", "") == 1.0
    assert m.token_f1("   ", "") == 1.0  # whitespace-only splits to no tokens


def test_token_f1_exactly_one_empty_is_zero():
    assert m.token_f1("metformin", "") == 0.0
    assert m.token_f1("", "metformin") == 0.0


def test_token_f1_repeated_tokens_use_multiset_not_set():
    # pred repeats "aspirin" 3x, gold has it once -> multiset intersection
    # caps at min(3,1)=1, not 3 (a naive set-based F1 would wrongly treat
    # all 3 predicted repeats as matches: precision=3/3=1 instead of 1/3).
    # common=1, precision=1/3, recall=1/1=1, f1=2*(1/3)*1/((1/3)+1)=0.5
    assert m.token_f1("aspirin aspirin aspirin", "aspirin") == pytest.approx(0.5, abs=1e-9)
