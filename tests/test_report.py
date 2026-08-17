"""tests/test_report.py — coverage for eval/report.py: the table renders
from a stub result set (never the real MedLoCoMo corpus, never HydraDB,
never an Anthropic API call — matching every other `tests/test_*.py` in
this repo), the honest-loss summary line fires exactly when it should,
underpowered categories are marked, and blended accuracy is impossible to
emit at the API level (E7-S3 honesty guards / collaborative/inbox/
004-to-claude.md).

`medmemgraph.eval.metrics` (E7-S3) is imported optimistically by
`report.py`. Tests whose *value* depends on a specific metrics.py function
(Wilson CI, McNemar, abstention_prf via metrics) are skipped, not faked,
when that function isn't importable — see `_needs_metrics`. Every other
test here (table shape, blended-accuracy refusal, underpowered marking,
summary-line branching, Pareto marking, truncation/dry-run caveats)
exercises report.py's own logic and does not depend on metrics.py at all.
"""

from __future__ import annotations

import dataclasses
import math

import pytest

from medmemgraph.eval import report

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_needs_metrics = pytest.mark.skipif(
    report._metrics is None, reason="medmemgraph.eval.metrics not importable in this checkout"
)


def _item(
    qa_id: str,
    question_type: str,
    scope: str,
    correct: bool,
    *,
    tokens: int = 100,
    latency_ms: float = 1.0,
    truncated: bool = False,
    pred_absent: bool | None = None,
) -> dict:
    return {
        "qa_id": qa_id,
        "question_type": question_type,
        "scope": scope,
        "correct": correct,
        "tokens": tokens,
        "latency_ms": latency_ms,
        "truncated": truncated,
        "pred_absent": pred_absent,
    }


def _six_category_items(*, prefix: str, correct_fn) -> list[dict]:
    """One stub item per (question_type, scope-appropriate) pairing,
    covering all six MedLoCoMo `question_type`s and both `scope`s, so a
    single system's `SystemResult` exercises the full per-category
    breakout in one call."""
    scopes = {
        "medical_reasoning": "single_admission",
        "care_plan_rationale": "single_admission",
        "longitudinal_progression": "cross_admission",
        "cross_admission_comparison": "cross_admission",
        "frequency_pattern": "cross_admission",
        "adversarial": "cross_admission",
    }
    items = []
    i = 0
    for qt, scope in scopes.items():
        for _ in range(10):
            i += 1
            qa_id = f"{prefix}-{qt}-{i}"
            items.append(_item(qa_id, qt, scope, correct_fn(qt, i)))
    return items


# ---------------------------------------------------------------------------
# 1. Blended accuracy is impossible to emit (E7-S3 honesty guard #1)
# ---------------------------------------------------------------------------


class TestBlendedAccuracyIsImpossibleToEmit:
    @pytest.mark.parametrize("bad_mode", ["blended", "overall", "all", "combined", ""])
    def test_accuracy_for_refuses_any_mode_but_the_two_slices(self, bad_mode):
        with pytest.raises(ValueError, match="blended|answerable|abstention"):
            report.accuracy_for(5, 10, mode=bad_mode)

    def test_accuracy_for_accepts_only_the_two_named_slices(self):
        assert report.accuracy_for(5, 10, mode="answerable") == pytest.approx(0.5)
        assert report.accuracy_for(3, 4, mode="abstention") == pytest.approx(0.75)

    def test_system_result_has_no_bare_or_blended_accuracy_field(self):
        fieldnames = {f.name for f in dataclasses.fields(report.SystemResult)}
        assert "accuracy" not in fieldnames
        assert "overall_accuracy" not in fieldnames
        assert "blended_accuracy" not in fieldnames
        # the only two accuracy-bearing properties are explicitly qualified
        assert hasattr(report.SystemResult, "answerable_accuracy")
        assert hasattr(report.SystemResult, "abstention_accuracy")

    def test_category_stat_has_no_bare_accuracy_field_either(self):
        fieldnames = {f.name for f in dataclasses.fields(report.CategoryStat)}
        assert "accuracy" not in fieldnames  # it's a computed property, gated by `mode`
        assert hasattr(report.CategoryStat, "accuracy")

    def test_rendered_report_never_emits_a_blended_accuracy_value(self):
        sys_a = report.build_system_result("fullctx", _six_category_items(prefix="a", correct_fn=lambda qt, i: i % 2 == 0))
        sys_b = report.build_system_result("dense", _six_category_items(prefix="b", correct_fn=lambda qt, i: i % 3 == 0))
        rep = report.build_report([sys_a, sys_b])
        text = report.render_terminal(rep) + report.render_markdown(rep)
        # "blended" may only appear in the module's own reassuring
        # declaration that a slice is *not* blended (e.g. the by-scope
        # section header) — never as a labelled "blended accuracy" value.
        assert "blended accuracy" not in text.lower()
        for line in text.lower().splitlines():
            if "blended" in line:
                assert "never blended" in line

    def test_master_table_accuracy_columns_are_always_qualified(self):
        sys_a = report.build_system_result("fullctx", _six_category_items(prefix="a", correct_fn=lambda qt, i: i % 2 == 0))
        rep = report.build_report([sys_a])
        header = report.render_master_table(rep).splitlines()[0]
        # every "accuracy"-bearing column name names its slice explicitly
        for token in header.split():
            if "acc" in token.lower():
                assert "answerable" in token.lower() or "abstain" in token.lower()

    def test_scope_breakout_never_pools_answerable_and_abstention_into_one_stat(self):
        # a scope bucket that mixes answerable and adversarial items must
        # produce TWO CategoryStats for that scope (one per mode), not one
        # pooled n/k pair.
        items = [
            _item("q1", "medical_reasoning", "cross_admission", True),
            _item("q2", "medical_reasoning", "cross_admission", False),
            _item("q3", "adversarial", "cross_admission", True),
        ]
        sys_result = report.build_system_result("fullctx", items)
        cross_stats = [s for s in sys_result.by_scope if s.category.startswith("cross_admission")]
        assert len(cross_stats) == 2
        modes = {s.mode for s in cross_stats}
        assert modes == {"answerable", "abstention"}
        answerable_stat = next(s for s in cross_stats if s.mode == "answerable")
        abstention_stat = next(s for s in cross_stats if s.mode == "abstention")
        assert answerable_stat.n == 2
        assert abstention_stat.n == 1
        # each CategoryStat's own accuracy is still computed through the
        # guarded accuracy_for — restricted to its own mode.
        assert answerable_stat.accuracy == pytest.approx(0.5)
        assert abstention_stat.accuracy == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 2. Table renders from a stub result set
# ---------------------------------------------------------------------------


class TestMasterTableRendersFromAStub:
    def _two_system_report(self):
        fullctx = report.build_system_result(
            "fullctx",
            _six_category_items(prefix="f", correct_fn=lambda qt, i: i <= 6),
            patient_ids=["STUB01"],
            judge_kind="token-overlap",
        )
        dense = report.build_system_result(
            "dense",
            _six_category_items(prefix="d", correct_fn=lambda qt, i: i <= 7),
            patient_ids=["STUB01"],
            judge_kind="token-overlap",
        )
        return report.build_report([fullctx, dense], run_config={"patient_id": "STUB01"})

    def test_one_row_per_system_in_master_table(self):
        rep = self._two_system_report()
        table = report.render_master_table(rep)
        assert "fullctx" in table
        assert "dense" in table
        # one data row per system beyond the header/separator lines
        lines = [l for l in table.splitlines() if l.strip()]
        assert len(lines) == 2 + len(rep.systems)

    def test_master_table_has_the_required_columns(self):
        rep = self._two_system_report()
        header = report.render_master_table(rep).splitlines()[0]
        for col in ("answerable_acc", "abstain_acc", "abstain P/R/F1", "recall@5", "recall@10", "ndcg@10", "mean_tok", "p50_ms", "p95_ms", "truncated"):
            assert col in header

    def test_per_category_breakout_covers_all_six_question_types(self):
        rep = self._two_system_report()
        fullctx = next(s for s in rep.systems if s.system_name == "fullctx")
        categories = [s.category for s in fullctx.by_question_type]
        assert categories == list(report.QUESTION_TYPES)

    def test_per_category_breakout_covers_both_scopes(self):
        rep = self._two_system_report()
        fullctx = next(s for s in rep.systems if s.system_name == "fullctx")
        scope_categories = {s.category.split(" — ")[0] for s in fullctx.by_scope}
        assert scope_categories == set(report.SCOPES)

    def test_render_terminal_and_render_markdown_do_not_raise(self):
        rep = self._two_system_report()
        terminal_text = report.render_terminal(rep)
        markdown_text = report.render_markdown(rep)
        assert "fullctx" in terminal_text
        assert "# medmemgraph eval report" in markdown_text

    def test_render_markdown_is_self_contained_and_stamped(self):
        rep = self._two_system_report()
        text = report.render_markdown(rep)
        assert rep.generated_at in text
        assert "run_config" in text
        assert "STUB01" in text
        assert "## Summary" in text
        assert "## Master table" in text
        assert "## Claim (frozen)" in text
        assert report.PARETO_SENTENCE in text

    def test_write_markdown_writes_a_file_under_the_given_dir(self, tmp_path):
        rep = self._two_system_report()
        path = report.write_markdown(rep, results_dir=tmp_path)
        assert path.exists()
        assert path.read_text() == report.render_markdown(rep)
        assert path.parent == tmp_path


# ---------------------------------------------------------------------------
# 3. The "full-context wins" summary line fires when it should
# ---------------------------------------------------------------------------


class TestSummaryLine:
    def _report_with_answerable_accuracies(self, *, fullctx_acc_of_100: int, other_acc_of_100: int):
        fullctx_items = [_item(f"q{i}", "medical_reasoning", "single_admission", i < fullctx_acc_of_100) for i in range(100)]
        other_items = [_item(f"q{i}", "medical_reasoning", "single_admission", i < other_acc_of_100) for i in range(100)]
        fullctx = report.build_system_result("fullctx", fullctx_items)
        other = report.build_system_result("dense", other_items)
        return report.build_report([fullctx, other])

    def test_fires_honest_loss_when_fullctx_has_the_highest_accuracy(self):
        rep = self._report_with_answerable_accuracies(fullctx_acc_of_100=80, other_acc_of_100=40)
        line = report.summary_line(rep)
        assert line.startswith("HONEST LOSS")
        assert report.PARETO_SENTENCE in line

    def test_fires_honest_loss_on_an_exact_tie_too(self):
        # tie must still count as "full-context is not beaten" — no memory
        # system may claim victory on an exact tie either.
        rep = self._report_with_answerable_accuracies(fullctx_acc_of_100=60, other_acc_of_100=60)
        line = report.summary_line(rep)
        assert line.startswith("HONEST LOSS")

    def test_does_not_fire_honest_loss_when_a_memory_system_scores_higher(self):
        rep = self._report_with_answerable_accuracies(fullctx_acc_of_100=40, other_acc_of_100=80)
        line = report.summary_line(rep)
        assert not line.startswith("HONEST LOSS")
        assert "dense" in line
        assert report.PARETO_SENTENCE in line

    def test_never_contains_a_banned_phrase_in_either_branch(self):
        for fullctx_acc, other_acc in [(80, 40), (40, 80), (50, 50)]:
            rep = self._report_with_answerable_accuracies(fullctx_acc_of_100=fullctx_acc, other_acc_of_100=other_acc)
            line = report.summary_line(rep).lower()
            for phrase in report._BANNED_PHRASES:
                assert phrase not in line

    def test_handles_missing_baseline_without_crashing(self):
        other = report.build_system_result("dense", [_item("q1", "medical_reasoning", "single_admission", True)])
        rep = report.build_report([other], baseline_name="fullctx")
        line = report.summary_line(rep)
        assert report.PARETO_SENTENCE in line

    def test_assert_no_banned_phrase_raises_on_injected_banned_text(self):
        with pytest.raises(AssertionError):
            report._assert_no_banned_phrase("Our system beats full-context on every category.")

    def test_assert_no_banned_phrase_accepts_the_frozen_pareto_sentence(self):
        report._assert_no_banned_phrase(report.PARETO_SENTENCE)  # must not raise


# ---------------------------------------------------------------------------
# 4. Underpowered rows are marked
# ---------------------------------------------------------------------------


class TestUnderpoweredMarking:
    def test_small_n_category_is_marked_underpowered(self):
        items = [_item(f"q{i}", "medical_reasoning", "single_admission", i % 2 == 0) for i in range(5)]
        sys_result = report.build_system_result("fullctx", items)
        stat = sys_result.by_question_type[0]
        assert stat.n == 5
        assert stat.underpowered is True

    def test_large_n_category_at_default_target_is_not_marked_underpowered(self):
        items = [_item(f"q{i}", "medical_reasoning", "single_admission", i % 2 == 0) for i in range(1000)]
        sys_result = report.build_system_result("fullctx", items)
        stat = sys_result.by_question_type[0]
        assert stat.n == 1000
        assert stat.underpowered is False

    def test_empty_category_is_underpowered_not_a_crash(self):
        items = [_item("q1", "medical_reasoning", "single_admission", True)]
        sys_result = report.build_system_result("fullctx", items)
        adversarial_stat = sys_result.abstention
        assert adversarial_stat.n == 0
        assert adversarial_stat.accuracy is None
        assert adversarial_stat.underpowered is True

    def test_minimum_detectable_effect_matches_hand_derivation_at_n30(self):
        # independent recomputation (not calling the function under test to
        # check itself) — standard two-sided z_(0.975)/z_(0.80) normal
        # quantiles, conservative p=0.5 variance.
        n = 30
        z_alpha, z_beta = 1.959963985, 0.8416212336
        se = math.sqrt(2 * 0.5 * 0.5 / n)
        expected = (z_alpha + z_beta) * se
        assert report.minimum_detectable_effect(n) == pytest.approx(expected, rel=1e-9)
        # sanity bound restated from the module docstring: ~0.36 at n=30.
        assert 0.35 < report.minimum_detectable_effect(30) < 0.37

    def test_minimum_detectable_effect_shrinks_as_n_grows(self):
        assert report.minimum_detectable_effect(1000) < report.minimum_detectable_effect(100) < report.minimum_detectable_effect(10)

    def test_minimum_detectable_effect_of_zero_or_negative_n_is_none(self):
        assert report.minimum_detectable_effect(0) is None
        assert report.minimum_detectable_effect(-5) is None


# ---------------------------------------------------------------------------
# 5. Pareto view
# ---------------------------------------------------------------------------


class TestParetoView:
    def test_a_cheaper_equally_accurate_system_dominates_full_context(self):
        fullctx_items = [_item(f"q{i}", "medical_reasoning", "single_admission", i < 50, tokens=70000, latency_ms=9000) for i in range(100)]
        dense_items = [_item(f"q{i}", "medical_reasoning", "single_admission", i < 55, tokens=2000, latency_ms=400) for i in range(100)]
        fullctx = report.build_system_result("fullctx", fullctx_items)
        dense = report.build_system_result("dense", dense_items)
        rep = report.build_report([fullctx, dense])
        assert rep.pareto_dominated["fullctx"] is True
        assert rep.pareto_dominated["dense"] is False

    def test_system_with_unknown_accuracy_is_not_silently_dropped(self):
        fullctx = report.build_system_result("fullctx", [_item("q1", "medical_reasoning", "single_admission", True)])
        empty = report.build_system_result("dense", [])  # n_items == 0, answerable_accuracy is None
        rep = report.build_report([fullctx, empty])
        assert rep.pareto_dominated["dense"] is False  # "not assessable", not "dominated"
        table = report.render_pareto(rep)
        assert "dense" in table

    def test_ascii_scatter_plots_every_system_with_a_known_accuracy(self):
        fullctx = report.build_system_result("fullctx", [_item("q1", "medical_reasoning", "single_admission", True, tokens=70000)])
        dense = report.build_system_result("dense", [_item("q1", "medical_reasoning", "single_admission", False, tokens=2000)])
        scatter = report.ascii_scatter([fullctx, dense])
        assert "fullctx" in scatter
        assert "dense" in scatter


# ---------------------------------------------------------------------------
# 6. Truncation / dry-run / provenance caveats
# ---------------------------------------------------------------------------


class TestCaveats:
    def test_truncation_is_recorded_and_surfaced(self):
        items = [_item("q1", "medical_reasoning", "single_admission", True, truncated=True)]
        sys_result = report.build_system_result("fullctx", items)
        assert sys_result.truncated is True
        assert sys_result.n_truncated == 1
        rep = report.build_report([sys_result])
        assert "TRUNCATED" in report.render_terminal(rep)

    def test_no_truncation_is_also_explicit_not_silent(self):
        items = [_item("q1", "medical_reasoning", "single_admission", True, truncated=False)]
        sys_result = report.build_system_result("fullctx", items)
        assert sys_result.truncated is False
        rep = report.build_report([sys_result])
        table = report.render_master_table(rep)
        assert "no" in table  # the truncated column reads "no", not blank

    def test_dry_run_is_surfaced_as_a_caveat(self):
        items = [_item("q1", "medical_reasoning", "single_admission", True)]
        sys_result = report.build_system_result("fullctx", items, dry_run=True)
        rep = report.build_report([sys_result])
        text = report.render_terminal(rep)
        assert "DRY-RUN" in text

    def test_missing_pred_absent_signal_produces_a_caveat_not_a_crash(self):
        items = [_item(f"q{i}", "adversarial", "cross_admission", True) for i in range(5)]  # no pred_absent given
        sys_result = report.build_system_result("fullctx", items)
        assert sys_result.abstention_prf is None
        assert sys_result.abstention_prf_caveat is not None
        rep = report.build_report([sys_result])
        assert "n/a" in report.render_master_table(rep)


# ---------------------------------------------------------------------------
# 7. Real results/*.json (harness.py write_results) shape adapter
# ---------------------------------------------------------------------------


class TestLoadFromHarnessRunDict:
    def _run_dict(self, system_name: str, *, correct_flags: list[bool]) -> dict:
        records = []
        for i, correct in enumerate(correct_flags):
            qt = "adversarial" if i % 5 == 4 else "medical_reasoning"
            scope = "cross_admission" if i % 2 == 0 else "single_admission"
            records.append(
                {
                    "qa_id": f"{system_name}-q{i}",
                    "question_type": qt,
                    "scope": scope,
                    "correct": correct,
                    "mode": "abstention" if qt == "adversarial" else "answerable",
                    "judge_kind": "token-overlap",
                    "judge_reason": "stub",
                    "tokens": 1000,
                    "latency_ms": 5.0,
                    "truncated": False,
                }
            )
        return {
            "patient_id": "STUB01",
            "system_name": system_name,
            "dry_run": True,
            "judge_kind": "token-overlap",
            "n_items": len(records),
            "records": records,
            "n_truncated": 0,
        }

    def test_adapts_a_harness_run_dict_into_a_system_result(self):
        d = self._run_dict("fullctx", correct_flags=[True, False, True, False, True] * 4)
        sys_result = report.load_system_result_from_run_dict(d)
        assert sys_result.system_name == "fullctx"
        assert sys_result.n_items == 20
        assert sys_result.patient_ids == ("STUB01",)
        assert sys_result.dry_run is True

    def test_load_results_dir_reads_every_matching_file(self, tmp_path):
        import json

        for name, flags in [("fullctx", [True, False] * 10), ("dense", [True] * 20)]:
            d = self._run_dict(name, correct_flags=flags)
            (tmp_path / f"STUB01__{name}.json").write_text(json.dumps(d))
        (tmp_path / "OTHERPATIENT__fullctx.json").write_text(json.dumps(self._run_dict("fullctx", correct_flags=[True])))

        systems = report.load_results_dir(tmp_path, "STUB01")
        names = sorted(s.system_name for s in systems)
        assert names == ["dense", "fullctx"]

    def test_build_report_for_patient_end_to_end(self, tmp_path):
        import json

        for name, flags in [("fullctx", [True, False] * 10), ("dense", [True] * 20)]:
            d = self._run_dict(name, correct_flags=flags)
            (tmp_path / f"STUB01__{name}.json").write_text(json.dumps(d))

        rep = report.build_report_for_patient("STUB01", results_dir=tmp_path)
        text = report.render_terminal(rep)
        assert "fullctx" in text and "dense" in text


# ---------------------------------------------------------------------------
# 8. metrics.py-dependent numbers — skipped, not faked, when unavailable
# ---------------------------------------------------------------------------


class TestMetricsOptionalDegradation:
    def test_metrics_available_is_false_for_a_nonexistent_function(self):
        assert report.metrics_available("this_function_does_not_exist") is False

    def test_wilson_ci_for_is_none_when_metrics_unavailable(self, monkeypatch):
        monkeypatch.setattr(report, "_metrics", None)
        assert report.wilson_ci_for(60, 100) is None

    def test_abstention_prf_for_is_none_when_metrics_unavailable(self, monkeypatch):
        monkeypatch.setattr(report, "_metrics", None)
        assert report.abstention_prf_for([True, True, False, False], [True, False, True, False]) is None

    def test_paired_significance_degrades_to_not_computed_without_metrics(self, monkeypatch):
        monkeypatch.setattr(report, "_metrics", None)
        fullctx = report.build_system_result("fullctx", [_item(f"q{i}", "medical_reasoning", "single_admission", i < 50) for i in range(100)])
        dense = report.build_system_result("dense", [_item(f"q{i}", "medical_reasoning", "single_admission", i < 60) for i in range(100)])
        sig = report.paired_significance(fullctx, [dense])
        assert sig["dense"]["p"] is None
        assert "not available" in sig["dense"]["note"] or "not computed" in sig["dense"]["note"]

    def test_build_report_never_raises_when_metrics_is_unavailable(self, monkeypatch):
        monkeypatch.setattr(report, "_metrics", None)
        fullctx = report.build_system_result("fullctx", _six_category_items(prefix="f", correct_fn=lambda qt, i: i <= 5))
        dense = report.build_system_result("dense", _six_category_items(prefix="d", correct_fn=lambda qt, i: i <= 6))
        rep = report.build_report([fullctx, dense])
        text = report.render_terminal(rep)
        assert "n/a" in text
        assert "not available" in text or "unavailable" in text


@_needs_metrics
class TestMetricsBackedNumbers:
    """These exercise the real medmemgraph.eval.metrics functions when the
    module is importable in this checkout — never a second implementation
    of wilson_interval/mcnemar_pvalue/abstention_prf here."""

    def test_wilson_ci_contains_the_point_estimate_and_is_not_a_negative_wald_interval(self):
        ci = report.wilson_ci_for(60, 100)
        assert ci is not None
        lo, hi = ci
        assert lo <= 0.6 <= hi
        # near-zero proportion: a Wald interval would go negative here.
        lo0, hi0 = report.wilson_ci_for(0, 20)
        assert lo0 >= 0.0

    def test_abstention_prf_for_matches_the_e7s3_worked_example(self):
        # E7-S3 AC3: pred abstain [T,T,F,F], gold [T,F,T,F] -> P=R=F1=0.5
        prf = report.abstention_prf_for([True, True, False, False], [True, False, True, False])
        assert prf["p"] == pytest.approx(0.5)
        assert prf["r"] == pytest.approx(0.5)
        assert prf["f1"] == pytest.approx(0.5)

    def test_paired_significance_uses_a_paired_test_not_an_unpaired_one(self):
        # 100 items, systems differ on exactly 10 (5 each way) -> a real
        # paired McNemar p-value in [0, 1].
        fullctx_items = [_item(f"q{i}", "medical_reasoning", "single_admission", i >= 5 and i < 55) for i in range(100)]
        dense_items = [_item(f"q{i}", "medical_reasoning", "single_admission", i < 50) for i in range(100)]
        fullctx = report.build_system_result("fullctx", fullctx_items)
        dense = report.build_system_result("dense", dense_items)
        sig = report.paired_significance(fullctx, [dense])
        p = sig["dense"]["p"]
        assert p is not None
        assert 0.0 <= p <= 1.0
        assert sig["dense"]["n_paired"] == 100

    def test_holm_bonferroni_is_delegated_to_metrics_not_reimplemented(self):
        # report.py must not define its own holm_bonferroni — it uses
        # medmemgraph.eval.metrics.holm_bonferroni directly.
        assert not hasattr(report, "holm_bonferroni")
