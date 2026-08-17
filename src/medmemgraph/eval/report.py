"""eval/report.py — the one table judges read: per-system accuracy/cost,
per-category breakout, paired significance, and a Pareto view, rendered to
the terminal and to a self-contained markdown file under `results/`.

This module is downstream of `eval/harness.py` (per-item records) and
`eval/metrics.py` (the statistical primitives: `mcnemar_pvalue`,
`wilson_interval`, `paired_bootstrap`, `abstention_prf`, `token_f1` —
E7-S3). `metrics.py` may not exist yet in a given checkout (it is being
written concurrently); this module imports it optimistically
(`_metrics`, set to `None` on `ImportError`) and degrades every
metrics-dependent number to an explicit "not computed" rather than
duplicating a second implementation of any of those five functions. What
never degrades is the table *shape* and the honesty guards below — those
do not need `metrics.py` to exist.

Honesty guards, implemented as code, not comments (collaborative/design/
stories/E7/E7-S3.md, collaborative/inbox/004-to-claude.md):

1. Never emit one blended accuracy across answerable + abstention items.
   `accuracy_for()` is the single place a proportion is computed in this
   module, and it *requires* `mode` to be exactly `"answerable"` or
   `"abstention"` — there is no third mode, and `SystemResult` has no
   bare `accuracy` field for a caller to reach for instead. Enforced by
   `test_report.py::test_blended_accuracy_is_impossible_to_emit`.
2. A category whose sample size cannot detect a `target_mde`-sized effect
   (paired two-proportion, alpha=0.05, power=0.80, normal approximation,
   conservative p=0.5 variance) is marked `underpowered=True` instead of
   silently reported as a normal comparison.
3. Full-context truncation is a first-class, printed field
   (`SystemResult.truncated` / `n_truncated`) — never silently dropped.
4. The Pareto claim sentence (`PARETO_SENTENCE`) is embedded verbatim in
   every rendered summary, and `_assert_no_banned_phrase` is run on every
   rendered report as a regression guard against ever emitting "beats
   full-context" / "outperforms full context" or an equivalent.
5. `summary_line()` says so in plain language when full-context wins on
   aggregate (pooled) answerable accuracy — the "honest-loss" case is not
   hideable in a cell; see the two pre-written README paragraphs this
   line is meant to select between.
"""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Mapping, Sequence

from medmemgraph.eval.harness import ADVERSARIAL_TYPE, QUESTION_TYPES, SCOPES

try:
    from medmemgraph.eval import metrics as _metrics
except ImportError:  # metrics.py (E7-S3) not landed yet in this checkout —
    _metrics = None  # every metrics-dependent number below degrades to
    # an explicit "not computed" (see module docstring point 0). Do not
    # write a second implementation of mcnemar_pvalue / wilson_interval /
    # paired_bootstrap / abstention_prf / token_f1 here.


def metrics_available(*names: str) -> bool:
    """Whether `medmemgraph.eval.metrics` is importable and exposes every
    name in `names`."""
    return _metrics is not None and all(hasattr(_metrics, n) for n in names)


# ---------------------------------------------------------------------------
# Frozen framing (collaborative/inbox/004-to-claude.md, E7-S3 contract)
# ---------------------------------------------------------------------------

PARETO_SENTENCE = (
    "Comparable accuracy at a fraction of tokens and latency; wins "
    "concentrated on cross-admission, temporal update, and abstention. "
    "Not a raw accuracy win over full-context."
)

_BANNED_PHRASES = (
    "beats full-context",
    "beats full context",
    "outperforms full context",
    "outperforms full-context",
    "beats fullctx",
)

FULLCTX_SYSTEM = "fullctx"

AccuracyMode = Literal["answerable", "abstention"]

DEFAULT_TARGET_MDE = 0.10
"""Default minimum-effect-size target (10 accuracy points) a category must
be powered to detect before it is reported as a normal comparison rather
than `underpowered`. Deliberately strict: literature/02's own methodology
note treats "30 questions per category" as a field-convention floor, not
proof of adequate power — at n=30, p=0.5, alpha=0.05, power=0.80 the
achievable MDE is ~0.36 (36 points), far short of this default. Expect
most per-category rows on a single-patient MedLoCoMo run to be marked
underpowered; that is the honest reading of a ~30-150 item run, not a bug
in this module."""

_Z_ALPHA_TWO_SIDED_05 = 1.959963985
_Z_BETA_POWER_80 = 0.8416212336


def _assert_no_banned_phrase(text: str) -> None:
    lowered = text.lower()
    for phrase in _BANNED_PHRASES:
        if phrase in lowered:
            raise AssertionError(
                f"report text contains a banned phrase ({phrase!r}). The frozen claim "
                "is Pareto — comparable accuracy, fewer tokens/latency, wins on "
                "cross-admission/temporal/abstention — never a raw accuracy win over "
                "full-context (collaborative/inbox/004-to-claude.md, E7-S3 banned "
                "approaches)."
            )


# ---------------------------------------------------------------------------
# The one guarded accuracy function in this module.
# ---------------------------------------------------------------------------


def _fraction(k: int, n: int) -> float | None:
    if n == 0:
        return None
    return k / n


def accuracy_for(k: int, n: int, *, mode: AccuracyMode) -> float | None:
    """Compute accuracy for exactly one slice of items. `mode` must be
    `"answerable"` or `"abstention"` — there is no blended mode, and this
    is checked at call time rather than left to caller discipline. This
    is the honesty guard from the API side: the only function in this
    module that turns (k, n) into a proportion refuses to do so across a
    mix of the two rubrics."""
    if mode not in ("answerable", "abstention"):
        raise ValueError(
            f"accuracy_for(mode={mode!r}) refused: accuracy must be reported "
            "separately per 'answerable' or 'abstention' slice — a blended "
            "accuracy across both is not a supported mode of this function "
            "(E7-S3 honesty guard: never emit a single blended accuracy)."
        )
    return _fraction(k, n)


# ---------------------------------------------------------------------------
# Minimum detectable effect / underpowered marking
# ---------------------------------------------------------------------------


def minimum_detectable_effect(
    n: int, *, alpha: float = 0.05, power: float = 0.80, p_baseline: float = 0.5
) -> float | None:
    """Rough single-arm MDE at sample size `n` for a paired proportion
    comparison, using the normal approximation with the conservative
    (maximum-variance) baseline proportion `p_baseline=0.5`. This is a
    sizing heuristic printed alongside the table — it is not a substitute
    for the paired McNemar test itself, which is the actual significance
    test this report defers to `metrics.mcnemar_pvalue` for."""
    if n is None or n <= 0:
        return None
    if alpha == 0.05 and power == 0.80:
        z_alpha, z_beta = _Z_ALPHA_TWO_SIDED_05, _Z_BETA_POWER_80
    else:  # pragma: no cover - only the defaults are exercised in tests
        z_alpha = math.sqrt(2) * _erfinv(1 - alpha)
        z_beta = math.sqrt(2) * _erfinv(2 * power - 1)
    se = (2 * p_baseline * (1 - p_baseline) / n) ** 0.5
    return (z_alpha + z_beta) * se


def _erfinv(x: float) -> float:  # pragma: no cover - only used for non-default alpha/power
    # Minimal Winitzki approximation; adequate for a sizing heuristic, not
    # used anywhere the default alpha=0.05/power=0.80 path is taken.
    a = 0.147
    ln1mx2 = math.log(1 - x * x)
    t1 = 2 / (math.pi * a) + ln1mx2 / 2
    return math.copysign(math.sqrt(math.sqrt(t1 * t1 - ln1mx2 / a) - t1), x)


# ---------------------------------------------------------------------------
# Metrics-optional wrappers (never a second implementation)
# ---------------------------------------------------------------------------


def wilson_ci_for(k: int, n: int) -> tuple[float, float] | None:
    """`(lo, hi)` via `metrics.wilson_interval`, or `None` if `n == 0` or
    `metrics.py` is unavailable. `wilson_interval` returns a
    `WilsonInterval` dataclass (`.lo`/`.hi`), not a bare tuple — unpacked
    here so the rest of this module has one stable `tuple | None` shape to
    render regardless of which `metrics.py` revision is on disk."""
    if n == 0:
        return None
    if not metrics_available("wilson_interval"):
        return None
    interval = _metrics.wilson_interval(k, n)
    return (interval.lo, interval.hi)


def abstention_prf_for(pred_absent: Sequence[bool], gold_absent: Sequence[bool]) -> dict | None:
    if not pred_absent or not metrics_available("abstention_prf"):
        return None
    return _metrics.abstention_prf(list(pred_absent), list(gold_absent))


# ---------------------------------------------------------------------------
# Percentiles (small local utility — not part of the metrics.py contract)
# ---------------------------------------------------------------------------


def _percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(pct / 100.0 * (len(ordered) - 1))))
    return ordered[idx]


# ---------------------------------------------------------------------------
# Data shapes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CategoryStat:
    """One row of a per-category breakout, restricted to exactly one
    accuracy rubric (`mode`). A scope (`single_admission` /
    `cross_admission`) contains both answerable and adversarial items in
    MedLoCoMo, so a scope is reported as *two* `CategoryStat`s (one per
    mode) rather than one n/k pair that would blend the two rubrics —
    see `build_system_result`."""

    category: str
    mode: AccuracyMode
    n: int
    k: int
    mean_tokens: float
    p50_latency_ms: float
    p95_latency_ms: float
    mde: float | None = None
    underpowered: bool = False

    @property
    def accuracy(self) -> float | None:
        return accuracy_for(self.k, self.n, mode=self.mode)


@dataclass(frozen=True)
class SystemResult:
    """One system's full result set for the report. Deliberately has no
    bare `accuracy` field — only `answerable_accuracy` and
    `abstention_accuracy`, each derived from a mode-tagged `CategoryStat`
    via the guarded `accuracy_for`."""

    system_name: str
    patient_ids: tuple[str, ...]
    n_items: int
    judge_kind: str
    answerable: CategoryStat
    abstention: CategoryStat
    by_question_type: tuple[CategoryStat, ...]
    by_scope: tuple[CategoryStat, ...]
    mean_tokens: float
    p50_latency_ms: float
    p95_latency_ms: float
    n_truncated: int
    answerable_ci: tuple[float, float] | None
    abstention_prf: dict | None
    abstention_prf_caveat: str | None
    recall_at_5: float | None = None
    recall_at_10: float | None = None
    ndcg_at_10: float | None = None
    n_judge_runs: int = 1
    judge_temperature: float = 0.0
    accuracy_sd: float | None = None
    dry_run: bool | None = None
    item_correctness: dict = field(default_factory=dict)

    @property
    def answerable_accuracy(self) -> float | None:
        return self.answerable.accuracy

    @property
    def abstention_accuracy(self) -> float | None:
        return self.abstention.accuracy

    @property
    def truncated(self) -> bool:
        return self.n_truncated > 0

    @property
    def needs_repeated_runs_note(self) -> bool:
        """Story requirement: mean±SD over >=3 judge runs *if*
        temperature > 0. True when temperature > 0 but fewer than 3 runs
        were actually done — the point estimate should not be read as a
        headline in that case."""
        return self.judge_temperature > 0 and self.n_judge_runs < 3


@dataclass(frozen=True)
class Report:
    systems: tuple[SystemResult, ...]
    baseline_name: str
    generated_at: str
    run_config: dict
    significance: dict
    pareto_dominated: dict
    target_mde: float


# ---------------------------------------------------------------------------
# Building a SystemResult from raw per-item records
# ---------------------------------------------------------------------------


def build_system_result(
    system_name: str,
    items: Sequence[Mapping],
    *,
    patient_ids: Sequence[str] = (),
    judge_kind: str = "unknown",
    n_truncated: int | None = None,
    recall_at_5: float | None = None,
    recall_at_10: float | None = None,
    ndcg_at_10: float | None = None,
    n_judge_runs: int = 1,
    judge_temperature: float = 0.0,
    accuracy_sd: float | None = None,
    dry_run: bool | None = None,
    target_mde: float = DEFAULT_TARGET_MDE,
) -> SystemResult:
    """Aggregate `items` (one dict per QA item: `qa_id`, `question_type`,
    `scope`, `correct`, `tokens`, `latency_ms`, `truncated`, and optional
    `pred_absent`) into a `SystemResult`. This is the single place raw
    per-item data becomes a table row — both the test-plan's stub result
    sets and `load_system_result_from_run_dict` (real `results/*.json`)
    go through it, so there is exactly one aggregation implementation."""
    items = list(items)
    n_items = len(items)

    def _mk_stat(category: str, mode: AccuracyMode, subset: list[Mapping]) -> CategoryStat:
        n = len(subset)
        k = sum(1 for it in subset if it["correct"])
        tokens = [it.get("tokens", 0) for it in subset]
        lat = [it.get("latency_ms", 0.0) for it in subset]
        mde = minimum_detectable_effect(n)
        return CategoryStat(
            category=category,
            mode=mode,
            n=n,
            k=k,
            mean_tokens=(sum(tokens) / n) if n else 0.0,
            p50_latency_ms=_percentile(lat, 50),
            p95_latency_ms=_percentile(lat, 95),
            mde=mde,
            underpowered=(mde is None) or (mde > target_mde),
        )

    answerable_items = [it for it in items if it["question_type"] != ADVERSARIAL_TYPE]
    abstention_items = [it for it in items if it["question_type"] == ADVERSARIAL_TYPE]

    answerable = _mk_stat("ANSWERABLE (pooled)", "answerable", answerable_items)
    abstention = _mk_stat("ABSTENTION (adversarial)", "abstention", abstention_items)

    type_buckets: dict[str, list[Mapping]] = defaultdict(list)
    for it in items:
        type_buckets[it["question_type"]].append(it)
    type_order = [t for t in QUESTION_TYPES if t in type_buckets]
    type_order += sorted(t for t in type_buckets if t not in QUESTION_TYPES)
    by_question_type = [
        _mk_stat(t, "abstention" if t == ADVERSARIAL_TYPE else "answerable", type_buckets[t])
        for t in type_order
    ]

    scope_buckets: dict[str, list[Mapping]] = defaultdict(list)
    for it in items:
        scope_buckets[it["scope"]].append(it)
    scope_order = [s for s in SCOPES if s in scope_buckets]
    scope_order += sorted(s for s in scope_buckets if s not in SCOPES)
    by_scope: list[CategoryStat] = []
    for s in scope_order:
        subset = scope_buckets[s]
        ans_subset = [it for it in subset if it["question_type"] != ADVERSARIAL_TYPE]
        abs_subset = [it for it in subset if it["question_type"] == ADVERSARIAL_TYPE]
        by_scope.append(_mk_stat(f"{s} — answerable", "answerable", ans_subset))
        by_scope.append(_mk_stat(f"{s} — abstention", "abstention", abs_subset))

    tokens_all = [it.get("tokens", 0) for it in items]
    lat_all = [it.get("latency_ms", 0.0) for it in items]
    mean_tokens = (sum(tokens_all) / n_items) if n_items else 0.0
    p50 = _percentile(lat_all, 50)
    p95 = _percentile(lat_all, 95)

    if n_truncated is None:
        n_truncated = sum(1 for it in items if it.get("truncated", False))

    item_correctness = {it["qa_id"]: bool(it["correct"]) for it in items if "qa_id" in it}

    pred_absent_raw = [it.get("pred_absent") for it in items]
    usable_idx = [i for i, p in enumerate(pred_absent_raw) if p is not None]
    pred_absent = [bool(pred_absent_raw[i]) for i in usable_idx]
    gold_absent = [items[i]["question_type"] == ADVERSARIAL_TYPE for i in usable_idx]
    prf = abstention_prf_for(pred_absent, gold_absent) if usable_idx else None

    caveat_parts: list[str] = []
    if not usable_idx:
        caveat_parts.append(
            "abstention P/R/F1 not computed: no item in this run carries a per-item "
            "system-abstained (pred_absent) signal yet."
        )
    else:
        if len(usable_idx) < n_items:
            caveat_parts.append(
                f"abstention P/R/F1 computed on {len(usable_idx)}/{n_items} items carrying a "
                "known system-abstained (pred_absent) signal; the remaining items were excluded "
                "rather than guessed."
            )
        if prf is None:
            caveat_parts.append(
                "abstention P/R/F1 not computed: medmemgraph.eval.metrics.abstention_prf is not available."
            )
    caveat = " ".join(caveat_parts) or None

    return SystemResult(
        system_name=system_name,
        patient_ids=tuple(patient_ids),
        n_items=n_items,
        judge_kind=judge_kind,
        answerable=answerable,
        abstention=abstention,
        by_question_type=tuple(by_question_type),
        by_scope=tuple(by_scope),
        mean_tokens=mean_tokens,
        p50_latency_ms=p50,
        p95_latency_ms=p95,
        n_truncated=n_truncated,
        answerable_ci=wilson_ci_for(answerable.k, answerable.n),
        abstention_prf=prf,
        abstention_prf_caveat=caveat,
        recall_at_5=recall_at_5,
        recall_at_10=recall_at_10,
        ndcg_at_10=ndcg_at_10,
        n_judge_runs=n_judge_runs,
        judge_temperature=judge_temperature,
        accuracy_sd=accuracy_sd,
        dry_run=dry_run,
        item_correctness=item_correctness,
    )


# ---------------------------------------------------------------------------
# Adapters for real results/<patient>__<system>.json (harness.py's
# write_results output shape)
# ---------------------------------------------------------------------------


def load_system_result_from_run_dict(d: Mapping, *, target_mde: float = DEFAULT_TARGET_MDE) -> SystemResult:
    """Adapt one `HarnessRun.to_dict()` payload (`results/*.json`) into a
    `SystemResult`. `pred_absent` can only be recovered for adversarial
    items (`mode="abstention"`'s `correct` bit literally means "declined
    correctly" there, per judge.py's routing) — `harness.py`'s `Record`
    does not currently persist raw answer text for any other item, so
    `pred_absent` is left `None` (unknown, not guessed) for answerable
    items. See `SystemResult.abstention_prf_caveat`."""
    items = []
    for r in d.get("records", []):
        qt = r["question_type"]
        pred_absent = r["correct"] if qt == ADVERSARIAL_TYPE else None
        items.append(
            {
                "qa_id": r["qa_id"],
                "question_type": qt,
                "scope": r["scope"],
                "correct": r["correct"],
                "tokens": r.get("tokens", 0),
                "latency_ms": r.get("latency_ms", 0.0),
                "truncated": r.get("truncated", False),
                "pred_absent": pred_absent,
            }
        )
    patient_ids = (d["patient_id"],) if d.get("patient_id") else ()
    return build_system_result(
        d["system_name"],
        items,
        patient_ids=patient_ids,
        judge_kind=d.get("judge_kind", "unknown"),
        n_truncated=d.get("n_truncated"),
        dry_run=d.get("dry_run"),
        target_mde=target_mde,
    )


def load_results_dir(
    results_dir: str | os.PathLike[str], patient_id: str, *, system_names: Sequence[str] | None = None
) -> list[SystemResult]:
    """Load every `results/<patient_id>__<system>.json` file for
    `patient_id`, optionally restricted to `system_names`."""
    out = []
    for path in sorted(Path(results_dir).glob(f"{patient_id}__*.json")):
        d = json.loads(path.read_text())
        if system_names is not None and d.get("system_name") not in system_names:
            continue
        out.append(load_system_result_from_run_dict(d))
    return out


# ---------------------------------------------------------------------------
# Paired significance (McNemar + Holm-Bonferroni)
# ---------------------------------------------------------------------------


def _align_by_qa_id(a: SystemResult, b: SystemResult) -> tuple[list[bool], list[bool]] | None:
    common = sorted(set(a.item_correctness) & set(b.item_correctness))
    if not common:
        return None
    return [a.item_correctness[q] for q in common], [b.item_correctness[q] for q in common]


def _paired_mde(mcnemar_result, *, baseline_acc: float | None) -> float | None:
    """The actual MDE (`metrics.mde`) for this specific paired comparison
    at the n actually used, deriving the phi correlation between the two
    systems' per-item correctness straight from McNemar's own 2x2 counts
    (a=both correct, b/c=discordant, d=both incorrect) — no second
    correlation primitive needed. `None` when `baseline_acc` sits at the
    0/1 boundary `metrics.mde` refuses, or when metrics.py lacks `mde`."""
    if baseline_acc is None or not (0 < baseline_acc < 1):
        return None
    if not metrics_available("mde"):
        return None
    a, b, c, d = mcnemar_result.a, mcnemar_result.b, mcnemar_result.c, mcnemar_result.d
    denom = math.sqrt(max((a + b) * (c + d) * (a + c) * (b + d), 1))
    correlation = ((a * d - b * c) / denom) if denom else 0.0
    correlation = max(-1.0, min(1.0, correlation))
    try:
        return _metrics.mde(mcnemar_result.n, baseline_acc, correlation).mde_pp
    except ValueError:
        return None


def paired_significance(
    baseline: SystemResult | None, others: Sequence[SystemResult], *, alpha: float = 0.05
) -> dict[str, dict]:
    """For each system in `others`, paired McNemar against `baseline` on
    the qa_ids they share, then `metrics.holm_bonferroni`-adjust across the
    family (both imported, never re-implemented — see module docstring).
    Attaches the paired MDE actually achievable at the shared n
    (`_paired_mde`). Degrades every field to an explicit "not computed" —
    never a fabricated number — when `baseline` is `None`, items don't
    overlap, or `metrics.py` (or the specific function needed) is not
    importable yet."""
    if baseline is None:
        return {
            o.system_name: {"p": None, "note": "no full-context baseline present in this run"}
            for o in others
        }
    if not metrics_available("mcnemar_test", "holm_bonferroni"):
        return {
            o.system_name: {
                "p": None,
                "note": "not computed — medmemgraph.eval.metrics.mcnemar_test/holm_bonferroni not available yet",
            }
            for o in others
        }

    raw_p: dict[str, float] = {}
    mde_pp: dict[str, float | None] = {}
    n_paired: dict[str, int] = {}
    notes: dict[str, str] = {}
    for o in others:
        pair = _align_by_qa_id(baseline, o)
        if pair is None:
            notes[o.system_name] = "no overlapping qa_ids with baseline; cannot pair"
            continue
        a_correct, b_correct = pair
        result = _metrics.mcnemar_test(a_correct, b_correct)
        raw_p[o.system_name] = result.p_value
        n_paired[o.system_name] = result.n
        mde_pp[o.system_name] = _paired_mde(result, baseline_acc=baseline.answerable_accuracy)

    names = list(raw_p)
    if names:
        adjusted = _metrics.holm_bonferroni([raw_p[name] for name in names], alpha=alpha)
    out: dict[str, dict] = {}
    for o in others:
        if o.system_name not in raw_p:
            out[o.system_name] = {"p": None, "note": notes.get(o.system_name, "not computed")}
            continue
        i = names.index(o.system_name)
        out[o.system_name] = {
            "p": raw_p[o.system_name],
            "adjusted_p": adjusted.adjusted_p[i],
            "reject": adjusted.reject[i],
            "n_paired": n_paired[o.system_name],
            "mde_pp": mde_pp[o.system_name],
            "note": "",
        }
    return out


# ---------------------------------------------------------------------------
# Pareto view
# ---------------------------------------------------------------------------


def pareto_frontier(systems: Sequence[SystemResult]) -> dict[str, bool]:
    """`{system_name: is_dominated}` on the (answerable accuracy, mean
    tokens, p50 latency) axes — explicitly *answerable* accuracy, never a
    blended number (honesty guard). A system with an unknown answerable
    accuracy cannot be assessed and is reported as not dominated (`False`)
    rather than silently dropped."""
    dominated: dict[str, bool] = {s.system_name: False for s in systems}
    for a in systems:
        if a.answerable_accuracy is None:
            continue
        for b in systems:
            if a is b or b.answerable_accuracy is None:
                continue
            not_worse = (
                b.answerable_accuracy >= a.answerable_accuracy
                and b.mean_tokens <= a.mean_tokens
                and b.p50_latency_ms <= a.p50_latency_ms
            )
            strictly_better = (
                b.answerable_accuracy > a.answerable_accuracy
                or b.mean_tokens < a.mean_tokens
                or b.p50_latency_ms < a.p50_latency_ms
            )
            if not_worse and strictly_better:
                dominated[a.system_name] = True
                break
    return dominated


def ascii_scatter(systems: Sequence[SystemResult], *, width: int = 50, height: int = 12) -> str:
    """Text scatter of answerable accuracy (y) vs mean tokens/query (x,
    log-scaled), precedented by LME-V2 Fig. 6. No plotting dependency is
    added for this — the parallel-columns table is the field norm anyway
    (survey 13) and this scatter is a secondary view of the same numbers."""
    pts = [(s.system_name, s.mean_tokens, s.answerable_accuracy) for s in systems if s.answerable_accuracy is not None]
    if not pts:
        return "(no system has a known answerable accuracy to plot)"
    xs = [math.log10(max(t, 1.0)) for _, t, _ in pts]
    ys = [acc for _, _, acc in pts]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(0.0, min(ys)), max(1.0, max(ys))
    if x_max == x_min:
        x_max = x_min + 1.0
    grid = [[" "] * width for _ in range(height)]
    labels = []
    for (name, tok, acc), x, y in zip(pts, xs, ys):
        col = int((x - x_min) / (x_max - x_min) * (width - 1))
        row = int((1 - (y - y_min) / (y_max - y_min)) * (height - 1))
        code = name[:1].upper()
        grid[row][col] = code
        labels.append(f"{code}={name} (tok={tok:.0f}, acc={acc:.3f})")
    lines = ["".join(row) for row in grid]
    lines.append("-" * width + f"  x=log10(mean tokens/query) [{10**x_min:.0f}..{10**x_max:.0f}], y=answerable accuracy [{y_min:.2f}..{y_max:.2f}]")
    lines.extend(labels)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


def build_report(
    systems: Sequence[SystemResult],
    *,
    baseline_name: str = FULLCTX_SYSTEM,
    run_config: Mapping | None = None,
    alpha: float = 0.05,
    target_mde: float = DEFAULT_TARGET_MDE,
) -> Report:
    systems = tuple(systems)
    by_name = {s.system_name: s for s in systems}
    baseline = by_name.get(baseline_name)
    others = [s for s in systems if s.system_name != baseline_name]
    significance = paired_significance(baseline, others, alpha=alpha)
    pareto_dominated = pareto_frontier(systems)
    return Report(
        systems=systems,
        baseline_name=baseline_name,
        generated_at=datetime.now(timezone.utc).isoformat(),
        run_config=dict(run_config or {}),
        significance=significance,
        pareto_dominated=pareto_dominated,
        target_mde=target_mde,
    )


def build_report_for_patient(
    patient_id: str,
    *,
    results_dir: str | os.PathLike[str] = "results",
    system_names: Sequence[str] | None = None,
    baseline_name: str = FULLCTX_SYSTEM,
    target_mde: float = DEFAULT_TARGET_MDE,
) -> Report:
    systems = load_results_dir(results_dir, patient_id, system_names=system_names)
    return build_report(
        systems,
        baseline_name=baseline_name,
        run_config={"patient_id": patient_id, "results_dir": str(results_dir)},
        target_mde=target_mde,
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    widths = [len(h) for h in headers]
    for r in rows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(c))

    def fmt_row(cells: Sequence[str]) -> str:
        return "  ".join(c.ljust(w) if i == 0 else c.rjust(w) for i, (c, w) in enumerate(zip(cells, widths)))

    lines = [fmt_row(headers), fmt_row(["-" * w for w in widths])]
    lines.extend(fmt_row(r) for r in rows)
    return "\n".join(lines)


def _fmt_num(x: float | None, spec: str = ".3f") -> str:
    return "n/a" if x is None else format(x, spec)


def _fmt_acc_cell(stat: CategoryStat, ci: tuple[float, float] | None = None) -> str:
    acc = stat.accuracy
    if acc is None:
        return "n/a"
    s = f"{acc:.3f}"
    if ci is not None:
        s += f" [{ci[0]:.3f},{ci[1]:.3f}]"
    if stat.underpowered:
        s += " (underpowered)"
    return s


def render_master_table(report: Report) -> str:
    headers = [
        "system",
        "n",
        "answerable_acc (95% CI)",
        "abstain_acc",
        "abstain P/R/F1",
        "recall@5",
        "recall@10",
        "ndcg@10",
        "mean_tok",
        "p50_ms",
        "p95_ms",
        "truncated",
        "dry_run",
    ]
    rows = []
    for s in report.systems:
        prf = s.abstention_prf
        prf_cell = f"{_fmt_num(prf['p'], '.2f')}/{_fmt_num(prf['r'], '.2f')}/{_fmt_num(prf['f1'], '.2f')}" if prf else "n/a"
        rows.append(
            [
                s.system_name,
                str(s.n_items),
                _fmt_acc_cell(s.answerable, s.answerable_ci),
                _fmt_acc_cell(s.abstention),
                prf_cell,
                _fmt_num(s.recall_at_5),
                _fmt_num(s.recall_at_10),
                _fmt_num(s.ndcg_at_10),
                f"{s.mean_tokens:.1f}",
                f"{s.p50_latency_ms:.2f}",
                f"{s.p95_latency_ms:.2f}",
                f"{s.n_truncated}/{s.n_items}" if s.truncated else "no",
                "yes" if s.dry_run else ("no" if s.dry_run is not None else "n/a"),
            ]
        )
    return _render_table(headers, rows)


def render_category_breakout(title: str, systems: Sequence[SystemResult], selector) -> str:
    headers = ["system", "category", "mode", "n", "accuracy", "mde", "underpowered"]
    rows = []
    for s in systems:
        for stat in selector(s):
            rows.append(
                [
                    s.system_name,
                    stat.category,
                    stat.mode,
                    str(stat.n),
                    _fmt_num(stat.accuracy),
                    _fmt_num(stat.mde),
                    "yes" if stat.underpowered else "no",
                ]
            )
    return f"-- {title} --\n" + _render_table(headers, rows)


def render_significance(report: Report) -> str:
    headers = ["system", "vs", "n_paired", "p (raw)", "p (Holm-adj)", "reject@alpha", "MDE@n", "note"]
    rows = []
    for name, res in report.significance.items():
        rows.append(
            [
                name,
                report.baseline_name,
                str(res.get("n_paired", "")) if res.get("n_paired") is not None else "n/a",
                _fmt_num(res.get("p"), ".4f"),
                _fmt_num(res.get("adjusted_p"), ".4f"),
                "yes" if res.get("reject") else "no",
                _fmt_num(res.get("mde_pp")),
                res.get("note", ""),
            ]
        )
    body = _render_table(headers, rows) if rows else "(no non-baseline systems in this run)"
    note = (
        "MDE@n is metrics.mde's paired minimum-detectable-effect for that row's actual shared "
        "sample size and observed baseline-vs-system correlation (not a generic heuristic). "
        f"Category rows in the breakout above use a separate, simpler single-arm sizing "
        f"heuristic (target={report.target_mde:.2f}) to mark 'underpowered' — see "
        "minimum_detectable_effect()'s docstring for why the two are not the same number."
    )
    return f"-- paired significance vs {report.baseline_name} (Holm-Bonferroni-adjusted) --\n{body}\n{note}"


def render_pareto(report: Report) -> str:
    headers = ["system", "answerable_acc", "mean_tok", "p50_ms", "dominated"]
    rows = [
        [
            s.system_name,
            _fmt_num(s.answerable_accuracy),
            f"{s.mean_tokens:.1f}",
            f"{s.p50_latency_ms:.2f}",
            "yes" if report.pareto_dominated.get(s.system_name) else "no",
        ]
        for s in report.systems
    ]
    table = _render_table(headers, rows)
    scatter = ascii_scatter(report.systems)
    return f"-- Pareto view (parallel columns) --\n{table}\n\n-- Pareto view (scatter, LME-V2 Fig. 6-style) --\n{scatter}"


def summary_line(report: Report) -> str:
    """The honesty-guard headline. Never says "beats"/"outperforms"
    full-context (`_assert_no_banned_phrase` is run on the result as a
    regression guard); says so in plain language when full-context wins
    on pooled answerable accuracy."""
    baseline = next((s for s in report.systems if s.system_name == report.baseline_name), None)
    if baseline is None or baseline.answerable_accuracy is None:
        line = (
            f"No {report.baseline_name!r} baseline with a known answerable accuracy in this run "
            f"— cannot compute the full-context comparison. {PARETO_SENTENCE}"
        )
        _assert_no_banned_phrase(line)
        return line

    others = [s for s in report.systems if s.system_name != baseline.system_name and s.answerable_accuracy is not None]
    if not others:
        line = f"Only {baseline.system_name} ran with a known answerable accuracy in this run. {PARETO_SENTENCE}"
        _assert_no_banned_phrase(line)
        return line

    best = max(others, key=lambda s: s.answerable_accuracy)
    if baseline.answerable_accuracy >= best.answerable_accuracy:
        line = (
            f"HONEST LOSS ON AGGREGATE ACCURACY: {baseline.system_name} has the highest point-estimate "
            f"pooled answerable accuracy in this run ({baseline.answerable_accuracy:.3f}); no memory "
            f"system here exceeds it on answerable accuracy alone. {PARETO_SENTENCE}"
        )
    else:
        line = (
            f"{best.system_name} shows a higher point-estimate pooled answerable accuracy than "
            f"{baseline.system_name} in this run ({best.answerable_accuracy:.3f} vs "
            f"{baseline.answerable_accuracy:.3f}) — read as directional pending the paired "
            f"significance test, not as a settled result. {PARETO_SENTENCE}"
        )
    _assert_no_banned_phrase(line)
    return line


def _caveats(report: Report) -> list[str]:
    notes: list[str] = []
    for s in report.systems:
        if s.dry_run:
            notes.append(
                f"{s.system_name}: DRY-RUN — this system ran with a stub answerer and the "
                "deterministic token-overlap judge fallback (no live LLM calls). Accuracy "
                "numbers are not evidence of real system quality; only the plumbing (table "
                "shape, cost accounting) should be read from a dry-run row."
            )
        if s.truncated:
            notes.append(f"{s.system_name}: TRUNCATED — {s.n_truncated}/{s.n_items} items had history truncated.")
        if s.abstention_prf_caveat:
            notes.append(f"{s.system_name}: {s.abstention_prf_caveat}")
        if s.needs_repeated_runs_note:
            notes.append(
                f"{s.system_name}: judge temperature={s.judge_temperature} > 0 but only "
                f"{s.n_judge_runs} run(s) — point estimate, not a mean±SD headline "
                "(story requirement: ≥ 3 judge runs when temperature > 0)."
            )
    if not metrics_available("wilson_interval"):
        notes.append("medmemgraph.eval.metrics.wilson_interval unavailable — Wilson CIs shown as n/a.")
    if not metrics_available("mcnemar_test", "holm_bonferroni"):
        notes.append(
            "medmemgraph.eval.metrics.mcnemar_test/holm_bonferroni unavailable — paired "
            "significance not computed."
        )
    if not metrics_available("abstention_prf"):
        notes.append("medmemgraph.eval.metrics.abstention_prf unavailable — abstention P/R/F1 shown as n/a.")
    return notes


def render_terminal(report: Report) -> str:
    parts = [
        f"=== medmemgraph eval report — generated {report.generated_at} — baseline={report.baseline_name} ===",
        render_master_table(report),
        "",
        render_category_breakout("by question_type", report.systems, lambda s: s.by_question_type),
        "",
        render_category_breakout("by scope (split answerable / abstention — never blended)", report.systems, lambda s: s.by_scope),
        "",
        render_significance(report),
        "",
        render_pareto(report),
        "",
        "-- caveats --",
        "\n".join(f"- {c}" for c in _caveats(report)) or "(none)",
        "",
        "-- summary --",
        summary_line(report),
    ]
    text = "\n".join(parts)
    _assert_no_banned_phrase(text)
    return text


def render_markdown(report: Report) -> str:
    stamp = report.generated_at
    patients = sorted({p for s in report.systems for p in s.patient_ids})
    header = (
        f"# medmemgraph eval report\n\n"
        f"Generated `{stamp}` · baseline=`{report.baseline_name}` · "
        f"target_mde=`{report.target_mde:.2f}` · patients=`{', '.join(patients) or 'unknown'}` · "
        f"run_config=`{json.dumps(report.run_config, sort_keys=True)}`\n"
    )
    parts = [
        header,
        "## Summary\n",
        f"**{summary_line(report)}**\n",
        "## Master table (one row per system)\n",
        "```text\n" + render_master_table(report) + "\n```\n",
        "## Per-category breakout\n",
        "```text\n" + render_category_breakout("by question_type", report.systems, lambda s: s.by_question_type) + "\n```\n",
        "```text\n"
        + render_category_breakout(
            "by scope (split answerable / abstention — never blended)", report.systems, lambda s: s.by_scope
        )
        + "\n```\n",
        "## Paired significance\n",
        "```text\n" + render_significance(report) + "\n```\n",
        "## Pareto view\n",
        "```text\n" + render_pareto(report) + "\n```\n",
        "## Caveats\n",
        "\n".join(f"- {c}" for c in _caveats(report)) or "(none)",
        "\n## Claim (frozen)\n",
        f"> {PARETO_SENTENCE}\n",
    ]
    text = "\n".join(parts)
    _assert_no_banned_phrase(text)
    return text


def write_markdown(
    report: Report, *, results_dir: str | os.PathLike[str] = "results", filename: str | None = None
) -> Path:
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if filename is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        patients = "-".join(sorted({p for s in report.systems for p in s.patient_ids})) or "unknown-patient"
        filename = f"report__{patients}__{stamp}.md"
    path = out_dir / filename
    path.write_text(render_markdown(report))
    return path


__all__ = [
    "PARETO_SENTENCE",
    "FULLCTX_SYSTEM",
    "DEFAULT_TARGET_MDE",
    "metrics_available",
    "accuracy_for",
    "minimum_detectable_effect",
    "wilson_ci_for",
    "abstention_prf_for",
    "CategoryStat",
    "SystemResult",
    "Report",
    "build_system_result",
    "load_system_result_from_run_dict",
    "load_results_dir",
    "paired_significance",
    "pareto_frontier",
    "ascii_scatter",
    "build_report",
    "build_report_for_patient",
    "render_master_table",
    "render_category_breakout",
    "render_significance",
    "render_pareto",
    "summary_line",
    "render_terminal",
    "render_markdown",
    "write_markdown",
]
