"""eval/harness.py — runs a named baseline system over one patient's real
QA items and prints a per-category results table.

Runs entirely without HydraDB (ARCHITECTURE.md §3.2 / PHASES.md leftover
gate row "No-memory + full-context on real QA ... Runs against
benchmark_qa.json for at least one patient without touching HydraDB"). This
module never imports hydra_client.

Table shape (story requirement 3): rows = `question_type`
(medical_reasoning, care_plan_rationale, longitudinal_progression,
cross_admission_comparison, frequency_pattern, adversarial), columns = n,
accuracy, mean tokens, p50/p95 latency ms. A second breakdown by `scope`
(single_admission vs cross_admission) is produced the same way. Both are
printed and written to `results/<patient_id>__<system_name>.json`.

Two headline numbers are reported *separately*, never blended: answerable
accuracy (pooled over every non-adversarial item) and abstention accuracy
(the adversarial category's own accuracy, which — per judge.py's routing —
means "did the system decline to answer"). ARCHITECTURE.md §1's Pareto
framing depends on this distinction staying visible; folding it into one
"overall accuracy" number would erase exactly the story the framing tells.

Invocation: ``uv run python -m medmemgraph.eval.harness --patient <id>
[--system nomem fullctx] [--dry-run] [--k N]``.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from medmemgraph.eval.baselines.fullctx import FullContextAnswerer
from medmemgraph.eval.baselines.nomem import NoMemoryAnswerer
from medmemgraph.eval.judge import Judge
from medmemgraph.eval.reader import ReaderChainOfNoteAnswerer, ReaderDirectAnswerer
from medmemgraph.eval.types import Answerer
from medmemgraph.pipeline.loader import Conversation, load_conversation, load_qa

# ---------------------------------------------------------------------------
# System registry
# ---------------------------------------------------------------------------

SYSTEM_FACTORIES: dict[str, Callable[..., Answerer]] = {
    NoMemoryAnswerer.name: NoMemoryAnswerer,
    FullContextAnswerer.name: FullContextAnswerer,
    ReaderDirectAnswerer.name: ReaderDirectAnswerer,
    ReaderChainOfNoteAnswerer.name: ReaderChainOfNoteAnswerer,
}
"""Registered baseline systems, keyed by name. Add a new baseline here to
make it runnable by name from the CLI / evaluate(). `reader_direct` /
`reader_con` (eval/reader.py) are the Chain-of-Note A/B pair: same
`mock_retrieve`-sourced evidence pack, only the reading strategy differs —
run both with `--system reader_direct reader_con` and diff the per-category
tables below to compare `mode="direct"` against `mode="chain_of_note"` on
identical retrieved evidence, per the story's A/B requirement."""

# Canonical row order — the six MedLoCoMo question_type categories named in
# the story, verified present in the real corpus (see dev.log.md entry for
# this story: sampled 2,892 items across 15 patients, all six present).
QUESTION_TYPES = (
    "medical_reasoning",
    "care_plan_rationale",
    "longitudinal_progression",
    "cross_admission_comparison",
    "frequency_pattern",
    "adversarial",
)
SCOPES = ("single_admission", "cross_admission")

ADVERSARIAL_TYPE = "adversarial"


# ---------------------------------------------------------------------------
# Result shapes
# ---------------------------------------------------------------------------


@dataclass
class Record:
    """One QA item's outcome under one system."""

    qa_id: str
    question_type: str
    scope: str
    correct: bool
    mode: str
    judge_kind: str
    judge_reason: str
    tokens: int
    latency_ms: float
    truncated: bool


@dataclass
class CategoryRow:
    category: str
    n: int
    accuracy: float | None
    mean_tokens: float
    p50_latency_ms: float
    p95_latency_ms: float


@dataclass
class HarnessRun:
    patient_id: str
    system_name: str
    dry_run: bool
    judge_kind: str
    n_items: int
    by_question_type: list[CategoryRow]
    by_scope: list[CategoryRow]
    answerable_accuracy: float | None
    abstention_accuracy: float | None
    n_truncated: int
    records: list[Record] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(pct / 100.0 * (len(ordered) - 1))))
    return ordered[idx]


def _accuracy(records: list[Record]) -> float | None:
    if not records:
        return None
    return sum(1 for r in records if r.correct) / len(records)


def _aggregate(
    records: list[Record], *, key: Callable[[Record], str], order: tuple[str, ...]
) -> list[CategoryRow]:
    buckets: dict[str, list[Record]] = defaultdict(list)
    for r in records:
        buckets[key(r)].append(r)

    ordered_keys = [k for k in order if k in buckets]
    ordered_keys += sorted(k for k in buckets if k not in order)

    rows = []
    for k in ordered_keys:
        items = buckets[k]
        tokens = [it.tokens for it in items]
        latencies = [it.latency_ms for it in items]
        rows.append(
            CategoryRow(
                category=k,
                n=len(items),
                accuracy=_accuracy(items),
                mean_tokens=(sum(tokens) / len(tokens)) if tokens else 0.0,
                p50_latency_ms=_percentile(latencies, 50),
                p95_latency_ms=_percentile(latencies, 95),
            )
        )
    return rows


# ---------------------------------------------------------------------------
# Core evaluation — testable without touching the corpus or the network
# ---------------------------------------------------------------------------


def evaluate(
    qa_items: list[dict],
    conversation: Conversation | None,
    *,
    patient_id: str,
    system_name: str,
    dry_run: bool = False,
    answerer_model: str | None = None,
    judge_model: str | None = None,
    k: int | None = None,
    rendering: str | None = None,
    retrieve_k: int | None = None,
    answerer: Answerer | None = None,
    judge: Judge | None = None,
) -> HarnessRun:
    """Run `system_name` (or an injected `answerer`) over `qa_items` and
    aggregate into a `HarnessRun`. Split from `run_harness()` so tests can
    exercise the whole scoring path against fixture QA items without the
    real MedLoCoMo corpus on disk.

    `rendering` (`json`/`prose`/`shuffled`) and `retrieve_k` (top-k passed
    to the retriever) are only meaningful for the `reader_direct`/
    `reader_con` systems (eval/reader.py) — harmlessly ignored for
    `nomem`/`fullctx`, which do not accept those constructor kwargs (see
    `_build_answerer`'s signature-introspection guard)."""
    if k is not None:
        qa_items = qa_items[:k]

    if answerer is None:
        answerer = _build_answerer(
            system_name, dry_run=dry_run, model=answerer_model, rendering=rendering, retrieve_k=retrieve_k
        )

    if judge is None:
        judge_kwargs: dict = {"force_fallback": dry_run}
        if judge_model:
            judge_kwargs["model"] = judge_model
        judge = Judge(**judge_kwargs)

    records: list[Record] = []
    for item in qa_items:
        result = answerer.answer(item["question"], conversation, patient_id=patient_id)
        verdict = judge.judge(
            question=item["question"],
            gold_answer=item["answer"],
            system_answer=result.text,
            question_type=item["question_type"],
        )
        records.append(
            Record(
                qa_id=item["qa_id"],
                question_type=item["question_type"],
                scope=item["scope"],
                correct=verdict.correct,
                mode=verdict.mode,
                judge_kind=verdict.judge_kind,
                judge_reason=verdict.reason,
                tokens=result.total_tokens,
                latency_ms=result.latency_ms,
                truncated=result.truncated,
            )
        )

    by_question_type = _aggregate(records, key=lambda r: r.question_type, order=QUESTION_TYPES)
    by_scope = _aggregate(records, key=lambda r: r.scope, order=SCOPES)

    answerable = [r for r in records if r.question_type != ADVERSARIAL_TYPE]
    adversarial = [r for r in records if r.question_type == ADVERSARIAL_TYPE]

    return HarnessRun(
        patient_id=patient_id,
        system_name=system_name,
        dry_run=dry_run,
        judge_kind=judge.kind,
        n_items=len(records),
        by_question_type=by_question_type,
        by_scope=by_scope,
        answerable_accuracy=_accuracy(answerable),
        abstention_accuracy=_accuracy(adversarial) if adversarial else None,
        n_truncated=sum(1 for r in records if r.truncated),
        records=records,
    )


def _build_answerer(
    system_name: str,
    *,
    dry_run: bool,
    model: str | None,
    rendering: str | None = None,
    retrieve_k: int | None = None,
) -> Answerer:
    factory = SYSTEM_FACTORIES.get(system_name)
    if factory is None:
        raise ValueError(
            f"unknown system {system_name!r}; known systems: {sorted(SYSTEM_FACTORIES)}"
        )
    if not dry_run and not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            f"system {system_name!r} is an LLM baseline and needs ANTHROPIC_API_KEY to "
            "call the Anthropic API. Set it in .env, or pass --dry-run to exercise the "
            "harness with a stub answerer and no API calls."
        )
    kwargs: dict = {"dry_run": dry_run}
    if model:
        kwargs["model"] = model
    # Only forward reader-specific kwargs to factories that actually accept
    # them (nomem/fullctx do not) — introspect rather than hard-coding a
    # system-name check, so a future baseline with the same kwarg names
    # picks this up for free.
    sig_params = inspect.signature(factory).parameters
    if rendering is not None and "rendering" in sig_params:
        kwargs["rendering"] = rendering
    if retrieve_k is not None and "k" in sig_params:
        kwargs["k"] = retrieve_k
    return factory(**kwargs)


def run_harness(
    patient_id: str,
    system_name: str,
    *,
    root: str | os.PathLike[str] | None = None,
    **kwargs,
) -> HarnessRun:
    """Load one patient's real conversation + QA (allowlisted loader — never
    HydraDB, never `formed_packet.json`) and evaluate `system_name` over
    them."""
    conversation = load_conversation(patient_id, root)
    qa_items = load_qa(patient_id, root)
    return evaluate(qa_items, conversation, patient_id=patient_id, system_name=system_name, **kwargs)


# ---------------------------------------------------------------------------
# Rendering + persistence
# ---------------------------------------------------------------------------

_HEADERS = ("category", "n", "accuracy", "mean_tokens", "p50_lat_ms", "p95_lat_ms")
_COL_WIDTHS = (26, 5, 10, 13, 12, 12)


def render_table(title: str, rows: list[CategoryRow]) -> str:
    lines = [f"-- {title} --"]
    header_line = "  ".join(h.ljust(w) if i == 0 else h.rjust(w) for i, (h, w) in enumerate(zip(_HEADERS, _COL_WIDTHS)))
    lines.append(header_line)
    for row in rows:
        cells = (
            row.category.ljust(_COL_WIDTHS[0]),
            str(row.n).rjust(_COL_WIDTHS[1]),
            ("n/a" if row.accuracy is None else f"{row.accuracy:.3f}").rjust(_COL_WIDTHS[2]),
            f"{row.mean_tokens:.1f}".rjust(_COL_WIDTHS[3]),
            f"{row.p50_latency_ms:.2f}".rjust(_COL_WIDTHS[4]),
            f"{row.p95_latency_ms:.2f}".rjust(_COL_WIDTHS[5]),
        )
        lines.append("  ".join(cells))
    return "\n".join(lines)


def render_run(run: HarnessRun) -> str:
    parts = [
        f"=== system={run.system_name} patient={run.patient_id} n_items={run.n_items} "
        f"dry_run={run.dry_run} judge={run.judge_kind} ===",
        render_table("by question_type", run.by_question_type),
        render_table("by scope", run.by_scope),
        f"answerable_accuracy={run.answerable_accuracy if run.answerable_accuracy is None else round(run.answerable_accuracy, 3)}  "
        f"abstention_accuracy={run.abstention_accuracy if run.abstention_accuracy is None else round(run.abstention_accuracy, 3)}  "
        f"n_truncated={run.n_truncated}/{run.n_items}",
    ]
    return "\n".join(parts)


def write_results(run: HarnessRun, results_dir: str | os.PathLike[str] = "results") -> Path:
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{run.patient_id}__{run.system_name}.json"
    payload = run.to_dict()
    payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(payload, indent=2))
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m medmemgraph.eval.harness",
        description=(
            "Run no-memory / full-context baselines over one patient's real "
            "MedLoCoMo QA and print a per-category results table. Never "
            "touches HydraDB."
        ),
    )
    parser.add_argument("--patient", required=True, help="MedLoCoMo subject_id")
    parser.add_argument(
        "--system",
        action="append",
        choices=sorted(SYSTEM_FACTORIES),
        help="baseline system to run (repeatable); default: all registered systems",
    )
    parser.add_argument("--root", default=None, help="MedLoCoMo corpus root (default: $MEDLOCOMO_ROOT or data/medlocomo)")
    parser.add_argument("--k", type=int, default=None, help="limit to the first N QA items (smoke testing)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="stub answerer, deterministic judge, zero API calls (CI-safe)",
    )
    parser.add_argument("--answerer-model", default=None, help="override the answerer model id")
    parser.add_argument("--judge-model", default=None, help="override the judge model id")
    parser.add_argument("--results-dir", default="results", help="directory to write per-run JSON into")
    parser.add_argument(
        "--rendering",
        choices=["json", "prose", "shuffled"],
        default=None,
        help="context-rendering strategy for reader_direct/reader_con (default: json)",
    )
    parser.add_argument(
        "--retrieve-k",
        type=int,
        default=None,
        help="top-k passed to the retriever for reader_direct/reader_con (default: 6)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    systems = args.system or list(SYSTEM_FACTORIES)

    for system_name in systems:
        try:
            run = run_harness(
                args.patient,
                system_name,
                root=args.root,
                dry_run=args.dry_run,
                k=args.k,
                answerer_model=args.answerer_model,
                judge_model=args.judge_model,
                rendering=args.rendering,
                retrieve_k=args.retrieve_k,
            )
        except RuntimeError as exc:
            print(f"skipping {system_name}: {exc}", file=sys.stderr)
            continue
        print(render_run(run))
        path = write_results(run, args.results_dir)
        print(f"wrote {path}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SYSTEM_FACTORIES",
    "QUESTION_TYPES",
    "SCOPES",
    "Record",
    "CategoryRow",
    "HarnessRun",
    "evaluate",
    "run_harness",
    "render_table",
    "render_run",
    "write_results",
    "main",
]
