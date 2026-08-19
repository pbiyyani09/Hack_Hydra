#!/usr/bin/env python
"""scripts/ingest_corpus.py — the corpus-scale ingest entry point.

`pipeline.ingest.ingest_patient` deliberately handles exactly ONE patient and
deliberately does NOT persist the two pieces of cross-run state it accepts. Its
own docstring assigns both jobs to "a corpus-scale caller's own loop"; this
script is that caller. Until it existed, `ingest_patient` had zero callers
anywhere in the repo — nothing had ever put a patient into HydraDB.

What this script owns, and `ingest_patient` deliberately does not:

  * **The loop**, and the gate re-running on every iteration of it
    (`assert_handcheck_passed()` is a cheap filesystem check per call, not a
    one-time bypass, so looping cannot skip it).
  * **`id_map` persistence** (`--id-map`). Node ids are deterministic SHA-256
    folds, so replay is idempotent *without* this file; the map matters for
    collision linear-probing, which only stays consistent across restarts if
    the map is carried over.
  * **`registry` persistence** (`--registry`). This is what makes entity
    resolution *incremental*: patient N+1's mentions are matched against the
    canonical entities patients 1..N already established, instead of every
    patient starting from an empty world.
  * **Index building** (`--build-indexes`). `PatientIndex.save` /
    `LexicalIndex.save` exist, are tested, and had no production caller, so
    `graph/retrieve.py` re-embedded every turn of every patient on every process
    start. Building once here turns that into a file read.
  * **Refusing to look like success when it isn't.** Any patient that fails is
    reported and the script exits non-zero.

Cost: one LLM call per admission (~30/patient). `medmemgraph.llm` caches every
completion to disk, so re-running after a crash — or after the HydraDB container
restarts, which wipes the graph under `CLOUD_PROVIDER=memory` — re-writes the
graph at ~$0 in API spend.

Usage:
    # smoke: 3 patients, real extraction
    uv run python scripts/ingest_corpus.py --limit 3

    # the real run
    uv run python scripts/ingest_corpus.py --limit 20

    # offline shape check, zero API cost (rule-based extractor)
    uv run python scripts/ingest_corpus.py --limit 1 --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from medmemgraph import llm
from medmemgraph.graph.lexical import LexicalIndex
from medmemgraph.graph.vector_index import DEFAULT_INDEX_DIR, PatientIndex
from medmemgraph.hydra_client import HydraClient
from medmemgraph.pipeline.extract import Extractor
from medmemgraph.pipeline.ids import IdMinter
from medmemgraph.pipeline.ingest import ingest_patient
from medmemgraph.pipeline.loader import list_patients, load_conversation
from medmemgraph.pipeline.resolve import CanonicalRegistry
from medmemgraph.pipeline.scale_gate import ScaleGateError

DEFAULT_STATE_DIR = Path("data/state")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python scripts/ingest_corpus.py",
        description="Ingest N MedLoCoMo patients into HydraDB, persisting id_map/registry between them.",
    )
    select = p.add_mutually_exclusive_group()
    select.add_argument("--patients", nargs="+", help="explicit subject_ids to ingest")
    select.add_argument(
        "--limit",
        type=int,
        help="ingest the first N patients by sorted subject_id (deterministic, "
        "so the selection is disclosable and not cherry-picked)",
    )
    p.add_argument("--root", default=None, help="corpus root (default: $MEDLOCOMO_ROOT or data/medlocomo)")
    p.add_argument(
        "--now",
        default=None,
        help="ISO timestamp recorded as the ingest clock reading (default: now, UTC). "
        "Explicit so a replay can reproduce a prior run's invalidation timestamps.",
    )
    p.add_argument("--id-map", default=str(DEFAULT_STATE_DIR / "id_map.json"))
    p.add_argument("--registry", default=str(DEFAULT_STATE_DIR / "registry.json"))
    p.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR))
    p.add_argument(
        "--build-indexes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="build and save the per-patient vector + lexical indexes (default: on)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="use the deterministic rule-based extractor instead of a real LLM call. "
        "Exercises the whole write path at zero API cost; the FACTS are not "
        "quality-bearing, so never report numbers from a --dry-run graph.",
    )
    p.add_argument("--batch-size", type=int, default=1000)
    return p


def _load_id_map(path: Path) -> dict[str, int]:
    try:
        return {k: int(v) for k, v in json.loads(path.read_text()).items()}
    except FileNotFoundError:
        return {}


def _save_id_map(path: Path, id_map: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(id_map, indent=0, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    now = args.now or datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    if args.patients:
        subject_ids = list(args.patients)
    else:
        available = list_patients(args.root)
        subject_ids = available[: args.limit] if args.limit else available

    if not subject_ids:
        print("no patients selected", file=sys.stderr)
        return 2

    id_map_path = Path(args.id_map)
    registry_path = Path(args.registry)
    id_map = _load_id_map(id_map_path)
    minter = IdMinter(id_map)
    registry = CanonicalRegistry.load_json(registry_path)

    print(f"ingesting {len(subject_ids)} patient(s); now={now}")
    print(f"  id_map   {id_map_path} ({len(id_map)} entries loaded)")
    print(f"  registry {registry_path} ({sum(len(v) for v in registry.by_patient.values())} entities loaded)")
    print(f"  extractor {'rule-based (--dry-run)' if args.dry_run else 'llm'}")
    print()

    failures: list[tuple[str, str]] = []
    totals = {"facts": 0, "written": 0, "turns": 0, "superseded": 0, "contradicted": 0}
    started = time.monotonic()

    # One client for the whole run rather than one per patient — ingest_patient
    # opens a short-lived client only when it is not given one.
    with HydraClient(transport="bolt") as client:
        for i, subject_id in enumerate(subject_ids, start=1):
            t0 = time.monotonic()
            prefix = f"[{i}/{len(subject_ids)}] {subject_id}"
            try:
                conversation = load_conversation(subject_id, args.root)
                report = ingest_patient(
                    subject_id,
                    now=now,
                    conversation=conversation,
                    client=client,
                    extractor=Extractor(dry_run=args.dry_run),
                    registry=registry,
                    id_map=minter,
                    batch_size=args.batch_size,
                )
            except ScaleGateError as exc:
                # The gate is a human sign-off, not a transient error. Stop the
                # whole run rather than reporting it once per patient.
                print(f"{prefix}: BLOCKED\n\n{exc}", file=sys.stderr)
                return 3
            except Exception as exc:  # noqa: BLE001 — one patient must not sink the run
                print(f"{prefix}: FAILED — {type(exc).__name__}: {exc}", file=sys.stderr)
                failures.append((subject_id, f"{type(exc).__name__}: {exc}"))
                continue

            inv = report.invalidation_report
            totals["facts"] += report.n_facts_extracted
            totals["written"] += report.n_facts_written
            totals["turns"] += report.n_turns_written
            totals["superseded"] += len(inv.supersedes_written)
            totals["contradicted"] += len(inv.contradicts_written)

            index_note = ""
            if args.build_indexes:
                try:
                    dense = PatientIndex(subject_id)
                    dense.build(conversation, report.facts)
                    dense.save(args.index_dir)
                    lex = LexicalIndex(subject_id)
                    lex.build(conversation)
                    lex.save(args.index_dir)
                    index_note = "  indexes:saved"
                except Exception as exc:  # noqa: BLE001 — indexes are a cache, not the graph
                    index_note = f"  indexes:FAILED({type(exc).__name__})"

            # Persist after EVERY patient, not at the end: a crash at patient 17
            # must not throw away the entity resolution done for 1..16.
            _save_id_map(id_map_path, dict(minter.id_map))
            registry_path.parent.mkdir(parents=True, exist_ok=True)
            registry.save_json(registry_path)

            print(
                f"{prefix}: facts={report.n_facts_extracted} written={report.n_facts_written} "
                f"skipped={report.write_report.facts_skipped} turns={report.n_turns_written} "
                f"supersedes={len(inv.supersedes_written)} contradicts={len(inv.contradicts_written)} "
                f"[{time.monotonic() - t0:.1f}s]{index_note}"
            )
            if report.write_report.skipped:
                print(f"    skips: {report.write_report.skip_summary()}")

    elapsed = time.monotonic() - started
    spent = _spend_usd()
    print()
    print(
        f"done in {elapsed:.0f}s — {len(subject_ids) - len(failures)}/{len(subject_ids)} patients; "
        f"facts {totals['written']}/{totals['facts']} written, {totals['turns']} turns, "
        f"{totals['superseded']} SUPERSEDES, {totals['contradicted']} CONTRADICTS"
    )
    if spent is not None:
        print(f"cumulative LLM spend (ledger, all runs): ${spent:.4f}")
    if failures:
        print(f"\n{len(failures)} patient(s) FAILED:", file=sys.stderr)
        for subject_id, reason in failures:
            print(f"  {subject_id}: {reason}", file=sys.stderr)
        return 1
    return 0


def _spend_usd() -> float | None:
    """Cumulative ledger spend, best-effort. The ledger persists across runs, so
    this is a lifetime figure against MEDMEMGRAPH_MAX_USD, not this run's cost."""
    try:
        data = json.loads((llm.CACHE_DIR / "ledger.json").read_text())
    except Exception:  # noqa: BLE001
        return None
    for key in ("total_usd", "committed_usd", "spent_usd"):
        if isinstance(data.get(key), (int, float)):
            return float(data[key])
    return None


if __name__ == "__main__":
    raise SystemExit(main())
