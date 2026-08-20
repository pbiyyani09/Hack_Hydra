"""FT-E3-S1: McNemar the already-measured turn-grain −9.0pp. Disk only."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# This script must not import reranker / embedders / run_sweep.
from medmemgraph.eval.existing_mcnemar import compute_from_rows, render_note

RAW_DEFAULT = Path("results/cpu_ablation_raw_items.json")
OUT_DEFAULT = Path("results/finetune-reranker/existing_turn_grain_mcnemar.md")


def main(argv: list[str] | None = None) -> int:
    del argv
    raw_path = RAW_DEFAULT
    if not raw_path.is_file():
        print(f"missing {raw_path}", file=sys.stderr)
        return 1
    rows = json.loads(raw_path.read_text(encoding="utf-8"))
    summary = compute_from_rows(rows)
    note = render_note(summary)
    OUT_DEFAULT.parent.mkdir(parents=True, exist_ok=True)
    OUT_DEFAULT.write_text(note, encoding="utf-8")
    print(note)
    print(f"wrote {OUT_DEFAULT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
