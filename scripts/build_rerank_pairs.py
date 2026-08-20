"""FT-E1-S2: write gold + arctic-s hard-neg pairs (allowlisted readers)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from medmemgraph.eval.rerank_pairs import build_pairs
from medmemgraph.eval.rerank_split import EVAL_TRIO

SPLIT_DEFAULT = Path("results/finetune-reranker/patient_split.json")


def main() -> int:
    split_path = SPLIT_DEFAULT
    if not split_path.is_file():
        print(f"missing {split_path}; run scripts/build_rerank_split.py first", file=sys.stderr)
        return 1
    split = json.loads(split_path.read_text(encoding="utf-8"))
    leaked = [p for p in EVAL_TRIO if p in split.get("train", []) or p in split.get("dev", [])]
    if leaked:
        print(f"eval trio in train/dev: {leaked}", file=sys.stderr)
        return 1
    manifest = build_pairs(split)
    print(json.dumps({k: manifest[k] for k in (
        "n_gold_turn", "n_gold_admission_expanded", "n_hardneg",
    )}, indent=2))
    print(f"subjects={len(manifest['subject_ids'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
