"""Write results/finetune-reranker/patient_split.json."""

from __future__ import annotations

import json
from pathlib import Path

from medmemgraph.eval.rerank_split import build_split

OUT = Path("results/finetune-reranker/patient_split.json")


def main() -> int:
    split = build_split()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
    print(
        f"train={len(split['train'])} dev={len(split['dev'])} eval={split['eval']}"
    )
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
