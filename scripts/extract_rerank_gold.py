"""Write the allowlisted MedLoCoMo gold set for rerank packing.

Train/dev only. Eval trio is refused. No Synthea; no formed_packet.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from medmemgraph.eval.rerank_gold import records_for_patient
from medmemgraph.eval.rerank_split import EVAL_TRIO
from scripts.finetune_minilm_reranker import SPLIT_DEFAULT

OUT_DIR = Path("data/reranker_ft")
MANIFEST = Path("results/finetune-reranker/gold_set.md")


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in rows:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> int:
    split = json.loads(SPLIT_DEFAULT.read_text(encoding="utf-8"))
    leaked = [p for p in EVAL_TRIO if p in split["train"] or p in split["dev"]]
    if leaked:
        raise SystemExit(f"eval trio in train/dev: {leaked}")
    counts = {}
    for name in ("train", "dev"):
        rows: list[dict] = []
        ids = list(split[name])
        for i, sid in enumerate(ids, 1):
            print(f"  gold {name} {i}/{len(ids)} {sid}", flush=True)
            rows.extend(records_for_patient(sid, split=name))
        path = OUT_DIR / f"gold_{name}.jsonl"
        _write(path, rows)
        n_turn = sum(1 for r in rows if r["gold_turn_ids"])
        n_units3 = sum(sum(1 for u in r["units"] if u["grade"] == 3) for r in rows)
        n_units2 = sum(sum(1 for u in r["units"] if u["grade"] == 2) for r in rows)
        counts[name] = {
            "path": str(path),
            "n_qa": len(rows),
            "n_turn_qa": n_turn,
            "n_grade3": n_units3,
            "n_grade2": n_units2,
        }
        print(f"{name}: {counts[name]}", flush=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(
        f"""# Rerank gold set (MedLoCoMo extract)

Allowlisted files only (`combined_conversation.json`, `benchmark_qa.json`).
Eval trio held out. Not Synthea. Not `formed_packet.json`.

Grades: 3 = gold turn_ids, 2 = ±1 neighbor, 1 = other same-admission
(ids only in `same_admission_other`).

- train: `{counts['train']}`
- dev: `{counts['dev']}`
""",
        encoding="utf-8",
    )
    print(MANIFEST.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
