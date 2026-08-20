"""Recompute Hit/nDCG @2,5,10,20 with the fixed IDCG (nDCG in [0, 1]).

Two separate sweeps — never the product grid (that would put Qwen
rerank on arctic-s at 123 s/query).

  GPU:  qwen3-0.6b + qwen3-rerank-0.6b
  CPU:  arctic-s + MiniLM ONNX int8 (unfinetuned) and the ORPO int8 arm
"""

from __future__ import annotations

import json
from pathlib import Path

from medmemgraph.eval.retrieval_eval import (
    RERANK_KS,
    aggregate_config,
    run_sweep,
    write_report,
)

TRIO = ["10056223", "10213338", "10312715"]
OUT = Path("results/finetune-reranker/ndcg_fixed.md")
JSON_OUT = Path("results/finetune-reranker/ndcg_fixed.json")


def _rows(sweep) -> list[dict]:
    rows = []
    for cfg in sweep.configs:
        agg = aggregate_config(cfg)
        row = {
            "embedder": cfg.embedder,
            "reranker": cfg.reranker,
            "n_admission": agg.n_admission,
            "n_turn": agg.n_turn,
            "hit_admission": {str(k): agg.hit_admission[k] for k in RERANK_KS},
            "ndcg_admission": {str(k): agg.ndcg_admission[k] for k in RERANK_KS},
            "hit_turn": {str(k): agg.hit_turn[k] for k in RERANK_KS},
            "ndcg_turn": {str(k): agg.ndcg_turn[k] for k in RERANK_KS},
        }
        for k in RERANK_KS:
            na = row["ndcg_admission"][str(k)]
            nt = row["ndcg_turn"][str(k)]
            if na == na and na > 1.0 + 1e-9:
                raise RuntimeError(f"admission nDCG@{k} > 1: {na} ({cfg.embedder}/{cfg.reranker})")
            if nt == nt and nt > 1.0 + 1e-9:
                raise RuntimeError(f"turn nDCG@{k} > 1: {nt} ({cfg.embedder}/{cfg.reranker})")
        rows.append(row)
    return rows


def _fmt(row: dict, grain: str) -> str:
    hit = row[f"hit_{grain}"]
    ndcg = row[f"ndcg_{grain}"]
    n = row["n_admission"] if grain == "admission" else row["n_turn"]
    cells = [f"{row['embedder']} + {row['reranker']}", str(n)]
    for k in RERANK_KS:
        cells.append(f"{hit[str(k)]:.3f}")
    for k in RERANK_KS:
        cells.append(f"{ndcg[str(k)]:.3f}")
    return "| " + " | ".join(cells) + " |"


def main() -> int:
    header = (
        "| System | n | Hit@2 | Hit@5 | Hit@10 | Hit@20 | "
        "nDCG@2 | nDCG@5 | nDCG@10 | nDCG@20 |"
    )
    sep = "|---|---|---|---|---|---|---|---|---|---|"

    print("=== GPU Qwen ===", flush=True)
    gpu = run_sweep(
        TRIO,
        embedders_=("qwen3-0.6b",),
        rerankers_=("qwen3-rerank-0.6b",),
        n_candidates=100,
        cache_path=None,
    )
    write_report(gpu, out_dir=Path("results"))

    print("=== CPU MiniLM ===", flush=True)
    cpu = run_sweep(
        TRIO,
        embedders_=("arctic-s",),
        rerankers_=(
            "ms-marco-minilm-l6-v2-onnx-int8",
            "ms-marco-minilm-l6-v2-ft-orpo-onnx-int8",
        ),
        n_candidates=100,
        cache_path=None,
    )
    write_report(cpu, out_dir=Path("results"))

    rows = _rows(gpu) + _rows(cpu)
    body = f"""# Hit / nDCG with fixed IDCG (nDCG in [0, 1])

Same 3 patients (`10056223`, `10213338`, `10312715`). Fixed
`metrics.ndcg_at_k`: IDCG uses the same `_is_relevant` labels as DCG.
Older admission nDCG columns that exceed 1.0 are the pre-fix bug.

GPU: `qwen3-0.6b` + `qwen3-rerank-0.6b` (not run on CPU).
CPU: `arctic-s` + MiniLM ONNX int8 (unfinetuned and CE-ORPO).

## Turn grain

{header}
{sep}
{chr(10).join(_fmt(r, "turn") for r in rows)}

## Admission grain

{header}
{sep}
{chr(10).join(_fmt(r, "admission") for r in rows)}
"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(body, encoding="utf-8")
    JSON_OUT.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(body)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
