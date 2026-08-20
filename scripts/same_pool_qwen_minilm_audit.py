"""Fairness audit: Hub Qwen, Qwen-FT, MiniLM-listwise on THE SAME pool.

If Qwen still loses nDCG on identical candidates, it is not an embedder
mismatch. Eval trio only. retrieve.py is not imported.
"""

from __future__ import annotations

import json
from pathlib import Path

from medmemgraph.eval.hybrid_pool import build_indexes, hybrid_search
from medmemgraph.eval.metrics import hit_at_k, ndcg_at_k
from medmemgraph.eval.retrieval_eval import (
    RERANK_KS,
    admission_only_evidence,
    build_reranker,
    turn_only_evidence,
)
from medmemgraph.pipeline.loader import load_conversation, load_qa

TRIO = ["10056223", "10213338", "10312715"]
OUT = Path("results/finetune-reranker/same_pool_audit.md")
JSON_OUT = Path("results/finetune-reranker/same_pool_audit.json")
RERANKERS = [
    "qwen3-rerank-0.6b",
    "qwen3-rerank-0.6b-ft-listwise",
    "ms-marco-minilm-l6-v2-ft-listwise-onnx-int8",
]


def _agg(rows: list[dict]) -> dict:
    n = len(rows)
    out = {"n": n}
    for k in RERANK_KS:
        out[f"hit@{k}"] = sum(r["hit"][k] for r in rows) / n
        out[f"ndcg@{k}"] = sum(r["ndcg"][k] for r in rows) / n
        if out[f"ndcg@{k}"] > 1.0 + 1e-6:
            raise RuntimeError(f"nDCG@{k} > 1")
    return out


def main() -> int:
    indexes = {}
    for pid in TRIO:
        indexes[pid] = build_indexes(pid, load_conversation(pid), dense_backend="arctic-s")

    pools: dict[tuple[str, str], list] = {}
    for pid in TRIO:
        for item in load_qa(pid):
            if not (item.get("evidence") or {}).get("admissions"):
                continue
            dense, lex = indexes[pid]
            pools[(pid, str(item["qa_id"]))] = (
                item,
                hybrid_search(item["question"], dense, lex, k=100),
            )

    arms = []
    for name in RERANKERS:
        print(f"rerank {name}…", flush=True)
        reranker = build_reranker(name)
        reranker.rerank("warmup", ["warmup"])
        adm: list[dict] = []
        turn: list[dict] = []
        for (pid, _qid), (item, cands) in pools.items():
            ranked = reranker.rerank(item["question"], [c.text for c in cands], top_k=None)
            reranked = [cands[i] for i, _ in ranked]
            ev = item["evidence"]
            adm_ev = admission_only_evidence(ev)
            adm.append(
                {
                    "hit": {k: hit_at_k(reranked, adm_ev, k) for k in RERANK_KS},
                    "ndcg": {k: ndcg_at_k(reranked, adm_ev, k) for k in RERANK_KS},
                }
            )
            trn = turn_only_evidence(ev)
            if trn is not None:
                turn.append(
                    {
                        "hit": {k: hit_at_k(reranked, trn, k) for k in RERANK_KS},
                        "ndcg": {k: ndcg_at_k(reranked, trn, k) for k in RERANK_KS},
                    }
                )
        arms.append({"reranker": name, "pool": "arctic-s+hyb", "admission": _agg(adm), "turn": _agg(turn)})
        print(json.dumps(arms[-1], indent=2), flush=True)

    hdr = "| reranker | n | Hit@2 | Hit@5 | Hit@10 | nDCG@2 | nDCG@5 | nDCG@10 |"
    sep = "|---|---|---|---|---|---|---|---|"

    def line(arm: dict, grain: str) -> str:
        a = arm[grain]
        cells = [arm["reranker"].replace("ms-marco-minilm-l6-v2-", "m-"), str(a["n"])]
        for k in (2, 5, 10):
            cells.append(f"{a[f'hit@{k}']:.3f}")
        for k in (2, 5, 10):
            cells.append(f"{a[f'ndcg@{k}']:.3f}")
        return "| " + " | ".join(cells) + " |"

    body = f"""# Same-pool audit (arctic-s hybrid, 100 cands)

If Hub/FT Qwen still trail MiniLM here, the gap is the reranker (train
budget / packing), not a different first-stage.

## Turn

{hdr}
{sep}
{chr(10).join(line(a, "turn") for a in arms)}

## Admission

{hdr}
{sep}
{chr(10).join(line(a, "admission") for a in arms)}
"""
    OUT.write_text(body, encoding="utf-8")
    JSON_OUT.write_text(json.dumps(arms, indent=2) + "\n", encoding="utf-8")
    print(body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
