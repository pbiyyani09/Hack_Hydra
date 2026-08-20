"""GPU Qwen-FT on exploded hybrid arctic-s pool (same as MiniLM packing eval)."""

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
OUT = Path("results/finetune-reranker/qwen_ft_exploded.md")
RERANKER = "qwen3-rerank-0.6b-ft-listwise"
POOL = 100


def _agg(rows: list[dict]) -> dict:
    n = len(rows)
    out = {"n": n}
    for k in RERANK_KS:
        out[f"hit@{k}"] = sum(r["hit"][k] for r in rows) / n
        out[f"ndcg@{k}"] = sum(r["ndcg"][k] for r in rows) / n
        if out[f"ndcg@{k}"] > 1.0 + 1e-6:
            raise RuntimeError("nDCG>1")
    out["mean_rel10"] = sum(r["rel10"] for r in rows) / n
    out["mean_rel_pool"] = sum(r["rel_pool"] for r in rows) / n
    return out


def main() -> int:
    indexes = {}
    for pid in TRIO:
        indexes[pid] = build_indexes(pid, load_conversation(pid), dense_backend="arctic-s")
    reranker = build_reranker(RERANKER)
    reranker.rerank("warmup", ["warmup"])
    adm, turn = [], []
    for pid in TRIO:
        dense, lex = indexes[pid]
        for item in load_qa(pid):
            ev = item.get("evidence") or {}
            if not ev.get("admissions"):
                continue
            cands = hybrid_search(item["question"], dense, lex, k=POOL, per_arm=POOL)
            ranked = reranker.rerank(item["question"], [c.text for c in cands], top_k=None)
            reranked = [cands[i] for i, _ in ranked]

            def rec(evidence: dict) -> dict:
                return {
                    "hit": {k: hit_at_k(reranked, evidence, k) for k in RERANK_KS},
                    "ndcg": {k: ndcg_at_k(reranked, evidence, k) for k in RERANK_KS},
                    "rel10": sum(1 for c in reranked[:10] if hit_at_k([c], evidence, 1)),
                    "rel_pool": sum(1 for c in reranked if hit_at_k([c], evidence, 1)),
                }

            adm_ev = admission_only_evidence(ev)
            adm.append(rec(adm_ev))
            trn = turn_only_evidence(ev)
            if trn is not None:
                turn.append(rec(trn))
    ta, aa = _agg(turn), _agg(adm)
    body = f"""# Qwen-FT (recipe-matched) exploded hybrid pool {POOL}

## Turn n={ta['n']}
Hit@2/5/10/20 = {ta['hit@2']:.3f} / {ta['hit@5']:.3f} / {ta['hit@10']:.3f} / {ta['hit@20']:.3f}
nDCG@2/5/10/20 = {ta['ndcg@2']:.3f} / {ta['ndcg@5']:.3f} / {ta['ndcg@10']:.3f} / {ta['ndcg@20']:.3f}
mean |rel@10|={ta['mean_rel10']:.2f} |rel@pool|={ta['mean_rel_pool']:.2f}

## Admission n={aa['n']}
Hit@2/5/10/20 = {aa['hit@2']:.3f} / {aa['hit@5']:.3f} / {aa['hit@10']:.3f} / {aa['hit@20']:.3f}
nDCG@2/5/10/20 = {aa['ndcg@2']:.3f} / {aa['ndcg@5']:.3f} / {aa['ndcg@10']:.3f} / {aa['ndcg@20']:.3f}
mean |rel@10|={aa['mean_rel10']:.2f} |rel@pool|={aa['mean_rel_pool']:.2f}
"""
    OUT.write_text(body, encoding="utf-8")
    print(body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
