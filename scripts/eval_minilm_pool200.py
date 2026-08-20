"""CPU MiniLM-listwise ONNX on hybrid pool 200 (rerank all 200).

Hypothesis: |rel@200|=1.89 vs |rel@100|=1.77; extra golds in 100-200
can be packed into top-10 and push turn nDCG@10 over 0.8.
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
OUT = Path("results/finetune-reranker/minilm_pack_exploded.md")
RERANKER = "ms-marco-minilm-l6-v2-ft-listwise"
POOL = 100


def _group_by_admission(items: list) -> list:
    """Passage-to-document firstP: keep CE order, but pack later turns of
    an earlier admission ahead of other admissions. Helps admission nDCG
    when the answering admission already won rank-1.
    """
    order: list[str] = []
    buckets: dict[str, list] = {}
    for it in items:
        sid = it.session_id
        if sid not in buckets:
            order.append(sid)
            buckets[sid] = []
        buckets[sid].append(it)
    out: list = []
    for sid in order:
        out.extend(buckets[sid])
    return out


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
    adm: list[dict] = []
    turn: list[dict] = []
    adm_g: list[dict] = []
    turn_g: list[dict] = []
    for pid in TRIO:
        dense, lex = indexes[pid]
        for item in load_qa(pid):
            ev = item.get("evidence") or {}
            if not ev.get("admissions"):
                continue
            cands = hybrid_search(item["question"], dense, lex, k=POOL, per_arm=POOL)
            ranked = reranker.rerank(item["question"], [c.text for c in cands], top_k=None)
            reranked = [cands[i] for i, _ in ranked]
            grouped = _group_by_admission(reranked)
            adm_ev = admission_only_evidence(ev)

            def rec(evidence: dict) -> dict:
                return {
                    "hit": {k: hit_at_k(reranked, evidence, k) for k in RERANK_KS},
                    "ndcg": {k: ndcg_at_k(reranked, evidence, k) for k in RERANK_KS},
                    "rel10": sum(1 for c in reranked[:10] if hit_at_k([c], evidence, 1)),
                    "rel_pool": sum(1 for c in reranked if hit_at_k([c], evidence, 1)),
                }

            adm.append(rec(adm_ev))
            adm_g.append(
                {
                    "hit": {k: hit_at_k(grouped, adm_ev, k) for k in RERANK_KS},
                    "ndcg": {k: ndcg_at_k(grouped, adm_ev, k) for k in RERANK_KS},
                    "rel10": sum(1 for c in grouped[:10] if hit_at_k([c], adm_ev, 1)),
                    "rel_pool": sum(1 for c in grouped if hit_at_k([c], adm_ev, 1)),
                }
            )
            trn = turn_only_evidence(ev)
            if trn is not None:
                turn.append(rec(trn))
                turn_g.append(
                    {
                        "hit": {k: hit_at_k(grouped, trn, k) for k in RERANK_KS},
                        "ndcg": {k: ndcg_at_k(grouped, trn, k) for k in RERANK_KS},
                        "rel10": sum(1 for c in grouped[:10] if hit_at_k([c], trn, 1)),
                        "rel_pool": sum(1 for c in grouped if hit_at_k([c], trn, 1)),
                    }
                )
    ta, aa = _agg(turn), _agg(adm)
    tg, ag = _agg(turn_g), _agg(adm_g)
    body = f"""# MiniLM-listwise CPU, hybrid pool {POOL}

## Turn n={ta['n']}
Hit@2/5/10/20 = {ta['hit@2']:.3f} / {ta['hit@5']:.3f} / {ta['hit@10']:.3f} / {ta['hit@20']:.3f}
nDCG@2/5/10/20 = {ta['ndcg@2']:.3f} / {ta['ndcg@5']:.3f} / {ta['ndcg@10']:.3f} / {ta['ndcg@20']:.3f}
mean |rel@10|={ta['mean_rel10']:.2f} |rel@pool|={ta['mean_rel_pool']:.2f}

## Admission n={aa['n']}
Hit@2/5/10/20 = {aa['hit@2']:.3f} / {aa['hit@5']:.3f} / {aa['hit@10']:.3f} / {aa['hit@20']:.3f}
nDCG@2/5/10/20 = {aa['ndcg@2']:.3f} / {aa['ndcg@5']:.3f} / {aa['ndcg@10']:.3f} / {aa['ndcg@20']:.3f}
mean |rel@10|={aa['mean_rel10']:.2f} |rel@pool|={aa['mean_rel_pool']:.2f}

## Turn after grouping by admission (firstP)
Hit@2/5/10/20 = {tg['hit@2']:.3f} / {tg['hit@5']:.3f} / {tg['hit@10']:.3f} / {tg['hit@20']:.3f}
nDCG@2/5/10/20 = {tg['ndcg@2']:.3f} / {tg['ndcg@5']:.3f} / {tg['ndcg@10']:.3f} / {tg['ndcg@20']:.3f}
mean |rel@10|={tg['mean_rel10']:.2f}

## Admission after grouping by admission (firstP)
Hit@2/5/10/20 = {ag['hit@2']:.3f} / {ag['hit@5']:.3f} / {ag['hit@10']:.3f} / {ag['hit@20']:.3f}
nDCG@2/5/10/20 = {ag['ndcg@2']:.3f} / {ag['ndcg@5']:.3f} / {ag['ndcg@10']:.3f} / {ag['ndcg@20']:.3f}
mean |rel@10|={ag['mean_rel10']:.2f}
"""
    OUT.write_text(body, encoding="utf-8")
    print(body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
