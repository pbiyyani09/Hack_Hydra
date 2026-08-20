"""First-stage Hit/nDCG ceiling: dense vs hybrid, no reranker.

CPU-only by default (arctic-s). Pass --gpu to add qwen3-0.6b.
Eval trio only. retrieve.py is not imported.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from medmemgraph.eval.hybrid_pool import build_indexes, expanded_hybrid_search, hybrid_search
from medmemgraph.eval.metrics import hit_at_k, ndcg_at_k
from medmemgraph.eval.retrieval_eval import (
    RERANK_KS,
    admission_only_evidence,
    turn_only_evidence,
)
from medmemgraph.graph.vector_index import PatientIndex
from medmemgraph.pipeline.loader import load_conversation, load_qa

TRIO = ["10056223", "10213338", "10312715"]
OUT = Path("results/finetune-reranker/first_stage_ceiling.md")
JSON_OUT = Path("results/finetune-reranker/first_stage_ceiling.json")


def _agg(rows: list[dict]) -> dict:
    n = len(rows)
    out = {"n": n}
    for k in RERANK_KS:
        out[f"hit@{k}"] = sum(r["hit"][k] for r in rows) / n
        out[f"ndcg@{k}"] = sum(r["ndcg"][k] for r in rows) / n
        if out[f"ndcg@{k}"] > 1.0 + 1e-6:
            raise RuntimeError(f"nDCG@{k} > 1")
    out["mean_n_rel_pool"] = sum(r["n_rel_pool"] for r in rows) / n
    out["mean_n_gold"] = sum(r["n_gold"] for r in rows) / n
    out["mean_n_rel10"] = sum(r["n_rel10"] for r in rows) / n
    return out


def run(embedder: str, hybrid: bool, *, expanded: bool = False) -> dict:
    indexes = {}
    for pid in TRIO:
        convo = load_conversation(pid)
        if hybrid:
            indexes[pid] = build_indexes(pid, convo, dense_backend=embedder)
        else:
            idx = PatientIndex(pid, backend=embedder, cache_path=None)
            idx.build(convo)
            indexes[pid] = idx

    adm: list[dict] = []
    turn: list[dict] = []
    for pid in TRIO:
        for item in load_qa(pid):
            ev = item.get("evidence") or {}
            if not ev.get("admissions"):
                continue
            if hybrid:
                dense, lex = indexes[pid]
                if expanded:
                    cands = expanded_hybrid_search(item["question"], dense, lex, k=100)
                else:
                    cands = hybrid_search(item["question"], dense, lex, k=100)
            else:
                cands = indexes[pid].search(item["question"], k=100)
            adm_ev = admission_only_evidence(ev)
            rec = {
                "hit": {k: hit_at_k(cands, adm_ev, k) for k in RERANK_KS},
                "ndcg": {k: ndcg_at_k(cands, adm_ev, k) for k in RERANK_KS},
                "n_rel_pool": sum(
                    1
                    for c in cands
                    if hit_at_k([c], adm_ev, 1)
                ),
                "n_rel10": sum(1 for c in cands[:10] if hit_at_k([c], adm_ev, 1)),
                "n_gold": len((adm_ev.get("admissions") or [])),
            }
            adm.append(rec)
            trn = turn_only_evidence(ev)
            if trn is not None:
                turn.append(
                    {
                        "hit": {k: hit_at_k(cands, trn, k) for k in RERANK_KS},
                        "ndcg": {k: ndcg_at_k(cands, trn, k) for k in RERANK_KS},
                        "n_rel_pool": sum(1 for c in cands if hit_at_k([c], trn, 1)),
                        "n_rel10": sum(1 for c in cands[:10] if hit_at_k([c], trn, 1)),
                        "n_gold": len(trn.get("turn_ids") or []),
                    }
                )
    return {
        "embedder": embedder,
        "hybrid": hybrid,
        "expanded": expanded,
        "admission": _agg(adm),
        "turn": _agg(turn),
    }


def _line(arm: dict, grain: str) -> str:
    a = arm[grain]
    cells = [
        f"{arm['embedder']}+{'hyb' if arm['hybrid'] else 'vec'}{'+rm3' if arm.get('expanded') else ''}",
        str(a["n"]),
        f"{a['mean_n_gold']:.2f}",
        f"{a['mean_n_rel_pool']:.2f}",
        f"{a['mean_n_rel10']:.2f}",
    ]
    for k in RERANK_KS:
        cells.append(f"{a[f'hit@{k}']:.3f}")
    for k in RERANK_KS:
        cells.append(f"{a[f'ndcg@{k}']:.3f}")
    return "| " + " | ".join(cells) + " |"


def main() -> int:
    use_gpu = "--gpu" in sys.argv
    arms = []
    print("CPU arctic-s dense…", flush=True)
    arms.append(run("arctic-s", False))
    print("CPU arctic-s hybrid…", flush=True)
    arms.append(run("arctic-s", True))
    print("CPU arctic-s hybrid+RM3…", flush=True)
    arms.append(run("arctic-s", True, expanded=True))
    if use_gpu:
        print("GPU qwen3-0.6b dense…", flush=True)
        arms.append(run("qwen3-0.6b", False))
        print("GPU qwen3-0.6b hybrid…", flush=True)
        arms.append(run("qwen3-0.6b", True))

    hdr = (
        "| stack | n | |gold| | |rel@100| | |rel@10| | "
        "Hit@2 | Hit@5 | Hit@10 | Hit@20 | "
        "nDCG@2 | nDCG@5 | nDCG@10 | nDCG@20 |"
    )
    sep = "|---|---|---|---|---|---|---|---|---|---|---|---|---|"
    body = f"""# First-stage ceiling (no reranker)

Hybrid = dense + BM25 RRF (k=60), pool 100, turn-deduped.
nDCG@10 ≥ 0.8 is impossible if |rel@100| is far below |gold| or
if |rel@10| stays sparse even after fusion.

## Turn

{hdr}
{sep}
{chr(10).join(_line(a, "turn") for a in arms)}

## Admission

{hdr}
{sep}
{chr(10).join(_line(a, "admission") for a in arms)}
"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(body, encoding="utf-8")
    JSON_OUT.write_text(json.dumps(arms, indent=2) + "\n", encoding="utf-8")
    print(body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
