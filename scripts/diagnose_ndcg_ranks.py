"""Where do relevant turns sit? Explains Hit@10 high / nDCG@10 ~0.5."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from medmemgraph.eval.metrics import _is_relevant, _parse_gold_evidence, ndcg_at_k
from medmemgraph.eval.retrieval_eval import (
    N_CANDIDATES_FOR_RERANK,
    admission_only_evidence,
    build_reranker,
    turn_only_evidence,
)
from medmemgraph.graph.vector_index import PatientIndex
from medmemgraph.pipeline.loader import load_conversation, load_qa

TRIO = ["10056223", "10213338", "10312715"]
OUT = Path("results/finetune-reranker/ndcg_rank_diagnosis.md")

BUCKETS = ((1, 1), (2, 2), (3, 5), (6, 10), (11, 20), (21, 100))


def _bucket(rank: int | None) -> str:
    if rank is None:
        return "miss"
    for a, b in BUCKETS:
        if a <= rank <= b:
            return f"{a}" if a == b else f"{a}-{b}"
    return "miss"


def _first_rank(reranked, gold) -> int | None:
    for i, item in enumerate(reranked, 1):
        if _is_relevant(item, gold):
            return i
    return None


def _n_rel(reranked, gold, k: int) -> int:
    return sum(1 for item in reranked[:k] if _is_relevant(item, gold))


def run_arm(embedder: str, reranker_name: str) -> dict:
    indexes = {}
    for pid in TRIO:
        idx = PatientIndex(pid, backend=embedder, cache_path=None)
        idx.build(load_conversation(pid))
        indexes[pid] = idx
    reranker = build_reranker(reranker_name)
    if reranker_name != "noop":
        reranker.rerank("warmup", ["warmup"])

    rows_adm: list[dict] = []
    rows_turn: list[dict] = []
    for pid in TRIO:
        for item in load_qa(pid):
            ev = item.get("evidence") or {}
            if not ev.get("admissions"):
                continue
            retrieved = indexes[pid].search(item["question"], k=500)
            cands = retrieved[:N_CANDIDATES_FOR_RERANK]
            ranked = reranker.rerank(item["question"], [c.text for c in cands], top_k=None)
            reranked = [cands[i] for i, _ in ranked]
            adm_gold = _parse_gold_evidence(admission_only_evidence(ev))
            rec = {
                "first": _first_rank(reranked, adm_gold),
                "n_rel10": _n_rel(reranked, adm_gold, 10),
                "n_rel_pool": _n_rel(reranked, adm_gold, len(reranked)),
                "ndcg10": ndcg_at_k(reranked, admission_only_evidence(ev), 10),
                "n_gold": len(adm_gold.admissions),
            }
            rows_adm.append(rec)
            trn_ev = turn_only_evidence(ev)
            if trn_ev is not None:
                trn_gold = _parse_gold_evidence(trn_ev)
                rows_turn.append(
                    {
                        "first": _first_rank(reranked, trn_gold),
                        "n_rel10": _n_rel(reranked, trn_gold, 10),
                        "n_rel_pool": _n_rel(reranked, trn_gold, len(reranked)),
                        "ndcg10": ndcg_at_k(reranked, trn_ev, 10),
                        "n_gold": len(trn_gold.turn_ids or ()),
                    }
                )
    return {"admission": rows_adm, "turn": rows_turn}


def _summarize(rows: list[dict]) -> dict:
    n = len(rows)
    firsts = Counter(_bucket(r["first"]) for r in rows)
    hits = [r for r in rows if r["first"] is not None and r["first"] <= 10]
    return {
        "n": n,
        "hit10": sum(1 for r in rows if r["first"] is not None and r["first"] <= 10) / n,
        "ndcg10": sum(r["ndcg10"] for r in rows) / n,
        "mean_n_rel10": sum(r["n_rel10"] for r in rows) / n,
        "mean_n_rel10_given_hit": (sum(r["n_rel10"] for r in hits) / len(hits)) if hits else 0.0,
        "mean_n_rel_pool": sum(r["n_rel_pool"] for r in rows) / n,
        "mean_n_gold": sum(r["n_gold"] for r in rows) / n,
        "mean_first_given_hit": (sum(r["first"] for r in hits) / len(hits)) if hits else None,
        "first_rank_hist": {k: firsts.get(k, 0) / n for k in ["1", "2", "3-5", "6-10", "11-20", "miss"]},
    }


def _hist_line(h: dict) -> str:
    return " | ".join(f"{k}={v*100:.1f}%" for k, v in h.items())


def main() -> int:
    arms = [
        ("qwen3-0.6b", "qwen3-rerank-0.6b", "GPU Qwen"),
        ("arctic-s", "ms-marco-minilm-l6-v2-onnx-int8", "CPU MiniLM"),
        ("arctic-s", "ms-marco-minilm-l6-v2-ft-orpo-onnx-int8", "CPU MiniLM-ORPO"),
    ]
    chunks = [
        "# Why nDCG@10 is ~0.5 while Hit@10 is 0.90+",
        "",
        "Hit@10 = at least one relevant turn in the top-10.",
        "nDCG@10 = those relevant turns must occupy the *best* ranks.",
        "First-rank histogram is the share of questions whose first relevant turn sits at that rank.",
        "",
    ]
    dump = {}
    for emb, rer, label in arms:
        print(f"running {label}…", flush=True)
        data = run_arm(emb, rer)
        adm = _summarize(data["admission"])
        trn = _summarize(data["turn"])
        dump[label] = {"admission": adm, "turn": trn}
        chunks += [
            f"## {label} (`{emb}` + `{rer}`)",
            "",
            f"### Turn (n={trn['n']}, mean |gold turns|={trn['mean_n_gold']:.2f})",
            f"- Hit@10={trn['hit10']:.3f}  nDCG@10={trn['ndcg10']:.3f}",
            f"- mean first-relevant rank | hit={trn['mean_first_given_hit']:.2f}",
            f"- mean gold turns in top-10={trn['mean_n_rel10']:.2f} (given hit {trn['mean_n_rel10_given_hit']:.2f}); in 100-pool {trn['mean_n_rel_pool']:.2f}",
            f"- first-relevant rank: {_hist_line(trn['first_rank_hist'])}",
            "",
            f"### Admission (n={adm['n']}, mean |gold admissions|={adm['mean_n_gold']:.2f})",
            f"- Hit@10={adm['hit10']:.3f}  nDCG@10={adm['ndcg10']:.3f}",
            f"- mean first-relevant rank | hit={adm['mean_first_given_hit']:.2f}",
            f"- mean gold-admission *turns* in top-10={adm['mean_n_rel10']:.2f} (given hit {adm['mean_n_rel10_given_hit']:.2f}); in 100-pool {adm['mean_n_rel_pool']:.2f}",
            f"- first-relevant rank: {_hist_line(adm['first_rank_hist'])}",
            "",
        ]
    OUT.write_text("\n".join(chunks), encoding="utf-8")
    Path("results/finetune-reranker/ndcg_rank_diagnosis.json").write_text(
        json.dumps(dump, indent=2) + "\n", encoding="utf-8"
    )
    print("\n".join(chunks))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
