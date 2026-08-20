"""How many gold turns enter the hybrid pool as k grows?"""

from __future__ import annotations

from medmemgraph.eval.hybrid_pool import build_indexes, hybrid_search
from medmemgraph.eval.metrics import hit_at_k
from medmemgraph.eval.retrieval_eval import admission_only_evidence, turn_only_evidence
from medmemgraph.pipeline.loader import load_conversation, load_qa

TRIO = ["10056223", "10213338", "10312715"]
KS = (10, 20, 50, 100, 200, 400)


def main() -> int:
    indexes = {}
    for pid in TRIO:
        indexes[pid] = build_indexes(pid, load_conversation(pid), dense_backend="arctic-s")
    turn = {k: [] for k in KS}
    adm = {k: [] for k in KS}
    n_turn = n_adm = 0
    n_gold_t = n_gold_a = 0
    for pid in TRIO:
        dense, lex = indexes[pid]
        for item in load_qa(pid):
            ev = item.get("evidence") or {}
            if not ev.get("admissions"):
                continue
            cands = hybrid_search(item["question"], dense, lex, k=400, per_arm=400)
            adm_ev = admission_only_evidence(ev)
            n_adm += 1
            n_gold_a += len(adm_ev.get("admissions") or [])
            for k in KS:
                adm[k].append(sum(1 for c in cands[:k] if hit_at_k([c], adm_ev, 1)))
            trn = turn_only_evidence(ev)
            if trn is None:
                continue
            n_turn += 1
            n_gold_t += len(trn.get("turn_ids") or [])
            for k in KS:
                turn[k].append(sum(1 for c in cands[:k] if hit_at_k([c], trn, 1)))
    print(f"turn n={n_turn} mean_gold={n_gold_t/n_turn:.2f}")
    for k in KS:
        rel = sum(turn[k]) / n_turn
        hit = sum(1 for x in turn[k] if x > 0) / n_turn
        print(f"  turn  k={k:3d} |rel|={rel:.2f} Hit={hit:.3f}")
    print(f"adm n={n_adm} mean_gold_adm={n_gold_a/n_adm:.2f}")
    for k in KS:
        rel = sum(adm[k]) / n_adm
        hit = sum(1 for x in adm[k] if x > 0) / n_adm
        print(f"  adm   k={k:3d} |rel|={rel:.2f} Hit={hit:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
