"""Graded lists for listwise nDCG training. Eval trio never appears."""

from __future__ import annotations

from typing import Any

from medmemgraph.eval.hybrid_pool import hybrid_search
from medmemgraph.eval.metrics import _is_relevant, _parse_gold_evidence
from medmemgraph.eval.rerank_split import EVAL_TRIO
from medmemgraph.eval.retrieval_eval import admission_only_evidence, turn_only_evidence
from medmemgraph.graph.lexical import LexicalIndex
from medmemgraph.graph.vector_index import PatientIndex

# Gold turn > neighbor of gold turn > other gold-admission turn > rest.
# Neighbors must lose to the answering turn (turn nDCG@2) but still beat
# other-admission distractors.
GRADE_TURN = 3.0
GRADE_NEIGHBOR = 2.0
GRADE_ADMISSION = 1.0
GRADE_NONE = 0.0
NEIGHBOR_RADIUS = 1


def grade_item(item, evidence: dict[str, Any]) -> float:
    trn = turn_only_evidence(evidence)
    if trn is not None:
        gold = _parse_gold_evidence(trn)
        if _is_relevant(item, gold):
            return GRADE_TURN
        if item.session_id in gold.admissions and item.turn_ids and gold.turn_ids:
            if any(abs(int(t) - int(g)) <= NEIGHBOR_RADIUS for t in item.turn_ids for g in gold.turn_ids):
                return GRADE_NEIGHBOR
    adm = admission_only_evidence(evidence)
    if adm.get("admissions") and _is_relevant(item, _parse_gold_evidence(adm)):
        return GRADE_ADMISSION
    return GRADE_NONE


def lists_for_patient(
    subject_id: str,
    qas: list[dict[str, Any]],
    dense: PatientIndex,
    lex: LexicalIndex,
    *,
    list_size: int = 32,
) -> list[dict[str, Any]]:
    if subject_id in EVAL_TRIO:
        raise RuntimeError(f"eval-trio subject {subject_id!r}")
    out: list[dict[str, Any]] = []
    for item in qas:
        ev = item.get("evidence") or {}
        if not ev.get("admissions"):
            continue
        pool = hybrid_search(item["question"], dense, lex, k=list_size)
        if not pool:
            continue
        labels = [grade_item(hit, ev) for hit in pool]
        if max(labels) <= 0:
            continue
        out.append(
            {
                "subject_id": subject_id,
                "qa_id": str(item["qa_id"]),
                "question": str(item["question"]),
                "docs": [h.text for h in pool],
                "labels": labels,
            }
        )
    return out
