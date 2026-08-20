"""Canonical rerank gold extracted from allowlisted MedLoCoMo files.

Readers are ``load_qa`` / ``load_conversation`` only. Eval trio is refused.
Does not open formed_packet.json or any per-admission artifact.

Grades (packing, not extra patients):
  3  every evidence.turn_id (must-cite answering turn)
  2  same-admission neighbor (±1 turn) of a gold turn
  1  other turns of gold admissions (implicit; listed as ids only)
  0  everything else (not stored)

Admission-only items have no grade-3 units; gold-admission turns are grade 1.
"""

from __future__ import annotations

from typing import Any

from medmemgraph.eval.rerank_split import EVAL_TRIO
from medmemgraph.graph.vector_index import format_turn
from medmemgraph.pipeline.loader import Conversation, load_conversation, load_qa

GRADE_TURN = 3
GRADE_NEIGHBOR = 2
GRADE_ADMISSION = 1
NEIGHBOR_RADIUS = 1


def gold_record(
    subject_id: str,
    qa: dict[str, Any],
    conversation: Conversation,
    *,
    split: str,
) -> dict[str, Any] | None:
    if subject_id in EVAL_TRIO:
        raise RuntimeError(f"eval-trio subject {subject_id!r}")
    ev = qa.get("evidence") or {}
    admissions = [str(a) for a in (ev.get("admissions") or [])]
    if not admissions:
        return None
    adm_set = set(admissions)
    raw_turns = ev.get("turn_ids")
    gold_turn_ids = {int(t) for t in raw_turns} if raw_turns else set()

    turns = [t for t in conversation.turns() if t.hadm_id in adm_set]
    by_hadm: dict[str, list] = {}
    for t in turns:
        by_hadm.setdefault(t.hadm_id, []).append(t)
    for lst in by_hadm.values():
        lst.sort(key=lambda x: x.turn_number)

    units: list[dict[str, Any]] = []
    neighbor_keys: set[tuple[str, int]] = set()
    if gold_turn_ids:
        gold_keys = {(t.hadm_id, t.turn_number) for t in turns if t.turn_number in gold_turn_ids}
        for t in turns:
            if t.turn_number not in gold_turn_ids:
                continue
            units.append(
                {
                    "hadm_id": t.hadm_id,
                    "turn_number": t.turn_number,
                    "grade": GRADE_TURN,
                    "kind": "gold_turn",
                    "text": format_turn(t),
                }
            )
            nums = [x.turn_number for x in by_hadm.get(t.hadm_id, [])]
            for n in nums:
                if abs(n - t.turn_number) <= NEIGHBOR_RADIUS and (t.hadm_id, n) not in gold_keys:
                    neighbor_keys.add((t.hadm_id, n))
        for t in turns:
            if (t.hadm_id, t.turn_number) not in neighbor_keys:
                continue
            units.append(
                {
                    "hadm_id": t.hadm_id,
                    "turn_number": t.turn_number,
                    "grade": GRADE_NEIGHBOR,
                    "kind": "neighbor",
                    "text": format_turn(t),
                }
            )
    same_adm = [
        {"hadm_id": t.hadm_id, "turn_number": t.turn_number}
        for t in turns
        if t.turn_number not in gold_turn_ids
        and (t.hadm_id, t.turn_number) not in neighbor_keys
    ]
    return {
        "subject_id": subject_id,
        "qa_id": str(qa["qa_id"]),
        "split": split,
        "question": str(qa["question"]),
        "answer": str(qa.get("answer") or ""),
        "question_type": str(qa.get("question_type") or ""),
        "scope": str(qa.get("scope") or ""),
        "gold_admissions": admissions,
        "gold_turn_ids": sorted(gold_turn_ids),
        "units": units,
        "same_admission_other": same_adm,
        "n_same_admission_other": len(same_adm),
    }


def records_for_patient(subject_id: str, *, split: str) -> list[dict[str, Any]]:
    if subject_id in EVAL_TRIO:
        raise RuntimeError(f"eval-trio subject {subject_id!r}")
    convo = load_conversation(subject_id)
    out: list[dict[str, Any]] = []
    for qa in load_qa(subject_id):
        rec = gold_record(subject_id, qa, convo, split=split)
        if rec is not None:
            out.append(rec)
    return out
