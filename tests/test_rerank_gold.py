"""Graded gold extract. No corpus IO beyond in-memory Conversation."""

from medmemgraph.eval.rerank_gold import GRADE_NEIGHBOR, GRADE_TURN, gold_record
from medmemgraph.eval.rerank_split import EVAL_TRIO
from medmemgraph.pipeline.loader import Admission, Conversation
import pytest


def _convo(subject_id: str, hadm: str, turns: list[tuple[int, str]]) -> Conversation:
    lines = tuple(
        {
            "turn_number": n,
            "time": "2124-01-01 00:00:00",
            "speaker": "Doctor",
            "text": text,
        }
        for n, text in turns
    )
    return Conversation(
        subject_id=subject_id,
        processed_hadm_ids=(hadm,),
        admissions=(
            Admission(
                hadm_id=hadm,
                admission_start="2124-01-01 00:00:00",
                admission_end="2124-01-02 00:00:00",
                conversation_lines=lines,
            ),
        ),
    )


def test_refuses_eval_trio():
    convo = _convo(EVAL_TRIO[0], "H1", [(1, "a")])
    qa = {"qa_id": "x", "question": "q", "evidence": {"admissions": ["H1"], "turn_ids": [1]}}
    with pytest.raises(RuntimeError, match="eval-trio"):
        gold_record(EVAL_TRIO[0], qa, convo, split="train")


def test_turn_gold_gets_grade_3_and_neighbors_2():
    convo = _convo("10569306", "H1", [(1, "a"), (2, "gold"), (3, "c"), (9, "far")])
    qa = {
        "qa_id": "q1",
        "question": "what",
        "answer": "gold",
        "question_type": "medical_reasoning",
        "scope": "single_admission",
        "evidence": {"admissions": ["H1"], "turn_ids": [2]},
    }
    rec = gold_record("10569306", qa, convo, split="train")
    assert rec is not None
    grades = {(u["turn_number"], u["grade"], u["kind"]) for u in rec["units"]}
    assert (2, GRADE_TURN, "gold_turn") in grades
    assert (1, GRADE_NEIGHBOR, "neighbor") in grades
    assert (3, GRADE_NEIGHBOR, "neighbor") in grades
    assert all(u["turn_number"] != 9 for u in rec["units"])
    assert {"hadm_id": "H1", "turn_number": 9} in rec["same_admission_other"]


def test_admission_only_has_no_grade_3():
    convo = _convo("10569306", "H1", [(1, "a"), (2, "b")])
    qa = {
        "qa_id": "q2",
        "question": "q",
        "evidence": {"admissions": ["H1"]},
    }
    rec = gold_record("10569306", qa, convo, split="dev")
    assert rec is not None
    assert rec["gold_turn_ids"] == []
    assert rec["units"] == []
    assert rec["n_same_admission_other"] == 2
