"""Offline pair-builder tests (FT-E1-S2). No arctic-s load."""

from __future__ import annotations

from medmemgraph.contracts import RetrieveItem
from medmemgraph.eval.rerank_pairs import build_pairs, is_gold_hit
from medmemgraph.eval.rerank_split import EVAL_TRIO
from medmemgraph.graph.vector_index import format_turn
from medmemgraph.pipeline.loader import Admission, Conversation


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


class _FakeIndex:
    def __init__(self, hits: list[RetrieveItem]):
        self._hits = hits
        self.built = False

    def build(self, conversation, facts=None) -> None:
        assert facts is None
        self.built = True

    def search(self, query: str, k: int) -> list[RetrieveItem]:
        return self._hits[:k]


def test_is_gold_hit_matches_admission_and_optional_turns():
    assert is_gold_hit("H1", [5], {"H1"}, {5})
    assert not is_gold_hit("H1", [9], {"H1"}, {5})
    assert is_gold_hit("H1", [9], {"H1"}, None)
    assert not is_gold_hit("H2", [5], {"H1"}, {5})


def test_turn_grain_gold_and_hardneg(tmp_path, monkeypatch):
    train_id = "train0001"
    convo = _convo(train_id, "H1", [(5, "gold text"), (9, "hardneg text")])
    gold_turn = convo.turns()[0]
    hard_turn = convo.turns()[1]
    hits = [
        RetrieveItem(
            text=format_turn(gold_turn),
            session_id="H1",
            turn_ids=[5],
            score=0.9,
            channel="vector",
        ),
        RetrieveItem(
            text=format_turn(hard_turn),
            session_id="H1",
            turn_ids=[9],
            score=0.8,
            channel="vector",
        ),
    ]
    qas = [
        {
            "qa_id": "q1",
            "question": "why metformin",
            "evidence": {"admissions": ["H1"], "turn_ids": [5]},
        }
    ]
    monkeypatch.setattr(
        "medmemgraph.eval.rerank_pairs.load_conversation", lambda sid, root=None: convo
    )
    monkeypatch.setattr("medmemgraph.eval.rerank_pairs.load_qa", lambda sid, root=None: qas)

    split = {
        "eval": list(EVAL_TRIO),
        "dev": [],
        "train": [train_id],
        "grain": "mixed-turn-preferred",
        "grain_rule": "test",
    }
    manifest = build_pairs(
        split,
        out_dir=tmp_path,
        index_factory=lambda sid: _FakeIndex(hits),
    )
    lines = (tmp_path / "train_pairs.jsonl").read_text(encoding="utf-8").strip().splitlines()
    recs = [__import__("json").loads(line) for line in lines]
    golds = [r for r in recs if r["kind"] == "gold"]
    negs = [r for r in recs if r["kind"] == "hardneg"]
    assert len(golds) == 1
    assert golds[0]["label"] == 1
    assert golds[0]["turn_number"] == 5
    assert golds[0]["passage"] == format_turn(gold_turn)
    assert len(negs) == 1
    assert negs[0]["label"] == 0
    assert negs[0]["turn_number"] == 9
    assert negs[0]["subject_id"] == train_id
    assert train_id not in EVAL_TRIO
    assert manifest["n_gold_turn"] == 1
    assert manifest["n_gold_admission_expanded"] == 0
    assert manifest["n_hardneg"] == 1
    for rec in recs:
        assert rec["subject_id"] not in EVAL_TRIO


def test_admission_only_item_expands_all_turns(tmp_path, monkeypatch):
    train_id = "train0002"
    convo = _convo(train_id, "H1", [(1, "t1"), (2, "t2")])
    qas = [
        {
            "qa_id": "q2",
            "question": "why admitted",
            "evidence": {"admissions": ["H1"]},
        }
    ]
    monkeypatch.setattr(
        "medmemgraph.eval.rerank_pairs.load_conversation", lambda sid, root=None: convo
    )
    monkeypatch.setattr("medmemgraph.eval.rerank_pairs.load_qa", lambda sid, root=None: qas)
    split = {"eval": list(EVAL_TRIO), "dev": [], "train": [train_id]}
    manifest = build_pairs(
        split,
        out_dir=tmp_path,
        index_factory=lambda sid: _FakeIndex([]),
    )
    recs = [
        __import__("json").loads(line)
        for line in (tmp_path / "train_pairs.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    golds = [r for r in recs if r["label"] == 1]
    assert {r["turn_number"] for r in golds} == {1, 2}
    assert manifest["n_gold_admission_expanded"] == 2
    assert manifest["n_gold_turn"] == 0
