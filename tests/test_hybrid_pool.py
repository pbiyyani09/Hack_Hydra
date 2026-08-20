"""Hybrid pool + graded labels. No models."""

from medmemgraph.contracts import RetrieveItem
from medmemgraph.eval.hybrid_pool import expanded_hybrid_search, fuse_dense_lexical, rm3_terms
from medmemgraph.eval.listwise_lists import GRADE_ADMISSION, GRADE_NEIGHBOR, GRADE_TURN, grade_item


def _it(sid: str, turns: list[int], text: str) -> RetrieveItem:
    return RetrieveItem(text=text, session_id=sid, turn_ids=turns, score=1.0, channel="vector")


def test_fuse_dedupes_overlapping_turns():
    dense = [_it("H1", [5], "dense-5"), _it("H1", [9], "dense-9")]
    lex = [_it("H1", [4, 5, 6], "window-5"), _it("H2", [1], "lex-h2")]
    fused = fuse_dense_lexical(dense, lex, k=10)
    keys = {(x.session_id, tuple(x.turn_ids)) for x in fused}
    # Window members become singleton turns so consecutive golds get
    # separate nDCG slots. Overlap with dense-5 does not drop 4 and 6.
    turns = {(x.session_id, x.turn_ids[0] if x.turn_ids else None) for x in fused}
    assert ("H1", 5) in turns
    assert ("H1", 9) in turns
    assert ("H2", 1) in turns
    assert ("H1", 4) in turns or ("H1", 6) in turns


def test_grades_turn_beats_admission():
    gold_turn = _it("H1", [5], "t5")
    neighbor = _it("H1", [6], "t6")
    sib = _it("H1", [9], "t9")
    other = _it("H2", [1], "x")
    ev = {"admissions": ["H1"], "turn_ids": [5]}
    assert grade_item(gold_turn, ev) == GRADE_TURN
    assert grade_item(neighbor, ev) == GRADE_NEIGHBOR
    assert grade_item(sib, ev) == GRADE_ADMISSION
    assert grade_item(other, ev) == 0.0


def test_rm3_skips_query_terms_and_stopwords():
    q = "was metformin started for diabetes"
    docs = [
        "metformin 500 mg diabetes nausea creatinine bump metformin metformin",
        "creatinine rose after metformin nausea nausea",
    ]
    terms = rm3_terms(q, docs, n_terms=5)
    assert "metformin" not in terms
    assert "diabetes" not in terms
    assert "the" not in terms
    assert "nausea" in terms
    assert "creatinine" in terms


def test_expanded_hybrid_falls_back_without_terms(monkeypatch):
    dense = [_it("H1", [5], "only")]
    lex = [_it("H1", [5], "only")]
    fused = fuse_dense_lexical(dense, lex, k=5)
    assert fused[0].session_id == "H1"
