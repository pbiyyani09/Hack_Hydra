"""tests/test_agent.py — `demo/agent.py` (`stories/E8/E8-S1.md`).

Everything here runs offline via `dry_run=True` -- `chat_once`'s `dry_run`
kwarg threads straight through to `eval.reader.read()`'s own deterministic
stub (`_stub_complete`), which never calls `medmemgraph.llm.complete()`, so
these tests need no API key, no network, no HydraDB, and no
`isolate_llm_module`-style fixture (unlike `tests/test_reader.py`, which
also exercises the real-call code path -- this file only needs the stub
path). Each test file in this project is self-contained (see
`eval/baselines/dense.py`'s own docstring for the same convention): the
`_item`/`_pack` helpers below are duplicated rather than imported from
`tests/test_reader.py` / `tests/test_retrieve.py`.
"""

from __future__ import annotations

import pytest

from medmemgraph.contracts import RetrieveItem, RetrieveResult
from medmemgraph.demo import agent
from medmemgraph.demo.agent import _REFUSAL_LINE, chat_once, main


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _item(
    text: str, session_id: str = "H1", turn_ids: list[int] | None = None,
    score: float = 0.9, channel: str = "vector",
) -> RetrieveItem:
    return RetrieveItem(text=text, session_id=session_id, turn_ids=turn_ids or [1], score=score, channel=channel)


def _absent_pack() -> RetrieveResult:
    """A structural-absence pack -- matches `retrieve()`'s own real
    invariant (`items=[]` whenever `structural_absence=True`, see
    `graph/retrieve.py`'s every return site)."""
    return RetrieveResult(items=[], route="graph", structural_absence=True, paths=[], latency_ms={"total": 1.0})


def _metformin_pack() -> RetrieveResult:
    """E7-S5's own worked example, reused here at the `chat_once` layer."""
    item = _item("Patient takes metformin 500mg", session_id="H1", turn_ids=[4])
    return RetrieveResult(items=[item], route="graph", structural_absence=False, paths=[], latency_ms={"total": 2.0})


class _CountingRetriever:
    """A scripted retriever that returns a fixed pack and counts calls --
    the vehicle for AC1/AC2's "Given a scripted retrieve" and the "no
    second retrieve" assertion."""

    def __init__(self, pack: RetrieveResult) -> None:
        self.pack = pack
        self.calls: list[tuple[str, str, int]] = []

    def __call__(self, question: str, patient_id: str, k: int) -> RetrieveResult:
        self.calls.append((question, patient_id, k))
        return self.pack


# ---------------------------------------------------------------------------
# 1. chat_once's return shape (the frozen contract)
# ---------------------------------------------------------------------------


class TestChatOnceShape:
    def test_returns_exactly_answer_retrieve_con(self):
        retriever = _CountingRetriever(_metformin_pack())
        result = chat_once("what dose of metformin?", "p1", retriever=retriever, dry_run=True)

        assert set(result.keys()) == {"answer", "retrieve", "con"}
        assert isinstance(result["answer"], str)
        assert isinstance(result["retrieve"], RetrieveResult)
        assert isinstance(result["con"], dict)

    def test_con_dict_carries_notes_abstain_citations(self):
        retriever = _CountingRetriever(_metformin_pack())
        result = chat_once("what dose of metformin?", "p1", retriever=retriever, dry_run=True)

        con = result["con"]
        assert set(con.keys()) >= {"notes", "abstain", "citations"}
        assert isinstance(con["notes"], list)
        assert all(isinstance(n, str) for n in con["notes"])
        assert isinstance(con["abstain"], bool)
        assert isinstance(con["citations"], list)

    def test_default_k_is_eight(self):
        retriever = _CountingRetriever(_metformin_pack())
        chat_once("q", "p1", retriever=retriever, dry_run=True)
        assert retriever.calls[0][2] == 8


# ---------------------------------------------------------------------------
# 2. AC1 -- structural_absence -> refusal, not a guessed fact.
# ---------------------------------------------------------------------------


class TestAbstentionPath:
    def test_structural_absence_forces_refusal_answer(self):
        retriever = _CountingRetriever(_absent_pack())
        result = chat_once("does the patient have a pacemaker?", "p1", retriever=retriever, dry_run=True)

        assert result["answer"] == _REFUSAL_LINE
        assert result["retrieve"].structural_absence is True
        assert result["con"]["abstain"] is True

    def test_structural_absence_does_not_confabulate_a_fact(self):
        retriever = _CountingRetriever(_absent_pack())
        result = chat_once("does the patient have a pacemaker?", "p1", retriever=retriever, dry_run=True)

        # The refusal must not mention pacemaker, or any invented clinical
        # content -- it is the fixed refusal line, nothing else.
        assert "pacemaker" not in result["answer"].lower()
        assert result["answer"] == "Not in this record"

    def test_structural_absence_true_forces_abstain_even_with_stray_items(self):
        # Defensive: even if a future retriever bug leaves items non-empty
        # alongside structural_absence=True, the reader's own stub (and
        # its real prompt -- STRUCTURAL_ABSENCE is embedded verbatim,
        # `reader.py::_build_user_content`) still forces abstention.
        stray_pack = RetrieveResult(
            items=[_item("unrelated evidence")], route="graph", structural_absence=True,
            paths=[], latency_ms={"total": 0.0},
        )
        retriever = _CountingRetriever(stray_pack)
        result = chat_once("q", "p1", retriever=retriever, dry_run=True)
        assert result["answer"] == _REFUSAL_LINE
        assert result["con"]["abstain"] is True

    def test_refusal_prints_the_literal_not_in_this_record_line(self, capsys):
        retriever = _CountingRetriever(_absent_pack())
        result = chat_once("does the patient have a pacemaker?", "p1", retriever=retriever, dry_run=True)
        agent._print_result(result)

        out = capsys.readouterr().out
        lines = [ln.strip() for ln in out.splitlines()]
        assert _REFUSAL_LINE in lines
        assert "structural_absence=true" in out


# ---------------------------------------------------------------------------
# 3. AC2 -- one retrieve call, pack-only answer, session_id citation.
# ---------------------------------------------------------------------------


class TestNormalAnswerPath:
    def test_uses_pack_only_no_second_retrieve(self):
        retriever = _CountingRetriever(_metformin_pack())
        chat_once("what dose of metformin does the patient take?", "p1", retriever=retriever, dry_run=True)

        assert len(retriever.calls) == 1

    def test_citations_include_the_items_session_id(self):
        retriever = _CountingRetriever(_metformin_pack())
        result = chat_once("what dose of metformin does the patient take?", "p1", retriever=retriever, dry_run=True)

        citations = result["con"]["citations"]
        assert citations, "expected at least one citation for a relevant item"
        assert any(c["session_id"] == "H1" and c["turn_ids"] == [4] for c in citations)

    def test_answer_is_not_the_refusal_when_evidence_is_relevant(self):
        retriever = _CountingRetriever(_metformin_pack())
        result = chat_once("what dose of metformin does the patient take?", "p1", retriever=retriever, dry_run=True)

        assert result["answer"] != _REFUSAL_LINE
        assert result["con"]["abstain"] is False
        assert "metformin" in result["answer"].lower()

    def test_route_and_citations_are_legible_in_printed_output(self, capsys):
        retriever = _CountingRetriever(_metformin_pack())
        result = chat_once("what dose of metformin does the patient take?", "p1", retriever=retriever, dry_run=True)
        agent._print_result(result)

        out = capsys.readouterr().out
        assert "route=graph" in out
        assert "H1" in out
        assert "[4]" in out


# ---------------------------------------------------------------------------
# 4. con.abstain alone (no structural_absence) still triggers the refusal.
# ---------------------------------------------------------------------------


class TestReaderOwnAbstentionAlsoRefuses:
    def test_no_overlapping_evidence_abstains_even_without_structural_absence(self):
        # structural_absence=False, but nothing in the pack bears on the
        # question -- the dry-run stub's own content-word filter abstains,
        # and chat_once must still surface a refusal (`con.abstain`).
        item = _item("Reports mild seasonal allergies, treated with loratadine.", session_id="H2", turn_ids=[9])
        pack = RetrieveResult(items=[item], route="vector", structural_absence=False, paths=[], latency_ms={"total": 0.0})
        retriever = _CountingRetriever(pack)

        result = chat_once("what is the cardiac ejection fraction?", "p1", retriever=retriever, dry_run=True)

        assert result["con"]["abstain"] is True
        assert result["answer"] == _REFUSAL_LINE


# ---------------------------------------------------------------------------
# 5. AC3 -- --help documents --patient, no HydraDB touch.
# ---------------------------------------------------------------------------


class TestHelpDoesNotRequireHydraDB:
    def test_help_documents_patient_flag(self, capsys, monkeypatch):
        def _boom(*_args, **_kwargs):
            raise AssertionError("retrieve() must not be called for --help")

        monkeypatch.setattr(agent, "graph_retrieve", _boom)

        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])

        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "--patient" in out

    def test_missing_required_patient_flag_exits_nonzero_without_hydradb(self, monkeypatch, capsys):
        def _boom(*_args, **_kwargs):
            raise AssertionError("retrieve() must not be called")

        monkeypatch.setattr(agent, "graph_retrieve", _boom)

        with pytest.raises(SystemExit) as exc_info:
            main([])

        assert exc_info.value.code != 0


# ---------------------------------------------------------------------------
# 6. main()'s prompt loop, offline via --dry-run and a scripted input().
# ---------------------------------------------------------------------------


class TestPromptLoop:
    def test_loop_answers_then_exits_on_exit_command(self, monkeypatch, capsys):
        answers = iter(["what dose of metformin does the patient take?", "exit"])
        monkeypatch.setattr("builtins.input", lambda *_a: next(answers))

        retriever = _CountingRetriever(_metformin_pack())
        monkeypatch.setattr(agent, "graph_retrieve", retriever)

        exit_code = main(["--patient", "p1", "--dry-run"])

        assert exit_code == 0
        assert len(retriever.calls) == 1
        out = capsys.readouterr().out
        assert "MedMemGraph chat" in out
        assert "route=graph" in out

    def test_loop_exits_cleanly_on_eof(self, monkeypatch, capsys):
        def _raise_eof(*_a):
            raise EOFError

        monkeypatch.setattr("builtins.input", _raise_eof)
        exit_code = main(["--patient", "p1", "--dry-run"])
        assert exit_code == 0

    def test_loop_skips_blank_lines_without_retrieving(self, monkeypatch):
        answers = iter(["   ", "exit"])
        monkeypatch.setattr("builtins.input", lambda *_a: next(answers))

        retriever = _CountingRetriever(_metformin_pack())
        monkeypatch.setattr(agent, "graph_retrieve", retriever)

        main(["--patient", "p1", "--dry-run"])
        assert len(retriever.calls) == 0

    def test_loop_survives_a_raising_chat_once(self, monkeypatch, capsys):
        answers = iter(["question one", "exit"])
        monkeypatch.setattr("builtins.input", lambda *_a: next(answers))

        def _boom(*_args, **_kwargs):
            raise RuntimeError("simulated LLM failure")

        monkeypatch.setattr(agent, "chat_once", _boom)

        exit_code = main(["--patient", "p1", "--dry-run"])

        assert exit_code == 0
        out = capsys.readouterr().out
        assert "error:" in out
