"""tests/test_scale_gate.py — `pipeline/scale_gate.py` (E2-S3) and the gate's
wiring into `pipeline/ingest.py` (E4-S4; decisions/005 Findings 2/3).

Every test in this file uses an isolated `tmp_path` fixture directory for
the gate, or monkeypatches `scale_gate.DEFAULT_HANDCHECK_DIR` at the real
`ingest_patient()` call site — **no test in this file ever writes
`fixtures/handcheck/PASSED`** (the hard rule this story states explicitly:
"No test may create PASSED as a side effect that leaks — use a tmp
fixture"). The one test that exercises the *real* default directory
(`test_default_handcheck_dir_is_currently_blocked`) only reads it — the real
`fixtures/handcheck/PENDING` marker (not `PASSED`) is what makes that
assertion true today, and this test does not (and structurally cannot)
change that.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from medmemgraph.contracts import EntityRef, mock_fact
from medmemgraph.graph.invalidate import InvalidationReport
from medmemgraph.graph.writer import WriteReport
from medmemgraph.pipeline import ingest
from medmemgraph.pipeline import scale_gate
from medmemgraph.pipeline.loader import Conversation
from medmemgraph.pipeline.scale_gate import (
    MIN_FACTS,
    ScaleGateError,
    assert_handcheck_passed,
    fact_from_dict,
    load_facts_jsonl,
)

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _valid_fact(i: int):
    """One `contracts.validate()`-clean `ClinicalFact`, `canonical_id=0` on
    both `subject`/`object` (E2-S3 AC2's own carve-out: 0 must be
    ACCEPTED — this is the post-mint-not-yet-run state a real hand-check
    run's facts are in) — every 4th row negated, alternating source_class,
    so the generated fixture also satisfies "at least one negated" / "at
    least two source_class values" the same way a real CHECKLIST.md must."""
    return mock_fact(
        fact_id=f"handcheck-fact-{i:05d}",
        patient_id="handcheck-patient",
        session_id="handcheck-admission",
        turn_ids=[i],
        subject=EntityRef(name="handcheck-patient", type="Patient", canonical_id=0),
        object=EntityRef(name=f"entity-{i}", type="Medication", canonical_id=0),
        polarity="negated" if i % 4 == 0 else "asserted",
        source_class="doctor" if i % 2 == 0 else "patient",
    )


def _write_facts_jsonl(path: Path, facts) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for fact in facts:
            fh.write(json.dumps(dataclasses.asdict(fact)))
            fh.write("\n")


def _write_passed(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("passed-by: test-fixture  date: 2026-08-16\n", encoding="utf-8")


def _green_handcheck_dir(tmp_path: Path, *, n_facts: int = MIN_FACTS) -> Path:
    """A fully-green, isolated `tmp_path` hand-check directory: `PASSED` +
    `facts.jsonl` with `n_facts` valid rows. Never touches the real
    `fixtures/handcheck/`."""
    d = tmp_path / "handcheck"
    facts = [_valid_fact(i) for i in range(n_facts)]
    _write_facts_jsonl(d / "facts.jsonl", facts)
    _write_passed(d / "PASSED")
    return d


# ---------------------------------------------------------------------------
# assert_handcheck_passed() — AC1: blocks with no PASSED.
# ---------------------------------------------------------------------------


class TestGateBlocksWithNoPassed:
    def test_missing_passed_raises_even_with_valid_facts_jsonl(self, tmp_path: Path) -> None:
        d = tmp_path / "handcheck"
        _write_facts_jsonl(d / "facts.jsonl", [_valid_fact(i) for i in range(MIN_FACTS)])
        # PASSED deliberately never written.
        with pytest.raises(ScaleGateError, match="PASSED"):
            assert_handcheck_passed(handcheck_dir=d)

    def test_missing_passed_and_missing_facts_jsonl_raises(self, tmp_path: Path) -> None:
        d = tmp_path / "handcheck"
        d.mkdir()
        with pytest.raises(ScaleGateError):
            assert_handcheck_passed(handcheck_dir=d)

    def test_default_handcheck_dir_state_matches_whether_passed_exists(self) -> None:
        """No `handcheck_dir=` override — exercises the REAL default directory
        (`fixtures/handcheck/`).

        Asserts the RULE, not a snapshot of the repo. The previous version of
        this test asserted the default dir is always blocked, which was true
        only while the repo shipped an unsigned gate; it started failing the
        moment a human legitimately reviewed the checklist and wrote `PASSED`
        (2026-08-17) — i.e. it failed on success. A gate test must track the
        gate's logic, not one moment in the repo's history.

        Read-only: this test never creates or removes `PASSED`. Only a human
        may write it (E2-S3 "Banned approaches")."""
        from medmemgraph.pipeline.scale_gate import DEFAULT_HANDCHECK_DIR

        passed_exists = (DEFAULT_HANDCHECK_DIR / "PASSED").is_file()
        if passed_exists:
            assert_handcheck_passed()  # green gate must not raise
        else:
            with pytest.raises(ScaleGateError, match="PASSED"):
                assert_handcheck_passed()

    def test_error_names_the_missing_path_and_next_step(self, tmp_path: Path) -> None:
        d = tmp_path / "handcheck"
        d.mkdir()
        with pytest.raises(ScaleGateError) as exc_info:
            assert_handcheck_passed(handcheck_dir=d)
        message = str(exc_info.value)
        assert str(d / "PASSED") in message
        assert "CHECKLIST" in message


# ---------------------------------------------------------------------------
# assert_handcheck_passed() — AC2: permits with PASSED + >=30 valid rows.
# ---------------------------------------------------------------------------


class TestGatePermitsWithPassed:
    def test_passed_plus_30_valid_rows_returns_none_without_raising(self, tmp_path: Path) -> None:
        d = _green_handcheck_dir(tmp_path)
        assert assert_handcheck_passed(handcheck_dir=d) is None  # does not raise

    def test_canonical_id_zero_is_accepted_not_rejected(self, tmp_path: Path) -> None:
        """E2-S3 AC2's own carve-out, stated verbatim: 'after mint,
        canonical_id may still be 0 — allow 0 in this gate'. Every fixture
        fact above already uses canonical_id=0 on both subject and object;
        this test pins that explicitly so a future accidental tightening of
        `validate()` (or a re-implementation here) cannot silently start
        rejecting 0."""
        d = _green_handcheck_dir(tmp_path)
        facts = load_facts_jsonl(d / "facts.jsonl")
        assert all(f.subject.canonical_id == 0 for f in facts)
        assert all(f.object.canonical_id == 0 for f in facts)
        assert_handcheck_passed(handcheck_dir=d)  # does not raise

    def test_fewer_than_min_facts_still_blocks_even_with_passed(self, tmp_path: Path) -> None:
        d = tmp_path / "handcheck"
        _write_facts_jsonl(d / "facts.jsonl", [_valid_fact(i) for i in range(MIN_FACTS - 1)])
        _write_passed(d / "PASSED")
        with pytest.raises(ScaleGateError, match=str(MIN_FACTS)):
            assert_handcheck_passed(handcheck_dir=d)

    def test_negative_canonical_id_blocks_even_with_passed(self, tmp_path: Path) -> None:
        d = tmp_path / "handcheck"
        facts = [_valid_fact(i) for i in range(MIN_FACTS)]
        facts[0] = dataclasses.replace(
            facts[0], object=EntityRef(name="bad", type="Medication", canonical_id=-1)
        )
        _write_facts_jsonl(d / "facts.jsonl", facts)
        _write_passed(d / "PASSED")
        with pytest.raises(ScaleGateError, match="validate"):
            assert_handcheck_passed(handcheck_dir=d)

    def test_bad_polarity_blocks_even_with_passed(self, tmp_path: Path) -> None:
        d = tmp_path / "handcheck"
        facts = [_valid_fact(i) for i in range(MIN_FACTS)]
        facts[0] = dataclasses.replace(facts[0], polarity="maybe")
        _write_facts_jsonl(d / "facts.jsonl", facts)
        _write_passed(d / "PASSED")
        with pytest.raises(ScaleGateError):
            assert_handcheck_passed(handcheck_dir=d)

    def test_bad_predicate_blocks_even_with_passed(self, tmp_path: Path) -> None:
        d = tmp_path / "handcheck"
        facts = [_valid_fact(i) for i in range(MIN_FACTS)]
        facts[0] = dataclasses.replace(facts[0], predicate="NOT_A_REAL_PREDICATE")
        _write_facts_jsonl(d / "facts.jsonl", facts)
        _write_passed(d / "PASSED")
        with pytest.raises(ScaleGateError):
            assert_handcheck_passed(handcheck_dir=d)

    def test_malformed_json_line_raises_scale_gate_error_not_a_raw_exception(self, tmp_path: Path) -> None:
        d = tmp_path / "handcheck"
        d.mkdir()
        (d / "facts.jsonl").write_text("{not valid json\n", encoding="utf-8")
        _write_passed(d / "PASSED")
        with pytest.raises(ScaleGateError):
            assert_handcheck_passed(handcheck_dir=d)


# ---------------------------------------------------------------------------
# fact_from_dict() / load_facts_jsonl() round trip.
# ---------------------------------------------------------------------------


class TestFactSerializationRoundTrip:
    def test_asdict_then_fact_from_dict_round_trips(self) -> None:
        original = _valid_fact(1)
        row = dataclasses.asdict(original)
        restored = fact_from_dict(row)
        assert restored == original

    def test_load_facts_jsonl_reads_every_row_in_order(self, tmp_path: Path) -> None:
        facts = [_valid_fact(i) for i in range(5)]
        path = tmp_path / "facts.jsonl"
        _write_facts_jsonl(path, facts)
        loaded = load_facts_jsonl(path)
        assert [f.fact_id for f in loaded] == [f.fact_id for f in facts]

    def test_load_facts_jsonl_skips_blank_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "facts.jsonl"
        facts = [_valid_fact(i) for i in range(3)]
        path.write_text(
            "\n".join(json.dumps(dataclasses.asdict(f)) for f in facts[:1])
            + "\n\n"
            + "\n".join(json.dumps(dataclasses.asdict(f)) for f in facts[1:])
            + "\n",
            encoding="utf-8",
        )
        loaded = load_facts_jsonl(path)
        assert len(loaded) == 3


# ---------------------------------------------------------------------------
# ingest.ingest_patient() — the gate wired into the product path
# (decisions/005 Finding 2). Monkeypatches `scale_gate.DEFAULT_HANDCHECK_DIR`
# so the REAL, unmocked `assert_handcheck_passed()` call inside
# `ingest_patient()` runs against an isolated tmp directory — this proves
# the actual wiring, not a stand-in for it, without ever touching the real
# `fixtures/handcheck/`.
# ---------------------------------------------------------------------------


class _ExplodingClient:
    """Any `.run()` call is a test failure — used to prove `ingest_patient`
    never reaches the graph at all while the gate is blocked."""

    def run(self, cypher: str, **params: object):  # pragma: no cover - should never execute
        raise AssertionError(f"HydraClient.run() must never be called while the gate is blocked: {cypher!r}")

    def close(self) -> None:
        pass


class TestIngestRefusesToRunWhileBlocked:
    def test_ingest_patient_raises_before_loading_the_conversation(self, tmp_path: Path, monkeypatch) -> None:
        blocked_dir = tmp_path / "handcheck"
        blocked_dir.mkdir()  # no PASSED, no facts.jsonl
        monkeypatch.setattr(scale_gate, "DEFAULT_HANDCHECK_DIR", blocked_dir)

        def _must_not_be_called(*args, **kwargs):
            raise AssertionError("load_conversation must never run while the gate is blocked")

        monkeypatch.setattr(ingest, "load_conversation", _must_not_be_called)

        with pytest.raises(ScaleGateError):
            ingest.ingest_patient(
                "some-subject",
                now="2026-08-16T00:00:00",
                client=_ExplodingClient(),
            )

    def test_ingest_patient_raises_even_when_a_conversation_is_already_supplied(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """The gate runs FIRST, unconditionally — even a caller that
        bypasses `load_conversation` entirely by passing its own
        `conversation=` still hits the gate before extraction runs."""
        blocked_dir = tmp_path / "handcheck"
        blocked_dir.mkdir()
        monkeypatch.setattr(scale_gate, "DEFAULT_HANDCHECK_DIR", blocked_dir)

        def _must_not_be_called(*args, **kwargs):
            raise AssertionError("extract_facts must never run while the gate is blocked")

        monkeypatch.setattr(ingest, "extract_facts", _must_not_be_called)
        conversation = Conversation(subject_id="p1", processed_hadm_ids=(), admissions=())

        with pytest.raises(ScaleGateError):
            ingest.ingest_patient(
                "p1",
                now="2026-08-16T00:00:00",
                conversation=conversation,
                client=_ExplodingClient(),
            )


class TestIngestProceedsOnceGateIsGreen:
    def test_ingest_patient_runs_the_full_pipeline_once_unblocked(self, tmp_path: Path, monkeypatch) -> None:
        """Green gate (isolated tmp dir) -> `ingest_patient` actually drives
        extract -> resolve -> `write_and_invalidate` (never the bare
        `write_facts` — decisions/005 Finding 2's own instruction: the
        product path must call the wrapper that runs invalidation too)."""
        green_dir = _green_handcheck_dir(tmp_path)
        monkeypatch.setattr(scale_gate, "DEFAULT_HANDCHECK_DIR", green_dir)

        conversation = Conversation(subject_id="p1", processed_hadm_ids=(), admissions=())
        produced_facts = [_valid_fact(i) for i in range(3)]

        calls: dict[str, object] = {}

        def fake_extract_facts(conv, admission=None, *, extractor=None):
            calls["extract_facts"] = (conv, admission, extractor)
            return produced_facts

        def fake_attach_canonical_ids(facts, *, registry=None, complete=None, id_map=None):
            calls["attach_canonical_ids"] = (facts, registry, complete, id_map)
            return facts

        def fake_write_and_invalidate(client, facts, id_map=None, *, now, batch_size=1000):
            calls["write_and_invalidate"] = (client, facts, id_map, now, batch_size)
            write_report = WriteReport(facts_in=len(facts), facts_written=len(facts))
            return write_report, InvalidationReport()

        monkeypatch.setattr(ingest, "extract_facts", fake_extract_facts)
        monkeypatch.setattr(ingest, "attach_canonical_ids", fake_attach_canonical_ids)
        monkeypatch.setattr(ingest, "write_and_invalidate", fake_write_and_invalidate)

        report = ingest.ingest_patient(
            "p1",
            now="2026-08-16T00:00:00",
            conversation=conversation,
            client=_ExplodingClient(),  # never actually .run(); write_and_invalidate is faked
        )

        assert "extract_facts" in calls
        assert "attach_canonical_ids" in calls
        assert "write_and_invalidate" in calls
        assert report.subject_id == "p1"
        assert report.n_facts_extracted == 3
        assert report.n_facts_written == 3
        assert report.facts == produced_facts
