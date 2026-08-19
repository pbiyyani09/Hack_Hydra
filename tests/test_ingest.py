"""tests/test_ingest.py — `pipeline.ingest.ingest_patient`, the one production
entry point that walks loader -> extract -> resolve -> write_and_invalidate.

This file did not exist before 2026-08-17. `ingest_patient` was written to fix
decisions/005 Finding 2 ("no production code called `write_facts` at all — only
tests did") and then itself had zero callers and zero tests, which is how the
entity-type drift (see `TestTotalSkipIsLoud`) survived: the module that made
ingest possible was never once run end to end.

Offline: a recording fake client (same shape as `tests/test_writer.py`'s) and an
injected `Extractor(complete_fn=...)`, so no HydraDB, no API key, and no network.
The handcheck gate is neutralized by monkeypatch (see `green_gate`) — these tests
must never depend on, or write, the real `fixtures/handcheck/PASSED`.
`tests/test_scale_gate.py` owns testing the gate itself.
"""

from __future__ import annotations

import json

import pytest

from medmemgraph.contracts import EntityRef, mock_fact
from medmemgraph.hydra_client import validate_dialect
from medmemgraph.pipeline import ingest as ingest_mod
from medmemgraph.pipeline.ingest import IngestError, ingest_patient
from medmemgraph.pipeline.loader import Admission, Conversation
from medmemgraph.pipeline.scale_gate import ScaleGateError

NOW = "2126-01-01T00:00:00"


class _RecordingClient:
    """Enforces the same dialect gate the real client does, without a network."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def run(self, cypher: str, **params: object) -> list[dict]:
        validate_dialect(cypher)
        self.calls.append((cypher, dict(params)))
        return []

    def close(self) -> None:  # pragma: no cover - ingest only closes what it opens
        pass


def _conversation(subject_id: str = "patient-0001") -> Conversation:
    admission = Admission(
        hadm_id="admission-0001",
        admission_start="2124-03-01 09:00:00",
        admission_end="2124-03-01 09:20:00",
        conversation_lines=(
            {
                "turn_number": 1,
                "time": "2124-03-01 09:00:00",
                "speaker": "Doctor",
                "text": "You are prescribed metformin for your diabetes.",
            },
        ),
    )
    return Conversation(
        subject_id=subject_id,
        processed_hadm_ids=("admission-0001",),
        admissions=(admission,),
    )


@pytest.fixture()
def green_gate(monkeypatch):
    """Neutralize the handcheck gate for tests about what happens AFTER it.

    Patched to a no-op rather than satisfied with real artifacts, deliberately:
    `ingest_patient` calls `assert_handcheck_passed()` with no arguments, so the
    only other way to make it pass is to create the real
    `fixtures/handcheck/PASSED`. No test may do that — a green gate must mean a
    human reviewed the extraction sample (E2-S3's "Banned approaches": never
    auto-pass the gate). `tests/test_scale_gate.py` owns testing the gate itself
    against a `tmp_path`."""
    monkeypatch.setattr(ingest_mod, "assert_handcheck_passed", lambda *a, **k: None)


class TestScaleGateRunsFirst:
    def test_blocked_gate_raises_before_anything_else_happens(self, monkeypatch):
        """The gate is line 1 for a reason: a red gate must not cost an LLM call."""

        def _blocked(*_a, **_k):
            raise ScaleGateError("fixtures/handcheck/PASSED does not exist")

        monkeypatch.setattr(ingest_mod, "assert_handcheck_passed", _blocked)

        def _must_not_run(*_a, **_k):  # pragma: no cover - asserts it is unreachable
            raise AssertionError("extraction ran despite a blocked scale gate")

        monkeypatch.setattr(ingest_mod, "extract_facts", _must_not_run)

        with pytest.raises(ScaleGateError):
            ingest_patient("patient-0001", now=NOW, conversation=_conversation())


class TestTotalSkipIsLoud:
    """The 2026-08-17 silent-skip regression guard.

    `write_facts` records a skipped fact and returns normally — correct for one
    bad row, catastrophic when it is every row, because ingest then "succeeds"
    against an empty graph. `ingest_patient` raises on a total skip; partial
    skips stay recorded-not-raised."""

    def test_all_facts_skipped_raises_ingest_error(self, green_gate, monkeypatch):
        unwritable = [
            mock_fact(
                fact_id=f"bad-{i}",
                object=EntityRef(name="mystery", type="NotARealEntityType", canonical_id=90 + i),
            )
            for i in range(3)
        ]
        monkeypatch.setattr(
            ingest_mod, "extract_facts", lambda *a, **k: list(unwritable)
        )
        client = _RecordingClient()

        with pytest.raises(IngestError) as exc_info:
            ingest_patient(
                "patient-0001", now=NOW, conversation=_conversation(), client=client
            )

        message = str(exc_info.value)
        assert "NONE were written" in message
        # The reason must be in the message — "3 skipped" with no cause is what
        # made this class of bug take a day to find.
        assert "NotARealEntityType" in message

    def test_partial_skip_does_not_raise(self, green_gate, monkeypatch):
        good = mock_fact(
            fact_id="good-1",
            object=EntityRef(name="metformin", type="Medication", canonical_id=11),
        )
        bad = mock_fact(
            fact_id="bad-1",
            object=EntityRef(name="mystery", type="NotARealEntityType", canonical_id=12),
        )
        monkeypatch.setattr(ingest_mod, "extract_facts", lambda *a, **k: [good, bad])

        report = ingest_patient(
            "patient-0001", now=NOW, conversation=_conversation(), client=_RecordingClient()
        )

        assert report.n_facts_written == 1
        assert report.write_report.facts_skipped == 1
        assert not report.write_report.total_skip

    def test_empty_fact_list_is_not_a_total_skip(self, green_gate, monkeypatch):
        """A patient whose conversation yields nothing extractable is a real,
        non-error outcome — `total_skip` requires facts to have gone IN."""
        monkeypatch.setattr(ingest_mod, "extract_facts", lambda *a, **k: [])

        report = ingest_patient(
            "patient-0001", now=NOW, conversation=_conversation(), client=_RecordingClient()
        )

        assert report.n_facts_extracted == 0
        assert not report.write_report.total_skip


class TestHappyPath:
    def test_real_extractor_output_shape_reaches_the_graph(self, green_gate, monkeypatch):
        """End to end with the entity types extraction actually emits (lowercase,
        pre-normalization) — proving the extract-time canonicalization lands."""
        from medmemgraph.pipeline.extract import Extractor

        payload = {
            "facts": [
                {
                    "subject_name": "patient",
                    "subject_type": "Patient",
                    "predicate_phrase": "is prescribed",
                    "object_name": "metformin",
                    "object_type": "medication",  # lowercase, as real runs emit
                    "assertion": "present",
                    "prn": False,
                    "time_expression": "",
                    "turn_ids": [1],
                    "confidence": 0.9,
                    "evidence_quote": "You are prescribed metformin",
                }
            ]
        }

        from medmemgraph import llm

        def _fake_complete(prompt: str, **kwargs):  # noqa: ANN001, ARG001
            return llm.LLMResponse(
                text=json.dumps(payload),
                parsed=payload,
                model="fake",
                prompt_tokens=10,
                completion_tokens=5,
                cost_usd=0.0,
                latency_ms=1.0,
                cached=False,
                attempts=1,
            )

        client = _RecordingClient()
        report = ingest_patient(
            "patient-0001",
            now=NOW,
            conversation=_conversation(),
            client=client,
            extractor=Extractor(complete_fn=_fake_complete),
        )

        assert report.n_facts_written == 1
        assert report.facts[0].object.type == "Medication"
        assert report.facts[0].object.canonical_id != 0
        # A `:Medication` vertex batch really was issued.
        assert any("SET n:Medication" in cypher for cypher, _ in client.calls)


class TestTurnProvenanceLayer:
    """`:Turn` / `CONTAINS` / `DRAWN_FROM` (added 2026-08-17).

    Before this, nothing in `src/` wrote a `:Turn` node, so
    `demo/provenance.py::provenance_chain` returned `turns: []` for every claim
    — its `MATCH (c:Claim {id:$cid})-[:DRAWN_FROM]->(t:Turn)` had no edges to
    walk. The demo's "here is the sentence that produced this version of the
    fact" beat had nothing behind it."""

    @staticmethod
    def _ingest(monkeypatch, facts):
        monkeypatch.setattr(ingest_mod, "extract_facts", lambda *a, **k: list(facts))
        client = _RecordingClient()
        report = ingest_patient(
            "patient-0001", now=NOW, conversation=_conversation(), client=client
        )
        return client, report

    def test_turn_nodes_and_edges_are_written(self, green_gate, monkeypatch):
        fact = mock_fact(
            fact_id="f-1",
            patient_id="patient-0001",
            session_id="admission-0001",
            turn_ids=[1],
            object=EntityRef(name="metformin", type="Medication", canonical_id=11),
        )
        client, report = self._ingest(monkeypatch, [fact])

        cyphers = [c for c, _ in client.calls]
        assert any("SET n:Turn" in c for c in cyphers)
        assert any("CONTAINS" in c for c in cyphers)
        assert any("DRAWN_FROM" in c for c in cyphers)
        assert report.n_turns_written == 1

    def test_turn_properties_are_the_names_provenance_reads(self, green_gate, monkeypatch):
        """`demo/provenance.py` selects `t.session_id`, `t.turn_id`, `t.raw_text`.
        Renaming any of them turns the provenance walk into rows of nulls, so the
        contract is asserted here rather than discovered on camera."""
        fact = mock_fact(
            fact_id="f-1",
            patient_id="patient-0001",
            session_id="admission-0001",
            turn_ids=[1],
            object=EntityRef(name="metformin", type="Medication", canonical_id=11),
        )
        client, _ = self._ingest(monkeypatch, [fact])

        turn_rows = [
            params["rows"] for cypher, params in client.calls if "SET n:Turn" in cypher
        ]
        assert turn_rows, "no :Turn vertex batch was issued"
        row = turn_rows[0][0]
        assert row["session_id"] == "admission-0001"
        assert row["turn_id"] == 1
        assert row["speaker"] == "Doctor"
        assert "metformin" in row["raw_text"]
        assert row["occurred_at"] == "2124-03-01 09:00:00"

    def test_drawn_from_anchors_on_the_same_claim_id_write_facts_minted(
        self, green_gate, monkeypatch
    ):
        """The subtle failure this guards: if `write_turns` re-derives claim ids
        from a *different* id map, the DRAWN_FROM edge upsert MATCHes nothing and
        every edge is silently skipped — no error, empty provenance."""
        from medmemgraph.pipeline.ids import mint_claim_id

        fact = mock_fact(
            fact_id="f-1",
            patient_id="patient-0001",
            session_id="admission-0001",
            turn_ids=[1],
            object=EntityRef(name="metformin", type="Medication", canonical_id=11),
        )
        client, _ = self._ingest(monkeypatch, [fact])

        claim_rows = [
            params["rows"] for cypher, params in client.calls if "SET n:Claim" in cypher
        ]
        drawn_rows = [
            params["rows"] for cypher, params in client.calls if "DRAWN_FROM" in cypher
        ]
        written_claim_id = claim_rows[0][0]["vertex"]
        assert drawn_rows[0][0]["source_vertex"] == written_claim_id
        assert written_claim_id == mint_claim_id("f-1")

    def test_fact_citing_a_turn_outside_the_conversation_is_recorded_not_silent(
        self, green_gate, monkeypatch
    ):
        fact = mock_fact(
            fact_id="f-dangling",
            patient_id="patient-0001",
            session_id="admission-0001",
            turn_ids=[999],  # this conversation has only turn 1
            object=EntityRef(name="metformin", type="Medication", canonical_id=11),
        )
        _, report = self._ingest(monkeypatch, [fact])

        assert report.turn_report is not None
        assert any("turn 999 not found" in p for _fid, ps in report.turn_report.skipped for p in ps)
