"""tests/test_writer.py — graph/writer.py (ARCHITECTURE.md §6.5,
stories/E4/E4-S4.md).

Live-engine reproduction this module's write shapes depend on (dev-backend,
2026-08-16; see `graph/writer.py`'s module docstring for the full write-up):
ARCHITECTURE.md §6.5's literal vertex-upsert shape,
`MERGE (n:Claim {id: row.vertex}) SET n.prop = row.prop`, is REJECTED live
against `ghcr.io/hydra-db/hydradb:0.1.1` —
`CypherSyntaxError('... UNWIND vertex upsert MERGE pattern matches only id;
apply labels with SET')`. The engine's actual required shape,
`MERGE (n {id: row.vertex}) SET n:Claim, n.prop = row.prop, ...`, was spiked
live (create-when-absent AND idempotent-merge-when-present, both confirmed
via a labelled-`MATCH` readback) and is what `graph/writer.py` emits.
`hydra_client._check_unlabelled_node` was extended with a narrow exception
for exactly this idiom — MATCH patterns remain fully guarded; re-verified
here (`test_match_based_unlabelled_pattern_is_still_rejected`) so a future
edit to that gate can't silently widen the hole.

Non-live tests use a `_RecordingClient` (records the Cypher + params it
would have sent, and independently runs every string through the real
`validate_dialect` so a shape that would be rejected by the live gate fails
these tests too) — `stories/E4/E4-S4.md` ACs 1/2. Live tests
(`@pytest.mark.live`) need a running HydraDB node on :7687 and
`HYDRA_AUTH_TOKEN` — AC3/4 plus this story's own test-plan (idempotent
double-write, sentinel round-trip, measured rows/sec).

Fixture ids: patient/fact identity keys are namespaced with a
`e4s4-writer-live-fixture` prefix, fixed (not time-based) — the same string
mints the same integer every run, so repeated suite runs converge onto the
same handful of nodes (MERGE-idempotent) rather than growing without bound,
and the long, distinctive hashed string makes collision with any other
agent's fixture (e.g. `test_run_dialect.py`'s literal `900001-900999` band,
`test_existence_both_directions.py`'s wall-clock-derived band) astronomically
unlikely — see `pipeline.ids`'s own collision-rate math (~2.7e-4 at
n=100,000; here n is a handful of long, distinct strings, not adjacent
small integers).
"""

from __future__ import annotations

import pytest

from medmemgraph import contracts
from medmemgraph.contracts import EntityRef, mock_fact
from medmemgraph.graph.writer import WriteReport, WriterError, write_facts
from medmemgraph.hydra_client import DialectError, HydraClient, validate_dialect
from medmemgraph.pipeline.ids import mint_claim_id, mint_entity_id, mint_patient_id

# ---------------------------------------------------------------------------
# Non-live: a fake client that records calls and independently enforces the
# same dialect gate `HydraClient.run` enforces, without touching the network.
# ---------------------------------------------------------------------------


class _RecordingClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def run(self, cypher: str, **params: object) -> list[dict]:
        validate_dialect(cypher)  # raises DialectError exactly like the real client
        self.calls.append((cypher, dict(params)))
        return []


class _FailingClient:
    """Raises on the Nth call (1-indexed) to exercise WriterError/partial
    progress. Every call before that is recorded like `_RecordingClient`."""

    def __init__(self, fail_on_call: int) -> None:
        self.fail_on_call = fail_on_call
        self.calls: list[tuple[str, dict]] = []

    def run(self, cypher: str, **params: object) -> list[dict]:
        validate_dialect(cypher)
        self.calls.append((cypher, dict(params)))
        if len(self.calls) == self.fail_on_call:
            raise RuntimeError("simulated engine-side failure")
        return []


def _fact(i: int, *, patient_id: str = "batch-patient", session_id: str = "batch-admission") -> contracts.ClinicalFact:
    return mock_fact(
        fact_id=f"batch-fact-{i:05d}",
        patient_id=patient_id,
        session_id=session_id,
        turn_ids=[i],
        subject=EntityRef(name=patient_id, type="Patient", canonical_id=1),
        object=EntityRef(name=f"med-{i}", type="Medication", canonical_id=5_000_000 + i),
    )


# ---------------------------------------------------------------------------
# Batch-shape unit tests — stories/E4/E4-S4.md AC1/AC2. No live node.
# ---------------------------------------------------------------------------


class TestBatchShape:
    def test_1001_facts_chunk_to_max_1000_rows_per_statement(self) -> None:
        client = _RecordingClient()
        facts = [_fact(i) for i in range(1001)]

        report = write_facts(client, facts)

        assert report.facts_written == 1001
        assert report.facts_skipped == 0
        assert client.calls, "expected at least one run() call"
        for cypher, params in client.calls:
            rows = params.get("rows")
            assert rows is not None, f"every statement must carry rows via a parameter: {cypher!r}"
            assert len(rows) <= 1000, f"statement carried {len(rows)} rows (> 1000): {cypher!r}"

        # Claim/HAS/ABOUT are each 1001 distinct rows (distinct fact_id /
        # object per fact) -> each needs exactly 2 statements (1000 + 1).
        claim_calls = [c for c, p in client.calls if "MERGE (n {id: row.vertex}) SET n:Claim" in c]
        assert len(claim_calls) == 2, f"expected 2 Claim batches for 1001 rows, got {len(claim_calls)}"

    def test_every_statement_starts_with_unwind_rows(self) -> None:
        client = _RecordingClient()
        write_facts(client, [_fact(i) for i in range(3)])
        assert client.calls
        for cypher, _ in client.calls:
            assert cypher.startswith("UNWIND $rows AS row"), cypher

    def test_no_bare_create_and_no_inline_unwind(self) -> None:
        client = _RecordingClient()
        write_facts(client, [_fact(i) for i in range(3)])
        for cypher, _ in client.calls:
            assert "CREATE" not in cypher.upper(), cypher
            assert "UNWIND $rows" in cypher
            assert "UNWIND [" not in cypher.replace(" ", "")

    def test_every_statement_passes_validate_dialect(self) -> None:
        # _RecordingClient already calls validate_dialect internally (it
        # would have raised already), but assert it explicitly too so this
        # test fails loudly and specifically if that wiring ever changes.
        client = _RecordingClient()
        write_facts(client, [_fact(i) for i in range(3)])
        for cypher, _ in client.calls:
            validate_dialect(cypher)  # must not raise


# ---------------------------------------------------------------------------
# Id-minting contract — no remint; subject/object canonical_id used as-is.
# ---------------------------------------------------------------------------


class TestIdMinting:
    def test_pre_minted_claim_id_is_reused_not_reminted(self) -> None:
        client = _RecordingClient()
        id_map: dict[str, int] = {}
        fact = _fact(0)
        pre_minted = mint_claim_id(fact.fact_id, id_map=id_map)

        write_facts(client, [fact], id_map=id_map)

        claim_rows = [
            row
            for cypher, params in client.calls
            if "SET n:Claim" in cypher
            for row in params["rows"]
        ]
        assert len(claim_rows) == 1
        assert claim_rows[0]["vertex"] == pre_minted

    def test_domain_entity_uses_canonical_id_as_is_no_remint(self) -> None:
        client = _RecordingClient()
        fact = mock_fact(
            object=EntityRef(name="metformin", type="Medication", canonical_id=424242)
        )
        write_facts(client, [fact])

        entity_rows = [
            row
            for cypher, params in client.calls
            if "SET n:Medication" in cypher
            for row in params["rows"]
        ]
        assert len(entity_rows) == 1
        assert entity_rows[0]["vertex"] == 424242

    def test_shared_id_map_gives_deterministic_replay_across_separate_calls(self) -> None:
        client_a = _RecordingClient()
        client_b = _RecordingClient()
        fact = _fact(0)

        shared_map: dict[str, int] = {}
        write_facts(client_a, [fact], id_map=shared_map)
        write_facts(client_b, [fact], id_map=shared_map)

        claim_vertex_a = next(
            row["vertex"]
            for cypher, params in client_a.calls
            if "SET n:Claim" in cypher
            for row in params["rows"]
        )
        claim_vertex_b = next(
            row["vertex"]
            for cypher, params in client_b.calls
            if "SET n:Claim" in cypher
            for row in params["rows"]
        )
        assert claim_vertex_a == claim_vertex_b


# ---------------------------------------------------------------------------
# ABOUT edge fan-out — object always, subject only when subject != Patient.
# ---------------------------------------------------------------------------


class TestAboutEdges:
    def test_subject_is_patient_writes_only_object_about_edge(self) -> None:
        client = _RecordingClient()
        fact = mock_fact(
            subject=EntityRef(name="patient-0001", type="Patient", canonical_id=1),
            object=EntityRef(name="metformin", type="Medication", canonical_id=2),
        )
        write_facts(client, [fact])
        about_rows = [
            row
            for cypher, params in client.calls
            if "MERGE (s)-[r:ABOUT " in cypher
            for row in params["rows"]
        ]
        assert len(about_rows) == 1
        assert about_rows[0]["destination_vertex"] == 2

    def test_subject_is_non_patient_writes_two_about_edges(self) -> None:
        client = _RecordingClient()
        fact = mock_fact(
            predicate="PRIMARY_CARE_PROVIDER_OF",
            subject=EntityRef(name="Dr. Lee", type="Provider", canonical_id=77),
            object=EntityRef(name="patient-0001", type="Provider", canonical_id=2),
        )
        write_facts(client, [fact])
        about_rows = [
            row
            for cypher, params in client.calls
            if "MERGE (s)-[r:ABOUT " in cypher
            for row in params["rows"]
        ]
        destinations = {row["destination_vertex"] for row in about_rows}
        assert destinations == {2, 77}


# ---------------------------------------------------------------------------
# Partial-failure handling — no transactions; WriterError carries progress.
# ---------------------------------------------------------------------------


class TestPartialFailure:
    def test_invalid_fact_is_skipped_not_raised(self) -> None:
        client = _RecordingClient()
        bad = mock_fact(fact_id="bad-1", predicate="NOT_A_REAL_PREDICATE")
        good = _fact(0)

        report = write_facts(client, [bad, good])

        assert report.facts_written == 1
        assert report.facts_skipped == 1
        assert report.skipped[0][0] == "bad-1"
        assert any("predicate" in p for p in report.skipped[0][1])

    def test_engine_failure_raises_writer_error_with_partial_report(self) -> None:
        client = _FailingClient(fail_on_call=2)  # Patient batch ok, Admission batch fails
        facts = [_fact(0)]

        with pytest.raises(WriterError) as exc_info:
            write_facts(client, facts)

        assert isinstance(exc_info.value.partial_report, WriteReport)
        assert exc_info.value.partial_report.statements_issued == 1

    def test_unrecognized_entity_type_is_skipped_not_raised(self) -> None:
        client = _RecordingClient()
        fact = mock_fact(
            fact_id="bad-entity-type",
            object=EntityRef(name="mystery", type="NotARealEntityType", canonical_id=99),
        )
        report = write_facts(client, [fact])
        assert report.facts_skipped == 1
        assert report.facts_written == 0


# ---------------------------------------------------------------------------
# The dialect gate's MERGE-vertex-upsert exception stays narrow — MATCH
# patterns must remain fully rejected (decisions/003's phantom-row bug is a
# MATCH/read-path defect; this exception must never leak into reads).
# ---------------------------------------------------------------------------


def test_match_based_unlabelled_pattern_is_still_rejected() -> None:
    with pytest.raises(DialectError):
        validate_dialect("MATCH (n {id: $id}) SET n:Patient")


def test_merge_with_extra_property_beyond_id_is_still_rejected() -> None:
    # The live engine's own restriction ("MERGE pattern matches only id");
    # the gate must not accidentally widen past exactly `{id: ...}`.
    with pytest.raises(DialectError):
        validate_dialect(
            "UNWIND $rows AS row MERGE (n {id: row.vertex, name: row.name}) SET n:Patient"
        )


# ---------------------------------------------------------------------------
# Live: requires HydraDB on :7687 and HYDRA_AUTH_TOKEN.
# ---------------------------------------------------------------------------

_LIVE_SESSION_ID = "e4s4-writer-live-fixture-admission"

# Each live test method below gets its OWN fixed patient_id, not a shared
# one. Ids here are deterministic hash-folds of fixed strings (never
# time-based), so reruns of this file converge onto the same handful of
# nodes rather than growing without bound — but that same determinism means
# a shared patient across test methods would accumulate every method's
# claims forever (idempotent replay is *per fact_id*, not "reset to zero").
# A `MATCH (p:Patient {id:$pid})-[:HAS]->(c:Claim)` count assertion would
# then drift upward on every subsequent suite run. One dedicated patient id
# per test method keeps each test's own count assertions stable forever.
_READBACK_PATIENT_ID = "e4s4-writer-live-fixture-patient-readback"
_PERF_PATIENT_ID = "e4s4-writer-live-fixture-patient-perf"
_ENTITY_PATIENT_ID = "e4s4-writer-live-fixture-patient-entity"


def _live_fact(i: int, *, patient_id: str, **overrides: object) -> contracts.ClinicalFact:
    med_name = f"e4s4-writer-live-fixture-med-{patient_id}-{i}"
    defaults = dict(
        fact_id=f"e4s4-writer-live-fixture-fact-{patient_id}-{i}",
        patient_id=patient_id,
        session_id=_LIVE_SESSION_ID,
        turn_ids=[i],
        subject=EntityRef(name=patient_id, type="Patient", canonical_id=1),
        object=EntityRef(
            name=med_name,
            type="Medication",
            # Mimics what Pipeline/ER (E3-S1) would already have filled in —
            # the writer never mints entity ids itself (§5.4/§6.4).
            canonical_id=mint_entity_id(patient_id, "Medication", med_name),
        ),
    )
    defaults.update(overrides)
    return mock_fact(**defaults)


@pytest.fixture(scope="module")
def live_client():
    c = HydraClient(transport="bolt")
    yield c
    c.close()


@pytest.mark.live
class TestLiveWriteAndReadback:
    def test_write_then_labelled_match_readback_and_idempotent_replay(self, live_client) -> None:
        facts = [
            _live_fact(1, patient_id=_READBACK_PATIENT_ID),
            _live_fact(2, patient_id=_READBACK_PATIENT_ID),
            _live_fact(3, patient_id=_READBACK_PATIENT_ID),
        ]
        pid = mint_patient_id(_READBACK_PATIENT_ID)

        report1 = write_facts(live_client, facts)
        assert report1.facts_written == 3
        assert report1.facts_skipped == 0

        rows = live_client.run(
            "MATCH (p:Patient {id: $pid})-[:HAS]->(c:Claim) "
            "RETURN c.fact_id AS fact_id, c.valid_to AS valid_to",
            pid=pid,
        )
        assert len(rows) == 3
        for row in rows:
            assert row["valid_to"] == contracts.SENTINEL_VALID_TO, (
                "a fact written with the sentinel must round-trip with the sentinel intact"
            )

        # Idempotent replay: same facts again must not duplicate nodes/edges.
        report2 = write_facts(live_client, facts)
        assert report2.facts_written == 3

        rows_after = live_client.run(
            "MATCH (p:Patient {id: $pid})-[:HAS]->(c:Claim) "
            "RETURN c.fact_id AS fact_id, c.valid_to AS valid_to",
            pid=pid,
        )
        assert len(rows_after) == 3, "second write duplicated Claim nodes/edges"
        assert {r["fact_id"] for r in rows_after} == {r["fact_id"] for r in rows}

        # count() over a raw node/relationship binding is rejected live
        # ("property values support integer, float, boolean, and string
        # literals" — spiked live, 2026-08-16); count a property instead.
        edge_count = live_client.run(
            "MATCH (p:Patient {id: $pid})-[:HAS]->(c:Claim) RETURN count(c.id) AS n",
            pid=pid,
        )
        assert edge_count == [{"n": 3}], "HAS edge count must stay 3 after a second identical write"

        patient_count = live_client.run(
            "MATCH (p:Patient {id: $pid}) RETURN count(p.id) AS n", pid=pid
        )
        assert patient_count == [{"n": 1}], "Patient must not be duplicated by repeated writes"

    def test_rows_per_sec_is_measured_and_reported(self, live_client) -> None:
        facts = [_live_fact(100 + i, patient_id=_PERF_PATIENT_ID) for i in range(25)]
        report = write_facts(live_client, facts)

        assert report.facts_written == 25
        assert report.wall_clock_s > 0
        assert report.rows_written > 0
        assert report.rows_per_sec == report.rows_written / report.wall_clock_s
        print(
            f"\n[writer rows/sec] rows_written={report.rows_written} "
            f"statements_issued={report.statements_issued} "
            f"wall_clock_s={report.wall_clock_s:.4f} "
            f"rows_per_sec={report.rows_per_sec:.1f}"
        )

    def test_object_and_subject_entity_nodes_are_labelled_and_readable(self, live_client) -> None:
        fact = _live_fact(200, patient_id=_ENTITY_PATIENT_ID)
        write_facts(live_client, [fact])
        rows = live_client.run(
            "MATCH (n:Medication {id: $id}) RETURN n.id AS id, n.name AS name",
            id=fact.object.canonical_id,
        )
        assert rows == [{"id": fact.object.canonical_id, "name": fact.object.name}]
