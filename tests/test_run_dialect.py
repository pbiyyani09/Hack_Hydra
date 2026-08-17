"""tests/test_run_dialect.py — dialect-gate coverage for hydra_client.run().

Pure-Python parametrized cases (`validate_dialect` directly) need no live
node and are NOT marked `live`. The `@pytest.mark.live` tests exercise the
real thing: driver patch, Bolt round-trip, HTTP round-trip, healthz.
"""

from __future__ import annotations

import pytest

from medmemgraph.hydra_client import DialectError, HydraClient, healthz, validate_dialect

# ---------------------------------------------------------------------------
# Illegal shapes — every rule named in ARCHITECTURE.md §8.3 / decisions/003.
# ---------------------------------------------------------------------------

ILLEGAL_QUERIES = [
    pytest.param(
        "MATCH (n:Patient {id: $id}) WHERE n.name IN $names RETURN n.id AS id",
        id="in-operator",
    ),
    pytest.param(
        "MATCH (n:Patient {id: $id}) WHERE n.name CONTAINS 'foo' RETURN n.id AS id",
        id="contains",
    ),
    pytest.param(
        "MATCH (n:Patient {id: $id}) WHERE n.name ENDS WITH 'foo' RETURN n.id AS id",
        id="ends-with",
    ),
    pytest.param(
        "MATCH (n:Patient {id: $id}) WHERE n.valid_to IS NULL RETURN n.id AS id",
        id="is-null",
    ),
    pytest.param(
        "MATCH (n:Patient {id: $id}) WHERE n.valid_to IS NOT NULL RETURN n.id AS id",
        id="is-not-null",
    ),
    pytest.param(
        "MATCH (n:Patient {id: $id}) RETURN CASE WHEN n.id > 0 THEN 1 ELSE 0 END AS flag",
        id="case",
    ),
    pytest.param(
        "MATCH (n:Claim {patient_id: $pid}) RETURN min(n.confidence) AS m",
        id="min-aggregate",
    ),
    pytest.param(
        "MATCH (n:Claim {patient_id: $pid}) RETURN max(n.confidence) AS m",
        id="max-aggregate",
    ),
    pytest.param(
        "MATCH (n:Patient {id: $id}) RETURN n",
        id="bare-return-n",
    ),
    pytest.param(
        "MATCH (n:Patient {id: $id}) RETURN *",
        id="return-star",
    ),
    pytest.param(
        "MATCH (n {id: $id}) RETURN n.id AS id",
        id="unlabelled-node-with-props",
    ),
    pytest.param(
        "MATCH (n)-[:HAS]->(c:Claim) RETURN c.id AS id",
        id="unlabelled-node-bare",
    ),
    pytest.param(
        "MATCH (a:Patient {id: $id})-[:HAS]->(b) RETURN b.id AS id",
        id="unlabelled-node-on-far-end",
    ),
    pytest.param(
        "UNWIND [1, 2, 3] AS x MATCH (n:Claim {id: x}) RETURN n.id AS id",
        id="inline-unwind-list",
    ),
    pytest.param(
        "BEGIN MATCH (n:Patient {id: 1}) RETURN n.id AS id",
        id="begin",
    ),
    pytest.param(
        "MATCH (n:Patient {id: 1}) RETURN n.id AS id COMMIT",
        id="commit",
    ),
    pytest.param(
        "MATCH (n:Patient {id: 1}) RETURN n.id AS id ROLLBACK",
        id="rollback",
    ),
    pytest.param(
        "MATCH (a:Patient {id: $id})-[:HAS]-(b:Claim) RETURN b.id AS id",
        id="undirected-relationship",
    ),
    pytest.param(
        "MATCH (a:Patient {id: $id})-[:HAS*]->(b:Claim) RETURN b.id AS id",
        id="unbounded-varlen-bare-star",
    ),
    pytest.param(
        "MATCH (a:Patient {id: $id})-[:HAS*1..]->(b:Claim) RETURN b.id AS id",
        id="unbounded-varlen-open-max",
    ),
    pytest.param(
        "MATCH (n:Patient {id: 1}) RETURN n.id AS id; "
        "MATCH (m:Patient {id: 2}) RETURN m.id AS id",
        id="multi-statement-semicolon",
    ),
]


@pytest.mark.parametrize("cypher", ILLEGAL_QUERIES)
def test_illegal_shape_raises_dialect_error(cypher: str) -> None:
    with pytest.raises(DialectError):
        validate_dialect(cypher)


# ---------------------------------------------------------------------------
# Legal shapes — copied verbatim (or parametrized) from ARCHITECTURE.md /
# PHASES.md's round-trip block and decisions/003's "correct" examples.
# ---------------------------------------------------------------------------

LEGAL_QUERIES = [
    pytest.param(
        "CREATE (a:Entity {id: 1, name: 'alpha'})"
        "-[:RELATES {relationship_id: 'ab'}]->"
        "(b:Entity {id: 2, name: 'beta'})",
        id="round-trip-create",
    ),
    pytest.param(
        "MATCH (a:Entity {id: 1})-[:RELATES]->(b:Entity) "
        "RETURN b.id AS id, b.name AS name",
        id="round-trip-match",
    ),
    pytest.param(
        "CALL algo.MSpaths({ "
        "sourceLabel: 'Entity', sourceProperty: 'name', "
        "sourceValues: ['alpha'], targetValues: ['beta'], "
        "pairwise: true, relTypes: ['RELATES'], relDirection: 'both', "
        "maxLen: 3, pathCount: 5, resultLimit: 100 "
        "}) YIELD path RETURN path",
        id="round-trip-mspaths-yield-path",
    ),
    pytest.param(
        "MATCH (n:Patient {id:500}) RETURN n.id AS id",
        id="existence-known-present",
    ),
    pytest.param(
        "MATCH (n:Patient {id:777777}) RETURN n.id AS id",
        id="existence-known-absent",
    ),
    pytest.param(
        "MATCH (n:Condition {id:500}) RETURN n.id AS id",
        id="existence-right-id-wrong-label",
    ),
    pytest.param(
        "MATCH (a:Patient {id:500})-[:HAS]->(b:Claim) RETURN b.id AS id",
        id="edge-anchored-existence",
    ),
    pytest.param(
        "MATCH (p:Patient {id: $pid})-[:HAS]->(c:Claim) "
        "WHERE c.valid_from <= $D AND c.valid_to > $D "
        "RETURN c.id AS id, c.valid_from AS valid_from, "
        "c.valid_to AS valid_to, c.predicate AS predicate",
        id="current-as-of-interval-query",
    ),
    pytest.param(
        "UNWIND $rows AS row MERGE (n:Claim {id: row.vertex}) "
        "SET n.fact_id = row.fact_id, n.patient_id = row.patient_id, "
        "n.valid_from = row.valid_from, n.valid_to = row.valid_to, "
        "n.status = row.status",
        id="unwind-merge-vertex-upsert-parameterized",
    ),
    pytest.param(
        "UNWIND $rows AS row "
        "MATCH (s:Patient {id: row.source_vertex}), (d:Claim {id: row.destination_vertex}) "
        "MERGE (s)-[r:HAS {id: row.relationship_vertex}]->(d) "
        "SET r.observed_at = row.observed_at",
        id="unwind-merge-relationship-upsert-parameterized",
    ),
    pytest.param(
        "MATCH (a:Patient {id:$id})-[:HAS*1..16]->(b:Claim) RETURN b.id AS id",
        id="bounded-varlen-explicit-max",
    ),
    # String-literal robustness — "both directions": a literal that merely
    # *contains* a reserved word must not false-positive.
    pytest.param(
        "MATCH (n:Claim {id: $id}) SET n.status = 'case closed' "
        "RETURN n.id AS id",
        id="string-literal-contains-case-not-rejected",
    ),
    pytest.param(
        "MATCH (n:Claim {id: $id}) SET n.notes = 'living in boston' "
        "RETURN n.id AS id",
        id="string-literal-contains-in-not-rejected",
    ),
    pytest.param(
        "MATCH (n:Claim {id: $id}) SET n.notes = 'the min(x) style call' "
        "RETURN n.id AS id",
        id="string-literal-contains-min-paren-not-rejected",
    ),
]


@pytest.mark.parametrize("cypher", LEGAL_QUERIES)
def test_legal_shape_passes_gate(cypher: str) -> None:
    validate_dialect(cypher)  # must not raise


def test_real_keyword_case_is_still_rejected_outside_a_string() -> None:
    """The other half of "test both directions": the same word, used as a
    real keyword rather than string content, must still be rejected."""
    with pytest.raises(DialectError):
        validate_dialect(
            "MATCH (n:Claim {id: $id}) RETURN CASE WHEN n.id > 0 THEN 'y' ELSE 'n' END AS flag"
        )


def test_real_in_keyword_is_still_rejected_outside_a_string() -> None:
    with pytest.raises(DialectError):
        validate_dialect(
            "MATCH (n:Claim {id: $id}) WHERE n.predicate IN $preds RETURN n.id AS id"
        )


def test_dialect_error_message_names_rule_and_fix() -> None:
    with pytest.raises(DialectError) as exc_info:
        validate_dialect("MATCH (n {id: 1}) RETURN n.id AS id")
    message = str(exc_info.value)
    assert "label" in message.lower()
    assert "fix" in message.lower()


# ---------------------------------------------------------------------------
# Live tests — real HydraDB node, both transports.
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_healthz_live() -> None:
    assert healthz() is True


@pytest.mark.live
def test_round_trip_bolt() -> None:
    client = HydraClient(transport="bolt")
    try:
        result = client.round_trip()
        # CREATE (not MERGE) is the prescribed shape (PHASES.md round-trip
        # block), so repeated round_trip() calls in the same process/session
        # accumulate additional alpha->beta edges rather than erroring — assert
        # on presence, not on an exact single-row list.
        assert {"id": 2, "name": "beta"} in result["match_rows"]
        assert len(result["path_rows"]) >= 1
    finally:
        client.close()


@pytest.mark.live
def test_round_trip_http() -> None:
    client = HydraClient(transport="http")
    try:
        result = client.round_trip()
        # CREATE (not MERGE) is the prescribed shape (PHASES.md round-trip
        # block), so repeated round_trip() calls in the same process/session
        # accumulate additional alpha->beta edges rather than erroring — assert
        # on presence, not on an exact single-row list.
        assert {"id": 2, "name": "beta"} in result["match_rows"]
        assert len(result["path_rows"]) >= 1
    finally:
        client.close()


@pytest.mark.live
def test_existence_fixture_both_directions_bolt() -> None:
    """Decision 003's own fixture, labelled, both directions, over run()."""
    client = HydraClient(transport="bolt")
    try:
        client.run(
            "CREATE (p:Patient {id: 900001, patient_id: 'fixture'})"
            "-[:HAS {id: 900002}]->(c:Claim {id: 900003, predicate: 'X'})"
        )
        present = client.run("MATCH (n:Patient {id: 900001}) RETURN n.id AS id")
        absent = client.run("MATCH (n:Patient {id: 900999}) RETURN n.id AS id")
        assert present == [{"id": 900001}]
        assert absent == []
    finally:
        client.close()
