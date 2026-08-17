"""tests/test_existence_both_directions.py — the abstention fixture (decisions/003).

`structural_absence` is MedMemGraph's headline capability: a missing node or
an absent path is a boolean top-k similarity can never produce, and it
covers 5,964 adversarial items — 33.3% of MedLoCoMo (decisions/003,
ARCHITECTURE.md §7.6, PHASES.md leftover-gate row "Existence fixture, both
directions"). If this file is wrong the system never abstains and nobody
notices until the numbers land.

Executed-verified defect this file pins (decisions/003): an UNLABELLED node
pattern matched by `id` (`MATCH (n {id: X}) RETURN n.id AS id`) returns a
row for ANY integer, existing or not. `count(*)` inherits the phantom. A
`WHERE` guard does not filter it. Adding a label predicate fixes it. Every
positive/negative case below uses the labelled shape; the one exception is
`test_unlabelled_match_is_a_trap`, which deliberately bypasses the
client-side dialect gate to prove the raw defect is real and pin it so a
future engine upgrade that fixes it makes that one test fail loudly.

No separate test-lead plan exists in this repo's coordination model (see
`collaborative/inbox/004-to-claude.md`); the authorities are `decisions/003`,
`ARCHITECTURE.md` §5-§8, and `PHASES.md`'s leftover-gate row for this file.

Requires a live HydraDB node (`docker ps` shows `hydradb`,
`curl -fsS http://127.0.0.1:9090/readyz` is 200) and `HYDRA_AUTH_TOKEN` in
the environment. All tests in this module are marked `live`.
"""

from __future__ import annotations

import os
import time

import pytest
import requests

from medmemgraph.graph.existence import exists
from medmemgraph.hydra_client import DialectError, HydraClient

pytestmark = pytest.mark.live

# ---------------------------------------------------------------------------
# Namespaced id range for this fixture only. Two constraints drive this:
#
# 1. Must not collide with round_trip()'s ids (1, 2) or test_run_dialect.py's
#    test_existence_fixture_both_directions_bolt range (900001-900999).
# 2. Bare-edge-pattern CREATE is the only write shape confirmed to work live
#    against this engine for a fresh (label, id) pair (a labelled MERGE
#    vertex-upsert, `MERGE (n:Patient {id: ...})`, was spiked live here and
#    the engine rejects it: "MERGE pattern matches only id; apply labels
#    with SET" — that shape needs an *unlabelled* MERGE pattern, which the
#    client-side dialect gate refuses per decisions/003, so it is not
#    available as a fixture-writing shape in this codebase). CREATE is not
#    idempotent — repeated CREATE calls duplicate nodes rather than
#    upserting (same behaviour test_run_dialect.py's own round_trip test
#    documents). A fixed id block would therefore make "exactly one row"
#    true only on a database that has never run this file before.
#
# The fix: derive the base id from the wall clock so every test-session
# invocation claims a fresh, effectively-never-before-used id block. That
# keeps "exactly one row" true on both a first run and a hundredth rerun,
# without requiring delete/cleanup capability this dialect doesn't have.
# ---------------------------------------------------------------------------
_BASE = 10_000_000 + (time.time_ns() // 1000) % 89_000_000
PATIENT_ID = _BASE + 1  # known-present :Patient
CLAIM_ID = _BASE + 2  # known-present :Claim, reached from PATIENT_ID via HAS
HAS_EDGE_ID = _BASE + 3
NEVER_CREATED_ID = _BASE + 999  # never inserted, under any label
NEVER_CREATED_SEED_ID = _BASE + 998  # never inserted; used as an edge-anchored seed


@pytest.fixture(scope="module")
def client():
    c = HydraClient(transport="bolt")
    # Bare node CREATE is rejected by the engine (decisions/003: "only
    # one-hop edge pattern..."); the fixture is minted as one edge pattern,
    # the same shape ARCHITECTURE.md §6.5 and PHASES.md's round-trip use.
    c.run(
        "CREATE (p:Patient {id: $pid, patient_id: 'existence-fixture'})"
        "-[:HAS {id: $eid}]->(cl:Claim {id: $cid, predicate: 'fixture'})",
        pid=PATIENT_ID,
        eid=HAS_EDGE_ID,
        cid=CLAIM_ID,
    )
    yield c
    c.close()


# ---------------------------------------------------------------------------
# 1-4: labelled MATCH, all four cells of the decision-003 truth table.
# ---------------------------------------------------------------------------


def test_labelled_match_known_present_returns_exactly_one_row(client: HydraClient) -> None:
    rows = client.run("MATCH (n:Patient {id: $id}) RETURN n.id AS id", id=PATIENT_ID)
    assert rows == [{"id": PATIENT_ID}]


def test_labelled_match_never_created_returns_zero_rows(client: HydraClient) -> None:
    rows = client.run("MATCH (n:Patient {id: $id}) RETURN n.id AS id", id=NEVER_CREATED_ID)
    assert rows == []


def test_labelled_match_wrong_label_returns_zero_rows(client: HydraClient) -> None:
    # Right id, wrong label — decision 003's third truth-table cell.
    rows = client.run("MATCH (n:Condition {id: $id}) RETURN n.id AS id", id=PATIENT_ID)
    assert rows == []


def test_edge_anchored_traversal_from_never_created_seed_returns_zero_rows(
    client: HydraClient,
) -> None:
    rows = client.run(
        "MATCH (a:Patient {id: $id})-[:HAS]->(b:Claim) RETURN b.id AS id",
        id=NEVER_CREATED_SEED_ID,
    )
    assert rows == []


def test_edge_anchored_traversal_from_known_seed_returns_a_row(client: HydraClient) -> None:
    # Companion positive case: the same edge-anchored shape returns a row
    # when the seed is real, so the negative case above is proven against a
    # query that is known to be capable of matching something.
    rows = client.run(
        "MATCH (a:Patient {id: $id})-[:HAS]->(b:Claim) RETURN b.id AS id",
        id=PATIENT_ID,
    )
    assert rows == [{"id": CLAIM_ID}]


# ---------------------------------------------------------------------------
# 5: the client-side gate refuses the unlabelled shape before it reaches the
# wire. This is what makes the trap below safe to leave in the codebase.
# ---------------------------------------------------------------------------


def test_run_rejects_unlabelled_match_before_it_reaches_the_wire(client: HydraClient) -> None:
    with pytest.raises(DialectError):
        client.run("MATCH (n {id: $id}) RETURN n.id AS id", id=NEVER_CREATED_ID)


# ---------------------------------------------------------------------------
# 6: the trap, documented as a trap. `HydraClient.run()` already refuses
# this shape (previous test), so this test talks to the wire directly,
# deliberately bypassing every client-side gate, to prove the underlying
# engine defect is real rather than the client's.
#
# PRODUCT CODE MUST NEVER QUERY THIS WAY. This helper exists only to pin the
# raw defect for this one test.
# ---------------------------------------------------------------------------


def _raw_http_query_bypassing_the_dialect_gate(cypher: str, **params: object) -> list[dict]:
    """Hit the live HydraDB HTTP endpoint directly with `requests`, skipping
    `HydraClient.run()` (and therefore `validate_dialect`) entirely — the
    same transport decision 003 used to reproduce the defect live. NOT a
    sanctioned query path; do not reuse this outside this trap test."""
    base = os.environ.get("HYDRA_HTTP_URL", "http://127.0.0.1:8443").rstrip("/")
    token = (
        os.environ.get("HYDRA_AUTH_TOKEN")
        or os.environ.get("GRAPH_AUTH_TOKEN")
        or "local-development-token-32-bytes"
    )
    resp = requests.post(
        f"{base}/v1/graphs/default/query",
        json={"cell_id": "cell-0", "query": cypher, "parameters": params},
        headers={
            "Authorization": f"Bearer {token}",
            "X-Graph-Namespace": "default",
            "Content-Type": "application/json",
        },
        timeout=30,
    )
    resp.raise_for_status()
    payload = resp.json()
    columns = payload.get("columns") or []
    rows_out: list[dict] = []
    for raw_row in payload.get("rows", []):
        row = {}
        for col, tagged in zip(columns, raw_row):
            row[col] = tagged.get("value") if isinstance(tagged, dict) else tagged
        rows_out.append(row)
    return rows_out


def test_unlabelled_match_is_a_trap() -> None:
    """PINS decisions/003's defect. Do not "fix" this test if it starts
    failing.

    A failure here means HydraDB stopped returning a phantom row for an
    unlabelled `{id: X}` lookup on a never-created id — that would be GOOD
    NEWS (the engine upgrade decision 003 anticipated landed), but it also
    means `hydra_client._check_unlabelled_node`'s rejection and this
    project's whole "always label the pattern" discipline should be
    revisited, loudly, as an architecture conversation — not silently
    edited here to keep the suite green.

    PRODUCT CODE MUST NEVER USE THE UNLABELLED SHAPE. `HydraClient.run()`
    already refuses it (see
    `test_run_rejects_unlabelled_match_before_it_reaches_the_wire` above);
    this test bypasses that gate on purpose to talk to the wire directly and
    prove the underlying engine's behaviour, not the client's.
    """
    rows = _raw_http_query_bypassing_the_dialect_gate(
        "MATCH (n {id: $id}) RETURN n.id AS id", id=NEVER_CREATED_ID
    )
    assert rows == [{"id": NEVER_CREATED_ID}], (
        "Expected the documented phantom row (decisions/003) for a "
        f"never-created id under an UNLABELLED match. Got {rows!r} instead — "
        "if this is [], the raw engine defect this test exists to pin may "
        "have been fixed upstream; read the docstring before touching this "
        "assertion, and open a collaborative/decisions/ note, not a quiet edit."
    )


# ---------------------------------------------------------------------------
# 7: existence.exists(label, node_id) -> bool, both directions.
# ---------------------------------------------------------------------------


def test_exists_true_for_known_present(client: HydraClient) -> None:
    assert exists("Patient", PATIENT_ID, client=client) is True


def test_exists_false_for_never_created(client: HydraClient) -> None:
    assert exists("Patient", NEVER_CREATED_ID, client=client) is False


def test_exists_false_for_right_id_wrong_label(client: HydraClient) -> None:
    assert exists("Condition", PATIENT_ID, client=client) is False


def test_exists_opens_and_closes_its_own_client_when_none_is_given() -> None:
    # No `client=` kwarg: exists() must open a short-lived connection itself
    # and still answer correctly both directions.
    assert exists("Patient", PATIENT_ID) is True
    assert exists("Patient", NEVER_CREATED_ID) is False


def test_exists_rejects_an_unsafe_or_empty_label() -> None:
    with pytest.raises(ValueError):
        exists("", PATIENT_ID)
    with pytest.raises(ValueError):
        exists("Patient {id: 1}) DETACH DELETE (n", PATIENT_ID)
