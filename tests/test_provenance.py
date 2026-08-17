"""tests/test_provenance.py — coverage for `demo/provenance.py` (E8-S2).

`test_source_has_no_banned_cypher_shape` is a fast, pure-Python source scan
(`ast`, same convention `tests/test_invalidate.py`'s
`test_no_delete_of_claim_anywhere_in_this_module` establishes: inspect only
the literal Cypher text passed to `client.run(...)` calls, not the module's
prose — several docstrings here explain, in English, why the module never
uses an unbounded `SUPERSEDES*`, and a naive whole-file substring search
would false-positive on that explanation). Every other test is
`@pytest.mark.live` and exercises the real engine end to end — same split
`tests/test_traverse.py`/`tests/test_invalidate.py` already use.

Fixture-writing constraints reused verbatim from `tests/test_traverse.py`
(same live-engine findings, not re-discovered): every fixture uses the two
live-proven MERGE idioms `graph/writer.py`/`graph/invalidate.py` themselves
use — vertex upsert (`MERGE (n {id: row.vertex}) SET n:Label, ...`) then
relationship upsert (`MATCH (s...), (d...) MERGE (s)-[r:TYPE {id:...}]->(d)
...`).
"""

from __future__ import annotations

import ast
import inspect
import time

import pytest

from medmemgraph.contracts import SENTINEL_VALID_TO
from medmemgraph.demo import provenance as prov
from medmemgraph.hydra_client import HydraClient, validate_dialect
from medmemgraph.pipeline.ids import mint_patient_id

# ---------------------------------------------------------------------------
# Pure test — no live node. E8-S2 AC2.
# ---------------------------------------------------------------------------


def test_source_has_no_banned_cypher_shape() -> None:
    """Acceptance criterion 2 (E8-S2): every Cypher string this module
    passes to `client.run(...)` passes `validate_dialect`, and none of them
    contains an unbounded `SUPERSEDES*` or an unlabelled node pattern (the
    dialect gate itself already rejects both, but this test also asserts
    the *literal substring* is never present, matching the acceptance
    criterion's own wording)."""
    tree = ast.parse(inspect.getsource(prov))
    cypher_strings: list[str] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in ("run", "run_paths")
            and node.args
        ):
            try:
                literal = ast.literal_eval(node.args[0])
            except (ValueError, TypeError):
                continue
            if isinstance(literal, str):
                cypher_strings.append(literal)

    assert cypher_strings, "expected at least one client.run(...) call to inspect"
    for cypher in cypher_strings:
        validate_dialect(cypher)  # must not raise
        assert "SUPERSEDES*" not in cypher
        assert "*" not in cypher  # no variable-length pattern of any kind in this module


def test_claim_id_must_be_int() -> None:
    with pytest.raises(ValueError, match="claim_id"):
        prov.provenance_chain(object(), patient_id="whoever", claim_id="not-an-int")  # type: ignore[arg-type]


def test_render_chain_of_empty_result_does_not_crash() -> None:
    text = prov.render_chain({"nodes": [], "edges": [], "turns": [], "paths": []})
    assert "No claim found" in text


# ---------------------------------------------------------------------------
# Live fixtures + tests.
# ---------------------------------------------------------------------------

_BASE = 500_000_000 + (time.time_ns() // 1000) % 400_000_000
_counter = 0


def _next_id() -> int:
    global _counter
    _counter += 1
    return _BASE + _counter


@pytest.fixture()
def client():
    c = HydraClient(transport="bolt")
    yield c
    c.close()


_DEFAULT_CLAIM_PROPS = dict(
    predicate="CURRENT_DOSAGE_OF",
    polarity="asserted",
    source_class="patient",
    confidence=0.9,
    session_id="admission-1",
    observed_at="2026-01-01T00:00:00",
    valid_from="2026-01-01T00:00:00",
    valid_to=SENTINEL_VALID_TO,
    invalidated_at=SENTINEL_VALID_TO,
    status="active",
)


def _mint_patient(client: HydraClient, node_id: int, patient_id: str) -> None:
    client.run(
        "UNWIND $rows AS row MERGE (n {id: row.vertex}) SET n:Patient, n.patient_id = row.patient_id",
        rows=[{"vertex": node_id, "patient_id": patient_id}],
    )


def _new_patient(client: HydraClient, patient_key: str) -> int:
    """The Patient node's `id` MUST be `mint_patient_id(patient_key)` — the
    SAME derivation `provenance_chain` itself uses internally to resolve
    `patient_id` -> node id (matching `graph/writer.py`/
    `graph/invalidate.py`'s own established fixture convention, e.g.
    `tests/test_invalidate.py::_new_patient_with_claim`). Using an
    arbitrary integer here (as `tests/test_traverse.py`'s fixtures do,
    since `traverse.paths_between` is seeded by an explicit caller-given
    id, never by `mint_patient_id`) would silently make every
    `provenance_chain` ownership check in this file fail to match."""
    pid = mint_patient_id(patient_key)
    _mint_patient(client, pid, patient_key)
    return pid


def _mint_claim(client: HydraClient, node_id: int, *, fact_id: str, **overrides: object) -> None:
    props = dict(_DEFAULT_CLAIM_PROPS, fact_id=fact_id, **overrides)
    props["vertex"] = node_id
    client.run(
        "UNWIND $rows AS row MERGE (n {id: row.vertex}) SET n:Claim, "
        "n.fact_id = row.fact_id, n.predicate = row.predicate, n.polarity = row.polarity, "
        "n.source_class = row.source_class, n.confidence = row.confidence, "
        "n.session_id = row.session_id, n.observed_at = row.observed_at, "
        "n.valid_from = row.valid_from, n.valid_to = row.valid_to, "
        "n.invalidated_at = row.invalidated_at, n.status = row.status",
        rows=[props],
    )


def _mint_turn(client: HydraClient, node_id: int, *, session_id: str, turn_id: int, raw_text: str) -> None:
    client.run(
        "UNWIND $rows AS row MERGE (n {id: row.vertex}) SET n:Turn, "
        "n.session_id = row.session_id, n.turn_id = row.turn_id, n.raw_text = row.raw_text",
        rows=[{"vertex": node_id, "session_id": session_id, "turn_id": turn_id, "raw_text": raw_text}],
    )


def _link(client: HydraClient, src_label: str, src_id: int, dst_label: str, dst_id: int, rel_type: str) -> int:
    edge_id = _next_id()
    client.run(
        f"UNWIND $rows AS row MATCH (s:{src_label} {{id: row.s}}), (d:{dst_label} {{id: row.d}}) "
        f"MERGE (s)-[r:{rel_type} {{id: row.e}}]->(d) SET r.observed_at = row.obs",
        rows=[{"s": src_id, "d": dst_id, "e": edge_id, "obs": "2026-01-01T00:00:00"}],
    )
    return edge_id


@pytest.mark.live
def test_provenance_chain_orders_supersession_pair_with_turns(client: HydraClient) -> None:
    """E8-S2 AC1 + the "closed interval is visible" fixture requirement:
    a superseded claim pair, both with `DRAWN_FROM` turns. Both claims
    appear, in chronological order, the `SUPERSEDES` edge is present, the
    old claim's `valid_to` is a real (non-sentinel) value while the new
    claim's is still the sentinel, and at least one turn's text is
    returned."""
    cold, cnew, told, tnew = _next_id(), _next_id(), _next_id(), _next_id()
    patient_id = f"e8s2-patient-{_next_id()}"
    pid = _new_patient(client, patient_id)
    _mint_claim(
        client,
        cold,
        fact_id=f"e8s2-fact-{pid}-old",
        valid_from="2026-01-01T00:00:00",
        valid_to="2026-02-01T00:00:00",
        status="invalidated",
    )
    _mint_claim(
        client,
        cnew,
        fact_id=f"e8s2-fact-{pid}-new",
        valid_from="2026-02-01T00:00:00",
    )
    _link(client, "Patient", pid, "Claim", cold, "HAS")
    _link(client, "Patient", pid, "Claim", cnew, "HAS")
    _link(client, "Claim", cnew, "Claim", cold, "SUPERSEDES")

    _mint_turn(client, told, session_id="admission-1", turn_id=4, raw_text="Metformin 500mg twice daily.")
    _mint_turn(
        client, tnew, session_id="admission-2", turn_id=7, raw_text="Increase metformin to 1000mg twice daily."
    )
    _link(client, "Claim", cold, "Turn", told, "DRAWN_FROM")
    _link(client, "Claim", cnew, "Turn", tnew, "DRAWN_FROM")

    result = prov.provenance_chain(client, patient_id=patient_id, claim_id=cnew)

    # Both claims appear, oldest first.
    assert [n["id"] for n in result["nodes"]] == [cold, cnew]

    # The SUPERSEDES edge is present (new -> old, this schema's own direction).
    assert result["edges"] == [{"type": "SUPERSEDES", "src": cnew, "dst": cold}]

    old_node, new_node = result["nodes"]
    assert old_node["predicate"] == "CURRENT_DOSAGE_OF"
    assert old_node["status"] == "invalidated"
    assert old_node["valid_to"] == "2026-02-01T00:00:00"  # real, closed interval
    assert new_node["status"] == "active"
    assert new_node["valid_to"] == SENTINEL_VALID_TO  # still open

    # resolution_reason is never written anywhere yet (module docstring
    # point 2) — this asserts the documented, honest default, not a guess.
    assert old_node["resolution_reason"] == ""
    assert new_node["resolution_reason"] == ""

    turn_texts = {t["text"] for t in result["turns"]}
    assert "Metformin 500mg twice daily." in turn_texts
    assert "Increase metformin to 1000mg twice daily." in turn_texts

    # Each turn is stamped with the claim it evidences (additive field).
    turns_by_claim = {t["claim_id"]: t["text"] for t in result["turns"]}
    assert turns_by_claim[cold] == "Metformin 500mg twice daily."
    assert turns_by_claim[cnew] == "Increase metformin to 1000mg twice daily."

    # algo.MSpaths payloads, not raw driver Records (AC3).
    assert result["paths"], "expected at least one algo.MSpaths payload"
    for payload in result["paths"]:
        assert set(payload) == {"path", "pathWeight", "pathCost", "claim_ids", "hop_count"}
        assert isinstance(payload["path"], str)

    rendered = prov.render_chain(result)
    assert "SUPERSEDES" in rendered
    assert "Metformin 500mg twice daily." in rendered
    assert "Increase metformin to 1000mg twice daily." in rendered
    assert str(cold) in rendered and str(cnew) in rendered


@pytest.mark.live
def test_provenance_chain_single_claim_has_no_history(client: HydraClient) -> None:
    """E8-S2 AC4: a claim with no SUPERSEDES/CONTRADICTS neighbor returns a
    single node and no edges, not an error."""
    cid = _next_id()
    patient_id = f"e8s2-lonely-{_next_id()}"
    pid = _new_patient(client, patient_id)
    _mint_claim(client, cid, fact_id=f"e8s2-fact-{pid}-solo", predicate="HAS_CONDITION")
    _link(client, "Patient", pid, "Claim", cid, "HAS")

    result = prov.provenance_chain(client, patient_id=patient_id, claim_id=cid)

    assert [n["id"] for n in result["nodes"]] == [cid]
    assert result["edges"] == []
    assert result["paths"] == []

    rendered = prov.render_chain(result)
    assert str(cid) in rendered
    assert "-->" not in rendered  # no transition to render


@pytest.mark.live
def test_provenance_chain_returns_empty_for_a_claim_the_patient_does_not_own(client: HydraClient) -> None:
    """PHI-scoping guard (module docstring point 3): a real claim id that
    belongs to a DIFFERENT patient must never surface under this
    patient_id — degrades to the fully-empty shape, not an exception and
    not a leaked cross-patient node."""
    cid = _next_id()
    owner_patient_id = f"e8s2-owner-{_next_id()}"
    other_patient_id = f"e8s2-other-{_next_id()}"

    owner_pid = _new_patient(client, owner_patient_id)
    _new_patient(client, other_patient_id)
    _mint_claim(client, cid, fact_id=f"e8s2-fact-{owner_pid}-owned")
    _link(client, "Patient", owner_pid, "Claim", cid, "HAS")
    # deliberately NOT linked to other_pid

    result = prov.provenance_chain(client, patient_id=other_patient_id, claim_id=cid)
    assert result == {"nodes": [], "edges": [], "turns": [], "paths": []}


@pytest.mark.live
def test_provenance_chain_returns_empty_for_a_never_created_claim_id(client: HydraClient) -> None:
    patient_id = f"e8s2-neverclaim-{_next_id()}"
    _new_patient(client, patient_id)

    never_created_claim_id = _next_id()
    result = prov.provenance_chain(client, patient_id=patient_id, claim_id=never_created_claim_id)
    assert result == {"nodes": [], "edges": [], "turns": [], "paths": []}
