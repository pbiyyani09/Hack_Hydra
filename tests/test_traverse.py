"""tests/test_traverse.py — graph/traverse.py (ARCHITECTURE.md §7.3, story
E4-S6, `algo.MSpaths` wrapper + ranking + time filtering + serialization).

Pure, fast tests (unmarked, no live node — `_cypher_string_literal`
escaping, `rank_paths`/`serialize_paths`/`filter_paths_by_time` over
directly-constructed `Path`/`GraphNode`/`GraphRelationship` objects) sit
beside `@pytest.mark.live` tests exercising the real engine end to end —
same split `tests/test_invalidate.py` and `tests/test_run_dialect.py`
already use in this repo.

Fixture-writing constraints (same live-engine findings `tests/
test_invalidate.py` and `graph/writer.py`/`graph/invalidate.py` already
document, reused here rather than re-discovered): bare `CREATE` only
succeeds as one fresh, one-hop, BOTH-endpoints-new edge pattern, which is
too restrictive for this file's richer fixtures (a 3-hop chain, a 4-node
cycle). Every fixture here instead uses the two proven MERGE idioms
`graph/writer.py`/`graph/invalidate.py` themselves use: vertex upsert
(`MERGE (n {id: row.vertex}) SET n:Label, ...`) then relationship upsert
(`MATCH (s...), (d...) MERGE (s)-[r:TYPE {id:...}]->(d) ...`) — both
already dialect-gate-approved and live-proven idempotent.
"""

from __future__ import annotations

import time

import pytest

from medmemgraph.contracts import SENTINEL_VALID_TO
from medmemgraph.graph import traverse as tr
from medmemgraph.hydra_client import GraphNode, GraphRelationship, HydraClient

# ---------------------------------------------------------------------------
# Pure tests — no live node.
# ---------------------------------------------------------------------------


def test_cypher_string_literal_escapes_quotes_and_backslashes() -> None:
    literal = tr._cypher_string_literal("Alzheimer's disease \\ test")
    assert literal == "'Alzheimer\\'s disease \\\\ test'"


def _synthetic_node(node_id: int, label: str, **props: object) -> GraphNode:
    return GraphNode(id=node_id, labels=frozenset({label}), properties=dict(props))


def _synthetic_claim(
    node_id: int,
    *,
    predicate: str = "TAKES_MEDICATION",
    polarity: str = "asserted",
    status: str = "active",
    valid_from: str = "2026-01-01T00:00:00",
    valid_to: str = SENTINEL_VALID_TO,
    confidence: float = 0.9,
    fact_id: str | None = None,
) -> GraphNode:
    return _synthetic_node(
        node_id,
        "Claim",
        predicate=predicate,
        polarity=polarity,
        status=status,
        valid_from=valid_from,
        valid_to=valid_to,
        confidence=confidence,
        fact_id=fact_id or f"fact-{node_id}",
    )


def _synthetic_rel(rel_id: int, rel_type: str, start_id: int, end_id: int) -> GraphRelationship:
    return GraphRelationship(id=rel_id, type=rel_type, start_id=start_id, end_id=end_id, properties={})


def _synthetic_path(
    nodes: list[GraphNode], rels: list[GraphRelationship], *, weight: float = 0.0, cost: float = 0.0
) -> tr.Path:
    return tr.Path(nodes=tuple(nodes), relationships=tuple(rels), path_weight=weight, path_cost=cost)


def test_filter_paths_by_time_pure_drops_closed_interval_and_keeps_live_one() -> None:
    patient = _synthetic_node(1, "Patient", patient_id="p1")
    closed_claim = _synthetic_claim(2, valid_from="2026-01-01T00:00:00", valid_to="2026-02-01T00:00:00")
    med = _synthetic_node(3, "Medication", name="metformin")
    path = _synthetic_path(
        [patient, closed_claim, med],
        [_synthetic_rel(10, "HAS", 1, 2), _synthetic_rel(11, "ABOUT", 2, 3)],
    )

    # as_of AFTER the claim's valid_to -> dropped (closed interval, no longer live)
    assert tr.filter_paths_by_time([path], "2026-03-01T00:00:00") == []
    # as_of BEFORE the claim's valid_from -> also dropped (not yet live)
    assert tr.filter_paths_by_time([path], "2025-12-01T00:00:00") == []
    # as_of WITHIN [valid_from, valid_to) -> kept
    assert tr.filter_paths_by_time([path], "2026-01-15T00:00:00") == [path]
    # exactly at valid_to -> excluded (half-open interval, matches close_interval's own semantics)
    assert tr.filter_paths_by_time([path], "2026-02-01T00:00:00") == []
    # exactly at valid_from -> included
    assert tr.filter_paths_by_time([path], "2026-01-01T00:00:00") == [path]


def test_filter_paths_by_time_ignores_non_claim_intervals() -> None:
    # a path with no :Claim node at all always passes (nothing to be "not live")
    patient = _synthetic_node(1, "Patient", patient_id="p1")
    med = _synthetic_node(2, "Medication", name="metformin")
    path = _synthetic_path([patient, med], [_synthetic_rel(10, "ABOUT", 1, 2)])
    assert tr.filter_paths_by_time([path], "2099-01-01T00:00:00") == [path]


def test_rank_paths_prefers_shorter_and_more_confident_paths() -> None:
    short_confident = _synthetic_path(
        [_synthetic_node(1, "Patient", patient_id="p1"), _synthetic_claim(2, confidence=0.95)],
        [_synthetic_rel(10, "HAS", 1, 2)],
    )
    long_unconfident = _synthetic_path(
        [
            _synthetic_node(1, "Patient", patient_id="p1"),
            _synthetic_claim(3, confidence=0.2),
            _synthetic_claim(4, confidence=0.2),
            _synthetic_claim(5, confidence=0.2),
        ],
        [
            _synthetic_rel(11, "HAS", 1, 3),
            _synthetic_rel(12, "SUPERSEDES", 3, 4),
            _synthetic_rel(13, "SUPERSEDES", 4, 5),
        ],
    )
    ranked = tr.rank_paths([long_unconfident, short_confident])
    assert ranked == [short_confident, long_unconfident]


def test_rank_paths_recency_component_is_neutral_without_as_of() -> None:
    old_claim_path = _synthetic_path(
        [_synthetic_node(1, "Patient", patient_id="p1"), _synthetic_claim(2, valid_from="2020-01-01T00:00:00")],
        [_synthetic_rel(10, "HAS", 1, 2)],
    )
    # no as_of/half_life given -> recency term is a no-op; ranking falls back to
    # hop-count/confidence only, so a single equally-scored path just returns as-is.
    assert tr.rank_paths([old_claim_path]) == [old_claim_path]


def test_serialize_paths_respects_token_budget_and_never_truncates_a_path() -> None:
    p1 = _synthetic_path(
        [_synthetic_node(1, "Patient", patient_id="p1"), _synthetic_node(2, "Medication", name="metformin")],
        [_synthetic_rel(10, "ABOUT", 1, 2)],
    )
    p2 = _synthetic_path(
        [_synthetic_node(1, "Patient", patient_id="p1"), _synthetic_node(3, "Medication", name="lisinopril")],
        [_synthetic_rel(11, "ABOUT", 1, 3)],
    )
    full_text = tr.serialize_paths([p1, p2], token_budget=10_000)
    assert "metformin" in full_text and "lisinopril" in full_text

    # a budget too small for even one path -> empty string, never a partial line
    tiny_text = tr.serialize_paths([p1, p2], token_budget=1)
    assert tiny_text == ""

    # a budget that fits exactly the first path's rendered text -> only that
    # path appears, whole (not cut mid-render); the second is skipped entirely.
    one_path_budget = len(tr._render_path(p1)) // 2  # generous token estimate for one short line
    text = tr.serialize_paths([p1, p2], token_budget=one_path_budget)
    assert tr._render_path(p1) in text
    assert "lisinopril" not in text


def test_serialize_paths_zero_budget_is_empty() -> None:
    p1 = _synthetic_path(
        [_synthetic_node(1, "Patient", patient_id="p1"), _synthetic_node(2, "Medication", name="metformin")],
        [_synthetic_rel(10, "ABOUT", 1, 2)],
    )
    assert tr.serialize_paths([p1], token_budget=0) == ""


def test_paths_between_rejects_unknown_seed_label() -> None:
    with pytest.raises(ValueError, match="schema.LABELS"):
        tr.paths_between(object(), [1], seed_label="NotARealLabel")  # type: ignore[arg-type]


def test_paths_between_rejects_target_ids_without_target_label() -> None:
    with pytest.raises(ValueError, match="target_label"):
        tr.paths_between(object(), [1], [2], seed_label="Patient")  # type: ignore[arg-type]


def test_paths_between_empty_seed_ids_short_circuits_without_a_client_call() -> None:
    # a bogus "client" (not a real HydraClient) would raise loudly if this
    # module tried to use it — proving the empty-seed-ids short circuit
    # (ARCHITECTURE §7.2: "if the seed set is empty, skip algo.MSpaths")
    # never even touches the client.
    assert tr.paths_between(object(), [], seed_label="Patient") == []  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Live fixtures + tests.
# ---------------------------------------------------------------------------

_BASE = 300_000_000 + (time.time_ns() // 1000) % 600_000_000
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
    predicate="TAKES_MEDICATION",
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


def _mint_entity(client: HydraClient, node_id: int, label: str, name: str) -> None:
    client.run(
        "UNWIND $rows AS row MERGE (n {id: row.vertex}) SET n:"
        + label
        + ", n.name = row.name, n.type = row.type",
        rows=[{"vertex": node_id, "name": name, "type": label}],
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
def test_known_two_hop_path_is_found(client: HydraClient) -> None:
    """Patient -HAS-> Claim -ABOUT-> Medication, exactly one 2-hop path."""
    pid, cid, med_id = _next_id(), _next_id(), _next_id()
    pkey = f"e4s6-2hop-{pid}"
    mname = f"metformin-{pid}"
    _mint_patient(client, pid, pkey)
    _mint_claim(client, cid, fact_id=f"fact-{pid}-1")
    _mint_entity(client, med_id, "Medication", mname)
    _link(client, "Patient", pid, "Claim", cid, "HAS")
    _link(client, "Claim", cid, "Medication", med_id, "ABOUT")

    paths = tr.paths_between(client, [pid], [med_id], seed_label="Patient", target_label="Medication")
    assert len(paths) == 1
    path = paths[0]
    assert path.hop_count == 2
    assert path.start_id == pid
    assert path.end_id == med_id
    assert path.claim_ids == (cid,)


@pytest.mark.live
def test_known_three_hop_path_is_found(client: HydraClient) -> None:
    """Patient -HAS-> ClaimNew -SUPERSEDES-> ClaimOld -ABOUT-> Medication:
    a realistic 3-hop chain (a dose-change claim reached via its superseded
    predecessor's own ABOUT edge)."""
    pid, cnew, cold, med_id = _next_id(), _next_id(), _next_id(), _next_id()
    pkey = f"e4s6-3hop-{pid}"
    mname = f"lisinopril-{pid}"
    _mint_patient(client, pid, pkey)
    _mint_claim(client, cnew, fact_id=f"fact-{pid}-new", valid_from="2026-02-01T00:00:00")
    _mint_claim(
        client,
        cold,
        fact_id=f"fact-{pid}-old",
        valid_from="2026-01-01T00:00:00",
        valid_to="2026-02-01T00:00:00",
        status="invalidated",
    )
    _mint_entity(client, med_id, "Medication", mname)
    _link(client, "Patient", pid, "Claim", cnew, "HAS")
    _link(client, "Claim", cnew, "Claim", cold, "SUPERSEDES")
    _link(client, "Claim", cold, "Medication", med_id, "ABOUT")

    paths = tr.paths_between(client, [pid], [med_id], seed_label="Patient", target_label="Medication", max_len=4)
    assert len(paths) == 1
    path = paths[0]
    assert path.hop_count == 3
    assert path.claim_ids == (cnew, cold)
    assert [rel.type for rel in path.relationships] == ["HAS", "SUPERSEDES", "ABOUT"]


@pytest.mark.live
def test_maxlen_above_16_is_clamped_and_query_still_succeeds(client: HydraClient) -> None:
    pid, cid, med_id = _next_id(), _next_id(), _next_id()
    pkey = f"e4s6-clamp-{pid}"
    mname = f"metformin-{pid}"
    _mint_patient(client, pid, pkey)
    _mint_claim(client, cid, fact_id=f"fact-{pid}-1")
    _mint_entity(client, med_id, "Medication", mname)
    _link(client, "Patient", pid, "Claim", cid, "HAS")
    _link(client, "Claim", cid, "Medication", med_id, "ABOUT")

    with pytest.warns(UserWarning, match="clamping"):
        paths = tr.paths_between(
            client, [pid], [med_id], seed_label="Patient", target_label="Medication", max_len=9001
        )
    assert len(paths) == 1  # the (clamped, engine-accepted) call still ran and found the path


@pytest.mark.live
def test_valid_time_filtering_drops_a_closed_interval_path_end_to_end(client: HydraClient) -> None:
    """A claim closed (SUPERSEDES'd) before the as_of date must not appear
    once `filter_paths_by_time` runs, even though `algo.MSpaths` itself
    happily returns it (the procedure has no interval concept at all)."""
    pid, cold, med_id = _next_id(), _next_id(), _next_id()
    pkey = f"e4s6-timefilter-{pid}"
    mname = f"metformin-{pid}"
    _mint_patient(client, pid, pkey)
    _mint_claim(
        client,
        cold,
        fact_id=f"fact-{pid}-closed",
        valid_from="2026-01-01T00:00:00",
        valid_to="2026-02-01T00:00:00",
        status="invalidated",
    )
    _mint_entity(client, med_id, "Medication", mname)
    _link(client, "Patient", pid, "Claim", cold, "HAS")
    _link(client, "Claim", cold, "Medication", med_id, "ABOUT")

    paths = tr.paths_between(client, [pid], [med_id], seed_label="Patient", target_label="Medication")
    assert len(paths) == 1  # algo.MSpaths itself doesn't know or care that this claim is closed

    as_of_after_close = "2026-06-01T00:00:00"
    assert tr.filter_paths_by_time(paths, as_of_after_close) == []

    as_of_while_live = "2026-01-15T00:00:00"
    assert len(tr.filter_paths_by_time(paths, as_of_while_live)) == 1


@pytest.mark.live
def test_safety_filter_rejects_a_selector_value_collision_across_patients(client: HydraClient) -> None:
    """Two different patients each have a Medication node named identically
    (a real collision surface, module docstring point 4: `name` is not
    patient-scoped in this schema). Seeding by patient A and targeting ONLY
    patient A's own medication id must never let patient B's same-named
    node leak into the results, even though both match the `name` selector
    algo.MSpaths is seeded by."""
    pid_a, cid_a, med_a = _next_id(), _next_id(), _next_id()
    pid_b, cid_b, med_b = _next_id(), _next_id(), _next_id()
    shared_name = f"collision-med-{pid_a}"

    _mint_patient(client, pid_a, f"e4s6-collide-a-{pid_a}")
    _mint_claim(client, cid_a, fact_id=f"fact-{pid_a}-a")
    _mint_entity(client, med_a, "Medication", shared_name)
    _link(client, "Patient", pid_a, "Claim", cid_a, "HAS")
    _link(client, "Claim", cid_a, "Medication", med_a, "ABOUT")

    _mint_patient(client, pid_b, f"e4s6-collide-b-{pid_b}")
    _mint_claim(client, cid_b, fact_id=f"fact-{pid_b}-b")
    _mint_entity(client, med_b, "Medication", shared_name)  # SAME name, different patient/node
    _link(client, "Patient", pid_b, "Claim", cid_b, "HAS")
    _link(client, "Claim", cid_b, "Medication", med_b, "ABOUT")

    # Seed by patient A only, but target med_b's REAL id (never patient A's
    # own medication) -- the resolved selector value ("shared_name") matches
    # BOTH med_a and med_b, but med_b is not reachable from patient A's own
    # subgraph, so this should still legitimately return nothing.
    cross_patient = tr.paths_between(
        client, [pid_a], [med_b], seed_label="Patient", target_label="Medication"
    )
    assert cross_patient == []

    # Sanity: patient A really can reach ITS OWN same-named medication.
    same_patient = tr.paths_between(
        client, [pid_a], [med_a], seed_label="Patient", target_label="Medication"
    )
    assert len(same_patient) == 1
    assert same_patient[0].end_id == med_a


@pytest.mark.live
def test_cyclic_fixture_matches_hand_computed_reachability(client: HydraClient) -> None:
    """Correctness guard for bugs #69/#71/#83 (literature/10 §E4,
    ARCHITECTURE §7.3): a known 3-cycle (a->b->c->a) plus a spur (a->d), all
    `:Claim` nodes linked by `SUPERSEDES`. Hand-computed truth: from `a`,
    outgoing-direction reachability within 2 hops is exactly {b, c, d}
    (b and d directly, c via a->b->c). Both `algo.MSpaths` (via
    `paths_between`, target_ids=None) and the equivalent raw bounded
    variable-length `MATCH` must agree with this hand-computed set. If a
    future engine build disagrees, that IS the #69/#71/#83 signal this test
    exists to catch — this module already exclusively uses `algo.MSpaths`
    everywhere (never a raw variable-length `MATCH`) specifically because
    of that documented risk; this test does not change that policy, it
    guards it.
    """
    a, b, c_node, d = _next_id(), _next_id(), _next_id(), _next_id()
    for node_id, letter in [(a, "a"), (b, "b"), (c_node, "c"), (d, "d")]:
        _mint_claim(client, node_id, fact_id=f"e4s6-cycle-{a}-{letter}")
    _link(client, "Claim", a, "Claim", b, "SUPERSEDES")
    _link(client, "Claim", b, "Claim", c_node, "SUPERSEDES")
    _link(client, "Claim", c_node, "Claim", a, "SUPERSEDES")  # closes the cycle
    _link(client, "Claim", a, "Claim", d, "SUPERSEDES")  # spur

    expected_reachable = {b, c_node, d}

    # algo.MSpaths, via this module's own wrapper (no fixed target -> SSpaths-like reachability)
    mspaths_paths = tr.paths_between(
        client, [a], None, seed_label="Claim", rel_types=["SUPERSEDES"], rel_direction="outgoing", max_len=2
    )
    mspaths_reachable = {p.end_id for p in mspaths_paths}
    assert mspaths_reachable == expected_reachable

    # The equivalent raw bounded variable-length MATCH, same fixture.
    # EXECUTED FINDING (live, not in literature/10): a variable-length MATCH's
    # source node pattern must be anchored by `id` specifically — matching by
    # any other property (e.g. `fact_id`) is rejected: "OpenCypher query is
    # not supported yet: variable-length MATCH requires a fixed source id".
    # This is a DIFFERENT constraint from algo.MSpaths's own `sourceProperty`
    # mechanism (module docstring point 1 in traverse.py), and one more
    # reason this module never uses a raw variable-length MATCH for real
    # traversal — it can't even be seeded the same way `paths_between`'s
    # public contract (integer ids resolved via a label+property lookup) is.
    raw_rows = client.run(
        "MATCH (n:Claim {id: $aid})-[:SUPERSEDES*1..2]->(m:Claim) RETURN m.id AS id",
        aid=a,
    )
    raw_reachable = {row["id"] for row in raw_rows}
    assert raw_reachable == expected_reachable, (
        "raw variable-length MATCH disagreed with the hand-computed truth on this cyclic "
        "fixture -- this IS the #69/#71/#83 signal (literature/10 §E4); algo.MSpaths is "
        "already used exclusively on every correctness-critical path in this module, so no "
        "code change follows from this assertion firing, but it must be investigated, not "
        "silenced."
    )
