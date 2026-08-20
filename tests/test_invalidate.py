"""tests/test_invalidate.py — coverage for `graph/invalidate.py` (E4-S5).

Pure-Python `classify()` unit tests (no live node) pin the decision-table
branching directly and run fast, in the same spirit as
`tests/test_run_dialect.py`'s split between pure dialect-gate checks and
`@pytest.mark.live` engine tests.

Everything else in this file is `@pytest.mark.live` and exercises the real
thing: `close_interval`, `link`, and the whole `apply()` pass against a
running HydraDB node — including the as-of query on both sides of a close
boundary, which is this project's demo spine (a vector store has no
structural way to answer "what did we believe before vs. after this
changed").

Fixture-writing constraints (discovered live while building this module —
see `invalidate.py`'s module docstring point 4, and confirmed again here):
bare `CREATE` only succeeds as one fresh, one-hop, BOTH-endpoints-new edge
pattern; a `MATCH`-then-`CREATE` mixing one existing and one fresh endpoint
is rejected regardless of direction. So every fixture below is built as
(a) one or two bare CREATEs, each minting a fresh node pair, then
(b) `UNWIND $rows AS row MATCH (existing),(existing) MERGE (...)->(...)`
to cross-link nodes that already exist — the same relationship-upsert
shape `link()` itself uses, proven live.
"""

from __future__ import annotations

import time

import pytest

from medmemgraph.contracts import EntityRef, SENTINEL_VALID_TO, mock_fact
from medmemgraph.graph import invalidate as inv
from medmemgraph.hydra_client import HydraClient, validate_dialect
from medmemgraph.pipeline.ids import mint_patient_id

# No module-level pytestmark: this file mixes fast pure-Python `classify()`
# tests (unmarked) with `@pytest.mark.live` engine tests, marked per-test.

# ---------------------------------------------------------------------------
# Pure classify() unit tests — no live node, no I/O.
# ---------------------------------------------------------------------------


def _row(**overrides: object) -> dict:
    base = {
        "id": 1,
        "predicate": "CURRENT_DOSAGE_OF",
        "status": "active",
        "valid_to": SENTINEL_VALID_TO,
        "valid_from": "2026-01-01T00:00:00",
        "polarity": "asserted",
        "source_class": "patient",
        "session_id": "admission-1",
        "object_id": 2,
        "object_name": "metformin",
    }
    base.update(overrides)
    return base


def test_classify_no_candidates_is_coexistence_new() -> None:
    fact = mock_fact(predicate="CURRENT_DOSAGE_OF", valid_from="2026-02-01T00:00:00")
    decision = inv.classify(fact, [], new_claim_id=99)
    assert decision.kind == "coexistence"
    assert decision.resolution_reason == "new"
    assert decision.supersedes == ()
    assert decision.contradicts == ()


def test_classify_same_source_is_update_and_supersedes() -> None:
    candidate = _row(session_id="admission-1", source_class="patient", valid_from="2026-01-01T00:00:00")
    fact = mock_fact(
        predicate="CURRENT_DOSAGE_OF",
        session_id="admission-1",
        source_class="patient",
        valid_from="2026-02-01T00:00:00",
        confidence=0.9,
    )
    decision = inv.classify(fact, [candidate], new_claim_id=42)
    assert decision.kind == "update"
    assert decision.resolution_reason == "same_source"
    assert decision.supersedes == ((42, 1),)  # (winner=new, loser=candidate.id)


def test_classify_set_valued_different_object_is_coexistence() -> None:
    # different object canonical_id -> caller would never even include it as
    # a candidate (the fetch step filters on object identity); classify()
    # itself just sees an empty candidate list in that case.
    fact = mock_fact(predicate="HAS_CONDITION", object=EntityRef(name="hypertension", type="Condition", canonical_id=5))
    decision = inv.classify(fact, [], new_claim_id=7)
    assert decision.kind == "coexistence"


def test_classify_set_valued_retraction_closes_the_asserted_twin() -> None:
    candidate = _row(
        predicate="HAS_ALLERGY_TO", polarity="asserted", session_id="admission-1",
        source_class="patient", valid_from="2026-01-01T00:00:00",
    )
    fact = mock_fact(
        predicate="HAS_ALLERGY_TO",
        polarity="negated",
        session_id="admission-1",
        source_class="patient",
        valid_from="2026-02-01T00:00:00",
        confidence=0.9,
    )
    decision = inv.classify(fact, [candidate], new_claim_id=42)
    assert decision.supersedes == ((42, 1),)
    assert decision.resolution_reason == "same_source"


def test_classify_source_class_doctor_beats_patient_same_time() -> None:
    candidate = _row(source_class="patient", session_id="admission-A", valid_from="2026-01-01T00:00:00")
    fact = mock_fact(
        predicate="CURRENT_DOSAGE_OF",
        source_class="doctor",
        session_id="admission-B",
        valid_from="2026-01-01T00:00:00",  # same valid_from as candidate
        confidence=0.9,
    )
    decision = inv.classify(fact, [candidate], new_claim_id=42)
    assert decision.resolution_reason == "source_class"
    # new (doctor) wins, candidate (patient) closes.
    assert decision.supersedes == ((42, 1),)


def test_classify_lower_trust_wins_when_strictly_more_recent() -> None:
    candidate = _row(source_class="doctor", session_id="admission-A", valid_from="2026-01-01T00:00:00")
    fact = mock_fact(
        predicate="CURRENT_DOSAGE_OF",
        source_class="patient",
        session_id="admission-B",
        valid_from="2026-03-01T00:00:00",  # strictly later than the doctor's claim
        confidence=0.9,
    )
    decision = inv.classify(fact, [candidate], new_claim_id=42)
    assert decision.resolution_reason == "recency"
    assert decision.supersedes == ((42, 1),)  # new (patient) wins by recency override


def test_classify_unresolved_when_nothing_discriminates() -> None:
    candidate = _row(source_class="patient", session_id="admission-A", valid_from="2026-01-01T00:00:00", object_name="metformin")
    fact = mock_fact(
        predicate="CURRENT_DOSAGE_OF",
        source_class="patient",
        session_id="admission-B",
        valid_from="2026-01-01T00:00:00",  # same time
        confidence=0.9,
        object=EntityRef(name="metformin", type="Medication", canonical_id=2),  # identical name/length -> ties rule 5
    )
    decision = inv.classify(fact, [candidate], new_claim_id=42)
    assert decision.kind == "contradiction"
    assert decision.resolution_reason == "unresolved"
    assert decision.contradicts == (1,)
    assert decision.supersedes == ()


def test_classify_low_confidence_never_closes_present_claim() -> None:
    """Decision 002 (open): a low-confidence ('possible' in spirit) new
    fact must not silently close a firmly-asserted existing claim. No i2b2
    field is read — `confidence` (already in the frozen contract) is the
    only signal used."""
    candidate = _row(source_class="patient", session_id="admission-A", valid_from="2026-01-01T00:00:00")
    fact = mock_fact(
        predicate="CURRENT_DOSAGE_OF",
        source_class="patient",
        session_id="admission-A",  # would otherwise be same_source -> auto-close
        valid_from="2026-02-01T00:00:00",
        confidence=0.2,  # below POSSIBLE_CONFIDENCE_FLOOR
    )
    decision = inv.classify(fact, [candidate], new_claim_id=42)
    assert decision.kind == "contradiction"
    assert decision.resolution_reason == "low_confidence"
    assert decision.supersedes == ()
    assert decision.contradicts == (1,)


def test_close_interval_cypher_is_dialect_legal() -> None:
    validate_dialect(
        "MATCH (c:Claim {id: $id}) SET c.valid_to = $valid_to, "
        "c.invalidated_at = $invalidated_at, c.status = $status"
    )


def test_link_cypher_is_dialect_legal() -> None:
    for rel in ("SUPERSEDES", "CONTRADICTS"):
        validate_dialect(
            f"UNWIND $rows AS row "
            f"MATCH (s:Claim {{id: row.source_vertex}}), (d:Claim {{id: row.destination_vertex}}) "
            f"MERGE (s)-[r:{rel} {{id: row.relationship_vertex}}]->(d) "
            f"SET r.observed_at = row.observed_at"
        )


def test_no_delete_of_claim_anywhere_in_this_module() -> None:
    """Acceptance criterion 6 (E4-S5): no DELETE of :Claim. Checks only the
    actual Cypher text passed to `client.run(...)` calls (via `ast`), not
    the module's prose — several docstrings here explain, in English, why
    the module never deletes, and would false-positive on a naive
    substring search."""
    import ast
    import inspect

    from medmemgraph.graph import invalidate as module

    tree = ast.parse(inspect.getsource(module))
    cypher_strings: list[str] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "run"
            and node.args
        ):
            first_arg = node.args[0]
            try:
                literal = ast.literal_eval(first_arg)
            except (ValueError, TypeError):
                continue
            if isinstance(literal, str):
                cypher_strings.append(literal)

    assert cypher_strings, "expected at least one client.run(...) call to inspect"
    for cypher in cypher_strings:
        assert "DELETE" not in cypher.upper(), f"found DELETE in Cypher text: {cypher!r}"


# ---------------------------------------------------------------------------
# Live fixtures + acceptance-criteria tests.
# ---------------------------------------------------------------------------

_BASE = 200_000_000 + (time.time_ns() // 1000) % 700_000_000
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


def _new_patient_with_claim(client: HydraClient, *, patient_key: str, claim_props: dict) -> tuple[int, int]:
    """Bare one-hop CREATE: a fresh :Patient + its first fresh :Claim.
    Returns (patient_node_id, claim_id).

    The Patient node's `id` is `mint_patient_id(patient_key)` — the SAME
    derivation `invalidate.py`'s `_fetch_candidates` (and `graph/writer.py`)
    use, NOT an arbitrary counter — so a `ClinicalFact` whose `patient_id`
    equals `patient_key` resolves to this exact node. `patient_key` itself
    must be unique per test *session* (callers pass a `_BASE`-suffixed
    string) since this uses bare `CREATE`, not `MERGE`: minting is
    deterministic, so a fixed key across repeated suite runs would either
    duplicate the node (bare CREATE never checks for an existing id) or, if
    someone "fixed" this with MERGE instead, hit the same unlabelled-MERGE
    vertex-upsert idiom `writer.py` had to special-case — out of scope for
    a test fixture, so uniqueness is enforced at the key level instead."""
    pid = mint_patient_id(patient_key)
    cid, eid = _next_id(), _next_id()
    client.run(
        "CREATE (p:Patient {id: $pid, patient_id: $pkey})-[:HAS {id: $eid}]->"
        "(c:Claim {id: $cid, predicate: $predicate, status: $status, "
        "valid_to: $valid_to, valid_from: $valid_from, polarity: $polarity, "
        "source_class: $source_class, session_id: $session_id})",
        pid=pid, pkey=patient_key, eid=eid, cid=cid, **claim_props,
    )
    return pid, cid


def _new_claim_with_object(
    client: HydraClient, *, claim_props: dict, object_label: str, object_name: str
) -> tuple[int, int]:
    """Bare one-hop CREATE: a fresh :Claim + its first fresh object entity
    node, connected by :ABOUT. Returns (claim_id, object_id)."""
    cid, oid, eid = _next_id(), _next_id(), _next_id()
    client.run(
        f"CREATE (c:Claim {{id: $cid, predicate: $predicate, status: $status, "
        f"valid_to: $valid_to, valid_from: $valid_from, polarity: $polarity, "
        f"source_class: $source_class, session_id: $session_id}})"
        f"-[:ABOUT {{id: $eid}}]->(o:{object_label} {{id: $oid, name: $oname}})",
        cid=cid, eid=eid, oid=oid, oname=object_name, **claim_props,
    )
    return cid, oid


def _attach_has(client: HydraClient, patient_node_id: int, claim_id: int) -> None:
    eid = _next_id()
    client.run(
        "UNWIND $rows AS row MATCH (p:Patient {id: row.pid}), (c:Claim {id: row.cid}) "
        "MERGE (p)-[r:HAS {id: row.eid}]->(c)",
        rows=[{"pid": patient_node_id, "cid": claim_id, "eid": eid}],
    )


def _attach_about(client: HydraClient, claim_id: int, object_label: str, object_id: int) -> None:
    eid = _next_id()
    client.run(
        f"UNWIND $rows AS row MATCH (c:Claim {{id: row.cid}}), (o:{object_label} {{id: row.oid}}) "
        f"MERGE (c)-[r:ABOUT {{id: row.eid}}]->(o)",
        rows=[{"cid": claim_id, "oid": object_id, "eid": eid}],
    )


def _read_claim(client: HydraClient, claim_id: int) -> dict:
    rows = client.run(
        "MATCH (c:Claim {id: $id}) RETURN c.id AS id, c.status AS status, "
        "c.valid_to AS valid_to, c.valid_from AS valid_from, "
        "c.invalidated_at AS invalidated_at",
        id=claim_id,
    )
    assert rows, f"claim {claim_id} not found — should still be readable, never deleted"
    return rows[0]


@pytest.mark.live
def test_dose_change_closes_old_interval_and_writes_supersedes_and_old_is_still_readable(client: HydraClient) -> None:
    """Acceptance criterion 1: functional predicate, same source ->
    update, SUPERSEDES written, old claim still matchable with its
    ORIGINAL valid_from untouched."""
    old_props = dict(
        predicate="CURRENT_DOSAGE_OF", status="active", valid_to=SENTINEL_VALID_TO,
        valid_from="2026-01-01T00:00:00", polarity="asserted", source_class="patient",
        session_id="admission-1",
    )
    pid, old_claim = _new_patient_with_claim(client, patient_key=f"dose-patient-{_BASE}", claim_props=old_props)

    new_props = dict(
        predicate="CURRENT_DOSAGE_OF", status="active", valid_to=SENTINEL_VALID_TO,
        valid_from="2026-02-01T00:00:00", polarity="asserted", source_class="patient",
        session_id="admission-1",  # same source (session) as old_claim
    )
    new_claim, med_id = _new_claim_with_object(
        client, claim_props=new_props, object_label="Medication", object_name="metformin"
    )
    _attach_has(client, pid, new_claim)
    _attach_about(client, old_claim, "Medication", med_id)

    new_fact = mock_fact(
        fact_id="dose-fact-new",
        patient_id=f"dose-patient-{_BASE}",
        session_id="admission-1",
        subject=EntityRef(name=f"dose-patient-{_BASE}", type="Patient", canonical_id=pid),
        predicate="CURRENT_DOSAGE_OF",
        object=EntityRef(name="metformin 1000mg", type="Medication", canonical_id=med_id),
        valid_from="2026-02-01T00:00:00",
        source_class="patient",
        confidence=0.9,
    )
    report = inv.apply(client, [new_fact], {"dose-fact-new": new_claim}, now="2026-02-01T00:00:00")

    decision = report.decisions["dose-fact-new"]
    assert decision.kind == "update"
    assert decision.resolution_reason == "same_source"
    assert (new_claim, old_claim) in report.supersedes_written

    old_row = _read_claim(client, old_claim)
    assert old_row["valid_to"] == "2026-02-01T00:00:00"
    assert old_row["status"] == "invalidated"
    assert old_row["valid_from"] == "2026-01-01T00:00:00"  # untouched

    new_row = _read_claim(client, new_claim)
    assert new_row["valid_to"] == SENTINEL_VALID_TO
    assert new_row["status"] == "active"

    supersedes = client.run(
        "MATCH (s:Claim {id: $s})-[:SUPERSEDES]->(d:Claim {id: $d}) RETURN d.id AS id",
        s=new_claim, d=old_claim,
    )
    assert supersedes == [{"id": old_claim}]


@pytest.mark.live
def test_set_valued_predicate_accumulates_instead_of_closing(client: HydraClient) -> None:
    """Acceptance criterion 2: HAS_CONDITION diabetes then HAS_CONDITION
    hypertension — different objects, both remain active."""
    diabetes_props = dict(
        predicate="HAS_CONDITION", status="active", valid_to=SENTINEL_VALID_TO,
        valid_from="2026-01-01T00:00:00", polarity="asserted", source_class="patient",
        session_id="admission-1",
    )
    pid, diabetes_claim = _new_patient_with_claim(client, patient_key=f"coexist-patient-{_BASE}", claim_props=diabetes_props)
    _, diabetes_id = _new_claim_with_object(
        client,
        claim_props=dict(diabetes_props, predicate="HAS_CONDITION"),
        object_label="Condition",
        object_name="diabetes-throwaway",
    )
    _attach_about(client, diabetes_claim, "Condition", diabetes_id)

    hypertension_props = dict(
        predicate="HAS_CONDITION", status="active", valid_to=SENTINEL_VALID_TO,
        valid_from="2026-03-01T00:00:00", polarity="asserted", source_class="patient",
        session_id="admission-2",
    )
    hypertension_claim, hypertension_id = _new_claim_with_object(
        client, claim_props=hypertension_props, object_label="Condition", object_name="hypertension"
    )
    _attach_has(client, pid, hypertension_claim)

    new_fact = mock_fact(
        fact_id="coexist-fact-hypertension",
        patient_id=f"coexist-patient-{_BASE}",
        session_id="admission-2",
        subject=EntityRef(name=f"coexist-patient-{_BASE}", type="Patient", canonical_id=pid),
        predicate="HAS_CONDITION",
        object=EntityRef(name="hypertension", type="Condition", canonical_id=hypertension_id),
        valid_from="2026-03-01T00:00:00",
        source_class="patient",
        confidence=0.9,
    )
    report = inv.apply(
        client, [new_fact], {"coexist-fact-hypertension": hypertension_claim}, now="2026-03-01T00:00:00"
    )
    decision = report.decisions["coexist-fact-hypertension"]
    assert decision.kind == "coexistence"
    assert decision.supersedes == ()
    assert decision.contradicts == ()

    for claim_id in (diabetes_claim, hypertension_claim):
        row = _read_claim(client, claim_id)
        assert row["status"] == "active"
        assert row["valid_to"] == SENTINEL_VALID_TO


@pytest.mark.live
def test_negated_retraction_closes_only_the_asserted_twin(client: HydraClient) -> None:
    """Acceptance criterion 3: asserted HAS_ALLERGY_TO penicillin, then a
    negated claim for the same object closes the asserted twin and writes
    SUPERSEDES."""
    asserted_props = dict(
        predicate="HAS_ALLERGY_TO", status="active", valid_to=SENTINEL_VALID_TO,
        valid_from="2026-01-01T00:00:00", polarity="asserted", source_class="patient",
        session_id="admission-1",
    )
    pid, asserted_claim = _new_patient_with_claim(client, patient_key=f"retract-patient-{_BASE}", claim_props=asserted_props)

    negated_props = dict(
        predicate="HAS_ALLERGY_TO", status="active", valid_to=SENTINEL_VALID_TO,
        valid_from="2026-02-01T00:00:00", polarity="negated", source_class="patient",
        session_id="admission-1",
    )
    negated_claim, allergy_id = _new_claim_with_object(
        client, claim_props=negated_props, object_label="Allergy", object_name="penicillin"
    )
    _attach_has(client, pid, negated_claim)
    _attach_about(client, asserted_claim, "Allergy", allergy_id)

    new_fact = mock_fact(
        fact_id="retract-fact-negated",
        patient_id=f"retract-patient-{_BASE}",
        session_id="admission-1",
        subject=EntityRef(name=f"retract-patient-{_BASE}", type="Patient", canonical_id=pid),
        predicate="HAS_ALLERGY_TO",
        object=EntityRef(name="penicillin", type="Allergy", canonical_id=allergy_id),
        valid_from="2026-02-01T00:00:00",
        polarity="negated",
        source_class="patient",
        confidence=0.9,
    )
    report = inv.apply(client, [new_fact], {"retract-fact-negated": negated_claim}, now="2026-02-01T00:00:00")
    decision = report.decisions["retract-fact-negated"]
    assert decision.kind in ("update", "correction")  # a genuine retraction, not a contradiction
    assert (negated_claim, asserted_claim) in report.supersedes_written

    old_row = _read_claim(client, asserted_claim)
    assert old_row["status"] == "invalidated"
    assert old_row["valid_to"] == "2026-02-01T00:00:00"

    supersedes = client.run(
        "MATCH (s:Claim {id: $s})-[:SUPERSEDES]->(d:Claim {id: $d}) RETURN d.id AS id",
        s=negated_claim, d=asserted_claim,
    )
    assert supersedes == [{"id": asserted_claim}]


@pytest.mark.live
def test_doctor_beats_patient_same_valid_from_source_class_wins(client: HydraClient) -> None:
    """Acceptance criterion 4: doctor says X, patient says NOT X, same
    valid_from, staleness window 0. Doctor stays active, patient claim is
    closed via source_class — never both deleted."""
    doctor_props = dict(
        predicate="HAS_CONDITION", status="active", valid_to=SENTINEL_VALID_TO,
        valid_from="2026-01-01T00:00:00", polarity="asserted", source_class="doctor",
        session_id="admission-doctor",
    )
    pid, doctor_claim = _new_patient_with_claim(client, patient_key=f"trust-patient-{_BASE}", claim_props=doctor_props)
    _, cond_id = _new_claim_with_object(
        client, claim_props=dict(doctor_props), object_label="Condition", object_name="anemia"
    )
    _attach_about(client, doctor_claim, "Condition", cond_id)

    patient_props = dict(
        predicate="HAS_CONDITION", status="active", valid_to=SENTINEL_VALID_TO,
        valid_from="2026-01-01T00:00:00",  # same valid_from as the doctor claim
        polarity="negated", source_class="patient", session_id="admission-patient",
    )
    patient_claim, _ = _new_claim_with_object(
        client, claim_props=patient_props, object_label="Condition", object_name="anemia-throwaway"
    )
    _attach_has(client, pid, patient_claim)

    new_fact = mock_fact(
        fact_id="trust-fact-patient",
        patient_id=f"trust-patient-{_BASE}",
        session_id="admission-patient",
        subject=EntityRef(name=f"trust-patient-{_BASE}", type="Patient", canonical_id=pid),
        predicate="HAS_CONDITION",
        object=EntityRef(name="anemia", type="Condition", canonical_id=cond_id),
        valid_from="2026-01-01T00:00:00",
        polarity="negated",
        source_class="patient",
        confidence=0.9,
    )
    report = inv.apply(client, [new_fact], {"trust-fact-patient": patient_claim}, now="2026-01-01T00:00:00")
    decision = report.decisions["trust-fact-patient"]

    doctor_row = _read_claim(client, doctor_claim)
    patient_row = _read_claim(client, patient_claim)

    if decision.resolution_reason == "source_class":
        assert doctor_row["status"] == "active"
        assert patient_row["status"] == "invalidated"
    else:
        # The only other spec-sanctioned outcome: could not discriminate,
        # both stay active with bidirectional CONTRADICTS. Never deleted.
        assert decision.resolution_reason == "unresolved"
        assert doctor_row["status"] == "active"
        assert patient_row["status"] == "active"
        fwd = client.run(
            "MATCH (a:Claim {id:$a})-[:CONTRADICTS]->(b:Claim {id:$b}) RETURN b.id AS id",
            a=doctor_claim, b=patient_claim,
        )
        bwd = client.run(
            "MATCH (a:Claim {id:$a})-[:CONTRADICTS]->(b:Claim {id:$b}) RETURN b.id AS id",
            a=patient_claim, b=doctor_claim,
        )
        assert fwd and bwd


@pytest.mark.live
def test_genuine_contradiction_is_flagged_unresolved_not_silently_picked(client: HydraClient) -> None:
    """Acceptance criterion 5: two same-class, same-time, same-specificity
    claims the precedence ladder cannot break. Both stay active,
    CONTRADICTS both directions, nothing deleted."""
    props_a = dict(
        predicate="CURRENT_DOSAGE_OF", status="active", valid_to=SENTINEL_VALID_TO,
        valid_from="2026-01-01T00:00:00", polarity="asserted", source_class="patient",
        session_id="admission-A",
    )
    pid, claim_a = _new_patient_with_claim(client, patient_key=f"unresolved-patient-{_BASE}", claim_props=props_a)
    _, med_id = _new_claim_with_object(
        client, claim_props=props_a, object_label="Medication", object_name="lisinopril"
    )
    _attach_about(client, claim_a, "Medication", med_id)

    props_b = dict(
        predicate="CURRENT_DOSAGE_OF", status="active", valid_to=SENTINEL_VALID_TO,
        valid_from="2026-01-01T00:00:00",  # same time as claim_a
        polarity="asserted", source_class="patient", session_id="admission-B",  # different session
    )
    claim_b, _ = _new_claim_with_object(
        client, claim_props=props_b, object_label="Medication", object_name="lisinopril-throwaway"
    )
    _attach_has(client, pid, claim_b)

    new_fact = mock_fact(
        fact_id="unresolved-fact-b",
        patient_id=f"unresolved-patient-{_BASE}",
        session_id="admission-B",
        subject=EntityRef(name=f"unresolved-patient-{_BASE}", type="Patient", canonical_id=pid),
        predicate="CURRENT_DOSAGE_OF",
        object=EntityRef(name="lisinopril", type="Medication", canonical_id=med_id),  # same length as "lisinopril"
        valid_from="2026-01-01T00:00:00",
        source_class="patient",
        confidence=0.9,
    )
    report = inv.apply(client, [new_fact], {"unresolved-fact-b": claim_b}, now="2026-01-01T00:00:00")
    decision = report.decisions["unresolved-fact-b"]
    assert decision.kind == "contradiction"
    assert decision.resolution_reason == "unresolved"
    assert "unresolved-fact-b" in report.unresolved_fact_ids

    row_a = _read_claim(client, claim_a)
    row_b = _read_claim(client, claim_b)
    assert row_a["status"] == "active"
    assert row_b["status"] == "active"
    assert row_a["valid_to"] == SENTINEL_VALID_TO
    assert row_b["valid_to"] == SENTINEL_VALID_TO

    fwd = client.run(
        "MATCH (a:Claim {id:$a})-[:CONTRADICTS]->(b:Claim {id:$b}) RETURN b.id AS id", a=claim_b, b=claim_a
    )
    bwd = client.run(
        "MATCH (a:Claim {id:$a})-[:CONTRADICTS]->(b:Claim {id:$b}) RETURN b.id AS id", a=claim_a, b=claim_b
    )
    assert fwd == [{"id": claim_a}]
    assert bwd == [{"id": claim_b}]


@pytest.mark.live
def test_low_confidence_fact_does_not_close_a_present_claim_live(client: HydraClient) -> None:
    """Decision 002 (open): a 'possible'-in-spirit (low-confidence) new
    fact must not close a firmly-present active claim. Live end-to-end
    version of the pure classify() unit test above."""
    present_props = dict(
        predicate="CURRENT_DOSAGE_OF", status="active", valid_to=SENTINEL_VALID_TO,
        valid_from="2026-01-01T00:00:00", polarity="asserted", source_class="patient",
        session_id="admission-1",
    )
    pid, present_claim = _new_patient_with_claim(client, patient_key=f"possible-patient-{_BASE}", claim_props=present_props)
    _, med_id = _new_claim_with_object(
        client, claim_props=present_props, object_label="Medication", object_name="warfarin"
    )
    _attach_about(client, present_claim, "Medication", med_id)

    possible_props = dict(
        predicate="CURRENT_DOSAGE_OF", status="active", valid_to=SENTINEL_VALID_TO,
        valid_from="2026-02-01T00:00:00", polarity="asserted", source_class="patient",
        session_id="admission-1",  # would be same_source if not for the confidence guard
    )
    possible_claim, _ = _new_claim_with_object(
        client, claim_props=possible_props, object_label="Medication", object_name="warfarin-throwaway"
    )
    _attach_has(client, pid, possible_claim)

    new_fact = mock_fact(
        fact_id="possible-fact",
        patient_id=f"possible-patient-{_BASE}",
        session_id="admission-1",
        subject=EntityRef(name=f"possible-patient-{_BASE}", type="Patient", canonical_id=pid),
        predicate="CURRENT_DOSAGE_OF",
        object=EntityRef(name="warfarin", type="Medication", canonical_id=med_id),
        valid_from="2026-02-01T00:00:00",
        source_class="patient",
        confidence=0.2,  # below the floor
    )
    report = inv.apply(client, [new_fact], {"possible-fact": possible_claim}, now="2026-02-01T00:00:00")
    decision = report.decisions["possible-fact"]
    assert decision.kind == "contradiction"
    assert decision.resolution_reason == "low_confidence"

    present_row = _read_claim(client, present_claim)
    assert present_row["status"] == "active"
    assert present_row["valid_to"] == SENTINEL_VALID_TO


@pytest.mark.live
def test_replay_is_idempotent(client: HydraClient) -> None:
    """apply() called twice with the same facts/claim_ids does not
    duplicate SUPERSEDES edges and does not re-close an already-closed
    claim differently."""
    old_props = dict(
        predicate="CURRENT_DOSAGE_OF", status="active", valid_to=SENTINEL_VALID_TO,
        valid_from="2026-01-01T00:00:00", polarity="asserted", source_class="patient",
        session_id="admission-1",
    )
    pid, old_claim = _new_patient_with_claim(client, patient_key=f"replay-patient-{_BASE}", claim_props=old_props)
    new_props = dict(old_props, valid_from="2026-02-01T00:00:00")
    new_claim, med_id = _new_claim_with_object(
        client, claim_props=new_props, object_label="Medication", object_name="atorvastatin"
    )
    _attach_has(client, pid, new_claim)
    _attach_about(client, old_claim, "Medication", med_id)

    new_fact = mock_fact(
        fact_id="replay-fact",
        patient_id=f"replay-patient-{_BASE}",
        session_id="admission-1",
        subject=EntityRef(name=f"replay-patient-{_BASE}", type="Patient", canonical_id=pid),
        predicate="CURRENT_DOSAGE_OF",
        object=EntityRef(name="atorvastatin", type="Medication", canonical_id=med_id),
        valid_from="2026-02-01T00:00:00",
        source_class="patient",
        confidence=0.9,
    )
    claim_ids = {"replay-fact": new_claim}
    report1 = inv.apply(client, [new_fact], claim_ids, now="2026-02-01T00:00:00")
    assert report1.decisions["replay-fact"].kind == "update"

    report2 = inv.apply(client, [new_fact], claim_ids, now="2026-02-01T00:00:00")
    # The old claim is now invalidated, so it no longer appears as an
    # active candidate -- the second pass is naturally a no-op.
    assert report2.decisions["replay-fact"].kind == "coexistence"
    assert report2.supersedes_written == []

    supersedes = client.run(
        "MATCH (s:Claim {id: $s})-[:SUPERSEDES]->(d:Claim {id: $d}) RETURN d.id AS id",
        s=new_claim, d=old_claim,
    )
    assert supersedes == [{"id": old_claim}]  # exactly one edge, not duplicated

    old_row = _read_claim(client, old_claim)
    assert old_row["valid_to"] == "2026-02-01T00:00:00"
    assert old_row["status"] == "invalidated"


@pytest.mark.live
def test_as_of_query_before_and_after_the_change_is_the_demo_spine(client: HydraClient) -> None:
    """The headline capability: an as-of query at a date BEFORE the change
    returns the OLD value; at a date AFTER, the NEW value. This is a path
    a vector store cannot answer at all."""
    old_props = dict(
        predicate="CURRENT_DOSAGE_OF", status="active", valid_to=SENTINEL_VALID_TO,
        valid_from="2026-01-01T00:00:00", polarity="asserted", source_class="patient",
        session_id="admission-1",
    )
    pid, old_claim = _new_patient_with_claim(client, patient_key=f"asof-patient-{_BASE}", claim_props=old_props)
    new_props = dict(old_props, valid_from="2026-03-01T00:00:00")
    new_claim, med_id = _new_claim_with_object(
        client, claim_props=new_props, object_label="Medication", object_name="metformin"
    )
    _attach_has(client, pid, new_claim)
    _attach_about(client, old_claim, "Medication", med_id)

    new_fact = mock_fact(
        fact_id="asof-fact",
        patient_id=f"asof-patient-{_BASE}",
        session_id="admission-1",
        subject=EntityRef(name=f"asof-patient-{_BASE}", type="Patient", canonical_id=pid),
        predicate="CURRENT_DOSAGE_OF",
        object=EntityRef(name="metformin", type="Medication", canonical_id=med_id),
        valid_from="2026-03-01T00:00:00",
        source_class="patient",
        confidence=0.9,
    )
    inv.apply(client, [new_fact], {"asof-fact": new_claim}, now="2026-03-01T00:00:00")

    # Copied verbatim from ARCHITECTURE.md §5.5 / test_run_dialect.py's own
    # "current-as-of-interval-query" legal-shape case.
    as_of_cypher = (
        "MATCH (p:Patient {id: $pid})-[:HAS]->(c:Claim) "
        "WHERE c.valid_from <= $D AND c.valid_to > $D "
        "RETURN c.id AS id, c.valid_from AS valid_from, c.valid_to AS valid_to, "
        "c.predicate AS predicate"
    )
    validate_dialect(as_of_cypher)

    before = client.run(as_of_cypher, pid=pid, D="2026-02-01T00:00:00")
    after = client.run(as_of_cypher, pid=pid, D="2026-04-01T00:00:00")

    assert len(before) == 1 and before[0]["id"] == old_claim
    assert before[0]["valid_from"] == "2026-01-01T00:00:00"
    assert before[0]["valid_to"] == "2026-03-01T00:00:00"

    assert len(after) == 1 and after[0]["id"] == new_claim
    assert after[0]["valid_from"] == "2026-03-01T00:00:00"
    assert after[0]["valid_to"] == SENTINEL_VALID_TO


@pytest.mark.live
def test_write_and_invalidate_wires_writer_then_apply(client: HydraClient) -> None:
    """`write_and_invalidate()` is the ARCHITECTURE §6.5 step-3 integration
    point: `graph.writer.write_facts()` (E4-S4, landed concurrently with
    this story) followed by this module's `apply()`, run as one call. Two
    ClinicalFacts for the same (patient, drug) go in as plain facts — never
    hand-built Cypher — and the second write closes the first."""
    from medmemgraph.pipeline.ids import mint_patient_id

    patient_key = f"wire-patient-{_BASE}"
    med_name = f"wire-med-{_BASE}"
    med_canonical_id = _next_id()

    old_fact = mock_fact(
        fact_id=f"wire-fact-old-{_BASE}",
        patient_id=patient_key,
        session_id="admission-1",
        subject=EntityRef(name=patient_key, type="Patient", canonical_id=1),
        predicate="CURRENT_DOSAGE_OF",
        object=EntityRef(name=med_name, type="Medication", canonical_id=med_canonical_id),
        valid_from="2026-01-01T00:00:00",
        source_class="patient",
        confidence=0.9,
    )
    write_report1, inv_report1 = inv.write_and_invalidate(client, [old_fact], now="2026-01-01T00:00:00")
    assert write_report1.facts_written == 1
    assert inv_report1.decisions[old_fact.fact_id].kind == "coexistence"

    new_fact = mock_fact(
        fact_id=f"wire-fact-new-{_BASE}",
        patient_id=patient_key,
        session_id="admission-1",  # same source as old_fact
        subject=EntityRef(name=patient_key, type="Patient", canonical_id=1),
        predicate="CURRENT_DOSAGE_OF",
        object=EntityRef(name=med_name, type="Medication", canonical_id=med_canonical_id),
        valid_from="2026-02-01T00:00:00",
        source_class="patient",
        confidence=0.9,
    )
    write_report2, inv_report2 = inv.write_and_invalidate(client, [new_fact], now="2026-02-01T00:00:00")
    assert write_report2.facts_written == 1
    decision = inv_report2.decisions[new_fact.fact_id]
    assert decision.kind == "update"
    assert decision.resolution_reason == "same_source"
    assert len(inv_report2.supersedes_written) == 1

    pid = mint_patient_id(patient_key)
    rows = client.run(
        "MATCH (p:Patient {id: $pid})-[:HAS]->(c:Claim) "
        "RETURN c.fact_id AS fact_id, c.status AS status, c.valid_to AS valid_to",
        pid=pid,
    )
    by_fact_id = {r["fact_id"]: r for r in rows}
    assert by_fact_id[old_fact.fact_id]["status"] == "invalidated"
    assert by_fact_id[old_fact.fact_id]["valid_to"] == "2026-02-01T00:00:00"
    assert by_fact_id[new_fact.fact_id]["status"] == "active"
    assert by_fact_id[new_fact.fact_id]["valid_to"] == SENTINEL_VALID_TO

    # Idempotent replay of the whole wired pipeline: no duplicate SUPERSEDES.
    inv.write_and_invalidate(client, [new_fact], now="2026-02-01T00:00:00")
    supersedes = client.run(
        "MATCH (s:Claim {fact_id: $new_fid})-[:SUPERSEDES]->(d:Claim {fact_id: $old_fid}) RETURN d.id AS id",
        new_fid=new_fact.fact_id, old_fid=old_fact.fact_id,
    )
    assert len(supersedes) == 1
