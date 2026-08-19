"""tests/test_retrieve.py — `graph/router.py` + `graph/fusion.py` +
`graph/retrieve.py` (ARCHITECTURE.md §7, stories E5-S1/E5-S4/E6-S2).

Split, same convention `tests/test_traverse.py`/`tests/test_invalidate.py`
already use: pure/fast tests (router rule, RRF math, the vector-route path
through `retrieve()` — no live node touched at all, since a pure-vector
route never opens a `HydraClient`) sit beside `@pytest.mark.live` tests that
exercise the real engine end to end for the graph arm and `structural_absence`.

Live fixtures use the two live-proven MERGE idioms `tests/test_traverse.py`
and `graph/writer.py`/`graph/invalidate.py` already document: vertex upsert
(`MERGE (n {id: row.vertex}) SET n:Label, ...`) then relationship upsert
(`MATCH (s...), (d...) MERGE (s)-[r:TYPE {id:...}]->(d) ...`) — bare `CREATE`
only supports one fresh, one-hop, both-endpoints-new edge pattern, too
restrictive for a multi-node fixture (decisions/003).
"""

from __future__ import annotations

import random
import time

import pytest

from medmemgraph.contracts import RetrieveItem, RetrieveResult
from medmemgraph.graph import fusion, retrieve as retrieve_mod, router
from medmemgraph.graph.lexical import LexicalIndex
from medmemgraph.graph.vector_index import PatientIndex
from medmemgraph.hydra_client import validate_dialect, HydraClient
from medmemgraph.pipeline.ids import mint_patient_id
from medmemgraph.pipeline.loader import Admission, Conversation

# ---------------------------------------------------------------------------
# router.py — pure, no live node.
# ---------------------------------------------------------------------------


def test_route_eval_cross_admission_routes_to_graph_no_flip() -> None:
    for scope, question_type in [("cross_admission", None), (None, "longitudinal_progression")]:
        decision = router.route_eval(scope, question_type, epsilon=0.0)
        assert decision.route == "graph"
        assert decision.epsilon_flipped is False


def test_route_eval_single_admission_routes_to_vector_no_flip() -> None:
    for scope, question_type in [("single_admission", None), (None, "medical_reasoning")]:
        decision = router.route_eval(scope, question_type, epsilon=0.0)
        assert decision.route == "vector"
        assert decision.epsilon_flipped is False


def test_route_eval_unclassified_defaults_to_hybrid() -> None:
    decision = router.route_eval(None, None, epsilon=0.0)
    assert decision.route == "hybrid"


def test_route_eval_epsilon_one_flips_graph_to_vector_deterministically() -> None:
    decision = router.route_eval("cross_admission", None, epsilon=1.0)
    assert decision.route == "vector"
    assert decision.epsilon_flipped is True
    assert decision.propensity == pytest.approx(1.0)


def test_route_live_cross_admission_keyword_routes_to_graph() -> None:
    decision = router.route_live("how has her A1c changed across admissions?", epsilon=0.0)
    assert decision.route == "graph"


def test_route_live_this_stay_keyword_routes_to_vector() -> None:
    decision = router.route_live("why did they choose this plan for this admission?", epsilon=0.0)
    assert decision.route == "vector"


def test_epsilon_flip_rate_is_within_tolerance_of_epsilon_over_1000_draws() -> None:
    """E5-S1 AC5: 1,000 draws at ε=0.05 on a fixed graph item, seeded rng —
    flip rate within 0.02 of 0.05."""
    rng = random.Random(20260816)
    n = 1000
    flips = sum(
        1 for _ in range(n) if router.route_eval("cross_admission", None, epsilon=0.05, rng=rng).epsilon_flipped
    )
    assert abs(flips / n - 0.05) <= 0.02


def test_route_log_records_a_nonzero_propensity_for_the_unchosen_action() -> None:
    """The whole point of ε-logging: a deterministic router logs propensity
    0 for the arm it never picks, which makes offline policy evaluation
    permanently impossible to retrofit. Here, epsilon=1.0 forces the router
    to log the arm the DETERMINISTIC rule would never have chosen for a
    cross_admission item ("vector") — with a real, non-zero propensity."""
    router.ROUTE_LOG.clear()

    unflipped = router.route_eval("cross_admission", None, epsilon=0.0)
    router.log_route_decision(question="q-unflipped", features={"scope": "cross_admission"}, decision=unflipped)

    flipped = router.route_eval("cross_admission", None, epsilon=1.0)
    router.log_route_decision(question="q-flipped", features={"scope": "cross_admission"}, decision=flipped)

    assert len(router.ROUTE_LOG) == 2
    assert router.ROUTE_LOG[0]["route"] == "graph"
    assert router.ROUTE_LOG[0]["propensity"] == pytest.approx(1.0)
    assert router.ROUTE_LOG[0]["epsilon_flipped"] is False

    # The unchosen-under-the-rule action ("vector") was logged WITH a
    # nonzero propensity, not silently dropped or logged at propensity 0.
    assert router.ROUTE_LOG[1]["route"] == "vector"
    assert router.ROUTE_LOG[1]["epsilon_flipped"] is True
    assert router.ROUTE_LOG[1]["propensity"] == pytest.approx(1.0)  # == epsilon here (epsilon=1.0)
    assert router.ROUTE_LOG[1]["propensity"] > 0.0


# ---------------------------------------------------------------------------
# fusion.py — pure RRF math, hand-built ranked lists.
# ---------------------------------------------------------------------------


def _item(text: str, *, channel: str = "vector") -> RetrieveItem:
    return RetrieveItem(text=text, session_id="s", turn_ids=[1], score=0.0, channel=channel)


def test_rrf_fuse_includes_items_unique_to_either_list_with_correct_scores() -> None:
    """E5-S4 AC3: item A is #1 lexical, absent from dense; item B is #1
    dense, absent from lexical. Both appear; scores equal 1/(60+rank) sums."""
    item_a = _item("evidence A", channel="lexical")
    item_b = _item("evidence B", channel="vector")
    dense = [item_b]
    lexical = [item_a]

    fused = fusion.rrf_fuse([dense, lexical], k=60)

    by_text = {item.text: item for item in fused}
    assert set(by_text) == {"evidence A", "evidence B"}
    assert by_text["evidence A"].score == pytest.approx(1.0 / (60 + 1))
    assert by_text["evidence B"].score == pytest.approx(1.0 / (60 + 1))


def test_rrf_fuse_sums_contributions_from_both_lists_and_ranks_descending() -> None:
    item_a = _item("A")
    item_b = _item("B")
    item_c = _item("C")
    dense = [item_a, item_b, item_c]  # ranks 1,2,3
    lexical = [item_b, item_a]  # ranks 1,2 — C absent from lexical

    fused = fusion.rrf_fuse([dense, lexical], k=60)
    by_text = {item.text: item.score for item in fused}

    assert by_text["A"] == pytest.approx(1.0 / 61 + 1.0 / 62)  # dense rank1 + lexical rank2
    assert by_text["B"] == pytest.approx(1.0 / 62 + 1.0 / 61)  # dense rank2 + lexical rank1
    assert by_text["C"] == pytest.approx(1.0 / 63)  # dense rank3 only

    # B > A > C by the arithmetic above (1/61+1/62 == 1/62+1/61, a tie —
    # both strictly exceed C's single-list contribution).
    ordered = [item.text for item in fused]
    assert ordered.index("C") == 2
    assert set(ordered[:2]) == {"A", "B"}


def test_rrf_fuse_weights_are_documented_and_equal_by_default() -> None:
    assert fusion.DEFAULT_WEIGHT == 1.0
    assert fusion.DEFAULT_RRF_K == 60


def test_rrf_fuse_rejects_mismatched_weights_length() -> None:
    with pytest.raises(ValueError):
        fusion.rrf_fuse([[_item("A")], [_item("B")]], weights=[1.0])


def test_rrf_fuse_empty_lists_returns_empty() -> None:
    assert fusion.rrf_fuse([[], []]) == []


# ---------------------------------------------------------------------------
# retrieve() — contract shape + vector route. No live node: a pure-vector
# route never opens a HydraClient (§7.1's own 26-point / 100-350x lesson).
# ---------------------------------------------------------------------------


def _synthetic_conversation(patient_id: str) -> Conversation:
    lines = [
        {
            "turn_number": 1,
            "time": "2026-01-01T00:00:00",
            "speaker": "Doctor",
            "text": "How is your blood pressure medication working for you?",
        },
        {
            "turn_number": 2,
            "time": "2026-01-01T00:01:00",
            "speaker": "Patient",
            "text": "I've been taking lisinopril every morning and it seems fine so far.",
        },
        {
            "turn_number": 3,
            "time": "2026-01-01T00:02:00",
            "speaker": "Doctor",
            "text": "Any dizziness or other side effects from the lisinopril dose?",
        },
    ]
    admission = Admission(
        hadm_id="admission-0001",
        admission_start="2026-01-01T00:00:00",
        admission_end="2026-01-01T01:00:00",
        conversation_lines=tuple(lines),
    )
    return Conversation(subject_id=patient_id, processed_hadm_ids=("admission-0001",), admissions=(admission,))


@pytest.fixture()
def registered_text_indexes():
    retrieve_mod.clear_index_cache()
    patient_id = "test-retrieve-vector-patient"
    conversation = _synthetic_conversation(patient_id)
    dense = PatientIndex(patient_id, cache_path=None)
    dense.build(conversation)
    lexical = LexicalIndex(patient_id)
    lexical.build(conversation)
    retrieve_mod.register_indexes(patient_id, dense, lexical)
    yield patient_id
    retrieve_mod.clear_index_cache()


def test_retrieve_returns_the_frozen_contract_shape(registered_text_indexes: str) -> None:
    result = retrieve_mod.retrieve(
        "why this medication plan for this admission?",
        registered_text_indexes,
        3,
        scope="single_admission",
        question_type="medical_reasoning",
        epsilon=0.0,
    )
    assert isinstance(result, RetrieveResult)
    assert isinstance(result.items, list)
    assert all(isinstance(it, RetrieveItem) for it in result.items)
    assert result.route in ("graph", "vector", "hybrid")
    assert isinstance(result.structural_absence, bool)
    assert isinstance(result.paths, list)
    assert set(result.latency_ms) == {"search", "total", "graph", "vector", "lexical", "rerank"}
    assert all(isinstance(v, (int, float)) for v in result.latency_ms.values())
    assert result.latency_ms["total"] >= 0.0


def test_retrieve_single_admission_routes_to_vector_and_opens_no_graph_client(
    registered_text_indexes: str,
) -> None:
    result = retrieve_mod.retrieve(
        "why this medication plan for this admission?",
        registered_text_indexes,
        3,
        scope="single_admission",
        question_type="medical_reasoning",
        epsilon=0.0,
    )
    assert result.route == "vector"
    assert result.structural_absence is False
    assert result.paths == []
    assert len(result.items) <= 3
    assert all(item.channel in ("vector", "lexical") for item in result.items)
    # No graph work was ever attempted on this route.
    assert result.latency_ms["graph"] == 0.0
    assert result.latency_ms["search"] == 0.0


def test_retrieve_vector_route_structural_absence_is_false_even_with_low_cosine(
    registered_text_indexes: str,
) -> None:
    """E6-S2 AC3: pure vector route, structural_absence is False even when
    the question has nothing to do with the indexed corpus (low cosine
    everywhere) — absence is a graph predicate, never a similarity
    threshold."""
    result = retrieve_mod.retrieve(
        "completely unrelated question about spacecraft telemetry",
        registered_text_indexes,
        3,
        scope="single_admission",
        question_type="medical_reasoning",
        epsilon=0.0,
    )
    assert result.route == "vector"
    assert result.structural_absence is False


def test_retrieve_k_truncates_the_fused_result(registered_text_indexes: str) -> None:
    result = retrieve_mod.retrieve(
        "medication",
        registered_text_indexes,
        1,
        scope="single_admission",
        question_type="medical_reasoning",
        epsilon=0.0,
    )
    assert len(result.items) <= 1


def test_retrieve_degrades_gracefully_when_no_text_corpus_exists() -> None:
    """A patient_id with no MedLoCoMo corpus AND no registered index ->
    text arm is [], never a raise, vector route still returns the frozen
    shape."""
    retrieve_mod.clear_index_cache()
    result = retrieve_mod.retrieve(
        "why this plan for this admission?",
        "patient-with-no-corpus-anywhere",
        3,
        scope="single_admission",
        question_type="medical_reasoning",
        epsilon=0.0,
    )
    assert result.route == "vector"
    assert result.items == []
    assert result.structural_absence is False


# ---------------------------------------------------------------------------
# retrieve() — graph route + structural_absence. Live node required.
# ---------------------------------------------------------------------------

_BASE = 20_000_000 + (time.time_ns() // 1000) % 79_000_000


def _next_id() -> int:
    global _BASE
    _BASE += 1
    return _BASE


_DEFAULT_CLAIM_PROPS = dict(
    predicate="HAS_ALLERGY_TO",
    polarity="asserted",
    source_class="patient",
    confidence=0.9,
    session_id="admission-1",
    observed_at="2026-01-01T00:00:00",
    valid_from="2026-01-01T00:00:00",
    valid_to="9999-12-31T00:00:00",
    invalidated_at="9999-12-31T00:00:00",
    status="active",
)


@pytest.fixture()
def client():
    c = HydraClient(transport="bolt")
    yield c
    c.close()


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
        "UNWIND $rows AS row MERGE (n {id: row.vertex}) SET n:" + label + ", n.name = row.name, n.type = row.type",
        rows=[{"vertex": node_id, "name": name, "type": label}],
    )


def _link(client: HydraClient, src_label: str, src_id: int, dst_label: str, dst_id: int, rel_type: str) -> None:
    edge_id = _next_id()
    client.run(
        f"UNWIND $rows AS row MATCH (s:{src_label} {{id: row.s}}), (d:{dst_label} {{id: row.d}}) "
        f"MERGE (s)-[r:{rel_type} {{id: row.e}}]->(d) SET r.observed_at = row.obs",
        rows=[{"s": src_id, "d": dst_id, "e": edge_id, "obs": "2026-01-01T00:00:00"}],
    )


@pytest.mark.live
def test_retrieve_cross_admission_routes_to_graph_and_finds_a_present_entity(client: HydraClient) -> None:
    """Positive direction of the both-directions structural_absence
    fixture: a patient WITH a matching domain entity -> `structural_absence`
    does NOT fire, the graph arm contributes at least one path, and the
    router's cross_admission rule surfaces as `route in {"graph","hybrid"}`."""
    suffix = _next_id()
    patient_key = f"retrieve-present-{suffix}"
    pid = mint_patient_id(patient_key)
    cid = _next_id()
    eid = _next_id()

    _mint_patient(client, pid, patient_key)
    _mint_claim(client, cid, fact_id=f"retrieve-present-fact-{suffix}")
    _mint_entity(client, eid, "Allergy", f"penicillin-{suffix}")
    _link(client, "Patient", pid, "Claim", cid, "HAS")
    _link(client, "Claim", cid, "Allergy", eid, "ABOUT")

    result = retrieve_mod.retrieve(
        "does she have any allergies?",
        patient_key,
        5,
        scope="cross_admission",
        question_type="cross_admission_comparison",
        epsilon=0.0,
        client=client,
    )

    assert result.route in ("graph", "hybrid")
    assert result.structural_absence is False
    assert len(result.paths) >= 1
    assert any(item.channel == "graph" for item in result.items)
    assert result.latency_ms["graph"] > 0.0


@pytest.mark.live
def test_retrieve_structural_absence_fires_for_a_patient_with_no_domain_entities(client: HydraClient) -> None:
    """Negative direction: a patient node that genuinely exists but has no
    domain entities to seed a walk from at all -> `structural_absence` is
    True and `items == []` (§7.6 rule 4, retrieve.py's own design decision
    2: this implementation's absence signal is patient-wide, not
    entity-specific — the coarsest, but always-honest, absence condition it
    can produce without an NER step)."""
    suffix = _next_id()
    patient_key = f"retrieve-absent-{suffix}"
    pid = mint_patient_id(patient_key)
    _mint_patient(client, pid, patient_key)  # patient exists; zero Claims, zero entities

    result = retrieve_mod.retrieve(
        "does she have any allergies?",
        patient_key,
        5,
        scope="cross_admission",
        question_type="cross_admission_comparison",
        epsilon=0.0,
        client=client,
    )

    assert result.route == "graph"
    assert result.structural_absence is True
    assert result.items == []
    assert result.paths == []


@pytest.mark.live
def test_retrieve_structural_absence_fires_for_a_never_created_patient(client: HydraClient) -> None:
    """A patient_id that was never written to the graph at all -> the cheap
    labelled patient-existence probe fails and `structural_absence` fires
    before any seed/MSpaths work is attempted."""
    result = retrieve_mod.retrieve(
        "does she have any allergies?",
        "retrieve-never-created-patient-does-not-exist",
        5,
        scope="cross_admission",
        question_type="cross_admission_comparison",
        epsilon=0.0,
        client=client,
    )
    assert result.structural_absence is True
    assert result.items == []
    assert result.latency_ms["search"] > 0.0


@pytest.mark.live
def test_retrieve_latency_ms_is_populated_on_the_live_graph_path(client: HydraClient) -> None:
    result = retrieve_mod.retrieve(
        "does she have any allergies?",
        "retrieve-never-created-patient-does-not-exist",
        5,
        scope="cross_admission",
        question_type="cross_admission_comparison",
        epsilon=0.0,
        client=client,
    )
    assert set(result.latency_ms) == {"search", "total", "graph", "vector", "lexical", "rerank"}
    assert result.latency_ms["total"] >= result.latency_ms["search"]


# ---------------------------------------------------------------------------
# retrieve_for_eval() — pins epsilon=0 for anything that reaches a results
# table (decisions/005 Finding 1: the reconciliation audit found `retrieve()`
# defaults to `DEFAULT_EPSILON=0.05` and neither `eval/harness.py` nor
# `eval/reader.py` referenced `epsilon` at all, so a reported table would
# have silently carried ~5% epsilon-flipped routes). No live node required
# — every case here stays on the pure-vector path via the same registered
# synthetic indexes `retrieve()`'s own vector-route tests above use.
# ---------------------------------------------------------------------------


def test_eval_epsilon_constant_is_zero() -> None:
    assert retrieve_mod.EVAL_EPSILON == 0.0


def test_retrieve_for_eval_rejects_an_explicit_epsilon_override(registered_text_indexes: str) -> None:
    with pytest.raises(ValueError):
        retrieve_mod.retrieve_for_eval(
            "why this medication plan?", registered_text_indexes, 3, epsilon=0.05
        )


def test_retrieve_for_eval_calls_retrieve_with_epsilon_zero_never_the_live_default(monkeypatch) -> None:
    """The exact regression this guards: `retrieve()`'s own default is
    `DEFAULT_EPSILON` (0.05, live use — E5-S1). `retrieve_for_eval` must
    always pass `epsilon=0.0` through explicitly rather than relying on
    (or silently inheriting) that default."""
    seen_epsilons: list[float | None] = []

    def spy_retrieve(question, patient_id, k, *, epsilon=retrieve_mod.DEFAULT_EPSILON, **kwargs):
        seen_epsilons.append(epsilon)
        return RetrieveResult(items=[], route="vector", structural_absence=False, paths=[], latency_ms={})

    monkeypatch.setattr(retrieve_mod, "retrieve", spy_retrieve)

    retrieve_mod.retrieve_for_eval("why this medication plan?", "some-patient", 3)

    assert seen_epsilons == [0.0]


def test_retrieve_for_eval_returns_the_frozen_contract_shape(registered_text_indexes: str) -> None:
    result = retrieve_mod.retrieve_for_eval(
        "why this medication plan for this admission?", registered_text_indexes, 3
    )
    assert isinstance(result, RetrieveResult)
    assert result.route in ("graph", "vector", "hybrid")


class TestEntityTimeline:
    """`entity_timeline` — complete, chronological coverage of one entity.

    The query top-k similarity structurally cannot express. Observed failure it
    fixes: asked to compare the etiology of headache between two dates,
    retrieval returned 6 items from 3 sessions and none were headache claims,
    while the graph held exactly three headache claims at the dates asked
    about."""

    def test_rejects_a_non_entity_label(self):
        """Label is interpolated into the query string (it cannot be a
        parameter in a node pattern), so it must be validated against the closed
        vocabulary or it is an injection point."""
        with pytest.raises(ValueError, match="not a domain entity label"):
            retrieve_mod.entity_timeline(object(), "p1", 123, "Claim")

    def test_label_injection_is_refused(self):
        with pytest.raises(ValueError):
            retrieve_mod.entity_timeline(object(), "p1", 123, "Symptom) RETURN 1 //")

    def test_query_is_dialect_legal(self):
        """Every node pattern labelled, relationship directed, no IN/IS NULL/
        CASE — the shapes `hydra_client.validate_dialect` rejects."""
        captured = {}

        class _FakeClient:
            def run(self, cypher, **params):
                captured["cypher"] = cypher
                captured["params"] = params
                return []

        retrieve_mod.entity_timeline(_FakeClient(), "10213338", 42, "Symptom")
        cypher = captured["cypher"]
        validate_dialect(cypher)  # must not raise
        assert "ORDER BY cl.valid_from" in cypher
        assert "(cl:Claim" in cypher and "(e:Symptom" in cypher
        assert captured["params"] == {"pid": "10213338", "eid": 42}

    def test_keys_on_claim_patient_id_not_the_patient_hub(self):
        """Walking (:Patient)-[:HAS]-> is what made algo.MSpaths exceed the
        engine's 30s timeout — :Patient has an edge to every claim. A property
        lookup on the claim sidesteps the hub."""
        captured = {}

        class _FakeClient:
            def run(self, cypher, **params):
                captured["cypher"] = cypher
                return []

        retrieve_mod.entity_timeline(_FakeClient(), "p", 1, "Medication")
        assert ":HAS" not in captured["cypher"]
        assert "patient_id: $pid" in captured["cypher"]

    def test_result_is_capped(self):
        class _FakeClient:
            def run(self, cypher, **params):
                return [{"id": i, "valid_from": f"210{i}-01-01"} for i in range(50)]

        rows = retrieve_mod.entity_timeline(_FakeClient(), "p", 1, "Symptom", limit=5)
        assert len(rows) == 5
