"""tests/test_schema.py — schema.py's frozen vocabulary (ARCHITECTURE.md §5,
stories/E4/E4-S3.md acceptance criteria). No live node required; every
assertion here is pure-Python or a `validate_dialect` static check.
"""

from __future__ import annotations

import pytest

from medmemgraph.contracts import DOMAIN_ENTITY_TYPES as CONTRACTS_DOMAIN_ENTITY_TYPES
from medmemgraph.contracts import SENTINEL_VALID_TO as CONTRACTS_SENTINEL
from medmemgraph.graph import schema
from medmemgraph.hydra_client import validate_dialect


def test_sentinel_reexported_and_matches_contracts() -> None:
    assert schema.SENTINEL_VALID_TO == "9999-12-31T00:00:00"
    assert schema.SENTINEL_VALID_TO is CONTRACTS_SENTINEL  # same object, not a redefinition


def test_current_as_of_cypher_passes_dialect_gate_and_names_the_right_labels() -> None:
    validate_dialect(schema.CURRENT_AS_OF_CYPHER)  # must not raise
    assert ":Patient" in schema.CURRENT_AS_OF_CYPHER
    assert ":Claim" in schema.CURRENT_AS_OF_CYPHER
    assert "$pid" in schema.CURRENT_AS_OF_CYPHER and "$D" in schema.CURRENT_AS_OF_CYPHER


def test_labels_match_frozen_set_exactly() -> None:
    # `Dosage` added 2026-08-17 with the entity-type normalization fix. A dose
    # is its own node, not a `:Medication`: `CURRENT_DOSAGE_OF` is one of only
    # three FUNCTIONAL_KEYS (the predicates that fire SUPERSEDES), so the
    # "furosemide 40mg -> 60mg" chain is the canonical invalidation-by-closing
    # walk. Folding doses into `:Medication` would let `resolve._similar` merge
    # "40mg" and "60mg" — trigram-similar — and erase that chain.
    assert schema.LABELS == frozenset(
        {
            "Patient",
            "Admission",
            "Turn",
            "Claim",
            "Condition",
            "Medication",
            "Allergy",
            "Symptom",
            "Procedure",
            "Provider",
            "Dosage",
        }
    )
    for banned in ("Episode", "HAS_CLAIM", "SAME_AS"):
        assert banned not in schema.LABELS


def test_domain_entity_labels_do_not_drift_from_the_wire_vocabulary() -> None:
    """`schema.DOMAIN_ENTITY_LABELS` is derived from
    `contracts.DOMAIN_ENTITY_TYPES`, not restated. Drift between the two is what
    made `writer.write_facts` silently skip 100% of real facts before the
    2026-08-17 fix — extraction emitted lowercase types, `label_for` accepted
    only the exact-cased labels, and the mismatch was recorded on
    `WriteReport.skipped` rather than raised."""
    assert schema.DOMAIN_ENTITY_LABELS == CONTRACTS_DOMAIN_ENTITY_TYPES
    assert schema.DOMAIN_ENTITY_LABELS <= schema.LABELS
    # The structural/provenance labels are never ABOUT targets.
    assert schema.DOMAIN_ENTITY_LABELS.isdisjoint({"Patient", "Admission", "Turn", "Claim"})


def test_rel_types_match_frozen_set_exactly() -> None:
    assert schema.REL_TYPES == frozenset(
        {"HAS", "ABOUT", "SUPERSEDES", "CONTRADICTS", "DRAWN_FROM", "ADMITTED", "CONTAINS"}
    )
    for banned in ("SAME_AS", "HAS_EPISODE", "FROM_EPISODE", "NEXT_ADMISSION", "COMPARED_TO"):
        assert banned not in schema.REL_TYPES


def test_claim_statuses_exactly_active_and_invalidated() -> None:
    assert schema.CLAIM_STATUSES == frozenset({"active", "invalidated"})
    for banned in ("superseded", "contested"):
        assert banned not in schema.CLAIM_STATUSES


def test_claim_properties_cover_architecture_5_3() -> None:
    expected = {
        "id",
        "fact_id",
        "patient_id",
        "session_id",
        "predicate",
        "polarity",
        "source_class",
        "confidence",
        "valid_from",
        "valid_to",
        "observed_at",
        "invalidated_at",
        "status",
    }
    assert set(schema.CLAIM_PROPERTIES) == expected


def test_functional_keys_split_matches_3_5():
    # HAD_INCIDENT added post-freeze (2026-08-16 PRN/vocabulary-gap bug fix)
    # as set-valued — see contracts.py's PREDICATES docstring and
    # graph/schema.py's FUNCTIONAL_KEYS comment for the provenance.
    functional = {k for k, v in schema.FUNCTIONAL_KEYS.items() if v == "functional"}
    set_valued = {k for k, v in schema.FUNCTIONAL_KEYS.items() if v == "set-valued"}
    assert functional == {
        "PRIMARY_CARE_PROVIDER_OF",
        "CURRENT_DOSAGE_OF",
        "TAKES_MEDICATION",
    }
    assert set_valued == {
        "HAS_CONDITION",
        "HAS_ALLERGY_TO",
        "REPORTS_SYMPTOM",
        "HAD_PROCEDURE",
        "HAD_INCIDENT",
    }


class TestLabelFor:
    def test_domain_entity_types_map_to_themselves(self) -> None:
        for entity_type in (
            "Condition",
            "Medication",
            "Allergy",
            "Symptom",
            "Procedure",
            "Provider",
        ):
            assert schema.label_for(entity_type) == entity_type

    def test_patient_type_maps_to_patient(self) -> None:
        assert schema.label_for("Patient") == "Patient"

    def test_unrecognized_type_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            schema.label_for("NotAnEntityType")

    def test_structural_labels_are_not_valid_entity_types(self) -> None:
        # Claim/Admission/Turn are structural labels, never an EntityRef.type.
        for structural in ("Claim", "Admission", "Turn"):
            with pytest.raises(ValueError):
                schema.label_for(structural)

    def test_injection_like_string_is_rejected_not_passed_through(self) -> None:
        with pytest.raises(ValueError):
            schema.label_for("Patient {id: 1}) DETACH DELETE (n")
