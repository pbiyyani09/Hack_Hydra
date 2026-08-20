"""Tests for the frozen contracts (ARCHITECTURE.md §4): ClinicalFact / retrieve() shapes,
validate(), and the mocks. Written by dev-python alongside the story (no separate test-lead
plan exists in this repo's coordination model — see collaborative/inbox/004-to-claude.md)."""

from dataclasses import replace

from medmemgraph.contracts import (
    PREDICATES,
    SENTINEL_VALID_TO,
    ClinicalFact,
    EntityRef,
    RetrieveItem,
    RetrieveResult,
    mock_fact,
    mock_retrieve,
    to_row,
    validate,
)


# ---------------------------------------------------------------------------
# Sentinel
# ---------------------------------------------------------------------------


def test_sentinel_is_exact_string():
    assert SENTINEL_VALID_TO == "9999-12-31T00:00:00"


def test_mock_fact_defaults_to_sentinel_valid_to():
    fact = mock_fact()
    assert fact.valid_to == "9999-12-31T00:00:00"


# ---------------------------------------------------------------------------
# mock_fact validates clean
# ---------------------------------------------------------------------------


def test_mock_fact_validates_clean():
    fact = mock_fact()
    assert validate(fact) == []


def test_mock_fact_overrides_apply():
    fact = mock_fact(patient_id="patient-9999", polarity="negated")
    assert fact.patient_id == "patient-9999"
    assert fact.polarity == "negated"
    # unrelated fields still valid / untouched
    assert validate(fact) == []


# ---------------------------------------------------------------------------
# validate() — each failure mode fires
# ---------------------------------------------------------------------------


def test_validate_rejects_predicate_outside_closed_vocab():
    fact = replace(mock_fact(), predicate="DOES_SOMETHING_UNLISTED")
    problems = validate(fact)
    assert any("predicate" in p for p in problems)


def test_validate_rejects_null_valid_to():
    fact = mock_fact()
    fact.valid_to = None  # type: ignore[assignment]
    problems = validate(fact)
    assert any("valid_to" in p for p in problems)


def test_validate_rejects_empty_valid_to():
    fact = mock_fact()
    fact.valid_to = ""
    problems = validate(fact)
    assert any("valid_to" in p for p in problems)


def test_validate_rejects_polarity_out_of_range():
    fact = mock_fact()
    fact.polarity = "maybe"  # type: ignore[assignment]
    problems = validate(fact)
    assert any("polarity" in p for p in problems)


def test_validate_rejects_polarity_possible_decision_002_stays_open():
    """Decision 002 (six-class i2b2 assertion) is open. The wire contract stays binary
    asserted|negated — 'possible' must be rejected, not silently accepted as a widened field."""
    fact = mock_fact()
    fact.polarity = "possible"  # type: ignore[assignment]
    problems = validate(fact)
    assert any("polarity" in p for p in problems)


def test_validate_rejects_source_class_out_of_range():
    fact = mock_fact()
    fact.source_class = "nurse"  # type: ignore[assignment]
    problems = validate(fact)
    assert any("source_class" in p for p in problems)


def test_validate_rejects_valid_from_after_valid_to():
    fact = mock_fact(valid_from="2027-01-01T00:00:00", valid_to="2026-01-01T00:00:00")
    problems = validate(fact)
    assert any("valid_from" in p for p in problems)


def test_validate_rejects_empty_turn_ids():
    fact = mock_fact(turn_ids=[])
    problems = validate(fact)
    assert any("turn_ids" in p for p in problems)


def test_validate_rejects_negative_subject_canonical_id():
    fact = mock_fact()
    fact.subject = EntityRef(name="x", type="Medication", canonical_id=-1)
    problems = validate(fact)
    assert any("subject.canonical_id" in p for p in problems)


def test_validate_rejects_negative_object_canonical_id():
    fact = mock_fact()
    fact.object = EntityRef(name="x", type="Medication", canonical_id=-5)
    problems = validate(fact)
    assert any("object.canonical_id" in p for p in problems)


def test_validate_returns_multiple_problems_when_multiple_fields_are_bad():
    fact = mock_fact(turn_ids=[], predicate="NOT_A_PREDICATE")
    fact.polarity = "maybe"  # type: ignore[assignment]
    problems = validate(fact)
    assert len(problems) >= 3


# ---------------------------------------------------------------------------
# Closed predicate vocabulary
# ---------------------------------------------------------------------------


def test_predicates_is_the_seven_from_architecture_section_4_plus_had_incident():
    # HAD_INCIDENT added post-freeze (2026-08-16 PRN/vocabulary-gap bug
    # story) — see contracts.py's own PREDICATES docstring for the real
    # observed-data justification ("had a fall" was undroppable-loss).
    assert PREDICATES == frozenset(
        {
            "TAKES_MEDICATION",
            "HAS_CONDITION",
            "HAS_ALLERGY_TO",
            "REPORTS_SYMPTOM",
            "HAD_PROCEDURE",
            "CURRENT_DOSAGE_OF",
            "PRIMARY_CARE_PROVIDER_OF",
            "HAD_INCIDENT",
        }
    )
    assert len(PREDICATES) == 8


# ---------------------------------------------------------------------------
# to_row() — flat dict of scalars only
# ---------------------------------------------------------------------------


def test_to_row_returns_only_scalars():
    fact = mock_fact()
    row = to_row(fact)
    for key, value in row.items():
        assert isinstance(value, (str, int, float, bool)), (
            f"{key!r} is {type(value)}, not a scalar"
        )


def test_to_row_has_no_nested_dict_or_raw_list_values():
    fact = mock_fact()
    row = to_row(fact)
    for value in row.values():
        assert not isinstance(value, (dict, list))


def test_to_row_flattens_subject_and_object():
    fact = mock_fact()
    row = to_row(fact)
    assert row["subject_name"] == fact.subject.name
    assert row["subject_type"] == fact.subject.type
    assert row["subject_canonical_id"] == fact.subject.canonical_id
    assert row["object_name"] == fact.object.name
    assert row["object_type"] == fact.object.type
    assert row["object_canonical_id"] == fact.object.canonical_id
    assert "subject" not in row
    assert "object" not in row


def test_to_row_preserves_scalar_fields():
    fact = mock_fact()
    row = to_row(fact)
    assert row["fact_id"] == fact.fact_id
    assert row["patient_id"] == fact.patient_id
    assert row["session_id"] == fact.session_id
    assert row["predicate"] == fact.predicate
    assert row["valid_from"] == fact.valid_from
    assert row["valid_to"] == fact.valid_to
    assert row["observed_at"] == fact.observed_at
    assert row["polarity"] == fact.polarity
    assert row["source_class"] == fact.source_class
    assert row["confidence"] == fact.confidence


# ---------------------------------------------------------------------------
# retrieve() mock shape
# ---------------------------------------------------------------------------


def test_mock_retrieve_returns_frozen_shape():
    result = mock_retrieve("does the patient take metformin?", "patient-0001", k=3)
    assert isinstance(result, RetrieveResult)
    assert len(result.items) == 3
    assert all(isinstance(item, RetrieveItem) for item in result.items)
    assert result.route in {"graph", "vector", "hybrid"}
    assert isinstance(result.structural_absence, bool)
    assert isinstance(result.paths, list)
    assert isinstance(result.latency_ms, dict)


def test_mock_retrieve_item_channel_is_legal():
    result = mock_retrieve("q", "patient-0001", k=1)
    assert result.items[0].channel in {"graph", "vector", "lexical"}


def test_mock_retrieve_zero_k_returns_empty_items():
    result = mock_retrieve("q", "patient-0001", k=0)
    assert result.items == []
