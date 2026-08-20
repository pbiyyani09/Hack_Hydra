"""Tests for deterministic integer id minting (ARCHITECTURE.md §5.4). Written by dev-python
alongside the story (no separate test-lead plan exists in this repo's coordination model — see
collaborative/inbox/004-to-claude.md, and tests/test_contracts.py's identical note).
"""

from __future__ import annotations

import inspect
import subprocess

import pytest

from medmemgraph.pipeline.ids import (
    MAX_ID,
    IdMinter,
    collision_self_check,
    fact_id,
    mint,
    mint_claim_id,
    mint_entity_id,
    mint_patient_id,
    normalize_key,
)

# ---------------------------------------------------------------------------
# normalize_key
# ---------------------------------------------------------------------------


def test_normalize_key_lowercases_strips_punctuation_and_collapses_whitespace():
    assert normalize_key("  METFORMIN, 500mg.  ") == "metformin 500mg"


def test_normalize_key_equal_for_case_and_punctuation_variants():
    assert normalize_key("Metformin") == normalize_key("metformin.") == normalize_key("  METFORMIN  ")


# ---------------------------------------------------------------------------
# Determinism — same process, repeated calls
# ---------------------------------------------------------------------------


def test_mint_entity_id_is_deterministic_across_calls():
    a = mint_entity_id("patient-0001", "Medication", "metformin")
    b = mint_entity_id("patient-0001", "Medication", "metformin")
    assert a == b


def test_mint_patient_id_is_deterministic_across_calls():
    a = mint_patient_id("10056223")
    b = mint_patient_id("10056223")
    assert a == b


def test_mint_claim_id_is_deterministic_across_calls():
    a = mint_claim_id("abc123fakehash")
    b = mint_claim_id("abc123fakehash")
    assert a == b


def test_mint_entity_id_normalizes_before_minting():
    """Surface-form variants that normalize to the same key mint to the same id."""
    a = mint_entity_id("patient-0001", "Medication", "Metformin")
    b = mint_entity_id("patient-0001", "Medication", "  METFORMIN  ")
    c = mint_entity_id("patient-0001", "Medication", "metformin.")
    assert a == b == c


# ---------------------------------------------------------------------------
# Determinism — fresh interpreter (separate process, no shared state)
# ---------------------------------------------------------------------------


def _run_in_fresh_interpreter(expr: str) -> str:
    """Spawn `uv run python -c <expr>` and return stdout, stripped. A fresh interpreter has an
    empty _DEFAULT_MINTER, so this actually exercises 'same inputs -> same id across processes
    and runs', not just 'across calls in one process'."""
    result = subprocess.run(
        ["uv", "run", "python", "-c", expr],
        cwd=str(_repo_root()),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"subprocess failed: {result.stderr}"
    return result.stdout.strip()


def _repo_root() -> str:
    import pathlib

    return str(pathlib.Path(__file__).resolve().parents[1])


@pytest.mark.timeout(60)
def test_mint_entity_id_deterministic_across_fresh_interpreters():
    expr = (
        "from medmemgraph.pipeline.ids import mint_entity_id; "
        "print(mint_entity_id('patient-0001', 'Medication', 'metformin'))"
    )
    first = _run_in_fresh_interpreter(expr)
    second = _run_in_fresh_interpreter(expr)
    in_process = mint_entity_id("patient-0001", "Medication", "metformin")
    assert first == second == str(in_process)


@pytest.mark.timeout(60)
def test_mint_patient_id_deterministic_across_fresh_interpreters():
    expr = (
        "from medmemgraph.pipeline.ids import mint_patient_id; "
        "print(mint_patient_id('10056223'))"
    )
    first = _run_in_fresh_interpreter(expr)
    second = _run_in_fresh_interpreter(expr)
    assert first == second == str(mint_patient_id("10056223"))


@pytest.mark.timeout(60)
def test_fact_id_deterministic_across_fresh_interpreters():
    expr = (
        "from medmemgraph.pipeline.ids import fact_id; "
        "print(fact_id(patient_id='patient-0001', session_id='admission-0001', "
        "turn_ids=[3, 1, 2], subject_canonical_id=1, predicate='TAKES_MEDICATION', "
        "object_canonical_id=2, polarity='asserted', valid_from='2101-01-01T00:00:00'))"
    )
    first = _run_in_fresh_interpreter(expr)
    second = _run_in_fresh_interpreter(expr)
    in_process = fact_id(
        patient_id="patient-0001",
        session_id="admission-0001",
        turn_ids=[3, 1, 2],
        subject_canonical_id=1,
        predicate="TAKES_MEDICATION",
        object_canonical_id=2,
        polarity="asserted",
        valid_from="2101-01-01T00:00:00",
    )
    assert first == second == in_process


# ---------------------------------------------------------------------------
# Non-negative, in-range (§5.4: positive 63-bit fold)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["metformin", "insulin", "lisinopril", "aspirin 81mg", "", "a" * 500, "🩺 unicode name"],
)
def test_mint_entity_id_is_non_negative_and_in_63_bit_range(name):
    if name == "":
        pytest.skip("empty canonical_name is a validation error, covered separately")
    minted = mint_entity_id("patient-0001", "Medication", name)
    assert 0 <= minted <= MAX_ID
    assert minted < (1 << 63)


def test_mint_patient_id_is_non_negative_and_in_range():
    minted = mint_patient_id("some-subject-id")
    assert 0 <= minted <= MAX_ID


def test_mint_claim_id_is_non_negative_and_in_range():
    minted = mint_claim_id("some-fact-hash")
    assert 0 <= minted <= MAX_ID


def test_max_id_is_positive_63_bit_ceiling():
    assert MAX_ID == (1 << 63) - 1


# ---------------------------------------------------------------------------
# Different inputs give different ids
# ---------------------------------------------------------------------------


def test_different_entity_names_give_different_ids():
    a = mint_entity_id("patient-0001", "Medication", "metformin")
    b = mint_entity_id("patient-0001", "Medication", "insulin")
    assert a != b


def test_different_entity_types_give_different_ids_for_same_name():
    a = mint_entity_id("patient-0001", "Medication", "chest pain")
    b = mint_entity_id("patient-0001", "Symptom", "chest pain")
    assert a != b


def test_different_patients_give_different_entity_ids_for_same_name():
    """Documented resolution in ids.py: entity identity keys are patient-scoped
    (`<Type>|<patient_id>|<normalized_name>`), not the bare `<Type>|<normalized_name>` that
    ARCHITECTURE.md §5.4's literal bullet shows, because this story's own mint_entity_id
    signature takes patient_id and the sibling ER story (E3-S1 AC2) requires two patients'
    'metformin' to never collapse onto one canonical node."""
    a = mint_entity_id("patient-AAAA", "Medication", "metformin")
    b = mint_entity_id("patient-BBBB", "Medication", "metformin")
    assert a != b


def test_different_patient_ids_give_different_patient_ids():
    a = mint_patient_id("10056223")
    b = mint_patient_id("10213338")
    assert a != b


def test_different_fact_ids_give_different_claim_ids():
    a = mint_claim_id("hash-one")
    b = mint_claim_id("hash-two")
    assert a != b


# ---------------------------------------------------------------------------
# id_map plumbing — idempotent replay across a fresh id_map "loaded" from a
# prior run, and cross-key isolation within one shared map.
# ---------------------------------------------------------------------------


def test_mint_with_explicit_id_map_is_idempotent_replay():
    first_run_map: dict[str, int] = {}
    first = mint_entity_id("patient-0001", "Medication", "metformin", id_map=first_run_map)

    # Simulate a fresh process loading the persisted map from the first run.
    reloaded_map = dict(first_run_map)
    second = mint_entity_id("patient-0001", "Medication", "metformin", id_map=reloaded_map)

    assert first == second
    assert reloaded_map == first_run_map


def test_mint_generic_primitive_matches_documented_shape():
    id_map: dict[str, int] = {}
    a = mint("Patient|10056223", id_map)
    b = mint("Patient|10056223", id_map)
    assert a == b
    assert id_map["Patient|10056223"] == a


def test_id_minter_linear_probes_past_a_forced_collision(monkeypatch):
    """A real SHA-256 collision can't be cheaply constructed, so force one white-box by
    monkeypatching the fold to a constant and confirming the documented probe behavior (§5.4
    point 3: 'On collision with a different key, increment the fold ... Never reuse an id.')."""
    import medmemgraph.pipeline.ids as ids_module

    monkeypatch.setattr(ids_module, "_fold_digest", lambda identity_key: 42)
    minter = IdMinter()

    first = minter.mint("key-a")
    second = minter.mint("key-b")
    third = minter.mint("key-c")

    assert first == 42
    assert second == 43  # probed forward past the occupied slot
    assert third == 44
    assert minter.id_map == {"key-a": 42, "key-b": 43, "key-c": 44}
    # Replay of an already-minted key still returns its original id, not a fresh probe.
    assert minter.mint("key-a") == 42


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_mint_entity_id_rejects_empty_patient_id():
    with pytest.raises(ValueError):
        mint_entity_id("", "Medication", "metformin")


def test_mint_entity_id_rejects_empty_canonical_name():
    with pytest.raises(ValueError):
        mint_entity_id("patient-0001", "Medication", "")


def test_mint_entity_id_rejects_canonical_name_that_normalizes_to_empty():
    with pytest.raises(ValueError):
        mint_entity_id("patient-0001", "Medication", "   ...,,,   ")


def test_mint_patient_id_rejects_empty_subject_id():
    with pytest.raises(ValueError):
        mint_patient_id("")


def test_mint_claim_id_rejects_empty_fact_id_value():
    with pytest.raises(ValueError):
        mint_claim_id("")


def test_fact_id_rejects_empty_turn_ids():
    with pytest.raises(ValueError):
        fact_id(
            patient_id="patient-0001",
            session_id="admission-0001",
            turn_ids=[],
            subject_canonical_id=1,
            predicate="TAKES_MEDICATION",
            object_canonical_id=2,
            polarity="asserted",
            valid_from="2101-01-01T00:00:00",
        )


# ---------------------------------------------------------------------------
# fact_id — stable under re-extraction, changes when meaning changes
# ---------------------------------------------------------------------------


def _base_fact_id_kwargs() -> dict:
    return dict(
        patient_id="patient-0001",
        session_id="admission-0001",
        turn_ids=[1, 2],
        subject_canonical_id=1,
        predicate="TAKES_MEDICATION",
        object_canonical_id=2,
        polarity="asserted",
        valid_from="2101-01-01T00:00:00",
    )


def test_fact_id_stable_under_re_extraction_of_same_content():
    """Two 'extraction runs' over identical turns, differing only in fields that are NOT part
    of the identity tuple (confidence, observed_at aren't even accepted by fact_id — see
    test_fact_id_signature_excludes_confidence_and_observed_at), must hash identically."""
    a = fact_id(**_base_fact_id_kwargs())
    b = fact_id(**_base_fact_id_kwargs())
    assert a == b


def test_fact_id_stable_regardless_of_turn_id_input_order():
    kwargs = _base_fact_id_kwargs()
    kwargs["turn_ids"] = [2, 1]
    a = fact_id(**_base_fact_id_kwargs())
    b = fact_id(**kwargs)
    assert a == b


def test_fact_id_changes_when_predicate_changes():
    a = fact_id(**_base_fact_id_kwargs())
    kwargs = _base_fact_id_kwargs()
    kwargs["predicate"] = "HAS_CONDITION"
    b = fact_id(**kwargs)
    assert a != b


def test_fact_id_changes_when_object_canonical_id_changes():
    a = fact_id(**_base_fact_id_kwargs())
    kwargs = _base_fact_id_kwargs()
    kwargs["object_canonical_id"] = 999
    b = fact_id(**kwargs)
    assert a != b


def test_fact_id_changes_when_polarity_changes():
    """asserted vs negated must not hash the same — a negation must not silently overwrite an
    assertion's claim node."""
    a = fact_id(**_base_fact_id_kwargs())
    kwargs = _base_fact_id_kwargs()
    kwargs["polarity"] = "negated"
    b = fact_id(**kwargs)
    assert a != b


def test_fact_id_changes_when_valid_from_changes():
    a = fact_id(**_base_fact_id_kwargs())
    kwargs = _base_fact_id_kwargs()
    kwargs["valid_from"] = "2102-06-01T00:00:00"
    b = fact_id(**kwargs)
    assert a != b


def test_fact_id_changes_when_turn_ids_change():
    a = fact_id(**_base_fact_id_kwargs())
    kwargs = _base_fact_id_kwargs()
    kwargs["turn_ids"] = [1, 2, 3]
    b = fact_id(**kwargs)
    assert a != b


def test_fact_id_signature_excludes_confidence_and_observed_at():
    """Cements the documented decision in ids.fact_id's docstring: confidence and observed_at
    are not identity fields and must not be able to perturb the hash, because they cannot even
    be passed in."""
    params = set(inspect.signature(fact_id).parameters)
    assert "confidence" not in params
    assert "observed_at" not in params


def test_fact_id_returns_stable_hex_digest_shape():
    value = fact_id(**_base_fact_id_kwargs())
    assert isinstance(value, str)
    assert len(value) == 64  # sha256 hexdigest
    int(value, 16)  # must be valid hex


# ---------------------------------------------------------------------------
# Collision sweep — realistic scale, >=100k synthetic entity names
# ---------------------------------------------------------------------------


def _synthetic_names(n: int) -> list[str]:
    """Realistic-ish synthetic entity names: drug/condition/symptom-shaped strings with varied
    length and structure, not a trivially patterned sequence, so the sweep says something about
    the fold, not about SHA-256 on `str(i)` specifically."""
    stems = [
        "metformin", "insulin", "lisinopril", "atorvastatin", "amlodipine", "losartan",
        "hydrochlorothiazide", "gabapentin", "sertraline", "omeprazole", "levothyroxine",
        "diabetes mellitus type 2", "hypertension", "chronic kidney disease", "chest pain",
        "shortness of breath", "penicillin", "sulfa drugs", "latex", "peanuts",
    ]
    names = []
    for i in range(n):
        stem = stems[i % len(stems)]
        # Vary casing, punctuation, dosage suffix, and a numeric salt so names are distinct.
        variant = i % 5
        if variant == 0:
            names.append(f"{stem} {i}mg")
        elif variant == 1:
            names.append(f"{stem.upper()}-{i}")
        elif variant == 2:
            names.append(f"{stem}, patient note #{i}")
        elif variant == 3:
            names.append(f"  {stem}   {i}  ")
        else:
            names.append(f"{stem}_{i}_variant")
    return names


@pytest.mark.timeout(120)
def test_collision_self_check_over_100k_synthetic_names_reports_zero_collisions():
    names = _synthetic_names(100_000)
    report = collision_self_check(names)

    assert report["n_names"] == 100_000
    assert report["n_unique_keys"] == 100_000, "synthetic generator produced a duplicate name"
    assert report["n_unique_ids"] == report["n_unique_keys"]
    assert report["final_id_collisions"] == [], (
        "IdMinter must never return the same id for two different keys — this would be a bug"
    )
    assert report["raw_fold_collisions"] == [], (
        f"expected 0 raw-fold collisions at n=100_000 in a 2**63 space "
        f"(birthday-bound expectation ~2.7e-4); got "
        f"{len(report['raw_fold_collisions'])}: {report['raw_fold_collisions'][:5]}"
    )
