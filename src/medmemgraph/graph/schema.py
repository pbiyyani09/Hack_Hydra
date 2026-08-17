"""schema.py — application schema convention for HydraDB OSS (ARCHITECTURE.md §5,
decisions/003). The engine has no DDL; this module is the one place the
label/relationship-type/property vocabulary lives, so no call site hand-writes
a label or relationship-type string, and no `:Claim` writer invents a property
name that drifts from the frozen shape.

Authority: ARCHITECTURE.md §5 (not the plan's `Episode` / `HAS_ADMISSION`
draft — inbox 004: ARCHITECTURE is the spec). Frozen shape (§5, reified
`:Claim` nodes):

    (:Patient)-[:HAS]->(:Claim)-[:ABOUT]->(:Condition|:Medication|:Allergy|:Symptom|:Procedure|:Provider)
    (:Claim)-[:SUPERSEDES]->(:Claim)
    (:Claim)-[:CONTRADICTS]->(:Claim)
    (:Claim)-[:DRAWN_FROM]->(:Turn)
    (:Patient)-[:ADMITTED]->(:Admission)-[:CONTAINS]->(:Turn)

Do NOT add `SAME_AS`, `HAS_CLAIM`, `HAS_EPISODE`, `FROM_EPISODE`,
`NEXT_ADMISSION`, or `COMPARED_TO` (the last is a cuttable BUILD stretch,
~4 engineer-hours, not required to freeze the schema).
"""

from __future__ import annotations

from medmemgraph.contracts import SENTINEL_VALID_TO

__all__ = [
    "SENTINEL_VALID_TO",
    "LABELS",
    "DOMAIN_ENTITY_LABELS",
    "REL_TYPES",
    "CLAIM_STATUSES",
    "CLAIM_PROPERTIES",
    "FUNCTIONAL_KEYS",
    "CURRENT_AS_OF_CYPHER",
    "label_for",
]

# ---------------------------------------------------------------------------
# Labels — ARCHITECTURE.md §5.1
# ---------------------------------------------------------------------------

LABELS = frozenset(
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
    }
)

# Domain-entity labels — the subset of LABELS that ABOUT edges may target
# (i.e. LABELS minus the structural/provenance labels Patient/Admission/
# Turn/Claim). `EntityRef.type` on a ClinicalFact is expected to name one of
# these, or "Patient" when the subject *is* the patient (§5.2: ABOUT targets
# "the claim's object, and, when the subject is not the patient, the
# subject" — so a subject EntityRef of type "Patient" never gets an ABOUT
# edge of its own; see graph/writer.py).
DOMAIN_ENTITY_LABELS = frozenset(
    {"Condition", "Medication", "Allergy", "Symptom", "Procedure", "Provider"}
)

# ---------------------------------------------------------------------------
# Relationship types — ARCHITECTURE.md §5.2. Every relationship carries its
# own integer `id`; `MERGE` on a relationship is by that `id` only.
# ---------------------------------------------------------------------------

REL_TYPES = frozenset(
    {
        "HAS",  # Patient -> Claim
        "ABOUT",  # Claim -> domain entity
        "SUPERSEDES",  # Claim -> Claim
        "CONTRADICTS",  # Claim -> Claim
        "DRAWN_FROM",  # Claim -> Turn
        "ADMITTED",  # Patient -> Admission
        "CONTAINS",  # Admission -> Turn
    }
)

# ---------------------------------------------------------------------------
# `:Claim` properties — ARCHITECTURE.md §5.3
# ---------------------------------------------------------------------------

CLAIM_STATUSES = frozenset({"active", "invalidated"})
"""`status` is exactly `active | invalidated` (ARCHITECTURE.md). Plan-draft
values `superseded` / `contested` are NOT stored as `status`: a SUPERSEDES
close sets `status="invalidated"`; an unresolved pair stays `status="active"`
on BOTH sides and is marked by bidirectional `CONTRADICTS` edges instead."""

CLAIM_PROPERTIES = (
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
)
"""One source of truth for the `:Claim` property set, in ARCHITECTURE.md
§5.3's order. `invalidated_at` uses `SENTINEL_VALID_TO` while the claim is
not closed — never null (no `IS NULL` in this dialect)."""

# ---------------------------------------------------------------------------
# FUNCTIONAL_KEYS — ARCHITECTURE.md §6.6 invalidation-by-closing, step 2/3:
# which predicates are functional over (patient, object) (a new claim closes
# the old one) vs set-valued (claims coexist unless explicitly retracted).
# Consumed by the invalidation pass (E4-S5); exported here because it is
# vocabulary, not write-path logic.
# ---------------------------------------------------------------------------

FUNCTIONAL_KEYS: dict[str, str] = {
    "PRIMARY_CARE_PROVIDER_OF": "functional",
    "CURRENT_DOSAGE_OF": "functional",
    "TAKES_MEDICATION": "functional",
    "HAS_CONDITION": "set-valued",
    "HAS_ALLERGY_TO": "set-valued",
    "REPORTS_SYMPTOM": "set-valued",
    "HAD_PROCEDURE": "set-valued",
    # HAD_INCIDENT added post-freeze (contracts.py's own docstring has the
    # provenance): set-valued like HAD_PROCEDURE/REPORTS_SYMPTOM — a second
    # fall does not retroactively "close" a first one, both are independent,
    # coexisting incidents over the patient's history.
    "HAD_INCIDENT": "set-valued",
}

# ---------------------------------------------------------------------------
# label_for — the one place an EntityRef.type string becomes a graph label.
# ---------------------------------------------------------------------------

_ENTITY_TYPE_TO_LABEL: dict[str, str] = {"Patient": "Patient"} | {
    label: label for label in DOMAIN_ENTITY_LABELS
}


def label_for(entity_type: str) -> str:
    """Map an `EntityRef.type` string (or a patient's own type) onto its
    graph label. Raises `ValueError` for anything outside the recognized
    set, so a call site can never build an unlabelled or mislabelled node
    pattern by passing through an unvalidated string (decisions/003).
    """
    try:
        return _ENTITY_TYPE_TO_LABEL[entity_type]
    except KeyError:
        raise ValueError(
            f"schema.label_for: {entity_type!r} is not a recognized entity type; "
            f"expected one of {sorted(_ENTITY_TYPE_TO_LABEL)}"
        ) from None


# ---------------------------------------------------------------------------
# CURRENT_AS_OF_CYPHER — ARCHITECTURE.md §5.5, copied verbatim (survey 05
# shape (b), projected legally: survey 05's own `RETURN m, r` is a parse
# error here — no bare node/relationship ever crosses the wire, §8.2).
# Caller always supplies `$pid` and `$D`; never depend on datetime()/now().
# ---------------------------------------------------------------------------

CURRENT_AS_OF_CYPHER = (
    "MATCH (p:Patient {id: $pid})-[:HAS]->(c:Claim) "
    "WHERE c.valid_from <= $D AND c.valid_to > $D "
    "RETURN c.id AS id, c.valid_from AS valid_from, c.valid_to AS valid_to, "
    "c.predicate AS predicate"
)
