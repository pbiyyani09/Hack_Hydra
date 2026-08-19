"""Frozen interface contracts — ClinicalFact (pipeline -> graph) and retrieve() (graph -> eval).

Copied field-for-field from `collaborative/design/ARCHITECTURE.md` §4 (itself copied from
`literature/GROK-INTAKE.md`). This module is the one place these shapes live. A change to a
field is a `collaborative/decisions/` file, not a quiet edit here.

Hard constraints this module encodes (see ARCHITECTURE.md and decisions/001, /003):
  - Sentinel `valid_to` is exactly "9999-12-31T00:00:00", never null/None. HydraDB has no
    `IS NULL` (literature/10 §A3), so a missing valid_to cannot be filtered for later.
  - `polarity` stays binary `asserted | negated`. Decision 002 (six-class i2b2 assertion) is
    OPEN — do not widen this field to match it.
  - `predicate` is a closed vocabulary (PREDICATES below). Extend only by adding to that one
    set; extraction that can't be canonicalized onto it should be dropped, not invented.
  - `canonical_id` and every HydraDB node `id` are non-negative integers (§5.4).
"""

from dataclasses import dataclass, field, replace
from typing import Literal

# ---------------------------------------------------------------------------
# Sentinel
# ---------------------------------------------------------------------------

SENTINEL_VALID_TO = "9999-12-31T00:00:00"
"""The only legal 'still open' value for ClinicalFact.valid_to / :Claim.valid_to. Never null."""

# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------

PREDICATES = frozenset(
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
"""Closed predicate vocabulary, ARCHITECTURE.md §4 field notes. Canonicalize onto this set at
emit time; skip a fact rather than invent a predicate outside it.

`HAD_INCIDENT` added post-freeze by the PRN/vocabulary-gap bug story
(2026-08-16), one deliberate, minimal extension driven by real observed
data — not a speculative addition. Real-run eyeball evidence
(`samples/extract-eyeball-10056223.md`, admission `23527958` turn 1: "you
had a fall this morning") showed `predicate_phrase='had a fall'` failing to
canonicalize onto any of the original seven predicates and being dropped as
`dropped_no_predicate` — a real information loss, since falls are
clinically significant (fall risk, injury, care planning). None of the
other six predicates is a semantic fit: not a medication, not a diagnosed
condition, not an ongoing subjective symptom, not a performed procedure.
`HAD_INCIDENT` covers a discrete clinical incident/accidental-injury event
(falls and similar), following this vocabulary's existing `HAD_PROCEDURE`
naming convention. Object type is `Condition` (not a new graph label) —
SNOMED CT codes "history of falls" (161899003) as a finding/situation, the
same coding category `HAS_CONDITION` objects already use in
`graph/schema.py`'s `DOMAIN_ENTITY_LABELS`; reusing it avoids widening
`graph/schema.py`'s frozen §5 label set, which this bug-fix story does not
otherwise touch. `graph/schema.py`'s `FUNCTIONAL_KEYS` and
`tests/test_contracts.py`'s closed-set assertion are updated in lockstep
(see those files) so this addition does not silently drift or break the
suite's own completeness guard in `graph/invalidate.py`."""

VALID_POLARITIES = frozenset({"asserted", "negated"})
"""Binary on the wire. Decision 002 (six-class i2b2 assertion) is open; do not widen this."""

VALID_SOURCE_CLASSES = frozenset({"doctor", "patient"})

# ---------------------------------------------------------------------------
# Entity-type vocabulary (2026-08-17 silent-skip bug fix)
# ---------------------------------------------------------------------------

DOMAIN_ENTITY_TYPES = frozenset(
    {"Condition", "Medication", "Allergy", "Symptom", "Procedure", "Provider", "Dosage"}
)
"""Canonical `EntityRef.type` values for a claim's object. `graph/schema.py`
derives `DOMAIN_ENTITY_LABELS` from this set, so the wire vocabulary and the
graph label vocabulary cannot drift apart.

Lives here, not in `graph/schema.py`, because `EntityRef.type` is decided in
Pipeline (`pipeline/extract.py`) and is consumed by THREE places that never
call `schema.label_for`:

  - `pipeline/ids.py::mint_entity_id` folds the type string into the minted
    node id hash (`<Type>|<patient_id>|<normalized_name>`);
  - `pipeline/resolve.py::block` partitions mentions on `(patient_id, entity_type)`;
  - `graph/writer.py` stores it verbatim as the node's `type` property.

That is why normalization must happen at emit time in `extract.py` and NOT by
loosening `schema.label_for`: a case-insensitive `label_for` would still let
`medication` and `Medication` mint two different ids and land in two different
blocking partitions, silently producing duplicate `:Medication` nodes with the
same `name` — a bug the label layer cannot see."""

ENTITY_TYPE_ALIASES: dict[str, str] = {
    # Canonical forms map to themselves so round-tripping is a no-op.
    **{t.lower(): t for t in DOMAIN_ENTITY_TYPES},
    "patient": "Patient",
    # Surface forms observed from real LLM extraction runs, plus the obvious
    # near-synonyms. `_CANDIDATE_FACT_SCHEMA` now pins `object_type` to an enum,
    # so this table is the belt-and-braces for older checkpoints and the
    # rule-based fallback path, not the primary defence.
    "diagnosis": "Condition",
    "disease": "Condition",
    "finding": "Condition",
    "drug": "Medication",
    "med": "Medication",
    "medicine": "Medication",
    "medications": "Medication",
    "allergen": "Allergy",
    "sign": "Symptom",
    "complaint": "Symptom",
    "symptoms": "Symptom",
    "surgery": "Procedure",
    "operation": "Procedure",
    "test": "Procedure",
    "imaging": "Procedure",
    "doctor": "Provider",
    "physician": "Provider",
    "clinician": "Provider",
    "dose": "Dosage",
}
"""Lowercased free-text entity type -> canonical `DOMAIN_ENTITY_TYPES` member.

Deliberately does NOT map `dosage -> Medication`. A dose string is its own
entity: `"60mg"` and `"80mg"` are trigram-similar, and `resolve._similar` would
merge them into one canonical node, collapsing exactly the dose-change history
that `CURRENT_DOSAGE_OF` (one of only three `FUNCTIONAL_KEYS`, i.e. one of the
three predicates that fire `SUPERSEDES`) exists to record."""


def normalize_entity_type(raw: str | None) -> str | None:
    """Canonicalize a free-text entity type onto `DOMAIN_ENTITY_TYPES`.

    Returns `None` for anything unmappable — the caller decides whether that is
    a drop or an error. Returning `None` rather than a `"Entity"` placeholder is
    deliberate: `"Entity"` is not a graph label, so it would be accepted here and
    then silently skipped much later by `graph/writer.py::_register_entity`,
    which is the exact failure this function exists to prevent.
    """
    if not raw:
        return None
    return ENTITY_TYPE_ALIASES.get(raw.strip().lower())

# ---------------------------------------------------------------------------
# Patient-ish placeholder detection (2026-08-17 subject/object role-reversal
# bug fix)
# ---------------------------------------------------------------------------

_PATIENT_ISH_TOKENS = frozenset(
    {
        "patient", "the patient", "pt", "the pt",
        "he", "she", "him", "her", "himself", "herself",
        "i", "me", "myself", "you", "yourself",
    }
)
"""Generic patient-referring placeholders a role-reversed extraction
candidate can leave behind in an entity-name field instead of a real
clinical entity name (e.g. a drug name). `pipeline/extract.py` emits
`subject_name=<drug>, object_name="patient"` on a reversed candidate
(the model's own system-prompt convention says `subject` is normally the
patient; on a reversal it swaps them) — this is the closed set of tokens
that name says "this is the patient", not a clinical entity. See
`docs/algorithms/extraction-and-temporal-normalization.md` §7."""


def is_patient_ish_token(name: str | None, patient_id: str = "") -> bool:
    """True when `name` is a generic patient-referring placeholder — a
    literal "patient"/"the patient"/pronoun, or (when `patient_id` is
    supplied) the patient's own id string — rather than a real clinical
    entity name.

    Shared by `pipeline/extract.py` (ingest-time reversal detection +
    repair: swap a reversed subject/object pair before constructing the
    `ClinicalFact`) and this module's own `validate()` (the last-line
    contract guard below: a patient-subject predicate's `object.name` must
    never be one of these, even if ingest-time repair somehow missed it).
    One definition, two call sites — see this bug's story text ("Do this
    generally, not as a TAKES_MEDICATION special case")."""
    if not name:
        return False
    normalized = name.strip().strip(".,!?;:").lower()
    if not normalized:
        return False
    if patient_id and normalized == patient_id.strip().lower():
        return True
    return normalized in _PATIENT_ISH_TOKENS

VALID_CHANNELS = frozenset({"graph", "vector", "lexical"})

VALID_ROUTES = frozenset({"graph", "vector", "hybrid"})

Polarity = Literal["asserted", "negated"]
SourceClass = Literal["doctor", "patient"]
Channel = Literal["graph", "vector", "lexical"]
Route = Literal["graph", "vector", "hybrid"]


# ---------------------------------------------------------------------------
# CONTRACT 1 — ClinicalFact. Pipeline emits, Graph consumes.
# ---------------------------------------------------------------------------


@dataclass
class EntityRef:
    """subject{...} / object{...} on a ClinicalFact. One canonical node per entity (survey 04);
    aliases live in a `mentioned_as` list on the graph side, not here."""

    name: str
    type: str
    canonical_id: int


@dataclass
class ClinicalFact:
    """Pipeline -> Graph. Field set is frozen (ARCHITECTURE.md §4). No additions, no renames."""

    fact_id: str
    patient_id: str
    session_id: str
    turn_ids: list[int]
    subject: EntityRef
    predicate: str
    object: EntityRef
    valid_from: str
    # kw_only so this is the only field with a default while field declaration order still
    # matches §4 exactly (dataclasses require non-default fields to precede default fields
    # unless the defaulted field is keyword-only).
    valid_to: str = field(default=SENTINEL_VALID_TO, kw_only=True)
    observed_at: str = ""
    polarity: Polarity = "asserted"
    source_class: SourceClass = "patient"
    confidence: float = 0.0


# ---------------------------------------------------------------------------
# CONTRACT 2 — retrieve(). Graph emits, Eval consumes.
# ---------------------------------------------------------------------------


@dataclass
class RetrieveItem:
    text: str
    session_id: str
    turn_ids: list[int]
    score: float
    channel: Channel


@dataclass
class RetrieveResult:
    items: list[RetrieveItem]
    route: Route
    structural_absence: bool
    paths: list
    latency_ms: dict
    """Per-stage wall-clock, keys: `search`, `total`, `graph`, `vector`,
    `lexical`, `rerank`.

    `rerank` added 2026-08-17 with the optional cross-encoder stage; it is
    always present and is 0.0 when no reranker is configured (the default).
    Reported as its own key rather than folded into `vector`/`lexical` because
    the reranker's cost IS the measurement — the model in that stage is
    targeted at CPU-only deployment, and a rerank latency hidden inside the
    retrieval numbers is exactly the figure that decision needs."""


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate(fact: ClinicalFact) -> list[str]:
    """Human-readable list of contract violations. Empty list means valid."""
    problems: list[str] = []

    if fact.predicate not in PREDICATES:
        problems.append(
            f"predicate {fact.predicate!r} is not in the closed vocabulary PREDICATES"
        )

    if not fact.valid_to:
        problems.append(
            "valid_to must not be null/empty; use SENTINEL_VALID_TO — HydraDB has no IS NULL"
        )

    if fact.polarity not in VALID_POLARITIES:
        problems.append(
            f"polarity {fact.polarity!r} must be one of {sorted(VALID_POLARITIES)} "
            "(decision 002 is open; binary only)"
        )

    if fact.source_class not in VALID_SOURCE_CLASSES:
        problems.append(
            f"source_class {fact.source_class!r} must be one of {sorted(VALID_SOURCE_CLASSES)}"
        )

    if fact.valid_from and fact.valid_to and fact.valid_from > fact.valid_to:
        problems.append(
            f"valid_from ({fact.valid_from!r}) must not be greater than valid_to "
            f"({fact.valid_to!r})"
        )

    if not fact.turn_ids:
        problems.append("turn_ids must not be empty; provenance requires at least one turn")

    if fact.subject.canonical_id < 0:
        problems.append(
            f"subject.canonical_id must be non-negative, got {fact.subject.canonical_id}"
        )

    if fact.object.canonical_id < 0:
        problems.append(
            f"object.canonical_id must be non-negative, got {fact.object.canonical_id}"
        )

    # 2026-08-17 subject/object role-reversal bug fix: a patient-subject
    # predicate's object.name must never be a patient-referring placeholder
    # (that is exactly the corrupted-drug-name defect — a fact that says the
    # patient TAKES_MEDICATION "patient"). `CURRENT_DOSAGE_OF` is the one
    # predicate with a genuinely different subject/object shape (medication
    # / dose, ARCHITECTURE.md §4) and is excluded; every other predicate has
    # the patient as subject by construction (`pipeline/extract.py`
    # `_handle_candidate`), so `fact.subject.name` doubles as the patient_id
    # to compare against. This is a last-line guard: `extract.py`'s own
    # ingest-time repair/drop logic should already have caught this before
    # a ClinicalFact is ever built, but a fact reaching `validate()` this
    # way must never be emitted silently either way.
    if fact.predicate != "CURRENT_DOSAGE_OF" and is_patient_ish_token(
        fact.object.name, fact.subject.name
    ):
        problems.append(
            f"object.name {fact.object.name!r} is a patient-referring placeholder, not a "
            f"real clinical entity, for predicate {fact.predicate!r} — likely an unrepaired "
            "subject/object role reversal from extraction"
        )

    return problems


# ---------------------------------------------------------------------------
# Mocks — so Pipeline and Evidence can build against a fake today.
# ---------------------------------------------------------------------------


def mock_fact(**overrides: object) -> ClinicalFact:
    """A ClinicalFact that validates clean. Pass field names to override, e.g.
    mock_fact(polarity="negated", patient_id="patient-0002")."""
    default = ClinicalFact(
        fact_id="mock-fact-0000000001",
        patient_id="patient-0001",
        session_id="admission-0001",
        turn_ids=[1, 2],
        subject=EntityRef(name="patient-0001", type="Patient", canonical_id=1),
        predicate="TAKES_MEDICATION",
        object=EntityRef(name="metformin", type="Medication", canonical_id=2),
        valid_from="2026-01-01T00:00:00",
        valid_to=SENTINEL_VALID_TO,
        observed_at="2026-01-01T00:00:00",
        polarity="asserted",
        source_class="patient",
        confidence=0.9,
    )
    return replace(default, **overrides)


def mock_retrieve(question: str, patient_id: str, k: int) -> RetrieveResult:
    """A RetrieveResult shaped like a real graph/vector answer, for Evidence to build against
    before Graph's retrieve() lands."""
    items = [
        RetrieveItem(
            text=f"mock evidence #{i} for patient {patient_id}: {question}",
            session_id=f"admission-{i:04d}",
            turn_ids=[i],
            score=round(1.0 - (i / k) * 0.1, 4) if k else 0.0,
            channel="vector",
        )
        for i in range(k)
    ]
    return RetrieveResult(
        items=items,
        route="vector",
        structural_absence=False,
        paths=[],
        latency_ms={"total": 0.0},
    )


# ---------------------------------------------------------------------------
# UNWIND row projection
# ---------------------------------------------------------------------------


def to_row(fact: ClinicalFact) -> dict:
    """Flatten a ClinicalFact into a flat dict of scalars — suitable as one row of the `$rows`
    parameter to an UNWIND MERGE/SET statement (ARCHITECTURE.md §6.5). No nested dicts: subject
    / object are exploded into subject_*/object_* keys; turn_ids (a list) is joined into a
    single scalar string since a bare list is not a scalar and the graph reaches turn text via
    DRAWN_FROM edges, not this property (§5.3)."""
    return {
        "fact_id": fact.fact_id,
        "patient_id": fact.patient_id,
        "session_id": fact.session_id,
        "turn_ids": ",".join(str(t) for t in fact.turn_ids),
        "subject_name": fact.subject.name,
        "subject_type": fact.subject.type,
        "subject_canonical_id": fact.subject.canonical_id,
        "predicate": fact.predicate,
        "object_name": fact.object.name,
        "object_type": fact.object.type,
        "object_canonical_id": fact.object.canonical_id,
        "valid_from": fact.valid_from,
        "valid_to": fact.valid_to,
        "observed_at": fact.observed_at,
        "polarity": fact.polarity,
        "source_class": fact.source_class,
        "confidence": fact.confidence,
    }
