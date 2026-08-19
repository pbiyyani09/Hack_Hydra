"""pipeline/extract.py — LLM structured extraction of ClinicalFacts from one
patient's dialogue, one admission at a time.

Design authority: ARCHITECTURE.md §6.1, `literature/14-information-extraction-
negation-temporal.md` §2 (Recommended Extraction Pipeline, Stages 1/2/6) and
Part 3 (assertion/negation), `decisions/002-polarity-field-underspecified.md`
(open — binary polarity stays the wire contract).

Pipeline shape (literature/14 §2, Stage 1+2 folded into one structured-output
call rather than a separate ConText pass):

    admission turns -> LLM structured extract (candidate fact + i2b2 tag +
    raw time expression + confidence) -> handling rules (this module) ->
    normalize.resolve_time / normalize.canonicalize_predicate -> ClinicalFact

**Handling rules (decisions/002, kept binary on purpose):**

| i2b2 tag                    | Action                                          |
|------------------------------|--------------------------------------------------|
| present                      | emit, polarity=asserted                          |
| absent                       | emit, polarity=negated (a denial is information) |
| possible                     | emit, polarity=asserted, confidence discounted    |
| conditional / hypothetical    | **do not emit** — skip beats mis-extracting       |
| not-associated-with-patient  | **do not attach to the patient** — drop           |

The i2b2 tag itself is never written onto the emitted `ClinicalFact` (the
frozen contract has no such field) — it is recorded only in this module's own
side log (`Extractor.assertion_log`) for later QC/EDA, exactly as decisions/002
requires ("a side log of i2b2 assertion tags is allowed... it must not drive
writes or closes").

**Transport: the `medmemgraph.llm` seam, not a direct provider SDK call.**
Every real inference call in this module routes through `llm.complete(...,
model=llm.EXTRACT_MODEL, schema=_EXTRACTION_SCHEMA)` — the single seam
`src/medmemgraph/llm.py` documents ("one place to cache, meter, cap spend,
and route, instead of three copies of the same plumbing drifting apart").
This module used to construct its own `anthropic.Anthropic()` client
directly; that import has been removed (`anthropic` is no longer a project
dependency — see `pyproject.toml`) precisely because a direct-SDK call site
duplicates key resolution, retry, schema-validation, caching, and
budget-capping logic that now lives in exactly one place. `_EXTRACTION_SCHEMA`
is a **plain** JSON Schema dict (no `additionalProperties`/`strict` flags
hand-added here) — the seam normalizes it per-provider itself (OpenAI strict
mode vs Google's restricted `Schema` subset); adding provider-specific flags
at this call site would be exactly the footgun the seam exists to prevent.

**Structured output, not "respond in JSON."** `llm.complete(schema=...)` uses
each provider's native structured-output mode rather than a bare "respond
only with JSON" instruction — schema-guided / tool-calling-style extraction
is the literature's own recommendation over prompted JSON mode both narrowly
(`literature/14` R-IE-34: TANL/SPAN-style structured generation beats BIO/
free-text for LLM-based clinical NER) and generally (`literature/17`
R-GSJ-001/R-GSJ-002: naive JSON-mode measurably degrades output quality on
non-classification tasks — up to a 63.1-point accuracy drop on one
Claude-3-Haiku benchmark — while schema-constrained decoding does not carry
that penalty) and matches this project's existing `eval/judge.py` precedent.

**No `dry_run` -> real inference; missing key -> raise, never silently
fall back.** The deterministic rule-based path below (a small NegEx/
ConText-style trigger-window matcher, literature/14 R-IE-01/R-IE-02 design
shape, over a modest clinical lexicon) still exists — it makes this module
free and offline to test — but it is now explicit, opt-in behaviour, not an
accidental default: `Extractor(dry_run=True)` (or `$MEDMEMGRAPH_
EXTRACT_DRY_RUN=1`) runs it, with zero network calls. `Extractor()`'s default
(`dry_run=False`) always attempts a real `llm.complete()` call; if no
OpenAI/Google key is configured anywhere `llm.py` looks, `llm.complete()`
itself raises `llm.MissingAPIKeyError` — a typed, actionable error naming
exactly which env vars / `.env` keys were checked — which this module lets
propagate unwrapped rather than catching it and quietly degrading to the
rule-based path. That silent-degrade behaviour (present in this module's
prior Anthropic-direct implementation) is the exact bug this rewiring
fixes: every fact this module emitted before today came from the rule-based
floor, not a real model, with nothing in the code or its output ever having
said so at the point of failure.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from typing import Callable

from medmemgraph import llm
from medmemgraph.contracts import (
    DOMAIN_ENTITY_TYPES,
    PREDICATES,
    SENTINEL_VALID_TO,
    ClinicalFact,
    EntityRef,
    is_patient_ish_token,
    normalize_entity_type,
    validate,
)
from medmemgraph.pipeline.ids import normalize_key
from medmemgraph.pipeline.loader import Admission, Conversation, Turn
from medmemgraph.pipeline.normalize import PREDICATE_DEFINITIONS, canonicalize_predicate, resolve_time

__all__ = [
    "ExtractionError",
    "AssertionLogEntry",
    "Extractor",
    "extract_facts",
]

_DRY_RUN_ENV_VAR = "MEDMEMGRAPH_EXTRACT_DRY_RUN"
"""Documented env-flag escape hatch for the fallback rule ("the stub runs
only when a caller passes dry_run=True, or sets the documented env flag") —
same `os.environ.get(NAME) == "1"` convention `eval/reader.py`'s
`MEDMEMGRAPH_USE_REAL_RETRIEVE` flag already uses in this codebase, not a
new pattern invented here."""


def _default_dry_run() -> bool:
    return os.environ.get(_DRY_RUN_ENV_VAR) == "1"


_I2B2_TAGS = (
    "present",
    "absent",
    "possible",
    "conditional",
    "hypothetical",
    "not-associated-with-patient",
)

_KNOWN_DECISIONS = (
    "emitted",
    "dropped_by_rule",
    "dropped_no_predicate",
    "dropped_invalid",
    "dropped_role_reversal_unrepairable",
    "dropped_unmappable_entity_type",
    "dropped_unusable_entity_name",
)
"""Every `AssertionLogEntry.decision` value this module ever writes —
`Extractor.decision_counts()`'s zero-fill keys, kept as one source of truth
next to the literal strings `_handle_candidate` uses below.

`dropped_role_reversal_unrepairable` (2026-08-17 subject/object
role-reversal bug fix): a candidate where BOTH subject_name and object_name
are patient-referring placeholders (or the would-be repair source is blank)
— no real clinical entity name survives on either side, so there is nothing
to swap into place. Distinct from `dropped_invalid` (a `validate()`
contract failure on an otherwise well-formed candidate) because this is
caught, and dropped, before a `ClinicalFact` is even constructed — see
`_handle_candidate`."""

_POSSIBLE_CONFIDENCE_DISCOUNT = 0.5
"""literature/14 §2 Stage 2: possible/uncertain -> asserted with confidence
discounted to ~0.4-0.6, never silently promoted to full assertion, never
miscoded as negation (R-IE-04's documented failure directions)."""

_PATIENT_SOURCED_CONDITION_DISCOUNT = 0.9
"""literature/14 §2 Stage 6, this survey's own design inference (explicitly
flagged there as needing day-1 validation, not literature-sourced): a
patient's own unconfirmed self-report of a formal diagnosis is plausibly
less authoritative than a clinician's direct assertion of the same fact.
Applied only to HAS_CONDITION facts whose source_class is "patient"."""

# NOTE: this module previously ran its own MAX_RETRIES=1 corrective-feedback
# retry loop on top of the raw Anthropic call (re-prompting with "your
# previous response did not match the schema" on a parse/shape failure).
# That is now `llm.complete()`'s job (schema-constrained decoding + up to
# 2 extra schema-mismatch round trips + up to DEFAULT_MAX_RETRIES transient-
# failure retries, all in the one seam) — duplicating a second retry loop
# here would be exactly the "three copies of the same plumbing drifting
# apart" llm.py's own module docstring warns against. See
# `Extractor._call_llm_extract` below.


class ExtractionError(RuntimeError):
    """Raised when the LLM is down, returns unparseable output, or the
    structured response fails schema validation after retry. Deliberately
    loud: the caller gets zero facts, never a silent degrade to raw-turn-only
    graph facts (E2-S1 banned approaches)."""


@dataclass
class AssertionLogEntry:
    """One row of the i2b2-tag side log (decisions/002: allowed, must not
    drive writes/closes, must not appear on the frozen `ClinicalFact`).

    `prn` (2026-08-16 PRN-drop bug fix): carries the as-needed/PRN nature of
    a `TAKES_MEDICATION` candidate as a **side-log fact attribute**, exactly
    as the story text asks ("carry the PRN nature as a fact attribute...
    rather than discarding the fact") — deliberately on this dataclass and
    NOT on `contracts.ClinicalFact`, which stays frozen (decisions/002: "do
    not widen the wire contract"). `False` for every non-PRN candidate and
    for every non-medication predicate.

    `repaired_role_reversal` (2026-08-17 subject/object role-reversal bug
    fix): True when `_handle_candidate` detected `object_name` holding a
    patient-referring placeholder (e.g. literal "patient") while
    `subject_name` held a real clinical entity name — the model returned
    the roles reversed relative to its own system-prompt convention — and
    repaired it by swapping the real entity name into the object position
    before this entry's `object_name`/emitted fact were finalized. A silent
    repair is only marginally better than a silent corruption, so this
    field exists to make the repair rate countable
    (`Extractor.decision_counts()['repaired_role_reversal']`), not just the
    fact that it happened once."""

    patient_id: str
    session_id: str
    turn_ids: list[int]
    i2b2_tag: str
    subject_name: str
    predicate_phrase: str
    object_name: str
    raw_time_expression: str
    evidence_quote: str
    decision: str  # "emitted" | "dropped_by_rule" | "dropped_no_predicate" |
    # "dropped_invalid" | "dropped_role_reversal_unrepairable"
    reason: str
    emitted_fact_id: str | None = None
    prn: bool = False
    repaired_role_reversal: bool = False


# ---------------------------------------------------------------------------
# Structured-output schema (a plain JSON Schema dict, passed to
# llm.complete(schema=...) — the seam normalizes it per-provider, see module
# docstring) and system prompt for the LLM path. Vocabulary text is
# generated from normalize.PREDICATE_DEFINITIONS so the prompt and the
# canonicalize pass never drift apart (single source of truth).
# ---------------------------------------------------------------------------

_CANDIDATE_FACT_SCHEMA = {
    "type": "object",
    "properties": {
        "subject_name": {
            "type": "string",
            "description": (
                "The PATIENT, almost always — write the literal word 'patient' "
                "here for every predicate except CURRENT_DOSAGE_OF (see the "
                "system prompt's Subject/object convention section). Never "
                "put a medication/condition/symptom/procedure name here "
                "unless predicate_phrase is a dosage relation — that is the "
                "one reversed-roles case this field exists to prevent."
            ),
        },
        "subject_type": {"type": "string"},
        "predicate_phrase": {
            "type": "string",
            "description": "An open phrase naming the relation; a later pass canonicalizes it onto the closed vocabulary or drops the fact.",
        },
        "object_name": {
            "type": "string",
            "description": (
                "The real clinical entity this fact is about — the "
                "medication/condition/symptom/procedure/allergen NAME (e.g. "
                "'prochlorperazine', 'diabetes'), never the word 'patient', "
                "'the patient', 'he'/'she'/'him'/'her', or any other word "
                "that just refers back to the patient. If you find yourself "
                "about to write a patient-referring word here, you have the "
                "subject/object roles backwards — put the entity name here "
                "and 'patient' in subject_name instead. See the worked "
                "example in the system prompt."
            ),
        },
        "object_type": {
            # Pinned to the closed vocabulary (2026-08-17). Previously a free
            # `{"type": "string"}`, which is why real runs emitted lowercase
            # `symptom`/`medication`/`dosage` and every fact was later skipped
            # by the graph writer. `_handle_candidate` still normalizes what
            # comes back — this enum stops the drift at the source, the alias
            # table in `contracts.py` catches whatever slips past it.
            "type": "string",
            "enum": sorted(DOMAIN_ENTITY_TYPES),
            "description": (
                "The kind of clinical entity named in object_name. "
                "'Dosage' is for a dose amount ('60mg') and pairs with the "
                "CURRENT_DOSAGE_OF predicate; the medication itself goes in "
                "subject_name for that predicate."
            ),
        },
        "assertion": {"type": "string", "enum": list(_I2B2_TAGS)},
        "prn": {
            "type": "boolean",
            "description": (
                "True only for a standing 'as needed' / PRN medication order "
                "(assertion='present') — the prescription is active now and "
                "the PATIENT decides when to take a dose within any stated "
                "limits, e.g. 'oxycodone for pain as needed', 'prochlorperazine "
                "for nausea if it comes back'. False for every regularly "
                "scheduled medication and every non-medication fact. Do not "
                "use this to mark a genuinely future-contingent prescription "
                "that does not exist yet (that is assertion='conditional', "
                "prn=false, and the fact is dropped)."
            ),
        },
        "time_expression": {
            "type": "string",
            "description": "The verbatim source-text span expressing when this holds (e.g. 'two weeks ago', 'since my last visit'), or '' if none is stated. Do not resolve it yourself.",
        },
        "turn_ids": {"type": "array", "items": {"type": "integer"}},
        "confidence": {"type": "number"},
        "evidence_quote": {
            "type": "string",
            "description": "A short verbatim quote from the turn text that supports this fact (Chain-of-Note style grounding).",
        },
    },
    "required": [
        "subject_name",
        "subject_type",
        "predicate_phrase",
        "object_name",
        "object_type",
        "assertion",
        "prn",
        "time_expression",
        "turn_ids",
        "confidence",
        "evidence_quote",
    ],
    "additionalProperties": False,
}

_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "facts": {"type": "array", "items": _CANDIDATE_FACT_SCHEMA},
    },
    "required": ["facts"],
    "additionalProperties": False,
}

_MAX_EXTRACT_TOKENS = 4096


def _predicate_vocab_block() -> str:
    lines = [f"- {name}: {defn}" for name, defn in PREDICATE_DEFINITIONS.items()]
    return "\n".join(lines)


_SYSTEM_PROMPT = (
    "You extract structured clinical facts from one hospital admission's "
    "doctor-patient dialogue turns. Emit one candidate fact per distinct "
    "clinical assertion.\n\n"
    "Subject/object convention: `subject` is normally the PATIENT unless "
    "the predicate is CURRENT_DOSAGE_OF, in which case `subject` is the "
    "medication and `object` is the dose (e.g. subject='furosemide', "
    "object='60mg'). Never attach a family member's condition to the "
    "patient as HAS_CONDITION — use the assertion tag "
    "'not-associated-with-patient' instead (see below) and let the pipeline "
    "drop it.\n\n"
    "Do NOT reverse subject and object. `object_name` must be the real "
    "clinical entity's name — the medication, condition, symptom, "
    "procedure, or allergen actually being talked about. It must NEVER be "
    "the word 'patient', 'the patient', or a pronoun ('he'/'she'/'him'/"
    "'her') standing in for the patient — that word belongs in "
    "`subject_name`, not `object_name`, for every predicate except "
    "CURRENT_DOSAGE_OF.\n"
    "  CORRECT: \"Prochlorperazine for nausea if it comes back.\" -> "
    "subject_name='patient', predicate_phrase='is prescribed', "
    "object_name='prochlorperazine'.\n"
    "  WRONG (roles reversed — never do this): subject_name="
    "'prochlorperazine', object_name='patient'. If you notice the entity "
    "name is about to land in subject_name instead of object_name, you have "
    "the roles backwards; swap them before responding.\n\n"
    "Predicate vocabulary (propose a phrase close to one of these; an exact "
    "hit is not required, a later pass canonicalizes or drops the fact):\n"
    f"{_predicate_vocab_block()}\n\n"
    "For every candidate fact, classify its assertion status using EXACTLY "
    "one of these six i2b2-style tags:\n"
    "- present: the concept currently holds for the patient. This INCLUDES "
    "a standing 'as needed' / PRN medication order — a PRN order is a REAL, "
    "currently-active prescription; the patient deciding WHEN to take a "
    "given dose does not make the prescription itself hypothetical or "
    "future-contingent. It also includes a doctor plainly stating that a "
    "medication/treatment IS being given/started/continued as part of "
    "today's actual care plan ('we'll give you X', 'we're starting you on "
    "X') — that is a real, present order, not a hypothetical, even though "
    "it is phrased with future-tense 'will'.\n"
    "- absent: explicitly denied/negated for the patient (e.g. 'no known "
    "drug allergies', 'not taking the mesalamine').\n"
    "- possible: hedged/uncertain (e.g. 'could be musculoskeletal', 'rule "
    "out sepsis').\n"
    "- conditional: the fact does NOT hold yet and will only come into "
    "existence if a stated future event happens (e.g. 'if the fever "
    "returns, we'll RESTART the antibiotic' — the antibiotic is stopped "
    "now, and only resumes if the fever comes back). Do NOT use "
    "conditional for PRN/as-needed dosing language — see 'present' above "
    "and the worked examples below; the distinguishing test is whether the "
    "'if'/'as needed' clause describes (a) WHEN to take an "
    "already-active medication for its stated indication (-> present, "
    "prn=true) or (b) a not-yet-started action gated on a future clinical "
    "change (-> conditional).\n"
    "- hypothetical: a counterfactual, not the patient's actual state (e.g. "
    "'if this were bacterial we would...'). A plain statement of today's "
    "actual treatment plan is never hypothetical, even in future tense — "
    "see 'present' above.\n"
    "- not-associated-with-patient: about someone else (e.g. 'my mother has "
    "diabetes', 'my other doctor said...' — attribute reported clinician "
    "speech to the reporting speaker, not this tag; this tag is only for a "
    "condition/fact belonging to a person OTHER than the patient).\n\n"
    "Worked examples contrasting present/PRN vs. genuinely conditional vs. "
    "hypothetical (study the contrast, do not pattern-match on the word "
    "'if' alone):\n"
    "- 'Take ibuprofen as needed for pain.' -> present, prn=true (a "
    "standing PRN order).\n"
    "- 'We're giving you oxycodone as needed. Take one every four hours if "
    "needed.' -> present, prn=true (the 'if needed' names the "
    "patient-discretionary dosing trigger, not a contingency on the "
    "prescription's existence).\n"
    "- 'Prochlorperazine for nausea if it comes back, and oxycodone for "
    "pain as needed.' -> BOTH present, prn=true (each medication is FOR a "
    "named symptom, taken if/when that symptom is present — this is PRN "
    "phrasing, not a not-yet-prescribed contingency).\n"
    "- 'We'll give you Tessalon Pearls to help with the cough.' -> present, "
    "prn=false (a plain active prescription being given today; there is no "
    "contingency at all, so this is not conditional and not hypothetical).\n"
    "- 'If the fever returns, restart the antibiotic.' -> conditional (the "
    "antibiotic is not currently being taken; a genuinely not-yet-true "
    "future action, gated on a status change, not a dosing trigger for an "
    "active order).\n"
    "- 'If this were bacterial, we would start antibiotics.' -> "
    "hypothetical (a counterfactual scenario the doctor is not asserting "
    "is the patient's actual state).\n\n"
    "prn (boolean, required on every candidate): true only when assertion="
    "'present' AND the fact is a standing as-needed/PRN medication order "
    "(see above); false for every regularly-scheduled medication and every "
    "non-medication fact.\n\n"
    "time_expression must be the verbatim source phrase for when the fact "
    "holds ('' if none stated) — do not resolve dates yourself; the "
    "pipeline resolves relative time against the turn's own timestamp. "
    "Corpus timestamps are shifted into the 2100s+ (MIMIC de-identification) "
    "— never assume the present era or flag a future-looking date as an "
    "anomaly.\n\n"
    "evidence_quote must be a short verbatim quote from the turn text "
    "supporting the fact (ground every extraction in the source text). "
    "turn_ids must be non-empty and reference the input turn_number(s) that "
    "support the fact. confidence is your own calibrated estimate in [0,1] "
    "for the assertion=present/absent case (before any pipeline discount).\n\n"
    "Respond with the required JSON only — one object per fact in `facts`. "
    "It is correct to return an empty `facts` list for a turn window with "
    "no extractable clinical content."
)


def _format_turns_block(turns: list[Turn]) -> str:
    return "\n".join(
        f"[turn {t.turn_number} | {t.time} | {t.speaker}] {t.text}" for t in turns
    )


# ---------------------------------------------------------------------------
# Placeholder fact_id (ARCHITECTURE §5.4 point 2 — real mint lands in E4-S3 /
# E3-S1; this story's placeholder is deterministic so re-running extraction
# on the same input is idempotent, but is explicitly NOT the frozen mint).
# ---------------------------------------------------------------------------


def _placeholder_fact_id(
    *,
    patient_id: str,
    session_id: str,
    subject_name: str,
    predicate: str,
    object_name: str,
    valid_from: str,
    polarity: str,
) -> str:
    key = "|".join(
        [patient_id, session_id, subject_name, predicate, object_name, valid_from, polarity]
    )
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]
    return f"pending-{digest}"


# ---------------------------------------------------------------------------
# Rule-based fallback lexicons — deliberately modest, NegEx/ConText-shaped
# (trigger word + local window), used only when Extractor.dry_run is True.
# ---------------------------------------------------------------------------

_MEDICATIONS = frozenset(
    {
        "metformin", "furosemide", "spironolactone", "escitalopram", "mesalamine",
        "lactulose", "rifaximin", "ciprofloxacin", "insulin", "lisinopril",
        "atorvastatin", "aspirin", "warfarin", "penicillin", "amoxicillin",
        "prednisone", "albuterol", "omeprazole", "metoprolol", "losartan",
        "sertraline", "gabapentin", "hydrochlorothiazide",
    }
)

_CONDITIONS = frozenset(
    {
        "diabetes", "cirrhosis", "ascites", "pneumonia", "hypertension",
        "hepatocellular carcinoma", "ulcerative colitis", "anxiety", "depression",
        "copd", "asthma", "kidney disease", "heart failure", "sepsis",
        "cancer", "pancreatitis", "arrhythmia",
    }
)

_SYMPTOMS = frozenset(
    {
        "swelling", "shortness of breath", "chest pain", "nausea", "fever",
        "cough", "dry cough", "fatigue", "dizziness", "headache", "vomiting",
        "diarrhea", "belly pain", "abdominal pain",
    }
)

_PROCEDURES = frozenset(
    {
        "biopsy", "ablation", "surgery", "transplant", "ultrasound", "echo",
        "ekg", "x-ray", "mri", "ct scan", "colonoscopy", "endoscopy",
    }
)

_ALLERGENS = frozenset(
    {"penicillin", "sulfa", "latex", "shellfish", "peanuts", "iodine"}
)

_LEXICON_TABLE: tuple[tuple[frozenset[str], str, str], ...] = (
    (_MEDICATIONS, "TAKES_MEDICATION", "Medication"),
    (_CONDITIONS, "HAS_CONDITION", "Condition"),
    (_SYMPTOMS, "REPORTS_SYMPTOM", "Symptom"),
    (_PROCEDURES, "HAD_PROCEDURE", "Procedure"),
    (_ALLERGENS, "HAS_ALLERGY_TO", "Allergen"),
)

_NEGATION_RE = re.compile(
    r"\b(?:not\s+(?:actually\s+)?(?:taking|on)|denies|denied|no\s+(?:known\s+)?|"
    r"never\s+(?:had|took)|isn'?t|is\s+not|negative\s+for|no\s+new)\b",
    re.IGNORECASE,
)
_POSSIBLE_RE = re.compile(
    r"\b(?:could\s+be|possibly|might\s+be|rule\s+out|suspect(?:ed)?|likely)\b",
    re.IGNORECASE,
)
_FAMILY_RE = re.compile(
    r"\b(?:mother|father|sister|brother|parent|grandmother|grandfather|"
    r"aunt|uncle|family\s+history)\b",
    re.IGNORECASE,
)
_CONDITIONAL_WINDOW_RE = re.compile(r"\bif\b", re.IGNORECASE)
_DOSAGE_RE = re.compile(r"(\d+(?:\.\d+)?)\s?mg\b", re.IGNORECASE)

_TIME_PHRASE_RE = re.compile(
    r"\b(?:\d+|a\s+couple(?:\s+of)?|a\s+few|several|one|two|three|four|five|"
    r"six|seven|eight|nine|ten)\s+(?:day|week|month|year)s?\s+ago\b"
    r"|\byesterday\b|\btoday\b"
    r"|\bsince\s+(?:my|our|the)?\s*last\s+visit\b"
    r"|\blast\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|week|month|year)\b"
    r"|\ba\s+while\s+(?:back|ago)\b|\bsome\s+time\s+ago\b",
    re.IGNORECASE,
)

_WINDOW_CHARS = 60


def _rule_based_candidates(turn: Turn) -> tuple[list[dict], list[dict]]:
    """NegEx/ConText-shaped trigger-window scan of one turn's text.

    Returns (candidates, skipped) — `candidates` are dicts shaped like the
    LLM schema's `facts` items (minus turn_ids, filled by the caller);
    `skipped` are dicts describing spans deliberately not emitted, for the
    same side-log / eyeball purpose the LLM path's drops serve.
    """
    text = turn.text
    lower = text.lower()
    candidates: list[dict] = []
    skipped: list[dict] = []

    for lexicon, predicate, otype in _LEXICON_TABLE:
        for term in lexicon:
            idx = lower.find(term)
            if idx == -1:
                continue
            window_start = max(0, idx - _WINDOW_CHARS)
            window = lower[window_start : idx + len(term) + 10]

            if _FAMILY_RE.search(window):
                skipped.append(
                    {
                        "reason": "not-associated-with-patient (rule-based: family-relation "
                        "term in window)",
                        "i2b2_tag": "not-associated-with-patient",
                        "span": term,
                        "predicate_phrase": predicate,
                        "object_name": term,
                    }
                )
                continue
            if _CONDITIONAL_WINDOW_RE.search(window):
                skipped.append(
                    {
                        "reason": "conditional/hypothetical (rule-based: 'if' in window)",
                        "i2b2_tag": "conditional",
                        "span": term,
                        "predicate_phrase": predicate,
                        "object_name": term,
                    }
                )
                continue

            assertion = "present"
            confidence = 0.55  # rule-based stub ceiling — deliberately modest,
            # see module docstring: this is a NegEx/ConText-shaped floor, not
            # the LLM path's expected quality.
            if _NEGATION_RE.search(window):
                assertion = "absent"
            elif _POSSIBLE_RE.search(window):
                assertion = "possible"

            time_match = _TIME_PHRASE_RE.search(lower)
            candidates.append(
                {
                    "subject_name": "PATIENT",
                    "subject_type": "Patient",
                    "predicate_phrase": predicate,
                    "object_name": term,
                    "object_type": otype,
                    "assertion": assertion,
                    "time_expression": time_match.group(0) if time_match else "",
                    "confidence": confidence,
                    "evidence_quote": text.strip()[:200],
                }
            )

    for m in _DOSAGE_RE.finditer(text):
        meds_in_turn = [med for med in _MEDICATIONS if med in lower]
        if len(meds_in_turn) != 1:
            continue  # ambiguous which medication the dose belongs to — skip
        candidates.append(
            {
                "subject_name": meds_in_turn[0],
                "subject_type": "Medication",
                "predicate_phrase": "CURRENT_DOSAGE_OF",
                "object_name": f"{m.group(1)}mg",
                "object_type": "Dosage",
                "assertion": "present",
                "time_expression": "",
                "confidence": 0.6,
                "evidence_quote": text.strip()[:200],
            }
        )

    return candidates, skipped


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------


@dataclass
class Extractor:
    """LLM structured extractor (via the `medmemgraph.llm` seam) with an
    explicit, non-silent rule-based fallback. Unlike the pre-rewire version,
    the LLM-vs-rule-based choice is `dry_run`, not key-presence — `dry_run`
    is decided once at construction (mirrors `eval/baselines/*.py`'s
    `dry_run` pattern), never per-call. `dry_run=False` (the default) always
    attempts a real call and lets `llm.py`'s own typed errors (most notably
    `MissingAPIKeyError` when no provider key is configured) propagate —
    see module docstring's "No dry_run -> real inference" section."""

    model: str = field(default_factory=lambda: llm.EXTRACT_MODEL)
    dry_run: bool = field(default_factory=_default_dry_run)
    checkpoint_path: str | os.PathLike[str] | None = None
    max_tokens: int = _MAX_EXTRACT_TOKENS
    use_cache: bool = True
    complete_fn: Callable[..., llm.LLMResponse] = llm.complete
    """The test/DI seam, replacing the old `client=` (Anthropic client)
    injection point now that the transport is a module-level function, not
    a client object: `(prompt, *, model, schema, system, max_tokens,
    temperature, dry_run, use_cache) -> llm.LLMResponse`, i.e. exactly
    `llm.complete`'s own signature. Defaults to the real `llm.complete`;
    tests inject a fake to run the LLM code path (`dry_run=False`) without
    any network access — see `tests/test_extract.py`'s `FakeComplete`."""

    assertion_log: list[AssertionLogEntry] = field(default_factory=list, init=False)
    total_cost_usd: float = field(default=0.0, init=False)
    total_prompt_tokens: int = field(default=0, init=False)
    total_completion_tokens: int = field(default=0, init=False)
    """Running totals across every real `complete_fn` call this Extractor
    instance has made (cache hits and dry_run add $0 / 0 tokens) — surfaced
    so a caller (the bulk-extraction script) can report real spend rather
    than reading it back out of `llm.get_ledger()` itself, per the story's
    "surface cost_usd and token counts... so the caller can report real
    spend" instruction."""
    _checkpoint: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._checkpoint = self._load_checkpoint()

    @property
    def kind(self) -> str:
        return "rule-based" if self.dry_run else "llm"

    # -- checkpointing ----------------------------------------------------

    def _load_checkpoint(self) -> dict:
        if self.checkpoint_path is None:
            return {}
        try:
            with open(self.checkpoint_path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save_checkpoint(self) -> None:
        if self.checkpoint_path is None:
            return
        with open(self.checkpoint_path, "w", encoding="utf-8") as fh:
            json.dump(self._checkpoint, fh)

    # -- extraction stats (2026-08-16 dropped_no_predicate-visibility fix) -

    def decision_counts(self) -> dict[str, int]:
        """Tally of `assertion_log` entries by `decision`, zero-filled for
        every known decision kind so a real-but-currently-zero count (e.g.
        no predicate-vocabulary misses in this run) is still structurally
        present rather than merely absent from a table.

        Exists specifically so `dropped_no_predicate` — previously visible
        only if a caller happened to filter/print `assertion_log` itself —
        is a first-class, always-reported number: "a vocabulary gap surfaces
        instead of quietly eating facts" (the bug this fixes). Callers
        (e.g. `scripts/generate_extract_eyeball.py`'s stats block) should
        prefer this over re-deriving the tally by hand.

        `repaired_role_reversal` (2026-08-17 subject/object role-reversal
        bug fix) is an **additional, non-exclusive** counter, not a
        `decision` partition member: it counts `entry.repaired_role_reversal
        is True` regardless of `entry.decision` (in practice always a subset
        of `emitted`, since an unrepairable reversal drops under
        `dropped_role_reversal_unrepairable` instead — see
        `_handle_candidate`). Unlike every other key here, it double-counts
        against its own `decision`, so `sum(counts.values()) ==
        len(assertion_log)` no longer holds once any repair has happened —
        deliberate: the point of this counter is exactly to make the repair
        *rate* visible (a silent repair is only marginally better than a
        silent corruption), not to preserve the old partition invariant.
        """
        counts = dict.fromkeys(_KNOWN_DECISIONS, 0)
        counts["repaired_role_reversal"] = 0
        for entry in self.assertion_log:
            counts[entry.decision] = counts.get(entry.decision, 0) + 1
            if entry.repaired_role_reversal:
                counts["repaired_role_reversal"] += 1
        return counts

    # -- public API ---------------------------------------------------------

    def extract(
        self, conversation: Conversation, admission: Admission | None = None
    ) -> list[ClinicalFact]:
        """dialogue -> list[ClinicalFact], one admission at a time
        (literature/14 §2: one LLM call per admission)."""
        if admission is not None:
            admissions = [admission]
            prior_end_lookup = self._prior_admission_ends(conversation, only=admission)
        else:
            admissions = list(conversation.admissions)
            prior_end_lookup = self._prior_admission_ends(conversation)

        facts: list[ClinicalFact] = []
        for adm in admissions:
            checkpoint_key = f"{conversation.subject_id}:{adm.hadm_id}"
            if checkpoint_key in self._checkpoint:
                facts.extend(
                    _fact_from_dict(row) for row in self._checkpoint[checkpoint_key]["facts"]
                )
                continue

            admission_facts = self._extract_admission(
                conversation.subject_id,
                adm,
                prior_admission_end=prior_end_lookup.get(adm.hadm_id),
            )
            facts.extend(admission_facts)

            if self.checkpoint_path is not None:
                self._checkpoint[checkpoint_key] = {
                    "facts": [_fact_to_dict(f) for f in admission_facts]
                }
                self._save_checkpoint()

        return facts

    @staticmethod
    def _prior_admission_ends(
        conversation: Conversation, *, only: Admission | None = None
    ) -> dict[str, str]:
        """hadm_id -> the immediately preceding admission's `admission_end`,
        for "since my last visit" resolution (normalize.resolve_time)."""
        out: dict[str, str] = {}
        admissions = conversation.admissions
        for i, adm in enumerate(admissions):
            if only is not None and adm.hadm_id != only.hadm_id:
                continue
            if i > 0:
                out[adm.hadm_id] = admissions[i - 1].admission_end
        return out

    # -- one admission ------------------------------------------------------

    def _extract_admission(
        self, patient_id: str, admission: Admission, *, prior_admission_end: str | None
    ) -> list[ClinicalFact]:
        turns = admission.turns()
        turns_by_number = {t.turn_number: t for t in turns}

        if self.dry_run:
            raw_facts = self._rule_based_extract(turns, patient_id=patient_id, session_id=admission.hadm_id)
        else:
            raw_facts = self._call_llm_extract(turns)

        facts: list[ClinicalFact] = []
        for raw in raw_facts:
            fact = self._handle_candidate(
                raw,
                patient_id=patient_id,
                admission=admission,
                turns_by_number=turns_by_number,
                prior_admission_end=prior_admission_end,
            )
            if fact is not None:
                facts.append(fact)
        return facts

    # -- LLM path -------------------------------------------------------

    def _call_llm_extract(self, turns: list[Turn]) -> list[dict]:
        """One `llm.complete()` call per admission (literature/14 §2: Stage
        1+2 folded into one structured-output call, unchanged from before).

        Retrying on a transient provider failure or a schema-mismatch is now
        `llm.complete()`'s own responsibility (see `_MAX_RETRIES`'s removal
        note above) rather than a second copy of that logic here. This
        module's job is narrower: make exactly one seam call per admission,
        surface its cost/tokens, and defensively re-check the parsed shape
        before handing it to the handling-rules pipeline.
        """
        user = (
            "Extract clinical facts from this admission's dialogue turns.\n\n"
            f"{_format_turns_block(turns)}"
        )
        try:
            response = self.complete_fn(
                user,
                model=self.model,
                schema=_EXTRACTION_SCHEMA,
                system=_SYSTEM_PROMPT,
                max_tokens=self.max_tokens,
                temperature=0.0,  # deterministic extraction, not creative generation
                dry_run=False,  # only reached when self.dry_run is False (see _extract_admission)
                use_cache=self.use_cache,
            )
        except llm.LLMError:
            # MissingAPIKeyError / BudgetExceeded / SchemaValidationError /
            # ProviderError already name exactly what happened and how to
            # fix it (llm.py's own docstrings) — propagate unwrapped rather
            # than flattening every failure into one generic
            # ExtractionError, so a caller (the bulk-extraction script) can
            # catch BudgetExceeded / MissingAPIKeyError specifically to stop
            # a whole multi-admission run cleanly instead of just skipping
            # one admission.
            raise
        except Exception as exc:  # noqa: BLE001 - re-raise typed, never silently fall back
            raise ExtractionError(f"LLM extraction call failed: {exc}") from exc

        # The call already happened (and, outside dry_run/cache-hit, already
        # cost money) regardless of what the defensive shape check below
        # decides — account for it before validating.
        self.total_cost_usd += response.cost_usd
        self.total_prompt_tokens += response.prompt_tokens
        self.total_completion_tokens += response.completion_tokens

        data = response.parsed
        if not isinstance(data, dict) or "facts" not in data:
            raise ExtractionError(
                f"extraction response missing required 'facts' key: {data!r}"
            )
        facts = data["facts"]
        if not isinstance(facts, list):
            raise ExtractionError(
                f"extraction response 'facts' is not a list: {type(facts)!r}"
            )
        return facts

    # -- rule-based path --------------------------------------------------

    def _rule_based_extract(
        self, turns: list[Turn], *, patient_id: str, session_id: str
    ) -> list[dict]:
        raw_facts: list[dict] = []
        for turn in turns:
            candidates, skipped = _rule_based_candidates(turn)
            for cand in candidates:
                cand["turn_ids"] = [turn.turn_number]
                raw_facts.append(cand)
            for skip in skipped:
                self.assertion_log.append(
                    AssertionLogEntry(
                        patient_id=patient_id,
                        session_id=session_id,
                        turn_ids=[turn.turn_number],
                        i2b2_tag=skip["i2b2_tag"],
                        subject_name="PATIENT",
                        predicate_phrase=skip["predicate_phrase"],
                        object_name=skip["object_name"],
                        raw_time_expression="",
                        evidence_quote=turn.text.strip()[:200],
                        decision="dropped_by_rule",
                        reason=skip["reason"],
                    )
                )
        return raw_facts

    # -- handling rules (decisions/002) + normalize + validate -----------

    def _handle_candidate(
        self,
        raw: dict,
        *,
        patient_id: str,
        admission: Admission,
        turns_by_number: dict[int, Turn],
        prior_admission_end: str | None,
    ) -> ClinicalFact | None:
        session_id = admission.hadm_id
        turn_ids = [int(t) for t in raw.get("turn_ids") or []]
        i2b2_tag = raw.get("assertion", "present")
        predicate_phrase = raw.get("predicate_phrase", "")
        object_name = raw.get("object_name", "")
        object_type = raw.get("object_type", "") or ""
        subject_name_raw = raw.get("subject_name", "")
        subject_type_raw = raw.get("subject_type", "") or ""
        time_expression = raw.get("time_expression", "") or ""
        evidence_quote = raw.get("evidence_quote", "")
        raw_confidence = float(raw.get("confidence", 0.5) or 0.5)
        # PRN nature carried as a side-log fact attribute only (see
        # AssertionLogEntry.prn docstring) — never onto ClinicalFact.
        # `.get(..., False)` keeps this backward-compatible with any
        # scripted/cached candidate dict that predates this field.
        prn = bool(raw.get("prn", False))
        # 2026-08-17 subject/object role-reversal bug fix: mutated below,
        # in the reversal-detection block, once `predicate` is known. Local
        # here (not just at that block) so the `log()` closure — defined
        # before that block runs — reads its live value at call time.
        repaired_role_reversal = False

        def log(decision: str, reason: str, fact_id: str | None = None) -> None:
            self.assertion_log.append(
                AssertionLogEntry(
                    patient_id=patient_id,
                    session_id=session_id,
                    turn_ids=turn_ids,
                    i2b2_tag=i2b2_tag,
                    subject_name=subject_name_raw,
                    predicate_phrase=predicate_phrase,
                    object_name=object_name,
                    raw_time_expression=time_expression,
                    evidence_quote=evidence_quote,
                    decision=decision,
                    reason=reason,
                    emitted_fact_id=fact_id,
                    prn=prn,
                    repaired_role_reversal=repaired_role_reversal,
                )
            )

        # --- decisions/002 handling rules -----------------------------
        if i2b2_tag in ("conditional", "hypothetical"):
            log("dropped_by_rule", f"i2b2 tag {i2b2_tag!r} — skip beats mis-extracting")
            return None
        if i2b2_tag == "not-associated-with-patient":
            log("dropped_by_rule", "not-associated-with-patient — not attached to the patient")
            return None

        if i2b2_tag == "present":
            polarity = "asserted"
            confidence = raw_confidence
        elif i2b2_tag == "absent":
            polarity = "negated"
            confidence = raw_confidence
        elif i2b2_tag == "possible":
            polarity = "asserted"  # never a third polarity value (decision 002)
            confidence = raw_confidence * _POSSIBLE_CONFIDENCE_DISCOUNT
        else:
            log("dropped_invalid", f"unrecognized assertion tag {i2b2_tag!r}")
            return None

        if not turn_ids:
            log("dropped_invalid", "no turn_ids — provenance requires at least one turn")
            return None

        predicate = canonicalize_predicate(predicate_phrase)
        if predicate is None or predicate not in PREDICATES:
            log(
                "dropped_no_predicate",
                f"predicate phrase {predicate_phrase!r} did not canonicalize onto PREDICATES",
            )
            return None

        # --- 2026-08-17 subject/object role-reversal repair -------------
        # Every predicate except CURRENT_DOSAGE_OF has the PATIENT as
        # subject by convention (the system prompt's own rule, and
        # `_handle_candidate` hardcodes `subject = patient_id` for exactly
        # this set below) — so `object_name` must always be a real clinical
        # entity, never a token that just re-names the patient.
        # CURRENT_DOSAGE_OF is excluded on purpose: its object is a dose
        # value ("60mg"), never patient-referring in the first place, so
        # this check would be a category error there (§5 of the algorithm
        # doc). Detect the reversal generically — not a TAKES_MEDICATION
        # special case — because the same failure has been observed on
        # HAS_CONDITION/HAD_PROCEDURE candidates too (see the story's own
        # evidence and docs/algorithms/extraction-and-temporal-
        # normalization.md §7).
        if predicate != "CURRENT_DOSAGE_OF" and is_patient_ish_token(object_name, patient_id):
            if is_patient_ish_token(subject_name_raw, patient_id) or not subject_name_raw.strip():
                # Unrepairable: no real clinical entity name survives on
                # either side (both patient-ish, or the would-be repair
                # source is blank) — nothing to swap into place. Drop
                # visibly rather than emit an object.name of "patient".
                log(
                    "dropped_role_reversal_unrepairable",
                    f"object_name {object_name!r} is a patient-referring placeholder and "
                    f"subject_name {subject_name_raw!r} has no recoverable entity name — "
                    "cannot repair a reversed subject/object pair",
                )
                return None
            # Reversed: the model's own subject_name field actually holds
            # the real clinical entity (e.g. a drug name); object_name is
            # just a patient-referring placeholder. Repair by promoting
            # subject_name (+ its type, when given) into the object
            # position — `subject` itself is unaffected below, since it is
            # already always hardcoded to patient_id for this predicate
            # family regardless of what the model sent.
            object_name, object_type = subject_name_raw, (subject_type_raw or object_type)
            repaired_role_reversal = True

        anchor_turn = turns_by_number.get(turn_ids[0])
        if anchor_turn is None:
            log("dropped_invalid", f"turn_id {turn_ids[0]} not found in this admission")
            return None

        valid_from, time_confidence = resolve_time(
            time_expression, anchor_turn.time, prior_admission_end=prior_admission_end
        )
        observed_at, _ = resolve_time("", anchor_turn.time)
        confidence = max(0.0, min(1.0, confidence * time_confidence))

        source_class = "doctor" if anchor_turn.speaker == "Doctor" else "patient"

        # literature/14 §2 Stage 6 discount (this survey's own inference,
        # flagged there as needing day-1 validation) — patient self-report
        # of a formal diagnosis, not a clinician assertion.
        if predicate == "HAS_CONDITION" and source_class == "patient":
            confidence *= _PATIENT_SOURCED_CONDITION_DISCOUNT

        if predicate == "CURRENT_DOSAGE_OF":
            subject = EntityRef(name=subject_name_raw or object_name, type="Medication", canonical_id=0)
        else:
            subject = EntityRef(name=patient_id, type="Patient", canonical_id=0)

        # Canonicalize the model's free-text `object_type` onto the closed
        # `DOMAIN_ENTITY_TYPES` vocabulary HERE, at emit time — not later in
        # `graph/schema.label_for`. Three downstream consumers read this string
        # raw and never call `label_for`: `ids.mint_entity_id` hashes it into
        # the node id, `resolve.block` partitions on it, and `writer` stores it
        # as the node's `type` property. Normalizing at the label layer instead
        # would leave `medication` and `Medication` minting two ids in two
        # blocking partitions — duplicate nodes the label check cannot detect.
        #
        # Before this fix `type` fell back to the literal `"Entity"`, which is
        # not a graph label, so `writer._register_entity` skipped the fact onto
        # `WriteReport.skipped` — recorded, never raised. Real runs emitted
        # lowercase types for every fact, so 100% were dropped and ingest
        # reported success with `facts_written == 0`.
        canonical_object_type = normalize_entity_type(object_type)
        if canonical_object_type is None:
            log(
                "dropped_unmappable_entity_type",
                f"object_type {object_type!r} does not map onto any DOMAIN_ENTITY_TYPES "
                f"member (object_name={object_name!r}) — dropping rather than emitting a "
                "fact the graph writer would silently skip",
            )
            return None

        # An entity name that survives as a non-empty STRING but normalizes to
        # nothing (all punctuation/symbols, e.g. "---" or "?") has no identity:
        # `ids.normalize_key` is what builds the `<normalized_name>` fragment of
        # the identity key, so `mint_entity_id` correctly refuses it with a
        # ValueError. That assertion is right and stays strict — but raising
        # from inside entity resolution aborts the ENTIRE patient, which is what
        # killed subject 10312715 on the first real 3-patient run after ~7
        # minutes of paid extraction had already succeeded. Catch it here, where
        # one bad candidate costs one dropped fact instead of one lost patient.
        if not normalize_key(object_name) or (
            subject.type != "Patient" and not normalize_key(subject.name)
        ):
            log(
                "dropped_unusable_entity_name",
                f"entity name normalizes to empty (subject={subject.name!r}, "
                f"object={object_name!r}) — no stable identity key can be derived, "
                "so this fact cannot be minted or merged",
            )
            return None

        object_ref = EntityRef(name=object_name, type=canonical_object_type, canonical_id=0)

        fact_id = _placeholder_fact_id(
            patient_id=patient_id,
            session_id=session_id,
            subject_name=subject.name,
            predicate=predicate,
            object_name=object_ref.name,
            valid_from=valid_from,
            polarity=polarity,
        )

        fact = ClinicalFact(
            fact_id=fact_id,
            patient_id=patient_id,
            session_id=session_id,
            turn_ids=turn_ids,
            subject=subject,
            predicate=predicate,
            object=object_ref,
            valid_from=valid_from,
            valid_to=SENTINEL_VALID_TO,
            observed_at=observed_at,
            polarity=polarity,
            source_class=source_class,
            confidence=round(confidence, 4),
        )

        problems = validate(fact)
        if problems:
            log("dropped_invalid", "; ".join(problems))
            return None

        log(
            "emitted",
            f"i2b2={i2b2_tag} -> polarity={polarity}"
            + (" (prn)" if prn else "")
            + (" (role-reversal repaired)" if repaired_role_reversal else ""),
            fact_id=fact.fact_id,
        )
        return fact


def _fact_to_dict(fact: ClinicalFact) -> dict:
    return {
        "fact_id": fact.fact_id,
        "patient_id": fact.patient_id,
        "session_id": fact.session_id,
        "turn_ids": fact.turn_ids,
        "subject": {"name": fact.subject.name, "type": fact.subject.type, "canonical_id": fact.subject.canonical_id},
        "predicate": fact.predicate,
        "object": {"name": fact.object.name, "type": fact.object.type, "canonical_id": fact.object.canonical_id},
        "valid_from": fact.valid_from,
        "valid_to": fact.valid_to,
        "observed_at": fact.observed_at,
        "polarity": fact.polarity,
        "source_class": fact.source_class,
        "confidence": fact.confidence,
    }


def _fact_from_dict(row: dict) -> ClinicalFact:
    return ClinicalFact(
        fact_id=row["fact_id"],
        patient_id=row["patient_id"],
        session_id=row["session_id"],
        turn_ids=row["turn_ids"],
        subject=EntityRef(**row["subject"]),
        predicate=row["predicate"],
        object=EntityRef(**row["object"]),
        valid_from=row["valid_from"],
        valid_to=row["valid_to"],
        observed_at=row["observed_at"],
        polarity=row["polarity"],
        source_class=row["source_class"],
        confidence=row["confidence"],
    )


def extract_facts(
    conversation: Conversation,
    admission: Admission | None = None,
    *,
    extractor: Extractor | None = None,
) -> list[ClinicalFact]:
    """dialogue -> list[ClinicalFact]. Story contract entry point.

    ``extractor`` is the test/DI seam: pass an
    ``Extractor(complete_fn=fake, ...)`` (or ``dry_run=True``) to run
    without a real API call. The default ``Extractor()`` always attempts a
    real ``llm.complete()`` call (``dry_run=False``); if no OpenAI/Google
    key is configured, that call raises ``llm.MissingAPIKeyError`` rather
    than silently degrading to the rule-based fallback — see
    ``Extractor``'s and this module's docstrings.
    """
    extractor = extractor or Extractor()
    return extractor.extract(conversation, admission)
