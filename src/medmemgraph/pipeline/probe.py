"""pipeline/probe.py — GT-first contradiction/update/correction probe.

MedLoCoMo has no knowledge-update or contradiction category (verified type
distribution: adversarial 33.3%, medical_reasoning 16.7%, care_plan_rationale
16.7%, longitudinal_progression 16.6%, cross_admission_comparison 8.6%,
frequency_pattern 8.2% — see the return note for this story). BEAM's own
paper names contradiction resolution the weakest category across every
method it tested: "all methods -- including ours -- perform strongest in
abstention and weakest in contradiction resolution" (`literature/08-
synthetic-multisession-data-generation.md` claim R-SYN-24). This module is
the evidence that MedMemGraph's bi-temporal invalidation (`graph/
invalidate.py`, story E4-S5) actually does something no MedLoCoMo item can
exercise.

--------------------------------------------------------------------------
THE ONE RULE THIS MODULE MUST NOT BREAK (literature/08 TL;DR, and its own
stated central finding): **fix the ground truth BEFORE generating the
dialogue that must contain it — never extract an answer key by re-reading a
finished transcript.** LongMemEval fixes the evidence statement by hand
before finalizing the session that carries it (R-SYN-04/06); MediLongChat and
ELICITED — the two 2026 papers doing exactly this task (structured patient
record -> longitudinal clinical dialogue) — both independently converge on
the same ordering (R-SYN-27, R-SYN-28). LoCoMo did the opposite --
"answers... are directly taken from the conversations... as much as
possible" (R-SYN-15) -- and an independent audit found a 6.4% gold-answer
error rate and a judge that accepted 62.81% of deliberately wrong answers
(R-SYN-16, R-SYN-17).

This module's enforcement of that rule is structural, not a promise: every
`_scenario_*` builder below constructs the complete ground truth (predicate,
kind, both values, both expected answers, the distinguishing text fragments
used to grade them) from **only** a `random.Random` — no `Conversation`, no
`Admission`, no turn text is in scope when a `_ScenarioSpec` is built. Only
afterward does `_try_build_item` open a real patient's dialogue and inject
verbatim quotes carrying that already-fixed ground truth. `tests/
test_probe.py::test_scenario_builders_need_no_corpus` calls every builder
with zero `Conversation` object ever constructed, which is impossible if a
builder needed dialogue text to compute an answer.

--------------------------------------------------------------------------
Three cases, not one (literature/08 Decision Question 1's "Contradiction/
correction/update taxonomy", quoted): an **update** is "a genuine change
over time attributable to the patient's condition... both values were true
at their respective times" (dose changed, drug discontinued, condition
resolved); a **correction** is "a clinician fixing a recording error" (chart
says an allergy, a later note says it was a charting error); a
**contradiction** is "two statements that cannot both be true and are never
resolved within the history" -- testing whether a system notices the
conflict, versus silently picking one. `PROBE_KINDS` below is exactly these
three; `ProbeItem.kind` is never a fourth value.

--------------------------------------------------------------------------
QC gates (literature/08 Decision Question 2, items 1-3; item 4 -- a human-
verification pass -- is explicitly NOT implemented here: it requires an
actual human and is disclosed as a follow-up, not silently skipped):

  1. *No-context baseline* (LastingBench's formal definition, R-SYN-33):
     a model given the bare question and zero evidence must fail. Real
     production code (`eval.reader.read`) is called with `items=[]`, which
     structurally abstains before ever touching a model -- this is a
     behavioral property of the shipped reader, not an assumption this
     module makes about it.
  2. *Leave-one-out evidence ablation* (literature/08's own name for the
     check; LastingBench-style leakage detection generalized): the specific
     text fragment that grounds an answer must be present in the two
     injected turns and **absent from the patient's real, un-injected
     dialogue** -- checked by scanning the actual corpus text, not asserted.
  3. *Last-session-only / recency baseline*: the fact that should only be
     recoverable from the EARLIER admission must be absent from the most
     recent admission of the actual injected conversation -- so a system
     that only reads the latest session cannot get the as-of answer right
     by recency alone.

Every generated item is graded against all three gates before it is
returned; a failing item is dropped and the reason is recorded (`build_probe
(..., return_report=True)`), never silently discarded. See `docs/algorithms/
contradiction-probe.md` for the reviewer-facing walkthrough, the honest
"what this probe cannot claim" section, and the real-corpus run's numbers.

--------------------------------------------------------------------------
Deliberate design notes / stated assumptions (not hidden):

* Signature. The direct dispatch names `build_probe(subject_id, n_patients,
  seed) -> list[ProbeItem]`. Read literally, `subject_id` and `n_patients`
  together are ambiguous (one patient, or a count of patients?). Resolved
  here, stated rather than silently picked: `subject_id` is an *optional*
  override -- when given, every generated item is built against that ONE
  real patient (useful for tests against a single fixture, and for a
  reproducible single-patient demo) and `n_patients` is then read as "how
  many probe items to build against this one patient". When `subject_id` is
  `None` (the default, real-run path), `n_patients` real MedLoCoMo patients
  are drawn deterministically (seeded) from `list_patients()`, one item per
  patient by default (`items_per_patient=1`), matching this story's "5-15
  patients, not 101" framing 1:1. If this reading is wrong, only
  `build_probe`'s first ~20 lines need to change; nothing downstream depends
  on it.
* Cross-boundary import. `_run_qc`'s no-context gate calls
  `medmemgraph.eval.reader.read` (Evidence-owned) from this Pipeline-owned
  module. `collaborative/design/stories/E7/E7-S4.md`'s on-disk packet splits
  generator (`pipeline/probe.py`) from scorer (`eval/probe.py`); the direct
  dispatch that authored *this* story asked for the QC gates -- including a
  no-context check -- to live in `pipeline/probe.py` itself. Both are true
  at once; flagged here for `dev-architect` rather than silently resolved,
  since it is a real, if narrow, ownership crossing (one read-only call into
  a well-tested, already-shipped function, not a reimplementation).
* QC gates are offline/deterministic by construction (no `ANTHROPIC_API_KEY`
  is configured in this sandbox -- confirmed by grepping `.env`, zero
  matches). They never call a live model. This is *stronger* than the
  literature's own no-context gate for one reason and weaker for another,
  both stated in `docs/algorithms/contradiction-probe.md` §5: stronger,
  because the injected facts are fabricated for this run and cannot have
  been memorized by any model's pretraining (Test of Time's own argument for
  synthetic data, R-SYN-35, applies here for free); weaker, because it means
  this module cannot measure whether a *specific* real model's priors
  happen to guess a common value (e.g. a plausible-sounding dose) -- only
  that the reader's own architecture cannot recover a fact it was never
  given. A live-LLM enhancement path is named as a follow-up, not built.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from pathlib import Path
from random import Random
from typing import Literal, Sequence

from medmemgraph.contracts import PREDICATES, VALID_POLARITIES, VALID_SOURCE_CLASSES
from medmemgraph.eval.reader import read as _reader_read
from medmemgraph.pipeline.loader import (
    Admission,
    Conversation,
    LoaderError,
    list_patients,
    load_conversation,
)

__all__ = [
    "ProbeKind",
    "PROBE_KINDS",
    "InjectedFact",
    "QCReport",
    "ProbeItem",
    "build_probe",
    "export",
    "qc_summary",
]

ProbeKind = Literal["update", "correction", "contradiction"]
PROBE_KINDS: tuple[ProbeKind, ...] = ("update", "correction", "contradiction")
"""literature/08 Decision Question 1's taxonomy, exactly these three and no
fourth value: `update` (both true at their own time), `correction` (the
earlier statement was a recording error), `contradiction` (unresolved
conflict, never reconciled in the record)."""


# ---------------------------------------------------------------------------
# Ground-truth-first records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InjectedFact:
    """One half of a `ProbeItem`'s ground truth: a single statement, the
    admission it belongs to, and the turn(s) that carry it. Mirrors the
    fields the story asks for verbatim ("the original fact, the superseding
    fact, the admission each belongs to... the turn ids where each was
    introduced")."""

    predicate: str
    entity_name: str
    entity_type: str
    value_text: str
    polarity: str
    session_id: str
    turn_ids: list[int]
    valid_from: str
    source_class: str
    speaker: str
    quote: str
    key_fragment: str
    """The minimal literal substring of `quote` that QC gates search for.
    Deliberately specific (usually ties the entity name to its value, e.g.
    "metformin is 500 mg") rather than a bare number/name alone, so a
    common value (a common dose, a common drug) does not trivially collide
    with unrelated real text elsewhere in the patient's own history."""


@dataclass
class QCReport:
    """literature/08 Decision Question 2's checklist, items 1-3, run against
    one `ProbeItem`. `.passed` gates whether `build_probe` returns the item
    at all."""

    evidence_ablation_ok: bool
    evidence_ablation_detail: str
    no_context_ok: bool
    no_context_detail: str
    last_session_only_ok: bool
    last_session_only_detail: str

    @property
    def passed(self) -> bool:
        return self.evidence_ablation_ok and self.no_context_ok and self.last_session_only_ok

    def failing_gates(self) -> list[str]:
        gates = []
        if not self.evidence_ablation_ok:
            gates.append("evidence_ablation")
        if not self.no_context_ok:
            gates.append("no_context")
        if not self.last_session_only_ok:
            gates.append("last_session_only")
        return gates

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "evidence_ablation_ok": self.evidence_ablation_ok,
            "evidence_ablation_detail": self.evidence_ablation_detail,
            "no_context_ok": self.no_context_ok,
            "no_context_detail": self.no_context_detail,
            "last_session_only_ok": self.last_session_only_ok,
            "last_session_only_detail": self.last_session_only_detail,
        }


@dataclass
class ProbeItem:
    """The story's required shape: entity, original fact, superseding fact,
    the admission each belongs to (`InjectedFact.session_id`), the expected
    answer at "now", the expected answer "as of" a stated earlier date, and
    the turn ids where each was introduced (`InjectedFact.turn_ids`)."""

    probe_id: str
    patient_id: str
    kind: ProbeKind
    entity_name: str
    entity_type: str
    predicate: str
    original_fact: InjectedFact
    superseding_fact: InjectedFact
    as_of_date: str
    question_now: str
    question_as_of: str
    expected_answer_now: str
    expected_answer_as_of: str
    injected_conversation: Conversation = field(repr=False)
    """The real patient's `Conversation`, copied (never the on-disk object
    mutated -- `Admission`/`Conversation` are frozen dataclasses; every
    injected admission is built via `dataclasses.replace`), with exactly the
    two quotes above appended as new turns. Never written back to
    `data/medlocomo/`; `export()` does not serialize this field."""
    qc: QCReport | None = None

    def summary(self) -> dict:
        """A JSON-friendly view (excludes `injected_conversation`) for
        logging / the return-note real-run report."""
        return {
            "probe_id": self.probe_id,
            "patient_id": self.patient_id,
            "kind": self.kind,
            "predicate": self.predicate,
            "entity_name": self.entity_name,
            "original_fact": {
                "value": self.original_fact.value_text,
                "polarity": self.original_fact.polarity,
                "session_id": self.original_fact.session_id,
                "turn_ids": list(self.original_fact.turn_ids),
                "valid_from": self.original_fact.valid_from,
                "quote": self.original_fact.quote,
            },
            "superseding_fact": {
                "value": self.superseding_fact.value_text,
                "polarity": self.superseding_fact.polarity,
                "session_id": self.superseding_fact.session_id,
                "turn_ids": list(self.superseding_fact.turn_ids),
                "valid_from": self.superseding_fact.valid_from,
                "quote": self.superseding_fact.quote,
            },
            "as_of_date": self.as_of_date,
            "question_now": self.question_now,
            "expected_answer_now": self.expected_answer_now,
            "question_as_of": self.question_as_of,
            "expected_answer_as_of": self.expected_answer_as_of,
            "qc": self.qc.to_dict() if self.qc else None,
        }


# ---------------------------------------------------------------------------
# Scenario templates -- the ground truth, built from a Random alone.
# ---------------------------------------------------------------------------


@dataclass
class _ScenarioSpec:
    kind: ProbeKind
    predicate: str
    entity_name: str
    entity_type: str
    old_polarity: str
    new_polarity: str
    old_value_text: str
    new_value_text: str
    old_quote: str
    new_quote: str
    old_speaker: str
    new_speaker: str
    old_fragment: str
    new_fragment: str
    expected_answer_now: str
    expected_answer_as_of: str
    question_now: str
    question_as_of_template: str  # contains a literal "{date}" placeholder


_MEDICATION_DOSE_POOL: tuple[tuple[str, int, int], ...] = (
    ("metformin", 500, 1000),
    ("lisinopril", 10, 20),
    ("furosemide", 20, 40),
    ("atorvastatin", 20, 40),
    ("levothyroxine", 50, 75),
)
_MED_DISCONTINUE_POOL = (
    "spironolactone",
    "omeprazole",
    "gabapentin",
    "hydrochlorothiazide",
    "sertraline",
)
_CONDITION_POOL = (
    "ascites",
    "peripheral edema",
    "atrial fibrillation",
    "acute kidney injury",
    "pleural effusion",
)
_ALLERGEN_POOL = ("penicillin", "sulfa", "shellfish", "latex", "codeine")
_PROVIDER_POOL = (
    "Dr. Alvarez",
    "Dr. Okafor",
    "Dr. Nazari",
    "Dr. Whitfield",
    "Dr. Ibarra",
    "Dr. Solheim",
)


def _scenario_dose_change(rng: Random) -> _ScenarioSpec:
    """update: "a medication dose changed" (story text)."""
    med, old_mg, new_mg = rng.choice(_MEDICATION_DOSE_POOL)
    return _ScenarioSpec(
        kind="update",
        predicate="CURRENT_DOSAGE_OF",
        entity_name=med,
        entity_type="Medication",
        old_polarity="asserted",
        new_polarity="asserted",
        old_value_text=f"{old_mg} mg",
        new_value_text=f"{new_mg} mg",
        old_quote=f"Your current dose of {med} is {old_mg} mg, taken once daily.",
        new_quote=(
            f"We're adjusting your {med} -- the new dose is {new_mg} mg once daily "
            "starting today."
        ),
        old_speaker="Doctor",
        new_speaker="Doctor",
        old_fragment=f"{med} is {old_mg} mg",
        new_fragment=f"new dose is {new_mg} mg",
        expected_answer_now=f"{new_mg} mg",
        expected_answer_as_of=f"{old_mg} mg",
        question_now=f"What is the patient's current dose of {med}?",
        question_as_of_template=f"As of {{date}}, what was the patient's dose of {med}?",
    )


def _scenario_med_discontinued(rng: Random) -> _ScenarioSpec:
    """update: "a drug discontinued" (story text)."""
    med = rng.choice(_MED_DISCONTINUE_POOL)
    return _ScenarioSpec(
        kind="update",
        predicate="TAKES_MEDICATION",
        entity_name=med,
        entity_type="Medication",
        old_polarity="asserted",
        new_polarity="negated",
        old_value_text=f"taking {med}",
        new_value_text=f"not taking {med} (discontinued)",
        old_quote=f"I've been taking {med} every morning since my last visit.",
        new_quote=f"We're stopping the {med} now -- you don't need it anymore.",
        old_speaker="Patient",
        new_speaker="Doctor",
        old_fragment=f"taking {med} every morning",
        new_fragment=f"stopping the {med} now",
        expected_answer_now=f"no, {med} was discontinued",
        expected_answer_as_of=f"yes, taking {med}",
        question_now=f"Is the patient currently taking {med}?",
        question_as_of_template=f"As of {{date}}, was the patient taking {med}?",
    )


def _scenario_condition_resolved(rng: Random) -> _ScenarioSpec:
    """update: "a condition resolved" (story text)."""
    cond = rng.choice(_CONDITION_POOL)
    return _ScenarioSpec(
        kind="update",
        predicate="HAS_CONDITION",
        entity_name=cond,
        entity_type="Condition",
        old_polarity="asserted",
        new_polarity="negated",
        old_value_text=f"has {cond}",
        new_value_text=f"{cond} resolved",
        old_quote=f"The imaging confirms you have {cond}; we'll keep monitoring it.",
        new_quote=(
            f"Great news: your {cond} has resolved and is no longer showing up on exam."
        ),
        old_speaker="Doctor",
        new_speaker="Doctor",
        old_fragment=f"confirms you have {cond}",
        new_fragment=f"{cond} has resolved",
        expected_answer_now=f"no, {cond} has resolved",
        expected_answer_as_of=f"yes, had {cond}",
        question_now=f"Does the patient currently have {cond}?",
        question_as_of_template=f"As of {{date}}, did the patient have {cond}?",
    )


def _scenario_allergy_corrected(rng: Random) -> _ScenarioSpec:
    """correction: "an allergy corrected" (story text) -- explicitly framed
    as a charting error in the injected text, not a change in the patient's
    actual state, per literature/08's correction/update distinction."""
    allergen = rng.choice(_ALLERGEN_POOL)
    return _ScenarioSpec(
        kind="correction",
        predicate="HAS_ALLERGY_TO",
        entity_name=allergen,
        entity_type="Allergen",
        old_polarity="asserted",
        new_polarity="negated",
        old_value_text=f"allergic to {allergen}",
        new_value_text="no known drug allergies",
        old_quote=f"I see the chart notes a {allergen} allergy -- I'll make sure that's flagged.",
        new_quote=(
            f"I double-checked that {allergen} allergy entry from before -- it was a "
            "charting error from a prior visit. You have no known drug allergies."
        ),
        old_speaker="Doctor",
        new_speaker="Doctor",
        old_fragment=f"chart notes a {allergen} allergy",
        new_fragment="no known drug allergies",
        expected_answer_now=(
            f"no, that earlier {allergen} allergy entry was a charting error -- "
            "no known drug allergies"
        ),
        expected_answer_as_of=f"yes, documented {allergen} allergy",
        question_now=f"Does the patient have a documented {allergen} allergy?",
        question_as_of_template=(
            f"As of {{date}}, did the patient have a documented {allergen} allergy?"
        ),
    )


def _scenario_pcp_contradiction(rng: Random) -> _ScenarioSpec:
    """contradiction: two mutually exclusive reports, never reconciled --
    literature/08's third taxonomy member, the one BEAM's own paper (R-SYN-
    24) names as the field's weakest-tested and least-verified category."""
    provider_a, provider_b = rng.sample(_PROVIDER_POOL, 2)
    return _ScenarioSpec(
        kind="contradiction",
        predicate="PRIMARY_CARE_PROVIDER_OF",
        entity_name="primary care provider",
        entity_type="Provider",
        old_polarity="asserted",
        new_polarity="asserted",
        old_value_text=provider_a,
        new_value_text=provider_b,
        old_quote=f"My primary care doctor is {provider_a}.",
        new_quote=f"My primary care doctor is {provider_b}.",
        old_speaker="Patient",
        new_speaker="Patient",
        old_fragment=provider_a,
        new_fragment=provider_b,
        expected_answer_now=(
            f"conflicting reports on file -- {provider_a} vs {provider_b}; the record "
            "never reconciles which is current"
        ),
        expected_answer_as_of=f"{provider_a} (the only report on file as of that date)",
        question_now="Who is the patient's primary care provider?",
        question_as_of_template=(
            "As of {date}, who had the patient named as their primary care provider?"
        ),
    )


_SCENARIO_BUILDERS: tuple = (
    _scenario_dose_change,
    _scenario_med_discontinued,
    _scenario_condition_resolved,
    _scenario_allergy_corrected,
    _scenario_pcp_contradiction,
)


def _validate_spec(spec: _ScenarioSpec) -> None:
    """Defensive, loud-fail sanity check on a scenario builder's own output
    -- a failure here is a bug in this module, not a data problem, so it
    raises rather than silently dropping (mirrors `contracts.validate`'s
    fail-loud discipline, just for this module's own internal invariant)."""
    assert spec.kind in PROBE_KINDS, f"unknown probe kind {spec.kind!r}"
    assert spec.predicate in PREDICATES, f"{spec.predicate!r} not in contracts.PREDICATES"
    assert spec.old_polarity in VALID_POLARITIES
    assert spec.new_polarity in VALID_POLARITIES
    assert spec.old_fragment.lower() in spec.old_quote.lower(), (
        f"scenario bug: old_fragment {spec.old_fragment!r} not a substring of "
        f"old_quote {spec.old_quote!r}"
    )
    assert spec.new_fragment.lower() in spec.new_quote.lower(), (
        f"scenario bug: new_fragment {spec.new_fragment!r} not a substring of "
        f"new_quote {spec.new_quote!r}"
    )


# ---------------------------------------------------------------------------
# Injection mechanics -- opens a real patient's dialogue, builds a COPY.
# ---------------------------------------------------------------------------


def _source_class(speaker: str) -> str:
    return "doctor" if speaker == "Doctor" else "patient"


def _append_turn(admission: Admission, speaker: str, text: str) -> tuple[dict, int, datetime]:
    """Build one new conversation-line dict, in the corpus's own on-disk
    shape (space-separated `time`, per `literature/14`/`normalize.py`'s
    documented format), anchored just after the admission's last real turn.
    Returns `(line, turn_number, time_dt)` -- never mutates `admission`."""
    existing = admission.conversation_lines
    last_turn_no = max((line["turn_number"] for line in existing), default=0)
    turn_no = last_turn_no + 1
    last_time_str = existing[-1]["time"] if existing else admission.admission_start
    time_dt = datetime.fromisoformat(last_time_str) + timedelta(minutes=5)
    line = {
        "turn_number": turn_no,
        "time": time_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "speaker": speaker,
        "text": text,
    }
    return line, turn_no, time_dt


def _pick_as_of(
    old_admission: Admission, new_admission: Admission, old_fact_dt: datetime, new_fact_dt: datetime
) -> datetime:
    """A timestamp strictly between the original fact and the superseding
    one -- "as of" that date, only the original fact is on record. Prefers
    the old admission's own `admission_end` (the natural "as of discharge"
    reading); falls back to the midpoint between the two facts' own times if
    that doesn't land strictly between them (defensive; real MedLoCoMo
    admissions for one patient do not overlap, so the fallback should not
    fire on real data)."""
    old_end = datetime.fromisoformat(old_admission.admission_end)
    if old_fact_dt < old_end < new_fact_dt:
        return old_end
    return old_fact_dt + (new_fact_dt - old_fact_dt) / 2


def _try_build_item(
    conv: Conversation,
    spec: _ScenarioSpec,
    rng: Random,
    *,
    probe_index: int,
    used_signatures: set[str],
) -> tuple[ProbeItem | None, str]:
    """One attempt: pick two real, chronologically distinct admissions of
    `conv`, verify the scenario's grounding fragments do not already occur
    anywhere in this patient's real (un-injected) dialogue, then build the
    injected copy. Returns `(item_or_None, reason)` -- `reason` is only
    meaningful when the first element is `None`."""
    _validate_spec(spec)

    signature = f"{spec.kind}:{spec.predicate}:{spec.entity_name}"
    if signature in used_signatures:
        return None, f"duplicate scenario signature {signature!r} already used for this patient"

    admissions = sorted(conv.admissions, key=lambda a: a.admission_start)
    if len(admissions) < 2:
        return None, "fewer than 2 admissions"

    old_idx = rng.randrange(0, len(admissions) - 1)
    new_idx = rng.randrange(old_idx + 1, len(admissions))
    old_admission = admissions[old_idx]
    new_admission = admissions[new_idx]

    real_text = " \n ".join(t.text for t in conv.turns()).lower()
    if spec.old_fragment.lower() in real_text or spec.new_fragment.lower() in real_text:
        return None, (
            f"grounding fragment collides with real corpus text for scenario {spec.kind}/"
            f"{spec.predicate}/{spec.entity_name}"
        )

    old_line, old_turn_no, old_time_dt = _append_turn(old_admission, spec.old_speaker, spec.old_quote)
    new_line, new_turn_no, new_time_dt = _append_turn(new_admission, spec.new_speaker, spec.new_quote)

    injected_old = replace(old_admission, conversation_lines=old_admission.conversation_lines + (old_line,))
    injected_new = replace(new_admission, conversation_lines=new_admission.conversation_lines + (new_line,))

    injected_admissions = []
    for a in conv.admissions:
        if a.hadm_id == old_admission.hadm_id:
            injected_admissions.append(injected_old)
        elif a.hadm_id == new_admission.hadm_id:
            injected_admissions.append(injected_new)
        else:
            injected_admissions.append(a)
    injected_conversation = replace(conv, admissions=tuple(injected_admissions))

    as_of_dt = _pick_as_of(old_admission, new_admission, old_time_dt, new_time_dt)
    as_of_display = as_of_dt.date().isoformat()

    probe_id = (
        f"probe-{conv.subject_id}-{spec.kind}-{old_admission.hadm_id}-"
        f"{new_admission.hadm_id}-{probe_index:03d}"
    )

    original_fact = InjectedFact(
        predicate=spec.predicate,
        entity_name=spec.entity_name,
        entity_type=spec.entity_type,
        value_text=spec.old_value_text,
        polarity=spec.old_polarity,
        session_id=old_admission.hadm_id,
        turn_ids=[old_turn_no],
        valid_from=old_time_dt.isoformat(),
        source_class=_source_class(spec.old_speaker),
        speaker=spec.old_speaker,
        quote=spec.old_quote,
        key_fragment=spec.old_fragment,
    )
    superseding_fact = InjectedFact(
        predicate=spec.predicate,
        entity_name=spec.entity_name,
        entity_type=spec.entity_type,
        value_text=spec.new_value_text,
        polarity=spec.new_polarity,
        session_id=new_admission.hadm_id,
        turn_ids=[new_turn_no],
        valid_from=new_time_dt.isoformat(),
        source_class=_source_class(spec.new_speaker),
        speaker=spec.new_speaker,
        quote=spec.new_quote,
        key_fragment=spec.new_fragment,
    )
    assert original_fact.source_class in VALID_SOURCE_CLASSES
    assert superseding_fact.source_class in VALID_SOURCE_CLASSES

    item = ProbeItem(
        probe_id=probe_id,
        patient_id=conv.subject_id,
        kind=spec.kind,
        entity_name=spec.entity_name,
        entity_type=spec.entity_type,
        predicate=spec.predicate,
        original_fact=original_fact,
        superseding_fact=superseding_fact,
        as_of_date=as_of_dt.isoformat(),
        question_now=spec.question_now,
        question_as_of=spec.question_as_of_template.format(date=as_of_display),
        expected_answer_now=spec.expected_answer_now,
        expected_answer_as_of=spec.expected_answer_as_of,
        injected_conversation=injected_conversation,
    )
    used_signatures.add(signature)
    return item, ""


# ---------------------------------------------------------------------------
# QC gates -- literature/08 Decision Question 2, items 1-3.
# ---------------------------------------------------------------------------


def _run_qc(item: ProbeItem, conv: Conversation) -> QCReport:
    """`conv` is the ORIGINAL, un-injected conversation (for the ablation
    gate's "without the evidence" half); `item.injected_conversation` is the
    actual artifact a downstream reader would see (for the recency gate)."""
    old_frag = item.original_fact.key_fragment.lower()
    new_frag = item.superseding_fact.key_fragment.lower()

    # --- Gate 1: leave-one-out evidence ablation -------------------------
    with_evidence_text = f"{item.original_fact.quote} {item.superseding_fact.quote}".lower()
    positive_control_ok = old_frag in with_evidence_text and new_frag in with_evidence_text

    real_text = " \n ".join(t.text for t in conv.turns()).lower()
    old_absent_without_evidence = old_frag not in real_text
    new_absent_without_evidence = new_frag not in real_text
    ablation_ok = positive_control_ok and old_absent_without_evidence and new_absent_without_evidence
    ablation_detail = (
        "answerable with the two injected turns and unanswerable without them"
        if ablation_ok
        else (
            f"positive_control_ok={positive_control_ok} "
            f"old_absent_without_evidence={old_absent_without_evidence} "
            f"new_absent_without_evidence={new_absent_without_evidence}"
        )
    )

    # --- Gate 2: no-context baseline --------------------------------------
    # Real production code, not a probe-local stand-in: eval.reader.read()'s
    # own "no items" branch structurally abstains before any model is
    # touched (see reader.py's `if not items:` early return).
    now_answer = _reader_read(item.question_now, [], mode="direct", dry_run=True)
    asof_answer = _reader_read(item.question_as_of, [], mode="direct", dry_run=True)
    no_ctx_ok = (
        now_answer.abstained
        and asof_answer.abstained
        and old_frag not in now_answer.text.lower()
        and new_frag not in now_answer.text.lower()
    )
    no_ctx_detail = (
        "reader.read() with zero evidence abstained on both questions"
        if no_ctx_ok
        else f"now.abstained={now_answer.abstained} asof.abstained={asof_answer.abstained}"
    )

    # --- Gate 3: last-session-only / recency baseline ----------------------
    recency_admission = max(item.injected_conversation.admissions, key=lambda a: a.admission_start)
    recency_text = " \n ".join(line["text"] for line in recency_admission.conversation_lines).lower()
    last_session_ok = old_frag not in recency_text
    last_session_detail = (
        f"old-fact fragment absent from the most recent admission "
        f"{recency_admission.hadm_id!r} -- the as-of answer genuinely needs the earlier one"
        if last_session_ok
        else (
            f"old-fact fragment leaked into the most recent admission "
            f"{recency_admission.hadm_id!r} -- recency alone would answer this"
        )
    )

    return QCReport(
        evidence_ablation_ok=ablation_ok,
        evidence_ablation_detail=ablation_detail,
        no_context_ok=no_ctx_ok,
        no_context_detail=no_ctx_detail,
        last_session_only_ok=last_session_ok,
        last_session_only_detail=last_session_detail,
    )


# ---------------------------------------------------------------------------
# build_probe -- the public entry point.
# ---------------------------------------------------------------------------

_ATTEMPT_BUDGET_PER_SLOT = 8
_MIN_ATTEMPT_BUDGET = 20


def build_probe(
    subject_id: str | None = None,
    n_patients: int = 8,
    seed: int = 0,
    *,
    root: str | os.PathLike[str] | None = None,
    items_per_patient: int = 1,
    return_report: bool = False,
) -> list[ProbeItem] | tuple[list[ProbeItem], dict]:
    """Ground-truth-first contradiction/update/correction probe over REAL
    MedLoCoMo dialogue (never `formed_packet.json`, never a synthetic-only
    patient -- this module opens dialogue exclusively via
    `pipeline.loader.load_conversation`/`list_patients`, the allowlisted
    loader). See the module docstring's "Deliberate design notes" for the
    `subject_id`/`n_patients` signature resolution.

    Returns only QC-passing items by default. Pass `return_report=True` to
    get `(items, report)` where `report` carries `n_requested`,
    `n_generated`, `n_dropped`, and a `dropped` list of
    `{patient_id, probe_id|None, stage, gates_failed, detail}` records --
    "drop any item that fails its gate and report the drop rate" (story
    text). This function has a declared stopping point: a bounded attempt
    budget, not an unbounded retry loop -- if the budget is exhausted before
    `n_patients` items are found, it returns fewer items and the report
    says so, rather than looping forever.
    """
    rng = Random(seed)

    if subject_id is not None:
        carriers = [subject_id] * max(n_patients, 0)
    else:
        candidates = list(list_patients(root))
        rng.shuffle(candidates)
        chosen: list[str] = []
        for sid in candidates:
            if len(chosen) >= n_patients:
                break
            try:
                conv = load_conversation(sid, root)
            except LoaderError:
                continue
            if len(conv.admissions) >= 2:
                chosen.append(sid)
        carriers = [sid for sid in chosen for _ in range(items_per_patient)]

    items: list[ProbeItem] = []
    dropped: list[dict] = []
    conv_cache: dict[str, Conversation] = {}
    used_signatures: dict[str, set[str]] = {}

    scenario_offset = rng.randrange(len(_SCENARIO_BUILDERS)) if _SCENARIO_BUILDERS else 0
    attempt_budget = max(len(carriers) * _ATTEMPT_BUDGET_PER_SLOT, _MIN_ATTEMPT_BUDGET)

    slot = 0
    attempts = 0
    while slot < len(carriers) and attempts < attempt_budget:
        attempts += 1
        sid = carriers[slot]

        conv = conv_cache.get(sid)
        if conv is None:
            try:
                conv = load_conversation(sid, root)
            except LoaderError as exc:
                dropped.append(
                    {
                        "patient_id": sid,
                        "probe_id": None,
                        "stage": "construction",
                        "gates_failed": [],
                        "detail": f"load failed: {exc}",
                    }
                )
                slot += 1
                continue
            conv_cache[sid] = conv

        if len(conv.admissions) < 2:
            dropped.append(
                {
                    "patient_id": sid,
                    "probe_id": None,
                    "stage": "construction",
                    "gates_failed": [],
                    "detail": "fewer than 2 admissions",
                }
            )
            slot += 1
            continue

        builder_idx = (scenario_offset + slot) % len(_SCENARIO_BUILDERS)
        spec = _SCENARIO_BUILDERS[builder_idx](rng)

        signatures = used_signatures.setdefault(sid, set())
        item, reason = _try_build_item(conv, spec, rng, probe_index=slot, used_signatures=signatures)
        if item is None:
            dropped.append(
                {
                    "patient_id": sid,
                    "probe_id": None,
                    "stage": "construction",
                    "gates_failed": [],
                    "detail": reason,
                }
            )
            continue

        qc = _run_qc(item, conv)
        item.qc = qc
        if qc.passed:
            items.append(item)
            slot += 1
        else:
            dropped.append(
                {
                    "patient_id": sid,
                    "probe_id": item.probe_id,
                    "stage": "qc",
                    "gates_failed": qc.failing_gates(),
                    "detail": "; ".join(
                        d
                        for ok, d in (
                            (qc.evidence_ablation_ok, qc.evidence_ablation_detail),
                            (qc.no_context_ok, qc.no_context_detail),
                            (qc.last_session_only_ok, qc.last_session_only_detail),
                        )
                        if not ok
                    ),
                }
            )

    if not return_report:
        return items

    report = {
        "n_requested": len(carriers),
        "n_generated": len(items),
        "n_dropped": len(dropped),
        "drop_rate": (len(dropped) / len(carriers)) if carriers else 0.0,
        "dropped": dropped,
    }
    return items, report


def qc_summary(report: dict) -> str:
    """Human-readable rollup of a `build_probe(..., return_report=True)`
    report -- per-gate drop counts, for the story's "report the numbers"
    requirement."""
    lines = [
        f"requested={report['n_requested']} generated={report['n_generated']} "
        f"dropped={report['n_dropped']} drop_rate={report['drop_rate']:.1%}"
    ]
    gate_counts: dict[str, int] = {}
    for d in report["dropped"]:
        if d["stage"] == "construction":
            gate_counts["construction"] = gate_counts.get("construction", 0) + 1
        for gate in d["gates_failed"]:
            gate_counts[gate] = gate_counts.get(gate, 0) + 1
    for gate in sorted(gate_counts):
        lines.append(f"  dropped by {gate}: {gate_counts[gate]}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# export -- benchmark_qa.json-shaped, so the existing Evidence harness can
# consume it with no special-casing (story requirement).
# ---------------------------------------------------------------------------


def export(items: Sequence[ProbeItem], path: str | os.PathLike[str]) -> Path:
    """Write `items` as `{"qas": [...]}`, byte-for-byte the same shape
    `pipeline.loader.load_qa` returns from a real `benchmark_qa.json`
    (`qa_id, scope, question_type, question, answer, evidence
    {admissions, turn_ids}`) -- verified in `tests/test_probe.py` by
    round-tripping the written file through `load_qa` itself, not just by
    eyeballing the JSON shape.

    Two QA rows per `ProbeItem`: `..._now` (needs both admissions to know
    the current, superseding state -- `scope="cross_admission"`) and
    `..._asof` (answerable from the earlier admission alone --
    `scope="single_admission"`). `question_type` is the item's `kind`
    (`update`/`correction`/`contradiction`) -- the harness's own
    `_aggregate()` buckets any category it doesn't already know about
    alphabetically after the six MedLoCoMo ones, so these three new rows
    show up as their own table rows with zero changes to `eval/harness.py`.
    """
    qas = []
    for item in items:
        qas.append(
            {
                "qa_id": f"{item.probe_id}__now",
                "scope": "cross_admission",
                "question_type": item.kind,
                "question": item.question_now,
                "answer": item.expected_answer_now,
                "evidence": {
                    "admissions": [item.original_fact.session_id, item.superseding_fact.session_id],
                    "turn_ids": list(item.original_fact.turn_ids) + list(item.superseding_fact.turn_ids),
                },
            }
        )
        qas.append(
            {
                "qa_id": f"{item.probe_id}__asof",
                "scope": "single_admission",
                "question_type": item.kind,
                "question": item.question_as_of,
                "answer": item.expected_answer_as_of,
                "evidence": {
                    "admissions": [item.original_fact.session_id],
                    "turn_ids": list(item.original_fact.turn_ids),
                },
            }
        )

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"qas": qas}, indent=2))
    return out_path
