"""tests/test_extract.py — the six i2b2 assertion-tag handling rules
(decisions/002), the dry_run/fail-loud transport contract, and cost/token
surfacing, all using a stubbed `complete_fn` (matching `llm.complete`'s own
call signature) so the default gate runs with zero network calls and no
provider key configured — `Extractor(complete_fn=fake)` forces the real
(non-dry_run) code path with a scripted response.

One test (`TestMissingApiKeyRaisesRatherThanSilentlyFallingBack
.test_default_construction_with_no_key_and_no_dry_run_raises`) deliberately
exercises the REAL `llm.complete` default end-to-end, with every key source
blanked out via `monkeypatch`, to prove the actual wiring — not just a mock
of it — raises rather than silently degrading. It never touches the real
repo `.env` or `data/llm_cache/` (isolated the same way `tests/test_llm.py`'s
own `isolate_llm_module` fixture isolates every test in that file) and never
makes a real network call (key resolution fails before any client is
constructed).

Synthetic fixture only (not vendored MedLoCoMo text) — a 6-10 turn admission
covering: an asserted medication, a denied allergy, a possible/uncertain
condition (with a relative-time expression to also exercise
`normalize.resolve_time` end-to-end), a conditional statement, a hypothetical
statement, and an other-subject (family-history) statement. Mirrors the
scenario the on-disk E2-S1 story's own acceptance criteria describe.
"""

from __future__ import annotations

import json

import pytest

from medmemgraph import llm
from medmemgraph.contracts import PREDICATES, SENTINEL_VALID_TO
from medmemgraph.pipeline.extract import ExtractionError, Extractor, extract_facts
from medmemgraph.pipeline.loader import Admission, Conversation

PATIENT_ID = "patient-0042"
HADM_ID = "admission-0007"

_TURNS = [
    {
        "turn_number": 1,
        "time": "2124-03-01 09:00:00",
        "speaker": "Doctor",
        "text": "You are prescribed metformin 500mg for your diabetes.",
    },
    {
        "turn_number": 2,
        "time": "2124-03-01 09:02:00",
        "speaker": "Patient",
        "text": "Okay, I'll take it. Do I have any allergies I should know about?",
    },
    {
        "turn_number": 3,
        "time": "2124-03-01 09:03:00",
        "speaker": "Doctor",
        "text": "You have no known penicillin allergy on file.",
    },
    {
        "turn_number": 4,
        "time": "2124-03-01 09:05:00",
        "speaker": "Patient",
        "text": "If this were bacterial, would you give me antibiotics?",
    },
    {
        "turn_number": 5,
        "time": "2124-03-01 09:06:00",
        "speaker": "Doctor",
        "text": "If this were bacterial we would start antibiotics, but it looks viral.",
    },
    {
        "turn_number": 6,
        "time": "2124-03-01 09:08:00",
        "speaker": "Patient",
        "text": "My mother has diabetes too, runs in the family.",
    },
    {
        "turn_number": 7,
        "time": "2124-03-01 09:10:00",
        "speaker": "Doctor",
        "text": (
            "Your creatinine looked high two weeks ago; it might suggest early "
            "kidney disease, we'll keep watching."
        ),
    },
    {
        "turn_number": 8,
        "time": "2124-03-01 09:12:00",
        "speaker": "Patient",
        "text": "Thanks doctor.",
    },
]


def _build_conversation() -> Conversation:
    admission = Admission(
        hadm_id=HADM_ID,
        admission_start=_TURNS[0]["time"],
        admission_end=_TURNS[-1]["time"],
        conversation_lines=tuple(_TURNS),
    )
    return Conversation(
        subject_id=PATIENT_ID,
        processed_hadm_ids=(HADM_ID,),
        admissions=(admission,),
    )


# ---------------------------------------------------------------------------
# FakeComplete — a scripted `llm.complete`-shaped callable, injected via
# `Extractor(complete_fn=...)`. Mirrors `tests/test_llm.py`'s own
# `FakeOpenAIChat` pattern: `responses` is a list of either a dict (the
# `facts` payload -> wrapped into a successful `llm.LLMResponse`, exactly
# what `response.parsed` already looks like on the real seam) or an
# Exception instance (raised as-is, e.g. a scripted `llm.MissingAPIKeyError`
# / `llm.SchemaValidationError` simulating what the real seam does after
# exhausting its own retries). Consumed in order, one entry per call.
# ---------------------------------------------------------------------------


class FakeComplete:
    def __init__(self, responses: list) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    def __call__(self, prompt: str, **kwargs) -> llm.LLMResponse:
        self.calls.append({"prompt": prompt, **kwargs})
        if not self._responses:
            raise RuntimeError("FakeComplete: no more scripted responses")
        item = self._responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return llm.LLMResponse(
            text=json.dumps(item),
            parsed=item,
            model=kwargs.get("model") or "fake-model",
            prompt_tokens=10,
            completion_tokens=5,
            cost_usd=0.0001,
            latency_ms=1.0,
            cached=False,
            attempts=1,
        )


def _candidate(
    *,
    predicate_phrase: str,
    object_name: str,
    object_type: str,
    assertion: str,
    turn_ids: list[int],
    confidence: float = 0.9,
    time_expression: str = "",
    subject_name: str = "PATIENT",
    subject_type: str = "Patient",
    prn: bool = False,
) -> dict:
    return {
        "subject_name": subject_name,
        "subject_type": subject_type,
        "predicate_phrase": predicate_phrase,
        "object_name": object_name,
        "object_type": object_type,
        "assertion": assertion,
        "prn": prn,
        "time_expression": time_expression,
        "turn_ids": turn_ids,
        "confidence": confidence,
        "evidence_quote": "evidence",
    }


_ALL_SIX_TAGS_PAYLOAD = {
    "facts": [
        _candidate(
            predicate_phrase="is prescribed",
            object_name="metformin",
            object_type="Medication",
            assertion="present",
            turn_ids=[1],
            confidence=0.9,
        ),
        _candidate(
            predicate_phrase="allergic to",
            object_name="penicillin",
            object_type="Allergen",
            assertion="absent",
            turn_ids=[3],
            confidence=0.9,
        ),
        _candidate(
            predicate_phrase="diagnosed with",
            object_name="early kidney disease",
            object_type="Condition",
            assertion="possible",
            turn_ids=[7],
            confidence=0.8,
            time_expression="two weeks ago",
        ),
        _candidate(
            predicate_phrase="is prescribed",
            object_name="antibiotics",
            object_type="Medication",
            assertion="conditional",
            turn_ids=[5],
        ),
        _candidate(
            predicate_phrase="is prescribed",
            object_name="antibiotics",
            object_type="Medication",
            assertion="hypothetical",
            turn_ids=[4],
        ),
        _candidate(
            predicate_phrase="diagnosed with",
            object_name="diabetes",
            object_type="Condition",
            assertion="not-associated-with-patient",
            turn_ids=[6],
        ),
    ]
}


# ---------------------------------------------------------------------------
# The six assertion-tag handling rules (decisions/002)
# ---------------------------------------------------------------------------


class TestSixAssertionTagHandlingRules:
    @pytest.fixture()
    def facts(self):
        conversation = _build_conversation()
        complete_fn = FakeComplete([_ALL_SIX_TAGS_PAYLOAD])
        extractor = Extractor(complete_fn=complete_fn)
        assert not extractor.dry_run, "default construction must attempt the real LLM code path"
        assert extractor.kind == "llm"
        return extractor.extract(conversation)

    def test_present_emits_asserted(self, facts):
        matches = [f for f in facts if f.predicate == "TAKES_MEDICATION" and f.object.name == "metformin"]
        assert len(matches) == 1
        assert matches[0].polarity == "asserted"
        assert matches[0].confidence == pytest.approx(0.9)

    def test_absent_emits_negated(self, facts):
        matches = [f for f in facts if f.predicate == "HAS_ALLERGY_TO"]
        assert len(matches) == 1
        assert matches[0].polarity == "negated"
        assert matches[0].object.name == "penicillin"

    def test_possible_emits_asserted_with_confidence_discount(self, facts):
        matches = [f for f in facts if f.object.name == "early kidney disease"]
        assert len(matches) == 1
        fact = matches[0]
        assert fact.polarity == "asserted"  # never a third polarity value
        assert fact.confidence < 0.8  # discounted below the raw 0.8

    def test_possible_time_expression_still_resolves_via_normalize(self, facts):
        matches = [f for f in facts if f.object.name == "early kidney disease"]
        fact = matches[0]
        # turn 7's own time is 2124-03-01; "two weeks ago" must move valid_from
        # backward, proving normalize.resolve_time ran on the LLM path too.
        assert fact.valid_from < "2124-03-01T09:10:00"
        assert fact.valid_from.startswith("2124-02-1")

    def test_conditional_is_not_emitted(self, facts):
        assert not any(f.predicate == "TAKES_MEDICATION" and f.turn_ids == [5] for f in facts)

    def test_hypothetical_is_not_emitted(self, facts):
        assert not any(f.predicate == "TAKES_MEDICATION" and f.turn_ids == [4] for f in facts)

    def test_not_associated_with_patient_is_not_emitted(self, facts):
        # No HAS_CONDITION diabetes fact at all — critically, none attached
        # to the patient either.
        assert not any(f.object.name == "diabetes" for f in facts)
        assert not any(
            f.predicate == "HAS_CONDITION" and f.subject.name == PATIENT_ID and f.object.name == "diabetes"
            for f in facts
        )

    def test_exactly_three_facts_survive_from_six_candidates(self, facts):
        # present, absent, possible emit; conditional/hypothetical/other-subject drop.
        assert len(facts) == 3

    def test_side_log_records_all_six_i2b2_tags(self):
        conversation = _build_conversation()
        complete_fn = FakeComplete([_ALL_SIX_TAGS_PAYLOAD])
        extractor = Extractor(complete_fn=complete_fn)
        extractor.extract(conversation)
        tags_seen = {entry.i2b2_tag for entry in extractor.assertion_log}
        assert tags_seen == {
            "present",
            "absent",
            "possible",
            "conditional",
            "hypothetical",
            "not-associated-with-patient",
        }
        # decisions/002: the i2b2 tag never drives a third polarity value.
        for entry in extractor.assertion_log:
            if entry.decision == "emitted":
                assert entry.i2b2_tag in ("present", "absent", "possible")


# ---------------------------------------------------------------------------
# Field-shape acceptance criteria
# ---------------------------------------------------------------------------


class TestEmittedFactFieldShapes:
    @pytest.fixture()
    def facts(self):
        conversation = _build_conversation()
        complete_fn = FakeComplete([_ALL_SIX_TAGS_PAYLOAD])
        extractor = Extractor(complete_fn=complete_fn)
        return extractor.extract(conversation)

    def test_polarity_is_binary(self, facts):
        for f in facts:
            assert f.polarity in ("asserted", "negated")

    def test_source_class_is_doctor_or_patient(self, facts):
        for f in facts:
            assert f.source_class in ("doctor", "patient")

    def test_valid_to_is_always_the_sentinel(self, facts):
        for f in facts:
            assert f.valid_to == SENTINEL_VALID_TO == "9999-12-31T00:00:00"

    def test_predicate_is_in_closed_vocabulary(self, facts):
        for f in facts:
            assert f.predicate in PREDICATES

    def test_turn_ids_non_empty(self, facts):
        for f in facts:
            assert len(f.turn_ids) > 0

    def test_session_id_equals_turn_hadm_id(self, facts):
        for f in facts:
            assert f.session_id == HADM_ID

    def test_source_class_matches_the_originating_turn_speaker(self, facts):
        met = next(f for f in facts if f.object.name == "metformin")
        assert met.source_class == "doctor"  # turn 1 is Doctor
        allergy = next(f for f in facts if f.predicate == "HAS_ALLERGY_TO")
        assert allergy.source_class == "doctor"  # turn 3 is Doctor


# ---------------------------------------------------------------------------
# LLM down / unparseable output -> raise, zero facts. `llm.LLMError`
# subclasses (SchemaValidationError, BudgetExceeded, MissingAPIKeyError,
# ProviderError) propagate UNWRAPPED (see extract.py's `_call_llm_extract`
# docstring) — they are already typed and actionable on their own; only a
# genuinely unexpected exception gets flattened into `ExtractionError`.
# ---------------------------------------------------------------------------


class TestLLMFailureRaisesRatherThanSilentlyFallingBack:
    def test_unexpected_exception_is_wrapped_as_extraction_error(self):
        complete_fn = FakeComplete([RuntimeError("simulated API outage")])
        conversation = _build_conversation()
        extractor = Extractor(complete_fn=complete_fn)
        with pytest.raises(ExtractionError):
            extractor.extract(conversation)

    def test_schema_validation_failure_from_the_seam_propagates_unwrapped(self):
        # llm.complete() already retries a schema mismatch internally (up to
        # 2 extra round trips — see tests/test_llm.py's own
        # TestSchemaValidation, not duplicated here). This simulates the
        # seam finally giving up and proves extract.py neither masks that
        # as a generic ExtractionError nor silently returns zero facts
        # without raising.
        complete_fn = FakeComplete([llm.SchemaValidationError("bad json after the seam's own retries")])
        conversation = _build_conversation()
        extractor = Extractor(complete_fn=complete_fn)
        with pytest.raises(llm.SchemaValidationError):
            extractor.extract(conversation)
        assert len(complete_fn.calls) == 1  # one seam call; its own retries are internal, not visible here

    def test_missing_facts_key_raises(self):
        complete_fn = FakeComplete([{"not_facts": []}])
        conversation = _build_conversation()
        extractor = Extractor(complete_fn=complete_fn)
        with pytest.raises(ExtractionError):
            extractor.extract(conversation)

    def test_facts_not_a_list_raises(self):
        complete_fn = FakeComplete([{"facts": "oops"}])
        conversation = _build_conversation()
        extractor = Extractor(complete_fn=complete_fn)
        with pytest.raises(ExtractionError):
            extractor.extract(conversation)


# ---------------------------------------------------------------------------
# Missing API key -> raise, never silently fall back (the fallback-rule
# behavioural change this story exists to make).
# ---------------------------------------------------------------------------


class TestMissingApiKeyRaisesRatherThanSilentlyFallingBack:
    def test_default_construction_with_no_key_and_no_dry_run_raises(self, tmp_path, monkeypatch):
        # Exercises the REAL default complete_fn (llm.complete) end-to-end —
        # not a mock of it — with every key source blanked out, isolated
        # from the real repo .env / data/llm_cache/ exactly like
        # tests/test_llm.py's own isolate_llm_module fixture does. Proves
        # the actual wiring raises rather than silently falling back to the
        # rule-based path.
        monkeypatch.setattr(llm, "CACHE_DIR", tmp_path / "llm_cache")
        monkeypatch.setattr(llm, "_ledger", None)
        monkeypatch.setattr(llm, "_openai_client", None)
        monkeypatch.setattr(llm, "_google_client", None)
        monkeypatch.setattr(llm, "_DEFAULT_DOTENV_PATH", tmp_path / "nonexistent.env")
        for name in ("OPENAI_API_KEY", "OPEN_AI_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY"):
            monkeypatch.delenv(name, raising=False)

        extractor = Extractor()  # real llm.complete, no injection, no dry_run
        assert extractor.kind == "llm"
        conversation = _build_conversation()
        with pytest.raises(llm.MissingAPIKeyError):
            extractor.extract(conversation)

    def test_injected_missing_key_error_propagates_unwrapped_not_swallowed(self):
        # Unit-level companion to the integration test above: extract.py's
        # own error-handling must not catch-and-degrade this specific
        # error into an empty facts list or a generic ExtractionError.
        complete_fn = FakeComplete(
            [llm.MissingAPIKeyError("OpenAI API key not found. Checked (in order): ...")]
        )
        extractor = Extractor(complete_fn=complete_fn)
        conversation = _build_conversation()
        with pytest.raises(llm.MissingAPIKeyError):
            extractor.extract(conversation)


# ---------------------------------------------------------------------------
# Cost / token surfacing (story requirement: "surface cost_usd and token
# counts from each response so the caller can report real spend").
# ---------------------------------------------------------------------------


class TestCostAndTokenSurfacing:
    def test_totals_accumulate_from_the_seam_response(self):
        complete_fn = FakeComplete([_ALL_SIX_TAGS_PAYLOAD])
        extractor = Extractor(complete_fn=complete_fn)
        conversation = _build_conversation()
        extractor.extract(conversation)
        # One admission -> one seam call -> FakeComplete's fixed per-call values.
        assert extractor.total_cost_usd == pytest.approx(0.0001)
        assert extractor.total_prompt_tokens == 10
        assert extractor.total_completion_tokens == 5

    def test_dry_run_never_incurs_cost_or_tokens(self):
        extractor = Extractor(dry_run=True)
        conversation = _build_conversation()
        extractor.extract(conversation)
        assert extractor.total_cost_usd == 0.0
        assert extractor.total_prompt_tokens == 0
        assert extractor.total_completion_tokens == 0


# ---------------------------------------------------------------------------
# Closed-vocabulary predicate drop (candidate that fails canonicalization)
# ---------------------------------------------------------------------------


class TestUncanonicalizablePredicateIsDropped:
    def test_off_vocabulary_predicate_phrase_is_dropped_not_invented(self):
        payload = {
            "facts": [
                _candidate(
                    predicate_phrase="enjoys hiking with",
                    object_name="hiking",
                    object_type="Hobby",
                    assertion="present",
                    turn_ids=[8],
                )
            ]
        }
        complete_fn = FakeComplete([payload])
        conversation = _build_conversation()
        extractor = Extractor(complete_fn=complete_fn)
        facts = extractor.extract(conversation)
        assert facts == []
        assert any(
            entry.decision == "dropped_no_predicate" for entry in extractor.assertion_log
        )

    def test_empty_turn_ids_is_dropped_not_a_validation_crash(self):
        payload = {
            "facts": [
                _candidate(
                    predicate_phrase="is prescribed",
                    object_name="metformin",
                    object_type="Medication",
                    assertion="present",
                    turn_ids=[],
                )
            ]
        }
        complete_fn = FakeComplete([payload])
        conversation = _build_conversation()
        extractor = Extractor(complete_fn=complete_fn)
        facts = extractor.extract(conversation)
        assert facts == []


# ---------------------------------------------------------------------------
# CURRENT_DOSAGE_OF subject convention (medication, not patient)
# ---------------------------------------------------------------------------


class TestCurrentDosageOfSubjectConvention:
    def test_subject_is_the_medication_not_the_patient(self):
        payload = {
            "facts": [
                _candidate(
                    predicate_phrase="current dosage of",
                    object_name="60mg",
                    object_type="Dosage",
                    assertion="present",
                    turn_ids=[1],
                    subject_name="metformin",
                    subject_type="Medication",
                )
            ]
        }
        complete_fn = FakeComplete([payload])
        conversation = _build_conversation()
        extractor = Extractor(complete_fn=complete_fn)
        (fact,) = extractor.extract(conversation)
        assert fact.predicate == "CURRENT_DOSAGE_OF"
        assert fact.subject.name == "metformin"
        assert fact.subject.type == "Medication"
        assert fact.object.name == "60mg"


# ---------------------------------------------------------------------------
# extract_facts() convenience entry point + admission scoping
# ---------------------------------------------------------------------------


class TestExtractFactsEntryPoint:
    def test_extract_facts_uses_injected_extractor(self):
        complete_fn = FakeComplete([_ALL_SIX_TAGS_PAYLOAD])
        extractor = Extractor(complete_fn=complete_fn)
        conversation = _build_conversation()
        facts = extract_facts(conversation, extractor=extractor)
        assert len(facts) == 3

    def test_admission_argument_scopes_to_one_admission(self):
        complete_fn = FakeComplete([_ALL_SIX_TAGS_PAYLOAD])
        extractor = Extractor(complete_fn=complete_fn)
        conversation = _build_conversation()
        facts = extract_facts(conversation, conversation.admissions[0], extractor=extractor)
        assert len(facts) == 3
        assert all(f.session_id == HADM_ID for f in facts)


# ---------------------------------------------------------------------------
# dry_run — runnable without any provider key, zero network calls
# (ENVIRONMENT note / the fallback rule's opt-in stub path)
# ---------------------------------------------------------------------------


class TestDryRunRunsWithoutApiKeyOrNetworkCalls:
    def test_dry_run_never_calls_complete_fn(self):
        # Even when a complete_fn IS supplied, dry_run must never reach it —
        # proves routing, not just the absence of a default client.
        never_call = FakeComplete([RuntimeError("dry_run must never reach complete_fn")])
        extractor = Extractor(dry_run=True, complete_fn=never_call)
        assert extractor.kind == "rule-based"
        conversation = _build_conversation()
        facts = extractor.extract(conversation)  # must not raise / hang
        assert isinstance(facts, list)
        assert never_call.calls == []
        for f in facts:
            assert f.polarity in ("asserted", "negated")
            assert f.predicate in PREDICATES

    def test_rule_based_catches_the_explicit_negation(self):
        extractor = Extractor(dry_run=True)
        conversation = _build_conversation()
        facts = extractor.extract(conversation)
        allergy_facts = [f for f in facts if f.predicate == "HAS_ALLERGY_TO"]
        assert allergy_facts, "rule-based path should catch the penicillin allergy mention"
        assert allergy_facts[0].polarity == "negated"

    def test_rule_based_drops_the_family_history_mention(self):
        extractor = Extractor(dry_run=True)
        conversation = _build_conversation()
        facts = extractor.extract(conversation)
        # Turn 6 ("My mother has diabetes too") must not produce a fact —
        # note turn 1 ("...for your diabetes") legitimately DOES produce a
        # patient-attached HAS_CONDITION diabetes fact; the assertion here
        # is scoped to turn 6 specifically, not "no diabetes fact anywhere".
        assert not any(f.object.name == "diabetes" and 6 in f.turn_ids for f in facts)
        skip_log = [
            e
            for e in extractor.assertion_log
            if e.decision == "dropped_by_rule" and e.turn_ids == [6]
        ]
        assert skip_log, "turn 6's family-history mention should be logged as a deliberate skip"
        assert skip_log[0].i2b2_tag == "not-associated-with-patient"

    def test_env_flag_forces_dry_run_by_default(self, monkeypatch):
        monkeypatch.setenv("MEDMEMGRAPH_EXTRACT_DRY_RUN", "1")
        extractor = Extractor()  # no explicit dry_run= kwarg
        assert extractor.dry_run is True
        assert extractor.kind == "rule-based"

    def test_explicit_dry_run_kwarg_overrides_absent_env_flag(self, monkeypatch):
        monkeypatch.delenv("MEDMEMGRAPH_EXTRACT_DRY_RUN", raising=False)
        extractor = Extractor(dry_run=True)
        assert extractor.dry_run is True


# ---------------------------------------------------------------------------
# 2026-08-16 PRN/vocabulary-gap bug fix regression tests.
#
# `samples/extract-eyeball-10056223.md` ("What it gets wrong", items 1-2)
# documented two real, reproducible extraction bugs from a real-inference
# run against subject 10056223: (1) PRN ("as needed") medication language
# was tagged `conditional` and silently dropped, losing a real active
# prescription off the medication list; (2) `predicate_phrase='had a fall'`
# failed to canonicalize onto `PREDICATES` and was dropped as
# `dropped_no_predicate`. Both fixes are prompt/vocabulary changes this
# module makes; these tests prove the PIPELINE's handling of the *fixed*
# model output (scripted `complete_fn`, zero network cost — this story
# explicitly does not re-run real inference), using the exact real quotes
# from the eyeball artifact as `evidence_quote`/turn text so the regression
# is traceable back to the bug report.
# ---------------------------------------------------------------------------

_PRN_TURNS = [
    {
        "turn_number": 1,
        "time": "2120-08-13 14:18:00",
        "speaker": "Doctor",
        "text": "Plus the new diuretics and oxycodone for pain as needed.",
    },
    {
        "turn_number": 2,
        "time": "2120-08-14 09:10:00",
        "speaker": "Doctor",
        "text": (
            "You'll continue ciprofloxacin for five more days to finish the "
            "course. Prochlorperazine for nausea if it comes back, and "
            "oxycodone for pain as needed."
        ),
    },
    {
        "turn_number": 3,
        "time": "2120-08-15 08:00:00",
        "speaker": "Doctor",
        "text": "We're giving you oxycodone as needed. Take one every four hours if needed.",
    },
    {
        "turn_number": 4,
        "time": "2120-08-16 10:30:00",
        "speaker": "Doctor",
        "text": "We heard a little wheeze in your lungs. We'll give you Tessalon Pearls to help with the cough.",
    },
    {
        "turn_number": 5,
        "time": "2120-08-17 07:45:00",
        "speaker": "Doctor",
        "text": "If the fever returns, restart the antibiotic.",
    },
    {
        "turn_number": 6,
        "time": "2120-08-17 07:47:00",
        "speaker": "Doctor",
        "text": "If this were bacterial we would start antibiotics, but it looks viral.",
    },
    {
        "turn_number": 7,
        "time": "2120-08-17 07:50:00",
        "speaker": "Patient",
        "text": "My mother has diabetes too, runs in the family.",
    },
    {
        "turn_number": 8,
        "time": "2120-08-17 08:00:00",
        "speaker": "Doctor",
        "text": "I understand you had a fall this morning and have been feeling very nauseous.",
    },
]

_PRN_PATIENT_ID = "patient-0099"
_PRN_HADM_ID = "admission-1010"


def _build_prn_conversation() -> Conversation:
    admission = Admission(
        hadm_id=_PRN_HADM_ID,
        admission_start=_PRN_TURNS[0]["time"],
        admission_end=_PRN_TURNS[-1]["time"],
        conversation_lines=tuple(_PRN_TURNS),
    )
    return Conversation(
        subject_id=_PRN_PATIENT_ID,
        processed_hadm_ids=(_PRN_HADM_ID,),
        admissions=(admission,),
    )


_PRN_BUGFIX_PAYLOAD = {
    "facts": [
        # Bug 1: "...oxycodone for pain as needed." -> present, prn=true
        # (was: conditional, dropped).
        _candidate(
            predicate_phrase="is prescribed",
            object_name="oxycodone",
            object_type="Medication",
            assertion="present",
            prn=True,
            turn_ids=[1],
        ),
        # Bug 1: "Prochlorperazine for nausea if it comes back..." -> the
        # "if" names the symptom dosing-trigger, not a not-yet-prescribed
        # contingency -> present, prn=true (was: conditional, dropped).
        _candidate(
            predicate_phrase="is prescribed",
            object_name="prochlorperazine",
            object_type="Medication",
            assertion="present",
            prn=True,
            turn_ids=[2],
        ),
        # Bug 1: "...and oxycodone for pain as needed." (same turn, second
        # medication) -> present, prn=true.
        _candidate(
            predicate_phrase="is prescribed",
            object_name="oxycodone",
            object_type="Medication",
            assertion="present",
            prn=True,
            turn_ids=[2],
        ),
        # Bug 1: "We're giving you oxycodone as needed..." -> present, prn=true.
        _candidate(
            predicate_phrase="is prescribed",
            object_name="oxycodone",
            object_type="Medication",
            assertion="present",
            prn=True,
            turn_ids=[3],
        ),
        # "We'll give you Tessalon Pearls..." — NO contingency at all, not
        # PRN language either -> present, prn=false. This is the "unambiguous
        # false positive" the story flags: it must never have dropped.
        _candidate(
            predicate_phrase="is prescribed",
            object_name="Tessalon Pearls",
            object_type="Medication",
            assertion="present",
            prn=False,
            turn_ids=[4],
        ),
        # Genuine conditional (not PRN): the antibiotic is not currently
        # being taken; it only (re)starts if the fever returns. Must still
        # drop.
        _candidate(
            predicate_phrase="is prescribed",
            object_name="the antibiotic",
            object_type="Medication",
            assertion="conditional",
            prn=False,
            turn_ids=[5],
        ),
        # Genuine hypothetical (counterfactual). Must still drop.
        _candidate(
            predicate_phrase="is prescribed",
            object_name="antibiotics",
            object_type="Medication",
            assertion="hypothetical",
            prn=False,
            turn_ids=[6],
        ),
        # Family-history mention. Must still drop and never attach to the
        # patient.
        _candidate(
            predicate_phrase="diagnosed with",
            object_name="diabetes",
            object_type="Condition",
            assertion="not-associated-with-patient",
            prn=False,
            turn_ids=[7],
        ),
        # Bug 2: "had a fall this morning" -> HAD_INCIDENT (was:
        # dropped_no_predicate, predicate_phrase='had a fall' unresolvable).
        _candidate(
            predicate_phrase="had a fall",
            object_name="fall",
            object_type="Condition",
            assertion="present",
            prn=False,
            turn_ids=[8],
        ),
    ]
}


class TestPRNMedicationsEmitAsPresent:
    """Bug 1 — PRN/"as needed" medication language must map to `present`
    (emit) with `prn=true` carried on the side log, not `conditional`
    (silently dropped)."""

    @pytest.fixture()
    def extractor_and_facts(self):
        complete_fn = FakeComplete([_PRN_BUGFIX_PAYLOAD])
        extractor = Extractor(complete_fn=complete_fn)
        conversation = _build_prn_conversation()
        facts = extractor.extract(conversation)
        return extractor, facts

    @pytest.mark.parametrize("turn_id,object_name", [(1, "oxycodone"), (2, "oxycodone"), (3, "oxycodone")])
    def test_every_as_needed_oxycodone_mention_emits(self, extractor_and_facts, turn_id, object_name):
        _, facts = extractor_and_facts
        matches = [
            f for f in facts if f.predicate == "TAKES_MEDICATION" and f.object.name == object_name and turn_id in f.turn_ids
        ]
        assert len(matches) == 1, f"expected exactly one emitted oxycodone fact for turn {turn_id}"
        assert matches[0].polarity == "asserted"

    def test_prochlorperazine_with_if_it_comes_back_still_emits_present(self, extractor_and_facts):
        _, facts = extractor_and_facts
        matches = [f for f in facts if f.object.name == "prochlorperazine"]
        assert len(matches) == 1
        assert matches[0].predicate == "TAKES_MEDICATION"
        assert matches[0].polarity == "asserted"

    def test_prn_facts_are_logged_with_prn_true(self, extractor_and_facts):
        extractor, _ = extractor_and_facts
        prn_objects = {"oxycodone", "prochlorperazine"}
        prn_entries = [
            e for e in extractor.assertion_log if e.decision == "emitted" and e.object_name in prn_objects
        ]
        assert len(prn_entries) == 4  # 3x oxycodone (turns 1/2/3) + prochlorperazine (turn 2)
        assert all(e.prn is True for e in prn_entries)

    def test_tessalon_pearls_emits_with_no_contingency_and_prn_false(self, extractor_and_facts):
        extractor, facts = extractor_and_facts
        matches = [f for f in facts if f.object.name == "Tessalon Pearls"]
        assert len(matches) == 1, "a plain active prescription with no contingency must never drop"
        assert matches[0].predicate == "TAKES_MEDICATION"
        assert matches[0].polarity == "asserted"
        log_entry = next(
            e for e in extractor.assertion_log if e.decision == "emitted" and e.object_name == "Tessalon Pearls"
        )
        assert log_entry.prn is False

    def test_total_prn_and_plain_present_medication_facts_emitted(self, extractor_and_facts):
        _, facts = extractor_and_facts
        med_facts = [f for f in facts if f.predicate == "TAKES_MEDICATION"]
        # 3x oxycodone (turns 1,2,3) + prochlorperazine (turn 2) + Tessalon Pearls (turn 4) = 5.
        assert len(med_facts) == 5


class TestGenuineConditionalAndHypotheticalStillDrop:
    """The `conditional`/`hypothetical` skip rule (decisions/002) is
    narrowed for PRN language, not removed — a genuine future-contingency
    conditional and a genuine counterfactual hypothetical must still be
    dropped, never emitted."""

    @pytest.fixture()
    def extractor_and_facts(self):
        complete_fn = FakeComplete([_PRN_BUGFIX_PAYLOAD])
        extractor = Extractor(complete_fn=complete_fn)
        conversation = _build_prn_conversation()
        facts = extractor.extract(conversation)
        return extractor, facts

    def test_if_the_fever_returns_restart_the_antibiotic_is_not_emitted(self, extractor_and_facts):
        _, facts = extractor_and_facts
        assert not any(f.object.name == "the antibiotic" for f in facts)

    def test_conditional_drop_is_logged_by_rule(self, extractor_and_facts):
        extractor, _ = extractor_and_facts
        entry = next(e for e in extractor.assertion_log if e.turn_ids == [5])
        assert entry.decision == "dropped_by_rule"
        assert entry.i2b2_tag == "conditional"

    def test_if_this_were_bacterial_hypothetical_is_not_emitted(self, extractor_and_facts):
        _, facts = extractor_and_facts
        assert not any(f.turn_ids == [6] for f in facts)

    def test_hypothetical_drop_is_logged_by_rule(self, extractor_and_facts):
        extractor, _ = extractor_and_facts
        entry = next(e for e in extractor.assertion_log if e.turn_ids == [6])
        assert entry.decision == "dropped_by_rule"
        assert entry.i2b2_tag == "hypothetical"


class TestFamilyHistoryStillNeverAttachesToPatient:
    @pytest.fixture()
    def extractor_and_facts(self):
        complete_fn = FakeComplete([_PRN_BUGFIX_PAYLOAD])
        extractor = Extractor(complete_fn=complete_fn)
        conversation = _build_prn_conversation()
        facts = extractor.extract(conversation)
        return extractor, facts

    def test_mothers_diabetes_never_attaches_to_the_patient(self, extractor_and_facts):
        _, facts = extractor_and_facts
        assert not any(f.object.name == "diabetes" for f in facts)
        assert not any(
            f.predicate == "HAS_CONDITION" and f.subject.name == _PRN_PATIENT_ID for f in facts
        )

    def test_family_history_drop_is_logged_by_rule(self, extractor_and_facts):
        extractor, _ = extractor_and_facts
        entry = next(e for e in extractor.assertion_log if e.turn_ids == [7])
        assert entry.decision == "dropped_by_rule"
        assert entry.i2b2_tag == "not-associated-with-patient"


class TestHadAFallNowCanonicalizes:
    """Bug 2 — 'had a fall' must canonicalize onto the new `HAD_INCIDENT`
    predicate instead of dropping as `dropped_no_predicate`."""

    @pytest.fixture()
    def extractor_and_facts(self):
        complete_fn = FakeComplete([_PRN_BUGFIX_PAYLOAD])
        extractor = Extractor(complete_fn=complete_fn)
        conversation = _build_prn_conversation()
        facts = extractor.extract(conversation)
        return extractor, facts

    def test_had_a_fall_emits_had_incident(self, extractor_and_facts):
        _, facts = extractor_and_facts
        matches = [f for f in facts if f.turn_ids == [8]]
        assert len(matches) == 1
        assert matches[0].predicate == "HAD_INCIDENT"
        assert matches[0].object.name == "fall"
        assert matches[0].subject.name == _PRN_PATIENT_ID

    def test_had_a_fall_is_not_logged_as_dropped_no_predicate(self, extractor_and_facts):
        extractor, _ = extractor_and_facts
        assert not any(
            e.decision == "dropped_no_predicate" and e.turn_ids == [8] for e in extractor.assertion_log
        )


class TestDecisionCountsSurfacesDroppedNoPredicate:
    """`Extractor.decision_counts()` (the extraction-stats visibility fix)
    always reports `dropped_no_predicate`, zero-filled, even when nothing
    dropped for that reason this run — so a vocabulary gap can never be
    silently absent from the stats."""

    def test_decision_counts_zero_fills_all_known_kinds(self):
        complete_fn = FakeComplete([_PRN_BUGFIX_PAYLOAD])
        extractor = Extractor(complete_fn=complete_fn)
        conversation = _build_prn_conversation()
        extractor.extract(conversation)
        counts = extractor.decision_counts()
        assert set(counts) == {"emitted", "dropped_by_rule", "dropped_no_predicate", "dropped_invalid"}
        assert counts["dropped_no_predicate"] == 0  # bug 2 is fixed: nothing drops for this reason here
        assert counts["dropped_by_rule"] == 3  # conditional + hypothetical + not-associated-with-patient
        assert counts["emitted"] == 6  # 5 medication facts + 1 HAD_INCIDENT fact
        assert sum(counts.values()) == len(extractor.assertion_log)

    def test_decision_counts_reports_a_real_dropped_no_predicate_when_it_happens(self):
        payload = {
            "facts": [
                _candidate(
                    predicate_phrase="enjoys hiking with",
                    object_name="hiking",
                    object_type="Hobby",
                    assertion="present",
                    turn_ids=[1],
                )
            ]
        }
        complete_fn = FakeComplete([payload])
        extractor = Extractor(complete_fn=complete_fn)
        conversation = _build_conversation()
        extractor.extract(conversation)
        assert extractor.decision_counts()["dropped_no_predicate"] == 1
