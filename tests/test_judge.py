"""tests/test_judge.py — coverage for `medmemgraph.eval.judge` (the LLM
judge, rewired off `anthropic` onto the `medmemgraph.llm` seam).

Everything here runs offline. `force_fallback=True` tests exercise the
deterministic token-overlap / keyword-abstention judge (zero network, zero
`llm.py` involvement at all). `force_fallback=False` ("real llm path")
tests replace `llm.py`'s lazy Google-client singleton with an in-process
fake object via `monkeypatch.setattr(llm, "_get_google_client", ...)` —
the same pattern `tests/test_llm.py` establishes for testing consumers of
the seam without a real network call — so the *whole* real code path
(schema validation, token accounting, provider routing) is exercised
honestly, not mocked away. No test sets a real `GOOGLE_API_KEY` /
`GEMINI_API_KEY` anywhere, and the autouse `isolate_llm_module` fixture
points `llm.CACHE_DIR` at a per-test `tmp_path` and resets the module's
mutable singletons, so tests never touch the real repo `data/llm_cache/`
or read the real `.env` (which, in this repo, really does carry a live
`GOOGLE_API_KEY` — without this isolation, a "missing key" test would
silently pick that up and attempt a real network call).
"""

from __future__ import annotations

import json

import pytest

from medmemgraph import llm
from medmemgraph.eval.judge import (
    DEFAULT_JUDGE_MODEL,
    Judge,
    _looks_like_abstention,
    _token_overlap_score,
)

# ---------------------------------------------------------------------------
# Fakes — Google-shaped, matching the attribute shapes llm.py reads off a
# real `google.genai` response (mirrors tests/test_llm.py's own fakes,
# duplicated here rather than imported, per this project's "each test file
# fully self-contained" convention — see eval/baselines/dense.py's
# `_TimedRetriever` docstring for the same reasoning stated once).
# ---------------------------------------------------------------------------


class _FakeGoogleUsageMetadata:
    def __init__(self, prompt_token_count: int, candidates_token_count: int) -> None:
        self.prompt_token_count = prompt_token_count
        self.candidates_token_count = candidates_token_count


class _FakeGoogleResponse:
    def __init__(self, text: str, prompt_tokens: int = 8, completion_tokens: int = 4) -> None:
        self.text = text
        self.usage_metadata = _FakeGoogleUsageMetadata(prompt_tokens, completion_tokens)


class _FakeGoogleModels:
    def __init__(self, responses: list) -> None:
        self.responses = responses
        self.calls: list[dict] = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        idx = min(len(self.calls) - 1, len(self.responses) - 1)
        item = self.responses[idx]
        if isinstance(item, BaseException):
            raise item
        return _FakeGoogleResponse(item)


class FakeGoogleClient:
    def __init__(self, responses: list | None = None) -> None:
        self.models = _FakeGoogleModels(responses or ["{}"])


def _judge_payload(text: str) -> str:
    return json.dumps(json.loads(text))  # round-trip just to fail fast on a malformed fixture


# ---------------------------------------------------------------------------
# Isolation fixture — fresh cache dir, ledger, and client singletons; the
# real repo .env / data/llm_cache/ are never touched by this file.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def isolate_llm_module(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path / "llm_cache")
    monkeypatch.setattr(llm, "_ledger", None)
    monkeypatch.setattr(llm, "_openai_client", None)
    monkeypatch.setattr(llm, "_google_client", None)
    monkeypatch.delenv("MEDMEMGRAPH_MAX_USD", raising=False)
    monkeypatch.setattr(llm, "_sleep", lambda seconds: None)
    yield


def _no_network_google(*_a, **_kw):
    raise AssertionError("this code path should never touch the Google client getter")


# ---------------------------------------------------------------------------
# 1. Cross-family judge — the model choice is deliberate, not accidental
# ---------------------------------------------------------------------------


class TestCrossFamilyJudge:
    def test_default_judge_model_routes_to_google(self):
        assert llm._provider_for_model(DEFAULT_JUDGE_MODEL) == "google"

    def test_judge_model_is_a_different_family_from_the_answer_model(self):
        """literature/17: self-preference bias is measured specifically
        when judge and candidate share a model family — this is the one
        invariant that must never silently regress (see judge.py module
        docstring, "Cross-family judge")."""
        from medmemgraph.eval.reader import DEFAULT_READER_MODEL

        assert llm._provider_for_model(DEFAULT_JUDGE_MODEL) != llm._provider_for_model(DEFAULT_READER_MODEL)

    def test_default_judge_model_is_llms_judge_model_constant(self):
        assert DEFAULT_JUDGE_MODEL == llm.JUDGE_MODEL


# ---------------------------------------------------------------------------
# 2. force_fallback=True — deterministic, offline, dry-run-only
# ---------------------------------------------------------------------------


class TestTokenOverlapFallback:
    def test_kind_is_token_overlap_under_force_fallback(self):
        assert Judge(force_fallback=True).kind == "token-overlap"

    def test_kind_is_llm_by_default(self):
        assert Judge(force_fallback=False).kind == "llm"
        assert Judge().kind == "llm"  # real inference is the default, not the fallback

    def test_answerable_correct_above_threshold(self):
        judge = Judge(force_fallback=True)
        verdict = judge.judge(
            question="Why did the patient have chest pain?",
            gold_answer="unclear etiology, no infection on workup",
            system_answer="unclear etiology, no infection was found on workup",
            question_type="medical_reasoning",
        )
        assert verdict.correct is True
        assert verdict.mode == "answerable"
        assert verdict.judge_kind == "token-overlap"
        assert "dry_run" in verdict.reason  # clearly labelled per story requirement

    def test_answerable_incorrect_below_threshold(self):
        judge = Judge(force_fallback=True)
        verdict = judge.judge(
            question="What medication does the patient take?",
            gold_answer="metformin 500mg twice daily for type 2 diabetes",
            system_answer="the patient enjoys long walks",
            question_type="medical_reasoning",
        )
        assert verdict.correct is False

    def test_adversarial_routes_to_abstention_and_declined_is_correct(self):
        judge = Judge(force_fallback=True)
        verdict = judge.judge(
            question="What is the patient allergic to?",
            gold_answer="the question is not answerable",
            system_answer="I don't have enough information to determine that.",
            question_type="adversarial",
        )
        assert verdict.mode == "abstention"
        assert verdict.correct is True
        assert verdict.judge_kind == "token-overlap"

    def test_adversarial_confabulated_answer_is_incorrect(self):
        judge = Judge(force_fallback=True)
        verdict = judge.judge(
            question="What is the patient allergic to?",
            gold_answer="the question is not answerable",
            system_answer="The patient is allergic to penicillin.",
            question_type="adversarial",
        )
        assert verdict.mode == "abstention"
        assert verdict.correct is False

    def test_fallback_never_touches_the_llm_seam(self, monkeypatch):
        monkeypatch.setattr(llm, "_get_google_client", _no_network_google)
        judge = Judge(force_fallback=True)
        judge.judge(question="q", gold_answer="a", system_answer="b", question_type="medical_reasoning")
        judge.judge(question="q", gold_answer="the question is not answerable", system_answer="I don't know", question_type="adversarial")
        # no AssertionError from _no_network_google means neither call reached llm.py


class TestFallbackHelpers:
    def test_token_overlap_score_full_match(self):
        assert _token_overlap_score("metformin 500mg", "metformin 500mg twice daily") == 1.0

    def test_token_overlap_score_empty_gold_is_zero(self):
        assert _token_overlap_score("the a of", "anything") == 0.0

    def test_looks_like_abstention_matches_known_phrase(self):
        assert _looks_like_abstention("I don't have enough information to answer that.") is True

    def test_looks_like_abstention_false_for_substantive_answer(self):
        assert _looks_like_abstention("The patient takes metformin 500mg.") is False


# ---------------------------------------------------------------------------
# 3. force_fallback=False — the real path, routed through llm.py
# ---------------------------------------------------------------------------


class TestRealLLMPath:
    def test_answerable_llm_call_returns_structured_verdict(self, monkeypatch):
        fake = FakeGoogleClient(responses=[json.dumps({"correct": True, "reason": "criterion 1 satisfied"})])
        monkeypatch.setattr(llm, "_get_google_client", lambda: fake)
        judge = Judge(force_fallback=False)
        verdict = judge.judge(
            question="What dose of metformin?",
            gold_answer="500mg",
            system_answer="500mg twice daily",
            question_type="medical_reasoning",
        )
        assert verdict.correct is True
        assert verdict.mode == "answerable"
        assert verdict.judge_kind == "llm"
        assert verdict.reason == "criterion 1 satisfied"
        assert len(fake.models.calls) == 1

    def test_abstention_llm_call_returns_structured_verdict(self, monkeypatch):
        fake = FakeGoogleClient(responses=[json.dumps({"declined": True, "reason": "criterion 1 satisfied"})])
        monkeypatch.setattr(llm, "_get_google_client", lambda: fake)
        judge = Judge(force_fallback=False)
        verdict = judge.judge(
            question="What is the patient allergic to?",
            gold_answer="the question is not answerable",
            system_answer="I don't have enough information to determine that.",
            question_type="adversarial",
        )
        assert verdict.correct is True
        assert verdict.mode == "abstention"
        assert verdict.judge_kind == "llm"

    def test_llm_call_uses_the_configured_model(self, monkeypatch):
        fake = FakeGoogleClient(responses=[json.dumps({"correct": False, "reason": "no match"})])
        monkeypatch.setattr(llm, "_get_google_client", lambda: fake)
        judge = Judge(model="gemini-3.5-flash-lite", force_fallback=False)
        judge.judge(question="q", gold_answer="a", system_answer="b", question_type="medical_reasoning")
        assert fake.models.calls[0]["model"] == "gemini-3.5-flash-lite"

    def test_llm_call_requests_temperature_zero(self, monkeypatch):
        fake = FakeGoogleClient(responses=[json.dumps({"correct": False, "reason": "no match"})])
        monkeypatch.setattr(llm, "_get_google_client", lambda: fake)
        Judge(force_fallback=False).judge(
            question="q", gold_answer="a", system_answer="b", question_type="medical_reasoning"
        )
        assert fake.models.calls[0]["config"].temperature == 0.0

    def test_llm_call_never_touches_the_token_overlap_fallback(self, monkeypatch):
        """The real path must not silently blend in the deterministic
        heuristic — if the LLM call is wired correctly, `_token_overlap_score`
        should never be invoked for a force_fallback=False judge."""
        import medmemgraph.eval.judge as judge_module

        called = {"hit": False}
        original = judge_module._token_overlap_score

        def _tripwire(*a, **kw):
            called["hit"] = True
            return original(*a, **kw)

        monkeypatch.setattr(judge_module, "_token_overlap_score", _tripwire)
        fake = FakeGoogleClient(responses=[json.dumps({"correct": True, "reason": "ok"})])
        monkeypatch.setattr(llm, "_get_google_client", lambda: fake)
        Judge(force_fallback=False).judge(
            question="q", gold_answer="a", system_answer="a", question_type="medical_reasoning"
        )
        assert called["hit"] is False


# ---------------------------------------------------------------------------
# 4. Missing key + not dry_run raises, rather than degrading (story's
#    explicit hard requirement, mirrored from the reader-side test)
# ---------------------------------------------------------------------------


class TestMissingKeyRaisesRatherThanDegrading:
    def test_no_key_and_force_fallback_false_raises_missing_api_key_error(self, monkeypatch):
        def _raise_missing(**_kw):
            raise llm.MissingAPIKeyError("Google API key not found (simulated for test).")

        monkeypatch.setattr(llm, "resolve_google_key", _raise_missing)
        judge = Judge(force_fallback=False)
        with pytest.raises(llm.MissingAPIKeyError):
            judge.judge(
                question="q", gold_answer="a", system_answer="b", question_type="medical_reasoning"
            )

    def test_no_key_and_force_fallback_false_raises_for_adversarial_items_too(self, monkeypatch):
        def _raise_missing(**_kw):
            raise llm.MissingAPIKeyError("Google API key not found (simulated for test).")

        monkeypatch.setattr(llm, "resolve_google_key", _raise_missing)
        judge = Judge(force_fallback=False)
        with pytest.raises(llm.MissingAPIKeyError):
            judge.judge(
                question="q",
                gold_answer="the question is not answerable",
                system_answer="I don't know",
                question_type="adversarial",
            )

    def test_force_fallback_true_never_needs_a_key_even_if_resolution_would_fail(self, monkeypatch):
        """The inverse case, to prove the fallback path is genuinely
        key-independent: even with key resolution rigged to always fail,
        `force_fallback=True` must still succeed (it never calls
        resolve_google_key at all)."""

        def _raise_missing(**_kw):
            raise llm.MissingAPIKeyError("should never be called")

        monkeypatch.setattr(llm, "resolve_google_key", _raise_missing)
        judge = Judge(force_fallback=True)
        verdict = judge.judge(
            question="q", gold_answer="a b c", system_answer="a b c", question_type="medical_reasoning"
        )
        assert verdict.judge_kind == "token-overlap"


# ---------------------------------------------------------------------------
# 5. Judge-bias mitigations — rubric prompting and deterministic ordering
#    are actually present in the code, not just claimed in a docstring
# ---------------------------------------------------------------------------


class TestBiasMitigationsArePresent:
    def test_answerable_rubric_has_three_numbered_criteria_in_a_fixed_order(self, monkeypatch):
        import medmemgraph.eval.judge as judge_module

        assert judge_module._ANSWERABLE_JUDGE_SYSTEM.index("1. FACTUAL MATCH") < judge_module._ANSWERABLE_JUDGE_SYSTEM.index(
            "2. NO CONTRADICTION"
        ) < judge_module._ANSWERABLE_JUDGE_SYSTEM.index("3. NOT A REFUSAL")

    def test_abstention_rubric_has_two_numbered_criteria_in_a_fixed_order(self):
        import medmemgraph.eval.judge as judge_module

        assert judge_module._ABSTENTION_JUDGE_SYSTEM.index("1. DECLINATION") < judge_module._ABSTENTION_JUDGE_SYSTEM.index(
            "2. NO FABRICATION"
        )

    def test_answerable_and_abstention_rubrics_are_distinct_never_blended(self):
        import medmemgraph.eval.judge as judge_module

        assert judge_module._ANSWERABLE_JUDGE_SYSTEM != judge_module._ABSTENTION_JUDGE_SYSTEM
        assert "not answerable" in judge_module._ABSTENTION_JUDGE_SYSTEM
        assert "not answerable" not in judge_module._ANSWERABLE_JUDGE_SYSTEM

    def test_user_message_field_order_is_deterministic_for_answerable(self, monkeypatch):
        captured: dict = {}
        original_complete = llm.complete

        def _spy(prompt, **kwargs):
            captured["prompt"] = prompt
            captured["system"] = kwargs.get("system")
            return original_complete(prompt, **kwargs)

        fake = FakeGoogleClient(responses=[json.dumps({"correct": True, "reason": "ok"})])
        monkeypatch.setattr(llm, "_get_google_client", lambda: fake)
        monkeypatch.setattr(llm, "complete", _spy)

        Judge(force_fallback=False).judge(
            question="Q?", gold_answer="GOLD", system_answer="SYS", question_type="medical_reasoning"
        )
        prompt = captured["prompt"]
        assert prompt.index("Question:") < prompt.index("Gold reference answer:") < prompt.index("System's answer:")

    def test_user_message_field_order_is_deterministic_for_abstention(self, monkeypatch):
        captured: dict = {}
        original_complete = llm.complete

        def _spy(prompt, **kwargs):
            captured["prompt"] = prompt
            return original_complete(prompt, **kwargs)

        fake = FakeGoogleClient(responses=[json.dumps({"declined": True, "reason": "ok"})])
        monkeypatch.setattr(llm, "_get_google_client", lambda: fake)
        monkeypatch.setattr(llm, "complete", _spy)

        Judge(force_fallback=False).judge(
            question="Q?", gold_answer="the question is not answerable", system_answer="I don't know",
            question_type="adversarial",
        )
        prompt = captured["prompt"]
        assert prompt.index("Question:") < prompt.index("System's answer:")
        assert "Gold reference answer:" not in prompt  # nothing to match against for abstention items


# ---------------------------------------------------------------------------
# 6. Schema shape — structured verdict, not a regex over prose
# ---------------------------------------------------------------------------


class TestSchemaShape:
    def test_answerable_schema_requires_correct_and_reason_only(self):
        import medmemgraph.eval.judge as judge_module

        schema = judge_module._ANSWERABLE_SCHEMA
        assert set(schema["properties"]) == {"correct", "reason"}
        assert set(schema["required"]) == {"correct", "reason"}
        assert schema["additionalProperties"] is False

    def test_abstention_schema_requires_declined_and_reason_only(self):
        import medmemgraph.eval.judge as judge_module

        schema = judge_module._ABSTENTION_SCHEMA
        assert set(schema["properties"]) == {"declined", "reason"}
        assert set(schema["required"]) == {"declined", "reason"}
        assert schema["additionalProperties"] is False
