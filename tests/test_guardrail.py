"""tests/test_guardrail.py — coverage for `medmemgraph.eval.guardrail` (the
cite-or-abstain grounding guardrail).

Everything here is offline. Most tests replace `medmemgraph.llm.complete`
with an in-process, queue-based fake (`_queued_complete`, mirroring
`tests/test_ragas_metrics.py`'s identical helper — each test file in this
project is fully self-contained rather than importing test helpers across
files, per that module's own stated convention) that returns real
`llm.LLMResponse` objects built from caller-supplied already-parsed
payloads, in call order. The short-circuit / disabled tests instead
monkeypatch `llm.complete` with a trip-wire that raises `AssertionError` if
it is ever called at all — proving "zero calls," not just "a fast result."

The autouse `isolate_llm_module` fixture points `llm.CACHE_DIR` at a
per-test `tmp_path` and resets the seam's mutable singletons (same
convention `test_judge.py`/`test_ragas_metrics.py`/`test_reader.py`
establish), so no test here touches the real repo `data/llm_cache/` or
reads the real `.env`.
"""

from __future__ import annotations

import json

import pytest

from medmemgraph import llm
from medmemgraph.contracts import RetrieveItem
from medmemgraph.eval import guardrail as gr

# ---------------------------------------------------------------------------
# Isolation fixture — same pattern as test_judge.py / test_ragas_metrics.py.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def isolate_llm_module(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path / "llm_cache")
    monkeypatch.setattr(llm, "_ledger", None)
    monkeypatch.setattr(llm, "_openai_client", None)
    monkeypatch.setattr(llm, "_google_client", None)
    monkeypatch.delenv("MEDMEMGRAPH_MAX_USD", raising=False)
    yield


def _no_network(*_a, **_kw):
    raise AssertionError("this code path should never touch a real network/GPU client")


# ---------------------------------------------------------------------------
# Fake `llm.complete` — queue of already-parsed payloads, returned in order
# (mirrors test_ragas_metrics.py's `_queued_complete` exactly).
# ---------------------------------------------------------------------------


def _queued_complete(payloads: list[dict]):
    it = iter(payloads)
    calls: list[dict] = []

    def _fake(
        prompt,
        *,
        model=None,
        schema=None,
        system=None,
        max_tokens=1024,
        temperature=0.0,
        dry_run=False,
        use_cache=True,
        max_retries=5,
    ):
        calls.append({"prompt": prompt, "system": system, "schema": schema, "model": model})
        try:
            parsed = next(it)
        except StopIteration as exc:
            raise AssertionError(
                f"fake llm.complete called more times ({len(calls)}) than payloads were queued "
                f"({len(payloads)}) -- prompt was: {prompt!r}"
            ) from exc
        return llm.LLMResponse(
            text=json.dumps(parsed), parsed=parsed, model=model or "fake-model",
            prompt_tokens=10, completion_tokens=5, cost_usd=0.0002,
            latency_ms=1.0, cached=False, attempts=1,
        )

    return _fake, calls


def _item(text: str, session_id: str = "H1", turn_ids=None, score: float = 0.9, channel: str = "vector") -> RetrieveItem:
    return RetrieveItem(text=text, session_id=session_id, turn_ids=turn_ids or [1], score=score, channel=channel)


# ---------------------------------------------------------------------------
# 1. check_grounding — a fully-cited answer passes
# ---------------------------------------------------------------------------


class TestFullyCitedAnswerPasses:
    def test_fully_cited_answer_is_grounded_with_citations(self, monkeypatch):
        items = [
            _item("2026-01-01: patient takes metformin 500mg twice daily.", session_id="H1", turn_ids=[4]),
            _item("2026-01-02: patient diagnosed with type 2 diabetes.", session_id="H1", turn_ids=[7]),
        ]
        fake, calls = _queued_complete(
            [
                {
                    "claims": [
                        {"text": "Patient takes metformin 500mg twice daily.", "status": "supported", "cited_item": 1, "reason": "item 1 states this"},
                        {"text": "Patient has type 2 diabetes.", "status": "supported", "cited_item": 2, "reason": "item 2 states this"},
                    ],
                    "confidence": 0.95,
                }
            ]
        )
        monkeypatch.setattr(llm, "complete", fake)
        report = gr.check_grounding(
            "The patient takes metformin 500mg twice daily for type 2 diabetes.",
            items,
            question="What medication does the patient take?",
            dry_run=False,
            use_cache=False,
        )
        assert report.is_grounded is True
        assert report.uncited_claims == []
        assert report.unsupported_claims == []
        assert report.confidence == pytest.approx(0.95)
        assert len(report.citations) == 2
        assert items[0] in report.citations and items[1] in report.citations
        assert report.n_judge_calls == 1
        assert report.cost_usd == pytest.approx(0.0002)
        assert len(calls) == 1, "must be exactly ONE judge call per answer (story cost discipline)"

    def test_general_knowledge_claim_is_uncited_not_unsupported_and_still_grounded(self, monkeypatch):
        """An "uncited" claim (general medical framing, no patient-specific
        fact) must not fail is_grounded -- only unsupported_claims does."""
        items = [_item("2026-01-01: patient takes metformin 500mg.", turn_ids=[4])]
        fake, calls = _queued_complete(
            [
                {
                    "claims": [
                        {"text": "Patient takes metformin 500mg.", "status": "supported", "cited_item": 1, "reason": "evidence states this"},
                        {"text": "Metformin is commonly prescribed for type 2 diabetes.", "status": "uncited", "cited_item": -1, "reason": "general knowledge, not patient-specific"},
                    ],
                    "confidence": 0.9,
                }
            ]
        )
        monkeypatch.setattr(llm, "complete", fake)
        report = gr.check_grounding("Patient takes metformin 500mg, commonly prescribed for type 2 diabetes.", items, dry_run=False, use_cache=False)
        assert report.is_grounded is True
        assert report.uncited_claims == ["Metformin is commonly prescribed for type 2 diabetes."]
        assert report.unsupported_claims == []
        assert len(calls) == 1


# ---------------------------------------------------------------------------
# 2. check_grounding — a fabricated medication is flagged unsupported
# ---------------------------------------------------------------------------


class TestFabricatedClaimFlaggedUnsupported:
    def test_fabricated_medication_is_flagged_unsupported(self, monkeypatch):
        items = [_item("2026-01-01: patient takes metformin 500mg for type 2 diabetes.", turn_ids=[4])]
        fake, calls = _queued_complete(
            [
                {
                    "claims": [
                        {"text": "Patient takes metformin 500mg.", "status": "supported", "cited_item": 1, "reason": "evidence states this"},
                        {"text": "Patient also takes lisinopril 40mg for hypertension.", "status": "unsupported", "cited_item": -1, "reason": "no evidence item mentions lisinopril or hypertension"},
                    ],
                    "confidence": 0.85,
                }
            ]
        )
        monkeypatch.setattr(llm, "complete", fake)
        report = gr.check_grounding(
            "Patient takes metformin 500mg and also takes lisinopril 40mg for hypertension.",
            items,
            dry_run=False,
            use_cache=False,
        )
        assert report.is_grounded is False
        assert report.unsupported_claims == ["Patient also takes lisinopril 40mg for hypertension."]
        assert len(report.citations) == 1  # only the real, supported claim contributes a citation
        assert len(calls) == 1

    def test_unrecognized_status_fails_safe_to_unsupported(self, monkeypatch):
        """A schema-shape drift (a status string outside the enum) must not
        be silently dropped or trusted as safe -- fail toward the dangerous
        case, same direction as llm.py's own overestimate-is-safe budget
        check."""
        items = [_item("evidence text", turn_ids=[1])]
        fake, calls = _queued_complete(
            [{"claims": [{"text": "some claim", "status": "maybe", "cited_item": -1, "reason": "ambiguous"}], "confidence": 0.5}]
        )
        monkeypatch.setattr(llm, "complete", fake)
        report = gr.check_grounding("some claim", items, dry_run=False, use_cache=False)
        assert report.is_grounded is False
        assert report.unsupported_claims == ["some claim"]

    def test_supported_claim_citing_nonexistent_item_is_downgraded_to_unsupported(self, monkeypatch):
        """An ungrounded citation (the judge claims 'supported' but names an
        item index that does not exist) is itself indistinguishable from a
        confabulation -- must never be trusted at face value."""
        items = [_item("only evidence item", turn_ids=[1])]
        fake, calls = _queued_complete(
            [{"claims": [{"text": "a claim", "status": "supported", "cited_item": 7, "reason": "bogus citation"}], "confidence": 0.9}]
        )
        monkeypatch.setattr(llm, "complete", fake)
        report = gr.check_grounding("a claim", items, dry_run=False, use_cache=False)
        assert report.is_grounded is False
        assert report.unsupported_claims == ["a claim"]
        assert report.citations == []


# ---------------------------------------------------------------------------
# 3. Zero-LLM-call short circuits
# ---------------------------------------------------------------------------


class TestStructuralAbsenceShortCircuits:
    def test_structural_absence_makes_no_llm_call(self, monkeypatch):
        monkeypatch.setattr(llm, "complete", _no_network)
        report = gr.check_grounding("NOT_IN_RECORD", [], structural_absence=True, dry_run=False)
        assert report.is_grounded is True
        assert report.n_judge_calls == 0
        assert report.cost_usd == 0.0
        assert report.shortcut_reason == "structural_absence"

    def test_structural_absence_short_circuits_even_if_the_answer_is_not_a_clean_abstention(self, monkeypatch):
        """The signal is trusted as given, not re-derived -- even a
        non-abstaining answer text does not trigger a call when the
        retrieval layer already says the entity is not in the record."""
        monkeypatch.setattr(llm, "complete", _no_network)
        report = gr.check_grounding("Patient takes atorvastatin.", [], structural_absence=True, dry_run=False)
        assert report.n_judge_calls == 0
        assert report.shortcut_reason == "structural_absence"


class TestAbstentionAnswerShortCircuits:
    def test_not_in_record_answer_makes_no_llm_call(self, monkeypatch):
        monkeypatch.setattr(llm, "complete", _no_network)
        items = [_item("some evidence", turn_ids=[1])]
        report = gr.check_grounding("NOT_IN_RECORD", items, dry_run=False)
        assert report.is_grounded is True
        assert report.n_judge_calls == 0
        assert report.shortcut_reason == "abstained_answer"

    def test_empty_answer_makes_no_llm_call(self, monkeypatch):
        monkeypatch.setattr(llm, "complete", _no_network)
        report = gr.check_grounding("", [_item("x", turn_ids=[1])], dry_run=False)
        assert report.n_judge_calls == 0
        assert report.is_grounded is True


class TestNoEvidenceShortCircuits:
    def test_non_abstaining_answer_with_zero_items_is_flagged_unsupported_with_no_call(self, monkeypatch):
        monkeypatch.setattr(llm, "complete", _no_network)
        report = gr.check_grounding("Patient takes warfarin 5mg.", [], structural_absence=False, dry_run=False)
        assert report.n_judge_calls == 0
        assert report.is_grounded is False
        assert report.unsupported_claims == ["Patient takes warfarin 5mg."]
        assert report.shortcut_reason == "no_evidence"


# ---------------------------------------------------------------------------
# 4. enforce() — policies
# ---------------------------------------------------------------------------


def _ungrounded_report() -> gr.GroundingReport:
    item = _item("real evidence", turn_ids=[1])
    return gr.GroundingReport(
        is_grounded=False,
        citations=[item],
        uncited_claims=["general framing statement"],
        unsupported_claims=["fabricated medication claim"],
        confidence=0.8,
        claims=[
            gr.ClaimVerdict(text="a real supported claim", status="supported", cited_item=item, reason="r"),
            gr.ClaimVerdict(text="general framing statement", status="uncited", cited_item=None, reason="r"),
            gr.ClaimVerdict(text="fabricated medication claim", status="unsupported", cited_item=None, reason="r"),
        ],
        cost_usd=0.0003,
        n_judge_calls=1,
    )


def _grounded_report() -> gr.GroundingReport:
    item = _item("real evidence", turn_ids=[1])
    return gr.GroundingReport(
        is_grounded=True,
        citations=[item],
        uncited_claims=[],
        unsupported_claims=[],
        confidence=0.95,
        claims=[gr.ClaimVerdict(text="a real supported claim", status="supported", cited_item=item, reason="r")],
        cost_usd=0.0002,
        n_judge_calls=1,
    )


class TestEnforceWarn:
    def test_warn_leaves_text_intact_even_when_ungrounded(self, monkeypatch):
        monkeypatch.setattr(llm, "complete", _no_network)
        answer = "a real supported claim. general framing statement. fabricated medication claim."
        result = gr.enforce(answer, [], "warn", report=_ungrounded_report())
        assert result.text == answer
        assert result.modified is False
        assert result.report.is_grounded is False  # the finding is still visible via .report

    def test_warn_leaves_text_intact_when_grounded(self):
        answer = "a real supported claim."
        result = gr.enforce(answer, [], "warn", report=_grounded_report())
        assert result.text == answer
        assert result.modified is False


class TestEnforceAbstain:
    def test_abstain_replaces_ungrounded_answer_with_refusal(self):
        answer = "a real supported claim. general framing statement. fabricated medication claim."
        result = gr.enforce(answer, [], "abstain", report=_ungrounded_report())
        assert result.text == "NOT_IN_RECORD"
        assert result.modified is True
        assert result.report.is_grounded is False

    def test_abstain_is_a_noop_when_already_grounded(self):
        answer = "a real supported claim."
        result = gr.enforce(answer, [], "abstain", report=_grounded_report())
        assert result.text == answer
        assert result.modified is False


class TestEnforceStrip:
    def test_strip_removes_unsupported_claim_keeps_supported_and_uncited(self):
        answer = "a real supported claim. general framing statement. fabricated medication claim."
        result = gr.enforce(answer, [], "strip", report=_ungrounded_report())
        assert "fabricated medication claim" not in result.text
        assert "a real supported claim" in result.text
        assert "general framing statement" in result.text
        assert result.modified is True

    def test_strip_is_a_noop_when_nothing_unsupported(self):
        answer = "a real supported claim."
        result = gr.enforce(answer, [], "strip", report=_grounded_report())
        assert result.text == answer
        assert result.modified is False

    def test_strip_collapses_to_abstain_text_when_everything_unsupported(self):
        item = None
        report = gr.GroundingReport(
            is_grounded=False, citations=[], uncited_claims=[], unsupported_claims=["only claim"],
            confidence=0.9,
            claims=[gr.ClaimVerdict(text="only claim", status="unsupported", cited_item=item, reason="r")],
            cost_usd=0.0002, n_judge_calls=1,
        )
        result = gr.enforce("only claim", [], "strip", report=report)
        assert result.text == "NOT_IN_RECORD"
        assert result.modified is True


class TestEnforceInvalidPolicy:
    def test_unknown_policy_raises(self):
        with pytest.raises(ValueError):
            gr.enforce("answer", [], "delete", report=_grounded_report())


# ---------------------------------------------------------------------------
# 5. enabled — genuine no-op
# ---------------------------------------------------------------------------


class TestDisabledIsANoOp:
    def test_disabled_makes_no_llm_call_and_leaves_text_unchanged(self, monkeypatch):
        monkeypatch.setattr(llm, "complete", _no_network)
        answer = "the patient takes a fabricated drug"
        items = [_item("unrelated evidence", turn_ids=[1])]
        result = gr.enforce(answer, items, "abstain", enabled=False)
        assert result.text == answer
        assert result.modified is False
        assert result.report.enabled is False
        assert result.report.n_judge_calls == 0
        assert result.report.shortcut_reason == "disabled"

    def test_disabled_is_a_noop_regardless_of_policy(self, monkeypatch):
        monkeypatch.setattr(llm, "complete", _no_network)
        answer = "some answer"
        for policy in sorted(gr.VALID_POLICIES):
            result = gr.enforce(answer, [], policy, enabled=False)
            assert result.text == answer
            assert result.modified is False


# ---------------------------------------------------------------------------
# 6. enforce() computes its own report when none is supplied
# ---------------------------------------------------------------------------


class TestEnforceComputesReportWhenNotSupplied:
    def test_enforce_calls_check_grounding_exactly_once_when_no_report_given(self, monkeypatch):
        items = [_item("patient takes metformin 500mg.", turn_ids=[1])]
        fake, calls = _queued_complete(
            [{"claims": [{"text": "Patient takes metformin 500mg.", "status": "supported", "cited_item": 1, "reason": "r"}], "confidence": 0.9}]
        )
        monkeypatch.setattr(llm, "complete", fake)
        result = gr.enforce("Patient takes metformin 500mg.", items, "warn", dry_run=False, use_cache=False)
        assert len(calls) == 1
        assert result.report.is_grounded is True

    def test_reusing_a_precomputed_report_never_calls_llm_complete(self, monkeypatch):
        monkeypatch.setattr(llm, "complete", _no_network)
        result = gr.enforce("some answer", [], "warn", report=_grounded_report())
        assert result.report.is_grounded is True


# ---------------------------------------------------------------------------
# 7. dry_run — never touches a real provider client
# ---------------------------------------------------------------------------


class TestDryRunNoNetwork:
    def test_dry_run_never_reaches_a_real_google_client(self, monkeypatch):
        monkeypatch.setattr(llm, "_get_google_client", _no_network)
        items = [_item("evidence text", turn_ids=[1])]
        report = gr.check_grounding("some non-abstaining answer", items, dry_run=True)
        assert isinstance(report, gr.GroundingReport)
        assert report.n_judge_calls == 1  # a real (stub) call was made, just never a network one


# ---------------------------------------------------------------------------
# 8. Confidence parsing is clipped to [0, 1]
# ---------------------------------------------------------------------------


class TestConfidenceParsing:
    def test_out_of_range_confidence_is_clipped(self, monkeypatch):
        items = [_item("evidence", turn_ids=[1])]
        fake, _ = _queued_complete([{"claims": [], "confidence": 5.0}])
        monkeypatch.setattr(llm, "complete", fake)
        report = gr.check_grounding("a plain factual claim", items, dry_run=False, use_cache=False)
        assert report.confidence == 1.0

    def test_missing_confidence_defaults_to_one(self, monkeypatch):
        items = [_item("evidence", turn_ids=[1])]
        fake, _ = _queued_complete([{"claims": []}])
        monkeypatch.setattr(llm, "complete", fake)
        report = gr.check_grounding("a plain factual claim", items, dry_run=False, use_cache=False)
        assert report.confidence == 1.0
