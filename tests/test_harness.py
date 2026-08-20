"""tests/test_harness.py — coverage for the eval harness (story deliverable
3/4/5): per-category table shape, adversarial-vs-answerable judge routing,
and full-context truncation being recorded.

Everything here runs against synthetic fixture data built directly from
`medmemgraph.pipeline.loader`'s public dataclasses, never the real MedLoCoMo
corpus and never HydraDB — the harness itself (`evaluate()`) does not care
where its `qa_items`/`Conversation` came from, which is exactly what makes
it CI-safe. `run_harness()`'s thin corpus-loading wrapper is exercised
separately by `tests/test_loader_allowlist.py`.

All harness-level tests here run in `dry_run=True` mode, i.e. zero Anthropic
API calls — this suite must pass with no `ANTHROPIC_API_KEY` set.
"""

from __future__ import annotations

import json

import pytest

from medmemgraph.eval.baselines.fullctx import FullContextAnswerer
from medmemgraph.eval.baselines.nomem import NoMemoryAnswerer
from medmemgraph.eval.harness import (
    ADVERSARIAL_TYPE,
    QUESTION_TYPES,
    SCOPES,
    CategoryRow,
    evaluate,
    render_run,
    render_table,
    write_results,
)
from medmemgraph.eval.judge import Judge
from medmemgraph.pipeline.loader import Admission, Conversation

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _turn(turn_number: int, speaker: str, text: str, time: str = "2126-01-01T09:00:00") -> dict:
    return {"turn_number": turn_number, "time": time, "speaker": speaker, "text": text}


@pytest.fixture
def small_conversation() -> Conversation:
    """Two admissions, two turns each — enough for a fullctx prompt, small
    enough to never trigger truncation at any realistic window."""
    admissions = (
        Admission(
            hadm_id="adm1",
            admission_start="2126-01-01",
            admission_end="2126-01-02",
            conversation_lines=(
                _turn(1, "Doctor", "How are you feeling today?"),
                _turn(2, "Patient", "I have had chest pain for two days."),
            ),
        ),
        Admission(
            hadm_id="adm2",
            admission_start="2126-02-01",
            admission_end="2126-02-02",
            conversation_lines=(
                _turn(1, "Doctor", "Any recurrence of the chest pain?", time="2126-02-01T09:00:00"),
                _turn(2, "Patient", "No, it fully resolved after discharge.", time="2126-02-01T09:05:00"),
            ),
        ),
    )
    return Conversation(subject_id="fixture-1", processed_hadm_ids=("adm1", "adm2"), admissions=admissions)


@pytest.fixture
def qa_items_all_categories() -> list[dict]:
    """One QA item per canonical question_type, covering both scopes, with
    an adversarial item whose gold answer matches the real corpus's
    verbatim string (see judge.py docstring / dev.log.md for the corpus
    check this literal string is grounded in)."""
    items = []
    for i, (qtype, scope) in enumerate(
        [
            ("medical_reasoning", "single_admission"),
            ("care_plan_rationale", "single_admission"),
            ("longitudinal_progression", "cross_admission"),
            ("cross_admission_comparison", "cross_admission"),
            ("frequency_pattern", "cross_admission"),
            (ADVERSARIAL_TYPE, "cross_admission"),
        ]
    ):
        items.append(
            {
                "qa_id": f"fixture_q{i}",
                "scope": scope,
                "question_type": qtype,
                "question": f"Fixture question #{i} about {qtype}?",
                "answer": "the question is not answerable" if qtype == ADVERSARIAL_TYPE else "chest pain resolved",
                "evidence": {"admissions": ["adm1", "adm2"]},
            }
        )
    return items


# ---------------------------------------------------------------------------
# 1. Per-category table shape
# ---------------------------------------------------------------------------


class TestPerCategoryTableShape:
    def test_by_question_type_covers_every_category_present(self, small_conversation, qa_items_all_categories):
        run = evaluate(
            qa_items_all_categories,
            small_conversation,
            patient_id="fixture-1",
            system_name="nomem",
            dry_run=True,
        )

        # One row per distinct question_type in the input, no more, no fewer.
        categories = {row.category for row in run.by_question_type}
        assert categories == set(QUESTION_TYPES)
        # n_items across the by-question_type rows sums to the input size.
        assert sum(row.n for row in run.by_question_type) == len(qa_items_all_categories)
        # Row order matches the canonical order named in the story, not
        # input order or alphabetical order.
        assert [row.category for row in run.by_question_type] == list(QUESTION_TYPES)

    def test_by_question_type_row_has_every_required_column(self, small_conversation, qa_items_all_categories):
        run = evaluate(
            qa_items_all_categories,
            small_conversation,
            patient_id="fixture-1",
            system_name="fullctx",
            dry_run=True,
        )
        for row in run.by_question_type:
            assert isinstance(row, CategoryRow)
            assert row.n >= 1
            assert row.accuracy is None or 0.0 <= row.accuracy <= 1.0
            assert row.mean_tokens >= 0.0
            assert row.p50_latency_ms >= 0.0
            assert row.p95_latency_ms >= 0.0

    def test_by_scope_breaks_out_single_vs_cross_admission(self, small_conversation, qa_items_all_categories):
        run = evaluate(
            qa_items_all_categories,
            small_conversation,
            patient_id="fixture-1",
            system_name="nomem",
            dry_run=True,
        )
        scopes = {row.category for row in run.by_scope}
        assert scopes <= set(SCOPES)
        assert sum(row.n for row in run.by_scope) == len(qa_items_all_categories)
        # Fixture has 2 single_admission + 4 cross_admission items.
        by_scope = {row.category: row.n for row in run.by_scope}
        assert by_scope["single_admission"] == 2
        assert by_scope["cross_admission"] == 4

    def test_render_table_and_render_run_do_not_raise_and_are_strings(self, small_conversation, qa_items_all_categories):
        run = evaluate(
            qa_items_all_categories,
            small_conversation,
            patient_id="fixture-1",
            system_name="nomem",
            dry_run=True,
        )
        text = render_table("by question_type", run.by_question_type)
        assert isinstance(text, str) and "medical_reasoning" in text
        full = render_run(run)
        assert isinstance(full, str) and "answerable_accuracy" in full

    def test_json_output_round_trips(self, tmp_path, small_conversation, qa_items_all_categories):
        run = evaluate(
            qa_items_all_categories,
            small_conversation,
            patient_id="fixture-1",
            system_name="nomem",
            dry_run=True,
        )
        path = write_results(run, results_dir=tmp_path)
        assert path.exists()
        loaded = json.loads(path.read_text())
        assert loaded["patient_id"] == "fixture-1"
        assert loaded["system_name"] == "nomem"
        assert len(loaded["by_question_type"]) == len(QUESTION_TYPES)
        assert len(loaded["records"]) == len(qa_items_all_categories)


# ---------------------------------------------------------------------------
# 2. Adversarial items route to abstention scoring
# ---------------------------------------------------------------------------


class TestAdversarialRoutesToAbstentionScoring:
    def test_adversarial_question_type_uses_abstention_mode(self):
        judge = Judge(force_fallback=True)
        verdict = judge.judge(
            question="What is the patient allergic to?",
            gold_answer="the question is not answerable",
            system_answer="I don't have enough information to determine that.",
            question_type=ADVERSARIAL_TYPE,
        )
        assert verdict.mode == "abstention"
        # The system declined -> correct, even though its text has ~zero
        # token overlap with the gold string "the question is not
        # answerable". This is the whole point of routing adversarial items
        # differently: comparing the answer text to the gold string would
        # be nonsensical here.
        assert verdict.correct is True

    def test_non_adversarial_question_type_uses_answerable_mode(self):
        judge = Judge(force_fallback=True)
        verdict = judge.judge(
            question="Why did the patient have chest pain?",
            gold_answer="unclear etiology, no infection on workup",
            system_answer="unclear etiology, no infection was found on workup",
            question_type="medical_reasoning",
        )
        assert verdict.mode == "answerable"

    def test_adversarial_item_with_substantive_answer_is_incorrect(self):
        """A system that confabulates an answer to an unanswerable question
        must score as WRONG under abstention scoring — it did not decline."""
        judge = Judge(force_fallback=True)
        verdict = judge.judge(
            question="What is the patient allergic to?",
            gold_answer="the question is not answerable",
            system_answer="The patient is allergic to penicillin.",
            question_type=ADVERSARIAL_TYPE,
        )
        assert verdict.mode == "abstention"
        assert verdict.correct is False

    def test_answerable_accuracy_and_abstention_accuracy_are_never_blended(
        self, small_conversation, qa_items_all_categories
    ):
        run = evaluate(
            qa_items_all_categories,
            small_conversation,
            patient_id="fixture-1",
            system_name="nomem",
            dry_run=True,
        )
        # Two separate numbers must exist, and neither is a function of
        # records outside its own category: answerable_accuracy is
        # computed only from non-adversarial records, abstention_accuracy
        # only from the adversarial record(s).
        answerable_records = [r for r in run.records if r.question_type != ADVERSARIAL_TYPE]
        adversarial_records = [r for r in run.records if r.question_type == ADVERSARIAL_TYPE]
        assert run.answerable_accuracy == pytest.approx(
            sum(1 for r in answerable_records if r.correct) / len(answerable_records)
        )
        assert run.abstention_accuracy == pytest.approx(
            sum(1 for r in adversarial_records if r.correct) / len(adversarial_records)
        )
        # A rendered summary line must carry both labelled separately, not
        # a single combined "accuracy" figure.
        summary = render_run(run)
        assert "answerable_accuracy=" in summary
        assert "abstention_accuracy=" in summary

    def test_harness_evaluate_records_mode_per_item(self, small_conversation, qa_items_all_categories):
        run = evaluate(
            qa_items_all_categories,
            small_conversation,
            patient_id="fixture-1",
            system_name="fullctx",
            dry_run=True,
        )
        modes = {r.question_type: r.mode for r in run.records}
        assert modes[ADVERSARIAL_TYPE] == "abstention"
        for qtype in QUESTION_TYPES:
            if qtype != ADVERSARIAL_TYPE:
                assert modes[qtype] == "answerable"


# ---------------------------------------------------------------------------
# 3. Truncation being recorded (fullctx, oldest-first)
# ---------------------------------------------------------------------------


class TestTruncationRecorded:
    @pytest.fixture
    def long_conversation(self) -> Conversation:
        """20 admissions, each with one long turn — big enough to force
        truncation under a small context window but well under any
        realistic one."""
        long_text = "A fairly long clinical sentence describing findings and history. " * 6
        admissions = tuple(
            Admission(
                hadm_id=f"adm{i}",
                admission_start=f"2126-{(i % 12) + 1:02d}-01",
                admission_end=f"2126-{(i % 12) + 1:02d}-02",
                conversation_lines=(_turn(1, "Doctor", long_text),),
            )
            for i in range(20)
        )
        return Conversation(
            subject_id="fixture-long",
            processed_hadm_ids=tuple(f"adm{i}" for i in range(20)),
            admissions=admissions,
        )

    def test_small_window_forces_truncation_and_records_it(self, long_conversation):
        answerer = FullContextAnswerer(dry_run=True, context_window_tokens=2_000)
        result = answerer.answer("What happened across admissions?", long_conversation, patient_id="fixture-long")
        assert result.truncated is True
        assert result.turns_dropped > 0
        assert result.tokens_dropped > 0

    def test_large_window_does_not_truncate(self, long_conversation):
        answerer = FullContextAnswerer(dry_run=True, context_window_tokens=190_000)
        result = answerer.answer("What happened across admissions?", long_conversation, patient_id="fixture-long")
        assert result.truncated is False
        assert result.turns_dropped == 0
        assert result.tokens_dropped == 0

    def test_truncation_drops_oldest_turns_first(self, long_conversation):
        """The oldest admission's turn text must be the first thing missing
        from the built context once truncation kicks in — this is the
        'truncate from the OLDEST end' contract, not just 'truncated=True'."""
        answerer = FullContextAnswerer(dry_run=True, context_window_tokens=1_500)
        context_text, truncated, turns_dropped, tokens_dropped = answerer._build_context(long_conversation)
        assert truncated is True
        # adm0 is the oldest (first in admission order); it must be gone.
        assert "adm0" not in context_text
        # adm19 is the newest; it must survive.
        assert "adm19" in context_text

    def test_harness_run_surfaces_truncation_count(self, long_conversation):
        qa_items = [
            {
                "qa_id": "trunc_q1",
                "scope": "cross_admission",
                "question_type": "longitudinal_progression",
                "question": "How did findings change over time?",
                "answer": "findings were stable",
                "evidence": {},
            }
        ]
        run = evaluate(
            qa_items,
            long_conversation,
            patient_id="fixture-long",
            system_name="fullctx",
            dry_run=True,
            answerer=FullContextAnswerer(dry_run=True, context_window_tokens=2_000),
        )
        assert run.n_truncated == 1
        assert run.records[0].truncated is True

    def test_nomem_never_truncates_because_it_never_reads_history(self, long_conversation):
        """nomem's contract is "no memory" — it must report truncated=False
        unconditionally, since it never builds a context to truncate."""
        answerer = NoMemoryAnswerer(dry_run=True)
        result = answerer.answer("Any question", long_conversation, patient_id="fixture-long")
        assert result.truncated is False
        assert result.turns_dropped == 0
        assert result.tokens_dropped == 0


# ---------------------------------------------------------------------------
# Honesty guard: a non-dry-run call with no API key fails loudly, not silently
# ---------------------------------------------------------------------------


class TestNoApiKeyFailsLoudly:
    def test_evaluate_without_dry_run_and_without_api_key_raises(
        self, monkeypatch, small_conversation, qa_items_all_categories
    ):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
            evaluate(
                qa_items_all_categories,
                small_conversation,
                patient_id="fixture-1",
                system_name="nomem",
                dry_run=False,
            )
