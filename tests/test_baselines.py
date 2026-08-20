"""tests/test_baselines.py — coverage for the rung-3/rung-4 baselines
(`eval/baselines/dense.py`, `eval/baselines/lexical.py`): each retrieves the
obviously-correct chunk for a hand-picked question, the chunk/k sweep runs,
token/latency accounting is populated, and both emit valid `RetrieveItem`s.

Everything here runs against a synthetic fixture `Conversation` built
directly from `medmemgraph.pipeline.loader`'s public dataclasses — never the
real MedLoCoMo corpus, never HydraDB, never an Anthropic API call (all
`Answerer`-level tests force `dry_run=True`), matching the offline-CI
convention every other `tests/test_*.py` in this repo already follows.
"""

from __future__ import annotations

import numpy as np
import pytest

from medmemgraph.contracts import RetrieveItem, RetrieveResult
from medmemgraph.eval.baselines.dense import (
    CHUNK_SIZE_SWEEP,
    K_SWEEP as DENSE_K_SWEEP,
    Chunk,
    DenseRAGAnswerer,
    DensePatientIndex,
    DenseRetriever,
    chunk_admission,
    chunk_conversation,
    dense_recall_sweep,
    embed,
)
from medmemgraph.eval.baselines.lexical import (
    K_SWEEP as LEXICAL_K_SWEEP,
    LexicalAnswerer,
    LexicalPatientIndex,
    LexicalRetriever,
    lexical_recall_sweep,
)
from medmemgraph.pipeline.loader import Admission, Conversation

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _turn(turn_number: int, speaker: str, text: str, time: str = "2126-01-01T09:00:00") -> dict:
    return {"turn_number": turn_number, "time": time, "speaker": speaker, "text": text}


@pytest.fixture
def allergy_conversation() -> Conversation:
    """Two admissions. Admission 1 carries one unambiguous, hand-picked fact
    (a penicillin allergy) that shares content words with the hand-picked
    test question below; admission 2 is unrelated filler so a retriever that
    ignores content and just returns "whatever is first" fails the test."""
    admissions = (
        Admission(
            hadm_id="adm1",
            admission_start="2126-01-01",
            admission_end="2126-01-03",
            conversation_lines=(
                _turn(1, "Doctor", "How are you feeling today?"),
                _turn(2, "Patient", "I have had chest pain for two days."),
                _turn(3, "Doctor", "Do you have any known drug allergies?"),
                _turn(4, "Patient", "Yes, I am allergic to penicillin, it gives me a bad rash."),
                _turn(5, "Doctor", "Noted, we will avoid penicillin-class antibiotics."),
            ),
        ),
        Admission(
            hadm_id="adm2",
            admission_start="2126-02-01",
            admission_end="2126-02-02",
            conversation_lines=(
                _turn(1, "Doctor", "Any recurrence of the chest pain?", time="2126-02-01T09:00:00"),
                _turn(2, "Patient", "No, it fully resolved after discharge.", time="2126-02-01T09:05:00"),
                _turn(3, "Doctor", "Any new symptoms since then?", time="2126-02-01T09:10:00"),
                _turn(4, "Patient", "I've been more tired than usual lately.", time="2126-02-01T09:15:00"),
            ),
        ),
    )
    return Conversation(
        subject_id="fixture-baselines", processed_hadm_ids=("adm1", "adm2"), admissions=admissions
    )


ALLERGY_QUESTION = "Is the patient allergic to penicillin or any other medication?"


@pytest.fixture
def qa_items_with_evidence() -> list[dict]:
    """QA items whose gold `evidence.admissions` names the admission the
    obviously-correct chunk lives in, so the sweep functions have a real
    gold signal to score against."""
    return [
        {
            "qa_id": "q1",
            "question": ALLERGY_QUESTION,
            "answer": "penicillin",
            "evidence": {"admissions": ["adm1"]},
        },
        {
            "qa_id": "q2",
            "question": "Has the patient's chest pain come back since discharge?",
            "answer": "no",
            "evidence": {"admissions": ["adm2"]},
        },
        {
            "qa_id": "q3",
            "question": "Is the patient more tired than usual recently?",
            "answer": "yes",
            "evidence": {"admissions": ["adm2"]},
        },
    ]


def _fixture_loader(conversation: Conversation):
    """A `conversation_loader`-shaped callable that ignores its arguments and
    always returns the fixture — keeps `DenseRetriever`/`LexicalRetriever`
    testable without touching the real corpus loader."""

    def _load(patient_id: str, root: object | None) -> Conversation:
        del patient_id, root
        return conversation

    return _load


# ---------------------------------------------------------------------------
# embed() — deterministic hashing-trick fallback
# ---------------------------------------------------------------------------


class TestEmbed:
    def test_deterministic_across_calls(self):
        a = embed(["metformin", "Glucophage"])
        b = embed(["metformin", "Glucophage"])
        assert np.array_equal(a, b)

    def test_same_string_same_vector_in_different_batches(self):
        """A vector for a given string must not depend on what else is in
        the same `embed()` call — required for query/corpus vectors embedded
        at different times to be cosine-comparable at all."""
        solo = embed(["penicillin allergy"])[0]
        batched = embed(["something else entirely", "penicillin allergy", "a third string"])[1]
        assert np.allclose(solo, batched)

    def test_l2_normalized(self):
        vectors = embed(["a short note", "a considerably longer clinical note about medications"])
        norms = np.linalg.norm(vectors, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_empty_input_shape(self):
        vectors = embed([])
        assert vectors.shape == (0, vectors.shape[1])

    def test_distinct_texts_yield_distinct_vectors(self):
        a, b = embed(["penicillin allergy causes a rash", "chest pain resolved after discharge"])
        assert not np.allclose(a, b)


# ---------------------------------------------------------------------------
# Chunking — fixed token size, session-bounded
# ---------------------------------------------------------------------------


class TestChunking:
    def test_chunk_admission_never_exceeds_admission_boundary(self, allergy_conversation):
        admission = allergy_conversation.admissions[0]
        chunks = chunk_admission(admission, chunk_size_tokens=20)
        assert all(c.session_id == "adm1" for c in chunks)
        # Every turn_number in every chunk actually belongs to this admission.
        valid_turn_numbers = {t.turn_number for t in admission.turns()}
        for c in chunks:
            assert set(c.turn_ids) <= valid_turn_numbers

    def test_chunk_conversation_covers_every_turn_exactly_once(self, allergy_conversation):
        chunks = chunk_conversation(allergy_conversation, chunk_size_tokens=15)
        seen: list[tuple[str, int]] = []
        for c in chunks:
            for t in c.turn_ids:
                seen.append((c.session_id, t))
        expected = [
            (admission.hadm_id, turn.turn_number)
            for admission in allergy_conversation.admissions
            for turn in admission.turns()
        ]
        assert sorted(seen) == sorted(expected)

    def test_smaller_chunk_size_yields_more_chunks(self, allergy_conversation):
        small = chunk_conversation(allergy_conversation, chunk_size_tokens=10)
        large = chunk_conversation(allergy_conversation, chunk_size_tokens=10_000)
        assert len(small) >= len(large)
        # A single huge budget puts each admission's turns in one chunk.
        assert len(large) == len(allergy_conversation.admissions)

    def test_chunk_is_frozen_dataclass_shape(self, allergy_conversation):
        chunks = chunk_conversation(allergy_conversation, chunk_size_tokens=20)
        assert chunks and isinstance(chunks[0], Chunk)
        assert isinstance(chunks[0].session_id, str)
        assert isinstance(chunks[0].turn_ids, list)
        assert isinstance(chunks[0].text, str)


# ---------------------------------------------------------------------------
# "Retrieves the obviously-correct chunk for a hand-picked question"
# ---------------------------------------------------------------------------


class TestDenseObviousChunk:
    def test_top_result_is_the_penicillin_chunk(self, allergy_conversation):
        index = DensePatientIndex(allergy_conversation, chunk_size_tokens=30)
        result = index.query(ALLERGY_QUESTION, k=2)
        assert result.items, "expected at least one retrieved chunk"
        top = result.items[0]
        assert top.session_id == "adm1"
        assert "penicillin" in top.text.lower()

    def test_unrelated_admission_does_not_outrank_the_correct_one(self, allergy_conversation):
        index = DensePatientIndex(allergy_conversation, chunk_size_tokens=30)
        result = index.query(ALLERGY_QUESTION, k=1)
        assert result.items[0].session_id == "adm1"


class TestLexicalObviousChunk:
    def test_top_result_is_the_penicillin_turn_window(self, allergy_conversation):
        index = LexicalPatientIndex(allergy_conversation, window=2)
        result = index.query(ALLERGY_QUESTION, k=2)
        assert result.items, "expected at least one retrieved item"
        top = result.items[0]
        assert top.session_id == "adm1"
        assert "penicillin" in top.text.lower()

    def test_session_aware_expansion_never_crosses_admission_boundary(self, allergy_conversation):
        """Turn 1 of adm2 is at the very start of its admission; a naive
        (non-session-aware) +/-2 expansion applied over a *flattened* turn
        list would pull in adm1's final turns as "neighbors". Assert that
        never happens."""
        index = LexicalPatientIndex(allergy_conversation, window=2)
        result = index.query("Any recurrence of the chest pain since discharge?", k=1)
        assert result.items
        top = result.items[0]
        assert top.session_id == "adm2"
        # Every window turn_id must exist in adm2's own turn list.
        adm2_turn_numbers = {t.turn_number for t in allergy_conversation.admissions[1].turns()}
        assert set(top.turn_ids) <= adm2_turn_numbers

    def test_expansion_window_pulls_in_neighbors(self, allergy_conversation):
        index = LexicalPatientIndex(allergy_conversation, window=2)
        result = index.query(ALLERGY_QUESTION, k=1)
        top = result.items[0]
        # The hit turn (4) plus its +/-2 window (2..5, clipped to [1,5])
        # should include more than just the single matching turn.
        assert len(top.turn_ids) > 1
        assert 4 in top.turn_ids


# ---------------------------------------------------------------------------
# Valid RetrieveItems
# ---------------------------------------------------------------------------


class TestValidRetrieveItems:
    def test_dense_items_are_well_formed(self, allergy_conversation):
        index = DensePatientIndex(allergy_conversation, chunk_size_tokens=20)
        result = index.query(ALLERGY_QUESTION, k=3)
        assert isinstance(result, RetrieveResult)
        assert result.route == "vector"
        for item in result.items:
            assert isinstance(item, RetrieveItem)
            assert isinstance(item.session_id, str) and item.session_id
            assert isinstance(item.turn_ids, list) and all(isinstance(t, int) for t in item.turn_ids)
            assert isinstance(item.score, float)
            assert item.channel == "vector"

    def test_lexical_items_are_well_formed(self, allergy_conversation):
        index = LexicalPatientIndex(allergy_conversation, window=2)
        result = index.query(ALLERGY_QUESTION, k=3)
        assert isinstance(result, RetrieveResult)
        assert result.route == "lexical"
        for item in result.items:
            assert isinstance(item, RetrieveItem)
            assert isinstance(item.session_id, str) and item.session_id
            assert isinstance(item.turn_ids, list) and all(isinstance(t, int) for t in item.turn_ids)
            assert isinstance(item.score, float)
            assert item.channel == "lexical"

    def test_empty_conversation_reports_structural_absence_not_a_crash(self):
        empty = Conversation(subject_id="empty", processed_hadm_ids=(), admissions=())
        dense_result = DensePatientIndex(empty, chunk_size_tokens=50).query("anything?", k=5)
        lexical_result = LexicalPatientIndex(empty, window=2).query("anything?", k=5)
        assert dense_result.items == [] and dense_result.structural_absence is True
        assert lexical_result.items == [] and lexical_result.structural_absence is True

    def test_retrievers_are_drop_in_compatible_with_mock_retrieve_signature(self, allergy_conversation):
        """Both retrievers must match `(question, patient_id, k) ->
        RetrieveResult` exactly, since that is the shape
        `eval.reader.ReaderAnswerer(retriever=...)` calls."""
        dense_retriever = DenseRetriever(chunk_size_tokens=20, conversation_loader=_fixture_loader(allergy_conversation))
        lexical_retriever = LexicalRetriever(window=2, conversation_loader=_fixture_loader(allergy_conversation))
        dense_out = dense_retriever(ALLERGY_QUESTION, "any-patient-id", 3)
        lexical_out = lexical_retriever(ALLERGY_QUESTION, "any-patient-id", 3)
        assert isinstance(dense_out, RetrieveResult)
        assert isinstance(lexical_out, RetrieveResult)

    def test_dense_retriever_caches_index_per_patient(self, allergy_conversation):
        calls = {"n": 0}

        def counting_loader(patient_id: str, root: object | None) -> Conversation:
            calls["n"] += 1
            return allergy_conversation

        retriever = DenseRetriever(chunk_size_tokens=20, conversation_loader=counting_loader)
        retriever("q1", "patient-x", 3)
        retriever("q2", "patient-x", 3)
        retriever("q3", "patient-x", 3)
        assert calls["n"] == 1, "expected the conversation to be loaded once, not once per question"


# ---------------------------------------------------------------------------
# Chunk/k sweep runs
# ---------------------------------------------------------------------------


class TestSweepRuns:
    def test_dense_sweep_covers_the_full_grid(self, allergy_conversation, qa_items_with_evidence):
        rows = dense_recall_sweep(qa_items_with_evidence, allergy_conversation)
        assert len(rows) == len(CHUNK_SIZE_SWEEP) * len(DENSE_K_SWEEP)
        seen = {(r["chunk_size"], r["k"]) for r in rows}
        assert seen == {(cs, k) for cs in CHUNK_SIZE_SWEEP for k in DENSE_K_SWEEP}
        for row in rows:
            assert row["n_items"] == len(qa_items_with_evidence)
            assert 0.0 <= row["admission_hit_rate"] <= 1.0
            assert row["mean_retrieved_tokens"] >= 0.0
            assert row["mean_latency_ms"] >= 0.0

    def test_dense_sweep_finds_the_gold_admission_at_least_once(self, allergy_conversation, qa_items_with_evidence):
        """With k covering the whole (tiny) fixture corpus, the correct
        admission must be found for at least the hand-picked allergy
        question -- a sweep that never hits gold at any setting would be a
        broken retriever, not just a "worst-case" baseline."""
        rows = dense_recall_sweep(qa_items_with_evidence, allergy_conversation, k_values=(10,))
        assert any(row["admission_hit_rate"] > 0 for row in rows)

    def test_lexical_sweep_covers_the_full_k_grid(self, allergy_conversation, qa_items_with_evidence):
        rows = lexical_recall_sweep(qa_items_with_evidence, allergy_conversation)
        assert len(rows) == len(LEXICAL_K_SWEEP)
        assert {r["k"] for r in rows} == set(LEXICAL_K_SWEEP)
        for row in rows:
            assert row["n_items"] == len(qa_items_with_evidence)
            assert 0.0 <= row["admission_hit_rate"] <= 1.0
            assert row["mean_retrieved_tokens"] >= 0.0
            assert row["mean_latency_ms"] >= 0.0

    def test_lexical_sweep_finds_the_gold_admission_at_least_once(self, allergy_conversation, qa_items_with_evidence):
        rows = lexical_recall_sweep(qa_items_with_evidence, allergy_conversation, k_values=(10,))
        assert any(row["admission_hit_rate"] > 0 for row in rows)


# ---------------------------------------------------------------------------
# Token / latency accounting populated (Answerer level, dry_run -- no API key)
# ---------------------------------------------------------------------------


class TestTokenLatencyAccounting:
    def test_dense_answerer_populates_tokens_and_latency(self, allergy_conversation):
        answerer = DenseRAGAnswerer(
            chunk_size_tokens=30, k=3, dry_run=True, conversation_loader=_fixture_loader(allergy_conversation)
        )
        result = answerer.answer(ALLERGY_QUESTION, None, patient_id="fixture-baselines")
        assert result.prompt_tokens > 0
        assert result.completion_tokens >= 0
        assert result.latency_ms > 0.0
        assert result.total_tokens == result.prompt_tokens + result.completion_tokens

    def test_lexical_answerer_populates_tokens_and_latency(self, allergy_conversation):
        answerer = LexicalAnswerer(
            window=2, k=3, dry_run=True, conversation_loader=_fixture_loader(allergy_conversation)
        )
        result = answerer.answer(ALLERGY_QUESTION, None, patient_id="fixture-baselines")
        assert result.prompt_tokens > 0
        assert result.completion_tokens >= 0
        assert result.latency_ms > 0.0

    def test_dense_answerer_name_and_mode(self):
        assert DenseRAGAnswerer.name == "dense"
        assert DenseRAGAnswerer.mode == "chain_of_note"

    def test_lexical_answerer_name_and_mode(self):
        assert LexicalAnswerer.name == "lexical"
        assert LexicalAnswerer.mode == "chain_of_note"

    def test_larger_chunk_size_changes_prompt_tokens(self, allergy_conversation):
        """A sanity check that chunk_size actually affects what gets sent to
        the reader (and therefore its token cost) -- not a hard-coded
        no-op."""
        small = DenseRAGAnswerer(
            chunk_size_tokens=10, k=5, dry_run=True, conversation_loader=_fixture_loader(allergy_conversation)
        )
        large = DenseRAGAnswerer(
            chunk_size_tokens=1000, k=5, dry_run=True, conversation_loader=_fixture_loader(allergy_conversation)
        )
        small_result = small.answer(ALLERGY_QUESTION, None, patient_id="fixture-baselines")
        large_result = large.answer(ALLERGY_QUESTION, None, patient_id="fixture-baselines")
        # Different chunk sizes produce a different number/shape of chunks,
        # so the rendered context (and thus prompt_tokens) should differ.
        assert small_result.prompt_tokens != large_result.prompt_tokens
