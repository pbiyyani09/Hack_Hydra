"""tests/test_channels.py — coverage for `graph/vector_index.py` and
`graph/lexical.py` (E5-S2, the "beside HydraDB" arm): per-patient brute-force
cosine + per-patient `bm25s` with +/-2 session-aware context expansion.

Two tiers, matching this repo's established convention
(`test_baselines.py` for synthetic-fixture behavioral coverage,
`test_loader_allowlist.py` for a real-corpus, `skipif`-guarded tier):

1. Synthetic-fixture tests (`Conversation` built directly from
   `pipeline.loader`'s public dataclasses, never the real corpus) —
   obvious-answer top-k retrieval for both channels, the +/-2 expansion
   actually widening the returned span (and clamping at an admission
   boundary), score ordering, empty-patient -> `[]`, and a save/load
   round trip for both channels. These run unconditionally, offline, no
   API key, matching every other `tests/test_*.py` in this repo.
2. A real-corpus, `skipif`-guarded tier (`TestRealPatient*`) that builds
   both indexes over one real MedLoCoMo patient and reports real build
   time, real on-disk-equivalent index size, and real p50/p90 query
   latency — the story's own "report real build time, index size, and
   p50 query latency per channel" requirement. Skipped, not failed, if
   `data/medlocomo/MedLoCoMo` has not been fetched (decisions/001: local
   data is fetched at setup, never vendored, so a fresh checkout without
   the fetch step must not fail this file outright).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from medmemgraph.contracts import RetrieveItem
from medmemgraph.graph.lexical import LexicalIndex
from medmemgraph.graph.vector_index import PatientIndex
from medmemgraph.pipeline.loader import Admission, Conversation, list_patients, load_conversation, load_qa

MEDLOCOMO_ROOT = os.environ.get("MEDLOCOMO_ROOT", "data/medlocomo")
CORPUS_DIR = Path(MEDLOCOMO_ROOT) / "MedLoCoMo"


# ---------------------------------------------------------------------------
# Synthetic fixtures — never the real corpus (see module docstring)
# ---------------------------------------------------------------------------


def _turn(turn_number: int, speaker: str, text: str, time: str = "2126-01-01T09:00:00") -> dict:
    return {"turn_number": turn_number, "time": time, "speaker": speaker, "text": text}


@pytest.fixture
def expansion_conversation() -> Conversation:
    """One 7-turn admission (E5-S2 AC3's own numbers: "a 7-turn admission
    and a BM25 hit on turn 4") plus a second, unrelated admission, so a
    lexical/vector hit on turn 4 of admission 1 can be checked both for
    "+/-2 expands to turns 2-6" *and* for "expansion never reaches into
    admission 2's turns". Turn 4 carries a distinctive, low-frequency
    token ("anaphylaxis") that appears nowhere else in either admission,
    so a query for that token has an unambiguous, obvious top-1 answer
    for both channels."""
    admissions = (
        Admission(
            hadm_id="adm-allergy",
            admission_start="2126-01-01",
            admission_end="2126-01-02",
            conversation_lines=(
                _turn(1, "Doctor", "How are you feeling this morning?"),
                _turn(2, "Patient", "A little anxious about the antibiotics, honestly."),
                _turn(3, "Doctor", "Understandable. Any past reactions to medications?"),
                _turn(4, "Patient", "Yes — I had anaphylaxis after amoxicillin as a child."),
                _turn(5, "Doctor", "Noted. We will avoid penicillin-class antibiotics entirely."),
                _turn(6, "Patient", "Thank you, that's a relief to hear."),
                _turn(7, "Doctor", "We'll use a cephalosporin instead and monitor you closely."),
            ),
        ),
        Admission(
            hadm_id="adm-unrelated",
            admission_start="2126-03-01",
            admission_end="2126-03-02",
            conversation_lines=(
                _turn(1, "Doctor", "How has your knee been since physical therapy started?", time="2126-03-01T09:00:00"),
                _turn(2, "Patient", "Much better, the swelling has mostly gone down.", time="2126-03-01T09:05:00"),
                _turn(3, "Doctor", "Great, let's continue the same exercise plan.", time="2126-03-01T09:10:00"),
            ),
        ),
    )
    return Conversation(subject_id="patient-expand", processed_hadm_ids=("adm-allergy", "adm-unrelated"), admissions=admissions)


@pytest.fixture
def edge_admission_conversation() -> Conversation:
    """A single 3-turn admission with the hit on turn 1 (the boundary
    case) — +/-2 expansion must clamp to the available window (turns
    1-3), never index out of range and never invent a turn 0 or turn 4
    that does not exist (E5-S2 AC3: "or the available window")."""
    admissions = (
        Admission(
            hadm_id="adm-edge",
            admission_start="2126-05-01",
            admission_end="2126-05-01",
            conversation_lines=(
                _turn(1, "Patient", "I've been having recurring migraines with visual aura."),
                _turn(2, "Doctor", "How long have the migraines been happening?"),
                _turn(3, "Patient", "About three weeks now, roughly twice a week."),
            ),
        ),
    )
    return Conversation(subject_id="patient-edge", processed_hadm_ids=("adm-edge",), admissions=admissions)


@pytest.fixture
def empty_conversation() -> Conversation:
    return Conversation(subject_id="patient-empty", processed_hadm_ids=(), admissions=())


# ---------------------------------------------------------------------------
# Vector channel
# ---------------------------------------------------------------------------


class TestPatientIndex:
    def test_obvious_answer_is_top1_and_tagged_vector(self, expansion_conversation: Conversation) -> None:
        index = PatientIndex("patient-expand", cache_path=None)
        index.build(expansion_conversation)
        results = index.search("Did you ever have anaphylaxis to a medication?", k=3)
        assert results, "expected at least one hit"
        top = results[0]
        assert isinstance(top, RetrieveItem)
        assert top.channel == "vector"
        assert top.turn_ids == [4]
        assert top.session_id == "adm-allergy"

    def test_scores_are_descending(self, expansion_conversation: Conversation) -> None:
        index = PatientIndex("patient-expand", cache_path=None)
        index.build(expansion_conversation)
        results = index.search("antibiotics and allergic reactions", k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_empty_patient_returns_empty_list(self, empty_conversation: Conversation) -> None:
        index = PatientIndex("patient-empty", cache_path=None)
        index.build(empty_conversation)
        assert index.search("anything at all", k=5) == []

    def test_indexes_both_turns_and_facts(self, expansion_conversation: Conversation) -> None:
        """ARCHITECTURE.md §7.4: "Index both turn text and emitted fact
        text." A fact whose text uses vocabulary absent from the raw
        turns must still be retrievable once passed to `build(facts=...)`."""
        from medmemgraph.contracts import EntityRef, mock_fact

        fact = mock_fact(
            patient_id="patient-expand",
            session_id="adm-allergy",
            turn_ids=[4],
            subject=EntityRef(name="patient-expand", type="Patient", canonical_id=1),
            predicate="HAS_ALLERGY_TO",
            object=EntityRef(name="amoxicillin", type="Medication", canonical_id=99),
            polarity="asserted",
        )
        index = PatientIndex("patient-expand", cache_path=None)
        index.build(expansion_conversation, facts=[fact])
        assert any(u.kind == "fact" for u in index.units)
        results = index.search("allergy to amoxicillin", k=3)
        assert any(r.text.startswith("[fact ") for r in results)

    def test_save_load_round_trip(self, expansion_conversation: Conversation, tmp_path: Path) -> None:
        cache_path = tmp_path / "embed_cache.npz"
        built = PatientIndex("patient-expand", cache_path=cache_path)
        built.build(expansion_conversation)
        built.save(tmp_path)

        loaded = PatientIndex.load("patient-expand", tmp_path, cache_path=cache_path)
        query = "anaphylaxis after amoxicillin"
        before = built.search(query, k=3)
        after = loaded.search(query, k=3)
        assert [(r.text, r.turn_ids, round(r.score, 6)) for r in before] == [
            (r.text, r.turn_ids, round(r.score, 6)) for r in after
        ]

    def test_load_missing_index_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            PatientIndex.load("no-such-patient", tmp_path)


# ---------------------------------------------------------------------------
# Lexical channel
# ---------------------------------------------------------------------------


class TestLexicalIndex:
    def test_obvious_answer_is_top1_and_tagged_lexical(self, expansion_conversation: Conversation) -> None:
        index = LexicalIndex("patient-expand")
        index.build(expansion_conversation)
        results = index.search("anaphylaxis amoxicillin", k=3)
        assert results
        top = results[0]
        assert top.channel == "lexical"
        assert 4 in top.turn_ids
        assert top.session_id == "adm-allergy"

    def test_plus_minus_2_expansion_widens_the_span(self, expansion_conversation: Conversation) -> None:
        """E5-S2 AC3, literal numbers: a hit on turn 4 of a 7-turn
        admission must expand to turns 2-6."""
        index = LexicalIndex("patient-expand", window=2)
        index.build(expansion_conversation)
        results = index.search("anaphylaxis amoxicillin", k=1)
        top = results[0]
        assert top.turn_ids == [2, 3, 4, 5, 6]
        for n in (2, 3, 4, 5, 6):
            assert f"turn {n}" in top.text
        # The expansion window is strictly wider than the single hit turn.
        assert len(top.turn_ids) > 1

    def test_expansion_never_crosses_an_admission_boundary(self, expansion_conversation: Conversation) -> None:
        index = LexicalIndex("patient-expand", window=2)
        index.build(expansion_conversation)
        results = index.search("anaphylaxis amoxicillin", k=1)
        top = results[0]
        assert top.session_id == "adm-allergy"
        assert all(1 <= t <= 7 for t in top.turn_ids), "expansion leaked outside the 7-turn admission"

    def test_expansion_clamps_at_the_available_window(self, edge_admission_conversation: Conversation) -> None:
        index = LexicalIndex("patient-edge", window=2)
        index.build(edge_admission_conversation)
        results = index.search("recurring migraines with visual aura", k=1)
        top = results[0]
        # Hit is turn 1 of a 3-turn admission: +/-2 would ask for turns
        # -1..3, clamped to the turns that actually exist, 1..3.
        assert top.turn_ids == [1, 2, 3]

    def test_scores_are_descending(self, expansion_conversation: Conversation) -> None:
        index = LexicalIndex("patient-expand")
        index.build(expansion_conversation)
        results = index.search("antibiotics allergic reaction", k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_empty_patient_returns_empty_list(self, empty_conversation: Conversation) -> None:
        index = LexicalIndex("patient-empty")
        index.build(empty_conversation)
        assert index.search("anything at all", k=5) == []

    def test_save_load_round_trip(self, expansion_conversation: Conversation, tmp_path: Path) -> None:
        built = LexicalIndex("patient-expand", window=2)
        built.build(expansion_conversation)
        built.save(tmp_path)

        loaded = LexicalIndex.load("patient-expand", tmp_path)
        query = "anaphylaxis amoxicillin"
        before = built.search(query, k=3)
        after = loaded.search(query, k=3)
        assert [(r.text, r.turn_ids, round(r.score, 6)) for r in before] == [
            (r.text, r.turn_ids, round(r.score, 6)) for r in after
        ]

    def test_load_missing_index_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            LexicalIndex.load("no-such-patient", tmp_path)


# ---------------------------------------------------------------------------
# pyproject.toml — no ANN dependency, ever (E5-S2 AC4)
# ---------------------------------------------------------------------------


def test_pyproject_has_no_ann_library() -> None:
    text = Path("pyproject.toml").read_text(encoding="utf-8")
    assert "numpy" in text
    assert "bm25s" in text
    for banned in ("faiss", "hnswlib", "annoy", "scann"):
        assert banned not in text.lower()


# ---------------------------------------------------------------------------
# Real-corpus tier — skipped, not failed, if the corpus was never fetched.
# Reports the story's required real build time / index size / p50-p90
# latency numbers for one real patient.
# ---------------------------------------------------------------------------

_real_corpus_skip = pytest.mark.skipif(
    not CORPUS_DIR.is_dir(),
    reason=f"real MedLoCoMo corpus not found at {CORPUS_DIR}",
)


def _percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(len(ordered) * pct))
    return ordered[idx]


def _measure_latencies_ms(search_fn, questions: list[str], k: int = 5) -> list[float]:
    latencies = []
    for question in questions:
        start = time.perf_counter()
        search_fn(question, k)
        latencies.append((time.perf_counter() - start) * 1000.0)
    return latencies


@_real_corpus_skip
class TestRealPatientChannels:
    """One real MedLoCoMo patient, both channels. `list_patients()[0]` is
    deterministic (the loader sorts subject ids), so this test targets the
    same patient on every run rather than a random one. Reports real
    numbers via `print` (visible under `pytest -v -s`) and asserts only
    generous, non-flaky sanity bounds — the actual measured numbers belong
    in the story's return note, not in a tight CI threshold that would
    make this test hostage to whichever machine runs it."""

    @staticmethod
    @pytest.fixture(scope="class")
    def patient_id() -> str:
        return list_patients()[0]

    @staticmethod
    @pytest.fixture(scope="class")
    def conversation(patient_id: str) -> Conversation:
        return load_conversation(patient_id)

    @staticmethod
    @pytest.fixture(scope="class")
    def sample_questions(patient_id: str) -> list[str]:
        qas = load_qa(patient_id)
        return [qa["question"] for qa in qas[:30]] or ["how has the patient been feeling"]

    def test_vector_build_search_and_report(
        self, patient_id: str, conversation: Conversation, sample_questions: list[str], tmp_path: Path
    ) -> None:
        index = PatientIndex(patient_id, cache_path=tmp_path / "embed_cache.npz")
        index.build(conversation)
        assert index.build_time_s > 0.0
        assert index.index_size_bytes() > 0
        assert len(index.units) == conversation.n_turns

        latencies = _measure_latencies_ms(index.search, sample_questions, k=5)
        p50, p90 = _percentile(latencies, 0.5), _percentile(latencies, 0.9)
        print(
            f"\n[vector] patient={patient_id} n_turns={conversation.n_turns} "
            f"n_units={len(index.units)} build_time_s={index.build_time_s:.4f} "
            f"index_size_bytes={index.index_size_bytes()} "
            f"p50_ms={p50:.4f} p90_ms={p90:.4f} n_queries={len(latencies)}"
        )
        # Generous bounds: literature/12 measured 0.168 ms/query at
        # n=3,000/dim=1024 on comparable CPU hardware [R-VEC-009]; this
        # patient's corpus is smaller. 500 ms is two-plus orders of
        # magnitude of headroom over that measurement, not a tight bound.
        assert p50 < 500.0
        assert index.build_time_s < 30.0

    def test_lexical_build_search_and_report(
        self, patient_id: str, conversation: Conversation, sample_questions: list[str]
    ) -> None:
        index = LexicalIndex(patient_id)
        index.build(conversation)
        assert index.build_time_s > 0.0
        assert index.index_size_bytes() > 0
        assert len(index.units) == conversation.n_turns

        latencies = _measure_latencies_ms(index.search, sample_questions, k=5)
        p50, p90 = _percentile(latencies, 0.5), _percentile(latencies, 0.9)
        print(
            f"\n[lexical] patient={patient_id} n_turns={conversation.n_turns} "
            f"n_units={len(index.units)} build_time_s={index.build_time_s:.4f} "
            f"index_size_bytes={index.index_size_bytes()} "
            f"p50_ms={p50:.4f} p90_ms={p90:.4f} n_queries={len(latencies)}"
        )
        assert p50 < 500.0
        assert index.build_time_s < 30.0

    def test_real_patient_save_load_round_trip(self, patient_id: str, conversation: Conversation, tmp_path: Path) -> None:
        vector_index = PatientIndex(patient_id, cache_path=tmp_path / "embed_cache.npz")
        vector_index.build(conversation)
        vector_index.save(tmp_path)
        loaded_vector = PatientIndex.load(patient_id, tmp_path, cache_path=tmp_path / "embed_cache.npz")

        lexical_index = LexicalIndex(patient_id)
        lexical_index.build(conversation)
        lexical_index.save(tmp_path)
        loaded_lexical = LexicalIndex.load(patient_id, tmp_path)

        query = conversation.turns()[len(conversation.turns()) // 2].text
        assert vector_index.search(query, k=3) == loaded_vector.search(query, k=3)
        assert lexical_index.search(query, k=3) == loaded_lexical.search(query, k=3)
