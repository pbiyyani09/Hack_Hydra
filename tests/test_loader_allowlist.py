"""tests/test_loader_allowlist.py — the enforcement point for decisions/001.

The loader must open exactly one path per patient per function: never a
glob, never a directory listing inside a patient dir. This suite doesn't
settle for asserting return values; it installs a guard that makes the
failure mode (a `glob("*/*/*.json")` accidentally picking up
`formed_packet.json`) impossible to pass silently: every path opened via
`builtins.open` *and* `pathlib.Path.open` during a load is recorded, and the
recorded set is asserted to contain only the allowed filename.

Runs against the real MedLoCoMo corpus (skipped if not present).
"""

from __future__ import annotations

import builtins
import json
import os
import pathlib
from pathlib import Path

import pytest

from medmemgraph.pipeline.loader import (
    ALLOWED_EVAL_FILENAME,
    ALLOWED_INGEST_FILENAME,
    Conversation,
    LoaderError,
    list_patients,
    load_conversation,
    load_qa,
)

# Named explicitly in decisions/001 and the story: every one of these is a
# per-admission or per-patient artifact generated *from* (or feeding into)
# formed_packet.json, and every one is out of bounds for ingestion/eval.
DISQUALIFIED_FILENAMES = frozenset(
    {
        "formed_packet.json",
        "qa.json",
        "summary.json",
        "prompt_record.json",
        "patient_summary.json",
        "cross_admission_qa.json",
    }
)

MEDLOCOMO_ROOT = os.environ.get("MEDLOCOMO_ROOT", "data/medlocomo")
CORPUS_DIR = Path(MEDLOCOMO_ROOT) / "MedLoCoMo"

pytestmark = pytest.mark.skipif(
    not CORPUS_DIR.is_dir(),
    reason=f"real MedLoCoMo corpus not found at {CORPUS_DIR}",
)


class OpenGuard:
    """Records every path opened via builtins.open / pathlib.Path.open.

    io.open and builtins.open are the same object at interpreter start, but
    pathlib.Path.open holds its own reference to io.open — patching
    builtins.open alone does not intercept `some_path.open()`. Patch both,
    per the story, so the guard catches either calling convention.
    """

    def __init__(self) -> None:
        self.opened: list[Path] = []
        self._real_builtin_open = builtins.open
        self._real_path_open = pathlib.Path.open

    def __enter__(self) -> "OpenGuard":
        guard = self

        def fake_builtin_open(file, *args, **kwargs):
            guard.opened.append(Path(os.fspath(file)))
            return guard._real_builtin_open(file, *args, **kwargs)

        def fake_path_open(self_path, *args, **kwargs):
            guard.opened.append(Path(self_path))
            return guard._real_path_open(self_path, *args, **kwargs)

        builtins.open = fake_builtin_open
        pathlib.Path.open = fake_path_open
        return self

    def __exit__(self, *exc_info) -> bool:
        builtins.open = self._real_builtin_open
        pathlib.Path.open = self._real_path_open
        return False

    def filenames(self) -> set[str]:
        return {p.name for p in self.opened}


def _assert_no_disqualified_names(opened: set[str]) -> None:
    hit = opened & DISQUALIFIED_FILENAMES
    assert not hit, f"loader opened disqualified file(s): {sorted(hit)}"


# --------------------------------------------------------------------------
# Real corpus, full sweep — the loader is the correctness firewall for every
# one of these 101 patients, not just a sample.
# --------------------------------------------------------------------------


def _real_subject_ids() -> list[str]:
    return list_patients()


def test_list_patients_finds_the_full_real_corpus():
    subjects = _real_subject_ids()
    assert len(subjects) == 101
    assert len(subjects) == len(set(subjects))  # unique


def test_list_patients_opens_nothing():
    """Discovering subject ids must not open a single file (stat only)."""
    with OpenGuard() as guard:
        subjects = list_patients()
    assert subjects
    assert guard.opened == []


def test_load_conversation_allowlist_full_corpus():
    """Every real patient: load_conversation opens ONLY combined_conversation.json."""
    all_opened: set[str] = set()
    for subject_id in _real_subject_ids():
        with OpenGuard() as guard:
            convo = load_conversation(subject_id)
        assert isinstance(convo, Conversation)
        assert convo.n_turns > 0
        opened = guard.filenames()
        assert opened == {ALLOWED_INGEST_FILENAME}, (
            f"subject {subject_id}: opened {opened}, expected only "
            f"{{{ALLOWED_INGEST_FILENAME!r}}}"
        )
        all_opened |= opened
    assert all_opened == {ALLOWED_INGEST_FILENAME}
    _assert_no_disqualified_names(all_opened)


def test_load_qa_allowlist_full_corpus():
    """Every real patient: load_qa opens ONLY benchmark_qa.json."""
    all_opened: set[str] = set()
    for subject_id in _real_subject_ids():
        with OpenGuard() as guard:
            qas = load_qa(subject_id)
        assert isinstance(qas, list)
        assert len(qas) > 0
        opened = guard.filenames()
        assert opened == {ALLOWED_EVAL_FILENAME}, (
            f"subject {subject_id}: opened {opened}, expected only "
            f"{{{ALLOWED_EVAL_FILENAME!r}}}"
        )
        all_opened |= opened
    assert all_opened == {ALLOWED_EVAL_FILENAME}
    _assert_no_disqualified_names(all_opened)


def test_real_patient_dirs_already_contain_decoys_and_stay_safe():
    """Sanity: the real corpus already places `patient_summary.json` and
    `cross_admission_qa.json` directly beside the allowed files in every
    patient dir. This proves the allowlist holds even without a synthetic
    fixture, on day one, against the actual leakage surface.
    """
    subject_id = _real_subject_ids()[0]
    patient_dir = CORPUS_DIR / subject_id
    present_decoys = {
        name for name in DISQUALIFIED_FILENAMES if (patient_dir / name).is_file()
    }
    assert present_decoys, "expected real corpus to carry >=1 decoy file per patient"

    with OpenGuard() as guard:
        load_conversation(subject_id)
        load_qa(subject_id)

    _assert_no_disqualified_names(guard.filenames())


# --------------------------------------------------------------------------
# Synthetic fixture: a patient dir with formed_packet.json and every other
# disqualified name sitting directly next to the allowed files (the exact
# shape decisions/001 warns a naive glob would pick up). Content is
# deliberately invalid JSON — a second, independent tripwire: if the loader
# ever opened one of these, parsing it would raise too.
# --------------------------------------------------------------------------

_VALID_CONVERSATION = {
    "subject_id": "DECOY0001",
    "processed_hadm_ids": ["H1"],
    "admissions": [
        {
            "hadm_id": "H1",
            "admission_start": "2124-01-01 00:00:00",
            "admission_end": "2124-01-02 00:00:00",
            "conversation_lines": [
                {
                    "turn_number": 1,
                    "time": "2124-01-01 00:00:00",
                    "speaker": "Doctor",
                    "text": "How are you feeling today?",
                },
                {
                    "turn_number": 2,
                    "time": "2124-01-01 00:05:00",
                    "speaker": "Patient",
                    "text": "Better, thank you.",
                },
            ],
        }
    ],
}

_VALID_QA = {
    "qas": [
        {
            "qa_id": "DECOY0001_q01",
            "scope": "single_admission",
            "question_type": "medical_reasoning",
            "question": "Why was the patient admitted?",
            "answer": "observation",
            "evidence": {"admissions": ["H1"], "turn_ids": [1]},
        }
    ]
}


@pytest.fixture
def decoy_patient_dir(tmp_path: Path) -> Path:
    subject_id = "DECOY0001"
    patient_dir = tmp_path / "MedLoCoMo" / subject_id
    patient_dir.mkdir(parents=True)

    (patient_dir / ALLOWED_INGEST_FILENAME).write_text(json.dumps(_VALID_CONVERSATION))
    (patient_dir / ALLOWED_EVAL_FILENAME).write_text(json.dumps(_VALID_QA))
    for decoy_name in DISQUALIFIED_FILENAMES:
        (patient_dir / decoy_name).write_text("{ this is not valid json, poison }")

    return tmp_path


def test_decoy_files_beside_the_allowed_ones_are_never_opened(decoy_patient_dir):
    with OpenGuard() as guard:
        convo = load_conversation("DECOY0001", root=decoy_patient_dir)
        qas = load_qa("DECOY0001", root=decoy_patient_dir)

    assert convo.subject_id == "DECOY0001"
    assert convo.n_turns == 2
    assert len(qas) == 1

    opened = guard.filenames()
    assert opened == {ALLOWED_INGEST_FILENAME, ALLOWED_EVAL_FILENAME}
    _assert_no_disqualified_names(opened)


def test_list_patients_ignores_a_decoy_only_directory(tmp_path: Path):
    """A subdirectory with only disqualified files (no combined_conversation.json)
    must not be reported as an ingestible patient."""
    corpus = tmp_path / "MedLoCoMo"
    good_dir = corpus / "GOOD001"
    good_dir.mkdir(parents=True)
    (good_dir / ALLOWED_INGEST_FILENAME).write_text(json.dumps(_VALID_CONVERSATION))

    decoy_only_dir = corpus / "DECOYONLY"
    decoy_only_dir.mkdir(parents=True)
    (decoy_only_dir / "formed_packet.json").write_text("{ poison }")

    subjects = list_patients(root=tmp_path)
    assert subjects == ["GOOD001"]


# --------------------------------------------------------------------------
# Structural checks (not the point of this file, but cheap and load-bearing:
# a loader that silently mis-shapes data is as dangerous as one that leaks).
# --------------------------------------------------------------------------


def test_load_conversation_shape_matches_contract():
    subject_id = _real_subject_ids()[0]
    convo = load_conversation(subject_id)

    assert convo.subject_id == subject_id
    assert convo.admissions
    assert convo.session_ids() == [a.hadm_id for a in convo.admissions]
    assert convo.n_turns == len(convo.turns())
    assert convo.token_estimate() > 0

    turn = convo.turns()[0]
    assert turn.speaker in {"Doctor", "Patient"}
    assert isinstance(turn.turn_number, int)
    assert isinstance(turn.time, str)
    assert isinstance(turn.text, str)


def test_load_qa_shape_matches_contract():
    subject_id = _real_subject_ids()[0]
    qas = load_qa(subject_id)

    assert qas
    expected_scopes = {"single_admission", "cross_admission"}
    expected_question_types = {
        "medical_reasoning",
        "care_plan_rationale",
        "longitudinal_progression",
        "cross_admission_comparison",
        "frequency_pattern",
        "adversarial",
    }
    for item in qas:
        assert item["scope"] in expected_scopes
        assert item["question_type"] in expected_question_types
        assert "qa_id" in item and "question" in item and "answer" in item
        assert "admissions" in item["evidence"]


def test_missing_patient_raises_loader_error(tmp_path: Path):
    with pytest.raises(LoaderError):
        load_conversation("NO_SUCH_PATIENT", root=tmp_path)


def test_pair_builder_on_one_train_patient_opens_only_allowlisted(tmp_path: Path):
    """FT-E1-S2: pair builder corpus reads stay on the two allowlisted names."""
    from medmemgraph.eval.rerank_pairs import build_pairs
    from medmemgraph.eval.rerank_split import EVAL_TRIO

    train_id = next(s for s in _real_subject_ids() if s not in EVAL_TRIO)

    class _EmptyIndex:
        def build(self, conversation, facts=None) -> None:
            del conversation, facts

        def search(self, query: str, k: int):
            del query, k
            return []

    split = {"eval": list(EVAL_TRIO), "dev": [], "train": [train_id]}
    with OpenGuard() as guard:
        build_pairs(
            split,
            out_dir=tmp_path,
            index_factory=lambda sid: _EmptyIndex(),
            only_subjects=[train_id],
        )
    opened = guard.filenames()
    _assert_no_disqualified_names(opened)
    corpus_names = {p.name for p in guard.opened if "MedLoCoMo" in p.parts}
    assert corpus_names <= {ALLOWED_INGEST_FILENAME, ALLOWED_EVAL_FILENAME}
    assert ALLOWED_INGEST_FILENAME in corpus_names
    assert ALLOWED_EVAL_FILENAME in corpus_names
