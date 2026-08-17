"""MedLoCoMo loader — the ingestion firewall.

See ``collaborative/decisions/001-medlocomo-packet-leakage.md``. Each patient
directory in the MedLoCoMo release also carries ``formed_packet.json`` (and
several other per-admission artifacts) — the *generation source* the dialogue
and the benchmark QA were built from. Ingesting any of it means reading the
answer key: results would look excellent and mean nothing.

This module opens exactly two kinds of file, by an allowlisted filename, at a
path it constructs directly:

    <root>/MedLoCoMo/<subject_id>/combined_conversation.json   (ingestion)
    <root>/MedLoCoMo/<subject_id>/benchmark_qa.json             (evaluation)

It never globs and never lists a patient directory looking for candidate
JSON files. ``list_patients`` lists the *corpus root* (one level up, to
discover subject ids) — it never lists, nor opens anything, inside an
individual patient directory beyond stat-checking the one allowed filename.

``tests/test_loader_allowlist.py`` is the enforcement point: it monkeypatches
``builtins.open`` and ``pathlib.Path.open`` for the duration of a load and
asserts the only filenames ever opened are the two named above.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

__all__ = [
    "ALLOWED_INGEST_FILENAME",
    "ALLOWED_EVAL_FILENAME",
    "LoaderError",
    "Turn",
    "Admission",
    "Conversation",
    "load_conversation",
    "load_qa",
    "list_patients",
]

# --------------------------------------------------------------------------
# Allowlist. These are the ONLY two filenames this module is permitted to
# open, anywhere. A denylist fails open (a new leaked filename slips through
# silently); this allowlist fails closed. See decisions/001.
# --------------------------------------------------------------------------

ALLOWED_INGEST_FILENAME = "combined_conversation.json"
ALLOWED_EVAL_FILENAME = "benchmark_qa.json"

_ROOT_ENV_VAR = "MEDLOCOMO_ROOT"
_DEFAULT_ROOT = "data/medlocomo"
_CORPUS_DIRNAME = "MedLoCoMo"

# Closed vocabulary confirmed against the real corpus (101 patients, 167,896
# turns) — see the dev-data return note for the scan. A speaker outside this
# set is corpus corruption, not a value to pass through silently.
_VALID_SPEAKERS = frozenset({"Doctor", "Patient"})


class LoaderError(RuntimeError):
    """A patient artifact is missing, unreadable, or fails the shape check.

    Raised instead of letting a raw ``KeyError``/``FileNotFoundError``
    propagate, so failures name the subject/admission/turn responsible
    instead of failing silently or with an opaque trace.
    """


def _resolve_root(root: str | os.PathLike[str] | None) -> Path:
    """``root`` defaults to ``$MEDLOCOMO_ROOT`` or ``data/medlocomo``."""
    if root is not None:
        return Path(root)
    return Path(os.environ.get(_ROOT_ENV_VAR, _DEFAULT_ROOT))


def _patient_dir(subject_id: str, root: str | os.PathLike[str] | None) -> Path:
    return _resolve_root(root) / _CORPUS_DIRNAME / str(subject_id)


def _read_json(path: Path, *, subject_id: str) -> dict:
    """Open exactly ``path`` and parse it as JSON. No fallback candidates."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError as exc:
        raise LoaderError(
            f"no {path.name} for subject {subject_id!r} at {path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise LoaderError(f"malformed JSON in {path}: {exc}") from exc


# --------------------------------------------------------------------------
# Typed conversation access
# --------------------------------------------------------------------------


class Turn(NamedTuple):
    """One flattened, admission-anchored conversation line."""

    hadm_id: str
    turn_number: int
    time: str
    speaker: str
    text: str


@dataclass(frozen=True)
class Admission:
    """One hospital admission (a "session") within a patient's history."""

    hadm_id: str
    admission_start: str
    admission_end: str
    conversation_lines: tuple[dict, ...]

    def turns(self) -> list[Turn]:
        return [
            Turn(
                hadm_id=self.hadm_id,
                turn_number=line["turn_number"],
                time=line["time"],
                speaker=line["speaker"],
                text=line["text"],
            )
            for line in self.conversation_lines
        ]


@dataclass(frozen=True)
class Conversation:
    """Parsed ``combined_conversation.json`` for one patient."""

    subject_id: str
    processed_hadm_ids: tuple[str, ...]
    admissions: tuple[Admission, ...]

    def turns(self) -> list[Turn]:
        """Flat turn list across every admission, admission order preserved."""
        flat: list[Turn] = []
        for admission in self.admissions:
            flat.extend(admission.turns())
        return flat

    def session_ids(self) -> list[str]:
        """``hadm_id`` of each admission, in admission order (session == admission)."""
        return [admission.hadm_id for admission in self.admissions]

    @property
    def n_turns(self) -> int:
        return sum(len(admission.conversation_lines) for admission in self.admissions)

    def token_estimate(self) -> int:
        """Rough token count over all turn text (doctor + patient).

        Uses tiktoken's ``cl100k_base`` encoding when available. Falls back
        to a ``chars // 4`` heuristic if tiktoken's encoding cache can't be
        loaded (e.g. an offline sandbox) — an estimate should degrade, not
        raise, when its only failure mode is a missing local cache.
        """
        text = " ".join(turn.text for turn in self.turns())
        try:
            import tiktoken

            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            return len(text) // 4


def _parse_conversation(raw: dict, *, subject_id: str, path: Path) -> Conversation:
    try:
        file_subject_id = str(raw["subject_id"])
        processed_hadm_ids = tuple(str(h) for h in raw["processed_hadm_ids"])
        raw_admissions = raw["admissions"]
    except KeyError as exc:
        raise LoaderError(f"{path} missing required key {exc}") from exc

    if file_subject_id != str(subject_id):
        raise LoaderError(
            f"{path} claims subject_id {file_subject_id!r}, "
            f"expected {subject_id!r} (directory/content mismatch)"
        )

    admissions: list[Admission] = []
    for adm in raw_admissions:
        try:
            hadm_id = str(adm["hadm_id"])
            admission_start = adm["admission_start"]
            admission_end = adm["admission_end"]
            lines = adm["conversation_lines"]
        except KeyError as exc:
            raise LoaderError(f"{path} admission missing required key {exc}") from exc

        for line in lines:
            speaker = line.get("speaker")
            if speaker not in _VALID_SPEAKERS:
                raise LoaderError(
                    f"{path} admission {hadm_id} turn {line.get('turn_number')} "
                    f"has speaker {speaker!r}, expected one of {sorted(_VALID_SPEAKERS)}"
                )

        admissions.append(
            Admission(
                hadm_id=hadm_id,
                admission_start=admission_start,
                admission_end=admission_end,
                conversation_lines=tuple(lines),
            )
        )

    return Conversation(
        subject_id=file_subject_id,
        processed_hadm_ids=processed_hadm_ids,
        admissions=tuple(admissions),
    )


def load_conversation(
    subject_id: str, root: str | os.PathLike[str] | None = None
) -> Conversation:
    """Load a patient's dialogue. Opens exactly one path — never a glob.

    ``<root>/MedLoCoMo/<subject_id>/combined_conversation.json``
    """
    path = _patient_dir(subject_id, root) / ALLOWED_INGEST_FILENAME
    raw = _read_json(path, subject_id=subject_id)
    return _parse_conversation(raw, subject_id=subject_id, path=path)


def load_qa(subject_id: str, root: str | os.PathLike[str] | None = None) -> list[dict]:
    """Load a patient's benchmark QA. Opens exactly one path — never a glob.

    ``<root>/MedLoCoMo/<subject_id>/benchmark_qa.json``

    Returns the ``qas`` list verbatim: each item is
    ``{qa_id, scope, question_type, question, answer, evidence}`` where
    ``evidence`` is ``{admissions: [...], turn_ids: [...]?}`` (``turn_ids``
    is present on roughly half of items — gold gives 100% admission-level
    and 50% turn-level evidence coverage).
    """
    path = _patient_dir(subject_id, root) / ALLOWED_EVAL_FILENAME
    raw = _read_json(path, subject_id=subject_id)
    try:
        return raw["qas"]
    except KeyError as exc:
        raise LoaderError(f"{path} missing required key {exc}") from exc


def list_patients(root: str | os.PathLike[str] | None = None) -> list[str]:
    """Subject ids with an ingestible conversation, sorted.

    Lists the corpus root (``<root>/MedLoCoMo/``) to discover subject-id
    subdirectories — this is a directory listing at the *corpus* level, not
    inside any individual patient directory, and does not open any file
    (only a stat check that the allowed filename exists).
    """
    corpus_dir = _resolve_root(root) / _CORPUS_DIRNAME
    if not corpus_dir.is_dir():
        raise LoaderError(f"MedLoCoMo corpus directory not found at {corpus_dir}")

    patients = [
        entry.name
        for entry in corpus_dir.iterdir()
        if entry.is_dir() and (entry / ALLOWED_INGEST_FILENAME).is_file()
    ]
    return sorted(patients)
