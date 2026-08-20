"""pipeline/scale_gate.py — the E2-S3 hand-check gate before scale ingest.

Design authority: `collaborative/design/stories/E2/E2-S3.md`, PHASES.md's own
risk-table row ("Hand-check ~30 extracted facts on day 1 against the source
turns. Do not scale ingestion before that passes."), and
`collaborative/decisions/005-audit-findings-and-epsilon-fix.md` Finding 3
(the reconciliation audit that found this whole gate missing).

This is the one function every corpus-scale entry point must call before
opening a second patient (E2-S3's own words): `assert_handcheck_passed()`.
**Default is BLOCKED** — any missing artifact (the `PASSED` marker, the
`facts.jsonl` sidecar, or a `facts.jsonl` row that fails
`contracts.validate()`) fails closed with a typed `ScaleGateError`, never a
silent pass. `fixtures/handcheck/PASSED` is a one-line file a **human** (or
the Evidence owner acting as second reader) writes only after filling
`CHECKLIST.md`'s `human: ok/bad` column — see that file's own instructions.

**This module never writes `PASSED` itself, under any circumstance.** That
is the entire point of the gate (E2-S3's own "Banned approaches": "Auto-
passing the gate from the extractor"); a coding agent writing that file
would defeat the one thing this module exists to guarantee. `scripts/
handcheck_extract.py` (which regenerates `facts.jsonl` / the `CHECKLIST.md`
skeleton) enforces the same discipline on its own side — see that script's
own docstring.

`handcheck_dir` is accepted on every function below (defaulting to the real
repo `fixtures/handcheck/`) purely so tests can point the gate at an
isolated `tmp_path` fixture instead of the real one — `tests/
test_scale_gate.py`'s own hard rule is "no test may create `PASSED` as a
side effect that leaks", which requires exactly this seam.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from medmemgraph.contracts import ClinicalFact, EntityRef, SENTINEL_VALID_TO, validate

__all__ = [
    "ScaleGateError",
    "MIN_FACTS",
    "PASSED_FILENAME",
    "FACTS_FILENAME",
    "CHECKLIST_FILENAME",
    "DEFAULT_HANDCHECK_DIR",
    "fact_from_dict",
    "load_facts_jsonl",
    "assert_handcheck_passed",
]

# `src/medmemgraph/pipeline/scale_gate.py` -> repo root is three levels up
# (pipeline/ -> medmemgraph/ -> src/ -> repo root) — same
# `Path(__file__).resolve().parents[N]` convention `llm.py`/`observability.py`
# already use in this codebase (they are one level shallower, hence `[2]`
# there vs `[3]` here).
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_HANDCHECK_DIR = REPO_ROOT / "fixtures" / "handcheck"

PASSED_FILENAME = "PASSED"
FACTS_FILENAME = "facts.jsonl"
CHECKLIST_FILENAME = "CHECKLIST.md"

MIN_FACTS = 30
"""E2-S3's own contract: "facts.jsonl has >= 30 valid ClinicalFact rows"."""


class ScaleGateError(RuntimeError):
    """Raised by `assert_handcheck_passed()` when the gate is not green.
    Every raise site below names exactly which artifact is missing/invalid
    and what a human needs to do about it — this is a blocking safety gate,
    not an assertion a caller is expected to silently retry past."""


def _resolve_dir(handcheck_dir: str | os.PathLike[str] | None) -> Path:
    return Path(handcheck_dir) if handcheck_dir is not None else DEFAULT_HANDCHECK_DIR


def fact_from_dict(row: dict) -> ClinicalFact:
    """Inverse of `dataclasses.asdict(fact)` — the shape
    `scripts/handcheck_extract.py` serializes each `facts.jsonl` line with.
    Mirrors `pipeline.extract._fact_from_dict`'s shape exactly (kept as an
    independent, public copy here rather than importing that module's
    private helper — this module has no other reason to depend on
    `pipeline.extract` at all, and the two shapes are a frozen contract
    (`contracts.ClinicalFact`), not something that should drift)."""
    return ClinicalFact(
        fact_id=row["fact_id"],
        patient_id=row["patient_id"],
        session_id=row["session_id"],
        turn_ids=list(row["turn_ids"]),
        subject=EntityRef(**row["subject"]),
        predicate=row["predicate"],
        object=EntityRef(**row["object"]),
        valid_from=row["valid_from"],
        valid_to=row.get("valid_to", SENTINEL_VALID_TO),
        observed_at=row.get("observed_at", ""),
        polarity=row.get("polarity", "asserted"),
        source_class=row.get("source_class", "patient"),
        confidence=row.get("confidence", 0.0),
    )


def load_facts_jsonl(path: str | os.PathLike[str]) -> list[ClinicalFact]:
    """One `ClinicalFact` per non-blank line of a JSON-lines file. Raises
    `ScaleGateError` (not a raw `json.JSONDecodeError`/`KeyError`) naming the
    offending line, so a malformed `facts.jsonl` fails the gate loudly
    rather than crashing some caller's stack trace with no context."""
    path = Path(path)
    facts: list[ClinicalFact] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ScaleGateError(f"scale gate BLOCKED: could not read {path}: {exc}") from exc

    for lineno, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ScaleGateError(f"scale gate BLOCKED: {path} line {lineno} is not valid JSON: {exc}") from exc
        try:
            facts.append(fact_from_dict(row))
        except (KeyError, TypeError) as exc:
            raise ScaleGateError(
                f"scale gate BLOCKED: {path} line {lineno} does not match the ClinicalFact shape: {exc}"
            ) from exc
    return facts


def assert_handcheck_passed(handcheck_dir: str | os.PathLike[str] | None = None) -> None:
    """Raises `ScaleGateError` unless ALL of the following hold; returns
    (does nothing) only when every one does:

    1. `<handcheck_dir>/PASSED` exists (a human-written marker — see module
       docstring; never written by this function or any other code path).
    2. `<handcheck_dir>/facts.jsonl` exists and parses as JSON-lines of
       `ClinicalFact`-shaped rows.
    3. It has at least `MIN_FACTS` (30) rows.
    4. Every row passes `contracts.validate()` — E2-S3 AC2's own carve-out
       ("after mint, canonical_id may still be 0 — allow 0 in this gate;
       reject negative / bad polarity / bad predicate") requires no special
       casing here: `validate()` already treats `canonical_id == 0` as
       clean and only rejects a *negative* one, alongside its other checks
       (closed predicate vocabulary, binary polarity, non-empty turn_ids,
       valid_from <= valid_to, non-null valid_to) — the exact rule E2-S3
       asks for, verbatim, already lives in the one frozen contract module
       rather than being re-implemented (and risking drift) here.

    `handcheck_dir` defaults to the real repo `fixtures/handcheck/` — pass
    an explicit path (a `tmp_path` fixture, in tests) to point the gate at
    an isolated directory instead.
    """
    directory = _resolve_dir(handcheck_dir)

    passed_path = directory / PASSED_FILENAME
    if not passed_path.is_file():
        raise ScaleGateError(
            f"scale gate BLOCKED: {passed_path} does not exist. E2-S3's hand-check "
            f"gate is not green — a human (or the Evidence owner acting as second "
            f"reader) must review {directory / CHECKLIST_FILENAME}'s real fact rows "
            f"against their source turns, fill the `human: ok/bad` column, and only "
            f"then write {PASSED_FILENAME} themselves. No coding agent may write "
            f"that file (E2-S3 'Banned approaches': 'Auto-passing the gate from the "
            f"extractor'). Scale ingest stays blocked until it exists."
        )

    facts_path = directory / FACTS_FILENAME
    if not facts_path.is_file():
        raise ScaleGateError(
            f"scale gate BLOCKED: {passed_path} exists but {facts_path} does not — "
            f"run `scripts/handcheck_extract.py` to (re)generate it before re-checking "
            f"the gate."
        )

    facts = load_facts_jsonl(facts_path)

    if len(facts) < MIN_FACTS:
        raise ScaleGateError(
            f"scale gate BLOCKED: {facts_path} has {len(facts)} row(s), need >= "
            f"{MIN_FACTS} (E2-S3's own contract)."
        )

    problems_by_fact: list[tuple[str, list[str]]] = []
    for fact in facts:
        problems = validate(fact)
        if problems:
            problems_by_fact.append((fact.fact_id, problems))

    if problems_by_fact:
        detail = "; ".join(f"{fid!r}: {', '.join(p)}" for fid, p in problems_by_fact[:5])
        more = "" if len(problems_by_fact) <= 5 else f" (+{len(problems_by_fact) - 5} more)"
        raise ScaleGateError(
            f"scale gate BLOCKED: {len(problems_by_fact)}/{len(facts)} row(s) in "
            f"{facts_path} fail contracts.validate(): {detail}{more}"
        )
