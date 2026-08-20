"""pipeline/ingest.py — the one-patient ingest entry point.

Design authority: `collaborative/design/stories/E4/E4-S4.md` ("Files to
touch": "pipeline/ingest.py — the one-patient entry that calls
`assert_handcheck_passed()` then extract -> normalize -> resolve -> write.
Multi-patient loops belong after E2-S3 is green.") and
`collaborative/decisions/005-audit-findings-and-epsilon-fix.md` Finding 2
(the reconciliation audit that found no production code called `write_facts`
or `write_and_invalidate` at all — only tests did — so, until this module
existed, there was no ingest entry point of ANY kind, gated or not).

**`assert_handcheck_passed()` runs first, unconditionally, on every call** —
before a single extract/normalize/resolve/write step happens for the one
patient this call is scoped to. `pipeline.scale_gate`'s own docstring:
default is BLOCKED; this module never works around that, never catches
`ScaleGateError`, and never writes `fixtures/handcheck/PASSED` itself — a
blocked gate means `ingest_patient()` raises and nothing downstream runs.

Pipeline shape, one patient at a time (ARCHITECTURE.md §6):

    load_conversation -> extract_facts (normalize.resolve_time /
    canonicalize_predicate already run INSIDE extract.py's own handling
    rules — see that module's docstring; there is no separate top-level
    "normalize" call here) -> resolve.attach_canonical_ids (entity
    resolution) -> graph.invalidate.write_and_invalidate (write + the
    invalidation pass, NOT the bare `graph.writer.write_facts` — Finding 2's
    own instruction: invalidation must actually run in the product path,
    not only in tests, so this module wires the wrapper that already exists
    for exactly this purpose rather than reaching into `writer.py` again).

**Multi-patient loops belong strictly after E2-S3's gate is green for
real.** This module deliberately exposes only `ingest_patient(subject_id,
...)` — a single subject per call. A caller wanting corpus-scale ingest
composes its own loop over `pipeline.loader.list_patients()` and calls this
function once per subject_id; `assert_handcheck_passed()` re-runs on every
one of those calls (a cheap filesystem check, not a one-time bypass), so
looping around this function cannot silently skip the gate either.

`now` (the ingest/system-clock reading `write_and_invalidate` needs for
every `invalidated_at`/`observed_at` its own pass writes) is a required,
caller-supplied keyword-only argument, never defaulted to a hidden
`datetime.now()` call inside this module — the same "no wall-clock read
buried in a callee" discipline `graph/invalidate.py`/ARCHITECTURE §8 already
establish and every one of that module's own tests exercises (`now=`
supplied explicitly at every call site, `tests/test_invalidate.py`).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from medmemgraph.contracts import ClinicalFact
from medmemgraph.graph.invalidate import InvalidationReport, write_and_invalidate
from medmemgraph.graph.writer import WriteReport
from medmemgraph.hydra_client import HydraClient
from medmemgraph.pipeline.extract import Extractor, extract_facts
from medmemgraph.pipeline.ids import IdMap, IdMinter
from medmemgraph.pipeline.loader import Conversation, load_conversation
from medmemgraph.pipeline.resolve import CanonicalRegistry, Complete, attach_canonical_ids
from medmemgraph.pipeline.scale_gate import assert_handcheck_passed

__all__ = ["IngestReport", "ingest_patient"]


@dataclass
class IngestReport:
    """What one `ingest_patient()` call did, for a caller (a script, a demo,
    a test) to report or assert on without re-deriving it from the two
    sub-reports by hand."""

    subject_id: str
    facts: list[ClinicalFact]
    write_report: WriteReport
    invalidation_report: InvalidationReport

    @property
    def n_facts_extracted(self) -> int:
        return len(self.facts)

    @property
    def n_facts_written(self) -> int:
        return self.write_report.facts_written


def ingest_patient(
    subject_id: str,
    *,
    now: str,
    root: str | os.PathLike[str] | None = None,
    conversation: Conversation | None = None,
    client: HydraClient | None = None,
    extractor: Extractor | None = None,
    registry: CanonicalRegistry | None = None,
    resolve_complete: Complete | None = None,
    id_map: IdMap | IdMinter | None = None,
    batch_size: int = 1000,
) -> IngestReport:
    """Ingest ONE patient's whole conversation (every admission the loaded
    `Conversation` carries — never a second patient; this function has no
    parameter that could name one). Order, unconditionally:

    1. `pipeline.scale_gate.assert_handcheck_passed()` — raises
       `ScaleGateError` and runs nothing else if the gate is not green.
    2. `pipeline.loader.load_conversation(subject_id, root)` (unless an
       already-loaded `conversation` is supplied — the injection point for
       tests and for a caller that already has one, e.g. a multi-admission
       incremental caller that loaded it once for other purposes too).
    3. `pipeline.extract.extract_facts(conversation, extractor=extractor)`
       — every admission, one LLM call per admission (`extract.py`'s own
       contract); `extractor` defaults to `Extractor()` (real inference —
       see that class's own "no dry_run -> real inference" docstring; pass
       `Extractor(dry_run=True)` explicitly for an offline/CI run).
    4. `pipeline.resolve.attach_canonical_ids(facts, registry=registry,
       complete=resolve_complete, id_map=id_map)` — entity resolution,
       rewriting `subject.canonical_id`/`object.canonical_id` on every fact
       in place.
    5. `graph.invalidate.write_and_invalidate(client, facts, id_map,
       now=now, batch_size=batch_size)` — writer + invalidation pass, run
       as the one integration point ARCHITECTURE §6.5 documents as one
       sequence (decisions/005 Finding 2: this is the wiring that was
       missing everywhere in product code before this module existed).

    `client=None` (the common case) opens a short-lived `HydraClient(
    transport="bolt")` for the duration of this one call and closes it in a
    `finally`, mirroring `graph/retrieve.py`'s own `_open_client()`
    convention; pass an already-open `client` to reuse one across several
    `ingest_patient()` calls instead.

    `id_map`/`registry` are the two pieces of cross-run state E3-S1/E4-S4
    document as the caller's job to persist for idempotent replay and
    incremental entity resolution across admissions/patients — this
    function accepts them but does not itself load or save them from disk;
    a corpus-scale caller's own loop is where that persistence belongs
    (module docstring: multi-patient looping is explicitly out of this
    module's scope).
    """
    assert_handcheck_passed()

    if conversation is None:
        conversation = load_conversation(subject_id, root)

    facts = extract_facts(conversation, extractor=extractor)

    attach_canonical_ids(
        facts,
        registry=registry,
        complete=resolve_complete,
        id_map=id_map,
    )

    owns_client = client is None
    active_client = client if client is not None else HydraClient(transport="bolt")
    try:
        write_report, invalidation_report = write_and_invalidate(
            active_client, facts, id_map, now=now, batch_size=batch_size
        )
    finally:
        if owns_client:
            active_client.close()

    return IngestReport(
        subject_id=subject_id,
        facts=facts,
        write_report=write_report,
        invalidation_report=invalidation_report,
    )
