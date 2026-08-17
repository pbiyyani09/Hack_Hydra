"""graph/writer.py — parameterized `UNWIND` writer for minted `ClinicalFact`s
(ARCHITECTURE.md §6.5, decisions/003, stories/E4/E4-S4.md).

Write order for one batch (ARCHITECTURE §6.5): vertices first (one label at
a time — `Patient`, `Admission`, `Claim`, then each domain-entity label),
then edges (`ADMITTED`, `HAS`, `ABOUT`). `SUPERSEDES` / `CONTRADICTS` and
`DRAWN_FROM`/`:Turn` (invalidation-by-closing, §6.6) are out of scope here —
E4-S5. `ClinicalFact` carries `turn_ids` (bare ints) but not the `speaker` /
`occurred_at` / `raw_text` a `:Turn` node requires (§5.1); writing a `:Turn`
node without them would leave a contract-violating stub, so this module does
not create `:Turn` nodes or `DRAWN_FROM` edges — a Pipeline story that also
has the source `Conversation` object is the natural owner of that edge.

--------------------------------------------------------------------------
Executed-verified live-engine correction to ARCHITECTURE.md §6.5's own copy
(dev-backend, 2026-08-16 — see `tests/test_writer.py` module docstring for
the full reproduction):

  ARCHITECTURE.md §6.5 documents `MERGE (n:Claim {id: row.vertex}) SET ...`
  (label INSIDE the MERGE pattern). Spiked live against
  `ghcr.io/hydra-db/hydradb:0.1.1`, that shape is REJECTED:
  `CypherSyntaxError('... UNWIND vertex upsert MERGE pattern matches only
  id; apply labels with SET')`. The engine's actual required shape puts the
  label on the SET, not the MERGE pattern:

      UNWIND $rows AS row MERGE (n {id: row.vertex}) SET n:Claim, n.prop = row.prop, ...

  This looks like exactly the unlabelled node pattern decisions/003 bans —
  and it *would* be, on a MATCH/read path (the phantom-row bug). But this is
  a MERGE, and MERGE's unlabelled-by-id lookup is a distinct, engine
  special-cased "vertex upsert" grammar production (confirmed by the
  engine's own error strings), not the general MATCH parser the phantom-row
  bug lives in — verified live for both create-when-absent and
  idempotent-merge-when-present. `hydra_client.py`'s dialect gate was
  extended with a narrow, evidence-scoped exception for exactly this shape
  (`_check_unlabelled_node` / `_VERTEX_UPSERT_MERGE_RE`); MATCH patterns are
  completely unaffected and remain fully guarded. This module always uses
  the corrected shape — do not "fix" it back to §6.5's literal text.

  The relationship-upsert shape (`MATCH (s:Label {id:...}), (d:Label
  {id:...}) MERGE (s)-[r:TYPE {id:...}]->(d) SET r.prop = ...`) *is* exactly
  as ARCHITECTURE.md §6.5 documents it and needed no correction — spiked
  live, works, and is idempotent.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

from medmemgraph import contracts
from medmemgraph.contracts import ClinicalFact
from medmemgraph.graph import schema
from medmemgraph.hydra_client import HydraClient
from medmemgraph.pipeline.ids import IdMap, IdMinter, mint_claim_id, mint_patient_id

MAX_BATCH_SIZE = 1000
"""Hard ceiling on rows per `UNWIND $rows` statement (ARCHITECTURE §6.5:
"One 1,000-row UNWIND is one commit"; writes serialize at ~225 stmt/s flat
in concurrency, so batching — not concurrency — is the lever)."""


# ---------------------------------------------------------------------------
# WriteReport
# ---------------------------------------------------------------------------


@dataclass
class WriteReport:
    """What one `write_facts` call did. No transactions exist (decisions/003,
    literature/10 §A2/§D4), so this also doubles as the "how far did we get"
    record a caller needs to decide whether a retry is safe — it always is,
    because every write in this module is `MERGE`-idempotent by minted id
    (§5.4): replaying the same fact list after a partial failure re-derives
    the same ids and re-applies the same `SET`s, never duplicating a node or
    edge."""

    facts_in: int = 0
    facts_written: int = 0
    facts_skipped: int = 0
    skipped: list[tuple[str, list[str]]] = field(default_factory=list)
    """`(fact_id, validate() problems)` for every input fact that failed
    `contracts.validate()` and was never sent to the engine."""

    nodes_written: int = 0
    edges_written: int = 0
    statements_issued: int = 0
    wall_clock_s: float = 0.0

    @property
    def rows_written(self) -> int:
        return self.nodes_written + self.edges_written

    @property
    def rows_per_sec(self) -> float:
        if self.wall_clock_s <= 0:
            return 0.0
        return self.rows_written / self.wall_clock_s


class WriterError(RuntimeError):
    """Raised when a batch statement fails against the engine after the
    dialect gate already accepted it (network error, an engine-side
    rejection, etc.). Carries the `WriteReport` accumulated up to the
    failure point so the caller can see exactly how much progress was made
    before deciding whether/how to retry — replaying the *same* `facts` list
    from scratch is always safe (see `WriteReport` docstring)."""

    def __init__(self, message: str, partial_report: WriteReport) -> None:
        super().__init__(message)
        self.partial_report = partial_report


# ---------------------------------------------------------------------------
# Batching helper
# ---------------------------------------------------------------------------


def _chunks(rows: list[dict[str, Any]], size: int) -> Iterator[list[dict[str, Any]]]:
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


# ---------------------------------------------------------------------------
# Cypher shapes — one label / one rel type per statement (ARCHITECTURE §6.5).
# ---------------------------------------------------------------------------


def _vertex_upsert_cypher(label: str, set_clauses: str) -> str:
    """`UNWIND $rows AS row MERGE (n {id: row.vertex}) SET n:<Label>, <set_clauses>`
    — the live-engine-verified shape (see module docstring). `label` is
    always drawn from `schema.LABELS` by the caller, never passed through
    from unvalidated input, so this is never an injection surface even
    though the label can't be a bind parameter (Cypher labels aren't
    parameterizable)."""
    return f"UNWIND $rows AS row MERGE (n {{id: row.vertex}}) SET n:{label}, {set_clauses}"


def _edge_upsert_cypher(rel_type: str, src_label: str, dst_label: str, set_clauses: str) -> str:
    """`UNWIND $rows AS row MATCH (s:<SrcLabel> {id:...}), (d:<DstLabel> {id:...})
    MERGE (s)-[r:<RelType> {id:...}]->(d) SET <set_clauses>` — exactly
    ARCHITECTURE.md §6.5's documented relationship-upsert shape, spiked live
    and confirmed to need no correction."""
    return (
        f"UNWIND $rows AS row "
        f"MATCH (s:{src_label} {{id: row.source_vertex}}), (d:{dst_label} {{id: row.destination_vertex}}) "
        f"MERGE (s)-[r:{rel_type} {{id: row.relationship_vertex}}]->(d) "
        f"SET {set_clauses}"
    )


_CLAIM_SET_CLAUSES = ", ".join(
    f"n.{prop} = row.{prop}" for prop in schema.CLAIM_PROPERTIES if prop != "id"
)
_PATIENT_SET_CLAUSES = "n.patient_id = row.patient_id"
_ADMISSION_SET_CLAUSES = "n.session_id = row.session_id, n.patient_id = row.patient_id"
_ENTITY_SET_CLAUSES = "n.name = row.name, n.type = row.type"
_EDGE_SET_CLAUSES = "r.observed_at = row.observed_at"


# ---------------------------------------------------------------------------
# write_facts
# ---------------------------------------------------------------------------


def write_facts(
    client: HydraClient,
    facts: list[ClinicalFact],
    id_map: IdMap | IdMinter | None = None,
    *,
    batch_size: int = 1000,
) -> WriteReport:
    """Write minted `ClinicalFact`s (plus their `Patient` / `Admission` /
    domain-entity nodes) to HydraDB with parameterized `UNWIND` batches.

    `id_map` (a plain dict or an `IdMinter`) is resolved ONCE into a single
    `pipeline.ids.IdMinter` and reused for every id minted during this call
    (Patient, Admission, Claim, and every `ADMITTED`/`HAS`/`ABOUT`
    relationship id) — `IdMinter` docstring: build one and call `.mint()`
    on it repeatedly, rather than a fresh one per key, so collision
    linear-probing (§5.4 point 3) sees every id minted in this batch, not
    just the ones minted through a single call. Pass a dict (or an
    `IdMinter` wrapping one) that the caller persists and reloads across
    process restarts for cross-run idempotent replay (§5.4 point 1:
    "persist the map in the Graph"); `None` mints a fresh in-memory map for
    this call only (idempotent *within* the call, the common case for a
    single ingest run or a test — determinism comes from the identity-key
    hash fold, not from cross-call state, so this only matters if two
    different identity keys in the same batch raw-fold to the same 63-bit
    value, astronomically unlikely at this project's scale).

    `subject.canonical_id` / `object.canonical_id` on each fact are used
    as-is for domain-entity node ids — this module never remints them
    (ARCHITECTURE §5.4/§6.4: "Mint in the Pipeline... Graph must not
    remint").

    Facts that fail `contracts.validate()` are skipped (recorded on the
    report, never sent to the engine) rather than raising — one malformed
    fact must not sink an otherwise-good batch. A fact is also skipped, for
    the same recorded-not-raised reason, if its `subject`/`object`
    `EntityRef.type` is not a recognized `schema` entity type (fails
    `schema.label_for`) — this can only happen if a caller bypasses
    Pipeline's own extraction/ER contract.

    Batches are capped at `min(batch_size, MAX_BATCH_SIZE)` rows per
    `UNWIND $rows` statement. No transactions exist (decisions/003); if a
    statement fails partway through the write order (vertices, then
    edges), this raises `WriterError` carrying the partial `WriteReport` —
    replaying the same `facts` list is always safe (every write here is
    `MERGE`-idempotent by minted id).
    """
    effective_batch_size = min(batch_size, MAX_BATCH_SIZE)
    if effective_batch_size <= 0:
        raise ValueError(f"write_facts: batch_size must be positive, got {batch_size}")

    minter: IdMinter = id_map if isinstance(id_map, IdMinter) else IdMinter(id_map)

    report = WriteReport(facts_in=len(facts))
    start = time.monotonic()

    # -- 1. Validate + assign ids ------------------------------------------
    patients: dict[int, dict[str, Any]] = {}
    admissions: dict[int, dict[str, Any]] = {}
    claims: dict[int, dict[str, Any]] = {}
    entities_by_label: dict[str, dict[int, dict[str, Any]]] = {}

    admitted_edges: dict[int, dict[str, Any]] = {}
    has_edges: dict[int, dict[str, Any]] = {}
    about_edges_by_label: dict[str, dict[int, dict[str, Any]]] = {}

    def _register_entity(entity: contracts.EntityRef) -> str | None:
        try:
            label = schema.label_for(entity.type)
        except ValueError as exc:
            return str(exc)
        if label in schema.DOMAIN_ENTITY_LABELS:
            entities_by_label.setdefault(label, {})[entity.canonical_id] = {
                "vertex": entity.canonical_id,
                "name": entity.name,
                "type": entity.type,
            }
        return None

    for fact in facts:
        problems = contracts.validate(fact)
        if problems:
            report.facts_skipped += 1
            report.skipped.append((fact.fact_id, problems))
            continue

        entity_error = _register_entity(fact.object)
        if entity_error is None and fact.subject.type != "Patient":
            entity_error = _register_entity(fact.subject)
        if entity_error is not None:
            report.facts_skipped += 1
            report.skipped.append((fact.fact_id, [entity_error]))
            continue

        patient_node_id = mint_patient_id(fact.patient_id, id_map=minter)
        admission_node_id = minter.mint(f"Admission|{fact.session_id}")
        claim_node_id = mint_claim_id(fact.fact_id, id_map=minter)

        patients[patient_node_id] = {"vertex": patient_node_id, "patient_id": fact.patient_id}
        admissions[admission_node_id] = {
            "vertex": admission_node_id,
            "session_id": fact.session_id,
            "patient_id": fact.patient_id,
        }

        row = contracts.to_row(fact)
        claims[claim_node_id] = {
            "vertex": claim_node_id,
            "fact_id": row["fact_id"],
            "patient_id": row["patient_id"],
            "session_id": row["session_id"],
            "predicate": row["predicate"],
            "polarity": row["polarity"],
            "source_class": row["source_class"],
            "confidence": row["confidence"],
            "valid_from": row["valid_from"],
            "valid_to": row["valid_to"],
            "observed_at": row["observed_at"],
            "invalidated_at": contracts.SENTINEL_VALID_TO,
            "status": "active",
        }

        admitted_edge_id = minter.mint(f"ADMITTED|{fact.patient_id}|{fact.session_id}")
        admitted_edges[admission_node_id] = {
            "source_vertex": patient_node_id,
            "destination_vertex": admission_node_id,
            "relationship_vertex": admitted_edge_id,
            "observed_at": fact.observed_at,
        }
        has_edge_id = minter.mint(f"HAS|{fact.fact_id}")
        has_edges[claim_node_id] = {
            "source_vertex": patient_node_id,
            "destination_vertex": claim_node_id,
            "relationship_vertex": has_edge_id,
            "observed_at": fact.observed_at,
        }

        object_edge_id = minter.mint(f"ABOUT|{fact.fact_id}|object")
        object_label = schema.label_for(fact.object.type)
        about_edges_by_label.setdefault(object_label, {})[object_edge_id] = {
            "source_vertex": claim_node_id,
            "destination_vertex": fact.object.canonical_id,
            "relationship_vertex": object_edge_id,
            "observed_at": fact.observed_at,
        }
        if fact.subject.type != "Patient":
            subject_edge_id = minter.mint(f"ABOUT|{fact.fact_id}|subject")
            subject_label = schema.label_for(fact.subject.type)
            about_edges_by_label.setdefault(subject_label, {})[subject_edge_id] = {
                "source_vertex": claim_node_id,
                "destination_vertex": fact.subject.canonical_id,
                "relationship_vertex": subject_edge_id,
                "observed_at": fact.observed_at,
            }

        report.facts_written += 1

    # -- 2. Write vertices, one label at a time ------------------------------
    def _write_vertex_batches(label: str, set_clauses: str, rows: Iterable[dict[str, Any]]) -> None:
        cypher = _vertex_upsert_cypher(label, set_clauses)
        _run_batches(client, cypher, list(rows), effective_batch_size, report, is_edge=False)

    _write_vertex_batches("Patient", _PATIENT_SET_CLAUSES, patients.values())
    _write_vertex_batches("Admission", _ADMISSION_SET_CLAUSES, admissions.values())
    _write_vertex_batches("Claim", _CLAIM_SET_CLAUSES, claims.values())
    for label, entities in entities_by_label.items():
        _write_vertex_batches(label, _ENTITY_SET_CLAUSES, entities.values())

    # -- 3. Write edges, one type/label-pair at a time -----------------------
    def _write_edge_batches(
        rel_type: str, src_label: str, dst_label: str, rows: Iterable[dict[str, Any]]
    ) -> None:
        cypher = _edge_upsert_cypher(rel_type, src_label, dst_label, _EDGE_SET_CLAUSES)
        _run_batches(client, cypher, list(rows), effective_batch_size, report, is_edge=True)

    _write_edge_batches("ADMITTED", "Patient", "Admission", admitted_edges.values())
    _write_edge_batches("HAS", "Patient", "Claim", has_edges.values())
    for dst_label, edges in about_edges_by_label.items():
        _write_edge_batches("ABOUT", "Claim", dst_label, edges.values())

    report.wall_clock_s = time.monotonic() - start
    return report


def _run_batches(
    client: HydraClient,
    cypher: str,
    rows: list[dict[str, Any]],
    batch_size: int,
    report: WriteReport,
    *,
    is_edge: bool,
) -> None:
    for batch in _chunks(rows, batch_size):
        if not batch:
            continue
        try:
            client.run(cypher, rows=batch)
        except Exception as exc:  # noqa: BLE001 — re-wrapped with partial progress, not swallowed
            raise WriterError(
                f"write_facts: statement failed after {report.statements_issued} prior "
                f"statement(s) succeeded ({report.rows_written} rows written so far); "
                f"replaying the same facts is safe (MERGE-idempotent by minted id). "
                f"Underlying error: {exc!r}",
                partial_report=report,
            ) from exc
        report.statements_issued += 1
        if is_edge:
            report.edges_written += len(batch)
        else:
            report.nodes_written += len(batch)
