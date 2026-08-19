"""graph/invalidate.py — invalidation-by-closing (ARCHITECTURE.md §6.6).

This is the mechanism the graph-native argument rests on: a superseded
:Claim is never deleted. Its interval is closed (`valid_to` set to the
successor's `valid_from`), its other properties are untouched, and a
`SUPERSEDES`/`CONTRADICTS` edge records why. "Show me the chain of updates
to this fact, with the evidence for each transition" is then a path query —
something a vector store has no structural way to answer.

Grounded in:
  - ARCHITECTURE.md §6.6 — the decision procedure this module implements.
  - literature/05 (temporal knowledge graphs) Q3/Q4 — interval mechanics,
    the functional/set-valued split, and the update-vs-correction language
    this module's `Decision.kind` reuses.
  - literature/09 (conflict resolution & provenance) Q1 — the seven-step
    precedence ladder, and its own base-rate warning (DECODE: 93% balanced-
    benchmark accuracy collapses to 23.94% precision at realistic base
    rates) for why "flag unresolved" is a first-class output, not a bug.
  - decisions/002 (OPEN) — polarity stays binary on the wire; a `possible`
    fact must not silently close a `present` one. Since there is no
    six-class field to read, this module uses `ClinicalFact.confidence`
    (already in the frozen contract) as the discount signal instead —
    see `POSSIBLE_CONFIDENCE_FLOOR` below.
  - decisions/003 — every node pattern carries a label; no unlabelled
    lookups anywhere in this module.

Engine reality check (executed live against ghcr.io/hydra-db/hydradb:0.1.1,
not assumed from the survey/ARCHITECTURE text — several of ARCHITECTURE
§6.5's *documented* shapes do not execute on this build; this module uses
only the shapes actually confirmed live, dated 2026-08-16):

  1. `UNWIND $rows AS row MERGE (n:Label {id: row.vertex}) SET ...` — the
     ARCHITECTURE §6.5 vertex-upsert shape — is REJECTED live, even when
     `id` already exists: "UNWIND vertex upsert MERGE pattern matches only
     id; apply labels with SET". (This corroborates, and generalizes, the
     narrower live finding already pinned by
     `tests/test_existence_both_directions.py`'s fixture comment about a
     *fresh* labelled MERGE.) `close_interval()` below therefore uses a
     plain, non-UNWIND `MATCH (c:Claim {id: $id}) SET ...` instead — proven
     live. A pleasant side effect: `MATCH` can never create a phantom claim
     the way an unknown-id `MERGE` could, so the "never MERGE an unknown
     id" hazard ARCHITECTURE warns about does not arise here at all.
  2. `UNWIND $rows AS row MATCH (a)-[:REL]->(b) ...` (a *traversal* inside
     an UNWIND-driven MATCH) is REJECTED: "UNWIND batch node patterns do
     not support labels" / "UNWIND batch supports one-hop relationships
     only" depending on shape. Object-identity resolution (`ABOUT` lookup)
     therefore runs one plain `MATCH (c:Claim {id:$cid})-[:ABOUT]->(e:Label)
     RETURN ...` per candidate claim (proven live) rather than a batched
     UNWIND traversal.
  3. `UNWIND $rows AS row MATCH (s:X {id:row.a}), (d:Y {id:row.b}) MERGE
     (s)-[r:REL {id:row.c}]->(d) SET r.prop = row.prop` — TWO independent,
     comma-separated, already-labelled node lookups (no inline traversal)
     followed by an edge `MERGE` — IS confirmed live, both for creating a
     fresh edge and (replayed) for re-`MERGE`ing the same edge id
     idempotently. This is exactly ARCHITECTURE §6.5's relationship-upsert
     shape and is what `link()` below uses, unchanged.
  4. Bare `CREATE` only ever succeeds as one fresh, one-hop, both-endpoints-
     new edge pattern (ARCHITECTURE's own "only one-hop edge patterns"
     message); a `MATCH`-then-`CREATE` mixing one existing and one fresh
     endpoint is rejected regardless of direction. This only affects how
     *fixtures* are built in `tests/test_invalidate.py` (this module itself
     never creates nodes) but is noted here since it shapes the test file.

None of this is a deviation from ARCHITECTURE's *intent* (never delete,
close-by-property-update, reify SUPERSEDES/CONTRADICTS) — only from the
literal Cypher text of §6.5's vertex-upsert example, which this survey
could not have verified without a live spike (ARCHITECTURE.md §8: "if the
query is not in survey 10, spike it on the live node first"). Flagged here,
not silently patched over, so E4-S3/E4-S4 (schema.py / writer.py, not yet
landed at the time this module was written) can reconcile against the same
live finding rather than re-discovering it.

Scope note (dependency gap, resolved mid-story — recorded, not scrubbed):
this story (E4-S5) depends on E4-S3 (`graph/schema.py`) and E4-S4
(`graph/writer.py`). Both landed *concurrently*, mid-way through this
story's implementation — this module's core decision procedure and its
live-verified Cypher shapes were written and tested against a
self-contained design first (no `schema` import, a locally-derived
predicate-classification table). Once `schema.py` landed with its own
`FUNCTIONAL_KEYS` — independently arriving at the identical partition of
`contracts.PREDICATES` this module had already derived — the local copy
was reconciled to import from `schema.py` instead (see the "Functional vs.
set-valued predicates" section below); `_fetch_object_id`'s label
validation was likewise reconciled onto `schema.label_for`. `apply()`
still takes an explicit `claim_ids: Mapping[fact_id, int]` for the
already-minted, already-written new-claim vertices — that is `writer.py`'s
job (now landed, and wired to call this module's `apply()` as its own
write order's final step; see `graph/writer.py`), not a boundary this
module should collapse. The one thing this module still mints locally is
the integer id of a `SUPERSEDES`/`CONTRADICTS` *edge* (`_fold_id`) —
nothing else mints those, so it remains this module's own concern, not a
duplicate of E3-S1's node-id minting.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Literal, Mapping, Sequence

from medmemgraph.contracts import PREDICATES, SENTINEL_VALID_TO, ClinicalFact
from medmemgraph.graph import schema
from medmemgraph.hydra_client import HydraClient
from medmemgraph.pipeline.ids import mint_patient_id

__all__ = [
    "FUNCTIONAL_PREDICATES",
    "SET_VALUED_PREDICATES",
    "is_functional",
    "POSSIBLE_CONFIDENCE_FLOOR",
    "STALENESS_WINDOW",
    "ClaimRow",
    "Decision",
    "InvalidationReport",
    "classify",
    "close_interval",
    "link",
    "apply",
    "write_and_invalidate",
]

# ---------------------------------------------------------------------------
# Functional vs. set-valued predicates — §6.6 step 2/3; literature/05 Q4's
# "relation type + uniqueness key" framing.
#
# Reconciliation note: this story (E4-S5) was written and its core decision
# procedure implemented, tested, and live-verified BEFORE `graph/schema.py`
# (E4-S3) landed — see this module's earlier "Scope note" paragraph above,
# which documents that dependency gap as it stood at the time. `schema.py`
# has since landed (concurrently) with its own `FUNCTIONAL_KEYS` table,
# independently arriving at the IDENTICAL partition of `contracts.PREDICATES`
# this module had already derived on its own from the same ARCHITECTURE §6.6
# text — cross-checked, not assumed. `schema.py`'s own docstring claims this
# vocabulary ("no call site hand-writes a label... this is the ONE place"),
# so this module now derives its sets from `schema.FUNCTIONAL_KEYS` rather
# than keeping a second, independently-hand-maintained copy — the "ONE
# place" the story asks for is `schema.py`, once it exists; this module
# still owns *when* that classification is queried against
# `contracts.PREDICATES`, i.e. the completeness assertion below.
# ---------------------------------------------------------------------------

FUNCTIONAL_PREDICATES = frozenset(
    p for p, kind in schema.FUNCTIONAL_KEYS.items() if kind == "functional"
)
SET_VALUED_PREDICATES = frozenset(
    p for p, kind in schema.FUNCTIONAL_KEYS.items() if kind == "set-valued"
)

# Defensive completeness check: every predicate in the frozen contract has
# exactly one cardinality. If PREDICATES is ever extended without updating
# schema.FUNCTIONAL_KEYS, this fails loudly at import time instead of
# silently treating the new predicate as neither functional nor set-valued.
assert FUNCTIONAL_PREDICATES | SET_VALUED_PREDICATES == PREDICATES, (
    "schema.FUNCTIONAL_KEYS has drifted from contracts.PREDICATES — update "
    "graph/schema.py's FUNCTIONAL_KEYS table."
)
assert not (FUNCTIONAL_PREDICATES & SET_VALUED_PREDICATES)


def is_functional(predicate: str) -> bool:
    """True for a predicate where, over (patient, object), at most one
    active claim can be true at a time (a new matching claim closes the
    old one). False for a predicate that accumulates (coexistence is the
    default; only an explicit same-object retraction closes a twin)."""
    if predicate in FUNCTIONAL_PREDICATES:
        return True
    if predicate in SET_VALUED_PREDICATES:
        return False
    raise ValueError(f"is_functional(): {predicate!r} is not in the closed PREDICATES vocabulary")


# ---------------------------------------------------------------------------
# Decision 002 (OPEN) — confidence discount stand-in for the i2b2 "possible"
# class. Polarity stays strictly binary on the wire (contracts.py); this
# module does NOT read any i2b2 field (there is none to read). Instead: a
# new fact whose extraction confidence is below this floor is never allowed
# to unilaterally close an existing active claim — it is routed to
# "contradiction"/unresolved instead, so a hedged ("rule out X") assertion
# can never silently overwrite a firmly-asserted ("has X") one. 0.5 is a
# documented, overridable default (module-level constant, not hard-coded
# inline), not a literature-sourced number — there is none to cite, since
# 002 itself is unclosed.
# ---------------------------------------------------------------------------
POSSIBLE_CONFIDENCE_FLOOR = 0.5

# ---------------------------------------------------------------------------
# MEASURED INTERACTION (2026-08-18, 20-patient corpus run) — read before
# tuning either constant.
#
# `POSSIBLE_CONFIDENCE_FLOOR` (0.5, here) and
# `extract._POSSIBLE_CONFIDENCE_DISCOUNT` (0.5, there) were chosen
# independently and land 0.05 apart in effect. An i2b2 `possible` fact starts
# around 0.9 and is halved to ~0.45, i.e. JUST under this floor — so rule 3
# below routes essentially EVERY hedged fact to `contradiction/low_confidence`
# rather than letting it participate in supersede arbitration.
#
# Corpus-scale effect: 6,917 CONTRADICTS vs 4,271 SUPERSEDES across 20
# patients; on subject 10912213, 196 of 860 claims (23%) sit at confidence
# 0.34-0.43 and are contradiction-bound by construction.
#
# This is defensible and is NOT being changed here: decisions/002 is open, and
# "a hedged fact can never close a firmer one" is the intended semantics. But
# the *magnitude* is an artifact of two constants meeting, not a considered
# calibration, and the contradiction rate should be read with that in mind.
# Changing either number re-writes invalidation semantics corpus-wide and
# requires a full re-ingest to observe.
# ---------------------------------------------------------------------------

# literature/09 Q1 rule 3: "only if it is not older than the current source
# by more than a declared staleness window". Frozen at 0 for this story
# (E4-S5.md): the lower-trust side overrides trust-class only if it is
# STRICTLY more recent, no grace period.
STALENESS_WINDOW = 0

# literature/09 Q1 rule 3 worked example / E4-S5.md: "source_class domain is
# the frozen binary doctor|patient. Trust: doctor > patient."
_TRUST = {"patient": 0, "doctor": 1}

# Object-label validation before interpolating an `EntityRef.type` string
# into a Cypher pattern (labels can't be bind parameters in this dialect) is
# `schema.label_for` — it raises `ValueError` for anything outside the
# recognized entity-type set, which is a *stricter*, single-source-of-truth
# check than a bare identifier-shape regex would be (api-security: allow-
# list, never block-list; the allow-list here is `schema.DOMAIN_ENTITY_LABELS`
# itself, not just "looks like a safe identifier").


def _fold_id(key: str) -> int:
    """A positive 63-bit fold of a SHA-256 digest of `key` (§5.4 point 3's
    formula), used ONLY to mint the integer id of a `SUPERSEDES`/
    `CONTRADICTS` *relationship* this module writes. This module does not
    mint :Claim or :Patient *node* ids — those are the caller's
    (Pipeline/E3-S1's) job, passed in via `claim_ids` / already present on
    `ClinicalFact.subject.canonical_id`. Relationship ids for these two
    edge types are this module's own concern since nothing else mints
    them. Purely a function of `key` (no counter, no persisted map), so
    replay always derives the same edge id and `MERGE` matches instead of
    duplicating — this is what makes `apply()` idempotent on replay."""
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) & 0x7FFFFFFFFFFFFFFF


# ---------------------------------------------------------------------------
# Decision shape
# ---------------------------------------------------------------------------

Kind = Literal["update", "correction", "contradiction", "coexistence"]
"""The four outcomes the task/story require distinguished. `update`: both
were true, at different times (normal forward supersession). `correction`:
the closed claim's own claimed validity period was wrong from the start
(same-or-later valid_from than the boundary it is closed at — a zero- or
negative-width remaining truth window). `contradiction`: precedence could
not (or, per decision 002's confidence floor, must not) discriminate a
winner; both stay active, linked `CONTRADICTS` both ways. `coexistence`:
nothing conflicts — a brand-new fact, a different set-valued object, or an
agreeing/corroborating duplicate; nothing is closed."""

ResolutionReason = str
# "same_source" | "source_class" | "recency" | "specificity" |
# "corroboration_count" | "unresolved" | "low_confidence" | "corroboration" |
# "new"

ClaimRow = Mapping[str, object]
"""One projected row from the candidate-lookup query: at least `id`,
`predicate`, `status`, `valid_to`, `valid_from`, `polarity`, `source_class`,
`session_id`, plus `object_id`/`object_name` once resolved via `ABOUT`."""


@dataclass(frozen=True)
class Decision:
    """The output of `classify()` — a plan, not yet applied. `apply()` (or
    a caller driving `close_interval`/`link` directly) executes it."""

    kind: Kind
    resolution_reason: ResolutionReason
    supersedes: tuple[tuple[int, int], ...] = ()
    """(winner_claim_id, loser_claim_id) pairs. The loser is closed
    (`close_interval`) and a `SUPERSEDES` edge is written winner -> loser.
    The loser is usually the pre-existing candidate, but can be the
    just-arrived new claim itself when precedence favors the existing
    claim (e.g. acceptance criterion 4's doctor-beats-patient case) — the
    tuple shape makes the direction explicit either way."""
    contradicts: tuple[int, ...] = ()
    """Candidate claim ids this fact genuinely conflicts with and cannot
    (or must not) be resolved against. Both stay `status="active"`;
    `apply()` writes `CONTRADICTS` both directions."""
    corroborates: tuple[int, ...] = ()
    """Candidate claim ids this fact agrees with / accumulates alongside.
    Reporting only — no graph write (ARCHITECTURE's frozen REL_TYPES has
    no corroboration edge; two claim nodes agreeing is discoverable via
    their shared `ABOUT` target without a new edge type)."""


@dataclass
class InvalidationReport:
    """The whole-pass result of `apply()`. Every field keyed/indexed so a
    demo or a test can show "why" without re-querying the graph."""

    decisions: dict[str, Decision] = field(default_factory=dict)  # fact_id -> Decision
    closed_claim_ids: list[int] = field(default_factory=list)
    supersedes_written: list[tuple[int, int]] = field(default_factory=list)
    contradicts_written: list[tuple[int, int]] = field(default_factory=list)
    unresolved_fact_ids: list[str] = field(default_factory=list)
    corroborated: list[tuple[str, int]] = field(default_factory=list)  # (fact_id, claim_id)


# ---------------------------------------------------------------------------
# classify() — pure, no I/O. §6.6's decision procedure + literature/09 Q1's
# precedence ladder.
# ---------------------------------------------------------------------------


def classify(
    new_fact: ClinicalFact,
    existing_claims: Sequence[ClaimRow],
    *,
    new_claim_id: int,
) -> Decision:
    """`existing_claims` must already be the *structurally related* active
    candidate set (§6.6 step 1): same `patient_id` + same `predicate` +
    same object `canonical_id`, `status="active"`, `valid_to=SENTINEL`.
    This function does no graph I/O — see `_fetch_candidates` for that."""
    predicate = new_fact.predicate
    if predicate not in PREDICATES:
        raise ValueError(
            f"classify(): predicate {predicate!r} is outside the closed PREDICATES vocabulary"
        )

    if not existing_claims:
        return Decision(kind="coexistence", resolution_reason="new")

    if predicate in SET_VALUED_PREDICATES:
        return _classify_set_valued(new_fact, existing_claims, new_claim_id=new_claim_id)
    return _resolve_against_candidates(new_fact, existing_claims, new_claim_id=new_claim_id)


def _classify_set_valued(
    new_fact: ClinicalFact, existing_claims: Sequence[ClaimRow], *, new_claim_id: int
) -> Decision:
    """§6.6 step 3: set-valued predicates coexist by default. An explicit
    retraction — `polarity` differing from an active same-object twin —
    is the one case that closes something, and it is treated as exactly
    the same arbitration problem as a functional conflict (whichever side,
    old or new, precedence favors, survives; §12/decision 002 note: this
    is the direction ARCHITECTURE documents explicitly — negated closes
    asserted; the symmetric case, asserted reversing an active negated
    twin, is this module's own explicit extension, stated here rather than
    silently assumed, since ARCHITECTURE §6.6 step 3 only spells out one
    direction)."""
    agreeing = [c for c in existing_claims if c["polarity"] == new_fact.polarity]
    opposing = [c for c in existing_claims if c["polarity"] != new_fact.polarity]

    if not opposing:
        if agreeing:
            return Decision(
                kind="coexistence",
                resolution_reason="corroboration",
                corroborates=tuple(c["id"] for c in agreeing),
            )
        return Decision(kind="coexistence", resolution_reason="new")

    return _resolve_against_candidates(new_fact, opposing, new_claim_id=new_claim_id)


def _resolve_against_candidates(
    new_fact: ClinicalFact, candidates: Sequence[ClaimRow], *, new_claim_id: int
) -> Decision:
    """The precedence ladder (literature/09 Q1), applied once a genuine
    same-object conflict exists (functional predicate, or a set-valued
    predicate's opposing-polarity twin)."""
    # decision 002's confidence-discount guard: a low-confidence ("possible"
    # in spirit) new fact must never silently close a firmer existing claim.
    if new_fact.confidence < POSSIBLE_CONFIDENCE_FLOOR:
        return Decision(
            kind="contradiction",
            resolution_reason="low_confidence",
            contradicts=tuple(c["id"] for c in candidates),
        )

    # Normally exactly one active candidate. If more than one exists (an ER
    # or prior-bug anomaly outside this story's scope), resolve against the
    # most recently valid one; the others are left untouched, not silently
    # dropped or auto-merged.
    candidate = max(candidates, key=lambda c: (str(c["valid_from"]), int(c["id"])))
    siblings = [c for c in candidates if c is not candidate and c["polarity"] == candidate["polarity"]]

    winner_side, reason = _resolve_conflict(new_fact, candidate, siblings, new_claim_id=new_claim_id)

    if winner_side is None:
        return Decision(
            kind="contradiction",
            resolution_reason=reason,
            contradicts=tuple(c["id"] for c in candidates),
        )

    if winner_side == "new":
        winner_id, loser_id, loser_valid_from = new_claim_id, candidate["id"], candidate["valid_from"]
    else:
        winner_id, loser_id, loser_valid_from = candidate["id"], new_claim_id, new_fact.valid_from

    # update vs. correction (task-required distinction, literature/05's own
    # vocabulary): compare the CLOSED claim's own valid_from against the
    # boundary it is closed at (always new_fact.valid_from, mechanically —
    # see close_interval). A positive-width remaining interval means the
    # closed claim really was true until the boundary (update, a genuine
    # real-world change). A zero-or-negative width (closed at or before its
    # own start) means it was never true to begin with (correction).
    kind: Kind = "update" if str(loser_valid_from) < str(new_fact.valid_from) else "correction"
    return Decision(kind=kind, resolution_reason=reason, supersedes=((winner_id, loser_id),))


def _resolve_conflict(
    new_fact: ClinicalFact,
    candidate: ClaimRow,
    siblings: Sequence[ClaimRow],
    *,
    new_claim_id: int,
) -> tuple[str | None, str]:
    """Returns (winner_side, resolution_reason). winner_side is "new",
    "old", or None (unresolved — rule 7)."""
    # Rule 1 — same source: not a conflict at all, an update. "Source" is
    # read as (source_class, session_id) — the finest-grained identity
    # ClinicalFact carries; two assertions in the very same admission
    # session by the same class of speaker are the same voice correcting
    # itself, not two competing sources.
    if new_fact.source_class == candidate["source_class"] and new_fact.session_id == candidate["session_id"]:
        return "new", "same_source"

    new_trust = _TRUST[new_fact.source_class]
    old_trust = _TRUST[candidate["source_class"]]

    if new_trust != old_trust:
        # Rule 3 — source-class trust, recency-overridable (staleness
        # window 0: the lower-trust side wins only if STRICTLY more recent).
        if new_trust > old_trust:
            higher_side, higher_vf = "new", new_fact.valid_from
            lower_side, lower_vf = "old", candidate["valid_from"]
        else:
            higher_side, higher_vf = "old", candidate["valid_from"]
            lower_side, lower_vf = "new", new_fact.valid_from
        if str(lower_vf) > str(higher_vf):
            return lower_side, "recency"
        return higher_side, "source_class"

    # Rule 4 — same trust class: recency.
    if str(new_fact.valid_from) != str(candidate["valid_from"]):
        return ("new", "recency") if str(new_fact.valid_from) > str(candidate["valid_from"]) else ("old", "recency")

    # Rule 5 — specificity: longer object name is the available proxy
    # (ClinicalFact carries no separate "detail" field to compare).
    new_len = len(new_fact.object.name or "")
    old_len = len(str(candidate.get("object_name") or ""))
    if new_len != old_len:
        return ("new", "specificity") if new_len > old_len else ("old", "specificity")

    # Rule 6 — corroboration count, by DISTINCT session, not distinct
    # mention (literature/09's own caveat about copying inflating naive
    # counts). `siblings` are other active candidates sharing candidate's
    # polarity — a real corroboration signal once more than one exists;
    # otherwise this rule is a tie (1 vs 1) and rule 7 fires, which is the
    # honest answer when there is nothing to count yet.
    old_corrob = len({str(c["session_id"]) for c in siblings} | {str(candidate["session_id"])})
    new_corrob = 1
    if new_corrob != old_corrob:
        return ("new", "corroboration_count") if new_corrob > old_corrob else ("old", "corroboration_count")

    # Rule 7 — flag unresolved. Not a failure: literature/09's own
    # base-rate warning (DECODE: 93% balanced-benchmark accuracy collapses
    # to 23.94% precision at realistic base rates) is exactly why silent
    # auto-resolution is not defensible here.
    return None, "unresolved"


# ---------------------------------------------------------------------------
# close_interval() / link() — the only two write shapes this module uses.
# Both proven live (module docstring point 1/3) — no UNWIND for the former.
# ---------------------------------------------------------------------------


def close_interval(
    client: HydraClient,
    claim_id: int,
    new_valid_from: str,
    *,
    invalidated_at: str,
) -> bool:
    """Close `claim_id`'s interval at `new_valid_from` (always the
    incoming fact's own `valid_from` — never a Cypher `now()`, per
    ARCHITECTURE §8's "no `datetime()`/`now()` built-in" rule). Sets
    `status="invalidated"`; never deletes; never touches any other
    property. `invalidated_at` (the *system*/ingest clock, distinct from
    `valid_to` — the bitemporal split literature/05 insists on) is
    caller-supplied, required, no default.

    Uses a plain `MATCH ... SET` (not the ARCHITECTURE §6.5 `UNWIND MERGE`
    vertex-upsert shape — see module docstring point 1: that shape is
    rejected live). `MATCH` can never create a node, so there is no
    "MERGE of an unknown id creates a phantom claim" hazard here at all.

    Always returns True (the write is issued unconditionally). A trailing
    `RETURN` after this `MATCH ... SET` was spiked live and rejected —
    "mutation queries cannot continue with MATCH, RETURN, or WITH after
    writes" — so this cannot itself report whether `claim_id` existed
    without a second round trip; callers only ever pass ids returned by
    the candidate lookup, so a missing id should not happen in practice.
    If `claim_id` does not exist, `MATCH` matches zero rows and `SET` is a
    silent no-op — never a phantom create, unlike `MERGE`."""
    client.run(
        "MATCH (c:Claim {id: $id}) SET c.valid_to = $valid_to, "
        "c.invalidated_at = $invalidated_at, c.status = $status",
        id=claim_id,
        valid_to=new_valid_from,
        invalidated_at=invalidated_at,
        status="invalidated",
    )
    return True


def link(
    client: HydraClient,
    new_claim_id: int,
    old_claim_id: int,
    rel: Literal["SUPERSEDES", "CONTRADICTS"],
    *,
    observed_at: str,
) -> int:
    """Write `(:new)-[:rel {id}]->(:old)`. Never `DELETE`s anything.
    `rel` is allow-listed (never a block-list) before being interpolated
    into the query text — Cypher relationship types cannot be bound as
    query parameters, so a validated literal is the only legal way to
    parameterize this at all.

    Uses ARCHITECTURE §6.5's documented relationship-upsert shape verbatim
    (module docstring point 3 — confirmed live, including idempotent
    replay). The edge's own integer id is minted deterministically
    (`_fold_id`) from `(rel, new_claim_id, old_claim_id)`, so calling this
    twice with the same arguments re-`MERGE`s the same edge rather than
    duplicating it.

    Returns the minted relationship id."""
    if rel not in ("SUPERSEDES", "CONTRADICTS"):
        raise ValueError(f"link(): rel must be SUPERSEDES or CONTRADICTS, got {rel!r}")
    edge_id = _fold_id(f"{rel}|{new_claim_id}|{old_claim_id}")
    client.run(
        f"UNWIND $rows AS row "
        f"MATCH (s:Claim {{id: row.source_vertex}}), (d:Claim {{id: row.destination_vertex}}) "
        f"MERGE (s)-[r:{rel} {{id: row.relationship_vertex}}]->(d) "
        f"SET r.observed_at = row.observed_at",
        rows=[
            {
                "source_vertex": new_claim_id,
                "destination_vertex": old_claim_id,
                "relationship_vertex": edge_id,
                "observed_at": observed_at,
            }
        ],
    )
    return edge_id


# ---------------------------------------------------------------------------
# Candidate lookup — §6.6 step 1, "structurally related active claims".
# ---------------------------------------------------------------------------


def _fetch_active_claims_for_predicate(
    client: HydraClient, patient_node_id: int, predicate: str
) -> list[dict]:
    """One labelled MATCH on the patient (ARCHITECTURE §5.5-style shape,
    already proven live by this module's own spike and by
    `tests/test_run_dialect.py`'s `current-as-of-interval-query` case).
    HydraDB has no `IN`, so predicate/status/sentinel filtering happens
    client-side on the projected columns (the story's own documented
    escape hatch for "no IN")."""
    rows = client.run(
        "MATCH (p:Patient {id: $pid})-[:HAS]->(c:Claim) "
        "RETURN c.id AS id, c.fact_id AS fact_id, c.predicate AS predicate, "
        "c.status AS status, c.valid_to AS valid_to, c.valid_from AS valid_from, "
        "c.polarity AS polarity, c.source_class AS source_class, "
        "c.session_id AS session_id",
        pid=patient_node_id,
    )
    return [
        dict(r)
        for r in rows
        if r.get("predicate") == predicate
        and r.get("status") == "active"
        and r.get("valid_to") == SENTINEL_VALID_TO
    ]


def _fetch_object_id(client: HydraClient, claim_id: int, entity_type: str) -> dict | None:
    """Resolve one candidate claim's `ABOUT` target. One `client.run()` per
    claim — a batched `UNWIND` version of this traversal is rejected live
    (module docstring point 2); candidate counts here are small (typically
    0-2 per new fact), unlike writer.py's bulk-ingest write path, so this
    is not a throughput concern for the invalidation pass.

    `entity_type` is `EntityRef.type` (e.g. `"Medication"`); `schema.label_for`
    both maps it onto its graph label and validates it (raises `ValueError`
    for anything outside `schema.DOMAIN_ENTITY_LABELS`), so this can never
    build a Cypher pattern from an unrecognized or unsafe type string."""
    label = schema.label_for(entity_type)
    rows = client.run(
        f"MATCH (c:Claim {{id: $cid}})-[:ABOUT]->(e:{label}) "
        "RETURN e.id AS object_id, e.name AS object_name",
        cid=claim_id,
    )
    if not rows:
        return None
    return {"object_id": rows[0]["object_id"], "object_name": rows[0].get("object_name")}


def _fetch_candidates(
    client: HydraClient, new_fact: ClinicalFact, *, new_claim_id: int
) -> list[ClaimRow]:
    """§6.6 step 1 end to end: active claims for (patient, predicate),
    narrowed to same object `canonical_id`.

    The patient node id is `pipeline.ids.mint_patient_id(new_fact.patient_id)`
    — the SAME derivation `graph/writer.py` uses to write the `Patient`
    node and its `HAS` edges (`Patient|<patient_id>`, §5.4), not
    `new_fact.subject.canonical_id`. Reconciled here after `writer.py`
    landed and made this concrete: `EntityRef.canonical_id` is an
    entity-resolution concern (Pipeline/E3-S1) scoped to *domain* entities;
    the patient's own graph identity is keyed on the top-level
    `patient_id` string field, independent of whatever an extractor filled
    into a `Patient`-typed `EntityRef.canonical_id`. Using the wrong one
    would silently query a different (or nonexistent) node and return an
    empty candidate set every time — a false "no conflict" — so this is
    correctness-critical, not stylistic.

    Excludes `new_claim_id` itself. By write order (§6.5 step 1 then step
    3), the new fact's own `:Claim` vertex is already written and already
    `status="active"` by the time invalidation runs — so it would
    otherwise show up as its own "candidate", self-matching and corrupting
    the decision (caught live while spiking this module: an unfiltered
    fetch resolved a fact against itself, producing a same-id
    winner/loser pair)."""
    patient_node_id = mint_patient_id(new_fact.patient_id)
    active = _fetch_active_claims_for_predicate(client, patient_node_id, new_fact.predicate)
    candidates: list[ClaimRow] = []
    for row in active:
        if row["id"] == new_claim_id:
            continue
        info = _fetch_object_id(client, row["id"], new_fact.object.type)
        if info is None or info["object_id"] != new_fact.object.canonical_id:
            continue
        merged = dict(row)
        merged["object_id"] = info["object_id"]
        merged["object_name"] = info["object_name"]
        candidates.append(merged)
    return candidates


# ---------------------------------------------------------------------------
# apply() — the whole pass.
# ---------------------------------------------------------------------------


def apply(
    client: HydraClient,
    new_facts: Sequence[ClinicalFact],
    claim_ids: Mapping[str, int],
    *,
    now: str,
) -> InvalidationReport:
    """Run the invalidation decision procedure for each of `new_facts`.

    Assumes every fact in `new_facts` has ALREADY been written as an
    active `:Claim` node (writer.py's job, once E4-S4 lands) with its
    minted integer id in `claim_ids[fact.fact_id]` — this module does not
    create claim vertices (`graph`'s boundary: "Pipeline does not close
    intervals" also reads, symmetrically, as "invalidation does not write
    facts"). `now` is the caller-supplied ingest clock, used for every
    `invalidated_at`/`observed_at` this pass writes (one ingest pass, one
    system-time reading — never a Cypher `now()`).

    Idempotent on replay: re-running `apply()` with the same `new_facts`
    and `claim_ids` either (a) finds the relevant candidate already
    `status="invalidated"` and therefore no longer in the active candidate
    set, so `classify()` naturally short-circuits to a no-op `coexistence`
    decision, or (b) re-derives the identical `Decision` and re-issues the
    identical `close_interval`/`link` calls, both of which are themselves
    idempotent (`MATCH+SET` reasserts the same values; `link`'s
    deterministic edge id makes `MERGE` re-match rather than duplicate).
    Either path leaves the graph state unchanged on a second run."""
    report = InvalidationReport()

    for fact in new_facts:
        if fact.fact_id not in claim_ids:
            raise ValueError(
                f"apply(): no minted claim id for fact_id {fact.fact_id!r} — "
                "the vertex must be written (writer.py) before invalidation runs"
            )
        new_claim_id = claim_ids[fact.fact_id]

        candidates = _fetch_candidates(client, fact, new_claim_id=new_claim_id)
        decision = classify(fact, candidates, new_claim_id=new_claim_id)
        report.decisions[fact.fact_id] = decision

        for winner_id, loser_id in decision.supersedes:
            close_interval(client, loser_id, fact.valid_from, invalidated_at=now)
            link(client, winner_id, loser_id, "SUPERSEDES", observed_at=now)
            report.closed_claim_ids.append(loser_id)
            report.supersedes_written.append((winner_id, loser_id))

        for old_id in decision.contradicts:
            link(client, new_claim_id, old_id, "CONTRADICTS", observed_at=now)
            link(client, old_id, new_claim_id, "CONTRADICTS", observed_at=now)
            report.contradicts_written.append((new_claim_id, old_id))
        if decision.contradicts:
            report.unresolved_fact_ids.append(fact.fact_id)

        for corrob_id in decision.corroborates:
            report.corroborated.append((fact.fact_id, corrob_id))

    return report


# ---------------------------------------------------------------------------
# write_and_invalidate() — the integration point ARCHITECTURE §6.5's write
# order names as step 3 ("Invalidation pass (§6.6), then UNWIND MATCH+MERGE
# SUPERSEDES/CONTRADICTS"), and E4-S5.md's own file list asks this story to
# wire up ("graph/writer.py — call the invalidation pass after vertex/edge
# upsert... Do not rewrite upserts.").
#
# Implemented here, as an ADDITIVE wrapper, rather than by editing
# `graph/writer.py`'s own `write_facts()` body: `writer.py` (E4-S4) landed
# concurrently with this story, already reviewed-ready
# ("ready for code-reviewer" per its own return note) — this function calls
# it unmodified, once, and then calls `apply()`, rather than reaching into
# an already-complete peer file's internals. "Do not rewrite upserts" is
# satisfied by construction: `write_facts` is called exactly as documented,
# nothing about it changes.
# ---------------------------------------------------------------------------


def write_and_invalidate(
    client: HydraClient,
    facts: Sequence[ClinicalFact],
    id_map=None,
    *,
    now: str,
    batch_size: int = 1000,
):
    """`graph.writer.write_facts(...)` followed by this module's `apply()`
    — the two write-order steps ARCHITECTURE §6.5 documents as one
    sequence, run as one call. Returns `(WriteReport, InvalidationReport)`.

    `id_map` is resolved into one shared `pipeline.ids.IdMinter` (same
    resolution `write_facts` itself does) and reused for both calls, so the
    claim id `apply()` needs for each just-written fact is *recomputed*
    (never re-minted — `IdMinter.mint` returns the cached value for an
    already-seen identity key) from the exact same minter `write_facts`
    consumed, guaranteeing they agree by construction rather than by
    convention. Facts `write_facts` skipped (failed `contracts.validate()`)
    are excluded from the invalidation pass — a fact that was never written
    as a `:Claim` node has nothing to invalidate against.
    """
    # Local imports: `graph.writer` (and `pipeline.ids`, already imported at
    # module scope) depend on this module not existing yet in their own
    # import graph — keeping the writer.py -> invalidate.py edge one-
    # directional (nothing in this module is imported by writer.py) avoids
    # a cycle while still letting this orchestration live beside `apply()`.
    from medmemgraph.graph import writer as _writer
    from medmemgraph.pipeline.ids import IdMinter, mint_claim_id

    minter = id_map if isinstance(id_map, IdMinter) else IdMinter(id_map)
    write_report = _writer.write_facts(client, list(facts), minter, batch_size=batch_size)

    skipped_fact_ids = {fact_id for fact_id, _problems in write_report.skipped}
    written_facts = [f for f in facts if f.fact_id not in skipped_fact_ids]
    claim_ids = {f.fact_id: mint_claim_id(f.fact_id, id_map=minter) for f in written_facts}

    invalidation_report = apply(client, written_facts, claim_ids, now=now)
    return write_report, invalidation_report
