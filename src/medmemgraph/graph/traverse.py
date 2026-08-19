"""graph/traverse.py — `algo.MSpaths` wrapper: seeding, ranking, time-filtering,
and bounded serialization of whole-path results (ARCHITECTURE.md §7.3, story
E4-S6, `collaborative/literature/10-hydradb-capability-audit.md` §A5).

This is the "Best Use of HydraDB" surface: weighted, bounded, multi-source/
multi-target whole-path enumeration is genuinely awkward to reproduce with a
vector store (no graph structure to walk) or a relational database (a
recursive CTE returns reachability, not a ranked, bounded path SET). Per the
award's own wording ("hard to pull off with traditional vector or relational
approaches", hackhydra.hydradb.com) — this module does not claim
"impossible," only "hard," and the docstrings below say so rather than
overclaiming.

--------------------------------------------------------------------------
Executed-verified live-engine corrections to ARCHITECTURE.md §7.3's own copy
(dev-backend, 2026-08-16 — every claim below was spiked against
ghcr.io/hydra-db/hydradb:0.1.1, not assumed from the survey text; see
`hydra_client.py`'s "5. Raw path hydration" section header for the
node/relationship wire-shape half of this investigation):

1. **`sourceProperty: 'id'` (ARCHITECTURE §7.3's own worked example, "when
   seeds are minted integers") DOES NOT WORK on this schema.** Spiked live:
   `sourceValues: ['<minted-id>']` (even correctly string-typed — see point
   2) against `sourceProperty: 'id'` returns ZERO rows for a freshly
   created, directly connected, sanity-checked pair. Root cause: our own
   `id` property is not stored in the queryable property bag at all — the
   engine folds it into the vertex's native Bolt identity instead (see
   `hydra_client.py`'s Finding 2). `algo.MSpaths`'s `sourceProperty`/
   `sourceValues` selector mechanism is a property-VALUE index lookup; `id`
   is not an indexed property here, it IS the vertex key. This module
   therefore seeds by a genuine stored STRING property per label instead
   (`_SELECTOR_PROPERTY` below: `name` for domain entities, `fact_id` for
   `:Claim`, `patient_id` for `:Patient`, `session_id` for `:Admission`),
   resolved from the caller's integer ids via one small lookup per id
   (`_resolve_selector_values`), and recovers the caller's real minted ids
   back out of the returned paths via `GraphNode.id` (never via a selector
   property) — see point 4 for why that resolution also needs a safety net.

2. **`sourceValues`/`targetValues` must be a list of Cypher STRING
   literals, syntactically** — `sourceValues: [123]` (bare integers) is a
   parse error: `"sourceValues must be a list of strings"`. Not documented
   in literature/10.

3. **`sourceValues`/`targetValues` cannot be bound Cypher parameters.**
   Spiked: `sourceValues: $source_values` (a plain list of strings, passed
   as an ordinary `run(cypher, source_values=[...])` parameter) is
   rejected: `"composite parameter $source_values is only supported as an
   UNWIND input"` — the same restriction literature/10 documents for a
   list of *maps*, confirmed here to also cover a plain list of strings.
   Consequence: every source/target value is built as an INLINE Cypher
   string literal, which makes this module a genuine injection surface
   (domain-entity `name` values originate from clinical-text extraction,
   not this module's own input) — `_cypher_string_literal` below
   backslash-escapes a backslash and a single quote before interpolating, spiked live against
   a name containing both an apostrophe and a backslash and confirmed to
   round-trip correctly (not merely "looks escaped").

4. **A resolved selector-property VALUE is not guaranteed unique across
   patients** (`name` on a domain entity, and — less obviously —
   `session_id` on `:Admission`, are both plausible collision surfaces:
   nothing in the frozen schema patient-scopes them). Seeding
   `algo.MSpaths` by such a value could therefore, in principle, pull in a
   same-named node belonging to a DIFFERENT patient — a correctness bug
   and a PHI-scoping concern for a per-patient memory graph, not just a
   ranking nuisance. `paths_between` closes this with a client-side
   allow-list check (api-security: allow-list, never block-list) — every
   returned path's start node id must be in the caller's own `seed_ids`,
   and its end node id (if `target_ids` given) must be in `target_ids` —
   using the caller's OWN integer ids as the source of truth, not
   whatever the property-value lookup happened to widen the search to.
   Any path that fails this check is dropped, never returned. See
   `paths_between`'s docstring for the full argument.

5. **`maxLen` above 16 is REJECTED live, not silently clamped by the
   engine**: `TransientError: "native_path_max_len rejected by admission
   control: actual 20 exceeds limit 16"`. This confirms clamping
   client-side (this story's own instruction) is a correctness requirement,
   not a defensive nicety — an un-clamped caller-supplied `max_len` would
   otherwise raise, not degrade.

6. **`algo.MSpaths`'s `RETURN` clause accepts only BARE yielded column
   names — no `AS` aliasing.** `RETURN path AS path` and `RETURN pathWeight
   AS w` both fail: `"unexpected tokens after native path procedure"`.
   `_build_mspaths_cypher` below always emits bare `RETURN path, pathWeight,
   pathCost` (a shape `hydra_client.py`'s dialect gate already special-cases
   as legal, `_ALLOWED_BARE_YIELD_COLUMNS`).

None of this is a deviation from ARCHITECTURE §7.3's *intent* (seed with
integer ids resolved from a semantic entry point, walk bounded+weighted
paths, project only `path`/`pathWeight`/`pathCost`) — only from the literal
Cypher text of its `sourceProperty: 'id'` example, which the section itself
anticipates might need live correction ("if the query is not in survey 10,
spike it on the live node first", ARCHITECTURE §8).
--------------------------------------------------------------------------

Deliverables (story E4-S6):
  - `paths_between` — the `algo.MSpaths` wrapper.
  - `Path` — typed result: node/edge sequence, `pathWeight`, `pathCost`,
    and `.claim_ids` for provenance.
  - `filter_paths_by_time` — client-side valid-time filtering (the
    procedure has no interval concept at all — ARCHITECTURE §7.3: "It is
    NOT valid-time aware... Filter returned path endpoints in the client").
  - `rank_paths` — client-side heuristic scorer. **Implementation choice,
    stated plainly per this story's own permission ("a well-documented
    simpler scorer is acceptable — say which you did")**: this module ships
    the geometric hop-decay heuristic literature/11 §3 synthesizes as the
    shared mathematical structure behind PPR/spreading-activation/PRA
    (`score(path) = γ^hop_count · claim_confidence · recency`), NOT exact
    PPR by power iteration. literature/11 §1 argues exact PPR is affordable
    at this project's scale (sub-10ms/query) and names it a legitimate
    BUILD candidate — it was not implemented here because this story's
    budget already covers `paths_between` (including the `sourceProperty:
    'id'` live-correction above), the `Path` type, time filtering,
    serialization, the hydra_client.py raw-path extension the whole thing
    depends on, and the cyclic correctness-guard test; a from-scratch PPR
    power-iteration re-ranker is a clean, isolated follow-on over the same
    `GraphPath` data this module already exports, not a redesign.
  - `serialize_paths` — bounded prompt text. Deliberately NOT an
    LLMLingua-class hard/extractive compressor: literature/15 R-QCC-033
    (single-source, 2026 preprint, hedged) measures such compressors
    breaking multi-hop bridge entities — the token that links a fact in one
    admission to a fact in another — at rates up to 60% on bridge
    questions, exactly this project's cross-admission retrieval shape.
    `serialize_paths` instead greedily includes whole, unmodified rendered
    paths in ranked order until the token budget is exhausted, skipping
    (never truncating) any path that would not fit — a path's bridge
    entity is never separated from the rest of that path's text.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from medmemgraph.contracts import SENTINEL_VALID_TO
from medmemgraph.graph import schema
from medmemgraph.hydra_client import GraphNode, GraphRelationship, HydraClient

__all__ = [
    "MAX_HOPS",
    "DEFAULT_REL_TYPES",
    "DEFAULT_GAMMA",
    "Path",
    "paths_between",
    "filter_paths_by_time",
    "rank_paths",
    "serialize_paths",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_HOPS = 16
"""Hard, non-configurable engine ceiling (literature/10 §A4,
`src/core/config.rs:45`'s `max_traversal_hops: 16`). `paths_between` clamps
to this and warns rather than letting the engine reject the request live."""

DEFAULT_REL_TYPES: tuple[str, ...] = ("ABOUT", "SUPERSEDES", "CONTRADICTS")
"""ARCHITECTURE §7.3's own example set, minus `DRAWN_FROM`/`ADMITTED`/
`CONTAINS` — `ADMITTED`/`CONTAINS` walk the Patient->Admission->Turn
substructure, not the claim/entity substructure this module's callers traverse
— and minus `HAS`.

**`HAS` (Patient -> Claim) is excluded as a correctness requirement at corpus
scale, not a tuning preference.**

`:Patient` is a hub: one node with an edge to every claim that patient has (586
for subject 10056223; ~900 for the largest patient in the 20-patient run).
Including `HAS` lets a walk reach that hub and fan back out across every claim,
so cost explodes with depth. Measured against the real graph, 2026-08-18:

    with `HAS`    : max_len 2-3 -> ~1.5s; max_len >= 4 -> the engine KILLS the
                    query ("native_path_neighbors exceeded query timeout after
                    29999 ms; limit is 29999 ms")
    without `HAS` : max_len 8   -> ~1.2s

`retrieve.DEFAULT_GRAPH_MAX_HOPS` is 8, so with `HAS` every graph-routed
question against a real patient timed out. And the failure was INVISIBLE:
`graph/retrieve.py` catches broadly and degrades to the text arm, so those
questions were silently answered from vector/lexical evidence.

Excluding `HAS` costs nothing semantically. Seeds are already patient-scoped
(`retrieve._fetch_patient_entities` only ever fetches that one patient's
entities), so a hub-mediated path asserts co-patient-hood that is already
guaranteed by construction. Every informative shape survives —
`Entity <-ABOUT- Claim`, `Claim -SUPERSEDES-> Claim`,
`Claim -CONTRADICTS-> Claim` — and observed real paths are 1-2 hops.

Pass `rel_types=` explicitly to widen or narrow this."""

DEFAULT_GAMMA = 0.85
"""Hop-decay constant for `rank_paths`'s heuristic scorer
(`score = γ^hop_count · ...`). literature/11 §3 frames this whole family
of scorers (PPR's damping factor, spreading activation's per-hop decay,
PRA's path-weighted walk probability) as "hand-tunable", not a
literature-sourced fixed value — 0.85 is a reasonable, documented default
(shorter paths favoured, but a well-evidenced 3-hop cross-admission chain
is not crushed relative to a 1-hop same-admission fact), overridable by
callers who calibrate against the eval harness."""

_VALID_REL_DIRECTIONS = frozenset({"both", "outgoing", "incoming"})
"""`'both'` and `'outgoing'` are executed-verified live (see module
docstring). `'incoming'` is `'outgoing'`'s documented symmetric complement
(literature/10 §A5's config-key list names `relDirection` as one of
these three without enumerating all three live) — inferred, not itself
independently spiked; flagged here rather than silently assumed to be
identical in every respect. Medium confidence on `'incoming'` specifically;
high on `'both'`/`'outgoing'`."""

_SELECTOR_PROPERTY: dict[str, str] = {
    "Patient": "patient_id",
    "Admission": "session_id",
    "Claim": "fact_id",
    **{label: "name" for label in schema.DOMAIN_ENTITY_LABELS},
}
"""Which genuine stored STRING property `algo.MSpaths` can seed by, per
label — see module docstring point 1 for why `id` itself cannot be used.
`:Turn` is deliberately absent (not written by this project's writer;
out of this module's traversal scope, same reasoning as
`DEFAULT_REL_TYPES`)."""


# ---------------------------------------------------------------------------
# Path — the typed result.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Path:
    """One whole path off an `algo.MSpaths` call, in walk order:
    `nodes[i]`/`nodes[i+1]` are connected by `relationships[i]`
    (`len(nodes) == len(relationships) + 1`). Thin wrapper over
    `hydra_client.GraphPath` plus the two extra scalar columns
    `algo.*paths` can yield (`pathWeight`, `pathCost` — 0.0 when the
    procedure call didn't request them)."""

    nodes: tuple[GraphNode, ...]
    relationships: tuple[GraphRelationship, ...]
    path_weight: float
    path_cost: float

    @property
    def hop_count(self) -> int:
        return len(self.relationships)

    @property
    def start_id(self) -> int:
        return self.nodes[0].id

    @property
    def end_id(self) -> int:
        return self.nodes[-1].id

    @property
    def claim_ids(self) -> tuple[int, ...]:
        """Minted `:Claim` node ids on this path, in walk order — the
        provenance chain a demo/prompt needs to point back at specific
        `ClinicalFact`s (via `pipeline.ids.mint_claim_id`'s inverse: the
        caller's own `fact_id -> claim_id` map, or `claim_fact_ids` below
        if only the string `fact_id` is available)."""
        return tuple(n.id for n in self.nodes if "Claim" in n.labels)

    @property
    def claim_fact_ids(self) -> tuple[str, ...]:
        """The string `fact_id` property of every `:Claim` node on this
        path, in walk order — the ORIGINAL `ClinicalFact.fact_id`, useful
        when a caller has facts keyed by that string rather than the
        minted integer id."""
        return tuple(
            str(n.properties["fact_id"])
            for n in self.nodes
            if "Claim" in n.labels and "fact_id" in n.properties
        )


# ---------------------------------------------------------------------------
# Cypher building — string-literal escaping, selector resolution, the call.
# ---------------------------------------------------------------------------


def _cypher_string_literal(value: str) -> str:
    """Build a single-quoted Cypher string literal from an arbitrary
    Python string, backslash-escaping `\\` and `'`. Required because
    `sourceValues`/`targetValues` cannot be bound parameters (module
    docstring point 3) — every value in this module's queries is an
    INLINE literal, so this is the one function standing between a
    clinical-text-derived entity name and a Cypher injection. Spiked live
    against a value containing both an apostrophe and a backslash and
    confirmed the escaped literal round-trips to the original string via
    a `MATCH ... {name: <literal>}` lookup — not merely "looks escaped"."""
    escaped = value.replace("\\", "\\\\").replace("'", "\\'")
    return f"'{escaped}'"


def _resolve_selector_values(
    client: HydraClient, label: str, ids: Sequence[int]
) -> dict[int, str]:
    """For each id in `ids`, look up the stored string property
    `algo.MSpaths` can actually seed by for `label` (module docstring
    point 1). One `MATCH` per id — batching via `UNWIND` is unavailable
    for a labelled node lookup in this dialect (same documented
    constraint `graph/invalidate.py`'s `_fetch_object_id` already used:
    "UNWIND batch node patterns do not support labels"); seed/target
    counts here are small (a handful of entity-similarity hits per
    query), so this is not a throughput concern, same reasoning as that
    sibling module's.

    Ids that don't resolve (node doesn't exist, or the selector property
    is unset) are simply absent from the returned dict — never raised —
    so a caller composing a seed set from an upstream similarity search
    can pass ids it isn't 100% sure still exist without this function
    itself becoming a second existence-check surface (that job belongs to
    `graph.existence.exists`, decisions/003)."""
    prop = _SELECTOR_PROPERTY.get(label)
    if prop is None:
        raise ValueError(
            f"paths_between: no known algo.MSpaths selector property for label {label!r}; "
            f"expected one of {sorted(_SELECTOR_PROPERTY)}"
        )
    resolved: dict[int, str] = {}
    for node_id in ids:
        if not isinstance(node_id, int) or isinstance(node_id, bool):
            raise ValueError(f"paths_between: seed/target ids must be int, got {node_id!r}")
        rows = client.run(f"MATCH (n:{label} {{id: $id}}) RETURN n.{prop} AS val", id=node_id)
        if rows and rows[0].get("val") is not None:
            resolved[node_id] = str(rows[0]["val"])
    return resolved


def _build_mspaths_cypher(
    *,
    seed_label: str,
    source_values: Sequence[str],
    target_label: str | None,
    target_values: Sequence[str] | None,
    rel_types: Sequence[str],
    rel_direction: str,
    max_len: int,
    path_count: int,
    result_limit: int,
    pairwise: bool,
) -> str:
    parts = [
        f"sourceLabel: {_cypher_string_literal(seed_label)}",
        f"sourceProperty: {_cypher_string_literal(_SELECTOR_PROPERTY[seed_label])}",
        "sourceValues: [" + ", ".join(_cypher_string_literal(v) for v in source_values) + "]",
    ]
    if target_values is not None:
        assert target_label is not None  # enforced by paths_between before this is called
        parts += [
            f"targetLabel: {_cypher_string_literal(target_label)}",
            f"targetProperty: {_cypher_string_literal(_SELECTOR_PROPERTY[target_label])}",
            "targetValues: [" + ", ".join(_cypher_string_literal(v) for v in target_values) + "]",
            f"pairwise: {'true' if pairwise else 'false'}",
        ]
    parts += [
        "relTypes: [" + ", ".join(_cypher_string_literal(rt) for rt in rel_types) + "]",
        f"relDirection: {_cypher_string_literal(rel_direction)}",
        f"maxLen: {int(max_len)}",
        f"pathCount: {int(path_count)}",
        f"resultLimit: {int(result_limit)}",
    ]
    config = ", ".join(parts)
    # Bare column names only — no `AS` aliasing (module docstring point 6).
    return f"CALL algo.MSpaths({{ {config} }}) YIELD path, pathWeight, pathCost RETURN path, pathWeight, pathCost"


# ---------------------------------------------------------------------------
# paths_between — the public wrapper.
# ---------------------------------------------------------------------------


def paths_between(
    client: HydraClient,
    seed_ids: Sequence[int],
    target_ids: Sequence[int] | None = None,
    *,
    seed_label: str,
    target_label: str | None = None,
    rel_types: Sequence[str] | None = None,
    rel_direction: str = "both",
    max_len: int = MAX_HOPS,
    path_count: int = 5,
    result_limit: int = 100,
    pairwise: bool = True,
) -> list[Path]:
    """Bounded, weighted, multi-source[/multi-target] whole-path
    enumeration via `algo.MSpaths` (ARCHITECTURE §7.3).

    `seed_ids` / `target_ids` are OUR minted integer node ids (the ids
    every other module in this project speaks in), NOT the value
    `algo.MSpaths` is actually seeded by on the wire — see module
    docstring point 1 for why `sourceProperty: 'id'` cannot work on this
    schema, and why `seed_label`/`target_label` (a single label covering
    the whole batch, matching `algo.MSpaths`'s own one-label-per-call
    shape) are required rather than inferred. A caller with a
    heterogeneous-label seed set calls this once per label and merges
    results client-side — inventing a mixed-label single call would not
    match the underlying primitive's actual shape.

    If `target_ids` is given, `target_label` is required (raises
    `ValueError` otherwise) and the call runs in `pairwise` mode (multi-
    source/multi-target path enumeration, ARCHITECTURE §7.3's own shape).
    If `target_ids` is omitted, this is an `SSpaths`-like "everything
    reachable from these seeds within `max_len` hops" call (spiked live:
    omitting `targetLabel`/`targetProperty`/`targetValues`/`pairwise`
    entirely returns all paths up to `max_len`, not just the longest).

    If `seed_ids` resolve to no valid selector value at all (none of them
    exist under `seed_label`, or the property is unset on all of them),
    this returns `[]` WITHOUT ever calling `algo.MSpaths` — ARCHITECTURE
    §7.2's own rule: "If the seed set is empty, skip algo.MSpaths... Do
    not invent a seed." Same rule applies symmetrically to a non-`None`
    but empty-after-resolution `target_ids`.

    **Safety filter (api-security: allow-list, never block-list;
    see module docstring point 4 for the concrete leak this defends
    against)**: every returned `Path`'s `start_id` is guaranteed to be one
    of the caller's own `seed_ids`, and — when `target_ids` was given —
    its `end_id` is guaranteed to be one of the caller's own `target_ids`.
    A path whose resolved-by-value endpoints don't actually match the
    caller's real integer ids (a `name`/`session_id` collision across two
    different patients' nodes, e.g. two "metformin" `:Medication` nodes)
    is silently dropped here, never returned — this project is a
    per-patient memory graph; a cross-patient path leaking through is a
    PHI-scoping bug, not a ranking nuisance, and this project has no other
    mechanism (patient-scoped selector property) to prevent it upstream of
    this check without a schema change out of this story's scope.

    `max_len` is clamped to `MAX_HOPS` (16) with a `warnings.warn` if the
    caller passed higher — the engine REJECTS (not silently truncates) a
    request above its hard-coded ceiling (module docstring point 5)."""
    if not seed_ids:
        return []
    if seed_label not in schema.LABELS:
        raise ValueError(f"paths_between: seed_label {seed_label!r} not in schema.LABELS")
    if target_ids is not None and target_label is None:
        raise ValueError("paths_between: target_label is required when target_ids is given")
    if target_label is not None and target_label not in schema.LABELS:
        raise ValueError(f"paths_between: target_label {target_label!r} not in schema.LABELS")

    resolved_rel_types = tuple(rel_types) if rel_types is not None else DEFAULT_REL_TYPES
    if not resolved_rel_types:
        raise ValueError("paths_between: rel_types must not be empty")
    for rt in resolved_rel_types:
        if rt not in schema.REL_TYPES:
            raise ValueError(f"paths_between: rel type {rt!r} not in schema.REL_TYPES")

    if rel_direction not in _VALID_REL_DIRECTIONS:
        raise ValueError(
            f"paths_between: rel_direction must be one of {sorted(_VALID_REL_DIRECTIONS)}, "
            f"got {rel_direction!r}"
        )

    if max_len > MAX_HOPS:
        warnings.warn(
            f"paths_between: max_len={max_len} exceeds HydraDB's hard-coded traversal "
            f"ceiling of {MAX_HOPS} hops (literature/10 §A4, not configurable — the engine "
            f"REJECTS a request above this ceiling live, TransientError "
            f"'native_path_max_len rejected by admission control'); clamping to {MAX_HOPS}.",
            stacklevel=2,
        )
        max_len = MAX_HOPS
    if max_len < 1:
        raise ValueError(f"paths_between: max_len must be >= 1, got {max_len}")
    if path_count < 1:
        raise ValueError(f"paths_between: path_count must be >= 1, got {path_count}")
    if result_limit < 1:
        raise ValueError(f"paths_between: result_limit must be >= 1, got {result_limit}")

    seed_ids = list(seed_ids)
    seed_value_by_id = _resolve_selector_values(client, seed_label, seed_ids)
    if not seed_value_by_id:
        return []  # ARCHITECTURE §7.2: skip the procedure, do not invent a seed.
    source_values = sorted(set(seed_value_by_id.values()))
    allowed_seed_ids = frozenset(seed_value_by_id)

    target_values: list[str] | None = None
    allowed_target_ids: frozenset[int] | None = None
    if target_ids is not None:
        target_ids = list(target_ids)
        target_value_by_id = _resolve_selector_values(client, target_label, target_ids)
        if not target_value_by_id:
            return []
        target_values = sorted(set(target_value_by_id.values()))
        allowed_target_ids = frozenset(target_value_by_id)

    cypher = _build_mspaths_cypher(
        seed_label=seed_label,
        source_values=source_values,
        target_label=target_label,
        target_values=target_values,
        rel_types=resolved_rel_types,
        rel_direction=rel_direction,
        max_len=max_len,
        path_count=path_count,
        result_limit=result_limit,
        pairwise=pairwise,
    )
    rows = client.run_paths(cypher)

    results: list[Path] = []
    for row in rows:
        graph_path = row["path"]
        if graph_path.nodes[0].id not in allowed_seed_ids:
            continue  # selector-value collision — see docstring's safety-filter paragraph
        if allowed_target_ids is not None and graph_path.nodes[-1].id not in allowed_target_ids:
            continue
        results.append(
            Path(
                nodes=graph_path.nodes,
                relationships=graph_path.relationships,
                path_weight=float(row.get("pathWeight") or 0.0),
                path_cost=float(row.get("pathCost") or 0.0),
            )
        )
    return results


# ---------------------------------------------------------------------------
# filter_paths_by_time — client-side valid-time filtering.
# ---------------------------------------------------------------------------


def _claim_live_at(node: GraphNode, as_of: str) -> bool:
    """A `:Claim` node is live at `as_of` iff `valid_from <= as_of <
    valid_to` (half-open interval, matching `graph/schema.py`'s
    `CURRENT_AS_OF_CYPHER` and `graph/invalidate.py.close_interval`'s own
    semantics: the closing boundary belongs to the successor claim, not
    the predecessor). A non-`:Claim` node (domain entity, Patient,
    Admission — nothing on this path's node list carries an interval)
    trivially passes, since it has no interval to be outside of."""
    valid_from = node.properties.get("valid_from")
    valid_to = node.properties.get("valid_to")
    if valid_from is None or valid_to is None:
        return True
    return valid_from <= as_of < valid_to


def filter_paths_by_time(paths: Sequence[Path], as_of: str) -> list[Path]:
    """Drop every path that crosses a `:Claim` not live at `as_of`.

    THIS HAPPENS CLIENT-SIDE, and must — `algo.MSpaths` is snapshot-scoped
    (one consistent SlateDB read view for the whole traversal) but has NO
    concept of a valid-time interval at all (ARCHITECTURE §7.3: "It is
    NOT valid-time aware... Filter returned path endpoints in the client
    by valid_from/valid_to against the question's as-of date"). A path can
    walk straight through a `SUPERSEDES`-closed, long-invalidated `:Claim`
    exactly as readily as a live one; this is the only place in the read
    path that distinguishes them. `as_of` uses the same sentinel
    convention as everywhere else in this project (`SENTINEL_VALID_TO`
    for "no upper bound") — string comparison, ISO-8601 lexical order,
    never a Cypher `now()` (ARCHITECTURE §8)."""
    return [
        path
        for path in paths
        if all(_claim_live_at(n, as_of) for n in path.nodes if "Claim" in n.labels)
    ]


# ---------------------------------------------------------------------------
# rank_paths — client-side heuristic scorer (module docstring: which
# scorer, and why, stated there rather than re-argued here).
# ---------------------------------------------------------------------------


def _path_confidence_component(path: Path) -> float:
    confidences = [
        n.properties.get("confidence")
        for n in path.nodes
        if "Claim" in n.labels and isinstance(n.properties.get("confidence"), (int, float))
    ]
    if not confidences:
        return 1.0
    return sum(confidences) / len(confidences)


def _path_recency_component(path: Path, *, as_of: str | None, half_life_days: float | None) -> float:
    """Exponential decay by days-old, relative to `as_of`, of the path's
    MOST RECENT `:Claim.valid_from`. Returns 1.0 (neutral) if `as_of` or
    `half_life_days` is unset, or the path carries no `:Claim` /
    `valid_from` to compare — recency is a bonus signal, never a
    disqualifier, since a structurally-relevant but old path can still be
    the only evidence available."""
    if not as_of or not half_life_days:
        return 1.0
    valid_froms = [
        n.properties.get("valid_from")
        for n in path.nodes
        if "Claim" in n.labels and n.properties.get("valid_from")
    ]
    if not valid_froms:
        return 1.0
    most_recent = max(valid_froms)
    # ISO-8601 date-prefix arithmetic only (YYYY-MM-DD...) — string slice,
    # not a datetime parse, to avoid a new dependency for a bonus signal;
    # matches this project's existing lexical-order convention for dates.
    try:
        days_diff = abs(_date_ordinal(as_of) - _date_ordinal(most_recent))
    except (ValueError, IndexError):
        return 1.0
    return 0.5 ** (days_diff / half_life_days)


def _date_ordinal(iso_string: str) -> int:
    """Rough day-ordinal for `YYYY-MM-DD...` (ignores time-of-day —
    good enough for a half-life recency weight, not a precise duration
    calculation). Raises `ValueError` on an unparseable prefix, caught by
    the one caller above."""
    year, month, day = int(iso_string[0:4]), int(iso_string[5:7]), int(iso_string[8:10])
    return year * 372 + month * 31 + day  # a monotonic-enough pseudo-ordinal, not calendar-exact


def rank_paths(
    paths: Sequence[Path],
    *,
    gamma: float = DEFAULT_GAMMA,
    as_of: str | None = None,
    recency_half_life_days: float | None = None,
) -> list[Path]:
    """Client-side path ranking — descending score, most useful first.

    `score(path) = γ^hop_count · mean(claim confidence) · recency(as_of)`
    (module docstring explains the choice of this simpler, well-documented
    heuristic over exact PPR by power iteration). `path_weight`/
    `path_cost` are NOT separately multiplied in here: in this project's
    current schema no edge carries a `weight`/`cost` property, so
    `algo.MSpaths` returns `pathWeight == hop_count` and `pathCost == 0`
    for every path (confirmed live) — folding `path_weight` in again would
    double-count hop-count decay, not add signal. Both are still recorded
    on every `Path` (see that dataclass) so a future schema that DOES
    configure edge weights/costs can extend this scorer without a shape
    change.

    `recency_half_life_days` is opt-in (`None` — the default — disables
    the recency term entirely, `_path_recency_component` returns 1.0):
    calling code that has no natural "as of when am I asking" date (a
    pure structural query) should not have an arbitrary recency bias
    silently injected."""
    scored = [
        (
            path,
            (gamma**path.hop_count)
            * _path_confidence_component(path)
            * _path_recency_component(path, as_of=as_of, half_life_days=recency_half_life_days),
        )
        for path in paths
    ]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return [path for path, _score in scored]


# ---------------------------------------------------------------------------
# serialize_paths — bounded prompt text (whole paths only, never truncated
# mid-path — see module docstring for the referential-dangling argument).
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _encoding() -> Any:
    import tiktoken

    return tiktoken.get_encoding("cl100k_base")


def _estimate_tokens(text: str) -> int:
    """Same tiktoken-cl100k_base-with-fallback pattern as
    `pipeline/loader.py`'s `Conversation.token_estimate()` and
    `eval/types.py`'s `estimate_tokens` — kept as a small LOCAL copy
    rather than importing `medmemgraph.eval.types` (Evidence's owned
    package, ARCHITECTURE §3.1/§3.2: Graph has no documented dependency
    edge onto Evidence; `eval/types.py` also pulls in
    `pipeline.loader.Conversation`, which this module has no other reason
    to depend on)."""
    if not text:
        return 0
    try:
        return len(_encoding().encode(text))
    except Exception:
        return max(1, len(text) // 4)


def _render_node(node: GraphNode) -> str:
    label = next(iter(node.labels), "Node")
    props = node.properties
    if label == "Claim":
        predicate = props.get("predicate", "?")
        polarity = props.get("polarity", "?")
        status = props.get("status", "?")
        valid_from = props.get("valid_from", "?")
        valid_to = props.get("valid_to", "?")
        interval = "ongoing" if valid_to == SENTINEL_VALID_TO else f"{valid_from}..{valid_to}"
        confidence = props.get("confidence")
        conf_suffix = f", conf={confidence:.2f}" if isinstance(confidence, (int, float)) else ""
        return f"Claim[{predicate} {polarity}, {status}, {interval}{conf_suffix}]"
    if label == "Patient":
        return f"Patient({props.get('patient_id', node.id)})"
    if label == "Admission":
        return f"Admission({props.get('session_id', node.id)})"
    name = props.get("name")
    if name:
        return f"{label}({name})"
    return f"{label}#{node.id}"


def _render_hop(path: Path, index: int) -> str:
    rel = path.relationships[index]
    from_node = path.nodes[index]
    if rel.start_id == from_node.id:
        return f"-[{rel.type}]->"
    return f"<-[{rel.type}]-"


def _render_path(path: Path) -> str:
    parts = [_render_node(path.nodes[0])]
    for i in range(path.hop_count):
        parts.append(_render_hop(path, i))
        parts.append(_render_node(path.nodes[i + 1]))
    return " ".join(parts)


def serialize_paths(paths: Sequence[Path], token_budget: int) -> str:
    """Render `paths` (already ranked — see `rank_paths`) into bounded
    prompt text, one path per line, greedily including paths in the given
    order until `token_budget` is exhausted.

    A path that would not fit in the REMAINING budget is skipped (not
    truncated) and the next, lower-ranked path is tried instead — the
    knapsack-shaped greedy heuristic literature/11 §3 recommends, without
    a claimed approximation-ratio guarantee (that survey explicitly
    declines to assert one; this docstring does the same). Every included
    path is rendered whole: no LLMLingua-class hard/extractive compressor
    ever runs over this text (module docstring: literature/15 R-QCC-033's
    referential-dangling failure mode — up to 60% of bridge entities
    dropped by such compressors on multi-hop evidence, which is exactly
    what a cross-admission path chain is). `token_budget <= 0` returns an
    empty string without inspecting any path."""
    if token_budget <= 0:
        return ""
    lines: list[str] = []
    used = 0
    for path in paths:
        text = _render_path(path)
        cost = _estimate_tokens(text) + (1 if lines else 0)  # +1 for the joining newline
        if used + cost > token_budget:
            continue
        lines.append(text)
        used += cost
    return "\n".join(lines)
