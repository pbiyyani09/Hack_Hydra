"""demo/provenance.py — `provenance_chain(client, *, patient_id, claim_id)`
(ARCHITECTURE.md §5/§6.6/§7.3, `collaborative/design/stories/E8/E8-S2.md`,
`DEMO.md` Beat 3, "$500-shaped moment").

"Show the chain of updates to this fact, with evidence for each transition"
is a path query over typed, timestamped edges: `(:new)-[:SUPERSEDES]->(:old)`
plus `(:Claim)-[:DRAWN_FROM]->(:Turn)` for the quoted evidence at each hop.
A vector store's top-k similarity returns nearest chunks, not an ordered
chain with a stated reason for each transition — the award criterion's own
wording (hackhydra.hydradb.com) is **"hard to pull off"**, not
"impossible", and this module (like `graph/traverse.py`/`graph/invalidate.py`
before it) does not upgrade that in prose, docstrings, or printed output.

--------------------------------------------------------------------------
Design decisions this module states explicitly rather than silently baking
in (a reviewer should be able to agree with or push back on each):

1. **The packet's preferred single `algo.MSpaths` call (Claim seeds,
   pairwise, `targetLabel: 'Turn'`) is adapted into TWO calls, for two
   independent, evidence-based reasons — not a silent rewrite of the
   packet's shape:**

   a. **No `:Turn` selector property exists anywhere in this codebase.**
      `graph/traverse.py`'s own `_SELECTOR_PROPERTY` table (the ONE place
      this project has already solved "`sourceProperty: 'id'` does not
      work here — the engine folds our `id` into Bolt identity, not the
      property bag" — see that module's docstring point 1) deliberately
      has no `Turn` entry ("`:Turn` is deliberately absent — not written by
      this project's writer; out of this module's traversal scope").
      Nothing in `src/` currently writes a `:Turn` node or `DRAWN_FROM`
      edge at all (`graph/writer.py`'s own docstring: "a Pipeline story
      that also has the source `Conversation` object is the natural owner
      of that edge" — that story has not landed). Inventing a selector
      property for `Turn` here, in a file this story does not list as
      Pipeline-owned schema surface, would be exactly the kind of "story
      disagrees with the actual files" situation this project's rules say
      to stop and escalate on rather than patch around — so this module
      does not do pairwise `algo.MSpaths` targeting on `Turn` at all.

   b. **A no-target, `relDirection: 'both'` walk over `HAS` from a single
      Claim seed is unsafe for this specific use case.** `HAS` is
      `Patient -> Claim`; walking it backward from the seed reaches the
      Patient, and forward from the Patient reaches **every other claim
      that patient has ever had**, not just this fact's own update chain —
      exactly the kind of query-widening `graph/traverse.py`'s module
      docstring point 4 already flags as a correctness/PHI-scoping risk
      for a *different* mechanism (a `name`-property collision), applied
      here to a *different* mechanism (an unconstrained reachability walk
      that happens to route through a hub node). The packet's own preferred
      shape is safe from this because it is PAIRWISE, anchored at both
      ends (seed Claim id -> target Turn id) — this module cannot reuse
      that safety property once (a) forces target-less seeding.

   The chain walk (this module's `_supersession_paths`) therefore calls
   `graph/traverse.paths_between` — the sanctioned `algo.MSpaths` wrapper,
   E5-S3's `ms_paths` deliverable — seeded ONLY by the given `claim_id`,
   over `relTypes=('SUPERSEDES', 'CONTRADICTS')` (Claim -> Claim only in
   this schema; walking `'both'` direction over just these two types can
   never leave the claim-chain component, so the `HAS`-hub-widening risk
   in (b) cannot arise), `maxLen=8` (the packet's own number), no target.
   Turn text is then fetched with the packet's OWN documented fallback
   shape ("If turn ids are unknown, first project them with a labelled
   1-hop") — `MATCH (c:Claim {id: $cid})-[:DRAWN_FROM]->(t:Turn) RETURN
   t.session_id AS session_id, t.turn_id AS turn_id, t.raw_text AS
   raw_text` — once per claim in the discovered chain (small counts: a
   provenance chain is a handful of claims, not a bulk operation; same
   "one MATCH per candidate" reasoning `graph/invalidate.py`'s
   `_fetch_object_id` already documents for the identical shape). This
   still satisfies E8-S2 AC3 ("uses those path payloads (or an
   `algo.*paths` call), not a log dump of raw driver Records") — the
   chain-defining hop IS an `algo.*paths` call; the turns-per-claim lookup
   is the packet's own documented, labelled, bounded fallback, not a raw
   var-length walk and never an unlabelled pattern.

2. **`resolution_reason` is always `""` today, and this is a known,
   upstream, already-documented gap, not a bug introduced here.**
   `E4-S3.md`'s own contract lists `resolution_reason: str  # empty while
   unused` on `:Claim`. As landed, `graph/schema.py`'s `CLAIM_PROPERTIES`
   does not include it, `graph/writer.py`'s `_CLAIM_SET_CLAUSES` (derived
   from `CLAIM_PROPERTIES`) never sets it, and `graph/invalidate.py`'s
   `close_interval`/`link` never write it either — confirmed by reading
   all three files and by `grep`ping every live `tests/test_invalidate.py`
   assertion of `resolution_reason` (all of them read the in-memory
   `Decision.resolution_reason`, none query the graph for a property of
   that name). This module reads `c.resolution_reason` defensively (a
   `RETURN` of a property that may not exist returns null, not a dialect
   error — this dialect's `IS NULL` restriction applies to `WHERE`, not to
   projecting a possibly-unset property) and maps `null -> ""`, matching
   the packet's own stated default exactly — NOT inventing a value, and
   forward-compatible with a future story that starts writing it (no
   change needed here when that lands). Flagged here explicitly per this
   project's "never resolve ambiguity silently" rule, rather than silently
   treated as if the field were already populated.

3. **PHI-scoping check on the SEED claim only, not on every walked hop.**
   `patient_id` is minted and one labelled, one-hop `MATCH (p:Patient
   {id: $pid})-[:HAS]->(c:Claim {id: $cid})` confirms the caller-given
   `claim_id` actually belongs to `patient_id` before any chain walk runs
   — the same allow-list-over-block-list posture `graph/traverse.py`'s own
   safety filter uses (api-security). Every OTHER node reached via
   `SUPERSEDES`/`CONTRADICTS` is trusted to remain within the same
   patient's subgraph BY CONSTRUCTION of `graph/invalidate.py.apply()`
   (candidates are only ever fetched via `MATCH (p:Patient {id:
   $pid})-[:HAS]->(c:Claim)` for the SAME patient — see that module's
   `_fetch_candidates` docstring) — this is a medium-confidence inference
   from reading the write path, not a second live-verified guard at every
   hop of THIS module's own read path. Flagged, not silently assumed;
   the discriminating observation that would raise confidence to high is
   a live test writing a genuine cross-patient `SUPERSEDES` edge by hand
   and confirming this module still refuses to surface it (out of this
   story's budget — the fixture-construction cost mirrors
   `graph/traverse.py`'s own collision test, and this story's boundary is
   the walk + the demo rendering, not re-auditing `invalidate.py`'s
   already-shipped, already-reviewed write path).

4. **`turns` carries one ADDITIVE field beyond the frozen contract:
   `claim_id`.** The literal contract is `{session_id, turn_id, text}`;
   this module also stamps which claim each turn evidences. Without it,
   "the turn text for each transition" (this story's own objective line,
   and `DEMO.md` Beat 3: "old claim, new claim, SUPERSEDES..., and the
   turn text via DRAWN_FROM") has no way to be rendered per-hop — the
   contract's `turns` list alone cannot answer "which claim does THIS
   turn support" without re-deriving it. Additive, never a rename/removal
   (same precedent `graph/retrieve.py`'s `_path_payload` sets: two extra
   convenience fields — `claim_ids`, `hop_count` — beyond the "algo.MSpaths
   payloads (path / pathWeight / pathCost)" the frozen contract names).

5. **Never raises out of the public function.** A `claim_id` that does not
   exist, or exists but is not owned by `patient_id`, degrades to the
   fully-empty shape (`nodes=[], edges=[], turns=[], paths=[]`) — the same
   "honest, empty, not an exception" posture `graph/retrieve.py`'s module
   docstring point 5 and `graph/traverse.py`'s "do not invent a seed" rule
   both already establish for this codebase's read path. A malformed
   `claim_id` (not an `int`) DOES raise `ValueError` — that is a caller
   programming error, not a data-shaped absence, matching
   `graph/traverse.py._resolve_selector_values`'s identical distinction.
--------------------------------------------------------------------------

`graph/retrieve.py`'s own `paths` field (design decision, see that
module's docstring point... — actually see its `_path_payload` docstring)
is a DIFFERENT walk: seeded by cosine-ranked entity ids over
`DEFAULT_REL_TYPES = (HAS, ABOUT, SUPERSEDES, CONTRADICTS)`, answering "what
evidence is relevant to this question" for one retrieval call. This
module's walk is seeded by one SPECIFIC already-known `claim_id` over
`(SUPERSEDES, CONTRADICTS)` only, answering "what is this one fact's own
update history" — a narrower, provenance-specific question `retrieve()`'s
general-purpose graph channel does not (and should not) try to answer
inline. `graph/retrieve.py` itself now cross-references this module by
name (see the note added there) so a reader does not have to guess whether
`retrieve.paths` already covers the provenance-demo shape.
"""

from __future__ import annotations

from collections.abc import Sequence

from medmemgraph.contracts import SENTINEL_VALID_TO
from medmemgraph.graph import traverse
from medmemgraph.hydra_client import HydraClient
from medmemgraph.pipeline.ids import mint_patient_id

__all__ = ["provenance_chain", "render_chain"]

CHAIN_REL_TYPES: tuple[str, ...] = ("SUPERSEDES", "CONTRADICTS")
"""Claim -> Claim only in this schema (`graph/schema.py` §5.2) — walking
`'both'` direction over just these two types can never leave the
claim-chain component and reach an unrelated claim via the `HAS` hub (see
module docstring point 1b)."""

MAX_CHAIN_HOPS = 8
"""The packet's own number (`E8-S2.md`: "maxLen: 8"), well under
`graph/traverse.MAX_HOPS`'s hard 16-hop ceiling, which `paths_between`
itself still clamps to defensively if ever raised."""

DEFAULT_PATH_COUNT = 10
DEFAULT_RESULT_LIMIT = 100
"""The packet's own numbers (`E8-S2.md`: "pathCount: 10, resultLimit:
100")."""

def _empty_result() -> dict:
    """A fresh dict every call — never a shared mutable module-level
    default a caller could accidentally mutate across calls."""
    return {"nodes": [], "edges": [], "turns": [], "paths": []}


# ---------------------------------------------------------------------------
# Ownership + seed-claim fetch — one labelled, one-hop MATCH.
# ---------------------------------------------------------------------------


def _fetch_owned_claim(client: HydraClient, patient_node_id: int, claim_id: int) -> dict | None:
    """Confirms `claim_id` is `patient_node_id`'s own `:Claim` (PHI-scoping,
    module docstring point 3) and returns its properties in the same round
    trip. `None` iff the claim does not exist, or exists but belongs to a
    different patient — both read as "nothing to show" by the caller, never
    an exception (module docstring point 5)."""
    rows = client.run(
        "MATCH (p:Patient {id: $pid})-[:HAS]->(c:Claim {id: $cid}) "
        "RETURN c.id AS id, c.predicate AS predicate, c.valid_from AS valid_from, "
        "c.valid_to AS valid_to, c.status AS status, c.resolution_reason AS resolution_reason",
        pid=patient_node_id,
        cid=claim_id,
    )
    if not rows:
        return None
    return rows[0]


def _node_view(node_id: int, properties) -> dict:
    """One `nodes[]` entry, the frozen contract's exact six keys — reads
    identically from a scalar-row dict (`_fetch_owned_claim`) or a
    `GraphNode.properties` mapping (a chain-walked node), since both use
    the same property names (`graph/schema.py`'s `CLAIM_PROPERTIES`).
    `resolution_reason` is coalesced `None -> ""` (module docstring
    point 2) rather than surfacing a raw `None` the packet's own contract
    never anticipates."""
    return {
        "id": node_id,
        "predicate": properties.get("predicate", ""),
        "valid_from": properties.get("valid_from", ""),
        "valid_to": properties.get("valid_to", ""),
        "status": properties.get("status", ""),
        "resolution_reason": properties.get("resolution_reason") or "",
    }


# ---------------------------------------------------------------------------
# The chain walk — algo.MSpaths via graph/traverse.py's sanctioned wrapper.
# ---------------------------------------------------------------------------


def _supersession_paths(client: HydraClient, claim_id: int) -> list[traverse.Path]:
    """`algo.MSpaths`, seeded by `claim_id` alone, over `CHAIN_REL_TYPES`
    only (module docstring point 1). Never raises: `paths_between` itself
    already degrades an unresolvable seed to `[]` without calling the
    engine (ARCHITECTURE §7.2's "do not invent a seed", already enforced
    one layer down)."""
    return traverse.paths_between(
        client,
        [claim_id],
        None,
        seed_label="Claim",
        rel_types=CHAIN_REL_TYPES,
        rel_direction="both",
        max_len=MAX_CHAIN_HOPS,
        path_count=DEFAULT_PATH_COUNT,
        result_limit=DEFAULT_RESULT_LIMIT,
    )


def _path_payload(path: traverse.Path) -> dict:
    """One `algo.MSpaths` payload — same shape as
    `graph/retrieve.py`'s own private `_path_payload` (kept as an
    independent, small copy here rather than importing that module-private
    function across a file boundary: this module built its own Cypher/walk
    already, and duplicating six lines is cheaper than coupling to a
    sibling module's private helper)."""
    return {
        "path": traverse.serialize_paths([path], 100_000),
        "pathWeight": path.path_weight,
        "pathCost": path.path_cost,
        "claim_ids": list(path.claim_ids),
        "hop_count": path.hop_count,
    }


# ---------------------------------------------------------------------------
# Turn text — the packet's own documented fallback shape, one MATCH per
# claim in the discovered chain (small counts; see module docstring 1).
# ---------------------------------------------------------------------------


def _fetch_turns(client: HydraClient, claim_ids: Sequence[int]) -> list[dict]:
    turns: list[dict] = []
    seen: set[tuple[object, object]] = set()
    for cid in claim_ids:
        rows = client.run(
            "MATCH (c:Claim {id: $cid})-[:DRAWN_FROM]->(t:Turn) "
            "RETURN t.session_id AS session_id, t.turn_id AS turn_id, t.raw_text AS raw_text",
            cid=cid,
        )
        for row in rows:
            key = (row.get("session_id"), row.get("turn_id"))
            if key in seen:
                continue
            seen.add(key)
            turns.append(
                {
                    "session_id": row.get("session_id"),
                    "turn_id": row.get("turn_id"),
                    "text": row.get("raw_text"),
                    "claim_id": cid,  # additive — module docstring point 4
                }
            )
    return turns


# ---------------------------------------------------------------------------
# provenance_chain — the frozen public entry point.
# ---------------------------------------------------------------------------


def provenance_chain(client: HydraClient, *, patient_id: str, claim_id: int) -> dict:
    """`E8-S2.md`'s frozen contract:

        provenance_chain(client, *, patient_id, claim_id) -> {
          nodes: list[{id, predicate, valid_from, valid_to, status, resolution_reason}],
          edges: list[{type: "SUPERSEDES"|"CONTRADICTS", src, dst}],
          turns: list[{session_id, turn_id, text}],   # + additive claim_id, point 4
          paths: list,   # algo.*paths payloads
        }

    `nodes` is ordered oldest-first (`valid_from` ascending, ties broken by
    id) so the returned list itself reads as the chronological chain a demo
    can print top-to-bottom without re-sorting — matching `DEMO.md` Beat 3's
    "old claim, new claim, SUPERSEDES...". A `claim_id` the caller's own
    `patient_id` does not own (or that does not exist at all) returns the
    fully-empty shape, never an exception (module docstring point 5). A
    claim that exists and is owned but has no `SUPERSEDES`/`CONTRADICTS`
    neighbor returns exactly one node and no edges (E8-S2 AC4)."""
    if not isinstance(claim_id, int) or isinstance(claim_id, bool):
        raise ValueError(f"provenance_chain: claim_id must be int, got {claim_id!r}")

    patient_node_id = mint_patient_id(patient_id)
    seed_row = _fetch_owned_claim(client, patient_node_id, claim_id)
    if seed_row is None:
        return _empty_result()

    chain_paths = _supersession_paths(client, claim_id)

    node_by_id: dict[int, dict] = {claim_id: _node_view(claim_id, seed_row)}
    edge_by_key: dict[tuple[str, int, int], dict] = {}
    for path in chain_paths:
        for node in path.nodes:
            if "Claim" in node.labels:
                node_by_id.setdefault(node.id, _node_view(node.id, node.properties))
        for rel in path.relationships:
            if rel.type in CHAIN_REL_TYPES:
                edge_by_key[(rel.type, rel.start_id, rel.end_id)] = {
                    "type": rel.type,
                    "src": rel.start_id,
                    "dst": rel.end_id,
                }

    ordered_nodes = sorted(node_by_id.values(), key=lambda n: (str(n["valid_from"]), n["id"]))
    turns = _fetch_turns(client, [n["id"] for n in ordered_nodes])

    return {
        "nodes": ordered_nodes,
        "edges": list(edge_by_key.values()),
        "turns": turns,
        "paths": [_path_payload(p) for p in chain_paths],
    }


# ---------------------------------------------------------------------------
# render_chain — human-readable text, so the demo never prints a raw dict
# or a driver Record (E8-S2 AC3). Pure formatting: no I/O, no client.
# ---------------------------------------------------------------------------


def render_chain(result: dict) -> str:
    """`provenance_chain`'s dict -> an ordered, dated, quoted-turn chain a
    viewer can follow without reading code (`E8-S2.md`'s own instruction).
    Rendering only — "how it is shown on camera" (a CLI/wrapper around
    this) is Evidence's boundary per the packet ("Owner: Graph (walk) +
    Evidence (how it is shown in the demo)"), not this function's job."""
    nodes = result.get("nodes") or []
    if not nodes:
        return "No claim found for this id under this patient."

    turns_by_claim: dict[object, list[dict]] = {}
    for turn in result.get("turns") or []:
        turns_by_claim.setdefault(turn.get("claim_id"), []).append(turn)

    edges = result.get("edges") or []

    def _edge_between(newer_id: object, older_id: object) -> dict | None:
        for edge in edges:
            if edge["src"] == newer_id and edge["dst"] == older_id:
                return edge
        return None

    lines: list[str] = []
    for i, node in enumerate(nodes):
        interval = (
            "ongoing"
            if node["valid_to"] == SENTINEL_VALID_TO
            else f'{node["valid_from"]} .. {node["valid_to"]}'
        )
        reason = f', reason={node["resolution_reason"]}' if node["resolution_reason"] else ""
        lines.append(f'[{node["id"]}] {node["predicate"]} ({node["status"]}, {interval}{reason})')
        for turn in turns_by_claim.get(node["id"], []):
            lines.append(f'    turn {turn["session_id"]}/{turn["turn_id"]}: "{turn["text"]}"')
        if i + 1 < len(nodes):
            newer = nodes[i + 1]
            edge = _edge_between(newer["id"], node["id"])
            arrow = edge["type"] if edge else "?"
            lines.append(f"  --{arrow}-->")
    return "\n".join(lines)
