"""graph/retrieve.py — `retrieve(question, patient_id, k) -> RetrieveResult`
(ARCHITECTURE.md §7, CONTRACT 2 in `contracts.py`; stories E5-S4 + E6-S2).
This is Evidence's ONE entry point into Graph's whole read path:

    route -> seed -> algo.MSpaths OR vector/lexical -> RRF fusion ->
    structural_absence -> pack

Ownership boundary: Evidence calls `retrieve()`; it does not reimplement
routing, fusion, or the abstention boolean (`router.py`/`fusion.py` own
those; this module orchestrates them). Pipeline does not issue MATCH.

--------------------------------------------------------------------------
Design decisions this module makes explicit rather than silently baking in
(each is a stated engineering judgment, flagged here for a reviewer to
agree with or push back on, not asserted as the one obvious reading of the
story packet):

1. **Patient existence is checked once, cheaply, on every route** (a single
   labelled `MATCH`, `graph/existence.exists`) — not only on the graph
   route. This is NOT the banned "pay for a graph walk on the vector route"
   (ARCHITECTURE §7.1's 26-point / 100-350x lesson refers to `algo.MSpaths`,
   never to a single O(1) labelled existence probe). A patient id that was
   never written at all has no data under ANY channel, so failing this one
   check early, before touching either arm, is strictly correct and cheap.
   §7.6 rule 3 ("vector routes do not set the flag from low cosine... unless
   a labelled seed check was also run and failed") explicitly permits this.

2. **`seed_entity_ids` is a soft, cosine-ranked top-k over the PATIENT'S
   OWN domain-entity names** (§7.2's own "brute-force cosine against the
   entity-name matrix" instruction), fetched fresh per call via one
   labelled `MATCH` per `schema.DOMAIN_ENTITY_LABELS` value — never
   invented, never unlabelled. This is coarser than true per-mention entity
   linking (which would require NER over the question, out of this story's
   scope): if a patient has ANY domain entities at all, `seed_entity_ids`
   will always return the top-k most similar ones, even for a question
   about something that specific patient's record does not actually
   contain. The `structural_absence` signal this module can honestly
   produce is therefore: "this patient has no domain entities to seed a
   walk from at all" (a real, testable, decisions/003-safe absence) — not
   "the specific entity named in the question does not exist for this
   patient" (a strictly finer-grained claim this architecture's read path,
   as specced, cannot make without an NER step nobody has built). Stated
   here rather than overclaimed in a docstring or a demo.

3. **`route=="hybrid"` degrades to text-only (`route="vector"` in the
   returned shape) rather than firing `structural_absence`, if the graph
   arm comes back empty.** §7.6's rules are written "for graph route"; a
   router-chosen `hybrid` already means "uncertain", and this module reads
   an empty graph corridor under an already-uncertain route as "answer from
   whatever text evidence exists" rather than "abstain" — Evidence still
   gets an honest `route` field (`"vector"`, not a lying `"hybrid"` with no
   graph contribution) and can layer its own refuse-vs-answer policy on top
   (§7.6's own closing line: "Evidence owns the policy on top"). A
   router-chosen `graph` route, by contrast, follows §7.6 literally:
   `structural_absence=True`, `items=[]`.

4. **Dense/lexical indexes are built lazily per `patient_id` and cached
   in-process** (`_INDEX_CACHE`), from `pipeline.loader.load_conversation`
   — the same allowlisted, packet-leakage-safe loader every other module in
   this project uses (decisions/001). `register_indexes()` is the test/
   production injection point for a caller that already has indexes built
   (or wants to test this module without touching the real MedLoCoMo
   corpus or paying an embedding cost per test).

5. **Never raises out of the public function** (E5-S4 AC4, extended here to
   both arms): a HydraDB outage, a missing/never-ingested patient
   directory, or any other failure degrades to whatever evidence the OTHER
   arm can still produce, always returning the frozen `RetrieveResult`
   shape with `paths=[]`.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import os
import time

import numpy as np

from medmemgraph.contracts import RetrieveItem, RetrieveResult
from medmemgraph.graph import fusion, schema, traverse
from medmemgraph.graph.existence import exists
from medmemgraph.graph.lexical import LexicalIndex
from medmemgraph.graph.router import DEFAULT_EPSILON, log_route_decision, route_eval, route_live
from medmemgraph.graph.vector_index import PatientIndex, embed_texts
from medmemgraph.hydra_client import HydraClient
from medmemgraph.observability import span
from medmemgraph.pipeline.ids import mint_patient_id
from medmemgraph.pipeline.loader import Conversation, LoaderError, load_conversation

__all__ = [
    "retrieve_for_eval",
    "EVAL_EPSILON",
    "retrieve",
    "seed_entity_ids",
    "register_indexes",
    "clear_index_cache",
]

DEFAULT_SEED_K = 8
"""§7.2's own worked shape ("top entity integer ids") — 8 is a small,
generous top-k for a per-patient entity-name matrix that is typically tens,
not thousands, of rows; overridable via `seed_entity_ids`'s own `k=`."""

DEFAULT_GRAPH_MAX_HOPS = 8
"""Patient -HAS-> Claim -ABOUT-> Entity is 2 hops; a SUPERSEDES chain adds a
small constant more. 8 comfortably covers this project's schema depth while
staying well under the engine's hard `maxLen<=16` ceiling
(`traverse.MAX_HOPS`) — `traverse.paths_between` itself still clamps
defensively if a caller ever raises this."""

_TOKEN_BUDGET_PER_PATH = 100_000
"""Effectively-unbounded per-call budget passed to `traverse.serialize_paths`
when rendering exactly ONE path into a `RetrieveItem.text` — this module
needs "render this one path", not "greedily pack a ranked path list into a
budget" (that packing already happened via `rank_paths` + this module's own
`k` truncation on the fused list)."""


# ---------------------------------------------------------------------------
# Text arm — dense (NumPy) + lexical (bm25s), fused to `text_rank`.
# ---------------------------------------------------------------------------

_INDEX_CACHE: dict[tuple[str, str | None], tuple[PatientIndex, LexicalIndex]] = {}


def register_indexes(
    patient_id: str,
    dense: PatientIndex,
    lexical: LexicalIndex,
    *,
    root: str | os.PathLike[str] | None = None,
) -> None:
    """Pre-populate the module-level index cache for `patient_id` without
    going through `pipeline.loader.load_conversation` — the injection point
    for tests (a small synthetic `Conversation`, no MedLoCoMo corpus, no
    embedding cost beyond the synthetic corpus) and for any production
    caller that already builds/persists these indexes elsewhere (e.g. an
    ingest pipeline that calls `PatientIndex.save`/`.load` — see
    `vector_index.py`)."""
    _INDEX_CACHE[(patient_id, str(root) if root is not None else None)] = (dense, lexical)


def clear_index_cache() -> None:
    """Test-hygiene helper: drop every cached index so the next call to
    `retrieve()` rebuilds/re-fetches from scratch."""
    _INDEX_CACHE.clear()


def _get_indexes(
    patient_id: str, *, root: str | os.PathLike[str] | None = None
) -> tuple[PatientIndex, LexicalIndex] | None:
    """Returns cached indexes, building+caching them on first use via the
    allowlisted loader (decisions/001). Returns `None` — never raises —
    when the patient has no ingestible conversation at all (never
    ingested, or a demo/test `patient_id` with no corpus backing): a
    missing text corpus degrades the text arm to `[]`, it does not fail
    `retrieve()` (design decision 5, module docstring)."""
    cache_key = (patient_id, str(root) if root is not None else None)
    cached = _INDEX_CACHE.get(cache_key)
    if cached is not None:
        return cached
    try:
        conversation: Conversation = load_conversation(patient_id, root=root)
    except (LoaderError, FileNotFoundError, OSError):
        return None
    dense = PatientIndex(patient_id)
    dense.build(conversation)
    lexical = LexicalIndex(patient_id)
    lexical.build(conversation)
    _INDEX_CACHE[cache_key] = (dense, lexical)
    return dense, lexical


def _text_channel(
    patient_id: str, question: str, k: int, *, root: str | os.PathLike[str] | None = None
) -> tuple[list[RetrieveItem], dict[str, float]]:
    """Always-built dense+lexical arm (ARCHITECTURE §7.5 step 1: "Always
    build dense + lexical. Fuse to `text_rank` via RRF.") — cheap relative
    to a graph walk, run regardless of route so the graph route's own
    "did text contribute anything unique" check (§7.5 step 3) has
    something to compare against."""
    latency = {"vector": 0.0, "lexical": 0.0}
    indexes = _get_indexes(patient_id, root=root)
    if indexes is None:
        return [], latency
    dense, lexical = indexes

    t0 = time.monotonic()
    with span("search", kind="RETRIEVER", channel="vector", k=k) as sp:
        dense_hits = dense.search(question, k)
        sp.set_attribute("candidate_count", len(dense_hits))
    latency["vector"] = (time.monotonic() - t0) * 1000

    t0 = time.monotonic()
    with span("search", kind="RETRIEVER", channel="lexical", k=k) as sp:
        lexical_hits = lexical.search(question, k)
        sp.set_attribute("candidate_count", len(lexical_hits))
    latency["lexical"] = (time.monotonic() - t0) * 1000

    with span("fuse", kind="CHAIN", channel="text") as sp:
        fused = fusion.rrf_fuse([dense_hits, lexical_hits])
        sp.set_attribute("candidate_count", len(fused))
    return fused, latency


# ---------------------------------------------------------------------------
# Graph arm — seed -> algo.MSpaths -> rank -> time-filter.
# ---------------------------------------------------------------------------


def _fetch_patient_entities(client: HydraClient, patient_node_id: int) -> list[tuple[int, str, str]]:
    """The per-patient "entity name matrix" §7.2 seeds from: one labelled
    `MATCH` per `schema.DOMAIN_ENTITY_LABELS` value (never an unlabelled
    pattern — decisions/003), unioned client-side. A patient with zero
    entities under a given label simply contributes nothing for that
    label — never invented. Returns `(node_id, name, label)` triples,
    de-duplicated by `node_id` (the same entity can be `ABOUT`-linked from
    more than one `:Claim`)."""
    seen: dict[int, tuple[str, str]] = {}
    for label in sorted(schema.DOMAIN_ENTITY_LABELS):
        rows = client.run(
            f"MATCH (p:Patient {{id: $pid}})-[:HAS]->(c:Claim)-[:ABOUT]->(e:{label}) "
            "RETURN e.id AS id, e.name AS name",
            pid=patient_node_id,
        )
        for row in rows:
            node_id = row["id"]
            if node_id not in seen:
                seen[node_id] = (str(row.get("name") or ""), label)
    return [(node_id, name, label) for node_id, (name, label) in seen.items()]


def seed_entity_ids(
    client: HydraClient, patient_node_id: int, question: str, *, k: int = DEFAULT_SEED_K
) -> dict[str, list[int]]:
    """ARCHITECTURE §7.2: natural-language question -> embedding ->
    brute-force cosine against THIS patient's own entity-name matrix ->
    top entity integer ids, grouped by label (`algo.MSpaths`/
    `traverse.paths_between` take one label per call — "a caller with a
    heterogeneous-label seed set calls this once per label and merges
    results client-side", `traverse.paths_between`'s own docstring).

    Empty candidate set (patient has no domain entities at all) -> `{}`
    without ever calling `embed_texts` — §7.2's own rule: "If the seed set
    is empty, skip `algo.MSpaths`... Do not invent a seed." Module
    docstring point 2 states plainly why this is coarser than true
    per-mention entity linking."""
    entities = _fetch_patient_entities(client, patient_node_id)
    if not entities:
        return {}
    names = [name for _, name, _ in entities]
    with span("embed", kind="EMBEDDING", entity_count=len(entities)) as sp:
        query_vec = embed_texts([question])[0]
        name_vecs = embed_texts(names)
        sp.set_attribute("vector_count", len(names) + 1)
    scores = name_vecs @ query_vec
    top_n = min(k, len(entities))
    order = np.argsort(-scores)[:top_n]

    by_label: dict[str, list[int]] = {}
    for i in order:
        node_id, _name, label = entities[int(i)]
        by_label.setdefault(label, []).append(node_id)
    return by_label


def _path_to_item(path: traverse.Path) -> RetrieveItem:
    """One `algo.MSpaths` path rendered as a `RetrieveItem` for fusion.
    `turn_ids` is deliberately `[]`: `:Claim` nodes do not carry a
    `turn_ids` property on the wire (`schema.CLAIM_PROPERTIES` has no such
    field — `graph/writer.py` never writes one), so this is an honest gap,
    not a silently-dropped value. `session_id` is the first `:Claim` on the
    path's own `session_id` property, if any."""
    session_id = ""
    for node in path.nodes:
        if "Claim" in node.labels:
            candidate = node.properties.get("session_id")
            if candidate:
                session_id = str(candidate)
                break
    text = traverse.serialize_paths([path], _TOKEN_BUDGET_PER_PATH)
    return RetrieveItem(text=text, session_id=session_id, turn_ids=[], score=path.path_weight, channel="graph")


def _path_payload(path: traverse.Path) -> dict:
    """One provenance-demo entry — the "algo.MSpaths payloads (path /
    pathWeight / pathCost)" the frozen contract names, plus two convenience
    fields (`claim_ids`, `hop_count`) a demo/prompt can use directly
    without re-walking `path.nodes`.

    E8-S2 note (documented per that story's own instruction to "fill/
    document `paths` for this walk if not already present"): this is
    already the general-purpose graph-channel walk's own path payload —
    seeded by cosine-ranked entity ids, over
    `DEFAULT_GRAPH_MAX_HOPS`/`DEFAULT_REL_TYPES` (`HAS, ABOUT, SUPERSEDES,
    CONTRADICTS`), answering "what evidence is relevant to this question"
    for one `retrieve()` call. `demo/provenance.py`'s own
    `provenance_chain()` is a DIFFERENT, narrower walk — seeded by one
    already-known `claim_id`, over `SUPERSEDES`/`CONTRADICTS` only,
    answering "what is this one fact's own update history" — not a second
    ad-hoc matcher duplicating this one, but a distinct question this
    module's own seeding (patient-wide entity cosine, not a specific claim
    id) cannot answer inline without changing what `retrieve()` returns.
    Nothing about fusion, routing, or this shape changes here."""
    return {
        "path": traverse.serialize_paths([path], _TOKEN_BUDGET_PER_PATH),
        "pathWeight": path.path_weight,
        "pathCost": path.path_cost,
        "claim_ids": list(path.claim_ids),
        "hop_count": path.hop_count,
    }


def _graph_channel(
    client: HydraClient, patient_node_id: int, question: str, *, as_of: str | None
) -> tuple[list[RetrieveItem], list[traverse.Path], bool]:
    """Returns `(items, ranked_paths, absent)`. `absent` is True iff
    seeding produced nothing (§7.6 rule 4: patient exists but no
    name-seeds -> path-absence) OR `algo.MSpaths` returned no path within
    the hop bound (§7.6 rule 2) — the two structural-absence conditions
    this module can fire, both computed via labelled lookups only."""
    by_label = seed_entity_ids(client, patient_node_id, question)
    if not by_label:
        return [], [], True

    all_paths: list[traverse.Path] = []
    with span(
        "traverse", kind="RETRIEVER", labels=sorted(by_label.keys()), max_hops=DEFAULT_GRAPH_MAX_HOPS
    ) as sp:
        for label, ids in by_label.items():
            all_paths.extend(
                traverse.paths_between(client, ids, seed_label=label, max_len=DEFAULT_GRAPH_MAX_HOPS)
            )
        sp.set_attribute("path_count", len(all_paths))

    if as_of:
        all_paths = traverse.filter_paths_by_time(all_paths, as_of)

    if not all_paths:
        return [], [], True

    with span("rerank", kind="RERANKER", candidate_count=len(all_paths)) as sp:
        ranked = traverse.rank_paths(all_paths, as_of=as_of)
    items = [_path_to_item(p) for p in ranked]
    return items, ranked, False


def _patient_exists(client: HydraClient, patient_node_id: int) -> bool:
    """§7.6 rule 1: labelled `MATCH` on the expected label, via the ONE
    sanctioned existence check (`graph/existence.exists`, decisions/003).
    Cheap (a single-row lookup), run on every route (module docstring
    point 1) — not the banned "pay for a graph walk on the vector route"."""
    return exists("Patient", patient_node_id, client=client)


# ---------------------------------------------------------------------------
# retrieve() — the frozen public entry point.
# ---------------------------------------------------------------------------


def _open_client() -> HydraClient:
    return HydraClient(transport="bolt")


def retrieve(
    question: str,
    patient_id: str,
    k: int,
    *,
    scope: str | None = None,
    question_type: str | None = None,
    as_of: str | None = None,
    client: HydraClient | None = None,
    epsilon: float = DEFAULT_EPSILON,
    rng=None,
    root: str | os.PathLike[str] | None = None,
) -> RetrieveResult:
    """`question` + `patient_id` (never crossed with any other patient —
    every graph query below is scoped by this one patient's minted node id)
    + `k` -> the frozen `RetrieveResult` shape (`contracts.py`).

    `scope`/`question_type` (gold labels, present at eval time via
    `pipeline.loader.load_qa`) route through `router.route_eval`; when both
    are `None` (the live-demo case, no gold labels), `router.route_live`'s
    deterministic keyword heuristic is used instead. `client`: reuse an
    open `HydraClient` (e.g. a test fixture, or a hot demo loop) — if
    omitted, and the router decision needs the graph arm at all, a
    short-lived one is opened and closed around this one call. Passing an
    explicit `client=` while the resolved route is `"vector"` is honored
    (module docstring point 1's cheap patient-existence probe still needs
    one) but a fresh client is never opened just to answer a pure-vector
    question if the caller supplied none and the text arm alone already
    satisfies the router's decision — see the early return below.

    Wrapped in one outer `span("retrieve", ...)` (`medmemgraph.observability`
    — a genuine no-op unless tracing is enabled) so a trace shows this
    call's own summary attributes (route, structural_absence, candidate/path
    counts, the per-stage latency this function already measures) alongside
    the route/embed/search/traverse/rerank/fuse child spans the stages below
    emit on their own. `_retrieve_impl` below is the unchanged original
    implementation; this function does not alter its behavior.
    """
    with span(
        "retrieve", kind="CHAIN", patient_id=patient_id, question_type=question_type, k=k
    ) as sp:
        result = _retrieve_impl(
            question,
            patient_id,
            k,
            scope=scope,
            question_type=question_type,
            as_of=as_of,
            client=client,
            epsilon=epsilon,
            rng=rng,
            root=root,
        )
        sp.set_attributes(
            {
                "route": result.route,
                "structural_absence": result.structural_absence,
                "candidate_count": len(result.items),
                "path_count": len(result.paths),
                "latency_vector_ms": result.latency_ms.get("vector"),
                "latency_lexical_ms": result.latency_ms.get("lexical"),
                "latency_graph_ms": result.latency_ms.get("graph"),
                "latency_search_ms": result.latency_ms.get("search"),
                "latency_total_ms": result.latency_ms.get("total"),
            }
        )
        return result


def _retrieve_impl(
    question: str,
    patient_id: str,
    k: int,
    *,
    scope: str | None = None,
    question_type: str | None = None,
    as_of: str | None = None,
    client: HydraClient | None = None,
    epsilon: float = DEFAULT_EPSILON,
    rng=None,
    root: str | os.PathLike[str] | None = None,
) -> RetrieveResult:
    """The actual read-path implementation `retrieve()` (above) wraps in
    one outer trace span. Everything below is unchanged from before this
    story added tracing, except for the route/search/fuse `with span(...)`
    blocks inline at each stage boundary — see `retrieve()`'s own docstring
    for the full public contract."""
    start = time.monotonic()
    latency_ms: dict[str, float] = {"search": 0.0, "total": 0.0, "graph": 0.0, "vector": 0.0, "lexical": 0.0}

    with span("route", kind="CHAIN", question_type=question_type, scope=scope) as sp:
        if scope is not None or question_type is not None:
            decision = route_eval(scope, question_type, epsilon=epsilon, rng=rng)
        else:
            decision = route_live(question, epsilon=epsilon, rng=rng)
        sp.set_attribute("route", decision.route)
    log_route_decision(
        question=question, features={"scope": scope, "question_type": question_type}, decision=decision
    )

    text_rank, text_latency = _text_channel(patient_id, question, k, root=root)
    latency_ms["vector"] = text_latency["vector"]
    latency_ms["lexical"] = text_latency["lexical"]

    if decision.route == "vector":
        # §7.1's own 26-point / 100-350x lesson: do NOT pay for a graph
        # walk on this route. No client is opened at all.
        latency_ms["total"] = (time.monotonic() - start) * 1000
        return RetrieveResult(
            items=text_rank[:k], route="vector", structural_absence=False, paths=[], latency_ms=latency_ms
        )

    # route is "graph" or "hybrid" — both need the engine.
    owns_client = client is None
    try:
        active_client = client if client is not None else _open_client()
    except Exception:
        # Engine unreachable before a single query was even issued —
        # degrade to whatever the text arm already produced (design
        # decision 5).
        latency_ms["total"] = (time.monotonic() - start) * 1000
        return RetrieveResult(
            items=text_rank[:k], route="vector", structural_absence=False, paths=[], latency_ms=latency_ms
        )

    try:
        patient_node_id = mint_patient_id(patient_id)

        t0 = time.monotonic()
        patient_present = _patient_exists(active_client, patient_node_id)
        latency_ms["search"] = (time.monotonic() - t0) * 1000

        if not patient_present:
            latency_ms["total"] = (time.monotonic() - start) * 1000
            return RetrieveResult(
                items=[], route=decision.route, structural_absence=True, paths=[], latency_ms=latency_ms
            )

        t0 = time.monotonic()
        graph_items, ranked_paths, graph_absent = _graph_channel(
            active_client, patient_node_id, question, as_of=as_of
        )
        latency_ms["graph"] = (time.monotonic() - t0) * 1000

        if graph_absent:
            if decision.route == "hybrid":
                # Design decision 3: an already-uncertain route degrades to
                # text-only rather than abstaining.
                latency_ms["total"] = (time.monotonic() - start) * 1000
                return RetrieveResult(
                    items=text_rank[:k],
                    route="vector",
                    structural_absence=False,
                    paths=[],
                    latency_ms=latency_ms,
                )
            latency_ms["total"] = (time.monotonic() - start) * 1000
            return RetrieveResult(
                items=[], route=decision.route, structural_absence=True, paths=[], latency_ms=latency_ms
            )

        with span("fuse", kind="CHAIN", channel="graph+text") as sp:
            fused = fusion.rrf_fuse([graph_items, text_rank])
            sp.set_attribute("candidate_count", len(fused))
        if decision.route == "hybrid":
            route_out = "hybrid"
        else:
            text_keys = {fusion.identity_key(item) for item in text_rank}
            graph_keys = {fusion.identity_key(item) for item in graph_items}
            # §7.5 step 3: "hybrid" only when text contributed a UNIQUE item.
            route_out = "hybrid" if (text_keys - graph_keys) else "graph"

        latency_ms["total"] = (time.monotonic() - start) * 1000
        return RetrieveResult(
            items=fused[:k],
            route=route_out,
            structural_absence=False,
            paths=[_path_payload(p) for p in ranked_paths],
            latency_ms=latency_ms,
        )
    except Exception:
        # E5-S4 AC4, extended to the graph arm: never raise out of the
        # public function. Whatever the text arm already produced still
        # answers the question, honestly labelled "vector".
        latency_ms["total"] = (time.monotonic() - start) * 1000
        return RetrieveResult(
            items=text_rank[:k], route="vector", structural_absence=False, paths=[], latency_ms=latency_ms
        )
    finally:
        if owns_client:
            active_client.close()


# ---------------------------------------------------------------------------
# Eval entry point — decisions/004 compliance
# ---------------------------------------------------------------------------

EVAL_EPSILON: float = 0.0
"""Epsilon for anything whose numbers reach a results table. Always 0.0.

`decisions/004-epsilon-router-log.md` and `inbox/006-to-claude.md` both promise,
in writing, that *"eval tables are run at epsilon=0"* so the headline route is
the frozen deterministic rule (cross-admission -> graph, single-admission ->
vector). `retrieve()` itself defaults to `DEFAULT_EPSILON` (0.05) because
E5-S1's contract specifies that for *live* use, where the exploration log is
the point.

The reconciliation audit found that nothing threaded epsilon=0 through
`eval/harness.py` or `eval/reader.py` -- both had zero references to it -- so a
reported table would silently have carried ~5% epsilon-flipped routes and
contradicted the written promise. It had not manifested only because the real
`retrieve()` was still opt-in behind `MEDMEMGRAPH_USE_REAL_RETRIEVE=1`.

Anything measured calls `retrieve_for_eval`. Anything live may call `retrieve`.
"""


def retrieve_for_eval(question: str, patient_id: str, k: int, **kwargs) -> RetrieveResult:
    """`retrieve()` pinned to the frozen deterministic router (epsilon=0).

    Use this from every evaluation path. Passing `epsilon` explicitly is a
    caller error rather than an override: a table produced with exploration on
    is not the system we claim to be reporting.
    """
    if "epsilon" in kwargs:
        raise ValueError(
            "retrieve_for_eval() pins epsilon=0 per decisions/004 (eval tables run "
            "at epsilon=0 so the headline route is the frozen rule). Call retrieve() "
            "directly if you genuinely want exploration -- but its numbers must not "
            "be reported as the system's."
        )
    return retrieve(question, patient_id, k, epsilon=EVAL_EPSILON, **kwargs)
