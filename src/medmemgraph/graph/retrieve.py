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
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping

import numpy as np

from medmemgraph.contracts import SENTINEL_VALID_TO, RetrieveItem, RetrieveResult
from medmemgraph.graph import embedders, fusion, schema, traverse
from medmemgraph.graph.existence import exists
from medmemgraph.graph.lexical import LexicalIndex
from medmemgraph.graph.reranker import reranker_from_env
from medmemgraph.graph.router import DEFAULT_EPSILON, log_route_decision, route_eval, route_live
from medmemgraph.graph.vector_index import DEFAULT_INDEX_DIR, PatientIndex, embed_texts
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
    "ReadPathConfig",
]

DEFAULT_RERANK_CANDIDATES = 50
"""Candidates fed to the reranker. `eval/retrieval_eval.py` sweeps at 100, a
number measured on a GPU; on the CPU-only deployment target 100 candidates
through a 0.6B causal reranker is seconds per query. 50 is the CPU-honest
default, tunable via MEDMEMGRAPH_RERANK_CANDIDATES."""

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

_INDEX_CACHE: dict[tuple[str, str | None, str], tuple[PatientIndex, LexicalIndex]] = {}
"""Keyed by (patient_id, root, embedder_name). The embedder is part of the key
because it became configurable: without it, swapping encoders mid-process
silently reuses the first one's vectors."""


def register_indexes(
    patient_id: str,
    dense: PatientIndex,
    lexical: LexicalIndex,
    *,
    root: str | os.PathLike[str] | None = None,
    embedder: str | None = None,
) -> None:
    """Pre-populate the module-level index cache for `patient_id` without
    going through `pipeline.loader.load_conversation` — the injection point
    for tests (a small synthetic `Conversation`, no MedLoCoMo corpus, no
    embedding cost beyond the synthetic corpus) and for any production
    caller that already builds/persists these indexes elsewhere (e.g. an
    ingest pipeline that calls `PatientIndex.save`/`.load` — see
    `vector_index.py`)."""
    key_embedder = embedder or getattr(dense, "backend_name", None) or embedders.DEFAULT_BACKEND_NAME
    _INDEX_CACHE[(patient_id, str(root) if root is not None else None, key_embedder)] = (
        dense,
        lexical,
    )


_PATIENT_ENTITY_CACHE: dict[int, list[tuple[int, str, str]]] = {}
"""`patient_node_id -> [(entity_id, name, label)]`, cached per process.

`_fetch_patient_entities` issues one labelled MATCH per domain label (7 queries)
and takes ~15s against a real patient's subgraph — and `seed_entity_ids` calls
it on EVERY question. Within one process the graph is static, so re-fetching per
question is pure waste: 24 QA items for one patient cost ~6 minutes of identical
queries.

Deliberately not invalidated on a timer. The only writer is
`pipeline/ingest.py`, which runs in its own process; a long-lived reader that
needs to observe new ingests calls `clear_index_cache()`."""


def clear_index_cache() -> None:
    """Test-hygiene helper: drop every cached index so the next call to
    `retrieve()` rebuilds/re-fetches from scratch."""
    _INDEX_CACHE.clear()
    _PATIENT_ENTITY_CACHE.clear()
    _PATIENT_ID_CACHE.clear()


@dataclass(frozen=True)
class ReadPathConfig:
    """Deployment knobs for the text arm, resolved once from the environment.

    Every field is OFF or default-valued unless explicitly set, so an unset
    environment reproduces the read path exactly as it behaved before these
    knobs existed. They are environment variables rather than `retrieve()`
    kwargs because the callers that need to vary them (`demo/agent.py`, the
    eval harness, `scripts/`) are all processes, not call sites — and because
    the point of the reranker seam is that a teammate can swap a checkpoint in
    without editing code.

        MEDMEMGRAPH_EMBED_BACKEND    embedder key (default: qwen3-0.6b)
        MEDMEMGRAPH_INDEX_DIR        load saved indexes from here (default: data/index)
        MEDMEMGRAPH_RERANKER         reranker key / HF id / local path (default: off)
        MEDMEMGRAPH_RERANK_CANDIDATES  pool size to rerank (default: 50)

    See `graph/reranker.py::spec_from_env` for the full reranker variable set.
    """

    embedder: str = embedders.DEFAULT_BACKEND_NAME
    index_dir: Path | None = None
    reranker: object | None = None
    n_candidates: int = DEFAULT_RERANK_CANDIDATES

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "ReadPathConfig":
        env = os.environ if env is None else env
        raw_dir = env.get("MEDMEMGRAPH_INDEX_DIR", str(DEFAULT_INDEX_DIR))
        try:
            n_candidates = int(env.get("MEDMEMGRAPH_RERANK_CANDIDATES") or DEFAULT_RERANK_CANDIDATES)
        except ValueError:
            n_candidates = DEFAULT_RERANK_CANDIDATES
        return cls(
            embedder=env.get("MEDMEMGRAPH_EMBED_BACKEND") or embedders.DEFAULT_BACKEND_NAME,
            index_dir=Path(raw_dir) if raw_dir else None,
            reranker=reranker_from_env(env),
            n_candidates=n_candidates,
        )


def _get_indexes(
    patient_id: str,
    *,
    root: str | os.PathLike[str] | None = None,
    cfg: ReadPathConfig | None = None,
) -> tuple[PatientIndex, LexicalIndex] | None:
    """Returns cached indexes, building+caching them on first use via the
    allowlisted loader (decisions/001). Returns `None` — never raises —
    when the patient has no ingestible conversation at all (never
    ingested, or a demo/test `patient_id` with no corpus backing): a
    missing text corpus degrades the text arm to `[]`, it does not fail
    `retrieve()` (design decision 5, module docstring).

    Tries a saved index (`cfg.index_dir`, written by
    `scripts/ingest_corpus.py`) before rebuilding. A rebuild re-embeds every
    turn of the patient's history — seconds per patient per process — which is
    pure waste across an eval run of hundreds of items. `PatientIndex.load`
    validates the saved `backend_name` and raises `BackendMismatchError` if it
    does not match the configured embedder, so a stale index from a different
    encoder can never be silently reused.

    The embedder name is part of the cache key. It has to be: the moment the
    encoder became configurable, a key of `(patient_id, root)` alone would hand
    back vectors built by whichever embedder happened to run first in the
    process."""
    cfg = cfg or ReadPathConfig.from_env()
    cache_key = (patient_id, str(root) if root is not None else None, cfg.embedder)
    cached = _INDEX_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if cfg.index_dir is not None:
        try:
            dense = PatientIndex.load(patient_id, cfg.index_dir, backend=cfg.embedder)
            lexical = LexicalIndex.load(patient_id, cfg.index_dir)
            _INDEX_CACHE[cache_key] = (dense, lexical)
            return dense, lexical
        except Exception:  # noqa: BLE001 — a cache miss/mismatch just means rebuild
            pass

    try:
        conversation: Conversation = load_conversation(patient_id, root=root)
    except (LoaderError, FileNotFoundError, OSError):
        return None
    dense = PatientIndex(patient_id, backend=cfg.embedder)
    dense.build(conversation)
    lexical = LexicalIndex(patient_id)
    lexical.build(conversation)
    _INDEX_CACHE[cache_key] = (dense, lexical)
    return dense, lexical


def _text_channel(
    patient_id: str,
    question: str,
    k: int,
    *,
    root: str | os.PathLike[str] | None = None,
    cfg: ReadPathConfig | None = None,
) -> tuple[list[RetrieveItem], dict[str, float]]:
    """Always-built dense+lexical arm (ARCHITECTURE §7.5 step 1: "Always
    build dense + lexical. Fuse to `text_rank` via RRF.") — cheap relative
    to a graph walk, run regardless of route so the graph route's own
    "did text contribute anything unique" check (§7.5 step 3) has
    something to compare against.

    The optional cross-encoder rerank stage lives HERE, on the text arm, and
    deliberately not on the final graph+text fusion. Graph items' `.text` is
    `traverse.serialize_paths(...)` — a rendered path, not prose — so a
    cross-encoder trained on clinical dialogue is out of distribution on them
    and would score them arbitrarily. The realistic failure is that it
    systematically demotes every real graph path, silently deleting the graph
    arm's contribution while `route` still reports `"graph"`. Reranking the text
    arm only also keeps the live path identical in shape to what
    `eval/retrieval_eval.py` measures."""
    cfg = cfg or ReadPathConfig.from_env()
    latency = {"vector": 0.0, "lexical": 0.0, "rerank": 0.0}
    indexes = _get_indexes(patient_id, root=root, cfg=cfg)
    if indexes is None:
        return [], latency
    dense, lexical = indexes

    # Widen the candidate pool only when something will actually re-rank it.
    n_cand = k if cfg.reranker is None else max(k, cfg.n_candidates)

    t0 = time.monotonic()
    with span("search", kind="RETRIEVER", channel="vector", k=n_cand) as sp:
        dense_hits = dense.search(question, n_cand)
        sp.set_attribute("candidate_count", len(dense_hits))
    latency["vector"] = (time.monotonic() - t0) * 1000

    t0 = time.monotonic()
    with span("search", kind="RETRIEVER", channel="lexical", k=n_cand) as sp:
        lexical_hits = lexical.search(question, n_cand)
        sp.set_attribute("candidate_count", len(lexical_hits))
    latency["lexical"] = (time.monotonic() - t0) * 1000

    with span("fuse", kind="CHAIN", channel="text") as sp:
        fused = fusion.rrf_fuse([dense_hits, lexical_hits])
        sp.set_attribute("candidate_count", len(fused))

    if cfg.reranker is None:
        # Byte-identical to the pre-reranker behaviour. Note this is an early
        # return rather than running a NoopReranker: `NoopReranker.rerank`
        # synthesizes descending scores, so routing the default path through it
        # would overwrite every `RetrieveItem.score`.
        return fused, latency

    t0 = time.monotonic()
    with span("rerank", kind="RERANKER", channel="text", model=cfg.reranker.name) as sp:
        ranked = cfg.reranker.rerank(question, [item.text for item in fused], top_k=None)
        # Truncate to 2*k — exactly the width RRF over two k-deep lists produced
        # before, so the downstream graph+text fusion sees the same cardinality
        # it always did, only reordered.
        fused = [replace(fused[i], score=score) for i, score in ranked][: 2 * k]
        sp.set_attribute("candidate_count", len(fused))
    latency["rerank"] = (time.monotonic() - t0) * 1000
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
    more than one `:Claim`).

    Cached per `patient_node_id` for the life of the process
    (`_PATIENT_ENTITY_CACHE`): this is 7 labelled queries costing ~15s against a
    real patient, and `seed_entity_ids` needs it on every single question."""
    cached = _PATIENT_ENTITY_CACHE.get(patient_node_id)
    if cached is not None:
        return cached
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
    result = [(node_id, name, label) for node_id, (name, label) in seen.items()]
    _PATIENT_ENTITY_CACHE[patient_node_id] = result
    return result

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
        # Asymmetric encoding: the question is a QUERY, the entity names are
        # documents. See `vector_index.embed_texts`'s own note on why encoding
        # both sides the same way collapses the score margins.
        query_vec = embed_texts([question], is_query=True)[0]
        name_vecs = embed_texts(names, is_query=False)
        sp.set_attribute("vector_count", len(names) + 1)
    scores = name_vecs @ query_vec
    top_n = min(k, len(entities))
    order = np.argsort(-scores)[:top_n]

    by_label: dict[str, list[int]] = {}
    for i in order:
        node_id, _name, label = entities[int(i)]
        by_label.setdefault(label, []).append(node_id)
    return by_label


DEFAULT_TIMELINE_ENTITIES = 4
"""How many seeded entities get a full timeline. Each costs one ordered claim
query plus one turn lookup per claim, so this bounds the round trips; the
remaining seeds still contribute path items as before."""

MAX_TIMELINE_CLAIMS = 12
"""Cap per entity. A patient on a long-term medication can have dozens of
claims about it; past a dozen the reader is reading a ledger, not evidence."""


def entity_timeline(
    client: HydraClient,
    patient_id: str,
    entity_id: int,
    label: str,
    *,
    limit: int = MAX_TIMELINE_CLAIMS,
) -> list[dict]:
    """EVERY claim about one entity, for one patient, in chronological order.

    This is the query top-k similarity structurally cannot express, and the
    reason this project is on a graph at all.

    The failure it fixes, observed directly on the benchmark: asked "compare the
    etiology of headache in 2160-08 and 2161-04", retrieval returned 6 items
    from 3 sessions and none of them were headache claims — while the graph held
    exactly three headache claims, at 2160-08, 2161-04 and 2163-04. Seeding had
    already picked the right entity. Everything after it went looking for the
    six most SIMILAR units instead of the ones that were actually ABOUT the
    thing being asked about.

    Complete coverage of a narrow slice beats a similar-looking sample of
    everything, for any question whose answer is a comparison or a trend.

    Dialect notes (`hydra_client.validate_dialect`): both node patterns are
    labelled, the relationship is directed, there is no `IN`, no `IS NULL`, no
    `CASE`. `ORDER BY` is engine-verified live (2026-08-19) — unlike
    `count()`/`collect()`, which the gate permits but which have no live usage
    in this tree, so ordering is done on the wire and everything else
    client-side.

    Keyed on `:Claim.patient_id`, NOT by walking `(:Patient)-[:HAS]->`: the
    `:Patient` node is a hub with an edge to every claim, and traversing it is
    what made `algo.MSpaths` exceed the engine's 30s timeout (see
    `traverse.DEFAULT_REL_TYPES`). A property lookup on the claim sidesteps the
    hub entirely.
    """
    if label not in schema.DOMAIN_ENTITY_LABELS:
        raise ValueError(f"{label!r} is not a domain entity label")
    rows = client.run(
        f"MATCH (cl:Claim {{patient_id: $pid}})-[:ABOUT]->(e:{label} {{id: $eid}}) "
        "RETURN cl.id AS id, cl.session_id AS session_id, cl.valid_from AS valid_from, "
        "cl.valid_to AS valid_to, cl.predicate AS predicate, cl.polarity AS polarity, "
        "cl.status AS status, cl.confidence AS confidence "
        "ORDER BY cl.valid_from",
        pid=patient_id,
        eid=entity_id,
    )
    return rows[:limit]


def _turn_text_for_claim(client: HydraClient, claim_id: int) -> tuple[str, list[int], str]:
    """`(text, turn_ids, session_id)` for one claim's source turns.

    Same query `demo/provenance.py::_fetch_turns` uses. This is what closes the
    `turn_ids=[]` gap `_path_to_item` documents: a graph item could previously
    say a dose changed but never quote the sentence that said so, and the gold
    answers on this benchmark are clinical content ("meningeal signs vs lupus
    flare"), not schema structure."""
    try:
        rows = client.run(
            "MATCH (c:Claim {id: $cid})-[:DRAWN_FROM]->(t:Turn) "
            "RETURN t.session_id AS session_id, t.turn_id AS turn_id, t.raw_text AS raw_text",
            cid=claim_id,
        )
    except Exception:  # noqa: BLE001 — evidence without a quote beats no evidence
        return "", [], ""
    texts, turn_ids, session_id = [], [], ""
    for r in rows:
        if r.get("raw_text"):
            texts.append(str(r["raw_text"]).strip())
        if r.get("turn_id") is not None:
            try:
                turn_ids.append(int(r["turn_id"]))
            except (TypeError, ValueError):
                pass
        session_id = session_id or str(r.get("session_id") or "")
    return " ".join(texts), turn_ids, session_id


CONTEXT_LABELS = ("Condition", "Procedure")
"""Entity labels fetched for admission context. A "compare the cause of X"
question is answered by a CONDITION, and diagnostic PROCEDUREs are what rule
causes in or out."""

MAX_CONTEXT_ADMISSIONS = 4
MAX_CONTEXT_PER_ADMISSION = 12


def admission_context(
    client: HydraClient, patient_id: str, session_id: str, *, limit: int = MAX_CONTEXT_PER_ADMISSION
) -> list[dict]:
    """What else was claimed in one admission — the co-occurrence hop.

    This exists because of a failure mode timelines alone cannot fix. Every
    `cross_admission_comparison` item on this benchmark asks to compare the
    ETIOLOGY of a symptom across dates, and every gold answer is a pair of
    causes: "meningeal signs vs lupus flare", "volume overload vs pulmonary
    edema", "liver enzyme elevation versus sepsis".

    The symptom is in the question; the cause is not. So similarity seeding —
    which embeds the QUESTION — can never retrieve it. Asked to compare the
    etiology of headache, nothing in the question resembles "lupus", and the
    system returned six items about headaches and answered "the etiologies were
    unspecified".

    The graph closes that gap structurally rather than semantically: a timeline
    tells us WHICH admissions the symptom appeared in, and the cause is
    whatever else was claimed in those same admissions. For the case above,
    admission 22661410 carries `lupus asserted` and `meningitis negated` on the
    same day as the headache — exactly the gold answer, one hop away and
    unreachable by any amount of embedding quality.

    Dialect-legal: labelled patterns, directed relationship, no `IN`/`IS NULL`/
    `CASE`. Ordered on the wire; label filtering is client-side because there is
    no `IN`."""
    out: list[dict] = []
    for label in CONTEXT_LABELS:
        try:
            rows = client.run(
                f"MATCH (cl:Claim {{patient_id: $pid, session_id: $sid}})-[:ABOUT]->(e:{label}) "
                "RETURN e.name AS name, cl.predicate AS predicate, cl.polarity AS polarity, "
                "cl.valid_from AS valid_from "
                "ORDER BY cl.valid_from",
                pid=patient_id,
                sid=session_id,
            )
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"admission context failed for {session_id}/{label}: {type(exc).__name__}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        for r in rows:
            r["label"] = label
        out.extend(rows)
    out.sort(key=lambda r: str(r.get("valid_from") or ""))
    return out[:limit]


def _context_item(session_id: str, rows: list[dict], anchor_name: str) -> RetrieveItem | None:
    """One admission's co-occurring claims, rendered compactly.

    Asserted and negated are kept separate and both are shown: "no signs of
    meningitis" is exactly as diagnostic as "lupus", and collapsing polarity
    would destroy the contrast these questions turn on."""
    if not rows:
        return None
    asserted = [r for r in rows if r.get("polarity") != "negated"]
    negated = [r for r in rows if r.get("polarity") == "negated"]
    when = str(rows[0].get("valid_from") or "?")[:10]
    lines = [
        f"CONTEXT for admission {session_id} (~{when}) — what else was on record "
        f"in the admission(s) where {anchor_name} appears:"
    ]
    if asserted:
        lines.append("  present: " + ", ".join(f"{r['name']} ({r['label']})" for r in asserted))
    if negated:
        lines.append("  ruled out: " + ", ".join(f"{r['name']} ({r['label']})" for r in negated))
    return RetrieveItem(
        text="\n".join(lines), session_id=session_id, turn_ids=[], score=0.95, channel="graph"
    )


def _timeline_to_item(
    entity_name: str, label: str, claims: list[dict], client: HydraClient
) -> RetrieveItem | None:
    """One entity's whole history rendered as a single evidence item.

    Deliberately ONE item, not one per claim: the point is that the reader sees
    the sequence together and can order it. Splitting it would put the claims
    back into competition with each other for top-k slots, which is the problem
    this is solving.

    `channel` stays `"graph"`. `contracts.Channel` is a frozen three-value
    Literal and widening it is a decisions/ file, not a quiet edit here."""
    if not claims:
        return None
    lines = [f"TIMELINE for {label}({entity_name}) — all {len(claims)} claim(s) on record, oldest first:"]
    turn_ids: list[int] = []
    session_id = ""
    for c in claims:
        quote, tids, sid = _turn_text_for_claim(client, c["id"])
        turn_ids.extend(tids)
        session_id = session_id or sid or str(c.get("session_id") or "")
        when = str(c.get("valid_from") or "?")
        state = "ongoing" if c.get("valid_to") == SENTINEL_VALID_TO else f"until {c.get('valid_to')}"
        line = (
            f"  [{when}] {c.get('predicate')} {c.get('polarity')} "
            f"({c.get('status')}, {state}, admission {c.get('session_id')})"
        )
        if quote:
            line += f' — "{quote[:220]}"'
        lines.append(line)
    return RetrieveItem(
        text="\n".join(lines),
        session_id=session_id,
        turn_ids=turn_ids[:24],
        score=1.0,
        channel="graph",
    )


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

    # Timelines FIRST. Path items answer "what is connected to what"; a timeline
    # answers "what happened to this thing, in what order" — which is the actual
    # question in every cross-admission and longitudinal item on this benchmark.
    # They lead the list so they survive the final top-k slice in _retrieve_impl.
    timeline_items = _timeline_items(client, patient_node_id, by_label)
    return timeline_items + items, ranked, False



def _timeline_items(
    client: HydraClient, patient_node_id: int, by_label: dict[str, list[int]]
) -> list[RetrieveItem]:
    """Build timelines for the top seeded entities, cheapest-first.

    Bounded by `DEFAULT_TIMELINE_ENTITIES` because each timeline costs one
    ordered query plus one turn lookup per claim. Failure of any single timeline
    is swallowed: a missing timeline degrades this back to the previous
    path-only behaviour, which is worse but not broken."""
    entities = _entity_names(client, patient_node_id)
    out: list[RetrieveItem] = []
    anchors: dict[str, list[str]] = {}
    for label, ids in by_label.items():
        if len(out) >= DEFAULT_TIMELINE_ENTITIES:
            break
        for entity_id in ids:
            # `break`, not `return` — returning here skipped the co-occurrence
            # step below entirely, and since the cap is hit on essentially every
            # real question, the context items were never emitted at all.
            if len(out) >= DEFAULT_TIMELINE_ENTITIES:
                break
            name = entities.get(entity_id)
            if not name:
                continue
            try:
                claims = entity_timeline(client, _patient_id_of(client, patient_node_id), entity_id, label)
                item = _timeline_to_item(name, label, claims, client)
            except Exception as exc:  # noqa: BLE001 — one timeline must not sink a query
                # WARN, don't swallow. A bare `continue` here hid a NameError on
                # the very first run of this code: every timeline died and the
                # channel silently degraded to the old path-only behaviour,
                # which looks exactly like "the feature is on but didn't help".
                warnings.warn(
                    f"timeline failed for {label}({name}): {type(exc).__name__}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            if item is not None:
                out.append(item)
                anchors.setdefault(name, []).extend(
                    str(c.get("session_id")) for c in claims if c.get("session_id")
                )
    out.extend(_context_items(client, _patient_id_of(client, patient_node_id), anchors))
    return out


def _context_items(
    client: HydraClient, patient_id: str, anchors: dict[str, list[str]]
) -> list[RetrieveItem]:
    """Co-occurrence context for the admissions the timelines landed in.

    Bounded by `MAX_CONTEXT_ADMISSIONS`: a symptom that recurs across a dozen
    admissions would otherwise pull the whole record back in, which is the
    behaviour this system exists to avoid."""
    seen: set[str] = set()
    items: list[RetrieveItem] = []
    for anchor_name, sessions in anchors.items():
        for session_id in sessions:
            if session_id in seen or len(items) >= MAX_CONTEXT_ADMISSIONS:
                continue
            seen.add(session_id)
            rows = admission_context(client, patient_id, session_id)
            item = _context_item(session_id, rows, anchor_name)
            if item is not None:
                items.append(item)
    return items


def _entity_names(client: HydraClient, patient_node_id: int) -> dict[int, str]:
    """`node_id -> name` for this patient's entities, off the same cached fetch
    `seed_entity_ids` uses, so this adds no queries."""
    return {eid: name for eid, name, _label in _fetch_patient_entities(client, patient_node_id)}


_PATIENT_ID_CACHE: dict[int, str] = {}


def _patient_id_of(client: HydraClient, patient_node_id: int) -> str:
    """The string `patient_id` for a minted node id. `:Claim.patient_id` stores
    the string form, and the timeline query keys on it to avoid the `:Patient`
    hub."""
    cached = _PATIENT_ID_CACHE.get(patient_node_id)
    if cached is not None:
        return cached
    rows = client.run(
        "MATCH (p:Patient {id: $pid}) RETURN p.patient_id AS patient_id", pid=patient_node_id
    )
    value = str(rows[0]["patient_id"]) if rows else ""
    _PATIENT_ID_CACHE[patient_node_id] = value
    return value


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
    latency_ms: dict[str, float] = {
        "search": 0.0,
        "total": 0.0,
        "graph": 0.0,
        "vector": 0.0,
        "lexical": 0.0,
        "rerank": 0.0,
    }

    with span("route", kind="CHAIN", question_type=question_type, scope=scope) as sp:
        if scope is not None or question_type is not None:
            decision = route_eval(scope, question_type, epsilon=epsilon, rng=rng)
        else:
            decision = route_live(question, epsilon=epsilon, rng=rng)
        sp.set_attribute("route", decision.route)
    log_route_decision(
        question=question, features={"scope": scope, "question_type": question_type}, decision=decision
    )

    cfg = ReadPathConfig.from_env()
    text_rank, text_latency = _text_channel(patient_id, question, k, root=root, cfg=cfg)
    latency_ms["vector"] = text_latency["vector"]
    latency_ms["lexical"] = text_latency["lexical"]
    # Copy EVERY key the text arm reports, not a hardcoded two. `rerank` was
    # added to `_text_channel`'s latency dict but not here, so the reranker's
    # cost — the single number the CPU-deployability decision rests on — was
    # measured, discarded, and reported as 0.0 to every caller.
    latency_ms.update(text_latency)

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
    except Exception as exc:  # noqa: BLE001 — never raise out of the public function
        # E5-S4 AC4, extended to the graph arm: never raise out of the
        # public function. Whatever the text arm already produced still
        # answers the question, honestly labelled "vector".
        #
        # WARN, don't swallow. Degrading silently here hid a total graph-arm
        # outage for an entire eval run on 2026-08-18: `algo.MSpaths` was
        # exceeding HydraDB's 30s query timeout on every real-scale patient
        # (see `traverse.DEFAULT_REL_TYPES` for the cause and fix), and because
        # this handler produced a perfectly well-formed vector-route result, the
        # numbers looked plausible and nothing anywhere said the graph had not
        # run. The degrade is still correct — a partial answer beats an
        # exception — but it must be audible.
        warnings.warn(
            f"graph arm failed for patient_id={patient_id!r}; degrading to the "
            f"text arm and reporting route='vector'. {type(exc).__name__}: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
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
