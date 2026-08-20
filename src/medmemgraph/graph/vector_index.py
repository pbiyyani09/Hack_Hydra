"""graph/vector_index.py — per-patient, in-process, brute-force-cosine vector
index. This is one half of the "beside HydraDB" arm (`lexical.py` is the
other): HydraDB OSS ships no vector index at all (confirmed by exhaustive
grep over `literature/10-hydradb-capability-audit.md`'s capability table),
so this whole layer is ours, and it runs entirely in-process — no server,
no extra container.

Authorities (copy the shape, do not re-derive it):

- `collaborative/design/ARCHITECTURE.md` §7.4 "Vector / lexical (Graph,
  beside the engine)": "Dense: one in-memory NumPy array of L2-normalized
  embeddings per patient_id. Brute-force cosine. Survey 12 measured
  0.17 ms/query at 3,000 vectors and 7.25 ms at 50,000 (AMD Ryzen 9 5900X,
  dim=1024). No FAISS, no HNSW, no ANN. Index both turn text and emitted
  fact text (survey 12: retrieve over both granularities)."
- `collaborative/literature/12-vector-search-and-hybrid-fusion.md` Q1/Q2:
  FAISS's own index-selection guidance places the approximate-ANN crossover
  at ~1M vectors [R-VEC-007/008]; this project's per-patient corpora are
  2-4 orders of magnitude below that, and brute-force cosine was measured
  directly in that survey at 0.168 ms/query (n=3,000, dim=1024) and
  7.25 ms/query even at n=50,000 [R-VEC-009/010] — no ANN library appears
  anywhere in this file, per the project's skip list.
- `collaborative/design/stories/E5/E5-S2.md` — the story this module
  implements (`PatientVectors`/`TextHit` in that packet's own sketch; this
  file follows the dispatch's own class-shape instructions — `PatientIndex`
  built and searched per patient, returning `contracts.RetrieveItem`
  directly rather than a second, field-identical dataclass — see "Contract"
  section of `docs/algorithms/vector-lexical-index.md` for the reasoning).

Banned by both the story and this project's decision log: FAISS, HNSW, any
ANN library, a clinical-specialized embedding model as the default, and a
single global matrix shared across patients (`E5-S2` "Banned approaches").

## Pluggable local embedder backend (this story)

`PatientIndex`'s *default* encoder is no longer the hashing trick below —
it is `graph/embedders.py`'s `qwen3-0.6b` local `sentence-transformers`
backend (see that module's own docstring for the full per-model
query/document prompt-convention writeup and citations). This module's own
`embed_texts`/`_hashing_embed`/`_EmbedCache`/`_embed_cached` are **kept,
unmodified**, below: they remain the live default for
`graph/retrieve.py::seed_entity_ids`'s entity-name similarity check, a
different call site this story's file list does not include and has no
reason to perturb (a different dim, a different cache schema, a different
accuracy/latency trade-off already reasoned through there). `PatientIndex`
itself now routes through `embedders.get_backend(...)` unless a caller
passes an explicit `embed_fn=` (a still-supported raw-callable escape
hatch, e.g. for a test that wants a synthetic, instant, zero-dependency
encoder) — see `PatientIndex.__init__`'s own docstring for exactly how the
two paths differ and why `cache_path`'s *default* is mode-aware.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from medmemgraph.contracts import ClinicalFact, RetrieveItem
from medmemgraph.graph import embedders
from medmemgraph.pipeline.loader import Conversation, Turn

__all__ = [
    "DEFAULT_EMBED_DIM",
    "DEFAULT_INDEX_DIR",
    "EmbedFn",
    "IndexedUnit",
    "PatientIndex",
    "embed_texts",
    "format_turn",
    "format_fact",
]

# ---------------------------------------------------------------------------
# Embedding — see docs/algorithms/vector-lexical-index.md §2 for the full
# "why not a clinical model, why not a hosted model by default" writeup.
# ---------------------------------------------------------------------------

DEFAULT_EMBED_DIM = 512
"""Hashing-trick output width. Matches `eval/baselines/dense.py`'s own
constant and citation (chosen once, independently, in each file per this
project's "code against the frozen contract, don't cross-import a private
helper from a sibling module" convention — see module docstring)."""

DEFAULT_INDEX_DIR = Path("data/index")

EmbedFn = Callable[[list[str]], np.ndarray]

_TOKEN_RE = re.compile(r"[a-z0-9]+")

_STOPWORDS = frozenset(
    {
        "the", "a", "an", "and", "or", "is", "are", "was", "were", "be", "been",
        "to", "of", "in", "on", "at", "for", "with", "it", "this", "that", "i",
        "you", "he", "she", "we", "they", "do", "does", "did", "have", "has",
        "had", "not", "no", "yes", "as", "by", "so", "if", "any", "there",
    }
)


def _tokenize(text: str) -> list[str]:
    return [tok for tok in _TOKEN_RE.findall(text.lower()) if tok not in _STOPWORDS]


def _hashing_embed(texts: list[str], dim: int = DEFAULT_EMBED_DIM) -> np.ndarray:
    """Deterministic signed-hashing-trick embedding (Weinberger, Dasgupta,
    Langford, Smola, Attenberg — "Feature Hashing for Large Scale
    Multitask Learning", ICML 2009, arXiv:0902.2206, §3): each surviving
    token is hashed via `hashlib.sha1` (stable across process restarts,
    unlike Python's salted-per-run builtin `hash()`) into a bucket index
    *and* a sign bit, so collisions partially cancel instead of purely
    accumulating. Log1p-dampened (sublinear term-frequency scaling, the
    same idea BM25/TF-IDF use) before L2 normalization. General-purpose by
    construction — it has zero domain knowledge of any kind, clinical or
    otherwise, which trivially satisfies this story's "do not default to a
    clinical-specialized model" instruction (see docs/algorithms note for
    the fuller embedding-model decision record)."""
    vectors = np.zeros((len(texts), dim), dtype=np.float32)
    for row, text in enumerate(texts):
        for tok in _tokenize(text):
            digest = hashlib.sha1(tok.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % dim
            sign = 1.0 if (digest[4] & 1) else -1.0
            vectors[row, bucket] += sign
    vectors = np.sign(vectors) * np.log1p(np.abs(vectors))
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vectors / norms).astype(np.float32)


def embed_texts(texts: list[str], *, dim: int = DEFAULT_EMBED_DIM) -> np.ndarray:
    """`(len(texts), dim)` L2-normalized float32 array — the frozen
    `E4-S2` `embed()` contract. Prefers the real Graph-owned
    `medmemgraph.embeddings.embed` if it has landed (same
    try-the-real-thing-first/fall-back pattern already used by
    `eval/baselines/dense.py::embed` and `eval/reader.py`'s
    `_default_retriever`); otherwise falls back to the deterministic
    hashing trick above, which *is* E4-S2's own specified default, not a
    competing invention. Delete this fork once `medmemgraph.embeddings`
    exists (same "delete the fork when it lands" convention `dense.py`
    already documents)."""
    if not texts:
        return np.zeros((0, dim), dtype=np.float32)
    try:
        from medmemgraph.embeddings import embed as _real_embed

        return _real_embed(texts)
    except ImportError:
        return _hashing_embed(texts, dim=dim)


def _content_hash(text: str) -> str:
    """SHA-256 of the exact indexed text — the on-disk embedding cache key
    (story requirement: "cache embeddings on disk keyed by content hash").
    `hashlib`, not the builtin `hash()`, for the same determinism-across-
    runs reason `_hashing_embed` uses `sha1` rather than `hash()`."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class _EmbedCache:
    """On-disk `content_hash -> vector` cache, one shared `.npz` file
    (embedding is a pure function of text alone, not of `patient_id`, so a
    global cache is correct, not merely convenient — the same clinical
    phrase appearing in two patients' histories embeds once). Cheap with
    the hashing-trick default; this is the load-bearing piece of
    infrastructure for the day a real hosted/local model replaces it,
    where every avoided re-embed call is real latency and, for a hosted
    model, real dollars."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._map: dict[str, np.ndarray] = {}
        if path.exists():
            with np.load(path, allow_pickle=False) as data:
                hashes = data["hashes"]
                vectors = data["vectors"]
                self._map = {h: vectors[i] for i, h in enumerate(hashes.tolist())}

    def get_many(self, texts: list[str]) -> tuple[dict[int, np.ndarray], list[int]]:
        hits: dict[int, np.ndarray] = {}
        misses: list[int] = []
        for i, text in enumerate(texts):
            vec = self._map.get(_content_hash(text))
            if vec is None:
                misses.append(i)
            else:
                hits[i] = vec
        return hits, misses

    def put_many(self, texts: list[str], vectors: np.ndarray) -> None:
        for text, vec in zip(texts, vectors):
            self._map[_content_hash(text)] = vec

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self._map:
            hashes = np.array(list(self._map.keys()))
            vectors = np.stack(list(self._map.values()))
        else:
            hashes = np.array([], dtype=str)
            vectors = np.zeros((0, DEFAULT_EMBED_DIM), dtype=np.float32)
        np.savez_compressed(self.path, hashes=hashes, vectors=vectors)


def _embed_cached(texts: list[str], embed_fn: EmbedFn, cache: "_EmbedCache | None", dim: int) -> np.ndarray:
    """`embed_fn` applied only to cache misses; hits are reused verbatim.
    Falls straight through to `embed_fn` with no cache bookkeeping when
    `cache is None` (tests that want a clean-room build)."""
    if not texts:
        return np.zeros((0, dim), dtype=np.float32)
    if cache is None:
        return embed_fn(texts)

    hits, miss_idx = cache.get_many(texts)
    vectors = np.zeros((len(texts), dim), dtype=np.float32)
    for i, vec in hits.items():
        vectors[i] = vec
    if miss_idx:
        miss_texts = [texts[i] for i in miss_idx]
        miss_vectors = embed_fn(miss_texts)
        for j, i in enumerate(miss_idx):
            vectors[i] = miss_vectors[j]
        cache.put_many(miss_texts, miss_vectors)
        cache.save()
    return vectors


# ---------------------------------------------------------------------------
# Chunking unit — literature/12 Q4: turn-level for raw dialogue (no
# pre-baked window; neighbor expansion, where it applies, happens at query
# time on the *lexical* channel per ReFind's own ablation — see
# `lexical.py`), plus atomic ClinicalFact text as a second granularity
# indexed into the same per-patient array (ARCHITECTURE.md §7.4: "Index
# both turn text and emitted fact text").
# ---------------------------------------------------------------------------


def format_turn(turn: Turn) -> str:
    """Same rendering convention `eval/baselines/dense.py`/`lexical.py`
    already use, reused here (not imported — see module docstring on why
    graph/ does not import from eval/) so a turn read via this index and a
    turn read via a baseline are byte-identical for a reviewer diffing the
    two."""
    return f"[admission {turn.hadm_id} turn {turn.turn_number} · {turn.time} · {turn.speaker}] {turn.text}"


def format_fact(fact: ClinicalFact) -> str:
    """Render one `ClinicalFact` as retrievable prose. `polarity="negated"`
    is rendered as an explicit denial (`ARCHITECTURE.md` §5.5 /
    `contracts.py`: retraction is `polarity=negated`, never a separate
    predicate) so the embedded text does not silently drop the negation —
    a fact and its denial must not embed to near-identical vectors."""
    verb = "does NOT have/report" if fact.polarity == "negated" else "has/reports"
    predicate_phrase = fact.predicate.replace("_", " ").lower()
    return (
        f"[fact {fact.fact_id} · admission {fact.session_id} · as of {fact.valid_from}] "
        f"{fact.subject.name} {verb} {predicate_phrase}: {fact.object.name}"
    )


@dataclass(frozen=True)
class IndexedUnit:
    """One row of the per-patient vector matrix: its rendered text, source
    session/turns for provenance, and which granularity it came from."""

    text: str
    session_id: str
    turn_ids: list[int]
    kind: str  # "turn" | "fact"


# ---------------------------------------------------------------------------
# PatientIndex
# ---------------------------------------------------------------------------


_UNSET_CACHE_PATH = object()
"""Sentinel distinguishing "caller left `cache_path` at its default" from
"caller explicitly passed `cache_path=<default literal path>`" —
`PatientIndex.__init__` needs this because the *meaning* of "default cache
path" is mode-aware (see below): the legacy `embed_fn=` path's default
cache file (`DEFAULT_INDEX_DIR / "embed_cache.npz"`, hash-only keys,
hashing-trick vectors) and the new `backend=` path's default cache file
(`embedders.DEFAULT_ST_CACHE_PATH`, `(model, role, hash)` keys, model
vectors) are two different, mutually-incompatible on-disk schemas. If both
paths shared one literal default path, whichever mode ran last in a
process would silently overwrite the other's cache file with its own
schema on `.save()`. A caller that explicitly passes `cache_path=None` (no
caching) or `cache_path=<some path>` (an explicit choice) is honored
exactly as given in either mode; only the *unstated* default is
mode-aware. `graph/retrieve.py::_get_indexes` calls `PatientIndex(patient_id)`
with every default left in place — this sentinel is what keeps that real
call site correct rather than hypothetical."""


class PatientIndex:
    """Per-patient, in-process brute-force-cosine vector index. One small
    NumPy array per `patient_id` — never a single global matrix (banned by
    the story: cross-patient leakage risk and it throws away the "just
    index into the right array" filtering-for-free property literature/12
    Q1/Q8 both call out as the whole reason ANN filtering machinery is
    unnecessary here).

    Two ways to supply an encoder, mutually exclusive:

    1. `backend=` (default `embedders.DEFAULT_BACKEND_NAME`, i.e.
       `"qwen3-0.6b"`) — a registry name (`"qwen3-0.6b"` | `"bge-m3"` |
       `"arctic-l-v2"` | `"openai"`) resolved via `embedders.get_backend`,
       or an already-constructed `EmbedderBackend`. Document text (turns,
       facts) and query text are routed through the backend's `encode(...,
       is_query=...)` with the correct role each time — `build()` always
       encodes documents, `search()` always encodes the query. This is the
       new default path this story adds.
    2. `embed_fn=` — a raw `Callable[[list[str]], np.ndarray]`, used
       identically for both queries and documents (no role distinction,
       the pre-this-story behavior, e.g. the module-level `embed_texts`
       hashing-trick default). Kept as an explicit escape hatch for a
       caller (or a test) that wants a synthetic, dependency-free encoder
       and does not want `embedders.py`/`sentence-transformers`/`torch`
       imported at all. Takes priority over `backend=` if both are given.

    `self.backend_name` records which encoder built this index's vectors
    (`"custom"` for the `embed_fn=` path — a raw callable has no stable
    registry name) and is persisted by `save()` / checked by `load()` so a
    stale index built with a different encoder can never be silently
    reused for search (`embedders.BackendMismatchError`)."""

    def __init__(
        self,
        patient_id: str,
        *,
        backend: "embedders.EmbedderBackend | str" = embedders.DEFAULT_BACKEND_NAME,
        embed_fn: EmbedFn | None = None,
        dim: int = DEFAULT_EMBED_DIM,
        cache_path: Path | None = _UNSET_CACHE_PATH,  # type: ignore[assignment]
    ) -> None:
        self.patient_id = patient_id
        if embed_fn is not None:
            resolved_cache_path = (
                (DEFAULT_INDEX_DIR / "embed_cache.npz") if cache_path is _UNSET_CACHE_PATH else cache_path
            )
            self.dim = dim
            self.backend_name = "custom"
            self._cache = _EmbedCache(resolved_cache_path) if resolved_cache_path is not None else None
            self._encode_doc: EmbedFn = embed_fn
            self._encode_query: EmbedFn = embed_fn
        else:
            resolved_cache_path = (
                embedders.DEFAULT_ST_CACHE_PATH if cache_path is _UNSET_CACHE_PATH else cache_path
            )
            resolved_backend = embedders.get_backend(backend, cache_path=resolved_cache_path)
            self.dim = resolved_backend.dim
            self.backend_name = resolved_backend.name
            self._cache = None  # the backend owns its own (model, role, hash)-keyed cache
            self._encode_doc = lambda texts: resolved_backend.encode(texts, is_query=False)
            self._encode_query = lambda texts: resolved_backend.encode(texts, is_query=True)
        self.units: list[IndexedUnit] = []
        self.vectors: np.ndarray = np.zeros((0, self.dim), dtype=np.float32)
        self.build_time_s: float = 0.0

    # -- build ---------------------------------------------------------

    def build(self, conversation: Conversation, facts: list[ClinicalFact] | None = None) -> None:
        """Index every turn in `conversation` (raw-dialogue granularity)
        plus, if supplied, every fact's rendered text (atomic-fact
        granularity) — both granularities in the *same* per-patient array,
        per `ARCHITECTURE.md` §7.4 and literature/12 Q4's three-way
        convergent evidence for indexing both rather than choosing one.
        `facts` is an explicit, optional argument rather than something
        this method extracts itself: wiring live extraction in here would
        couple this module to the Pipeline track's LLM-backed extractor,
        which this story does not own and which is not guaranteed
        available (no API key) in every environment this index must build
        in — callers that have facts (post-extraction) pass them; callers
        that only have raw dialogue (this story's own tests, and any
        early-pipeline caller) still get a fully working turn-level index."""
        start = time.monotonic()
        units: list[IndexedUnit] = [
            IndexedUnit(text=format_turn(turn), session_id=turn.hadm_id, turn_ids=[turn.turn_number], kind="turn")
            for turn in conversation.turns()
        ]
        if facts:
            units.extend(
                IndexedUnit(text=format_fact(fact), session_id=fact.session_id, turn_ids=list(fact.turn_ids), kind="fact")
                for fact in facts
            )
        self.units = units
        texts = [u.text for u in units]
        if not texts:
            self.vectors = np.zeros((0, self.dim), dtype=np.float32)
        elif self._cache is not None:
            # Legacy `embed_fn=` path only — the backend path's cache lives
            # inside the backend itself (see `__init__`) and is never `None`
            # here unless `embed_fn=` was given.
            self.vectors = _embed_cached(texts, self._encode_doc, self._cache, self.dim)
        else:
            self.vectors = self._encode_doc(texts)
        self.build_time_s = time.monotonic() - start

    # -- search ----------------------------------------------------------

    def search(self, query: str, k: int) -> list[RetrieveItem]:
        """Top-`k` by cosine similarity. Both sides are L2-normalized, so
        the dot product *is* cosine similarity (`ARCHITECTURE.md` §7.4,
        literature/12 Q2). Empty index -> `[]`, never an invented hit
        (E5-S2 AC2). Query text is embedded via `self._encode_query` — the
        `is_query=True` role, distinct from `build()`'s `is_query=False`
        document role, for any backend whose model card documents an
        asymmetric query/document prompt convention (`graph/embedders.py`
        module docstring)."""
        if not self.units:
            return []
        q = self._encode_query([query])[0]
        scores = self.vectors @ q
        k_eff = min(k, len(self.units))
        top_idx = np.argsort(-scores)[:k_eff]
        return [
            RetrieveItem(
                text=self.units[i].text,
                session_id=self.units[i].session_id,
                turn_ids=list(self.units[i].turn_ids),
                score=float(scores[i]),
                channel="vector",
            )
            for i in top_idx
        ]

    # -- persistence -------------------------------------------------------

    def index_size_bytes(self) -> int:
        """On-disk footprint estimate: the vectors array plus a rough JSON
        metadata estimate — reported by the story's "report build time and
        index size" acceptance line."""
        return int(self.vectors.nbytes) + sum(len(u.text.encode("utf-8")) for u in self.units)

    @staticmethod
    def _paths_for(patient_id: str, index_dir: Path) -> tuple[Path, Path]:
        safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", patient_id)
        return (
            index_dir / f"{safe_id}.vector.npy",
            index_dir / f"{safe_id}.vector.meta.json",
        )

    def _paths(self, index_dir: Path) -> tuple[Path, Path]:
        return self._paths_for(self.patient_id, index_dir)

    def save(self, index_dir: str | Path = DEFAULT_INDEX_DIR) -> Path:
        """Persist this patient's built index to `index_dir` (default
        `data/index/`, gitignored — local data per decisions/001) so a
        rebuild is not needed every process start. Two files: the vectors
        array (`.npy`) and a JSON sidecar with `patient_id`, `dim`,
        `backend_name` (the encoder that produced these vectors — see
        `load()`), and each unit's `text`/`session_id`/`turn_ids`/`kind`."""
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        vectors_path, meta_path = self._paths(index_dir)
        np.save(vectors_path, self.vectors)
        meta = {
            "patient_id": self.patient_id,
            "dim": self.dim,
            "backend_name": self.backend_name,
            "build_time_s": self.build_time_s,
            "units": [
                {"text": u.text, "session_id": u.session_id, "turn_ids": u.turn_ids, "kind": u.kind}
                for u in self.units
            ],
        }
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        return vectors_path

    @classmethod
    def load(
        cls,
        patient_id: str,
        index_dir: str | Path = DEFAULT_INDEX_DIR,
        *,
        backend: "embedders.EmbedderBackend | str | None" = None,
        embed_fn: EmbedFn | None = None,
        cache_path: Path | None = _UNSET_CACHE_PATH,  # type: ignore[assignment]
    ) -> "PatientIndex":
        """Inverse of `save`. Raises `FileNotFoundError` (not a silent
        empty index) if no saved index exists for `patient_id` — a missing
        index and an empty patient are different facts and must not be
        conflated (mirrors `existence.py`'s "do not invent a result"
        discipline for a different failure mode). File existence is
        checked *before* any encoder is constructed, so a missing-index
        call never pays a model-load cost on its way to raising.

        Encoder resolution, in order:

        1. `embed_fn=` given -> use it verbatim (legacy path).
        2. `backend=` given -> resolve it via `embedders.get_backend`.
        3. Neither given -> use whatever `backend_name` the saved index's
           own metadata recorded, so a bare `load(patient_id)` call
           correctly reconstructs the *original* encoder rather than
           silently defaulting to today's global default (which may have
           changed since the index was built).
        4. Neither given *and* the saved metadata has no `backend_name`
           at all (an index saved before this story existed) -> raise
           `embedders.BackendMismatchError` rather than guessing an
           encoder for vectors of unknown provenance.

        Whichever encoder is resolved, its `.name` is then compared
        against the saved `backend_name` (when the latter is present) and
        a mismatch raises `embedders.BackendMismatchError` — this is the
        "a mismatched backend name on load raises rather than silently
        comparing incompatible vectors" guarantee. A caller that
        explicitly asks for the wrong encoder is stopped just as reliably
        as one that asks for nothing at all."""
        index_dir = Path(index_dir)
        vectors_path, meta_path = cls._paths_for(patient_id, index_dir)
        if not vectors_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"no saved vector index for patient_id={patient_id!r} under {index_dir}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        saved_backend_name = meta.get("backend_name")

        if embed_fn is not None:
            index = cls(patient_id, embed_fn=embed_fn, dim=meta.get("dim", DEFAULT_EMBED_DIM), cache_path=cache_path)
        elif backend is not None:
            index = cls(patient_id, backend=backend, cache_path=cache_path)
        elif saved_backend_name:
            index = cls(patient_id, backend=saved_backend_name, cache_path=cache_path)
        else:
            raise embedders.BackendMismatchError(
                f"saved index for patient_id={patient_id!r} at {meta_path} has no recorded "
                "backend_name (built before pluggable embedders existed) and no backend=/"
                "embed_fn= was given to load() to disambiguate — refusing to guess an encoder "
                "for vectors of unknown provenance"
            )

        if saved_backend_name and index.backend_name != saved_backend_name:
            raise embedders.BackendMismatchError(
                f"saved index for patient_id={patient_id!r} was built with backend "
                f"{saved_backend_name!r} but load() resolved backend {index.backend_name!r}; "
                "refusing to silently compare incompatible vectors"
            )

        index.dim = meta["dim"]
        index.build_time_s = meta.get("build_time_s", 0.0)
        index.units = [
            IndexedUnit(text=u["text"], session_id=u["session_id"], turn_ids=list(u["turn_ids"]), kind=u["kind"])
            for u in meta["units"]
        ]
        index.vectors = np.load(vectors_path)
        return index
