"""graph/embedders.py — pluggable local (and hosted) embedding backends for
`graph/vector_index.py`. This module owns the *encoder*; `vector_index.py`
still owns the brute-force cosine *index* unchanged (`PatientIndex.search`'s
`self.vectors @ q` math is not touched by this story at all).

## Why this module exists

`vector_index.PatientIndex` previously defaulted to a deterministic
signed-hashing-trick embedding (see that module's own `_hashing_embed`,
kept in place, unmodified, as the fallback for the module's *other*,
untouched caller — `graph/retrieve.py::seed_entity_ids`'s entity-name
similarity check, which is out of this story's file scope). A hashing
trick has zero semantic knowledge by construction; a real sentence-
embedding model measurably beats it on retrieval quality (this project's
own `literature/12-vector-search-and-hybrid-fusion.md` Q3: a general-
domain 335M model beat every medical-specific model tested on real EHR
retrieval, and beat a random-order baseline is what an 8.9B *clinical*
model scored *worse* than — general-purpose semantic embedding, not
domain-specialization, is the load-bearing choice). This module is the
pluggable, swappable layer that lets `PatientIndex` use a real model
by default while keeping the encoder swap a one-line change, never an
index rewrite.

## The GPU-tier local ablation candidates — verified reachable on HF, and
## each model's own card read directly (not from training-data memory)
## before writing the prompt-formatting code below

All three were smoke-tested in this session on the project's own RTX 3090
(see the dev-ml return note / `.claude/logs/dev.log.md` entry for this
story for the pasted real throughput/VRAM numbers) and all three report
`hidden_size=1024` in their own `config.json` — a deliberate, load-bearing
fact this module leans on: every GPU-tier local backend emits the *same*
output dimension, so an ablation *within that tier* is a clean swap (no
downstream dimensionality confound in `PatientIndex`) and
`SentenceTransformerEmbedder.dim` is read live from the loaded model
(`SentenceTransformer.get_embedding_dimension()`), never hardcoded, so a
future model-card change cannot silently drift out of sync with this
module's own claim. This same-dim property does **not** extend across
tiers — see "The CPU-tier local ablation candidates" below, which are all
384-dim by construction (a real, deliberate dimensionality drop, not a
bug): `PatientIndex` persists `backend_name` and refuses to `load()` an
index built by a different encoder regardless of whether the dimensions
happen to match (`embedders.BackendMismatchError`), so a dimension change
across an ablation run is never silently unsafe — see §"Why a persisted
`backend_name` guard, not a dimension check" below.

- **`qwen3-0.6b` -> `Qwen/Qwen3-Embedding-0.6B`** (default). Its own
  `config_sentence_transformers.json` (read directly from the locally
  cached snapshot in this session, not recalled from memory) ships
  `"prompts": {"query": "Instruct: Given a web search query, retrieve
  relevant passages that answer the query\\nQuery:", "document": ""}` and
  `"default_prompt_name": null` — i.e. the instruction prefix is
  opt-in per call (`prompt_name="query"`), applies *only* to queries, and
  documents get an explicit empty prefix (no instruction at all). The
  model's own README repeats this in prose: "No need to add instruction
  for retrieval documents" and quantifies the cost of skipping the query
  instruction ("approximately 1% to 5%" retrieval-quality drop). 32K
  (32768-token) context, dim up to 1024 with MRL support (this module
  uses the full 1024, not a truncated MRL slice — see "why no
  `truncate_dim`" note on `SentenceTransformerEmbedder`).
- **`bge-m3` -> `BAAI/bge-m3`**. Its `config_sentence_transformers.json`
  carries no `prompts` key at all (unlike the other two), and its own
  README states this in prose, unambiguously: "The only difference is
  that the BGE-M3 model no longer requires adding instructions to the
  queries." Symmetric: queries and documents are embedded with the exact
  same raw text, no prefix either side. (Sentence-transformers itself
  reports `model.prompts == {"query": "", "document": ""}` for this model
  at runtime — an internal default template, not a model-card-documented
  convention — so this module treats `bge-m3` as having no registered
  query prompt name at all, rather than relying on that empty-string
  default, which is the more explicit and self-documenting choice.)
  8192-token context (`config.json` `max_position_embeddings=8194`).
- **`arctic-l-v2` -> `Snowflake/snowflake-arctic-embed-l-v2.0`**. Its
  `config_sentence_transformers.json` ships `"prompts": {"query": "query:
  "}` (trailing space, no `document` entry at all) and its own README's
  worked Sentence-Transformers example is explicit: `query_embeddings =
  model.encode(queries, prompt_name="query")` immediately followed by
  `document_embeddings = model.encode(documents)` with **no** prompt
  argument on the document side. Same asymmetric shape as `qwen3-0.6b`,
  different literal prefix. 8192-token context.

## The CPU-tier local ablation candidates — added for the CPU-deployment
## ablation story (`literature/20-cpu-inference-and-deployment-optimization.md`
## §Q1/Q2: the reranker, not the query encoder, is this project's dominant
## CPU-latency risk, but the encoder's own re-measured 2-thread-CPU cost is
## still the number a t3.medium demo box has to pay on every query)

Five 22-33M-parameter, 384-dim models, ~18-27x smaller than `qwen3-0.6b`
(595.8M params) by this module's own live `sum(p.numel() for p in
model.parameters())` count. All five verified reachable on the Hub
(`huggingface_hub.model_info`, run directly in this session) before
registration; each model's own card was read directly out of the locally
cached HF snapshot (downloading it first where not already present) — not
recalled from training-data memory, same discipline as the GPU-tier three
above. Getting a model's query/document convention wrong is, per this
story's own brief, "the single most common way an embedder ablation
becomes meaningless" — two of these five needed a mechanism the GPU-tier
three never exercised (see "Two new prompt mechanisms" below).

- **`minilm-l6` -> `sentence-transformers/all-MiniLM-L6-v2`** — the
  canonical CPU baseline named by the story. No `config_sentence_
  transformers.json` `prompts` key at all, and the model's own README
  ("Usage (Sentence-Transformers)" section, read directly from the local
  cache) shows a plain `model.encode(sentences)` call with no instruction
  anywhere. **Symmetric, no prefix either side** — same shape as `bge-m3`.
  22.7M params, dim 384, 256-token context.
- **`bge-small` -> `BAAI/bge-small-en-v1.5`**. Its own
  `config_sentence_transformers.json` carries **no `prompts` key** (unlike
  `bge-m3`, which explicitly documents having none — this model's own
  README instead states the instruction in prose only, never registering
  it as a loadable `sentence-transformers` prompt template): "we suggest to
  add the instruction to the query" for retrieval tasks, quoting the exact
  string `"Represent this sentence for searching relevant passages: "`,
  and explicitly "In all cases, **no instruction** needs to be added to
  passages." This is a real, documented, **asymmetric** convention that
  the model's own on-disk config does not expose via `prompt_name=` —
  `query_prompt_text` (a new `_ModelSpec` field, see below) carries the
  literal string instead, passed to `SentenceTransformer.encode(...,
  prompt=...)` rather than `prompt_name=...`. 33.4M params, dim 384,
  512-token context.
- **`arctic-s` -> `Snowflake/snowflake-arctic-embed-s`**. Unlike
  `arctic-l-v2`, this smaller sibling's own `config_sentence_transformers.
  json` ships a *different* literal query instruction —
  `"prompts": {"query": "Represent this sentence for searching relevant
  passages: "}` (no `document` entry), confirmed by the README's own
  worked example: `model.encode(queries, prompt_name="query")` immediately
  followed by `model.encode(documents)` with no prompt argument. Same
  registered-`prompt_name` mechanism as `arctic-l-v2`/`qwen3-0.6b` (existing
  `query_prompt_name` field, no new mechanism needed) — **a different
  literal instruction string per Arctic model size is itself a finding**:
  do not assume one Arctic size's card generalizes to another's. 33.2M
  params, dim 384, 512-token context.
- **`gte-small` -> `thenlper/gte-small`**. No `config_sentence_
  transformers.json` exists for this model at all (predates that file's
  convention — a 2023-era model card, `AutoModel`/`average_pool`-based
  usage example, not `sentence-transformers`-first). Its own "Usage" section
  shows both the raw-transformers example and the `sentence-transformers`
  example (`model.encode(sentences)`) applying **no instruction to either
  side** — read directly from the cached README, not inferred from the
  sibling `gte-Qwen2` instruction-tuned family's different convention
  (confirmed by checking, not assumed by name-family analogy). **Symmetric,
  no prefix either side.** 33.4M params, dim 384, 512-token context.
- **`e5-small` -> `intfloat/e5-small-v2`**. No `config_sentence_
  transformers.json` exists (same 2023-era-card reason as `gte-small`). Its
  own README's FAQ section is explicit and is the first model in this
  registry with a genuinely new shape: **both sides get an instruction**,
  not just the query — "Use `"query: "` and `"passage: "` correspondingly
  for asymmetric tasks such as passage retrieval." This is not "asymmetric"
  in the `qwen3-0.6b`/`arctic-*` sense (instruction-on-query-only); it is
  "different-instruction-on-each-side." `_ModelSpec` needed a second new
  field, `document_prompt_text`, to express it (see below) — the first
  registry row in this module where `SentenceTransformerEmbedder._forward`
  passes a `prompt=` on the *document* encode path at all. 33.4M params,
  dim 384, 512-token context.

### Two new prompt mechanisms this tier required, beyond `query_prompt_name`

The GPU-tier three above are fully expressed by one field,
`_ModelSpec.query_prompt_name: str | None`, passed to `SentenceTransformer.
encode(..., prompt_name=...)` — a lookup into the *model's own* registered
`self.prompts` dict (populated from `config_sentence_transformers.json`).
That mechanism silently cannot express two real conventions this tier's
own model cards document:

1. **A query instruction the card documents in prose but the model's own
   `config_sentence_transformers.json` never registers** (`bge-small`).
   `prompt_name="query"` would raise `KeyError` against this model's own
   (empty) `self.prompts` dict — sentence-transformers still exposes a
   literal-text escape hatch for exactly this case, `encode(...,
   prompt=<raw string>)`, which bypasses the model's own registry entirely.
   New field: `_ModelSpec.query_prompt_text: str | None`.
2. **An instruction on the *document* side, not just the query side**
   (`e5-small`). Every model registered before this story treats "document"
   as either "no instruction" or "the empty-string default" — never a real
   prefix. New field: `_ModelSpec.document_prompt_text: str | None`.

Both new fields default to `None` and are mutually exclusive with
`query_prompt_name` in every registry row (never two query mechanisms set
on the same spec) — `SentenceTransformerEmbedder._forward` below checks
`query_prompt_name` first, falls back to `query_prompt_text` for the query
side, and independently applies `document_prompt_text` on the document
side regardless of which query mechanism (if any) is in play. This is
purely additive: every existing registry row (`qwen3-0.6b`, `bge-m3`,
`arctic-l-v2`) leaves both new fields at their `None` default and is
byte-for-byte unaffected.

### Why a persisted `backend_name` guard, not a dimension check

`vector_index.PatientIndex.load()` (unchanged by this story — see that
module's own docstring) compares the *name* of the resolved backend
against the saved index's own `backend_name` metadata, never the
dimension. This matters concretely now that this registry spans two
dimensions (1024 for the GPU tier, 384 for the CPU tier): two different
384-dim models (say `minilm-l6` and `gte-small`) would pass a dimension-
only check while producing semantically incompatible vector spaces — a
`load()` that only checked `dim` would silently let that through.
`tests/test_embedders.py::TestPatientIndexBackendWiring::
test_load_with_mismatched_backend_raises_across_dimension_tiers` proves
this holds across a dimension change (384-dim `minilm-l6` saved, `qwen3-
0.6b` 1024-dim requested at `load()`), not just within one.

**OpenAI (`openai`, kept for comparison, never the default)**: OpenAI's
own embeddings guide (https://platform.openai.com/docs/guides/embeddings,
fetched and read directly in this session, 2026-08-16) documents no
query/document prefix convention anywhere — every example embeds queries
and documents with plain, unmodified text. `OpenAIEmbedder.encode`
therefore applies no per-role formatting; `is_query` is accepted for
`EmbedderBackend` protocol conformance but is a documented no-op on this
backend.

## The cache: `(model_name, role, content_hash)`, not just `(model_name,
## content_hash)`

The story's own spec names the cache key as `(model_name, content_hash)`.
This module deliberately extends that to a 3-tuple by folding in `role`
(`"query"` | `"document"`): two of the three local backends apply a
*different* prefix to the same raw string depending on which role it is
embedded as (`qwen3-0.6b`, `arctic-l-v2` above). A 2-tuple key would let a
query string that happens to collide, byte-for-byte, with an indexed
turn/fact string return the *other* role's cached vector — silently wrong,
not merely stale. This is a stated, reasoned deviation from the story's
literal 2-tuple, not an unstated one; see `_DiskEmbedCache`'s own
docstring for the mechanics.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "EmbedderBackend",
    "SentenceTransformerEmbedder",
    "OpenAIEmbedder",
    "EMBEDDER_REGISTRY",
    "DEFAULT_BACKEND_NAME",
    "DEFAULT_ST_CACHE_PATH",
    "get_backend",
    "list_backends",
    "UnknownBackendError",
    "BackendMismatchError",
]


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class EmbedderBackend(Protocol):
    """The frozen shape every backend in this module satisfies, and the
    shape `vector_index.PatientIndex` codes against instead of a concrete
    class — swapping `qwen3-0.6b` for `bge-m3` for `openai` is a
    `backend="..."` string change, never an import change.

    `encode()` takes an extra `is_query` keyword beyond the story's
    literal `encode(texts, batch_size) -> np.ndarray` sketch — required to
    thread the query/document formatting difference documented in the
    module docstring through to the model call; every call site in this
    project (`vector_index.PatientIndex.build`/`.search`) passes it
    explicitly rather than relying on the default, so which role is being
    encoded is always an explicit fact at the call site, not an implicit
    one."""

    name: str
    dim: int
    max_seq: int

    def encode(
        self, texts: list[str], *, batch_size: int = 64, is_query: bool = False
    ) -> np.ndarray: ...


class UnknownBackendError(ValueError):
    """Raised by `get_backend` for a name not in `EMBEDDER_REGISTRY` (and
    not `"openai"`) — never silently falls back to a different backend."""


class BackendMismatchError(ValueError):
    """Raised by `vector_index.PatientIndex.load` when the backend
    resolved for a `load()` call does not match the backend name recorded
    in the saved index's own metadata (or no backend was recorded at all
    and none was supplied to disambiguate) — see that module's own
    docstring on `load()`."""


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Defense-in-depth re-normalization applied after every backend's own
    forward pass, regardless of whether the underlying library already
    normalized (sentence-transformers' `normalize_embeddings=True` does,
    but fp16 rounding can drift a norm a few ULPs off 1.0, and a future
    backend added to this module might not normalize at all) — the
    `EmbedderBackend.encode` contract is L2-normalized output, enforced
    here once rather than trusted per-backend. Mirrors
    `vector_index._hashing_embed`'s own explicit norm-then-guard pattern
    (zero-vector rows guarded against division by zero) for exactly the
    same reason."""
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.size == 0:
        return vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vectors / norms).astype(np.float32)


# ---------------------------------------------------------------------------
# Registry — one row per local ablation candidate. `hf_id` and
# `query_prompt_name` are the two facts this module needed to verify per
# model (see module docstring for what was found and how); `dim`/`max_seq`
# are deliberately *not* stored here — they are read live off the loaded
# model (`SentenceTransformerEmbedder.__init__`) so this table cannot drift
# out of sync with the model itself.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ModelSpec:
    hf_id: str
    query_prompt_name: str | None = None
    """Passed to `SentenceTransformer.encode(..., prompt_name=...)` when
    encoding queries — a lookup into the *model's own* registered
    `self.prompts` dict (populated from `config_sentence_transformers.
    json`). `None` means either this model's own card documents no query
    instruction convention at all (`bge-m3`, `minilm-l6`, `gte-small`), or
    it documents one that its own config does not register, in which case
    `query_prompt_text` below is used instead (never both on the same
    spec) — `prompt_name` is then simply omitted from the `encode()` call,
    not passed as an empty string, which is the more explicit choice (see
    module docstring)."""
    query_prompt_text: str | None = None
    """Raw literal prefix text, passed to `SentenceTransformer.encode(...,
    prompt=...)` — bypasses the model's own `.prompts` registry entirely.
    Used when a model's *card* documents a query instruction that its own
    `config_sentence_transformers.json` does not register as a loadable
    `prompt_name` (`bge-small`), or that predates that file's existence
    altogether. Mutually exclusive with `query_prompt_name` in every
    registry row below — never both set for the same model (see module
    docstring, "Two new prompt mechanisms")."""
    document_prompt_text: str | None = None
    """Raw literal prefix text applied to the *document* side, via the same
    `encode(..., prompt=...)` mechanism as `query_prompt_text`. Every model
    registered before this story's CPU-tier addition needs no document-side
    prefix (`None`); `e5-small` is the first exception (its own README:
    'Use "query: " and "passage: " correspondingly')."""
    card_note: str = ""
    """One-line summary of what this model's own card says about
    query/document formatting — surfaced by `list_backends(verbose=True)`
    and quoted in the README algorithm section so a reviewer can check
    the claim against the cited source without re-deriving it."""


EMBEDDER_REGISTRY: dict[str, _ModelSpec] = {
    "qwen3-0.6b": _ModelSpec(
        hf_id="Qwen/Qwen3-Embedding-0.6B",
        query_prompt_name="query",
        card_note=(
            "config_sentence_transformers.json: query prompt "
            "'Instruct: Given a web search query, retrieve relevant "
            "passages that answer the query\\nQuery:', document prompt "
            "'' (empty). Asymmetric: instruction on queries only."
        ),
    ),
    "bge-m3": _ModelSpec(
        hf_id="BAAI/bge-m3",
        query_prompt_name=None,
        card_note=(
            "README: 'The only difference is that the BGE-M3 model no "
            "longer requires adding instructions to the queries.' "
            "Symmetric: no prefix either side."
        ),
    ),
    "arctic-l-v2": _ModelSpec(
        hf_id="Snowflake/snowflake-arctic-embed-l-v2.0",
        query_prompt_name="query",
        card_note=(
            "config_sentence_transformers.json: query prompt 'query: ', "
            "no document prompt entry; README's own ST example calls "
            "`model.encode(documents)` with no prompt argument. "
            "Asymmetric: instruction on queries only."
        ),
    ),
    # -- CPU-tier (22-33M params, 384-dim) — see module docstring "The
    # CPU-tier local ablation candidates" for the full per-model writeup.
    "minilm-l6": _ModelSpec(
        hf_id="sentence-transformers/all-MiniLM-L6-v2",
        card_note=(
            "No config_sentence_transformers.json prompts key; README's "
            "own Sentence-Transformers usage example: plain "
            "`model.encode(sentences)`, no instruction either side. "
            "Symmetric: no prefix either side."
        ),
    ),
    "bge-small": _ModelSpec(
        hf_id="BAAI/bge-small-en-v1.5",
        query_prompt_text="Represent this sentence for searching relevant passages: ",
        card_note=(
            "config_sentence_transformers.json has NO prompts key (unlike "
            "bge-m3, which explicitly has none by design) -- the "
            "instruction lives only in README prose: 'we suggest to add "
            "the instruction to the query' quoting 'Represent this "
            "sentence for searching relevant passages: '; 'In all cases, "
            "no instruction needs to be added to passages.' Asymmetric, "
            "applied via encode(prompt=...) not prompt_name= (the model's "
            "own config never registers it as a loadable prompt name)."
        ),
    ),
    "arctic-s": _ModelSpec(
        hf_id="Snowflake/snowflake-arctic-embed-s",
        query_prompt_name="query",
        card_note=(
            "config_sentence_transformers.json: query prompt 'Represent "
            "this sentence for searching relevant passages: ' (DIFFERENT "
            "literal text from arctic-l-v2's 'query: ' -- do not assume "
            "one Arctic model size's card generalizes to another's), no "
            "document prompt entry; README's own ST example confirms. "
            "Asymmetric: instruction on queries only."
        ),
    ),
    "gte-small": _ModelSpec(
        hf_id="thenlper/gte-small",
        card_note=(
            "No config_sentence_transformers.json exists (pre-dates that "
            "file's convention). README's own Sentence-Transformers usage "
            "example: plain `model.encode(sentences)`, no instruction "
            "either side (confirmed directly, not inferred from the "
            "sibling instruction-tuned gte-Qwen2 family's different "
            "convention). Symmetric: no prefix either side."
        ),
    ),
    "e5-small": _ModelSpec(
        hf_id="intfloat/e5-small-v2",
        query_prompt_text="query: ",
        document_prompt_text="passage: ",
        card_note=(
            "No config_sentence_transformers.json exists. README's own "
            "FAQ, verbatim: 'Use \"query: \" and \"passage: \" "
            "correspondingly for asymmetric tasks such as passage "
            "retrieval.' The first model in this registry where BOTH "
            "sides get an instruction, not just the query."
        ),
    ),
}

DEFAULT_BACKEND_NAME = "qwen3-0.6b"
DEFAULT_ST_CACHE_PATH = Path("data/index/st_embed_cache.npz")
"""Deliberately a *different* file from `vector_index.DEFAULT_INDEX_DIR /
'embed_cache.npz'` (the legacy hashing-trick cache used by that module's
own `embed_fn=` escape hatch) — the two caches have incompatible on-disk
schemas (compound-key vs hash-only, different dims), and reusing one path
for both would mean whichever cache saved last silently clobbers the
other's file. See `vector_index.PatientIndex.__init__`'s own comment on
the sentinel-based default-path split."""


def list_backends() -> list[str]:
    """Every backend name `get_backend` accepts: the three GPU-tier
    candidates, the five CPU-tier candidates, plus `"openai"`. Used by
    `tests/test_embedders.py` and available to a future CLI ablation
    sweep."""
    return [*EMBEDDER_REGISTRY.keys(), "openai"]


# ---------------------------------------------------------------------------
# Disk cache — (model_name, role, content_hash) -> vector
# ---------------------------------------------------------------------------


def _content_hash(text: str) -> str:
    """SHA-256 of the exact text being embedded — same determinism-
    across-restarts reasoning as `vector_index._content_hash` (not
    reused directly: that function lives in a sibling module this story
    does not import from, to keep the two caches' failure modes
    independent — a bug in one cache's (de)serialization cannot corrupt
    the other's)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


_KEY_SEP = "\x1f"
"""ASCII unit separator — never appears in a model name, role, or hex
digest, so splitting on it to recover the 3-tuple is unambiguous without
any escaping logic."""


class _DiskEmbedCache:
    """One shared `.npz` file, `(model_name, role, content_hash) ->
    float32 vector`. See module docstring for why `role` is folded into
    the key beyond the story's literal `(model_name, content_hash)` pair.

    Read-tolerant: a cache file that fails to parse (truncated write,
    foreign schema, hand-edited) is treated as a cold cache — logged and
    skipped, never a crash. A cache exists purely for latency/cost; losing
    it can never be a correctness bug, only a performance one."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._map: dict[tuple[str, str, str], np.ndarray] = {}
        self._dirty = False
        if self.path.exists():
            try:
                with np.load(self.path, allow_pickle=False) as data:
                    packed_keys = data["keys"]
                    vectors = data["vectors"]
                    for i, packed in enumerate(packed_keys.tolist()):
                        model_name, role, content_hash = packed.split(_KEY_SEP)
                        self._map[(model_name, role, content_hash)] = vectors[i]
            except Exception:  # noqa: BLE001 - a bad cache file must never be fatal
                logger.warning("embedders: could not read cache at %s; starting cold", self.path)
                self._map = {}

    def get_many(
        self, model_name: str, role: str, texts: list[str]
    ) -> tuple[dict[int, np.ndarray], list[int]]:
        hits: dict[int, np.ndarray] = {}
        misses: list[int] = []
        for i, text in enumerate(texts):
            vec = self._map.get((model_name, role, _content_hash(text)))
            if vec is None:
                misses.append(i)
            else:
                hits[i] = vec
        return hits, misses

    def put_many(self, model_name: str, role: str, texts: list[str], vectors: np.ndarray) -> None:
        for text, vec in zip(texts, vectors):
            self._map[(model_name, role, _content_hash(text))] = np.asarray(vec, dtype=np.float32)
        self._dirty = True

    def save(self) -> None:
        if not self._dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self._map:
            packed_keys = np.array(
                [f"{m}{_KEY_SEP}{r}{_KEY_SEP}{h}" for (m, r, h) in self._map.keys()]
            )
            vectors = np.stack(list(self._map.values()))
        else:
            packed_keys = np.array([], dtype=str)
            vectors = np.zeros((0, 0), dtype=np.float32)
        np.savez_compressed(self.path, keys=packed_keys, vectors=vectors)
        self._dirty = False


# ---------------------------------------------------------------------------
# SentenceTransformerEmbedder
# ---------------------------------------------------------------------------

# Model-weights-level cache: (hf_id, device) -> loaded `SentenceTransformer`.
# Deliberately separate from the *disk embedding cache* above — the two are
# orthogonal costs (GPU-resident weights vs. a per-text vector on disk) and
# conflating them would mean two `SentenceTransformerEmbedder` instances for
# the same model but different `cache_path`s (e.g. two tests, each with its
# own `tmp_path`) would either reload the ~0.6-0.6B-parameter model twice
# (slow) or silently share one instance's cache_path against the other's
# wishes (surprising). Keying the *model* cache by `(hf_id, device)` only,
# and re-deriving a fresh lightweight wrapper (dim/max_seq/prompt name/cache
# object) on every `SentenceTransformerEmbedder.__init__` call, gets both
# properties at once: one real GPU load per (model, device) per process, and
# every caller's own `cache_path` honored exactly as given.
_MODEL_CACHE: dict[tuple[str, str], object] = {}
_MODEL_CACHE_LOCK = threading.Lock()


def _load_sentence_transformer(hf_id: str, device: str):
    """Returns a process-wide-shared `SentenceTransformer` for `(hf_id,
    device)`, loading it (fp16 `model_kwargs` on CUDA, plain fp32 on CPU —
    the story's own "fp16 on CUDA with CPU fallback" instruction) on first
    request only. `torch`/`sentence_transformers` are imported here, not
    at module level, so importing `graph.embedders` (e.g. transitively via
    `graph.vector_index`) never pays the multi-second torch import cost
    unless a `SentenceTransformerEmbedder` is actually constructed."""
    cache_key = (hf_id, device)
    with _MODEL_CACHE_LOCK:
        cached = _MODEL_CACHE.get(cache_key)
        if cached is not None:
            return cached
        import torch
        from sentence_transformers import SentenceTransformer

        model_kwargs = {"torch_dtype": torch.float16} if device == "cuda" else {}
        model = SentenceTransformer(hf_id, device=device, model_kwargs=model_kwargs)
        _MODEL_CACHE[cache_key] = model
        return model


class SentenceTransformerEmbedder:
    """Wraps one `sentence-transformers` model as an `EmbedderBackend`.
    `name` is the short registry key (`"qwen3-0.6b"` etc.), not the raw HF
    id, so `vector_index.PatientIndex`'s persisted `backend_name` (and the
    load-time mismatch check) reads cleanly."""

    def __init__(
        self,
        model_key: str,
        *,
        device: str | None = None,
        cache_path: Path | None = DEFAULT_ST_CACHE_PATH,
        batch_size: int = 64,
    ) -> None:
        spec = EMBEDDER_REGISTRY.get(model_key)
        if spec is None:
            raise UnknownBackendError(
                f"unknown local embedder backend {model_key!r}; registered: "
                f"{sorted(EMBEDDER_REGISTRY)}"
            )
        import torch

        self.name = model_key
        self._hf_id = spec.hf_id
        self._query_prompt_name = spec.query_prompt_name
        self._query_prompt_text = spec.query_prompt_text
        self._document_prompt_text = spec.document_prompt_text
        self._batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = _load_sentence_transformer(spec.hf_id, self.device)
        # `get_embedding_dimension`/`get_max_seq_length`, not the deprecated
        # `get_sentence_embedding_dimension` alias (confirmed live in this
        # session: the deprecated name raises a `FutureWarning` on this
        # installed sentence-transformers version).
        self.dim: int = int(self._model.get_embedding_dimension())
        self.max_seq: int = int(self._model.get_max_seq_length())
        self._cache = _DiskEmbedCache(cache_path) if cache_path is not None else None

    def encode(
        self, texts: list[str], *, batch_size: int | None = None, is_query: bool = False
    ) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        role = "query" if is_query else "document"
        bs = batch_size or self._batch_size
        if self._cache is None:
            return self._forward(texts, batch_size=bs, is_query=is_query)

        hits, miss_idx = self._cache.get_many(self.name, role, texts)
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, vec in hits.items():
            vectors[i] = vec
        if miss_idx:
            miss_texts = [texts[i] for i in miss_idx]
            miss_vectors = self._forward(miss_texts, batch_size=bs, is_query=is_query)
            for j, i in enumerate(miss_idx):
                vectors[i] = miss_vectors[j]
            self._cache.put_many(self.name, role, miss_texts, miss_vectors)
            self._cache.save()
        return vectors

    def _forward(self, texts: list[str], *, batch_size: int, is_query: bool) -> np.ndarray:
        """The one place this class calls the model. `truncate_dim` is
        deliberately never passed even though `qwen3-0.6b` supports MRL
        (Matryoshka) truncation down to 32 dims — the module docstring's
        "same dim across all three GPU-tier backends, no downstream
        dimensionality confound *within that tier*" property depends on
        always using the full 1024-dim output.

        Query-side mechanism, checked in order (mutually exclusive per
        registry row — see `_ModelSpec` docstring): a registered
        `prompt_name` (looked up in the model's own `self.prompts` dict)
        takes priority over a raw literal `query_prompt_text` (bypasses
        that dict entirely) if a spec somehow set both, though no row in
        `EMBEDDER_REGISTRY` currently does. Document-side: `prompt=` is
        applied independently of whichever query mechanism (if any) is in
        play — `e5-small` is the only registry row that sets
        `document_prompt_text` today (module docstring, "Two new prompt
        mechanisms")."""
        kwargs: dict = dict(
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        if is_query:
            if self._query_prompt_name:
                kwargs["prompt_name"] = self._query_prompt_name
            elif self._query_prompt_text:
                kwargs["prompt"] = self._query_prompt_text
        elif self._document_prompt_text:
            kwargs["prompt"] = self._document_prompt_text
        vectors = self._model.encode(list(texts), **kwargs)
        return _l2_normalize(np.asarray(vectors, dtype=np.float32))


# ---------------------------------------------------------------------------
# OpenAIEmbedder — kept for comparison, never the default
# ---------------------------------------------------------------------------


class OpenAIEmbedder:
    """Wraps `medmemgraph.llm.embed` (the project's single LLM/embedding
    seam — disk cache, budget cap, Ledger accounting all already live
    there) as an `EmbedderBackend`, so a hosted-model comparison run is
    `backend="openai"`, not a second HTTP client. Never the default
    (`vector_index.PatientIndex`'s own default is `qwen3-0.6b`) — kept
    reachable behind the identical protocol purely for an ablation/
    comparison run, per this story's own instruction."""

    _MAX_SEQ_TOKENS = {
        "text-embedding-3-small": 8191,
        "text-embedding-3-large": 8191,
    }
    """OpenAI's documented per-request token ceiling for the embeddings
    endpoint (https://platform.openai.com/docs/guides/embeddings, fetched
    and read directly in this session, 2026-08-16); 8191 is the fallback
    for any other embedding model name."""

    def __init__(self, model: str | None = None, *, dry_run: bool = False, use_cache: bool = True) -> None:
        from medmemgraph import llm

        self._llm = llm
        self.model = model or llm.EMBED_MODEL
        self.name = f"openai:{self.model}"
        # `llm._EMBED_DIMENSIONS` is a private constant of a sibling,
        # already-existing module — reused deliberately rather than
        # duplicated (this project's established "one seam" convention,
        # `llm.py`'s own module docstring), not a public-interface change.
        self.dim: int = llm._EMBED_DIMENSIONS.get(self.model, llm._DEFAULT_DRY_RUN_EMBED_DIM)
        self.max_seq: int = self._MAX_SEQ_TOKENS.get(self.model, 8191)
        self._dry_run = dry_run
        self._use_cache = use_cache

    def encode(
        self, texts: list[str], *, batch_size: int | None = None, is_query: bool = False
    ) -> np.ndarray:
        # `batch_size`/`is_query` accepted for `EmbedderBackend` protocol
        # conformance; neither changes behavior here — OpenAI's own
        # embeddings guide documents no query/document prefix convention
        # (module docstring), and `llm.embed` does its own internal
        # batching/caching/budget accounting per call, not per chunk.
        del batch_size, is_query
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = self._llm.embed(
            list(texts), model=self.model, dry_run=self._dry_run, use_cache=self._use_cache
        )
        return _l2_normalize(np.asarray(vectors, dtype=np.float32))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_backend(
    backend: "EmbedderBackend | str",
    *,
    cache_path: Path | None = DEFAULT_ST_CACHE_PATH,
    device: str | None = None,
    batch_size: int = 64,
) -> "EmbedderBackend":
    """Resolves a backend name (`"qwen3-0.6b"` | `"bge-m3"` |
    `"arctic-l-v2"` | `"openai"`) — or passes through an object that
    already satisfies `EmbedderBackend` unchanged, so a caller (or a test)
    can hand in a hand-built fake without going through this registry at
    all. Raises `UnknownBackendError` for any other string rather than
    guessing — never a silent fallback to the default model."""
    if not isinstance(backend, str):
        return backend
    if backend == "openai":
        return OpenAIEmbedder()
    if backend in EMBEDDER_REGISTRY:
        return SentenceTransformerEmbedder(
            backend, device=device, cache_path=cache_path, batch_size=batch_size
        )
    raise UnknownBackendError(
        f"unknown embedder backend {backend!r}; registered: {list_backends()}"
    )
