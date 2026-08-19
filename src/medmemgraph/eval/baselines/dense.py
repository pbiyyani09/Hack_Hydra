"""eval/baselines/dense.py — Rung 3 of the baseline ladder: plain chunked
dense (vector) retrieval over one patient's turn history.

`literature/02-memory-benchmarks-and-evaluation.md` §"Baseline ladder"
names this rung explicitly: "fixed-size chunked RAG at multiple chunk sizes
and k" (Mem0's own baseline sweep: 128/256/512/1024/2048/4096/8192-token
chunks x k in {1,2}, R-MEM-058) and "dense embedding retrieval" (used in
MedLoCoMo and BEAM's own RAG baseline, R-MEM-049/R-MEM-024). This module
implements that rung **independently and plainly** — fixed chunk size,
top-k, one embedding function, no rerank, no router, no fusion (story
requirement) — so it stays a real baseline and not "our system wearing a
disguise": it does not import the graph, the router, or the fusion layer,
only the corpus loader and the Chain-of-Note reader every other system in
the results table is also read by (`eval/reader.py`).

Runs entirely without HydraDB — same convention as `nomem.py`/`fullctx.py`.

--------------------------------------------------------------------------
embed() — why a hashing-trick fallback, not sentence-transformers
--------------------------------------------------------------------------
`collaborative/design/stories/E4/E4-S2.md` is the one place this project
already decided the `embed(texts) -> np.ndarray` contract (Graph-owned,
`src/medmemgraph/embeddings.py`): "shape (len(texts), dim), L2-normalized,
dtype float32. Default implementation: deterministic hashing-trick (no
network, no FAISS)... Banned approaches: Adding FAISS / HNSW /
sentence-transformers as a required dep." That module has not landed yet
(verified by grep — no `embeddings.py`, no `def embed` anywhere in this
tree as of this story). Per this project's established "do not block on
another track — code against the frozen contract and import the real
implementation behind a try/except" convention (already used by
`eval/reader.py::_default_retriever` for `retrieve()`), this file
implements the *same* default-fallback contract E4-S2 already specifies —
not a competing implementation — and imports the real
`medmemgraph.embeddings.embed` first if it exists, deleting this fork once
it lands.

The fallback is the signed hashing trick (Weinberger, Dasgupal, Langford,
Smola, Attenberg — "Feature Hashing for Large Scale Multitask Learning",
ICML 2009, arXiv:0902.2206): each token is hashed via `hashlib.sha1`
(stable across process restarts — Python's builtin `hash()` is salted per
run unless `PYTHONHASHSEED` is fixed, `hashlib` is not) into a bucket
index *and* a sign bit, so hash collisions partially cancel instead of
purely accumulating (§3 of that paper). This is why `--no-api-key` runs of
this baseline are still deterministic (same requirement `judge.py`/
`reader.py`'s stub paths already satisfy) and why `ARCHITECTURE.md`'s own
"Dense: one in-memory NumPy array of L2-normalized embeddings per
patient_id. Brute-force cosine." bullet is satisfied literally, without
committing this story to a heavyweight new dependency the story's file
list does not include (adding one — torch/sentence-transformers/an
API-keyed hosted embedder not already wired in this repo — would itself be
a new-dependency HALT per this agent's contract; E4-S2 already reasoned
through exactly this trade-off and banned it).

--------------------------------------------------------------------------
Chunking
--------------------------------------------------------------------------
Fixed **token-size**, non-overlapping, **session-bounded** chunks: chunk
boundaries never cross an admission boundary (a chunk's `RetrieveItem.
session_id` must name exactly one admission for the Recall@k/NDCG@k gold
comparison in `literature/02`'s ladder — and for E7-S2's own admission-hit
contract — to mean anything). Two chunk sizes are swept (`CHUNK_SIZE_SWEEP`)
per Mem0's own token-size grid (R-MEM-058 above); two top-k values are
swept (`K_SWEEP`), matching the Recall@5/@10 convention this project's
retrieval metrics use elsewhere.

--------------------------------------------------------------------------
Read by the same reader as everyone else
--------------------------------------------------------------------------
`DenseRAGAnswerer` subclasses `eval.reader.ReaderAnswerer` and supplies only
a different `retriever` callable (`DenseRetriever`) — the Chain-of-Note
reading strategy, prompt, schema, and token/latency accounting are the
exact same code path `reader_con`/`reader_direct` and (in `lexical.py`) the
BM25 baseline all go through. This is deliberate, per the story: "every
system in the table must be read by the same reader under the same mode,
or the comparison is confounded" — the *only* thing that varies across
`dense`/`lexical`/`reader_con` is which evidence gets retrieved.
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, replace
from typing import Callable

import numpy as np

from medmemgraph.contracts import RetrieveItem, RetrieveResult
from medmemgraph.eval.reader import DEFAULT_READER_MODEL, ReaderAnswerer
from medmemgraph.eval.types import AnswerResult, estimate_tokens
from medmemgraph.pipeline.loader import Admission, Conversation, load_conversation

__all__ = [
    "DEFAULT_EMBED_DIM",
    "DEFAULT_CHUNK_SIZE_TOKENS",
    "DEFAULT_K",
    "CHUNK_SIZE_SWEEP",
    "K_SWEEP",
    "embed",
    "Chunk",
    "chunk_admission",
    "chunk_conversation",
    "DensePatientIndex",
    "DenseRetriever",
    "DenseRAGAnswerer",
    "dense_recall_sweep",
]

# ---------------------------------------------------------------------------
# embed() — see module docstring.
# ---------------------------------------------------------------------------

DEFAULT_EMBED_DIM = 512
"""Hashing-trick output width. Not the real embedding model's dim (that
model does not exist in this repo yet) — a cheap, fast, collision-tolerable
width for per-patient-scale corpora (hundreds, not millions, of chunks;
ARCHITECTURE.md §"Dense" measured brute-force cosine cost at this scale
directly, R-VEC-009/010 in literature/12)."""

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# A short, hand-picked stopword list — not corpus-derived IDF (that would
# make embed() batch-relative and break "the same string embeds to the same
# vector no matter what else is in the call", E4-S2 AC5) — just enough to
# stop "the"/"a"/"is" from dominating every chunk's hashed bag-of-words
# before L2 normalization.
_STOPWORDS = frozenset(
    {
        "the", "a", "an", "and", "or", "is", "are", "was", "were", "be", "been",
        "to", "of", "in", "on", "at", "for", "with", "it", "this", "that", "i",
        "you", "he", "she", "we", "they", "do", "does", "did", "have", "has",
        "had", "not", "no", "yes", "as", "by", "so", "if", "any", "there",
    }
)


def _tokenize_for_embed(text: str) -> list[str]:
    return [tok for tok in _TOKEN_RE.findall(text.lower()) if tok not in _STOPWORDS]


def _hashing_embed(texts: list[str], dim: int = DEFAULT_EMBED_DIM) -> np.ndarray:
    """Deterministic signed-hashing-trick embedding — see module docstring
    (Weinberger et al. 2009, arXiv:0902.2206, §3: signed hashing so
    collisions partially cancel rather than purely accumulate). Row i is
    `texts[i]`'s embedding; L2-normalized; zero rows (no tokens survived
    stopword filtering) are left as the zero vector rather than divided by
    zero."""
    vectors = np.zeros((len(texts), dim), dtype=np.float32)
    for row, text in enumerate(texts):
        for tok in _tokenize_for_embed(text):
            digest = hashlib.sha1(tok.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % dim
            sign = 1.0 if (digest[4] & 1) else -1.0
            vectors[row, bucket] += sign
    # log1p-dampen repeated-token dominance (sublinear TF scaling, the same
    # idea BM25/TF-IDF use) before normalizing, sign-preserving.
    vectors = np.sign(vectors) * np.log1p(np.abs(vectors))
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vectors / norms).astype(np.float32)


def embed(texts: list[str]) -> np.ndarray:
    """`embed(texts) -> np.ndarray`, shape `(len(texts), dim)`, L2-normalized
    float32 — E4-S2's frozen contract. Prefers the real Graph-owned
    `medmemgraph.embeddings.embed` if it has landed (same
    try-the-real-thing-first / fall back pattern as `eval/reader.py`'s
    `_default_retriever`); otherwise uses the deterministic hashing-trick
    fallback above, which is *the same default* E4-S2 itself specifies, not
    a parallel invention. Delete this fork once `medmemgraph.embeddings`
    exists (the story's own "delete the fork when it lands" convention)."""
    if not texts:
        return np.zeros((0, DEFAULT_EMBED_DIM), dtype=np.float32)
    try:
        from medmemgraph.embeddings import embed as _real_embed

        return _real_embed(texts)
    except ImportError:
        return _hashing_embed(texts, dim=DEFAULT_EMBED_DIM)


# ---------------------------------------------------------------------------
# Chunking — fixed token size, session-bounded, non-overlapping.
# ---------------------------------------------------------------------------

CHUNK_SIZE_SWEEP = (256, 1024)
"""Two chunk sizes swept (story requirement), drawn from Mem0's own
chunked-RAG baseline grid (128/256/512/1024/2048/4096/8192 tokens,
`literature/02` R-MEM-058) — a "small" and a "large" granularity from that
grid, not two arbitrary numbers."""

K_SWEEP = (5, 10)
"""Two top-k values swept (story requirement); 5 and 10 match this
project's Recall@5/@10 convention (E7-S2/E7-S3) rather than Mem0's own
k in {1,2}, since this project's retrieval-quality reporting is built
around @5/@10."""

DEFAULT_CHUNK_SIZE_TOKENS = CHUNK_SIZE_SWEEP[0]
DEFAULT_K = K_SWEEP[0]


def _format_turn(turn) -> str:
    return f"[admission {turn.hadm_id} turn {turn.turn_number} · {turn.time} · {turn.speaker}] {turn.text}"


@dataclass(frozen=True)
class Chunk:
    """One fixed-token-size window of consecutive turns, all from a single
    admission (never split across a session boundary)."""

    session_id: str
    turn_ids: list[int]
    text: str


def chunk_admission(admission: Admission, chunk_size_tokens: int) -> list[Chunk]:
    """Walk `admission`'s turns in order, accumulating lines into the
    current chunk until adding the next line would exceed
    `chunk_size_tokens` (estimated via `eval.types.estimate_tokens`, the
    same tiktoken-with-fallback estimator `fullctx.py` uses), then start a
    new chunk. A single turn longer than the budget still becomes its own
    (over-budget) chunk rather than being split mid-turn — chunking never
    fabricates a turn boundary that was not in the source data."""
    chunks: list[Chunk] = []
    current_lines: list[str] = []
    current_turn_ids: list[int] = []
    current_tokens = 0

    for turn in admission.turns():
        line = _format_turn(turn)
        line_tokens = estimate_tokens(line)
        if current_lines and current_tokens + line_tokens > chunk_size_tokens:
            chunks.append(
                Chunk(session_id=admission.hadm_id, turn_ids=list(current_turn_ids), text="\n".join(current_lines))
            )
            current_lines, current_turn_ids, current_tokens = [], [], 0
        current_lines.append(line)
        current_turn_ids.append(turn.turn_number)
        current_tokens += line_tokens

    if current_lines:
        chunks.append(
            Chunk(session_id=admission.hadm_id, turn_ids=list(current_turn_ids), text="\n".join(current_lines))
        )
    return chunks


def chunk_conversation(conversation: Conversation, chunk_size_tokens: int) -> list[Chunk]:
    """Chunk every admission independently (session-bounded, see
    `chunk_admission`) and concatenate — admission order preserved, matching
    `Conversation.turns()`'s own ordering convention."""
    chunks: list[Chunk] = []
    for admission in conversation.admissions:
        chunks.extend(chunk_admission(admission, chunk_size_tokens))
    return chunks


# ---------------------------------------------------------------------------
# Index + retrieval — brute-force cosine, per ARCHITECTURE.md's "Dense" bullet
# ---------------------------------------------------------------------------


class DensePatientIndex:
    """One patient's chunk vectors, held in memory as a single L2-normalized
    NumPy array (`ARCHITECTURE.md` §"Dense": "one in-memory NumPy array of
    L2-normalized embeddings per patient_id. Brute-force cosine... No FAISS,
    no HNSW, no ANN" — measured 0.17-7.25 ms/query at this project's scale,
    literature/12 R-VEC-009/010). Built once per (patient, chunk_size);
    `DenseRetriever` below caches instances across a whole QA sweep so
    repeated queries do not re-chunk/re-embed the same history."""

    def __init__(self, conversation: Conversation, chunk_size_tokens: int) -> None:
        self.chunk_size_tokens = chunk_size_tokens
        self.chunks: list[Chunk] = chunk_conversation(conversation, chunk_size_tokens)
        texts = [c.text for c in self.chunks]
        self.vectors: np.ndarray = embed(texts) if texts else np.zeros((0, DEFAULT_EMBED_DIM), dtype=np.float32)

    def query(self, question: str, k: int) -> RetrieveResult:
        start = time.monotonic()
        if not self.chunks:
            latency_ms = (time.monotonic() - start) * 1000.0
            return RetrieveResult(
                items=[], route="vector", structural_absence=True, paths=[], latency_ms={"total": latency_ms}
            )

        q_vec = embed([question])[0]
        # Both sides are L2-normalized, so the dot product IS cosine
        # similarity — no separate division needed.
        scores = self.vectors @ q_vec
        k_eff = min(k, len(self.chunks))
        top_idx = np.argsort(-scores)[:k_eff]

        items = [
            RetrieveItem(
                text=self.chunks[i].text,
                session_id=self.chunks[i].session_id,
                turn_ids=list(self.chunks[i].turn_ids),
                score=float(scores[i]),
                channel="vector",
            )
            for i in top_idx
        ]
        latency_ms = (time.monotonic() - start) * 1000.0
        return RetrieveResult(
            items=items, route="vector", structural_absence=False, paths=[], latency_ms={"total": latency_ms}
        )


class DenseRetriever:
    """Callable `(question, patient_id, k) -> RetrieveResult` — the same
    shape as `contracts.mock_retrieve`/the real `retrieve()`, so it drops
    straight into `eval.reader.ReaderAnswerer(retriever=...)`. Caches one
    `DensePatientIndex` per `patient_id` (this baseline's `chunk_size_tokens`
    is fixed per retriever instance, so the cache key is just the patient)
    so a whole QA sweep chunks/embeds each patient's history exactly once,
    not once per question."""

    def __init__(
        self,
        *,
        chunk_size_tokens: int = DEFAULT_CHUNK_SIZE_TOKENS,
        root: object | None = None,
        conversation_loader: Callable[[str, object | None], Conversation] = load_conversation,
    ) -> None:
        self.chunk_size_tokens = chunk_size_tokens
        self.root = root
        self._conversation_loader = conversation_loader
        self._cache: dict[str, DensePatientIndex] = {}

    def _index_for(self, patient_id: str) -> DensePatientIndex:
        index = self._cache.get(patient_id)
        if index is None:
            conversation = self._conversation_loader(patient_id, self.root)
            index = DensePatientIndex(conversation, self.chunk_size_tokens)
            self._cache[patient_id] = index
        return index

    def __call__(self, question: str, patient_id: str, k: int) -> RetrieveResult:
        return self._index_for(patient_id).query(question, k)


class _TimedRetriever:
    """Wraps a retriever callable and records the wall-clock cost of its
    most recent call. Duplicated (not shared) between `dense.py` and
    `lexical.py` on purpose — the story's "implement them independently and
    plainly" instruction is about not sharing retrieval/router/fusion
    machinery between baselines; this eight-line timing shim is not
    algorithmic content, but keeping each baseline file fully self-contained
    (no cross-baseline import) avoids even the appearance of shared
    plumbing a reviewer would have to untangle."""

    def __init__(self, inner: Callable[[str, str, int], RetrieveResult]) -> None:
        self._inner = inner
        self.last_latency_ms = 0.0

    def __call__(self, question: str, patient_id: str, k: int) -> RetrieveResult:
        start = time.monotonic()
        result = self._inner(question, patient_id, k)
        self.last_latency_ms = (time.monotonic() - start) * 1000.0
        return result


class DenseRAGAnswerer(ReaderAnswerer):
    """Rung 3 of the baseline ladder, registered under `name = "dense"`.
    Subclasses `ReaderAnswerer` purely to supply `DenseRetriever` as the
    retrieval source — reading strategy, prompt, schema, and per-item token
    accounting are 100% shared with `reader_direct`/`reader_con` (see module
    docstring: "read by the same reader as everyone else").

    The one deliberate override of `ReaderAnswerer.answer()`: retrieval
    latency (chunking is cached, but the cosine search itself is not free)
    is added into the returned `AnswerResult.latency_ms`, so the harness's
    latency column reports the *full* per-query cost of this system
    (retrieve + read), not just the read step — story requirement: "Record
    tokens consumed and latency per query for both — the Pareto claim lives
    or dies on these." Token accounting needs no such override: prompt
    tokens are already a function of the retrieved evidence's size via
    `render_context()` inside `read()`, so a larger/smaller chunk size or k
    is already faithfully reflected in `prompt_tokens` with zero extra code.
    """

    name = "dense"
    mode = "chain_of_note"

    def __init__(
        self,
        *,
        chunk_size_tokens: int = DEFAULT_CHUNK_SIZE_TOKENS,
        rendering: str = "json",
        k: int = DEFAULT_K,
        dry_run: bool = False,
        model: str = DEFAULT_READER_MODEL,
        client: object | None = None,
        root: object | None = None,
        conversation_loader: Callable[[str, object | None], Conversation] = load_conversation,
    ) -> None:
        self.chunk_size_tokens = chunk_size_tokens
        self._timed_retriever = _TimedRetriever(
            DenseRetriever(chunk_size_tokens=chunk_size_tokens, root=root, conversation_loader=conversation_loader)
        )
        super().__init__(
            rendering=rendering, k=k, dry_run=dry_run, model=model, client=client, retriever=self._timed_retriever
        )

    def answer(
        self,
        question: str,
        conversation: Conversation | None,
        *,
        patient_id: str,
        scope: str | None = None,
        question_type: str | None = None,
    ) -> AnswerResult:
        result = super().answer(
            question, conversation, patient_id=patient_id, scope=scope, question_type=question_type
        )
        return replace(result, latency_ms=result.latency_ms + self._timed_retriever.last_latency_ms)


# ---------------------------------------------------------------------------
# Chunk/k sweep — retrieval-only, LLM-independent (see docs/algorithms/
# dense-lexical-baselines.md for why this, not dry-run QA accuracy, is what
# decides "which chunk size and k won").
# ---------------------------------------------------------------------------


def _admission_hit(items: list[RetrieveItem], gold_admissions: list[str]) -> bool:
    gold = set(gold_admissions)
    return any(item.session_id in gold for item in items)


def dense_recall_sweep(
    qa_items: list[dict],
    conversation: Conversation,
    *,
    chunk_sizes: tuple[int, ...] = CHUNK_SIZE_SWEEP,
    k_values: tuple[int, ...] = K_SWEEP,
) -> list[dict]:
    """Retrieval-only sweep across the `chunk_sizes` x `k_values` grid: for
    each QA item, does the gold admission (`evidence.admissions`, 100%
    coverage per `pipeline/loader.py`) appear among the retrieved chunks'
    `session_id`s? This is deliberately **not** routed through an LLM judge
    at all — it is meaningful with or without `ANTHROPIC_API_KEY`, unlike
    the harness's `--dry-run` answerable_accuracy (a stub-vs-token-overlap
    artifact — see the prior `[dev-ml]` dev.log.md entries' own disclaimer
    on that number). This is what actually decides "which chunk size and k
    won" for this story's return note.

    Not `metrics.recall_at_k`/`ndcg_at_k` (E7-S2's on-disk story assigns
    those, and a canonical `Recall@k`/`NDCG@k`, to `eval/metrics.py`, which
    does not exist in this repo as of this story — verified by grep, not
    assumed) — this is a narrower, single-purpose "did the top-k contain the
    right session at all" hit-rate local to this sweep, named
    `admission_hit_rate` rather than `recall` to avoid colliding with that
    future canonical name.

    Returns one row per `(chunk_size, k)` combination:
    `{chunk_size, k, n_items, admission_hit_rate, mean_retrieved_tokens,
    mean_latency_ms}`.
    """
    rows: list[dict] = []
    for chunk_size in chunk_sizes:
        index = DensePatientIndex(conversation, chunk_size)
        for k in k_values:
            hits = 0
            tokens: list[int] = []
            latencies: list[float] = []
            for item in qa_items:
                result = index.query(item["question"], k)
                gold_admissions = item.get("evidence", {}).get("admissions", [])
                if _admission_hit(result.items, gold_admissions):
                    hits += 1
                tokens.append(sum(estimate_tokens(it.text) for it in result.items))
                latencies.append(result.latency_ms.get("total", 0.0))
            n = len(qa_items)
            rows.append(
                {
                    "chunk_size": chunk_size,
                    "k": k,
                    "n_items": n,
                    "admission_hit_rate": (hits / n) if n else None,
                    "mean_retrieved_tokens": (sum(tokens) / n) if n else 0.0,
                    "mean_latency_ms": (sum(latencies) / n) if n else 0.0,
                }
            )
    return rows
