"""eval/baselines/lexical.py — Rung 4 of the baseline ladder: `bm25s` over
turn-granularity units, engineered with the ReFind session-aware
context-window expansion, not a bare BM25 index.

`literature/12-vector-search-and-hybrid-fusion.md` §5 is the reason "bare
BM25" is explicitly banned by this story: ReFind (arXiv 2608.12888)
indexes an "unmodified chat archive with a plain turn-granularity BM25
inverted index — standard Robertson & Zaragoza scoring with k1=1.2,
b=0.75" and, on its own six-benchmark evaluation, attains the *highest*
mean accuracy of any compared system (58.2), above even the strongest
graph baseline HippoRAG 2 (53.2) [R-VEC-062]. But ReFind's own ablation is
explicit about *why*: a "generic agentic BM25" control — same retrieval
loop, but returning only raw turn-level BM25 hits, no session-aware
context expansion, no RRF, no temporal filter — scores 14.5 / 7.1 points
*below* the full system on LongMemEval-S/M [R-VEC-063]. Of the controls
layered on top of bare BM25, removing **context-window expansion**
(returning the +/-2 neighboring turns around a BM25 hit) causes the single
largest accuracy drop (9.2 points on the S subset) — bigger than removing
RRF reranking (-3.9) or temporal filtering (-1.9) [R-VEC-064]. A bare BM25
implementation would understate this rung by exactly the margin that
finding names, and "engineer it properly" (the story's own words) means:
implement that one highest-value control, session-aware (never expand
across an admission boundary — a neighboring turn from a *different*
hospital stay is not "local context", it is a different session's text
attributed to the wrong evidence).

This baseline does *not* implement RRF reranking or temporal filtering —
those are two more of ReFind's ablated controls, worth real accuracy
(-3.9 / -1.9 points respectively) but smaller than context expansion, and
adding them would blur rung 4 into rung 5's "fusion" territory, which the
story explicitly excludes ("no rerank... no fusion" is dense.py's
constraint; this file's analogous constraint is "the ReFind context
expansion, not the rest of ReFind's agentic loop").

Runs entirely without HydraDB — same convention as `nomem.py`/`fullctx.py`/
`dense.py`. Read by the same Chain-of-Note reader as every other system in
the results table (see `dense.py`'s module docstring for why that is
deliberate, not incidental).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Callable

import bm25s

from medmemgraph.contracts import RetrieveItem, RetrieveResult
from medmemgraph.eval.reader import DEFAULT_READER_MODEL, ReaderAnswerer
from medmemgraph.eval.types import AnswerResult, estimate_tokens
from medmemgraph.pipeline.loader import Conversation, Turn, load_conversation

__all__ = [
    "DEFAULT_WINDOW",
    "DEFAULT_K",
    "K_SWEEP",
    "TurnUnit",
    "LexicalPatientIndex",
    "LexicalRetriever",
    "LexicalAnswerer",
    "lexical_recall_sweep",
]

DEFAULT_WINDOW = 2
"""+/-2 neighboring turns, exactly ReFind's own ablated expansion window
(literature/12 R-VEC-064: "returning the +/-2 neighboring turns around a
BM25 hit")."""

K_SWEEP = (5, 10)
"""Same two k values `dense.py` sweeps, so a per-k dense-vs-lexical
comparison at matched k is directly readable off the two sweep tables."""

DEFAULT_K = K_SWEEP[0]


def _format_turn(turn: Turn) -> str:
    return f"[admission {turn.hadm_id} turn {turn.turn_number} · {turn.time} · {turn.speaker}] {turn.text}"


# ---------------------------------------------------------------------------
# Turn-granularity index + session-aware context expansion
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TurnUnit:
    """One indexed turn: its admission, its position within that
    admission's own turn list (used to expand a window without crossing an
    admission boundary), and its rendered text."""

    session_id: str
    turn_number: int
    position: int
    text: str


class LexicalPatientIndex:
    """Turn-granularity `bm25s` index for one patient, with session-aware
    +/-`window` context expansion applied at query time (ReFind's
    highest-value ablated control, see module docstring)."""

    def __init__(self, conversation: Conversation, *, window: int = DEFAULT_WINDOW) -> None:
        self.window = window
        self._admission_turns: dict[str, list[Turn]] = {
            admission.hadm_id: admission.turns() for admission in conversation.admissions
        }
        self.units: list[TurnUnit] = [
            TurnUnit(session_id=hadm_id, turn_number=turn.turn_number, position=pos, text=_format_turn(turn))
            for hadm_id, turns in self._admission_turns.items()
            for pos, turn in enumerate(turns)
        ]

        self._bm25: bm25s.BM25 | None = None
        if self.units:
            corpus_texts = [u.text for u in self.units]
            corpus_tokens = bm25s.tokenize(corpus_texts, stopwords="english", show_progress=False)
            self._bm25 = bm25s.BM25()
            self._bm25.index(corpus_tokens, show_progress=False)

    def query(self, question: str, k: int) -> RetrieveResult:
        start = time.monotonic()
        if self._bm25 is None or not self.units:
            latency_ms = (time.monotonic() - start) * 1000.0
            return RetrieveResult(
                items=[], route="lexical", structural_absence=True, paths=[], latency_ms={"total": latency_ms}
            )

        query_tokens = bm25s.tokenize([question], stopwords="english", show_progress=False)
        k_eff = min(k, len(self.units))
        doc_ids, scores = self._bm25.retrieve(query_tokens, k=k_eff, show_progress=False)

        items = [self._expand(self.units[idx], float(score)) for idx, score in zip(doc_ids[0], scores[0])]
        latency_ms = (time.monotonic() - start) * 1000.0
        return RetrieveResult(
            items=items, route="lexical", structural_absence=False, paths=[], latency_ms={"total": latency_ms}
        )

    def _expand(self, unit: TurnUnit, score: float) -> RetrieveItem:
        """Session-aware +/-`window` context expansion (see module
        docstring). `unit.position` is this turn's index within its *own*
        admission's turn list, so the slice below can never reach into a
        different admission's turns — "session-aware" is exactly this
        constraint, not a euphemism for it."""
        turns = self._admission_turns[unit.session_id]
        lo = max(0, unit.position - self.window)
        hi = min(len(turns), unit.position + self.window + 1)
        window_turns = turns[lo:hi]
        return RetrieveItem(
            text="\n".join(_format_turn(t) for t in window_turns),
            session_id=unit.session_id,
            turn_ids=[t.turn_number for t in window_turns],
            score=score,
            channel="lexical",
        )


class LexicalRetriever:
    """Callable `(question, patient_id, k) -> RetrieveResult` — same shape
    as `contracts.mock_retrieve`/`dense.DenseRetriever`, so it drops into
    `eval.reader.ReaderAnswerer(retriever=...)` identically. Caches one
    `LexicalPatientIndex` per patient across a whole QA sweep."""

    def __init__(
        self,
        *,
        window: int = DEFAULT_WINDOW,
        root: object | None = None,
        conversation_loader: Callable[[str, object | None], Conversation] = load_conversation,
    ) -> None:
        self.window = window
        self.root = root
        self._conversation_loader = conversation_loader
        self._cache: dict[str, LexicalPatientIndex] = {}

    def _index_for(self, patient_id: str) -> LexicalPatientIndex:
        index = self._cache.get(patient_id)
        if index is None:
            conversation = self._conversation_loader(patient_id, self.root)
            index = LexicalPatientIndex(conversation, window=self.window)
            self._cache[patient_id] = index
        return index

    def __call__(self, question: str, patient_id: str, k: int) -> RetrieveResult:
        return self._index_for(patient_id).query(question, k)


class _TimedRetriever:
    """See `dense.py::_TimedRetriever` docstring for why this eight-line
    shim is deliberately duplicated rather than imported from `dense.py` —
    keeping this baseline file import-independent of the dense baseline."""

    def __init__(self, inner: Callable[[str, str, int], RetrieveResult]) -> None:
        self._inner = inner
        self.last_latency_ms = 0.0

    def __call__(self, question: str, patient_id: str, k: int) -> RetrieveResult:
        start = time.monotonic()
        result = self._inner(question, patient_id, k)
        self.last_latency_ms = (time.monotonic() - start) * 1000.0
        return result


class LexicalAnswerer(ReaderAnswerer):
    """Rung 4 of the baseline ladder, registered under `name = "lexical"`.
    See `dense.DenseRAGAnswerer`'s docstring — identical reasoning for
    subclassing `ReaderAnswerer` and for the retrieval-latency override on
    `answer()`; only the retriever differs."""

    name = "lexical"
    mode = "chain_of_note"

    def __init__(
        self,
        *,
        window: int = DEFAULT_WINDOW,
        rendering: str = "json",
        k: int = DEFAULT_K,
        dry_run: bool = False,
        model: str = DEFAULT_READER_MODEL,
        client: object | None = None,
        root: object | None = None,
        conversation_loader: Callable[[str, object | None], Conversation] = load_conversation,
    ) -> None:
        self.window = window
        self._timed_retriever = _TimedRetriever(
            LexicalRetriever(window=window, root=root, conversation_loader=conversation_loader)
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
# k sweep — retrieval-only, LLM-independent (mirrors dense.dense_recall_sweep)
# ---------------------------------------------------------------------------


def _admission_hit(items: list[RetrieveItem], gold_admissions: list[str]) -> bool:
    gold = set(gold_admissions)
    return any(item.session_id in gold for item in items)


def lexical_recall_sweep(
    qa_items: list[dict],
    conversation: Conversation,
    *,
    k_values: tuple[int, ...] = K_SWEEP,
    window: int = DEFAULT_WINDOW,
) -> list[dict]:
    """Retrieval-only k sweep, LLM-independent — same `admission_hit_rate`
    definition and same non-canonical-metrics-name caveat as
    `dense.dense_recall_sweep` (see that function's docstring). There is no
    "chunk size" axis here (this baseline indexes at fixed turn granularity
    plus a fixed +/-2 expansion window, per ReFind's own ablated design —
    sweeping the window size is future work, not this story's ask), so this
    sweep varies only `k`, over the same `K_SWEEP` values `dense.py` uses,
    specifically so the two sweep tables are comparable at matched k."""
    index = LexicalPatientIndex(conversation, window=window)
    rows: list[dict] = []
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
                "k": k,
                "n_items": n,
                "admission_hit_rate": (hits / n) if n else None,
                "mean_retrieved_tokens": (sum(tokens) / n) if n else 0.0,
                "mean_latency_ms": (sum(latencies) / n) if n else 0.0,
            }
        )
    return rows
