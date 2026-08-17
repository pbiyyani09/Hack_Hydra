"""graph/lexical.py — per-patient `bm25s` lexical index, beside HydraDB.
The other half of the "beside HydraDB" arm (`vector_index.py` is the
dense half): HydraDB OSS has no full-text/BM25 index (confirmed absent
from the OSS engine by exhaustive grep over
`literature/10-hydradb-capability-audit.md`), so lexical retrieval also
runs entirely in-process.

Authorities:

- `collaborative/design/ARCHITECTURE.md` §7.4: "Lexical: `bm25s` over the
  same corpus. MIT, pure Python + NumPy. ReFind's headline is real;
  ReFind's own ablation says plain BM25 is not the whole story
  (session-aware aggregation, context-window expansion, temporal filter).
  Build `bm25s` first; add turn-window expansion if the lexical baseline
  looks weak."
- `collaborative/literature/12-vector-search-and-hybrid-fusion.md` §5/Q5:
  ReFind (arXiv 2608.12888) indexes raw turns with plain
  Robertson-Zaragoza BM25 (`k1=1.2, b=0.75`) and tops six conversational-
  memory benchmarks [R-VEC-061/062]. But its own ablation shows a "generic
  agentic BM25" control (no session-aware RRF, no context-window
  expansion, no temporal filter) scores 14.5 / 7.1 points *below* the
  full system on LongMemEval-S/M [R-VEC-063], and of those controls,
  removing context-window expansion (the +/-2 neighboring turns around a
  hit) causes the single *largest* accuracy drop (9.2 points on the S
  subset) — bigger than removing RRF (-3.9) or temporal filtering (-1.9)
  [R-VEC-064]. This story is explicitly the engineered version, not a
  bare BM25 index: `search()` always applies the +/-2 session-aware
  expansion below. (Session-level RRF and temporal filtering are *out* of
  this story per E5-S2's own "Lexical extras" note — those are cut #2 and
  a separate fusion-stage concern, `ARCHITECTURE.md` §7.5, E5-S4.)
- `collaborative/design/stories/E5/E5-S2.md` — `PatientLexical` in that
  packet's own sketch; this file follows the dispatch's own class-shape
  instruction (`LexicalIndex`, matching `PatientIndex`'s
  build/search/save/load shape so the router can treat channels
  uniformly — see `docs/algorithms/vector-lexical-index.md`).

Banned: session-level RRF (cut, deferred to E5-S4's fusion stage),
temporal filtering, any LLM/untrained reranker (RMM 58.8% -> 31.0%;
literature/12 R-VEC-080), a second, forked BM25 implementation elsewhere
in the Graph-owned tree (`E5-S2` Boundary).
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import bm25s

from medmemgraph.contracts import RetrieveItem
from medmemgraph.pipeline.loader import Conversation
from medmemgraph.graph.vector_index import DEFAULT_INDEX_DIR, format_turn

__all__ = [
    "DEFAULT_WINDOW",
    "DEFAULT_INDEX_DIR",
    "TurnUnit",
    "LexicalIndex",
]

DEFAULT_WINDOW = 2
"""+/-2 neighboring turns — ReFind's own ablated expansion window
(literature/12 R-VEC-064). Not configurable away from 2 by this story's
tests without a stated reason; 2 is the number the cited ablation
measured, not an arbitrary default."""


@dataclass(frozen=True)
class TurnUnit:
    """One indexed turn: its admission, this turn's position within that
    *same* admission's own turn list (so expansion can never reach across
    an admission boundary), and its rendered text."""

    session_id: str
    turn_number: int
    position: int
    text: str


@dataclass(frozen=True)
class _ExpandTurn:
    """Minimal, already-rendered stand-in for `pipeline.loader.Turn`,
    carrying only what `_expand` ever reads (`turn_number` for
    `RetrieveItem.turn_ids`, `text` for the joined window). Used
    uniformly by both `build()` (freshly rendered via `format_turn`) and
    `load()` (reconstructed from the saved sidecar) so `_expand` has
    exactly one code path regardless of whether the index was just built
    or loaded from disk — `format_turn` is called at most once per turn,
    at build time, never again at expansion time (a saved index does not
    keep the original `hadm_id`/`time`/`speaker` fields `format_turn`
    would need to re-render)."""

    turn_number: int
    text: str


class LexicalIndex:
    """Per-patient `bm25s` index over raw turns, with session-aware
    +/-`window` context expansion applied at query time. Same
    build/search/save/load shape as `vector_index.PatientIndex` (story
    requirement: "Same interface shape ... so the router can treat
    channels uniformly")."""

    def __init__(self, patient_id: str, *, window: int = DEFAULT_WINDOW) -> None:
        self.patient_id = patient_id
        self.window = window
        self._admission_turns: dict[str, list[_ExpandTurn]] = {}
        self.units: list[TurnUnit] = []
        self._bm25: bm25s.BM25 | None = None
        self.build_time_s: float = 0.0

    # -- build ---------------------------------------------------------

    def build(self, conversation: Conversation) -> None:
        """Index every turn in `conversation` at turn granularity
        (literature/12 Q4/Q5: turn-level indexing *with* neighbor
        expansion at retrieval time, not a pre-baked window baked into the
        index). Facts are not indexed here — atomic-fact text is a dense-
        retrieval-side granularity per `ARCHITECTURE.md` §7.4's own split
        (dense indexes "turn text and emitted fact text"; the lexical
        bullet only ever says "the same corpus" of turns) and this
        story's contract does not ask for a second, lexical-side fact
        index."""
        start = time.monotonic()
        self._admission_turns = {
            admission.hadm_id: [_ExpandTurn(turn_number=t.turn_number, text=format_turn(t)) for t in admission.turns()]
            for admission in conversation.admissions
        }
        self.units = [
            TurnUnit(session_id=hadm_id, turn_number=et.turn_number, position=pos, text=et.text)
            for hadm_id, turns in self._admission_turns.items()
            for pos, et in enumerate(turns)
        ]
        self._bm25 = None
        if self.units:
            corpus_texts = [u.text for u in self.units]
            corpus_tokens = bm25s.tokenize(corpus_texts, stopwords="english", show_progress=False)
            self._bm25 = bm25s.BM25()
            self._bm25.index(corpus_tokens, show_progress=False)
        self.build_time_s = time.monotonic() - start

    # -- search ----------------------------------------------------------

    def search(self, query: str, k: int) -> list[RetrieveItem]:
        """Top-`k` BM25 hits, each expanded to its +/-`window` session-
        local context window. Empty index -> `[]` (E5-S2 AC2, mirrored
        from the vector channel — do not invent a hit)."""
        if self._bm25 is None or not self.units:
            return []
        query_tokens = bm25s.tokenize([query], stopwords="english", show_progress=False)
        k_eff = min(k, len(self.units))
        doc_ids, scores = self._bm25.retrieve(query_tokens, k=k_eff, show_progress=False)
        return [self._expand(self.units[idx], float(score)) for idx, score in zip(doc_ids[0], scores[0])]

    def _expand(self, unit: TurnUnit, score: float) -> RetrieveItem:
        """Session-aware +/-`window` expansion. `unit.position` is this
        turn's index within its *own* admission's turn list (set at build
        time from `enumerate(turns)` per admission), so the slice below
        can never reach into a different admission's turns — "session-
        aware" is exactly this constraint, not a euphemism for it
        (E5-S2 AC3: "the hit's text includes turns 2-6 (or the available
        window)")."""
        turns = self._admission_turns[unit.session_id]
        lo = max(0, unit.position - self.window)
        hi = min(len(turns), unit.position + self.window + 1)
        window_turns = turns[lo:hi]
        return RetrieveItem(
            text="\n".join(t.text for t in window_turns),
            session_id=unit.session_id,
            turn_ids=[t.turn_number for t in window_turns],
            score=score,
            channel="lexical",
        )

    # -- persistence -------------------------------------------------------

    def index_size_bytes(self) -> int:
        """In-memory footprint estimate: the BM25 sparse-score arrays plus
        the indexed turn text — same reporting shape as
        `PatientIndex.index_size_bytes`."""
        text_bytes = sum(len(u.text.encode("utf-8")) for u in self.units)
        if self._bm25 is None:
            return text_bytes
        score_bytes = sum(
            int(arr.nbytes) for key, arr in self._bm25.scores.items() if hasattr(arr, "nbytes")
        )
        return text_bytes + score_bytes

    def _safe_id(self) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]", "_", self.patient_id)

    def _paths(self, index_dir: Path) -> tuple[Path, Path]:
        safe_id = self._safe_id()
        return index_dir / f"{safe_id}.bm25", index_dir / f"{safe_id}.bm25.meta.json"

    def save(self, index_dir: str | Path = DEFAULT_INDEX_DIR) -> Path:
        """Persist this patient's built index to `index_dir` (default
        `data/index/`, gitignored per decisions/001) so a rebuild is not
        needed every process start. `bm25s`'s own `.save()` writes the
        sparse score arrays + vocab to a directory; a JSON sidecar carries
        `patient_id`, `window`, and every unit's `session_id`/
        `turn_number`/`position`/`text` (needed to reconstruct
        `_admission_turns` for expansion after a fresh-process load,
        without re-reading `combined_conversation.json`)."""
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        bm25_dir, meta_path = self._paths(index_dir)
        if self._bm25 is not None:
            self._bm25.save(str(bm25_dir), show_progress=False)
        meta = {
            "patient_id": self.patient_id,
            "window": self.window,
            "build_time_s": self.build_time_s,
            "has_index": self._bm25 is not None,
            "units": [
                {"session_id": u.session_id, "turn_number": u.turn_number, "position": u.position, "text": u.text}
                for u in self.units
            ],
        }
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        return bm25_dir

    @classmethod
    def load(cls, patient_id: str, index_dir: str | Path = DEFAULT_INDEX_DIR, *, window: int | None = None) -> "LexicalIndex":
        """Inverse of `save`. Raises `FileNotFoundError` if no saved index
        exists for `patient_id` — a missing index and an empty patient are
        different facts (same discipline as `PatientIndex.load`)."""
        index_dir = Path(index_dir)
        safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", patient_id)
        bm25_dir = index_dir / f"{safe_id}.bm25"
        meta_path = index_dir / f"{safe_id}.bm25.meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"no saved lexical index for patient_id={patient_id!r} under {index_dir}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        index = cls(patient_id, window=window if window is not None else meta["window"])
        index.build_time_s = meta.get("build_time_s", 0.0)
        index.units = [
            TurnUnit(session_id=u["session_id"], turn_number=u["turn_number"], position=u["position"], text=u["text"])
            for u in meta["units"]
        ]
        # Reconstruct each admission's turn list, position-ordered, as the
        # same `_ExpandTurn` stand-ins `build()` populates — `format_turn`
        # output was already baked into `unit.text` at build time, so
        # `_expand` re-joins the *saved* per-turn text uniformly on both
        # the built and the loaded path (see `_ExpandTurn` docstring).
        by_session: dict[str, list[tuple[int, _ExpandTurn]]] = {}
        for unit in index.units:
            by_session.setdefault(unit.session_id, []).append(
                (unit.position, _ExpandTurn(turn_number=unit.turn_number, text=unit.text))
            )
        index._admission_turns = {
            session_id: [t for _, t in sorted(pairs, key=lambda p: p[0])] for session_id, pairs in by_session.items()
        }
        if meta.get("has_index") and bm25_dir.exists():
            index._bm25 = bm25s.BM25.load(str(bm25_dir), load_corpus=False, show_progress=False)
        else:
            index._bm25 = None
        return index
