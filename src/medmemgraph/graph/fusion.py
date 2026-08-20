"""graph/fusion.py — Reciprocal Rank Fusion, `k=60` (ARCHITECTURE.md §7.5,
story E5-S4, `literature/12` — RRF is the field default named independently
by three separate papers in that survey).

Fuses ranked lists whose scores are **not mutually comparable**: cosine
similarity (dense), BM25 (lexical), and a client-side heuristic path score
(`γ^hop_count · confidence · recency`, `graph/traverse.py`'s `rank_paths`)
live on three different, unnormalized scales. RRF sidesteps ever having to
normalize them by fusing on RANK POSITION alone, never on the raw score
value:

    RRF(d) = Σ_i  1 / (k + rank_i(d))

(Cormack, Clarke & Buettcher, "Reciprocal Rank Fusion outperforms Condorcet
and Individual Rank Learning Methods", SIGIR 2009 — the citation
`literature/12` traces the `k=60` default to, alongside two later
independent confirmations; see that survey for the full citation chain.)
Ranks are 1-based; a document/path absent from one input list simply
contributes 0 to that list's term — never a penalty, never an imputed rank.

Per-channel weights are exposed (`weights=`) with a documented EQUAL default
(1.0 each) — **not a derived optimum**: `literature/13` found no principled
published method for splitting one token/relevance budget across
heterogeneous retrieval channels, so this default is a stated engineering
judgment, overridable by a caller that calibrates against the eval harness,
never a result this module claims to have derived.

Banned (ARCHITECTURE §7.5, this module's own story E5-S4): score
normalization, a per-query LLM (DAT-style) weight, and any weighted-sum of
cosine vs `pathWeight` vs BM25.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace

from medmemgraph.contracts import RetrieveItem

__all__ = ["DEFAULT_RRF_K", "DEFAULT_WEIGHT", "IdentityKeyFn", "identity_key", "rrf_fuse"]

DEFAULT_RRF_K = 60
"""The field default (`literature/12`) — three independent papers converge
on this exact constant. Not tuned per-project; there is no evidence base
in this project's own survey for deviating from it."""

DEFAULT_WEIGHT = 1.0
"""Equal per-channel weight — the documented "no principled method to split
the budget" default (module docstring). Overridable per-call via
`weights=`, never silently re-derived elsewhere."""

IdentityKeyFn = Callable[[RetrieveItem], tuple]


def identity_key(item: RetrieveItem) -> tuple:
    """Default de-duplication key across ranked lists: `(session_id,
    turn_ids, text)`. Two items are "the same piece of evidence" for RRF
    purposes iff they render to the exact same provenance + text — a graph
    path and a lexical turn window naturally never collide under this key
    (their `text` never matches byte-for-byte), which is intentional: this
    module does not attempt cross-channel entity resolution, only exact
    de-duplication of genuinely repeated hits (e.g. the same turn window
    surfacing from both the dense and lexical arm)."""
    return (item.session_id, tuple(item.turn_ids), item.text)


def rrf_fuse(
    ranked_lists: Sequence[Sequence[RetrieveItem]],
    *,
    k: int = DEFAULT_RRF_K,
    weights: Sequence[float] | None = None,
    key_fn: IdentityKeyFn = identity_key,
) -> list[RetrieveItem]:
    """Reciprocal Rank Fusion over `ranked_lists` (each already sorted
    best-first — RRF only ever looks at POSITION, never at a list's own
    score scale). Returns items sorted descending by fused RRF score.

    Each returned item is a `replace()` of whichever input item first
    contributed that de-duplication key at its BEST (lowest-numbered) rank
    across all lists — `.score` is OVERWRITTEN with the fused RRF score
    (`RetrieveItem.score`'s one field, never a second parallel field —
    `contracts.py`'s shape is frozen) and `.channel` is left as that
    winning item's own channel, so a caller can still tell which arm
    produced the representative rendering of a given piece of evidence even
    after fusion. A key present in more than one list keeps the rendering
    from whichever list ranked it best, on the reasoning that the
    best-ranking channel's own text/provenance is the more representative
    one — never a synthesized merge of two different renderings.

    Empty `ranked_lists` (or every list empty) returns `[]`, not an error —
    fusing nothing is a valid, silent no-op."""
    if weights is None:
        weights = [DEFAULT_WEIGHT] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError(
            f"rrf_fuse: weights (len={len(weights)}) must have the same length as "
            f"ranked_lists (len={len(ranked_lists)})"
        )

    scores: dict[tuple, float] = {}
    best_item: dict[tuple, RetrieveItem] = {}
    best_rank: dict[tuple, int] = {}

    for ranked_list, weight in zip(ranked_lists, weights):
        for rank, item in enumerate(ranked_list, start=1):
            key = key_fn(item)
            scores[key] = scores.get(key, 0.0) + weight * (1.0 / (k + rank))
            if key not in best_rank or rank < best_rank[key]:
                best_rank[key] = rank
                best_item[key] = item

    fused = [replace(best_item[key], score=score) for key, score in scores.items()]
    fused.sort(key=lambda it: it.score, reverse=True)
    return fused
