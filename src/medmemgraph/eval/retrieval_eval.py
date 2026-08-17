"""eval/retrieval_eval.py — the IR metric sweep and embedder/reranker
ablation. This answers the one question the whole project rests on: is
retrieval actually working? If recall is poor, every downstream accuracy
number in `eval/report.py`'s Pareto table is measuring the reader's ability
to guess, not the memory system's ability to find the right evidence.

--------------------------------------------------------------------------
Two-stage pipeline, two metric families
--------------------------------------------------------------------------
This module scores the exact two-stage retrieve-then-rerank pipeline
`graph/reranker.py` already implements (`two_stage_retrieve`): a cheap
bi-encoder (`graph/vector_index.py::PatientIndex`, one of `graph/
embedders.py`'s three registered backbones) fetches a wide candidate pool;
an optional trained cross-encoder (`graph/reranker.py`'s `NoopReranker` /
`CrossEncoderReranker`) reorders a shortlist of it. `ai-engineering` ch6's
own hybrid-search framing is exactly this shape: "cheap retriever fetches
candidates, expensive one reranks."

Two metric families, one per stage, per the dispatching story:

  * **Bi-encoder stage (pre-rerank):** `Recall@{10,20,50,100,500}` — the
    *fraction* of gold evidence units found in the wide candidate pool.
    This is the stage that decides the retrieval *ceiling*: nothing the
    reranker or reader does later can recover a gold unit the bi-encoder
    never surfaced at all.
  * **Post-rerank:** `Hit@{2,5,10,20}` and `nDCG@{2,5,10,20}` — the
    *rank quality* of the final, reranked shortlist actually handed to a
    reader. `Hit@k` (any gold unit in the top-k, binary) is deliberately
    not `Recall@k` (fraction of gold units in the top-k) at this stage —
    see `eval/metrics.py::hit_at_k`'s own docstring for why conflating
    them silently misrepresents a system that found "some evidence" as
    having found "most of the evidence."

`N_CANDIDATES_FOR_RERANK` (100) is the shortlist size handed to the
reranker — reranking the full 500-candidate bi-encoder pool with a
cross-encoder would be the expensive-stage-does-cheap-stage's-job mistake
`ai-engineering` ch6 warns against; 100 is generous headroom over every
`RERANK_KS` value (max 20) while keeping the sweep's wall-clock/GPU cost
tractable (measured directly on this project's RTX 3090 in this session —
see the dev-log return note for this story for the pasted throughput
numbers).

--------------------------------------------------------------------------
Grounding — both granularities, every table states both n's
--------------------------------------------------------------------------
`eval/metrics.py::recall_at_k`/`ndcg_at_k`/`hit_at_k` auto-select grain
per QA item (turn-level if the item's own gold `evidence.turn_ids` is
present, else admission-level) — correct for a single scalar-per-item
call, but this story requires BOTH granularities scored explicitly and
reported with their own denominators (admission ids are on 100% of items;
turn ids on ~50% — `pipeline/loader.py`'s own docstring). `admission_only_
evidence`/`turn_only_evidence` below construct two independent evidence
views per item (stripping/keeping `turn_ids`) so the same three metric
functions are called twice, once per grain, rather than reimplemented —
story requirement: "Reuse recall_at_k and ndcg_at_k rather than
reimplementing." An item with no turn-level gold contributes `None` to
every turn-grain field, not a zero-filled row — `aggregate_config` below
divides by the count of non-`None` items for the turn-level n, never by
the full item count (`designing-ml-systems` ch06's slice-based-evaluation
discipline: a metric silently averaged over the wrong denominator is a
Simpson's-paradox trap, not an honest number).

--------------------------------------------------------------------------
Why this module retrieves once per (patient, embedder) and reuses it
across every reranker, rather than calling `reranker.two_stage_retrieve`
--------------------------------------------------------------------------
`graph/reranker.py::two_stage_retrieve(query, retrieve_fn=..., ...)` is the
right shape for a single production query, where the retrieve stage runs
exactly once per call by construction. This module runs a 3x3 embedder x
reranker grid over the SAME QA items, so re-running the (identical,
bi-encoder-only-depends-on-embedder) retrieve stage once per reranker would
be 3x wasted GPU work with zero effect on the numbers — `run_sweep` below
retrieves each (patient, embedder) pair's candidates exactly once per
embedder and replays that fixed candidate pool through every registered
reranker, a deliberate, documented divergence from `two_stage_retrieve`'s
per-call convention, not a reimplementation of its logic (reranking itself
is still a bare `reranker.rerank(query, docs, top_k=None)` call, the same
one `two_stage_retrieve` makes internally).

--------------------------------------------------------------------------
Sample size — `metrics.mde` decides what this run can detect
--------------------------------------------------------------------------
This project's own MDE reference table (`.claude/logs/dev.log.md` /
this story's own dispatch text): at n=250, detectable paired-proportion
effects run 12.28pp (uncorrelated systems) to 5.49pp (highly correlated);
at n=1000, 6.14pp to 2.74pp. Retrieval metrics are local-GPU-cheap (no API
budget consumed), so this module prefers a larger n than the LLM-scored
evals in `eval/report.py` can afford, subject to the real wall-clock
measured directly on this project's hardware in this session (patient
index build ~3.6s/patient/embedder; warm cross-encoder rerank of 100
candidates ~0.09-0.41s/query depending on backbone — see the dev-log
return note). `mde_table` below computes `metrics.mde(n, ...)` at the
ACTUAL n a given sweep run achieved (never the aspirational 250/1000
reference points), so every reported n carries its own honest detectable-
effect number, and `paired_reranker_vs_noop`'s `underpowered` flag marks
any observed delta smaller than what the paired n at hand could actually
detect (mirrors `eval/report.py::_paired_mde`'s own "derive the phi
correlation from McNemar's own 2x2 counts" construction — not
reimplemented, independently re-derived here because `report.py`'s
version is a private module-level helper, not an exported name).

--------------------------------------------------------------------------
The Recall@500 ceiling — reported, not treated as discriminating
--------------------------------------------------------------------------
A patient corpus in this project is ~1,000-3,000 turns (`pipeline/
loader.py`; the smoke-tested patient in this session had 1,451 turns).
Recall@500 therefore retrieves roughly a third to a half of an entire
patient's indexed units — not a "did retrieval work" signal but a "how
much of the whole patient history is in the candidate pool" ceiling check,
and it saturates toward 1.0 by construction as k approaches the corpus
size. `render_markdown` below prints `RECALL_CEILING_NOTE` verbatim
alongside every Recall@500 column so a reader does not mistake ceiling
saturation for retrieval quality — the informative Recall@k rows are the
low-k ones (10/20/50).

--------------------------------------------------------------------------
Admission-grain nDCG can legitimately exceed 1.0 — verified, not a bug
--------------------------------------------------------------------------
Discovered by inspecting the real sweep's own numbers (several
admission-grain nDCG cells came back > 1.0) and root-caused BEFORE being
reported, per this project's "adversarial self-check" rule — reproduced
directly against `eval/metrics.py::ndcg_at_k` with a hand-built minimal
case (`m.ndcg_at_k([Item("adm-X",[1]), Item("adm-X",[2]), Item("adm-X",[3]),
Item("adm-X",[4])], {"admissions": ["adm-X"]}, 10) == 2.56...`) before
being surfaced here. The cause: `ndcg_at_k`'s own IDCG uses `ideal_count =
min(k, n_gold_units)` (`eval/metrics.py`, pre-existing, not redefined by
this module — story requirement: reuse it verbatim). At admission grain,
`n_gold_units` is the number of DISTINCT gold admissions (often 1-4), but
`PatientIndex` indexes one unit PER TURN, so a reranked top-k can contain
several turns that all legitimately belong to that one gold admission —
each individually admission-relevant (rel=1), summed into DCG, while IDCG
stays capped at the 1-4-term ideal. The bound `nDCG<=1` assumes at most
`n_gold_units` retrieved items can be relevant; that assumption holds at
turn grain (this project's index has exactly one unit per turn, so at most
one retrieved item can carry any given gold turn id — confirmed empirically
in the real sweep: every turn-grain nDCG cell in this story's results
stayed within [0,1]) but does NOT hold at admission grain whenever more
than `n_gold_units` distinct turns from the correct admission(s) are
retrieved, which is common and not a failure mode — finding many turns from
the right admission is what a good retriever is SUPPOSED to do.

Practical reading: `Hit@k` and `Recall@k` are bounded and directly
comparable at both grains, always. `nDCG@k` at TURN grain is the standard,
bounded-in-[0,1] ranking-quality signal. `nDCG@k` at ADMISSION grain is
still a valid RELATIVE (higher-is-better, same-scale-across-configs)
ranking-quality signal — every config in a given sweep is scored by the
exact same `ndcg_at_k` definition against the exact same gold, so
config-vs-config comparisons remain fair — but it is not a `[0,1]`-bounded
proportion and must never be read or reported as one. `render_markdown`
prints `NDCG_ADMISSION_CAVEAT` verbatim beside every admission-grain nDCG
column for exactly this reason.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

from medmemgraph.contracts import RetrieveItem
from medmemgraph.eval import metrics
from medmemgraph.graph.reranker import CrossEncoderReranker, NoopReranker, RerankerBackend
from medmemgraph.graph.vector_index import PatientIndex
from medmemgraph.pipeline.loader import load_conversation, load_qa

__all__ = [
    "EMBEDDERS",
    "RERANKERS",
    "RECALL_KS",
    "RERANK_KS",
    "N_CANDIDATES_FOR_RERANK",
    "RECALL_CEILING_NOTE",
    "NDCG_ADMISSION_CAVEAT",
    "admission_only_evidence",
    "turn_only_evidence",
    "ItemMetrics",
    "evaluate_item",
    "ConfigResult",
    "SweepResult",
    "build_reranker",
    "run_sweep",
    "AggregateRow",
    "aggregate_config",
    "mde_table",
    "MdeTableRow",
    "paired_reranker_vs_noop",
    "render_markdown",
    "render_terminal_summary",
    "write_report",
]

# ---------------------------------------------------------------------------
# Ablation grid — the story's exact axes.
# ---------------------------------------------------------------------------

EMBEDDERS: tuple[str, ...] = ("qwen3-0.6b", "bge-m3", "arctic-l-v2")
"""`graph/embedders.py::EMBEDDER_REGISTRY` keys, verbatim — this module
never invents a fourth embedder name of its own."""

RERANKERS: tuple[str, ...] = ("noop", "qwen3-rerank-0.6b", "bge-rerank-v2-m3")
"""`"noop"` -> `graph.reranker.NoopReranker`; the other two ->
`graph.reranker.REGISTERED_MODELS` keys, verbatim."""

RECALL_KS: tuple[int, ...] = (10, 20, 50, 100, 500)
RERANK_KS: tuple[int, ...] = (2, 5, 10, 20)
N_CANDIDATES_FOR_RERANK = 100
"""Bi-encoder shortlist size handed to the reranker — see module
docstring's "why 100" note."""

RECALL_CEILING_NOTE = (
    "Recall@500 retrieves roughly a sixth to a half of an entire patient's "
    "indexed units on this project's corpora and saturates toward 1.0 as k "
    "approaches corpus size by construction. It is a ceiling check, not a "
    "discriminating measurement — the informative Recall@k rows are the "
    "low-k ones (10/20/50)."
)

NDCG_ADMISSION_CAVEAT = (
    "nDCG@k at admission grain can legitimately exceed 1.0: eval/metrics.py's "
    "ndcg_at_k caps IDCG at min(k, n_gold_admissions), but multiple distinct "
    "turns from one correct admission can all be individually relevant, "
    "inflating DCG past that cap (verified directly against eval/metrics.py "
    "with a hand-built case, see retrieval_eval.py module docstring). Read "
    "these values as a relative, same-scale-across-configs ranking signal, "
    "never as a bounded [0,1] proportion. nDCG@k at TURN grain does not have "
    "this property (one indexed unit per turn) and stays in [0,1]."
)


# ---------------------------------------------------------------------------
# Grounding — both granularities, from the same MedLoCoMo `evidence` dict.
# ---------------------------------------------------------------------------


def admission_only_evidence(evidence: dict) -> dict:
    """Strip any `turn_ids` key so `metrics.recall_at_k`/`ndcg_at_k`/
    `hit_at_k` score at admission grain regardless of whether this QA item
    also happens to carry turn-level gold — admission ids are on 100% of
    items (`pipeline/loader.py`), so this view is always scoreable."""
    return {"admissions": list(evidence.get("admissions") or [])}


def turn_only_evidence(evidence: dict) -> dict | None:
    """`None` when this QA item carries no turn-level gold (~50% of items,
    `pipeline/loader.py`) — the caller MUST skip turn-grain scoring for it
    entirely (never zero-fill), so the turn-level n in every aggregate
    table is the true count of groundable items, not the full item count."""
    turn_ids = evidence.get("turn_ids")
    if not turn_ids:
        return None
    return {"admissions": list(evidence.get("admissions") or []), "turn_ids": list(turn_ids)}


# ---------------------------------------------------------------------------
# Per-item metrics.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ItemMetrics:
    patient_id: str
    qa_id: str
    embedder: str
    reranker: str
    n_candidates: int
    """How many bi-encoder candidates were actually handed to the
    reranker for this item (`<= N_CANDIDATES_FOR_RERANK`; a small patient
    index can return fewer)."""
    retrieve_latency_ms: float
    rerank_latency_ms: float
    recall_admission: dict[int, float]
    recall_turn: dict[int, float] | None
    hit_admission: dict[int, float]
    hit_turn: dict[int, float] | None
    ndcg_admission: dict[int, float]
    ndcg_turn: dict[int, float] | None


def evaluate_item(
    *,
    patient_id: str,
    qa_id: str,
    embedder: str,
    reranker: str,
    bi_encoder_retrieved: list[RetrieveItem],
    reranked: list[RetrieveItem],
    evidence: dict,
    retrieve_latency_ms: float,
    rerank_latency_ms: float,
) -> ItemMetrics:
    """One QA item's full metric row: `Recall@RECALL_KS` against the
    bi-encoder's own wide candidate pool (`bi_encoder_retrieved`,
    pre-rerank order), and `Hit@RERANK_KS`/`nDCG@RERANK_KS` against the
    reranked shortlist (`reranked`, post-rerank order) — both computed at
    BOTH grains via `admission_only_evidence`/`turn_only_evidence`, reusing
    `metrics.recall_at_k`/`ndcg_at_k`/`hit_at_k` verbatim (never
    reimplemented — story requirement)."""
    adm_ev = admission_only_evidence(evidence)
    trn_ev = turn_only_evidence(evidence)

    recall_admission = {k: metrics.recall_at_k(bi_encoder_retrieved, adm_ev, k) for k in RECALL_KS}
    recall_turn = {k: metrics.recall_at_k(bi_encoder_retrieved, trn_ev, k) for k in RECALL_KS} if trn_ev else None

    hit_admission = {k: metrics.hit_at_k(reranked, adm_ev, k) for k in RERANK_KS}
    hit_turn = {k: metrics.hit_at_k(reranked, trn_ev, k) for k in RERANK_KS} if trn_ev else None

    ndcg_admission = {k: metrics.ndcg_at_k(reranked, adm_ev, k) for k in RERANK_KS}
    ndcg_turn = {k: metrics.ndcg_at_k(reranked, trn_ev, k) for k in RERANK_KS} if trn_ev else None

    return ItemMetrics(
        patient_id=patient_id,
        qa_id=qa_id,
        embedder=embedder,
        reranker=reranker,
        n_candidates=len(reranked),
        retrieve_latency_ms=retrieve_latency_ms,
        rerank_latency_ms=rerank_latency_ms,
        recall_admission=recall_admission,
        recall_turn=recall_turn,
        hit_admission=hit_admission,
        hit_turn=hit_turn,
        ndcg_admission=ndcg_admission,
        ndcg_turn=ndcg_turn,
    )


# ---------------------------------------------------------------------------
# Grid runner.
# ---------------------------------------------------------------------------


@dataclass
class ConfigResult:
    embedder: str
    reranker: str
    items: list[ItemMetrics]
    build_time_s_mean: float
    """Mean per-patient `PatientIndex.build()` wall-clock cost for this
    embedder (reranker-independent; recorded once per embedder, copied
    onto every reranker's `ConfigResult` for that embedder so every row
    of the final table is self-contained)."""
    embedder_vram_mb: float | None
    """Incremental CUDA memory allocated while loading/building this
    embedder's weights + index (delta across `run_sweep`'s own embedder
    loop iteration) — `None` on a CPU-only machine, never a fabricated
    number."""
    reranker_vram_mb: float | None
    """Incremental CUDA memory allocated while loading this reranker's
    weights (0.0 for `"noop"`, which loads nothing)."""
    n_patients: int


@dataclass
class SweepResult:
    configs: list[ConfigResult]
    patient_ids: list[str]
    n_candidates: int


def build_reranker(name: str) -> RerankerBackend:
    """`"noop"` -> `NoopReranker()`; any `graph.reranker.REGISTERED_MODELS`
    key -> a lazily-loaded `CrossEncoderReranker(name)` (loading happens on
    first `.rerank()` call, `run_sweep` forces it once via a warmup call so
    the load cost is measured and excluded from the timed per-item loop)."""
    if name == "noop":
        return NoopReranker()
    return CrossEncoderReranker(name)


def _cuda_mem_mb() -> float | None:
    """Current CUDA-allocated memory in MB, or `None` if torch/CUDA is
    unavailable — VRAM reporting degrades to "not measured," never a
    fabricated 0.0 (this project's hard "paste real numbers" rule)."""
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard project dependency
        return None
    if not torch.cuda.is_available():
        return None
    return torch.cuda.memory_allocated() / 1e6


def run_sweep(
    patient_ids: Sequence[str],
    *,
    embedders_: Sequence[str] = EMBEDDERS,
    rerankers_: Sequence[str] = RERANKERS,
    n_candidates: int = N_CANDIDATES_FOR_RERANK,
    root: object | None = None,
    cache_path: object | None = None,
) -> SweepResult:
    """Runs the full `embedders_ x rerankers_` grid over every QA item of
    every patient in `patient_ids`. For each embedder, the bi-encoder
    candidate pool (`Recall@RECALL_KS`'s own top-`max(RECALL_KS)` search)
    is computed exactly ONCE per (patient, QA item) and replayed through
    every reranker in `rerankers_` — see module docstring's "why this
    module retrieves once per (patient, embedder)" note.

    An item whose gold `evidence` carries no admissions at all (should
    never happen — `pipeline/loader.py` documents 100% admission coverage
    — but a real corpus can always surprise) is skipped, never silently
    scored as "0 relevant" (this project's hard "never silently score an
    ungroundable item" rule, `eval/metrics.py::_parse_gold_evidence`'s own
    convention, mirrored here at the sweep level rather than caught as an
    uncaught `ValueError` mid-run).
    """
    configs: list[ConfigResult] = []

    for embedder_name in embedders_:
        vram_before = _cuda_mem_mb()
        indexes: dict[str, PatientIndex] = {}
        build_times: list[float] = []
        for pid in patient_ids:
            conversation = load_conversation(pid, root)
            t0 = time.monotonic()
            index = PatientIndex(pid, backend=embedder_name, cache_path=cache_path)
            index.build(conversation)
            build_times.append(time.monotonic() - t0)
            indexes[pid] = index
        vram_after = _cuda_mem_mb()
        embedder_vram_mb = (
            max(0.0, vram_after - vram_before) if (vram_before is not None and vram_after is not None) else None
        )

        # Bi-encoder stage — once per (patient, QA item) for this embedder,
        # reused by every reranker below.
        bi_stage: dict[tuple[str, str], tuple[list[RetrieveItem], float, dict]] = {}
        for pid in patient_ids:
            for item in load_qa(pid, root):
                if not (item.get("evidence") or {}).get("admissions"):
                    continue  # ungroundable — never silently scored
                t0 = time.monotonic()
                retrieved = indexes[pid].search(item["question"], k=max(RECALL_KS))
                latency_ms = (time.monotonic() - t0) * 1000.0
                bi_stage[(pid, item["qa_id"])] = (retrieved, latency_ms, item)

        reranker: RerankerBackend | None = None
        for reranker_name in rerankers_:
            # `CrossEncoderReranker` is NOT process-wide-cached the way
            # `graph/embedders.py`'s sentence-transformer weights are (see
            # that module's own `_MODEL_CACHE` docstring) -- each reranker
            # in this inner loop is a fresh object. Without explicitly
            # dropping the PREVIOUS reranker before measuring THIS one's
            # baseline, PyTorch's caching allocator can reuse the
            # just-freed blocks for the new model's load, making
            # `vram_after_r - vram_before_r` net out near zero (or even go
            # negative, clamped to 0.0) even though a real ~1GB-class model
            # was loaded. Verified directly in this story's own first sweep
            # run (`bge-rerank-v2-m3` reported a 0.0 MB delta in every row)
            # and root-caused with a minimal standalone repro before this
            # fix: the first attempt at this fix (a separate `prev_reranker`
            # tracking variable, deleted while the loop's own `reranker`
            # name still held a live second reference to the same object)
            # still failed identically — Python's `for` loop does not
            # rescope its target variable per iteration, so a *second* name
            # bound to the same object keeps its refcount above zero and
            # blocks release. Using exactly ONE name (`reranker` itself,
            # `del`eted and not reassigned until the new model has already
            # been measured) is what actually drops the refcount to zero in
            # time — re-verified with the same standalone repro after this
            # rewrite (see the dev-log return note for this story for the
            # pasted before/after numbers).
            if reranker is not None:
                del reranker
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:  # pragma: no cover - torch is a hard project dependency
                    pass

            vram_before_r = _cuda_mem_mb()
            reranker = build_reranker(reranker_name)
            if reranker_name != "noop":
                reranker.rerank("warmup query", ["warmup document"])  # force the lazy load, once
            vram_after_r = _cuda_mem_mb()
            reranker_vram_mb = (
                max(0.0, vram_after_r - vram_before_r)
                if (vram_before_r is not None and vram_after_r is not None)
                else None
            )

            item_metrics: list[ItemMetrics] = []
            for (pid, qa_id), (retrieved, retrieve_latency_ms, item) in bi_stage.items():
                candidates = retrieved[:n_candidates]
                docs = [c.text for c in candidates]
                t0 = time.monotonic()
                ranked_pairs = reranker.rerank(item["question"], docs, top_k=None)
                rerank_latency_ms = (time.monotonic() - t0) * 1000.0
                reranked = [candidates[idx] for idx, _score in ranked_pairs]

                item_metrics.append(
                    evaluate_item(
                        patient_id=pid,
                        qa_id=qa_id,
                        embedder=embedder_name,
                        reranker=reranker_name,
                        bi_encoder_retrieved=retrieved,
                        reranked=reranked,
                        evidence=item["evidence"],
                        retrieve_latency_ms=retrieve_latency_ms,
                        rerank_latency_ms=rerank_latency_ms,
                    )
                )

            configs.append(
                ConfigResult(
                    embedder=embedder_name,
                    reranker=reranker_name,
                    items=item_metrics,
                    build_time_s_mean=(sum(build_times) / len(build_times)) if build_times else 0.0,
                    embedder_vram_mb=embedder_vram_mb,
                    reranker_vram_mb=reranker_vram_mb,
                    n_patients=len(patient_ids),
                )
            )

    return SweepResult(configs=configs, patient_ids=list(patient_ids), n_candidates=n_candidates)


# ---------------------------------------------------------------------------
# Aggregation.
# ---------------------------------------------------------------------------


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


@dataclass(frozen=True)
class AggregateRow:
    embedder: str
    reranker: str
    n_admission: int
    """Denominator for every `*_admission` column — always the full item
    count for this config (admission-level gold is on 100% of items)."""
    n_turn: int
    """Denominator for every `*_turn` column — the count of items that
    actually carried turn-level gold (~50%), never the full item count."""
    recall_admission: dict[int, float]
    recall_turn: dict[int, float]
    hit_admission: dict[int, float]
    hit_turn: dict[int, float]
    ndcg_admission: dict[int, float]
    ndcg_turn: dict[int, float]
    mean_retrieve_latency_ms: float
    mean_rerank_latency_ms: float
    build_time_s_mean: float
    embedder_vram_mb: float | None
    reranker_vram_mb: float | None

    @property
    def total_vram_mb(self) -> float | None:
        if self.embedder_vram_mb is None or self.reranker_vram_mb is None:
            return None
        return self.embedder_vram_mb + self.reranker_vram_mb


def aggregate_config(config: ConfigResult) -> AggregateRow:
    """Mean every per-item metric across `config.items`, admission-grain
    over the full item set and turn-grain over the (smaller) subset that
    actually carries turn-level gold — the two denominators are reported
    separately as `n_admission`/`n_turn` (never blended into one n)."""
    items = config.items
    turn_items = [it for it in items if it.recall_turn is not None]

    recall_admission = {k: _mean([it.recall_admission[k] for it in items]) for k in RECALL_KS}
    hit_admission = {k: _mean([it.hit_admission[k] for it in items]) for k in RERANK_KS}
    ndcg_admission = {k: _mean([it.ndcg_admission[k] for it in items]) for k in RERANK_KS}

    if turn_items:
        recall_turn = {k: _mean([it.recall_turn[k] for it in turn_items]) for k in RECALL_KS}  # type: ignore[index]
        hit_turn = {k: _mean([it.hit_turn[k] for it in turn_items]) for k in RERANK_KS}  # type: ignore[index]
        ndcg_turn = {k: _mean([it.ndcg_turn[k] for it in turn_items]) for k in RERANK_KS}  # type: ignore[index]
    else:
        recall_turn = {k: float("nan") for k in RECALL_KS}
        hit_turn = {k: float("nan") for k in RERANK_KS}
        ndcg_turn = {k: float("nan") for k in RERANK_KS}

    return AggregateRow(
        embedder=config.embedder,
        reranker=config.reranker,
        n_admission=len(items),
        n_turn=len(turn_items),
        recall_admission=recall_admission,
        recall_turn=recall_turn,
        hit_admission=hit_admission,
        hit_turn=hit_turn,
        ndcg_admission=ndcg_admission,
        ndcg_turn=ndcg_turn,
        mean_retrieve_latency_ms=_mean([it.retrieve_latency_ms for it in items]),
        mean_rerank_latency_ms=_mean([it.rerank_latency_ms for it in items]),
        build_time_s_mean=config.build_time_s_mean,
        embedder_vram_mb=config.embedder_vram_mb,
        reranker_vram_mb=config.reranker_vram_mb,
    )


# ---------------------------------------------------------------------------
# MDE — what this run's actual n can detect (metrics.mde, never a bare
# assertion of adequate power).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MdeTableRow:
    n: int
    correlation: float
    mde_pp: float


def mde_table(
    n: int,
    *,
    baseline_acc: float = 0.5,
    correlations: tuple[float, ...] = (0.0, 0.5, 0.9),
    power: float = 0.8,
    alpha: float = 0.05,
) -> list[MdeTableRow]:
    """`metrics.mde(n, baseline_acc, correlation).mde_pp` at the ACTUAL n
    a sweep run achieved, for a spread of assumed correlations (0.0
    uncorrelated -> 0.9 highly correlated, matching this project's own
    reference table at n=250/n=1000 quoted in module docstring).
    `baseline_acc=0.5` is the conservative (maximum-variance) choice, same
    convention `eval/report.py::minimum_detectable_effect` uses."""
    return [
        MdeTableRow(n=n, correlation=rho, mde_pp=metrics.mde(n, baseline_acc, rho, power=power, alpha=alpha).mde_pp)
        for rho in correlations
    ]


def _phi_correlation(mc: "metrics.McNemarResult") -> float:
    """Pearson/phi correlation between two systems' per-item binary
    correctness, derived directly from McNemar's own 2x2 counts —
    independently re-derived from `eval/report.py::_paired_mde`'s
    identical construction (that helper is a private module-level name,
    not exported, so it is re-derived here rather than imported)."""
    a, b, c, d = mc.a, mc.b, mc.c, mc.d
    denom = math.sqrt(max((a + b) * (c + d) * (a + c) * (b + d), 1))
    corr = ((a * d - b * c) / denom) if denom else 0.0
    return max(-1.0, min(1.0, corr))


def paired_reranker_vs_noop(
    sweep: SweepResult, *, k: int = 10, alpha: float = 0.05
) -> dict[str, dict[str, dict]]:
    """For each embedder, paired-compares every non-`"noop"` reranker
    against `"noop"` on the SAME qa_ids (every config answers the same
    queries, so paired McNemar/bootstrap is strictly more powerful than an
    unpaired test — `eval/metrics.py` module docstring, R-CTX-31) at
    `Hit@k`/`nDCG@k` (admission grain): paired McNemar on the binary
    `Hit@k` outcome, Holm-Bonferroni-adjusted across the reranker family
    within that embedder (mirrors `eval/report.py::paired_significance`'s
    exact idiom), plus a paired bootstrap test on the continuous `nDCG@k`
    delta. Each comparison also carries its own achievable MDE
    (`metrics.mde` at the paired n, correlation from `_phi_correlation`)
    and an `underpowered` flag: `True` when the observed `hit_delta`
    magnitude is smaller than what that paired n could actually detect —
    never silently reported as a normal, trustworthy comparison.

    Returns `{embedder: {reranker: {...}}}`.
    """
    hit_by_key: dict[tuple[str, str], dict[str, bool]] = {}
    ndcg_by_key: dict[tuple[str, str], dict[str, float]] = {}
    for c in sweep.configs:
        hit_by_key[(c.embedder, c.reranker)] = {it.qa_id: bool(it.hit_admission[k]) for it in c.items}
        ndcg_by_key[(c.embedder, c.reranker)] = {it.qa_id: it.ndcg_admission[k] for it in c.items}

    embedders_seen = sorted({c.embedder for c in sweep.configs})
    out: dict[str, dict[str, dict]] = {}
    for embedder in embedders_seen:
        noop_hits = hit_by_key.get((embedder, "noop"))
        if noop_hits is None:
            continue
        rerankers_for_embedder = sorted(
            {c.reranker for c in sweep.configs if c.embedder == embedder} - {"noop"}
        )
        family_p: list[float] = []
        family_names: list[str] = []
        family_results: dict[str, dict] = {}

        for reranker_name in rerankers_for_embedder:
            other_hits = hit_by_key.get((embedder, reranker_name))
            if other_hits is None:
                continue
            shared_ids = sorted(set(noop_hits) & set(other_hits))
            if not shared_ids:
                continue

            noop_correct = [noop_hits[i] for i in shared_ids]
            other_correct = [other_hits[i] for i in shared_ids]
            mc = metrics.mcnemar_test(noop_correct, other_correct)

            noop_ndcg = np.array([ndcg_by_key[(embedder, "noop")][i] for i in shared_ids])
            other_ndcg = np.array([ndcg_by_key[(embedder, reranker_name)][i] for i in shared_ids])
            pb = metrics.paired_bootstrap_test(other_ndcg, noop_ndcg, n_resamples=2000, seed=0)

            hit_delta = (sum(other_correct) - sum(noop_correct)) / len(shared_ids)
            baseline_acc = (sum(noop_correct) + sum(other_correct)) / (2 * len(shared_ids))
            paired_mde_pp = None
            if 0.0 < baseline_acc < 1.0:
                corr = _phi_correlation(mc)
                try:
                    paired_mde_pp = metrics.mde(mc.n, baseline_acc, corr).mde_pp
                except ValueError:
                    paired_mde_pp = None

            family_p.append(mc.p_value)
            family_names.append(reranker_name)
            family_results[reranker_name] = {
                "n_paired": mc.n,
                "hit_delta": hit_delta,
                "hit_p_raw": mc.p_value,
                "ndcg_delta": pb.observed_diff,
                "ndcg_p": pb.p_value,
                "paired_mde_pp": paired_mde_pp,
                "underpowered": (paired_mde_pp is not None and abs(hit_delta) < paired_mde_pp),
            }

        if family_p:
            adjusted = metrics.holm_bonferroni(family_p, alpha=alpha)
            for i, name in enumerate(family_names):
                family_results[name]["hit_p_holm"] = adjusted.adjusted_p[i]
                family_results[name]["reject_holm"] = adjusted.reject[i]

        out[embedder] = family_results
    return out


# ---------------------------------------------------------------------------
# Rendering.
# ---------------------------------------------------------------------------


def _fmt(x: float) -> str:
    return "n/a" if (x is None or math.isnan(x)) else f"{x:.3f}"


def render_markdown(sweep: SweepResult, *, grain: str) -> str:
    """One markdown table for `grain in ("admission", "turn")` — every row
    is one `(embedder, reranker)` config, its own n stated inline (never
    borrowed from the other grain's table)."""
    if grain not in ("admission", "turn"):
        raise ValueError(f"grain must be 'admission' or 'turn', got {grain!r}")

    rows = [aggregate_config(c) for c in sweep.configs]
    lines: list[str] = []
    lines.append(f"## Retrieval eval — {grain}-level grounding")
    lines.append("")
    lines.append(f"Patients: `{', '.join(sweep.patient_ids)}` · n_candidates_for_rerank=`{sweep.n_candidates}`")
    lines.append("")
    if grain == "admission":
        lines.append(f"> {RECALL_CEILING_NOTE}")
        lines.append("")
        lines.append(f"> {NDCG_ADMISSION_CAVEAT}")
        lines.append("")

    header = (
        ["embedder", "reranker", "n"]
        + [f"Recall@{k}" for k in RECALL_KS]
        + [f"Hit@{k}" for k in RERANK_KS]
        + [f"nDCG@{k}" for k in RERANK_KS]
        + ["retrieve_ms", "rerank_ms", "embed_vram_mb", "rerank_vram_mb"]
    )
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))

    for row in rows:
        n = row.n_admission if grain == "admission" else row.n_turn
        recall = row.recall_admission if grain == "admission" else row.recall_turn
        hit = row.hit_admission if grain == "admission" else row.hit_turn
        ndcg = row.ndcg_admission if grain == "admission" else row.ndcg_turn
        cells = (
            [row.embedder, row.reranker, str(n)]
            + [_fmt(recall[k]) for k in RECALL_KS]
            + [_fmt(hit[k]) for k in RERANK_KS]
            + [_fmt(ndcg[k]) for k in RERANK_KS]
            + [
                f"{row.mean_retrieve_latency_ms:.2f}",
                f"{row.mean_rerank_latency_ms:.2f}",
                ("n/a" if row.embedder_vram_mb is None else f"{row.embedder_vram_mb:.1f}"),
                ("n/a" if row.reranker_vram_mb is None else f"{row.reranker_vram_mb:.1f}"),
            ]
        )
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def render_significance_section(sweep: SweepResult, *, k: int = 10) -> str:
    comparisons = paired_reranker_vs_noop(sweep, k=k)
    lines = [f"## Paired significance — reranker vs. noop, Hit@{k}/nDCG@{k} (admission grain)", ""]
    if not comparisons:
        lines.append("(no noop baseline present in this sweep — comparison not computed)")
        return "\n".join(lines)
    header = [
        "embedder", "reranker", "n_paired", "hit_delta", "hit_p_holm", "reject_holm",
        "ndcg_delta", "ndcg_p", "paired_mde_pp", "underpowered",
    ]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for embedder, per_reranker in comparisons.items():
        for reranker_name, r in per_reranker.items():
            mde_cell = "n/a" if r["paired_mde_pp"] is None else f"{r['paired_mde_pp']:.4f}"
            cells = [
                embedder, reranker_name, str(r["n_paired"]),
                f"{r['hit_delta']:+.4f}", f"{r['hit_p_holm']:.4f}", ("yes" if r["reject_holm"] else "no"),
                f"{r['ndcg_delta']:+.4f}", f"{r['ndcg_p']:.4f}", mde_cell,
                ("yes" if r["underpowered"] else "no"),
            ]
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_mde_section(n_admission: int, n_turn: int) -> str:
    lines = ["## Minimum detectable effect at this run's actual n", ""]
    lines.append(
        "Sample-size sizing table (`metrics.mde`, alpha=0.05, power=0.80, "
        "baseline_acc=0.5 conservative-variance) at the n THIS run actually "
        "achieved — never the aspirational 250/1000 reference points."
    )
    lines.append("")
    lines.append("| grain | n | correlation | detectable effect (pp) |")
    lines.append("|---|---|---|---|")
    for label, n in (("admission", n_admission), ("turn", n_turn)):
        if n <= 0:
            lines.append(f"| {label} | {n} | n/a | n/a (no groundable items) |")
            continue
        for row in mde_table(n):
            lines.append(f"| {label} | {row.n} | {row.correlation:.2f} | {row.mde_pp * 100:.2f} |")
    return "\n".join(lines)


def render_terminal_summary(sweep: SweepResult) -> str:
    """Short terminal-friendly summary — full detail is in the markdown
    files `write_report` saves to `results/`."""
    lines = ["Retrieval eval sweep summary", "=" * 29]
    for grain in ("admission", "turn"):
        lines.append("")
        lines.append(render_markdown(sweep, grain=grain))
    lines.append("")
    n_admission = sweep.configs[0].n_patients if sweep.configs else 0
    rows_adm = [aggregate_config(c) for c in sweep.configs]
    n_adm = rows_adm[0].n_admission if rows_adm else 0
    n_trn = rows_adm[0].n_turn if rows_adm else 0
    lines.append("")
    lines.append(render_mde_section(n_adm, n_trn))
    lines.append("")
    lines.append(render_significance_section(sweep))
    return "\n".join(lines)


def write_report(sweep: SweepResult, *, out_dir: Path = Path("results")) -> dict[str, Path]:
    """Writes one markdown file per granularity plus a combined summary
    (with the MDE table and paired-significance section) to `out_dir`
    (default `results/`, gitignored — curated artifacts get `git add -f`
    at freeze, per this project's convention). Returns the written paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    paths: dict[str, Path] = {}
    for grain in ("admission", "turn"):
        text = render_markdown(sweep, grain=grain)
        path = out_dir / f"retrieval_eval__{grain}__{ts}.md"
        path.write_text(text, encoding="utf-8")
        paths[grain] = path

    rows = [aggregate_config(c) for c in sweep.configs]
    n_adm = rows[0].n_admission if rows else 0
    n_trn = rows[0].n_turn if rows else 0
    summary_text = "\n\n".join(
        [
            "# Retrieval eval — full sweep report",
            f"Generated {ts} · patients=`{', '.join(sweep.patient_ids)}`",
            render_markdown(sweep, grain="admission"),
            render_markdown(sweep, grain="turn"),
            render_mde_section(n_adm, n_trn),
            render_significance_section(sweep),
        ]
    )
    summary_path = out_dir / f"retrieval_eval__summary__{ts}.md"
    summary_path.write_text(summary_text, encoding="utf-8")
    paths["summary"] = summary_path
    return paths
