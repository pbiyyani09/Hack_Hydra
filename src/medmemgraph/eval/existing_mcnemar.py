"""Pair already-measured sweep rows for the FT-E3-S1 disk-only McNemar.

No models are loaded. Rows with ``hit_turn is None`` are dropped, never
zero-filled. Join key is ``(patient_id, qa_id)``.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

from medmemgraph.eval import metrics

GPU_BEST = ("qwen3-0.6b", "qwen3-rerank-0.6b")
CPU_BEST = ("arctic-s", "ms-marco-minilm-l6-v2-onnx-int8")


def _is_config(row: Mapping[str, Any], embedder: str, reranker: str) -> bool:
    return row.get("embedder") == embedder and row.get("reranker") == reranker


def pair_turn_hit10(
    rows: Sequence[Mapping[str, Any]],
    *,
    gpu: tuple[str, str] = GPU_BEST,
    cpu: tuple[str, str] = CPU_BEST,
) -> tuple[list[bool], list[bool]]:
    """Return paired GPU/CPU turn-grain Hit@10 booleans.

    Items whose ``hit_turn`` is ``None`` on either side are dropped from
    *both* vectors. Missing turn-grain rows are never treated as misses.
    """
    gpu_by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    cpu_by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (str(row["patient_id"]), str(row["qa_id"]))
        if _is_config(row, *gpu):
            gpu_by_key[key] = row
        elif _is_config(row, *cpu):
            cpu_by_key[key] = row

    gpu_correct: list[bool] = []
    cpu_correct: list[bool] = []
    for key in sorted(set(gpu_by_key) & set(cpu_by_key)):
        g_hit = gpu_by_key[key].get("hit_turn")
        c_hit = cpu_by_key[key].get("hit_turn")
        if g_hit is None or c_hit is None:
            continue
        gpu_correct.append(bool(g_hit["10"]))
        cpu_correct.append(bool(c_hit["10"]))
    return gpu_correct, cpu_correct


def phi_from_2x2(mc: metrics.McNemarResult) -> float:
    """Same construction as ``retrieval_eval._phi_correlation``."""
    a, b, c, d = mc.a, mc.b, mc.c, mc.d
    denom = math.sqrt(max((a + b) * (c + d) * (a + c) * (b + d), 1))
    corr = ((a * d - b * c) / denom) if denom else 0.0
    return max(-1.0, min(1.0, corr))


def summarize_turn_gap(
    gpu_correct: Sequence[bool], cpu_correct: Sequence[bool]
) -> dict[str, Any]:
    """McNemar + Holm + MDE. ``correct_a`` = GPU, ``correct_b`` = CPU."""
    mc = metrics.mcnemar_test(gpu_correct, cpu_correct)
    holm = metrics.holm_bonferroni([mc.p_value])
    n = mc.n
    gpu_hit = sum(gpu_correct) / n
    cpu_hit = sum(cpu_correct) / n
    delta = cpu_hit - gpu_hit
    phi = phi_from_2x2(mc)
    baseline = (gpu_hit + cpu_hit) / 2.0
    paired_mde = metrics.mde(n=n, baseline_acc=baseline, correlation=phi)
    ref_mdes = {
        rho: metrics.mde(n=n, baseline_acc=baseline, correlation=rho).mde_pp
        for rho in (0.0, 0.5, 0.9)
    }
    # mde_pp is in probability units (0-1) per metrics.MDEResult.
    underpowered = abs(delta) < paired_mde.mde_pp
    return {
        "n_paired": n,
        "gpu_hit": gpu_hit,
        "cpu_hit": cpu_hit,
        "delta": delta,
        "mc": mc,
        "holm_p": holm.adjusted_p[0],
        "reject": holm.reject[0],
        "phi": phi,
        "mde_pp": paired_mde.mde_pp,
        "ref_mdes": ref_mdes,
        "underpowered": underpowered,
    }


def render_note(summary: Mapping[str, Any]) -> str:
    n = summary["n_paired"]
    n_line = (
        f"n_paired = {n} (handoff claimed 231; using the JSON)."
        if n != 231
        else "n_paired = 231 (matches the handoff)."
    )
    delta_pp = summary["delta"] * 100.0
    gpu_pp = summary["gpu_hit"] * 100.0
    cpu_pp = summary["cpu_hit"] * 100.0
    mde_pp_display = summary["mde_pp"] * 100.0
    ref = summary["ref_mdes"]
    reject = "yes" if summary["reject"] else "no"
    under = "yes" if summary["underpowered"] else "no"
    return f"""# Existing turn-grain Hit@10 gap — McNemar (disk only)

This is the *current* gap, not a close-claim. No model was loaded.

## Turn grain (the target)

{n_line}

| System | Hit@10 turn |
|---|---|
| GPU `qwen3-0.6b` + `qwen3-rerank-0.6b` | {summary["gpu_hit"]:.3f} ({gpu_pp:.1f}%) |
| CPU `arctic-s` + `ms-marco-minilm-l6-v2-onnx-int8` | {summary["cpu_hit"]:.3f} ({cpu_pp:.1f}%) |
| CPU − GPU | {summary["delta"]:+.4f} ({delta_pp:+.1f} pp) |

- McNemar p = {summary["mc"].p_value:.4g} (exact={summary["mc"].exact}; b={summary["mc"].b} GPU-hit/CPU-miss, c={summary["mc"].c} GPU-miss/CPU-hit)
- Holm-adjusted p = {summary["holm_p"]:.4g} (family of 1)
- Reject H0 at α=0.05: **{reject}**
- φ (paired correctness) = {summary["phi"]:.3f}
- Paired MDE = {mde_pp_display:.2f} pp (probability units {summary["mde_pp"]:.4f})
- Underpowered (`|Δ|` < paired MDE): **{under}**
- Reference MDEs at this n (ρ=0 / 0.5 / 0.9): {ref[0.0]*100:.2f} / {ref[0.5]*100:.2f} / {ref[0.9]*100:.2f} pp

## Admission grain — **not the target** (quoted, not recomputed)

> Hit@10 admission (n=462) | 0.926 | −2.8pp (`hit_delta=-0.0281`, Holm
> `p=0.259`, **not** significant, underpowered at this n)
>
> Source: `results/cpu_ablation_report.md` line 31 and paired table row
> `arctic-s | ms-marco-minilm-l6-v2-onnx-int8 | 462 | -0.0281 | 0.2586 | no`.
"""


def compute_from_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    gpu, cpu = pair_turn_hit10(list(rows))
    return summarize_turn_gap(gpu, cpu)
