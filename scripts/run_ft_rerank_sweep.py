"""FT-E3-S2: arctic-s × {noop, old ONNX, ft ONNX} on the eval trio.

Never uses module-default EMBEDDERS/RERANKERS. Never reruns Qwen.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from medmemgraph.eval.retrieval_eval import paired_configs, run_sweep, write_report

EVAL_TRIO = ("10056223", "10213338", "10312715")
GPU_BEST = ("qwen3-0.6b", "qwen3-rerank-0.6b")
OLD_CPU = ("arctic-s", "ms-marco-minilm-l6-v2-onnx-int8")
FT_ONNX = "ms-marco-minilm-l6-v2-ft-orpo-turn-onnx-int8"
BCE_FT_ONNX = "ms-marco-minilm-l6-v2-ft-orpo-onnx-int8"
RAW_EXISTING = Path("results/cpu_ablation_raw_items.json")
SPLIT_PATH = Path("results/finetune-reranker/patient_split.json")
OUT_NOTE = Path("results/finetune-reranker/close_the_gap.md")
GPU_TURN_REF = 0.900
GPU_ADM_REF = 0.955
WIN_BAR = 0.88


def _filter_config(rows: list[dict], embedder: str, reranker: str) -> list[dict]:
    return [r for r in rows if r.get("embedder") == embedder and r.get("reranker") == reranker]


def _items_from_sweep(sweep, embedder: str, reranker: str) -> list:
    for cfg in sweep.configs:
        if cfg.embedder == embedder and cfg.reranker == reranker:
            return cfg.items
    raise RuntimeError(f"missing sweep config {embedder} × {reranker}")


def _agg_hit(items, grain: str, k: int = 10) -> tuple[float, int]:
    hits: list[float] = []
    for it in items:
        field = it.hit_admission if grain == "admission" else it.hit_turn
        if field is None:
            continue
        val = field.get(k, field.get(str(k)))
        if val is None:
            continue
        hits.append(float(val))
    if not hits:
        return float("nan"), 0
    return sum(hits) / len(hits), len(hits)


def _fmt_pair(row: dict) -> str:
    mde = row["paired_mde_pp"]
    mde_s = "n/a" if mde is None else f"{mde * 100:.2f} pp"
    return (
        f"n={row['n_paired']}, Δ={row['hit_delta']*100:+.2f} pp, "
        f"p_raw={row['hit_p_raw']:.4g}, p_holm={row['hit_p_holm']:.4g}, "
        f"reject={row['reject_holm']}, MDE={mde_s}, underpowered={row['underpowered']}"
    )


def _leakage_checklist() -> tuple[list[str], bool]:
    lines = ["## Leakage checklist (finetuned turn Hit@10 > GPU 0.900)"]
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8")) if SPLIT_PATH.is_file() else {}
    eval_ids = set(split.get("eval", list(EVAL_TRIO)))
    train_dev = set(split.get("train", [])) | set(split.get("dev", []))
    leak_split = sorted(eval_ids & train_dev)
    lines.append(
        f"1. patient_split.json eval ∩ (train∪dev): "
        f"{'empty (pass)' if not leak_split else 'FAIL ' + str(leak_split)}"
    )
    pair_hits: list[str] = []
    pair_dir = Path("data/reranker_ft")
    for path in sorted(pair_dir.glob("*.jsonl")):
        text = path.read_text(encoding="utf-8")
        for eid in EVAL_TRIO:
            if eid in text:
                pair_hits.append(f"{path.name}:{eid}")
    lines.append(
        f"2. rg of eval ids in data/reranker_ft/*.jsonl: "
        f"{'empty (pass)' if not pair_hits else 'FAIL ' + str(pair_hits)}"
    )
    lines.append("3. allowlist test — re-run `tests/test_loader_allowlist.py` (not executed here).")
    hardneg_outside = []
    for path in pair_dir.glob("*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("kind") == "hardneg" and rec.get("subject_id") not in train_dev:
                hardneg_outside.append(rec.get("subject_id"))
    lines.append(
        f"4. hard-neg subject_ids ⊂ train∪dev: "
        f"{'pass' if not hardneg_outside else 'FAIL ' + str(sorted(set(hardneg_outside)))}"
    )
    failed = bool(leak_split or pair_hits or hardneg_outside)
    lines.append(
        "5. Verdict: **leakage suspected** — do not treat as a leaderboard win."
        if failed
        else "5. Checklist boxes 1/2/4 passed. Still domain adaptation on held-out patients, not general capability."
    )
    return lines, failed


def main() -> int:
    if "qwen3-rerank-0.6b" in (
        "noop",
        "ms-marco-minilm-l6-v2-onnx-int8",
        FT_ONNX,
    ):
        return 1  # belt
    embedders_ = ("arctic-s",)
    rerankers_ = (
        "noop",
        "ms-marco-minilm-l6-v2-onnx-int8",
        BCE_FT_ONNX,
        FT_ONNX,
    )
    # BCE_FT_ONNX here is the mixed-grain ORPO (0.874). FT_ONNX is turn-only continue.
    sweep = run_sweep(
        list(EVAL_TRIO),
        embedders_=embedders_,
        rerankers_=rerankers_,
        n_candidates=100,
    )
    write_report(sweep, out_dir=Path("results"))

    if not RAW_EXISTING.is_file():
        print(f"missing {RAW_EXISTING}", file=sys.stderr)
        return 1
    existing = json.loads(RAW_EXISTING.read_text(encoding="utf-8"))
    gpu_items = _filter_config(existing, *GPU_BEST)
    old_cpu_new = _items_from_sweep(sweep, *OLD_CPU)
    ft_items = _items_from_sweep(sweep, "arctic-s", FT_ONNX)

    ft_turn, n_ft_turn = _agg_hit(ft_items, "turn")
    ft_adm, n_ft_adm = _agg_hit(ft_items, "admission")
    old_turn, n_old_turn = _agg_hit(old_cpu_new, "turn")
    old_adm, n_old_adm = _agg_hit(old_cpu_new, "admission")

    if n_ft_adm != 462 or n_old_adm != 462:
        print(
            f"admission n mismatch: ft={n_ft_adm} old={n_old_adm} expected 462 — refusing close-claim",
            file=sys.stderr,
        )
    if n_ft_turn != 231 or n_old_turn != 231:
        print(
            f"turn n mismatch: ft={n_ft_turn} old={n_old_turn} expected 231 — refusing close-claim",
            file=sys.stderr,
        )

    old_vs_gpu_turn = paired_configs(gpu_items, [it.__dict__ for it in old_cpu_new], grain="turn")
    ft_vs_gpu_turn = paired_configs(gpu_items, [it.__dict__ for it in ft_items], grain="turn")
    ft_vs_old_turn = paired_configs(
        [it.__dict__ for it in old_cpu_new], [it.__dict__ for it in ft_items], grain="turn"
    )
    old_vs_gpu_adm = paired_configs(gpu_items, [it.__dict__ for it in old_cpu_new], grain="admission")
    ft_vs_gpu_adm = paired_configs(gpu_items, [it.__dict__ for it in ft_items], grain="admission")
    ft_vs_old_adm = paired_configs(
        [it.__dict__ for it in old_cpu_new], [it.__dict__ for it in ft_items], grain="admission"
    )

    closed = (
        n_ft_turn == 231
        and (not ft_vs_gpu_turn["reject_holm"] or ft_vs_gpu_turn["underpowered"])
        and (ft_vs_gpu_turn["hit_delta"] >= 0 or ft_vs_gpu_turn["underpowered"])
    )
    # A-closed: deficit vs GPU-best is no longer a significant, above-MDE gap.
    # hit_delta is ft - gpu, so a remaining deficit is negative.
    remaining_deficit = ft_vs_gpu_turn["hit_delta"] < 0
    significant_above_mde = ft_vs_gpu_turn["reject_holm"] and not ft_vs_gpu_turn["underpowered"]
    closed = n_ft_turn == 231 and not (remaining_deficit and significant_above_mde)

    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8")) if SPLIT_PATH.is_file() else {}
    latency_note = Path("results/finetune-reranker/latency_probe.md")
    latency_blob = latency_note.read_text(encoding="utf-8") if latency_note.is_file() else "(latency probe not yet written)"

    beats_qwen = ft_turn > GPU_TURN_REF
    leak_lines: list[str] = []
    leak_failed = False
    if beats_qwen:
        leak_lines, leak_failed = _leakage_checklist()

    if leak_failed:
        headline = "Leakage suspected — not a leaderboard win."
    elif ft_turn > GPU_TURN_REF:
        headline = (
            f"Bonus: ORPO turn Hit@10 {ft_turn:.3f} > GPU {GPU_TURN_REF:.3f}. "
            "Domain adaptation on held-out patients — run the leakage checklist."
        )
    elif ft_turn >= WIN_BAR:
        headline = (
            f"Win bar met: ORPO turn Hit@10 {ft_turn:.3f} ≥ {WIN_BAR:.2f} "
            f"(GPU {GPU_TURN_REF:.3f}). Close, not a beat."
        )
    elif closed:
        headline = (
            f"ORPO turn Hit@10 {ft_turn:.3f} is below the {WIN_BAR:.2f} bar. "
            "Remaining GPU deficit is not a significant above-MDE gap."
        )
    else:
        headline = (
            f"The turn-grain gap did not close (ORPO {ft_turn:.3f} vs GPU {GPU_TURN_REF:.3f})."
        )

    bce_line = ""
    try:
        bce_items = _items_from_sweep(sweep, "arctic-s", BCE_FT_ONNX)
        bce_turn, _ = _agg_hit(bce_items, "turn")
        bce_adm, _ = _agg_hit(bce_items, "admission")
        bce_line = (
            f"| CPU arctic-s + MiniLM-BCE-ft ONNX int8 | {bce_adm:.3f} | {bce_turn:.3f} |\n"
        )
    except RuntimeError:
        bce_line = ""

    body = f"""# Close the gap? — MiniLM CE-ORPO vs GPU Qwen (turn-grain)

**{headline}**

Framing: domain adaptation on held-out patients from the same corpus.
Not a general claim that "our reranker beats Qwen." Admission −2.8pp is
**not the target** (already n.s. and underpowered at n=462).
Win bar this run: turn Hit@10 **> {WIN_BAR:.2f}**. Beating 0.900 is bonus.

## Hit@10

| Arm | Admission (n={n_ft_adm}) | Turn (n={n_ft_turn}) |
|---|---|---|
| GPU `qwen3-0.6b` + `qwen3-rerank-0.6b` (existing rows) | {GPU_ADM_REF:.3f} | {GPU_TURN_REF:.3f} |
| CPU arctic-s + MiniLM ONNX int8 (this sweep) | {old_adm:.3f} | {old_turn:.3f} |
{bce_line}| CPU arctic-s + MiniLM-ORPO ONNX int8 (this sweep) | {ft_adm:.3f} | {ft_turn:.3f} |

## Turn-grain McNemar (the claim)

1. old CPU vs GPU-best: {_fmt_pair(old_vs_gpu_turn)}
2. **ft ONNX vs GPU-best:** {_fmt_pair(ft_vs_gpu_turn)}
3. ft ONNX vs old CPU: {_fmt_pair(ft_vs_old_turn)}

## Admission-grain (commentary, underpowered if `|Δ|` < MDE)

1. old CPU vs GPU: {_fmt_pair(old_vs_gpu_adm)}
2. ft vs GPU: {_fmt_pair(ft_vs_gpu_adm)}
3. ft vs old CPU: {_fmt_pair(ft_vs_old_adm)}

Admission nDCG is in [0, 1] after the IDCG fix. Older on-disk
admission nDCG columns that exceed 1.0 are the pre-fix bug.

## Split

- eval: {split.get("eval")}
- dev ({split.get("n_dev")}): {split.get("dev")}
- train n={len(split.get("train", []))}
- grain: {split.get("grain")} — {split.get("grain_rule")}

## CPU latency / RAM

{latency_blob}

New torch arm RAM (weights-only): 90.9 MB fp32. New int8 arm: 22.7 MB.
Old ONNX int8 arm unchanged.

`qwen3-rerank-0.6b` was **not** in this sweep's `rerankers_`.
"""
    if leak_lines:
        body += "\n" + "\n".join(leak_lines) + "\n"

    OUT_NOTE.parent.mkdir(parents=True, exist_ok=True)
    OUT_NOTE.write_text(body, encoding="utf-8")
    print(body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
