"""nDCG sweep: hybrid first stage + (baseline | listwise) rerankers.

Two stacks, never crossed:
  GPU: qwen3-0.6b ± BM25, qwen3-rerank (Hub or local FT)
  CPU: arctic-s ± BM25, MiniLM ONNX (old or listwise)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from medmemgraph.eval.hybrid_pool import build_indexes, hybrid_search
from medmemgraph.eval.metrics import ndcg_at_k
from medmemgraph.eval.retrieval_eval import (
    RERANK_KS,
    admission_only_evidence,
    build_reranker,
    turn_only_evidence,
)
from medmemgraph.eval import metrics as M
from medmemgraph.pipeline.loader import load_conversation, load_qa

TRIO = ["10056223", "10213338", "10312715"]
OUT = Path("results/finetune-reranker/ndcg_improved.md")
JSON_OUT = Path("results/finetune-reranker/ndcg_improved.json")
OUT_CPU = Path("results/finetune-reranker/ndcg_improved_cpu.md")
JSON_CPU = Path("results/finetune-reranker/ndcg_improved_cpu.json")
OUT_GPU = Path("results/finetune-reranker/ndcg_improved_gpu.md")
JSON_GPU = Path("results/finetune-reranker/ndcg_improved_gpu.json")


def _agg(pairs: list[dict]) -> dict:
    n = len(pairs)
    out = {"n": n}
    for k in RERANK_KS:
        out[f"hit@{k}"] = sum(p["hit"][k] for p in pairs) / n
        out[f"ndcg@{k}"] = sum(p["ndcg"][k] for p in pairs) / n
        if out[f"ndcg@{k}"] > 1.0 + 1e-6:
            raise RuntimeError(f"nDCG@{k} > 1: {out[f'ndcg@{k}']}")
    return out


def run_arm(embedder: str, reranker_name: str, *, hybrid: bool) -> dict:
    indexes = {}
    for pid in TRIO:
        convo = load_conversation(pid)
        if hybrid:
            indexes[pid] = build_indexes(pid, convo, dense_backend=embedder)
        else:
            from medmemgraph.graph.vector_index import PatientIndex

            idx = PatientIndex(pid, backend=embedder, cache_path=None)
            idx.build(convo)
            indexes[pid] = idx
    reranker = build_reranker(reranker_name)
    if reranker_name != "noop":
        reranker.rerank("warmup", ["warmup"])

    adm_rows: list[dict] = []
    turn_rows: list[dict] = []
    for pid in TRIO:
        for item in load_qa(pid):
            ev = item.get("evidence") or {}
            if not ev.get("admissions"):
                continue
            if hybrid:
                dense, lex = indexes[pid]
                cands = hybrid_search(item["question"], dense, lex, k=100)
            else:
                cands = indexes[pid].search(item["question"], k=100)
            ranked = reranker.rerank(item["question"], [c.text for c in cands], top_k=None)
            reranked = [cands[i] for i, _ in ranked]
            adm_ev = admission_only_evidence(ev)
            adm_rows.append(
                {
                    "hit": {k: M.hit_at_k(reranked, adm_ev, k) for k in RERANK_KS},
                    "ndcg": {k: ndcg_at_k(reranked, adm_ev, k) for k in RERANK_KS},
                }
            )
            trn = turn_only_evidence(ev)
            if trn is not None:
                turn_rows.append(
                    {
                        "hit": {k: M.hit_at_k(reranked, trn, k) for k in RERANK_KS},
                        "ndcg": {k: ndcg_at_k(reranked, trn, k) for k in RERANK_KS},
                    }
                )
    return {
        "embedder": embedder,
        "reranker": reranker_name,
        "hybrid": hybrid,
        "admission": _agg(adm_rows),
        "turn": _agg(turn_rows),
    }


def _line(arm: dict, grain: str) -> str:
    a = arm[grain]
    cells = [
        f"{arm['embedder']}+{'hyb' if arm['hybrid'] else 'vec'}",
        arm["reranker"].replace("ms-marco-minilm-l6-v2-", "m-"),
        str(a["n"]),
    ]
    for k in RERANK_KS:
        cells.append(f"{a[f'hit@{k}']:.3f}")
    for k in RERANK_KS:
        cells.append(f"{a[f'ndcg@{k}']:.3f}")
    return "| " + " | ".join(cells) + " |"


def _ckpt_exists(key: str) -> bool:
    from medmemgraph.graph.reranker import REGISTERED_MODELS

    spec = REGISTERED_MODELS.get(key)
    if spec is None:
        return False
    root = Path(spec.hf_id)
    if spec.backend == "onnx":
        return (root / (spec.onnx_file or "onnx/model_qint8_avx512.onnx")).is_file()
    return (root / "config.json").is_file()


def main() -> int:
    qwen_ft = "qwen3-rerank-0.6b-ft-listwise"
    mini_lw = "ms-marco-minilm-l6-v2-ft-listwise-onnx-int8"
    mini_orpo = "ms-marco-minilm-l6-v2-ft-orpo-onnx-int8"

    cpu_only = "--cpu-only" in sys.argv
    gpu_only = "--gpu-only" in sys.argv
    plan: list[tuple[str, str, bool, str]] = []
    if not cpu_only:
        plan.extend(
            [
                ("qwen3-0.6b", "noop", False, "GPU first-stage dense"),
                ("qwen3-0.6b", "noop", True, "GPU first-stage hybrid"),
                ("qwen3-0.6b", "qwen3-rerank-0.6b", False, "GPU Hub Qwen dense"),
                ("qwen3-0.6b", "qwen3-rerank-0.6b", True, "GPU Hub Qwen hybrid"),
            ]
        )
        if _ckpt_exists(qwen_ft):
            plan.append(("qwen3-0.6b", qwen_ft, False, "GPU Qwen-FT dense"))
            plan.append(("qwen3-0.6b", qwen_ft, True, "GPU Qwen-FT hybrid"))
        else:
            print(f"skip {qwen_ft}: no checkpoint", flush=True)
    if not gpu_only:
        plan.extend(
            [
                ("arctic-s", "noop", False, "CPU first-stage dense"),
                ("arctic-s", "noop", True, "CPU first-stage hybrid"),
                ("arctic-s", mini_orpo, False, "CPU MiniLM-ORPO dense"),
                ("arctic-s", mini_orpo, True, "CPU MiniLM-ORPO hybrid"),
            ]
        )
        if _ckpt_exists(mini_lw):
            plan.append(("arctic-s", mini_lw, False, "CPU MiniLM-listwise dense"))
            plan.append(("arctic-s", mini_lw, True, "CPU MiniLM-listwise hybrid"))
        else:
            print(f"skip {mini_lw}: no ONNX export", flush=True)

    arms = []
    for embedder, reranker, hybrid, label in plan:
        print(f"{label}…", flush=True)
        arms.append(run_arm(embedder, reranker, hybrid=hybrid))

    hdr = (
        "| stack | reranker | n | Hit@2 | Hit@5 | Hit@10 | Hit@20 | "
        "nDCG@2 | nDCG@5 | nDCG@10 | nDCG@20 |"
    )
    sep = "|---|---|---|---|---|---|---|---|---|---|---|"
    body = f"""# Improved nDCG sweep (hybrid + listwise)

Bar: nDCG@10 ≥ 0.8. First-stage turn recall and |Rel| still cap this;
say so if the bar is missed.

## Turn

{hdr}
{sep}
{chr(10).join(_line(a, "turn") for a in arms)}

## Admission

{hdr}
{sep}
{chr(10).join(_line(a, "admission") for a in arms)}
"""
    out_md, out_json = OUT, JSON_OUT
    if cpu_only:
        out_md, out_json = OUT_CPU, JSON_CPU
    elif gpu_only:
        out_md, out_json = OUT_GPU, JSON_GPU
    out_md.write_text(body, encoding="utf-8")
    out_json.write_text(json.dumps(arms, indent=2) + "\n", encoding="utf-8")
    print(body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
