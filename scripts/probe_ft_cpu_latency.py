"""2-thread CPU latency on real clinical text (FT-E2-S2)."""

from __future__ import annotations

import time
from pathlib import Path

import torch

from medmemgraph.graph.reranker import CrossEncoderReranker
from medmemgraph.pipeline.loader import load_conversation

PATIENT = "10056223"
N_CAND = 100
N_MEASURED = 7
OUT = Path("results/finetune-reranker/latency_probe.md")
BASELINE_RERANK_MS = 365.3
BASELINE_E2E_MS = 421.6
ARCTIC_ENCODE_MS = 54.96
ARCTIC_SEARCH_MS = 1.322
FT_KEY = "ms-marco-minilm-l6-v2-ft-medlocomo-onnx-int8"


def main() -> int:
    torch.set_num_threads(2)
    convo = load_conversation(PATIENT)
    turns = convo.turns()
    docs = [
        f"[admission {t.hadm_id} turn {t.turn_number} · {t.time} · {t.speaker}] {t.text}"
        for t in turns[:N_CAND]
    ]
    if len(docs) < N_CAND:
        docs = (docs * ((N_CAND // max(len(docs), 1)) + 1))[:N_CAND]
    questions = [
        "What medications was the patient taking at discharge?",
        "Why was the patient admitted?",
        "Did the patient report chest pain?",
        "What was the discharge disposition?",
        "Were there any medication changes across admissions?",
        "What comorbidities are documented?",
        "What was the most recent lab abnormality?",
        "Was anticoagulation discussed?",
    ]
    reranker = CrossEncoderReranker(FT_KEY)
    # untimed warmup
    reranker.rerank(questions[0], docs)
    measured: list[float] = []
    for q in questions[1 : 1 + N_MEASURED]:
        t0 = time.perf_counter()
        reranker.rerank(q, docs)
        measured.append((time.perf_counter() - t0) * 1000.0)
    mean_ms = sum(measured) / len(measured)
    e2e = mean_ms + ARCTIC_ENCODE_MS + ARCTIC_SEARCH_MS
    ratio = mean_ms / BASELINE_RERANK_MS
    failed_cpu = mean_ms > 3 * BASELINE_RERANK_MS and mean_ms >= 1000.0
    note = f"""# FT MiniLM ONNX int8 — 2-thread CPU latency

- device: CPU, `torch.set_num_threads(2)`
- text: real clinical dialogue, patient `{PATIENT}` via `load_conversation` only
- pool: {N_CAND} candidates; 1 untimed warmup + {N_MEASURED} measured queries
- ONNX Runtime providers on this box are CPU-class only (no CUDA claimed)
- mean rerank_ms: **{mean_ms:.1f}** (samples: {", ".join(f"{x:.1f}" for x in measured)})
- projected e2e (arctic-s encode+search {ARCTIC_ENCODE_MS}+{ARCTIC_SEARCH_MS} ms, not re-measured): **{e2e:.1f} ms**
- comparison: un-finetuned MiniLM-int8 rerank **{BASELINE_RERANK_MS} ms** / e2e **{BASELINE_E2E_MS} ms**
- ratio vs 365.3 ms: {ratio:.2f}×
- **±2–3× caveat:** the ablation's real-query CPU timings never reconciled with the 0.124 s synthetic microbenchmark; load-average swing 9.88 → 1.82 was also unisolated. Treat this number as same-protocol, not absolute.

{"**CPU-product requirement failed:** int8 is slower than ~3× 365.3 ms and no longer sub-second." if failed_cpu else "Still in the sub-second class under this protocol (or within 3× of 365.3 ms)."}
"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(note, encoding="utf-8")
    print(note)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
