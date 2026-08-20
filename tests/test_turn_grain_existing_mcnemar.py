"""Offline pairing tests for FT-E3-S1. No models."""

from __future__ import annotations

from medmemgraph.eval.existing_mcnemar import pair_turn_hit10


def _row(emb: str, rer: str, pid: str, qid: str, hit10: float | None) -> dict:
    hit_turn = None if hit10 is None else {"2": 0.0, "5": 0.0, "10": hit10, "20": hit10}
    return {
        "embedder": emb,
        "reranker": rer,
        "patient_id": pid,
        "qa_id": qid,
        "hit_admission": {"10": 1.0},
        "hit_turn": hit_turn,
    }


def test_pairs_on_patient_qa_and_drops_missing_turn():
    gpu_e, gpu_r = "qwen3-0.6b", "qwen3-rerank-0.6b"
    cpu_e, cpu_r = "arctic-s", "ms-marco-minilm-l6-v2-onnx-int8"
    rows = [
        _row(gpu_e, gpu_r, "p1", "q1", 1.0),
        _row(cpu_e, cpu_r, "p1", "q1", 0.0),
        _row(gpu_e, gpu_r, "p1", "q2", None),  # drop
        _row(cpu_e, cpu_r, "p1", "q2", 1.0),
        _row(gpu_e, gpu_r, "p2", "q3", 0.0),
        _row(cpu_e, cpu_r, "p2", "q3", 1.0),
        _row("other", "noop", "p1", "q1", 1.0),
    ]
    gpu, cpu = pair_turn_hit10(rows)
    assert len(gpu) == len(cpu) == 2
    assert gpu == [True, False]
    assert cpu == [False, True]


def test_script_does_not_import_models():
    from pathlib import Path

    src = Path("scripts/turn_grain_existing_mcnemar.py").read_text(encoding="utf-8")
    assert "import run_sweep" not in src
    assert "from medmemgraph.eval.retrieval_eval" not in src
    assert "medmemgraph.graph.reranker" not in src
    assert "medmemgraph.graph.embedders" not in src
