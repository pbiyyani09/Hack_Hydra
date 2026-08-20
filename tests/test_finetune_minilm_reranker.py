"""Offline trainer guards (FT-E2-S1). Does not fit the 23M model."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import finetune_minilm_reranker as ft


def test_student_is_minilm_l6():
    src = Path("scripts/finetune_minilm_reranker.py").read_text(encoding="utf-8")
    assert 'cross-encoder/ms-marco-MiniLM-L-6-v2' in src
    assert "bge-m3" not in src
    assert "TinyBERT" not in src
    assert "qwen3-rerank" not in src
    assert "teacher" not in src.lower() or "No teacher" in src or "no teacher" in src
    assert ft.STUDENT_HF_ID == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert ft.STUDENT_PARAMS == 22_714_113


def test_refuse_eval_trio_before_train(tmp_path, monkeypatch):
    rows = [{"subject_id": "10056223", "question": "q", "passage": "p", "label": 1, "kind": "gold"}]
    with pytest.raises(SystemExit, match="eval-trio"):
        ft.refuse_eval_trio(rows, [])


def test_load_pairs_and_eval_scan(tmp_path):
    path = tmp_path / "pairs.jsonl"
    path.write_text(
        json.dumps({"subject_id": "train01", "kind": "gold", "label": 1}) + "\n"
        + json.dumps({"subject_id": "10213338", "kind": "hardneg", "label": 0}) + "\n",
        encoding="utf-8",
    )
    rows = ft.load_pairs(path)
    assert ft.eval_trio_subjects_in(rows) == ["10213338"]


def test_choose_device_yields_when_other_pid(monkeypatch):
    monkeypatch.setattr(ft, "other_gpu_compute_pids", lambda self_pid=None: [99999])
    device, reason = ft.choose_device()
    assert device == "cpu"
    assert "yielded" in reason


def test_rebalance_keeps_turn_golds_and_caps_admission():
    rows = []
    for i in range(40):
        rows.append({
            "subject_id": "t1", "qa_id": "q_adm", "kind": "gold",
            "grain": "admission_expanded", "question": "q", "passage": f"p{i}", "label": 1,
        })
    rows.append({
        "subject_id": "t1", "qa_id": "q_turn", "kind": "gold",
        "grain": "turn", "question": "q", "passage": "gold-turn", "label": 1,
    })
    rows.append({
        "subject_id": "t1", "qa_id": "q_turn", "kind": "hardneg",
        "question": "q", "passage": "neg", "label": 0,
    })
    kept, stats = ft.rebalance_pairs(rows, max_adm_gold_per_qa=8, seed=20260817)
    assert stats["out_gold_turn"] == 1
    assert stats["out_gold_adm"] == 8
    assert stats["out_hardneg"] == 1
    assert any(r["passage"] == "gold-turn" for r in kept)
