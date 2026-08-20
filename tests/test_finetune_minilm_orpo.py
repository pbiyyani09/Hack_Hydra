"""Offline CE-ORPO tests. Does not fit MiniLM."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from medmemgraph.eval.rerank_orpo import (
    Triple,
    build_triples_from_rows,
    orpo_loss,
)
from medmemgraph.eval.rerank_split import EVAL_TRIO


def test_orpo_loss_decreases_when_winner_pulls_ahead():
    s_w = torch.tensor([0.0, 0.0])
    s_l = torch.tensor([0.0, 0.0])
    tied = float(orpo_loss(s_w, s_l, lambda_or=1.0))
    better = float(orpo_loss(s_w + 2.0, s_l - 2.0, lambda_or=1.0))
    worse = float(orpo_loss(s_w - 2.0, s_l + 2.0, lambda_or=1.0))
    assert better < tied < worse


def test_orpo_loss_is_sft_plus_ranknet():
    s_w = torch.tensor([1.0])
    s_l = torch.tensor([-0.5])
    got = float(orpo_loss(s_w, s_l, lambda_or=0.5))
    sft = float(-torch.nn.functional.logsigmoid(s_w))
    rn = float(-torch.nn.functional.logsigmoid(s_w - s_l))
    assert got == pytest.approx(sft + 0.5 * rn, abs=1e-6)


def test_triples_exclude_same_admission_and_eval():
    rows = [
        {
            "subject_id": "t1", "qa_id": "q1", "question": "why ablation",
            "passage": "gold turn", "label": 1, "kind": "gold", "grain": "turn",
            "hadm_id": "H1", "turn_number": 5,
        },
        {
            "subject_id": "t1", "qa_id": "q1", "question": "why ablation",
            "passage": "same adm hardneg", "label": 0, "kind": "hardneg",
            "hadm_id": "H1", "turn_number": 9,
        },
        {
            "subject_id": "t1", "qa_id": "q1", "question": "why ablation",
            "passage": "other adm hardneg", "label": 0, "kind": "hardneg",
            "hadm_id": "H2", "turn_number": 1,
        },
    ]
    triples = build_triples_from_rows(rows, k_neg=4)
    assert len(triples) == 1
    assert triples[0].rejected == "other adm hardneg"
    assert triples[0].rejected_hadm == "H2"


def test_admission_only_item_emits_triples(tmp_path):
    rows = [
        {
            "subject_id": "t2", "qa_id": "q2", "question": "why admitted",
            "passage": f"t{i}", "label": 1, "kind": "gold",
            "grain": "admission_expanded", "hadm_id": "H1", "turn_number": i,
        }
        for i in range(20)
    ] + [
        {
            "subject_id": "t2", "qa_id": "q2", "question": "why admitted",
            "passage": "neg", "label": 0, "kind": "hardneg",
            "hadm_id": "H9", "turn_number": 1,
        }
    ]
    triples = build_triples_from_rows(rows, k_neg=1, max_adm_golds=4, seed=20260817)
    assert 1 <= len(triples) <= 4  # max_adm_golds=4, one other-adm neg
    assert all(t.grain == "admission_expanded" for t in triples)
    assert all(t.rejected == "neg" for t in triples)


def test_eval_trio_row_raises():
    rows = [
        {
            "subject_id": EVAL_TRIO[0], "qa_id": "q", "question": "q",
            "passage": "p", "label": 1, "kind": "gold", "grain": "turn",
            "hadm_id": "H1", "turn_number": 1,
        }
    ]
    with pytest.raises(RuntimeError, match="eval-trio"):
        build_triples_from_rows(rows)


def test_script_is_minilm_and_orpo():
    from scripts.finetune_minilm_orpo import STUDENT_HF_ID
    from scripts.finetune_minilm_orpo import CHECKPOINT as ORPO_CKPT

    src = Path("scripts/finetune_minilm_orpo.py").read_text(encoding="utf-8")
    assert STUDENT_HF_ID == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert "orpo_loss" in src
    assert "qwen3-rerank" not in src
    assert "ft-orpo" in str(ORPO_CKPT)
    assert "ft-medlocomo" not in str(ORPO_CKPT)
