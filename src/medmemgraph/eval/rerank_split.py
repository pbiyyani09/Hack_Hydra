"""Deterministic patient-level train/dev/eval split for reranker finetune."""

from __future__ import annotations

import random
from typing import Any

from medmemgraph.pipeline.loader import list_patients

EVAL_TRIO = ("10056223", "10213338", "10312715")
DEFAULT_N_DEV = 10
DEFAULT_SEED = 20260817


def build_split(
    root: object | None = None, *, n_dev: int = DEFAULT_N_DEV, seed: int = DEFAULT_SEED
) -> dict[str, Any]:
    patients = list_patients(root)
    missing = [p for p in EVAL_TRIO if p not in patients]
    if missing:
        raise RuntimeError(f"eval trio not in corpus: {missing}")
    remaining = [p for p in patients if p not in EVAL_TRIO]
    if len(remaining) < n_dev:
        raise RuntimeError("not enough non-eval patients for dev")
    rng = random.Random(seed)
    dev = sorted(rng.sample(remaining, n_dev))
    train = sorted(p for p in remaining if p not in set(dev))
    return {
        "eval": list(EVAL_TRIO),
        "dev": dev,
        "train": train,
        "n_dev": n_dev,
        "seed": seed,
        "grain": "mixed-turn-preferred",
        "grain_rule": (
            "turn-level positives when evidence.turn_ids exist; "
            "else all turns of gold admissions as positives"
        ),
        "source": "list_patients(); eval trio locked; Random(20260817).sample",
    }
