"""CE-ORPO triples and loss for MiniLM (scalar logit, not a causal LLM).

ORPO (Hong et al. 2024): L = L_SFT + λ L_OR
  L_SFT = −log σ(s_w)                 # gold is the chosen class
  L_OR  = −log σ(s_w − s_l)           # log-odds ratio of 1-logit CE

Same student, pairwise objective. Same-admission hard-negs are excluded
from the rejected slot: they are turn-grain negatives but admission-grain
positives, and pushing them down is what collapsed admission Hit@10.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

from medmemgraph.eval.rerank_split import EVAL_TRIO

DEFAULT_LAMBDA = 1.0
DEFAULT_K_NEG = 4
DEFAULT_MAX_ADM_GOLDS = 2
DEFAULT_SEED = 20260817


@dataclass(frozen=True)
class Triple:
    subject_id: str
    qa_id: str
    question: str
    chosen: str
    rejected: str
    chosen_hadm: str
    rejected_hadm: str
    grain: str  # "turn" | "admission_expanded"


def orpo_loss(s_w, s_l, *, lambda_or: float = DEFAULT_LAMBDA):
    """L = −logσ(s_w) − λ −logσ(s_w − s_l). Mean over the batch."""
    import torch
    import torch.nn.functional as F

    sft = -F.logsigmoid(s_w)
    odds = -F.logsigmoid(s_w - s_l)
    return (sft + lambda_or * odds).mean()


def build_triples_from_rows(
    rows: Iterable[dict[str, Any]],
    *,
    k_neg: int = DEFAULT_K_NEG,
    max_adm_golds: int = DEFAULT_MAX_ADM_GOLDS,
    seed: int = DEFAULT_SEED,
    exclude_same_admission: bool = True,
) -> list[Triple]:
    """Group by (subject, qa). Prefer turn golds. Reject other-admission hard-negs."""
    import random

    rng = random.Random(seed)
    by_qa: dict[tuple[str, str], dict[str, list]] = defaultdict(
        lambda: {"question": None, "turn": [], "adm": [], "neg": []}
    )
    for row in rows:
        sid = str(row["subject_id"])
        if sid in EVAL_TRIO:
            raise RuntimeError(f"eval-trio subject {sid!r} in triple builder")
        key = (sid, str(row["qa_id"]))
        bucket = by_qa[key]
        bucket["question"] = str(row["question"])
        kind = row.get("kind")
        rec = {
            "passage": str(row["passage"]),
            "hadm_id": str(row.get("hadm_id", "")),
        }
        if kind == "gold" and row.get("grain") == "admission_expanded":
            bucket["adm"].append(rec)
        elif kind == "gold":
            bucket["turn"].append(rec)
        elif kind == "hardneg":
            bucket["neg"].append(rec)

    triples: list[Triple] = []
    for (sid, qid), bucket in by_qa.items():
        question = bucket["question"] or ""
        golds = bucket["turn"]
        grain = "turn"
        if not golds:
            golds = list(bucket["adm"])
            grain = "admission_expanded"
            if len(golds) > max_adm_golds:
                golds = rng.sample(golds, max_adm_golds)
        gold_hadm = {g["hadm_id"] for g in golds} | {g["hadm_id"] for g in bucket["adm"]}
        negs = list(bucket["neg"])
        if exclude_same_admission:
            negs = [n for n in negs if n["hadm_id"] not in gold_hadm]
        if not golds or not negs:
            continue
        for gold in golds:
            take = negs if len(negs) <= k_neg else rng.sample(negs, k_neg)
            for neg in take:
                triples.append(
                    Triple(
                        subject_id=sid,
                        qa_id=qid,
                        question=question,
                        chosen=gold["passage"],
                        rejected=neg["passage"],
                        chosen_hadm=gold["hadm_id"],
                        rejected_hadm=neg["hadm_id"],
                        grain=grain,
                    )
                )
    rng.shuffle(triples)
    return triples


def iter_pairs(path) -> Iterator[dict[str, Any]]:
    import json
    from pathlib import Path

    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def triples_stats(triples: list[Triple]) -> dict[str, int]:
    n_turn = sum(1 for t in triples if t.grain == "turn")
    return {
        "n_triples": len(triples),
        "n_turn": n_turn,
        "n_admission_expanded": len(triples) - n_turn,
        "n_subjects": len({t.subject_id for t in triples}),
        "n_qa": len({(t.subject_id, t.qa_id) for t in triples}),
    }
