"""FT-E2-S1: finetune MiniLM-L-6-v2 on gold + hard-neg pairs.

Yields the 3090 if another compute process owns it. Refuses eval-trio
subject_ids before any training step.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from medmemgraph.eval.rerank_split import EVAL_TRIO

STUDENT_HF_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"
STUDENT_PARAMS = 22_714_113
SPLIT_DEFAULT = Path("results/finetune-reranker/patient_split.json")
TRAIN_PAIRS = Path("data/reranker_ft/train_pairs.jsonl")
DEV_PAIRS = Path("data/reranker_ft/dev_pairs.jsonl")
CHECKPOINT = Path("data/reranker_ft/ms-marco-minilm-l6-v2-ft-medlocomo")
LOG_PATH = Path("results/finetune-reranker/train_log.md")
SEED = 20260817


def load_pairs(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def iter_pairs(path: Path):
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def eval_trio_subjects_in(rows: list[dict]) -> list[str]:
    found: list[str] = []
    for row in rows:
        sid = str(row.get("subject_id", ""))
        if sid in EVAL_TRIO and sid not in found:
            found.append(sid)
    return found


def other_gpu_compute_pids(self_pid: int | None = None) -> list[int]:
    """PIDs of other compute processes on the GPU, if any."""
    self_pid = os.getpid() if self_pid is None else self_pid
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return []
    if proc.returncode != 0:
        return []
    pids: list[int] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            pid = int(line.split(",")[0].strip())
        except ValueError:
            continue
        if pid != self_pid:
            pids.append(pid)
    return pids


def choose_device() -> tuple[str, str]:
    """Return (device, reason). Never grab the 3090 if extract owns it."""
    others = other_gpu_compute_pids()
    if others:
        return "cpu", f"yielded 3090; other compute pids={others}"
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda", "3090 idle; short MiniLM fit allowed"
    except ImportError:
        pass
    return "cpu", "cuda unavailable; training on CPU"


def refuse_eval_trio(train_rows: list[dict], dev_rows: list[dict]) -> None:
    leaked = eval_trio_subjects_in(train_rows) + eval_trio_subjects_in(dev_rows)
    if leaked:
        raise SystemExit(
            f"eval-trio subject_id in pair files: {sorted(set(leaked))}; "
            "refusing to train"
        )


def _count_kind(rows: list[dict], kind: str) -> int:
    return sum(1 for r in rows if r.get("kind") == kind)


def rebalance_pairs(
    rows, *, max_adm_gold_per_qa: int = 16, seed: int = SEED
) -> tuple[list[dict], dict[str, int]]:
    """Keep every turn-grain gold and every hard-neg. Cap admission-expanded
    golds per QA so they cannot drown the turn-grain target (~60× more
    pairs on disk). Pair files themselves are unfiltered (A-grain).

    Admission golds are reservoir-sampled so a 1.5M-pair file never sits
    fully in RAM.
    """
    import random

    rng = random.Random(seed)
    kept: list[dict] = []
    adm_by_qa: dict[tuple[str, str], list[dict]] = {}
    adm_seen: dict[tuple[str, str], int] = {}
    stats = {
        "in_total": 0,
        "in_gold_turn": 0,
        "in_gold_adm": 0,
        "in_hardneg": 0,
        "out_gold_turn": 0,
        "out_gold_adm": 0,
        "out_hardneg": 0,
    }
    for row in rows:
        stats["in_total"] += 1
        kind = row.get("kind")
        grain = row.get("grain")
        if kind == "hardneg":
            kept.append(row)
            stats["in_hardneg"] += 1
            stats["out_hardneg"] += 1
        elif kind == "gold" and grain == "admission_expanded":
            stats["in_gold_adm"] += 1
            key = (str(row["subject_id"]), str(row["qa_id"]))
            adm_seen[key] = adm_seen.get(key, 0) + 1
            bucket = adm_by_qa.setdefault(key, [])
            n = adm_seen[key]
            if len(bucket) < max_adm_gold_per_qa:
                bucket.append(row)
            else:
                j = rng.randint(1, n)
                if j <= max_adm_gold_per_qa:
                    bucket[j - 1] = row
        else:
            kept.append(row)
            stats["in_gold_turn"] += 1
            stats["out_gold_turn"] += 1
    for group in adm_by_qa.values():
        kept.extend(group)
        stats["out_gold_adm"] += len(group)
    rng.shuffle(kept)
    stats["out_total"] = len(kept)
    stats["max_adm_gold_per_qa"] = max_adm_gold_per_qa
    return kept, stats


def main() -> int:
    if not SPLIT_DEFAULT.is_file():
        print(f"missing {SPLIT_DEFAULT}", file=sys.stderr)
        return 1
    if not TRAIN_PAIRS.is_file() or not DEV_PAIRS.is_file():
        print("missing pair files; run scripts/build_rerank_pairs.py", file=sys.stderr)
        return 1

    split = json.loads(SPLIT_DEFAULT.read_text(encoding="utf-8"))
    # Scan subject_id only — do not materialize 1.5M admission golds first.
    leaked: list[str] = []
    for path in (TRAIN_PAIRS, DEV_PAIRS):
        leaked.extend(eval_trio_subjects_in(iter_pairs(path)))
    if leaked:
        raise SystemExit(
            f"eval-trio subject_id in pair files: {sorted(set(leaked))}; "
            "refusing to train"
        )

    train_rows = iter_pairs(TRAIN_PAIRS)
    dev_rows = iter_pairs(DEV_PAIRS)

    device, device_reason = choose_device()
    t0 = time.monotonic()

    import random

    import numpy as np
    import torch
    from sentence_transformers import CrossEncoder, InputExample
    from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
    from torch.utils.data import DataLoader

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if device == "cuda":
        torch.cuda.manual_seed_all(SEED)

    train_kept, train_bal = rebalance_pairs(train_rows)
    dev_kept, dev_bal = rebalance_pairs(dev_rows)
    train_examples = [
        InputExample(texts=[r["question"], r["passage"]], label=float(r["label"]))
        for r in train_kept
    ]
    dev_examples = [
        InputExample(texts=[r["question"], r["passage"]], label=float(r["label"]))
        for r in dev_kept
    ]
    if not train_examples:
        print("no train pairs", file=sys.stderr)
        return 1

    batch_size = 32
    epochs = 1
    loader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    steps_per_epoch = max(1, len(train_examples) // batch_size)
    warmup = max(10, int(0.1 * steps_per_epoch * epochs))
    eval_steps = max(50, steps_per_epoch)

    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    model = CrossEncoder(STUDENT_HF_ID, num_labels=1, device=device)
    evaluator = None
    if dev_examples:
        evaluator = CEBinaryClassificationEvaluator.from_input_examples(
            dev_examples, name="dev-patients"
        )

    # ST 5.7 CrossEncoder.fit requires the optional `datasets` extra.
    # old_fit is the pre-v4 loop and needs no new dependency (uv only).
    model.old_fit(
        train_dataloader=loader,
        evaluator=evaluator,
        epochs=epochs,
        warmup_steps=warmup,
        evaluation_steps=eval_steps if evaluator is not None else 0,
        output_path=str(CHECKPOINT),
        save_best_model=True,
        optimizer_params={"lr": 2e-5},
        use_amp=(device == "cuda"),
        show_progress_bar=True,
    )
    if not (CHECKPOINT / "config.json").exists():
        model.save(str(CHECKPOINT))

    wall_h = (time.monotonic() - t0) / 3600.0
    log = f"""# MiniLM gold+hardneg train log

- student: `{STUDENT_HF_ID}` ({STUDENT_PARAMS} params)
- split: `{SPLIT_DEFAULT}` (seed {split.get("seed", SEED)})
- train patients: {len(split.get("train", []))}
- dev patients: {len(split.get("dev", []))}
- eval patients: {len(split.get("eval", []))} (held out)
- train pairs on disk: {train_bal["in_total"]} (gold_turn={train_bal["in_gold_turn"]}, gold_adm={train_bal["in_gold_adm"]}, hardneg={train_bal["in_hardneg"]})
- train pairs used: {train_bal}
- dev pairs on disk: {dev_bal["in_total"]} (gold_turn={dev_bal["in_gold_turn"]}, gold_adm={dev_bal["in_gold_adm"]}, hardneg={dev_bal["in_hardneg"]})
- dev pairs used: {dev_bal}
- device: **{device}** — {device_reason}
- wall hours: {wall_h:.3f}
- seed: {SEED}
- CrossEncoder.old_fit (ST 5.7 `fit` needs `datasets`; not added): BinaryCrossEntropy (num_labels=1), batch={batch_size}, epochs={epochs}, lr=2e-5, warmup={warmup}, eval_steps={eval_steps}
- checkpoint: `{CHECKPOINT}`
- 3090 yielded: {"yes" if device == "cpu" and "yielded" in device_reason else "no"}
"""
    LOG_PATH.write_text(log, encoding="utf-8")
    print(log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
