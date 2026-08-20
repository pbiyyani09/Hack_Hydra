"""CE-ORPO finetune of MiniLM-L-6-v2. Yields the 3090. Refuses eval trio.

Does not overwrite the first pointwise-BCE checkpoint.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from medmemgraph.eval.rerank_orpo import (
    DEFAULT_K_NEG,
    DEFAULT_LAMBDA,
    DEFAULT_MAX_ADM_GOLDS,
    Triple,
    build_triples_from_rows,
    iter_pairs,
    orpo_loss,
    triples_stats,
)
from medmemgraph.eval.rerank_split import EVAL_TRIO
from scripts.finetune_minilm_reranker import (
    DEV_PAIRS,
    SEED,
    SPLIT_DEFAULT,
    STUDENT_HF_ID,
    STUDENT_PARAMS,
    TRAIN_PAIRS,
    choose_device,
    eval_trio_subjects_in,
)

CHECKPOINT = Path(
    os.environ.get("ORPO_CHECKPOINT", "data/reranker_ft/ms-marco-minilm-l6-v2-ft-orpo")
)
INIT_FROM = os.environ.get("ORPO_INIT_FROM", STUDENT_HF_ID)
LOG_PATH = Path(os.environ.get("ORPO_LOG", "results/finetune-reranker/train_orpo_log.md"))
LR = float(os.environ.get("ORPO_LR", "1e-5"))
EPOCHS = int(os.environ.get("ORPO_EPOCHS", "4"))
BATCH = 16
MAX_LEN = 512
MAX_ADM = int(os.environ.get("ORPO_MAX_ADM", str(DEFAULT_MAX_ADM_GOLDS)))
K_NEG = int(os.environ.get("ORPO_K_NEG", str(DEFAULT_K_NEG)))


def _refuse_eval(path: Path) -> None:
    leaked = eval_trio_subjects_in(iter_pairs(path))
    if leaked:
        raise SystemExit(
            f"eval-trio subject_id in {path}: {sorted(set(leaked))}; refusing to train"
        )


def _score_pairs(model, queries: list[str], passages: list[str], device: str):
    tok = model.tokenizer(
        list(queries),
        list(passages),
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    )
    tok = {k: v.to(device) for k, v in tok.items()}
    logits = model.model(**tok).logits
    return logits.view(-1)


def _batches(items: list[Triple], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def pairwise_acc(model, triples: list[Triple], device: str, *, cap: int = 2048) -> float:
    import torch

    if not triples:
        return float("nan")
    sample = triples[:cap]
    model.model.eval()
    correct = 0
    n = 0
    with torch.no_grad():
        for batch in _batches(sample, BATCH):
            qs = [t.question for t in batch]
            s_w = _score_pairs(model, qs, [t.chosen for t in batch], device)
            s_l = _score_pairs(model, qs, [t.rejected for t in batch], device)
            correct += int((s_w > s_l).sum().item())
            n += len(batch)
    model.model.train()
    return correct / max(n, 1)


def main() -> int:
    if not TRAIN_PAIRS.is_file() or not DEV_PAIRS.is_file():
        print("missing pair files", file=sys.stderr)
        return 1
    _refuse_eval(TRAIN_PAIRS)
    _refuse_eval(DEV_PAIRS)
    split = json.loads(SPLIT_DEFAULT.read_text(encoding="utf-8"))
    for sid in list(split.get("train", [])) + list(split.get("dev", [])):
        if sid in EVAL_TRIO:
            raise SystemExit(f"eval trio in split train/dev: {sid}")

    device, device_reason = choose_device()
    t0 = time.monotonic()

    import random

    import numpy as np
    import torch
    from sentence_transformers import CrossEncoder
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import LambdaLR

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if device == "cuda":
        torch.cuda.manual_seed_all(SEED)

    train_triples = build_triples_from_rows(
        iter_pairs(TRAIN_PAIRS), seed=SEED, k_neg=K_NEG, max_adm_golds=MAX_ADM
    )
    dev_triples = build_triples_from_rows(
        iter_pairs(DEV_PAIRS), seed=SEED + 1, k_neg=K_NEG, max_adm_golds=MAX_ADM
    )
    if not train_triples:
        print("no train triples", file=sys.stderr)
        return 1

    model = CrossEncoder(INIT_FROM, num_labels=1, device=device)
    model.model.train()
    opt = AdamW(model.model.parameters(), lr=LR, weight_decay=0.01)
    steps_per_epoch = max(1, (len(train_triples) + BATCH - 1) // BATCH)
    total_steps = steps_per_epoch * EPOCHS
    warmup = max(10, int(0.1 * total_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return float(step) / float(max(1, warmup))
        return max(0.0, float(total_steps - step) / float(max(1, total_steps - warmup)))

    sched = LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None

    CHECKPOINT.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    best_acc = -1.0
    history: list[str] = []
    global_step = 0
    for epoch in range(1, EPOCHS + 1):
        random.shuffle(train_triples)
        model.model.train()
        running = 0.0
        n_batches = 0
        for batch in _batches(train_triples, BATCH):
            qs = [t.question for t in batch]
            chosen = [t.chosen for t in batch]
            rejected = [t.rejected for t in batch]
            opt.zero_grad(set_to_none=True)
            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    s_w = _score_pairs(model, qs, chosen, device)
                    s_l = _score_pairs(model, qs, rejected, device)
                    loss = orpo_loss(s_w, s_l, lambda_or=DEFAULT_LAMBDA)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                s_w = _score_pairs(model, qs, chosen, device)
                s_l = _score_pairs(model, qs, rejected, device)
                loss = orpo_loss(s_w, s_l, lambda_or=DEFAULT_LAMBDA)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.model.parameters(), 1.0)
                opt.step()
            sched.step()
            running += float(loss.detach())
            n_batches += 1
            global_step += 1
            if n_batches % 200 == 0:
                print(
                    f"epoch {epoch} step {n_batches}/{steps_per_epoch} "
                    f"loss={running / n_batches:.4f}",
                    flush=True,
                )
        dev_acc = pairwise_acc(model, dev_triples, device)
        train_acc = pairwise_acc(model, train_triples, device, cap=1024)
        line = (
            f"epoch {epoch}: train_loss={running / max(n_batches, 1):.4f} "
            f"train_pair_acc@{min(1024, len(train_triples))}={train_acc:.3f} "
            f"dev_pair_acc={dev_acc:.3f}"
        )
        history.append(line)
        print(line, flush=True)
        if dev_acc >= best_acc:
            best_acc = dev_acc
            model.save(str(CHECKPOINT))
            print(f"saved best to {CHECKPOINT} (dev_pair_acc={dev_acc:.3f})", flush=True)

    wall_h = (time.monotonic() - t0) / 3600.0
    tstats = triples_stats(train_triples)
    dstats = triples_stats(dev_triples)
    log = f"""# MiniLM CE-ORPO train log

- student: `{STUDENT_HF_ID}` ({STUDENT_PARAMS} params)
- objective: L = −logσ(s_w) + λ −logσ(s_w − s_l), λ={DEFAULT_LAMBDA}
- rejected slot: other-admission hard-negs only (k_neg={DEFAULT_K_NEG})
- admission golds/QA cap: {DEFAULT_MAX_ADM_GOLDS}
- split: `{SPLIT_DEFAULT}` seed {split.get("seed", SEED)}
- train patients: {len(split.get("train", []))} · dev: {len(split.get("dev", []))} · eval held out: {len(split.get("eval", []))}
- train triples: {tstats}
- dev triples: {dstats}
- lr={LR} epochs={EPOCHS} batch_triples={BATCH} warmup_steps={warmup} max_len={MAX_LEN}
- device: **{device}** — {device_reason}
- wall hours: {wall_h:.3f}
- best dev pairwise acc: {best_acc:.3f}
- history: {history}
- checkpoint: `{CHECKPOINT}` (does not overwrite the pointwise-BCE FT)
- 3090 yielded: {"yes" if device == "cpu" and "yielded" in device_reason else "no"}
"""
    LOG_PATH.write_text(log, encoding="utf-8")
    print(log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
