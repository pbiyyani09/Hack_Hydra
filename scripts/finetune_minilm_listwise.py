"""Listwise LambdaLoss MiniLM. Train on GPU; export is CPU ONNX later.

Graded labels: gold turn=2, other gold-admission turn=1, else 0.
Hybrid BM25+dense lists from train/dev patients only.
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

from medmemgraph.eval.hybrid_pool import build_indexes
from medmemgraph.eval.listwise_lists import lists_for_patient
from medmemgraph.eval.rerank_split import EVAL_TRIO
from medmemgraph.pipeline.loader import load_conversation, load_qa
from scripts.finetune_minilm_reranker import (
    SEED,
    SPLIT_DEFAULT,
    STUDENT_HF_ID,
    choose_device,
    eval_trio_subjects_in,
)

CHECKPOINT = Path("data/reranker_ft/ms-marco-minilm-l6-v2-ft-listwise")
LOG = Path("results/finetune-reranker/train_listwise_log.md")
LIST_SIZE = 32
LR = 5e-6
EPOCHS = int(os.environ.get("LISTWISE_EPOCHS", "2"))
BATCH_LISTS = 1
MINI_BATCH_DOCS = 16


def _load_split() -> dict:
    split = json.loads(SPLIT_DEFAULT.read_text(encoding="utf-8"))
    leaked = [p for p in EVAL_TRIO if p in split["train"] or p in split["dev"]]
    if leaked:
        raise SystemExit(f"eval trio in train/dev: {leaked}")
    return split


def _cache_path(name: str) -> Path:
    return Path(f"data/reranker_ft/listwise_{name}.jsonl")


def _save_lists(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in rows:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _load_lists(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _build_all(subjects: list[str], backend: str) -> list[dict]:
    rows: list[dict] = []
    for i, sid in enumerate(subjects, 1):
        print(f"  lists {backend} {i}/{len(subjects)} {sid}", flush=True)
        convo = load_conversation(sid)
        dense, lex = build_indexes(sid, convo, dense_backend=backend)
        rows.extend(lists_for_patient(sid, load_qa(sid), dense, lex, list_size=LIST_SIZE))
        if eval_trio_subjects_in(rows):
            raise SystemExit("eval trio leaked into lists")
    return rows


def main() -> int:
    split = _load_split()
    device, reason = choose_device()
    t0 = time.monotonic()

    train_cache, dev_cache = _cache_path("train_arctic_pack"), _cache_path("dev_arctic_pack")
    if train_cache.is_file() and dev_cache.is_file():
        print(f"loading cached lists {train_cache} {dev_cache}", flush=True)
        train_rows = _load_lists(train_cache)
        dev_rows = _load_lists(dev_cache)
        if eval_trio_subjects_in(train_rows + dev_rows):
            raise SystemExit("eval trio in cached lists")
    else:
        print("building train lists (arctic-s + BM25)…", flush=True)
        train_rows = _build_all(list(split["train"]), "arctic-s")
        print("building dev lists…", flush=True)
        dev_rows = _build_all(list(split["dev"]), "arctic-s")
        _save_lists(train_cache, train_rows)
        _save_lists(dev_cache, dev_rows)
    print(f"train lists={len(train_rows)} dev lists={len(dev_rows)}", flush=True)
    if not train_rows:
        print("no train lists", file=sys.stderr)
        return 1

    import math
    import random

    import numpy as np
    import torch
    from sentence_transformers import CrossEncoder
    from torch.optim import AdamW

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if device == "cuda":
        torch.cuda.manual_seed_all(SEED)

    init = os.environ.get("LISTWISE_INIT", STUDENT_HF_ID)
    model = CrossEncoder(init, num_labels=1, device=device)
    opt = AdamW(model.model.parameters(), lr=LR, weight_decay=0.01)

    def _score(query: str, docs: list[str]) -> torch.Tensor:
        tok = model.tokenizer(
            [query] * len(docs),
            docs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        tok = {k: v.to(device) for k, v in tok.items()}
        return model.model(**tok).logits.view(-1)

    def _listnet(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        q = torch.softmax(labels.float(), dim=0)
        logp = torch.log_softmax(scores.float(), dim=0)
        return -(q * logp).sum()
    CHECKPOINT.mkdir(parents=True, exist_ok=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)

    def _eval(rows: list[dict], cap: int = 200) -> float:
        model.model.eval()
        total = 0.0
        n = 0
        with torch.no_grad():
            for rec in rows[:cap]:
                scores = _score(rec["question"], rec["docs"]).detach().cpu().tolist()
                order = list(np.argsort(-np.asarray(scores)))
                labels = rec["labels"]
                dcg = sum(labels[idx] / math.log2(r + 2) for r, idx in enumerate(order[:10]))
                ideal = sorted(labels, reverse=True)
                idcg = sum(g / math.log2(r + 2) for r, g in enumerate(ideal[:10]))
                if idcg > 0:
                    total += dcg / idcg
                    n += 1
        model.model.train()
        return total / max(n, 1)

    best = -1.0
    history: list[str] = []
    model.model.train()
    for epoch in range(1, EPOCHS + 1):
        random.shuffle(train_rows)
        running = 0.0
        steps = 0
        skipped = 0
        for rec in train_rows:
            labs = rec["labels"]
            if max(labs) <= 0:
                skipped += 1
                continue
            opt.zero_grad(set_to_none=True)
            scores = _score(rec["question"], rec["docs"])
            labels = torch.tensor(labs, dtype=torch.float32, device=device)
            loss = _listnet(scores, labels)
            if not torch.isfinite(loss):
                skipped += 1
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.model.parameters(), 1.0)
            opt.step()
            running += float(loss.detach())
            steps += 1
            if steps % 200 == 0:
                print(
                    f"epoch {epoch} step {steps} loss={running / steps:.4f} skip={skipped}",
                    flush=True,
                )
        dev_ndcg = _eval(dev_rows)
        line = f"epoch {epoch}: loss={running / max(steps, 1):.4f} dev_list_ndcg10={dev_ndcg:.3f}"
        history.append(line)
        print(line, flush=True)
        if dev_ndcg >= best:
            best = dev_ndcg
            model.save(str(CHECKPOINT))
            print(f"saved {CHECKPOINT}", flush=True)

    wall = (time.monotonic() - t0) / 3600.0
    LOG.write_text(
        f"""# MiniLM listwise (LambdaLoss NDCG-Loss2++)

- init: `{init}`
- lists: hybrid arctic-s + BM25, size={LIST_SIZE}
- grades: turn=2, same-admission=1, other=0
- train lists={len(train_rows)} dev={len(dev_rows)}
- lr={LR} epochs={EPOCHS} device={device} ({reason})
- wall hours={wall:.3f} best_dev_list_ndcg10={best:.3f}
- history: {history}
- checkpoint: `{CHECKPOINT}`
""",
        encoding="utf-8",
    )
    print(LOG.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
