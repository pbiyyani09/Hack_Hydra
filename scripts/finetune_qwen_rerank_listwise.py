"""Listwise finetune of qwen3-rerank-0.6b on GPU. Same split/grades as MiniLM.

Saves a local causal-LM checkpoint. Inference stays GPU (this stack is
the GPU arm). Does not overwrite the Hub Qwen weights.
"""

from __future__ import annotations

import gc
import json
import math
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
from medmemgraph.graph.reranker import REGISTERED_MODELS
from medmemgraph.pipeline.loader import load_conversation, load_qa
from scripts.finetune_minilm_reranker import SEED, SPLIT_DEFAULT, choose_device

CHECKPOINT = Path("data/reranker_ft/qwen3-rerank-0.6b-ft-listwise")
LOG = Path("results/finetune-reranker/train_qwen_listwise_log.md")
LIST_SIZE = 16  # 0.6B CE; keep lists short
TRAIN_LIST_KEEP = 8
MAX_TRAIN_LISTS = 6000
UNFREEZE_LAST = 8
MAX_LEN = 512
LR = 5e-6
EPOCHS = 1
QWEN_KEY = "qwen3-rerank-0.6b"
TRAIN_CACHE = Path("data/reranker_ft/listwise_train_qwen.jsonl")
DEV_CACHE = Path("data/reranker_ft/listwise_dev_qwen.jsonl")


def _save_lists(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in rows:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _load_lists(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _build_all(subjects: list[str]) -> list[dict]:
    rows: list[dict] = []
    for i, sid in enumerate(subjects, 1):
        print(f"  qwen-lists {i}/{len(subjects)} {sid}", flush=True)
        if sid in EVAL_TRIO:
            raise SystemExit(f"eval trio {sid}")
        convo = load_conversation(sid)
        dense, lex = build_indexes(sid, convo, dense_backend="qwen3-0.6b")
        rows.extend(lists_for_patient(sid, load_qa(sid), dense, lex, list_size=LIST_SIZE))
    return rows


def _load_or_build(split: dict) -> tuple[list[dict], list[dict]]:
    if TRAIN_CACHE.is_file() and DEV_CACHE.is_file():
        print(f"loading cached Qwen lists {TRAIN_CACHE} {DEV_CACHE}", flush=True)
        return _load_lists(TRAIN_CACHE), _load_lists(DEV_CACHE)
    print("building Qwen+BM25 train lists…", flush=True)
    train_rows = _build_all(list(split["train"]))
    print("building Qwen+BM25 dev lists…", flush=True)
    dev_rows = _build_all(list(split["dev"]))
    _save_lists(TRAIN_CACHE, train_rows)
    _save_lists(DEV_CACHE, dev_rows)
    return train_rows, dev_rows


def main() -> int:
    split = json.loads(SPLIT_DEFAULT.read_text(encoding="utf-8"))
    if os.environ.get("MEDMEM_FORCE_GPU") == "1":
        device, reason = "cuda", "MEDMEM_FORCE_GPU=1"
    else:
        device, reason = choose_device()
    if device != "cuda":
        print(f"Qwen listwise needs GPU; got {device} ({reason})", file=sys.stderr)
        return 1
    t0 = time.monotonic()
    train_rows, dev_rows = _load_or_build(split)
    print(f"train={len(train_rows)} dev={len(dev_rows)}", flush=True)

    import random

    import torch
    from torch.optim import AdamW
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    def _shrink(rec: dict, keep: int) -> dict:
        labs = rec["labels"]
        pos = [i for i, lab in enumerate(labs) if lab > 0]
        neg = [i for i, lab in enumerate(labs) if lab <= 0]
        extra = max(0, keep - len(pos))
        chosen = pos + (random.sample(neg, extra) if extra and len(neg) >= extra else neg[:extra])
        chosen = chosen[:keep]
        if not chosen:
            return rec
        return {
            **rec,
            "docs": [rec["docs"][i] for i in chosen],
            "labels": [labs[i] for i in chosen],
        }

    train_rows = [_shrink(r, TRAIN_LIST_KEEP) for r in train_rows]
    dev_rows = [_shrink(r, TRAIN_LIST_KEEP) for r in dev_rows]
    random.shuffle(train_rows)
    train_rows = train_rows[:MAX_TRAIN_LISTS]
    print(f"after shrink/cap train={len(train_rows)} keep={TRAIN_LIST_KEEP}", flush=True)

    gc.collect()
    torch.cuda.empty_cache()

    spec = REGISTERED_MODELS[QWEN_KEY]
    # Always the Hub tokenizer (local save_pretrained has triggered a
    # mistral-regex warning on reload). Eval uses the same Hub recipe.
    tok = AutoTokenizer.from_pretrained(spec.hf_id, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(spec.hf_id, torch_dtype=torch.float32).to(device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.train()
    torch.backends.cuda.matmul.allow_tf32 = True
    for param in model.parameters():
        param.requires_grad = False
    for layer in model.model.layers[-UNFREEZE_LAST:]:
        for param in layer.parameters():
            param.requires_grad = True
    for param in model.lm_head.parameters():
        param.requires_grad = True
    if hasattr(model.model, "norm"):
        for param in model.model.norm.parameters():
            param.requires_grad = True
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"model loaded; trainable={n_train / 1e6:.1f}M "
        f"cuda mem={torch.cuda.memory_allocated() / 1e9:.2f} GB",
        flush=True,
    )
    prefix = (
        "<|im_start|>system\nJudge whether the Document meets the "
        'requirements based on the Query and the Instruct provided. '
        'Note that the answer can only be "yes" or "no".<|im_end|>\n'
        "<|im_start|>user\n"
    )
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    instruction = spec.instruction or (
        "Given a clinical conversation search query, retrieve the passage "
        "that most directly answers the query"
    )
    yes_id = tok.convert_tokens_to_ids("yes")
    no_id = tok.convert_tokens_to_ids("no")
    prefix_ids = tok.encode(prefix, add_special_tokens=False)
    suffix_ids = tok.encode(suffix, add_special_tokens=False)
    budget = MAX_LEN - len(prefix_ids) - len(suffix_ids)
    opt = AdamW((p for p in model.parameters() if p.requires_grad), lr=LR, weight_decay=0.01)

    def yes_logits(query: str, docs: list[str], *, chunk: int = 2) -> torch.Tensor:
        # Same ID-concat recipe as _Qwen3YesNoBackend.score_batch so the
        # last token is the judgment position (not a truncated suffix).
        parts: list[torch.Tensor] = []
        for start in range(0, len(docs), chunk):
            bodies = [
                f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"
                for doc in docs[start : start + chunk]
            ]
            enc = tok(
                bodies,
                padding=False,
                truncation="longest_first",
                max_length=budget,
                add_special_tokens=False,
            )
            for i, ids in enumerate(enc["input_ids"]):
                enc["input_ids"][i] = prefix_ids + ids + suffix_ids
            batch = tok.pad(enc, padding=True, return_tensors="pt")
            batch = {k: v.to(device) for k, v in batch.items()}
            hidden = model.model(**batch).last_hidden_state[:, -1, :]
            logits = model.lm_head(hidden)
            parts.append(logits[:, yes_id] - logits[:, no_id])
            del hidden, logits, batch
        return torch.cat(parts, dim=0)

    def listnet(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Softmax CE between label distribution and score distribution.
        lab = labels.to(scores.device).float()
        lab = lab - lab.max()
        q = torch.softmax(lab, dim=0)
        p = torch.log_softmax(scores.float(), dim=0)
        return -(q * p).sum()

    def eval_ndcg(rows: list[dict], cap: int = 80) -> float:
        model.eval()
        tot = 0.0
        n = 0
        with torch.no_grad():
            for rec in rows[:cap]:
                s = yes_logits(rec["question"], rec["docs"])
                order = torch.argsort(s, descending=True).tolist()
                labels = rec["labels"]
                dcg = sum(labels[i] / math.log2(r + 2) for r, i in enumerate(order[:10]))
                ideal = sorted(labels, reverse=True)
                idcg = sum(g / math.log2(r + 2) for r, g in enumerate(ideal[:10]))
                if idcg > 0:
                    tot += dcg / idcg
                    n += 1
        model.train()
        return tot / max(n, 1)

    best = -1.0
    history: list[str] = []
    for epoch in range(1, EPOCHS + 1):
        random.shuffle(train_rows)
        running = 0.0
        steps = 0
        skipped = 0
        for rec in train_rows:
            opt.zero_grad(set_to_none=True)
            labs = rec["labels"]
            if max(labs) <= 0:
                skipped += 1
                continue
            scores = yes_logits(rec["question"], rec["docs"])
            labels = torch.tensor(labs, device=device)
            loss = listnet(scores, labels)
            if not torch.isfinite(loss.detach()):
                skipped += 1
                continue
            loss.backward()
            trainable = [p for p in model.parameters() if p.requires_grad]
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in trainable):
                opt.zero_grad(set_to_none=True)
                skipped += 1
                continue
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            running += float(loss.detach())
            steps += 1
            if steps == 1 or steps % 20 == 0:
                elapsed = time.monotonic() - t0
                print(
                    f"epoch {epoch} step {steps}/{len(train_rows)} "
                    f"loss={running / steps:.4f} skip={skipped} "
                    f"sec/step={elapsed / max(steps, 1):.2f}",
                    flush=True,
                )
        if steps < 50:
            print(f"too few finite steps ({steps}); skip={skipped}", file=sys.stderr)
            return 1
        dev = eval_ndcg(dev_rows)
        line = f"epoch {epoch}: loss={running / max(steps, 1):.4f} dev_list_ndcg10={dev:.3f}"
        history.append(line)
        print(line, flush=True)
        if dev >= best:
            best = dev
            CHECKPOINT.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(CHECKPOINT))
            tok.save_pretrained(str(CHECKPOINT))
            print(f"saved {CHECKPOINT}", flush=True)

    wall = (time.monotonic() - t0) / 3600.0
    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text(
        f"""# Qwen3-Reranker listwise

- lists: hybrid qwen3-0.6b + BM25, mined size={LIST_SIZE}, train keep={TRAIN_LIST_KEEP}
- unfreeze last {UNFREEZE_LAST} layers + lm_head; max_len={MAX_LEN}
- grades: turn=2, same-admission=1, other=0
- train={len(train_rows)} dev={len(dev_rows)}
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
