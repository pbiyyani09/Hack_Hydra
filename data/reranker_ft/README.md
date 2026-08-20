# Finetuned MiniLM reranker weights

Int8 ONNX cross-encoder checkpoints, finetuned from
`cross-encoder/ms-marco-MiniLM-L6-v2` on MedLoCoMo train patients to
improve retrieval nDCG.

`data/` is gitignored repo-wide (see `.gitignore`, "Local data — fetched
at setup, never vendored"). These files are the one deliberate
exception, tracked via `git add -f`: they are model weights, not corpus
data, and the registry keys in `graph/reranker.py` resolve to these
paths, so the `-onnx-int8` arms work on a fresh clone with no download
step.

## What is here

Four arms, each a `sentence_transformers.CrossEncoder` directory loaded
with `backend="onnx"` and `file_name="onnx/model_qint8_avx512.onnx"`:

| directory | registry key | objective |
|---|---|---|
| `…-ft-listwise-onnx` | `ms-marco-minilm-l6-v2-ft-listwise-onnx-int8` | listwise |
| `…-ft-orpo-onnx` | `ms-marco-minilm-l6-v2-ft-orpo-onnx-int8` | CE-ORPO |
| `…-ft-orpo-turn-onnx` | `ms-marco-minilm-l6-v2-ft-orpo-turn-onnx-int8` | CE-ORPO, turn grain |
| `…-ft-medlocomo-onnx` | `ms-marco-minilm-l6-v2-ft-medlocomo-onnx-int8` | pointwise BCE |

22.0 MiB of graph per arm (22.7 MB RAM, 22.7 M params), 95 MB total.

**`…-ft-listwise-onnx` is the best arm.** The only apples-to-apples
comparison is `same_pool_audit` — one pool built once (arctic-s + BM25
hybrid, 100 candidates), all three rerankers looped over that identical
pool, turn grain, n=231:

| reranker | Hit@2 | nDCG@10 |
|---|---|---|
| `…-ft-listwise-onnx-int8` | **0.948** | **0.756** |
| `qwen3-rerank-0.6b-ft-listwise` | 0.892 | 0.702 |
| `qwen3-rerank-0.6b` (stock) | 0.701 | 0.556 |

The higher **0.778** figure is the same arm at pool **200**. No Qwen arm
was ever run at pool 200, and widening the pool 100 -> 200 is itself
worth about +0.021, so 0.778 must NOT be compared against any Qwen
number — use 0.756 for that.

Note there is **no same-pool unfinetuned-MiniLM control**. What
finetuning MiniLM bought over stock MiniLM is therefore unmeasured on
this pool; the 0.461 stock figure in `ndcg_fixed.md` is a dense-only,
non-hybrid pool and is not comparable.

Full sweep in `results/finetune-reranker/` (gitignored — local only).

## What is not here, and why

- **fp32 ONNX** (`_fp32/model.onnx`, 87 MB/arm) — int8 is the deploy
  target; the fp32 graph is an export intermediate.
- **`onnx/model_quantized.onnx`** — verified byte-identical to
  `model_qint8_avx512.onnx` (same md5), so it is a pure duplicate.
- **Torch checkpoints** (`model.safetensors`, 88 MB/arm) — the four
  `ms-marco-minilm-l6-v2-ft-*` (non-`-onnx`) registry keys point at
  these and will **not** resolve on a fresh clone. Rerun the training
  scripts in `scripts/` to regenerate them.
  Note: the torch listwise directory was overwritten by a later packing
  epoch and no longer corresponds to `…-ft-listwise-onnx`. The ONNX
  graph here is the pre-pack checkpoint that produced 0.778 — it is the
  artifact the number belongs to.
- **`qwen3-rerank-0.6b-ft-listwise`** (2.3 GB) — a single file well over
  GitHub's 100 MB per-file limit. Local only.
- **Training data** (`*_pairs.jsonl`, `gold_*.jsonl`,
  `listwise_*.jsonl`, ~1.2 GB) — built from MedLoCoMo; regenerate with
  `scripts/build_rerank_pairs.py` and `scripts/extract_rerank_gold.py`.

## Reproducing

Patient-level split, eval trio `10056223` / `10213338` / `10312715` held
out of training. Train, export and sweep:

```
uv run python scripts/extract_rerank_gold.py
uv run python scripts/build_rerank_pairs.py
uv run python scripts/finetune_minilm_listwise.py
uv run python scripts/export_ft_minilm_onnx_int8.py
uv run python scripts/run_ft_rerank_sweep.py
```

These arms are **eval-only**: `retrieve.py` does not import
`graph/reranker.py`, so nothing here is on the serving path.
