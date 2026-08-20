# Trained cross-encoder reranking (the optional retrieve-then-rerank second stage)

`src/medmemgraph/graph/reranker.py`

This is the algorithm-level explanation for a paper reviewer: what this
module actually computes, which evidence and which model-card equations
each design decision traces to, and — honestly — what this session's real
runs could and could not demonstrate. `documentor` weaves this into the
end-of-bundle README narrative; it does not rewrite this section (same
convention as `docs/algorithms/vector-lexical-index.md` /
`dense-lexical-baselines.md`).

## 1. Why this module exists, and why it is not wired into `retrieve()`

`graph/reranker.py` is deliberately **not imported by `retrieve.py` or
`router.py`**. The frozen `retrieve()` contract (`contracts.py` CONTRACT
2) is untouched by this file — nothing in this session's demo/eval path
changes shape or behavior because this module now exists. It exists so
`eval/`'s baseline ladder can run reranking as **one more arm of an
ablation** — `NoopReranker` vs. a trained cross-encoder — the same
"isolate the retrieval variable, measure it honestly" discipline
`eval/baselines/dense.py`/`lexical.py` already apply to chunk size and
`k`.

That framing matters because this project's own planning documents
(`.grok/plans/phases-medmemgraph-2026-08-16.md`, `research/GROK-
INTAKE.md`) list "LLM reranker" as something Graph "must not
redesign"/"do not build," citing a sibling survey's finding that a
**prompted, untrained**, reflection-style LLM rerank step measured
accuracy dropping 58.8% → 31.0%, below the naive baseline, recovering
only once fine-tuned
(`research/01-agent-memory-architectures.md`, claim AM-063). Read
literally, that ban is about the *prompted/untrained* failure mode, not
about reranking as a category, and this project's own dedicated
retrieval-engineering survey draws exactly that line:

> "A trained cross-encoder reranker (e.g. `bge-reranker-v2-m3`) is a
> defensible, low-risk *optional* second-stage add-on if time permits —
> a prompted/untrained LLM or off-the-shelf-domain reranking step is
> not, and should not be added."
> — `collaborative/literature/12-vector-search-and-hybrid-fusion.md` Q7

> "reranking is safe when it is a *trained, task-appropriate* model, and
> risky when it is either an untrained/prompted LLM step or an
> off-the-shelf model borrowed from a different domain (general web
> search) without validation on this project's own conversational-
> clinical data."
> — same survey, closing summary

So this module implements the *safe* half of that distinction — a
fine-tuned relevance-scoring model, no prompt, no free-form generation —
never the banned half, and it stays off the frozen retrieve path so it
cannot regress the graded pipeline even if that validation has not
happened yet. Two honest caveats the survey itself states, carried
forward here rather than smoothed over:

1. **"Trained" is not a blanket safety guarantee on this project's own
   data.** The survey's own positive evidence (R-VEC-078: cross-encoders
   competitive with GPT-4 listwise reranking in-domain; R-VEC-082:
   `bge-reranker-v2-m3` used as a named production baseline elsewhere) is
   about trained cross-encoders *in general*. A sibling, independent
   ablation in the same survey (R-VEC-080) found an off-the-shelf,
   non-domain-tuned web-search cross-encoder *hurting* conversational-
   memory Hit@1 by 6.9 points in one clean test. Neither model registered
   below has been validated on this project's own held-out
   MedLoCoMo/`benchmark_qa.json` slice — that validation is exactly what
   `NoopReranker` as a first-class ablation arm is *for* (§3), not a
   claim this module can make on its own.
2. **`literature/12` names `bge-reranker-v2-m3` specifically** as its
   evidenced pick; `Qwen/Qwen3-Reranker-0.6B` (this module's default,
   named explicitly by the story that dispatched this module) is not
   evaluated anywhere in that survey. It is registered because it is
   independently well-documented as a *trained* (not prompted)
   relevance-scoring model — the same structural safety argument applies
   — but it carries the same "not yet validated on our data" caveat.

Do not "simplify" this module into a prompted-LLM rerank step later.
That is the one failure mode every citation above measured as harmful,
and the one thing this module deliberately is not.

## 2. Two registered backbones, two different scoring mechanisms

`REGISTERED_MODELS` holds two entries, both independently verified
reachable on the Hub (`huggingface_hub.model_info`, run directly in this
sandbox before this module was written) and both already present in this
environment's local HF cache:

| key | HF id | `kind` | scoring mechanism |
|---|---|---|---|
| `qwen3-rerank-0.6b` (default) | `Qwen/Qwen3-Reranker-0.6B` | `causal_yesno` | next-token yes/no logits of a fine-tuned causal LM |
| `bge-rerank-v2-m3` | `BAAI/bge-reranker-v2-m3` | `seq_classification` | a trained regression head (standard cross-encoder) |

**`bge-rerank-v2-m3` is a standard cross-encoder.** It is an
`AutoModelForSequenceClassification` checkpoint (XLM-RoBERTa backbone) —
its own model card's "Huggingface Transformers" usage section shows the
forward pass producing a relevance logit directly:

```python
scores = model(**inputs, return_dict=True).logits.view(-1,).float()
```

`sentence_transformers.CrossEncoder` wraps exactly this model shape, so
`_SentenceTransformersCrossEncoderBackend` uses it directly, with
`activation_fn=torch.nn.Sigmoid()` at `predict()` time to map the raw
logit onto `[0, 1]` (the same normalization `FlagEmbedding`'s own
`compute_score(..., normalize=True)` documents applying) — purely for
score comparability with the other backbone's already-`[0, 1]`
yes-probability; sigmoid is monotonic, so it never changes the induced
ranking, only the reported score's scale.

**`qwen3-rerank-0.6b` is not a regression-head cross-encoder at all — it
is a fine-tuned causal LM scored on its own next-token logits.** This is
the part the dispatching story called out explicitly ("Qwen3-Reranker
uses a specific yes/no logit scoring format rather than a plain
regression head... scoring it as a generic cross-encoder produces
plausible-looking but wrong rankings"), and it is verified against the
model's own card (huggingface.co/Qwen/Qwen3-Reranker-0.6B, "Transformers
Usage" section, fetched 2026-08-16), quoted here in full since it is the
load-bearing artifact `_Qwen3YesNoBackend` reproduces line-for-line:

```python
# format_instruction
def format_instruction(instruction, query, doc):
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"

# tokenizer / model setup
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Reranker-0.6B", padding_side="left")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-Reranker-0.6B").eval()
token_false_id = tokenizer.convert_tokens_to_ids("no")
token_true_id = tokenizer.convert_tokens_to_ids("yes")

prefix = ("<|im_start|>system\nJudge whether the Document meets the requirements "
          "based on the Query and the Instruct provided. Note that the answer can "
          "only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)

# process_inputs
def process_inputs(pairs):
    inputs = tokenizer(pairs, padding=False, truncation="longest_first",
                        return_attention_mask=False,
                        max_length=max_length - len(prefix_tokens) - len(suffix_tokens))
    for i, ele in enumerate(inputs["input_ids"]):
        inputs["input_ids"][i] = prefix_tokens + ele + suffix_tokens
    inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)
    return inputs

# compute_logits
@torch.no_grad()
def compute_logits(inputs):
    batch_scores = model(**inputs).logits[:, -1, :]
    true_vector = batch_scores[:, token_true_id]
    false_vector = batch_scores[:, token_false_id]
    batch_scores = torch.stack([false_vector, true_vector], dim=1)
    batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
    return batch_scores[:, 1].exp().tolist()
```

Three things about this recipe that are easy to get subtly wrong (and
why `_Qwen3YesNoBackend` is careful about each):

- **`padding_side="left"` is load-bearing, not cosmetic.**
  `compute_logits` reads `logits[:, -1, :]` — the LAST sequence position
  — for every row in a batch. With right-padding, the last position of a
  shorter row would be a pad token, not the real final token, and the
  yes/no logits read off it would be meaningless. Left-padding keeps the
  real final token in the last position for every row regardless of that
  row's own length.
- **The score is a genuine probability, not a bare logit.** The final
  `log_softmax` is taken over exactly the two-element vector
  `{no_logit, yes_logit}`, then the yes-term is exponentiated. This is
  what makes `yes_prob + no_prob == 1` for each row — a bare
  `yes_logit` alone would not be comparable across rows the way a
  properly normalized probability is.
- **No `model.generate()` call anywhere.** The assistant turn is
  pre-seeded with an empty `<think>\n\n</think>\n\n` block precisely so
  the model is scored on the logits at that *exact* next position — one
  forward pass, no decoding loop. This is also why this is correctly
  described as a *trained relevance classifier accessed through a causal
  LM's weights*, not "prompting an LLM to rank documents" — there is no
  free-form generation step for a prompt-injection/format-drift failure
  mode to live in.

Loading `Qwen/Qwen3-Reranker-0.6B` with `AutoModelForSequenceClassification`
(or any generic pooled-cross-encoder wrapper that assumes a classification
head) would attach a **randomly initialized** head to a model that was
never trained with one — the forward pass would run without error and
produce plausible-*looking* scores that are pure noise. This is the exact
trap the dispatching story's "read the model card and implement it
correctly" line warns about, and it is why this module has two backend
classes rather than one generic `AutoModelForSequenceClassification`
path.

## 3. `NoopReranker` — "no rerank" as a first-class ablation arm

`NoopReranker.rerank(query, docs, top_k)` returns `docs`' input order,
unscored (technically: strictly-decreasing synthetic scores, so a caller
that re-sorts by score reconstructs the exact input order too — see the
class docstring). It implements the same `RerankerBackend` protocol as
`CrossEncoderReranker`, so `two_stage_retrieve` (§4) never special-cases
it. This is a direct response to §1's honest caveat: the only way to
know, for *this* project's own data, whether a trained cross-encoder
helps or hurts is to run the identical pipeline with and without it and
compare — exactly the validation `literature/12`'s own Q7 recommendation
calls for before shipping reranking at all. Without `NoopReranker` as a
peer arm (rather than a caller-side `if reranker: ...` branch), that
comparison would be two different code paths, not one controlled
variable.

## 4. Two-stage retrieval: bi-encoder top-N, then rerank to top-k, latency recorded separately

`two_stage_retrieve(query, retrieve_fn=..., reranker=..., n_candidates=N,
top_k=k)` is the "cheap retriever fetches candidates, expensive one
reranks" sequence `ai-engineering` ch06's own hybrid-search framing
names directly, made measurable: it calls `retrieve_fn(query,
n_candidates)` (any bi-encoder/lexical/fused first stage with that
shape — not hardcoded to one channel), times it, then calls
`reranker.rerank(query, [item.text for item in candidates], top_k=top_k)`,
timing that separately. The returned `TwoStageResult` carries
`retrieve_latency_ms` and `rerank_latency_ms` as two independent numbers
— the story's own stated reason: "reranking is the expensive stage and
the Pareto claim depends on knowing its cost." §5 below shows exactly how
expensive, measured directly in this environment.

Reranked items keep their original `channel`/`session_id`/`turn_ids`
(`contracts.VALID_CHANNELS` is frozen and has no "reranked" value, so the
bi-encoder's own channel label is the honest one to keep — the same
convention `fusion.rrf_fuse` already uses for its own re-scored output);
only `.score` is overwritten, to the reranker's own score.

## 5. Honest residual

- **Neither registered model has been validated on this project's own
  held-out data yet** (§1 caveat 1) — this module makes that validation
  *possible* (`NoopReranker` as a controlled peer arm, `two_stage_retrieve`
  reporting both stages' cost) but does not itself run or report that
  comparison against `benchmark_qa.json`'s gold evidence. That is a
  follow-on ablation for `eval/`'s baseline ladder, not this story's own
  scope.
- **`qwen3-rerank-0.6b` (the default) is unevaluated in `literature/12`;
  `bge-rerank-v2-m3` is the survey's own evidenced pick** (§1 caveat 2).
  Both are registered and both are exercised end-to-end by
  `tests/test_reranker.py`'s real-model tier; picking a shipped default
  between them, if either is shipped at all, is a decision for whoever
  runs the held-out-data validation above, not one this module makes for
  them.
- **fp16 batching is not bit-identical to fp32 batching, and that is
  expected, not a bug.** Measured directly while writing
  `tests/test_reranker.py`: scoring the same 6 documents in one fp16
  batch vs. one-at-a-time in fp16 produced score deltas up to ~1.7e-3
  (different per-batch padding width changes the accumulated rounding in
  fp16 attention/matmul kernels); the identical comparison in fp32
  dropped that delta to ~1e-6-1e-7. The batching-invariance test
  (`test_batching_does_not_change_scores`) therefore runs in fp32
  specifically, to isolate "is the batching logic correct" (no
  misaligned indices, correct left-padding, no cross-row contamination)
  from "what does fp16 cost in precision" (a separate, known, accepted
  trade-off — this project runs fp16 by default for latency/VRAM,
  per §5 below and the CUDA hardware this project's memory already
  verifies).
- **No query-specific instruction tuning was attempted** for
  `qwen3-rerank-0.6b` beyond a single fixed clinical-domain instruction
  string (`RerankerModelSpec.instruction`); the model card notes
  "tailored instructions specific to their tasks and scenarios" typically
  add 1-5% — a real, cheap, unexplored lever for whoever runs the
  held-out validation above.

## 6. Real numbers

Measured directly in this environment (RTX 3090, 24 GB VRAM, torch
2.13.0+cu130, both models loaded fp16 on CUDA — see the dev-ml return
note / `.claude/logs/dev.log.md` entry for this story for the exact
commands):

100 synthetic clinical-note-style candidate documents, `top_k=10`, mean
of 5 timed calls after one untimed warm-up call (load time excluded):

| reranker | load time | VRAM after load | peak VRAM during rerank | rerank(100 docs) mean latency |
|---|---|---|---|---|
| `NoopReranker` | 0 s | 0 GB | 0 GB | 0.002 ms |
| `qwen3-rerank-0.6b` | 4.73 s | 1.20 GB | 2.80 GB | 442.7 ms |
| `bge-rerank-v2-m3` | 3.04 s | 1.14 GB | 1.17 GB | 81.7 ms |

Both trained backbones correctly promoted the hand-built, deliberately
paraphrased correct document (semantically about "hypertension"/
"lisinopril" but sharing almost no literal tokens with a "blood
pressure"/"medication"-worded query) from last-of-5 under naive
keyword-overlap ranking to first-of-5 — see §7/`tests/test_reranker.py`
for the exact case and both models' real scores.

This is the direct, measured evidence for the story's own warning that
"reranking is the expensive stage": `qwen3-rerank-0.6b` costs roughly
five orders of magnitude more wall-clock time than `NoopReranker` on the
same 100-candidate pool, and `bge-rerank-v2-m3` — smaller and a
regression head rather than a full causal-LM forward pass — costs about
5.4x less than `qwen3-rerank-0.6b` for the same job. Any Pareto claim
that includes a reranked arm must charge this cost honestly against that
arm's accuracy gain, not report accuracy alone.

## 7. Verification (original story)

`uv run pytest tests/test_reranker.py -v` — 20 tests, all passing, ~10-12s
wall clock in this environment (dominated by two real model loads, `qwen3-
rerank-0.6b` and `bge-reranker-v2-m3`, each a few seconds). See the
dev-ml return note / `.claude/logs/dev.log.md` entry for this story for
the pasted real output, the hand-built promotion case's real scores from
both backbones, and the §6 latency/VRAM table's raw numbers.

## 8. CPU-optimized registry extension (this story)

### 8.1 Why — a real, measured deployment gap

The live demo has to run on a CPU-only AWS EC2 box (2 vCPU / 4 GB,
`t3.medium` class) — GPU instances are out on cost. Both backbones §2–§6
document are GPU-class: `bge-rerank-v2-m3` measured **13.65 s** for a
100-candidate rerank on a 2-thread CPU (project measurement, not this
session's); `qwen3-rerank-0.6b`, a full causal-LM forward pass, is slower
still per candidate. 13.65 s per query is not a product; the smallest
known-good CPU number in hand before this story
(`cross-encoder/ms-marco-MiniLM-L-6-v2`, 0.50 s/100 docs) was unvalidated
for quality on this project's clinical-dialogue data — MiniLM-L-6-v2 is
23M params, English-only, MS MARCO-trained, and assuming it transfers to
clinical conversation without checking is the same category of error as
assuming a GPU latency number transfers to CPU. This story adds seven
CPU-appropriate arms spanning that unmeasured quality/latency/memory
frontier.

### 8.2 The seven new arms, verified before registration

Every candidate's Hub id was independently verified reachable
(`huggingface_hub.model_info`, run directly in this sandbox) before it was
added — all six candidates named in the story, plus the pre-existing
checkpoint reused for the ONNX arm, were reachable; none had to be
reported missing. Each `config.json` was then read directly (not assumed)
to confirm architecture and head shape:

| key | HF id | architecture | `id2label` | params (exact, see §8.3) |
|---|---|---|---|---|
| `ms-marco-tinybert-l2-v2` | `cross-encoder/ms-marco-TinyBERT-L-2-v2` | `BertForSequenceClassification` | `{"0": "LABEL_0"}` | 4,386,561 |
| `ms-marco-minilm-l6-v2` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | `BertForSequenceClassification` | `{"0": "LABEL_0"}` | 22,714,113 |
| `ms-marco-minilm-l6-v2-onnx-int8` | *(same repo, `onnx/model_qint8_avx512.onnx`)* | *(same, ONNX-exported, int8)* | — | 22,714,113 |
| `ms-marco-minilm-l12-v2` | `cross-encoder/ms-marco-MiniLM-L-12-v2` | `BertForSequenceClassification` | `{"0": "LABEL_0"}` | 33,360,897 |
| `jina-rerank-v1-turbo-en` | `jinaai/jina-reranker-v1-turbo-en` | `JinaBertForSequenceClassification` (custom, `trust_remote_code`) | *(none declared)* | 37,771,777 |
| `mxbai-rerank-xsmall-v1` | `mixedbread-ai/mxbai-rerank-xsmall-v1` | `DebertaV2ForSequenceClassification` | `{"0": "LABEL_0"}` | 70,830,337 |
| `bge-rerank-base` | `BAAI/bge-reranker-base` | `XLMRobertaForSequenceClassification` | `{"0": "LABEL_0"}` | 278,044,931 |

Every one of the six library-built-in architectures resolves to a
**genuine single-label (`num_labels=1`) trained regression head** — the
same "real cross-encoder head, no yes/no-token trick" shape as
`bge-rerank-v2-m3` §2 — so all seven route through the existing
`_SentenceTransformersCrossEncoderBackend`; none needed a new `kind`. The
one architectural wrinkle, `jina-rerank-v1-turbo-en`, ships a custom
`modeling_bert.py` referenced by its own `config.json` `auto_map`; its
model card's own `sentence-transformers` usage example passes
`trust_remote_code=True` to `CrossEncoder(...)` explicitly
(huggingface.co/jinaai/jina-reranker-v1-turbo-en, "Usage" §2, fetched
2026-08-17) — reproduced via `RerankerModelSpec.trust_remote_code`.

### 8.3 `params` — exact, not a rounded model-card figure

`RerankerModelSpec.params` is the sum of `huggingface_hub.
get_safetensors_metadata(hf_id).parameter_count` across dtypes — this
reads only the small JSON header a safetensors file carries, never
downloads the weight tensors themselves, and is therefore both fast and
exact (not the model card's rounded prose number). Run directly in this
sandbox for all nine registered entries (including the two pre-existing
ones, backfilled by this story):

| key | params |
|---|---|
| `ms-marco-tinybert-l2-v2` | 4,386,561 |
| `ms-marco-minilm-l6-v2` (+ its onnx-int8 twin) | 22,714,113 |
| `ms-marco-minilm-l12-v2` | 33,360,897 |
| `jina-rerank-v1-turbo-en` | 37,771,777 |
| `mxbai-rerank-xsmall-v1` | 70,830,337 |
| `bge-rerank-base` | 278,044,931 |
| `bge-rerank-v2-m3` | 567,755,777 |
| `qwen3-rerank-0.6b` | 595,776,512 |

The ablation spans the requested ~4M–568M range and beyond (the
pre-existing 596M default stays registered as the upper anchor).

### 8.4 `approx_ram_mb` — weights-only, and the one place that convention
### had to be broken

Every `backend="torch"` entry's `approx_ram_mb` is `params * 4 bytes /
1e6` (fp32 — `CrossEncoderReranker._ensure_loaded` already forces
`fp16=False` off CUDA, unchanged by this story). The one `backend="onnx"`
entry is `params * 1 byte / 1e6` (int8). This is a **weights-only** floor
— no activation buffers, no framework overhead — stated as such in the
field's own docstring so a reader sizing the 4 GB box against it knows
what it is and is not.

**One entry breaks that convention deliberately, because keeping it would
have been dishonest:** measured directly in this sandbox
(`transformers.AutoModelForSequenceClassification.from_pretrained(...,
trust_remote_code=True)`, then `m.named_buffers()`), `jina-rerank-v1-turbo-en`
allocates a single non-parameter buffer, `bert.encoder.alibi`, shape
`(1, 12, 8192, 8192)` float32 — **3,221,225,472 bytes, ~3.22 GB** — at
model-construction time, sized for the model's full 8192-token max
context regardless of the actual input length used. Isolated directly:
RSS jumped ~6.7 GB during `CrossEncoder(...)` construction itself in a
clean subprocess, not during `.predict()`; the model's own real trained
weights are ~151 MB (confirmed via `sum(p.numel()*p.element_size() for p
in m.parameters())`), so the ALiBi buffer alone is **~21x** the model's
own weight footprint, and alone exceeds this story's entire 4 GB target.
`jina-rerank-v1-turbo-en`'s `approx_ram_mb` is therefore set to the
measured real total (151.1 + 3221.4 ≈ **3372.5 MB**), not the naive
`params * 4` figure (151.1 MB) every other entry uses — see the registry
entry's own inline comment and `test_reranker.py::
test_jina_ram_estimate_reflects_the_measured_alibi_buffer_not_just_weights`,
which guards against a future edit silently reverting this back to the
misleading formula.

**Practical read:** despite being one of the smaller candidates by
parameter count (37.8M, between `ms-marco-minilm-l12-v2` and
`mxbai-rerank-xsmall-v1`), `jina-rerank-v1-turbo-en` is the single worst
fit for this story's 4 GB CPU box of everything registered — a finding
the params column alone would have completely hidden, and exactly the
kind of thing this ablation exists to catch before it becomes a demo-day
surprise.

### 8.5 The ONNX/int8 arm — cheap because the maintainers already did the export

`cross-encoder/ms-marco-MiniLM-L-6-v2`'s own Hub repo already ships
several pre-exported ONNX graphs, including a maintainer-produced,
AVX512-tuned int8-quantized one, `onnx/model_qint8_avx512.onnx`
(`huggingface_hub.list_repo_files`, verified directly in this sandbox
before this entry was registered). `ms-marco-minilm-l6-v2-onnx-int8`
simply points `sentence_transformers.CrossEncoder`'s own native
`backend="onnx"` support (shipped in this project's pinned
`sentence-transformers>=5.7.0`, backed by `optimum[onnxruntime]`, added to
`pyproject.toml` for this story) at that file via
`model_kwargs={"file_name": ...}`. No custom `optimum.onnxruntime` export
script, no local PTQ pass was needed — this is exactly the "if it is cheap
to do" case the story anticipated, and it was: one registry entry, no new
backend class, no new `kind`.

Measured directly in this sandbox, `torch.set_num_threads(2)` (this
story's 2 vCPU deployment target), 100 synthetic clinical-note-style
candidate pairs, one untimed warm-up call excluded:

| variant | mean latency, 100 docs |
|---|---|
| `ms-marco-minilm-l6-v2` (native fp32 torch) | 0.423 s |
| `ms-marco-minilm-l6-v2-onnx-int8` | 0.124 s |

(Same figures as §8.6's table below, restated here for the direct
torch-vs-onnx comparison; §8.6 documents the exact measurement
methodology once for all seven arms.)

A ~3.4x wall-clock speedup on identical hardware, on top of the ~4x smaller
weights-only memory footprint (90.9 MB fp32 vs. 22.7 MB int8 — §8.4). On
a 4 GB box where `bge-rerank-v2-m3` alone (fp32, ~2.27 GB) already
consumes most of the budget, the int8 MiniLM-L6 arm is the only registered
entry here that comfortably coexists with the rest of this project's
CPU-resident process (embedder + lexical index + Python/torch runtime
overhead) with real headroom to spare — quantization is a **memory-fit
lever here, not only a speed one**.

### 8.6 Real numbers — load time, RSS, and 100-candidate latency, all seven arms

Measured directly in this sandbox: `torch.set_num_threads(2)`, each model
loaded and measured in its own clean subprocess (isolates one model's RSS
from another's — a shared process would over/under-attribute memory
across models), `device="cpu"` explicit (not "cuda if available" — this
sandbox has a GPU, but the CPU path is what this story exists to
measure), 100 synthetic candidate pairs, `top_k=10`, mean of 5 timed calls
after one untimed warm-up call (`resource.getrusage(...).ru_maxrss`, KB→MB,
Linux):

| key | load time | RSS delta at load | mean latency, 100 docs |
|---|---|---|---|
| `ms-marco-tinybert-l2-v2` | 4.84 s | 283.6 MB | 0.031 s |
| `ms-marco-minilm-l6-v2-onnx-int8` | 5.49 s | 332.9 MB | 0.124 s |
| `ms-marco-minilm-l6-v2` | 4.46 s | 329.0 MB | 0.423 s |
| `ms-marco-minilm-l12-v2` | 4.71 s | 370.1 MB | 0.874 s |
| `jina-rerank-v1-turbo-en` | 7.02 s | **6946.3 MB** | 0.596 s |
| `mxbai-rerank-xsmall-v1` | 4.34 s | 668.0 MB | 1.866 s |
| `bge-rerank-base` | 4.64 s | 904.9 MB | 3.061 s |

("RSS delta at load" is whole-process peak-RSS growth from just before
construction to just after the lazy load completes — necessarily larger
than §8.4's weights-only `approx_ram_mb` for every model, since it also
counts the tokenizer, the `transformers`/`sentence-transformers` runtime
objects, and (for `jina-rerank-v1-turbo-en` specifically) the §8.4 ALiBi
buffer; the two numbers are not meant to match exactly, but the *jina*
row's order-of-magnitude gap from every other row's RSS delta is the same
finding as §8.4, independently confirmed via a second measurement method.)

`ms-marco-minilm-l6-v2-onnx-int8`'s **latency ranks 2nd-fastest** of all
seven (behind only the 4.4M-param TinyBERT floor) while its **RSS delta
ranks among the smallest** — the two-lever case (quantization for both
speed and memory) the story anticipated, confirmed.

### 8.7 An honest quality signal this session's own measurement caught

Story acceptance only required "each registered model loads, scores,
respects `top_k`" (`tests/test_reranker.py::TestCpuAblationModelsReal`) —
not that every arm passes the §5/§6 hand-built promotion case
(`PROMOTION_QUERY`/`PROMOTION_DOCS`, the "lisinopril for hypertension"
item worded to share almost no literal tokens with the query). Run
against all seven anyway, for the same paraphrase-robustness signal §6
reports for the two GPU-class backbones:

| key | promotes the correct doc to rank 1? | top score for the correct doc |
|---|---|---|
| `bge-rerank-base` | **yes** | 0.997 |
| `jina-rerank-v1-turbo-en` | **yes** | 0.549 |
| `mxbai-rerank-xsmall-v1` | no (0.770 vs. 0.752 — narrowly ranked 2nd) | 0.752 |
| `ms-marco-minilm-l12-v2` | no (ranks a high-literal-overlap distractor 1st, 0.967 vs. 0.954) | 0.954 |
| `ms-marco-minilm-l6-v2` | no (0.971 vs. 0.755) | 0.755 |
| `ms-marco-minilm-l6-v2-onnx-int8` | no (0.972 vs. 0.717) | 0.717 |
| `ms-marco-tinybert-l2-v2` | no (0.993 vs. effectively last) | ~0.0002 |

Only `bge-rerank-base` and `jina-rerank-v1-turbo-en` (both larger,
non-MS-MARCO-only training) promote the semantically-correct,
lexically-dissimilar document past a high-literal-overlap distractor on
this one hand-built case; every MS-MARCO-trained arm (`ms-marco-*`,
regardless of size) and `mxbai-rerank-xsmall-v1` rank the distractor
first instead. This is **one illustrative case, not a held-out-data
evaluation** — the same §1/§5 caveat the original two backbones carry
applies here with the same force — but it is a real, measured signal that
smaller/weaker MS-MARCO-only cross-encoders may lean harder on literal
token overlap than semantic match, worth weighing against their latency
advantage in whatever held-out ablation run comes next. It is exactly why
this story's tests do not assert a per-model promotion requirement (see
`TestCpuAblationModelsReal`'s own comment) — asserting it would have
required either quietly dropping five of seven models' coverage or baking
a known-false claim into the suite.

### 8.8 Honest residual (this story)

- **None of the nine registered models has been validated on this
  project's own held-out clinical-dialogue data** — same caveat as §1/§5,
  now covering seven more arms. §8.7's promotion case is one illustrative
  signal, not that validation.
- **`jina-rerank-v1-turbo-en` is registered but not recommended for the 4
  GB deployment target** — §8.4's ALiBi-buffer finding disqualifies it on
  memory alone, independent of its (otherwise reasonable, §8.7) quality
  and its fast §8.6 latency. It stays registered because the story asked
  for reachable-and-correct registration, not a memory-based exclusion
  list, and because the finding itself is only useful if the model stays
  inspectable through the same registry as everything else.
- **The ONNX/int8 arm exists for exactly one base model
  (`ms-marco-minilm-l6-v2`)**, chosen as "the winner-so-far" because it
  was the one CPU-oriented candidate with a real measured latency number
  already in hand before this story (0.50 s/100 docs, a different
  machine) — not because it is this session's own measured best on either
  §8.6 latency or §8.7 quality (`ms-marco-tinybert-l2-v2` is faster;
  `bge-rerank-base` is the only MS-MARCO-free promotion pass). A follow-on
  ablation with real per-query gold labels, not this story's one
  illustrative case, is what should actually pick a shipped default.
- **§8.6's RSS-delta numbers are whole-process measurements** (tokenizer,
  runtime objects, and — for `jina-rerank-v1-turbo-en` — its ALiBi
  buffer, all included), not the `approx_ram_mb` field's weights-only
  convention; the two are cross-checked against each other in §8.6's own
  note, not asserted identical.

## 8a. Finetuned MiniLM arms (eval-only, 2026-08-17)

Two **new** registry keys, added after a patient-level gold+hardneg
finetune on MedLoCoMo train patients (eval trio
`10056223` / `10213338` / `10312715` held out):

- `ms-marco-minilm-l6-v2-ft-medlocomo` — local torch checkpoint
- `ms-marco-minilm-l6-v2-ft-medlocomo-onnx-int8` — locally exported
  int8 ONNX (`onnx/model_qint8_avx512.onnx` under the export dir, **not**
  the Hub-shipped un-finetuned graph)

The original `ms-marco-minilm-l6-v2-onnx-int8` arm is unchanged. These
keys are **eval-only**. This module is still **not imported by
`retrieve.py`**. Wiring a CE arm into `retrieve()` / demo defaults is out
until an explicit post-freeze go.

A second student objective, **CE-ORPO** (`…-ft-orpo` / `…-ft-orpo-onnx-int8`),
is the same MiniLM with L = −logσ(s_w) + λ −logσ(s_w−s_l) on
(gold, other-admission hard-neg) triples. Best measured turn Hit@10 on
the eval trio is **0.874** (unfinetuned 0.810, pointwise-BCE FT 0.866,
GPU 0.900). The 0.88 bar was not met. Hit@2 / nDCG recovered vs BCE.
Admission Hit@10 is still below the unfinetuned MiniLM. See
`results/finetune-reranker/close_the_gap.md`.

## 9. Verification (this story)

`uv run pytest tests/test_reranker.py -v` — 38 tests, all passing, ~25-27s
wall clock in this environment (dominated by nine real model loads across
Tiers 2 and 3). `-m "not slow"` skips the 21 load-heavy CPU-ablation tests
(`TestCpuAblationModelsReal`, parametrized over all seven §8.2 arms),
dropping to ~5-6s for a fast dev loop. See the dev-ml return note /
`.claude/logs/dev.log.md` entry for this story for the pasted real output
and the raw numbers behind every §8 table.
