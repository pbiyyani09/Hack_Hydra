# Vector and lexical per-patient indexes (the "beside HydraDB" arm)

`src/medmemgraph/graph/vector_index.py`, `src/medmemgraph/graph/lexical.py`

This is the algorithm-level explanation for a paper reviewer: what these two
modules actually compute, which paper/finding each design decision traces
to, and — honestly — what this session's real-corpus run could and could
not demonstrate. `documentor` weaves this into the end-of-bundle README
narrative; it does not rewrite this section (same convention as
`docs/algorithms/dense-lexical-baselines.md`).

## 1. Why this arm exists at all

HydraDB OSS ships no vector index and no full-text/BM25 index — confirmed
absent from the engine by exhaustive grep over
`collaborative/literature/10-hydradb-capability-audit.md`'s capability
table. Retrieval that isn't a graph traversal has to live somewhere; per
`ARCHITECTURE.md` §7.4 ("Vector / lexical (Graph, beside the engine)") it
lives in-process, next to the engine, not inside it. `vector_index.py` and
`lexical.py` are that layer: one small NumPy array and one small `bm25s`
index per `patient_id`, held in the Python process, queried with a Python
function call, never a network round trip to HydraDB.

Both modules expose the same shape — `build(...)`, `search(query, k)`,
`save(index_dir)`, `load(patient_id, index_dir)` — so a router (E5-S4, not
this story) can treat the two channels uniformly and fuse their results
with Reciprocal Rank Fusion (`ARCHITECTURE.md` §7.5) without a channel-
specific code path. Both return `contracts.RetrieveItem` directly rather
than a second, field-identical dataclass (`E5-S2`'s own sketch names a
`TextHit` type with the exact same five fields `RetrieveItem` already has:
`text`, `session_id`, `turn_ids`, `score`, `channel`) — reusing the frozen
contract instead of forking a parallel one it would have to be kept in
sync with by hand.

## 2. Chunking unit: turn-level for raw dialogue, plus a second, fact-level
granularity

`literature/12-vector-search-and-hybrid-fusion.md` Q4 converges on this
from three independent angles: (a) Dense X Retrieval's own honest
counter-finding that proposition-level retrieval is *not* a universal
win — it underperforms passage-level retrieval in-domain, and its
advantage concentrates on generalization/long-tail cases [R-VEC-054/055]
— exactly the "sometimes the raw turn has context the isolated fact lost,
sometimes the atomic fact is precisely what's needed" profile; (b) SeCom's
finding that every tested memory-unit granularity (turn/session/summary)
has real limitations on its own [R-VEC-058]; (c) this project's own
entity-resolution survey's finding that fact-augmented retrieval keys
improve recall@k by 9.4 points and downstream QA accuracy by 5.4 points
over flat indexing. `ARCHITECTURE.md` §7.4 states the resulting design
directly: "Index both turn text and emitted fact text."

`vector_index.PatientIndex.build(conversation, facts=None)` implements
exactly that: every turn in `conversation` is indexed at turn granularity
(`IndexedUnit.kind == "turn"`), and if `facts` (a list of
`contracts.ClinicalFact`) is supplied, each fact's rendered text is
indexed alongside the turns in the *same* per-patient array
(`IndexedUnit.kind == "fact"`). `facts` is an explicit, optional argument
rather than something `build()` extracts itself — wiring live extraction
in here would couple this module to the Pipeline track's LLM-backed
extractor (E2-S1), which this story does not own and which needs an API
key this environment does not always have. Callers that have
post-extraction facts pass them; callers that only have raw dialogue (this
story's own tests, and any early-pipeline caller) still get a fully
working turn-level index.

For the raw-content granularity specifically, both channels index at
**turn level**, not a pre-baked sliding window: literature/12 Q4's own
words are "turn-level indexing *with* neighbor-expansion at retrieval
time, rather than pre-baking large fixed windows into the index" — the one
design ReFind's own ablation actually measured (see §4 below). Dense
retrieval does not additionally apply neighbor expansion at query time —
`ARCHITECTURE.md` §7.4's "Dense" bullet says nothing about a window, and
ReFind's ablation is specifically about *lexical* hits, not dense chunk
neighborhoods — so `PatientIndex.search` returns exactly the matched turn
(or fact), and `LexicalIndex.search` is the one that expands (§4).

## 3. Vector channel: brute-force cosine, deliberately no ANN library

`literature/12` Q1: FAISS's own index-selection guidance places the
approximate-search crossover at roughly 1M vectors [R-VEC-007/008]. This
project's per-patient corpora are 2-4 orders of magnitude below that. The
survey's own directly-measured benchmark on comparable CPU hardware (AMD
Ryzen 9 5900X) found brute-force cosine over 3,000 vectors at 1,024
dimensions took 0.168 ms/query, and even 50,000 vectors (an unrealistic
upper bound — the whole 100-patient corpus pooled) took 7.25 ms/query
[R-VEC-009/010]. `PatientIndex` therefore holds one small
`(n_units, dim)` L2-normalized `float32` NumPy array per patient and
scores every query by `self.vectors @ q` (both sides normalized, so the
dot product *is* cosine similarity, per `ARCHITECTURE.md` §7.4 and
literature/12 Q2's own primary recommendation: "plain NumPy brute-force
cosine ... one small in-memory array per patient"). No FAISS, no HNSW, no
ANN library appears anywhere in `pyproject.toml` (`E5-S2` AC4, enforced by
`test_channels.py::test_pyproject_has_no_ann_library`) — building an ANN
index at this scale would add a tuning surface and a recall/latency
trade-off that measurement shows does not need to exist.

**Original session's measurement, one real MedLoCoMo patient (`10056223`,
1,451 turns across 27 admissions, dim=512, deterministic hashing-trick
encoder):** build time 0.10 s, on-disk-equivalent index size ~3.05 MB
(`3,195,909` bytes), p50 query latency 2.0-3.0 ms, p90 3.0-3.6 ms, over 30
real benchmark questions from that patient's own `benchmark_qa.json` (see
§6 for the full numbers and how they were produced). The absolute latency
was higher than literature/12's own 0.168 ms figure for a comparable
vector count because that measurement included Python-level `embed()` call
overhead for the query text on every search (the survey's number is pure
NumPy dot-product cost); the qualitative conclusion — sub-5-ms brute-force
*search* at this project's real per-patient scale, no ANN needed —
replicates directly and still holds after §4's encoder swap below (the
NumPy cosine math this section describes is completely unchanged; only
the *encoder* producing the vectors that math runs over changed).
**Re-measured with the new default encoder (`qwen3-0.6b`, dim=1024, same
patient) in the pluggable-embedder story:** build time 3.40 s, on-disk-
equivalent index size 6,167,557 bytes, p50 query latency 240-241 ms, p90
244-245 ms, same 30 questions — see the new §4 below for why per-query
latency rose by roughly two orders of magnitude (a real GPU forward pass
per query vs. a hash lookup) and why this is still the right trade for
this project's own retrieval-quality goal.

## 4. Embedding model: a pluggable local backend, general-purpose by
construction, not a clinical model, and — as of this story — the real
thing rather than a hashing-trick placeholder

`literature/12` Q3's single most load-bearing piece of evidence is a
head-to-head ablation on real EHR retrieval tasks: a general-domain 335M
model (BGE-large-en) beat every medical-specific model tested, including
an 8.9B clinical model (Gatortron-large) that scored *worst* of all seven
models and even underperformed a random-chunk-ordering baseline on one
task [R-VEC-032/033/034]. The survey's own recommendation is a strong
general-purpose model as the primary path — hosted *or* local, whichever
is available. Earlier sessions (`collaborative/design/stories/E4/E4-S2.md`)
had no local-embedding library wired in yet and settled for a
deterministic hashing-trick default instead, explicitly flagged in this
document's own §7 as a genuine trade-off, not a literature-first-choice
pick. **This story resolves that gap**: `sentence-transformers` and
`torch` are now project dependencies (`pyproject.toml`), and
`PatientIndex`'s default encoder is a real, general-purpose, *local*
sentence-embedding model — free (no per-call cost, no network dependency
once the model weights are cached), and, per the ablation above, a better
retrieval signal than either the hashing trick or a clinical-specialized
model would be.

### 4.1 Architecture: `graph/embedders.py` owns the encoder, `PatientIndex`
still owns the index

`graph/embedders.py` is a new module, separate from this file, that
exposes:

- **`EmbedderBackend`** — the protocol every encoder in this module
  satisfies: `encode(texts, *, batch_size, is_query) -> np.ndarray`
  (L2-normalized), plus `name`/`dim`/`max_seq`. `PatientIndex` codes
  against this protocol, never a concrete class, so swapping encoders is
  a `backend="..."` string change.
- **`SentenceTransformerEmbedder`** — wraps one `sentence-transformers`
  model: fp16 weights on CUDA (`model_kwargs={"torch_dtype":
  torch.float16}`), fp32 on CPU (automatic fallback when no CUDA device is
  available), batched (`batch_size=64` default), with an on-disk cache
  (§4.4).
- **`OpenAIEmbedder`** — wraps this project's own `llm.embed()` seam (the
  same disk cache / budget cap / Ledger accounting every other LLM call in
  this codebase already goes through), kept reachable behind the identical
  protocol for a hosted-model comparison run, never the default.
- **`EMBEDDER_REGISTRY`** — the three local ablation candidates below,
  plus `get_backend("openai")` for the fourth. `PatientIndex(patient_id)`
  with every default left in place resolves `backend="qwen3-0.6b"`.

`PatientIndex.build()`/`.search()` are the only two call sites that
actually invoke an encoder, and they now pass the correct role explicitly:
`build()` always calls `encode(texts, is_query=False)` (every turn and
fact is a *document*); `search()` always calls `encode([query],
is_query=True)` (the one string being searched for is always a *query*).
This `is_query` flag is the mechanism that makes §4.2 below operationally
real rather than merely documented.

### 4.2 The three local candidates, and what each model's own card says
about query/document formatting — read directly from the artifact, not
recalled from training-data memory

All three were verified reachable on Hugging Face and were already present
in this session's local HF cache (`~/.cache/huggingface/hub`); their
`config_sentence_transformers.json` and `README.md` were read directly out
of that cache before writing a single line of prompt-formatting code
(`graph/embedders.py`'s own module docstring reproduces the exact JSON/
prose found). This mattered concretely: getting a model's query/document
convention wrong is, per this story's own brief, "the single most common
way an embedder ablation becomes meaningless" — silently prepending the
wrong prefix (or none where one is required) costs several points of
recall without ever raising an error.

| Registry key | HF id | Query prompt | Document prompt | Source |
|---|---|---|---|---|
| `qwen3-0.6b` (default) | `Qwen/Qwen3-Embedding-0.6B` | `"Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"` | none (empty) | `config_sentence_transformers.json`'s own `prompts` dict; README: "No need to add instruction for retrieval documents," and quantifies skipping the query instruction as an "approximately 1% to 5%" retrieval-quality cost |
| `bge-m3` | `BAAI/bge-m3` | none | none | README, verbatim: "The only difference is that the BGE-M3 model no longer requires adding instructions to the queries." Symmetric — the only one of the three with no asymmetry at all |
| `arctic-l-v2` | `Snowflake/snowflake-arctic-embed-l-v2.0` | `"query: "` | none | `config_sentence_transformers.json`'s `prompts.query`; README's own Sentence-Transformers usage example calls `model.encode(queries, prompt_name="query")` immediately followed by `model.encode(documents)` with no prompt argument at all |

Two of three (`qwen3-0.6b`, `arctic-l-v2`) are **asymmetric**: the exact
same raw string embeds to a *different* vector depending on whether it is
being treated as a query or a document. One (`bge-m3`) is **symmetric**:
identical raw text embeds identically either way. `graph/embedders.py`'s
`EMBEDDER_REGISTRY` encodes this per-model, as data (`query_prompt_name:
str | None`), not as scattered per-model `if` branches, and
`tests/test_embedders.py::TestPromptFormatting` asserts both directions
directly against real model output — `qwen3-0.6b`'s query/document vectors
for identical text are asserted *not* `np.allclose`; `bge-m3`'s are
asserted `np.testing.assert_allclose` (equal).

A fourth fact, also read directly from the artifact rather than assumed:
all three models report `hidden_size=1024` in their own `config.json`, so
every local candidate in this registry emits the same output dimension —
an ablation across backends changes only which model produced the
vectors, never a downstream dimensionality confound in `PatientIndex`.
`qwen3-0.6b` additionally supports Matryoshka (MRL) truncation down to 32
dims; `SentenceTransformerEmbedder` never passes `truncate_dim`, precisely
to preserve this same-dim-across-backends property.

**OpenAI, for comparison**: OpenAI's own embeddings guide
(https://platform.openai.com/docs/guides/embeddings, fetched and read
directly in this session) documents no query/document prefix convention
at all — every worked example embeds queries and documents identically.
`OpenAIEmbedder.encode` therefore applies no per-role formatting;
`is_query` is accepted (protocol conformance) and is a documented no-op.

### 4.3 Why local, why this default, and the honest cost of the swap

Local wins over hosted on every axis that matters for this project: **free**
(no per-token cost, and this project runs under a hard-capped LLM budget
elsewhere — `llm.py`'s `MEDMEMGRAPH_MAX_USD` — that an embedding call would
otherwise eat into), **no network dependency** once weights are cached
(this session's own environment already had all three models cached
locally), and **measurably competitive-to-better on MTEB-style benchmarks**
than the previous hashing-trick default, which had *zero* semantic
signal by construction. The measured cost of the swap, honestly: per-query
search latency rose by roughly two orders of magnitude versus the hashing
trick — a single GPU forward pass through a 0.6B-parameter model, even
batched-of-one, is inherently more expensive than a `sha1` hash lookup.
§3's own re-measurement (one real MedLoCoMo patient) shows this
concretely: p50 query latency rose from 2-3 ms (hashing trick) to 240-241
ms (`qwen3-0.6b`) — still three orders of magnitude below any latency
budget this project has stated anywhere, and dominated by fixed per-call
GPU dispatch overhead for a batch-of-one query, not by anything that scales
with corpus size (the brute-force cosine *search* itself is still
sub-millisecond at this project's per-patient scale, per §3).

**This session's own throughput/VRAM measurement, all three models, RTX
3090, fp16, batch_size=64, 320 synthetic clinical-turn-shaped documents
(~35 words each) plus 30 single-query encode calls:**

| Backend | Load time | Encode throughput | Peak VRAM | Query p50 / p90 |
|---|---|---|---|---|
| `qwen3-0.6b` | 5.87 s | 247.2 texts/s | 1,783.0 MB | 24.24 ms / 24.75 ms |
| `bge-m3` | 3.88 s | 1,200.1 texts/s | 1,180.5 MB | 13.39 ms / 14.05 ms |
| `arctic-l-v2` | 4.41 s | 1,342.7 texts/s | 1,183.5 MB | 13.31 ms / 13.57 ms |

`qwen3-0.6b` is 4-5x slower per document than the other two despite a
comparable parameter count — plausibly its Qwen3-LM-derived, 28-layer,
originally-causal-decoder architecture versus the other two's shallower
encoder-only (BERT/XLM-R-family) architectures, but that causal
explanation is an inference from the model cards' own stated layer counts,
not something this session verified by profiling the forward pass
layer-by-layer — flagged as such rather than stated as settled fact. All
three comfortably coexist in VRAM simultaneously (well under 6 GB combined
against this project's 24 GB card) if a future ablation script wants to
hold all three loaded at once. Full pasted `pytest` output and the
measurement script's own numbers are in the dev-ml return note /
`.claude/logs/dev.log.md` entry for this story.

### 4.4 The disk cache: `(model_name, role, content_hash)`, not just
`(model_name, content_hash)`

`SentenceTransformerEmbedder` keeps an on-disk `.npz` cache
(`data/index/st_embed_cache.npz` by default, gitignored) so a re-run never
re-embeds text it has already embedded — this story's own "so re-runs are
free" requirement. The literal spec named the key as `(model_name,
content_hash)`; this implementation deliberately extends it to a 3-tuple
by folding in `role` (`"query"` | `"document"`). The reason is §4.2's own
finding: two of the three local backends apply a *different* prompt to
the same raw string depending on its role, so a 2-tuple key would let a
query string that happens to collide, byte-for-byte, with an indexed
turn/fact string silently return the *other* role's cached vector — wrong,
not merely stale. `tests/test_embedders.py::TestDiskCache::
test_cache_key_distinguishes_query_role_from_document_role` asserts this
directly. The cache is read-tolerant (a corrupt or foreign-schema file is
logged and treated as cold, never fatal — a cache exists purely for
latency/cost and losing it can only ever be a performance regression, not
a correctness one) and is deliberately a *different file* from this
module's own legacy `_EmbedCache` (§6) — the two have incompatible
schemas and would silently clobber each other if they shared one path.

`tests/test_embedders.py::TestDiskCache::
test_disk_cache_survives_a_fresh_instance_so_reruns_are_free` verifies the
actual story requirement end-to-end: encode a text with one
`SentenceTransformerEmbedder` instance, construct a **second**, unrelated
instance pointed at the same `cache_path` (simulating a fresh process
re-run), monkeypatch that second instance's forward pass to raise if
ever called, and confirm the cache hit still returns a byte-identical
(`np.array_equal`) vector with the forward pass never invoked.

### 4.5 `embed_fn=` — the raw-callable escape hatch, and why `embed_texts()`
below is untouched

`PatientIndex(patient_id, embed_fn=...)` still accepts a raw
`Callable[[list[str]], np.ndarray]`, used identically for queries and
documents (no role distinction) — the pre-this-story behavior, kept as an
explicit escape hatch for a caller or test that wants a synthetic,
dependency-free encoder without importing `torch`/`sentence-transformers`
at all. This module's own `embed_texts()`/`_hashing_embed()`/
`_EmbedCache`/`_embed_cached` below are **kept exactly as they were** —
this story's file list does not include `graph/retrieve.py`, and that
module's `seed_entity_ids()` function still calls `embed_texts()` directly
for its entity-name similarity check, a different call site with its own
already-reasoned-through dim/accuracy/latency trade-off that this story
has no mandate to perturb.

## 4.6 The CPU-tier ablation candidates: why they exist, what each model's
own card says, and the measured numbers

**The problem this closes.** The live demo must run on a CPU-only AWS
EC2 box, t3.medium class (2 vCPU / 4 GB, no GPU) —
`literature/20-cpu-inference-and-deployment-optimization.md`'s own
headline finding is that this project's *reranker*, not the query
encoder, is the dominant CPU-latency risk (a derived 56-172 s estimate for
100-candidate `bge-reranker-v2-m3` reranking on 2-thread CPU, vs. the
query encoder's already-known several-hundred-millisecond cost) — but the
encoder's own cost is still real money on every query, and `qwen3-0.6b`
(595.8M params, this module's own live `sum(p.numel() for p in
model.parameters())` count) was never chosen with CPU deployment in mind
(§4.3: local-vs-hosted and general-vs-clinical, never local-vs-CPU-
budget). This section registers five 22-33M-parameter, 384-dim candidates
— 18-27x smaller than `qwen3-0.6b` by live parameter count — as
`EMBEDDER_REGISTRY` rows in the identical shape as the GPU-tier three, so
swapping tiers is the same one-line `backend="..."` change §4.1 already
established, never a second code path.

### 4.6.1 Five candidates, and what each model's own card says — read
directly, not from training-data memory

All five were verified reachable on the Hub (`huggingface_hub.model_info`,
run directly in this session) before registration. `minilm-l6` was already
present in this project's local HF cache; the other four were downloaded
in this session (their `config_sentence_transformers.json`/`README.md`
read directly out of the resulting local snapshot, same discipline §4.2
already established for the GPU tier) — all five are now cached locally,
so `tests/test_embedders.py`'s CPU-tier tests need no network call to run.

| Registry key | HF id | Params | Query prompt | Document prompt | Mechanism | Source |
|---|---|---|---|---|---|---|
| `minilm-l6` | `sentence-transformers/all-MiniLM-L6-v2` | 22.7M | none | none | — | No `prompts` key in config; README's own ST example: plain `model.encode(sentences)` |
| `bge-small` | `BAAI/bge-small-en-v1.5` | 33.4M | `"Represent this sentence for searching relevant passages: "` | none | `query_prompt_text` (raw `prompt=`, new) | Config has **no** `prompts` key at all (unlike `bge-m3`); README prose only: "we suggest to add the instruction to the query... In all cases, no instruction needs to be added to passages" |
| `arctic-s` | `Snowflake/snowflake-arctic-embed-s` | 33.2M | `"Represent this sentence for searching relevant passages: "` | none | `query_prompt_name="query"` (existing mechanism) | `config_sentence_transformers.json` `prompts.query` — a **different literal string** from `arctic-l-v2`'s `"query: "`; README's own ST example confirms |
| `gte-small` | `thenlper/gte-small` | 33.4M | none | none | — | No `config_sentence_transformers.json` at all (pre-dates that file); README's own ST example: plain `model.encode(sentences)` |
| `e5-small` | `intfloat/e5-small-v2` | 33.4M | `"query: "` | `"passage: "` | `query_prompt_text` + `document_prompt_text` (both new) | No `config_sentence_transformers.json`; README FAQ, verbatim: "Use `"query: "` and `"passage: "` correspondingly for asymmetric tasks" |

Every real load's dim is 384 (`SentenceTransformerEmbedder.dim`, read live
— never hardcoded, same discipline §4.2 uses), confirmed directly by
`tests/test_embedders.py::TestCPUTierRegistry::
test_all_five_cpu_tier_candidates_report_384_dim` against all five real
models, not a subset.

**Two conventions this tier needed that the GPU tier never exercised**
(`_ModelSpec` gained two new optional fields, `query_prompt_text` /
`document_prompt_text`, both `None`-default and additive — every existing
GPU-tier row is untouched):

1. **A query instruction the card documents in prose but the model's own
   `config_sentence_transformers.json` never registers** (`bge-small`).
   `encode(..., prompt_name="query")` would raise `KeyError` against this
   model's own (empty) `self.prompts` dict; `encode(..., prompt=<raw
   string>)` bypasses that dict entirely and is what `query_prompt_text`
   routes to. Proven directly, not just inferred from "vectors differ":
   `tests/test_embedders.py::TestCPUTierPromptFormatting::
   test_bge_small_query_vector_matches_the_raw_literal_prompt_call` asserts
   this module's own query-encode output is **byte-identical** (`atol=
   1e-6`) to a hand-built `SentenceTransformer(...).encode(text, prompt=
   "Represent this sentence for searching relevant passages: ")` call
   against the same raw model.
2. **An instruction on the *document* side, not just the query side**
   (`e5-small`) — every model registered before this tier applies an
   instruction (if any) to queries only. `document_prompt_text` is checked
   independently of whichever query mechanism is in play (`Sentence
   TransformerEmbedder._forward`). Proven the same way, on the document
   path specifically: `test_e5_small_document_vector_matches_the_raw_
   passage_prefixed_call` asserts this module's document-encode output
   matches a raw `prompt="passage: "` call, and — the sharper assertion —
   is **not** `np.allclose` to a raw no-prefix call, i.e. the prefix is
   provably reaching the model, not silently dropped.

`arctic-s` is registered via the *existing* `query_prompt_name` mechanism
(no new field needed) but is its own small finding: its query instruction
string is different literal text from `arctic-l-v2`'s, despite both models
being the same family at different sizes —
`test_arctic_s_uses_registered_prompt_name_with_a_different_literal_than_
arctic_l_v2` asserts this directly against both models' own resolved
`self.prompts["query"]`, so a future contributor cannot assume one
Arctic-family card generalizes to a different size in that family.

### 4.6.2 Measured numbers — GPU and 2-thread CPU, this session's own
hardware (AMD Ryzen 9 5900X, RTX 3090; **not** a t3.medium — see the
honesty note below)

Same GPU methodology as §4.3's own table (320 synthetic clinical-turn-
shaped documents ~35 words each, batch_size=64, fp16, 30 single-query
`encode(..., is_query=True)` calls through the real `SentenceTransformer
Embedder` — i.e. through the actual registered prompt, not a bypass) plus
a new CPU measurement: `torch.set_num_threads(2)`, fp32 (CPU never uses
fp16 — `SentenceTransformerEmbedder`'s own "fp16 on CUDA, fp32 on CPU"
rule, §4.1), one warm-up call excluded, 15 single-query calls timed.

**GPU (RTX 3090, fp16):**

| Backend | Params | Load | Throughput | Peak VRAM | Query p50 / p90 |
|---|---|---|---|---|---|
| `minilm-l6` | 22.7M | 4.12 s | 1,040.3 texts/s | 85.6 MB | 3.86 ms / 4.26 ms |
| `bge-small` | 33.4M | 1.45 s | 4,647.4 texts/s | 105.8 MB | 6.88 ms / 7.23 ms |
| `arctic-s` | 33.2M | 1.56 s | 4,557.4 texts/s | 105.6 MB | 6.79 ms / 7.00 ms |
| `gte-small` | 33.4M | 1.40 s | 5,226.1 texts/s | 105.8 MB | 6.29 ms / 6.48 ms |
| `e5-small` | 33.4M | 1.37 s | 4,795.8 texts/s | 106.6 MB | 6.24 ms / 6.38 ms |
| `qwen3-0.6b` (default, re-measured for comparison) | 595.8M | 2.71 s | 342.3 texts/s | 1,673.9 MB | 25.65 ms / 25.86 ms |

**2-thread CPU — the number that matters for the t3.medium serving
target:**

| Backend | Load | Query p50 / p90 (min-max) | Process RSS |
|---|---|---|---|
| `minilm-l6` | 1.45 s | **8.83 ms** / 8.92 ms (8.75-8.98) | 1,480.1 MB |
| `bge-small` | 1.49 s | 18.58 ms / 18.80 ms (18.33-21.30) | 1,615.2 MB |
| `arctic-s` | 1.42 s | 18.19 ms / 18.42 ms (18.06-22.59) | 1,655.1 MB |
| `gte-small` | 1.43 s | 17.05 ms / 17.23 ms (16.96-19.90) | 1,660.8 MB |
| `e5-small` | 1.40 s | 18.57 ms / 18.83 ms (17.88-19.11) | 1,660.8 MB |
| `qwen3-0.6b` (default, re-measured) | 2.39 s | **342.96 ms** / 345.96 ms (338.82-350.31) | 5,183.2 MB |

Every CPU-tier candidate is **18-39x faster** than `qwen3-0.6b` at 2-thread
single-query encode, measured on the same hardware with the same
methodology (`minilm-l6` 38.8x; `gte-small` 20.1x; `arctic-s` 18.9x;
`bge-small`/`e5-small` 18.5x). Process RSS tells the more decisive story
for this project's actual 4 GB deployment box: every CPU-tier candidate
fits comfortably under 1.7 GB, while `qwen3-0.6b` alone measures 5.18 GB —
**already over this box's entire 4 GB budget on the encoder alone**,
independently confirming `literature/20`'s own flagged risk ("this
project's two models may not fit in 4GB of RAM at all at their native
FP32/BF16 precision") from the encoder side specifically, not just the
combined-with-reranker estimate that survey computed.

**An honesty note on the `qwen3-0.6b` baseline number, kept visible rather
than smoothed over.** `literature/20`'s own dispatching brief quotes "query
encode of one sentence with Qwen3-Embedding-0.6B takes 471 ms on 2-thread
CPU" as ground truth, and that survey's own record is explicit that it
"did not independently re-measure" that figure. This session did
re-measure it directly, twice — once with the query left unprompted
(bypassing this project's own registered instruction entirely, 248.43 ms —
an apples-to-oranges number, discarded) and once through the real,
registered `embedders.get_backend("qwen3-0.6b").encode([...], is_query=
True)` path with the model's own instruction prefix actually applied (the
honest, production-shaped number): **342.96 ms**, not 471 ms — roughly 27%
lower. Two plausible, non-exclusive explanations, neither confirmed in
this session: (1) this machine (AMD Ryzen 9 5900X, no AVX-512/VNNI at
all) is a materially different, and on raw per-core clock/IPC plausibly
faster, CPU than the mixed Skylake-SP/Cascade-Lake fleet AWS's own t3
product page documents for a real t3.medium — `literature/20` itself
flags this exact uncertainty (R-CPU-016/017) and this session's own
measurement is **on the same non-target workstation the whole project's
prior 471 ms figure was quoted from**, not on an actual t3.medium either,
so neither number should be read as a t3.medium prediction; (2) this is a
shared, multi-agent development box (this session found a sibling
`test_reranker.py::TestCpuAblationModelsReal` concurrently landed by a
different dev-ml run during this same story) — the original 471 ms figure
may have been measured under real, transient CPU contention from other
concurrently running agent processes that this session's own near-idle
load average (0.40-0.65 at measurement time) did not have. **What both
numbers agree on, and what this ablation actually needs, is unaffected by
which is more precise**: `qwen3-0.6b`'s 2-thread CPU query cost is in the
several-hundred-millisecond range, one to two orders of magnitude above
every CPU-tier candidate registered here — the qualitative finding this
whole story exists to establish, not the exact millisecond count.

### 4.6.3 What this section does NOT claim

Same discipline as §4.3/§7 below for the GPU tier: **no retrieval-quality
(Recall@k/nDCG@k/Hit@k) number is reported for any CPU-tier candidate
here.** This section verifies reachability, correct per-model prompt
formatting (§4.6.1, proven against the real model, not merely asserted),
and measured engineering cost (§4.6.2) — exactly this story's own
acceptance line. Whether `minilm-l6`'s English-only, MS-MARCO/general-
domain training transfers to this project's clinical-dialogue corpus
without a real accuracy cost is the open question this ablation was
commissioned to eventually answer for the *reranker* side (`literature/20`
Q2's "measure the same-day accuracy check against this project's own
held-out set" instruction) and is equally open, unanswered here, for the
encoder side — `eval/retrieval_eval.py`'s own sweep (already covers three
GPU-tier embedders x three rerankers) is the natural place a CPU-tier
Recall@k/Hit@k run would land, not this module.

## 5. Lexical channel: `bm25s` plus the +/-2 session-aware context expansion

`literature/12` §5/Q5 (and the ReFind full-text read that survey performed,
arXiv 2608.12888) is unambiguous that a *bare* BM25 index is the wrong
build: ReFind's own ablation shows a "generic agentic BM25" control (same
retrieval loop, but no session-aware RRF, no context-window expansion, no
temporal filter) scores 14.5 / 7.1 points *below* the full ReFind system on
LongMemEval-S/M [R-VEC-063]. Of the three controls layered on top of bare
BM25, removing context-window expansion — returning the +/-2 neighboring
turns around a hit — causes the single *largest* accuracy drop (9.2 points
on the S subset), bigger than removing RRF reranking (-3.9) or temporal
filtering (-1.9) [R-VEC-064]. `E5-S2`'s own scope line draws the boundary
explicitly: context expansion is **in** this story; session-level RRF and
temporal filtering are **out** (cut #2, deferred to the fusion stage,
`ARCHITECTURE.md` §7.5 / E5-S4).

`LexicalIndex` therefore always applies the expansion — it is not an
optional flag a caller can silently skip into a bare-BM25 comparison.
`build()` tokenizes and indexes every turn (`bm25s.tokenize(...,
stopwords="english")`, `bm25s.BM25().index(...)`) and separately records,
for every turn, which admission it belongs to and its zero-based position
*within that admission's own turn list*. `search()` retrieves the top-`k`
raw hits, then `_expand()` slices `[position - window, position + window]`
against that same admission's turn list only — never a different
admission's turns, which is what "session-aware" means operationally, not
just as a name (`E5-S2` AC3's literal fixture: a 7-turn admission, a hit on
turn 4, expansion to turns 2-6 — `test_channels.py`'s
`test_plus_minus_2_expansion_widens_the_span` reproduces those exact
numbers; `test_expansion_clamps_at_the_available_window` covers the edge
case where the hit is near an admission boundary and the window has to
clamp rather than reach past turn 1 or the last turn).

**This session's own measurement, the same real patient:** build time
0.06 s, on-disk-equivalent index size ~482 KB (`492,917` bytes), p50 query
latency ~0.20 ms, p90 ~0.22-0.27 ms over the same 30 real questions. The
lexical channel is both smaller on disk and faster per query than the
vector channel at this scale — consistent with `bm25s`'s own published
throughput claims [R-VEC-022] and with there being no `embed()` call
(hashing or otherwise) on the query path for this channel at all.

## 6. Persistence: save/load, and an on-disk embedding cache keyed by
content hash

Both `PatientIndex.save`/`load` and `LexicalIndex.save`/`load` persist to
`data/index/` (gitignored — local, generated data per
`decisions/001-medlocomo-packet-leakage.md`'s "never vendor local data"
discipline, same as the corpus fetch itself) so a process restart does not
have to rebuild every patient's index from `combined_conversation.json`
again. `PatientIndex` writes its vectors as a `.npy` array plus a JSON
metadata sidecar (each unit's text/session/turn-ids/kind, **and, as of
this story, `backend_name`** — the encoder that produced these vectors);
`LexicalIndex` delegates the BM25 sparse-score arrays to `bm25s.BM25`'s
own `.save()`/`.load()` and keeps a parallel JSON sidecar carrying enough
per-turn metadata (`session_id`, `turn_number`, `position`, the
already-rendered `text`) to reconstruct the +/-2 expansion state without
re-parsing the source conversation. Both `load()` classmethods raise
`FileNotFoundError` rather than silently returning an empty index when no
save exists for a `patient_id` — a missing index and a genuinely empty
patient are different facts, the same discipline `graph/existence.py`
already applies to a different failure mode ("do not invent a result").

**`PatientIndex.load()`'s backend guard (this story).** A saved index now
carries the name of the encoder that built it. `load()` resolves an
encoder — from an explicit `backend=`/`embed_fn=` argument, or, if
neither is given, from the saved `backend_name` itself — and then checks
that the resolved encoder's `.name` matches what was saved, raising
`embedders.BackendMismatchError` on any mismatch, and also raising when
the saved index has *no* `backend_name` at all (built before this story
existed) and the caller supplied nothing to disambiguate. This exists
because vectors from two different encoders are not comparable at all —
same shape, wildly different semantics — and a stale index silently
reused with a new encoder would not error, it would just return wrong
answers with high-looking cosine scores. File existence is checked
*before* any encoder is constructed (`_paths_for` is a `staticmethod`,
callable without a live instance), so a `load()` call for a genuinely
missing patient never pays a model-load cost on its way to raising
`FileNotFoundError` — `tests/test_embedders.py::
test_load_missing_index_raises_before_touching_any_backend` asserts this
by monkeypatching `embedders.get_backend` to explode if `load()` ever
reaches it on that path.

`graph/embedders.py`'s own `(model_name, role, content_hash)`-keyed disk
cache (§4.4) is a *separate* piece of infrastructure from the legacy
`vector_index._EmbedCache` described next — the two caches serve two
different, unrelated call sites (`PatientIndex`'s default backend path vs.
`embed_texts()`'s hashing-trick fallback for `graph/retrieve.py`) and
were kept deliberately non-overlapping (different default file paths) so
neither can silently clobber the other's incompatible on-disk schema.

The legacy on-disk embedding cache (`vector_index._EmbedCache`, one shared
`data/index/embed_cache.npz` keyed by `sha256(text)` alone — no model
name, because this cache predates any pluggable-backend concept) still
backs the `embed_fn=` escape hatch and `embed_texts()`'s own hashing-trick
default (§4.5); it is untouched by this story. It is a global cache, not
a per-patient one: embedding is a pure function of text alone, so the same
clinical phrase recurring across two different patients' histories embeds
exactly once.

Round-trip correctness (`test_channels.py`'s `test_save_load_round_trip`
tests, both channels, plus `TestRealPatientChannels::
test_real_patient_save_load_round_trip` on the real corpus, **now
exercising the real `qwen3-0.6b` backend rather than the hashing trick**)
asserts the saved-then-loaded index returns bit-identical `RetrieveItem`s
— text, turn ids, and score — to the freshly built one, not merely "the
same number of results."; `tests/test_embedders.py::
TestPatientIndexBackendWiring::test_save_then_load_with_matching_backend_round_trips`
adds the same assertion for an explicitly-named non-default backend
(`bge-m3`), and `test_load_with_mismatched_backend_raises` /
`test_load_with_no_recorded_backend_and_no_override_raises` cover the two
ways the new guard is meant to fire.

## 7. Honest residual

- **This story resolves the previous session's own flagged residual**
  ("no real semantic embedding model is wired in by default") — see §4.
  It also introduces a new, smaller one immediately below.
- **No cross-backend retrieval-quality ablation was run as part of this
  story.** §4.3 reports load time / throughput / VRAM / query latency for
  all three local candidates — real, measured engineering numbers — but
  *not* Recall@k/NDCG@k for `qwen3-0.6b` vs. `bge-m3` vs. `arctic-l-v2`
  against `benchmark_qa.json`'s gold evidence. That is squarely
  `eval/metrics.py`'s own recall-sweep territory (E7-S2), not this
  module's — this story's own acceptance line asks for reachability,
  correct prompt formatting, and measured throughput/VRAM, not a winner
  declared among the three. Which of the three actually retrieves best on
  this project's real clinical QA is therefore still an open, empirically
  answerable question, not a claim either this document or the code makes.
- **§4.6's five CPU-tier candidates carry the identical residual, stated
  again explicitly because it is the single most consequential open
  question this addition creates**: zero retrieval-quality measurement.
  §4.6.2 reports real, measured latency/VRAM/RSS numbers; whether
  `minilm-l6`'s (or any of the other four's) general-domain training
  transfers to this project's clinical-dialogue corpus without a real
  accuracy cost is not answered by this story and must not be assumed
  from the GPU-tier's own `literature/12` Q3 finding (general-domain
  beating clinical-specialized at 335M/8.9B scale) — that evidence was
  never generated at the 22-33M parameter scale this section registers.
  §4.6.3 states this residual in full.
- **The `qwen3-0.6b` 471 ms vs. 342.96 ms CPU-latency discrepancy (§4.6.2)
  is flagged, not resolved.** This session re-measured the story's own
  quoted baseline directly rather than trusting it blind, found a real
  ~27% gap, offered two plausible non-exclusive explanations (different,
  non-target hardware; possible transient multi-agent CPU contention at
  the time of the original measurement), and could not conclusively
  isolate which — stated as an open question rather than picking one and
  presenting it as settled.
- **The `qwen3-0.6b` vs. `bge-m3`/`arctic-l-v2` throughput gap (§4.3) is
  reported, not explained down to the mechanism** — the causal-decoder-
  architecture hypothesis offered there is a plausible inference from each
  model's own stated layer count, not something this session verified by
  profiling. Flagged explicitly rather than stated as settled.
- **No `Recall@k`/`NDCG@k` canonical metric is computed here.** This
  module reports build time, index size, and query latency (this story's
  own acceptance line); retrieval-quality metrics against
  `benchmark_qa.json`'s gold evidence belong to `eval/metrics.py` (not yet
  on disk, per E7-S2) and to that story's own recall sweep, not to this
  one. The real-patient test's questions are drawn from that patient's
  real QA set purely as realistic, non-synthetic query text for latency
  measurement — the test does not assert those questions' *answers* are
  retrieved (no gold-evidence comparison is made here).
- **Fusion (RRF, `k=60`) is not this module's job.** `ARCHITECTURE.md`
  §7.5 and `E5-S4` own combining the `vector` and `lexical` channels (and,
  where routed, the graph channel) into one ranked list. `PatientIndex`
  and `LexicalIndex` each return their own channel's top-`k` independently
  by design, tagged `channel="vector"`/`"lexical"` on every
  `RetrieveItem`, so a fusion stage has exactly what it needs and nothing
  this module would have to un-fuse.
- **`eval/baselines/lexical.py` (E7-S2, a sibling story) independently
  forked its own turn-level `bm25s` + +/-2-expansion implementation**
  before this module existed — verified by reading that file, which
  predates this story landing `graph/lexical.py`. That is a pre-existing,
  out-of-order sequencing fact (E7-S2 was implemented and merged before
  E5-S2), not a defect this story introduces or is positioned to fix
  unilaterally: `eval/baselines/lexical.py` is explicitly documented (its
  own module docstring) as an *intentionally independent* baseline
  implementation ("a baseline that quietly shares MedMemGraph's own
  retrieval machinery is ... the system under test wearing a disguise"),
  so importing this module's `LexicalIndex` into that file would
  contradict that baseline's own stated purpose, even though it would
  remove the duplication. Flagged for the team, not silently resolved
  here.

## 8. Verification

- `uv run pytest tests/test_channels.py -v -s` — 18 tests, all passing.
  **Originally** 0.5-0.6 s wall clock (hashing-trick default). **As of
  this story**, 18/18 still pass, now 23-24 s wall clock — the expected,
  foreseeable cost of the default encoder becoming a real GPU-resident
  model (one-time ~4-7 s model load, shared process-wide across every
  `PatientIndex` constructed in the same run, plus real per-query GPU
  forward passes instead of instant hash lookups) rather than a
  regression. §4.3's own table is where the "is this cost worth it"
  trade-off is argued.
- `uv run pytest tests/test_embedders.py -v` — 47 tests (30 from the
  original pluggable-embedder story, +17 for this CPU-tier addition:
  registry-data assertions for all five new `_ModelSpec` rows, real
  query/document prompt-formatting proofs for all five against the real
  model — including the two new-mechanism proofs, §4.6.1 — and the
  `PatientIndex` backend-wiring/mismatch-guard tests re-verified with a
  384-dim CPU-tier backend, including a mismatch that crosses the
  1024-dim/384-dim tier boundary) — all passing, 24.6 s wall clock (all
  eight local models load and encode within that time; no network call
  needed, every model already locally cached as of this story).
- `uv run pytest tests/test_retrieve.py -v -m "not live"` — 18/18 still
  passing; `graph/retrieve.py::_get_indexes`'s bare `PatientIndex(patient_id)`
  call site (the one real, unmodified call site this story's default-encoder
  change reaches without any code change there) still resolves correctly
  to the new default.

See the dev-ml return note / `.claude/logs/dev.log.md` entry for this
story for the pasted real `pytest` output and the real build-time/
index-size/latency/throughput/VRAM numbers this section and §4.3 quote.
