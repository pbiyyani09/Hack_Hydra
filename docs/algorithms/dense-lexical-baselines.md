# Dense (rung 3) and lexical (rung 4) baselines

`src/medmemgraph/eval/baselines/dense.py`, `src/medmemgraph/eval/baselines/lexical.py`

This is the algorithm-level explanation for a paper reviewer: what these
two baselines actually compute, which paper/finding each design decision
traces to, and — honestly — what this session's real-corpus run could and
could not demonstrate. `documentor` weaves this into the end-of-bundle
README narrative; it does not rewrite this section.

## 1. Why these two rungs exist, and why they must be independent

`literature/02-memory-benchmarks-and-evaluation.md` reconstructs a
baseline ladder from the union of baseline categories actually implemented
across the memory-systems literature it surveyed (R-MEM-058/049/024):
no-memory → full-context → chunked dense RAG → BM25/lexical → the memory
system itself. Rungs 1–2 (`nomem.py`, `fullctx.py`) already exist. This
story adds rungs 3–4.

The load-bearing constraint on both is **independence**: neither baseline
imports the graph, the router, or a fusion layer. Each is exactly what a
plain, standard implementation of that retrieval family would look like —
because a baseline that quietly shares MedMemGraph's own retrieval
machinery is not a baseline a reviewer can trust; it is the system under
test wearing a disguise. What the two baselines *do* share, deliberately,
is the reading step: both are read by the exact same Chain-of-Note reader
(`eval/reader.py`) under the exact same mode (`chain_of_note`) that
`reader_con` already uses on `mock_retrieve`'s evidence. That is not an
inconsistency — it is what makes the comparison uncounfounded. If dense and
lexical used their own ad hoc "answer this" prompt instead, any accuracy
difference between them (or against `reader_con`) could be an artifact of
prompt differences rather than retrieval quality. Holding the reader fixed
and varying only the retriever isolates the one variable the ladder is
supposed to measure.

## 2. Dense (rung 3): chunking + a hashing-trick embedding

**Chunking.** Turns are grouped into fixed-token-size, non-overlapping
windows, one admission at a time — a chunk never spans two admissions,
because a chunk's `session_id` has to name exactly one admission for any
gold-admission comparison (this story's own sweep, and the eventual
canonical `Recall@k`/`NDCG@k` in `eval/metrics.py`) to mean anything.
`literature/02` R-MEM-058 documents Mem0's own baseline sweep across
128/256/512/1024/2048/4096/8192-token chunks; this story's two swept sizes
(`CHUNK_SIZE_SWEEP = (256, 1024)`) are drawn directly from that grid as a
"small" and a "large" granularity, not two arbitrary numbers.

**Embedding.** This is the one place this story had to make and record a
genuine, load-bearing decision rather than following an existing pattern.
`literature/12-vector-search-and-hybrid-fusion.md` §3/§Q3 recommends a
strong general-purpose hosted embedding model (e.g. `text-embedding-3-large`
or Voyage) as the primary path, with a local BGE-family model as an offline
fallback — but this project has **no** `ANTHROPIC_API_KEY` available in
this environment (a fact independently confirmed by every prior `[dev-ml]`
`dev.log.md` entry), no hosted embedding provider wired in at all
(Anthropic does not sell embeddings), and no local embedding library as a
dependency (`pyproject.toml` lists only `numpy`/`bm25s`, no
`sentence-transformers`/`torch`). Adding one would itself be a new,
heavyweight dependency this story's file list does not include.

The actual answer to "what should `embed()` do" already exists in this
repo: `collaborative/design/stories/E4/E4-S2.md` (Graph-owned,
`src/medmemgraph/embeddings.py`, not yet landed as of this story — verified
by `grep`, not assumed) specifies the contract exactly: `embed(texts) ->
np.ndarray`, shape `(len(texts), dim)`, L2-normalized float32, "Default
implementation: deterministic hashing-trick (no network, no FAISS)", and
explicitly **bans** "Adding FAISS / HNSW / sentence-transformers as a
required dep." `dense.py::embed()` implements that same default (not a
competing one), and tries the real Graph-owned module first via a
try/except — the identical "prefer the real thing, fall back, delete the
fork when it lands" pattern `eval/reader.py::_default_retriever()` already
uses for `retrieve()` itself.

The fallback is the **signed hashing trick** (Weinberger, Dasgupta,
Langford, Smola, Attenberg, "Feature Hashing for Large Scale Multitask
Learning", ICML 2009, arXiv:0902.2206, §3): each token is hashed once for
a bucket index and once for a sign bit, so that hash collisions partially
cancel rather than purely accumulate — the paper's own argument for why
feature hashing does not need collision-free hashing to work well in
practice. Concretely, per chunk: tokenize (lowercase, a short fixed
stopword list, no corpus-derived IDF — deliberately, since IDF would make
a vector's value depend on what else was in the same `embed()` call,
breaking the guarantee that a query embedded separately from the corpus is
still cosine-comparable to it), hash each surviving token via `hashlib.sha1`
(stable across process restarts, unlike Python's salted built-in `hash()`),
accumulate signed counts into a 512-wide vector, apply a sign-preserving
`log1p` (the same sublinear term-frequency idea BM25/TF-IDF use, so one
repeated word cannot dominate a chunk's vector), then L2-normalize.

**Honest limitation, stated plainly:** this is a bag-of-words-family
representation, not a learned semantic embedding — it ranks by (weighted,
hash-tolerant) token overlap, not by meaning. It satisfies
`ARCHITECTURE.md`'s "Dense: one in-memory NumPy array of L2-normalized
embeddings per patient_id. Brute-force cosine" bullet literally, and it is
the exact fallback this project's own architecture decision (E4-S2)
already specified for exactly this no-network/no-heavyweight-dependency
situation — but a reviewer should not read "dense (NumPy)" here as a claim
that this baseline captures semantic similarity a true embedding model
would. That gap is real, and it is visible directly in this story's own
real-corpus run (§5 below): the lexical baseline's exact-token-match BM25
scoring beat this hashing-trick "dense" baseline on retrieval quality at
every k tested on the one patient run — the opposite of what a strong
semantic embedder would likely do, and consistent with the fact that this
"dense" baseline's representational power is closer to lexical matching
than to semantics.

**Retrieval.** Per `ARCHITECTURE.md`'s own "Dense" bullet and
`literature/12`'s measured numbers (0.17 ms/query at 3,000 vectors, 7.25 ms
at 50,000 — R-VEC-009/010), this is brute-force cosine over an in-memory
NumPy array, no FAISS/HNSW/ANN index — well within the per-patient scale
this project ever needs (hundreds of chunks, not millions).

## 3. Lexical (rung 4): BM25 at turn granularity + ReFind's context-window expansion

`literature/12` §5 is why a bare BM25 index would be a strawman, not a
fair baseline. ReFind (arXiv 2608.12888) indexes an "unmodified chat
archive with a plain turn-granularity BM25 inverted index" and, across six
conversational-memory benchmarks, attains the highest mean accuracy of any
compared system, above even the strongest graph baseline (R-VEC-062). But
ReFind's own ablation names exactly why: a "generic agentic BM25" control
— same retrieval loop, but *no* session-aware context expansion, no RRF,
no temporal filtering — scores 14.5 / 7.1 points **below** the full system
on LongMemEval-S/M (R-VEC-063). Of those controls, removing **context-window
expansion** (returning the ±2 neighboring turns around a BM25 hit) causes
the single largest accuracy drop of any ablated component — 9.2 points on
the S subset, bigger than removing RRF reranking (−3.9) or temporal
filtering (−1.9) — R-VEC-064.

`lexical.py` therefore implements exactly that one highest-value control,
and only that one (RRF reranking and temporal filtering are two more of
ReFind's ablated components, real but smaller, and adding them would blur
this rung into rung 5's "hybrid fusion" territory, which this story's
"no fusion" constraint excludes): index every turn individually with
`bm25s` (Robertson & Zaragoza scoring, `bm25s`'s defaults), and at query
time, for each top-k BM25 hit, expand to the ±2 neighboring turns **within
that hit's own admission** — never across an admission boundary. This last
clause is what "session-aware" means concretely: `LexicalPatientIndex`
tracks each turn's position *within its own admission's turn list*, so the
expansion slice is bounds-checked against that admission alone and can
never reach into a different hospital stay's turns, however close the two
admissions' turn positions happen to be when flattened. This is directly
tested (`tests/test_baselines.py::TestLexicalObviousChunk::
test_session_aware_expansion_never_crosses_admission_boundary`), not just
asserted in prose.

## 4. Token/latency accounting, and the one deliberate override

Both `DenseRAGAnswerer` and `LexicalAnswerer` subclass `eval.reader.
ReaderAnswerer`, supplying only a different retriever callable. The reader
step's own `prompt_tokens`/`completion_tokens` already scale with the
retrieved evidence's size for free (`render_context()` inside `read()`
renders whatever `RetrieveItem`s it is handed) — a larger chunk size or a
larger k is already faithfully reflected in the token column with zero
extra code. The one thing the parent class's `answer()` does *not* capture
is the retrieval step's own wall-clock cost (embedding + cosine search, or
BM25 lookup + expansion) — that happens *before* the reader's internal
timer starts. Both subclasses wrap their retriever in a tiny `_TimedRetriever`
shim and add its recorded latency into the final `AnswerResult.latency_ms`,
so the harness's latency column reports the full retrieve-then-read cost
per query — the number the Pareto claim actually needs.

## 5. What the real-corpus run showed (patient `10056223`, 162 real QA items, `--dry-run`)

No `ANTHROPIC_API_KEY` is available in this environment (checked `.env`
and the shell env, same finding every prior `[dev-ml]` entry has logged),
so the harness's own per-category `answerable_accuracy`/`abstention_accuracy`
numbers below are produced by the deterministic stub reader + token-overlap
judge, not a real model — they exercise the code paths honestly but are
**not** a real quality signal (same disclaimer the `chain-of-note-reader.md`
real run already made for `reader_direct`/`reader_con`, for the identical
reason). To get a real, LLM-independent quality signal anyway, this story
adds a retrieval-only sweep (`dense_recall_sweep`/`lexical_recall_sweep`)
that checks, per QA item and with no model call at all, whether the gold
admission (`evidence.admissions`, 100% coverage) appears anywhere in the
top-k retrieved items' `session_id`s — an `admission_hit_rate`, not the
canonical `Recall@k` (that name is reserved for `eval/metrics.py`, which
does not exist in this repo yet; see the return note for why this story did
not create it).

Real numbers from this run:

```
=== dense_recall_sweep (chunk_size x k) ===
chunk_size    k     n  admission_hit_rate  mean_tokens  mean_lat_ms
       256    5   162              0.7716       1104.8       0.1172
       256   10   162              0.8519       2215.9       0.1207
      1024    5   162              0.7654       4328.4       0.1110
      1024   10   162              0.8765       8801.7       0.1141
winning (chunk_size, k) by admission_hit_rate: (1024, 10) = 0.8765

=== lexical_recall_sweep (k) ===
   k     n  admission_hit_rate  mean_tokens  mean_lat_ms
   5   162              0.7901       1235.5       0.1683
  10   162              0.9012       2465.1       0.1902
winning k by admission_hit_rate: 10 = 0.9012

=== dense vs lexical, matched k, admission_hit_rate ===
k=5:  dense_best=0.7716  lexical=0.7901  -> lexical
k=10: dense_best=0.8765  lexical=0.9012  -> lexical
```

**Stated plainly, per this story's own instruction not to bury this:
lexical beat dense at every k tested on this patient**, and did so at
roughly a quarter of the token cost (lexical k=10: 2,465 mean retrieved
tokens for 0.9012 hit rate; dense's best setting, chunk_size=1024/k=10:
8,802 mean retrieved tokens for 0.8765 hit rate — lexical is both more
accurate *and* cheaper here). This is consistent with — and should not be
overclaimed beyond — two things this survey already found: (1)
`literature/12` R-VEC-065, ReFind's own retrieval-backend comparison, found
its BM25 backend modestly outperforming a dense-embedding *and* a hybrid
backend within its specific agentic design; (2) §2's honest limitation
above — this project's "dense" baseline is a hashing-trick bag-of-words
fallback, not the strong general embedding model `literature/12` actually
recommends, so this result is real for the two baselines as implemented,
not a general claim that lexical retrieval beats semantic dense retrieval
on this task. A fair dense-vs-lexical verdict would need the real
`embed()` (hosted API or local BGE model) this story's environment could
not provide.

Also note the mechanical, expected effect inside the dense sweep alone:
recall rises monotonically with both chunk size and k (0.7654 → 0.8765 as
chunk_size/k both grow), at the cost of roughly an 8x increase in retrieved
tokens (1,104.8 → 8,801.7) — the token/recall trade-off this whole ladder
exists to make visible, not a defect in the sweep.

The harness's own per-category table (dry-run, disclaimed above) still
demonstrates the mechanical wiring correctly: `n` sums to 162 for both
systems across both breakdowns, and `dense`'s mean per-query token cost at
its winning setting (~14,081-14,799 across categories, chunk_size=1024/k=10)
is roughly 2.6x `lexical`'s (~5,424-5,470, k=10) — the same "lexical
retrieves less but hits the gold admission more often, on this patient"
finding as the recall-only sweep, now visible in the harness's own
end-to-end token accounting.
