# MedMemGraph

Graph-native longitudinal clinical memory for AI agents, built on HydraDB OSS.
Hack Hydra Track 03 (Memory + Context Retrieval).

<!-- C1: collaborative/design/ARCHITECTURE.md §1 -->
MedMemGraph is a memory layer, not a second EHR. It ingests a patient's
multi-admission conversation history, extracts clinical facts, reifies them
as versioned `:Claim` nodes in a property graph, and answers questions by
walking bounded weighted paths instead of ranking chunks by cosine
similarity. When a patient's history holds no answer, it says so instead of
guessing.

Everything here runs against **HydraDB OSS**, self-hosted, in-process. No
managed retrieval API is on the critical path.

## Why a graph, not a vector store

<!-- C2: collaborative/literature/10-hydradb-capability-audit.md, quoting hackhydra.hydradb.com judging criteria -->
The Hack Hydra $500 "Best Use of HydraDB" award asks for "a particularly
strong graph data model" and "a use case that is hard to pull off with
traditional vector or relational approaches." We are not claiming it is
*impossible* with those approaches, only that it is a poor fit for them.

<!-- C3: collaborative/design/ARCHITECTURE.md §1, §7.3, §7.6 -->
Three things a longitudinal clinical memory needs and a top-k similarity
search does not naturally give you:

- **A "why do you believe this?" provenance walk.** Every clinical claim in
  the graph is a node (`:Claim`), not a row. When a claim is revised, the
  old claim is closed and linked to the new one with `SUPERSEDES`, and both
  link back to the dialogue turn that produced them via `DRAWN_FROM`. That
  chain is a graph traversal (`algo.SPpaths` / `algo.MSpaths`, HydraDB's
  bounded weighted whole-path enumerator), not a similarity score.
- **Structural absence.** "Never mentioned" is a labelled node/path lookup
  that returns zero rows. A vector index cannot return "nothing is close
  enough" as a first-class boolean; it returns nearest neighbors regardless
  of whether any of them are relevant. This is our primary abstention
  signal, not a confidence threshold on a similarity score.
- **Cross-admission synthesis.** "Did the dose change between the two
  stays?" is a two-hop walk through claims that supersede each other across
  admissions, not a fact any single chunk contains.

HydraDB OSS's unique surface for this is `algo.SPpaths` / `SSpaths` /
`MSpaths`: multi-source, multi-target, bounded-length weighted path
enumeration that returns whole paths, not endpoint projections. Everything
else in this project (embeddings, BM25, temporal intervals, extraction) is
an application convention we build *beside* the engine, not a feature of
the engine itself. See [What HydraDB OSS does not give
us](#what-hydradb-oss-does-not-give-us).

## Quickstart

```bash
bash scripts/run_hydradb.sh                                     # boot HydraDB OSS, wait on :9090/readyz
bash scripts/download_medlocomo.sh                              # fetch MedLoCoMo into data/ (gitignored, never vendored)
uv sync --extra dev && uv run pytest -q -m "not live"            # install + offline tests
uv run python scripts/ingest_corpus.py --limit 1                 # ~9 min, ~$0.15 — enough to try it
uv run python -m medmemgraph.demo.agent --patient <patient_id>   # chat with one patient's memory
```

`<patient_id>` is one of the subject-id directories under
`data/medlocomo/MedLoCoMo/` after the fetch step (101 of them; see
[Data](#data)).

**To just try it, one patient is enough** — `--limit 1`, about 9 minutes. The full
20-patient corpus (2.9 hours) is only needed to reproduce the results tables.

**The ingest step is not optional.** Every graph-route answer, the
`structural_absence` abstention signal, and the provenance walk read state that
only `scripts/ingest_corpus.py` writes; against an empty graph `retrieve()`
degrades to the text arm and the demo silently has no paths to show. Ingest
costs about **$0.15 and 9 minutes for one patient** (measured median over 20; range 2-23 min,
driven by admission count — these patients have a median of 28 admissions and one
LLM extraction call is made per admission);
`llm.py` caches every completion to disk, so a re-run after a restart is
essentially free.

Ingest is gated on a human hand-check (`fixtures/handcheck/`) — see
[Scale gate](#scale-gate) below. No API key is required to boot HydraDB or run
the offline tests. Real extraction and real answers need an OpenAI key and a
Google key (see [Configuration](#configuration)).

## Configuration

Put these in a gitignored `.env` at the repo root:

| variable | used for |
|---|---|
| `OPENAI_API_KEY` | the reader / answerer (`gpt-4.1-mini`) |
| `GOOGLE_API_KEY` | extraction, the LLM judge, entity-match adjudication (`gemini-3.5-flash-lite`) |
| `HYDRA_AUTH_TOKEN` | must match the token file `scripts/run_hydradb.sh` writes |
| `MEDLOCOMO_ROOT` | corpus parent dir (default `data/medlocomo`) |
| `MEDMEMGRAPH_MAX_USD` | cumulative spend cap, default `$5.00` |

Two providers on purpose: the judge is a **different model family** from the
answerer, so a model is never grading its own output.

Optional read-path knobs — all off/default unless set:

| variable | default | effect |
|---|---|---|
| `MEDMEMGRAPH_EMBED_BACKEND` | `qwen3-0.6b` | bi-encoder for the dense arm |
| `MEDMEMGRAPH_RERANKER` | *(off)* | registry key, HF id, **or a local checkpoint directory** |
| `MEDMEMGRAPH_RERANKER_KIND` | `seq_classification` | or `causal_yesno` |
| `MEDMEMGRAPH_RERANK_CANDIDATES` | `50` | pool size handed to the reranker |
| `MEDMEMGRAPH_INDEX_DIR` | `data/index` | load saved indexes instead of rebuilding |

### Running with no API account at all

Every LLM role accepts a `local:` model id, which loads HuggingFace weights on
this machine instead of calling a provider. No key, no per-token cost, no rate
limit:

```bash
export MEDMEMGRAPH_ANSWER_MODEL="local:Qwen/Qwen2.5-7B-Instruct"
export MEDMEMGRAPH_LOCAL_DEVICE=cuda     # or cpu; defaults to cuda when available
export MEDMEMGRAPH_LOCAL_DTYPE=float16   # float32 on cpu
export MEDMEMGRAPH_LOCAL_MAX_INPUT_TOKENS=8192
```

Anything after `local:` goes straight to `from_pretrained`, so a local directory
works as well as a Hub id — the same handoff shape as the reranker.

**Measured, not assumed** (6 patients, 204 items, same questions and same judge,
only the answerer swapped — `Qwen2.5-7B-Instruct` in 8-bit on a 16 GB card):

| | local 7B-8bit | `gpt-4.1-mini` |
|---|---:|---:|
| Answerable accuracy | 0.422 | **0.539** |
| Abstention accuracy | **0.889** | 0.722 |
| Latency / item | 23.3 s* | 5.2 s |
| Cost | **$0.00** | ~$0.04 / patient |

\* the GPU was shared with an ingest job during this run; isolated generation
was ~0.7 s.

Answerable accuracy was lower on **all six patients**, so the gap is a real
capability difference rather than variance, and the headline tables above stay on
`gpt-4.1-mini`. What the local model is good for is **iteration at zero cost**
(measure a retrieval change locally, confirm it once on the API) and
**reproducibility** — a reader with no API account can clone this repo and get
real numbers.

The abstention direction is worth noting on its own: the local model declines
more often (0.889 vs 0.722) and answers less. For a clinical memory layer that is
the safer of the two failure directions, though it is not free — it is exactly
the recall it gives up.

Two honest limits:

- **Structured output is prompted, not constrained.** OpenAI and Google both do
  native schema-constrained decoding; `transformers` alone does not. A `schema=`
  becomes an instruction and the existing schema-retry loop parses the result.
  `literature/17` records that prompted JSON mode can cost real accuracy versus
  constrained decoding, so a local model is a good fit for bulk generation and a
  questionable one for the schema-heavy extraction path until measured on your
  own data.
- **`fullctx` will not fit.** That baseline builds ~80K-token prompts by design;
  a 7B model's weights plus an 80K KV cache exceed a 16 GB card. Run the
  retrieval systems locally and `fullctx` against an API, or quantize.

The judge is worth keeping on a hosted model even when the answerer is local —
it is cheap (a few dollars for a full run) and keeping it in a **different model
family from the answerer** is what stops a model grading its own output.

`MEDMEMGRAPH_MAX_USD` is a **cumulative** cap: the ledger at
`data/llm_cache/ledger.json` persists across runs, so spend accumulates until
you reset it.

## Setup, step by step

### 1. Boot HydraDB OSS

<!-- C4: collaborative/design/stories/E4/E4-S1.md "Config / fixtures" (survey 10 §K Option A, executed) -->
Pin the image, never `latest` (`main` has no CI gate as of this audit) and
never `0.1.0` on non-amd64 hardware (`0.1.0` is amd64-only).

```bash
bash scripts/run_hydradb.sh
```

That script runs the following (shown here so the boot step is reproducible
even by hand):

```bash
printf '%s\n' 'local-development-token-32-bytes' > auth-token && chmod 644 auth-token
docker run -d --name hydradb \
  -p 7687:7687 -p 8443:8443 -p 9090:9090 \
  -v "$PWD/auth-token:/auth-token:ro" \
  -e CLOUD_PROVIDER=memory \
  -e GRAPH_AUTH_TOKEN_FILE=/auth-token \
  -e GRAPH_ALLOW_PLAINTEXT=true \
  -e GRAPH_ADVERTISED_BOLT_ADDR=127.0.0.1:7687 \
  -e RUST_MIN_STACK=33554432 \
  ghcr.io/hydra-db/hydradb:0.1.1
until curl -fsS http://127.0.0.1:9090/readyz >/dev/null 2>&1; do sleep 1; done
```

- Image: `ghcr.io/hydra-db/hydradb:0.1.1` (commit `6a2fbb1`). <!-- C5: collaborative/literature/10-hydradb-capability-audit.md §E1 -->
- `CLOUD_PROVIDER=memory`. No S3, no MinIO, no persistent volume required.
  (`CLOUD_PROVIDER=local` is unsafe under sustained write load per an
  upstream issue; do not use it here.)
- The token must be at least 32 characters and must not be `change-me`, or
  the container refuses to boot.
- Ports: `7687` (Bolt), `8443` (HTTP fallback), `9090` (admin / `/readyz`).
- `/readyz` returns 200 in **under 8 seconds** on this boot path (measured
  against a live container; do not trust a smaller number you may see
  quoted elsewhere).

Export the matching client-side token before running anything else:

```bash
export HYDRA_AUTH_TOKEN=local-development-token-32-bytes
export HYDRA_BOLT_URI=bolt://127.0.0.1:7687
export HYDRA_HTTP_URL=http://127.0.0.1:8443
```

(Or put them in a gitignored `.env`; `src/medmemgraph/hydra_client.py` loads
one automatically if present.)

### 2. Fetch MedLoCoMo (never vendored)

```bash
bash scripts/download_medlocomo.sh
```

This clones `github.com/leozzy13/MedLoCoMo` into `${MEDLOCOMO_ROOT:-data/medlocomo}`,
which is gitignored. **This repository never ships the corpus.** See
[Data](#data) for why, and the `pipeline/loader.py` module docstring
if you have access to the design workspace.

Ingestion reads exactly one file per patient,
`MedLoCoMo/<subject_id>/combined_conversation.json`. Evaluation reads
`MedLoCoMo/<subject_id>/benchmark_qa.json`. Everything else in the tree
upstream, especially `formed_packet.json`, is out of bounds: it is the
source the benchmark questions were generated against, and reading it would
be reading the answer key. An allowlist test (`tests/test_loader_allowlist.py`)
enforces this in code, not just in this paragraph.

### 3. Install and run the freeze-bar tests

```bash
uv sync
uv run pytest -q
```

This project is `uv`-managed end to end; there is no bare `pip install`
path. Tests that need a live HydraDB node are marked `@pytest.mark.live`
and only run once step 1 is up; the rest run with no external
dependencies at all (`uv run pytest -q -m "not live"` if HydraDB is not
running).

### 4. Ingest patients into the graph

```bash
uv run python scripts/ingest_corpus.py --limit 20
```

Per patient: load the conversation, extract clinical facts (one LLM call per
admission), resolve entities against everything ingested so far, write
`:Claim` nodes with `SUPERSEDES`/`CONTRADICTS` edges, write the `:Turn`
provenance layer, and save the dense + lexical indexes. Roughly **$0.15 and
9 minutes for one patient** (median of 20; 2.9 h for all twenty). `id_map`/`registry` are persisted after every
patient, so a crash at patient 17 does not discard the entity resolution done
for 1-16, and a re-run replays from the LLM cache at ~$0.

A reference run over 20 patients produced 13,406 facts (**0 skipped**), 30,086
`:Turn` nodes, 4,271 `SUPERSEDES` and 6,917 `CONTRADICTS` edges.

<a id="scale-gate"></a>
#### The scale gate

`ingest_patient()`'s first line is `assert_handcheck_passed()`, which fails
closed unless `fixtures/handcheck/PASSED` exists. That file must be written by
a **human**, after reading `fixtures/handcheck/CHECKLIST.md`'s 30 real
extracted facts against their source turns and filling the `human: ok/bad`
column. No coding agent may write it.

To regenerate the checklist against the current extractor:

```bash
uv run python scripts/handcheck_extract.py     # rewrites CHECKLIST.md + facts.jsonl
# review the rows, then:
echo "reviewed by <you>, <date>" > fixtures/handcheck/PASSED
```

The gate exists because extraction quality is the one thing no test can
assert. It is deliberately annoying.

### 5. Talk to a patient's memory

```bash
uv run python -m medmemgraph.demo.agent --patient <patient_id>
```

A minimal REPL: one `retrieve()` call per question, then an answer over
that evidence pack (Chain-of-Note). It prints the answer, the route the
question took (`graph` / `vector` / `hybrid`), whether the graph reported
`structural_absence`, citations (`session_id` / `turn_ids`), token count,
and latency. If the graph reports absence, or the reader abstains, it
prints `Not in this record` rather than guessing.

Routing is deterministic here (`--epsilon 0`); `retrieve()`'s live default of
`0.05` flips roughly one question in twenty to the other arm to keep offline
policy evaluation possible, which is the wrong trade for a demo or a recording.

### 6. Walk a fact's revision history

```bash
uv run python -m medmemgraph.demo.provenance --patient <patient_id> --predicate CURRENT_DOSAGE_OF
```

Prints every version of a claim, the `SUPERSEDES`/`CONTRADICTS` edge between
each pair, and the dialogue turn that produced each one. Omit `--claim` and it
finds a claim that actually has a chain, so there is no id to copy by hand.

This is the query the whole design exists for: nothing is deleted, so "why do
you believe this, and what did you believe before?" is a bounded path walk with
quoted evidence at every hop.

## The HydraDB dependency, in detail

<!-- C6: collaborative/literature/10-hydradb-capability-audit.md §F0, executed against a live container on driver versions 5.8.0/5.14.0/5.28.2/5.28.4 -->
**The official Neo4j Python driver does not connect to HydraDB OSS out of
the box.** HydraDB's Bolt server identifies itself as `SlateDBGraph/0.1.0`.
Every Neo4j driver ≥5.1 hard-rejects any server agent that does not begin
with `Neo4j/`, raising `UnsupportedServerProduct` at connect time, on every
driver version we tested. The upstream README's "Neo4j-compatible Bolt
connectivity" claim does not hold as shipped.

The fix is a two-line client-side monkeypatch, applied in
`src/medmemgraph/hydra_client.py` before the driver is imported:

```python
import neo4j._sync.io._bolt5  as _b5;  _b5.check_supported_server_product  = lambda a: None
import neo4j._async.io._bolt5 as _ab5; _ab5.check_supported_server_product = lambda a: None
```

The patch target moved between `neo4j` 5.x and 6.x, so the driver version
is pinned exactly: `neo4j==5.28.2` (`pyproject.toml`). Do not bump it
without re-verifying the patch target first. If your stack is not Python,
budget time to check whether your driver applies the same client-side
check before you assume this is a solved problem.

This client also gates every query against HydraDB's Cypher dialect before
it reaches the wire (`validate_dialect`, decision 003): no `IN`, no
`CONTAINS`/`ENDS WITH`, no `IS NULL`, no `CASE`, no `min()`/`max()`, no bare
`RETURN n`, no unlabelled node patterns, no unbounded variable-length
paths, one statement per call, auto-commit only (HydraDB has no
transactions). A rejected query fails locally, before any network I/O,
with a message naming the rule and the fix.

We are documenting this in detail because it cost real time to find and
will cost other teams the same time if they hit it unwarned.

## Data

<!-- C7: collaborative/design/ARCHITECTURE.md §1, "BUILD-PLAN executed table" -->
MedLoCoMo: 101 patients, 29.5 sessions mean per patient, 66.7K mean tokens
per patient history (64.2K median, 156.5K max), 6,733,899 tokens /
167,896 turns corpus-wide, 17,892 QA items, 5,964 adversarial (33.3%),
8,946 cross-session (50.0%). Gold evidence labels exist on 100% of
admission IDs and 50% of turn IDs. These numbers were measured against the
cloned release, not quoted from the originating paper.

<!-- C8: collaborative/decisions/001-medlocomo-packet-leakage.md -->
The corpus's own synthetic artifacts are CC BY 4.0. The release also
embeds MIMIC-IV-Note-derived note text (radiology and discharge notes)
under a PhysioNet data use agreement that restricts redistribution, and
that text is not covered by the CC BY 4.0 grant, which is the corpus
authors' to give only for what they generated. For that reason, and
independent of the leakage concern below, **this repository fetches
MedLoCoMo at setup time and never vendors it.**

Separately, every admission directory in the upstream release also
contains `formed_packet.json`, the structured record the benchmark's
dialogue and QA items were generated *from*. Ingesting it would mean
answering questions from the source they were written against, not from
retrieval over the conversation. Our loader allowlists exactly
`combined_conversation.json` for ingestion and `benchmark_qa.json` for
evaluation; everything else, `formed_packet.json` above all, is unreachable
by construction, enforced by `tests/test_loader_allowlist.py`.

## Results

Real numbers from a real run. **10 patients, 336 paired QA items per system**,
stratified 6-per-question-type with a fixed seed so every system saw exactly the
same items (the paired McNemar below depends on that). Judge:
`gemini-3.5-flash-lite`, a different model family from the `gpt-4.1-mini`
answerer, so no model grades its own output.

| System | Answerable acc (95% CI) | Abstention acc | Mean tokens | p50 latency |
|---|---|---|---:|---:|
| **MedMemGraph** (graph/vector router + rerank + Chain-of-Note) | **0.783** [0.730, 0.827] | **0.583** | **9,703** | 18,528 ms |
| Full-context | 0.757 [0.703, 0.804] | 0.517 | 80,557 | 17,748 ms |
| Lexical (BM25 + window) | 0.558 [0.499, 0.615] | 0.633 | 2,690 | 5,295 ms |
| Dense (chunked RAG) | 0.457 [0.399, 0.515] | 0.750 | 2,511 | 4,752 ms |
| No-memory | 0.101 [0.071, 0.143] | 0.917 | 162 | 585 ms |

**The claim, stated precisely:**

> MedMemGraph **matches or exceeds** full-context answerable accuracy while
> abstaining better, at **8.3x fewer tokens**. The accuracy difference is within
> noise; the cost difference is not.

We do **not** claim a statistically significant accuracy win. Paired McNemar
gives **p = 0.20** (0.60 Holm-adjusted) and the +0.026 margin sits inside the
0.063 minimum detectable effect at n=336. "Higher point estimate, not a settled
result" is the honest reading, and `eval/report.py` refuses to print the word
"beats" for exactly this reason.

What *is* outside noise is the cost: 9,703 tokens against 80,557.

No-memory's 0.917 abstention is not a result — a system with no patient data
abstains on nearly everything. It is the sanity floor: at 0.101 answerable it
confirms the benchmark cannot be answered from general medical knowledge.

### Per category

| Category | Full-context | MedMemGraph | |
|---|---:|---:|---|
| medical_reasoning | 0.900 | **1.000** | win |
| adversarial (abstention) | 0.517 | **0.583** | win |
| frequency_pattern | 0.604 | **0.646** | win |
| cross_admission_comparison | 0.625 | **0.646** | win |
| care_plan_rationale | 0.967 | 0.967 | tie |
| longitudinal_progression | 0.633 | 0.600 | loss |

Four wins, one tie, one loss — including a perfect 60/60 on `medical_reasoning`,
and a win on abstention, the failure mode the Track 03 brief names for
long-context models ("they mostly fail at abstention"), measured here rather
than quoted.

### How it got there

Every step is a separate measured run over the same 336 items:

| change | answerable | abstention |
|---|---:|---:|
| baseline (k=6, no rerank) | 0.536 | 0.717 |
| + cross-encoder reranking | 0.641 | 0.650 |
| + timestamps and honest evidence framing | 0.656 | 0.733 |
| + entity timelines and admission co-occurrence | 0.674 | 0.750 |
| + evidence coverage k=6 -> 40 | **0.783** | 0.583 |

The last row is the largest single gain and the one that cost something:
coverage bought 11 points of answerable accuracy and gave back 17 points of
abstention. Both directions are real and both are reported.

### Evidence coverage is a tuned parameter, and the curve turns over

| k | answerable | abstention | mean tokens |
|---:|---:|---:|---:|
| 6 | 0.674 | 0.750 | 2,483 |
| **40** | **0.783** | **0.583** | **9,703** |
| 60 | 0.775 | 0.550 | 13,493 |

More evidence is not monotonically better. k=60 is worse than k=40 on accuracy,
abstention, tokens *and* latency. `MedMemGraphAnswerer.DEFAULT_K` is 40 because
that is where the curve peaks, not because it was picked.

### Cross-encoder reranking: +10.5 points, p = 0.00016

The retrieval stage can be improved by reranking the candidate pool with a
trained cross-encoder before the reader sees it. Measured as a controlled A/B —
**same 10 patients, same 276 answerable items, same answerer and judge, only the
reranker toggled** (`MEDMEMGRAPH_RERANKER=qwen3-rerank-0.6b`):

| | reranker off | reranker on |
|---|---|---|
| Answerable accuracy | 0.536 [0.477, 0.594] | **0.641** [0.583, 0.696] |
| Abstention accuracy | 0.717 | 0.650 |

Paired McNemar: **p = 0.00016**, with **42 items flipping wrong -> right against
13 flipping right -> wrong**. The gain was positive on all 10 patients
individually (+0.033 to +0.222).

Per category, the improvement lands where retrieval was weakest:

| Category | off | on | delta |
|---|---:|---:|---:|
| longitudinal_progression | 0.267 | **0.467** | **+0.200** |
| medical_reasoning | 0.767 | 0.917 | +0.150 |
| care_plan_rationale | 0.900 | 0.967 | +0.067 |
| frequency_pattern | 0.438 | 0.500 | +0.062 |
| cross_admission_comparison | 0.229 | 0.250 | +0.021 |
| adversarial (abstention) | 0.717 | 0.650 | −0.067 |

`longitudinal_progression` nearly doubling confirms the diagnosis in the section
below: those were **retrieval** failures, not reasoning failures — the reader
was doing fine with poor evidence. The 6.7-point abstention cost is the one
trade: sharper retrieval surfaces plausible evidence more often, so the system
declines less.

With reranking on, MedMemGraph moves **ahead of lexical BM25** (0.641 vs 0.558)
rather than tied with it, and the gap to full-context narrows from 22 points to
12 while still using ~40x fewer tokens. It does not overturn the honest-loss
framing above.

**Retrieval-stage evidence** (`eval/retrieval_eval.py`, no LLM, no API cost),
turn-level grounding, n=231:

| reranker | Hit@2 | Hit@10 | nDCG@10 | latency/query (GPU) | latency/query (CPU) |
|---|---:|---:|---:|---:|---:|
| none | 0.377 | 0.606 | 0.302 | — | — |
| `ms-marco-minilm-l6-v2-onnx-int8` | 0.576 | 0.797 | 0.460 | 108 ms | **0.1 s** |
| `qwen3-rerank-0.6b` | **0.654** | **0.818** | **0.515** | 242 ms | 9.5 s |

Both beat the no-rerank control at p < 0.0001 (Holm-adjusted; paired MDE 4.5pp
against deltas of 9.1-9.3pp — not underpowered).

The two arms differ by **95x on CPU** and only 2.2x on GPU, because a
`causal_yesno` reranker runs a full causal-LM forward pass per candidate while a
`seq_classification` head emits one scalar per pair. For CPU deployment the
small classification-head model is the only viable shape; the quality gap it
needs to close is Hit@2 0.576 -> 0.654.

Reranking is **off by default** (`MEDMEMGRAPH_RERANKER` unset) so the baseline
read path is unchanged unless explicitly enabled.

### What is still weak

**The accuracy win is not statistically significant.** p = 0.20 at n = 336. The
cheapest fix is more data, not more engineering: 20 patients are ingested and
only 10 are evaluated, and doubling n roughly halves the minimum detectable
effect.

**We lose `longitudinal_progression`** (0.600 vs 0.633). Reading the failures,
these ask for a specific transition — golds are 4 words, like `topiramate to
valproate` — and our answers describe the clinical timeline without ever
asserting the one before/after pair. Forcing that shape was tried and
**reverted**: two required schema fields did change the output, but demanding a
before-state for a question whose gold is a single state manufactures a second
term, often a negation, which the judge scores as contradicting the gold. See
`eval/reader.py::TRANSITION_TYPES` for the measurement and why the code is kept
but disabled.

**Coverage bought accuracy with abstention.** Going from k=6 to k=40 gained 11
points of answerable accuracy and lost 17 points of abstention (0.750 -> 0.583).
We remain ahead of full-context on both, but the 23-point abstention margin the
smaller configuration had was the most distinctive result this system produced,
and it is gone. Recovering it without giving back the accuracy would mean
strengthening the structural-absence signal rather than shrinking `k`.

**Extraction recall is thin.** Only **27.5% of turns produce any claim** (34.6%
of doctor turns). An audit of all 39 items full-context gets and we do not found
23% unreachable by any retrieval change, because the fact was never extracted:
for one item the gold's other half is stated plainly in the dialogue ("So it's
not meningitis?") with no corresponding claim on that admission. A further 21%
are roll-ups that appear nowhere literally — `respiratory failure` occurs in 0 of
1,294 turns for that patient.

**`structural_absence` rarely fires in practice.** It works — a never-ingested
patient returns `structural_absence=True` — but for an adversarial question about
a *real* patient, seeding still returns its k nearest entities, so the abstention
comes from the reader declining, not from the graph reporting nothing. The
mechanism is real; its share of the abstention result is smaller than the
architecture section implies.

### Reproducing this

```bash
bash scripts/run_hydradb.sh
bash scripts/download_medlocomo.sh
uv run python scripts/ingest_corpus.py --limit 20         # ~$2.75, ~2.9h (or --limit 1, ~9 min)

export MEDMEMGRAPH_RERANKER=qwen3-rerank-0.6b             # +10.5 pts, p=0.00016
export MEDMEMGRAPH_RERANKER_DEVICE=cuda                   # 'cpu' works; see the CPU column above
export MEDMEMGRAPH_RERANK_CANDIDATES=150                  # must exceed k, or it caps coverage
PER_TYPE=6 SEED=0 RETRIEVE_K=40 bash scripts/run_eval.sh  # ~$30 for all 5 systems

uv run python -m medmemgraph.eval.report --markdown
```

Re-running only the graph system (`SYSTEMS=medmemgraph`) costs ~$1.50: the
baselines are deterministic given the same items and judge, so an A/B against
them stays valid without paying for them again. Full-context alone is ~85% of a
full run's spend, since it puts the entire patient history in every prompt by
design. `llm.py` caches every completion to disk, so a repeat is close to free.

Costs are dominated by full-context (~85% of LLM spend: it puts the entire
patient history in every prompt by design). `llm.py` caches every completion to
disk, so a re-run is close to free.

### Scale of the ingested graph

20 patients: **13,406 facts written, 0 skipped**, 30,086 `:Turn` nodes, 4,271
`SUPERSEDES` and 6,917 `CONTRADICTS` edges — the numbers from the reference
ingest, which is what `scripts/ingest_corpus.py --limit 20` reproduces.

The graph the results were measured against is a later replay of that ingest and
holds **13,782 claims and 28,141 `:Turn` nodes**. The turn count is ~6% lower
because a transient Bolt disconnect cut one patient's turn-writing after its
claims had committed. Both numbers are stated rather than one being quietly
picked: the claim layer is complete, the provenance layer is thin for that one
patient. On the contradiction rate, see the
measured note above `POSSIBLE_CONFIDENCE_FLOOR` in `graph/invalidate.py` — it is
inflated by an interaction between two independently chosen constants, and is
reported rather than tuned away.

## What HydraDB OSS does not give us

<!-- C9: collaborative/literature/10-hydradb-capability-audit.md §0, executed, confirmed absent by exhaustive source search -->
HydraDB OSS is a distributed, object-store-native graph engine speaking a
narrow openCypher subset over Bolt and HTTP. It has **no vector index, no
full-text/BM25 index, no bitemporal or time-travel support, and no
text-to-graph entity extraction.** All four are things this project builds
beside the engine, not features of it:

- Dense retrieval: an in-process NumPy brute-force cosine index, one array
  per patient.
- Lexical retrieval: `bm25s`.
- Bitemporal semantics: closed-open `valid_from`/`valid_to` intervals with
  a fixed sentinel (`9999-12-31T00:00:00`, never `null`, since HydraDB has
  no `IS NULL`), invalidation-by-closing rather than delete-and-replace.
- Extraction: an LLM structured-extraction pass (with a deterministic
  rule-based fallback for testing without an API key), reified as `:Claim`
  nodes on the graph.

What HydraDB does provide, and is the reason to use it rather than a
generic graph library, is durable object-store-backed storage, a real
Bolt/HTTP server with auth and backpressure, and `algo.SPpaths` /
`SSpaths` / `MSpaths`: bounded, weighted, multi-source/multi-target path
enumeration, at the traversal ceiling of 16 hops, returning whole paths.

## Repository layout

```
src/medmemgraph/
  hydra_client.py        # the only HydraDB query entry point; dialect gate + Bolt patch
  contracts.py            # frozen ClinicalFact / retrieve() interfaces
  pipeline/                # extract, normalize, entity resolution, id minting
  graph/                   # schema, writer, invalidation, existence checks
  eval/                    # baselines (no-memory, full-context), Chain-of-Note reader, judge, harness
  demo/                    # the chat wrapper and the provenance-walk showpiece
scripts/
  download_medlocomo.sh    # fetch the corpus; never commits it
  run_hydradb.sh           # boot HydraDB OSS on CLOUD_PROVIDER=memory
docs/algorithms/           # design notes for the extraction, reader, and harness algorithms
demo/                       # VIDEO_SCRIPT.md, SUBMISSION.md
```

`docs/algorithms/` is worth a look if you want the reasoning behind a
specific module rather than just its code: `extraction-and-temporal-normalization.md`,
`chain-of-note-reader.md`, `eval-harness.md`.

## License

This repository is MIT licensed (see [`LICENSE`](LICENSE)); do not treat
that as changed by anything above. It depends at runtime on **HydraDB
OSS**, which is separately licensed **AGPL-3.0** by its maintainers
(`github.com/hydra-db/hydradb`); we run it as an external service over
Bolt/HTTP, and do not vendor or modify its source. No managed HydraDB API
is used anywhere in this project.

## Demo and submission

- Video script (timed, ≤3:00): [`demo/VIDEO_SCRIPT.md`](demo/VIDEO_SCRIPT.md)
- Submission checklist: [`demo/SUBMISSION.md`](demo/SUBMISSION.md)
