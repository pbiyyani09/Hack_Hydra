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
bash scripts/run_hydradb.sh                                    # boot HydraDB OSS, wait on :9090/readyz
bash scripts/download_medlocomo.sh                              # fetch MedLoCoMo into data/ (gitignored, never vendored)
uv sync && uv run pytest -q                                     # install + freeze-bar tests
uv run python -m medmemgraph.demo.agent --patient <patient_id>  # chat with one patient's memory
```

`<patient_id>` is one of the subject-id directories under
`data/medlocomo/MedLoCoMo/` after the fetch step (101 of them; see
[Data](#data)). No API key is required to boot HydraDB or run the tests.
An LLM API key is required for real extraction and real answers; without
one, the LLM-dependent components fall back to deterministic stand-ins
(clearly labelled; see [Results](#results)).

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
[Data](#data) for why, and [decisions/001](collaborative/decisions/001-medlocomo-packet-leakage.md)
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

### 4. Talk to a patient's memory

```bash
uv run python -m medmemgraph.demo.agent --patient <patient_id>
```

A minimal REPL: one `retrieve()` call per question, then an answer over
that evidence pack (Chain-of-Note). It prints the answer, the route the
question took (`graph` / `vector` / `hybrid`), whether the graph reported
`structural_absence`, citations (`session_id` / `turn_ids`), token count,
and latency. If the graph reports absence, or the reader abstains, it
prints `Not in this record` rather than guessing.

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

**There are no results here yet.** As of this writing, no managed LLM API
key is configured in the environment this README was written in, so every
LLM-dependent component (extraction, the Chain-of-Note reader, the LLM
judge) runs on a deterministic fallback. Fallback numbers are not quality
signals; do not read anything below as a benchmark result until the freeze
run replaces it.

You can run the harness right now, with no API key, and get a real (if
uninteresting) number back, because the fallback path is itself real code:

```bash
uv run python -m medmemgraph.eval.harness --patient <patient_id> --system nomem --dry-run
```

This exercises a stub answerer and a deterministic token-overlap judge. It
confirms the harness runs end to end, nothing more; it is not evidence of
anything about memory quality.

<!--
RESULTS: filled at freeze from the eval harness output.
  uv run python -m medmemgraph.eval.harness --patient <id> --system nomem fullctx reader_direct reader_con
  Per-category + abstention aggregation from src/medmemgraph/eval/metrics.py (E7-S3: mcnemar_pvalue,
  wilson_interval, paired_bootstrap, abstention_prf, token_f1). Every cell below is a placeholder,
  not a measurement. Do not fill this table with anything that is not the output of a real run.
-->

**Answerable categories** (judge accuracy; abstention is reported
separately below, never blended into this table):

| System | medical_reasoning | care_plan_rationale | longitudinal_progression | cross_admission_comparison | frequency_pattern | mean tokens | p50 latency (ms) | p95 latency (ms) |
|---|---|---|---|---|---|---|---|---|
| No-memory | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Full-context | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| MedMemGraph (graph/vector router + Chain-of-Note) | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

**Abstention** (adversarial slice, precision/recall/F1, gold = "the
question is not answerable"; a required row, never folded into the table
above):

| System | Precision | Recall | F1 | n |
|---|---|---|---|---|
| No-memory | TBD | TBD | TBD | TBD |
| Full-context | TBD | TBD | TBD | TBD |
| MedMemGraph | TBD | TBD | TBD | TBD |

Paired McNemar p-values (MedMemGraph vs. full-context) and 95% Wilson
confidence intervals per category ship in `results/*.json` and
`eval/metrics.py` output; summarized here at freeze if they change the
read of the table above.

**The claim, stated plainly (pick exactly one of the two paragraphs below
at freeze and delete the other):**

> Comparable accuracy at a fraction of tokens and latency; wins
> concentrated on cross-admission synthesis, temporal update, and
> abstention. Not a raw accuracy win over full-context.

> Full-context has higher aggregate answerable accuracy than MedMemGraph
> on this benchmark. MedMemGraph's case is a Pareto one on cost, not on
> raw accuracy: a fraction of the tokens and latency, comparable-to-lower
> answerable accuracy, and wins on the cross-admission, temporal, and
> abstention slices where full-context struggles most. We are not claiming
> an accuracy win over full-context.

We are not claiming a raw-accuracy win over full-context, and we are not
going to. Prior work on episodic-memory layers for LLMs generally does not
win on raw accuracy against long context either; if the same holds here,
that is stated above in plain words, not buried.

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
