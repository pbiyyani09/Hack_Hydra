# Submission form — draft answers

Paste-ready. Fields marked 🔴 need your input. Form:
<https://forms.gle/WEwqEmmN7Bkp4HyJ6> · Track 03 (Memory + Context Retrieval)

---

## Project name

```
MedMemGraph
```

## Short project description

```
Graph-native longitudinal clinical memory for AI agents, built on HydraDB OSS.
It ingests a patient's multi-admission dialogue history, extracts clinical facts
as versioned :Claim nodes, and answers questions by walking bounded paths through
the graph instead of ranking chunks by similarity. On a 336-question benchmark it
matches a full-context baseline's accuracy using 8x fewer tokens, while abstaining
better when the answer genuinely is not in the record.
```

## Problem being addressed

```
A patient's history spans dozens of hospital admissions. The two standard ways to
give an AI agent that memory both fail in specific, measurable ways.

Put the whole history in the prompt and it works, but it costs ~80,000 tokens per
question, and long-context models are poor at recognising when a question simply
has no answer in the record — they answer anyway.

Retrieve by embedding similarity and you lose chronology and revision. A vector
index cannot tell you that a dose was 200mg, then 300mg, then reduced; it returns
whichever chunks look most like the question.

Both share a deeper failure. Ask "compare the etiology of this patient's headache
between two admissions" and the answer is "meningeal signs versus lupus flare" —
but the word "lupus" never appears in the question. No amount of embedding quality
retrieves a fact that is not similar to what was asked. That is a structural
limit of similarity search, not a tuning problem.

And in a clinical setting, a confidently wrong medication history is a patient
safety failure, so knowing when to say "not in this record" is not a nice-to-have.
```

## What you built

```
An end-to-end memory layer, running against self-hosted HydraDB OSS.

INGEST: an allowlisted loader reads one file per patient, an LLM extracts clinical
facts with i2b2 assertion handling (negations kept, hypotheticals dropped, family
history never attached to the patient), relative times are resolved against each
turn's own clock, entities are resolved across admissions, and facts are written
as :Claim nodes with deterministic integer ids so a crashed ingest replays
identically. 20 patients: 13,406 facts written, 0 skipped, ~28-30k :Turn nodes.

REVISION AS DATA: when a fact changes, the old claim is not deleted. Its validity
interval is closed and a SUPERSEDES or CONTRADICTS edge records why. 4,271
SUPERSEDES edges across the corpus. Every claim links back via DRAWN_FROM to the
dialogue turn that produced it, so "why do you believe this?" is a path walk with
quoted evidence at every hop.

RETRIEVAL: a frozen router sends cross-admission questions to the graph and
single-admission lookups to a vector/lexical arm, fused with reciprocal rank
fusion and reranked by a trained cross-encoder. Two graph-native retrieval modes
do the work similarity cannot: an entity timeline (every claim about an entity,
chronologically, with source turns) and admission co-occurrence (what else was
claimed in the admissions where this symptom appeared — which is where the cause
lives).

EVALUATION: a five-system ladder — no-memory floor, full-context, dense RAG,
lexical BM25, and MedMemGraph — scored by an LLM judge from a different model
family than the answerer, with paired McNemar tests, Wilson intervals, and
abstention reported separately from answerable accuracy and never blended.
```

## Deployed project link

```
No hosted link. This is a self-hosted CLI product: it runs against a HydraDB
container you control, and the corpus is MIMIC-IV-derived clinical text under a
PhysioNet data use agreement that prohibits redistribution — so a public demo
would mean publishing restricted patient data. The demo video shows the system
running end to end, and one patient ingests in about 9 minutes if you want to run
it yourself.
```

## 🔴 How the project uses the HydraDB Open Source Repo

> This is judged criterion #2 and the $500 Best Use award. Strongest version:

```
HydraDB is the memory. Every fact, every revision, and every provenance link
lives in it, and the read path is built on its native traversal.

THE QUERY THAT MAKES THE CASE. Ask "compare the etiology of headache in 2160-08
and 2161-04". The gold answer is "meningeal signs versus lupus flare". The word
"lupus" is not in the question, so similarity search cannot reach it at any
embedding quality — we measured this: retrieval returned six items from three
sessions, none of them the relevant claims.

The graph reaches it in one hop. An entity timeline returns all three headache
claims chronologically — 2160-08, 2161-04, 2163-04, exactly the dates asked
about — and tells us which admissions they occurred in. The cause is then
whatever else was claimed in those same admissions: admission 22661410 carries
"lupus asserted" and "meningitis negated" on the same day, with the lumbar
puncture that ruled it out. That is a structural answer to a question similarity
search is structurally unable to answer.

WHAT WE USE. Labelled property graph with reified :Claim nodes, so a revision is
a node with an interval rather than an overwrite. algo.MSpaths for bounded,
weighted, multi-source whole-path enumeration — it returns paths, not endpoint
projections, which is what makes a provenance walk one query. Closed-open validity
intervals with a sentinel, so "what did we believe on this date" is a range
predicate. Labelled existence checks as a first-class absence signal.

WHAT WE WOULD LOSE WITHOUT IT. A relational recursive CTE returns reachability,
not a ranked bounded path set with the evidence attached. A vector store has no
path concept at all, and no way to return "nothing is connected here" as a
boolean rather than a low similarity score.

WHAT WE LEARNED THE HARD WAY, and contributed back as documentation: the Neo4j
driver rejects HydraDB's server agent and needs a two-line client-side patch;
the Cypher surface is a narrow subset (no IN, no IS NULL, no CASE, no min/max);
and traversing through the :Patient hub node exceeds the 30-second query timeout
at real corpus scale — excluding that one edge type took a query from timing out
to 1.2 seconds. All three are written up in the repo for the next team.
```

## Tech stack used

```
HydraDB OSS 0.1.1 (Rust, AGPL-3.0, self-hosted via Docker, CLOUD_PROVIDER=memory)
Python 3.13, uv
neo4j==5.28.2 (pinned — client-side patch target moves in 6.x)
Qwen3-Embedding-0.6B via sentence-transformers; bm25s for lexical
Cross-encoder reranking (Qwen3-Reranker-0.6B; also a fine-tuned MiniLM int8 ONNX)
OpenAI gpt-4.1-mini (answering) and Google gemini-3.5-flash-lite (extraction and
  the LLM judge) — deliberately different families so the judge never grades its
  own answerer. Optional local inference via transformers for zero-cost runs.
Arize Phoenix / OpenTelemetry instrumentation
Dataset: MedLoCoMo (fetched at setup, never vendored — see licensing note)
```

## 🔴 Team members and individual contributions

```
<your name>  — <e.g. pipeline, graph model, retrieval, evaluation harness>
<teammate>   — reranker fine-tuning (MiniLM cross-encoder, int8 ONNX export;
                turn-level Hit@2 0.654 -> 0.797 against the Qwen3 baseline at
                2.3x lower latency)
```

## GitHub repository link

```
https://github.com/pbiyyani09/Hack_Hydra
```

## 🔴 3-minute demo video link

```
<unlisted YouTube URL — upload demo/medmemgraph_demo.mp4, then OPEN IT FROM A
LOGGED-OUT BROWSER before pasting it here>
```

---

## Before you submit

- [ ] PR #2 merged, so `main` has the video
- [ ] Repo is public — open it in a private window
- [ ] Video link works logged-out
- [ ] Numbers quoted here match README § Results (0.783 / 0.757 / 8.3x)
- [ ] Do not write "beats full-context" anywhere — p = 0.20, it is a match, not
      a win, and `eval/report.py` refuses to print the word for that reason
