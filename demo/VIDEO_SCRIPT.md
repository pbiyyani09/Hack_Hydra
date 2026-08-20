# MedMemGraph: demo video script

Four beats, in this order, no extras. Total run time budget **150s of a
180s hard cap** (30s slack for pacing; do not spend the slack on new
content: see [What this video does not
do](#what-this-video-does-not-do)). Content past 3:00 may not be reviewed,
per the hackathon's own rule.

Record after the feature freeze (2026-08-18) with real `OPENAI_API_KEY` and
`GOOGLE_API_KEY` values configured (this project does not use Anthropic) and
at least one patient ingested (`scripts/ingest_corpus.py`) behind a passed
hand-check (`fixtures/handcheck/PASSED`, E2-S3). Every command below assumes HydraDB
OSS is already booted (`bash scripts/run_hydradb.sh`) and the demo patient
exists in the graph.

**Before recording (not on camera):**

1. Pick the demo `patient_id` and confirm it is ingested.
2. Pick a cross-admission question with a real path in the graph. The
   running example in the design docs is "did the furosemide dose change
   between the two admissions"; use it **only if** that patient's two
   admissions both made it into the ingested graph. Otherwise substitute
   any real dose-change or diagnosis-update pair for the demo patient. Do
   not fabricate a question the graph cannot actually answer.
3. Pick one adversarial (never-mentioned) question for the same patient.
4. Note the `claim_id` of a claim with a real `SUPERSEDES` chain for Beat 3
   (read it off the graph or off `samples/extract-eyeball-<id>.md`-style
   output; there is no on-camera step for this lookup).

---

## Beat 1: Cross-admission question (~45s), running total 0:00-0:45

| Time | Shot | Words / on-screen |
|---|---|---|
| 0:00-0:05 | Title card: "MedMemGraph: graph-native clinical memory on HydraDB OSS" | VO: "This is MedMemGraph, built on HydraDB OSS, running self-hosted, no managed API." |
| 0:05-0:10 | Terminal: run `uv run python -m medmemgraph.demo.agent --patient <patient_id>` | (typed on screen) |
| 0:10-0:20 | Type the cross-admission question at the prompt. **Pause recording / cut here** until the real answer prints; do not speed this up artificially. | e.g. "Did the furosemide dose change between the two admissions?" (or the substituted real question; see prep step 2) |
| 0:20-0:40 | Terminal shows the answer, then `route=graph`, the path, citations (`session_id`/`turn_ids`), token count, latency. Cursor/highlight on the `route=graph` line specifically. | VO: "One retrieve call. The router sent this to the graph because it spans two admissions. That's a real path walk, not a ranked list of chunks. Here's the route line, and the citation back to the exact turn." |
| 0:40-0:45 | Hold on the citation line. | VO: "Fewer tokens than reading both admissions in full, and it's provable which turn the answer came from." |

**Must show:** the literal `route=graph` line and a path, not a wall of
retrieved chunks. **Must not show:** a second `retrieve()` call for the
same question.

---

## Beat 2: Adversarial abstention (~30s), running total 0:45-1:15

Same CLI session, no restart.

| Time | Shot | Words / on-screen |
|---|---|---|
| 0:45-0:50 | On-screen label: "Beat 2: abstention" | (silent or short VO cue) |
| 0:50-1:00 | Type the adversarial / never-mentioned question at the same prompt. **Pause recording** until the real answer prints. | e.g. a fact never present anywhere in this patient's record |
| 1:00-1:10 | Terminal shows `structural_absence=true` and the literal line `Not in this record`. | VO: "This patient's history never mentions this. The graph reports structural absence, a missing node or a missing path, and the agent declines instead of guessing." |
| 1:10-1:15 | Hold on the `Not in this record` line. | VO: "No nearest-chunk filler." |

**Must show:** `structural_absence=true` and the literal `Not in this
record` line. **Must not show:** any nearest-chunk filler, or an
unlabelled-`MATCH`-style existence check standing in for this (that check
is a known trap in this codebase; see `decisions/003`).

---

## Beat 3: Provenance walk (~50s), running total 1:15-2:05

| Time | Shot | Words / on-screen |
|---|---|---|
| 1:15-1:20 | On-screen label: "Beat 3: provenance walk" | VO: "Now the shot that's hard to do with a vector store: showing your work." |
| 1:20-1:30 | Run the provenance walk for the pre-selected `claim_id` (see `src/medmemgraph/demo/provenance.py`, `provenance_chain(client, patient_id=..., claim_id=...)`). **Pause recording** until real output prints. | on-screen command, e.g. `uv run python -m medmemgraph.demo.provenance --patient <patient_id> --claim <claim_id>`¹ |
| 1:30-1:45 | Output shows the old claim and the new claim as two nodes, and the `SUPERSEDES` (or `CONTRADICTS`) edge between them. | VO: "Old claim, new claim, and the edge that connects them. Not an overwrite; both are still in the graph." |
| 1:45-2:00 | Output shows the turn text for both claims via `DRAWN_FROM`. | VO: "And here's the actual sentence from each conversation that produced each version." |
| 2:00-2:05 | Hold on the full chain. | VO: "That's a bounded path walk through a graph a stranger can audit, not a similarity score." |

¹ `demo/provenance.py`'s contract (`provenance_chain(...)`) is fixed by
E8-S2; whether it ships with a `__main__` CLI like `demo/agent.py`'s, or is
called from a short driver script, is that story's call. If no CLI exists
by recording day, call the function directly and print its return value:

```
uv run python -c "
from medmemgraph.hydra_client import HydraClient
from medmemgraph.demo.provenance import provenance_chain
with HydraClient() as client:
    print(provenance_chain(client, patient_id='<patient_id>', claim_id=<claim_id>))
"
```

Either way, the walk itself must be `algo.SPpaths` / `algo.MSpaths` with
labelled seeds and `maxLen <= 8`, never an unlabelled `MATCH` and never an
unbounded `SUPERSEDES*` (that shape is explicitly rejected upstream in
this codebase; see `ARCHITECTURE.md` §12).

**Must show:** both claims, the `SUPERSEDES`/`CONTRADICTS` edge, and at
least one turn's real text. **Must not show:** a raw dump of driver
records, or the illegal `OPTIONAL MATCH` + unbounded `SUPERSEDES*` query
pattern this codebase explicitly bans.

---

## Beat 4: The Pareto sentence (~25s), running total 2:05-2:30

| Time | Shot | Words / on-screen |
|---|---|---|
| 2:05-2:15 | On screen: the results table from `README.md` § Results. Abstention is its own column; tokens and latency are in the same table. The numbers are measured and final — no `TBD` cells remain. | (table on screen, no VO yet) |
| 2:15-2:25 | Speak the sentence the measured table supports — and only that sentence. | VO: "Point seven eight three against full-context's point seven five seven, on the same three hundred and thirty-six questions, using eight times fewer tokens — and abstaining better. The accuracy difference is inside the noise at this sample size, so we call it a match, not a win. The cost difference is not inside the noise." |
| 2:25-2:30 | Closing card: "MedMemGraph. Built on HydraDB OSS 0.1.1, AGPL-3.0. github.com/pbiyyani09/Hack_Hydra" | (text on screen) |

**Must say on camera:** that the accuracy difference is **within noise**
(p = 0.20 at n = 336) and that the cost difference is not. Saying "we beat
full-context" without that qualifier over-reads a p = 0.20 result, and a judge
who checks `eval/report.py` will find it refuses to print the word "beats" for
exactly this reason. Claiming the match honestly is stronger than claiming a win
that the statistics do not support.

**Also worth 5 seconds if the pacing allows:** we win `medical_reasoning`
60/60 and win abstention (0.583 vs 0.517) — the failure mode the track
description names for long-context models.

**Must name on camera:** HydraDB OSS, and that it is used self-hosted (no
managed API).

---

## Total

45 + 30 + 50 + 25 = **150 seconds**, against a 180-second hard cap. The
30-second slack is pacing buffer for the two "pause until real output"
beats (1 and 3), not room for a fifth beat.

## What this video does not do

- No fifth beat, no new retrieval channel introduced for the camera.
- No claim of a *statistically significant* accuracy win. The higher point
  estimate is real and sayable; "beats" is not, at p = 0.20.
- No demo of `CLOUD_PROVIDER=local`, a managed HydraDB endpoint, or any
  network call that is not this project's own self-hosted container.
- No six-class polarity framing (decision 002 stays open; the demo speaks
  only to `asserted`/`negated`).
