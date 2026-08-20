# The Chain-of-Note reader — extract-then-reason on already-retrieved evidence

Owner: Evidence. Code: `src/medmemgraph/eval/reader.py`. Tests:
`tests/test_reader.py`. Harness wiring: `src/medmemgraph/eval/harness.py`
(`SYSTEM_FACTORIES["reader_direct"]`, `SYSTEM_FACTORIES["reader_con"]`).

This document explains the one piece of this codebase that is a direct,
citable implementation of a published technique rather than harness
plumbing: `reader.read()`, the Chain-of-Note extract-then-reason prompt.
`collaborative/literature/15-query-understanding-and-context-compression.md`
identifies it as "the single best cost/benefit item in this entire survey"
— a pure prompt-template change, zero extra retrieval calls, zero training
— and `collaborative/literature/02-memory-benchmarks-and-evaluation.md`
independently corroborates the same magnitude from the same primary source.
This module is that technique, made runnable and A/B-able on this
project's own data rather than assumed to transfer.

## 1. What problem this solves, and why it is "near-free"

Every baseline in this harness (`nomem`, `fullctx`) answers a question by
handing an LLM some text and asking for an answer in one shot. That "read
once, answer" strategy has a measured, specific weakness that is not about
*retrieval* quality at all — it shows up even when retrieval is perfect.
LongMemEval's own reading-strategy ablation, run with **oracle retrieval
held constant** (so the only variable is how the same retrieved evidence is
formatted and read), found: **"even with perfect retrieval, a suboptimal
reading strategy results in up to a 10-point absolute performance drop
compared to the best approach for GPT-4o"** (LongMemEval §5.5, Fig. 6 —
`literature/15` R-QCC-043, cross-verified against `literature/02` R-MEM-012
at the same source location). The fix the paper measured that gain from is
not a different retriever — it is instructing the model to **"first
extract information from each memory item and then reason based on these
notes"** (§5.5, R-QCC-042): an explicit two-step extract-then-reason
process instead of a single-pass read.

The original Chain-of-Note paper (Yu, Zhang, Pan, Ma, Wang, Yu — *Chain-of-
Note: Enhancing Robustness in Retrieval-Augmented Language Models*, EMNLP
2023, arXiv:2311.09210) reports two further numbers that are directly load-
bearing for this project specifically: **+7.9 EM given entirely noisy
retrieved documents**, and — critically, given this project's 33.3%
adversarial/abstention question mix — **+10.5 points in rejection rate**
(correctly abstaining) **for questions outside the model's knowledge scope**
(`literature/15` R-QCC-044). That second number is a direct, first-party
demonstration that reading more carefully does not just improve answer
quality — it measurably improves the model's ability to recognize *when it
does not have enough grounded information to answer at all*, which is
exactly the discrimination MedMemGraph's adversarial subset requires.

Mechanistically, a third source this project's survey cites explains *why*
this works: LLMs' internal representations often **do** correctly encode
the position and content of the right information even from deep in a long
context, but the model "often fails to leverage this in generating accurate
responses" — a "know but don't tell" gap between internal retrieval and
output generation (`literature/15` R-QCC-040). Forcing an explicit,
separately-scored extraction step closes that gap by making the model
write down what it found *before* it is allowed to reason from it, instead
of implicitly and silently deciding what mattered while composing the
final answer in one pass.

## 2. Two modes, one A/B surface

`read(question, items, mode)` implements both sides of the ablation on the
*same* evidence pack, so the harness can measure the delta on this
project's own data instead of assuming LongMemEval's number transfers
(`literature/15`'s own closing gap: "never measured together, on this
project's own MedLoCoMo-shaped data"):

- **`mode="direct"`** — the ablation baseline. One instruction: read the
  evidence, answer directly, or say `NOT_IN_RECORD`. No extraction step, no
  per-item notes, no citations.
- **`mode="chain_of_note"`** — extract-then-reason in a single completion
  (the paper's own two-step structure fits in one call; this module never
  spends a second model call or a second retrieval call to do it — see
  §5). Step 1 asks for exactly one short note per retrieved item, in order:
  what it says about the question, or the literal word `IRRELEVANT`. Step 2
  answers *using only those notes*, never outside knowledge and never an
  item marked `IRRELEVANT`; if every note is `IRRELEVANT` — or the caller
  flagged `structural_absence=True`, the graph-native "no connecting
  evidence at all" signal (`ARCHITECTURE.md` §7.6, `contracts.RetrieveResult
  .structural_absence`) — the model must answer the literal string
  `NOT_IN_RECORD` and set `abstained=true`.

Both modes are registered as full harness systems — `reader_direct` and
`reader_con` — via `ReaderDirectAnswerer`/`ReaderChainOfNoteAnswerer` in
`reader.py`, which retrieve evidence (`contracts.mock_retrieve` today, the
real graph-backed `retrieve()` behind `$MEDMEMGRAPH_USE_REAL_RETRIEVE=1`
once Graph lands it — this project's established "do not block on Graph"
convention) and pass it straight to `read()`. Because both classes satisfy
the same `Answerer` protocol as `nomem`/`fullctx`, they get the harness's
entire per-category table, judge routing, and answerable/abstention-
accuracy split *for free* — `uv run python -m medmemgraph.eval.harness
--patient <id> --system reader_direct reader_con` runs the whole A/B in one
invocation with directly comparable rows.

## 3. Abstention is a structured field, not a parsed phrase

Both modes ask the model for structured JSON (`output_config.format`,
`json_schema`, matching the pattern already established in `judge.py`):

```json
{"answer": "...", "abstained": true|false}
```

(`chain_of_note` additionally requires a `notes` array — see §4.) `abstained`
comes directly off this field — `reader.py` never regexes the answer text
for refusal-shaped phrases the way `judge.py`'s deterministic fallback does
(that keyword list is a *fallback for when there is no model to ask at
all*, not a substitute for a real model's own structured judgment). This
matters because the abstention literature this project's own survey cites
is explicit that a generator's own *confidence* is unreliable exactly under
the conditions abstention exists to catch — but a *structured decision the
model is explicitly asked to make and commit to in the schema* is a
different, stronger claim than an implicit confidence score
(`literature/06-abstention-and-calibration.md` R-ABST-13/28: raw confidence
is overconfident under distribution shift, and capable models tend to
answer incorrectly rather than abstain when context is checkably
insufficient — precisely the failure a forced, schema-validated
`abstained` field is built to prevent from hiding inside fluent prose).
`test_reader.py::TestAbstentionIsAStructuredField` proves this directly:
a scripted completion sets `abstained=True` while the answer text contains
no refusal-shaped words at all, and `Answer.abstained` is still `True`.

Two structural, zero-model-call abstentions also exist, both defensible as
*stronger* than anything the generator could contribute: an empty evidence
pack (`items == []`, nothing to extract or reason over — `06-abstention-
and-calibration.md`'s pre-generation-structural-check-beats-generator's-own-
confidence framing, applied literally), and the deterministic dry-run
stub's own policy of abstaining when `structural_absence=True` (§6).

## 4. "Do not invent turns" as code, not a prompt request

The Chain-of-Note step 2 instruction says "never state a session id or turn
id that was not given to you" — but a prompt instruction is not a
guarantee. `reader._reconcile_notes` makes it one: it builds exactly one
`Note` per retrieved item, in item order, and **every** `Note.session_id`
/`Note.turn_ids` is copied verbatim from the real `RetrieveItem` — never
from the model's JSON output. The model's raw per-item output is used only
to *look up* a matching note by exact `(session_id, turn_ids)` key; if the
model skipped an item, or named a session/turn combination that does not
exactly match a retrieved item (invented or garbled a citation), that raw
note simply fails to match and the item defaults to a non-citable
`IRRELEVANT` note instead of trusting an ungrounded claim.
`Answer.citations` — `[{session_id, turn_ids}, ...]` for every note marked
relevant — is therefore a *derived*, provably-grounded property, not a
second thing the model has to get right on its own
(`test_reader.py::TestCitationsAreGroundedNeverInvented` exercises both the
invented-session and the skipped-item case directly).

This also produces one of the two concrete, non-abstract things Chain-of-
Note buys over the direct baseline for this project's provenance story:
`mode="direct"` has no extraction step and therefore `Answer.notes`/
`Answer.citations` are always empty — direct mode cannot answer "why do you
believe that" with a specific turn; chain_of_note mode always can, or
explicitly declines.

## 5. One completion, not two, and never a second retrieval

`read()` is a pure function over an already-retrieved `items: list
[RetrieveItem]` — it never calls the retriever or the corpus loader itself
(`tests/test_reader.py::TestReadNeverRetrieves` source-greps the function
body for this). The extract step and the reason step both happen inside
**one** model completion (the paper's own two-step description does not
require two calls, and the story explicitly bans a second retrieval hop
"let me search again" as a banned approach) — cheaper, and it means the
harness's token/latency columns for `reader_con` are directly comparable to
`reader_direct`'s, not inflated by an extra round trip.

## 6. Three swappable context-rendering strategies — a genuine open question, not a default

`literature/15` §10 surfaces a real, unresolved tension between two well-
evidenced findings measured on different task shapes:

- LongMemEval's own ablation (§2 above) found **structured, JSON-per-item**
  rendering contributes to the +10-point reading-accuracy gain.
- Chroma Research's "Context Rot" report — 18 frontier models, Anthropic /
  OpenAI / Google / Alibaba families — found the opposite intuition holds
  for haystack-style long-context tasks: **"models perform worse when the
  haystack preserves a logical flow of ideas. Shuffling the haystack and
  removing local coherence consistently improves performance ... across all
  18 models tested"** (`literature/15` R-QCC-037).

Neither source's task shape is this project's bounded, pre-filtered,
graph-traversal-shaped evidence set, and the survey explicitly declines to
adjudicate which finding transfers here. So `render_context(items,
rendering)` implements all three rather than picking one:

| `rendering` | What it does | Motivated by |
|---|---|---|
| `"json"` | One JSON object per item: `session_id`, `turn_ids`, `time` (if detected), `channel`, `score`, `text` | LongMemEval's structured-context finding (R-QCC-043) |
| `"prose"` | One bracketed natural-language line per item, in retrieval order, with the same provenance fields rendered before the text | The plain baseline both other strategies are compared against |
| `"shuffled"` | The same per-item line format as `"prose"`, but **display order** is randomized (`random.Random`, seed configurable, default deterministic) | Chroma's coherence-hurts-attention finding (R-QCC-037) |

Randomizing `"shuffled"` reorders *presentation* only — each line still
carries the item's real `Item N` ordinal (its position in the original,
un-shuffled `items` list), so provenance and citation lookup never become
ambiguous just because the reading order was scrambled. The harness (§2)
exposes `--rendering {json,prose,shuffled}` so which strategy wins on this
project's actual retrieved evidence is a measured harness output, not an
assumption baked into the code.

## 7. Timestamps: rendered explicitly where present, never invented where absent

`RetrieveItem` (the frozen CONTRACT 2 shape, `contracts.py`) carries
`session_id`/`turn_ids` but has no separate `time` field — that contract is
frozen and out of this story's scope to change. Every rendering strategy
above therefore does two things: it always renders `session_id`+`turn_ids`
explicitly (the provenance the contract *does* guarantee), and it looks for
a leading bracketed timestamp already present in `item.text` — the format
EHR-RAG's own template produces, `"[time] type - description (value:
value)"` (`literature/15` §11, Appendix A, R-QCC-045) — splitting it into
its own `time` field when found (`_split_timestamp`) and reporting it
honestly as unknown when not found, rather than fabricating one. This
follows directly from this project's own cited evidence that LLMs are
specifically weak at duration arithmetic and sensitive to where a date sits
in a sentence (Test of Time, `literature/05` R-TKG-10/11; Premise Order
Matters, R-TKG-12, both cited via `literature/15` §11): put the timestamp
in a fixed, easy-to-find position ahead of the fact, every time it is
available, instead of leaving it embedded mid-sentence or silently absent.

## 8. Running the A/B, and an honest account of what this environment could measure

`uv run pytest tests/test_reader.py -v` → **31 passed** (offline, no
`ANTHROPIC_API_KEY` — every test uses either `dry_run=True` or a scripted
fake Anthropic-shaped client, per this project's established no-key-
required convention).

**No `ANTHROPIC_API_KEY` is available in this environment** (checked
`.env`, the shell environment — same finding the harness/judge story's own
return note recorded). This means the real A/B this story asks for —
`reader_direct` vs `reader_con`, live LLM, on real MedLoCoMo QA — could not
be executed in this session, and it would be dishonest to report `--dry-
run` accuracy numbers as if they measured Chain-of-Note's real effect. What
*was* run, against the real corpus, is the mechanics check:

```
env -u ANTHROPIC_API_KEY uv run python -m medmemgraph.eval.harness \
  --patient 10056223 --system reader_direct --system reader_con --dry-run --k 40
```

Both systems ran end-to-end against real `benchmark_qa.json` items (patient
`10056223`, first 40 QA items, all four `question_type`s that patient has
represented), produced full per-`question_type`/per-`scope` tables, and
wrote `results/10056223__reader_{direct,con}.json` — proving the two
systems are correctly registered, correctly retrieve via `mock_retrieve`,
and correctly route through the harness's existing judge/aggregation
machinery with no code changes there. Actual pasted output:

```
=== system=reader_direct patient=10056223 n_items=40 dry_run=True judge=token-overlap ===
-- by question_type --
category                        n    accuracy    mean_tokens    p50_lat_ms    p95_lat_ms
medical_reasoning               5       0.000          911.8          0.31        106.79
care_plan_rationale             7       0.000          928.0          0.31          0.32
longitudinal_progression       16       0.000          737.2          0.26          0.27
adversarial                    12       0.000          859.0          0.31          0.31
-- by scope --
single_admission                19       0.000          910.5          0.31          0.32
cross_admission                 21       0.000          755.2          0.26          0.28
answerable_accuracy=0.0  abstention_accuracy=0.0  n_truncated=0/40

=== system=reader_con patient=10056223 n_items=40 dry_run=True judge=token-overlap ===
-- by question_type --
category                        n    accuracy    mean_tokens    p50_lat_ms    p95_lat_ms
medical_reasoning               5       0.000         1564.0          0.54          0.58
care_plan_rationale              7       0.000         1591.9          0.53          0.54
longitudinal_progression        16       0.000         1274.9          0.44          0.45
adversarial                     12       0.000         1483.0          0.52          0.54
-- by scope --
single_admission                 19       0.000         1564.9          0.53          0.55
cross_admission                  21       0.000         1305.9          0.44          0.52
answerable_accuracy=0.0  abstention_accuracy=0.0  n_truncated=0/40
```

**Both systems score exactly 0.0 on every row, including
`abstention_accuracy`. This is not a Chain-of-Note finding — it is a mock-
data artifact, and reporting it as anything else would be exactly the kind
of dishonest number this project's framing exists to prevent.** Root
cause, traced rather than assumed: `contracts.mock_retrieve` (the stand-in
for the graph's not-yet-landed `retrieve()`, per this project's "do not
block on Graph" convention) returns item text of the literal form
`"mock evidence #{i} for patient {patient_id}: {question}"` — a templated
echo of the question itself, carrying zero real clinical content. Two
independent, compounding consequences follow directly, both mechanical, not
about reading strategy:

1. **`answerable_accuracy=0.0` for both systems**: the token-overlap judge
   (the no-API-key fallback) scores a system answer against the *gold*
   answer's content words (e.g. "unclear etiology, no infection on
   workup"). Neither stub's answer text — built entirely from the mock
   item text — can contain any of those words, because the mock item never
   contained real clinical content to extract in the first place. A live
   LLM reading real retrieved evidence would not have this failure mode;
   this is specific to `mock_retrieve`'s placeholder text.
2. **`abstention_accuracy=0.0` for both systems, on the 12 adversarial
   items**: `reader_con`'s dry-run stub marks an item relevant when it
   shares a content word with the question (§6's coarse relevance proxy,
   documented in `reader._stub_complete`) — and `mock_retrieve`'s item text
   *always* contains the full question, so every item is trivially
   "relevant" and the stub never reaches its own abstain branch.
   `reader_direct`'s stub never filters at all by design (the "just answer,
   no filtering" ablation). Both degenerate to always-answer for the same
   underlying reason: the mock evidence is not adversarial-shaped noisy
   evidence, it is an echo chamber.

**What this run does honestly demonstrate**: the reader/harness plumbing
is correct end-to-end against real QA data (`n` sums to 40 across both
groupings for both systems; `reader_con`'s `mean_tokens` is consistently
~1.7x `reader_direct`'s at identical `n_items`, entirely attributable to
the extra per-item notes step — a real, structural cost difference the
harness surfaces correctly, not a fluke). **What it cannot demonstrate is
Chain-of-Note's actual effect on accuracy or abstention** — that requires
either (a) `ANTHROPIC_API_KEY` so both `reader_direct`/`reader_con` and the
LLM judge run for real (the single next command: rerun the line above
without `--dry-run`), or (b) the real graph-backed `retrieve()` landing so
`mock_retrieve`'s degenerate echo text is replaced with genuine retrieved
clinical evidence (`$MEDMEMGRAPH_USE_REAL_RETRIEVE=1`, already wired and
ready in `reader._default_retriever`). Neither was available in this
session. Per the story's own instruction — "if Chain-of-Note does not help
on our data, say so plainly; that is a finding, not a failure" — the
honest finding here is narrower still: **this data shape (mock echo text)
cannot show Chain-of-Note helping or hurting either way**, because there is
no real signal in the input for a reading strategy to extract in the first
place. That is a fact about the fixture, not about the technique.

## 9. The `llm.py`-seam rewiring — real inference is now reachable, and honestly what that does and does not close

§8 above is an accurate historical record and is left unedited, but its
premise ("no `ANTHROPIC_API_KEY` is available in this environment") is now
the wrong question to ask. This module (dev-ml, judge.py/reader.py story)
was rewired off a direct `anthropic` SDK call — which this project's
dependencies no longer even include — onto `medmemgraph.llm.complete()`,
the shared seam. Concretely: `read(..., dry_run=False)` (the default) now
always attempts a real call using `DEFAULT_READER_MODEL = llm.ANSWER_MODEL`
(OpenAI, `gpt-4.1-mini`); if no OpenAI key is configured, it raises
`llm.MissingAPIKeyError` immediately rather than the old behavior (silently
falling back to the offline stub, forever, because `ANTHROPIC_API_KEY` was
never set to begin with). This closes the exact bug §8 was unknowingly
demonstrating: every dry-run number reported above was never one accidental
key away from becoming real — it was reachable only through `--dry-run`
now, whereas before, real inference was unreachable *no matter what a
caller intended*, silently.

**This environment does have a working `OPEN_AI_KEY` / `GOOGLE_API_KEY` in
`.env`** (verified live, not assumed — see `.claude/logs/dev.log.md`'s
`[dev-ml]` entry for this story). A single real, non-dry-run call was made
to prove the rewired path reaches a real provider end-to-end, not just
that it parses under a scripted fake client in `pytest`:

```
read("What dose of metformin does the patient take?",
     [RetrieveItem(text="Patient takes metformin 500mg for type 2 diabetes.",
                    session_id="H1", turn_ids=[4], score=0.9, channel="vector")],
     "chain_of_note", dry_run=False)
```

Real result: `text="The patient takes metformin 500mg."`,
`abstained=False`, `citations=[{"session_id": "H1", "turn_ids": [4]}]` —
correct, grounded, and citing the real item, not a hallucinated one. Total
added spend for this call plus one real judge call verifying the same
rewiring on the judge side: **$0.0005** (ledger before → after: $0.0012 →
$0.0017, tracked in `data/llm_cache/ledger.json`).

**What this does NOT close**: a full real-provider `reader_direct` vs
`reader_con` A/B across a real patient's QA set — the actual measurement
§8's own framing was built toward — was not re-run in this story (out of
this rewiring story's scope: it is a `judge.py`/`reader.py` correctness
story, not an evaluation-run story, and a 40-item × 2-system × real-judge
sweep is a deliberately larger spend/time commitment than a plumbing
verification needs). The mock-data ceiling §8 identifies
(`contracts.mock_retrieve`'s templated echo text carrying zero real
clinical content) is **also unchanged** by this rewiring — real inference
reading fake evidence still cannot demonstrate Chain-of-Note's effect
either way, for the same reason §8 gives. Running the real A/B for real —
`uv run python -m medmemgraph.eval.harness --patient 10056223 --system
reader_direct --system reader_con --k 40` (no `--dry-run`) — remains the
next concrete step, and it is now a real-inference run for the first time,
not a silently-fake one, but it was not executed as part of this story.
