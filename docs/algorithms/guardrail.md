# The cite-or-abstain grounding guardrail

Owner: Evidence/eval. Code: `src/medmemgraph/eval/guardrail.py`. Tests:
`tests/test_guardrail.py`. Wired (optional, off by default) into:
`src/medmemgraph/eval/reader.py`.

This document explains the piece of the pipeline that turns "grounded" from
an aspiration into something enforced: for one already-generated clinical
answer, does it actually rest on the evidence it was given, or does it
contain a claim nobody can point to a source for — which, in a clinical
memory system, is indistinguishable from a confabulation. A confabulated
medication history is this project's own stated worst-case failure mode;
this module is the mechanism that catches it before it reaches a
reader/clinician, not just a number reported after the fact.

## 1. What this measures, and how it differs from the two metrics next to it

Three modules in this repo all touch "did retrieval/generation work," and
it is worth being precise about which question each one answers, because
they can legitimately disagree without either being wrong:

| Module | Question | When it runs |
|---|---|---|
| `eval/retrieval_eval.py` (Recall@k, Hit@k, nDCG@k) | Did retrieval *find* the annotated gold evidence at all? | Pre-generation, against gold labels |
| `eval/ragas_metrics.py::faithfulness` | As a *research metric*, what fraction of an answer's claims are inferable from context? | Post-generation, reported, not enforced |
| `eval/guardrail.py` (this module) | For *this one answer, right now*, should it ship as written, be trimmed, or be refused? | Post-generation, **enforced** |

`ragas_metrics.py`'s own module docstring names this module explicitly and
states the relationship precisely: "the two are meant to be independent
estimates of the same property... not merged into one function." They are
kept genuinely independent (separate prompts, separate code paths, no
shared claim-decomposition function) so that if they ever disagree on a
real answer, that disagreement is itself a signal worth looking at, not an
artifact of one calling the other.

## 2. The algorithm — decompose, then classify, in ONE call

`literature/06-abstention-and-calibration.md`'s own "Decision Questions"
section names this module's shape by name as its recommended Stage 3: a
generation-side groundedness check that "fires only if [pre-generation]
Stage 1 and 2 both pass but the retrieved evidence is thin/ambiguous...
Decompose the drafted answer into claims and verify each against the
retrieved subgraph via an NLI-style entailment check (RAGAS-faithfulness
pattern, R-ABST-25) — one to two extra LLM calls, no sampling." RAGAS's own
Faithfulness definition (R-ABST-25, arXiv:2309.15217): decompose the answer
into atomic statements `S` via an LLM prompt, then verify each against the
retrieved context via a second LLM prompt; `F = |V| / |S|`.

`check_grounding()` runs that same decompose-then-verify shape, with two
deliberate extensions beyond RAGAS's bare boolean:

**Three-way classification, not a boolean.** RAGAS's verification step is
binary (`attributable: true/false`). Collapsing every claim into "cited" vs
"not cited" either over-punishes routine clinical framing ("this is
commonly prescribed for...") for lacking a citation it never needed, or
under-punishes a genuine fabrication by filing it in the same bucket as
harmless commentary. So every claim gets one of three labels:

- **`supported`** — a specific patient-record fact (medication, dose,
  diagnosis, event, date) directly stated or entailed by one numbered
  evidence item. Cited to that item.
- **`unsupported`** — a specific patient-record fact absent from, or
  contradicted by, every evidence item. **This is the dangerous case** —
  `GroundingReport.is_grounded` is `False` if and only if this list is
  non-empty. `uncited_claims` (below) never triggers this.
- **`uncited`** — general medical knowledge, reasoning, or transitional
  language that never claimed a patient-specific fact at all. Needs no
  citation, and does not fail groundedness — this is what keeps the
  guardrail from flagging ordinary clinical prose as a violation just
  because it did not cite a source for saying something unremarkable.

**Explicit, verified citation, not just a bit.** Each `"supported"` claim
names *which* numbered item grounds it, and `_parse_claims` cross-checks
that index is real before trusting it — a claim marked "supported" that
cites an item outside `[1, len(items)]` is downgraded to `"unsupported"`.
This is the same guarantee `eval/reader.py::_reconcile_notes` already
applies to the *reader's* own per-item notes ("never trust a citation the
model claims without verifying it's real"), applied here to the *judge's*
output instead of the answerer's.

**Cost discipline: exactly ONE judge call per answer.** `ragas_metrics.py`'s
Faithfulness deliberately spends one decomposition call plus one
verification call *per claim* (a documented, different cost-shape choice —
see `docs/algorithms/ragas-metrics.md` §2). This module's story imposed a
tighter constraint ("this adds one judge-model call per answer"), so
decomposition and classification happen in a single schema'd
`llm.complete()` call (`_GROUNDING_SCHEMA`): one array of `{text, status,
cited_item, reason}` objects, plus one overall `confidence`.

## 3. `structural_absence` is reused, never rediscovered

`retrieve()`'s `structural_absence` flag means "the graph itself found no
connecting evidence for this patient at all" — a pre-generation, structural
fact, always paired with `items == []` (`contracts.RetrieveResult`,
ARCHITECTURE.md §7.6). `literature/06` R-ABST-28 (the *Sufficient Context*
study, Joren et al.) found capable models "often output incorrect answers
instead of abstaining" even when context insufficiency is, in principle,
checkable before generation — the literature-backed argument for why a
**pre-generation structural check outranks a post-generation judge call**
re-deriving the same fact. `check_grounding(..., structural_absence=True)`
therefore short-circuits immediately, with **zero** `llm.complete` calls —
verified directly in this module's live run (§6) via a trip-wire that
raises if the function is ever called, not just a timing check. Two more
zero-call short circuits exist for the same reason (spending a judge call
would either be a wasted no-op, or worse, an unreliable inference standing
in for something already knowable for free):

- The answer is empty or already looks like a decline
  (`judge._looks_like_abstention`, reused rather than re-implemented — an
  honest refusal asserts no facts to check).
- `retrieved_items` is empty even though `structural_absence` was not set
  (a defensive fallback) — zero context means any patient-specific claim
  in a non-abstaining answer is unsupported *by construction*; the whole
  answer is flagged as one hazard claim deterministically, the same "no
  context → unsupported by construction" convention `ragas_metrics.py`'s
  own `faithfulness()` already documents for its empty-context case.

Structural absence is *pre*-generation; this module's check is
*post*-generation — complementary signals along the pipeline, never one
re-deriving the other.

## 4. `enforce()` — three policies, one deliberately inert by default

`enforce(answer, items, policy)` wraps `check_grounding()` with an action:

- **`"warn"` (the default)** — never touches `text`. The finding lives
  entirely in `EnforcedAnswer.report` (which every policy returns
  regardless). Silently rewriting a clinical answer is its own hazard
  (the story's own framing) — so the *default* behavior of this module
  makes zero content changes, and a caller must opt in to anything
  stronger. This also matters given the honesty caveat in §5 below: a
  heuristic classifier automatically rewriting or suppressing clinical
  text on every flagged case would itself be a hazard if the flagging has
  a meaningful false-positive rate, which is exactly the caveat R-ABST-49
  raises about this whole technique family.
- **`"strip"`** — removes only the `unsupported` claims, rebuilding `text`
  from the surviving `supported`/`uncited` claim texts. If nothing
  survives, the result is the same `"NOT_IN_RECORD"` sentinel `"abstain"`
  uses (there is no partial answer left to return).
- **`"abstain"`** — replaces the entire `text` with the literal
  `"NOT_IN_RECORD"` sentinel (matching `eval/reader.py`'s own abstention
  convention exactly, so downstream consumers already recognizing that
  string treat a guardrail-triggered refusal identically to one the reader
  produced on its own) whenever `is_grounded` is `False`; a no-op
  otherwise.

`enabled=False` is a **genuine** no-op: `check_grounding`/`llm.complete`
are never called at all, not merely skipped after a cheap check —
`tests/test_guardrail.py::TestDisabledIsANoOp` proves this with a
trip-wire, and the live run (§6) proves it against the real module, not a
test double.

## 5. Honesty — what this classifier is, and is not, evidence of

`literature/06` R-ABST-49 measured RAGAS-Faithfulness itself at only 69.0%
binary hallucination-classification accuracy on HaluBench (15,000
context-question-answer triples) — well below purpose-built judges like
Lynx-70B (87.4%) or even GPT-4o used directly as a judge (86.5%). This is a
different, harder measurement from RAGAS's own separately-reported 0.95
pairwise-preference-agreement number on a 50-page curated dataset
(R-ABST-27) — the same survey flags substituting one for the other as a
config-mismatch-class error. This module inherits the decompose-then-verify
*shape* RAGAS validated, not a promise that its own numbers transfer
unchanged; the honest framing is "a real, useful heuristic signal," not "an
oracle." Concretely, this shapes two design choices already described
above, restated here as the reason, not just the rule: `enforce()` defaults
to `"warn"` rather than `"abstain"` (a fail-closed policy is only as good
as the classifier gating it), and every `GroundingReport` carries its own
`confidence` so a caller can weight how much to trust one specific verdict
rather than treating every flag as equally certain.

## 6. Live-verified real run

Real `gpt-4.1-mini` reader call and real `gemini-3.5-flash-lite` judge
calls (not `dry_run`, not a fake) — the same faithful/deliberately-wrong
control pair `docs/algorithms/ragas-metrics.md` §5 uses, applied to this
module instead:

**Evidence** (two real-clinical-shaped items, one admission):
1. `[admission-0042, turn 17]` "Patient reports taking metformin 500mg
   twice daily for type 2 diabetes, well-controlled per last A1c of 6.8."
2. `[admission-0042, turn 19]` "Home medication reconciliation confirms
   metformin 500mg PO BID; no other diabetes medications listed."

**Step 1 — a real reader answer** (`reader.read(..., dry_run=False)`,
`gpt-4.1-mini`): `"The patient takes metformin 500mg twice daily (BID) for
diabetes."` — `abstained=False`, 601 prompt tokens / 105 completion tokens.

**Step 2 — guardrail check on that real, grounded answer**
(`gemini-3.5-flash-lite`, real call): `is_grounded=True`,
`citations=[('admission-0042', [17])]`, `uncited_claims=[]`,
`unsupported_claims=[]`, `confidence=1.0`, **cost_usd=$0.000202**,
`n_judge_calls=1`.

**Step 3 — a deliberately fabricated answer**, same evidence: `"The
patient takes metformin 500mg twice daily for type 2 diabetes, and was
also started on lisinopril 20mg daily for hypertension management."`
Real guardrail verdict: `is_grounded=False`,
`unsupported_claims=["The patient was started on lisinopril 20mg daily
for hypertension management."]`, reason: *"There is no mention of
lisinopril or hypertension in the provided evidence items."*
`citations=[('admission-0042', [17])]` (the real metformin claim still
cites correctly — the guardrail did not throw away the good half of the
answer). **cost_usd=$0.000268**, `n_judge_calls=1`.

**One caught fabrication, discriminating cleanly**: the metformin claim
scored `supported` in both runs; the invented lisinopril/hypertension claim
scored `unsupported` only when actually present — the classifier is
responding to content, not defaulting to "always flag" or "never flag."

**Step 4 — `enforce()` applied to the fabricated report, all three
policies, on the real result**:

| policy | `modified` | `text` |
|---|---|---|
| `warn` | `False` | *(unchanged — full fabricated text, including lisinopril)* |
| `strip` | `True` | `"The patient takes metformin 500mg twice daily for type 2 diabetes."` |
| `abstain` | `True` | `"NOT_IN_RECORD"` |

`strip` correctly kept the real metformin claim and removed only the
fabricated one — it did not collapse to a full refusal when part of the
answer was genuinely grounded.

**Step 5 — `structural_absence` short circuit, against the real module**:
`check_grounding("NOT_IN_RECORD", [], structural_absence=True, dry_run=False)`
with `llm.complete` replaced by a trip-wire that raises `AssertionError` if
called at all: **zero calls made**, `shortcut_reason="structural_absence"`,
`is_grounded=True`.

**Per-answer cost, measured, not modeled**: the two real guardrail checks
above cost **$0.000202** and **$0.000268** — mean **≈$0.000235/answer**.
Linear projection (never a token-count model) for a full harness sweep:

| projected run size | projected guardrail-only cost |
|---|---|
| 40 QA items (one `--dry-run` mechanics-check patient, `chain-of-note-reader.md` §8's own sample size) | $0.0094 |
| 964 items (one full real MedLoCoMo patient's QA set, `eval-harness.md`'s own corpus number) | $0.227 |

Cumulative ledger for this session's real calls (`data/llm_cache/ledger.json`,
persisted, shared with every prior real-inference story in this project):
`gpt-4.1-mini` 8 calls / $0.0061 total; `gemini-3.5-flash-lite` 213 calls /
$0.4275 total — this story's own contribution was one real reader call plus
two real guardrail calls, a few tenths of a cent, well inside the
$5.00 `MEDMEMGRAPH_MAX_USD` cap.

See `.claude/logs/dev.log.md`'s `[dev-ml]` entry for this story for the
full pasted `pytest -v` output and the raw script this live run's numbers
are taken from.
