# The evaluation harness — no-memory and full-context baselines

Owner: Evidence. Code: `src/medmemgraph/eval/{harness,judge,types}.py`,
`src/medmemgraph/eval/baselines/{nomem,fullctx}.py`. Tests:
`tests/test_harness.py`.

This document explains three pieces of the harness that a reviewer would
otherwise have to reconstruct from the code: the baseline ladder framing,
the full-context truncation policy, and the judge's abstention routing.
Everything here runs without HydraDB — that is deliberate (see "Why no
graph" below), not an oversight.

## 1. The baseline ladder, and why full-context is allowed to win

MedMemGraph's headline claim is Pareto, not a raw-accuracy win over
full-context (`collaborative/design/ARCHITECTURE.md` §1): *comparable
accuracy at a fraction of the tokens and latency, with the wins concentrated
on cross-admission synthesis, temporal update, and abstention*. Mem0's own
paper reports full-context beating Mem0 by six points on the same backbone;
an independent 2026 replication has long-context beating mem0-based memory
by 33+ points. A harness that quietly disadvantages full-context — by
truncating it without saying so, by using a stronger model for the memory
system, by blending metrics that favor one baseline — would make the Pareto
claim unfalsifiable. So the two baselines in this module are built to be as
strong as their category allows:

- **`nomem` (`NoMemoryAnswerer`)** answers from the bare question text, with
  an explicit system-prompt instruction *not* to invent patient-specific
  facts. This is the sanity floor: it has structurally no way to get a
  patient-specific question right except by chance or by the question
  itself leaking the answer. Any system — including MedMemGraph — scoring
  near this baseline is not doing its job.
- **`fullctx` (`FullContextAnswerer`)** puts the patient's entire turn
  history in the prompt, oldest-first, and is *expected* to score well. Its
  cost (tokens, latency) is what the Pareto argument is actually made
  against — see the `mean_tokens` column in the harness table, which for a
  real patient is roughly 300–400× `nomem`'s, purely from carrying the
  history.

Both baselines share one contract, enforced by `harness.evaluate()` calling
every system through the same `Answerer.answer(question, conversation, *,
patient_id)` signature: `nomem` receives the `Conversation` object but
never reads a single field of it (see `del conversation, patient_id` at the
top of `NoMemoryAnswerer.answer`) — this is checked by
`test_nomem_never_truncates_because_it_never_reads_history`, which is really
a proxy for "nomem never touches the history at all."

**Why no graph.** `nomem`/`fullctx` never import `hydra_client` or anything
graph-facing. This is intentional per `ARCHITECTURE.md` §3.2: *"Evidence
baselines --no graph--> Evidence harness (unblocks eval on day one)."* The
graph-backed `retrieve()` route is a third system that will plug into the
same `evaluate()` function later via the same `Answerer` protocol — nothing
in this harness needs to change to add it.

## 2. Full-context truncation — the honesty constraint

A silently truncated baseline is a dishonest baseline: it looks like it had
access to the full patient history when it did not, and any accuracy loss
gets misattributed to the model instead of to the missing context. The
`FullContextAnswerer._build_context` method enforces two rules:

1. **Truncate from the oldest end.** `Conversation.turns()` returns turns in
   admission order, oldest first (`pipeline/loader.py`). When the estimated
   token count of the flattened turn list exceeds the configured budget
   (`context_window_tokens - reserved tokens for system prompt / question /
   output`), the method drops turns from the *front* of the list — the
   earliest admissions — one at a time, re-summing as it goes, until the
   remainder fits (or exactly one turn remains, which is never dropped).
   This mirrors what a real deployment would do: keep the most recent,
   presumably most relevant, history when a window forces a choice.
2. **Record it on the result, always.** Every `AnswerResult` carries
   `truncated: bool`, `turns_dropped: int`, and `tokens_dropped: int`. The
   harness aggregates `n_truncated` per run so a reader can see, in the
   printed summary line, exactly how many of the patient's QA items were
   answered against a truncated history. Truncation is computed identically
   in `--dry-run` mode — only the network call is skipped — so the
   truncation *decision* is exercised and testable without an API key
   (`tests/test_harness.py::TestTruncationRecorded`).

Token counts here are estimates (`eval/types.estimate_tokens`, tiktoken's
`cl100k_base` encoding with a `chars // 4` fallback — the same
degrade-don't-raise pattern `Conversation.token_estimate()` already
established), not the exact count the target model's own tokenizer would
produce. The budget is deliberately conservative (a flat 4,000-token
reservation for system prompt + question + output) specifically so that
estimation error pushes toward *safe* (a slightly early truncation) rather
than *unsafe* (a request that overflows the real context window and fails
outright).

For the primary MedLoCoMo corpus this policy is close to inert in practice:
mean patient history is ~66.7K tokens, max 156.5K (`ARCHITECTURE.md`'s
executed corpus table), comfortably under this module's default 190K-token
budget for `claude-haiku-4-5`'s 200K context window. The infrastructure
exists — and is unit-tested under an artificially small window — for the
outlier patient or a smaller-context model where it does bind.

## 3. The judge — abstention routing and the two backends

`judge.Judge.judge(question, gold_answer, system_answer, question_type)` is
the harness's only scoring authority, and it makes exactly one branching
decision:

```
question_type == "adversarial"  →  _judge_abstention(...)   (mode="abstention")
otherwise                       →  _judge_answerable(...)   (mode="answerable")
```

This branch exists because MedLoCoMo's adversarial items (33.3% of the
corpus, the single largest question category) carry the *literal* gold
string `"the question is not answerable"` — verified by reading the real
corpus (`data/medlocomo/MedLoCoMo/*/benchmark_qa.json`; every sampled
adversarial item across 15 patients / 964 items carries exactly that
string). Scoring a system's answer against that string with ordinary
semantic-correctness judgment is meaningless — there is no substantive
answer to match, and a system that says "the question is not answerable"
verbatim would trivially "match" while a system that correctly declines in
different words would not. `_judge_abstention` instead asks the narrower
question *did the system decline to answer*, and scores `correct =
declined`. A system that confabulates a plausible-sounding answer to an
unanswerable question is judged **wrong**, even though it produced fluent,
on-topic text — this is the abstention story the graph's `structural_absence`
signal (`ARCHITECTURE.md` §7.6) is ultimately built to win, and the harness
has to be able to detect the failure mode it is trying to measure.

The consequence surfaces one level up, in `harness.evaluate()`:
`answerable_accuracy` is computed only from non-adversarial records, and
`abstention_accuracy` only from adversarial ones — the two numbers are never
averaged into a single headline figure. Blending them would let a system
buy a good-looking overall score by refusing everything (abstention
accuracy is trivially maximized by never answering) or by ignoring the
adversarial third of the benchmark entirely — either failure mode is
exactly what the Pareto framing is supposed to surface, not hide.

**Two judge backends**, selected once at `Judge.__init__` time and recorded
on every `HarnessRun` as `judge_kind`. As of the `llm.py`-seam rewiring
(dev-ml, judge.py/reader.py story), the previous `ANTHROPIC_API_KEY`-
presence-based auto-detection is gone — `anthropic` was removed from this
project's dependencies once `llm.py` replaced it, so that auto-detection
would now *always* pick the fallback, silently, forever, which is exactly
the bug this rewiring fixes:

- **`llm`** (`force_fallback=False`, the default): a real call through
  `medmemgraph.llm.complete()` — the single seam every LLM-dependent module
  in this project shares — using `model=DEFAULT_JUDGE_MODEL`
  (`llm.JUDGE_MODEL`, Google `gemini-3.5-flash-lite` by default). A cheap/
  fast model is appropriate here because *judging is easier than
  generating*, i.e. a weaker model can often reliably judge a stronger one.
  **Deliberately a different provider family from the answering systems**
  (`llm.ANSWER_MODEL`, OpenAI `gpt-4.1-mini`) — `collaborative/literature/
  17-genai-structured-generation-and-judging.md` (Part C.11–12) found
  self-preference bias skewing scores by up to 10 points on a medical
  rubric benchmark specifically when judge and candidate share a model
  family; this is argued once, in full, in `judge.py`'s own module
  docstring, not left implicit. Uses `llm.complete(schema=...)` — provider-
  native structured output, never a bare "respond in JSON" instruction —
  so the response is guaranteed to parse as the expected JSON shape.
  `temperature=0.0` is always requested (both `gemini-3.5-flash-lite` and
  `gpt-4.1-mini` accept it unconditionally; the old `supports_temperature()`
  guard was a Claude-4.6+-specific quirk that no longer applies once the
  provider changed). Two further, measurement-backed judge-bias
  mitigations are built into the prompts themselves (not just the model
  choice): a **rubric-style system prompt** with fixed, numbered criteria
  (literature/17 R-GSJ-035/053), and **deterministic ordering** of the
  question/gold-answer/system-answer fields in the user message, held
  constant across every call (R-GSJ-030) — see `judge.py`'s module
  docstring for the full argument and citations.
  If no API key is configured, the very first real call raises
  `llm.MissingAPIKeyError` immediately, naming exactly which env var to
  set — there is no silent degradation to the fallback below.
- **`token-overlap`** (`--dry-run`'s `force_fallback=True`, the *only* way
  to reach this backend now — it is no longer also the accidental default
  when a key happens to be missing): a deterministic, clearly-labelled
  heuristic — no network call, no nondeterminism. For answerable items it
  computes the fraction of the gold answer's stopword-stripped content
  words that also appear in the system's answer, and calls it correct at a
  fixed 0.5 threshold. For adversarial items it checks the system's answer
  against a fixed list of abstention phrasings (`"not enough information"`,
  `"cannot determine"`, `"i don't know"`, …). This backend exists so the
  harness never *requires* an API key to run end-to-end in `--dry-run`
  mode (story requirement) — but it is a cruder proxy for correctness than
  the LLM judge, and every `JudgeResult` and every persisted record says
  which backend produced it (`judge_kind`), so a reader is never left
  guessing which numbers are "real."

Live-verified in this session (real Google call, `gemini-3.5-flash-lite`,
cost $0.0002 total for both items — see `.claude/logs/dev.log.md`'s
`[dev-ml]` entry for this story): an answerable item scored `correct=True`
with reason *"FACTUAL MATCH drove the verdict as the system's answer
correctly identifies the medication and dose stated in the gold
reference"*, and an adversarial item scored `correct=True` (i.e. the
system correctly declined) with reason *"The system correctly identified
that there was not enough information to answer the question and declined
to provide a fabricated allergy"* — both rubric-cited, both real network
calls, not stub output.

## 4. Reading the table

Rows are `question_type` (the canonical six:
`medical_reasoning, care_plan_rationale, longitudinal_progression,
cross_admission_comparison, frequency_pattern, adversarial`, in that fixed
order) or `scope` (`single_admission, cross_admission`) — two independent
groupings of the same records, both printed, both written to the per-run
JSON. Columns are `n`, `accuracy`, `mean_tokens`, `p50_lat_ms`, `p95_lat_ms`
— cost (tokens, latency) sits next to correctness in every row on purpose,
because the token/latency cost *is* half of the claim being evaluated, not
an afterthought appended to an accuracy table.

`accuracy` on the `adversarial` row is abstention accuracy, not answerable
accuracy — reading it as "how often the system got the (nonexistent)
right answer" would be a category error; reading it as "how often the
system correctly said it didn't know" is the intended meaning, consistent
with the two-headline-number rule in §3 above.
