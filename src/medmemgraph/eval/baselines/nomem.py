"""eval/baselines/nomem.py — the no-memory baseline: answers from the
question alone, with zero patient history. This is the sanity floor named in
ARCHITECTURE.md's evaluation framing and the BUILD-PLAN baseline ladder:
anything MedMemGraph or any other system scores *near* this means the memory
layer added nothing.

Runs entirely without HydraDB — that is the point (ARCHITECTURE.md §3.2:
"Evidence.baselines --no graph--> Evidence.harness (unblocks eval on day
one)"). This module never imports hydra_client or contracts' graph-facing
pieces.

A `Conversation` is accepted in `.answer(...)` for signature parity with
`FullContextAnswerer` (harness.py calls every system the same way) but is
never read here — that is the whole contract of "no memory". Nothing in this
file constructs a prompt containing patient history.
"""

from __future__ import annotations

import os
import time

from medmemgraph import llm
from medmemgraph.eval.types import AnswerResult, estimate_tokens
from medmemgraph.pipeline.loader import Conversation

DEFAULT_MODEL = os.environ.get("MEDMEMGRAPH_ANSWERER_MODEL", llm.ANSWER_MODEL)
"""Default answerer model. claude-haiku-4-5 is the fast/cheap current model
(claude-api skill, `shared/models.md`) — chosen so the baseline ladder is
affordable to run across a whole patient's QA set; override via
$MEDMEMGRAPH_ANSWERER_MODEL or the `model=` constructor kwarg."""

_SYSTEM_PROMPT = (
    "You are a clinical question-answering assistant. Answer the question "
    "using ONLY general medical knowledge. You have NOT been given this "
    "patient's chart, dialogue history, or any prior admission — you have no "
    "memory of this patient at all, and you must not invent or guess "
    "patient-specific facts (dates, medications, prior visits). If the "
    "question cannot be answered without patient-specific history you do "
    "not have, say so plainly, e.g. 'I don't have enough patient "
    "information to answer this.' Answer in one or two sentences."
)


class NoMemoryAnswerer:
    """Answers strictly from the bare question text via the Anthropic API.

    `dry_run=True` skips the API call entirely and returns a deterministic
    stub — used by `--dry-run` (story requirement 5) so the harness is
    testable in CI without an API key.
    """

    name = "nomem"

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        dry_run: bool = False,
        client: object | None = None,
        max_tokens: int = 512,
    ) -> None:
        self.model = model
        self.dry_run = dry_run
        self.max_tokens = max_tokens
        # `client` is accepted for signature parity with the other baselines and
        # is inert: real calls go through `llm.complete()`, the one seam every
        # LLM-dependent module in this project shares. This class previously did
        # `import anthropic` here — a package removed from `pyproject.toml` when
        # that seam replaced it, so constructing this baseline for a real run
        # raised `ModuleNotFoundError`. Rung 1 of the baseline ladder could not
        # run at all (2026-08-17).
        self._client = client

    def answer(
        self,
        question: str,
        conversation: Conversation | None,
        *,
        patient_id: str,
        scope: str | None = None,
        question_type: str | None = None,
    ) -> AnswerResult:
        del conversation, patient_id  # contract: no-memory never reads history
        # Accepted for Answerer-protocol parity; this system does no
        # retrieval, so there is no router for the gold labels to reach.
        del scope, question_type

        if self.dry_run:
            return self._stub_answer(question)

        response = llm.complete(
            question,
            model=self.model,
            system=_SYSTEM_PROMPT,
            max_tokens=self.max_tokens,
            temperature=0.0,
        )
        return AnswerResult(
            text=response.text,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            latency_ms=response.latency_ms,
        )

    def _stub_answer(self, question: str) -> AnswerResult:
        """Deterministic, API-free stand-in for --dry-run / CI. Deliberately
        never claims patient-specific knowledge, matching the real system
        prompt's abstention instruction, so dry-run exercises the same
        abstention-detection code path the real judge would see."""
        start = time.monotonic()
        text = (
            "[dry-run stub, no-memory] I don't have enough patient "
            f"information to answer: {question[:80]}"
        )
        latency_ms = max((time.monotonic() - start) * 1000.0, 0.001)
        return AnswerResult(
            text=text,
            prompt_tokens=estimate_tokens(question) + estimate_tokens(_SYSTEM_PROMPT),
            completion_tokens=estimate_tokens(text),
            latency_ms=latency_ms,
        )


__all__ = ["NoMemoryAnswerer", "DEFAULT_MODEL"]
