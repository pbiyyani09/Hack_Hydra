"""eval/baselines/fullctx.py — the full-context baseline: the entire
patient history goes in the prompt. This is the strongest baseline in the
literature and the one the Pareto pitch has to be honest about
(ARCHITECTURE.md §1: "Mem0's own paper has full-context at 72.90% against
Mem0 at 66.88% on the same backbone ... Do not claim an accuracy win over
full-context"). This baseline is expected to score well; the harness's job
is to show what it *costs* in tokens and latency next to that accuracy, not
to make it lose.

Runs entirely without HydraDB — same as nomem.py.

Truncation contract (story requirement 2): if the patient's full turn
history exceeds the configured model window, this class truncates from the
OLDEST end (drops the earliest turns first, keeps the most recent ones) and
records that truncation happened on the returned `AnswerResult`
(`truncated`, `turns_dropped`, `tokens_dropped`). A silently truncated
baseline is a dishonest baseline (`decisions/001`'s framing, applied here to
truncation instead of leakage) — nothing about this class is allowed to drop
context without saying so on the result it returns.
"""

from __future__ import annotations

import os
import time

from medmemgraph.eval.types import AnswerResult, estimate_tokens
from medmemgraph.pipeline.loader import Conversation

DEFAULT_MODEL = os.environ.get("MEDMEMGRAPH_ANSWERER_MODEL", "claude-haiku-4-5")

_SYSTEM_PROMPT = (
    "You are a clinical question-answering assistant with FULL access to "
    "this patient's dialogue history below, spanning every hospital "
    "admission on record. Use ONLY the provided history and general medical "
    "knowledge — do not invent facts not present in the history. If the "
    "history does not contain enough information to answer, say so plainly, "
    "e.g. 'The provided history does not contain enough information to "
    "answer this.' Answer in one or two sentences."
)

# Reserved for the system prompt, the question, and the model's own output —
# subtracted from the raw context window to get the budget available for
# patient-history turn text. Conservative on purpose: this is an estimate
# (see estimate_tokens), and an over-budget request is a hard API failure
# while an under-budget one just truncates a little more than strictly
# necessary.
_RESERVED_TOKENS = 4_000


class FullContextAnswerer:
    """Answers with the patient's entire turn history in the prompt,
    oldest-first, truncated from the oldest end when it does not fit."""

    name = "fullctx"

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        dry_run: bool = False,
        client: object | None = None,
        max_tokens: int = 512,
        context_window_tokens: int = 190_000,
    ) -> None:
        self.model = model
        self.dry_run = dry_run
        self.max_tokens = max_tokens
        self.context_budget_tokens = max(
            1_000, context_window_tokens - _RESERVED_TOKENS - max_tokens
        )
        self._client = client
        if not dry_run and self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()

    def answer(
        self, question: str, conversation: Conversation | None, *, patient_id: str
    ) -> AnswerResult:
        del patient_id
        context_text, truncated, turns_dropped, tokens_dropped = self._build_context(
            conversation
        )

        if self.dry_run:
            return self._stub_answer(
                question, context_text, truncated, turns_dropped, tokens_dropped
            )

        user_content = (
            f"PATIENT HISTORY (oldest first):\n{context_text}\n\nQUESTION: {question}"
        )
        start = time.monotonic()
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )
        latency_ms = (time.monotonic() - start) * 1000.0
        text = next((b.text for b in response.content if b.type == "text"), "")
        usage = response.usage
        return AnswerResult(
            text=text,
            prompt_tokens=usage.input_tokens,
            completion_tokens=usage.output_tokens,
            latency_ms=latency_ms,
            truncated=truncated,
            turns_dropped=turns_dropped,
            tokens_dropped=tokens_dropped,
        )

    def _build_context(
        self, conversation: Conversation | None
    ) -> tuple[str, bool, int, int]:
        """Flatten every admission's turns, oldest admission first
        (`Conversation.turns()`: "Flat turn list across every admission,
        admission order preserved" — pipeline/loader.py), and drop from the
        front (the oldest turns) until the estimated token count fits the
        budget. Always keeps at least the single most recent turn."""
        if conversation is None:
            return "", False, 0, 0

        turns = conversation.turns()
        lines = [
            f"[admission {t.hadm_id} turn {t.turn_number} · {t.time} · {t.speaker}] {t.text}"
            for t in turns
        ]
        token_counts = [estimate_tokens(line) for line in lines]
        total = sum(token_counts)

        if total <= self.context_budget_tokens or len(lines) <= 1:
            return "\n".join(lines), False, 0, 0

        dropped_tokens = 0
        dropped_turns = 0
        i = 0
        # Drop oldest-first (front of the list) until it fits, but never
        # drop the last remaining turn.
        while total > self.context_budget_tokens and i < len(lines) - 1:
            dropped_tokens += token_counts[i]
            dropped_turns += 1
            total -= token_counts[i]
            i += 1

        return "\n".join(lines[i:]), True, dropped_turns, dropped_tokens

    def _stub_answer(
        self,
        question: str,
        context_text: str,
        truncated: bool,
        turns_dropped: int,
        tokens_dropped: int,
    ) -> AnswerResult:
        """Deterministic, API-free stand-in for --dry-run / CI. Truncation
        is computed above regardless of dry_run — the whole path, including
        the truncation decision, runs under --dry-run (story requirement
        5); only the network call is skipped."""
        start = time.monotonic()
        text = (
            "[dry-run stub, full-context] The provided history does not "
            f"contain enough information to answer: {question[:80]}"
        )
        latency_ms = max((time.monotonic() - start) * 1000.0, 0.001)
        return AnswerResult(
            text=text,
            prompt_tokens=estimate_tokens(context_text)
            + estimate_tokens(question)
            + estimate_tokens(_SYSTEM_PROMPT),
            completion_tokens=estimate_tokens(text),
            latency_ms=latency_ms,
            truncated=truncated,
            turns_dropped=turns_dropped,
            tokens_dropped=tokens_dropped,
        )


__all__ = ["FullContextAnswerer", "DEFAULT_MODEL"]
