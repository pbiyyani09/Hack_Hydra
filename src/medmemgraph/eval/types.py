"""eval/types.py — shared shapes for the eval harness: baselines, harness,
and judge all import from here so there is exactly one `AnswerResult` /
token-estimation implementation instead of three drifting copies.

Not one of the four files named in the story, but a small, deliberate
addition: `nomem.py`, `fullctx.py`, `harness.py`, and `judge.py` all need the
same "how did the system answer, at what token/latency cost" shape, and
`Conversation.token_estimate()` in `pipeline/loader.py` already established
the tiktoken-with-fallback pattern this module reuses for per-string
estimates (loader's version only estimates a whole conversation).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal, Protocol

from medmemgraph.pipeline.loader import Conversation

# ---------------------------------------------------------------------------
# Token estimation — same tiktoken-cl100k_base-with-fallback approach as
# Conversation.token_estimate() (pipeline/loader.py), generalized to any
# string so the fullctx truncation logic can budget line-by-line.
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _encoding():
    import tiktoken

    return tiktoken.get_encoding("cl100k_base")


def estimate_tokens(text: str) -> int:
    """Best-effort token count for `text`. Uses tiktoken's cl100k_base
    encoding when available; falls back to ``chars // 4`` if the encoding
    cache can't be loaded (e.g. an offline sandbox) — an estimate should
    degrade, not raise, matching ``Conversation.token_estimate()``."""
    if not text:
        return 0
    try:
        return len(_encoding().encode(text))
    except Exception:
        return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Model-compatibility guard — Claude Opus 5 / Sonnet 5 / Fable 5 / Mythos 5
# and the 4.6-4.8 Opus/Sonnet family reject any non-default
# temperature/top_p/top_k with a 400 ("Sampling parameters removed" /
# "rejected" — claude-api skill, shared/model-migration.md). Only
# claude-haiku-4-5 and legacy models still accept it. The judge wants
# temperature=0 for repeatability; this guard keeps that request valid if
# MEDMEMGRAPH_JUDGE_MODEL is ever pointed at a newer model.
# ---------------------------------------------------------------------------

_NO_TEMPERATURE_MODELS = frozenset(
    {
        "claude-opus-5",
        "claude-sonnet-5",
        "claude-fable-5",
        "claude-mythos-5",
        "claude-opus-4-8",
        "claude-opus-4-7",
        "claude-opus-4-6",
        "claude-sonnet-4-6",
    }
)


def supports_temperature(model: str) -> bool:
    """Whether `model` accepts a non-default `temperature` parameter."""
    return model not in _NO_TEMPERATURE_MODELS


# ---------------------------------------------------------------------------
# AnswerResult — what every baseline system returns for one QA item.
# ---------------------------------------------------------------------------


@dataclass
class AnswerResult:
    """One system's answer to one question, plus the cost it paid for it.

    `truncated` / `turns_dropped` / `tokens_dropped` are populated only by
    context-bounded systems (fullctx); nomem always reports
    truncated=False since it never builds a context to truncate.
    """

    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    truncated: bool = False
    turns_dropped: int = 0
    tokens_dropped: int = 0
    abstained: bool = False
    """Whether the system itself declined to answer (its own structured
    `abstained` flag, e.g. `reader.Answer.abstained` — never inferred by
    regexing `.text`). Defaults False so baselines that never abstain
    (nomem/fullctx) don't have to set it. Added so a results file can
    distinguish "model declined but judge disagreed" from "model never
    declined" (previously dropped at the `AnswerResult` boundary and lost
    before it ever reached a `Record`)."""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class Answerer(Protocol):
    """The interface every baseline system implements. `harness.py` calls
    `.answer(...)` uniformly across systems; a no-memory system is free to
    (and, per its contract, must) ignore the `conversation` argument."""

    name: str

    def answer(
        self,
        question: str,
        conversation: Conversation | None,
        *,
        patient_id: str,
        scope: str | None = None,
        question_type: str | None = None,
    ) -> AnswerResult: ...
    """`scope` / `question_type` are the benchmark's own gold labels for the
    item. Retrieval-backed systems forward them so `router.route_eval` (the
    frozen rule under test) runs instead of `route_live`'s demo-time keyword
    heuristic; systems that do no retrieval ignore them. Optional so an
    implementation that predates them still satisfies the protocol."""


# ---------------------------------------------------------------------------
# JudgeResult — what judge.Judge.judge() returns for one QA item.
# ---------------------------------------------------------------------------

JudgeMode = Literal["answerable", "abstention"]
JudgeKind = Literal["llm", "token-overlap"]


@dataclass
class JudgeResult:
    """`mode` records which scoring path fired (answerable correctness vs.
    adversarial abstention — story requirement: score these separately,
    never blended) and `judge_kind` records which judge backend produced
    the verdict (real LLM vs. the deterministic no-API-key fallback), so a
    results table can always be traced back to how a number was made."""

    correct: bool
    reason: str
    mode: JudgeMode
    judge_kind: JudgeKind


__all__ = [
    "AnswerResult",
    "Answerer",
    "JudgeResult",
    "JudgeMode",
    "JudgeKind",
    "estimate_tokens",
    "supports_temperature",
]
