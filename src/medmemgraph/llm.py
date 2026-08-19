"""llm.py — the single seam every LLM-dependent component routes through.

Why this file exists: every LLM-dependent component in this repo
(`pipeline/extract.py`, `eval/judge.py`, `eval/reader.py`) is currently
running on a deterministic fallback because no provider was wired — none of
them import this module yet (rewiring them is a later, separate story; this
story only builds the seam, per explicit instruction not to touch those
three files). Now that OpenAI and Google keys exist, this module is where
every real inference call will flow: one place to cache, meter, cap spend,
and route, instead of three copies of the same plumbing drifting apart.

## Key resolution (never logs a key value, anywhere)

`.env` at the repo root is non-standard: the OpenAI key is named
`OPEN_AI_KEY` (not `OPENAI_API_KEY`), several lines have a space before
`=`, and `python-dotenv`'s own parser silently drops at least one line in
this file (confirmed live in this session: `dotenv_values()` warns
"could not parse statement starting at line 2" and the returned dict is
missing `SEMANTIC_SCHOLAR_API_KEY` even though every other line, including
the space-before-`=` ones, comes through fine). Because a parser that
silently drops lines elsewhere in the file is not a parser to trust blindly
for the two keys this module actually needs, `.env` is read here with a
small hand-rolled, maximally-permissive line parser (`_parse_dotenv`)
instead of `python-dotenv` — a bad line is skipped, not fatal to the whole
file.

Resolution order (env always beats `.env`; the canonical name always beats
the alias, within each source):
  OpenAI: env `OPENAI_API_KEY` -> env `OPEN_AI_KEY` -> .env `OPENAI_API_KEY`
          -> .env `OPEN_AI_KEY`
  Google: env `GOOGLE_API_KEY` -> env `GEMINI_API_KEY` -> .env
          `GOOGLE_API_KEY` -> .env `GEMINI_API_KEY`

## Model policy — cost-minimal, no reasoning models

`EXTRACT_MODEL` / `ANSWER_MODEL` / `JUDGE_MODEL` / `EMBED_MODEL` below are
all non-reasoning, cheapest-tier models, each overridable via a
`MEDMEMGRAPH_*_MODEL` env var. `JUDGE_MODEL` is deliberately a *different
model family* (Google) from `ANSWER_MODEL` (OpenAI): literature/17
(`collaborative/literature/17-genai-structured-generation-and-judging.md`)
documents self-preference bias large enough to flip head-to-head rankings
— one cited study found answer-order manipulation flipped 66 of 80
verdicts — and cross-family judging is the measurement-supported
mitigation. This is a deliberate design choice, argued once here, not an
accident of which key happened to be available.

`gpt-5*` / `o*` / `*-pro` (reasoning/premium tiers) are never a hardcoded
default here (`_assert_not_reasoning_default` self-checks the literal
fallback strings at import time as a regression guard) — but nothing stops
a caller from passing one explicitly via `model=...` or a
`MEDMEMGRAPH_*_MODEL` env override, which is an explicit choice, not a
default.

## Structured output — native schema, never a bare "respond in JSON"

When `schema` (a JSON Schema dict) is given, `complete()` uses each
provider's native structured-output mode (OpenAI
`response_format={"type": "json_schema", ...}` with `strict: True`;
Google `response_mime_type="application/json"` + `response_schema=...`)
rather than a prompted "respond in JSON" instruction — literature/17 found
naive JSON-mode measurably degrades content quality (up to 63 points on one
benchmark) while true schema-constrained decoding does not. The parsed
object is validated against `schema` with a small hand-rolled validator
(`_validate_against_schema` — this repo has no `jsonschema` dependency and
adding one is out of this story's scope) and retried on mismatch.

## Cost control — the point of this module

1. A disk cache at `CACHE_DIR` (default `data/llm_cache/`, override via
   `MEDMEMGRAPH_LLM_CACHE_DIR`), keyed by SHA-256 of
   `(kind, model, system, prompt-or-text, schema, temperature)` — `kind`
   ("complete" vs "embed") is a namespacing addition beyond the literal
   `(model, system, prompt, schema, temperature)` tuple, so the two call
   types can never collide on the same key. A cache hit costs $0, makes no
   network call, and is marked `cached=True`.
2. A process-wide `Ledger`, persisted to `CACHE_DIR/ledger.json`, tracking
   calls/cache-hits/tokens/USD per model, so cost survives across runs, not
   just within one process.
3. `MEDMEMGRAPH_MAX_USD` (default `$5.00`, read fresh from the environment
   on every call, not frozen at import) is a hard cap: before any real
   network call, the estimated cost (local token estimate for the prompt,
   `max_tokens` as the worst-case completion estimate — an overestimate
   stops early, which is the safe direction) is reserved against the cap;
   `BudgetExceeded` is raised *before* the request goes out if committed +
   in-flight + estimated would exceed it.
4. `PRICING` is a module constant, $/1,000,000 tokens, per model. See the
   comment above `PRICING` for exactly what was verified, how, and when.
5. `dry_run=True` returns a deterministic stub (schema-shaped when a schema
   is given) with zero network calls and $0 cost — tests and CI cost
   nothing.

## Truncation — detected, escalated, and costed (not silently repeated)

A schema'd response cut off by `max_tokens` (provider `finish_reason`
`length`/`MAX_TOKENS`, or — when a provider/SDK doesn't expose that — a JSON
parse failure whose completion-token count is at the ceiling) is a
DIFFERENT failure from a genuine schema violation, and gets a different
response: `_complete_with_schema_retry` escalates `max_tokens` for the next
attempt (`_next_truncation_max_tokens`, capped at
`_TRUNCATION_MAX_TOKENS_CAP`) instead of re-issuing an identical doomed
request, `LLMResponse.truncated` exposes it as a distinct, countable outcome
on success, and `TruncationError` (a `SchemaValidationError` subtype)
distinguishes it on exhaustion. `DEFAULT_SCHEMA_MAX_TOKENS` also raises the
starting point for schema'd calls that don't pass `max_tokens` explicitly.
Every real, provider-billed token spent on a discarded attempt along the
way — truncated or not — is recorded via `Ledger.record_failed_attempt`,
surfaced as its own line in `Ledger.report()`, so that spend is no longer
invisible. See each symbol's own docstring for the full detail; this
section exists so a reader of the module docstring doesn't miss that the
behavior exists at all.

## Known limitation, stated rather than hidden

The cache key deliberately does NOT include `max_tokens` or `max_retries`
(the story's own key tuple is `(model, system, prompt, schema,
temperature)`) — two calls that differ only in `max_tokens` will share a
cache entry. Acceptable for this project's use (callers pass a consistent
`max_tokens` per call site) but worth knowing if that stops being true. This
is unchanged by the truncation-escalation retry above: escalation only ever
raises `max_tokens` on a *within-call* retry attempt (never persisted to the
cache key), and a cache HIT always short-circuits before any real call is
even considered, so it can never observe an in-flight escalation.

`Ledger.check_budget`'s pre-flight cap check also stays exactly as loose as
it was before this fix: it reserves one call's worth of estimated cost
(`max_tokens` as the worst case) before `_complete_with_schema_retry` runs,
which can itself make up to `_MAX_SCHEMA_RETRIES + 1` real, billed attempts
under that single reservation — a pre-existing looseness this story did not
ask to close, and a real (if narrow) gap: `total_cost_usd` — what
`check_budget` sums against the cap — still counts only successful/
committed spend, not `total_failed_cost_usd`. Flagged, not silently
decided either way; see those two properties' docstrings.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, NamedTuple

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class LLMError(Exception):
    """Base class for every error this module raises on purpose."""


class MissingAPIKeyError(LLMError):
    """Raised when a required provider API key cannot be resolved from any
    source. Never carries a key value — only names which env vars / .env
    keys were checked."""


class BudgetExceeded(LLMError):
    """Raised instead of making a call that would push cumulative spend
    (this process's Ledger) past MEDMEMGRAPH_MAX_USD."""


class SchemaValidationError(LLMError):
    """Raised when a schema'd response still fails validation after all
    retries."""


class TruncationError(SchemaValidationError):
    """Raised when a schema'd response is STILL being cut off by
    ``max_tokens`` after every truncation-escalation retry is exhausted (see
    ``_next_truncation_max_tokens``) — a distinct, narrower subtype of
    ``SchemaValidationError``, not a sibling of it: a truncated response
    genuinely IS a schema-validation failure (incomplete JSON never
    validates), so existing callers that only catch the base
    ``SchemaValidationError`` keep working unmodified; new callers that want
    to react differently to "ran out of tokens" vs "the model ignored the
    schema shape" can ``except TruncationError`` first.

    Before this fix both causes raised the exact same
    ``SchemaValidationError`` with no way to tell them apart even though
    they need different responses: a genuine schema violation means the
    schema (or the model) is wrong; a truncation means the request needs
    more headroom, not a different schema."""


class LocalModelError(LLMError):
    """A local (`local:`) model could not be loaded or run.

    Raised by the memory pre-flight before any weight is read, and for bad
    quantization settings. An `LLMError` subclass so existing `except LLMError`
    handlers — `pipeline/resolve.py`'s entity-match degrade, for instance —
    treat it like any other provider failure rather than crashing a run."""


class ProviderError(LLMError):
    """Raised when a provider call fails fatally (non-retryable) or retries
    are exhausted. Wraps the original exception via `raise ... from exc`."""


# ---------------------------------------------------------------------------
# Key resolution
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DOTENV_PATH = _REPO_ROOT / ".env"


def _parse_dotenv(path: Path) -> dict[str, str]:
    """Small, maximally-permissive `.env` line parser.

    Deliberately not `python-dotenv`: confirmed live in this session that
    `dotenv_values()` on this repo's real `.env` warns "could not parse
    statement starting at line 2" and silently drops that line's key
    (`SEMANTIC_SCHOLAR_API_KEY`) from the returned dict, while every other
    line — including the several with a space before `=` — comes through
    fine. A parser that silently drops lines elsewhere in the file is not
    trustworthy for the two keys this module actually needs, so: read the
    file ourselves, one line at a time, tolerate anything we don't
    understand (skip it), never raise on a malformed line.
    """
    result: dict[str, str] = {}
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return result
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key:
            result[key] = value
    return result


def _resolve_key(
    names: list[str],
    *,
    env: Mapping[str, str],
    dotenv_map: Mapping[str, str],
) -> str | None:
    """`names` in priority order, env source before .env source (matching
    the story's exact order: every env name is checked before any .env
    name)."""
    for name in names:
        value = env.get(name)
        if value and value.strip():
            return value.strip()
    for name in names:
        value = dotenv_map.get(name)
        if value and value.strip():
            return value.strip()
    return None


def resolve_openai_key(
    *,
    env: Mapping[str, str] | None = None,
    dotenv_path: str | Path | None = None,
) -> str:
    """Resolve the OpenAI key: env `OPENAI_API_KEY` -> env `OPEN_AI_KEY` ->
    .env `OPENAI_API_KEY` -> .env `OPEN_AI_KEY`. Raises `MissingAPIKeyError`
    naming exactly what was checked (never the value) if none resolve."""
    env = env if env is not None else os.environ
    path = Path(dotenv_path) if dotenv_path is not None else _DEFAULT_DOTENV_PATH
    dotenv_map = _parse_dotenv(path)
    key = _resolve_key(["OPENAI_API_KEY", "OPEN_AI_KEY"], env=env, dotenv_map=dotenv_map)
    if not key:
        raise MissingAPIKeyError(
            "OpenAI API key not found. Checked (in order): env OPENAI_API_KEY, "
            "env OPEN_AI_KEY, .env OPENAI_API_KEY, .env OPEN_AI_KEY (looked in "
            f"{path}). Set one of these before calling an OpenAI model — e.g. "
            "`export OPENAI_API_KEY=sk-...` or add `OPEN_AI_KEY=...` to .env."
        )
    return key


def resolve_google_key(
    *,
    env: Mapping[str, str] | None = None,
    dotenv_path: str | Path | None = None,
) -> str:
    """Resolve the Google key: env `GOOGLE_API_KEY` -> env `GEMINI_API_KEY`
    -> .env `GOOGLE_API_KEY` -> .env `GEMINI_API_KEY`. Raises
    `MissingAPIKeyError` naming exactly what was checked (never the value)
    if none resolve."""
    env = env if env is not None else os.environ
    path = Path(dotenv_path) if dotenv_path is not None else _DEFAULT_DOTENV_PATH
    dotenv_map = _parse_dotenv(path)
    key = _resolve_key(["GOOGLE_API_KEY", "GEMINI_API_KEY"], env=env, dotenv_map=dotenv_map)
    if not key:
        raise MissingAPIKeyError(
            "Google API key not found. Checked (in order): env GOOGLE_API_KEY, "
            "env GEMINI_API_KEY, .env GOOGLE_API_KEY, .env GEMINI_API_KEY "
            f"(looked in {path}). Set one of these before calling a Gemini "
            "model — e.g. `export GOOGLE_API_KEY=...` or add "
            "`GOOGLE_API_KEY=...` to .env (get one at "
            "https://aistudio.google.com/apikey)."
        )
    return key


# ---------------------------------------------------------------------------
# Model policy
# ---------------------------------------------------------------------------

_BANNED_DEFAULT_PREFIXES = ("gpt-5", "o1", "o3", "o4", "o-")
_BANNED_DEFAULT_SUFFIXES = ("-pro",)


def _assert_not_reasoning_default(name: str, value: str) -> None:
    """Self-check applied only to this module's own hardcoded fallback
    strings (not to env-overridden or per-call values — an explicit
    override is exactly the escape hatch the story allows)."""
    lowered = value.lower()
    if lowered.startswith(_BANNED_DEFAULT_PREFIXES) or lowered.endswith(_BANNED_DEFAULT_SUFFIXES):
        raise LLMError(
            f"{name} defaults to {value!r}, which looks like a reasoning/"
            "premium-tier model — banned as a hardcoded default per project "
            "cost policy (minimal spend, no reasoning models). Pass it "
            "explicitly via model=... if that model is really what you want."
        )


_HARDCODED_EXTRACT_MODEL = "gemini-3.5-flash-lite"
_HARDCODED_ANSWER_MODEL = "gpt-4.1-mini"
_HARDCODED_JUDGE_MODEL = "gemini-3.5-flash-lite"
_HARDCODED_EMBED_MODEL = "text-embedding-3-small"

for _name, _value in (
    ("EXTRACT_MODEL", _HARDCODED_EXTRACT_MODEL),
    ("ANSWER_MODEL", _HARDCODED_ANSWER_MODEL),
    ("JUDGE_MODEL", _HARDCODED_JUDGE_MODEL),
):
    _assert_not_reasoning_default(_name, _value)
del _name, _value

EXTRACT_MODEL = os.environ.get("MEDMEMGRAPH_EXTRACT_MODEL", _HARDCODED_EXTRACT_MODEL)
"""Google, gemini-3.5-flash-lite: the bulk pass over the ~6.7M-token corpus
— cheapest non-reasoning tier, this is the highest-volume call site."""

ANSWER_MODEL = os.environ.get("MEDMEMGRAPH_ANSWER_MODEL", _HARDCODED_ANSWER_MODEL)
"""OpenAI, gpt-4.1-mini: the reader / Chain-of-Note answerer."""

JUDGE_MODEL = os.environ.get("MEDMEMGRAPH_JUDGE_MODEL", _HARDCODED_JUDGE_MODEL)
"""Google, gemini-3.5-flash-lite: DIFFERENT FAMILY from ANSWER_MODEL,
deliberately — see the module docstring's "Model policy" section for the
literature/17 self-preference-bias argument."""

EMBED_MODEL = os.environ.get("MEDMEMGRAPH_EMBED_MODEL", _HARDCODED_EMBED_MODEL)
"""OpenAI, text-embedding-3-small: cheapest embedding tier."""


def resolve_key_for_model(model: str) -> str:
    """The API key `model` would use, resolved the same way a real call resolves
    it. Raises `LLMError` (`MissingAPIKeyError`, or a routing error for an
    unroutable model name) if it is not configured.

    Exists so a caller can answer "is real inference actually available?" without
    probing one hardcoded env-var name. There are four accepted key spellings
    across two providers and `.env` is also consulted, so an
    `os.environ.get("SOME_KEY")` test gives false negatives — which is exactly
    how `pipeline/resolve.py` ended up warning that it was degrading to
    exact-match-only on runs that had perfectly good credentials.

    Never returns the key in a log or an error message; only to the caller."""
    provider = _provider_for_model(model)
    if provider == "local":
        return ""  # weights on disk; nothing to authenticate
    return resolve_openai_key() if provider == "openai" else resolve_google_key()


def _provider_for_model(model: str) -> str:
    """Route by model-name prefix: `gpt-`/`o<digit>`/`text-embedding-` ->
    "openai"; `gemini-`/`gemma-` -> "google"."""
    m = model.strip().lower()
    if m.startswith(LOCAL_MODEL_PREFIX):
        return "local"
    if m.startswith("gemini-") or m.startswith("gemma-"):
        return "google"
    if m.startswith("gpt-") or m.startswith("text-embedding") or (
        len(m) >= 2 and m[0] == "o" and m[1].isdigit()
    ):
        return "openai"
    raise LLMError(
        f"Cannot route model {model!r} to a provider: expected a "
        "gpt-*/o<digit>*/text-embedding-* prefix (OpenAI), a "
        f"gemini-*/gemma-* prefix (Google), or a {LOCAL_MODEL_PREFIX!r} prefix "
        "(local weights, e.g. 'local:Qwen/Qwen2.5-7B-Instruct')."
    )


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------


class _Price(NamedTuple):
    input: float   # USD per 1,000,000 input/prompt tokens
    output: float  # USD per 1,000,000 output/completion tokens


# Verified 2026-08-17 in this session (fetched live, not from training-data
# memory) against two independent sources that agreed exactly:
#   - the OpenRouter public models API, https://openrouter.ai/api/v1/models
#     (per-token prices: gpt-4.1-mini prompt=4e-7/completion=1.6e-6;
#     gemini-2.5-flash-lite prompt=1e-7/completion=4e-7)
#   - BerriAI/litellm's community-maintained pricing table,
#     raw.githubusercontent.com/BerriAI/litellm/main/litellm/
#     model_prices_and_context_window_backup.json (same two models, same
#     per-token numbers exactly; also the only source of the two that lists
#     an embeddings price at all — OpenRouter does not proxy embeddings)
# Not fetched from either provider's own pricing page directly (JS-rendered,
# not scrapeable via `curl` in this sandbox) — two independent third-party
# aggregators agreeing exactly is the verification basis, stated plainly
# rather than claimed as "fetched from the primary source."
PRICING: dict[str, _Price] = {
    "gpt-4.1-mini": _Price(input=0.40, output=1.60),            # verified 2026-08-17
    "gemini-2.5-flash-lite": _Price(input=0.10, output=0.40),   # verified 2026-08-17 (404s on this key)
    # gemini-2.5-flash-lite 404s ("no longer available to new users") on the project key,
    # so 3.5-flash-lite is the shipped default. Same lite tier; priced conservatively HIGH
    # until a real quote is confirmed, so the budget cap trips early rather than late.
    "gemini-3.5-flash-lite": _Price(input=0.20, output=0.80),   # conservative estimate 2026-08-17
    "gemini-3.5-flash": _Price(input=0.40, output=1.60),        # conservative estimate 2026-08-17
    "text-embedding-3-small": _Price(input=0.02, output=0.0),   # verified 2026-08-17 (litellm only)
}

_FALLBACK_PRICE = _Price(input=15.0, output=75.0)
"""Deliberately conservative-HIGH placeholder for any model string not in
PRICING (e.g. a caller passes something we have no verified price for).
Roughly comparable to the priciest top-tier/reasoning-model rates publicly
observed — NOT a real quote for any specific model — chosen specifically so
an un-priced model trips the budget cap quickly instead of silently costing
$0 and defeating it. An overestimate stops early; that is the safe
direction (see module docstring)."""


def _cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    # Local weights cost nothing per token. Without this they fall through to
    # `_FALLBACK_PRICE` ($15/$75 per 1M, deliberately punitive for un-priced
    # models) and the budget cap refuses free inference within a few calls.
    # Electricity and the GPU are real costs, but they are not per-token and
    # this ledger measures per-token API spend.
    if model.startswith(LOCAL_MODEL_PREFIX):
        return 0.0
    price = PRICING.get(model, _FALLBACK_PRICE)
    return (prompt_tokens / 1_000_000) * price.input + (completion_tokens / 1_000_000) * price.output


DEFAULT_MAX_USD = 5.00


def _max_usd() -> float:
    """Read fresh every call (not frozen at import) so a test or a caller can
    `monkeypatch`/`setenv` it per-call.

    Checks `os.environ` FIRST, then falls back to `.env` — the same two-source
    order `resolve_openai_key`/`resolve_google_key` use, and for the same
    reason. Reading only `os.environ` made the effective cap depend on import
    ORDER: a value in `.env` took effect only if some other module happened to
    have called `load_dotenv()` first (`eval/judge.py` and `hydra_client.py`
    both do, at import). So a process that imported `llm` alone silently fell
    back to the $5.00 default, while a harness process saw the configured $50 —
    same repo, same `.env`, different cap, no warning either way. Observed
    2026-08-18: a bare `llm.complete()` refused a call for exceeding a cap the
    operator had explicitly raised.

    A spend limit whose value depends on which module imported first is not a
    spend limit."""
    raw = os.environ.get("MEDMEMGRAPH_MAX_USD")
    if raw is None or not raw.strip():
        raw = _parse_dotenv(_DEFAULT_DOTENV_PATH).get("MEDMEMGRAPH_MAX_USD")
    if raw is None or not raw.strip():
        return DEFAULT_MAX_USD
    try:
        return float(raw)
    except ValueError:
        return DEFAULT_MAX_USD


# ---------------------------------------------------------------------------
# Token estimation (local, offline — same tiktoken-cl100k_base-with-fallback
# shape as pipeline/loader.py's Conversation.token_estimate(), reimplemented
# self-contained here rather than importing across module boundaries)
# ---------------------------------------------------------------------------

_encoding_cache: list[Any] = []


def _encoding() -> Any:
    if not _encoding_cache:
        import tiktoken

        _encoding_cache.append(tiktoken.get_encoding("cl100k_base"))
    return _encoding_cache[0]


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    try:
        return len(_encoding().encode(text))
    except Exception:
        return max(1, len(text) // 4)


def _estimate_tokens_offline(text: str) -> int:
    """Character-based token estimate that NEVER attempts network I/O —
    unlike `_estimate_tokens`, which may lazily download tiktoken's
    cl100k_base BPE file on first use if it isn't already present in
    tiktoken's local cache. `dry_run=True`'s "no network call, ever" is a
    hard requirement (not just a best-effort default like the real-call
    path's pre-flight budget estimate), so the dry_run code paths use this
    instead of `_estimate_tokens`."""
    if not text:
        return 0
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Ledger — process-wide (or explicitly-scoped) spend/usage accounting
# ---------------------------------------------------------------------------


@dataclass
class _ModelStats:
    calls: int = 0
    cache_hits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    failed_attempts: int = 0
    failed_prompt_tokens: int = 0
    failed_completion_tokens: int = 0
    failed_cost_usd: float = 0.0
    """`failed_*` fields (all default 0/0.0, so an old `ledger.json` written
    before this fix — which has none of these keys — still loads cleanly via
    `_ModelStats(**stats)`) track schema-retry attempts that produced a REAL,
    billed completion (truncated or schema-invalid) which was then discarded
    in favor of a later retry — tokens a provider charged for that every cost
    figure this project reported before this fix silently excluded. Kept
    fully separate from `calls`/`cost_usd`/etc. above: those fields keep
    meaning exactly what they meant before (the call actually returned to
    the caller, cache hit or not) — nothing about existing cost/call
    arithmetic changes. See `Ledger.record_failed_attempt`."""


CACHE_DIR = Path(os.environ.get("MEDMEMGRAPH_LLM_CACHE_DIR", "data/llm_cache"))
"""Mutable module global (deliberately not captured as a function default
argument anywhere in this file) so a test can
`monkeypatch.setattr(llm, "CACHE_DIR", tmp_path)` and every cache/ledger
path lookup that happens after that point sees the new value."""


class Ledger:
    """Tracks calls/cache-hits/tokens/USD per model, persisted to disk as
    JSON so cost survives across process runs, not just within one.

    `check_budget` / `release_reservation` implement a reserve-then-commit
    pattern so concurrent callers (`complete_many`) cannot collectively
    overshoot the cap: the estimated cost of a call is added to
    `_reserved_usd` (under the lock) *before* the network round trip, and
    only released after — so a second concurrent call's budget check sees
    the first call's in-flight spend, not just what has already been
    `record`-ed.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else (CACHE_DIR / "ledger.json")
        self._lock = threading.RLock()
        self._by_model: dict[str, _ModelStats] = {}
        self._reserved_usd = 0.0
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        for model, stats in raw.get("by_model", {}).items():
            try:
                self._by_model[model] = _ModelStats(**stats)
            except TypeError:
                continue

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"by_model": {m: asdict(s) for m, s in self._by_model.items()}}
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    @property
    def total_cost_usd(self) -> float:
        """Successful/committed spend only — exactly what this property
        meant before this fix (the call that was actually returned to a
        caller, cache hit or not). Does NOT include `total_failed_cost_usd`
        below; see that property's docstring for why, and for what still
        does account for it (`check_budget`'s cap check does not currently
        include failed-attempt cost either — a known, explicitly-scoped-out
        gap, not an oversight; flagged in this story's return note)."""
        with self._lock:
            return sum(s.cost_usd for s in self._by_model.values())

    @property
    def total_failed_cost_usd(self) -> float:
        """Real, provider-billed cost of every discarded schema-retry
        attempt (truncated or schema-invalid) across all models — the
        "floor vs actual" gap this fix makes visible. See
        `record_failed_attempt`."""
        with self._lock:
            return sum(s.failed_cost_usd for s in self._by_model.values())

    def record(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
        *,
        cached: bool = False,
    ) -> None:
        with self._lock:
            stats = self._by_model.setdefault(model, _ModelStats())
            stats.calls += 1
            if cached:
                stats.cache_hits += 1
            stats.prompt_tokens += prompt_tokens
            stats.completion_tokens += completion_tokens
            stats.cost_usd += cost_usd
            self._save()

    def record_failed_attempt(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
    ) -> None:
        """Record a schema-retry attempt that produced a real (provider-
        billed) completion which was then discarded — truncated by
        max_tokens, or valid JSON that failed schema validation — and
        retried rather than returned to the caller.

        Deliberately a separate method from `record()`, not an optional
        `failed=True` flag on it: `record()`'s `calls`/`cache_hits` counters
        mean "an outcome was returned to a caller" everywhere else in this
        module and in every existing test — overloading it here would make
        that meaning ambiguous. A failed attempt is never cached (there is
        nothing worth caching) and never counted as a `call` in that sense;
        it lands only in the dedicated `failed_*` fields, which `report()`
        below surfaces as its own line so the previously-invisible waste is
        visible instead of merely absent — see decisions/ audit context in
        this module's own story history for why this method exists at all.
        """
        with self._lock:
            stats = self._by_model.setdefault(model, _ModelStats())
            stats.failed_attempts += 1
            stats.failed_prompt_tokens += prompt_tokens
            stats.failed_completion_tokens += completion_tokens
            stats.failed_cost_usd += cost_usd
            self._save()

    def check_budget(self, estimated_additional_usd: float, cap_usd: float) -> None:
        """Raise BudgetExceeded if committed + already-in-flight + this
        call's estimate would exceed `cap_usd`; otherwise reserve the
        estimate (caller must `release_reservation` the same amount once
        the call finishes, success or failure)."""
        with self._lock:
            committed = sum(s.cost_usd for s in self._by_model.values())
            projected = committed + self._reserved_usd + estimated_additional_usd
            if projected > cap_usd:
                raise BudgetExceeded(
                    f"Refusing this call: committed ${committed:.4f} + in-flight "
                    f"${self._reserved_usd:.4f} + this call's estimate "
                    f"${estimated_additional_usd:.4f} = ${projected:.4f} would "
                    f"exceed the ${cap_usd:.2f} cap (MEDMEMGRAPH_MAX_USD). Raise "
                    "MEDMEMGRAPH_MAX_USD, pass dry_run=True, or reduce "
                    "max_tokens / batch size."
                )
            self._reserved_usd += estimated_additional_usd

    def release_reservation(self, estimated_additional_usd: float) -> None:
        with self._lock:
            self._reserved_usd = max(0.0, self._reserved_usd - estimated_additional_usd)

    def report(self) -> str:
        """Two blocks: successful/committed calls (unchanged shape from
        before this fix), then — its own line/section, never merged into
        the first table or its TOTAL — failed schema-retry attempts: real,
        provider-billed tokens spent on a truncated or schema-invalid
        response that was discarded and retried. Before this fix that spend
        was recorded nowhere, so every cost figure this method ever printed
        was a floor, not an actual; this section is what makes the gap
        visible rather than silently absent."""
        with self._lock:
            header = f"{'model':32s} {'calls':>6s} {'hits':>6s} {'prompt_tok':>12s} {'compl_tok':>12s} {'cost_usd':>10s}"
            lines = [header, "-" * len(header)]
            total_calls = total_hits = total_prompt = total_completion = 0
            total_cost = 0.0
            total_failed_attempts = total_failed_prompt = total_failed_completion = 0
            total_failed_cost = 0.0
            for model in sorted(self._by_model):
                s = self._by_model[model]
                lines.append(
                    f"{model:32s} {s.calls:6d} {s.cache_hits:6d} {s.prompt_tokens:12d} "
                    f"{s.completion_tokens:12d} {s.cost_usd:10.4f}"
                )
                total_calls += s.calls
                total_hits += s.cache_hits
                total_prompt += s.prompt_tokens
                total_completion += s.completion_tokens
                total_cost += s.cost_usd
                total_failed_attempts += s.failed_attempts
                total_failed_prompt += s.failed_prompt_tokens
                total_failed_completion += s.failed_completion_tokens
                total_failed_cost += s.failed_cost_usd
            lines.append("-" * len(header))
            lines.append(
                f"{'TOTAL':32s} {total_calls:6d} {total_hits:6d} {total_prompt:12d} "
                f"{total_completion:12d} {total_cost:10.4f}"
            )
            if self._reserved_usd:
                lines.append(f"(in-flight, not yet committed: ${self._reserved_usd:.4f})")

            lines.append("")
            failed_header = (
                f"{'model (failed attempts)':32s} {'attempts':>8s} {'prompt_tok':>12s} "
                f"{'compl_tok':>12s} {'cost_usd':>10s}"
            )
            lines.append(failed_header)
            lines.append("-" * len(failed_header))
            if total_failed_attempts:
                for model in sorted(self._by_model):
                    s = self._by_model[model]
                    if not s.failed_attempts:
                        continue
                    lines.append(
                        f"{model:32s} {s.failed_attempts:8d} {s.failed_prompt_tokens:12d} "
                        f"{s.failed_completion_tokens:12d} {s.failed_cost_usd:10.4f}"
                    )
            lines.append(
                f"{'FAILED (discarded, still billed)':32s} {total_failed_attempts:8d} "
                f"{total_failed_prompt:12d} {total_failed_completion:12d} {total_failed_cost:10.4f}"
            )
            return "\n".join(lines)

    def reset(self) -> None:
        with self._lock:
            self._by_model = {}
            self._reserved_usd = 0.0
            self._save()


_ledger: Ledger | None = None
_ledger_singleton_lock = threading.Lock()


def get_ledger() -> Ledger:
    """The process-wide Ledger singleton, created on first use.

    Double-checked locking: `complete_many`'s whole point is concurrent
    callers, and an unsynchronized "if None: create" here would let two
    threads each construct their own `Ledger` instance (each with its own
    unshared `RLock`) pointed at the same `ledger.json` — a real bug found
    and fixed during this story's own test run, not a hypothetical: it
    surfaced as concurrent, un-mutexed `_save()` calls corrupting the
    tmp-file rename (`FileNotFoundError` on `tmp.replace(path)`) under
    `complete_many`'s `test_respects_max_concurrency` test.
    """
    global _ledger
    if _ledger is None:
        with _ledger_singleton_lock:
            if _ledger is None:
                _ledger = Ledger()
    return _ledger


def reset_ledger(path: str | Path | None = None) -> Ledger:
    """Replace the process-wide Ledger singleton with a fresh one — a test
    or CLI convenience, not part of the normal call path."""
    global _ledger
    with _ledger_singleton_lock:
        _ledger = Ledger(path=path)
    return _ledger


# ---------------------------------------------------------------------------
# Disk cache
# ---------------------------------------------------------------------------


def _cache_key(
    *,
    kind: str,
    model: str,
    system: str | None,
    prompt_or_text: str,
    schema: dict | None,
    temperature: float | None,
) -> str:
    payload = {
        "kind": kind,
        "model": model,
        "system": system,
        "prompt": prompt_or_text,
        "schema": schema,
        "temperature": temperature,
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.json"


def _cache_get(key: str) -> dict | None:
    path = _cache_path(key)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _cache_put(key: str, payload: dict) -> None:
    path = _cache_path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Schema validation — a small, deliberately-narrow JSON Schema subset
# (object/properties/required/additionalProperties, array/items,
# string/integer/number/boolean/null/object/array types, enum) matching
# what this project's own structured-output schemas actually look like
# (flat objects of primitives — see eval/judge.py's _ANSWERABLE_SCHEMA for
# the existing precedent). Not a general JSON Schema implementation; this
# repo has no `jsonschema` dependency and adding one is out of this story's
# scope (a new dependency is a HALT trigger, not a default reflex).
# ---------------------------------------------------------------------------


def _type_matches(value: Any, expected: Any) -> bool:
    types = expected if isinstance(expected, list) else [expected]
    for t in types:
        if t == "object" and isinstance(value, dict):
            return True
        if t == "array" and isinstance(value, list):
            return True
        if t == "string" and isinstance(value, str):
            return True
        if t == "integer" and isinstance(value, int) and not isinstance(value, bool):
            return True
        if t == "number" and isinstance(value, (int, float)) and not isinstance(value, bool):
            return True
        if t == "boolean" and isinstance(value, bool):
            return True
        if t == "null" and value is None:
            return True
    return False


def _check_node(value: Any, schema: dict, path: str, errors: list[str]) -> None:
    if not isinstance(schema, dict):
        return
    expected_type = schema.get("type")
    if expected_type and not _type_matches(value, expected_type):
        errors.append(f"{path}: expected type {expected_type!r}, got {type(value).__name__}")
        return
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} not in enum {schema['enum']!r}")
        return
    if expected_type == "object" and isinstance(value, dict):
        props = schema.get("properties", {}) or {}
        for req in schema.get("required", []) or []:
            if req not in value:
                errors.append(f"{path}: missing required property {req!r}")
        if schema.get("additionalProperties") is False:
            extra = sorted(set(value) - set(props))
            if extra:
                errors.append(f"{path}: unexpected properties {extra!r}")
        for k, v in value.items():
            if k in props:
                _check_node(v, props[k], f"{path}.{k}", errors)
    elif expected_type == "array" and isinstance(value, list):
        items_schema = schema.get("items")
        if isinstance(items_schema, dict):
            for i, item in enumerate(value):
                _check_node(item, items_schema, f"{path}[{i}]", errors)


def _validate_against_schema(data: Any, schema: dict) -> list[str]:
    """Return a list of human-readable violations; empty list == valid."""
    errors: list[str] = []
    _check_node(data, schema, "$", errors)
    return errors


def _stub_value_for_schema(schema: dict) -> Any:
    """A deterministic, schema-shaped placeholder value — used by
    dry_run=True so callers that read `.parsed` still get something
    structurally sane, without any network call."""
    if not isinstance(schema, dict):
        return None
    if "enum" in schema and schema["enum"]:
        return schema["enum"][0]
    t = schema.get("type")
    if isinstance(t, list):
        t = t[0] if t else None
    if t == "object" or (t is None and "properties" in schema):
        props = schema.get("properties", {}) or {}
        return {k: _stub_value_for_schema(v) for k, v in props.items()}
    if t == "array":
        items = schema.get("items")
        return [_stub_value_for_schema(items)] if isinstance(items, dict) else []
    if t == "integer":
        return 0
    if t == "number":
        return 0.0
    if t == "boolean":
        return False
    if t == "null":
        return None
    return ""


# ---------------------------------------------------------------------------
# Retry / backoff
# ---------------------------------------------------------------------------

DEFAULT_MAX_RETRIES = 5
_RETRY_BASE_DELAY_S = 1.0
_RETRY_MAX_DELAY_S = 30.0
_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
_MAX_SCHEMA_RETRIES = 2  # extra full round trips (on top of the first) on schema mismatch

_TRUNCATION_ESCALATION_FACTOR = 4
_TRUNCATION_MAX_TOKENS_CAP = 32_768
"""Truncation-specific retry policy (see `_complete_with_schema_retry`).

The measured defect: a 60-85 turn admission's structured-extraction response
was cut off by `max_tokens=4096` near 12.4-12.8k output characters, and the
un-fixed retry re-issued the *identical* oversized request — same
`max_tokens`, guaranteed to truncate again. `_next_truncation_max_tokens`
instead escalates by `_TRUNCATION_ESCALATION_FACTOR` each time truncation is
detected, so the existing `_MAX_SCHEMA_RETRIES + 1` = 3 attempts spend their
budget on genuinely more headroom (e.g. 4096 -> 16384 -> 32768) rather than
repeating a doomed call. Capped at `_TRUNCATION_MAX_TOKENS_CAP` — well within
what both provider families support for structured JSON output — so a
pathological input cannot balloon a single call's cost/latency unboundedly;
if the cap is reached and the response is still truncated, `TruncationError`
surfaces that plainly (see the raise at the bottom of
`_complete_with_schema_retry`) instead of failing silently or looping
forever."""


def _next_truncation_max_tokens(current: int) -> int:
    """Escalate `max_tokens` after a detected truncation. Never decreases
    (even if `current` is already at or past the cap, `min(...)` just holds
    it there — the loop's own attempt limit is what eventually stops
    retrying, not this function)."""
    return min(current * _TRUNCATION_ESCALATION_FACTOR, _TRUNCATION_MAX_TOKENS_CAP)


def _sleep(seconds: float) -> None:  # pragma: no cover - trivial, monkeypatched in tests
    time.sleep(seconds)


def _is_retryable(exc: BaseException) -> bool:
    """Duck-typed on purpose: real `openai.APIConnectionError` (covers
    `APITimeoutError`, its subclass) is always retryable; anything —
    real SDK exception or a test double — carrying a `.status_code` or
    `.code` (google.genai.errors.APIError's attribute) integer in
    `_RETRYABLE_STATUS_CODES` or >= 500 is retryable. Everything else
    (4xx auth/bad-request/schema-shape errors) is fatal."""
    try:
        import openai

        if isinstance(exc, openai.APIConnectionError):
            return True
    except ImportError:  # pragma: no cover - openai is a hard dependency
        pass
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(exc, "code", None)
    if isinstance(status, int):
        return status in _RETRYABLE_STATUS_CODES or status >= 500
    return False


def _call_with_retry(fn: Any, *, provider: str, max_retries: int) -> tuple[Any, Any, int]:
    """Call `fn()` (a zero-arg callable performing one network round trip,
    returning `(payload, usage)`), retrying retryable failures with bounded
    exponential backoff + jitter. Returns `(payload, usage, attempts)`.
    Raises `MissingAPIKeyError` immediately, unwrapped (it is specific and
    actionable on its own). Raises `ProviderError` — naming what to do next
    — once retries are exhausted or the error is fatal."""
    attempt = 0
    delay = _RETRY_BASE_DELAY_S
    while True:
        attempt += 1
        try:
            payload, usage = fn()
            return payload, usage, attempt
        except MissingAPIKeyError:
            raise
        except Exception as exc:
            retryable = _is_retryable(exc)
            if not retryable or attempt > max_retries:
                reason = "retries exhausted" if retryable else "not retryable"
                next_step = (
                    "back off, lower max_concurrency, or try again later"
                    if retryable
                    else "inspect the error above (likely a bad request, an "
                    "incompatible schema, or an auth problem) — retrying will "
                    "not help until it is fixed"
                )
                raise ProviderError(
                    f"{provider} call failed on attempt {attempt} ({reason}): "
                    f"{exc}. Next step: {next_step}."
                ) from exc
            _sleep(min(delay, _RETRY_MAX_DELAY_S) + random.uniform(0, delay * 0.25))
            delay = min(delay * 2, _RETRY_MAX_DELAY_S)


# ---------------------------------------------------------------------------
# Provider clients — lazy singletons, so key resolution (and therefore
# MissingAPIKeyError) only happens the first time a real call is actually
# needed, never for dry_run or cache-hit-only usage.
# ---------------------------------------------------------------------------

_openai_client: Any = None
_google_client: Any = None
_client_singleton_lock = threading.Lock()


def _get_openai_client() -> Any:
    global _openai_client
    if _openai_client is None:
        with _client_singleton_lock:
            if _openai_client is None:
                import openai

                _openai_client = openai.OpenAI(api_key=resolve_openai_key())
    return _openai_client


def _get_google_client() -> Any:
    global _google_client
    if _google_client is None:
        with _client_singleton_lock:
            if _google_client is None:
                from google import genai

                _google_client = genai.Client(api_key=resolve_google_key())
    return _google_client


def reset_clients() -> None:
    """Drop both cached client singletons — a test/CLI convenience so the
    next real call re-resolves keys and rebuilds a fresh client (also lets
    tests monkeypatch `_get_openai_client`/`_get_google_client` cleanly
    without a stale singleton from an earlier test winning)."""
    global _openai_client, _google_client
    _openai_client = None
    _google_client = None


class _Usage(NamedTuple):
    prompt_tokens: int
    completion_tokens: int
    truncated: bool = False
    """True when the provider itself reported this completion was cut off
    by max_tokens (OpenAI ``finish_reason == "length"``; Google
    ``candidates[0].finish_reason`` containing ``MAX_TOKENS``). Defaults to
    False so every existing construction site (``embed()``'s usage, cache
    replay, dry_run) that never mentions truncation stays exactly as
    truthful as it always was — this field is additive, not a behavior
    change for call sites that don't set it."""


# ---------------------------------------------------------------------------
# complete()
# ---------------------------------------------------------------------------

DEFAULT_MAX_TOKENS = 1024
"""Default ``max_tokens`` for ``complete()`` calls that pass no ``schema``
(free-form text, e.g. a judge verdict + one-line reason). Unchanged by this
fix — free-form completions are typically short, and every current schema'd
call site already overrides this explicitly. See ``DEFAULT_SCHEMA_MAX_TOKENS``
below for the schema'd-call default, which this fix DOES raise."""

DEFAULT_SCHEMA_MAX_TOKENS = 8192
"""Default ``max_tokens`` for ``complete()`` calls that DO pass a ``schema``,
used only when the caller doesn't pass ``max_tokens`` explicitly (every
current in-repo call site does pass it explicitly — e.g.
``pipeline/extract.py``'s own ``_MAX_EXTRACT_TOKENS = 4096`` — so this new
default protects *future* schema'd callers, not a regression fix for an
existing one).

Was silently ``DEFAULT_MAX_TOKENS`` (1024) before this fix — far below what
structured extraction over a real clinical dialogue turn needs. The
measured defect this raise responds to: high-turn (60-85 turn) admissions
reproducibly truncated a 4096-token structured-extraction response near
12.4-12.8k output characters, losing that admission's entire fact set
(measured admission failure rate 3/29 and 3/27, ~10-11%). 8192 is a better
starting point, not a promise that nothing ever truncates again — pair it
with the truncation-specific escalating retry below
(``_next_truncation_max_tokens``), which is the actual fix for the reported
failure rate; this default just means fewer calls need to escalate at all."""


@dataclass
class LLMResponse:
    text: str
    parsed: Any | None
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_ms: float
    cached: bool
    attempts: int
    truncated: bool = False
    """True when at least one attempt behind this response was detected as
    truncated by max_tokens (finish_reason ``length``/``MAX_TOKENS``, or the
    JSON-parse-failure-plus-token-ceiling heuristic) before a later,
    escalated-max_tokens retry produced this response. A distinct,
    countable outcome — callers can report ``sum(r.truncated for r in
    results)`` instead of that recovery being silently invisible. Defaults
    to False so every pre-existing construction site (dry_run, and cache
    hits from before this field existed) stays truthful without special-
    casing."""


def _openai_complete_once(
    *,
    model: str,
    system: str | None,
    prompt: str,
    schema: dict | None,
    max_tokens: int,
    temperature: float,
) -> tuple[str, _Usage]:
    client = _get_openai_client()
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_tokens,
        "temperature": temperature,
    }
    if schema is not None:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "response",
                "schema": _to_openai_schema(schema),
                "strict": True,
            },
        }
    resp = client.chat.completions.create(**kwargs)
    choice = resp.choices[0]
    text = choice.message.content or ""
    truncated = getattr(choice, "finish_reason", None) == "length"
    usage = _Usage(
        prompt_tokens=resp.usage.prompt_tokens if resp.usage else 0,
        completion_tokens=resp.usage.completion_tokens if resp.usage else 0,
        truncated=truncated,
    )
    return text, usage


def _to_openai_schema(schema: dict) -> dict:
    """Normalize a plain JSON Schema for OpenAI strict structured output.

    The two providers want *opposite* things from the same schema, which is a
    live footgun for callers:

    - OpenAI strict mode **requires** ``additionalProperties: false`` on every
      object, and requires ``required`` to list **every** property. Omitting
      either is a hard 400, not a soft degradation.
    - Google's API **rejects** ``additionalProperties`` outright (also a 400),
      which is why ``_to_google_schema`` strips it.

    Normalizing here rather than at each call site means ``extract.py`` /
    ``judge.py`` / ``reader.py`` write one plain schema and it works on both
    providers. Hand-adding these flags at every nesting level is the kind of
    thing that gets forgotten in one branch and fails mid-bulk-run.

    Pure and recursive; does not mutate the caller's dict.
    """
    if not isinstance(schema, dict):
        return schema

    out: dict[str, Any] = {}
    for key, value in schema.items():
        if isinstance(value, dict):
            out[key] = (
                {k: _to_openai_schema(v) for k, v in value.items()}
                if key in ("properties", "$defs", "definitions")
                else _to_openai_schema(value)
            )
        elif isinstance(value, list):
            out[key] = [_to_openai_schema(v) for v in value]
        else:
            out[key] = value

    if out.get("type") == "object":
        out["additionalProperties"] = False
        # Strict mode requires every declared property to appear in `required`.
        props = out.get("properties")
        if isinstance(props, dict) and props:
            out["required"] = list(props.keys())
    return out


def _to_google_schema(schema: dict) -> Any:
    """Convert a plain JSON Schema dict to a Gemini-public-API-compatible
    `google.genai.types.Schema`.

    Load-bearing fix, found live (not from memory) while running this
    story's own required smoke call: passing a JSON Schema dict with
    `additionalProperties` straight through as `response_schema` gets a
    real 400 from the Gemini public API — `Unknown name
    "additional_properties" at 'generation_config.response_schema': Cannot
    find field` — because the public `GEMINI_API` surface supports only a
    restricted OpenAPI-3.0-like subset of `Schema` (unlike `VERTEX_AI`,
    which does accept it; confirmed by inspecting
    `types.Schema.from_json_schema`'s own `api_option` parameter). The SDK
    ships exactly the right conversion for this:
    `types.Schema.from_json_schema(api_option="GEMINI_API")` silently drops
    fields the public API doesn't support (verified live: the same schema
    that 400'd as a raw dict round-trips to a working call once converted
    this way) rather than erroring, which is the right default here — a
    caller's schema is allowed to carry `additionalProperties` (OpenAI's
    strict mode requires it) and just have that one field ignored on the
    Google path, rather than fail the call.
    """
    from google.genai import types

    json_schema = types.JSONSchema.model_validate(schema)
    return types.Schema.from_json_schema(
        json_schema=json_schema, api_option="GEMINI_API", raise_error_on_unsupported_field=False,
    )


def _google_complete_once(
    *,
    model: str,
    system: str | None,
    prompt: str,
    schema: dict | None,
    max_tokens: int,
    temperature: float,
) -> tuple[str, _Usage]:
    from google.genai import types

    client = _get_google_client()
    config_kwargs: dict[str, Any] = {
        "max_output_tokens": max_tokens,
        "temperature": temperature,
    }
    if system:
        config_kwargs["system_instruction"] = system
    if schema is not None:
        config_kwargs["response_mime_type"] = "application/json"
        config_kwargs["response_schema"] = _to_google_schema(schema)
    config = types.GenerateContentConfig(**config_kwargs)
    resp = client.models.generate_content(model=model, contents=prompt, config=config)
    text = resp.text or ""
    usage_meta = resp.usage_metadata
    truncated = False
    candidates = getattr(resp, "candidates", None) or []
    if candidates:
        finish_reason = getattr(candidates[0], "finish_reason", None)
        if finish_reason is not None:
            # Real SDK: a `types.FinishReason` enum member (`.name ==
            # "MAX_TOKENS"`). Test doubles / a future SDK version: a plain
            # string or int — `str(...)` handles all three uniformly.
            reason_str = getattr(finish_reason, "name", None) or str(finish_reason)
            truncated = "MAX_TOKENS" in reason_str.upper()
    usage = _Usage(
        prompt_tokens=(usage_meta.prompt_token_count or 0) if usage_meta else 0,
        completion_tokens=(usage_meta.candidates_token_count or 0) if usage_meta else 0,
        truncated=truncated,
    )
    return text, usage



# ---------------------------------------------------------------------------
# Local provider — HuggingFace weights on this machine's own GPU/CPU.
# ---------------------------------------------------------------------------

LOCAL_MODEL_PREFIX = "local:"
"""Model-id prefix that routes to on-disk weights: `local:Qwen/Qwen2.5-7B-Instruct`.

Everything after the prefix is handed verbatim to `transformers`
`from_pretrained`, so it accepts a Hub id OR a local directory."""

LOCAL_DEVICE_ENV = "MEDMEMGRAPH_LOCAL_DEVICE"
LOCAL_DTYPE_ENV = "MEDMEMGRAPH_LOCAL_DTYPE"
LOCAL_MAX_INPUT_TOKENS_ENV = "MEDMEMGRAPH_LOCAL_MAX_INPUT_TOKENS"
LOCAL_QUANT_ENV = "MEDMEMGRAPH_LOCAL_QUANT"
"""`8bit` / `4bit` / unset. Quantized loading needs `bitsandbytes` and trades a
little quality for a lot of VRAM: Qwen2.5-7B is ~15.2 GB in fp16 (does not fit a
16 GB card beside an embedder) and ~7.6 GB in 8-bit (fits with room)."""

LOCAL_SKIP_PREFLIGHT_ENV = "MEDMEMGRAPH_LOCAL_SKIP_PREFLIGHT"
"""Escape hatch for the memory pre-flight below. Set to `1` only if you know the
estimate is wrong for your setup — the check exists because the failure mode it
prevents is not an exception."""

_HOST_HEADROOM_GB = 2.0
"""Spare host RAM to leave unclaimed. `from_pretrained` peaks above the largest
shard (tokenizer, safetensors mmap, dtype conversion), and a *global* OOM does
not politely kill just this process."""

_VRAM_HEADROOM_GB = 1.5
"""Spare VRAM for the embedder (~1.2 GB), any reranker, and the KV cache."""


def _dir_size_gb(path: "pathlib.Path") -> float:
    """On-disk size in GB, counting each physical file ONCE.

    Deduplicated by `(st_dev, st_ino)` because a HuggingFace cache stores each
    weight once under `blobs/` and symlinks it into `snapshots/`; a naive
    `rglob` sum counts both and reports exactly double. That bug shipped in this
    guard's first revision and made it refuse a 7B/8-bit load needing ~7.1 GB by
    sizing it at 14.2 GB — and it only appeared once the model was cached, so
    the first (uncached) run passed and the second did not."""
    total = 0
    seen: set[tuple[int, int]] = set()
    for f in path.rglob("*"):
        try:
            st = f.stat()
        except OSError:
            continue
        if not f.is_file():
            continue
        key = (st.st_dev, st.st_ino)
        if key in seen:
            continue
        seen.add(key)
        total += st.st_size
    return total / 1024**3


def _preflight_local_load(hf_id: str, device: str, quant: str | None) -> None:
    """Refuse a local model load that the machine cannot survive.

    **Why this is a hard gate and not a warning.** On 2026-08-18 a
    `Qwen2.5-3B-Instruct` fp16 load on a 15.9 GB WSL2 box triggered a kernel
    *global* OOM (`anon-rss 8.3 GB`). The kernel did not kill just the loader —
    the VM went down, which took the HydraDB container with it, and because that
    container runs `CLOUD_PROVIDER=memory` the entire ingested graph (13,406
    facts, 30,086 turns, 4h of work) was gone. VS Code was holding 8 GB at the
    time, leaving 6.5 GB against a load that peaked at 8.3 GB.

    An exception costs a run. A global OOM costs the machine. Anything that can
    take down the host has to refuse first and explain, so this raises
    `LocalModelError` naming the numbers before a single weight is read.

    Estimates come from the real on-disk size of the HF snapshot when it is
    already cached, so the common path is measured rather than guessed. Set
    `$MEDMEMGRAPH_LOCAL_SKIP_PREFLIGHT=1` to override.
    """
    if os.environ.get(LOCAL_SKIP_PREFLIGHT_ENV) == "1":
        return

    import pathlib as _pathlib

    import torch

    # Weight size: prefer the cached snapshot on disk; fall back to a
    # parameter-count guess parsed from the model id ("...-7B-..." -> 7).
    weights_gb: float | None = None
    try:
        from huggingface_hub import snapshot_download

        snapshot_dir = _pathlib.Path(
            snapshot_download(hf_id, local_files_only=True, allow_patterns=["*.json"])
        )
        # Size the SNAPSHOT dir, not its grandparent. The grandparent is the
        # model repo root holding both `blobs/` and `snapshots/`, and the latter
        # symlinks into the former — sizing it reports double.
        weights_gb = _dir_size_gb(snapshot_dir)
    except Exception:  # noqa: BLE001 — not cached yet, or a local path
        candidate = _pathlib.Path(hf_id)
        if candidate.is_dir():
            weights_gb = _dir_size_gb(candidate)
    if weights_gb is None or weights_gb <= 0:
        match = re.search(r"(\d+(?:\.\d+)?)\s*[bB]\b", hf_id)
        params_b = float(match.group(1)) if match else 7.0
        weights_gb = params_b * 2.0  # fp16: 2 bytes/param

    if quant == "8bit":
        weights_gb *= 0.5
    elif quant == "4bit":
        weights_gb *= 0.28

    # Host RAM: with device_map streaming, peak is roughly the largest shard.
    # Shards are conventionally <=5 GB, so bound by min(weights, 5) + headroom.
    peak_host_gb = min(weights_gb, 5.0) + _HOST_HEADROOM_GB
    available_gb = _available_ram_gb()
    if available_gb is not None and peak_host_gb > available_gb:
        raise LocalModelError(
            f"Refusing to load {hf_id!r}: needs ~{peak_host_gb:.1f} GB host RAM "
            f"(largest shard + {_HOST_HEADROOM_GB:.1f} GB headroom) but only "
            f"{available_gb:.1f} GB is available. A load this size triggered a "
            f"kernel global OOM on this machine before, which rebooted the VM and "
            f"destroyed the in-memory graph. Free host RAM (closing extra VS Code "
            f"windows recovered ~5 GB), pick a smaller model, set "
            f"${LOCAL_QUANT_ENV}=8bit, or override with "
            f"${LOCAL_SKIP_PREFLIGHT_ENV}=1."
        )

    if device.startswith("cuda") and torch.cuda.is_available():
        free_bytes, _total = torch.cuda.mem_get_info(0)
        free_vram_gb = free_bytes / 1024**3
        needed_vram_gb = weights_gb + _VRAM_HEADROOM_GB
        if needed_vram_gb > free_vram_gb:
            hint = (
                f" Try ${LOCAL_QUANT_ENV}=8bit (roughly halves it)."
                if quant is None
                else f" Try ${LOCAL_QUANT_ENV}=4bit."
            )
            raise LocalModelError(
                f"Refusing to load {hf_id!r} on {device}: needs ~{needed_vram_gb:.1f} GB "
                f"VRAM (weights ~{weights_gb:.1f} GB + {_VRAM_HEADROOM_GB:.1f} GB for the "
                f"embedder/reranker/KV cache) but only {free_vram_gb:.1f} GB is free.{hint}"
            )


def _available_ram_gb() -> float | None:
    """Host RAM currently available, from `/proc/meminfo`. `MemAvailable` (not
    `MemFree`) is the kernel's own estimate of what a new allocation can claim
    without swapping. Returns None off Linux."""
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024**2
    except OSError:
        return None
    return None


_DEFAULT_LOCAL_MAX_INPUT_TOKENS = 8192
"""Prompt-side truncation guard. A local model has a hard context window and,
unlike an API, no server-side error telling you that you exceeded it — it will
happily produce garbage instead. Truncation here is reported on the response
(`truncated=True`) rather than applied silently."""

_local_models: dict[str, Any] = {}


def _get_local_model(hf_id: str):
    """Load (once per process) and cache a local causal LM + tokenizer.

    Kept in a module-level dict rather than an lru_cache so a caller can drop a
    model to reclaim VRAM: `llm._local_models.clear()`."""
    cached = _local_models.get(hf_id)
    if cached is not None:
        return cached

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = os.environ.get(LOCAL_DEVICE_ENV) or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype_name = os.environ.get(LOCAL_DTYPE_ENV) or ("float16" if device == "cuda" else "float32")
    quant = (os.environ.get(LOCAL_QUANT_ENV) or "").strip().lower() or None
    if quant not in (None, "8bit", "4bit"):
        raise LocalModelError(
            f"${LOCAL_QUANT_ENV}={quant!r} is not recognized; use '8bit', '4bit', or leave it unset."
        )

    # Refuse before reading a single weight — see _preflight_local_load for why
    # this is a hard gate rather than a warning.
    _preflight_local_load(hf_id, device, quant)

    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    quant_config = None
    if quant is not None:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:  # pragma: no cover - transformers is a hard dep
            raise LocalModelError(f"quantization requested but unavailable: {exc}") from exc
        if quant == "8bit":
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, dtype_name),
                bnb_4bit_quant_type="nf4",
            )
    # `device_map=` (accelerate), NOT `.from_pretrained(...).to(device)`.
    #
    # The `.to(device)` form materializes the ENTIRE model in system RAM first
    # and only then copies it to the GPU, so peak host memory is the full model
    # size no matter how much VRAM is free. That killed a 3B fp16 load on this
    # 15 GB WSL2 box (kernel OOM at 8.3 GB RSS) while the GPU sat with 15.8 GB
    # free — the load never got far enough to use it. `device_map` streams the
    # checkpoint shard by shard straight onto the target device, so peak host
    # memory is one shard, not one model.
    #
    # An earlier revision of this function avoided `accelerate` deliberately, to
    # keep the dependency list unchanged. That was the wrong trade: it bought a
    # shorter dependency list and paid with a loader that cannot load anything
    # bigger than free system RAM, which on this machine is smaller than the GPU.
    load_kwargs: dict[str, Any] = {
        "device_map": device,
        "low_cpu_mem_usage": True,
    }
    if quant_config is not None:
        # bitsandbytes owns the on-device dtype; passing `dtype=` alongside it
        # is ignored at best and conflicting at worst.
        load_kwargs["quantization_config"] = quant_config
    else:
        load_kwargs["dtype"] = getattr(torch, dtype_name)
    model = AutoModelForCausalLM.from_pretrained(hf_id, **load_kwargs)
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    _local_models[hf_id] = (tokenizer, model, device)
    return _local_models[hf_id]


def _local_complete_once(
    *,
    model: str,
    system: str | None,
    prompt: str,
    schema: dict | None,
    max_tokens: int,
    temperature: float,
) -> tuple[str, "_Usage"]:
    """One generation against local weights. Same contract as the OpenAI and
    Google `_*_complete_once` functions, so the retry/schema/cache/ledger
    machinery above is untouched.

    **Structured output is prompted, not constrained.** OpenAI and Google both
    offer native schema-constrained decoding; `transformers` alone does not, and
    adding a constrained-decoding dependency (outlines/guidance) is a bigger
    change than this seam. So when `schema` is given, the schema is appended to
    the system prompt as an instruction and the model's JSON is parsed by the
    SAME `_complete_with_schema_retry` loop that already handles a provider
    returning malformed JSON.

    That is a real quality difference, stated rather than hidden: this project's
    own `literature/17` notes that prompted JSON mode can cost significant
    accuracy versus constrained decoding. A local model is therefore the right
    choice for cost-free bulk generation and the wrong choice for the
    schema-heavy extraction path unless you have measured it on your own data.
    """
    import torch

    hf_id = model[len(LOCAL_MODEL_PREFIX):]
    tokenizer, lm, device = _get_local_model(hf_id)

    sys_text = system or ""
    if schema is not None:
        sys_text = (
            f"{sys_text}\n\nRespond with ONLY a single valid JSON object matching "
            f"this schema. No prose, no markdown fence, no trailing commentary. "
            f"Start your reply with '{{' and end it with '}}'.\n"
            f"Schema: {json.dumps(schema)}\n"
            f"Skeleton to fill in (same keys, same nesting):\n{_schema_skeleton(schema)}"
        ).strip()

    messages = ([{"role": "system", "content": sys_text}] if sys_text else []) + [
        {"role": "user", "content": prompt}
    ]
    try:
        text_in = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:  # noqa: BLE001 — model without a chat template
        text_in = (f"{sys_text}\n\n" if sys_text else "") + prompt

    max_input = int(os.environ.get(LOCAL_MAX_INPUT_TOKENS_ENV) or _DEFAULT_LOCAL_MAX_INPUT_TOKENS)
    enc = tokenizer(text_in, return_tensors="pt", truncation=True, max_length=max_input)
    prompt_tokens = int(enc["input_ids"].shape[-1])
    input_truncated = prompt_tokens >= max_input
    enc = {k: v.to(lm.device) for k, v in enc.items()}

    with torch.no_grad():
        out = lm.generate(
            **enc,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated = out[0][prompt_tokens:]
    completion_tokens = int(generated.shape[-1])
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()

    if schema is not None:
        text = _extract_json_object(text)

    return text, _Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        # Either side hitting its ceiling is a real truncation the caller must
        # be able to see; the schema-retry loop uses this to widen max_tokens.
        truncated=input_truncated or completion_tokens >= max_tokens,
    )



def _schema_skeleton(schema: dict) -> str:
    """A minimal filled-in example of `schema`, for the prompted-structured-output
    path.

    A schema alone underspecifies the target for a small local model: given only
    the JSON Schema, a 7B emits valid JSON most of the time but not every time,
    and an intermittent failure is worse than a consistent one because it shows
    up as a scattering of wrong answers rather than an obvious bug. Showing the
    literal shape to fill in is the cheapest reliability win available without
    adding a constrained-decoding dependency."""

    def _example(node: dict):
        t = node.get("type")
        if t == "object":
            return {k: _example(v) for k, v in (node.get("properties") or {}).items()}
        if t == "array":
            return [_example(node.get("items") or {"type": "string"})]
        if t == "boolean":
            return True
        if t == "integer":
            return 0
        if t == "number":
            return 0.0
        return "..."

    return json.dumps(_example(schema), indent=None)


def _extract_json_object(text: str) -> str:
    """Pull the outermost balanced JSON object out of a model completion.

    Prompted structured output is best-effort: a local model that is told "reply
    with only JSON" mostly complies, but sometimes wraps the object in a
    markdown fence, prefixes it with "Here is the JSON:", or appends a sentence
    of commentary. Each of those makes `json.loads` fail on output that is
    otherwise perfectly good, and the caller then burns its retry budget and
    finally reports the item as a wrong answer — a measurement error dressed up
    as a model error.

    Brace-counting rather than a regex, and string-aware so a `{` or `}` inside a
    quoted note (clinical text contains both) does not throw off the depth
    count."""
    if not text:
        return text
    fenced = text.strip()
    if fenced.startswith("```"):
        fenced = fenced.split("\n", 1)[-1] if "\n" in fenced else fenced
        fenced = fenced.rsplit("```", 1)[0]
    start = fenced.find("{")
    if start == -1:
        return fenced.strip()

    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(fenced)):
        ch = fenced[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return fenced[start : i + 1]
    # Unbalanced (usually a genuine truncation) — hand back what we have so the
    # retry loop sees a real JSONDecodeError and can widen max_tokens.
    return fenced[start:].strip()


def _record_failed_schema_attempt(ledger: Ledger, model: str, usage: _Usage) -> None:
    """Every schema-retry attempt that reaches this point produced a REAL
    completion — real tokens the provider already billed — that is being
    discarded (truncated, or valid JSON that failed schema validation) in
    favor of a retry. Unlike a transient 429/500 (handled entirely inside
    `_call_with_retry`, which never returns a usable `usage` because no
    completion was generated), this usage is real and, before this fix, was
    simply overwritten by the next loop iteration and never recorded
    anywhere — the exact gap this function closes."""
    cost = _cost_usd(model, usage.prompt_tokens, usage.completion_tokens)
    ledger.record_failed_attempt(model, usage.prompt_tokens, usage.completion_tokens, cost)


def _complete_with_schema_retry(
    *,
    provider: str,
    model: str,
    system: str | None,
    prompt: str,
    schema: dict | None,
    max_tokens: int,
    temperature: float,
    max_retries: int,
    ledger: Ledger,
) -> tuple[str, Any, _Usage, int, bool]:
    """Returns `(text, parsed, usage, total_attempts, truncated_recovery)`.
    `truncated_recovery` is True iff at least one attempt along the way was
    detected as truncated before this call ultimately succeeded — the value
    `complete()` surfaces on `LLMResponse.truncated` (see that field's
    docstring). On exhaustion, raises `TruncationError` (a
    `SchemaValidationError` subtype) if the LAST attempt was a truncation,
    else the plain `SchemaValidationError` as before.
    """
    total_attempts = 0
    last_violation = "(no attempt recorded)"
    last_was_truncated = False
    any_truncated = False
    usage = _Usage(0, 0)
    current_max_tokens = max_tokens
    for _ in range(_MAX_SCHEMA_RETRIES + 1):

        def _once() -> tuple[str, _Usage]:
            if provider == "openai":
                return _openai_complete_once(
                    model=model, system=system, prompt=prompt, schema=schema,
                    max_tokens=current_max_tokens, temperature=temperature,
                )
            if provider == "local":
                return _local_complete_once(
                    model=model, system=system, prompt=prompt, schema=schema,
                    max_tokens=current_max_tokens, temperature=temperature,
                )
            return _google_complete_once(
                model=model, system=system, prompt=prompt, schema=schema,
                max_tokens=current_max_tokens, temperature=temperature,
            )

        text, usage, attempts = _call_with_retry(_once, provider=provider, max_retries=max_retries)
        total_attempts += attempts
        if schema is None:
            return text, None, usage, total_attempts, any_truncated

        truncated_this_attempt = usage.truncated
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            if not truncated_this_attempt and usage.completion_tokens >= current_max_tokens:
                # Fallback heuristic (story-specified): the provider didn't
                # (or its SDK version doesn't) expose a finish_reason we
                # recognized, but the completion consumed every token we
                # asked for and the JSON never closed — same signature as an
                # explicit truncation, detected a different way.
                truncated_this_attempt = True
            last_was_truncated = truncated_this_attempt
            last_violation = f"response was not valid JSON: {exc}"
            if truncated_this_attempt:
                last_violation += (
                    f" — looks like max_tokens={current_max_tokens} truncation "
                    "(finish_reason or token-ceiling signal), not a schema defect"
                )
            _record_failed_schema_attempt(ledger, model, usage)
            if truncated_this_attempt:
                any_truncated = True
                current_max_tokens = _next_truncation_max_tokens(current_max_tokens)
            continue

        violations = _validate_against_schema(parsed, schema)
        if not violations:
            return text, parsed, usage, total_attempts, any_truncated
        # Valid, complete JSON that just doesn't match the schema is a
        # genuine schema violation, NOT truncation — unless the provider's
        # own finish_reason explicitly disagrees (rare: cut off exactly at a
        # coincidentally-valid-but-wrong-shape JSON boundary).
        last_was_truncated = truncated_this_attempt
        last_violation = "; ".join(violations)
        _record_failed_schema_attempt(ledger, model, usage)
        if truncated_this_attempt:
            any_truncated = True
            current_max_tokens = _next_truncation_max_tokens(current_max_tokens)

    if last_was_truncated:
        raise TruncationError(
            f"{model} response was still truncated by max_tokens after "
            f"{total_attempts} attempt(s) across {_MAX_SCHEMA_RETRIES + 1} schema "
            f"retries, even after escalating max_tokens up to {current_max_tokens} "
            f"(cap {_TRUNCATION_MAX_TOKENS_CAP}): {last_violation}. The completion "
            "is being cut off before the JSON closes, not rejected for its shape — "
            "raise max_tokens further at the call site, or reduce how much you're "
            "asking the model to extract in one call (e.g. split a high-turn "
            "admission)."
        )
    raise SchemaValidationError(
        f"{model} response did not satisfy the given schema after "
        f"{total_attempts} attempt(s) across {_MAX_SCHEMA_RETRIES + 1} schema "
        f"retries: {last_violation}. Check the schema is compatible with "
        f"{provider}'s structured-output constraints (OpenAI strict mode "
        "requires additionalProperties: false and every property required), "
        "or relax it."
    )


def _dry_run_complete(
    *, model: str, system: str | None, prompt: str, schema: dict | None
) -> LLMResponse:
    start = time.monotonic()
    if schema is not None:
        parsed = _stub_value_for_schema(schema)
        text = json.dumps(parsed)
    else:
        parsed = None
        text = f"[dry_run stub response for model={model} — no network call made]"
    prompt_tokens = _estimate_tokens_offline((system or "") + prompt)
    completion_tokens = _estimate_tokens_offline(text)
    latency_ms = (time.monotonic() - start) * 1000.0
    return LLMResponse(
        text=text, parsed=parsed, model=model, prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens, cost_usd=0.0, latency_ms=latency_ms,
        cached=False, attempts=1,
    )


def complete(
    prompt: str,
    *,
    model: str | None = None,
    schema: dict | None = None,
    system: str | None = None,
    max_tokens: int | None = None,
    temperature: float = 0.0,
    dry_run: bool = False,
    use_cache: bool = True,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> LLMResponse:
    """The single entry point every real inference call in this project
    should eventually route through (see module docstring — rewiring
    `extract.py` / `judge.py` / `reader.py` to actually call this is a
    later story, not this one).

    `model=None` defaults to `ANSWER_MODEL` (the most general-purpose of
    the four named defaults) — a stated assumption: the story specifies the
    four per-role model constants but not what a caller-agnostic
    `complete()` call defaults to when it isn't told which role it's
    serving. Every real caller (extract/judge/reader, once wired) is
    expected to pass `model=` explicitly.

    `max_tokens=None` (the new default — was a fixed `DEFAULT_MAX_TOKENS`
    before this fix) resolves to `DEFAULT_SCHEMA_MAX_TOKENS` when `schema`
    is given, else `DEFAULT_MAX_TOKENS` — see both constants' docstrings.
    Every current schema'd call site in this repo already passes
    `max_tokens=` explicitly, so this only changes behavior for a caller
    that doesn't; it does not change what happens on truncation regardless
    of which value is in play, or what value a caller passes explicitly —
    see `_complete_with_schema_retry`'s escalating retry for that.
    """
    model = model or ANSWER_MODEL
    if max_tokens is None:
        max_tokens = DEFAULT_SCHEMA_MAX_TOKENS if schema is not None else DEFAULT_MAX_TOKENS
    temperature = float(temperature)
    start = time.monotonic()

    if dry_run:
        return _dry_run_complete(model=model, system=system, prompt=prompt, schema=schema)

    key = _cache_key(
        kind="complete", model=model, system=system, prompt_or_text=prompt,
        schema=schema, temperature=temperature,
    )
    if use_cache:
        hit = _cache_get(key)
        if hit is not None:
            get_ledger().record(
                model, hit.get("prompt_tokens", 0), hit.get("completion_tokens", 0),
                0.0, cached=True,
            )
            return LLMResponse(
                text=hit["text"], parsed=hit.get("parsed"), model=model,
                prompt_tokens=hit.get("prompt_tokens", 0),
                completion_tokens=hit.get("completion_tokens", 0),
                cost_usd=0.0, latency_ms=(time.monotonic() - start) * 1000.0,
                cached=True, attempts=0,
                # `.get(..., False)`: an entry written by a pre-fix cache
                # (no "truncated" key at all) still hits and still returns a
                # valid LLMResponse — the cache-key computation above is
                # completely unchanged by this fix, only what's stored under
                # a hit gained one more (backward-compatible) field.
                truncated=hit.get("truncated", False),
            )

    provider = _provider_for_model(model)
    est_prompt_tokens = _estimate_tokens((system or "") + prompt)
    estimate = _cost_usd(model, est_prompt_tokens, max_tokens)
    ledger = get_ledger()
    ledger.check_budget(estimate, _max_usd())
    try:
        text, parsed, usage, attempts, truncated = _complete_with_schema_retry(
            provider=provider, model=model, system=system, prompt=prompt,
            schema=schema, max_tokens=max_tokens, temperature=temperature,
            max_retries=max_retries, ledger=ledger,
        )
    finally:
        ledger.release_reservation(estimate)

    cost = _cost_usd(model, usage.prompt_tokens, usage.completion_tokens)
    ledger.record(model, usage.prompt_tokens, usage.completion_tokens, cost, cached=False)

    if use_cache:
        _cache_put(key, {
            "text": text, "parsed": parsed,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "truncated": truncated,
        })

    return LLMResponse(
        text=text, parsed=parsed, model=model,
        prompt_tokens=usage.prompt_tokens, completion_tokens=usage.completion_tokens,
        cost_usd=cost, latency_ms=(time.monotonic() - start) * 1000.0,
        cached=False, attempts=attempts, truncated=truncated,
    )


def complete_many(
    prompts: list[str],
    *,
    max_concurrency: int = 4,
    model: str | None = None,
    schema: dict | None = None,
    system: str | None = None,
    max_tokens: int | None = None,
    temperature: float = 0.0,
    dry_run: bool = False,
    use_cache: bool = True,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> list[LLMResponse]:
    """`complete()` over a batch of prompts, bounded by `max_concurrency`.
    Each call independently reserves-then-commits against the same process
    Ledger (see `Ledger.check_budget`), so a burst of concurrent calls
    cannot collectively overshoot `MEDMEMGRAPH_MAX_USD`. Returns results in
    input order (not completion order). If any call fails, every already
    -submitted call is still allowed to finish (already-paid-for requests
    aren't wasted by early cancellation) and the first failure is then
    raised.

    `max_tokens=None` forwards to `complete()` unchanged, which resolves it
    per-call the same schema-aware way (see `complete()`'s own docstring) —
    every caller of `complete_many` benefits from the same truncation fix
    without needing to know about it."""
    if not prompts:
        return []
    workers = max(1, min(max_concurrency, len(prompts)))
    results: list[LLMResponse | None] = [None] * len(prompts)

    def _work(i: int, p: str) -> None:
        results[i] = complete(
            p, model=model, schema=schema, system=system, max_tokens=max_tokens,
            temperature=temperature, dry_run=dry_run, use_cache=use_cache,
            max_retries=max_retries,
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_work, i, p) for i, p in enumerate(prompts)]
        first_exc: BaseException | None = None
        for f in futures:
            try:
                f.result()
            except Exception as exc:  # noqa: BLE001 - deliberately broad, re-raised below
                if first_exc is None:
                    first_exc = exc
        if first_exc is not None:
            raise first_exc
    return results  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# embed()
# ---------------------------------------------------------------------------

_EMBED_DIMENSIONS = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072}
_DEFAULT_DRY_RUN_EMBED_DIM = 8


def _dry_run_vector(model: str, text: str) -> list[float]:
    """Deterministic, offline, schema-free stub embedding — same text under
    the same model always produces the same vector, no network call."""
    dim = _EMBED_DIMENSIONS.get(model, _DEFAULT_DRY_RUN_EMBED_DIM)
    seed = hashlib.sha256(f"{model}:{text}".encode("utf-8")).digest()
    values: list[float] = []
    counter = 0
    while len(values) < dim:
        chunk = hashlib.sha256(seed + counter.to_bytes(4, "big")).digest()
        for b in chunk:
            if len(values) >= dim:
                break
            values.append((b / 255.0) * 2.0 - 1.0)
        counter += 1
    return values


def _embed_once(provider: str, model: str, texts: list[str]) -> tuple[list[list[float]], _Usage]:
    def _once() -> tuple[list[list[float]], _Usage]:
        if provider == "openai":
            client = _get_openai_client()
            resp = client.embeddings.create(model=model, input=texts)
            vectors = [list(d.embedding) for d in resp.data]
            usage = _Usage(prompt_tokens=resp.usage.prompt_tokens if resp.usage else 0, completion_tokens=0)
            return vectors, usage
        client = _get_google_client()
        resp = client.models.embed_content(model=model, contents=texts)
        vectors = [list(e.values or []) for e in (resp.embeddings or [])]
        estimated = sum(_estimate_tokens(t) for t in texts)
        return vectors, _Usage(prompt_tokens=estimated, completion_tokens=0)

    vectors, usage, _attempts = _call_with_retry(_once, provider=provider, max_retries=DEFAULT_MAX_RETRIES)
    return vectors, usage


def embed(
    texts: list[str],
    *,
    model: str | None = None,
    dry_run: bool = False,
    use_cache: bool = True,
) -> list[list[float]]:
    """Batched embeddings, with the same disk cache and Ledger accounting
    as `complete()`. Cache is per-text (keyed like `complete()`'s, with
    `kind="embed"`), so overlapping batches across calls reuse hits; only
    the uncached subset of a batch triggers one real network call."""
    model = model or EMBED_MODEL
    if not texts:
        return []
    if dry_run:
        return [_dry_run_vector(model, t) for t in texts]

    provider = _provider_for_model(model)
    ledger = get_ledger()
    keys = [
        _cache_key(kind="embed", model=model, system=None, prompt_or_text=t, schema=None, temperature=None)
        for t in texts
    ]
    results: list[list[float] | None] = [None] * len(texts)

    if use_cache:
        for i, key in enumerate(keys):
            hit = _cache_get(key)
            if hit is not None:
                results[i] = hit["vector"]
                ledger.record(model, hit.get("tokens", 0), 0, 0.0, cached=True)

    miss_indices = [i for i, v in enumerate(results) if v is None]
    if miss_indices:
        miss_texts = [texts[i] for i in miss_indices]
        estimate = _cost_usd(model, sum(_estimate_tokens(t) for t in miss_texts), 0)
        ledger.check_budget(estimate, _max_usd())
        try:
            vectors, usage = _embed_once(provider, model, miss_texts)
        finally:
            ledger.release_reservation(estimate)
        cost = _cost_usd(model, usage.prompt_tokens, 0)
        ledger.record(model, usage.prompt_tokens, 0, cost, cached=False)
        for idx, text, vector in zip(miss_indices, miss_texts, vectors):
            results[idx] = vector
            if use_cache:
                _cache_put(keys[idx], {"vector": vector, "tokens": _estimate_tokens(text)})

    return results  # type: ignore[return-value]


__all__ = [
    "resolve_key_for_model",
    "LLMError",
    "MissingAPIKeyError",
    "LocalModelError",
    "BudgetExceeded",
    "SchemaValidationError",
    "TruncationError",
    "ProviderError",
    "resolve_openai_key",
    "resolve_google_key",
    "EXTRACT_MODEL",
    "ANSWER_MODEL",
    "JUDGE_MODEL",
    "EMBED_MODEL",
    "PRICING",
    "DEFAULT_MAX_USD",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_SCHEMA_MAX_TOKENS",
    "CACHE_DIR",
    "LLMResponse",
    "Ledger",
    "get_ledger",
    "reset_ledger",
    "reset_clients",
    "complete",
    "complete_many",
    "embed",
]
