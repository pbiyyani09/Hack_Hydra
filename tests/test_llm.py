"""tests/test_llm.py — offline tests for medmemgraph.llm.

Every test in this file is offline: no test may make a real network call.
Provider clients are always either (a) never reached (dry_run / cache-hit
paths, proven via a trip-wire monkeypatch that raises if the client getter
is ever invoked) or (b) replaced with an in-process fake object via
`monkeypatch.setattr(llm, "_get_openai_client", ...)` /
`monkeypatch.setattr(llm, "_get_google_client", ...)`. No test sets a real
`OPENAI_API_KEY` / `GOOGLE_API_KEY` anywhere, and the autouse
`isolate_llm_module` fixture points `CACHE_DIR` at a per-test `tmp_path` and
resets the module's mutable singletons, so tests never touch the real repo
`data/llm_cache/` or read the real `.env`.
"""

from __future__ import annotations

import json
import threading
import time

import pytest

from medmemgraph import llm


# ---------------------------------------------------------------------------
# Fakes — minimal objects matching exactly the attribute shapes llm.py reads
# off real openai / google-genai SDK response objects (verified against the
# installed SDKs — openai==3.1.0, google-genai==2.18.1 — before writing
# this module; see the dev-python return note for what was introspected).
# ---------------------------------------------------------------------------


class _FakeUsage:
    def __init__(self, prompt_tokens: int, completion_tokens: int = 0) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str, finish_reason: str = "stop") -> None:
        self.message = _FakeMessage(content)
        self.finish_reason = finish_reason


class _FakeChatCompletion:
    def __init__(
        self,
        content: str,
        prompt_tokens: int = 10,
        completion_tokens: int = 5,
        finish_reason: str = "stop",
    ) -> None:
        self.choices = [_FakeChoice(content, finish_reason=finish_reason)]
        self.usage = _FakeUsage(prompt_tokens, completion_tokens)


class Truncated:
    """Marks a scripted response (for either fake client below) as cut off
    by max_tokens — the OpenAI fake reports `finish_reason="length"`, the
    Google fake reports a `MAX_TOKENS` finish_reason, and both size
    `completion_tokens` to exactly the caller's requested `max_tokens` (the
    real-world signature: the provider used every token it was given).
    Lets a test script "this specific attempt was truncated" without
    constructing a full fake SDK response object by hand."""

    def __init__(self, text: str) -> None:
        self.text = text


class _FakeEmbeddingRow:
    def __init__(self, vector: list[float]) -> None:
        self.embedding = vector


class _FakeEmbeddingResponse:
    def __init__(self, vectors: list[list[float]], prompt_tokens: int = 3) -> None:
        self.data = [_FakeEmbeddingRow(v) for v in vectors]
        self.usage = _FakeUsage(prompt_tokens)


class FakeOpenAIChat:
    """Records every call; `responses` is a list of a str (success content),
    a `Truncated(text)` (success content, but `finish_reason="length"` and
    `completion_tokens` pinned to the requested `max_completion_tokens`), or
    an Exception instance (raised) — consumed in order; the last entry
    repeats once exhausted."""

    def __init__(self, responses: list) -> None:
        self.responses = responses
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        idx = min(len(self.calls) - 1, len(self.responses) - 1)
        item = self.responses[idx]
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, Truncated):
            return _FakeChatCompletion(
                item.text,
                completion_tokens=kwargs["max_completion_tokens"],
                finish_reason="length",
            )
        return _FakeChatCompletion(item)


class FakeOpenAIEmbeddings:
    def __init__(self, dim: int = 4) -> None:
        self.dim = dim
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        texts = kwargs["input"]
        vectors = [[float(len(t))] * self.dim for t in texts]
        return _FakeEmbeddingResponse(vectors, prompt_tokens=sum(len(t) for t in texts))


class FakeOpenAIClient:
    def __init__(self, responses: list | None = None, embed_dim: int = 4) -> None:
        self.chat = type("Chat", (), {})()
        self.chat.completions = FakeOpenAIChat(responses or ["ok"])
        self.embeddings = FakeOpenAIEmbeddings(dim=embed_dim)


class _FakeGoogleUsageMetadata:
    def __init__(self, prompt_token_count: int, candidates_token_count: int) -> None:
        self.prompt_token_count = prompt_token_count
        self.candidates_token_count = candidates_token_count


class _FakeGoogleCandidate:
    def __init__(self, finish_reason: str | None) -> None:
        self.finish_reason = finish_reason


class _FakeGoogleResponse:
    def __init__(
        self,
        text: str,
        prompt_tokens: int = 8,
        completion_tokens: int = 4,
        finish_reason: str | None = None,
    ) -> None:
        self.text = text
        self.usage_metadata = _FakeGoogleUsageMetadata(prompt_tokens, completion_tokens)
        # A real response always carries >=1 candidate; finish_reason=None
        # here models "no candidates" so the truncation-detection code's
        # `if candidates:` guard is exercised by the non-truncated tests too.
        self.candidates = [_FakeGoogleCandidate(finish_reason)] if finish_reason is not None else []


class _FakeGoogleModels:
    def __init__(self, responses: list) -> None:
        self.responses = responses
        self.calls: list[dict] = []
        self.embed_calls: list[dict] = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        idx = min(len(self.calls) - 1, len(self.responses) - 1)
        item = self.responses[idx]
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, Truncated):
            config = kwargs["config"]
            return _FakeGoogleResponse(
                item.text,
                completion_tokens=config.max_output_tokens,
                finish_reason="MAX_TOKENS",
            )
        return _FakeGoogleResponse(item)

    def embed_content(self, **kwargs):
        self.embed_calls.append(kwargs)
        texts = kwargs["contents"]
        embeddings = [type("E", (), {"values": [float(len(t))] * 4})() for t in texts]
        return type("R", (), {"embeddings": embeddings})()


class FakeGoogleClient:
    def __init__(self, responses: list | None = None) -> None:
        self.models = _FakeGoogleModels(responses or ["ok"])


class RetryableError(Exception):
    def __init__(self, status_code: int = 429) -> None:
        super().__init__(f"retryable {status_code}")
        self.status_code = status_code


class FatalError(Exception):
    def __init__(self, status_code: int = 400) -> None:
        super().__init__(f"fatal {status_code}")
        self.status_code = status_code


# ---------------------------------------------------------------------------
# Isolation fixture — every test gets a fresh cache dir, ledger, and client
# singletons; the real repo .env / data/llm_cache/ are never touched.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def isolate_llm_module(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path / "llm_cache")
    monkeypatch.setattr(llm, "_ledger", None)
    monkeypatch.setattr(llm, "_openai_client", None)
    monkeypatch.setattr(llm, "_google_client", None)
    monkeypatch.delenv("MEDMEMGRAPH_MAX_USD", raising=False)
    # Point `.env` resolution at a path that does not exist. This module's own
    # docstring already promised tests "never ... read the real `.env`", but
    # nothing enforced it: key resolution and (since 2026-08-18) the spend cap
    # both fall back to `_DEFAULT_DOTENV_PATH`, so the repo's real `.env` was
    # one `delenv` away from leaking into a test. It did, the moment
    # `_max_usd()` learned to read `.env`: the cap-default test started seeing
    # the operator's real $50 instead of the $5 default.
    monkeypatch.setattr(llm, "_DEFAULT_DOTENV_PATH", tmp_path / "nonexistent.env")
    monkeypatch.setattr(llm, "_sleep", lambda seconds: None)
    yield


def _no_network_openai(*_a, **_kw):
    raise AssertionError("dry_run/cache-hit path touched the OpenAI client getter")


def _no_network_google(*_a, **_kw):
    raise AssertionError("dry_run/cache-hit path touched the Google client getter")


# ---------------------------------------------------------------------------
# 1. Key resolution
# ---------------------------------------------------------------------------


class TestKeyResolution:
    def test_openai_env_beats_dotenv(self, tmp_path):
        dotenv = tmp_path / ".env"
        dotenv.write_text("OPENAI_API_KEY=from-dotenv\n")
        key = llm.resolve_openai_key(env={"OPENAI_API_KEY": "from-env"}, dotenv_path=dotenv)
        assert key == "from-env"

    def test_openai_canonical_name_beats_alias_in_env(self):
        key = llm.resolve_openai_key(
            env={"OPENAI_API_KEY": "canonical", "OPEN_AI_KEY": "alias"},
            dotenv_path="/nonexistent/.env",
        )
        assert key == "canonical"

    def test_openai_alias_env_beats_dotenv_canonical(self, tmp_path):
        dotenv = tmp_path / ".env"
        dotenv.write_text("OPENAI_API_KEY=from-dotenv-canonical\n")
        key = llm.resolve_openai_key(env={"OPEN_AI_KEY": "from-env-alias"}, dotenv_path=dotenv)
        assert key == "from-env-alias"

    def test_openai_falls_back_to_dotenv_open_ai_key_alias(self, tmp_path):
        # Mirrors this repo's real, non-standard .env: OPEN_AI_KEY (not
        # OPENAI_API_KEY), with a space before "=".
        dotenv = tmp_path / ".env"
        dotenv.write_text("OPEN_AI_KEY =sk-real-repo-shaped-value\n")
        key = llm.resolve_openai_key(env={}, dotenv_path=dotenv)
        assert key == "sk-real-repo-shaped-value"

    def test_openai_missing_everywhere_raises_actionable_error(self, tmp_path):
        dotenv = tmp_path / ".env"
        dotenv.write_text("SOME_OTHER_KEY=x\n")
        with pytest.raises(llm.MissingAPIKeyError) as excinfo:
            llm.resolve_openai_key(env={}, dotenv_path=dotenv)
        msg = str(excinfo.value)
        assert "OPENAI_API_KEY" in msg and "OPEN_AI_KEY" in msg

    def test_google_env_beats_dotenv_and_alias_order(self, tmp_path):
        dotenv = tmp_path / ".env"
        dotenv.write_text("GOOGLE_API_KEY=from-dotenv\n")
        key = llm.resolve_google_key(
            env={"GEMINI_API_KEY": "alias-env"}, dotenv_path=dotenv
        )
        assert key == "alias-env"

    def test_google_dotenv_canonical(self, tmp_path):
        dotenv = tmp_path / ".env"
        dotenv.write_text("GOOGLE_API_KEY=from-dotenv-canonical\n")
        key = llm.resolve_google_key(env={}, dotenv_path=dotenv)
        assert key == "from-dotenv-canonical"

    def test_google_missing_raises(self, tmp_path):
        dotenv = tmp_path / ".env"
        dotenv.write_text("# nothing relevant\n")
        with pytest.raises(llm.MissingAPIKeyError):
            llm.resolve_google_key(env={}, dotenv_path=dotenv)

    def test_a_resolved_secret_value_never_leaks_into_a_later_error(self, monkeypatch):
        # A real key DOES resolve here, then a downstream (fake) provider
        # call fails — the resolved secret must not appear anywhere in the
        # resulting ProviderError's message.
        distinctive_secret = "sk-DO-NOT-LEAK-THIS-EXACT-STRING-98765"
        monkeypatch.setattr(
            llm, "resolve_openai_key", lambda **_kw: distinctive_secret
        )

        class _CapturingClient:
            def __init__(self, key: str) -> None:
                self.key = key
                self.chat = type("Chat", (), {})()
                self.chat.completions = FakeOpenAIChat([FatalError(400)])

        monkeypatch.setattr(llm, "_openai_client", None)
        monkeypatch.setattr(
            llm,
            "_get_openai_client",
            lambda: _CapturingClient(llm.resolve_openai_key()),
        )
        with pytest.raises(llm.ProviderError) as excinfo:
            llm.complete("hi", model="gpt-4.1-mini", use_cache=False)
        assert distinctive_secret not in str(excinfo.value)


class TestDotenvParsing:
    def test_skips_comments_and_blank_lines(self, tmp_path):
        p = tmp_path / ".env"
        p.write_text("# comment\n\nOPENAI_API_KEY=abc\n\n# trailing\n")
        parsed = llm._parse_dotenv(p)
        assert parsed == {"OPENAI_API_KEY": "abc"}

    def test_handles_space_before_equals(self, tmp_path):
        p = tmp_path / ".env"
        p.write_text("OPEN_AI_KEY =value-with-space-before-eq\n")
        parsed = llm._parse_dotenv(p)
        assert parsed["OPEN_AI_KEY"] == "value-with-space-before-eq"

    def test_strips_surrounding_quotes(self, tmp_path):
        p = tmp_path / ".env"
        p.write_text('GOOGLE_API_KEY="quoted-value"\n')
        parsed = llm._parse_dotenv(p)
        assert parsed["GOOGLE_API_KEY"] == "quoted-value"

    def test_a_malformed_line_does_not_take_down_the_rest_of_the_file(self, tmp_path):
        # Shaped like this repo's real .env: one line that concatenates
        # onto the previous value with no separating newline, which trips
        # up python-dotenv's own parser (confirmed live against the real
        # file during development) but must not stop us from finding the
        # keys we actually need on later, well-formed lines.
        p = tmp_path / ".env"
        p.write_text(
            'SEMANTIC_SCHOLAR_API_KEY="abc"MEDLOCOMO_ROOT=data/medlocomo\n'
            "OPENAI_API_KEY=still-findable\n"
        )
        parsed = llm._parse_dotenv(p)
        assert parsed["OPENAI_API_KEY"] == "still-findable"

    def test_missing_file_returns_empty_dict(self, tmp_path):
        assert llm._parse_dotenv(tmp_path / "does-not-exist.env") == {}


# ---------------------------------------------------------------------------
# 2. Routing
# ---------------------------------------------------------------------------


class TestRouting:
    @pytest.mark.parametrize(
        "model,expected",
        [
            ("gpt-4.1-mini", "openai"),
            ("gpt-4o", "openai"),
            ("o1", "openai"),
            ("o3-mini", "openai"),
            ("text-embedding-3-small", "openai"),
            ("gemini-2.5-flash-lite", "google"),
            ("gemini-1.5-pro", "google"),
            ("gemma-2-9b", "google"),
        ],
    )
    def test_provider_for_model(self, model, expected):
        assert llm._provider_for_model(model) == expected

    def test_unroutable_model_raises(self):
        with pytest.raises(llm.LLMError):
            llm._provider_for_model("claude-3-5-sonnet")


class TestModelDefaults:
    def test_defaults_are_not_reasoning_models(self):
        for model in (llm.EXTRACT_MODEL, llm.ANSWER_MODEL, llm.JUDGE_MODEL):
            llm._assert_not_reasoning_default("X", model)  # must not raise

    @pytest.mark.parametrize("bad", ["gpt-5-mini", "o3-pro", "o1", "gemini-3.0-pro"])
    def test_banned_defaults_are_rejected_by_the_guard(self, bad):
        with pytest.raises(llm.LLMError):
            llm._assert_not_reasoning_default("X", bad)

    def test_judge_and_answer_are_different_families(self):
        assert llm._provider_for_model(llm.ANSWER_MODEL) != llm._provider_for_model(llm.JUDGE_MODEL)


# ---------------------------------------------------------------------------
# 3. dry_run — never touches the network
# ---------------------------------------------------------------------------


class TestDryRun:
    def test_complete_dry_run_makes_no_network_call(self, monkeypatch):
        monkeypatch.setattr(llm, "_get_openai_client", _no_network_openai)
        monkeypatch.setattr(llm, "_get_google_client", _no_network_google)
        result = llm.complete("hello", model="gpt-4.1-mini", dry_run=True)
        assert result.cached is False
        assert result.cost_usd == 0.0
        assert result.text

    def test_complete_dry_run_google_model_makes_no_network_call(self, monkeypatch):
        monkeypatch.setattr(llm, "_get_openai_client", _no_network_openai)
        monkeypatch.setattr(llm, "_get_google_client", _no_network_google)
        result = llm.complete("hello", model="gemini-2.5-flash-lite", dry_run=True)
        assert result.cost_usd == 0.0

    def test_complete_dry_run_with_schema_returns_shaped_stub(self, monkeypatch):
        monkeypatch.setattr(llm, "_get_openai_client", _no_network_openai)
        schema = {
            "type": "object",
            "properties": {"correct": {"type": "boolean"}, "reason": {"type": "string"}},
            "required": ["correct", "reason"],
            "additionalProperties": False,
        }
        result = llm.complete("q", model="gpt-4.1-mini", schema=schema, dry_run=True)
        assert result.parsed == {"correct": False, "reason": ""}
        assert not llm._validate_against_schema(result.parsed, schema)

    def test_embed_dry_run_makes_no_network_call(self, monkeypatch):
        monkeypatch.setattr(llm, "_get_openai_client", _no_network_openai)
        monkeypatch.setattr(llm, "_get_google_client", _no_network_google)
        vectors = llm.embed(["a", "b"], model="text-embedding-3-small", dry_run=True)
        assert len(vectors) == 2
        assert len(vectors[0]) == 1536

    def test_embed_dry_run_is_deterministic(self):
        v1 = llm.embed(["same text"], dry_run=True)
        v2 = llm.embed(["same text"], dry_run=True)
        assert v1 == v2

    def test_embed_dry_run_differs_for_different_text(self):
        v1 = llm.embed(["text one"], dry_run=True)
        v2 = llm.embed(["text two"], dry_run=True)
        assert v1 != v2

    def test_dry_run_never_writes_to_cache(self, monkeypatch):
        monkeypatch.setattr(llm, "_get_openai_client", _no_network_openai)
        llm.complete("hello", model="gpt-4.1-mini", dry_run=True)
        assert not llm.CACHE_DIR.exists() or not list(llm.CACHE_DIR.glob("*.json"))

    def test_dry_run_never_touches_tiktoken_network_path(self, monkeypatch):
        # Even if tiktoken's encoding were somehow unavailable/unreachable,
        # dry_run must still succeed via the offline char-based estimator.
        def _boom():
            raise AssertionError("dry_run reached the tiktoken-backed estimator")

        monkeypatch.setattr(llm, "_encoding", _boom)
        result = llm.complete("hello world", model="gpt-4.1-mini", dry_run=True)
        assert result.prompt_tokens > 0


# ---------------------------------------------------------------------------
# 4. Cache
# ---------------------------------------------------------------------------


class TestCache:
    def test_cache_hit_returns_cached_true_and_skips_network(self, monkeypatch):
        fake = FakeOpenAIClient(responses=["first response"])
        monkeypatch.setattr(llm, "_get_openai_client", lambda: fake)

        first = llm.complete("hi", model="gpt-4.1-mini", use_cache=True)
        assert first.cached is False
        assert len(fake.chat.completions.calls) == 1

        # Now make any further network access an error, and re-request the
        # exact same call — must be served entirely from cache.
        monkeypatch.setattr(llm, "_get_openai_client", _no_network_openai)
        second = llm.complete("hi", model="gpt-4.1-mini", use_cache=True)
        assert second.cached is True
        assert second.attempts == 0
        assert second.cost_usd == 0.0
        assert second.text == first.text

    def test_cache_key_ignores_dict_key_order_in_schema(self, monkeypatch):
        fake = FakeOpenAIClient(responses=[json.dumps({"a": 1, "b": 2})])
        monkeypatch.setattr(llm, "_get_openai_client", lambda: fake)
        schema1 = {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}}
        schema2 = {"properties": {"b": {"type": "integer"}, "a": {"type": "integer"}}, "type": "object"}
        llm.complete("hi", model="gpt-4.1-mini", schema=schema1)
        monkeypatch.setattr(llm, "_get_openai_client", _no_network_openai)
        second = llm.complete("hi", model="gpt-4.1-mini", schema=schema2)
        assert second.cached is True

    def test_different_prompt_is_a_cache_miss(self, monkeypatch):
        fake = FakeOpenAIClient(responses=["resp"])
        monkeypatch.setattr(llm, "_get_openai_client", lambda: fake)
        llm.complete("prompt one", model="gpt-4.1-mini")
        llm.complete("prompt two", model="gpt-4.1-mini")
        assert len(fake.chat.completions.calls) == 2

    def test_use_cache_false_bypasses_cache(self, monkeypatch):
        fake = FakeOpenAIClient(responses=["resp"])
        monkeypatch.setattr(llm, "_get_openai_client", lambda: fake)
        llm.complete("hi", model="gpt-4.1-mini", use_cache=False)
        llm.complete("hi", model="gpt-4.1-mini", use_cache=False)
        assert len(fake.chat.completions.calls) == 2

    def test_cache_records_a_ledger_entry_at_zero_cost(self, monkeypatch):
        fake = FakeOpenAIClient(responses=["resp"])
        monkeypatch.setattr(llm, "_get_openai_client", lambda: fake)
        llm.complete("hi", model="gpt-4.1-mini")
        llm.complete("hi", model="gpt-4.1-mini")
        stats = llm.get_ledger()._by_model["gpt-4.1-mini"]
        assert stats.calls == 2
        assert stats.cache_hits == 1


# ---------------------------------------------------------------------------
# 5. Ledger arithmetic
# ---------------------------------------------------------------------------


class TestLedger:
    def test_record_accumulates_per_model(self, tmp_path):
        ledger = llm.Ledger(path=tmp_path / "ledger.json")
        ledger.record("model-a", 100, 50, 0.01)
        ledger.record("model-a", 200, 100, 0.02)
        ledger.record("model-b", 10, 10, 0.001)
        assert ledger._by_model["model-a"].calls == 2
        assert ledger._by_model["model-a"].prompt_tokens == 300
        assert ledger._by_model["model-a"].completion_tokens == 150
        assert ledger._by_model["model-a"].cost_usd == pytest.approx(0.03)
        assert ledger.total_cost_usd == pytest.approx(0.031)

    def test_cached_record_increments_hits_not_cost(self, tmp_path):
        ledger = llm.Ledger(path=tmp_path / "ledger.json")
        ledger.record("model-a", 100, 50, 0.0, cached=True)
        assert ledger._by_model["model-a"].calls == 1
        assert ledger._by_model["model-a"].cache_hits == 1
        assert ledger.total_cost_usd == 0.0

    def test_persists_and_reloads_from_disk(self, tmp_path):
        path = tmp_path / "ledger.json"
        ledger1 = llm.Ledger(path=path)
        ledger1.record("model-a", 100, 50, 0.5)
        ledger2 = llm.Ledger(path=path)
        assert ledger2.total_cost_usd == pytest.approx(0.5)
        assert ledger2._by_model["model-a"].calls == 1

    def test_reserve_then_release_round_trips_to_zero_impact(self, tmp_path):
        ledger = llm.Ledger(path=tmp_path / "ledger.json")
        ledger.check_budget(0.5, cap_usd=1.0)
        assert ledger._reserved_usd == pytest.approx(0.5)
        ledger.release_reservation(0.5)
        assert ledger._reserved_usd == pytest.approx(0.0)

    def test_reservation_blocks_a_second_concurrent_overshoot(self, tmp_path):
        ledger = llm.Ledger(path=tmp_path / "ledger.json")
        ledger.check_budget(0.6, cap_usd=1.0)  # reserves 0.6, ok
        with pytest.raises(llm.BudgetExceeded):
            ledger.check_budget(0.6, cap_usd=1.0)  # 0.6 + 0.6 > 1.0

    def test_report_contains_model_names_and_total(self, tmp_path):
        ledger = llm.Ledger(path=tmp_path / "ledger.json")
        ledger.record("gpt-4.1-mini", 10, 5, 0.001)
        report = ledger.report()
        assert "gpt-4.1-mini" in report
        assert "TOTAL" in report

    def test_reset_clears_state(self, tmp_path):
        ledger = llm.Ledger(path=tmp_path / "ledger.json")
        ledger.record("model-a", 10, 5, 0.1)
        ledger.reset()
        assert ledger.total_cost_usd == 0.0
        assert ledger._by_model == {}


# ---------------------------------------------------------------------------
# 6. BudgetExceeded
# ---------------------------------------------------------------------------


class TestBudget:
    def test_raises_before_the_network_call_when_cap_would_be_exceeded(self, monkeypatch):
        monkeypatch.setenv("MEDMEMGRAPH_MAX_USD", "0.0000001")
        monkeypatch.setattr(llm, "_get_openai_client", _no_network_openai)
        with pytest.raises(llm.BudgetExceeded):
            llm.complete("hello", model="gpt-4.1-mini")

    def test_does_not_raise_when_estimate_exactly_meets_the_cap(self, monkeypatch):
        est = llm._cost_usd("gpt-4.1-mini", llm._estimate_tokens("hi"), llm.DEFAULT_MAX_TOKENS)
        monkeypatch.setenv("MEDMEMGRAPH_MAX_USD", f"{est:.10f}")
        fake = FakeOpenAIClient(responses=["ok"])
        monkeypatch.setattr(llm, "_get_openai_client", lambda: fake)
        result = llm.complete("hi", model="gpt-4.1-mini")
        assert result.text == "ok"

    def test_unpriced_model_uses_conservative_high_fallback(self):
        cost = llm._cost_usd("some-unpriced-model-xyz", 1_000_000, 1_000_000)
        cheapest_known = min(p.input + p.output for p in llm.PRICING.values())
        assert cost > cheapest_known

    def test_default_cap_is_five_dollars_when_unset(self, monkeypatch):
        monkeypatch.delenv("MEDMEMGRAPH_MAX_USD", raising=False)
        assert llm._max_usd() == llm.DEFAULT_MAX_USD == 5.00

    def test_budget_check_reads_env_fresh_not_frozen_at_import(self, monkeypatch):
        monkeypatch.setenv("MEDMEMGRAPH_MAX_USD", "123.0")
        assert llm._max_usd() == 123.0
        monkeypatch.setenv("MEDMEMGRAPH_MAX_USD", "0.5")
        assert llm._max_usd() == 0.5


# ---------------------------------------------------------------------------
# 7. Schema validation + retry
# ---------------------------------------------------------------------------

_SCHEMA = {
    "type": "object",
    "properties": {"correct": {"type": "boolean"}, "reason": {"type": "string"}},
    "required": ["correct", "reason"],
    "additionalProperties": False,
}


class TestSchemaValidation:
    def test_validator_accepts_a_conforming_object(self):
        assert llm._validate_against_schema({"correct": True, "reason": "ok"}, _SCHEMA) == []

    def test_validator_flags_missing_required(self):
        errors = llm._validate_against_schema({"correct": True}, _SCHEMA)
        assert any("reason" in e for e in errors)

    def test_validator_flags_wrong_type(self):
        errors = llm._validate_against_schema({"correct": "yes", "reason": "ok"}, _SCHEMA)
        assert any("correct" in e for e in errors)

    def test_validator_flags_additional_properties(self):
        errors = llm._validate_against_schema(
            {"correct": True, "reason": "ok", "extra": 1}, _SCHEMA
        )
        assert any("extra" in e for e in errors)

    def test_validator_recurses_into_nested_object(self):
        nested_schema = {
            "type": "object",
            "properties": {"inner": _SCHEMA},
            "required": ["inner"],
        }
        errors = llm._validate_against_schema({"inner": {"correct": True}}, nested_schema)
        assert any("reason" in e for e in errors)

    def test_validator_checks_array_items(self):
        arr_schema = {"type": "array", "items": {"type": "integer"}}
        assert llm._validate_against_schema([1, 2, 3], arr_schema) == []
        errors = llm._validate_against_schema([1, "two", 3], arr_schema)
        assert errors

    def test_validator_checks_enum(self):
        enum_schema = {"type": "string", "enum": ["a", "b"]}
        assert llm._validate_against_schema("a", enum_schema) == []
        assert llm._validate_against_schema("z", enum_schema)

    def test_complete_retries_on_schema_mismatch_then_succeeds(self, monkeypatch):
        responses = [
            json.dumps({"correct": True}),  # missing "reason" -> invalid
            "not json at all",  # invalid JSON -> invalid
            json.dumps({"correct": True, "reason": "third time"}),  # valid
        ]
        fake = FakeOpenAIClient(responses=responses)
        monkeypatch.setattr(llm, "_get_openai_client", lambda: fake)
        result = llm.complete("q", model="gpt-4.1-mini", schema=_SCHEMA)
        assert result.parsed == {"correct": True, "reason": "third time"}
        assert result.attempts == 3
        assert len(fake.chat.completions.calls) == 3

    def test_complete_raises_schema_validation_error_after_exhausting_retries(self, monkeypatch):
        fake = FakeOpenAIClient(responses=[json.dumps({"correct": True})])  # always invalid
        monkeypatch.setattr(llm, "_get_openai_client", lambda: fake)
        with pytest.raises(llm.SchemaValidationError):
            llm.complete("q", model="gpt-4.1-mini", schema=_SCHEMA)
        assert len(fake.chat.completions.calls) == llm._MAX_SCHEMA_RETRIES + 1

    def test_google_schema_path_uses_response_schema_config(self, monkeypatch):
        fake = FakeGoogleClient(responses=[json.dumps({"correct": True, "reason": "ok"})])
        monkeypatch.setattr(llm, "_get_google_client", lambda: fake)
        result = llm.complete("q", model="gemini-2.5-flash-lite", schema=_SCHEMA)
        assert result.parsed == {"correct": True, "reason": "ok"}
        sent_config = fake.models.calls[0]["config"]
        assert sent_config.response_mime_type == "application/json"

    def test_google_schema_conversion_strips_additional_properties(self):
        # Regression test for a real bug found live while running this
        # story's own required smoke call: the Gemini public API rejects
        # `additionalProperties` in `response_schema` outright (400
        # "Unknown name additional_properties") even though OpenAI's strict
        # structured-output mode requires that same field to be present.
        # `_to_google_schema` must convert via the GEMINI_API surface,
        # which silently drops it rather than failing the whole call.
        converted = llm._to_google_schema(_SCHEMA)
        dumped = converted.model_dump(exclude_none=True)
        assert "additional_properties" not in dumped
        assert dumped["required"] == ["correct", "reason"]


# ---------------------------------------------------------------------------
# 8. Retry / backoff (transient errors)
# ---------------------------------------------------------------------------


class TestRetryBackoff:
    def test_retries_a_retryable_error_then_succeeds(self, monkeypatch):
        responses = [RetryableError(429), RetryableError(503), "finally ok"]
        fake = FakeOpenAIClient(responses=responses)
        monkeypatch.setattr(llm, "_get_openai_client", lambda: fake)
        sleeps = []
        monkeypatch.setattr(llm, "_sleep", lambda s: sleeps.append(s))
        result = llm.complete("hi", model="gpt-4.1-mini")
        assert result.text == "finally ok"
        assert result.attempts == 3
        assert len(sleeps) == 2

    def test_fatal_error_is_not_retried(self, monkeypatch):
        fake = FakeOpenAIClient(responses=[FatalError(400)])
        monkeypatch.setattr(llm, "_get_openai_client", lambda: fake)
        with pytest.raises(llm.ProviderError):
            llm.complete("hi", model="gpt-4.1-mini")
        assert len(fake.chat.completions.calls) == 1

    def test_retries_exhausted_raises_provider_error(self, monkeypatch):
        fake = FakeOpenAIClient(responses=[RetryableError(500)])  # always retryable
        monkeypatch.setattr(llm, "_get_openai_client", lambda: fake)
        with pytest.raises(llm.ProviderError):
            llm.complete("hi", model="gpt-4.1-mini", max_retries=2)
        assert len(fake.chat.completions.calls) == 3  # 1 initial + 2 retries

    def test_missing_api_key_is_not_wrapped_or_retried(self, monkeypatch):
        def _raise_missing():
            raise llm.MissingAPIKeyError("no key configured")

        monkeypatch.setattr(llm, "resolve_openai_key", lambda **_kw: _raise_missing())
        with pytest.raises(llm.MissingAPIKeyError):
            llm.complete("hi", model="gpt-4.1-mini", use_cache=False)

    def test_is_retryable_duck_types_on_status_code_or_code(self):
        assert llm._is_retryable(RetryableError(429)) is True
        assert llm._is_retryable(RetryableError(503)) is True
        assert llm._is_retryable(FatalError(400)) is False
        assert llm._is_retryable(FatalError(404)) is False

        class FakeGoogleError(Exception):
            code = 500

        assert llm._is_retryable(FakeGoogleError()) is True


# ---------------------------------------------------------------------------
# 9. complete_many
# ---------------------------------------------------------------------------


class TestCompleteMany:
    def test_returns_results_in_input_order(self, monkeypatch):
        """`complete_many` must place prompt i's response at index i regardless
        of which thread finishes first.

        The fake here maps PROMPT -> response, rather than handing out responses
        in arrival order like `FakeOpenAIChat` does. That matters: with
        `max_concurrency=2` an arrival-ordered fake races, so `p0` could be
        served `r1` and the test failed intermittently while `complete_many`
        (which assigns `results[i]` by index) was perfectly correct. An
        arrival-ordered fake cannot distinguish "results came back shuffled"
        from "the fake shuffled them", so it could never have tested this
        invariant. Also adds a barrier so the completion order is *guaranteed*
        reversed, making the assertion deterministic instead of merely usually
        true."""
        started = threading.Barrier(2, timeout=5)

        class PromptKeyedChat:
            def create(self, **kwargs):
                prompt = kwargs["messages"][-1]["content"]
                if prompt in ("p0", "p1"):
                    # Force both to be in flight, then let p1 answer first.
                    started.wait()
                    if prompt == "p0":
                        time.sleep(0.05)
                return _FakeChatCompletion(prompt.replace("p", "r"))

        fake = FakeOpenAIClient()
        fake.chat.completions = PromptKeyedChat()
        monkeypatch.setattr(llm, "_get_openai_client", lambda: fake)

        results = llm.complete_many(
            ["p0", "p1", "p2"], model="gpt-4.1-mini", max_concurrency=2, use_cache=False
        )
        assert [r.text for r in results] == ["r0", "r1", "r2"]

    def test_respects_max_concurrency(self, monkeypatch):
        lock = threading.Lock()
        state = {"current": 0, "max_seen": 0}

        class SlowFakeChat:
            def create(self, **kwargs):
                with lock:
                    state["current"] += 1
                    state["max_seen"] = max(state["max_seen"], state["current"])
                time.sleep(0.02)
                with lock:
                    state["current"] -= 1
                return _FakeChatCompletion("ok")

        fake = FakeOpenAIClient()
        fake.chat.completions = SlowFakeChat()
        monkeypatch.setattr(llm, "_get_openai_client", lambda: fake)
        llm.complete_many(
            [f"p{i}" for i in range(8)], model="gpt-4.1-mini", max_concurrency=3, use_cache=False
        )
        assert state["max_seen"] <= 3

    def test_empty_prompts_returns_empty_list(self):
        assert llm.complete_many([]) == []

    def test_budget_exceeded_propagates_from_a_worker(self, monkeypatch):
        monkeypatch.setenv("MEDMEMGRAPH_MAX_USD", "0.0000001")
        monkeypatch.setattr(llm, "_get_openai_client", _no_network_openai)
        with pytest.raises(llm.BudgetExceeded):
            llm.complete_many(["p0", "p1", "p2"], model="gpt-4.1-mini", max_concurrency=2)


# ---------------------------------------------------------------------------
# 10. embed()
# ---------------------------------------------------------------------------


class TestEmbed:
    def test_openai_embed_routes_and_returns_vectors(self, monkeypatch):
        fake = FakeOpenAIClient()
        monkeypatch.setattr(llm, "_get_openai_client", lambda: fake)
        vectors = llm.embed(["a", "bb"], model="text-embedding-3-small")
        assert len(vectors) == 2
        assert len(fake.embeddings.calls) == 1

    def test_google_embed_routes_correctly(self, monkeypatch):
        fake = FakeGoogleClient()
        monkeypatch.setattr(llm, "_get_google_client", lambda: fake)
        vectors = llm.embed(["a", "bb"], model="gemini-embedding-001")
        assert len(vectors) == 2
        assert len(fake.models.embed_calls) == 1

    def test_embed_empty_list_short_circuits(self, monkeypatch):
        monkeypatch.setattr(llm, "_get_openai_client", _no_network_openai)
        assert llm.embed([]) == []

    def test_embed_caches_per_text_partial_hit(self, monkeypatch):
        fake = FakeOpenAIClient()
        monkeypatch.setattr(llm, "_get_openai_client", lambda: fake)
        llm.embed(["alpha", "beta"], model="text-embedding-3-small")
        assert len(fake.embeddings.calls) == 1
        assert fake.embeddings.calls[0]["input"] == ["alpha", "beta"]

        llm.embed(["beta", "gamma"], model="text-embedding-3-small")
        assert len(fake.embeddings.calls) == 2
        # only the uncached text should have been sent
        assert fake.embeddings.calls[1]["input"] == ["gamma"]

    def test_embed_budget_exceeded_before_network(self, monkeypatch):
        # embedding is priced far lower than chat completion (no output
        # cost, $0.02/1M input) — cap at exactly $0 so *any* positive
        # estimate exceeds it, rather than relying on a tiny-but-not-tiny-
        # enough cap value.
        monkeypatch.setenv("MEDMEMGRAPH_MAX_USD", "0")
        monkeypatch.setattr(llm, "_get_openai_client", _no_network_openai)
        with pytest.raises(llm.BudgetExceeded):
            llm.embed(["some text"], model="text-embedding-3-small")


# ---------------------------------------------------------------------------
# 11. LLMResponse shape / general integration smoke (still offline)
# ---------------------------------------------------------------------------


class TestIntegrationOffline:
    def test_complete_without_schema_has_none_parsed(self, monkeypatch):
        fake = FakeOpenAIClient(responses=["plain text answer"])
        monkeypatch.setattr(llm, "_get_openai_client", lambda: fake)
        result = llm.complete("q", model="gpt-4.1-mini")
        assert result.parsed is None
        assert result.text == "plain text answer"
        assert result.cost_usd > 0.0

    def test_complete_default_model_is_answer_model(self, monkeypatch):
        fake = FakeOpenAIClient(responses=["ok"])
        monkeypatch.setattr(llm, "_get_openai_client", lambda: fake)
        result = llm.complete("q")
        assert result.model == llm.ANSWER_MODEL

    def test_cost_matches_pricing_table(self, monkeypatch):
        fake = FakeOpenAIClient(responses=["ok"])
        fake.chat.completions = FakeOpenAIChat(["ok"])
        monkeypatch.setattr(llm, "_get_openai_client", lambda: fake)
        result = llm.complete("q", model="gpt-4.1-mini")
        price = llm.PRICING["gpt-4.1-mini"]
        expected = (result.prompt_tokens / 1_000_000) * price.input + (
            result.completion_tokens / 1_000_000
        ) * price.output
        assert result.cost_usd == pytest.approx(expected)


# ---------------------------------------------------------------------------
# 12. Truncation — detected, distinguished from a schema violation, and
# escalated (not repeated identically) — the story's Defect 1.
# ---------------------------------------------------------------------------


class TestTruncationDetectionAndEscalation:
    def test_openai_finish_reason_length_is_detected_and_recovers(self, monkeypatch):
        responses = [
            Truncated('{"correct": true, "reason": "cut off h'),  # incomplete JSON
            json.dumps({"correct": True, "reason": "recovered"}),
        ]
        fake = FakeOpenAIClient(responses=responses)
        monkeypatch.setattr(llm, "_get_openai_client", lambda: fake)
        result = llm.complete("q", model="gpt-4.1-mini", schema=_SCHEMA, max_tokens=100)
        assert result.parsed == {"correct": True, "reason": "recovered"}
        assert result.truncated is True

    def test_google_max_tokens_finish_reason_is_detected_and_recovers(self, monkeypatch):
        responses = [
            Truncated('{"correct": true, "reason": "cut'),
            json.dumps({"correct": True, "reason": "ok"}),
        ]
        fake = FakeGoogleClient(responses=responses)
        monkeypatch.setattr(llm, "_get_google_client", lambda: fake)
        result = llm.complete("q", model="gemini-2.5-flash-lite", schema=_SCHEMA, max_tokens=50)
        assert result.parsed == {"correct": True, "reason": "ok"}
        assert result.truncated is True

    def test_retry_escalates_max_tokens_rather_than_repeating_identically(self, monkeypatch):
        responses = [
            Truncated('{"correct": true, "reason": "cut off h'),
            json.dumps({"correct": True, "reason": "recovered"}),
        ]
        fake = FakeOpenAIClient(responses=responses)
        monkeypatch.setattr(llm, "_get_openai_client", lambda: fake)
        llm.complete("q", model="gpt-4.1-mini", schema=_SCHEMA, max_tokens=100)
        first_max_tokens = fake.chat.completions.calls[0]["max_completion_tokens"]
        second_max_tokens = fake.chat.completions.calls[1]["max_completion_tokens"]
        assert first_max_tokens == 100
        assert second_max_tokens > first_max_tokens  # escalated, not identical

    def test_truncation_heuristic_fires_on_json_failure_at_the_token_ceiling(self, monkeypatch):
        # No finish_reason exposed at all (candidates=[]) -- but the
        # completion consumed exactly max_tokens and the JSON never closed.
        # The story's own fallback heuristic must still catch this.
        class _NoFinishReasonChat:
            def __init__(self) -> None:
                self.calls: list[dict] = []

            def create(self, **kwargs):
                self.calls.append(kwargs)
                if len(self.calls) == 1:
                    return _FakeChatCompletion(
                        "incomplete json without a closing brace",
                        completion_tokens=kwargs["max_completion_tokens"],
                        finish_reason="stop",  # deliberately NOT "length"
                    )
                return _FakeChatCompletion(json.dumps({"correct": True, "reason": "ok"}))

        fake = FakeOpenAIClient()
        fake.chat.completions = _NoFinishReasonChat()
        monkeypatch.setattr(llm, "_get_openai_client", lambda: fake)
        result = llm.complete("q", model="gpt-4.1-mini", schema=_SCHEMA, max_tokens=20)
        assert result.truncated is True
        assert fake.chat.completions.calls[1]["max_completion_tokens"] > 20

    def test_a_genuine_schema_violation_is_not_flagged_as_truncated_and_does_not_escalate(
        self, monkeypatch
    ):
        # Valid, complete JSON, finish_reason="stop" -- just the wrong
        # shape. Must raise the plain SchemaValidationError, NOT
        # TruncationError, and every retry must use the SAME max_tokens.
        fake = FakeOpenAIClient(responses=[json.dumps({"correct": True})])  # missing "reason"
        monkeypatch.setattr(llm, "_get_openai_client", lambda: fake)
        with pytest.raises(llm.SchemaValidationError) as excinfo:
            llm.complete("q", model="gpt-4.1-mini", schema=_SCHEMA, max_tokens=50)
        assert not isinstance(excinfo.value, llm.TruncationError)
        used = {c["max_completion_tokens"] for c in fake.chat.completions.calls}
        assert used == {50}

    def test_truncation_exhausted_raises_truncation_error_distinct_from_schema_error(
        self, monkeypatch
    ):
        fake = FakeOpenAIClient(responses=[Truncated("still not valid json")])  # always truncated
        monkeypatch.setattr(llm, "_get_openai_client", lambda: fake)
        with pytest.raises(llm.TruncationError) as excinfo:
            llm.complete("q", model="gpt-4.1-mini", schema=_SCHEMA, max_tokens=64)
        # TruncationError IS-A SchemaValidationError, so old code that only
        # catches the base class keeps working unmodified.
        assert isinstance(excinfo.value, llm.SchemaValidationError)
        used = [c["max_completion_tokens"] for c in fake.chat.completions.calls]
        assert used == sorted(used)  # non-decreasing -- never re-issued identically after a step up
        assert len(set(used)) > 1  # it actually escalated at least once
        assert max(used) <= llm._TRUNCATION_MAX_TOKENS_CAP

    def test_truncation_escalation_is_capped(self):
        assert (
            llm._next_truncation_max_tokens(llm._TRUNCATION_MAX_TOKENS_CAP)
            == llm._TRUNCATION_MAX_TOKENS_CAP
        )
        far_below_cap = 1000
        assert (
            llm._next_truncation_max_tokens(far_below_cap)
            == far_below_cap * llm._TRUNCATION_ESCALATION_FACTOR
        )
        past_cap = llm._TRUNCATION_MAX_TOKENS_CAP * 10
        assert llm._next_truncation_max_tokens(past_cap) == llm._TRUNCATION_MAX_TOKENS_CAP

    def test_default_schema_max_tokens_used_when_caller_does_not_override(self, monkeypatch):
        fake = FakeOpenAIClient(responses=[json.dumps({"correct": True, "reason": "ok"})])
        monkeypatch.setattr(llm, "_get_openai_client", lambda: fake)
        llm.complete("q", model="gpt-4.1-mini", schema=_SCHEMA)  # no max_tokens passed
        assert fake.chat.completions.calls[0]["max_completion_tokens"] == llm.DEFAULT_SCHEMA_MAX_TOKENS

    def test_default_max_tokens_unchanged_for_a_non_schema_call(self, monkeypatch):
        fake = FakeOpenAIClient(responses=["plain text"])
        monkeypatch.setattr(llm, "_get_openai_client", lambda: fake)
        llm.complete("q", model="gpt-4.1-mini")  # no schema, no max_tokens override
        assert fake.chat.completions.calls[0]["max_completion_tokens"] == llm.DEFAULT_MAX_TOKENS

    def test_successful_first_attempt_is_not_flagged_truncated(self, monkeypatch):
        fake = FakeOpenAIClient(responses=[json.dumps({"correct": True, "reason": "ok"})])
        monkeypatch.setattr(llm, "_get_openai_client", lambda: fake)
        result = llm.complete("q", model="gpt-4.1-mini", schema=_SCHEMA)
        assert result.truncated is False


# ---------------------------------------------------------------------------
# 13. Failed-attempt ledger accounting — the story's Defect 2.
# ---------------------------------------------------------------------------


class TestFailedAttemptAccounting:
    def test_ledger_record_failed_attempt_accumulates_separately_from_record(self, tmp_path):
        ledger = llm.Ledger(path=tmp_path / "ledger.json")
        ledger.record_failed_attempt("model-a", 100, 50, 0.02)
        ledger.record_failed_attempt("model-a", 200, 100, 0.04)
        stats = ledger._by_model["model-a"]
        assert stats.failed_attempts == 2
        assert stats.failed_prompt_tokens == 300
        assert stats.failed_completion_tokens == 150
        assert stats.failed_cost_usd == pytest.approx(0.06)
        assert ledger.total_failed_cost_usd == pytest.approx(0.06)
        # Wholly separate from the successful/"committed" accounting.
        assert stats.calls == 0
        assert stats.cost_usd == 0.0
        assert ledger.total_cost_usd == 0.0

    def test_a_discarded_schema_retry_attempts_real_tokens_land_in_the_ledger(self, monkeypatch):
        responses = [
            json.dumps({"correct": True}),  # real completion, fails schema -- discarded
            json.dumps({"correct": True, "reason": "ok"}),  # returned to the caller
        ]
        fake = FakeOpenAIClient(responses=responses)
        monkeypatch.setattr(llm, "_get_openai_client", lambda: fake)
        llm.complete("q", model="gpt-4.1-mini", schema=_SCHEMA)
        stats = llm.get_ledger()._by_model["gpt-4.1-mini"]
        assert stats.failed_attempts == 1
        assert stats.failed_prompt_tokens > 0
        assert stats.failed_completion_tokens > 0
        assert stats.failed_cost_usd > 0.0
        # The one successful attempt is still counted exactly as before.
        assert stats.calls == 1

    def test_failed_attempt_cost_does_not_change_the_returned_responses_own_cost(self, monkeypatch):
        responses = [
            json.dumps({"correct": True}),  # discarded
            json.dumps({"correct": True, "reason": "ok"}),  # returned
        ]
        fake = FakeOpenAIClient(responses=responses)
        monkeypatch.setattr(llm, "_get_openai_client", lambda: fake)
        result = llm.complete("q", model="gpt-4.1-mini", schema=_SCHEMA)
        price = llm.PRICING["gpt-4.1-mini"]
        expected = (result.prompt_tokens / 1_000_000) * price.input + (
            result.completion_tokens / 1_000_000
        ) * price.output
        assert result.cost_usd == pytest.approx(expected)

    def test_report_shows_a_dedicated_failed_attempt_line(self, tmp_path):
        ledger = llm.Ledger(path=tmp_path / "ledger.json")
        ledger.record("gpt-4.1-mini", 10, 5, 0.001)
        ledger.record_failed_attempt("gpt-4.1-mini", 500, 300, 0.15)
        report = ledger.report()
        assert "FAILED" in report
        assert "gpt-4.1-mini" in report
        assert "0.1500" in report  # the wasted cost is visible, not merely absent

    def test_report_failed_section_present_even_with_zero_failures(self, tmp_path):
        ledger = llm.Ledger(path=tmp_path / "ledger.json")
        ledger.record("gpt-4.1-mini", 10, 5, 0.001)
        report = ledger.report()
        assert "FAILED" in report

    def test_old_ledger_json_without_failed_fields_still_loads(self, tmp_path):
        path = tmp_path / "ledger.json"
        path.write_text(json.dumps({
            "by_model": {
                "gpt-4.1-mini": {
                    "calls": 3, "cache_hits": 1, "prompt_tokens": 100,
                    "completion_tokens": 50, "cost_usd": 0.01,
                }
            }
        }))
        ledger = llm.Ledger(path=path)
        assert ledger._by_model["gpt-4.1-mini"].calls == 3
        assert ledger._by_model["gpt-4.1-mini"].failed_attempts == 0
        assert ledger.total_failed_cost_usd == 0.0


# ---------------------------------------------------------------------------
# 14. Cache-key semantics are unchanged by this fix.
# ---------------------------------------------------------------------------


class TestCacheKeySemanticsUnchanged:
    def test_a_pre_fix_cache_entry_without_the_truncated_field_still_hits(self, monkeypatch):
        # Simulates a cache file written by the pre-fix code: same key
        # computation (`_cache_key` is untouched by this story), payload
        # missing the new "truncated" key entirely.
        key = llm._cache_key(
            kind="complete", model="gpt-4.1-mini", system=None,
            prompt_or_text="hi", schema=None, temperature=0.0,
        )
        llm.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        (llm.CACHE_DIR / f"{key}.json").write_text(json.dumps({
            "text": "pre-fix cached text", "parsed": None,
            "prompt_tokens": 12, "completion_tokens": 7,
        }))
        monkeypatch.setattr(llm, "_get_openai_client", _no_network_openai)
        result = llm.complete("hi", model="gpt-4.1-mini", use_cache=True)
        assert result.cached is True
        assert result.text == "pre-fix cached text"
        assert result.truncated is False  # missing key defaults gracefully

    def test_cache_key_computation_ignores_truncation_and_max_tokens_exactly_as_before(self):
        # Same (kind, model, system, prompt, schema, temperature) -> same
        # key, regardless of max_tokens -- the documented, unchanged
        # limitation this story was explicitly told to preserve.
        key_a = llm._cache_key(
            kind="complete", model="gpt-4.1-mini", system=None,
            prompt_or_text="hi", schema=None, temperature=0.0,
        )
        key_b = llm._cache_key(
            kind="complete", model="gpt-4.1-mini", system=None,
            prompt_or_text="hi", schema=None, temperature=0.0,
        )
        assert key_a == key_b


class TestBudgetCapResolvesFromDotenvToo:
    """`MEDMEMGRAPH_MAX_USD` must resolve the same two ways the API keys do:
    `os.environ` first, then `.env`.

    Before 2026-08-18 it read only `os.environ`, so a value in `.env` took
    effect only if some other module had already called `load_dotenv()` —
    `eval/judge.py` and `hydra_client.py` both do at import, `llm` alone does
    not. The effective spend cap therefore depended on IMPORT ORDER: the same
    repo with the same `.env` enforced $50 in a harness process and $5 in a
    bare one, silently, in both directions."""

    def test_env_var_wins(self, monkeypatch, tmp_path):
        dotenv = tmp_path / ".env"
        dotenv.write_text("MEDMEMGRAPH_MAX_USD=99\n")
        monkeypatch.setattr(llm, "_DEFAULT_DOTENV_PATH", dotenv)
        monkeypatch.setenv("MEDMEMGRAPH_MAX_USD", "12.5")
        assert llm._max_usd() == 12.5

    def test_falls_back_to_dotenv_when_env_var_absent(self, monkeypatch, tmp_path):
        dotenv = tmp_path / ".env"
        dotenv.write_text("MEDMEMGRAPH_MAX_USD=42\n")
        monkeypatch.setattr(llm, "_DEFAULT_DOTENV_PATH", dotenv)
        monkeypatch.delenv("MEDMEMGRAPH_MAX_USD", raising=False)
        assert llm._max_usd() == 42.0

    def test_default_when_neither_source_has_it(self, monkeypatch, tmp_path):
        monkeypatch.setattr(llm, "_DEFAULT_DOTENV_PATH", tmp_path / "nonexistent.env")
        monkeypatch.delenv("MEDMEMGRAPH_MAX_USD", raising=False)
        assert llm._max_usd() == llm.DEFAULT_MAX_USD

    def test_garbage_value_falls_back_to_default_rather_than_crashing(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(llm, "_DEFAULT_DOTENV_PATH", tmp_path / "nonexistent.env")
        monkeypatch.setenv("MEDMEMGRAPH_MAX_USD", "not-a-number")
        assert llm._max_usd() == llm.DEFAULT_MAX_USD


class TestLocalProvider:
    """`local:<hf_id>` — on-disk weights, no API, no key, no per-token cost.

    Added 2026-08-18 so the eval can run when an API account is exhausted. The
    tests below never load real weights: `_get_local_model` is monkeypatched, so
    this class stays offline like every other test in this file."""

    MODEL = "local:some-org/some-model"

    def test_routes_to_the_local_provider(self):
        assert llm._provider_for_model(self.MODEL) == "local"
        assert llm._provider_for_model("gpt-4.1-mini") == "openai"
        assert llm._provider_for_model("gemini-3.5-flash-lite") == "google"

    def test_needs_no_api_key(self, monkeypatch):
        """Weights are on disk. Key resolution must not raise, and must not
        fall through to a provider resolver that would."""
        def _explode():  # pragma: no cover - asserts it is never called
            raise AssertionError("local model tried to resolve a provider key")

        monkeypatch.setattr(llm, "resolve_openai_key", _explode)
        monkeypatch.setattr(llm, "resolve_google_key", _explode)
        assert llm.resolve_key_for_model(self.MODEL) == ""

    def test_costs_nothing_rather_than_the_unpriced_fallback(self):
        """Without an explicit zero, a local model hits `_FALLBACK_PRICE`
        ($15/$75 per 1M, deliberately punitive) and the budget cap refuses free
        inference within a handful of calls."""
        assert llm._cost_usd(self.MODEL, 1_000_000, 1_000_000) == 0.0
        assert llm._cost_usd("gpt-4.1-mini", 1_000_000, 0) > 0.0

    def test_unroutable_model_names_still_raise(self):
        with pytest.raises(llm.LLMError, match="Cannot route model"):
            llm._provider_for_model("mistral-7b")

    def test_schema_is_appended_to_the_system_prompt(self, monkeypatch):
        """`transformers` has no constrained decoding, so a schema becomes an
        instruction and the existing schema-retry loop parses the result. The
        test asserts the schema actually reaches the model."""
        seen = {}

        class _Tok:
            pad_token_id = 0
            pad_token = None
            eos_token = "</s>"

            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
                seen["messages"] = messages
                return "PROMPT"

            def __call__(self, text, return_tensors=None, truncation=None, max_length=None):
                import torch
                return {"input_ids": torch.zeros((1, 5), dtype=torch.long)}

            def decode(self, ids, skip_special_tokens=True):
                return '{"ok": true}'

        class _LM:
            device = "cpu"

            def generate(self, **kwargs):
                import torch
                return torch.zeros((1, 8), dtype=torch.long)

        monkeypatch.setattr(llm, "_get_local_model", lambda hf_id: (_Tok(), _LM(), "cpu"))
        schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}
        text, usage = llm._local_complete_once(
            model=self.MODEL, system="You are a judge.", prompt="q",
            schema=schema, max_tokens=32, temperature=0.0,
        )
        assert text == '{"ok": true}'
        system_msg = seen["messages"][0]["content"]
        assert "You are a judge." in system_msg
        assert '"ok"' in system_msg, "schema must reach the model as an instruction"
        assert usage.prompt_tokens == 5

    def test_markdown_fence_is_stripped(self, monkeypatch):
        """Small local models wrap JSON in ```json fences despite instructions;
        an unstripped fence fails json.loads and burns a schema retry."""
        class _Tok:
            pad_token_id = 0
            pad_token = None
            eos_token = "</s>"
            def apply_chat_template(self, m, tokenize=False, add_generation_prompt=True): return "P"
            def __call__(self, text, return_tensors=None, truncation=None, max_length=None):
                import torch
                return {"input_ids": torch.zeros((1, 3), dtype=torch.long)}
            def decode(self, ids, skip_special_tokens=True):
                return '```json\n{"ok": true}\n```'

        class _LM:
            device = "cpu"
            def generate(self, **kwargs):
                import torch
                return torch.zeros((1, 6), dtype=torch.long)

        monkeypatch.setattr(llm, "_get_local_model", lambda hf_id: (_Tok(), _LM(), "cpu"))
        text, _ = llm._local_complete_once(
            model=self.MODEL, system=None, prompt="q", schema={"type": "object"},
            max_tokens=16, temperature=0.0,
        )
        assert json.loads(text) == {"ok": True}

    def test_input_truncation_is_reported_not_silent(self, monkeypatch):
        """A local model has a hard context window and — unlike an API — no
        server-side error when you exceed it; it just produces worse output. So
        the truncation must surface on the usage record, where the schema-retry
        loop and the caller can both see it."""
        class _Tok:
            pad_token_id = 0
            pad_token = None
            eos_token = "</s>"
            def apply_chat_template(self, m, tokenize=False, add_generation_prompt=True): return "P"
            def __call__(self, text, return_tensors=None, truncation=None, max_length=None):
                import torch
                assert truncation is True, "prompt must be truncated, never sent over-length"
                # Simulate the tokenizer clamping to max_length.
                return {"input_ids": torch.zeros((1, max_length), dtype=torch.long)}
            def decode(self, ids, skip_special_tokens=True): return "answer"

        class _LM:
            device = "cpu"
            def generate(self, **kwargs):
                import torch
                return torch.zeros((1, kwargs["input_ids"].shape[-1] + 3), dtype=torch.long)

        monkeypatch.setattr(llm, "_get_local_model", lambda hf_id: (_Tok(), _LM(), "cpu"))
        monkeypatch.setenv(llm.LOCAL_MAX_INPUT_TOKENS_ENV, "128")
        _text, usage = llm._local_complete_once(
            model=self.MODEL, system=None, prompt="x " * 10_000, schema=None,
            max_tokens=32, temperature=0.0,
        )
        assert usage.prompt_tokens == 128
        assert usage.truncated is True

    def test_completion_hitting_max_tokens_is_also_flagged(self, monkeypatch):
        class _Tok:
            pad_token_id = 0
            pad_token = None
            eos_token = "</s>"
            def apply_chat_template(self, m, tokenize=False, add_generation_prompt=True): return "P"
            def __call__(self, text, return_tensors=None, truncation=None, max_length=None):
                import torch
                return {"input_ids": torch.zeros((1, 4), dtype=torch.long)}
            def decode(self, ids, skip_special_tokens=True): return "truncated output"

        class _LM:
            device = "cpu"
            def generate(self, **kwargs):
                import torch  # produced exactly max_new_tokens -> ran out of room
                return torch.zeros((1, 4 + kwargs["max_new_tokens"]), dtype=torch.long)

        monkeypatch.setattr(llm, "_get_local_model", lambda hf_id: (_Tok(), _LM(), "cpu"))
        _text, usage = llm._local_complete_once(
            model=self.MODEL, system=None, prompt="q", schema=None,
            max_tokens=16, temperature=0.0,
        )
        assert usage.completion_tokens == 16
        assert usage.truncated is True


class TestLocalLoadPreflight:
    """`_preflight_local_load` — the gate that stops a local model taking the
    machine down.

    Every test fakes the machine's RAM/VRAM rather than reading the real values,
    so the suite asserts the DECISION RULE and cannot go flaky when the box it
    runs on happens to be busy or idle."""

    @staticmethod
    def _fake_machine(monkeypatch, *, ram_gb: float, vram_gb: float | None):
        monkeypatch.setattr(llm, "_available_ram_gb", lambda: ram_gb)
        import torch

        if vram_gb is None:
            monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        else:
            monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
            monkeypatch.setattr(
                torch.cuda, "mem_get_info", lambda _i=0: (int(vram_gb * 1024**3), int(16 * 1024**3))
            )
        # Force the parameter-count fallback rather than a cached-snapshot read,
        # so size comes from the model id and the test is hermetic.
        monkeypatch.setattr(llm, "_dir_size_gb", lambda _p: 0.0)

    def test_refuses_when_host_ram_is_the_binding_constraint(self, monkeypatch):
        """The real 2026-08-18 failure: plenty of VRAM, not enough host RAM.
        `from_pretrained` stages weights in host memory, so VRAM being free is
        no protection at all."""
        self._fake_machine(monkeypatch, ram_gb=1.0, vram_gb=15.0)
        with pytest.raises(llm.LocalModelError) as exc:
            llm._preflight_local_load("Qwen/Qwen2.5-7B-Instruct", "cuda", None)
        msg = str(exc.value)
        assert "host RAM" in msg
        assert "1.0 GB is available" in msg

    def test_refuses_when_vram_is_the_binding_constraint(self, monkeypatch):
        self._fake_machine(monkeypatch, ram_gb=64.0, vram_gb=14.7)
        with pytest.raises(llm.LocalModelError) as exc:
            llm._preflight_local_load("Qwen/Qwen2.5-7B-Instruct", "cuda", None)
        assert "VRAM" in str(exc.value)

    def test_error_names_the_actionable_fix(self, monkeypatch):
        """A refusal that does not say what to do next just moves the problem."""
        self._fake_machine(monkeypatch, ram_gb=64.0, vram_gb=14.7)
        with pytest.raises(llm.LocalModelError) as exc:
            llm._preflight_local_load("Qwen/Qwen2.5-7B-Instruct", "cuda", None)
        assert llm.LOCAL_QUANT_ENV in str(exc.value)

    def test_8bit_makes_a_refused_model_fit(self, monkeypatch):
        """The measured case: 7B fp16 needs ~15.5 GB and is refused; the same
        model in 8-bit needs ~7.5 GB and is allowed."""
        self._fake_machine(monkeypatch, ram_gb=64.0, vram_gb=14.7)
        with pytest.raises(llm.LocalModelError):
            llm._preflight_local_load("Qwen/Qwen2.5-7B-Instruct", "cuda", None)
        llm._preflight_local_load("Qwen/Qwen2.5-7B-Instruct", "cuda", "8bit")  # must not raise

    def test_small_model_on_a_healthy_machine_is_allowed(self, monkeypatch):
        self._fake_machine(monkeypatch, ram_gb=12.0, vram_gb=14.7)
        llm._preflight_local_load("Qwen/Qwen2.5-3B-Instruct", "cuda", None)

    def test_cpu_device_skips_the_vram_check_but_not_the_ram_check(self, monkeypatch):
        self._fake_machine(monkeypatch, ram_gb=1.0, vram_gb=None)
        with pytest.raises(llm.LocalModelError) as exc:
            llm._preflight_local_load("Qwen/Qwen2.5-7B-Instruct", "cpu", None)
        assert "host RAM" in str(exc.value)
        assert "VRAM" not in str(exc.value)

    def test_escape_hatch_bypasses_everything(self, monkeypatch):
        self._fake_machine(monkeypatch, ram_gb=0.1, vram_gb=0.1)
        monkeypatch.setenv(llm.LOCAL_SKIP_PREFLIGHT_ENV, "1")
        llm._preflight_local_load("Qwen/Qwen2.5-72B-Instruct", "cuda", None)

    def test_unparseable_model_id_assumes_7b_rather_than_zero(self, monkeypatch):
        """An unknown id must fail SAFE — assume something big — never assume 0 GB
        and wave a 70B model through."""
        self._fake_machine(monkeypatch, ram_gb=64.0, vram_gb=1.0)
        with pytest.raises(llm.LocalModelError):
            llm._preflight_local_load("some-org/mystery-model", "cuda", None)

    def test_dir_size_counts_symlinked_blobs_once(self, tmp_path):
        """A HuggingFace cache stores each weight once under `blobs/` and
        symlinks it into `snapshots/`. Summing both reports exactly double,
        which made this guard refuse a load that fit comfortably."""
        blobs = tmp_path / "blobs"
        snaps = tmp_path / "snapshots" / "abc"
        blobs.mkdir(parents=True)
        snaps.mkdir(parents=True)
        payload = b"x" * (4 * 1024 * 1024)  # 4 MB
        (blobs / "weight.bin").write_bytes(payload)
        (snaps / "weight.bin").symlink_to(blobs / "weight.bin")

        size_gb = llm._dir_size_gb(tmp_path)
        expected_gb = len(payload) / 1024**3
        assert size_gb == pytest.approx(expected_gb, rel=0.01), (
            "symlinked blob counted more than once"
        )

    def test_bad_quant_value_is_rejected_with_the_valid_options(self, monkeypatch):
        monkeypatch.setenv(llm.LOCAL_QUANT_ENV, "3bit")
        with pytest.raises(llm.LocalModelError) as exc:
            llm._get_local_model("Qwen/Qwen2.5-3B-Instruct")
        assert "8bit" in str(exc.value) and "4bit" in str(exc.value)


class TestPromptedSchemaRobustness:
    """`_extract_json_object` / `_schema_skeleton` — the prompted-structured-output
    path used by `local:` models.

    OpenAI and Google constrain decoding to the schema; `transformers` does not,
    so a local model is *asked* for JSON and mostly complies. The failures are
    cosmetic — a markdown fence, a "Here is the JSON:" preamble, a trailing
    sentence — but each one makes `json.loads` fail on otherwise-correct output,
    burns the retry budget, and surfaces as a wrong answer. That is a measurement
    error wearing a model error's clothes, which is exactly what this project's
    eval discipline exists to prevent."""

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ('{"a": 1}', {"a": 1}),
            ('```json\n{"a": 1}\n```', {"a": 1}),
            ('```\n{"a": 1}\n```', {"a": 1}),
            ('Here is the JSON:\n{"a": 1}', {"a": 1}),
            ('{"a": 1}\n\nHope that helps!', {"a": 1}),
            ('{"n": [{"x": 1}, {"x": 2}], "a": "y"}', {"n": [{"x": 1}, {"x": 2}], "a": "y"}),
        ],
    )
    def test_recovers_json_from_common_wrappers(self, raw, expected):
        assert json.loads(llm._extract_json_object(raw)) == expected

    def test_braces_inside_strings_do_not_break_depth_counting(self):
        """Clinical notes really do contain braces and quotes; a regex-based
        extractor mis-slices on them."""
        raw = '{"note": "gave {60 mg} then said \\"stop\\"", "ok": true}'
        assert json.loads(llm._extract_json_object(raw)) == {
            "note": 'gave {60 mg} then said "stop"',
            "ok": True,
        }

    def test_unbalanced_output_is_returned_for_the_retry_loop_to_reject(self):
        """A genuinely truncated object must still fail to parse — silently
        repairing it would invent content the model never produced."""
        raw = '{"notes": [{"session_id": "abc"'
        with pytest.raises(json.JSONDecodeError):
            json.loads(llm._extract_json_object(raw))

    def test_skeleton_mirrors_the_schema_shape(self):
        schema = {
            "type": "object",
            "properties": {
                "notes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "turn_ids": {"type": "array", "items": {"type": "integer"}},
                            "relevant": {"type": "boolean"},
                        },
                    },
                },
                "answer": {"type": "string"},
            },
        }
        skeleton = json.loads(llm._schema_skeleton(schema))
        assert set(skeleton) == {"notes", "answer"}
        assert isinstance(skeleton["notes"], list)
        assert set(skeleton["notes"][0]) == {"turn_ids", "relevant"}
        assert skeleton["notes"][0]["turn_ids"] == [0]
        assert skeleton["notes"][0]["relevant"] is True
