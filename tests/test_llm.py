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
        fake = FakeOpenAIClient(responses=["r0", "r1", "r2"])
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
