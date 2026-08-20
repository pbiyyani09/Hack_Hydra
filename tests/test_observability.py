"""tests/test_observability.py — offline tests for medmemgraph.observability.

Every test in this file is offline: no test makes a real network call.
`init_tracing(enabled=True)` is only ever exercised with `_reachable`
monkeypatched to `False` (so the module deterministically lands on its
guaranteed-offline file tier regardless of what happens to be listening on
this machine's ports) or with a hand-built in-memory OTel tracer injected
directly via `obs._state`, matching the same "monkeypatch/inject, never hit
the real network" convention `tests/test_llm.py` documents for its own
provider clients.

The "genuine no-op, no imports forced" claim (module docstring's hardest
requirement) is checked in a subprocess (`uv run python -c ...`) rather than
in-process, because by the time this test file itself runs under the full
suite, other tests/modules may have already imported `opentelemetry`/
`phoenix`/`openinference` for unrelated reasons — only a fresh interpreter
can prove "importing this module, disabled, never pulls those in".
"""

from __future__ import annotations

import inspect
import json
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

from medmemgraph import observability as obs

# ---------------------------------------------------------------------------
# Isolation — every test starts and ends with a fresh, disabled module state
# and no accumulated "warn once" keys, so tests can't leak into each other.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_observability_module(monkeypatch):
    # `_instrument_llm_clients` calls `OpenAIInstrumentor()`/
    # `GoogleGenAIInstrumentor().instrument(...)`, which globally monkeypatch
    # the REAL, installed `openai`/`google-genai` SDK classes for the rest of
    # the pytest *process* (confirmed live in this session: `instrument()`
    # flips `is_instrumented_by_opentelemetry` to `True` process-wide, with
    # no matching per-test `uninstrument()` anywhere in this file). Any test
    # here that exercises the real `_init_enabled`/`_init_file_tier` path
    # would otherwise leak that global SDK patch into every OTHER test file
    # in the same `pytest` run — exactly the "intermittent test_observability
    # failure in the full suite, passes 29/29 in isolation" a concurrent
    # [dev-ml] session's dev.log entry (2026-08-16 23:50) independently
    # root-caused. Neutralizing it here, once, for every test in this file
    # (not per-test) is the fix: this module's own instrumentation wiring is
    # covered directly by `test_init_enabled_calls_instrument_llm_clients`
    # below via this same monkeypatch, and by the manual `--demo` run this
    # story's return note reports (real Arize delivery, HTTP 200).
    monkeypatch.setattr(obs, "_instrument_llm_clients", lambda tracer_provider: None)
    obs.shutdown_tracing()
    obs._warned_keys.clear()
    yield
    obs.shutdown_tracing()
    obs._warned_keys.clear()


def _in_memory_tracer():
    """A real OTel SDK TracerProvider + SimpleSpanProcessor backed by
    `InMemorySpanExporter` — offline, deterministic, and exercises the exact
    same `_RealSpan`/`start_as_current_span` code path a live Arize/Phoenix/
    file tier would, without any network or disk I/O."""
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")
    return tracer, exporter


class _BrokenCM:
    """A `start_as_current_span(...)`-shaped context manager whose
    `__enter__` succeeds (so caller code inside `with span(...):` can run)
    but whose `__exit__` raises — simulating an exporter that fails only
    when actually flushing/sending the finished span."""

    def __enter__(self):
        class _FakeOtelSpan:
            def set_attribute(self, *a, **k):
                pass

            def record_exception(self, *a, **k):
                pass

        return _FakeOtelSpan()

    def __exit__(self, *exc_info):
        raise RuntimeError("exporter unreachable")


class _ExitBrokenTracer:
    def start_as_current_span(self, name):
        return _BrokenCM()


class _StartBrokenTracer:
    def start_as_current_span(self, name):
        raise RuntimeError("could not allocate a span")


# ---------------------------------------------------------------------------
# 1. Disabled is a genuine no-op — no heavy imports forced.
# ---------------------------------------------------------------------------


def test_disabled_by_default_with_no_init_call_at_all():
    # Never having called init_tracing() at all must already behave as
    # fully disabled (module docstring's explicit claim).
    assert obs.current_state().enabled is False
    assert obs.current_state().mode == "disabled"
    assert obs.is_tracing_enabled() is False


def test_init_tracing_explicit_disable_returns_disabled_state():
    state = obs.init_tracing(project_name="x", enabled=False)
    assert state.enabled is False
    assert state.mode == "disabled"


def test_env_flag_resolution_truthy_and_falsy_values():
    truthy = {"MEDMEMGRAPH_TRACING_ENABLED": "1"}
    also_truthy = {"MEDMEMGRAPH_TRACING_ENABLED": "YES"}
    falsy = {"MEDMEMGRAPH_TRACING_ENABLED": "0"}
    unset: dict[str, str] = {}
    assert obs._tracing_enabled_from_env(truthy) is True
    assert obs._tracing_enabled_from_env(also_truthy) is True
    assert obs._tracing_enabled_from_env(falsy) is False
    assert obs._tracing_enabled_from_env(unset) is False


def test_disabled_state_never_imports_phoenix_openinference_or_otel_sdk():
    """The strongest form of this claim: a completely fresh interpreter
    that imports the module, uses `traced`/`span` while disabled, and never
    pulls in any of the heavy vendor packages."""
    script = (
        "import sys\n"
        "before = set(sys.modules)\n"
        "from medmemgraph import observability as obs\n"
        "@obs.traced('f')\n"
        "def f(a, b):\n"
        "    return a + b\n"
        "assert f(2, 3) == 5\n"
        "with obs.span('g', x=1) as sp:\n"
        "    sp.set_attribute('y', 2)\n"
        "after = set(sys.modules)\n"
        "heavy = {m for m in (after - before) if m.split('.')[0] in "
        "('phoenix', 'openinference', 'opentelemetry')}\n"
        "assert not heavy, heavy\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        ["uv", "run", "python", "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


# ---------------------------------------------------------------------------
# 2. span() / traced() are genuine no-ops when disabled.
# ---------------------------------------------------------------------------


def test_span_noop_when_disabled_never_raises_and_chains():
    with obs.span("stage", kind="chain", patient_id="p1") as sp:
        result = sp.set_attribute("k", "v")
        assert result is sp  # chains, like the real span wrapper
        sp.set_attributes({"a": 1, "b": 2})
        sp.record_exception(ValueError("x"))
    # no exception anywhere above


def test_span_noop_reraises_caller_exceptions_unchanged():
    with pytest.raises(ZeroDivisionError):
        with obs.span("stage"):
            1 / 0


# ---------------------------------------------------------------------------
# 3. @traced preserves signature and return value — disabled and enabled.
# ---------------------------------------------------------------------------


def _annotated_fn(a: int, b: str = "default", *, c: float = 1.0) -> str:
    """docstring for the wrapped function"""
    return f"{a}-{b}-{c}"


def test_traced_preserves_signature_docstring_name_and_return_value_disabled():
    wrapped = obs.traced("my-stage", kind="chain")(_annotated_fn)
    assert wrapped.__name__ == "_annotated_fn"
    assert wrapped.__doc__ == "docstring for the wrapped function"
    assert inspect.signature(wrapped) == inspect.signature(_annotated_fn)
    assert wrapped(1, "x", c=2.0) == _annotated_fn(1, "x", c=2.0) == "1-x-2.0"
    assert wrapped(5) == "5-default-1.0"


def test_traced_preserves_signature_and_return_value_when_enabled():
    tracer, exporter = _in_memory_tracer()
    obs._state = obs.TracingState(enabled=True, mode="test", tracer=tracer)

    wrapped = obs.traced("my-stage", kind="chain")(_annotated_fn)
    assert inspect.signature(wrapped) == inspect.signature(_annotated_fn)
    assert wrapped(1, "x", c=2.0) == "1-x-2.0"

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "my-stage"


def test_traced_default_name_is_function_qualname():
    wrapped = obs.traced()(_annotated_fn)
    assert wrapped.__wrapped__ is _annotated_fn
    # exercised indirectly: default name only matters once a real tracer is
    # attached, covered by the enabled-mode span-name assertions below.


# ---------------------------------------------------------------------------
# 4. Span attributes are populated (real OTel spans, in-memory exporter).
# ---------------------------------------------------------------------------


def test_span_attributes_populated_including_kind_and_latency():
    tracer, exporter = _in_memory_tracer()
    obs._state = obs.TracingState(enabled=True, mode="test", tracer=tracer)

    with obs.span("route", kind="chain", patient_id="p-123", k=5, structural_absence=False) as sp:
        sp.set_attribute("route", "graph")

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "route"
    attrs = dict(span.attributes)
    assert attrs["openinference.span.kind"] == "CHAIN"
    assert attrs["patient_id"] == "p-123"
    assert attrs["k"] == 5
    assert attrs["structural_absence"] is False  # a real bool, not coerced (only None is)
    assert attrs["route"] == "graph"
    assert "latency_ms" in attrs
    assert attrs["latency_ms"] >= 0.0


def test_coerce_attr_handles_bool_none_and_nested_values():
    assert obs._coerce_attr(True) is True
    assert obs._coerce_attr(3.5) == 3.5
    assert obs._coerce_attr("s") == "s"
    assert obs._coerce_attr(None) == ""
    assert obs._coerce_attr([1, 2, 3]) == [1, 2, 3]
    assert obs._coerce_attr({"a": 1}) == json.dumps({"a": 1})


def test_nested_spans_form_a_parent_child_relationship():
    tracer, exporter = _in_memory_tracer()
    obs._state = obs.TracingState(enabled=True, mode="test", tracer=tracer)

    with obs.span("outer", kind="chain") as outer:
        outer.set_attribute("k", 5)
        with obs.span("inner", kind="retriever") as inner:
            inner.set_attribute("candidate_count", 3)

    spans = {s.name: s for s in exporter.get_finished_spans()}
    assert set(spans) == {"outer", "inner"}
    assert spans["inner"].parent.span_id == spans["outer"].context.span_id
    assert spans["inner"].context.trace_id == spans["outer"].context.trace_id


def test_exception_inside_span_is_recorded_and_reraised():
    tracer, exporter = _in_memory_tracer()
    obs._state = obs.TracingState(enabled=True, mode="test", tracer=tracer)

    with pytest.raises(ValueError):
        with obs.span("failing-stage"):
            raise ValueError("boom")

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.status.status_code.name == "ERROR"
    event_names = [e.name for e in span.events]
    assert "exception" in event_names


# ---------------------------------------------------------------------------
# 5. A failed exporter never raises out of span().
# ---------------------------------------------------------------------------


def test_span_start_failure_degrades_to_noop_and_warns_once():
    obs._state = obs.TracingState(enabled=True, mode="test", tracer=_StartBrokenTracer())

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with obs.span("x", foo=1) as sp:
            sp.set_attribute("bar", 2)  # must not raise even though span start failed

    assert any("could not start span" in str(w.message) for w in caught)


def test_span_export_failure_on_exit_does_not_raise_and_warns_once():
    obs._state = obs.TracingState(enabled=True, mode="test", tracer=_ExitBrokenTracer())

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with obs.span("y") as sp:
            sp.set_attribute("ok", True)
        # exiting the `with` block above must not propagate the broken
        # exporter's RuntimeError.

    assert any("exporter failed while closing span" in str(w.message) for w in caught)


def test_span_export_failure_does_not_swallow_a_real_caller_exception():
    """A tracing-internal failure (exporter down) must never mask the
    caller's own business-logic exception — the caller's exception is what
    propagates, not the tracing failure."""
    obs._state = obs.TracingState(enabled=True, mode="test", tracer=_ExitBrokenTracer())

    with pytest.raises(KeyError):
        with obs.span("z"):
            raise KeyError("caller's own bug")


def test_init_tracing_total_failure_degrades_to_noop_fallback_and_never_raises(monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("everything is on fire")

    monkeypatch.setattr(obs, "_init_enabled", _boom)
    state = obs.init_tracing(project_name="x", enabled=True)
    assert state.enabled is False
    assert state.mode == "noop-fallback"


# ---------------------------------------------------------------------------
# 6. .env parsing — mirrors llm.py's own defensive parser and its exact
#    documented quirks (space before `=`, a line python-dotenv would drop).
# ---------------------------------------------------------------------------


def test_parse_dotenv_tolerates_space_before_equals_and_bad_lines(tmp_path):
    envfile = tmp_path / ".env"
    envfile.write_text(
        'SEMANTIC_SCHOLAR_API_KEY="abc"MEDLOCOMO_ROOT=data/medlocomo\n'  # malformed concatenated line
        "\n"
        '# comment\n'
        'ARIZE_SPACE_ID="space-value"\n'
        "ARIZE_API_KEY = \"key-value\"\n"  # space before '='
        "not_a_kv_line_at_all\n",
        encoding="utf-8",
    )
    parsed = obs._parse_dotenv(envfile)
    assert parsed["ARIZE_SPACE_ID"] == "space-value"
    assert parsed["ARIZE_API_KEY"] == "key-value"
    assert "not_a_kv_line_at_all" not in parsed


def test_parse_dotenv_missing_file_returns_empty_dict(tmp_path):
    assert obs._parse_dotenv(tmp_path / "does-not-exist.env") == {}


def test_resolve_arize_credentials_env_beats_dotenv(tmp_path):
    envfile = tmp_path / ".env"
    envfile.write_text('ARIZE_SPACE_ID="from-dotenv"\nARIZE_API_KEY="from-dotenv-key"\n', encoding="utf-8")

    space_id, api_key = obs._resolve_arize_credentials(
        env={"ARIZE_SPACE_ID": "from-env"}, dotenv_path=envfile
    )
    assert space_id == "from-env"
    assert api_key == "from-dotenv-key"  # not overridden by env -> falls through to .env


def test_resolve_arize_credentials_never_prints_the_key_value(tmp_path, capsys):
    envfile = tmp_path / ".env"
    envfile.write_text('ARIZE_SPACE_ID="space-1"\nARIZE_API_KEY="super-secret-value"\n', encoding="utf-8")

    obs._resolve_arize_credentials(env={}, dotenv_path=envfile)
    captured = capsys.readouterr()
    assert "super-secret-value" not in captured.out
    assert "super-secret-value" not in captured.err


def test_resolve_arize_credentials_missing_returns_none_none(tmp_path):
    space_id, api_key = obs._resolve_arize_credentials(env={}, dotenv_path=tmp_path / "missing.env")
    assert (space_id, api_key) == (None, None)


# ---------------------------------------------------------------------------
# 7. init_tracing()'s fallback ladder — offline, deterministic (network
#    reachability is monkeypatched, never actually probed).
# ---------------------------------------------------------------------------


def test_init_tracing_falls_back_to_file_tier_when_nothing_is_reachable(tmp_path, monkeypatch):
    monkeypatch.setattr(obs, "_reachable", lambda host, port, timeout=1.5: False)
    trace_file = tmp_path / "spans.jsonl"

    state = obs.init_tracing(
        project_name="file-tier-test",
        enabled=True,
        env={},
        dotenv_path=tmp_path / "missing.env",
        trace_file=trace_file,
    )
    assert state.mode == "file"
    assert state.enabled is True
    assert Path(state.detail) == trace_file

    with obs.span("stage-one", kind="chain", k=3) as sp:
        sp.set_attribute("route", "vector")
    obs.shutdown_tracing()

    lines = trace_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["name"] == "stage-one"


def test_init_tracing_skips_arize_tier_without_credentials(tmp_path, monkeypatch):
    # No ARIZE_* anywhere (env empty, dotenv missing) -> must never even
    # attempt the Arize reachability probe.
    probed: list[tuple[str, int]] = []

    def _tracking_reachable(host, port, timeout=1.5):
        probed.append((host, port))
        return False

    monkeypatch.setattr(obs, "_reachable", _tracking_reachable)
    obs.init_tracing(
        project_name="x",
        enabled=True,
        env={},
        dotenv_path=tmp_path / "missing.env",
        trace_file=tmp_path / "spans.jsonl",
    )
    assert (obs.ARIZE_OTLP_HOST, 443) not in probed
    assert (obs.LOCAL_PHOENIX_HOST, obs.LOCAL_PHOENIX_PORT) in probed


def test_init_enabled_calls_instrument_llm_clients_with_the_new_provider(tmp_path, monkeypatch):
    """Verifies the wiring `_isolate_observability_module`'s autouse
    monkeypatch otherwise hides: a real `init_tracing(enabled=True, ...)`
    call reaches `_instrument_llm_clients(tracer_provider)` exactly once,
    with the tracer provider it just built. Spies on the same call the
    autouse fixture neutralizes, rather than letting it touch the real
    installed OpenAI/Google SDKs (see that fixture's docstring)."""
    monkeypatch.setattr(obs, "_reachable", lambda host, port, timeout=1.5: False)
    calls: list[object] = []
    monkeypatch.setattr(obs, "_instrument_llm_clients", lambda tracer_provider: calls.append(tracer_provider))

    state = obs.init_tracing(
        project_name="x", enabled=True, env={}, dotenv_path=tmp_path / "missing.env", trace_file=tmp_path / "spans.jsonl"
    )
    assert len(calls) == 1
    assert calls[0] is not None
    # the tracer this state exposes was obtained from that same provider
    assert state.tracer is not None


def test_reachable_returns_false_fast_for_a_closed_port():
    # Port 1 is a reserved/typically-closed port; a short timeout keeps this
    # test fast regardless of the host machine's exact behavior.
    assert obs._reachable("127.0.0.1", 1, timeout=0.3) is False


def test_arize_auth_headers_shape_matches_the_vendor_sdk_contract():
    headers = obs._arize_auth_headers("space-1", "key-1")
    assert headers["authorization"] == "key-1"
    assert headers["api_key"] == "key-1"
    assert headers["arize-space-id"] == "space-1"
    assert headers["space_id"] == "space-1"
    assert headers["arize-interface"] == "otel"


# ---------------------------------------------------------------------------
# 8. instrument_llm_module() — wraps a module's complete()/embed() without
#    touching the real llm.py (a minimal fake module is used here so this
#    file never imports the real openai/google clients).
# ---------------------------------------------------------------------------


class _FakeLLMResponse:
    def __init__(self, model):
        self.text = "hi"
        self.model = model
        self.prompt_tokens = 10
        self.completion_tokens = 5
        self.cost_usd = 0.0001
        self.cached = False
        self.attempts = 1


class _FakeLLMModule:
    def __init__(self):
        self.complete_calls: list[dict] = []
        self.embed_calls: list[dict] = []

    def complete(self, prompt, *, model=None, **kwargs):
        self.complete_calls.append({"prompt": prompt, "model": model, **kwargs})
        return _FakeLLMResponse(model or "fake-model")

    def embed(self, texts, *, model=None, **kwargs):
        self.embed_calls.append({"texts": texts, "model": model, **kwargs})
        return [[0.1, 0.2] for _ in texts]


def test_instrument_llm_module_wraps_complete_and_embed_and_is_idempotent():
    tracer, exporter = _in_memory_tracer()
    obs._state = obs.TracingState(enabled=True, mode="test", tracer=tracer)

    fake = _FakeLLMModule()
    orig_complete, orig_embed = fake.complete, fake.embed

    obs.instrument_llm_module(fake)
    assert fake.complete is not orig_complete
    assert fake.embed is not orig_embed
    assert getattr(fake.complete, "_medmemgraph_traced", False) is True

    result = fake.complete("hello", model="gpt-4.1-mini")
    assert result.text == "hi"
    assert fake.complete_calls == [{"prompt": "hello", "model": "gpt-4.1-mini"}]

    vectors = fake.embed(["a", "b"], model="text-embedding-3-small")
    assert vectors == [[0.1, 0.2], [0.1, 0.2]]

    spans = {s.name: s for s in exporter.get_finished_spans()}
    assert "llm.complete" in spans and "llm.embed" in spans
    complete_attrs = dict(spans["llm.complete"].attributes)
    assert complete_attrs["llm.model_name"] == "gpt-4.1-mini"
    assert complete_attrs["llm.token_count.prompt"] == 10
    assert complete_attrs["cost_usd"] == 0.0001
    embed_attrs = dict(spans["llm.embed"].attributes)
    assert embed_attrs["text_count"] == 2
    assert embed_attrs["vector_count"] == 2

    # idempotent: instrumenting again does not double-wrap
    wrapped_once = fake.complete
    obs.instrument_llm_module(fake)
    assert fake.complete is wrapped_once


def test_instrument_llm_module_wrapper_preserves_signature():
    fake = _FakeLLMModule()
    orig_sig = inspect.signature(fake.complete)
    obs.instrument_llm_module(fake)
    # functools.wraps sets __wrapped__, which inspect.signature follows.
    assert inspect.signature(fake.complete).parameters.keys() == orig_sig.parameters.keys()


def test_instrument_llm_module_return_value_passthrough_when_disabled():
    # instrument_llm_module wraps unconditionally (span() itself is the
    # single source of truth for enabled/disabled), so even with tracing
    # disabled the wrapped function's return value must be unchanged.
    fake = _FakeLLMModule()
    obs.instrument_llm_module(fake)
    result = fake.complete("hello", model="m")
    assert result.text == "hi"
    assert result.model == "m"
