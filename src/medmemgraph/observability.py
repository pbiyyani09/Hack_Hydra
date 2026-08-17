"""observability.py — OpenTelemetry / Arize Phoenix tracing for MedMemGraph.

## Why this file exists

The user wants traceability: a single place to see how one question flows
through the whole read path (`retrieve -> route -> embed -> search ->
rerank -> traverse -> fuse -> read`), not just the raw LLM calls. This
module is that seam: it configures an OpenTelemetry `TracerProvider`, wires
the vendor auto-instrumentors for the OpenAI and Google GenAI SDKs, and
gives the rest of the codebase two small, always-safe primitives —
`@traced(...)` and `span(...)` — to mark the non-LLM pipeline stages.

## Hard requirement: tracing must never be able to break a run

Every path in this module that could touch the network (registering an
exporter, starting a span, ending a span, instrumenting a client) is wrapped
so that a failure degrades to a warning and a no-op, never an exception that
propagates into caller code. The one exception, by design, is the caller's
own business-logic exception raised *inside* a `with span(...):` block —
that is not this module's to swallow; it is recorded on the span (so the
trace shows where things broke) and then re-raised unchanged.

## Enable/disable

Tracing is **opt-in**, off by default. `init_tracing(enabled=None)` resolves
`enabled` from the `MEDMEMGRAPH_TRACING_ENABLED` env var (truthy strings:
`1`/`true`/`yes`/`on`; anything else, including unset, is disabled) unless
an explicit `True`/`False` is passed. When disabled — which includes never
having called `init_tracing()` at all, since every entry point below is a
genuine no-op until `init_tracing()` flips `_state.enabled` — this module
never imports `phoenix`, `openinference`, or `opentelemetry.sdk.*`. Only
this file's own stdlib-only top-level imports ever run.

## Fallback ladder (never dies, never silently drops the "where did this
question's evidence come from" record)

1. **Arize AX (Arize Cloud)** — `ARIZE_SPACE_ID` + `ARIZE_API_KEY` resolved
   (env beats `.env`, `.env` parsed with the same small hand-rolled,
   permissive line parser `llm.py` uses — see that module's docstring for
   exactly why `python-dotenv` is not trusted on this repo's real `.env`;
   never logs a key value) AND a quick TCP probe to `otlp.arize.com:443`
   succeeds.
2. **Local Phoenix** — a TCP probe to `localhost:6006` succeeds (a Phoenix
   server the user happens to have running locally, `docker run
   arizephoenix/phoenix` or `phoenix serve`).
3. **File-based OTel exporter** — always succeeds (local disk, no network):
   one JSON span per line, appended to `MEDMEMGRAPH_TRACE_FILE` (default
   `data/traces/spans.jsonl`). This is the guaranteed-offline tier.
4. **No-op** — tracing disabled, or every tier above raised during setup.
   `init_tracing` never raises; it always returns a `TracingState` and
   prints (once) which mode is active.

`init_tracing` picks the first tier that actually works and says so.

## Instrumentation strategy — wrap, never restructure

Two of this module's own docstring's instructions apply the same principle
to two different files:

- `llm.py` is 1300 lines with 82 tests other agents depend on. This module
  never edits it. `instrument_llm_module()` monkeypatches only the two
  public entry points other code actually calls (`complete`, `embed`) with
  a `functools.wraps`-preserving span wrapper, from the outside, only when
  tracing is enabled. In addition, `openinference-instrumentation-openai`
  and `-google-genai` auto-instrument the underlying `openai`/`google.genai`
  SDK client classes those functions call internally (zero-code, vendor
  auto-instrumentation) — so a real API call shows both a project-level
  `llm.complete`/`llm.embed` span (our cost/cache/latency accounting) and a
  nested provider-level `LLM`/`EMBEDDING` span (the vendor's own
  prompt/response/token-usage capture).
  Known limitation, stated rather than hidden: `complete_many`'s internal
  fan-out runs each `complete()` call on a `ThreadPoolExecutor` worker
  thread; OpenTelemetry's context (which span is "current") is carried via
  `contextvars` and is not automatically propagated to a thread spawned by
  `submit()`. Each of those nested `llm.complete` spans is therefore
  correctly recorded, but may appear as its own root rather than a child of
  whatever span called `complete_many` — an honest trace-shape gap, not a
  silently wrong parent/child link.
- `graph/retrieve.py` names its own stage boundaries in code (route, the
  text arm's dense+lexical search, the graph arm's seed/traverse/rerank,
  fuse) inside a handful of already-small, already-labelled functions/
  blocks. That module directly imports `traced`/`span` from here and wraps
  its own stage boundaries in place — the additive, always-safe idiom this
  module exists to support — rather than being monkeypatched from outside.
  When tracing is disabled this costs one attribute-dict-free boolean check
  per stage; behavior, return values, and every existing test are
  unaffected (verified: `uv run pytest tests/test_retrieve.py` after the
  edit, see this story's dev.log entry).
"""

from __future__ import annotations

import functools
import json
import os
import socket
import sys
import threading
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, TypeVar

__all__ = [
    "TracingState",
    "init_tracing",
    "shutdown_tracing",
    "current_state",
    "is_tracing_enabled",
    "traced",
    "span",
    "instrument_llm_module",
    "TracingError",
]

# ---------------------------------------------------------------------------
# Config surface
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DOTENV_PATH = _REPO_ROOT / ".env"
_DEFAULT_TRACE_FILE = Path(os.environ.get("MEDMEMGRAPH_TRACE_FILE", "data/traces/spans.jsonl"))

ENV_TRACING_FLAG = "MEDMEMGRAPH_TRACING_ENABLED"

# Arize's real OTLP-ingest endpoint and auth-header contract, NOT the
# Phoenix-Cloud `/s/<space_id>` + Bearer-token shape `phoenix.otel.register`
# infers for the `app.phoenix.arize.com` hostname. Confirmed live in this
# session by downloading and reading (not installing — that would be a new,
# out-of-scope dependency) the `arize-otel` wheel from PyPI: `Endpoint.ARIZE
# = "https://otlp.arize.com/v1"` and `_get_arize_auth_headers(space_id,
# api_key)` in `arize/otel/otel.py` (package version 0.13.0). Replicated
# here using only the already-installed `arize-phoenix-otel`'s generic
# `endpoint=`/`headers=`/`protocol=` passthrough — `phoenix.otel.register`
# and `arize.otel.register` are both thin convenience wrappers over the same
# underlying `opentelemetry-exporter-otlp-proto-http` exporter, confirmed by
# reading both sources: neither wrapper rewrites an explicitly-passed
# `endpoint`, so this is not a guess dressed up as a fact.
ARIZE_OTLP_HOST = "otlp.arize.com"
ARIZE_OTLP_ENDPOINT = f"https://{ARIZE_OTLP_HOST}/v1"
LOCAL_PHOENIX_HOST = "localhost"
LOCAL_PHOENIX_PORT = 6006
LOCAL_PHOENIX_ENDPOINT = f"http://{LOCAL_PHOENIX_HOST}:{LOCAL_PHOENIX_PORT}"


def _arize_auth_headers(space_id: str, api_key: str) -> dict[str, str]:
    """Exact header set `arize-otel` 0.13.0 sends (see constant comment
    above) — `authorization` is the RAW api key, not a `Bearer `-prefixed
    value (Arize's own convention, different from Phoenix Cloud's). Never
    logged; only ever placed directly into the OTLP exporter's headers."""
    return {
        "authorization": api_key,
        "api_key": api_key,  # deprecated alias, kept for parity with the vendor SDK
        "arize-space-id": space_id,
        "space_id": space_id,  # deprecated alias, kept for parity with the vendor SDK
        "arize-interface": "otel",
    }

# openinference semantic-convention attribute keys, spelled out as literal
# strings (confirmed live against `openinference.semconv.trace.SpanAttributes`
# in this session) so this module never needs that import at all when
# tracing is disabled.
_ATTR_SPAN_KIND = "openinference.span.kind"


class TracingError(Exception):
    """This module's own errors. Never allowed to escape `init_tracing`,
    `traced`, or `span` — always caught internally and downgraded to a
    single warning plus a no-op fallback."""


# ---------------------------------------------------------------------------
# .env parsing — defensive, mirrors llm.py's `_parse_dotenv` exactly (see
# that module's docstring: `python-dotenv` silently drops a line in this
# repo's real `.env`). Deliberately re-implemented here, not imported from
# `llm.py`, so this module has zero coupling and zero import cost of its
# own when tracing is disabled.
# ---------------------------------------------------------------------------


def _parse_dotenv(path: Path) -> dict[str, str]:
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


def _resolve_arize_credentials(
    *, env: Mapping[str, str] | None = None, dotenv_path: str | Path | None = None
) -> tuple[str | None, str | None]:
    """`(space_id, api_key)`. Env always beats `.env`. Never logs either
    value — only ever returns them for direct use as OTel exporter headers."""
    env = env if env is not None else os.environ
    path = Path(dotenv_path) if dotenv_path is not None else _DEFAULT_DOTENV_PATH
    dotenv_map = _parse_dotenv(path)
    space_id = (env.get("ARIZE_SPACE_ID") or dotenv_map.get("ARIZE_SPACE_ID") or "").strip()
    api_key = (env.get("ARIZE_API_KEY") or dotenv_map.get("ARIZE_API_KEY") or "").strip()
    return (space_id or None), (api_key or None)


def _tracing_enabled_from_env(env: Mapping[str, str] | None = None) -> bool:
    env = env if env is not None else os.environ
    raw = env.get(ENV_TRACING_FLAG)
    if raw is None:
        return False
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _reachable(host: str, port: int, timeout: float = 1.5) -> bool:
    """Best-effort, short-timeout TCP probe used only to CHOOSE a fallback
    tier faster than waiting on a real exporter's own timeout. Never raises;
    a probe failure just means "try the next tier", not "the run is broken"."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass
class TracingState:
    """The result of `init_tracing()`. `mode` is one of: `"disabled"`
    (never enabled, or explicitly disabled), `"arize"`, `"local-phoenix"`,
    `"file"`, or `"noop-fallback"` (enabled was requested but every tier —
    including the guaranteed-offline file tier — failed; this should only
    happen if the local filesystem itself is unwritable)."""

    enabled: bool = False
    mode: str = "disabled"
    project_name: str = "medmemgraph"
    tracer: Any = None
    detail: str = ""


_state = TracingState()
_state_lock = threading.RLock()
_warned_keys: set[str] = set()


def _warn_once(key: str, message: str) -> None:
    if key in _warned_keys:
        return
    _warned_keys.add(key)
    warnings.warn(f"[medmemgraph.observability] {message}", RuntimeWarning, stacklevel=3)


def current_state() -> TracingState:
    return _state


def is_tracing_enabled() -> bool:
    return _state.enabled


# ---------------------------------------------------------------------------
# init_tracing()
# ---------------------------------------------------------------------------


def init_tracing(
    project_name: str = "medmemgraph",
    enabled: bool | None = None,
    *,
    env: Mapping[str, str] | None = None,
    dotenv_path: str | Path | None = None,
    trace_file: str | Path | None = None,
) -> TracingState:
    """Sets up the OTel tracer provider and auto-instruments the OpenAI and
    Google GenAI clients (`llm.py`'s two provider SDKs). Safe to call more
    than once (e.g. once per test) — each call fully replaces the prior
    state. Never raises: any failure in any tier degrades to the next tier,
    and total failure degrades to a `"noop-fallback"` `TracingState` plus
    one warning, not an exception.
    """
    global _state
    env = env if env is not None else os.environ
    resolved_enabled = enabled if enabled is not None else _tracing_enabled_from_env(env)

    with _state_lock:
        if not resolved_enabled:
            _state = TracingState(enabled=False, mode="disabled", project_name=project_name)
            return _state

        try:
            new_state = _init_enabled(
                project_name, env=env, dotenv_path=dotenv_path, trace_file=trace_file
            )
        except Exception as exc:  # noqa: BLE001 - deliberately broad, see module docstring
            _warn_once(
                "init-total-failure",
                f"tracing init failed end-to-end ({exc!r}); degrading to local no-op. "
                "The run continues untraced.",
            )
            new_state = TracingState(
                enabled=False, mode="noop-fallback", project_name=project_name, detail=repr(exc)
            )
        _state = new_state
        print(
            f"[medmemgraph.observability] tracing mode: {_state.mode}"
            + (f" ({_state.detail})" if _state.detail else "")
        )
        return _state


def _init_enabled(
    project_name: str,
    *,
    env: Mapping[str, str],
    dotenv_path: str | Path | None,
    trace_file: str | Path | None,
) -> TracingState:
    from phoenix.otel import register as phoenix_register

    space_id, api_key = _resolve_arize_credentials(env=env, dotenv_path=dotenv_path)

    if space_id and api_key and _reachable(ARIZE_OTLP_HOST, 443):
        try:
            tracer_provider = phoenix_register(
                endpoint=ARIZE_OTLP_ENDPOINT,
                headers=_arize_auth_headers(space_id, api_key),
                protocol="http/protobuf",
                project_name=project_name,
                batch=True,
                verbose=False,
                auto_instrument=False,
                set_global_tracer_provider=False,
            )
            _instrument_llm_clients(tracer_provider)
            tracer = tracer_provider.get_tracer(project_name)
            return TracingState(
                enabled=True,
                mode="arize",
                project_name=project_name,
                tracer=tracer,
                detail=f"Arize ({ARIZE_OTLP_ENDPOINT}), space {space_id[:8]}…",
            )
        except Exception as exc:  # noqa: BLE001
            _warn_once("arize-register-failed", f"Arize Cloud registration failed ({exc!r}); trying local Phoenix.")

    if _reachable(LOCAL_PHOENIX_HOST, LOCAL_PHOENIX_PORT):
        try:
            tracer_provider = phoenix_register(
                endpoint=LOCAL_PHOENIX_ENDPOINT,
                project_name=project_name,
                batch=True,
                verbose=False,
                auto_instrument=False,
                set_global_tracer_provider=False,
            )
            _instrument_llm_clients(tracer_provider)
            tracer = tracer_provider.get_tracer(project_name)
            return TracingState(
                enabled=True,
                mode="local-phoenix",
                project_name=project_name,
                tracer=tracer,
                detail=LOCAL_PHOENIX_ENDPOINT,
            )
        except Exception as exc:  # noqa: BLE001
            _warn_once("local-phoenix-register-failed", f"local Phoenix registration failed ({exc!r}); falling back to a file exporter.")

    # File tier — local disk only, no network, always succeeds unless the
    # filesystem itself is unwritable (in which case the caller's own
    # try/except in init_tracing() downgrades this to "noop-fallback").
    return _init_file_tier(project_name, trace_file=trace_file)


def _init_file_tier(project_name: str, *, trace_file: str | Path | None) -> TracingState:
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider as SDKTracerProvider
    from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

    path = Path(trace_file) if trace_file is not None else _DEFAULT_TRACE_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    out = path.open("a", encoding="utf-8")
    # Compact single-line JSON per span (true JSON Lines) — the SDK's own
    # default formatter pretty-prints with indent=4, which is not JSONL.
    exporter = ConsoleSpanExporter(out=out, formatter=lambda readable_span: readable_span.to_json(indent=None) + "\n")
    resource = Resource.create({"openinference.project.name": project_name, "service.name": project_name})
    provider = SDKTracerProvider(resource=resource)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    _instrument_llm_clients(provider)
    tracer = provider.get_tracer(project_name)
    return TracingState(
        enabled=True, mode="file", project_name=project_name, tracer=tracer, detail=str(path)
    )


def _instrument_llm_clients(tracer_provider: Any) -> None:
    """Vendor zero-code auto-instrumentation for the two SDKs `llm.py`
    calls (`openai.OpenAI`, `google.genai.Client`) — patches the SDK client
    classes, not `llm.py` itself. Idempotent (checked via each
    instrumentor's own `is_instrumented_by_opentelemetry`), so calling
    `init_tracing()` more than once never double-instruments."""
    try:
        from openinference.instrumentation.openai import OpenAIInstrumentor

        instrumentor = OpenAIInstrumentor()
        if not instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.instrument(tracer_provider=tracer_provider)
    except Exception as exc:  # noqa: BLE001
        _warn_once("openai-instrument-failed", f"could not auto-instrument the OpenAI client ({exc!r}); llm.complete spans still work, provider-level detail will be missing.")

    try:
        from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor

        instrumentor = GoogleGenAIInstrumentor()
        if not instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.instrument(tracer_provider=tracer_provider)
    except Exception as exc:  # noqa: BLE001
        _warn_once("google-genai-instrument-failed", f"could not auto-instrument the Google GenAI client ({exc!r}); llm.complete/embed spans still work, provider-level detail will be missing.")

    instrument_llm_module()


def shutdown_tracing() -> None:
    """Flushes/closes the active exporter (if any) and resets to disabled.
    Test hygiene + a clean `--demo` exit; never raises."""
    global _state
    with _state_lock:
        tracer = _state.tracer
        if tracer is not None:
            try:
                provider = getattr(tracer, "_tracer_provider", None) or _trace_get_tracer_provider_safe()
                if provider is not None and hasattr(provider, "shutdown"):
                    provider.shutdown()
            except Exception as exc:  # noqa: BLE001
                _warn_once("shutdown-failed", f"tracer shutdown raised ({exc!r}); ignored.")
        _state = TracingState()


def _trace_get_tracer_provider_safe() -> Any | None:
    try:
        from opentelemetry import trace as trace_api

        return trace_api.get_tracer_provider()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Attribute coercion — OTel attributes must be a bool/int/float/str or a
# homogeneous sequence of one of those; anything else (dict, None, a
# dataclass, ...) is coerced to a JSON string rather than silently dropped
# or raising.
# ---------------------------------------------------------------------------


def _coerce_attr(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, tuple)) and all(isinstance(v, (str, bool, int, float)) for v in value):
        return list(value)
    try:
        return json.dumps(value, default=str)
    except Exception:
        return str(value)


# ---------------------------------------------------------------------------
# Span wrappers — one uniform interface (`set_attribute`/`set_attributes`/
# `record_exception`) regardless of whether tracing is enabled, so caller
# code never has to branch on it.
# ---------------------------------------------------------------------------


class _NoOpSpan:
    __slots__ = ()

    def set_attribute(self, key: str, value: Any) -> "_NoOpSpan":
        return self

    def set_attributes(self, attrs: Mapping[str, Any]) -> "_NoOpSpan":
        return self

    def record_exception(self, exc: BaseException) -> "_NoOpSpan":
        return self


_NOOP_SPAN = _NoOpSpan()


class _RealSpan:
    """Wraps a real OTel span. Every method swallows its own exceptions —
    a broken exporter must never propagate into caller code (module
    docstring's hard requirement)."""

    __slots__ = ("_otel_span",)

    def __init__(self, otel_span: Any) -> None:
        self._otel_span = otel_span

    def set_attribute(self, key: str, value: Any) -> "_RealSpan":
        try:
            self._otel_span.set_attribute(key, _coerce_attr(value))
        except Exception:  # noqa: BLE001
            pass
        return self

    def set_attributes(self, attrs: Mapping[str, Any]) -> "_RealSpan":
        for key, value in attrs.items():
            self.set_attribute(key, value)
        return self

    def record_exception(self, exc: BaseException) -> "_RealSpan":
        try:
            self._otel_span.record_exception(exc)
        except Exception:  # noqa: BLE001
            pass
        return self


@contextmanager
def span(name: str, *, kind: str = "CHAIN", **attributes: Any) -> Iterator[Any]:
    """Context manager for a single pipeline stage. A genuine no-op
    (`yield`s `_NOOP_SPAN`, starts nothing, imports nothing new) when
    tracing is disabled. When enabled: starts a real OTel span named
    `name`, tags it `openinference.span.kind=kind` plus every `attributes`
    kwarg, and always closes it — a caller's own exception inside the
    `with` block is recorded on the span and re-raised unchanged; a
    tracing-internal failure (span could not start, or the exporter raised
    on flush) is caught, warned once, and never re-raised.
    """
    state = _state
    if not state.enabled or state.tracer is None:
        yield _NOOP_SPAN
        return

    try:
        raw_cm = state.tracer.start_as_current_span(name)
        otel_span = raw_cm.__enter__()
    except Exception as exc:  # noqa: BLE001
        _warn_once(f"span-start-{name}", f"could not start span {name!r} ({exc!r}); this call proceeds untraced.")
        yield _NOOP_SPAN
        return

    wrapped = _RealSpan(otel_span)
    wrapped.set_attribute(_ATTR_SPAN_KIND, kind.upper())
    wrapped.set_attributes(attributes)

    t0 = time.monotonic()
    exc_info: tuple[Any, Any, Any] = (None, None, None)
    try:
        yield wrapped
    except BaseException:
        exc_info = sys.exc_info()
        raise
    finally:
        wrapped.set_attribute("latency_ms", (time.monotonic() - t0) * 1000.0)
        try:
            raw_cm.__exit__(*exc_info)
        except Exception as exc:  # noqa: BLE001
            _warn_once(f"span-exit-{name}", f"exporter failed while closing span {name!r} ({exc!r}); trace may be incomplete, run continues.")


F = TypeVar("F", bound=Callable[..., Any])


def traced(name: str | None = None, *, kind: str = "CHAIN", **default_attrs: Any) -> Callable[[F], F]:
    """Decorator form of `span(...)` for a whole function. Preserves the
    wrapped function's signature (`functools.wraps` sets `__wrapped__`,
    which `inspect.signature` follows automatically) and its return value
    unchanged — the wrapper adds exactly one `with span(...):` around the
    call, nothing else. `name` defaults to the function's own
    `__qualname__` when omitted."""

    def decorator(func: F) -> F:
        span_name = name or func.__qualname__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with span(span_name, kind=kind, **default_attrs):
                return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


# ---------------------------------------------------------------------------
# llm.py instrumentation — wrap, never restructure (see module docstring).
# ---------------------------------------------------------------------------

_TRACED_MARKER = "_medmemgraph_traced"


def instrument_llm_module(module: Any | None = None) -> None:
    """Monkeypatches `llm.py`'s two public entry points (`complete`,
    `embed`) with a span-emitting wrapper, without changing a single line
    of `llm.py` itself. Idempotent (checks `_TRACED_MARKER`) — safe to call
    more than once, and safe to call regardless of `_state.enabled`: the
    wrapper always delegates its actual span to `span()`, which is itself
    the single source of truth for enabled/disabled (a genuine no-op when
    disabled), so wrapping never has to be "undone" if tracing later turns
    off. (This function is only ever invoked from the enabled code path in
    `_init_enabled` -> `_instrument_llm_clients`, but does not depend on
    that — checking `_state.enabled` here would race against `init_tracing`
    assigning the new state, which is exactly the bug this comment is
    warning a future editor away from re-introducing.)"""
    if module is None:
        from medmemgraph import llm as module  # local import: only touched when enabled

    _wrap_complete(module)
    _wrap_embed(module)


def _wrap_complete(module: Any) -> None:
    original = getattr(module, "complete", None)
    if original is None or getattr(original, _TRACED_MARKER, False):
        return

    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        requested_model = kwargs.get("model") or "default"
        with span("llm.complete", kind="LLM", model=requested_model) as sp:
            result = original(*args, **kwargs)
            sp.set_attributes(
                {
                    "llm.model_name": getattr(result, "model", requested_model),
                    "llm.token_count.prompt": getattr(result, "prompt_tokens", None),
                    "llm.token_count.completion": getattr(result, "completion_tokens", None),
                    "cost_usd": getattr(result, "cost_usd", None),
                    "cached": getattr(result, "cached", None),
                    "attempts": getattr(result, "attempts", None),
                }
            )
            return result

    setattr(wrapper, _TRACED_MARKER, True)
    module.complete = wrapper


def _wrap_embed(module: Any) -> None:
    original = getattr(module, "embed", None)
    if original is None or getattr(original, _TRACED_MARKER, False):
        return

    @functools.wraps(original)
    def wrapper(texts: list[str], *args: Any, **kwargs: Any) -> Any:
        requested_model = kwargs.get("model") or "default"
        with span("llm.embed", kind="EMBEDDING", model=requested_model, text_count=len(texts)) as sp:
            result = original(texts, *args, **kwargs)
            sp.set_attribute("vector_count", len(result) if result is not None else 0)
            return result

    setattr(wrapper, _TRACED_MARKER, True)
    module.embed = wrapper


# ---------------------------------------------------------------------------
# `--demo` — one-command, offline-safe, end-to-end trace of a single
# question through the whole named pipeline shape.
# ---------------------------------------------------------------------------


def _run_demo(question: str, patient_id: str) -> None:
    state = init_tracing(project_name="medmemgraph-demo")

    from medmemgraph import llm
    from medmemgraph.graph.retrieve import retrieve

    with span("read", kind="CHAIN", patient_id=patient_id, question=question) as read_sp:
        # The real retrieve() — already instrumented in graph/retrieve.py —
        # produces the retrieve/route/embed/search/traverse/rerank/fuse
        # sub-spans on its own. `retrieve()` never raises (design decision
        # 5 in that module), so this is safe even with no HydraDB engine
        # reachable and no ingested corpus for `patient_id`: it degrades to
        # an honest, empty, still-fully-traced result.
        result = retrieve(question, patient_id, k=5)
        response = llm.complete(
            f"Using only this evidence, answer: {question}\n\nEvidence: "
            f"{[item.text for item in result.items]}",
            model=llm.ANSWER_MODEL,
            dry_run=True,
        )
        read_sp.set_attributes(
            {
                "route": result.route,
                "structural_absence": result.structural_absence,
                "candidate_count": len(result.items),
                "answer_preview": response.text[:120],
            }
        )

    shutdown_tracing()

    print(f"\nDemo trace emitted for patient_id={patient_id!r}, mode={state.mode}.")
    if state.mode == "arize":
        print(f"View it in Arize AX: https://app.arize.com/ (traces sent to {ARIZE_OTLP_ENDPOINT})")
    elif state.mode == "local-phoenix":
        print(f"View it at your local Phoenix UI: {LOCAL_PHOENIX_ENDPOINT}")
    elif state.mode == "file":
        print(f"Traces written as JSON lines to: {state.detail}")
    else:
        print(
            "Tracing did not activate (mode="
            f"{state.mode!r}). Set {ENV_TRACING_FLAG}=1 and re-run to record a trace."
        )


def _main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m medmemgraph.observability",
        description="Emit one complete, offline-safe trace of a single question through the retrieval pipeline.",
    )
    parser.add_argument("--demo", action="store_true", help="run the one-command demo trace")
    parser.add_argument("--question", default="Has the patient's metformin dose changed recently?")
    parser.add_argument("--patient-id", default="demo-patient-000")
    args = parser.parse_args(argv)

    if not args.demo:
        parser.print_help()
        return 1

    os.environ.setdefault(ENV_TRACING_FLAG, "1")
    _run_demo(args.question, args.patient_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
