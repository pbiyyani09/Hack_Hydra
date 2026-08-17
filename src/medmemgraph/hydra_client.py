"""hydra_client.py — Graph-owned, dialect-safe client for HydraDB OSS 0.1.1.

Every query in this project goes through :func:`HydraClient.run` (flat
scalar-dict rows) or its raw-path sibling :func:`HydraClient.run_paths`
(added E4-S6/traverse.py, for ``algo.*paths`` ``YIELD path`` results only —
see "5. Raw path hydration" below for why a second method exists at all).
Both call :func:`validate_dialect` before anything reaches the wire; there is
no ungated entry point — see ``ARCHITECTURE.md`` §8 and ``decisions/003``.

Two independent, board-frozen facts drive this file's shape:

1. **The Bolt driver incompatibility** (``literature/10`` §F0, executed).
   HydraDB's Bolt server identifies itself as ``SlateDBGraph/0.1.0``. Every
   official Neo4j Python driver >=5.1 hard-rejects any server agent that does
   not start with ``Neo4j/``, raising ``UnsupportedServerProduct`` at connect
   time. The README's "Neo4j-compatible Bolt connectivity" claim is wrong as
   shipped. The fix is a two-line client-side monkeypatch, applied below
   *before* any driver or session is created. The patch target moved between
   neo4j 5.x and 6.x, hence the exact pin (``neo4j==5.28.2`` in
   ``pyproject.toml``) — do not bump the driver without re-verifying the
   patch target.

2. **The Cypher dialect is a narrow subset** (``literature/10`` §A,
   ``decisions/003``). ``run()`` rejects the illegal shapes *before* sending
   anything to the engine, so a bad query fails loudly and locally with a
   message naming the rule and the fix, instead of surfacing days later as a
   silently wrong number (an unlabelled node pattern is the worst case:
   ``decisions/003`` — it returns a phantom row for ANY id and defeats
   ``structural_absence`` abstention on 33.3% of the primary benchmark).

Auto-commit only. ``session.run()`` is the only Bolt call path that works —
``driver.execute_query()``, ``session.execute_read()``/``execute_write()``,
and ``session.begin_transaction()`` all open an explicit transaction, and
HydraDB rejects ``BEGIN``/``COMMIT``/``ROLLBACK`` at the protocol layer
(no transactions at all — ``literature/10`` §A2/§D4). Do not use them here.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Mapping

import requests

try:
    from dotenv import load_dotenv

    load_dotenv()  # picks up a local, gitignored .env if present; no-op otherwise
except ImportError:  # pragma: no cover - python-dotenv is a hard dependency, but be defensive
    pass


# --------------------------------------------------------------------------
# 1. The Bolt driver patch — applied at import time, BEFORE any connect.
#
#    HydraDB server agent: "SlateDBGraph/0.1.0". neo4j==5.28.2's own check
#    (neo4j/_sync/io/_bolt5.py, neo4j/_async/io/_bolt5.py — each module
#    imports `check_supported_server_product` by name from `_common.py` and
#    calls it as a bare global lookup within its own module namespace) is
#    neutered by rebinding that name in the *importing* module, which is
#    exactly where the call site resolves it. Patching `_common.py` instead
#    would not intercept the call, since `_bolt5.py` already bound its own
#    reference at import time.
# --------------------------------------------------------------------------
def _apply_bolt_driver_patch() -> None:
    import neo4j._async.io._bolt5 as _ab5
    import neo4j._sync.io._bolt5 as _b5

    _b5.check_supported_server_product = lambda agent: None
    _ab5.check_supported_server_product = lambda agent: None


_apply_bolt_driver_patch()

from neo4j import GraphDatabase  # noqa: E402  (import after the patch, deliberately)


# --------------------------------------------------------------------------
# 2. Dialect gate — decision 003 + literature/10 §A, enforced client-side.
# --------------------------------------------------------------------------
class DialectError(ValueError):
    """Raised by :func:`validate_dialect` / :func:`HydraClient.run` when a
    Cypher string violates HydraDB's dialect. The message always names the
    rule that fired and the fix — see ``ARCHITECTURE.md`` §8.3."""


_STRING_LITERAL_RE = re.compile(r"'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\"", re.DOTALL)


def _strip_string_literals(cypher: str) -> str:
    """Blank out the contents of every quoted string literal (same length,
    non-keyword filler), so the keyword-matching checks below never
    false-positive on a literal that merely *contains* a reserved word (e.g.
    a note text "...case closed..." must not trip the CASE rule) — and so a
    structural character (a stray ``(`` inside a string) can never be
    mistaken for query structure either. Must run before every other check.
    """
    return _STRING_LITERAL_RE.sub(lambda m: "\0" * len(m.group(0)), cypher)


def _split_top_level(s: str, sep: str = ",") -> list[str]:
    """Split on `sep` at bracket-depth 0 only, so `count(*)` or a nested
    `{...}` map does not get split internally."""
    parts: list[str] = []
    depth = 0
    buf: list[str] = []
    for ch in s:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == sep and depth == 0:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    parts.append("".join(buf))
    return parts


def _check_multi_statement(stripped: str, original: str) -> str | None:
    if ";" in stripped:
        return (
            "Multi-statement request (`;`-separated): HydraDB parses and plans "
            "exactly one statement per request (literature/10 §A0/§A2). "
            "Fix: issue one run() call per statement."
        )
    return None


def _check_transactions(stripped: str, original: str) -> str | None:
    m = re.search(r"\b(BEGIN|COMMIT|ROLLBACK)\b", stripped, re.IGNORECASE)
    if m:
        return (
            f"`{m.group(1).upper()}` is rejected: HydraDB has no explicit "
            "transactions, auto-commit only (literature/10 §A2/§D4, "
            "ARCHITECTURE.md §8.1). Fix: use plain session.run() (auto-commit) "
            "via this client; never execute_query/execute_read/execute_write/"
            "begin_transaction."
        )
    return None


def _check_in(stripped: str, original: str) -> str | None:
    if re.search(r"\bIN\b", stripped, re.IGNORECASE):
        return (
            "`IN` is not implemented (WHERE supports only =, <>, <, >, <=, >=, "
            "STARTS WITH — literature/10 §A3). Fix: replace `WHERE x.prop IN "
            "$list` with an `UNWIND $list AS v ...` batch, or seed "
            "`algo.MSpaths(..., sourceValues: $ids, ...)` directly with the ids."
        )
    return None


def _check_contains(stripped: str, original: str) -> str | None:
    if re.search(r"\bCONTAINS\b", stripped, re.IGNORECASE):
        return (
            "`CONTAINS` is rejected (only `STARTS WITH` is a legal string "
            "predicate — literature/10 §A3). Fix: use `STARTS WITH` for a "
            "prefix match, or filter client-side after a broader labelled MATCH."
        )
    if re.search(r"\bENDS\s+WITH\b", stripped, re.IGNORECASE):
        return (
            "`ENDS WITH` is rejected (only `STARTS WITH` is legal — "
            "literature/10 §A3). Fix: filter client-side after a broader "
            "labelled MATCH."
        )
    return None


def _check_is_null(stripped: str, original: str) -> str | None:
    if re.search(r"\bIS\s+(NOT\s+)?NULL\b", stripped, re.IGNORECASE):
        return (
            "`IS NULL`/`IS NOT NULL` is rejected — there is no NULL test "
            "(literature/10 §A3/§C4). Fix: use the sentinel convention "
            '(`valid_to = "9999-12-31T00:00:00"`) and compare with an explicit '
            "`=`/`<>`/range operator instead of a NULL test."
        )
    return None


def _check_case(stripped: str, original: str) -> str | None:
    if re.search(r"\bCASE\b", stripped, re.IGNORECASE):
        return (
            "`CASE` is not in HydraDB's implemented AST (literature/10 "
            "§A1/§A3). Fix: branch client-side on the returned rows, or issue "
            "one query per branch."
        )
    return None


def _check_min_max(stripped: str, original: str) -> str | None:
    # Require the call-syntax `(` right after the name, not just the substring
    # — otherwise identifiers like `maxLen` or `pathCost` would false-positive.
    if re.search(r"\bmin\s*\(", stripped, re.IGNORECASE):
        return (
            "`min(...)` is not an implemented aggregate (only count/sum/avg/"
            "collect — literature/10 §A3). Fix: `ORDER BY <prop> LIMIT 1`, or "
            "resolve client-side."
        )
    if re.search(r"\bmax\s*\(", stripped, re.IGNORECASE):
        return (
            "`max(...)` is not an implemented aggregate (only count/sum/avg/"
            "collect — literature/10 §A3). Fix: `ORDER BY <prop> DESC LIMIT 1`, "
            "or resolve client-side."
        )
    return None


def _check_inline_unwind(stripped: str, original: str) -> str | None:
    if re.search(r"\bUNWIND\s*\[", stripped, re.IGNORECASE):
        return (
            "Inline `UNWIND [...]` list literal: the UNWIND batch input must "
            "be a parameter (literature/10 §A2/§D2, decisions/003). "
            "Fix: pass rows as `$rows` and write `UNWIND $rows AS row ...`."
        )
    return None


_REL_RE = re.compile(r"(<?)-\[[^\]]*\]-(>?)")


def _check_undirected(stripped: str, original: str) -> str | None:
    for m in _REL_RE.finditer(stripped):
        if not m.group(1) and not m.group(2):
            return (
                "Undirected relationship pattern `-[...]-`: every relationship "
                "pattern must be directed (literature/10 §A3, decisions/003). "
                "Fix: pick a direction, `-[...]->` or `<-[...]-`."
            )
    return None


def _check_unbounded_varlen(stripped: str, original: str) -> str | None:
    for m in re.finditer(r"\[[^\]]*\]", stripped):
        content = m.group(0)
        if "*" in content and not re.search(r"\*\s*\d*\s*\.\.\s*\d+", content):
            return (
                "Unbounded variable-length relationship (`*` or `*min..` with "
                "no explicit max): the hop ceiling is mandatory in the query "
                "text and the hard limit is 16 (literature/10 §A4). "
                "Fix: write an explicit bound, e.g. `*1..16` — or, on any "
                "correctness-critical path, prefer `algo.MSpaths`/`algo.SPpaths` "
                "instead of a raw variable-length MATCH (ARCHITECTURE.md §8.5; "
                "bugs #69/#71/#83 give silent wrong answers on cyclic graphs)."
            )
    return None


# A node pattern's `(` is never immediately preceded (no whitespace) by an
# identifier character — that shape is a function/procedure call, e.g.
# `algo.MSpaths(...)` or `count(...)`, not a node pattern, and is excluded by
# this negative lookbehind rather than mis-flagged as an unlabelled node.
_NODE_PATTERN_RE = re.compile(r"(?<![A-Za-z0-9_])\(([^()]*)\)")
_LABELLED_VAR_RE = re.compile(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*:")

# --------------------------------------------------------------------------
# Vertex-upsert idiom exception — executed-verified live against
# ghcr.io/hydra-db/hydradb:0.1.1, 2026-08-16 (dev-backend, E4-S4 spike;
# see tests/test_writer.py header for the pinned reproduction).
#
# ARCHITECTURE.md §6.5's documented vertex-upsert shape,
# `MERGE (n:Claim {id: row.vertex}) SET n.prop = row.prop`, is REJECTED by
# the live engine: `CypherSyntaxError('... UNWIND vertex upsert MERGE
# pattern matches only id; apply labels with SET')`. The engine's OWN
# required shape is `MERGE (n {id: row.vertex}) SET n:Claim, n.prop = ...`
# — the label goes on the SET, not in the MERGE pattern
# (`'... UNWIND vertex upsert requires exactly one SET label'` confirms
# this is a distinct, special-cased grammar production, not general MERGE).
#
# This is the one node-pattern shape this gate exempts from decision 003's
# "every node pattern is labelled" rule, and only this shape: the MERGE
# operand's braces contain *exactly* `id: <expr>` (nothing else — matching
# the engine's own "matches only id" restriction) and the very next clause
# is `SET <same var>:<Label>` (a label assignment, not a property set).
#
# This is safe, not a loophole reopening decisions/003's phantom-row bug:
# that bug is specific to the MATCH/read code path (an unlabelled
# `MATCH (n {id: X})` returns a row for ANY integer, existing or not).
# MERGE's unlabelled-by-id lookup is a different, engine-special-cased
# "vertex upsert" path — spiked live both for correctness-on-create and
# for idempotency (same row merged twice produces exactly one node,
# confirmed via a labelled MATCH count both times). MATCH patterns are
# completely unaffected by this exception: `MATCH (n {id: $id}) SET n:X`
# still raises DialectError, verified in tests/test_writer.py.
# --------------------------------------------------------------------------
_VERTEX_UPSERT_MERGE_RE = re.compile(
    r"\bMERGE\s*(\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\{\s*id\s*:\s*[^{},]+\}\s*\))\s*"
    r"SET\s+([A-Za-z_][A-Za-z0-9_]*)\s*:\s*[A-Za-z_][A-Za-z0-9_]*\b",
    re.IGNORECASE,
)


def _vertex_upsert_safe_spans(stripped: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for m in _VERTEX_UPSERT_MERGE_RE.finditer(stripped):
        merge_var, set_var = m.group(2), m.group(3)
        if merge_var == set_var:
            spans.append(m.span(1))
    return spans


def _check_unlabelled_node(stripped: str, original: str) -> str | None:
    safe_spans = _vertex_upsert_safe_spans(stripped)
    occurrences = [
        m
        for m in _NODE_PATTERN_RE.finditer(stripped)
        if not any(s <= m.start() and m.end() <= e for s, e in safe_spans)
    ]

    # A later bare `(x)` that re-references a variable already bound with a
    # label earlier in the SAME statement is legal Cypher, not a fresh
    # by-id lookup — e.g. the documented relationship-upsert shape
    # (ARCHITECTURE.md §6.5): `MATCH (s:Patient {...}), (d:Claim {...})
    # MERGE (s)-[r:HAS {...}]->(d)`. Collect every variable that got a
    # label anywhere in this statement first, then only flag a bare `(x)`
    # when `x` was never labelled at all.
    labelled_vars: set[str] = set()
    for m in occurrences:
        before_brace = m.group(1).split("{", 1)[0]
        var_match = _LABELLED_VAR_RE.match(before_brace)
        if var_match:
            labelled_vars.add(var_match.group(1))

    for m in occurrences:
        content = m.group(1)
        before_brace = content.split("{", 1)[0]  # a label's `:` is never inside `{...}` props
        if ":" in before_brace:
            continue  # this occurrence carries its own label
        bare_var = content.strip()
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", bare_var) and bare_var in labelled_vars:
            continue  # legal: bare reference to a variable labelled elsewhere in this statement
        return (
            "Unlabelled node pattern `(...)`: every node pattern must "
            "carry a label, or be a bare reference to a variable already "
            "labelled earlier in the same statement (decisions/003 — an "
            "unlabelled `MATCH (n {id: X})` is vertex-id *resolution*, not "
            "existence; it returns a phantom row for ANY integer id and "
            "silently defeats structural_absence abstention). "
            "Fix: add a label, e.g. `(n:Patient {id: X})`."
        )
    return None


_CLAUSE_BOUNDARY_RE = re.compile(
    r"\b(?:RETURN|ORDER\s+BY|LIMIT|SKIP|UNION(?:\s+ALL)?|YIELD)\b", re.IGNORECASE
)
_ALLOWED_BARE_YIELD_COLUMNS = {"path", "pathweight", "pathcost"}


def _check_bare_return(stripped: str, original: str) -> str | None:
    boundaries = list(_CLAUSE_BOUNDARY_RE.finditer(stripped))
    has_yield = any(b.group(0).upper() == "YIELD" for b in boundaries)
    for i, b in enumerate(boundaries):
        if b.group(0).upper() != "RETURN":
            continue
        start = b.end()
        end = boundaries[i + 1].start() if i + 1 < len(boundaries) else len(stripped)
        projection = stripped[start:end]
        for raw_item in _split_top_level(projection):
            item = raw_item.strip()
            if not item:
                continue
            if item == "*":
                return (
                    "`RETURN *` is a parse error: projections must be "
                    "`<binding>.<property>` or an aggregate (literature/10 "
                    "§A3). Fix: name explicit `binding.property` columns."
                )
            expr = re.split(r"\bAS\b", item, maxsplit=1, flags=re.IGNORECASE)[0].strip()
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", expr):
                if expr.lower() in _ALLOWED_BARE_YIELD_COLUMNS and has_yield:
                    # Legal exception: `CALL algo.*paths(...) YIELD path RETURN
                    # path` — the only bare bindings the engine accepts, because
                    # `path`/`pathWeight`/`pathCost` carry a wire-safe Path
                    # value, unlike a bare node/relationship binding
                    # (literature/10 §A5/§F1; ARCHITECTURE.md §4 "never a bare
                    # node").
                    continue
                return (
                    f"`RETURN {expr}` projects a bare binding: no Node/"
                    "Relationship value ever crosses the wire, and an "
                    "unqualified binding is also decision 003's phantom-node "
                    "lure (literature/10 §A3/§F1). Fix: project "
                    f"`{expr}.<property>` explicitly — or, if `{expr}` is a "
                    "`YIELD`-bound path column, it must be exactly `path`, "
                    "`pathWeight`, or `pathCost` from an `algo.*paths` CALL."
                )
    return None


_DIALECT_CHECKS = (
    _check_multi_statement,
    _check_transactions,
    _check_in,
    _check_contains,
    _check_is_null,
    _check_case,
    _check_min_max,
    _check_inline_unwind,
    _check_undirected,
    _check_unbounded_varlen,
    _check_unlabelled_node,
    _check_bare_return,
)


def validate_dialect(cypher: str) -> None:
    """Raise :class:`DialectError` if `cypher` violates HydraDB's Cypher
    dialect. Called by :meth:`HydraClient.run` before every query, on both
    transports, so a bad shape never reaches the wire."""
    stripped = _strip_string_literals(cypher)
    for check in _DIALECT_CHECKS:
        message = check(stripped, cypher)
        if message is not None:
            raise DialectError(message)


# --------------------------------------------------------------------------
# 3. HydraQueryError — engine-side failures, distinct from a dialect-gate
#    rejection so callers can tell "your query text is illegal" apart from
#    "the engine ran it and it failed" (e.g. 421 not_cell_writer, timeout).
# --------------------------------------------------------------------------
class HydraQueryError(RuntimeError):
    """Raised when the HydraDB engine itself rejects or fails a query that
    passed the local dialect gate (network error, non-200 HTTP response,
    Bolt driver exception, etc.)."""


def _unwrap_http_value(tagged: Any) -> Any:
    """Unwrap one HTTP query-response value out of its tagged-union envelope
    (literature/10 §F2): ``{"type": ..., "value": ...}``.

    ``vertex_id`` / ``integer`` / ``signed_integer`` / ``float`` / ``boolean``
    / ``string`` collapse to the plain Python value; ``list`` recurses;
    ``null`` becomes ``None``. ``path`` is passed through as the raw
    ``{"type": "path", "value": {...}}`` envelope unchanged — this function
    is only ever called on non-path columns by :meth:`HydraClient.run`
    (callers with a ``path``/``pathWeight``/``pathCost`` YIELD use
    :meth:`HydraClient.run_paths` instead, see "5. Raw path hydration"
    below, which spiked and now documents the exact ``path`` JSON shape
    literature/10 §F2 left unspecified).
    """
    if not isinstance(tagged, dict) or "type" not in tagged:
        return tagged
    t = tagged.get("type")
    v = tagged.get("value")
    if t == "null":
        return None
    if t == "list":
        return [_unwrap_http_value(item) for item in (v or [])]
    # vertex_id / integer / signed_integer / float / boolean / string all
    # collapse to `v` unchanged; `path` also passes `v` through raw here
    # (its structure is handled by `_hydrate_http_path`, not this function).
    return v


# --------------------------------------------------------------------------
# 5. Raw path hydration — `algo.*paths` `YIELD path`/`pathWeight`/`pathCost`.
#
# Added E4-S6 (`graph/traverse.py`'s story) after spiking, live against
# ghcr.io/hydra-db/hydradb:0.1.1, 2026-08-16, why `HydraClient.run()`'s
# existing `record.data()` (Bolt) / `_unwrap_http_value` (HTTP) machinery
# cannot carry a `path` column's node labels or our own minted node `id`s —
# both of which `graph/traverse.py`'s story requires ("the :Claim ids on
# the path so provenance survives to the demo").
#
# EXECUTED FINDING 1 (Bolt): `neo4j._data.RecordExporter` (what
# `Record.data()` calls) collapses a `Node` to `dict(node)` — properties
# only, labels dropped — and a `Relationship` to just its type name string
# — properties (including our own minted `id`) dropped too. Confirmed by
# printing the RAW `neo4j.graph.Path`/`Node`/`Relationship` objects
# `session.run()` (bypassing `.data()`) actually returns: `Node.labels` IS
# populated (`frozenset({'Patient'})` etc.) and `Relationship.items()` DOES
# still carry our minted `id` property — `.data()` throws both away, not
# the driver.
#
# EXECUTED FINDING 2 (Bolt, load-bearing, corrects ARCHITECTURE.md §7.3's
# own worked guidance "`sourceProperty`: `'id'` when seeds are minted
# integers"): our own `id` property is NOT present in a Bolt-hydrated
# `Node`'s property map at all — `dict(node)` never contains an `id` key,
# for ANY node in this schema. Instead, the driver's `Node.element_id`
# (legacy `Node.id`) numerically EQUALS our minted `id` value exactly
# (verified: a node created with `{id: 981995962, ...}` comes back with
# `element_id='981995962'`). The engine folds our `id` property into the
# vertex's native Bolt identity rather than storing it in the queryable
# property bag — consistent with decisions/003's "MERGE matches on id
# only" (`id` IS the vertex key here, not an ordinary property) and with
# `algo.MSpaths`'s own `sourceProperty: 'id'` + `sourceValues: ['<id>']`
# (even string-typed, per Finding 3) returning zero rows for a freshly
# created, directly-connected, sanity-checked pair — there is no `id`
# property for that selector to match against. `graph/traverse.py` seeds
# `algo.MSpaths` by a genuine stored string property per label instead
# (`name` for domain entities, `fact_id` for Claim, etc.) and recovers the
# caller's real minted `id` back out of the returned `Path` via
# `GraphNode.id` below (`Node.element_id`), not via any selector property —
# see that module's own docstring for the full design and the client-side
# allow-list check this implies (a `name` collision across two different
# patients' domain-entity nodes must never leak a cross-patient path).
#
# EXECUTED FINDING 3: `algo.MSpaths`'s `sourceValues`/`targetValues` must be
# a list of Cypher STRING literals syntactically — `sourceValues: [123]`
# (bare integers) is rejected at parse time: `"sourceValues must be a list
# of strings"`. `maxLen` > 16 is REJECTED live (`TransientError:
# "native_path_max_len rejected by admission control: actual 20 exceeds
# limit 16"`), not silently clamped by the engine — confirming
# ARCHITECTURE.md §7.3's ceiling is a hard requirement the CALLER must
# clamp to before sending, exactly as this story instructs.
#
# EXECUTED FINDING 4 (HTTP): unlike Bolt, the HTTP transport's `path` JSON
# value IS fully self-describing and includes everything Bolt's `.data()`
# drops — `{"nodes": [{"id": int, "labels": [str], "properties": {tagged
# scalars}}, ...], "relationships": [{"id": <wire int>, "edge_type": str,
# "src": int, "dst": int, "properties": {"id": {"type": "integer", "value":
# <our minted rel id>}, ...}}, ...]}`. Node `id` here IS the real minted
# id (unlike Bolt's dropped-from-properties case) and `labels` is populated
# directly — no `element_id` indirection needed. This shape was previously
# undocumented in literature/10 §F2 (`_unwrap_http_value`'s docstring said
# so); now recorded here from one live, single-path spike. Confidence:
# high for node/relationship field shapes (self-describing, unambiguous);
# medium for whether `nodes`/`relationships` array ORDER always matches
# walk order on more complex (branching-free, but longer or `both`-
# direction-backward) paths — only one 2-hop straight-line path was
# spiked. `_hydrate_http_path` below trusts the server's given order
# (matches the one case tested); this is flagged, not silently assumed.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GraphNode:
    """One node on a raw `algo.*paths` result, id/labels/properties all
    intact (unlike `HydraClient.run()`'s `record.data()` rows)."""

    id: int
    labels: frozenset[str]
    properties: dict[str, Any]


@dataclass(frozen=True)
class GraphRelationship:
    """One relationship on a raw `algo.*paths` result. `id` is OUR minted
    relationship id (the `id` property every writer.py/invalidate.py edge
    carries), not the transport's own internal element id — recovered from
    the relationship's property bag (Bolt) or its tagged `properties.id`
    (HTTP), falling back to the transport's own element id only if no `id`
    property was ever set (should not happen for any edge this project
    writes, but this is a decode step, not a write-path assumption, so it
    degrades rather than raises)."""

    id: int
    type: str
    start_id: int
    end_id: int
    properties: dict[str, Any]


@dataclass(frozen=True)
class GraphPath:
    """One raw path off an `algo.*paths` YIELD, in walk order:
    `nodes[i]` and `nodes[i+1]` are connected by `relationships[i]`
    (so `len(nodes) == len(relationships) + 1`). `relationships[i]`'s own
    `start_id`/`end_id` reflect the edge's true stored direction, which can
    differ from the walk order when `relDirection: 'both'` traverses a hop
    backward — callers that care about arrow direction (e.g. rendering)
    should compare `relationships[i].start_id` against `nodes[i].id`."""

    nodes: tuple[GraphNode, ...]
    relationships: tuple[GraphRelationship, ...]


def _hydrate_bolt_path(path: Any) -> GraphPath:
    """`neo4j.graph.Path` -> `GraphPath`, recovering what `Record.data()`
    drops (Finding 1/2 above): node labels from `Node.labels`, node id from
    `Node.element_id` (our minted `id`, folded into Bolt identity — never in
    the property map), relationship id from its own `id` property (still
    present on a raw `Relationship`, unlike on a raw `Node`)."""
    nodes = tuple(
        GraphNode(id=int(n.element_id), labels=frozenset(n.labels), properties=dict(n))
        for n in path.nodes
    )
    relationships = []
    for r in path.relationships:
        props = dict(r)
        minted_id = props.pop("id", None)
        rel_id = int(minted_id) if minted_id is not None else int(r.element_id)
        start_id = int(r.start_node.element_id) if r.start_node is not None else -1
        end_id = int(r.end_node.element_id) if r.end_node is not None else -1
        relationships.append(
            GraphRelationship(id=rel_id, type=r.type, start_id=start_id, end_id=end_id, properties=props)
        )
    return GraphPath(nodes=nodes, relationships=tuple(relationships))


def _unwrap_http_property_value(tagged: Any) -> Any:
    """Unwrap ONE node/relationship property value inside a `path` JSON
    payload — `{"String": "foo"}`, `{"Integer": 5}`, `{"Float": 0.9}`, a
    single-key internally-tagged variant dict.

    EXECUTED FINDING (2026-08-16, live spike, not in literature/10 — a
    SECOND, DIFFERENT tagged-union encoding from the one `_unwrap_http_value`
    handles): the outer query-result envelope uses `{"type": ..., "value":
    ...}` (literature/10 §F2, what `_unwrap_http_value` unwraps), but a
    `path` value's node/relationship `properties` maps use this other,
    single-key `{"<Variant>": <value>}` shape instead — confirmed via a raw
    HTTP response dump: `"properties": {"id": {"Integer": 973149930}}`.
    Passing this shape through `_unwrap_http_value` silently no-ops (its
    `"type" not in tagged` guard returns the tagged dict unchanged instead
    of the value), which is exactly the bug this separate function exists
    to avoid — caught live (a `TypeError` on `int({"Integer": ...})`)
    before this ever reached `graph/traverse.py`'s caller."""
    if not isinstance(tagged, dict) or len(tagged) != 1:
        return tagged
    ((_variant, unwrapped),) = tagged.items()
    return unwrapped


def _hydrate_http_path(value: Mapping[str, Any]) -> GraphPath:
    """HTTP `path` value (Finding 4 above) -> `GraphPath`. Node `id`/
    `labels` are given directly; relationship `id` is recovered from its
    own tagged `properties.id` the same way Bolt's is (our minted id, not
    the transport's internal `"id"` field, which is a small per-query wire
    counter, not our identity — mirrors `_hydrate_bolt_path`'s same
    distinction). Property VALUES are unwrapped with
    `_unwrap_http_property_value`, not `_unwrap_http_value` — see that
    function's docstring for why the two are not interchangeable here."""
    nodes = tuple(
        GraphNode(
            id=n["id"],
            labels=frozenset(n.get("labels") or ()),
            properties={
                k: _unwrap_http_property_value(v) for k, v in (n.get("properties") or {}).items()
            },
        )
        for n in value.get("nodes", [])
    )
    relationships = []
    for r in value.get("relationships", []):
        props = {
            k: _unwrap_http_property_value(v) for k, v in (r.get("properties") or {}).items()
        }
        minted_id = props.pop("id", None)
        rel_id = int(minted_id) if minted_id is not None else int(r["id"])
        relationships.append(
            GraphRelationship(id=rel_id, type=r["edge_type"], start_id=r["src"], end_id=r["dst"], properties=props)
        )
    return GraphPath(nodes=nodes, relationships=tuple(relationships))


# --------------------------------------------------------------------------
# 6. HydraClient — run() is the ONLY *scalar-row* query entry point, both
#    transports; run_paths() is its raw-path sibling for algo.*paths calls.
# --------------------------------------------------------------------------
class HydraClient:
    """Dialect-safe client for HydraDB OSS 0.1.1, Bolt (default) or HTTP.

    ``run(cypher, **params) -> list[dict]`` is the only query entry point.
    Both transports share the same dialect gate and return the same shape —
    the seam between them is a constructor argument, not a rewrite
    (ARCHITECTURE.md §8: "the seam is a 30-minute swap only if every call
    already goes through run()").
    """

    def __init__(
        self,
        transport: str = "bolt",
        *,
        bolt_uri: str | None = None,
        http_base_url: str | None = None,
        token: str | None = None,
        database: str = "default",
        namespace: str = "default",
        graph_id: str = "default",
        cell_id: str = "cell-0",
        http_timeout: float = 30.0,
    ) -> None:
        if transport not in ("bolt", "http"):
            raise ValueError(f"HydraClient: transport must be 'bolt' or 'http', got {transport!r}")
        self.transport = transport

        # Secrets from config/env only — never a literal in source. Local dev
        # sets HYDRA_AUTH_TOKEN (directly or via a gitignored .env; see
        # decisions/003 provenance / ARCHITECTURE.md §9 boot recipe for the
        # token-file requirement on the server side — this is the client-side
        # counterpart, the same shared secret).
        self._token = token or os.environ.get("HYDRA_AUTH_TOKEN") or os.environ.get("GRAPH_AUTH_TOKEN")
        if not self._token:
            raise RuntimeError(
                "HydraClient: no auth token. Set HYDRA_AUTH_TOKEN (env var or a "
                "gitignored .env — see literature/10 §E3 for the token's own "
                "≥32-char, not-'change-me' requirement on the server side)."
            )

        if transport == "bolt":
            self._bolt_uri = bolt_uri or os.environ.get("HYDRA_BOLT_URI", "bolt://127.0.0.1:7687")
            self._database = database
            self._driver = GraphDatabase.driver(self._bolt_uri, auth=("neo4j", self._token))
        else:
            self._http_base_url = (
                http_base_url or os.environ.get("HYDRA_HTTP_URL", "http://127.0.0.1:8443")
            ).rstrip("/")
            self._namespace = namespace
            self._graph_id = graph_id
            self._cell_id = cell_id
            self._http_timeout = http_timeout
            self._http_session = requests.Session()

    # -- the one query entry point -----------------------------------------
    def run(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        """Run one Cypher statement, auto-commit, and return a list of plain
        dicts. Raises :class:`DialectError` before sending anything to the
        engine if `cypher` violates the dialect gate."""
        validate_dialect(cypher)
        if self.transport == "bolt":
            return self._run_bolt(cypher, params)
        return self._run_http(cypher, params)

    def _run_bolt(self, cypher: str, params: Mapping[str, Any]) -> list[dict[str, Any]]:
        # session.run() ONLY — auto-commit. driver.execute_query() and
        # session.execute_read()/execute_write()/begin_transaction() all open
        # an explicit transaction, and HydraDB's Bolt layer FAILS any BEGIN at
        # the protocol level (no transactions at all — see module docstring
        # and literature/10 §A2/§D4). Do not "upgrade" this to execute_query.
        with self._driver.session(database=self._database) as session:
            # No `session.reset()` here: neo4j==5.28.2's sync `Session` has
            # no `reset()` method (verified via introspection, 2026-08-16 —
            # `dir(neo4j.Session)` does not list it). The previous version of
            # this method called it anyway, so every Bolt-side engine
            # failure (e.g. a CypherSyntaxError) was masked by an unrelated
            # `AttributeError: 'Session' object has no attribute 'reset'`,
            # hiding the real error from every caller. The `with` block's
            # `__exit__` already closes/releases the session on any
            # exception, which is all cleanup this auto-commit-only client
            # needs (decision 003 / literature/10 §F1) — just let the
            # original exception propagate.
            result = session.run(cypher, dict(params))
            return [record.data() for record in result]

    def _run_http(self, cypher: str, params: Mapping[str, Any]) -> list[dict[str, Any]]:
        url = f"{self._http_base_url}/v1/graphs/{self._graph_id}/query"
        headers = {
            "Authorization": f"Bearer {self._token}",
            "X-Graph-Namespace": self._namespace,
            "Content-Type": "application/json",
        }
        body: dict[str, Any] = {
            "cell_id": self._cell_id,
            "query": cypher,
            "parameters": dict(params),
        }
        rows_out: list[dict[str, Any]] = []
        columns: list[str] = []
        cursor: Any = None
        while True:
            if cursor is not None:
                body = {**body, "cursor": cursor}
            try:
                resp = self._http_session.post(
                    url, json=body, headers=headers, timeout=self._http_timeout
                )
            except requests.RequestException as exc:
                raise HydraQueryError(f"HydraDB HTTP query failed to send: {exc}") from exc
            if resp.status_code != 200:
                raise HydraQueryError(
                    f"HydraDB HTTP query failed: {resp.status_code} {resp.text}"
                )
            payload = resp.json()
            columns = payload.get("columns") or columns
            for raw_row in payload.get("rows", []):
                rows_out.append(
                    {col: _unwrap_http_value(v) for col, v in zip(columns, raw_row)}
                )
            cursor = payload.get("next_cursor")
            if not cursor:
                break
        return rows_out

    # -- the raw-path query entry point -------------------------------------
    def run_paths(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        """Run one `algo.*paths` `YIELD path`/`pathWeight`/`pathCost` Cypher
        statement and return rows whose `path` column is a `GraphPath`
        (labels, our minted node/relationship `id`s, and properties all
        intact) instead of `run()`'s `record.data()`-flattened list — see
        "5. Raw path hydration" above for why `run()`'s shape cannot carry
        this. `pathWeight`/`pathCost` columns, if yielded, pass through as
        plain ints unchanged on both transports.

        Same dialect gate as `run()` (`validate_dialect` first, always) —
        this is an additive sibling entry point, not a bypass. Only
        meaningful for `CALL algo.*paths(...) YIELD path[, pathWeight]
        [, pathCost] RETURN path[, pathWeight][, pathCost]` shapes; calling
        it on an ordinary scalar-row query works but is pointless (no
        column will be a `GraphPath`, and you pay the same round trip
        `run()` would have)."""
        validate_dialect(cypher)
        if self.transport == "bolt":
            return self._run_bolt_paths(cypher, params)
        return self._run_http_paths(cypher, params)

    def _run_bolt_paths(self, cypher: str, params: Mapping[str, Any]) -> list[dict[str, Any]]:
        from neo4j.graph import Path as _BoltPath  # local: only this method needs it

        with self._driver.session(database=self._database) as session:
            result = session.run(cypher, dict(params))
            rows: list[dict[str, Any]] = []
            for record in result:
                row: dict[str, Any] = {}
                for key, value in zip(record.keys(), record.values()):
                    row[key] = _hydrate_bolt_path(value) if isinstance(value, _BoltPath) else value
                rows.append(row)
            return rows

    def _run_http_paths(self, cypher: str, params: Mapping[str, Any]) -> list[dict[str, Any]]:
        # Deliberately NOT sharing a helper with `_run_http`: same cursor-
        # paging shape, but the per-value unwrap differs (path columns need
        # `_hydrate_http_path`, not `_unwrap_http_value`) — kept as a
        # separate, near-duplicate method so `_run_http`'s already-tested
        # body is never touched by this story.
        url = f"{self._http_base_url}/v1/graphs/{self._graph_id}/query"
        headers = {
            "Authorization": f"Bearer {self._token}",
            "X-Graph-Namespace": self._namespace,
            "Content-Type": "application/json",
        }
        body: dict[str, Any] = {
            "cell_id": self._cell_id,
            "query": cypher,
            "parameters": dict(params),
        }
        rows_out: list[dict[str, Any]] = []
        columns: list[str] = []
        cursor: Any = None
        while True:
            if cursor is not None:
                body = {**body, "cursor": cursor}
            try:
                resp = self._http_session.post(
                    url, json=body, headers=headers, timeout=self._http_timeout
                )
            except requests.RequestException as exc:
                raise HydraQueryError(f"HydraDB HTTP query failed to send: {exc}") from exc
            if resp.status_code != 200:
                raise HydraQueryError(
                    f"HydraDB HTTP query failed: {resp.status_code} {resp.text}"
                )
            payload = resp.json()
            columns = payload.get("columns") or columns
            for raw_row in payload.get("rows", []):
                row: dict[str, Any] = {}
                for col, tagged in zip(columns, raw_row):
                    if isinstance(tagged, dict) and tagged.get("type") == "path":
                        row[col] = _hydrate_http_path(tagged.get("value") or {})
                    else:
                        row[col] = _unwrap_http_value(tagged)
                rows_out.append(row)
            cursor = payload.get("next_cursor")
            if not cursor:
                break
        return rows_out

    def close(self) -> None:
        if self.transport == "bolt":
            self._driver.close()
        else:
            self._http_session.close()

    def __enter__(self) -> "HydraClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- self-tests ----------------------------------------------------------
    def round_trip(self) -> dict[str, Any]:
        """Connectivity self-test using ONLY the legal, labelled shapes from
        PHASES.md's round-trip block (survey 10, labelled per decision 003).
        Returns the rows from each step for the caller to assert on. Note:
        the CREATE step is a bare `CREATE`, not `MERGE` — exactly as
        specified — so repeated calls each add a new 'alpha'->'beta' edge;
        that is a property of the fixture, not a defect in this method.
        """
        self.run(
            "CREATE (a:Entity {id: 1, name: 'alpha'})"
            "-[:RELATES {relationship_id: 'ab'}]->"
            "(b:Entity {id: 2, name: 'beta'})"
        )
        match_rows = self.run(
            "MATCH (a:Entity {id: 1})-[:RELATES]->(b:Entity) "
            "RETURN b.id AS id, b.name AS name"
        )
        path_rows = self.run(
            "CALL algo.MSpaths({ "
            "sourceLabel: 'Entity', sourceProperty: 'name', "
            "sourceValues: ['alpha'], targetValues: ['beta'], "
            "pairwise: true, relTypes: ['RELATES'], relDirection: 'both', "
            "maxLen: 3, pathCount: 5, resultLimit: 100 "
            "}) YIELD path RETURN path"
        )
        return {"match_rows": match_rows, "path_rows": path_rows}


def healthz(admin_url: str | None = None, timeout: float = 5.0) -> bool:
    """Hit HydraDB's admin `/readyz` on :9090. True iff HTTP 200."""
    url = admin_url or os.environ.get("HYDRA_ADMIN_URL", "http://127.0.0.1:9090/readyz")
    try:
        resp = requests.get(url, timeout=timeout)
        return resp.status_code == 200
    except requests.RequestException:
        return False
