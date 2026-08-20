"""demo/agent.py — the Beat 1 / Beat 2 chat wrapper (`stories/E8/E8-S1.md`).

A stranger chats about **one** patient's history through a minimal CLI. The
whole module is exactly two calls wired together, printed legibly:

    retrieve(question, patient_id, k) -> pack        (graph/retrieve.py)
    answer_con(question, pack) -> {notes, abstain,
                                    citations, ...}   (eval/reader.py)

No new retrieval channel. No schema/fusion/predicate change. No demo chrome
beyond this REPL — the only "chrome" here is legible print formatting so a
screen recording reads cleanly (`demo/VIDEO_SCRIPT.md` Beats 1-2).

--------------------------------------------------------------------------
Design decisions stated explicitly, per this project's convention of not
silently baking in a judgment call (see `graph/retrieve.py`'s own numbered
list for the pattern this follows):

1. **`answer_con` is `eval.reader.read(..., mode="chain_of_note")`, aliased
   at import time.** The story packet's contract names a function
   `answer_con(question, pack) -> {notes, answer, abstain, citations}`
   (`E7-S5.md`), but no symbol of that name exists anywhere in this repo —
   `grep -rn answer_con` (excluding this file) returns nothing. What was
   actually built for E7-S5 is `eval/reader.py::read()`, whose
   `mode="chain_of_note"` path returns an `Answer` dataclass carrying
   exactly the same information (`text`, `abstained`, `notes`,
   `.citations` property of `[{session_id, turn_ids}, ...]`) under
   different names, and which every other consumer in this codebase
   already wires up this same way (`eval/reader.py::ReaderAnswerer.answer`:
   `pack = retriever(...); read(question, pack.items, mode,
   structural_absence=pack.structural_absence, ...)`). This module follows
   that established, already-proven wiring rather than inventing a second
   one. `read` is imported `as answer_con` purely so the call site below
   reads the same as the packet's own prose.

2. **`chat_once`'s top-level `answer` field is normalized to the literal
   string `"Not in this record"` whenever `pack.structural_absence` or
   `Answer.abstained` is true**, rather than passed through as the reader's
   raw `"NOT_IN_RECORD"` token. Both are unambiguously refusals (AC1 only
   requires "a refusal, not a guessed fact"), but the demo scripts
   (`DEMO.md`, `VIDEO_SCRIPT.md` Beat 2) require this exact human-legible
   line to appear on screen ("the literal line `Not in this record`"). One
   normalization point keeps the returned dict, the printed line, and the
   recorded demo consistent instead of three near-duplicate strings.

3. **`chat_once` never catches exceptions; `main()`'s prompt loop does.**
   `retrieve()` is documented to never raise (design decision 5 in
   `graph/retrieve.py`), but `answer_con`/`read()` can (e.g.
   `llm.MissingAPIKeyError`, a live provider error) when `dry_run=False`.
   Swallowing that inside `chat_once` would let a real error masquerade as
   "not in record" — the opposite of "do not confabulate". `chat_once`
   stays a pure, honest function; only the REPL (`main()`, where a crash
   mid-recording is the actual failure mode to guard against) catches and
   prints an `error:` line and keeps going.

4. **No `--model` / `--rendering` / `--reader` flags.** The story bans
   "demo chrome beyond a REPL" and any new retrieval/fusion surface;
   exposing reader internals as CLI knobs is exactly that kind of chrome
   for a three-minute demo video. `--dry-run` is kept because it is the
   project-wide convention for "run this without an API key/cost"
   (`eval/harness.py --dry-run`), and it is what makes `tests/test_agent.py`
   free and offline, per this story's own fixtures note.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import inspect
from typing import Callable

from medmemgraph.contracts import RetrieveResult
from medmemgraph.eval.reader import Answer, read as answer_con
from medmemgraph.graph.retrieve import retrieve as graph_retrieve

__all__ = ["chat_once", "main"]

DEFAULT_K = 8
"""Matches the contract's own default (`E8-S1.md`) and `retrieve.py`'s
`DEFAULT_SEED_K` — not a coincidence, both size a per-question top-k."""

_REFUSAL_LINE = "Not in this record"
"""The literal line `DEMO.md`/`VIDEO_SCRIPT.md` require on screen for Beat
2. No trailing punctuation — matches both docs' exact quoting."""

Retriever = Callable[..., RetrieveResult]


# ---------------------------------------------------------------------------
# chat_once — the frozen contract.
# ---------------------------------------------------------------------------


def _con_dict(answer: Answer) -> dict:
    """`Answer` -> the `con` dict the contract names ("notes / abstain /
    citations"), plus the token/latency fields `main()`'s printer needs
    (step 3 of the contract: "token/latency if present"). `notes` is
    flattened to `list[str]` (one note's text per retrieved item, in item
    order) per E7-S5's own frozen `answer_con` return shape (`notes:
    list[str]`) — `Answer.notes` is the richer `list[Note]` this project's
    actual reader produces; this is the one place that richer shape is
    narrowed back down to what the packet asked for."""
    return {
        "notes": [note.text for note in answer.notes],
        "abstain": answer.abstained,
        "citations": answer.citations,
        "mode": answer.mode,
        "rendering": answer.rendering,
        "prompt_tokens": answer.prompt_tokens,
        "completion_tokens": answer.completion_tokens,
        "total_tokens": answer.total_tokens,
        "latency_ms": answer.latency_ms,
    }



def _accepts_epsilon(retriever) -> bool:
    """Does this retriever take an `epsilon=` keyword?

    Introspected rather than isinstance-checked so any caller-supplied
    retriever participates without knowing about the flag. Unintrospectable
    callables answer False — degrading to the plain 3-arg call, never a
    TypeError mid-demo."""
    try:
        params = inspect.signature(retriever).parameters
    except (TypeError, ValueError):
        return False
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return True
    return "epsilon" in params


def chat_once(
    question: str,
    patient_id: str,
    *,
    k: int = DEFAULT_K,
    epsilon: float | None = 0.0,
    retriever: Retriever | None = None,
    dry_run: bool = False,
) -> dict:
    """`retrieve(question, patient_id, k)` -> `answer_con(question, pack)`
    -> the frozen return shape (`E8-S1.md`):

        {answer: str, retrieve: RetrieveResult, con: dict}

    Exactly one retrieve call, exactly one answer call — never a second
    retrieval hop (E7-S5 AC3's own banned-approach, re-stated here since
    this is the module that would be tempted to add one for a "let me
    check again" refinement).

    `retriever`: injection point for a scripted `RetrieveResult`
    (`tests/test_agent.py`) — defaults to the real, graph-backed
    `graph.retrieve.retrieve`. Never `contracts.mock_retrieve` by default;
    unlike `eval/reader.py`'s harness wiring, this is the live demo path,
    not an offline eval system.

    `dry_run`: threaded straight through to `answer_con` (`read`)'s own
    `dry_run` — the deterministic offline stub, no API key, no cost, no
    network call. `False` (real inference) is the default, matching
    `read()`'s own "real is the default, dry_run is opt-in" rule; the
    recorded demo (`VIDEO_SCRIPT.md`) must run with real inference.
    """
    retriever = retriever or graph_retrieve
    # epsilon=0 by default HERE, unlike `retrieve()`'s live default of 0.05.
    # That 5% exploration exists to log a non-zero propensity for the unchosen
    # arm so offline policy evaluation stays possible later — worth it for a
    # deployed system collecting data, actively harmful for a demo, where it
    # means roughly one in twenty questions silently takes the wrong route and
    # a recorded take is ruined for no visible reason. Pass --epsilon 0.05 to
    # restore exploration.
    # Forward `epsilon` only to a retriever that accepts it. The real
    # `graph.retrieve.retrieve` does; the simple 3-arg callables injected by
    # tests and by `contracts.mock_retrieve` do not, and passing it blindly
    # turns every injected retriever into a TypeError.
    pack: RetrieveResult = (
        retriever(question, patient_id, k, epsilon=epsilon)
        if epsilon is not None and _accepts_epsilon(retriever)
        else retriever(question, patient_id, k)
    )

    con_answer = answer_con(
        question,
        pack.items,
        "chain_of_note",
        structural_absence=pack.structural_absence,
        dry_run=dry_run,
        # The demo keeps the fuller clinical answer. `commit_style` (on by
        # default) instructs the reader to compress a progression into the one
        # "X to Y" transition MedLoCoMo's gold answers are written as — which
        # wins points on the benchmark and is the wrong thing to show a
        # clinician, who wants the intervening detail and the citations. Same
        # retrieval, same facts, different framing; the README says so rather
        # than letting this screen imply the eval answers look like this.
        commit_style=False,
    )

    refusing = pack.structural_absence or con_answer.abstained
    answer_text = _REFUSAL_LINE if refusing else con_answer.text

    return {
        "answer": answer_text,
        "retrieve": pack,
        "con": _con_dict(con_answer),
    }


# ---------------------------------------------------------------------------
# main() — CLI: --patient <id>, then a prompt loop.
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m medmemgraph.demo.agent",
        description=(
            "Chat about ONE patient's history: retrieve() once, answer via "
            "Chain-of-Note, print the route/absence/citations. "
            "See collaborative/design/stories/E8/E8-S1.md."
        ),
    )
    parser.add_argument(
        "--patient",
        required=True,
        metavar="PATIENT_ID",
        help="patient_id to chat about, e.g. a MedLoCoMo subject id such as 10056223",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
        help=f"top-k evidence items retrieved per question (default: {DEFAULT_K})",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.0,
        help=(
            "router exploration rate (default: 0.0, deterministic). retrieve()'s "
            "own live default is 0.05, which flips roughly 1 question in 20 to "
            "the other arm to keep offline policy evaluation possible; that is "
            "the wrong trade for a demo or a recording. Pass 0.05 to restore it."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "answer with the deterministic offline stub instead of a real LLM "
            "call -- no API key needed, no cost. For rehearsal only; the "
            "recorded demo must NOT use this flag."
        ),
    )
    return parser


def _fmt_bool(value: bool) -> str:
    """`True`/`False` -> `true`/`false` -- the exact casing `DEMO.md` /
    `VIDEO_SCRIPT.md` quote for the on-screen `structural_absence=` line."""
    return "true" if value else "false"


def _fmt_citations(citations: list[dict]) -> str:
    if not citations:
        return "(none)"
    return "; ".join(f"{c['session_id']} turns={c['turn_ids']}" for c in citations)


def _fmt_path(pack: RetrieveResult) -> str | None:
    """One representative graph path, truncated -- "a path, not a wall of
    retrieved chunks" (`VIDEO_SCRIPT.md` Beat 1's own must-show line).
    Returns `None` when the route never touched the graph arm (`paths` is
    always `[]` for a pure-vector route -- `retrieve.py`'s own contract)."""
    if not pack.paths:
        return None
    first = pack.paths[0]
    text = str(first.get("path", "")) if isinstance(first, dict) else str(first)
    more = f" (+{len(pack.paths) - 1} more path(s))" if len(pack.paths) > 1 else ""
    if len(text) > 240:
        text = text[:240] + "..."
    return text + more


def _print_result(result: dict) -> None:
    pack: RetrieveResult = result["retrieve"]
    con: dict = result["con"]
    refusing = pack.structural_absence or con["abstain"]

    print()
    if refusing:
        print(_REFUSAL_LINE)
    else:
        print(f"answer: {result['answer']}")

    print(f"route={pack.route}  structural_absence={_fmt_bool(pack.structural_absence)}")

    path_line = _fmt_path(pack)
    if path_line is not None:
        print(f"path: {path_line}")

    print(f"citations: {_fmt_citations(con['citations'])}")

    print(
        f"tokens: prompt={con['prompt_tokens']} completion={con['completion_tokens']} "
        f"total={con['total_tokens']}"
    )
    retrieve_ms = pack.latency_ms.get("total", 0.0) if pack.latency_ms else 0.0
    print(f"latency_ms: retrieve={retrieve_ms:.1f}  answer={con['latency_ms']:.1f}")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: `--patient <id>`, then a prompt loop. `argv=None`
    (the real `python -m medmemgraph.demo.agent` path) reads `sys.argv`
    via `argparse`'s own default; an explicit `argv` is the test seam
    (`tests/test_agent.py`) -- `argparse.parse_args([...])` never touches
    real `sys.argv` either way, so no monkeypatching is required to test
    `--help` (AC3: must not require HydraDB to print help -- nothing below
    this function's `--help` branch is reached in that case, since
    `argparse` prints help and exits before `chat_once` is ever imported
    from a call, let alone invoked)."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    print(f"MedMemGraph chat -- patient {args.patient!r}. Ctrl-D or 'exit' to quit.")
    if args.epsilon:
        print(f"(--epsilon {args.epsilon}: routing is randomized, not deterministic)")
    if args.dry_run:
        print("(--dry-run: offline stub answers, no API key, no LLM cost)")

    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break

        try:
            result = chat_once(
                question,
                args.patient,
                k=args.k,
                epsilon=args.epsilon,
                dry_run=args.dry_run,
            )
        except Exception as exc:  # noqa: BLE001 - a bad turn must not kill the demo loop
            print(f"error: {exc}")
            continue

        _print_result(result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
