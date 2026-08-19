"""graph/router.py — the frozen routing rule (ARCHITECTURE.md §7.1, story E5-S1).

`cross_admission -> graph`, `single_admission -> vector`, uncertain -> `hybrid`.
Evidence-backed, not a hedge: `literature/03` finds graph structure wins on
multi-hop and synthesis questions but **loses simple fact lookup by up to 26
points at 100-350x the query-token cost** of a vector arm. MedLoCoMo's own
gold labels split ~50/50 between the two scopes (`scope`/`question_type` on
every QA item — `pipeline.loader.load_qa`).

Two entry points:
  - `route_eval` — the eval-time router. Uses the benchmark's own gold
    `scope`/`question_type` labels; this is the *rule under test*, not a
    heuristic guess.
  - `route_live` — the demo-time router. No gold labels exist at demo time;
    a small, deterministic keyword / admission-count heuristic stands in.

Both apply the SAME epsilon-randomization step before returning: with
probability `epsilon` (default 0.05), a `graph`/`vector` decision is flipped
to the other action (`hybrid` never flips — there is no second action
defined for it to explore against). This is NOT a learned/contextual bandit
(`literature/19` gates that behind a headroom diagnostic this project has
not run) — it is the smallest possible instrument that keeps the door open
for one later, by logging a non-zero propensity for the action that
*wasn't* taken. A fully deterministic router logs propensity 0 for the
unchosen arm on every single row, which makes offline policy evaluation
(IPS/DR/SNIPS) mathematically impossible to retrofit after the fact — this
costs about 30 minutes today and cannot be added later (ARCHITECTURE §7.1,
BUILD-PLAN.html).

Banned (story's own list, repeated here so a reviewer doesn't have to go
hunting): a trained classifier / LinUCB / Thompson sampling, routing
everything to graph, forking `Route` to add a `"lexical"` value (lexical is
a `channel` on `RetrieveItem`, never a `route` — `contracts.py`), and using
an LLM to classify the question.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

from medmemgraph.contracts import Route

__all__ = [
    "CROSS_ADMISSION_QUESTION_TYPES",
    "SINGLE_ADMISSION_QUESTION_TYPES",
    "DEFAULT_EPSILON",
    "RouteDecision",
    "route_eval",
    "route_live",
    "log_route_decision",
    "ROUTE_LOG",
]

# ---------------------------------------------------------------------------
# The frozen category rule — ARCHITECTURE.md §7.1 / E5-S1's own contract.
# ---------------------------------------------------------------------------

CROSS_ADMISSION_QUESTION_TYPES = frozenset(
    {"longitudinal_progression", "cross_admission_comparison", "frequency_pattern"}
)
SINGLE_ADMISSION_QUESTION_TYPES = frozenset({"medical_reasoning", "care_plan_rationale"})

DEFAULT_EPSILON = 0.05
"""ARCHITECTURE §7.1 / BUILD-PLAN.html's own default. Small, not zero: big
enough to produce a usable offline-eval sample over a 5-10k question corpus,
small enough that the live demo is not visibly flip-flopping."""


@dataclass(frozen=True)
class RouteDecision:
    route: Route
    reason: str
    epsilon_flipped: bool
    propensity: float
    subtype: str | None
    split: str | None


# ---------------------------------------------------------------------------
# ε-logging — "log (question, features, route, ε-flag, outcome)".
# ---------------------------------------------------------------------------

ROUTE_LOG: list[dict] = []
"""In-process, append-only log of every route decision this module has
produced: `{question, features, route, epsilon_flipped, propensity, reason,
outcome}`. Deliberately a plain in-memory list, not a file/DB sink — the
story's own scope is "make future offline learning possible", not "ship a
production logging pipeline". A caller that wants persistence reads this
list (e.g. from `retrieve.py`, once per call) or calls `log_route_decision`
itself and writes the record out. `outcome` is `None` at decision time
(this module cannot know whether the retrieval that follows actually helped
answer the question) — a caller that does know fills it in later by mutating
the returned dict in place, or by keeping its own join on `question`."""


def log_route_decision(
    *, question: str, features: dict, decision: RouteDecision, outcome: dict | None = None
) -> dict:
    """Append one structured record to `ROUTE_LOG` and return it. `features`
    is whatever the caller used to route — the gold `scope`/`question_type`
    for `route_eval`, or the live heuristic's own signals for `route_live`.
    This is what makes offline policy evaluation possible later: every row
    carries the actual `route` taken AND its `propensity` under the
    randomized policy, including rows where an ε-flip picked the arm the
    deterministic rule would NOT have chosen — those rows are exactly the
    ones a deterministic router could never have logged with a non-zero
    propensity."""
    record = {
        "question": question,
        "features": dict(features),
        "route": decision.route,
        "epsilon_flipped": decision.epsilon_flipped,
        "propensity": decision.propensity,
        "reason": decision.reason,
        "outcome": outcome,
    }
    ROUTE_LOG.append(record)
    return record


# ---------------------------------------------------------------------------
# ε-flip — shared by both entry points.
# ---------------------------------------------------------------------------


def _flip(base_route: Route, epsilon: float, rng: random.Random | None) -> tuple[Route, bool, float]:
    """Randomize a `graph`/`vector` base decision. `hybrid` never flips (no
    second action is defined for it — the router is already declining to
    pick a side) and is returned as a deterministic choice with propensity
    1.0. Returns `(route, epsilon_flipped, propensity)`; `propensity` is the
    probability, under this randomized policy, of the RETURNED route —
    `1 - epsilon` when not flipped, `epsilon` when flipped."""
    if base_route == "hybrid":
        return base_route, False, 1.0
    draw = (rng or random).random()
    if draw < epsilon:
        flipped_route: Route = "vector" if base_route == "graph" else "graph"
        return flipped_route, True, epsilon
    return base_route, False, 1.0 - epsilon


# ---------------------------------------------------------------------------
# route_eval — eval-time router, gold scope/question_type.
# ---------------------------------------------------------------------------


def route_eval(
    scope: str | None,
    question_type: str | None,
    *,
    epsilon: float = DEFAULT_EPSILON,
    rng: random.Random | None = None,
) -> RouteDecision:
    """Frozen eval rule (copied verbatim from ARCHITECTURE §7.1 / E5-S1):

        if question_type in cross or scope == "cross_admission": graph
        elif question_type in single or scope == "single_admission": vector
        else: hybrid

    `adversarial` is not itself a route signal — it is not a member of
    either `question_type` set and is not the string `"cross_admission"` /
    `"single_admission"`, so an adversarial item with neither gold label
    recognized falls through to `hybrid`, same as any other unclassified
    item; the router still maps by `scope` when present, exactly as this
    story's own note requires ("adversarial is not a reason to skip the
    graph seed check later")."""
    if question_type in CROSS_ADMISSION_QUESTION_TYPES or scope == "cross_admission":
        base_route: Route = "graph"
        reason = "cross_admission_rule"
    elif question_type in SINGLE_ADMISSION_QUESTION_TYPES or scope == "single_admission":
        base_route = "vector"
        reason = "single_admission_rule"
    else:
        base_route = "hybrid"
        reason = "unclassified_defaults_to_hybrid"

    route, flipped, propensity = _flip(base_route, epsilon, rng)
    if flipped:
        reason = f"{reason}+epsilon_flip"
    return RouteDecision(
        route=route,
        reason=reason,
        epsilon_flipped=flipped,
        propensity=propensity,
        subtype=question_type,
        split=scope,
    )


# ---------------------------------------------------------------------------
# route_live — demo-time router, deterministic keyword heuristic.
# ---------------------------------------------------------------------------

_GRAPH_KEYWORD_RE = re.compile(
    r"compar|progress|over time|how often|how many times|across admission"
    r"|between (?:the )?(?:two |both )?admission|between (?:the )?(?:two |both )?(?:stay|visit)"
    r"|since (?:the|her|his|my) last",
    re.IGNORECASE,
)
"""ARCHITECTURE §7.1's own live heuristic: "comparison / progression / "over
time" / "how often" / "how many times" / "compared to last admission"".

`between ... admission(s)/stay(s)/visit(s)` added 2026-08-18: the ARCHITECTURE
list covers "across admission" but not "between admissions", and "between" is
the more natural phrasing — `demo/VIDEO_SCRIPT.md`'s OWN example question ("Did
the furosemide dose change between the two admissions?") missed the pattern and
routed to `hybrid`. This heuristic only runs at demo time (`route_live`);
evaluation uses the benchmark's gold `scope`/`question_type` labels via
`route_eval`, so no reported number depends on it.
`compar` (not `comparison`/`compared`) and `progress` (not `progression`)
are deliberately truncated stems so both noun and verb forms match; `across
admission` matches "across admissions" as a substring (AC4's own worked
example, "how has her A1c changed across admissions?")."""

_VECTOR_KEYWORD_RE = re.compile(r"why this plan|this stay|this admission|labs this stay", re.IGNORECASE)
"""ARCHITECTURE §7.1's own live heuristic: "why this plan" / "this stay" /
"this admission" / labs-this-stay language."""


def route_live(
    question: str,
    *,
    n_admissions_mentioned: int = 0,
    epsilon: float = DEFAULT_EPSILON,
    rng: random.Random | None = None,
) -> RouteDecision:
    """Deterministic demo-time heuristic (no gold labels exist at demo
    time, no trained classifier, no LLM classification — story's own banned
    list). `subtype`/`split` are `None`: there is no gold `question_type`/
    `scope` to report at live-demo time."""
    if n_admissions_mentioned > 1 or _GRAPH_KEYWORD_RE.search(question):
        base_route: Route = "graph"
        reason = "live_heuristic_cross_admission_signal"
    elif _VECTOR_KEYWORD_RE.search(question):
        base_route = "vector"
        reason = "live_heuristic_single_admission_signal"
    else:
        base_route = "hybrid"
        reason = "live_heuristic_uncertain"

    route, flipped, propensity = _flip(base_route, epsilon, rng)
    if flipped:
        reason = f"{reason}+epsilon_flip"
    return RouteDecision(
        route=route,
        reason=reason,
        epsilon_flipped=flipped,
        propensity=propensity,
        subtype=None,
        split=None,
    )
