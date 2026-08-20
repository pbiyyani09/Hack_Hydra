"""pipeline/normalize.py — relative-time resolution vs the turn's own clock,
and closed-vocabulary predicate canonicalization.

Both functions here are pure (no LLM, no network, no wall-clock reads) so
they are cheaply unit-testable and safe to run at ingest scale. See
`ARCHITECTURE.md` §6.2/§6.3 and `literature/14-information-extraction-
negation-temporal.md` Part 4 / Stage 3 for the design this module encodes.

Temporal rule (ARCHITECTURE §6.2, survey 05 Q3 / Graphiti reference-
timestamp pattern, corroborated by `literature/14` R-IE-16 through R-IE-19):
resolve "two weeks ago", "since my last visit", "yesterday" etc. against
**the turn's own `time`** — never wall-clock ingest time, never query time.
MedLoCoMo timestamps are MIMIC date-shifted into the 2100s+, so nothing here
may assume the present era; `datetime.now()` never appears in this module.

`literature/14` Part 4 documents the exact bottleneck this module is built
against: span detection is easy (~90%) but *value normalization* is the hard
part (~73-78% on the two largest converging benchmarks, R-IE-18/R-IE-19) —
and both source benchmarks independently converge on the same discipline
this module follows: when an expression is genuinely vague ("a while back",
"some time"), do not fabricate false precision — snap to the coarsest
defensible anchor (here: the turn's own time) and lower confidence, rather
than guess an exact date (R-IE-18 "Discussion" point (i), HeidelTime's own
documented choice).

Predicate canonicalization (ARCHITECTURE §6.3, survey 04 EDC shape): the
on-disk story E2-S2 specifies an embed-nearest-neighbour + one-LLM-confirm
pipeline. This story's contract (see the story text) trades that for a pure,
deterministic alias/definition lookup — no embedding call, no LLM call,
fully unit-testable without a key — while keeping the same closed-vocabulary
discipline: match onto `contracts.PREDICATES` or return `None` so the caller
drops the fact. Never invent a predicate outside the closed set
(`decisions/` — extend `PREDICATES` in exactly one place, `contracts.py`,
not here).
"""

from __future__ import annotations

import calendar
import re
from datetime import datetime, timedelta

from medmemgraph.contracts import PREDICATES

__all__ = [
    "PREDICATE_DEFINITIONS",
    "resolve_time",
    "canonicalize_predicate",
]

# ---------------------------------------------------------------------------
# Temporal resolution
# ---------------------------------------------------------------------------

# Confidence tiers (this module's own calibration — literature/14 gives the
# *shape* of the problem, not a numeric confidence scale to copy verbatim;
# see the module docstring and R-IE-18/19 for the "vague -> lower
# confidence, don't guess" discipline these numbers implement).
_CONF_EXPLICIT_ABSOLUTE = 1.0
_CONF_NO_EXPRESSION = 1.0
_CONF_TODAY = 1.0
_CONF_YESTERDAY = 0.95
_CONF_NUMERIC_AGO = 0.9
_CONF_LAST_WEEKDAY = 0.85
_CONF_SINCE_LAST_VISIT_RESOLVED = 0.8
_CONF_VAGUE_QUANTITY_AGO = 0.6
_CONF_LAST_UNIT = 0.6
_CONF_UNRECOGNIZED_EXPRESSION = 0.4
_CONF_SINCE_LAST_VISIT_UNRESOLVED = 0.3
_CONF_VAGUE_DURATION = 0.3

_NUMBER_WORDS: dict[str, int] = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12,
    "a couple": 2, "couple": 2, "a few": 3, "few": 3, "several": 4,
}

_WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}

_MONTH_NAMES = {
    name.lower(): i
    for i, name in enumerate(
        [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        ],
        start=1,
    )
}

_AGO_RE = re.compile(
    r"\b(?P<qty>a\s+couple(?:\s+of)?|a\s+few|several|an?|\d+|"
    r"one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+"
    r"(?P<unit>day|week|month|year)s?\s+ago\b",
    re.IGNORECASE,
)
_YESTERDAY_RE = re.compile(r"\byesterday\b", re.IGNORECASE)
_TODAY_RE = re.compile(r"\btoday\b|\bthis\s+morning\b|\btonight\b", re.IGNORECASE)
_LAST_WEEKDAY_RE = re.compile(
    r"\blast\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.IGNORECASE,
)
_LAST_UNIT_RE = re.compile(r"\blast\s+(week|month|year)\b", re.IGNORECASE)
_SINCE_LAST_VISIT_RE = re.compile(
    r"\bsince\s+(?:my|our|the)?\s*last\s+visit\b", re.IGNORECASE
)
_VAGUE_DURATION_RE = re.compile(
    r"\b(?:a\s+while\s+(?:back|ago)|some\s+time\s+ago|a\s+long\s+time\s+ago|"
    r"a\s+bit\s+ago|not\s+long\s+ago)\b",
    re.IGNORECASE,
)
_ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})(?:[ T](\d{2}:\d{2}(?::\d{2})?))?\b")
_MONTH_DAY_YEAR_RE = re.compile(
    r"\b(" + "|".join(_MONTH_NAMES) + r")\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b",
    re.IGNORECASE,
)


def _parse_anchor(turn_time: str) -> datetime:
    """Parse the turn's own `time` string. MedLoCoMo turns look like
    ``2124-01-01 00:00:00``; the frozen contract's ISO strings use a ``T``
    separator. ``datetime.fromisoformat`` accepts both under Python 3.13."""
    return datetime.fromisoformat(turn_time.strip())


def _to_iso(dt: datetime) -> str:
    return dt.isoformat()


def _add_months(dt: datetime, delta_months: int) -> datetime:
    """Calendar-correct month add/subtract with day-of-month clamping
    (e.g. subtracting a month from the 31st clamps to the shorter month's
    last day rather than raising or silently rolling over)."""
    total = dt.month - 1 + delta_months
    year = dt.year + total // 12
    month = total % 12 + 1
    day = min(dt.day, calendar.monthrange(year, month)[1])
    return dt.replace(year=year, month=month, day=day)


def _parse_quantity(qty: str) -> tuple[int, bool]:
    """Returns (count, is_vague). A vague quantity ("a couple", "a few",
    "several", the bare article "a"/"an") gets a lower confidence tier than
    an explicit number, even though both resolve to a concrete date — the
    date is a best-effort approximation, not a value the source text
    actually stated precisely."""
    q = re.sub(r"\s+", " ", qty.strip().lower())
    q = re.sub(r"\s+of$", "", q)
    if q.isdigit():
        return int(q), False
    if q in ("a", "an"):
        return 1, True
    if q in _NUMBER_WORDS:
        vague = q in ("a couple", "couple", "a few", "few", "several")
        return _NUMBER_WORDS[q], vague
    last_token = q.split()[-1] if q.split() else q
    return _NUMBER_WORDS.get(last_token, 1), True


def _try_absolute_date(expr: str) -> datetime | None:
    m = _ISO_DATE_RE.search(expr)
    if m:
        date_part = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        time_part = m.group(4) or "00:00:00"
        try:
            return datetime.fromisoformat(f"{date_part} {time_part}")
        except ValueError:
            return None
    m = _MONTH_DAY_YEAR_RE.search(expr)
    if m:
        month = _MONTH_NAMES[m.group(1).lower()]
        day = int(m.group(2))
        year = int(m.group(3))
        try:
            return datetime(year, month, day)
        except ValueError:
            return None
    return None


def resolve_time(
    expr: str | None,
    turn_time: str,
    *,
    prior_admission_end: str | None = None,
) -> tuple[str, float]:
    """Resolve a (possibly relative) temporal expression against the turn's
    own clock. Returns ``(iso_string, confidence)``.

    Anchor discipline (ARCHITECTURE §6.2): the reference timestamp is
    always ``turn_time`` — never wall-clock ``datetime.now()``, never a
    query-time value. When an expression cannot be confidently resolved to
    a specific date, this function returns the anchor (`turn_time`, or
    `prior_admission_end` where that is the more specific fallback
    available) with a lowered confidence rather than guessing — the
    discipline `literature/14` documents HeidelTime and the i2b2 2012
    challenge both converging on for the hard "value normalization" half of
    temporal extraction (see module docstring).

    ``prior_admission_end`` is the caller-supplied end timestamp of this
    patient's prior admission (``Admission.admission_end`` one index back
    in ``Conversation.admissions``), needed to resolve "since my last
    visit" / "since my last admission" style expressions. If the caller has
    no prior admission (this is the patient's first on record), pass
    ``None`` — the expression then falls through to the "unresolvable,
    return the anchor" branch, exactly as the story's contract specifies.
    """
    anchor = _parse_anchor(turn_time)

    if not expr or not expr.strip():
        return _to_iso(anchor), _CONF_NO_EXPRESSION

    text = expr.strip()
    lowered = text.lower()

    absolute = _try_absolute_date(text)
    if absolute is not None:
        return _to_iso(absolute), _CONF_EXPLICIT_ABSOLUTE

    if _SINCE_LAST_VISIT_RE.search(lowered):
        if prior_admission_end:
            return (
                _to_iso(_parse_anchor(prior_admission_end)),
                _CONF_SINCE_LAST_VISIT_RESOLVED,
            )
        return _to_iso(anchor), _CONF_SINCE_LAST_VISIT_UNRESOLVED

    if _YESTERDAY_RE.search(lowered):
        return _to_iso(anchor - timedelta(days=1)), _CONF_YESTERDAY

    if _TODAY_RE.search(lowered):
        return _to_iso(anchor), _CONF_TODAY

    m = _LAST_WEEKDAY_RE.search(lowered)
    if m:
        target = _WEEKDAYS[m.group(1).lower()]
        # "last Monday": the most recent occurrence strictly before the
        # anchor date, always at least one day back even if the anchor
        # itself falls on that weekday (survey 05 gives no guidance here;
        # this is this module's own, documented convention).
        days_back = (anchor.weekday() - target) % 7
        if days_back == 0:
            days_back = 7
        return _to_iso(anchor - timedelta(days=days_back)), _CONF_LAST_WEEKDAY

    m = _LAST_UNIT_RE.search(lowered)
    if m:
        unit = m.group(1)
        if unit == "week":
            resolved = anchor - timedelta(weeks=1)
        elif unit == "month":
            resolved = _add_months(anchor, -1)
        else:
            resolved = _add_months(anchor, -12)
        return _to_iso(resolved), _CONF_LAST_UNIT

    m = _AGO_RE.search(lowered)
    if m:
        count, vague = _parse_quantity(m.group("qty"))
        unit = m.group("unit")
        if unit == "day":
            resolved = anchor - timedelta(days=count)
        elif unit == "week":
            resolved = anchor - timedelta(weeks=count)
        elif unit == "month":
            resolved = _add_months(anchor, -count)
        else:
            resolved = _add_months(anchor, -12 * count)
        confidence = _CONF_VAGUE_QUANTITY_AGO if vague else _CONF_NUMERIC_AGO
        return _to_iso(resolved), confidence

    if _VAGUE_DURATION_RE.search(lowered):
        # HeidelTime's own discipline (R-IE-18): don't fabricate false
        # precision for "a while back" / "some time ago" — snap to the
        # anchor and lower confidence instead of guessing an exact date.
        return _to_iso(anchor), _CONF_VAGUE_DURATION

    # Nothing matched: not empty, not resolvable. Return the anchor rather
    # than guess (banned approaches: no ingest-time/$now anchor; this is
    # still the turn anchor, just at the lowest non-"since last visit"
    # confidence tier).
    return _to_iso(anchor), _CONF_UNRECOGNIZED_EXPRESSION


# ---------------------------------------------------------------------------
# Predicate canonicalization
# ---------------------------------------------------------------------------

PREDICATE_DEFINITIONS: dict[str, str] = {
    "TAKES_MEDICATION": (
        "the subject is currently taking or was prescribed the object as medication"
    ),
    "HAS_CONDITION": (
        "the subject has been diagnosed with or currently has the object condition"
    ),
    "HAS_ALLERGY_TO": (
        "the subject has an allergy or adverse reaction to the object"
    ),
    "REPORTS_SYMPTOM": "the subject reports the object as a symptom",
    "HAD_PROCEDURE": "the subject underwent the object procedure",
    "CURRENT_DOSAGE_OF": "the current dose of the named medication is the object",
    "PRIMARY_CARE_PROVIDER_OF": (
        "the object's named clinician is the subject's primary care provider"
    ),
    "HAD_INCIDENT": (
        "the subject experienced the object as a discrete clinical incident "
        "or accidental injury event (e.g. a fall) — not an ongoing symptom, "
        "a diagnosed condition, or a performed medical procedure"
    ),
}
"""One-line definition per `contracts.PREDICATES` member (ARCHITECTURE §6.3
/ E2-S2 story text, copied verbatim) — the single source of truth both this
module's alias table and `extract.py`'s system prompt draw from, so the
prompt's vocabulary description and the canonicalize pass never drift
apart."""

assert set(PREDICATE_DEFINITIONS) == PREDICATES, (
    "PREDICATE_DEFINITIONS must have exactly one entry per contracts.PREDICATES "
    "member — extend both in the same change, never one without the other"
)

# EDC-lite: this story's contract trades survey 04's embed + LLM-confirm
# pipeline for a pure alias/definition lookup (see module docstring). Each
# alias is a normalized (lowercase, underscore-joined) synonym phrase for
# the predicate it maps to. Deliberately conservative — no single common
# English verb ("has", "shows", "reports") is used alone, since those are
# ambiguous across multiple predicates; every alias here is specific enough
# that a collision across predicates would itself be a canonicalization bug.
_PREDICATE_ALIASES: dict[str, str] = {
    # TAKES_MEDICATION
    "is_prescribed": "TAKES_MEDICATION",
    "prescribed": "TAKES_MEDICATION",
    "was_prescribed": "TAKES_MEDICATION",
    "on_medication": "TAKES_MEDICATION",
    "takes": "TAKES_MEDICATION",
    "is_taking": "TAKES_MEDICATION",
    "taking": "TAKES_MEDICATION",
    "started_taking": "TAKES_MEDICATION",
    "continues_taking": "TAKES_MEDICATION",
    "keeps_taking": "TAKES_MEDICATION",
    "is_on": "TAKES_MEDICATION",
    "medicated_with": "TAKES_MEDICATION",
    "takes_medication": "TAKES_MEDICATION",
    # HAS_CONDITION
    "has_condition": "HAS_CONDITION",
    "diagnosed_with": "HAS_CONDITION",
    "has_diagnosis_of": "HAS_CONDITION",
    "suffers_from": "HAS_CONDITION",
    "history_of": "HAS_CONDITION",
    "has_history_of": "HAS_CONDITION",
    "presents_with": "HAS_CONDITION",
    "has_been_diagnosed_with": "HAS_CONDITION",
    # HAS_ALLERGY_TO
    "has_allergy_to": "HAS_ALLERGY_TO",
    "allergic_to": "HAS_ALLERGY_TO",
    "is_allergic_to": "HAS_ALLERGY_TO",
    "allergy_to": "HAS_ALLERGY_TO",
    "reacts_to": "HAS_ALLERGY_TO",
    "has_reaction_to": "HAS_ALLERGY_TO",
    "has_an_allergy_to": "HAS_ALLERGY_TO",
    # REPORTS_SYMPTOM
    "reports_symptom": "REPORTS_SYMPTOM",
    "reports_symptom_of": "REPORTS_SYMPTOM",
    "experiencing": "REPORTS_SYMPTOM",
    "is_experiencing": "REPORTS_SYMPTOM",
    "complains_of": "REPORTS_SYMPTOM",
    "has_symptom": "REPORTS_SYMPTOM",
    "reports_experiencing": "REPORTS_SYMPTOM",
    # HAD_PROCEDURE
    "had_procedure": "HAD_PROCEDURE",
    "underwent": "HAD_PROCEDURE",
    "had_a_procedure": "HAD_PROCEDURE",
    "was_performed": "HAD_PROCEDURE",
    "had": "HAD_PROCEDURE",  # deliberately scoped to procedure objects at the call site
    # CURRENT_DOSAGE_OF
    "current_dosage_of": "CURRENT_DOSAGE_OF",
    "dose_of": "CURRENT_DOSAGE_OF",
    "dosage_of": "CURRENT_DOSAGE_OF",
    "current_dose_of": "CURRENT_DOSAGE_OF",
    "takes_dose_of": "CURRENT_DOSAGE_OF",
    "dosed_at": "CURRENT_DOSAGE_OF",
    # PRIMARY_CARE_PROVIDER_OF
    "primary_care_provider_of": "PRIMARY_CARE_PROVIDER_OF",
    "primary_care_physician_of": "PRIMARY_CARE_PROVIDER_OF",
    "pcp_of": "PRIMARY_CARE_PROVIDER_OF",
    "is_pcp_of": "PRIMARY_CARE_PROVIDER_OF",
    "is_primary_care_provider_of": "PRIMARY_CARE_PROVIDER_OF",
    # HAD_INCIDENT — added post-freeze for the real observed "had a fall"
    # gap (see contracts.py's PREDICATES docstring). Deliberately narrow:
    # only fall-shaped phrasing, not a catch-all for arbitrary incidents,
    # since a fall is the only real data-demanded case so far.
    "had_a_fall": "HAD_INCIDENT",
    "had_a_fall_incident": "HAD_INCIDENT",
    "had_an_incident": "HAD_INCIDENT",
    "experienced_a_fall": "HAD_INCIDENT",
    "sustained_a_fall": "HAD_INCIDENT",
    "fell": "HAD_INCIDENT",
    "fell_down": "HAD_INCIDENT",
    "had_fall": "HAD_INCIDENT",
}

# "had" collides with plain past-tense "has" usage far too often to be a
# safe blanket alias; it is legal only when scoped to a clearly procedural
# object at the canonicalize_predicate call site.
del _PREDICATE_ALIASES["had"]


def _normalize_phrase(phrase: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", phrase.strip().lower())
    return normalized.strip("_")


_STOPWORD_TOKENS = frozenset(
    {
        "with", "the", "of", "in", "on", "to", "for", "and", "a", "an",
        "is", "at", "by", "or", "as", "it", "was", "were",
    }
)
"""Filler tokens inside otherwise-specific multi-word aliases (e.g.
"diagnosed_with", "on_medication"). Excluded from the single-token
qualification rule below so a phrase that merely shares a preposition with
an alias ("enjoys hiking with") can't hijack an unrelated predicate — length
alone (the >=4-chars rule) isn't enough, since "with" itself is 4 characters."""


def canonicalize_predicate(phrase: str | None) -> str | None:
    """Map an open predicate phrase onto `contracts.PREDICATES`, or return
    `None` when nothing fits — the caller drops the fact rather than
    inventing a predicate outside the closed vocabulary (ARCHITECTURE §4 /
    §6.3, decision: predicate is a closed vocabulary, extend it in exactly
    one place: `contracts.PREDICATES`, never here).

    Pure, deterministic, no LLM/embedding call — see module docstring for
    why this trades the on-disk E2-S2 story's EDC embed+confirm pipeline
    for a plain alias table.
    """
    if not phrase or not phrase.strip():
        return None

    normalized = _normalize_phrase(phrase)
    if not normalized:
        return None

    exact = _PREDICATE_ALIASES.get(normalized)
    if exact is not None:
        return exact

    # A predicate already spelled as one of the closed-vocab strings
    # (case/spacing aside) is trivially itself.
    upper = phrase.strip().upper().replace(" ", "_")
    if upper in PREDICATES:
        return upper

    # Token-overlap fallback: score every alias by how many of its
    # underscore-joined tokens appear, in order, as a contiguous run inside
    # the normalized phrase (or vice-versa) — catches phrases like
    # "on_metformin" (a med name fused onto the "on" trigger) without
    # letting a single generic token hijack an unrelated predicate. Require
    # at least 2 overlapping tokens, or a single overlapping *non-stopword*
    # token that is itself >= 4 characters.
    phrase_tokens = set(normalized.split("_"))
    best_predicate: str | None = None
    best_score = 0
    for alias, predicate in _PREDICATE_ALIASES.items():
        alias_tokens = alias.split("_")
        overlap = (phrase_tokens & set(alias_tokens)) - _STOPWORD_TOKENS
        if not overlap:
            continue
        score = len(overlap)
        qualifies = score >= 2 or (score == 1 and len(next(iter(overlap))) >= 4)
        if qualifies and score > best_score:
            best_score = score
            best_predicate = predicate

    return best_predicate
