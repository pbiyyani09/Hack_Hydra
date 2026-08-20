"""tests/test_normalize.py — pure-function tests for
`medmemgraph.pipeline.normalize`. No LLM, no network, no wall-clock reads:
every case is deterministic input -> deterministic output.

Anchor discipline under test (ARCHITECTURE §6.2): the reference timestamp is
always the turn's own `time`, never `datetime.now()`. Test 3 below patches
`datetime.now` to a wall-clock date that is centuries away from the corpus
era specifically to prove the module never reads it.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from medmemgraph.contracts import SENTINEL_VALID_TO
from medmemgraph.pipeline.normalize import canonicalize_predicate, resolve_time

ANCHOR = "2124-03-15 10:00:00"  # a Wednesday, MIMIC-shifted (2100s) era


# ---------------------------------------------------------------------------
# resolve_time
# ---------------------------------------------------------------------------


class TestResolveTimeFixtureTable:
    """The story's own fixture table, verbatim (date equality is enough per
    the acceptance criteria; time-of-day may stay at the turn's own time)."""

    @pytest.mark.parametrize(
        "expr,turn_time,expected_date_prefix",
        [
            ("two weeks ago", "2124-03-15 10:00:00", "2124-03-01"),
            ("yesterday", "2124-03-15 10:00:00", "2124-03-14"),
            ("", "2124-03-15 10:00:00", "2124-03-15"),
        ],
    )
    def test_fixture_table(self, expr, turn_time, expected_date_prefix):
        iso, confidence = resolve_time(expr, turn_time)
        assert iso.startswith(expected_date_prefix)
        assert 0.0 <= confidence <= 1.0


class TestResolveTimeAbsoluteAndEmpty:
    def test_no_expression_returns_turn_time_at_full_confidence(self):
        iso, confidence = resolve_time("", ANCHOR)
        assert iso == "2124-03-15T10:00:00"
        assert confidence == 1.0

    def test_none_expression_treated_as_no_expression(self):
        iso, confidence = resolve_time(None, ANCHOR)
        assert iso == "2124-03-15T10:00:00"
        assert confidence == 1.0

    def test_absolute_iso_date_in_text(self):
        iso, confidence = resolve_time("on 2124-01-05", ANCHOR)
        assert iso == "2124-01-05T00:00:00"
        assert confidence == 1.0

    def test_absolute_month_name_date(self):
        iso, confidence = resolve_time("January 5, 2124", ANCHOR)
        assert iso == "2124-01-05T00:00:00"
        assert confidence == 1.0


class TestResolveTimeRelativeAgo:
    def test_yesterday(self):
        iso, confidence = resolve_time("yesterday", ANCHOR)
        assert iso == "2124-03-14T10:00:00"
        assert confidence == pytest.approx(0.95)

    def test_today(self):
        iso, confidence = resolve_time("today", ANCHOR)
        assert iso == "2124-03-15T10:00:00"
        assert confidence == pytest.approx(1.0)

    def test_numeric_days_ago(self):
        iso, confidence = resolve_time("14 days ago", ANCHOR)
        assert iso == "2124-03-01T10:00:00"
        assert confidence == pytest.approx(0.9)

    def test_spelled_out_number_weeks_ago(self):
        iso, confidence = resolve_time("two weeks ago", ANCHOR)
        assert iso == "2124-03-01T10:00:00"
        assert confidence == pytest.approx(0.9)

    def test_months_ago_calendar_correct(self):
        iso, _ = resolve_time("three months ago", ANCHOR)
        assert iso.startswith("2123-12-15")

    def test_years_ago(self):
        iso, _ = resolve_time("two years ago", ANCHOR)
        assert iso.startswith("2122-03-15")

    def test_vague_quantity_ago_is_lower_confidence_than_numeric(self):
        vague_iso, vague_conf = resolve_time("a couple of weeks ago", ANCHOR)
        numeric_iso, numeric_conf = resolve_time("two weeks ago", ANCHOR)
        assert vague_iso == numeric_iso  # both resolve to the same date (2 weeks)
        assert vague_conf < numeric_conf

    def test_a_few_days_ago(self):
        iso, confidence = resolve_time("a few days ago", ANCHOR)
        assert iso.startswith("2124-03-12")
        assert confidence < 0.9  # vague quantity, lower tier than explicit numeric

    def test_month_boundary_day_clamping(self):
        # Jan 31 minus 1 month must not raise / roll over illegally.
        iso, _ = resolve_time("one month ago", "2124-01-31 09:00:00")
        assert iso.startswith("2123-12-31") or iso.startswith("2123-12-30")


class TestResolveTimeLastWeekday:
    def test_last_monday_from_wednesday(self):
        # ANCHOR (2124-03-15) is a Wednesday; the most recent prior Monday
        # is 2124-03-13.
        assert datetime.fromisoformat(ANCHOR).strftime("%A") == "Wednesday"
        iso, confidence = resolve_time("last Monday", ANCHOR)
        assert iso.startswith("2124-03-13")
        resolved_weekday = datetime.fromisoformat(iso).strftime("%A")
        assert resolved_weekday == "Monday"
        assert confidence == pytest.approx(0.85)

    def test_last_weekday_always_goes_back_at_least_one_full_week_when_same_day(self):
        # The anchor itself is a Wednesday; "last Wednesday" must not return
        # the anchor's own date.
        iso, _ = resolve_time("last Wednesday", ANCHOR)
        assert iso.startswith("2124-03-08")

    def test_last_week_unit(self):
        iso, confidence = resolve_time("last week", ANCHOR)
        assert iso.startswith("2124-03-08")
        assert confidence == pytest.approx(0.6)

    def test_last_month_unit(self):
        iso, _ = resolve_time("last month", ANCHOR)
        assert iso.startswith("2124-02-15")


class TestResolveTimeSinceMyLastVisit:
    """"since my last visit" needs a prior admission end (needs_prior_admission_end
    per the story text) — accepted as an argument, not looked up internally."""

    def test_resolved_against_prior_admission_end(self):
        prior_end = "2124-02-01 09:00:00"
        iso, confidence = resolve_time(
            "since my last visit", ANCHOR, prior_admission_end=prior_end
        )
        assert iso == "2124-02-01T09:00:00"
        assert confidence == pytest.approx(0.8)

    def test_unresolvable_without_prior_admission_returns_anchor_with_lower_confidence(self):
        iso, confidence = resolve_time("since my last visit", ANCHOR, prior_admission_end=None)
        assert iso == "2124-03-15T10:00:00"  # returns the anchor, does not guess
        assert confidence < 0.5  # lowered, not full confidence

    def test_since_our_last_visit_variant(self):
        prior_end = "2124-02-01 09:00:00"
        iso, _ = resolve_time("since our last visit", ANCHOR, prior_admission_end=prior_end)
        assert iso == "2124-02-01T09:00:00"


class TestResolveTimeVagueDurations:
    def test_a_while_back_snaps_to_anchor_not_a_guess(self):
        iso, confidence = resolve_time("a while back", ANCHOR)
        assert iso == "2124-03-15T10:00:00"
        assert confidence < 0.5

    def test_some_time_ago_snaps_to_anchor(self):
        iso, confidence = resolve_time("some time ago", ANCHOR)
        assert iso == "2124-03-15T10:00:00"
        assert confidence < 0.5


class TestResolveTimeUnrecognized:
    def test_unrecognized_expression_returns_anchor_not_a_guess(self):
        iso, confidence = resolve_time("during the eclipse", ANCHOR)
        assert iso == "2124-03-15T10:00:00"
        assert 0.0 < confidence < 0.5


class TestResolveTimeNeverAnchorsToWallClock:
    def test_wall_clock_freeze_does_not_leak_into_relative_resolution(self, monkeypatch):
        """datetime.now patched to 2026-08-16 (today, per the harness) —
        resolving a relative phrase against a 2124 turn anchor must never
        produce a 2026 date. normalize.py never calls datetime.now()/utcnow()
        at all, so this test also guards against a future regression that
        introduces one."""
        import datetime as datetime_module

        class _FrozenDateTime(datetime_module.datetime):
            @classmethod
            def now(cls, tz=None):
                return cls(2026, 8, 16, 12, 0, 0, tzinfo=tz)

        monkeypatch.setattr(datetime_module, "datetime", _FrozenDateTime)

        iso, _ = resolve_time("two weeks ago", ANCHOR)
        assert "2026" not in iso
        assert iso.startswith("2124-03-01")


class TestResolveTimeSentinelDiscipline:
    """This module never produces the SENTINEL_VALID_TO string — it only
    resolves `valid_from`; the caller (extract.py) always writes the sentinel
    into valid_to separately. Documented here so the two never drift."""

    def test_resolve_time_output_is_never_the_sentinel(self):
        for expr in ("", "yesterday", "two weeks ago", "since my last visit"):
            iso, _ = resolve_time(expr, ANCHOR, prior_admission_end="2124-01-01 00:00:00")
            assert iso != SENTINEL_VALID_TO


# ---------------------------------------------------------------------------
# canonicalize_predicate
# ---------------------------------------------------------------------------


class TestCanonicalizePredicateSynonymClusters:
    @pytest.mark.parametrize(
        "phrase",
        ["is prescribed", "on medication", "takes", "is taking", "prescribed"],
    )
    def test_takes_medication_cluster(self, phrase):
        assert canonicalize_predicate(phrase) == "TAKES_MEDICATION"

    @pytest.mark.parametrize(
        "phrase", ["diagnosed with", "has history of", "suffers from"]
    )
    def test_has_condition_cluster(self, phrase):
        assert canonicalize_predicate(phrase) == "HAS_CONDITION"

    @pytest.mark.parametrize("phrase", ["allergic to", "has a reaction to", "allergy to"])
    def test_has_allergy_to_cluster(self, phrase):
        assert canonicalize_predicate(phrase) == "HAS_ALLERGY_TO"

    @pytest.mark.parametrize("phrase", ["reports symptom", "complains of", "experiencing"])
    def test_reports_symptom_cluster(self, phrase):
        assert canonicalize_predicate(phrase) == "REPORTS_SYMPTOM"

    @pytest.mark.parametrize("phrase", ["underwent", "had a procedure"])
    def test_had_procedure_cluster(self, phrase):
        assert canonicalize_predicate(phrase) == "HAD_PROCEDURE"

    @pytest.mark.parametrize("phrase", ["current dosage of", "dose of", "dosage of"])
    def test_current_dosage_of_cluster(self, phrase):
        assert canonicalize_predicate(phrase) == "CURRENT_DOSAGE_OF"

    @pytest.mark.parametrize(
        "phrase", ["primary care provider of", "primary care physician of", "pcp of"]
    )
    def test_primary_care_provider_of_cluster(self, phrase):
        assert canonicalize_predicate(phrase) == "PRIMARY_CARE_PROVIDER_OF"

    @pytest.mark.parametrize(
        "phrase",
        [
            "had a fall",  # the exact real-world miss: samples/extract-eyeball-10056223.md
            "had a fall incident",
            "had an incident",
            "experienced a fall",
            "sustained a fall",
            "fell",
            "fell down",
            "had fall",
        ],
    )
    def test_had_incident_cluster(self, phrase):
        # 2026-08-16 vocabulary-gap bug fix: previously `canonicalize_predicate`
        # returned None for every phrase here (dropped as
        # `dropped_no_predicate`); HAD_INCIDENT now covers them.
        assert canonicalize_predicate(phrase) == "HAD_INCIDENT"


class TestCanonicalizePredicateExactHits:
    def test_already_closed_vocab_string_case_insensitive(self):
        assert canonicalize_predicate("TAKES_MEDICATION") == "TAKES_MEDICATION"
        assert canonicalize_predicate("takes_medication") == "TAKES_MEDICATION"

    def test_underscore_or_space_separated_alias_both_work(self):
        assert canonicalize_predicate("is_prescribed") == "TAKES_MEDICATION"
        assert canonicalize_predicate("is prescribed") == "TAKES_MEDICATION"


class TestCanonicalizePredicateNoneCase:
    @pytest.mark.parametrize(
        "phrase",
        ["enjoys hiking", "walks the dog", "likes pizza", "went on vacation"],
    )
    def test_unrelated_phrase_returns_none(self, phrase):
        assert canonicalize_predicate(phrase) is None

    def test_empty_string_returns_none(self):
        assert canonicalize_predicate("") is None

    def test_none_input_returns_none(self):
        assert canonicalize_predicate(None) is None

    def test_whitespace_only_returns_none(self):
        assert canonicalize_predicate("   ") is None


class TestCanonicalizePredicateNeverInventsAPredicate:
    def test_return_value_is_always_a_member_of_predicates_or_none(self):
        from medmemgraph.contracts import PREDICATES

        phrases = [
            "is prescribed", "diagnosed with", "allergic to", "reports symptom",
            "underwent", "current dosage of", "primary care provider of",
            "enjoys hiking", "", None, "STOPS_MEDICATION", "CARE_PLAN",
        ]
        for phrase in phrases:
            result = canonicalize_predicate(phrase)
            assert result is None or result in PREDICATES
