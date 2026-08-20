"""tests/test_resolve.py — cross-session entity resolution (E3-S1: pipeline/
resolve.py). Written alongside the story (no separate test-lead artifact
exists in this repo's coordination model, same convention every prior
`[dev-ml]` entry in `.claude/logs/dev.log.md` has noted).

Every `complete` (the LLM-call injection seam, mirrors `Extractor(client=
fake)` / `Judge(client=fake)`) used below is a scripted stub — this file
runs fully offline, no `ANTHROPIC_API_KEY` required, same discipline as
`tests/test_extract.py` / `tests/test_reader.py`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from medmemgraph.contracts import ClinicalFact, EntityRef
from medmemgraph.pipeline.ids import mint_entity_id
from medmemgraph.pipeline.resolve import (
    MAX_CLUSTER_SIZE,
    CanonicalEntity,
    CanonicalRegistry,
    Mention,
    attach_canonical_ids,
    block,
    blocking_stats,
    match,
    resolve,
)

FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "er" / "alias_metformin.json"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fact(
    *,
    patient_id: str = "patient-0001",
    session_id: str = "admission-0001",
    turn_ids: list[int] | None = None,
    predicate: str = "TAKES_MEDICATION",
    object_name: str,
    object_type: str = "Medication",
    valid_from: str = "2124-01-01T00:00:00",
    fact_id: str | None = None,
) -> ClinicalFact:
    turn_ids = turn_ids or [1]
    return ClinicalFact(
        fact_id=fact_id or f"fact-{object_name}-{session_id}-{turn_ids[0]}",
        patient_id=patient_id,
        session_id=session_id,
        turn_ids=turn_ids,
        subject=EntityRef(name=patient_id, type="Patient", canonical_id=0),
        predicate=predicate,
        object=EntityRef(name=object_name, type=object_type, canonical_id=0),
        valid_from=valid_from,
    )


def _mention(
    name: str,
    *,
    entity_type: str = "Medication",
    patient_id: str = "patient-0001",
    session_id: str = "admission-0001",
    turn_ids: list[int] | None = None,
) -> Mention:
    return Mention(
        name=name,
        entity_type=entity_type,
        patient_id=patient_id,
        session_id=session_id,
        turn_ids=turn_ids or [1],
    )


def _load_alias_fixture() -> tuple[str, list[ClinicalFact]]:
    with open(FIXTURE_PATH, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    patient_id = data["patient_id"]
    facts = [
        _fact(
            patient_id=patient_id,
            session_id=row["session_id"],
            turn_ids=row["turn_ids"],
            predicate=row["predicate"],
            object_name=row["object_name"],
            object_type=row["object_type"],
            valid_from=row["valid_from"],
            fact_id=row["fact_id"],
        )
        for row in data["facts"]
    ]
    return patient_id, facts


_MENTION_RE = re.compile(
    r"Mention A: '(?P<a>[^']*)'.*Mention B: '(?P<b>[^']*)'", re.DOTALL
)


def _names_from_prompt(user: str) -> tuple[str, str]:
    m = _MENTION_RE.search(user)
    assert m is not None, f"could not parse mention names out of prompt: {user!r}"
    return m.group("a"), m.group("b")


def _always_match(system: str, user: str, schema: dict) -> dict:
    return {"same_entity": True, "confidence": 0.9, "reason": "scripted: always match"}


def _never_match(system: str, user: str, schema: dict) -> dict:
    return {"same_entity": False, "confidence": 0.9, "reason": "scripted: never match"}


def _raising_complete(system: str, user: str, schema: dict) -> dict:
    raise RuntimeError("scripted LLM outage")


def _scripted_pairwise(matches: dict[frozenset[str], bool]):
    """Build a `complete` stub keyed by an unordered pair of raw mention
    names — mirrors this module's own order-independent cache key so a
    test can script exactly which pairs match without caring about call
    order. Asking about an unscripted pair is a test bug, not a silent
    default — raise loudly."""

    def _complete(system: str, user: str, schema: dict) -> dict:
        a, b = _names_from_prompt(user)
        key = frozenset((a, b))
        if key not in matches:
            raise AssertionError(f"unscripted pair asked: {a!r} vs {b!r}")
        same = matches[key]
        return {
            "same_entity": same,
            "confidence": 0.9 if same else 0.05,
            "reason": f"scripted: {a!r} vs {b!r} -> {same}",
        }

    return _complete


def _counting(complete):
    calls: list[tuple[str, str]] = []

    def _wrapped(system: str, user: str, schema: dict) -> dict:
        calls.append(_names_from_prompt(user))
        return complete(system, user, schema)

    _wrapped.calls = calls  # type: ignore[attr-defined]
    return _wrapped


# ---------------------------------------------------------------------------
# block() — cheap candidate generation
# ---------------------------------------------------------------------------


class TestBlock:
    def test_obvious_case_and_punctuation_variants_land_in_one_block(self):
        mentions = [
            _mention("Metformin"),
            _mention("metformin."),
            _mention("  METFORMIN  "),
        ]
        blocks = block(mentions)
        assert len(blocks) == 1
        assert len(blocks[0]) == 3

    def test_token_overlap_variant_lands_with_its_neighbour(self):
        mentions = [
            _mention("metformin 500mg tablet"),
            _mention("metformin 500mg"),
        ]
        blocks = block(mentions)
        assert len(blocks) == 1

    def test_known_brand_generic_pair_blocks_together_via_gazetteer(self):
        # The one bridge this module's documented gazetteer explicitly
        # provides (module docstring: bounded stand-in for a missing
        # embedding backend) — direct evidence it fires.
        mentions = [_mention("metformin"), _mention("Glucophage")]
        blocks = block(mentions)
        assert len(blocks) == 1

    def test_unrelated_medications_land_in_different_blocks(self):
        mentions = [_mention("metformin"), _mention("furosemide")]
        blocks = block(mentions)
        assert len(blocks) == 2

    def test_never_crosses_patient(self):
        mentions = [
            _mention("metformin", patient_id="patient-A"),
            _mention("metformin", patient_id="patient-B"),
        ]
        blocks = block(mentions)
        assert len(blocks) == 2, "identical name, different patient — must never share a block"

    def test_never_crosses_entity_type(self):
        mentions = [
            _mention("ozempic", entity_type="Medication"),
            _mention("ozempic", entity_type="Condition"),
        ]
        blocks = block(mentions)
        assert len(blocks) == 2

    def test_documented_gap_colloquial_description_does_not_block_with_generic(self):
        """Honest negative case (module docstring's stated limitation): a
        purely descriptive reference with no lexical/gazetteer overlap
        does NOT block against the drug it colloquially refers to. This is
        not a bug to silently patch — it is the real, reported boundary of
        a no-new-dependency lexical blocker; see docs/algorithms/
        entity-resolution.md and the real-corpus run for the honest
        write-up."""
        mentions = [_mention("metformin"), _mention("the 500mg one")]
        blocks = block(mentions)
        assert len(blocks) == 2

    def test_empty_input(self):
        assert block([]) == []


class TestBlockingStats:
    def test_reduction_ratio_is_positive_when_blocking_helps(self):
        mentions = (
            [_mention("metformin") for _ in range(5)]
            + [_mention("furosemide") for _ in range(5)]
            + [_mention("spironolactone") for _ in range(5)]
        )
        blocks = block(mentions)
        stats = blocking_stats(mentions, blocks)
        assert stats["n_mentions"] == 15
        assert stats["pairs_naive"] == 15 * 14 // 2
        assert stats["pairs_after_blocking"] < stats["pairs_naive"]
        assert 0.0 < stats["reduction_ratio"] < 1.0

    def test_empty_mentions_reduction_ratio_is_zero_not_a_crash(self):
        stats = blocking_stats([], [])
        assert stats["reduction_ratio"] == 0.0
        assert stats["pairs_naive"] == 0


# ---------------------------------------------------------------------------
# match() — LLM adjudication, cached, LLM-failure-safe
# ---------------------------------------------------------------------------


class TestMatch:
    def test_exact_normalized_key_never_calls_the_llm(self):
        a, b = _mention("Metformin"), _mention("metformin.")

        def _boom(system, user, schema):
            raise AssertionError("must not call the LLM for an exact-key pair")

        matched, confidence, reason = match(a, b, complete=_boom)
        assert matched is True
        assert confidence == 1.0
        assert "exact" in reason

    def test_cross_patient_never_matches_without_calling_the_llm(self):
        a = _mention("metformin", patient_id="patient-A")
        b = _mention("metformin", patient_id="patient-B")

        def _boom(system, user, schema):
            raise AssertionError("must not call the LLM for a cross-patient pair")

        matched, confidence, reason = match(a, b, complete=_boom)
        assert matched is False
        assert confidence == 0.0
        assert "cross-patient" in reason

    def test_fuzzy_pair_calls_the_scripted_complete(self):
        a, b = _mention("acme drug variant one"), _mention("acme drug variant two")
        stub = _counting(_always_match)
        matched, confidence, reason = match(a, b, complete=stub)
        assert matched is True
        assert confidence == 0.9
        assert len(stub.calls) == 1

    def test_complete_raising_is_caught_and_returns_no_match(self):
        a, b = _mention("acme drug variant one"), _mention("acme drug variant two")
        matched, confidence, reason = match(a, b, complete=_raising_complete)
        assert matched is False
        assert confidence == 0.0
        assert "unavailable" in reason or "failed" in reason

    def test_cache_avoids_a_second_llm_call_for_the_same_pair(self):
        a, b = _mention("acme drug variant one"), _mention("acme drug variant two")
        stub = _counting(_always_match)
        cache: dict = {}
        match(a, b, complete=stub, cache=cache)
        match(a, b, complete=stub, cache=cache)
        match(b, a, complete=stub, cache=cache)  # order-independent key
        assert len(stub.calls) == 1

    def test_no_llm_configured_and_no_complete_degrades_to_no_match(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        a, b = _mention("acme drug variant one"), _mention("acme drug variant two")
        matched, confidence, reason = match(a, b)
        assert matched is False


# ---------------------------------------------------------------------------
# resolve() — transitivity guard (dispatch: "construct one deliberately")
# ---------------------------------------------------------------------------


class TestTransitivityGuard:
    def test_representative_anchoring_stops_a_bad_chain_merge(self):
        """A = 'acme drug variant one', B = '...variant two', C = '...variant
        three'. Scripted: A-B match, A-C do NOT match, B-C WOULD match if
        ever asked. Naive connected-components (chain through B) would
        merge {A, B, C} into one entity. Representative-anchored clustering
        must not: C is only ever compared against the cluster's fixed
        representative (A), never against B, so it correctly forms its own
        canonical entity."""
        a = _mention("acme drug variant one")
        b = _mention("acme drug variant two")
        c = _mention("acme drug variant three")

        # Sanity check the blocking premise: all three land in one block
        # (required for this to be a meaningful test of the *clustering*
        # guard rather than an accidental blocking miss).
        assert len(block([a, b, c])) == 1

        matches = {
            frozenset({a.name, b.name}): True,
            frozenset({a.name, c.name}): False,
            frozenset({b.name, c.name}): True,  # never actually asked — see assertion below
        }
        stub = _counting(_scripted_pairwise(matches))

        canonicals = resolve([a, b, c], complete=stub, id_map={})

        assert len(canonicals) == 2
        sizes = sorted(len(e.aliases) for e in canonicals)
        assert sizes == [1, 2], "expected {A,B} merged and {C} separate, not all three merged"

        # The guard's whole point: B vs C is never actually asked, because
        # C is only ever compared against the cluster's representative (A).
        asked_pairs = {frozenset(pair) for pair in stub.calls}
        assert frozenset({b.name, c.name}) not in asked_pairs

    def test_max_cluster_size_caps_fuzzy_growth_but_not_exact_matches(self):
        # Fuzzy case: MAX_CLUSTER_SIZE+2 distinct-but-always-matching mentions
        # must split into more than one cluster once the FUZZY-JOIN cap binds
        # (the representative itself plus MAX_CLUSTER_SIZE fuzzy joins = a
        # first-cluster ceiling of MAX_CLUSTER_SIZE+1 members — the cap counts
        # joins, not the representative).
        mentions = [_mention(f"acme drug variant {i}") for i in range(MAX_CLUSTER_SIZE + 2)]
        canonicals = resolve(mentions, complete=_always_match, id_map={})
        assert len(canonicals) >= 2
        assert all(len(e.aliases) <= MAX_CLUSTER_SIZE + 1 for e in canonicals)
        assert sum(len(e.aliases) for e in canonicals) == MAX_CLUSTER_SIZE + 2

        # Exact-key case: many identical mentions of the same real drug
        # (a realistic chronic-medication scenario) must NOT be split by
        # the cap — see resolve.py's own comment on why this exemption
        # exists (a false split would recreate the duplicate-node problem
        # ER exists to prevent).
        identical = [_mention("metformin") for _ in range(MAX_CLUSTER_SIZE + 5)]

        def _boom(system, user, schema):
            raise AssertionError("exact-key mentions must never reach the LLM")

        canonicals = resolve(identical, complete=_boom, id_map={})
        assert len(canonicals) == 1
        assert len(canonicals[0].mentions) == MAX_CLUSTER_SIZE + 5

    def test_cap_counts_fuzzy_joins_not_total_membership_regression(self):
        """Real-corpus bug, found by actually running
        `scripts/generate_resolve_eyeball.py` against 64 real admissions,
        fixed here (see resolve.py's `resolve()` comment at the fuzzy-join
        cap for the full story) — regression-locked so it cannot silently
        come back. A cluster that has already grown past MAX_CLUSTER_SIZE
        via harmless EXACT matches (a common medication mentioned very
        often) must still have a single later FUZZY candidate compared
        against it — the cap must bind on fuzzy-join count, never on raw
        total membership, or a frequently-mentioned entity could never
        absorb a late paraphrase/brand-name alias. That is exactly the
        story's own headline under-merge failure mode, reproduced by the
        wrong mechanism if this regresses."""
        many_exact = [_mention("metformin") for _ in range(MAX_CLUSTER_SIZE + 20)]
        # Shares the token "metformin" with the exact-match block (token-
        # Jaccard 0.5, clears BLOCK_SIMILARITY_THRESHOLD) so it lands in the
        # SAME block and actually exercises the fuzzy path — but is not
        # exact-key-equal (must go through match(), not the exact fast path).
        one_fuzzy_alias = _mention("metformin xr")
        mentions = many_exact + [one_fuzzy_alias]

        matches = {frozenset({"metformin", one_fuzzy_alias.name}): True}
        stub = _counting(_scripted_pairwise(matches))

        canonicals = resolve(mentions, complete=stub, id_map={})

        # The fuzzy candidate MUST have actually been asked about (not
        # silently skipped due to the exact-grown cluster's raw size).
        assert len(stub.calls) == 1
        assert len(canonicals) == 1
        assert one_fuzzy_alias.name in canonicals[0].aliases


# ---------------------------------------------------------------------------
# Cross-patient isolation (AC2)
# ---------------------------------------------------------------------------


class TestCrossPatientNeverMerges:
    def test_same_name_different_patients_get_different_canonical_ids(self):
        a = _mention("metformin", patient_id="patient-A")
        b = _mention("metformin", patient_id="patient-B")

        def _boom(system, user, schema):
            raise AssertionError("must never call the LLM across patients")

        canonicals = resolve([a, b], complete=_boom, id_map={})
        assert len(canonicals) == 2
        ids = {e.canonical_id for e in canonicals}
        assert len(ids) == 2


# ---------------------------------------------------------------------------
# AC4 — complete raising creates a new canonical, never a wrong merge
# ---------------------------------------------------------------------------


class TestCompleteRaisesNoMerge:
    def test_llm_failure_on_a_fuzzy_candidate_yields_two_canonicals_not_one(self):
        a = _mention("acme drug variant one")
        b = _mention("acme drug variant two")
        canonicals = resolve([a, b], complete=_raising_complete, id_map={})
        assert len(canonicals) == 2, "an LLM failure must never silently merge"


# ---------------------------------------------------------------------------
# Ids stable on re-run (AC3 / dispatch)
# ---------------------------------------------------------------------------


class TestIdsStableOnRerun:
    def test_same_mentions_same_fresh_id_map_mint_the_same_id_twice(self):
        mentions = [_mention("metformin"), _mention("furosemide")]

        id_map_1: dict = {}
        canonicals_1 = resolve(mentions, complete=_never_match, id_map=id_map_1)

        # Simulate "fresh id_map loaded from the first run" (AC3's own
        # wording) by re-running against a NEW dict seeded from run 1's
        # persisted map, not the same live object.
        id_map_2 = dict(id_map_1)
        canonicals_2 = resolve(mentions, complete=_never_match, id_map=id_map_2)

        ids_1 = {e.canonical_name: e.canonical_id for e in canonicals_1}
        ids_2 = {e.canonical_name: e.canonical_id for e in canonicals_2}
        assert ids_1 == ids_2


# ---------------------------------------------------------------------------
# Incremental: new admission does not renumber existing entities
# ---------------------------------------------------------------------------


class TestIncrementalResolution:
    def test_second_admission_does_not_renumber_first_admissions_entities(self):
        registry = CanonicalRegistry()
        id_map: dict = {}

        admission_1_facts = [
            _fact(session_id="adm-01", object_name="metformin", turn_ids=[1]),
            _fact(session_id="adm-01", object_name="furosemide", turn_ids=[2]),
        ]
        attach_canonical_ids(
            admission_1_facts, registry=registry, complete=_never_match, id_map=id_map
        )
        metformin_id_v1 = admission_1_facts[0].object.canonical_id
        furosemide_id_v1 = admission_1_facts[1].object.canonical_id
        patient_id = admission_1_facts[0].patient_id
        assert {e.canonical_id for e in registry.get(patient_id)} == {
            metformin_id_v1,
            furosemide_id_v1,
        }

        # A second, later admission: repeats metformin under an exact-key
        # variant, and introduces a genuinely new medication.
        admission_2_facts = [
            _fact(session_id="adm-02", object_name="Metformin.", turn_ids=[9]),
            _fact(session_id="adm-02", object_name="aspirin", turn_ids=[11]),
        ]
        attach_canonical_ids(
            admission_2_facts, registry=registry, complete=_never_match, id_map=id_map
        )

        # Admission 1's own fact objects (not touched by call 2 at all) —
        # their ids must be byte-for-byte identical to before call 2 ran.
        assert admission_1_facts[0].object.canonical_id == metformin_id_v1
        assert admission_1_facts[1].object.canonical_id == furosemide_id_v1

        # Admission 2's repeat mention of the SAME drug must resolve onto
        # the SAME (not a new) canonical id.
        assert admission_2_facts[0].object.canonical_id == metformin_id_v1

        # The registry's own record of furosemide is untouched/not renumbered.
        furosemide_entries = [
            e for e in registry.get(patient_id) if e.canonical_name == "furosemide"
        ]
        assert len(furosemide_entries) == 1
        assert furosemide_entries[0].canonical_id == furosemide_id_v1

        # A brand-new entity (aspirin) was minted, not folded into anything.
        assert admission_2_facts[1].object.canonical_id not in {metformin_id_v1, furosemide_id_v1}


# ---------------------------------------------------------------------------
# Aliases accumulate onto one canonical id (dispatch)
# ---------------------------------------------------------------------------


class TestAliasesAccumulate:
    def test_repeated_exact_key_and_gazetteer_variants_all_land_on_one_canonical(self):
        mentions = [
            _mention("metformin"),
            _mention("Metformin."),
            _mention("Glucophage"),
            _mention("  METFORMIN  "),
        ]
        canonicals = resolve(mentions, complete=_always_match, id_map={})
        assert len(canonicals) == 1
        entity = canonicals[0]
        assert set(entity.aliases) == {"metformin", "Metformin.", "Glucophage", "  METFORMIN  "}
        assert len(entity.mentions) == 4


# ---------------------------------------------------------------------------
# The alias fixture (E3-S1.md AC1's worked example) — run through
# attach_canonical_ids end to end, honest about what does and does not merge.
# ---------------------------------------------------------------------------


class TestAliasFixtureEndToEnd:
    def test_metformin_glucophage_and_case_variant_share_one_canonical(self):
        """AC1's worked example, as far as a no-new-dependency lexical
        blocker (module docstring) can honestly take it: 'metformin',
        'Glucophage' (gazetteer-bridged), and 'METFORMIN.' (exact-key
        variant) all land on ONE canonical id, with all three (plus the
        distinct 'furosemide') from the fixture accounted for."""
        patient_id, facts = _load_alias_fixture()
        registry = CanonicalRegistry()
        attach_canonical_ids(facts, registry=registry, complete=_always_match, id_map={})

        by_name = {f.fact_id: f for f in facts}
        metformin_id = by_name["fixture-0001"].object.canonical_id
        glucophage_id = by_name["fixture-0002"].object.canonical_id
        case_variant_id = by_name["fixture-0004"].object.canonical_id
        furosemide_id = by_name["fixture-0005"].object.canonical_id

        assert metformin_id == glucophage_id == case_variant_id
        assert furosemide_id != metformin_id

        entities = registry.get(patient_id)
        merged = next(e for e in entities if e.canonical_id == metformin_id)
        assert {"metformin", "Glucophage", "METFORMIN."}.issubset(set(merged.aliases))

    def test_colloquial_description_is_the_documented_gap_not_a_silent_merge(self):
        """The one fixture case this design honestly does NOT resolve: 'the
        500mg one' never blocks against any metformin variant (no lexical
        or gazetteer overlap), so it ends up as its own, separate canonical
        entity — reported here explicitly, not hidden, per the story's own
        instruction to 'say plainly where it over- or under-merged.'"""
        patient_id, facts = _load_alias_fixture()
        registry = CanonicalRegistry()
        attach_canonical_ids(facts, registry=registry, complete=_always_match, id_map={})

        by_name = {f.fact_id: f for f in facts}
        metformin_id = by_name["fixture-0001"].object.canonical_id
        colloquial_id = by_name["fixture-0003"].object.canonical_id
        assert colloquial_id != metformin_id


# ---------------------------------------------------------------------------
# attach_canonical_ids — Patient subject minted directly, no ER
# ---------------------------------------------------------------------------


class TestAttachCanonicalIds:
    def test_patient_subject_gets_a_canonical_id_without_any_llm_call(self):
        fact = _fact(object_name="metformin")
        assert fact.subject.canonical_id == 0

        def _boom(system, user, schema):
            raise AssertionError("Patient identity must never go through match()")

        attach_canonical_ids([fact], complete=_boom, id_map={})
        assert fact.subject.canonical_id > 0
        assert fact.object.canonical_id > 0

    def test_two_facts_sharing_an_object_mention_get_the_same_canonical_id(self):
        f1 = _fact(session_id="adm-01", object_name="metformin", turn_ids=[1])
        f2 = _fact(session_id="adm-02", object_name="metformin", turn_ids=[9])
        attach_canonical_ids([f1, f2], complete=_never_match, id_map={})
        assert f1.object.canonical_id == f2.object.canonical_id

    def test_matches_the_directly_minted_entity_id(self):
        """`attach_canonical_ids`'s canonical_id for an unambiguous single
        mention must equal calling `ids.mint_entity_id` directly with that
        mention's own (patient_id, type, name) — resolve.py must not
        invent a second, divergent id scheme."""
        fact = _fact(patient_id="patient-0099", object_name="metformin")
        id_map: dict = {}
        attach_canonical_ids([fact], complete=_never_match, id_map=id_map)
        expected = mint_entity_id("patient-0099", "Medication", "metformin", id_map={})
        assert fact.object.canonical_id == expected


# ---------------------------------------------------------------------------
# CanonicalRegistry round-trip
# ---------------------------------------------------------------------------


class TestCanonicalRegistryRoundTrip:
    def test_to_dict_from_dict_round_trip(self):
        entity = CanonicalEntity(
            canonical_id=42,
            canonical_name="metformin",
            entity_type="Medication",
            patient_id="patient-0001",
            aliases=["metformin", "Glucophage"],
            mentions=[_mention("metformin"), _mention("Glucophage")],
        )
        registry = CanonicalRegistry()
        registry.replace_patient("patient-0001", [entity])

        restored = CanonicalRegistry.from_dict(registry.to_dict())
        restored_entities = restored.get("patient-0001")
        assert len(restored_entities) == 1
        assert restored_entities[0].canonical_id == 42
        assert restored_entities[0].aliases == ["metformin", "Glucophage"]
        assert len(restored_entities[0].mentions) == 2

    def test_save_and_load_json(self, tmp_path):
        entity = CanonicalEntity(
            canonical_id=7,
            canonical_name="furosemide",
            entity_type="Medication",
            patient_id="patient-0002",
        )
        registry = CanonicalRegistry()
        registry.replace_patient("patient-0002", [entity])
        path = tmp_path / "registry.json"
        registry.save_json(path)

        loaded = CanonicalRegistry.load_json(path)
        assert loaded.get("patient-0002")[0].canonical_id == 7

    def test_load_missing_file_returns_empty_registry(self, tmp_path):
        loaded = CanonicalRegistry.load_json(tmp_path / "does-not-exist.json")
        assert loaded.all() == []


# ---------------------------------------------------------------------------
# AC5 — no SAME_AS relationship type anywhere in the source tree
# ---------------------------------------------------------------------------


_SAME_AS_LITERAL_RE = re.compile(r"""["']SAME_AS["']|:\s*SAME_AS\b""")
"""Matches SAME_AS used as an actual value — a quoted string literal (e.g.
a REL_TYPES entry or a Cypher parameter) or Cypher's `:SAME_AS` relationship-
type token — but NOT a backtick-quoted prose mention inside a docstring/
comment explaining why it is deliberately absent (this module's own module
docstring, and `graph/schema.py`'s "Do NOT add `SAME_AS`" comment, both
legitimately name the string in prose; a blind substring search would
false-positive on exactly the sentences documenting the ban)."""


class TestNoSameAsRelationshipType:
    def test_source_tree_has_no_same_as_relationship_type(self):
        src_root = Path(__file__).resolve().parents[1] / "src" / "medmemgraph"
        offenders = []
        for path in src_root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if _SAME_AS_LITERAL_RE.search(text):
                offenders.append(str(path))
        assert offenders == [], f"SAME_AS used as a real value/rel-type in: {offenders}"

    def test_prose_mentions_of_same_as_are_documenting_the_ban_not_using_it(self):
        """Sanity-checks the regex above isn't just failing to match at all
        — confirms it correctly distinguishes prose from a real literal."""
        assert _SAME_AS_LITERAL_RE.search("Do NOT add `SAME_AS` edges.") is None
        assert _SAME_AS_LITERAL_RE.search('REL_TYPES = {"SAME_AS"}') is not None
        assert _SAME_AS_LITERAL_RE.search("MATCH (a)-[:SAME_AS]->(b)") is not None
