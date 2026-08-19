"""pipeline/resolve.py — cross-session entity resolution: the same real-world
entity ("metformin" / "Glucophage" / "the 500mg one") collapses onto ONE
canonical node per patient, never across patients.

Design authority: ARCHITECTURE.md §6.3 (Entity resolution) and §5.4
(deterministic ids), `collaborative/literature/04-entity-resolution-and-
ontology.md` ("survey 04"), `collaborative/design/stories/E3/E3-S1.md`.

Pipeline shape (ARCHITECTURE §6.3, survey 04 Q1, Zep-shaped, no trained
matcher — R-ER-04/05/06: zero-shot LLM matching is within a few F1 points of
a fine-tuned matcher and does not suffer the 22-61 F1 out-of-distribution
collapse a fine-tuned matcher shows on unseen entities, which is exactly
what a memory system meeting new entities every session would hit):

    ClinicalFacts -> Mentions -> block() -> match() within each block
    -> resolve() clusters (transitivity-guarded) -> CanonicalEntity
    -> ids.mint_entity_id -> attach_canonical_ids() rewrites facts in place

One canonical node per entity, aliases folded into a list property
(`mentioned_as`), NOT alias nodes joined by `SAME_AS` edges (ARCHITECTURE
§5.2, survey 04 Q3 / R-ER-22: this is Zep's own evaluated production
pattern, and the reason is architectural — a `SAME_AS` subgraph forces every
downstream traversal to first collapse through those edges, which is
expensive in an engine with a 16-hop ceiling and no `IN`). This module does
not decide graph representation — it hands the Graph writer a `canonical_id`
per entity and a `mentioned_as` alias LIST (see `CanonicalEntity.aliases`);
serializing that list into HydraDB's delimited-string property is the
Graph's job (E3-S1.md's own Boundary note), not this module's.

--------------------------------------------------------------------------
Reconciling two spec documents for this exact story (stated here, not
silently picked, per this project's established convention — see the
`[dev-ml]` E2/E7-S5 entries in `.claude/logs/dev.log.md`):

  * This story's own direct dispatch (my authoritative context) asks for a
    four-function API — `block(mentions) -> list[list[Mention]]`,
    `match(a, b) -> (bool, float, str)`, `resolve(mentions) ->
    list[CanonicalEntity]`, `attach_canonical_ids(facts) ->
    list[ClinicalFact]` — with blocking built from "normalized string,
    token overlap, character n-gram / MinHash-ish" (no embedding call
    named).
  * The on-disk `E3-S1.md` / ARCHITECTURE §6.3 describe the same pipeline
    at a different altitude and additionally specify an **embedding-cosine
    kNN blocking signal** over "name + one-turn context" as the second
    blocking pass, specifically to catch the paraphrase/brand-name case
    ("metformin"/"Glucophage") that no string key can ever catch — this is
    a linguistic fact, not a tuning choice: brand and generic drug names
    share no substrings, so no purely lexical similarity metric can bridge
    them.

Followed the direct dispatch's literal function set, and did NOT add a real
embedding call: this repo has no embedding client, no embedding dependency
in `pyproject.toml`, and no `ANTHROPIC_API_KEY` configured in this sandbox
(same finding every prior `[dev-ml]` entry in this log has made) — Anthropic
does not itself expose an embeddings endpoint (checked the `claude-api`
skill directly, not from memory, per this repo's own trigger rule for any
Anthropic-API code: zero hits for "embed" in that skill's docs). Adding one
would mean a new dependency this story was not scoped for (a HALT trigger
per the hard rules), not a same-story implementation choice — flagging this
for `dev-architect`/a future story rather than quietly reaching for one.

This is the exact same trade `normalize.py` already made and documented for
predicate canonicalization (EDC embed+LLM-confirm traded for a pure alias
table) — same precedent, same honesty discipline: the resulting gap is
real, and it is reported plainly, not hidden. Where pure lexical blocking
provably cannot bridge a brand/generic or purely descriptive pair, this
module adds exactly ONE additional non-ML, zero-parameter, hand-reviewable
signal (`_KNOWN_ALIAS_GROUPS`, a short static gazetteer of well-known
brand<->generic pairs for this corpus's own documented medication lexicon,
same category of artifact as `normalize._PREDICATE_ALIASES`) as a scoped
stand-in for the missing embedding backend. It is bounded and audited: it
catches only the finite pairs listed, and it does NOT generalize to open
colloquial descriptions ("my diabetes pill", "the 500mg one") — that gap is
real, documented here, and reported plainly in the real-corpus run (see
`docs/algorithms/entity-resolution.md` and the dev.log return note) rather
than glossed over. If/when a real embedding backend is wired in, `resolve()`
and `block()` both already thread an `embed` callable through so the
gazetteer becomes redundant rather than load-bearing.

`match()` follows ARCHITECTURE §6.3 point 3's *pairwise* branch (not the
listwise branch it also names for blocks > 3-4 candidates, R-ER-10's ~16 F1
listwise gain) — direct dispatch's literal `match(a, b)` signature is
pairwise, and pairwise is what `resolve()`'s representative-anchored
clustering (below) is built on specifically because it is also the
transitivity guard. Cost is bounded the same way this project always bounds
LLM cost: an exact-normalized-key short-circuit that never calls the LLM at
all, aggressive caching (by normalized pair key, reusable across the whole
resolve() call and, if the caller persists the cache dict, across future
incremental runs too), and the alias gazetteer resolving the corpus's own
worked example without any LLM call.

--------------------------------------------------------------------------
Transitivity guard (dispatch: "Connected-components clustering has a
well-known transitivity failure mode. Guard against it."):

Naive union-find over pairwise match edges lets A-matches-B and B-matches-C
merge {A, B, C} even when A and C are NOT the same entity — a real risk
once similarity thresholds are lenient enough to catch real paraphrases.
This module never does connected-components. Instead, each block resolves
via single-pass **representative-anchored clustering**: the first mention
opens a cluster as its representative; every later candidate in the block
is matched only against each open cluster's REPRESENTATIVE, never against
an arbitrary existing member — so a mention joins a cluster only on direct
pairwise support with the anchor, not by chaining through an intermediate
member (`MAX_CLUSTER_SIZE` below is a second, independent safety cap for
the same failure mode, per the dispatch's "cap cluster size and/or require
pairwise support" — both are implemented, not either/or).
--------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Callable, Literal, MutableMapping, Sequence

from medmemgraph import llm
from medmemgraph.contracts import ClinicalFact
from medmemgraph.pipeline.ids import IdMap, IdMinter, mint_entity_id, mint_patient_id, normalize_key

__all__ = [
    "Mention",
    "CanonicalEntity",
    "CanonicalRegistry",
    "MAX_CLUSTER_SIZE",
    "BLOCK_SIMILARITY_THRESHOLD",
    "block",
    "blocking_stats",
    "match",
    "resolve",
    "attach_canonical_ids",
]

logger = logging.getLogger(__name__)

Role = Literal["subject", "object"]

# ---------------------------------------------------------------------------
# Mention / CanonicalEntity
# ---------------------------------------------------------------------------


@dataclass
class Mention:
    """One raw surface-form appearance of an entity, carrying enough
    provenance to rewrite the `ClinicalFact` it came from.

    `context` is a documented hook for a future embedding signal ("name +
    one-turn context", ARCHITECTURE §6.3 point 2) — not populated by
    `attach_canonical_ids` today, since `ClinicalFact` carries only bare
    `turn_ids` ints, not turn text (§5.3); left `""` by default and never
    required by any function in this module.

    `source_canonical` is set only for the internal pseudo-mentions
    `resolve()` synthesizes from `existing` canonicals (see `resolve()`'s
    docstring) — never set on a real mention derived from a `ClinicalFact`.
    """

    name: str
    entity_type: str
    patient_id: str
    session_id: str = ""
    turn_ids: list[int] = field(default_factory=list)
    fact_index: int | None = None
    role: Role | None = None
    context: str = ""
    source_canonical: "CanonicalEntity | None" = None


@dataclass
class CanonicalEntity:
    """One canonical node's worth of state: `canonical_id` from
    `ids.mint_entity_id`, the chosen `canonical_name`, every surface form
    ever seen (`aliases`, de-duplicated, insertion-ordered — the
    `mentioned_as` set), and the full turn provenance of every mention that
    merged into it.

    `canonical_name` is chosen once (the first-seen mention's raw name in
    the input order given) and never changes on a later incremental call —
    load-bearing for id stability: `ids.mint_entity_id` hashes on
    `normalize_key(canonical_name)`, so changing the chosen name would
    change the minted id and violate "ids are stable on re-run" /
    "incremental resolution does not renumber existing entities".
    """

    canonical_id: int
    canonical_name: str
    entity_type: str
    patient_id: str
    aliases: list[str] = field(default_factory=list)
    mentions: list[Mention] = field(default_factory=list)

    def add_alias(self, name: str) -> None:
        if name not in self.aliases:
            self.aliases.append(name)

    def to_dict(self) -> dict:
        return {
            "canonical_id": self.canonical_id,
            "canonical_name": self.canonical_name,
            "entity_type": self.entity_type,
            "patient_id": self.patient_id,
            "aliases": list(self.aliases),
            "mentions": [
                {
                    "name": m.name,
                    "entity_type": m.entity_type,
                    "patient_id": m.patient_id,
                    "session_id": m.session_id,
                    "turn_ids": list(m.turn_ids),
                    "fact_index": m.fact_index,
                    "role": m.role,
                }
                for m in self.mentions
                if m.source_canonical is None  # never persist a pseudo-mention
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CanonicalEntity":
        return cls(
            canonical_id=data["canonical_id"],
            canonical_name=data["canonical_name"],
            entity_type=data["entity_type"],
            patient_id=data["patient_id"],
            aliases=list(data.get("aliases", [])),
            mentions=[
                Mention(
                    name=m["name"],
                    entity_type=m["entity_type"],
                    patient_id=m["patient_id"],
                    session_id=m.get("session_id", ""),
                    turn_ids=list(m.get("turn_ids", [])),
                    fact_index=m.get("fact_index"),
                    role=m.get("role"),
                )
                for m in data.get("mentions", [])
            ],
        )


# ---------------------------------------------------------------------------
# CanonicalRegistry — the "per-patient canonical registry that can be loaded
# and extended" the story asks for. Incremental behaviour lives here: pass
# `registry.get(patient_id)` in as `resolve()`'s `existing=`, and the
# returned canonicals (mutated in place for matches, new ones appended for
# misses) are what the caller should persist back for the next admission.
# ---------------------------------------------------------------------------


@dataclass
class CanonicalRegistry:
    """`patient_id -> list[CanonicalEntity]`, JSON-round-trippable so it can
    be loaded fresh at the start of a run and saved back at the end,
    without ever reprocessing admissions 1..N to resolve admission N+1
    (dispatch: "resolving admission N+1 must not require reprocessing
    1..N")."""

    by_patient: dict[str, list[CanonicalEntity]] = field(default_factory=dict)

    def get(self, patient_id: str) -> list[CanonicalEntity]:
        return self.by_patient.get(patient_id, [])

    def all(self) -> list[CanonicalEntity]:
        out: list[CanonicalEntity] = []
        for entities in self.by_patient.values():
            out.extend(entities)
        return out

    def replace_patient(self, patient_id: str, entities: list[CanonicalEntity]) -> None:
        self.by_patient[patient_id] = entities

    def to_dict(self) -> dict:
        return {
            patient_id: [e.to_dict() for e in entities]
            for patient_id, entities in self.by_patient.items()
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CanonicalRegistry":
        return cls(
            by_patient={
                patient_id: [CanonicalEntity.from_dict(e) for e in entities]
                for patient_id, entities in data.items()
            }
        )

    def save_json(self, path: str | os.PathLike[str]) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load_json(cls, path: str | os.PathLike[str]) -> "CanonicalRegistry":
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return cls.from_dict(json.load(fh))
        except FileNotFoundError:
            return cls()


# ---------------------------------------------------------------------------
# Blocking — cheap candidate generation before any LLM call.
# ---------------------------------------------------------------------------

BLOCK_SIMILARITY_THRESHOLD = 0.5
"""Similarity floor (max of token-Jaccard, char-trigram Dice) for two
mentions to land in the same block when their normalized keys are not
identical. Deliberately generous (recall over precision) — a false-positive
block placement only costs one extra `match()` call/cache lookup; a
false-negative means two real mentions of the same entity are never even
compared, which is the failure `attach_canonical_ids` cannot recover from
downstream (survey 05 gap ARCHITECTURE §6.3 names: "failed ER degrades to
new entity" — that degrade is only safe if blocking, at minimum, is
recall-generous)."""

MAX_CLUSTER_SIZE = 12
"""Hard cap on how many FUZZY (LLM-adjudicated, non-exact-key) joins a
single cluster may accept within one `resolve()` call — the second,
independent transitivity-guard mechanism (module docstring). Deliberately
NOT a cap on total membership: an unbounded run of exact-key joins (e.g. the
same drug mentioned verbatim across 30 admissions) carries zero transitivity
risk and must never be blocked by this cap, or a common entity would be
unable to ever absorb one later paraphrase/alias candidate purely because
its mention count is high — see the real-corpus bug this exact scenario
caught, documented at the cap's call site in `resolve()`. Twelve fuzzy joins
is a headroom-not-a-tight-fit choice: a real patient's medication list
rarely exceeds single digits of distinct entities, so this should
essentially never bind an honestly-matched cluster, only a runaway one."""

_KNOWN_ALIAS_GROUPS: dict[str, str] = {
    # normalized brand/colloquial name -> normalized generic/canonical group
    # key. Deliberately small and hand-reviewable (see module docstring for
    # why this exists and what it does not do). Scoped to this corpus's own
    # documented medication lexicon (extract.py's rule-based-fallback
    # `_MEDICATIONS` set) plus their most common US brand names.
    "glucophage": "metformin",
    "lasix": "furosemide",
    "aldactone": "spironolactone",
    "lexapro": "escitalopram",
    "asacol": "mesalamine",
    "pentasa": "mesalamine",
    "kristalose": "lactulose",
    "enulose": "lactulose",
    "xifaxan": "rifaximin",
    "cipro": "ciprofloxacin",
    "novolin": "insulin",
    "humalog": "insulin",
    "lantus": "insulin",
    "prinivil": "lisinopril",
    "zestril": "lisinopril",
    "lipitor": "atorvastatin",
    "aspirin low dose": "aspirin",
    "amoxil": "amoxicillin",
    "coumadin": "warfarin",
    "jantoven": "warfarin",
    "pen vk": "penicillin",
    "deltasone": "prednisone",
    "ventolin": "albuterol",
    "proventil": "albuterol",
    "prilosec": "omeprazole",
    "lopressor": "metoprolol",
    "toprol": "metoprolol",
    "cozaar": "losartan",
    "zoloft": "sertraline",
    "neurontin": "gabapentin",
    "microzide": "hydrochlorothiazide",
}


def _token_set(name: str) -> set[str]:
    return set(normalize_key(name).split())


def _char_trigrams(name: str) -> set[str]:
    s = normalize_key(name).replace(" ", "")
    if len(s) < 3:
        return {s} if s else set()
    return {s[i : i + 3] for i in range(len(s) - 2)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _dice(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return 2 * len(a & b) / (len(a) + len(b))


def _blocking_group_key(mention: Mention) -> str:
    """Normalized name, folded through the small alias gazetteer so a known
    brand name lands on the same blocking key as its generic (module
    docstring's documented, bounded gazetteer)."""
    normalized = normalize_key(mention.name)
    return _KNOWN_ALIAS_GROUPS.get(normalized, normalized)


def _similar(a: Mention, b: Mention) -> bool:
    if _blocking_group_key(a) == _blocking_group_key(b):
        return True
    name_a, name_b = normalize_key(a.name), normalize_key(b.name)
    token_sim = _jaccard(_token_set(name_a), _token_set(name_b))
    trigram_sim = _dice(_char_trigrams(name_a), _char_trigrams(name_b))
    return max(token_sim, trigram_sim) >= BLOCK_SIMILARITY_THRESHOLD


def block(mentions: Sequence[Mention]) -> list[list[Mention]]:
    """Cheap candidate generation before any LLM call.

    Partitions `mentions` by `(patient_id, entity_type)` first — never
    compares across patients (hard rule, AC2) and never compares a
    `:Medication` mention against a `:Condition` mention (two different
    domain-entity labels are never the same real-world entity in this
    schema, ARCHITECTURE §5.1). Within a partition, single-pass greedy
    assignment: a mention joins the first existing block whose blocking-key
    (alias-gazetteer-folded normalized name) matches exactly, OR whose
    similarity (token-Jaccard / char-trigram Dice, see `_similar`) to that
    block's first (representative) member clears
    `BLOCK_SIMILARITY_THRESHOLD`; otherwise it opens a new block.

    Deterministic in input order (no sorting, no randomness) — required for
    `resolve()`'s "ids are stable on re-run" guarantee downstream, since
    `resolve()`'s representative choice (and therefore `canonical_name`,
    and therefore the minted id) depends on which mention block() puts
    first in each block.
    """
    partitions: dict[tuple[str, str], list[list[Mention]]] = {}
    for m in mentions:
        key = (m.patient_id, m.entity_type)
        blocks_for_partition = partitions.setdefault(key, [])
        placed = False
        for b in blocks_for_partition:
            if _similar(m, b[0]):
                b.append(m)
                placed = True
                break
        if not placed:
            blocks_for_partition.append([m])

    out: list[list[Mention]] = []
    for key in partitions:
        out.extend(partitions[key])
    return out


def blocking_stats(mentions: Sequence[Mention], blocks: Sequence[Sequence[Mention]]) -> dict:
    """Reduction-ratio report: brute-force pairwise comparisons
    (`C(n, 2)` over all mentions) vs. pairwise comparisons actually needed
    after blocking (`sum(C(|block|, 2))`). Survey 04's own formal framing
    (R-ER-27): Reduction Ratio is one of blocking's three defining axes."""
    n = len(mentions)
    pairs_naive = n * (n - 1) // 2
    pairs_after = sum(len(b) * (len(b) - 1) // 2 for b in blocks)
    reduction_ratio = 1.0 - (pairs_after / pairs_naive) if pairs_naive else 0.0
    return {
        "n_mentions": n,
        "n_blocks": len(blocks),
        "pairs_naive": pairs_naive,
        "pairs_after_blocking": pairs_after,
        "reduction_ratio": reduction_ratio,
        "largest_block": max((len(b) for b in blocks), default=0),
    }


# ---------------------------------------------------------------------------
# match() — zero-shot LLM pairwise adjudication, cached, LLM-failure-safe.
# ---------------------------------------------------------------------------

_MATCH_SYSTEM_PROMPT = (
    "You are adjudicating whether two mentions extracted from a patient's "
    "clinical dialogue refer to the SAME real-world entity (e.g. the same "
    "medication, condition, symptom, procedure, or allergen), possibly "
    "under different surface forms (brand vs. generic drug name, a "
    "colloquial description, a typo, an abbreviation). Judge real-world "
    "identity, not surface similarity — 'metformin' and 'Glucophage' are "
    "the same entity (brand/generic pair); 'metformin' and 'insulin' are "
    "not, even though both are diabetes medications. Respond with the "
    "required JSON only."
)

_MATCH_SCHEMA = {
    "type": "object",
    "properties": {
        "same_entity": {"type": "boolean"},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["same_entity", "confidence", "reason"],
    "additionalProperties": False,
}

_MATCH_MAX_TOKENS = 300

Complete = Callable[[str, str, dict], dict]
"""`(system, user, schema) -> parsed_json_dict` — the same shape
`eval/judge.py._call_llm` / `pipeline/extract.py._call_llm` already use in
this codebase. Injected for testing (mirrors `Extractor(client=fake)` /
`Judge(client=fake)`); defaults to a lazily-constructed real Anthropic call
when `ANTHROPIC_API_KEY` is configured."""


class MatchUnavailableError(RuntimeError):
    """Raised internally when no LLM is configured and no `complete`
    override was supplied. Always caught by `match()` itself (see
    `match()`'s docstring: match() never raises) — this class exists so the
    reason is typed and greppable, not to escape `match()`."""


def _match_user_prompt(a: Mention, b: Mention) -> str:
    return (
        f"Mention A: {a.name!r} (type: {a.entity_type})\n"
        f"Mention B: {b.name!r} (type: {b.entity_type})\n\n"
        "Are these the same real-world entity?"
    )


MATCH_MODEL = os.environ.get("MEDMEMGRAPH_MATCH_MODEL", llm.EXTRACT_MODEL)
"""Entity-resolution adjudicator. Defaults to `llm.EXTRACT_MODEL` (the cheap
Google tier) — this is a high-volume, low-difficulty yes/no judgement, and
`match()` short-circuits on exact key and caches by normalized name pair, so
real calls are only the genuinely fuzzy candidates."""


def _default_complete(system: str, user: str, schema: dict) -> dict:
    """Route entity matching through `medmemgraph.llm`, the one seam every real
    inference call in this project uses.

    Rewired 2026-08-17. This function previously gated on `ANTHROPIC_API_KEY`
    and then did `import anthropic` — but `anthropic` was removed from
    `pyproject.toml` when the `llm.py` seam replaced it, so the import could
    only ever have raised. In practice the key was never set, so `match()`
    caught `MatchUnavailableError` and returned "no merge" for EVERY fuzzy pair:
    entity resolution silently ran as blocking-key-exact-match only. That is
    why a real corpus produced `shortness of breath` and `short of breath` as
    two separate canonical entities, and why the E3-S1 headline case
    (metformin / Glucophage / "the 500mg one") could never merge on real data.
    """
    try:
        response = llm.complete(
            user,
            model=MATCH_MODEL,
            schema=schema,
            system=system,
            max_tokens=_MATCH_MAX_TOKENS,
            temperature=0.0,
        )
    except llm.LLMError as exc:
        # Typed, so `match()` degrades to "no merge" for this pair rather than
        # sinking the whole resolve() call — same contract as before, now with
        # a real reason attached instead of a missing-key stub.
        raise MatchUnavailableError(f"llm.complete failed for entity match: {exc}") from exc
    if not isinstance(response.parsed, dict):
        raise MatchUnavailableError(
            f"entity-match response was not a JSON object: {response.parsed!r}"
        )
    return response.parsed


def _pair_cache_key(a: Mention, b: Mention) -> tuple[str, str]:
    """Order-independent cache key on normalized name pair — "cache
    aggressively by normalized pair key" (dispatch). Deliberately NOT
    keyed on patient_id/entity_type: two mentions are only ever compared
    within one blocking partition anyway (same patient, same type), so the
    normalized-name pair alone is already unambiguous within a resolve()
    call, and staying name-only lets the SAME cache (if the caller persists
    it) short-circuit an identical brand/generic comparison the next time
    it recurs for a *different* patient too."""
    return tuple(sorted((normalize_key(a.name), normalize_key(b.name))))  # type: ignore[return-value]


def match(
    a: Mention,
    b: Mention,
    *,
    complete: Complete | None = None,
    cache: MutableMapping[tuple, tuple] | None = None,
    stats: MutableMapping[str, int] | None = None,
) -> tuple[bool, float, str]:
    """Zero-shot LLM adjudication: "is `a` the same real-world entity as
    `b`?". Returns `(matched, confidence, reason)`.

    Never raises. Three fast paths avoid an LLM call entirely (cheapest
    discriminating check first): (1) cross-patient pairs are always
    `False` by construction (AC2's hard rule enforced here too, not just in
    `block()`); (2) an exact normalized-key match (post alias-gazetteer
    folding) is always `True` at confidence 1.0 — no reasonable LLM would
    say "metformin" and "Metformin." are different entities, and asking
    costs money for a certain answer; (3) a cache hit on the normalized
    pair key returns the previously-adjudicated result.

    On an LLM failure (down, no key configured, malformed response), this
    function catches it internally and returns `(False, 0.0, reason)` —
    "on LLM failure: do not merge" (E3-S1.md contract point 3; AC4).
    Failure-to-merge is the conservative direction: a missed merge is a
    duplicate node the ER pass can fix on the NEXT admission once the LLM
    is available again; a wrong merge silently breaks invalidation
    (ARCHITECTURE §6.3's own stated reason) and there is no future pass
    that un-merges it (survey 04 R-ER-41: the literature explicitly flags
    merge-correction as under-addressed).
    """
    if a.patient_id != b.patient_id:
        return False, 0.0, "cross-patient comparison is never permitted"

    if _blocking_group_key(a) == _blocking_group_key(b):
        if stats is not None:
            stats["exact_key_matches"] = stats.get("exact_key_matches", 0) + 1
        return True, 1.0, "exact normalized-key match (post alias-gazetteer fold)"

    cache_key = _pair_cache_key(a, b)
    if cache is not None and cache_key in cache:
        if stats is not None:
            stats["cache_hits"] = stats.get("cache_hits", 0) + 1
        return cache[cache_key]

    completer = complete or _default_complete
    try:
        data = completer(_MATCH_SYSTEM_PROMPT, _match_user_prompt(a, b), _MATCH_SCHEMA)
        matched = bool(data["same_entity"])
        confidence = float(data.get("confidence", 0.5 if matched else 0.0))
        reason = str(data.get("reason", ""))
        result = (matched, confidence, reason)
        if stats is not None:
            stats["llm_calls"] = stats.get("llm_calls", 0) + 1
    except Exception as exc:  # noqa: BLE001 - never propagate; "do not merge" on any failure
        result = (False, 0.0, f"no merge — LLM adjudication unavailable/failed: {exc}")
        if stats is not None:
            stats["llm_failures"] = stats.get("llm_failures", 0) + 1

    if cache is not None:
        cache[cache_key] = result
    return result


# ---------------------------------------------------------------------------
# resolve() — block, then representative-anchored clustering within each
# block (the transitivity guard), incremental against `existing`.
# ---------------------------------------------------------------------------


def _llm_configured(complete: Complete | None) -> bool:
    """Is a real adjudicator reachable? An injected `complete` always counts;
    otherwise the default path needs whichever provider key `MATCH_MODEL` routes
    to. Checked by resolving the key, not by probing one hardcoded env var name
    — `llm.py` accepts four spellings across two providers and also reads `.env`,
    so an env-var membership test gave false negatives and warned about degrading
    on runs that were about to work fine."""
    if complete is not None:
        return True
    try:
        llm.resolve_key_for_model(MATCH_MODEL)
    except llm.LLMError:
        return False
    return True


def resolve(
    mentions: Sequence[Mention],
    *,
    existing: Sequence[CanonicalEntity] = (),
    complete: Complete | None = None,
    cache: MutableMapping[tuple, tuple] | None = None,
    stats: MutableMapping[str, int] | None = None,
    id_map: IdMap | IdMinter | None = None,
    max_cluster_size: int = MAX_CLUSTER_SIZE,
) -> list[CanonicalEntity]:
    """`mentions` -> canonical entities, per patient, incremental against
    `existing`.

    Two-phase per (patient, entity_type, blocking-key-group) block:

    Phase 1 — incremental match against `existing`. Each `existing`
    canonical is turned into a synthetic representative `Mention`
    (`source_canonical` set) and fed into the SAME `block()` call as the
    new mentions, so an existing canonical only ever gets compared to new
    mentions blocking places alongside it. Every new mention in a block
    that also contains an existing-canonical pseudo-mention is matched
    against that pseudo-mention first; a positive match appends the new
    mention as an alias onto the EXISTING canonical (never reminting,
    never renumbering — "incremental resolution of a new admission does
    not renumber existing entities").

    Phase 2 — brand-new clusters. Whatever new mentions did not match any
    existing canonical are clustered among themselves within their block
    via single-pass representative-anchored clustering (module docstring's
    transitivity guard): the first unresolved mention in a block opens a
    cluster as its representative; each later mention is matched only
    against open clusters' representatives (never a non-representative
    member — this is what makes it a guard, not connected components) and
    joins the first one that matches, subject to `max_cluster_size`; a
    mention matching none opens its own new cluster. Each resulting
    cluster mints ONE new `CanonicalEntity` via `ids.mint_entity_id`.

    Degrade discipline (E3-S1.md: "Document the degrade in a log line; do
    not silently skip"): if no LLM is configured (no `complete` override
    and no `ANTHROPIC_API_KEY`), this function logs exactly one `warning`
    up front and proceeds — every fuzzy (non-exact-key) candidate pair
    then simply fails to merge (see `match()`'s docstring), which is the
    documented "degrades to: blocking-key exact match only" behaviour, not
    a crash and not a silent skip.

    Returns the FULL current canonical set touched by this call: every
    `existing` canonical passed in (whether or not it received a new
    alias this call) plus every brand-new canonical minted this call, for
    every patient present in `mentions` or `existing`.
    """
    if not _llm_configured(complete):
        logger.warning(
            "pipeline.resolve: no API key resolvable for MATCH_MODEL=%s and no "
            "complete() override supplied — degrading to blocking-key-exact-match "
            "only; fuzzy (non-exact-key) candidates will NOT be merged this run, so "
            "surface variants of one entity ('shortness of breath' / 'short of "
            "breath') will become separate canonical nodes",
            MATCH_MODEL,
        )

    by_patient_existing: dict[str, list[CanonicalEntity]] = {}
    for c in existing:
        by_patient_existing.setdefault(c.patient_id, []).append(c)

    by_patient_new: dict[str, list[Mention]] = {}
    for m in mentions:
        by_patient_new.setdefault(m.patient_id, []).append(m)

    patient_ids = list(dict.fromkeys([*by_patient_existing, *by_patient_new]))
    out: list[CanonicalEntity] = []

    for patient_id in patient_ids:
        existing_for_patient = by_patient_existing.get(patient_id, [])
        new_for_patient = by_patient_new.get(patient_id, [])

        pseudo = [
            Mention(
                name=c.canonical_name,
                entity_type=c.entity_type,
                patient_id=c.patient_id,
                source_canonical=c,
            )
            for c in existing_for_patient
        ]

        pool = pseudo + new_for_patient
        blocks = block(pool)

        new_canonicals: list[CanonicalEntity] = []

        for blk in blocks:
            anchors = [m for m in blk if m.source_canonical is not None]
            unresolved = [m for m in blk if m.source_canonical is None]

            if anchors:
                still_unresolved: list[Mention] = []
                for m in unresolved:
                    matched_anchor: CanonicalEntity | None = None
                    for anchor in anchors:
                        matched, _conf, _reason = match(
                            m, anchor, complete=complete, cache=cache, stats=stats
                        )
                        if matched:
                            matched_anchor = anchor.source_canonical
                            break
                    if matched_anchor is not None:
                        matched_anchor.add_alias(m.name)
                        matched_anchor.mentions.append(m)
                    else:
                        still_unresolved.append(m)
                unresolved = still_unresolved

            # Phase 2 — representative-anchored clustering (transitivity guard).
            # The size cap bounds only FUZZY-JOIN growth (a running per-cluster
            # counter, `fuzzy_joins[id(cluster)]` below) — NOT total membership.
            # An exact blocking-group-key match (identical/near-identical
            # surface form, e.g. the same drug mentioned in 30 admissions)
            # carries no transitivity risk at all — it is a direct, unambiguous
            # identity, not an inference chained through an intermediate weak
            # link — so it always joins and never counts against the cap.
            #
            # Real-corpus bug found and fixed here (not hand-waved — caught by
            # actually running `scripts/generate_resolve_eyeball.py` against
            # patient 13813803's 64 real admissions, not by the unit tests
            # alone): an EARLIER version of this loop capped on `len(cluster)`
            # (total membership) instead of fuzzy-join count. A cluster that
            # had already grown past `max_cluster_size` purely via harmless
            # exact matches (e.g. "endoscopy" mentioned 36 times) then
            # silently SKIPPED comparing any later fuzzy candidate against it
            # at all — `match()` was never even called, so a legitimate
            # paraphrase candidate for a COMMON entity would have been
            # dropped for the wrong reason (the entity's sheer mention count,
            # not real transitivity risk). That is exactly the story's
            # headline under-merge failure mode (a frequently-mentioned drug
            # failing to absorb a late-arriving brand-name/paraphrase alias),
            # reproduced by the very mechanism meant to guard against a
            # different failure mode. Fixed by capping fuzzy-join count only.
            clusters: list[list[Mention]] = []
            fuzzy_joins: dict[int, int] = {}
            for m in unresolved:
                joined = False
                for cluster in clusters:
                    representative = cluster[0]
                    exact = _blocking_group_key(m) == _blocking_group_key(representative)
                    if not exact and fuzzy_joins.get(id(cluster), 0) >= max_cluster_size:
                        continue
                    matched, _conf, _reason = match(
                        m, representative, complete=complete, cache=cache, stats=stats
                    )
                    if matched:
                        cluster.append(m)
                        if not exact:
                            fuzzy_joins[id(cluster)] = fuzzy_joins.get(id(cluster), 0) + 1
                        joined = True
                        break
                if not joined:
                    clusters.append([m])

            for cluster in clusters:
                representative = cluster[0]
                canonical_name = representative.name
                entity_type = representative.entity_type
                canonical_id = mint_entity_id(
                    patient_id, entity_type, canonical_name, id_map=id_map
                )
                entity = CanonicalEntity(
                    canonical_id=canonical_id,
                    canonical_name=canonical_name,
                    entity_type=entity_type,
                    patient_id=patient_id,
                    aliases=[],
                    mentions=list(cluster),
                )
                for m in cluster:
                    entity.add_alias(m.name)
                new_canonicals.append(entity)

        out.extend(existing_for_patient)
        out.extend(new_canonicals)

    return out


# ---------------------------------------------------------------------------
# attach_canonical_ids() — the pipeline entry point.
# ---------------------------------------------------------------------------


def attach_canonical_ids(
    facts: list[ClinicalFact],
    *,
    registry: CanonicalRegistry | None = None,
    complete: Complete | None = None,
    cache: MutableMapping[tuple, tuple] | None = None,
    stats: MutableMapping[str, int] | None = None,
    id_map: IdMap | IdMinter | None = None,
) -> list[ClinicalFact]:
    """Rewrite `subject.canonical_id` / `object.canonical_id` on every fact,
    IN PLACE (`EntityRef` is a plain, non-frozen dataclass — see
    `contracts.py`), so downstream `UNWIND MERGE` writes land on one node
    per entity. Also returns `facts` for chaining convenience.

    `EntityRef.type == "Patient"` is minted directly via
    `ids.mint_patient_id` — patient identity is exact-by-`subject_id`, not
    fuzzy; it never goes through `block()`/`match()`. Every other
    `EntityRef` becomes a `Mention` and goes through `resolve()`.

    `registry` is the incremental seam: pass the SAME `CanonicalRegistry`
    across calls (loaded via `CanonicalRegistry.load_json` at the start of
    a run, saved back via `.save_json` at the end) so a later admission's
    facts resolve against everything already known without reprocessing
    earlier admissions' facts (module docstring / dispatch's incremental
    requirement). Omitting `registry` runs single-shot (fresh, empty
    registry each call) — correct for one-off/test use, but each call then
    starts from zero prior knowledge.
    """
    registry = registry if registry is not None else CanonicalRegistry()

    mentions: list[Mention] = []
    for i, fact in enumerate(facts):
        for role, ref in (("subject", fact.subject), ("object", fact.object)):
            if ref.type == "Patient":
                ref.canonical_id = mint_patient_id(ref.name, id_map=id_map)
                continue
            mentions.append(
                Mention(
                    name=ref.name,
                    entity_type=ref.type,
                    patient_id=fact.patient_id,
                    session_id=fact.session_id,
                    turn_ids=list(fact.turn_ids),
                    fact_index=i,
                    role=role,  # type: ignore[arg-type]
                )
            )

    canonicals = resolve(
        mentions,
        existing=registry.all(),
        complete=complete,
        cache=cache,
        stats=stats,
        id_map=id_map,
    )

    # Persist the full (existing + newly-minted) canonical set per patient.
    by_patient_result: dict[str, list[CanonicalEntity]] = {}
    for entity in canonicals:
        by_patient_result.setdefault(entity.patient_id, []).append(entity)
    for patient_id, entities in by_patient_result.items():
        registry.replace_patient(patient_id, entities)

    # Rewrite facts using ONLY this call's own Mention objects (matched by
    # Python object identity), never a mention's fact_index blindly. This
    # matters specifically for `existing` canonicals: their `.mentions`
    # list accumulates across every past `attach_canonical_ids` call, and a
    # historical mention's `fact_index` points into a *different* (past)
    # `facts` list — blindly trusting every `entity.mentions` record here
    # would silently mutate the wrong fact (or index out of range) on the
    # second and later incremental call. Filtering to `id(m) in this_call`
    # is what makes incremental resolution safe.
    this_call_ids = {id(m) for m in mentions}
    for entity in canonicals:
        for m in entity.mentions:
            if id(m) not in this_call_ids:
                continue
            if m.fact_index is None or m.role is None:
                continue  # defensive only; every real mention built above sets both
            ref = getattr(facts[m.fact_index], m.role)
            ref.canonical_id = entity.canonical_id

    return facts
