"""Deterministic integer id minting — ARCHITECTURE.md §5.4.

HydraDB node ids are non-negative integers and `MERGE` matches on `id` **only**
(`literature/10` §J.3). HydraDB has no transactions, so a crashed-and-restarted
writer must re-derive the *same* id for the *same* real-world thing on replay,
or a partial ingest turns into duplicate nodes. That is the one job of this
module: turn a stable "identity key" into a stable, collision-resistant,
non-negative integer, and turn a `ClinicalFact`'s identity fields into a
stable content hash (`fact_id`). Nothing here talks to HydraDB; the graph
writer consumes these ids, it does not remint them (§5.4, E3-S1 concurrency
note).

Do NOT allocate ids from a counter. A crashed writer plus a restarted counter
duplicates nodes; `MERGE` cannot tell the duplicate from the original.

--------------------------------------------------------------------------
Two API surfaces are implemented here, on purpose, because two documents in
this repo describe the same §5.4 scheme at different altitudes and this
module has to satisfy both:

  * This story's own spec: `mint_entity_id`, `mint_claim_id`, `mint_patient_id`,
    `fact_id` — one ergonomic function per node kind.
  * `collaborative/design/stories/E3/E3-S1.md` (the ER story, which also lists
    `ids.py` under "Files to touch" and notes "E4-S3 imports mint; it does not
    rewrite it"): a generic `mint(identity_key, id_map) -> int` primitive plus
    `normalize_key`.

The four ergonomic mint_* functions are thin wrappers that build the §5.4
identity key and call the same underlying primitive (`IdMinter.mint`, exposed
module-level as `mint()`). One scheme, two doors in. See the "Reconciling two
spec documents" note below `mint_entity_id` for the one place these two
sources actually disagreed and how it was resolved.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, MutableMapping, Sequence

# ---------------------------------------------------------------------------
# Bit width & fold — ARCHITECTURE.md §5.4 point 3: "positive 63-bit fold of a
# cryptographic digest of the identity key." §5.4 specifies the width, so we
# use it exactly (63-bit) rather than picking our own.
# ---------------------------------------------------------------------------

BIT_WIDTH = 63
MAX_ID = (1 << BIT_WIDTH) - 1  # 9_223_372_036_854_775_807 — inclusive ceiling

IdMap = MutableMapping[str, int]
"""identity_key -> minted integer id. The one piece of state that must be
persisted (by the Graph, per §5.4 point 1: 'Mint in the Pipeline, persist the
map in the Graph') and reloaded across runs for idempotent replay."""


def _fold_digest(identity_key: str) -> int:
    """SHA-256 the identity key, take the first 8 bytes as a big-endian
    unsigned 64-bit int, mask off the top bit. Deterministic, non-negative,
    inside [0, MAX_ID]. SHA-256 (not a fast/insecure hash) so an adversarial
    or merely unlucky input can't be hand-crafted to force collisions the way
    it could against e.g. Python's salted `hash()` or a CRC."""
    digest = hashlib.sha256(identity_key.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big")
    return value & MAX_ID


# ---------------------------------------------------------------------------
# normalize_key — §5.4's "<normalized_name>"; wording ("lowercase, strip
# punctuation/whitespace") taken from E3-S1.md's identical spec for the same
# function so ER's blocking key and this module's identity key never drift
# apart.
# ---------------------------------------------------------------------------

_NON_WORD_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+", re.UNICODE)


def normalize_key(name: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace. Used to build the
    `<normalized_name>` fragment of a domain-entity identity key so that
    "Metformin", "metformin.", and "  METFORMIN  " mint to the same node.

    >>> normalize_key("  METFORMIN, 500mg.  ")
    'metformin 500mg'
    """
    lowered = name.strip().lower()
    no_punct = _NON_WORD_RE.sub(" ", lowered)
    return _WHITESPACE_RE.sub(" ", no_punct).strip()


# ---------------------------------------------------------------------------
# IdMinter — the stateful primitive. §5.4 point 3: keep a client-side id_map
# keyed by identity key; on collision with a *different* key, linear-probe
# forward and record the assignment; never reuse an id.
# ---------------------------------------------------------------------------


class IdMinter:
    """Wraps a persistable `id_map` (identity_key -> int) plus an in-memory
    reverse-lookup set of ids already assigned, so repeated `.mint()` calls
    against the *same* minter are O(1) amortized instead of re-scanning
    `id_map.values()` (a plain dict) on every call — the difference between
    the >=100k-name collision sweep finishing in under a second and not
    finishing at all.

    The reverse set (`_used_ids`) is a derived cache, not state to persist:
    on load, rebuild it once from the persisted `id_map` (done in
    `__init__`) and it is correct from then on.
    """

    def __init__(self, id_map: IdMap | None = None) -> None:
        self.id_map: IdMap = id_map if id_map is not None else {}
        self._used_ids: set[int] = set(self.id_map.values())

    def mint(self, identity_key: str) -> int:
        existing = self.id_map.get(identity_key)
        if existing is not None:
            return existing
        candidate = _fold_digest(identity_key)
        while candidate in self._used_ids:
            # Linear probe. §5.4: "On collision with a different key,
            # increment the fold ... Never reuse an id." Wraps mod 2**63 via
            # the mask, which only matters if the id space is ever near
            # exhausted (it will not be at this project's scale).
            candidate = (candidate + 1) & MAX_ID
        self.id_map[identity_key] = candidate
        self._used_ids.add(candidate)
        return candidate


_DEFAULT_MINTER = IdMinter()
"""Module-level minter used when a caller doesn't supply its own `id_map` —
this is what makes `mint_entity_id("p1", "Medication", "metformin")` work
with no third argument and still be idempotent *within one process*. It does
NOT persist across process restarts; pass an explicit `id_map` (loaded from
wherever the Graph persisted the last run's map) for cross-run idempotent
replay (§5.4 point 4, and this story's AC: "same identity key minted twice,
fresh id_map loaded from the first run -> identical integer")."""


def _resolve_minter(id_map: IdMap | IdMinter | None) -> IdMinter:
    if id_map is None:
        return _DEFAULT_MINTER
    if isinstance(id_map, IdMinter):
        return id_map
    return IdMinter(id_map)


def mint(identity_key: str, id_map: IdMap) -> int:
    """Generic primitive matching E3-S1.md's documented shape,
    `mint(identity_key, id_map) -> int`, for code that only has a plain dict
    and an identity key (e.g. a future `schema.py` that "imports mint; does
    not rewrite it").

    This constructs a throwaway `IdMinter` per call, which rebuilds the O(n)
    reverse-lookup set every time — fine for the low call volumes of a real
    ingest run or one-off lookups, *not* for a tight loop minting many ids
    against the same `id_map`. For that (e.g. `collision_self_check` below),
    construct one `IdMinter` and call `.mint()` on it repeatedly instead.
    """
    return IdMinter(id_map).mint(identity_key)


# ---------------------------------------------------------------------------
# mint_entity_id / mint_claim_id / mint_patient_id — this story's ergonomic
# wrappers. Each builds the §5.4 identity key for its node kind and mints it.
# ---------------------------------------------------------------------------


def mint_patient_id(subject_id: str, *, id_map: IdMap | IdMinter | None = None) -> int:
    """§5.4: `Patient|<subject_id>`."""
    if not subject_id:
        raise ValueError("subject_id must be non-empty")
    return _resolve_minter(id_map).mint(f"Patient|{subject_id}")


def mint_entity_id(
    patient_id: str,
    entity_type: str,
    canonical_name: str,
    *,
    id_map: IdMap | IdMinter | None = None,
) -> int:
    """Domain entity (`:Condition` / `:Medication` / `:Allergy` / `:Symptom` /
    `:Procedure` / `:Provider`) id.

    Reconciling two spec documents (stated here, not silently picked):
    ARCHITECTURE.md §5.4's literal bullet for a domain entity is
    `<Type>|<normalized_name>` — no patient scoping. Taken literally, two
    different patients' "metformin" would mint to the *one* shared node.
    But this story's own signature takes `patient_id` as the first
    positional argument, and the sibling ER story (E3-S1.md, AC2) requires
    "metformin for patient A and metformin for patient B ... receive
    different canonical ids (never cross-patient)" — this is a per-patient
    longitudinal memory graph, not a shared drug ontology; two patients'
    "metformin" claims should not collapse onto one canonical node just
    because the surface form matches.

    Resolution (medium-high confidence; flagged to dev-architect in the
    return note rather than assumed silently): the identity key is
    `<Type>|<patient_id>|<normalized_name>`, i.e. §5.4's bullet extended
    with the patient scope this story's signature already implies. If that
    reading is wrong, the fix is a one-line change to the f-string below;
    nothing else in this module depends on the exact key shape.
    """
    if not patient_id:
        raise ValueError("patient_id must be non-empty")
    if not entity_type:
        raise ValueError("entity_type must be non-empty")
    normalized = normalize_key(canonical_name)
    if not normalized:
        raise ValueError(f"canonical_name normalizes to empty string: {canonical_name!r}")
    identity_key = f"{entity_type}|{patient_id}|{normalized}"
    return _resolve_minter(id_map).mint(identity_key)


def mint_claim_id(fact_id_value: str, *, id_map: IdMap | IdMinter | None = None) -> int:
    """§5.4: `Claim|<fact_id>`. Parameter named `fact_id_value` (not `fact_id`)
    so it doesn't shadow the `fact_id()` function below in this module; the
    story's documented signature is `mint_claim_id(fact_id) -> int` and this
    is call-compatible with that (pass positionally or as `fact_id_value=`)."""
    if not fact_id_value:
        raise ValueError("fact_id_value must be non-empty")
    return _resolve_minter(id_map).mint(f"Claim|{fact_id_value}")


# ---------------------------------------------------------------------------
# fact_id — the deterministic content hash, ARCHITECTURE.md §5.4 point 2.
# ---------------------------------------------------------------------------


def fact_id(
    *,
    patient_id: str,
    session_id: str,
    turn_ids: Sequence[int],
    subject_canonical_id: int,
    predicate: str,
    object_canonical_id: int,
    polarity: str,
    valid_from: str,
) -> str:
    """Stable hex digest of a `ClinicalFact`'s identity key, exactly the
    tuple ARCHITECTURE.md §5.4 point 2 names: `(patient_id, session_id,
    sorted turn_ids, subject.canonical_id, predicate, object.canonical_id,
    polarity, valid_from)`. This is the replay key: re-running extraction
    over the same turns should reproduce the same `fact_id` so the writer's
    `MERGE` updates the existing `:Claim` instead of duplicating it.

    Which fields participate, and why (§5.4 names this exact tuple; the
    reasoning for each inclusion/exclusion below is this module's argument
    for *why* that tuple is the right one, not a proposal to change it):

    IN the hash:
      - `patient_id`, `session_id`, `turn_ids` (sorted) — *where the claim
        comes from*. Two structurally-identical claims drawn from different
        turns/sessions are still provenance-distinct rows; conflating them
        would let one claim's `DRAWN_FROM` edges silently absorb another's
        source turns.
      - `subject_canonical_id`, `predicate`, `object_canonical_id` — *what
        the claim says*. This is the actual assertion; changing any one of
        these is a different claim by construction (see the "changes when
        predicate/object changes" test below).
      - `polarity` — asserted vs negated are opposite claims about the same
        subject/predicate/object; hashing them the same would let a later
        negation silently overwrite instead of contradict/supersede.
      - `valid_from` — the start of the claim's validity interval is part of
        its temporal identity: "started metformin in March" and "started
        metformin in June" are different claims even with identical
        subject/predicate/object/polarity. (This does assume the Normalize
        step, §6.2, resolves the same source text to the same `valid_from`
        on re-extraction; that determinism is Normalize's contract to keep,
        not something this module can enforce.)

    OUT of the hash (both explicitly excluded by §5.4's tuple, and each
    excluded for an independent reason so this isn't just "because the spec
    said so"):
      - `confidence` — a model's self-reported certainty about *the same*
        claim. Two extraction passes over identical turns can legitimately
        return 0.87 vs 0.91 confidence for what is obviously the same fact;
        hashing confidence in would mint a fresh `:Claim` node on every
        re-run purely from extractor jitter, defeating idempotent replay —
        exactly the failure mode §5.4 exists to prevent.
      - `observed_at` — "when it was said" (assertion time), distinct from
        `valid_from` ("when it became true", §5.3). `turn_ids` + `session_id`
        already pin the claim to a specific point in the dialogue, so
        `observed_at` is a derived/display field here, not an identity
        field; including it would again let harmless re-normalization noise
        (a slightly different assertion-time parse) fork the id.
    """
    if not turn_ids:
        raise ValueError("turn_ids must be non-empty")
    sorted_turns = ",".join(str(t) for t in sorted(turn_ids))
    payload = "\x1f".join(
        [
            patient_id,
            session_id,
            sorted_turns,
            str(subject_canonical_id),
            predicate,
            str(object_canonical_id),
            polarity,
            valid_from,
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Collision self-check — a fast sweep to sanity-check the fold's collision
# resistance at ingest scale before trusting it on the real corpus.
# ---------------------------------------------------------------------------


def collision_self_check(
    names: Iterable[str],
    *,
    entity_type: str = "CollisionSelfCheck",
    patient_id: str = "collision-self-check",
) -> dict:
    """Mint an entity id for every name in `names` (one shared `IdMinter`,
    so it is O(n) for even very large `names`) and report on collisions.

    `IdMinter.mint` always linear-probes past an occupied slot, so a
    *final*-id collision between two different identity keys should be
    impossible by construction — if `final_id_collisions` is ever non-empty,
    that is a bug in `IdMinter`, not a hash birthday-bound event, and should
    be treated as a hard failure. What this sweep actually measures is the
    *raw* fold's collision rate (`raw_fold_collisions`): how often two
    different names hash to the same 63-bit value *before* probing moves one
    of them. At n names in a 2**63 space the expected raw-collision count is
    roughly n**2 / 2**64 (birthday bound) — for n = 100_000 that is about
    2.7e-4, i.e. an honest run should report zero.

    Returns
    -------
    dict with keys:
      n_names             — count of names consumed from the iterable.
      n_unique_keys        — count of distinct identity keys (dedups exact
                              repeats in `names`, which are not collisions).
      n_unique_ids          — count of distinct minted integer ids; should
                              equal n_unique_keys always.
      raw_fold_collisions   — list of (first_key, second_key, raw_fold) for
                              every raw-fold clash observed (expected: []).
      final_id_collisions   — list of (first_key, second_key, id) for any
                              case where two different keys minted to the
                              same *final* id (expected: [], always — a
                              non-empty list here is an IdMinter bug).
    """
    minter = IdMinter()
    seen_keys: set[str] = set()
    raw_fold_of: dict[int, str] = {}
    raw_collisions: list[tuple[str, str, int]] = []
    final_id_of: dict[int, str] = {}
    final_collisions: list[tuple[str, str, int]] = []
    n = 0

    for name in names:
        n += 1
        identity_key = f"{entity_type}|{patient_id}|{normalize_key(name)}"
        if identity_key in seen_keys:
            continue  # duplicate synthetic name -> same key, not a collision
        seen_keys.add(identity_key)

        raw = _fold_digest(identity_key)
        prior_raw_owner = raw_fold_of.get(raw)
        if prior_raw_owner is not None and prior_raw_owner != identity_key:
            raw_collisions.append((prior_raw_owner, identity_key, raw))
        else:
            raw_fold_of[raw] = identity_key

        minted = minter.mint(identity_key)
        prior_final_owner = final_id_of.get(minted)
        if prior_final_owner is not None and prior_final_owner != identity_key:
            final_collisions.append((prior_final_owner, identity_key, minted))
        else:
            final_id_of[minted] = identity_key

    return {
        "n_names": n,
        "n_unique_keys": len(seen_keys),
        "n_unique_ids": len(final_id_of),
        "raw_fold_collisions": raw_collisions,
        "final_id_collisions": final_collisions,
    }
