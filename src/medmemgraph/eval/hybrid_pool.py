"""Dense + BM25 RRF candidate pool for nDCG-oriented eval/train.

Eval-only. Does not import into retrieve.py. Lexical windows and dense
turns are de-duplicated by (session_id, turn_number) so the same
evidence is not scored twice.
"""

from __future__ import annotations

import re

from medmemgraph.contracts import RetrieveItem
from medmemgraph.graph.fusion import rrf_fuse
from medmemgraph.graph.lexical import LexicalIndex
from medmemgraph.graph.vector_index import PatientIndex
from medmemgraph.pipeline.loader import Conversation

RRF_K = 60
POOL_PER_ARM = 100
_RM3_STOP = frozenset(
    "a an the of to in for on and or is was were be been being this that "
    "it with from at by as not no yes patient doctor admission turn".split()
)
_TOKEN = re.compile(r"[a-z0-9]+")


def _core_turns(item: RetrieveItem) -> list[tuple[str, int]]:
    if item.turn_ids:
        return [(item.session_id, int(t)) for t in item.turn_ids]
    return [(item.session_id, -1)]


def fuse_dense_lexical(
    dense: list[RetrieveItem],
    lexical: list[RetrieveItem],
    *,
    k: int = POOL_PER_ARM,
) -> list[RetrieveItem]:
    fused = rrf_fuse([dense, lexical], k=RRF_K)
    kept: list[RetrieveItem] = []
    seen: set[tuple[str, int]] = set()
    for item in fused:
        keys = _core_turns(item)
        # Keep a window if it contributes ANY new turn. Skipping on
        # *any* overlap dropped BM25 +/-2 neighbors and made consecutive
        # gold turns count as one nDCG item.
        if keys and all(key in seen for key in keys):
            continue
        kept.append(item)
        seen.update(keys)
        if len(kept) >= k:
            break
    return explode_to_turns(kept)[:k]


def explode_to_turns(items: list[RetrieveItem]) -> list[RetrieveItem]:
    """Split multi-turn BM25 windows into one RetrieveItem per turn.

    Hit@k still fires if any member is gold. nDCG needs one list slot per
    gold turn; a 5-turn window scored as a single blob caps nDCG when
    evidence.turn_ids are consecutive.
    """
    out: list[RetrieveItem] = []
    seen: set[tuple[str, int]] = set()
    for item in items:
        tids = list(item.turn_ids) if item.turn_ids else [-1]
        if len(tids) <= 1:
            key = (item.session_id, tids[0])
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
            continue
        for tid in tids:
            key = (item.session_id, int(tid))
            if key in seen:
                continue
            seen.add(key)
            out.append(
                RetrieveItem(
                    text=item.text,
                    session_id=item.session_id,
                    turn_ids=[int(tid)],
                    score=item.score,
                    channel=item.channel,
                )
            )
    return out


def build_indexes(
    patient_id: str, conversation: Conversation, *, dense_backend: str
) -> tuple[PatientIndex, LexicalIndex]:
    dense = PatientIndex(patient_id, backend=dense_backend, cache_path=None)
    dense.build(conversation)
    lex = LexicalIndex(patient_id)
    lex.build(conversation)
    return dense, lex


def _turn_text_map(dense: PatientIndex) -> dict[tuple[str, int], str]:
    out: dict[tuple[str, int], str] = {}
    for unit in dense.units:
        if getattr(unit, "kind", "turn") != "turn":
            continue
        if unit.turn_ids:
            out[(unit.session_id, int(unit.turn_ids[0]))] = unit.text
    return out


def hybrid_search(
    query: str,
    dense: PatientIndex,
    lex: LexicalIndex,
    *,
    k: int = POOL_PER_ARM,
    per_arm: int = POOL_PER_ARM,
) -> list[RetrieveItem]:
    fused = fuse_dense_lexical(dense.search(query, per_arm), lex.search(query, per_arm), k=k)
    tmap = _turn_text_map(dense)
    out: list[RetrieveItem] = []
    for item in fused:
        tid = item.turn_ids[0] if item.turn_ids else None
        text = tmap.get((item.session_id, int(tid))) if tid is not None else None
        if text is None:
            out.append(item)
            continue
        out.append(
            RetrieveItem(
                text=text,
                session_id=item.session_id,
                turn_ids=[int(tid)],
                score=item.score,
                channel=item.channel,
            )
        )
    return out


def rm3_terms(query: str, feedback_docs: list[str], *, n_terms: int = 8) -> list[str]:
    """Cheap RM3-style expansion terms from pseudo-relevant docs.

    No gold answer. Terms already in the query are skipped. Used only to
    lift first-stage recall of the second gold turn.
    """
    qset = set(_TOKEN.findall(query.lower()))
    tf: dict[str, int] = {}
    for doc in feedback_docs:
        for tok in _TOKEN.findall(doc.lower()):
            if len(tok) < 3 or tok in _RM3_STOP or tok in qset or tok.isdigit():
                continue
            tf[tok] = tf.get(tok, 0) + 1
    return [w for w, _ in sorted(tf.items(), key=lambda kv: (-kv[1], kv[0]))[:n_terms]]


def expanded_hybrid_search(
    query: str,
    dense: PatientIndex,
    lex: LexicalIndex,
    *,
    k: int = POOL_PER_ARM,
    per_arm: int = POOL_PER_ARM,
) -> list[RetrieveItem]:
    """Original hybrid RRF-fused with an RM3-expanded hybrid query."""
    base = hybrid_search(query, dense, lex, k=k, per_arm=per_arm)
    terms = rm3_terms(query, [h.text for h in base[:10]])
    if not terms:
        return base
    expanded = f"{query} {' '.join(terms)}"
    extra = hybrid_search(expanded, dense, lex, k=k, per_arm=per_arm)
    return fuse_dense_lexical(base, extra, k=k)
