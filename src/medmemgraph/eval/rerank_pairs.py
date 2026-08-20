"""Gold + arctic-s hard-negative pairs for MiniLM finetune (FT-E1-S2).

Readers are ``load_qa`` / ``load_conversation`` only. Hard-neg mining
runs on train (resp. dev) patients — never the eval trio.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from medmemgraph.contracts import RetrieveItem
from medmemgraph.eval.rerank_split import EVAL_TRIO
from medmemgraph.graph.vector_index import PatientIndex, format_turn
from medmemgraph.pipeline.loader import Conversation, Turn, load_conversation, load_qa

DEFAULT_OUT_DIR = Path("data/reranker_ft")
HARDNEG_K = 20
# Dedicated arctic-s cache. The shared `data/index/st_embed_cache.npz` mixes
# embedder dims and `save()` raises ValueError on np.stack.
ARCTIC_CACHE = Path("data/reranker_ft/arctic_s_embed_cache.npz")


def is_gold_hit(
    session_id: str,
    turn_ids: Sequence[int],
    gold_admissions: set[str],
    gold_turn_ids: set[int] | None,
) -> bool:
    """Same rule as ``metrics._is_relevant`` (copied, not imported)."""
    if session_id not in gold_admissions:
        return False
    if gold_turn_ids is None:
        return True
    return bool(set(turn_ids) & gold_turn_ids)


def gold_turns_for_item(conversation: Conversation, evidence: dict[str, Any]) -> tuple[list[Turn], str]:
    """Return (gold turns, grain tag). grain is ``turn`` or ``admission_expanded``."""
    admissions = {str(a) for a in (evidence.get("admissions") or [])}
    raw_turn_ids = evidence.get("turn_ids")
    if raw_turn_ids:
        wanted = {int(t) for t in raw_turn_ids}
        turns = [
            t
            for t in conversation.turns()
            if t.hadm_id in admissions and t.turn_number in wanted
        ]
        return turns, "turn"
    turns = [t for t in conversation.turns() if t.hadm_id in admissions]
    return turns, "admission_expanded"


def _pair_record(
    *,
    subject_id: str,
    qa_id: str,
    question: str,
    passage: str,
    label: int,
    hadm_id: str,
    turn_number: int,
    kind: str,
    split: str,
) -> dict[str, Any]:
    return {
        "subject_id": subject_id,
        "qa_id": qa_id,
        "question": question,
        "passage": passage,
        "label": label,
        "hadm_id": hadm_id,
        "turn_number": turn_number,
        "kind": kind,
        "split": split,
    }


def _default_index(subject_id: str) -> PatientIndex:
    # cache_path=None: the shared ST cache save() rewrites the whole npz on
    # every query miss and becomes the long pole. Pair mining does not need
    # a persisted embed cache (story forbids editing embedders.py).
    return PatientIndex(subject_id, backend="arctic-s", cache_path=None)


def _assert_not_eval(subject_id: str) -> None:
    if subject_id in EVAL_TRIO:
        raise RuntimeError(f"eval-trio subject {subject_id!r} must not enter pair building")


def _pairs_for_subject(
    subject_id: str,
    *,
    split_name: str,
    root: object | None,
    hardneg_k: int,
    index_factory: Callable[[str], Any],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    _assert_not_eval(subject_id)
    conversation = load_conversation(subject_id, root)
    items = load_qa(subject_id, root)
    index = index_factory(subject_id)
    index.build(conversation)

    pairs: list[dict[str, Any]] = []
    counts = {"n_gold_turn": 0, "n_gold_admission_expanded": 0, "n_hardneg": 0}
    for item in items:
        evidence = item.get("evidence") or {}
        admissions = [str(a) for a in (evidence.get("admissions") or [])]
        if not admissions:
            continue
        qa_id = str(item["qa_id"])
        question = str(item["question"])
        gold, grain = gold_turns_for_item(conversation, evidence)
        gold_admissions = set(admissions)
        raw_turn_ids = evidence.get("turn_ids")
        gold_turn_ids = {int(t) for t in raw_turn_ids} if raw_turn_ids else None

        for turn in gold:
            rec = _pair_record(
                subject_id=subject_id,
                qa_id=qa_id,
                question=question,
                passage=format_turn(turn),
                label=1,
                hadm_id=turn.hadm_id,
                turn_number=turn.turn_number,
                kind="gold",
                split=split_name,
            )
            rec["grain"] = grain
            pairs.append(rec)
            if grain == "turn":
                counts["n_gold_turn"] += 1
            else:
                counts["n_gold_admission_expanded"] += 1

        hits: list[RetrieveItem] = index.search(question, k=hardneg_k)
        kept = 0
        for hit in hits:
            if is_gold_hit(hit.session_id, hit.turn_ids, gold_admissions, gold_turn_ids):
                continue
            turn_number = int(hit.turn_ids[0]) if hit.turn_ids else -1
            pairs.append(
                _pair_record(
                    subject_id=subject_id,
                    qa_id=qa_id,
                    question=question,
                    passage=hit.text,
                    label=0,
                    hadm_id=str(hit.session_id),
                    turn_number=turn_number,
                    kind="hardneg",
                    split=split_name,
                )
            )
            counts["n_hardneg"] += 1
            kept += 1
            if kept >= hardneg_k:
                break
    return pairs, counts


def build_pairs(
    split: dict[str, Any],
    root: object | None = None,
    *,
    hardneg_k: int = HARDNEG_K,
    out_dir: Path | str = DEFAULT_OUT_DIR,
    index_factory: Callable[[str], Any] | None = None,
    only_subjects: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Write train/dev jsonl + manifest. Returns the manifest dict."""
    factory = index_factory or _default_index
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for sid in list(split.get("train", [])) + list(split.get("dev", [])):
        _assert_not_eval(str(sid))
    for sid in split.get("eval", []):
        if only_subjects is not None and str(sid) in set(only_subjects):
            raise RuntimeError(f"refusing to mine pairs on eval subject {sid!r}")

    train_path = out / "train_pairs.jsonl"
    dev_path = out / "dev_pairs.jsonl"
    totals = {
        "n_gold_turn": 0,
        "n_gold_admission_expanded": 0,
        "n_hardneg": 0,
    }
    written_subjects: list[str] = []

    def _load_done(path: Path) -> list[str]:
        if not path.is_file():
            return []
        return [str(s) for s in json.loads(path.read_text(encoding="utf-8"))]

    def _rewrite_without(path: Path, drop: set[str]) -> None:
        if not path.is_file() or not drop:
            return
        kept = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if str(rec["subject_id"]) not in drop:
                kept.append(line)
        path.write_text(("\n".join(kept) + ("\n" if kept else "")), encoding="utf-8")

    for split_name, path in (("train", train_path), ("dev", dev_path)):
        subjects = [str(s) for s in split.get(split_name, [])]
        if only_subjects is not None:
            allow = set(only_subjects)
            subjects = [s for s in subjects if s in allow]
        done_path = out / f"{split_name}_done.json"
        already = set(_load_done(done_path)) if only_subjects is None else set()
        # Drop any partial subject left in jsonl but not marked done.
        if path.is_file() and only_subjects is None:
            present: set[str] = set()
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    present.add(str(json.loads(line)["subject_id"]))
            _rewrite_without(path, present - already)
        handle = path.open("a" if already else "w", encoding="utf-8")
        done_list = list(already)
        try:
            for i, sid in enumerate(subjects, 1):
                if sid in already:
                    written_subjects.append(sid)
                    print(f"[{split_name} {i}/{len(subjects)}] skip {sid} (already written)", flush=True)
                    continue
                print(f"[{split_name} {i}/{len(subjects)}] {sid} …", flush=True)
                pairs, counts = _pairs_for_subject(
                    sid,
                    split_name=split_name,
                    root=root,
                    hardneg_k=hardneg_k,
                    index_factory=factory,
                )
                for rec in pairs:
                    if rec["subject_id"] in EVAL_TRIO:
                        raise RuntimeError("eval-trio subject leaked into a pair record")
                    handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
                handle.flush()
                done_list.append(sid)
                done_path.write_text(json.dumps(done_list) + "\n", encoding="utf-8")
                if sid not in written_subjects:
                    written_subjects.append(sid)
                print(
                    f"  gold_turn={counts['n_gold_turn']} "
                    f"gold_adm={counts['n_gold_admission_expanded']} "
                    f"hardneg={counts['n_hardneg']}",
                    flush=True,
                )
        finally:
            handle.close()

    expected = set(str(s) for s in split.get("train", [])) | set(str(s) for s in split.get("dev", []))
    if only_subjects is None and set(written_subjects) != expected:
        missing = expected - set(written_subjects)
        extra = set(written_subjects) - expected
        raise RuntimeError(f"subject set mismatch; missing={sorted(missing)} extra={sorted(extra)}")

    totals = {"n_gold_turn": 0, "n_gold_admission_expanded": 0, "n_hardneg": 0}
    for path in (train_path, dev_path):
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            kind = rec.get("kind")
            if kind == "hardneg":
                totals["n_hardneg"] += 1
            elif kind == "gold":
                if rec.get("grain") == "admission_expanded":
                    totals["n_gold_admission_expanded"] += 1
                else:
                    totals["n_gold_turn"] += 1

    manifest = {
        "hardneg_k": hardneg_k,
        "n_gold_turn": totals["n_gold_turn"],
        "n_gold_admission_expanded": totals["n_gold_admission_expanded"],
        "n_hardneg": totals["n_hardneg"],
        "subject_ids": sorted(written_subjects),
        "train_subjects": [str(s) for s in split.get("train", [])]
        if only_subjects is None
        else [s for s in split.get("train", []) if s in set(only_subjects)],
        "dev_subjects": [str(s) for s in split.get("dev", [])]
        if only_subjects is None
        else [s for s in split.get("dev", []) if s in set(only_subjects)],
        "eval_subjects_held_out": list(EVAL_TRIO),
        "grain": split.get("grain", "mixed-turn-preferred"),
        "grain_rule": split.get("grain_rule"),
    }
    (out / "pairs_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
