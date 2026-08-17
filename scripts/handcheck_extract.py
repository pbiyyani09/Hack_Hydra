#!/usr/bin/env python
"""scripts/handcheck_extract.py — (re)generate the E2-S3 formal hand-check
gate artifacts: `fixtures/handcheck/facts.jsonl` and a `CHECKLIST.md`
skeleton with the `human: ok/bad` column left **empty**.

This is the formal gate script (`collaborative/design/stories/E2/E2-S3.md`),
distinct from the lighter leftover-gate sample
`scripts/generate_extract_eyeball.py` already produces (that script's own
docstring says so explicitly). This script never writes
`fixtures/handcheck/PASSED` — that is a human act (or the Evidence owner
acting as second reader), performed only after reading `CHECKLIST.md`'s real
rows against their source turns and filling the `human` column by hand. See
`src/medmemgraph/pipeline/scale_gate.py`'s module docstring for the full
discipline this script is one half of.

**Real inference is the default** (`extract.py`'s contract: no `dry_run` ->
a real `llm.complete()` call; a missing key raises `llm.MissingAPIKeyError`
rather than silently degrading). Pass `--dry-run` to force the deterministic
rule-based fallback instead (offline, zero API cost — useful for
re-generating the skeleton's *shape* without spending anything, though a
`PASSED` gate should always be checked against a real-inference run).

Default subject/admissions (`10056223`, anchor `26605038`, extra
`20971116`) intentionally match `scripts/generate_extract_eyeball.py`'s own
defaults — the two admissions this script needs were already extracted for
real while building that artifact, so `medmemgraph.llm`'s disk cache
(`data/llm_cache/`) makes a default re-run of this script a $0.00 cache hit,
not a fresh paid call. This is NOT guaranteed to be byte-identical to a
prior run's output, though: the cache is keyed on (model, system, prompt,
schema, temperature), so a code change that alters the system prompt or
schema (e.g. a predicate-vocabulary or handling-rule fix) changes the key
and forces a fresh call the next time *that* combination runs — real
`temperature=0.0` calls have also been observed to vary slightly between
two genuinely-fresh calls with an identical prompt (see
`samples/extract-eyeball-10056223.md`, 'Honest read on quality' item 3).
This script's own "Known systematic findings" section below is generated
FROM WHATEVER THIS RUN ACTUALLY PRODUCED, not copied from that older
artifact, precisely because of this — see `_check_known_findings_status()`.
Pass `--subject`/`--hadm-id`/`--extra-hadm-ids` to point this at a
different patient instead (a fresh subject means fresh, real, billed
calls).

Usage:
    # real inference (default, cache-hit for the default subject) ->
    # fixtures/handcheck/facts.jsonl + fixtures/handcheck/CHECKLIST.md
    uv run python scripts/handcheck_extract.py

    # rule-based fallback (explicit opt-in, zero API calls)
    uv run python scripts/handcheck_extract.py --dry-run
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import json
from pathlib import Path

from medmemgraph import llm
from medmemgraph.contracts import PREDICATES, ClinicalFact
from medmemgraph.pipeline.extract import Extractor
from medmemgraph.pipeline.loader import Admission, Conversation, Turn, load_conversation
from medmemgraph.pipeline.normalize import resolve_time
from medmemgraph.pipeline.scale_gate import (
    CHECKLIST_FILENAME,
    DEFAULT_HANDCHECK_DIR,
    FACTS_FILENAME,
    MIN_FACTS,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _check_known_findings_status(extractor: Extractor) -> tuple[str, str]:
    """Evidence-based status for the two review prompts, computed from
    THIS run's own `extractor.assertion_log` plus the live
    `contracts.PREDICATES` set — never a hardcoded claim about whether
    either finding is "still open", since that is exactly the kind of
    stale assertion this script must not make (a fix landing in
    `pipeline/extract.py`/`pipeline/normalize.py` after this docstring was
    written would silently make a hardcoded claim wrong). Returns
    `(prn_status, predicate_vocab_status)`, each a short prose line to
    render directly under its review prompt.
    """
    emitted_prn = [e for e in extractor.assertion_log if e.decision == "emitted" and e.prn]
    dropped_prn_conditional = [
        e
        for e in extractor.assertion_log
        if e.decision == "dropped_by_rule" and e.i2b2_tag == "conditional"
    ]
    if emitted_prn:
        prn_status = (
            f"**Evidence from THIS run:** {len(emitted_prn)} PRN-tagged medication candidate(s) "
            f"were EMITTED (not dropped) in this run's own admissions, e.g. "
            f"`{emitted_prn[0].predicate_phrase} {emitted_prn[0].object_name}` (turn "
            f"{emitted_prn[0].turn_ids}) — consistent with a fix already being active in this "
            f"checkout of `pipeline/extract.py`. Please still confirm the emitted fact's "
            f"clinical framing (a bare `asserted` `TAKES_MEDICATION`, with the PRN nature "
            f"recorded only in the side log, never on the frozen `ClinicalFact` wire contract) "
            f"is the right call, not just that it is no longer silently dropped."
        )
    elif dropped_prn_conditional:
        prn_status = (
            f"**Evidence from THIS run:** {len(dropped_prn_conditional)} medication candidate(s) "
            f"were still dropped as `conditional`, e.g. `{dropped_prn_conditional[0].object_name}` "
            f"(turn {dropped_prn_conditional[0].turn_ids}) — this run does not show the fix "
            f"described above as active; please treat prompt 1 as fully open."
        )
    else:
        prn_status = (
            "**Evidence from THIS run:** no PRN-shaped `conditional` drop AND no PRN-flagged "
            "emission occurred in this run's own admissions either way — this run is silent on "
            "whether the fix is active; treat prompt 1 as open until independently confirmed "
            "(e.g. against `samples/extract-eyeball-10056223.md`'s own admission "
            "`20971116` turn 52)."
        )

    if "HAD_INCIDENT" in PREDICATES:
        predicate_vocab_status = (
            "**Evidence from the live `contracts.PREDICATES` set (checked when this file was "
            "generated, not assumed):** `HAD_INCIDENT` IS present in the closed vocabulary — a "
            "fall-shaped candidate (`normalize.canonicalize_predicate('had a fall')` and its "
            "documented alias list) would now canonicalize rather than `dropped_no_predicate`. "
            "This patient's own two admissions run here do not happen to contain a fall-shaped "
            "utterance, so this is a vocabulary-level check, not a same-run behavioral "
            "reproduction — please still confirm `HAD_INCIDENT`'s scope (deliberately narrow: "
            "fall-shaped phrasing only, per `normalize.py`'s own comment) is the right level of "
            "vocabulary extension, not just that SOME extension landed."
        )
    else:
        predicate_vocab_status = (
            "**Evidence from the live `contracts.PREDICATES` set:** no fall-specific predicate "
            "is present — the vocabulary gap `samples/extract-eyeball-10056223.md` originally "
            "found (`'had a fall'` -> `dropped_no_predicate`) has not been closed; treat prompt "
            "2 as fully open."
        )

    return prn_status, predicate_vocab_status


def _run_admission_catching_errors(
    extractor: Extractor, conversation: Conversation, admission: Admission
) -> tuple[list[ClinicalFact], Exception | None]:
    """Same discipline as `generate_extract_eyeball.py`'s own helper: a
    per-admission `llm.LLMError` must not abort the whole run before
    anything is written, nor be swallowed silently — it is reported in the
    generated `CHECKLIST.md` instead."""
    try:
        facts = extractor.extract(conversation, admission)
        return facts, None
    except llm.LLMError as exc:
        return [], exc


def _turn_lookup(order: list[Admission]) -> dict[str, dict[int, Turn]]:
    return {adm.hadm_id: {t.turn_number: t for t in adm.turns()} for adm in order}


def _snippet(text: str, limit: int = 160) -> str:
    text = text.strip().replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _write_facts_jsonl(facts: list[ClinicalFact], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for fact in facts:
            fh.write(json.dumps(dataclasses.asdict(fact), sort_keys=False))
            fh.write("\n")


def _write_checklist(
    *,
    facts: list[ClinicalFact],
    turns_by_admission: dict[str, dict[int, Turn]],
    subject_id: str,
    admissions_used: list[str],
    extractor_kind: str,
    dropped_prn: list[tuple],
    dropped_predicate_gap: list[tuple],
    prn_status: str,
    predicate_vocab_status: str,
    path: Path,
) -> None:
    lines: list[str] = []
    lines.append("# E2-S3 hand-check gate — CHECKLIST")
    lines.append("")
    lines.append(
        "This is the formal E2-S3 hand-check gate "
        "(`collaborative/design/stories/E2/E2-S3.md`), regenerated by "
        "`scripts/handcheck_extract.py`. **Scale ingest is blocked until a "
        "human fills the `human: ok/bad` column below and writes "
        "`fixtures/handcheck/PASSED` themselves** — see "
        "`src/medmemgraph/pipeline/scale_gate.py`. No coding agent may "
        "write `PASSED`."
    )
    lines.append("")
    lines.append(f"- **subject_id:** `{subject_id}`")
    lines.append(f"- **Admissions covered:** {', '.join(admissions_used)}")
    lines.append(f"- **Extractor path used:** `{extractor_kind}`")
    lines.append(f"- **Rows:** {len(facts)} (>= {MIN_FACTS} required)")
    n_negated = sum(1 for f in facts if f.polarity == "negated")
    source_classes = sorted({f.source_class for f in facts})
    lines.append(f"- **Negated rows:** {n_negated} (>= 1 required)")
    lines.append(f"- **source_class values present:** {source_classes} (>= 2 required if both speakers occur)")
    lines.append(
        "- **No `combined_conversation.json` is vendored here** — only short "
        "(<=160 char) verbatim snippets of the source turn are quoted below, "
        "per this story's own discipline."
    )
    lines.append("")

    lines.append("## Required review prompts — please specifically re-check these two")
    lines.append("")
    lines.append(
        "These are the two **systematic** (not one-off) findings the ORIGINAL "
        "real-inference eyeball run surfaced (`samples/extract-eyeball-10056223.md`, "
        "'Honest read on quality' items 1 and 2) — recurring across multiple admissions "
        "at the time that artifact was generated. **A status line, computed fresh from "
        "THIS run rather than copied from that older artifact, follows each prompt** — "
        "code may have changed between when that artifact was written and when this "
        "checklist was generated, so do not assume either finding is still open just "
        "because it is listed here; read the status line. Please form an explicit "
        "opinion on each regardless of status."
    )
    lines.append("")
    lines.append(
        "1. **PRN (\"as needed\") medications being tagged `conditional` and dropped "
        "entirely** rather than emitted as a real, currently-active prescription "
        "(e.g. \"the new diuretics and oxycodone for pain as needed\" -> "
        "`TAKES_MEDICATION oxycodone` skipped, not emitted with a caveat). Is dropping "
        "a PRN order the correct behavior for a medication list, or should the "
        "`conditional` tag's definition be loosened for PRN language specifically? See "
        "decisions/002's handling-rules table."
    )
    lines.append(f"   {prn_status}")
    lines.append(
        "2. **Predicate-vocabulary coverage.** A clinically real candidate fact (e.g. "
        "\"you had a fall this morning\") can fail to canonicalize onto the closed "
        "`PREDICATES` set (`normalize.canonicalize_predicate` returns `None`) and is "
        "dropped as `dropped_no_predicate` — silently, from a downstream point of "
        "view. Does the closed predicate vocabulary (`normalize.PREDICATE_"
        "DEFINITIONS`) need extending before a full-corpus run, or is dropping the "
        "right call for anything outside it?"
    )
    lines.append(f"   {predicate_vocab_status}")
    lines.append("")

    # Supplementary detail tables — ONLY rendered when this run's own
    # assertion_log actually contains the relevant case (the one-line
    # evidence statements above already cover the "nothing of this shape
    # happened" case precisely, including the emitted-not-dropped case
    # these tables cannot show at all, so no redundant "(none)" filler is
    # printed here when they're empty).
    if dropped_prn:
        lines.append("### PRN-medication drops observed in *this* run")
        lines.append("")
        lines.append("| admission | turn | predicate/object | reason | source snippet |")
        lines.append("|---|---|---|---|---|")
        for hadm_id, turn_ids, predicate_phrase, object_name, reason, snippet in dropped_prn:
            lines.append(f"| {hadm_id} | {turn_ids} | {predicate_phrase}/{object_name} | {reason} | {snippet} |")
        lines.append("")

    if dropped_predicate_gap:
        lines.append("### Predicate-vocabulary-gap drops observed in *this* run")
        lines.append("")
        lines.append("| admission | turn | predicate phrase | object | source snippet |")
        lines.append("|---|---|---|---|---|")
        for hadm_id, turn_ids, predicate_phrase, object_name, snippet in dropped_predicate_gap:
            lines.append(f"| {hadm_id} | {turn_ids} | {predicate_phrase!r} | {object_name} | {snippet} |")
        lines.append("")

    lines.append("## Facts to hand-check")
    lines.append("")
    lines.append(
        "| # | fact_id | predicate | object | polarity | source_class | turn_id | "
        "turn_time | snippet | human: ok/bad | notes |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for i, fact in enumerate(facts, start=1):
        turn_map = turns_by_admission.get(fact.session_id, {})
        turn = turn_map.get(fact.turn_ids[0]) if fact.turn_ids else None
        snippet = _snippet(turn.text) if turn else ""
        turn_time = turn.time if turn else ""
        object_display = fact.object.name
        if fact.subject.type != "Patient":
            # CURRENT_DOSAGE_OF-shaped facts: subject is the medication, not
            # the patient — show both so the reviewer isn't reading a bare
            # dose with no medication name attached.
            object_display = f"{fact.subject.name} → {fact.object.name}"
        note = ""
        anchor_iso = resolve_time("", turn.time)[0] if turn else ""
        if anchor_iso and fact.valid_from != anchor_iso:
            note = f"relative-time resolved: valid_from={fact.valid_from} (turn time {anchor_iso})"
        lines.append(
            f"| {i} | `{fact.fact_id}` | {fact.predicate} | {object_display} | "
            f"{fact.polarity} | {fact.source_class} | {fact.turn_ids} | {turn_time} | "
            f"{snippet} |  | {note} |"
        )
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _generate(args: argparse.Namespace) -> tuple[Path, Path]:
    conversation = load_conversation(args.subject, args.root)
    admissions_by_id = {a.hadm_id: a for a in conversation.admissions}
    anchor = admissions_by_id[args.hadm_id]

    extractor = Extractor(dry_run=args.dry_run)
    order = [anchor] + [
        admissions_by_id[h] for h in args.extra_hadm_ids if h in admissions_by_id and h != args.hadm_id
    ]

    all_facts: list[ClinicalFact] = []
    errors: list[tuple[str, Exception]] = []
    for adm in order:
        facts, err = _run_admission_catching_errors(extractor, conversation, adm)
        all_facts.extend(facts)
        if err is not None:
            errors.append((adm.hadm_id, err))
        if len(all_facts) >= args.max_rows and adm.hadm_id != args.hadm_id:
            break

    anchor_facts = sorted((f for f in all_facts if f.session_id == args.hadm_id), key=lambda f: f.turn_ids[0])
    extra_facts = sorted(
        (f for f in all_facts if f.session_id != args.hadm_id), key=lambda f: (f.session_id, f.turn_ids[0])
    )
    ordered_facts = anchor_facts + extra_facts
    selected = ordered_facts[: args.max_rows]

    if len(selected) < MIN_FACTS:
        print(
            f"WARNING: only {len(selected)} facts extracted (< {MIN_FACTS} required by "
            "the gate) — pass more/other --extra-hadm-ids or a different --subject.",
        )

    turns_by_admission = _turn_lookup(order)

    dropped = [e for e in extractor.assertion_log if e.decision != "emitted"]
    dropped_prn: list[tuple] = []
    dropped_predicate_gap: list[tuple] = []
    for e in dropped:
        turn = turns_by_admission.get(e.session_id, {}).get(e.turn_ids[0]) if e.turn_ids else None
        snippet = _snippet(turn.text) if turn else ""
        if e.decision == "dropped_by_rule" and e.i2b2_tag == "conditional" and "as needed" in (turn.text.lower() if turn else ""):
            dropped_prn.append((e.session_id, e.turn_ids, e.predicate_phrase, e.object_name, e.reason, snippet))
        elif e.decision == "dropped_no_predicate":
            dropped_predicate_gap.append((e.session_id, e.turn_ids, e.predicate_phrase, e.object_name, snippet))

    prn_status, predicate_vocab_status = _check_known_findings_status(extractor)

    handcheck_dir = Path(args.handcheck_dir) if args.handcheck_dir else DEFAULT_HANDCHECK_DIR
    facts_path = handcheck_dir / FACTS_FILENAME
    checklist_path = handcheck_dir / CHECKLIST_FILENAME

    _write_facts_jsonl(selected, facts_path)
    _write_checklist(
        facts=selected,
        turns_by_admission=turns_by_admission,
        subject_id=args.subject,
        admissions_used=[adm.hadm_id for adm in order],
        extractor_kind=extractor.kind,
        dropped_prn=dropped_prn,
        dropped_predicate_gap=dropped_predicate_gap,
        prn_status=prn_status,
        predicate_vocab_status=predicate_vocab_status,
        path=checklist_path,
    )

    print(
        f"wrote {facts_path} ({len(selected)} rows) and {checklist_path} — "
        f"extractor.kind={extractor.kind}, run date {_dt.date.today().isoformat()}. "
        f"PASSED was NOT written (never is, by this script)."
    )
    if errors:
        print(f"NOTE: {len(errors)} admission(s) raised llm.LLMError and were skipped: {errors}")
    return facts_path, checklist_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", default="10056223", help="MedLoCoMo subject_id (default: matches the leftover-gate eyeball's own subject, cache-hit-cheap to re-run)")
    parser.add_argument("--hadm-id", default="26605038", help="anchor admission (hadm_id)")
    parser.add_argument(
        "--extra-hadm-ids",
        nargs="*",
        default=["20971116"],
        help="same-patient admissions to pull additional facts from, in order, until --max-rows is reached",
    )
    parser.add_argument("--max-rows", type=int, default=30)
    parser.add_argument("--root", default=None, help="MedLoCoMo corpus root (default: $MEDLOCOMO_ROOT or data/medlocomo)")
    parser.add_argument(
        "--handcheck-dir",
        default=None,
        help="override fixtures/handcheck/ (default: pipeline.scale_gate.DEFAULT_HANDCHECK_DIR)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="force the deterministic rule-based fallback (Extractor(dry_run=True)) — offline, zero API cost",
    )
    args = parser.parse_args()
    _generate(args)


if __name__ == "__main__":
    main()
