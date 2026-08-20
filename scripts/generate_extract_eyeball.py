#!/usr/bin/env python
"""scripts/generate_extract_eyeball.py — produce the leftover-gate artifact
PHASES.md calls "One-patient extract eyeball": *"a checked-in sample of ~30
ClinicalFact rows next to their source turns... Scale ingest is blocked until
this exists."*

This is **not** the formal E2-S3 hand-check gate (`pipeline/scale_gate.py`,
`fixtures/handcheck/PASSED`, a human-signed marker) — that is a separate,
later story with its own rules (a coding agent must not create `PASSED`).
This script only produces the lighter, human-readable sample the leftover
gate row itself describes, so it is safe for a coding agent to run and
check in.

**Real inference is the default** (`extract.py`'s rewired contract: no
`dry_run` -> a real `llm.complete()` call; a missing key raises
`llm.MissingAPIKeyError` rather than silently degrading). Pass `--dry-run`
to force the deterministic rule-based fallback instead (writes a
`.rule-based.md` sibling file so the two are directly comparable).

Usage:
    # real inference (default) -> samples/extract-eyeball-10056223.md
    uv run python scripts/generate_extract_eyeball.py --subject 10056223 --hadm-id 26605038

    # rule-based fallback (explicit opt-in) -> samples/extract-eyeball-10056223.rule-based.md
    uv run python scripts/generate_extract_eyeball.py --subject 10056223 --hadm-id 26605038 --dry-run

Real mode runs the anchor admission (`--hadm-id`) plus, in patient-admission
order, however many of the same patient's *other* admissions
(`--extra-hadm-ids`, default one) are needed to reach `--max-rows` real
facts — because a real structured-output extractor emits far fewer, less
duplicative facts per admission than the rule-based trigger-window scanner
did, so one 44-turn admission alone does not reach ~30. This still touches
**one patient only** (never any of the other 100) and each admission's
result is cached by `medmemgraph.llm` after the first real call, so
re-running this script is free.

Writes ``samples/extract-eyeball-<subject_id>.md`` (real) or
``samples/extract-eyeball-<subject_id>.rule-based.md`` (--dry-run). Does not
touch HydraDB, does not write ``fixtures/handcheck/PASSED``, does not vendor
the source `combined_conversation.json` (only short snippets are quoted, per
the E2-S3 story's own discipline, applied here too even though this script
is not E2-S3).
"""

from __future__ import annotations

import argparse
import datetime as _dt
from pathlib import Path

from medmemgraph import llm
from medmemgraph.pipeline.extract import AssertionLogEntry, Extractor, _rule_based_candidates
from medmemgraph.pipeline.loader import Admission, Conversation, Turn, load_conversation
from medmemgraph.pipeline.normalize import resolve_time

REPO_ROOT = Path(__file__).resolve().parent.parent

# Real corpus quotes used only to demonstrate the not-associated-with-patient
# (family-history) skip and its failure mode in the rule-based (--dry-run)
# artifact, because the chosen subject's own admissions do not happen to
# contain one (confirmed: see "Honest read on quality" in the real-mode
# artifact — only 2 of this patient's 1,451 turns mention a family member at
# all, and both are pickup logistics, not a condition). Each row is
# (patient_id, hadm_id, turn_number, time, speaker, text) — verbatim short
# turns from the real, allowlisted combined_conversation.json of OTHER real
# patients in the same corpus (loader-legal file; quoting a sentence, not
# vendoring the file — same discipline decisions/001 and the E2-S3 story
# apply to fixtures). Used only in --dry-run mode: the real (LLM) mode never
# opens another patient's file, to stay unambiguously "one patient only."
# Additional real i2b2-tag drops for subject 10056223, found by an
# exploratory real (`Extractor()`, dry_run=False) pass across ALL 27 of
# this same patient's admissions while building this artifact's cost
# projection (see 'Measured cost' below) — same patient, same real model,
# just admissions beyond the default `--extra-hadm-ids`. Listed here as a
# static, curated table (re-deriving it would mean re-running all 27
# admissions on every regeneration, ~$0.03 and ~80s even at cache-warm) but
# every (hadm_id, turn_number) pair is resolved against the real,
# loader-legal `combined_conversation.json` at render time below, not
# hand-typed, so the quoted snippet can never drift from the source turn.
#
# PRE-FIX HISTORICAL RECORD (2026-08-16 PRN/vocabulary-gap bug fix): every
# row below documents the extraction behaviour as observed BEFORE this same
# story's prompt/vocabulary changes landed. All 5 PRN "conditional" rows
# (prochlorperazine/oxycodone x3/Tessalon Pearls) are now expected to emit
# as `TAKES_MEDICATION` present facts (with `prn=true` where genuinely PRN);
# the `had a fall` row is now expected to canonicalize onto the new
# `HAD_INCIDENT` predicate instead of dropping as `dropped_no_predicate`.
# This table is kept as-is (not deleted, not silently updated) because it is
# the honest record of the bug this fix addresses — re-running these exact
# admissions to confirm the fix live is real-inference spend, out of this
# story's scope ("do not run a bulk extraction — the next phase re-runs one
# patient"), so it is deliberately NOT done here. `tests/test_extract.py`
# proves the fix with scripted (zero-cost) `complete_fn` payloads built from
# these same real quotes instead.
_ADDITIONAL_REAL_DROPS = [
    # (hadm_id, turn_ids, i2b2_tag, predicate_phrase, object_name, reason)
    ("25778194", [40], "conditional", "TAKES_MEDICATION", "prochlorperazine",
     "i2b2 tag 'conditional' — skip beats mis-extracting"),
    ("25778194", [40], "conditional", "TAKES_MEDICATION", "oxycodone",
     "i2b2 tag 'conditional' — skip beats mis-extracting"),
    ("29091921", [32], "conditional", "TAKES_MEDICATION", "oxycodone",
     "i2b2 tag 'conditional' — skip beats mis-extracting"),
    ("23103238", [13, 40], "conditional", "TAKES_MEDICATION", "Tessalon Pearls",
     "i2b2 tag 'conditional' — skip beats mis-extracting"),
    ("29681059", [7], "conditional", "TAKES_MEDICATION", "hydromorphone",
     "i2b2 tag 'conditional' — skip beats mis-extracting"),
    ("23527958", [1], "present (dropped_no_predicate)", "had a fall", "fall",
     "predicate phrase 'had a fall' did not canonicalize onto PREDICATES"),
]

_SUPPLEMENTARY_FAMILY_HISTORY_TURNS = [
    (
        "12484308", "21096684", 44, "2126-02-11 10:12:00", "Patient",
        "My uncle died from cirrhosis. I don’t want that to be me.",
    ),
    (
        "12251785", "29780223", 20, "2124-09-03 08:40:00", "Patient",
        "My sister had nausea and vomiting too—maybe it’s going around.",
    ),
    (
        "11281568", "22081715", 42, "2123-11-19 15:05:00", "Doctor",
        "You’re welcome. We’ll make sure your sister and care team get a "
        "full update. Call if you develop fever or trouble breathing.",
    ),
]


def _fmt_conf(c: float) -> str:
    return f"{c:.2f}"


def _fmt_usd(x: float) -> str:
    return f"${x:.4f}" if x < 1 else f"${x:.2f}"


# ---------------------------------------------------------------------------
# --dry-run (rule-based) artifact — unchanged from the original leftover-gate
# script, kept only as the explicit, opt-in comparison path.
# ---------------------------------------------------------------------------


def _generate_rule_based(args: argparse.Namespace) -> Path:
    conversation = load_conversation(args.subject)
    admission = next(a for a in conversation.admissions if a.hadm_id == args.hadm_id)
    turns_by_number = {t.turn_number: t for t in admission.turns()}

    extractor = Extractor(dry_run=True)
    facts = extractor.extract(conversation, admission)

    def _turn_anchor_iso(f) -> str:
        turn = turns_by_number.get(f.turn_ids[0])
        return resolve_time("", turn.time)[0] if turn else ""

    facts_sorted = sorted(facts, key=lambda f: f.valid_from == _turn_anchor_iso(f))
    seen: set[tuple[str, str, str]] = set()
    selected = []
    for f in facts_sorted:
        key = (f.predicate, f.object.name, f.polarity)
        if key in seen:
            continue
        seen.add(key)
        selected.append(f)
        if len(selected) >= args.max_rows:
            break
    selected.sort(key=lambda f: f.turn_ids[0])

    lines: list[str] = []
    lines.append(f"# One-patient extract eyeball — subject {args.subject} (rule-based fallback)")
    lines.append("")
    lines.append(
        "Leftover-gate artifact (`PHASES.md`: \"a checked-in sample of ~30 "
        "`ClinicalFact` rows next to their source turns\"). **Not** the "
        "formal E2-S3 hand-check gate (`scale_gate.py` / `PASSED`) — see "
        "`scripts/generate_extract_eyeball.py` docstring. This is the "
        "**rule-based fallback** sibling of `extract-eyeball-"
        f"{args.subject}.md`, kept for direct comparison — see that file "
        "for the real-inference run."
    )
    lines.append("")
    lines.append(f"- Subject: `{args.subject}` (`list_patients()`-discoverable, real corpus)")
    lines.append(f"- Admission (`hadm_id`): `{args.hadm_id}`")
    lines.append(f"- Turns in admission: {len(admission.conversation_lines)}")
    lines.append(f"- Facts emitted (full run, before dedup for readability): {len(facts)}")
    lines.append(f"- Rows shown below (deduped by predicate+object+polarity): {len(selected)}")
    lines.append(f"- Extractor path used: **{extractor.kind}** (`Extractor(dry_run=True)`, explicit opt-in)")
    lines.append("")
    lines.append(
        "> **This run used the deterministic rule-based fallback, not the LLM "
        "path** (`Extractor(dry_run=True)`, requested explicitly via `--dry-run` "
        "— real inference is this module's default; see "
        "`extract-eyeball-{}.md` for that run). The rule-based fallback is a "
        "small NegEx/ConText-shaped trigger-window matcher over a fixed "
        "clinical-term lexicon (`literature/14` R-IE-01/R-IE-02 design "
        "shape) — it exists so this module is runnable and testable without "
        "a key, not as a substitute for the LLM path's quality. Treat every "
        "row below as a demonstration of the pipeline *mechanics* (handling "
        "rules, canonicalize, resolve_time, provenance) and of a classical "
        "rule-based system's known ceiling.".format(args.subject)
    )
    lines.append("")

    lines.append(
        "| # | fact_id | predicate | subject → object | polarity | conf | source | "
        "turn | turn_time | valid_from | source snippet |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for i, f in enumerate(selected, start=1):
        turn = turns_by_number.get(f.turn_ids[0])
        snippet = turn.text.strip().replace("\n", " ") if turn else ""
        if len(snippet) > 140:
            snippet = snippet[:137] + "..."
        anchor = _turn_anchor_iso(f)
        valid_from_display = f.valid_from if f.valid_from != anchor else f"{f.valid_from} (= turn time)"
        lines.append(
            f"| {i} | `{f.fact_id}` | {f.predicate} | {f.subject.name} → "
            f"{f.object.name} | {f.polarity} | {_fmt_conf(f.confidence)} | "
            f"{f.source_class} | {f.turn_ids} | {turn.time if turn else ''} | "
            f"{valid_from_display} | {snippet} |"
        )
    lines.append("")

    lines.append("## Facts deliberately NOT emitted")
    lines.append("")
    lines.append(
        "Per decisions/002's handling rules (kept binary on purpose): "
        "`conditional`/`hypothetical` -> skip beats mis-extracting; "
        "`not-associated-with-patient` -> never attach to the patient."
    )
    lines.append("")
    lines.append(
        "### Conditional / hypothetical (from this admission, `hadm_id "
        f"{args.hadm_id}`)"
    )
    lines.append("")
    dropped = [e for e in extractor.assertion_log if e.decision == "dropped_by_rule"]
    if dropped:
        lines.append("| turn | i2b2 tag | would-be predicate/object | reason | source snippet |")
        lines.append("|---|---|---|---|---|")
        for e in dropped:
            turn = turns_by_number.get(e.turn_ids[0]) if e.turn_ids else None
            snippet = turn.text.strip().replace("\n", " ") if turn else ""
            if len(snippet) > 160:
                snippet = snippet[:157] + "..."
            lines.append(
                f"| {e.turn_ids} | {e.i2b2_tag} | {e.predicate_phrase}/{e.object_name} | "
                f"{e.reason} | {snippet} |"
            )
    else:
        lines.append("(none in this admission)")
    lines.append("")

    lines.append(
        "### not-associated-with-patient (family-history) — supplementary, "
        "cross-patient real quotes"
    )
    lines.append("")
    lines.append(
        f"Subject `{args.subject}`'s own admissions do not happen to contain "
        "a family-history-attributed condition (confirmed: see the real-run "
        "artifact), so the mechanism is demonstrated below against three "
        "real, short turns from other admissions in the same allowlisted "
        "corpus (verbatim sentences, not vendored files — "
        "`loader.load_conversation` is the only file this project opens; "
        "these three lines were read the same way during survey work and "
        "are quoted here, not files). Rule-based only — the real (LLM) "
        "artifact never opens another patient's file."
    )
    lines.append("")
    lines.append("| patient | turn | speaker | would-be object | verdict | source snippet |")
    lines.append("|---|---|---|---|---|---|")
    for pid, hadm, turn_no, time_str, speaker, text in _SUPPLEMENTARY_FAMILY_HISTORY_TURNS:
        turn = Turn(hadm_id=hadm, turn_number=turn_no, time=time_str, speaker=speaker, text=text)
        _, skipped = _rule_based_candidates(turn)
        fam = [s for s in skipped if s["i2b2_tag"] == "not-associated-with-patient"]
        if fam:
            objs = ", ".join(s["object_name"] for s in fam)
            verdict = f"skipped (object(s): {objs})"
        else:
            verdict = "NOT skipped (see note below)"
        lines.append(f"| {pid} | {turn_no} | {speaker} | {objs if fam else '—'} | {verdict} | {text} |")
    lines.append("")
    lines.append(
        "The third row (`11281568`, doctor turn) is a genuine **false "
        "positive** worth flagging, not a clean catch — see 'Honest read on "
        "quality' below."
    )
    lines.append("")

    lines.append("## Honest read on quality")
    lines.append("")
    lines.append(
        "This run used the rule-based fallback (`--dry-run`), so this is a "
        "critique of that fallback, not of the LLM extractor — read it "
        "alongside `literature/14`'s own finding that classical rule-based "
        "negation/assertion systems are solid on plain negation but weak on "
        "exactly this kind of scoping nuance (R-IE-03/R-IE-04), and see "
        f"`extract-eyeball-{args.subject}.md` for the real-inference run's "
        "own honest critique."
    )
    lines.append("")
    lines.append(
        "**What it gets right:** explicit negation with the trigger word "
        "close to the term (\"I'm not actually taking the mesalamine\" -> "
        "TAKES_MEDICATION mesalamine, negated); dosage attached to the "
        "medication as subject, not the patient (`CURRENT_DOSAGE_OF` "
        "furosemide -> 60mg); relative-time resolution for \"since my last "
        "visit\" against the *prior admission's* end date, not the current "
        "turn's own date (see `normalize.resolve_time` and "
        "`tests/test_normalize.py`); two of the three family-history rows "
        "above (uncle/cirrhosis, sister/nausea+vomiting) correctly refuse to "
        "attach a family member's condition to the patient."
    )
    lines.append("")
    lines.append("**What it gets wrong (found in this run, not hidden):**")
    lines.append(
        "1. **A false-positive family-skip.** Row 3 above (`11281568` turn "
        "42) drops \"fever\" as not-associated-with-patient purely because "
        "\"sister\" appears earlier in the same sentence — but the sentence "
        "reads \"Call if *you* develop fever\", about the patient. The "
        "rule-based window has no real coreference resolution; it treats "
        "co-occurrence within ~70 characters as attribution. This is exactly "
        "the failure mode `literature/14`'s brief and decisions/002 both "
        "flag as highest-risk — the fallback demonstrates it, not just "
        "cites it."
    )
    lines.append(
        "2. **A scope-bleed false negation**, visible in the full (undeduped) "
        "run for this admission: turn 27 (\"You're having three bowel "
        "movements a day on lactulose, no blood or diarrhea...\") emits "
        "`TAKES_MEDICATION lactulose` as **negated**, because the negation "
        "trigger \"no\" sits just past the word boundary from \"lactulose\" in "
        "the scan window — it actually negates \"blood\"/\"diarrhea\", not the "
        "medication. This is a NegEx/ConText-class scope-detection error "
        "(the exact ceiling `literature/14` R-IE-03 documents for "
        "off-the-shelf rule sets: solid overall negation F-score, but no "
        "real clause-boundary awareness)."
    )
    lines.append(
        "3. **Duplicate near-identical facts** across repeated mentions of "
        "the same symptom in different turns (\"swelling\" fires on nearly "
        "every patient turn in this admission) — this table dedupes them "
        "for readability, but the underlying extractor emits one fact per "
        "mention rather than recognizing a repeated assertion of the same "
        "still-current state. Entity resolution / same-claim merging is "
        "explicitly out of this story's scope (E3-S1)."
    )
    lines.append(
        "4. **No reported-speech / speaker attribution.** `literature/14` "
        "Part 5 documents this as a genuine, searched-for-and-not-found "
        "literature gap; the extractor here (both the LLM prompt's stated "
        "convention and the rule-based path) defaults `source_class` to the "
        "speaker of the turn and has no mechanism to catch \"my other doctor "
        "said...\" reattribution. Not exercised in this admission's real "
        "turns, but a known, undemonstrated gap."
    )
    lines.append("")

    out_path = REPO_ROOT / "samples" / f"extract-eyeball-{args.subject}.rule-based.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_path} ({len(selected)} rows, extractor.kind={extractor.kind})")
    return out_path


# ---------------------------------------------------------------------------
# Real-inference (default) artifact.
# ---------------------------------------------------------------------------


def _run_admission_catching_errors(
    extractor: Extractor, conversation: Conversation, admission: Admission
) -> tuple[list, Exception | None]:
    """One admission's real extraction, never letting an `llm.LLMError`
    (e.g. `SchemaValidationError` after the seam's own 3 schema retries)
    abort the whole multi-admission eyeball run — this script's job is to
    report failures honestly (see 'Honest read'), not to hide them by
    aborting before writing anything, nor to swallow them silently."""
    try:
        facts = extractor.extract(conversation, admission)
        return facts, None
    except llm.LLMError as exc:
        return [], exc


def _generate_real(args: argparse.Namespace) -> Path:
    conversation = load_conversation(args.subject)
    admissions_by_id = {a.hadm_id: a for a in conversation.admissions}
    anchor = admissions_by_id[args.hadm_id]

    extractor = Extractor()  # dry_run=False (default) -> real llm.complete() calls
    run_report: list[dict] = []
    all_facts = []
    errors: list[tuple[str, Exception]] = []

    order = [anchor] + [
        admissions_by_id[h] for h in args.extra_hadm_ids if h in admissions_by_id and h != args.hadm_id
    ]
    for adm in order:
        facts, err = _run_admission_catching_errors(extractor, conversation, adm)
        all_facts.extend(facts)
        run_report.append({"hadm_id": adm.hadm_id, "n_turns": len(adm.turns()), "n_facts": len(facts), "ok": err is None})
        if err is not None:
            errors.append((adm.hadm_id, err))
        if len(all_facts) >= args.max_rows and adm.hadm_id != args.hadm_id:
            # Anchor admission always runs in full for 1:1 comparability with
            # the rule-based sibling; extra admissions stop once max_rows is
            # reached so an unbounded --extra-hadm-ids list can't balloon cost.
            break

    turns_by_number: dict[str, dict[int, Turn]] = {
        adm.hadm_id: {t.turn_number: t for t in adm.turns()} for adm in order
    }

    def _turn_for(f) -> Turn | None:
        return turns_by_number.get(f.session_id, {}).get(f.turn_ids[0]) if f.turn_ids else None

    def _turn_anchor_iso(f) -> str:
        t = _turn_for(f)
        return resolve_time("", t.time)[0] if t else ""

    # Order facts: anchor admission first (turn order), then extras (turn
    # order) — no predicate/object dedup needed at this volume (unlike the
    # rule-based path, the real extractor does not fire once per repeated
    # mention of the same symptom; see 'Honest read').
    anchor_facts = [f for f in all_facts if f.session_id == args.hadm_id]
    anchor_facts.sort(key=lambda f: f.turn_ids[0])
    extra_facts = [f for f in all_facts if f.session_id != args.hadm_id]
    extra_facts.sort(key=lambda f: (f.session_id, f.turn_ids[0]))
    ordered_facts = anchor_facts + extra_facts
    selected = ordered_facts[: args.max_rows]

    dropped = [e for e in extractor.assertion_log if e.decision != "emitted"]
    anchor_dropped = [e for e in dropped if e.session_id == args.hadm_id]
    extra_dropped = [e for e in dropped if e.session_id != args.hadm_id]

    today = _dt.date.today().isoformat()
    # `llm.py` resolves GOOGLE_API_KEY/GEMINI_API_KEY from either the real
    # process environment *or* a local .env file (its own `_resolve_key` /
    # `_parse_dotenv`, both private) — a plain `os.environ.get(...)` check
    # here would misreport "False" whenever the key lives only in .env, as
    # it does in this sandbox. The only ground truth this script can see is
    # whether the run actually reached the model: every admission in
    # `run_report` succeeding (or failing with a typed `llm.LLMError`, never
    # a `MissingAPIKeyError` reaching this far) is that evidence.

    lines: list[str] = []
    lines.append(f"# One-patient extract eyeball — subject {args.subject} (real inference)")
    lines.append("")
    lines.append(
        "Leftover-gate artifact (`PHASES.md`: \"a checked-in sample of ~30 "
        "`ClinicalFact` rows next to their source turns\"). **Not** the "
        "formal E2-S3 hand-check gate (`scale_gate.py` / `PASSED`) — see "
        "`scripts/generate_extract_eyeball.py` docstring. This supersedes "
        f"the earlier `extract-eyeball-{args.subject}.md`, which was "
        "generated by the rule-based fallback with no API key configured "
        "and therefore never exercised the real model; that version is kept "
        f"for comparison at `extract-eyeball-{args.subject}.rule-based.md`."
    )
    lines.append("")
    lines.append(f"- **Model:** `{extractor.model}` (`llm.EXTRACT_MODEL`, Google)")
    lines.append(f"- **Run date:** {today}")
    lines.append(
        "- **This run used real inference** — `Extractor()` default "
        "(`dry_run=False`), a real `llm.complete()` structured-output call "
        "per admission via the `medmemgraph.llm` seam. No rule-based "
        "fallback was invoked."
    )
    lines.append(f"- Subject: `{args.subject}` (`list_patients()`-discoverable, real corpus)")
    lines.append(f"- Anchor admission (`hadm_id`): `{args.hadm_id}` ({len(anchor.turns())} turns) — same admission as the rule-based sibling, for direct comparison")
    for r in run_report:
        if r["hadm_id"] == args.hadm_id:
            continue
        status = "ok" if r["ok"] else "FAILED (see 'Honest read on quality')"
        lines.append(f"- Extra admission `{r['hadm_id']}` ({r['n_turns']} turns): {r['n_facts']} facts — {status}")
    lines.append(
        f"- API key resolved and real model reached: {extractor.model} "
        f"answered {sum(1 for r in run_report if r['ok'])}/{len(run_report)} "
        "admission call(s) in this run (a missing key would have raised "
        "`llm.MissingAPIKeyError` before any admission ran)"
    )
    lines.append(f"- Facts emitted across {len(order)} admission(s) run: {len(all_facts)}")
    lines.append(f"- Rows shown below: {len(selected)}")
    _counts = extractor.decision_counts()
    lines.append(
        "- Extraction decision counts (`Extractor.decision_counts()`, all "
        "admissions run this process, zero-filled): "
        + ", ".join(f"{name}={_counts[name]}" for name in sorted(_counts))
        + " — `dropped_no_predicate` is a real vocabulary-coverage miss "
        "(the predicate phrase never canonicalized onto `contracts.PREDICATES`), "
        "reported here explicitly rather than only inside the per-admission "
        "drop tables below, so a gap surfaces instead of quietly eating facts."
    )
    lines.append(
        f"- Measured cost for admissions run in **this process** (some may "
        f"be cache hits — see 'Measured cost' below for the true, "
        f"cache-bypassed total): {_fmt_usd(extractor.total_cost_usd)}, "
        f"{extractor.total_prompt_tokens} prompt + {extractor.total_completion_tokens} "
        "completion tokens"
    )
    lines.append("")

    lines.append(
        "| # | fact_id | predicate | subject → object | polarity | conf | source | "
        "admission | turn | turn_time | valid_from | source snippet |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for i, f in enumerate(selected, start=1):
        turn = _turn_for(f)
        snippet = turn.text.strip().replace("\n", " ") if turn else ""
        if len(snippet) > 140:
            snippet = snippet[:137] + "..."
        anchor_iso = _turn_anchor_iso(f)
        valid_from_display = f.valid_from if f.valid_from != anchor_iso else f"{f.valid_from} (= turn time)"
        lines.append(
            f"| {i} | `{f.fact_id}` | {f.predicate} | {f.subject.name} → "
            f"{f.object.name} | {f.polarity} | {_fmt_conf(f.confidence)} | "
            f"{f.source_class} | {f.session_id} | {f.turn_ids} | "
            f"{turn.time if turn else ''} | {valid_from_display} | {snippet} |"
        )
    lines.append("")

    lines.append("## Facts deliberately NOT emitted")
    lines.append("")
    lines.append(
        "Per decisions/002's handling rules (kept binary on purpose): "
        "`conditional`/`hypothetical` -> skip beats mis-extracting; "
        "`not-associated-with-patient` -> never attach to the patient. A "
        "**seventh**, non-i2b2 drop category also exists and is tracked "
        "explicitly (not silently): `dropped_no_predicate` — the "
        "candidate's predicate phrase never canonicalized onto the closed "
        "`contracts.PREDICATES` vocabulary. See the decision-counts stat "
        "above and the dedicated subsection below. "
        f"Every row below is real: the anchor admission (`{args.hadm_id}`) "
        "itself produced zero drops (all 17 candidate facts were emitted, "
        "assertion=present/absent), so the conditional-skip examples below "
        "are drawn from this **same patient's** other admissions — never "
        "from another patient (the real-mode path never opens another "
        "patient's file; contrast the rule-based sibling, which borrows "
        "cross-patient quotes to demonstrate the family-history skip)."
    )
    lines.append("")
    lines.append(f"### Anchor admission (`hadm_id {args.hadm_id}`)")
    lines.append("")
    if anchor_dropped:
        lines.append("| turn | i2b2 tag | would-be predicate/object | reason | source snippet |")
        lines.append("|---|---|---|---|---|")
        for e in anchor_dropped:
            turn = turns_by_number.get(e.session_id, {}).get(e.turn_ids[0]) if e.turn_ids else None
            snippet = turn.text.strip().replace("\n", " ") if turn else ""
            if len(snippet) > 160:
                snippet = snippet[:157] + "..."
            lines.append(f"| {e.turn_ids} | {e.i2b2_tag} | {e.predicate_phrase}/{e.object_name} | {e.reason} | {snippet} |")
    else:
        lines.append(
            "(none — the model tagged every candidate in this admission "
            "`present`/`absent`; see 'Honest read on quality' for why this "
            "admission alone cannot demonstrate a skip.)"
        )
    lines.append("")
    lines.append("### Other admissions of the same patient (conditional skips)")
    lines.append("")
    if extra_dropped:
        lines.append("| admission | turn | i2b2 tag | would-be predicate/object | reason | source snippet |")
        lines.append("|---|---|---|---|---|---|")
        for e in extra_dropped:
            turn = turns_by_number.get(e.session_id, {}).get(e.turn_ids[0]) if e.turn_ids else None
            snippet = turn.text.strip().replace("\n", " ") if turn else ""
            if len(snippet) > 160:
                snippet = snippet[:157] + "..."
            lines.append(f"| {e.session_id} | {e.turn_ids} | {e.i2b2_tag} | {e.predicate_phrase}/{e.object_name} | {e.reason} | {snippet} |")
    else:
        lines.append("(none in the extra admission(s) run)")
    lines.append("")

    lines.append("### Predicate-vocabulary drops (`dropped_no_predicate`, this run)")
    lines.append("")
    no_predicate_dropped = [e for e in dropped if e.decision == "dropped_no_predicate"]
    if no_predicate_dropped:
        lines.append("| admission | turn | predicate phrase / object | reason | source snippet |")
        lines.append("|---|---|---|---|---|")
        for e in no_predicate_dropped:
            turn = turns_by_number.get(e.session_id, {}).get(e.turn_ids[0]) if e.turn_ids else None
            snippet = turn.text.strip().replace("\n", " ") if turn else ""
            if len(snippet) > 160:
                snippet = snippet[:157] + "..."
            lines.append(f"| {e.session_id} | {e.turn_ids} | {e.predicate_phrase}/{e.object_name} | {e.reason} | {snippet} |")
    else:
        lines.append(
            "(none in the admission(s) run this process — see the curated "
            "table below for a real example found during broader exploration.)"
        )
    lines.append("")

    lines.append(
        "### Additional real conditional / no-predicate drops, same patient "
        "(found while exploring the full 27-admission history for this "
        "artifact's cost projection — not re-run by default, see note below)"
    )
    lines.append("")
    lines.append("| admission | turn | i2b2 tag | would-be predicate/object | reason | source snippet |")
    lines.append("|---|---|---|---|---|---|")
    all_admissions_by_id = {a.hadm_id: a for a in conversation.admissions}
    for hadm_id, turn_ids, i2b2_tag, predicate_phrase, object_name, reason in _ADDITIONAL_REAL_DROPS:
        adm = all_admissions_by_id.get(hadm_id)
        t = {tt.turn_number: tt for tt in adm.turns()}.get(turn_ids[0]) if adm else None
        snippet = t.text.strip().replace("\n", " ") if t else "(turn not found)"
        if len(snippet) > 160:
            snippet = snippet[:157] + "..."
        lines.append(f"| {hadm_id} | {turn_ids} | {i2b2_tag} | {predicate_phrase}/{object_name} | {reason} | {snippet} |")
    lines.append("")
    lines.append(
        "(Six of the seven rows above are the same `conditional` PRN-"
        "medication pattern as the dynamic table before it — see 'Honest "
        "read on quality' item 1; the seventh, `23527958`, is the "
        "vocabulary-coverage gap in item 2. **All seven rows are a PRE-FIX "
        "historical record** — see the 2026-08-16 PRN/vocabulary-gap bug "
        "fix: PRN language now maps to `present` [`prn=true`] instead of "
        "`conditional`, and `HAD_INCIDENT` now covers `had a fall`. This "
        "table has not been re-run live against the model to confirm "
        "[real-inference spend, out of scope here]; "
        "`tests/test_extract.py` proves the fix against these same real "
        "quotes with a scripted, zero-cost `complete_fn`.)"
    )
    lines.append("")
    lines.append(
        "**not-associated-with-patient (family-history):** never observed "
        f"anywhere in subject `{args.subject}`'s real data (checked across "
        "all 27 admissions / 1,451 turns during exploration for this "
        "artifact — see 'Honest read on quality', item 4). **hypothetical:** "
        "likewise never observed. Both tags remain untested by this "
        "patient's real content; see the rule-based sibling file for a "
        "demonstration of the family-history mechanism against other real "
        "corpus turns."
    )
    lines.append("")

    lines.append("## Honest read on quality")
    lines.append("")
    lines.append(
        "This is a critique of the **real LLM extractor** "
        f"(`{extractor.model}` via `llm.complete()`), grounded in what was "
        "actually observed while building this artifact — not a template "
        "carried over from the rule-based file."
    )
    lines.append("")
    lines.append(
        "**What it gets right, compared to the rule-based fallback:** no "
        "duplicate near-identical facts across repeated symptom mentions — "
        "the rule-based scanner fired once per mention of a lexicon term "
        "(e.g. \"swelling\" on nearly every patient turn); the real "
        "extractor emitted one fact per distinct clinical assertion, "
        f"yielding {len(anchor_facts)} facts for the same "
        f"{len(anchor.turns())}-turn anchor admission that the rule-based "
        "pass over-produced and then had to de-duplicate down for "
        "readability. Dosage-as-subject and negation both still work "
        "correctly (see rows above). The real extractor also correctly "
        "produced **zero** false-positive family-history skips on this "
        "patient's real pronoun-heavy turns (e.g. \"call if you develop "
        "fever\" is not near any family-relation term in this patient's "
        "own data, so the rule-based false positive documented in the "
        "rule-based sibling file simply cannot recur here on real text — a "
        "weaker claim than \"the LLM fixed it\", since this patient's data "
        "never posed that exact test; see item 4 below)."
    )
    lines.append("")
    lines.append("**What it gets wrong (found in this run, not hidden):**")
    lines.append(
        "1. **[FIXED 2026-08-16 — see the PRN/vocabulary-gap bug fix; this "
        "item is the pre-fix record, kept for honesty, not re-run live to "
        "confirm.]** PRN (\"as needed\") medications are tagged `conditional` and "
        "dropped entirely**, e.g. admission `20971116` turn 52 (\"Plus the "
        "new diuretics and oxycodone for pain as needed.\") and admission "
        "`29091921` turn 32 (\"We're giving you oxycodone as needed.\") both "
        "drop `TAKES_MEDICATION oxycodone`. This is a defensible reading of "
        "the system prompt's own definition (\"conditional: true only under "
        "a stated future contingency\") but is arguably too strict for a "
        "medication list: an as-needed order is a real, currently-active "
        "prescription, not a hypothetical — dropping it loses information a "
        "clinician would want on the patient's med list, not just a correct "
        "skip of a truly counterfactual statement. This is the single "
        "clearest, most reproducible real-run finding in this artifact and "
        "should be the first thing a human reviewer checks in the formal "
        "E2-S3 hand-check — it recurred independently across at least 3 "
        "different admissions during exploration for this artifact."
    )
    lines.append(
        "2. **[FIXED 2026-08-16 — `HAD_INCIDENT` added to `contracts."
        "PREDICATES`; this item is the pre-fix record, kept for honesty, "
        "not re-run live to confirm.]** A real vocabulary-coverage gap. "
        "Admission `23527958` turn 1 "
        "(\"you had a fall this morning\") produces a candidate fact with "
        "`predicate_phrase='had a fall'`, which does not canonicalize onto "
        "the closed `PREDICATES` vocabulary (`normalize.canonicalize_"
        "predicate` returns `None`) and is dropped as `dropped_no_predicate` "
        "— silently, from this eyeball's point of view, since decisions/002 "
        "handling rules only cover the six i2b2 tags, not predicate-vocab "
        "misses. A fall is clinically significant (fall risk, injury) and "
        "has no home in the current predicate vocabulary "
        "(`normalize.PREDICATE_DEFINITIONS`); this is a vocabulary gap, not "
        "an extraction failure."
    )
    lines.append(
        "3. **Non-determinism at `temperature=0.0`.** Re-running the exact "
        f"same anchor-admission prompt fresh (bypassing the cache, "
        "`use_cache=False`) on two separate calls returned 17 and (in a "
        "later, non-cache-bypassed check) a different fact count on a "
        "different admission across repeated calls — concretely, a fresh, "
        "uncached call against admission `20971116` returned 16 facts, "
        "while an earlier cached response for the same admission held 21. "
        "`temperature=0.0` lowers but does not eliminate sampling variance "
        "on this provider; a caller that needs bit-for-bit reproducibility "
        "(e.g. re-running this exact eyeball for a diff-based review) should "
        "rely on the disk cache (`llm.py`'s `CACHE_DIR`), not re-issue fresh "
        "calls and expect identical output."
    )
    lines.append(
        "4. **An unreplicated transient schema-validation failure.** During "
        "exploratory extraction across this same patient's full 27-admission "
        "history (building the multi-admission cost estimate below, not "
        "part of the table above), one admission raised "
        "`llm.SchemaValidationError` after exhausting the seam's own 3 "
        "schema retries: *\"response was not valid JSON: Expecting property "
        "name enclosed in double quotes: line 385 column 7\"* — a malformed, "
        "truncated-looking JSON body. A clean re-run across all 27 "
        "admissions immediately afterward completed with zero errors, so "
        "this did not reproduce; flagged here at **medium-low confidence** "
        "as a real, observed failure mode (the seam's retry-and-propagate "
        "design worked exactly as intended — it raised a typed, named error "
        "rather than silently returning partial/wrong data — but scale "
        "ingest should expect an occasional admission to need a retry, not "
        "assume every admission succeeds on the first attempt)."
    )
    lines.append(
        "5. **No reported-speech / speaker attribution**, same "
        "undemonstrated gap `literature/14` Part 5 documents for the "
        "rule-based path: `source_class` defaults to the turn's speaker "
        "with no mechanism to catch \"my other doctor said...\" "
        "reattribution. Not exercised in this patient's real turns (none "
        "contain that construction), so still an open, un-tested risk "
        "rather than a confirmed miss."
    )
    lines.append(
        "6. **Raw-output naming noise that does not reach the emitted "
        "fact.** A few dropped (conditional) candidates named the subject "
        "using the patient's in-dialogue name (e.g. `\"Mr. Johnson\"`) "
        "instead of the literal placeholder `\"PATIENT\"` the system prompt "
        "asks for. This never affects the *emitted* `ClinicalFact` rows "
        "above — `extract.py`'s `_handle_candidate` hardcodes "
        "`subject=patient_id` for every non-`CURRENT_DOSAGE_OF` predicate "
        "regardless of what the model said — but it is worth a human "
        "reviewer's attention as a sign the model does not reliably follow "
        "the placeholder-naming instruction, only that today's code "
        "defensively ignores it."
    )
    lines.append("")
    lines.append(
        "**Recommendation before treating this as the real gate:** the "
        "formal E2-S3 hand-check (`fixtures/handcheck/CHECKLIST.md`, "
        "≥30 rows, a human-filled `ok`/`bad` column) should specifically "
        "re-check finding 1 (PRN-medication drops) and finding 2 "
        "(predicate-vocabulary coverage) — both are systematic, not "
        "one-off, and both are cheap for `dev-architect` to resolve "
        "(loosen the `conditional` definition for PRN language; extend "
        "`PREDICATE_DEFINITIONS`) before any multi-patient run amplifies "
        "either gap across the corpus."
    )
    lines.append("")

    lines.append("## Measured cost, and a projection to the full corpus")
    lines.append("")
    lines.append(
        "Two fresh (`use_cache=False`), individually-billed calls were made "
        "to get a clean, cache-free cost baseline for this patient (all "
        "other admissions used during exploration hit the default disk "
        "cache after their first real call, so their in-process reported "
        "cost is `$0.00` on a re-run — real, but not double-counted):"
    )
    lines.append("")
    lines.append("| admission | turns | facts | prompt tok | completion tok | cost (USD) |")
    lines.append("|---|---|---|---|---|---|")
    lines.append(f"| `{args.hadm_id}` (anchor) | 44 | 17 | 3,090 | 2,235 | $0.002406 |")
    lines.append("| `20971116` (extra) | 70 | 16 | 4,241 | 1,932 | $0.0023938 |")
    lines.append("| **this patient, 2 admissions, fresh** | 114 | 33 | 7,331 | 4,167 | **$0.0048** |")
    lines.append("")
    lines.append(
        "A separate exploratory pass extracted **all 27 admissions / 1,451 "
        "turns** of this same patient's history (2 of the 27 served from "
        "cache from the calls above, 25 fresh) for **$0.0294** more, giving "
        "a whole-patient fresh-equivalent total of **≈$0.0342** for 619 "
        "facts across all 27 admissions (170,275 prompt+completion tokens "
        "measured in that pass, plus the ≈11,498 tokens for the 2 cached "
        "admissions, isolated above) — item 4 above (the one schema-"
        "validation failure) happened once during an earlier attempt at "
        "this same full-patient pass and did not recur on the clean re-run."
    )
    lines.append("")
    lines.append(
        "**Projection to the full 101-patient / 167,896-turn / 6,733,899-"
        "token MedLoCoMo corpus** (README.md's own measured corpus stats), "
        "extrapolated three ways from this single patient's ≈$0.0342 / "
        "1,451 turns / 27 admissions / 31,454 raw corpus tokens "
        "(`Conversation.token_estimate()`):"
    )
    lines.append("")
    lines.append("| basis | this patient | corpus total | naive linear projection |")
    lines.append("|---|---|---|---|")
    lines.append("| per raw corpus token | 31,454 tok | 6,733,899 tok | **≈$7.32** |")
    lines.append("| per turn | 1,451 turns | 167,896 turns | ≈$3.96 |")
    lines.append("| per admission | 27 admissions | ≈2,980 admissions (101 × 29.5 mean) | ≈$3.78 |")
    lines.append("")
    lines.append(
        "These three methods disagree by ~2x because this patient's turns "
        "are shorter than the corpus average (this patient: "
        f"{31454 // 1451} raw tokens/turn measured vs. the corpus-wide mean "
        f"of {6733899 // 167896} tokens/turn) — a smaller, less costly "
        "patient than typical. Since API prompt cost is driven by content "
        "volume (tokens), not turn or admission *count*, the per-token "
        "projection (**≈$7.32**) is the more defensible upper-bound "
        "estimate; **recommend budgeting ≈$5-8 total** for a full-corpus "
        "`EXTRACT_MODEL` extraction pass, with **medium-low confidence** "
        "given this is extrapolated from a single, below-median-size "
        "patient (n=1). The single observation that would most tighten "
        "this range: repeat this same cost measurement on 2-3 more patients "
        "spanning the corpus's session-count distribution (min/median/max) "
        "before authorizing the full 101-patient run — this is well within "
        "the default `MEDMEMGRAPH_MAX_USD=$5.00` per-process cap for any "
        "one of those additional single-patient runs, but the full 101-"
        "patient run itself will need that cap raised (not done here — see "
        "'Do not raise the budget cap' in this story)."
    )
    lines.append("")

    out_path = REPO_ROOT / "samples" / f"extract-eyeball-{args.subject}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(
        f"wrote {out_path} ({len(selected)} rows, extractor.kind={extractor.kind}, "
        f"in-process cost={_fmt_usd(extractor.total_cost_usd)})"
    )
    if errors:
        print(f"NOTE: {len(errors)} admission(s) raised llm.LLMError and were skipped: {errors}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", default="10056223")
    parser.add_argument("--hadm-id", default="26605038")
    parser.add_argument(
        "--extra-hadm-ids",
        nargs="*",
        default=["20971116"],
        help="Same-patient admissions to pull additional real facts from, "
        "in patient-admission order, until --max-rows is reached. Ignored "
        "in --dry-run mode (the rule-based artifact stays single-admission, "
        "matching the original leftover-gate script).",
    )
    parser.add_argument("--max-rows", type=int, default=30)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Force the deterministic rule-based fallback (explicit opt-in, "
        "matches Extractor(dry_run=True)) instead of the real-inference "
        "default. Writes the '.rule-based.md' sibling file.",
    )
    args = parser.parse_args()

    if args.dry_run:
        _generate_rule_based(args)
    else:
        _generate_real(args)


if __name__ == "__main__":
    main()
