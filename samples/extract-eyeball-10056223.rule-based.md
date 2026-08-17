# One-patient extract eyeball — subject 10056223 (rule-based fallback)

Leftover-gate artifact (`PHASES.md`: "a checked-in sample of ~30 `ClinicalFact` rows next to their source turns"). **Not** the formal E2-S3 hand-check gate (`scale_gate.py` / `PASSED`) — see `scripts/generate_extract_eyeball.py` docstring. This is the **rule-based fallback** sibling of `extract-eyeball-10056223.md`, kept for direct comparison — see that file for the real-inference run.

- Subject: `10056223` (`list_patients()`-discoverable, real corpus)
- Admission (`hadm_id`): `26605038`
- Turns in admission: 44
- Facts emitted (full run, before dedup for readability): 56
- Rows shown below (deduped by predicate+object+polarity): 30
- Extractor path used: **rule-based** (`Extractor(dry_run=True)`, explicit opt-in)

> **This run used the deterministic rule-based fallback, not the LLM path** (`Extractor(dry_run=True)`, requested explicitly via `--dry-run` — real inference is this module's default; see `extract-eyeball-10056223.md` for that run). The rule-based fallback is a small NegEx/ConText-shaped trigger-window matcher over a fixed clinical-term lexicon (`literature/14` R-IE-01/R-IE-02 design shape) — it exists so this module is runnable and testable without a key, not as a substitute for the LLM path's quality. Treat every row below as a demonstration of the pipeline *mechanics* (handling rules, canonicalize, resolve_time, provenance) and of a classical rule-based system's known ceiling.

| # | fact_id | predicate | subject → object | polarity | conf | source | turn | turn_time | valid_from | source snippet |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `pending-2ef4071f593406794183` | REPORTS_SYMPTOM | 10056223 → swelling | asserted | 0.55 | doctor | [1] | 2122-08-12 23:42:00 | 2122-08-12T23:42:00 (= turn time) | Hello, Mr. Johnson. I'm Dr. Patel, one of the hospitalists. I understand you came in because of swelling in your legs and belly? |
| 2 | `pending-86d5467b3d4317371954` | REPORTS_SYMPTOM | 10056223 → cough | asserted | 0.55 | doctor | [3] | 2122-08-12 23:48:00 | 2122-08-12T23:48:00 (= turn time) | I see. You mentioned some shortness of breath and a dry cough with activity too? |
| 3 | `pending-48bba9ce80e814a961bb` | REPORTS_SYMPTOM | 10056223 → dry cough | asserted | 0.55 | doctor | [3] | 2122-08-12 23:48:00 | 2122-08-12T23:48:00 (= turn time) | I see. You mentioned some shortness of breath and a dry cough with activity too? |
| 4 | `pending-31314d42d99c3308f98c` | REPORTS_SYMPTOM | 10056223 → fever | asserted | 0.55 | patient | [4] | 2122-08-12 23:50:00 | 2122-08-12T23:50:00 (= turn time) | Yeah, a little. Not bad, but I notice it when I walk. I also thought I had a fever at home, but I don’t feel hot now. |
| 5 | `pending-eda6878c4867095797d7` | REPORTS_SYMPTOM | 10056223 → chest pain | asserted | 0.55 | doctor | [5] | 2122-08-12 23:53:00 | 2122-08-12T23:53:00 (= turn time) | Okay. Any chest pain recently? |
| 6 | `pending-1c2dbede5c59b11d5f6b` | REPORTS_SYMPTOM | 10056223 → shortness of breath | asserted | 0.52 | patient | [6] | 2122-08-12 23:55:00 | 2122-08-11T23:55:00 | I had some yesterday—right in the middle of my chest, with pressure down my right arm and some shortness of breath. But it’s gone now. |
| 7 | `pending-a05b73c5cbb795c41ede` | HAD_PROCEDURE | 10056223 → ekg | asserted | 0.55 | doctor | [7] | 2122-08-12 23:58:00 | 2122-08-12T23:58:00 (= turn time) | Thanks for letting me know. We checked your EKG, and it looks stable, no new changes. Your vitals are normal now, and your lungs sound cl... |
| 8 | `pending-09474cd75ac114ea96ab` | HAD_PROCEDURE | 10056223 → transplant | asserted | 0.55 | patient | [8] | 2122-08-13 00:01:00 | 2122-08-13T00:01:00 (= turn time) | Good. I’ve been on the transplant list for a while, and I just feel frustrated. The swelling isn’t going down like I hoped, even with the... |
| 9 | `pending-7d83f4c311cf6af79141` | TAKES_MEDICATION | 10056223 → furosemide | asserted | 0.44 | patient | [11] | 2122-08-13 07:33:00 | 2122-07-13T12:09:00 | I’ve been on 60 mg of furosemide since my last visit. Could that be too much? |
| 10 | `pending-245c0caaa8154671d86d` | CURRENT_DOSAGE_OF | furosemide → 60mg | asserted | 0.60 | patient | [11] | 2122-08-13 07:33:00 | 2122-08-13T07:33:00 (= turn time) | I’ve been on 60 mg of furosemide since my last visit. Could that be too much? |
| 11 | `pending-3db25864aebc03f54406` | TAKES_MEDICATION | 10056223 → spironolactone | asserted | 0.55 | doctor | [12] | 2122-08-13 07:36:00 | 2122-08-13T07:36:00 (= turn time) | It’s appropriate, but it can lower potassium. We’re adding more spironolactone—from 200 to 300 mg daily. It helps keep potassium up and r... |
| 12 | `pending-088d8446fb43e9812809` | CURRENT_DOSAGE_OF | spironolactone → 300mg | asserted | 0.60 | doctor | [12] | 2122-08-13 07:36:00 | 2122-08-13T07:36:00 (= turn time) | It’s appropriate, but it can lower potassium. We’re adding more spironolactone—from 200 to 300 mg daily. It helps keep potassium up and r... |
| 13 | `pending-570498d54d59ca61c57f` | HAS_CONDITION | 10056223 → anxiety | asserted | 0.28 | doctor | [16] | 2122-08-13 07:47:00 | 2122-08-13T07:47:00 (= turn time) | We reviewed it. No EKG changes, no enzyme rise, and your echo recently was okay. It could be musculoskeletal or related to anxiety. But w... |
| 14 | `pending-11e127efd44e5a09b4d3` | HAD_PROCEDURE | 10056223 → ekg | negated | 0.55 | doctor | [16] | 2122-08-13 07:47:00 | 2122-08-13T07:47:00 (= turn time) | We reviewed it. No EKG changes, no enzyme rise, and your echo recently was okay. It could be musculoskeletal or related to anxiety. But w... |
| 15 | `pending-4589e2d435feb1b36dc0` | HAD_PROCEDURE | 10056223 → echo | negated | 0.55 | doctor | [16] | 2122-08-13 07:47:00 | 2122-08-13T07:47:00 (= turn time) | We reviewed it. No EKG changes, no enzyme rise, and your echo recently was okay. It could be musculoskeletal or related to anxiety. But w... |
| 16 | `pending-37897afc44a1826e6764` | TAKES_MEDICATION | 10056223 → escitalopram | asserted | 0.55 | patient | [17] | 2122-08-13 07:50:00 | 2122-08-13T07:50:00 (= turn time) | I’ve been anxious lately. Still waiting on the transplant. And I’ve been taking escitalopram—helps a little. |
| 17 | `pending-3441ba34ef5dc005d0f1` | HAS_CONDITION | 10056223 → depression | asserted | 0.55 | doctor | [18] | 2122-08-13 07:53:00 | 2122-08-13T07:53:00 (= turn time) | Good. We’ll continue that. Depression and anxiety are common with chronic illness, and treating them is part of your care. |
| 18 | `pending-4d3d1ccf30bcd083a116` | HAS_CONDITION | 10056223 → ascites | negated | 0.55 | doctor | [19] | 2122-08-13 10:15:00 | 2122-08-13T10:15:00 (= turn time) | We got your ultrasound results. Your liver still shows cirrhosis and some ascites, but no new tumors. The blood vessels are open and work... |
| 19 | `pending-1c5ecf44cdeae577ea69` | HAS_CONDITION | 10056223 → cirrhosis | asserted | 0.55 | doctor | [19] | 2122-08-13 10:15:00 | 2122-08-13T10:15:00 (= turn time) | We got your ultrasound results. Your liver still shows cirrhosis and some ascites, but no new tumors. The blood vessels are open and work... |
| 20 | `pending-cb8a972ef349b2c9d785` | HAD_PROCEDURE | 10056223 → ultrasound | asserted | 0.55 | doctor | [19] | 2122-08-13 10:15:00 | 2122-08-13T10:15:00 (= turn time) | We got your ultrasound results. Your liver still shows cirrhosis and some ascites, but no new tumors. The blood vessels are open and work... |
| 21 | `pending-e0f2bdf92247f1a4a521` | HAD_PROCEDURE | 10056223 → x-ray | asserted | 0.55 | patient | [22] | 2122-08-13 10:24:00 | 2122-08-13T10:24:00 (= turn time) | And my chest X-ray? I had a dry cough. |
| 22 | `pending-ef953a346fe4a42555d8` | HAS_CONDITION | 10056223 → pneumonia | negated | 0.55 | doctor | [23] | 2122-08-13 10:27:00 | 2122-08-13T10:27:00 (= turn time) | No pneumonia or fluid in the lungs. The heart looks normal in size. Your shortness of breath is likely from the fluid retention, not hear... |
| 23 | `pending-97b259381f6f9c696ee4` | HAS_CONDITION | 10056223 → heart failure | asserted | 0.28 | doctor | [23] | 2122-08-13 10:27:00 | 2122-08-13T10:27:00 (= turn time) | No pneumonia or fluid in the lungs. The heart looks normal in size. Your shortness of breath is likely from the fluid retention, not hear... |
| 24 | `pending-af6498bcbc28b942ce31` | TAKES_MEDICATION | 10056223 → mesalamine | negated | 0.55 | patient | [26] | 2122-08-13 10:36:00 | 2122-08-13T10:36:00 (= turn time) | And my ulcerative colitis? I’m not actually taking the mesalamine. |
| 25 | `pending-67138b6ae29d9a3c9309` | HAS_CONDITION | 10056223 → ulcerative colitis | asserted | 0.49 | patient | [26] | 2122-08-13 10:36:00 | 2122-08-13T10:36:00 (= turn time) | And my ulcerative colitis? I’m not actually taking the mesalamine. |
| 26 | `pending-0dd9476f24a286f50413` | TAKES_MEDICATION | 10056223 → lactulose | negated | 0.55 | doctor | [27] | 2122-08-13 10:39:00 | 2122-08-13T10:39:00 (= turn time) | You’re having three bowel movements a day on lactulose, no blood or diarrhea. If you’re not having flares, we can hold off on mesalamine ... |
| 27 | `pending-1bfeb7304cc59d701f72` | TAKES_MEDICATION | 10056223 → lactulose | asserted | 0.55 | doctor | [35] | 2122-08-13 14:18:00 | 2122-08-13T14:18:00 (= turn time) | Same as before, but spironolactone is now 300 mg daily. Furosemide stays at 60 mg. Keep taking lactulose, rifaximin, and the others as pr... |
| 28 | `pending-ce37602efe553d361180` | TAKES_MEDICATION | 10056223 → rifaximin | asserted | 0.55 | doctor | [35] | 2122-08-13 14:18:00 | 2122-08-13T14:18:00 (= turn time) | Same as before, but spironolactone is now 300 mg daily. Furosemide stays at 60 mg. Keep taking lactulose, rifaximin, and the others as pr... |
| 29 | `pending-68bf9160abc686168f34` | TAKES_MEDICATION | 10056223 → ciprofloxacin | asserted | 0.55 | patient | [36] | 2122-08-13 14:21:00 | 2122-08-13T14:21:00 (= turn time) | And the ciprofloxacin? I’ve been on it for infection prevention. |
| 30 | `pending-50ff70b418b9ac5b493a` | HAS_CONDITION | 10056223 → ascites | asserted | 0.55 | doctor | [37] | 2122-08-13 14:24:00 | 2122-08-13T14:24:00 (= turn time) | Yes, continue that. It helps prevent infections like SBP, especially with ascites. |

## Facts deliberately NOT emitted

Per decisions/002's handling rules (kept binary on purpose): `conditional`/`hypothetical` -> skip beats mis-extracting; `not-associated-with-patient` -> never attach to the patient.

### Conditional / hypothetical (from this admission, `hadm_id 26605038`)

| turn | i2b2 tag | would-be predicate/object | reason | source snippet |
|---|---|---|---|---|
| [27] | conditional | TAKES_MEDICATION/mesalamine | conditional/hypothetical (rule-based: 'if' in window) | You’re having three bowel movements a day on lactulose, no blood or diarrhea. If you’re not having flares, we can hold off on mesalamine for now. |
| [27] | conditional | REPORTS_SYMPTOM/diarrhea | conditional/hypothetical (rule-based: 'if' in window) | You’re having three bowel movements a day on lactulose, no blood or diarrhea. If you’re not having flares, we can hold off on mesalamine for now. |
| [40] | conditional | REPORTS_SYMPTOM/abdominal pain | conditional/hypothetical (rule-based: 'if' in window) | Keep your hepatology follow-up in four days. Watch for increased swelling, confusion, fever, or abdominal pain. Call if any of those happen. |

### not-associated-with-patient (family-history) — supplementary, cross-patient real quotes

Subject `10056223`'s own admissions do not happen to contain a family-history-attributed condition (confirmed: see the real-run artifact), so the mechanism is demonstrated below against three real, short turns from other admissions in the same allowlisted corpus (verbatim sentences, not vendored files — `loader.load_conversation` is the only file this project opens; these three lines were read the same way during survey work and are quoted here, not files). Rule-based only — the real (LLM) artifact never opens another patient's file.

| patient | turn | speaker | would-be object | verdict | source snippet |
|---|---|---|---|---|---|
| 12484308 | 44 | Patient | cirrhosis | skipped (object(s): cirrhosis) | My uncle died from cirrhosis. I don’t want that to be me. |
| 12251785 | 20 | Patient | vomiting, nausea | skipped (object(s): vomiting, nausea) | My sister had nausea and vomiting too—maybe it’s going around. |
| 11281568 | 42 | Doctor | fever | skipped (object(s): fever) | You’re welcome. We’ll make sure your sister and care team get a full update. Call if you develop fever or trouble breathing. |

The third row (`11281568`, doctor turn) is a genuine **false positive** worth flagging, not a clean catch — see 'Honest read on quality' below.

## Honest read on quality

This run used the rule-based fallback (`--dry-run`), so this is a critique of that fallback, not of the LLM extractor — read it alongside `literature/14`'s own finding that classical rule-based negation/assertion systems are solid on plain negation but weak on exactly this kind of scoping nuance (R-IE-03/R-IE-04), and see `extract-eyeball-10056223.md` for the real-inference run's own honest critique.

**What it gets right:** explicit negation with the trigger word close to the term ("I'm not actually taking the mesalamine" -> TAKES_MEDICATION mesalamine, negated); dosage attached to the medication as subject, not the patient (`CURRENT_DOSAGE_OF` furosemide -> 60mg); relative-time resolution for "since my last visit" against the *prior admission's* end date, not the current turn's own date (see `normalize.resolve_time` and `tests/test_normalize.py`); two of the three family-history rows above (uncle/cirrhosis, sister/nausea+vomiting) correctly refuse to attach a family member's condition to the patient.

**What it gets wrong (found in this run, not hidden):**
1. **A false-positive family-skip.** Row 3 above (`11281568` turn 42) drops "fever" as not-associated-with-patient purely because "sister" appears earlier in the same sentence — but the sentence reads "Call if *you* develop fever", about the patient. The rule-based window has no real coreference resolution; it treats co-occurrence within ~70 characters as attribution. This is exactly the failure mode `literature/14`'s brief and decisions/002 both flag as highest-risk — the fallback demonstrates it, not just cites it.
2. **A scope-bleed false negation**, visible in the full (undeduped) run for this admission: turn 27 ("You're having three bowel movements a day on lactulose, no blood or diarrhea...") emits `TAKES_MEDICATION lactulose` as **negated**, because the negation trigger "no" sits just past the word boundary from "lactulose" in the scan window — it actually negates "blood"/"diarrhea", not the medication. This is a NegEx/ConText-class scope-detection error (the exact ceiling `literature/14` R-IE-03 documents for off-the-shelf rule sets: solid overall negation F-score, but no real clause-boundary awareness).
3. **Duplicate near-identical facts** across repeated mentions of the same symptom in different turns ("swelling" fires on nearly every patient turn in this admission) — this table dedupes them for readability, but the underlying extractor emits one fact per mention rather than recognizing a repeated assertion of the same still-current state. Entity resolution / same-claim merging is explicitly out of this story's scope (E3-S1).
4. **No reported-speech / speaker attribution.** `literature/14` Part 5 documents this as a genuine, searched-for-and-not-found literature gap; the extractor here (both the LLM prompt's stated convention and the rule-based path) defaults `source_class` to the speaker of the turn and has no mechanism to catch "my other doctor said..." reattribution. Not exercised in this admission's real turns, but a known, undemonstrated gap.
