# Submission checklist

Deadline: **2026-08-20 23:59 PT** (hard stop). Internal target: **file by
midday PT on 2026-08-20**, so the form is not a 23:00 surprise. Feature
freeze: end of day 2026-08-18; after that, only fixes the numbers demand,
the demo, and this checklist.

This is a checklist, not prose. Check every box against the real repo
state on the commit you are about to submit, not against memory of an
earlier state.

## Eligibility gates

- [ ] Repo is public: `github.com/pbiyyani09/Hack_Hydra`.
- [ ] Repo has an OSS license at its root (`LICENSE`, MIT) that was not
      changed silently during this sprint.
- [ ] Demo video is ≤ 3:00. Content past 3:00 may not be reviewed, so cut
      before that mark; do not rely on judges stopping early.
- [ ] Video is hosted somewhere judges can watch without asking for
      access. Unlisted YouTube is acceptable, but confirm no "request
      access" gate is in front of it. Test the link from a logged-out /
      incognito browser before filing.
- [ ] Every commit that matters for judging is dated **on or after
      2026-08-12**. (The two commits currently on `main` are dated
      2026-08-13; anything committed from here forward trivially clears
      this bar. Just don't backdate or rebase across it.)

## Repo hygiene (re-run on the exact commit you submit, not an earlier one)

- [ ] `git ls-files | rg -n "formed_packet|combined_conversation|MedLoCoMo/"`
      returns nothing. Zero MedLoCoMo blobs, zero `formed_packet.json`,
      anywhere in the tracked tree.
- [ ] `git ls-files | rg -n "\.env$|auth-token$"` returns nothing. No
      secrets or auth tokens tracked.
- [ ] `rg -n "beat full-context|outperforms full context" README.md
      demo/VIDEO_SCRIPT.md` returns nothing. No accuracy-win claim
      anywhere stranger-facing.
- [ ] `.gitignore` still covers `data/`, `.env`, `auth-token`, and the
      usual Python build artifacts (`__pycache__/`, `.pytest_cache/`,
      `*.egg-info/`, `dist/`, `build/`). Diff it against the last known-good
      version if anyone touched it during the sprint.
- [ ] Image pin is stated explicitly in `README.md`:
      `ghcr.io/hydra-db/hydradb:0.1.1` (not `latest`, not `0.1.0`).

## Freeze-bar tests, re-run on the submission commit

```bash
uv sync
uv run pytest -q                    # full suite, needs HydraDB up (bash scripts/run_hydradb.sh) for @pytest.mark.live tests
uv run pytest -q -m "not live"      # offline subset only, no HydraDB required
```

- [ ] Both commands were actually re-run on the commit about to be
      submitted, not on an earlier one. A green run from yesterday does
      not certify today's diff.
- [ ] If any test is skipped or `xfail`, that is a known, explained state,
      not a silent regression; check the skip reason.

## Submission form fields

Confirm the actual Hack Hydra form before filing; the fields below are the
ones known at the time this checklist was written. Fill in real values, not
placeholders, before submitting.

- [ ] **Project name:** MedMemGraph
- [ ] **Description:** one or two sentences, pulled from `README.md`'s
      opening paragraph; do not write a new one that drifts from it.
- [ ] **Problem:** longitudinal clinical memory across many admissions;
      long-context and vector-only memory layers lose chronology, miss
      overwrites, and confabulate on questions with no answer in the
      record. Cite the same framing used in `README.md` / `ARCHITECTURE.md`
      §1, do not invent a different pitch for the form.
- [ ] **Deployed link:** this project is a local, self-hosted CLI demo
      against a self-hosted HydraDB container, not a hosted web app.
      Confirm with the actual form whether "deployed link" accepts N/A /
      "local demo, see video" for a project shaped this way, or whether a
      link is mandatory; do not fabricate a URL to satisfy the field.
- [ ] **How HydraDB is used, and what the project would lose without
      it:** the graph holds versioned `:Claim` nodes with `SUPERSEDES` /
      `CONTRADICTS` edges, and answers cross-admission questions with
      `algo.MSpaths` (bounded, weighted, multi-source/multi-target whole-path
      enumeration) and answers "was this ever mentioned" with a labelled
      existence check. Without HydraDB: no bounded whole-path retrieval
      primitive (a relational recursive CTE does not return ranked paths;
      a vector store has no path concept at all), and no cheap way to get
      a genuine "never mentioned" boolean instead of a low-similarity
      guess. See `README.md` § Why a graph, not a vector store for the
      full argument.
- [ ] **Tech stack:** HydraDB OSS `0.1.1` (Rust, AGPL-3.0, self-hosted),
      Python 3.13, `neo4j==5.28.2` (patched client), `uv`, NumPy
      brute-force cosine, `bm25s`, `sentence-transformers` (Qwen3-Embedding-0.6B),
      OpenAI `gpt-4.1-mini` (Chain-of-Note reading) and Google
      `gemini-3.5-flash-lite` (extraction, LLM judge, entity-match
      adjudication) — deliberately different model families so the judge
      never grades its own answerer. Deterministic offline fallbacks exist
      behind an explicit `--dry-run`, never as a silent default.
- [ ] **Team:** fill in actual names/roles before filing; do not leave
      this checklist's placeholder text in the form.
- [ ] **GitHub link:** `https://github.com/pbiyyani09/Hack_Hydra`
- [ ] **Demo video link:** the unlisted YouTube (or equivalent) URL from
      the eligibility gates above, confirmed reachable logged-out.

## File before 23:59 PT, aim for midday

- [ ] Form submitted.
- [ ] Confirmation (email / on-screen receipt) saved somewhere the team
      can find it if there's a dispute later.
