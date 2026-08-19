#!/usr/bin/env bash
# scripts/run_eval.sh — the full baseline ladder over N patients, then the
# pooled report.
#
# `eval/harness.py` is per-patient by design, so a corpus-scale evaluation is a
# loop over (patient x system). This script is that loop, plus the two things
# that make its numbers trustworthy:
#
#   * EVERY system sees the SAME items. `--stratify-per-type` + a fixed
#     `--seed` guarantee it. This is not cosmetic: `metrics.mcnemar_test` and
#     `report._align_by_qa_id` are both PAIRED per qa_id, so systems drawing
#     different samples silently degrades to comparing different benchmarks.
#   * An absolute LLM cache dir. `llm.CACHE_DIR` defaults to a RELATIVE
#     `data/llm_cache`, so running from another directory silently starts a
#     fresh cache AND a fresh ledger — resetting the spend cap without saying so.
#
# Cost note: `fullctx` puts the entire patient history (~66.7K tokens) in the
# prompt for every question and is ~85% of the bill. Do NOT shrink its context
# window to save money — handicapping the strongest baseline is exactly the
# dishonesty this project's own eval discipline forbids. Cut items per type, or
# cut patients, which keeps the pairing intact.
set -euo pipefail

PER_TYPE="${PER_TYPE:-6}"
SEED="${SEED:-0}"
N_PATIENTS="${N_PATIENTS:-20}"
RESULTS_DIR="${RESULTS_DIR:-results}"
# The retrieval ladder. reader_direct/reader_con are deliberately NOT here: they
# read `mock_retrieve` evidence so that READING STRATEGY is the only variable
# between them, which makes their accuracy meaningless next to systems that do
# real retrieval. Run them as their own A/B:
#   SYSTEMS="reader_direct reader_con" RESULTS_DIR=results/reading-ablation ./scripts/run_eval.sh
SYSTEMS="${SYSTEMS:-nomem fullctx dense lexical medmemgraph}"

export MEDMEMGRAPH_LLM_CACHE_DIR="${MEDMEMGRAPH_LLM_CACHE_DIR:-$PWD/data/llm_cache}"

patients=$(uv run python -c "
from medmemgraph.pipeline.loader import list_patients
print(' '.join(list_patients()[:$N_PATIENTS]))")

echo "systems  : $SYSTEMS"
echo "patients : $(echo "$patients" | wc -w)"
echo "sampling : --stratify-per-type $PER_TYPE --seed $SEED"
echo "cache    : $MEDMEMGRAPH_LLM_CACHE_DIR"
echo

for patient in $patients; do
  for system in $SYSTEMS; do
    out="$RESULTS_DIR/${patient}__${system}.json"
    if [ -f "$out" ]; then
      echo "  skip $patient/$system (already have $out)"
      continue
    fi
    echo "  run  $patient/$system"
    uv run python -m medmemgraph.eval.harness \
      --patient "$patient" \
      --system "$system" \
      --stratify-per-type "$PER_TYPE" \
      --seed "$SEED" \
      --results-dir "$RESULTS_DIR" \
      >/dev/null || echo "    FAILED $patient/$system" >&2
  done
done

echo
echo "=== pooled report ==="
uv run python -m medmemgraph.eval.report --results-dir "$RESULTS_DIR" --markdown
