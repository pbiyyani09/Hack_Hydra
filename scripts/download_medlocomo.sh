#!/usr/bin/env bash
# Fetch MedLoCoMo. NEVER vendor it into this repo: the release embeds
# MIMIC-IV-Note-derived text under a PhysioNet DUA. See decisions/001.
set -euo pipefail
DEST="${MEDLOCOMO_ROOT:-data/medlocomo}"
if [ -d "$DEST/MedLoCoMo" ]; then echo "already present: $DEST"; exit 0; fi
mkdir -p "$(dirname "$DEST")"
git clone --depth 1 https://github.com/leozzy13/MedLoCoMo "$DEST"
echo "MedLoCoMo at $DEST/MedLoCoMo ($(ls "$DEST/MedLoCoMo" | wc -l) patient dirs)"
echo "Ingestion reads ONLY combined_conversation.json; eval reads benchmark_qa.json."
