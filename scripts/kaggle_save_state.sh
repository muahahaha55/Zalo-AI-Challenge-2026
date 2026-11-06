#!/usr/bin/env bash
# Last cell of a Kaggle session: keep only what the next session needs, then "Save Version" (Save & Run All is not needed).
# Kept in the notebook output: state/processed (ASR transcripts, rebuilt SFT data), checkpoints/, outputs/ (+ .done_* markers, logs).
# Removed: the lifemate/ working copy (code + the symlink to the read-only raw data); kaggle_setup.sh rebuilds it from lifemate-code.
# Next session: attach THIS notebook's output as an extra input; kaggle_setup.sh restores state, checkpoints and stage markers.
set -euo pipefail
W=${KROOT:-/kaggle}/working
[ -d "$W/lifemate/data/processed" ] || { echo "nothing to save: $W/lifemate/data/processed missing"; exit 1; }
rm -rf "$W/state"
mkdir -p "$W/state"
cp -r "$W/lifemate/data/processed" "$W/state/processed"
rm -f "$W/state/processed/dataset-metadata.json"
rm -rf "$W/lifemate"
du -sh "$W"/state "$W"/checkpoints "$W"/outputs 2>/dev/null
echo "saved. Now 'Save Version' (private notebook). Output must stay < 20 GB."
