#!/usr/bin/env bash
# Prepare PRIVATE Kaggle datasets from the local machine (Git Bash). It only prepares folders and
# dataset-metadata.json; it never uploads. Upload yourself with the printed commands (docs/KAGGLE.md).
#   KAGGLE_USER=<your kaggle username> bash scripts/kaggle_pack.sh code|processed|raw|models
set -euo pipefail
cd "$(dirname "$0")/.."
KAGGLE_USER=${KAGGLE_USER:-}   # optional: only needed for the CLI upload path; the web UI needs no metadata
STAGE=kaggle_upload            # git-ignored

meta() {  # meta <dir> <slug> <title>
  [ -n "$KAGGLE_USER" ] || return 0
  printf '{\n  "title": "%s",\n  "id": "%s/%s",\n  "licenses": [{"name": "other"}]\n}\n' "$3" "$KAGGLE_USER" "$2" > "$1/dataset-metadata.json"
}
hint() {  # hint <dir> [extra flags]
  echo "Prepared: $1"
  if [ -z "$KAGGLE_USER" ]; then
    echo "Web upload: kaggle.com > Datasets > New Dataset > drag this folder (or its zip) in, set visibility to Private."
    echo "(CLI upload instead: rerun with KAGGLE_USER=<name>, which also writes dataset-metadata.json.)"
    return 0
  fi
  echo "Upload (creates a PRIVATE dataset; verify it says Private on its Kaggle page afterwards):"
  echo "  kaggle datasets create -p \"$1\" ${2:-}"
  echo "Update later:"
  echo "  kaggle datasets version -p \"$1\" -m \"<message>\" ${2:-}"
}

case "${1:-}" in
  code)       # source only (tracked + new, honours .gitignore): no data, weights, secrets
    rm -rf "$STAGE/lifemate-code"; mkdir -p "$STAGE/lifemate-code"
    git ls-files -z --cached --others --exclude-standard | xargs -0 -I{} cp --parents {} "$STAGE/lifemate-code/"
    meta "$STAGE/lifemate-code" lifemate-code "lifemate-code"
    .venv/Scripts/python.exe -I -c "import shutil; shutil.make_archive('$STAGE/lifemate-code', 'zip', '$STAGE/lifemate-code')" 2>/dev/null       || python -I -c "import shutil; shutil.make_archive('$STAGE/lifemate-code', 'zip', '$STAGE/lifemate-code')"
    echo "Zip for the web UI: $STAGE/lifemate-code.zip"
    hint "$STAGE/lifemate-code" ;;
  processed)  # SFT jsonl, dev ids, ASR transcripts, caches: small, re-upload as often as needed
    test -f data/processed/dev_ids.txt || { echo "data/processed/dev_ids.txt missing: run scripts/build_sft.py first"; exit 1; }
    meta data/processed lifemate-processed "lifemate-processed"
    hint data/processed ;;
  raw)        # competition data in place (no copy). --dir-mode zip is required or sub-folders are skipped
    test -d "data/raw/train+public_test" || { echo "data/raw/train+public_test missing"; exit 1; }
    meta "data/raw/train+public_test" lifemate-raw "lifemate-raw"
    hint "data/raw/train+public_test" "--dir-mode zip" ;;
  models)     # weights: Qwen3-4B (safetensors) + PhoWhisper-large converted bin -> safetensors; see scripts/prepare_models.py
    if [ ! -f "$STAGE/lifemate-models/manifest.json" ]; then
      .venv/Scripts/python.exe -I -X utf8 scripts/prepare_models.py 2>/dev/null || python -I -X utf8 scripts/prepare_models.py
    fi
    meta "$STAGE/lifemate-models" lifemate-models "lifemate-models"
    echo "Checksums and parameter counts are in $STAGE/lifemate-models/manifest.json; copy them into docs/MODELS.md"
    hint "$STAGE/lifemate-models" "--dir-mode zip" ;;
  *) echo "usage: KAGGLE_USER=<name> bash scripts/kaggle_pack.sh code|processed|raw|models"; exit 2 ;;
esac
