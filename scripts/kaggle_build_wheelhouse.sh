#!/usr/bin/env bash
# Build an offline wheelhouse. Run ONCE in a Kaggle notebook with Internet ON and NO competition data attached
# (only the public pip index is contacted). Then publish /kaggle/working/wheelhouse as a PRIVATE dataset
# named lifemate-wheels; every later notebook runs with Internet OFF.
#   bash scripts/kaggle_build_wheelhouse.sh [requirements-kaggle.txt]
set -euo pipefail
REQ="${1:-requirements-kaggle.txt}"
OUT=/kaggle/working/wheelhouse
mkdir -p "$OUT"
# keep the image's own numpy/torch: pin the resolver to the versions already installed
python -m pip list --format=freeze | grep -iE '^(torch|torchvision|torchaudio|numpy)==' > /tmp/constraints.txt || true
python -m pip download -q -d "$OUT" -r "$REQ" -c /tmp/constraints.txt
# the big GPU stack is already in the image; drop whatever the resolver pulled in for it
rm -f "$OUT"/torch-* "$OUT"/torchvision-* "$OUT"/torchaudio-* "$OUT"/triton-* "$OUT"/nvidia_* "$OUT"/numpy-* 2>/dev/null || true
python --version
du -sh "$OUT"
ls "$OUT" | wc -l
# dry run: does the offline install resolve?
python -m pip install --no-index --find-links "$OUT" --dry-run -r "$REQ" >/dev/null && echo "offline install check: OK"
cp "$REQ" "$OUT"/
echo "Now publish $OUT as a PRIVATE dataset (lifemate-wheels). Python version must match later notebooks (same Kaggle image)."
