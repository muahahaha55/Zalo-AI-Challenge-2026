#!/usr/bin/env bash
# Build the submission image and the file to upload (organisers' guide 2026: docker save | gzip, md5sum, Google Drive).
#   bash scripts/build_submission.sh <TeamName> [tag]          (tag default v1)
#   TORCH_INDEX=https://download.pytorch.org/whl/cu128 bash scripts/build_submission.sh test   # local test build
#   GZIP_CMD=pigz ...        faster compression on a many-core machine (apt install pigz)
#   SKIP_MODEL_CHECK=1 ...   build with test models (e.g. small ones) that are not in models.lock.json
# Output: zac2026_<TeamName>.tar.gz and zac2026_<TeamName>.tar.gz.md5 in the repo root (git-ignored).
set -euo pipefail
cd "$(dirname "$0")/.."
TEAM=${1:?usage: build_submission.sh <TeamName> [tag]}
TAG=${2:-v1}
IMAGE="zac2026-$(echo "$TEAM" | tr '[:upper:]' '[:lower:]'):$TAG"
PY=${PY:-.venv/Scripts/python}; [ -x "$PY" ] || PY=python3

if [ -n "$(git status --porcelain --untracked-files=no)" ]; then echo "WARNING: uncommitted changes; COMMIT.txt will say dirty"; fi
{ git rev-parse HEAD; [ -z "$(git status --porcelain --untracked-files=no)" ] || echo dirty; } > docker/COMMIT.txt

if [ "${SKIP_MODEL_CHECK:-0}" != 1 ]; then "$PY" -I -X utf8 scripts/check_models.py --models models; fi
"$PY" -I -X utf8 scripts/check_compliance.py

docker build -f docker/Dockerfile ${TORCH_INDEX:+--build-arg TORCH_INDEX=$TORCH_INDEX} -t "$IMAGE" .
OUT="zac2026_${TEAM}.tar.gz"
docker save "$IMAGE" | ${GZIP_CMD:-gzip} > "$OUT"       # GZIP_CMD=pigz uses every core (same .gz format)
md5sum "$OUT" | tee "$OUT.md5"
ls -la "$OUT"
echo "Upload $OUT to Google Drive, share 'Anyone with the link', then paste the link and the MD5 above into the portal."
