#!/bin/bash
# Organisers' entry point: docker run --gpus '"device=0"' -v <data>:/data:ro -v <result>:/result <image> /bin/bash /code/predict.sh
# Reads /data/<split>/ and writes /result/submission.jsonl (plus load / predict times on stdout). No network needed.
set -euo pipefail
cd /code
python3 /code/preprocessing.py
python3 /code/predict.py "$@"
