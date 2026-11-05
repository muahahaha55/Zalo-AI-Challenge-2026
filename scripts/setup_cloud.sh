#!/usr/bin/env bash
# Set up a rented Linux GPU box (vast.ai / Kaggle) for LifeMate. Run from the repo root on the remote machine.
#   bash scripts/setup_cloud.sh
# Data is NOT in git: copy data/raw/train+public_test and data/processed to the box with rsync/scp (see docs/CLOUD.md).
set -euo pipefail

command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
sudo -n true 2>/dev/null && SUDO=sudo || SUDO=""
$SUDO apt-get update -qq && $SUDO apt-get install -y -qq libsndfile1 ffmpeg git rsync tmux >/dev/null

python3 -m venv .venv
. .venv/bin/activate
python -m pip install -q -U pip
python -m pip install -q torch                      # default Linux wheel ships CUDA
python -m pip install -q -r requirements-gpu-linux.txt
python -I -X utf8 - <<'PY'
import torch, transformers, peft
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("transformers", transformers.__version__, "peft", peft.__version__)
try:
    import vllm; print("vllm", vllm.__version__)
except Exception as e:
    print("vllm not available:", e)
PY
git config core.hooksPath .githooks
echo "setup done. Next: check data is in place with: ls data/raw/train+public_test/train | head"
