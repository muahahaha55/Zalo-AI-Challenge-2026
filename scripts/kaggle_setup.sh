#!/usr/bin/env bash
# First cell of every Kaggle notebook (Internet OFF). Attach these PRIVATE datasets to the notebook:
# lifemate-code, lifemate-raw, lifemate-processed, lifemate-models, lifemate-wheels.
# Result: /kaggle/working/lifemate is the repo root with data wired in, and kaggle_env.sh holds env vars + paths.
#   bash "$(find /kaggle/input -name kaggle_setup.sh -print -quit)"
set -euo pipefail
KROOT=${KROOT:-/kaggle}          # overridable only to test these scripts on a fake tree
IN=$KROOT/input
W=$KROOT/working/lifemate

find_dir() {  # find_dir <marker path relative to the dataset root>  -> prints the dataset root
  local m
  m=$(find "$IN" -maxdepth 6 -path "*/$1" -print -quit 2>/dev/null || true)
  [ -n "$m" ] || { echo "ERROR: no attached dataset contains $1" >&2; exit 1; }
  echo "${m%/$1}"
}

CODE=$(find_dir "src/lifemate/agent.py")
RAW=$(find_dir "train/train.jsonl")
PROC=$(find_dir "dev_ids.txt")
QWEN=$(find_dir "qwen3-4b/config.json")/qwen3-4b
PHO=$(find_dir "phowhisper-large/config.json")/phowhisper-large
WHEELS=$(find "$IN" -maxdepth 6 -type d -name wheelhouse -print -quit 2>/dev/null || true)

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
rm -rf "$W"
mkdir -p "$W/data/raw" "$W/data/processed" "$KROOT"/working/outputs "$KROOT"/working/checkpoints
cp -r "$CODE"/. "$W"/
ln -s "$RAW" "$W/data/raw/train+public_test"            # read-only mount; the big audio stays on /kaggle/input
cp -r "$PROC"/. "$W/data/processed"/                     # writable copy: scripts write ASR/SFT outputs here
ln -sfn "$KROOT"/working/outputs "$W/outputs"
ln -sfn "$KROOT"/working/checkpoints "$W/checkpoints"

# Resume from a previous session: its notebook output is attached as an input (see kaggle_save_state.sh)
PREV=$(find "$IN" -maxdepth 6 -type d -path "*/state/processed" -print -quit 2>/dev/null || true)
if [ -n "$PREV" ]; then
  PREV=${PREV%/state/processed}
  echo "restoring previous session from $PREV"
  cp -r "$PREV/state/processed"/. "$W/data/processed"/             # newer ASR transcripts / SFT data override the base copy
  [ -d "$PREV/checkpoints" ] && cp -r "$PREV/checkpoints"/. "$KROOT"/working/checkpoints/
  [ -d "$PREV/outputs" ] && cp -r "$PREV/outputs"/. "$KROOT"/working/outputs/
fi

if [ -n "$WHEELS" ]; then
  python -m pip install -q --no-index --find-links "$WHEELS" -r "$W/requirements-kaggle.txt"
else
  echo "WARNING: no wheelhouse attached; relying on the packages preinstalled in the image"
fi

cat > "$W/kaggle_env.sh" <<ENV
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export QWEN=$QWEN PHO=$PHO
cd $W
ENV
. "$W/kaggle_env.sh"
python -I -X utf8 -c '
import torch, transformers, peft
print("torch", torch.__version__, "| gpus:", torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
print("transformers", transformers.__version__, "| peft", peft.__version__)
'
echo "setup done. In every later cell:  source /kaggle/working/lifemate/kaggle_env.sh"
