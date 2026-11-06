#!/usr/bin/env bash
# Run run_infer.py on every visible GPU in parallel (one process per GPU, sample-sharded), then merge.
#   bash scripts/infer_multi_gpu.sh <split> <out.jsonl> [extra run_infer.py args, not --split/--out/--shard]
# e.g. bash scripts/infer_multi_gpu.sh train outputs/dev.jsonl --ids data/processed/dev_ids.txt --backend hf --model $QWEN --text clean
set -euo pipefail
SPLIT=$1
OUT=$2
shift 2
N=$(python -I -c "import torch; print(torch.cuda.device_count())")
BASE="${OUT%.jsonl}"
rm -f "$BASE".part*.jsonl
pids=()
for ((k = 0; k < N; k++)); do
  CUDA_VISIBLE_DEVICES=$k python -I -X utf8 scripts/run_infer.py --split "$SPLIT" --shard "$k/$N" --out "$BASE.part$k.jsonl" "$@" \
    > "$BASE.part$k.log" 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
[ "$fail" = 0 ] || { echo "a shard failed; see $BASE.part*.log"; exit 1; }
SUBSET=""
for arg in "$@"; do
  case "$arg" in --ids|--limit) SUBSET="--allow-missing" ;; esac   # subset run: not every id of the split is expected
done
python -I -X utf8 scripts/merge_shards.py --key id $SUBSET --order-from "data/raw/train+public_test/$SPLIT/$SPLIT.jsonl" --out "$OUT" "$BASE".part?.jsonl
