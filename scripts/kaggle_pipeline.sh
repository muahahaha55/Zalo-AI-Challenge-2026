#!/usr/bin/env bash
# HANDOFF.md sections 1-2 as resumable stages for a multi-GPU box (Kaggle L4x4; also fine on 1 GPU, e.g. a 4090).
# Run inside the notebook after kaggle_setup.sh:
#   source /kaggle/working/lifemate/kaggle_env.sh
#   bash scripts/kaggle_pipeline.sh <stage>   stages: asr_base | sft_asr_data | train | asr_ft | sft_ft_data | aug_data | eval_dev | status
#   SMOKE=1 bash scripts/kaggle_pipeline.sh <stage>   tiny run to test the plumbing (never touches real artifacts);
#   SMOKE_STEPS=30 makes the train stage run 30 optimizer steps instead of 3, to read the s/step speed
#   DRYRUN=1 ...  print the commands only;  FORCE=1 ... redo a stage that is already marked done
# Each stage writes outputs/logs/<stage>.log and outputs/.done_<stage>; re-running a finished stage is a no-op.
# Env: QWEN, PHO (model dirs, set by kaggle_env.sh); NGPU (default: all visible); MAXLEN (default 5120: the longest sample is 4.4K tokens)
#      RUN (adapter name, default q4b_lora; use e.g. q4b_aug for a new run), SKIP_ASR=1 (train only the LLM, keep checkpoints/phowl_ft), AUG=1 / HIST=1 / RETRACT=1 (train also on sft_train_aug.jsonl / sft_train_hist.jsonl / sft_train_retract.jsonl from stage aug_data),
#      SAVE_LIMIT (checkpoints kept, default 2), EPOCHS_SFT / EPOCHS_ASR (default 3); MAX_HOURS (SFT wall-clock budget, default 10.5); TAG (eval_dev: phowl or phowl_ft)
set -euo pipefail
cd "$(dirname "$0")/.."
STAGE=${1:-status}
QWEN=${QWEN:?source kaggle_env.sh first}
PHO=${PHO:?source kaggle_env.sh first}
NGPU=${NGPU:-$(python -I -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)}
MAXLEN=${MAXLEN:-5120}
RUN=${RUN:-q4b_lora}
EPOCHS_SFT=${EPOCHS_SFT:-3}
EPOCHS_ASR=${EPOCHS_ASR:-3}
mkdir -p outputs/logs checkpoints data/processed
PY="python -I -X utf8"
D=data/processed
SPLITS="train public_test"

# SMOKE never touches real artifacts: *_smoke ASR tags, checkpoints/smoke, and the data-building stages are skipped
LIMIT=""; SMOKE_SFT=""; SMOKE_ASR=""; SMOKE_INF=""; SFX=""; CK=checkpoints
if [ "${SMOKE:-0}" = 1 ]; then
  LIMIT="--limit 16"; SMOKE_SFT="--max-steps ${SMOKE_STEPS:-3}"; SMOKE_ASR="--max-steps ${SMOKE_STEPS:-3}"; SMOKE_INF="--limit 8"; SFX="_smoke"; CK=checkpoints/smoke
fi

run() { echo "+ $*"; [ "${DRYRUN:-0}" = 1 ] || eval "$*"; }

# asr_shards <model> <tag>: transcribe every split on all GPUs (one process per GPU), then merge the shards
asr_shards() {
  local model=$1 tag=$2$SFX n=$NGPU split k
  for split in $SPLITS; do
    if [ -s "$D/asr_${tag}_${split}.jsonl" ] && [ "${SMOKE:-0}" != 1 ]; then echo "asr_${tag}_${split}.jsonl exists, skipping"; continue; fi
    rm -f "$D"/asr_${tag}_${split}.part*.jsonl
    for ((k = 0; k < n; k++)); do
      run "CUDA_VISIBLE_DEVICES=$k $PY scripts/run_asr.py --model $model --tag $tag --split $split --shard $k/$n $LIMIT &"
    done
    run "wait"
    run "$PY scripts/merge_shards.py --key path --out $D/asr_${tag}_${split}.jsonl $D/asr_${tag}_${split}.part*.jsonl"
  done
}

stage_body() {
  case "$STAGE" in
    asr_base)        # HANDOFF 1: PhoWhisper-large on train + public_test (all wavs incl. history turns), then WER
      asr_shards "$PHO" phowl
      if [ "${SMOKE:-0}" != 1 ]; then run "$PY scripts/asr_wer.py --tag phowl"; fi ;;   # expect ~26.2% overall
    sft_asr_data)    # HANDOFF 1: ASR variant of the SFT data -> sft_train_asr.jsonl / sft_dev_asr.jsonl
      if [ "${SMOKE:-0}" = 1 ]; then echo "skipped in SMOKE (would overwrite real data)"; return 0; fi
      run "$PY scripts/build_sft.py --asr-tag phowl" ;;
    train)           # HANDOFF 2.1 + 2.2: ASR LoRA on the last GPU, SFT LoRA on the others (sequential if only 1 GPU)
      local sft_gpus=0 asr_gpu=0 nsft=1 data="$D/sft_train.jsonl"
      if [ "$NGPU" -ge 2 ]; then asr_gpu=$((NGPU - 1)); nsft=$((NGPU - 1)); sft_gpus=$(seq -s, 0 $((NGPU - 2))); fi
      if [ -s "$D/sft_train_asr.jsonl" ]; then data="$data $D/sft_train_asr.jsonl"; fi
      if [ "${AUG:-0}" = 1 ]; then
        if [ -s "$D/sft_train_aug.jsonl" ]; then data="$data $D/sft_train_aug.jsonl"; else echo "AUG=1 but $D/sft_train_aug.jsonl is missing: run stage aug_data first"; exit 1; fi
      fi
      if [ "${HIST:-0}" = 1 ]; then
        if [ -s "$D/sft_train_hist.jsonl" ]; then data="$data $D/sft_train_hist.jsonl"; else echo "HIST=1 but $D/sft_train_hist.jsonl is missing: run stage aug_data first"; exit 1; fi
      fi
      if [ "${RETRACT:-0}" = 1 ]; then
        if [ -s "$D/sft_train_retract.jsonl" ]; then data="$data $D/sft_train_retract.jsonl"; else echo "RETRACT=1 but $D/sft_train_retract.jsonl is missing: run stage aug_data first"; exit 1; fi
      fi
      local asr_cmd="CUDA_VISIBLE_DEVICES=$asr_gpu $PY scripts/train_asr.py --model $PHO --out $CK/phowl_ft --epochs $EPOCHS_ASR --save-steps 100 $SMOKE_ASR"
      local sft_cmd="CUDA_VISIBLE_DEVICES=$sft_gpus torchrun --nproc_per_node=$nsft scripts/train_sft.py --model $QWEN --data $data --out $CK/$RUN --save-limit ${SAVE_LIMIT:-2} --epochs $EPOCHS_SFT --max-len $MAXLEN --global-batch 16 --save-steps 100 --max-hours ${MAX_HOURS:-10.5} --resume $SMOKE_SFT"
      if [ "${SKIP_ASR:-0}" = 1 ]; then      # LLM only: never touches checkpoints/phowl_ft; all GPUs go to the SFT
        nsft=$NGPU; sft_gpus=$(seq -s, 0 $((NGPU - 1)))
        sft_cmd="CUDA_VISIBLE_DEVICES=$sft_gpus torchrun --nproc_per_node=$nsft ${sft_cmd#*torchrun --nproc_per_node=* }"
        run "$sft_cmd"
      elif [ "$NGPU" -ge 2 ]; then
        run "$asr_cmd > outputs/logs/train_asr.log 2>&1 &"
        run "$sft_cmd"
        run "wait"
      else
        run "$asr_cmd"
        run "$sft_cmd"
      fi ;;
    asr_ft)          # HANDOFF 2.1: re-transcribe with the fine-tuned ASR, compare WER on dev
      if [ ! -d "$CK/phowl_ft/merged" ] && [ "${DRYRUN:-0}" != 1 ]; then echo "$CK/phowl_ft/merged missing: run stage train first"; exit 1; fi
      asr_shards "$CK/phowl_ft/merged" phowl_ft
      if [ "${SMOKE:-0}" != 1 ]; then
        run "$PY scripts/asr_wer.py --tag phowl_ft --dev-only"
        run "$PY scripts/asr_wer.py --tag phowl --dev-only"      # baseline to compare (expect ~25.4%)
      fi ;;
    sft_ft_data)
      if [ "${SMOKE:-0}" = 1 ]; then echo "skipped in SMOKE (would overwrite real data)"; return 0; fi
      run "$PY scripts/build_sft.py --asr-tag phowl_ft" ;;
    aug_data)        # train-only augmentation: synthetic out-of-scope samples + shuffled/dropped-app variants (see build_sft.py)
      if [ "${SMOKE:-0}" = 1 ]; then echo "skipped in SMOKE (would overwrite real data)"; return 0; fi
      run "$PY scripts/build_sft.py --asr-tag phowl_ft --aug-oos ${AUG_OOS:-100} --aug-variants ${AUG_VARIANTS:-1} --hist-max ${HIST_MAX:-700} --retract-copies ${RETRACT_COPIES:-2} --retract-replace ${RETRACT_REPLACE:-60} --retract-cancel ${RETRACT_CANCEL:-20}" ;;
    eval_dev)        # HANDOFF 2.3: dev score with the LoRA adapter; TAG=phowl (default) or phowl_ft
      local tag=${TAG:-phowl} txt
      txt="asr:$tag"; if [ "${SMOKE:-0}" = 1 ]; then txt=clean; fi   # smoke ASR files only cover a few wavs
      run "bash scripts/infer_multi_gpu.sh train outputs/dev_${RUN}_$tag$SFX.jsonl --ids $D/dev_ids.txt --backend hf --model $QWEN --adapter $CK/$RUN --text $txt $SMOKE_INF"
      if [ "${SMOKE:-0}" != 1 ]; then run "$PY scripts/eval.py --sub outputs/dev_${RUN}_$tag.jsonl --ids $D/dev_ids.txt"; fi ;;
    status)
      ls outputs/.done_* 2>/dev/null || echo "no stage finished yet"
      ls "$D" checkpoints 2>/dev/null ;;
    *) echo "unknown stage: $STAGE (asr_base|sft_asr_data|train|asr_ft|sft_ft_data|aug_data|eval_dev|status)"; exit 2 ;;
  esac
}

if [ "$STAGE" = status ]; then stage_body; exit 0; fi
if [ -e "outputs/.done_$STAGE" ] && [ "${FORCE:-0}" != 1 ] && [ "${SMOKE:-0}" != 1 ]; then
  echo "stage $STAGE already done (FORCE=1 to redo)"; exit 0
fi
echo "== stage $STAGE | GPUs: $NGPU | $(date -u +%FT%TZ)"
if [ "${DRYRUN:-0}" = 1 ]; then stage_body; exit 0; fi
stage_body 2>&1 | tee "outputs/logs/$STAGE.log"
[ "${SMOKE:-0}" = 1 ] || touch "outputs/.done_$STAGE"
echo "== stage $STAGE finished"
