"""LoRA SFT of a causal LM on the step-wise agent data (loss only on the assistant turn being predicted).

Each assistant turn of a sample becomes one training example whose prompt is rendered EXACTLY like inference
(apply_chat_template(messages[:k], add_generation_prompt=True, enable_thinking=False)), so train/inference match.

  python -I -X utf8 scripts/train_sft.py --model Qwen/Qwen3-4B --data data/processed/sft_train.jsonl data/processed/sft_train_asr.jsonl \
      --out checkpoints/q4b_lora --epochs 3
  smoke test: add --max-steps 3 --model Qwen/Qwen3-0.6B
Multi-GPU (Kaggle L4x4, data-parallel; one full copy of the model per GPU):
  torchrun --nproc_per_node=4 scripts/train_sft.py --model <local dir> --data ... --out ... --global-batch 16       --save-steps 200 --max-hours 10.5 --resume
"""
import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint


def warmup_kwargs(ratio):
    """transformers 5 removed warmup_ratio (a float warmup_steps < 1 means a ratio there); 4.x only knows warmup_ratio."""
    import dataclasses
    return {"warmup_ratio": ratio} if "warmup_ratio" in {f.name for f in dataclasses.fields(TrainingArguments)} else {"warmup_steps": ratio}


class TimeLimit(TrainerCallback):
    """Stop (after saving) once the wall-clock budget is used, so a Kaggle session never ends mid-checkpoint.
    The decision is all-reduced so every rank stops on the same step (otherwise DDP would hang)."""

    def __init__(self, hours):
        self.t0, self.limit = time.time(), hours * 3600

    def on_step_end(self, args, state, control, **kw):
        if state.global_step % 10:
            return
        over = float(time.time() - self.t0 > self.limit)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            t = torch.tensor([over], device=f"cuda:{args.local_rank}" if args.local_rank >= 0 else "cuda")
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX)
            over = t.item()
        if over:
            control.should_save = True
            control.should_training_stop = True


def step_examples(messages, tok, max_len):
    out = []
    for k, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        prompt = tok.apply_chat_template(messages[:k], add_generation_prompt=True, tokenize=False, enable_thinking=False)
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        t_ids = tok(m["content"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
        if len(p_ids) + len(t_ids) > max_len:
            continue
        out.append({"input_ids": p_ids + t_ids, "labels": [-100] * len(p_ids) + t_ids})
    return out


class Collate:
    def __init__(self, pad_id):
        self.pad = pad_id

    def __call__(self, batch):
        n = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), n), self.pad, dtype=torch.long)
        lab = torch.full((len(batch), n), -100, dtype=torch.long)
        att = torch.zeros((len(batch), n), dtype=torch.long)
        for i, b in enumerate(batch):
            L = len(b["input_ids"])
            ids[i, :L] = torch.tensor(b["input_ids"]); lab[i, :L] = torch.tensor(b["labels"]); att[i, :L] = 1
        return {"input_ids": ids, "labels": lab, "attention_mask": att}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--accum", type=int, default=16, help="per-GPU gradient accumulation")
    ap.add_argument("--global-batch", type=int, default=0, help="if set, accum = global_batch // (bs * n_gpus)")
    ap.add_argument("--save-steps", type=int, default=0, help="0 = save each epoch, else every N optimizer steps")
    ap.add_argument("--max-hours", type=float, default=0, help="stop and save after this many hours (0 = no limit)")
    ap.add_argument("--save-limit", type=int, default=2, help="checkpoints kept (raise it to compare epochs)")
    ap.add_argument("--resume", action="store_true", help="continue from the last checkpoint in --out if there is one")
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)

    world = int(os.environ.get("WORLD_SIZE", 1))
    main_proc = int(os.environ.get("RANK", 0)) == 0
    if a.global_batch:
        a.accum = max(1, a.global_batch // (a.bs * world))
    tok = AutoTokenizer.from_pretrained(a.model)
    ex = []
    for f in a.data:
        for l in open(f, encoding="utf-8"):
            ex += step_examples(json.loads(l)["messages"], tok, a.max_len)
    if main_proc:
        print(f"{len(ex)} step examples; mean len {np.mean([len(e['input_ids']) for e in ex]):.0f}; "
              f"{world} GPU(s) x bs {a.bs} x accum {a.accum} = global batch {world * a.bs * a.accum}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    if main_proc:
        model.print_trainable_parameters()
    args = TrainingArguments(
        output_dir=a.out, per_device_train_batch_size=a.bs, gradient_accumulation_steps=a.accum, num_train_epochs=a.epochs,
        max_steps=a.max_steps, learning_rate=a.lr, lr_scheduler_type="cosine", **warmup_kwargs(0.03), logging_steps=10,
        save_strategy="steps" if a.save_steps else "epoch", save_steps=a.save_steps or 500, save_total_limit=a.save_limit,
        ddp_find_unused_parameters=False, bf16=torch.cuda.is_bf16_supported(), fp16=not torch.cuda.is_bf16_supported(),
        seed=a.seed, data_seed=a.seed, report_to="none", remove_unused_columns=False,
        dataloader_num_workers=0)
    trainer = Trainer(model=model, args=args, train_dataset=ex, data_collator=Collate(tok.pad_token_id),
                      callbacks=[TimeLimit(a.max_hours)] if a.max_hours else [])
    last = get_last_checkpoint(a.out) if (a.resume and os.path.isdir(a.out)) else None
    if last and main_proc:
        print("resuming from", last, flush=True)
    trainer.train(resume_from_checkpoint=last)
    if trainer.is_world_process_zero():
        model.save_pretrained(a.out)
        tok.save_pretrained(a.out)
        print("saved", a.out)


if __name__ == "__main__":
    main()
