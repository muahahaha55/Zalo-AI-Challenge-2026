"""LoRA fine-tune of PhoWhisper on train audio (final request + history user turns), dev samples held out.
Targets are normalised transcripts (lowercase, no punctuation, digits kept) like PhoWhisper's own style.
At the end the LoRA is merged and saved to <out>/merged so scripts/run_asr.py --model <out>/merged works unchanged.

  python -I -X utf8 scripts/train_asr.py --model vinai/PhoWhisper-large --out checkpoints/phowl_ft --epochs 3
  smoke test: --model vinai/PhoWhisper-small --max-steps 3
Only train audio is used (never public_test / private_test).
"""
import argparse
import json
import random
import re
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from peft import LoraConfig, get_peft_model
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperForConditionalGeneration, WhisperProcessor

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "train+public_test" / "train"
OUT = ROOT / "data" / "processed"


def warmup_kwargs(ratio):
    """transformers 5 removed warmup_ratio (a float warmup_steps < 1 means a ratio there); 4.x only knows warmup_ratio."""
    import dataclasses
    return {"warmup_ratio": ratio} if "warmup_ratio" in {f.name for f in dataclasses.fields(Seq2SeqTrainingArguments)} else {"warmup_steps": ratio}


def norm(s):
    s = unicodedata.normalize("NFC", s).lower()
    return " ".join(re.sub(r"[^\w\s]", " ", s).split())


class Collate:
    def __init__(self, proc, start_id):
        self.p, self.start = proc, start_id

    def __call__(self, batch):
        wavs = []
        for path, _ in batch:
            x, sr = sf.read(DATA / path, dtype="float32")
            assert sr == 16000
            wavs.append(x if x.ndim == 1 else x.mean(1))
        feats = self.p.feature_extractor(wavs, sampling_rate=16000, return_tensors="pt").input_features
        lab = [self.p.tokenizer(t).input_ids for _, t in batch]
        n = max(map(len, lab))
        labels = torch.full((len(lab), n), -100, dtype=torch.long)
        for i, l in enumerate(lab):
            labels[i, : len(l)] = torch.tensor(l)
        if (labels[:, 0] == self.start).all():       # decoder prepends the start token itself
            labels = labels[:, 1:]
        return {"input_features": feats, "labels": labels}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vinai/PhoWhisper-large")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--save-steps", type=int, default=0, help="0 = save each epoch")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)

    dev = set((OUT / "dev_ids.txt").read_text().split())
    rows = []
    for l in open(DATA / "train.jsonl", encoding="utf-8"):
        it = json.loads(l)
        if it["id"] in dev:
            continue
        rows.append((it["audio"], norm(it["audio_text"])))
        rows += [(h["audio"], norm(h["audio_text"])) for h in it["history"] if h["role"] == "user"]
    print(len(rows), "training utterances (dev samples excluded)", flush=True)

    proc = WhisperProcessor.from_pretrained(a.model, language="vietnamese", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(a.model, dtype=torch.bfloat16)
    model.config.forced_decoder_ids = None
    model.generation_config.language, model.generation_config.task = "vi", "transcribe"
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05,
                                             target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]))
    model.print_trainable_parameters()
    args = Seq2SeqTrainingArguments(
        output_dir=a.out, per_device_train_batch_size=a.bs, gradient_accumulation_steps=a.accum, num_train_epochs=a.epochs,
        max_steps=a.max_steps, learning_rate=a.lr, lr_scheduler_type="cosine", **warmup_kwargs(0.05), logging_steps=10,
        save_strategy="steps" if a.save_steps else "epoch", save_steps=a.save_steps or 500, save_total_limit=2,
        ddp_find_unused_parameters=False, bf16=True, seed=a.seed, data_seed=a.seed, report_to="none",
        remove_unused_columns=False, dataloader_num_workers=0, label_names=["labels"])
    start = model.config.decoder_start_token_id
    trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=rows, data_collator=Collate(proc, start))
    trainer.train()
    if trainer.is_world_process_zero():       # only one rank merges and writes (DDP runs this script once per GPU)
        merged = model.merge_and_unload()
        merged.config.use_cache = True
        merged.save_pretrained(Path(a.out) / "merged")
        proc.save_pretrained(Path(a.out) / "merged")
        print("saved", Path(a.out) / "merged")


if __name__ == "__main__":
    main()
