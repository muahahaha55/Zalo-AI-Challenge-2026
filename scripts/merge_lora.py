"""Merge the LoRA adapter into the base LLM and save the result as one fine-tuned model folder.

  python -I -X utf8 scripts/merge_lora.py --base <Qwen3-4B folder> --adapter checkpoints/q4b_lora --out checkpoints/q4b_merged

Why: the organisers' Docker guide says the base model used in training must not be shipped in the image, only the
fine-tuned weights used at inference. The merge is done in float32 and saved in bfloat16 (the dtype used at inference);
rounding can change a few greedy outputs, so the merged model MUST be re-evaluated on dev (clean and ASR) and on
public_test before it replaces base + adapter. The tokenizer and chat template are copied from the adapter folder (they
are the ones used in training). Parameter count after merging: the base count (the adapter adds nothing).
"""
import argparse
import shutil
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.float32)
    model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload()
    model = model.to(torch.bfloat16)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    tok_src = a.adapter if (Path(a.adapter) / "tokenizer_config.json").is_file() else a.base
    AutoTokenizer.from_pretrained(tok_src).save_pretrained(out)
    for extra in ("chat_template.jinja",):
        if (Path(tok_src) / extra).is_file():
            shutil.copy2(Path(tok_src) / extra, out / extra)
    n = sum(p.numel() for p in model.parameters())
    print(f"merged model saved to {out}: {n / 1e9:.4f}B parameters, dtype bfloat16")


if __name__ == "__main__":
    main()
