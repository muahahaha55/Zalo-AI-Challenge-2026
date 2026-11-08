"""Dev-time, GPU (Kaggle): write the spoken request for each generated sample with a LOCAL language model, then let the
same model check that the text asks for exactly those calls.
  python -I -X utf8 scripts/gen_text.py --model /kaggle/input/<qwen3-14b> --inp data/processed/callgen_pending.jsonl \
      --out data/processed/callgen_text.jsonl [--batch 32] [--seed 42] [--no-verify]
The model runs offline on the machine (never a third-party API: RULES 1.3 / team decision). Its size does not matter for
the submission because it only builds training data; bigger writes and judges better (14B-32B fits a 96 GB card).
Output rows = input rows + "audio_text" (cleaned to one line) + "verified" (bool, unless --no-verify). build_sft.py
--callgen drops rows with verified == false and applies lifemate.callgen.text_ok. Rows already in --out are skipped."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lifemate.callgen import clean_text, is_yes, verify_prompt  # noqa: E402
from lifemate.jsonl import read_jsonl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="local folder of an instruction model (Qwen3-14B or similar)")
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--no-verify", action="store_true")
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    set_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda").eval()

    def complete(prompts: list[str], sample: bool, max_new: int) -> list[str]:
        chats = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False) for p in prompts]
        enc = tok(chats, return_tensors="pt", padding=True).to("cuda")
        kw = {"do_sample": True, "temperature": a.temperature, "top_p": 0.95} if sample else {"do_sample": False}
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new, pad_token_id=tok.pad_token_id, **kw)
        return tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    rows = read_jsonl(a.inp)
    out = Path(a.out)
    done = {r["id"] for r in read_jsonl(out)} if out.is_file() else set()
    todo = [r for r in rows if r["id"] not in done]
    print(f"{len(rows)} rows, {len(done)} already done, {len(todo)} to write")
    kept = 0
    with out.open("a", encoding="utf-8", newline="\n") as f:
        for s in range(0, len(todo), a.batch):
            chunk = todo[s : s + a.batch]
            texts = [clean_text(t) for t in complete([r["prompt"] for r in chunk], True, a.max_new_tokens)]
            verdicts = [None] * len(chunk)
            if not a.no_verify:
                verdicts = [is_yes(v) for v in complete([verify_prompt(r, t) for r, t in zip(chunk, texts, strict=True)], False, 4)]
            for r, t, v in zip(chunk, texts, verdicts, strict=True):
                row = {**r, "audio_text": t}
                if v is not None:
                    row["verified"] = v
                    kept += v
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(f"  {min(s + a.batch, len(todo))}/{len(todo)}  verified so far {kept}", flush=True)


if __name__ == "__main__":
    main()
