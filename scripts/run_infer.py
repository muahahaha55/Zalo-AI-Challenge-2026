"""Run the agent over a split and write a submission.jsonl.
  python -I -X utf8 scripts/run_infer.py --split train --ids data/processed/dev_ids.txt --backend oracle --out outputs/dev_oracle.jsonl
  python -I -X utf8 scripts/run_infer.py --split train --ids ... --backend hf --model Qwen/Qwen3-4B --text clean --out ...
  python -I -X utf8 scripts/run_infer.py --split public_test --backend hf --model ... --text asr:phowl --out outputs/pub.jsonl
--text clean (train only: gold audio_text) | asr:<tag> (data/processed/asr_<tag>_<split>.jsonl)"""
import argparse
import importlib
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "train+public_test"
sys.path.insert(0, str(ROOT / "src"))
from lifemate.agent import OracleBackend, run_sample  # noqa: E402
from lifemate.prompt import build_messages, load_catalog  # noqa: E402


def hf_backend(model_path, adapter=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path)
    kw = dict(dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    model = AutoModelForCausalLM.from_pretrained(model_path, **kw).cuda().eval()
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).eval()
    print("LLM params (B):", sum(p.numel() for p in model.parameters()) / 1e9, flush=True)

    def gen(messages):
        ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", enable_thinking=False,
                                      return_dict=True).to("cuda")
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=256, do_sample=False, temperature=None, top_p=None, top_k=None)
        return tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
    return gen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--ids")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", help="K/N: process only samples K, K+N, ... (one process per GPU; merge with merge_shards.py)")
    ap.add_argument("--backend", choices=["oracle", "hf"], required=True)
    ap.add_argument("--model")
    ap.add_argument("--adapter")
    ap.add_argument("--text", default="clean")
    ap.add_argument("--no-rules", action="store_true", help="disable the deterministic post-processing (src/lifemate/rules.py)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    random.seed(0)

    sys.path.insert(0, str(DATA / a.split))
    ToolEnv = importlib.import_module("zaic_tools").ToolEnv
    cat = load_catalog(DATA / a.split / f"{a.split}_tools_catalog.json")
    items = [json.loads(l) for l in open(DATA / a.split / f"{a.split}.jsonl", encoding="utf-8")]
    if a.ids:
        keep = set(Path(a.ids).read_text().split())
        items = [i for i in items if i["id"] in keep]
    if a.limit:
        items = items[: a.limit]
    if a.shard:
        k, n = map(int, a.shard.split("/"))
        items = items[k::n]
    asr = {}
    if a.text.startswith("asr:"):
        for l in open(ROOT / "data" / "processed" / f"asr_{a.text[4:]}_{a.split}.jsonl", encoding="utf-8"):
            d = json.loads(l); asr[d["path"]] = d["text"]
    gen = hf_backend(a.model, a.adapter) if a.backend == "hf" else None

    times = []
    with open(a.out, "w", encoding="utf-8") as fout:
        for n, it in enumerate(items):
            t0 = time.time()
            if asr:
                transcript = asr[it["audio"]]
                htexts = [asr[h["audio"]] for h in it["history"] if h["role"] == "user"]
            else:
                transcript = it["audio_text"]
                htexts = [h["audio_text"] for h in it["history"] if h["role"] == "user"]
            env = ToolEnv(a.split, it["id"], root=str(DATA))        # exactly one env per sample
            g = OracleBackend(build_messages(it, cat, transcript, htexts)) if a.backend == "oracle" else gen
            calls, h = run_sample(it, env, cat, transcript, htexts, g, use_rules=not a.no_rules)
            times.append(time.time() - t0)
            fout.write(json.dumps({"id": it["id"], "tool_calls": calls, "state_hash": h}, ensure_ascii=False) + "\n")
            if n % 20 == 0:
                print(f"{n + 1}/{len(items)}  avg {sum(times) / len(times):.2f}s", flush=True)
    ts = sorted(times)
    print(f"done {len(items)} samples; mean {sum(ts) / len(ts):.2f}s p95 {ts[int(.95 * (len(ts) - 1))]:.2f}s max {ts[-1]:.2f}s")


if __name__ == "__main__":
    main()
