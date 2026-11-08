"""Dev-time: generate new training samples (calls + results, no text yet) by transplanting real call segments.
Usage: python -I -X utf8 scripts/gen_calls.py --n 3000 [--seed 42] [--out data/processed/callgen_pending.jsonl]
Writes one row per sample with the instruction ("prompt") for a LOCAL language model (scripts/gen_text.py, run on Kaggle).
Dev samples (data/processed/dev_ids.txt) are never used as base or donor. See lifemate.callgen."""
import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lifemate.callgen import describe, generate, style_examples  # noqa: E402
from lifemate.jsonl import read_jsonl, write_jsonl  # noqa: E402
from lifemate.paths import data_root, processed_dir, split_dir  # noqa: E402
from lifemate.pipeline import load_tool_env_class  # noqa: E402
from lifemate.prompt import load_catalog  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(processed_dir() / "callgen_pending.jsonl"))
    a = ap.parse_args()
    train = split_dir("train")
    cat = load_catalog(train / "train_tools_catalog.json")
    dev = set((processed_dir() / "dev_ids.txt").read_text(encoding="utf-8").split())
    items = [it for it in read_jsonl(train / "train.jsonl") if it["id"] not in dev]
    env_cls = load_tool_env_class(train)
    samples, stats = generate(items, cat, lambda i: env_cls("train", i, root=str(data_root())), random.Random(a.seed), a.n)
    rng = random.Random(a.seed + 1)
    rows = [{**s, "prompt": describe(s, style_examples(s, items, rng))} for s in samples]
    write_jsonl(a.out, rows)
    print(f"{len(rows)} samples -> {a.out}; stats {stats}")


if __name__ == "__main__":
    main()
