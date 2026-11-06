"""Merge per-GPU shard files into one jsonl (dev tool, no competition data is read except the files you pass).

  # submissions from run_infer.py --shard: restore the split's sample order and check nothing is missing
  python -I -X utf8 scripts/merge_shards.py --key id --order-from data/raw/train+public_test/public_test/public_test.jsonl \
      --out outputs/pub.jsonl outputs/pub.part*.jsonl
  # ASR shards from run_asr.py --shard
  python -I -X utf8 scripts/merge_shards.py --key path --out data/processed/asr_phowl_ft_public_test.jsonl \
      data/processed/asr_phowl_ft_public_test.part*.jsonl
"""
import argparse
import json
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True, choices=["id", "path"])
    ap.add_argument("--order-from", help="split jsonl whose 'id' order (and id set) the output must follow; --key id only")
    ap.add_argument("--allow-missing", action="store_true",
                    help="subset runs (--ids/--limit): ids of the split that no shard produced are dropped instead of being an error")
    ap.add_argument("--out", required=True)
    ap.add_argument("parts", nargs="+")
    a = ap.parse_args()

    rows = {}
    for f in a.parts:
        for l in open(f, encoding="utf-8"):
            if l.strip():
                d = json.loads(l)
                rows[d[a.key]] = d              # a later duplicate (e.g. a resumed shard) overwrites an earlier one
    keys = sorted(rows)
    if a.order_from:
        want = [json.loads(l)["id"] for l in open(a.order_from, encoding="utf-8") if l.strip()]
        missing = [k for k in want if k not in rows]
        extra = [k for k in rows if k not in set(want)]
        if extra or (missing and not a.allow_missing):
            sys.exit(f"ERROR: {len(missing)} ids missing, {len(extra)} unexpected (first missing: {missing[:3]})")
        keys = [k for k in want if k in rows]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        for k in keys:
            f.write(json.dumps(rows[k], ensure_ascii=False) + "\n")
    print(f"merged {len(a.parts)} files -> {a.out}: {len(keys)} rows")


if __name__ == "__main__":
    main()
