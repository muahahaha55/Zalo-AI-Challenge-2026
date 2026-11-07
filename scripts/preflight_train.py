"""Check SFT data files before spending GPU time on them.
  python -I -X utf8 scripts/preflight_train.py --data data/processed/sft_train.jsonl data/processed/sft_train_asr.jsonl \
      --dev-ids data/processed/dev_ids.txt --max-len 5120 --tokenizer Qwen/Qwen3-4B [--hold-apps flights,bike_rental]
Checks per file: conversation structure, duplicate ids, rows that come from dev samples, mentions of held-out apps, and (with a
tokenizer) how many training steps are longer than --max-len and would be dropped. Exit code 1 if any problem is found."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lifemate.datacheck import check_rows  # noqa: E402
from lifemate.jsonl import iter_jsonl  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--dev-ids", help="file with dev sample ids: no row may come from them")
    ap.add_argument("--hold-apps", default="", help="comma-separated apps that must not appear anywhere (unseen-apps experiment)")
    ap.add_argument("--max-len", type=int, default=0, help="the trainer's --max-len (needs --tokenizer)")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--max-problems", type=int, default=10, help="problems printed per file")
    a = ap.parse_args()

    dev = set(Path(a.dev_ids).read_text().split()) if a.dev_ids else None
    hold = {x.strip() for x in a.hold_apps.split(",") if x.strip()} or None
    tok = None
    if a.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.tokenizer)
    bad = 0
    total_steps = 0
    for path in a.data:
        rep = check_rows(path, iter_jsonl(path), dev, hold, tok, a.max_len or None)
        total_steps += rep.step_examples
        extra = f", longest step {rep.longest} tokens, {rep.too_long} over max-len {a.max_len}" if tok and a.max_len else ""
        print(f"{'OK  ' if not rep.problems else 'FAIL'} {path}: {rep.rows} rows, {rep.step_examples} training steps{extra}")
        for p in rep.problems[: a.max_problems]:
            print("      -", p)
        if len(rep.problems) > a.max_problems:
            print(f"      ... {len(rep.problems) - a.max_problems} more")
        bad += bool(rep.problems)
    print(f"total training steps per epoch: {total_steps}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
