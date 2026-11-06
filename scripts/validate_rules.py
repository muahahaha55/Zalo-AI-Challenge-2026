"""Measure src/lifemate/rules.py on the train labels (dev tool; uses clean transcripts audio_text, or ASR text with --asr).

  .venv/Scripts/python.exe -I -X utf8 scripts/validate_rules.py [--asr data/processed/asr_phowl_ft_train.jsonl] [--show 12]

A) accuracy: start from a deliberately WRONG value, check whether the rule restores the gold value exactly.
B) false fires: start from the GOLD value (a perfect model); the rule must change nothing.
"""
import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lifemate.rules import HHMM, ISO_D, ISO_DT, extract_when, fix_args, spoken_numbers  # noqa: E402

SLIPS = [timedelta(hours=1), timedelta(hours=-1), timedelta(days=1), timedelta(days=-1), timedelta(hours=12), timedelta(days=1, hours=1)]


def wrong_value(v: str, n: int) -> str:
    """A realistic model slip (1 h, 1 day, AM/PM, ...), rotating deterministically."""
    slip = SLIPS[n % len(SLIPS)]
    m = ISO_DT.match(v)
    if m:
        dt = datetime(int(m[1]), int(m[2]), int(m[3]), int(m[4]), int(m[5])) + slip
        return dt.strftime("%Y-%m-%dT%H:%M") + m[6]
    if HHMM.match(v):
        return f"{(int(v[:2]) + int(slip.total_seconds() // 3600)) % 24:02d}:{v[3:]}"
    return (datetime.fromisoformat(v) + timedelta(days=abs(slip.days) or 1)).strftime("%Y-%m-%d")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asr")
    ap.add_argument("--show", type=int, default=10)
    a = ap.parse_args()
    asr = {}
    if a.asr:
        asr = {json.loads(l)["path"]: json.loads(l)["text"] for l in open(a.asr, encoding="utf-8")}
    items = [json.loads(l) for l in open(ROOT / "data/raw/train+public_test/train/train.jsonl", encoding="utf-8")]

    tot = Counter(); bad_examples = []; false_fires = []
    for it in items:
        text = asr.get(it["audio"], it["audio_text"]) if asr else it["audio_text"]
        now = it["context"]["current_time"]
        for c in it["label"]["tool_calls"]:
            args = c["arguments"] or {}
            dt_keys = [k for k, v in args.items() if isinstance(v, str) and (ISO_DT.match(v) or HHMM.match(v) or ISO_D.match(v))]
            if len(dt_keys) == 1:
                k = dt_keys[0]
                tot["gold time args (1 datetime in call)"] += 1
                when = extract_when(text, datetime.fromisoformat(now))
                if when is None:
                    continue
                tot["rule has an opinion"] += 1
                tot["_n"] += 1
                new, ch = fix_args(c["name"], {**args, k: wrong_value(args[k], tot["_n"])}, text, now)
                if new[k] == args[k]:
                    tot["  fully restored"] += 1
                else:
                    tot["  NOT restored"] += 1
                    bad_examples.append((it["id"], text[:90], now, f"{c['name']}.{k}", args[k], new[k]))
                _, ch2 = fix_args(c["name"], args, text, now)
                if ch2:
                    tot["  false fire on a correct value"] += 1
                    false_fires.append((it["id"], text[:90], ch2))
            for k, v in args.items():
                if "phone" in k and isinstance(v, str) and v.isdigit():
                    nums = spoken_numbers(text)
                    tot["gold phone args"] += 1
                    if len(nums) == 1:
                        tot["  phone: one spoken number"] += 1
                        tot["  phone: spoken == gold"] += nums[0] == v
                        _, ch3 = fix_args(c["name"], {**args, k: "0000000000"}, text, now)
                    if len(nums) == 1 and nums[0] != v:
                        bad_examples.append((it["id"], text[:90], now, f"{c['name']}.{k}", v, nums[0]))
    tot.pop("_n", None)
    for k, v in tot.items():
        print(f"{k:48s} {v}")
    print(f"\n{min(a.show, len(bad_examples))} lỗi điển hình (id | câu | now | tham số | gold | luật):")
    for e in bad_examples[: a.show]:
        print(" -", " | ".join(map(str, e)))
    print(f"\n{min(a.show, len(false_fires))} lần luật sửa nhầm giá trị đúng:")
    for e in false_fires[: a.show]:
        print(" -", e)


if __name__ == "__main__":
    main()
