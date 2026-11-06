"""WER/CER of an ASR output against train audio_text (final request + history user turns).
Usage: python -I -X utf8 scripts/asr_wer.py --tag phowl [--dev-only]"""
import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

try:
    import jiwer
except ImportError:          # e.g. a Kaggle notebook without Internet: same corpus-level WER/CER in pure Python
    class jiwer:             # noqa: N801  (mimics the two functions used below)
        @staticmethod
        def _edits(r, h):
            prev = list(range(len(h) + 1))
            for i, x in enumerate(r, 1):
                cur = [i]
                for j, y in enumerate(h, 1):
                    cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
                prev = cur
            return prev[-1]

        @classmethod
        def _rate(cls, refs, hyps, split):
            if isinstance(refs, str):
                refs, hyps = [refs], [hyps]
            edits = sum(cls._edits(split(r), split(h)) for r, h in zip(refs, hyps))
            return edits / max(1, sum(len(split(r)) for r in refs))

        @classmethod
        def wer(cls, refs, hyps):
            return cls._rate(refs, hyps, str.split)

        @classmethod
        def cer(cls, refs, hyps):
            return cls._rate(refs, hyps, list)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lifemate.normalize import words_to_digits  # noqa: E402

DATA = ROOT / "data" / "raw" / "train+public_test" / "train"
OUT = ROOT / "data" / "processed"


NO_NUM = False


def norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s).lower()
    s = " ".join(re.sub(r"[^\w\s]", " ", s).split())
    return s if NO_NUM else words_to_digits(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--dev-only", action="store_true")
    ap.add_argument("--no-num", action="store_true", help="do not convert spoken numbers to digits")
    a = ap.parse_args()
    global NO_NUM
    NO_NUM = a.no_num
    hyp = {}
    for l in open(OUT / f"asr_{a.tag}_train.jsonl", encoding="utf-8"):
        d = json.loads(l)
        hyp[d["path"]] = d["text"]
    dev = set(open(OUT / "dev_ids.txt").read().split()) if a.dev_only else None
    refs, hyps, meta = [], [], []
    for l in open(DATA / "train.jsonl", encoding="utf-8"):
        it = json.loads(l)
        if dev is not None and it["id"] not in dev:
            continue
        pairs = [(it["audio"], it["audio_text"], "final")] + [(h["audio"], h["audio_text"], "hist") for h in it["history"] if h["role"] == "user"]
        for path, ref, kind in pairs:
            if path in hyp:
                refs.append(norm(ref)); hyps.append(norm(hyp[path])); meta.append(kind)
    print(f"n={len(refs)} utterances")
    print("WER %.2f%%  CER %.2f%%" % (100 * jiwer.wer(refs, hyps), 100 * jiwer.cer(refs, hyps)))
    for k in ("final", "hist"):
        idx = [i for i, m in enumerate(meta) if m == k]
        if idx:
            print(f"  {k}: WER {100 * jiwer.wer([refs[i] for i in idx], [hyps[i] for i in idx]):.2f}%")
    worst = sorted(range(len(refs)), key=lambda i: -jiwer.wer(refs[i], hyps[i]))[:8]
    for i in worst:
        print(" REF:", refs[i], "\n HYP:", hyps[i])


if __name__ == "__main__":
    main()
