"""WER/CER of an ASR output against train audio_text (final request + history user turns).
Usage: python -I -X utf8 scripts/asr_wer.py --tag phowl [--dev-only]"""
import argparse
import json
import re
import unicodedata
from pathlib import Path

import jiwer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "train+public_test" / "train"
OUT = ROOT / "data" / "processed"


def norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s).lower()
    return " ".join(re.sub(r"[^\w\s]", " ", s).split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--dev-only", action="store_true")
    a = ap.parse_args()
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
