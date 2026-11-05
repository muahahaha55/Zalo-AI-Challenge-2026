"""Build SFT data from train: data/processed/sft_{train,dev}.jsonl (+ dev_ids.txt) and verify round-trip.
Usage: python -I -X utf8 scripts/build_sft.py [--tokenizer Qwen/Qwen3-4B]"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "train+public_test" / "train"
OUT = ROOT / "data" / "processed"
sys.path.insert(0, str(ROOT / "src"))
from lifemate.prompt import load_catalog, build_messages, END  # noqa: E402


def groups(items, thr=0.6):
    """Union-find over near-duplicate transcripts (Jaccard of word sets >= thr) so dev has no template leakage."""
    toks = [set(re.findall(r"\w+", it["audio_text"].lower())) for it in items]
    par = list(range(len(items)))

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            u = len(toks[i] | toks[j])
            if u and len(toks[i] & toks[j]) / u >= thr:
                par[find(i)] = find(j)
    return [find(i) for i in range(len(items))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--asr-tag", default=None, help="also write *_asr.jsonl with ASR transcripts (data/processed/asr_<tag>_train.jsonl)")
    a = ap.parse_args()
    items = [json.loads(l) for l in open(DATA / "train.jsonl", encoding="utf-8")]
    cat = load_catalog(DATA / "train_tools_catalog.json")
    g = groups(items)
    ids = sorted(set(g))
    random.Random(a.seed).shuffle(ids)
    dev_groups, n = set(), 0
    for gid in ids:  # fill dev with whole groups until ~10% of samples
        if n >= 0.1 * len(items):
            break
        dev_groups.add(gid)
        n += g.count(gid)
    OUT.mkdir(parents=True, exist_ok=True)
    asr = {}
    if a.asr_tag:
        for l in open(OUT / f"asr_{a.asr_tag}_train.jsonl", encoding="utf-8"):
            d = json.loads(l)
            asr[d["path"]] = d["text"]
    tr, dv, tra, dva = [], [], [], []
    for it, gi in zip(items, g):
        htexts = [h["audio_text"] for h in it["history"] if h["role"] == "user"]
        msgs = build_messages(it, cat, it["audio_text"], htexts)
        # round trip: assistant turns must decode back to the gold calls
        dec = [json.loads(m["content"]) for m in msgs if m["role"] == "assistant" and m["content"] != END]
        gold = [{"name": c["name"], "arguments": c["arguments"]} for c in it["label"]["tool_calls"]]
        assert dec == gold, it["id"]
        calls = it["label"]["tool_calls"]
        assert (msgs[-1]["content"] == END) == (not calls or calls[-1]["name"] != "sys.ask_missing_info"), it["id"]
        (dv if gi in dev_groups else tr).append({"id": it["id"], "messages": msgs})
        if asr:   # same targets, but the request and history turns are the ASR transcripts (noisy, as at inference)
            at = asr[it["audio"]]
            ah = [asr[h["audio"]] for h in it["history"] if h["role"] == "user"]
            (dva if gi in dev_groups else tra).append({"id": it["id"], "messages": build_messages(it, cat, at, ah)})
    for name, rows in (("sft_train", tr), ("sft_dev", dv), ("sft_train_asr", tra), ("sft_dev_asr", dva)):
        if not rows:
            continue
        with open(OUT / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (OUT / "dev_ids.txt").write_text("\n".join(r["id"] for r in dv) + "\n")
    print(f"groups={len(set(g))} train={len(tr)} dev={len(dv)} (dev groups={len(dev_groups)})")
    if a.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.tokenizer)
        L = sorted(len(tok.apply_chat_template(r["messages"], tokenize=True)) for r in tr + dv)
        S = sorted(len(tok(r["messages"][1]["content"])["input_ids"]) for r in tr + dv)
        q = lambda v, p: v[int(p * (len(v) - 1))]
        print("full-sample tokens  mean=%.0f p50=%d p95=%d max=%d" % (sum(L) / len(L), q(L, .5), q(L, .95), L[-1]))
        print("first user msg tok  mean=%.0f p50=%d p95=%d max=%d" % (sum(S) / len(S), q(S, .5), q(S, .95), S[-1]))


if __name__ == "__main__":
    main()
