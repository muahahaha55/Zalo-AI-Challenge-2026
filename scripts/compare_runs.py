"""Paired comparison of two dev runs (dev has only 155 samples, so look at per-sample differences, not just the means).

  python -I -X utf8 scripts/compare_runs.py --a outputs/dev_runs/dev_sft_phowl_ft.jsonl --b outputs/dev_q4b_hist_phowl_ft.jsonl \
      [--ids data/processed/dev_ids.txt] [--names old,new]

Prints: both means, mean paired difference with a bootstrap 95% interval (resampling samples), counts of samples that got
better / worse / equal, the same split by sample kind (exec/ask/oos), and the most changed sample ids.
A difference whose interval contains 0 is not evidence of improvement. Train gold only (dev samples); never public_test."""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lifemate.metric import load_fuzzy_params, score_sample  # noqa: E402

DATA = ROOT / "data" / "raw" / "train+public_test" / "train"


def kind(calls):
    if not calls:
        return "oos"
    return "ask" if any(c["name"] == "sys.ask_missing_info" for c in calls) else "exec"


def per_sample(path, items, gh, fuzzy):
    sub = {}
    for l in open(path, encoding="utf-8"):
        d = json.loads(l)
        sub[d["id"]] = d
    out = {}
    for it in items:
        s = sub.get(it["id"], {"tool_calls": [], "state_hash": None})
        out[it["id"]] = score_sample(s["tool_calls"], s["state_hash"], it["label"]["tool_calls"], gh[it["id"]], fuzzy)
    return out


def bootstrap_ci(diffs, n=5000, seed=0):
    rng = random.Random(seed)
    m = len(diffs)
    means = sorted(sum(diffs[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--ids")
    ap.add_argument("--names", default="A,B")
    a = ap.parse_args()
    na, nb = a.names.split(",")
    items = [json.loads(l) for l in open(DATA / "train.jsonl", encoding="utf-8")]
    if a.ids:
        keep = set(Path(a.ids).read_text().split())
        items = [i for i in items if i["id"] in keep]
    gh = json.loads((ROOT / "data" / "processed" / "gold_hash_train.json").read_text(encoding="utf-8"))
    fuzzy = load_fuzzy_params(DATA / "train_tools_catalog.json")
    sa, sb = per_sample(a.a, items, gh, fuzzy), per_sample(a.b, items, gh, fuzzy)
    ids = [i["id"] for i in items]
    kinds = {i["id"]: kind(i["label"]["tool_calls"]) for i in items}
    d = [sb[i] - sa[i] for i in ids]
    n = len(ids)
    lo, hi = bootstrap_ci(d)
    print(f"N={n}   {na}: {100 * sum(sa.values()) / n:.2f}   {nb}: {100 * sum(sb.values()) / n:.2f}")
    print(f"paired diff ({nb}-{na}): {100 * sum(d) / n:+.2f} pts   95% bootstrap interval [{100 * lo:+.2f}, {100 * hi:+.2f}]"
          f"   -> {'distinguishable from 0' if lo > 0 or hi < 0 else 'NOT distinguishable from 0'}")
    better = sum(x > 1e-9 for x in d)
    worse = sum(x < -1e-9 for x in d)
    print(f"samples better {better}, worse {worse}, equal {n - better - worse}")
    by = defaultdict(list)
    for i, x in zip(ids, d):
        by[kinds[i]].append(x)
    for k in ("exec", "ask", "oos"):
        v = by.get(k, [])
        if v:
            print(f"  {k:5s} n={len(v):3d}  diff {100 * sum(v) / len(v):+6.2f}   better {sum(x > 1e-9 for x in v)}  worse {sum(x < -1e-9 for x in v)}")
    top = sorted(zip(ids, d), key=lambda t: -abs(t[1]))[:8]
    print("largest changes:", ", ".join(f"{i}({100 * x:+.0f})" for i, x in top))


if __name__ == "__main__":
    main()
