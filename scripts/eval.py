"""Score a submission.jsonl against train gold.  Usage:
   python -I -X utf8 scripts/eval.py --sub outputs/sub.jsonl [--ids dev_ids.txt]
   python -I -X utf8 scripts/eval.py --selftest          # gold -> 100.0, empty submission -> oos rate
Gold hashes are produced by replaying the gold calls on the official ToolEnv (cached in data/processed/)."""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "train+public_test"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(DATA / "train"))
from zaic_tools import ToolEnv                      # noqa: E402
from lifemate.metric import load_fuzzy_params, score_sample  # noqa: E402

CACHE = ROOT / "data" / "processed" / "gold_hash_train.json"


def load_train():
    with open(DATA / "train" / "train.jsonl", encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def gold_hashes(items):
    if CACHE.exists():
        return json.loads(CACHE.read_text(encoding="utf-8"))
    out, bad = {}, []
    for it in items:
        env = ToolEnv("train", it["id"], root=str(DATA))
        for c in it["label"]["tool_calls"]:
            r = env.call(c["name"], c["arguments"])
            if not r.get("ok", True) and r.get("ok") is False:
                bad.append((it["id"], c["name"], r.get("error")))
        out[it["id"]] = env.get_state_hash()
    if bad:
        print("WARNING gold calls rejected by env:", bad[:5], len(bad))
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(out), encoding="utf-8")
    return out


def kind(it):
    tc = it["label"]["tool_calls"]
    if not tc:
        return "oos"
    return "ask" if any(c["name"] == "sys.ask_missing_info" for c in tc) else "exec"


def evaluate(sub, items, gh, fuzzy):
    rows = []
    for it in items:
        s = sub.get(it["id"], {"tool_calls": [], "state_hash": None})
        rows.append((it, score_sample(s["tool_calls"], s["state_hash"], it["label"]["tool_calls"], gh[it["id"]], fuzzy)))
    return rows


def report(rows):
    n = len(rows)
    print(f"N={n}  score={100 * sum(s for _, s in rows) / n:.2f}")
    groups = defaultdict(list)
    for it, s in rows:
        groups["kind=" + kind(it)].append(s)
        groups["gold_calls=" + str(min(len(it["label"]["tool_calls"]), 5))].append(s)
        groups["history=" + str(bool(it["history"]))].append(s)
    for k in sorted(groups):
        v = groups[k]
        print(f"  {k:16s} n={len(v):5d}  mean={100 * sum(v) / len(v):6.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub")
    ap.add_argument("--ids", help="file with one id per line (e.g. dev split)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    items = load_train()
    if a.ids:
        keep = set(Path(a.ids).read_text().split())
        items = [i for i in items if i["id"] in keep]
    fuzzy = load_fuzzy_params(DATA / "train" / "train_tools_catalog.json")
    gh = gold_hashes(load_train())
    if a.selftest:
        gold = {i["id"]: {"tool_calls": i["label"]["tool_calls"], "state_hash": gh[i["id"]]} for i in items}
        print("-- gold as submission (expect 100)"); report(evaluate(gold, items, gh, fuzzy))
        print("-- empty submission (expect = oos share)"); report(evaluate({}, items, gh, fuzzy))
    else:
        sub = {}
        for l in open(a.sub, encoding="utf-8"):
            d = json.loads(l); sub[d["id"]] = d
        report(evaluate(sub, items, gh, fuzzy))
