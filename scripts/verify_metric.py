"""Independent checks that our scorer/hash pipeline agrees with what the organisers ship. Run: python -I -X utf8 scripts/verify_metric.py"""
import json, re, sys, importlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "train+public_test"
sys.path.insert(0, str(ROOT / "src"))
from lifemate.metric import load_fuzzy_params, count_matches, token_f1  # noqa: E402


def env_for(split):
    for k in [k for k in sys.modules if k == "zaic_tools" or k.startswith("zaic_tools.")]:
        del sys.modules[k]
    sys.path.insert(0, str(DATA / split))
    mod = importlib.import_module("zaic_tools")
    sys.path.pop(0)
    return mod


# A. External ground truth: organiser's sample_submission.jsonl (3 lines) -> replay with public zaic_tools, compare full hash
print("A. sample_submission.jsonl replay vs organiser hashes")
m = env_for("public_test")
for l in open(DATA / "public_test" / "sample_submission.jsonl", encoding="utf-8"):
    d = json.loads(l)
    env = m.ToolEnv("public_test", d["id"], root=str(DATA))
    rejected = [c["name"] for c in d["tool_calls"] if env.call(c["name"], c["arguments"]).get("ok") is False]
    h = env.get_state_hash()
    print(f"  {d['id']}: ours={h[:12]} theirs={d['state_hash'][:12]} match={h == d['state_hash']} rejected={rejected}")

# B. Gold hash vs label.final_state: replay gold calls; for every app present in final_state, canon(env state) must equal canon(final_state)
print("B. replay gold vs label.final_state (train)")
m = env_for("train")
canon = importlib.import_module("zaic_tools.utils").canon
items = [json.loads(l) for l in open(DATA / "train" / "train.jsonl", encoding="utf-8")]
bad = []
for it in items:
    env = m.ToolEnv("train", it["id"], root=str(DATA))
    for c in it["label"]["tool_calls"]:
        env.call(c["name"], c["arguments"])
    got, want = canon(env.get_state()), canon(it["label"]["final_state"])
    if any(got.get(a) != want[a] for a in want):
        bad.append(it["id"])
print(f"  mismatches: {len(bad)}/{len(items)} {bad[:5]}")

# C. Determinism: a second fresh env replaying the same gold gives the same hash
env1 = m.ToolEnv("train", items[0]["id"], root=str(DATA)); env2 = m.ToolEnv("train", items[0]["id"], root=str(DATA))
for c in items[0]["label"]["tool_calls"]:
    env1.call(c["name"], c["arguments"]); env2.call(c["name"], c["arguments"])
print("C. determinism:", env1.get_state_hash() == env2.get_state_hash())

# D. Sensitivity to the one ambiguous choice (token definition for token-F1): perturb every fuzzy gold value
print("D. fuzzy token-definition sensitivity (whitespace vs regex word tokens)")
fz = load_fuzzy_params(DATA / "train" / "train_tools_catalog.json")
import unicodedata
from collections import Counter

def f1_regex(a, b):
    t = lambda s: re.findall(r"\w+", unicodedata.normalize("NFC", s).lower())
    ta, tb = t(a), t(b); c = sum((Counter(ta) & Counter(tb)).values())
    if not ta and not tb: return 1.0
    if not c: return 0.0
    p, r = c / len(ta), c / len(tb); return 2 * p * r / (p + r)

vals = [(c["name"], k, v) for it in items for c in it["label"]["tool_calls"] for k, v in c["arguments"].items() if k in fz.get(c["name"], ()) and isinstance(v, str)]
dis = tot = 0
for _, _, v in vals:
    w = v.split()
    for pert in (" ".join(w[:-1]), " ".join(w[1:]), v.rstrip(".,!") , " ".join(w + ["nhé"]), " ".join(w[: max(1, len(w) // 2)])):
        tot += 1
        dis += (token_f1(pert, v) > 0.6) != (f1_regex(pert, v) > 0.6)
print(f"  fuzzy gold values: {len(vals)}; perturbations: {tot}; decisions that differ between tokenizers: {dis} ({100*dis/max(tot,1):.2f}%)")
