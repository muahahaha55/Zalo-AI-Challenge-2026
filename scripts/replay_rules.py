"""Offline effect of src/lifemate/rules.py on an existing dev submission (no model needed).

  .venv/Scripts/python.exe -I -X utf8 scripts/replay_rules.py --sub outputs/logs/dev_sft_phowl.jsonl [--asr data/processed/asr_phowl_train.jsonl]

Each predicted call is passed through fix_args and executed on a fresh train ToolEnv (one per sample), then scored against gold with the
same metric as eval.py. A control run without the rules must reproduce the original score. Dev tool: train data only.
"""
import argparse
import importlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("lm_eval", ROOT / "scripts" / "eval.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
from lifemate.metric import load_fuzzy_params, score_sample  # noqa: E402
from lifemate.rules import fix_args  # noqa: E402


def replay(it, calls, text, use_rules):
    env = ev.ToolEnv("train", it["id"], root=str(ev.DATA))
    changes = []
    for c in calls:
        args, ch = fix_args(c["name"], c["arguments"], text, it["context"]["current_time"]) if use_rules else (c["arguments"], [])
        changes += ch
        env.call(c["name"], args)
    return env.get_tool_calls(), env.get_state_hash(), changes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", required=True)
    ap.add_argument("--asr", help="ASR jsonl of the transcript the model saw (default: clean audio_text, optimistic)")
    ap.add_argument("--ids", default=str(ROOT / "data/processed/dev_ids.txt"))
    a = ap.parse_args()
    asr = {json.loads(l)["path"]: json.loads(l)["text"] for l in open(a.asr, encoding="utf-8")} if a.asr else {}
    keep = set(Path(a.ids).read_text().split())
    items = [i for i in ev.load_train() if i["id"] in keep]
    sub = {json.loads(l)["id"]: json.loads(l) for l in open(a.sub, encoding="utf-8")}
    fuzzy = load_fuzzy_params(ev.DATA / "train" / "train_tools_catalog.json")
    gh = ev.gold_hashes(ev.load_train())

    tot0 = tot1 = 0.0
    better, worse, fired = [], [], 0
    for it in items:
        text = asr.get(it["audio"], it["audio_text"])
        calls = sub[it["id"]]["tool_calls"]
        c0, h0, _ = replay(it, calls, text, False)
        c1, h1, ch = replay(it, calls, text, True)
        s0 = score_sample(c0, h0, it["label"]["tool_calls"], gh[it["id"]], fuzzy)
        s1 = score_sample(c1, h1, it["label"]["tool_calls"], gh[it["id"]], fuzzy)
        tot0 += s0; tot1 += s1
        if ch:
            fired += 1
            (better if s1 > s0 else worse if s1 < s0 else []).append((it["id"], s0, s1, ch))
    n = len(items)
    print(f"N={n}  control (no rules, replayed) = {100 * tot0 / n:.2f}   with rules = {100 * tot1 / n:.2f}   delta = {100 * (tot1 - tot0) / n:+.2f}")
    print(f"samples where a rule fired: {fired}; improved: {len(better)}; got worse: {len(worse)}")
    for tag, lst in (("IMPROVED", better), ("WORSE", worse)):
        for sid, s0, s1, ch in lst:
            print(f"  {tag} {sid}: {s0:.2f} -> {s1:.2f}  {ch}")


if __name__ == "__main__":
    main()
