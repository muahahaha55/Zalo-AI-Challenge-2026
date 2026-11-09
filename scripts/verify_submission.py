"""Pre-submission check (CPU only). Mirrors what the organisers do: replay the submitted tool_calls in order on a fresh
ToolEnv of the split with its own zaic_tools, then compare the hash with the submitted one.

  python -I -X utf8 scripts/verify_submission.py --split public_test --sub outputs/sub_public_v1.jsonl
  (--allow-partial: only check the ids present, e.g. for sample_submission.jsonl)

Checks: one line per sample (no missing/extra/duplicate ids, split order), UTF-8 JSONL keys, no rejected call
(schema error), no tool outside installed_apps + sys.ask_missing_info, hash reproduced by the replay.
Dev tool: it uses env.call/get_state_hash only, never reads *_environment/ itself.
"""
import argparse
import importlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lifemate.paths import data_root  # noqa: E402

DATA = data_root()      # data/raw/train+public_test, or LIFEMATE_DATA_ROOT (the folder that holds the split folders)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="public_test")
    ap.add_argument("--sub", required=True)
    ap.add_argument("--allow-partial", action="store_true")
    a = ap.parse_args()

    sys.path.insert(0, str(DATA / a.split))
    ToolEnv = importlib.import_module("zaic_tools").ToolEnv
    items = [json.loads(l) for l in open(DATA / a.split / f"{a.split}.jsonl", encoding="utf-8") if l.strip()]
    order = [i["id"] for i in items]
    installed = {i["id"]: set(i["context"]["installed_apps"]) for i in items}

    problems, rows = [], []
    with open(a.sub, "rb") as f:
        raw = f.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        sys.exit(f"FAIL: not UTF-8: {e}")
    for n, l in enumerate(text.splitlines(), 1):
        if not l.strip():
            problems.append(f"line {n}: empty line")
            continue
        try:
            d = json.loads(l)
        except json.JSONDecodeError as e:
            problems.append(f"line {n}: invalid JSON ({e})")
            continue
        if set(d) != {"id", "tool_calls", "state_hash"}:
            problems.append(f"line {n}: keys {sorted(d)} != id/tool_calls/state_hash")
        rows.append(d)

    ids = [r.get("id") for r in rows]
    dup = [k for k, c in Counter(ids).items() if c > 1]
    missing = [k for k in order if k not in set(ids)]
    extra = [k for k in ids if k not in set(order)]
    if dup:
        problems.append(f"duplicate ids: {dup[:5]}")
    if extra:
        problems.append(f"unknown ids: {extra[:5]}")
    if missing and not a.allow_partial:
        problems.append(f"{len(missing)} ids missing (first: {missing[:3]})")
    if not missing and not extra and not dup and ids != order:
        problems.append("ids are not in the split's order (allowed, but sample_submission uses split order)")

    ok = bad_hash = rejected = foreign = 0
    for r in rows:
        sid = r.get("id")
        if sid not in installed:
            continue
        env = ToolEnv(a.split, sid, root=str(DATA))
        for c in r.get("tool_calls", []):
            name = c.get("name", "")
            if name != "sys.ask_missing_info" and name.split(".")[0] not in installed[sid]:
                foreign += 1
                problems.append(f"{sid}: tool {name} not in installed_apps")
            res = env.call(name, c.get("arguments", {}))
            if res.get("ok") is False:
                rejected += 1
                problems.append(f"{sid}: call {name} rejected: {res.get('error')}")
        if env.get_state_hash() == r.get("state_hash"):
            ok += 1
        else:
            bad_hash += 1
            problems.append(f"{sid}: state_hash differs from replay")

    n_calls = [len(r.get("tool_calls", [])) for r in rows]
    print(f"{a.sub}: {len(rows)} lines / {len(order)} samples in {a.split}")
    print(f"  hash reproduced: {ok}   hash differs: {bad_hash}   rejected calls: {rejected}   tools outside installed_apps: {foreign}")
    print(f"  empty tool_calls: {sum(1 for x in n_calls if x == 0)}   mean calls/sample: {sum(n_calls) / max(len(n_calls), 1):.2f}   max: {max(n_calls, default=0)}")
    if problems:
        print("FAIL:")
        for p in problems[:25]:
            print("  -", p)
        if len(problems) > 25:
            print(f"  ... and {len(problems) - 25} more")
        sys.exit(1)
    print("OK: safe to run `python -m zaic_tools --split public_test --root . --validate <sub>` and submit (after approval)")


if __name__ == "__main__":
    main()
