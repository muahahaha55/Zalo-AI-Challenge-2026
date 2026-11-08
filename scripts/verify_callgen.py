"""Dev-time, GPU (Kaggle): round-trip check of generated samples. A local model reads the generated request and acts in the
base sample's environment through our normal agent loop; the sample is kept only if it reaches the same final state as
the label (same state hash). This catches text that does not ask for what the label does, which neither the writer
model's own yes/no judgement nor lexical checks reliably catch.
  python -I -X utf8 scripts/verify_callgen.py --model /kaggle/input/<qwen3-14b> --inp data/processed/callgen_text.jsonl \
      --out data/processed/callgen_checked.jsonl [--adapter checkpoints/q4b_lora]
Use a model DIFFERENT from the one being trained, ideally a bigger untuned one: with our own fine-tuned model the filter
would keep only what that model already gets right. Output rows = input rows + "roundtrip" (bool). Only rows that pass
lifemate.callgen.text_ok (and are not verified == false) are checked; the others are written with roundtrip false.
Rows already in --out are skipped (resumable)."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lifemate.agent import Agent, AgentConfig  # noqa: E402
from lifemate.callgen import text_ok  # noqa: E402
from lifemate.jsonl import read_jsonl  # noqa: E402
from lifemate.paths import data_root, split_dir  # noqa: E402
from lifemate.pipeline import HFBackend, load_tool_env_class  # noqa: E402
from lifemate.prompt import load_catalog  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--time-budget", type=float, default=60.0, help="seconds per sample (generous: this is not the submission)")
    a = ap.parse_args()

    train = split_dir("train")
    cat = load_catalog(train / "train_tools_catalog.json")
    env_cls = load_tool_env_class(train)
    new_env = lambda i: env_cls("train", i, root=str(data_root()))  # noqa: E731
    agent = Agent(cat, HFBackend(a.model, a.adapter), AgentConfig(time_budget_s=a.time_budget))

    out = Path(a.out)
    done = {r["id"] for r in read_jsonl(out)} if out.is_file() else set()
    rows = [r for r in read_jsonl(a.inp) if r["id"] not in done]
    passed = checked = 0
    with out.open("a", encoding="utf-8", newline="\n") as f:
        for n, r in enumerate(rows, 1):
            ok = r.get("verified") is not False and text_ok(r["audio_text"], r, cat)[0]
            if ok:
                gold = new_env(r["base"])
                for c in r["label"]["tool_calls"]:
                    gold.call(c["name"], c["arguments"])
                outcome = agent.run(r, new_env(r["base"]), r["audio_text"], [])
                # same final state, and the same tools used (a read-only label leaves the state unchanged, so the hash
                # alone would accept an agent that did nothing)
                names = {c["name"] for c in outcome.tool_calls}
                ok = outcome.state_hash == gold.get_state_hash() and {c["name"] for c in r["label"]["tool_calls"]} <= names
                checked += 1
                passed += ok
            f.write(json.dumps({**r, "roundtrip": ok}, ensure_ascii=False) + "\n")
            if n % 50 == 0:
                f.flush()
                print(f"  {n}/{len(rows)}  checked {checked}, passed {passed}", flush=True)
    print(f"done: checked {checked}, passed {passed}")


if __name__ == "__main__":
    main()
