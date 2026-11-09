"""Dev-time, GPU (Kaggle): round-trip check of generated samples. A local model reads the generated request and acts in the
base sample's environment through our normal agent loop; the sample passes if it reaches the same final state as the label
(same state hash) and uses at least the label's tools.
  python -I -X utf8 scripts/verify_callgen.py --model $QWEN --adapter checkpoints/q4b_lora --inp data/processed/callgen_text.jsonl \
      --out data/processed/callgen_checked.jsonl --check-all --control data/processed/dev_ids.txt --control-n 100

Which verifier? The agent prompt asks for ONE JSON tool call per turn, a protocol only a model trained on it follows reliably
(an untuned base model may fail even on good text). Hence the CONTROL: the same verifier is run on real dev samples (their real
clean text and label), and its pass rate there is the ceiling. A verifier whose control pass rate is low (< ~30%) is useless as
a filter. With the fine-tuned adapter the pass rate on generated samples, compared with the control, is a measure of how
consistent the generated text is with its label; as a FILTER it keeps mostly what the model already solves (easy samples),
so prefer reading the numbers over blindly filtering with it.

--check-all   run the agent on every row (also those rejected by the cheap filters), then print the pass rate of the rows
              that passed the filters against the rest: if both are alike, the cheap filters carry no information.
Output rows = input rows + "prefilter_ok", "agent_ok" (the agent reached the label) and "roundtrip" (= both; what build_sft.py reads). Rows already in --out are skipped."""
import argparse
import json
import sys
from collections import Counter
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
    ap.add_argument("--check-all", action="store_true", help="also run rows that the cheap filters reject (diagnostic)")
    ap.add_argument("--control", default=None, help="file of real train/dev sample ids (dev_ids.txt) for the positive control")
    ap.add_argument("--control-n", type=int, default=100)
    ap.add_argument("--time-budget", type=float, default=60.0, help="seconds per sample (generous: this is not the submission)")
    a = ap.parse_args()

    train = split_dir("train")
    cat = load_catalog(train / "train_tools_catalog.json")
    env_cls = load_tool_env_class(train)
    new_env = lambda i: env_cls("train", i, root=str(data_root()))  # noqa: E731
    agent = Agent(cat, HFBackend(a.model, a.adapter), AgentConfig(time_budget_s=a.time_budget))

    def passes(sample: dict, base_id: str, text: str, history_texts: list[str]) -> bool:
        gold = new_env(base_id)
        for c in sample["label"]["tool_calls"]:
            gold.call(c["name"], c["arguments"])
        outcome = agent.run(sample, new_env(base_id), text, history_texts)
        # same final state AND the label's tools (a read-only label leaves the state unchanged, so the hash alone would
        # accept an agent that did nothing)
        used = {c["name"] for c in outcome.tool_calls}
        return outcome.state_hash == gold.get_state_hash() and {c["name"] for c in sample["label"]["tool_calls"]} <= used

    if a.control:
        ids = Path(a.control).read_text(encoding="utf-8").split()[: a.control_n]
        real = {it["id"]: it for it in read_jsonl(train / "train.jsonl")}
        ok = 0
        for i in ids:
            it = real[i]
            ok += passes(it, i, it["audio_text"], [h["audio_text"] for h in it["history"] if h["role"] == "user"])
        print(f"CONTROL: {ok}/{len(ids)} real samples pass ({100 * ok / len(ids):.1f}%): the ceiling of this verifier", flush=True)

    out = Path(a.out)
    done = {r["id"] for r in read_jsonl(out)} if out.is_file() else set()
    rows = [r for r in read_jsonl(a.inp) if r["id"] not in done]
    tally: Counter = Counter()
    with out.open("a", encoding="utf-8", newline="\n") as f:
        for n, r in enumerate(rows, 1):
            pre = r.get("verified") is not False and text_ok(r["audio_text"], r, cat)[0]
            ok = False
            if pre or a.check_all:
                ok = passes(r, r["base"], r["audio_text"], [])
                tally[(pre, ok)] += 1
            # "roundtrip" is what build_sft.py reads: kept only if the cheap filters AND the agent agree; "agent_ok" is the diagnostic
            f.write(json.dumps({**r, "prefilter_ok": pre, "agent_ok": ok, "roundtrip": pre and ok}, ensure_ascii=False) + "\n")
            if n % 50 == 0:
                f.flush()
                print(f"  {n}/{len(rows)}  {dict(tally)}", flush=True)
    for pre in (True, False):
        total = tally[(pre, True)] + tally[(pre, False)]
        if total:
            print(f"prefilter {'passed' if pre else 'rejected'}: {tally[(pre, True)]}/{total} pass the round trip "
                  f"({100 * tally[(pre, True)] / total:.1f}%)")


if __name__ == "__main__":
    main()
