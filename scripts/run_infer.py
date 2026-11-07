"""Run the agent over a split and write a submission.jsonl.
  python -I -X utf8 scripts/run_infer.py --split train --ids data/processed/dev_ids.txt --backend oracle --out outputs/dev_oracle.jsonl
  python -I -X utf8 scripts/run_infer.py --split train --ids ... --backend hf --model Qwen/Qwen3-4B --text clean --out ...
  python -I -X utf8 scripts/run_infer.py --split public_test --backend hf --model ... --text asr:phowl --out outputs/pub.jsonl
--text clean (train only: gold audio_text) | asr:<tag> (data/processed/asr_<tag>_<split>.jsonl)
--time-budget S  stop starting new generation steps once S seconds are used on a sample (0 = off; the contest limit is 30 s)
--resume         continue an interrupted run from <out>.part (rows go there and the file is renamed to <out> only when the run ends)"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lifemate.agent import AgentConfig  # noqa: E402
from lifemate.determinism import seed_everything  # noqa: E402
from lifemate.jsonl import ResumableJsonlWriter  # noqa: E402
from lifemate.pipeline import HFBackend, SplitRunner, Transcripts  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--ids")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", help="K/N: process only samples K, K+N, ... (one process per GPU; merge with merge_shards.py)")
    ap.add_argument("--backend", choices=["oracle", "hf"], required=True)
    ap.add_argument("--model")
    ap.add_argument("--adapter")
    ap.add_argument("--text", default="clean")
    ap.add_argument("--no-rules", action="store_true", help="disable the deterministic post-processing (src/lifemate/rules.py)")
    ap.add_argument("--time-budget", type=float, default=0.0)
    ap.add_argument("--resume", action="store_true", help="continue an interrupted run from <out>.part")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    seed_everything(0)

    backend = None
    if a.backend == "hf":
        backend = HFBackend(a.model, a.adapter)
        print("LLM params (B):", backend.param_count() / 1e9, flush=True)
    config = AgentConfig(use_rules=not a.no_rules, time_budget_s=a.time_budget or None)
    runner = SplitRunner(a.split, config, Transcripts.from_spec(a.text, a.split), backend=backend, oracle=a.backend == "oracle")
    ids = Path(a.ids).read_text().split() if a.ids else None
    shard = tuple(map(int, a.shard.split("/"))) if a.shard else None
    items = runner.items(ids, a.limit, shard)

    def progress(n, total, avg):
        if (n - 1) % 20 == 0:
            print(f"{n}/{total}  avg {avg:.2f}s", flush=True)

    with ResumableJsonlWriter(a.out, resume=a.resume) as writer:    # LF, flushed per row, renamed to --out only on success
        todo = [i for i in items if i["id"] not in writer.done]
        for row in runner.run(todo, progress):
            writer.write(row)
    print("done", runner.stats.summary())


if __name__ == "__main__":
    main()
