"""Transcribe every wav of a split (final request + history user turns) with a Whisper-family model.
Resumable; greedy; fp16. Output: data/processed/asr_<tag>_<split>.jsonl  {"path": "audio/<id>/<id>.wav", "text": ...}
Usage: python -I -X utf8 scripts/run_asr.py --model vinai/PhoWhisper-large --tag phowl --split train [--limit 40] [--bs 8]"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lifemate.asr import WhisperTranscriber  # noqa: E402
from lifemate.jsonl import iter_jsonl  # noqa: E402
from lifemate.paths import processed_dir, split_dir  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", help="file with sample ids; only transcribe audio of those samples")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--shard", help="K/N: transcribe every N-th wav starting at K; writes asr_<tag>_<split>.partK.jsonl "
                                    "(merge with merge_shards.py --key path)")
    a = ap.parse_args()
    random.seed(0)

    root = split_dir(a.split)
    paths = sorted(p.relative_to(root).as_posix() for p in (root / "audio").rglob("*.wav"))
    if a.ids:
        keep = set(Path(a.ids).read_text().split())
        paths = [p for p in paths if p.split("/")[1] in keep]
    if a.limit:
        paths = paths[: a.limit]
    out = processed_dir()
    out.mkdir(parents=True, exist_ok=True)
    if a.shard:
        k, n = map(int, a.shard.split("/"))
        paths = paths[k::n]
    out_path = out / f"asr_{a.tag}_{a.split}{f'.part{k}' if a.shard else ''}.jsonl"
    done = {d["path"] for d in iter_jsonl(out_path)} if out_path.exists() else set()
    todo = [p for p in paths if p not in done]
    print(f"{len(paths)} wavs, {len(done)} done, {len(todo)} to do", flush=True)
    if not todo:
        return

    asr = WhisperTranscriber(a.model, batch_size=a.bs)
    print("params (B):", asr.param_count() / 1e9, flush=True)
    t0 = time.time()
    with open(out_path, "a", encoding="utf-8", newline="\n") as fout:
        for i in range(0, len(todo), a.bs):
            batch = todo[i : i + a.bs]
            for p, text in zip(batch, asr.transcribe([root / p for p in batch]), strict=True):
                fout.write(json.dumps({"path": p, "text": text}, ensure_ascii=False) + "\n")
            fout.flush()
            if (i // a.bs) % 10 == 0:
                print(f"{i + len(batch)}/{len(todo)}  {time.time() - t0:.0f}s", flush=True)
    print("done", time.time() - t0, "s")


if __name__ == "__main__":
    main()
