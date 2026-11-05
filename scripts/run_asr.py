"""Transcribe every wav of a split (final request + history user turns) with a Whisper-family model.
Resumable; greedy; fp16. Output: data/processed/asr_<tag>_<split>.jsonl  {"path": "audio/<id>/<id>.wav", "text": ...}
Usage: python -I -X utf8 scripts/run_asr.py --model vinai/PhoWhisper-large --tag phowl --split train [--limit 40] [--bs 8]"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "train+public_test"
OUT = ROOT / "data" / "processed"


def degenerate(text: str) -> bool:
    w = text.split()
    g = [" ".join(w[i : i + 3]) for i in range(len(w) - 2)]
    return bool(g) and max(g.count(x) for x in set(g)) >= 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", help="file with sample ids; only transcribe audio of those samples")
    ap.add_argument("--bs", type=int, default=8)
    a = ap.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)

    audio_root = DATA / a.split / "audio"
    paths = sorted(p.relative_to(DATA / a.split).as_posix() for p in audio_root.rglob("*.wav"))
    if a.ids:
        keep = set(Path(a.ids).read_text().split())
        paths = [p for p in paths if p.split("/")[1] in keep]
    if a.limit:
        paths = paths[: a.limit]
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / f"asr_{a.tag}_{a.split}.jsonl"
    done = set()
    if out_path.exists():
        done = {json.loads(l)["path"] for l in open(out_path, encoding="utf-8")}
    todo = [p for p in paths if p not in done]
    print(f"{len(paths)} wavs, {len(done)} done, {len(todo)} to do", flush=True)
    if not todo:
        return

    proc = WhisperProcessor.from_pretrained(a.model)
    model = WhisperForConditionalGeneration.from_pretrained(a.model, torch_dtype=torch.float16).cuda().eval()
    print("params (B):", sum(p.numel() for p in model.parameters()) / 1e9, flush=True)
    t0 = time.time()
    with open(out_path, "a", encoding="utf-8") as fout:
        for i in range(0, len(todo), a.bs):
            batch = todo[i : i + a.bs]
            wavs = []
            for p in batch:
                x, sr = sf.read(DATA / a.split / p, dtype="float32")
                assert sr == 16000, (p, sr)
                wavs.append(x if x.ndim == 1 else x.mean(1))
            feats = proc.feature_extractor(wavs, sampling_rate=16000, return_tensors="pt").input_features.cuda().half()
            with torch.no_grad():
                ids = model.generate(feats, language="vi", task="transcribe", num_beams=1, do_sample=False, max_new_tokens=160)
            texts = proc.batch_decode(ids, skip_special_tokens=True)
            for k, t in enumerate(texts):  # guard against Whisper repetition loops: redo that item with an n-gram block
                if degenerate(t):
                    with torch.no_grad():
                        ids1 = model.generate(feats[k : k + 1], language="vi", task="transcribe", num_beams=1, do_sample=False,
                                              max_new_tokens=100, no_repeat_ngram_size=4)
                    texts[k] = proc.batch_decode(ids1, skip_special_tokens=True)[0]
            for p, t in zip(batch, texts):
                fout.write(json.dumps({"path": p, "text": t.strip()}, ensure_ascii=False) + "\n")
            fout.flush()
            if (i // a.bs) % 10 == 0:
                print(f"{i + len(batch)}/{len(todo)}  {time.time() - t0:.0f}s", flush=True)
    print("done", time.time() - t0, "s")


if __name__ == "__main__":
    main()
