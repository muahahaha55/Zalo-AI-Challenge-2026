"""Pre-flight check of the mounted data (no model, no writes): which split is in /data, how many samples, missing audio.
Fails early with a clear message instead of failing inside the model run."""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src" if (HERE / "src").is_dir() else HERE.parent / "src"))   # /code/src in the image, ../src in the repo
from lifemate.container import find_split  # noqa: E402


def main() -> int:
    data = Path(os.environ.get("LIFEMATE_DATA_DIR", "/data"))
    print("preprocessing...", flush=True)
    split, flat = find_split(data, os.environ.get("LIFEMATE_SPLIT") or None)
    split_dir = data if flat else data / split
    items = [json.loads(line) for line in open(split_dir / f"{split}.jsonl", encoding="utf-8") if line.strip()]
    wavs = [it["audio"] for it in items] + [h["audio"] for it in items for h in it.get("history", []) if h.get("role") == "user" and h.get("audio")]
    missing = [w for w in wavs if not (split_dir / w).is_file()]
    for need in (f"{split}_tools_catalog.json", "zaic_tools"):
        if not (split_dir / need).exists():
            print(f"preprocessing: missing {split_dir / need}", flush=True)
            return 1
    print(f"preprocessing: split {split!r}, {len(items)} samples, {len(wavs)} audio files, missing audio: {len(missing)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
