"""Check a models/ folder against models.lock.json before the final `docker build` (so the image cannot silently ship a
wrong, partial or unexpected model).
  python -I -X utf8 scripts/check_models.py [--models models] [--lock models.lock.json] [--skip-hash]
For each of models/llm, models/adapter, models/asr: every required file exists, every locked weight file exists with the locked
sha256, no weight file outside the lock, no pickle-based file (.bin/.pt/.pth/.pkl/.ckpt). Then the parameter limits are
checked with lifemate.params. Exit code 1 on any problem. Reads no competition data."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lifemate.manifest import sha256_file  # noqa: E402
from lifemate.params import check_limits, count_path  # noqa: E402

PICKLE_SUFFIXES = {".bin", ".pt", ".pth", ".pkl", ".pickle", ".ckpt"}


def check_role(role: str, folder: Path, spec: dict, skip_hash: bool) -> list[str]:
    problems = []
    if not folder.is_dir():
        return [f"{role}: folder {folder} is missing"]
    for name in spec["required"]:
        if not (folder / name).is_file():
            problems.append(f"{role}: required file {name} is missing")
    for name, want in spec["weights"].items():
        path = folder / name
        if not path.is_file():
            problems.append(f"{role}: weight file {name} is missing")
        elif not skip_hash and sha256_file(path) != want:
            problems.append(f"{role}: {name} has the wrong sha256 (not the model that was measured)")
    for path in folder.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix in PICKLE_SUFFIXES:
            problems.append(f"{role}: pickle-based file {path.name} (RULES 3.2)")
        if path.suffix == ".safetensors" and path.name not in spec["weights"]:
            problems.append(f"{role}: unexpected weight file {path.name} (not in models.lock.json)")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=str(ROOT / "models"))
    ap.add_argument("--lock", default=str(ROOT / "models.lock.json"))
    ap.add_argument("--skip-hash", action="store_true", help="only check files and parameters (fast; for a dry run)")
    a = ap.parse_args()
    lock = json.loads(Path(a.lock).read_text(encoding="utf-8"))
    models = Path(a.models)
    problems: list[str] = []
    for role in ("llm", "adapter", "asr"):
        problems += check_role(role, models / role, lock[role], a.skip_hash)
    if not problems:
        llm, adapter, asr = (count_path(models / r) for r in ("llm", "adapter", "asr"))
        print(f"parameters: LLM {llm / 1e9:.4f}B + adapter {adapter / 1e9:.4f}B, ASR {asr / 1e9:.4f}B, system {(llm + adapter + asr) / 1e9:.4f}B")
        problems += check_limits(llm + adapter, llm + adapter + asr)
    for p in problems:
        print("PROBLEM:", p)
    print("models OK" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
