"""Count the parameters of the system that will be shipped and check the contest limits (LLM < 5B, whole system < 6B).
  python -I -X utf8 scripts/count_params.py --llm models/llm --adapter models/adapter --asr models/asr [--extra models/other ...]
Each argument is a .safetensors file or a folder of them. The adapter counts as part of the LLM (it is applied at run time).
Exit code 1 when a limit is reached."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lifemate.params import check_limits, count_path  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", required=True, help="base LLM weights")
    ap.add_argument("--adapter", help="LoRA adapter (counted with the LLM)")
    ap.add_argument("--asr", required=True)
    ap.add_argument("--extra", nargs="*", default=[], help="any other model in the system (classifier, second ASR, ...)")
    a = ap.parse_args()

    llm = count_path(a.llm)
    adapter = count_path(a.adapter) if a.adapter else 0
    asr = count_path(a.asr)
    extra = [count_path(p) for p in a.extra]
    llm_total = llm + adapter
    total = llm_total + asr + sum(extra)
    print(f"LLM base    {llm / 1e9:8.4f}B")
    print(f"LoRA adapter{adapter / 1e9:8.4f}B   (applied at run time, counted with the LLM)")
    print(f"LLM total   {llm_total / 1e9:8.4f}B   limit < 5B")
    print(f"ASR         {asr / 1e9:8.4f}B")
    for p, n in zip(a.extra, extra, strict=True):
        print(f"extra {p}  {n / 1e9:.4f}B")
    print(f"SYSTEM      {total / 1e9:8.4f}B   limit < 6B   headroom {(6e9 - total) / 1e9:.4f}B")
    problems = check_limits(llm_total, total)
    for p in problems:
        print("LIMIT REACHED:", p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
