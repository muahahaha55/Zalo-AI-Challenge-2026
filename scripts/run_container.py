"""Container entry point (see src/lifemate/container.py). Typical use inside the image:
  python -I -X utf8 scripts/run_container.py                      # reads /data, writes result/submission.jsonl
  python -I -X utf8 scripts/run_container.py --data <folder> --backend null --asr none --out outputs/plumbing.jsonl   # plumbing test"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lifemate.container import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
