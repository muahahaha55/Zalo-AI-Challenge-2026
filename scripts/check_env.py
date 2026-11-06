"""Sanity-check the dev environment (no access to *_environment/)."""
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "train+public_test"

for m in ["torch", "transformers", "peft", "soundfile", "librosa", "jiwer", "jsonschema", "numpy"]:
    try:
        mod = importlib.import_module(m)
        print(f"ok   {m} {getattr(mod, '__version__', '')}")
    except Exception as e:
        print(f"FAIL {m}: {e}")

import torch

print("cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")

# zaic_tools of each split must be importable; only the official API is used.
for split in ["train", "public_test"]:
    sys.path.insert(0, str(DATA / split))
    for k in [k for k in sys.modules if k == "zaic_tools" or k.startswith("zaic_tools.")]:
        del sys.modules[k]
    import json

    from zaic_tools import ToolEnv
    with open(DATA / split / f"{split}.jsonl", encoding="utf-8") as f:
        sid = json.loads(f.readline())["id"]
    env = ToolEnv(split, sid, root=str(DATA))
    print(split, "ToolEnv ok, hash:", env.get_state_hash())
    sys.path.pop(0)
