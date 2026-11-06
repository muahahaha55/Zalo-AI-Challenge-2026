"""Where things live. One place instead of path constants repeated in every script.

LIFEMATE_DATA_ROOT      folder that holds the split folders (train/, public_test/, ...). Default: data/raw/train+public_test.
                        In the final container this is where the organisers' data is mounted.
LIFEMATE_PROCESSED_DIR  intermediate files (ASR transcripts, SFT data). Default: data/processed.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def data_root() -> Path:
    return Path(os.environ.get("LIFEMATE_DATA_ROOT") or ROOT / "data" / "raw" / "train+public_test")


def split_dir(split: str) -> Path:
    return data_root() / split


def processed_dir() -> Path:
    return Path(os.environ.get("LIFEMATE_PROCESSED_DIR") or ROOT / "data" / "processed")
