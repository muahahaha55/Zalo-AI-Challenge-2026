"""Exact parameter counts from the headers of .safetensors files (no model is loaded, no GPU needed).

The contest limits the whole system (< 6B) and the LLM (< 5B). Counting from the files that are actually shipped avoids trusting
model cards, and it includes a LoRA adapter that is applied at run time instead of being merged into the base weights.
Note: weights tied to another tensor (e.g. an LM head sharing the embedding matrix) are stored once and counted once, which
matches `sum(p.numel() for p in model.parameters())`.
"""
from __future__ import annotations

import json
import math
import struct
from pathlib import Path

LLM_LIMIT = 5_000_000_000
TOTAL_LIMIT = 6_000_000_000


def count_file(path: str | Path) -> int:
    """Number of elements of all tensors in one .safetensors file, read from its header only."""
    with open(path, "rb") as f:
        (size,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(size))
    return sum(math.prod(t["shape"]) for name, t in header.items() if name != "__metadata__")


def count_path(path: str | Path) -> int:
    """A .safetensors file, or every .safetensors file directly inside a folder."""
    p = Path(path)
    files = [p] if p.is_file() else sorted(p.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no .safetensors files in {p}")
    return sum(count_file(f) for f in files)


def check_limits(llm: int, total: int) -> list[str]:
    problems = []
    if llm >= LLM_LIMIT:
        problems.append(f"LLM has {llm / 1e9:.4f}B parameters, the limit is below 5B")
    if total >= TOTAL_LIMIT:
        problems.append(f"system has {total / 1e9:.4f}B parameters, the limit is below 6B")
    return problems
