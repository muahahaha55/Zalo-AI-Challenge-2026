"""Small JSON Lines helpers (UTF-8, one object per line, blank lines ignored)."""
from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def read_jsonl(path: str | Path) -> list[dict]:
    return list(iter_jsonl(path))


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    n = 0
    with open(path, "w", encoding="utf-8", newline="\n") as f:     # LF on every OS, so files are byte-identical across machines
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n
