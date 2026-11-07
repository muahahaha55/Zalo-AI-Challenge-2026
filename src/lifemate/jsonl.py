"""Small JSON Lines helpers (UTF-8, LF, one object per line, blank lines ignored)."""
from __future__ import annotations

import json
import os
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


class ResumableJsonlWriter:
    """Write rows one by one, safely.

    Rows go to `<path>.part` (flushed after each row); only when the `with` block ends without an exception is the file
    renamed to `<path>`, so the final path never holds a half-written result. With `resume=True` an existing `.part` is
    continued: its complete rows are kept (`done` lists their `key` values), and a truncated last line left by a crash is cut off.
    """

    def __init__(self, path: str | Path, resume: bool = False, key: str = "id"):
        self.path = Path(path)
        self.part = self.path.with_name(self.path.name + ".part")
        self.key = key
        self.done: set[str] = set()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        keep = 0
        if resume and self.part.exists():
            with open(self.part, "rb") as f:
                for line in f:
                    try:
                        row = json.loads(line)
                    except ValueError:
                        break                                   # truncated or corrupt tail: everything from here is dropped
                    if not line.endswith(b"\n"):
                        break
                    self.done.add(row[self.key])
                    keep += len(line)
            with open(self.part, "r+b") as f:
                f.truncate(keep)
        self._file = open(self.part, "ab" if keep else "wb")    # noqa: SIM115 - closed in __exit__

    def write(self, row: dict) -> None:
        self._file.write((json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8"))
        self._file.flush()

    def __enter__(self) -> ResumableJsonlWriter:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._file.close()
        if exc_type is None:
            os.replace(self.part, self.path)
