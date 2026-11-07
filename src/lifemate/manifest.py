"""Provenance manifests: which command, code version and inputs produced a file, with checksums.

A manifest answers "which data was this adapter trained on?" months later. It records no timestamps, so the same inputs and
arguments give a byte-identical manifest.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

from .paths import ROOT

CHUNK = 1 << 20


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def git_state(cwd: Path = ROOT) -> dict:
    """Current commit and whether tracked files have uncommitted changes; empty values when git is unavailable."""
    def run(*args: str) -> str | None:
        try:
            done = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=20, check=True)
            return done.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain", "--untracked-files=no")
    return {"commit": commit, "dirty": None if status is None else bool(status)}


def file_entry(path: str | Path, count_rows: bool = False) -> dict:
    p = Path(path)
    entry = {"path": p.as_posix(), "sha256": sha256_file(p), "bytes": p.stat().st_size}
    if count_rows:
        with open(p, "rb") as f:
            entry["rows"] = sum(1 for line in f if line.strip())
    return entry


def build_manifest(tool: str, args: dict, inputs: Iterable[str | Path], outputs: Iterable[str | Path]) -> dict:
    return {
        "tool": tool,
        "args": args,
        "git": git_state(),
        "python": sys.version.split()[0],
        "inputs": [file_entry(p) for p in inputs if Path(p).is_file()],
        "outputs": [file_entry(p, count_rows=str(p).endswith(".jsonl")) for p in outputs],
    }


def write_manifest(path: str | Path, tool: str, args: dict, inputs: Iterable[str | Path], outputs: Iterable[str | Path]) -> dict:
    manifest = build_manifest(tool, args, inputs, outputs)
    Path(path).write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return manifest
