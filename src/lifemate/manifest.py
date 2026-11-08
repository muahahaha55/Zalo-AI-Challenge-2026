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


LIBRARIES = ("torch", "transformers", "peft", "accelerate", "tokenizers", "safetensors")


def library_versions(names: Iterable[str] = LIBRARIES) -> dict[str, str]:
    from importlib import metadata
    out = {}
    for name in names:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return out


def model_entry(folder: str | Path | None) -> dict | None:
    """Weight files (*.safetensors) of a model folder with checksums; None when no folder is given or it is not a folder."""
    if not folder or not Path(folder).is_dir():
        return None
    files = sorted(Path(folder).rglob("*.safetensors"))
    return {"path": Path(folder).as_posix(),
            "weights": [{"name": p.relative_to(folder).as_posix(), "sha256": sha256_file(p), "bytes": p.stat().st_size} for p in files]}


def build_run_manifest(tool: str, args: dict, models: dict[str, str | Path | None], inputs: Iterable[str | Path],
                       output: str | Path, stats: dict) -> dict:
    """Provenance of an inference output: code version, libraries, model weights, input files, arguments, result checksum.
    Answers "which adapter and which transcripts produced this submission?". `stats` (timings, stop reasons) is the only
    part that is not reproducible byte for byte."""
    manifest = build_manifest(tool, args, inputs, [output])
    manifest["libraries"] = library_versions()
    manifest["models"] = {role: model_entry(path) for role, path in models.items()}
    manifest["stats"] = stats
    return manifest


def write_run_manifest(path: str | Path, *build_args, **build_kwargs) -> dict:
    manifest = build_run_manifest(*build_args, **build_kwargs)
    Path(path).write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return manifest
