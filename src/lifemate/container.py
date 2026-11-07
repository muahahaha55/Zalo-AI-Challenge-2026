"""Entry point for the final container: read the mounted split, run ASR + agent per sample, write result/submission.jsonl.

The organisers mount a folder "with the same structure as public_test" at /data. Two layouts are accepted and detected:
  flat   : /data IS the split folder          (/data/audio/, /data/<split>.jsonl, /data/zaic_tools/, ...)
  nested : /data holds the split folder       (/data/<split>/audio/, /data/<split>/<split>.jsonl, ...)
The official ToolEnv wants `root/<split>/...`, so for the flat layout a link `<stage>/<split> -> /data` is created.
Which layout the organisers really use is unconfirmed; use --split to disambiguate if detection reports ambiguity.

Offline by construction: Hugging Face offline flags are forced before any model is loaded, and no network library is used.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .agent import AgentConfig
from .determinism import seed_everything
from .jsonl import ResumableJsonlWriter
from .pipeline import HFBackend, LiveTranscripts, NullBackend, SplitRunner, StubTranscripts, Transcripts

IGNORED_JSONL = {"sample_submission"}
PREFERRED_SPLIT = "private_test"


class LayoutError(RuntimeError):
    """The mounted folder does not look like a split, or it is ambiguous."""


@dataclass(frozen=True)
class Layout:
    split: str
    data_root: Path        # the folder that holds the split folder (what ToolEnv calls `root`)
    flat: bool             # True when the mounted folder is itself the split folder


def find_split(data_dir: Path, split: str | None = None) -> tuple[str, bool]:
    """Return (split name, flat) by looking at the mounted folder only (no side effects)."""
    if not data_dir.is_dir():
        raise LayoutError(f"{data_dir} is not a folder")
    nested = sorted(d.name for d in data_dir.iterdir() if d.is_dir() and (d / f"{d.name}.jsonl").is_file())
    flat = sorted(p.stem for p in data_dir.glob("*.jsonl") if p.stem not in IGNORED_JSONL) if (data_dir / "audio").is_dir() else []
    found = [(name, False) for name in nested] + [(name, True) for name in flat]
    if split:
        found = [c for c in found if c[0] == split]
    if len(found) == 1:
        return found[0]
    preferred = [c for c in found if c[0] == PREFERRED_SPLIT]
    if len(preferred) == 1:
        return preferred[0]
    raise LayoutError(f"cannot tell the layout of {data_dir}: nested splits {nested}, flat splits {flat}"
                      + (f", requested split {split!r}" if split else "") + "; pass --split")


def link_directory(link: Path, target: Path) -> None:
    """Make `link` point at `target` (symlink; on Windows without symlink rights, a directory junction)."""
    if link.is_symlink():
        link.unlink()
    elif link.exists():
        raise LayoutError(f"{link} exists and is not a link")
    try:
        os.symlink(target, link, target_is_directory=True)
    except OSError:
        if os.name != "nt":
            raise
        subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], check=True, capture_output=True)


def resolve_layout(data_dir: str | Path, split: str | None = None, stage_dir: str | Path | None = None) -> Layout:
    data_dir = Path(data_dir).resolve()
    name, flat = find_split(data_dir, split)
    if not flat:
        return Layout(name, data_dir, False)
    stage = Path(stage_dir) if stage_dir else Path(tempfile.mkdtemp(prefix="lifemate_root_"))
    stage.mkdir(parents=True, exist_ok=True)
    link_directory(stage / name, data_dir)
    return Layout(name, stage, True)


def force_offline() -> None:
    """No network at inference (RULES 2.4): make Hugging Face libraries refuse to go online and skip hub kernels."""
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[key] = "1"
    os.environ["USE_HUB_KERNELS"] = "NO"


def build_transcripts(spec: str, split_dir: Path) -> Transcripts:
    """spec: a Whisper model folder (live recognition), 'clean' (gold text, train only) or 'none' (empty text, plumbing tests)."""
    if spec == "clean":
        return Transcripts.clean()
    if spec == "none":
        return StubTranscripts()
    from .asr import WhisperTranscriber
    return LiveTranscripts(WhisperTranscriber(spec), split_dir)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    env = os.environ.get
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=env("LIFEMATE_DATA_DIR", "/data"), help="mounted folder (default /data)")
    ap.add_argument("--out", default=env("LIFEMATE_OUT", "result/submission.jsonl"))
    ap.add_argument("--split", default=None, help="split name; detected when omitted")
    ap.add_argument("--backend", choices=["hf", "null"], default="hf", help="null: answer END everywhere (plumbing test)")
    ap.add_argument("--llm", default=env("LIFEMATE_LLM", "/models/llm"))
    ap.add_argument("--adapter", default=env("LIFEMATE_ADAPTER", "/models/adapter"), help="LoRA folder, or 'none'")
    ap.add_argument("--asr", default=env("LIFEMATE_ASR", "/models/asr"), help="Whisper folder, 'clean' or 'none'")
    ap.add_argument("--time-budget", type=float, default=25.0, help="seconds per sample incl. recognition (contest limit: 30)")
    ap.add_argument("--no-rules", action="store_true")
    ap.add_argument("--resume", action="store_true", help="continue an interrupted run from <out>.part")
    ap.add_argument("--limit", type=int, default=0, help="first N samples only (testing)")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    force_offline()
    seed_everything(0)
    layout = resolve_layout(args.data, args.split)
    split_dir = layout.data_root / layout.split
    print(f"split {layout.split!r} ({'flat' if layout.flat else 'nested'} layout) at {args.data}", flush=True)

    backend = NullBackend()
    if args.backend == "hf":
        backend = HFBackend(args.llm, None if args.adapter == "none" else args.adapter)
    runner = SplitRunner(layout.split, AgentConfig(use_rules=not args.no_rules, time_budget_s=args.time_budget or None),
                         build_transcripts(args.asr, split_dir), backend=backend, data_root=layout.data_root)
    items = runner.items(limit=args.limit)

    def progress(n: int, total: int, avg: float) -> None:
        if (n - 1) % 20 == 0:
            print(f"{n}/{total}  avg {avg:.2f}s", flush=True)

    with ResumableJsonlWriter(args.out, resume=args.resume) as writer:
        todo = [i for i in items if i["id"] not in writer.done]
        for row in runner.run(todo, progress):
            writer.write(row)
    print("done", runner.stats.summary(), flush=True)
    return 0
