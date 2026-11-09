"""End-to-end check of the final solution on a machine with a GPU (and Docker). Runs the container N times on a split,
then reports: did every run finish, are the outputs byte-identical, how long did samples take, do the submitted hashes
replay (verify_submission.py), and how many lines differ from a reference submission.

  # the real image (Docker + NVIDIA container toolkit):
  python -I -X utf8 scripts/verify_image.py --image lifemate --data-root /path/to/folder-that-holds-public_test \\
      --runs 3 --out-dir check_run [--ref outputs/sub_public_v1.jsonl] [--limit 20]
  # the same code path without Docker (e.g. a Kaggle notebook, weights in /kaggle/...):
  python -I -X utf8 scripts/verify_image.py --no-docker --data-root ... --runs 3 --out-dir check_run \\
      -- --llm /path/llm --adapter /path/adapter --asr /path/asr

Everything after `--` is passed to the container entry point (scripts/run_container.py). The data folder is mounted read-only
and the network is disabled (`--network none`). Writes <out-dir>/report.json and prints a summary; exit code 1 on any problem.
It reads no *_environment/ folder itself (verify_submission.py replays calls through ToolEnv)."""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def gpu_info() -> str:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30).stdout.strip()
        return out.replace("\n", " | ")
    except (OSError, subprocess.SubprocessError):
        return "nvidia-smi not available"


def parse_done(text: str) -> dict:
    """Numbers from the container's last line: 'done N samples; mean Xs p95 Ys max Zs; stop reasons {...}'."""
    m = re.search(r"done (\d+) samples; mean ([\d.]+)s p95 ([\d.]+)s max ([\d.]+)s; stop reasons (\{.*\})", text)
    if not m:
        return {}
    return {"samples": int(m[1]), "mean_s": float(m[2]), "p95_s": float(m[3]), "max_s": float(m[4]), "stop_reasons": m[5]}


def run_once(args, run_dir: Path, extra: list[str]) -> tuple[int, str, float]:
    run_dir.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    if args.no_docker:
        cmd = [sys.executable, "-I", "-X", "utf8", str(ROOT / "scripts" / "run_container.py"), "--data", str(Path(args.data_root) / args.split),
               "--out", str(run_dir / "submission.jsonl"), "--split", args.split, *extra]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    else:
        cmd = ["docker", "run", "--rm", "--gpus", "all", "--network", "none",
               "-v", f"{Path(args.data_root).resolve() / args.split}:/data:ro", "-v", f"{run_dir.resolve()}:/app/result", args.image, *extra]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return proc.returncode, proc.stdout + proc.stderr, time.monotonic() - start


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="lifemate")
    ap.add_argument("--data-root", required=True, help="folder that holds the split folder (e.g. the one containing public_test/)")
    ap.add_argument("--split", default="public_test")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--out-dir", default="check_run")
    ap.add_argument("--ref", default=None, help="a reference submission to compare with (e.g. outputs/sub_public_v1.jsonl)")
    ap.add_argument("--limit", type=int, default=0, help="first N samples only (quick test)")
    ap.add_argument("--no-docker", action="store_true", help="run scripts/run_container.py directly instead of the image")
    ap.add_argument("--no-verify", action="store_true", help="skip verify_submission.py (needs the split's zaic_tools)")
    args, extra = ap.parse_known_args()
    extra = [x for x in extra if x != "--"]
    out_dir = Path(args.out_dir)
    problems: list[str] = []
    report: dict = {"gpu": gpu_info(), "image": None if args.no_docker else args.image, "runs": []}
    print("GPU:", report["gpu"], flush=True)

    outputs = []
    for r in range(1, args.runs + 1):
        run_dir = out_dir / f"run{r}"
        code, log, seconds = run_once(args, run_dir, extra)
        sub = run_dir / "submission.jsonl"
        entry = {"run": r, "exit_code": code, "wall_s": round(seconds, 1), **parse_done(log)}
        if code != 0 or not sub.is_file():
            problems.append(f"run {r}: exit code {code}, output file {'present' if sub.is_file() else 'MISSING'}")
            (run_dir / "log.txt").write_text(log[-4000:], encoding="utf-8")
            print(log[-1500:])
        else:
            entry["sha256"] = sha256(sub)
            entry["rows"] = sum(1 for _ in open(sub, "rb"))
            outputs.append(sub)
        report["runs"].append(entry)
        print(f"run {r}: {entry}", flush=True)

    if len(outputs) >= 2:
        same = len({sha256(o) for o in outputs}) == 1
        report["byte_identical"] = same
        print("byte-identical across runs:", same)
        if not same:
            problems.append("runs are not byte-identical (RULES 2.5)")

    if outputs and not args.no_verify:
        env = {**os.environ, "LIFEMATE_DATA_ROOT": str(Path(args.data_root).resolve())}
        cmd = [sys.executable, "-I", "-X", "utf8", str(ROOT / "scripts" / "verify_submission.py"), "--split", args.split, "--sub", str(outputs[0])]
        if args.limit:
            cmd.append("--allow-partial")
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
        report["verify_submission"] = {"exit_code": proc.returncode, "tail": proc.stdout.strip().splitlines()[-3:]}
        print("verify_submission:", "OK" if proc.returncode == 0 else "FAILED", *report["verify_submission"]["tail"], sep="\n  ")
        if proc.returncode != 0:
            problems.append("verify_submission.py failed")

    if outputs and args.ref:
        ref = {json.loads(line)["id"]: line.strip() for line in open(args.ref, encoding="utf-8") if line.strip()}
        got = {json.loads(line)["id"]: line.strip() for line in open(outputs[0], encoding="utf-8") if line.strip()}
        differ = sorted(i for i in got if i in ref and got[i] != ref[i])
        report["vs_reference"] = {"compared": sum(1 for i in got if i in ref), "different": len(differ), "first_ids": differ[:10]}
        print(f"vs reference: {report['vs_reference']}")

    slow = [e for e in report["runs"] if e.get("max_s", 0) > 30]
    if slow:
        problems.append("a sample took more than 30 s (contest limit)")
    report["problems"] = problems
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print("\nPROBLEMS:" if problems else "\nALL CHECKS PASSED", *problems, sep="\n  ")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
