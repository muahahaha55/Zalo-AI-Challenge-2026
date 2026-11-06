"""Local pre-upload check for the Kaggle workflow (docs/KAGGLE.md). Reads no file contents under the dataset's app-state folders;
it only looks at names, sizes and line counts of the files listed below.

  .venv/Scripts/python.exe -I -X utf8 scripts/kaggle_preflight.py
Exit code 1 if any hard check fails.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "train+public_test"
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "kaggle_upload" / "lifemate-models"
bad = 0


def check(ok: bool, msg: str, hard: bool = True):
    global bad
    print(("[ok]   " if ok else ("[FAIL] " if hard else "[warn] ")) + msg)
    if not ok and hard:
        bad += 1


def find_bash():
    """A bash that really runs. On Windows `bash` may be the WSL launcher without a distro; Git Bash is the one we want."""
    for cand in [shutil.which("bash"), "C:/Program Files/Git/bin/bash.exe", "C:/Program Files/Git/usr/bin/bash.exe"]:
        if cand and Path(cand).exists():
            r = subprocess.run([cand, "-c", "echo ok"], capture_output=True, text=True)
            if r.returncode == 0 and r.stdout.strip() == "ok":
                return cand
    return None


def size_gb(p: Path) -> float:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e9


def lines(p: Path) -> int:
    with open(p, encoding="utf-8") as f:
        return sum(1 for _ in f)


def main():
    print("== raw data")
    for rel in ["train/train.jsonl", "public_test/public_test.jsonl", "train/audio", "public_test/audio",
                "train/zaic_tools", "train/train_tools_catalog.json", "public_test/public_test_tools_catalog.json"]:
        check((RAW / rel).exists(), f"data/raw/train+public_test/{rel}")
    if RAW.exists():
        print(f"       raw size {size_gb(RAW):.2f} GB")

    print("== processed data")
    for f in ["dev_ids.txt", "sft_train.jsonl", "sft_dev.jsonl"]:
        check((PROC / f).exists(), f"data/processed/{f}")
    if (PROC / "sft_train.jsonl").exists():
        n, d = lines(PROC / "sft_train.jsonl"), lines(PROC / "sft_dev.jsonl")
        check((n, d) == (1390, 155), f"train={n} dev={d} samples (HANDOFF expects 1390 / 155)")
    check(not (PROC / "dev_ids.txt").exists() or lines(PROC / "dev_ids.txt") == 155, "dev_ids.txt has 155 ids")
    for f in sorted(PROC.glob("asr_*smoke*")):
        check(False, f"leftover smoke file {f.name}", hard=False)

    print("== models (kaggle_upload/lifemate-models)")
    mf = MODELS / "manifest.json"
    check(mf.exists(), "manifest.json (run scripts/prepare_models.py)")
    if mf.exists():
        for name, m in json.loads(mf.read_text(encoding="utf-8")).items():
            ok = all((MODELS / name / fn).exists() for fn in m.get("sha256", {})) and bool(m.get("sha256"))
            check(ok and not list((MODELS / name).glob("*.bin")), f"{name}: safetensors only, {m['params_B']:.3f}B params, {m['license']}")
        print(f"       models size {size_gb(MODELS):.2f} GB")

    print("== code")
    bash = find_bash()
    check(bash is not None, "a working bash found (needed for bash -n)", hard=False)
    for sh in sorted((ROOT / "scripts").glob("*.sh")) + sorted((ROOT / ".githooks").glob("*")):
        check(b"\r\n" not in sh.read_bytes(), f"LF line endings: {sh.relative_to(ROOT)}")
        if sh.suffix == ".sh" and bash:
            r = subprocess.run([bash, "-n", sh.relative_to(ROOT).as_posix()], capture_output=True, text=True, cwd=ROOT)
            check(r.returncode == 0, f"bash -n {sh.name}" + (f": {r.stderr.strip()}" if r.returncode else ""))
    for py in sorted((ROOT / "scripts").glob("*.py")) + sorted((ROOT / "src").rglob("*.py")):
        r = subprocess.run([sys.executable, "-m", "py_compile", str(py)], capture_output=True, text=True)
        check(r.returncode == 0, f"py_compile {py.relative_to(ROOT)}")
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "check_compliance.py")], capture_output=True, text=True, cwd=ROOT)
    check(r.returncode == 0, "check_compliance.py: " + (r.stdout.strip().splitlines() or [r.stderr.strip()])[-1])
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], capture_output=True, text=True, cwd=ROOT)
    check(r.returncode == 0, "pytest: " + (r.stdout.strip().splitlines() or ["?"])[-1])

    print("== secrets / hygiene")
    for p in [ROOT / "kaggle_upload", PROC, RAW]:
        hits = [f for f in p.rglob("*") if f.is_file() and f.name in ("kaggle.json", ".env")] if p.exists() else []
        check(not hits, f"no kaggle.json/.env inside {p.relative_to(ROOT)}")
    check((Path.home() / ".kaggle" / "kaggle.json").exists(), "~/.kaggle/kaggle.json present (needed to upload)", hard=False)
    print("\nRESULT:", "FAIL" if bad else "ready to pack and upload (see docs/KAGGLE.md)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
