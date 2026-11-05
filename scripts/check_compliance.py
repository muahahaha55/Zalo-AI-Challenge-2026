"""Static compliance checks for docs/RULES.md.

  python scripts/check_compliance.py            # inference code + all tracked files
  python scripts/check_compliance.py --staged   # staged files only (used by .githooks/pre-commit)

Part A (inference code): no *_environment/state.json access, only validate/call/get_tool_calls/get_state_hash,
one ToolEnv per file, no network libs, no trust_remote_code/pickle.
Part B (repo hygiene): no data/weights/secrets/large files committed.
This is a best-effort lint, not a proof of compliance; see RULES.md section 7.
"""
import argparse
import fnmatch
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Code that runs at inference time (dev-only scripts such as eval.py / verify_metric.py are deliberately excluded:
# they use get_state on train, which is allowed in development).
INFERENCE_GLOBS = ["src/lifemate/*.py", "scripts/run_infer.py", "docker/*.py", "docker/**/*.py", "inference/*.py"]
INFERENCE_EXCLUDE = ["src/lifemate/metric.py", "src/lifemate/__init__.py"]

FORBIDDEN = [  # (regex, why)
    (r"_environment", "reads *_environment/ (forbidden)"),
    (r"state\.json", "reads state.json (forbidden)"),
    (r"\bget_state\b(?!_hash)", "env.get_state is train-only"),
    (r"\.states\b", "env.states is not allowed at inference"),
    (r"\bExecutor\b", "Executor is not allowed at inference"),
    (r"\bSimulator\b", "Simulator is not allowed at inference"),
    (r"\b(deepcopy|copy\.copy)\b", "cloning an env is not allowed"),
    (r"^\s*(import|from)\s+(requests|urllib|httpx|aiohttp|http\.client|socket|openai|anthropic)\b", "network library in inference code"),
    (r"trust_remote_code\s*=\s*True", "trust_remote_code=True is forbidden"),
    (r"\bpickle\b", "pickle is forbidden"),
]
SECRET_PATTERNS = [
    (r"hf_[A-Za-z0-9]{30,}", "Hugging Face token"),
    (r"sk-[A-Za-z0-9_-]{20,}", "API key (sk-...)"),
    (r"AKIA[0-9A-Z]{16}", "AWS access key"),
    (r"-----BEGIN [A-Z ]*PRIVATE KEY-----", "private key"),
    (r"(?i)(api[_-]?key|secret|token|password)\s*[:=]\s*['\"][A-Za-z0-9_\-]{16,}['\"]", "hard-coded credential"),
]
BANNED_PATHS = ["data/*", "*_environment/*", "*.zip", "*.wav", "*.safetensors", "*.bin", "*.pt", "*.pth", "*.gguf", "*.ckpt",
                ".env", ".env.*", "*.pem", "id_rsa*", "id_ed25519*", ".venv/*"]
MAX_BYTES = 5 * 1024 * 1024


def git_files(staged: bool) -> list[str]:
    cmd = ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"] if staged else ["git", "ls-files"]
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8").stdout
    return [l for l in out.splitlines() if l]


def strip_comments_and_docstrings(src: str) -> str:
    """Keep real code only, so prose such as 'never read state.json' in a docstring does not trip the lint."""
    import io
    import tokenize
    out = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type in (tokenize.COMMENT,):
                continue
            if tok.type == tokenize.STRING and tok.string.lstrip("rbfRBF").startswith(('"""', "'''")):
                out.append('""')
                continue
            out.append(tok.string if tok.type != tokenize.NEWLINE else "\n")
    except tokenize.TokenError:
        return src
    return " ".join(out)


def check_inference(problems: list[str]):
    files = set()
    for g in INFERENCE_GLOBS:
        files.update(p for p in ROOT.glob(g) if p.is_file())
    for p in sorted(files):
        rel = p.relative_to(ROOT).as_posix()
        if rel in INFERENCE_EXCLUDE:
            continue
        raw = p.read_text(encoding="utf-8")
        code = raw  # patterns are line-oriented; check raw lines but skip pure comment lines
        for n, line in enumerate(code.splitlines(), 1):
            s = line.strip()
            if s.startswith("#"):
                continue
            for rx, why in FORBIDDEN:
                if re.search(rx, line) and not _in_docstring(raw, n):
                    problems.append(f"{rel}:{n}: {why}: {s[:80]}")
        if len(re.findall(r"\bToolEnv\s*\(",strip_comments_and_docstrings(raw))) > 1:
            problems.append(f"{rel}: more than one ToolEnv( construction (exactly one env per sample)")


_doc_cache: dict[int, set] = {}


def _in_docstring(raw: str, lineno: int) -> bool:
    key = hash(raw)
    if key not in _doc_cache:
        import ast
        lines = set()
        try:
            for node in ast.walk(ast.parse(raw)):
                if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                    b = node.body
                    if b and isinstance(b[0], ast.Expr) and isinstance(getattr(b[0], "value", None), ast.Constant) and isinstance(b[0].value.value, str):
                        lines.update(range(b[0].lineno, b[0].end_lineno + 1))
        except SyntaxError:
            pass
        _doc_cache[key] = lines
    return lineno in _doc_cache[key]


def check_repo(files: list[str], problems: list[str]):
    for rel in files:
        if any(fnmatch.fnmatch(rel, pat) for pat in BANNED_PATHS):
            problems.append(f"{rel}: path is not allowed in git (data / weights / secrets / env)")
            continue
        p = ROOT / rel
        if not p.is_file():
            continue
        if p.stat().st_size > MAX_BYTES:
            problems.append(f"{rel}: {p.stat().st_size / 1e6:.1f} MB > 5 MB limit")
            continue
        if rel == "scripts/check_compliance.py":   # holds the patterns themselves
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for rx, why in SECRET_PATTERNS:
            if re.search(rx, text):
                problems.append(f"{rel}: possible secret ({why})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--staged", action="store_true")
    a = ap.parse_args()
    problems: list[str] = []
    check_inference(problems)
    check_repo(git_files(a.staged), problems)
    if problems:
        print("COMPLIANCE CHECK FAILED:")
        for x in problems:
            print("  -", x)
        sys.exit(1)
    print("compliance check passed")


if __name__ == "__main__":
    main()
