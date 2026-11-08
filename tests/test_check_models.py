import hashlib
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("check_models", ROOT / "scripts" / "check_models.py")
cm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cm)


def make(tmp_path: Path, data: bytes = b"weights"):
    folder = tmp_path / "asr"
    folder.mkdir()
    (folder / "config.json").write_text("{}")
    (folder / "model.safetensors").write_bytes(data)
    return folder, {"required": ["config.json"], "weights": {"model.safetensors": hashlib.sha256(b"weights").hexdigest()}}


def test_matching_folder_has_no_problems(tmp_path):
    folder, spec = make(tmp_path)
    assert cm.check_role("asr", folder, spec, skip_hash=False) == []


def test_wrong_hash_missing_file_pickle_and_extra_weights_are_reported(tmp_path):
    folder, spec = make(tmp_path, data=b"other")
    (folder / "config.json").unlink()
    (folder / "extra.bin").write_bytes(b"x")
    (folder / "second.safetensors").write_bytes(b"y")
    text = " | ".join(cm.check_role("asr", folder, spec, skip_hash=False))
    for needle in ("config.json is missing", "wrong sha256", "pickle-based file extra.bin", "unexpected weight file second.safetensors"):
        assert needle in text


def test_skip_hash_ignores_content_but_not_presence(tmp_path):
    folder, spec = make(tmp_path, data=b"other")
    assert cm.check_role("asr", folder, spec, skip_hash=True) == []
    assert cm.check_role("asr", tmp_path / "nope", spec, skip_hash=True) == [f"asr: folder {tmp_path / 'nope'} is missing"]
