import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lifemate import manifest  # noqa: E402


def test_sha256_matches_known_value(tmp_path):
    p = tmp_path / "a.txt"
    p.write_bytes(b"abc")
    assert manifest.sha256_file(p) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_manifest_lists_inputs_outputs_rows_and_is_reproducible(tmp_path):
    src, out = tmp_path / "in.json", tmp_path / "out.jsonl"
    src.write_text("{}", encoding="utf-8")
    out.write_text('{"id": 1}\n\n{"id": 2}\n', encoding="utf-8")
    missing = tmp_path / "optional-input-that-does-not-exist"
    path = tmp_path / "M.json"
    m1 = manifest.write_manifest(path, "unit", {"seed": 1}, [src, missing], [out])
    assert [e["path"].endswith("in.json") for e in m1["inputs"]] == [True]            # absent inputs are skipped, not an error
    assert m1["outputs"][0]["rows"] == 2 and m1["args"] == {"seed": 1} and m1["tool"] == "unit"
    first = path.read_bytes()
    manifest.write_manifest(path, "unit", {"seed": 1}, [src, missing], [out])
    assert path.read_bytes() == first                                                  # no timestamps: same input, same bytes
    assert json.loads(first)["outputs"][0]["sha256"] == manifest.sha256_file(out)


def test_git_state_never_raises(tmp_path):
    state = manifest.git_state(tmp_path)                                               # not a repository
    assert set(state) == {"commit", "dirty"}
