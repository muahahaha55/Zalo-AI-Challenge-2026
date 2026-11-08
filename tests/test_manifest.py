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


def test_run_manifest_records_weights_inputs_output_and_stats(tmp_path):
    import json

    from lifemate.manifest import model_entry, write_run_manifest
    model = tmp_path / "adapter"
    model.mkdir()
    (model / "adapter_model.safetensors").write_bytes(b"abc")
    (model / "config.json").write_text("{}")
    data = tmp_path / "in.jsonl"
    data.write_text('{"id": 1}\n')
    out = tmp_path / "out.jsonl"
    out.write_text('{"id": 1}\n{"id": 2}\n')
    path = tmp_path / "m.json"
    write_run_manifest(path, "t", {"a": 1}, {"llm": None, "adapter": model, "asr": tmp_path / "missing"}, [data], out, {"samples": 2})
    m = json.loads(path.read_text(encoding="utf-8"))
    assert m["models"]["adapter"]["weights"][0]["name"] == "adapter_model.safetensors"      # config.json is not a weight file
    assert m["models"]["adapter"]["weights"][0]["sha256"] == model_entry(model)["weights"][0]["sha256"]
    assert m["models"]["llm"] is None and m["models"]["asr"] is None
    assert m["outputs"][0]["rows"] == 2 and m["inputs"][0]["path"].endswith("in.jsonl")
    assert m["stats"] == {"samples": 2}
    assert isinstance(m["libraries"], dict)
