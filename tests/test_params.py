import json
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lifemate import params  # noqa: E402


def write_safetensors(path: Path, shapes: dict[str, list[int]]) -> None:
    """A valid .safetensors file with float16 zeros: header only matters for counting, but the layout is the real one."""
    header, offset = {"__metadata__": {"format": "pt"}}, 0
    for name, shape in shapes.items():
        n = 2
        for d in shape:
            n *= d
        header[name] = {"dtype": "F16", "shape": shape, "data_offsets": [offset, offset + n]}
        offset += n
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(offset))


def test_count_file_counts_elements_and_ignores_metadata(tmp_path):
    f = tmp_path / "m.safetensors"
    write_safetensors(f, {"a": [2, 3], "b": [4], "scalar": []})
    assert params.count_file(f) == 6 + 4 + 1


def test_count_path_accepts_a_folder_of_shards(tmp_path):
    write_safetensors(tmp_path / "model-1.safetensors", {"a": [10, 10]})
    write_safetensors(tmp_path / "model-2.safetensors", {"b": [5]})
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    assert params.count_path(tmp_path) == 105
    assert params.count_path(tmp_path / "model-2.safetensors") == 5
    with pytest.raises(FileNotFoundError):
        params.count_path(tmp_path / "empty-folder-does-not-exist")


def test_limits_are_strict_and_independent():
    assert params.check_limits(4_999_999_999, 5_999_999_999) == []
    assert len(params.check_limits(5_000_000_000, 5_000_000_001)) == 1               # LLM at the limit is not below it
    assert len(params.check_limits(1, 6_000_000_000)) == 1
    assert len(params.check_limits(5_000_000_000, 6_000_000_000)) == 2
