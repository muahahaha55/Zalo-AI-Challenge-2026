import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lifemate import jsonl, paths  # noqa: E402


def test_data_root_and_processed_dir_can_be_overridden(monkeypatch, tmp_path):
    monkeypatch.setenv("LIFEMATE_DATA_ROOT", str(tmp_path / "d"))
    monkeypatch.setenv("LIFEMATE_PROCESSED_DIR", str(tmp_path / "p"))
    assert paths.data_root() == tmp_path / "d"
    assert paths.split_dir("public_test") == tmp_path / "d" / "public_test"
    assert paths.processed_dir() == tmp_path / "p"
    monkeypatch.delenv("LIFEMATE_DATA_ROOT")
    assert paths.data_root() == paths.ROOT / "data" / "raw" / "train+public_test"


def test_jsonl_round_trip_keeps_unicode_and_skips_blank_lines(tmp_path):
    p = tmp_path / "x.jsonl"
    assert jsonl.write_jsonl(p, [{"a": "Phạm Hùng"}, {"b": [1, 2]}]) == 2
    assert "Phạm Hùng" in p.read_text(encoding="utf-8")              # ensure_ascii=False
    p.write_text(p.read_text(encoding="utf-8") + "\n\n", encoding="utf-8")
    assert jsonl.read_jsonl(p) == [{"a": "Phạm Hùng"}, {"b": [1, 2]}]


def test_write_jsonl_uses_lf_on_every_platform(tmp_path):
    p = tmp_path / "x.jsonl"
    jsonl.write_jsonl(p, [{"a": 1}, {"a": 2}])
    assert p.read_bytes() == b'{"a": 1}\n{"a": 2}\n'
