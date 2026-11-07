import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lifemate.jsonl import ResumableJsonlWriter  # noqa: E402


def lines(path):
    return path.read_bytes().split(b"\n")[:-1]


def test_success_renames_part_to_final_with_lf(tmp_path):
    out = tmp_path / "r" / "x.jsonl"
    with ResumableJsonlWriter(out) as w:
        w.write({"id": "a", "v": "Phạm"})
        w.write({"id": "b"})
        assert not out.exists() and out.with_name("x.jsonl.part").exists()          # nothing at the final path yet
    assert out.exists() and not out.with_name("x.jsonl.part").exists()
    assert out.read_bytes() == '{"id": "a", "v": "Phạm"}\n{"id": "b"}\n'.encode()


def test_exception_leaves_part_and_never_creates_the_final_file(tmp_path):
    out = tmp_path / "x.jsonl"
    with pytest.raises(RuntimeError), ResumableJsonlWriter(out) as w:
        w.write({"id": "a"})
        raise RuntimeError("crash")
    assert not out.exists() and out.with_name("x.jsonl.part").exists()


def test_previous_final_file_survives_a_failed_run(tmp_path):
    out = tmp_path / "x.jsonl"
    out.write_text('{"id": "old"}\n', encoding="utf-8")
    with pytest.raises(RuntimeError), ResumableJsonlWriter(out) as w:
        w.write({"id": "new"})
        raise RuntimeError("crash")
    assert out.read_text(encoding="utf-8") == '{"id": "old"}\n'


def test_resume_keeps_complete_rows_and_cuts_a_truncated_tail(tmp_path):
    out = tmp_path / "x.jsonl"
    part = tmp_path / "x.jsonl.part"
    part.write_bytes(b'{"id": "a"}\n{"id": "b"}\n{"id": "c", "tru')
    with ResumableJsonlWriter(out, resume=True) as w:
        assert w.done == {"a", "b"}
        w.write({"id": "c"})
    assert [json.loads(x)["id"] for x in lines(out)] == ["a", "b", "c"]


def test_a_complete_last_line_without_newline_is_not_trusted(tmp_path):
    out = tmp_path / "x.jsonl"
    (tmp_path / "x.jsonl.part").write_bytes(b'{"id": "a"}\n{"id": "b"}')
    with ResumableJsonlWriter(out, resume=True) as w:
        assert w.done == {"a"}                                                      # b may have been cut mid-write


def test_without_resume_an_old_part_is_discarded(tmp_path):
    out = tmp_path / "x.jsonl"
    (tmp_path / "x.jsonl.part").write_bytes(b'{"id": "stale"}\n')
    with ResumableJsonlWriter(out) as w:
        assert w.done == set()
        w.write({"id": "fresh"})
    assert lines(out) == [b'{"id": "fresh"}']
