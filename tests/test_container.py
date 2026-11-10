"""Layout detection and the container entry point. The end-to-end tests need the competition data and are skipped without it."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lifemate import container  # noqa: E402
from lifemate.paths import split_dir  # noqa: E402

HAVE_DATA = (split_dir("public_test") / "public_test.jsonl").exists()


def make_split(folder: Path, name: str) -> Path:
    (folder / "audio").mkdir(parents=True)
    (folder / f"{name}.jsonl").write_text("", encoding="utf-8")
    return folder


def test_flat_layout_is_detected(tmp_path):
    make_split(tmp_path, "private_test")
    (tmp_path / "sample_submission.jsonl").write_text("", encoding="utf-8")          # never mistaken for a split
    assert container.find_split(tmp_path) == ("private_test", True)


def test_nested_layout_is_detected(tmp_path):
    make_split(tmp_path / "private_test", "private_test")
    assert container.find_split(tmp_path) == ("private_test", False)


def test_ambiguity_needs_split_unless_private_test_is_present(tmp_path):
    make_split(tmp_path / "train", "train")
    make_split(tmp_path / "public_test", "public_test")
    with pytest.raises(container.LayoutError, match="--split"):
        container.find_split(tmp_path)
    assert container.find_split(tmp_path, "train") == ("train", False)
    make_split(tmp_path / "private_test", "private_test")
    assert container.find_split(tmp_path) == ("private_test", False)               # the contest split wins


def test_unknown_or_missing_folder_is_an_error(tmp_path):
    with pytest.raises(container.LayoutError):
        container.find_split(tmp_path / "nope")
    with pytest.raises(container.LayoutError):
        container.find_split(tmp_path)                                              # nothing in it


def test_flat_layout_gets_a_link_so_the_official_root_convention_holds(tmp_path):
    data = make_split(tmp_path / "mounted", "private_test")
    (data / "marker.txt").write_text("x", encoding="utf-8")
    layout = container.resolve_layout(data, stage_dir=tmp_path / "stage")
    assert layout.flat and layout.split == "private_test" and layout.data_root == tmp_path / "stage"
    assert (layout.data_root / "private_test" / "marker.txt").read_text(encoding="utf-8") == "x"   # root/<split>/... resolves


def test_nested_layout_needs_no_link(tmp_path):
    make_split(tmp_path / "private_test", "private_test")
    layout = container.resolve_layout(tmp_path)
    assert not layout.flat and layout.data_root == tmp_path.resolve()


def test_force_offline_sets_the_flags(monkeypatch):
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "USE_HUB_KERNELS"):
        monkeypatch.delenv(key, raising=False)
    container.force_offline()
    import os
    assert [os.environ[k] for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")] == ["1", "1", "1"]
    assert os.environ["USE_HUB_KERNELS"] == "NO"


@pytest.mark.skipif(not HAVE_DATA, reason="competition data not available")
def test_end_to_end_plumbing_on_public_test_and_resume(tmp_path, monkeypatch):
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "USE_HUB_KERNELS"):
        monkeypatch.setenv(key, "")                                                 # main() forces these; restored after the test
    out = tmp_path / "result" / "submission.jsonl"
    args = ["--data", str(split_dir("public_test")), "--backend", "null", "--asr", "none", "--limit", "5", "--out", str(out)]
    assert container.main(args) == 0
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 5 and all(r["tool_calls"] == [] and len(r["state_hash"]) == 64 for r in rows)
    assert all(isinstance(r["time"], int) and r["time"] >= 0 for r in rows)            # organisers' format: ms per sample
    assert not out.with_name(out.name + ".part").exists()                           # finished: no leftover partial file

    # simulate a crash after 2 rows plus a half-written third line, then resume
    part = out.with_name(out.name + ".part")
    first = out.read_text(encoding="utf-8").splitlines()
    part.write_text("\n".join(first[:2]) + "\n" + first[2][:20], encoding="utf-8")
    out.unlink()
    assert container.main([*args, "--resume"]) == 0
    def strip_time(lines):
        return [{k: v for k, v in json.loads(x).items() if k != "time"} for x in lines]
    assert strip_time(out.read_text(encoding="utf-8").splitlines()) == strip_time(first)   # same answers as the uninterrupted run
