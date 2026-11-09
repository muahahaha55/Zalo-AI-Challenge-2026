import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("verify_image", ROOT / "scripts" / "verify_image.py")
vi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vi)


def test_parse_done_reads_the_container_summary_line():
    log = "x\ndone 634 samples; mean 1.72s p95 3.48s max 4.96s; stop reasons {'end': 500, 'ask': 134}\n"
    got = vi.parse_done(log)
    assert got["samples"] == 634 and got["max_s"] == 4.96 and got["p95_s"] == 3.48 and "'ask': 134" in got["stop_reasons"]
    assert vi.parse_done("crashed") == {}


def test_sha256_of_a_file(tmp_path):
    p = tmp_path / "a"
    p.write_bytes(b"abc")
    assert vi.sha256(p) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert sys.version_info >= (3, 12)
