import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lifemate.metric import count_matches, score_sample

FZ = {"notes.create": {"title"}}
g = [{"name": "a.read", "arguments": {}}, {"name": "a.write", "arguments": {"id": "x", "n": 5}}]


def test_exact():
    assert score_sample(g, "h", g, "h", FZ) == 1.0


def test_order_ignored():
    assert score_sample(g[::-1], "h", g, "h", FZ) == 1.0


def test_hash_mismatch_zero():
    assert score_sample(g, "z", g, "h", FZ) == 0.0


def test_extra_call_lowers_precision():
    s = score_sample(g + [{"name": "a.read", "arguments": {"q": 1}}], "h", g, "h", FZ)
    assert abs(s - 0.8) < 1e-9  # P=2/3, R=1


def test_missing_call_lowers_recall():
    assert abs(score_sample(g[1:], "h", g, "h", FZ) - 2 / 3) < 1e-9


def test_oos():
    assert score_sample([], "h", [], "h2", FZ) == 1.0
    assert score_sample(g[:1], "h", [], "h", FZ) == 0.0


def test_fuzzy_threshold():
    gold = [{"name": "notes.create", "arguments": {"title": "mua sữa cho bé"}}]
    ok = [{"name": "notes.create", "arguments": {"title": "Mua sữa cho bé nhé"}}]
    bad = [{"name": "notes.create", "arguments": {"title": "họp lúc ba giờ"}}]
    assert count_matches(ok, gold, FZ) == 1 and count_matches(bad, gold, FZ) == 0


def test_max_matching_not_greedy():
    f = {"t": {"x"}}
    gold = [{"name": "t", "arguments": {"x": "a b c d e"}}, {"name": "t", "arguments": {"x": "a b c d"}}]
    sub = [{"name": "t", "arguments": {"x": "a b c d"}}, {"name": "t", "arguments": {"x": "a b c d e"}}]
    assert count_matches(sub, gold, f) == 2
