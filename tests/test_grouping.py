import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lifemate.grouping import near_duplicate_groups  # noqa: E402


def brute_force(texts, thr=0.6):
    """The original all-pairs scan, kept here as the reference implementation."""
    toks = [set(re.findall(r"\w+", t.lower())) for t in texts]
    par = list(range(len(texts)))

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            u = len(toks[i] | toks[j])
            if u and len(toks[i] & toks[j]) / u >= thr:
                par[find(i)] = find(j)
    return [find(i) for i in range(len(texts))]


def test_simple_groups():
    g = near_duplicate_groups(["bật đèn phòng khách", "bật đèn phòng khách nhé", "chuyển tiền cho mẹ", ""])
    assert g[0] == g[1] and g[2] != g[0] and len({g[0], g[2], g[3]}) == 3


def test_identical_to_brute_force_on_random_texts_including_labels():
    rng = random.Random(7)
    vocab = [f"w{i}" for i in range(25)] + ["tôi", "nhé", "giúp", "cho"] * 6      # a few very frequent words, many rare ones
    for trial in range(40):
        texts = [" ".join(rng.choice(vocab) for _ in range(rng.randint(0, 9))) for _ in range(rng.randint(0, 60))]
        for thr in (0.3, 0.5, 0.6, 0.8, 1.0):
            assert near_duplicate_groups(texts, thr) == brute_force(texts, thr), (trial, thr)


def test_threshold_edge_is_inclusive():
    # {a,b,c} vs {a,b,c,d,e}: Jaccard = 3/5 = 0.6 exactly -> linked at 0.6, not at 0.61
    g = near_duplicate_groups(["a b c", "a b c d e"], 0.6)
    assert g[0] == g[1]
    g = near_duplicate_groups(["a b c", "a b c d e"], 0.61)
    assert g[0] != g[1]
