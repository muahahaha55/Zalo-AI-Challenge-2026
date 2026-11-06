"""Near-duplicate grouping of short texts, used to keep template-identical samples on the same side of a split.

Two texts are linked when the Jaccard similarity of their word sets is >= threshold; groups are the connected components
(union-find). The all-pairs scan is O(n^2) set operations; this uses a prefix-filter set-similarity join instead:
tokens are ordered rarest-first, only a short prefix of each set is indexed/probed (two sets with Jaccard >= t must
share a token in their prefixes), candidates are size-filtered and then checked EXACTLY, so the result is identical
to the brute-force scan (see tests/test_grouping.py) while the work stays near-linear for typical text.

The numeric group labels are reproduced exactly too: the dev split is chosen by shuffling those labels, so a different
label would silently change the dev set and invalidate every score measured on it. To get the same labels, the linked
pairs are collected and then merged in the order the all-pairs scan would have merged them (i ascending, j ascending).
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

_WORD = re.compile(r"\w+")


def tokens(text: str) -> frozenset[str]:
    return frozenset(_WORD.findall(text.lower()))


class _DisjointSet:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        self.parent[self.find(a)] = self.find(b)


def near_duplicate_groups(texts: list[str], threshold: float = 0.6) -> list[int]:
    """Group label per text (the label is the component's representative index)."""
    sets = [tokens(t) for t in texts]
    freq = Counter(tok for s in sets for tok in s)
    ordered = [sorted(s, key=lambda t: (freq[t], t)) for s in sets]      # rarest token first
    edges: list[tuple[int, int]] = []
    index: dict[str, list[int]] = defaultdict(list)
    for i, toks in enumerate(ordered):
        size = len(toks)
        if size == 0:
            continue                                     # empty sets are never similar to anything (union size 0 in the scan)
        prefix = toks[: size - math.ceil(threshold * size - 1e-9) + 1]
        seen: set[int] = set()
        for tok in prefix:
            for j in index[tok]:
                if j in seen:
                    continue
                seen.add(j)
                sj = len(ordered[j])
                if sj < threshold * size - 1e-9 or size < threshold * sj - 1e-9:     # size filter
                    continue
                inter = len(sets[i] & sets[j])
                if inter / (size + sj - inter) >= threshold:
                    edges.append((j, i) if j < i else (i, j))
        for tok in prefix:
            index[tok].append(i)
    dsu = _DisjointSet(len(texts))
    for a, b in sorted(edges):                           # same merge order (and so same representatives) as the scan
        dsu.union(a, b)
    return [dsu.find(i) for i in range(len(texts))]
