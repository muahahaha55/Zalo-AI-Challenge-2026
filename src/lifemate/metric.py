"""Re-implementation of the official LifeMate score (from the task statement; the official scorer is not shipped).

Per sample:
  gold empty            -> 1 if submitted list is empty else 0
  hash differs or M = 0 -> 0
  otherwise             -> F1 with P = M/(M+U), R = M/|G|
A submitted call matches a gold call when the tool name is equal, every non-fuzzy argument is identical and every
fuzzy argument (catalog `x-match: fuzzy`) has token-F1 > 0.6 (lower-cased, NFC, whitespace tokens). A call matches at
most one call; order is ignored, so we use a maximum bipartite matching.
"""
from __future__ import annotations

import json
import unicodedata
from collections import Counter
from pathlib import Path

FUZZY_THRESHOLD = 0.6


def load_fuzzy_params(*catalog_paths) -> dict[str, set[str]]:
    """{tool_name: {fuzzy param names}} from one or more *_tools_catalog.json files."""
    out: dict[str, set[str]] = {}
    for p in catalog_paths:
        cat = json.loads(Path(p).read_text(encoding="utf-8"))
        for app in cat["apps"]:
            for t in app["tools"]:
                props = t.get("parameters", {}).get("properties", {})
                out.setdefault(t["name"], set()).update(k for k, v in props.items() if v.get("x-match") == "fuzzy")
    return out


def _tokens(s) -> list[str]:
    return unicodedata.normalize("NFC", str(s)).lower().split()


def token_f1(a, b) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta and not tb:
        return 1.0
    common = sum((Counter(ta) & Counter(tb)).values())
    if common == 0:
        return 0.0
    p, r = common / len(ta), common / len(tb)
    return 2 * p * r / (p + r)


def calls_match(sub: dict, gold: dict, fuzzy: dict[str, set[str]]) -> bool:
    if sub.get("name") != gold.get("name"):
        return False
    sa, ga = sub.get("arguments") or {}, gold.get("arguments") or {}
    if set(sa) != set(ga):
        return False
    fz = fuzzy.get(gold["name"], set())
    for k, gv in ga.items():
        sv = sa[k]
        if k in fz and isinstance(sv, str) and isinstance(gv, str):
            if token_f1(sv, gv) <= FUZZY_THRESHOLD:
                return False
        elif sv != gv or type(sv) is not type(gv) and not (isinstance(sv, (int, float)) and isinstance(gv, (int, float))):
            return False
    return True


def count_matches(sub_calls: list[dict], gold_calls: list[dict], fuzzy: dict[str, set[str]]) -> int:
    """Size of a maximum matching between submitted and gold calls (augmenting paths)."""
    adj = [[j for j, g in enumerate(gold_calls) if calls_match(s, g, fuzzy)] for s in sub_calls]
    match_of_gold = [-1] * len(gold_calls)

    def try_assign(i, seen):
        for j in adj[i]:
            if j in seen:
                continue
            seen.add(j)
            if match_of_gold[j] == -1 or try_assign(match_of_gold[j], seen):
                match_of_gold[j] = i
                return True
        return False

    return sum(try_assign(i, set()) for i in range(len(sub_calls)))


def score_sample(sub_calls, sub_hash, gold_calls, gold_hash, fuzzy) -> float:
    if not gold_calls:
        return 1.0 if not sub_calls else 0.0
    if sub_hash != gold_hash:
        return 0.0
    m = count_matches(sub_calls, gold_calls, fuzzy)
    if m == 0:
        return 0.0
    u = len(sub_calls) - m
    p, r = m / (m + u), m / len(gold_calls)
    return 2 * p * r / (p + r)


def call_f1(sub_calls, gold_calls, fuzzy) -> float:
    """Same as score_sample but ignoring the hash (the tie-break score)."""
    return score_sample(sub_calls, "x", gold_calls, "x", fuzzy)
