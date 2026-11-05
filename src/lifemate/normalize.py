"""Vietnamese spoken-number -> digits. Used to (a) measure WER fairly and (b) later post-process ASR text.

Conservative: a run of number words that cannot be parsed is left untouched.
Handles: 'một trăm hai mươi ba triệu' -> 123 triệu, 'hai triệu tám' -> 2800000 (colloquial tail), 'hai triệu rưỡi',
'mười lăm', 'hai mươi mốt', digit strings like 'không chín một hai ...'.
"""
from __future__ import annotations

import re

DIGITS = {"không": 0, "một": 1, "mốt": 1, "hai": 2, "ba": 3, "bốn": 4, "tư": 4, "năm": 5, "lăm": 5,
          "sáu": 6, "bảy": 7, "bẩy": 7, "tám": 8, "chín": 9}
UNITS = {"nghìn": 10**3, "ngàn": 10**3, "triệu": 10**6, "tỷ": 10**9, "tỉ": 10**9}
OTHERS = {"mười", "mươi", "chục", "trăm", "linh", "lẻ"}
NUMWORDS = set(DIGITS) | set(UNITS) | OTHERS


def _parse(tokens: list[str]) -> int | None:
    total, group, cur, last_unit, seen_unit = 0, 0, None, None, False
    for t in tokens:
        if t in DIGITS:
            if cur is not None:
                return None
            cur = DIGITS[t]
        elif t == "mười":
            if cur is not None:
                return None
            group += 10
        elif t in ("mươi", "chục"):
            group += (cur if cur is not None else 1) * 10
            cur = None
        elif t == "trăm":
            group += (cur if cur is not None else 1) * 100
            cur = None
        elif t in ("linh", "lẻ"):
            continue
        elif t in UNITS:
            v = group + (cur or 0)
            if v == 0:
                return None
            total += v * UNITS[t]
            last_unit, seen_unit, group, cur = UNITS[t], True, 0, None
    rest = group + (cur or 0)
    if rest:
        if seen_unit and last_unit >= 1000 and rest < 10:
            total += rest * last_unit // 10                  # 'hai triệu tám' = 2.8 triệu
        elif seen_unit and last_unit >= 10**6 and rest < 1000 and rest % 100 == 0:
            total += rest * 1000                             # 'hai triệu hai trăm' = 2.2 triệu
        else:
            total += rest
    return total


def words_to_digits(text: str) -> str:
    toks = text.split()
    out, i = [], 0
    while i < len(toks):
        if toks[i].lower() not in NUMWORDS:
            out.append(toks[i]); i += 1
            continue
        j = i
        while j < len(toks) and toks[j].lower() in NUMWORDS:
            j += 1
        run = [t.lower() for t in toks[i:j]]
        rung = j
        half = False
        if j < len(toks) and toks[j].lower() == "rưỡi" and any(t in UNITS for t in run):   # 'hai triệu rưỡi'
            half, rung = True, j + 1
        val: int | None
        if len(run) >= 3 and all(t in DIGITS for t in run):          # spoken digit string (phone numbers, ...)
            val = None
            out.append("".join(str(DIGITS[t]) for t in run))
        else:
            val = _parse(run)
            if val is None:
                out.extend(toks[i:j])
            else:
                if half:
                    unit = next(UNITS[t] for t in reversed(run) if t in UNITS)
                    val += unit // 2
                out.append(str(val))
        i = rung if (val is not None and half) else j
    return " ".join(out)
