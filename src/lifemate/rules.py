"""Deterministic post-processing of a predicted tool call (Step 7).

The LLM often gets the *intent* right and the arithmetic wrong (8 giờ kém 15 -> 20:45, "thứ ba tuần sau" off by a day, spoken phone
digits garbled). These rules re-derive such values from the transcript and the current time, and only fire when they are unambiguous:
  * time/date: the utterance has exactly ONE clock mention and at most one date mention, no relative shift ("muộn thêm", "nữa", ...),
    and the call has exactly ONE datetime-shaped argument. Only the parts the transcript states are overwritten (date and/or h:m).
  * phone: the utterance contains exactly one spoken digit run of 9-11 digits and the predicted phone differs from it.
Values are recognised by SHAPE (ISO datetime / date / HH:MM / digit string), not by tool name, so unseen apps are covered too.
Pure functions, no I/O, no environment access.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from .normalize import NUMWORDS, words_to_digits

ISO_DT = re.compile(r"^(\d{4})-(\d\d)-(\d\d)T(\d\d):(\d\d)(.*)$")
ISO_D = re.compile(r"^\d{4}-\d\d-\d\d$")
HHMM = re.compile(r"^(\d\d):(\d\d)$")
PERIOD = r"(sáng|trưa|chiều|tối|đêm)"
SHIFT_WORDS = re.compile(r"\b(muộn|sớm|lùi|dời|hoãn|nữa|cộng thêm|giờ cũ|giờ mới|theo giờ|trước giờ|sau giờ|thêm \d+ (?:phút|tiếng)|\d+ (?:phút|tiếng))\b")
SELF_CORRECT = re.compile(r"\b(à không|à mà|à thôi|nhầm|ý là|đổi lại|sửa lại|khỏi)\b")
MAX_SHIFT_HOURS = 25      # a genuine model slip is 1 hour or 1 day; a bigger gap means the stated time belongs to something else
WEEKDAY = {"2": 0, "3": 1, "4": 2, "5": 3, "6": 4, "7": 5}


def norm(text: str) -> str:
    t = unicodedata.normalize("NFC", text).lower()
    t = re.sub(r"[,.;!?…\"']", " ", t)
    return words_to_digits(" ".join(t.split()))


@dataclass
class When:
    clock: tuple[int, int, str | None] | None = None      # (hour, minute, period)
    day: date | None = None


def _period_near(t: str, start: int, end: int) -> str | None:
    after = re.match(rf"\s*(?:\d{{1,2}}\s+)?{PERIOD}\b", t[end:end + 18])
    if after:
        return after.group(1)
    before = re.search(rf"{PERIOD}(?: nay| mai| hôm nay| nào| hôm đó)?\s*$", t[max(0, start - 16):start])
    return before.group(1) if before else None


def _clocks(t: str):
    found = []
    for m in re.finditer(r"\b(\d{1,2}) (?:giờ|h)\b(?: (kém) (\d{1,2})| (rưỡi)| (\d{1,2})(?: phút)?\b(?! (?:tiếng|ngày|tháng|người|cái|lần|km|cây|số|tuổi|món|phần|bạn)))?", t):
        h = int(m.group(1))
        if m.group(2):
            h, mi = h - 1, 60 - int(m.group(3))
        elif m.group(4):
            mi = 30
        elif m.group(5):
            mi = int(m.group(5))
        else:
            mi = 0
        if not (0 <= h <= 24 and 0 <= mi < 60) or re.match(r" (nữa|sau|đồng hồ)", t[m.end():m.end() + 10]):
            continue
        nxt = t[m.end():].split(" ")[1:2]          # a number word right after the clock that words_to_digits could not read
        if nxt and not m.group(4) and (nxt[0] in NUMWORDS or nxt[0] in ("rưỡi", "kém")):   # (e.g. 'năm giờ bốn lăm'): do not guess
            return None
        found.append((m.start(), m.end(), h, mi))
    for m in re.finditer(r"\b(\d{1,2}) rưỡi\b", t):
        if not any(s <= m.start() < e for s, e, _, _ in found):
            found.append((m.start(), m.end(), int(m.group(1)), 30))
    for m in re.finditer(r"\b(\d{1,2}):(\d{2})\b", t):
        found.append((m.start(), m.end(), int(m.group(1)), int(m.group(2))))
    found.sort()
    return found


def _days(t: str, today: date, clock_spans):
    out = set()
    if re.search(r"\b(hôm nay|bữa nay|(?:sáng|trưa|chiều|tối|đêm) nay)\b", t):
        out.add(today)
    near_clock = lambda i: any(abs(i - s) <= 18 or abs(i - e) <= 18 for s, e in clock_spans)
    for m in re.finditer(r"\bmai\b", t):          # 'mai' is also a first name and ASR output is lower-case: require time context
        ctx = t[max(0, m.start() - 6):m.start()]
        if re.search(r"(ngày|sáng|trưa|chiều|tối|đêm)\s*$", ctx) or near_clock(m.start()):
            out.add(today + timedelta(days=1))
    if re.search(r"\b(ngày kia|ngày mốt|mốt)\b", t):
        out.add(today + timedelta(days=2))
    if re.search(r"\bhôm qua\b", t):
        out.add(today - timedelta(days=1))
    monday = today - timedelta(days=today.weekday())
    for m in re.finditer(r"\b(?:thứ (\d)|(chủ nhật))(?: tuần (này|sau|tới|trước))?", t):
        wd = 6 if m.group(2) else WEEKDAY.get(m.group(1))
        if wd is None:
            continue
        qual = m.group(3)
        if qual in ("sau", "tới"):
            out.add(monday + timedelta(days=7 + wd))
        elif qual == "này":
            out.add(monday + timedelta(days=wd))
        elif qual == "trước":
            out.add(monday + timedelta(days=wd - 7))
        else:
            out.add(today + timedelta(days=(wd - today.weekday()) % 7 or 7))
    for m in re.finditer(r"\b(?:ngày|mùng|mồng) (\d{1,2})(?: tháng (\d{1,2}))?\b", t):
        d, mo = int(m.group(1)), int(m.group(2) or today.month)
        try:
            cand = date(today.year, mo, d)
            if cand < today and not m.group(2):
                cand = date(today.year + (mo == 12), mo % 12 + 1, d)
            out.add(cand)
        except ValueError:
            pass
    return out


def extract_when(transcript: str, now: datetime) -> When | None:
    """What the utterance says about time, or None when it is absent, relative, or ambiguous."""
    t = norm(transcript)
    if SHIFT_WORDS.search(t) or SELF_CORRECT.search(t):
        return None
    clocks = _clocks(t)
    if clocks is None:
        return None
    if len({(h, m) for _, _, h, m in clocks}) > 1:
        return None
    days = _days(t, now.date(), [(s, e) for s, e, _, _ in clocks])
    if len(days) > 1:
        return None
    clock = None
    if clocks:
        s, e, h, m = clocks[0]
        clock = (h, m, _period_near(t, s, e))
    if clock is None and not days:
        return None
    return When(clock, next(iter(days), None))


def _resolve_hour(h: int, period: str | None, pred_hour: int | None) -> int | None:
    if h == 24:
        return 0
    if period in ("chiều", "tối"):
        return h + 12 if h < 12 else h
    if period == "trưa":
        return h + 12 if 1 <= h <= 5 else h
    if period == "sáng":
        return 0 if h == 12 else h
    if period == "đêm":
        return h + 12 if 6 <= h < 12 else (None if h == 12 else h)
    if h >= 13 or pred_hour is None:
        return h if h >= 13 else None
    return h % 12 + (12 if pred_hour >= 12 else 0)      # no am/pm word: keep the model's half of the day


def _fix_datetime(v: str, when: When) -> str | None:
    m = ISO_DT.match(v)
    y, mo, d, hh, mm, rest = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5)), m.group(6)
    if when.day:
        y, mo, d = when.day.year, when.day.month, when.day.day
    if when.clock:
        h = _resolve_hour(when.clock[0], when.clock[2], hh)
        if h is None:
            return None
        hh, mm = h, when.clock[1]
    try:
        old = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5)))
        new = datetime(y, mo, d, hh, mm)
    except ValueError:
        return None
    explicit_ampm = bool(when.clock and when.clock[2])
    if abs(new - old) > timedelta(hours=MAX_SHIFT_HOURS) and not (old.date() == new.date() and explicit_ampm):
        return None
    return f"{y:04d}-{mo:02d}-{d:02d}T{hh:02d}:{mm:02d}{rest}"


def _fix_hhmm(v: str, when: When) -> str | None:
    if not when.clock:
        return None
    h = _resolve_hour(when.clock[0], when.clock[2], int(v[:2]))
    return None if h is None else f"{h:02d}:{when.clock[1]:02d}"


def spoken_numbers(transcript: str, lo: int = 9, hi: int = 11) -> list[str]:
    return [tok for tok in norm(transcript).split() if tok.isdigit() and lo <= len(tok) <= hi]


def fix_args(name: str, args: dict, transcript: str, current_time: str) -> tuple[dict, list[str]]:
    """Return (possibly corrected arguments, list of human-readable changes). Never raises."""
    changes: list[str] = []
    try:
        new = dict(args)
        now = datetime.fromisoformat(current_time)
        dt_keys = [k for k, v in args.items() if isinstance(v, str) and (ISO_DT.match(v) or HHMM.match(v) or ISO_D.match(v))]
        if len(dt_keys) == 1:
            k, v = dt_keys[0], args[dt_keys[0]]
            when = extract_when(transcript, now)
            if when:
                nv = _fix_datetime(v, when) if ISO_DT.match(v) else _fix_hhmm(v, when) if HHMM.match(v) else (
                    when.day.isoformat() if when.day else None)
                if nv and nv != v:
                    new[k] = nv
                    changes.append(f"{name}.{k}: {v} -> {nv}")
        elif len(dt_keys) > 1:          # e.g. start + end: which one carries the stated clock is unknown, but the stated DAY is shared
            when = extract_when(transcript, now)
            if when and when.day:
                for k in dt_keys:
                    m = ISO_DT.match(args[k]) if isinstance(args[k], str) else None
                    if m:
                        old = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
                        if old != when.day and abs((old - when.day).days) <= 1:
                            nv = f"{when.day.isoformat()}T{m.group(4)}:{m.group(5)}{m.group(6)}"
                            new[k] = nv
                            changes.append(f"{name}.{k}: {args[k]} -> {nv}")
        nums = set(spoken_numbers(transcript))
        for k, v in args.items():
            if "phone" in k and isinstance(v, str) and v.isdigit() and len(nums) == 1 and v not in nums:
                nv = next(iter(nums))
                new[k] = nv
                changes.append(f"{name}.{k}: {v} -> {nv}")
        return new, changes
    except Exception:
        return args, []
