"""Training calls with correct labels by construction: transplant real call segments into other samples' environments.

A *segment* is a run of consecutive gold calls of one app (reads followed by writes). It is replayed in the environment
of another train sample (the *base*) whose installed apps include every tool of the segment:
  * ids (parameters marked x-ref) are re-bound: the donor's id is located in the donor's results by its key path
    (e.g. events/id) and replaced by a value found at the same path in the base environment's own results;
  * ISO dates are shifted by the difference between the two `current_time` days;
  * enum parameters may be swapped for another allowed value;
  * every call goes through env.validate and env.call; a rejected call drops the whole sample.
Because ids always come from the base environment's real results, the label is consistent with the state. The request
text does not exist yet: `describe()` builds the instruction for a LOCAL language model that writes it (dev time only).

The module never constructs an environment itself (`env_factory` is injected), and only TRAIN-split items may be passed in.
"""
from __future__ import annotations

import json
import random
import re
import unicodedata
from collections.abc import Callable
from datetime import date, timedelta

ISO_DATE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})")
ENUM_SWAP_PROB = 0.3
SEGMENT_COUNT_WEIGHTS = {1: 0.45, 2: 0.4, 3: 0.15}
MAX_CALLS = 7          # the agent allows 8 steps (calls + END); real train answers have at most 6 calls


def app_of(tool_name: str) -> str:
    return tool_name.split(".", 1)[0]


def _app_runs(item: dict) -> list[list[tuple[dict, int]]]:
    out: list[list[tuple[dict, int]]] = []
    for i, c in enumerate(item["label"]["tool_calls"]):
        if app_of(c["name"]) == "sys":
            break
        if out and app_of(out[-1][-1][0]["name"]) == app_of(c["name"]):
            out[-1].append((c, i))
        else:
            out.append([(c, i)])
    return out


def _refs_into(run: list[tuple[dict, int]], results: list[dict], cat: dict) -> bool:
    """True if a call of `run` uses an id that appears in `results` (so the run depends on them)."""
    for c, _ in run:
        props = cat.get(c["name"], {}).get("parameters", {}).get("properties", {})
        for k, v in c["arguments"].items():
            if "x-ref" in props.get(k, {}) and any(find_path(one, results) for one in (v if isinstance(v, list) else [v])):
                return True
    return False


def segments(item: dict, cat: dict) -> list[list[tuple[dict, int]]]:
    """Self-contained pieces of the gold answer that can be replayed elsewhere.
    Runs of one app; a run that uses ids read by the previous run is merged into it (contacts.search -> sms.send).
    A piece is kept only if it ends with a write, or is the whole answer: a read cut out of a longer chain would teach
    the model to stop after reading. sys.ask_missing_info ends the list."""
    runs = _app_runs(item)
    results = item["label"]["tool_results"]
    merged: list[list[tuple[dict, int]]] = []
    for run in runs:
        if merged and _refs_into(run, results[merged[-1][0][1]: merged[-1][-1][1] + 1], cat):
            merged[-1] = merged[-1] + run
            continue
        merged.append(run)
    n_calls = len(item["label"]["tool_calls"])
    return [s for s in merged if cat.get(s[-1][0]["name"], {}).get("kind") == "write" or len(s) == n_calls]


def _walk(node, path: tuple, value):
    """Yield the key path (list indexes dropped) at which `value` occurs in a nested result."""
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _walk(v, path + (k,), value)
    elif isinstance(node, list):
        for v in node:
            yield from _walk(v, path, value)
    elif node == value:
        yield path


def find_path(value: str, results: list[dict]) -> tuple | None:
    """Path of the most recent result field equal to `value`, or None."""
    for res in reversed(results):
        for path in _walk(res, (), value):
            return path
    return None


def values_at(path: tuple, results: list[dict]) -> list:
    """Distinct values found at `path` in the results, in order of first appearance (latest result first)."""
    seen: list = []

    def visit(node, depth):
        if depth == len(path):
            if not isinstance(node, (dict, list)) and node not in seen:
                seen.append(node)
            return
        if isinstance(node, dict) and path[depth] in node:
            visit(node[path[depth]], depth + 1)
        elif isinstance(node, list):
            for v in node:
                visit(v, depth)

    for res in reversed(results):
        visit(res, 0)
    return seen


def shift_dates(value, days: int):
    """Shift every ISO date inside a string argument (also in lists) by `days`."""
    if isinstance(value, list):
        return [shift_dates(v, days) for v in value]
    if isinstance(value, str) and days:
        m = ISO_DATE.match(value)
        if m:
            d = date(int(m[1]), int(m[2]), int(m[3])) + timedelta(days=days)
            return d.isoformat() + value[10:]
    return value


def _day(item: dict) -> date:
    return date.fromisoformat(item["context"]["current_time"][:10])


def eligible(segment: list[tuple[dict, int]], base: dict) -> bool:
    installed = {t for info in base["context"]["installed_apps"].values() for t in info["tools"]}
    return all(c["name"] in installed for c, _ in segment)


def transplant(segment: list[tuple[dict, int]], donor: dict, base: dict, env, seen: list[dict], cat: dict,
               rng: random.Random) -> tuple[list[dict], list[dict]] | None:
    """Replay the segment in `env` and return (calls, results), or None when anything does not fit.
    `seen` are the results already produced in this sample (ids may be taken from them)."""
    delta = (_day(base) - _day(donor)).days
    donor_results = donor["label"]["tool_results"]
    calls: list[dict] = []
    results: list[dict] = []
    for call, idx in segment:
        props = cat[call["name"]].get("parameters", {}).get("properties", {})
        args = {}
        for k, v in call["arguments"].items():
            p = props.get(k, {})
            if "x-ref" in p:
                vals = v if isinstance(v, list) else [v]
                if not vals:
                    return None
                path = find_path(vals[0], donor_results[:idx])    # where the donor read this id (all list items share it)
                if path is None:
                    return None
                pool = values_at(path, seen + results)
                if not pool or len(pool) < len(vals):
                    return None
                picked = rng.sample(pool, len(vals))
                args[k] = picked if isinstance(v, list) else picked[0]
                continue
            if "enum" in p and rng.random() < ENUM_SWAP_PROB:
                args[k] = rng.choice(p["enum"])
                continue
            args[k] = shift_dates(v, delta)
        if not env.validate(call["name"], args).get("ok", False):
            return None
        res = env.call(call["name"], args)
        if not res.get("ok", False):
            return None
        calls.append({"name": call["name"], "arguments": args})
        results.append(res)
    return calls, results


def generate(items: list[dict], cat: dict, env_factory: Callable[[str], object], rng: random.Random, n: int,
             max_tries_factor: int = 30, weights: dict[int, float] | None = None) -> tuple[list[dict], dict]:
    """Up to n new samples (no text yet). Returns (samples, stats). Reproducible for a given seed and item order.
    `weights`: share of samples built from 1, 2, 3 ... segments (default SEGMENT_COUNT_WEIGHTS). Long chains are the ones the
    text filters reject most often, so ask for more of them when the output will be filtered."""
    bases = [it for it in items if not it["history"]]
    donors = [it for it in items if segments(it, cat)]
    out: list[dict] = []
    stats = {"tries": 0, "no_segment": 0, "rejected": 0, "duplicate": 0}
    seen_keys: set[str] = set()
    shares = weights or SEGMENT_COUNT_WEIGHTS
    counts, probs = list(shares), list(shares.values())
    while len(out) < n and stats["tries"] < n * max_tries_factor:
        stats["tries"] += 1
        base = rng.choice(bases)
        want = rng.choices(counts, probs)[0]
        env = env_factory(base["id"])
        calls: list[dict] = []
        results: list[dict] = []
        used: list[str] = []
        for _ in range(want):
            donor = rng.choice(donors)
            apps_so_far = {app_of(c["name"]) for c in calls}           # one sub-request per app: no repeated reads
            segs = [s for s in segments(donor, cat) if eligible(s, base) and not {app_of(c["name"]) for c, _ in s} & apps_so_far
                    and len(calls) + len(s) <= MAX_CALLS]
            if not segs:
                stats["no_segment"] += 1
                continue
            res = transplant(rng.choice(segs), donor, base, env, results, cat, rng)
            if res is None:
                stats["rejected"] += 1
                break
            calls += res[0]
            results += res[1]
            used.append(donor["id"])
        else:
            if not calls:
                continue
            key = json.dumps(calls, ensure_ascii=False, sort_keys=True)
            if key in seen_keys:
                stats["duplicate"] += 1
                continue
            seen_keys.add(key)
            out.append({"id": f"{base['id']}#cg{len(out):05d}", "context": base["context"], "history": [],
                        "label": {"tool_calls": calls, "tool_results": results}, "base": base["id"], "donors": used})
    return out, stats


_ID_LIKE = re.compile(r"\b[a-z]{1,8}_\d+\b")
_TOOL_LIKE = re.compile(r"\b[a-z_]+\.[a-z_]+\b")
MIN_WORDS, MAX_WORDS = 3, 60
MIN_ARG_COVERAGE = 0.5
VIETNAMESE_MARKS = set("àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ")
MIN_WORDS_FOR_DIACRITICS = 6


def foreign_script(text: str) -> bool:
    """True if a letter is not Latin script (CJK, Cyrillic, Thai, ...): models sometimes switch language mid-sentence."""
    return any(ch.isalpha() and not unicodedata.name(ch, "").startswith("LATIN") for ch in text)


def lacks_vietnamese(text: str) -> bool:
    """A real spoken request of a few words always carries Vietnamese diacritics; text without any is not Vietnamese."""
    return len(text.split()) >= MIN_WORDS_FOR_DIACRITICS and not (set(text.lower()) & VIETNAMESE_MARKS)


def _words(text: str) -> list[str]:
    return re.findall(r"\w+", unicodedata.normalize("NFC", text).lower())


def free_text_args(sample: dict, cat: dict) -> list[str]:
    """String arguments the speaker must have said (titles, names, message content, rooms): not ids, enums or dates."""
    out = []
    for c in sample["label"]["tool_calls"]:
        props = cat[c["name"]].get("parameters", {}).get("properties", {})
        for k, v in c["arguments"].items():
            p = props.get(k, {})
            if isinstance(v, str) and len(v) >= 2 and "x-ref" not in p and "enum" not in p and not ISO_DATE.match(v):
                out.append(v)
    return out


def clean_text(raw: str) -> str:
    """First non-empty line of the model output, without surrounding quotes or a leftover thinking block."""
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    for line in raw.splitlines():
        line = line.strip().strip('"“”\'').strip()
        if line:
            return line
    return ""


def text_ok(text: str, sample: dict, cat: dict) -> tuple[bool, str]:
    """(keep?, reason). Rejects texts that leak ids or tool names, or whose length is implausible for a spoken request."""
    n = len(text.split())
    if n < MIN_WORDS or n > MAX_WORDS:
        return False, "length"
    if foreign_script(text):
        return False, "foreign_script"
    if lacks_vietnamese(text):
        return False, "not_vietnamese"
    if _ID_LIKE.search(text) or _TOOL_LIKE.search(text):
        return False, "id_or_tool_name"
    name = sample["context"].get("user_profile", {}).get("name", "")
    if name and name.lower() in text.lower():
        return False, "third_person"
    words = set(_words(text))
    for value in free_text_args(sample, cat):
        need = _words(value)
        if len(need) >= 2 and sum(w in words for w in need) / len(need) < MIN_ARG_COVERAGE:
            return False, "missing_args"
    for c in sample["label"]["tool_calls"]:
        props = cat[c["name"]].get("parameters", {}).get("properties", {})
        for k, v in c["arguments"].items():
            if "x-ref" in props.get(k, {}):
                for one in v if isinstance(v, list) else [v]:
                    if isinstance(one, str) and one in text:
                        return False, "id_or_tool_name"
    return True, "ok"


def apply_holdout(sample: dict, hold_apps: set) -> dict | None:
    """'Unseen apps' experiment: None if the sample's calls use a held app, else a copy without those apps installed."""
    if {app_of(c["name"]) for c in sample["label"]["tool_calls"]} & hold_apps:
        return None
    new = json.loads(json.dumps(sample))
    new["context"]["installed_apps"] = {a: v for a, v in sample["context"]["installed_apps"].items() if a not in hold_apps}
    return new


def _brief(calls: list[dict]) -> str:
    return "; ".join(json.dumps(c, ensure_ascii=False) for c in calls)


def style_examples(sample: dict, items: list[dict], rng: random.Random, k: int = 3) -> list[tuple[str, list[dict]]]:
    """k real (request, gold calls) pairs that use NONE of the sample's apps: they show how real users talk, and a model
    that copies one of them produces text about other apps, which the verification pass then rejects. (The donors'
    own requests are not used: copying them gives text that half matches the new calls, the hardest error to catch.)"""
    apps = {app_of(c["name"]) for c in sample["label"]["tool_calls"]}
    pool = [it for it in items if it["label"]["tool_calls"] and not it["history"]
            and not {app_of(c["name"]) for c in it["label"]["tool_calls"]} & apps]
    return [(it["audio_text"], it["label"]["tool_calls"]) for it in rng.sample(pool, min(k, len(pool)))]


def verify_prompt(sample: dict, text: str) -> str:
    """Yes/no question for the local model: does `text` ask for exactly these operations? Answer starts with CÓ or KHÔNG."""
    ops = "\n".join(f"- {json.dumps(c, ensure_ascii=False)}" for c in sample["label"]["tool_calls"])
    return (
        f"Người dùng nói với trợ lý: \"{text}\"\n"
        f"Trợ lý thực hiện các thao tác:\n{ops}\n"
        "Câu nói có yêu cầu ĐÚNG và ĐỦ các thao tác này (các thao tác đọc/tra cứu để lấy thông tin là được phép), "
        "và KHÔNG yêu cầu thêm việc nào khác không? Trả lời đúng một từ: CÓ hoặc KHÔNG."
    )


def is_yes(answer: str) -> bool:
    return clean_text(answer).upper().startswith("CÓ")


def describe(sample: dict, examples: list[tuple[str, list[dict]]], limit: int = 400) -> str:
    """Instruction for the local language model that writes the spoken request for this sample.
    `examples`: real (request, gold calls) pairs that show the register of real speakers (see style_examples)."""
    lines = [f"- {json.dumps(c, ensure_ascii=False)} -> {json.dumps(r, ensure_ascii=False)[:limit]}"
             for c, r in zip(sample["label"]["tool_calls"], sample["label"]["tool_results"], strict=True)]
    shots = "\n".join(f"- Câu nói: {t}\n  Thao tác: {_brief(c)[:limit]}" for t, c in examples)
    return (
        "Bạn viết câu mà NGƯỜI DÙNG nói trực tiếp với trợ lý giọng nói trên điện thoại (ngôi thứ nhất, câu ra lệnh hoặc nhờ vả).\n"
        f"Bây giờ là {sample['context']['current_time']}.\n"
        "Ví dụ câu nói thật của người dùng (chỉ để học CÁCH NÓI; nội dung của chúng không liên quan, đừng chép):\n" + shots + "\n"
        "Lần này trợ lý cần thực hiện đúng các thao tác sau (kèm kết quả trả về):\n" + "\n".join(lines) + "\n"
        "Viết một câu nói (có thể hai vế) dẫn tới ĐÚNG các thao tác ghi trên, không thêm việc khác. Nói tự nhiên, ngắn gọn, "
        "đời thường như các ví dụ; gọi người, đồ vật, sự kiện bằng tên trong kết quả, không bao giờ đọc mã ID, tên tool hay tên "
        "tham số; không xưng tên người dùng, không kể lại theo ngôi thứ ba. Chỉ in ra câu nói."
    )
