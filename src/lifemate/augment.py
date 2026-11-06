"""Training-data builders that need no model: every transformation is deterministic given a seed.

history_samples: turn each earlier (user, assistant) pair of `history` into its own SFT sample. The gold of those
turns is already in the data (assistant tool_calls + tool_results), is schema-valid and uses installed apps, but it is
unused by the main samples. Only TRAIN-split samples may be passed in (never dev).
"""
from __future__ import annotations

import json
import random
import re

from .prompt import build_messages


def _turns(item: dict):
    """Yield (user_index_in_history, user_turn, assistant_turn) for each complete earlier exchange."""
    h = item["history"]
    for k in range(len(h) - 1):
        if h[k]["role"] == "user" and h[k + 1]["role"] == "assistant":
            yield k, h[k], h[k + 1]


def _text_picker(asr: dict, use_asr: bool):
    """Function giving the text of a user turn: its ASR transcript (noisy, as at inference) or the clean gold text."""
    if use_asr:
        return lambda turn: asr[turn["audio"]]
    return lambda turn: turn["audio_text"]


QUOTAS = {"multi": 0.40, "none": 0.20, "one_write": 0.25, "one_read": 0.15}


def history_samples(train_items: list[dict], cat: dict, asr: dict, rng: random.Random, max_samples: int,
                    quotas: dict | None = None, asr_share: float = 0.5):
    """Return (rows, stats). Turns are bucketed by call pattern and drawn with fixed shares (QUOTAS) so that one bucket
    cannot starve the others; a bucket that is short hands its unused quota to the remaining buckets (in QUOTAS order).
    A row's transcripts are the ASR text with probability asr_share (when `asr` is given), else the clean audio_text."""
    quotas = quotas or QUOTAS
    cand = {k: [] for k in quotas}
    for it in train_items:
        for k, u, a in _turns(it):
            calls, results = a.get("tool_calls") or [], a.get("tool_results") or []
            if len(calls) != len(results) or any(c["name"] not in cat for c in calls):
                continue                                  # defensive; the real data has none of these
            if len(calls) >= 2:
                key = "multi"
            elif not calls:
                key = "none"
            else:
                key = "one_write" if cat[calls[0]["name"]].get("kind") == "write" else "one_read"
            cand[key].append((it, k, u, a, calls, results))
    for v in cand.values():
        rng.shuffle(v)
    take = {key: min(len(cand[key]), int(round(max_samples * q))) for key, q in quotas.items()}
    spare = max_samples - sum(take.values())
    for key in quotas:                                    # redistribute what short buckets could not fill
        extra = min(spare, len(cand[key]) - take[key])
        take[key] += max(0, extra)
        spare -= max(0, extra)
    chosen = [x for key in quotas for x in cand[key][: take[key]]]
    rows = []
    for it, k, u, _assistant, calls, results in chosen:
        use_asr = bool(asr) and rng.random() < asr_share
        hist = it["history"][:k]                          # earlier turns only
        pick = _text_picker(asr, use_asr)
        htexts = [pick(t) for t in hist if t["role"] == "user"]
        sample = {**it, "history": hist, "label": {"tool_calls": calls, "tool_results": results}}
        rows.append({"id": f"{it['id']}#h{k // 2}", "messages": build_messages(sample, cat, pick(u), htexts)})
    rng.shuffle(rows)
    stats = {key: len(v) for key, v in cand.items()}
    stats["taken"] = take
    stats["selected"] = len(rows)
    return rows, stats


def gold_apps(item: dict) -> set:
    """Apps the gold answer depends on. sys.ask_missing_info names the tool it asks about, and that app counts too:
    it must stay installed (and is 'used' by the sample) or the ask would point at a tool that does not exist."""
    apps = set()
    for c in item["label"]["tool_calls"]:
        name = c["name"]
        if name == "sys.ask_missing_info":
            name = str((c.get("arguments") or {}).get("tool_name", ""))
        apps.add(name.split(".")[0])
    return apps - {"sys", ""}


def history_apps(item: dict) -> set:
    return {c["name"].split(".")[0] for h in item["history"] if h["role"] == "assistant"
            for c in (h.get("tool_calls") or [])} - {"sys"}


def split_holdout(items: list[dict], hold_apps: set):
    """'Unseen apps' experiment. Returns (train_items, eval_ids).
    eval_ids: every sample whose gold calls use a held app -> evaluated by a model that never saw those tools.
    train_items: the remaining samples with the held apps REMOVED from installed_apps, and dropped entirely if their
    history already called a held app (so no trace of a held tool is left anywhere in the training prompts)."""
    eval_ids = [it["id"] for it in items if gold_apps(it) & hold_apps]
    ev = set(eval_ids)
    train = []
    for it in items:
        if it["id"] in ev or history_apps(it) & hold_apps:
            continue
        new = json.loads(json.dumps(it))   # plain-data copy of a sample (no env involved)
        new["context"]["installed_apps"] = {a: v for a, v in it["context"]["installed_apps"].items() if a not in hold_apps}
        train.append(new)
    return train, eval_ids


def with_apps(item: dict, apps: dict) -> dict:
    """Copy of a sample whose context lists exactly `apps` (an ordered dict app -> {'tools': [...]})."""
    new = json.loads(json.dumps(item))                    # plain-data copy of a sample (no env involved)
    new["context"]["installed_apps"] = apps
    return new


def synthetic_samples(train_items: list[dict], cat: dict, asr: dict, rng: random.Random, n_oos: int, n_variants: int):
    """Train-split augmentation (never call it with dev samples). Returns (rows, stats).

    oos      : a history-free exec sample whose gold apps are ALL removed from installed_apps; target = END at once
               (no installed app can serve the request any more).
    variants : per sample, shuffled app/tool order with random irrelevant apps dropped. Gold apps are always kept, and for
               `ask` samples that includes the app the question is about (see gold_apps), same targets as the original.
    The order of rng calls is part of the contract: it makes the output reproducible for a given seed."""
    rows: list[dict] = []
    stats = {"oos": 0, "variants": 0}

    def add(it: dict, tag: str, calls: list | None = None) -> None:
        use_asr = bool(asr) and rng.random() < 0.5
        pick = _text_picker(asr, use_asr)
        sample = it if calls is None else {**it, "label": {**it["label"], "tool_calls": calls, "tool_results": []}}
        history = [pick(h) for h in it["history"] if h["role"] == "user"]
        rows.append({"id": f"{it['id']}#{tag}", "messages": build_messages(sample, cat, pick(it), history)})

    execs = [it for it in train_items if it["label"]["tool_calls"] and not it["history"]
             and all(c["name"] != "sys.ask_missing_info" for c in it["label"]["tool_calls"])]
    rng.shuffle(execs)
    for it in execs:
        if stats["oos"] >= n_oos:
            break
        used = gold_apps(it)
        left = {a: v for a, v in it["context"]["installed_apps"].items() if a not in used}
        if not used or not left:
            continue
        keys = list(left)
        rng.shuffle(keys)
        add(with_apps(it, {a: left[a] for a in keys}), "oos", calls=[])
        stats["oos"] += 1
    for it in train_items:
        for v in range(n_variants):
            used = gold_apps(it)
            others = [a for a in it["context"]["installed_apps"] if a not in used]
            kept = [a for a in others if rng.random() < 0.7]
            keys = [a for a in it["context"]["installed_apps"] if a in used or a in kept]
            rng.shuffle(keys)
            apps = {}
            for a in keys:
                tools = list(it["context"]["installed_apps"][a]["tools"])
                rng.shuffle(tools)
                apps[a] = {**it["context"]["installed_apps"][a], "tools": tools}
            add(with_apps(it, apps), f"aug{v}")
            stats["variants"] += 1
    rng.shuffle(rows)
    return rows, stats


# How speakers retract in the real data (98 train samples, 6.3%): a request, a marker, then either the request that
# counts or "never mind". Markers below are taken from those samples.
RETRACT_MARKERS = ["à thôi", "mà thôi", "à khoan", "ờ mà khoan", "à không", "thôi", "ờ thôi"]
CANCEL_ENDINGS = ["khỏi", "thôi khỏi", "khỏi làm nữa", "không cần nữa", "thôi bỏ đi"]
_TRAILING = " .,;!?…"
# markers of a retraction / negation that make the scope of a composed 'never mind' ambiguous, plus the real-data pattern
_AMBIGUOUS = re.compile(r"à khoan|à thôi|mà thôi|\bthôi\b|khỏi|à không|không phải|nhầm|\bkhoan\b|\bđừng\b", re.IGNORECASE)
REAL_RETRACTION = re.compile(r"à khoan|à thôi|mà thôi|thôi khỏi|khỏi|không phải", re.IGNORECASE)
MAX_RETRACTED_WORDS = 25


def is_simple_request(text: str, max_words: int = MAX_RETRACTED_WORDS) -> bool:
    """One short clause with no retraction/negation of its own: safe to use as the 'taken back' part of a composition,
    because 'never mind' can only mean that whole request."""
    t = text.strip().rstrip(_TRAILING)
    return bool(t) and len(t.split()) <= max_words and "," not in t and not _AMBIGUOUS.search(t)


def _retractable(it: dict, pool_apps: set | None = None) -> bool:
    calls = it["label"]["tool_calls"]
    ok = bool(calls) and not it["history"] and all(c["name"] != "sys.ask_missing_info" for c in calls)
    return ok and (pool_apps is None or gold_apps(it) <= pool_apps)


def retraction_samples(train_items: list[dict], cat: dict, rng: random.Random, n_replace: int, n_cancel: int):
    """Compositions that teach 'ignore what the speaker took back'. TRAIN samples only. Text is clean (no audio exists
    for a composed sentence), history-free samples only. Returns (rows, stats).

    replace : '<request B>, <marker>, <request A>'  -> gold of A. B was actionable in A's context (its apps are installed
              there) but is retracted, so none of B's calls may appear in the answer.
    cancel  : '<request B>, <marker> <cancel>'       -> no tool call at all (END), the whole request was withdrawn.
    Each composition stays on A's (or B's) own context, so ids, tool results and the answer remain consistent."""
    base = [it for it in train_items if _retractable(it) and is_simple_request(it["audio_text"])]
    rows: list[dict] = []
    stats = {"replace": 0, "cancel": 0}

    def clause(it: dict) -> str:
        return it["audio_text"].strip().rstrip(_TRAILING)

    def add(it: dict, text: str, label: dict, tag: str) -> None:
        sample = {**it, "label": label}
        rows.append({"id": f"{it['id']}#{tag}", "messages": build_messages(sample, cat, text, [])})

    answerable = [it for it in train_items if it["label"]["tool_calls"] and not it["history"]     # A may also be an ask sample
                  and not _AMBIGUOUS.search(it["audio_text"])]
    rng.shuffle(answerable)
    for a in answerable:
        if stats["replace"] >= n_replace:
            break
        installed = set(a["context"]["installed_apps"])
        partners = [b for b in base if b["id"] != a["id"] and gold_apps(b) <= installed]
        if not partners:
            continue
        b = rng.choice(partners)
        text = f"{clause(b)}, {rng.choice(RETRACT_MARKERS)}, {a['audio_text'].strip()}"
        add(a, text, a["label"], f"retr{stats['replace']}")
        stats["replace"] += 1
    shuffled = list(base)
    rng.shuffle(shuffled)
    for b in shuffled:
        if stats["cancel"] >= n_cancel:
            break
        text = f"{clause(b)}, {rng.choice(RETRACT_MARKERS)} {rng.choice(CANCEL_ENDINGS)}"
        add(b, text, {"tool_calls": [], "tool_results": []}, f"cancel{stats['cancel']}")
        stats["cancel"] += 1
    rng.shuffle(rows)
    return rows, stats


def retraction_oversample(train_items: list[dict], cat: dict, asr: dict, rng: random.Random, copies: int):
    """Real self-correction samples (6% of train), repeated `copies` times. Labels are the real ones, so nothing can be mislabelled;
    copy 0 keeps the clean text and later copies use the ASR text when available, so repeats are not identical."""
    rows = []
    for it in train_items:
        if not REAL_RETRACTION.search(it["audio_text"]):
            continue
        for c in range(copies):
            use_asr = bool(asr) and c % 2 == 1
            pick = _text_picker(asr, use_asr)
            history = [pick(h) for h in it["history"] if h["role"] == "user"]
            rows.append({"id": f"{it['id']}#rx{c}", "messages": build_messages(it, cat, pick(it), history)})
    rng.shuffle(rows)
    return rows
