"""Training-data builders that need no model: every transformation is deterministic given a seed.

history_samples: turn each earlier (user, assistant) pair of `history` into its own SFT sample. The gold of those
turns is already in the data (assistant tool_calls + tool_results), is schema-valid and uses installed apps, but it is
unused by the main samples. Only TRAIN-split samples may be passed in (never dev).
"""
from __future__ import annotations

import json
import random

from .prompt import build_messages


def _turns(item: dict):
    """Yield (user_index_in_history, user_turn, assistant_turn) for each complete earlier exchange."""
    h = item["history"]
    for k in range(len(h) - 1):
        if h[k]["role"] == "user" and h[k + 1]["role"] == "assistant":
            yield k, h[k], h[k + 1]


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
    for it, k, u, a, calls, results in chosen:
        use_asr = bool(asr) and rng.random() < asr_share
        hist = it["history"][:k]                          # earlier turns only
        pick = lambda turn: asr[turn["audio"]] if use_asr else turn["audio_text"]
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
