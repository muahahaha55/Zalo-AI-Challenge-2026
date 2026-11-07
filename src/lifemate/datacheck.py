"""Pre-flight checks for SFT data, so a GPU run is not wasted on a data mistake.

Pure functions over the rows written by scripts/build_sft.py ({"id": ..., "messages": [...]}). The tokenizer is optional:
without it only the structural checks run.
"""
from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field

END = "END"


@dataclass
class FileReport:
    path: str
    rows: int = 0
    step_examples: int = 0            # one per assistant turn: what scripts/train_sft.py trains on
    too_long: int = 0                 # step examples that train_sft would silently drop (longer than max_len)
    longest: int = 0
    problems: list[str] = field(default_factory=list)


def check_conversation(row: dict) -> list[str]:
    """Structure the trainer and the inference loop rely on:
    system, user, then (assistant, user)* and a last assistant turn which is END or a lone sys.ask_missing_info;
    every other assistant turn is one JSON tool call (no END in the middle, ask only last)."""
    rid, msgs = row.get("id", "?"), row.get("messages") or []
    problems: list[str] = []
    if len(msgs) < 3:
        return [f"{rid}: fewer than 3 messages"]
    roles = [m.get("role") for m in msgs]
    if roles[:2] != ["system", "user"]:
        problems.append(f"{rid}: must start with system, user (got {roles[:2]})")
    for k in range(2, len(msgs)):
        if roles[k] != ("assistant" if k % 2 == 0 else "user"):
            problems.append(f"{rid}: role {roles[k]!r} at position {k} breaks the assistant/user alternation")
            break
    if roles[-1] != "assistant":
        problems.append(f"{rid}: does not end with an assistant turn")
    assistants = [m["content"] for m in msgs if m.get("role") == "assistant"]
    for i, content in enumerate(assistants):
        last = i == len(assistants) - 1
        if content == END:
            if not last:
                problems.append(f"{rid}: END before the last assistant turn")
            continue
        try:
            call = json.loads(content)
            ok = isinstance(call, dict) and isinstance(call.get("name"), str) and isinstance(call.get("arguments"), dict)
        except ValueError:
            ok = False
        if not ok:
            problems.append(f"{rid}: assistant turn {i} is neither END nor a JSON tool call")
        elif call["name"] == "sys.ask_missing_info" and not last:
            problems.append(f"{rid}: sys.ask_missing_info must be the last call")
    if assistants and assistants[-1] != END and "sys.ask_missing_info" not in assistants[-1]:
        problems.append(f"{rid}: ends with a tool call that is not sys.ask_missing_info (END is missing)")
    return problems


def base_id(row_id: str) -> str:
    """Sample id without the augmentation suffix: 'train_0007#aug0' -> 'train_0007'."""
    return row_id.split("#", 1)[0]


def step_token_lengths(messages: list[dict], tokenizer) -> list[int]:
    """Token length of every training step (prompt up to the turn + the turn itself + the end token), the same way
    scripts/train_sft.py builds its examples."""
    lengths = []
    for k, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        prompt = tokenizer.apply_chat_template(messages[:k], add_generation_prompt=True, tokenize=False, enable_thinking=False)
        target = m["content"] + "<|im_end|>"
        lengths.append(len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
                       + len(tokenizer(target, add_special_tokens=False)["input_ids"]))
    return lengths


def check_rows(path: str, rows: Iterable[dict], dev_ids: set[str] | None = None, hold_apps: set[str] | None = None,
               tokenizer=None, max_len: int | None = None) -> FileReport:
    rep = FileReport(path)
    seen: set[str] = set()
    pattern = re.compile(r"\b(" + "|".join(map(re.escape, sorted(hold_apps))) + r")\.") if hold_apps else None
    for row in rows:
        rep.rows += 1
        rid = row.get("id", "?")
        if rid in seen:
            rep.problems.append(f"{rid}: duplicate id in this file")
        seen.add(rid)
        rep.problems += check_conversation(row)
        if dev_ids is not None and base_id(rid) in dev_ids:
            rep.problems.append(f"{rid}: comes from a dev sample (leak)")
        if pattern is not None and any(pattern.search(m["content"]) for m in row["messages"]):
            rep.problems.append(f"{rid}: mentions a held-out app")
        rep.step_examples += sum(m.get("role") == "assistant" for m in row.get("messages", []))
        if tokenizer is not None and max_len:
            for n in step_token_lengths(row["messages"], tokenizer):
                rep.longest = max(rep.longest, n)
                rep.too_long += n > max_len
    return rep
