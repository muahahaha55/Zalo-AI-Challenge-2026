"""Step-by-step agent loop around the official ToolEnv.

generate(messages) -> str is any backend (HF, vLLM, oracle). The loop only uses ToolEnv.validate / call /
get_tool_calls / get_state_hash. Invalid calls are never executed, so they never reach the submission.
"""
from __future__ import annotations

import json
from typing import Callable

from .prompt import END, build_messages, call_text, dumps, result_message
from .rules import fix_args

MAX_STEPS = 8


def parse_call(text: str):
    """Return (name, arguments) or None."""
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`").removeprefix("json").strip()
    try:
        d = json.loads(t.splitlines()[0] if t else "")
    except Exception:
        try:
            d = json.loads(t)
        except Exception:
            return None
    if not isinstance(d, dict) or not isinstance(d.get("name"), str):
        return None
    args = d.get("arguments", {})
    return (d["name"], args) if isinstance(args, dict) else None


def run_sample(item: dict, env, catalog: dict, transcript: str, history_texts: list[str],
               generate: Callable[[list[dict]], str], max_steps: int = MAX_STEPS, retries: int = 1, use_rules: bool = True):
    msgs = build_messages(item, catalog, transcript, history_texts, with_target=False)
    seen, errors = set(), 0
    for _ in range(max_steps):
        out = generate(msgs).strip()
        if out.upper().startswith(END):
            break
        parsed = parse_call(out)
        if parsed is None:
            errors += 1
            if errors > retries:
                break
            msgs += [{"role": "assistant", "content": out}, {"role": "user", "content": "LỖI: đầu ra không phải JSON tool call hợp lệ."}]
            continue
        name, args = parsed
        if use_rules:                                  # Step 7: re-derive times / spoken phone numbers deterministically
            args, _ = fix_args(name, args, transcript, item["context"]["current_time"])
        key = dumps({"name": name, "arguments": args})
        v = env.validate(name, args)
        if not v.get("ok", False):
            errors += 1
            if errors > retries:
                break
            msgs += [{"role": "assistant", "content": out}, {"role": "user", "content": "LỖI: " + str(v.get("error"))}]
            continue
        if key in seen and catalog.get(name, {}).get("kind") != "write":   # repeated identical reads are never useful
            break                                                         # (gold does repeat some identical writes, e.g. redeem x2)
        seen.add(key)
        res = env.call(name, args)
        msgs += [{"role": "assistant", "content": call_text({"name": name, "arguments": args})},
                 {"role": "user", "content": result_message(res)}]
        if name == "sys.ask_missing_info":
            break
    return env.get_tool_calls(), env.get_state_hash()


class OracleBackend:
    """Replays the gold assistant turns of a train sample (harness self-test)."""

    def __init__(self, sample_msgs):
        self.targets = [m["content"] for m in sample_msgs if m["role"] == "assistant"]
        self.i = 0

    def __call__(self, messages):
        out = self.targets[self.i]
        self.i += 1
        return out
