"""Step-by-step agent loop around the official ToolEnv.

`generate(messages) -> str` is any backend (HF, vLLM, oracle). The loop only uses ToolEnv.validate / call /
get_tool_calls / get_state_hash. Invalid calls are never executed, so they never reach the submission.

Safety properties (a sample that times out or crashes scores 0, so the loop must always hand the env back):
  * time budget: no new generation starts when the time already used plus the slowest step so far would exceed it;
  * backend errors (CUDA OOM, tokenizer failures, ...) end the sample instead of raising, keeping every call made so far.
"""
from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from .prompt import END, build_messages, call_text, dumps, result_message
from .rules import fix_args

MAX_STEPS = 8
Generate = Callable[[list[dict]], str]


class StopReason(str, Enum):
    END = "end"                      # the model said END
    ASK = "ask"                      # sys.ask_missing_info was issued (always the last call)
    MAX_STEPS = "max_steps"
    PARSE_ERROR = "parse_error"      # output was not a JSON tool call, retries used up
    INVALID_CALL = "invalid_call"    # schema validation failed, retries used up
    REPEATED_READ = "repeated_read"  # the same read call again (never useful)
    TIME_BUDGET = "time_budget"
    BACKEND_ERROR = "backend_error"


@dataclass(frozen=True)
class AgentConfig:
    max_steps: int = MAX_STEPS
    retries: int = 1                 # corrections allowed after a malformed or invalid call
    use_rules: bool = True           # deterministic time / phone-number fixes (rules.py)
    time_budget_s: float | None = None


@dataclass
class Outcome:
    tool_calls: list[dict]
    state_hash: str
    stop_reason: StopReason
    steps: int = 0
    elapsed_s: float = 0.0
    error: str | None = None


def parse_call(text: str) -> tuple[str, dict] | None:
    """Return (name, arguments) or None."""
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`").removeprefix("json").strip()
    try:
        d = json.loads(t.splitlines()[0] if t else "")
    except Exception:  # noqa: BLE001 - any malformed output simply means "no call"
        try:
            d = json.loads(t)
        except Exception:  # noqa: BLE001
            return None
    if not isinstance(d, dict) or not isinstance(d.get("name"), str):
        return None
    args = d.get("arguments", {})
    return (d["name"], args) if isinstance(args, dict) else None


class Agent:
    """One instance per catalog and backend; `run` handles one sample on the (single) ToolEnv created for it."""

    def __init__(self, catalog: dict, generate: Generate, config: AgentConfig | None = None, clock: Callable[[], float] = time.monotonic):
        self.catalog = catalog
        self.generate = generate
        self.config = config or AgentConfig()
        self._clock = clock

    def run(self, item: dict, env, transcript: str, history_texts: list[str], generate: Generate | None = None) -> Outcome:
        cfg, gen = self.config, generate or self.generate
        msgs = build_messages(item, self.catalog, transcript, history_texts, with_target=False)
        start = self._clock()
        seen: set[str] = set()
        errors = steps = 0
        slowest = 0.0
        reason = StopReason.MAX_STEPS
        error: str | None = None

        for _ in range(cfg.max_steps):
            if cfg.time_budget_s is not None and (self._clock() - start) + slowest >= cfg.time_budget_s:
                reason = StopReason.TIME_BUDGET
                break
            t0 = self._clock()
            try:
                out = gen(msgs).strip()
            except Exception as exc:  # noqa: BLE001 - never lose the env: it already holds the calls made so far
                reason, error = StopReason.BACKEND_ERROR, f"{type(exc).__name__}: {exc}"
                break
            slowest = max(slowest, self._clock() - t0)
            steps += 1
            if out.upper().startswith(END):
                reason = StopReason.END
                break
            parsed = parse_call(out)
            if parsed is None:
                errors += 1
                if errors > cfg.retries:
                    reason = StopReason.PARSE_ERROR
                    break
                msgs += [{"role": "assistant", "content": out}, {"role": "user", "content": "LỖI: đầu ra không phải JSON tool call hợp lệ."}]
                continue
            name, args = parsed
            if cfg.use_rules:                          # Step 7: re-derive times / spoken phone numbers deterministically
                args, _ = fix_args(name, args, transcript, item["context"]["current_time"])
            key = dumps({"name": name, "arguments": args})
            check = env.validate(name, args)
            if not check.get("ok", False):
                errors += 1
                if errors > cfg.retries:
                    reason = StopReason.INVALID_CALL
                    break
                msgs += [{"role": "assistant", "content": out}, {"role": "user", "content": "LỖI: " + str(check.get("error"))}]
                continue
            if key in seen and self.catalog.get(name, {}).get("kind") != "write":   # repeated identical reads are never useful
                reason = StopReason.REPEATED_READ                                    # (gold does repeat some identical writes)
                break
            seen.add(key)
            result = env.call(name, args)
            msgs += [{"role": "assistant", "content": call_text({"name": name, "arguments": args})},
                     {"role": "user", "content": result_message(result)}]
            if name == "sys.ask_missing_info":
                reason = StopReason.ASK
                break
        return Outcome(env.get_tool_calls(), env.get_state_hash(), reason, steps, self._clock() - start, error)


def run_sample(item: dict, env, catalog: dict, transcript: str, history_texts: list[str], generate: Generate,
               max_steps: int = MAX_STEPS, retries: int = 1, use_rules: bool = True):
    """Backwards-compatible wrapper: returns (tool_calls, state_hash)."""
    agent = Agent(catalog, generate, AgentConfig(max_steps=max_steps, retries=retries, use_rules=use_rules))
    outcome = agent.run(item, env, transcript, history_texts)
    return outcome.tool_calls, outcome.state_hash


class OracleBackend:
    """Replays the gold assistant turns of a train sample (harness self-test)."""

    def __init__(self, sample_msgs: list[dict]):
        self.targets = [m["content"] for m in sample_msgs if m["role"] == "assistant"]
        self.i = 0

    def __call__(self, messages: list[dict]) -> str:
        out = self.targets[self.i]
        self.i += 1
        return out
