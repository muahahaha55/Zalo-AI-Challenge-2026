"""Agent loop behaviour with a scripted backend and a fake env (no data, no GPU)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lifemate.agent import Agent, AgentConfig, StopReason, parse_call, run_sample  # noqa: E402

CAT = {n: {"name": n, "kind": k, "description": "", "parameters": {"type": "object", "properties": {}}}
       for n, k in (("a.read", "read"), ("a.write", "write"), ("sys.ask_missing_info", "meta"))}
ITEM = {"id": "s", "audio": "a.wav", "audio_text": "x", "history": [],
        "context": {"current_time": "2026-11-16T10:00:00+07:00", "timezone": "Asia/Ho_Chi_Minh", "location": {}, "user_profile": {},
                    "installed_apps": {"a": {"tools": ["a.read", "a.write"]}}}}
READ = '{"name":"a.read","arguments":{}}'
WRITE = '{"name":"a.write","arguments":{"x":"1"}}'
WRITE2 = '{"name":"a.write","arguments":{"x":"2"}}'
ASK = '{"name":"sys.ask_missing_info","arguments":{"tool_name":"a.write","params":["x"]}}'


class FakeEnv:
    def __init__(self, bad_tools=()):
        self.calls, self.bad = [], set(bad_tools)

    def validate(self, name, args):
        return {"ok": False, "error": "bad"} if name in self.bad else {"ok": True}

    def call(self, name, args):
        self.calls.append({"name": name, "arguments": args})
        return {"ok": True}

    def get_tool_calls(self):
        return list(self.calls)

    def get_state_hash(self):
        return f"h{len(self.calls)}"


def scripted(*outputs):
    it = iter(outputs)
    return lambda messages: next(it)


def run(outputs, env=None, **cfg):
    env = env or FakeEnv()
    return Agent(CAT, scripted(*outputs), AgentConfig(use_rules=False, **cfg)).run(ITEM, env, "x", []), env


def test_end_stops_the_loop():
    out, _ = run([READ, "END"])
    assert out.stop_reason is StopReason.END and [c["name"] for c in out.tool_calls] == ["a.read"] and out.steps == 2


def test_ask_is_last_and_stops():
    out, _ = run([READ, ASK, "SHOULD NOT BE REACHED"])
    assert out.stop_reason is StopReason.ASK and out.tool_calls[-1]["name"] == "sys.ask_missing_info"


def test_malformed_output_gets_one_correction_then_stops():
    out, _ = run(["not json", READ, "END"])
    assert out.stop_reason is StopReason.END and len(out.tool_calls) == 1
    out, _ = run(["bad", "worse", READ])
    assert out.stop_reason is StopReason.PARSE_ERROR and out.tool_calls == []


def test_invalid_call_is_never_executed():
    out, env = run([WRITE, WRITE], env=FakeEnv(bad_tools={"a.write"}))
    assert out.stop_reason is StopReason.INVALID_CALL and env.calls == []


def test_repeated_read_stops_but_repeated_write_is_allowed():
    out, _ = run([READ, READ, "END"])
    assert out.stop_reason is StopReason.REPEATED_READ and len(out.tool_calls) == 1
    out, _ = run([WRITE, WRITE, "END"])
    assert out.stop_reason is StopReason.END and len(out.tool_calls) == 2


def test_max_steps():
    out, _ = run([READ, WRITE, WRITE2, "END"], max_steps=2)
    assert out.stop_reason is StopReason.MAX_STEPS and len(out.tool_calls) == 2


def test_backend_error_keeps_calls_made_so_far():
    state = {"n": 0}

    def backend(messages):
        state["n"] += 1
        if state["n"] == 1:
            return READ
        raise RuntimeError("CUDA out of memory")

    env = FakeEnv()
    out = Agent(CAT, backend, AgentConfig(use_rules=False)).run(ITEM, env, "x", [])
    assert out.stop_reason is StopReason.BACKEND_ERROR and "out of memory" in out.error
    assert [c["name"] for c in out.tool_calls] == ["a.read"] and out.state_hash == "h1"


def test_time_budget_stops_before_a_step_that_would_not_fit():
    now = [0.0]

    def backend(messages):
        now[0] += 4.0                                   # every generation takes 4 s
        return READ if now[0] < 5 else WRITE

    agent = Agent(CAT, backend, AgentConfig(use_rules=False, time_budget_s=10.0), clock=lambda: now[0])
    out = agent.run(ITEM, FakeEnv(), "x", [])
    # after 2 steps 8 s are used and the slowest step is 4 s: a third step would end at 12 s > 10 s
    assert out.stop_reason is StopReason.TIME_BUDGET and out.steps == 2 and out.elapsed_s == 8.0


def test_no_budget_means_no_limit():
    out, _ = run([READ, WRITE, "END"], time_budget_s=None)
    assert out.stop_reason is StopReason.END


def test_run_sample_wrapper_returns_tuple():
    env = FakeEnv()
    calls, state_hash = run_sample(ITEM, env, CAT, "x", [], scripted(READ, "END"), use_rules=False)
    assert [c["name"] for c in calls] == ["a.read"] and state_hash == "h1"


def test_parse_call_accepts_fenced_json_and_rejects_garbage():
    assert parse_call('```json\n{"name":"a.read","arguments":{}}\n```') == ("a.read", {})
    assert parse_call('{"name":"a.read"}') == ("a.read", {})
    assert parse_call("END") is None
    assert parse_call('{"arguments":{}}') is None
    assert parse_call('{"name":"a","arguments":[1]}') is None


def test_backend_that_accepts_a_deadline_gets_the_remaining_budget():
    now = [0.0]
    seen = []

    class Backend:
        accepts_deadline = True
        outputs = iter([READ, WRITE, "END"])

        def __call__(self, messages, max_time=None):
            seen.append(max_time)
            now[0] += 3.0
            return next(self.outputs)

    agent = Agent(CAT, Backend(), AgentConfig(use_rules=False, time_budget_s=20.0), clock=lambda: now[0])
    agent.run(ITEM, FakeEnv(), "x", [])
    assert seen == [20.0, 17.0, 14.0]               # remaining budget before each step


def test_deadline_is_not_passed_without_a_budget_or_to_plain_backends():
    seen = []

    class Backend:
        accepts_deadline = True

        def __call__(self, messages, max_time=None):
            seen.append(max_time)
            return "END"

    Agent(CAT, Backend(), AgentConfig(use_rules=False)).run(ITEM, FakeEnv(), "x", [])          # no budget: nothing to cap
    assert seen == [None]
    def plain(messages):                                  # no accepts_deadline: a stray max_time would raise TypeError -> BACKEND_ERROR
        return "END"

    out = Agent(CAT, plain, AgentConfig(use_rules=False, time_budget_s=5.0)).run(ITEM, FakeEnv(), "x", [])
    assert out.stop_reason is StopReason.END


def test_remaining_budget_never_drops_below_the_minimum():
    now = [0.0]
    seen = []

    class Backend:
        accepts_deadline = True

        def __call__(self, messages, max_time=None):
            seen.append(max_time)
            now[0] += 0.0
            return READ if len(seen) == 1 else "END"

    agent = Agent(CAT, Backend(), AgentConfig(use_rules=False, time_budget_s=0.2), clock=lambda: now[0])
    agent.run(ITEM, FakeEnv(), "x", [])
    assert seen[0] == 0.5
