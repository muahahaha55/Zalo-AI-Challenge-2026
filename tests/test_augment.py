import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lifemate.augment import history_samples
from lifemate.prompt import END

CAT = {
    "a.read": {"name": "a.read", "kind": "read", "description": "", "parameters": {"type": "object", "properties": {}}},
    "a.write": {"name": "a.write", "kind": "write", "description": "", "parameters": {"type": "object", "properties": {"x": {"type": "string"}}}},
    "sys.ask_missing_info": {"name": "sys.ask_missing_info", "kind": "meta", "description": "", "parameters": {"type": "object", "properties": {}}},
}


def item(i, hist):
    return {"id": i, "audio": f"audio/{i}/{i}.wav", "audio_text": "final", "history": hist,
            "context": {"current_time": "2026-11-16T10:00:00+07:00", "timezone": "Asia/Ho_Chi_Minh", "location": {}, "user_profile": {},
                        "installed_apps": {"a": {"tools": ["a.read", "a.write"]}}},
            "label": {"tool_calls": [], "tool_results": []}}


def turn(n, text, calls):
    return [{"role": "user", "audio": f"audio/s/s_u{n}.wav", "audio_text": text},
            {"role": "assistant", "text": "ok", "tool_calls": calls, "tool_results": [{"ok": True}] * len(calls)}]


C1, C2 = {"name": "a.read", "arguments": {}}, {"name": "a.write", "arguments": {"x": "1"}}


def test_each_exchange_becomes_a_sample_with_only_earlier_history():
    hist = turn(1, "u1", [C1]) + turn(2, "u2", [C1, C2]) + turn(3, "u3", [])
    rows, st = history_samples([item("s", hist)], CAT, {}, random.Random(0), 10)
    assert len(rows) == 3 and st["selected"] == 3
    by = {r["id"]: r["messages"] for r in rows}
    m2 = by["s#h1"]                                   # second exchange
    assert "u1" in m2[1]["content"] and "[YÊU CẦU]\nu2" in m2[1]["content"] and "u3" not in m2[1]["content"]
    assert [x["role"] for x in m2[2:]] == ["assistant", "user", "assistant", "user", "assistant"] and m2[-1]["content"] == END
    assert by["s#h2"][-1]["content"] == END and len(by["s#h2"]) == 3   # no call -> END at once


def test_quotas_do_not_starve_any_bucket():
    hist = []
    n = 0
    for calls, count in (([C1, C2], 10), ([], 10), ([C2], 10), ([C1], 10)):
        for _ in range(count):
            n += 1
            hist += turn(n, f"t{n}", calls)
    rows, st = history_samples([item("s", hist)], CAT, {}, random.Random(1), 20)
    assert st["taken"] == {"multi": 8, "none": 4, "one_write": 5, "one_read": 3} and len(rows) == 20


def test_short_bucket_hands_quota_to_others():
    hist = turn(1, "a", [C1, C2]) + turn(2, "b", [C2]) + turn(3, "c", [C2]) + turn(4, "d", [C1]) + turn(5, "e", [C1])
    rows, st = history_samples([item("s", hist)], CAT, {}, random.Random(1), 4)
    assert len(rows) == 4 and st["taken"]["none"] == 0 and st["taken"]["multi"] == 1


def test_asr_text_used_for_request_and_history():
    hist = turn(1, "clean1", [C1]) + turn(2, "clean2", [C1])
    asr = {"audio/s/s_u1.wav": "ASR1", "audio/s/s_u2.wav": "ASR2"}
    rows, _ = history_samples([item("s", hist)], CAT, asr, random.Random(0), 10, asr_share=1.0)
    text = {r["id"]: r["messages"][1]["content"] for r in rows}["s#h1"]
    assert "ASR1" in text and "[YÊU CẦU]\nASR2" in text and "clean" not in text


def test_unknown_tool_is_skipped():
    bad = turn(1, "u", [{"name": "zzz.nope", "arguments": {}}])
    rows, _ = history_samples([item("s", bad)], CAT, {}, random.Random(0), 5)
    assert rows == []
