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


def test_split_holdout_hides_apps_everywhere():
    from lifemate.augment import split_holdout
    def it(i, gold_app, hist_app=None, apps=("a", "h", "k")):
        calls = [{"name": f"{gold_app}.t", "arguments": {}}] if gold_app else []
        hist = turn(1, "u", [{"name": f"{hist_app}.t", "arguments": {}}]) if hist_app else []
        x = item(i, hist)
        x["label"] = {"tool_calls": calls, "tool_results": []}
        x["context"]["installed_apps"] = {a: {"tools": [f"{a}.t"]} for a in apps}
        return x
    items = [it("s1", "a"), it("s2", "h"), it("s3", "a", hist_app="h"), it("s4", None), it("s5", "k", hist_app="a")]
    train, ev = split_holdout(items, {"h"})
    assert ev == ["s2"]                                       # gold uses a held app
    assert [t["id"] for t in train] == ["s1", "s4", "s5"]       # s3 dropped: its history called a held app
    assert all("h" not in t["context"]["installed_apps"] for t in train)
    assert "h" in items[0]["context"]["installed_apps"]         # inputs are not mutated


def test_gold_apps_counts_the_app_an_ask_is_about():
    from lifemate.augment import gold_apps
    x = item("s", [])
    x["label"] = {"tool_calls": [{"name": "contacts.search", "arguments": {}},
                                 {"name": "sys.ask_missing_info", "arguments": {"tool_name": "phone.call", "params": ["contact_id"]}}],
                  "tool_results": []}
    assert gold_apps(x) == {"contacts", "phone"}
    x["label"]["tool_calls"] = [{"name": "sys.ask_missing_info", "arguments": {"tool_name": "flights.search", "params": ["date"]}}]
    assert gold_apps(x) == {"flights"}


def _plain_cat(apps):
    cat = {f"{a}.t": {"name": f"{a}.t", "kind": "read", "description": "", "parameters": {"type": "object", "properties": {}}} for a in apps}
    cat["sys.ask_missing_info"] = CAT["sys.ask_missing_info"]
    return cat


def test_variants_never_drop_the_app_an_ask_is_about():
    from lifemate.augment import synthetic_samples
    x = item("s", [])
    x["label"] = {"tool_calls": [{"name": "sys.ask_missing_info", "arguments": {"tool_name": "b.t", "params": ["x"]}}], "tool_results": []}
    x["context"]["installed_apps"] = {a: {"tools": [f"{a}.t"]} for a in "abcdefgh"}
    rows, _ = synthetic_samples([x], _plain_cat("abcdefgh"), {}, random.Random(3), 0, 60)
    assert len(rows) == 60
    assert all("b.t(" in r["messages"][1]["content"] for r in rows)     # the tool the ask refers to is always still listed


def test_oos_samples_end_immediately_and_hide_all_gold_apps():
    from lifemate.augment import synthetic_samples
    x = item("s", [])
    x["label"] = {"tool_calls": [{"name": "a.t", "arguments": {}}], "tool_results": [{"ok": True}]}
    x["context"]["installed_apps"] = {a: {"tools": [f"{a}.t"]} for a in "abc"}
    rows, st = synthetic_samples([x], _plain_cat("abc"), {}, random.Random(0), 1, 0)
    assert st["oos"] == 1 and [m["role"] for m in rows[0]["messages"]] == ["system", "user", "assistant"]
    assert rows[0]["messages"][-1]["content"] == "END" and "a.t(" not in rows[0]["messages"][1]["content"]
