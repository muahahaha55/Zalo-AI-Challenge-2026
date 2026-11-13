import copy
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lifemate.masking import mask_sample, masked_samples
from lifemate.prompt import END


def _tool(name, desc="", props=None):
    return {"name": name, "app": name.split(".")[0], "kind": "read", "description": desc,
            "parameters": {"type": "object", "properties": props or {}}}


CAT = {
    "contacts.search": _tool("contacts.search", "Tìm danh bạ", {"query": {"type": "string"}}),
    "chat.send": _tool("chat.send", "Gửi tin; lấy id từ contacts.search trước",
                       {"contact_id": {"type": "string", "x-ref": "contacts.contacts"}, "text": {"type": "string"}}),
    "chat.read": _tool("chat.read", "Đọc tin"),
    "music.play": _tool("music.play", "Phát nhạc"),
    "sys.ask_missing_info": _tool("sys.ask_missing_info", "Hỏi lại", {"tool_name": {"type": "string"}, "params": {"type": "array"}}),
}
REAL_WORDS = {"contacts", "chat", "music", "search", "send", "read", "play"}


def sample(sid="s1", calls=None, history=None):
    calls = calls if calls is not None else [{"name": "contacts.search", "arguments": {"query": "Lan"}},
                                             {"name": "chat.send", "arguments": {"contact_id": "c1", "text": "hi"}}]
    return {"id": sid, "audio": f"audio/{sid}.wav", "audio_text": "nhắn Lan hi", "history": history or [],
            "context": {"current_time": "2026-11-16T10:00:00+07:00", "timezone": "Asia/Ho_Chi_Minh", "location": {}, "user_profile": {},
                        "installed_apps": {"contacts": {"tools": ["contacts.search"]}, "chat": {"tools": ["chat.send", "chat.read"]},
                                           "music": {"tools": ["music.play"]}}},
            "label": {"tool_calls": calls, "tool_results": [{"ok": True}] * len(calls)}}


def _words(text):
    return set(re.findall(r"[a-z_]+", text))


def test_names_are_masked_consistently_and_input_is_untouched():
    it = sample()
    before = copy.deepcopy(it)
    new, cat, alias = mask_sample(it, CAT, random.Random(0))
    assert it == before                                                    # no mutation of the original
    assert set(alias) == {"contacts.search", "chat.send", "chat.read", "music.play"}
    assert [c["name"] for c in new["label"]["tool_calls"]] == [alias["contacts.search"], alias["chat.send"]]
    assert new["label"]["tool_calls"][1]["arguments"] == {"contact_id": "c1", "text": "hi"}   # arguments unchanged
    assert set(new["context"]["installed_apps"]) == {a.split(".")[0] for a in alias.values()}
    assert alias["chat.send"].split(".")[0] == alias["chat.read"].split(".")[0]           # one alias per app
    assert set(cat) == set(alias.values()) | {"sys.ask_missing_info"}
    send = cat[alias["chat.send"]]
    assert send["parameters"]["properties"]["contact_id"]["x-ref"] == alias["contacts.search"].split(".")[0] + ".contacts"
    assert alias["contacts.search"] in send["description"] and "contacts.search" not in send["description"]
    assert set(send["parameters"]["properties"]) == {"contact_id", "text"}                  # parameter names kept


def test_no_real_app_or_tool_word_left_in_the_prompt():
    hist = [{"role": "user", "audio": "audio/h.wav", "audio_text": "mở nhạc"},
            {"role": "assistant", "text": "ok", "tool_calls": [{"name": "music.play", "arguments": {}}], "tool_results": [{"ok": True}]}]
    rows, st = masked_samples([sample(history=hist)], CAT, {}, random.Random(1), 5)
    assert st["masked"] == 1 and rows[0]["id"] == "s1#mask"
    text = "\n".join(m["content"] for m in rows[0]["messages"] if m["role"] != "system")
    refs = re.findall(r"ref=([a-z0-9]+)\.([a-z_]+)", text)
    assert refs and all(app not in REAL_WORDS for app, _ in refs)          # x-ref: app part masked ...
    text = re.sub(r"ref=[a-z0-9]+\.[a-z_]+", "ref=", text)                 # ... collection word kept on purpose (it is meaning, like a description)
    leaked = _words(text) & REAL_WORDS
    assert not leaked, leaked


def test_ask_tool_name_follows_the_alias():
    ask = [{"name": "sys.ask_missing_info", "arguments": {"tool_name": "chat.send", "params": ["text"]}}]
    new, _, alias = mask_sample(sample(calls=ask), CAT, random.Random(2))
    c = new["label"]["tool_calls"][0]
    assert c["name"] == "sys.ask_missing_info" and c["arguments"] == {"tool_name": alias["chat.send"], "params": ["text"]}
    rows, _ = masked_samples([sample(calls=ask)], CAT, {}, random.Random(2), 1)
    assert rows[0]["messages"][-1]["content"] != END                       # an ask sample still ends on the ask


def test_targets_round_trip_and_end_with_END():
    rows, _ = masked_samples([sample()], CAT, {}, random.Random(3), 1)
    msgs = rows[0]["messages"]
    assistants = [m["content"] for m in msgs if m["role"] == "assistant"]
    assert assistants[-1] == END
    calls = [json.loads(x) for x in assistants[:-1]]
    tool_lines = msgs[1]["content"].split("[TOOL]\n", 1)[1].split("\n[LỊCH SỬ]", 1)[0].splitlines()
    listed = {line.split("(", 1)[0] for line in tool_lines}
    assert all(c["name"] in listed for c in calls)                        # every target names a tool listed in the prompt


def test_aliases_differ_between_samples_and_are_reproducible():
    items = [sample("a"), sample("b")]
    r1, _ = masked_samples(items, CAT, {}, random.Random(7), 2)
    r2, _ = masked_samples(items, CAT, {}, random.Random(7), 2)
    assert r1 == r2                                                         # same seed, same rows
    by_id = {r["id"]: r for r in r1}
    first = lambda r: json.loads(next(m["content"] for m in r["messages"] if m["role"] == "assistant"))["name"]  # noqa: E731
    assert first(by_id["a#mask"]) != first(by_id["b#mask"])                 # a fresh alias table per sample


def test_asr_text_and_count_limit():
    items = [sample(f"s{i}") for i in range(10)]
    asr = {f"audio/s{i}.wav": f"asr {i}" for i in range(10)}
    rows, st = masked_samples(items, CAT, asr, random.Random(4), 6)
    assert len(rows) == 6 and st["masked"] == 6 and len({r["id"] for r in rows}) == 6
    assert 0 < st["asr"] < 6                                                # mix of ASR and clean text (seeded)
    assert sum(r["messages"][1]["content"].endswith("[YÊU CẦU]\n" + f"asr {r['id'][1:-5]}") for r in rows) == st["asr"]


def test_unknown_tool_is_skipped():
    it = sample()
    it["context"]["installed_apps"]["x"] = {"tools": ["x.unknown"]}
    rows, st = masked_samples([it], CAT, {}, random.Random(0), 1)
    assert rows == [] and st["skipped_unknown_tool"] == 1
