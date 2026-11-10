import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lifemate import callgen as cg

CAT = {
    "cal.list": {"name": "cal.list", "kind": "read", "parameters": {"properties": {"day": {"type": "string"}}}},
    "cal.cancel": {"name": "cal.cancel", "kind": "write", "parameters": {"properties": {"event_id": {"type": "string", "x-ref": "cal.events"},
                                                                       "mode": {"enum": ["a", "b"]}}}},
    "sms.send": {"name": "sms.send", "kind": "write", "parameters": {"properties": {"to": {"type": "string", "x-ref": "cal.x"}}}},
}


class FakeEnv:
    """cal.list returns the events of this environment; cal.cancel only accepts known ids."""
    def __init__(self, ids):
        self.ids = ids
        self.log = []

    def validate(self, name, args):
        return {"ok": True}

    def get_state_hash(self):
        return str(len(self.log))             # every call changes the state in this fake

    def call(self, name, args):
        self.log.append((name, args))
        if name == "cal.list":
            return {"ok": True, "events": [{"id": i} for i in self.ids]}
        if name == "cal.cancel":
            return {"ok": args["event_id"] in self.ids}
        return {"ok": True}


def item(iid, ids, day, tools):
    return {"id": iid, "history": [], "context": {"current_time": f"{day}T08:00:00+07:00",
            "installed_apps": {"cal": {"tools": ["cal.list", "cal.cancel"]}, "sms": {"tools": ["sms.send"]}}},
            "label": {"tool_calls": [{"name": "cal.list", "arguments": {"day": f"{day}"}},
                                     {"name": "cal.cancel", "arguments": {"event_id": "evt_1", "mode": "a"}}],
                      "tool_results": [{"ok": True, "events": [{"id": "evt_1"}, {"id": "evt_2"}]}, {"ok": True}]}}


def test_find_path_and_values():
    res = [{"ok": True, "events": [{"id": "x"}, {"id": "y"}]}]
    assert cg.find_path("y", res) == ("events", "id")
    assert cg.find_path("zzz", res) is None
    assert cg.values_at(("events", "id"), res) == ["x", "y"]


def test_shift_dates():
    assert cg.shift_dates("2026-11-28T09:00:00+07:00", 2) == "2026-11-30T09:00:00+07:00"
    assert cg.shift_dates("hello", 2) == "hello"
    assert cg.shift_dates(["2026-11-30"], -1) == ["2026-11-29"]


def test_segments_group_by_app_and_stop_at_ask():
    it = item("a", ["e1"], "2026-11-01", None)
    it["label"]["tool_calls"] += [{"name": "sms.send", "arguments": {}}, {"name": "sys.ask_missing_info", "arguments": {}}]
    assert [len(s) for s in cg.segments(it, CAT)] == [2, 1]


def test_segments_merge_cross_app_dependency_and_drop_cut_reads():
    it = item("a", ["e1"], "2026-11-01", None)
    it["label"]["tool_calls"] = [{"name": "cal.list", "arguments": {}}, {"name": "sms.send", "arguments": {"to": "evt_2"}}]
    it["label"]["tool_results"] = [{"ok": True, "events": [{"id": "evt_2"}]}, {"ok": True}]
    assert [[c["name"] for c, _ in seg] for seg in cg.segments(it, CAT)] == [["cal.list", "sms.send"]]
    it["label"]["tool_calls"][1]["arguments"] = {}                 # no dependency: the lone read is a cut chain -> dropped
    assert [[c["name"] for c, _ in seg] for seg in cg.segments(it, CAT)] == [["sms.send"]]
    it["label"]["tool_calls"] = it["label"]["tool_calls"][:1]       # a read that IS the whole answer is kept
    assert [[c["name"] for c, _ in seg] for seg in cg.segments(it, CAT)] == [["cal.list"]]


def test_text_filter():
    s = item("a", ["e1"], "2026-11-01", None)
    s["context"]["user_profile"] = {"name": "Anh Nam"}
    assert cg.text_ok("hủy giúp tôi lịch đá bóng chiều nay", s, CAT) == (True, "ok")
    assert cg.text_ok("hủy lịch evt_1 giúp tôi nhé", s, CAT)[1] == "id_or_tool_name"
    assert cg.text_ok("Anh Nam muốn hủy lịch đá bóng", s, CAT)[1] == "third_person"
    assert cg.text_ok("hủy", s, CAT)[1] == "length"
    s["label"]["tool_calls"][0]["arguments"]["day"] = "đá bóng chiều"   # free text the speaker must have said
    assert cg.text_ok("hủy giúp tôi lịch đá bóng chiều nay", s, CAT) == (True, "ok")
    assert cg.text_ok("đặt báo thức sáu giờ sáng mai nhé", s, CAT)[1] == "missing_args"
    assert cg.clean_text('<think>x</think>\n "hủy lịch nhé"\n giải thích') == "hủy lịch nhé"


def test_transplant_rebinds_ids_to_base_environment():
    donor = item("d", ["evt_1", "evt_2"], "2026-11-01", None)
    base = item("b", ["zz_7"], "2026-11-11", None)
    env = FakeEnv(["zz_7"])
    out = cg.transplant(cg.segments(donor, CAT)[0], donor, base, env, [], CAT, random.Random(1))
    assert out is not None
    calls, results = out
    assert calls[1]["arguments"]["event_id"] == "zz_7"          # id comes from the base environment, not the donor
    assert calls[0]["arguments"]["day"] == "2026-11-11"          # date shifted by the 10-day difference
    assert all(r["ok"] for r in results)


def test_transplant_fails_without_a_matching_read():
    donor = item("d", ["evt_1"], "2026-11-01", None)
    seg = [cg.segments(donor, CAT)[0][1]]                              # only the cancel, its read is missing
    assert cg.transplant(seg, donor, donor, FakeEnv(["evt_1"]), [], CAT, random.Random(1)) is None


def test_generate_is_reproducible_and_labels_replay():
    items = [item("a", ["e1", "e2"], "2026-11-01", None), item("b", ["f1"], "2026-11-09", None)]
    run = lambda: cg.generate(items, CAT, lambda i: FakeEnv(["e1", "e2"] if i == "a" else ["f1"]), random.Random(5), 4)  # noqa: E731
    (s1, _), (s2, _) = run(), run()
    assert s1 == s2 and s1
    for s in s1:
        ids = {"a": ["e1", "e2"], "b": ["f1"]}[s["base"]]
        assert all(c["arguments"]["event_id"] in ids for c in s["label"]["tool_calls"] if c["name"] == "cal.cancel")
        assert "cal.list" in cg.describe(s, [("hủy lịch", s["label"]["tool_calls"])])


def test_foreign_script_and_missing_diacritics_are_rejected():
    s = item("a", ["e1"], "2026-11-01", None)
    assert cg.text_ok("hủy giúp tôi lịch đá bóng chiều nay", s, CAT)[0]
    assert cg.text_ok("hủy giúp tôi lịch 取消 chiều nay nhé", s, CAT)[1] == "foreign_script"
    assert cg.text_ok("huy giup toi lich da bong chieu nay", s, CAT)[1] == "not_vietnamese"
    assert cg.text_ok("cancel my football meeting this afternoon", s, CAT)[1] == "not_vietnamese"


def test_generate_honours_segment_count_weights():
    items = [item("a", ["e1", "e2"], "2026-11-01", None), item("b", ["f1"], "2026-11-09", None)]
    env = lambda i: FakeEnv(["e1", "e2"] if i == "a" else ["f1"])  # noqa: E731
    only_one, _ = cg.generate(items, CAT, env, random.Random(3), 6, weights={1: 1.0})
    assert only_one and all(len({c["name"].split(".")[0] for c in s["label"]["tool_calls"]}) == 1 for s in only_one)


def test_ids_keep_their_kind_and_stay_consistent_inside_a_sample():
    class Env:
        calls = 0

        def validate(self, name, args):
            return {"ok": True}

        def get_state_hash(self):
            return str(self.calls)

        def call(self, name, args):
            self.calls += 1
            if name == "dev.list":
                return {"ok": True, "devices": [{"id": "light_a", "type": "light"}, {"id": "ac_1", "type": "climate"},
                                                {"id": "light_b", "type": "light"}]}
            return {"ok": True}

    cat = {"dev.list": {"name": "dev.list", "kind": "read", "parameters": {"properties": {}}},
           "dev.light": {"name": "dev.light", "kind": "write", "parameters": {"properties": {"device_id": {"x-ref": "dev.devices"}}}},
           "dev.note": {"name": "dev.note", "kind": "write", "parameters": {"properties": {"device_id": {"x-ref": "dev.devices"}}}}}
    donor = {"id": "d", "context": {"current_time": "2026-11-01T08:00:00+07:00", "installed_apps": {"dev": {"tools": list(cat)}}},
             "history": [], "label": {
                 "tool_calls": [{"name": "dev.list", "arguments": {}}, {"name": "dev.light", "arguments": {"device_id": "light_a"}},
                                {"name": "dev.note", "arguments": {"device_id": "light_a"}}],
                 "tool_results": [{"ok": True, "devices": [{"id": "light_a", "type": "light"}, {"id": "ac_9", "type": "climate"}]},
                                  {"ok": True}, {"ok": True}]}}
    for seed in range(20):
        out = cg.transplant(cg.segments(donor, cat)[0], donor, donor, Env(), [], cat, random.Random(seed))
        assert out is not None
        picked = [c["arguments"]["device_id"] for c in out[0] if "device_id" in c["arguments"]]
        assert picked[0] == picked[1] and picked[0] in ("light_a", "light_b")       # same kind, and the same device twice
    refs = cg.resolve_refs(out[0], out[1], cat)
    assert refs[picked[0]]["type"] == "light"
    sample = {"label": {"tool_calls": out[0], "tool_results": out[1]}, "refs": refs, "context": {"current_time": "2026-11-01T08:00:00+07:00"}}
    assert picked[0] in cg.verify_prompt(sample, "bật đèn lên") and "tương ứng với" in cg.describe(sample, [])


def test_time_stated_in_the_text_must_match_the_label():
    cat = {**CAT, "alarm.create": {"name": "alarm.create", "kind": "write", "parameters": {"properties": {"time": {"type": "string"}}}}}
    s = item("a", ["e1"], "2026-11-01", None)
    s["label"]["tool_calls"] = [{"name": "alarm.create", "arguments": {"time": "2026-11-02T05:00:00+07:00"}}]
    s["label"]["tool_results"] = [{"ok": True}]
    assert cg.text_ok("hẹn chuông năm giờ sáng mai cho cậu bạn của tôi nhé", s, cat) == (True, "ok")
    assert cg.text_ok("hẹn chuông sáu giờ sáng mai cho cậu bạn của tôi nhé", s, cat)[1] == "time_conflict"
    assert cg.text_ok("hẹn chuông năm giờ sáng ngày kia cho cậu bạn của tôi nhé", s, cat)[1] == "time_conflict"


def test_ids_resolve_to_the_listing_not_to_a_later_receipt():
    cat = {"bank.list": {"name": "bank.list", "kind": "read", "parameters": {"properties": {}}},
           "bank.pay": {"name": "bank.pay", "kind": "write", "parameters": {"properties": {"account_id": {"x-ref": "bank.accounts"}}}}}
    calls = [{"name": "bank.list", "arguments": {}}, {"name": "bank.pay", "arguments": {"account_id": "acc_02"}}]
    results = [{"ok": True, "accounts": [{"id": "acc_01", "name": "thanh toán"}, {"id": "acc_02", "name": "tiết kiệm"}]},
               {"ok": True, "item": {"id": "tx_01", "account_id": "acc_02", "amount": -5}}]
    assert cg.find_path("acc_02", results) == ("accounts", "id")
    assert cg.resolve_refs(calls, results, cat)["acc_02"]["name"] == "tiết kiệm"
    # a later update of the same record must not hide what the id referred to before the write
    results2 = [{"ok": True, "events": [{"id": "e1", "start": "old"}]}, {"ok": True, "item": {"id": "e1", "start": "new"}}]
    assert cg.record_of("e1", ("events", "id"), results2)["start"] == "old"


def test_calls_the_environment_silently_refused_are_dropped():
    assert cg.soft_failure({"ok": True, "done": False, "message": "assignee phải là một trong ['Lan']"})
    assert cg.soft_failure({"ok": True, "message": "món 'x' không có trong thực đơn"})
    assert not cg.soft_failure({"ok": True, "message": {"id": "msg_01", "content": "hi"}})     # a sent message is an object
    assert not cg.soft_failure({"ok": True, "item": {"id": "e1"}})

    class Env:
        def validate(self, name, args):
            return {"ok": True}

        def get_state_hash(self):
            return "h"

        def call(self, name, args):
            return {"ok": True, "done": False, "message": "không tìm thấy"}

    cat = {"x.do": {"name": "x.do", "kind": "write", "parameters": {"properties": {}}}}
    donor = {"id": "d", "context": {"current_time": "2026-11-01T08:00:00+07:00", "installed_apps": {"x": {"tools": ["x.do"]}}},
             "history": [], "label": {"tool_calls": [{"name": "x.do", "arguments": {}}], "tool_results": [{"ok": True}]}}
    assert cg.transplant(cg.segments(donor, cat)[0], donor, donor, Env(), [], cat, random.Random(0)) is None


def test_a_sample_never_has_fewer_segments_than_requested():
    """Asking for 3 segments when only 1 app is available must produce nothing, not a one-segment sample (the old loop
    skipped the missing segments, so --weights was almost ignored: 86% of the A5 retry samples had a single segment)."""
    items = [item("a", ["e1", "e2"], "2026-11-01", None), item("b", ["f1"], "2026-11-09", None)]
    env = lambda i: FakeEnv(["e1", "e2"] if i == "a" else ["f1"])  # noqa: E731
    out, stats = cg.generate(items, CAT, env, random.Random(4), 5, weights={3: 1.0}, max_tries_factor=5)
    assert out == [] and stats["no_segment"] > 0


def test_times_that_end_up_in_the_past_and_writes_that_change_nothing_are_dropped():
    from datetime import datetime
    now_d, now_b = datetime(2026, 11, 1, 8, 0), datetime(2026, 11, 1, 18, 0)
    assert cg.moved_into_the_past("2026-11-01T15:00:00+07:00", "2026-11-01T15:00:00+07:00", now_d, now_b)
    assert not cg.moved_into_the_past("2026-11-01T07:00:00+07:00", "2026-11-01T07:00:00+07:00", now_d, now_b)   # was past already
    assert not cg.moved_into_the_past("2026-11-02T15:00:00+07:00", "2026-11-02T15:00:00+07:00", now_d, now_b)   # tomorrow
    assert not cg.moved_into_the_past("evt_1", "evt_2", now_d, now_b)

    class NoopEnv:
        def validate(self, name, args):
            return {"ok": True}

        def get_state_hash(self):
            return "same"

        def call(self, name, args):
            return {"ok": True}

    cat = {"x.set": {"name": "x.set", "kind": "write", "parameters": {"properties": {}}}}
    donor = {"id": "d", "context": {"current_time": "2026-11-01T08:00:00+07:00", "installed_apps": {"x": {"tools": ["x.set"]}}},
             "history": [], "label": {"tool_calls": [{"name": "x.set", "arguments": {}}], "tool_results": [{"ok": True}]}}
    assert cg.transplant(cg.segments(donor, cat)[0], donor, donor, NoopEnv(), [], cat, random.Random(0)) is None
