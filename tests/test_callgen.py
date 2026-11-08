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
