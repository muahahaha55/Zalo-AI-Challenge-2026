from datetime import datetime

from lifemate.rules import extract_when, fix_args

NOW = "2026-11-24T10:15:00+07:00"       # a Tuesday


def fix(args, text, name="reminders.create"):
    return fix_args(name, args, text, NOW)[0]


def test_kem_15_is_quarter_to():
    out = fix({"content": "x", "time": "2026-11-25T20:45:00+07:00"}, "Nhắc tôi uống thuốc lúc 8 giờ kém 15 tối mai với.")
    assert out["time"] == "2026-11-25T19:45:00+07:00"


def test_spoken_hour_with_period():
    out = fix({"time": "2026-11-25T18:30:00+07:00"}, "Đặt báo thức sáu rưỡi sáng mai", "alarm.create")
    assert out["time"] == "2026-11-25T06:30:00+07:00"


def test_next_week_weekday():
    w = extract_when("ba giờ chiều thứ ba tuần sau", datetime.fromisoformat("2026-10-01T09:00:00+07:00"))
    assert str(w.day) == "2026-10-06" and w.clock[:2] == (3, 0)


def test_start_end_only_date_is_fixed():
    args = {"title": "t", "start": "2026-10-07T15:00:00+07:00", "end": "2026-10-07T16:00:00+07:00"}
    out = fix_args("calendar.create_event", args, "ba giờ chiều thứ ba tuần sau", "2026-10-01T09:00:00+07:00")[0]
    assert out["start"].startswith("2026-10-06T15:00") and out["end"].startswith("2026-10-06T16:00")


def test_spoken_phone_overrides_garbled_digits():
    out = fix_args("contacts.update", {"contact_id": "c_01", "phone": "09123456850"},
                   "số mới là không chín một hai, ba bốn năm, sáu bảy tám năm", NOW)[0]
    assert out["phone"] == "09123456785"


def test_untouched_when_ambiguous_or_relative():
    base = {"time": "2026-11-25T09:00:00+07:00"}
    for text in ["Nhắc tôi dời lịch muộn thêm một tiếng", "đặt báo thức năm giờ, à không, năm rưỡi",
                 "nhắc lúc bảy giờ sáng và mười giờ tối", "năm giờ bốn lăm em nhé", "Mai Anh nhờ gọi lại"]:
        assert fix(base, text) == base, text


def test_big_gap_is_not_overridden():
    base = {"time": "2026-12-20T20:00:00+07:00"}          # stated time belongs to something else (far from the stated one)
    assert fix(base, "nhắc tôi lúc 8 giờ sáng mai") == base


def test_errors_never_raise():
    assert fix_args("x", {"time": "not-a-date"}, "", "garbage") == ({"time": "not-a-date"}, [])
