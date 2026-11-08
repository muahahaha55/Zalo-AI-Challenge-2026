import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lifemate import leakcheck  # noqa: E402

DATA = ["Nhắc em đặt lịch cắt tóc lúc 9 giờ sáng thứ bảy tuần này nhé.", "Bật đèn."]    # invented sentences, never real data


def index():
    return leakcheck.build_index(DATA)


def test_a_whole_long_transcript_is_found_whatever_the_case_and_punctuation():
    windows, wholes = index()
    assert leakcheck.find_leaks("# comment: NHẮC EM đặt lịch cắt tóc, lúc 9 giờ sáng thứ bảy tuần này nhé!!", windows, wholes)


def test_a_long_enough_fragment_is_found():
    windows, wholes = index()
    hit = leakcheck.find_leaks("example: đặt lịch cắt tóc lúc 9 giờ sáng thứ bảy", windows, wholes)
    assert hit[0] == "đặt lịch cắt tóc lúc 9 giờ sáng"          # the first 8-word window that matches


def test_short_overlap_and_unrelated_text_pass():
    windows, wholes = index()
    assert leakcheck.find_leaks("Hẹn giờ cắt tóc lúc 9 giờ sáng nay.", windows, wholes) == []   # shares only a short phrase
    assert leakcheck.find_leaks("Bật đèn.", windows, wholes) == []                                            # under 20 characters
    assert leakcheck.find_leaks("def f(): return 1", windows, wholes) == []


def test_normalize_is_nfc_and_ignores_punctuation():
    import unicodedata
    decomposed = unicodedata.normalize("NFD", "Tắt, đèn!")
    assert leakcheck.normalize(decomposed) == leakcheck.normalize("tắt đèn") == "tắt đèn"
