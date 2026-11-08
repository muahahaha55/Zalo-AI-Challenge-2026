import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lifemate.normalize import words_to_digits as w

CASES = {
    "một trăm hai mươi ba triệu": "123000000",
    "hai triệu tám": "2800000",
    "hai triệu rưỡi": "2500000",
    "năm trăm nghìn": "500000",
    "mười lăm": "15",
    "hai mươi mốt": "21",
    "một trăm linh năm": "105",
    "chuyển hai trăm nghìn cho mẹ": "chuyển 200000 cho mẹ",
    "nhắc tôi lúc tám giờ": "nhắc tôi lúc 8 giờ",
    "không ba bảy hai bốn năm sáu tám chín": "037245689",
    "bật đèn phòng khách": "bật đèn phòng khách",
    "ba chục nghìn": "30000",
}


def test_cases():
    for src, exp in CASES.items():
        assert w(src) == exp, (src, w(src), exp)
