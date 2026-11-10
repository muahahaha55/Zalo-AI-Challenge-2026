"""Write submission/predict_notebook.ipynb (organisers' timing notebook: seed, load models, read test cases, predict)."""
import json
import sys
from pathlib import Path

cells = [
    ("markdown", """# LifeMate: notebook đo thời gian (theo hướng dẫn Docker của BTC)
Chạy **Run All**. Kết quả ghi vào `/result/jupyter_submission.jsonl`, cùng định dạng `submission.jsonl` (có trường `time`, mili giây mỗi mẫu).
Notebook dùng **đúng cùng đường mã và cấu hình** với `predict.py` (`lifemate.container.build_runner`), nên `tool_calls` và `state_hash` giống hệt; chỉ `time` khác.
Không cần mạng; mọi model nằm sẵn trong `/code/saved_models`."""),
    ("code", """# Bước 1: cố định seed (cùng seed với predict.py; giải mã greedy nên kết quả không phụ thuộc lấy mẫu ngẫu nhiên)
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/code/src")
from lifemate.container import SEED, build_runner, parse_args  # noqa: E402
from lifemate.determinism import seed_everything  # noqa: E402

seed_everything(SEED)
print("seed", SEED)"""),
    ("code", """# Bước 2: nạp mô hình và tài nguyên (LLM Qwen3-4B đã fine-tune, ASR PhoWhisper-large đã fine-tune), không dùng API ngoài
args = parse_args([])                      # mặc định giống hệt predict.py: /data, /code/saved_models/*, ngân sách 25 s/mẫu, luật bật
t0 = time.perf_counter()
runner, layout, adapter = build_runner(args)
print(f"load model: {int((time.perf_counter() - t0) * 1000)} ms; split {layout.split}; adapter {adapter or 'none (merged)'}")"""),
    ("code", """# Bước 3: đọc các test case từ /data/<split>/<split>.jsonl
test_cases = runner.items(limit=args.limit)
print(len(test_cases), "test cases")"""),
    ("code", """# Bước 4: dự đoán từng mẫu, đo thời gian từ lúc bắt đầu mẫu (trước khi đọc audio) đến khi có kết quả
all_result = []
for item in test_cases:
    t1 = time.perf_counter()
    outcome = runner.run_item(item)            # ASR -> LLM -> env.call(...) -> ... trên đúng một ToolEnv của mẫu
    ms = int((time.perf_counter() - t1) * 1000)
    all_result.append({"id": item["id"], "tool_calls": outcome.tool_calls, "state_hash": outcome.state_hash, "time": ms})

Path("/result").mkdir(parents=True, exist_ok=True)
with open("/result/jupyter_submission.jsonl", "w", encoding="utf-8", newline="\\n") as f:
    for r in all_result:
        f.write(json.dumps(r, ensure_ascii=False) + "\\n")
times = sorted(r["time"] for r in all_result)
print(f"{len(all_result)} mẫu, trung bình {sum(times) / max(1, len(times)):.0f} ms, lớn nhất {times[-1] if times else 0} ms")
print("output saved in /result/jupyter_submission.jsonl")"""),
]


def cell(kind, text):
    lines = text.split("\n")
    src = [line + "\n" for line in lines[:-1]] + [lines[-1]]
    c = {"cell_type": kind, "metadata": {}, "source": src}
    if kind == "code":
        c.update({"execution_count": None, "outputs": []})
    return c


nb = {"cells": [cell(k, t) for k, t in cells], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
Path(sys.argv[1]).write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
print("written", sys.argv[1])
