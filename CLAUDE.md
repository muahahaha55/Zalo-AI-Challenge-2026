# LifeMate — Zalo AI Challenge 2026

Agent biến yêu cầu giọng nói tiếng Việt thành chuỗi tool call. Kế hoạch: `docs/PLAN.md`. **Quy tắc bắt buộc: `docs/RULES.md`** (đọc trước khi làm việc liên quan đến dữ liệu, inference, bí mật, git hoặc nộp bài).

## Quy tắc cứng (tóm tắt)
- Không bao giờ đọc `*_environment/` (kể cả `state.json`). Dữ liệu app chỉ qua `zaic_tools`; lúc inference chỉ `validate`, `call`, `get_tool_calls`, `get_state_hash`; một `ToolEnv` mỗi mẫu.
- Không gán nhãn public/private test; không gửi dữ liệu cuộc thi cho dịch vụ bên thứ ba; `data/` không vào git.
- Inference offline, tất định (greedy, seed cố định), < 6B tham số (LLM < 5B, ngữ cảnh ≤ 16K), ≤ 30 s/mẫu.
- Không đưa bí mật vào git/chat; không `trust_remote_code`; không force-push; không `--no-verify`.
- **Hỏi trước** khi: push, nộp leaderboard/Docker, chi tiền (GPU thuê), gửi dữ liệu ra ngoài, xóa dữ liệu/model.

## Lệnh thường dùng (Windows, Python 3.12, venv `.venv`)
- Chạy Python: `.venv/Scripts/python -I -X utf8 <script>` (từ thư mục gốc repo).
- Kiểm tra tuân thủ: `python scripts/check_compliance.py`; test: `pytest tests` (94 test); lint: `ruff check .` (cài bằng `pip install -r requirements-dev.txt`). Hook pre-commit chạy cả ba.
- Chấm nội bộ: `scripts/eval.py --sub <file> [--ids data/processed/dev_ids.txt]`; kiểm bộ chấm: `scripts/verify_metric.py`.
- Hook pre-commit: `git config core.hooksPath .githooks` (đã bật ở máy này).

## Cấu trúc
- `src/lifemate/` (thư viện, có test, lint đầy đủ): `agent.py` (lớp `Agent`: vòng lặp từng bước, giới hạn thời gian, bắt lỗi backend, lý do dừng), `pipeline.py` (`SplitRunner`, `Transcripts`, `HFBackend`: dùng chung cho `run_infer.py` và container), `prompt.py` (dựng prompt, dùng chung SFT và suy luận), `metric.py` (chấm điểm), `augment.py` (mẫu lịch sử, biến thể, giấu app), `callgen.py` (ghép đoạn call thật sang môi trường khác, lọc câu; `scripts/gen_calls.py`, `gen_text.py`, `verify_callgen.py`), `grouping.py` (nhóm câu gần giống, prefix-filter), `rules.py`, `normalize.py`, `paths.py` (đường dẫn, ghi đè bằng `LIFEMATE_DATA_ROOT`), `jsonl.py` (có `ResumableJsonlWriter`: ghi nguyên tử, chạy tiếp), `asr.py` (`WhisperTranscriber`), `container.py` (chạy trong container: tự phát hiện bố cục `/data`, offline, ghi `result/submission.jsonl`), `datacheck.py` + `scripts/preflight_train.py` (kiểm tra dữ liệu trước khi train), `manifest.py` (nguồn gốc dữ liệu), `params.py` + `scripts/count_params.py` (đếm tham số từ file), `determinism.py`. Docker: `docker/` và `docs/DOCKER.md` (bản nháp, chưa dựng được vì Docker không chạy).
- `scripts/`: `run_asr.py`, `asr_wer.py`, `build_sft.py`, `run_infer.py`, `eval.py`, ...
- `data/raw/train+public_test/` (dữ liệu gốc, ngoài git), `data/processed/` (đầu ra trung gian, ngoài git).
- Máy local chỉ có RTX 4050 6 GB: train và đo thời gian chạy trên vast.ai/Kaggle.
