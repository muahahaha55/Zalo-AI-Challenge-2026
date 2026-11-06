# LifeMate — Zalo AI Challenge 2026

Agent biến yêu cầu giọng nói tiếng Việt thành chuỗi tool call.
Pipeline: **ASR (PhoWhisper-large + LoRA) → Qwen3-4B (SFT LoRA) → vòng lặp từng bước: sinh 1 call → `validate` → `call` → đưa kết quả lại → … → `END`**.
Ràng buộc: offline, tất định (greedy, seed cố định), tổng < 6B tham số (LLM < 5B), ≤ 30 s/mẫu.

> **Đọc trước khi làm bất cứ việc gì: [docs/RULES.md](docs/RULES.md)** (luật dữ liệu, bí mật, git, nộp bài). Vi phạm có thể bị loại.
> Dữ liệu cuộc thi **không** có trong repo và không được chia sẻ: mỗi người tự tải từ cổng cuộc thi bằng tài khoản của mình.

## Mốc thời gian
| Hạn | Việc |
|---|---|
| 16/11 23:59 | Nộp Docker (tối đa 2 lần, chỉ lần cuối được chấm) |
| 20/11 09:00 | Phát private_test |
| 23/11 23:59 | Nộp kết quả private |

## Tiến độ (cập nhật 2026-11-06)
Điểm **dev nội bộ** (155 mẫu giữ riêng từ train; chưa có điểm leaderboard):

| Kịch bản | Điểm dev |
|---|---|
| Qwen3-4B zero-shot | 7,35 (nộp rỗng: 7,18) |
| SFT + ASR gốc | 47,75 |
| SFT + ASR fine-tune (`phowl_ft`) — **cấu hình nộp hiện tại** | **56,33** |
| SFT + transcript sạch (trần của LLM hiện tại) | 65,69 |

Chi tiết, nguyên nhân mất điểm và nhật ký: [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md). Kết luận chính: **điểm nghẽn là SFT của LLM** (quá khớp, loại ask/oos, app lạ), không phải ASR hay luật giờ.

## Việc còn lại (nhận việc bằng cách ghi tên vào cột "Ai")
Số Step theo [docs/PLAN.md](docs/PLAN.md). Không dùng audio hay nhãn public/private test để huấn luyện (RULES mục 1).

| # | Việc | Step | Ghi chú | Ai |
|---|---|---|---|---|
| 1 | Nộp public lần đầu, ghi điểm vào `EXPERIMENTS.md` | 4 | Cần người phụ trách đồng ý (RULES 5.1). Xem mục "Nộp public" bên dưới | |
| 2 | Huấn luyện lại SFT với augmentation (oos tổng hợp + xáo thứ tự app), lưu checkpoint theo epoch, so với `q4b_lora` | 5 | `docs/KAGGLE.md` mục 4b. Đã có code, chưa chạy | |
| 3 | Dữ liệu tổng hợp cho 11 app mới của public (chỉ dùng **schema** catalog) | 6 | 65% mẫu public có app lạ. Cần LLM lớn ở giai đoạn phát triển (không dùng lúc inference) | |
| 4 | Dev "app lạ": giấu vài app khỏi SFT để đo khái quát hóa thật | 5/6 | Hiện dev chỉ có app đã thấy nên điểm dev lạc quan | |
| 5 | Cổng ask/oos bảo thủ, constrained decoding | 8 | Gọi thừa 1 tool khi oos là 0 điểm | |
| 6 | Đo thời gian p95 trên A30 và đếm tham số cuối | 11 | Kaggle chỉ để tham khảo | |
| 7 | Docker (nhúng trọng số, `--network none`, chạy 3–5 lần so từng byte) | 12 | Hạn 16/11. Chờ hướng dẫn đóng gói của BTC | |
| 8 | Dọn Kaggle khi xong: xóa dataset raw/processed và output notebook chứa transcript | — | `docs/KAGGLE.md` mục 0 | |

## Cấu trúc repo
```
src/lifemate/    metric.py (bộ chấm), prompt.py (dựng prompt, dùng chung SFT và inference),
                 agent.py (vòng lặp từng bước), normalize.py (số đọc → chữ số), rules.py (hậu xử lý giờ/số điện thoại)
scripts/         run_asr.py, asr_wer.py, build_sft.py, train_asr.py, train_sft.py, run_infer.py, eval.py, verify_metric.py,
                 analyze_dev.py, replay_rules.py, validate_rules.py, check_compliance.py, check_env.py
                 kaggle_*.sh|py, infer_multi_gpu.sh, merge_shards.py, prepare_models.py   (chạy trên Kaggle)
                 setup_cloud.sh                                                            (chạy trên vast.ai)
tests/           test_metric.py, test_normalize.py, test_rules.py
notebooks/       submit-public.ipynb (chạy public_test → submission.jsonl), test-dev-set.ipynb (đo dev vòng 2),
                 testmodel-zaloai.ipynb (bản cũ, kiểm tra dataset Kaggle). Đã xóa output; sửa đường dẫn input theo tài khoản Kaggle của bạn
docs/            xem bảng dưới
data/ outputs/ checkpoints/ kaggle_upload/    ngoài git (dữ liệu, kết quả, trọng số)
```

## Tài liệu
| File | Nội dung |
|---|---|
| [docs/RULES.md](docs/RULES.md) | **Luật bắt buộc** |
| [docs/PLAN.md](docs/PLAN.md) | Kế hoạch 13 step và rủi ro |
| [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) | Điểm, phân tích lỗi, nhật ký thí nghiệm (ghi mọi lần chạy mới vào đây) |
| [docs/KAGGLE.md](docs/KAGGLE.md) | Chạy trên Kaggle (Internet OFF, dataset riêng tư) |
| [docs/CLOUD.md](docs/CLOUD.md) | Chạy trên vast.ai |
| [docs/HANDOFF.md](docs/HANDOFF.md) | Yêu cầu bàn giao huấn luyện (bước 1–3) |
| [docs/MODELS.md](docs/MODELS.md) | Licence, phiên bản, số tham số, sha256 các model |

## Bắt đầu (máy local, Windows, Python 3.12)
```bash
python -m venv .venv && .venv/Scripts/python -m pip install -r requirements.txt   # torch cài riêng theo CUDA của máy
git config core.hooksPath .githooks                       # bật hook kiểm tra luật trước mỗi commit (bắt buộc)
# giải nén train+public_test.zip (lấy từ cổng cuộc thi) vào data/raw/ → data/raw/train+public_test/{train,public_test}
.venv/Scripts/python -I -X utf8 scripts/check_env.py       # mọi dòng phải "ok"
.venv/Scripts/python -I -X utf8 scripts/verify_metric.py   # bộ chấm: gold = 100, nộp rỗng ≈ 7
.venv/Scripts/python -m pytest tests -q                    # 17 test
.venv/Scripts/python scripts/check_compliance.py           # kiểm tra tuân thủ
```
Máy local (RTX 4050 6 GB) chỉ để viết code, chấm điểm, test. **Huấn luyện và đo thời gian chạy trên Kaggle hoặc vast.ai.**

## Tái lập kết quả hiện tại (Kaggle)
Làm theo `docs/KAGGLE.md` (dataset riêng tư: code, raw, models, và output notebook trước). Các stage: `asr_base → sft_asr_data → train → asr_ft → sft_ft_data → eval_dev` bằng `bash scripts/kaggle_pipeline.sh <stage>`.
Bản chạy hiện tại: seed 42, LoRA r=64 (LLM) / r=32 (ASR), 3 epoch. Checksum adapter và ASR đã gộp nằm trong `docs/EXPERIMENTS.md` mục 2. **Trọng số không nằm trong repo**: xin qua link Drive riêng của đội.

## Nộp public
1. Chạy `notebooks/submit-public.ipynb` trên Kaggle (private, Internet OFF) → `submission.jsonl` (634 dòng, một dòng/mẫu).
2. Kiểm tra: `cd data/raw/train+public_test/public_test && python -m zaic_tools --split public_test --root .. --validate <đường dẫn submission.jsonl>` → 0 lỗi schema.
3. **Hỏi người phụ trách**, rồi nộp ở mục Submission của cổng (tối đa 5 lần/ngày). Không dùng lượt nộp để dò đáp án.
4. Ghi cấu hình, sha256 và điểm vào `docs/EXPERIMENTS.md`.

## Quy trình làm việc nhóm
- Mỗi người làm trên **nhánh riêng** (`ten/viec`), mở PR vào `main`; không push thẳng `main`, không force-push, không `--no-verify`.
- Mỗi thí nghiệm mới: thêm một dòng vào bảng nhật ký của `docs/EXPERIMENTS.md` (cấu hình, seed, commit, điểm dev, thời gian). Chỉ kết luận khi chênh lệch lớn hơn nhiễu của dev (155 mẫu, mỗi mẫu 0,65 điểm).
- **Hỏi trước khi**: push lên remote, nộp leaderboard hoặc Docker, thuê GPU, gửi dữ liệu ra ngoài, xóa dữ liệu hay model.
- Không đưa vào git: dữ liệu cuộc thi, audio, transcript, dự đoán, trọng số, khóa/token (`.gitignore` và hook đã chặn phần lớn).
- Không đọc thư mục `*_environment/` (kể cả `state.json`); dữ liệu app chỉ qua `zaic_tools`.
