# Bàn giao bước 0: gộp LoRA vào Qwen3-4B và đo lại dev

Ngày 2026-11-11. Người chạy: thuan (Kaggle, 1 GPU RTX PRO 6000 Blackwell), làm đúng "Bước 0" của `docs/HUONG_DAN_GPU_08-11.md`. Không sửa mã của Phương (`scripts/merge_lora.py` chạy nguyên bản). Mã là commit `b2853d2`. Số liệu tổng hợp nằm ở `docs/results/merge/`. Trọng số `q4b_merged` và `phowl_ft/merged` **không** vào git; chuyển cho chủ dự án qua Google Drive riêng (mục 5).

## 1. Tóm tắt
1. **Điểm dev của model đã gộp (không adapter) trùng hoàn toàn với `q4b_lora`**: chữ sạch **66,82** và ASR `phowl_ft` **56,98**, đúng hai mốc của Phương. Chênh từng cặp **+0,00** (khoảng tin cậy [+0,00; +0,00]), 155/155 mẫu bằng điểm.
2. Có **4 mẫu (chữ sạch) và 3 mẫu (ASR) khác nhau về `tool_calls`/`state_hash`** giữa hai model (151/155 và 152/155 giống hệt), nhưng điểm của chúng không đổi. Đây là sai khác làm tròn bf16 mà `merge_lora.py` đã nêu trước (gộp float32, lưu bfloat16); không ảnh hưởng điểm dev.
3. Gộp mất **9 giây**; model đã gộp **4,0225B tham số**, lưu bfloat16 trong một file `model.safetensors` (khoảng 8 GB). Cộng với ASR thì khớp mức 5,566B mà `docs/TRANG_THAI.md` dự kiến (chưa đếm lại bằng `count_params.py`).
4. sha256 của `phowl_ft/merged/model.safetensors` là `cffba381…`, **trùng giá trị ASR trong `models.lock.json`**.
5. Theo hướng dẫn: dev không tụt, nên đủ điều kiện chuyển hai thư mục cho chủ dự án để dựng image.

## 2. Cấu hình
`merge_lora.py --base <Qwen3-4B> --adapter checkpoints/q4b_lora --out checkpoints/q4b_merged` (gộp float32, lưu bfloat16, tokenizer và chat template lấy từ thư mục adapter). Đo dev bằng `infer_multi_gpu.sh train ... --model checkpoints/q4b_merged --text clean` và `--text asr:phowl_ft` (không `--adapter`), luật bật, chấm bằng `eval.py`. Mốc `q4b_lora` **được chạy lại trong cùng phiên** (Qwen3-4B + adapter) để so cùng môi trường: ra đúng 66,82 và 56,98. Môi trường: `torch 2.11.0+cu130`, CUDA 13.0, Python 3.13.16, `peft 0.21.2`, `transformers 5.20.2`, `accelerate 1.15.0`, VRAM 102 GB.

## 3. Điểm dev (155 mẫu)
| | `q4b_lora` | `q4b_merged` | Chênh từng cặp |
|---|---|---|---|
| Chữ sạch | 66,82 | 66,82 | +0,00 [+0,00; +0,00] (0 tốt lên, 0 tệ đi, 155 bằng) |
| ASR `phowl_ft` | 56,98 | 56,98 | +0,00 [+0,00; +0,00] (0, 0, 155) |

Theo loại (giống hệt nhau ở hai model): chữ sạch exec 65,73, ask 75,64, oos 58,33; ASR exec 57,99, ask 63,33, oos 33,33. Số mẫu giống hệt về `tool_calls` và `state_hash` giữa hai model: chữ sạch 151/155, ASR 152/155.

## 4. Điều chưa kết luận được
1. Chỉ đo dev 155 mẫu. `HUONG_DAN_GPU` và `TRANG_THAI` còn yêu cầu đo lại trên `public_test` với model đã gộp (A7a, notebook `a7a-verify-real` dùng đúng `q4b_merged`); chưa làm ở bước này.
2. 4 và 3 mẫu khác về tool_calls nhưng bằng điểm: tôi chưa xem chúng là mẫu nào, và chưa phân tích xem đó là cả hai cùng sai hay khác nhau nhưng cùng được chấm đúng.
3. Kết quả cell kiểm tra sha256 adapter (cell M-1) không nằm trong zip. Việc dev trùng đúng 66,82 và 56,98 phù hợp với adapter đúng, nhưng sha256 adapter chưa được đối chiếu trong bản bàn giao này.
4. `models.lock.json` hiện còn ghi shard Qwen3-4B gốc cho LLM; chủ dự án cập nhật bằng `check_models.py --update-lock` sau khi nhận `q4b_merged`.

## 5. Chuyển trọng số cho chủ dự án (việc của người chạy GPU)
Tải từ tab Output của notebook `merge-lora` (từng file): `q4b_merged/*` (6 file, trong đó `model.safetensors` khoảng 8 GB), `phowl_ft/merged/*` (khoảng 3 GB), và `SHA256SUMS_merged.txt`. Đưa lên Google Drive riêng (không công khai). Sau khi tải, đặt `SHA256SUMS_merged.txt` trong thư mục chứa `q4b_merged/` và `phowl_ft/merged/` rồi chạy `sha256sum -c SHA256SUMS_merged.txt`. sha256 của hai file trọng số chính:
- `q4b_merged/model.safetensors`: `652e96ae…` (đầy đủ trong `docs/results/merge/SHA256SUMS_merged.txt`)
- `phowl_ft/merged/model.safetensors`: `cffba381…`

## 6. File
| File | Nội dung | Trong git? |
|---|---|---|
| `docs/results/merge/diem_*.txt` | Điểm dev của `q4b_lora` và `q4b_merged` (sạch và ASR) | có |
| `docs/results/merge/so_sanh_lora_merged.txt` | So từng cặp và số mẫu khác nhau | có |
| `docs/results/merge/log_gop.txt`, `torch_cuda.txt`, `SHA256SUMS_merged.txt`, `*.zip.sha256` | Log gộp, môi trường, sha256 | có |
| `merge_results.zip` (55 KB) | Gồm cả dự đoán dev (`dev_*.jsonl`) | **không**, gửi qua Drive riêng |
| `q4b_merged/`, `phowl_ft/merged/` | Trọng số | **không**, qua Drive riêng (mục 5) |

sha256 zip: `55aa5948…`.

## 7. Dòng nhật ký đề xuất cho `docs/EXPERIMENTS.md`
| Ngày | Commit | Việc | Điểm | Ghi chú |
|---|---|---|---|---|
| 2026-11-11 | `b2853d2` | **Bước 0**: gộp LoRA `q4b_lora` vào Qwen3-4B (`merge_lora.py`), đo dev với model đã gộp, không adapter | 66,82 (sạch) và 56,98 (ASR `phowl_ft`), bằng `q4b_lora` | chênh từng cặp +0,00 ở cả hai (155/155 bằng điểm); 4 và 3 mẫu khác về tool_calls nhưng bằng điểm (làm tròn bf16); gộp 9 s; 4,0225B tham số; sha256 ASR trùng `models.lock.json`. Chi tiết: `docs/BAN_GIAO_MERGE.md` |
