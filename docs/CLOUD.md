# Chạy trên GPU thuê (vast.ai / Kaggle)

Làm theo `docs/RULES.md` mục 3: khóa SSH riêng cho dự án, đặt **trần chi phí trước khi chạy**, hủy máy và xóa dữ liệu khi xong. Dữ liệu cuộc thi chỉ nằm trên máy thuê của đội, không đưa vào image hay dataset công khai.

## 1. Chọn máy
- GPU 24 GB (A30, RTX 4090, A5000, A100), đĩa ≥ 100 GB, Ubuntu có CUDA 12.x. Để đo thời gian gần với chấm bài, ưu tiên **A30**.
- Ước lượng sơ bộ (chưa đo, cần xác nhận ở lần chạy đầu): SFT Qwen3-4B LoRA khoảng 3–5 giờ cho ~19.000 mẫu bước × ~1.9K token; fine-tune PhoWhisper-large LoRA khoảng 1–2 giờ.
- Trần chi phí đề xuất cho mỗi lượt: số giờ dự kiến × giá/giờ × 1,5. Dừng và báo nếu vượt.

## 2. Đưa mã và dữ liệu lên máy
```bash
# Trên máy local (Git Bash/WSL). <PORT> <HOST> lấy từ trang vast.ai.
ssh -p <PORT> root@<HOST> "mkdir -p /workspace/lifemate"
git ls-files -z | rsync -av --from0 --files-from=- -e "ssh -p <PORT>" ./ root@<HOST>:/workspace/lifemate/
rsync -av --progress -e "ssh -p <PORT>" data/raw data/processed root@<HOST>:/workspace/lifemate/data/
```
(Hoặc `git clone` repo rồi chỉ rsync `data/`.) Sau đó trên máy thuê: `cd /workspace/lifemate && bash scripts/setup_cloud.sh`. Dùng `tmux` cho job dài.

## 3. Các lượt chạy theo thứ tự
| # | Lệnh (trên máy thuê, `source .venv/bin/activate`) | Mục đích |
|---|---|---|
| 1 | `python scripts/run_infer.py --split train --ids data/processed/dev_ids.txt --backend hf --model Qwen/Qwen3-4B --text clean --out outputs/dev_q4b_clean.jsonl` | baseline zero-shot, transcript sạch (trần LLM) |
| 2 | như trên với `--text asr:phowl` | baseline với ASR thật |
| 3 | `python scripts/eval.py --sub outputs/dev_q4b_clean.jsonl --ids data/processed/dev_ids.txt` | điểm dev theo loại mẫu |
| 4 | `python scripts/train_asr.py --model vinai/PhoWhisper-large --out checkpoints/phowl_ft --epochs 3` | fine-tune ASR (song song nếu có 2 máy) |
| 5 | `python scripts/train_sft.py --model Qwen/Qwen3-4B --data data/processed/sft_train.jsonl data/processed/sft_train_asr.jsonl --out checkpoints/q4b_lora --epochs 3` | SFT LLM |
| 6 | `python scripts/run_infer.py ... --adapter checkpoints/q4b_lora` rồi `eval.py` | đo dev sau SFT |
| 7 | `python scripts/run_infer.py --split public_test --text asr:<tag> ... --out outputs/pub.jsonl` | bài nộp public (chỉ nộp sau khi được đồng ý) |

Sau mỗi lượt: ghi cấu hình, seed, commit và điểm vào `docs/EXPERIMENTS.md`.
Sau khi ASR đã fine-tune: chạy lại `run_asr.py --model checkpoints/phowl_ft/merged` cho train và public_test, rồi `build_sft.py --asr-tag <tag>` để làm lại biến thể ASR của dữ liệu SFT.

## 4. Lấy kết quả về và dọn dẹp
```bash
rsync -av -e "ssh -p <PORT>" root@<HOST>:/workspace/lifemate/{outputs,checkpoints} ./   # adapter + submission, không kéo dữ liệu
```
Cuối cùng: **hủy máy trên vast.ai** (xóa instance, không chỉ dừng) để ngừng tính phí và xóa dữ liệu trên đó.
