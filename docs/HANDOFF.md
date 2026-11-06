# Bàn giao: train trên máy có RTX 4090 (24 GB)

Dành cho thành viên đội sẽ chạy huấn luyện. Đọc `docs/RULES.md` trước (đặc biệt mục 1–3). Dữ liệu cuộc thi **không** có trong GitHub và chỉ dành cho đội đã đăng ký: lấy bộ dữ liệu từ trang cuộc thi bằng tài khoản của chính bạn (file `train+public_test.zip`), không nhận qua kênh công khai.

## 0. Chuẩn bị (một lần)
```bash
git clone https://github.com/muahahaha55/Zalo-AI-Challenge-2026.git && cd Zalo-AI-Challenge-2026
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu124   # Linux: pip install torch
pip install -r requirements.txt
git config core.hooksPath .githooks
mkdir -p data/raw && unzip train+public_test.zip -d data/raw      # kết quả: data/raw/train+public_test/{train,public_test,README.md}
python scripts/check_env.py                                       # mọi dòng phải "ok", ToolEnv ok cho train và public_test
```
Dùng `python -I -X utf8 <script>` trên Windows để tránh lỗi mã hóa. Chỉ `data/raw` và `data/processed` chứa dữ liệu, cả hai đã nằm ngoài git.

## 1. Dựng lại dữ liệu trung gian (không có trong git)
Thứ tự bắt buộc, vì các file sau phụ thuộc file trước:
```bash
python scripts/build_sft.py                       # tạo dev_ids.txt, sft_train.jsonl, sft_dev.jsonl (chia dev cố định bằng seed 42)
python scripts/run_asr.py --model vinai/PhoWhisper-large --tag phowl --split train
python scripts/run_asr.py --model vinai/PhoWhisper-large --tag phowl --split public_test
python scripts/build_sft.py --asr-tag phowl      # thêm sft_train_asr.jsonl, sft_dev_asr.jsonl
python scripts/asr_wer.py --tag phowl             # kỳ vọng: WER toàn bộ khoảng 26,2%
```
Kiểm tra hai điểm: `build_sft.py` in `train=1390 dev=155`; WER gần 26%. Nếu khác nhiều, dừng và báo.

## 2. Huấn luyện
1. **ASR (tùy chọn nhưng dự kiến)**: `python scripts/train_asr.py --model vinai/PhoWhisper-large --out checkpoints/phowl_ft --epochs 3`. Sau đó chạy lại `run_asr.py --model checkpoints/phowl_ft/merged --tag phowl_ft` cho cả hai split, rồi `build_sft.py --asr-tag phowl_ft`, và đo `asr_wer.py --tag phowl_ft --dev-only` để so với 25,4% ở dev. Chỉ dùng bản fine-tune nếu WER dev giảm rõ.
2. **LLM**: `python scripts/train_sft.py --model Qwen/Qwen3-4B --data data/processed/sft_train.jsonl data/processed/sft_train_asr.jsonl --out checkpoints/q4b_lora --epochs 3`. Nếu hết VRAM: giảm `--max-len` xuống 6144, hoặc `--rank 32`.
3. **Đánh giá trên dev** (không dùng public_test để tinh chỉnh):
```bash
python scripts/run_infer.py --split train --ids data/processed/dev_ids.txt --backend hf --model Qwen/Qwen3-4B \
    --adapter checkpoints/q4b_lora --text asr:phowl --out outputs/dev_sft.jsonl
python scripts/eval.py --sub outputs/dev_sft.jsonl --ids data/processed/dev_ids.txt
```
Ghi số liệu vào `docs/EXPERIMENTS.md` (cấu hình, seed, commit, điểm dev, thời gian mỗi mẫu).

## 3. Gửi lại những gì
- **Gửi**: thư mục adapter (`checkpoints/q4b_lora`, `checkpoints/phowl_ft/merged` nếu dùng), `docs/EXPERIMENTS.md` đã cập nhật, log huấn luyện, `outputs/dev_*.jsonl` điểm dev. Kèm checksum (`sha256sum`) và link Drive riêng của đội.
- **Không gửi / không commit**: dữ liệu, audio, `data/processed`, state, trọng số trong git. Không đưa token vào chat.
- Việc `git push`, nộp leaderboard và nộp Docker vẫn hỏi người phụ trách trước.

## 4. Lưu ý đã biết
- Thời gian huấn luyện chưa đo trên 4090. Trên RTX 4050, một mẫu SFT mất khoảng 13 giây (không dùng được); trên 4090 dự kiến nhanh hơn nhiều nhưng cần ghi lại số thật ở epoch đầu.
- ASR huấn luyện trên transcript chữ thường không dấu câu, trong khi biến thể "sạch" của SFT giữ chữ hoa/dấu câu; xử lý sau khi có ASR đã fine-tune.
- Điểm Qwen3-1.7B zero-shot trên dev chỉ là kiểm tra đường chạy (0,00), không phải baseline của Qwen3-4B.
- Bộ chấm nội bộ chưa hiệu chuẩn với leaderboard (xem `docs/EXPERIMENTS.md`).
