# Tái tạo quá trình huấn luyện (LifeMate)

Thư mục này có toàn bộ mã huấn luyện (`src/`, `scripts/`, `configs/`) và các tệp yêu cầu thư viện. Mã giống hệt repo của đội ở commit ghi trong `COMMIT.txt`.

## Môi trường đã dùng
- 1 GPU NVIDIA RTX PRO 6000 Blackwell (96 GB) trên Kaggle; Python 3.13; `torch 2.11.0+cu130`; `transformers 5.20.2`, `peft 0.21.2`, `accelerate 1.15.0` (danh sách đầy đủ: `requirements-kaggle.txt` và bản `pip freeze` trong repo của đội).
- GPU từ 24 GB là đủ (thời gian dài hơn). Cần Internet để tải model gốc.

## Dữ liệu
- **Dữ liệu train và public_test của BTC** (BTC đã có), đặt ở `data/raw/train+public_test/` (gồm `train/` và `public_test/` như bản phát hành). Đội không phát tán lại dữ liệu này.
- Mọi dữ liệu trung gian (bản chép ASR, dữ liệu SFT) được **sinh lại bằng mã** từ dữ liệu trên; không có dữ liệu ngoài.

## Model gốc (tải từ Hugging Face; không nằm trong Docker)
- LLM: `Qwen/Qwen3-4B` (https://huggingface.co/Qwen/Qwen3-4B)
- ASR: `vinai/PhoWhisper-large` (https://huggingface.co/vinai/PhoWhisper-large); `scripts/prepare_models.py` đổi bản gốc (pickle) sang safetensors.

## Các bước (đúng thứ tự đã chạy để ra model nộp)
Từ thư mục gốc của mã, với `QWEN=<thư mục Qwen3-4B>` và `PHO=<thư mục PhoWhisper-large>`:
```bash
pip install -r requirements-kaggle.txt                      # torch cài sẵn theo GPU
python scripts/prepare_models.py                             # tải model gốc, đổi PhoWhisper sang safetensors
# 1. ASR gốc chép toàn bộ audio train và public_test (văn bản nhiễu cho SFT)
python scripts/run_asr.py --model $PHO --tag phowl --split train
python scripts/run_asr.py --model $PHO --tag phowl --split public_test
# 2. Dựng dữ liệu SFT (chia dev cố định, seed 42) gồm bản câu sạch và bản câu ASR
python scripts/build_sft.py --asr-tag phowl
# 3. Fine-tune ASR (LoRA r=32, 3 epoch, lr 1e-4, batch 8 x accum 2, seed 42; mẫu dev bị loại khỏi train), tự gộp vào checkpoints/phowl_ft/merged
python scripts/train_asr.py --model $PHO --out checkpoints/phowl_ft --epochs 3 --save-steps 100
# 4. Fine-tune LLM (LoRA r=64, alpha 128, dropout 0,05, lr 2e-4 cosine, warmup 3%, 3 epoch, batch tổng 16, max-len 5120, seed 42)
torchrun --nproc_per_node=1 scripts/train_sft.py --model $QWEN --data data/processed/sft_train.jsonl data/processed/sft_train_asr.jsonl \
    --out checkpoints/q4b_lora --epochs 3 --max-len 5120 --global-batch 16 --save-limit 2
# 5. Gộp LoRA vào Qwen3-4B (bản dùng lúc suy luận)
python scripts/merge_lora.py --base $QWEN --adapter checkpoints/q4b_lora --out checkpoints/q4b_merged
```
Kết quả: `checkpoints/q4b_merged` (LLM) và `checkpoints/phowl_ft/merged` (ASR), là `saved_models/llm` và `saved_models/asr` trong Docker. `scripts/kaggle_pipeline.sh` chạy cùng các bước này theo từng giai đoạn.

## Kiểm tra
- `python scripts/eval.py --sub <file> --ids data/processed/dev_ids.txt`: chấm trên dev bằng bộ chấm nội bộ (khớp hash của BTC trên mẫu nộp mẫu).
- `python scripts/check_models.py`: so sha256 trọng số với `models.lock.json`.

## Seed và tính tái lập
Seed 42 cho chia dev, train ASR và train LLM; seed 0 và giải mã greedy khi suy luận. Huấn luyện trên GPU có thể khác nhau ở bit cuối giữa các lần chạy hoặc các loại GPU (thao tác cộng dồn không xác định của cuDNN/cuBLAS), nên model tái tạo có thể lệch nhẹ so với model nộp; checksum model nộp nằm trong `models.lock.json`.
