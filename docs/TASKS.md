# Giao việc vòng tới (từ 07/11)

Đọc `docs/RULES.md` trước. Mốc cứng: nộp Docker **16/11 23:59** (nên nộp sáng 16/11). Mỗi việc dưới đây ghi rõ ai làm, lệnh chạy, và gửi lại gì. Mọi số điểm so với **dev của cấu hình hiện tại: 56,33 (ASR `phowl_ft`) và 65,69 (transcript sạch)**, đều ghi vào `docs/EXPERIMENTS.md`.

## A. Việc cho bạn có GPU (Kaggle)

### A1. Train lại LLM với dữ liệu lịch sử (ưu tiên số 1, ~2,5 giờ)
Đổi **đúng một thứ** so với `q4b_lora` (thêm 700 mẫu lấy từ các lượt trong `history`), để so sánh sạch. Lý do dựng lại biến thể ASR bằng `phowl`: `q4b_lora` đã train bằng biến thể ASR gốc, còn file `sft_train_asr.jsonl` hiện trên Kaggle đã bị ghi đè bằng bản `phowl_ft`.

```bash
git pull origin main                       # cần bản có src/lifemate/augment.py và cờ HIST/SKIP_ASR
source kaggle_env.sh                       # nạp $QWEN, $PHO như mọi lần

# 1) dựng lại dữ liệu: ASR gốc (đúng như lần train đầu) + 700 mẫu lịch sử
python -I -X utf8 scripts/build_sft.py --asr-tag phowl --hist-max 700
wc -l data/processed/sft_train.jsonl data/processed/sft_train_asr.jsonl data/processed/sft_train_hist.jsonl
#   kỳ vọng: 1390 / 1390 / 700. Nếu khác, dừng và báo.

# 2) train chỉ LLM (SKIP_ASR=1: KHÔNG train lại ASR, KHÔNG ghi đè checkpoints/phowl_ft)
SKIP_ASR=1 HIST=1 RUN=q4b_hist EPOCHS_SFT=3 SAVE_LIMIT=4 bash scripts/kaggle_pipeline.sh train
```
`SAVE_LIMIT=4` giữ checkpoint theo bước để chọn epoch. **Không dùng `AUG=1` ở lượt này** (tránh lẫn hai thay đổi). Nếu còn GPU rảnh, chạy riêng một lượt `AUG=1 HIST=1 RUN=q4b_hist_aug` (cần chạy stage `aug_data` trước, nó dựng lại file `sft_train_aug.jsonl`).

Chấm dev (cả hai kịch bản) rồi gửi lại:
```bash
TAG=phowl_ft RUN=q4b_hist bash scripts/kaggle_pipeline.sh eval_dev          # so với 56,33
bash scripts/infer_multi_gpu.sh train outputs/dev_q4b_hist_clean.jsonl --ids data/processed/dev_ids.txt \
     --backend hf --model $QWEN --adapter checkpoints/q4b_hist --text clean
python -I -X utf8 scripts/eval.py --sub outputs/dev_q4b_hist_clean.jsonl --ids data/processed/dev_ids.txt   # so với 65,69
```
**Gửi lại:** 2 file `outputs/dev_q4b_hist_*.jsonl`, số điểm (tổng, exec/ask/oos), thời gian mỗi mẫu, đường cong loss, `sha256sum` adapter. Tôi sẽ so từng cặp mẫu với `q4b_lora` (dev chỉ 155 mẫu, sai số khoảng ±4 điểm nên không chỉ nhìn điểm tổng).

### A2. Đo WER của `phowl_ft` trên chính audio train (~5 phút, kiểm tra một giả thuyết)
Giả thuyết: ASR đã fine-tune trên audio train nên gần như không nhiễu khi chạy lại trên train, vì vậy biến thể ASR của SFT dựng từ `phowl_ft` không giống nhiễu thật.
```bash
python -I -X utf8 scripts/asr_wer.py --tag phowl_ft            # toàn bộ train
python -I -X utf8 scripts/asr_wer.py --tag phowl_ft --dev-only # đã biết: 9,77%
```
Gửi cả hai con số. Nếu WER trên train **thấp hơn nhiều** dev (ví dụ dưới 3%), giả thuyết đúng và ta sẽ làm ASR out-of-fold.

### A3. Kiểm tra A30 hoặc GPU tương đương (nếu có)
Chỉ cần chạy `scripts/run_infer.py` trên 50 mẫu dev và ghi mean/p95/max thời gian mỗi mẫu. Kaggle p95 hiện khoảng 3 giây; vẫn chưa có số liệu trên phần cứng chấm (1×A30).

## B. Việc làm tay, không cần GPU (ai cũng làm được)

| # | Việc | Làm thế nào | Gửi lại |
|---|---|---|---|
| B1 | Xác nhận mọi Kaggle dataset và notebook chứa dữ liệu ở chế độ **Private**, Internet OFF, và không còn output notebook chứa transcript | Kaggle > Datasets/Notebooks > Settings; xem `docs/KAGGLE.md` mục 0 | ảnh chụp trạng thái (che tên người khác) hoặc xác nhận bằng chữ |
| B2 | Đối chiếu checksum Qwen3-4B với bản chính thức | Tải `Qwen/Qwen3-4B` từ Hugging Face (tài khoản của bạn), `sha256sum` 3 file shard, so với bảng trong `docs/MODELS.md` | kết quả khớp / không khớp. Adapter được train trên bản lấy từ một Kaggle dataset, không phải bản tải trực tiếp |
| B3 | Điền licence thật cho các model | Đọc model card của `vinai/PhoWhisper-large` và `Qwen/Qwen3-4B`, ghi tên licence và đường dẫn vào cột Licence của `docs/MODELS.md`. Hiện bảng ghi "tag từ Hugging Face, cần đối chiếu" | PR hoặc ghi chú vào file |
| B4 | Hỏi BTC (cổng cuộc thi hoặc kênh hỗ trợ) | (a) **Hướng dẫn đóng gói Docker** (đề nói có tài liệu riêng, ta chưa nhận); (b) cách đếm tham số: theo checkpoint phát hành hay theo phần thực sự nạp; (c) lệnh chạy và đường dẫn cụ thể khi họ mount `/data` | câu trả lời nguyên văn |
| B5 | Báo hạn mức GPU Kaggle còn lại trong tuần | trang Settings của tài khoản | số giờ còn lại; quyết định cắt thử nghiệm nào |
| B6 | Kiểm tra mẫu | chạy thử A1 bước 1 và gửi dòng `wc -l` | 3 con số |

## C. Việc của tôi (máy local, không cần GPU)
Đã xong: khai thác lượt `history` (ý 1), cờ `HIST` và `SKIP_ASR` trong pipeline, tài liệu này.

Tiếp theo, theo thứ tự:
1. Đổi thực thể đồng bộ trong câu, tham số và kết quả (ý 2; rủi ro: đổi sót một chỗ sẽ tạo nhãn sai mà vẫn trông hợp lệ).
2. Mẫu `ask` bằng cách xóa tham số khỏi câu, và mẫu `oos` thêm (ý 4).
3. Dev theo từng bước và dev app lạ (giấu vài app khỏi train, đo trên mẫu dùng chúng).
4. Script so sánh từng cặp mẫu giữa hai bản chạy dev.
5. Khung Docker v0 (chạy offline, đọc `/data`, ghi `result/submission.jsonl`), chờ B4(a) để chỉnh cho đúng.

## D. Điều kiện chốt
- **Đến 13/11:** chốt cấu hình, ngừng thêm tính năng. Sau mốc này chỉ sửa lỗi và đóng gói.
- **Chỉ giữ một thay đổi nếu** dev tăng rõ khi so từng cặp mẫu, thời gian p95 vẫn thoải mái, và hai lần chạy cho kết quả giống từng byte.
- Không dùng lượt nộp public để dò đáp án (RULES 5.1); chỉ nộp khi dev tăng rõ, cần người phụ trách đồng ý.
- Không đưa dữ liệu, audio, transcript, dự đoán, trọng số vào git. Gửi adapter và file dev qua Drive riêng của đội.
