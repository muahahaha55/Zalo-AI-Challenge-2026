# Giao việc vòng tới (từ 07/11)

Đọc `docs/RULES.md` trước. Mốc cứng: nộp Docker **16/11 23:59** (nên nộp sáng 16/11). Mỗi việc dưới đây ghi rõ ai làm, lệnh chạy, và gửi lại gì. Mọi số điểm so với **dev của cấu hình hiện tại: 56,33 (ASR `phowl_ft`) và 65,69 (transcript sạch)**, đều ghi vào `docs/EXPERIMENTS.md`.

## 0. Lưu ý bắt buộc sau lượt A1 (07/11), đọc trước khi chạy tiếp

1. **Notebook và output trên Kaggle phải ở chế độ Private** (RULES 1.3). Phát hiện ngày 07/11: notebook `train-hist` đang **công khai**, và output của nó chứa dẫn xuất của dữ liệu cuộc thi (bản chép lời ASR của audio train và public_test, các file `sft_*.jsonl`, `dev_ids.txt`, `gold_hash_train.json`). **Chủ notebook cần đặt Private ngay** (Settings, Sharing), kiểm tra mọi notebook và dataset khác có gắn dữ liệu, và gỡ các phiên bản đã lưu nếu cần. Sau này mỗi notebook mới phải đặt Private **trước khi** chạy cell đầu tiên.
2. **Stage `train` và `eval_dev` cần `FORCE=1`** khi các dấu `outputs/.done_*` được khôi phục từ phiên trước (không có thì stage báo "already done" và không chạy gì). Log nay được **nối thêm** (`tee -a`), không còn ghi đè.
3. **Muốn chọn epoch thì đặt `SAVE_STEPS` bằng số bước mỗi epoch** (ví dụ khoảng 650 cho lượt có 10.404 bước mỗi epoch) cùng `SAVE_LIMIT` đủ lớn (6 trở lên). Mỗi checkpoint khoảng 1,5 GB và output Kaggle giới hạn 20 GB (lượt A1 đã dùng khoảng 14 GB), nên xóa các checkpoint cũ trước.
4. **Gửi `pip freeze`** (Docker cần ghim đúng phiên bản): thêm một cell vào notebook, chạy trong phiên Kaggle đã train, rồi gửi file: `pip freeze > /kaggle/working/freeze.txt; python -c "import torch,transformers,peft;print(torch.__version__,transformers.__version__,peft.__version__)"`. Phiên bản bàn giao A1 ghi: transformers 5.20.2, peft 0.21.2, torch 2.11.0+cu130, chưa có file chứng minh.

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

### A1b. Thêm dữ liệu "lời rút lại" (làm sau A1, so với A1)
Người nói tự sửa lời ("à khoan", "à thôi", "khỏi") chiếm 6,3% train, trong đó 31% là mẫu ngoài phạm vi. Trên dev các mẫu này chỉ đạt 40,6 điểm so với 67,6 của phần còn lại (chỉ 11 mẫu nên chưa chắc), model hay làm luôn phần đã bị rút lại. Dữ liệu gồm: các mẫu rút lại **thật** nhân bản 2 lần (nhãn thật, an toàn), cộng vài chục câu ghép "câu ngắn, à thôi, yêu cầu thật" và "câu ngắn, à thôi khỏi" (câu bị rút lại luôn là câu đơn, ngắn, không chứa sẵn dấu hiệu rút lại hay phủ định, để phạm vi "bỏ đi" không mơ hồ).
```bash
bash scripts/kaggle_pipeline.sh aug_data      # FORCE=1 nếu đã chạy; sinh thêm sft_train_retract.jsonl (khoảng 254 mẫu)
SKIP_ASR=1 HIST=1 RETRACT=1 RUN=q4b_retract EPOCHS_SFT=3 SAVE_LIMIT=4 bash scripts/kaggle_pipeline.sh train
```
Chỉ giữ nếu dev tăng so với A1 và mẫu có lời rút lại không tệ đi (xem `scripts/compare_runs.py`). Để chỉ dùng mẫu thật, đặt `RETRACT_REPLACE=0 RETRACT_CANCEL=0` khi chạy `aug_data`.

### A3. Kiểm tra A30 hoặc GPU tương đương (nếu có)
Chỉ cần chạy `scripts/run_infer.py` trên 50 mẫu dev và ghi mean/p95/max thời gian mỗi mẫu. Kaggle p95 hiện khoảng 3 giây; vẫn chưa có số liệu trên phần cứng chấm (1×A30).

### A4. Thử "app lạ" (sau A1; đo khái quát hóa thật, ~2 giờ vì ít mẫu hơn)
Hiện dev chỉ có app đã thấy, nên điểm dev lạc quan so với public (65% mẫu public có app lạ). Bộ chia này giấu 6 app (`flights, bike_rental, package_tracking, shopping_list, podcast_radio, loyalty`) khỏi mọi chỗ trong dữ liệu train, kể cả khỏi danh sách app trong prompt, rồi đo trên các mẫu dùng chúng.
```bash
python -I -X utf8 scripts/build_sft.py --asr-tag phowl --hold-apps flights,bike_rental,package_tracking,shopping_list,podcast_radio,loyalty --hist-max 300
#   ghi vào data/processed/holdout/ (không đụng file chính). Kỳ vọng: ~1180 mẫu train, 174 mẫu đánh giá
python -I -X utf8 scripts/train_sft.py --model $QWEN --data data/processed/holdout/sft_train.jsonl data/processed/holdout/sft_train_asr.jsonl \
     --out checkpoints/q4b_holdout --epochs 3 --max-len 5120 --global-batch 16 --save-limit 2
bash scripts/infer_multi_gpu.sh train outputs/unseen_q4b_holdout.jsonl --ids data/processed/holdout/eval_ids.txt \
     --backend hf --model $QWEN --adapter checkpoints/q4b_holdout --text clean
python -I -X utf8 scripts/eval.py --sub outputs/unseen_q4b_holdout.jsonl --ids data/processed/holdout/eval_ids.txt
```
Gửi điểm (tổng, exec/ask/oos) và file đầu ra. Đây là mốc "chưa thấy app" để so các ý sau (che tên, ghép app). Mỗi lần so thêm một ý thì train lại với cùng bộ chia này.

**Lưu ý quan trọng sau khi kéo bản mới:** bản cũ của bước sinh biến thể (`--aug-variants`) có lỗi: với mẫu `ask`, nó coi app mà câu hỏi nhắc tới là "không liên quan" và có thể xóa app đó khỏi danh sách đã cài (161/217 mẫu `ask` bị ảnh hưởng, khoảng 30% biến thể của chúng bị hỏng). Đã sửa. **File `sft_train_aug.jsonl` đã dựng trước đó trên Kaggle cần dựng lại** (`bash scripts/kaggle_pipeline.sh aug_data` với `FORCE=1`) trước khi dùng `AUG=1`.

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
Đã xong: khai thác lượt `history` (ý 1), cờ `HIST` và `SKIP_ASR` trong pipeline, bộ chia "app lạ" (A4), `scripts/compare_runs.py` (so từng cặp mẫu, có khoảng tin cậy; trên dev chỉ phát hiện được chênh lệch từ khoảng 6 điểm trở lên), sửa lỗi biến thể `ask`, tài liệu này.

Tiếp theo, theo thứ tự:
1. ~~Đổi thực thể đồng bộ~~ **bỏ**: phân tích lỗi giá trị trên dev cho thấy chủ yếu là số học thời gian ("trước mười lăm phút" từ giờ giao hàng, "thứ ba tuần sau" lệch một ngày) và chọn ID, không phải đổi tên người hay địa điểm.
2. Mẫu `ask` bằng cách xóa tham số khỏi câu: cần xác định đúng đoạn bị xóa trong câu nói, nhãn sai nhưng trông hợp lệ nếu xác định sai; chưa làm.
3. Dev theo từng bước (cần chạy model nên làm khi có GPU).
4. Khung Docker v0 (chạy offline, đọc `/data`, ghi `result/submission.jsonl`), chờ B4(a) để chỉnh cho đúng.

## D. Điều kiện chốt
- **Đến 13/11:** chốt cấu hình, ngừng thêm tính năng. Sau mốc này chỉ sửa lỗi và đóng gói.
- **Chỉ giữ một thay đổi nếu** dev tăng rõ khi so từng cặp mẫu, thời gian p95 vẫn thoải mái, và hai lần chạy cho kết quả giống từng byte.
- Không dùng lượt nộp public để dò đáp án (RULES 5.1); chỉ nộp khi dev tăng rõ, cần người phụ trách đồng ý.
- Không đưa dữ liệu, audio, transcript, dự đoán, trọng số vào git. Gửi adapter và file dev qua Drive riêng của đội.
