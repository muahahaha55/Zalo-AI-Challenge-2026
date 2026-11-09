# Hướng dẫn chạy trên GPU (phiên 08/11): torch, A4, A5

Dành cho thành viên chạy GPU trên Kaggle. Chi tiết từng lệnh ở `docs/TASKS.md` (mục 0, A4, A4b, A5); file này chỉ nói thứ tự, điều kiện và cái cần gửi lại.

## Quy tắc không đổi
- Không đưa dữ liệu cuộc thi (câu nói, bản chép lời, call, catalog) vào Claude, Copilot hay bất kỳ API ngoài nào. Model sinh câu phải chạy **trên máy Kaggle**.
- Notebook và dataset có dữ liệu luôn **Private**; đặt Private **trước khi** chạy cell đầu tiên. Internet **Off**.
- Không ghi đè adapter `q4b_lora` (adapter đang dùng cho bản nộp). Mỗi thí nghiệm dùng `RUN=` tên mới.
- Hết hoặc sắp hết giờ GPU thì báo ngay kèm số giờ còn lại, để cắt bớt thí nghiệm theo thứ tự ở dưới.

## Bước 0. Đưa mã mới lên Kaggle (thành viên chạy GPU tự làm)
Dataset mã trên Kaggle đang là bản cũ: chưa có `callgen`, `check_models`, các sửa pipeline.
```bash
git clone https://github.com/muahahaha55/Zalo-AI-Challenge-2026.git     # bản sạch, không có data/
cd Zalo-AI-Challenge-2026 && git log --oneline -1                        # phải là 783526d hoặc mới hơn
bash scripts/kaggle_pack.sh code                                          # tạo kaggle_upload/lifemate-code.zip (chỉ đóng gói, không tải lên)
```
Rồi: Kaggle > Datasets > `lifemate-code` > New Version > kéo zip vào > kiểm tra dataset vẫn **Private**. Trong notebook, gắn phiên bản mới nhất của dataset.

Kiểm tra nhanh trong notebook (không cần GPU): `python -I -X utf8 scripts/preflight_train.py` phải báo đạt, và `ls` thấy `src/lifemate/callgen.py`.

## Bước 1. Gửi lại ngay (5 phút, phiên có GPU)
```python
import torch, sys; print(torch.__version__, torch.version.cuda, sys.version)
```
Gửi nguyên dòng in ra. File `pip freeze` lần trước lấy trong phiên **chỉ có CPU** (torch 2.11.0+cpu), nên chưa biết bản CUDA đã dùng để train. Docker cần biết.

## Cập nhật 09/11: A4 đã xong (cảm ơn). Việc tiếp theo là A4b, rồi A5 có điều kiện
Kéo bản mới của `main` (commit `3ee6803` trở lên) và gắn lại dataset mã mới nếu chưa. Chi tiết A4b ở `docs/TASKS.md` mục A4b.

### A4b: tham chiếu "app đã thấy" (khoảng 10 phút GPU, làm trước)
Chạy chính `q4b_holdout` (adapter còn trong output notebook A4) trên 132 mẫu dev dùng app đã thấy, chữ sạch, luật bật như A4:
```bash
bash scripts/infer_multi_gpu.sh train outputs/seen_q4b_holdout.jsonl --ids data/processed/holdout/dev_ids.txt      --backend hf --model $QWEN --adapter checkpoints/q4b_holdout --text clean
python -I -X utf8 scripts/eval.py --sub outputs/seen_q4b_holdout.jsonl --ids data/processed/holdout/dev_ids.txt
```
Nếu tiện, chạy thêm `q4b_holdout_hist` trên cùng tệp. Gửi lại: điểm tổng, exec, ask, oos của từng cái.
**Cách đọc:** khoảng cách = (điểm app đã thấy) trừ (41,93 của app lạ). Dưới khoảng 6 điểm: **bỏ A5**. Trên khoảng 15 điểm: **làm A5**. Ở giữa: báo lại, ta cân nhắc theo giờ GPU còn lại.

### A5: chỉ làm sau khi biết kết quả A4b
Cần một model lớn hơn 1,7B cho bước viết câu (1,7B đã thử và viết sai gần hết; mức đủ tốt chưa đo). **Thử theo thứ tự rẻ đến đắt: Qwen3-8B và Qwen3.5-9B (thử song song, mỗi cái 200 mẫu), rồi 14B, rồi 32B** (32B bf16 vừa 96 GB, 72B không vừa), mỗi lần 200 mẫu rồi đọc 30 câu, dừng ở model nhỏ nhất cho câu chấp nhận được. **Model viết câu không phải model nộp bài**: bài nộp vẫn là Qwen3-4B + adapter. Model này chỉ tạo dữ liệu, không nằm trong bài nộp nên giới hạn 5B/6B không áp dụng. Cách lấy mà vẫn giữ Internet Off: trong notebook, Add Input > Models, chọn biến thể Qwen3 tương ứng trong Kaggle Models; không tải từ Hugging Face trong phiên. Không đưa dữ liệu cuộc thi lên Kaggle Model công khai. Nếu Kaggle không có sẵn, báo lại để tìm cách khác.
Bước thử (chỉ 200 mẫu, khoảng 15 đến 20 phút kể cả nạp model):
```bash
python -I -X utf8 scripts/gen_calls.py --n 200
python -I -X utf8 scripts/gen_text.py --model <thư mục model 8B, 14B hoặc 32B> --inp data/processed/callgen_pending.jsonl        --out data/processed/callgen_text.jsonl --batch 32
```
Qwen3.5-9B là model đa phương thức, chưa thử nạp: `gen_text.py` tự thử `AutoModelForImageTextToText` nếu nạp thường lỗi; nếu vẫn lỗi hoặc mẫu chat không nhận `enable_thinking`, bỏ qua và báo lỗi nguyên văn. Bước kiểm tra khứ hồi `verify_callgen.py` dùng `AutoModelForCausalLM`, nên dùng Qwen3-8B (hoặc 14B) cho bước đó, không dùng Qwen3.5.
Gửi lại **30 dòng ngẫu nhiên** của `callgen_text.jsonl` (trường `audio_text` và `label.tool_calls`) để duyệt chất lượng câu **trước khi** sinh số lượng lớn hoặc train. Không train A5 khi chưa có duyệt.

## Bước 2. A4: đo "app lạ" (ưu tiên số 1, hạn đề xuất **10/11**)
Giấu 6 app (`flights, bike_rental, package_tracking, shopping_list, podcast_radio, loyalty`) khỏi mọi dữ liệu train, train, rồi đo trên các mẫu dùng chúng. Làm đúng theo mục A4 trong `docs/TASKS.md`.

Chạy hai cấu hình trên cùng bộ chia để so:
| Tên | Dữ liệu train | Mục đích |
|---|---|---|
| `q4b_holdout` | `sft_train.jsonl` + `sft_train_asr.jsonl` của holdout | mốc: model chưa thấy app nào trong 6 app |
| `q4b_holdout_hist` | như trên + `sft_train_hist.jsonl` (`--hist-max 300`) | xem dữ liệu lịch sử có giúp trên app lạ không |

Gửi lại: điểm tổng và điểm exec / ask / oos của từng cấu hình, và file đầu ra `outputs/unseen_*.jsonl`.

**Lưu ý:** lệnh A4 trong TASKS dùng `--asr-tag phowl` (bản ghi của ASR gốc), khác với `q4b_lora` (dùng `phowl_ft`). Vì thế **A4 không so trực tiếp với điểm dev 56,98 của `q4b_lora`**; chỉ so các cấu hình A4 với nhau.

## Bước 3. A6 (ASRX): ĐÃ RÚT LẠI, không chạy
Tiền đề của A6 sai (`q4b_lora` đã học trên bản ghi `phowl`, và `aug_data` ghi lại `sft_train_asr.jsonl` bằng `phowl_ft`), nên A6 không đo được điều định đo. Mã ASRX đã gỡ khỏi `main`. Cảm ơn bên chạy GPU đã đối chiếu và phát hiện. Làm A4 trước, rồi A5 nếu cần.

## Bước 4. A5: dữ liệu ghép đoạn call (chỉ khi A4 cho thấy app lạ là vấn đề)
Không chạy đủ số lượng ngay. Làm bước thử trước:
```bash
python -I -X utf8 scripts/gen_calls.py --n 200
python -I -X utf8 scripts/gen_text.py --model <thư mục Qwen3 8B/14B/32B trong Kaggle Model hoặc Dataset Private> \
       --inp data/processed/callgen_pending.jsonl --out data/processed/callgen_text.jsonl --batch 32
```
Gửi lại **30 dòng ngẫu nhiên** của `callgen_text.jsonl` (trường `audio_text` và `label.tool_calls`) để duyệt chất lượng câu **trước khi** sinh số lượng lớn. Model 1,7B đã thử và viết sai gần hết, nên đừng dùng cỡ đó.

## Thứ tự cắt khi thiếu giờ GPU
1. Bước 1 (torch) luôn làm, rất rẻ.
2. A4 hai cấu hình; nếu thiếu giờ, chỉ chạy `q4b_holdout`.
3. A5 (chỉ khi A4 cho thấy app lạ là vấn đề).

Nếu đến **10/11** mà A4 chưa chạy được thì bỏ A4 và A5, giữ cấu hình hiện tại (`q4b_lora` + ASR đã fine-tune + luật) và dồn sức vào Docker.

## Mốc thời gian
| Ngày | Việc |
|---|---|
| 10/11 | hạn A4 |
| 13/11 | chốt cấu hình, ngừng thêm tính năng |
| 14 đến 15/11 | dựng Docker cuối |
| 16/11 | nộp Docker (nên buổi sáng) |
| 20/11 | private_test |
