# Ý tưởng: che tên app và tool khi train để model đọc mô tả (A8, ĐÃ CHẠY: không đạt)

> **Kết quả (13/11, 4 × L4):** không đạt tiêu chí. App lạ 41,93 → 44,49 (+2,56 [−3,91; +9,01]), app đã thấy 66,03 → 67,67 (+1,64). exec tăng ở cả hai tập và mẫu gọi tool ngoài nhãn giảm (đúng chiều giả thuyết), nhưng trong nhiễu; ask tụt. Giữ bài nộp hiện tại. Chi tiết: `docs/BAN_GIAO_A8.md`.

Ngày 2026-11-13. Người viết: thuan (cùng Claude). Trạng thái lúc viết: **mã đã viết và kiểm tra trên máy local, chưa train**. Phương đã quyết dừng train sau A5, rồi cho phép thử A8. Mã mới **không đổi** đầu ra cũ của `build_sft.py` (đã kiểm từng byte, mục 5) và không đụng đến suy luận.

## 1. Vấn đề cần giải
- App lạ là chỗ mất điểm lớn nhất: cùng một model, **41,93** trên app lạ so với **66,03** trên app đã thấy (A4, A4b). Public có **65%** mẫu chứa app lạ; điểm public 48,74 thấp hơn dev ASR (56,98) khoảng 8 điểm.
- Lỗi chính trên app lạ là **chọn nhầm tool**: 48 đến 52% mẫu 0 điểm gọi một tool có thật nhưng ngoài tập tool của nhãn (`docs/BAN_GIAO_A4.md`).
- Thêm dữ liệu cùng kiểu không giúp: A1 (`history`) không phân biệt được với 0; A5 (907 mẫu tự sinh) làm exec đứng yên, ask và oos tụt (`docs/BAN_GIAO_A5_TRAIN1.md`).

Giả thuyết: model chọn tool theo **tên đã thuộc** trong train (loss rơi tới 0,004) thay vì đọc mô tả. Với app lạ, tên không có trong trí nhớ nên nó chọn tool "quen" gần nhất.

## 2. Ý tưởng và nguồn
**Che tên (function masking)**, theo Hammer (Lin và cộng sự, 2024, arXiv:2410.04587): trong một bản sao của mẫu train, mọi tên app và tên tool bị thay bằng chuỗi ngẫu nhiên, **nhất quán trong mẫu đó và khác nhau giữa các mẫu**. Model không thể chọn tool theo tên nhớ được nữa, phải đọc mô tả, đúng thứ mà app lạ lúc thi cung cấp. Hammer dùng cách này trên các model cỡ 1,5B đến 7B (có Hammer-4B dựng từ Qwen1.5-4B-Chat). RoTBench (Ye và cộng sự, EMNLP 2024) cho thấy chỉ đổi tên tool cũng đủ làm model không dùng được tool đó.

**Kèm 100 mẫu oos** (`--aug-oos 100`, đã có trong `build_sft.py` từ 06/11 nhưng chưa từng train): bỏ hết app mà nhãn dùng, đích là `END`. Hammer thêm khoảng 10% mẫu "không có hàm phù hợp" và thấy mức đó tốt nhất trong thiết lập của họ; 100 mẫu ở đây là khoảng 3% số hàng train (3640) và 8,5% số mẫu gốc (1180). A5 làm ask và oos tụt, và giả thuyết đã ghi là A5 không có mẫu loại này; thêm oos là để bù chiều đó.

**Mức độ bằng chứng (nói thẳng):** Hammer **không có bảng so sánh có che và không che** trên cùng benchmark; ablation duy nhất (Qwen2-1.5B) chỉ cho xu hướng qua đồ thị: che nhiều thì học chậm hơn trên tập train nhưng tổng quát tốt hơn sang API-Bank. Mức tăng lớn trong bảng của họ gộp cả che tên lẫn dữ liệu thêm. Không bài nào làm với tiếng Việt, đầu vào ASR hay metric "lệch state hash là 0 điểm". Vì vậy chỉ được coi là giả thuyết đáng thử, chưa phải kết quả.

## 3. Che gì, giữ gì (và lý do)
| Thành phần trong prompt | Xử lý | Lý do |
|---|---|---|
| Tên tool `app.tool` trong `[TOOL]`, nhãn, call trong `[LỊCH SỬ]` | `<bí danh app>.<bí danh tool>` | Mục đích chính |
| Bí danh app | một bí danh cho mọi tool cùng app trong mẫu | Giữ cấu trúc nhóm theo app |
| `tool_name` của `sys.ask_missing_info` | theo bí danh | Nếu không thì nhãn ask chỉ vào tool không có trong prompt |
| Tên tool được nhắc trong mô tả (`ewallet.pay_qr` nhắc `food_delivery.list_restaurants`) | theo bí danh, kể cả tool của app không cài | Nếu không thì tên thật lọt ra (đã gặp: 251 mẫu trước khi sửa) |
| `x-ref` (`contacts.contacts`) | che phần app, **giữ** từ sau dấu chấm | Từ sau dấu chấm là nghĩa ("danh bạ"), giống mô tả |
| **Tên tham số** | **giữ nguyên** | Prompt (`prompt.compact_tool`) không in mô tả tham số; che tên thì model không còn gì để đoán nghĩa. Đây là chỗ **khác Hammer** (Hammer che cả tên tham số vì họ có mô tả tham số) |
| `sys.ask_missing_info`, mô tả tool, câu yêu cầu, kết quả tool | giữ nguyên | Lúc thi chúng y như vậy |

Bí danh là chữ thường và số, 4 đến 6 ký tự, bắt đầu bằng chữ, không trùng bất kỳ từ nào trong tên app hoặc tool thật. Mỗi mẫu một bảng bí danh mới; seed cố định nên chạy lại ra đúng từng byte.

## 4. Mã đã viết
- `src/lifemate/masking.py` (mới): `mask_sample(item, catalog, rng)` trả về mẫu và catalog đã che cùng bảng bí danh, không sửa đầu vào; `masked_samples(train_items, catalog, asr, rng, n)` sinh tối đa `n` hàng `"<id>#mask"`, mỗi mẫu gốc nhiều nhất một lần, chữ ASR với xác suất 0,5 khi có `--asr-tag` (như `synthetic_samples`), và kiểm khứ hồi nhãn trên từng hàng.
- `scripts/build_sft.py`: thêm cờ `--aug-mask N` (mặc định 0, tức tắt) ghi `sft_train_mask.jsonl`; seed riêng (`seed + 5`), chỉ lấy mẫu phần train, chạy được với `--hold-apps`.
- `tests/test_masking.py` (mới, 7 test): che nhất quán, không sửa đầu vào, không còn từ thật nào của app hay tool trong prompt (trừ từ sau dấu chấm của `x-ref`, cố ý), `tool_name` của ask theo bí danh, khứ hồi nhãn và `END`, bí danh khác nhau giữa các mẫu và tái lập theo seed, trộn chữ ASR, bỏ mẫu có tool không có trong catalog.
- **Không sửa** `prompt.py`, `agent.py`, `pipeline.py`, `train_sft.py`: suy luận và bài nộp hiện tại không đổi.

## 5. Đã kiểm tra (máy local, dữ liệu train thật, chỉ in số tổng)
| Kiểm tra | Kết quả |
|---|---|
| `pytest tests` | 116 test qua (gồm 7 test mới) |
| `ruff check .`, `scripts/check_compliance.py` | qua |
| Che toàn bộ phần train của bộ chia A4 (1312 mẫu) | 1312 hàng, 0 mẫu bị bỏ, 0 tên tool thật trong nhãn, 0 tên tool hoặc tiền tố app thật trong prompt, 0 nhắc tới app bị giấu |
| Độ dài | token của tin nhắn đầu (400 hàng) tăng trung vị **+11,8%** (p95 +16,5%, tối đa +20,6%; bí danh ngẫu nhiên tách nhiều token hơn tên thật) |
| `build_sft.py --asr-tag phowl --hold-apps <6 app> --aug-oos 100 --aug-mask 1180` | 1180 hàng che (579 dùng chữ ASR `phowl`), 100 hàng oos |
| `preflight_train.py` (tokenizer Qwen3-4B, `--max-len 5120`, `--dev-ids`, `--hold-apps`) | **OK** cả 4 file; bước dài nhất của file che 4621 token, **0 bước quá 5120**; **11437 bước mỗi epoch** (A4 là 7558, A5 là 10549) |
| Các file cũ khi bật cờ | `sft_train`, `sft_train_asr`, `sft_dev`, `sft_dev_asr`, `sft_train_aug`, `eval_ids` **trùng từng byte** với lần chạy không có `--aug-mask` |
| Tái lập | chạy lại cùng lệnh ra `sft_train_mask.jsonl` trùng từng byte |

## 6. Thí nghiệm đề xuất (A8, một lần train)
Cùng bộ chia, cùng công thức và cùng tiêu chí như A5, để so thẳng với mốc `q4b_holdout`:
```bash
HOLD=flights,bike_rental,package_tracking,shopping_list,podcast_radio,loyalty
python -X utf8 scripts/build_sft.py --asr-tag phowl --hold-apps $HOLD --aug-oos 100 --aug-mask 1180
H=data/processed/holdout
python -X utf8 scripts/preflight_train.py --data $H/sft_train.jsonl $H/sft_train_asr.jsonl $H/sft_train_aug.jsonl $H/sft_train_mask.jsonl \
  --dev-ids $H/dev_ids.txt --hold-apps $HOLD --tokenizer $QWEN --max-len 5120
python -X utf8 scripts/train_sft.py --model $QWEN --data $H/sft_train.jsonl $H/sft_train_asr.jsonl $H/sft_train_aug.jsonl $H/sft_train_mask.jsonl \
  --out checkpoints/q4b_holdout_mask --epochs 3 --max-len 5120 --global-batch 16 --save-limit 2
bash scripts/infer_multi_gpu.sh train outputs/unseen_q4b_holdout_mask.jsonl --ids $H/eval_ids.txt --backend hf --model $QWEN --adapter checkpoints/q4b_holdout_mask --text clean
bash scripts/infer_multi_gpu.sh train outputs/seen_q4b_holdout_mask.jsonl --ids $H/dev_ids.txt --backend hf --model $QWEN --adapter checkpoints/q4b_holdout_mask --text clean
python -X utf8 scripts/compare_runs.py --a <unseen_q4b_holdout.jsonl của A4> --b outputs/unseen_q4b_holdout_mask.jsonl --ids $H/eval_ids.txt --names q4b_holdout,q4b_holdout_mask
python -X utf8 scripts/compare_runs.py --a <seen_q4b_holdout.jsonl của A4b> --b outputs/seen_q4b_holdout_mask.jsonl --ids $H/dev_ids.txt --names q4b_holdout,q4b_holdout_mask
```
- **Tiêu chí (giữ như A5):** app lạ tăng **ít nhất 5 điểm** so với 41,93 **và** app đã thấy không giảm quá 3 điểm so với 66,03.
- **Thời gian ước:** 11437 bước mỗi epoch; A5 với 10549 bước mất 2 giờ 52 phút, nên khoảng **3 giờ 5 phút đến 3 giờ 20 phút** train (prompt che dài hơn khoảng 12%), cộng khoảng 20 phút đo, trên một GPU RTX PRO 6000. Trên **L4 x4** (24 GB mỗi card) ước **4 đến 6 giờ** train (chưa đo); khi có nhiều GPU phải train bằng `torchrun --nproc_per_node=<số GPU>` để batch tổng vẫn là 16 (chạy `python train_sft.py` thì Trainer tự dùng DataParallel và batch tổng thành 16 nhân số GPU). Notebook `a8-mask-holdout` tự làm việc này. So với mốc `q4b_holdout` (train trên 1 GPU RTX PRO 6000) có thêm một nguồn sai khác nhỏ: khác GPU và chia dữ liệu theo DDP, cùng batch tổng và cùng số bước.
- Gộp hai thay đổi (che tên và oos) trong một lần train nên **không tách được** cái nào có tác dụng; chấp nhận vì chỉ có thời gian cho một lần. Đọc thêm theo loại mẫu: exec trên app lạ cho biết phần che tên; ask và oos cho biết phần oos.

## 7. Nếu đạt: bản nộp
Train toàn bộ (không `--hold-apps`): `build_sft.py --asr-tag phowl --aug-oos 100 --aug-mask 1390`, train `sft_train + sft_train_asr + sft_train_aug + sft_train_mask` 3 epoch như `q4b_lora`, gộp bằng `merge_lora.py`, đo dev (phải không thấp hơn 66,82 sạch và 56,98 qua ASR), rồi dựng image, A7a và Test Solution. Khoảng **3,5 giờ train + 1 giờ gộp và đo + dựng image**: tức phải bắt đầu chậm nhất sáng 14/11 để kịp Test Solution trước 23:59 ngày 15/11. Trễ mốc nào thì **giữ bài nộp hiện tại (48,74)**. `submission/training_code/README.md` phải thêm hai cờ này nếu dùng.

## 8. Rủi ro và giới hạn
- Bằng chứng gián tiếp (mục 2). Theo phán đoán, khả năng đạt ngưỡng +5 là **dưới 50%**; nếu đạt đúng +5 trên app lạ thì điểm public ước tăng khoảng +2 đến +3 (app lạ chỉ là một phần của 65% mẫu có app lạ).
- Che tên làm việc học trên tập train chậm hơn (đúng như Hammer báo), nên app đã thấy có thể giảm.
- Một lần train, một seed, một cách chia app như A4 và A5; chênh dưới khoảng 5 điểm nằm trong nhiễu (khoảng tin cậy của A5 rộng khoảng 13 điểm).
- Tên tham số không che (mục 3), nên model vẫn có thể dựa vào tên tham số quen.

## 9. Nguồn
- Lin, Q. và cộng sự (2024). *Hammer: Robust Function-Calling for On-Device Language Models via Function Masking*. arXiv:2410.04587.
- Ye, J. và cộng sự (2024). *RoTBench: A Multi-Level Benchmark for Evaluating the Robustness of Large Language Models in Tool Learning*. EMNLP 2024. arXiv:2401.08326.
- Hsieh, C.-Y. và cộng sự (2023). *Tool Documentation Enables Zero-Shot Tool-Usage with Large Language Models*. arXiv:2308.00675.
