# Bàn giao A8: che tên app và tool khi train (`q4b_holdout_mask`, bộ chia A4)

Ngày 2026-11-13. Người chạy: thuan (Kaggle, **4 GPU L4**), notebook `notebooks/a8-mask-holdout.ipynb`, mã ở commit `d5f376c` (nhánh `thuandev`). Ý tưởng, nguồn và các kiểm tra trước khi chạy: `docs/Y_TUONG_CHE_TEN.md`. Số liệu tổng hợp ở `docs/results/a8/`. Dự đoán (`unseen_*`, `seen_*`) là dữ liệu cuộc thi nên **không** vào git, gửi qua Drive riêng (mục 7).

## 1. Tóm tắt
1. **A8 KHÔNG ĐẠT tiêu chí đặt trước** (giống A5: app lạ tăng ít nhất 5 điểm và app đã thấy không giảm quá 3 điểm). App lạ (174 mẫu): **41,93 → 44,49, chênh từng cặp +2,56** (khoảng tin cậy 95% [−3,91; +9,01], 31 tốt lên, 23 tệ đi, 120 bằng). App đã thấy (132 mẫu): **66,03 → 67,67, +1,64** [−4,84; +8,01] (17, 14, 101).
2. **Khác A5, lần này chiều hướng đúng chỗ:** exec tăng ở cả hai tập (+3,21 trên app lạ, +4,40 trên app đã thấy; A5 là −0,64 và −0,04), và số mẫu exec gọi tool ngoài tập tool của nhãn giảm (app lạ 42 → 33, app đã thấy 20 → 14). Đây đúng là lỗi mà việc che tên nhắm tới. Nhưng cả hai khoảng tin cậy đều chứa 0, nên **chưa phân biệt được với nhiễu**.
3. **ask vẫn tụt, giống A5:** app lạ 35,56 → 33,33 (3 tốt, 4 tệ), app đã thấy 62,92 → 48,33 (1 tốt, 3 tệ); số lần dừng bằng ask trên app lạ 27 → 19 (nhãn có 21). 100 mẫu oos không bù được, vì chúng dạy dừng ngay chứ không dạy hỏi lại. oos trên app đã thấy không đổi (92,86).
4. Theo quy tắc đã đặt: **không train bản nộp; giữ bài nộp hiện tại** (`q4b_lora` đã gộp, public 48,74). Quyết định cuối thuộc Phương (mục 6).
5. Mốc `q4b_holdout` chấm lại ra đúng 41,93 và 66,03 trước khi so; dữ liệu dựng trên Kaggle trùng từng byte với bản dựng ở máy local (sha256 `sft_train_mask.jsonl` `98c6666a…`), nên phép so hợp lệ.

## 2. Cấu hình
Dữ liệu: `build_sft.py --asr-tag phowl --hold-apps flights,bike_rental,package_tracking,shopping_list,podcast_radio,loyalty --aug-oos 100 --aug-mask 1180`: 1180 mẫu train, 174 eval, 132 dev; 100 mẫu oos; 1180 mẫu che tên (579 dùng chữ ASR `phowl`). Preflight: 4 file OK, bước dài nhất 4621 token, 0 bước quá `--max-len 5120`, **11437 bước mỗi epoch** (A4 7558, A5 10549).

Train: `torchrun --nproc_per_node=4 train_sft.py --data sft_train sft_train_asr sft_train_aug sft_train_mask --epochs 3 --max-len 5120 --global-batch 16 --save-limit 2 --max-hours 9.5`, tức **4 GPU × bs 1 × accum 4 = batch tổng 16**, 2145 bước tối ưu, **đủ 3 epoch** (không chạm giới hạn giờ). Train **22290 s (6 giờ 11 phút)**, cả cell 23100 s; ước trước của tôi là 4 đến 6 giờ. Loss 1,063 → 0,0020 (`train_loss` trung bình 0,0482; A5 0,0507, A4 0,0612). Adapter sha256 `21f38a19bc6cfeb4a5f718b2106e1f4ec44bbe1d995eb8c94b7d9a734f73b067`.

Đo: `infer_multi_gpu.sh` (4 tiến trình, mỗi GPU một phần) trên `holdout/eval_ids.txt` và `holdout/dev_ids.txt`, chữ sạch, luật bật, **tên thật** (suy luận không che gì), như A4, A4b và A5. Môi trường: `torch 2.11.0+cu130`, `transformers 5.20.2`, `peft 0.21.2`, `accelerate 1.15.0`, 4 × L4 24 GB.

## 3. Điểm
| | `q4b_holdout` (mốc) | `q4b_holdout_mask` | Chênh từng cặp |
|---|---|---|---|
| **App lạ** tổng (174) | 41,93 | **44,49** | +2,56 [−3,91; +9,01] |
| exec (153) | 42,80 | 46,02 | +3,21 (28 tốt, 19 tệ) |
| ask (21) | 35,56 | 33,33 | −2,22 (3 tốt, 4 tệ) |
| **App đã thấy** tổng (132) | 66,03 | **67,67** | +1,64 [−4,84; +8,01] |
| exec (102) | 62,84 | 67,25 | +4,40 (16 tốt, 11 tệ) |
| ask (16) | 62,92 | 48,33 | **−14,58** (1 tốt, 3 tệ) |
| oos (14) | 92,86 | 92,86 | 0,00 |

So với A5 (cùng bộ chia, cùng mốc): app lạ A5 −2,58, A8 **+2,56**; app đã thấy A5 −2,10, A8 **+1,64**; exec app lạ A5 −0,64, A8 **+3,21**.

Theo số call trên app lạ (A8): 1 call 61,81 (48 mẫu); 2 call 43,02 (74); 3 call 33,78 (32); 4 call 25,00 (13); 5 call 26,35 (7). Có `history` / không: 47,23 / 37,49.

## 4. Kiểm tra cơ chế: gọi tool ngoài tập tool của nhãn (`docs/results/a8/goi_tool_ngoai_nhan.txt`)
Đếm trên mẫu exec: mẫu có ít nhất một call dùng tool không có trong nhãn.
| | Mốc | A8 |
|---|---|---|
| App lạ (153 exec) | 42 (27%); trong 74 mẫu 0 điểm có 34 | **33 (22%)**; trong 68 mẫu 0 điểm có 26 |
| App đã thấy (102 exec) | 20 (20%); trong 32 mẫu 0 điểm có 14 | **14 (14%)**; trong 28 mẫu 0 điểm có 11 |

Chiều hướng khớp giả thuyết (model đọc mô tả thay vì nhớ tên nên ít chọn nhầm tool), nhưng mức giảm là 9 và 6 mẫu, chưa đủ để kết luận. Cách đếm này khác cách đếm "48 đến 52% mẫu 0 điểm" trong `docs/BAN_GIAO_A4.md` (ở đây 34/74 = 46% với mốc), nên không so thẳng hai con số.

## 5. Điều chưa kết luận được
1. Một lần train, một seed, một cách chia app. +2,56 và +1,64 đều nằm trong nhiễu; chỉ có thể nói A8 **không đạt ngưỡng +5**, và **chiều hướng tốt hơn A5**.
2. Gộp hai thay đổi (che tên và 100 oos) nên không tách được phần của từng cái. Phần exec tăng có lẽ chủ yếu do che tên (100 mẫu oos nhắm vào việc dừng, không vào chọn tool), nhưng chưa tách được; phần ask tụt chưa rõ nguyên nhân (cả A5 và A8 đều thêm mẫu không có ask).
3. Mốc `q4b_holdout` train trên 1 GPU RTX PRO 6000, A8 trên 4 GPU L4 (DDP): cùng batch tổng 16 và cùng số bước, nhưng khác GPU và thứ tự chia dữ liệu. Đây là một nguồn sai khác nhỏ, chưa đo riêng.
4. Đo trên chữ sạch (như A4, A4b, A5); chưa đo qua ASR.
5. **Thời gian trên L4** (không phải câu hỏi của A8 nhưng cần biết): Qwen3-4B kèm adapter chưa gộp, chữ sạch, mỗi mẫu trung bình khoảng 7 s, p95 9 đến 17 s, **chậm nhất 25,67 s** (gần giới hạn 30 s). A7a trên RTX PRO 6000 với model đã gộp và ASR sống: chậm nhất 5,67 s. GPU máy chấm chưa rõ; nên có số đo trên GPU giống máy chấm (A7b) trước khi nộp.

## 6. Đề xuất (Phương quyết định)
1. **Giữ bài nộp hiện tại** (`q4b_lora` đã gộp, public 48,74), đúng quy tắc đặt trước. Lý do thêm: train bản nộp toàn bộ trên L4 x4 ước 7 đến 8 giờ, cộng gộp, đo dev, dựng image và Test Solution, rất sát hạn 23:59 ngày 15/11; lợi ích nếu có chỉ khoảng +1 đến +2 điểm public và chưa chắc chắn; ask đang tụt.
2. Nếu Phương vẫn muốn dùng A8 (tức bỏ qua tiêu chí): train toàn bộ theo `docs/Y_TUONG_CHE_TEN.md` mục 7, và chỉ nộp khi dev qua ASR **không thấp hơn 56,98** và Test Solution chạy qua.
3. Hướng tiếp (sau cuộc thi, hoặc cho vòng sau): giữ che tên nhưng thêm mẫu ask (không chỉ oos), vì cả A5 và A8 đều cho thấy thêm mẫu "gọi tool" làm model ít hỏi lại.

## 7. File
| File | Nội dung | Trong git? |
|---|---|---|
| `docs/results/a8/phan_quyet_va_so_sanh.txt` | Phán quyết theo tiêu chí và hai kết quả `compare_runs` | có |
| `docs/results/a8/diem_app_la.txt`, `diem_app_da_thay.txt` | Điểm chi tiết (theo số call, `history`, loại mẫu) | có |
| `docs/results/a8/kiem_tra_moc.txt` | Chấm lại hai mốc (41,93 và 66,03) | có |
| `docs/results/a8/goi_tool_ngoai_nhan.txt` | Kiểm tra cơ chế (mục 4) | có |
| `docs/results/a8/dung_du_lieu.txt`, `preflight.txt`, `thoi_gian.txt`, `adapter_sha256.txt`, `loss_q4b_holdout_mask.txt`, `torch_cuda.txt`, `a8_results.zip.sha256` | Dựng dữ liệu, preflight, thời gian train và đo, sha256, đường cong loss, môi trường | có |
| `a8_results.zip` (67 KB) | Gồm `unseen_q4b_holdout_mask.jsonl`, `seen_q4b_holdout_mask.jsonl` (dự đoán trên dữ liệu cuộc thi), log đầy đủ | **không**, gửi qua Drive riêng |

sha256 zip: `c45f1c44…` (đầy đủ trong `docs/results/a8/a8_results.zip.sha256`).

## 8. Dòng nhật ký đề xuất cho `docs/EXPERIMENTS.md`
| Ngày | Commit | Việc | Điểm | Ghi chú |
|---|---|---|---|---|
| 2026-11-13 | `d5f376c` | **A8** `q4b_holdout_mask`: bộ chia A4 (1180 mẫu + bản ASR `phowl`) + 100 mẫu oos + 1180 bản sao **che tên app và tool** (`build_sft.py --aug-mask`, theo Hammer, arXiv:2410.04587), 3 epoch, 4 × L4, chữ sạch | app lạ **44,49** (mốc 41,93), app đã thấy **67,67** (mốc 66,03) | **Không đạt** (cần app lạ +5): +2,56 [−3,91; +9,01] (31/23/120) và +1,64 [−4,84; +8,01] (17/14/101). Khác A5: **exec tăng** (+3,21 và +4,40) và mẫu exec gọi tool ngoài nhãn giảm (42→33 và 20→14), đúng chiều giả thuyết nhưng trong nhiễu. ask vẫn tụt (−2,22 và −14,58). Giữ bài nộp hiện tại. Train 6 giờ 11 phút trên 4 × L4. Adapter sha256 `21f38a19…`. Chi tiết: `docs/BAN_GIAO_A8.md` |
