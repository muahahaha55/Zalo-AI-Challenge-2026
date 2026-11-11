# Bàn giao A5 train lần 1: `q4b_holdout_a5` (bộ chia A4 + 907 mẫu A5)

Ngày 2026-11-11. Người chạy: thuan (Kaggle, 1 GPU RTX PRO 6000 Blackwell), làm đúng "Bước 1" của `docs/HUONG_DAN_GPU_08-11.md` và `docs/TASKS.md` mục "A5, bước tiếp theo (10/11)". Không sửa mã của Phương. Mã là commit `b2853d2`. Số liệu tổng hợp nằm ở `docs/results/a5t/`. Dự đoán (`unseen_*`, `seen_*`) là dữ liệu cuộc thi nên **không** vào git, gửi qua Drive riêng (mục 7).

## 1. Tóm tắt
1. **Lần train 1 KHÔNG ĐẠT tiêu chí đặt trước.** Điểm app lạ (174 mẫu): **41,93 → 39,35, chênh từng cặp −2,58** (khoảng tin cậy 95% [−9,09; +4,05], 25 tốt lên, 29 tệ đi, 120 bằng). Cần ≥ +5: **không đạt**, và cận trên của khoảng tin cậy (+4,05) cũng dưới +5.
2. Điểm app đã thấy (132 mẫu): 66,03 → 63,93, chênh −2,10 [−7,67; +3,20] (12 tốt, 13 tệ, 107 bằng). Cần không giảm quá 3 điểm: đạt.
3. Theo quy tắc đã đặt: **không làm lần train 2** (`q4b_a5` trên toàn bộ train); giữ `q4b_lora` (nay là bản đã gộp, xem `docs/BAN_GIAO_MERGE.md`).
4. **Toàn bộ phần âm nằm ở loại `ask` và `oos`; `exec` gần như không đổi** (mục 4). A5 không giúp exec trên app lạ, loại mẫu mà dữ liệu này nhắm tới.
5. Mốc `q4b_holdout` chấm lại ra đúng 41,93 và 66,03 trước khi so, nên phép so hợp lệ.

## 2. Cấu hình
Dữ liệu dựng bằng `build_sft.py --asr-tag phowl --hold-apps flights,bike_rental,package_tracking,shopping_list,podcast_radio,loyalty --hist-max 300 --callgen data/processed/callgen_checked_all.jsonl`: **1180 mẫu train, 174 eval, 132 dev, và `callgen -> sft_train_cg.jsonl: 907 samples; {'not_train': 1791, 'roundtrip_failed': 1302, 'ok': 907}`** (đúng dòng Phương yêu cầu gửi). Train đúng công thức A4 (`q4b_holdout`: `sft_train` + `sft_train_asr`, 3 epoch, `--max-len 5120 --global-batch 16 --save-limit 2`) cộng `holdout/sft_train_cg.jsonl`, **không** thêm `sft_train_hist`. Preflight: 1180, 1180 và 907 hàng, 0 hàng quá `--max-len`, 10549 bước mỗi epoch (A4 là 7558). Đo bằng `infer_multi_gpu.sh ... --text clean` trên `holdout/eval_ids.txt` và `holdout/dev_ids.txt`, luật bật, như A4 và A4b. Môi trường như A4. `callgen_checked_all.jsonl` là output của notebook `a5-gen` (sha256 `ba436701…`).

Train: 10351 s (2 giờ 52 phút; ước ban đầu của tôi là 3 đến 3,5 giờ), loss cuối khoảng 0,0026 (`train_loss` trung bình 0,0507, A4 là 0,0612). Adapter sha256: `fd9612213bd51bdda9fcd3d1d1d21dd8bf84ca79c73bc72b52e6c97da4ef6d34`.

## 3. Điểm
| | `q4b_holdout` (mốc) | `q4b_holdout_a5` | Chênh từng cặp |
|---|---|---|---|
| **App lạ** tổng (174) | 41,93 | **39,35** | −2,58 [−9,09; +4,05] |
| exec (153) | 42,80 | 42,16 | −0,64 (22 tốt, 22 tệ) |
| ask (21) | 35,56 | 18,89 | **−16,67** (3 tốt, 7 tệ) |
| **App đã thấy** tổng (132) | 66,03 | **63,93** | −2,10 [−7,67; +3,20] |
| exec (102) | 62,84 | 62,80 | −0,04 (12 tốt, 9 tệ) |
| ask (16) | 62,92 | 52,08 | **−10,83** (0 tốt, 3 tệ) |
| oos (14) | 92,86 | 85,71 | **−7,14** (0 tốt, 1 tệ) |

Theo số call trên app lạ (mốc → mới): 1 call 53,47 → 52,08; 2 call 41,71 → 37,93; 3 call 30,51 → 32,14; 4 call 28,85 → 13,46 (13 mẫu); 5 call 41,59 → 48,17 (7 mẫu). Có `history` / không: 42,75 / 39,84 → 41,77 / 33,19. Các nhóm 4 và 5 call chỉ 13 và 7 mẫu, nên thuộc nhiễu.

Thời gian mỗi mẫu: app lạ mean 1,62 s, p95 3,52 s, max 4,73 s; app đã thấy mean 1,55 s, p95 3,07 s, max 4,10 s. Lý do dừng app lạ: `end` 158, `ask` 14, `max_steps` 1, `repeated_read` 1.

## 4. Vì sao điểm tổng giảm: do ask và oos, không do exec
- **exec** (loại mẫu A5 nhắm tới) không đổi: −0,64 trên app lạ (22 mẫu tốt lên và 22 tệ đi), −0,04 trên app đã thấy.
- **ask** tụt ở cả hai tập (−16,67 và −10,83) và **oos** đã thấy −7,14. Đây là các loại mẫu mà A5 không có.
- **Số lần dừng bằng `ask` giảm**: app lạ 27 (`q4b_holdout`, `docs/BAN_GIAO_A4.md`) → 14 (nhãn có 21 mẫu ask); app đã thấy 19 (`docs/BAN_GIAO_A4b.md`) → 16 (nhãn 16). Model hỏi lại ít hơn.
- **Giả thuyết (chưa kiểm chứng):** theo `callgen.py`, mẫu A5 không bao giờ chứa `ask` hay mẫu ngoài phạm vi (chuỗi nào có `sys.*` thì bị cắt, mẫu không có call bị bỏ), nên 907 mẫu thêm đều là "gọi tool" và có thể kéo model về phía luôn gọi tool. Muốn xác nhận cần thí nghiệm riêng (ví dụ trộn thêm mẫu ask và oos vào dữ liệu A5); không làm khi chưa có quyết định.
- Số mẫu ask chỉ 21 và 16, oos 14, nên từng mức giảm riêng lẻ chưa chắc chắn (khoảng tin cậy tổng thể chứa 0 ở cả hai tập).

## 5. Điều chưa kết luận được
1. Một lần train, một seed, một cách chia app (như A4). Chênh −2,58 và −2,10 đều nằm trong nhiễu; chỉ có thể nói A5 **không cho thấy cải thiện** ở ngưỡng +5, chứ không chứng minh A5 gây hại.
2. Giả thuyết ở mục 4 chưa được thử.
3. Chỉ dùng 907 mẫu (bộ chia A4 loại 1791 mẫu); với train toàn bộ (1550 mẫu) kết quả có thể khác, nhưng lần 2 không được phép theo tiêu chí đã đặt.
4. Đo trên chữ sạch (như A4 và A4b); chưa đo với ASR thật.

## 6. Đề xuất (Phương quyết định)
1. **Không train lần 2.** Giữ `q4b_lora` (bản đã gộp, dev 66,82 và 56,98) làm bài nộp.
2. Nếu còn giờ GPU và Phương muốn thử tiếp: sinh dữ liệu A5 có thêm mẫu ask và oos (hoặc trộn A5 với mẫu ask và oos thật của train), train lại cùng công thức rồi đo trên cùng hai tập, so với 41,93 và 66,03. Chỉ làm nếu kịp trước mốc chốt cấu hình 13/11.
3. Chạy A7a với `q4b_merged` (notebook `a7a-verify-real`) để chốt đường đi tới bài nộp.

## 7. File
| File | Nội dung | Trong git? |
|---|---|---|
| `docs/results/a5t/phan_quyet_va_so_sanh.txt` | Phán quyết theo tiêu chí và hai kết quả `compare_runs` | có |
| `docs/results/a5t/diem_app_la.txt`, `diem_app_da_thay.txt` | Điểm chi tiết (theo số call, `history`, loại mẫu) | có |
| `docs/results/a5t/kiem_tra_moc.txt` | Chấm lại hai mốc (41,93 và 66,03) | có |
| `docs/results/a5t/dung_du_lieu.txt`, `preflight.txt`, `thoi_gian.txt`, `adapter_sha256.txt`, `loss_q4b_holdout_a5.txt`, `torch_cuda.txt`, `*.zip.sha256` | Dựng dữ liệu, preflight, thời gian, sha256, đường cong loss, môi trường | có |
| `a5_train1_results.zip` (59 KB) | Gồm `unseen_q4b_holdout_a5.jsonl`, `seen_q4b_holdout_a5.jsonl` (dự đoán trên dữ liệu cuộc thi) và log | **không**, gửi qua Drive riêng |

sha256 zip: `fa214fad…` (đầy đủ trong `docs/results/a5t/a5_train1_results.zip.sha256`).

## 8. Dòng nhật ký đề xuất cho `docs/EXPERIMENTS.md`
| Ngày | Commit | Việc | Điểm | Ghi chú |
|---|---|---|---|---|
| 2026-11-11 | `b2853d2` | **A5 train lần 1** `q4b_holdout_a5`: bộ chia A4 (1180 mẫu + bản ASR `phowl`) + 907 mẫu A5, không `history`, 3 epoch, chữ sạch | app lạ **39,35** (mốc 41,93), app đã thấy **63,93** (mốc 66,03) | **Không đạt** (cần app lạ +5): chênh −2,58 [−9,09; +4,05] (25/29/120) và −2,10 [−7,67; +3,20] (12/13/107). exec không đổi (−0,64 và −0,04); ask tụt (−16,67 và −10,83), oos đã thấy −7,14; số lần dừng bằng ask giảm (27→14 và 19→16). Giả thuyết chưa kiểm chứng: A5 không có mẫu ask/oos. Không làm lần 2. 2 giờ 52 phút. Adapter sha256 `fd961221…`. Chi tiết: `docs/BAN_GIAO_A5_TRAIN1.md` |
