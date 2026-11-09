# Bàn giao kết quả A4b (tham chiếu "app đã thấy")

Ngày 2026-11-09. Người chạy: thuan (Kaggle, 1 GPU RTX PRO 6000 Blackwell), theo `docs/TASKS.md` mục A4b. Không sửa mã của Phương. Mã dùng là commit `07d4cf8`. Số liệu thô nằm ở `docs/results/a4b/`; file dự đoán (`seen_*.jsonl`) **không** vào git, gửi qua Drive riêng (xem mục 8).

## 1. Tóm tắt
1. Chạy chính `q4b_holdout` và `q4b_holdout_hist` (adapter đã train ở A4, sha256 trùng giá trị ghi ở A4) trên 132 mẫu dev dùng app **đã thấy**, chữ sạch, luật bật: **66,03** và **62,52**. Cùng hai model trên 174 mẫu app **lạ** (A4): 41,93 và 43,37.
2. **Khoảng cách đã thấy trừ lạ của `q4b_holdout`: +24,10 (cách TASKS tính), +20,92 nếu chỉ so exec và ask (cùng loại mẫu).** Khoảng tin cậy 95% của số sau: [+10,56; +31,41]. **Trên ngưỡng 15 của Phương** theo cả hai cách tính.
3. Kết quả giữ nguyên sau khi cân bằng cơ cấu mẫu (có lịch sử hay không, số call): khoảng cách `q4b_holdout` từ +19,7 đến +21,4.
4. Với `q4b_holdout_hist` khoảng cách nhỏ hơn: +19,15 (TASKS), +16,40 (exec+ask, khoảng tin cậy [+5,58; +27,15]), sau chuẩn hóa +15,8 đến +16,4: **sát ngưỡng 15**.
5. Theo quy tắc đã đặt ("trên khoảng 15 điểm thì làm A5"), A4b **ủng hộ làm A5**. Mức khoảng cách (khoảng 20) là ước lượng điểm; khoảng tin cậy rộng (cận dưới 10,6) nên "trên 15" là khả dĩ cao chứ chưa chắc chắn.
6. Dữ liệu `history` (300 mẫu) trên app đã thấy: **−3,51** (khoảng tin cậy [−9,43; +2,37], không phân biệt được với 0), cộng với +1,44 ở A4 (cũng không phân biệt được với 0): **không có bằng chứng `history` giúp**, và tín hiệu oos tăng ở A1 không lặp lại ở đây (mục 5).

## 2. Cấu hình
Không huấn luyện gì mới. `infer_multi_gpu.sh train ... --ids data/processed/holdout/dev_ids.txt --backend hf --model <Qwen3-4B> --adapter checkpoints/<q4b_holdout hoặc q4b_holdout_hist> --text clean`, rồi `eval.py` trên cùng tệp id (đúng lệnh TASKS A4b). `holdout/dev_ids.txt` có 132 mẫu: **102 exec, 16 ask, 14 oos**, không trùng với `eval_ids.txt` (app lạ). Adapter sha256: `q4b_holdout` = `3fb678f3…`, `q4b_holdout_hist` = `d425d17c…` (đối chiếu trong notebook trước khi chạy). Môi trường như A4: `torch 2.11.0+cu130`, CUDA 13.0, Python 3.13.16, `peft 0.21.2`, `transformers 5.20.2`.

## 3. Điểm trên app đã thấy (132 mẫu)
| | `q4b_holdout` | `q4b_holdout_hist` |
|---|---|---|
| **Tổng** | **66,03** | **62,52** |
| exec (102) | 62,84 | 61,24 |
| ask (16) | 62,92 | 50,42 |
| oos (14) | 92,86 | 85,71 |
| Có lịch sử (70) / không (62) | 59,51 / 73,40 | 60,18 / 65,18 |
| Thời gian mỗi mẫu | mean 1,63 s, p95 3,47 s, max 6,21 s | mean 1,58 s, p95 3,42 s, max 4,10 s |
| Lý do dừng | `end` 113, `ask` 19 | `end` 116, `ask` 15, `repeated_read` 1 |

Tham chiếu: `q4b_lora` (train đủ 1390 mẫu) được 66,82 trên dev chữ sạch, gần bằng 66,03 của `q4b_holdout` (train 1180 mẫu): bỏ 15% dữ liệu train gần như không làm giảm điểm trên app đã thấy.

## 4. Khoảng cách đã thấy trừ lạ (kết quả chính; bootstrap hai mẫu độc lập)
| Nhóm | Đã thấy | Lạ | **Khoảng cách** | Khoảng tin cậy 95% |
|---|---|---|---|---|
| **`q4b_holdout`** tổng (cách TASKS) | 66,03 | 41,93 | **+24,10** | [+13,87; +34,09] |
| chỉ exec + ask | 62,85 | 41,93 | **+20,92** | [+10,56; +31,41] |
| exec | 62,84 | 42,80 | +20,04 | [+8,56; +31,08] |
| ask (n = 16 và 21) | 62,92 | 35,56 | +27,36 | [−1,37; +56,27] |
| **`q4b_holdout_hist`** tổng | 62,52 | 43,37 | +19,15 | [+8,79; +29,31] |
| chỉ exec + ask | 59,77 | 43,37 | +16,40 | [+5,58; +27,15] |
| exec | 61,24 | 44,88 | +16,36 | [+4,83; +27,52] |
| ask | 50,42 | 32,38 | +18,04 | [−11,43; +47,32] |

**Vì sao tôi in cả hai cách tính:** tập đã thấy có 14 mẫu oos (92,86 điểm) mà tập app lạ không có, nên "tổng" thổi phồng khoảng cách khoảng 3 điểm. Hai tập cũng khác cơ cấu (53% mẫu có lịch sử ở tập đã thấy, 72% ở tập app lạ). Chuẩn hóa theo cơ cấu tập app lạ (chỉ exec+ask):

| Chuẩn hóa theo | `q4b_holdout` | `q4b_holdout_hist` |
|---|---|---|
| có/không lịch sử | +19,66 | +16,35 |
| số call gold (1, 2, 3, 4+) | +21,41 | +15,76 |
| lịch sử × số call | +21,11 | +16,32 |

Khoảng cách không đến từ khác biệt cơ cấu. Nhóm ask chỉ có 16 và 21 mẫu nên khoảng tin cậy rất rộng (chứa 0).

## 5. Dữ liệu `history` trên app đã thấy (so từng cặp, `q4b_holdout_hist` trừ `q4b_holdout`)
Chênh **−3,51**, khoảng tin cậy 95% [−9,43; +2,37], 14 tốt lên, 16 tệ đi, 102 bằng. Theo loại: exec −1,60 (13 và 12), ask −12,50 (1 tốt lên, 3 tệ đi), oos −7,14 (0 tốt lên, 1 tệ đi). Từ chối nhầm: 0/118 (`q4b_holdout`) và 2/118 (`q4b_holdout_hist`); số dự đoán `ask`: 19 và 15 (nhãn: 16); số call trung bình 2,17 và 2,08.
So với các lần trước: A1 (700 mẫu `history`) cho oos 4/12 lên 7/12 (3 tốt lên, 0 tệ đi); ở đây (300 mẫu) oos không tăng. Tóm lại, qua A1, A4 và A4b, `history` không có tác dụng ổn định trên oos, exec hay ask.

## 6. Điều chưa kết luận được
1. "Trên 15" dựa trên ước lượng điểm khoảng 20 với khoảng tin cậy [10,6; 31,4]; tập đã thấy (132 mẫu) và tập app lạ (174 mẫu) là hai tập khác nhau, không phải so cặp trên cùng mẫu.
2. Chỉ một cách chia app (giấu 6 app cụ thể); `loyalty` kém nhất (A4: 22,03 và 18,05). Khoảng cách có thể phụ thuộc vào app nào bị giấu.
3. Khoảng cách này đo trên chữ sạch. Với ASR thật khoảng cách có thể khác; chưa đo.
4. Public có 65% mẫu chứa app lạ, nhưng mức "lạ" của app public (11 app mới, 34 tool mới) có thể khác mức "lạ" của 6 app bị giấu.

## 7. Đề xuất (Phương quyết định)
1. **Làm A5** theo quy tắc đã đặt, bắt đầu bằng bước thử 200 mẫu (`gen_calls.py --n 200` và `gen_text.py`) với model sinh câu theo thứ tự 8B, rồi 14B, rồi 32B; gửi 30 dòng để duyệt chất lượng trước khi sinh nhiều hoặc train. Điều kiện hiện thiếu: cần model Qwen3-8B (hoặc lớn hơn) làm Kaggle Model hoặc dataset Private.
2. Khi đo A5, so trên đúng bộ chia A4 (`holdout/eval_ids.txt`) và dùng cả hai tham chiếu của A4b: mục tiêu là thu hẹp khoảng cách +20 (tức đưa điểm app lạ 41,93 lên gần 66).
3. Không đưa `history` vào cấu hình nộp: chưa có bằng chứng giúp, và có dấu hiệu nhẹ làm tệ ask (−12,50 trên 16 mẫu, chưa chắc).
4. Giữ `q4b_lora` làm cấu hình nộp cho đến khi có cải thiện đo được.

## 8. File
| File | Nội dung | Trong git? |
|---|---|---|
| `docs/results/a4b/diem_seen_*.txt`, `bao_cao_khoang_cach.txt`, `chuan_hoa_theo_co_cau.txt`, `thoi_gian.txt`, `torch_cuda.txt`, `a4b_results.zip.sha256` | Điểm, khoảng cách, so cặp, chuẩn hóa, thời gian, môi trường | có |
| `a4b_results.zip` (27 KB) | Gồm cả `seen_*.jsonl` (dự đoán trên dữ liệu cuộc thi) | **không**, gửi qua Drive riêng |

## 9. Dòng nhật ký đề xuất cho `docs/EXPERIMENTS.md`
| Ngày | Commit | Việc | Điểm | Ghi chú |
|---|---|---|---|---|
| 2026-11-09 | `07d4cf8` | A4b: `q4b_holdout` và `q4b_holdout_hist` trên 132 mẫu dev app đã thấy (102 exec, 16 ask, 14 oos), chữ sạch, luật bật | 66,03 và 62,52 | khoảng cách đã thấy − lạ của `q4b_holdout`: +24,10 (TASKS), +20,92 chỉ exec+ask [+10,56; +31,41], chuẩn hóa cơ cấu +19,7 đến +21,4: trên ngưỡng 15 (làm A5); `q4b_holdout_hist` +16,4 (sát ngưỡng); `history` trên app đã thấy −3,51 [−9,43; +2,37] |
