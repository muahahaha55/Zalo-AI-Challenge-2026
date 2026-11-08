# Bàn giao kết quả A4 ("app lạ")

Ngày 2026-11-08. Người chạy: thuan (Kaggle, 1 GPU RTX PRO 6000 Blackwell), theo `docs/TASKS.md` mục A4. Không sửa mã của Phương. Mã dùng là commit `3fd1c60`. Số liệu thô nằm ở `docs/results/a4/`; file dự đoán chi tiết (`unseen_*.jsonl`) **không** đưa vào git, gửi qua Drive riêng (xem mục 8).

## 1. Tóm tắt
1. Giấu 6 app (`flights, bike_rental, package_tracking, shopping_list, podcast_radio, loyalty`) khỏi train rồi đo trên 174 mẫu dùng chúng (chữ sạch): **`q4b_holdout` đạt 41,93; `q4b_holdout_hist` đạt 43,37**.
2. Thêm 300 mẫu `history`: chênh **+1,44** (khoảng tin cậy 95% [−4,69; +7,60], 28 tốt lên, 22 tệ đi, 124 bằng): **không phân biệt được với 0**, giống kết luận ở A1.
3. **Lỗi chính trên app lạ là chọn sai tool**: 51,7% (`q4b_holdout`) và 48,2% (`q4b_holdout_hist`) số mẫu 0 điểm gọi một tool ngoài tập tool của nhãn. Sai tham số là 33–34%; thiếu hoặc thừa call khoảng 14–16%; từ chối gần như không có (1/174).
4. Điểm thô so với dev trong phân phối: `q4b_lora` với chữ sạch đạt 66,82 (exec 65,73, ask 75,64) nên app lạ có thể làm mất khoảng 24 điểm (exec −23, ask −40). **Đây chưa phải phép so chính thức** (mục 7).
5. Nếu cần quyết định A5 (ghép đoạn call), A4 ủng hộ hướng này về mặt mức độ ảnh hưởng và loại lỗi (chọn sai tool), nhưng xem mục 7 về điểm chưa được đo.

## 2. Cấu hình hai lần chạy
| | `q4b_holdout` (mốc) | `q4b_holdout_hist` |
|---|---|---|
| Dữ liệu train | `holdout/sft_train` + `holdout/sft_train_asr` (ASR `phowl`): 2 × 1180 hàng | cộng `holdout/sft_train_hist` 300 hàng |
| Bước huấn luyện mỗi epoch | 7.558 | 8.226 |
| Lệnh | đúng lệnh TASKS A4: `train_sft.py ... --epochs 3 --max-len 5120 --global-batch 16 --save-limit 2` | cùng lệnh, thêm file hist vào `--data` |
| Số bước tối ưu / thời gian train | 1.419 / 7.758 s (2 giờ 9 phút) | 1.545 / 8.422 s (2 giờ 20 phút) |
| Checkpoint còn lại | 946, 1419 (lưu mỗi epoch) | 1030, 1545 |
| Loss cuối | khoảng 0,0025–0,0039 | khoảng 0,0051–0,0124 (xem `loss_*.txt`) |
| sha256 adapter | `3fb678f340662f002a6bb4d42c28f91848a6d631839c1d9ff5b900bf33199a0a` | `d425d17c31ce87e87515b2356ba6402c1943de4919b20cd89b57b2569c8b0910` |

Dữ liệu dựng bằng `build_sft.py --asr-tag phowl --hold-apps <6 app> --hist-max 300`: 1180 mẫu train, 174 mẫu đánh giá, 0 hàng quá `--max-len`, đúng kỳ vọng của TASKS. Đánh giá bằng `infer_multi_gpu.sh ... --text clean` trên `holdout/eval_ids.txt` rồi `eval.py`; luật `rules.py` bật mặc định.

## 3. Điểm trên 174 mẫu app lạ (chữ sạch)
| | `q4b_holdout` | `q4b_holdout_hist` |
|---|---|---|
| **Tổng** | **41,93** | **43,37** |
| exec (153 mẫu) | 42,80 | 44,88 |
| ask (21 mẫu) | 35,56 | 32,38 |
| oos | không có (eval toàn mẫu dùng app bị giấu) | không có |
| 1 call gold (48) | 53,47 | 61,11 |
| 2 call (74) | 41,71 | 39,77 |
| 3 call (32) | 30,51 | 34,38 |
| 4 call (13) | 28,85 | 26,92 |
| 5 call (7) | 41,59 | 31,43 |
| Có `history` (125) / không (49) | 42,75 / 39,84 | 45,79 / 37,21 |
| Cơ cấu điểm: 0 / một phần / tối đa | 50,0% / 19,0% / 31,0% | 48,9% / 17,8% / 33,3% |

Đã chấm lại độc lập bằng `scripts/eval.py` và `compare_runs.py` ở máy local: cùng 41,93 và 43,37.

## 4. So từng cặp mẫu (`scripts/compare_runs.py`, `q4b_holdout_hist` trừ `q4b_holdout`)
Chênh **+1,44**, khoảng tin cậy 95% **[−4,69; +7,60]**, 28 tốt lên, 22 tệ đi, 124 bằng. Theo loại: exec +2,08 (22 tốt lên, 18 tệ đi), ask −3,17 (6 và 4). Số mẫu ask chỉ 21 nên mức −3,17 thuộc nhiễu.

## 5. Theo từng app bị giấu (một mẫu có thể dùng nhiều app; 2/174 mẫu dùng từ 2 app)
| App | n | `q4b_holdout` | `q4b_holdout_hist` | Chênh |
|---|---|---|---|---|
| flights | 33 | 41,41 | 41,11 | −0,30 |
| bike_rental | 30 | 46,17 | 47,94 | +1,77 |
| loyalty | 29 | **22,03** | **18,05** | −3,98 |
| podcast_radio | 28 | 55,95 | 58,33 | +2,38 |
| package_tracking | 25 | 34,00 | 40,67 | +6,67 |
| shopping_list | 23 | 51,67 | 58,19 | +6,52 |

`loyalty` kém nhất ở cả hai cấu hình (khoảng 20 điểm). Chênh theo từng app dựa trên 23–33 mẫu nên đều thuộc nhiễu; không kết luận `history` giúp app nào.

## 6. Phân loại lỗi thô (mẫu 0 điểm; chỉ so tên tool với nhãn)
| Loại | `q4b_holdout` (87 mẫu) | `q4b_holdout_hist` (85 mẫu) |
|---|---|---|
| Gọi tool SAI (có tool ngoài nhãn) | 45 (51,7%) | 41 (48,2%) |
| Đúng bộ tool, sai tham số hoặc hash | 29 (33,3%) | 29 (34,1%) |
| Đúng loại tool nhưng thiếu call | 9 (10,3%) | 10 (11,8%) |
| Đúng loại tool nhưng thừa call | 3 (3,4%) | 4 (4,7%) |
| Không gọi gì (từ chối) | 1 (1,1%) | 1 (1,2%) |

**Từ chối nhầm:** 1/174 ở cả hai (rủi ro mà dữ liệu `history` có thể gây, vì 20% mẫu `history` là lượt không gọi tool: không xảy ra trên app lạ). Số mẫu dự đoán `ask`: 27 (`q4b_holdout`) và 16 (`q4b_holdout_hist`) so với 21 nhãn; số call trung bình 2,24 và 2,28. Thời gian mỗi mẫu của `q4b_holdout`: mean 1,72 s, p95 3,48 s, max 4,96 s; lý do dừng: `end` 146, `ask` 27, `repeated_read` 1 (log thời gian của `q4b_holdout_hist` chưa được lưu vào zip).

## 7. Điều chưa kết luận được
1. **Không có tham chiếu "app đã thấy" cùng mô hình.** Con số "mất khoảng 24 điểm" lấy từ `q4b_lora` trên dev (66,82), nhưng khác tập mẫu (155 so với 174), khác dữ liệu train (1390 so với 1180 mẫu), khác loại ASR trong dữ liệu train. Phép so sạch cần chạy chính `q4b_holdout` trên `holdout/dev_ids.txt` (132 mẫu dev dùng app đã thấy, không nằm trong train): khoảng 5 phút trên Kaggle, adapter còn trong output notebook A4. **Đề xuất làm trước khi lấy con số 24 điểm làm căn cứ.**
2. Bộ đánh giá chỉ có exec và ask, không có oos, nên A4 không đo được việc "từ chối đúng" trên app lạ.
3. Mọi chênh giữa hai cấu hình đều nằm trong nhiễu (174 mẫu; mỗi mẫu 0,57 điểm).
4. Loại "gọi tool SAI" gộp cả trường hợp gọi tool của app đã thấy thay cho tool của app lạ, và gọi tool cùng app nhưng sai tên; chưa tách.

## 8. Môi trường và file
Môi trường train (theo yêu cầu Bước 1 trong hướng dẫn GPU): `torch 2.11.0+cu130`, CUDA `13.0`, Python `3.13.16 (main, Oct 1 2026, 15:40:24) [GCC 13.3.0]`; `peft 0.21.2`, `transformers 5.20.2`; GPU RTX PRO 6000 Blackwell Server Edition. So với Docker hiện tại (`torch 2.6.0+cu124`, `python:3.12`), môi trường train lệch về torch và Python.

| File | Nội dung | Trong git? |
|---|---|---|
| `docs/results/a4/diem_*.txt`, `so_sanh_tung_cap.txt`, `phan_tich_theo_app.txt` | Điểm, so từng cặp, phân tích theo app và lỗi | có |
| `docs/results/a4/loss_*.txt`, `adapter_sha256.txt`, `torch_cuda.txt` | Đường cong loss, sha256 adapter, môi trường | có |
| `a4_results.zip` (99 KB, sha256 trong `a4_results.zip.sha256`) | Gồm cả `unseen_*.jsonl` (dự đoán trên dữ liệu cuộc thi) | **không**, gửi qua Drive riêng |

## 9. Đề xuất (Phương quyết định)
1. **A4b (khoảng 10 phút GPU):** chạy `q4b_holdout` trên `holdout/dev_ids.txt` để có tham chiếu "đã thấy" cùng mô hình, rồi mới đo khoảng cách thật giữa app đã thấy và app lạ.
2. **Dữ liệu `history`:** không có bằng chứng giúp app lạ (+1,44, nằm trong nhiễu), cũng không gây từ chối nhầm. Cùng kết luận với A1: chưa đủ để đổi cấu hình nộp. Giữ `q4b_lora`.
3. **A5 (ghép đoạn call):** loại lỗi chiếm nhiều nhất là chọn sai tool trên app lạ, phù hợp với mục tiêu của A5; điều kiện theo TASKS là "A4 cho thấy app lạ là vấn đề". Nên xác nhận bằng A4b trước. A5 cần một model từ 14B trở lên làm dataset Private trên Kaggle (hiện chưa có).
4. Hạn A4 (10/11) đã đáp ứng.

## 10. Dòng nhật ký đề xuất cho `docs/EXPERIMENTS.md`
| Ngày | Commit | Việc | Điểm | Ghi chú |
|---|---|---|---|---|
| 2026-11-08 | `3fd1c60` | A4: giấu 6 app, `q4b_holdout` và `q4b_holdout_hist` (+300 mẫu `history`), 3 epoch, chữ sạch, 174 mẫu app lạ | 41,93 và 43,37 | chênh +1,44 [−4,69; +7,60], không phân biệt được với 0; 48–52% mẫu 0 điểm gọi sai tool; từ chối nhầm 1/174; 2 giờ 9 phút và 2 giờ 20 phút; adapter sha256 `3fb678f3…` và `d425d17c…`; chưa có tham chiếu app đã thấy cùng mô hình |
