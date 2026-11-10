# Bàn giao A5: sinh số lượng dữ liệu (trước khi train)

Ngày 2026-11-10. Người chạy: thuan (Kaggle, 1 GPU RTX PRO 6000 Blackwell), làm đúng bước 1 đến 3 của `docs/TASKS.md` mục "A5, bước tiếp theo (10/11)". Không sửa mã của Phương. Mã dùng là commit `78c70a3` (dataset `lifemate-code` dựng lại từ commit này; cell kiểm tra chặn nếu mã cũ). **Chưa train gì**, đúng dặn "gửi lại trước khi train". Số liệu tổng hợp nằm ở `docs/results/a5g/`. Mẫu sinh ra (`callgen_checked_all.jsonl`) và 30 dòng để duyệt (`a5g_sample30_kept.txt`) là dữ liệu cuộc thi nên **không** vào git, gửi qua Drive riêng (mục 7).

## 1. Tóm tắt
1. Chạy đủ ba bước, không lỗi: sinh 4000 mẫu call, hai model viết câu (Qwen3-8B cho nửa đầu, Qwen3.5-9B cho nửa sau, `--no-verify`), khứ hồi bằng Qwen3-4B + `q4b_lora` (3 tiến trình song song, cả ba kết thúc bình thường).
2. **Được giữ (lọc rẻ và khứ hồi): 1550/4000 (38,8%).** Theo model viết câu: 8B 725/2011 (36,1%), 9B 825/1989 (41,5%).
3. **Số mẫu dùng được cho bộ chia A4: 907** (dòng xem trước của `build_sft.py`: `callgen -> sft_train_cg.jsonl: 907 samples; {'not_train': 1791, 'roundtrip_failed': 1302, 'ok': 907}`). Con số "tối đa 1509" trong bản tổng hợp chưa trừ 1791 mẫu có mẫu gốc hoặc donor nằm ngoài phần train của bộ chia A4; dùng **907**.
4. Tỷ lệ giữ **giảm theo số call** (56% với 1 call, 15 đến 21% với 5 đến 7 call), và mẫu **nhiều app (3 đến 4 app) ít hơn khoảng một nửa** so với train thật. Cả hai đáng cân nhắc trước khi train (mục 4).
5. **Chưa ai đọc nội dung 30 dòng** (tôi chỉ dùng số tổng hợp). Chờ Phương duyệt `a5g_sample30_kept.txt` (30 dòng của mẫu **được giữ**, kèm mục id trỏ tới) trước khi train.

## 2. Cấu hình
Hai model viết câu là bản gốc (sha256 khớp Hugging Face: `Qwen3-8B` 5/5 shard, `Qwen3.5-9B` 4/4 shard; nguồn Kaggle Models `manojkumarcs28/qwen3-8b` và `mirzamilanfarabi/qwen3.5`, biến thể `9b`), nạp bằng `AutoModelForCausalLM`. Người kiểm tra khứ hồi: Qwen3-4B + adapter `q4b_lora` từ output `Hello_done`. Môi trường: `torch 2.11.0+cu130`, CUDA 13.0, Python 3.13.16, `peft 0.21.2`, `transformers 5.20.2`.

Lệnh (đúng TASKS): `gen_calls.py --n 4000 --seed 42 --weights 1:0.35,2:0.4,3:0.2,4:0.05` (9583 lần thử: 4709 bị môi trường từ chối, 689 trùng, 185 không tìm được đoạn); chia đôi bằng `split -n l/2`; `gen_text.py --batch 32 --no-verify` (569 s cho 2011 mẫu với 8B, 523 s cho 1989 mẫu với 9B, gồm nạp model); khứ hồi 3 tiến trình, **không** dùng `--check-all` và `--control` (khác bước thử 200 mẫu; không có số CONTROL ở lần này).

## 3. Kết quả
| | Kết quả |
|---|---|
| Qua luật `text_ok` | 3081/4000 (77,0%) |
| Lý do `text_ok` loại | thiếu nội dung tham số 439, giờ mâu thuẫn nhãn 358, lộ id/tên tool 57, độ dài 50, chữ lạ (`foreign_script`) 15 |
| Khứ hồi đạt, trong mẫu qua lọc rẻ | 1550/3081 (50,3%): ba tiến trình 479/998 (48,0%), 527/1041 (50,6%), 544/1042 (52,2%) |
| **Được giữ** | **1550/4000 (38,8%)** |
| Câu chép y hệt câu thật trong train | 0/4000 |
| Ký tự Hán/Nhật/Hàn | 9 câu (đều bị loại, 0 trong số được giữ) |
| Câu trùng nhau trong số được giữ | 6 |
| Câu rỗng | 0 |
| Số từ mỗi câu được giữ | trung vị 21, p90 38, max 59 |
| Mẫu dùng 1 trong 6 app A4 giấu | 97/4000 (2,4%), 41 trong số được giữ (bị `build_sft` bỏ ở bộ chia A4) |

Theo số call (tổng, qua lọc rẻ, được giữ, % giữ):
| Số call | Tổng | Qua lọc rẻ | Được giữ | % giữ |
|---|---|---|---|---|
| 1 | 908 | 748 | 505 | 56% |
| 2 | 1053 | 875 | 477 | 45% |
| 3 | 899 | 661 | 308 | 34% |
| 4 | 609 | 448 | 162 | 27% |
| 5 | 293 | 191 | 61 | 21% |
| 6 | 146 | 99 | 22 | 15% |
| 7 | 92 | 59 | 15 | 16% |

Theo số app trong mẫu (tổng, được giữ): 1 app 1509 và 854; 2 app 1736 và 595; 3 app 633 và 89; 4 app 120 và 12; 5 app 2 và 0. App nhiều nhất trong số được giữ: reminders 410, calendar 266, chat_ott 229, alarm 206, banking 167, smart_home 136, notes 132, tasks 99.

**Từ 4000 đến 907 (xem trước `build_sft.py`, bộ chia A4):** 1791 mẫu (45%) có mẫu gốc hoặc donor không thuộc phần train của bộ chia này (`not_train`); trong 2209 mẫu còn lại, 1302 không qua bộ lọc (`roundtrip_failed`, gồm cả mẫu hỏng ở lọc rẻ lẫn khứ hồi), còn **907** (`ok`; không mẫu nào bị loại vì chép câu thật hay `text_ok`).

## 4. Điều cần cân nhắc trước khi train
1. **Lệch về chuỗi ngắn:** từ 3 call trở lên chỉ 568/2039 mẫu được giữ (27,9%), còn 1 đến 2 call là 982/1961 (50,1%). A4 cho thấy chuỗi dài là chỗ app lạ mất điểm nhiều nhất. Dữ liệu 907 mẫu của bộ chia A4 sẽ nghiêng về chuỗi ngắn tương tự (chưa có bảng riêng theo số call cho 907 mẫu).
2. **Thiếu mẫu 3 đến 4 app:** trong 1550 mẫu giữ, 1/2/3/4 app là 55,1%/38,4%/5,7%/0,8%, còn train thật là 48/37/11/4% (theo TASKS). Mẫu 3 đến 4 app mất nhiều nhất ở khâu lọc (3 app: 89/633 giữ, 14%; 4 app: 12/120, 10%). Nếu Phương muốn bù, có thể sinh lại với trọng số lệch lên nhiều đoạn (không làm khi chưa được dặn).
3. **Không có CONTROL ở lần này:** tỷ lệ khứ hồi đạt 50,3% so với 57 đến 61% ở bước thử 200 mẫu và 82% mức trần trên mẫu thật; mẫu lần này có nhiều chuỗi dài hơn (51% mẫu từ 3 call) nên khó so trực tiếp.
4. **Ước lượng thời gian train** (suy từ tỷ lệ số hàng so với A4, chưa đo): A4 train 2360 hàng mất 7758 s; thêm 907 hàng (tổng 3267) cho khoảng 3 giờ, nên dự trù **3 đến 3,5 giờ**, ít hơn mức 3,5 đến 4,5 giờ Phương dự kiến vì dữ liệu thêm ít hơn dự kiến (khoảng 2000). Mẫu A5 nhiều call dài hơn nên có thể nhỉnh hơn.

## 5. Điều chưa kết luận được
1. Chất lượng câu và nhãn **chưa duyệt**; tỷ lệ khứ hồi chỉ cho biết agent `q4b_lora` đạt cùng trạng thái, và bộ lọc giữ phần model đã giải được (xem `docs/BAN_GIAO_A5_LAI.md` mục 5).
2. Chưa có điểm nào chứng minh dữ liệu này giúp app lạ; lần train 1 (`q4b_holdout_a5`) mới đo được.
3. Số 907 phụ thuộc vào cách chia của A4; nếu train toàn bộ (lần 2) số mẫu dùng được sẽ lớn hơn (tối đa 1550 trước khi trừ mẫu dev; ước lượng chưa kiểm).

## 6. Đề xuất (Phương quyết định)
1. **Duyệt 30 dòng** `a5g_sample30_kept.txt` (mẫu được giữ, mỗi dòng ghi `writer` là 8b hay 9b, `text_ok`, `agent_ok` và mục id trỏ tới).
2. Nếu đạt: chạy notebook train lần 1 (`notebooks/a5-train-holdout.ipynb`) với 907 mẫu; notebook tự dừng nếu bộ chia không ra 1180/174/132 hoặc nếu chấm lại hai file mốc không ra 41,93 và 66,03.
3. Cân nhắc điểm 4.1 và 4.2: có muốn sinh thêm hoặc đổi trọng số trước khi train không (tốn thêm khoảng 1 đến 2 giờ GPU cho sinh và khứ hồi).

## 7. File
| File | Nội dung | Trong git? |
|---|---|---|
| `docs/results/a5g/tong_hop.txt` | Bản tổng hợp số đếm của 4000 mẫu | có |
| `docs/results/a5g/xem_truoc_build_sft.txt` | Dòng `callgen -> sft_train_cg.jsonl: 907 samples` và các số khác của `build_sft` | có |
| `docs/results/a5g/log_*.txt` | Log sinh call, viết câu (hai model), khứ hồi (ba tiến trình); chỉ có số đếm | có |
| `docs/results/a5g/model_8b_9b.txt`, `torch_cuda.txt`, `*.sha256` | Thông tin và đối chiếu sha256 model, môi trường, sha256 của zip và của `callgen_checked_all.jsonl` | có |
| `a5_gen_results.zip` (18 KB) | Gồm `a5g_sample30_kept.txt/.jsonl` (30 dòng của mẫu được giữ) và log | **không**, gửi qua Drive riêng |
| `a5_data/callgen_checked_all.jsonl` (trong output notebook `a5-gen` trên Kaggle) | 4000 mẫu và cờ kiểm tra; notebook train lấy file này | **không** (ở lại trên Kaggle) |

sha256 zip: `0c94a331…`; sha256 `callgen_checked_all.jsonl`: `ba436701…` (đầy đủ trong thư mục `docs/results/a5g/`).

## 8. Dòng nhật ký đề xuất cho `docs/EXPERIMENTS.md`
| Ngày | Commit | Việc | Điểm | Ghi chú |
|---|---|---|---|---|
| 2026-11-10 | `78c70a3` | **A5 sinh số lượng**: 4000 mẫu call (seed 42, `--weights 1:0.35,2:0.4,3:0.2,4:0.05`, mã đã sửa), Qwen3-8B và Qwen3.5-9B viết câu (một nửa mỗi model, `--no-verify`), khứ hồi `q4b_lora` 3 tiến trình, chưa train | không có điểm | Được giữ **1550/4000 (38,8%)** (8B 36,1%, 9B 41,5%); **907 mẫu dùng được cho bộ chia A4**; % giữ giảm theo số call (1 call 56%, 5 đến 7 call 15 đến 21%) và mẫu 3 đến 4 app ít (5,7% và 0,8% so với 11% và 4% ở train thật); 0 câu chép câu thật, 0 chữ Hán trong số giữ. viết câu 569 s và 523 s. Chờ duyệt 30 dòng. Chi tiết: `docs/BAN_GIAO_A5_GEN.md` |
