# Bàn giao kết quả A5 làm lại (bước thử 200 mẫu với mã đã sửa)

Ngày 2026-11-10. Người chạy: thuan (Kaggle, 1 GPU RTX PRO 6000 Blackwell), làm đúng các bước trong `docs/TASKS.md` mục "A5, kết quả duyệt lần thử đầu" và `docs/HUONG_DAN_GPU_08-11.md`. Không sửa mã của Phương. Mã dùng là commit `962ca6c` (dataset `lifemate-code` dựng lại từ commit này). **Chưa train gì.** Số liệu tổng hợp nằm ở `docs/results/a5r/`. Câu nói sinh ra, cờ kiểm tra từng mẫu (`callgen_checked_*.jsonl`) và 30 dòng để duyệt (`a5r_sample30_*.txt`) là dữ liệu cuộc thi nên **không** vào git, gửi qua Drive riêng (mục 8). **Số liệu của lần thử đầu (`docs/BAN_GIAO_A5_THU.md`) bị thay bằng bản này**, vì lần đó tính trên nhãn có id sai loại.

## 1. Tóm tắt
1. Làm đủ 5 bước của Phương cho cả Qwen3-8B và Qwen3.5-9B trên **cùng 200 mẫu call** (seed 42, `--weights 1:0.3,2:0.4,3:0.3`, mã mới: 98/200 mẫu có `refs`).
2. **Ba số theo yêu cầu** (khứ hồi bằng Qwen3-4B + adapter `q4b_lora`, `--check-all`, `--control-n 100`):

   | | Qwen3-8B | Qwen3.5-9B |
   |---|---|---|
   | **CONTROL** (100 mẫu thật, mức trần) | **82,0%** | **82,0%** |
   | Đạt khứ hồi, mẫu **qua** bộ lọc rẻ | 63/104 (60,6%) | 102/178 (57,3%) |
   | Đạt khứ hồi, mẫu **bị loại** bởi bộ lọc rẻ | 40/96 (41,7%) | 7/22 (31,8%) |

3. CONTROL 82% **cao hơn mức kỳ vọng 60 đến 70%** của Phương: người kiểm tra tin cậy được trên mẫu thật.
4. Tỷ lệ mẫu sinh ra mà agent đạt khứ hồi, trên **tất cả** 200 mẫu: **8B 103 (51,5%), 9B 109 (54,5%)**, tức khoảng 63% đến 66% mức trần 82%. So từng cặp cùng mẫu call, hai model **không khác nhau** (9B hơn 6 mẫu; 36 tốt lên, 30 tệ đi, 134 bằng).
5. Bộ lọc rẻ **có mang thông tin** ở cả hai model (mẫu qua lọc đạt khứ hồi cao hơn mẫu bị loại: +18,9 điểm [+5,3; +32,5] ở 8B, +25,5 [+4,7; +46,3] ở 9B), nhưng **phần "tự chấm CÓ/KHÔNG" của chính model viết câu là phần kém thông tin nhất và lệch theo độ dài chuỗi** (mục 4). Số mẫu "được giữ" khác nhau nhiều (8B 63, 9B 102) chủ yếu vì bước này, chứ không vì chất lượng câu khác nhau.
6. **Chưa ai đọc nội dung 30 dòng** (tôi chỉ dùng số tổng hợp). Chất lượng câu và nhãn vẫn cần Phương duyệt (mục 6).

## 2. Cấu hình
Giống lần thử đầu về model: Qwen3-8B (Kaggle Models `manojkumarcs28/qwen3-8b`, sha256 5/5 shard trùng `Qwen/Qwen3-8B`) và Qwen3.5-9B (`mirzamilanfarabi/qwen3.5`, biến thể `9b`, 4/4 shard trùng `Qwen/Qwen3.5-9B`); cả hai nạp bằng `AutoModelForCausalLM`, **không cần đường dự phòng** (log không có dòng `failed`). Người kiểm tra: Qwen3-4B + adapter `q4b_lora` lấy từ output notebook `Hello_done`, `--time-budget` mặc định 60 s. Môi trường: `torch 2.11.0+cu130`, CUDA 13.0, Python 3.13.16, `peft 0.21.2`, `transformers 5.20.2`, `accelerate 1.15.0`, VRAM 102 GB.

Thời gian: `gen_text` 57 s mỗi model (gồm nạp model); `verify_callgen` 628 s (8B) và 517 s (9B), mỗi lần gồm 100 mẫu CONTROL và 200 mẫu (`--check-all`). Cả phiên đủ trong khoảng 1 giờ như Phương dự kiến.

## 3. Kết quả trên 200 mẫu
| | Qwen3-8B | Qwen3.5-9B |
|---|---|---|
| Model tự chấm CÓ (`verified`) | 121/200 (60,5%) | 192/200 (96,0%) |
| Qua luật `text_ok` (bản mới) | 179/200 (89,5%) | 183/200 (91,5%) |
| Qua bộ lọc rẻ (cả hai) | 104/200 (52,0%) | 178/200 (89,0%) |
| Agent đạt khứ hồi (`agent_ok`, tất cả mẫu) | 103/200 (51,5%) | 109/200 (54,5%) |
| **Được giữ** (lọc rẻ và khứ hồi) | **63/200 (31,5%)** | **102/200 (51,0%)** |
| Lý do `text_ok` loại | giờ/số điện thoại mâu thuẫn nhãn 13, thiếu nội dung tham số 7, lộ id/tên tool 1 | 8, 6, 3 |
| Câu chép y hệt câu thật trong train | 0 | 2 (1,0%) |
| Ký tự Hán/Nhật/Hàn | 0 | 0 |
| Câu trùng nhau trong 200 | 0 | 0 |
| Độ dài (số từ) | trung vị 18, p90 24, max 48 | trung vị 20,5, p90 33, max 58 |

Mẫu dùng 1 trong 6 app A4 giấu: 3/200 (1,5%), được giữ 1 ở mỗi model. Số app mỗi mẫu 1/2/3 = 165/30/5.

Theo số call (tổng | qua lọc rẻ | `agent_ok` | được giữ):
| Số call | Tổng | 8B | 9B |
|---|---|---|---|
| 1 | 85 | 70 \| 54 \| 47 | 74 \| 53 \| 49 |
| 2 | 64 | 29 \| 29 \| 15 | 58 \| 36 \| 33 |
| 3 | 30 | 4 \| 12 \| 1 | 28 \| 11 \| 11 |
| 4 | 15 | 1 \| 6 \| 0 | 13 \| 7 \| 7 |
| 5 đến 7 | 6 | 0 \| 2 \| 0 | 5 \| 2 \| 2 |

## 4. Điều cần chú ý: bước "tự chấm CÓ/KHÔNG" kém thông tin và làm lệch độ dài
- **8B:** trả lời CÓ cho 97,6% mẫu 1 call nhưng chỉ 50,0% mẫu 2 call, **16,7% mẫu 3 call, 6,7% mẫu 4 call**, trong khi `agent_ok` ở 3 đến 4 call vẫn khoảng 40%. Nó loại chuỗi dài theo độ dài chứ không theo việc câu có đúng hay không: được giữ chỉ **1/51** mẫu từ 3 call trở lên.
- **9B:** trả lời CÓ cho 192/200 mẫu (96%; từ 86,7% đến 100% ở mọi độ dài), gần như không loại gì; được giữ 20/51 mẫu từ 3 call trở lên.
- Riêng bước tự chấm, tỉ lệ đạt khứ hồi của nhóm CÓ so với nhóm KHÔNG chỉ hơn nhẹ: 8B 56,2% so với 44,3%; 9B 55,2% so với 37,5% (nhóm KHÔNG chỉ 8 mẫu). Luật `text_ok` phân biệt rõ hơn: qua 54,7% (8B) và 56,8% (9B), bị loại 23,8% (21 mẫu) và 29,4% (17 mẫu).
- **Phép so tham khảo (chưa phải đề xuất mã):** nếu bỏ bước tự chấm và chỉ giữ mẫu qua `text_ok` **và** `agent_ok`, 8B giữ 98/200 và 9B giữ 104/200, cùng **20 mẫu từ 3 call trở lên** ở cả hai. Tức hai model viết câu tương đương nhau theo thước đo khứ hồi; hiệu "63 so với 102" nằm ở bước tự chấm. `gen_text.py` đã có `--no-verify`.

## 5. Điều chưa kết luận được
1. **Chất lượng câu và nhãn chưa duyệt**; tỷ lệ khứ hồi chỉ cho biết agent `q4b_lora` đạt cùng trạng thái. Mẫu không đạt có thể là câu xấu hoặc mẫu khó (khó thì có ích khi train), và mẫu đạt có thể là mẫu dễ (bộ lọc theo agent giữ phần model đã giải được).
2. Mức trần 82% đo trên 100 mẫu **thật** đầu `dev_ids.txt`; chưa biết mức trần theo số call, nên không biết 40% ở chuỗi 3 đến 4 call là kém hay là mức bình thường của người kiểm tra với chuỗi dài. Mẫu sinh ra cũng khác phân phối so với mẫu thật.
3. Các khác biệt giữa hai model đo trên 200 mẫu, một seed; nhóm "bị loại" của 9B chỉ có 22 mẫu và nhóm KHÔNG của bước tự chấm ở 9B chỉ 8 mẫu, nên các con số đó rất thiếu chắc.
4. 9B chép y hệt câu thật ở 2/200 mẫu (8B: 0): nhỏ, nhưng cần theo dõi khi sinh nhiều.
5. Kết quả của cell A-1 (đối chiếu sha256 của `q4b_lora` với `models.lock.json`) không nằm trong zip. CONTROL 82% phù hợp với adapter đúng; cần người chạy xác nhận dòng `q4b_lora: TRÙNG adapter của bài nộp`.

## 6. Đề xuất (Phương quyết định)
1. **Duyệt 30 dòng** `a5r_sample30_8b.txt` và `a5r_sample30_9b.txt` (cùng 30 id, đặt cạnh nhau; mỗi dòng ghi `verified`, `text_ok`, `agent_ok` và mục id trỏ tới). Đây là bước quyết định, vì các tỷ lệ trên chỉ là thước đo tự động.
2. Nếu 30 dòng đạt: chọn model viết câu theo chất lượng đọc được và tốc độ (hai model đều khoảng 1 phút cho 200 mẫu; 8B không phải dùng đường PyTorch chậm của 9B). Cân nhắc bỏ bước tự chấm của model viết câu (mục 4) hoặc dùng nó chỉ để loại, rồi lọc bằng `text_ok` và khứ hồi.
3. Khi sinh nhiều (`--n 6000`): chuỗi từ 3 call trở lên chiếm khoảng một phần tư mẫu (51/200) và chỉ khoảng 40% trong số đó qua khứ hồi, nên số mẫu dài còn lại ít; nên sinh dư hơn nữa cho chuỗi dài nếu muốn A5 dạy được chuỗi dài. Mẫu dùng app A4 giấu chỉ 1,5% và bị loại khi train theo bộ chia A4.
4. Không train A5 khi chưa có duyệt (đúng hướng dẫn).

## 7. Đối chiếu với các bước Phương yêu cầu
| Bước | Kết quả |
|---|---|
| 1. Kéo `main` (từ `620b3f7`), dựng lại dataset mã | Làm: `962ca6c`; cell A-1 chặn nếu mã cũ |
| 2. `gen_calls --n 200 --weights 1:0.3,2:0.4,3:0.3` | Làm |
| 3. Viết câu bằng Qwen3-8B và Qwen3.5-9B, hai tệp riêng | Làm |
| 4. Khứ hồi với `q4b_lora`, `--check-all`, `--control ... --control-n 100`, ba số | Làm; ba số ở mục 1 |
| 5. Gửi: tổng hợp, ba số, 30 dòng mới (cùng 30 id cho hai model, kèm `verified`, `text_ok`, `agent_ok`) | Đủ; 30 id trùng nhau giữa hai model |

## 8. File
| File | Nội dung | Trong git? |
|---|---|---|
| `docs/results/a5r/tong_hop_8b.txt`, `tong_hop_9b.txt` | Bản tổng hợp số đếm từng model | có |
| `docs/results/a5r/so_sanh_8b_9b.txt` | So hai model, tính thông tin của từng bộ lọc, theo số call, phép so bỏ bước tự chấm | có |
| `docs/results/a5r/log_verify_*.txt`, `log_gen_text_*.txt`, `log_gen_calls.txt` | Log chạy (chỉ số đếm) | có |
| `docs/results/a5r/model_8b_9b.txt`, `torch_cuda.txt`, `a5_retry_results.zip.sha256` | Thông tin model và đối chiếu sha256, môi trường, sha256 của zip | có |
| `a5_retry_results.zip` (292 KB) | Gồm `callgen_checked_8b/9b.jsonl` (200 câu và call mỗi model, cờ kiểm tra) và `a5r_sample30_*.txt/.jsonl` | **không**, gửi qua Drive riêng |

sha256 của zip: `493dd6d9…` (đầy đủ trong `a5_retry_results.zip.sha256`).

## 9. Dòng nhật ký đề xuất cho `docs/EXPERIMENTS.md`
| Ngày | Commit | Việc | Điểm | Ghi chú |
|---|---|---|---|---|
| 2026-11-10 | `962ca6c` | **A5 làm lại** 200 mẫu call (seed 42, `--weights 1:0.3,2:0.4,3:0.3`, mã mới), Qwen3-8B và Qwen3.5-9B viết câu, khứ hồi bằng Qwen3-4B + `q4b_lora`, chưa train | không có điểm (chưa train) | **CONTROL 82/100**; đạt khứ hồi mẫu qua/bị loại bởi lọc rẻ: 8B 63/104 (60,6%) và 40/96 (41,7%), 9B 102/178 (57,3%) và 7/22 (31,8%); `agent_ok` trên 200 mẫu: 8B 103, 9B 109 (hai model không khác nhau, 36/30/134); được giữ 8B 63, 9B 102, nhưng khác biệt do bước tự chấm (8B CÓ 97,6% với 1 call và 6,7% với 4 call; 9B CÓ 96%); bỏ bước tự chấm thì giữ 98 và 104, cùng 20 mẫu ≥3 call. 57 s gen_text, 628 s và 517 s khứ hồi. Thay số lần thử đầu. Chất lượng câu **chưa duyệt** (chờ 30 dòng). Chi tiết: `docs/BAN_GIAO_A5_LAI.md` |
