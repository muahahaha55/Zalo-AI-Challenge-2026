# Bàn giao kết quả A5, bước thử 200 mẫu (Qwen3-8B và Qwen3.5-9B)

Ngày 2026-11-09. Người chạy: thuan (Kaggle, 1 GPU RTX PRO 6000 Blackwell), theo `docs/TASKS.md` mục A5 và `docs/HUONG_DAN_GPU_08-11.md`. Không sửa mã của Phương. Mã dùng là commit `404d56d` (dataset `lifemate-code` dựng lại từ commit này). **Chưa train gì.** Số liệu tổng hợp nằm ở `docs/results/a5/`; câu nói sinh ra (`callgen_text_*.jsonl`) và 30 dòng để duyệt (`a5_sample30.txt`) là dữ liệu cuộc thi nên **không** vào git, gửi qua Drive riêng (mục 7).

## 1. Tóm tắt
1. Cả hai model nạp và chạy được trên Kaggle, không lỗi: **Qwen3-8B 49 giây, Qwen3.5-9B 53 giây** cho 200 mẫu (gồm nạp model, viết câu và tự chấm). Qwen3.5-9B nạp bằng `AutoModelForCausalLM` ngay lần đầu, **không phải dùng đường dự phòng** `AutoModelForImageTextToText` (log không có dòng `AutoModelForCausalLM failed`).
2. Hai model đều là bản gốc: sha256 các shard trùng `Qwen/Qwen3-8B` (5/5) và `Qwen/Qwen3.5-9B` (4/4) trên Hugging Face, dù nguồn Kaggle là bản của người dùng khác (mục 2).
3. Tỷ lệ mẫu được giữ sau lọc (model tự chấm CÓ **và** qua luật `text_ok`): **8B 148/200 (74,0%), 9B 171/200 (85,5%)**. Không câu nào chép y hệt câu thật trong train (0/200 cả hai), không câu trùng nhau.
4. **Điều tôi chưa làm: chưa đánh giá chất lượng câu.** Chưa ai đọc 30 dòng. Tỷ lệ "CÓ" do chính model viết tự chấm, nên không đủ làm bằng chứng (với Qwen3-1.7B, model tự chấm trả lời CÓ cả 24/24 câu, gồm câu sai). Cần Phương duyệt `a5_sample30.txt` trước khi quyết định (mục 6).
5. Hai điểm bộ lọc hiện tại chưa xử lý (mục 4): (a) **câu còn chữ Hán/Nhật/Hàn lọt qua lọc**: 1 câu ở 8B và 2 câu ở 9B, cả ba đều được giữ; (b) tỷ lệ giữ **giảm mạnh theo số call** (8B: 92% với 1 call, 40% với 4 call), nên dữ liệu sau lọc lệch về chuỗi ngắn.

## 2. Cấu hình
| | Qwen3-8B | Qwen3.5-9B |
|---|---|---|
| Nguồn trên Kaggle | Kaggle Models `manojkumarcs28/qwen3-8b` (bản người dùng khác đăng) | Kaggle Models `mirzamilanfarabi/qwen3.5` (bản người dùng khác đăng, biến thể `9b`) |
| Kiểm tra bản gốc | **5/5 shard trùng sha256** với `Qwen/Qwen3-8B` | **4/4 shard trùng sha256** với `Qwen/Qwen3.5-9B` |
| `model_type`, kích thước | `qwen3`, hidden 4096, 36 tầng, 16,38 GB | `qwen3_5` (đa phương thức), hidden 4096, 32 tầng, 19,31 GB |
| Cách nạp | `AutoModelForCausalLM` | `AutoModelForCausalLM` (đường dự phòng không cần) |
| Thời gian `gen_text` (gồm nạp) | 49 s | 53 s |

Dùng chung: `gen_calls.py --n 200 --seed 42` (200 mẫu call giống hệt nhau giữa hai lượt: 579 lần thử, 75 bị từ chối bởi `validate/call`, 8 trùng), `gen_text.py --batch 32` (nhiệt độ 0,8, `max_new_tokens` 96, mặc định của script). Môi trường: `torch 2.11.0+cu130`, CUDA 13.0, Python 3.13.16, `transformers 5.20.2`, `accelerate 1.15.0`, VRAM 102 GB. `dev_ids.txt` có đúng 155 id (notebook dừng nếu khác).

Với Qwen3.5-9B, log `transformers` có 4 cảnh báo `... falling back to its reference PyTorch implementation because causal_conv1d / flash-linear-attention is not installed ... much slower`. Vẫn chạy nhanh ở quy mô 200 mẫu; chưa biết ảnh hưởng khi sinh số lượng lớn.

## 3. Kết quả trên 200 mẫu
| | Qwen3-8B | Qwen3.5-9B |
|---|---|---|
| Model tự chấm CÓ (`verified`) | 158/200 (79,0%) | 179/200 (89,5%) |
| Qua luật `text_ok` | 186/200 (93,0%) | 191/200 (95,5%) |
| **Qua cả hai (được giữ)** | **148/200 (74,0%)** | **171/200 (85,5%)** |
| Lý do `text_ok` loại | thiếu nội dung tham số 13, ngôi thứ ba 1 | thiếu nội dung tham số 8, lộ ID hoặc tên tool 1 |
| Câu chép y hệt câu thật trong train | 0/200 | 0/200 |
| Câu trùng nhau trong 200 | 0 | 0 |
| Độ dài (số từ) | trung vị 18, p90 25, min 5, max 48 | trung vị 21,5, p90 36, min 8, max 56 |
| Tham chiếu: câu thật trong train (1000 câu đầu) | trung vị 22, p90 35 | |
| Câu có dấu `?` | 7 | 3 (câu thật: 29/1000) |
| Cùng một mẫu, hai model viết giống hệt | 0/200 | |

Hai model giữ cùng 135 mẫu; chỉ 8B giữ 13 mẫu, chỉ 9B giữ 36 mẫu, 16 mẫu không model nào giữ được. Quyết định CÓ/KHÔNG của hai model trùng nhau ở 155/200 mẫu. Trong 30 dòng ngẫu nhiên (cùng 30 id ở hai lượt, seed 0): 8B có 19/30 qua cả hai bộ lọc, 9B có 24/30. Độ dài của 9B gần câu thật hơn (trung vị 21,5 so với 22), 8B ngắn hơn (18).

Cơ cấu 200 mẫu call (giống hệt giữa hai lượt): số call 1/2/3/4/5/6/7 = 90/59/29/15/2/4/1; số app mỗi mẫu 1/2/3 = 171/23/6; app nhiều nhất: reminders 41, calendar 32, smart_home 29, chat_ott 19, alarm 15, banking 14. Mẫu dùng 1 trong 6 app A4 giấu: 5/200 (2,5%); sau lọc còn 4 (8B) và 5 (9B). Với bộ chia A4 các mẫu này bị bỏ khi train.

## 4. Hai điểm cần chú ý ở bước lọc
**(a) Chữ Hán/Nhật/Hàn lọt qua lọc.** Một câu của 8B và hai câu của 9B chứa ký tự CJK (Qwen thỉnh thoảng đổi ngôn ngữ giữa câu); cả ba đều nằm trong số được giữ, vì `text_ok` không kiểm tra loại ký tự và model tự chấm vẫn trả lời CÓ. Thêm một dòng nữa: một câu của 9B không có chữ cái tiếng Việt có dấu nào (không nằm trong số được giữ). Đề xuất (Phương quyết định): thêm luật loại câu có ký tự ngoài bảng chữ cái tiếng Việt và các dấu câu thông thường. Tôi chưa sửa mã.

**(b) Tỷ lệ giữ giảm theo số call** (giữ/tổng):
| Số call | Tổng | 8B giữ | 9B giữ |
|---|---|---|---|
| 1 | 90 | 83 (92%) | 85 (94%) |
| 2 | 59 | 41 (69%) | 51 (86%) |
| 3 | 29 | 15 (52%) | 20 (69%) |
| 4 | 15 | 6 (40%) | 10 (67%) |
| 5 đến 7 | 7 | 3 | 5 |

A4/A4b cho thấy app lạ mất điểm nhiều nhất ở chuỗi dài (3 call khoảng 30 điểm). Nếu sinh số lượng lớn rồi lọc như hiện tại, dữ liệu A5 sẽ ít chuỗi dài hơn dự định, nhất là với 8B. 9B giữ nhiều mẫu dài hơn nhưng chưa biết đó là câu tốt hơn hay model tự chấm dễ dãi hơn (CÓ 89,5% so với 79,0%); chỉ đọc 30 dòng mới phân biệt được.

## 5. Điều chưa kết luận được
1. **Chất lượng câu: chưa đo.** Số trong mục 3 chỉ là tỷ lệ qua hai bộ lọc tự động. Không có số nào cho biết câu có yêu cầu đúng và đủ các call hay không.
2. 9B giữ nhiều hơn có thể do viết tốt hơn, hoặc do dễ dãi khi tự chấm, hoặc do câu dài và nhiều chi tiết hơn nên dễ qua luật "thiếu nội dung tham số". Chưa tách được.
3. Mỗi lượt là 200 mẫu, một seed; chưa đo độ biến động giữa các lần chạy.
4. Chưa chạy bước khứ hồi (`verify_callgen.py`), lớp lọc mà `TASKS.md` gọi là quan trọng nhất; sau lọc hiện tại các tỷ lệ trên mới là lọc lớp 1 và lớp 2.
5. Thời gian cho số lượng lớn chỉ ước lượng thô: nếu tuyến tính, 6000 mẫu cho `gen_text` khoảng 15 đến 30 phút mỗi model (200 mẫu mất chưa đầy 1 phút kể cả nạp). Chưa đo, và 9B chạy bằng đường PyTorch chậm nên có thể dài hơn.

## 6. Đề xuất (Phương quyết định)
1. **Duyệt 30 dòng trước.** `a5_sample30.txt` trong mỗi zip có 30 dòng với cùng 30 id ở cả hai lượt, nên có thể đặt cạnh nhau (cùng cuộc gọi, hai câu của hai model). Mỗi dòng ghi sẵn `verified` và `text_ok`, để thấy bộ lọc có bắt được câu xấu không (đặc biệt các dòng `verified=True` mà vẫn sai).
2. Nếu 9B đạt: dùng 9B cho bước viết câu, nhưng bước khứ hồi (`verify_callgen.py`) dùng Qwen3-8B hoặc 14B như đã ghi trong hướng dẫn, vì script đó dùng `AutoModelForCausalLM`. Nếu cả hai chưa đạt: thử 14B rồi 32B (cùng 200 mẫu seed 42, notebook đã hỗ trợ `GEN_SIZE=14b` và `32b`).
3. Trước khi sinh nhiều: thêm luật loại ký tự lạ (mục 4a) và cân nhắc sinh dư mẫu nhiều call (mục 4b). Với bộ chia A4 còn nhiều mẫu bị bỏ (ở thử nghiệm local là 183/300), nên cần `--n 6000` như đã ghi.
4. Không train A5 khi chưa có duyệt (đúng hướng dẫn).

## 7. File
| File | Nội dung | Trong git? |
|---|---|---|
| `docs/results/a5/tong_hop_8b.txt`, `tong_hop_9b.txt` | Bản tổng hợp số đếm của từng lượt | có |
| `docs/results/a5/so_sanh_8b_9b.txt` | So hai model, đối chiếu câu thật, số câu CJK | có |
| `docs/results/a5/log_gen_text_*.txt`, `log_gen_calls.txt` | Log chạy (chỉ có số đếm tiến độ) | có |
| `docs/results/a5/model_8b.txt`, `model_9b.txt` | Đường dẫn model, sha256 `config.json`, kết quả đối chiếu sha256 từng shard | có |
| `docs/results/a5/torch_cuda.txt`, `*.zip.sha256` | Môi trường; sha256 hai zip | có |
| `a5_trial_8b_results.zip` (137 KB), `a5_trial_9b_results.zip` (140 KB) | Gồm cả `callgen_text_*.jsonl` (200 câu và call) và `a5_sample30.txt/.jsonl` | **không**, gửi qua Drive riêng |

sha256 hai zip: `1541c5e4…` (8B) và `bf52b1c1…` (9B).

## 8. Dòng nhật ký đề xuất cho `docs/EXPERIMENTS.md`
| Ngày | Commit | Việc | Điểm | Ghi chú |
|---|---|---|---|---|
| 2026-11-09 | `404d56d` | A5 thử 200 mẫu: Qwen3-8B và Qwen3.5-9B viết câu cho cùng 200 mẫu call (seed 42), chưa train | không có điểm (chưa train) | giữ sau lọc: 8B 148/200 (74,0%), 9B 171/200 (85,5%); 49 s và 53 s; 0 câu chép câu thật; câu CJK lọt lọc: 1 (8B) và 2 (9B); tỷ lệ giữ giảm theo số call (8B: 92% với 1 call, 40% với 4 call); chất lượng câu **chưa duyệt** (chờ 30 dòng) |
