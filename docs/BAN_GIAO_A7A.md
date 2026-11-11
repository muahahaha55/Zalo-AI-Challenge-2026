# Bàn giao A7a: chạy toàn bộ giải pháp với trọng số thật (ASR sống + LLM đã gộp) trên `public_test`

Ngày 2026-11-11. Người chạy: thuan (Kaggle, 1 GPU RTX PRO 6000 Blackwell), làm đúng mục "A7a" của `docs/TASKS.md`. Không sửa mã của Phương (`scripts/verify_image.py` chạy nguyên bản; chỉ sửa bản sao trên Kaggle bằng `sed` để bỏ cờ `-I -X utf8` cho hợp Kaggle). Mã là commit `b2853d2`. Số liệu tổng hợp nằm ở `docs/results/a7a/`. `report.json` chỉ chứa số đếm, thời gian và id mẫu, không có nội dung dữ liệu; file `submission.jsonl` của ba lần chạy không nằm trong zip.

## 1. Tóm tắt
1. **Tất cả kiểm tra đạt** (log cuối: `ALL CHECKS PASSED`, `problems: []`).
2. **Ba lần chạy cho cùng câu trả lời** (`answers_identical: true`, bỏ trường `time`). sha256 file đầu ra khác nhau giữa ba lần vì mỗi dòng có trường `time` (mili giây); phần câu trả lời mới là thứ cần trùng.
3. `verify_submission`: **hash phát lại 634/634, 0 hash lệch, 0 lệnh bị từ chối, 0 tool ngoài app cài**.
4. **Mẫu chậm nhất 5,67 s** (ba lần: 5,67 / 5,63 / 5,65 s), dưới 30 s với dư địa lớn, nhưng đo trên GPU mạnh (mục 4).
5. **48/634 dòng (7,6%) khác `sub_public_v1.jsonl`.** Kỳ vọng của `TASKS.md` là trùng 634/634 hoặc ghi số dòng lệch kèm phân tích. Nguyên nhân **chưa tách được** (mục 5).

## 2. Cấu hình
`verify_image.py --no-docker --data-root data/raw/train+public_test --split public_test --runs 3 --out-dir check_a7a --ref <sub_public_v1.jsonl> -- --llm <q4b_merged> --adapter none --asr <phowl_ft/merged>`. Chạy cùng đường mã `run_container.py` như bản nộp: ASR sống (`phowl_ft` đã gộp) rồi Qwen3-4B đã gộp LoRA (`q4b_merged`, **không adapter**), offline. Đầu vào của model đã gộp là output notebook `merge-lora` (xem `docs/BAN_GIAO_MERGE.md`, sha256 `model.safetensors` `652e96ae…`). Bài nộp tham chiếu `sub_public_v1.jsonl` là bản sao `outputs/submission.jsonl` (634 dòng). Lưu ý: `report.json` không ghi đường dẫn LLM; việc dùng `q4b_merged` dựa vào cách dựng notebook `a7a-verify-real`, không phải vào một dòng trong báo cáo. Môi trường: GPU `NVIDIA RTX PRO 6000 Blackwell Server Edition`, driver 580.178.04, `torch 2.11.0+cu130`, CUDA 13.0, Python 3.13.16, `peft 0.21.2`, `transformers 5.20.2`, `accelerate 1.15.0`.

Ngoài lượt đầy đủ (3 lần × 634 mẫu) còn một lượt thử nhanh (2 lần × 20 mẫu) làm trước để bắt lỗi cấu hình.

## 3. Kết quả đối chiếu tiêu chí
| Tiêu chí | Kết quả |
|---|---|
| Ba lần trùng câu trả lời | **Đạt** (`answers_identical: true`) |
| Hash phát lại khớp | **Đạt**: 634/634, 0 lệch |
| Lệnh bị từ chối / tool ngoài app cài | **Đạt**: 0 / 0 |
| Mẫu chậm nhất < 30 s | **Đạt**: 5,67 s |
| Trùng `sub_public_v1.jsonl` | **48/634 lệch** (7,6%), cần giải thích |

Thời gian:
| Lần | Tổng | mean/mẫu | p95 | max |
|---|---|---|---|---|
| 1 | 1083,8 s | 1,63 s | 3,00 s | 5,67 s |
| 2 | 906,5 s | 1,42 s | 2,77 s | 5,63 s |
| 3 | 907,1 s | 1,42 s | 2,77 s | 5,65 s |

Lần 1 chậm hơn vì chạy nguội. Lý do dừng giống hệt ở cả ba lần: `end` 546, `ask` 75, `repeated_read` 6, `max_steps` 5, `invalid_call` 2. Có 33 mẫu `tool_calls` rỗng (5,2%), trung bình 2,31 call mỗi mẫu, nhiều nhất 8. Tổng thời gian cả phiên 2898 s.

Lượt thử nhanh 20 mẫu: 2 lần trùng nhau, hash 20/20, max 3,53 s, và 1/20 lệch tham chiếu (`public_0023`, cũng nằm trong 48 mẫu của lượt đầy đủ).

Mười id đầu tiên trong 48 mẫu lệch (ghi trong `report.json`): `public_0023, 0026, 0028, 0055, 0098, 0112, 0113, 0133, 0181, 0189`. Danh sách đủ 48 id không có trong báo cáo.

## 4. Điều chưa kết luận được
1. **Nguyên nhân 48 dòng lệch.** Hai khả năng, không tách được từ dữ liệu này:
   - **Model đã gộp:** trên dev, `q4b_merged` khác `q4b_lora` về `tool_calls` ở 4/155 mẫu (chữ sạch) và 3/155 (ASR), điểm không đổi (`docs/BAN_GIAO_MERGE.md`). Đó là khoảng 2 đến 3%, thấp hơn 7,6%.
   - **ASR sống:** bài nộp tham chiếu dùng bản chép lời `phowl_ft` tính sẵn theo lô; A7a chạy ASR từng mẫu, có thể ra chữ hơi khác nên LLM trả lời khác. Chưa so bản chép lời của hai đường.
   Chưa biết 48 dòng lệch tốt hơn hay xấu hơn bản tham chiếu: `public_test` không có nhãn nên không chấm được.
2. **Thời gian trên GPU của máy chấm.** Đo trên Blackwell 97 GB. GPU chấm chưa rõ (A30 hay RTX 5090, `docs/TRANG_THAI.md` ghi "cần hỏi BTC"); A7b (Docker, GPU 24 GB) mới cho số sát hơn. Mốc 5,67 s còn dư rất nhiều so với 30 s, nhưng chưa phải số ở máy chấm.
3. Đây là chạy `--no-docker`. Chưa kiểm được image Docker thật (A7b, người khác phụ trách) và chưa kiểm việc nạp model từ thư mục `models/` đúng bố cục image.
4. `models.lock.json` còn ghi shard Qwen3-4B gốc cho LLM; cần `check_models.py --update-lock` sau khi nhận `q4b_merged` (xem `docs/BAN_GIAO_MERGE.md` mục 4).

## 5. Đề xuất (Phương quyết định)
1. Coi A7a **đạt** về đúng đắn, tất định và thời gian; ghi nhận 48 dòng lệch kèm hai nguyên nhân chưa tách.
2. Muốn tách nguyên nhân thì chạy thêm trên Kaggle (khoảng 30 phút): suy luận `public_test` với **cùng bản chép lời tính sẵn** cho (a) Qwen3-4B + adapter và (b) `q4b_merged`, rồi so mỗi bên với `sub_public_v1.jsonl`. (a) là mẫu đối chứng, phải ra 0 dòng lệch (cùng adapter, cùng chữ chép); nếu không, đường chạy chưa tất định như tưởng. Số dòng lệch của (b) là phần do gộp bf16; phần còn lại của 48 (xấp xỉ 48 trừ số của (b)) là do ASR sống. Chỉ làm nếu cần chọn giữa bản gộp và bản adapter để dựng image.
3. Chuyển `q4b_merged/*`, `phowl_ft/merged/*`, `SHA256SUMS_merged.txt` (xem `docs/BAN_GIAO_MERGE.md` mục 5) cho chủ dự án qua Drive riêng để dựng image và làm A7b.

## 6. File
| File | Nội dung | Trong git? |
|---|---|---|
| `docs/results/a7a/report_634.json` | Báo cáo lượt đầy đủ (3 lần × 634 mẫu) | có |
| `docs/results/a7a/report_nhanh_20.json` | Báo cáo lượt thử nhanh (2 lần × 20 mẫu) | có |
| `docs/results/a7a/log_verify_image.txt`, `torch_cuda.txt`, `a7a_results.zip.sha256` | Log tổng hợp, môi trường, sha256 zip | có |
| `a7a_results.zip` (3 KB) | Gồm hai `report.json`, log, `torch_cuda.txt`, `MANIFEST_sha256.txt` | **không**, gửi qua Drive riêng |

sha256 zip: `5b443d3b…` (đầy đủ trong `docs/results/a7a/a7a_results.zip.sha256`).

## 7. Dòng nhật ký đề xuất cho `docs/EXPERIMENTS.md`
| Ngày | Commit | Việc | Điểm | Ghi chú |
|---|---|---|---|---|
| 2026-11-11 | `b2853d2` | **A7a**: chạy toàn bộ giải pháp trên `public_test` (634 mẫu × 3 lần), ASR sống `phowl_ft/merged` + `q4b_merged` không adapter, `--no-docker`, RTX PRO 6000 | không chấm được (không nhãn) | Đạt: 3 lần trùng câu trả lời, hash 634/634, 0 lệnh bị từ chối, max 5,67 s (mean 1,42 đến 1,63 s). 48/634 dòng lệch `sub_public_v1.jsonl`, nguyên nhân chưa tách (gộp bf16 hay ASR sống). Chi tiết: `docs/BAN_GIAO_A7A.md` |
