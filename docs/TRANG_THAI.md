# Trang tình trạng (một trang, cập nhật 10/11 tối)

Mục đích: ai mở repo cũng biết **đang làm gì, đã quyết gì, còn thiếu gì, ai làm**. Chi tiết từng thí nghiệm ở `docs/EXPERIMENTS.md`, việc cho người chạy GPU ở `docs/TASKS.md`, đóng gói ở `docs/DOCKER.md`.

## 1. Bài nộp hiện tại (đã đo)
Qwen3-4B + LoRA (`q4b_lora`) và PhoWhisper-large đã fine-tune, luật `rules.py` bật. Dev 56,98 (qua ASR) và 66,82 (chữ sạch); **public 48,56** (nộp lần 1). Hệ thống 5,698B tham số (adapter chạy kèm), dưới 6B.

## 2. Các thí nghiệm đã xong
| Thí nghiệm | Kết quả | Quyết định |
|---|---|---|
| A1: thêm dữ liệu từ `history` | không thấy khác biệt | không dùng |
| A4: giấu 6 app, đo trên app lạ | **41,93** (app lạ) | thí nghiệm gốc để đo A5 |
| A4b: cùng model trên app đã thấy | **66,03**; **khoảng cách 21 điểm** | app lạ là điểm yếu chính, đáng làm A5 |
| A6: thêm câu ASR nhiễu | tiền đề sai (model đã học trên ASR gốc) | **rút lại** |
| A5: 4000 mẫu tự sinh, train theo bộ chia A4 | app lạ 39,35 (mốc 41,93), exec không đổi | **không đạt, dừng** (mục 3) |
| Gộp LoRA vào Qwen3-4B | dev trùng 66,82 và 56,98 | dùng cho bài nộp |
| A7a: toàn bộ giải pháp trên public_test | đúng, tất định, max 5,67 s | đạt (mục 4) |

## 3. A5 (dữ liệu tự sinh): đã thử và DỪNG (11/11)
Sinh 4000 mẫu (giữ 1550; dùng được 907 cho bộ chia A4), train `q4b_holdout_a5` theo bộ chia A4, đo trên app lạ (174 mẫu) và app đã thấy (132 mẫu):

| | `q4b_holdout` (mốc) | `q4b_holdout_a5` | Chênh từng cặp |
|---|---|---|---|
| App lạ | 41,93 | **39,35** | −2,58 [−9,09; +4,05] |
| App lạ, loại exec (nhắm vào đây) | 42,80 | 42,16 | −0,64 |
| App đã thấy | 66,03 | 63,93 | −2,10 |

**Tiêu chí đặt trước (app lạ tăng ít nhất 5 điểm): không đạt.** Đáng chú ý là loại exec, loại mẫu A5 nhắm tới, **không đổi** (−0,64), còn ask và oos tụt. Giả thuyết (chưa kiểm chứng): mẫu A5 chỉ gồm gọi tool, nên kéo model bớt hỏi lại. **Quyết định: không train lần 2, không sinh thêm dữ liệu; giữ `q4b_lora` (bản đã gộp) làm bài nộp.** Điểm yếu app lạ (chênh 21 điểm so với app đã thấy) vẫn còn, ta không có cách đã chứng minh để thu hẹp nó trước hạn chốt.

## 4. Bản đã gộp LoRA và A7a (đã làm)
- **Gộp LoRA vào Qwen3-4B:** dev trùng hoàn toàn (66,82 sạch, 56,98 ASR), 4,0225B tham số, một file `model.safetensors` (khoảng 8 GB) sha256 `652e96ae…`. Đủ điều kiện làm bài nộp (BTC không cho đưa base model vào image).
- **A7a (public_test 634 mẫu, ASR sống, 3 lần):** trùng câu trả lời, hash 634/634, 0 lệnh bị từ chối, **mẫu chậm nhất 5,67 s**; **48/634 dòng khác `sub_public_v1.jsonl`** (nguyên nhân chưa tách: gộp bf16 hay ASR sống). Đo trên Blackwell 97 GB, chưa phải máy chấm.

## 5. Đóng gói nộp (BTC hướng dẫn 10/11)
| Mục | Tình trạng |
|---|---|
| Khung `/code` (predict.sh, predict.py, notebook, README, training_code) | **đã viết và chạy thử trong Docker với model nhỏ** (GPU, hai lần giống nhau, notebook, Jupyter) |
| Dockerfile (CUDA 13.2, không ENTRYPOINT, torch 2.11.0, đã gỡ `cuda-compat`) | **đã dựng và thử với model nhỏ** (20 GB); bản thật chưa dựng |
| Gộp LoRA vào Qwen3-4B (`scripts/merge_lora.py`) | **xong**, dev trùng; trọng số đang ở bạn kia, **chờ chuyển qua Drive** |
| Test Solution của BTC | 1 lượt/ngày, **đóng 23:59 ngày 15/11** |
| Nộp chính thức | **23:59 ngày 16/11**, tối đa 2 lượt, lượt cuối được chấm |
| GPU của máy chấm | **chưa rõ** (RTX 5090 hay A30), cần hỏi BTC |
| Dữ liệu huấn luyện trong gói | **cần quyết**: tôi đề xuất chỉ nộp mã tái tạo, không đăng dữ liệu dẫn xuất công khai |

## 6. Mốc
| Ngày | Việc |
|---|---|
| 11/11 | Nhận hai thư mục model từ bạn kia, dựng image thật, tải lên Drive, **Test Solution lượt đầu** |
| 12 đến 13/11 | Đọc kết quả BTC, sửa theo log; nộp kết quả public của đường chạy cuối để so với 48,56 |
| 13/11 | **Chốt cấu hình** (đã quyết: `q4b_lora` bản gộp, trừ khi có lỗi) |
| 14 đến 15/11 | Test Solution thêm lượt, kiểm tra cuối |
| 16/11 | Nộp chính thức (Drive "Anyone with the link" + MD5), **trước 00:00 ngày 16/11 phải chốt danh sách thành viên đội** |
| 20/11, 23/11 | Private test, nộp kết quả cuối |

## 6b. Từ khi người giữ GPU xong đến lúc có bài nộp (việc của chủ dự án)
| # | Việc | Phụ thuộc | Ước lượng |
|---|---|---|---|
| 1 | Nhận `q4b_merged` và `phowl_ft/merged` (khoảng 11 GB) về máy, đặt vào `models/llm` và `models/asr`, xóa `models/adapter` | bước 0 của người giữ GPU | 1 giờ (tải) |
| 2 | `python scripts/check_models.py --update-lock` (ghi sha256 mới), kiểm giới hạn tham số (kỳ vọng 5,566B) | 1 | 5 phút |
| 3 | `bash scripts/build_submission.sh <TeamName>`: dựng image (cu130 như lúc train), `docker save | gzip`, MD5 | 2, Docker chạy, ổ C còn trống khoảng 60 GB | khoảng 1,5 giờ (dựng khoảng 45 phút, nén khoảng 30 đến 40 phút) |
| 4 | Tải `zac2026_<TeamName>.tar.gz` (khoảng 20 GB) lên Google Drive, chia sẻ "Anyone with the link" | 3, mạng | khoảng 1 đến 3 giờ |
| 5 | **Test Solution lượt đầu** (ô Google URL và MD5 trên cổng) | 4 | 1 lượt/ngày, hạn 15/11 |
| 6 | Đọc kết quả BTC: lỗi thì xem log, sửa, dựng lại, thử lượt kế (ngày sau) | 5 | theo log |
| 7 | **Chốt cấu hình 13/11:** `q4b_lora` đã gộp hoặc `q4b_a5` (nếu thí nghiệm đạt và dev không giảm) | kết quả A5, bước 0 | quyết định |
| 8 | Nếu dùng A5: sinh lại 4000 mẫu bằng bộ sinh đã sửa, train toàn bộ, gộp, đo dev, rồi lặp 1 đến 6 | 7 | khoảng 7 giờ GPU thêm, chỉ khi A5 đạt tiêu chí |
| 9 | Kiểm tra cuối: README (`submission/README.md`, `training_code/README.md`) khớp cấu hình thật; `check_compliance`; chạy lặp ba lần | 7 | 1 giờ |
| 10 | **Chốt danh sách thành viên đội trước 00:00 ngày 16/11** (cổng khóa sau đó) | | 5 phút |
| 11 | **Nộp chính thức qua form "Solution Submission"** (link Drive, MD5): nộp lượt đầu **sớm** (15/11 hoặc sáng 16/11), giữ lượt thứ hai để sửa; BTC chỉ chấm lượt cuối | 9 | hạn 23:59 ngày 16/11 |
| 12 | Sau 20/11: BTC chạy image trên private_test; nộp kết quả cuối theo thông báo của BTC (hạn 23/11) | | |

**Điểm có thể trượt:** (a) tải 20 GB lên Drive chậm; (b) lỗi chỉ lộ khi BTC chạy (cu130, GPU 5090 hay A30); (c) mô hình gộp đo lại dev tụt điểm (khi đó phải xử lý trước khi nộp); (d) ổ C: gần đầy làm Docker sập.

## 7. Việc đang chờ
| Việc | Ai |
|---|---|
| Chuyển `q4b_merged/` (8 GB), `phowl_ft/merged/` (3 GB) và `SHA256SUMS_merged.txt` qua Drive riêng; gửi thêm `submission.jsonl` của A7a (đầu ra public_test của đường chạy cuối) | bạn kia (GPU) |
| Giải phóng ổ C: (còn 43 GB; cần khoảng 45 GB cho dựng và xuất), mở lại Docker Desktop | bạn |
| Thư hỏi BTC (GPU chấm, dữ liệu huấn luyện, gộp LoRA) | bạn gửi, tôi soạn |
| Chốt danh sách thành viên đội trước 00:00 ngày 16/11 | bạn |
