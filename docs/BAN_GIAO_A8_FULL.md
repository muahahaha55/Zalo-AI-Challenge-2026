# Bàn giao A8 đầy đủ (`q4b_a8_full`): train bản nộp có che tên trên toàn bộ train, đo dev

Ngày 2026-11-15. Người chạy: thuan (Kaggle, **4 GPU L4**), notebook `notebooks/a8-full.ipynb` với cell F-3 sửa cho L4 x4 (bản sửa chạy trên Kaggle không nằm trong repo; ba thay đổi liệt kê ở mục 2). Mã: dataset `lifemate-code` có `masking.py` (commit `d5f376c` trở lên). Ý tưởng: `docs/Y_TUONG_CHE_TEN.md`; hướng dẫn gốc của Phương: `docs/HUONG_DAN_A8_FULL.md`. Số liệu tổng hợp ở `docs/results/a8_full/`. Dự đoán dev (`dev_a8_merged_*.jsonl`) là dữ liệu cuộc thi nên **không** vào git, gửi qua Drive riêng.

## 1. Tóm tắt
1. **Phán quyết F-6: ĐẠT cả ba điều kiện dev.** Dev sạch **68,94** (mốc 66,82, +2,12); dev ASR `phowl_ft` **58,43** (mốc 56,98, +1,45); ask (sạch) **73,08** (ngưỡng 70,64; mốc 75,64 nên vẫn tụt 2,56). exec 66,54 (mốc 65,73), oos 83,33 (mốc 58,33).
2. **Nhưng bản này, theo luật của hướng dẫn, phải bỏ:** train **dừng ở bước 2410/2505 (epoch 2,886)** vì `TimeLimit` (`--max-hours 8.5`); notebook in `CHẠM GIỚI HẠN ... BẢN NÀY BỎ` và hướng dẫn ghi bản dừng sớm không dùng được. Điểm giảm nhẹ (không đổi luật): lịch học cosine còn **0,35% đỉnh** ở bước 2410, loss cuối 0,053, và dev đo đúng trên checkpoint này. Phương quyết định.
3. **Phần tăng gần như chỉ đến từ oos.** Trong +2,12: oos **+1,94** (12 mẫu dev, khoảng 3 mẫu đúng thêm), exec +0,61, ask −0,43 (`docs/results/a8_full/dong_gop_theo_loai.txt`). Dev 155 mẫu nhiễu khoảng ±4 điểm, nên không đủ để nói A8 đầy đủ tốt hơn bài hiện tại. Khớp với A8 trên bộ chia A4 (+2,56, khoảng tin cậy chứa 0, ask tụt).
4. **Không kịp Test Solution (đóng 23:59 ngày 15/11).** Có kết quả lúc 22:46; dựng image v2 mất 2 đến 2,5 giờ theo `docs/THUE_MAY_DUNG_IMAGE.md`. Theo hướng dẫn: **bỏ v2, giữ bài v1** (`zac2026_ManchesterCity.tar.gz`, MD5 `6e2d32ad…`, Test Solution đạt ngày 14/11). Chưa dựng image nào, chưa đổi `models.lock.json`.
5. Tốc độ suy luận trên L4 với model đã gộp: trung bình khoảng 3,9 s mỗi mẫu, chậm nhất **10,31 s** (8 phân đoạn, 4 GPU, hai lần đo sạch và ASR; chữ ASR là bản `phowl_ft` đã nhận dạng sẵn, không gồm thời gian nhận dạng thực).

## 2. Cấu hình
Dữ liệu (`build_sft.py --asr-tag phowl --aug-oos 100 --aug-mask 1390`): `sft_train` 1390 mẫu + `sft_train_asr` 1390 + `sft_train_aug` 100 mẫu oos + `sft_train_mask` 1390 bản sao che tên (655 dùng chữ ASR; 0 mẫu bị bỏ). Bộ chia dev **không đổi** (155 mẫu, 141 nhóm; F-2 kiểm `cmp` với bản trước khi dựng). Preflight: 4 file OK, bước dài nhất 4680 token, 0 bước quá `--max-len 5120`, **13 360 bước mỗi epoch**.

Train: `torchrun --nproc_per_node=4 scripts/train_sft.py --data sft_train sft_train_asr sft_train_aug sft_train_mask --epochs 3 --max-len 5120 --global-batch 16 --save-limit 2 --max-hours 8.5`, tức 4 GPU × bs 1 × accum 4 = batch tổng 16, kế hoạch 2505 bước tối ưu. Khác `notebooks/a8-full.ipynb` của Phương chỉ ở cell F-3: cho phép 1 hoặc 4 GPU, `--max-hours` 5 → 8,5, ngưỡng cảnh báo 17 800 → 30 000 giây.

Môi trường: `torch 2.11.0+cu130`, `transformers 5.20.2`, `peft 0.21.2`, `accelerate 1.15.0`, 4 × L4 24 GB (`docs/results/a8_full/torch_cuda.txt`).

## 3. Thời gian
| | |
|---|---|
| Cả cell F-3 (từ lúc chạy `torchrun`) | **30 797 s (8 giờ 33 phút)** |
| Thanh tiến trình / `train_runtime` | 7 giờ 01 phút (25 280 s), 2410 bước |
| Gộp LoRA (F-4) | 22 s |

Ước trước của tôi là 7 đến 8,5 giờ; thực tế vượt giới hạn. **Có một khoảng chưa giải thích được:** chênh giữa cell (30 797 s) và `train_runtime` (25 280 s) là khoảng **1,5 giờ**, trong khi lần A8 trước trên cùng loại máy chênh 13 phút (23 100 s so với 22 290 s). Log chỉ có mốc thời gian lúc khởi động (06:14:43) và lúc thoát (14:47:57); không có dòng nào cho biết thời gian đó mất vào đâu (nạp model từ `/kaggle/input`, tokenize 13 360 mẫu và khởi động DDP đều nằm trước khi vòng train bắt đầu). Theo mã (`scripts/train_sft.py`), `TimeLimit` tính từ lúc dựng `Trainer`; nếu vậy, giới hạn 8,5 giờ chạm ở 7 giờ 01 của thanh tiến trình nghĩa là khoảng 1,5 giờ đã trôi qua giữa lúc dựng `Trainer` và bước đầu tiên. Log không cho biết vì sao, nên không có số liệu để nói chắc nguyên nhân.

## 4. Điểm dev (155 mẫu)
| | A8 đầy đủ | Mốc (`q4b_lora` đã gộp) | Chênh |
|---|---|---|---|
| Dev sạch | **68,94** | 66,82 | +2,12 |
| ask / exec / oos (sạch) | 73,08 / 66,54 / 83,33 | 75,64 / 65,73 / 58,33 | −2,56 / +0,81 / +25,00 |
| Dev ASR `phowl_ft` | **58,43** | 56,98 | +1,45 |
| ask / exec / oos (ASR) | 65,38 / 56,90 / 58,33 | (chưa có mốc theo loại) | |

Chi tiết theo số call và `history`: `docs/results/a8_full/phan_quyet_va_diem.txt`. Chưa có khoảng tin cậy từng cặp: dự đoán mốc `q4b_lora` đã gộp trên dev không có trong zip này.

## 5. Điều chưa kết luận được
1. Một lần train, một seed, dev 155 mẫu (12 mẫu oos). +2,12 và +1,45 nằm trong nhiễu; oos chiếm gần hết phần tăng.
2. Gộp hai thay đổi (che tên và 100 oos), không tách được phần của từng cái. Với dev chỉ có 12 mẫu oos, phần oos tăng có thể đơn thuần do 100 mẫu oos thêm vào.
3. Bản dừng ở epoch 2,886 (xem mục 1 điểm 2), nên không phải đúng công thức "3 epoch" của `q4b_lora`.
4. Chưa đo trên bộ chia A4 hay public; không có số public cho bản này.

## 6. Đề xuất (Phương quyết định)
1. **Giữ v1** cho nộp chính thức ngày 16/11. Lý do: quá giờ chốt (v2 không kịp qua Test Solution), lợi ích dev nằm trong nhiễu và chủ yếu từ oos, ask vẫn tụt, bản train dừng sớm.
2. Cổng nộp chính thức chỉ chấm lượt cuối. **Không nên** dùng lượt hai cho v2 chưa qua Test Solution: nếu image lỗi thì bài chấm cuối lỗi.
3. Không xóa notebook và output Kaggle (`checkpoints/q4b_a8_merged`, khoảng 8 GB) đến hết cuộc thi, phòng khi Phương muốn dùng; sha256 `model.safetensors` ở mục 7.

## 7. File và sha256
| File | Nội dung | Trong git? |
|---|---|---|
| `docs/results/a8_full/phan_quyet_va_diem.txt` | Phán quyết F-6 nguyên văn và hai bảng điểm | có |
| `docs/results/a8_full/dong_gop_theo_loai.txt` | Phân rã +2,12 theo loại mẫu | có |
| `docs/results/a8_full/thoi_gian.txt`, `dung_du_lieu.txt`, `preflight.txt` | Thời gian, dựng dữ liệu (sha256 4 file), preflight | có |
| `docs/results/a8_full/loss_q4b_a8_full.txt` | 241 dòng loss và `train_runtime` | có |
| `docs/results/a8_full/sha256_adapter_va_model_gop.txt`, `torch_cuda.txt`, `a8_full_results.zip.sha256` | sha256 và môi trường | có |
| `a8_full_results.zip` (70 KB) | Gồm `dev_a8_merged_clean.jsonl`, `dev_a8_merged_phowl_ft.jsonl` (dự đoán dev), log đầy đủ (kể cả `a8f_train.log`) | **không**, gửi qua Drive riêng |

- sha256 zip: `a114b20a31d70c55d50313a5c89e6ed3c2a6a7c9a63284cfceb5d7f543669cf1` (15 file trong zip khớp `MANIFEST_sha256.txt`, 0 lỗi).
- sha256 adapter `q4b_a8_full/adapter_model.safetensors`: `0da49d47270fed3058ca6d8ef322ddb2ad7764020149103877e4a1a39fb4fb09`.
- sha256 `q4b_a8_merged/model.safetensors`: `d863ed5f834199c9a3698d55d93b42627b26f8ea7fbef803920024d550d12b93` (4,0225B tham số, bfloat16).

## 8. Dòng nhật ký đề xuất cho `docs/EXPERIMENTS.md`
| Ngày | Commit | Việc | Điểm | Ghi chú |
|---|---|---|---|---|
| 2026-11-15 | `d5f376c` (mã), notebook `a8-full` (F-3 sửa cho L4 x4) | **A8 đầy đủ** `q4b_a8_full`: toàn bộ train (1390 + bản ASR) + 100 oos + 1390 bản che tên, 4 × L4, `--max-hours 8.5` | dev sạch **68,94** (mốc 66,82), dev ASR **58,43** (mốc 56,98); ask 73,08 | Đạt F-6 nhưng **dừng ở epoch 2,886** (chạm giờ) nên bỏ theo luật; +1,94 trong +2,12 đến từ oos (12 mẫu). Không dựng image: hết giờ Test Solution. Giữ v1. |
