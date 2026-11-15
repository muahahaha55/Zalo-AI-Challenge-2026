# A8 đầy đủ: thử cuối cho bài nộp (hướng dẫn cho thuan, 15/11)

**Mục tiêu:** train bản nộp có che tên app và tool trên **toàn bộ train**, gộp LoRA, đo dev. Nếu dev không tệ đi, dựng image v2 và đưa qua Test Solution **trước 23:59 hôm nay (15/11)**. **Bài hiện tại (`zac2026_ManchesterCity.tar.gz`, MD5 `6e2d32ad…`, Test Solution đạt ngày 14/11) là bài an toàn: không xóa, không ghi đè, không đổi link.** Bản mới chỉ thay khi qua đủ các cổng dưới đây; không qua thì bỏ, không tiếc.

**Kỳ vọng thực tế:** thí nghiệm A8 trên bộ chia A4 cho app lạ +2,56 (khoảng tin cậy chứa 0) và ask tụt. Lợi ích có thể chỉ +1 đến 2 điểm, hoặc không có. Đây là một lần thử, không phải cải tiến đã chứng minh.

## 0. Việc phải kịp (Test Solution đóng 23:59 ngày 15/11, mỗi ngày 1 lượt)
| Bước | Ai | Thời gian | Cổng dừng |
|---|---|---|---|
| 1. Chạy `notebooks/a8-full.ipynb` (dựng dữ liệu, train, gộp, đo dev) | thuan, Kaggle **RTX PRO 6000** | khoảng 4 đến 4,5 giờ | train chạm 5 giờ hoặc dev không đạt: **dừng, báo Phương** |
| 2. Phương xem phán quyết, quyết định | Phương | 10 phút | không đạt: giữ bài cũ |
| 3. Phương cập nhật `models.lock.json` và README, push (mục 3) | Phương | 15 phút | |
| 4. Thuê máy ảo, tải trọng số, dựng image v2, nén, tải lên Drive (mục 4) | thuan | khoảng 2 đến 2,5 giờ | |
| 5. Test Solution với link và MD5 của v2 | Phương | 10 phút | phải **đạt** mới được coi là nộp được |

**Giờ chốt cứng:** nếu đến lúc bắt đầu bước 4 mà còn dưới **2,5 giờ** đến 23:59, **bỏ A8 đầy đủ, giữ bài cũ**. Đừng cố vì sẽ nộp bản chưa qua test, mà bản cuối cùng mới được chấm.

**Nếu chỉ có L4 x4:** không chạy (train 7 đến 8 giờ, không kịp). Báo Phương ngay.

## 1. Chạy notebook (bước 1)
Notebook `notebooks/a8-full.ipynb` (đã có trên nhánh `main`, kéo về). Làm theo mục "Trước khi chạy" ở đầu notebook: Private, Internet Off, RTX PRO 6000, bốn input (nhớ **không gắn** output A4, A4b, A5, A8, merge-lora). Thứ tự: P-1 đến P-3, F-1 đến F-8. Điểm cần nhìn:
- **F-2** phải in `dev_ids.txt trùng bản trước khi dựng` và 1390 / 100 / 1390 mẫu. Không đúng thì dừng, gửi output.
- **F-3** train 3 epoch. Nếu in `CHẠM GIỚI HẠN --max-hours 5`, bản này bỏ.
- **F-6** in phán quyết. Ba điều kiện: dev sạch ≥ 66,82; dev ASR `phowl_ft` ≥ 56,98; ask (sạch) ≥ 70,64 (mốc 75,64 trừ 5). Dev 155 mẫu nhiễu khoảng ±4 điểm; "đạt" chỉ nghĩa là đủ điều kiện để Phương cân nhắc.
- **F-8** chạy cuối rồi **Save Version** (Private).

Gửi Phương: `a8_full_results.zip` (qua Drive riêng vì có dự đoán dev), **dòng phán quyết F-6 chép nguyên văn**, và sha256 của `checkpoints/q4b_a8_merged/model.safetensors`.

## 2. Phương quyết định (bước 2)
Chỉ tiếp tục nếu phán quyết ĐẠT cả ba điều kiện **và** còn đủ thời gian (giờ chốt cứng ở mục 0). Dữ liệu cuộc thi trong `a8_full_results.zip` không vào git.

## 3. Cập nhật mã trước khi dựng (Phương, bước 3)
Image phải dựng từ **một commit sạch**. Phương làm trước, rồi mới báo thuan dựng:
1. `models.lock.json`: thay sha256 của `llm/model.safetensors` bằng giá trị thuan gửi (ASR giữ nguyên `cffba381…`).
2. `submission/training_code/README.md`: bước 2 thành `python scripts/build_sft.py --asr-tag phowl --aug-oos 100 --aug-mask 1390`; bước 4 thêm hai file dữ liệu: `--data data/processed/sft_train.jsonl data/processed/sft_train_asr.jsonl data/processed/sft_train_aug.jsonl data/processed/sft_train_mask.jsonl`; bước 5 giữ nguyên. Ghi: "mẫu che tên: `lifemate.masking` thay tên app và tool bằng chuỗi ngẫu nhiên theo từng mẫu (Hammer, arXiv:2410.04587)".
3. `submission/README.md`: mục "Dữ liệu huấn luyện" thêm 1390 bản sao che tên và 100 mẫu ngoài phạm vi.
4. `python scripts/check_compliance.py`, commit, push; báo thuan hash commit.

## 4. Dựng image v2 (thuan, bước 4)
Làm đúng `docs/THUE_MAY_DUNG_IMAGE.md`, với **các khác biệt**:
- **Trọng số:** LLM từ output notebook `a8-full` (`checkpoints/q4b_a8_merged/*`) đặt vào `models/llm`; ASR vẫn là `phowl_ft/merged` (như v1, nguồn Kaggle `merge-lora` hoặc Drive). Không có `models/adapter`.
- **Mã:** clone repo, `git checkout <hash commit của Phương>` (commit có `models.lock.json` mới). **Không chạy `--update-lock`** (sẽ làm cây mã "dirty" và `COMMIT.txt` ghi dirty); chỉ chạy `python3 scripts/check_models.py`, phải in `models OK`.
- **Dựng:** `GZIP_CMD=pigz bash scripts/build_submission.sh ManchesterCity v2` (tên đội giữ nguyên `ManchesterCity`, thẻ `v2`).
- **Đặt tên tệp khi tải lên:** đổi thành `zac2026_ManchesterCity_v2.tar.gz` (và tệp `.md5` kèm theo) **trước khi tải lên Drive**, để **không ghi đè tệp v1**. Chỉ chia sẻ "Anyone with the link" cho tệp v2; tệp v1 giữ nguyên.
- Gửi Phương: link Drive v2 và **MD5**.

## 5. Test Solution (Phương, bước 5)
Dán link v2 và MD5 v2 vào ô Test Solution. **Đạt:** dev đã thỏa, Phương chọn bản nào nộp chính thức ngày 16/11 (v1 hoặc v2). **Không đạt hoặc trễ giờ:** nộp v1 như kế hoạch, bỏ v2.

## Nếu có sự cố
- Notebook lỗi hoặc dev tụt: dừng và báo Phương, không tự đổi tham số.
- Trễ giờ chốt: bỏ, chuyển sang nộp v1 sáng 16/11.
- Không xóa gì (notebook, output, máy ảo thì xóa sau khi tải lên xong) cho đến khi cuộc thi kết thúc.
