# Vòng private: chạy giải pháp đã nộp trên private_test và nộp kết quả (21 đến 23/11)

**Luật BTC:** private_test phát hành 9:00 ngày 20/11. Đội chạy **giải pháp đã nộp và đã đóng băng** trên tập này, nộp file JSONL ở mục Submission của cổng, **hạn 23:59 ngày 23/11**. BTC chạy lại Docker đã nộp để đối chiếu (máy chấm: **1 GPU A30**); nếu lần chạy lại khác kết quả đã nộp, BTC chạy Docker 5 lần và lấy điểm trung bình. Bảng private hiện theo thứ tự thời gian nộp.

**Nguyên tắc:** chạy đúng bài đã nộp, không sửa gì: mã commit `submission-v1` (`98d3199`), LLM `q4b_merged` (sha256 `652e96ae…`), ASR `phowl_ft/merged` (`cffba381…`), ASR chạy sống. Không train, không sửa luật, không chỉnh theo dữ liệu private (gán nhãn private_test bị cấm).

## Dữ liệu private (đã kiểm ở máy Phương, 21/11)
625 mẫu, đủ âm thanh, catalog `2.7.0`, không tool nào đổi schema so với train. Khó hơn public: **19/65 app không có trong train** (618/625 mẫu có cài ít nhất một app lạ), **37% mẫu có lịch sử từ 9 lượt** (public 7%), âm thanh mỗi mẫu trung vị 20 s, dài nhất 75 s (public 83 s). Kỳ vọng điểm private thấp hơn public 48,74, do điểm yếu app lạ đã đo được.

## Việc của thuan (Kaggle, khoảng 1,5 giờ)
1. Tải `private_test.zip` từ cổng (thành viên đội tải được), giải nén, tạo dataset Kaggle **Private** tên `lifemate-private` chứa thư mục `private_test/` (bỏ `__MACOSX` và `.DS_Store`). **Không** để Public, không chia sẻ cho ai ngoài đội.
2. Kéo `main`, mở `notebooks/private-run.ipynb`: Private, Internet Off, GPU **RTX PRO 6000** (không có thì L4), gắn 3 input như đầu notebook (`lifemate-code`, `lifemate-private`, output `merge-lora`).
3. Chạy P-1 đến P-3, R-1 đến R-5 theo thứ tự rồi Save Version.
   - **R-1** phải in `24/24 file trùng commit nộp 98d3199` và `TRÙNG bài nộp` cho cả llm và asr. Lệch thì dừng.
   - **R-2** thử 20 mẫu, phải `ALL CHECKS PASSED`.
   - **R-3** 625 mẫu × 2 lần: cần hai lần trùng câu trả lời, `verify_submission` khớp 625/625 hash, mẫu chậm nhất dưới 25 s.
4. Gửi Phương qua **Drive riêng**: `private_results.zip` (có `private_submission.jsonl` và `report.json`), kèm output nguyên văn của R-4. Không đưa file này lên git hay nơi công khai.

## Việc của Phương (sau khi nhận file)
1. Kiểm lại ở máy (CPU, chỉ phát lại call bằng `zaic_tools`, không đọc `private_test_environment/`):
   ```bash
   LIFEMATE_DATA_ROOT=data/raw/private .venv/Scripts/python -I -X utf8 scripts/verify_submission.py --split private_test --sub <private_submission.jsonl>
   ```
   phải báo 625/625 hash khớp, 0 call ngoài app cài.
2. Nộp `private_submission.jsonl` ở mục Submission của cổng (vòng private). Nộp sớm (bảng xếp theo thời gian nộp); không cần đợi sát hạn.
3. Chụp màn hình kết quả, ghi vào `docs/TRANG_THAI.md`.

## Lưu ý
- Kết quả trên GPU Kaggle có thể khác vài dòng so với lần BTC chạy lại trên A30 (làm tròn bf16 khác GPU, ngân sách 25 s mỗi mẫu trên GPU chậm hơn). Theo luật, khi đó BTC lấy trung bình 5 lần chạy Docker, không bị loại. Không cố "sửa" cho khớp.
- Trên public, A7a chạy cùng đường này cho kết quả trùng nhau qua 3 lần, chậm nhất 5,67 s, và bài nộp từ đường này được 48,74.
- Không xóa notebook, output Kaggle, dataset `lifemate-private` và tệp image trên Drive cho đến khi BTC công bố kết quả (2/12).
