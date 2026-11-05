# Nhật ký thí nghiệm

Mỗi dòng: ngày, commit, cấu hình, seed, điểm dev (nội bộ), điểm leaderboard (nếu có), ghi chú.

| Ngày | Commit | Thí nghiệm | Điểm dev | Leaderboard | Ghi chú |
|---|---|---|---|---|---|
| 2026-11-04 | 85027c2 | Metric self-test: gold làm submission | 100,00 | — | `scripts/eval.py --selftest` |
| 2026-11-04 | 85027c2 | Submission rỗng | 7,18 | — | = tỉ lệ mẫu ngoài phạm vi |
| 2026-11-05 | 0073e13 | Agent loop + oracle backend (train) | 100,00 | — | harness đúng; sau khi cho phép lặp call ghi giống nhau |
| 2026-11-05 | 0073e13 | PhoWhisper-large greedy, WER train | WER 26,18% (chuẩn hóa số) | — | yêu cầu cuối 25,19%; dev 25,41% (chưa chuẩn hóa số) |
| 2026-11-05 | 0073e13 | Qwen3-1.7B zero-shot, clean text, 30 mẫu dev | 0,00 | — | HF trên RTX 4050; 25,8 s/mẫu; không đọc dữ liệu trước khi ghi |

## Mâu thuẫn / chưa chắc với bộ chấm của BTC
- Cách tách token cho token-F1 (đang dùng khoảng trắng; lệch 0,73% quyết định ở phép thử độ nhạy).
- So khớp `params` của `sys.ask_missing_info` (đang so tuyệt đối kể cả thứ tự).
- Cần hiệu chuẩn lại sau lần nộp public đầu tiên.
