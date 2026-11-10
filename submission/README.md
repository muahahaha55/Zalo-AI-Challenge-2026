# LifeMate: giải pháp (Zalo AI Challenge 2026)

Agent nghe yêu cầu bằng giọng nói tiếng Việt, rồi gọi tool từng bước trong môi trường giả lập cho đến khi xong việc.

## Tổng quan
```
audio (wav) --> ASR: PhoWhisper-large đã fine-tune --> văn bản
văn bản + ngữ cảnh + lịch sử + danh sách tool đã cài --> LLM: Qwen3-4B đã fine-tune (LoRA)
   vòng lặp: LLM sinh ĐÚNG MỘT tool call dạng JSON -> env.validate -> env.call -> kết quả đưa lại cho LLM -> ... -> END
kết quả: env.get_tool_calls(), env.get_state_hash(), thời gian mỗi mẫu
```

## Mô hình
| Thành phần | Model | Tham số | Ghi chú |
|---|---|---|---|
| ASR | `vinai/PhoWhisper-large`, fine-tune LoRA r=32 trên audio train (rồi gộp) | 1,543B | WER dev 25,4% -> 9,8% |
| LLM | `Qwen/Qwen3-4B`, fine-tune LoRA r=64 (alpha 128, dropout 0,05, mọi lớp chiếu), rồi gộp vào trọng số | 4,0225B | ngữ cảnh dùng tối đa khoảng 6K token (prompt dài nhất khoảng 4,4K), dưới giới hạn 16K |
| **Tổng** | | **5,566B** | dưới 6B; LLM dưới 5B |

Không dùng API ngoài; mọi model nằm trong `/code/saved_models`; suy luận offline.

## Suy luận (`predict.py`)
- **Prompt:** hệ thống ngắn mô tả giao thức (mỗi lượt đúng một dòng: một tool call JSON hoặc `END`), rồi thời gian hiện tại, vị trí, hồ sơ người dùng, danh sách tool đã cài (rút gọn schema), lịch sử hội thoại (văn bản ASR của các lượt trước, call và kết quả), và yêu cầu.
- **Vòng lặp agent** (`src/lifemate/agent.py`): giải mã greedy (tất định), tối đa 8 bước; call sai schema được báo lỗi lại cho LLM và cho sửa 1 lần; đọc lặp y hệt thì dừng; `sys.ask_missing_info` là bước cuối; mỗi mẫu đúng một `ToolEnv`.
- **Hậu xử lý tất định** (`src/lifemate/rules.py`): tính lại giờ/ngày và số điện thoại nói bằng chữ từ câu ASR khi không mơ hồ.
- **Thời gian:** ngân sách 25 giây mỗi mẫu (gồm cả ASR), giới hạn cứng mỗi bước sinh; trung bình khoảng 1,7 giây/mẫu trên GPU 96 GB khi thử.
- Chỉ dùng `validate`, `call`, `get_tool_calls`, `get_state_hash`; không đọc `*_environment/`.

## Dữ liệu huấn luyện
- Dữ liệu train của BTC (1.545 mẫu; 155 mẫu giữ làm dev, chia theo nhóm câu gần giống nhau để không rò).
- Văn bản đầu vào lúc train: câu sạch và bản chép của ASR gốc (nhiễu, để model quen lỗi nhận dạng).
- Mục tiêu: chuỗi call gold từng bước (mỗi lượt một call, rồi `END`).
- Chi tiết tái tạo ở `training_code/README.md`.

## Tái lập
Seed cố định (train 42, suy luận 0), giải mã greedy. Phiên bản thư viện ghim theo môi trường đã train (`requirements.txt`).
