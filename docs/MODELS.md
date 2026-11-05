# Mô hình và tài nguyên bên ngoài

Mỗi mô hình thêm vào hệ thống phải có dòng ở đây (RULES.md mục 2.7). Điều kiện: phát hành trước 26/10/2026, licence cho phép, tổng tham số < 6B, LLM < 5B.
Số tham số trong cột này là số **đã đếm bằng code** (`sum(p.numel())`); ô "?" là chưa đếm.

| Vai trò | Mô hình | Tham số (B) | Licence | Nguồn | Ghi chú |
|---|---|---|---|---|---|
| ASR | `vinai/PhoWhisper-large` | 1,543 (đã đếm) | ? kiểm tra model card | Hugging Face | greedy, fp16; fine-tune bằng LoRA, gộp trước khi dùng |
| LLM (dự kiến) | `Qwen/Qwen3-4B` | ? (chưa đếm) | ? Apache-2.0 theo hiểu biết, cần xác nhận | Hugging Face | SFT bằng LoRA; thinking tắt |
| LLM (chỉ để thử đường chạy) | `Qwen/Qwen3-1.7B`, `Qwen/Qwen3-0.6B` | 1,721 / 0,636 | ? | Hugging Face | không đưa vào hệ thống nộp |
| ASR (chỉ để thử script) | `vinai/PhoWhisper-small` | 0,245 | ? | Hugging Face | không đưa vào hệ thống nộp |

TODO: điền licence thật từ model card, thêm checksum file trọng số.
