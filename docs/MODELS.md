# Mô hình và tài nguyên bên ngoài

Mỗi mô hình thêm vào hệ thống phải có dòng ở đây (RULES.md mục 2.7). Điều kiện: phát hành trước 26/10/2026, licence cho phép, tổng tham số < 6B, LLM < 5B.
Số tham số trong cột này là số **đã đếm bằng code** (`sum(p.numel())`, `scripts/prepare_models.py`); ô "?" là chưa đếm. Licence lấy từ tag của repo trên Hugging Face (2026-11-05), cần đối chiếu lại với model card/LICENSE trước khi nộp.

| Vai trò | Mô hình | Tham số (B) | Licence | Revision (HF) | Ghi chú |
|---|---|---|---|---|---|
| ASR | `vinai/PhoWhisper-large` | 1,5433 (đã đếm) | BSD-3-Clause | `b9136a44b5f2ca664bd0b8f74baecf1715f6eeeb` (2024-11-12) | repo chỉ có `pytorch_model.bin`; đã đổi một lần sang safetensors fp32 (đã kiểm tra từng tensor giống hệt). Greedy, fp16; fine-tune bằng LoRA, gộp trước khi dùng |
| LLM | `Qwen/Qwen3-4B` | 4,0225 (đã đếm, < 5B) | Apache-2.0 | `1cfa9a720891` (2025-07-26) | safetensors chính thức; SFT bằng LoRA; thinking tắt |
| LLM (chỉ để thử đường chạy) | `Qwen/Qwen3-1.7B`, `Qwen/Qwen3-0.6B` | 1,721 / 0,636 | ? | ? | không đưa vào hệ thống nộp |
| ASR (chỉ để thử script) | `vinai/PhoWhisper-small` | 0,245 | ? | ? | không đưa vào hệ thống nộp |

Tổng hai mô hình dự kiến: 1,5433 + 4,0225 = **5,566B < 6B** (còn dư địa khoảng 0,43B; không đủ chỗ cho một mô hình thứ ba lớn hơn PhoBERT-base 0,135B hay e5-small 0,12B, cần đếm lại nếu thêm classifier Step 8).

## Checksum (sha256) các file trọng số gửi lên Kaggle
Nguồn: `kaggle_upload/lifemate-models/manifest.json` (sinh bởi `scripts/prepare_models.py`).

`qwen3-4b`
- `model-00001-of-00003.safetensors` 328a91d3122359d5547f9d79521205bc0a46e1f79a792dfe650e99fc2d651223
- `model-00002-of-00003.safetensors` 6cd087b316306a68c562436b5492edbcf6e16c6dba3a1308279caa5a58e21ca5
- `model-00003-of-00003.safetensors` e4bf436957184f4eeb86a80e9db394503f1f56446b2e6b7edeac5b81470f4ca1

`phowhisper-large` (bản đã đổi sang safetensors, không phải file gốc của repo)
- `model-00001-of-00002.safetensors` 4855e8b43394e0378a0fbc85945f7539f9ce0196ef568548d44b3555bd24d59c
- `model-00002-of-00002.safetensors` f06ccdf86a8b2260267a1bb5b1db17855979e220d85688377ab7eefb84017de2

TODO: đối chiếu LICENSE trên model card PhoWhisper (BSD-3-Clause theo tag), và xác nhận điều khoản của đề cho phép dùng mô hình này.
