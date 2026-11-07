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

## Đếm tham số của hệ thống sẽ nộp (hiệu chỉnh 2026-11-07)

Đếm từ header file `.safetensors` (`python -I -X utf8 scripts/count_params.py --llm <thư mục> --adapter <thư mục> --asr <thư mục>`; không cần nạp model).

| Thành phần | Tham số | Ghi chú |
|---|---|---|
| Qwen3-4B (cơ sở) | 4.022.468.096 | từ header của bản trên Hugging Face (máy này không giữ bản cơ sở) |
| Adapter LoRA r=64 (`q4b_lora`) | 132.120.576 | đếm trên file thật; cộng thêm nếu adapter được áp dụng lúc chạy |
| PhoWhisper-large đã fine-tune (đã gộp) | 1.543.304.960 | đếm trên file thật |
| **LLM tổng (cơ sở + adapter)** | **4.154.588.672** (4,1546B) | giới hạn < 5B |
| **Cả hệ thống, adapter chạy kèm** | **5.697.893.632** (5,6979B) | giới hạn < 6B, còn dư 0,3021B |
| Cả hệ thống, adapter đã gộp vào cơ sở | 5.565.773.056 (5,5658B) | con số 5,566B ghi ở các nơi khác là trường hợp này |

Cả hai cách đều dưới giới hạn. Nếu thêm mô hình thứ ba (bộ phân loại), dư địa là khoảng 0,30B khi adapter chạy kèm, và 0,43B khi đã gộp.
Gộp adapter vào cơ sở giảm số tham số và bước tính, nhưng làm tròn bf16 khi gộp có thể làm vài đầu ra khác với bản đã đo, nên nếu gộp thì phải đo lại dev.

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
