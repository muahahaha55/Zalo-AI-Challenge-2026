# Đóng gói Docker (bản nháp, chưa xác nhận với ban tổ chức)

Trạng thái 2026-11-07. **Chưa có hướng dẫn đóng gói của ban tổ chức** (đã gửi email hỏi). Mọi thứ dưới đây dựa trên đề bài: BTC mount thư mục private_test (cấu trúc như public_test) vào `/data`, chạy script suy luận, lấy kết quả ở `result/submission.jsonl`, container chạy offline. Khi có tài liệu chính thức, đối chiếu và sửa.

## Đã kiểm chứng và chưa kiểm chứng

| Phần | Trạng thái |
|---|---|
| Lõi chạy trong container (`src/lifemate/container.py`, `scripts/run_container.py`) | **Đã chạy thật bằng Python** trên máy này với `public_test`: cả hai bố cục `/data`, kết quả hợp lệ 634/634 hash, ghi vào `result/submission.jsonl` từ một thư mục làm việc khác, chạy tiếp sau lỗi, chặn mạng ở mức socket |
| ASR sống + LLM với mạng bị chặn | **Đã chạy** với model nhỏ (PhoWhisper-small và Qwen3-0.6B), 3 mẫu: 0 lần cố kết nối mạng |
| `docker build` / `docker run` | **Chưa chạy**: Docker Desktop không chạy trên máy phát triển nên chưa có image nào được dựng |
| Ảnh GPU (`docker/Dockerfile`) | **Chưa dựng**, chưa chạy trên GPU nào; phiên bản thư viện chưa ghim |
| Bố cục `/data` thật của BTC | **Chưa biết**, nên mã nhận cả hai kiểu |
| Thời gian trên A30, chạy lặp giống từng byte trên GPU | **Chưa đo** |

## Bố cục `/data` được nhận
- **Phẳng:** `/data` chính là thư mục split (`/data/audio/`, `/data/<split>.jsonl`, `/data/zaic_tools/`, ...). Mã tạo một liên kết `<tạm>/<split>` trỏ vào `/data` vì `ToolEnv` mong đợi `root/<split>/...`.
- **Lồng:** `/data/<split>/...`.
- Tên split tự phát hiện (ưu tiên `private_test`). Nếu mơ hồ, mã dừng với thông báo rõ, và dùng `--split`.
- `zaic_tools` được nạp từ chính thư mục split (mỗi split mang bản riêng).

## Hai image
| Image | Mục đích |
|---|---|
| `docker/Dockerfile.lite` | kiểm tra đường ống: không GPU, không torch, không model; backend rỗng (`END` ở mọi mẫu) và không ASR. Kiểm tra bố cục, `zaic_tools`, offline, file kết quả, hash |
| `docker/Dockerfile` | bản đầy đủ: PhoWhisper đã fine-tune + Qwen3-4B + adapter LoRA |

### Kiểm thử đường ống (không cần GPU; cần bật Docker Desktop)
```bash
docker build -f docker/Dockerfile.lite -t lifemate-lite .
mkdir result
docker run --rm --network none -v "<đường dẫn thư mục public_test>:/data" -v "$PWD/result:/app/result" lifemate-lite
python -I -X utf8 scripts/verify_submission.py --split public_test --sub result/submission.jsonl      # kỳ vọng 634/634 hash khớp
```
Chạy hai lần và so sánh hai file bằng `cmp` để kiểm tra tái lập.

### Ảnh đầy đủ
1. Chuẩn bị thư mục `models/` ngoài git (đã nằm trong `.gitignore`): `models/llm` (Qwen3-4B cơ sở, safetensors + tokenizer), `models/adapter` (adapter LoRA), `models/asr` (PhoWhisper đã gộp).
2. Ghim phiên bản trong `requirements-inference.txt` theo môi trường đã train (`pip freeze` trên Kaggle).
3. `docker build -f docker/Dockerfile -t lifemate .`
4. `docker run --rm --gpus all --network none -v "<split>:/data" -v "$PWD/result:/app/result" lifemate`

Tham số của container (`--llm`, `--adapter`, `--asr`, `--time-budget`, `--no-rules`, `--resume`, `--limit`) có giá trị mặc định lấy từ biến môi trường trong Dockerfile. `--time-budget` mặc định 25 giây mỗi mẫu **gồm cả nhận dạng giọng nói**; giới hạn của đề là 30 giây.

## Danh sách kiểm tra trước khi nộp (khớp RULES 5.2)
- [ ] Đếm tham số: `python -I -X utf8 scripts/count_params.py --llm models/llm --adapter models/adapter --asr models/asr` (hiện 5,698B với adapter chạy kèm; xem `docs/MODELS.md`).
- [ ] Offline thật: `scripts/check_offline.py` (chặn socket) chạy được với model thật đã đóng gói.
- [ ] Thời gian: p95 mỗi mẫu trên A30 (hoặc GPU tương đương) trong 30 giây, có dư địa.
- [ ] Chạy 3–5 lần liên tiếp cho file kết quả giống từng byte.
- [ ] `scripts/verify_submission.py` đạt trên kết quả chạy `public_test` trong container.
- [ ] `scripts/check_compliance.py` đạt.
- [ ] Đối chiếu với tài liệu chính thức của BTC khi nhận được.
