# Đóng gói Docker (bản nháp, chưa xác nhận với ban tổ chức)

Trạng thái 2026-11-07 (cuối ngày). **Chưa có hướng dẫn đóng gói của ban tổ chức** (đã gửi email hỏi). Mọi thứ dưới đây dựa trên đề bài: BTC mount thư mục private_test (cấu trúc như public_test) vào `/data`, chạy script suy luận, lấy kết quả ở `result/submission.jsonl`, container chạy offline. Khi có tài liệu chính thức, đối chiếu và sửa.

## Đã kiểm chứng và chưa kiểm chứng

| Phần | Trạng thái |
|---|---|
| Lõi chạy trong container (`src/lifemate/container.py`, `scripts/run_container.py`) | **Đã chạy thật bằng Python** trên máy này với `public_test`: cả hai bố cục `/data`, kết quả hợp lệ 634/634 hash, ghi vào `result/submission.jsonl` từ một thư mục làm việc khác, chạy tiếp sau lỗi, chặn mạng ở mức socket |
| ASR sống + LLM với mạng bị chặn | **Đã chạy** với model nhỏ (PhoWhisper-small và Qwen3-0.6B), 3 mẫu: 0 lần cố kết nối mạng |
| Image `Dockerfile.lite` | **Đã dựng và chạy thật (07/11)** trên Docker Desktop (Linux/amd64, 178 MB): `--network none`, `public_test` mount **chỉ đọc**; bố cục phẳng và lồng cho file **trùng từng byte**; hai lần chạy trùng từng byte; 634/634 hash khớp khi phát lại bằng `zaic_tools` trên máy Windows (hash giống nhau giữa Linux và Windows); `--split` bị thiếu thì báo lỗi rõ; `/data` rỗng thoát mã 1 và không ghi file |
| Ảnh GPU (`docker/Dockerfile`) | **Đã dựng và chạy thật (07/11) với model nhỏ** (Qwen3-0.6B, PhoWhisper-small, không adapter): `docker build` thành công (13,4 GB), `--gpus all` thấy RTX 4050, chạy `--network none` với dữ liệu chỉ đọc, nhận dạng giọng nói sống + LLM trên GPU, 3 mẫu hợp lệ, trung bình 3,79 giây/mẫu. **Chưa chạy với Qwen3-4B, PhoWhisper-large, adapter thật**; chưa đo trên A30 |
| Phiên bản thư viện trong image | **Đã ghim theo môi trường train và đã dựng lại (09/11)**: trong container `torch 2.11.0+cu126` (CUDA 12.6), `transformers 5.20.2`, `peft 0.21.2`, `accelerate 1.15.0`, `tokenizers 0.23.3`, Python 3.12. Môi trường train trên Kaggle là `torch 2.11.0+cu130`, Python 3.13, GPU RTX PRO 6000 Blackwell (`docs/results/a4/torch_cuda.txt`). **Cố ý chọn cu126 thay cho cu130**: cu130 cần driver NVIDIA từ 580 trở lên, driver máy chấm (A30) chưa biết, trong khi cu12x chạy được từ khoảng 525 (tương thích phiên bản phụ). Chạy `--gpus all --network none` 6 mẫu `public_test`, hai lần trùng từng byte (model nhỏ, không adapter). **Chưa biết** liệu kết quả có giữ nguyên giữa Blackwell (train), Ada (máy này) và Ampere (A30), hay giữa cu130 và cu126: phải đo trên GPU Ampere |
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
1. Chuẩn bị thư mục `models/` ngoài git (kích thước image: bản thử với model nhỏ là 13,4 GB; với Qwen3-4B và PhoWhisper-large sẽ khoảng 11 GB trọng số cộng khoảng 6 GB thư viện, nên cần tính chỗ chứa và đường tải lên) (đã nằm trong `.gitignore`): `models/llm` (Qwen3-4B cơ sở, safetensors + tokenizer), `models/adapter` (adapter LoRA), `models/asr` (PhoWhisper đã gộp).
2. Ghim phiên bản trong `requirements-inference.txt` theo môi trường đã train (`pip freeze` trên Kaggle).
3. `docker build -f docker/Dockerfile -t lifemate .`
4. `docker run --rm --gpus all --network none -v "<split>:/data" -v "$PWD/result:/app/result" lifemate`

Tham số của container (`--llm`, `--adapter`, `--asr`, `--time-budget`, `--no-rules`, `--resume`, `--limit`) có giá trị mặc định lấy từ biến môi trường trong Dockerfile. `--time-budget` mặc định 25 giây mỗi mẫu **gồm cả nhận dạng giọng nói**; giới hạn của đề là 30 giây.

## Danh sách kiểm tra trước khi nộp (khớp RULES 5.2)
- [ ] Kiểm tra thư mục `models/` bằng `python -I -X utf8 scripts/check_models.py` (so sha256 của Qwen3-4B, adapter và ASR đã gộp với `models.lock.json`, cấm file pickle, rồi kiểm giới hạn tham số). Chạy **trước** `docker build` cuối.
- [ ] Đếm tham số: `python -I -X utf8 scripts/count_params.py --llm models/llm --adapter models/adapter --asr models/asr` (hiện 5,698B với adapter chạy kèm; xem `docs/MODELS.md`).
- [ ] Offline thật: `scripts/check_offline.py` (chặn socket) chạy được với model thật đã đóng gói.
- [ ] Thời gian: p95 mỗi mẫu trên A30 (hoặc GPU tương đương) trong 30 giây, có dư địa.
- [ ] Chạy 3–5 lần liên tiếp cho file kết quả giống từng byte.
- [ ] `scripts/verify_submission.py` đạt trên kết quả chạy `public_test` trong container.
- [ ] `scripts/check_compliance.py` đạt.
- [ ] Lưu bản ghi nguồn gốc của lần chạy kiểm tra cuối: thêm `--manifest /app/result/run_manifest.json` (hoặc biến `LIFEMATE_MANIFEST`) khi **tự kiểm tra**; file ghi commit, phiên bản thư viện, sha256 trọng số, file đầu vào, sha256 kết quả và thời gian. Mặc định **tắt** trong container nộp, vì BTC thu thư mục `result/` và việc băm trọng số mất khoảng một phút. `scripts/run_infer.py` ghi `<out>.manifest.json` mặc định (`--manifest none` để tắt).
- [ ] Đối chiếu với tài liệu chính thức của BTC khi nhận được.

## Kiểm tra image thật (máy có Docker và GPU 24 GB, ví dụ RTX 4090 hoặc 3090)
Chưa từng chạy với trọng số thật; làm trước 15/11 (task A7b trong `docs/TASKS.md`).
1. **Điều kiện máy:** Docker với hỗ trợ GPU (Linux: `nvidia-container-toolkit`; Windows: Docker Desktop với WSL2), driver NVIDIA từ khoảng 525 trở lên (image dùng CUDA 12.6), khoảng 40 GB đĩa trống, Internet khi **dựng** (không cần khi chạy).
2. **Lấy mã:** `git clone https://github.com/muahahaha55/Zalo-AI-Challenge-2026.git` (bản sạch; dữ liệu cuộc thi để ở thư mục riêng, **không** nằm trong repo).
3. **Ráp `models/` trong thư mục repo** (ngoài git): `models/llm` = Qwen3-4B gốc (3 file `model-0000x-of-00003.safetensors`, `config.json`, tokenizer), `models/adapter` = `adapter_model.safetensors` + `adapter_config.json` của `q4b_lora`, `models/asr` = PhoWhisper-large đã gộp (`model.safetensors`, `config.json`, tokenizer). Rồi: `python -I -X utf8 scripts/check_models.py` phải in "models OK" (so sha256 với `models.lock.json`).
4. **Dựng:** `docker build -f docker/Dockerfile -t lifemate .` (lần đầu mất khoảng 40 phút nếu mạng chậm; kích thước khoảng 17 đến 20 GB).
5. **Chạy kiểm tra:** `python -I -X utf8 scripts/verify_image.py --image lifemate --data-root <thư mục chứa public_test> --runs 3 --out-dir check_a7b --ref <sub_public_v1.jsonl>`. Thử nhanh trước bằng `--limit 20 --runs 2`.
6. **Gửi lại:** `check_a7b/report.json` (GPU, driver, thời gian, tất định, số dòng lệch so với bài nộp tham chiếu). Không gửi `submission.jsonl` qua kênh công khai.
7. **Đạt khi:** ba lần chạy trùng từng byte, `verify_submission` báo 634/634 hash khớp, mẫu chậm nhất (`max_s`) dưới 30 giây (nên dưới 15 giây để có dư địa), số dòng lệch so với bài nộp tham chiếu nhỏ và giải thích được.
