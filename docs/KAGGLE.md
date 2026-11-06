# Chạy trên Kaggle (GPU L4 x4, Internet OFF)

Thực hiện `docs/HANDOFF.md` (yêu cầu của Baron Phương) trên Kaggle. Được phép theo `docs/RULES.md` mục 1.3 (đã đồng ý 2026-11-05) với điều kiện: **dataset riêng tư, notebook riêng tư**. Notebook không có mạng nên mọi thứ phải có sẵn dưới dạng dataset.

## 0. Quy tắc khi dùng Kaggle
- Mọi dataset và notebook phải ở chế độ **Private**; kiểm tra lại trên trang web sau khi tạo. Không "Make Public", không thêm cộng tác viên.
- Không đặt token Kaggle/HF vào notebook hay output. `kaggle.json` nằm ở `~/.kaggle/` (ngoài git).
- Khi xong đợt làm việc: xóa dataset `lifemate-raw`, `lifemate-processed` và các phiên bản notebook output chứa transcript/dự đoán. Chỉ giữ adapter/checkpoint nếu cần.
- `kaggle_upload/` nằm trong `.gitignore`.

## 1. Chuẩn bị trên máy local (đã làm xong phần lớn)
| Việc | Lệnh | Trạng thái |
|---|---|---|
| Giải nén dữ liệu | `unzip data/raw/train+public_test.zip -d data/raw` | xong |
| Venv + thư viện + kaggle CLI | `.venv/Scripts/python.exe -m pip install -r requirements.txt kaggle huggingface_hub torch` | xong |
| Dựng SFT (HANDOFF bước 1, phần không cần GPU) | `.venv/Scripts/python.exe -I -X utf8 scripts/build_sft.py --tokenizer Qwen/Qwen3-4B` | xong: train=1390 dev=155 |
| Chuẩn bị trọng số (Qwen3-4B + PhoWhisper-large đổi `.bin` sang safetensors) | `.venv/Scripts/python.exe -I -X utf8 scripts/prepare_models.py` | xem `kaggle_upload/lifemate-models/manifest.json` |
| Kiểm tra trước khi upload | `.venv/Scripts/python.exe -I -X utf8 scripts/kaggle_preflight.py` | chạy lại trước mỗi lần upload |

Thống kê SFT thật (Qwen3 tokenizer): mẫu dài nhất 4.521 token, p95 3.210; 4.420 mẫu bước, 8,4M token mỗi epoch cho bản sạch (gấp đôi khi thêm bản ASR). Vì vậy `MAXLEN` mặc định là 5120, không cần 8192.

## 2. Năm dataset riêng tư
| Dataset | Nội dung | Lệnh chuẩn bị (Git Bash) |
|---|---|---|
| `lifemate-code` | mã nguồn (không data) | `KAGGLE_USER=<tên> bash scripts/kaggle_pack.sh code` |
| `lifemate-raw` | `data/raw/train+public_test` | `... kaggle_pack.sh raw` |
| `lifemate-processed` | `data/processed` | `... kaggle_pack.sh processed` |
| `lifemate-models` | Qwen3-4B + PhoWhisper-large (safetensors) | `... kaggle_pack.sh models` |
| `lifemate-wheels` | wheel cài offline | mục 3, tạo trong notebook Kaggle |

`kaggle_pack.sh` chỉ ghi `dataset-metadata.json` rồi **in lệnh `kaggle datasets create ...` để bạn tự chạy**. `raw` và `models` cần `--dir-mode zip` (đã in sẵn), nếu không thư mục con bị bỏ qua. Mã và `processed` đổi thường xuyên: cập nhật bằng `kaggle datasets version -p <thư mục> -m "..."`. Với dataset `raw` nên tạo trước vì nặng nhất (~1,1 GB).

Ghi licence + `sha256` + số tham số từ `manifest.json` vào `docs/MODELS.md` (RULES 2.7, 3.2).

## 3. Wheelhouse (Internet ON, một lần)
Notebook mới, bật Internet, chỉ gắn `lifemate-code` (không chứa dữ liệu cuộc thi; **không gắn** raw/processed/models):
```bash
%%bash
cd "$(dirname "$(find /kaggle/input -name kaggle_build_wheelhouse.sh -print -quit)")/.." && bash scripts/kaggle_build_wheelhouse.sh
```
Lưu output thành dataset `lifemate-wheels` (Private). Torch/numpy dùng bản có sẵn của image Kaggle. vLLM và xgrammar **không** đưa vào (dễ lệch phiên bản torch); trên Kaggle dùng backend HF. Dùng đúng image Python giống notebook làm việc.

## 4. Notebook làm việc (Internet OFF, GPU L4 x4, cell `%%bash`)
Gắn 5 dataset. Cell 1:
```bash
%%bash
bash "$(find /kaggle/input -name kaggle_setup.sh -print -quit)"
```
Các cell sau bắt đầu bằng `source /kaggle/working/lifemate/kaggle_env.sh`. Chọn stage bằng `bash scripts/kaggle_pipeline.sh <stage>`:

| Stage | Tương ứng HANDOFF | Việc | Ước lượng (chưa đo) |
|---|---|---|---|
| (kiểm tra) | — | `python -I -X utf8 scripts/verify_metric.py && python -m pytest tests -q` | 1 phút |
| `SMOKE=1` mọi stage | — | chạy thử ống dẫn với dữ liệu rất nhỏ, không đụng file thật | vài phút mỗi stage |
| `asr_base` | 1: ASR PhoWhisper-large cho train + public_test, rồi WER (kỳ vọng ~26,2%) | 4 GPU song song | ~20–40 phút |
| `sft_asr_data` | 1: `build_sft.py --asr-tag phowl` | CPU | ~1 phút |
| `train` | 2.1 + 2.2: ASR LoRA trên GPU cuối, SFT LoRA trên các GPU còn lại (`--resume`, `--max-hours 10.5`) | 3 GPU SFT + 1 GPU ASR | SFT ~1,5–2 giờ/epoch, 3 epoch ~5 giờ (rất thô) |
| `asr_ft` | 2.1: ASR đã fine-tune cho cả hai split, so WER dev với bản gốc (~25,4%) | 4 GPU | ~20–40 phút |
| `sft_ft_data` | 2.1: `build_sft.py --asr-tag phowl_ft` | CPU | ~1 phút |
| `eval_dev` | 2.3: điểm dev với adapter (`TAG=phowl` hoặc `phowl_ft`) | 4 GPU | ~vài phút |

Ví dụ:
```bash
%%bash
source /kaggle/working/lifemate/kaggle_env.sh
SMOKE=1 bash scripts/kaggle_pipeline.sh asr_base     # thử trước, rồi chạy thật không có SMOKE
bash scripts/kaggle_pipeline.sh asr_base
bash scripts/kaggle_pipeline.sh sft_asr_data
```
Mọi stage ghi log vào `outputs/logs/<stage>.log` và đánh dấu `outputs/.done_<stage>`; chạy lại stage đã xong thì bỏ qua (`FORCE=1` để làm lại). Chọn biến: `EPOCHS_SFT`, `EPOCHS_ASR`, `MAXLEN`, `NGPU`, `MAX_HOURS`; `DRYRUN=1` để chỉ in lệnh.

Ghi chú:
- Hạn mức ~12 giờ/phiên và giờ GPU hằng tuần có hạn; xem trang Settings của tài khoản. **Xem tốc độ (giây/bước) trong vài phút đầu của `train` rồi tính lại số epoch**; `MAX_HOURS` bảo đảm dừng đúng lúc đã lưu checkpoint. Ước lượng ở trên dựa trên 16,8M token/epoch và 3 L4 ở ~30 TFLOPs hiệu dụng mỗi card.
- OOM: giảm `MAXLEN` hoặc thêm `--rank 32` vào `train_sft.py`.
- Nhật ký mỗi stage vào `docs/EXPERIMENTS.md` (cấu hình, seed, commit, điểm dev, thời gian).


## 4b. Vòng 2: luật + augmentation (sau khi có điểm dev vòng 1)
Code mới (cần upload lại zip `lifemate-code`): `src/lifemate/rules.py` (giờ/ngày/số điện thoại, tự chạy trong agent, tắt bằng `--no-rules`), `build_sft.py --aug-oos/--aug-variants`, stage `aug_data`, biến `RUN`, `AUG`, `SAVE_LIMIT`.
```bash
%%bash
source /kaggle/working/lifemate/kaggle_env.sh
bash scripts/kaggle_pipeline.sh aug_data                                     # sinh sft_train_aug.jsonl (CPU, ~1 phút)
RUN=q4b_aug AUG=1 EPOCHS_SFT=2 SAVE_LIMIT=4 bash scripts/kaggle_pipeline.sh train      # ~2,5 giờ trên 1 GPU lớn, ước lượng
RUN=q4b_aug TAG=phowl_ft FORCE=1 bash scripts/kaggle_pipeline.sh eval_dev    # điểm dev (đã gồm luật)
```
So sánh với q4b_lora (dev 47,75): xem `exec/ask/oos`. Phân tích lỗi và replay luật chạy ở máy local: `scripts/analyze_dev.py`, `scripts/replay_rules.py`, `scripts/validate_rules.py`.

### Qua phiên sau
Cuối mỗi phiên chạy `bash scripts/kaggle_save_state.sh`, rồi **Save Version** (notebook riêng tư). Phiên sau gắn thêm output của notebook đó làm input; `kaggle_setup.sh` tự khôi phục transcript ASR, SFT đã dựng lại, `checkpoints/` (để `--resume`) và các dấu `.done_*`. Output tối đa ~20 GB.

## 5. Gửi lại những gì (HANDOFF mục 3)
Từ `/kaggle/working`: `checkpoints/q4b_lora`, `checkpoints/phowl_ft/merged` (nếu dùng), `outputs/logs`, `outputs/dev_*.jsonl`. Chạy `sha256sum` rồi gửi qua link Drive riêng của đội. Không gửi dữ liệu, audio, `data/processed`. Việc `git push`, nộp leaderboard, nộp Docker vẫn hỏi người phụ trách trước.

## 6. Đo thời gian: L4 không thay được A30
L4 có băng thông bộ nhớ thấp hơn A30 nhiều (~300 so với ~930 GB/s) nên decode LLM trên L4 chậm hơn. Nếu p95 trên L4 < 30 giây thì trên A30 gần như chắc chắn đạt (kết quả thận trọng), nhưng số liệu cuối cho checklist Docker (RULES 5.2) vẫn cần đo lại trên A30 (vast.ai). Kaggle không có Docker nên bước `docker run --network none` cũng làm ở vast.ai.

## 7. Việc của bạn
1. Upload bằng **web** (không cần CLI, không cần `kaggle.json`): Kaggle > Datasets > New Dataset > kéo thả, chọn **Private**. `kaggle_setup.sh` tìm dataset theo file đánh dấu nên tên dataset và cách lồng thư mục không quan trọng:
   - code: `kaggle_upload/lifemate-code.zip` (tạo bằng `bash scripts/kaggle_pack.sh code`)
   - raw: file gốc `data/raw/train+public_test.zip` (Kaggle tự giải nén)
   - processed: thư mục `data/processed`
   - models: hai thư mục `qwen3-4b` và `phowhisper-large` trong `kaggle_upload/lifemate-models/` (~14 GB; nếu trình duyệt hay đứt thì dùng CLI)

   Đường CLI (tùy chọn): `~/.kaggle/kaggle.json` rồi `KAGGLE_USER=<tên> bash scripts/kaggle_pack.sh <mục>` và chạy lệnh được in ra.
2. Chạy `kaggle_preflight.py` trước khi upload; sau khi tạo, xác nhận mỗi dataset đang ở chế độ Private.
3. Tạo `lifemate-wheels` (mục 3).
4. Chạy cell setup + `SMOKE=1` từng stage, rồi các stage thật; gửi lại log nếu có lỗi.
