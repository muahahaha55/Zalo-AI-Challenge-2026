# Chiến lược ra bài nộp public đầu tiên (nhanh, không cản việc train tiếp)

Lập 2026-11-06. Mục tiêu: có `submission.jsonl` cho public_test **hôm nay**, dùng đúng những gì đã có (`q4b_lora` + ASR `phowl_ft`), không chờ bản huấn luyện lại. Lần nộp này để **hiệu chuẩn bộ chấm và đo khoảng cách dev ↔ public (65% mẫu public có app lạ)**, không phải để lấy điểm cao. Mỗi lần nộp leaderboard cần người phụ trách đồng ý (RULES 5.1).

## Vì sao chia việc như dưới đây
- Điểm nghẽn là GPU: Qwen3-4B bf16 cần khoảng 8 GB cho trọng số, máy local RTX 4050 6 GB **không chạy được**. Lượng tính thì nhỏ: 634 mẫu × ~1,65 s ≈ **17 phút trên 1 GPU** (đo trên RTX PRO 6000; chưa đo ở GPU khác), ASR đã chạy sẵn.
- Dữ liệu, ASR `phowl_ft` cho public_test (file `data/processed/asr_phowl_ft_public_test.jsonl`, stage `asr_ft` đã xong) và adapter đều **đã nằm trên Kaggle** của bạn kia. Chạy ở đó thì không phải chuyển gì ngoài file kết quả nhỏ (634 dòng).
- Mọi bước còn lại (kiểm tra bài nộp) chạy được ở máy local, CPU.

## Phân việc

| # | Ai | Việc | Ước lượng | Ra gì |
|---|---|---|---|---|
| A | Bạn kia (Kaggle, trước khi bấm train mới) | chạy **một lệnh** suy luận public_test (bên dưới) | ~17 phút (1 GPU) hoặc ~9 phút (2 GPU) | `outputs/sub_public_v1.jsonl` |
| B | Bạn kia | gửi file đó (nhỏ, qua kênh riêng của đội) rồi **tiếp tục train** | vài phút | file gửi đi |
| C | Máy local | `verify_submission.py` + `zaic_tools --validate` | < 3 phút | báo OK/FAIL |
| D | Người phụ trách | quyết định nộp, tải file lên cổng | — | điểm leaderboard |
| E | Máy local | ghi điểm vào `docs/EXPERIMENTS.md`, so với dev | 10 phút | số hiệu chuẩn |

Bạn kia chỉ phải làm A và B. Cả hai không đổi code, không đổi checkpoint, nên không ảnh hưởng lần train tiếp theo, miễn là A chạy **trước** khi GPU bị chiếm cho lần train mới (hoặc chạy ở phiên Kaggle thứ hai nếu còn hạn mức).

## A. Lệnh cho bạn kia (Kaggle, thư mục gốc repo, đã `source kaggle_env.sh`)
```bash
# kiểm tra đủ đồ (phải thấy cả 3)
ls data/processed/asr_phowl_ft_public_test.jsonl checkpoints/q4b_lora/adapter_model.safetensors
wc -l data/raw/train+public_test/public_test/public_test.jsonl        # kỳ vọng 634

# chạy suy luận trên mọi GPU rồi gộp theo đúng thứ tự mẫu (tự báo lỗi nếu thiếu id)
bash scripts/infer_multi_gpu.sh public_test outputs/sub_public_v1.jsonl \
  --backend hf --model $QWEN --adapter checkpoints/q4b_lora --text asr:phowl_ft

wc -l outputs/sub_public_v1.jsonl                                       # kỳ vọng 634
sha256sum outputs/sub_public_v1.jsonl                                   # gửi kèm file
```
Cấu hình này giống hệt lần đo dev 56,33 (adapter `q4b_lora`, ASR `phowl_ft`, luật `rules.py` mặc định bật). **Đừng** thêm `--no-rules`, `--ids`, `--limit`. Nếu một shard lỗi, xem `outputs/sub_public_v1.part*.log`.

Hai lưu ý cho A:
- Phải dùng đúng `--adapter checkpoints/q4b_lora` (adapter đã train bằng ASR gốc, đã đo). Không dùng checkpoint của lần train mới nếu nó đã ghi đè.
- File ASR `phowl_ft` cho public_test phải tồn tại và đủ 1.933 dòng (`wc -l`). Nếu không, chạy `bash scripts/kaggle_pipeline.sh asr_ft` (stage cũ sẽ bỏ qua nếu file đã có).

## C. Kiểm tra ở máy local (CPU)
Đặt file vào `outputs/sub_public_v1.jsonl`, rồi:
```bash
.venv/Scripts/python -I -X utf8 scripts/verify_submission.py --split public_test --sub outputs/sub_public_v1.jsonl
cd "data/raw/train+public_test" && PYTHONPATH=public_test python -m zaic_tools --split public_test --root . --validate ../../../outputs/sub_public_v1.jsonl
```
`verify_submission.py` phát lại mọi `tool_calls` trên `ToolEnv` mới (cách BTC chấm), đòi **634/634 hash khớp, 0 call bị từ chối, 0 tool ngoài `installed_apps`**, đủ id, đúng thứ tự. Bản thử trên `sample_submission.jsonl` của BTC báo OK (3/3 hash khớp); một bản cố tình làm hỏng bị bắt cả ba lỗi (hash sai, tool lạ, thiếu id). Chỉ khi cả hai kiểm tra qua mới đưa người phụ trách quyết định nộp.

## Ngưỡng "đúng thì mới nộp"
- Hash khớp 100% và 0 call bị từ chối. Nếu có call bị từ chối thì agent đang sinh sai schema, **sửa trước khi nộp** vì mỗi call lỗi là một call thừa.
- Tỉ lệ mẫu `tool_calls` rỗng: tham chiếu train là ~7% (ngoài phạm vi). Nếu public ra rất lệch (ví dụ 0% hoặc >25%) thì hơi đáng ngờ, nhưng chưa phải lỗi (public có app lạ nên có thể khác).
- Số call trung bình mỗi mẫu: train ~2,3. Lệch nhiều là dấu hiệu model bỏ qua bước đọc hoặc lặp call.

## E. Diễn giải điểm khi có (để lần nộp này có giá trị)
Điểm dev tham chiếu cho cấu hình này là **56,33**. Hai khả năng:
- Public thấp hơn nhiều (ví dụ < 40): đúng như dự đoán do 65% mẫu có app lạ → ưu tiên Step 6 (dữ liệu tổng hợp cho 11 app mới) và augmentation hơn là tinh chỉnh ASR.
- Public gần dev (≥ 50): model khái quát tốt hơn lo ngại → ưu tiên điểm nghẽn SFT (ask/oos, call thừa).
Ghi cả điểm leaderboard lẫn chênh lệch so với dev vào `docs/EXPERIMENTS.md`. Với public_test không có gold nên **không thể chấm lại bằng bộ chấm nội bộ**; chỉ so được tổng điểm và hành vi tổng quát (số call, tỉ lệ rỗng). Vì vậy việc hiệu chuẩn bộ chấm (cách tách token, so `params` của `ask_missing_info`) chỉ được kiểm chứng gián tiếp.

## Sau lần nộp này (song song với train của bạn kia)
1. Lần train sau (augmentation, `RUN=q4b_aug`) lưu checkpoint theo epoch (`SAVE_LIMIT=4`), để biết quá khớp từ epoch nào.
2. Dựng "dev app lạ" (giấu vài app khỏi SFT) để đo khái quát hóa thật, không phải chờ leaderboard.
3. Mỗi bản adapter mới: chạy lại A–C đúng quy trình trên, rồi so với `sub_public_v1` về số call và tỉ lệ rỗng trước khi dùng một lượt nộp (tối đa 5 lượt/ngày).
4. Kế hoạch dài vẫn theo `docs/PLAN.md` (Step 6, 8, 11, 12). Mốc cứng: Docker 16/11 23:59.

## Rủi ro đã biết
- **GPU khác nhau cho kết quả hơi khác**: HF generate greedy trên GPU/phiên bản thư viện khác có thể cho vài mẫu khác nhau. Không ảnh hưởng việc nộp hôm nay, nhưng ảnh hưởng quy tắc tái lập khi nộp Docker (RULES 2.5); cần chạy lặp ở Step 11.
- **Model nền khác nguồn**: adapter được huấn luyện trên bản Qwen3-4B lấy từ một Kaggle dataset, không từ Hugging Face trực tiếp. Checksum 3 shard ghi trong `docs/MODELS.md`; cần đối chiếu với bản chính thức trước khi nộp Docker. Việc này không chặn bài nộp public hôm nay.
- **Dữ liệu trên Kaggle**: dataset và notebook phải ở chế độ Private (RULES 1.3). Chưa xác nhận.
- Đây là một lần nộp, không phải kết quả cuối: không dùng lượt nộp để dò đáp án test (RULES 5.1).
