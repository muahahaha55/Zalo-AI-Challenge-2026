# Kế hoạch hành động — LifeMate (Zalo AI Challenge 2026)

Ngày lập: 2026-11-04. Mốc cứng: **16/11 23:59** nộp Docker; **20/11 09:00** phát private_test; **23/11 23:59** nộp kết quả private.
Còn 12 ngày đến hạn Docker, nên mỗi step bên dưới có ngày dự kiến, đầu ra cụ thể và tiêu chí "xong".

## Quyết định đã chốt (2026-11-04)

1. **Agent sinh từng bước** (mỗi bước thấy kết quả của bước trước), không sinh cả danh sách call một lần. Lý do: ID và số tiền chỉ biết sau khi đọc; call chỉ bị từ chối khi sai schema, còn chọn sai ID thì tool vẫn chạy và hash lệch.
2. **Fine-tune ASR nằm trong kế hoạch chính** (Step 9 nâng lên bắt buộc, làm sớm từ 8/11, chạy song song với SFT LLM), không còn là phương án dự phòng.
3. Ưu tiên giao hàng an toàn trước (baseline → SFT → Docker chạy được), rồi mới thêm dữ liệu tổng hợp và ensemble ASR.

## Tóm tắt chiến lược (một đoạn)

Pipeline: **ASR → LLM (Qwen3-4B, SFT) → vòng lặp từng bước: sinh 1 tool call → `env.validate` → `env.call` → đưa kết quả lại → … → kết thúc**. Luật cứng xử lý số, giờ, địa chỉ, khớp tên. Điểm = 0 nếu hash sai và mọi call thừa đều trừ precision, nên ưu tiên **ít call, đúng call**.

Hai phát hiện từ data quyết định thứ tự việc:
1. 65% mẫu public_test cài ít nhất một app **không có trong train** (11 app, 34 tool mới) → model phải học "đọc schema rồi gọi", không học thuộc tên tool.
2. Train đã có transcript (`audio_text`) → có thể huấn luyện LLM trên text sạch + text nhiễu ASR, tách việc ASR và LLM thành hai bài toán độc lập.

Phân công môi trường: máy local (RTX 4050 6 GB) để viết code, chấm điểm, chạy ASR nhỏ. **Train, benchmark thời gian, test Docker trên vast.ai** (nên thuê A30 hoặc 4090/A100 để đo gần đúng phần cứng chấm). Kaggle dùng thêm cho các job ngắn.

---

## Step 1 — Bộ chấm nội bộ (4/11–5/11, local)

**Việc làm**
- Viết `src/lifemate/metric.py`: cài lại đúng công thức của đề.
  - Mẫu gold rỗng: 1 điểm khi và chỉ khi submit rỗng.
  - Mẫu còn lại: 0 nếu `state_hash` khác hoặc M=0; ngược lại F1 từ P=M/(M+U), R=M/|G|.
  - Khớp call: cùng tên, tham số không-fuzzy giống hệt, tham số fuzzy (`x-match: fuzzy`) token-F1 > 0,6 sau lowercase + NFC; mỗi call khớp tối đa một lần, không xét thứ tự.
- Tính `state_hash` gold bằng cách **chạy lại gold tool_calls trên ToolEnv của train** rồi so với hash đã chạy; đây cũng kiểm tra xem hash có ổn định không.
- `scripts/eval.py`: đọc submission + train gold, in điểm tổng và điểm theo loại mẫu (exec / ask / oos), theo số call gold, theo app.
- Cách chia: giữ 10% train làm dev; chia theo **nhóm câu nói gần giống nhau** (data có nhiều mẫu gần trùng lời, ví dụ "nếu mai mưa thì dời lịch…") để dev không bị rò rỉ.

**Xong khi:** chạy gold của chính train qua `eval.py` ra đúng 100,0; chạy submission rỗng ra đúng tỉ lệ oos (khoảng 7%).

## Step 2 — Dựng dữ liệu SFT từ train (5/11–6/11, local)

**Việc làm**
- Mỗi mẫu train khai triển thành các **bước** (mỗi bước = 1 lần sinh):
  - Đầu vào: bối cảnh (giờ, múi giờ, profile), schema nén của `installed_apps`, history (text + tool_calls/results của các lượt trước), transcript yêu cầu cuối, các call đã gọi cùng kết quả ở bước trước trong lượt này.
  - Đầu ra: call kế tiếp (JSON), hoặc `END` khi đã xong.
- Nén schema thành chữ ký ngắn: `app.tool(arg:type*, ...)` kèm mô tả ngắn, enum, `x-ref`, `x-one-of`. Đo token, mục tiêu < 3K token cho phần schema.
- Chỉ tính loss trên phần đầu ra.
- Quy ước cho oos: đầu ra đầu tiên là `END` (list rỗng). Cho ask: `sys.ask_missing_info` là call cuối.
- Định dạng lưu: `data/processed/sft_train.jsonl` và `sft_dev.jsonl` (không commit).

**Xong khi:** có tập SFT, thống kê độ dài token (p95 < 8K), và kiểm tra lại: ghép các bước của một mẫu → thu được đúng chuỗi gold.

## Step 3 — Chạy ASR và chọn ASR (5/11–7/11, local + cloud)

**Việc làm**
- Chạy PhoWhisper-large (1,55B, fp16 khoảng 3 GB, vừa 4050) trên **toàn bộ audio train + public_test**, kể cả `_u1.wav`, `_u2.wav` của history. Cache vào `data/processed/asr_<model>.jsonl` (greedy, `num_beams=1`, seed cố định).
- Đo WER/CER so với `audio_text` trên train; báo cáo riêng cho các nhóm: độ dài, mẫu có số tiền/giờ, tên riêng (tên liên hệ, tên thiết bị). Chưa có nhãn vùng miền, nên cân nhắc thêm một proxy (ví dụ nhóm theo từ vựng) hoặc chấp nhận WER chung.
- Thử thêm `PhoWhisper-medium` và `Whisper-large-v3-turbo`; chọn bằng WER **và** điểm cuối của pipeline (Step 6), không chỉ WER.
- Chuẩn hóa văn bản ASR (số đọc, chính tả tên/ thiết bị) viết riêng ở Step 7.

**Xong khi:** có bảng WER các ứng viên, và file transcript của train/public_test cho từng ứng viên ASR.

**Điểm quyết định (7/11):** chọn 1 ASR chính. Fine-tune ASR (Step 9) đã nằm trong đường chính; ensemble 2 ASR chỉ làm nếu có lợi trên điểm cuối.

## Step 4 — Baseline agent zero-shot và lần nộp public đầu tiên (6/11–8/11, cloud)

**Việc làm**
- Viết `src/lifemate/agent.py`: vòng lặp theo Step 2, dùng Qwen3-4B (bf16, greedy) qua vLLM trên vast.ai. Mỗi bước: sinh call → `env.validate` → nếu hợp lệ thì `env.call` và đưa kết quả lại; nếu không hợp lệ thì **không gọi** (tránh call lỗi, vì call lỗi vẫn bị tính là thừa), cho model một lần sửa, rồi thôi.
- Giới hạn số bước (tối đa khoảng 8). Một `ToolEnv` mỗi mẫu; chỉ dùng `validate`, `call`, `get_tool_calls`, `get_state_hash`; không đụng `*_environment/`.
- `scripts/run_infer.py --split public_test`: ghi `submission.jsonl`, đo thời gian mỗi mẫu.
- Dùng `python -m zaic_tools --validate submission.jsonl` trước khi nộp.
- Nộp lên cổng để lấy điểm mốc (tối đa 5 lần/ngày).

**Xong khi:** có điểm dev nội bộ + điểm leaderboard của baseline. Mọi cải tiến sau đó phải so với số này.

## Step 5 — SFT Qwen3-4B (8/11–10/11, vast.ai)

**Việc làm**
- LoRA (r=64) hoặc full fine-tune trên tập Step 2. Bắt đầu bằng LoRA vì nhanh hơn; thử full fine-tune nếu còn thời gian.
- **Chống học vẹt tên tool** (vì public có 34 tool mới): ngẫu nhiên đổi thứ tự tool/app trong prompt, thỉnh thoảng bỏ bớt app không liên quan, và (tùy chọn) đổi tên tool/tham số kiểu đồng nghĩa trong một phần nhỏ mẫu. Luôn giữ schema trong prompt.
- Huấn luyện trên **trộn** transcript sạch (`audio_text`) và transcript ASR (nhiễu thật), khoảng 50/50.
- Lưu checkpoint, log loss, đo điểm dev bằng Step 1 sau mỗi lần chạy.

**Xong khi:** điểm dev vượt rõ baseline Step 4, và điểm theo từng loại (exec/ask/oos) được báo cáo riêng.

## Step 6 — Dữ liệu tổng hợp cho phần thiếu (9/11–11/11, vast.ai/local)

Chạy song song với Step 5; dùng LLM lớn (chỉ ở giai đoạn phát triển, không dùng lúc inference) sinh thêm dữ liệu.

**Việc làm**
- **11 app mới của public** (ecommerce, event_tickets, grocery, habit_tracker, pantry, parking, public_transit, recipes, stocks, train_bus_tickets, translator): sinh câu lệnh + chuỗi call đúng theo schema trong `public_test_tools_catalog.json`. Chỉ dùng **schema** catalog; **không** dùng nội dung `public_test.jsonl` hay audio public làm nhãn (luật cấm gán nhãn public/private test).
- **Lớp khó**: ngoài phạm vi (app gần giống nhưng chưa cài, người dùng tự hủy "à thôi", câu chào hỏi, hỏi lại chuyện cũ) và thiếu thông tin (thiếu `x-one-of`, tên liên hệ trùng).
- Kiểm tra tự động mọi mẫu tổng hợp: chạy các call qua `ToolEnv`/schema; loại mẫu lỗi.
- Với mẫu cần trạng thái app, chỉ dùng được các mẫu mà ta thực thi được bằng env hợp lệ, vì ta không đọc `state.json` của public_test.

**Xong khi:** có tập tổng hợp đã lọc, huấn luyện lại, điểm dev không giảm và điểm trên nhóm "app lạ" (mô phỏng bằng cách giấu một số app train khỏi SFT) tăng.

## Step 7 — Luật xác định và hậu xử lý (9/11–11/11, local)

Làm ngay khi có baseline để biết model sai ở đâu; thêm dần, mỗi luật phải tăng điểm dev.

**Việc làm**
- Số đọc → số: "năm trăm nghìn" → 500000, "hai triệu rưỡi", "ba chục", v.v.
- Giờ/ngày tương đối → ISO theo `current_time` + `timezone`: "ba giờ chiều mai", "sáu rưỡi sáng", "thứ sáu tuần sau".
- Điền địa chỉ nhà/cơ quan từ `user_profile`.
- Khớp tên liên hệ/thiết bị sau khi đọc dữ liệu qua tool (fuzzy match transcript với tên trả về từ `contacts.search`, `smart_home.get_status`).
- Ràng buộc `x-ref`: sau lần đọc, chỉ cho phép ID có trong kết quả.
- Gỡ call trùng; không bao giờ phát call đọc thừa.

**Xong khi:** mỗi luật có test nhỏ trong `tests/` và đóng góp điểm dev được ghi lại trong log thử nghiệm.

## Step 8 — Giải mã ràng buộc và cổng 3 lớp (10/11–12/11, vast.ai)

**Việc làm**
- Constrained decoding (xgrammar/outlines) theo JSON schema của tool trong `installed_apps`; sau lần đọc thì thu hẹp `x-ref` vào các ID hợp lệ.
- Quyết định lớp execute / ask_missing / oos: ban đầu dùng chính LLM (END đầu tiên = oos). Nếu lớp oos hoặc ask sai nhiều, thêm classifier nhỏ (PhoBERT-base 0,135B hoặc e5-small 0,12B) làm cổng, ngưỡng **bảo thủ** cho oos, vì gọi thừa một tool là 0 điểm.
- Đảm bảo `sys.ask_missing_info` luôn là call cuối và tham số `tool_name`, `params` đúng định dạng.

**Xong khi:** tỉ lệ call bị schema-reject = 0 trên dev; điểm oos và ask đạt hoặc vượt Step 5.

## Step 9 — Fine-tune ASR (8/11–12/11, vast.ai, song song với Step 5)

Đây là bước chính thức, không còn tùy chọn. Vẫn đo trước: so điểm dev khi dùng transcript sạch `audio_text` với transcript ASR, để biết trần cải thiện và quyết định bỏ bao nhiêu công vào đây.

- Dữ liệu: audio train (1.545 mẫu + audio các lượt user trong history) với `audio_text`. Giữ 10% làm dev ASR, chia cùng nhóm với dev của LLM để không rò rỉ.
- Không dùng audio public_test/private_test để train hay để tạo nhãn (luật cấm).
- Fine-tune PhoWhisper bằng LoRA (hoặc full nếu đủ VRAM), seed cố định; đánh giá bằng WER/CER và điểm cuối của pipeline.
- Thêm prompt biasing bằng từ khóa các app đang cài và tên thực thể.
- Ensemble 2 ASR (tổng < 6B cùng LLM) chỉ khi còn thời gian và có lợi trên điểm cuối.
- Cuối mỗi lần: lưu checkpoint, ghi vào `docs/EXPERIMENTS.md`.

- Fine-tune PhoWhisper bằng LoRA trên audio train (1.545 mẫu + audio history), kèm prompt biasing từ khóa các app đang cài.
- Hoặc ensemble 2 ASR (PhoWhisper-medium + Whisper-turbo, tổng 1,58B) và đưa cả hai hypothesis vào prompt LLM.
- Mỗi lựa chọn phải qua bộ chấm cuối của pipeline, không chỉ WER.

## Step 10 — Phân tích lỗi và vòng cải tiến (liên tục từ 8/11)

- Sau mỗi lần chạy dev: xuất bảng 50 mẫu điểm thấp nhất, gán nhãn nguyên nhân (ASR, chọn sai tool, sai tham số, thiếu/thừa call, sai lớp).
- Ưu tiên sửa nhóm nguyên nhân có tổng điểm mất lớn nhất.
- Ghi mọi thử nghiệm (cấu hình, điểm dev, điểm leaderboard) vào `docs/EXPERIMENTS.md`.
- Chỉ nộp leaderboard khi điểm dev tăng; giữ quota 5 lần/ngày.

## Step 11 — Kiểm tra ràng buộc kỹ thuật (12/11–13/11, vast.ai)

- **Đếm tham số bằng code** (`sum(p.numel())` cho mọi model, gồm embedding, classifier): toàn hệ thống < 6B, LLM < 5B. Bỏ model nào làm vượt.
- **Thời gian**: ≤ 30 s/mẫu trên A30 (ASR + LLM nhiều bước + env); đo p95, mục tiêu p95 < 15 s để có dư địa. Cho `max_new_tokens` và số bước tối đa hợp lý.
- **Tất định**: cố định seed (Python/NumPy/PyTorch), greedy; chạy 3–5 lần liên tiếp, so submission từng byte.
- **Offline**: không tải gì lúc chạy; đặt `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`.
- **Licence**: kiểm tra từng model và dữ liệu (model phát hành trước 26/10/2026, licence cho phép).
- **Tuân thủ**: quét code để chắc chắn không mở/đọc `*_environment/`, không dùng `get_state`, `states`, `Executor`, `Simulator`, không clone env.

## Step 12 — Đóng gói Docker và thử như người chấm (13/11–15/11)

- Image Linux có CUDA, cài `requirements-gpu-linux.txt`, **nhúng sẵn trọng số** (không tải mạng).
- Entry point: đọc split ở `/data`, ghi `result/submission.jsonl`. Chờ hướng dẫn đóng gói riêng của BTC để khớp lệnh chạy chính xác; kiểm tra khi có tài liệu.
- Kiểm thử: `docker run --gpus all --network none -v <public_test>:/data ...` trên vast.ai, so điểm với lần chạy thường.
- Giữ kích thước image hợp lý; chuẩn bị checksum và link Google Drive công khai.
- Cần kèm trong gói nộp: code huấn luyện, code inference, dữ liệu dùng để huấn luyện (theo đề).

## Step 13 — Nộp và vòng private (16/11 → 23/11)

- **16/11 trước 23:59**: nộp Docker qua form "Solution Submission" (tối đa 2 lần, chỉ lần cuối được chấm). Nộp sớm trước hạn ít nhất vài giờ.
- **20/11 09:00**: tải private_test → chạy đúng image đã đóng băng → kiểm tra bằng `--validate` → nộp `submission.jsonl` trước 23/11 23:59. Không sửa code sau khi đóng băng.
- Kết quả top 2 công bố 2/12; BTC chạy lại image để kiểm chứng (nếu khác sẽ lấy trung bình 5 lần).

---

## Lịch tổng quan

| Ngày | Việc chính |
|---|---|
| 4–5/11 | Step 1 (metric), bắt đầu Step 2, 3 |
| 6–8/11 | Xong Step 2, 3; Step 4 baseline; nộp public lần đầu |
| 8–11/11 | Step 5 (SFT LLM), Step 9 (fine-tune ASR), Step 6 (tổng hợp), Step 7 (luật) song song |
| 10–13/11 | Step 8 (decoding + cổng), hoàn tất Step 9, Step 10 liên tục |
| 12–13/11 | Step 11 (tham số, thời gian, tất định) |
| 13–15/11 | Step 12 (Docker, thử offline) |
| 16/11 | Step 13: nộp Docker (nộp sớm trong ngày) |
| 20–23/11 | Chạy private, nộp kết quả |

## Điểm quyết định / dự phòng

| Mốc | Câu hỏi | Nếu không đạt |
|---|---|---|
| 7/11 | PhoWhisper-large có WER chấp nhận được không? | Đổi sang medium/ensemble (Step 9) |
| 9/11 | Qwen3-4B SFT có vượt baseline rõ không? | Thử Qwen2.5-3B / Phi-4-mini; giảm độ dài prompt |
| 12/11 | p95 thời gian > 20 s? | Cắt số bước, giảm ASR (medium), rút gọn prompt |
| 13/11 | Tham số > 6B? | Bỏ classifier / đổi ASR nhỏ hơn |
| 14/11 | Docker chưa chạy được offline? | Dừng thêm tính năng, chỉ sửa đóng gói |

## Rủi ro chính

1. **App lạ ở public/private** → mất điểm nếu model học vẹt. Giảm bằng Step 5 (augmentation) và Step 6.
2. **Call thừa** (đặc biệt đọc thừa) giảm precision; oos mà gọi tool là 0 điểm.
3. **Hết giờ 30 s** nếu vòng lặp nhiều bước; mẫu timeout = 0 điểm.
4. **Không tái lập** (non-determinism của GPU/vLLM) → chạy nhiều lần kiểm tra, cố định seed và batch.
5. **Vi phạm luật dữ liệu** (đọc `state.json`, gán nhãn public/private) bị loại; quét code ở Step 11.
6. **Thời gian**: chỉ 12 ngày; giữ ưu tiên: Step 1 → 4 → 5 → 7/8 → 11 → 12. Step 6 là bổ sung khi còn thời gian; Step 9 đã nằm trong đường chính.
