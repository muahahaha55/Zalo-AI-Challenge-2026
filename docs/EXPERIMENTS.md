# Nhật ký thí nghiệm và tổng hợp tiến độ

Cập nhật 2026-11-06. Mọi điểm là **điểm dev nội bộ** (155 mẫu giữ riêng từ train, chia theo nhóm câu gần giống, seed 42), tính bằng bộ chấm tự cài (`scripts/eval.py`). **Chưa có điểm leaderboard**, chưa nộp lần nào. Dev là app đã thấy trong train, còn public có 65% mẫu chứa app lạ, nên điểm public nhiều khả năng thấp hơn.

## 1. Tóm tắt nhanh (đọc phần này trước)
**Pipeline hiện tại:** ASR PhoWhisper-large (đã fine-tune LoRA) → Qwen3-4B (SFT LoRA) → vòng lặp từng bước: sinh 1 tool call → `validate` → `call` → đưa kết quả lại → … → `END`. Tổng tham số 5,566B nếu adapter đã gộp vào Qwen3-4B (4,0225B + PhoWhisper 1,5433B), **5,698B nếu adapter LoRA (0,1321B) chạy kèm lúc suy luận**; cả hai < 6B (đếm từ header file, xem `docs/MODELS.md`).

| Kịch bản đo (cùng adapter `q4b_lora`) | Điểm dev | exec | ask | oos |
|---|---|---|---|---|
| Zero-shot Qwen3-4B (không SFT), text sạch | **7,35** | 8,45 | 1,92 | 8,33 |
| SFT + ASR gốc `phowl` (WER dev 25,4%) | **47,75** | 44,03 | 63,46 | 50,00 |
| SFT + ASR đã fine-tune `phowl_ft` (WER dev 9,8%) | **56,33** | 57,13 | 63,33 | 33,33 |
| SFT + transcript sạch (trần của LLM hiện tại) | **65,69** | 64,24 | 75,64 | 58,33 |
| Tham chiếu: nộp rỗng / gold | 7,18 / 100 | | | |

Đọc kết quả:
1. **SFT là nguồn điểm duy nhất**: zero-shot chỉ ngang nộp rỗng (7,35 so với 7,18).
2. **Fine-tune ASR đáng giá**: WER dev 25,43% → 9,77%, điểm dev +8,6 (47,75 → 56,33). Bản nộp nên dùng `phowl_ft`.
3. **LLM vẫn là điểm nghẽn chính**: ngay với transcript hoàn hảo chỉ đạt 65,69, tức mất 34 điểm. Cải thiện tiếp phải nhắm vào SFT (dữ liệu, quá khớp, loại mẫu ask/oos), không phải ASR.
4. **Thời gian không phải vấn đề**: mean 1,65 s, p95 3,17 s, max 4,36 s mỗi mẫu (HF generate, 1 GPU RTX PRO 6000 Blackwell, chưa gồm ASR ~0,06 s/file). Giới hạn là 30 s. Chưa đo trên A30.
5. Dev chỉ có 155 mẫu (mỗi mẫu 0,65 điểm) và oos chỉ 12 mẫu (mỗi mẫu 8,3% của loại đó): chênh lệch nhỏ, nhất là ở oos, chưa đủ để kết luận.

**Quan sát cần chú ý:** loss train của SFT rơi từ 0,92 xuống 0,004 trong khi dev chỉ 47,75–65,69, cùng với việc model thay giá trị đúng bằng giá trị quen trong train ("hồ Tây" → "Hồ Gươm"), là dấu hiệu quá khớp.

## 2. Đã làm gì, ở đâu (tái lập)
| Bước | Kết quả | Cách chạy |
|---|---|---|
| Bộ chấm, SFT, agent, ASR (Step 1–4 của PLAN) | xong, có test | `scripts/eval.py`, `build_sft.py`, `agent.py`, `run_asr.py` |
| Chạy trên Kaggle, GPU riêng tư (RULES 1.3) | xong | `docs/KAGGLE.md`, `notebooks/`, `scripts/kaggle_pipeline.sh` (stage `asr_base`, `sft_asr_data`, `train`, `asr_ft`, `sft_ft_data`, `eval_dev`) |
| Fine-tune ASR | LoRA r=32, 3 epoch, 843 bước, 21 phút; dev loại khỏi train ASR | stage `train`; adapter gộp `checkpoints/phowl_ft/merged` |
| SFT Qwen3-4B | LoRA r=64, 3 epoch, 1659 bước, batch 16, max-len 5120, **2,5 giờ**; dữ liệu = `sft_train` (sạch) + `sft_train_asr` (ASR `phowl` gốc) | stage `train`; adapter `checkpoints/q4b_lora` (528 MB) |
| Phân tích lỗi (Step 10) | có script | `scripts/analyze_dev.py` |
| Luật giờ/ngày/số điện thoại (Step 7, một phần) | có, hiệu quả nhỏ (mục 4) | `src/lifemate/rules.py`, test `tests/test_rules.py`, `validate_rules.py`, `replay_rules.py` |
| Augmentation (oos tổng hợp + biến thể) | **đã sinh dữ liệu, chưa huấn luyện** | `build_sft.py --aug-oos 100 --aug-variants 1`; `RUN=q4b_aug AUG=1 EPOCHS_SFT=2 bash scripts/kaggle_pipeline.sh train` |

Checksum sha256 (từ gói bàn giao `SHA256SUMS.txt`): adapter `q4b_lora/adapter_model.safetensors` = `25e86a6bcad0336c02253a396249160c72ca548297e94a20646b49dcc910d868`; ASR đã gộp `phowl_ft_merged/model.safetensors` = `cffba381278e485f69e14888536e610ab415727dc1bde031b17097a0eff64d2c`. Checksum model gốc và licence ở `docs/MODELS.md`.

## 3. Điểm mất ở đâu (`scripts/analyze_dev.py`, mỗi mẫu một nguyên nhân chính)
| Nguyên nhân chính | ASR gốc (mất 52,3đ) | ASR fine-tune (mất 43,7đ) | Sạch (mất 34,3đ) |
|---|---|---|---|
| exec: sai tham số (hash lệch) | 21,9đ (34 mẫu) | 18,1đ (28 mẫu) | 13,6đ (21 mẫu) |
| exec: thừa/thiếu call (các biến thể) | 17,1đ | 13,0đ | 9,9đ |
| oos nhưng vẫn gọi tool | 3,9đ (6/12 mẫu) | 5,2đ (8/12 mẫu) | 3,2đ (5/12 mẫu) |
| ask nhưng đi thực thi / exec nhưng hỏi lại | 6,8đ (8 + 4 mẫu) | 5,8đ (7 + 2 mẫu) | 7,4đ (6 + 6 mẫu) |

- Tham số sai nhiều nhất là **văn bản tự do** (nội dung tin nhắn/nhắc, tên, địa điểm, từ khóa) và **tham số thừa/thiếu** (ví dụ `contacts.search` gọi không có `query`); sau đó là ID lấy từ kết quả đọc, giờ/ngày, số điện thoại.
- **Cả 6 mẫu oos sai ở lần đo ASR gốc đều là app cần thiết không được cài** (chưa kiểm tra tay các mẫu oos sai ở hai kịch bản kia), model chọn tool "gần giống" (không có app đổi tiền → gọi `subscriptions.cancel`, không có app đi chợ → gọi `package_tracking`…). Đây là lý do thêm dữ liệu oos tổng hợp.
- Request có điều kiện ("nếu… thì gọi…") làm model gọi thừa call.
- Mẫu càng nhiều call gold càng dễ sai (1 call: 77%, 2–3 call: ~49–50% với `phowl_ft`).

## 4. Luật xác định: đo lại và đính chính
Luật `rules.py` chỉ chạm giờ/ngày tương đối → ISO và số điện thoại đọc thành chữ số, và rất thận trọng (chỉ sửa khi câu có đúng một mốc giờ, không có từ dịch giờ tương đối, giá trị model lệch ≤25 giờ). Trên nhãn train luật sửa nhầm 7/198 trường hợp (3,5%), phone 13/13 đúng.

**Đính chính:** lần đo đầu cho +3,71 điểm là **lạc quan quá**, vì luật được áp trên transcript sạch trong khi model đã đọc ASR nhiễu. Đo lại đúng (`replay_rules.py --asr`, cùng transcript model đã thấy):

| Kịch bản | Không luật | Có luật | Chênh |
|---|---|---|---|
| ASR gốc | 47,75 | 47,75 | **+0,00** (1 mẫu tốt hơn, 1 mẫu tệ hơn) |
| ASR fine-tune | 56,33 | 56,98 | **+0,65** |
| Sạch | 65,69 | 66,82 | +1,13 |

Lý do: ASR làm hỏng chính cụm giờ ("8 giờ kém 15" → "tám giờ **trẻ** mười lăm"), nên luật không chạy hoặc chạy sai; và khi text sạch thì model đã tự đúng phần lớn. Luật vẫn bật mặc định (rủi ro thấp, lợi ích nhỏ); đừng kỳ vọng nó giải quyết lỗi tham số. Không nên đầu tư thêm vào luật giờ trước khi giải quyết điểm nghẽn ở SFT.

## 5. Việc tiếp theo và các điểm chưa biết
**Đang chờ quyết định/chạy:**
1. Huấn luyện lại với augmentation (oos tổng hợp 100 mẫu + biến thể xáo thứ tự/bỏ app) 2 epoch, rồi so oos/ask/exec với `q4b_lora`. Nên lưu checkpoint theo epoch (`SAVE_LIMIT=4`) để biết quá khớp từ epoch nào.
2. **Nộp public lần đầu** để hiệu chuẩn bộ chấm và đo khoảng cách dev ↔ public (app lạ). Cần người phụ trách đồng ý (RULES 5.1); đề xuất đợi bản huấn luyện lại.
3. Một "dev app lạ" (giấu vài app khỏi SFT) để đo năng lực khái quát hóa thật; hiện chưa có.

**Chưa làm so với PLAN:** cổng ask/oos bảo thủ, constrained decoding, dữ liệu tổng hợp cho 11 app mới của public, kiểm tra thời gian trên A30, Docker, tìm cách đo lệch định dạng chữ hoa/dấu câu giữa ASR (chữ thường) và biến thể "sạch".

**Lưu ý:** `sft_train_asr.jsonl` trên Kaggle hiện là biến thể `phowl_ft` (bị ghi đè sau stage `sft_ft_data`), trong khi `q4b_lora` được huấn luyện bằng bản `phowl` gốc. Mã chưa commit (cột commit ghi `c662dbe+`).

## 6. Nhật ký chi tiết
Mỗi dòng: ngày, commit, cấu hình, điểm dev, leaderboard, ghi chú. Seed 42 ở mọi lần chạy. Thí nghiệm trên Kaggle dùng 1 GPU RTX PRO 6000 Blackwell 98 GB, transformers 5.20.2.

| Ngày | Commit | Thí nghiệm | Điểm dev | Leaderboard | Ghi chú |
|---|---|---|---|---|---|
| 2026-11-04 | 85027c2 | Metric self-test: gold làm submission | 100,00 | — | `scripts/eval.py --selftest` |
| 2026-11-04 | 85027c2 | Submission rỗng | 7,18 | — | = tỉ lệ mẫu ngoài phạm vi |
| 2026-11-05 | 0073e13 | Agent loop + oracle backend (train) | 100,00 | — | harness đúng; sau khi cho phép lặp call ghi giống nhau |
| 2026-11-05 | 0073e13 | PhoWhisper-large greedy, WER train | WER 26,18% (chuẩn hóa số) | — | yêu cầu cuối 25,19%; dev 25,41% (chưa chuẩn hóa số) |
| 2026-11-05 | 0073e13 | Qwen3-1.7B zero-shot, clean text, 30 mẫu dev | 0,00 | — | HF trên RTX 4050; 25,8 s/mẫu; chỉ kiểm tra đường chạy |
| 2026-11-05 | c662dbe+ | Dựng SFT (`build_sft.py --tokenizer Qwen/Qwen3-4B`) | — | — | train=1390 dev=155 (141 nhóm dev). Mẫu mean 1941, p95 3210, max 4521 token; 4420 mẫu bước, 8,4M token/epoch (bản sạch). Đủ cho `--max-len 5120` |
| 2026-11-06 | c662dbe+ | Fine-tune ASR: PhoWhisper-large LoRA r=32, 3 epoch (843 bước), dev loại khỏi train ASR | — | — | **WER dev 25,43% → 9,77%** (CER 14,83% → 6,14%); final 25,68% → 10,65%, history 25,24% → 9,11% (n=548 câu). ASR lại train 5036 wav 306 s, public_test 1933 wav 127 s. Dev cùng phân phối với train nên mức giảm trên test thật có thể nhỏ hơn |
| 2026-11-06 | c662dbe+ | SFT Qwen3-4B LoRA r=64, 3 epoch (1659 bước, batch 16, max-len 5120), `sft_train` + `sft_train_asr` (phowl) | — | — | 2,5 giờ; loss 0,92 → 0,004. Adapter 528 MB |
| 2026-11-06 | c662dbe+ | Dev: adapter q4b_lora + ASR gốc `phowl` | **47,75** | — | exec 44,03, ask 63,46, oos 50,00; theo số call gold: 0→50,0, 1→64,4, 2→40,4, 3→32,5, 4→41,1, 5→64,3 |
| 2026-11-06 | c662dbe+ | Dev: q4b_lora + ASR fine-tune `phowl_ft` | **56,33** | — | exec 57,13, ask 63,33, oos 33,33; 1 call 77,0 / 2 call 48,9 / 3 call 50,3; mean 1,65 s, p95 3,17 s, max 4,36 s mỗi mẫu |
| 2026-11-06 | c662dbe+ | Dev: q4b_lora + transcript sạch (trần LLM) | **65,69** | — | exec 64,24, ask 75,64, oos 58,33; mean 1,56 s, p95 2,96 s, max 4,08 s |
| 2026-11-06 | c662dbe+ | Baseline zero-shot Qwen3-4B (không adapter), text sạch | **7,35** | — | oos 8,33, exec 8,45, ask 1,92; mean 1,68 s, p95 3,46 s, max 5,68 s |
| 2026-11-06 | c662dbe+ | Luật `rules.py` áp lên dự đoán dev (`replay_rules.py`) | ASR gốc +0,00; `phowl_ft` +0,65; sạch +1,13 | — | xem mục 4; số +3,71 báo trước đó là **sai do dùng transcript sạch** |
| 2026-11-06 | c662dbe+ | `build_sft.py --aug-oos 100 --aug-variants 1` → `sft_train_aug.jsonl` (1490 mẫu) | chưa đo | — | 100 oos tổng hợp (bỏ mọi app gold dùng, target `END`) + 1390 biến thể xáo thứ tự/bỏ app không liên quan; chỉ từ phần train, đã kiểm tra không lẫn dev. Cần huấn luyện lại |
| 2026-11-06 | c662dbe+ | **Nộp public lần 1**: `q4b_lora` + ASR `phowl_ft`, luật `rules.py` bật (`outputs/sub_public_v1.jsonl`) | dev tương ứng (luật bật) 56,98 | **48,56** | 634/634 hash phát lại khớp, 0 call bị từ chối. Chênh −8,4 so với dev có luật (56,98; số 56,33 cũ đo khi chưa bật luật; dev 155 mẫu). Phân bố: oos 5,0% (train 7,2%), ask 12,3% (14,0%), exec 82,6%; calls/mẫu 2,35. Mẫu có app lạ: 410/634 (65%), 282 mẫu đã gọi tool của app lạ (464/1488 call, đều hợp lệ). Trên mẫu có app lạ: oos 3,7%, ask 9,8% so với 7,6% và 17,0% ở mẫu chỉ app quen. Không có đáp án public nên chưa tách được nguyên nhân |
| 2026-11-07 | `e04264b` | **A1** `q4b_hist`: SFT LoRA r=64, 3 epoch (1953 bước), `sft_train`+`sft_train_asr` (`phowl`)+700 mẫu từ `history`, `SKIP_ASR=1`; so với `q4b_lora` chạy lại cùng phiên (luật bật) | 57,81 (ASR `phowl_ft`) và 69,90 (sạch) so với 56,98 và 66,82 | — | So từng cặp: +0,83 [−4,74; +6,35] (17/15/123) và +3,08 [−2,59; +9,09] (20/14/121): **không phân biệt được với 0**. Toàn bộ mức tăng nằm ở oos (4/12→7/12 ASR, 7/12→10/12 sạch): ở ASR là `0978, 0982, 1513`, ở sạch là `1316, 1513, 1572` (khác nhau, chỉ `1513` chung), 0 mẫu tệ đi; ngoài oos thì exec −1,06/+2,08 và ask −1,79/−2,56. Từ chối nhầm mới: 1/143 mỗi kịch bản (`1548` ở ASR, `1164` ở sạch). 2 giờ 57 phút, 5,47 s/bước. Adapter sha256 `a5d8879ec16f1f298ba0c612f4f525f330eaa4646b615ee2bd85e507cab36d9a`. Kết luận: giữ `q4b_lora` làm mốc nộp cho đến khi có A4 (app lạ) |

**Hiệu chỉnh mốc (07/11):** hai điểm dev 56,33 và 65,69 ở các dòng trên đo **khi chưa bật luật `rules.py`** (xem mục 4); với luật bật, `q4b_lora` đạt 56,98 và 66,82 (bản chạy lại trên Kaggle cho đúng các số này; khác file cũ ở 3 mẫu ASR và 2 mẫu sạch, đều là chỉnh sửa kiểu luật). Bài nộp public 48,56 đã bật luật nên mốc dev tương ứng là 56,98.

## 7. Mâu thuẫn / chưa chắc với bộ chấm của BTC
- Cách tách token cho token-F1 (đang dùng khoảng trắng; lệch 0,73% quyết định ở phép thử độ nhạy).
- So khớp `params` của `sys.ask_missing_info` (đang so tuyệt đối kể cả thứ tự).
- Cần hiệu chuẩn lại sau lần nộp public đầu tiên.
