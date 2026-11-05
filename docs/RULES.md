# Quy tắc an toàn, bảo mật và quy trình — LifeMate (ZAC 2026)

Áp dụng cho mọi người và mọi công cụ (kể cả trợ lý AI) làm việc trong repo này. Nguồn: đề bài, `README.md` của gói dữ liệu, quy định của cuộc thi. Nơi nào quy tắc có thể kiểm tra tự động thì có ghi cách kiểm tra.
Mức: **CẤM** = vi phạm có thể bị loại hoặc gây thiệt hại không sửa được. **PHẢI** = bắt buộc làm. **HỎI TRƯỚC** = cần người phụ trách đồng ý rõ ràng trước khi làm.

## 1. Dữ liệu cuộc thi

1. **CẤM** chia sẻ, phát tán, đăng lại bộ dữ liệu (audio, jsonl, catalog, `zaic_tools`, state). Dữ liệu chỉ dành cho đội đã đăng ký.
   - `data/` luôn nằm ngoài git (`.gitignore`). Không commit mẫu dữ liệu, không dán nội dung data vào issue/PR/chat công khai.
   - Hook pre-commit chặn file nằm trong `data/` và file > 5 MB.
2. **CẤM** gán nhãn `public_test` hoặc `private_test` (thủ công, hoặc bằng dịch vụ/mô hình ngoài) để huấn luyện hay tinh chỉnh. Gồm cả pseudo-label, self-training trên transcript hay tool call dự đoán của test, và chọn siêu tham số bằng cách đọc nhãn suy ra từ test.
   - Được: chạy ASR trên audio test **để suy luận** (đúng pipeline nộp bài). Được: dùng `public_test_tools_catalog.json` (schema) để sinh dữ liệu tổng hợp.
3. Dữ liệu tổng hợp **được** sinh bằng LLM/TTS ngoài **chỉ trong giai đoạn phát triển**, và **không** dùng các dịch vụ đó lúc inference.
   - **PHẢI** ghi lại nguồn, mô hình sinh và prompt vào `docs/EXPERIMENTS.md` hoặc `docs/DATA_PROVENANCE.md`.
   - **HỎI TRƯỚC** khi gửi bất kỳ nội dung nào của dữ liệu cuộc thi (audio, transcript, state, tool call) tới dịch vụ bên thứ ba (API LLM, TTS, nhận diện giọng nói đám mây, Kaggle Dataset công khai...). Mặc định: không gửi.
4. Dữ liệu train dùng để huấn luyện thì **được**. Nhãn train, `get_state()` chỉ dùng cho train, không bao giờ dùng lúc inference.

## 2. Luật thi lúc suy luận (inference) — vi phạm có thể bị loại

1. **CẤM** đọc, sao chép, phân tích mọi file dưới `*_environment/` (kể cả `state.json`). Mọi dữ liệu app chỉ lấy qua `zaic_tools`.
   - Chặn: quyền Claude Code từ chối đọc `**/*_environment/**` (`.claude/settings.json`); `scripts/check_compliance.py` quét code inference tìm `_environment`, `state.json`.
2. Lúc inference **chỉ** dùng `ToolEnv.validate`, `call`, `get_tool_calls`, `get_state_hash`. **CẤM** `get_state`, `states`, `Executor`, `Simulator`, nhân bản/sao chép env.
   - `get_state()` chỉ được dùng ở script phát triển trên train (`eval.py`, `verify_metric.py`).
3. **PHẢI** tạo đúng **một** `ToolEnv` cho mỗi mẫu. Mọi call đã gọi (kể cả call bị từ chối) phải nằm trong submission. **CẤM** tạo lại env để thử rồi làm lại từ state ban đầu.
4. **PHẢI** chạy hoàn toàn offline, không API ngoài (Docker chạy với `--network none`; đặt `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`). **CẤM** `requests`, `urllib`, `httpx`, `socket`, SDK LLM thương mại trong code inference.
5. **PHẢI** tất định: cố định seed (Python, NumPy, PyTorch), giải mã greedy hoặc temperature 0, cho ASR và LLM.
6. **Giới hạn tài nguyên**:
   - Tổng tham số cả hệ thống **< 6B**; LLM **< 5B** và ngữ cảnh **≤ 16.384 token**. Đếm bằng code (`sum(p.numel())`), gồm embedding và mọi mô hình phụ, không tin con số trên model card.
   - **≤ 30 giây/mẫu** trên 1×A30; quá giờ hoặc lỗi thực thi = 0 điểm. Mục tiêu nội bộ: p95 < 15 giây.
7. Mô hình và dữ liệu công khai phải phát hành **trước 26/10/2026** và có licence cho phép. **PHẢI** ghi tên, phiên bản, licence và URL vào `docs/MODELS.md` khi thêm model.

## 3. Bảo mật bí mật và hạ tầng

1. **CẤM** đưa token, API key, mật khẩu, khóa SSH riêng vào git, chat, log, notebook. Lưu bí mật trong biến môi trường hoặc file `.env`/`~/.ssh` (đã nằm trong `.gitignore`).
   - Hook pre-commit quét mẫu token (Hugging Face `hf_...`, `sk-...`, AWS, khóa riêng). Nếu bí mật lỡ lộ: coi như đã mất, **thu hồi và tạo mới**, không chỉ xóa khỏi lịch sử.
2. Tải model chỉ từ nguồn tin cậy (tổ chức chính thức trên Hugging Face). **CẤM** `trust_remote_code=True` và nạp file `pickle`/`.bin` không rõ nguồn; ưu tiên `safetensors`. Ghi checksum model vào `docs/MODELS.md`.
3. Máy thuê (vast.ai, Kaggle): dùng khóa SSH riêng cho dự án, không dùng lại khóa cá nhân. Không để dữ liệu cuộc thi trên máy dùng chung hoặc image công khai. **PHẢI** hủy máy thuê và xóa dữ liệu trên đó khi xong việc (cũng tránh tốn tiền).
4. File tải về hoặc giải nén ngoài dự án là **không đáng tin**: để trong thư mục riêng, không chạy mã trong đó, chạy Python đọc chúng với `-I`.
5. **HỎI TRƯỚC** mọi khoản chi: thuê GPU, nạp tiền, dịch vụ trả phí. Đặt trần chi phí cho mỗi lần chạy trước khi bắt đầu (số giờ × giá/giờ).

## 4. Quy trình mã nguồn (git)

1. **CẤM** `git push --force`, `git reset --hard` lên `main`, sửa lịch sử đã push. **CẤM** commit: dữ liệu, trọng số (`*.safetensors`, `*.bin`, `*.pt`, `*.gguf`), checkpoint, log lớn, `.venv`, bí mật. Trọng số lưu ngoài git (Drive/Hub riêng) kèm checksum.
2. **HỎI TRƯỚC** khi `git push`, mở PR, hoặc làm bất kỳ việc nào công khai hóa nội dung ra ngoài máy. Chỉ commit khi được yêu cầu. Commit nhỏ, thông điệp nêu rõ thay đổi; không dùng `--no-verify` để bỏ qua hook.
3. Trước khi commit: `python scripts/check_compliance.py` và `pytest tests` phải qua (hook tự chạy). Cài hook: `git config core.hooksPath .githooks`.
4. Mỗi thay đổi ảnh hưởng điểm số phải có số liệu trước/sau trên dev, ghi vào `docs/EXPERIMENTS.md` (cấu hình, seed, commit, điểm dev, điểm leaderboard nếu có).

## 5. Quy trình nộp bài

1. **Lần nộp leaderboard** (tối đa 5/ngày): **HỎI TRƯỚC**. Trước khi nộp: `python -m zaic_tools --split public_test --root . --validate submission.jsonl` qua, đủ số dòng (một dòng/mẫu), UTF-8 JSONL. Không dùng lượt nộp để dò đáp án (dò nhãn test bằng leaderboard là gán nhãn gián tiếp, cấm).
2. **Nộp Docker** (hạn 16/11 23:59, tối đa 2 lần, chỉ lần cuối được chấm): **HỎI TRƯỚC**. Checklist bắt buộc, tick đủ mới nộp:
   - [ ] Đếm tham số: tổng < 6B, LLM < 5B.
   - [ ] p95 thời gian/mẫu đo trên A30 (hoặc GPU tương đương) nằm trong 30 giây, có dự phòng.
   - [ ] Chạy 3–5 lần liên tiếp cho submission giống hệt từng byte.
   - [ ] Chạy với `--network none`, không tải gì lúc chạy.
   - [ ] `check_compliance.py` qua trên toàn bộ code trong image.
   - [ ] Trọng số có sẵn trong image, licence đã ghi trong `docs/MODELS.md`.
   - [ ] Kèm mã huấn luyện, mã inference, dữ liệu dùng để huấn luyện (theo đề), checksum và link Drive công khai.
3. **Đóng băng**: sau khi nộp Docker, không sửa code/trọng số của bản nộp. Vòng private chạy đúng image đã đóng băng.
4. Vòng private (20/11 09:00–23/11 23:59): chỉ chạy image đã đóng băng; không tinh chỉnh bằng private_test.

## 6. Quy tắc riêng cho trợ lý AI (Claude Code) làm việc trong repo

1. Đọc dữ liệu chỉ qua các file được phép (`*.jsonl`, catalog, audio, `audio_text` của train). Không mở `*_environment/` (đã bị chặn bằng quyền).
2. Không tự làm các việc **HỎI TRƯỚC** ở trên: push, nộp bài, chi tiền, gửi dữ liệu ra ngoài, tạo máy thuê, xóa dữ liệu hay model.
3. Không dán bí mật vào đầu ra. Gặp token trong file, báo cho người dùng thay vì sao chép.
4. Báo cáo trung thực: nếu test hoặc kiểm tra thất bại, nói rõ kèm kết quả; nếu một bước bị bỏ qua, nói rõ.
5. Khi bộ chấm nội bộ, `zaic_tools` hoặc đề bài mâu thuẫn nhau: bản của BTC là chuẩn; ghi lại điểm mâu thuẫn trong `docs/EXPERIMENTS.md` và hỏi người dùng.

## 7. Công cụ thực thi các quy tắc

| Quy tắc | Cơ chế |
|---|---|
| Không đọc `*_environment/` | `.claude/settings.json` (từ chối đọc) + `scripts/check_compliance.py` |
| Chỉ dùng API env cho phép, một `ToolEnv`/mẫu, không mạng, không `trust_remote_code` | `scripts/check_compliance.py` (quét mã inference) |
| Không commit data/trọng số/bí mật/file lớn | `.githooks/pre-commit` + `.gitignore` |
| Không `git push --force` | `.claude/settings.json` (từ chối lệnh) |
| Đếm tham số, thời gian, tất định, offline | Checklist mục 5.2, thực hiện ở Step 11 trong `docs/PLAN.md` |

Giới hạn của việc thực thi: chặn quyền chỉ bắt được các đường đọc phổ biến (công cụ Read, một số lệnh shell), không thay thế được kỷ luật. Quy tắc 2.1 và 2.2 chủ yếu dựa vào việc mọi người tự giữ và vào kiểm tra tĩnh ở trên.
