# Giao việc vòng tới (từ 07/11)

Đọc `docs/RULES.md` trước. Mốc cứng: nộp Docker **16/11 23:59** (nên nộp sáng 16/11). Mỗi việc dưới đây ghi rõ ai làm, lệnh chạy, và gửi lại gì. Mọi số điểm so với **dev của cấu hình hiện tại: 56,33 (ASR `phowl_ft`) và 65,69 (transcript sạch)**, đều ghi vào `docs/EXPERIMENTS.md`.

## 0. Lưu ý bắt buộc sau lượt A1 (07/11), đọc trước khi chạy tiếp

1. **Notebook và output trên Kaggle phải ở chế độ Private** (RULES 1.3). Phát hiện ngày 07/11: notebook `train-hist` đang **công khai**, và output của nó chứa dẫn xuất của dữ liệu cuộc thi (bản chép lời ASR của audio train và public_test, các file `sft_*.jsonl`, `dev_ids.txt`, `gold_hash_train.json`). **Chủ notebook cần đặt Private ngay** (Settings, Sharing), kiểm tra mọi notebook và dataset khác có gắn dữ liệu, và gỡ các phiên bản đã lưu nếu cần. Sau này mỗi notebook mới phải đặt Private **trước khi** chạy cell đầu tiên.
2. **Stage `train` và `eval_dev` cần `FORCE=1`** khi các dấu `outputs/.done_*` được khôi phục từ phiên trước (không có thì stage báo "already done" và không chạy gì). Log nay được **nối thêm** (`tee -a`), không còn ghi đè.
3. **Muốn chọn epoch thì đặt `SAVE_STEPS` bằng số bước mỗi epoch** (ví dụ khoảng 650 cho lượt có 10.404 bước mỗi epoch) cùng `SAVE_LIMIT` đủ lớn (6 trở lên). Mỗi checkpoint khoảng 1,5 GB và output Kaggle giới hạn 20 GB (lượt A1 đã dùng khoảng 14 GB), nên xóa các checkpoint cũ trước.
4. **`pip freeze` đã nhận (08/11, `docs/env/kaggle_freeze_2026-11-08.txt`)** và Docker đã ghim theo đó (transformers 5.20.2, peft 0.21.2, accelerate 1.15.0). **Còn thiếu bản torch CUDA:** file lấy trong phiên chỉ có CPU (`torch 2.11.0+cpu`). Trong phiên **có GPU**, chạy `python -c "import torch,sys;print(torch.__version__, torch.version.cuda, sys.version)"` và gửi dòng in ra.
5. **Mã trên Kaggle phải là bản mới nhất** (từ commit `783526d` trở đi): thành viên chạy GPU tự đóng gói và tải lên (`bash scripts/kaggle_pack.sh code`, rồi New Version của dataset mã, giữ **Private**) và gắn phiên bản mới vào notebook trước khi chạy A4, A5, A6. Thiếu bước này thì ``callgen` và các sửa pipeline không có.
6. **Thứ tự và hạn:** A4 (hạn 10/11) trước, rồi **A4b** (10 phút) rồi A5 chỉ khi A4b cho thấy khoảng cách app lạ so với app đã thấy đủ lớn. **A6 (ASRX) đã rút lại**, xem cuối mục A. Nếu đến 10/11 A4 chưa chạy được thì bỏ A4 và A5. Hướng dẫn chạy từng bước gửi riêng.

## A. Việc cho bạn có GPU (Kaggle)

### A1. Train lại LLM với dữ liệu lịch sử (ưu tiên số 1, ~2,5 giờ)
Đổi **đúng một thứ** so với `q4b_lora` (thêm 700 mẫu lấy từ các lượt trong `history`), để so sánh sạch. Lý do dựng lại biến thể ASR bằng `phowl`: `q4b_lora` đã train bằng biến thể ASR gốc, còn file `sft_train_asr.jsonl` hiện trên Kaggle đã bị ghi đè bằng bản `phowl_ft`.

```bash
git pull origin main                       # cần bản có src/lifemate/augment.py và cờ HIST/SKIP_ASR
source kaggle_env.sh                       # nạp $QWEN, $PHO như mọi lần

# 1) dựng lại dữ liệu: ASR gốc (đúng như lần train đầu) + 700 mẫu lịch sử
python -I -X utf8 scripts/build_sft.py --asr-tag phowl --hist-max 700
wc -l data/processed/sft_train.jsonl data/processed/sft_train_asr.jsonl data/processed/sft_train_hist.jsonl
#   kỳ vọng: 1390 / 1390 / 700. Nếu khác, dừng và báo.

# 2) train chỉ LLM (SKIP_ASR=1: KHÔNG train lại ASR, KHÔNG ghi đè checkpoints/phowl_ft)
SKIP_ASR=1 HIST=1 RUN=q4b_hist EPOCHS_SFT=3 SAVE_LIMIT=4 bash scripts/kaggle_pipeline.sh train
```
`SAVE_LIMIT=4` giữ checkpoint theo bước để chọn epoch. **Không dùng `AUG=1` ở lượt này** (tránh lẫn hai thay đổi). Nếu còn GPU rảnh, chạy riêng một lượt `AUG=1 HIST=1 RUN=q4b_hist_aug` (cần chạy stage `aug_data` trước, nó dựng lại file `sft_train_aug.jsonl`).

Chấm dev (cả hai kịch bản) rồi gửi lại:
```bash
TAG=phowl_ft RUN=q4b_hist bash scripts/kaggle_pipeline.sh eval_dev          # so với 56,33
bash scripts/infer_multi_gpu.sh train outputs/dev_q4b_hist_clean.jsonl --ids data/processed/dev_ids.txt \
     --backend hf --model $QWEN --adapter checkpoints/q4b_hist --text clean
python -I -X utf8 scripts/eval.py --sub outputs/dev_q4b_hist_clean.jsonl --ids data/processed/dev_ids.txt   # so với 65,69
```
**Gửi lại:** 2 file `outputs/dev_q4b_hist_*.jsonl`, số điểm (tổng, exec/ask/oos), thời gian mỗi mẫu, đường cong loss, `sha256sum` adapter. Tôi sẽ so từng cặp mẫu với `q4b_lora` (dev chỉ 155 mẫu, sai số khoảng ±4 điểm nên không chỉ nhìn điểm tổng).

### A2. Đo WER của `phowl_ft` trên chính audio train (~5 phút, kiểm tra một giả thuyết)
Giả thuyết: ASR đã fine-tune trên audio train nên gần như không nhiễu khi chạy lại trên train, vì vậy biến thể ASR của SFT dựng từ `phowl_ft` không giống nhiễu thật.
```bash
python -I -X utf8 scripts/asr_wer.py --tag phowl_ft            # toàn bộ train
python -I -X utf8 scripts/asr_wer.py --tag phowl_ft --dev-only # đã biết: 9,77%
```
Gửi cả hai con số. Nếu WER trên train **thấp hơn nhiều** dev (ví dụ dưới 3%), giả thuyết đúng và ta sẽ làm ASR out-of-fold.

### A1b. Thêm dữ liệu "lời rút lại" (làm sau A1, so với A1)
Người nói tự sửa lời ("à khoan", "à thôi", "khỏi") chiếm 6,3% train, trong đó 31% là mẫu ngoài phạm vi. Trên dev các mẫu này chỉ đạt 40,6 điểm so với 67,6 của phần còn lại (chỉ 11 mẫu nên chưa chắc), model hay làm luôn phần đã bị rút lại. Dữ liệu gồm: các mẫu rút lại **thật** nhân bản 2 lần (nhãn thật, an toàn), cộng vài chục câu ghép "câu ngắn, à thôi, yêu cầu thật" và "câu ngắn, à thôi khỏi" (câu bị rút lại luôn là câu đơn, ngắn, không chứa sẵn dấu hiệu rút lại hay phủ định, để phạm vi "bỏ đi" không mơ hồ).
```bash
bash scripts/kaggle_pipeline.sh aug_data      # FORCE=1 nếu đã chạy; sinh thêm sft_train_retract.jsonl (khoảng 254 mẫu)
SKIP_ASR=1 HIST=1 RETRACT=1 RUN=q4b_retract EPOCHS_SFT=3 SAVE_LIMIT=4 bash scripts/kaggle_pipeline.sh train
```
Chỉ giữ nếu dev tăng so với A1 và mẫu có lời rút lại không tệ đi (xem `scripts/compare_runs.py`). Để chỉ dùng mẫu thật, đặt `RETRACT_REPLACE=0 RETRACT_CANCEL=0` khi chạy `aug_data`.

### A3. Kiểm tra A30 hoặc GPU tương đương (nếu có)
Chỉ cần chạy `scripts/run_infer.py` trên 50 mẫu dev và ghi mean/p95/max thời gian mỗi mẫu. Kaggle p95 hiện khoảng 3 giây; vẫn chưa có số liệu trên phần cứng chấm (1×A30).

### A4. Thử "app lạ" (sau A1; đo khái quát hóa thật, ~2 giờ vì ít mẫu hơn)
Hiện dev chỉ có app đã thấy, nên điểm dev lạc quan so với public (65% mẫu public có app lạ). Bộ chia này giấu 6 app (`flights, bike_rental, package_tracking, shopping_list, podcast_radio, loyalty`) khỏi mọi chỗ trong dữ liệu train, kể cả khỏi danh sách app trong prompt, rồi đo trên các mẫu dùng chúng.
```bash
python -I -X utf8 scripts/build_sft.py --asr-tag phowl --hold-apps flights,bike_rental,package_tracking,shopping_list,podcast_radio,loyalty --hist-max 300
#   ghi vào data/processed/holdout/ (không đụng file chính). Kỳ vọng: ~1180 mẫu train, 174 mẫu đánh giá
python -I -X utf8 scripts/train_sft.py --model $QWEN --data data/processed/holdout/sft_train.jsonl data/processed/holdout/sft_train_asr.jsonl \
     --out checkpoints/q4b_holdout --epochs 3 --max-len 5120 --global-batch 16 --save-limit 2
bash scripts/infer_multi_gpu.sh train outputs/unseen_q4b_holdout.jsonl --ids data/processed/holdout/eval_ids.txt \
     --backend hf --model $QWEN --adapter checkpoints/q4b_holdout --text clean
python -I -X utf8 scripts/eval.py --sub outputs/unseen_q4b_holdout.jsonl --ids data/processed/holdout/eval_ids.txt
```
Lệnh trên dùng `--asr-tag phowl` (bản ghi của ASR gốc) còn `q4b_lora` dùng `phowl_ft`, nên **không so điểm A4 với điểm dev của `q4b_lora`**; chỉ so các cấu hình A4 với nhau (`q4b_holdout` và `q4b_holdout_hist`, cái thứ hai thêm `sft_train_hist.jsonl` vào `--data`).
Gửi điểm (tổng, exec/ask/oos) và file đầu ra. Đây là mốc "chưa thấy app" để so các ý sau (che tên, ghép app). Mỗi lần so thêm một ý thì train lại với cùng bộ chia này.

**Lưu ý quan trọng sau khi kéo bản mới:** bản cũ của bước sinh biến thể (`--aug-variants`) có lỗi: với mẫu `ask`, nó coi app mà câu hỏi nhắc tới là "không liên quan" và có thể xóa app đó khỏi danh sách đã cài (161/217 mẫu `ask` bị ảnh hưởng, khoảng 30% biến thể của chúng bị hỏng). Đã sửa. **File `sft_train_aug.jsonl` đã dựng trước đó trên Kaggle cần dựng lại** (`bash scripts/kaggle_pipeline.sh aug_data` với `FORCE=1`) trước khi dùng `AUG=1`.

### A5. Dữ liệu mới: ghép đoạn call thật vào môi trường khác (sau A4, so trên bộ chia A4)
Dữ liệu này có nhãn đúng ngay từ khi dựng. Các bước:
1. Lấy một đoạn call liên tiếp của cùng một app trong một mẫu train.
2. Chạy lại đoạn đó trong môi trường của một mẫu train khác có cài app ấy:
   - ID được lấy lại từ kết quả đọc của chính môi trường mới;
   - ngày được dời theo `current_time` của môi trường mới;
   - giá trị enum có thể được đổi.
3. Mọi call phải qua `validate` và `call`. Chỉ cần một call bị từ chối là bỏ cả mẫu.
4. Câu nói do **model chạy trên máy** viết. Thử Qwen3-8B và Qwen3.5-9B (song song), rồi 14B, rồi 32B (32B bf16 vừa card 96 GB) và dừng ở model nhỏ nhất cho câu chấp nhận được (chưa đo ngưỡng; 1,7B không dùng được). Model này chỉ tạo dữ liệu nên giới hạn 5B/6B của bài nộp không áp dụng. Thêm vào notebook dưới dạng Kaggle Model hoặc dataset **Private**. Không gửi dữ liệu sang bất kỳ API bên ngoài nào. Prompt đưa 3 câu thật của **app khác** làm mẫu cách nói.
5. Lọc qua 3 lớp:
   - luật đơn giản: bỏ câu lộ ID hoặc tên tool, câu kể theo ngôi thứ ba, câu thiếu nội dung chữ của tham số, câu quá ngắn hoặc quá dài;
   - chính model viết tự trả lời CÓ/KHÔNG;
   - **khứ hồi (quan trọng nhất)**: một model khác đọc câu, chạy agent trong môi trường đó, và chỉ giữ mẫu nếu ra cùng trạng thái và gọi cùng các tool với nhãn.

Kiểm tra trên máy local (08/11, không có GPU lớn):
- **Phần call:** 2000 mẫu, với 40 app và 117 tool. Phát lại khớp 2000/2000 kết quả, và agent chạy với "oracle" (phát lại nhãn) đạt 2000/2000. Có 23% mẫu mang chuỗi tool chưa từng có trong train, và 95% mẫu kết thúc bằng thao tác ghi.
- **Phần chữ, thử với Qwen3-1.7B:** **câu sai gần hết** (chép câu mẫu hoặc tả sai thao tác). Model tự chấm vẫn trả lời CÓ cho cả 24/24 câu, luật đơn giản chỉ loại được 7/24, còn bước khứ hồi loại được cả 5/5 câu sai đã thử. Kết luận: **model nhỏ không dùng được**. Phải đọc mẫu đầu ra của model 14B trở lên trước khi train. Luật "thiếu nội dung tham số" cũng loại nhầm 24% câu thật, nên bỏ đi một số mẫu tốt.
```bash
python -I -X utf8 scripts/gen_calls.py --n 6000                     # CPU, vài phút -> data/processed/callgen_pending.jsonl
python -I -X utf8 scripts/gen_text.py --model /kaggle/input/<qwen3-14b> --inp data/processed/callgen_pending.jsonl \
       --out data/processed/callgen_text.jsonl --batch 64              # GPU; chạy lại thì tự bỏ qua dòng đã có
# DỪNG Ở ĐÂY lần đầu: chạy với --n 200, gửi 30 dòng ngẫu nhiên của callgen_text.jsonl (trường audio_text + label.tool_calls) để duyệt
python -I -X utf8 scripts/verify_callgen.py --model /kaggle/input/<qwen3-14b> --inp data/processed/callgen_text.jsonl \
       --out data/processed/callgen_checked.jsonl                      # GPU, vài giây mỗi mẫu; dùng model CHƯA tinh chỉnh
# A4 có và không có dữ liệu mới (cùng bộ chia, chỉ khác file train):
python -I -X utf8 scripts/build_sft.py --asr-tag phowl --hold-apps flights,bike_rental,package_tracking,shopping_list,podcast_radio,loyalty \
       --hist-max 300 --callgen data/processed/callgen_checked.jsonl   # thêm holdout/sft_train_cg.jsonl, đã bỏ mẫu dùng app bị giấu
```
Train như A4 nhưng thêm `data/processed/holdout/sft_train_cg.jsonl` vào `--data` (đặt tên `q4b_holdout_cg`), rồi so với `q4b_holdout` trên `holdout/eval_ids.txt` bằng `scripts/compare_runs.py`. Chỉ dùng cho bản nộp nếu điểm trên app lạ tăng rõ. Với bộ chia A4, nhiều mẫu bị bỏ vì mẫu gốc nằm ngoài phần train của bộ chia đó (ở thử nghiệm local là 183/300), nên cần sinh dư (`--n 6000`).

### A7. Chạy toàn bộ giải pháp với trọng số THẬT (không phụ thuộc A5; làm trước 15/11)
Mục đích: biết giải pháp cuối chạy đúng ở dạng nộp (ASR sống cộng LLM, offline, ghi `result/`), mẫu chậm nhất dưới 30 giây, kết quả tất định, và hash phát lại khớp. Hiện mới thử bằng model nhỏ. Công cụ: `scripts/verify_image.py` (chạy N lần, so byte, đo thời gian, gọi `verify_submission.py`, so với bài nộp tham chiếu, ghi `report.json`).
**A7a: trên Kaggle (RTX PRO 6000, không có Docker; chạy cùng đường mã `run_container.py`)**:
```bash
python -I -X utf8 scripts/verify_image.py --no-docker --data-root data/raw/train+public_test --split public_test --runs 3     --out-dir check_a7a --ref outputs/sub_public_v1.jsonl -- --llm $QWEN --adapter checkpoints/q4b_lora --asr checkpoints/phowl_ft/merged
```
Gửi `check_a7a/report.json` (không chứa nội dung dữ liệu). Kỳ vọng: 634/634 hash khớp, ba lần trùng từng byte, và **trùng 634/634 dòng với `sub_public_v1.jsonl`** (nếu lệch thì ghi số dòng lệch: do ASR sống trong lần này thay vì bản chép sẵn, hoặc do luật; báo lại để phân tích).
**A7b: trên máy có Docker và GPU 24 GB (RTX 4090 hoặc 3090)**: xem `docs/DOCKER.md` mục "Kiểm tra image thật". Lệnh chính:
```bash
python -I -X utf8 scripts/check_models.py --models models        # trước khi dựng
docker build -f docker/Dockerfile -t lifemate .
python -I -X utf8 scripts/verify_image.py --image lifemate --data-root <thư mục chứa public_test> --runs 3 --out-dir check_a7b --ref sub_public_v1.jsonl
```
Làm A7b trên 4090 rồi, nếu có thể, trên 3090 (cùng đời Ampere với A30 máy chấm); so `report.json` của hai máy và số dòng lệch nhau.

### A5, kết quả duyệt lần thử đầu (09/11) và bản sửa: LÀM LẠI bước thử 200 mẫu
Đọc 30 dòng của 9B (8B tương tự) cho thấy **lỗi nằm ở bộ sinh call, không phải ở model viết câu**:
- **Id lấy ngẫu nhiên không xét loại:** trong 49 call `smart_home` của lần thử, **29 call (59%) có `device_id` sai loại** (`light_set` trỏ vào điều hòa `ac_bedroom`, `climate_set` trỏ vào đèn). Nhãn "hợp lệ" với môi trường nhưng vô nghĩa, và model viết câu không thể mô tả đúng ("tắt đèn phòng ngủ nhỏ" nhưng nhãn là điều hòa).
- **Id không nhất quán trong mẫu:** "xem tin nhắn với Lan rồi gửi vào đó" nhưng `read_chat` và `send_message` dùng hai `chat_id` khác nhau.
- **Người viết câu không thấy mục id trỏ tới:** kết quả bị cắt 400 ký tự nên mục được chọn có thể nằm ngoài phần được cắt.
- Ngoài ra: chữ Hán lọt qua lọc (đã sửa ở `dfdde56`), có câu thêm việc không có trong call (ví dụ thêm "nhắc xem lại kết quả chuyển khoản") hoặc sai số tiền.
Bản sửa (commit sau `dfdde56`): id chỉ được đổi sang bản ghi **cùng loại** (`type`/`kind`/`category`), cùng một id gốc luôn ra cùng một id mới trong một mẫu, hai id gốc khác nhau không bị gộp, và mỗi mẫu mang thêm `refs` (mục mà từng id trỏ tới) để người viết và bước tự chấm đọc được. Kiểm tra local: 2000 mẫu, phát lại và oracle khớp 2000/2000, **0/459** call `smart_home` sai loại.
**Việc cần làm:** làm lại bước thử 200 mẫu với mã mới. Các tỉ lệ giữ 74% và 85,5% của lần thử đầu **không dùng được nữa** vì tính trên dữ liệu có nhãn sai.
**Các bước làm lại (khoảng 1 giờ GPU cho cả hai model):**
1. Kéo `main` mới nhất (từ commit `620b3f7`), dựng lại dataset mã, gắn vào notebook.
2. Sinh call, ưu tiên chuỗi dài vì chuỗi dài bị lọc nhiều nhất:
   `python -I -X utf8 scripts/gen_calls.py --n 200 --weights 1:0.3,2:0.4,3:0.3`
3. Viết câu bằng từng model (Qwen3-8B và Qwen3.5-9B) với `gen_text.py`, như lần trước, ra hai tệp riêng.
4. **Mới: kiểm tra khứ hồi cho từng tệp**, dùng adapter `q4b_lora` làm người kiểm tra (nó biết giao thức một dòng JSON mỗi lượt; model gốc chưa tinh chỉnh thường không làm được và sẽ cho 0%):
   `python -I -X utf8 scripts/verify_callgen.py --model $QWEN --adapter checkpoints/q4b_lora --inp <tệp câu> --out <tệp kiểm> --check-all --control data/processed/dev_ids.txt --control-n 100`
   Lệnh in 3 số cần gửi: **CONTROL** (tỉ lệ đạt trên 100 mẫu thật, là mức trần, kỳ vọng khoảng 60 đến 70%), tỉ lệ đạt của mẫu **qua** bộ lọc rẻ, và tỉ lệ đạt của mẫu **bị loại** bởi bộ lọc rẻ. Nếu hai tỉ lệ sau gần nhau thì bộ lọc rẻ chưa mang thông tin gì.
5. Gửi: bản tổng hợp như lần trước, ba số ở bước 4 cho mỗi model, và **30 dòng ngẫu nhiên** mới (cùng 30 id cho hai model, kèm `verified`, `text_ok`, `agent_ok`). Chưa train.
Cách đọc: tỉ lệ đạt của mẫu sinh ra so với CONTROL cho biết câu sinh ra nhất quán với nhãn đến đâu. Mẫu không đạt có thể là câu xấu hoặc mẫu khó (khó thì lại có ích cho train), nên đây là thước đo, không phải bộ lọc cuối.

### A5, bước tiếp theo (10/11): sinh số lượng, train theo bộ chia A4, rồi mới train bản nộp
Kết quả làm lại cho thấy câu và nhãn đủ tốt để đi tiếp (mẫu đạt khứ hồi đều hợp lý, mẫu lỗi bị bắt), và hai model viết câu tương đương nhau. Chưa có điểm nào chứng minh A5 giúp ích: bước sau phải đo trên app lạ.
**Lỗi tìm thêm khi kiểm tra số liệu (10/11), đã sửa trong mã:** môi trường đôi khi trả `ok: true` cho lệnh không làm gì (`done: false` hoặc `message` là chữ báo lỗi: món không có trong thực đơn, gán việc cho người không tồn tại, số dư không đủ). Bản sinh cũ giữ các lệnh này: 7/200 mẫu (3,5%) của lần làm lại, trong khi train thật chỉ có 1/1545. Đã loại ở bước sinh (`soft_failure`). **Mẫu đã sinh từ mã cũ phải bỏ**, sinh lại bằng mã mới.
**Lỗi thứ hai tìm thấy khi kiểm tra (10/11), đã sửa:** `--weights` gần như không có tác dụng. Khi không tìm được đoạn phù hợp, vòng lặp cũ bỏ qua và vẫn phát ra mẫu với ÍT đoạn hơn yêu cầu, nên **172/200 mẫu (86%) của lần làm lại chỉ có 1 đoạn** dù đã xin 30/40/30. Các mẫu "chuỗi dài" của lần đó thực ra là một app với nhiều call, không phải nhiều việc ghép lại. Bản sửa không bao giờ phát mẫu thiếu đoạn (thử tối đa 25 người cho trước khi bỏ). Kiểm tra: với `--weights 1:0.25,2:0.35,3:0.3,4:0.1` ra đúng phân bố (32/39/25/4%).
**Kết quả kiểm tra ý "viết câu theo từng đoạn":** trên 200 mẫu làm lại, chỉ 25 mẫu có 2 đoạn và 3 mẫu có 3 đoạn, nên chưa đủ để kết luận. Số có được: khứ hồi đạt 55 đến 59% với 1 đoạn và 24 đến 28% với 2 đoạn (khoảng tin cậy rộng, 14 đến 48%); tỉ lệ này gần với mô hình độc lập (p^2 = 31 đến 35%) và với việc người kiểm tra cũng yếu hơn ở chuỗi dài (điểm dev thật: 77 với 1 call, 49 với 2 call). Phần mất điểm vì câu viết kém (`text_ok` đạt 84 đến 92%) không tăng theo số đoạn. **Chưa có bằng chứng cho thấy viết theo đoạn sẽ tốt hơn**, nên không làm; nếu muốn kiểm tra sau, sinh lại với mã mới (đủ mẫu nhiều đoạn) rồi đo lại.
**Thay đổi so với bước thử:** (1) trộn **cả hai model** viết câu (một nửa mẫu mỗi model, để đa dạng văn phong); (2) **bỏ bước tự chấm** của model viết câu (`gen_text.py --no-verify`, cho kết quả không thông tin và tốn thời gian); (3) khứ hồi chạy **3 tiến trình song song** trên cùng GPU (96 GB đủ chỗ; giải mã từng mẫu là tác vụ chờ nên song song gần như nhân thông lượng); (4) `--weights 1:0.35,2:0.4,3:0.2,4:0.05` (số **đoạn** trộn vào một mẫu; train thật có 48% mẫu một app, 37% hai app, 11% ba app, 4% bốn app).
```bash
# 1. sinh call (CPU), chia đôi cho hai model viết câu
python -I -X utf8 scripts/gen_calls.py --n 4000 --weights 1:0.35,2:0.4,3:0.2,4:0.05
split -n l/2 -d data/processed/callgen_pending.jsonl data/processed/pend_     # pend_00 (cho 8B), pend_01 (cho 9B)
# 2. viết câu (GPU), không tự chấm
python -I -X utf8 scripts/gen_text.py --model <Qwen3-8B>   --inp data/processed/pend_00 --out data/processed/txt_8b.jsonl --batch 32 --no-verify
python -I -X utf8 scripts/gen_text.py --model <Qwen3.5-9B> --inp data/processed/pend_01 --out data/processed/txt_9b.jsonl --batch 32 --no-verify
# 3. khứ hồi bằng q4b_lora, 3 tiến trình song song (cat rồi chia ba theo dòng), bỏ --check-all
cat data/processed/txt_8b.jsonl data/processed/txt_9b.jsonl > data/processed/txt_all.jsonl
split -n l/3 -d data/processed/txt_all.jsonl data/processed/vpart_
for k in 00 01 02; do python -I -X utf8 scripts/verify_callgen.py --model $QWEN --adapter checkpoints/q4b_lora    --inp data/processed/vpart_$k --out data/processed/chk_$k.jsonl & done; wait
cat data/processed/chk_0*.jsonl > data/processed/callgen_checked_all.jsonl
```
Gửi lại **trước khi train**: số mẫu sinh, số mẫu được giữ (`roundtrip` true) theo số call, và 30 dòng ngẫu nhiên **của mẫu được giữ**.
**Hai lần train, theo thứ tự (không làm lần hai nếu lần một không đạt):**
1. **Bộ chia A4:** `build_sft.py --asr-tag phowl --hold-apps flights,bike_rental,package_tracking,shopping_list,podcast_radio,loyalty --hist-max 300 --callgen data/processed/callgen_checked_all.jsonl` (đã tự bỏ mẫu dùng app bị giấu và câu chép y hệt câu thật), train `q4b_holdout_a5` đúng công thức A4 nhưng thêm `holdout/sft_train_cg.jsonl` vào `--data`, KHÔNG thêm `sft_train_hist` (history không có bằng chứng giúp). Đo trên `holdout/eval_ids.txt` (app lạ) và `holdout/dev_ids.txt` (app đã thấy), chữ sạch, so với `q4b_holdout`: 41,93 và 66,03.
2. **Chỉ khi (1) đạt:** train `q4b_a5` trên toàn bộ train (không giấu app) + `sft_train_cg.jsonl`, đo dev 155 mẫu chữ sạch và ASR `phowl_ft`, so với `q4b_lora` (66,82 và 56,98 với luật bật).
**Tiêu chí quyết định (đặt trước khi xem số):** (1) đạt nếu điểm app lạ tăng **ít nhất 5 điểm** (so từng cặp bằng `compare_runs.py`) và điểm app đã thấy không giảm quá 3 điểm. (2) dùng cho bản nộp nếu dev ASR không giảm so với `q4b_lora` và dev sạch không giảm quá 2 điểm. Dev nhiễu, chênh dưới khoảng 6 điểm không phân biệt được, nên quyết định chính dựa vào (1).
**Thời gian dự kiến:** sinh và viết câu khoảng 30 phút, khứ hồi khoảng 45 đến 60 phút với 3 tiến trình, train mỗi lần khoảng 3,5 đến 4,5 giờ (dữ liệu thêm khoảng 2000 mẫu). Làm lần (1) xong trong đêm 10/11, lần (2) trong ngày 11 đến 12/11, để chốt cấu hình 13/11. Nếu thiếu giờ GPU, báo ngay: chỉ làm (1) và dùng nó để quyết định có đáng làm (2) hay không.

### A4b. Tham chiếu "app đã thấy" cho A4 (khoảng 10 phút GPU; làm trước A5)
**Kết quả (09/11):** khoảng cách đã thấy − lạ của `q4b_holdout` là +20,9 (chỉ exec+ask, [+10,6; +31,4]), trên ngưỡng 15 nên **làm A5** (bước thử 200 mẫu trước, chưa train). Xem `docs/BAN_GIAO_A4b.md`.
A4 cho 41,93 trên 174 mẫu dùng app bị giấu, nhưng chưa biết model đó đạt bao nhiêu trên app **đã thấy**, nên chưa biết app lạ làm mất bao nhiêu điểm. Con số "mất khoảng 24 điểm" hiện lấy từ `q4b_lora` trên dev (66,82), khác tập mẫu, khác dữ liệu train, khác loại ASR trong train: không so được.
Cách đo đúng: chạy chính `q4b_holdout` (adapter còn trong output notebook A4) trên `data/processed/holdout/dev_ids.txt` (132 mẫu dev dùng app đã thấy, không nằm trong train của bộ chia này), chữ sạch, luật bật như A4:
```bash
bash scripts/infer_multi_gpu.sh train outputs/seen_q4b_holdout.jsonl --ids data/processed/holdout/dev_ids.txt      --backend hf --model $QWEN --adapter checkpoints/q4b_holdout --text clean
python -I -X utf8 scripts/eval.py --sub outputs/seen_q4b_holdout.jsonl --ids data/processed/holdout/dev_ids.txt
```
Gửi điểm (tổng, exec, ask, oos). Khoảng cách `seen − unseen` của **cùng một model** mới là số đo cho A5: nếu dưới khoảng 6 điểm thì A5 không đáng làm; nếu trên 15 điểm thì A5 đáng làm. Nếu có thể, chạy thêm `q4b_holdout_hist` trên cùng tệp để so.

### A6 (ASRX): đã rút lại (08/11)
Ý định ban đầu: thêm các mẫu train với bản ghi của ASR gốc `phowl`, vì tưởng LLM chỉ thấy bản ghi gần sạch của `phowl_ft`. **Tiền đề sai**: `q4b_lora` và `q4b_hist` đã được train với bản ghi `phowl` (ASR gốc, WER 25%), nên dữ liệu "ASRX" trùng 1390/1390 hàng với dữ liệu đã dùng. Cộng thêm việc `aug_data` ghi lại `sft_train_asr.jsonl` bằng bản `phowl_ft`, chạy A6 chỉ thêm một bản gần sạch, không đo được điều đã dự định. Mã (`--asr-extra-tag`, cờ `ASRX`) đã gỡ. Phát hiện bởi bên chạy GPU khi đối chiếu dữ liệu.
Hệ quả: khoảng 15 điểm mất do ASR trên dev không phải do LLM chưa thấy chữ nhiễu. Hướng còn lại là cải thiện chính ASR hoặc đưa nhiều kết quả ASR vào prompt (chưa làm, chưa đo).

## B. Việc làm tay, không cần GPU (ai cũng làm được)

| # | Việc | Làm thế nào | Gửi lại |
|---|---|---|---|
| B1 | Xác nhận mọi Kaggle dataset và notebook chứa dữ liệu ở chế độ **Private**, Internet OFF, và không còn output notebook chứa transcript | Kaggle > Datasets/Notebooks > Settings; xem `docs/KAGGLE.md` mục 0 | ảnh chụp trạng thái (che tên người khác) hoặc xác nhận bằng chữ |
| B2 | Đối chiếu checksum Qwen3-4B với bản chính thức | Tải `Qwen/Qwen3-4B` từ Hugging Face (tài khoản của bạn), `sha256sum` 3 file shard, so với bảng trong `docs/MODELS.md` | kết quả khớp / không khớp. Adapter được train trên bản lấy từ một Kaggle dataset, không phải bản tải trực tiếp |
| B3 | Điền licence thật cho các model | Đọc model card của `vinai/PhoWhisper-large` và `Qwen/Qwen3-4B`, ghi tên licence và đường dẫn vào cột Licence của `docs/MODELS.md`. Hiện bảng ghi "tag từ Hugging Face, cần đối chiếu" | PR hoặc ghi chú vào file |
| B4 | Hỏi BTC (cổng cuộc thi hoặc kênh hỗ trợ) | (a) **Hướng dẫn đóng gói Docker** (đề nói có tài liệu riêng, ta chưa nhận); (b) cách đếm tham số: theo checkpoint phát hành hay theo phần thực sự nạp; (c) lệnh chạy và đường dẫn cụ thể khi họ mount `/data` | câu trả lời nguyên văn |
| B5 | Báo hạn mức GPU Kaggle còn lại trong tuần | trang Settings của tài khoản | số giờ còn lại; quyết định cắt thử nghiệm nào |
| B6 | Kiểm tra mẫu | chạy thử A1 bước 1 và gửi dòng `wc -l` | 3 con số |

## C. Việc của tôi (máy local, không cần GPU)
Đã xong: khai thác lượt `history` (ý 1), cờ `HIST` và `SKIP_ASR` trong pipeline, bộ chia "app lạ" (A4), `scripts/compare_runs.py` (so từng cặp mẫu, có khoảng tin cậy; trên dev chỉ phát hiện được chênh lệch từ khoảng 6 điểm trở lên), sửa lỗi biến thể `ask`, tài liệu này.

Tiếp theo, theo thứ tự:
1. ~~Đổi thực thể đồng bộ~~ **bỏ**: phân tích lỗi giá trị trên dev cho thấy chủ yếu là số học thời gian ("trước mười lăm phút" từ giờ giao hàng, "thứ ba tuần sau" lệch một ngày) và chọn ID, không phải đổi tên người hay địa điểm.
2. Mẫu `ask` bằng cách xóa tham số khỏi câu: cần xác định đúng đoạn bị xóa trong câu nói, nhãn sai nhưng trông hợp lệ nếu xác định sai; chưa làm.
3. Dev theo từng bước (cần chạy model nên làm khi có GPU).
4. Khung Docker v0 (chạy offline, đọc `/data`, ghi `result/submission.jsonl`), chờ B4(a) để chỉnh cho đúng.

## D. Điều kiện chốt
- **Đến 13/11:** chốt cấu hình, ngừng thêm tính năng. Sau mốc này chỉ sửa lỗi và đóng gói.
- **Chỉ giữ một thay đổi nếu** dev tăng rõ khi so từng cặp mẫu, thời gian p95 vẫn thoải mái, và hai lần chạy cho kết quả giống từng byte.
- Không dùng lượt nộp public để dò đáp án (RULES 5.1); chỉ nộp khi dev tăng rõ, cần người phụ trách đồng ý.
- Không đưa dữ liệu, audio, transcript, dự đoán, trọng số vào git. Gửi adapter và file dev qua Drive riêng của đội.
