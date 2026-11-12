# Dựng và tải lên image nộp bài ở máy có đường truyền tốt

Dành cho người có máy Linux (hoặc Windows với Docker Desktop) có **Internet không giới hạn** và Docker. Lý do: image nộp bài nặng khoảng 20 GB, tệp `.tar.gz` nặng khoảng 18 GB, và phải tải lên Google Drive. Máy của chủ dự án không đủ dung lượng mạng để tải trọng số xuống rồi tải tệp này lên, nên việc dựng và tải lên làm ở nơi có đường truyền.

## Cần có
- Docker chạy được (`docker run hello-world`). **Không cần GPU** để dựng; cần GPU chỉ nếu muốn thử chạy (bước 7, không bắt buộc).
- Khoảng 70 GB đĩa trống (image 20 GB, tệp xuất 18 GB, bộ nhớ đệm), Python 3 và `git`.
- Trọng số (đã có sẵn ở người giữ GPU): thư mục `q4b_merged/` (LLM đã gộp LoRA, khoảng 8 GB) và `phowl_ft/merged/` (ASR đã gộp, khoảng 3 GB), kèm `SHA256SUMS_merged.txt`.
- Tài khoản Google Drive có ít nhất 25 GB trống.

## Các bước
```bash
# 1. mã (bản sạch, không có dữ liệu cuộc thi)
git clone https://github.com/muahahaha55/Zalo-AI-Challenge-2026.git && cd Zalo-AI-Challenge-2026 && git log --oneline -1

# 2. đặt trọng số: LLM đã gộp vào models/llm, ASR đã gộp vào models/asr (KHÔNG có models/adapter)
mkdir -p models/llm models/asr
cp -r <đường dẫn>/q4b_merged/*        models/llm/
cp -r <đường dẫn>/phowl_ft/merged/*   models/asr/
( cd <thư mục chứa q4b_merged và phowl_ft> && sha256sum -c SHA256SUMS_merged.txt )     # phải OK hết

# 3. ghi khóa sha256 mới của hai thư mục model và kiểm tra giới hạn tham số (kỳ vọng hệ thống 5,566B, LLM dưới 5B)
python3 scripts/check_models.py --update-lock
python3 scripts/check_models.py                      # phải in "models OK"

# 4. dựng image, xuất tệp nộp và MD5 (khoảng 1 đến 2 giờ; tên đội viết liền, không dấu)
bash scripts/build_submission.sh <TeamName>          # ra zac2026_<TeamName>.tar.gz và .md5 ở thư mục gốc

# 5. tải zac2026_<TeamName>.tar.gz lên Google Drive, chia sẻ "Anyone with the link", chép link và dòng MD5
```
Gửi lại cho chủ dự án: **link Drive, mã MD5, và `models.lock.json` mới** (hoặc commit lên nhánh của bạn). Chủ dự án sẽ nhập link và MD5 vào ô Test Solution của cổng.

## Lưu ý
- Nếu bước 4 báo lỗi mạng ở `apt-get update` (không phân giải được `archive.ubuntu.com`), thêm `--build-arg APT_MIRROR=http://mirror.bizflycloud.vn/ubuntu` vào lệnh `docker build` trong `scripts/build_submission.sh` (chỉ cần trên mạng bị chặn; mặc định để trống).
- Bản dựng mặc định dùng `torch 2.11.0` bản `cu130` như lúc train; không cần GPU để dựng.
- **Không** đưa trọng số hoặc tệp `.tar.gz` lên chỗ công khai ngoài link Drive nộp cho BTC.
- Bước 7 (không bắt buộc, cần GPU NVIDIA và driver từ 580 trở lên): kiểm tra image chạy được:
  `python3 scripts/verify_image.py --image zac2026-<teamname>:v1 --data-root <thư mục chứa public_test> --split public_test --runs 2 --limit 20 --out-dir check_img`
  rồi gửi `check_img/report.json`.
