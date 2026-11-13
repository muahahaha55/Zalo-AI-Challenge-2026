# Thuê một máy ảo để dựng và tải lên image nộp bài (hướng dẫn cho người triển khai)

**Việc cần làm:** dựng image nộp bài (khoảng 20 GB), xuất `zac2026_<TeamName>.tar.gz` (khoảng 18 GB), tải lên Google Drive, gửi cho chủ dự án **link Drive, mã MD5 và `models.lock.json` mới**. Máy của chủ dự án không đủ dung lượng mạng, và các máy GPU 4090/3090 đã hết chỗ, nên làm trên một **máy ảo thuê theo giờ, không cần GPU**: đường truyền của máy chủ trung tâm dữ liệu tải 11 GB trọng số và tải lên 18 GB chỉ mất vài chục phút.

**Hạn:** Test Solution đóng **23:59 ngày 15/11** (1 lượt/ngày), nộp chính thức **23:59 ngày 16/11**. Nên tải lên xong trong **14/11** để còn một ngày sửa theo log của BTC. Thời gian làm khoảng 3 đến 5 giờ.

## 1. Thuê máy (không cần GPU)
Cấu hình tối thiểu: **Ubuntu 22.04, 8 vCPU, 16 GB RAM, đĩa từ 160 GB, đường truyền không giới hạn**, vị trí bất kỳ, đăng nhập bằng khóa SSH.
- Gợi ý: **Hetzner Cloud CX43** (8 vCPU, 16 GB, 160 GB, 20 TB lưu lượng), giá niêm yết khoảng **0,026 euro mỗi giờ** (khoảng 16,5 euro mỗi tháng), tức cả buổi làm việc dưới **1 euro**. Hetzner đôi khi yêu cầu xác minh tài khoản mới (có thể mất vài giờ), nên đăng ký sớm.
- Thay thế nếu Hetzner chậm: DigitalOcean, Vultr hoặc Linode (máy 8 vCPU, 16 GB, đĩa từ 160 GB, cùng yêu cầu). Giá thay đổi theo từng nhà cung cấp, kiểm tra trên trang của họ.
- **Không dùng** máy GPU kiểu vast.ai hay RunPod loại "container": chúng thường chặn Docker bên trong. Nếu dùng vast.ai thì phải chọn loại **máy ảo (VM)**.
- Ai trả tiền: người triển khai tạo tài khoản, ghi lại tổng chi phí để chủ dự án hoàn lại hoặc chia.

## 2. Cài công cụ trên máy ảo (SSH vào, chạy với `root`)
```bash
apt-get update && apt-get install -y git python3 python3-pip unzip pigz rclone
curl -fsSL https://get.docker.com | sh              # cài Docker
docker run --rm hello-world                          # kiểm tra
df -h /                                              # cần khoảng 70 GB trống
```

## 3. Lấy trọng số lên máy ảo
Trọng số (`q4b_merged/` 8 GB và `phowl_ft/merged/` 3 GB, kèm `SHA256SUMS_merged.txt`) nằm ở Kaggle (output của notebook `merge-lora`) và ở file `lifemate_weights.zip` trên Drive của bạn. Có hai cách:

**Cách A (đơn giản nhất): từ Kaggle.**
```bash
pip install kaggle
mkdir -p ~/.kaggle && nano ~/.kaggle/kaggle.json      # dán token Kaggle CỦA BẠN (đăng nhập Kaggle, Settings, API); chmod 600 sau đó
chmod 600 ~/.kaggle/kaggle.json
kaggle kernels output <tên-người-dùng>/merge-lora -p /root/weights
```
**Cách B: từ Drive** (cùng cấu hình `rclone` ở mục 4):
```bash
rclone copy gdrive:lifemate_weights.zip /root/ --progress && unzip /root/lifemate_weights.zip -d /root/weights
```
Sau đó kiểm tra: `cd /root/weights && sha256sum -c SHA256SUMS_merged.txt` phải OK hết (và `q4b_merged/model.safetensors` bắt đầu bằng `652e96ae`, `phowl_ft/merged/model.safetensors` bắt đầu bằng `cffba381`).

## 4. Cấu hình rclone với Google Drive (để tải tệp nộp lên)
Máy ảo không có trình duyệt nên dùng cách ủy quyền từ máy của bạn:
1. Trên **máy tính của bạn** (cài rclone: https://rclone.org/downloads/) chạy `rclone authorize "drive"`, đăng nhập đúng tài khoản Google có Drive dung lượng lớn, rồi **chép đoạn token** in ra.
2. Trên máy ảo: `rclone config` → `n` (remote mới) → tên `gdrive` → loại `drive` → Enter cho client id và secret → scope `1` (toàn quyền) → Enter nhiều lần → "Use auto config?" chọn **`n`** → dán token → "Configure as Shared Drive?" `n` → `y`.
3. Kiểm tra: `rclone lsd gdrive:`.
**Token này là bí mật** (cho phép truy cập Drive của bạn): không dán vào chat hay lên GitHub, và xóa khỏi máy ảo khi xong (mục 8).

## 5. Lấy mã và đặt model
```bash
git clone https://github.com/muahahaha55/Zalo-AI-Challenge-2026.git && cd Zalo-AI-Challenge-2026
git log --oneline -1                                      # phải là baba2e6 hoặc mới hơn
mkdir -p models/llm models/asr                            # KHÔNG có models/adapter (LoRA đã gộp)
cp -r /root/weights/q4b_merged/*        models/llm/
cp -r /root/weights/phowl_ft/merged/*   models/asr/
python3 scripts/check_models.py --update-lock             # ghi khóa sha256 mới cho hai thư mục model
python3 scripts/check_models.py                           # phải in: parameters ... system 5.5xxxB ... models OK
```

## 6. Dựng và xuất image
```bash
GZIP_CMD=pigz bash scripts/build_submission.sh <TeamName>      # tên đội viết liền, không dấu; ví dụ LifeMateTeam
```
Mất khoảng 30 đến 60 phút (tải thư viện CUDA, dựng, nén). Kết quả: `zac2026_<TeamName>.tar.gz` và `zac2026_<TeamName>.tar.gz.md5` ở thư mục gốc. Kiểm tra nhanh image (không cần GPU):
```bash
docker run --rm --entrypoint /bin/bash zac2026-<teamname>:v1 -c "ls /code /code/saved_models && python3 -c 'import torch,transformers;print(torch.__version__, transformers.__version__)'"
```
Phải thấy `predict.sh`, `predict.py`, `predict_notebook.ipynb`, `saved_models` (có `llm` và `asr`), và `torch 2.11.0+cu130`.

## 7. Tải lên Drive và lấy link
```bash
rclone copy zac2026_<TeamName>.tar.gz gdrive: --progress --drive-chunk-size 256M      # tiếp tục được nếu đứt: chạy lại cùng lệnh
cat zac2026_<TeamName>.tar.gz.md5
```
Rồi trên giao diện Drive web: chuột phải vào tệp, **Chia sẻ**, chọn **"Bất kỳ ai có đường liên kết" (Anyone with the link), quyền Người xem**, sao chép liên kết dạng `https://drive.google.com/file/d/<mã>/view`. Mở liên kết bằng cửa sổ ẩn danh để chắc tải được.

## 8. Gửi kết quả và dọn dẹp
- Gửi chủ dự án (kênh riêng): **link Drive, mã MD5 (dòng trong tệp `.md5`), chi phí thuê, và `models.lock.json` mới** (chép nội dung, hoặc đẩy lên nhánh `thuandev`).
- **Xóa máy ảo** (để ngừng tính tiền), xóa `~/.config/rclone/rclone.conf` và `~/.kaggle/kaggle.json` nếu còn, và thu hồi quyền truy cập rclone trong tài khoản Google (https://myaccount.google.com/permissions) nếu muốn.
- Tệp trọng số và `.tar.gz` chỉ để ở link Drive nộp cho BTC; không đăng ở chỗ khác.

## Nếu gặp lỗi
- `apt-get update` không phân giải được tên miền: thử lại sau, hoặc thêm `--build-arg APT_MIRROR=<một gương Ubuntu gần máy>` vào lệnh `docker build` trong `scripts/build_submission.sh`.
- Hết đĩa: xóa `/root/weights` sau khi đã chép vào `models/`, hoặc `docker system prune -af`.
- `build_submission.sh` dừng ở `check_compliance`: gửi nguyên thông báo cho chủ dự án.
