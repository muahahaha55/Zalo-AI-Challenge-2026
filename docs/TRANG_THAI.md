# Trang tình trạng (một trang, cập nhật 10/11 tối)

Mục đích: ai mở repo cũng biết **đang làm gì, đã quyết gì, còn thiếu gì, ai làm**. Chi tiết từng thí nghiệm ở `docs/EXPERIMENTS.md`, việc cho người chạy GPU ở `docs/TASKS.md`, đóng gói ở `docs/DOCKER.md`.

## 1. Bài nộp hiện tại (đã đo)
Qwen3-4B + LoRA (`q4b_lora`) và PhoWhisper-large đã fine-tune, luật `rules.py` bật. Dev 56,98 (qua ASR) và 66,82 (chữ sạch); **public 48,56** (nộp lần 1). Hệ thống 5,698B tham số (adapter chạy kèm), dưới 6B.

## 2. Các thí nghiệm đã xong
| Thí nghiệm | Kết quả | Quyết định |
|---|---|---|
| A1: thêm dữ liệu từ `history` | không thấy khác biệt | không dùng |
| A4: giấu 6 app, đo trên app lạ | **41,93** (app lạ) | thí nghiệm gốc để đo A5 |
| A4b: cùng model trên app đã thấy | **66,03**; **khoảng cách 21 điểm** | app lạ là điểm yếu chính, đáng làm A5 |
| A6: thêm câu ASR nhiễu | tiền đề sai (model đã học trên ASR gốc) | **rút lại** |
| A5 thử 200 mẫu (hai lần) | lần 1 nhãn sai, lần 2 chấp nhận được | sửa bộ sinh rồi sinh số lượng |

## 3. A5 (dữ liệu tự sinh) đang ở đâu
**A5 là gì:** lấy các đoạn call thật trong train, chạy lại trong môi trường của mẫu khác (id đổi theo môi trường mới, nhãn đúng sẵn), rồi nhờ **model chạy trên máy (Qwen3-8B và Qwen3.5-9B)** viết câu nói. Lọc 3 lớp: luật đơn giản, rồi **khứ hồi** (adapter `q4b_lora` đọc câu, chạy agent, chỉ giữ mẫu ra đúng trạng thái như nhãn).

**Đã sinh 4000 mẫu (đã chạy xong trên Kaggle, model viết câu là 8B và 9B, mỗi cái một nửa):**
| Bước | Số mẫu |
|---|---|
| Sinh ra | 4000 |
| Qua luật lọc rẻ | 3081 |
| **Được giữ sau khứ hồi (dùng cho bản nộp)** | **1550** (8B 725, 9B 825) |
| Trong đó dùng được cho thí nghiệm theo bộ chia A4 | **907** |

**Vì sao 907 chứ không phải 1550:** thí nghiệm A4 giấu 6 app và 155+ mẫu dev khỏi train. Một mẫu A5 dựng từ mẫu nền và các mẫu cho; nếu **bất kỳ** mẫu nào trong số đó nằm ngoài phần train của bộ chia A4 (thuộc app bị giấu hoặc thuộc dev) thì phải bỏ để không rò. Quá trình: 4000, bỏ 1791 mẫu dính vào phần ngoài train, còn 2209, bỏ 1302 mẫu không qua lọc, còn **907**. Với bản nộp thật (không giấu app) dùng được 1550.

**Chất lượng (đọc tay 30 mẫu được giữ):** 28/30 đúng và tự nhiên. Một lỗi đã tìm ra và **đã sửa trong mã** (câu nhắn cho Minh nhưng nội dung còn gọi tên "Sơn" chép từ mẫu cho): ảnh hưởng khoảng 3% dữ liệu đã sinh, nên không chặn thí nghiệm này.

**Về "model to hơn" (Qwen3.6-27B, Gemma 4 31B):** không cần. 8B và 9B đã sinh xong 4000 mẫu, và hai model cho kết quả khứ hồi ngang nhau (khoảng 50%). Điểm nghẽn là bộ lọc khứ hồi, không phải sức viết của model.

## 4. Việc tiếp theo cho A5 (người chạy GPU)
1. Train `q4b_holdout_a5`: dữ liệu bộ chia A4 + 907 mẫu A5 (không có `history`), đúng công thức A4.
2. Đo trên app lạ (so với **41,93**) và app đã thấy (so với **66,03**).
3. **Tiêu chí đặt trước:** app lạ tăng **ít nhất 5 điểm** và app đã thấy không giảm quá 3 điểm. Nếu đạt: train bản nộp với 1550 mẫu (bộ sinh đã sửa). Nếu không: giữ `q4b_lora`.
Thời gian: khoảng 3,5 đến 4,5 giờ GPU mỗi lần train.

## 5. Đóng gói nộp (BTC hướng dẫn 10/11)
| Mục | Tình trạng |
|---|---|
| Khung `/code` (predict.sh, predict.py, notebook, README, training_code) | **đã viết**, chạy thử không Docker với backend rỗng |
| Dockerfile (CUDA 13.2, không ENTRYPOINT, torch 2.11.0) | **đã viết, chưa dựng** (Docker tắt) |
| Gộp LoRA vào Qwen3-4B (`scripts/merge_lora.py`) | script xong; **chưa chạy**; cần GPU và **phải đo lại dev** trước khi dùng |
| Test Solution của BTC | 1 lượt/ngày, **đóng 23:59 ngày 15/11** |
| Nộp chính thức | **23:59 ngày 16/11**, tối đa 2 lượt, lượt cuối được chấm |
| GPU của máy chấm | **chưa rõ** (RTX 5090 hay A30), cần hỏi BTC |
| Dữ liệu huấn luyện trong gói | **cần quyết**: tôi đề xuất chỉ nộp mã tái tạo, không đăng dữ liệu dẫn xuất công khai |

## 6. Mốc
| Ngày | Việc |
|---|---|
| 11 đến 12/11 | Train A5 (bộ chia A4), gộp LoRA và đo lại, dựng image, **Test Solution lượt đầu** |
| 13/11 | **Chốt cấu hình** (q4b_lora hay q4b_a5) |
| 14 đến 15/11 | Kiểm tra Docker, Test Solution thêm lượt, sửa theo log |
| 16/11 | Nộp chính thức (Drive "Anyone with the link" + MD5), **trước 00:00 ngày 16/11 phải chốt danh sách thành viên đội** |
| 20/11, 23/11 | Private test, nộp kết quả cuối |

## 7. Việc đang chờ
| Việc | Ai |
|---|---|
| Push 2 commit (bộ khung nộp, bộ lọc tên cũ) | tôi, chờ bạn đồng ý |
| Thư hỏi BTC (GPU, dữ liệu huấn luyện, base model khi gộp LoRA) | bạn gửi, tôi soạn |
| Train A5 theo bộ chia A4 | bạn kia (GPU) |
| Bật Docker để dựng thử image | bạn |
