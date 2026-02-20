# Tan-224308-DH22TIN03-HethongXacThucAnhThat-Gia

Đồ án: Hệ thống xác thực ảnh thật và ảnh do AI tạo.

## 1. Tổng quan
Dự án gồm 2 phần:
- backend: xử lý dữ liệu, train/evaluate mô hình, API FastAPI, Grad-CAM.
- frontend: giao diện upload ảnh và hiển thị kết quả dự đoán.

Mô hình đang dùng chính:
- EfficientNet-B0 (PyTorch) cho bài toán phân loại 2 lớp: real và fake.

## 2. Cấu trúc thư mục

```text
backend/
  api.py
  train.py
  evaluate.py
  chia_du_lieu.py
  tron_du_lieu_2_nguon.py
  model_two_branch.py
  grad_cam.py

frontend/
  index.html
  style.css
  app.js

du_lieu/
  train/that, train/gia
  val/that,   val/gia
  test/that,  test/gia
```

## 3. Những gì đã làm

### 3.1. Chuẩn bị dữ liệu
- Đã tạo script chia dữ liệu: `backend/chia_du_lieu.py`.
- Đã tạo script trộn dữ liệu 2 nguồn fake có kiểm soát: `backend/tron_du_lieu_2_nguon.py`.
- Quy tắc trộn hiện tại:
  - `that`: chỉ lấy từ 1 nguồn (`dataset1`).
  - `gia`: trộn `dataset1` 70% + `dataset2` 30%.
  - Cân bằng lớp theo từng split: `that = gia`.
  - Tỉ lệ split: `train/val/test = 80/10/10`.

### 3.2. Train mô hình
- File: backend/train.py.
- Backbone: `EfficientNet-B0` pretrained.
- Output: 2 lớp.
- Augmentation train:
  - `Resize(256) -> CenterCrop(224) -> RandomHorizontalFlip -> ColorJitter -> GaussianBlur.
- Normalize theo ImageNet mean/std.
- Optimizer: Adam (`lr=1e-4`), loss: CrossEntropyLoss.
- Lưu model tốt nhất theo validation accuracy: best_efficientnet_b0.pth.
# Trước đó có sử dụng ResNet18 và ResNet50 :
Lý do chọn EfficientNet-B0 thay vì ResNet18 và ResNet50:
- So với ResNet50: EfficientNet-B0 nhẹ hơn đáng kể, ít tốn VRAM hơn, phù hợp với laptop.
- So với ResNet18: EfficientNet-B0 thường cho chất lượng đặc trưng tốt hơn ở cùng mức tài nguyên, nên cân bằng tốt giữa tốc độ và độ chính xác.
- Tổng thể: EfficientNet-B0 là điểm cân bằng hợp lý cho máy hiện tại (dễ train, ít lỗi thiếu bộ nhớ, chất lượng tốt).

### 3.3. Đánh giá mô hình
- File: backend/evaluate.py.
- Chức năng:
  - Tính `accuracy`, `precision`, `recall`, `F1-score`.
  - In confusion matrix.
  - Vẽ ROC curve và lưu ảnh.
  - In xác suất dự đoán của từng ảnh test.

### 3.4. Backend API
- File: backend/api.py.
- Endpoint chính: `POST /predict`.
- Nhận ảnh upload, tiền xử lý như pipeline inference, trả về:
  - label: real hoặc fake
  - confidence: độ tin cậy

### 3.5. Frontend
- `frontend/index.html`, `frontend/style.css`, `frontend/app.js`.
- Chức năng:
  - Upload ảnh.
  - Xem preview ảnh.
  - Gọi API `/predict` bằng fetch.
  - Hiển thị nhãn dự đoán và xác suất.

### 3.6. Grad-CAM (giải thích mô hình)
- File: backend/grad_cam.py.
- Chức năng:
  - Hook vào convolution layer cuối.
  - Sinh heatmap.
  - Overlay heatmap lên ảnh gốc.
  - Có thể tích hợp vào FastAPI endpoint.

### 3.7. Mô hình mở rộng 2 nhánh
- File: backend/model_two_branch.py.
- Nhánh 1: RGB qua EfficientNet backbone.
- Nhánh 2: FFT magnitude qua CNN 3 lớp conv.
- Nối feature 2 nhánh để phân loại 2 lớp.


## Train model : 
- Theo tỷ lệ là train/val/test : 80/10/10
- Real : 15k hình ảnh
- Fake : 15k Hình ảnh
- train khoảng 15-20 epochs