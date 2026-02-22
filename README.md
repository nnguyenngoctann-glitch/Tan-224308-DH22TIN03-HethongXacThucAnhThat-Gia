# Tan-224308-DH22TIN03-HethongXacThucAnhThat-Gia

Đồ án hệ thống xác thực ảnh thật và ảnh do AI tạo.

## 1. Tổng quan
- Dự án có 2 phần chính:
  - backend: xử lý dữ liệu, huấn luyện, đánh giá, API, Grad CAM
  - frontend: giao diện tải ảnh và hiển thị kết quả
- Mô hình chính đang dùng: EfficientNet-B0 cho phân loại 2 lớp gia và that

## 2. Cấu trúc thư mục

```text
backend/
  api.py
  train.py
  evaluate.py
  chia_du_lieu.py
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

## 3. Các phần đã làm

### 3.1 Huấn luyện mô hình
- Tệp: backend/train.py
- Backbone: EfficientNet-B0 pretrained
- Đầu ra: 2 lớp gia và that
- Tăng cường dữ liệu khi huấn luyện:
  - Resize 256
  - CenterCrop 224
  - RandomHorizontalFlip
  - ColorJitter
  - GaussianBlur
- Chuẩn hóa theo ImageNet mean và std
- Tối ưu: Adam, tốc độ học 1e-4
- Hàm mất mát: CrossEntropyLoss
- Lưu mô hình tốt nhất theo độ chính xác tập val vào best_efficientnet_b0.pth

Lý do chọn EfficientNet-B0 vì trước đó có sử dụng qua ResNet50 và ResNet18:
- Nhẹ hơn ResNet50, phù hợp máy có tài nguyên vừa phải
- Thường cho đặc trưng tốt hơn ResNet18 trong nhiều trường hợp
- Cân bằng tốt giữa tốc độ và độ chính xác

### 3.2 Đánh giá mô hình
- Tệp: backend/evaluate.py
- Có các chỉ số:
  - accuracy
  - precision
  - recall
  - F1 score
- Có confusion matrix
- Có ROC curve và lưu ảnh kết quả

### 3.3 API backend
- Tệp: backend/api.py
- Endpoint:
  - POST /predict
  - POST /predict-with-cam
- Trả về nhãn và độ tin cậy
- Có trạng thái uncertain khi độ tin cậy thấp hơn ngưỡng

### 3.4 Frontend
- Tệp: frontend/index.html, frontend/style.css, frontend/app.js
- Chức năng:
  - tải ảnh từ máy
  - kéo thả ảnh
  - xem ảnh xem trước
  - gọi API và hiển thị kết quả
  - hiển thị Grad CAM gồm heatmap và ảnh overlay

### 3.5 Grad CAM
- Tệp: backend/grad_cam.py
- Chức năng:
  - hook vào lớp tích chập cuối
  - sinh heatmap
  - ghép heatmap lên ảnh gốc
- Tích hợp vào endpoint POST /predict-with-cam

## 4. Dữ liệu đang dùng để huấn luyện
- Tỉ lệ chia: train val test bằng 80 10 10
- Số lượng hiện tại trong du_lieu:
  - gia: 15500 ảnh
  - that: 15500 ảnh
  - tổng: 31000 ảnh
- Theo từng tập:
  - train: gia 12400, that 12400
  - val: gia 1550, that 1550
  - test: gia 1550, that 1550

## 5. Nguồn dữ liệu
- du_lieu_nguon/dataset chính:
  - gia 30000
  - that 30000
- du_lieu_nguon/dataset phụ:
  - gia 7536
  - that 20000

## 6. Xử lý ảnh mờ để giảm sai số ảnh chất lượng thấp
- Ảnh mờ được tạo từ dữ liệu đã có nhãn, nhãn giữ nguyên
- Mỗi lớp bổ sung 500 ảnh mờ:
  - kiểu hide: 250
  - kiểu gopro: 250
- Chia đều theo 80 10 10:
  - train: 400
  - val: 50
  - test: 50
 

 ## Cách chạy nhanh : 
-B1 : -m uvicorn backend.api:app --host 0.0.0.0 --port 8000
-B2 : truy cập http://localhost:8000
