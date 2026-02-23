# Hệ Thống Xác Thực Ảnh Thật/Giả

Đồ án xây dựng hệ thống phân loại ảnh thật và ảnh do AI tạo, có API suy luận và giao diện web trực quan kèm Grad-CAM để giải thích kết quả.

## 1. Mục tiêu dự án
- Phân loại ảnh thành 2 lớp: that (ảnh thật) và gia (ảnh AI).
- Trả về nhãn + độ tin cậy dự đoán.
- Hỗ trợ trạng thái uncertain khi độ tin cậy thấp.
- Hiển thị Grad-CAM để người dùng thấy vùng mô hình tập trung.

## 2. Công nghệ sử dụng
### 2.1 Backend và AI/ML
- Python: ngôn ngữ chính để xây dựng toàn bộ backend và pipeline ML.
- PyTorch: framework huấn luyện và suy luận mô hình học sâu.
- torchvision: cung cấp EfficientNet-B0, transform ảnh và ImageFolder.
- FastAPI: xây dựng REST API phục vụ dự đoán.
- Uvicorn: ASGI server chạy dịch vụ FastAPI.

### 2.2 Xử lý ảnh và trực quan
- Pillow (PIL): đọc/ghi/chuyển đổi ảnh.
- numpy: xử lý dữ liệu mảng trong quá trình tạo CAM.
- matplotlib: vẽ ROC curve và lưu biểu đồ đánh giá.

### 2.3 Frontend
- HTML: cấu trúc giao diện.
- CSS: định dạng giao diện và responsive.
- JavaScript: xử lý upload/kéo-thả ảnh, gọi API và hiển thị kết quả.

### 2.4 Mô hình sử dụng
- EfficientNet-B0 (pretrained + fine-tuning): backbone chính cho bài toán phân loại 2 lớp thật/giả.

## 3. Cấu trúc thư mục
```text
backend/
  api.py              # API suy luận và phục vụ frontend
  train.py            # Huấn luyện mô hình
  evaluate.py         # Đánh giá mô hình trên test set
  chia_du_lieu.py     # Chia dữ liệu train/val/test
  grad_cam.py         # Sinh heatmap/overlay Grad-CAM

frontend/
  index.html          # Giao diện chính
  style.css           # Giao diện hiển thị
  app.js              # Tương tác người dùng và gọi API

du_lieu/
  train/that, train/gia
  val/that,   val/gia
  test/that,  test/gia

best_efficientnet_b0.pth   # Checkpoint mô hình tốt nhất
requirements.txt           # Danh sách thư viện
```

## 4. Phân tích hệ thống
### 4.1 Kiến trúc tổng thể
- Tầng dữ liệu: du_lieu được chia thành train/val/test.
- Tầng huấn luyện: train.py huấn luyện model và lưu checkpoint tốt nhất.
- Tầng đánh giá: evaluate.py tính Accuracy, Precision, Recall, F1, ROC/AUC.
- Tầng suy luận: api.py nạp checkpoint, cung cấp endpoint dự đoán.
- Tầng giao diện: frontend upload ảnh, hiển thị kết quả và Grad-CAM.

### 4.2 Luồng hoạt động chính
1. Chuẩn hóa dữ liệu bằng chia_du_lieu.py với tỉ lệ 80/10/10.
2. Huấn luyện EfficientNet-B0 trong train.py.
3. Đánh giá chất lượng bằng evaluate.py.
4. Khởi chạy API bằng api.py.
5. Người dùng upload ảnh qua frontend, hệ thống trả kết quả + CAM.

### 4.3 Phân tích chức năng từng module
- backend/chia_du_lieu.py:
  - Thu thập ảnh từ nguồn, chia train/val/test theo tỉ lệ cấu hình.
  - Copy ảnh vào cấu trúc dữ liệu đích để train/evaluate.
- backend/train.py:
  - Dùng EfficientNet-B0 pretrained, thay classifier 2 lớp.
  - Áp dụng augmentation (Resize, Crop, Flip, ColorJitter, Blur).
  - Tối ưu bằng Adam, loss CrossEntropyLoss, lưu best checkpoint.
- backend/evaluate.py:
  - Đánh giá trên test set với Accuracy, Precision, Recall, F1.
  - In confusion matrix, vẽ ROC curve và tính AUC.
- backend/api.py:
  - Endpoint GET /health kiểm tra trạng thái hệ thống.
  - Endpoint POST /predict trả label và confidence.
  - Endpoint POST /predict-with-cam trả thêm heatmap/overlay dạng base64.
  - Có ngưỡng uncertain_threshold để giảm dự đoán thiếu chắc chắn.
- backend/grad_cam.py:
  - Hook vào lớp conv cuối của model.
  - Tạo heatmap vùng chú ý và ghép overlay lên ảnh gốc.
- frontend/index.html + style.css + app.js:
  - Upload/kéo-thả ảnh, preview ảnh.
  - Gọi API và hiển thị kết quả dự đoán.
  - Hiển thị Grad-CAM trực quan cho người dùng.

### 4.4 Dữ liệu hiện tại
- Dữ liệu train/val/test cân bằng 2 lớp.
- Quy mô hiện dùng:
  - Tổng 31,000 ảnh (gia 15,500 và that 15,500)
  - train: 24,800 (12,400 mỗi lớp)
  - val: 3,100 (1,550 mỗi lớp)
  - test: 3,100 (1,550 mỗi lớp)
- Có bổ sung ảnh mờ để cải thiện độ bền với ảnh chất lượng thấp.

## 5. Tiến độ hiện tại của dự án
Mức độ hoàn thiện tổng thể ước tính: 85%.

### 5.1 Đã hoàn thành
- Hoàn chỉnh pipeline train và lưu checkpoint tốt nhất.
- Có module evaluate với bộ chỉ số đánh giá chính.
- Có API suy luận ổn định cho 2 endpoint dự đoán.
- Có frontend demo end-to-end.
- Có Grad-CAM tích hợp vào luồng dự đoán.
- Có tài liệu mô tả chi tiết và báo cáo tiến độ.

### 5.2 Đang thực hiện
- Rà soát thêm trường hợp dự đoán sai.
- Tinh chỉnh ngưỡng uncertain để tăng độ ổn định.
- Tối ưu nội dung báo cáo học thuật.

## 6. Cách chạy nhanh
```bash
pip install -r requirements.txt
python -m uvicorn backend.api:app --host 0.0.0.0 --port 8000
```
Truy cập: http://localhost:8000
