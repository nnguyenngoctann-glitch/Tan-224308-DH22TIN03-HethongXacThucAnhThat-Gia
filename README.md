# Tan-224308-DH22TIN03-HethongXacThucAnhThat-Gia
Đồ án: Hệ thống xác thực ảnh thật, ảnh AI, ảnh chỉnh sửa.

## Tổng quan
Dự án dùng mô hình học sâu để phân loại ảnh thành 3 lớp:
- real
- ai_generated
- edited

Hệ thống gồm 2 phần:
- Backend API để nhận ảnh và trả kết quả dự đoán.
- Frontend để người dùng tải ảnh và xem kết quả trực quan.

## Công nghệ sử dụng
- Python
- PyTorch, Torchvision (huấn luyện và suy luận model)
- FastAPI, Uvicorn (xây dựng API)
- Pillow, NumPy (xử lý ảnh/dữ liệu)
- scikit-learn, matplotlib, seaborn (đánh giá và vẽ biểu đồ)
- HTML, CSS, JavaScript (giao diện web)

## Quy trình xây dựng dự án
1. Chuẩn bị dữ liệu ảnh theo thư mục lớp (real, ai_generated, edited).
2. Tiền xử lý ảnh: resize 224x224, normalize theo ImageNet, augment cho tập train.
3. Xây dựng mô hình ResNet50 theo hướng transfer learning.
4. Huấn luyện model với CrossEntropyLoss + Adam.
5. Đánh giá trên validation: accuracy, confusion matrix, classification report.
6. Lưu model tốt nhất tại model/best_model.pth.
7. Triển khai model qua API FastAPI (/predict, /predict_batch).
8. Kết nối frontend để upload ảnh và hiển thị kết quả.

## Cấu trúc chính
```text
backend/main.py            # API FastAPI
training/train.py          # Huấn luyện model
training/dataset_loader.py # Nạp dữ liệu + transforms
model/model.py             # Kiến trúc model
utils/metrics.py           # Metrics và biểu đồ
utils/predict.py           # Dự đoán ảnh bằng script
frontend/                  # Giao diện web
```

## Cách chạy nhanh
1. Cài thư viện:
```bash
pip install -r requirements.txt
```

2. Huấn luyện:
```bash
python -m training.train
```

3. Chạy hệ thống:
```bash
python -m backend.main
```

Mở:
- http://localhost:8000 (giao diện)
- http://localhost:8000/docs (API docs)
