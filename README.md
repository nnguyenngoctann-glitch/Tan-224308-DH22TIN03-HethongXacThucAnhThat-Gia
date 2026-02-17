# Tan-224308-DH22TIN03-HethongXacThucAnhThat-Gia
Đồ án: Hệ thống xác thực ảnh thật và ảnh AI.

## Tổng quan
Dự án sử dụng mô hình học sâu để phân loại ảnh thành 2 lớp:
- real: ảnh thật
- ai_generated: ảnh do AI tạo

Hệ thống gồm:
- Backend API để nhận ảnh và trả kết quả dự đoán
- Frontend để tải ảnh và hiển thị kết quả

## Công nghệ sử dụng
- Python
- PyTorch, Torchvision
- FastAPI, Uvicorn
- Pillow, NumPy
- scikit-learn, matplotlib, seaborn
- HTML, CSS, JavaScript

## Quy trình xây dựng
1. Chuẩn bị dữ liệu theo 2 lớp real và ai_generated
2. Tiền xử lý ảnh: resize 224x224, normalize theo ImageNet, augment cho train
3. Xây dựng mô hình ResNet50 theo transfer learning
4. Huấn luyện mô hình với CrossEntropyLoss và Adam
5. Đánh giá bằng accuracy, confusion matrix, classification report
6. Lưu model tốt nhất tại model/best_model.pth
7. Triển khai API qua FastAPI với endpoint predict và predict_batch
8. Kết nối frontend để upload ảnh và xem kết quả

## Dataset train hiện tại
- Đường dẫn: dataset/train_binary_balanced_10000
- ai_generated: 10000 ảnh
- real: 10000 ảnh

## Dataset validation hiện tại
- Đường dẫn: dataset/val_binary_balanced
- ai_generated: 804 ảnh
- real: 804 ảnh

## Trạng thái dữ liệu
- Đã làm sạch trùng ảnh trong từng bộ
- Đã loại trùng chéo giữa train và validation
- Train và validation hiện không bị leakage dữ liệu

## Tỷ lệ train/validation khi huấn luyện
- Train: 80%
- Validation: 20% (chia từ dataset train trong quá trình training)
