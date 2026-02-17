# Tan-224308-DH22TIN03-HethongXacThucAnhThat-Gia
Đồ án: Hệ thống xác thực ảnh thật, ảnh AI, ảnh chỉnh sửa.

## Tổng quan
Dự án dùng mô hình học sâu để phân loại ảnh thành 2 lớp:
- real - ảnh thật
- ai_generated - ảnh do AI tạo

Hệ thống gồm 2 phần:
- Backend API để nhận ảnh và trả kết quả dự đoán.
- Frontend để người dùng tải ảnh và xem kết quả trực quan.

## Công nghệ và thư viện sử dụng
- Python
- PyTorch, Torchvision dùng để huấn luyện và suy luận model
- FastAPI xây dựng API và chạy trên Uvicorn
- Pillow,  NumPy dùng xử lý ảnh/dữ liệu
- scikit-learn, matplotlib, seaborn dùng để đánh giá và vẽ biểu đồ
- HTML, CSS, JavaScript dùng để thiết kế giao diện web

## Quy trình xây dựng dự án
1. Chuẩn bị dữ liệu ảnh theo thư mục lớp (`real`, `ai_generated`).
(Hiện tại em đang train lại vì đang bị chênh lệch số hình của các lớp )
2. Tiền xử lý ảnh: resize `224x224`, normalize theo ImageNet, augment cho tập train.
3. Xây dựng mô hình `ResNet50` theo hướng transfer learning.
(trước đó em có sử dụng ResNet18 vì tỷ lệ chính xác còn thấp nên em chuyển qua ResNet50)
4. Huấn luyện model với `CrossEntropyLoss` + `Adam`.
5. Đánh giá trên validation: accuracy, confusion matrix, classification report.
6. Lưu model tốt nhất tại `model/best_model.pth`.
7. Triển khai model qua API `FastAPI` (`/predict`, `/predict_batch`).
8. Kết nối frontend để upload ảnh và hiển thị kết quả.

## Dataset để train :
- ai_generated : 6000 ảnh
- real : 6000 ảnh
## Chia train/val (80/20) :
- train : 80%
- Validation (hình để test model ) : 20%

