# Nghiên Cứu Kiến Trúc Attention, Transformer & Vision Transformer (ViT)

Kho lưu trữ mã nguồn mở của Nhóm 1 phục vụ đồ án thực hành môn Deep Learning - Giảng viên hướng dẫn: Thầy Đặng N.H. Thành. Dự án tập trung triển khai từ con số 0 (From Scratch) hai kiến trúc mạng lớn dựa trên cơ chế Attention ứng dụng vào bài toán phân loại.

## 📂 Tổ chức Cấu trúc Mã nguồn
Hệ thống mã nguồn được module hóa thành các không gian riêng biệt tương ứng với nhiệm vụ phân công:
```text
attention-transformer-vit/
│
├── src/
│   ├── text_classification/       # PHẦN CHUNG: Khối xử lý Văn bản
│   │   ├── model.py               # Lõi Transformer mạng xử lý chuỗi tuần tự (Huyền)
│   │   └── dataset.py             # Thực thể quản lý cấu trúc dữ liệu văn bản (Hoàng)
│   │
│   └── image_classification/      # PHẦN RIÊNG: Khối xử lý Hình ảnh
│       ├── model.py               # Lõi mạng Vision Transformer (ViT) xử lý không gian (Trí)
│       └── dataset.py             # Thực thể quản lý và biến đổi hình ảnh (Augmentation) (Tâm)
│
├── main_text.py                   # Điểm kích hoạt huấn luyện Text Classification
├── main_image.py                  # Điểm kích hoạt huấn luyện Image Classification
└── README.md                      # Tài liệu điều hướng hệ thống