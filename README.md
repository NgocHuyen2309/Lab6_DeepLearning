# Gói hoàn thiện nhiệm vụ của Trí - Image Classification

## 1. File đã hoàn thiện

Copy 2 file sau vào đúng vị trí trong repo:

```text
src/image_classification/dataset.py
main_image.py
```

## 2. Điểm đã xử lý đúng yêu cầu

- Tích hợp thành công tập dữ liệu `CIFAR-10` từ thư viện `torchvision`.
- Có tiền xử lý bằng `Resize` về 224x224, `Normalize`, và tăng cường dữ liệu `RandomHorizontalFlip`.
- Trả về `images` có shape `[batch_size, 3, 224, 224]` và `labels` đúng để đưa vào Transformer.
- `main_image.py` đã kết nối DataLoader thật với mô hình Vision Transformer thu gọn (ViT-Tiny) để tránh lỗi VRAM.
- Training loop có đủ Train Loss, Train Accuracy, Validation Loss, Validation Accuracy.
- Tự xuất `results/vit_training_log.csv` để nhóm phân tích vẽ biểu đồ và đưa vào báo cáo.
- Tự lưu biểu đồ loss/accuracy nếu môi trường có `matplotlib`.

## 3. Cài thư viện

Cài đặt các gói PyTorch và công cụ trực quan:

```bash
pip install torch torchvision torchaudio matplotlib tqdm
```

*(Lưu ý: Nếu dùng GPU NVIDIA trên Windows, nhớ ưu tiên cài pytorch-cuda phù hợp)*

## 4. Chạy thử nhanh để kiểm tra pipeline

```bash
python main_image.py --epochs 1 --max-train-samples 2000 --max-val-samples 500
```

## 5. Chạy thực nghiệm để lấy số liệu báo cáo

```bash
python main_image.py --epochs 10 --batch-size 32 --lr 3e-4
```

## 6. Kết quả cần nộp cho Hân/Thành

Sau khi chạy xong, gửi các file trong thư mục `results/`:

```text
results/vit_training_log.csv
results/vit_training_curves.png
```

## 7. Đoạn mô tả ngắn đưa vào báo cáo Word

Trong phần thực nghiệm Image Classification, nhóm sử dụng tập dữ liệu CIFAR-10 gồm 10 lớp đối tượng cơ bản. Dữ liệu hình ảnh được tiền xử lý Resize lên 224x224 để phù hợp với chuẩn của ViT, sau đó chuẩn hóa phân phối màu sắc (Normalize) và áp dụng lật ảnh ngẫu nhiên để tăng cường dữ liệu. Mô hình Vision Transformer nhận đầu vào dạng `images` có shape `[batch_size, 3, 224, 224]`, thực hiện chia nhỏ ảnh thành các patches 16x16 và tính toán qua mạng Transformer Encoder để xuất ra tensor logits có shape `[batch_size, 10]`. Quá trình huấn luyện sử dụng AdamW ghi lại `train_loss`, `train_accuracy`, `val_loss`, `val_accuracy` theo từng epoch và xuất ra file CSV cùng biểu đồ trực quan để phục vụ phân tích hội tụ/overfitting trong báo cáo.
