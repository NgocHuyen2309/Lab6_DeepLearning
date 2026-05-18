# Gói hoàn thiện nhiệm vụ của Duy Hoàng - Text Classification

## 1. File đã hoàn thiện

Copy 2 file sau vào đúng vị trí trong repo:

```text
src/text_classification/dataset.py
main_text.py
```

## 2. Điểm đã xử lý đúng yêu cầu

- `TextClassificationDataset` kế thừa `torch.utils.data.Dataset`.
- Có tokenizer, xây vocabulary, padding/truncation về `max_seq_len`.
- Trả về `input_ids`, `padding_mask`, `label` đúng để đưa vào Transformer.
- `main_text.py` đã xóa Dummy Data và thay bằng DataLoader AG News thật.
- Training loop có đủ Train Loss, Train Accuracy, Validation Loss, Validation Accuracy.
- Tự xuất `results/text_training_log.csv` để nhóm phân tích vẽ biểu đồ và đưa vào báo cáo.
- Tự lưu biểu đồ loss/accuracy nếu môi trường có `matplotlib`.

## 3. Cài thư viện

Ưu tiên cách 1:

```bash
pip install datasets matplotlib
```

Nếu môi trường không dùng HuggingFace datasets, có thể dùng torchtext tương thích với PyTorch:

```bash
pip install torchtext matplotlib
```

Hoặc tải AG News CSV đặt tại:

```text
data/ag_news_csv/train.csv
data/ag_news_csv/test.csv
```

## 4. Chạy thử nhanh để kiểm tra pipeline

```bash
python main_text.py --epochs 1 --max-train-samples 2000 --max-val-samples 500
```

## 5. Chạy thực nghiệm để lấy số liệu báo cáo

```bash
python main_text.py --epochs 5 --batch-size 32 --max-seq-len 128 --lr 1e-4
```

## 6. Kết quả cần nộp cho Hân/Thành

Sau khi chạy xong, gửi các file trong thư mục `results/`:

```text
results/text_training_log.csv
results/text_training_curves_loss.png
results/text_training_curves_accuracy.png
results/best_text_transformer.pt
```

## 7. Đoạn mô tả ngắn đưa vào báo cáo Word

Trong phần thực nghiệm Text Classification, nhóm sử dụng tập dữ liệu AG News gồm 4 lớp chủ đề tin tức. Dữ liệu văn bản được tiền xử lý bằng tokenizer dạng từ/tách dấu câu, sau đó ánh xạ token sang chỉ số trong từ điển. Mỗi câu được chuẩn hóa về cùng độ dài `max_seq_len = 128` bằng truncation và padding. Mô hình Transformer nhận đầu vào dạng `input_ids` có shape `[batch_size, max_seq_len]` và `padding_mask` có cùng shape để phân biệt token thật với vị trí `<pad>`. Quá trình huấn luyện ghi lại `train_loss`, `train_accuracy`, `val_loss`, `val_accuracy` theo từng epoch và xuất ra file CSV để phục vụ phân tích hội tụ/overfitting trong báo cáo.
