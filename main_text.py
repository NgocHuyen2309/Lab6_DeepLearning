import torch
import torch.nn as nn
import torch.optim as optim
from src.text_classification.model import TextTransformer

# TODO (@Hoàng): Import DataLoader thật từ file dataset.py của bạn vào đây
# from src.text_classification.dataset import get_text_dataloaders

def main():
    # 1. CẤU HÌNH SIÊU THAM SỐ (Hyperparameters)
    VOCAB_SIZE = 10000     
    NUM_CLASSES = 4        # VD: Dataset AG News có 4 nhãn
    MAX_SEQ_LEN = 128
    BATCH_SIZE = 32
    EPOCHS = 10            # Số vòng lặp qua toàn bộ dữ liệu
    LEARNING_RATE = 1e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] Khởi tạo mô hình Text Transformer trên thiết bị: {DEVICE}")
    model = TextTransformer(
        vocab_size=VOCAB_SIZE, 
        num_classes=NUM_CLASSES, 
        d_model=256, 
        n_heads=8, 
        d_ff=512, 
        num_layers=4, 
        max_seq_len=MAX_SEQ_LEN,
        dropout=0.1
    ).to(DEVICE)

    # 2. TẢI DỮ LIỆU (DATA LOADERS)
    # TODO (@Hoàng): Mở comment 2 dòng dưới khi đã viết xong DataLoader
    # train_loader, val_loader = get_text_dataloaders(BATCH_SIZE, MAX_SEQ_LEN)
    print("[INFO] Đang chờ Hoàng nạp Dataloader dữ liệu thật...")

    # Tạm thời tạo Dataloader giả (Dummy DataLoader) để test luồng
    dummy_inputs = torch.randint(0, VOCAB_SIZE, (100, MAX_SEQ_LEN))
    dummy_labels = torch.randint(0, NUM_CLASSES, (100,))
    train_loader = [(dummy_inputs[i:i+BATCH_SIZE], dummy_labels[i:i+BATCH_SIZE]) for i in range(0, 100, BATCH_SIZE)]
    val_loader = train_loader # Tạm dùng chung cho Val

    # 3. HÀM LOSS VÀ OPTIMIZER
    criterion = nn.CrossEntropyLoss()
    # Với Transformer, AdamW thường giúp mô hình hội tụ tốt và tránh Overfitting hơn Adam
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # 4. VÒNG LẶP HUẤN LUYỆN (TRAINING LOOP)
    print("[INFO] Bắt đầu quá trình huấn luyện...")
    for epoch in range(EPOCHS):
        
        # --- PHASE 1: TRAIN ---
        model.train()
        total_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_inputs, batch_labels in train_loader:
            batch_inputs, batch_labels = batch_inputs.to(DEVICE), batch_labels.to(DEVICE)
            padding_mask = (batch_inputs != 0).to(DEVICE) # Tạo mask bỏ qua các token 0 (Padding)

            optimizer.zero_grad()
            logits = model(batch_inputs, padding_mask=padding_mask)
            loss = criterion(logits, batch_labels)
            
            loss.backward()
            # Kỹ thuật Gradient Clipping: Ngăn chặn lỗi nổ đạo hàm (Exploding Gradient) rất hay gặp ở Transformer
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            correct_train += (predictions == batch_labels).sum().item()
            total_train += batch_labels.size(0)

        avg_train_loss = total_train_loss / len(train_loader)
        train_acc = (correct_train / total_train) * 100

        # --- PHASE 2: EVALUATION (ĐÁNH GIÁ TẬP VALIDATION) ---
        model.eval()
        total_val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad(): # Tắt tính đạo hàm để tiết kiệm RAM và tăng tốc độ
            for batch_inputs, batch_labels in val_loader:
                batch_inputs, batch_labels = batch_inputs.to(DEVICE), batch_labels.to(DEVICE)
                padding_mask = (batch_inputs != 0).to(DEVICE)

                logits = model(batch_inputs, padding_mask=padding_mask)
                loss = criterion(logits, batch_labels)

                total_val_loss += loss.item()
                predictions = torch.argmax(logits, dim=-1)
                correct_val += (predictions == batch_labels).sum().item()
                total_val += batch_labels.size(0)

        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = (correct_val / total_val) * 100

        # In kết quả từng Epoch cho Lê Hân ghi nhận và vẽ biểu đồ
        print(f"Epoch [{epoch+1:02d}/{EPOCHS}] | "
              f"Train Loss: {avg_train_loss:.4f} - Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f} - Val Acc: {val_acc:.2f}%")

    print("[SUCCESS] Huấn luyện hoàn tất!")
    # Lưu trọng số mô hình (Model Weights)
    # torch.save(model.state_dict(), "transformer_text_model.pth")

if __name__ == "__main__":
    main()