import torch
import torch.nn as nn
import torch.optim as optim
from src.text_classification.model import TextTransformer

def main():
    # 1. Cấu hình siêu tham số (Hyperparameters)
    VOCAB_SIZE = 10000     # Kích thước từ điển
    NUM_CLASSES = 4        # VD: AG News có 4 nhãn
    MAX_SEQ_LEN = 128
    BATCH_SIZE = 16
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] Khởi tạo mô hình Transformer trên {DEVICE}...")
    model = TextTransformer(
        vocab_size=VOCAB_SIZE, 
        num_classes=NUM_CLASSES, 
        d_model=256, 
        n_heads=8, 
        d_ff=512, 
        num_layers=4, 
        max_seq_len=MAX_SEQ_LEN
    ).to(DEVICE)

    # 2. Tạo Dummy Data (Dữ liệu giả) để Test Pipeline
    # TODO (@Hoàng): Chỗ này Hoàng sẽ thay bằng Dataloader tải dữ liệu AG News hoặc IMDb thật nhé!
    print("[INFO] Đang tạo dữ liệu giả (Dummy Data) để test luồng Tensor...")
    dummy_input = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, MAX_SEQ_LEN)).to(DEVICE)
    # Mask: Giả sử một số vị trí cuối là padding (giá trị 0)
    dummy_mask = (dummy_input != 0).to(DEVICE) 
    dummy_labels = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,)).to(DEVICE)

    # 3. Loss và Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 4. Huấn luyện 1 Step (Sanity Check)
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    logits = model(dummy_input, padding_mask=dummy_mask)
    loss = criterion(logits, dummy_labels)
    
    # Backward pass
    loss.backward()
    optimizer.step()

    print(f"[SUCCESS] Pipeline chạy thành công!")
    print(f"Shape đầu vào: {dummy_input.shape} -> Shape đầu ra (Logits): {logits.shape}")
    print(f"Giá trị Loss ở bước thử nghiệm: {loss.item():.4f}")
    print("Sẵn sàng giao cho team Thực nghiệm!")

if __name__ == "__main__":
    main()