import torch
import torch.nn as nn
import torch.optim as optim

# Import kiến trúc mạng từ phần của Trí
from src.image_classification.model import VisionTransformer

# TODO [TÂM]: Import DataLoader và Data Augmentation từ file dataset.py của bạn
# from src.image_classification.dataset import get_cifar10_dataloaders 

def train_model():
    # 1. CẤU HÌNH THIẾT BỊ (Tự động nhận diện CUDA)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Bắt đầu huấn luyện ViT trên thiết bị: {device}")

    # 2. KHỞI TẠO MÔ HÌNH (ViT-Base chuẩn 85M Params - Code bởi Trí)
    model = VisionTransformer(
        img_size=224, 
        patch_size=16, 
        in_channels=3, 
        num_classes=10, 
        embed_dim=768, 
        depth=12, 
        num_heads=12
    ).to(device)

    # 3. TẢI DỮ LIỆU
    # TODO [TÂM]: Khởi tạo dataloader của bạn ở đây
    # train_loader, val_loader = get_cifar10_dataloaders(batch_size=64)
    print("[INFO] Đang chờ nạp dữ liệu từ dataset.py...")

    # 4. CẤU HÌNH HÀM LOSS VÀ OPTIMIZER
    # Gợi ý cho Tâm: ViT thường hội tụ tốt hơn với AdamW thay vì Adam/SGD thường
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)

    # 5. VÒNG LẶP HUẤN LUYỆN (TRAINING LOOP)
    epochs = 10
    
    # TODO [TÂM]: Viết chi tiết vòng lặp Train & Eval ở dưới đây
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        """ 
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        """
        
        # In log ra màn hình cho đội Phân tích lấy số liệu vẽ chart
        # print(f"Epoch [{epoch+1}/{epochs}] - Loss: {running_loss/len(train_loader):.4f}")
        
    print("[INFO] Đã hoàn thành bộ khung, chờ Chí Tâm hoàn thiện dữ liệu!")

if __name__ == "__main__":
    train_model()