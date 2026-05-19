import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import kiến trúc mạng từ phần của Trí
from src.image_classification.model import VisionTransformer
from src.image_classification.dataset import get_cifar10_dataloaders 

def train_model():
    # 1. CẤU HÌNH THIẾT BỊ
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Bắt đầu huấn luyện ViT trên thiết bị: {device}")

    # 2. KHỞI TẠO MÔ HÌNH 
    model = VisionTransformer(
        img_size=224,      # Khớp với dataset.py (resize 224x224)
        patch_size=16,     # 14x14 = 196 patches
        in_channels=3, 
        num_classes=10,    # CIFAR-10 có 10 lớp
        embed_dim=192,     # Giảm mạnh để tiết kiệm VRAM (ViT-Tiny)
        depth=6,           # 6 Transformer blocks
        num_heads=3        # 3 attention heads (192/3=64 head_dim)
    ).to(device)

    # 3. TẢI DỮ LIỆU
    print("[INFO] Đang nạp dữ liệu từ dataset.py...")
    train_loader, val_loader = get_cifar10_dataloaders(batch_size=32)  # Giảm từ 64 → 32

    # 4. CẤU HÌNH HÀM LOSS VÀ OPTIMIZER
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)

    # 5. VÒNG LẶP HUẤN LUYỆN
    epochs = 10
    
    # Khởi tạo mảng lưu trữ số liệu vẽ biểu đồ
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }
    
    for epoch in range(epochs):
        # --- PHASE 1: TRAINING ---
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        train_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1:02d}/{epochs}] Train", leave=False, unit="batch")
        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            # Cập nhật live loss & acc trên thanh tiến trình
            train_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.*correct_train/total_train:.1f}%")
            
        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = (correct_train / total_train) * 100
        
        # --- PHASE 2: VALIDATION ---
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch [{epoch+1:02d}/{epochs}]   Val", leave=False, unit="batch")
            for images, labels in val_bar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                val_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.*correct_val/total_val:.1f}%")
                
        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc = (correct_val / total_val) * 100
        
        # Lưu số liệu
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_acc'].append(epoch_val_acc)
        
        print(f"Epoch [{epoch+1:02d}/{epochs}] | "
              f"Train Loss: {epoch_train_loss:.4f} - Train Acc: {epoch_train_acc:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} - Val Acc: {epoch_val_acc:.2f}%")
        

    print("[INFO] Đã hoàn thành huấn luyện, đang trích xuất biểu đồ...")
    
    # 6. VẼ VÀ LƯU BIỂU ĐỒ
    plt.figure(figsize=(14, 5))
    
    # Biểu đồ Loss
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), history['train_loss'], label='Train Loss', marker='o')
    plt.plot(range(1, epochs + 1), history['val_loss'], label='Validation Loss', marker='o')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Biểu đồ Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), history['train_acc'], label='Train Accuracy', marker='o')
    plt.plot(range(1, epochs + 1), history['val_acc'], label='Validation Accuracy', marker='o')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('vit_training_metrics.png')
    print("[SUCCESS] Đã lưu biểu đồ thành công vào tệp 'vit_training_metrics.png'")

if __name__ == "__main__":
    train_model()