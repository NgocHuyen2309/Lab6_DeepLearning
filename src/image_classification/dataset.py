import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_cifar10_dataloaders(batch_size=64):
    """
    Hàm khởi tạo Data Augmentation và DataLoader cho CIFAR-10.
    """
    # Các bước Data Augmentation cho tập Train
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)), # Shape: [3, 224, 224]
        transforms.RandomHorizontalFlip(), # Tăng cường dữ liệu lật ảnh
        transforms.ToTensor(),
        # Normalize đưa pixel về không gian phân phối chuẩn
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), 
    ])

    # Tập Validation không dùng Augmentation, chỉ Resize và Normalize
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)), # Shape: [3, 224, 224]
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Tải dataset
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_val)

    # Đóng gói vào DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader