"""
main_text.py - Thực nghiệm Transformer cho Text Classification
--------------------------------------------------------------
File này đã thay Dummy Data bằng DataLoader AG News thật do Duy Hoàng phụ trách.

Cách chạy nhanh để kiểm tra pipeline:
    python main_text.py --epochs 1 --max-train-samples 2000 --max-val-samples 500

Cách chạy đầy đủ hơn:
    python main_text.py --epochs 5 --batch-size 32 --max-seq-len 128

Kết quả đầu ra:
    results/text_training_log.csv       # Loss/Accuracy từng epoch
    results/text_training_curves.png    # Biểu đồ train/validation
    results/best_text_transformer.pt    # Checkpoint tốt nhất theo validation accuracy
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import time
from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim

from src.text_classification.dataset import PAD_IDX, build_ag_news_dataloaders
from src.text_classification.model import TextTransformer


def set_seed(seed: int) -> None:
    """Cố định seed để kết quả thực nghiệm dễ lặp lại."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def forward_transformer(
    model: nn.Module,
    input_ids: torch.Tensor,
    padding_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Gọi model linh hoạt để tương thích với nhiều cách Huyền đặt forward().

    Chuẩn team đang dùng:
        logits = model(input_ids, padding_mask=padding_mask)
    """
    try:
        return model(input_ids, padding_mask=padding_mask)
    except TypeError:
        try:
            return model(input_ids, padding_mask)
        except TypeError:
            return model(input_ids)


def calculate_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = torch.argmax(logits, dim=1)
    correct = (predictions == labels).sum().item()
    return correct / labels.size(0)


def train_one_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_idx, batch in enumerate(dataloader, start=1):
        batch = move_batch_to_device(batch, device)
        input_ids = batch["input_ids"]          # Shape: [Batch, Max_seq_len]
        padding_mask = batch["padding_mask"]    # Shape: [Batch, Max_seq_len], True = token thật
        labels = batch["label"]                 # Shape: [Batch]

        optimizer.zero_grad(set_to_none=True)

        logits = forward_transformer(model, input_ids, padding_mask)
        loss = criterion(logits, labels)

        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (torch.argmax(logits, dim=1) == labels).sum().item()
        total_samples += batch_size

        if batch_idx == 1:
            print("[DEBUG] Batch đầu tiên:")
            print(f"        input_ids shape    : {tuple(input_ids.shape)}")
            print(f"        padding_mask shape : {tuple(padding_mask.shape)}")
            print(f"        logits shape       : {tuple(logits.shape)}")

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        input_ids = batch["input_ids"]
        padding_mask = batch["padding_mask"]
        labels = batch["label"]

        logits = forward_transformer(model, input_ids, padding_mask)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (torch.argmax(logits, dim=1) == labels).sum().item()
        total_samples += batch_size

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


def save_log_csv(history: List[Dict[str, float]], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = [
        "epoch",
        "train_loss",
        "train_accuracy",
        "val_loss",
        "val_accuracy",
        "epoch_time_sec",
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def save_training_plot(history: List[Dict[str, float]], output_path: str) -> None:
    """Lưu biểu đồ Loss/Accuracy để nhóm phân tích đưa vào báo cáo."""
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] Không thể vẽ biểu đồ vì thiếu matplotlib: {exc}")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    epochs = [row["epoch"] for row in history]

    plt.figure()
    plt.plot(epochs, [row["train_loss"] for row in history], marker="o", label="Train Loss")
    plt.plot(epochs, [row["val_loss"] for row in history], marker="o", label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Text Transformer - Loss theo Epoch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    loss_plot_path = output_path.replace(".png", "_loss.png")
    plt.savefig(loss_plot_path, dpi=150)
    plt.close()

    plt.figure()
    plt.plot(epochs, [row["train_accuracy"] for row in history], marker="o", label="Train Accuracy")
    plt.plot(epochs, [row["val_accuracy"] for row in history], marker="o", label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Text Transformer - Accuracy theo Epoch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    acc_plot_path = output_path.replace(".png", "_accuracy.png")
    plt.savefig(acc_plot_path, dpi=150)
    plt.close()

    print(f"[INFO] Đã lưu biểu đồ: {loss_plot_path}")
    print(f"[INFO] Đã lưu biểu đồ: {acc_plot_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Transformer from scratch on AG News")

    # Data
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--max-vocab-size", type=int, default=10000)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--max-train-samples", type=int, default=None, help="Dùng để debug nhanh; mặc định dùng toàn bộ train")
    parser.add_argument("--max-val-samples", type=int, default=None, help="Dùng để debug nhanh; mặc định dùng toàn bộ test")
    parser.add_argument("--num-workers", type=int, default=0)

    # Training
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    # Model - giữ cùng cấu hình với skeleton ban đầu của nhóm
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)

    # Output
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--save-checkpoint", action="store_true", default=True)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 72)
    print("THỰC NGHIỆM TEXT CLASSIFICATION - AG NEWS")
    print("=" * 72)
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Hyperparameters:")
    print(f"       epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")
    print(f"       max_seq_len={args.max_seq_len}, max_vocab_size={args.max_vocab_size}")
    print(f"       d_model={args.d_model}, n_heads={args.n_heads}, d_ff={args.d_ff}, num_layers={args.num_layers}")

    # 1. Load AG News thật thay cho Dummy Data
    train_loader, val_loader, vocab, num_classes = build_ag_news_dataloaders(
        data_dir=args.data_dir,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        max_vocab_size=args.max_vocab_size,
        min_freq=args.min_freq,
        num_workers=args.num_workers,
        seed=args.seed,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
    )

    vocab_size = len(vocab)
    print(f"[INFO] Khởi tạo mô hình Transformer trên {device}...")
    model = TextTransformer(
        vocab_size=vocab_size,
        num_classes=num_classes,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        max_seq_len=args.max_seq_len,
    ).to(device)

    # 2. Loss và Optimizer
    # PAD_IDX được dùng trong Dataset; CrossEntropyLoss chỉ tính trên nhãn lớp, không cần ignore_index.
    _ = PAD_IDX
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 3. Training Loop đầy đủ: log Loss/Accuracy từng epoch
    best_val_accuracy = 0.0
    history: List[Dict[str, float]] = []
    checkpoint_path = os.path.join(args.output_dir, "best_text_transformer.pt")

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            grad_clip=args.grad_clip,
        )
        val_metrics = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        epoch_time = time.time() - start_time
        row = {
            "epoch": epoch,
            "train_loss": round(train_metrics["loss"], 6),
            "train_accuracy": round(train_metrics["accuracy"], 6),
            "val_loss": round(val_metrics["loss"], 6),
            "val_accuracy": round(val_metrics["accuracy"], 6),
            "epoch_time_sec": round(epoch_time, 2),
        }
        history.append(row)

        print(
            f"[EPOCH {epoch:02d}/{args.epochs}] "
            f"train_loss={row['train_loss']:.4f} | train_acc={row['train_accuracy']:.4f} | "
            f"val_loss={row['val_loss']:.4f} | val_acc={row['val_accuracy']:.4f} | "
            f"time={row['epoch_time_sec']:.2f}s"
        )

        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            if args.save_checkpoint:
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "vocab": vocab,
                        "args": vars(args),
                        "best_val_accuracy": best_val_accuracy,
                    },
                    checkpoint_path,
                )
                print(f"[INFO] Đã lưu checkpoint tốt nhất: {checkpoint_path}")

    # 4. Xuất CSV + biểu đồ cho Hân/Thành phân tích trong báo cáo
    csv_path = os.path.join(args.output_dir, "text_training_log.csv")
    plot_path = os.path.join(args.output_dir, "text_training_curves.png")
    save_log_csv(history, csv_path)
    save_training_plot(history, plot_path)

    print("=" * 72)
    print("[SUCCESS] Hoàn thành thực nghiệm Text Classification bằng dữ liệu AG News thật.")
    print(f"[RESULT] Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"[OUTPUT] CSV log: {csv_path}")
    print(f"[OUTPUT] Checkpoint: {checkpoint_path if args.save_checkpoint else 'Không lưu'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
