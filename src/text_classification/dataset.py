"""
dataset.py - Phần Duy Hoàng
--------------------------------
Tiền xử lý dữ liệu văn bản cho bài toán Text Classification bằng Transformer.

Điểm chính để báo cáo:
- Class TextClassificationDataset kế thừa torch.utils.data.Dataset.
- Tokenizer đơn giản bằng regex, không phụ thuộc thư viện nặng.
- Padding / Truncation cố định về max_seq_len.
- Trả về input_ids, padding_mask, label để đưa trực tiếp vào TextTransformer.
- Có hàm build_ag_news_dataloaders() để tải AG News thật, thay Dummy Data trong main_text.py.

Lưu ý:
- padding_mask = True tại vị trí token thật, False tại vị trí PAD.
- PAD token có id = 0; UNK token có id = 1.
"""

from __future__ import annotations

import csv
import os
import random
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_IDX = 0
UNK_IDX = 1


def basic_tokenizer(text: str) -> List[str]:
    """
    Tokenizer nhẹ cho tiếng Anh.

    Ví dụ:
        "Apple releases new iPhone!" -> ["apple", "releases", "new", "iphone", "!"]
    """
    text = str(text).lower().strip()
    return re.findall(r"[a-z0-9]+|[^\s\w]", text)


def build_vocab(
    texts: Iterable[str],
    max_vocab_size: int = 10000,
    min_freq: int = 2,
) -> Dict[str, int]:
    """
    Xây dựng từ điển token -> id từ tập train.

    Hai id đầu tiên được cố định:
    - 0: <pad>
    - 1: <unk>
    """
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(basic_tokenizer(text))

    vocab: Dict[str, int] = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
    for token, freq in counter.most_common(max_vocab_size - len(vocab)):
        if freq >= min_freq and token not in vocab:
            vocab[token] = len(vocab)
    return vocab


@dataclass
class TextExample:
    text: str
    label: int


class TextClassificationDataset(Dataset):
    """
    Dataset chuẩn cho Transformer Text Classification.

    Input thô:
        text: chuỗi văn bản
        label: nhãn số, đã chuẩn hóa về 0..num_classes-1

    Output mỗi sample:
        input_ids: Tensor shape [max_seq_len]
        padding_mask: Tensor shape [max_seq_len]
            True  = token thật
            False = padding
        label: Tensor scalar
    """

    def __init__(
        self,
        examples: Sequence[TextExample | Tuple[str, int]],
        vocab: Dict[str, int],
        max_seq_len: int = 128,
    ) -> None:
        self.examples: List[TextExample] = []
        for example in examples:
            if isinstance(example, TextExample):
                self.examples.append(example)
            else:
                text, label = example
                self.examples.append(TextExample(text=str(text), label=int(label)))

        self.vocab = vocab
        self.max_seq_len = int(max_seq_len)
        self.pad_idx = self.vocab.get(PAD_TOKEN, PAD_IDX)
        self.unk_idx = self.vocab.get(UNK_TOKEN, UNK_IDX)

    def __len__(self) -> int:
        return len(self.examples)

    def encode(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Chuyển văn bản -> input_ids và padding_mask.

        Shape trước padding: [seq_len_thật]
        Shape sau padding/truncate: [max_seq_len]
        """
        tokens = basic_tokenizer(text)
        token_ids = [self.vocab.get(token, self.unk_idx) for token in tokens]

        # Truncation: cắt bớt nếu câu dài hơn max_seq_len
        token_ids = token_ids[: self.max_seq_len]
        real_len = len(token_ids)

        # Padding: thêm PAD_IDX vào cuối câu nếu chưa đủ max_seq_len
        if real_len < self.max_seq_len:
            token_ids = token_ids + [self.pad_idx] * (self.max_seq_len - real_len)

        input_ids = torch.tensor(token_ids, dtype=torch.long)  # Shape: [max_seq_len]
        padding_mask = input_ids != self.pad_idx               # Shape: [max_seq_len]
        return input_ids, padding_mask

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        example = self.examples[index]
        input_ids, padding_mask = self.encode(example.text)
        label = torch.tensor(example.label, dtype=torch.long)

        return {
            "input_ids": input_ids,         # Shape: [max_seq_len]
            "padding_mask": padding_mask,   # Shape: [max_seq_len]
            "label": label,
        }


def _normalize_ag_news_label(label: int) -> int:
    """AG News trong torchtext dùng nhãn 1..4, HuggingFace dùng 0..3."""
    label = int(label)
    return label - 1 if 1 <= label <= 4 else label


def _read_ag_news_csv(csv_path: str) -> List[TextExample]:
    """
    Đọc AG News dạng CSV nếu người dùng đã tải sẵn.

    Hỗ trợ format phổ biến:
    - Không header: label,title,description
    - Có header: label/title/description hoặc text/label
    """
    examples: List[TextExample] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        sample = f.read(2048)
        f.seek(0)
        has_header = csv.Sniffer().has_header(sample)

        if has_header:
            reader = csv.DictReader(f)
            for row in reader:
                label_value = row.get("label") or row.get("class") or row.get("target")
                title = row.get("title", "")
                description = row.get("description", "") or row.get("desc", "")
                text = row.get("text") or f"{title} {description}".strip()
                if label_value is None or not text:
                    continue
                examples.append(TextExample(text=text, label=_normalize_ag_news_label(int(label_value))))
        else:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 2:
                    continue
                label_value = int(row[0])
                text = " ".join(row[1:]).strip()
                examples.append(TextExample(text=text, label=_normalize_ag_news_label(label_value)))

    return examples


def _load_ag_news_from_local_csv(data_dir: str) -> Optional[Tuple[List[TextExample], List[TextExample]]]:
    """Tìm data/ag_news_csv/train.csv và data/ag_news_csv/test.csv."""
    candidates = [
        data_dir,
        os.path.join(data_dir, "ag_news_csv"),
        os.path.join(data_dir, "ag_news"),
    ]
    for folder in candidates:
        train_path = os.path.join(folder, "train.csv")
        test_path = os.path.join(folder, "test.csv")
        if os.path.exists(train_path) and os.path.exists(test_path):
            print(f"[INFO] Đang đọc AG News từ CSV local: {folder}")
            return _read_ag_news_csv(train_path), _read_ag_news_csv(test_path)
    return None


def _load_ag_news_from_huggingface() -> Optional[Tuple[List[TextExample], List[TextExample]]]:
    """Tải AG News qua HuggingFace datasets nếu môi trường đã cài datasets."""
    try:
        from datasets import load_dataset  # type: ignore
    except Exception:
        return None

    try:
        print("[INFO] Đang tải AG News bằng HuggingFace datasets...")
        raw = load_dataset("ag_news")
        train_examples = [
            TextExample(text=item["text"], label=int(item["label"]))
            for item in raw["train"]
        ]
        test_examples = [
            TextExample(text=item["text"], label=int(item["label"]))
            for item in raw["test"]
]
        return train_examples, test_examples
    except Exception as exc:
        print(f"[WARN] Không tải được AG News từ HuggingFace datasets: {exc}")
        return None


def _load_ag_news_from_torchtext(root: str) -> Optional[Tuple[List[TextExample], List[TextExample]]]:
    """Tải AG News qua torchtext nếu môi trường hỗ trợ torchtext."""
    try:
        from torchtext.datasets import AG_NEWS  # type: ignore
    except Exception:
        return None

    try:
        print("[INFO] Đang tải AG News bằng torchtext...")
        train_iter, test_iter = AG_NEWS(root=root, split=("train", "test"))
        train_examples = [
            TextExample(text=text, label=_normalize_ag_news_label(label))
            for label, text in train_iter
        ]
        test_examples = [
            TextExample(text=text, label=_normalize_ag_news_label(label))
            for label, text in test_iter
        ]
        return train_examples, test_examples
    except Exception as exc:
        print(f"[WARN] Không tải được AG News từ torchtext: {exc}")
        return None


def load_ag_news_examples(data_dir: str = "data") -> Tuple[List[TextExample], List[TextExample]]:
    """
    Load AG News thật theo thứ tự ưu tiên:
    1. CSV local: data/ag_news_csv/train.csv + test.csv
    2. HuggingFace datasets
    3. torchtext
    """
    local_data = _load_ag_news_from_local_csv(data_dir)
    if local_data is not None:
        return local_data

    hf_data = _load_ag_news_from_huggingface()
    if hf_data is not None:
        return hf_data

    torchtext_data = _load_ag_news_from_torchtext(root=data_dir)
    if torchtext_data is not None:
        return torchtext_data

    raise RuntimeError(
        "Không tìm thấy/tải được AG News. Hãy thử một trong các cách sau:\n"
        "1) Cài: pip install datasets\n"
        "2) Hoặc cài torchtext tương thích với PyTorch\n"
        "3) Hoặc tải AG News CSV vào data/ag_news_csv/train.csv và test.csv"
    )


def _limit_examples(
    examples: List[TextExample],
    max_samples: Optional[int],
    seed: int,
) -> List[TextExample]:
    """Giới hạn số mẫu để debug nhanh nhưng vẫn dùng dữ liệu thật."""
    if max_samples is None or max_samples <= 0 or max_samples >= len(examples):
        return examples
    rng = random.Random(seed)
    indices = list(range(len(examples)))
    rng.shuffle(indices)
    selected = indices[:max_samples]
    return [examples[i] for i in selected]


def build_ag_news_dataloaders(
    data_dir: str = "data",
    max_seq_len: int = 128,
    batch_size: int = 32,
    max_vocab_size: int = 10000,
    min_freq: int = 2,
    num_workers: int = 0,
    seed: int = 42,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, Dict[str, int], int]:
    """
    Hàm chính để main_text.py gọi.

    Returns:
        train_loader, val_loader, vocab, num_classes
    """
    train_examples, val_examples = load_ag_news_examples(data_dir=data_dir)

    train_examples = _limit_examples(train_examples, max_train_samples, seed)
    val_examples = _limit_examples(val_examples, max_val_samples, seed)

    vocab = build_vocab(
        (example.text for example in train_examples),
        max_vocab_size=max_vocab_size,
        min_freq=min_freq,
    )

    train_dataset = TextClassificationDataset(
        examples=train_examples,
        vocab=vocab,
        max_seq_len=max_seq_len,
    )
    val_dataset = TextClassificationDataset(
        examples=val_examples,
        vocab=vocab,
        max_seq_len=max_seq_len,
    )

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    labels = [example.label for example in train_examples + val_examples]
    num_classes = 4
    
    print("[INFO] Dataset AG News đã sẵn sàng:")
    print(f"       Train samples: {len(train_dataset):,}")
    print(f"       Val/Test samples: {len(val_dataset):,}")
    print(f"       Vocab size: {len(vocab):,}")
    print(f"       Num classes: {num_classes}")
    print(f"       Max sequence length: {max_seq_len}")

    return train_loader, val_loader, vocab, num_classes
