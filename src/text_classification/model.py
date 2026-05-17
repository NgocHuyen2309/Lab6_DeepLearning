import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class PositionalEncoding(nn.Module):
    """
    Mã hóa vị trí sử dụng hàm Sine và Cosine.
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Shape của pe: [1, max_len, d_model]
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor đầu vào có shape [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return x

class MultiHeadAttentionFromScratch(nn.Module):
    """
    Cơ chế Multi-Head Attention xây dựng hoàn toàn từ các phép chiếu tuyến tính.
    """
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model phải chia hết cho n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        # Các lớp Linear để chiếu Q, K, V
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        
        # Lớp Linear đầu ra
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None):
        batch_size = q.size(0)
        
        # 1. Thực hiện Linear Projection và chia thành n_heads
        # Shape: [batch_size, seq_len, d_model] -> [batch_size, n_heads, seq_len, d_head]
        Q = self.q_linear(q).view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_linear(k).view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_linear(v).view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        
        # 2. Scaled Dot-Product Attention
        # scores shape: [batch_size, n_heads, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        
        if mask is not None:
            # Mask những vị trí padding (giá trị 0) bằng -1e9 để softmax tiến về 0
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention_weights = F.softmax(scores, dim=-1)
        
        # context shape: [batch_size, n_heads, seq_len, d_head]
        context = torch.matmul(attention_weights, V)
        
        # 3. Nối các heads lại (Concatenation)
        # Shape: [batch_size, seq_len, d_model]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # 4. Chiếu qua lớp Linear cuối cùng
        output = self.out_linear(context)
        
        return output, attention_weights

class PositionWiseFeedForward(nn.Module):
    """Mạng Feed-Forward phi tuyến cho từng vị trí token."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerEncoderLayer(nn.Module):
    """Một khối (Block) của Transformer Encoder."""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttentionFromScratch(d_model, n_heads)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Sub-layer 1: Multi-Head Attention -> Add & Norm
        attn_out, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_out))
        
        # Sub-layer 2: Feed Forward -> Add & Norm
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))
        
        return x

class TextTransformer(nn.Module):
    """Kiến trúc Transformer phân loại văn bản hoàn chỉnh."""
    def __init__(self, vocab_size: int, num_classes: int, d_model: int = 256, 
                 n_heads: int = 8, d_ff: int = 512, num_layers: int = 4, max_seq_len: int = 512, dropout: float = 0.1):
        super().__init__()
        # Lớp Embedding: chuyển index từ thành Vector
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)
        
        # Các lớp Encoder (Tạo danh sách các block)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])
        
        # Lớp phân loại cuối cùng
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x shape: [batch_size, seq_len]
        # Xử lý mask để đưa vào Attention (chuyển sang dạng [batch_size, 1, 1, seq_len])
        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)
            
        out = self.embedding(x)
        out = self.pos_encoder(out)
        
        for layer in self.layers:
            out = layer(out, padding_mask)
            
        # Global Average Pooling: Lấy trung bình thông tin tất cả các từ trong câu
        out = out.mean(dim=1) 
        
        # Đưa qua bộ phân loại
        logits = self.classifier(out)
        return logits