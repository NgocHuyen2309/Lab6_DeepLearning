import torch
import torch.nn as nn

# ==============================================================================
# 1. PATCH EMBEDDING (Băm ảnh thành các mảnh và Ánh xạ sang Vector)
# ==============================================================================
class PatchEmbedding(nn.Module):
    """
    Theo bài báo, ảnh 2D (H x W x C) sẽ được cắt thành các patches 2D, 
    sau đó duỗi thẳng và nhân với ma trận tuyến tính E để tạo thành các Patch Embeddings.
    Cách tối ưu nhất trong PyTorch là dùng Conv2d với kernel_size = stride = patch_size.
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Lớp Convolution đóng vai trò Linear Projection (Ánh xạ tuyến tính E)
        self.proj = nn.Conv2d(
            in_channels, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # x: [B, 3, 224, 224] -> proj(x): [B, 768, 14, 14]
        x = self.proj(x)
        
        # Duỗi phẳng 2 chiều không gian (14x14 = 196 patches)
        # flatten(2) -> [B, 768, 196]
        # transpose -> [B, 196, 768] (Đúng chuẩn [Batch, Sequence_Length, Embed_Dim])
        x = x.flatten(2).transpose(1, 2)
        return x
# ==============================================================================
# 2. MULTI-HEAD SELF-ATTENTION (MSA)
# ==============================================================================
class Attention(nn.Module):
    """
    Cài đặt cơ chế Scaled Dot-Product Attention:
    Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
    """
    def __init__(self, dim=768, num_heads=12, qkv_bias=True, dropout_rate=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5 # Hệ số 1 / sqrt(d_k)
        
        # Gộp chung Q, K, V vào 1 lớp Linear để tăng tốc độ tính toán phần cứng
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        B, N, C = x.shape
        # Lấy Q, K, V và chia thành nhiều Head
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] # Mỗi Tensor có shape: [B, Heads, N, Head_Dim]
        
        # Tính Năng lượng Attention (Q * K^T) * scale
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Phân bố Softmax
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Nhân với Value (V) và gộp các Heads lại
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        # Đi qua Linear Output
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
# ==============================================================================
# 3. TRANSFORMER ENCODER BLOCK & MLP
# ==============================================================================
class MLP(nn.Module):
    """Mạng truyền thẳng (Feed Forward) đặt cuối mỗi Transformer Block"""
    def __init__(self, in_features, hidden_features, act_layer=nn.GELU, drop=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    """Một khối Transformer Encoder hoàn chỉnh (Gồm LayerNorm, MSA, MLP và Residual Connection)"""
    def __init__(self, dim=768, num_heads=12, mlp_ratio=4.0, qkv_bias=True, drop=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, dropout_rate=drop)
        self.norm2 = nn.LayerNorm(dim)
        
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=hidden_features, drop=drop)

    def forward(self, x):
        # Chú ý: Ở ViT, LayerNorm được đặt TRƯỚC Attention/MLP (Pre-Norm)
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
# ==============================================================================
# 4. VISION TRANSFORMER (TỔNG THỂ)
# ==============================================================================
class VisionTransformer(nn.Module):
    """
    Ráp nối toàn bộ các thành phần: Patch Embedding -> CLS Token -> Positional Encoding 
    -> Transformer Blocks -> MLP Head (Classification)
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=1000, 
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, drop_rate=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # Khởi tạo [CLS] Token (Dùng để đại diện cho toàn bộ bức ảnh khi phân loại)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Khởi tạo Positional Embeddings (Cộng thêm thông tin vị trí không gian cho patches)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Xây dựng các lớp Transformer (Depth = 12 cho bản ViT-Base)
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, drop=drop_rate)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Head phân loại (Classification Head)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Khởi tạo trọng số ngẫu nhiên ban đầu
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B = x.shape[0]
        
        # 1. Băm ảnh và nhúng
        x = self.patch_embed(x)
        
        # 2. Gắn [CLS] Token vào đầu chuỗi
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # 3. Cộng Positional Embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # 4. Đi qua các Transformer Blocks
        for block in self.blocks:
            x = block(x)
            
        x = self.norm(x)
        
        # 5. Lấy output của [CLS] token (nằm ở vị trí index 0) đưa vào phân loại
        cls_output = x[:, 0]
        out = self.head(cls_output)
        
        return out