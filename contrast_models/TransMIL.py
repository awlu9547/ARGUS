import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
from operator import mul
from einops import rearrange


class PPEG(nn.Module):
    def __init__(self, embed_dim=512, k=3):
        super(PPEG, self).__init__()
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=k, padding=k//2, groups=embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, N, C = x.shape
        H, W = int(N**0.5), int(N**0.5)
        feat = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        x_ppeg = self.proj(feat).permute(0, 2, 3, 1).reshape(B, H*W, C)
        x = x_ppeg + x
        x = self.norm(x)
        return x


class TransMIL(nn.Module):
    def __init__(self, input_dim=512, num_classes=4, pos_enc='PPEG', dropout=0.25):
        super(TransMIL, self).__init__()
        self.pos_enc = pos_enc
        self.num_classes = num_classes

        # Patch embedding
        self.cls_token = nn.Parameter(torch.randn(1, 1, input_dim))
        self._init_weights(self.cls_token)

        # Position Encoding
        if self.pos_enc == 'PPEG':
            self.pos_enc = PPEG(input_dim)
        else:
            self.pos_enc = nn.PositionalEncoding(input_dim, dropout=dropout)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=8, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Classification head
        self.head = nn.Sequential(
            nn.Linear(input_dim, num_classes)
        )

        self.dropout = nn.Dropout(dropout)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x, return_WSI_feature=False, return_WSI_attn=False):
        """
        Args:
            x: 输入 patch 特征 shape (N, input_dim)
        Returns:
            logits: 分类 logit shape (1, num_classes)
            feature: WSI-level 表征（可选）
            attn_weights: 注意力权重（可选）
        """
        x = x.unsqueeze(0)  # Add batch dim: (1, N, D)

        # 将 cls token 添加到序列开头
        cls_token = self.cls_token.expand(1, -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        # Positional encoding
        if self.pos_enc.__class__.__name__ == 'PositionalEncoding':
            x = self.pos_enc(x)
        elif self.pos_enc.__class__.__name__ == 'PPEG':
            x = self.pos_enc(x)

        # Transformer 编码器
        x = self.transformer(x)

        # 提取 cls token 的输出
        M = x[:, 0]  # (B=1, D)
        logits = self.head(self.dropout(M))

        output = {'logits': logits}

        if return_WSI_feature:
            output['WSI_feature'] = M.cpu().numpy()

        if return_WSI_attn:
            # 获取 transformer 中各层的 attention weights（可选）
            output['WSI_attn'] = self.get_attention_weights(x)

        return output

    def get_attention_weights(self, x):
        """获取 Transformer 中所有 attention heads 的平均权重"""
        B, N, C = x.shape
        attn_weights = []
        for layer in self.transformer.layers:
            qkv = layer.self_attn.in_proj_weight
            q, k, v = torch.chunk(qkv, 3, dim=0)
            q, k, v = [rearrange(mat, '(h d) c -> h 1 d c', h=8) for mat in (q, k, v)]
            q = q @ x.transpose(-2, -1)
            k = k @ x
            attn = (q @ k) * (1.0 / (C // 8)**0.5)
            attn = F.softmax(attn, dim=-1)
            attn_weights.append(attn.detach().cpu())
        return torch.mean(torch.stack(attn_weights), dim=0)