"""Classifier heads for the Fisher UC present/absent experiment."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PresentHead(nn.Module):
    """Binary present/absent classifier.

    Input is a single 128D scene-query feature. The output logit is positive for
    target-present and non-positive for no-target at the default threshold 0.
    """

    def __init__(self, D=128, hidden=128, p_drop=0.1):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(D, hidden),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, 1),
        )

    def forward(self, g_global):
        return self.ffn(g_global).squeeze(-1)


class GaussianAttrConvPoolFormerHead(nn.Module):
    """Encode top-k Gaussian+UC tokens into one 128D classifier feature.

    The renderer builds one token per top-k Gaussian:
        [query-conditioned g, xyz, scale, rotation, opacity, SH color, UC]

    This head maps each token to 128D, performs local Conv1d pooling along the
    score-ranked top-k sequence, then uses a lightweight CLS transformer.
    """

    def __init__(self, attr_dim, D=128, pooled_tokens=64,
                 num_self_layers=2, num_self_heads=4,
                 ffn_dim=256, p_drop=0.0, kernel_size=5):
        super().__init__()
        self.attr_dim = int(attr_dim)
        self.D = int(D)
        self.pooled_tokens = int(pooled_tokens)
        if self.pooled_tokens < 1:
            raise ValueError("pooled_tokens must be >= 1")
        kernel_size = int(kernel_size)
        if kernel_size < 1:
            raise ValueError("kernel_size must be >= 1")
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd so Conv1d preserves rank length")
        padding = kernel_size // 2

        self.attr_mlp = nn.Sequential(
            nn.Linear(self.attr_dim, D),
            nn.GELU(),
            nn.LayerNorm(D),
            nn.Linear(D, D),
            nn.GELU(),
            nn.LayerNorm(D),
        )
        self.conv_pool = nn.Sequential(
            nn.Conv1d(D, D, kernel_size=kernel_size, padding=padding),
            nn.GELU(),
            nn.AvgPool1d(kernel_size=2, stride=2, ceil_mode=True),
            nn.Conv1d(D, D, kernel_size=kernel_size, padding=padding),
            nn.GELU(),
            nn.AvgPool1d(kernel_size=2, stride=2, ceil_mode=True),
            nn.Conv1d(D, D, kernel_size=3, padding=1),
            nn.GELU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D,
            nhead=num_self_heads,
            dim_feedforward=ffn_dim,
            dropout=p_drop,
            batch_first=True,
            activation="gelu",
        )
        self.self_attn = nn.TransformerEncoder(
            encoder_layer, num_layers=num_self_layers
        )
        self.cls_token = nn.Parameter(torch.randn(1, D) * 0.02)
        self.out_norm = nn.LayerNorm(D)

    def forward(self, attr_topk, return_stats=False):
        if attr_topk.dim() != 2 or attr_topk.shape[-1] != self.attr_dim:
            raise ValueError(
                f"attr_topk must be [K, {self.attr_dim}], got {tuple(attr_topk.shape)}"
            )
        if attr_topk.shape[0] < 1:
            raise ValueError("GaussianAttrConvPoolFormerHead requires K >= 1")

        tokens = self.attr_mlp(attr_topk)                         # [K, D]
        h = tokens.transpose(0, 1).unsqueeze(0)                   # [1, D, K]
        h = self.conv_pool(h)                                     # [1, D, K']
        pre_adaptive_tokens = h.shape[-1]
        if pre_adaptive_tokens > self.pooled_tokens:
            h = F.adaptive_avg_pool1d(h, self.pooled_tokens)
        pooled_tokens = h.squeeze(0).transpose(0, 1)              # [M, D]

        cls = self.cls_token.to(dtype=pooled_tokens.dtype, device=pooled_tokens.device)
        seq = torch.cat([cls, pooled_tokens], dim=0).unsqueeze(0) # [1, 1+M, D]
        seq_out = self.self_attn(seq)
        cls_out = self.out_norm(seq_out[0, 0])
        if return_stats:
            return cls_out, {
                "input_tokens": attr_topk.new_tensor(float(attr_topk.shape[0])),
                "pre_adaptive_tokens": attr_topk.new_tensor(float(pre_adaptive_tokens)),
                "pooled_tokens": attr_topk.new_tensor(float(pooled_tokens.shape[0])),
            }
        return cls_out
