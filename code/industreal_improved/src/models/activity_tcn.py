"""Minimal TCN for per-frame activity classification (Opus 141 ACT-ARCH-4 Phase 1).

Frozen ConvNeXt-Tiny backbone features (768-dim) fed into a small TCN
with dilations [1, 2, 4]. Conv1D over time with residual connections.
"""

import torch.nn as nn
import torch.nn.functional as F


class TCNBlock(nn.Module):
    """Temporal Conv1D block with dilated convolution + residual."""

    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.shortcut = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        # x: [B, T, C]
        x = x.transpose(1, 2)  # [B, C, T]
        residual = self.shortcut(x)
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        return F.gelu(out + residual).transpose(1, 2)  # [B, T, C]


class ActivityTCN(nn.Module):
    """Multi-task activity head: per-frame ConvNeXt features -> TCN -> classifier.

    Phase 1 of TCN+ViT (Opus 141 ACT-ARCH-4): minimal Conv1D only.
    Receptive field = 1 + 2 + 4 = 7 frames at kernel_size=3.
    """

    def __init__(self, in_dim=768, num_classes=69, hidden=256, levels=3):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden)
        dilations = [2**i for i in range(levels)]  # 1, 2, 4
        self.blocks = nn.ModuleList(
            [TCNBlock(hidden, hidden, kernel_size=3, dilation=d, dropout=0.1) for d in dilations]
        )
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, x):
        # x: [B, T, 768]
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.classifier(x)  # [B, T, num_classes]
