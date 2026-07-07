"""TCN+ViT for per-frame activity classification (Opus 141 ACT-ARCH-4 Phase 2).

Two-stream multi-head architecture:
  Stream 1 (TCN): multi-scale dilated Conv1D [1,2,4,8], receptive field ~15 frames
  Stream 2 (TemporalViT): 4-layer transformer encoder, 8 heads, learned positional embeddings
  Fusion: late concatenation -> MLP -> classifier

Frozen ConvNeXt-Tiny backbone features (768-dim) feed into both streams.
"""
import torch
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
        x = x.transpose(1, 2)  # [B, C, T]
        residual = self.shortcut(x)
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        return F.gelu(out + residual).transpose(1, 2)


class TemporalViT(nn.Module):
    """ViT over per-frame features with learned positional embeddings.

    Treats each frame as a token; applies multi-head self-attention
    over the temporal sequence (T=16 tokens per clip).
    """
    def __init__(self, in_dim=768, embed_dim=512, num_heads=8, num_layers=4, max_len=32):
        super().__init__()
        self.proj = nn.Linear(in_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
            dropout=0.1, activation='gelu', batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: [B, T, 768]
        B, T, C = x.shape
        x = self.proj(x) + self.pos_embed[:, :T]
        x = self.encoder(x)
        return self.norm(x)  # [B, T, embed_dim]


class ActivityTCNViT(nn.Module):
    """Two-stream activity head: TCN + TemporalViT + late fusion.

    TCN: dilated Conv1D [1,2,4,8], receptive field ~15 frames
    ViT: 4 layers, 8 heads, dim 512, learned positional embeddings
    Fusion: concat TCN-pool + ViT-pool -> MLP -> classifier
    """
    def __init__(self, in_dim=768, num_classes=69, tcn_hidden=256, vit_dim=512, fusion_hidden=512):
        super().__init__()
        self.tcn_input = nn.Linear(in_dim, tcn_hidden)
        dilations = [1, 2, 4, 8]
        self.tcn_blocks = nn.ModuleList([
            TCNBlock(tcn_hidden, tcn_hidden, kernel_size=3, dilation=d, dropout=0.1)
            for d in dilations
        ])
        self.vit = TemporalViT(in_dim, vit_dim, num_heads=8, num_layers=4)
        self.fusion = nn.Sequential(
            nn.Linear(tcn_hidden + vit_dim, fusion_hidden),
            nn.LayerNorm(fusion_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_hidden, num_classes)
        )

    def forward(self, x):
        # TCN stream
        tcn_x = self.tcn_input(x)
        for block in self.tcn_blocks:
            tcn_x = block(tcn_x)
        tcn_pool = tcn_x.mean(dim=1)  # [B, tcn_hidden]

        # ViT stream
        vit_x = self.vit(x)  # [B, T, vit_dim]
        vit_pool = vit_x.mean(dim=1)  # [B, vit_dim]

        # Late fusion
        fused = torch.cat([tcn_pool, vit_pool], dim=-1)
        return self.fusion(fused)  # [B, num_classes]