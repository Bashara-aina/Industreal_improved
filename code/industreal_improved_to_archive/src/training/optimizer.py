"""Optimizer and scheduler builders — aliased for external test fixtures."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    LinearLR,
    SequentialLR,
)
from src import config as C
from src.models.model import POPWMultiTaskModel


def build_optimizer(model: torch.nn.Module) -> torch.optim.Optimizer:
    """Build AdamW optimizer with differential LR for backbone/heads/bias."""
    backbone_params, head_params, bias_params = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(ln in name for ln in ['layer0', 'layer1', 'layer2', 'layer3', 'layer4']):
            backbone_params.append(param)
        elif 'bias' in name:
            bias_params.append(param)
        else:
            head_params.append(param)

    backbone_lr = C.BASE_LR * 0.1
    head_lr = C.BASE_LR
    bias_lr = head_lr * 0.3

    param_groups = [
        {'params': backbone_params, 'lr': backbone_lr},
        {'params': head_params,     'lr': head_lr},
        {'params': bias_params,     'lr': bias_lr},
    ]
    return AdamW(param_groups, weight_decay=C.WEIGHT_DECAY)


def build_scheduler(optimizer: torch.optim.Optimizer):
    """Build SequentialLR with warmup + cosine annealing."""
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=C.WARMUP_EPOCHS)
    if getattr(C, 'USE_COSINE_ANNEALING', False):
        cosine = CosineAnnealingWarmRestarts(
            optimizer, T_0=C.T_0, T_mult=C.T_mult, eta_min=1e-6
        )
    else:
        cosine = CosineAnnealingLR(
            optimizer, T_max=C.EPOCHS - C.WARMUP_EPOCHS, eta_min=1e-6
        )
    return SequentialLR(optimizer, [warmup, cosine], milestones=[C.WARMUP_EPOCHS])