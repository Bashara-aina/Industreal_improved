"""Optimizer and scheduler builders — aliased for external test fixtures."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    LinearLR,
    OneCycleLR,
    SequentialLR,
)
from src import config as C
from src.models.model import POPWMultiTaskModel


def build_optimizer(model: torch.nn.Module) -> torch.optim.Optimizer:
    """Build AdamW optimizer with differential LR for backbone/heads/bias.

    Per paper §Implementation:
      - Backbone LR = 5e-5 (0.01× head LR with fine-tuning, frozen for linear probe)
      - Head LR = 5e-4
      - Weight decay = 5e-2 (bias/norm excluded)
    """
    # [FREEZE_BACKBONE] Freeze entire backbone when True (linear probe mode)
    if bool(getattr(C, 'FREEZE_BACKBONE', True)):
        for name, param in model.named_parameters():
            if name.startswith('backbone.'):
                param.requires_grad = False

    backbone_params, head_params, no_decay_params = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith('backbone.'):
            backbone_params.append(param)
        elif 'bias' in name or 'norm' in name or 'LayerNorm' in name or 'GroupNorm' in name:
            no_decay_params.append(param)
        else:
            head_params.append(param)

    _bb_mult = float(getattr(C, 'BACKBONE_LR_MULT', 0.01))
    backbone_lr = C.BASE_LR * _bb_mult
    head_lr = C.BASE_LR
    no_decay_lr = head_lr * 0.3

    # Per paper §Implementation Table: AdamW (β₁=0.9, β₂=0.999)
    param_groups = [
        {'params': backbone_params, 'lr': backbone_lr, 'weight_decay': C.WEIGHT_DECAY},
        {'params': head_params,     'lr': head_lr,     'weight_decay': C.WEIGHT_DECAY},
        {'params': no_decay_params, 'lr': no_decay_lr, 'weight_decay': 0.0},
    ]
    return AdamW(param_groups, betas=(0.9, 0.999))


def build_scheduler(optimizer: torch.optim.Optimizer):
    """Build scheduler per paper: Warmup (2 ep) → OneCycleLR."""
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=C.WARMUP_EPOCHS)
    if getattr(C, 'ONE_CYCLE_LR', False):
        scheduler = OneCycleLR(
            optimizer,
            max_lr=[pg['lr'] for pg in optimizer.param_groups],
            total_steps=C.EPOCHS - C.WARMUP_EPOCHS,
            pct_start=0.3,
            div_factor=10.0,
            final_div_factor=1000.0,
        )
    elif getattr(C, 'USE_COSINE_ANNEALING', False):
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=C.T_0, T_mult=C.T_mult, eta_min=1e-6
        )
    else:
        scheduler = CosineAnnealingLR(
            optimizer, T_max=C.EPOCHS - C.WARMUP_EPOCHS, eta_min=1e-6
        )
    return SequentialLR(optimizer, [warmup, scheduler], milestones=[C.WARMUP_EPOCHS])