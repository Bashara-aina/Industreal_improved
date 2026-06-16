"""
Tier 3.9 — Knowledge Distillation Infrastructure
=================================================
Distill from dedicated baselines into the unified multi-task model.

Teachers:
  - YOLOv8m (fine-tuned on IndustReal) → soft detection targets
  - MViTv2 (K400-pretrained) → soft activity logits

Method:
  - Logit distillation: KL divergence at temperature T (soft targets from teacher)
  - Box distillation: MSE on box predictions (localization from teacher)
  - Combined with hard-label loss via configurable weights

The paper's thesis ("one model matches N specialists") is proven when the
distilled unified model matches the specialist baselines.

Usage:
    # Generate teacher predictions (once, offline):
    python src/training/distillation.py --generate --teacher yolov8m \
        --data-path data/IndustReal --output runs/teacher_preds/

    # Train with distillation:
    # Set config: USE_DISTILLATION=True, DISTILL_TEACHER_DET_CKPT=..., etc.
    # Then run train.py normally — MultiTaskLoss.forward() checks these flags.
"""

import argparse
import json
import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ============================================================================
# Distillation Loss Functions
# ============================================================================
def detection_logit_distillation_loss(student_logits: torch.Tensor,
                                       teacher_logits: torch.Tensor,
                                       temperature: float = 3.0,
                                       alpha: float = 0.5) -> torch.Tensor:
    """KL divergence between student and teacher detection logits.

    Args:
        student_logits: [B, N, C] — student raw logits (pre-sigmoid)
        teacher_logits: [B, N, C] — teacher soft targets (pre-sigmoid)
        temperature: smoothing temperature
        alpha: weight of distillation vs 0 (1=full distill, 0=none)

    Returns:
        loss: scalar
    """
    # Soft targets at temperature
    with torch.no_grad():
        teacher_probs = F.sigmoid(teacher_logits / temperature)

    student_probs = F.sigmoid(student_logits / temperature)

    # Binary KL for each class at each location
    # KL(p||q) = p*log(p/q) + (1-p)*log((1-p)/(1-q))
    kl_pos = teacher_probs * torch.log(teacher_probs / (student_probs + 1e-8))
    kl_neg = (1 - teacher_probs) * torch.log((1 - teacher_probs) / (1 - student_probs + 1e-8))
    kl = (kl_pos + kl_neg).mean()

    return alpha * kl * (temperature ** 2)


def activity_distillation_loss(student_logits: torch.Tensor,
                                teacher_logits: torch.Tensor,
                                temperature: float = 3.0,
                                alpha: float = 0.3) -> torch.Tensor:
    """KL divergence between student and teacher activity logits.

    Args:
        student_logits: [B, num_classes] — student raw logits
        teacher_logits: [B, num_classes] — teacher soft targets
        temperature: smoothing temperature

    Returns:
        loss: scalar
    """
    with torch.no_grad():
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)

    # KL(p||q) = sum(p * log(p/q)) = sum(p * (log(p) - log(q)))
    kl = F.kl_div(
        student_log_probs,
        teacher_probs.detach(),
        reduction='batchmean',
        log_target=False,
    )

    return alpha * kl * (temperature ** 2)


def box_distillation_loss(student_boxes: torch.Tensor,
                           teacher_boxes: torch.Tensor,
                           alpha: float = 0.2) -> torch.Tensor:
    """MSE between student and teacher box predictions.

    Only applied where the teacher has confident detections.

    Args:
        student_boxes: [B, N, 4] — student predicted boxes
        teacher_boxes: [B, N, 4] — teacher predicted boxes
        alpha: weight of distillation

    Returns:
        loss: scalar
    """
    mask = (teacher_boxes.sum(dim=-1) > 0).float()  # [B, N]
    if mask.sum() == 0:
        return torch.tensor(0.0)

    diff = (student_boxes - teacher_boxes.detach()) ** 2
    loss = (diff.sum(dim=-1) * mask).sum() / mask.sum()

    return alpha * loss


# ============================================================================
# Teacher Prediction Generator (offline, run once)
# ============================================================================
class TeacherPredictionGenerator:
    """Generate and cache teacher predictions for distillation.

    This runs the teacher model (YOLOv8m, MViTv2, etc.) over the training set
    once, saving soft predictions to disk. The student then loads these
    predictions during training (no teacher forward pass needed each epoch).
    """

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._predictions: Dict[str, Dict] = {}

    def add_sample(self, frame_id: str,
                   teacher_det_logits: Optional[np.ndarray] = None,
                   teacher_det_boxes: Optional[np.ndarray] = None,
                   teacher_act_logits: Optional[np.ndarray] = None):
        """Adda teacher prediction for one frame."""
        self._predictions[frame_id] = {
            'det_logits': teacher_det_logits,
            'det_boxes': teacher_det_boxes,
            'act_logits': teacher_act_logits,
        }

    def save(self):
        """Save all predictions to disk as .pkl files."""
        np.savez_compressed(
            self.output_dir / 'teacher_predictions.npz',
            **{k: {
                kk: vv for kk, vv in v.items() if vv is not None
            } for k, v in self._predictions.items()}
        )
        # Also save metadata
        meta = {
            'num_frames': len(self._predictions),
            'output_dims': {
                'det_logits': self._predictions[list(self._predictions.keys())[0]]['det_logits'].shape
                if self._predictions and list(self._predictions.values())[0].get('det_logits') is not None
                else None,
                'act_logits': self._predictions[list(self._predictions.keys())[0]]['act_logits'].shape
                if self._predictions and list(self._predictions.values())[0].get('act_logits') is not None
                else None,
            }
        }
        with open(self.output_dir / 'teacher_meta.json', 'w') as f:
            json.dump(meta, f, indent=2)

        logger.info(f"[Distill] Saved {len(self._predictions)} teacher predictions "
                    f"to {self.output_dir}")


class TeacherPredictionLoader:
    """Load cached teacher predictions during training."""

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self._data: Optional[Dict] = None
        self._meta: Optional[Dict] = None

    def load(self):
        """Load the prediction cache into memory."""
        self._data = np.load(self.cache_dir / 'teacher_predictions.npz',
                             allow_pickle=True)
        with open(self.cache_dir / 'teacher_meta.json', 'r') as f:
            self._meta = json.load(f)
        logger.info(f"[Distill] Loaded {self._meta['num_frames']} teacher predictions")

    def get(self, frame_id: str) -> Optional[Dict]:
        """Get teacher predictions for a specific frame."""
        if self._data is None:
            self.load()
        try:
            return dict(self._data[frame_id].item())
        except (KeyError, AttributeError):
            return None


# ============================================================================
# Combined Distillation Module (callable from MultiTaskLoss)
# ============================================================================
class DistillationLoss(nn.Module):
    """Distillation loss wrapper configured from config flags.

    Used transparently: if USE_DISTILLATION=False, this module is a no-op.
    """

    def __init__(self, temperature: float = 3.0,
                 det_weight: float = 0.5, act_weight: float = 0.3):
        super().__init__()
        self.temperature = temperature
        self.det_weight = det_weight
        self.act_weight = act_weight
        self.teacher_loader: Optional[TeacherPredictionLoader] = None

    def set_teacher_cache(self, cache_dir: str):
        self.teacher_loader = TeacherPredictionLoader(cache_dir)
        self.teacher_loader.load()

    def forward(self, student_outputs: Dict[str, torch.Tensor],
                teacher_outputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute distillation loss components.

        Args:
            student_outputs: model forward outputs (cls_preds, activity, etc.)
            teacher_outputs: teacher predictions (loaded from cache)

        Returns:
            total_distill_loss, metrics_dict
        """
        total = torch.tensor(0.0)
        metrics = {}

        # Detection distillation
        if 'det_logits' in teacher_outputs and 'cls_preds' in student_outputs:
            det_loss = detection_logit_distillation_loss(
                student_outputs['cls_preds'],
                teacher_outputs['det_logits'],
                self.temperature, self.det_weight,
            )
            total = total + det_loss
            metrics['distill_det'] = det_loss.item()

        if 'det_boxes' in teacher_outputs and 'reg_preds' in student_outputs:
            box_loss = box_distillation_loss(
                student_outputs['reg_preds'],
                teacher_outputs['det_boxes'],
                self.det_weight * 0.5,
            )
            total = total + box_loss
            metrics['distill_box'] = box_loss.item()

        # Activity distillation
        if 'act_logits' in teacher_outputs and 'act_logits' in student_outputs:
            act_loss = activity_distillation_loss(
                student_outputs['act_logits'],
                teacher_outputs['act_logits'],
                self.temperature, self.act_weight,
            )
            total = total + act_loss
            metrics['distill_act'] = act_loss.item()

        metrics['distill_total'] = total.item()
        return total, metrics


# ============================================================================
# CLI — Generate teacher predictions
# ============================================================================
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Tier 3.9: Knowledge Distillation')
    ap.add_argument('--generate', action='store_true', help='Generate teacher predictions')
    ap.add_argument('--output', type=str, default='runs/teacher_preds',
                    help='Output directory for teacher predictions')
    ap.add_argument('--data-path', type=str, help='Path to dataset')

    args = ap.parse_args()

    if args.generate:
        gen = TeacherPredictionGenerator(args.output)
        logger.info("[Distill] Teacher prediction generation — run your teacher model "
                    "over the dataset and call gen.add_sample() per frame")
        logger.info("[Distill] See module docstring for integration instructions")
    else:
        ap.print_help()
