"""Tests for VarifocalLoss and AsymmetricLoss.

Covers:
  - VarifocalLoss: IoU-aware classification, alpha/gamma params
  - AsymmetricLoss: gamma_neg=4.0, gamma_pos=0.0, clip=0.05
"""

import torch
import pytest

from src.losses.varifocal_loss import VarifocalLoss
from src.losses.asymmetric_loss import AsymmetricLoss


# ============================================================================
# Varifocal Loss
# ============================================================================


class TestVarifocalLoss:
    """IoU-aware classification loss (Zhang et al. 2021 CVPR Oral)."""

    def test_loss_finite_and_positive(self):
        loss_fn = VarifocalLoss(alpha=0.75, gamma=2.0)
        pred = torch.randn(8, 4)
        target = torch.zeros(8, 4)
        target[:, 0] = 0.8  # first class has IoU=0.8 for positive samples
        loss = loss_fn(pred, target)
        assert torch.isfinite(loss), "Loss should be finite"
        assert loss.item() > 0, "Loss should be positive"

    def test_perfect_prediction_low_loss(self):
        """Very confident predictions should give low loss."""
        loss_fn = VarifocalLoss(alpha=0.75, gamma=2.0)
        target = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
        # Very confident logits → sigmoid → near 1.0 / 0.0
        pred = torch.tensor([[10.0, -10.0], [-10.0, -10.0]])
        loss = loss_fn(pred, target)
        assert loss.item() < 0.1, "Confident predictions should give low loss"

    def test_worst_prediction_high_loss(self):
        """Completely wrong predictions should give high loss."""
        loss_fn = VarifocalLoss(alpha=0.75, gamma=2.0)
        target = torch.tensor([[0.8, 0.0], [0.0, 0.0]])
        # Completely wrong: predict low where target is high and vice versa
        pred = torch.tensor([[-10.0, 10.0], [10.0, 10.0]])
        loss = loss_fn(pred, target)
        assert loss.item() > 1.0, "Wrong predictions should give high loss"

    def test_alpha_param_affects_loss(self):
        """Different alpha values should produce different losses."""
        target = torch.tensor([[0.8, 0.0], [0.0, 0.0]])
        pred = torch.randn(2, 2)
        loss1 = VarifocalLoss(alpha=0.25, gamma=2.0)(pred, target)
        loss2 = VarifocalLoss(alpha=0.75, gamma=2.0)(pred, target)
        assert loss1 != pytest.approx(loss2.item(), rel=1e-3), "Alpha should affect loss value"

    def test_gamma_param_affects_loss(self):
        """Different gamma values should produce different losses."""
        target = torch.tensor([[0.8, 0.0], [0.0, 0.0]])
        pred = torch.randn(2, 2)
        loss1 = VarifocalLoss(alpha=0.75, gamma=0.0)(pred, target)
        loss2 = VarifocalLoss(alpha=0.75, gamma=2.0)(pred, target)
        assert loss1 != pytest.approx(loss2.item(), rel=1e-3), "Gamma should affect loss value"

    def test_all_negative_samples(self):
        """When all targets are zero (negative), loss should still be defined."""
        loss_fn = VarifocalLoss(alpha=0.75, gamma=2.0)
        pred = torch.randn(4, 3)
        target = torch.zeros(4, 3)
        loss = loss_fn(pred, target)
        assert torch.isfinite(loss), "Loss with all negative should be finite"
        assert loss.item() > 0, "Loss should be positive"

    def test_single_sample(self):
        """Loss should work with single sample."""
        loss_fn = VarifocalLoss(alpha=0.75, gamma=2.0)
        pred = torch.randn(1, 5)
        target = torch.zeros(1, 5)
        target[0, 0] = 0.9
        loss = loss_fn(pred, target)
        assert torch.isfinite(loss)


# ============================================================================
# Asymmetric Loss
# ============================================================================


class TestAsymmetricLoss:
    """Asymmetric loss for multi-label classification (Ridnik et al. 2021 ICCV)."""

    def test_loss_finite_and_positive(self):
        loss_fn = AsymmetricLoss(gamma_neg=4.0, gamma_pos=0.0, clip=0.05)
        pred = torch.randn(8, 4)
        target = (torch.sigmoid(torch.randn(8, 4)) > 0.5).float()
        loss = loss_fn(pred, target)
        assert torch.isfinite(loss), "Loss should be finite"
        assert loss.item() > 0, "Loss should be positive"

    def test_gamma_neg_greater_than_gamma_pos(self):
        """gamma_neg >> gamma_pos means negative examples are down-weighted more."""
        loss_fn = AsymmetricLoss(gamma_neg=4.0, gamma_pos=0.0, clip=0.05)
        pred = torch.tensor([[2.0, 0.0]])
        target = torch.tensor([[1.0, 0.0]])
        loss = loss_fn(pred, target)
        assert torch.isfinite(loss)

    def test_perfect_prediction_low_loss(self):
        """Near-perfect predictions should give low loss."""
        loss_fn = AsymmetricLoss(gamma_neg=4.0, gamma_pos=0.0, clip=0.05)
        # Perfect: correct class logit high, negative class logit very negative
        pred = torch.tensor([[10.0, -10.0]])
        target = torch.tensor([[1.0, 0.0]])
        loss = loss_fn(pred, target)
        assert loss.item() < 0.1, "Near-perfect prediction should have low loss"

    def test_wrong_prediction_high_loss(self):
        """Wrong predictions should give high loss."""
        loss_fn = AsymmetricLoss(gamma_neg=4.0, gamma_pos=0.0, clip=0.05)
        pred = torch.tensor([[-10.0, 10.0]])
        target = torch.tensor([[1.0, 0.0]])
        loss = loss_fn(pred, target)
        assert loss.item() > 1.0, "Wrong prediction should have high loss"

    def test_clip_prevents_extreme_values(self):
        """Clip parameter prevents log(0) by bounding sigmoid probabilities."""
        loss_fn_asym = AsymmetricLoss(gamma_neg=4.0, gamma_pos=0.0, clip=0.05)
        pred = torch.tensor([[100.0, -100.0]])
        target = torch.tensor([[1.0, 0.0]])
        loss_asym = loss_fn_asym(pred, target)
        assert torch.isfinite(loss_asym), "Loss with clip should be finite"

    def test_all_positive(self):
        """All-positive targets should be handled."""
        loss_fn = AsymmetricLoss(gamma_neg=4.0, gamma_pos=0.0, clip=0.05)
        pred = torch.randn(4, 3)
        target = torch.ones(4, 3)
        loss = loss_fn(pred, target)
        assert torch.isfinite(loss)

    def test_all_negative(self):
        """All-negative targets should be handled."""
        loss_fn = AsymmetricLoss(gamma_neg=4.0, gamma_pos=0.0, clip=0.05)
        pred = torch.randn(4, 3)
        target = torch.zeros(4, 3)
        loss = loss_fn(pred, target)
        assert torch.isfinite(loss)

    def test_zero_gamma_pos_gives_focal_positive(self):
        """gamma_pos=0 means positive examples have no focal down-weighting."""
        loss_fn_pos0 = AsymmetricLoss(gamma_neg=4.0, gamma_pos=0.0, clip=0.05)
        loss_fn_pos2 = AsymmetricLoss(gamma_neg=4.0, gamma_pos=2.0, clip=0.05)
        pred = torch.tensor([[1.0, -1.0]])
        target = torch.tensor([[1.0, 0.0]])
        l0 = loss_fn_pos0(pred, target)
        l2 = loss_fn_pos2(pred, target)
        assert l0 != pytest.approx(l2.item(), rel=1e-3), "gamma_pos should affect loss value"
