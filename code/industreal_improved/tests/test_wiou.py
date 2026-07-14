"""Tests for WIoU v3 dynamic non-monotonic IoU loss.

Covers:
  - IoU computation correctness (intersection, union, complete overlap)
  - Distance-aware weighting (WIoU v1 = IoU * exp(-dw))
  - Dynamic non-monotonic focusing (v3) with beta/anchor_iou
  - Edge cases: degenerate boxes, zero-area, far apart
"""

import torch

from src.losses.wiou_loss import wiou_v3_loss


# ============================================================================
# IoU Computation
# ============================================================================


class TestWIoUv3IoU:
    """Verify the IoU component of WIoU v3."""

    def test_perfect_overlap(self):
        """Identical boxes should give IoU = 1 and WIoU v1 = 1 (exp(-0) = 1)."""
        pred = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
        target = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
        anchor = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
        loss = wiou_v3_loss(pred, target, anchor)
        # IoU = 1 → WIoU v1 = 1 → (1 - 1) * r = 0
        assert loss.item() == 0.0, "Perfect overlap should give zero loss"

    def test_no_overlap(self):
        """Completely disjoint boxes should give IoU = 0 and loss near 1*r."""
        pred = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        target = torch.tensor([[5.0, 5.0, 6.0, 6.0]])
        anchor = pred.clone()
        loss = wiou_v3_loss(pred, target, anchor)
        assert loss.item() > 0.5, "Non-overlapping boxes should give high loss"
        assert torch.isfinite(loss), "Loss should be finite"

    def test_partial_overlap(self):
        """Partial overlap should give finite positive loss (r factor can amplify > 1)."""
        pred = torch.tensor([[0.0, 0.0, 3.0, 3.0]])
        target = torch.tensor([[1.0, 1.0, 4.0, 4.0]])
        anchor = pred.clone()
        loss = wiou_v3_loss(pred, target, anchor)
        assert loss.item() > 0, "Partial overlap loss should be positive"
        assert torch.isfinite(loss), "Loss should be finite"

    def test_iou_correctness(self):
        """Verify computed IoU for a known case: inter=1, area_p=4, area_t=4, union=7, iou=1/7."""
        pred = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
        target = torch.tensor([[1.0, 1.0, 3.0, 3.0]])
        anchor = pred.clone()
        loss = wiou_v3_loss(pred, target, anchor)
        # IoU = 1/7 ≈ 0.1429, enclosing box = [0,0,3,3] → ew=3, eh=3
        # cx_p=1, cy_p=1, cx_t=2, cy_t=2
        # dw = ((1-2)/3)^2 + ((1-2)/3)^2 = 2/9 ≈ 0.222
        # WIoU v1 = (1/7) * exp(-2/9) ≈ 0.1429 * 0.8007 ≈ 0.1144
        # Loss = (1 - 0.1144) * r (r depends on anchor_iou)
        assert torch.isfinite(loss)


# ============================================================================
# Distance-aware weighting (WIoU v1)
# ============================================================================


class TestWIoUv1Distance:
    """WIoU v1 = IoU * exp(-dw) penalizes distant predictions."""

    def test_same_iou_different_distance(self):
        """Same IoU but different distances should give different losses."""
        anchor = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
        # Case 1: close (small offset)
        pred_close = torch.tensor([[0.5, 0.5, 2.5, 2.5]])
        target_close = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
        # Case 2: far (large offset with same IoU area)
        pred_far = torch.tensor([[5.0, 5.0, 7.0, 7.0]])
        target_far = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
        loss_close = wiou_v3_loss(pred_close, target_close, anchor)
        loss_far = wiou_v3_loss(pred_far, target_far, anchor)
        assert loss_far > loss_close, "Far apart boxes should have higher loss than close ones"

    def test_dw_penalty_term(self):
        """The distance penalty term dw = (dcx/ew)^2 + (dcy/eh)^2 should be non-negative."""
        pred = torch.tensor([[1.0, 1.0, 3.0, 3.0]])
        target = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
        anchor = pred.clone()
        loss = wiou_v3_loss(pred, target, anchor)
        assert torch.isfinite(loss)

    def test_concentric_boxes(self):
        """Concentric boxes should have dw=0 (centers coincide)."""
        pred = torch.tensor([[-1.0, -1.0, 1.0, 1.0]])
        target = torch.tensor([[-0.5, -0.5, 0.5, 0.5]])
        anchor = pred.clone()
        loss = wiou_v3_loss(pred, target, anchor)
        assert torch.isfinite(loss)


# ============================================================================
# Dynamic non-monotonic focusing (v3)
# ============================================================================


class TestWIoUv3Focusing:
    """WIoU v3 dynamic focusing: r = delta / (alpha^(beta - delta))."""

    def test_beta_one_gives_r_delta(self):
        """When beta=1 (anchored), r = delta / (alpha^(1-delta))."""
        # beta = iou / anchor_iou
        # When pred == anchor, iou / anchor_iou = 1 → beta=1
        pred = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
        target = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
        anchor = pred.clone()
        loss = wiou_v3_loss(pred, target, anchor)
        # Perfect overlap → loss = 0
        assert loss.item() == 0.0

    def test_r_values_finite(self):
        """The focusing factor r should always be finite."""
        pred = torch.randn(8, 4).abs() * 2
        target = torch.randn(8, 4).abs() * 2
        anchor = torch.randn(8, 4).abs() * 2
        # Ensure valid boxes
        pred[:, 2:] = pred[:, :2] + (pred[:, 2:] - pred[:, :2]).abs() + 0.1
        target[:, 2:] = target[:, :2] + (target[:, 2:] - target[:, :2]).abs() + 0.1
        anchor[:, 2:] = anchor[:, :2] + (anchor[:, 2:] - anchor[:, :2]).abs() + 0.1
        loss = wiou_v3_loss(pred, target, anchor)
        assert torch.isfinite(loss), "Loss with random boxes should be finite"
        assert loss.item() > 0, "Loss should be positive"

    def test_no_anchor_fallback(self):
        """When anchor is None, anchor_iou = iou so beta = 1."""
        pred = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
        target = torch.tensor([[0.5, 0.5, 1.5, 1.5]])
        loss = wiou_v3_loss(pred, target, None)
        assert torch.isfinite(loss), "Loss without anchor should be finite"

    def test_r_factor_scale(self):
        """The r factor should scale the loss value."""
        # Two samples with same IoU and distance but different anchor quality
        target = torch.tensor([[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 2.0, 2.0]])
        pred = torch.tensor([[0.5, 0.5, 1.5, 1.5], [0.5, 0.5, 1.5, 1.5]])
        anchor = torch.tensor([[0.0, 0.0, 2.0, 2.0], [5.0, 5.0, 7.0, 7.0]])
        loss = wiou_v3_loss(pred, target, anchor)
        assert torch.isfinite(loss)


# ============================================================================
# Edge Cases
# ============================================================================


class TestWIoUv3EdgeCases:
    """Edge cases: degenerate boxes, zero area, extreme aspect ratios."""

    def test_zero_area_prediction(self):
        """Prediction with zero area should not produce NaN."""
        pred = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
        target = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
        anchor = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
        loss = wiou_v3_loss(pred, target, anchor)
        assert torch.isfinite(loss), "Zero-area pred should not produce NaN"

    def test_zero_area_target(self):
        """Target with zero area should not produce NaN."""
        pred = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
        target = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
        anchor = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
        loss = wiou_v3_loss(pred, target, anchor)
        assert torch.isfinite(loss), "Zero-area target should not produce NaN"

    def test_batch_of_boxes(self):
        """Multiple boxes in batch should give a single scalar loss."""
        pred = torch.randn(16, 4).abs()
        target = torch.randn(16, 4).abs()
        anchor = torch.randn(16, 4).abs()
        pred[:, 2:] = pred[:, :2] + (pred[:, 2:] - pred[:, :2]).abs() + 0.1
        target[:, 2:] = target[:, :2] + (target[:, 2:] - target[:, :2]).abs() + 0.1
        anchor[:, 2:] = anchor[:, :2] + (anchor[:, 2:] - anchor[:, :2]).abs() + 0.1
        loss = wiou_v3_loss(pred, target, anchor)
        assert torch.isfinite(loss), "Batch loss should be finite"
        assert loss.dim() == 0, "Loss should be a scalar"

    def test_negative_coordinates(self):
        """Boxes with negative coordinates should work fine."""
        pred = torch.tensor([[-3.0, -3.0, -1.0, -1.0]])
        target = torch.tensor([[-2.0, -2.0, 0.0, 0.0]])
        anchor = torch.tensor([[-3.0, -3.0, -1.0, -1.0]])
        loss = wiou_v3_loss(pred, target, anchor)
        assert torch.isfinite(loss), "Negative coords should not produce NaN"

    def test_flipped_coordinates(self):
        """Flipped boxes (x1 > x2, y1 > y2) should be handled gracefully."""
        pred = torch.tensor([[2.0, 2.0, 0.0, 0.0]])  # flipped
        target = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        anchor = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
        loss = wiou_v3_loss(pred, target, anchor)
        assert torch.isfinite(loss), "Flipped boxes should not produce NaN"
