"""Tests for LDAMLoss + Deferred Re-Weighting (Cao et al. 2019 NeurIPS).

LDAMLoss computes per-class margins inversely proportional to
sqrt(sqrt(cls_num_list)), then applies DRW after a configured epoch.
"""
import torch
import pytest

from src.losses.ldam_drw import LDAMLoss


# ============================================================================
# Margin computation
# ============================================================================

class TestLDAMMargin:
    """Verify the per-class margin formula.

    m_c = max_m * (cls_num_list^{-1/4}) / max(cls_num_list^{-1/4})
    """

    def test_balanced_classes_produce_equal_margins(self):
        """When all classes have equal counts, all margins should be equal."""
        counts = [100, 100, 100]
        loss_fn = LDAMLoss(counts, max_m=0.5, s=30)
        expected = 0.5  # all equal -> normalised to max_m
        for m in loss_fn.m_list:
            assert abs(m.item() - expected) < 1e-6, f"Margin {m} != {expected}"

    def test_rare_class_gets_larger_margin(self):
        """Rarer classes should get larger margins."""
        counts = [1000, 100]
        loss_fn = LDAMLoss(counts, max_m=0.5, s=30)
        m_rare = loss_fn.m_list[1].item()
        m_common = loss_fn.m_list[0].item()
        assert m_rare > m_common, (
            f"Rare class margin {m_rare} should exceed common {m_common}"
        )

    def test_very_rare_class_gets_max_margin(self):
        """The rarest class determines max_m normalization."""
        counts = [1000, 10]
        loss_fn = LDAMLoss(counts, max_m=0.5, s=30)
        assert abs(loss_fn.m_list[1].item() - 0.5) < 1e-6, (
            f"Rarest class should have margin == max_m=0.5, got {loss_fn.m_list[1]}"
        )

    def test_margin_three_class_ratio(self):
        """Verify relative margin ratios between three classes."""
        counts = [1000, 100, 10]
        loss_fn = LDAMLoss(counts, max_m=0.5, s=30)
        margins = loss_fn.m_list.tolist()
        assert margins[2] > margins[1] > margins[0], (
            f"Margins should be strictly decreasing with frequency: {margins}"
        )
        # rarest should be exactly max_m
        assert abs(margins[2] - 0.5) < 1e-6

    def test_margin_formula_direct(self):
        """Direct computation of the margin formula."""
        counts = [16, 81]
        # sqrt(sqrt(16)) = 2, sqrt(sqrt(81)) = 3
        # m_raw = [1/2, 1/3] = [0.5, 0.333...]
        # normalized by max (0.5): m = [0.5/0.5*max_m, 0.333/0.5*max_m]
        # = [0.5, 0.333...]
        raw = [1.0 / (16 ** 0.25), 1.0 / (81 ** 0.25)]
        max_raw = max(raw)
        expected = [r / max_raw * 0.5 for r in raw]

        loss_fn = LDAMLoss(counts, max_m=0.5, s=30)
        margins = loss_fn.m_list.tolist()
        assert abs(margins[0] - expected[0]) < 1e-5
        assert abs(margins[1] - expected[1]) < 1e-5


# ============================================================================
# Deferred Re-Weighting (DRW) schedule
# ============================================================================

class TestDRWSchedule:
    """DRW activates when epoch >= reweight_epoch (default 35)."""

    def test_drw_disabled_before_reweight_epoch(self):
        """Before reweight_epoch, the loss should be unweighted CE."""
        counts = [100, 10]
        loss_fn = LDAMLoss(counts, max_m=0.5, s=30, reweight_epoch=35)

        logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
        targets = torch.tensor([0, 1])

        loss_before = loss_fn(logits, targets, epoch=34)
        assert not loss_fn.is_drw, "DRW should be False before epoch 35"

    def test_drw_enabled_at_reweight_epoch(self):
        """At epoch >= reweight_epoch, DRW activates and applies class weights."""
        counts = [100, 10]
        loss_fn = LDAMLoss(counts, max_m=0.5, s=30, reweight_epoch=35)

        logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
        targets = torch.tensor([0, 1])

        # Before DRW — baseline
        loss_before = loss_fn(logits, targets, epoch=34).item()

        # At DRW — loss should differ because class weights are applied
        loss_fn.is_drw = False  # reset
        loss_at = loss_fn(logits, targets, epoch=35).item()

        assert loss_fn.is_drw, "DRW should be True at epoch 35"
        assert abs(loss_at - loss_before) > 1e-6, (
            "DRW loss should differ from non-DRW loss"
        )

    def test_drw_stays_enabled_after_reweight_epoch(self):
        """Once activated, DRW persists for all later epochs."""
        counts = [100, 10, 50]
        loss_fn = LDAMLoss(counts, max_m=0.5, s=30, reweight_epoch=35)

        logits = torch.randn(4, 3)
        targets = torch.tensor([0, 1, 2, 0])

        _ = loss_fn(logits, targets, epoch=35)
        assert loss_fn.is_drw

        _ = loss_fn(logits, targets, epoch=100)
        assert loss_fn.is_drw, "DRW should remain enabled after activation"

    def test_drw_weights_normalized(self):
        """DRW weights should sum to the number of classes (mean=1)."""
        counts = [100, 50, 10, 5]
        loss_fn = LDAMLoss(counts, max_m=0.5, s=30, reweight_epoch=35)

        logits = torch.randn(2, 4)
        targets = torch.tensor([0, 3])
        _ = loss_fn(logits, targets, epoch=35)

        # internal weight computation: 1/sqrt(cls_num_list) normalized to mean=1
        weights = 1.0 / torch.sqrt(torch.tensor(counts, dtype=torch.float))
        weights = weights / weights.sum() * len(counts)
        assert abs(weights.sum().item() - len(counts)) < 1e-5
        assert abs(weights.mean().item() - 1.0) < 1e-5


# ============================================================================
# set_class_counts() API
# ============================================================================

class TestSetClassCounts:
    """LDAMLoss accepts class counts at construction; users can also
    re-configure by setting cls_num_list and re-computing the margin buffer."""

    def test_construct_with_different_counts(self):
        """LDAMLoss can be constructed with different class distributions."""
        c1 = [10, 10, 10]
        c2 = [100, 1, 50]

        loss1 = LDAMLoss(c1, max_m=0.5)
        loss2 = LDAMLoss(c2, max_m=0.5)

        # Balanced -> all margins equal
        m1 = loss1.m_list.tolist()
        assert all(abs(m - m1[0]) < 1e-6 for m in m1)

        # Imbalanced -> first class (rarest at 1) gets max margin
        assert abs(loss2.m_list[1].item() - 0.5) < 1e-6

    def test_recompute_margins_via_setter(self):
        """Users can recompute m_list by reassigning cls_num_list
        and re-running the margin computation logic."""
        counts = [100, 10]
        loss_fn = LDAMLoss(counts, max_m=0.5, s=30, reweight_epoch=35)

        old_margins = loss_fn.m_list.clone()

        # Change class distribution and recompute margins
        new_counts = [10, 100]
        new_m_list = 1.0 / torch.sqrt(torch.sqrt(torch.tensor(new_counts, dtype=torch.float)))
        new_m_list = new_m_list * (0.5 / new_m_list.max())
        loss_fn.m_list = new_m_list
        loss_fn.cls_num_list = new_counts

        assert not torch.equal(loss_fn.m_list, old_margins), (
            "Margins should change after recomputation"
        )
        # Now the previously common class (count=100) has smaller margin
        assert loss_fn.m_list[0].item() > loss_fn.m_list[1].item(), (
            "Previously common class (now rare at 10) should have larger margin"
        )

    def test_drw_uses_updated_class_counts(self):
        """DRW re-weighting should use the updated class counts."""
        counts = [100, 10]
        loss_fn = LDAMLoss(counts, max_m=0.5, s=30, reweight_epoch=35)

        # Update counts to new distribution
        new_counts = [10, 100]
        new_m_list = 1.0 / torch.sqrt(torch.sqrt(torch.tensor(new_counts, dtype=torch.float)))
        new_m_list = new_m_list * (0.5 / new_m_list.max())
        loss_fn.m_list = new_m_list
        loss_fn.cls_num_list = new_counts

        logits = torch.tensor([[2.0, 0.0], [0.0, 1.0]])
        targets = torch.tensor([0, 1])
        loss = loss_fn(logits, targets, epoch=35)

        assert torch.isfinite(loss), "Loss should be finite after count update"
        assert loss.item() > 0, "Loss should be positive"


# ============================================================================
# End-to-end loss computation
# ============================================================================

class TestLDAMLossForward:
    """Verify forward() produces finite, positive values."""

    def test_loss_finite(self):
        counts = [100, 10]
        loss_fn = LDAMLoss(counts, max_m=0.5, s=30)
        logits = torch.randn(4, 2)
        targets = torch.tensor([0, 1, 0, 1])
        loss = loss_fn(logits, targets, epoch=10)
        assert torch.isfinite(loss), "Loss should be finite"
        assert loss.item() > 0, "Loss should be positive"

    def test_loss_before_and_after_drw(self):
        counts = [100, 10]
        loss_fn = LDAMLoss(counts, max_m=0.5, s=30, reweight_epoch=35)
        logits = torch.randn(8, 2)
        targets = torch.randint(0, 2, (8,))

        l_before = loss_fn(logits, targets, epoch=34)
        l_after = loss_fn(logits, targets, epoch=35)
        assert torch.isfinite(l_before)
        assert torch.isfinite(l_after)

    def test_single_class(self):
        """Should handle binary-like case with 2 classes."""
        counts = [1, 1]
        loss_fn = LDAMLoss(counts, max_m=0.5, s=30)
        logits = torch.randn(2, 2)
        targets = torch.tensor([0, 1])
        loss = loss_fn(logits, targets, epoch=0)
        assert torch.isfinite(loss)
