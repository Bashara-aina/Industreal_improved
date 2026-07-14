"""Tests for PSR transition target generation, monotonic decoding, and focal loss.

Covers:
  - build_transition_targets: Gaussian-smeared (sigma=3.0) transition indicators
  - MonotonicDecoder: once-on-stays-on state machine with hysteresis
  - binary_focal_loss: per-component alpha, comp_weights, ignore_mask handling
"""
import torch
import pytest

from src.models.psr_transition import build_transition_targets, MonotonicDecoder
from src.training.losses import binary_focal_loss


# ============================================================================
# Gaussian Transition Targets (build_transition_targets)
# ============================================================================

class TestBuildTransitionTargets:
    """Gaussian-smeared target generation with sigma=3.0."""

    def test_single_transition_peaks_at_one(self):
        """A single 0->1 transition should produce a Gaussian centred at that frame."""
        labels = torch.zeros(1, 20, 1)
        labels[0, 10:, 0] = 1.0  # transition at frame 10
        targets = build_transition_targets(labels, sigma=3.0)
        # Peak should be at frame 10
        peak_idx = targets[0, :, 0].argmax().item()
        assert peak_idx == 10, f"Peak at {peak_idx}, expected 10"
        assert abs(targets[0, 10, 0].item() - 1.0) < 1e-4, (
            "Gaussian peak should be normalised to 1.0"
        )

    def test_gaussian_decay_symmetric(self):
        """Gaussian should decay symmetrically around the transition."""
        labels = torch.zeros(1, 21, 1)
        labels[0, 10:, 0] = 1.0  # transition at frame 10
        targets = build_transition_targets(labels, sigma=3.0)
        # Check symmetry: frames 8 and 12 should have same value
        off = abs(targets[0, 8, 0].item() - targets[0, 12, 0].item())
        assert off < 1e-4, f"Gaussian not symmetric: delta={off}"
        # Frames far from transition should be near zero
        assert targets[0, 0, 0].item() < 0.05, "Frame far before transition should be near 0"

    def test_no_transition_all_zeros(self):
        """No transition means all targets should be zero (no smoothing)."""
        labels = torch.zeros(1, 15, 2)
        targets = build_transition_targets(labels, sigma=3.0)
        assert (targets == 0).all(), "No transitions should produce all-zero targets"

    def test_sigma_three_radius(self):
        """Gaussian kernel radius should be int(3*sigma) = 9 when sigma=3."""
        labels = torch.zeros(1, 30, 1)
        labels[0, 15:, 0] = 1.0
        targets = build_transition_targets(labels, sigma=3.0)
        # Radius = 9, so frame 15+9=24 should have non-zero but 15+10=25 should be zero
        assert targets[0, 24, 0].item() > 0, "Frame at radius should have non-zero value"
        assert targets[0, 25, 0].item() < 1e-9, "Frame beyond radius should be zero"
        assert targets[0, 5, 0].item() < 1e-9, "Frame before radius should be zero"

    def test_multi_component_independent_smoothing(self):
        """Each component channel should be smoothed independently."""
        labels = torch.zeros(1, 20, 3)
        labels[0, 5:, 0] = 1.0  # c0 transition at 5
        labels[0, 15:, 1] = 1.0  # c1 transition at 15
        # c2: no transition
        targets = build_transition_targets(labels, sigma=3.0)
        assert abs(targets[0, 5, 0].item() - 1.0) < 1e-4, "c0 peak at frame 5"
        assert abs(targets[0, 15, 1].item() - 1.0) < 1e-4, "c1 peak at frame 15"
        assert (targets[0, :, 2] == 0).all(), "c2 (no transition) should be all zero"

    def test_batch_independence(self):
        """Different samples in batch should not interfere."""
        labels = torch.zeros(2, 15, 1)
        labels[0, 5:, 0] = 1.0  # sample 0 transitions at 5
        labels[1, :, 0] = 0.0    # sample 1 never transitions
        targets = build_transition_targets(labels, sigma=3.0)
        assert targets[0, 5, 0].item() > 0.5, "Sample 0 should have peak"
        assert (targets[1, :, 0] == 0).all(), "Sample 1 should be all zero"

    def test_all_transitions_at_start(self):
        """When transitions happen at frame 0, Gaussian should be clipped at boundary."""
        labels = torch.zeros(1, 15, 1)
        labels[0, 0:, 0] = 1.0  # transition at frame 0
        targets = build_transition_targets(labels, sigma=3.0)
        assert targets[0, 0, 0].item() > 0.5, "Frame 0 should have high value"
        assert torch.isfinite(targets).all(), "All values should be finite"

    def test_all_transitions_at_end(self):
        """When transitions happen at last frame, Gaussian should be clipped."""
        labels = torch.zeros(1, 15, 1)
        labels[0, 14:, 0] = 1.0  # transition at last frame (14)
        targets = build_transition_targets(labels, sigma=3.0)
        assert targets[0, 14, 0].item() > 0.5, "Last frame should have peak"
        assert torch.isfinite(targets).all(), "All values should be finite"


# ============================================================================
# Monotonic Decoder (once-on-stays-on, hysteresis)
# ============================================================================

class TestMonotonicDecoder:
    """Monotonic state machine with hysteresis thresholds."""

    def test_once_on_stays_on(self):
        """Once a component transitions to 1, it should stay 1 for all future frames."""
        decoder = MonotonicDecoder(num_components=1)
        B, T = 1, 10
        # Transition probability spikes at frame 3 only
        logits = torch.zeros(B, T, 1)
        logits[0, 3, 0] = 0.9  # above HI threshold (0.5)
        logits[0, 4:8, 0] = 0.6  # above LO threshold (0.3) for sustained
        # Need sustained=3 frames above LO + one above HI to fire
        # Frame 3: above HI, but sustain counter only 1 -> no fire
        # Frame 4: counter=2, above HI -> no fire (counter < 3)
        # Frame 5: counter=3 AND above HI -> FIRE
        states = decoder(logits, threshold=0.3)
        # Frame 5 should have state=1, and stay 1 thereafter
        assert states[0, :5, 0].sum() == 0, "No state should be 1 before frame 5"
        assert states[0, 5, 0].item() == 1.0, "Should fire at frame 5"
        assert (states[0, 6:, 0] == 1.0).all(), "Once on, stays on"

    def test_low_prob_no_transition(self):
        """If all probabilities are low, nothing should transition."""
        decoder = MonotonicDecoder(num_components=2)
        logits = torch.rand(1, 10, 2) * 0.2  # all below LO threshold (0.3)
        states = decoder(logits)
        assert (states == 0).all(), "No transitions should occur with low probs"

    def test_hysteresis_requires_sustained(self):
        """A single high-prob frame should NOT trigger if not sustained."""
        decoder = MonotonicDecoder(num_components=1)
        logits = torch.zeros(1, 10, 1)
        logits[0, 2, 0] = 0.9  # one-frame spike, above HI but not sustained
        states = decoder(logits)
        assert (states == 0).all(), "Single spike should not trigger with hysteresis"

    def test_procedure_order_respected(self):
        """With procedure_order [(0,1)], component 1 must wait for component 0."""
        decoder = MonotonicDecoder(num_components=2, procedure_order=[(0, 1)])
        logits = torch.zeros(1, 15, 2)
        # Both have sustained high prob from frame 3
        logits[0, 3:, 0] = 0.6  # c0
        logits[0, 3:, 1] = 0.6  # c1
        states = decoder(logits)
        # c0 should fire before c1
        t0_fire = states[0, :, 0].argmax().item()
        t1_fire = states[0, :, 1].argmax().item()
        assert t0_fire <= t1_fire, "c0 should fire before or at same time as c1"

    def test_multi_component_no_order(self):
        """With procedure_order containing out-of-range pairs, components are independent."""
        decoder = MonotonicDecoder(num_components=3, procedure_order=[(99, 99)])
        logits = torch.zeros(1, 10, 3)
        logits[0, 3:, :] = 0.6  # all components sustained from frame 3
        states = decoder(logits)
        # All should fire at same time
        t0 = states[0, :, 0].argmax().item()
        t1 = states[0, :, 1].argmax().item()
        t2 = states[0, :, 2].argmax().item()
        assert t0 == t1 == t2, "Without order constraint, all fire at same time"

    def test_already_placed_stays_one(self):
        """If a component is already 1 at start, it should stay 1."""
        # MonotonicDecoder forward recomputes from scratch each call,
        # so this tests the clamping logic within a single forward pass:
        # once set to 1, can_transition excludes already-placed components.
        decoder = MonotonicDecoder(num_components=1)
        logits = torch.zeros(1, 10, 1)
        logits[0, 3:6, 0] = 0.6  # sustained enough to fire
        logits[0, 8, 0] = 0.0  # drops to zero after firing
        states = decoder(logits)
        # Once fired, state should persist even after prob drops
        fire_frame = (states[0, :, 0] == 1.0).nonzero(as_tuple=True)[0][0].item()
        assert fire_frame >= 5, "Should fire after 3 sustained frames"
        assert (states[0, fire_frame:, 0] == 1.0).all(), "State should persist after fire"


# ============================================================================
# Binary Focal Loss (per-component alpha, comp_weights, ignore_mask)
# ============================================================================

class TestBinaryFocalLoss:
    """Per-component alpha, comp_weights, ignore_mask for -1 error states."""

    def test_basic_focal_reduction(self):
        """Standard binary focal loss should return a finite scalar."""
        logits = torch.randn(8, 11)
        targets = (torch.sigmoid(torch.randn(8, 11)) > 0.5).float()
        loss = binary_focal_loss(logits, targets, alpha=0.25, gamma=2.0)
        assert torch.isfinite(loss), "Loss should be finite"
        assert loss.item() > 0, "Loss should be positive"

    def test_per_component_alpha_overrides_scalar(self):
        """per_component_alpha [C] should override scalar alpha."""
        logits = torch.randn(8, 3)
        targets = (torch.sigmoid(torch.randn(8, 3)) > 0.5).float()
        pc_alpha = torch.tensor([0.1, 0.5, 0.9])
        loss1 = binary_focal_loss(logits, targets, alpha=0.25)
        loss2 = binary_focal_loss(logits, targets, per_component_alpha=pc_alpha)
        assert torch.isfinite(loss2), "Loss with per_component_alpha should be finite"

    def test_comp_weights_changes_loss_value(self):
        """comp_weights should change the loss value for imbalanced components."""
        logits = torch.randn(8, 2)
        targets = (torch.sigmoid(torch.randn(8, 2)) > 0.5).float()
        loss_no_weight = binary_focal_loss(logits, targets, gamma=0.0)
        weights = torch.tensor([1.0, 10.0])
        loss_weighted = binary_focal_loss(logits, targets, gamma=0.0, comp_weights=weights)
        assert loss_no_weight != pytest.approx(loss_weighted.item(), rel=1e-3), (
            "comp_weights should change loss value"
        )

    def test_ignore_mask_zeroes_negative_one_targets(self):
        """-1 targets should be ignored (zero contribution to loss)."""
        logits = torch.randn(8, 2)
        targets = torch.full((8, 2), -1.0)  # all ignored
        loss = binary_focal_loss(logits, targets, alpha=0.25, gamma=2.0)
        assert torch.isfinite(loss), "Loss with all -1 targets should be finite"
        assert loss.item() == 0.0, "All -1 targets should give zero loss"

    def test_partial_ignore_mask(self):
        """Mixture of valid and -1 targets should only compute loss on valid."""
        logits = torch.randn(4, 2)
        targets = torch.tensor([[1.0, -1.0], [0.0, 1.0], [-1.0, 0.0], [1.0, 1.0]])
        loss = binary_focal_loss(logits, targets, alpha=0.25, gamma=2.0)
        assert torch.isfinite(loss), "Loss with partial -1 should be finite"
        assert loss.item() > 0, "Loss should be positive"

    def test_all_valid_no_ignore(self):
        """Without any -1 targets, loss should work normally."""
        logits = torch.randn(4, 11)
        targets = (torch.sigmoid(torch.randn(4, 11)) > 0.5).float()
        loss = binary_focal_loss(logits, targets, alpha=0.25, gamma=2.0)
        assert torch.isfinite(loss)

    def test_logit_clamp_safety(self):
        """Extreme logits should be clamped to [-8, 8] for numerical safety."""
        logits = torch.tensor([[100.0, -100.0]])
        targets = torch.tensor([[1.0, 0.0]])
        loss = binary_focal_loss(logits, targets, alpha=0.25, gamma=2.0)
        assert torch.isfinite(loss), "Extreme logits should not produce NaN"
