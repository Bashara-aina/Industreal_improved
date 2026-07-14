#!/usr/bin/env python3
"""
ST-Pose Invariance Tests — verifies the pose head, renormalization, loss, and
bootstrap CI computation without running a full training loop.

Tests:
  1. Pose head produces 6D output of correct shape
  2. Renormalization step does not produce NaN
  3. Cosine/geodesic loss is finite on sample
  4. Bootstrap CI calculation produces a CI with lower <= mean <= upper

Usage:
    cd /path/to/industreal_improved && python -m pytest tests/test_st_pose_invariance.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

# Path setup
_SCRIPT_DIR = Path(__file__).resolve().parent
_WORK_DIR = _SCRIPT_DIR  # tests/
_PROJECT_ROOT = _WORK_DIR.parent
_CODE_ROOT = _PROJECT_ROOT / "code" / "industreal_improved"
for _p in [
    str(_CODE_ROOT),
    str(_CODE_ROOT / "src"),
    str(_CODE_ROOT / "src" / "models"),
    str(_CODE_ROOT / "src" / "training"),
    str(_CODE_ROOT / "src" / "evaluation"),
    str(_CODE_ROOT / "src" / "data"),
    str(_PROJECT_ROOT),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torch")

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import random


# ===========================================================================
# Import the module under test
# ===========================================================================

# We import directly from the script path so we can test the functions
# without relying on the module structure
_SCRIPT_PATH = _PROJECT_ROOT / "scripts" / "train_st_pose.py"
spec = None
_st_pose = None

# Execute the script to get its symbols
import importlib.util

spec = importlib.util.spec_from_file_location("train_st_pose", str(_SCRIPT_PATH))
_st_pose = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_st_pose)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture(scope="module")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module")
def model(device):
    """Create a minimal ST-Pose model with all parameters initialised."""
    m = _st_pose.MViTv2STPose(freeze_backbone=True)
    m = m.to(device)
    m.eval()
    return m


@pytest.fixture(scope="module")
def sample_batch(device):
    """Create a random batch simulating [B, C, H, W] input."""
    B = 4
    frames = torch.randn(B, 3, 224, 224, device=device)
    # GT pose: unit-norm fwd and up
    fwd_gt = torch.randn(B, 3, device=device)
    fwd_gt = F.normalize(fwd_gt, dim=1)
    up_gt = torch.randn(B, 3, device=device)
    up_gt = F.normalize(up_gt, dim=1)
    targets = torch.cat([fwd_gt, up_gt], dim=1)  # [B, 6]
    return frames, targets


# ===========================================================================
# Test 1: Pose head produces 6D output of correct shape
# ===========================================================================


class TestPoseHeadOutputShape:
    """Verify the pose MLP head produces [B, 6] from [B, 768] features."""

    def test_head_produces_6d(self, model, device):
        """Forward pass through the full model gives [B, 6]."""
        batch = torch.randn(2, 3, 224, 224, device=device)
        with torch.no_grad():
            out = model(batch)
        assert isinstance(out, torch.Tensor), f"Expected Tensor, got {type(out)}"
        assert out.shape == (2, 6), f"Expected (2, 6), got {out.shape}"

    def test_head_mlp_channels(self):
        """MLP layers: 768 -> 256 -> 6."""
        mlp = nn.Sequential(
            nn.Linear(768, 256),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(256, 6),
        )
        x = torch.randn(4, 768)
        out = mlp(x)
        assert out.shape == (4, 6), f"Expected (4, 6), got {out.shape}"

    def test_pose_dim_constant(self):
        """POSE_DIM is 6."""
        assert _st_pose.POSE_DIM == 6, f"Expected 6, got {_st_pose.POSE_DIM}"


# ===========================================================================
# Test 2: Renormalization step does not produce NaN
# ===========================================================================


class TestRenormalization:
    """Verify renormalize_pose handles edge cases without NaN."""

    def test_renormalize_normal_input(self, device):
        """Normal random inputs produce finite fwd/up unit vectors."""
        raw = torch.randn(4, 6, device=device)
        fwd, up = _st_pose.renormalize_pose(raw)
        assert torch.isfinite(fwd).all(), "fwd has NaN/Inf"
        assert torch.isfinite(up).all(), "up has NaN/Inf"
        # Check unit norm
        fwd_norms = fwd.norm(dim=1)
        up_norms = up.norm(dim=1)
        assert torch.allclose(fwd_norms, torch.ones_like(fwd_norms), atol=1e-6)
        assert torch.allclose(up_norms, torch.ones_like(up_norms), atol=1e-6)

    def test_renormalize_zero_input(self, device):
        """All-zero input: should not produce NaN (epsilon guard)."""
        raw = torch.zeros(2, 6, device=device)
        fwd, up = _st_pose.renormalize_pose(raw)
        assert torch.isfinite(fwd).all(), "fwd has NaN/Inf on zero input"
        assert torch.isfinite(up).all(), "up has NaN/Inf on zero input"
        # Norms should be finite (not NaN)
        assert not torch.isnan(fwd.norm(dim=1)).any()

    def test_renormalize_extreme_input(self, device):
        """Extreme values should not produce NaN."""
        raw = torch.full((2, 6), 1e10, device=device, dtype=torch.float32)
        fwd, up = _st_pose.renormalize_pose(raw)
        assert torch.isfinite(fwd).all(), "fwd has NaN/Inf on extreme input"
        assert torch.isfinite(up).all(), "up has NaN/Inf on extreme input"

    def test_renormalize_preserves_separate_channels(self, device):
        """Fwd and up should be independently normalized."""
        raw = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]], device=device)
        fwd, up = _st_pose.renormalize_pose(raw)
        assert torch.allclose(fwd[0], torch.tensor([1.0, 0.0, 0.0], device=device), atol=1e-6)
        assert torch.allclose(up[0], torch.tensor([0.0, 1.0, 0.0], device=device), atol=1e-6)


# ===========================================================================
# Test 3: Cosine/geodesic loss is finite on sample
# ===========================================================================


class TestCosinePoseLoss:
    """Verify cosine_pose_loss produces finite, sensible values."""

    def test_loss_finite(self, device):
        """Random predictions and targets produce finite loss."""
        pred = torch.randn(4, 6, device=device)
        target = torch.randn(4, 6, device=device)
        loss = _st_pose.cosine_pose_loss(pred, target)
        assert torch.isfinite(loss), f"Loss is not finite: {loss}"
        assert loss >= 0, f"Loss should be non-negative, got {loss}"

    def test_perfect_prediction_zero_loss(self, device):
        """Predictions equal to targets (after renormalization) give zero loss."""
        fwd = torch.tensor([[1.0, 0.0, 0.0]], device=device)
        up = torch.tensor([[0.0, 1.0, 0.0]], device=device)
        target = torch.cat([fwd, up], dim=1)
        pred = target.clone()
        loss = _st_pose.cosine_pose_loss(pred, target)
        assert loss < 1e-6, f"Perfect prediction loss should be ~0, got {loss}"

    def test_opposite_prediction_max_loss(self, device):
        """Predicting opposite direction should give max loss (~4.0)."""
        target = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]], device=device)
        pred = torch.tensor([[-1.0, 0.0, 0.0, 0.0, -1.0, 0.0]], device=device)
        loss = _st_pose.cosine_pose_loss(pred, target)
        # Opposite direction: cos = -1, so 1 - (-1) = 2 per channel, total = 4
        assert abs(loss.item() - 4.0) < 1e-4, f"Opposite loss should be ~4.0, got {loss}"

    def test_loss_grad_flows(self, device):
        """Loss is differentiable; gradients flow through the MLP."""
        model = _st_pose.MViTv2STPose(freeze_backbone=True).to(device)
        model.train()
        # Must enable gradients through backbone for the test to pass
        for p in model.backbone.parameters():
            p.requires_grad = True
        batch = torch.randn(2, 3, 224, 224, device=device)
        targets = torch.randn(2, 6, device=device)
        pred = model(batch)
        loss = _st_pose.cosine_pose_loss(pred, targets)
        loss.backward()
        # Check that head parameters received gradients
        head_params = [p for p in model.pose_head.parameters() if p.grad is not None]
        assert len(head_params) > 0, "No gradients flowing to pose head"
        for p in head_params:
            assert torch.isfinite(p.grad).all(), f"Non-finite gradient in {p.shape}"


# ===========================================================================
# Test 3b: Angular MAE computation
# ===========================================================================


class TestAngularMAE:
    """Verify angular_mae_per_frame produces correct values."""

    def test_perfect_prediction_zero_mae(self, device):
        """Perfect fwd/up prediction yields 0 degree MAE."""
        target = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]], device=device)
        pred = target.clone()
        fwd_mae, up_mae = _st_pose.angular_mae_per_frame(pred, target)
        assert fwd_mae.item() < 1e-4, f"Fwd MAE should be ~0, got {fwd_mae}"
        assert up_mae.item() < 1e-4, f"Up MAE should be ~0, got {up_mae}"

    def test_opposite_prediction_180_deg(self, device):
        """Opposite direction yields 180 degree MAE."""
        target = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]], device=device)
        pred = torch.tensor([[-1.0, 0.0, 0.0, 0.0, -1.0, 0.0]], device=device)
        fwd_mae, up_mae = _st_pose.angular_mae_per_frame(pred, target)
        assert abs(fwd_mae.item() - 180.0) < 1e-4, f"Fwd MAE should be ~180, got {fwd_mae}"
        assert abs(up_mae.item() - 180.0) < 1e-4, f"Up MAE should be ~180, got {up_mae}"

    def test_45_degree_mae(self, device):
        """45-degree offset yields ~45 degree MAE."""
        cos45 = 2**0.5 / 2
        target = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]], device=device)
        pred = torch.tensor([[cos45, cos45, 0.0, cos45, cos45, 0.0]], device=device)
        fwd_mae, up_mae = _st_pose.angular_mae_per_frame(pred, target)
        assert abs(fwd_mae.item() - 45.0) < 1e-4, f"Fwd MAE should be ~45, got {fwd_mae}"
        assert abs(up_mae.item() - 45.0) < 1e-4, f"Up MAE should be ~45, got {up_mae}"


# ===========================================================================
# Test 4: Bootstrap CI produces lower <= mean <= upper
# ===========================================================================


class TestBootstrapCI:
    """Verify bootstrap_ci invariants."""

    def test_ci_order_invariant(self):
        """CI lower <= mean <= upper for uniform random data."""
        rng = random.Random(42)
        values = [rng.gauss(0, 1) for _ in range(100)]
        mean, lo, hi = _st_pose.bootstrap_ci(values, n_resamples=1000, seed=42)
        assert lo <= mean <= hi, f"CI invariant violated: {lo} <= {mean} <= {hi}"

    def test_ci_narrow_with_low_variance(self):
        """Low-variance data produces narrow CI."""
        values = [0.5 + rng.uniform(-0.01, 0.01) for rng in [random.Random(42)] for _ in range(200)]
        values = [0.5 + random.Random(42).uniform(-0.01, 0.01) for _ in range(200)]
        mean, lo, hi = _st_pose.bootstrap_ci(values, n_resamples=1000, seed=42)
        width = hi - lo
        assert width < 0.1, f"Low-variance CI too wide: {width}"

    def test_ci_weighted(self):
        """Weighted bootstrap CI with frame weights is consistent."""
        values = [1.0, 2.0, 3.0]
        weights = [2.0, 1.0, 1.0]
        mean, lo, hi = _st_pose.bootstrap_ci(values, weights, n_resamples=1000, seed=42)
        # Weighted mean = (2*1 + 1*2 + 1*3) / (2+1+1) = 7/4 = 1.75
        expected_mean = (2 * 1.0 + 1 * 2.0 + 1 * 3.0) / 4.0
        assert abs(mean - expected_mean) < 1e-6, (
            f"Weighted mean mismatch: {mean} != {expected_mean}"
        )
        assert lo <= mean <= hi, f"CI invariant violated: {lo} <= {mean} <= {hi}"

    def test_ci_empty_returns_nan(self):
        """Empty input returns NaN for all three values."""
        mean, lo, hi = _st_pose.bootstrap_ci([], n_resamples=1000, seed=42)
        assert math.isnan(mean), "Mean should be NaN for empty input"
        assert math.isnan(lo), "Lo should be NaN for empty input"
        assert math.isnan(hi), "Hi should be NaN for empty input"

    def test_ci_single_value(self):
        """Single value returns that value for mean, spread depends on resampling."""
        mean, lo, hi = _st_pose.bootstrap_ci([3.14], n_resamples=1000, seed=42)
        assert abs(mean - 3.14) < 1e-6, f"Mean mismatch: {mean}"
        # lo and hi can differ due to resampling of the single element
        assert lo <= mean <= hi, f"CI invariant violated: {lo} <= {mean} <= {hi}"


# ===========================================================================
# Test 4b: End-to-end evaluate_pose returns correct structure
# ===========================================================================


class TestEvaluatePoseStructure:
    """Verify evaluate_pose returns dict matching bootstrap_ci.json structure."""

    def test_evaluate_pose_returns_expected_keys(self, model, sample_batch, device):
        """Evaluate on a single batch returns bootstrap_ci-like structure."""
        frames, targets = sample_batch
        dataset = torch.utils.data.TensorDataset(frames, targets)
        loader = torch.utils.data.DataLoader(dataset, batch_size=4)

        results = _st_pose.evaluate_pose(model, loader, device=device, max_batches=1)

        assert "head_pose_forward" in results, "Missing head_pose_forward"
        assert "head_pose_up" in results, "Missing head_pose_up"
        assert "metadata" in results, "Missing metadata"

        fwd = results["head_pose_forward"]
        up = results["head_pose_up"]

        assert "headline_weighted_mean_deg" in fwd
        assert "bootstrap_95_ci_deg" in fwd
        assert len(fwd["bootstrap_95_ci_deg"]) == 2
        assert (
            fwd["bootstrap_95_ci_deg"][0]
            <= fwd["headline_weighted_mean_deg"]
            <= fwd["bootstrap_95_ci_deg"][1]
        )

        assert "headline_weighted_mean_deg" in up
        assert "bootstrap_95_ci_deg" in up
        assert len(up["bootstrap_95_ci_deg"]) == 2
        assert (
            up["bootstrap_95_ci_deg"][0]
            <= up["headline_weighted_mean_deg"]
            <= up["bootstrap_95_ci_deg"][1]
        )

    def test_evaluate_pose_mirrors_bootstrap_ref_structure(self, model, sample_batch, device):
        """Output key structure mirrors bootstrap_ci.json."""
        frames, targets = sample_batch
        dataset = torch.utils.data.TensorDataset(frames, targets)
        loader = torch.utils.data.DataLoader(dataset, batch_size=4)
        results = _st_pose.evaluate_pose(model, loader, device=device, max_batches=1)

        # Reference structure from bootstrap_ci.json
        expected_fwd_keys = {
            "headline_weighted_mean_deg",
            "bootstrap_95_ci_deg",
            "bootstrap_method",
            "n_frames",
        }
        expected_up_keys = {
            "headline_weighted_mean_deg",
            "bootstrap_95_ci_deg",
            "bootstrap_method",
            "n_frames",
        }

        assert expected_fwd_keys.issubset(results["head_pose_forward"].keys()), (
            f"Missing keys in fwd: {expected_fwd_keys - results['head_pose_forward'].keys()}"
        )
        assert expected_up_keys.issubset(results["head_pose_up"].keys()), (
            f"Missing keys in up: {expected_up_keys - results['head_pose_up'].keys()}"
        )


# Need math for NaN test
import math


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
