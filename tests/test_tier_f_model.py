#!/usr/bin/env python3
"""Tests for the Tier F multi-task model (175 §3.1-3.2).

Verifies that:
  1. Model instantiates correctly (backbone + 4 heads).
  2. Temporal forward mode produces correct output shapes.
  3. Detection forward mode produces correct output shapes.
  4. All heads produce tensors with valid shapes and dtypes.
  5. Parameter counts are roughly as specified (~60M total).

Marked ``xfail`` if Hiera backbone cannot be loaded (architecture absent
from installed timm).  Pretrained-weight download failures are **not**
a skip condition -- the architecture must run without pretrained weights.
"""

import sys
from pathlib import Path

import pytest
import torch

# Ensure the model module is importable
_ROOT = Path(__file__).resolve().parent.parent / "code" / "industreal_improved"
sys.path.insert(0, str(_ROOT))

# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture(scope="module")
def model():
    """Instantiate TierFModel with random weights (no pretrained download)."""
    from src.models.tier_f_model import TierFModel

    m = TierFModel(
        num_classes_det=24,
        num_classes_act=75,
        num_components_psr=11,
        pose_dim=6,
        pretrained=False,
    )
    m.eval()
    return m


@pytest.fixture(scope="module")
def temporal_input():
    """Random clip for temporal mode: ``[B=2, T=16, 3, 224, 224]``."""
    return torch.randn(2, 16, 3, 224, 224)


@pytest.fixture(scope="module")
def detection_input():
    """Random frame for detection mode: ``[B=2, 3, 224, 224]``.

    Note: timm's Hiera-B has fixed position-embedding windows sized for
    224x224.  Higher resolution (e.g. 448-640 as specified in 175 §3.1)
    requires adapting the backbone's internal pos_embed, unroll, and
    reroll state -- left as production work.
    """
    return torch.randn(2, 3, 224, 224)


# ===========================================================================
# Tests
# ===========================================================================


def test_model_instantiation(model):
    """Model should instantiate with expected components."""
    from src.models.tier_f_model import (
        FPN,
        ActivityHead,
        DetectionHead,
        PSRHead,
        PoseHead,
        TierFModel,
    )

    assert isinstance(model, TierFModel)
    assert isinstance(model.fpn, FPN)
    assert isinstance(model.detection_head, DetectionHead)
    assert isinstance(model.activity_head, ActivityHead)
    assert isinstance(model.psr_head, PSRHead)
    assert isinstance(model.pose_head, PoseHead)


def test_temporal_mode_shapes(model, temporal_input):
    """Temporal forward produces correct output shapes."""
    with torch.no_grad():
        out = model(temporal_input, mode="temporal")

    assert "act_logits" in out, "Missing act_logits"
    assert "psr_logits" in out, "Missing psr_logits"
    assert "pose_6d" in out, "Missing pose_6d"

    B = temporal_input.shape[0]
    T = temporal_input.shape[1]

    # Activity: (B, 75)
    assert out["act_logits"].shape == (B, 75), (
        f"Expected (B, 75), got {out['act_logits'].shape}"
    )

    # PSR: (B, T, 11)
    assert out["psr_logits"].shape == (B, T, 11), (
        f"Expected (B, {T}, 11), got {out['psr_logits'].shape}"
    )

    # Pose: (B, 6)
    assert out["pose_6d"].shape == (B, 6), (
        f"Expected (B, 6), got {out['pose_6d'].shape}"
    )

    # All should be float tensors
    for key, tensor in out.items():
        assert tensor.is_floating_point(), f"{key} is not floating point"


def test_detection_mode_shapes(model, detection_input):
    """Detection forward produces correct output shapes."""
    with torch.no_grad():
        out = model(detection_input, mode="detection")

    assert "det_cls_logits" in out, "Missing det_cls_logits"
    assert "det_box_logits" in out, "Missing det_box_logits"

    cls_list = out["det_cls_logits"]
    box_list = out["det_box_logits"]

    # Should have 3 FPN levels
    assert len(cls_list) == 3, f"Expected 3 FPN levels, got {len(cls_list)}"
    assert len(box_list) == 3, f"Expected 3 FPN levels, got {len(box_list)}"

    B = detection_input.shape[0]
    H, W = detection_input.shape[2:]

    # Check stride relationships: FPN levels correspond to strides
    # [8, 16, 32] relative to input (approximately)
    # Input 448 -> strides 8=56, 16=28, 32=14
    for level_idx, (cls_t, box_t) in enumerate(zip(cls_list, box_list)):
        assert cls_t.shape[0] == B, f"Level {level_idx} cls batch mismatch"
        assert box_t.shape[0] == B, f"Level {level_idx} box batch mismatch"
        # Cls: (B, 24, H_i, W_i)
        assert cls_t.shape[1] == 24, (
            f"Level {level_idx} cls expected 24 ch, got {cls_t.shape[1]}"
        )
        # Box: (B, 4*16=64, H_i, W_i)
        assert box_t.shape[1] == 64, (
            f"Level {level_idx} box expected 64 ch, got {box_t.shape[1]}"
        )
        # Spatial dims should be decreasing across levels
        assert cls_t.shape[-1] == box_t.shape[-1], (
            f"Level {level_idx} cls/box width mismatch"
        )
        assert cls_t.shape[-2] == box_t.shape[-2], (
            f"Level {level_idx} cls/box height mismatch"
        )

    # Strides: 224/(8, 16, 32) = (28, 14, 7)
    assert cls_list[0].shape[-2] == 28, (
        f"P3 expected height 28, got {cls_list[0].shape[-2]}"
    )
    assert cls_list[1].shape[-2] == 14, (
        f"P4 expected height 14, got {cls_list[1].shape[-2]}"
    )
    assert cls_list[2].shape[-2] == 7, (
        f"P5 expected height 7, got {cls_list[2].shape[-2]}"
    )

    # All should be float tensors
    for key, tensor_list in out.items():
        for level_idx, tensor in enumerate(tensor_list):
            assert tensor.is_floating_point(), (
                f"{key}[{level_idx}] is not floating point"
            )


def test_unknown_mode(model, temporal_input):
    """Unknown mode raises ValueError."""
    with pytest.raises(ValueError, match="Unknown mode"):
        model(temporal_input, mode="invalid")


def test_param_count(model):
    """Total parameter count should be approximately 60M."""
    counts = model.get_param_counts()
    total = counts["total"]

    # Spec says ~60M. Allow ±15% for stub head variations.
    expected = 60_000_000
    tolerance = 0.15
    assert abs(total - expected) / expected < tolerance, (
        f"Total params {total:,} outside ±{tolerance*100:.0f}% of {expected:,}. "
        f"Component counts: {counts}"
    )

    # Backbone should dominate
    assert counts.get("backbone", 0) > 40_000_000, (
        f"Backbone too small: {counts.get('backbone', 0):,}"
    )


def test_forward_batch_consistency(model, temporal_input):
    """Temporal forward should be batch-independent (two identical frames
    produce same per-frame output)."""
    # Create batch where one sample is duplicated
    dup = temporal_input[:1].repeat(2, 1, 1, 1, 1)
    with torch.no_grad():
        out = model(dup, mode="temporal")

    # Both samples should have identical outputs
    assert torch.allclose(out["act_logits"][0], out["act_logits"][1], atol=1e-6)
    assert torch.allclose(out["pose_6d"][0], out["pose_6d"][1], atol=1e-6)
    assert torch.allclose(out["psr_logits"][0], out["psr_logits"][1], atol=1e-6)
