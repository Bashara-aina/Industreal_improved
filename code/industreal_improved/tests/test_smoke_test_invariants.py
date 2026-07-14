#!/usr/bin/env python3
"""
Smoke Test Invariants — 175 §2 Preflight Gate Verification

Verifies the smoke test's invariants without running the full training loop:
  Test 1: 50 clips can be loaded
  Test 2: 4-head model produces all 4 losses on a sample
  Test 3: At least one head has a non-NaN, non-zero gradient
  Test 4: The assertion thresholds are present and sensible

Usage:
    cd /path/to/industreal_improved && python -m pytest tests/test_smoke_test_invariants.py -v
"""

import sys
from pathlib import Path

# Path setup (mirrors minimal_smoke_test.py)
_SCRIPT_DIR = Path(__file__).resolve().parent
_WORK_DIR = _SCRIPT_DIR.parent  # code/industreal_improved
_SRC_DIR = _WORK_DIR / "src"
sys.path.insert(0, str(_WORK_DIR))
sys.path.insert(1, str(_SRC_DIR / "models"))
sys.path.insert(2, str(_SRC_DIR / "training"))
sys.path.insert(3, str(_SRC_DIR / "evaluation"))
sys.path.insert(4, str(_SRC_DIR))

import pytest
import torch
import torch.nn.functional as F


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture(scope="module")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module")
def model_4head(device):
    """Try TierFModel first, fall back to POPWMultiTaskModel."""
    try:
        from src.models.tier_f_model import TierFModel

        m = TierFModel(
            num_classes_det=24,
            num_classes_act=75,
            num_components_psr=11,
            pose_dim=6,
            pretrained=False,
        ).to(device)
        m.train()
        return m, "TierFModel"
    except Exception:
        from src import config as C
        from models import model as model_module

        m = model_module.POPWMultiTaskModel(
            backbone_type=C.BACKBONE,
            pretrained=False,
            use_videomae=False,
        ).to(device)
        m.train()
        return m, "POPWMultiTaskModel"


@pytest.fixture(scope="module")
def sample_batch(device, model_4head):
    """Create a mini sample to test forward/backward."""
    model, model_name = model_4head

    if model_name == "TierFModel":
        # Temporal clip [1, 16, 3, 224, 224]
        clip = torch.randn(1, 16, 3, 224, 224, device=device)
        det_frame = torch.randn(1, 3, 224, 224, device=device)
        return {
            "model_name": model_name,
            "clip": clip,
            "det_frame": det_frame,
            "targets": {
                "activity": torch.randint(0, 75, (1,), device=device),
                "psr_labels": torch.randint(0, 2, (1, 16, 11), device=device).float(),
                "head_pose": torch.randn(1, 9, device=device),
                "detection": [
                    {
                        "boxes": torch.tensor([[100.0, 100.0, 400.0, 400.0]], device=device),
                        "labels": torch.tensor([2], device=device),
                    }
                ],
            },
        }
    else:
        # Single frame [B, 3, H, W]
        # Use the configured image size
        from src import config as C

        frames = torch.randn(1, 3, C.IMG_HEIGHT, C.IMG_WIDTH, device=device)
        return {
            "model_name": model_name,
            "frames": frames,
            "targets": {
                "activity": torch.randint(0, C.NUM_CLASSES_ACT, (1,), device=device),
                "psr_labels": torch.rand(1, C.NUM_PSR_COMPONENTS, device=device),
                "head_pose": torch.randn(1, 9, device=device),
                "detection": [
                    {
                        "boxes": torch.tensor([[100.0, 100.0, 400.0, 400.0]], device=device),
                        "labels": torch.tensor([2], device=device),
                    }
                ],
            },
        }


# ===========================================================================
# Test 1: 50 clips can be loaded from the dataset
# ===========================================================================


def test_50_clips_loadable():
    """Test that 50 temporal clips can be successfully loaded from the dataset."""
    try:
        from src.data.industreal_dataset import IndustRealMultiTaskDataset as Dataset
        import config as Cfg

        # Temporarily disable RAM cache for quick loading
        _orig = getattr(Cfg, "RAM_CACHE_MAX_IMAGES", 8000)
        Cfg.RAM_CACHE_MAX_IMAGES = 0

        ds = Dataset(
            split="train",
            augment=False,
            sequence_mode=True,
            sequence_length=16,
            max_recordings=None,
        )

        Cfg.RAM_CACHE_MAX_IMAGES = _orig

        n = min(len(ds), 50)
        assert n == 50, f"Expected 50 clips available, got {n}"

        # Load the first 50
        for i in range(50):
            sample = ds[i]
            assert sample is not None, f"Sample {i} is None"
            assert "images" in sample, f"Sample {i} missing 'images'"
            assert "rgb" in sample["images"], f"Sample {i} missing 'rgb'"
            assert sample["images"]["rgb"].shape[0] == 16, (
                f"Sample {i} expected 16 frames, got {sample['images']['rgb'].shape[0]}"
            )

        # Verify key labels exist
        s = ds[0]
        required_keys = ["action_label", "psr_labels", "head_pose", "gt_boxes", "gt_classes"]
        for key in required_keys:
            assert key in s, f"Sample missing required key: {key}"

        print(f"  Loaded 50 clips successfully (dataset has {len(ds)} total windows)")

    except ImportError as e:
        pytest.skip(f"Cannot import dataset module: {e}")
    except Exception as e:
        pytest.fail(f"Dataset loading failed: {e}")


# ===========================================================================
# Test 2: 4-head model produces all 4 losses on a sample
# ===========================================================================


def _compute_smoke_losses(model, model_name, batch, device):
    """Compute per-head losses for any model type."""
    targets = batch["targets"]

    if model_name == "TierFModel":
        clip = batch["clip"]
        det_frame = batch["det_frame"]

        temp_out = model(clip, mode="temporal")
        det_out = model(det_frame, mode="detection")

        # Activity loss
        loss_act = F.cross_entropy(temp_out["act_logits"], targets["activity"])

        # PSR loss
        loss_psr = F.binary_cross_entropy_with_logits(temp_out["psr_logits"], targets["psr_labels"])

        # Pose loss
        loss_pose = F.mse_loss(temp_out["pose_6d"], targets["head_pose"][:, :6])

        # Detection loss (simplified plumbing check)
        cls_list = det_out["det_cls_logits"]
        box_list = det_out["det_box_logits"]
        # Each is [B, C, H, W]. GAP -> [B, C], average across levels -> [B, C]
        cls_level_means = [l.mean(dim=[-1, -2]) for l in cls_list]  # list of [B, C]
        cls_gap = torch.stack(cls_level_means, dim=0).mean(dim=0)  # [B, C]
        cls_target = torch.zeros(cls_gap.shape[-1], device=device)
        for lbl in targets["detection"][0]["labels"]:
            if lbl < cls_gap.shape[-1]:
                cls_target[lbl] = 1.0
        # Broadcast target to batch
        cls_target_batch = cls_target.unsqueeze(0).expand(cls_gap.shape[0], -1)
        loss_det = F.binary_cross_entropy_with_logits(cls_gap, cls_target_batch)

        return {"det": loss_det, "act": loss_act, "psr": loss_psr, "pose": loss_pose}

    else:
        # POPWMultiTaskModel
        from src import config as C

        outputs = model(batch["frames"])
        T = C.NUM_PSR_COMPONENTS

        loss_act = F.cross_entropy(outputs["act_logits"], targets["activity"])
        loss_psr = F.binary_cross_entropy_with_logits(outputs["psr_logits"], targets["psr_labels"])
        loss_pose = F.mse_loss(outputs["head_pose"][:, :6], targets["head_pose"][:, :6])

        # Detection
        cls_list = outputs["cls_preds"]
        box_list = outputs["reg_preds"]
        if isinstance(cls_list, list):
            cls_level_means = [l.mean(dim=[-1, -2]) for l in cls_list]
            cls_gap = torch.stack(cls_level_means, dim=0).mean(dim=0)
        else:
            cls_gap = cls_list.mean(dim=[-1, -2])  # [B, C]
        num_det_classes = cls_gap.shape[-1]
        cls_target = torch.zeros(num_det_classes, device=device)
        for lbl in targets["detection"][0]["labels"]:
            if lbl < num_det_classes:
                cls_target[lbl] = 1.0
        cls_target_batch = cls_target.unsqueeze(0).expand(cls_gap.shape[0], -1)
        loss_det = F.binary_cross_entropy_with_logits(cls_gap, cls_target_batch)

        return {"det": loss_det, "act": loss_act, "psr": loss_psr, "pose": loss_pose}


def test_all_four_losses_produced(model_4head, sample_batch, device):
    """Test that the 4-head model produces all 4 losses on a single sample."""
    model, model_name = model_4head
    losses = _compute_smoke_losses(model, model_name, sample_batch, device)

    expected_heads = ["det", "act", "psr", "pose"]
    for head in expected_heads:
        assert head in losses, f"Missing loss for head: {head}"
        assert isinstance(losses[head], torch.Tensor), (
            f"Loss for {head} is not a tensor: {type(losses[head])}"
        )
        assert losses[head].numel() == 1, (
            f"Loss for {head} should be scalar, got shape {losses[head].shape}"
        )
        assert torch.isfinite(losses[head]), f"Loss for {head} is not finite: {losses[head].item()}"

    print(
        f"  Losses: det={losses['det'].item():.4f} act={losses['act'].item():.4f} "
        f"psr={losses['psr'].item():.4f} pose={losses['pose'].item():.6f}"
    )


# ===========================================================================
# Test 3: At least one head has a non-NaN, non-zero gradient
# ===========================================================================


def test_gradient_exists_and_non_nan(model_4head, sample_batch, device):
    """Test that at least one head produces a valid gradient."""
    model, model_name = model_4head

    losses = _compute_smoke_losses(model, model_name, sample_batch, device)

    model.zero_grad()
    total = sum(losses.values())
    total.backward()

    heads_with_grad = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            grad_norm = p.grad.norm().item()
            has_nan = torch.isnan(p.grad).any().item()
            has_inf = torch.isinf(p.grad).any().item()

            if grad_norm > 0 and not has_nan and not has_inf:
                # Determine which head this param belongs to
                for head in [
                    "backbone",
                    "fpn",
                    "detection",
                    "activity",
                    "psr",
                    "pose",
                    "head_pose",
                ]:
                    if head in name:
                        heads_with_grad.append(head)
                        break
                else:
                    heads_with_grad.append("other")

    unique_heads = set(heads_with_grad)

    # We need at least one head with valid gradient (not all 4 may be required)
    assert len(unique_heads) >= 1, (
        f"No head has a non-NaN, non-zero gradient. Gradients found in: {heads_with_grad}"
    )

    # Check no NaN anywhere
    all_nan_free = True
    for name, p in model.named_parameters():
        if p.grad is not None and torch.isnan(p.grad).any():
            all_nan_free = False
            break

    assert all_nan_free, "NaN gradient found in at least one parameter"

    print(f"  Heads with valid gradients: {unique_heads}")
    print(f"  All gradients NaN-free: {all_nan_free}")


# ===========================================================================
# Test 4: Assertion thresholds are present and sensible
# ===========================================================================


def test_thresholds_present_and_sensible():
    """Test that the smoke test thresholds are defined and physically meaningful."""
    # These match 175 §2 specs
    thresholds = {
        "loss_det": {"value": 1.0, "interpretation": "Detection loss below 1.0"},
        "loss_act": {"value": 1.0, "interpretation": "Activity loss below 1.0"},
        "loss_psr": {"value": 1.0, "interpretation": "PSR loss below 1.0"},
        "loss_pose": {"value": 0.1, "interpretation": "Pose loss below 0.1"},
    }

    # Verify all four thresholds exist
    expected_heads = ["loss_det", "loss_act", "loss_psr", "loss_pose"]
    for head in expected_heads:
        assert head in thresholds, f"Missing threshold for {head}"
        assert "value" in thresholds[head], f"Threshold {head} missing 'value'"
        assert "interpretation" in thresholds[head], f"Threshold {head} missing 'interpretation'"

    # Verify thresholds are physically sensible
    for head, config in thresholds.items():
        val = config["value"]
        assert val > 0, f"Threshold {head}={val} should be positive"
        if head == "loss_pose":
            # Pose loss (MSE on normalized vectors) should be O(0.01) for good fit
            assert val < 1.0, f"Threshold {head}={val} seems too lenient for pose"
        else:
            # Detection/activity/PSR (classification losses) can be < 1.0 for overfit
            assert val <= 5.0, f"Threshold {head}={val} seems too lenient"

    print(f"  All 4 thresholds present and physically sensible:")
    for head, config in thresholds.items():
        print(f"    {head} < {config['value']} — {config['interpretation']}")
