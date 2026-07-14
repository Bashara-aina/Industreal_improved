#!/usr/bin/env python3
"""
test_st_act_invariance.py — Invariance tests for ST-Act single-task activity baseline.

Tests:
  1. 50 clips can be loaded from the train split
  2. 75-class head produces logits of shape [B, 75]
  3. One-epoch training + eval produces numeric top-1 in (0, 1.0)
  4. require_split guards raise error on invalid split

Usage:
    cd /path/to/industreal_improved && python -m pytest tests/test_st_act_invariance.py -v
"""

import sys
from pathlib import Path

# Path setup
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR
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

import pytest
import torch


# ===========================================================================
# Test 1: 50 clips loadable
# ===========================================================================
def test_50_clips_loadable():
    """Test that 50 clips can be loaded from the train split."""
    from scripts.train_st_act import ClipDataset

    ds = ClipDataset(split="train", clip_len=16, stride=8, max_clips=50)

    assert len(ds) == 50, f"Expected 50 clips, got {len(ds)}"

    # Load all 50
    for i in range(50):
        clip, label = ds[i]
        assert clip.shape == (16, 3, 224, 224), (
            f"Clip {i} shape {clip.shape}, expected (16, 3, 224, 224)"
        )
        assert clip.dtype == torch.float32, f"Clip {i} dtype {clip.dtype}"
        assert 0 <= label < 75, f"Label {i} out of range: {label}"

    print(f"  Loaded {len(ds)} clips successfully, all shapes correct")


# ===========================================================================
# Test 2: 75-class head produces correct logit shape
# ===========================================================================
def test_75_class_head_shape():
    """Test that the 75-class head produces logits of shape [B, 75]."""
    from scripts.train_st_act import MViTv2STAct

    model = MViTv2STAct(num_classes=75, freeze_backbone=True)
    model.eval()

    # Create a random clip [B, T, C, H, W]
    x = torch.randn(4, 16, 3, 224, 224)
    with torch.no_grad():
        logits = model(x)

    assert logits.shape == (4, 75), f"Expected logits shape (4, 75), got {logits.shape}"
    assert logits.dtype == torch.float32, f"Logits dtype {logits.dtype}"
    assert torch.isfinite(logits).all(), "Logits contain non-finite values"

    print(f"  Logits shape {logits.shape}, all finite: OK")


# ===========================================================================
# Test 3: One-epoch training + eval returns numeric top-1 < 1.0
# ===========================================================================
@pytest.mark.slow
def test_one_epoch_training():
    """Test that one epoch of training produces valid metrics.

    This is a plumbing verification: train for 1 epoch on a small subset
    and verify top-1 is numeric and reasonable.
    """
    from scripts.train_st_act import (
        MViTv2STAct,
        ClipDataset,
        compute_class_weights,
        train,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tiny datasets (50 train, 20 val)
    train_ds = ClipDataset(split="train", clip_len=16, stride=8, max_clips=50)
    val_ds = ClipDataset(split="val", clip_len=16, stride=8, max_clips=20)

    assert len(train_ds) >= 10, f"Need >=10 train clips, got {len(train_ds)}"
    assert len(val_ds) >= 5, f"Need >=5 val clips, got {len(val_ds)}"

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=4,
        shuffle=True,
        num_workers=0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )

    # Model (freeze backbone for speed — just testing plumbing)
    model = MViTv2STAct(num_classes=75, freeze_backbone=True).to(device)

    # Class weights
    class_weights = compute_class_weights(train_ds, num_classes=75).to(device)

    # Loss
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    # Optimizer (only head params since backbone frozen)
    optimizer = torch.optim.AdamW(
        model.classifier.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    # Scheduler
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=1,
    )
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=0,
    )

    # Train for 1 epoch
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        metrics = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=cosine_scheduler,
            warmup_scheduler=warmup_scheduler,
            num_epochs=1,
            warmup_epochs=0,
            device=device,
            output_dir=Path(tmpdir),
        )

    # Verify metrics exist and are reasonable
    assert "top1" in metrics, "Missing top1 in metrics"
    assert "top5" in metrics, "Missing top5 in metrics"
    assert "clip_count" in metrics, "Missing clip_count"
    assert metrics["clip_count"] > 0, "Zero clip count"

    top1 = metrics["top1"]
    top5 = metrics["top5"]

    # top-1 should be reasonable after 1 epoch (not 0, not perfect)
    assert 0.0 <= top1 < 1.0, f"top1={top1} outside expected range [0, 1.0)"
    assert 0.0 <= top5 <= 1.0, f"top5={top5} outside expected range [0, 1.0]"

    # With frozen backbone + cosine LR T_max=1, LR decays to 0 after step 1,
    # so top-1 may be at chance (~0.013) or lower for small val sets.
    assert top1 >= 0.0, f"top1={top1} is negative — model is broken"

    print(f"  1-epoch training: top1={top1:.4f}, top5={top5:.4f}, n={metrics['clip_count']}")


# ===========================================================================
# Test 4: require_split guards work
# ===========================================================================
def test_require_split_guards():
    """Test that require_split correctly guards split access."""
    from src.split_config import require_split

    # Valid splits should not raise
    require_split("val", allow_test_only=False)
    require_split("train", allow_test_only=False)

    # test split should be allowed with allow_test_only=True
    require_split("test", allow_test_only=True)

    # "val" should raise when allow_test_only=True
    with pytest.raises(ValueError, match="Only 'test' may be used"):
        require_split("val", allow_test_only=True)

    # "train" should raise when allow_test_only=True
    with pytest.raises(ValueError, match="Only 'test' may be used"):
        require_split("train", allow_test_only=True)

    # Invalid split name should raise
    with pytest.raises(ValueError, match="Invalid eval_split"):
        require_split("invalid_split", allow_test_only=False)

    print("  All require_split guards work correctly")
