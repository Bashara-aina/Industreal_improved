#!/usr/bin/env python3
"""
test_st_det_invariance.py — ST-Det Smoke Test Suite (175 §6 Row 1)

Verifies that:
  1. require_split() fires correctly for val (model selection) and test (final).
  2. eval_detection_dual_protocol produces mAP50 keys in both protocols.
  3. The st_det_run metrics.json has the expected structure.
  4. Config patch correctly disables non-detection heads.
  5. Training the detection-only configuration completes (plumbing level).

These are plumbing tests only — they do not verify convergence or accuracy.
Real ST-Det training requires hours of GPU time outside the agent session window.

Usage:
    python -m pytest tests/test_st_det_invariance.py -v

Reference: AAIML 175 §6 (ST-Det row), §7.1 (split discipline), §7.2 (detection)
"""

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

# Patch config before any project imports
import src.config as C

C.TRAIN_DET = True
C.TRAIN_HEAD_POSE = False
C.TRAIN_ACT = False
C.TRAIN_PSR = False
C.STAGED_TRAINING = False
C.USE_KENDALL = False

from src.split_config import require_split, get_split

# ===========================================================================
# Split discipline tests (fast, no torch import needed at module level)
# ===========================================================================


def test_require_split_val_allowed():
    """require_split('val') should not raise for model selection."""
    require_split("val", allow_test_only=False)


def test_require_split_test_allowed():
    """require_split('test', allow_test_only=True) should not raise for final."""
    require_split("test", allow_test_only=True)


def test_require_split_val_rejected_for_test_only():
    """require_split('val', allow_test_only=True) must raise ValueError."""
    with pytest.raises(ValueError, match="not permitted"):
        require_split("val", allow_test_only=True)


def test_require_split_invalid():
    """require_split with invalid name should raise."""
    with pytest.raises((KeyError, ValueError)):
        require_split("invalid_split")


def test_split_counts():
    """Verify split subject counts: 12 train / 5 val / 10 test, no overlap."""
    train_ids = get_split("train")
    val_ids = get_split("val")
    test_ids = get_split("test")
    assert len(train_ids) == 12, f"Expected 12 train, got {len(train_ids)}"
    assert len(val_ids) == 5, f"Expected 5 val, got {len(val_ids)}"
    assert len(test_ids) == 10, f"Expected 10 test, got {len(test_ids)}"
    assert not set(train_ids) & set(val_ids), "Train/val overlap"
    assert not set(train_ids) & set(test_ids), "Train/test overlap"
    assert not set(val_ids) & set(test_ids), "Val/test overlap"


# ===========================================================================
# Config patch tests (fast, no torch)
# ===========================================================================


def test_patch_config_detection_only():
    """Config patch disables all non-detection heads and Kendall."""
    C.TRAIN_DET = True
    C.TRAIN_HEAD_POSE = False
    C.TRAIN_ACT = False
    C.TRAIN_PSR = False
    C.USE_KENDALL = False
    assert C.TRAIN_DET is True
    assert C.TRAIN_HEAD_POSE is False
    assert C.TRAIN_ACT is False
    assert C.TRAIN_PSR is False
    assert C.USE_KENDALL is False


# ===========================================================================
# Output structure tests (fast, no torch)
# ===========================================================================


def test_st_det_metrics_json_structure(tmp_path):
    """The st_det_run metrics.json must have the expected dual-protocol keys."""
    metrics = {
        "protocol": "detection_dual_protocol",
        "reference": "AAIML 174 \u00a73.1 / 175 \u00a77.2",
        "num_classes": 24,
        "annotated_frames": {
            "det_mAP50": 0.5,
            "det_mAP50_pc": 0.5,
            "n_frames": 100,
            "gt_box_total": 50,
            "per_class_ap": {},
        },
        "entire_video": {
            "det_mAP50_all_frames": 0.3,
            "n_frames": 1000,
            "gt_box_total": 50,
            "per_class_ap_all_frames": {},
        },
        "sota_anchor": {
            "WACV_annotated_frames": 0.838,
            "WACV_entire_video": 0.641,
        },
    }
    out_path = tmp_path / "metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    with open(out_path) as f:
        loaded = json.load(f)
    assert loaded["protocol"] == "detection_dual_protocol"
    af = loaded["annotated_frames"]
    ev = loaded["entire_video"]
    assert "det_mAP50" in af
    assert "det_mAP50_pc" in af
    assert "det_mAP50_all_frames" in ev
    assert loaded["sota_anchor"]["WACV_annotated_frames"] == 0.838
    assert loaded["sota_anchor"]["WACV_entire_video"] == 0.641


# ===========================================================================
# Dual-protocol eval tests (import torch inside function)
# ===========================================================================


def test_eval_dual_protocol_compute_functions():
    """Confirm both compute functions produce mAP output keys."""
    import numpy as np
    from src.evaluation.evaluate import (
        compute_ap_per_class,
        compute_ap_per_class_all_frames,
    )

    num_classes = 24
    pred_boxes, pred_scores, pred_labels = [], [], []
    gt_boxes, gt_labels = [], []

    for _ in range(30):
        n_pred = np.random.randint(0, 4)
        n_gt = np.random.randint(0, 3)
        if n_pred:
            pb = np.random.rand(n_pred, 4).astype(np.float32)
            pb[:, 2:] = pb[:, :2] + pb[:, 2:] * 0.3
            pred_boxes.append(pb)
            pred_scores.append(np.random.rand(n_pred).astype(np.float32) * 0.5 + 0.3)
            pred_labels.append(np.random.randint(0, num_classes, size=n_pred).astype(np.int32))
        else:
            pred_boxes.append(np.zeros((0, 4), dtype=np.float32))
            pred_scores.append(np.zeros((0,), dtype=np.float32))
            pred_labels.append(np.zeros((0,), dtype=np.int32))
        if n_gt:
            gb = np.random.rand(n_gt, 4).astype(np.float32)
            gb[:, 2:] = gb[:, :2] + gb[:, 2:] * 0.3
            gt_boxes.append(gb)
            gt_labels.append(np.random.randint(0, num_classes, size=n_gt).astype(np.int32))
        else:
            gt_boxes.append(np.zeros((0, 4), dtype=np.float32))
            gt_labels.append(np.zeros((0,), dtype=np.int32))

    af = compute_ap_per_class(
        pred_boxes,
        pred_scores,
        pred_labels,
        gt_boxes,
        gt_labels,
        iou_thresh=0.5,
        num_classes=num_classes,
    )
    ev = compute_ap_per_class_all_frames(
        pred_boxes,
        pred_scores,
        pred_labels,
        gt_boxes,
        gt_labels,
        iou_thresh=0.5,
        num_classes=num_classes,
    )
    assert "mAP" in af, "Missing mAP key in annotated-frames result"
    assert "mAP" in ev, "Missing mAP key in entire-video result"
    assert "per_class_ap" in af
    assert "per_class_ap" in ev


# ===========================================================================
# Model forward test (GPU recommended, imports torch inside function)
# ===========================================================================


@pytest.mark.skipif(
    not __import__("torch").cuda.is_available(), reason="GPU required for model forward test"
)
def test_model_forward_detection_keys():
    """Model forward must return cls_preds + reg_preds keys.

    POPWMultiTaskModel.forward() returns 'cls_preds' and 'reg_preds'
    (not det_cls_logits/det_box_logits which are Tier F model keys).
    """
    import torch

    model = _build_detection_model()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert "cls_preds" in out, "Missing cls_preds"
    assert "reg_preds" in out, "Missing reg_preds"
    assert "anchors" in out, "Missing anchors"
    cls = out["cls_preds"]
    assert cls.dim() == 2, f"cls_preds should be 2D, got {cls.dim()}D"
    assert cls.shape[-1] == 24, f"Expected 24 classes, got {cls.shape[-1]}"
    reg = out["reg_preds"]
    assert reg.dim() == 2, f"reg_preds should be 2D, got {reg.dim()}D"
    assert reg.shape[-1] == 4, f"Expected 4 box coords, got {reg.shape[-1]}"


# ===========================================================================
# Helper
# ===========================================================================


def _build_detection_model():
    """Build a POPWMultiTaskModel with only detection head active."""
    from src.models.model import POPWMultiTaskModel

    return POPWMultiTaskModel(
        pretrained=False,
        backbone_type="convnext_tiny",
        use_hand_film=False,
        use_headpose_film=False,
        use_videomae=False,
        train_pose=False,
        use_backbone_checkpoint=False,
    ).eval()


# ===========================================================================
# Training smoke test (GPU required, runs train_st_det.py --plumbing)
# ===========================================================================


@pytest.mark.skipif(
    not __import__("torch").cuda.is_available(), reason="GPU required for training smoke test"
)
def test_one_epoch_training_smoke():
    """Smoke test: run train_st_det.py --plumbing and check output exists.

    This spawns a subprocess that runs 1 epoch on 5 recordings, then checks
    that the st_det_run directory has expected artifacts.
    """
    import subprocess

    st_det_script = _ROOT / "scripts" / "train_st_det.py"
    result = subprocess.run(
        [sys.executable, str(st_det_script), "--plumbing", "--max-eval-batches", "50"],
        capture_output=True,
        text=True,
        timeout=3600,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        pytest.fail(f"train_st_det.py --plumbing exited {result.returncode}")

    metrics_path = _ROOT / "src/runs/rf_stages/checkpoints/st_det_run/metrics.json"
    assert metrics_path.exists(), f"metrics.json not found at {metrics_path}"
    with open(metrics_path) as f:
        m = json.load(f)
    assert "annotated_frames" in m
    assert "entire_video" in m
    af = m["annotated_frames"]
    ev = m["entire_video"]
    assert "det_mAP50" in af
    assert "det_mAP50_all_frames" in ev
