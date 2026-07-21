# IMP-11: Task-Aligned Learning (TAL) for One-Stage Detection

## Summary

Implements **Task-aligned one-stage object detection (TOOD, AAAI 2021)** which
jointly optimizes classification and localization with a task-aligned loss.
TOOD replaces standard focal loss with an IoU-gated alignment signal that
down-weights anchors whose predicted box does not align well with the GT box.

**Expected improvement**: +3.2 AP over ATSS (per TOOD paper Table 1).

## What changed

1. **`--use-tal` CLI flag** (default `False`, backward compatible).
   When enabled, the classification loss for each positive anchor is weighted by
   `IoU(pred_box, GT_box) ** tal_alpha`, where `tal_alpha=2.0` matches the
   TOOD default.

2. **TAL-only classification weighting** — regression (CIoU) is left unchanged,
   matching the TOOD design where alignment only gates the classification branch.

3. **`--tal-alpha`** controls the alignment exponent (default 2.0).

4. **`--tal-lr-mult`** boosts the detection head LR by 2x (default) when TAL is
   active, as recommended by the TOOD paper for TAL-specific heads.

## Design

TAL computes per-location classification weights as follows:

```
alignment_metric = IoU(pred_box, GT_box)      # per positive anchor
loc_weight       = max(alignment over anchors at location) ** tal_alpha
cls_loss         = loc_weight * alpha * focal_weight * BCE
```

Background locations keep `weight=1.0` (no down-weighting), following standard
focal loss.

The TAL path uses the same focal-loss alpha/focal_weight formulation as standard
sigmoid focal loss, but each positive location's contribution is scaled by its
alignment metric. This ensures that only well-localized positive predictions
contribute strongly to the classification gradient, while poorly-localized ones
are suppressed.

## Files modified

- **`train_mtl_v3.py`**:
  - `detection_loss()`: Added `use_tal` / `tal_alpha` parameters.
  - TAL branch: when `use_tal=True` and `pos_mask.any()`, classification loss
    uses the alignment-weighted focal formulation.
  - `multi_task_loss_v3()`: Plumbed `use_tal` / `tal_alpha` through to
    `detection_loss()`.
  - CLI: Added `--use-tal`, `--tal-alpha`, `--tal-lr-mult`.
  - LR: `det_lr_mult_eff` computed as `args.det_lr_mult * tal_lr_mult` when TAL
    is active, applied to all optimizer param-group setups (LLRD and non-LLRD,
    Phase 1 and Phase 2).

## Usage

```bash
python train_mtl_v3.py --use-tal --tal-alpha 2.0 --tal-lr-mult 2.0
```

To compare against baseline (no TAL):

```bash
python train_mtl_v3.py
```

## Reference

- Feng et al. "TOOD: Task-aligned One-stage Object Detection" AAAI 2021.
  [[arXiv](https://arxiv.org/abs/2108.07755)]
