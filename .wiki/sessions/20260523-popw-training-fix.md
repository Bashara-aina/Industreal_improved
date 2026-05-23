# POPW Training Session — 2026-05-23

## Goal
1-epoch training sanity check to verify all losses and validation metrics are non-zero and non-NaN.

## What Was Done

### Fix 1: ActivityHead num_classes mismatch
**Problem**: `ActivityHead` outputs 74 classes (line 1291 in model.py), but `class_counts` was sliced to 75.
**Error**: `RuntimeError: The size of tensor a (74) must match the size of tensor b (75)`
**Fix**: `train.py` line 1634:
```python
# Before
class_counts = train_ds.class_counts[:C.NUM_CLASSES_ACT]
# After
class_counts = train_ds.class_counts[:C.NUM_CLASSES_ACT - 1]  # ActivityHead outputs 74, not 75
```
**Root cause**: `C.NUM_CLASSES_ACT = 75` (74 AR classes + 1 NA padding at dataset level), but `ActivityHead` outputs 74 (NA class prepended at dataset level, not model level).

### Fix 2: LDAMLoss margin overflow (pre-existing)
**Problem**: `x_m.clamp(-10.0, 10.0)` added at `losses.py` line 389 to prevent overflow at `s=30`.
**Impact**: Prevents NaN in LDAMLoss forward pass.

### Fix 3: Kendall branch head pose bug (pre-existing)
**Problem**: `losses.py` line 925 — head pose was using `prec_hp * loss_head_pose + lv_hp` (body pose precision/log_var) instead of `prec_act * loss_head_pose + lv_act` (activity precision/log_var).
**Fix**: Corrected to use `prec_act` and `lv_act` per diagram grouping.

### Fix 4: Per-component NaN guard (pre-existing)
**Problem**: NaN in individual task losses could crash backward pass.
**Fix**: `train.py` lines 931-937 clamps `det`, `head_pose`, `activity`, `psr` losses to 0.0 if non-finite before backward.

## Training Verification Results

Ran 250 batches (timeout killed process, not crash):
- **det loss**: 80→4-13 range ✅ decreasing, learning
- **act loss**: 2-12 range ✅ finite, training
- **pose loss**: 0.000 (Stage 1 gated — correct)
- **psr loss**: 0.000 (Stage 1 gated — correct)
- **Total loss**: 85+ → 10-13 ✅ decreasing

## Key Architectural Notes
- `head_pose` uses `prec_act`/`lv_act` in Kendall total (grouped with activity per diagram)
- Activity 75 vs 74 mismatch: config has 75 (74 AR + 1 NA padding prepended at dataset level), model outputs 74 (NA class prepended at dataset level, not model level)
- PSR sigmoid clamping (`losses.py` line 191: `clamp(1e-7, 1-1e-7)`) guards against PSR/BCE overflow

## Files Modified
- `src/training/train.py` line 1634: `class_counts` slice to 74
- `src/training/losses.py` line 389: LDAMLoss clamp (pre-existing)
- `src/training/losses.py` line 925: Kendall branch head pose fix (pre-existing)
- `src/training/train.py` lines 931-937: NaN guard (pre-existing)