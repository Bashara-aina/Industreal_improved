# PSR Gradient Deep Debug — File-157 DETACH Fix Verification

**Date:** 2026-07-07
**Status:** Root cause found. PSR head gradient is killed by staged-training parameter freezing in `_set_stage_requires_grad`, NOT by FPN detach.

## Symptom

- DETACH_PSR_FPN=False confirmed set in wrapper output
- PSR head gradient RMS=0.00e+00 reported by head liveness probe
- PSR activations on seq frames are +4700 (forward pass works, head is producing output)
- Loss function receives the logits and computes loss, but backward produces no gradient for PSR head weights

## Root Cause: Dual Kill

There are TWO mechanisms that independently kill the PSR head gradient during stages 1 and 2:

### Primary Kill (Parameter Freezing): `_set_stage_requires_grad` in `train.py:779-803`

During stages 1 and 2, `_set_stage_requires_grad` explicitly freezes PSR head parameters:

```python
# train.py:779-782 (stage 1)
for name, p in model.named_parameters():
    if 'activity_head' in name or 'psr_head' in name:
        if _REINIT_HEADS_ACTIVE and 'activity_head' in name:
            continue  # activity skips freeze when reinit-active
        p.requires_grad = False
```

```python
# train.py:797-803 (stage 2)
for name, p in model.named_parameters():
    if 'activity_head' in name or 'psr_head' in name:
        if _REINIT_HEADS_ACTIVE and 'activity_head' in name:
            continue  # activity skips freeze when reinit-active
        p.requires_grad = False
```

Key observation: `activity_head` has a conditional bypass (`_REINIT_HEADS_ACTIVE and 'activity_head' in name`), but `psr_head` has NO bypass. `psr_head` is ALWAYS frozen during stages 1 and 2 regardless of any flag.

When `requires_grad=False`, PyTorch autograd does not track the parameter. The PSR head weights receive NO gradient accumulation regardless of what the loss function computes. The forward pass works (hence +4700 activations), but `.backward()` produces no `.grad` for any `psr_head.*` parameter.

### Secondary Kill (Loss Weighting): Criterion Kendall staging in `losses.py:1745-1777`

Even if parameter freezing were fixed, the criterion's Kendall staging would independently kill the PSR gradient by zeroing the precision weight:

```python
# losses.py:1756-1768
if stage == 1:
    prec_psr = prec_psr * 0   # kills PSR gradient contribution
    lv_psr = lv_psr * 0       # kills log_var_psr gradient
elif stage == 2:
    prec_psr = prec_psr * 0   # kills PSR gradient contribution
    lv_psr = lv_psr * 0       # kills log_var_psr gradient
```

When `prec_psr = 0`, the Kendall total computed at line 1828 is:

```python
total = total + prec_psr * (loss_psr * _psr_w) + lv_psr
#       leaf(0) +      0 * (loss_psr * 20.0) + 0
#       = 0 + 0 + 0 = 0
```

The total loss is zero. While `total` has a computation graph (`SumBackward`), the gradient of `prec_psr * (loss_psr * _psr_w)` w.r.t. `loss_psr` is `prec_psr = 0`. So the scalar 0 multiplier blocks gradient propagation to PSR head weights through the loss.

### Why DETACH_PSR_FPN=False Doesn't Help

The DETACH_PSR_FPN flag controls whether FPN features are `.detach()`'d before feeding into the PSR head (model.py:2104-2107). With `DETACH_PSR_FPN=False`:

```python
# model.py:2104-2107
if getattr(C, 'DETACH_PSR_FPN', False):
    p3_t = p3_t.detach()  # NOT executed when False
    p4_t = p4_t.detach()
    p5_t = p5_t.detach()
```

The features are NOT detached, so the PSR head backward CAN flow into FPN/backbone. However:

1. **The PSR head params are frozen** (`requires_grad=False` in stages 1/2) — they get no gradient regardless
2. **The train.py seq-path gradient snapshot/restore** (lines 1397-1399) correctly restores only backbone and FPN gradients to preserve accumulated non-seq gradients — this part of the DETACH fix works correctly
3. **The PSR head params would still get zero gradient** because the Kendall total zeroes their loss contribution

### Why the seq batch sets `train_psr=True` but criterion still stages it

In train.py seq path (lines 1308-1311):
```python
criterion.train_psr = True
criterion.train_pose = False
criterion.train_act = False
criterion.train_det = False
```

This tells the criterion which tasks to compute losses for, but the Kendall staging block at lines 1745-1777 checks `self._current_epoch` and overrides `prec_psr` regardless of which `train_*` flags are set. There is no `_is_psr_only_batch` check.

## Fix Required

### Fix 1 (Critical): Remove PSR head parameter freezing in stages 1 and 2

In `train.py`, the `_set_stage_requires_grad` function at lines 779-782 and 799-803 should add `psr_head` to the reinit-active bypass (same pattern as `activity_head`):

```python
if 'activity_head' in name or 'psr_head' in name:
    if _REINIT_HEADS_ACTIVE and ('activity_head' in name or 'psr_head' in name):
        continue
    p.requires_grad = False
```

Or better, skip freezing PSR head entirely when CFG_TRAIN_PSR is True with sequence mode, since the seq batch path explicitly manages PSR training.

### Fix 2 (Secondary): Bypass Kendall staging for PSR-only batches

In `losses.py:1745-1777`, add a guard so that staging does not zero PSR precision when this is a PSR-only batch (all other train_* flags are False):

```python
_is_psr_only_batch = self.train_psr and not (self.train_det or self.train_pose or self.train_act)
if stage == 1:
    if not _is_psr_only_batch:
        prec_hp = prec_hp * 0
        lv_hp = lv_hp * 0
        prec_psr = prec_psr * 0
        lv_psr = lv_psr * 0
```

## Configuration Context

The user runs with:
- DETACH_PSR_FPN=False (confirmed correct in wrapper)
- Staged training active (default)
- Epochs in stage 1-2 range (epoch < 16)
- KENDALL_FIXED_WEIGHTS=False (default, so standard Kendall path with staging)
- PSR head was re-initialized with LeakyReLU (Opus 140 repair)

## Verification

To verify the fix, the head liveness probe output (train.py:2040-2115) should show `psr=NonZeroGradNorm` instead of `psr=0.00e+00` on seq batch steps during stages 1 and 2.
