# PSR V4 Design: Why V3 gradient is still DEAD, and how to fix V4

## 1. The V3 Problem

- V3 has DETACH_PSR_FPN=False set (wrapper overrides stage_rf4 preset which has detach_psr_fpn=True)
- V3 has post_gelu +4708 (PSR output head activations are alive - forward pass works)
- V3 has PSR head gradient RMS=0.00e+00 (DEAD) - no gradient reaches ANY of the 88 PSR head params
- V3 has psr=0.0000 loss in ALL training steps (no PSR loss ever fires)
- Resolved config confirms DETACH_PSR_FPN=False, MIXED_PRECISION=True, USE_PSR_TRANSITION=True, KENDALL_FIXED_WEIGHTS=True
- This is the 5th exhibit of "code that exists but does not execute"

```
[LIVENESS_GRAD step=2001] ...
  psr_head:DEAD[RMS=0.00e+00|n=88]
  psr_heads:[h0=0.00e+00[DEAD],h1=0.00e+00[DEAD],...,h10=0.00e+00[DEAD]]
```

## 2. Why V3 Fails (root cause analysis)

The PSR gradient death has THREE independent causes, all active simultaneously:

### Cause 1: USE_PSR_TRANSITION=True kills PSR loss on normal batches

In `losses.py:1436-1459`:
```python
if self.use_psr_transition:
    if outputs['psr_logits'].dim() == 3:
        # ... compute transition loss (sequence batches only) ...
    else:
        # Per-frame batch (dim==2): skip PSR loss.
        loss_psr = zero  # <-- THIS KILLS PSR GRADIENT
```

With `USE_PSR_TRANSITION=True` (stage_rf4 preset default), PSR loss is ONLY computed on sequence batches (dim==3). Normal batches (dim==2), which are 3 out of every 4 batches when PSR_SEQ_EVERY_N_BATCHES=4, get zero PSR loss.

### Cause 2: Sequence batch PSR loss is structurally near-zero

Even on the 1-in-4 sequence batches, the `build_transition_targets` function (psr_transition.py) produces Gaussian-smoothed transition targets. For sequences where no component transitions occur (common in short 4-frame windows with fill-forward labels), ALL transition targets are zero. The `binary_focal_loss` with all-zero targets still produces a small loss (~0.04 per element), but this is further:

- Smoothed by `_smooth_cap` (passes through for small losses)
- Multiplied by `PSR_WEIGHT=20.0` under KENDALL_FIXED_WEIGHTS
- Divided by `accum_steps=4`

The net effect: PSR produces a tiny, sporadic gradient that is effectively zero compared to the dominant detection/activity gradients.

### Cause 3: DETACH_PSR_FPN was the WRONG fix target

The V3 fix assumed the gradient death was caused by `detach_psr_fpn=True` in the preset. But the resolved config shows DETACH_PSR_FPN=False is correctly applied. The gradient is DEAD for a completely different reason: the PSR LOSS is structurally zero for most batches. DETACH_PSR_FPN controls whether PSR gradients reach the BACKBONE, not whether PSR head parameters get gradient.

### Evidence chain

```
psr=0.0000 (loss) → loss_psr=0 → no gradient → psr_head:DEAD

Every [DEBUG] line in the training log shows psr=0.0000 because:
- Normal batches: loss_psr = zero (use_psr_transition=True, dim==2)
- Sequence batches: loss is NOT logged to the DEBUG line
  (seq batch uses pbar.set_postfix_str, not logger.info)
```

## 3. V4 Design

### 3.1 V4 Forces - Direct edits to fix the gradient path

Three independent fixes, each sufficient alone. All three together guarantee gradient flow.

#### Fix A: Disable USE_PSR_TRANSITION for PSR repair training

**File**: `src/config.py` or wrapper
**Mechanism**: Override `USE_PSR_TRANSITION = False` in the wrapper (same pattern as DETACH_PSR_FPN override).
**Effect**: PSR loss fires on EVERY batch (dim==2 path), giving dense gradient signal via simple focal loss on per-frame labels. No more skipped batches.

```python
# In wrapper, after preset application:
C.USE_PSR_TRANSITION = False
```

#### Fix B: Force sequence batch on every step (PSR_SEQ_EVERY_N_BATCHES=1)

**File**: wrapper or env var
**Mechanism**: Set env var `PSR_SEQ_EVERY_N_BATCHES=1` so every batch is a sequence batch.
**Effect**: Even without Fix A, every batch gets dim==3 PSR loss. But the transition targets may still produce near-zero loss for short windows.

#### Fix C: Use PSR-only single-task preset (ablation_psr_only)

**File**: V4 launch script
**Mechanism**: Replace `--preset stage_rf4` with `--preset ablation_psr_only` (defined in config.py line 1728).
**Effect**: Only PSR head trains. No detection/activity/pose gradients to compete. PSR gets 100% of optimizer budget. No Kendall weighting interference.

The ablation_psr_only preset has:
- `train_det: False, train_act: False, train_psr: True, train_head_pose: False`
- `use_psr_transition: False` (directly from the preset)
- `PSR_SEQ_EVERY_N_BATCHES=1` (every batch is a sequence batch)

### 3.2 V4 Launch

```bash
# V4: PSR-only training with dense gradient path
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 \
DETACH_PSR_FPN=False \
USE_PSR_TRANSITION=False \
PSR_SEQ_EVERY_N_BATCHES=1 \
KENDALL_FIXED_WEIGHTS=1 \
python3 -u scripts/train_psr_repair_wrapper.py \
    --preset ablation_psr_only \
    --batch-size 2 \
    --resume src/runs/rf_stages/checkpoints/best.pth \
    --start-epoch 18 \
    --max-epochs 23
```

**Why these changes:**
- `--preset ablation_psr_only`: PSR-only preset (no detection/activity interference)
- `USE_PSR_TRANSITION=False`: Dense per-frame focal loss on EVERY batch
- `PSR_SEQ_EVERY_N_BATCHES=1`: Every batch uses sequence path with causal transformer
- `DETACH_PSR_FPN=False`: Allow PSR gradient to flow into backbone (safe in PSR-only mode)
- Resume from epoch 18 best.pth (current best F1=0.7018)
- Run for 5 epochs (18-23), sufficient to measure gradient flow

**Verification checks:**
1. `psr=` in [DEBUG] log lines shows NON-ZERO values
2. `psr_head:ALIVE[RMS=...]` in [LIVENESS_GRAD] - PSR head params show non-zero gradient
3. `psr_heads:[h0=...ALIVE,...]` - individual output heads show gradient
4. PSR F1 > 0.70 after 5 epochs

### 3.3 V4 Expected Outcomes

| Scenario | F1 after 5 epochs | Diagnosis |
|----------|-------------------|-----------|
| **V4 F1 > 0.78** | Implementation fix works - gradient flows | PSR can learn |
| **V4 F1 ~ 0.70** | Gradient flows but PSR capacity is limit | Architectural issue |
| **V4 F1 < 0.70** | Gradient still dead or data issue | Deeper problem |

If V4 F1 > 0.78: Merge into full multi-task training with USE_PSR_TRANSITION re-enabled (but with gradient warmup).
If V4 F1 ~ 0.70: PSR head capacity limits (try wider transformer or more layers).
If V4 F1 < 0.70: Investigate LeakyReLU gradient 0.01 factor (try ReLU instead).

### 3.4 Risk Mitigation

- **Checkpoint safety**: Resume from best.pth, not crash_recovery.pth. Best.pth represents epoch 18 F1=0.7018.
- **Minimal changes**: Only preset/env var changes. No code modifications to model.py, losses.py, or train.py.
- **Quick loop**: 5 epochs at PSR-only preset should complete in <3 hours on RTX 3060.
- **Rollback**: If V4 fails, the checkpoint is preserved. Training can resume from best.pth with original params.
