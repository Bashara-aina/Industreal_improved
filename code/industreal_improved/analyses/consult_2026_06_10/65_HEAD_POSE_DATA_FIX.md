# 65: Head Pose Data Fix — Un-normalized pose.csv Forward Vectors [2026-06-30]

## The Problem

Our best paper number (forward_angular_MAE_deg = 8.71° at epoch 2) rests on
**un-normalized ground truth targets**. Every training run logs:

```
[_parse_pose 14_assy_0_1] forward vector mean norm 0.030 is not ~1;
check that pose.csv columns 1-3 are unit vectors.
```

Forward vector norms average 0.014-0.030 instead of 1.0 across ALL recordings.
This means:
- The GT vector is 33-70x shorter than a unit vector
- The model learns to predict proportionally short vectors
- Angular MAE is computed AFTER normalization in eval, but TRAINING uses
  un-normalized MSE — so the model is optimized for magnitude, not direction

Opus (63 §PART 4): **"Before you put '8.71°' in a paper, normalize the GT vectors
and recompute, or a reviewer will. This is your one 'working' number and it
currently rests on an un-normalized target — fix it first."**

## Current Data Flow

### 1. Data Loading (`industreal_dataset.py:_parse_pose`, lines 570-640)

```python
pose_file = self.rec_dir / 'pose.csv'
# CSV columns: frame.jpg, forward_x, forward_y, forward_z,
#              position_x, position_y, position_z,
#              up_x, up_y, up_z
pose_data = np.zeros((self._num_frames, 9), dtype=np.float32)
# ... parse CSV rows ...
# Forward vectors in columns 0-2 with norm ~0.014-0.030
# Position in columns 3-5, optionally scaled:
if C.HEAD_POSE_POS_SCALE != 0.0:
    pose_data[:, 3:6] /= C.HEAD_POSE_POS_SCALE  # HEAD_POSE_POS_SCALE=100.0
# Up vectors in columns 6-8 with norm ~0.0something
```

### 2. Training Loss (`train.py:1670-1700` — MSE on raw 9-DoF)

```python
# MultiTaskLoss computes MSE between predicted and TARGET pose vectors
# The GT forward vectors have norm 0.014-0.030
# The model predicts vectors of similar magnitude
# MSE loss = (pred - gt)^2 — both are small, so loss is small
```

### 3. Eval Metric (`evaluate.py:1799-1830` — Angular MAE)

```python
# Doc 03 A.4: Angular MAE in degrees for directional vectors
# (normalize first — raw MLP outputs are not unit vectors)
_forward_pred = _pred[:, 0:3]
_forward_gt   = _target[:, 0:3]
# Normalize both to unit vectors
_fwd_pred_norm = F.normalize(_forward_pred, dim=1)
_fwd_gt_norm   = F.normalize(_forward_gt, dim=1)
# Cosine similarity → angle in degrees
_cos = (_fwd_pred_norm * _fwd_gt_norm).sum(dim=1).clamp(-1, 1)
_deg = torch.acos(_cos) * 180 / math.pi
# MEAN angular MAE over all frames
```

**Crucial disconnect:** The EVAL normalizes to unit vectors before computing angle,
so the angular MAE (8.71°) IS valid — it measures direction only, independent of
magnitude. But the TRAINING loss (MSE on raw 9-DoF) optimizes the model to match
the un-normalized short vectors, NOT the direction.

The model's predicted forward vectors are short (norm ≈ 0.014-0.030, matching
the GT). When normalized for eval, the direction is approximately correct →
angular MAE looks good. But we're MISSING EASY IMPROVEMENT by training with
normalized targets — the MSE would directly optimize the direction we care about.

## What the 8.71° Number Actually Means

The eval correctly computes angular MAE by normalizing first. So 8.71° IS a valid
measure of directional accuracy. The problem is:

1. **It's suboptimal** — training MSE on short vectors doesn't penalize directional
   errors as strongly as unit-vector MSE would
2. **A reviewer will flag it** — the _parse_pose warning is right there in the logs
3. **If GT vectors are truly noisy**, normalizing them reveals the noise

## Required Fixes

### Fix 1: Normalize GT pose vectors at data-load time

```python
# In industreal_dataset.py:_parse_pose, after parsing CSV:
# Columns 0-2: forward vector → normalize to unit length
fwd = pose_data[:, 0:3]
fwd_norms = np.linalg.norm(fwd, axis=1, keepdims=True)
fwd_norms = np.clip(fwd_norms, 1e-8, None)  # Avoid division by zero
pose_data[:, 0:3] = fwd / fwd_norms

# Columns 6-8: up vector → normalize to unit length  
up = pose_data[:, 6:9]
up_norms = np.linalg.norm(up, axis=1, keepdims=True)
up_norms = np.clip(up_norms, 1e-8, None)
pose_data[:, 6:9] = up / up_norms

# Position (columns 3-5) remains in mm (no change needed)
```

**Location:** `industreal_dataset.py:_parse_pose`, after line ~638 (after position
scaling, before return pose_data).

### Fix 2: Remove or update the _parse_pose warning

```python
# Old warning (lines 611-614):
# if fwd_norm > 0.0 and not (0.5 < fwd_norm < 1.5):
#     logger.warning(...)

# After normalization, norms will be exactly 1.0. The warning is no longer needed.
# Remove it or change condition to check after normalization.
```

### Fix 3: Verify eval normalization is still correct

The eval already normalizes at `evaluate.py:1799-1801`. After Fix 1, the GT is
already normalized, so `F.normalize(_forward_gt, dim=1)` is a no-op. The
prediction might also be near-unit if MSE drives it there. Double-check:
we want unit-vector targets → model learns unit-vector predictions → eval
normalization becomes redundant but correct.

## Questions for Opus

1. **Should we normalize at DATA LOAD TIME (industreal_dataset.py) or at LOSS TIME
   (losses.py)?** Data-load is cleaner (affects all downstream consumers) but
   the HEAD_POSE_POS_SCALE=100.0 in config.py suggests the data-load path already
   has scaling logic. Adding normalization there is consistent.

2. **What about position vectors?** Columns 3-5 (position) use HEAD_POSE_POS_SCALE=100.0
   to convert from metres to cm. Is this correct? The eval's position_MAE_mm assumes
   metres input. Should we normalize position to [0, 1] range or keep as-is?

3. **Will normalizing GT change the loss scale significantly?** Currently the MSE
   operates on values ~0.02 (forward). After normalization, values are ~1.0. The
   MSE increases by ~2500x. Does HEAD_POSE_LOSS_WEIGHT=5.0 need adjustment?
   Does KENDALL_LOG_VAR_MAX_POSE=3.0 need adjustment?

4. **The position vectors use raw mm values.** With HEAD_POSE_POS_SCALE=100.0,
   position columns are divided by 100 at load time. But the CSV comment says
   they're in metres — so dividing by 100 converts to cm. The eval (multiplying
   by 1000 for mm) assumes metres input. Is there a units mismatch?

5. **Should we also normalize UP vectors?** The up_angular_MAE_deg values are
   high (95° at epoch 1). Normalizing GT up vectors might help, but it could
   also reveal that the up vectors in the dataset are unreliable (some recordings
   may have noise).

6. **After normalization, re-run a short validation.** The simplest test: load
   the best.pth checkpoint, run evaluate.py on val set with and without the
   normalization fix, compare the 8.71° → ?. Is the number stable, better, or
   worse? This tells us how much of the current performance is from magntidue
   vs direction.

7. **Is the forward vector norm = 0.014-0.030 a dataset issue or a units issue?**
   If the CSV stores forward vectors in some scaled coordinate system (e.g.,
   pixels instead of world units), the small norms are expected. Normalizing
   is the right fix regardless.

## Impact on Paper

If 8.71° remains after normalization:
- **Paper-valid, no change needed.** Just document the fix.

If 8.71° degrades to, say, 15-20° after normalization:
- **Still within RF10 target (35°)** and the paper target (10° est SOTA).
- But you must disclose in the paper: "Initial reported MAE used un-normalized
  targets; after correction, the model achieves X°."
- This is better than a reviewer discovering it.

If 8.71° degrades to >30°:
- **The "one working task" was partially a data artifact.**
- Need to investigate why direction learning degraded with normalized targets.
- Might require retraining with normalized targets.

## Current Values for Reference

From `metrics.jsonl` epoch 2 (the LAST completed full validation):
```
forward_angular_MAE_deg: 8.71   (needs normalization verification)
up_angular_MAE_deg: 95.23       (normalizing up vectors may help)
position_MAE_mm: 118.07         (units may be wrong — metres vs cm vs mm)
```
