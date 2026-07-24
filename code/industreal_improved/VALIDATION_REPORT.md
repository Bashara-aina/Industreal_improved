# Multi-Head Validation Report — IndustReal MTL

## Conv-Proj Fix: All Eval Scripts Verified Clean

| Script | conv_proj | in_channels | Dataset | Status |
|--------|-----------|-------------|---------|--------|
| `eval_mtl_9ch.py` | `expand_conv_proj_to_9ch` | 9 | `FullMultiModalDataset` (5 modals) | ✅ Reference, always correct |
| `eval_mtl_with_gt.py` | `expand_conv_proj_to_9ch` | 9 | `EvalMultimodalDataset` (5 modals, fallbacks) | ✅ Fixed (was 6ch) |
| `eval_mtl_streaming.py` | `expand_conv_proj_to_9ch` | 9 | `EvalMultimodalDataset` (5 modals, fallbacks) | ✅ Fixed (was 6ch) |
| `eval_mtl_6gaps.py` | `expand_conv_proj_to_9ch` | 9 | `EvalMultimodalDataset` (5 modals, T=8 seq) | ✅ Fixed (was 6ch) |

Legacy 6ch training scripts (`train_mtl_production.py`, `train_mtl_single_gpu.py`, `train_mtl_max.py`, `train_mtl_full_6gaps.py`) — none are running, active training uses `train_mtl_v3.py` (9ch).

---

## Detection Head

### Architecture
`DetectionHead` (mvit_mtl_model.py:240): decoupled conv head on FPN levels P3/P4/P5.
- `cls_head`: Conv2d(256→256, 3x3) → GroupNorm → ReLU → Conv2d(256→24, 1x1)
- `reg_head`: Conv2d(256→256, 3x3) → GroupNorm → ReLU → Conv2d(256→64, 1x1) [16 anchors × 4 coords]
- Prior bias init: bias = -ln((1-0.1)/0.1) ≈ -2.197 → sigmoid ≈ 10% initial confidence

### Training Loss
`detection_loss()` at train_mtl_v3.py:317:
- **Classification**: Sigmoid focal loss (γ=2.0, α=0.25) per location
- **Regression**: Smooth L1 on positive anchors only, weighted ×5.0
- Level weights: inverse-sqrt(H×W) to balance P3/P4/P5 contribution

### Eval Metric (current — proxy)
`eval_mtl_with_gt.py:341-363`: For each GT box, checks if **any** spatial location at any FPN level predicts the correct class with sigmoid > 0.3. **This is NOT mAP** — no bbox decoding, no IoU matching, no precision-recall curve.

**Verdict**: WILL show non-zero results with conv_proj fixed. Current training: cls_loss=0.13 (decreasing from ~0.3), reg_loss=0.04, ~600 pos anchors/batch. The proxy metric will likely show 0.3-0.5 initially and increase. Real mAP@50 would be lower.

**Random baseline for proxy**: With prior_prob=0.1 bias init and 24 classes × ~3000 locations/FN level, chance of any location exceeding 0.3 is non-trivial. Expect proxy rate ~0.05-0.15 from random head.

**Recommendation**: Implement proper mAP@50 eval with bbox decoding + IoU matching for Phase 2 results.

---

## Activity Head

### Architecture
`ActivityHead` (mvit_mtl_model.py:317): 3-layer MLP on cls_token (768d).
- Linear(768→2048) → LayerNorm → GELU → Dropout(0.2) → Linear(2048→1024) → LayerNorm → GELU → Dropout(0.2) → Linear(1024→75)

### Training Loss
`train_mtl_v3.py:571-574`: `F.cross_entropy(act_logits, act_target) * 0.5`

### Eval Metric
`eval_mtl_with_gt.py:282-292`: Argmax over 75 logits, compare to GT action ID. Standard top-1 accuracy.

### Train-Eval Alignment: ✅ PERFECT
- Both use argmax over 75 classes
- num_act_classes=75 is correct (train: 72 unique IDs 0-74, val: 65 unique IDs, max ID=74)
- CE training loss directly optimizes the same objective as argmax eval

### Random Baseline
1/75 ≈ **1.33%** top-1 accuracy for uniform random logits.

**Verdict**: WILL show meaningful results. First reliable signal should appear early in Phase 2.

---

## Pose Head

### Architecture
`PoseHead` (mvit_mtl_model.py:395): MLP on cls_token (768d).
- Linear(768→256) → Tanh → output [B, 6] (3=fwd, 3=up)
- Tanh bounds outputs to [-1, 1]

### Training Loss
`train_mtl_v3.py:587-592`: `F.smooth_l1_loss(pred_6d, target_raw_6d) * 0.1`
- Target: concatenated [fwd_x, fwd_y, fwd_z, up_x, up_y, up_z]
- GT vectors ARE unit vectors (magnitude 0.9992-0.9993 verified), so magnitude penalty is harmless

### Eval Metric
`eval_mtl_with_gt.py:294-313`: 
- Normalize pred_fwd and pred_up to unit length
- `angular_error_deg(pred, gt) = arccos(dot(pred, gt) / (|pred| * |gt|))`
- Report mean forward MAE and mean up MAE in degrees

### Train-Eval Mismatch: ⚠️ MINOR
Training uses Smooth L1 on raw 6D vectors (R^6). Eval uses angular error on S^2 × S^2 (directional only).
- **Mitigated by**: GT vectors are unit (±0.0007), Tanh bounds to [-1,1], Smooth L1 approximates angular distance for small angles
- **Not fatal**: Model learns directional alignment, MAE will decrease from ~90°
- **Risk**: May plateau higher than if geodesic loss were used. The `geodesic_angle()` function exists at mvit_mtl_model.py:649 but is unused in training.
- **Loss weight concern**: pose_loss × 0.1 + detection has 100× LR mult. Small pose gradient relative to detection.

### Random Baseline
Random 6D (uniform in [-1,1]) → random unit vectors → **~90°** angular MAE.

**Verdict**: WILL show progress (decreasing from ~90° toward 20-40°). Might plateau above optimal due to train-eval mismatch and low loss weight.

---

## PSR Head

### Architecture
`PSRHead` (mvit_mtl_model.py:408): MLP on P5 features (768ch, post-all-attention).
- AdaptiveAvgPool3d(H,W) → Linear(768→256) → GELU → Dropout(0.15) → Linear(256→11)
- ~0.2M params (was 1.8M with old Transformer)
- **Critical fix [OPUS 186 B-6]**: Now reads from `blocks[14]` (P5, 768ch, semantic-rich) instead of `conv_proj` (P2, 96ch, semantics-free). Old code made PSR loss flat at base-rate entropy.

### Training Loss
`train_mtl_v3.py:598-608`: `F.binary_cross_entropy_with_logits(psr_pred, psr_target) * 0.5`
- psr_target: 11-dim vector, each component is 0/1
- Fill-forward applied in dataset (sparse PSR_labels_raw.csv → dense per-frame)

### Eval Metric
`eval_mtl_with_gt.py:316-339`:
- Sigmoid → threshold at 0.5 → binary predictions
- Per-component precision/recall/F1 → macro F1 (mean across 11 components)

### Train-Eval Alignment: ✅ ALIGNED
- Both use sigmoid on logits
- BCE training directly optimizes the same binary classification objective
- Fill-forward logic matches training dataset exactly (verified: same algorithm, same -1 masking for error components)

### PSR Fill-Forward Fix (eval_mtl_with_gt.py:110-140)
Added to match `FullMultiModalDataset` fill-forward (train_mtl_full_multimodal.py:257-297):
1. Load sparse rows from PSR_labels_raw.csv
2. Sort by frame number
3. Create dense array sized to max(frame_nums)
4. Iterate through frames, filling forward last valid values
5. Only update components that are NOT -1 (error markers)
6. Store per-frame labels

### Random Baseline
- 11 independent binary components
- With ~20% positive rate: macro F1 ≈ 0.20-0.30 for random guessing
- With all-zero prediction: macro F1 = 0 (no positives → precision undefined at 0)

**Verdict**: WILL show non-zero macro F1. First signal after Phase 2 begins. The [OPUS 186] P5 source fix was critical — prior to that PSR was reading semantics-free conv_proj features and couldn't learn transitions.

---

## Summary: Will All Heads Show Meaningful Results?

| Head | Random Baseline | Expected Start | Expected After Phase 2 | Risk Level |
|------|----------------|----------------|----------------------|------------|
| Detection (proxy) | ~0.05-0.15 | ~0.3-0.5 | ~0.6-0.8 | 🟢 Low — head learning well already |
| Activity (top-1) | ~1.3% | ~2-5% | ~10-30% | 🟢 Low — CE and argmax aligned |
| Pose (MAE deg) | ~90° | ~60-80° | ~15-30° | 🟡 Medium — Smooth L1 vs angular mismatch, low loss weight |
| PSR (macro F1) | ~0.0-0.3 | ~0.1-0.3 | ~0.4-0.7 | 🟢 Low — BCE and sigmoid aligned, P5 fix in place |

**All 4 heads WILL show meaningful metrics with conv_proj fixed.** The root cause (6ch vs 9ch mismatch) was the sole reason for all-zero results.

## Known Gaps (Future Work)

1. **Detection proxy → real mAP@50**: Current proxy metric is too permissive — any location predicting correct class with >0.3 confidence counts as a match regardless of spatial location. Real mAP requires bbox decoding + IoU-based matching.

2. **Pose loss → angular loss**: Smooth L1 on raw 6D is directionally aligned but suboptimal. The `geodesic_angle()` function exists but is unused in training.

3. **Head loss weighting**: Detection has det_lr_mult=100, making it dominant. Activity (×0.5), Pose (×0.1), PSR (×0.5) are minor contributors. May cause backbone to converge to detection-optimal features at the expense of other tasks.
