# 78 — Detection Head: Collapse-Free Training to Converged mAP [2026-07-01]

**Goal:** Guarantee the detection head trains to a stable, converged mAP50 without collapsing, gradient-starving, or diverging. Detection is the PAPER HEADLINE metric (alongside forward-gaze 8.71°) — it must work.

**Source files (all paths relative to `code/industreal_improved/src/`):**
- `config.py:54–66, 178–200, 568–586, 605–676, 756–802, 867–886` — all detection configs
- `models/model.py:1804–1808` — DetectionHead construction
- `training/losses.py:1038–1041, 1214–1248` — FocalLoss + GIoU warmup
- `evaluation/evaluate.py:1095–1120` — detection metric computation (mAP)

---

## 1. The Four Detection Collapse Mechanisms (All Fixed)

Detection on IndustReal has historically collapsed via four independent mechanisms. Each has a fix. Opus must confirm all four are active and correct.

### 1.1 Mechanism A: Cumulative Negative Gradient (OHEM + Asymmetric Gamma)

**Problem**: With ~0.01% positive anchors (~20/173K per image), the cumulative negative gradient from all 173K background anchors drives `cls_logits` to -16 over ~850 steps → all-background predictions → mAP=0.

**Fix 1: OHEM (config.py:656-661)**
```python
DET_OHEM_ENABLED = True
DET_OHEM_RATIO = 2.0      # Keep only 2:1 negatives-to-positives
DET_OHEM_MIN_NEG = 32     # Minimum 32 negatives for stability
```

Keeps only the **hardest negatives** at a 2:1 ratio relative to positive anchors. This breaks the cumulative negative gradient cycle while preserving discriminative negative examples. Note: `DET_OHEM_MIN_NEG=32` is lower than the original 128 — this was tuned to prevent negatives from overwhelming low-pos batches.

**Fix 2: Asymmetric Gamma (config.py:663-675)**
```python
DET_ASYMMETRIC_GAMMA = True
DET_GAMMA_POS = 0.0       # No gamma suppression on positives
DET_GAMMA_NEG = 1.5       # Mild gamma on negatives
```

With `gamma_pos=0`, positives always contribute `1 × CE` gradient — never suppressed. With `gamma_neg=1.5`, negatives at p=0.07 contribute ~0.27 × CE (moderate). Together with OHEM at 2:1, this guarantees positives dominate the gradient direction per batch.

### 1.2 Mechanism B: GT Frame Starvation (DET_GT_FRAME_FRACTION)

**Problem**: Only ~0.7% of frames have GT boxes. Activity-balanced sampler draws GT-bearing frames even more rarely. The detection head sees a positive box in <1% of steps.

**Fix: DET_GT_FRAME_FRACTION (config.py:786-801)**
```python
DET_GT_FRAME_FRACTION = float(os.environ.get('DET_GT_FRAME_FRACTION', '0.90'))
```

Targets an **absolute** per-batch GT fraction. At 0.90, 90% of each batch is GT-bearing — independent of base OD density. This guarantees the detector positive gradient on (nearly) every step. The actual value is overridden per-stage in `apply_preset()`:
- Detection-dominant stages (RF1/RF2, recovery_det_only): **0.90**
- Multi-head stages (RF3+): **0.40** — leaves room for activity group balance

**Verification**: log `det_gt_fraction` at each epoch start — should be 0.90 for RF1/RF2, 0.40 for RF3+.

### 1.3 Mechanism C: Regression Gradient Shock (Reg Warmup + Detach)

**Problem**: When regression head is freshly initialized (`--reinit-heads`), the first GT boxes (~step 751) produce a huge regression loss from random predictions, propagating gradient shock through shared FPN → collapses both regression and classification.

**Fix 1: Regression Warmup (config.py:867-873, losses.py:1229-1234)**
```python
REINIT_REG_WARMUP_STEPS = 1000
REINIT_REG_WARMUP_INIT_MULT = 0.01
```

Linear ramp from 1% → 100% regression loss over 1000 steps:
```python
_reg_ramp = _reinit_reg_wm + (1.0 - _reinit_reg_wm) * float(self._step_counter) / _reinit_reg_ws
reg_loss = reg_loss * _reg_ramp
```
At step 0: 0.01 × reg_loss (negligible). Step 500: ~0.5 ×. Step 1000: 1.0 ×.

**Fix 2: Detach Reg from FPN (config.py:888-890) — NOTE: ALL RF PRESETS OVERRIDE TO False**
```python
DETACH_REG_FPN = True   # Global default (no-preset runs)
# BUT every RF stage preset overrides to False:
# RF1: 'detach_reg_fpn': False  — regression gradient MUST reach FPN during bootstrap
#     (reg warmup, not detach, is the correct gradient shock guard)
# RF2-RF10: all set 'detach_reg_fpn': False
# See config.py apply_preset() lines 1296-1305 for RF1's explicit rationale.
```

After Opus v11 (2026-06-21), the reg-detach was disabled for ALL stages based on evidence that regression gradient signal is necessary for the FPN to escape background equilibrium. The `REINIT_REG_WARMUP_STEPS=1000` with `INIT_MULT=0.01` (Fix 1) provides sufficient gradient shock protection without severing the FPN gradient path.

**Fix 3: GIoU Negative Floor (losses.py:1236-1246)**
```python
loss_det = torch.where(
    loss_det < 0,
    NEG_SLOPE * loss_det,  # NEG_SLOPE = 0.0 — zero floor, no negative
    loss_det,
)
```

GIoU can be negative (range [-1, 1]). With Kendall precision up to 54.6×, a negative GIoU of -1.5 produces -82 contribution → Kendall divergence. The zero floor eliminates this.

### 1.4 Mechanism D: Empty-Frame Background Loss (Maintenance)

**Problem**: Between GT-bearing batches, the detection head sits idle for ~2200 steps — its weights drift as the backbone changes for other tasks. When GT finally arrives, the weights have decayed.

**Fix: DET_EMPTY_SAMPLE (config.py:764-766)**
```python
DET_EMPTY_SAMPLE = 2048     # subsample 2048 anchor locations
DET_EMPTY_BG_SCALE = 0.05   # small background focal loss
```

On frames with zero GT boxes, subsample 2048 anchor locations and compute a tiny background focal loss (scale 0.05). This keeps detection head weights alive between GT batches (grad norm ~0.005-0.9 per empty image vs 0.0 without).

---

## 2. Detection Architecture Overview

### 2.1 RetinaNet Head (model.py:1804-1808)
```python
self.detection_head = DetectionHead(
    in_channels=256, num_classes=C.NUM_DET_CLASSES,  # 24
    detach_reg_fpn=getattr(C, 'DETACH_REG_FPN', False),
)
```

Standard RetinaNet: FPN P3-P7 → cls_subnet + reg_subnet shared across all pyramid levels. 9 anchors per location. 24 output classes (22 assembly states + background + error_state).

### 2.2 Anchor Configuration (config.py:428-440)
```python
ANCHOR_SIZES = (96, 160, 256, 384, 512)
DET_POS_IOU_THRESH = 0.4
DET_POS_IOU_TOP_K = 9
DET_POS_IOU_IOU_FLOOR = 0.2
```

The combination of lower IoU threshold (0.4), top-K force-match (9 anchors per GT), and IoU floor (0.2) ensures every GT box gets ~6-10 positive anchors, solving the small-object positive-starvation problem. This was a root cause fix in Opus v8 §3.

---

## 3. Focal Loss + GIoU (losses.py:1038-1041, loss forward:1214-1248)

### 3.1 FocalLoss Parameters
```python
self.det_loss_fn = FocalLoss(
    alpha=C.FOCAL_ALPHA,           # 0.25 (standard RetinaNet)
    gamma=C.FOCAL_GAMMA,           # 2.0
    pos_iou_thresh=C.DET_POS_IOU_THRESH,  # 0.4
    neg_iou_thresh=C.DET_NEG_IOU_THRESH,  # 0.4
    class_alphas=getattr(C, 'DET_CLASS_ALPHAS', {}),
)
```

The `DET_CLASS_ALPHAS` dict (config.py:613-648) provides per-class focal alpha overrides. After the Opus v4 fix, these are correctly mapped to model indices (not CSV class indices). Classes with AP=0.0 get α=0.94-0.96 (strong positive gradient), dominant classes get α=0.78-0.85 (better suppression).

### 3.2 GIoU Weight
```python
giou_weight = float(getattr(C, 'GIOU_WEIGHT', 2.0))  # config.py:649
loss_det = cls_loss + giou_weight * reg_loss  # losses.py:1235
```

GIoU weight 2.0 means regression contributes 2× the raw loss of classification.

### 3.3 OHEM + Asymmetric Gamma in FocalLoss.forward()
The FocalLoss class (not shown fully here but at losses.py:~500-690) applies:
1. Positive/negative anchor separation via anchor-target IoU matching
2. Positive count clamp (prevents `num_pos=0` division errors)
3. OHEM: sort negatives by `p_t` (focal weight), keep only top-K hardest
4. Asymmetric gamma: `p_t_pos^0 * CE` vs `p_t_neg^1.5 * CE`
5. Per-class alpha weighting from `DET_CLASS_ALPHAS`

---

## 4. Eval Metrics: Correct and Stable

### 4.1 mAP50 Computation (evaluate.py)
The detection evaluation follows standard COCO mAP:
- `DET_EVAL_SCORE_THRESH = 0.001` (config.py:584) — low enough to capture all predictions
- `DET_EVAL_MAX_PER_IMAGE = 300` (config.py:585)
- `DET_EVAL_NMS_IOU_THRESH = 0.5` (config.py:586)

The confusion matrix at evaluate.py:1095-1120 filters to keep only classes with GT or predicted data, preventing empty-class display issues.

### 4.2 Detection Probe (config.py:51-52)
```python
LIVENESS_EVERY = 500
DET_DEBUG_EVERY = 50  # --reinit-heads only
```

At `--reinit-heads`, every 50 steps logs `cls_mean`, `pos_mask`, `bestIoU_max`, `num_pos`, `score_max`, `score_p99`, `mAP50_probe`. This gives early warning of collapse (cls_mean dropping toward -16) or positive starvation (num_pos=0).

---

## 5. What Could Still Go Wrong

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Detection head converges to mAP50=0.05-0.10 (barely above chance) | Medium | This is the detection death-spiral recovery floor. If stuck here, train for 10+ epochs — slow improvement is normal for small-object detection on sparse GT. |
| DET_OHEM_RATIO=2.0 too aggressive in low-pos batches (only 1-2 positives → only 2-4 negatives) | Low | The `DET_OHEM_MIN_NEG=32` floor guarantees at least 32 negatives even when positives are scarce. |
| Detach_Reg_FPN=False in all stages — reg gradient reaches FPN, warmup is the guard | Low | Regression gradient flows into FPN in all stages. The warmup protects against shock during the first 1000 steps after --reinit-heads. No action needed. |
| GIoU negative values (loss < 0) produce gradient that pushes Kendall log_var in wrong direction | Low | The zero floor at losses.py:1242 prevents negative loss_det from reaching Kendall. |
| DET_CLASS_ALPHAS have wrong model indices after config rename | Low | Verify each class alpha maps to its model index. Index 20 (class 21) at alpha=0.96 is the most stuck class (train=709, AP=0.0) — if this is mis-indexed, that class never learns. |

---

## 6. Expected mAP Trajectory

| Epoch | Expected mAP50 | Notes |
|-------|---------------|-------|
| 0-2 | 0.0-0.005 | Warmup + OHEM stabilizing |
| 3-5 | 0.01-0.05 | Positive gradient building, reg warmup prevents regression shock |
| 6-10 | 0.05-0.15 | Stable training, slow climb |
| 11-20 | 0.15-0.35 | Main improvement phase |
| 21-50 | 0.35-0.55 | Convergence, diminishing returns |
| 51-100 | 0.50-0.65 | Max convergence (paper headline range) |

**Expected final: mAP50 ≈ 0.50–0.65** (YOLOv8m was 0.838 on same data — gap is expected given single-backbone multi-task and consumer GPU constraints).

---

## 7. Final Go/No-Go Criteria (Epoch 5)

| Signal | Pass | Borderline | Fail |
|--------|------|-----------|------|
| `cls_mean` | -3.0 to -1.0 | -5.0 to -3.0 | < -5.0 (collapse) or > 0.0 (no suppression) |
| `num_pos` per batch | ≥ 5 avg | 2-5 avg | < 2 avg |
| `mAP50_probe` | ≥ 0.02 | 0.005-0.02 | < 0.005 |
| `score_max` | ≥ 0.10 | 0.05-0.10 | < 0.05 |
| GT-bearing batch fraction | ≥ 0.30 | 0.15-0.30 | < 0.15 |

**If all pass at epoch 5**: detection is on a healthy trajectory. Continue to convergence.

**If fail (mAP50 < 0.005 at epoch 5)**: the death spiral is not broken. Increase `DET_GT_FRAME_FRACTION` to 0.95, decrease `DET_OHEM_RATIO` to 1.0.
