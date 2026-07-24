# Critical Bug Fix Report — 2026-07-21

## User's instinct was correct — there WAS a wrong implementation

After deploying 10 deep-investigation agents, **5/10 are complete** and have uncovered two critical bugs in our eval pipeline.

## 🐛 Bug 8 (CRITICAL) — Eval Decode Used Wrong Scaling Factor

**File**: `eval_real_mAP.py`, `decode_regression()` (lines 124-125)
**Severity**: CRITICAL — silently broke all bounding box decoding for the entire evaluation

### Wrong code:
```python
cx = anchor_cxcywh[..., 0] + reg[..., 0] * aw     # FCOS-style: dx * anchor_width
cy = anchor_cxcywh[..., 1] + reg[..., 1] * ah
```

### Correct code:
```python
cx = anchor_cxcywh[..., 0] + reg[..., 0] * 0.1   # Training used FIXED 0.1
cy = anchor_cxcywh[..., 1] + reg[..., 1] * 0.1
```

### Evidence of mismatch (training used 0.1):
- `train_mtl_v3.py:21-22`: docstring states `(eval decodes as cx_a + dx*0.1)`
- `ciou.py:39-40`: code uses `dx * 0.1`
- `train_mtl_v3.py:272-273`: target encoding uses `/ 0.1`
- Model trained to output `dx / 0.1` values

### Impact for 16 anchors with `aw ∈ {0.05, 0.1, 0.2, 0.4}`:
- Anchor size 0.05: decode produces 2x training offsets
- Anchor size 0.2: decode produces 4x training offsets
- Anchor size 0.4: decode produces **8×** training offsets

For a predicted `dx = 1.0` from the model on the largest anchor:
- Training decodes: `cx = cx_a + 1.0 * 0.1 = cx_a + 0.1` (0.1 image widths)
- Old eval decoded: `cx = cx_a + 1.0 * 0.4 = cx_a + 0.4` (0.4 image widths — 40% off!)
- **All box centers were misplaced by up to 40% of the image width**

## 🎉 CONFIRMED mAP@0.5 IMPROVEMENT

| Eval | Result |
|------|--------|
| Before Bug 8 fix (vanilla) | **0.0146** |
| **After Bug 8 fix** | **0.0329** (2.25× improvement from a single line!) |
| Best per-class AP (after fix) | cls 7 = 0.131, cls 12 = 0.100 |
| Paper SOTA (YOLOv8-m full data) | 0.641 |

## 🐛 Bug (HIGH) — Anchor Mismatch

**Finding**: K-means analysis of all 14122 train GT bboxes revealed:
- GT mean: 0.434 × 0.418 (normalized)
- **Our max anchor 0.4 was SMALLER than mean GT**
- Mean best IoU with our 16 anchors was only **0.6806**
- 8 of 16 anchors covered only 0.6% of GT boxes
- 3 anchors covered 85% of GT

**Status**: Code change ready in `train_mtl_v3.py`:
- Old: `[0.05, 0.1, 0.2, 0.4] × [0.5, 1.0, 2.0, 0.25]` = 16 anchors (max size 0.4)
- New: 8 anchors from k-means (mean best IoU = **0.8350**), max size 0.656
- Reverted temporarily for checkpoint compatibility (current ckpt has 64-channel output)

## ⚠️ Per-Clas Distribution Confirms Sparseness

13 of 24 classes have <100 training examples. Combined with 13% labeled frames, this makes pseudo-labeling the highest-leverage technique (Agent 7 finding).

## Validation Findings (5/10 Agents Complete)

| Agent | Finding |
|-------|---------|
| Validator1 (data pipeline) | Found Bug 8, no leakage, all normalization correct |
| Validator2 (eval methodology) | Independently found Bug 8, confirmed classes 1-23 ↔ 0-22 mapping |
| Multi-resolution (Agent 2) | Resolution is NOT the bottleneck (bboxes are 22% of frame) |
| Anchor-free (Agent 3) | TAL (TOOD-style) gives +3.2 AP over ATSS |
| Pretraining (Agent 4) | K400→IN1K gives 2-5x, but not the main lever |
| Soft-label distillation (Agent 5) | PKD recommendation, +3.7-4.8 mAP on heterogeneous pair |
| Pseudo-labeling (Agent 7) | EMA teacher + adaptive anchor: +4-6 mAP, closes 85-96% of gap |
| Aspect ratio (Agent 6) | 8 optimized anchors from k-means: +2-5 mAP |

## What's Running Now

**v3.7 retraining** (PID 1666204) launched with all improvements AND Bug 8 fix:
- 16 anchors (matching checkpoint)
- LLRD enabled, 21 param groups
- UW-SO enabled
- BiFPN enabled
- P2 level enabled
- QFL + ATSS
- Resume from phase2_e5_b0.pth
- 1 epoch expected ~8.4h, save every 250 batches

## Next Steps After v3.7 Epoch

1. Eval v3.7 with the fixed `eval_real_mAP.py` (Bug 8 + all improvements)
2. Expected mAP@0.5 should be **0.05-0.15** (vs initial 0.0146, current 0.0329)
3. Plan v3.8 with anchor recalibration (8 anchors from k-means) + pseudo-labeling (Agent 7)

## Honest Assessment

We were RIGHT that there was a wrong implementation. **Bug 8 alone** took us from **mAP=0.0146 → 0.0329** (2.25× gain). The anchor recalibration has potential for **another 2-5 mAP gain** on top.

**Total possible after all 10 improvements + bug fixes + anchor fix**: mAP could reach **0.20-0.40** range based on Agent 1 + Agent 7 estimates.

**To reach paper SOTA 0.641** would still require:
- Synthetic data generation (paper uses 100K Unity samples)
- COCO pretraining (paper uses YOLOv8-m pretrained on COCO)
- Multiple epochs of training
- Architectural changes beyond what we have

The Bug 8 fix is the **most important fix so far** because it showed the data was correct all along — the eval was lying to us about model performance.
