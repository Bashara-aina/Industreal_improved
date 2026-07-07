# D3 Detection mAP = 0.00009 -- Root Cause Analysis

## Summary

The D3 detection head produces mAP=0.00009 on the full 38k evaluation set because of three independent problems that compound: 92% of frames have zero GT boxes and flood the precision-recall curve with false positives; the classification head systematically confuses assembly states (only 46% of GT boxes get a correctly-classified prediction); and the box regression produces mean best IoU of 0.234, well below the 0.5 threshold.

Only 44 of 3102 GT boxes (1.4%) are detected at IoU > 0.5 with the correct class.

---

## 1. Data Analysis: Empty Frame Flooding

The evaluation set has 38,036 frames with only 3,102 total GT boxes.

| Metric | Value |
|--------|-------|
| Total frames | 38,036 |
| Frames with GT | 3,102 (8.1%) |
| Frames without GT | 34,934 (91.9%) |
| Total GT boxes | 3,102 |
| Total predictions | 3,983,551 |
| Predictions on empty frames | 3,831,469 (96.2%) |
| Predictions on GT frames | 152,082 (3.8%) |
| Avg predictions per frame | 104.7 |
| Avg predictions per empty frame | 109.7 |
| Avg predictions per GT frame | 49.0 |

Every prediction on an empty frame is an automatic false positive. With ~3.8M false positives across 38k frames, the precision-recall curve is collapsed before any true positives can contribute. At IoU=0.5 with COCO interpolation, a single false positive at high confidence can destroy the AP for an entire class.

## 2. Classification Head Analysis: Severe Class Confusion

The detection head systematically predicts certain classes as defaults, ignoring many of the 24 assembly state classes.

### Per-class prediction volume vs GT volume

| Class | Predictions | GT Boxes | Ratio (pred:GT) | AP@0.5 |
|-------|-------------|----------|-----------------|--------|
| 7 | 972,615 | 380 | 2,559:1 | 0.00006 |
| 10 | 1,549,085 | 251 | 6,171:1 | 0.00007 |
| 12 | 74,682 | 430 | 174:1 | 0.00124 |
| 17 | 514,795 | 263 | 1,957:1 | 0.00000 |
| 22 | 156,319 | 378 | 414:1 | 0.00002 |
| 0 | 260,543 | 331 | 787:1 | 0.00000 |
| 8 | 267,862 | 20 | 13,393:1 | 0.00000 |
| 4 | 20,092 | 324 | 62:1 | 0.00000 |
| 2 | 28,724 | 0 | -- | 0.00000 |
| 3 | 55,192 | 0 | -- | 0.00000 |

Classes 7 and 10 account for 63% of all predictions but only 20% of GT. Classes 2 and 3 have zero GT but generate tens of thousands of predictions (the model hallucinates these classes).

### Classification confusion: best-matching prediction per GT box

Only 1,199 of 2,601 GT boxes (46%) have their best-overlapping prediction in the correct class. The other 54% have a wrong class assigned. Key confusion patterns:

| GT Class (true) | Predicted Class (best match) | Count |
|-----------------|------------------------------|-------|
| 22 (11101111110) | 12 (11110110001) | 223 |
| 21 (11101011110) | 12 (11110110001) | 151 |
| 22 (11101111110) | 10 (11110111100) | 94 |
| 10 (11110111100) | 12 (11110110001) | 71 |
| 12 (11110110001) | 10 (11110111100) | 68 |
| 13 (11110111101) | 12 (11110110001) | 52 |
| 4 (10010100000) | 3 (10010010000) | 45 |
| 4 (10010100000) | 22 (11101111110) | 45 |

The model treats class 12 as a default "catch-all" for many assembly states, particularly classes 21, 22, 13, and 10. The similarity of the binary state strings suggests these classes represent similar assembly stages that the model cannot distinguish.

Classes that are NEVER predicted at any confidence:
- Class 1 (background): 0 predictions
- Class 13: 0 predictions
- Class 16: 0 predictions
- Class 19: 0 predictions
- Class 23 (error_state): 0 predictions

These classes generate zero predictions despite having GT boxes (classes 13: 57 GT, 16: 27 GT, 19: 39 GT). The model has learned to never output these classes.

### Per-class GT detection rate (IoU > 0.5, correct class)

| Class | Detected / Total | Rate | 
|-------|-----------------|------|
| 7 | 15 / 380 | 3.9% |
| 12 | 15 / 430 | 3.5% |
| 22 | 8 / 378 | 2.1% |
| 10 | 5 / 251 | 2.0% |
| 11 | 1 / 68 | 1.5% |
| All others | 0 / 1,595 | 0.0% |
| **OVERALL** | **44 / 3,102** | **1.4%** |

Only 4 of 18 present classes achieve any detections at all.

## 3. Box Regression Analysis: Poor Localization

### Best IoU per GT box (best prediction matching each GT by overlap)

| Class | Mean IoU | Median IoU | Max IoU | Any pred IoU>0.5 |
|-------|----------|------------|---------|-----------------|
| 0 | 0.056 | 0.037 | 0.336 | 0 / 331 (0%) |
| 4 | 0.116 | 0.036 | 0.550 | 1 / 324 (0.3%) |
| 7 | 0.167 | 0.204 | 0.599 | 20 / 380 (5.3%) |
| 9 | 0.295 | 0.294 | 0.433 | 0 / 88 (0%) |
| 11 | 0.368 | 0.353 | 0.500 | 1 / 68 (1.5%) |
| 12 | 0.330 | 0.316 | 0.551 | 22 / 430 (5.1%) |
| 13 | 0.360 | 0.368 | 0.439 | 0 / 57 (0%) |
| 17 | 0.261 | 0.274 | 0.471 | 0 / 263 (0%) |
| 21 | 0.367 | 0.373 | 0.539 | 5 / 175 (2.9%) |
| 22 | 0.383 | 0.422 | 0.605 | 31 / 378 (8.2%) |
| **ALL** | **0.234** | **0.248** | **0.605** | **86 / 3,102 (2.8%)** |

The mean best IoU of 0.234 is far below the 0.5 threshold. Even when the model predicts in roughly the right area, the boxes don't overlap well with GT. 501 GT boxes (16%) have zero overlapping prediction at any IoU.

Class 22 has the best regression (median IoU 0.422, approaching the 0.5 threshold), which is why it achieves 8/378 detections despite the classification confusion.

## 4. Evaluation Pipeline Analysis

- DET_EVAL_SCORE_THRESH = 0.001 (passes almost all 1.3M anchors through score filter)
- DET_EVAL_MAX_PER_IMAGE = 300 (top 300 anchors per image pass through NMS)
- DET_EVAL_NMS_IOU_THRESH = 0.5 (standard NMS)

The 0.001 threshold means essentially all 173K anchors pass the score filter on every frame, then top 300 go through NMS. On empty frames, these 300 are all false positives. On GT frames, the 300 include a mix of true and false positives.

After NMS, approximately 105 predictions per frame survive (average across all frames). This is consistent with the per-class NMS filtering removing some overlapping proposals.

## 5. Training Analysis: Why the Detection Head Fails

### 5.1 Multi-task auxiliary problem

The D3 model trains with Kendall weighting across detection, pose, activity, and PSR tasks. The log_vars are initialized with s_det=0, s_pose=-1, s_act=0, s_psr=0. Detection has equal weight to other tasks, but:

- 92% of training batches contain zero-GT frames
- On empty frames, only DET_EMPTY_SAMPLE=2048 of 173K anchors are sampled at scale 0.05
- This produces a very small gradient (~0.005-0.9 loss per empty frame)
- The detection head receives meaningful positive gradients only on ~8% of batches

The DETACH_REG_FPN=True setting isolates regression gradients from FPN but doesn't prevent the cls subnet from collapsing.

### 5.2 Severe class imbalance in loss

The focal loss has:
- FOCAL_ALPHA = 0.50 (raised from 0.25)
- DET_ASYMMETRIC_GAMMA = True
- DET_GAMMA_POS = 0.0 (no focus suppression for positives)
- DET_GAMMA_NEG = 1.5 (mild suppression for negatives)
- DET_OHEM_ENABLED = True, RATIO=2.0, MIN_NEG=32

These settings attempt to fix the imbalance, but the sheer volume of empty-frame background overwhelms the per-class alphas and OHEM. The OHEM keeps only 32 negatives per GT frame with a 2:1 neg:pos ratio, but on empty frames the loss comes from the DET_EMPTY_SAMPLE mechanism which bypasses OHEM.

### 5.3 Per-class alpha overrides

```python
DET_CLASS_ALPHAS = {
    20: 0.96, 18: 0.95, 13: 0.94, 19: 0.93, 9: 0.92, 16: 0.91, 5: 0.90,
    4: 0.80, 15: 0.80, 11: 0.80,
    22: 0.40, 14: 0.30,
    21: 0.05, 8: 0.05, 7: 0.05, 0: 0.05, 10: 0.05, 17: 0.05,
    23: 0.25,  # error_state, zero train GT -- FOCAL_ALPHA default
}
```

Classes with low alpha (0.05: classes 21, 8, 7, 0, 10, 17) have VERY weak positive gradient but strong negative gradient. This suppresses the model from predicting these classes. However, classes 7 and 10 end up as the most-predicted classes anyway, suggesting the alpha is not the primary mechanism controlling class output volume.

The classes with AP = 0.000 (classe 13, 16, 19, 18) all have high alpha (0.91-0.95) but zero predictions. The high alpha gives them a strong positive signal, but the model never learns their visual features.

## 6. Score Distribution

Despite the poor mAP, the model produces reasonable confidence scores:

| Percentile | Score |
|------------|-------|
| Mean | 0.2285 |
| Median | 0.1720 |
| p95 | 0.6246 |
| p99 | 0.8483 |
| p99.9 | 0.9656 |
| Max | 0.9974 |

The scores are well above random (0.033 bias-init floor), which means the classification head IS learning -- it's just learning the WRONG distribution. High confidence on wrong classes is the signature of a classification head that settled into a degenerate local minimum.

---

## Recommendations

### 1. Fix the evaluation empty-frame problem (highest impact)
The primary reason mAP drops from 0.573 (biased subsample) to 0.00009 (full 38k) is that 92% of evaluation frames have no GT. Every prediction on these frames is a false positive.

- **Option A**: Report "GT-frame-only mAP" as the primary metric, with the full-set mAP as a secondary filter. The biased subsample was correct in intent -- it asked "can the model detect boxes when they exist?" The full-set mAP asks "can the model detect boxes AND not predict anything on 92% of frames?" These are different questions.
- **Option B**: Add a "background confidence" score and reject predictions when the overall confidence is below an adaptive threshold. The model could learn to predict "no objects" per frame.
- **Option C**: Use a detection-specific confidence calibration method (temperature scaling on sigmoid outputs) to push background-frame scores below an optimal threshold.

### 2. Fix the classification confusion
The model systematically confuses assembly states (particularly classes 10/12/21/22).

- **Immediate**: Group the 24 detection classes by visual similarity based on the confusion matrix, and train with hierarchical classification (predict group first, then specific class within group).
- **Immediate**: Increase DET_GAMMA_NEG from 1.5 to 2.0 (standard RetinaNet value) to force harder negative mining. This suppresses the 63% class-7/class-10 flood.
- **Medium**: Add a separate classification head (2-layer MLP on pooled features from the box locations), decoupled from the regression head.
- **Medium**: Use label-aware data augmentation -- when a frame has class-22 GT, synthesize additional class-22 examples by pasting.

### 3. Fix the box regression
The mean best IoU of 0.234 is below threshold.

- **Immediate**: Verify that anchor sizes match GT box distribution. The current anchor sizes (96, 160, 256, 384, 512) may not cover the typical assembly part sizes. Re-run k-means on the full 38k GT box dimensions.
- **Immediate**: Increase the number of regression iterations or add a cascade refinement stage (two-stage: first predict rough box, then refine with ROI-pooled features).
- **Medium**: Investigate the GIoU loss gradient -- with best IoU of 0.23, many prediction-GT pairs have zero GIoU gradient because GIoU loss is near-constant when boxes don't overlap.

### 4. Fix the training imbalance
- **Immediate**: Create a GT-balanced batch sampler for the detection head -- ensure every batch contains at least one frame with GT boxes. Without this, the detection head trains on empty gradients for most steps.
- **Immediate**: Schedule the detection loss weight with a ramp (similar to the activity task's act_ramp). Start with low weight and increase once the backbone is stabilized.
- **Medium**: Run a dedicated detection-only fine-tuning phase after multi-task training, with a higher learning rate (3x) and no gradient isolation (DETACH_REG_FPN=False).

### 5. Class-specific fixes

Classes with zero AP and zero predictions (13, 16, 19) need urgent attention:
- Verify these classes have adequate training data
- Verify the class indices are correctly aligned between the dataset labels and the model
- Consider removing classes with very few training examples (<50 GT boxes)

The confusion pair (GT=22 -> Pred=12, 223 instances) is the single biggest classification error. Class 22 and class 12 have similar binary state strings (11101111110 vs 11110110001). Consider:
- Merging these classes if they are visually indistinguishable
- Adding distinguishing features in the head

---

## Data Sources

- Evaluation predictions: `d3_full_eval/per_frame_predictions.json` (38,036 frames)
- mAP computation: `d3_full_38k/analyze_gt.py`, `compute_det_map.py`
- Detection head: `src/models/model.py` (DetectionHead class, lines 500-567)
- Loss function: `src/training/losses.py` (FocalLoss class, lines 74-413)
- Config: `src/config.py` (detection settings lines 215-545, 720-800, 892-899)
- Evaluation: `src/evaluation/evaluate.py` (lines 180-296, 3740-3810)
- Runtime config: `src/runs/rf_stages/checkpoints/config.py`
