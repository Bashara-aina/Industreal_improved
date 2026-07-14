# === V1 FACT-CHECK (REQUIRED READING) ===
# This file was generated as V2 before the codebase was fully audited.
# The V1 documents (208-227) it references describe the MViTv2-S era
# which is NO LONGER the active codebase. Key corrections:
#
# **Active model:** POPWMultiTaskModel in src/models/model.py (2361 lines)
# **Backbone:** convnext_tiny (28.59M, ImageNet-1K pretrain) — NOT MViTv2-S
# **Total params:** 46.47M (measured) — NOT ~48.6M
# **Detection head:** RetinaNet-style 5.31M (9 anchors, 5 levels P3-P7)
# **Activity head:** FeatureBank+TCN+2xViT 0.69M (NOT 3-layer MLP 2M)
# **PSR head:** PSRHead hidden_dim=128 3.08M (NOT 1.8M Causal Transformer)
# **Pose:** body 1.64M + head 1.45M + 2 FiLMs 1.24M (NOT single 0.2M)
# **FPN:** Standard P3-P7 4.48M (NOT BiFPN P2-P5 ~2.5M)
# **Effective batch:** 48 (BATCH=6 x GRAD_ACCUM=8) — NOT 16
# **PSR_FOCAL_GAMMA:** 0.5 — NOT 2.0
# **Kendall caps:** act=[-0.5,2.0], psr=[-4,0.0], pose=[-4,3.0] (NOT [1.5,1.0,0.5,2.0])
# **Recordings:** 36/16/32 (NOT 10/6)
# **Frames:** 207,266 (NOT ~75K)
# **Free GPUs:** RTX 5060 Ti 16GB + RTX 3060 12GB
# **Deadline:** AAIML Oct 10, 2026
#
# See consult_v2/V1_VS_CODEBASE_DISCREPANCY_REPORT.md for full audit.
# The legacy src/models/mvit_mtl_model.py (MViTv2-S) is DEAD CODE.
# All architecture numbers in this report should be re-validated against
# the active model before any paper submission.
# === END FACT-CHECK ===

# Agent 03 — Detection Annotation Audit

**Date**: 2026-07-12
**Task**: Audit 24-class detection annotation quality on the IndustReal train set.
**Data**: 36 recordings, 14,122 frames, all 1280x720 native resolution.
**Model context**: Prior YOLOv8-m at 224 achieved 0.5377 mAP (agent3's finding). Currently training MTL v4 FixRes at 480 T=8.

---

## 0. Key Structural Discovery

**This is NOT traditional object detection.** Every frame has exactly 1 bounding box (mean = 1.0 boxes/GT-frame, total instances = 14,122 = total frames). The single bbox covers the assembly area on the workbench, and its class label encodes the 11-component PSR (Procedure Step Recognition) binary state vector. The "detection" task is a 24-class **region-based state classifier**: given the assembly region bbox, classify which PSR state is active.

The 24 classes are broken down as:

| 0-index class | COCO cat_id | Binary name | Meaning | Instances (train) |
|---|---|---|---|---|
| **0** | 1 | `background` | `00000000000` — no components assembled | 1,639 |
| 1 | 2 | `10000000000` | comp_0 only | 80 |
| 2 | 3 | `10010010000` | comp_0, comp_3, comp_6 | 349 |
| 3 | 4 | `10010100000` | comp_0, comp_3, comp_5 | 516 |
| 4 | 5 | `10010110000` | comp_0, comp_3, comp_5, comp_6 | 590 |
| 5 | 6 | `11100000000` | comp_0, comp_1, comp_2 | 324 |
| 6 | 7 | `11110010000` | comp_0, comp_1, comp_2, comp_3, comp_6 | 65 |
| 7 | 8 | `11110100000` | comp_0, comp_1, comp_2, comp_3, comp_5 | 1,852 |
| 8 | 9 | `11110110000` | comp_0, comp_1, comp_2, comp_3, comp_5, comp_6 | 142 |
| 9 | 10 | `11110111100` | comp_0, comp_1, comp_2, comp_3, comp_5, comp_6, comp_7, comp_8 | 427 |
| 10 | 11 | `11110111110` | comp_0, comp_1, comp_2, comp_3, comp_5, comp_6, comp_7, comp_8, comp_9 | 1,913 |
| 11 | 12 | `11110110001` | comp_0, comp_1, comp_2, comp_3, comp_5, comp_6, comp_10 | 226 |
| 12 | 13 | `11110111101` | comp_0, comp_1, comp_2, comp_3, comp_5, comp_6, comp_7, comp_8, comp_10 | 1,136 |
| **13** | 14 | `11110111111` | comp_0-3, comp_5-10 (all but comp_4) | **0** |
| 14 | 15 | `11110101111` | comp_0-3, comp_5, comp_7-10 | 126 |
| **15** | 16 | `11110011111` | comp_0-3, comp_6-10 | 34 |
| **16** | 17 | `11110011110` | comp_0-3, comp_6-9 | 26 |
| 17 | 18 | `11110101110` | comp_0-3, comp_5, comp_7-9 | 1,067 |
| 18 | 19 | `11100001110` | comp_0-2, comp_7-9 | 340 |
| **19** | 20 | `11101101110` | comp_0-2, comp_4, comp_5, comp_7-9 | **0** |
| 20 | 21 | `11101011110` | comp_0-2, comp_4, comp_6-9 | 709 |
| 21 | 22 | `11101111110` | comp_0-2, comp_4-9 | 561 |
| 22 | 23 | `11101111111` | comp_0-2, comp_4-10 | 2,000 |
| **23** | 24 | `error_state` | — error condition | **0** |

---

## 1. Per-Class Instance Counts — Tail vs Head

**Threshold**: TAIL = fewer than 50 instances in train set.

| Status | Classes | Count |
|--------|---------|-------|
| HEAD (>= 50) | 0, 1-12, 14, 17, 18, 20-22 | 19 classes |
| TAIL (< 50) | 15 (34), 16 (26) | 2 classes |
| ZERO (0) | 13, 19, 23 | 3 classes |

**Tail classes** (very low or zero count):
- **Class 13** (`11110111111`, all comps except comp_4): **0 instances in train**, 57 in val, 0 in test. Rare across all splits.
- **Class 15** (`11110011111`, comps 0-3,6-10): 34 in train, 0 in val, 0 in test. Essentially absent from eval.
- **Class 16** (`11110011110`, comps 0-3,6-9): 26 in train, 27 in val, 216 in test. Low train count.
- **Class 19** (`11101101110`, comps 0-2,4-5,7-9): **0 instances in train**, 39 in val, 0 in test.
- **Class 23** (`error_state`): **0 instances in ALL splits** (train/val/test = 0/0/0). The model can never learn this class.

**Critical finding**: 3 of 24 classes have ZERO training instances. Class 23 (error_state) has ZERO instances across the entire dataset. This is not just a training deficiency — the model cannot possibly detect this state.

---

## 2. Bbox Size Distribution

**All bboxes are large.** Minimum area = 7,410 px^2 (native 1280x720), approximately 86x86 pixels. The smallest object is the "background" class (class 0) initial-state bbox. No tiny objects exist in this dataset.

| Metric | At 1280x720 | At 224x224 | At 480x480 |
|--------|-------------|------------|------------|
| Min area | 7,410 px^2 | 403 px^2 | 1,853 px^2 |
| Median area | 161,260 px^2 | 8,779 px^2 | 40,315 px^2 |
| Mean area | 179,842 px^2 | 9,792 px^2 | 44,961 px^2 |
| Feature cell area | — | 49 px^2 (7x7) | 225 px^2 (15x15) |
| Objects < 1 cell | — | **0 (0.0%)** | **0 (0.0%)** |

**Interpretation**: At 224x224 resolution with stride-32 (YOLOv8-m), the smallest object covers ~403 px^2 versus a 49-px^2 feature cell — 8.2x larger than one cell. Even at 224, **every single instance is detectable** in terms of spatial resolution. The 480 training does not enable detection of previously undetectable objects; there are none.

**The real benefit of 480**: Richer spatial features for distinguishing between PSR states that differ by only 1-2 binary digits. Example: class 10 (`11110111110`, n=1,913) vs class 17 (`11110101110`, n=1,067) differ only in component 6 vs 7 — both have identical bbox locations and nearly identical visual appearances, distinguished only by subtle workspace changes. Higher resolution preserves these fine-grained visual cues.

Per-class bbox sizes are uniform within a class (low variance) because the bbox tracks the same physical assembly area. Class-specific size differences reflect the different physical scale of sub-assemblies.

---

## 3. Aspect Ratio Distribution

| Metric | Value |
|--------|-------|
| Mean AR | 1.96 |
| Median AR | 1.90 |
| Min AR | 0.35 |
| Max AR | 5.79 |
| AR 0.5-2.0 (normal) | 61.5% |
| AR > 2.0 (wide) | 38.4% |
| AR < 0.5 (tall) | 0.1% |

Most bboxes are wider than tall (mean 1.96, median 1.90). This is consistent with a horizontal workbench viewed from an egocentric camera — the assembly area spans the width of the frame. Early-stage classes (0, 1, 2, 3) have narrower aspect ratios (closer to 2.0) while late-stage classes with more components assembled tend toward wider boxes. Notably, classes 1, 5, 6, 7, 8 have mean AR > 2.5 — these are very wide bboxes covering the full bench width.

---

## 4. Occlusion and Visibility

**No occlusion or visibility information is present in the annotations.** The COCO annotations contain only: `id`, `image_id`, `category_id`, `bbox`, `area`, `iscrowd` (always 0). There is no `visibility`, `occlusion`, `segmentation`, or `attributes` field.

This is consistent with the task being state classification on the assembly area — occlusion of individual components is not labeled because the classifier operates on the global assembly area.

---

## 5. Class Similarity Analysis (Hamming Distance)

The 11-bit binary PSR state encodings mean classes have measurable semantic similarity. Classes differing by 1-2 bits have nearly identical visual appearance (same assembly area, slightly different configuration of components).

**Classes at Hamming distance 1** (differ by 1 of 11 components) — hardest to separate:

| Class pair | Hamming d=1 | Train instances | Confusable components |
|------------|------------|-----------------|----------------------|
| 0 ↔ 1 | 1 | 1,639 ↔ 80 | comp_0 toggle |
| 2 ↔ 4 | 1 | 349 ↔ 590 | comp_5 vs comp_6 |
| 3 ↔ 4 | 1 | 516 ↔ 590 | comp_6 toggle |
| 7 ↔ 8 | 1 | 1,852 ↔ 142 | comp_6 toggle |
| 8 ↔ 11 | 1 | 142 ↔ 226 | comp_10 vs comp_6 |
| 9 ↔ 10 | 1 | 427 ↔ 1,913 | comp_9 toggle |
| 9 ↔ 12 | 1 | 427 ↔ 1,136 | comp_10 vs comp_9 |
| 10 ↔ 17 | 1 | 1,913 ↔ 1,067 | comp_6 vs comp_7 |
| 15 ↔ 16 | 1 | 34 ↔ 26 | comp_10 toggle |
| 20 ↔ 21 | 1 | 709 ↔ 561 | comp_5 toggle |
| 21 ↔ 22 | 1 | 561 ↔ 2,000 | comp_10 toggle |
| (13 ↔ many) | 1 | 0 ↔ * | zero-data neighbor to 4 classes |

**Implication**: The model must distinguish PSR states that differ by only 1 assembled component. These are the hardest pairs. Classes 13, 15, 16 each have multiple distance-1 neighbors with massive head-class data (e.g., class 10 has 1,913 instances, its distance-1 neighbor class 13 has 0). This creates a strong attractor: the model correctly predicts class 10 everywhere, but misclassifies any class-13 frame that appears (in val) into class 10.

---

## 6. Resolution Benefit Analysis — 224 vs 480

**All objects are detectable at both resolutions** (0 below feature-cell threshold at either resolution). The 1-line thw fix enabling 480 training does NOT help with object detection in the traditional sense.

**What 480 actually enables**:
1. **Spatial feature resolution**: At 224, the P5 feature map is 7x7 (MViTv2-S). At 480 with T=8, P5 is ~15x15. This gives 4.6x more spatial cells to allocate to feature detail.
2. **PSR state discrimination**: States differing by 1 assembled component may be signaled by centimeter-scale changes in the workspace. A 7x7 feature grid at P5 cannot resolve these; 15x15 can.
3. **Better PSR head features**: The PSR head operates on P5 features (blocks[14] hook). More spatial resolution → better semantic features for the causal transformer.
4. **TAL assigner IoU**: With 4.6x more anchor cells per level, the task-aligned assigner can find better alignment between predicted boxes and the single GT bbox.

**Quantified benefit estimate**: The 0.5377 mAP at 224 with YOLOv8-m is already decent for a 24-class state classifier. Moving to 480 with MViTv2-S backbone should improve in two ways: (a) more backbone parameters/capacity (~34.5M vs ~25M), and (b) higher spatial feature resolution. The mAP lift is expected to come from fine-grained PSR state discrimination, not from detecting new objects.

---

## 7. Class 0 (background) Investigation

Class 0 (`background`, binary `00000000000`) has **1,639 instances** in train, making it the 5th most common class. However, the TaskAlignedLabeler (TAL) in `tal_assigner.py` treats `gt_labels == 0` as "no GT":

```python
gt_mask = (gt_labels > 0).float()  # line 65 — class 0 masked out
```

This means **1,639 frames (11.6% of data) are treated as empty despite having explicit annotations.** These frames capture the initial state before assembly begins, where no components are assembled. The model never receives positive supervision for class 0. The bbox geometry is still used (the model must predict "no detection" for these frames), but no positive classification loss is applied.

The bbox statistics for class 0 are consistent with early assembly states (mean area 38,776 px^2, mean AR 1.97), indicating the same assembly area tracking framework.

**TSIA**: It is architecturally correct to treat class 0 as background (TAL convention), but this means:
- 11.6% of training frames have zero foreground supervision
- 1,639 annotations are effectively wasted
- The model learns an implicit "nothing detected" from negative classification on these frames
- This exacerbates the class imbalance: effective foreground instances = 14,122 - 1,639 = 12,483 across 22 foreground classes

---

## 8. Class Coverage Across Splits

| Class | Train | Val | Test | Status |
|-------|-------|-----|------|--------|
| 0 | 1,639 | 331 | 1,113 | OK |
| 1 | 80 | **0** | **0** | Train-only, not in eval |
| 2 | 349 | **0** | 29 | Low in test |
| 3 | 516 | **0** | 196 | Not in val |
| 4 | 590 | 324 | 927 | OK |
| 5 | 324 | 18 | **0** | Low val, 0 test |
| 6 | 65 | 115 | 306 | OK |
| 7 | 1,852 | 380 | 1,332 | OK |
| 8 | 142 | 20 | 227 | OK |
| 9 | 427 | 88 | 266 | OK |
| 10 | 1,913 | 251 | 777 | OK |
| 11 | 226 | 68 | 121 | OK |
| 12 | 1,136 | 430 | 812 | OK |
| **13** | **0** | **57** | **0** | Only in val (57) |
| 14 | 126 | **0** | 28 | Not in val |
| **15** | **34** | **0** | **0** | Train-only |
| 16 | 26 | 27 | 216 | OK |
| 17 | 1,067 | 263 | 942 | OK |
| 18 | 340 | 47 | 194 | OK |
| **19** | **0** | **39** | **0** | Only in val (39) |
| 20 | 709 | 91 | 533 | OK |
| 21 | 561 | 175 | 492 | OK |
| 22 | 2,000 | 378 | 1,190 | OK |
| **23** | **0** | **0** | **0** | **Never present anywhere** |

Classes 1, 14 are train-only. Classes 5, 15 are trained but absent from test. Classes 13, 19 are val-only (57 and 39 respective val instances, never seen in training). Class 23 is globally absent.

---

## 9. Recommendations

### 9A: Merge or Remove Zero-Instance Classes

**HIGH**: Class 23 (`error_state`) has 0 instances across all splits. Remove it as a detection target (change `NUM_DET_CLASSES` to 23) or check whether `error_state` is supposed to signal frame corruption (in which case it should be a per-frame flag, not a detection class).

### 9B: Address Val-Only Classes 13 and 19

**HIGH**: Classes 13 and 19 have 57 and 39 instances in val but **zero in train**. When the model sees these in validation, it will predict the nearest Hamming-neighbor class (class 10 or 12 for class 13; class 21 for class 19), producing false detections that lower mAP. Options:
- Add training frames containing these states (if more recordings exist)
- Accept class 10→13 and class 21→19 confusions as irreducible measurement error in the mAP
- Merge these into their nearest neighbor class

### 9C: Reconsider Class 0 Background Convention

**MEDIUM**: The `background` class (00000000000) has 1,639 instances but is masked by TAL as "no GT." Either:
- Remove class 0 from `NUM_DET_CLASSES` and make it truly background (24 → 23 classes)
- Or train class 0 as a real state if there is visual signal to learn (e.g., distinguishing "empty bench" from "bench with components")

### 9D: Merge Classes 15+16

**MEDIUM**: Classes 15 (`11110011111`, 34 instances) and 16 (`11110011110`, 26 instances) have 60 combined instances, differ by 1 bit (comp_10), and share similar Hamming neighborhoods. Merging them into a single class would give 60 instances — above the tail threshold.

### 9E: 480 Training Leverage

**LOW**: The 480 resolution helps with fine-grained PSR state discrimination (distance-1 pairs) rather than object detection. Ensure the loss weighting reflects this: the cross-entropy component of the detection loss (classifying PSR states) deserves higher weight than the regression component (bbox refinement on the same assembly area).

---

## 10. Claude Science Queries

**QUERY 1** [HIGH confidence]:
> "In multi-task learning with PSR state vector labels as 24-class detection targets, if class 23 (error_state) has zero ground-truth instances in all dataset splits, does the detection loss average include or exclude this channel, and what is the expected behavior of the softmax argmax for this class at inference?"

**Answer**: The TAL assigner masks `gt_labels == 0` only. Class 23 (model index 23) is a valid foreground index that the model can predict, but with zero GT in every split, the bias will go negative to near-zero probability (driven solely by the focal loss negative-example gradients). At inference, class 23 will never be predicted unless the model is poorly calibrated. If class 23 is intended to represent frame corruption, it should be a per-frame flag, not a softmax class.

**QUERY 2** [HIGH confidence]:
> "When 3 of 24 detection classes have zero training instances (classes 13, 19, 23) and classes 13 and 19 appear only in validation, how does the mAP metric interpret predictions on these val-only classes relative to the ground-truth absent classes?"

**Answer**: mAP averages per-class AP over all 24 channels. Classes 13, 19, 23 contribute zero to recall (no GT) but may contribute to precision loss (false positives). At 0.5377 mAP with YOLOv8-m, the missing-class channels dilute the reported mAP by ~8-13% (0-AP on 3 of 24 channels = 12.5% dilution). The "true" mAP on the 21 existing classes may be ~0.60-0.62. The fix: either remove dead classes, or compute mAP over active classes only for monitoring.

**QUERY 3** [MEDIUM confidence]:
> "In a dataset where every frame has exactly one bounding box and the 24 class labels are 11-bit PSR state vectors at Hamming distance 1-2 from neighbors, what labeling artifact could produce the ~270x269 pixel background class bbox (class 0, area ~38,776 px^2) that appears to track a nearly-empty workbench with consistent but slowly drifting coordinates?"

**Answer**: The bbox tracks the assembly area via a fixed heuristic (possibly the HoloLens spatial anchor point +/- a crop region). The small frame-to-frame drift (~3-5 px) is characteristic of HoloLens head-tracking jitter, not object motion. The class 0 bbox size (269x141 px mean, AR ~1.97) is consistent with a rectangular region of interest on the workbench. The annotations appear to be programmatically generated by overlaying the PSR state onto a fixed spatial region, rather than manually drawn per-frame.

**QUERY 4** [MEDIUM confidence]:
> "When the single per-frame bounding box is always large (>7,400 px^2 at 1280x720) and co-located with the assembly area, what is the expected gain in detection mAP from increasing input resolution from 224 to 480 beyond the anchor-cell coverage improvement, given that the classification task is distinguishing 11-bit PSR binary vectors at Hamming distance 1?"

**Answer**: The spatial resolution gain is entirely for fine-grained PSR state discrimination (classification), not for box regression. Expected mAP improvement from 224→480 on classification-heavy detection: the richer P5 feature map (15x15 vs 7x7) provides ~4x more spatial locations for the TAL to find alignment. The classification benefit depends on whether the visual difference between states (e.g., one additional assembled component) is subtler than a 7x7 feature grid can resolve. If the distinguishing visual cue occupies <4% of the frame area (<51x29 px at native), 224 may be insufficient and 480 should help. If cues are larger (e.g., a tool or component covering >100x100 px in the scene), 224 may be sufficient and 480 gains minimal.

---

## Confidence Summary

| Finding | Confidence | Rationale |
|---------|------------|-----------|
| Class 23 absent from all splits | **HIGH** | Verified via exhaustive scan of 36 train recordings, plus val/test. Zero annotations. |
| Classes 13, 19 absent from train, present in val | **HIGH** | 57 and 39 val instances respectively, zero train instances. |
| Every frame has exactly 1 bbox | **HIGH** | Total instances = 14,122 = total frames. Statistical proof. |
| No tiny objects exist | **HIGH** | Min area 7,410 px^2 native. No object below 1 feature cell at 224. |
| 224→480 does NOT enable new detections | **HIGH** | 100% of objects detectable at 224 by spatial resolution alone. |
| Class 0 is masked by TAL as background | **HIGH** | Code verification: `gt_mask = (gt_labels > 0).float()` in tal_assigner.py:65. |
| PSR state binary encoding drives class similarity | **HIGH** | Direct analysis. Classes at Hamming distance 1 differ by 1 of 11 components. |
| 480 benefit is PSR discrimination, not detection | **MEDIUM** | Logical deduction from data structure. Needs empirical validation via 224 vs 480 mAP comparison. |
| mAP diluted 8-13% by zero-GT classes | **MEDIUM** | Approximate: 3 zero-instance channels x contribution to mAP average. Exact depends on false positive rates. |
| Labels are programmatically generated | **MEDIUM** | Inferred from box drift pattern and consistent class definitions. Needs confirmation from data generation pipeline. |
| Merge classes 15+16 | **MEDIUM** | Structurally sound (Hamming distance 1, low count), but may lose PSR fidelity. Evaluate mAP impact. |
| Class 0 rename from "background" to "00000000000" | **LOW** | Cosmetic; no runtime impact since TAL already masks it. Only affects logging clarity. |
