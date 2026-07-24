# Forensics Report: Detection Regression in Checkpoint B vs A

## Aggregate Summary

| Metric | A (phase2_e1_b0) | B (phase2_e0_b26000) | Delta |
|---|---|---|---|
| Det mAP@0.5 | **0.554** | **0.366** | **-0.189 (-34%)** |
| Act Top-1 | 19.4% | 20.3% | +0.9 pp |
| Pose MAE | 6.33 deg | 6.61 deg | +0.28 deg |
| PSR F1 | 0.568 | 0.569 | +0.001 |

The regression is **isolated to detection**. Activity, pose, and PSR are flat or slightly better.

---

## 1. Per-Class AP Table (24 detection classes)

Only 4 classes have ground-truth annotations in the 2000-frame sample. The other 20 classes have zero GT boxes in these frames, so AP is undefined/0.0 for both.

| Model idx | Cat ID | Class Name | AP_A | AP_B | Delta |
|---|---|---|---|---|---|
| 0 | 1 | background | 0.0442 | 0.0087 | -0.0355 |
| 1 | 2 | 10000000000 | 0.0000 | 0.0000 | 0.0000 |
| 2 | 3 | 10010010000 | 0.0000 | 0.0000 | 0.0000 |
| 3 | 4 | 10010100000 | 0.0000 | 0.0000 | 0.0000 |
| **4** | **5** | **10010110000** | **0.5632** | **0.3465** | **-0.2167** |
| 5 | 6 | 11100000000 | 0.0000 | 0.0000 | 0.0000 |
| 6 | 7 | 11110010000 | 0.0000 | 0.0000 | 0.0000 |
| **7** | **8** | **11110100000** | **0.9617** | **0.9313** | **-0.0304** |
| 8 | 9 | 11110110000 | 0.0000 | 0.0000 | 0.0000 |
| 9 | 10 | 11110111100 | 0.0000 | 0.0000 | 0.0000 |
| **10** | **11** | **11110111110** | **0.6485** | **0.1759** | **-0.4726** |
| 11-23 | 12-24 | (11 states + error) | 0.0000 | 0.0000 | 0.0000 |

---

## 2. Top 5 Most Regressed Classes (largest AP drop)

| Rank | Model idx | Class Name | AP_A | AP_B | Delta |
|---|---|---|---|---|---|
| 1 | 10 | 11110111110 | 0.6485 | 0.1759 | **-0.4726** |
| 2 | 4 | 10010110000 | 0.5632 | 0.3465 | -0.2167 |
| 3 | 0 | background | 0.0442 | 0.0087 | -0.0355 |
| 4 | 7 | 11110100000 | 0.9617 | 0.9313 | -0.0304 |

## 3. Top 5 Most Improved Classes

None improved. All 4 measurable classes regressed.

---

## 4. Per-Component PSR F1 Table (11 components)

| Comp | Name | F1_A | F1_B | Delta |
|---|---|---|---|---|
| 0 | comp0 | 1.0000 | 1.0000 | 0.0000 |
| 1 | comp1 | 0.9380 | 0.9303 | -0.0077 |
| 2 | comp2 | 0.9330 | 0.9296 | -0.0034 |
| 3 | comp3 | 0.8809 | 0.9004 | **+0.0195** |
| 4 | comp4 | 0.0000 | 0.0000 | 0.0000 |
| 5 | comp5 | 0.9027 | 0.9032 | +0.0005 |
| 6 | comp6 | 0.9009 | 0.9092 | **+0.0083** |
| 7 | comp7 | 0.0000 | 0.0000 | 0.0000 |
| 8 | comp8 | 0.0000 | 0.0000 | 0.0000 |
| 9 | comp9 | 0.0000 | 0.0000 | 0.0000 |
| 10 | comp10 | 0.6923 | 0.6837 | -0.0086 |

PSR is **not regressed**. Components 3 and 6 actually improved slightly. Components 4, 7, 8, 9 have zero F1 on both (no positive GT in the sample).

---

## 5. Detection Confidence Distribution

| Threshold | A (count) | A (%) | B (count) | B (%) |
|---|---|---|---|---|
| Total raw preds | 1,057,155 | 100% | 1,591,510 | 100% |
| > 0.05 | 1,057,155 | 100.0% | 1,591,510 | 100.0% |
| > 0.20 | 154,304 | 14.6% | 241,163 | 15.2% |
| > 0.50 | 11,077 | 1.05% | 40,658 | 2.55% |
| Mean conf | 0.121 | -- | 0.130 | -- |
| Median conf | 0.082 | -- | 0.086 | -- |

| Property | A | B | Interpretation |
|---|---|---|---|
| Predictions per frame | 20.8 | 30.4 | B predicts **46% more boxes** |
| High-conf preds (>0.5) | 11,077 | 40,658 | B has **3.7x more** high-conf predictions |
| mAP at same high conf | 0.554 | 0.366 | But B's extra high-conf boxes are **wrong** |

---

## 6. Diagnosis

### The regression is in the **detection classification (cls) head**, not the regression (reg) head.

**Evidence:**

1. **B predicts 50% more boxes per frame** (30.4 vs 20.8) at the same 0.05 threshold. The cls head is firing on more anchor points.

2. **B has 3.7x more predictions above 0.5 confidence** (40,658 vs 11,077), yet mAP dropped 34%. This is the hallmark of **cls head collapse** -- the model produces confident-but-wrong class predictions. If the reg head were the problem, we would see properly classified boxes with poor IoU, not a flood of extra boxes.

3. **Per-class AP tells the same story:**
   - Class 10 (11110111110): AP drops **73%** (0.649 -> 0.176). This is the most-affected class.
   - Class 4 (10010110000): AP drops **38%** (0.563 -> 0.347).
   - Class 7 (11110100000): only drops 3%, preserved well.
   - The regression is **class-specific**, not uniform, which points to cls head overfitting or weight imbalance.

4. **Other heads are unaffected.** Activity (+0.9pp), Pose (+0.28 deg), and PSR (flat) all confirm the backbone feature quality is preserved. If the backbone had degraded, all tasks would suffer. The regression is **head-localized**.

5. **Confidence calibration is intact** (mean/median conf nearly identical). B isn't under-confident -- it's just wrong more often.

### Root cause hypothesis

The checkpoint B is from **step 26,000 of phase 2** (early), while A is from **epoch 1 end** (later). The phase 2 training likely uses different loss weighting or data distribution that initially harms the detection cls head. The cls head may be receiving **insufficient gradient signal** relative to the other heads (activity/pose/psr dominate via uncertainty weighting), causing it to diverge early before converging later. The later epoch (A) recovered.

### To pinpoint further

- Compare the UW-SO loss weights at step 26k vs epoch 1 end
- Run detection eval at intermediate steps (10k, 18k, 22k) to see when the regression starts
- Evaluate per-class recall to distinguish "cls head not detecting" from "cls head predicting wrong class"
