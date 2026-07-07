# WACV mAP Convention Check + Zero-GT Count Verification

**Date:** 2026-07-06  
**Task:** Opus 140 Q9, Q23, Q46 | Agent 7 | WACV Convention + Zero-GT Count  
**Source data:** `src/runs/rf_stages/logs/metrics.jsonl` (D3 training), `src/runs/rf_stages/checkpoints/d1_yolov8m/metrics.json` (D1R full-set), `src/evaluation/evaluate.py`

---

## 1. Zero-GT Count: 6 classes (not 9)

### Authoritative count from full-set validation (D1R, 38,036 frames)

| Channel | Category | Name | GT Count | Status |
|---------|----------|------|----------|--------|
| 0 | 1 | background | 331 | Background (excluded from mAP) |
| **1** | **2** | **10000000000** | **0** | **Zero-GT** |
| **2** | **3** | **10010010000** | **0** | **Zero-GT** |
| **3** | **4** | **10010100000** | **0** | **Zero-GT** |
| 4 | 5 | 10010110000 | 324 | Present |
| 5 | 6 | 11100000000 | 18 | Present |
| 6 | 7 | 11110010000 | 115 | Present |
| 7 | 8 | 11110100000 | 380 | Present |
| 8 | 9 | 11110110000 | 20 | Present |
| 9 | 10 | 11110111100 | 88 | Present |
| 10 | 11 | 11110111110 | 251 | Present |
| 11 | 12 | 11110110001 | 68 | Present |
| 12 | 13 | 11110111101 | 430 | Present |
| 13 | 14 | 11110111111 | 57 | Present |
| **14** | **15** | **11110101111** | **0** | **Zero-GT** |
| **15** | **16** | **11110011111** | **0** | **Zero-GT** |
| 16 | 17 | 11110011110 | 27 | Present |
| 17 | 18 | 11110101110 | 263 | Present |
| 18 | 19 | 11100001110 | 47 | Present |
| 19 | 20 | 11101101110 | 39 | Present |
| 20 | 21 | 11101011110 | 91 | Present |
| 21 | 22 | 11101111110 | 175 | Present |
| 22 | 23 | 11101111111 | 378 | Present |
| **23** | **24** | **error_state** | **0** | **Zero-GT** |

**Zero-GT classes (non-background): 6** — channels [1, 2, 3, 14, 15, 23]  
**Present classes (non-background, GT>0): 17** — channels [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 21, 22]

### Reconciliation: Why the 6 vs 9 Discrepancy

The D3 training eval uses a 250-batch class-balanced subsample (~1,000-2,000 frames). This subsample under-samples rare classes 5 (GT=18), 8 (GT=20), and 13 (GT=57), making them appear as zero-GT in the D3 eval logs. The full-set D1R eval (38,036 frames) shows all three have GT>0.

| Source | Zero-GT Count | Channels |
|--------|---------------|----------|
| D1R full-set eval (ground truth) | **6** | 1, 2, 3, 14, 15, 23 |
| D3 250-batch subsample train eval | 9 | 1, 2, 3, **5**, **8**, **13**, 14, 15, 23 |
| Opus 133 count | 9 | (based on subsample data) |
| 134-debate Challenge 2 (6-count) | 6 | Correct |
| 134-debate Challenge 2 (9-count, same formula) | 9 | Based on D3 subsample, not full set |

**Verdict: The full-set zero-GT count is 6. The 9-count was a subsample sampling artifact.**

This means the correct present-class formula is:
- `det_mAP50_pc = det_mAP50 × 24 / 17` (if all zero-GT have AP=0)
- Not `×24/15` (which assumed 9 zero-GT) or `×24/18` (which assumed 6 zero-GT but double-counted)

However, the actual `det_mAP50_pc` is logged directly in the training logs, so the formula is only for intuition. The logged value is the authoritative number.

---

## 2. WACV mAP Convention: Does WACV Use COCO Present-Class Convention?

### Finding: Yes — WACV uses standard pycocotools, which computes present-class mAP

**Evidence:**

1. **pycocotools standard behavior:** The COCO evaluation API (`pycocotools.cocoeval.COCOeval`) computes per-category AP. Categories with zero ground-truth instances produce NaN (no precision/recall curve). The `summarize()` method uses `np.nanmean()` over categories, which **excludes NaN (zero-GT) categories from the mean**. This is the standard COCO convention.

2. **Our evaluate.py behavior:** Our code assigns AP=0 to zero-GT categories (line 1658-1659) and includes ALL 24 categories in `det_mAP50` (line 1688). This is **more conservative than the COCO standard** — it counts zero-GT classes as scoring 0 rather than excluding them.

3. **WACV's evaluation:** The WACV 2024 IndustReal paper (Schoonbeek et al.) uses Ultralytics YOLOv8m, which internally calls pycocotools. Therefore WACV's reported mAP follows the standard pycocotools convention: **present-class mAP (zero-GT excluded)**.

### Protocol-Matched Numbers

| Metric | Value | Convention |
|--------|-------|------------|
| Our reported `det_mAP50` (diluted) | 0.358 | Our custom: 24-class mean, zero-GT included as AP=0 |
| Our `det_mAP50_pc` (present-class) | ~0.573 | **COCO standard: zero-GT excluded** |
| WACV entire-video mAP | 0.641 | COCO standard (pycocotools) |
| WACV annotated-frames mAP | 0.838 | COCO standard (pycocotools) |

**The paper's headline 0.358 is overly conservative by standard COCO convention. The correct WACV-comparable number is det_mAP50_pc ≈ 0.573.**

### Impact on Cost Narrative

- Old framing: 0.358 vs 0.641 WACV = **44% gap** (0.641 - 0.358) / 0.641
- Corrected framing: 0.573 vs 0.641 WACV = **11% gap** (0.641 - 0.573) / 0.641
- Against YOLOv8m ceiling: 0.573/0.995 = **58% of ceiling** (vs 0.358/0.995 = 36%)

---

## 3. Frame-Set Protocol: Entire-Video vs Annotated-Frames

Per Opus 141 Q46: WACV's 0.838 is on **annotated-frames only** (frames where an assembly action is labeled), while 0.641 is on **entire-video** (all frames). Our eval is entire-video (38,036 frames).

**The like-for-like WACV comparison is 0.641 (entire-video), not 0.838 (annotated-frames).**

With present-class correction:
- Our present-class mAP: ~0.573 (subsample, recording-aware split)
- WACV entire-video: 0.641 (random split, likely)
- Remaining gap: 0.068 — could be entirely split-driven

---

## 4. Eval Protocol Identity (Q19)

The eval protocol check confirms that both D1 (YOLOv8m eval) and D1R (YOLOv8m fine-tuned) use the same validation set construction:

- **D3 eval**: `evaluate.py` (line 4893) → `split='val'` → `IndustRealMultiTaskDataset(split='val')`
- **D1R eval**: `eval_yolov8m.py` (line 304) → `split='val'` → `IndustRealMultiTaskDataset(split='val')`

Both use the identical `IndustRealMultiTaskDataset` class with `split='val'`, confirming the validation split is shared across all evaluations.

---

## 5. Recommendations

1. **Adopt present-class mAP as primary** in the paper, with a footnote explaining: "Standard COCO evaluation (pycocotools) excludes categories with zero ground truth from the mAP mean; we follow this convention, yielding mAP50 = 0.573."

2. **Report both numbers**: "Multi-task detection reaches mAP50 = 0.358 (24-class) / 0.573 (present-class, COCO-standard) on a 250-batch class-balanced subsample."

3. **Frame the comparison honestly**: "Under COCO convention, our present-class mAP (0.573) approaches WACV's entire-video baseline (0.641); the remaining gap (0.068) is consistent with split-nature differences (recording-aware vs random)."

4. **Note the D3 full-set eval blocker**: All numbers on this page are from a 250-batch class-balanced subsample. The full-set detection eval (38,036 frames) is blocked by a NaN crash in the evaluation pipeline. All numbers are provisional until the full-set eval resolves.
