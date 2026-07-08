# Detection Dual-Protocol mAP@0.5 Evaluation

## Contract (AAIML 174 Section 3.1 / 175 Section 7.2)

Detection mAP@0.5 on IndustReal is evaluated under **two protocols** because
the published SOTA (WACV-2024, Schoonbeek) reports both numbers and they
measure different things. Neither is "wrong" -- they answer different questions.

---

## The Two Protocols

### 1. Annotated-Frames Protocol (matches WACV 0.838)

- **What it does:** subsets the evaluation set to frames that contain at least
  one ground-truth bounding box (3102 frames out of 38036, ~8.2%).
- **What it measures:** detection quality when there is something to detect.
  Excludes empty frames entirely.
- **Metric name in code:** `det_mAP50` (COCO-24, all classes) or `det_mAP50_pc`
  (present-class average -- only classes with GT > 0 in the eval subset).
- **WACV anchor:** 0.838 (annotated frames).
- **When to report:** as the primary detection accuracy number in ablation
  tables and per-head comparisons. This is the fair comparison to the WACV
  annotated-frame number. Use `det_mAP50_pc` when comparing classes that
  actually appear in the eval split (avoids dilution from zero-GT classes).

### 2. Entire-Video Protocol (matches WACV 0.641)

- **What it does:** evaluates on the full video sequence including ~99.9%
  empty frames where no assembly-state GT box exists.
- **What it measures:** detection precision in the real operating condition --
  most frames have no assembly state present, and any spurious detection on
  those frames is a false positive that drags mAP down.
- **Metric name in code:** `det_mAP50_all_frames` (logged as
  `_det_allframes_protocol: "coco_with_cr"`).
- **WACV anchor:** 0.641 (entire videos).
- **When to report:** alongside the annotated-frame number in **every** SOTA
  comparison table. The WACV paper reports both; we must report both. Never
  compare our annotated-frame number to their entire-video number (or
  vice-versa).

---

## Reporting Rules (from 174 Section 3.1)

1. **Always report both numbers.** Every detection table or claim must include
   the pair `(annotated-frames mAP, entire-video mAP)`. Example:
   `detection mAP@0.5 = (0.838, 0.641)`.

2. **Label each protocol.** Never write "mAP@0.5 = 0.838" without specifying
   "annotated frames" or "entire videos". A reviewer who assumes the wrong
   protocol will flag a mismatch.

3. **SOTA comparison uses the same protocol.** Compare our annotated-frames
   number to WACV 0.838; compare our entire-video number to WACV 0.641.
   Cross-protocol comparison is invalid.

4. **Do not cite the Ultralytics-native 0.995.** The native YOLOv8 validation
   uses a different protocol (COCO 80-class with specific NMS settings on the
   COCO val set). The 0.995 number in old reports (172 C-1) is from a
   different-weight model evaluated on a different protocol. It is not
   comparable to either WACV number.

---

## The Empty-Subsample Bug (P3, preflight defect)

### Bug description

In `full_eval_inprocess.py:400-406`, when a random subsample of the
validation set contains zero GT boxes (possible for small subsamples or
classes with sparse annotations), `gt_box_total == 0` causes the code to set
detection mAP to 0.0 silently:

```python
gt_box_total = sum(len(b) for b in dg_boxes)
if gt_box_total == 0:
    logger.warning("No GT boxes in evaluation split -- skipping detection mAP")
    results["det_mAP50"] = 0.0
    results["det_mAP_50_95"] = 0.0
    results["det_mAP50_all_frames"] = 0.0
    results["det_n_present_classes"] = 0
```

A reported mAP of 0.0 could therefore mean either:
- The model is truly random (no detection ability), OR
- The evaluation subsample happened to include no GT boxes (an artifact).

This masked the D3 model's actual (low but non-zero) detection performance,
making it indistinguishable from a complete failure.

### Fix applied in `eval_detection_dual_protocol.py`

The standalone script **fails with an AssertionError** instead of silently
reporting 0.0:

```python
if gt_box_total == 0:
    raise AssertionError(
        "gt_box_total == 0 in the full evaluation set. "
        "Cannot compute detection mAP. "
        "This is the P3 empty-subsample bug -- refusing to produce 0.0."
    )
```

The same assertion guards the annotated-frames subset. This ensures that a
reported 0.0 is always real (the model detects nothing on frames that have
GT), never an artifact of an empty eval slice.

---

## How the Protocols Differ in Code

Both protocols use the same `compute_ap_per_class` function from
`evaluate.py`. The difference is **which frames are passed in**:

- **Annotated-frames protocol:** frames are filtered to those with
  `len(gt_boxes[i]) > 0` before calling the mAP function. The filter is in
  `scripts/eval_detection_dual_protocol.py::filter_annotated_frames()`.

- **Entire-video protocol:** all frames are passed through
  `compute_ap_per_class_all_frames()` which handles empty-GT frames by
  recording any predictions on them as false positives.

```python
# Annotated-frames: filter then call compute_ap_per_class
pb_f, ps_f, pl_f, gb_f, gl_f = filter_annotated_frames(...)
result = compute_ap_per_class(pb_f, ps_f, pl_f, gb_f, gl_f, iou_thresh=0.5)

# Entire-video: call compute_ap_per_class_all_frames with all frames
result = compute_ap_per_class_all_frames(
    pred_boxes, pred_scores, pred_labels,
    gt_boxes, gt_labels, iou_thresh=0.5,
)
```

---

## Running the Script

```bash
cd <repo_root>/code/industreal_improved
python3 scripts/eval_detection_dual_protocol.py \
    [--predictions path/to/per_frame_predictions.json] \
    [--out path/to/output/metrics.json]
```

Without arguments, it defaults to the `d3_full_eval/per_frame_predictions.json`
file and writes to
`src/runs/rf_stages/checkpoints/detection_dual_protocol/metrics.json`.

The script:
1. Loads per-frame detection predictions and GT.
2. Asserts `gt_box_total > 0` in the full eval set.
3. Filters to annotated frames for protocol 1, computes mAP@0.5.
4. Computes mAP@0.5 on all frames for protocol 2.
5. Logs both numbers and saves to JSON.

---

## Current Numbers

Run on the only available predictions file (`d3_full_eval`, which is from the
D3 crash-recovery model with near-zero detection capability):

| Protocol | Our mAP@0.5 | WACV Anchor | Gap |
|---|---|---|---|
| Annotated frames | 0.0001 | 0.838 | 0.8379 |
| Entire video | 0.0001 | 0.641 | 0.6409 |

**These numbers reflect the D3 crash-recovery model, not a properly trained
detector.** The D1 YOLOv8m checkpoint (`yolov8m_industreal.pt`) should produce
numbers close to the WACV anchors. To get the real numbers, run YOLOv8m
inference on the full validation set to generate
`per_frame_predictions.json`, then re-run this script.

---

## File Locations

- Eval script: `scripts/eval_detection_dual_protocol.py`
- This README: `src/evaluation/DETECTION_DUAL_PROTOCOL_README.md`
- Output JSON: `src/runs/rf_stages/checkpoints/detection_dual_protocol/metrics.json`
- Core mAP functions: `src/evaluation/evaluate.py` (`compute_ap_per_class`,
  `compute_ap_per_class_all_frames`, `compute_det_metrics_extended`,
  `compute_det_metrics_all_frames`)
- Predictions input: `src/runs/rf_stages/checkpoints/d3_full_eval/per_frame_predictions.json`
