# D3 Full-38k Detection mAP

Computed from `per_frame_predictions.json` (v1 full eval on all 38,036 frames).

| Metric | Value |
|--------|-------|
| mAP@0.5 | 0.0001 |
| mAP@0.5:0.95 | 0.0000 |
| Total frames | 38036 |
| Total GT boxes | 3102 |
| Detection classes | 24 |

## Per-Class AP@0.5

| Class ID | AP@0.5 |
|----------|--------|
| 0 | 0.0000 |
| 4 | 0.0000 |
| 5 | 0.0000 |
| 6 | 0.0000 |
| 7 | 0.0001 |
| 8 | 0.0000 |
| 9 | 0.0000 |
| 10 | 0.0001 |
| 11 | 0.0002 |
| 12 | 0.0012 |
| 13 | 0.0000 |
| 16 | 0.0000 |
| 17 | 0.0000 |
| 18 | 0.0000 |
| 19 | 0.0000 |
| 20 | 0.0000 |
| 21 | 0.0000 |
| 22 | 0.0000 |

## Per-Class AP@0.5:0.95

| Class ID | AP@0.5:0.95 |
|----------|------------|
| 0 | 0.0000 |
| 4 | 0.0000 |
| 5 | 0.0000 |
| 6 | 0.0000 |
| 7 | 0.0000 |
| 8 | 0.0000 |
| 9 | 0.0000 |
| 10 | 0.0000 |
| 11 | 0.0000 |
| 12 | 0.0001 |
| 13 | 0.0000 |
| 16 | 0.0000 |
| 17 | 0.0000 |
| 18 | 0.0000 |
| 19 | 0.0000 |
| 20 | 0.0000 |
| 21 | 0.0000 |
| 22 | 0.0000 |

Computed with COCO-style all-point interpolation.
Average of 10 IoU thresholds (0.50:0.05:0.95) for mAP@0.5:0.95.
GPU not required — all computation on CPU from saved predictions.

**Interpretation**: The full-38k mAP is ~0.0001 (0.01%), a dramatic drop from the earlier
250-batch subsample (0.573). The subsample used only frames with GT boxes; the full-38k
set has 38036 frames but only 3102 GT boxes (99.9% of frames have zero GT). The D3
multi-task detection head was trained as an auxiliary loss and produces ~105 predictions
per frame — nearly all false positives on empty frames, collapsing the precision-recall
curve. This is consistent with a model that never received focused detection training.
The 18 present classes all have very few GT instances (8-430 boxes per class).

**Disclosure update**: The previous "present-class mAP50=0.573" was based on a 250-batch
subsample that only evaluated frames WITH GT boxes. The correct full-38k number for
SOTA reporting is 0.00009 (present-class, COCO convention). Zero-GT classes (6: [1,2,3,14,15,23])
are excluded per COCO standard.
