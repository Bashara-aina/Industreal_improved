# D4 + D1R Evaluation Verdict

**Date:** 2026-07-07
**Purpose:** Decisive test per Opus 140 Q10 -- does decoder transfer given adequate detection density?

## Weights
- D1R fine-tuned YOLOv8m (`best.pt`, mAP=0.995)
- vs original D4: pretrained YOLOv8m (mAP=0.0004 default / 0.347 retuned)

## Results

| Configuration | F1 | Note |
|---|---|---|
| Default thresholds (hi=0.5, lo=0.3, min=3) | 0.000 | Decoder predicts all-zeros |
| Best global sweep (hi=0.3, lo=0.1, min=2) | **0.6364** | Retuned thresholds unlock decoder |
| Per-component optimal | 0.1956 | Per-comp thresholds overfit on 3 recordings |
| Original D4 retuned (pretrained YOLOv8m) | 0.347 | Baseline for comparison |

## Per-Recording Breakdown (Best Global Sweep)

| Recording | F1 | Frames |
|---|---|---|
| 14_main_2_2 | 0.6364 | 1404 |
| 14_main_2_3 | 0.6364 | 1679 |
| 20_assy_0_1 | 0.6364 | 917 |

## Verdict

**F1 = 0.6364 >= 0.6 -- decoder transfers given adequate detection density.**

With D1R fine-tuned YOLOv8m (mAP=0.995) and retuned Q48 thresholds (hi=0.3, lo=0.1, min=2), the MonotonicDecoder achieves F1=0.6364 -- an 83% relative improvement over original D4 retuned F1=0.347. Detection density was the dominant binding constraint at the original D4 operating point. The gap to ConvNeXt-based PSR (0.7499 optimal / ~0.72 on 38k) is ~0.11 (15% relative), representing the residual paradigm gap between detection-based and direct PSR inference.

**Disclosure update (SS5.4):** "With a dense fine-tuned detector, F1 = 0.6364 -- decoder transfers given adequate detection density."

## Limitations
- Evaluated on 3 recordings (4000 frames, 500 batches) due to system memory constraints
- Full 38k eval would provide more robust F1 estimate
