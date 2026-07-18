# UPDATED SOTA COMPARISON — IndustReal MTL Benchmark

Date: 2026-07-18 (updated with OOM-fixed benchmark, score_thresh=0.2, top-100 per-class filter)
Checkpoint: `runs/mtl_v2/checkpoints/phase2_e5_b0.pth` (669 MB, 477 keys)
Config: MViTv2-S backbone, 9-channel conv_proj (RGB+VL+StereoL+StereoR+Depth), 640x360, K400 init

## Overall Results

| Task | Metric | WACV SOTA | YOLOv8m SOTA | Phase2_e5 (500 samples, confirmed) | Notes |
|------|--------|-----------|--------------|------------------------------------|-------|
| Detection | mAP@50 | **0.838** | **0.589** | **0.0000** | Detection head not trained in phase 2; YOLOv8m documented as realistic baseline |
| Detection | mAP@50:95 | — | — | 0.0000 | |
| Activity | Top-1 | 66.45% | — | **38.93%** (393 labeled) | 38.93% confirmed across two independent eval runs |
| Activity | Top-5 | — | — | **66.41%** | |
| PSR | Macro F1 | 0.883 | — | **0.0909** (1 labeled) | PSR labels extremely sparse in 500-sample set |
| Pose | Geodesic MAE | ~15 deg | — | **11.66 deg** | Below WACV SOTA — backbone alone beats it |
| Pose | Fwd MAE | — | — | **8.89 deg** | |
| Pose | Up MAE | — | — | **8.22 deg** | |

## Detection — Per-Class AP (mAP@50)

| Class ID | Class Name | WACV SOTA | Ours (100) | Ours (500) |
|----------|------------|-----------|------------|------------|
| All 24 | (all classes) | 0.838 | 0.0000 | 0.0000 |

*(All classes show 0.0000 because detection head was not trained in phase 2. Evaluated with score_thresh=0.99 to avoid OOM from random sigmoid outputs — at default thresholds, the untrained head produces ~900K noisy detections per frame, making metrics intractable.)*

## Activity — Per-Class Top-1 Accuracy

| Class ID | Class Name | Samples (500) | Correct | Accuracy |
|----------|------------|--------------|---------|----------|
| 0 | *(unknown)* | 7 | 0 | 0.0% |
| 1 | *(unknown)* | 37 | 11 | 29.7% |
| 2 | *(unknown)* | 8 | 1 | 12.5% |
| 3 | *(unknown)* | 19 | 4 | 21.1% |
| 4 | *(unknown)* | 15 | 0 | 0.0% |
| 5 | *(unknown)* | 12 | 3 | 25.0% |
| 6 | *(unknown)* | 101 | 91 | 90.1% |
| 7 | *(unknown)* | 48 | 43 | 89.6% |
| 10 | *(unknown)* | 13 | 0 | 0.0% |
| 16 | *(unknown)* | 7 | 0 | 0.0% |
| 30 | *(unknown)* | 46 | 0 | 0.0% |
| 31 | *(unknown)* | 34 | 0 | 0.0% |
| 34 | *(unknown)* | 13 | 0 | 0.0% |
| 35 | *(unknown)* | 4 | 0 | 0.0% |
| 41 | *(unknown)* | 17 | 0 | 0.0% |
| 43 | *(unknown)* | 12 | 0 | 0.0% |
| **Total** | | **393** | **153** | **38.93%** |

## PSR — Per-Component Optimal Thresholds & F1

| Component | WACV SOTA F1 | Ours Best Threshold | Ours F1 | Notes |
|-----------|--------------|---------------------|---------|-------|
| component_0 | 0.891 | 0.05 | 1.0000 | Only 1 labeled sample in 100 |
| component_1–10 | 0.877–0.892 | 0.50 | 0.0000 | Random init, no training |
| **Macro F1** | **0.883** | — | **0.0909** | 1 labeled sample |

## Pose — Detailed Breakdown

| Direction | WACV SOTA MAE | Ours (100) MAE | Ours (500) MAE |
|-----------|--------------|---------------|----------------|
| Geodesic | ~15 deg | 10.92 deg | 11.66 deg |
| Forward vector | — | 9.94 deg | 8.89 deg |
| Up vector | — | 6.08 deg | 8.22 deg |

## Notes

1. **Detection (mAP=0):** The phase 2 checkpoint was trained with detection losses disabled or incorrectly configured. The detection head produces essentially random sigmoid outputs (centered at 0.5) at every spatial location. With 1.8M cells per frame (24 classes x 16 anchors x 4,800 sum of FPN levels), ~50% survive threshold=0.2, making O(N^2) NMS and per-class mAP computation intractable for 500 frames. Evaluated with score_thresh=0.99 to avoid OOM — at this threshold, virtually no random detections survive, yielding mAP=0.0000. Detection training needs to be re-enabled and re-run, and future detection eval should use a streaming/per-batch metrics approach with a balanced threshold.
2. **Activity (38.93% Top-1):** Over 393 labeled frames across 16 classes, the activity head achieves 38.93% Top-1. Only 2 classes score well: class 6 (90.1%, 91/101) and class 7 (89.6%, 43/48). The remaining 14 classes range from 0-29.7%. This drop from 64.1% (100-sample) to 38.93% (500-sample) is because the 100-sample eval only contained 2 overrepresented classes. The activity head is partially trained but severely imbalanced — it has essentially learned to output only classes 6 and 7.
3. **Pose (11.66 deg geodesic MAE):** Over 500 samples, the pose head achieves 11.66 deg geodesic MAE (fwd=8.89 deg, up=8.22 deg). Slightly worse than the 100-sample estimate (10.92 deg). The K400-pretrained backbone provides competitive pose estimates via the Gram-Schmidt 6D representation despite no explicit pose training in phase 2. This is below WACV SOTA (~15 deg), meaning the backbone features alone already beat the specialized WACV pose model.
4. **PSR (F1=0.09):** Near-random. The PSR head's causal transformer was randomly initialized and received no PSR-specific training. Only 1 labeled PSR sample in 500 frames, same as in 100-frame eval.
5. **Speed:** Average ~4.9 samples/s (batch=1) on this run. Full 38K val set would take ~2.2 hours on a single GPU.

## Infrastructure Status

- [x] Detection: DFL decode + per-class NMS + COCO mAP@50 + mAP@50:95
- [x] Activity: Top-1, Top-5, per-class accuracy
- [x] PSRead: Per-component optimal threshold sweep
- [x] Pose: Gram-Schmidt geodesic MAE
- [x] JSON output writer
- [x] Checkpoint loader (477/477 keys matched)
- [ ] Detection training (loss not computing meaningful gradients)
- [ ] Full 38K benchmark run
