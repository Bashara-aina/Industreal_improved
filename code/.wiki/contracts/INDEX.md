# Contract Index — POPW & IndustReal Improvements

## Summary
14 contracts created for improving PopW and IndustReal models to beat benchmark targets.

## Contract List

| # | Priority | Title | Files Modified | Dependencies |
|---|----------|-------|----------------|--------------|
| 1 | HIGH | Create PopW improved folder structure | popw_main_improved/* | none |
| 2 | HIGH | ConvNeXt-Tiny backbone for PopW | model.py | 1 |
| 3 | HIGH | OKS Loss for PopW pose | losses.py | 1 |
| 4 | HIGH | GCN skeleton for PopW | model.py | 1 |
| 5 | HIGH | Create IndustReal improved folder | industreal_improved/* | none |
| 6 | HIGH | ConvNeXt-Tiny backbone for IndustReal | model.py | 5 |
| 7 | HIGH | TMA Cell for IndustReal | model.py | 5 |
| 8 | HIGH | Temporal Bank T=8+T=32 for IndustReal | model.py | 5 |
| 9 | MEDIUM | Label Smoothing for PopW | train.py | 1 |
| 10 | MEDIUM | Temporal Augmentation for PopW | ikea_dataset.py | 1 |
| 11 | MEDIUM | Spatial Augmentation for both | *_dataset.py | 1, 5 |
| 12 | MEDIUM | TTA horizontal flip for PopW | evaluate.py | 1 |
| 13 | LOW | Cosine Annealing with Warmup | train.py | 1, 5 |
| 14 | LOW | ONNX Export | export_onnx.py | 2, 6 |

## Execution Order

### Batch 1 (Serial)
1. Contract #1: PopW folder structure
2. Contract #5: IndustReal folder structure

### Batch 2 (Parallel after #1)
- Contract #2: ConvNeXt-Tiny PopW
- Contract #3: OKS Loss PopW
- Contract #4: GCN Skeleton PopW

### Batch 3 (Parallel after #5)
- Contract #6: ConvNeXt-Tiny IndustReal
- Contract #7: TMA Cell IndustReal
- Contract #8: Temporal Bank IndustReal

### Batch 4 (Parallel)
- Contract #9: Label Smoothing
- Contract #10: Temporal Augmentation
- Contract #11: Spatial Augmentation
- Contract #12: TTA
- Contract #13: Cosine Annealing

### Batch 5 (Final)
- Contract #14: ONNX Export

## Target Metrics

### PopW (IKEA ASM)
| Task | Target | Baseline |
|------|--------|----------|
| Activity Top-1 | >64.15% | I3D RGB+pose |
| Activity mcAP (csv) | >84.47% | PTMA |
| Pose PCK@10px | >64.3% | MaskRCNN-ft |
| Detection AP@0.5 | >85.3% | ResNeXt-101-FPN |

### IndustReal
| Task | Target | Baseline |
|------|--------|----------|
| Activity Top-1 | >66.45% | MViTv2 Kinetics |
| ASD mAP@0.5 | >83.8% | YOLOv8m |
| PSR F1 | >0.901 | STORM-PSR |

### Efficiency (both)
- Params: < 49M
- FPS: > 291 (PTMA baseline)
