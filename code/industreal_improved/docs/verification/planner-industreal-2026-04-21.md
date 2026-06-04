## Plan: IndustReal POPW Adaptation
Date: 2026-04-21
Type: FEATURE (adapt IKEA POPW codebase for IndustReal dataset)

Context gathered:
- IKEA popw_main: 6 files (config.py, ikea_dataset.py, model.py, losses.py, train.py, evaluate.py)
- benchmark.py is architecture-agnostic, can be copied as-is
- IndustReal: single egocentric RGB (1280×720), 84 recordings across train/val/test
- AR_labels.csv: recording_id, action_class_id (0-74 non-consecutive), action_desc, start_frame, end_frame
- OD_labels.json: COCO format, 24 categories (binary assembly states + error_state + background)
- PSR_labels_raw.csv: per-recording sparse state changes (11 components), not per-frame
- pose.csv: 9 DoF head pose (forward vec 3, position 3, up vec 3)
- hands.csv: 52 coordinates per frame (26 joints × 2 hands)
- PSR_labels.csv: completed_frame, procedure_step_id (0-35), description

Risk assessment:
- PSR task definition is ambiguous — resolved by treating as 11-component multi-label classification
- Non-consecutive action class IDs (0-74 gaps) — use dict mapping for compact tensors
- Single camera (not 3) — much simpler dataset and model
- COCO detection format is identical — can reuse extraction code from ikea_dataset.py

Approach:
- 7 contracts: config, dataset, model, losses, train, evaluate, benchmark
- PSR head: lightweight per-frame 11-dim binary classifier on C5 features
- Pose head: HeadPoseHead regressing 9 DoF from C5 GAP + FC
- Single-camera data loading (much simpler than IKEA multi-camera)

## Execution Order
Serial (must run in sequence):
  1. Contract #1: config.py (foundation — all others import it)
  2. Contract #2: industreal_dataset.py (depends on config)
  3. Contract #3: model.py (depends on config)
  4. Contract #4: losses.py (depends on config + model)
  5. Contract #5: train.py (depends on all above)
  6. Contract #6: evaluate.py (depends on all above)
  7. Contract #7: benchmark.py (depends on config + model)

Parallel: none (strictly sequential dependency chain)

Final gate: Contract #5 train.py dry-run must succeed (model builds, loss computes, dataset loads)

## Risks
| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| PSR_labels_raw interpolation algorithm wrong (sparse → dense per-frame) | H | H | Test interpolation: compare output shape against expected frame count per recording |
| Non-consecutive AR action class IDs cause tensor indexing errors | M | H | Clamp all action IDs to [0, NUM_CLASSES_ACT-1]; use dict mapping not direct indexing |
| Head pose normalization not matching training distribution | M | M | Compute mean/std from pose.csv during dataset init; store in config or compute on first load |
| PSR evaluation metrics (F1@T, Edit Score) implementation bugs | M | M | Test compute_psr_metrics with synthetic data (known state changes) before real evaluation |
| CUDA OOM due to 1280×720 images (vs IKEA 640×480) | H | M | Reduce BATCH_SIZE to 4 if needed; eval uses batch=2 |
| 24 COCO detection classes may have very few annotations per class | M | M | Check class distribution; if imbalanced, adjust FocalLoss alpha per class |
| PSR head NaN during early training (BCE logits extreme) | M | L | Add PSR loss to MultiTaskLoss Kendall with neutral start (log_var=0), clip logits |
