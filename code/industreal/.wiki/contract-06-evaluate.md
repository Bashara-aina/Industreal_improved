### CONTRACT #6: Create IndustReal evaluation metrics

WHAT:
  Create `/home/newadmin/swarm-bot/project/popw/working/code/industreal/evaluate.py` — evaluation script adapted from IKEA POPW evaluate.py for IndustReal. Returns detection mAP@0.5, activity macro-F1/accuracy/top-5, head pose MAE per DoF, PSR F1@10/25/50 and Edit Score. Full evaluate_all() function.

FILES:
  READ:  /media/newadmin/master/POPW/popw_main/evaluate.py (full source, reuse compute_activity_metrics, compute_det_metrics_extended)
  WRITE: /home/newadmin/swarm-bot/project/popw/working/code/industreal/evaluate.py

  Key changes from IKEA evaluate.py:
  1. Import MultiTaskIndustReal instead of MultiTaskIKEA
  2. Import IndustRealMultiTaskDataset and its collate_fn
  3. Use C.NUM_DET_CLASSES=24, C.NUM_CLASSES_ACT=74, C.PSR_NUM_COMPONENTS=11, C.NUM_KEYPOINTS=9
  4. replace compute_pose_metrics_extended with compute_head_pose_metrics:
     - Mean Absolute Error per DoF (9 values)
     - Mean Absolute Error overall (scalar)
     - MAE per dimension: forward_x, forward_y, forward_z, pos_x, pos_y, pos_z, up_x, up_y, up_z
     - Optional: R² correlation coefficient per DoF
  5. Add compute_psr_metrics:
     - PSR is multi-label: predict binary state for 11 components per frame
     - Compute per-component F1, then macro-F1 across components
     - F1@10/25/50: temporal F1 with frame tolerance of 10/25/50 frames
       (treating each component as a binary sequence, compute F1 allowing ±10/25/50 frame tolerance)
     - Edit Score: Levenshtein distance between predicted and ground-truth state change frames
     - Overall PSR F1 = macro-F1 across 11 components at exact frame matching
  6. Keep compute_activity_metrics (same format as IKEA, works with 74 classes)
  7. Keep compute_det_metrics_extended (same format, works with COCO format)
  8. evaluate_all() collects: act_preds/gt, head_pose_preds/gt, psr_preds/gt, det boxes/scores/labels
  9. Returns dict with: det_mAP50, det_mAP_50_95, act_accuracy, act_macro_f1, act_weighted_f1, act_top5, head_pose_MAE (per DoF and overall), psr_F1, psr_F1_tol10, psr_F1_tol25, psr_F1_tol50, psr_edit_score

  PSR F1@T (tolerance) algorithm:
  ```
  For each component (11 total):
    1. Get predicted binary sequence (N frames)
    2. Get ground-truth binary sequence
    3. Find change points in both sequences (frames where state flips 0→1)
    4. For each GT change point, find predicted change within ±T frames
    5. True Positive = predicted change point within tolerance of GT
    6. False Positive = predicted change point not matched to any GT within tolerance
    7. False Negative = GT change point not matched to any prediction within tolerance
    8. F1 = 2*TP / (2*TP + FP + FN)
  Overall F1@T = macro-average over 11 components
  ```

DONE_WHEN:
  - /home/newadmin/swarm-bot/project/popw/working/code/industreal/evaluate.py exists and >400 lines
  - evaluate_all() function defined
  - compute_head_pose_metrics() returns MAE per DoF and overall
  - compute_psr_metrics() returns F1, F1@10/25/50, Edit Score
  - compute_activity_metrics() and compute_det_metrics_extended() work unchanged
  - Standalone CLI works: python evaluate.py --checkpoint path/to/checkpoint.pth

PROOF_FORMAT:
  python3 -c "
import sys; sys.path.insert(0, '/home/newadmin/swarm-bot/project/popw/working/code/industreal')
import config as C
import torch
import numpy as np

from evaluate import compute_head_pose_metrics, compute_psr_metrics, compute_activity_metrics

# Test head pose metrics
pred_pose = np.random.randn(100, 9).astype(np.float32)
gt_pose = np.random.randn(100, 9).astype(np.float32)
# Add some variation
pred_pose[:50] = gt_pose[:50] + 0.1
metrics = compute_head_pose_metrics(pred_pose, gt_pose)
required = ['head_pose_MAE', 'head_pose_MAE_per_dof', 'head_pose_MAE_overall']
for k in required:
    assert k in metrics, f'Missing head pose metric: {k}'
print(f'Head pose MAE overall: {metrics[\"head_pose_MAE_overall\"]:.4f}')

# Test PSR metrics
psr_pred = np.random.randint(0, 2, (100, 11)).astype(np.float32)
psr_gt = psr_pred.copy()
# Make 20% different
flip_idx = np.random.choice([True, False], size=(100, 11))
psr_gt = np.where(flip_idx, 1 - psr_gt, psr_gt)
psr_metrics = compute_psr_metrics(psr_pred, psr_gt)
required_psr = ['psr_F1', 'psr_F1_tol10', 'psr_F1_tol25', 'psr_F1_tol50', 'psr_edit_score']
for k in required_psr:
    assert k in psr_metrics, f'Missing PSR metric: {k}'
print(f'PSR F1: {psr_metrics[\"psr_F1\"]:.4f}, Edit: {psr_metrics[\"psr_edit_score\"]:.4f}')

print('evaluate.py OK')
"

BLOCKER_IF:
  - evaluate.py fails to import
  - Head pose metrics missing required fields
  - PSR metrics missing required fields

DEPENDS_ON: 1, 2, 3, 4 (all prior contracts must be complete)
