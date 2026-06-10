### CONTRACT #5: Create IndustReal training script

WHAT:
  Create `/home/newadmin/swarm-bot/project/popw/working/code/industreal/train.py` — training script adapted from IKEA POPW train.py for IndustReal dataset. Jointly trains detection + head pose + activity + PSR. Mixed precision, gradient accumulation, Kendall uncertainty weighting for 4 tasks, class-balanced sampling, early stopping.

FILES:
  READ:  /media/newadmin/master/POPW/popw_main/train.py (full source, reuse training loop structure)
  WRITE: /home/newadmin/swarm-bot/project/popw/working/code/industreal/train.py

  Key changes from IKEA train.py:
  1. Import IndustRealMultiTaskDataset and its collate_fn instead of IKEAMultiTaskDataset
  2. Import MultiTaskIndustReal instead of MultiTaskIKEA
  3. Import losses module (not _ikea_ds_module)
  4. Import evaluate_all from evaluate.py (IndustReal version)
  5. Use C.NUM_CLASSES_ACT (74), C.NUM_DET_CLASSES (24), C.PSR_NUM_COMPONENTS (11)
  6. Detection evaluation: C.NUM_DET_CLASSES=24
  7. Activity evaluation: C.NUM_CLASSES_ACT=74, C.ACT_CLASS_NAMES from config
  8. Head pose evaluation: new metric (mean absolute error per DoF, PCK at fixed thresholds adapted for head pose)
  9. PSR evaluation: F1@10/25/50 (temporal F1 at 10/25/50 frame tolerance), Edit Score
  10. Combined validation metric: w_det*det_mAP50 + w_act*act_macro_f1 + w_pose*pose_MAE_inverse + w_psr*psr_F1
      - Use inverse MAE for pose: combined_pose = 1.0 / (1.0 + mean_abs_error)
      - Default weights: w_det=0.35, w_act=0.35, w_pose=0.15, w_psr=0.15
  11. Remove IKEAMultiTaskDataset-specific config (DATASET_MODE, DETECTION_MODE, USE_FILM)
  12. Remove _ikea_ds_module COCO cache clearing (use dataset.clear_coco_cache())
  13. Criterion: pass PSR_NUM_COMPONENTS for 4-task Kendall weighting
  14. Keep: AMP, gradient accumulation, NaN guard, early stopping, checkpoint saving, JSONL logging
  15. Validation: evaluate_all returns dict with det_mAP50, act_macro_f1, pose_mean_mae, psr_F1

  Combined metric computation:
  ```python
  combined = w_det * det_mAP50 + w_act * act_macro_f1 + w_pose * (1/(1+pose_mae)) + w_psr * psr_F1
  ```

  NaN handling for PSR: if psr_F1 is NaN (no positive labels in val), exclude from combined and renormalize weights.

DONE_WHEN:
  - /home/newadmin/swarm-bot/project/popw/working/code/industreal/train.py exists and >400 lines
  - Can import and instantiate MultiTaskIndustReal model
  - train.py runs without errors on first epoch (dry run with DEBUG_MODE if needed)
  - --resume flag works with checkpoint loading
  - JSONL logging includes all 4 task metrics

PROOF_FORMAT:
  python3 -c "
import sys; sys.path.insert(0, '/home/newadmin/swarm-bot/project/popw/working/code/industreal')
import config as C
import torch

# Test model can be built
from model import MultiTaskIndustReal
model = MultiTaskIndustReal(pretrained=True)
print(f'Model built OK, params: {sum(p.numel() for p in model.parameters()):,}')

# Test loss can be built
from losses import MultiTaskLoss
criterion = MultiTaskLoss(num_classes=C.NUM_CLASSES_ACT)
print('Loss built OK')

# Verify 4-task Kendall weights exist
assert hasattr(criterion, 'log_var_det')
assert hasattr(criterion, 'log_var_pose')
assert hasattr(criterion, 'log_var_act')
assert hasattr(criterion, 'log_var_psr'), 'Missing log_var_psr for 4th task'
print('4-task Kendall OK')

# Test forward pass with loss
images = torch.randn(1, 3, C.IMG_HEIGHT, C.IMG_WIDTH)
outputs = model(images)
targets = {
    'detection': [{'boxes': torch.randn(0, 4), 'labels': torch.zeros(0, dtype=torch.long)}],
    'keypoints': torch.randn(1, 17, 2),
    'visibility': torch.ones(1, 17),
    'kpt_confidence': torch.ones(1, 17),
    'activity': torch.tensor([0]),
    'head_pose': torch.randn(1, 9),
    'psr_labels': torch.randint(0, 2, (1, 11)).float(),
}
total_loss, ld = criterion(outputs, targets)
print(f'Loss total: {ld[\"total\"]:.4f}')
print('train.py setup OK')
"

BLOCKER_IF:
  - train.py fails to import any module
  - Model forward raises shape errors
  - 4-task Kendall weighting not properly implemented

DEPENDS_ON: 1, 2, 3, 4 (all prior contracts must be complete)
