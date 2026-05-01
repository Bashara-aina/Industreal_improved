### CONTRACT #4: Create IndustReal loss functions

WHAT:
  Create `/home/newadmin/swarm-bot/project/popw/working/code/industreal/losses.py` — loss functions for all 4 tasks: FocalLoss (detection, unchanged from IKEA), MSELoss/HeadPoseLoss (9-DoF head pose regression), ClassBalancedFocalLoss (activity, unchanged from IKEA), BCELoss (PSR multi-label), MultiTaskLoss (Kendall uncertainty weighting for all 4 tasks).

FILES:
  READ:  /media/newadmin/master/POPW/popw_main/losses.py (full source, reuse FocalLoss, ClassBalancedFocalLoss, MultiTaskLoss)
  WRITE: /home/newadmin/swarm-bot/project/popw/working/code/industreal/losses.py

  Key changes:
  1. HeadPoseLoss: MSE on normalized 9-DoF head pose. Normalize each dimension by its training set std. Use only frames where confidence > threshold (filter low-confidence pose predictions).
  2. PSRLoss: BCEWithLogitsLoss for 11-component multi-label classification. Sum over components, mean over batch.
  3. MultiTaskLoss: extend Kendall uncertainty weighting to 4 tasks. Add log_var_psr parameter. Activity warmup applies only to activity task (not PSR). PSR task has no warmup.
  4. ClassBalancedFocalLoss: set_class_counts must work with 74 activity classes.
  5. FocalLoss: identical to IKEA (anchor matching, focal loss, smooth L1 regression).

  Kendall uncertainty for 4 tasks:
  - log_var_det = 0.0 (neutral start)
  - log_var_pose = -1.0 (compensate for smaller loss scale at init)
  - log_var_act = 0.0 (neutral start)
  - log_var_psr = 0.0 (neutral start)
  - Combined: sum over all 4 tasks of [exp(-log_var_t) * L_t + log_var_t]

  Activity warmup: same as IKEA (ACT_WARMUP_EPOCHS ramp), does NOT affect PSR.

DONE_WHEN:
  - /home/newadmin/swarm-bot/project/popw/working/code/industreal/losses.py exists and >300 lines
  - FocalLoss (detection), HeadPoseLoss (9-DoF MSE), BCELoss (PSR), ClassBalancedFocalLoss (activity), MultiTaskLoss all defined
  - MultiTaskLoss.set_class_counts() works for activity
  - MultiTaskLoss.set_epoch() handles activity warmup correctly
  - MultiTaskLoss forward returns loss_dict with 4 task losses + Kendall weights
  - Loss can be computed from model outputs and targets

PROOF_FORMAT:
  python3 -c "
import sys, torch
sys.path.insert(0, '/home/newadmin/swarm-bot/project/popw/working/code/industreal')
import config as C
from losses import FocalLoss, HeadPoseLoss, BCELoss, ClassBalancedFocalLoss, MultiTaskLoss

# Test HeadPoseLoss
head_pose_loss = HeadPoseLoss()
pred = torch.randn(4, 9)
gt = torch.randn(4, 9)
vis = torch.ones(4, 9)  # all visible
loss = head_pose_loss(pred, gt, vis)
assert loss.numel() == 1 or loss.dim() == 0, f'HeadPoseLoss should return scalar, got shape {loss.shape}'
print(f'HeadPoseLoss: {loss.item():.4f}')

# Test BCELoss
psr_loss_fn = BCELoss()
psr_pred = torch.randn(4, 11)
psr_gt = torch.randint(0, 2, (4, 11)).float()
loss = psr_loss_fn(psr_pred, psr_gt)
assert loss.numel() == 1 or loss.dim() == 0
print(f'BCELoss: {loss.item():.4f}')

# Test MultiTaskLoss
mtl = MultiTaskLoss(num_classes=C.NUM_CLASSES_ACT)
outputs = {
    'cls_preds': torch.randn(2, 57600, C.NUM_DET_CLASSES),
    'reg_preds': torch.randn(2, 57600, 4),
    'anchors': torch.randn(57600, 4),
    'head_pose': torch.randn(2, 9),
    'act_logits': torch.randn(2, C.NUM_CLASSES_ACT),
    'psr_logits': torch.randn(2, 11),
}
targets = {
    'detection': [{'boxes': torch.randn(3, 4), 'labels': torch.randint(1, C.NUM_DET_CLASSES, (3,))} for _ in range(2)],
    'keypoints': torch.randn(2, 17, 2),
    'visibility': torch.ones(2, 17),
    'kpt_confidence': torch.ones(2, 17),
    'activity': torch.randint(0, C.NUM_CLASSES_ACT, (2,)),
    'head_pose': torch.randn(2, 9),
    'psr_labels': torch.randint(0, 2, (2, 11)).float(),
}
total, ld = mtl(outputs, targets)
assert 'total' in ld
assert 'psr' in ld
print(f'MultiTaskLoss total: {ld[\"total\"]:.4f}')
print('losses.py OK')
"

BLOCKER_IF:
  - Source losses.py not readable
  - Any loss computation raises shape mismatch
  - Kendall uncertainty not extended to 4 tasks properly

DEPENDS_ON: 1, 3 (config.py and model.py must exist first)
