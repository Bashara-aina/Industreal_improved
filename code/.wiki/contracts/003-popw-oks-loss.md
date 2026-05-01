### CONTRACT #3: Implement OKS Loss for PopW pose estimation

WHAT:
  Replace Wing Loss with OKS (Object Keypoint Similarity) loss in losses.py for POPW pose head, improving PCK@10px accuracy on IKEA ASM benchmarks.

FILES:
  READ:  /home/newadmin/swarm-bot/project/popw/working/code/popw_main_improved/losses.py
  WRITE: /home/newadmin/swarm-bot/project/popw/working/code/popw_main_improved/losses.py
  RUN:   python -c "import sys; sys.path.insert(0, '/home/newadmin/swarm-bot/project/popw/working/code/popw_main_improved'); from losses import OKSLoss; print('OKSLoss imported successfully')"

DONE_WHEN:
  - OKSLoss class implemented in losses.py with forward(keypoints_pred, keypoints_gt, keypoint_scores) signature
  - OKS formula: OKS = exp(-d²/(2*s²*k²)) where d=distance, s=scale, k=per-keypoint constant
  - Per-keypoint k constants for COCO 17-keypoint layout defined
  - OKSLoss combines: (1) heatmap MSE loss, (2) OKS-based coordinate loss
  - Compatible with existing PoseHead output (heatmaps + coords)
  - Config flag `USE_OKS_LOSS = True` enables it in train.py
  - Train.py uses `OKSLoss()` when `C.USE_OKS_LOSS` is True

PROOF_FORMAT:
  CODE: `python -c "
import torch
import sys
sys.path.insert(0, '/home/newadmin/swarm-bot/project/popw/working/code/popw_main_improved')
from losses import OKSLoss
loss_fn = OKSLoss()
B, K = 2, 17
pred = torch.randn(B, K, 2)
target = torch.randn(B, K, 2)
scores = torch.rand(B, K).sigmoid()
loss = loss_fn(pred, target, scores)
print(f'OKS Loss: {loss.item():.4f}')
"` → outputs loss value

BLOCKER_IF:
  - OKSLoss produces NaN gradients
  - Per-keypoint constants not defined for COCO 17-keypoint format
  - Integration with PoseHead breaks existing code

DEPENDS_ON: 1
