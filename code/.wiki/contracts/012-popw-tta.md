### CONTRACT #12: Add Test-Time Augmentation (horizontal flip) to PopW

WHAT:
  Implement test-time augmentation with horizontal flip for PopW, averaging predictions from original and flipped images to improve activity recognition accuracy.

FILES:
  READ:  /home/newadmin/swarm-bot/project/popw/working/code/popw_main_improved/evaluate.py
  WRITE: /home/newadmin/swarm-bot/project/popw/working/code/popw_main_improved/evaluate.py
  RUN:   python -c "import sys; sys.path.insert(0, '/home/newadmin/swarm-bot/project/popw/working/code/popw_main_improved'); from evaluate import evaluate_with_tta; print('evaluate_with_tta imported')"

DONE_WHEN:
  - evaluate_with_tta(model, dataloader, device) function implemented
  - Original image forward pass
  - Horizontally flipped image forward pass (flip RGB channels or transform)
  - Activity logits averaged: act_logits = (act_orig + act_flipped) / 2
  - Pose keypoints: x_coord = IMG_WIDTH - x_coord after flip
  - Detection boxes: x_min = IMG_WIDTH - x_max, x_max = IMG_WIDTH - x_min
  - TTA disabled when USE_TTA = False (config flag)
  - TTA enabled by default in eval mode

PROOF_FORMAT:
  CODE: `python -c "
import torch
import sys
sys.path.insert(0, '/home/newadmin/swarm-bot/project/popw/working/code/popw_main_improved')
from evaluate import evaluate_with_tta
print('TTA function available')
"`

BLOCKER_IF:
  - TTA doubles inference time (may not meet 291 FPS target)
  - Flip keypoint transformation produces incorrect coordinates

DEPENDS_ON: 1
