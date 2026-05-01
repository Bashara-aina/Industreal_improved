### CONTRACT #10: Add Temporal Augmentation (random frame stride) to PopW

WHAT:
  Implement temporal augmentation with random frame stride during training to improve temporal robustness for IKEA ASM activity recognition and mcAP metrics.

FILES:
  READ:  /home/newadmin/swarm-bot/project/popw/working/code/popw_main_improved/ikea_dataset.py
  WRITE: /home/newadmin/swarm-bot/project/popw/working/code/popw_main_improved/ikea_dataset.py
  RUN:   python -c "import sys; sys.path.insert(0, '/home/newadmin/swarm-bot/project/popw/working/code/popw_main_improved'); from ikea_dataset import IKEAFrameDataset; print('IKEAFrameDataset imported')"

DONE_WHEN:
  - Config flag TRAIN_FRAME_STRIDE_RANGE = [3, 7] for random stride selection
  - Random frame stride applied per-video during training
  - Uniform sampling maintained for validation (EVAL_FRAME_STRIDE=1)
  - Augmentation respects temporal consistency within a video
  - Random stride between 3-7 frames per step

PROOF_FORMAT:
  CODE: `python -c "
import random
import sys
sys.path.insert(0, '/home/newadmin/swarm-bot/project/popw/working/code/popw_main_improved')
stride = random.randint(3, 7)
print(f'Random frame stride: {stride}')
"`

BLOCKER_IF:
  - Random stride causes index out of bounds in frame sampling
  - Augmentation breaks temporal label alignment

DEPENDS_ON: 1
