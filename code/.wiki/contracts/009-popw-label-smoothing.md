### CONTRACT #9: Add Label Smoothing to PopW activity classification

WHAT:
  Implement label smoothing (smoothing=0.1) for CrossEntropyLoss in PopW activity classification to improve generalization on IKEA ASM 33-class activity recognition.

FILES:
  READ:  /home/newadmin/swarm-bot/project/popw/working/code/popw_main_improved/train.py
  WRITE: /home/newadmin/swarm-bot/project/popw/working/code/popw_main_improved/train.py
  RUN:   python -c "import sys; sys.path.insert(0, '/home/newadmin/swarm-bot/project/popw/working/code/popw_main_improved'); from train import train_one_epoch; print('train_one_epoch imported')"

DONE_WHEN:
  - Label smoothing enabled via config: LABEL_SMOOTHING = 0.1
  - CrossEntropyLoss with label_smoothing=0.1 used for activity classification loss
  - Compatible with class weights for imbalanced classes (if USE_CLASS_WEIGHTS=True)
  - Smoothed labels computed as: (1 - eps) * one_hot + eps / num_classes

PROOF_FORMAT:
  CODE: `python -c "
import torch
import torch.nn as nn
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
logits = torch.randn(4, 33)
labels = torch.randint(0, 33, (4,))
loss = loss_fn(logits, labels)
print(f'Label smoothing loss: {loss.item():.4f}')
"`

BLOCKER_IF:
  - label_smoothing parameter not available in PyTorch CrossEntropyLoss
  - Class weights + label smoothing interaction causes NaN

DEPENDS_ON: 1
