### CONTRACT #8: Implement Two-Level Temporal Bank T=8 + T=32 for IndustReal

WHAT:
  Implement two-level temporal bank module processing sequences at T=8 (short-term) and T=32 (long-term) temporal strides, concatenating features for comprehensive temporal modeling.

FILES:
  READ:  /home/newadmin/swarm-bot/project/popw/working/code/industreal_improved/model.py
  WRITE: /home/newadmin/swarm-bot/project/popw/working/code/industreal_improved/model.py
  RUN:   python -c "import sys; sys.path.insert(0, '/home/newadmin/swarm-bot/project/popw/working/code/industreal_improved'); from model import TemporalBankModule; print('TemporalBankModule imported')"

DONE_WHEN:
  - TemporalBankModule class implemented with forward(frames: List[Tensor]) -> Tensor
  - Level 1: T=8 frames → Conv1D temporal encoding (kernel=3, channels=256)
  - Level 2: T=32 frames → Conv1D temporal encoding (kernel=3, channels=256)
  - Concatenates [B, 512] features from both levels for activity head
  - Frame subsampling: every 4th frame for T=32, every frame for T=8
  - Compatible with existing dataset loader frame sampling
  - Config flags: TEMPORAL_BANK_T8=8, TEMPORAL_BANK_T32=32
  - Feature concatenation: torch.cat([t8_features, t32_features], dim=-1)

PROOF_FORMAT:
  CODE: `python -c "
import torch
import sys
sys.path.insert(0, '/home/newadmin/swarm-bot/project/popw/working/code/industreal_improved')
from model import TemporalBankModule
T8, T32 = 8, 32
B, C = 2, 256
bank = TemporalBankModule(feat_channels=C, hidden_channels=256)
# Simulate T=8 short-term frames
t8_frames = [torch.randn(B, C, 1, 1) for _ in range(T8)]
# Simulate T=32 long-term frames
t32_frames = [torch.randn(B, C, 1, 1) for _ in range(T32)]
out = bank(t8_frames, t32_frames)
print(f'TemporalBank output shape: {out.shape}')
"` → torch.Size([2, 512])

BLOCKER_IF:
  - Temporal bank causes memory explosion with T=32 sequences
  - Frame stride subsampling produces incorrect temporal alignment
  - Feature concatenation causes dimension mismatch with ActivityHead

DEPENDS_ON: 5
