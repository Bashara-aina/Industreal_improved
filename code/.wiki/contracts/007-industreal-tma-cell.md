### CONTRACT #7: Implement GRU-based TMA Cell for IndustReal

WHAT:
  Implement Temporal Masked Attention (TMA) cell inspired by PTMA (arXiv:2508.17025) using GRU recurrence and masked self-attention for temporal sequence modeling in IndustReal activity recognition.

FILES:
  READ:  /home/newadmin/swarm-bot/project/popw/working/code/industreal_improved/model.py
  WRITE: /home/newadmin/swarm-bot/project/popw/working/code/industreal_improved/model.py
  RUN:   python -c "import sys; sys.path.insert(0, '/home/newadmin/swarm-bot/project/popw/working/code/industreal_improved'); from model import TMACell; print('TMACell imported')"

DONE_WHEN:
  - TMACell class implemented with forward(sequence: List[Tensor]) -> Tensor interface
  - GRU layer: hidden_size=256, num_layers=1, batch_first=True
  - Masked self-attention: embed_dim=256, num_heads=4, mask_future=True (causal attention)
  - Probabilistic modeling component (PTMA inspiration): outputs mean + logvar for distribution
  - Sequence length: T=8 frames per clip
  - Returns aggregated temporal features [B, 256] for activity classification
  - Config flag `USE_TMA_CELL = True` enables it in MultiTaskIndustReal.forward()
  - TMA processes C5 features before ActivityHead

PROOF_FORMAT:
  CODE: `python -c "
import torch
import sys
sys.path.insert(0, '/home/newadmin/swarm-bot/project/popw/working/code/industreal_improved')
from model import TMACell
T, B, C = 8, 2, 768
cell = TMACell(feat_channels=C, hidden_size=256, num_heads=4)
sequence = [torch.randn(B, C, 20, 30) for _ in range(T)]
out = cell(sequence)
print(f'TMA output shape: {out.shape}')
"` → torch.Size([2, 256])

BLOCKER_IF:
  - GRU hidden state not properly initialized
  - Masked attention produces NaN
  - Sequence processing breaks batch dimension handling

DEPENDS_ON: 5
