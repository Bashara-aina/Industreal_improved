### CONTRACT #13: Add Cosine Annealing with Warmup scheduler to PopW and IndustReal

WHAT:
  Implement cosine annealing learning rate scheduler with linear warmup for both PopW and IndustReal training loops.

FILES:
  READ:  /home/newadmin/swarm-bot/project/popw/working/code/popw_main_improved/train.py
         /home/newadmin/swarm-bot/project/popw/working/code/industreal_improved/train.py
  WRITE: /home/newadmin/swarm-bot/project/popw/working/code/popw_main_improved/train.py
         /home/newadmin/swarm-bot/project/popw/working/code/industreal_improved/train.py
  RUN:   grep -c "CosineAnnealingWarmRestarts" /home/newadmin/swarm-bot/project/popw/working/code/popw_main_improved/train.py

DONE_WHEN:
  - Scheduler: torch.optim.lr_scheduler.CosineAnnealingWarmRestarts with T_0=10, T_mult=2
  - Warmup: LinearLR for first WARMUP_EPOCHS=5 with start_factor=0.1
  - Combined scheduler: SequentialLR with warmup then cosine
  - Config: USE_COSINE_ANNEALING = True
  - Works with frozen backbone (only head params learning rate applied)

PROOF_FORMAT:
  CODE: `python -c "
import torch
optimizer = torch.optim.Adam([torch.randn(10, 10).requires_grad_()], lr=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
print(f'Cosine scheduler: {scheduler}')
"`

BLOCKER_IF:
  - Frozen backbone causes param group mismatch
  - T_0 too small causes oscillation

DEPENDS_ON: 1, 5
