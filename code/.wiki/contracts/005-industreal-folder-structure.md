### CONTRACT #5: Create improved folder structure for IndustReal

WHAT:
  Create `/home/newadmin/swarm-bot/project/popw/working/code/industreal_improved/` directory with ConvNeXt-Tiny backbone, TMA cell, and temporal bank integration for IndustReal.

FILES:
  READ:  /home/newadmin/swarm-bot/project/popw/working/code/industreal/model.py
         /home/newadmin/swarm-bot/project/popw/working/code/industreal/config.py
         /home/newadmin/swarm-bot/project/popw/working/code/industreal/train.py
         /home/newadmin/swarm-bot/project/popw/working/code/industreal/evaluate.py
         /home/newadmin/swarm-bot/project/popw/working/code/industreal/losses.py
         /home/newadmin/swarm-bot/project/popw/working/code/industreal/industreal_dataset.py
  WRITE: /home/newadmin/swarm-bot/project/popw/working/code/industreal_improved/model.py
         /home/newadmin/swarm-bot/project/popw/working/code/industreal_improved/config.py
         /home/newadmin/swarm-bot/project/popw/working/code/industreal_improved/train.py
         /home/newadmin/swarm-bot/project/popw/working/code/industreal_improved/evaluate.py
         /home/newadmin/swarm-bot/project/popw/working/code/industreal_improved/losses.py
         /home/newadmin/swarm-bot/project/popw/working/code/industreal_improved/industreal_dataset.py
  RUN:   mkdir -p /home/newadmin/swarm-bot/project/popw/working/code/industreal_improved

DONE_WHEN:
  - Directory `/home/newadmin/swarm-bot/project/popw/working/code/industreal_improved/` exists
  - All 6 Python files are copied to the improved directory
  - `model.py` imports `timm` for ConvNeXt-Tiny
  - Config sets `BACKBONE = 'convnext_tiny'` and `USE_TMA_CELL = True` and `USE_TEMPORAL_BANK = True`
  - TemporalBankModule class with T=8 and T=32 temporal levels exists in model.py
  - TMACell class (GRU-based temporal masked attention) exists in model.py

PROOF_FORMAT:
  FILE_OP: `ls -la /home/newadmin/swarm-bot/project/popw/working/code/industreal_improved/`
  CODE: `python -c "import timm; print('timm available')"`
  CODE: `grep -c "TMACell" /home/newadmin/swarm-bot/project/popw/working/code/industreal_improved/model.py`
  CODE: `grep -c "TemporalBankModule" /home/newadmin/swarm-bot/project/popw/working/code/industreal_improved/model.py`

BLOCKER_IF:
  - timm not installed
  - Original industreal files cannot be read
  - Temporal processing changes break existing data loader compatibility

DEPENDS_ON: none
