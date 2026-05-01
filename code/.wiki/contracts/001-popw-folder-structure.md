### CONTRACT #1: Create improved folder structure for PopW

WHAT:
  Create `/home/newadmin/swarm-bot/project/popw/working/code/popw_main_improved/` directory with all required Python files copied from original popw_main, then apply ConvNeXt-Tiny backbone replacement and OKS Loss integration.

FILES:
  READ:  /media/newadmin/master/POPW/popw_main/model.py
         /media/newadmin/master/POPW/popw_main/config.py
         /media/newadmin/master/POPW/popw_main/train.py
         /media/newadmin/master/POPW/popw_main/evaluate.py
         /media/newadmin/master/POPW/popw_main/losses.py
         /media/newadmin/master/POPW/popw_main/ikea_dataset.py
  WRITE: /home/newadmin/swarm-bot/project/popw/working/code/popw_main_improved/model.py
         /home/newadmin/swarm-bot/project/popw/working/code/popw_main_improved/config.py
         /home/newadmin/swarm-bot/project/popw/working/code/popw_main_improved/train.py
         /home/newadmin/swarm-bot/project/popw/working/code/popw_main_improved/evaluate.py
         /home/newadmin/swarm-bot/project/popw/working/code/popw_main_improved/losses.py
         /home/newadmin/swarm-bot/project/popw/working/code/popw_main_improved/ikea_dataset.py
  RUN:   mkdir -p /home/newadmin/swarm-bot/project/popw/working/code/popw_main_improved

DONE_WHEN:
  - Directory `/home/newadmin/swarm-bot/project/popw/working/code/popw_main_improved/` exists
  - All 6 Python files are copied to the improved directory
  - `model.py` imports `timm` for ConvNeXt-Tiny
  - `losses.py` contains OKS Loss implementation
  - `config.py` sets `BACKBONE = 'convnext_tiny'` and `USE_OKS_LOSS = True`

PROOF_FORMAT:
  FILE_OP: `ls -la /home/newadmin/swarm-bot/project/popw/working/code/popw_main_improved/`
  CODE: `python -c "import timm; print(timm.list_models('convnext_tiny*')[0])"` → convnext_tiny

BLOCKER_IF:
  - timm is not installed (`import timm` fails)
  - ConvNeXt-Tiny pretrained weights not available in timm
  - Original popw_main files cannot be read

DEPENDS_ON: none
