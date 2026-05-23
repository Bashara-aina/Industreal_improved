#!/bin/bash
# =============================================================================
# 10% Subset Quick Training — Verify Losses Are Non-NaN / Non-Zero
# =============================================================================
# Goals:
#   - 10% dataset (SUBSET_RATIO=0.10)
#   - No staged training — all 5 heads active from epoch 0
#   - Fast eval (500 batches) to get val metrics every epoch
#   - 5 epochs to see convergence with non-NaN losses
# =============================================================================

set -e

PROJ_DIR="/home/newadmin/swarm-bot/project/popw/working/code/industreal_improved_to_archive/src"
# FIXED: crash_recovery.pth is at the repo root, not in src/runs/
CKPT="/home/newadmin/popw/project/working/code/industreal_improved_to_archieve/checkpoints/popw_v2/crash_recovery.pth"

echo "============================================"
echo "10% subset quick training (5 epochs)"
echo "============================================"
echo "Checkpoint: $CKPT"
echo ""

cd "$PROJ_DIR"

# --- Patch config.py for 10% subset, no staged training, fast eval ---
python -c "
import re

cfg_path = 'config.py'
with open(cfg_path, 'r') as f:
    cfg = f.read()

patches = {
    'SUBSET_RATIO':     '0.10   # 10% subset for quick training',
    'EVAL_MAX_BATCHES': '500   # ~4 min eval per epoch',
    'VAL_BATCH_SIZE':   '2   # RTX 3060 stable',
    'VAL_NUM_WORKERS':  '0   # No subprocess workers',
    'STAGED_TRAINING':  'False   # All 5 heads active from epoch 0',
    'BENCHMARK_MODE':   'True   # VAL_EVERY=1 every epoch',
    'MAX_EPOCHS':       '5   # 5 epochs enough to see convergence',
    'TRAIN_MAX_STEPS':  '0   # Disable step limit (use epochs)',
}

for key, val in patches.items():
    cfg = re.sub(rf'^{key}\s*=.*', f'{key} = {val}', cfg, flags=re.M)

with open(cfg_path, 'w') as f:
    f.write(cfg)
print('config.py patched:')
for k, v in patches.items():
    print(f'  {k} = {v}')
"

echo ""
echo "Starting training..."
echo "============================================"

cd "$PROJ_DIR"
exec python training/train.py \
    --resume "$CKPT" \
    --subset-ratio 0.10 \
    --seed 42 \
    --no-staged-training \
    --max-epochs 5 \
    --num-workers 0 \
    2>&1