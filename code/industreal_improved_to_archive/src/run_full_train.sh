#!/bin/bash
# =============================================================================
# Full-Dataset Single-Seed Training Launcher
# =============================================================================
# Features:
#   - Full dataset (subset_ratio=1.0)
#   - Single seed (42)
#   - No staged training — all 5 heads active from epoch 0
#   - Eval every epoch with full validation set (EVAL_MAX_BATCHES=0)
#   - Resume from crash_recovery.pth
#
# RTX 3060 12GB constraints:
#   - VAL_BATCH_SIZE=2 (stability), VAL_NUM_WORKERS=0 (no SHM crash)
#   - BATCH_SIZE=4, GRAD_ACCUM_STEPS=8 (effective batch 32)
# =============================================================================

set -e

PROJ_DIR="/home/newadmin/swarm-bot/project/popw/working/code/industreal_improved_to_archive/src"
CKPT="$PROJ_DIR/runs/full_multi_task_tma_tbank_benchmark/checkpoints/crash_recovery.pth"
CONFIG="$PROJ_DIR/src/config.py"

echo "============================================"
echo "Full-dataset single-seed training"
echo "============================================"
echo "Checkpoint : $CKPT"
echo "Project dir: $PROJ_DIR"
echo ""

# --- Patch config.py to enable full evaluation and RTX 3060 settings ---
cd "$PROJ_DIR"   # PROJ_DIR = .../src/  (config.py is at $PROJ_DIR/config.py)

python -c "
import re

cfg_path = 'config.py'
with open(cfg_path, 'r') as f:
    cfg = f.read()

originals = {}
for key in ['EVAL_MAX_BATCHES', 'VAL_BATCH_SIZE', 'VAL_NUM_WORKERS', 'STAGED_TRAINING', 'BENCHMARK_MODE']:
    m = re.search(rf'^{key}\s*=\s*(.+)', cfg, re.M)
    originals[key] = m.group(1).strip() if m else 'not found'

print(f'  EVAL_MAX_BATCHES: {originals[\"EVAL_MAX_BATCHES\"]} → 0 (full eval)')
print(f'  VAL_BATCH_SIZE:   {originals[\"VAL_BATCH_SIZE\"]} → 2 (RTX 3060)')
print(f'  VAL_NUM_WORKERS:  {originals[\"VAL_NUM_WORKERS\"]} → 0 (no SHM workers)')
print(f'  STAGED_TRAINING: {originals[\"STAGED_TRAINING\"]} → False')
print(f'  BENCHMARK_MODE:   {originals[\"BENCHMARK_MODE\"]} → True')

cfg = re.sub(r'^EVAL_MAX_BATCHES\s*=.*', 'EVAL_MAX_BATCHES = 500   # ~4 min eval (capped for SIGTERM-safe)', cfg, flags=re.M)
cfg = re.sub(r'^VAL_BATCH_SIZE\s*=.*',   'VAL_BATCH_SIZE   = 2   # RTX 3060: batch=2 stable', cfg, flags=re.M)
cfg = re.sub(r'^VAL_NUM_WORKERS\s*=.*',  'VAL_NUM_WORKERS  = 0   # No subprocess workers (avoids SHM crash)', cfg, flags=re.M)
cfg = re.sub(r'^STAGED_TRAINING\s*=.*', 'STAGED_TRAINING  = False   # All 5 heads active from epoch 0', cfg, flags=re.M)
cfg = re.sub(r'^BENCHMARK_MODE\s*=.*',   'BENCHMARK_MODE   = True    # VAL_EVERY=1 every epoch', cfg, flags=re.M)

with open(cfg_path, 'w') as f:
    f.write(cfg)
print('  config.py patched successfully.')
"

echo ""
echo "Starting training..."
echo "============================================"

# Run training
cd "$PROJ_DIR"
exec python ../files/train.py \
    --resume "$CKPT" \
    --subset-ratio 1.0 \
    --seed 42 \
    --no-staged-training \
    --max-epochs 100 \
    --num-workers 0 \
    2>&1
