#!/bin/bash
# =============================================================================
# POPW — Full-Dataset 100-Epoch Production Training
# =============================================================================
# Purpose : Train POPW on 100% IndustReal/IKEA ASM dataset for paper results.
#           All metrics needed for popw_paper_improved.tex are captured.
#
# Features:
#   • Full dataset (subset_ratio=1.0, no staging)
#   • 100 epochs, all 5 heads active from epoch 0
#   • Full evaluation every epoch (no batch cap)
#   • Best + last checkpoint per epoch saved to runs/popw_full_100e/
#   • metrics.jsonl + CSV for all paper metrics
#   • Kendall log_vars tracked per epoch
#
# Metrics captured for paper (popw_paper_improved.tex):
#   • ASD mAP@0.5 (bbox frames) + mAP@[0.5:0.95] + mAP@0.5 (all frames)
#   • Activity: Top-1, Top-5, Macro-F1, Weighted-F1, Frame-Acc (all + no NA)
#   • Head Pose: Forward MAE (°), Up MAE (°), Position MAE (mm), Overall raw
#   • PSR: F1@±3, F1@±5, Edit Score, POS
#   • Assembly State: F1@1, Top-1 Acc, MAP@R(+)
#   • Error Verification: AP, F1, Precision, Recall
#   • Kendall weights: log_var_{det,pose,act,psr}
#   • Combined metric for model selection
#
# RTX 3060 12GB safe settings applied automatically.
# =============================================================================

set -e

PROJ_DIR="/home/newadmin/swarm-bot/project/popw/working/code/industreal_improved_to_archive/src"
RUN_NAME="popw_full_100e"
LOG_DIR="$PROJ_DIR/runs/$RUN_NAME/logs"
CKPT_DIR="$PROJ_DIR/runs/$RUN_NAME/checkpoints"
CKPT="$CKPT_DIR/crash_recovery.pth"

mkdir -p "$LOG_DIR" "$CKPT_DIR"

echo "============================================"
echo "POPW — Full 100-epoch production training"
echo "============================================"
echo "Run name   : $RUN_NAME"
echo "Checkpoint : $CKPT"
echo "Logs       : $LOG_DIR"
echo "Dataset    : 100% (subset-ratio=1.0)"
echo "Epochs     : 100"
echo "Staging    : OFF (all 5 heads from epoch 0)"
echo "Eval       : full validation set every epoch"
echo "============================================"
echo ""

# --- Patch config.py with RTX 3060 safe settings for long run ---
cd "$PROJ_DIR"

python -c "
import re

cfg_path = 'config.py'
with open(cfg_path, 'r') as f:
    cfg = f.read()

patches = {
    'EVAL_MAX_BATCHES': ('-1',       'full val set every epoch (no cap, -1=unlimited)'),
    'USE_VIDEOMAE':     ('True',     'enable VideoMAE V2 stream for +5-7% activity boost'),
    'BATCH_SIZE':       ('6',         'push VRAM: batch=6 for RTX 3060 12GB'),
    'GRAD_ACCUM_STEPS': ('6',        'effective batch = 6×6 = 36 (~paper 32)'),
    'VAL_BATCH_SIZE':   ('8',        'validation benefits from larger batch'),
    'VAL_NUM_WORKERS':  ('4',        '4 workers for val prefetch'),
    'STAGED_TRAINING': ('False',    'all heads active from epoch 0'),
    'BENCHMARK_MODE':   ('True',     'VAL_EVERY=1 every epoch'),
    'NUM_WORKERS':      ('8',        '8 workers — 64GB RAM available'),
    'PIN_MEMORY':       ('True',     'faster dataloader'),
    'TORCH_NUM_THREADS':('12',       'CPU threads for 64GB machine'),
    'CUDA_MEMORY_FRACTION': ('0.92', '92% VRAM'),
}

for key, (val, desc) in patches.items():
    m = re.search(rf'^{key}\s*=\s*(.+)', cfg, re.M)
    old = m.group(1).strip() if m else 'not set'
    if re.search(rf'^{key}\s*=', cfg, re.M):
        cfg = re.sub(rf'^{key}\s*=.*', f'{key} = {val}', cfg, flags=re.M)
    else:
        cfg += f'\n{key} = {val}  # added by run_full_production.sh'
    print(f'  {key:25s}  {old:>12s} → {val}  ({desc})')

with open(cfg_path, 'w') as f:
    f.write(cfg)
print('  config.py patched.')
"

echo ""
echo "Starting training at $(date)"
echo "============================================"

# --- Run training (no tee — tee causes 8KB Python stdout buffering) ---
cd "$PROJ_DIR"

# SIGTERM handler is already built into train.py (Bashara 2026-05-08)
# SIGUSR1 → gracefully finish epoch and save
# SIGTERM  → save crash_recovery.pth and exit cleanly
PYTHONUNBUFFERED=1 nohup python -u training/train.py \
    --no-staged-training \
    --subset-ratio 1.0 \
    --seed 42 \
    --max-epochs 100 \
    > "$LOG_DIR/train.log" 2>&1 &
TRAIN_PID=$!
echo $TRAIN_PID > "$CKPT_DIR/train_pid"
echo "Training started (PID=$TRAIN_PID). Monitor with:"
echo "  tail -f $LOG_DIR/train.log"
echo ""
echo "To stop: kill -9 $(cat $CKPT_DIR/train_pid 2>/dev/null) 2>/dev/null"
echo "To check GPU: nvidia-smi --query-gpu=memory.used --format=csv,noheader"

echo "To stop: kill -9 \$(cat $CKPT_DIR/train_pid 2>/dev/null) 2>/dev/null"
echo "To check GPU: nvidia-smi --query-gpu=memory.used --format=csv,noheader"
echo ""
echo "============================================"
echo "Training is running in background."
echo "Log: $LOG_DIR/train.log"
echo "Checkpoints: $CKPT_DIR"
echo "============================================"
echo ""
echo "After training finishes, extract CSV with:"
echo "  python -c \"\$(cat <<'PYEOF'
import json,csv
from pathlib import Path
log=Path('runs/popw_full_100e/logs/metrics.jsonl')
out=Path('runs/popw_full_100e/logs/paper_metrics.csv')
rows=[]
for line in open(log):
 d=json.loads(line); v=d.get('val',{}); t=d.get('train',{}); m=v or {}; tr=t or {}
 rows.append({'epoch':d.get('epoch',''),'lr':d.get('lr',''),
  'det_mAP50_bbox':m.get('det_mAP50',''),'det_mAP50_all':m.get('det_mAP50_all_frames',''),'det_mAP50_95':m.get('det_mAP_50_95',''),
  'act_top1':m.get('act_accuracy',''),'act_top5':m.get('act_top5_accuracy',''),'act_macro_f1':m.get('act_macro_f1',''),'act_weighted_f1':m.get('act_weighted_f1',''),'act_frame_acc':m.get('act_frame_accuracy',''),'act_macro_recall':m.get('act_macro_recall',''),
  'hp_forward_deg':m.get('forward_angular_MAE_deg',''),'hp_up_deg':m.get('up_angular_MAE_deg',''),'hp_pos_mm':m.get('position_MAE_mm',''),'hp_overall_raw':m.get('head_pose_MAE',''),
  'psr_f1_3':m.get('psr_f1_at_t',''),'psr_f1_5':m.get('psr_f1_at_t5',''),'psr_edit':m.get('psr_edit_score',''),'psr_pos':m.get('psr_pos',''),
  'as_f1_1':m.get('as_f1',''),'as_top1':m.get('as_top1_accuracy',''),'as_map_r':m.get('as_map_at_r',''),
  'err_ap':m.get('ev_ap',''),'err_f1':m.get('ev_f1',''),
  'log_var_det':tr.get('log_var_det',''),'log_var_pose':tr.get('log_var_pose',''),'log_var_act':tr.get('log_var_act',''),'log_var_psr':tr.get('log_var_psr',''),
  'train_loss':tr.get('total',''),'train_det_loss':tr.get('det',''),'train_pose_loss':tr.get('head_pose',''),'train_act_loss':tr.get('activity',''),'train_psr_loss':tr.get('psr',''),
  'combined_metric':m.get('combined','')})
cols=list(rows[0].keys()) if rows else []
w=csv.DictWriter(open(out,'w'),fieldnames=cols);w.writeheader();w.writerows(rows)
print(f'CSV: {out} ({len(rows)} epochs, {len(cols)} cols)')
PYEOF
)\""