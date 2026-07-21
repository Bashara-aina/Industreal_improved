#!/bin/bash
# =============================================================================
# RECOVERY RETRAIN — 25% subset, 3 epochs, FP32, full collapse-recovery guards
# (opus 2026-06-10, RC-13/14/15/16/17/19/20/21/22 + P1-P10 + R1+R2 applied)
#
# Why this differs from run_reinit_fp32_bs2.sh (5% / 1 epoch / collapsed further):
#   1. SUBSET 0.25 not 0.05. Opus: "data breadth is the binding constraint, not
#      step count." 0.05 mostly re-fits 12 of 75 activity classes and a handful
#      of GT boxes. 0.25 ≈ 18-19 recs × ~700 frames ≈ 12-13K frames.
#   2. 3 EPOCHS not 1. Gives Adam enough horizon to escape the small-magnitude
#      init region the heads start in.
#   3. --reset-optimizer-for-reinit (R1). The 5%/bs=2 retrain ran with stale
#      Adam m/v from the collapsed-43 ckpt applied to fresh reinit weights.
#      Step-0 cls loss = 197,844; by end of epoch, act→1 class, PSR→1 pattern,
#      det→flat (std=0). R1 zeros m/v for reinit'd head params.
#   4. --reset-kendall-log-vars (R2). Epoch-43 log_vars were learned against a
#      collapsed state. R2 forces them back to neutral (det=0, head_pose=-1,
#      act=0, psr=0) so the loss-balancing starts fresh.
#   5. --max-epochs 46. Resumes at epoch 43, runs 3 more = epoch 44, 45, 46.
#   6. FP32 (--no-amp). AMP was off in the 5%/bs=2 retrain too; this is the
#      stable path. ~7.5 h/epoch at 0.25/bs=2 on RTX 3060 → ~22 h total.
#   7. USE_EMA=False, USE_MIXUP=False, CUTMIX_ALPHA=0.0 already in config.py.
#   8. Resume source: best.pth (epoch 43, 19:42 timestamp — bit-identical to
#      crash_recovery.pth, so either works). Using best.pth since it's the
#      "official" run checkpoint.
#
# Sanity checks before launching (all will print to log):
#   - epoch == 43 in source ckpt
#   - post-reinit: 0 NaN/Inf params
#   - post-reset: optimizer m/v is zero for 169 head tensors
#   - 0/100 NaN gradients in first 100 steps (FP32 proven stable)
#   - loss should start ~5-20, NOT ~500
#
# Run:
#   cd /home/newadmin/swarm-bot/project/popw/working/code/industreal_improved
#   bash run_recovery_retrain_25pct.sh
# =============================================================================
set -e

PROJ="/home/newadmin/swarm-bot/project/popw/working/code/industreal_improved"
RUN_NAME="recovery_25pct_3ep_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$PROJ/src/runs/$RUN_NAME/logs"
CKPT_DIR="$PROJ/src/runs/$RUN_NAME/checkpoints"
mkdir -p "$LOG_DIR" "$CKPT_DIR"

SRC_CKPT="$PROJ/src/runs/full_multi_task_tma_tbank_benchmark/checkpoints/best.pth"
LOG="$LOG_DIR/train.log"

# Kill any lingering GPU processes from prior runs
pkill -f "training/train.py" 2>/dev/null || true
sleep 2
nvidia-smi --gpu-reset 2>/dev/null || true
sleep 3

# PYTHONPATH (consistent with run_reinit_fp32_bs2.sh)
export PYTHONPATH="$PROJ:$PROJ/src:$PROJ/src/models:$PROJ/src/training:$PROJ/src/evaluation:$PROJ/src/data:$PROJ/src/utils"
export SUBSET_RATIO=0.25
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export MALLOC_ARENA_MAX=4
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRAIN_MAX_STEPS=0
export EVAL_MAX_BATCHES=20

echo "============================================================"
echo "RECOVERY RETRAIN (opus 2026-06-10, P1-P10 + R1+R2 applied)"
echo "  Subset  : 0.25 (~18-19 recs, ~12-13K frames)"
echo "  Epochs  : 3 (epoch 44, 45, 46)"
echo "  BS      : 2 (FP32, seq-mode PSR T=4 fits in 11.4 GB)"
echo "  Reinit  : det/act/psr with documented priors"
echo "  Guards  : R1 (zero Adam m/v), R2 (reset Kendall log_vars)"
echo "  Source  : $SRC_CKPT"
echo "  Log     : $LOG"
echo "  RUN     : $RUN_NAME"
echo "============================================================"
echo ""
echo "Pre-flight: source ckpt info"
python - <<EOF 2>&1 | tee -a "$LOG"
import torch
ck = torch.load("$SRC_CKPT", map_location='cpu', weights_only=False)
print(f"  epoch={ck.get('epoch')}  step={ck.get('step')}  best_metric={ck.get('best_metric', 0.0):.4f}")
nan_params = [n for n, p in ck.get('model', {}).items() if torch.isnan(p).any() or torch.isinf(p).any()]
print(f"  NaN/Inf params in source model state: {len(nan_params)}")
if nan_params[:5]:
    print(f"  first 5: {nan_params[:5]}")
EOF

cd "$PROJ/src"

python -u training/train.py \
    --resume "$SRC_CKPT" \
    --reinit-heads \
    --reset-optimizer-for-reinit \
    --reset-kendall-log-vars \
    --no-staged-training \
    --no-amp \
    --max-epochs 46 \
    --subset-ratio 0.25 \
    --num-workers 0 \
    --batch-size 2 \
    --seed 42 \
    2>&1 | tee "$LOG"

EXIT=${PIPESTATUS[0]}
echo ""
echo "============================================================"
if [ $EXIT -eq 0 ]; then
    echo "RESULT: SUCCESS — recovery retrain completed"
    echo "  Checkpoints: $CKPT_DIR"
    echo "  Eval: bash run_eval_post_retrain_fp32.sh $CKPT_DIR"
else
    echo "RESULT: FAILED with exit code $EXIT"
    echo "  Check tail of $LOG for crash trace"
fi
echo "Log: $LOG"
echo "============================================================"
exit $EXIT
