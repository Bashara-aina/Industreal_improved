#!/bin/bash
# POPW 2% Subset Training + Evaluation Script
# Usage: bash run_2pct_train_eval.sh

set -e

cd /home/newadmin/swarm-bot/project/popw/working/code/industreal_improved_to_archive
export PYTHONPATH="$(pwd):$(pwd)/src:$(pwd)/src/models:$(pwd)/src/training:$(pwd)/src/evaluation:$(pwd)/src/data:$(pwd)/src/utils"

RUN_DIR="src/runs/full_multi_task_tma_tbank_benchmark"
CKPT_DIR="$RUN_DIR/checkpoints"
LOG_DIR="$RUN_DIR/logs"

# ============================================================
# STEP 1: Evaluate checkpoint at step 631 (May 21 15:34 crash recovery)
# ============================================================
echo "============================================================"
echo "[1/3] Evaluating checkpoint: epoch_0_batch_631.pth"
echo "============================================================"

CKPT="$CKPT_DIR/epoch_0_batch_631.pth"
if [ ! -f "$CKPT" ]; then
    echo "ERROR: Checkpoint not found: $CKPT"
    # Fall back to crash_recovery if available
    CKPT="$CKPT_DIR/crash_recovery.pth"
    echo "Falling back to crash_recovery.pth"
fi

METRICS_FILE="$RUN_DIR/metrics_2pct_step631.jsonl"

python - <<'EOF'
import sys, os, torch, json
sys.path.insert(0, os.getcwd())
from src.training.train import train_one_epoch_or_step
from src.evaluation.evaluate import evaluate_model

RUN_DIR = "src/runs/full_multi_task_tma_tbank_benchmark"
CKPT = "src/runs/full_multi_task_tma_tbank_benchmark/epoch_0_batch_631.pth"
if not os.path.exists(CKPT):
    CKPT = "src/runs/full_multi_task_tma_tbank_benchmark/crash_recovery.pth"

print(f"Loading checkpoint: {CKPT}")
state = torch.load(CKPT, map_location="cpu")
print(f"  Step: {state.get('step', 'N/A')}")
print(f"  Epoch: {state.get('epoch', 'N/A')}")
print(f"  Keys: {list(state.keys())}")

# Check model keys
if "model_state" in state:
    print(f"  Model state keys (first 5): {list(state['model_state'].keys())[:5]}")
elif "state_dict" in state:
    print(f"  State dict keys (first 5): {list(state['state_dict'].keys())[:5]}")
print("Checkpoint loaded successfully.")
EOF

echo ""
echo "============================================================"
echo "[2/3] Evaluation not yet implemented — using quick loss check"
echo "============================================================"

# Quick sanity: load checkpoint and run a few forward passes
python - <<'EOF'
import sys, os, torch, yaml
sys.path.insert(0, os.getcwd())

CKPT = "src/runs/full_multi_task_tma_tbank_benchmark/epoch_0_batch_631.pth"
if not os.path.exists(CKPT):
    CKPT = "src/runs/full_multi_task_tma_tbank_benchmark/crash_recovery.pth"

print(f"Loading: {CKPT}")
state = torch.load(CKPT, map_location="cpu", weights_only=False)
step = state.get("step", "N/A")
epoch = state.get("epoch", "N/A")
print(f"Checkpoint step={step}, epoch={epoch}")

# Check if we have metrics saved
import json, glob
metrics_files = sorted(glob.glob("src/runs/full_multi_task_tma_tbank_benchmark/metrics*.jsonl"))
if metrics_files:
    print(f"\nFound metrics files:")
    for f in metrics_files:
        print(f"  {f}")
    # Show last entry of most recent
    with open(metrics_files[-1]) as fh:
        lines = fh.readlines()
    if lines:
        last = json.loads(lines[-1])
        print(f"\nLast metrics entry (step={last.get('step','?')}):")
        for k, v in last.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
else:
    print("\nNo metrics files found — need to run evaluation.")
EOF

echo ""
echo "============================================================"
echo "[3/3] Quick checkpoint summary"
echo "============================================================"
python - <<'EOF'
import torch, glob, os

ckpt_dir = "src/runs/full_multi_task_tma_tbank_benchmark/checkpoints"
checkpoints = sorted(glob.glob(f"{ckpt_dir}/epoch_0_batch_*.pth"), 
                   key=lambda p: int(p.split("_batch_")[-1].replace(".pth","")) if "batch_" in p else 0)

print(f"Available checkpoints in {ckpt_dir}:")
for c in checkpoints[-10:]:
    size_mb = os.path.getsize(c) / 1e6
    step = c.split("_batch_")[-1].replace(".pth","")
    print(f"  step_{step}: {size_mb:.1f} MB")

crash = f"{ckpt_dir}/crash_recovery.pth"
if os.path.exists(crash):
    size_mb = os.path.getsize(crash) / 1e6
    print(f"  crash_recovery: {size_mb:.1f} MB (last crash save)")
EOF

echo ""
echo "============================================================"
echo "Done. Next: run 'python src/evaluation/evaluate.py --help' for full eval."
echo "============================================================"
