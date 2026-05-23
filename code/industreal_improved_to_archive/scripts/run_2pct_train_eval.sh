#!/bin/bash
# ============================================================
# POPW 2% Subset — Train to step 200 → Evaluate → Metrics
# Single script. Run in terminal:
#   bash run_2pct_train_eval.sh
# ============================================================
set -e
WDIR="/home/newadmin/swarm-bot/project/popw/working/code/industreal_improved_to_archive"
cd "$WDIR"

# Kill any lingering GPU processes
pkill -f "train.py" 2>/dev/null || true
sleep 2
nvidia-smi --gpu-reset 2>/dev/null || true
sleep 3

# Export PYTHONPATH
export PYTHONPATH="$WDIR:$WDIR/src:$WDIR/src/models:$WDIR/src/training:$WDIR/src/evaluation:$WDIR/src/data:$WDIR/src/utils"
export TRAIN_MAX_STEPS=200        # ← key: enforces 200-step early stop
export CFG_ARGS="--subset-ratio 0.02 --no-staged-training --seed 42"

LOG="$WDIR/src/runs/full_multi_task_tma_tbank_benchmark/logs/train_2pct.log"
mkdir -p "$(dirname "$LOG")"

echo "============================================================"
echo "POPW 2% Subset Training"
echo "  TRAIN_MAX_STEPS=$TRAIN_MAX_STEPS"
echo "  Log: $LOG"
echo "============================================================"

# Run training
python src/training/train.py $CFG_ARGS \
    2>&1 | tee "$LOG" &

TPID=$!
echo "Training PID: $TPID"

# Wait (max 90 min for 200 steps at ~1.5it/s ≈ ~2.2 min)
echo "Waiting for training to finish (or reach step 200)..."
wait $TPID
TRAIN_EXIT=$?
echo "Training exit code: $TRAIN_EXIT"

# ── Find best checkpoint ───────────────────────────────────
CKPT_DIR="$WDIR/src/runs/full_multi_task_tma_tbank_benchmark/checkpoints"
# epoch_0_batch_*.pth files, highest step first
CKPT=$(ls -t "$CKPT_DIR"/epoch_0_batch_[0-9]*.pth 2>/dev/null | head -1)
if [ -z "$CKPT" ]; then
    CKPT="$CKPT_DIR/crash_recovery.pth"
fi
echo ""
echo "Using checkpoint: $CKPT"
STEP=$(echo "$CKPT" | grep -o 'batch_[0-9]*' | grep -o '[0-9]*')
echo "Checkpoint step: $STEP"

# ── Evaluate ─────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "Running evaluation..."
echo "============================================================"

python - <<PYEOF
import sys, os, torch, json, traceback
sys.path.insert(0, os.getcwd())

ckpt_path = "$CKPT"
if not os.path.exists(ckpt_path):
    print(f"ERROR: {ckpt_path} not found"); sys.exit(1)

print(f"Loading: {ckpt_path}")
state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
step = state.get("step", state.get("global_step", "N/A"))
epoch = state.get("epoch", "N/A")
print(f"  step={step}, epoch={epoch}")

# ── Quick val loss (50 batches) ──────────────────────────────
try:
    from src.models.model import POPWMultiTaskModel
    from src.data.industreal_dataset import IndustRealMultiTaskDataset, collate_fn
    from src.training.losses import MultiTaskLoss
    from torch.utils.data import DataLoader
    import torch.nn as nn

    device = torch.device("cuda")

    model = POPWMultiTaskModel()
    model = model.to(device)
    sd = state.get("model")
    result = model.load_state_dict(sd, strict=False)
    if result.missing_keys:
        print(f"  Missing keys (expected): {result.missing_keys}")
    if result.unexpected_keys:
        print(f"  Unexpected keys: {result.unexpected_keys}")
    model = model.eval()
    print(f"Model loaded, device={device}")

    # Build criterion (used for loss computation)
    criterion = MultiTaskLoss(
        num_classes_act=75,
        num_psr_components=11,
        train_det=True,
        train_pose=True,
        train_act=True,
        train_psr=True,
        use_kendall=True,
    ).to(device)

    print("Computing validation loss (50 batches)...")
    val_ds = IndustRealMultiTaskDataset(
        split='val',
        img_size=(720, 1280),
        augment=False,
        seed=42,
        max_recordings=None,
    )
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0, pin_memory=True, collate_fn=collate_fn)

    totals = {"loss": 0.0, "det": 0.0, "pose": 0.0, "act": 0.0, "psr": 0.0}
    n = 0
    with torch.no_grad():
        for batch in val_loader:
            try:
                # collate_fn returns (images, targets) tuple — unpack correctly
                images, targets = batch
                frames = images.to(device)
                targets = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in targets.items()}
                outputs = model(frames)
                loss_dict, _ = criterion(outputs, targets)

                for k in totals:
                    v = loss_dict.get(k, torch.tensor(0.0))
                    totals[k] += v.item() if hasattr(v, "item") else float(v)
                n += 1
                if n >= 50:
                    break
            except Exception as e:
                print(f"  Batch {n} error: {e}")
                traceback.print_exc()
                continue

    if n > 0:
        metrics = {
            "checkpoint_step": int(step) if isinstance(step, (int,float)) else -1,
            "checkpoint_epoch": str(epoch),
            "val_loss": totals["loss"]/n,
            "val_det": totals["det"]/n,
            "val_pose": totals["pose"]/n,
            "val_act": totals["act"]/n,
            "val_psr": totals["psr"]/n,
            "n_batches": n,
        }
        out = "/home/newadmin/swarm-bot/project/popw/working/code/industreal_improved_to_archive/src/runs/full_multi_task_tma_tbank_benchmark/metrics_2pct_eval.json"
        with open(out, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nSaved: {out}")
        print("\n=== VAL METRICS ===")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    else:
        print("No batches processed.")

except Exception as e:
    print(f"Error during evaluation: {e}")
    traceback.print_exc()
    # Print checkpoint info at least
    print("\nCheckpoint info:")
    print(f"  step: {step}")
    print(f"  epoch: {epoch}")
    print(f"  keys: {list(state.keys())[:10]}")
PYEOF

echo ""
echo "============================================================"
echo "Done. Metrics → src/runs/.../metrics_2pct_eval.json"
echo "============================================================"