# Training Scripts

Main training scripts for the multi-task IndustReal model.

## Main Scripts

### `train_mtl_mvit.py`
**Main MTL training script.** Trains the full MTLMViTModel with all 4 heads
(detection, activity, pose, PSR) on full multimodal data (5 modalities).

Key flags:
- `--num-anchors {8,16}` — Anchor configuration
- `--use-tal` — Enable TAL alignment (TOOD-style)
- `--use-llrd` — Layer-wise learning rate decay
- `--use-uw-so` — Uncertainty-weighted loss (Kendall)
- `--use-logit-bias-update` — Per-class bias update (default OFF)
- `--use-class-balanced-sampling` — Class-balanced batch sampling
- `--use-mosaic` / `--use-copy-paste` — Augmentations

### `train_v8_multitask.py`
V8 multi-task pipeline (alternative).

### `train_st_act.py` / `train_st.py`
Single-task baselines (training each head independently).

### `decoupled_act_retrain.py`
Decoupled activity classifier retrain.

### `train_psr_repair_wrapper.py`
Wrapper for PSR head repair training with bf16 mixed precision.

### `st_act_standalone.py`
Single-task activity training without DataLoader.

## Launcher Shell Scripts (`.sh`)

Various launcher scripts for different training configs:
- `train_v5b_fresh.sh`, `train_v5c_fresh_f1fix2.sh`, `train_v5_multitask.sh`
- `train_singletask_*.sh` — Single-task launchers
- `train_psr_head_repair.sh`, `train_psr_repair_v3.sh`, `train_psr_repair_v4.sh`
- `train_activity_tcn.sh` — Activity-only TCN training
- `train_finetune_backbone.sh` — Backbone fine-tuning
- `train_mvit_finetune.sh` — MViT fine-tuning
- `train_isolated.sh` — Isolated head training
- `train_st_baselines.sh` — Single-task baselines

## Run Scripts (`.sh`)

Various run scripts for executing experiments:
- `run_2pct_eval_only.sh`, `run_2pct_train_eval.sh` — 2% dataset experiments
- `run_ablation_suite.sh` — Ablation studies
- `run_d1_yolov8m_eval.sh`, `run_d3_full_eval.sh`, `run_d4_yolov8m_psr.sh`
  — YOLOv8 baseline evaluations
- `run_eval_*.sh` — Eval launchers
- `run_fresh_smoke_patched.sh` — Smoke tests
- `run_kurin_baseline.sh` — Kurin baseline
- `run_overfit_*.sh` — Overfit probes
- `run_psr_kendall_fixed.sh` — PSR + Kendall
- `run_recovery_retrain_25pct.sh` — Recovery retraining
- `run_reinit_*.sh` — Re-init experiments
- `run_rf4_probe.sh` — RF4 probe
- `run_smoke_fp32*.sh` — FP32 smoke tests
- `run_st_baselines.sh` — Single-task baselines

## Usage

```bash
# Main MTL training (resumes from checkpoint)
python scripts/train/train_mtl_mvit.py \
    --resume runs/mtl_v3.7/checkpoints/phase2_e0_b18000.pth \
    --phase1-epochs 0 \
    --phase2-epochs 1 \
    --batch-size 2 \
    --use-llrd \
    --use-uw-so \
    --num-anchors 16 \
    --output-dir runs/mtl_v3.8

# OR use a launcher
bash scripts/train/train_v5_multitask.sh
```