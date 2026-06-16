# R3 Restart Plan — IndustReal Training Pipeline

## Summary

This document consolidates ALL fixes applied through fix1-fix19 into a single restart plan for the R3 training run. The model uses `crash_recovery.pth` (epoch ~49 checkpoint from the original collapsed run) with `--reinit-heads` to reinitialize the three collapsed heads (detection, activity, PSR) + FPN while preserving the backbone and pose heads.

---

## 1. ALL Fixes Applied (fix1-fix19)

### Fixes in `src/config.py`

| Area | Setting | Old Value | New Value | Purpose |
|------|---------|-----------|-----------|---------|
| **PSR Sequence Mode** | `USE_PSR_SEQUENCE_MODE` | False | True | Enable temporal transformer with T=2 for PSR gradient flow |
| | `PSR_SEQUENCE_LENGTH` | 4 | 2 | Fits RTX 3060 12GB with gradient checkpointing |
| | `PSR_SEQ_EVERY_N_BATCHES` | 8 | 6 | More sequence batches for temporal learning |
| | `PSR_SEQ_LOSS_SCALE` | 1.0 | 1.5 | Amplify Transformer gradient signal |
| **Gradient Checkpointing** | `USE_BACKBONE_CHECKPOINT` | False | True | ~50% activation memory reduction during backprop |
| **Liveness Monitoring** | `LIVENESS_EVERY` | 200 | 100 | 2x frequency — tighter monitoring |
| | `LIVENESS_GRAD_EVERY` | 200 | 200 | Kept at 200 for grad-norm liveness |
| **Activity Dominance Control** | `ACTIVITY_HEAD_GRAD_CLIP` | 0.5 | 0.1 | Prevent activity dominating backbone gradients |
| | `ACTIVITY_LOSS_WEIGHT` | 1.0 | 0.2 | Down-weight activity 80% before Kendall |
| **Per-Task Kendall Bounds** | `KENDALL_LOG_VAR_MIN_ACT` | 0.0 | -0.5 | Activity can precision-boost moderately |
| | `KENDALL_LOG_VAR_MAX_PSR` | 2.0 | 0.0 | PSR can't be suppressed (min prec=1.0) |
| | `KENDALL_LOG_VAR_MAX_POSE` | 2.0 | 3.0 | Pose can be suppressed (was dominating) |
| **PSR Step Warmup** | `PSR_WARMUP_STEPS` | 6000 | 4000 | Reduced — sequence mode provides temporal signal |
| | `PSR_WARMUP_INIT_MULT` | 3.0 | 2.0 | Gentler warmup start |
| | `PSR_WEIGHT` | 60 | 30 | Reduced — sequence mode reduces need for aggressive doubling |
| | `PSR_FOCAL_GAMMA` | 1.0 | 1.5 | Harder negative mining with sequence mode |
| | `USE_PSR_TRANSITION` | False | True (paper_run) | Gaussian-smeared transition targets |
| **Detection Eval** | `DET_EVAL_SCORE_THRESH` | 0.02 | 0.001 | YOLOv8 comparability |
| **Mixed Precision** | `MIXED_PRECISION` | False | True | Model is stable — 0 GRAD_NAN events over 1100+ steps |
| **Head Pose** | `USE_GEO_HEAD_POSE` | False | True | 6D rotation (Zhou) — strictly better |
| **Pose Loss Weight** | `POSE_LOSS_WEIGHT` | 0.02 | 0.01 | Pose at 4.46 (strongest), less weight needed |
| **PSR Sensitivity** | `PSR_SENSITIVITY_WEIGHT` | 0.0 | 0.01 | NaN root cause fixed (std correction=0) |
| **Per-Component PSR Weights** | `PSR_COMP_WEIGHTS` | Uniform | [1.0, 1.21, ...4.61] | Inverse prevalence weighting for rare steps |

### Fixes in `src/training/train.py`

| Area | Fix | Detail |
|------|-----|--------|
| **Reinit Heads** | `_reinit_dead_heads()` | Re-initializes det/act/psr heads + FPN (8 Conv2d modules) from Kaiming/Xavier priors |
| | Det cls_score | pi=0.1 prior (was 0.05) — sigmoid starts near 0.1 |
| | PSR output heads | **std=0.02, zero bias** (was Xavier uniform + bias=-0.2). Moderate init prevents sigmoid saturation → (1-p_t)^gamma ~0 under focal loss |
| | PSR transformer | **NEW: Xavier-uniform reinit** — original ckpt weights produced extreme outputs (std~86) with mismatched per_frame_mlp/FPN after reinit |
| | Activity head | Xavier-uniform ViT blocks, std=0.02 classifiers, bias=-0.5 |
| | FPN | Kaiming-uniform a=1 + zero bias across all 8 Conv2d modules |
| **EMA Shadow** | Re-anchor | After reinit, EMA shadow for det/act/psr/fpn params is reset to fresh weights |
| **Kendall Reset** | log_var reset | All 4 Kendall log_vars set to 0.0 (neutral) after --reinit-heads |
| **Det Warmup** | HARDCODED in train.py | 50 zero-grad + 200-step linear ramp (250 total). `DET_WARMUP_STEPS` config never wired — see config.py:53 comment |
| **PSR Warmup** | 200-step 2x grad | PSR output head gets 2x gradient multiplier for first 200 steps after reinit |
| **Mixed Precision** | Amp GradScaler | Re-enabled with NaN guards, isfinite checks, RC-29 telemetry |
| **GPU Memory** | expandable_segments | PYTORCH_ALLOC_CONF=expandable_segments:True — prevents fragmentation OOM |
| **Config Snapshot** | config.py copy | Saves snapshot of config to checkpoint directory |

### Fixes in `src/training/losses.py`

| Area | Fix | Detail |
|------|-----|--------|
| **Kendall Bounds** | Per-task clamp | Activity: clamp(-0.5, 2.0), PSR: clamp(-4.0, 0.0), Pose: clamp(-4.0, 3.0) |
| **Activity Weight** | `ACTIVITY_LOSS_WEIGHT` | Applied before Kendall precision: `prec_act * (loss_act * _act_w) + lv_act` |
| **PSR Step Warmup** | Precision multiplier | Decays from `PSR_WARMUP_INIT_MULT=2.0` to 1.0 over `PSR_WARMUP_STEPS=4000` steps |
| **PSR Per-Component Weights** | `comp_weights` | `PSR_COMP_WEIGHTS` normalize per-component loss by mean weight |
| **Liveness Probe** | Frequency doubled | Every 100 steps (was 200), includes per-component PSR breakdown + GPU memory |
| **Per-Component PSR Loss** | `_psr_per_component` | Logged in liveness — min/max/mean across 11 components |
| **NaN Guards** | PSR sensitivity | `std(correction=0)` verified safe — no NaN events in 1100+ steps |

---

## 2. Current Training Status

**As of 2026-06-15 08:47 UTC:**

- **Process**: Training IS running (PID 3482339)
- **Command**: `python train.py --resume crash_recovery.pth --reinit-heads`
- **Config preset**: `paper_run` with ALL fixes applied
- **Checkpoint**: `src/runs/full_multi_task_tma_tbank/checkpoints/crash_recovery.pth` (epoch ~49)
- **Log**: `src/runs/full_multi_task_tma_tbank/logs/train.log` (launched via `scripts/launch_r25_fix.sh`)
- **Checkpoints from original run**: `best.pth`, `latest.pth`, `crash_recovery.pth`, `best_pre_restart.pth`, `latest_pre_restart.pth`

### Pre-Restart R2.5 Run State

The original run used `scripts/launch_r25_fix.sh` which launches to `runs/paper_run_r25_fix_20260615.log`. The current running process appears to be a direct launch using the `full_multi_task_tma_tbank` checkpoint directory.

---

## 3. What to Monitor During R3

### A. Immediate (first 200 steps)

| Signal | What to Look For | Action if Bad |
|--------|------------------|---------------|
| Liveness output | ALL 4 heads show "ALIVE" (det>0.01, act>0.001, psr>0.001, hp>0.001) | Kill and debug dead head |
| Grad norms | `GRAD_NAN=0`, grad norms > 1e-6 for all heads | Check head gradient flow |
| Kendall log_vars | All near 0.0 (neutral — just reset); no divergence | Watch for one log_var going to -4 or +2 |
| PSR per-component | PSR loss > 1e-3 (not DEAD at 1e-4 floor); ideally 0.01-0.1 | PSR head not learning — check seq batch count |

### B. Short-term (steps 200-2000)

| Signal | What to Look For | Action if Bad |
|--------|------------------|---------------|
| Activity loss | Trending down from ~4-5 toward stable baseline | Activity head stuck (stuck at 2/75 classes) |
| Detection mAP | Should exceed 0.001 by epoch 1-2 | Det head still collapsed |
| PSR loss | Above 0.001 floor consistently | Temporal transformer not engaging |
| GPU memory | <11.5 GiB allocated (stay under 12GB) | OOM — reduce batch size or seq length |

### C. Long-term (epochs 5-100)

| Signal | What to Look For | Action if Bad |
|--------|------------------|---------------|
| Activity Top-1 | >10% after epoch 5, trending toward 40-50% | Activity head needs LDAM re-enable |
| Detection mAP@0.5 | >0.02 after epoch 5 | Detection not learning |
| PSR F1 | >0.2 after epoch 10 | PSR needs more sequence batches |
| Kendall balance | All log_vars in [-1, 1] range | Kendall divergence (one task dominating) |

### D. Infrastructure

- **Monitor**: `bash /home/newadmin/swarm-bot/scripts/monitor_r25_training.sh` — auto-detects PID + latest log
- **Tmux monitor**: `tmux attach -t r3_monitor` (auto-launched by restart script)
- **Kill**: `bash scripts/kill_training.sh` — SIGTERM first, SIGKILL after 30s timeout, reports last 20 log lines
- **Log tail**: `tail -f src/runs/full_multi_task_tma_tbank/logs/train.log`
- **GPU check**: `watch -n 5 nvidia-smi`
- **Optimizer check**: `grep "optimizer windows" src/runs/full_multi_task_tma_tbank/logs/train.log` — verify optimizer.step() is committing (not silently skipped by AMP GradScaler)

---

## 4. Success Criteria per Head

### Detection Head (ASD)

| Metric | Threshold | Notes |
|--------|-----------|-------|
| mAP@0.5 | > 0.05 at epoch 5 | Bootstrap target from `recovery_det_only` logic |
| mAP@0.5 | > 0.10 at epoch 20 | Acceptable production quality |
| Score separation | sigmoid std > 0.05 across predictions | Indicates head is not collapsed |
| Positive match rate | > 0.1% of anchors matched to GT | RetinaNet healthy matching |

### Activity Head (AR)

| Metric | Threshold | Notes |
|--------|-----------|-------|
| Top-1 accuracy | > 10% at epoch 5 | Beats random (1/75 = 1.3%) |
| Top-1 accuracy | > 30% at epoch 20 | Trending toward paper target |
| Effective classes | > 10 classes with non-zero predictions | Avoid 1-2 class collapse |
| Activity loss | 1.0-5.0 (not >40 or <0.1) | Smooth cap at 80.0; healthy range |

### PSR Head

| Metric | Threshold | Notes |
|--------|-----------|-------|
| PSR per-frame loss | > 0.001 (not at 1e-4 floor) | Indicates head is alive |
| PSR F1 (macro) | > 0.05 at epoch 10 | Above random baseline |
| Logit std | > 0.01 across per-component logits | Not collapsed to constant |
| Temporal smooth loss | Decreasing over time | Transformer learning transitions |

### Head Pose (9-DoF)

| Metric | Threshold | Notes |
|--------|-----------|-------|
| Position MAE | < 0.5 (in HEAD_POSE_POS_SCALE=100 units) | Well below naive mean predictor |
| Direction MAE | < 30 degrees | Geometry-aware head (6D rotation) |
| Loss | 0.01-1.0 | Healthy range |

---

## 5. Files & Scripts Reference

| File | Purpose |
|------|---------|
| `scripts/launch_r25_fix.sh` | Original launch script for R2.5 fix run |
| `scripts/restart_r3_training.sh` | R3 restart script — kill + backup + launch + monitor |
| `scripts/kill_training.sh` | Graceful kill script (SIGTERM > 30s wait > SIGKILL) |
| `/home/newadmin/swarm-bot/scripts/monitor_r25_training.sh` | Training monitor (auto-detect PID + log) |
| `src/config.py` | All training hyperparameters (paper_run preset) |
| `src/training/train.py` | Training loop, reinit logic, optimizer setup |
| `src/training/losses.py` | MultiTaskLoss with Kendall weighting, per-task bounds |
| `src/models/model.py` | Full model architecture (ConvNeXt + FPN + 5 heads) |

### Checkpoint Reference

| File | Size | Origin |
|------|------|--------|
| `crash_recovery.pth` | 805 MB | Epoch ~49 from original run -- reinit heads starting point |
| `best.pth` | 492 MB | Best checkpoint from original run (pre-collapse) |
| `latest.pth` | 514 MB | Latest checkpoint before reinit |
| `best_pre_restart.pth` | 492 MB | Copy of best.pth before R2.5 restart |

---

## 6. Quick Reference — Key Config Values

```python
# PSR Sequence Mode
USE_PSR_SEQUENCE_MODE = True      # [FIX 2026-06-15] Temporal context for PSR
PSR_SEQUENCE_LENGTH = 2            # T=2 with gradient checkpointing
PSR_SEQ_EVERY_N_BATCHES = 6        # 16.7% of batches are sequence

# Activity Dominance Control
ACTIVITY_HEAD_GRAD_CLIP = 0.1      # Prevent activity dominating backbone
ACTIVITY_LOSS_WEIGHT = 0.2         # Down-weight 80% before Kendall

# Kendall Bounds
KENDALL_LOG_VAR_MIN_ACT = -0.5     # Activity can boost precision moderately
KENDALL_LOG_VAR_MAX_PSR = 0.0      # PSR can't be suppressed
KENDALL_LOG_VAR_MAX_POSE = 3.0     # Pose can be suppressed

# PSR Warmup
PSR_WARMUP_STEPS = 4000            # Precision ramp duration
PSR_WARMUP_INIT_MULT = 2.0         # Start precision multiplier
PSR_WEIGHT = 30.0                  # PSR weight before Kendall
```
