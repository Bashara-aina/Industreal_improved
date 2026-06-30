# 59: Opus Master Prompt v12 — Activity Collapse Crisis [2026-06-30]

## Project Context

We are training a multi-task model for assembly verification (IndustReal dataset):
5 tasks on ConvNeXt-Tiny backbone with FPN. After 10 days of training across
6 full attempts, we have NEVER produced a validation result with act_macro_f1 > 0.002.

## Current Hardware
- GPU 0: NVIDIA RTX 3060 12GB — IDLE (never used)
- GPU 1: NVIDIA RTX 5060 Ti 16GB — 100% training
- CPU: 12 cores, HDD dataset (3,667 train frames, 1,928 val frames)
- Python 3.13, PyTorch 2.12, CUDA 13.0

## Current Training State (as of 2026-06-30 18:50 UTC)
- Stage: RF4 (all 5 heads, 50% data subset)
- Epoch: 3/23
- PID: 3369770 (just launched with classifier reinit + 20x LR)
- Speed: 1.2 batch/s, ~48 min/epoch
- Best metric (combined): 0.44499 (from stage_history, not current run)

## The Activity Head Collapse — PRIMARY PROBLEM

The activity head (8.2M params, 15.2% of total) collapses to predicting 1-4 out of
75 possible classes within 1 epoch and NEVER recovers. The gradient norm is
stationary at 0.010 regardless of:

| Intervention | Result |
|-------------|--------|
| Default config (blend=0.05, lr=1x) | act_macro_f1 = 0.0002 (random) |
| blend=0.70, lr=1x | Collapse: 4/75 classes, class 12=94% |
| blend=1.00, lr=3x, GC | Collapse: 4/75 classes, class 12=87.5% |
| blend=1.00, lr=10x, GC | Collapse: 4/75 classes, class 12=98% |
| **blend=1.00, lr=20x, GC, classifier REINIT** | **AWAITING VALIDATION** |

### Activity Head Architecture
```
proj_features(1048→512) → 2-layer ViT → TCN → LayerNorm → Linear(512→75)
  Input: c5_mod_blend(512) + det_conf(512) + pool(p4.detach()) = 1048-D
  Gradient path: CE → classifier → LayerNorm → TCN → ViT → proj_features
    → c5_mod_blend → c5_mod → FPN → backbone
  Sequence length: 1 (no staging, single frame per step)
  Feature bank in the middle: proj_feat → feature_bank(video_ids) → bank_output
```

### Comparison: Other Heads' Gradient Norms
```
PSR head:       3.180 (312x activity)
Detection head: 0.479 (47x activity)
Head pose:      0.561 (55x activity)
Pose head:      0.443 (43x activity)
Backbone:       2.366 (232x activity)
FPN:            1.154 (113x activity)
ACTIVITY HEAD:  0.010 (1x — the baseline)
```

The effective weight update at ACTIVITY_LR_MULTIPLIER=20x:
0.010 gradient × 1.0e-2 lr = 1.0e-4 per step

## Training Loss Breakdown (last completed epoch 2)
```
Train: loss=8.591  det=1.124  pose=1.699  act=0.600  psr=1.147
```

Activity loss is 0.600 — NOT zero. The loss computes and backpropagates.
The gradient just vanishes in the long chain through ViT → TCN → feature_bank.

## PSR Oscillation — SECONDARY PROBLEM
PSR head oscillates on a ~500-step cycle:
- step=2000: ALIVE[2.48]
- step=3000: DEAD[1e-06]
- step=3500: ALIVE[0.08]
- step=4000: DEAD[1e-06]

Per-component heads (h0-h10) stay ALIVE throughout. The oscillation is in the
sequence output head's total gradient norm. This may be from seq_batch/det_batch
alternation. PSR val metrics have been 0.0000 across all runs.

## Detection Progress
- det_mAP50: 0.023 (ep1) → 0.053 (ep2)
- det_mAP50_pc: 0.033 (ep1) → 0.079 (ep2)
- cls_mean: -3.5 to -5.1 (oscillating with seq/det cycle)
- DET_PROBE verdict: LOCALIZING (921-1354 preds at IoU>0.5)
- Trend: ~0.025 mAP50 improvement per epoch. Need 0.20 for RF4 gate.

## Head Pose
- forward_angular_MAE_deg: 49.99° (ep1) → 8.71° (ep2)
- Already exceeds RF10 final target (35°)
- NOTE: pose.csv forward vectors have norms 0.014-0.030 instead of 1.0 (not normalized)

## Infrastructure Failures
- 5 process deaths in last 12 hours (CUDA hangs, DataLoader deadlocks, stale heartbeats)
- 0 out of 5 validation attempts completed a full epoch
- Only 200-batch gate evals have succeeded (GATE_EVAL_MAX_BATCHES=200)
- GPU 0 (RTX 3060) completely idle — no multi-GPU config
- HDD bottleneck: 1.2 batch/s with NUM_WORKERS=0

### Active Fixes
1. Pre-val checkpoint: latest.pth saved before validation (0 epoch loss on crash)
2. NUM_WORKERS=0: no DataLoader deadlocks
3. Watchdog thread: auto-kills after 600s hang
4. GPU heartbeat with PID check: prevents false watchdog kills
5. Evaluate_all ThreadPoolExecutor: 1200s timeout (but cannot interrupt CUDA kernel)

## All Changes Made This Session (Jun 30)

### config.py
- ACTIVITY_HEAD_GRAD_CLIP: 0.3 → 1.0
- ACTIVITY_LR_MULTIPLIER: (new) 3.0 → 10.0 → 20.0
- ACTIVITY_GRAD_BLEND_RATIO: 0.70 → 1.00
- NUM_WORKERS: 4 → 0
- VAL_NUM_WORKERS: 2 → 0

### train.py
- Pre-val checkpoint: latest.pth saved after training, before validation
- activity_head split into separate param group from psr_head
- Gradient centralization for activity head (AMP + FP32 paths)
- Watchdog thread with PID verification
- CSV header fix: pd.read_csv(names=[...])
- OneCycleLR max_lr updated for new param group structure

### Checkpoint
- best.pth reinitialized: activity_head classifier LayerNorm→default, weight→xavier, bias→0

## Stage History

| Stage | Data | Epochs | Best Metric | Best det_mAP50_pc |
|-------|------|--------|-------------|:-:|
| RF1 (det only) | 20% | 10 | 0.184 | 0.184 (det) |
| RF2 (det+pose) | 35% | 2 | 0.31 | 0.31 |
| RF3 (det+pose+act) | 35% | 2 | 0.2592 | 0.31 |
| RF4 (CURRENT, all) | 50% | 3/23 | 0.445 | 0.079 |

## RF4 Gate Targets (Must pass to advance)
| Condition | Threshold | Best Ever | Probability |
|-----------|:-:|:-:|:-:|
| det_mAP50_pc ≥ 0.20 | 0.20 | 0.079 (ep2) | 30% |
| act_top1 ≥ 0.06 | 0.06 | 0.0 | <5% |
| psr_f1_at_t ≥ 0.05 | 0.05 | 0.0 | 10% |
| head_pose_MAE ≤ 65° | 65° | 8.71° | 100% |

## RF10 Final Targets
| Target | Requirement | Progress | Feasibility |
|--------|-------------|----------|:-:|
| det_mAP50_pc ≥ 0.30 | YOLOv8m=0.838 | 0.079 | <5% |
| act_top1 ≥ 0.18 | MViTv2=0.653 | ~0.0 | <1% |
| psr_f1_at_t ≥ 0.16 | B2=0.731 | 0.0 | <1% |
| head_pose_MAE ≤ 35° | Est SOTA=10° | 8.71° | 90% |

## Paper Targets (AHFE 2026 Hawaii)
Act_top1 target: 0.375 (raw multi-task, paper §3.7.1)
Current: ~0.0. Gap: 37.5 percentage points.
