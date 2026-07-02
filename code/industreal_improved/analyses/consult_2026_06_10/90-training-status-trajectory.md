# 01 — Training Status & Trajectory

## Current Run State

- **PID:** 554647 (alive, running ~21h)
- **Epoch:** 2/99 (batch ~830/4387, 19%)
- **Config:** stage_rf4 preset, BATCH_SIZE=6, GRAD_ACCUM_STEPS=8, EFFECTIVE_BATCH=48
- **GPU:** NVIDIA RTX 5060 Ti (16GB), VRAM: 1.36GB alloc / 7.72GB reserved
- **Compute:** ~149% CPU util, RTX 5060 Ti at ~95% GPU util
- **Process:** RNl (real-time priority, nice +10)

## Critical Concerns

1. **LR scaling mismatch** — EFFECTIVE_BATCH=48 vs the paper's 16 (3x larger). Per the linear scaling rule (Goyal et al. 2017), LR should scale proportionally with batch size when all else is equal. The LR was not adjusted for the 3x batch increase. This means the effective per-sample LR is 1/3 of the paper's intended value, which may explain slow convergence across all heads.

2. **OneCycleLR pct_start discrepancy** — The canonical scheduler builder in `src/training/optimizer.py:58` uses `pct_start=0.3` (peak LR at epoch ~31), but `src/training/train.py:3708` builds the scheduler inline with `pct_start=0.1` (peak at epoch ~10). Most documentation files cite 0.1, but the "official" scheduler module has 0.3. Verify which code path is actually active — they produce peak LR at epochs 31 vs 10, a 3x difference in warmup duration.

3. **PSR warmup is very brief** — PSR_WARMUP_STEPS=500 is only ~12-20 minutes of training (at ~0.7 batch/s). This is 500 optimizer steps total, after which the PSR precision multiplier reaches 1.0 and full loss weight applies. For a head being trained from scratch, this is a very short adaptation period.

4. **Dual GPU resource underutilization** — GPU 0 (RTX 3060 12GB) is idle at ~400MB used for display + VSCode. It could run subprocess evaluation or host a parallel validation loop. Only GPU 1 (RTX 5060 Ti 16GB) is training.

5. **Detection confidence stuck at bias init** — DET_PROBE shows score_p50=0.036, which is essentially unchanged from the bias initialization of -3.4 (~sigmoid(-3.4)=0.033) after 5000+ batches. The classifier sigmoid outputs have not shifted above the bias floor, indicating the classification subnet may not be learning discriminative scores despite producing good bounding boxes (bestIoU_max=0.85-0.97).

6. **Pose loss magnitude-matching concern** — The pose loss dropping to ~0.1 may reflect magnitude-matching rather than directional convergence. The GT forward vectors in pose.csv have norms of 0.014-0.030 (not unit vectors). The model could achieve MSE=0.1 by simply outputting near-zero vectors that match the GT magnitude, without learning correct orientation. The eval normalizes before computing angular MAE, so angular MAE may degrade once this normalization is applied at data-load time.

7. **Activity collapse improving but not resolved** — The activity head shows a clear improvement trajectory (1 class predicted at epoch 0 vs 3 at epoch 2 vs 5+ in later epoch 2 samples), and entropy has risen from 0 to 1.036 nats. However, 5+ out of 69 classes is still deep in collapse territory (gate requires >=10). The activity ramp-up (ACT_RAMP_EPOCHS=5) only recently ended, so genuine convergence may begin at epoch 3+.

## Hardware Context

The system has two GPUs:
- GPU 0: RTX 3060 12GB (used for display + VSCode, ~400MB used) — **IDLE, could run subprocess eval**
- GPU 1: RTX 5060 Ti 16GB (training, primary CUDA device)

Overcommit memory was set to mode 1 (permissive) because Chrome's VSZ allocations consumed the strict mode budget. Commit charge is now 53GB vs 110GB limit.

The idle RTX 3060 presents an opportunity: it could run a parallel evaluation loop or validation process without interfering with training on the 5060 Ti. This would solve the current inability to run validation without hanging the training process.

## Complete Training History (RF4)

### RF4 Run History

| Run | Epochs Reached | Batches | Killed By | Notes |
|---|---|---|---|---|
| rf4_run_20260702_010027 | 0 | 0 | Unknown | Crash at launch |
| rf4_run_20260702_010938 | 0 | 410/6580 | Unknown | Died early epoch 0 |
| rf4_run_20260702_081721 | 0 | 4990/13161 | Unknown | Batch=2, furthest epoch 0 |
| rf4_run_20260702_100557 | 0 | 980/13161 | Unknown | |
| rf4_run_20260702_103014 | 0 | 330/13161 | Unknown | |
| rf4_run_20260702_104019 | 0 | 0 | Unknown | |
| rf4_run_20260702_104258 | 0 | 1450/6580 | Unknown | |
| rf4_run_20260702_112450 | 1 | 310/13161 | Watchdog (1216s stale) | Batch=2, only one that hit epoch 1 |
| rf4_batch6_clean | 0 | 208/4387 | ENOMEM | overcommit=2 killed alloc |
| rf4_batch6_131538 | 0 (completed) | 4387/4387 | CUDA hang in val | Epoch 0 val hung after activity |
| rf4_batch6_135539 | 0->1->2->death | Multiple | Watchdog (post-eval stale) + hangs | 3 deaths total |
| **rf4_batch6_204203 (current)** | **2/99** | **~830/4387 (19%)** | **STILL ALIVE** | **Most successful RF4 run ever — first to reach epoch 2 with all heads alive** |

### Key Timeline

- **2026-07-02 13:09** — First clean epoch-0 launch (batch=2, RTX 3060)
- **2026-07-02 13:40** — Restart on RTX 5060 Ti with batch=6
- **2026-07-02 15:42** — Epoch 0 complete! First RF4 run to finish an epoch
- Three deaths from watchdog and CUDA hangs during eval
- **2026-07-02 20:42** — Current run resumed from epoch 2 checkpoint
- **2026-07-02 21:39** — Reached epoch 2 batch 830 (19%), still alive and stable

## Loss Trajectory (Epoch 0 -> 1 -> 2)

### Epoch 0 Losses (every 500 batches)

| Batch | Total | Det(c) | Pose | Act | PSR |
|---|---|---|---|---|---|
| 500 | 21.95 | 2.57 | 8.38 | 1.02 | 0.78 |
| 1000 | 22.65 | 3.09 | 7.17 | 1.14 | 0.88 |
| 1500 | 20.14 | 2.10 | 7.32 | 0.91 | 0.76 |
| 2000 | 19.50 | 2.26 | 6.67 | 0.98 | 0.77 |
| 2500 | 17.22 | 2.28 | 6.21 | 0.89 | 0.60 |
| 3000 | 18.27 | 2.42 | 5.99 | 0.92 | 0.63 |
| 3500 | 16.61 | 2.67 | 4.22 | 1.09 | 0.68 |
| 3900 | 14.00 | 1.34 | 4.53 | 0.97 | 0.50 |

### Epoch 1 Losses (every 500 batches)

| Batch | Total | Det(c) | Pose | Act | PSR |
|---|---|---|---|---|---|
| 498 | 9.59 | 1.33 | 0.69 | 0.97 | 0.47 |
| 998 | 8.45 | 2.22 | 1.50 | 1.13 | 0.14 |
| 1498 | 5.97 | 1.43 | 1.51 | 0.79 | 0.00 |
| 1998 | 3.77 | 1.43 | 0.09 | 0.95 | 0.00 |
| 2498 | 4.98 | 1.42 | 1.49 | 0.78 | 0.00 |
| 2998 | 6.44 | 1.50 | 1.79 | 0.78 | 0.00 |
| 3498 | 4.01 | 1.57 | 0.15 | 1.04 | 0.00 |
| 3998 | 7.50 | 2.62 | 1.61 | 1.08 | 0.00 |

### Epoch 2 Losses (every 500 batches, with live sample)

| Batch | Total | Det(c) | Pose | Act | PSR |
|---|---|---|---|---|---|
| 110 | 5.52 | 1.19 | 1.39 | 1.60 | 0.00 |
| 610 | 4.56 | 1.23 | 0.17 | 2.15 | 0.00 |
| 1110 | 4.44 | 1.30 | 0.13 | 1.70 | 0.00 |
| 830 (live) | 4.47 | 1.67 | 0.92 | 0.91 | 0.00 |

Note: epoch 2 data is sparse because batch logging only occurs every 500 batches. The live value at batch 830 is averaged from recent non-seq steps.

## Key Observations

1. **Pose converged extremely fast** — dropped from 8.38->0.69 in the first 500 batches of epoch 1. Now sits at 0.1-1.5 range. The pose head (WingLoss on 9-DoF head pose) is essentially learned. **However**, see Critical Concern #6: this may be magnitude-matching, not directional convergence. The GT forward vectors have norms 0.014-0.030, and MSE=0.1 can be achieved by outputting vectors of matching small magnitude regardless of direction.

2. **PSR collapsed to zero** — went from 0.78->0.47->0.14->0.00 across epoch 0->1 boundary. This is PSR sequence loss being zero on non-seq batches. The seq batches (every 2nd batch) show `seq=1` with only PSR loss. However, PSR logits on seq batches show pre_linear mean=~0.01, std=~1.6, post_gelu mean=~0.26 — the PSR head is still alive but producing near-zero loss on non-seq batches is expected.

3. **Activity is near random but improving** — 0.8-2.1 range, no clear downward trend yet. Activity ramp-up is 5 epochs, we are at epoch 2. The warmup (2 epochs) just ended. Expect activity to start converging from epoch 3+. Diversity has improved from 1 class (epoch 0) to 3 classes (epoch 2 val) to 5+ classes in live running. Entropy went from 0 to 1.036 nats. This is the first RF4 run showing any activity diversity improvement.

4. **Detection is stable** — det(c) (class loss) is 1.2-2.6, det(g) (giou box loss) is 0.06-0.61. Both in normal range for 4-task multi-training with Kendall. However, DET_PROBE shows score_p50=0.036, essentially the bias init value of ~0.033, meaning the classifier confidence has not separated from background after 5000+ batches.

5. **Weight decay (wd)** has been 0.24-0.31 across epochs — consistent with GRAD_CLIP_NORM=5.0 and WEIGHT_DECAY=1e-3. Much healthier than the old 5e-2 that dominated gradients.

6. **VRAM stable** at 7.4-7.9GB reserved on 16GB card. Plenty of headroom for larger batch or additional tasks.

7. **PSR warmup duration concern** — PSR_WARMUP_STEPS=500 completes in ~12-20 minutes of training. This may be too short for stable PSR head adaptation, potentially causing a loss spike when the precision multiplier reaches 1.0. Prior runs used 2000-6000 steps.

8. **Activity collapse status**: The trajectory is encouraging but not resolved — 1 class (epoch 0) -> 3 classes (epoch 2 step-val) -> 5+ classes (live). Entropy 0 -> 1.036 nats. The gate requires >=10 distinct classes and >=1.5 nats entropy for RF4 pass. This is the first time any improvement has been measured.
