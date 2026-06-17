# R2.5 Training — Complete Run History & Current Status
> Generated 2026-06-15 08:00 UTC — Training ALIVE

## Current Run (paper_run_r25_fix_20260615)

| Field | Value |
|-------|-------|
| Status | ACTIVE |
| PID | 2904634 |
| Epoch | 48 / 100 |
| Step | 5133 / 12579 (40.8%) |
| Runtime | 68 min (epoch), ~68h total |
| GRAD_NAN | 0 |
| SIGTERM | 0 |
| Log size | 2.1 MB |

### Per-Head Liveness (step 5000)

| Head | Loss | Grad (first layer) | Grad (last layer) | Status |
|------|------|--------------------|--------------------|--------|
| Detection | 1e-6 DEAD | 5.80e-02 ALIVE | 6.17e-02 ALIVE | Oscillating NO_GRAD/ALIVE |
| Activity | 1.12e+00 | 5.06e-02 ALIVE | 5.09e-02 ALIVE | HEALTHY |
| PSR | 5.44e-02 | 3.58e-02 ALIVE | 0.00e+00 DEAD | Alive in loss, bias head DEAD |
| HeadPose | 1.79e-02 | 8.75e-02 ALIVE | 4.97e-02 ALIVE | HEALTHY |
| Pose | 6.06e+00 | 1.72e-02 ALIVE | 5.99e-05 ALIVE | Weak last-layer grad |

### Kendall log_vars
```
det=0.000  head_pose=-1.268  act=0.000  psr=0.000
```
Activity log_var=0.000 (at clamp floor — Kendall cannot precision-boost activity above 1.0)

### Loss Pattern (epoch 48)
- Total loss: 5-17, highly variable per batch
- Activity dominates: 0.4-11.0 (wide swings, rare spikes to 11+)
- Pose: 0.005-0.9 (intermittent spikes)
- PSR: 0.006-0.13 (stable but low)
- Detection: mostly 1e-6 (background frames), occasional 1-4 (objects present)
- WD: 0.25 constant

### PSR Warmup Status
- PSR_WARMUP_STEPS = 6000
- PSR_WARMUP_INIT_MULT = 3.0
- Current step ~5133: **85.6% through warmup**
- Current mult ≈ 1.29× (ramping down from 3.0 toward 1.0)
- When complete (~step 6000): PSR precision mult = 1.0, no more gradient boost

---

## Complete Run History (R2.5)

| # | Log | Epochs | Steps | Outcome | GRAD_NAN | Key Detail |
|---|-----|--------|-------|---------|----------|------------|
| 1 | paper_run_r25_fix_20260615 | **48+** | **5133+** | **RUNNING** | **0** | First run past epoch 0 |
| 2 | paper_run_r25_intervention_20260614 | 0 | ~12579 | IndexError | 0 | act_clip_ids vs activity_mask mismatch at epoch-end eval |
| 3 | paper_run_r25_3ep_fixed2_20260614 | 0 | 8639 | SIGTERM | 0 | PSR=0.0000010 (completely dead), killed |
| 4 | paper_run_r25_fixed_20260614 | 0 | 1297 | SIGTERM | 0 | Early kill during debugging |
| 5 | paper_run_r25_20260614 | 0 | 226 | SIGTERM | 0 | Very early kill |
| 6 | full_100e_fp32_fix3_20260614 | 0 | 6594 | DataLoader killed | 206 | 206/206 skipped windows |
| 7 | full_100e_fp32_fix2_20260614 | 0 | 193 | Det loss 13.3 | N/A | Detection loss dominated |
| 8 | paper_run_r25_diag_20260614 | 0 | ~110 | NaN in LogBackward0 | Yes | Died immediately |
| 9 | paper_run_r25_3ep_20260614 | 0 | 0 | DataLoader killed | N/A | Never started |

### Death Analysis

| Run | Killed By | Root Cause |
|-----|-----------|------------|
| R2.5 intervention | IndexError | activity_mask size != act_clip_ids at eval — FIXED in current run |
| R2.5 3ep_fixed2 | SIGTERM | PSR completely dead (0.0000010), likely manual kill after detecting PSR collapse |
| FP32 fix3 | DataLoader SIGTERM + GRAD_NAN cascade | Detection head gradients overflowed (206/206 skipped windows) — AMP instability |
| R2.5 diag | NaN in LogBackward0 | Kendall log_var instability + PSR focal loss → numerical collapse |
| R2.5 3ep | DataLoader killed | DataLoader worker crashed on startup (likely OOM or stale multiprocessing) |

---

## Source File Changes (uncommitted, vs gap-closure commit)

### config.py
- BATCH_SIZE=1→2, GRAD_ACCUM_STEPS=32→16 (effective 32 unchanged)
- PSR_WEIGHT=60 (was 30), POSE_LOSS_WEIGHT=0.02 (was 0.001)
- ACTIVITY_HEAD_GRAD_CLIP=0.1, ACTIVITY_LOSS_WEIGHT=0.3
- Kendall bounds: ACT min=0, PSR max=0, Pose max=0
- PSR_WARMUP_STEPS=6000 (was 3000), PSR_WARMUP_INIT_MULT=3.0
- USE_PSR_SEQUENCE_MODE=False (was True — OOM)
- PSR_SEQ_EVERY_N_BATCHES=8 (was 4)

### train.py
- Activity head per-head grad clip at 0.1 (both AMP and FP32 paths)
- Gradient clip moved BEFORE NaN check (was after — fixed ordering bug)
- empty_cache() before sequence batches
- Missing targets.to(device) for keypoints, pose_confidence

### losses.py
- Detection focal: out-of-range label protection with clamp
- CBFocalLoss: activity target OOB guard with scatter_ fix
- POSE_LOSS_WEIGHT config-driven (was hardcoded 0.001)
- PSR sensitivity: batch>1 guard + correction=0 (fix NaN div by zero)

### evaluate.py
- activity_mask eval: skip act_clip_ids when act_valid[i] is False
- Segment eval: skip label==0 (NA segment)

---

## Architecture Constants (52.5M params)

| Component | Detail |
|-----------|--------|
| Backbone | ConvNeXt Tiny (28.6M) |
| Detection | RetinaNet FPN, 24 ASD classes |
| Activity | 75-class, CB-Focal + label smoothing |
| Pose | 17-keypoint Wing Loss |
| HeadPose | 9-DoF MSE with geo rotation |
| PSR | Causal Transformer (3L, 4H, d=256), 36×11 components |
| Total | 52.5M params |
| TMA | TMA Cell enabled |
| TemporalBank | Enabled, slot_overwrite=False |
| EMA | Enabled |
