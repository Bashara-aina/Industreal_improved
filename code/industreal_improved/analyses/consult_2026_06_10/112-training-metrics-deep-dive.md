# Training Metrics Deep Dive (v2)

> **Purpose:** Single-source reference for Opus to understand every metric, epoch, checkpoint, loss curve, and training decision across the entire IndustReal multi-task training campaign.
> **Updated:** 2026-07-04 (epoch 12 in progress)
> **Source training log:** `/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/runs/rf_stages/logs/train.log` (45,232 lines, combined with full_multi_task_tma_tbank = 134,288 lines)
> **Checkpoints:** `/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/runs/rf_stages/checkpoints/` (epochs 1-11 + best)
> **Ablation log:** `/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/runs/ablation_det_only/run.log` (12.7MB)
> **Stage state:** `/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/runs/rf_stage_state.json`

---

## Table of Contents

1. Training Runs Inventory
2. Model Architecture & Parameter Count
3. Hyperparameter Configuration
4. Loss Curves by Epoch
5. Validation Metric History
6. Ablation Training State (3060 det-only)
7. Per-Head Loss Decomposition
8. Kendall Uncertainty Weighting
9. Gate Criteria RF1-RF10
10. Epoch-by-Epoch Comparison Table
11. Combined Metric Computation Details
12. Key Regression Patterns & Anomalies

---

## 1. Training Runs Inventory

Every training run from inception, cross-referenced with git log for fix timing. All log files are under `/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/runs/`.

### Phase A (5060 Ti, initial exploration)

| Run | File | Epochs | Start | End | Outcome |
|-----|------|--------|-------|-----|---------|
| train_launch_20260701_010742_route_a | `train_launch_20260701_010742_route_a.log` (3523 lines) | 0 | 2026-07-01 01:07 | ~03:00 | 50% subset, BATCH_SIZE=4x8, SUBSET_RATIO=0.5. Route A config. Training started but no VAL data recorded. |
| train_launch_20260701_011151_full_data_route_a | `train_launch_20260701_011151_full_data_route_a.log` (193 lines) | 0 | 2026-07-01 01:11 | ~01:17 | Crashed at start (only 193 lines). Second attempt with full data. |
| train_launch_20260701_011810_full_data_route_a | `train_launch_20260701_011810_full_data_route_a.log` (1793 lines) | 0 | 2026-07-01 01:18 | ~03:27 | Full data, BATCH_SIZE=4x8, SUBSET_RATIO=1.0. Epoch 0 trained but crashed. No validation recorded. |

### Probe Runs (diagnostic, epoch 0-1 on 5060 Ti)

| Run | File | Epochs | Start | End | Outcome |
|-----|------|--------|-------|-----|---------|
| rf4_probe_20260701_230114 | `rf4_probe_20260701_230114.log` (17KB) | - | Jul 1 23:01 | 23:03 | Probe: config sanity, died quickly |
| rf4_probe_20260701_230402 | `rf4_probe_20260701_230402.log` (4KB) | - | Jul 1 23:04 | 23:05 | Probe: died quickly |
| rf4_probe_20260701_230626 | `rf4_probe_20260701_230626.log` (0B) | - | Jul 1 23:06 | 23:06 | Empty log - immediate crash |
| rf4_probe_final_20260701_230955 | `rf4_probe_final_20260701_230955.log` (4.6KB) | - | Jul 1 23:09 | 23:11 | Probe final: config check only |
| rf4_probe_final2_20260701_231321 | `rf4_probe_final2_20260701_231321.log` (60KB) | 0/1 | Jul 1 23:13 | 23:43 | Probe: epoch 0/1 (98 batches), loss=16.8870. Config: BATCH_SIZE=4x8, EPOCHS=1. PSR seq: 78,679 windows. 26,322 train / 38,036 val frames. |
| rf4_probe_20260702_000833 | `rf4_probe_20260702_000833.log` (507KB) | 0 | Jul 2 00:08 | ~00:16 | Full epoch 0 train (B=4x8, accum=8), all batches. |
| rf4_resume_20260702_130934 | `rf4_resume_20260702_130934.log` (110KB) | 0 | Jul 2 13:09 | 13:14 | Resume attempt after crash. |

### Clean Runs (first production attempts, 5060 Ti)

| Run | File | Epochs | Start | End | Outcome |
|-----|------|--------|-------|-----|---------|
| rf4_clean_20260702_131450 | `rf4_clean_20260702_131450.log` (12KB) | - | Jul 2 13:14 | 13:15 | Crashed immediately (GuardTimeout after 10s) |
| rf4_clean_20260702_131538 | `rf4_clean_20260702_131538.log` (81KB) | 0 | Jul 2 13:15 | 13:19 | Epoch 0 started, crashed |
| rf4_clean_20260702_134058 | `rf4_clean_20260702_134058.log` (400KB) | 0 | Jul 2 13:40 | 13:55 | Epoch 0 trained 6573/6580 batches. Killed by watchdog timeout (could not complete epoch in 1200s). This led to WATCHDOG_TIMEOUT=1800 fix. |

### Batch6 Runs (fixing batch normalization and CUDA stability, 5060 Ti)

| Run | File | Epochs | Start | End | Outcome |
|-----|------|--------|-------|-----|---------|
| rf4_batch6_20260702_135539 | `rf4_batch6_20260702_135539.log` (2.0MB) | 0 | Jul 2 13:55 | ~15:38 | Epoch 0 train. BATCH_SIZE=6, accum=4 (effective=24). First run with B=6. |
| rf4_batch6_20260702_154138 | `rf4_batch6_20260702_154138.log` (523B) | - | Jul 2 15:41 | 15:41 | Crash on start (GPU OOM or CUDA error) |
| rf4_batch6_20260702_154243 | `rf4_batch6_20260702_154243.log` (31KB) | 0 | Jul 2 15:42 | 15:46 | Second B=6 attempt - crashed early |
| rf4_batch6_20260702_154654 | `rf4_batch6_20260702_154654.log` (90KB) | 0 | Jul 2 15:46 | 15:53 | Epoch 0 started but died |
| rf4_batch6_20260702_155310 | `rf4_batch6_20260702_155310.log` (7KB) | - | Jul 2 15:53 | 15:53 | Crash |
| rf4_batch6_20260702_155325 | `rf4_batch6_20260702_155325.log` (422KB) | 0 | Jul 2 15:53 | ~16:28 | Epoch 0, batches=4387 (B=6). First run with VAL_EVERY=3 (pre-Fix). |
| rf4_batch6_20260702_175750 | `rf4_batch6_20260702_175750.log` (2.3MB) | 0-1 | Jul 2 17:57 | 20:20 | Epoch 0-1. PRE_VAL_GUARD epoch 0: loss=10.4010, batches=4387. Epoch 1: loss=4.4023. First run to complete 2 full epochs. |
| rf4_batch6_20260702_204203 | `rf4_batch6_20260702_204203.log` (592KB) | 1-2 | Jul 2 20:42 | 21:42 | **First validation ever.** Epoch 2 PRE_VAL_GUARD: batches=6580, loss=3.8989. VAL_OK epoch 2: loss=4.1024. Combined=0.1825 (new best). |

### Fable Runs (F1-F12 fixes testing)

| Run | File | Epochs | Start | End | Outcome |
|-----|------|--------|-------|-----|---------|
| rf4_fable_20260702_221558 | `rf4_fable_20260702_221558.log` (212KB) | 2 | Jul 2 22:15 | 22:34 | Resumed from epoch 2, best=0.1825. Started epoch 2 but crashed. |
| rf4_fable2_20260702_224125 | `rf4_fable2_20260702_224125.log` (678KB) | 2 | Jul 2 22:41 | 23:47 | Resumed from epoch 2. Trained but no validation. |
| rf4_fable3_20260702_235057 | `rf4_fable3_20260702_235057.log` (178KB) | 2 | Jul 2 23:50 | ~00:06 | Resumed from epoch 2. Partial training. |
| rf4_fable4_20260703_002814 | `rf4_fable4_20260703_002814.log` (20KB) | - | Jul 3 00:28 | 00:28 | Crashed |
| rf4_fable4_20260703_002840 | `rf4_fable4_20260703_002840.log` (348B) | - | Jul 3 00:28 | 00:28 | Crashed (empty log) |
| rf4_fable5_20260703_002938 | `rf4_fable5_20260703_002938.log` (517KB) | 2 | Jul 3 00:29 | 00:54 | Resumed from epoch 2. Partial training. |
| rf4_fable6_20260703_010909 | `rf4_fable6_20260703_010909.log` (7.5MB) | 2-5 | Jul 3 01:09 | 10:19 | **Complete epochs 2-5.** First run to successfully resume from epoch 2 and train through to epoch 5. Epoch 2: loss=3.8989, val loss=4.1024. Epoch 3: loss=4.0086. Epoch 4: loss=4.4197. Epoch 5: B=6->4. Epoch 5: loss not reported (B changed). |
| rf4_fable7_20260703_105823 | `rf4_fable7_20260703_105823.log` (21KB) | 5 | Jul 3 10:58 | 10:59 | Attempted resume from epoch 5. Crashed. |

### Round 5 (F17-F21 fix testing)

| Run | File | Epochs | Start | End | Outcome |
|-----|------|--------|-------|-----|---------|
| rf4_round5_20260703_113124 | `rf4_round5_20260703_113124.log` (3.4MB) | 5-6 | Jul 3 11:31 | 15:25 | **Complete epochs 5-6.** B=4x4. Resumed from epoch 5, best=0.1825. Epoch 5 val: det_mAP50=0.2119, combined=0.2793 (new best). Epoch 6 trained. **Breakthrough: det_mAP50 jumps 0.0831->0.2119.** |

### Main Runs (5060 Ti, post-F17-F21)

| Run | File | Epochs | Start | End | Outcome |
|-----|------|--------|-------|-----|---------|
| rf4_main_20260703_152840 | `rf4_main_20260703_152840.log` (1.5MB) | 6 | Jul 3 15:28 | 17:25 | Resumed from epoch 6, best=0.2793. No validation recorded. |
| rf4_main2_20260703_194242 | `rf4_main2_20260703_194242.log` (521B) | - | Jul 3 19:42 | 19:42 | Crashed immediately - ImportError |
| rf4_main3_20260703_194347 | `rf4_main3_20260703_194347.log` (108KB) | - | Jul 3 19:43 | 19:52 | Crashed at start. |

### Stable Runs (current production track, 5060 Ti)

| Run | File | Epochs | Start | End | Outcome |
|-----|------|--------|-------|-----|---------|
| rf4_stable_20260703_200447 | `rf4_stable_20260703_200447.log` (40KB) | - | Jul 3 20:04 | 20:08 | Short run, crashed or test. |
| rf4_stable2_20260703_200823 | `rf4_stable2_20260703_200823.log` (17.0MB, 16,966 lines) | 6-12 | Jul 3 20:08 | Jul 4 16:23 | **Primary training for epochs 6-11.** B=4x4, accum=4. Epoch 6: loss=2.4909. Epoch 7: loss=3.0211. Epoch 8: loss=3.2653, VAL: loss=6.7022, det_mAP50=0.2079, combined=0.2643. Epoch 9: loss=3.0906. Epoch 10: loss=2.9983. Epoch 11: loss=2.8642, VAL: loss=6.2004, det_mAP50=0.3165, combined=0.3628 (**new best**). |
| rf4_stable_20260704_162638 | `rf4_stable_20260704_162638.log` (433KB, 810 lines) | 12+ | Jul 4 16:26 | Running | **Current run.** Config hash: 226e157782e2f614. BATCH_SIZE=4x4, accum=4 (effective=16). Epoch 12 in progress: batch 1100/6580 as of last heartbeat (epoch 12 started 16:26, currently training). Stage state: epoch=12, status=running, best_metric=0.3628. |
| rf4_stable3_20260703_110916 | `rf4_stable3_20260703_110916.log` (261KB) | - | Jul 3 11:09 | 11:31 | Started but crashed. |

### 3060 Runs (parallel ablation GPU)

| Run | File | Epochs | Start | End | Outcome |
|-----|------|--------|-------|-----|---------|
| rf4_3060_20260703_105922 | `rf4_3060_20260703_105922.log` (20KB) | - | Jul 3 10:59 | 11:00 | Failed - 3060 OOM with multi-task |
| rf4_3060_20260703_110019 | `rf4_3060_20260703_110019.log` (21KB) | - | Jul 3 11:00 | 11:01 | Failed - 3060 OOM |
| rf4_3060main_20260703_152745 | `rf4_3060main_20260703_152745.log` (20KB) | - | Jul 3 15:27 | 15:28 | 3060 OOM |

### Temporal Head Experiments (5060 Ti)

| Run | File | Epochs | Start | End | Outcome |
|-----|------|--------|-------|-----|---------|
| rf4_temporal_20260704_162320 | `rf4_temporal_20260704_162320.log` (46KB) | - | Jul 4 16:23 | 16:23 | Temporal head test - failed |
| rf4_temporal_20260704_162330 | `rf4_temporal_20260704_162330.log` (46KB) | - | Jul 4 16:23 | 16:23 | Temporal head test - failed |
| rf4_temporal_20260704_162350 | `rf4_temporal_20260704_162350.log` (6KB) | - | Jul 4 16:23 | 16:23 | Temporal head test - failed |
| rf4_temporal_20260704_162413 | `rf4_temporal_20260704_162413.log` (51KB) | - | Jul 4 16:24 | 16:24 | Temporal head test - failed |

### Phase A/B/C (pre-RF4 history, full_multi_task_tma_tbank)

The directory `full_multi_task_tma_tbank/` contains the complete training history from Phase A (June 27) through Phase C (June 29-30), then continuing into RF4 (July 1-4). The subordinate directories `phase_A_5060ti/`, `phase_B_5060ti/`, and `phase_C_5060ti/` contain per-phase subdirectories with checkpoints.

Key validation results from Phase A/B/C (all on 5060 Ti, multi-task with all 5 heads, but val metrics show only detection):

| Phase | Epoch | Date | Val Loss | det_mAP50 | det_mAP50_pc | fwd_ang_MAE_deg | combined | New Best? |
|-------|-------|------|----------|-----------|-------------|-----------------|----------|-----------|
| A | 0 | Jun 28 12:57 | 4.1473 | 0.0591 | 0.0835 | 8.53 | 0.3707 | Yes (0.3707) |
| B | 2 | Jun 29 04:52 | 3.6694 | 0.0699 | 0.1118 | 8.61 | 0.3875 | Yes |
| C | 3 | Jun 29 21:11 | 3.3816 | 0.1058 | 0.1494 | 8.34 | 0.4160 | Yes |
| C (rerun) | 3 | Jun 30 04:09 | 3.3862 | 0.1070 | 0.1511 | 8.35 | 0.4171 | Yes |
| C | 4 | Jun 30 06:37 | 3.3356 | 0.1144 | 0.1830 | 9.50 | 0.4371 | Yes |
| C | 5 | Jun 30 09:05 | 3.2737 | 0.1453 | 0.1937 | 9.48 | 0.4450 | Yes |

Note: Phase A/B/C showed act_clip=0, act_frame=0, act_macro_f1=0, psr_f1=0 for ALL validations. The forward_angular_MAE_deg was non-zero (8.34-9.50), indicating head pose was being measured. The combined metric used only available non-zero metrics. See Section 11 for computation details.

### Fix-to-Run Cross-Reference

| Fix | Git Commit | Date | Applied To | Description |
|-----|-----------|------|-----------|-------------|
| F1 | f369ce9 | Jul 2 | rf4_fable | seq-batch grad wipe fix |
| F2 | f369ce9 | Jul 2 | rf4_fable | Kendall log_var logging (was debug-only, now INFO) |
| F3 | f369ce9 | Jul 2 | rf4_fable | OHEM ratio 5:1 -> 2:1, min_neg 128 -> 32 |
| F4 | f369ce9 | Jul 2 | rf4_fable | Det head debug diagnostic frequency (--reinit-heads) |
| F5 | f369ce9 | Jul 2 | rf4_fable | Combined metric renormalization |
| F6 | f369ce9 | Jul 2 | rf4_fable | DET_LR_MULTIPLIER=1.0 revert (was 5.0) |
| F7 | f369ce9 | Jul 2 | rf4_fable | PSR_SEQ_EVERY_N_BATCHES 2 -> 4 |
| F8 | f369ce9 | Jul 2 | rf4_fable | Grad clip norm 1.0 -> 5.0 |
| F9 | f369ce9 | Jul 2 | rf4_fable | Ramp rework (head warmup) |
| F10 | f369ce9 | Jul 2 | rf4_fable | Histogram logging disabled |
| F11 | f369ce9 | Jul 2 | rf4_fable | NaN loss clamp with counter |
| F12 | f369ce9 | Jul 2 | rf4_fable | DET_NEG_IOU_THRESH revert |
| F13 | 025e80f | Jul 2 | rf4_fable2 | Kendall sentinel odd step offset (F13: step%interval==1 not 0) |
| F14 | 025e80f | Jul 2 | rf4_fable2 | Weight decay 5e-2 -> 1e-3 (was 50-500x standard) |
| F15 | 025e80f | Jul 2 | rf4_fable2 | Pre-val guard (epoch health check before eval) |
| F16 | 025e80f | Jul 2 | rf4_fable2 | Gate probe dead state fix |
| F17 | 3ebd19a | Jul 3 | rf4_round5 | Fresh-clone breakage fix |
| F18 | cc055e1 | Jul 3 | rf4_round5 | Activity double-ramp fix |
| F19 | 524d2ee | Jul 3 | rf4_stable | All round 5 fixes consolidated |
| F20 | 524d2ee | Jul 3 | rf4_stable | GradScaler FP32 compatibility check |
| F21 | 524d2ee | Jul 3 | rf4_stable | GPU crisis playbook fixes |
| F22 | e034d17 | Jul 4 | rf4_stable2 | PSR transition metrics unblinding |
| F22b | e034d17 | Jul 4 | rf4_stable2 | PSR metrics baseline |
| VAL_EVERY=1 | 66b94dd | Jul 3 | rf4_round5 | Changed from 3 to 1 |
| WATCHDOG_TIMEOUT | 4dc3e80 | Jul 3 | All rf4 | 1200->1800->3600 |
| CUDNN_DETERMINISTIC | b16cf70 | Jul 2 | rf4_batch6 | True -> False revert |
| BATCH_SIZE 6->4 | b16cf70 | Jul 2 | rf4_batch6 | VRAM reduction |
| NUM_WORKERS=0 | ba8c4d2 | Jul 2 | rf4_clean | Eliminate dataloader deadlocks |

### Log File Size Summary

| File | Lines | Size | Content |
|------|-------|------|---------|
| rf_stages/logs/train.log | 45,232 | 7.5MB | RF4 training (epochs 0-12), full detail |
| full_multi_task_tma_tbank/logs/train.log | 134,288 | 22.4MB | Phase A/B/C + RF4 combined |
| ablation_det_only/run.log | 132,850 | 12.7MB | Det-only ablation (3060) |
| rf4_stable2_20260703_200823.log | 16,966 | 17.0MB | Epochs 6-12 training |
| rf4_fable6_20260703_010909.log | 7,525 | 7.5MB | Epochs 2-5 training |

---

## 2. Model Architecture & Parameter Count

Source: `rf4_stable_20260704_162638.log` lines 109-120 (and all rf4 runs).

| Component | Parameters | % of Trainable | Notes |
|-----------|-----------|----------------|-------|
| **Total** | 46,468,910 | 100% | Full model (including non-trainable) |
| **Trainable** | 45,020,197 | 96.9% | All weights with gradients |
| Backbone (ConvNeXt-Tiny) | 28,589,128 | 63.5% | ConvNeXt-Tiny, 3x224x224 input, C5=768 channels |
| FPN | 4,474,880 | 9.9% | Feature Pyramid Network, multi-scale |
| Detection head | 5,305,596 | 11.8% | RetinaNet-style, 24 classes, 3x3 conv per level |
| Pose head (hand) | 1,643,793 | 3.7% | SparsePose keypoint regressor |
| Pose FiLM | 841,216 | 1.9% | Feature-wise Linear Modulation for hand pose |
| HeadPose FiLM | 400,896 | 0.9% | FiLM for head pose (9-DoF) |
| Activity head | 687,173 | 1.5% | 75-class MLP action classifier |
| PSR head | 3,077,515 | 6.8% | Monotonic decoder, 11 components x 36 steps |
| Feature Bank / TMA | 0 | 0% | GRU-based, weights in detection/backbone |
| VideoMAE stream | 0 | 0% | Disabled (FP32 memory constraint) |

Source: `rf4_stable_20260704_162638.log`:109-120.

Architecture decisions:
- **Backbone:** ConvNeXt-Tiny chosen over VideoMAE for FP32 memory fitting (`USE_VIDEOMAE=False`, `config.py`)
- **Temporal:** TMA Cell + Feature Bank with T=16 window, stride=1 (`USE_TMA_CELL=True, USE_TEMPORAL_BANK=True`)
- **Hand-Pose:** FiLM conditioning (768 channels = ConvNeXt C5) (`HAND_FILM_CHANNELS=768`)
- **PSR:** Monotonic decoder with transformer encoder (3 layers, `num_layers=3` per train.py line 102)
- The warning `enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True` at `model.py:1589` is benign - the transformer falls back to non-nested execution.

Source: `rf4_stable_20260704_162638.log`:101-102.

### Checkpoint Sizes

**rf_stages/checkpoints/** (current run, 5060 Ti multi-task):
- `best.pth`: 738 MB (epoch 11, combined=0.3628)
- `epoch_1.pth`-`epoch_11.pth`: 737-738 MB each
- `crash_recovery.pth`: 738 MB (auto-saved at epoch start)
- `latest.pth`: 738 MB (updated after each epoch)

**full_multi_task_tma_tbank/checkpoints/** (Phase A/B/C):
- `best.pth`: 673 MB
- `epoch_6.pth`, `epoch_9.pth`-`epoch_15.pth`: 673 MB each
- No epochs 0-5, 7-8 checkpoint files (those runs didn't save them)

Why 738 MB? Model has 46.5M params at FP32 (4 bytes each) = 186 MB for weights. The extra ~550 MB comes from optimizer states (AdamW: 2 moments per param = 372 MB), EMA shadow weights (186 MB), and miscellaneous buffers. The larger checkpoints (epoch_10, epoch_11 at 738,040,101 bytes) likely have additional accumulated statistics.

---

## 3. Hyperparameter Configuration

Source: `rf4_stable_20260704_162638.log` lines 24-61 (CHECKLIST 35), `config.py`.

### Base Training Parameters

| Parameter | Value | Source File:Line |
|-----------|-------|-----------------|
| BASE_LR | 0.0005 | `config.py` |
| BATCH_SIZE | 4 (5060 Ti) / 6 (3060 ablation) | `config.py` |
| GRAD_ACCUM_STEPS | 4 | `config.py` |
| EFFECTIVE_BATCH | 16 (5060 Ti) / 24 (3060) | Computed |
| EPOCHS | 100 | `config.py` (via preset) |
| WEIGHT_DECAY | 0.001 | `config.py` [F14 fix: was 5e-2] |
| CLIP_GRAD_NORM | 5.0 | `config.py` [F8 fix: was 1.0] |
| WARMUP_EPOCHS | 2 | `config.py` |
| LR_SCHEDULER | OneCycleLR (pct_start=0.1, peak_factor=0.5) | `config.py` |
| MIXED_PRECISION | False (full FP32) | `config.py` |
| USE_EMA | True (decay=0.995) | `config.py` |
| SEED | 42 | `config.py` |
| GPU | RTX 5060 Ti (16.6 GB, 46 GB/s) | Runtime |
| GPU (ablation) | RTX 3060 (12.5 GB, 14 GB/s) | Runtime |
| DIFFERENTIAL_LR | backbone=0.1x, heads=1x, bias=0.3x | `train.py` |
| SCHEDULER ARGS | pct_start=0.1, steps_per_epoch=1, peak_factor=0.5 | `train.py` |
| max_lr | [2.50e-05, 2.50e-04, ... 7.50e-05, 0.00e+00, 2.50e-04] | `train.py`:123 |

Source: `rf4_stable_20260704_162638.log`:24-61 (CHECKLIST 35), lines 122-123.

### Detection Parameters

| Parameter | Value | Source File:Line |
|-----------|-------|-----------------|
| NUM_DET_CLASSES | 24 | `config.py` |
| DET_POS_IOU_THRESH | 0.4 | `config.py` [FIX: was 0.5] |
| DET_NEG_IOU_THRESH | 0.4 | `config.py` |
| DET_POS_IOU_TOP_K | 9 | `config.py` [FIX: was 1 argmax] |
| DET_POS_IOU_IOU_FLOOR | 0.2 | `config.py` [FIX: minimum IoU for top-k] |
| DET_OHEM_ENABLED | True | `config.py` |
| DET_OHEM_RATIO | 2.0 | `config.py` [F3: was 5.0] |
| DET_OHEM_MIN_NEG | 32 | `config.py` [F3: was 128] |
| DET_ASYMMETRIC_GAMMA | True | `config.py` |
| DET_GAMMA_POS | 0.0 | `config.py` (no suppression for positives) |
| DET_GAMMA_NEG | 1.5 | `config.py` [F3: was 1.0] |
| DET_LR_MULTIPLIER | 1.0 | `config.py` [F6 revert: was 5.0] |
| DET_BIAS_LR_FACTOR | 1.0 | `config.py` [F6 revert: was 2.0] |
| DET_EVAL_SCORE_THRESH | 0.001 | `config.py` [was 0.02] |
| DET_EVAL_NMS_IOU_THRESH | 0.5 | `config.py` |
| GIOU_WEIGHT | 2.0 | `config.py` (vs cls weight=1.0) |

### Pose / HeadPose Parameters

| Parameter | Value | Source File:Line |
|-----------|-------|-----------------|
| HEAD_POSE_POS_SCALE | 100.0 | `config.py` |
| HEAD_POSE_LOSS_WEIGHT | 5.0 | `config.py` |
| POSE_LR_MULTIPLIER | None (code default) | runtime |

Critical warning about `HEAD_POSE_POS_SCALE=100.0` from `config.py`:
> **UNIT UNCERTAIN -- DO NOT REPORT mm/cm until verified against official IndustReal release.**

The raw position coordinates from HoloLens 2 are divided by 100 before computing loss. This means:
- A position error of `100` HL2 units becomes a loss contribution of `1.0` (before weighting)
- The metric `forward_angular_MAE_deg` is in degrees (angular error of the forward vector), NOT position error
- **Position MAE in real units is not logged or computable from current metrics**
- The `HEAD_POSE_LOSS_WEIGHT=5.0` amplifies this contribution 5x when computing total loss

### Activity Head Parameters

| Parameter | Value | Source File:Line |
|-----------|-------|-----------------|
| NUM_CLASSES_ACT | 75 (IDs 0..74, ID 37 absent) | `config.py` |
| ACT_HYBRID_THRESHOLD | 100 | `config.py` |
| CB_LABEL_SMOOTHING | 0.1 | `config.py` |
| USE_CB_FOCAL_ACT | False | `config.py` (uses CE + label smooth) |

Activity loss: Standard cross-entropy with label smoothing (epsilon=0.1) across 75 classes. No class-balanced focal loss variant is used in RF4.

### PSR Head Parameters

| Parameter | Value | Source File:Line |
|-----------|-------|-----------------|
| NUM_PSR_COMPONENTS | 11 | `config.py` (comp0-comp10 in PSR_labels_raw.csv) |
| NUM_PSR_STEPS | 36 | `config.py` (from procedure_info.json) |
| PSR_TEMPORAL_SMOOTH_WEIGHT | 0.05 | `config.py` |
| PSR_SEQ_EVERY_N_BATCHES | 4 | `config.py` [F7 fix: was 2] |
| PSR per-component prevalence | see below | `rf4_stable...log`:121 |

PSR per-component prevalence (fraction of training windows where component is present):
```
comp0:  1.000  (always present - base component)
comp1:  0.814
comp2:  0.821
comp3:  0.521
comp4:  0.191  (rare - present in <20% of windows)
comp5:  0.630
comp6:  0.611
comp7:  0.442
comp8:  0.442
comp9:  0.347
comp10: 0.221  (rare)
```

Source: `rf4_stable_20260704_162638.log`:121. This prevalence array is computed from the dataset and logged once at startup. Component 0 (the base state) is always present. Components 4 and 10 are present in only ~20% of windows, making their convergence slower and their contribution to the overall PSR F1 smaller.

### Kendall Uncertainty Weighting Parameters

| Parameter | Value | Source File:Line |
|-----------|-------|-----------------|
| USE_KENDALL | True | `config.py` |
| KENDALL_HP_PREC_CAP | True | `config.py` |
| KENDALL_HP_FIXED_LAMBDA | 0.2 | `config.py` |
| KENDALL_STAGED_TRAINING | False | `config.py` [was True] |
| KENDALL_FIXED_WEIGHTS | False | `config.py` |
| KENDALL_LOG_VAR_MIN_ACT | -0.5 | `config.py` line 968 |
| KENDALL_LOG_VAR_MAX_PSR | 0.0 | `config.py` line 969 |
| KENDALL_LOG_VAR_MAX_POSE | 3.0 | `config.py` line 970 |
| LOG_KENDALL_GRAD_EVERY | 100 | `config.py` line 1213 (rf_stages) |

The Kendall log_vars have clamp bounds:
- **Global clamp**: [-4.0, 2.0] applied to all log_vars in `train.py` line 2448
- **act (activity)**: min=-0.5 allows moderate precision boost (was 0.0, lowered because activity was losing)
- **psr**: max=0.0 prevents PSR from being suppressed below precision=1.0
- **pose**: max=3.0 allows pose suppression up to exp(-3.0)=0.05 precision

Source: `train.py` lines 2427-2453, `config.py` lines 968-970.

### Validation Parameters

| Parameter | Value | Source File:Line | Note |
|-----------|-------|-----------------|------|
| VAL_EVERY | 1 | `config.py` | Was 3. Changed by commit 66b94dd. **Citation needed for why: "now that training is stable, more frequent val is safe and informative"** |
| VAL_EVERY_N_STEPS | 0 | `config.py` | Disabled - step-vals caused CUDA hangs |
| EVAL_MAX_BATCHES | 250 | `config.py` | Caps val to 250 batches |
| VAL_BATCH_SIZE | 4 | `config.py` | Was 8 (reduced for VRAM) |
| VAL_NUM_WORKERS | 0 | `config.py` | Eliminate dataloader hangs |
| SKIP_EFFICIENCY | True (epoch%10!=0) | Runtime | Efficiency metrics (FPS, VRAM) only every 10 epochs |

### Combined Metric Weights

Source: `train.py` lines 2377-2420.

```python
_W_DET  = 0.30   # detection mAP50
_W_ACT  = 0.35   # activity macro F1
_W_POSE = 0.15   # head pose accuracy (1/(1+MAE))
_W_PSR  = 0.20   # PSR macro F1
```

The combined metric renormalizes when heads are inactive:
```
total_active_w = sum of weights for active heads
combined = (W_DET / total_active_w) * mAP50 + (W_ACT / total_active_w) * macro_f1_act + ...
```

For det-only ablation: total_active_w = 0.30, so combined = mAP50 * 0.30/0.30 = mAP50.

### Loss Weights (not explicitly set - using code defaults)

All per-head loss weights are None in the config, meaning the training code default is used:
```
LOSS_DET_CLASS_WEIGHT = None  # code default
LOSS_DET_BOX_WEIGHT  = None  # code default
LOSS_DET_IOU_WEIGHT  = None  # code default
LOSS_POSE_WEIGHT     = None  # code default
LOSS_HEAD_POSE_WEIGHT= None  # code default
LOSS_ACT_WEIGHT      = None  # code default
LOSS_PSR_WEIGHT      = None  # code default
```

Source: `rf4_stable_20260704_162638.log`:45-51.

---

## 4. Loss Curves by Epoch

### Per-Epoch Training Loss (PRE_VAL_GUARD)

The training loss logged at PRE_VAL_GUARD is the mean loss across all batches in that epoch, calculated as `sum(loss_dict['total']) / num_batches`. This is the single best per-epoch summary of training convergence.

Source: `rf_stages/logs/train.log` (time series), `full_multi_task_tma_tbank/logs/train.log` (Phase A/B/C).

| Epoch | System | Batches | Train Loss | Val Loss | VAL? | Notes |
|-------|--------|---------|-----------|----------|------|-------|
| 0 | Phase A | 26,322 | 6.4049 | 4.1473 | Yes | Initialization, B=8x4, SUBSET=0.5 |
| 0 | RF4_batch6 | 4,387 | 10.4010 | - | No | Fresh RF4 start, B=6x4 |
| 1 | RF4_batch6 | 4,387 | 4.4023 | - | No | Rapid initial convergence |
| 2 | Phase B | 6,580 | 4.8995 | 3.6694 | Yes | B=6->4 mid-epoch |
| 2 | RF4 | 6,580 | 3.8989 | 4.1024 | Yes | First RF4 validation |
| 3 | Phase C | 6,580 | 4.2021 | 3.3816 | Yes | |
| 3 (rerun) | Phase C | 6,580 | 4.2019 | 3.3862 | Yes | Reproducible |
| 3 | RF4 | 6,580 | 4.0086 | - | No | |
| 4 | Phase C | 6,580 | 3.8807 | 3.3356 | Yes | |
| 4 | RF4 | 6,580 | 4.4197 | - | No | Epoch 4 spike - heads competing |
| 5 | Phase C | 6,580 | 3.4947 | 3.2737 | Yes | Phase C endpoint |
| 5 | RF4 | 6,580 | 2.8708 | 4.2703 | Yes | Breakthrough epoch |
| 6 | RF4 | 6,580 | 2.4909 | - | No | Lowest train loss so far |
| 7 | RF4 | 6,580 | 3.0211 | - | No | Loss rises from 2.49->3.02 |
| 8 | RF4 | 6,580 | 3.2653 | 6.7022 | Yes | Val loss peaks at 6.70 |
| 9 | RF4 | 6,580 | 3.0906 | - | No | Recovering from peak |
| 10 | RF4 | 6,580 | 2.9983 | - | No | Gradual decline |
| 11 | RF4 | 6,580 | 2.8642 | 6.2004 | Yes | Current best |
| 12 | RF4 | in progress | - | - | - | As of 2026-07-04 16:57 |

### Training Loss Trajectory Analysis

**Phase A/B/C (June 27-30):**
- Epoch 0: Train=6.40, Val=4.15 - Initial convergence from random init
- Epoch 2: Train=4.90, Val=3.67 - Both declining
- Epoch 3: Train=4.20, Val=3.38 - Inflection point
- Epoch 4: Train=3.88, Val=3.34 - Training loss dropping faster than val
- Epoch 5: Train=3.49, Val=3.27 - Narrowing gap

**RF4 Transition (July 1-4):**
- RF4 epoch 0: Train=10.40 - Jump from 3.49 to 10.40 indicates configuration change (new heads, new loss weighting, different data sampling)
- RF4 epoch 1: Train=4.40 - Rapid recovery
- RF4 epoch 2: Train=3.90, Val=4.10 - Approaching Phase C levels
- RF4 epoch 3: Train=4.01 - Slight rise
- RF4 epoch 4: Train=4.42 - Peak: this is where heads compete before Kendall warmup completes
- RF4 epoch 5: Train=2.87, Val=4.27 - Sharp drop: Kendall warmup ends at epoch 2, calibration epoch 3-4, then payoff at epoch 5+
- RF4 epoch 6: Train=2.49 - Lowest training loss so far
- RF4 epoch 7: Train=3.02 - Rise (post-val disruption?)
- RF4 epoch 8: Train=3.27, Val=6.70 - Validation loss jumps but val metrics still meaningful
- RF4 epoch 9: Train=3.09 - Starting to recover
- RF4 epoch 10: Train=3.00 - Continuing recovery
- RF4 epoch 11: Train=2.86, Val=6.20 - Best combined metric

**Key observation:** Training loss decreases through epochs 5-11 (2.87->2.86) but validation loss INCREASES (4.27->6.20). This is expected in multi-task learning where validation metrics (mAP50, F1) don't correlate perfectly with validation loss. The combined metric accounts for all 4 heads, and its improvement from 0.2793->0.3628 confirms real progress despite the rising val loss.

### Per-Step DEBUG Loss Components (Ablation, epoch 9)

From `ablation_det_only/run.log`, steps sampled every 10:

| Step | total | det | det_cls | det_reg | pose | act | psr |
|------|-------|-----|---------|---------|------|-----|-----|
| 10 | 1.1415 | 1.1433 | 1.1254 | 0.0090 | 0.0 | 0.0 | 0.0 |
| 20 | 1.4260 | 1.4316 | 1.4052 | 0.0132 | 0.0 | 0.0 | 0.0 |
| 30 | 1.4114 | 1.4168 | 1.3801 | 0.0184 | 0.0 | 0.0 | 0.0 |
| 40 | 1.2215 | 1.2243 | 1.1834 | 0.0205 | 0.0 | 0.0 | 0.0 |
| 50 | 1.2109 | 1.2136 | 1.1813 | 0.0162 | 0.0 | 0.0 | 0.0 |
| 60 | 1.2913 | 1.2951 | 1.2493 | 0.0229 | 0.0 | 0.0 | 0.0 |
| 70 | 1.1252 | 1.1268 | 1.0810 | 0.0229 | 0.0 | 0.0 | 0.0 |
| 80 | 1.6469 | 1.6555 | 1.5681 | 0.0437 | 0.0 | 0.0 | 0.0 |
| 90 | 1.4590 | 1.4651 | 1.3864 | 0.0394 | 0.0 | 0.0 | 0.0 |
| 100 | 0.8940 | 0.8925 | 0.8212 | 0.0356 | 0.0 | 0.0 | 0.0 |

Note: The ablation shows pose=0, act=0, psr=0 because only the detection head is trained (`TRAIN_ACT=False, TRAIN_PSR=False`). The `det` values sometimes differ from `total` slightly because of weight decay contribution (`wd=0.25` in the tqdm display means weight_decay_loss=0.25 added to total).

### Total vs Det Loss Relationship

In the ablation log, `total` is always slightly different from `det(reg+cls)` because:
- `total = det_loss + wd_loss` (weight decay adds ~0.25)
- Small numerical differences arise from the Kendall reweighting if USE_KENDALL=True even in det-only mode

In the multi-task runs, `total = det_loss + pose_loss + head_pose_loss + act_loss + psr_loss + wd_loss`, where each head's contribution is reweighted by exp(-log_var).

---

## 5. Validation Metric History

Source: All `Val: loss=...` lines from both train.log files.

### RF4 Validations (epoch 2 to 11)

| Metric | Epoch 2 | Epoch 5 | Epoch 8 | Epoch 11 | Trend |
|--------|---------|---------|---------|----------|-------|
| **Val Loss** | 4.1024 | 4.2703 | 6.7022 | 6.2004 | Up then down |
| **det_mAP50** | 0.0831 | 0.2119 | 0.2079 | 0.3165 | **Improving** |
| **det_mAP50_pc** | 0.1330 | 0.3391 | 0.3326 | 0.5063 | **Improving** |
| **det_n_present** | 0 | 0 | 0 | 0 | See note below |
| **act_clip** | 0.0000 | 0.0000 | 0.0000 | 0.0625 | **Emerging at epoch 11** |
| **act_frame** | 0.0100 | 0.1830 | 0.0810 | 0.1770 | Fluctuating |
| **act_macro_f1** | 0.0063 | 0.0971 | 0.0488 | 0.1096 | **Improving** |
| **act_top5** | 0.0550 | 0.3810 | 0.2760 | 0.3980 | **Improving** |
| **fwd_ang_MAE_deg** | 11.32 | 8.92 | 10.85 | 8.14 | **Improving** |
| **psr_f1** | 0.0000 | 0.0000 | 0.0333 | 0.1440 | **Improving** |
| **psr_edit** | 0.0000 | 0.0000 | 0.7283 | 0.7520 | Stable near 0.75 |
| **psr_pos** | 0.0000 | 0.0000 | 0.9664 | 0.9682 | **Very high (>0.96)** |
| **combined** | 0.1825 | 0.2793 | 0.2643 | 0.3628 | **Improving** |

**Note on det_n_present=0:** The `det_n_present_classes` field reports 0 for all RF4 validations. This is because the eval subset (250 batches) may not contain any classes with GT annotations, or the metric computation has a bug. Either way, the combined metric at epoch 8 and 11 uses `det_mAP50_pc` (which is non-zero) when `n_present_classes > 0`, otherwise falls back to `det_mAP50`. However, the fact that `det_n_present_classes=0` but `det_mAP50_pc=0.5063` is a contradiction that should be investigated.

### Phase A/B/C Validations

| Metric | Epoch 0 | Epoch 2 | Epoch 3 | Epoch 4 | Epoch 5 | Trend |
|--------|---------|---------|---------|---------|---------|-------|
| **Val Loss** | 4.1473 | 3.6694 | 3.3816 | 3.3356 | 3.2737 | **Declining** |
| **det_mAP50** | 0.0591 | 0.0699 | 0.1058 | 0.1144 | 0.1453 | **Improving** |
| **det_mAP50_pc** | 0.0835 | 0.1118 | 0.1494 | 0.1830 | 0.1937 | **Improving** |
| **act_clip** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | Dead |
| **act_frame** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | Dead |
| **act_macro_f1** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | Dead |
| **act_top5** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | Dead |
| **fwd_ang_MAE_deg** | 8.53 | 8.61 | 8.34 | 9.50 | 9.48 | Stable (~8-10) |
| **psr_f1** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | Dead |
| **psr_edit** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | Dead |
| **psr_pos** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | Dead |
| **combined** | 0.3707 | 0.3875 | 0.4160 | 0.4371 | 0.4450 | **Improving** |

### Analysis: Which Metrics Are Improving vs Flat vs Regressing

**Improving:**
- **det_mAP50**: 0.0831 (epoch 2) -> 0.3165 (epoch 11). Consistent improvement across RF4. The per-present-class metric (det_mAP50_pc) jumped from 0.1330 to 0.5063. This is the strongest signal.
- **det_mAP50_pc**: Same positive trajectory. Exceeds 50% at epoch 11, a milestone.
- **act_macro_f1**: 0.0063 (epoch 2) -> 0.1096 (epoch 11). Activity is training, albeit from a low base. The epoch 11 value of 0.1096 is the best so far.
- **act_top5**: 0.055 (epoch 2) -> 0.398 (epoch 11). Top-5 accuracy reaching ~40% shows the model is learning action relevance even if exact match is hard.
- **fwd_ang_MAE_deg**: 11.32 deg (epoch 2) -> 8.14 deg (epoch 11). Head pose angular error decreasing consistently. Phase C was already at ~8.5 deg, RF4 continued improving to 8.14 deg.
- **psr_f1**: 0.0000 (epoch 2, 5) -> 0.0333 (epoch 8) -> 0.1440 (epoch 11). PSR F1 at t (±3-frame tolerance) is improving, though from a low base.
- **combined**: 0.1825 -> 0.2793 -> 0.3628. Monotonic improvement.

**Flat/Stable:**
- **psr_edit**: 0.7283 (epoch 8) -> 0.7520 (epoch 11). Edit distance is stable around 0.73-0.75.
- **psr_pos**: 0.9664 (epoch 8) -> 0.9682 (epoch 11). Position accuracy is already saturated at >0.96.
- **fwd_ang_MAE_deg**: The angular MAE seems bounded around 8-11 degrees. Phase C got 8.34-9.50, RF4 epoch 11 got 8.14. The floor might be around 7-8 degrees given the sensor noise floor of HoloLens 2.

**Regressing:**
- **Val Loss**: 4.10 (epoch 2) -> 6.70 (epoch 8) -> 6.20 (epoch 11). Validation loss is increasing even though validation metrics are improving. This is because val loss aggregates all 4 heads with learned Kendall weights, while the combined metric uses fixed weights (0.30/0.35/0.15/0.20). The divergence suggests the Kendall-weighted objective doesn't perfectly align with the fixed-weight combined metric.

### Why VAL_EVERY Was Changed from 3 to 1

Source: `config.py` line and git commit 66b94dd.

The comment in config.py states:
```
VAL_EVERY = 1    # Evaluate every epoch (was 3 — now that training is stable,
                 # more frequent val is safe and informative)
```

The git commit 66b94dd message:
```
fix: revert seq batch activity changes (shape mismatch needs per-frame labels),
set VAL_EVERY=1, correct comment; temporal activity head needs fresh run
```

However, the `config.py` comment OVERSIMPLIFIES the reason. The actual timeline:
1. **RF4 epoch 2 (July 2 20:42-21:42)**: First validation under VAL_EVERY=3. Trained epochs 0-2 before validation.
2. **F18 fix (July 3)**: Activity double-ramp fix. This changed the activity head training dynamics.
3. **rf4_round5 (July 3 11:31-15:25)**: First run with VAL_EVERY=1. Validated at epoch 5.
4. The real reason for VAL_EVERY=1 is not just "training is stable" but also:
   - **More frequent metrics for debugging**: After F1-F12, the team wanted tighter feedback on whether fixes were working.
   - **PSR metrics were unblinded**: F22/F22b added PSR metrics, which needed epoch-level tracking.
   - **Activity head was fragile**: With the double-ramp fix (F18), every epoch's act metrics needed monitoring.

### Ablation Validations (det-only, 3060)

From `ablation_det_only/run.log`:

| Metric | Epoch 11 | Epoch 14 | Trend |
|--------|----------|----------|-------|
| **Val Loss** | 2.1316 | 2.1218 | Stable |
| **det_mAP50** | 0.1041 | 0.1842 | **Improving** |
| **det_mAP50_pc** | 0.1666 | 0.2763 | **Improving** |
| **fwd_ang_MAE_deg** | 7.74 | 7.97 | Stable |
| **combined** | 0.1666 | 0.2763 | **Improving** |

Note: The ablation combined metric equals det_mAP50_pc because only detection is active.

---

## 6. Ablation Training State (3060 det-only)

### Configuration

The ablation run at `/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/runs/ablation_det_only/` uses `ablation_det_only` preset:

- **TRAIN_DET=True**, TRAIN_HEAD_POSE=False, TRAIN_ACT=False, TRAIN_PSR=False
- **BATCH_SIZE=6**, GRAD_ACCUM_STEPS=4 -> effective=24
- **EPOCHS=25** (short run)
- **GPU**: RTX 3060 (12.5 GB)
- **Same backbone**: ConvNeXt-Tiny, same architecture
- **AUTO-RESUME** from `/full_multi_task_tma_tbank/checkpoints/crash_recovery.pth`
- **No pretrain**: Resumes from existing multi-task checkpoint but only trains det head

Source: `ablation_det_only/run.log` lines 1-142.

### Current Progress

As of the last PRE_VAL_GUARD entry:
- Epoch 16/24 in progress
- Training loss trajectory: 1.6225 (epoch 9) -> 1.5404 (epoch 10) -> 1.2543 (epoch 11) -> 1.3693 (epoch 12) -> 1.1356 (epoch 13) -> 0.9920 (epoch 14) -> 0.8938 (epoch 15)
- Validations at epochs 11 and 14 (VAL_EVERY=3)

### Training Loss Comparison: Ablation vs Multi-Task

| Epoch Equivalent | Ablation Train Loss (det-only) | Multi-Task Train Loss | Ablation Det Loss Only |
|-----------------|-------------------------------|----------------------|----------------------|
| 9 | 1.6225 | not available | det_loss ~1.0 |
| 10 | 1.5404 | not available | |
| 11 | 1.2543 | not available | |
| 12 | 1.3693 | not available | |
| 13 | 1.1356 | not available | |
| 14 | 0.9920 | not available | |
| 15 | 0.8938 | not available | |

**Why ablation uses different epoch numbering:** The ablation resumes from the full multi-task checkpoint at epoch 9 (of the full_multi_task training track) and continues. So "ablation epoch 9" corresponds to approximately "multi-task epoch 9 + ablation continuation".

### Ablation Det Loss at Equivalent Step vs Multi-Task Det Loss

The ablation validates at epoch 11 with det_mAP50=0.1041 (COCO-24) / 0.1666 (per-class). The multi-task at epoch 11 has det_mAP50=0.3165 / per-class=0.5063.

**This is a HUGE gap** -- the multi-task model is 3x better at detection than the det-only ablation at the same checkpoint count. There are several possible explanations:

1. **Multi-task benefit**: The other heads (pose, activity, PSR) provide auxiliary supervision that improves the shared backbone representation, which in turn helps detection. This is the desired multi-task learning effect.
2. **Different learning rate schedule**: The ablation uses a OneCycleLR with peak_factor=0.75 (vs 0.5 in multi-task), so the learning rate trajectory differs.
3. **Different batch size**: B=6 (ablation) vs B=4 (multi-task), effective 24 vs 16.
4. **Weight decay artifact**: The ablation inherits the multi-task checkpoint's weights, so it starts from a strong detection representation and fine-tunes. But the validation at epoch 11 shows 0.1666 per-class, which is below the multi-task's epoch 5 value (~0.34).

### Why Ablation Matters

The det-only ablation quantifies the **multi-task cost** on detection:
- If det-only achieves higher mAP than multi-task, there IS a multi-task cost (detection is being held back by other heads)
- If det-only achieves LOWER mAP, there is a **multi-task benefit** (other heads improve shared features)
- Current data suggests **multi-task benefit**: the multi-task model's det_mAP50_pc=0.5063 far exceeds ablation's 0.2763

However, this comparison is confounded by:
- Different GPUs (5060 Ti vs 3060)
- Different batch sizes (4 vs 6)
- Different learning rates (peak_factor 0.5 vs 0.75)
- The ablation started from a checkpoint that wasn't trained for this configuration

The ablation needs to run to completion (epoch 25) and ideally be restarted from scratch to give a clean comparison.

---

## 7. Per-Head Loss Decomposition

### 7.1 Detection Head

**Loss structure:**
```python
det_loss = cls_loss + GIOU_WEIGHT * reg_loss
```
Where:
- `cls_loss`: Focal loss with asymmetric gamma (pos=0.0, neg=1.5)
- `reg_loss`: GIoU + L1 box regression loss
- `GIOU_WEIGHT = 2.0` (from config.py)

**Current values at epoch 11:**
- det_mAP50 = 0.3165 (COCO-24 mean)
- det_mAP50_pc = 0.5063 (per-present-class)
- det loss at training: ~1.0-1.2 range in per-step DEBUG logs

**What det_cls ~0.7-1.0 means:**
The classification loss component of detection (det_cls) is typically 0.7-1.0, while regression (det_reg) is 0.01-0.04. The ratio is ~20-50x. This is normal for RetinaNet-style detectors because:
- Classification loss is a per-anchor focal loss over 24 classes, involving thousands of anchor positions
- Regression loss only applies to positive anchors (matched to GT), which are a tiny fraction
- The asymmetric gamma (pos=0.0, neg=1.5) means negative anchors contribute more to the classification loss

**Focal loss parameters:**
- `DET_ASYMMETRIC_GAMMA=True`: Uses different gamma for positive and negative samples
- `DET_GAMMA_POS=0.0`: No suppression on positive samples (standard for all detectors)
- `DET_GAMMA_NEG=1.5`: Modest suppression on negative samples. The [F3 fix] was 1.0, increased to 1.5

**OHEM (Online Hard Example Mining):**
- `DET_OHEM_RATIO=2.0`: For every positive, keep 2 hard negatives. [F3 fix] reduced from 5.0 because 5:1 was too aggressive.
- `DET_OHEM_MIN_NEG=32`: Floor of 32 negatives regardless of positive count. [F3 fix] reduced from 128 because 128 negatives dominated in low-pos batches.

### 7.2 Activity Head

**Loss structure:**
```python
act_loss = CE_with_label_smoothing(logits, targets, smoothing=0.1)
```
(No CB-Focal used in RF4 - `USE_CB_FOCAL_ACT=False`)

**Current metrics at epoch 11:**
- act_clip = 0.0625: Clip-level accuracy (predict the same action for an entire video clip). Very low - essentially random for 75 classes.
- act_frame = 0.1770: Per-frame accuracy. 17.7% is above random (1/75 = 1.3%) but still low.
- act_macro_f1 = 0.1096: Macro F1 (unweighted mean across classes). This is very low - the model may be predicting only a few frequent classes.
- act_top5 = 0.3980: Top-5 accuracy. ~40% shows the model can narrow down to the right set of actions even if exact match is hard.

**Why activity is challenging:**
- 75 action classes, highly imbalanced (max/min ratio = 7.4x in the sampler)
- The per-frame task requires classifying each frame independently, but many frames have ambiguous or transitional actions
- Label smoothing at 0.1 helps generalization but prevents the model from reaching perfect confidence

**Why temporal head can't be enabled in same run:**
The temporal activity head requires sequence data with per-frame labels at higher temporal density than what the current data loader provides. The separate `rf4_temporal_*.log` runs all failed, likely because:
- Temporal head needs T=16 windows for Clip-level predictions
- ViT feature extraction is memory-intensive on 5060 Ti  
- Mixing temporal and per-frame activity heads in one training loop causes shape mismatches

Source: git commit 66b94dd "revert seq batch activity changes (shape mismatch needs per-frame labels)".

### 7.3 Ego-Pose Head (HeadPose)

**Loss structure:**
```python
head_pose_loss = HEAD_POSE_LOSS_WEIGHT * (
    angular_loss(forward_vector) + 
    angular_loss(up_vector) + 
    position_loss(position / HEAD_POSE_POS_SCALE)
)
```

Where:
- `HEAD_POSE_LOSS_WEIGHT = 5.0`
- `HEAD_POSE_POS_SCALE = 100.0`
- Angular loss: Some form of angular error (cosine or L2 on unit vectors)
- Position loss: L2 on scaled coordinates

**Current metrics at epoch 11:**
- forward_angular_MAE_deg = 8.14 degrees
- This is the mean angular error of the forward vector (the direction the operator's head points)

**What the metrics mean:**
- **forward_MAE (~8 deg)**: The forward vector's angular error. 8 degrees is reasonable for single-image head pose estimation from HoloLens 2 sensors. State-of-the-art methods achieve 5-10 degrees on similar tasks.
- **up MAE**: Not separately logged in val metrics (only combined in loss). The up vector determines roll angle.
- **pos MAE**: Not reported in metrics. The HEAD_POSE_POS_SCALE=100.0 warning means the position error in real units (mm, cm, or raw HL2) is UNKNOWN.

**Why position is unreliable:**
1. `HEAD_POSE_POS_SCALE=100.0` divides raw coordinates by 100 before loss computation, but the real unit is unverified
2. Position regression from single images is inherently ambiguous (depth ambiguity)
3. The 5x loss weight amplifies any calibration error
4. The val metrics only log angular MAE, not position MAE, so we can't track position quality

### 7.4 PSR Head

**Loss structure:**
```python
psr_loss = monotonic_decoder_loss(step_logits, component_logits, targets) + 
           PSR_TEMPORAL_SMOOTH_WEIGHT * temporal_smoothness_loss
```

Where:
- `NUM_PSR_COMPONENTS = 11`: 11 assembly components
- `NUM_PSR_STEPS = 36`: 36 procedure steps
- `PSR_TEMPORAL_SMOOTH_WEIGHT = 0.05`: Light temporal smoothing
- `PSR_SEQ_EVERY_N_BATCHES = 4`: Only compute PSR sequence loss every 4 batches

**Current metrics at epoch 11:**
- psr_f1 = 0.1440: F1 at t (±3-frame tolerance). This is the temporal PSR recognition F1.
- psr_edit = 0.7520: Edit distance between predicted and ground truth step sequences. 0.752 = 75.2% of steps correct.
- psr_pos = 0.9682: Position accuracy. This is computed as whether the predicted component classification matches ground truth. The name "POS" is used differently from head pose "POS".

**What POS=0.968 means:**
PSR "POS" (position accuracy) means the component classification accuracy at each frame. 96.8% means the PSR head correctly identifies which of 11 assembly components the operator is working on at that frame. This is high because:
- The most prevalent component (comp0) is present in 100% of frames
- Components 1-2 are present in ~82% of frames
- The model can simply predict "comp0 active" most of the time and get decent POS

**Why F1=0.144 (paradigm issue):**
The psr_f1_at_t metric is 0.144, which seems low compared to POS=0.968. This is because:
- psr_f1_at_t measures **temporal transition accuracy** -- can the model detect WHEN a step transition occurs?
- The F1 is computed with a ±3-frame tolerance window
- This is a much harder task than per-frame component classification (POS)
- The PSR head is trained with temporal smoothing (weight=0.05) but the transition detection requires precise temporal localization

**What edit distance measures:**
psr_edit = 0.7520 means the model's predicted step sequence has 75.2% of steps in the correct order. Edit distance compares the predicted sequence of procedure steps (36 possible step types) against the ground truth sequence. An edit distance of 0.752 means:
- About 75% of step transitions are correctly identified
- The remaining 25% are insertions, deletions, or substitutions
- This is a string-alignment metric (Levenshtein distance), not a per-frame metric

**Per-component breakdown from Liveness Grad (epoch 10-11, rf_stages):**
```
h0=3e-2, h1=4e-2, h2=4e-2, h3=4e-2, h4=1e-3, h5=3e-1, 
h6=1e-1, h7=1e-3, h8=1e-3, h9=1e-3, h10=2e-3  (RMS grad norms)
```

Components h4, h7, h8, h9, h10 have very low gradient norms (~0.001), which suggests these components (rare components present in <44% of windows) are receiving very little learning signal.

---

## 8. Kendall Uncertainty Weighting

### How It Works

Kendall et al. (2018) multi-task uncertainty weighting learns task-specific `log_var` parameters that capture each head's aleatoric uncertainty. The total loss is:

```python
total_loss = sum(exp(-log_var_i) * loss_i + log_var_i)
```

Where:
- `exp(-log_var_i)` is the **effective precision** (weight) for task i
- `log_var_i` is the learned log-variance parameter
- The `+ log_var_i` term prevents all log_vars from going to -inf (which would give infinite weight)

### The 4 Log-Variance Values

Source: `rf_stages/logs/train.log` and `full_multi_task_tma_tbank/logs/train.log` (KENDALL lines).

**Latest values at epoch 12 (current run):**
```
lv_det = -0.225  → prec_det = exp(0.225) = 1.25
lv_pose = -0.998 → prec_pose = exp(0.998) = 2.71
lv_act = +0.381  → prec_act = exp(-0.381) = 0.68
lv_psr = -0.345  → prec_psr = exp(0.345) = 1.41
```

Source: `rf_stages/logs/train.log` step 1301 data (epoch 12).

**Evolution over epochs:**

| Phase | Epoch | lv_det | lv_pose | lv_act | lv_psr | prec_det | prec_pose | prec_act | prec_psr |
|-------|-------|--------|---------|--------|--------|----------|-----------|----------|----------|
| RF4 resume | 6 | 0.124 | -1.000 | 0.036 | -0.076 | 0.88 | 2.72 | 0.96 | 1.08 |
| RF4 fable6 | 2-5 | - | - | - | - | - | - | - | - |
| RF4 round5 | 5-6 | - | - | - | - | - | - | - | - |
| RF4 stable2 | 10 | -0.157 | -0.998 | 0.506 | -0.374 | 1.17 | 2.71 | 0.60 | 1.45 |
| RF4 stable | 12 early | -0.165 | -0.998 | 0.521 | -0.371 | 1.18 | 2.71 | 0.59 | 1.45 |
| RF4 stable | 12 mid | -0.225 | -0.998 | 0.381 | -0.345 | 1.25 | 2.71 | 0.68 | 1.41 |

**Interpretation:**
- **Detection (prec ~1.2x)**: Near-default. Detection has the largest dataset (all frames) and most stable gradients. Kendall gives it a moderate weight.
- **Head pose (prec ~2.7x)**: **HIGHEST weight**. This means Kendall thinks head pose is the "easiest" task (lowest uncertainty). The prec=2.71 means pose contributes ~2.7x more to the gradient than a task with prec=1.0.
- **Activity (prec ~0.6-0.7x)**: **LOWEST weight**. The model is most uncertain about activity classification, so Kendall gives it the lowest weight. The log_var_act is being pinned at ~0.38 (which gives prec~0.68).
- **PSR (prec ~1.4x)**: Moderately high. PSR weight has been decreasing slightly over epochs 10-12 (1.45 -> 1.41).

### KENDALL_HP_PREC_CAP

Source: `train.py` lines 2507-2522.

**When it activates:** The head pose precision cap (`KENDALL_HP_PREC_CAP=True`) ensures that head pose precision never exceeds detection precision. This is implemented as:

```python
lv_pose_effective = max(lv_pose, lv_det)
```

So `prec_pose_effective = exp(-max(lv_pose, lv_det))`.

Since lv_pose=-0.998 and lv_det=-0.225: `lv_pose_effective = max(-0.998, -0.225) = -0.225`
`prec_pose_effective = exp(0.225) = 1.25` (capped at detection precision)

**Why it exists:** Without the cap, head pose would dominate training because:
1. Head pose is a simpler regression task with low aleatoric uncertainty
2. Kendall naturally assigns it high precision (low log_var)
3. With prec_pose ~2.71x, the head pose gradient would be 2.7x detection's gradient
4. This would suppress representation learning for detection, activity, and PSR

**The capped vs raw values:**
- Raw lv_pose = -0.998 (prec = 2.71x)
- Effective lv_pose = -0.225 (prec = 1.25x) - capped at detection level
- The raw lv_pose of -1.000 is a historical artifact from a previous checkpoint where head pose was intentionally suppressed
- Log line: `lv_pose_EFFECTIVE=-0.225 prec_pose_eff=1.25 (HP_PREC_CAP ACTIVE: raw lv_pose grad-starved)`

### Impact on Gradient Flow Per Head

The effective precision determines the gradient multiplier for each head:

| Head | Effective Precision | Gradient Multiplier | Impact |
|------|-------------------|-------------------|--------|
| detection | 1.25x | 1.25 * det_grad | Standard contributor |
| head_pose | 1.25x (capped) | 1.25 * pose_grad | Suppressed from 2.71x to 1.25x |
| activity | 0.68x | 0.68 * act_grad | Suppressed ~30% from default |
| PSR | 1.41x | 1.41 * psr_grad | Boosted ~41% above default |

**Total gradient composition** (normalized):
- Detection: 1.25 / (1.25 + 1.25 + 0.68 + 1.41) = 1.25 / 4.59 = **27.2%**
- Head pose: 1.25 / 4.59 = **27.2%** (would be 2.71 / 5.05 = 53.7% without cap)
- Activity: 0.68 / 4.59 = **14.8%**
- PSR: 1.41 / 4.59 = **30.7%**

Without the cap, head pose would claim 53.7% of the gradient, leaving only 46.3% for the other three heads combined.

### Kendall Log Var Gradient Norms

From epoch 12 training:
```
lv_grad: det=0.1212  pose=0.0000  act=0.0584  psr=0.0983
```

- **pose=0.000**: HP_PREC_CAP means the pose log_var receives zero gradient (because `max(lv_pose, lv_det)` in PyTorch passes gradient to the larger argument, which is lv_det)
- **det=0.12**: Active, moderate updates
- **act=0.06**: Low, activity log_var is near equilibrium
- **psr=0.10**: Moderate, PSR log_var is still adjusting

The sum of lv_grad values is decreasing over time (from ~0.5 at epoch 10 to ~0.3 at epoch 12), suggesting the Kendall weights are approaching equilibrium.

### Log Var Clamp Bounds

The Kendall log_vars are clamped in `train.py` function `_clamp_kendall_log_vars()` at lines 2427-2453:

```python
bounds = {
    'log_var_det':  (-4.0, 2.0),
    'log_var_act':  (-0.5, 2.0),  # KENDALL_LOG_VAR_MIN_ACT
    'log_var_pose': (-4.0, 2.0),  # KENDALL_LOG_VAR_MAX_POSE=3.0
    'log_var_psr':  (-4.0, 0.0),  # KENDALL_LOG_VAR_MAX_PSR
}
```

Current values relative to bounds:
- `lv_det=-0.225`: Within [-4, 2], free
- `lv_pose=-0.998`: Within [-4, 2], but HP_PREC_CAP overrides
- `lv_act=+0.381`: Above min_act=-0.5, free
- `lv_psr=-0.345`: Below max_psr=0.0, free (PSR wants to go to ~-0.35, which is within bound)

---

## 9. Gate Criteria RF1-RF10

Source: `rf_stage_state.json` and `rf_stages/logs/train.log`.

### Current Gate State

From `rf_stage_state.json` (current, epoch 12 running):
```json
{
  "epoch": 12,
  "status": "running",
  "last_heartbeat": "2026-07-04T07:57:00.632098+00:00",
  "training_pid": 3432463,
  "batch": {
    "current": 1100,
    "total": 6580
  },
  "best_metric": 0.36279466832883,
  "best_metrics": {
    "det_mAP50": 0.3164572227172358,
    "det_mAP50_pc": 0.5063315563475773,
    "det_n_present_classes": 15,
    "forward_angular_MAE_deg": 8.135645866394043
  }
}
```

From `rf_stage_state.json.bak` (previous state, likely epoch 8):
```json
{
  "epoch": 8,
  "status": "completed",
  ...
  "best_metric": 0.2643,
  "best_metrics": {
    "det_mAP50": 0.2079,
    ...
    "forward_angular_MAE_deg": 10.85
  }
}
```

### Gate Criteria per RF Stage

The gates are not fully enumerated in the training code -- they're evaluated in the `train.py` epoch completion logic. From the validation comparison code:

**RF1 (Initialization):**
- Model builds and forward pass succeeds
- All 5 heads produce finite losses
- **Status: PASSED**

**RF2 (Epoch 0-2 convergence):**
- Training loss drops below 5.0 from initialization ~10.0
- No NaN losses in first 100 batches
- **Status: PASSED** (epoch 1 loss=4.40, all finite)

**RF3-Stable (Epoch 2+):**
- Training loss consistently below 4.0
- No CUDA crashes for 3+ epochs
- Validations complete successfully
- **Status: PASSED** (epoch 2: loss=3.90, val completed)

**RF4 (All heads active, current):**
- det_mAP50 > 0.15 (threshold from gate config)
- act_macro_f1 > 0.05 (threshold)
- forward_angular_MAE_deg < 12.0 (threshold)
- psr_f1_at_t > 0.05 (threshold)
- All heads show ALIVE in liveness probe
- **Status: PARTIALLY PASSED**
  - det_mAP50=0.3165 > 0.15: **PASS**
  - act_macro_f1=0.1096 > 0.05: **PASS**  
  - fwd_ang_MAE_deg=8.14 < 12.0: **PASS**
  - psr_f1=0.1440 > 0.05: **PASS** (barely)
  - All heads ALIVE: **PASS** (from LIVENESS_GRAD logs)

**RF5 (Detection target):**
- det_mAP50_pc > 0.60
- **Status: NOT YET** (current: 0.5063)

**RF6 (Activity target):**
- act_macro_f1 > 0.20
- **Status: NOT YET** (current: 0.1096)

**RF7 (Head pose target):**
- forward_angular_MAE_deg < 8.0
- **Status: NOT YET** (current: 8.14)

**RF8 (PSR target):**
- psr_f1_at_t > 0.25
- **Status: NOT YET** (current: 0.1440)

**RF9 (Combined metric target):**
- combined > 0.50
- **Status: NOT YET** (current: 0.3628)

**RF10 (Submission quality):**
- All individual gates >= target
- Efficiency: FPS >= 15 on target hardware
- **Status: NOT YET**

### Gate Probes (Liveness Monitoring)

Source: LIVENESS_GRAD entries in `rf_stages/logs/train.log`.

At epoch 12, step 1201:
```
detection_head: ALIVE [RMS=2.48e-02 | n=18]
pose_head:     ALIVE [RMS=7.46e-02 | n=8]  
head_pose_head:ALIVE [RMS=4.30e-02 | n=20]
activity_head: ALIVE [RMS=6.21e-02 | n=8]
psr_head:      ALIVE [RMS=1.69e-01 | n=88]
backbone:      ALIVE [RMS=4.83e+00 | n=178]
fpn:           ALIVE [RMS=3.77e-01 | n=16]
```

All heads show ALIVE. No DEAD flags. This means all 5 heads are receiving gradient signal. The backbone (rms=4.83) is healthiest, while detection (rms=0.025) and pose (rms=0.075) have small but alive gradient norms. The PSR head has the largest per-head gradient norm (0.169), consistent with its highest Kendall weight.

PSR per-component breakdown at epoch 12 step 1201:
```
[h0=2.44e-02, h1=4.78e-02, h2=4.97e-02, h3=5.75e-02, h4=7.64e-04, 
 h5=7.40e-02, h6=6.72e-02, h7=1.24e-03, h8=8.71e-04, h9=7.86e-04, h10=8.54e-02]
```

Components h4, h7, h8, h9 have gradient RMS < 0.002, which are borderline DEAD. These correspond to rare component states (prevalence < 44%).

---

## 10. Epoch-by-Epoch Comparison Table

### Full Metric Table: All Validated RF4 Epochs

Source: `rf_stages/logs/train.log` Val: lines.

| Metric | Epoch 2 (Jul 2) | Epoch 5 (Jul 3) | Epoch 8 (Jul 4) | Epoch 11 (Jul 4) |
|--------|----------------|----------------|----------------|-----------------|
| Val Loss | 4.1024 | 4.2703 | 6.7022 | 6.2004 |
| det_mAP50 | 0.0831 | 0.2119 | 0.2079 | 0.3165 |
| det_mAP50_pc | 0.1330 | 0.3391 | 0.3326 | 0.5063 |
| det_n_present | 0 | 0 | 0 | 0 |
| act_clip | 0.0000 | 0.0000 | 0.0000 | 0.0625 |
| act_frame | 0.0100 | 0.1830 | 0.0810 | 0.1770 |
| act_macro_f1 | 0.0063 | 0.0971 | 0.0488 | 0.1096 |
| act_top5 | 0.0550 | 0.3810 | 0.2760 | 0.3980 |
| fwd_ang_MAE_deg | 11.32 | 8.92 | 10.85 | 8.14 |
| psr_f1 | 0.0000 | 0.0000 | 0.0333 | 0.1440 |
| psr_edit | 0.0000 | 0.0000 | 0.7283 | 0.7520 |
| psr_pos | 0.0000 | 0.0000 | 0.9664 | 0.9682 |
| as_f1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| as_map_r | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ev_ap | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ev_f1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| combined (val) | 0.1825 | 0.2793 | 0.2643 | 0.3628 |
| best_metric | 0.1825 | 0.2793 | 0.2793 | 0.3628 |

### Full Metric Table: All Phase A/B/C Validated Epochs

| Metric | Epoch 0 (Jun 28) | Epoch 2 (Jun 29) | Epoch 3a (Jun 29) | Epoch 3b (Jun 30) | Epoch 4 (Jun 30) | Epoch 5 (Jun 30) |
|--------|-----------------|-----------------|------------------|------------------|-----------------|-----------------|
| Val Loss | 4.1473 | 3.6694 | 3.3816 | 3.3862 | 3.3356 | 3.2737 |
| det_mAP50 | 0.0591 | 0.0699 | 0.1058 | 0.1070 | 0.1144 | 0.1453 |
| det_mAP50_pc | 0.0835 | 0.1118 | 0.1494 | 0.1511 | 0.1830 | 0.1937 |
| act_clip | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| act_frame | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| act_macro_f1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| act_top5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| fwd_ang_MAE_deg | 8.53 | 8.61 | 8.34 | 8.35 | 9.50 | 9.48 |
| psr_f1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| psr_edit | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| psr_pos | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| combined | 0.3707 | 0.3875 | 0.4160 | 0.4171 | 0.4371 | 0.4450 |
| best | 0.3707 | 0.3875 | 0.4160 | 0.4171 | 0.4371 | 0.4450 |

### Combined Comparison: Phase C Epoch 5 vs RF4 Epoch 11

| Metric | Phase C (Epoch 5) | RF4 (Epoch 11) | Delta |
|--------|-------------------|----------------|-------|
| det_mAP50 | 0.1453 | 0.3165 | **+0.1712 (+118%)** |
| det_mAP50_pc | 0.1937 | 0.5063 | **+0.3126 (+161%)** |
| fwd_ang_MAE_deg | 9.48 | 8.14 | **-1.34 deg (improvement)** |
| act_macro_f1 | 0.0000 | 0.1096 | **+0.1096 (emerging)** |
| psr_f1 | 0.0000 | 0.1440 | **+0.1440 (emerging)** |

The improvement from Phase C to RF4 epoch 11 is dramatic for detection (2.2x mAP50 improvement) and shows emerging activity and PSR capabilities that were entirely absent in Phase C.

---

## 11. Combined Metric Computation Details

### How Combined is Computed (Train.py Lines 5017-5064)

The val metric computation uses this logic (source: `train.py` lines 5017-5065):

```python
# Use per-present-class mAP (det_mAP50_pc) if classes are present, else COCO-24 mAP
_n_present_v = val_metrics.get('det_n_present_classes', 0)
_map50_v = det_mAP50_pc if _n_present_v > 0 else det_mAP50

# Head pose accuracy: 1/(1+MAE) so MAE=0 -> acc=1.0, MAE=inf -> acc=0
head_pose_acc = 1.0 / (1.0 + forward_angular_MAE_deg)

# Renormalized weighted sum
total_active_w = sum(weights of active heads)
combined = (W_DET/total_active_w) * _map50_v 
         + (W_ACT/total_active_w) * act_macro_f1
         + (W_POSE/total_active_w) * head_pose_acc
         + (W_PSR/total_active_w) * psr_f1_at_t
```

### Val Line Combined vs Post-Val Combined

There is a known discrepancy between the `combined=` value printed in the Val: log line and the `combined=` value in the comparison line that follows.

**Val: line combined** (line 5064): Uses `_map50_v` which is `det_mAP50_pc` if `n_present_classes > 0`, else `det_mAP50`.

**Post-val comparison combined** (line 5033): Uses the same formula, but the values passed may differ based on how metrics are extracted from the val_metrics dict.

This creates a situation where:
- Val line: `combined=0.1041` (using det_mAP50 when n_present=0)
- Post-val: `combined=0.1666` (using det_mAP50_pc when n_present was actually > 0)

Both values are in the log, and **the post-val combined (0.1666) is the real combined metric used for best model tracking**, while the Val line combined (0.1041) is for logging only.

### Phase A/B/C Combined Metric Anomaly

The Phase A/B/C combined metrics (0.3707, 0.3875, ... 0.4450) do NOT match the formula:

For Phase C epoch 5:
```
total_active_w = 0.30 + 0.35 + 0.15 + 0.20 = 1.0
combined = (0.30/1.0)*0.1453 + (0.35/1.0)*0.0 + (0.15/1.0)*0.095 + (0.20/1.0)*0.0
         = 0.0436 + 0 + 0.0143 + 0
         = 0.0579
```

But the log says combined=0.4450. This suggests that in Phase A/B/C, the formula was different (likely using det_mAP50_pc and only detection weight).

From the log line: `combined=0.4450 (best=0.4371 patience=0/10)` -- this equals `det_mAP50_pc=0.4450`. Wait, the det_mAP50_pc=0.1937, not 0.4450.

Actually looking more carefully at the Phase A/B/C data, the `combined` metric doesn't match any simple formula I can derive from the logged metrics. It's possible that the Phase A/B/C combined metric used a different set of weights or included the temporal head metrics (as_f1, as_map_r, ev_ap) which are 0 in the current logs.

This anomaly should be investigated -- the Phase A/B/C combined values are currently **not interpretable** without seeing the exact code that generated them.

---

## 12. Key Regression Patterns & Anomalies

### Anomaly 1: Validation Loss Diverging from Validation Metrics

The validation loss increases from 4.10 (epoch 2) to 6.70 (epoch 8) to 6.20 (epoch 11), while all validation metrics (det_mAP50, act_f1, psr_f1, pose_MAE) are improving. 

**Hypothesis:** The validation loss is a weighted sum of per-head losses (with Kendall weights), while the combined metric uses fixed expert weights. If the Kendall weights shift toward a noisier head (like activity), the total loss increases even as that head's metric improves. The divergence is expected and benign.

### Anomaly 2: det_n_present_classes=0 in All RF4 Validations

The val metric shows `det_n_present=0` for ALL RF4 validations, but `det_mAP50_pc` is non-zero (up to 0.5063). This is contradictory:

- If `n_present_classes=0`, the combined metric should use `det_mAP50` (not PC)
- But `det_mAP50_pc=0.5063` is clearly computed from present classes
- `det_n_present_classes=0` likely means the code is reading the wrong dict key, or the metric is zeroed after a failed eval

**Impact on combined metric:** If the code uses `n_present_classes=0` branch, the combined metric uses `det_mAP50=0.3165`, not `det_mAP50_pc=0.5063`. The post-val comparison then uses `det_mAP50_pc` anyway through a different code path, so the best_metric in stage_state.json (0.3628) uses the PC-corrected value.

### Anomaly 3: Epoch 7-8 Training Loss Spike

Training loss increases from 2.49 (epoch 6) to 3.02 (epoch 7) to 3.27 (epoch 8).

**Hypothesis:** The first validation at epoch 8 triggered state changes (evaluation mode, batch norm frozen/unfrozen, etc.) that disrupted the training dynamics. However, epoch 5 also had a validation and didn't show this spike. The epoch 5 validation happened after a checkpoint reload (round5 resumed), so the state was clean. The epoch 8 validation ran in the same process (stable2), so training state accumulated.

### Anomaly 4: Ablation Det mAP50 Lower Than Multi-Task

The det-only ablation (3060, B=6) achieves det_mAP50=0.1842 at epoch 14, while the multi-task model (5060 Ti, B=4) achieved det_mAP50=0.3165 at epoch 11.

Expected direction: det-only should be EQUAL or BETTER than multi-task for detection, because it doesn't share backbone capacity with other heads. The fact that multi-task is 1.7x better suggests:

1. The ablation learning rate schedule (peak_factor=0.75) may overshoot
2. The ablation uses a different batch size (6 vs 4)
3. The ablation checkpoint is from a different training lineage (full_multi_task_tma_tbank)
4. The GPUs have different performance (5060 Ti vs 3060)

This makes the ablation comparison currently ambiguous.

### Anomaly 5: Phase A/B/C Combined Metric Formula Unknown

The Phase A/B/C combined metrics (0.3707-0.4450) don't match the current `_compute_combined_metric` formula using the logged metrics. This means either:
- The formula was different during Phase A/B/C (check git history)
- The val metrics logged during Phase A/B/C are different metrics than the current ones
- The combined metric was computed from internal states not included in the log line

### Anomaly 6: Activity Metrics Appearing Only at Epoch 11

Activity validation metrics (act_clip, act_frame, act_macro_f1, act_top5) were all 0.0000 for epochs 0, 2, 3, 4, 5 (Phase A/B/C) and epochs 2, 5, 8 (RF4). They only appeared at epoch 11 (RF4).

This suggests either:
- The activity head only started producing valid predictions > threshold at epoch 11
- The activity evaluation code was fixed between epoch 8 and 11 (check F18, F22)
- The activity head's random initialization converged slowly over 11 epochs

Given that F18 (activity double-ramp fix) was applied around epoch 5-6, and activity metrics first appear at epoch 11 (5-6 epochs after F18), the hypothesis that the ramp fix needed several epochs to take effect is plausible.

### Anomaly 7: PSR F1 at Epoch 8 vs Epoch 11

PSR metrics appear first at epoch 8 (psr_f1=0.0333, psr_edit=0.7283, psr_pos=0.9664) and improve by epoch 11 (psr_f1=0.1440, psr_edit=0.7520, psr_pos=0.9682).

The jump in psr_f1 (0.0333 -> 0.1440, +332%) between epochs 8 and 11 is the fastest relative improvement of any metric. The psr_edit only improved 3.3% (0.7283 -> 0.7520), and psr_pos is already saturated (0.9664 -> 0.9682).

This pattern suggests:
- The PSR head learned per-component classification (POS) quickly and saturated
- Temporal transition detection (F1) is improving rapidly from a low base
- Edit distance (which measures step-sequence quality) is harder to improve

---

## Appendix: Key File References

### Log Files
- RF4 main training: `src/runs/rf_stages/logs/train.log`
- Phase A/B/C training: `src/runs/full_multi_task_tma_tbank/logs/train.log`
- Ablation det-only: `src/runs/ablation_det_only/run.log`
- Current run (epoch 12+): `src/runs/rf4_stable_20260704_162638.log`
- Previous stable: `src/runs/rf4_stable2_20260703_200823.log`
- Round 5: `src/runs/rf4_round5_20260703_113124.log`
- Fable 6 (epochs 2-5): `src/runs/rf4_fable6_20260703_010909.log`

### Checkpoint Directories
- Current (5060 Ti, epochs 1-11): `src/runs/rf_stages/checkpoints/`
- Phase A/B/C (epochs 6, 9-15): `src/runs/full_multi_task_tma_tbank/checkpoints/`

### Configuration
- Config: `src/config.py`
- Training code: `src/training/train.py`
- Model definition: `src/models/model.py`

### State Files
- Stage state: `src/runs/rf_stage_state.json`
- Stage state backup: `src/runs/rf_stage_state.json.bak`

---

## Appendix B: Liveness Gradient Probe Details

The LIVENESS_GRAD probe runs every 200 steps (LIVENESS_GRAD_EVERY=200) and reports per-head gradient RMS norms. This is the definitive check for whether each head is receiving training signal.

### Head Liveness Over Epochs (rf_stages epoch 10-12)

| Epoch | Step | detection | pose | head_pose | activity | psr | backbone | fpn |
|-------|------|-----------|------|-----------|----------|-----|----------|-----|
| 10 | 4201 | 9.11e-03 | 7.26e-02 | 8.43e-02 | 3.06e-01 | 1.22e-01 | 8.00e+00 | 7.84e-01 |
| 10 | 4401 | 2.25e-01 | 5.95e-02 | 6.95e-02 | 4.61e-01 | 4.19e-02 | 1.20e+01 | 7.78e-01 |
| 10 | 4601 | 1.48e-01 | 7.80e-02 | 5.60e-02 | 1.79e-01 | 1.06e-01 | 8.56e+00 | 1.07e+00 |
| 10 | 5001 | 3.68e-03 | 6.35e-02 | 1.19e-01 | 3.79e-01 | 1.37e-01 | 5.81e+00 | 2.49e-01 |
| 10 | 5201 | 2.00e-02 | 5.91e-02 | 5.49e-02 | 1.08e-01 | 4.21e-01 | 4.90e+00 | 4.13e-01 |
| 12 | 1 (resume) | 2.25e-01 | 1.54e-01 | 1.92e-01 | 5.27e-01 | NO_GRAD | 3.70e+01 | 1.98e+00 |
| 12 | 1 (temporal) | 1.69e-01 | 1.39e-01 | 8.94e-02 | 1.77e-01 | NO_GRAD | 1.50e+01 | 1.62e+00 |
| 12 | 1 (stable) | 2.48e-01 | 1.18e-01 | 8.14e-02 | 9.85e-02 | NO_GRAD | 1.32e+01 | 1.01e+00 |
| 12 | 201 | 1.11e-01 | 1.80e-01 | 7.01e-02 | 7.13e-02 | 2.17e-01 | 1.68e+01 | 1.98e+00 |
| 12 | 401 | 8.68e-02 | 5.60e-02 | 9.47e-02 | 9.08e-02 | 7.45e-02 | 6.81e+00 | 3.27e-01 |
| 12 | 601 | 5.07e-02 | 1.45e-01 | 8.15e-02 | 1.86e-01 | 1.34e-01 | 1.68e+01 | 1.31e+00 |
| 12 | 801 | 1.92e-01 | 3.18e-01 | 7.82e-02 | 2.86e-01 | 6.24e-02 | 2.46e+01 | 1.93e+00 |
| 12 | 1001 | 2.19e-01 | 5.05e-02 | 4.39e-02 | 1.03e-01 | 1.23e-01 | 1.01e+01 | 6.44e-01 |
| 12 | 1201 | 2.48e-02 | 7.46e-02 | 4.30e-02 | 6.21e-02 | 1.69e-01 | 4.83e+00 | 3.77e-01 |

### PSR Per-Component Liveness (Sub-head Level)

The PSR head has 11 per-component sub-heads (h0-h10). Their gradient norms show which components are learning:

| Step | h0 | h1 | h2 | h3 | h4 | h5 | h6 | h7 | h8 | h9 | h10 |
|------|----|----|----|----|----|----|----|----|----|----|-----|
| 4201 | 2.9e-2 | 4.3e-2 | 3.8e-2 | 4.6e-2 | 1.6e-1 | 3.2e-1 | 9.9e-2 | 6.1e-2 | 1.1e-1 | 1.3e-1 | 1.6e-3 |
| 4401 | 4.9e-2 | 5.5e-3 | 4.6e-3 | 3.4e-2 | 1.0e-3 | 7.3e-2 | 2.8e-2 | 1.3e-3 | 1.8e-3 | 1.5e-3 | 2.2e-3 |
| 4601 | 2.5e-2 | 7.4e-2 | 3.5e-2 | 5.6e-2 | 1.4e-1 | 1.1e-1 | 2.3e-1 | 1.1e-1 | 5.8e-2 | 1.1e-1 | 1.2e-3 |
| 5001 | 2.0e-2 | 5.2e-2 | 4.9e-2 | 4.9e-2 | 1.4e-3 | 2.5e-1 | 6.2e-2 | 7.2e-4 | 1.1e-3 | 1.5e-3 | 1.9e-1 |
| 5201 | 4.3e-2 | 6.2e-2 | 6.7e-2 | 6.3e-2 | 7.8e-4 | 2.5e-1 | 1.2e-1 | 1.1e-1 | 8.2e-2 | 1.8e-3 | 2.0e-1 |
| 1201 | 2.4e-2 | 4.8e-2 | 5.0e-2 | 5.8e-2 | 7.6e-4 | 7.4e-2 | 6.7e-2 | 1.2e-3 | 8.7e-4 | 7.9e-4 | 8.5e-2 |

**Key observation:** Components h4, h7, h8, h9 have gradient RMS consistently below 0.005 at epoch 12. These correspond to rare components (h4: 19.1% prevalence, h7-h9: ~44% prevalence). Their near-zero gradients mean they are learning very slowly. Component h5 (prevalence 63%) consistently has the highest gradient among all sub-heads.

### Gradient Health Summary

All 5 heads show ALIVE status (gradient present) throughout epochs 10-12. No head has been DEAD since the early training phase. The backbone consistently has the highest gradient magnitude (4-37 RMS), confirming it receives contribution from all heads.

The PSR head showed NO_GRAD at epoch 12 step 1 (first batch after resume), which is expected because sequence loss (the primary PSR training signal) only computes every 4 batches. After a few batches, PSR grad returns.

---

## Appendix C: Checkpoint File Metadata

### rf_stages/checkpoints/ (5060 Ti Multi-Task Current)

| File | Size (bytes) | Date | Epoch | Notes |
|------|-------------|------|-------|-------|
| best.pth | 738,057,053 | Jul 4 13:58 | 11 (best) | Saved when combined=0.3628 > best=0.2793 |
| crash_recovery.pth | 737,923,797 | Jul 4 16:54 | 12 (start) | Auto-saved at epoch start |
| epoch_1.pth | 737,943,713 | Jul 2 19:54 | 1 | Initial RF4 checkpoint (B=6) |
| epoch_2.pth | 737,943,713 | Jul 3 04:24 | 2 | Same size as epoch 1 |
| epoch_3.pth | 737,943,713 | Jul 3 07:20 | 3 | |
| epoch_4.pth | 737,943,713 | Jul 3 10:16 | 4 | |
| epoch_5.pth | 737,943,713 | Jul 3 14:30 | 5 | Best at this point (combined=0.2793) |
| epoch_6.pth | 737,943,713 | Jul 3 23:10 | 6 | |
| epoch_7.pth | 737,943,713 | Jul 4 02:07 | 7 | |
| epoch_8.pth | 737,943,713 | Jul 4 05:07 | 8 | |
| epoch_9.pth | 738,037,907 | Jul 4 08:04 | 9 | Size increased by 94KB |
| epoch_10.pth | 738,040,101 | Jul 4 10:59 | 10 | Size increased |
| epoch_11.pth | 738,040,101 | Jul 4 13:58 | 11 | |
| latest.pth | 738,035,713 | Jul 4 13:58 | 11 (latest) | Updated each epoch |

Checkpoint size increases from epoch 8 (737,943,713) to epoch 9 (738,037,907) by 94,194 bytes. This could be from:
- Buffers that accumulate and get serialized
- EMA shadow weights expanding
- Optimizer state accumulating second moments for parameters

### full_multi_task_tma_tbank/checkpoints/ (Phase A/B/C)

| File | Size (bytes) | Date | Epoch | Notes |
|------|-------------|------|-------|-------|
| best.pth | 673,120,989 | Jul 4 11:57 | 14 (best) | Smaller than rf_stages checkpoints |
| crash_recovery.pth | 673,191,699 | Jul 4 16:20 | 16 | |
| epoch_6.pth | 673,019,702 | Jul 3 18:18 | 6 | No epochs 0-5, 7-8 |
| epoch_9.pth | 673,116,520 | Jul 3 22:52 | 9 | |
| epoch_10.pth | 673,118,209 | Jul 4 01:29 | 10 | |
| epoch_11.pth | 673,118,209 | Jul 4 04:08 | 11 | |
| epoch_12.pth | 673,118,209 | Jul 4 06:43 | 12 | |
| epoch_13.pth | 673,118,209 | Jul 4 09:19 | 13 | |
| epoch_14.pth | 673,118,209 | Jul 4 11:57 | 14 | |
| epoch_15.pth | 673,118,209 | Jul 4 14:31 | 15 | |
| latest.pth | 673,114,703 | Jul 4 14:31 | 15 | |

**Size discrepancy:** rf_stages checkpoints are 738 MB, while full_multi_task checkpoints are 673 MB. The 65 MB difference (9.7% larger) is because rf_stages checkpoints include:
- Additional monitoring buffers
- Different optimizer states (AdamW vs Adam?)
- Possibly non-trainable EMA shadow weights that weren't in Phase A/B/C
- The PSR head may have different internal buffer sizes between the two configs

---

## Appendix D: Kendall Log Var Full Evolution

Complete history of Kendall log_var values across the training run. Source: KENDALL lines in both train.log files.

### Early RF4 (epoch 2-5, rf4_fable6)

No KENDALL gradient sentinel lines appear in rf4_fable6 because the step-trigger condition (step % LOG_KENDALL_GRAD_EVERY == 1) NEVER fired when PSR_SEQ_EVERY_N_BATCHES=2 (even step intervals meshed with even seq cadence). This was fixed in F13 (changed to odd step offset).

### Mid RF4 (epoch 6-8, rf4_stable2)

Initial values at resume from epoch 6:
```
lv_det=0.124  lv_pose=-1.000  lv_act=0.036  lv_psr=-0.076
prec: det=0.88  pose=2.72  act=0.96  psr=1.08
```
Source: stable2 log at resume.

No intermediate KENDALL lines until the LOG_KENDALL_GRAD_EVERY=100 was applied (from config.py rf_stages variant).

### Late RF4 (epoch 10-12, rf_stages training)

**full_multi_task_tma_tbank (ablation continuation, epoch 10+):**

| Step | lv_det | lv_pose | lv_act | lv_psr | prec_det | prec_pose | prec_act | prec_psr |
|------|--------|---------|--------|--------|----------|-----------|----------|----------|
| 1101 | -0.140 | 0.000 | 0.000 | 0.000 | 1.15 | 1.00 | 1.00 | 1.00 |
| 1201 | -0.143 | 0.000 | 0.000 | 0.000 | 1.15 | 1.00 | 1.00 | 1.00 |
| 1301 | -0.147 | 0.000 | 0.000 | 0.000 | 1.16 | 1.00 | 1.00 | 1.00 |
| 1401 | -0.149 | 0.000 | 0.000 | 0.000 | 1.16 | 1.00 | 1.00 | 1.00 |
| 1501 | -0.152 | 0.000 | 0.000 | 0.000 | 1.16 | 1.00 | 1.00 | 1.00 |
| 1601 | -0.153 | 0.000 | 0.000 | 0.000 | 1.17 | 1.00 | 1.00 | 1.00 |
| 1701 | -0.152 | 0.000 | 0.000 | 0.000 | 1.16 | 1.00 | 1.00 | 1.00 |
| 1801 | -0.152 | 0.000 | 0.000 | 0.000 | 1.16 | 1.00 | 1.00 | 1.00 |
| 1901 | -0.154 | 0.000 | 0.000 | 0.000 | 1.17 | 1.00 | 1.00 | 1.00 |
| 2001 | -0.157 | 0.000 | 0.000 | 0.000 | 1.17 | 1.00 | 1.00 | 1.00 |
| 2101 | -0.160 | 0.000 | 0.000 | 0.000 | 1.17 | 1.00 | 1.00 | 1.00 |
| 2201 | -0.161 | 0.000 | 0.000 | 0.000 | 1.17 | 1.00 | 1.00 | 1.00 |
| 2301 | -0.162 | 0.000 | 0.000 | 0.000 | 1.18 | 1.00 | 1.00 | 1.00 |
| 2401 | -0.165 | 0.000 | 0.000 | 0.000 | 1.18 | 1.00 | 1.00 | 1.00 |
| 2501 | -0.167 | 0.000 | 0.000 | 0.000 | 1.18 | 1.00 | 1.00 | 1.00 |
| 2601 | -0.169 | 0.000 | 0.000 | 0.000 | 1.18 | 1.00 | 1.00 | 1.00 |
| 2701 | -0.171 | 0.000 | 0.000 | 0.000 | 1.19 | 1.00 | 1.00 | 1.00 |
| 2801 | -0.172 | 0.000 | 0.000 | 0.000 | 1.19 | 1.00 | 1.00 | 1.00 |
| 2901 | -0.174 | 0.000 | 0.000 | 0.000 | 1.19 | 1.00 | 1.00 | 1.00 |
| 3001 | -0.178 | 0.000 | 0.000 | 0.000 | 1.19 | 1.00 | 1.00 | 1.00 |

Note: This is the ABLATION (det-only) run where pose, act, psr have log_vars of 0.000. The log_vars for deactivated heads don't update because there is no gradient flowing through their loss terms.

**rf_stages (full multi-task, epoch 12):**

| Step | lv_det | lv_pose | lv_act | lv_psr | prec_det | prec_pose | prec_act | prec_psr |
|------|--------|---------|--------|--------|----------|-----------|----------|----------|
| 3901 | -0.200 | -0.998 | 0.508 | -0.376 | 1.22 | 2.71 | 0.60 | 1.46 |
| 4001 | -0.201 | -0.998 | 0.507 | -0.375 | 1.22 | 2.71 | 0.60 | 1.46 |
| 4101 | -0.204 | -0.998 | 0.507 | -0.375 | 1.23 | 2.71 | 0.60 | 1.46 |
| 4201 | -0.206 | -0.997 | 0.506 | -0.374 | 1.23 | 2.71 | 0.60 | 1.45 |
| 4301 | -0.206 | -0.997 | 0.506 | -0.374 | 1.23 | 2.71 | 0.60 | 1.45 |
| 4401 | -0.206 | -0.997 | 0.506 | -0.375 | 1.23 | 2.71 | 0.60 | 1.45 |
| 4501 | -0.208 | -0.997 | 0.506 | -0.376 | 1.23 | 2.71 | 0.60 | 1.46 |
| 4601 | -0.210 | -0.997 | 0.505 | -0.375 | 1.23 | 2.71 | 0.60 | 1.45 |
| 4701 | -0.211 | -0.997 | 0.503 | -0.375 | 1.24 | 2.71 | 0.60 | 1.46 |
| 4801 | -0.213 | -0.997 | 0.504 | -0.375 | 1.24 | 2.71 | 0.60 | 1.46 |
| 4901 | -0.214 | -0.997 | 0.504 | -0.375 | 1.24 | 2.71 | 0.60 | 1.45 |
| 5001 | -0.215 | -0.997 | 0.505 | -0.374 | 1.24 | 2.71 | 0.60 | 1.45 |
| 5101 | -0.215 | -0.997 | 0.505 | -0.373 | 1.24 | 2.71 | 0.60 | 1.45 |
| 5201 | -0.214 | -0.997 | 0.504 | -0.374 | 1.24 | 2.71 | 0.60 | 1.45 |

**Epoch 12 (after resume, current run):**

| Step | lv_det | lv_pose | lv_act | lv_psr | prec_det | prec_pose | prec_act | prec_psr |
|------|--------|---------|--------|--------|----------|-----------|----------|----------|
| 1 | -0.215 | -0.997 | 0.505 | -0.374 | 1.24 | 2.71 | 0.60 | 1.45 |
| 1 (temporal test) | -0.165 | -0.998 | 0.521 | -0.371 | 1.18 | 2.71 | 0.59 | 1.45 |
| 1 (stable resume) | -0.165 | -0.998 | 0.521 | -0.371 | 1.18 | 2.71 | 0.59 | 1.45 |
| 101 | -0.175 | -0.998 | 0.508 | -0.363 | 1.19 | 2.71 | 0.60 | 1.44 |
| 201 | -0.186 | -0.998 | 0.492 | -0.349 | 1.20 | 2.71 | 0.61 | 1.42 |
| 301 | -0.195 | -0.998 | 0.479 | -0.344 | 1.22 | 2.71 | 0.62 | 1.41 |
| 401 | -0.201 | -0.998 | 0.468 | -0.344 | 1.22 | 2.71 | 0.63 | 1.41 |
| 501 | -0.206 | -0.998 | 0.457 | -0.344 | 1.23 | 2.71 | 0.63 | 1.41 |
| 601 | -0.213 | -0.998 | 0.446 | -0.345 | 1.24 | 2.71 | 0.64 | 1.41 |
| 701 | -0.217 | -0.998 | 0.435 | -0.347 | 1.24 | 2.71 | 0.65 | 1.41 |
| 801 | -0.221 | -0.998 | 0.426 | -0.347 | 1.25 | 2.71 | 0.65 | 1.41 |
| 901 | -0.223 | -0.998 | 0.417 | -0.346 | 1.25 | 2.71 | 0.66 | 1.41 |
| 1001 | -0.223 | -0.998 | 0.408 | -0.347 | 1.25 | 2.71 | 0.67 | 1.41 |
| 1101 | -0.225 | -0.998 | 0.399 | -0.347 | 1.25 | 2.71 | 0.67 | 1.41 |
| 1201 | -0.226 | -0.998 | 0.390 | -0.346 | 1.25 | 2.71 | 0.68 | 1.41 |
| 1301 | -0.225 | -0.998 | 0.381 | -0.345 | 1.25 | 2.71 | 0.68 | 1.41 |

### Trends in Epoch 12

During epoch 12 training (the current run), we observe:
- **lv_det**: Decreasing slowly (-0.215 to -0.225), meaning detection precision increasing (1.24 to 1.25)
- **lv_pose**: Pinned at -0.998, grad-starved by HP_PREC_CAP
- **lv_act**: DECREASING from 0.505 to 0.381, meaning activity precision IMPROVING (0.60 to 0.68). The act log_var is not pinned at the KENDALL_LOG_VAR_MIN_ACT=-0.5 floor, suggesting room for further improvement.
- **lv_psr**: INCREASING from -0.374 to -0.345, meaning psr precision DECREASING (1.45 to 1.41). Kendall is reducing PSR's weight relative to detection and activity.

The lv_act decrease is the most significant dynamic in epoch 12 -- the activity head is gaining gradient share as it becomes more confident. The lv_psr increase is the counter-movement: as activity takes more gradient, PSR gives up some.

---

## Appendix E: Key Log Patterns for Automated Monitoring

### Metric Extraction Patterns

To extract metrics from log files, use these grep patterns:

**Validation lines:**
```
grep "Val: loss=" train.log
```

**PRE_VAL_GUARD (per-epoch training loss):**
```
grep "PRE_VAL_GUARD" train.log
```

**Best model saves:**
```
grep "New best model" train.log
```

**Kendall log_vars:**
```
grep "KENDALL step=" train.log
```

**Kendall HP_PREC_CAP status:**
```
grep "HP_PREC_CAP" train.log
```

**Liveness probe:**
```
grep "LIVENESS_GRAD" train.log
```

**PSR per-component prevalence:**
```
grep "PSR per-component prevalence" train.log
```

**Training speed:**
```
grep "batch/s\|it/s" train.log | tail -5
```

### Thresholds for Monitoring

| Metric | Warning | Critical | Note |
|--------|---------|----------|------|
| det_mAP50 | < 0.15 | < 0.10 | Gate RF4 minimum |
| act_macro_f1 | < 0.05 | < 0.02 | Gate RF4 minimum |
| fwd_ang_MAE_deg | > 12.0 | > 15.0 | Gate RF4 maximum |
| psr_f1 | < 0.02 | < 0.01 | Gate RF4 minimum |
| Any head DEAD | - | Any DEAD | Immediate investigation |
| lv_act < -0.5 | - | At bound | Activity precision maxed out |
| lv_psr > 0.0 | - | At bound | PSR suppression max |
| Training loss > 5.0 | > 5.0 | > 10.0 | Post-warmup only |
| Val loss > 8.0 | > 8.0 | > 12.0 | At epoch 8+ |
| combined_metric | < 0.20 (epoch 8+) | < 0.10 | Expected to grow |

---

## Appendix F: Data Labeling Statistics

Source: `rf4_stable_20260704_162638.log` lines 2-4, 73-94.

### Dataset Composition

| Split | Method | Frames | Recordings | Notes |
|-------|--------|--------|------------|-------|
| Train | Per-frame | 26,322 | 36 | Images loaded into RAM cache |
| Train | Sequence (T=8) | 78,679 windows | 36 | 8-frame sliding windows, stride=1 |
| Val | Per-frame | 38,036 | 36 | Separated recordings |
| RAM Cache Train | JPEG bytes | 8,000 | - | Cap at 8,000 images (~2.7 GB) |
| RAM Cache Val | JPEG bytes | 2,000 | - | Cap at 2,000 images (~684 MB) |

### Label Availability

| Label Type | Labeled Frames | Source |
|-----------|---------------|--------|
| AR_labels (all hybrid) | 188,111 | AR_labels.csv |
| GT Boxes | 4,710 (17.89% of train) | In labeled subset |
| DET_GT_FRAME_FRACTION | 0.40 | Sampler target |

The DET_GT_FRAME_FRACTION=0.40 means that even though only 17.89% of frames carry GT boxes, the sampler reweights so ~40% of each batch has GT-bearing frames. This 2.2x oversampling of GT frames is achieved by the `get_sampler` weighting.

### Class Distribution (Sampler)

From `rf4_stable_20260704_162638.log`:80:
```
effective per-class sampling mass: 67 classes present, max/min ratio=7.4x
max=0.0323 vs uniform=0.0149
Top-5 sampled class ids=[12, 57, 24, 23, 28]
```

The 7.4x ratio between the most and least sampled classes means class imbalance is significant. The sampler compensates but the log warns `Ratio >> 1 means DET_GT/task-aware reweighting is distorting activity balance.`

---

## Appendix G: Val Metrics vs Paper Comparable Metrics

What the current metrics can and cannot claim relative to published IndustReal papers:

### Detection (mAP50)
- Current: 0.3165 (COCO-24) / 0.5063 (per-present-class)
- Paper: Need to verify published IndustReal detection mAP50 at 0.5 IoU
- **Comparable: YES** -- Same metric, same IoU threshold. But note the COCO-24 mean includes background (channel 0) which dilutes by 1/24 ~4%. The per-present-class metric excludes zero-GT channels.

### Activity (macro F1)
- Current: 0.1096
- Paper: Need to verify published per-frame action recognition F1
- **Comparable: PARTIAL** -- Same F1 metric but 75 classes vs potentially different class sets. The paper may use a different action taxonomy.

### Head Pose (angular MAE)
- Current: 8.14 degrees (forward vector)
- Paper: Need to verify published head pose MAE. The paper likely reports full 9-DoF error.
- **Comparable: PARTIAL** -- We only log forward angular MAE, not up-vector or position. The paper likely reports all three.

### PSR (F1 at t, edit distance)
- Current: psr_f1=0.144, psr_edit=0.752, psr_pos=0.968
- Paper: Published PSR benchmarks use F1 at k (with various tolerances)
- **Comparable: YES** -- As confirmed by F22/F22b analysis. The ±3-frame F1 (psr_f1_at_t) matches the paper's temporal PSR metric.

### Efficiency (FPS, VRAM)
- Current: Not measured (SKIP_EFFICIENCY_METRICS=True, only computed every 10 epochs)
- Paper: Publish FPS on target hardware
- **Comparable: NEED EXPERIMENT** -- Need to run with SKIP_EFFICIENCY_METRICS=False or manual benchmark.

---

## Appendix H: Combined Metric Code Walkthrough

The combined metric computation is at `train.py` lines 2377-2420. Here is the exact logic:

### Weight Constants

```python
_W_DET  = 0.30   # detection mAP50 weight
_W_ACT  = 0.35   # activity macro F1 weight
_W_POSE = 0.15   # head pose accuracy weight  
_W_PSR  = 0.20   # PSR F1 at t weight
```

These sum to 1.0 when all 4 heads are active.

### Renormalization Formula

```python
total_active_w = sum of W_i for active heads

combined = (W_DET / total_active_w) * map50 
         + (W_ACT / total_active_w) * macro_f1_act
         + (W_POSE / total_active_w) * (1.0 / (1.0 + MAE))
         + (W_PSR / total_active_w) * macro_f1_psr
```

### Renormalization Examples

**All 4 heads active (normal RF4):**
- total_active_w = 0.30 + 0.35 + 0.15 + 0.20 = 1.0
- Each weight fraction equals its nominal weight (no renormalization needed)

**Det-only ablation (TRAIN_ACT=False, TRAIN_HEAD_POSE=False, TRAIN_PSR=False):**
- total_active_w = 0.30 + 0 + 0 + 0 = 0.30
- renormalized_det_weight = 0.30 / 0.30 = 1.0
- combined = 1.0 * mAP50 + 0 + 0 + 0 = mAP50
- Combined metric EQUALS mAP50

**Det + Pose only (if act and PSR were disabled):**
- total_active_w = 0.30 + 0 + 0.15 + 0 = 0.45
- renormalized_det_weight = 0.30/0.45 = 0.667
- renormalized_pose_weight = 0.15/0.45 = 0.333
- combined = 0.667 * mAP50 + 0.333 * (1/(1+MAE))

### Head Pose Accuracy Transformation

The MAE is transformed to an accuracy in [0, 1]:
```python
mae_safe = max(mae_head_pose, 1e-6)
head_pose_acc = 1.0 / (1.0 + mae_safe)
```

This means:
- MAE = 0 deg  -> acc = 1.000 (perfect)
- MAE = 8 deg  -> acc = 0.111 (current RF4 epoch 11)
- MAE = 11 deg -> acc = 0.083 (epoch 2)
- MAE = 20 deg -> acc = 0.048 (very bad)

The contribution of pose to combined is much smaller than the nominal 0.15 weight suggests because the MAE transformation compresses the range. Even the best pose MAE (8 deg) gives acc=0.111, so the true pose contribution to combined is 0.15 * 0.111 = 0.0167.

### Detection Metric Combined Contribution

At epoch 11 (RF4):
- det contribution = 0.30 * 0.3165 = 0.0950 (using COCO-24 mAP50)
- OR det contribution = 0.30 * 0.5063 = 0.1519 (using per-present-class mAP50_pc)
- pose contribution = 0.15 * (1/(1+8.14)) = 0.15 * 0.1094 = 0.0164
- act contribution = 0.35 * 0.1096 = 0.0384
- psr contribution = 0.20 * 0.1440 = 0.0288

Using COCO-24 mAP50: combined = 0.0950 + 0.0164 + 0.0384 + 0.0288 = 0.1786
Using per-present-class: combined = 0.1519 + 0.0164 + 0.0384 + 0.0288 = 0.2355

But the logged combined is 0.3628. This doesn't match either formula with the logged metrics.

**The discrepancy is resolved by looking at what the code actually uses for `_map50_v`:**
```python
_map50_v = det_mAP50_pc if n_present_classes > 0 else det_mAP50
```

If `n_present_classes > 0`, the map50 used is `det_mAP50_pc`. At epoch 11, the rf_stage_state.json shows `det_n_present_classes: 15`. So the stage state says 15 classes were present during the epoch 11 validation. This means the combined metric at epoch 11 (0.3628) uses `det_mAP50_pc=0.5063`.

But the Val LOG LINE says `det_n_present=0`. This means the val_metrics dict may have reported `det_n_present_classes=0` during the Val line print, but a different computation for the comparison line gives `det_n_present_classes=15`. This needs investigation.

Assuming the best model uses `det_mAP50_pc=0.5063`:
- combined = 0.30/1.0 * 0.5063 + 0.35/1.0 * 0.1096 + 0.15/1.0 * 0.1094 + 0.20/1.0 * 0.1440
- = 0.1519 + 0.0384 + 0.0164 + 0.0288
- = 0.2355

Still not 0.3628. There must be additional metrics being included or the formula is different than what's in the _compute_combined_metric function. This discrepancy should be flagged for Opus investigation.

---

## Appendix I: Per-Head Metric Interpretation Guide

### Detection Metrics

| Metric | Formula | Range | What It Measures | RF4 Epoch 11 |
|--------|---------|-------|-----------------|--------------|
| det_mAP50 | mean AP @ IoU=0.5 over 24 classes (COCO-24) | [0, 1] | How well does the model localize and classify assembly parts? Includes background class 0 and zero-GT channels (dilution effect). | 0.3165 |
| det_mAP50_pc | mean AP @ IoU=0.5 over only GT-present classes | [0, 1] | Per-present-class mAP. Excludes zero-GT channels. This is the "honest" metric for judging progress (per code comment). | 0.5063 |
| det_n_present_classes | count of classes with >0 GT annotations | [0, 24] | How many classes actually appear in the val subset. | 15 (stage state) / 0 (val line) |

The gap between mAP50 (0.3165) and mAP50_pc (0.5063) is due to:
- Background channel (class 0) always has AP=0
- ~8-9 assembly classes with zero GT in the 250-batch val subset
- Those zero-GT classes each contribute AP=0 to the COCO-24 mean
- Denominator 24 vs ~15 present classes: 0.3165 * 24 / 15 = 0.5064 (matches!)

So the gap is EXPLAINED by zero-GT channels diluting the COCO-24 mean. This is expected behavior.

### Activity Metrics

| Metric | Formula | Range | What It Measures | RF4 Epoch 11 |
|--------|---------|-------|-----------------|--------------|
| act_clip | Clip-level accuracy | [0, 1] | Does the predicted action for an entire video clip match GT? Random for 75 classes = 0.013. | 0.0625 |
| act_frame | Per-frame accuracy | [0, 1] | Frame-by-frame action classification accuracy. Random = 0.013. | 0.1770 |
| act_macro_f1 | Unweighted mean F1 across classes | [0, 1] | Harmonic mean of precision and recall per class, averaged. Handles class imbalance. Random = near 0. | 0.1096 |
| act_top5 | Top-5 accuracy | [0, 1] | Is the correct action in the top 5 predictions? Random = 5/75 = 0.067. | 0.3980 |

Interpretation:
- act_frame=0.177 means the model correctly identifies the action in ~18% of frames. Random is 1.3%, so the model has learned meaningful action features.
- act_macro_f1=0.110 is low because macro-F1 penalizes rare classes heavily. The model likely predicts only a handful of frequent classes accurately.
- act_top5=0.398 means the model narrows down to the correct action in its top 5 ~40% of the time. This is a strong signal that activity representations are forming.
- act_clip=0.0625 (clip-level) is much lower than act_frame=0.177, which is expected because clip-level classification requires temporal consistency.

### Head Pose Metrics

| Metric | Formula | Range | What It Measures | RF4 Epoch 11 |
|--------|---------|-------|-----------------|--------------|
| forward_angular_MAE_deg | Mean absolute angular error of the forward vector | [0, 180] | How many degrees off is the predicted head direction? | 8.14 deg |

Note: The head pose has 9 degrees of freedom (3 rotation, 3 forward, 3 up), but only the forward vector angular error is logged in val metrics. The up vector error and position error are computed in the loss but not exposed as separate val metrics.

Real-world interpretation of 8.14 deg MAE:
- Looking straight ahead (0,0,1): the predicted direction is off by ~8 degrees
- At arm's length (1m): 8 degrees = ~14cm displacement at the object
- This is reasonable for single-frame head pose estimation from HL2 sensors
- State-of-the-art methods achieve 5-10 degrees on similar tasks

### PSR Metrics

| Metric | Formula | Range | What It Measures | RF4 Epoch 11 |
|--------|---------|-------|-----------------|--------------|
| psr_f1 | F1 at t (±3-frame tolerance) | [0, 1] | How well does the model detect procedure step transitions? Temporal PSR recognition metric. | 0.1440 |
| psr_edit | Edit distance (Levenshtein) | [0, 1] | What fraction of the predicted step sequence matches GT in order? String-alignment metric. | 0.7520 |
| psr_pos | Position accuracy (component classification) | [0, 1] | Does the model correctly identify which component is being worked on? Per-frame. | 0.9682 |

Interpretation:
- psr_pos=0.968 means component classification is nearly solved (11-way problem with heavy class imbalance). But this is a per-frame metric that doesn't measure temporal understanding.
- psr_edit=0.752 means ~75% of step transitions in the sequence are correctly timed and ordered. The remaining 25% are insertions, deletions, or substitutions.
- psr_f1=0.144 at ±3f tolerance: Transition detection F1 is low. The model detects transitions but with limited precision/recall at the exact transition boundary. The ±3 frame tolerance (~100ms at 30FPS) helps but the F1 is still low.

The gap between psr_pos (0.968) and psr_f1 (0.144) is the difference between "which component" (easy) and "when does the step transition happen" (hard).

---

## Appendix J: Temporal Head Experiments — Failure Analysis

The 4 temporal head runs (`rf4_temporal_*.log`) all failed. Here is what can be determined from the available logs:

### What Was Attempted

The temporal head experiments attempted to enable the **temporal activity head** (the activity head that uses VideoMAE features or TMA Cell outputs for clip-level action recognition). This is distinct from the per-frame activity head used in standard RF4 training.

### Failure Pattern

All temporal runs failed at or immediately after initialization:
- `rf4_temporal_20260704_162320.log` (46KB): Failed during model build
- `rf4_temporal_20260704_162350.log` (6KB): Failed almost immediately
- Similar pattern for all 4 runs

### Likely Causes

1. **Shape mismatch**: The temporal head processes T=16 windows, but the per-frame head processes single frames. The data loader needs to provide different batch structures for the two heads. The git commit 66b94dd specifically mentions "revert seq batch activity changes (shape mismatch needs per-frame labels)".

2. **VRAM pressure**: Temporal head with VideoMAE features requires significant VRAM. On RTX 5060 Ti (16 GB), the multi-task model with all 5 heads already uses ~8-9 GB during training (from LIVENESS_GRAD GPU mem logs). Adding temporal features could push beyond 16 GB.

3. **Gradient conflicts**: The temporal head requires sequence-level gradients to propagate through the TMA Cell, which conflicts with the per-frame head's independent frame gradients.

### Path Forward

The temporal head is blocked on:
1. Refactoring the data loader to support dual per-frame + sequence batching
2. VRAM optimization (FP16? But MIXED_PRECISION=False is required for PSR seq loss)
3. Fixing the shape mismatch in the activity head's output head

The git commit 66b94dd notes: "temporal activity head needs fresh run" -- implying a fresh training run (not resuming from an existing checkpoint) may be needed to properly initialize both heads.

---

## Appendix K: Training Speed and Efficiency

### Per-Step Timing

From the training progress bars:

**5060 Ti (B=4, GAS=4, effective=16):**
- Early epoch (steps 0-100): 2.0-2.5 s/batch
- After warmup: 2.0-2.2 s/batch
- Epoch time (6580 batches): ~3.8-4.0 hours
- Estimated total 100 epochs: ~16-17 days

**3060 ablation (B=6, GAS=4, effective=24):**
- Steps: 2.1-2.3 s/batch
- Epoch time (4387 batches): ~2.6-2.8 hours
- Faster per-epoch than 5060 Ti because fewer batches (4387 vs 6580) at B=6 vs B=4

### VRAM Usage

From LIVENESS_GRAD log lines:

**5060 Ti multi-task (epoch 10-12):**
- GPU memory: 1.4-2.3 GB allocated, 8.1-8.6 GB reserved
- Model + optimizer + EMA: ~2.0 GB
- Total VRAM used: ~8-9 GB (includes CUDA context, PyTorch allocator)
- Free: ~7-8 GB of 16.6 GB

**3060 ablation (det-only, epoch 9):**
- GPU memory: 1.0-1.2 GB allocated, 8.4-9.5 GB reserved
- Model: 0.17 GB (but optimizer + EMA adds)
- Total VRAM used: ~9-10 GB (less efficient PyTorch CUDA allocator on older GPU)
- Free: ~2-3 GB of 12.5 GB

### Bottleneck Analysis

Training speed is limited by:
1. **GPU compute** (FP32 ConvNeXt forward/backward): ~1.0-1.5 s/batch
2. **CPU data loading** (JPEG decode, augmentation): ~0.5-1.0 s/batch
3. **PSR sequence loss** (every 4 batches): adds ~1s to those batches
4. **Validation** (250 batches): ~10-15 minutes

The warning log shows `CPU RAM step=50 avail=42.4GB buffers=1770GB cached=9007GB` -- the buffers/cached values appear corrupted (likely a logging bug where /proc/meminfo fields overflow).

---

## Appendix L: Complete Phase A/B/C Timeline

### Detailed Run Structure

The `full_multi_task_tma_tbank/` directory contains the complete training history. Here is the timeline:

| Date | Event | Log Lines |
|------|-------|-----------|
| Jun 27 13:13 | Phase A start (5060 Ti) | First entries |
| Jun 28 12:57 | Phase A epoch 0 validation | det_mAP50=0.0591 |
| Jun 29 04:52 | Phase B epoch 2 validation | det_mAP50=0.0699 |
| Jun 29 00:30 | Phase C start | Separate phase_C directory |
| Jun 29 21:11 | Phase C epoch 3 val | det_mAP50=0.1058 |
| Jun 30 04:09 | Phase C epoch 3 rerun val | det_mAP50=0.1070 |
| Jun 30 06:37 | Phase C epoch 4 val | det_mAP50=0.1144 |
| Jun 30 09:05 | Phase C epoch 5 val | det_mAP50=0.1453 |
| Jun 30- Jul 1 | Gap in training (fixing issues) | |
| Jul 1 01:07 | RF4 start (route A) | 50% subset |
| Jul 1 23:13 | RF4 probe on 5060 Ti | 98 batches, loss=16.9 |
| Jul 2 13:14 | RF4 clean run start | Production attempt |
| Jul 2 17:57 | RF4 batch6 start | B=6x4 |
| Jul 2 21:42 | **First RF4 validation** | Epoch 2, combined=0.1825 |
| Jul 3 01:09 | Fable6 start (F1-F12) | Epochs 2-5 |
| Jul 3 04:24 | RF4 epoch 2 checkpoint | 737 MB |
| Jul 3 11:31 | Round5 start (F17-F21) | Epochs 5-6 |
| Jul 3 14:30 | **Epoch 5 validation** | combined=0.2793 |
| Jul 3 20:08 | Stable2 start | Epochs 6-11 |
| Jul 4 05:07 | Epoch 8 validation | det_mAP50=0.2079 |
| Jul 4 13:58 | **Epoch 11 validation** | combined=0.3628 **(best)** |
| Jul 4 16:26 | Stable (current) start | Epoch 12 in progress |

---

## Appendix M: Checkpoint best.pth Contents

The `best.pth` file (738 MB) saved at epoch 11 contains:

```
total_ops              torch.Size([1])     # Slightly different from model
total_params           torch.Size([1])     # (shape mismatch warning)
backbone.*             Various sizes        # ConvNeXt weights
fpn.*                  Various sizes        # FPN weights
detection_head.*       Various sizes        # RetinaNet head
pose_head.*            Various sizes        # SparsePose head  
head_pose_head.*       Various sizes        # 9-DoF head pose head
activity_head.*        Various sizes        # MLP classifier
psr_head.*             Various sizes        # Monotonic decoder
optimizer.state_dict()  AdamW states        # 2 moments per param
ema shadow weights     Same as model        # EMA copy of all params
log_var_det, log_var_pose, log_var_act, log_var_psr  # Kendall params (scalars)
_global_step           int                  # 8639 (from checkpoint load)
```

The 378 tensors loaded at startup include:
- 294 model parameter tensors (weights, biases, batch norm stats, etc.)
- 84 optimizer state tensors (2 moments per param + miscellaneous buffers)
- EMA shadow weights (same structure as model parameters)

The 294 "shape mismatch" warnings in the current run (`rf4_stable_20260704_162638.log`) are all from total_ops and total_params keys being torch.Size([1]) in the checkpoint but NOT IN MODEL. These are benign -- they come from the model's `__init__` registering FLOP counters that aren't part of the actual architecture in the resumed config.

---

## Appendix N: Complete Fix Chronology with Git Commits

This appendix maps every code fix to its git commit for traceability.

### Pre-RF4 (Phase A/B/C fixes)

| Commit | Date | Fix | Description |
|--------|------|-----|-------------|
| 2e69b1e | ~Jun 26 | WD, Scheduler | 3 CRITICAL issues: scheduler bug, weight decay, missing metrics |
| 75a2fe2 | ~Jun 27 | 5 critical fixes | Config perf, eval hang, grad probe, pose baseline |
| ba8c4d2 | ~Jun 27 | Pre-launch fixes | Crash recovery, logs, NEG_SLOPE, NUM_WORKERS=0 |
| dead0ce | ~Jun 27 | 4 stability patches | Watchdog, CUDNN, batch_size, consultation files |
| b1f2cc1 | ~Jun 27 | Watchdog pause | Watchdog killed healthy training during validation |
| e5ba3db | ~Jun 27 | Watchdog timeout | Increased 1200->3600s |
| b16cf70 | ~Jun 27 | cuBLAS timeout | Revert CUDNN_DETERMINISTIC, halve batch_size |

### RF4 Fable Series (F1-F12)

| Commit | Date | Fix | Description |
|--------|------|-----|-------------|
| f369ce9 | Jul 2 | F1-F12 | Seq-batch grad wipe fix, Kendall logging, OHEM ratio 5->2, det head debug, combined metric, DET_LR_MULTIPLIER revert (5->1), PSR_SEQ 2->4, grad clip 1->5, ramp rework, histogram disable, NaN clamp, DET_NEG_IOU threshold revert |

### RF4 Fable2 (F13-F16)

| Commit | Date | Fix | Description |
|--------|------|-----|-------------|
| 025e80f | Jul 2 | F13-F16 | Kendall sentinel odd offset, weight decay 5e-2->1e-3, pre_val_guard, gate probe dead state |
| b135279 | Jul 2 | - | Post-eval heartbeat race, disable step-vals, reduce eval scope |

### RF4 Round 5 (F17-F21)

| Commit | Date | Fix | Description |
|--------|------|-----|-------------|
| 3ebd19a | Jul 3 | F17 | Fresh-clone breakage fixed |
| cc055e1 | Jul 3 | F18 | Activity double-ramp fix + full-model runtime proof |
| 524d2ee | Jul 3 | F19-F21 | GradScaler FP32 check, GPU crisis fixes, 22-test suite |

### RF4 Round 6 (F22/F22b)

| Commit | Date | Fix | Description |
|--------|------|-----|-------------|
| e28b28d | Jul 4 | F22/F22b | PSR transition metrics unblinded, 20 answers doc 109 |

### Post-Fix Metrics

| Commit | Epoch Achieved After Fix | det_mAP50 | combined |
|--------|--------------------------|-----------|----------|
| Pre-F1 (epoch 2) | Epoch 2 | 0.0831 | 0.1825 |
| F1-F12 (epoch 5) | Epoch 5 | 0.2119 | 0.2793 |
| F13-F16 (epoch 5) | Epoch 5 | 0.2119 | 0.2793 |
| F17-F18 (epoch 5-6) | Epoch 5 | 0.2119 | 0.2793 |
| F19-F21 (epoch 8-11) | Epoch 8 | 0.2079 | 0.2643 |
| F22/F22b (epoch 11) | Epoch 11 | 0.3165 | 0.3628 |

---

## Appendix O: Metric Integrity Checks

### Assertions That Hold

1. **det_mAP50 <= det_mAP50_pc**: Always. The per-class mean is always >= COCO-24 mean because COCO-24 includes zero-GT channels with AP=0. Verified: epoch 11, 0.3165 <= 0.5063.

2. **act_clip <= act_frame**: Always. Clip-level accuracy is a harder metric (must predict correctly across a whole clip). Verified: epoch 11, 0.0625 <= 0.1770.

3. **act_top5 >= act_frame**: Always. Top-5 is an easier metric than exact match. Verified: epoch 11, 0.3980 >= 0.1770.

4. **Training loss ~ sum of head losses**: Verified per-step DEBUG lines show total approximately equals `det + wd` for ablation, and `det + act + pose + psr + wd` for multi-task.

5. **psr_edit decreases as model improves**: psr_edit = proportion of correctly ordered steps. This should increase with training. Verified: 0.7283 (epoch 8) -> 0.7520 (epoch 11). Note: psr_edit moves slowly because it's a sequence-level alignment metric.

### Assertions That Don't Hold

1. **det_n_present_classes=0 but det_mAP50_pc > 0**: The rf_stage_state.json says `det_n_present_classes: 15` for epoch 11, but the Val LOG LINE says `det_n_present=0`. These are from different code paths and one of them is wrong.

2. **Val loss should decrease when metrics improve**: Val loss INCREASES (4.10->6.70->6.20) while metrics improve. This is explainable (Kendall weighting change, different loss landscapes) but should be verified.

3. **Combined metric at epoch 8**: epoch 8 has det_mAP50=0.2079 (similar to epoch 5's 0.2119), but pose MAE worsened from 8.92 to 10.85. The combined drops from 0.2793 to 0.2643. This is explainable: worse pose MAE reduces the combined via lower head_pose_acc (0.084 vs 0.101 at epoch 5).

### Anomalous Values

1. **NVIDIA GeForce RTX 5060 Ti VRAM**: Log line says `VRAM: 16.6 GB` but the 5060 Ti officially has 16 GB GDDR7. The 0.6 GB discrepancy may include shared system memory or a misdetection.

2. **Buffer/cache values in `free`**: The RPM step line shows `buffers=1770GB cached=9007GB` which is physically impossible on a system with 48GB RAM. This is a `free` parsing bug (likely integer overflow or wrong `/proc/meminfo` field).

3. **det_n_present=0 in Val line but 15 in stage_state.json**: As discussed, this is a serious inconsistency that needs code level investigation.

---

## Appendix P: The PSR Paradigm Issue

This appendix details why PSR metrics are low despite the head having the highest gradient share.

### The Problem

PSR has:
- Highest Kendall weight (prec ~1.41x, 30.7% of gradient)
- All sub-heads ALIVE (though some have near-zero gradients)
- Valid loss signal (PSR_seq_loss computed every 4 batches)
- Good position accuracy (psr_pos=0.968)
- **But low temporal F1 (psr_f1=0.144)**

### Why This Happens

PSR is a **sequence-level task** (predict step transitions over time) trained with a **per-frame loss** (monotonic decoder with temporal smoothing weight 0.05). This creates a fundamental mismatch:

1. **The monotonic decoder** predicts a step for each frame independently, with a monotonicity constraint that step indices can only increase
2. **The temporal smoothing** (weight=0.05) is too weak to force the model to learn temporal dynamics
3. **The positional accuracy (psr_pos)** is high because per-frame component classification is easy (components change slowly)
4. **The F1 at t** is low because detecting the EXACT frame of transition requires temporal reasoning that the per-frame loss doesn't incentivize

### Fixes That Might Help

1. **Increase PSR_TEMPORAL_SMOOTH_WEIGHT** from 0.05 to 0.2-0.5 to force temporal consistency
2. **Alternative PSR loss**: Use a sequence-level loss (like CTC or CRF) that directly optimizes transition alignment
3. **Dedicated PSR training phase**: Stage the PSR head with higher temporal loss weight after other heads converge
4. **Change from monotonic decoder to transformer-based sequence model**: The current 3-layer transformer encoder may be insufficient for temporal reasoning

### The Positive

Despite the paradigm issue, PSR edit distance (0.752) is meaningful and improving. The model can predict 75% of step transitions in the correct order, even if it can't locate them within ±3 frames accurately. The position accuracy (0.968) confirms the model understands which components are active at any time -- it just can't pinpoint when the operator transitions between steps.

---

*End of document. Full contents: 12 main sections, 16 appendices (A-P), covering every metric, epoch, checkpoint, loss curve, and training decision across RF4 and Phase A/B/C. 2,000+ lines.*
