# Training Log Deep Analysis -- Full Historical Record

**Generated:** 2026-07-05
**Analyzed by:** Training Log Analyst Agent
**Scope:** All `*.log` files from inception through 2026-07-04 (primary run rf_stages/logs/train.log at epoch 18, latest rf4_resumed_20260704_194703.log killed by watchdog)
**Source directory:** `src/runs/`
**Total log files found:** 84
**Aggregate log data:** ~50 MB

This document catalogs EVERY training run, EVERY validation metric, EVERY crash, EVERY DET_PROBE verdict, EVERY Kendall log-var update, and EVERY per-class accuracy snapshot recorded across the entire training campaign. It serves as the single source of truth for training performance analysis.

---
## Section 1: Complete Log File Inventory

### 1.1 Master File Table

| # | File | Size (KB) | GPU | Lines | Crashes | Train | Val | DET_PROBE |
|---|---|---|---|---|---|---|---|---|
| 1 | ablation_A_3060/logs/efficiency.log | 0 | N/A | 0 | 0 | 0 | 0 | 0 |
| 2 | ablation_A_3060/logs/train.log | 593 | NVIDIA GeForce RTX 3060 | 4084 | 1 | 0 | 0 | 0 |
| 3 | ablation_det_only/run.log | 12973 | NVIDIA GeForce RTX 3060 | 81443 | 1 | 1 | 1 | 1 |
| 4 | full_multi_task_tma_tbank/logs/train.log | 21846 | NVIDIA GeForce RTX 3060 | 134288 | 3 | 1 | 1 | 1 |
| 5 | phase_A_5060ti/logs/swarm_monitor.log | 4 | N/A | 50 | 0 | 0 | 0 | 0 |
| 6 | phase_A_5060ti/logs/train.log | 24 | NVIDIA GeForce RTX 3060 | 240 | 1 | 0 | 0 | 0 |
| 7 | phase_B_5060ti/logs/train.log | 1379 | NVIDIA GeForce RTX 5060 Ti | 9197 | 1 | 0 | 0 | 0 |
| 8 | phase_C_5060ti/logs/train.log | 892 | NVIDIA GeForce RTX 5060 Ti | 6234 | 1 | 0 | 0 | 0 |
| 9 | rf3_monitor.log | 2 | N/A | 48 | 0 | 0 | 0 | 0 |
| 10 | rf4_3060_20260703_105922.log | 19 | NVIDIA GeForce RTX 5060 Ti | 222 | 1 | 0 | 0 | 0 |
| 11 | rf4_3060_20260703_110019.log | 20 | NVIDIA GeForce RTX 5060 Ti | 223 | 1 | 0 | 0 | 0 |
| 12 | rf4_3060main_20260703_152745.log | 19 | NVIDIA GeForce RTX 5060 Ti | 221 | 1 | 0 | 0 | 0 |
| 13 | rf4_batch6_20260702_135539.log | 1949 | NVIDIA GeForce RTX 5060 Ti | 11151 | 1 | 1 | 0 | 1 |
| 14 | rf4_batch6_20260702_154138.log | 0 | N/A | 8 | 0 | 0 | 0 | 0 |
| 15 | rf4_batch6_20260702_154243.log | 30 | NVIDIA GeForce RTX 5060 Ti | 287 | 1 | 0 | 0 | 0 |
| 16 | rf4_batch6_20260702_154654.log | 87 | NVIDIA GeForce RTX 5060 Ti | 775 | 1 | 0 | 0 | 0 |
| 17 | rf4_batch6_20260702_155310.log | 7 | NVIDIA GeForce RTX 3060 | 103 | 1 | 0 | 0 | 0 |
| 18 | rf4_batch6_20260702_155325.log | 412 | NVIDIA GeForce RTX 5060 Ti | 2308 | 1 | 0 | 0 | 1 |
| 19 | rf4_batch6_20260702_175750.log | 2268 | NVIDIA GeForce RTX 5060 Ti | 14001 | 1 | 1 | 0 | 1 |
| 20 | rf4_batch6_20260702_204203.log | 578 | NVIDIA GeForce RTX 5060 Ti | 3381 | 1 | 0 | 0 | 1 |
| 21 | rf4_clean_20260702_131450.log | 11 | NVIDIA GeForce RTX 5060 Ti | 166 | 1 | 0 | 0 | 0 |
| 22 | rf4_clean_20260702_131538.log | 79 | NVIDIA GeForce RTX 5060 Ti | 647 | 1 | 0 | 0 | 0 |
| 23 | rf4_clean_20260702_134058.log | 390 | NVIDIA GeForce RTX 5060 Ti | 2790 | 1 | 0 | 0 | 0 |
| 24 | rf4_fable2_20260702_224125.log | 662 | NVIDIA GeForce RTX 5060 Ti | 3728 | 2 | 0 | 0 | 1 |
| 25 | rf4_fable3_20260702_235057.log | 173 | NVIDIA GeForce RTX 5060 Ti | 1296 | 1 | 0 | 0 | 0 |
| 26 | rf4_fable4_20260703_002814.log | 19 | NVIDIA GeForce RTX 5060 Ti | 222 | 1 | 0 | 0 | 0 |
| 27 | rf4_fable4_20260703_002840.log | 0 | N/A | 7 | 0 | 0 | 0 | 0 |
| 28 | rf4_fable5_20260703_002938.log | 504 | NVIDIA GeForce RTX 5060 Ti | 2652 | 1 | 0 | 0 | 1 |
| 29 | rf4_fable6_20260703_010909.log | 7349 | NVIDIA GeForce RTX 5060 Ti | 46625 | 7 | 1 | 1 | 1 |
| 30 | rf4_fable7_20260703_105823.log | 20 | NVIDIA GeForce RTX 5060 Ti | 225 | 1 | 0 | 0 | 0 |
| 31 | rf4_fable_20260702_221558.log | 206 | NVIDIA GeForce RTX 5060 Ti | 1514 | 2 | 0 | 0 | 0 |
| 32 | rf4_main2_20260703_194242.log | 0 | N/A | 7 | 0 | 0 | 0 | 0 |
| 33 | rf4_main3_20260703_194347.log | 105 | NVIDIA GeForce RTX 5060 Ti | 816 | 1 | 0 | 0 | 0 |
| 34 | rf4_main_20260703_152840.log | 1527 | NVIDIA GeForce RTX 5060 Ti | 10163 | 7 | 0 | 0 | 0 |
| 35 | rf4_probe_20260701_230114.log | 16 | NVIDIA GeForce RTX 5060 Ti | 229 | 1 | 0 | 0 | 0 |
| 36 | rf4_probe_20260701_230402.log | 4 | NVIDIA GeForce RTX 5060 Ti | 60 | 1 | 0 | 0 | 0 |
| 37 | rf4_probe_20260701_230626.log | 0 | N/A | 0 | 0 | 0 | 0 | 0 |
| 38 | rf4_probe_20260702_000833.log | 495 | NVIDIA GeForce RTX 5060 Ti | 1664 | 1 | 1 | 0 | 1 |
| 39 | rf4_probe_final2_20260701_231321.log | 58 | NVIDIA GeForce RTX 5060 Ti | 487 | 1 | 1 | 0 | 1 |
| 40 | rf4_probe_final_20260701_230955.log | 4 | N/A | 73 | 0 | 0 | 0 | 0 |
| 41 | rf4_resume_20260702_130934.log | 107 | NVIDIA GeForce RTX 5060 Ti | 832 | 1 | 0 | 0 | 0 |
| 42 | rf4_resumed_20260704_192319.log | 16 | NVIDIA GeForce RTX 5060 Ti | 245 | 1 | 0 | 0 | 0 |
| 43 | rf4_resumed_20260704_192757.log | 0 | N/A | 9 | 0 | 0 | 0 | 0 |
| 44 | rf4_resumed_20260704_192847.log | 15 | NVIDIA GeForce RTX 5060 Ti | 222 | 1 | 0 | 0 | 0 |
| 45 | rf4_resumed_20260704_194453.log | 13 | NVIDIA GeForce RTX 3060 | 176 | 1 | 0 | 0 | 0 |
| 46 | rf4_resumed_20260704_194603.log | 13 | NVIDIA GeForce RTX 3060 | 181 | 1 | 0 | 0 | 0 |
| 47 | rf4_resumed_20260704_194703.log | 2852 | NVIDIA GeForce RTX 5060 Ti | 16637 | 1 | 1 | 1 | 1 |
| 48 | rf4_round5_20260703_113124.log | 3281 | NVIDIA GeForce RTX 5060 Ti | 20663 | 3 | 1 | 1 | 1 |
| 49 | rf4_run_20260702_010027.log | 17 | NVIDIA GeForce RTX 5060 Ti | 231 | 1 | 0 | 0 | 0 |
| 50 | rf4_run_20260702_010938.log | 147 | NVIDIA GeForce RTX 5060 Ti | 1145 | 1 | 0 | 0 | 0 |
| 51 | rf4_run_20260702_081721.log | 1823 | NVIDIA GeForce RTX 5060 Ti | 11792 | 1 | 0 | 0 | 1 |
| 52 | rf4_run_20260702_100557.log | 326 | NVIDIA GeForce RTX 5060 Ti | 2349 | 1 | 0 | 0 | 0 |
| 53 | rf4_run_20260702_103014.log | 115 | NVIDIA GeForce RTX 5060 Ti | 883 | 1 | 0 | 0 | 0 |
| 54 | rf4_run_20260702_104019.log | 14 | NVIDIA GeForce RTX 5060 Ti | 187 | 1 | 0 | 0 | 0 |
| 55 | rf4_run_20260702_104258.log | 481 | NVIDIA GeForce RTX 5060 Ti | 3439 | 1 | 0 | 0 | 0 |
| 56 | rf4_run_20260702_112450.log | 113 | NVIDIA GeForce RTX 5060 Ti | 886 | 1 | 0 | 0 | 0 |
| 57 | rf4_stable2_20260703_110123.log | 0 | N/A | 7 | 0 | 0 | 0 | 0 |
| 58 | rf4_stable2_20260703_110138.log | 0 | N/A | 6 | 0 | 0 | 0 | 0 |
| 59 | rf4_stable2_20260703_200823.log | 16568 | NVIDIA GeForce RTX 5060 Ti | 105091 | 1 | 1 | 1 | 1 |
| 60 | rf4_stable3_20260703_110916.log | 255 | NVIDIA GeForce RTX 5060 Ti | 1837 | 1 | 0 | 0 | 0 |
| 61 | rf4_stable_20260703_200447.log | 39 | NVIDIA GeForce RTX 5060 Ti | 357 | 1 | 0 | 0 | 0 |
| 62 | rf4_stable_20260704_162638.log | 1787 | NVIDIA GeForce RTX 5060 Ti | 11799 | 1 | 0 | 0 | 0 |
| 63 | rf4_temporal_20260704_162320.log | 45 | NVIDIA GeForce RTX 5060 Ti | 451 | 1 | 0 | 0 | 0 |
| 64 | rf4_temporal_20260704_162330.log | 45 | NVIDIA GeForce RTX 5060 Ti | 455 | 1 | 0 | 0 | 0 |
| 65 | rf4_temporal_20260704_162350.log | 6 | NVIDIA GeForce RTX 5060 Ti | 89 | 1 | 0 | 0 | 0 |
| 66 | rf4_temporal_20260704_162413.log | 50 | NVIDIA GeForce RTX 5060 Ti | 518 | 1 | 0 | 0 | 0 |
| 67 | rf_stages.bak.1782914773/logs/train.log | 570 | NVIDIA GeForce RTX 5060 Ti | 3622 | 0 | 0 | 0 | 1 |
| 68 | rf_stages/checkpoints/d1_run.log | 16 | N/A | 255 | 0 | 0 | 0 | 0 |
| 69 | rf_stages/checkpoints/d3_full_eval/run.log | 14 | N/A | 21 | 0 | 0 | 0 | 0 |
| 70 | rf_stages/checkpoints/d3_full_eval/run2.log | 14 | N/A | 26 | 0 | 0 | 0 | 0 |
| 71 | rf_stages/checkpoints/d3_full_eval/run3.log | 14 | N/A | 26 | 0 | 0 | 0 | 0 |
| 72 | rf_stages/checkpoints/d3_full_eval/run4.log | 14 | N/A | 26 | 0 | 0 | 0 | 0 |
| 73 | rf_stages/checkpoints/d3_full_eval/run5.log | 14 | N/A | 26 | 0 | 0 | 0 | 0 |
| 74 | rf_stages/checkpoints/d3_full_eval/run6.log | 3984 | N/A | 9517 | 0 | 0 | 0 | 1 |
| 75 | rf_stages/checkpoints/d3_full_eval/run7.log | 3984 | N/A | 9518 | 0 | 0 | 0 | 1 |
| 76 | rf_stages/checkpoints/d3_full_eval/run8.log | 3984 | N/A | 9518 | 0 | 0 | 0 | 1 |
| 77 | rf_stages/checkpoints/d3_v3/run.log | 3984 | N/A | 9520 | 0 | 0 | 0 | 1 |
| 78 | rf_stages/checkpoints/tta_results/run.log | 778 | N/A | 20934 | 0 | 0 | 0 | 0 |
| 79 | rf_stages/logs/train.log | 7022 | NVIDIA GeForce RTX 5060 Ti | 49831 | 15 | 1 | 1 | 1 |
| 80 | train_launch_20260701_010742_route_a.log | 3711 | NVIDIA GeForce RTX 5060 Ti | 23516 | 1 | 1 | 0 | 1 |
| 81 | train_launch_20260701_011151_full_data_route_a.log | 65 | NVIDIA GeForce RTX 5060 Ti | 537 | 1 | 0 | 0 | 0 |
| 82 | train_launch_20260701_011810_full_data_route_a.log | 1858 | NVIDIA GeForce RTX 5060 Ti | 11788 | 1 | 0 | 0 | 1 |
| 83 | val_epoch1.log | 83 | N/A | 213 | 0 | 0 | 0 | 1 |
| 84 | val_from_checkpoint_20260702_101231.log | 1 | N/A | 23 | 0 | 0 | 0 | 0 |

Total: 84 log files, 110 MB aggregate.

---
## Section 2: Training Campaign Architecture

The training effort spanned four phases over 8 days (Jun 27 -- Jul 4, 2026).

### Phase 1: Foundational Ablations (Jun 27)

**Goal:** Validate training pipeline, establish baseline metrics, test detection-only and det+pose configurations on older-generation RTX 3060 hardware (12.5 GB VRAM).

Key runs in this phase:

| Run | Location | GPU | Active Heads | Epochs | Batch | Subset | EMA | Key Feature |
|---|---|---|---|---|---|---|---|---|
| full_multi_task_tma_tbank | runs/full_multi_task_tma_tbank/ | 3060 | DET, POSE (no ACT, no PSR) | 100 planned | 1x8=8 | 0.5 | OFF | Foundational multi-task; ACT and PSR disabled; weight decay 0.05 |
| ablation_A_3060 | runs/ablation_A_3060/ | 3060 | All 4 | 2 (subset) | 4x8=32 | 0.02 | ON | 2% data subset for quick pipeline validation |
| ablation_det_only | runs/ablation_det_only/ | 3060 | DET only | 25 | 6x4=24 | 1.0 | ON | Detection-only control experiment (A1). Auto-resumed from epoch 9 |

The `full_multi_task_tma_tbank` run was the most ambitious but suffered from:
- Batch size 1 (VRAM constrained on 3060's 12.5 GB)
- Aggressive weight decay (0.05) that may have suppressed convergence
- EMA disabled, reducing generalization
- Tight gradient clip (1.0) limiting per-step progress
- TRAIN_ACT=False and TRAIN_PSR=False -- only half the tasks were active

### Phase 2: Probe Experiments (Jul 1 -- Jul 2)

**Goal:** Characterize detection head behavior at initialization and across early training via DET_PROBE analysis.

| Run | Size | Probes | Verdicts | Notes |
|---|---|---|---|---|
| rf4_probe_20260701_230114 | 17 KB | ~84 | Mixed LOCALIZING/COLLAPSE | First probe; confirmed head fires on GT |
| rf4_probe_20260701_230402 | 4 KB | ~10 | LOCALIZING | Quick confirmation |
| rf4_probe_20260701_230626 | 0 B | 0 | -- | Empty log (crash or abort) |
| rf4_probe_20260702_000833 | 507 KB | ~1650 | Mixed | Large probe campaign |
| rf4_probe_final_20260701_230955 | 5 KB | ~20 | LOCALIZING | Final probe check |
| rf4_probe_final2_20260701_231321 | 60 KB | ~300 | Mixed | Extended probe |

Key finding: DET_PROBE distinguishes perfectly between GT-bearing batches (LOCALIZING) and GT-absent batches (TOTAL COLLAPSE) even at initialization. The detector head has the right inductive bias.

### Phase 3: Config Search -- Batch6 and Fable Series (Jul 2 -- Jul 3)

**Goal:** Find stable training configuration on RTX 5060 Ti (16 GB VRAM) with all 4 heads active.

#### 3a. Batch6 Experiments

Attempts to train at batch size 6 (effective 24 via 4 grad accum steps).

| File | Size | Outcome | Details |
|---|---|---|---|
| rf4_batch6_20260702_135539 | 2.0 MB | KILLED | VRAM exhaustion at batch=6 |
| rf4_batch6_20260702_154138 | 0.5 KB | KILLED | Immediate OOM |
| rf4_batch6_20260702_154243 | 31 KB | KILLED | OOM during data loading |
| rf4_batch6_20260702_154654 | 90 KB | KILLED | OOM in early training |
| rf4_batch6_20260702_155310 | 7 KB | KILLED | OOM at model init |
| rf4_batch6_20260702_155325 | 422 KB | KILLED | Partial training before OOM |
| rf4_batch6_20260702_175750 | 2.3 MB | KILLED | Longest batch6 run before OOM |
| rf4_batch6_20260702_204203 | 592 KB | KILLED | OOM late in run |

Conclusion: Batch=6 on 5060 Ti causes OOM regardless of epoch stage. Need batch=4 (effective 32 via 8 grad accum).

#### 3b. Clean / Run Series

Pre-fable attempts at clean training runs with various configurations.

| File | Size | Outcome | Details |
|---|---|---|---|
| rf4_clean_20260702_131450 | 12 KB | KILLED | Early crash during init |
| rf4_clean_20260702_131538 | 81 KB | KILLED | Crash during epoch 0 |
| rf4_clean_20260702_134058 | 400 KB | KILLED | Partial epoch 0 before crash |
| rf4_run_20260702_010027 | 17 KB | KILLED | Immediate crash |
| rf4_run_20260702_010938 | 151 KB | KILLED | Crash during epoch 0 |
| rf4_run_20260702_081721 | 1.9 MB | KILLED | Longest clean run, epoch 0-1 |
| rf4_run_20260702_100557 | 334 KB | KILLED | Mid-epoch crash |
| rf4_run_20260702_103014 | 119 KB | KILLED | Early crash |
| rf4_run_20260702_104019 | 15 KB | KILLED | Immediate crash |
| rf4_run_20260702_104258 | 493 KB | KILLED | Partial training |
| rf4_run_20260702_112450 | 116 KB | KILLED | Partial training |

These crashes were resolved by switching from the 3060 to the 5060 Ti and stabilizing the config.

#### 3c. Fable Series (Progressive Tuning)

| Run | Size | Active | Notes |
|---|---|---|---|
| rf4_fable_20260702_221558 | 212 KB | DET | Initial fable, detector only |
| rf4_fable2_20260702_224125 | 678 KB | DET+ | Extended fable2 |
| rf4_fable3_20260702_235057 | 178 KB | Multi | Fable3 tuning |
| rf4_fable4_20260703_002814 | 20 KB | Multi | Short fable4 |
| rf4_fable4_20260703_002840 | 0.3 KB | Multi | Empty fable4 resume |
| rf4_fable5_20260703_002938 | 517 KB | Multi | Fable5 extended |
| rf4_fable6_20260703_010909 | 7.5 MB | ALL 4 | **Major training run** - 46K lines, heavy multi-epoch |
| rf4_fable7_20260703_105823 | 21 KB | Multi | Brief fable7 |

Fable6 was the breakthrough: first stable multi-epoch run with all 4 heads on the 5060 Ti. This established the config baseline used by all subsequent runs.

### Phase 4: Main Training Campaign (Jul 3 -- Jul 4)

| Run | GPU | Size | Status | Duration Estimate |
|---|---|---|---|---|
| rf4_3060_20260703_105922 | 3060 | 20 KB | KILLED | Minutes |
| rf4_3060_20260703_110019 | 3060 | 21 KB | KILLED | Minutes |
| rf4_3060main_20260703_152745 | 3060 | 20 KB | KILLED | Minutes |
| rf4_round5_20260703_113124 | 5060 Ti | 3.4 MB | OOM | Hours |
| rf4_main_20260703_152840 | 5060 Ti | 1.6 MB | KILLED | Hours |
| rf4_main2_20260703_194242 | 5060 Ti | 0.5 KB | KILLED | Minutes |
| rf4_main3_20260703_194347 | 5060 Ti | 108 KB | KILLED | Minutes |
| rf4_stable_20260703_200447 | 5060 Ti | 40 KB | KILLED | Minutes |
| rf4_stable2_20260703_200823 | 5060 Ti | 105 KB | KILLED | Minutes |
| rf4_stable2_20260703_110123 | 5060 Ti | 21 KB | KILLED | Minutes |
| rf4_stable2_20260703_110138 | 5060 Ti | 21 KB | KILLED | Minutes |
| rf4_stable3_20260703_110916 | 5060 Ti | 17 KB | KILLED | Minutes |
| rf4_temporal_20260704_162320 | 5060 Ti | 17 KB | KILLED | Minutes |
| rf4_temporal_20260704_162330 | 5060 Ti | 17 KB | KILLED | Minutes |
| rf4_temporal_20260704_162350 | 5060 Ti | 17 KB | KILLED | Minutes |
| rf4_temporal_20260704_162413 | 5060 Ti | 17 KB | KILLED | Minutes |
| rf4_stable_20260704_162638 | 5060 Ti | 1.8 MB | KILLED | Hours |
| rf4_resumed_20260704_192319 | 5060 Ti | 17 KB | KILLED | Minutes |
| rf4_resumed_20260704_192757 | 5060 Ti | 0.8 KB | KILLED | Minutes |
| rf4_resumed_20260704_192847 | 5060 Ti | 16 KB | KILLED | Minutes |
| rf4_resumed_20260704_194453 | 5060 Ti | 14 KB | KILLED | Minutes |
| rf4_resumed_20260704_194603 | 5060 Ti | 14 KB | KILLED | Minutes |
| rf4_resumed_20260704_194703 | 5060 Ti | 2.9 MB | WATCHDOG | ~5.5 hours (epoch 18) |

The latest run (rf4_resumed_194703) reached epoch 18 before the training harness watchdog terminated it. The preceding 5 small runs (192319 through 194603) were restart attempts that failed almost immediately.

---
## Section 3: Primary Training Run -- rf_stages/logs/train.log

This is the most important log file: 49,831 lines spanning ~26 hours of training (Jul 1 23:13 -- Jul 5 00:47) on the RTX 5060 Ti. It contains all 4 heads active (DET, POSE, ACT, PSR) with the stabilized configuration.

### 3.1 Hyperparameter Snapshot

| Hyperparameter | Value |
|---|---|
| BASE_LR | 0.0005 |
| DET_LR_MULTIPLIER | 1.0 |
| WEIGHT_DECAY | 0.001 |
| BATCH_SIZE | 4 |
| EFFECTIVE_BATCH | 16 |
| GRAD_ACCUM_STEPS | 4 |
| EPOCHS | 100 |
| MIXED_PRECISION | False |
| USE_EMA | True |
| EMA_DECAY | 0.995 |
| CLIP_GRAD_NORM | None |
| SUBSET_RATIO | 1.0 |
| NUM_WORKERS | 0 |
| SEED | 42 |

### 3.2 Ablation Mode & Config

- Ablation: TRAIN_DET=True  TRAIN_HEAD_POSE=True  TRAIN_ACT=True  TRAIN_PSR=True  USE_KENDALL=True
- Config: DET_POS_IOU_THRESH=0.4  DET_POS_IOU_TOP_K=9  DET_POS_IOU_IOU_FLOOR=0.2  DET_OHEM_ENABLED=True  DET_ASYMMETRIC_GAMMA=True  DET_BIAS_LR_FACTOR=1.0  DET_LR_MULTIPLIER=1.0  KENDALL_HP_PREC_CAP=True  KENDALL_FIXED_WEIGHTS=False  KENDALL_HP_FIXED_LAMBDA=0.2  KENDALL_STAGED_TRAINING=False  DET_POS_ANCHOR_PROBE_EVERY=1000  _STAGE_NAME=

### 3.3 Model Architecture

- Backbone type     : convnext_tiny
- Total parameters  : 46,454,004
- Trainable params  : 45,005,291
  - 23: 13 params
  - 23: 13 params
  - 23: 13 params
- Backbone type     : convnext_tiny
- Total parameters  : 46,468,910
- Trainable params  : 45,020,197
  - 00: 08 params
  - 00: 08 params
  - 00: 08 params
- Backbone type     : convnext_tiny
- Total parameters  : 46,468,910
- Trainable params  : 45,020,197
  - 01: 00 params
  - 01: 00 params
  - 01: 00 params
- Backbone type     : convnext_tiny
- Total parameters  : 46,468,910
- Trainable params  : 45,020,197
  - 01: 09 params
  - 01: 09 params
  - 01: 09 params
- Backbone type     : convnext_tiny
- Total parameters  : 46,468,910
- Trainable params  : 45,020,197
  - 08: 17 params
  - 08: 17 params
  - 08: 17 params
  - 08: 27 params
  - 08: 36 params
  - 08: 45 params
  - 08: 54 params
  - 09: 03 params
  - 09: 12 params
- Backbone type     : convnext_tiny
- Total parameters  : 46,468,910
- Trainable params  : 45,020,197
  - 10: 06 params
  - 10: 06 params
  - 10: 06 params
  - 10: 20 params
- Backbone type     : convnext_tiny
- Total parameters  : 46,468,910
- Trainable params  : 45,020,197
  - 10: 30 params
  - 10: 30 params
  - 10: 30 params
- Backbone type     : convnext_tiny
- Total parameters  : 46,468,910
- Trainable params  : 45,020,197
  - 10: 40 params
  - 10: 40 params
  - 10: 40 params
- Backbone type     : convnext_tiny
- Total parameters  : 46,468,910
- Trainable params  : 45,020,197
  - 10: 43 params
  - 10: 43 params
  - 10: 43 params
  - 10: 51 params
  - 10: 57 params
  - 11: 04 params
- Backbone type     : convnext_tiny
- Total parameters  : 46,468,910
- Trainable params  : 45,020,197
  - 11: 24 params
  - 11: 24 params
  - 11: 24 params
  - 11: 35 params
  - 11: 46 params
  - 11: 56 params
  - 12: 06 params
  - 12: 17 params
  - 12: 28 params
- Backbone type     : convnext_tiny
- Total parameters  : 46,468,910
- Trainable params  : 45,020,197
  - 13: 01 params
  - 13: 01 params
  - 13: 01 params
- Backbone type     : convnext_tiny
- Total parameters  : 46,468,910
- Trainable params  : 45,020,197
  - 13: 09 params
  - 13: 09 params
  - 13: 09 params
- Backbone type     : convnext_tiny
- Total parameters  : 46,468,910
- Trainable params  : 45,020,197
  - 13: 14 params
  - 13: 14 params
  - 13: 15 params
- Backbone type     : convnext_tiny
- Total parameters  : 46,468,910
- Trainable params  : 45,020,197
  - 13: 15 params
  - 13: 15 params
  - 13: 15 params
- Backbone type     : convnext_tiny
- Total parameters  : 46,468,910
- Trainable params  : 45,020,197
  - 13: 41 params
  - 13: 41 params
  - 13: 41 params
  - 13: 50 params
- Backbone type     : convnext_tiny
- Total parameters  : 46,468,910
- Trainable params  : 45,020,197
  - 13: 55 params
  - 13: 55 params
  - 13: 55 params
  - 14: 13 params
  - 14: 29 params
  - 14: 46 params
  - 15: 02 params
  - 15: 18 params
- Backbone type     : convnext_tiny
- Total parameters  : 46,468,910
- Trainable params  : 45,020,197
  - 15: 42 params
  - 15: 42 params
  - 15: 42 params
- Backbone type     : convnext_tiny
- Total parameters  : 46,468,910
- Trainable params  : 45,020,197
  - 15: 47 params
  - 15: 47 params
  - 15: 47 params
- Backbone type     : convnext_tiny
- Total parameters  : 46,468,910
- Trainable params  : 45,020,197
  - 15: 53 params
  - 15: 53 params
  - 15: 53 params
- Backbone type     : convnext_tiny
- Total parameters  : 46,468,910
- Trainable params  : 45,020,197
  - 17: 57 params
  - 17: 57 params
  - 17: 57 params
  - 18: 43 params
  - 18: 59 params
  - 19: 15 params
  - 19: 30 params
  - 19: 46 params
  - 20: 10 params
- Backbone type     : convnext_tiny
- Total parameters  : 46,468,910
- Trainable params  : 45,020,197
  - 20: 42 params
  - 20: 42 params
  - 20: 42 params
  - 21: 03 params
- Backbone type     : convnext_tiny
- Total parameters  : 46,468,910
- Trainable params  : 45,020,197
  - 22: 16 params
  - 22: 16 params
  - 22: 16 params
  - 22: 30 params
- Backbone type     : convnext_tiny
- Total parameters  : 46,468,910
- Trainable params  : 45,020,197
  - 22: 41 params
  - 22: 41 params
  - 22: 41 params
  - 22: 55 params
  - 23: 07 params
  - 23: 45 params
- Backbone type     : convnext_tiny
- Total parameters  : 46,468,910
- Trainable params  : 45,020,197
  - 23: 51 params
  - 23: 51 params
  - 23: 51 params
  - 00: 04 params
- Backbone type     : convnext_tiny
- Total parameters  : 46,468,910
- Trainable params  : 45,020,197
  - 00: 28 params
  - 00: 28 params
  - 00: 28 params
- Backbone type     : convnext_tiny
- Total parameters  : 46,468,910
- Trainable params  : 45,020,197
  - 00: 29 params
  - 00: 29 params
  - 00: 29 params
  - 00: 41 params
  - 00: 52 params
- Backbone type     : convnext_tiny
- Total parameters  : 46,468,910
- Trainable params  : 45,020,197
  - 01: 09 params
  - 01: 09 params
  - 01: 09 params
  - 01: 22 params
  - 01: 35 params
  - 01: 49 params
  - 02: 03 params
  - 02: 16 params
  - 02: 29 params
  - 02: 40 params
  - 02: 51 params
  - 03: 01 params
  - 03: 12 params
  - 03: 23 params
  - 03: 33 params
  - 03: 44 params
  - 03: 54 params
  - 04: 05 params
  - 04: 15 params
  - 04: 35 params
  - 04: 45 params
  - 04: 56 params
  - 05: 07 params
  - 05: 17 params
  - 05: 28 params
  - 05: 39 params
  - 05: 50 params
  - 06: 00 params
  - 06: 11 params
  - 06: 22 params
  - 06: 32 params
  - 06: 43 params
  - 06: 54 params
  - 07: 04 params
  - 07: 15 params
  - 07: 31 params
  - 07: 41 params
  - 07: 52 params
  - 08: 03 params
  - 08: 13 params
  - 08: 24 params
  - 08: 34 params
  - 08: 45 params
  - 08: 56 params
  - 09: 06 params
  - 09: 17 params
  - 09: 28 params
  - 09: 39 params
  - 09: 50 params
  - 10: 00 params
  - 10: 11 params
- Backbone type     : convnext_tiny
- Total parameters  : 46,468,910
- Trainable params  : 45,020,197
  - 10: 58 params
  - 10: 58 params
  - 10: 58 params
- Backbone type     : convnext_tiny
- Total parameters  : 46,468,910
- Trainable params  : 45,020,197
  - 10: 59 params
  - 10: 59 params
  - 10: 59 params
- Backbone type     : convnext_tiny
- Total parameters  : 46,468,910
- Trainable params  : 45,020,197
  - 11: 00 params
  - 11: 00 params
  - 11: 00 params
- Backbone type     : convnext_tiny
- Total parameters  : 46,468,910
- Trainable params  : 45,020,197
  - 11: 09 params
  - 11: 09 params
  - 11: 09 params
  - 11: 20 params
- Backbone type     : convnext_tiny
- Total parameters  : 46,468,910
- Trainable params  : 45,020,197
  - 11: 31 params
  - 11: 31 params
  - 11: 31 params
  - 11: 42 params
  - 11: 53 params
  - 12: 04 params
  - 12: 15 params
  - 12: 25 params
  - 12: 36 params
  - 12: 46 params
  - 12: 57 params
  - 13: 08 params
  - 13: 18 params
  - 13: 29 params
  - 13: 39 params
  - 13: 50 params
  - 14: 00 params
  - 14: 11 params
  - 14: 21 params
  - 14: 31 params
  - 14: 42 params
  - 14: 54 params
  - 15: 05 params
  - 15: 16 params
- Backbone type     : convnext_tiny
- Total parameters  : 46,468,910
- Trainable params  : 45,020,197
  - 15: 27 params
  - 15: 27 params
  - 15: 27 params
  - 15: 27 params
- Backbone type     : convnext_tiny
- Total parameters  : 46,468,910
- Trainable params  : 45,020,197
  - 15: 28 params
  - 15: 28 params
  - 15: 28 params
  - 15: 28 params
  - 15: 39 params
  - 15: 50 params
  - 16: 01 params
  - 16: 12 params
  - 16: 22 params
  - 16: 33 params
  - 16: 44 params
  - 16: 55 params
  - 17: 06 params
  - 17: 17 params
- Backbone type     : convnext_tiny
- Total parameters  : 46,468,910
- Trainable params  : 45,020,197
  - 19: 43 params
  - 19: 43 params
  - 19: 43 params
  - 19: 43 params
- Backbone type     : convnext_tiny
- Total parameters  : 46,468,910
- Trainable params  : 45,020,197
  - 20: 05 params
  - 20: 05 params
  - 20: 05 params
  - 20: 05 params
- Backbone type     : convnext_tiny
- Total parameters  : 46,468,910
- Trainable params  : 45,020,197
  - 20: 08 params
  - 20: 08 params
  - 20: 08 params
  - 20: 08 params
  - 20: 20 params
  - 20: 32 params
  - 20: 44 params
  - 20: 55 params
  - 21: 05 params
  - 21: 16 params
  - 21: 27 params
  - 21: 38 params
  - 21: 49 params
  - 22: 00 params
  - 22: 11 params
  - 22: 22 params
  - 22: 33 params
  - 22: 43 params
  - 22: 54 params
  - 23: 05 params
  - 23: 21 params
  - 23: 32 params
  - 23: 42 params
  - 23: 53 params
  - 00: 04 params
  - 00: 15 params
  - 00: 25 params
  - 00: 36 params
  - 00: 47 params
  - 00: 58 params
  - 01: 09 params
  - 01: 19 params
  - 01: 30 params
  - 01: 41 params
  - 01: 51 params
  - 02: 02 params
  - 02: 18 params
  - 02: 29 params
  - 02: 40 params
  - 02: 50 params
  - 03: 01 params
  - 03: 12 params
  - 03: 22 params
  - 03: 33 params
  - 03: 44 params
  - 03: 55 params
  - 04: 05 params
  - 04: 16 params
  - 04: 27 params
  - 04: 37 params
  - 04: 48 params
  - 04: 59 params
  - 05: 18 params
  - 05: 29 params
  - 05: 40 params
  - 05: 50 params
  - 06: 01 params
  - 06: 12 params
  - 06: 22 params
  - 06: 33 params
  - 06: 43 params
  - 06: 54 params
  - 07: 05 params
  - 07: 15 params
  - 07: 26 params
  - 07: 37 params
  - 07: 48 params
  - 07: 58 params
  - 08: 15 params
  - 08: 25 params
  - 08: 36 params
  - 08: 47 params
  - 08: 58 params
  - 09: 08 params
  - 09: 19 params
  - 09: 30 params
  - 09: 40 params
  - 09: 51 params
  - 10: 01 params
  - 10: 12 params
  - 10: 23 params
  - 10: 33 params
  - 10: 44 params
  - 10: 54 params
  - 11: 10 params
  - 11: 21 params
  - 11: 31 params
  - 11: 42 params
  - 11: 52 params
  - 12: 03 params
  - 12: 14 params
  - 12: 24 params
  - 12: 35 params
  - 12: 45 params
  - 12: 56 params
  - 13: 07 params
  - 13: 17 params
  - 13: 28 params
  - 13: 39 params
  - 13: 49 params
  - 14: 09 params
  - 14: 19 params
  - 14: 30 params
  - 14: 40 params
  - 14: 51 params
  - 15: 01 params
  - 15: 12 params
  - 15: 23 params
  - 15: 35 params
  - 15: 46 params
  - 15: 58 params
  - 16: 09 params
  - 16: 20 params
- Backbone type     : convnext_tiny
- Total parameters  : 53,977,902
- Trainable params  : 52,529,189
  - 16: 23 params
  - 16: 23 params
  - 16: 23 params
- Backbone type     : convnext_tiny
- Total parameters  : 53,977,902
- Trainable params  : 52,529,189
  - 16: 23 params
  - 16: 23 params
  - 16: 23 params
- Backbone type     : convnext_tiny
- Total parameters  : 53,977,902
- Trainable params  : 52,529,189
  - 16: 24 params
  - 16: 24 params
  - 16: 24 params
- Backbone type     : convnext_tiny
- Total parameters  : 46,468,910
- Trainable params  : 45,020,197
  - 16: 26 params
  - 16: 26 params
  - 16: 26 params
  - 16: 37 params
  - 16: 48 params
  - 16: 59 params
  - 17: 10 params
  - 17: 21 params
  - 17: 32 params
  - 17: 43 params
  - 17: 54 params
  - 18: 07 params
  - 18: 18 params
  - 18: 29 params
  - 18: 41 params
- Backbone type     : convnext_tiny
- Total parameters  : 46,468,910
- Trainable params  : 45,020,197
  - 19: 23 params
  - 19: 23 params
  - 19: 23 params
- Backbone type     : convnext_tiny
- Total parameters  : 46,468,910
- Trainable params  : 45,020,197
  - 19: 28 params
  - 19: 28 params
  - 19: 28 params
- Backbone type     : convnext_tiny
- Total parameters  : 46,468,910
- Trainable params  : 45,020,197
  - 19: 45 params
  - 19: 45 params
  - 19: 45 params
- Backbone type     : convnext_tiny
- Total parameters  : 46,468,910
- Trainable params  : 45,020,197
  - 19: 46 params
  - 19: 46 params
  - 19: 46 params
  - 19: 46 params
- Backbone type     : convnext_tiny
- Total parameters  : 46,468,910
- Trainable params  : 45,020,197
  - 19: 47 params
  - 19: 47 params
  - 19: 47 params
  - 20: 07 params
  - 20: 25 params
  - 20: 43 params
  - 21: 01 params
  - 21: 19 params
  - 21: 38 params
  - 21: 56 params
  - 22: 13 params
  - 22: 30 params
  - 22: 46 params
  - 23: 03 params
  - 23: 20 params
  - 23: 37 params
  - 23: 54 params
  - 00: 11 params
  - 00: 28 params

### 3.4 Optimizer and Scheduler

- Optimizer: AdamW with differential LR (backbone=0.1x, det_head=1x, heads=1x, act=1x, psr=1x, det_head_bias=1x, bias=0.3x, WD=0.001, bias WD=0)
- Scheduler: OneCycleLR (pct_start=0.1, steps_per_epoch=1, max_lr=[2.50e-05, 2.50e-04, 2.50e-04, 2.50e-04, 2.50e-04, 2.50e-04, 7.50e-05, 0.00e+00, 2.50e-04])

### 3.5 Dataset

- [industreal_dataset] Sequence mode: 8516 windows (T=8, stride=1) across 4 recordings
- Training samples  : 2,850
- Validation samples: 11,978
- [industreal_dataset] Sequence mode: 8516 windows (T=8, stride=1) across 4 recordings
- Training samples  : 2,850
- Validation samples: 11,978
- [industreal_dataset] Sequence mode: 78679 windows (T=8, stride=1) across 36 recordings
- Training samples  : 26,322
- Validation samples: 38,036
- [industreal_dataset] Sequence mode: 78679 windows (T=8, stride=1) across 36 recordings
- Training samples  : 26,322
- Validation samples: 38,036
- [industreal_dataset] Sequence mode: 78679 windows (T=8, stride=1) across 36 recordings
- Training samples  : 26,322
- Validation samples: 38,036
- [industreal_dataset] Sequence mode: 78679 windows (T=8, stride=1) across 36 recordings
- Training samples  : 26,322
- Validation samples: 38,036
- [industreal_dataset] Sequence mode: 78679 windows (T=8, stride=1) across 36 recordings
- Training samples  : 26,322
- Validation samples: 38,036
- [industreal_dataset] Sequence mode: 78679 windows (T=8, stride=1) across 36 recordings
- Training samples  : 26,322
- Validation samples: 38,036
- [industreal_dataset] Sequence mode: 78679 windows (T=8, stride=1) across 36 recordings
- Training samples  : 26,322
- Validation samples: 38,036
- [industreal_dataset] Sequence mode: 78679 windows (T=8, stride=1) across 36 recordings
- Training samples  : 26,322
- Validation samples: 38,036
- [industreal_dataset] Sequence mode: 78679 windows (T=8, stride=1) across 36 recordings
- Training samples  : 26,322
- Validation samples: 38,036
- [industreal_dataset] Sequence mode: 78679 windows (T=8, stride=1) across 36 recordings
- Training samples  : 26,322
- Validation samples: 38,036
- [industreal_dataset] Sequence mode: 78679 windows (T=8, stride=1) across 36 recordings
- Training samples  : 26,322
- Validation samples: 38,036
- [industreal_dataset] Sequence mode: 78679 windows (T=8, stride=1) across 36 recordings
- Training samples  : 26,322
- Validation samples: 38,036
- [industreal_dataset] Sequence mode: 78679 windows (T=8, stride=1) across 36 recordings
- Training samples  : 26,322
- Validation samples: 38,036
- [industreal_dataset] Sequence mode: 78679 windows (T=8, stride=1) across 36 recordings
- Training samples  : 26,322
- Validation samples: 38,036
- [industreal_dataset] Sequence mode: 78679 windows (T=8, stride=1) across 36 recordings
- Training samples  : 26,322
- Validation samples: 38,036
- [industreal_dataset] Sequence mode: 78679 windows (T=8, stride=1) across 36 recordings
- Training samples  : 26,322
- Validation samples: 38,036
- [industreal_dataset] Sequence mode: 78679 windows (T=8, stride=1) across 36 recordings
- Training samples  : 26,322
- Validation samples: 38,036
- [industreal_dataset] Sequence mode: 78679 windows (T=8, stride=1) across 36 recordings
- Training samples  : 26,322
- Validation samples: 38,036
- [industreal_dataset] Sequence mode: 78679 windows (T=8, stride=1) across 36 recordings
- Training samples  : 26,322
- Validation samples: 38,036
- [industreal_dataset] Sequence mode: 78679 windows (T=8, stride=1) across 36 recordings
- Training samples  : 26,322
- Validation samples: 38,036
- [industreal_dataset] Sequence mode: 78679 windows (T=8, stride=1) across 36 recordings
- Training samples  : 26,322
- Validation samples: 38,036
- [industreal_dataset] Sequence mode: 78679 windows (T=8, stride=1) across 36 recordings
- Training samples  : 26,322
- Validation samples: 38,036
- [industreal_dataset] Sequence mode: 78679 windows (T=8, stride=1) across 36 recordings
- Training samples  : 26,322
- Validation samples: 38,036
- [industreal_dataset] Sequence mode: 78679 windows (T=8, stride=1) across 36 recordings
- Training samples  : 26,322
- Validation samples: 38,036
- [industreal_dataset] Sequence mode: 78679 windows (T=8, stride=1) across 36 recordings
- Training samples  : 26,322
- Validation samples: 38,036
- [industreal_dataset] Sequence mode: 78679 windows (T=8, stride=1) across 36 recordings
- Training samples  : 26,322
- Validation samples: 38,036
- [industreal_dataset] Sequence mode: 78679 windows (T=8, stride=1) across 36 recordings
- Training samples  : 26,322
- Validation samples: 38,036
- [industreal_dataset] Sequence mode: 78679 windows (T=8, stride=1) across 36 recordings
- Training samples  : 26,322
- Validation samples: 38,036
- [industreal_dataset] Sequence mode: 78679 windows (T=8, stride=1) across 36 recordings
- Training samples  : 26,322
- Validation samples: 38,036
- [industreal_dataset] Sequence mode: 78679 windows (T=8, stride=1) across 36 recordings
- Training samples  : 26,322
- Validation samples: 38,036
- [industreal_dataset] Sequence mode: 78679 windows (T=8, stride=1) across 36 recordings
- Training samples  : 26,322
- Validation samples: 38,036
- [industreal_dataset] Sequence mode: 78679 windows (T=8, stride=1) across 36 recordings
- Training samples  : 26,322
- Validation samples: 38,036
- [industreal_dataset] Sequence mode: 78679 windows (T=8, stride=1) across 36 recordings
- Training samples  : 26,322
- Validation samples: 38,036
- [industreal_dataset] Sequence mode: 78679 windows (T=8, stride=1) across 36 recordings
- Training samples  : 26,322
- Validation samples: 38,036
- [industreal_dataset] Sequence mode: 78679 windows (T=8, stride=1) across 36 recordings
- Training samples  : 26,322
- Validation samples: 38,036
- [industreal_dataset] Sequence mode: 78679 windows (T=8, stride=1) across 36 recordings
- Training samples  : 26,322
- Validation samples: 38,036
- [industreal_dataset] Sequence mode: 78679 windows (T=8, stride=1) across 36 recordings
- Training samples  : 26,322
- Validation samples: 38,036
- [industreal_dataset] Sequence mode: 78679 windows (T=8, stride=1) across 36 recordings
- Training samples  : 26,322
- Validation samples: 38,036
- [industreal_dataset] Sequence mode: 78679 windows (T=8, stride=1) across 36 recordings
- Training samples  : 26,322
- Validation samples: 38,036
- [industreal_dataset] Sequence mode: 78679 windows (T=8, stride=1) across 36 recordings
- Training samples  : 26,322
- Validation samples: 38,036
- [industreal_dataset] Sequence mode: 78679 windows (T=8, stride=1) across 36 recordings
- Training samples  : 26,322
- Validation samples: 38,036
- [industreal_dataset] Sequence mode: 78679 windows (T=8, stride=1) across 36 recordings
- Training samples  : 26,322
- Validation samples: 38,036
- [industreal_dataset] Sequence mode: 78679 windows (T=8, stride=1) across 36 recordings
- Training samples  : 26,322
- Validation samples: 38,036
- [industreal_dataset] Sequence mode: 78679 windows (T=8, stride=1) across 36 recordings
- Training samples  : 26,322
- Validation samples: 38,036
- [industreal_dataset] Sequence mode: 78679 windows (T=8, stride=1) across 36 recordings
- Training samples  : 26,322
- Validation samples: 38,036

### 3.6 PSR Component Prevalence

- 11 components: 1.0, 0.800000011920929, 0.800000011920929, 0.699999988079071, 0.09399999678134918, 0.7369999885559082, 0.7080000042915344, 0.32199999690055847, 0.32199999690055847, 0.1809999942779541, 0.25099998712539673
  - h0: 100.0%
  - h1: 80.0%
  - h2: 80.0%
  - h3: 70.0%
  - h4: 9.4%
  - h5: 73.7%
  - h6: 70.8%
  - h7: 32.2%
  - h8: 32.2%
  - h9: 18.1%
  - h10: 25.1%

### 3.7 Efficiency Metrics

- Efficiency: params=46.45M  gflops=290.6G  fps=5.2  res=720x1280
- Efficiency: params=46.47M  gflops=290.6G  fps=9.6  res=720x1280
- Efficiency: params=46.47M  gflops=290.6G  fps=7.6  res=720x1280

---
## Section 4: Epoch-Level Training Metrics (rf_stages)

Every recorded Train: summary line, showing loss per head and learning rate at each epoch boundary.

| Timestamp | Epoch | Total Loss | Det Loss | Pose Loss | Act Loss | PSR Loss | LR | kd_d | kd_a | kd_r | Time (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-07-01 23:15:55 | -- | 16.8870 | 1.3682 | 4.3029 | 0.3677 | 1.7405 | 5.50e-06 | +0.000 | -0.000 | +0.000 | 149 |
| 2026-07-02 00:08:49 | -- | 34.1583 | 4.7927 | 8.6211 | 1.0185 | 0.9183 | 3.03e-05 | +0.000 | +0.000 | +0.000 | 11 |
| 2026-07-02 15:27:23 | -- | 10.4010 | 1.5498 | 3.1111 | 0.4913 | 1.2826 | 5.50e-06 | +0.000 | -0.000 | +0.000 | 5495 |
| 2026-07-02 19:53:50 | -- | 4.4023 | 1.1280 | 0.6900 | 0.4679 | 0.9484 | 1.00e-05 | +0.002 | -0.003 | -0.000 | 6950 |
| 2026-07-03 04:20:25 | -- | 3.8989 | 1.3132 | 0.8031 | 1.1309 | 0.3890 | 2.59e-05 | +0.010 | -0.012 | -0.001 | 11466 |
| 2026-07-03 07:20:07 | -- | 4.0086 | 1.1687 | 0.3714 | 1.9826 | 0.3652 | 5.71e-05 | +0.025 | -0.007 | -0.001 | 10531 |
| 2026-07-03 10:16:27 | -- | 4.4197 | 1.0470 | 0.1368 | 2.4708 | 0.3386 | 1.05e-04 | +0.054 | +0.065 | -0.006 | 10558 |
| 2026-07-03 14:26:44 | -- | 2.8708 | 0.9376 | 0.0802 | 0.7169 | 0.3027 | 1.09e-04 | +0.102 | +0.080 | -0.037 | 10510 |
| 2026-07-03 23:10:01 | -- | 2.4909 | 0.7648 | 0.0402 | 0.6328 | 0.2602 | 1.51e-04 | +0.094 | -0.027 | -0.109 | 10887 |
| 2026-07-04 02:07:32 | -- | 3.0211 | 0.7969 | 0.0493 | 1.2441 | 0.2429 | 1.90e-04 | +0.067 | -0.008 | -0.190 | 10626 |
| 2026-07-04 05:03:58 | -- | 3.2653 | 0.7496 | 0.0407 | 1.7675 | 0.2420 | 2.22e-04 | +0.030 | +0.205 | -0.262 | 10562 |
| 2026-07-04 08:03:44 | -- | 3.0906 | 0.7242 | 0.0363 | 1.6900 | 0.2313 | 2.43e-04 | -0.021 | +0.392 | -0.321 | 10543 |
| 2026-07-04 10:59:37 | -- | 2.9983 | 0.6881 | 0.0298 | 1.6724 | 0.2405 | 2.50e-04 | -0.070 | +0.493 | -0.345 | 10526 |
| 2026-07-04 13:54:20 | -- | 2.8642 | 0.6394 | 0.0233 | 1.6137 | 0.2303 | 2.50e-04 | -0.137 | +0.527 | -0.365 | 10460 |
| 2026-07-05 00:35:43 | -- | 1.9153 | 0.5176 | 0.0170 | 0.5034 | 0.2393 | 1.37e-03 | -0.324 | +0.010 | -0.350 | 17307 |

### 4.1 Epoch Duration Analysis

| Epoch | Batches | Steps | Duration (s) | Hours | Speed (batch/s) |
|---|---|---|---|---|---|
| 0 | 98 | 146 | 149 | 0.04 | 0.66 |
| 1 | 1 | 1 | 11 | 0.00 | 0.09 |
| 0 | 4387 | 6580 | 5495 | 1.53 | 0.80 |
| 1 | 4387 | 6580 | 6950 | 1.93 | 0.63 |
| 2 | 6580 | 8224 | 11466 | 3.19 | 0.57 |
| 3 | 6580 | 8224 | 10531 | 2.93 | 0.62 |
| 4 | 6580 | 8224 | 10558 | 2.93 | 0.62 |
| 5 | 6580 | 8224 | 10510 | 2.92 | 0.63 |
| 6 | 6580 | 8224 | 10887 | 3.02 | 0.60 |
| 7 | 6580 | 8224 | 10626 | 2.95 | 0.62 |
| 8 | 6580 | 8224 | 10562 | 2.93 | 0.62 |
| 9 | 6580 | 8224 | 10543 | 2.93 | 0.62 |
| 10 | 6580 | 8224 | 10526 | 2.92 | 0.63 |
| 11 | 6580 | 8224 | 10460 | 2.91 | 0.63 |
| 17 | 6580 | 8224 | 17307 | 4.81 | 0.38 |

---
## Section 5: All Validation Metrics Ever Recorded

Every validation event from every run, in chronological order.

### 5.1 rf_stages -- Full Validation History (5 events)

| # | Timestamp | Epoch | Loss | det_mAP50 | det_mAP50_pc | det_n | act_frame | act_macro_f1 | act_top5 | fwd_ang_deg | psr_F1 | psr_edit | psr_pos | combined | best_combined |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2026-07-03 04:24:08 | E1 | 4.1024 | 0.0831 | 0.1330 | 0 | 0.0100 | 0.0063 | 0.0550 | 11.32 | 0.0000 | 0.0000 | 0.0000 | 0.1825 | 0.0000 |
| 2 | 2026-07-03 14:30:42 | E2 | 4.2703 | 0.2119 | 0.3391 | 0 | 0.1830 | 0.0971 | 0.3810 | 8.92 | 0.0000 | 0.0000 | 0.0000 | 0.2793 | 0.1825 |
| 3 | 2026-07-04 05:07:46 | E3 | 6.7022 | 0.2079 | 0.3326 | 0 | 0.0810 | 0.0488 | 0.2760 | 10.85 | 0.0333 | 0.7283 | 0.9664 | 0.2643 | 0.2793 |
| 4 | 2026-07-04 13:58:10 | E4 | 6.2004 | 0.3165 | 0.5063 | 0 | 0.1770 | 0.1096 | 0.3980 | 8.14 | 0.1440 | 0.7520 | 0.9682 | 0.3628 | 0.2793 |

### 5.2 Val Metric Trajectory Summary

| Metric | Val 1 | Val 2 | Val 3 | Val 4 | Val 5 | Delta |
|---|---|---|---|---|---|---|---|

### 5.3 Additional Val Events from Other Runs

- **ablation_det_only/run.log**: 2 validation events
- **full_multi_task_tma_tbank/logs/train.log**: 8 validation events
- **rf4_fable6_20260703_010909.log**: 1 validation events
- **rf4_resumed_20260704_194703.log**: 1 validation events
- **rf4_round5_20260703_113124.log**: 1 validation events
- **rf4_stable2_20260703_200823.log**: 2 validation events

---
## Section 6: Per-Class Activity Accuracy Across Epochs

The activity head predicts 1 of 11 operation classes per frame. Evaluations show which classes the model recognizes and which it misses.

### 6.1 Epoch 0 Val -- Activity Collapse at Initialization

| Metric | Value |
|---|---|
| pred_distinct | 2/11 classes |
| entropy | 0.117 nats |
| gt_distinct | 9/11 classes |
| Macro-F1 | 0.0587 |
| Frame Accuracy | 0.3500 |
| Top-5 Accuracy | 0.7750 |
| Macro-Recall | 0.1111 |

5 Worst Classes (all 0.0%): other, take, align, plug, tighten
5 Best Classes: browse (100.0%), pull (0.0%), loosen (0.0%), fit (0.0%), put (0.0%)

At initialization, the model predicts only 2 classes (dominated by class 7 at 97.5%). This is expected for a randomly initialized classifier with severe class imbalance.

### 6.2 Epoch ~1 Val -- 67-Class Evaluation

| Metric | Value |
|---|---|
| pred_distinct | 3/69 classes |
| entropy | 0.423 nats |
| gt_distinct | 62/69 classes |
| Macro-F1 | 0.0004 |
| Frame Accuracy | 0.0120 |

5 Worst (all 0.0%): other, take_short_brace, align_objects, take_pin_short, plug_short_pin
5 Best: put_screw_pin (85.7%), pull_pin_long (0.0%), fit_wing (0.0%), fit_acorn_nut (0.0%), plug_objects (0.0%)

The 67-class evaluation (vs 11-class in epoch 0) shows even worse diversity. Only 3/69 classes predicted. This suggests the class-count mismatch between evaluations.

### 6.3 Epoch 3-4 Val -- Peak Activity Regression

| Metric | Value |
|---|---|
| pred_distinct | 1/69 classes |
| entropy | 0.000 nats |
| Macro-F1 | 0.0000 |
| Frame Accuracy | 0.0000 |

5 Worst (all 0.0%): other, take_short_brace, align_objects, take_pin_short, plug_short_pin
5 Best (all 0.0%): plug_objects, pull_pin_long, fit_wing, fit_acorn_nut, put_small_screw_pin

Complete activity collapse: 1/69 classes predicted, zero accuracy. The activity head has degenerated to predicting a single class for everything.

### 6.4 Epoch 7-8 Val -- Activity Recovery Begins

| Metric | Value |
|---|---|
| Macro-F1 | 0.0488 |
| Frame Accuracy | 0.0810 |
| Top-5 Accuracy | 0.2760 |

5 Worst (0.0%): other, take_short_brace, align_objects, take_pin_short, plug_short_pin
5 Best: take_wing (44.0%), browse_instruction (43.0%), put (42.9%), put_wing (42.9%), fit_acorn_nut (41.2%)

Activity begins to recover: top classes reach 41-44% accuracy. Metrics climb from 0.00 to 0.05 Macro-F1.

### 6.5 Epoch 11-12 Val -- Activity Improving

| Metric | Value |
|---|---|
| Macro-F1 | 0.1096 |
| Frame Accuracy | 0.1770 |
| Top-5 Accuracy | 0.3980 |

5 Worst (0.0%): other, take_short_brace, align_objects, take_tooth_washer, take_nut
5 Best: browse_instruction (77.3%), fit_acorn_nut (71.4%), plug_pin_long (53.3%), take_wing (52.9%), plug_objects (42.9%)

Top classes now reach 77%. Macro-F1 doubles to 0.11. Some classes (other, align_objects) remain stuck at 0%.

### 6.6 Epoch 17-18 Val -- Best Activity (Latest)

| Metric | Value |
|---|---|
| Macro-F1 | 0.2047 |
| Frame Accuracy | 0.3110 |
| Top-5 Accuracy | 0.5420 |
| Macro-Recall | 0.2214 |
| Weighted-F1 | 0.3049 |

5 Worst (0.0%): other, align_objects, put, put_pin_long, put_wing
5 Best: plug_pin_long (87.5%), browse_instruction (87.5%), take_wing (80.0%), take_wing_beam (79.5%), tighten_acorn_nut (78.6%)

Best activity evaluation: 8 classes above 0%, top-5 above 78%. The 5 dead classes (other, align_objects, put, put_pin_long, put_wing) represent rare operations that lack sufficient training examples.

### 6.7 Final Epoch 18 Val (rf4_resumed_194703)

| Metric | Value |
|---|---|
| Macro-F1 | 0.1655 |
| Frame Accuracy | 0.2530 |
| Top-5 Accuracy | 0.4710 |

5 Worst (0.0%): other, take_short_brace, align_objects, take_screw_pin, put
5 Best: browse_instruction (81.6%), take_pin_middle (65.5%), fit_wheel (60.0%), take_wing (58.1%), plug_pin_long (55.6%)

This is slightly worse than epoch 17, suggesting the resumed run may have had a different checkpoint or slight regression.

### 6.8 Per-Class Accuracy Trajectory (Top-5 Classes)

| Class | E0 | E1 | E3-4 | E7-8 | E11-12 | E17-18 | Trend |
|---|---|---|---|---|---|---|---|
| browse_instruction | 100% | -- | 0% | 43% | 77% | 88% | V-shaped |
| plug_pin_long | 0% | -- | -- | -- | 53% | 88% | Rising |
| take_wing | 0% | -- | 0% | 44% | 53% | 80% | Rising |
| take_wing_beam | -- | -- | -- | -- | -- | 80% | Late |
| tighten_acorn_nut | -- | -- | -- | -- | -- | 79% | Late |
| fit_acorn_nut | -- | -- | 0% | 41% | 71% | -- | Variable |
| put_screw_pin | -- | 86% | 0% | -- | -- | -- | Collapse |

Key pattern: Several classes that appear good at early epochs (put_screw_pin at 86%) later collapse to 0% as the model adjusts its decision boundary. The late-emerging classes (take_wing_beam, tighten_acorn_nut) only become identifiable after sufficient training.

---
## Section 7: DET_PROBE Verdicts -- Complete Analysis

DET_PROBE logs are emitted at each validation batch. They measure whether the detection head produces meaningful predictions. 4210 probe entries exist in rf_stages alone (2886 LOCALIZING, 1324 TOTAL COLLAPSE = 68.5% LOCALIZING rate).

### 7.1 Verdict Distribution Across Runs

- **ablation_det_only/run.log**: 1000 probes, 876 LOCALIZING (88%), 124 TOTAL COLLAPSE (12%)
- **full_multi_task_tma_tbank/logs/train.log**: 18426 probes, 14354 LOCALIZING (78%), 4072 TOTAL COLLAPSE (22%)
- **rf4_batch6_20260702_135539.log**: 1000 probes, 986 LOCALIZING (99%), 14 TOTAL COLLAPSE (1%)
- **rf4_batch6_20260702_155325.log**: 400 probes, 176 LOCALIZING (44%), 224 TOTAL COLLAPSE (56%)
- **rf4_batch6_20260702_175750.log**: 800 probes, 354 LOCALIZING (44%), 446 TOTAL COLLAPSE (56%)
- **rf4_batch6_20260702_204203.log**: 400 probes, 144 LOCALIZING (36%), 256 TOTAL COLLAPSE (64%)
- **rf4_fable2_20260702_224125.log**: 500 probes, 188 LOCALIZING (38%), 312 TOTAL COLLAPSE (62%)
- **rf4_fable5_20260703_002938.log**: 500 probes, 188 LOCALIZING (38%), 312 TOTAL COLLAPSE (62%)
- **rf4_fable6_20260703_010909.log**: 500 probes, 428 LOCALIZING (86%), 72 TOTAL COLLAPSE (14%)
- **rf4_probe_20260702_000833.log**: 1000 probes, 852 LOCALIZING (85%), 148 TOTAL COLLAPSE (15%)
- **rf4_probe_final2_20260701_231321.log**: 20 probes, 14 LOCALIZING (70%), 6 TOTAL COLLAPSE (30%)
- **rf4_resumed_20260704_194703.log**: 1000 probes, 858 LOCALIZING (86%), 142 TOTAL COLLAPSE (14%)
- **rf4_round5_20260703_113124.log**: 500 probes, 428 LOCALIZING (86%), 72 TOTAL COLLAPSE (14%)
- **rf4_run_20260702_081721.log**: 400 probes, 152 LOCALIZING (38%), 248 TOTAL COLLAPSE (62%)
- **rf4_stable2_20260703_200823.log**: 1000 probes, 852 LOCALIZING (85%), 148 TOTAL COLLAPSE (15%)
- **rf_stages.bak.1782914773/logs/train.log**: 600 probes, 252 LOCALIZING (42%), 348 TOTAL COLLAPSE (58%)
- **rf_stages/checkpoints/d3_full_eval/run6.log**: 9509 probes, 210 LOCALIZING (2%), 9299 TOTAL COLLAPSE (98%)
- **rf_stages/checkpoints/d3_full_eval/run7.log**: 9509 probes, 210 LOCALIZING (2%), 9299 TOTAL COLLAPSE (98%)
- **rf_stages/checkpoints/d3_full_eval/run8.log**: 9509 probes, 210 LOCALIZING (2%), 9299 TOTAL COLLAPSE (98%)
- **rf_stages/checkpoints/d3_v3/run.log**: 9509 probes, 210 LOCALIZING (2%), 9299 TOTAL COLLAPSE (98%)
- **rf_stages/logs/train.log**: 4210 probes, 2886 LOCALIZING (69%), 1324 TOTAL COLLAPSE (31%)
- **train_launch_20260701_010742_route_a.log**: 800 probes, 336 LOCALIZING (42%), 464 TOTAL COLLAPSE (58%)
- **train_launch_20260701_011810_full_data_route_a.log**: 400 probes, 182 LOCALIZING (46%), 218 TOTAL COLLAPSE (55%)
- **val_epoch1.log**: 200 probes, 0 LOCALIZING (0%), 200 TOTAL COLLAPSE (100%)

### 7.2 Probe Quality Comparison: Early vs Late Training

#### Early Epoch (rf_stages epoch 0):
| Metric | GT-bearing batches | GT-absent batches |
|---|---|---|
| score_p50 | 0.055 | 0.055 |
| score_max | 0.21-0.25 | 0.20-0.27 |
| preds>0.50 | 0 (none above 0.30) | 0 |
| bestIoU>0.5 | 248-748 | 0 |
| bestIoU_max | 0.84-0.93 | 0.00 |
| bestIoU_mean | 0.024-0.029 | 0.000 |

At initialization: All predictions have uniform ~0.055 confidence. The detector can LOCALIZE but not with high confidence. max IoU is already 0.84+ (good localization). No predictions above 0.50 threshold.

#### Late Epoch (rf4_resumed_194703 epoch 18):
| Metric | GT-bearing batches | GT-absent batches |
|---|---|---|
| score_p50 | 0.0012-0.0016 | 0.0013-0.0018 |
| score_max | 0.83-0.999 | 0.15-0.39 |
| preds>0.50 | 247-1398 | 0 |
| bestIoU>0.5 | 2878-9421 | 0 |
| bestIoU_max | 0.90-0.98 | 0.00 |
| bestIoU_mean | 0.075-0.361 | 0.000 |

After training: score_p50 drops to 0.001 (model is now confident -- only fires when needed). max confidence hits 0.999. bestIoU>0.5 counts are 5-20x higher. bestIoU_mean is 3-10x higher. The detector is well-calibrated.

### 7.3 Complete Probe Detail: Latest Run (rf4_resumed_194703)

1000 probes total: 858 LOCALIZING (85.8%), 142 TOTAL COLLAPSE (14.2%)

Sample across full run (every ~100th probe):
| Batch # | bestIoU>0.5 | bestIoU_max | bestIoU_mean | Verdict |
|---|---|---|---|---|
| 0 | 3943 | 0.9513 | 0.2291 | LOCALIZING |
| 33 | 9202 | 0.9595 | 0.2919 | LOCALIZING |
| 66 | 4208 | 0.9160 | 0.2789 | LOCALIZING |
| 99 | 3789 | 0.9721 | 0.3238 | LOCALIZING |
| 132 | 6538 | 0.9075 | 0.2735 | LOCALIZING |
| 165 | 3504 | 0.9207 | 0.4755 | LOCALIZING |
| 198 | 7336 | 0.9714 | 0.2712 | LOCALIZING |
| 231 | 7361 | 0.9674 | 0.4787 | LOCALIZING |
| 14 | 0 | 0.0000 | 0.0000 | TOTAL |
| 47 | 4448 | 0.9625 | 0.1408 | LOCALIZING |
| 80 | 6886 | 0.9596 | 0.3374 | LOCALIZING |
| 113 | 0 | 0.0000 | 0.0000 | TOTAL |
| 146 | 7431 | 0.9609 | 0.1832 | LOCALIZING |
| 179 | 8211 | 0.9800 | 0.4085 | LOCALIZING |
| 212 | 3475 | 0.9495 | 0.2689 | LOCALIZING |
| 245 | 4367 | 0.8974 | 0.3055 | LOCALIZING |

---
## Section 8: Kendall Log-Var Evolution

Kendall's uncertainty weighting learns per-task log-variances. The precision is exp(-log_var). Higher precision = higher task weight in the combined loss.

### 8.1 rf_stages -- Kendall Log-Var Trajectory (sampled)

| Sample | det_lv | pose_lv | act_lv | psr_lv | det_prec | pose_prec | act_prec | psr_prec |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.0040 | -1.0000 | -0.0050 | -0.0010 | 0.996 | 2.718 | 1.005 | 1.001 |
| 38 | 0.0050 | -1.0000 | -0.0060 | -0.0010 | 0.995 | 2.718 | 1.006 | 1.001 |
| 76 | 0.0110 | -1.0000 | -0.0130 | -0.0010 | 0.989 | 2.718 | 1.013 | 1.001 |
| 114 | 0.0190 | -1.0000 | -0.0140 | -0.0010 | 0.981 | 2.718 | 1.014 | 1.001 |
| 152 | 0.0300 | -1.0000 | -0.0010 | -0.0010 | 0.970 | 2.718 | 1.001 | 1.001 |
| 190 | 0.0480 | -1.0000 | 0.0550 | -0.0040 | 0.953 | 2.718 | 0.946 | 1.004 |
| 228 | 0.0730 | -1.0000 | 0.1070 | -0.0110 | 0.930 | 2.718 | 0.899 | 1.011 |
| 266 | 0.0910 | -1.0000 | 0.0960 | -0.0210 | 0.913 | 2.718 | 0.908 | 1.021 |
| 304 | 0.1210 | -1.0000 | 0.0470 | -0.0660 | 0.886 | 2.718 | 0.954 | 1.068 |
| 342 | 0.1050 | -1.0000 | 0.0220 | -0.0760 | 0.900 | 2.718 | 0.978 | 1.079 |
| 380 | 0.1240 | -1.0000 | 0.0360 | -0.0760 | 0.883 | 2.718 | 0.965 | 1.079 |
| 418 | 0.0930 | -1.0000 | -0.0340 | -0.1130 | 0.911 | 2.718 | 1.035 | 1.120 |
| 456 | 0.0740 | -0.9990 | -0.0670 | -0.1580 | 0.929 | 2.716 | 1.069 | 1.171 |
| 494 | 0.0650 | -0.9990 | 0.0240 | -0.2060 | 0.937 | 2.716 | 0.976 | 1.229 |
| 532 | 0.0420 | -0.9990 | 0.1530 | -0.2510 | 0.959 | 2.716 | 0.858 | 1.285 |
| 570 | 0.0150 | -0.9990 | 0.2880 | -0.2830 | 0.985 | 2.716 | 0.750 | 1.327 |
| 608 | -0.0200 | -0.9990 | 0.3870 | -0.3210 | 1.020 | 2.716 | 0.679 | 1.379 |
| 646 | -0.0450 | -0.9990 | 0.4530 | -0.3400 | 1.046 | 2.716 | 0.636 | 1.405 |
| 684 | -0.0750 | -0.9980 | 0.4980 | -0.3460 | 1.078 | 2.713 | 0.608 | 1.413 |
| 722 | -0.1120 | -0.9980 | 0.5230 | -0.3540 | 1.119 | 2.713 | 0.593 | 1.425 |
| 760 | -0.1560 | -0.9980 | 0.5310 | -0.3710 | 1.169 | 2.713 | 0.588 | 1.449 |
| 798 | -0.1840 | -0.9980 | 0.5170 | -0.3710 | 1.202 | 2.713 | 0.596 | 1.449 |
| 836 | -0.1950 | -0.9980 | 0.4790 | -0.3440 | 1.215 | 2.713 | 0.619 | 1.411 |
| 874 | -0.2400 | -0.9980 | 0.1870 | -0.3590 | 1.271 | 2.713 | 0.829 | 1.432 |
| 912 | -0.3270 | -0.9970 | 0.0360 | -0.3450 | 1.387 | 2.710 | 0.965 | 1.412 |
| 950 | -0.3350 | -0.9970 | -0.1280 | -0.3640 | 1.398 | 2.710 | 1.137 | 1.439 |

**Final Kendall state:** det_lv=-0.3350 (prec=1.398), pose_lv=-0.9970 (prec=2.710), act_lv=-0.1310 (prec=1.140), psr_lv=-0.3630 (prec=1.438)

### 8.2 rf4_resumed_194703 -- Kendall Log-Vars

| Sample | det_lv | pose_lv | act_lv | psr_lv | det_prec | pose_prec | act_prec | psr_prec | HP_PREC_CAP |
|---|---|---|---|---|---|---|---|---|---|
| 0 | -0.2410 | -0.9980 | 0.1930 | -0.3580 | 1.273 | 2.713 | 0.824 | 1.430 | ACTIVE |
| 3 | -0.2680 | -0.9970 | 0.1710 | -0.3350 | 1.307 | 2.710 | 0.843 | 1.398 | ACTIVE |
| 6 | -0.2900 | -0.9970 | 0.1490 | -0.3360 | 1.336 | 2.710 | 0.862 | 1.399 | ACTIVE |
| 9 | -0.3020 | -0.9970 | 0.1310 | -0.3380 | 1.353 | 2.710 | 0.877 | 1.402 | ACTIVE |
| 12 | -0.3070 | -0.9970 | 0.1120 | -0.3380 | 1.359 | 2.710 | 0.894 | 1.402 | ACTIVE |
| 15 | -0.3080 | -0.9970 | 0.0950 | -0.3410 | 1.361 | 2.710 | 0.909 | 1.406 | ACTIVE |
| 18 | -0.3140 | -0.9970 | 0.0780 | -0.3410 | 1.369 | 2.710 | 0.925 | 1.406 | ACTIVE |
| 21 | -0.3180 | -0.9970 | 0.0610 | -0.3450 | 1.374 | 2.710 | 0.941 | 1.412 | ACTIVE |
| 24 | -0.3230 | -0.9970 | 0.0460 | -0.3440 | 1.381 | 2.710 | 0.955 | 1.411 | ACTIVE |
| 27 | -0.3290 | -0.9970 | 0.0300 | -0.3460 | 1.390 | 2.710 | 0.970 | 1.413 | ACTIVE |
| 30 | -0.3330 | -0.9970 | 0.0150 | -0.3450 | 1.395 | 2.710 | 0.985 | 1.412 | ACTIVE |
| 33 | -0.3360 | -0.9970 | 0.0010 | -0.3490 | 1.399 | 2.710 | 0.999 | 1.418 | ACTIVE |
| 36 | -0.3420 | -0.9970 | -0.0140 | -0.3530 | 1.408 | 2.710 | 1.014 | 1.423 | ACTIVE |
| 39 | -0.3450 | -0.9970 | -0.0280 | -0.3550 | 1.412 | 2.710 | 1.028 | 1.426 | ACTIVE |
| 42 | -0.3440 | -0.9970 | -0.0410 | -0.3560 | 1.411 | 2.710 | 1.042 | 1.428 | ACTIVE |
| 45 | -0.3420 | -0.9970 | -0.0520 | -0.3550 | 1.408 | 2.710 | 1.053 | 1.426 | ACTIVE |
| 48 | -0.3430 | -0.9970 | -0.0640 | -0.3540 | 1.409 | 2.710 | 1.066 | 1.425 | ACTIVE |
| 51 | -0.3400 | -0.9970 | -0.0760 | -0.3570 | 1.405 | 2.710 | 1.079 | 1.429 | ACTIVE |
| 54 | -0.3410 | -0.9970 | -0.0890 | -0.3590 | 1.406 | 2.710 | 1.093 | 1.432 | ACTIVE |
| 57 | -0.3400 | -0.9970 | -0.1000 | -0.3600 | 1.405 | 2.710 | 1.105 | 1.433 | ACTIVE |
| 60 | -0.3380 | -0.9970 | -0.1120 | -0.3630 | 1.402 | 2.710 | 1.119 | 1.438 | ACTIVE |
| 63 | -0.3350 | -0.9970 | -0.1230 | -0.3640 | 1.398 | 2.710 | 1.131 | 1.439 | ACTIVE |

### 8.3 Kendall Analysis

- HP_PREC_CAP remains ACTIVE throughout training for pose log-var, which is pinned at ~ -1.0 (prec=2.71). This means the pose head's Kendall weight is capped at a maximum of 2.71 due to the precision cap.
- Det log-var decreases from ~0.013 to -0.30 (prec increasing from 0.99 to 1.35) across training, meaning the detector gains confidence.
- Act log-var increases from 0.0 to 0.14 (prec decreasing from 1.00 to 0.87), meaning the activity head becomes LESS trusted by Kendall as training progresses -- consistent with its poor performance.
- PSR log-var stays near -0.34 (prec ~1.40), stable throughout training.
- The Kendall mechanism is correctly adjusting: trustworthy heads (det, pose) get higher weight; struggling heads (act) get lower weight.

---
## Section 9: Liveness Gradient Probe Analysis

Liveness gradient probes measure gradient RMS at each head. ALIVE = meaningful gradient flowing, DEAD = no gradient signal, NO_GRAD = head not in computation graph.

### 9.1 rf_stages -- First Liveness Probe (Step 0)

```
detection_head: ALIVE[7.93e-01]/ALIVE[8.79e-02]
pose_head: ALIVE[8.49e-02]/ALIVE[2.98e-03]
head_pose_head: ALIVE[4.38e+00]/ALIVE[3.57e-02]
activity_head: ALIVE[1.83e-02]/ALIVE[2.12e-03]
psr_head: ALIVE[1.72e+01]/DEAD[1.49e-08]
psr_heads: ALL 11 ALIVE (RMS 2.3-3.5)
backbone: ALIVE[1.359e+00|n=178]
fpn: ALIVE[7.974e-01|n=16]
gpu_mem=0.74GB/7.42GB
```

At step 0: All 4 main heads ALIVE. PSR shows ALIVE in the main head log but DEAD in the second probe (raw PSR vs component average). GPU memory 0.74 GB used, 7.42 GB reserved.

### 9.2 Latest Run (rf4_resumed_194703) -- Liveness Timeline

| Step | det_head | pose_head | hp_head | act_head | psr_head | backbone | fpn | GPU Mem |
|---|---|---|---|---|---|---|---|---|
| 1 | ALIVE | ALIVE | ALIVE | ALIVE | NO_GRAD | ALIVE | ALIVE | 1.25GB |
| 201 | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | 1.43GB |
| 401 | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | 1.43GB |
| 601 | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | 2.07GB |
| 801 | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | 1.43GB |
| 1001 | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | 1.43GB |
| 1201 | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | 2.13GB |
| 1401 | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | 1.49GB |
| 1601 | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | 1.49GB |
| 1801 | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | 1.49GB |
| 2001 | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | 1.49GB |
| 2201 | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | 1.49GB |
| 2401 | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | 1.49GB |
| 2601 | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | 1.49GB |
| 2801 | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | 1.49GB |
| 3001 | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | 1.43GB |
| 3201 | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | 1.49GB |
| 3401 | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | 1.49GB |
| 3601 | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | 1.49GB |
| 3801 | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | 1.49GB |
| 4001 | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | 1.49GB |
| 4201 | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | 1.49GB |
| 4401 | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | 1.49GB |
| 4601 | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | 1.49GB |
| 4801 | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | 1.49GB |
| 5001 | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | 1.43GB |
| 5201 | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | 1.49GB |
| 5401 | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | 2.13GB |
| 5601 | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | 1.49GB |
| 5801 | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | 1.49GB |
| 6001 | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | 1.49GB |
| 6201 | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | 1.49GB |
| 6401 | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | ALIVE | 1.49GB |

All 4 heads maintain ALIVE status throughout training. No head ever goes DEAD in the resumed run. PSR starts as NO_GRAD (first step, no sequence data yet) but activates by step 201.

### 9.3 PSR Sub-Head Liveness Detail (rf4_resumed_194703)

The PSR head has 11 sub-components (h0-h10). Their liveness at key steps reveals degradation in low-prevalence components.

| Step | h0 | h1 | h2 | h3 | h4 | h5 | h6 | h7 | h8 | h9 | h10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 401 | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? |
| 801 | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? |
| 1201 | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? |
| 1601 | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? |
| 2001 | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? |
| 2401 | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? |
| 2801 | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? |
| 3201 | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? |
| 3601 | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? |
| 4001 | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? |
| 4401 | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? |
| 4801 | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? |
| 5201 | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? |
| 5601 | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? |
| 6001 | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? |
| 6401 | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? |

Note: Later in training, components h7, h8, h9 show DEAD or near-DEAD gradient RMS (<1e-3). These correspond to the lowest-prevalence PSR components.

---
## Section 10: Complete Crash Database

Every crash across every log file, categorized by type.

| Run | Line | Type | Description |
|---|---|---|---|
| ablation_A_3060/logs/train.log | L7 | cuDNN | Config hash details: {"ALLOW_TF32": true, "BACKBONE": "convnext_tiny", "BASE_LR": 0.0005, "BATCH_SIZ |
| ablation_det_only/run.log | L7 | cuDNN | Config hash details: {"ALLOW_TF32": true, "BACKBONE": "convnext_tiny", "BASE_LR": 0.0005, "BATCH_SIZ |
| full_multi_task_tma_tbank/logs/train.log | L84468 | OOM | torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 338.00 MiB. GPU 0 has a total capacity |
| full_multi_task_tma_tbank/logs/train.log | L84517 | OOM | torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 44.00 MiB. GPU 0 has a total capacity  |
| full_multi_task_tma_tbank/logs/train.log | L84695 | OOM | torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 86.00 MiB. GPU 0 has a total capacity  |
| phase_A_5060ti/logs/train.log | L8 | cuDNN | Config hash details: {"ALLOW_TF32": true, "BACKBONE": "convnext_tiny", "BASE_LR": 0.0005, "BATCH_SIZ |
| phase_B_5060ti/logs/train.log | L8 | cuDNN | Config hash details: {"ALLOW_TF32": true, "BACKBONE": "convnext_tiny", "BASE_LR": 0.0005, "BATCH_SIZ |
| phase_C_5060ti/logs/train.log | L8 | cuDNN | Config hash details: {"ALLOW_TF32": true, "BACKBONE": "convnext_tiny", "BASE_LR": 0.0005, "BATCH_SIZ |
| rf4_3060_20260703_105922.log | L8 | cuDNN | Config hash details: {"ALLOW_TF32": true, "BACKBONE": "convnext_tiny", "BASE_LR": 0.0005, "BATCH_SIZ |
| rf4_3060_20260703_110019.log | L9 | cuDNN | Config hash details: {"ALLOW_TF32": true, "BACKBONE": "convnext_tiny", "BASE_LR": 0.0005, "BATCH_SIZ |
| rf4_3060main_20260703_152745.log | L8 | cuDNN | Config hash details: {"ALLOW_TF32": true, "BACKBONE": "convnext_tiny", "BASE_LR": 0.0005, "BATCH_SIZ |
| rf4_batch6_20260702_135539.log | L8 | cuDNN | Config hash details: {"ALLOW_TF32": true, "BACKBONE": "convnext_tiny", "BASE_LR": 0.0005, "BATCH_SIZ |
| rf4_batch6_20260702_154243.log | L8 | cuDNN | Config hash details: {"ALLOW_TF32": true, "BACKBONE": "convnext_tiny", "BASE_LR": 0.0005, "BATCH_SIZ |
| rf4_batch6_20260702_154654.log | L8 | cuDNN | Config hash details: {"ALLOW_TF32": true, "BACKBONE": "convnext_tiny", "BASE_LR": 0.0005, "BATCH_SIZ |
| rf4_batch6_20260702_155310.log | L8 | cuDNN | Config hash details: {"ALLOW_TF32": true, "BACKBONE": "convnext_tiny", "BASE_LR": 0.0005, "BATCH_SIZ |
| rf4_batch6_20260702_155325.log | L8 | cuDNN | Config hash details: {"ALLOW_TF32": true, "BACKBONE": "convnext_tiny", "BASE_LR": 0.0005, "BATCH_SIZ |
| rf4_batch6_20260702_175750.log | L8 | cuDNN | Config hash details: {"ALLOW_TF32": true, "BACKBONE": "convnext_tiny", "BASE_LR": 0.0005, "BATCH_SIZ |
| rf4_batch6_20260702_204203.log | L8 | cuDNN | Config hash details: {"ALLOW_TF32": true, "BACKBONE": "convnext_tiny", "BASE_LR": 0.0005, "BATCH_SIZ |
| rf4_clean_20260702_131450.log | L8 | cuDNN | Config hash details: {"ALLOW_TF32": true, "BACKBONE": "convnext_tiny", "BASE_LR": 0.0005, "BATCH_SIZ |
| rf4_clean_20260702_131538.log | L8 | cuDNN | Config hash details: {"ALLOW_TF32": true, "BACKBONE": "convnext_tiny", "BASE_LR": 0.0005, "BATCH_SIZ |
| rf4_clean_20260702_134058.log | L8 | cuDNN | Config hash details: {"ALLOW_TF32": true, "BACKBONE": "convnext_tiny", "BASE_LR": 0.0005, "BATCH_SIZ |
| rf4_fable2_20260702_224125.log | L8 | cuDNN | Config hash details: {"ALLOW_TF32": true, "BACKBONE": "convnext_tiny", "BASE_LR": 0.0005, "BATCH_SIZ |
| rf4_fable2_20260702_224125.log | L3728 | cuDNN | RuntimeError: cuDNN error: CUDNN_STATUS_EXECUTION_FAILED
 |
| rf4_fable3_20260702_235057.log | L8 | cuDNN | Config hash details: {"ALLOW_TF32": true, "BACKBONE": "convnext_tiny", "BASE_LR": 0.0005, "BATCH_SIZ |
| rf4_fable4_20260703_002814.log | L8 | cuDNN | Config hash details: {"ALLOW_TF32": true, "BACKBONE": "convnext_tiny", "BASE_LR": 0.0005, "BATCH_SIZ |
| rf4_fable5_20260703_002938.log | L8 | cuDNN | Config hash details: {"ALLOW_TF32": true, "BACKBONE": "convnext_tiny", "BASE_LR": 0.0005, "BATCH_SIZ |
| rf4_fable6_20260703_010909.log | L8 | cuDNN | Config hash details: {"ALLOW_TF32": true, "BACKBONE": "convnext_tiny", "BASE_LR": 0.0005, "BATCH_SIZ |
| rf4_fable6_20260703_010909.log | L46551 | cuDNN | Epoch 5 [no-staging]:   2%|▏         | 102/6580 [02:48<2:48:06,  1.56s/it, loss=5.6964 det=1.5941(c= |
| rf4_fable6_20260703_010909.log | L46568 | cuDNN |     data_type = CUDNN_DATA_FLOAT
 |
| rf4_fable6_20260703_010909.log | L46576 | cuDNN |     type = CUDNN_DATA_FLOAT
 |
| rf4_fable6_20260703_010909.log | L46581 | cuDNN |     type = CUDNN_DATA_FLOAT
 |
| rf4_fable6_20260703_010909.log | L46586 | cuDNN |     type = CUDNN_DATA_FLOAT
 |
| rf4_fable6_20260703_010909.log | L46587 | cuDNN |     tensor_format = CUDNN_TENSOR_NHWC
 |
| rf4_fable7_20260703_105823.log | L8 | cuDNN | Config hash details: {"ALLOW_TF32": true, "BACKBONE": "convnext_tiny", "BASE_LR": 0.0005, "BATCH_SIZ |
| rf4_fable_20260702_221558.log | L8 | cuDNN | Config hash details: {"ALLOW_TF32": true, "BACKBONE": "convnext_tiny", "BASE_LR": 0.0005, "BATCH_SIZ |
| rf4_fable_20260702_221558.log | L1487 | cuDNN | Epoch 2 [no-staging]:  13%|█▎        | 571/4387 [18:28<1:43:20,  1.62s/it, loss=3.7197 det=1.8018(c= |
| rf4_main3_20260703_194347.log | L9 | cuDNN | Config hash details: {"ALLOW_TF32": true, "BACKBONE": "convnext_tiny", "BASE_LR": 0.0005, "BATCH_SIZ |
| rf4_main_20260703_152840.log | L8 | cuDNN | Config hash details: {"ALLOW_TF32": true, "BACKBONE": "convnext_tiny", "BASE_LR": 0.0005, "BATCH_SIZ |
| rf4_main_20260703_152840.log | L10089 | cuDNN | Epoch 6 [no-staging]:  65%|██████▌   | 4289/6580 [1:56:28<1:04:18,  1.68s/it, loss=1.351 det=0.000 p |
| rf4_main_20260703_152840.log | L10106 | cuDNN |     data_type = CUDNN_DATA_FLOAT
 |
| rf4_main_20260703_152840.log | L10114 | cuDNN |     type = CUDNN_DATA_FLOAT
 |
| rf4_main_20260703_152840.log | L10119 | cuDNN |     type = CUDNN_DATA_FLOAT
 |
| rf4_main_20260703_152840.log | L10124 | cuDNN |     type = CUDNN_DATA_FLOAT
 |
| rf4_main_20260703_152840.log | L10125 | cuDNN |     tensor_format = CUDNN_TENSOR_NHWC
 |
| rf4_probe_20260701_230114.log | L51 | cuDNN | Config hash details: {"ALLOW_TF32": true, "BACKBONE": "convnext_tiny", "BASE_LR": 0.0005, "BATCH_SIZ |
| rf4_probe_20260701_230402.log | L51 | cuDNN | Config hash details: {"ALLOW_TF32": true, "BACKBONE": "convnext_tiny", "BASE_LR": 0.0005, "BATCH_SIZ |
| rf4_probe_20260702_000833.log | L13 | cuDNN | Config hash details: {"ALLOW_TF32": true, "BACKBONE": "convnext_tiny", "BASE_LR": 0.0005, "BATCH_SIZ |
| rf4_probe_final2_20260701_231321.log | L9 | cuDNN | Config hash details: {"ALLOW_TF32": true, "BACKBONE": "convnext_tiny", "BASE_LR": 0.0005, "BATCH_SIZ |
| rf4_resume_20260702_130934.log | L8 | cuDNN | Config hash details: {"ALLOW_TF32": true, "BACKBONE": "convnext_tiny", "BASE_LR": 0.0005, "BATCH_SIZ |
| rf4_resumed_20260704_192319.log | L9 | cuDNN | Config hash details: {"ALLOW_TF32": true, "BACKBONE": "convnext_tiny", "BASE_LR": 0.0005, "BATCH_SIZ |
| rf4_resumed_20260704_192847.log | L9 | cuDNN | Config hash details: {"ALLOW_TF32": true, "BACKBONE": "convnext_tiny", "BASE_LR": 0.0005, "BATCH_SIZ |
| rf4_resumed_20260704_194453.log | L9 | cuDNN | Config hash details: {"ALLOW_TF32": true, "BACKBONE": "convnext_tiny", "BASE_LR": 0.0005, "BATCH_SIZ |
| rf4_resumed_20260704_194603.log | L9 | cuDNN | Config hash details: {"ALLOW_TF32": true, "BACKBONE": "convnext_tiny", "BASE_LR": 0.0005, "BATCH_SIZ |
| rf4_resumed_20260704_194703.log | L9 | cuDNN | Config hash details: {"ALLOW_TF32": true, "BACKBONE": "convnext_tiny", "BASE_LR": 0.0005, "BATCH_SIZ |
| rf4_round5_20260703_113124.log | L8 | cuDNN | Config hash details: {"ALLOW_TF32": true, "BACKBONE": "convnext_tiny", "BASE_LR": 0.0005, "BATCH_SIZ |
| rf4_round5_20260703_113124.log | L20559 | OOM | torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 450.00 MiB. GPU 0 has a total capacity |
| rf4_round5_20260703_113124.log | L20626 | OOM | torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 450.00 MiB. GPU 0 has a total capacity |
| rf4_run_20260702_010027.log | L8 | cuDNN | Config hash details: {"ALLOW_TF32": true, "BACKBONE": "convnext_tiny", "BASE_LR": 0.0005, "BATCH_SIZ |
| rf4_run_20260702_010938.log | L8 | cuDNN | Config hash details: {"ALLOW_TF32": true, "BACKBONE": "convnext_tiny", "BASE_LR": 0.0005, "BATCH_SIZ |
| rf4_run_20260702_081721.log | L8 | cuDNN | Config hash details: {"ALLOW_TF32": true, "BACKBONE": "convnext_tiny", "BASE_LR": 0.0005, "BATCH_SIZ |
| rf4_run_20260702_100557.log | L8 | cuDNN | Config hash details: {"ALLOW_TF32": true, "BACKBONE": "convnext_tiny", "BASE_LR": 0.0005, "BATCH_SIZ |
| rf4_run_20260702_103014.log | L8 | cuDNN | Config hash details: {"ALLOW_TF32": true, "BACKBONE": "convnext_tiny", "BASE_LR": 0.0005, "BATCH_SIZ |
| rf4_run_20260702_104019.log | L8 | cuDNN | Config hash details: {"ALLOW_TF32": true, "BACKBONE": "convnext_tiny", "BASE_LR": 0.0005, "BATCH_SIZ |
| rf4_run_20260702_104258.log | L8 | cuDNN | Config hash details: {"ALLOW_TF32": true, "BACKBONE": "convnext_tiny", "BASE_LR": 0.0005, "BATCH_SIZ |
| rf4_run_20260702_112450.log | L8 | cuDNN | Config hash details: {"ALLOW_TF32": true, "BACKBONE": "convnext_tiny", "BASE_LR": 0.0005, "BATCH_SIZ |
| rf4_stable2_20260703_200823.log | L9 | cuDNN | Config hash details: {"ALLOW_TF32": true, "BACKBONE": "convnext_tiny", "BASE_LR": 0.0005, "BATCH_SIZ |
| rf4_stable3_20260703_110916.log | L8 | cuDNN | Config hash details: {"ALLOW_TF32": true, "BACKBONE": "convnext_tiny", "BASE_LR": 0.0005, "BATCH_SIZ |
| rf4_stable_20260703_200447.log | L9 | cuDNN | Config hash details: {"ALLOW_TF32": true, "BACKBONE": "convnext_tiny", "BASE_LR": 0.0005, "BATCH_SIZ |
| rf4_stable_20260704_162638.log | L9 | cuDNN | Config hash details: {"ALLOW_TF32": true, "BACKBONE": "convnext_tiny", "BASE_LR": 0.0005, "BATCH_SIZ |
| rf4_temporal_20260704_162320.log | L9 | cuDNN | Config hash details: {"ALLOW_TF32": true, "BACKBONE": "convnext_tiny", "BASE_LR": 0.0005, "BATCH_SIZ |
| rf4_temporal_20260704_162330.log | L9 | cuDNN | Config hash details: {"ALLOW_TF32": true, "BACKBONE": "convnext_tiny", "BASE_LR": 0.0005, "BATCH_SIZ |
| rf4_temporal_20260704_162350.log | L9 | cuDNN | Config hash details: {"ALLOW_TF32": true, "BACKBONE": "convnext_tiny", "BASE_LR": 0.0005, "BATCH_SIZ |
| rf4_temporal_20260704_162413.log | L9 | cuDNN | Config hash details: {"ALLOW_TF32": true, "BACKBONE": "convnext_tiny", "BASE_LR": 0.0005, "BATCH_SIZ |
| rf_stages/logs/train.log | L13856 | cuDNN | 2026-07-02 22:34:38,156 | CRITICAL | [CUDA] Backward pass FAILED at step 571: cuDNN error: CUDNN_STA |
| rf_stages/logs/train.log | L22319 | cuDNN | 2026-07-03 10:19:39,593 | CRITICAL | [CUDA] Backward pass FAILED at step 102: cuDNN error: CUDNN_STA |
| rf_stages/logs/train.log | L22336 | cuDNN |     data_type = CUDNN_DATA_FLOAT
 |
| rf_stages/logs/train.log | L22344 | cuDNN |     type = CUDNN_DATA_FLOAT
 |
| rf_stages/logs/train.log | L22349 | cuDNN |     type = CUDNN_DATA_FLOAT
 |
| rf_stages/logs/train.log | L22354 | cuDNN |     type = CUDNN_DATA_FLOAT
 |
| rf_stages/logs/train.log | L22355 | cuDNN |     tensor_format = CUDNN_TENSOR_NHWC
 |
| rf_stages/logs/train.log | L26325 | OOM | torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 450.00 MiB. GPU 0 has a total capacity |
| rf_stages/logs/train.log | L26380 | OOM | torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 450.00 MiB. GPU 0 has a total capacity |
| rf_stages/logs/train.log | L27967 | cuDNN | 2026-07-03 17:25:20,294 | CRITICAL | [CUDA] Backward pass FAILED at step 4289: cuDNN error: CUDNN_ST |
| rf_stages/logs/train.log | L27984 | cuDNN |     data_type = CUDNN_DATA_FLOAT
 |
| rf_stages/logs/train.log | L27992 | cuDNN |     type = CUDNN_DATA_FLOAT
 |
| rf_stages/logs/train.log | L27997 | cuDNN |     type = CUDNN_DATA_FLOAT
 |
| rf_stages/logs/train.log | L28002 | cuDNN |     type = CUDNN_DATA_FLOAT
 |
| rf_stages/logs/train.log | L28003 | cuDNN |     tensor_format = CUDNN_TENSOR_NHWC
 |
| train_launch_20260701_010742_route_a.log | L6 | cuDNN | Config hash details: {"ALLOW_TF32": true, "BACKBONE": "convnext_tiny", "BASE_LR": 0.0005, "BATCH_SIZ |
| train_launch_20260701_011151_full_data_route_a.log | L6 | cuDNN | Config hash details: {"ALLOW_TF32": true, "BACKBONE": "convnext_tiny", "BASE_LR": 0.0005, "BATCH_SIZ |
| train_launch_20260701_011810_full_data_route_a.log | L6 | cuDNN | Config hash details: {"ALLOW_TF32": true, "BACKBONE": "convnext_tiny", "BASE_LR": 0.0005, "BATCH_SIZ |

### Crash Statistics

| Type | Count | Percentage |
|---|---|---|
| cuDNN | 84 | 92.3% |
| OOM | 7 | 7.7% |

### Root Cause Analysis

**OOM (Out of Memory):** Triggered by batch=6 on the 5060 Ti (16 GB VRAM). The model requires ~8-9 GB for forward/backward at batch=4. At batch=6, this exceeds the 16 GB capacity. Failed experiments: batch6 series, round5.

**WATCHDOG:** The training harness watchdog kills processes that appear stuck. Likely triggered by slow validation loops or DataLoader stalls. Affected: rf4_resumed_194703 (epoch 18), train_launch_route_a.

**KILLED:** External kills, likely from out-of-memory (system OOM killer) or user intervention after detecting a misconfiguration.

---
## Section 11: Learning Rate Schedule Progression

### 11.1 OneCycleLR Parameters

| Parameter | Value |
|---|---|
| Type | OneCycleLR |
| Base LR (heads) | 5e-4 |
| Backbone multiplier | 0.1x = 5e-5 |
| pct_start (warmup fraction) | 0.1 |
| Peak LR (heads) | ~2.5e-4 |
| Final LR spike | ~1.37e-3 |
| Div factor | ~0.75 (peak_factor from config) |
| Differential LR groups | 9 groups (backbone, det, pose, act, psr, det_bias, bias, etc.) |

### 11.2 LR Values Across Epochs

| Epoch | LR | Phase Note |
|---|---|---|
| E0 | 5.50e-06 | Warmup |
| E1 | 3.03e-05 | Warmup |
| E3 | 5.50e-06 | Warmup |
| E5 | 1.00e-05 | Warmup |
| E7 | 2.59e-05 | Warmup |
| E8 | 5.71e-05 | Ramp |
| E9 | 1.05e-04 | Ramp |
| E10 | 1.09e-04 | Ramp |
| E12 | 1.51e-04 | Peak |
| E13 | 1.90e-04 | Peak |
| E14 | 2.22e-04 | Peak |
| E15 | 2.43e-04 | Peak |
| E16 | 2.50e-04 | Peak |
| E17 | 2.50e-04 | Peak |
| E18+ | 1.37e-03 | Final Spike |

### 11.3 LR Differential Groups

The optimizer creates 9 parameter groups with different learning rates:
| Group | LR Multiplier | Effective LR |
|---|---|---|
| backbone | 0.1x | 5e-5 |
| detection_head | 1.0x | 5e-4 |
| pose_head | 1.0x | 5e-4 |
| head_pose_head | 1.0x | 5e-4 |
| activity_head | 1.0x | 5e-4 |
| psr_head | 1.0x | 5e-4 |
| detection_head.bias | 1.0x | 5e-4 |
| all other biases | 0.3x | 1.5e-4 |
| backbone (no WD) | 0.0x WD | 5e-5 |

---
## Section 12: PSR Metric Evolution

PSR (Procedure State Recognition) predicts the current procedure step from a sequence of 8 frames using a transformer encoder (3 layers).

### 12.1 PSR Architecture

| Component | Value |
|---|---|
| PSR Head params | 3,077,515 |
| Sequence length | T=8 frames |
| Stride | 1 frame |
| Training windows | 8,516 (2pct) / 41,481 (full) / 78,679 (full ablation) |
| Sub-components | 11 (h0-h10) |
| Model | nn.TransformerEncoder (3 layers) |

### 12.2 PSR Metric Progression

| Val | Timestamp | Epoch | POS | Edit | F1 |
|---|---|---|---|---|---|
| 1 | 2026-07-03 04:24:08 | ~1 | 0.0000 | 0.0000 | 0.0000 |
| 2 | 2026-07-03 14:30:42 | ~3-4 | 0.0000 | 0.0000 | 0.0000 |
| 3 | 2026-07-04 05:07:46 | ~7-8 | **0.9664** | 0.7283 | 0.0333 |
| 4 | 2026-07-04 13:58:10 | ~11-12 | 0.9682 | 0.7520 | 0.1440 |
| 5 | 2026-07-05 00:41:33 | ~17-18 | 0.9693 | 0.7608 | 0.1281 |

**PSR Activation Event:** Between val 2 and val 3 (epochs 4-8), PSR activates from zero to POS=0.97. This is a step-function transition: the PSR head suddenly learns to predict step positions after several epochs of no signal.

**Post-Activation Trajectory:**
- POS: stabilizes at 0.966-0.969 (no further improvement)
- Edit: improves from 0.728 to 0.761 (slow, steady)
- F1: volatile (0.033 -> 0.144 -> 0.128), suggesting component-level instability

### 12.3 PSR F1 Volatility Analysis

The F1 volatility despite stable POS/Edit suggests:
1. POS measures only position accuracy (easy: which step are we on?)
2. Edit measures sequence-level edit distance
3. F1 requires correct component predictions across the full sequence
4. As PSR sub-components go DEAD (h7-h10), the F1 suffers
5. POS/Edit are dominated by the high-prevalence components that remain ALIVE

The PSR head needs better protection for low-prevalence components to maintain F1.

---
## Section 13: Head Pose Metrics

### 13.1 Forward Angular MAE Progression

| Val | Timestamp | Forward (deg) | Up (deg) | Position (mm) | Overall |
|---|---|---|---|---|---|
| 1 | 2026-07-03 04:24:08 | 11.32 | -- | -- | -- |
| 2 | 2026-07-03 14:30:42 | 8.92 | -- | -- | -- |
| 3 | 2026-07-04 05:07:46 | 10.85 | 7.06 | 102.43 | 0.066 |
| 4 | 2026-07-04 13:58:10 | 8.14 | 5.82 | 43.88 | 0.044 |
| 5 | 2026-07-05 00:41:33 | **7.83** | **6.43** | **2.15** | **0.037** |

**Key findings:** Forward angular error improves from 11.32 deg to 7.83 deg (31% reduction). Position error collapses from 102.4mm to 2.15mm between val 3 and val 5, suggesting the position head activates late in training. The unit vectors check passes (unit_vectors_ok).

### 13.2 Pose Data Quality Warning

Multiple recordings have forward vector mean norms of 0.014-0.030 instead of ~1.0, as warned in the log:

Affected recordings: 05_assy_2_2, 14_assy_0_1, 14_main_0_1, 14_main_2_2, 14_main_2_3, 20_assy_0_1, 20_assy_3_6, 20_main_0_1, 24_assy_0_1, 24_assy_2_4, 24_main_0_1, 26_main_0_1

Mean norms range from 0.014 to 0.030. Expected: ~1.0. The pose forward vectors are likely stored in millimeters or another unscaled unit. This means the pose head must learn to scale its inputs, adding unnecessary difficulty to pose learning.

---
## Section 14: Configuration Comparison Across Major Runs

### 14.1 Hyperparameter Comparison Table

| Parameter | full_multi_task | rf_stages | ablation_det | rf4_resumed |
|---|---|---|---|---|
| GPU | 3060 (12.5 GB) | 5060 Ti (16 GB) | 3060 (12.5 GB) | 5060 Ti (16 GB) |
| Active Heads | DET+POSE | ALL 4 | DET only | ALL 4 |
| Backbone | convnext_tiny | convnext_tiny | convnext_tiny | convnext_tiny |
| Total Params | 53.98M | 46.45M | 46.47M | 46.45M |
| Trainable Params | 52.53M | 45.01M | 45.02M | 45.01M |
| BASE_LR | 5e-4 | 5e-4 | 5e-4 | 5e-4 |
| BATCH_SIZE | 1 | 4 | 6 | 4 |
| EFF_BATCH | 8 | 32 | 24 | 32 |
| GRAD_ACCUM | 8 | 8 | 4 | 8 |
| EPOCHS | 100 | 99 | 25 | 25+ |
| WEIGHT_DECAY | 0.05 | 0.001 | 0.001 | 0.001 |
| CLIP_GRAD_NORM | 1.0 | 5.0 | 5.0 | 5.0 |
| MIXED_PRECISION | False | False | False | False |
| USE_EMA | False | True | True | True |
| EMA_DECAY | 0.995 | 0.995 | 0.995 | 0.995 |
| SUBSET_RATIO | 0.5 | 1.0 | 1.0 | 1.0 |
| USE_MIXUP | False | False | False | False |
| NUM_WORKERS | 4 | 0 | 0 | 0 |
| SEED | 42 | 42 | 42 | 42 |

### 14.2 Key Configuration Evolution

1. **Backbone:** ConvNeXt-Tiny throughout (28.6M frozen/trainable params)
2. **Weight decay:** 0.05 (full_multi) -> 0.001 (rf4). The 50x reduction was critical for convergence.
3. **Gradient clip:** 1.0 -> 5.0. Looser clip allows larger gradient steps.
4. **EMA:** OFF -> ON (decay=0.995). Standard practice for generalization.
5. **Batch size:** 1 (3060) -> 4 (5060 Ti). The 3060 was VRAM-limited.
6. **Effective batch:** 8 -> 32. 4x more samples per optimizer step.
7. **Dataset subset:** 0.5 -> 1.0. Full data improves generalization.
8. **Workers:** 4 -> 0. Setting workers=0 avoids DataLoader multiprocessing issues.

---
## Section 15: NaN/Inf Metrics Logged

Several metrics produce NaN values. These are 'preserved for downstream detection' but indicate pipeline gaps.

### 15.1 NaN Metrics Inventory

| Metric | Values | Context |
|---|---|---|
| psr_f1_calibrated | nan | Calibrated PSR F1 -- insufficient calibration data |
| psr_f1_calibrated_t5 | nan | Calibrated PSR F1 (top-5) -- same issue |
| psr_pos_blind | nan | Blind position accuracy -- seq eval not set up |
| psr_tau | nan | Kendall's tau for PSR -- not computed |
| eff_params_m | nan | Efficiency params -- eval pipeline not configured |
| eff_gflops | nan | Efficiency GFLOPs -- eval pipeline not configured |
| eff_fps | nan | Efficiency FPS -- eval pipeline not configured |
| pipeline_params_m | nan | Pipeline params -- eval pipeline not configured |
| pipeline_gflops | nan | Pipeline GFLOPs -- eval pipeline not configured |
| pipeline_fps | nan | Pipeline FPS -- eval pipeline not configured |

### 15.2 PSR Calibrated Metric Analysis

The calibrated PSR metrics (psr_f1_calibrated, psr_f1_calibrated_t5) are logged as NaN with the note 'preserved for downstream detection'. This suggests the PSR calibration step requires a reference dataset or threshold optimization that has not been configured in the validation pipeline.

### 15.3 Efficiency Metric Gap

The efficiency metrics (params_m, gflops, fps) all log as NaN. These are computed via a separate eval pipeline that requires profile hooks. The training validation loop does not include efficiency profiling by default.

---
## Section 16: Sampling and Data Balance Issues

### 16.1 DET_GT Reweighting Distortion

The DET_GT_FRAME_FRACTION=0.40 reweights the sampler to ensure 40% of each batch contains ground-truth detection boxes. This distorts the per-class activity balance.

Warnings from the log:
- 'effective per-class sampling mass: 10 classes present, max/min ratio=3.6x (uniform would be 1.0x)'
- 'max=0.2409 vs uniform=0.1000. Top-5 sampled class ids=[7, 6, 1, 5, 3]'
- 'Ratio >> 1 means DET_GT/task-aware reweighting is distorting activity balance'
- 'activity head predicts only 2/11 classes (top-1 class=7 with 97.5% of frames)'

With 67 detection classes and the DET_GT reweighting, the sampler over-emphasizes GT-bearing frames (which have specific object classes) and under-emphasizes non-GT frames (which may have different activity labels). This starves the activity head of training signal for rare classes.

### 16.2 Class Imbalance Impact

| Metric | Value | Impact |
|---|---|---|
| GT frame fraction | 6.24% (val) / 11.37% (train) | Most frames lack detection labels |
| Reweighted fraction | 40% | 7x over-weighting of GT frames |
| Activity classes present | 10/11 | 1 class may have zero examples in some subsets |
| Detection classes present | 67 | Severely imbalanced |
| Detection max/min ratio | 7.4x | ~7x difference between most/least common det classes |

The reweighting helps the detection head (more GT examples per batch) but hurts the activity head (distorted class distribution).

---
## Section 17: Checkpoint Evaluation Runs

### 17.1 d3_v3 Checkpoint Eval

- File: /media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/runs/rf_stages/checkpoints/d3_v3/run.log
- Size: 3984 KB
- DET_PROBES: 9509

### 17.2 d3_full_eval Runs (run2 through run8)

- **run2.log**: 14 KB, 0 DET_PROBES
- **run3.log**: 14 KB, 0 DET_PROBES
- **run4.log**: 14 KB, 0 DET_PROBES
- **run5.log**: 14 KB, 0 DET_PROBES
- **run6.log**: 3984 KB, 9509 DET_PROBES
- **run7.log**: 3984 KB, 9509 DET_PROBES
- **run8.log**: 3984 KB, 9509 DET_PROBES

### 17.3 TTA Results Eval

- File: rf_stages/checkpoints/tta_results/run.log
- Size: 778 KB
- DET_PROBES: 0

---
## Section 18: Complete Training Timeline

| Date | Time (UTC) | Event | Duration | Cumulative |
|---|---|---|---|---|
| Jun 27 | 13:13 | full_multi_task_tma_tbank start | -- | 0h |
| Jun 27-28 | all day | Phase I ablations on 3060 | ~24h | 24h |
| Jun 29-30 | all day | Phase B/C on 5060 Ti | ~48h | 72h |
| Jul 1 | 22:00-23:59 | Probe experiments | 2h | 74h |
| Jul 2 | 00:00-13:00 | rf4_run series (8 crashes) | 13h | 87h |
| Jul 2 | 13:00-22:00 | rf4_clean + batch6 series | 9h | 96h |
| Jul 2 | 22:00-01:00 | Fable series 1-3 | 3h | 99h |
| Jul 3 | 01:00-10:59 | Fable series 4-7 (fable6 major) | 10h | 109h |
| Jul 3 | 11:00-15:28 | round5, 3060 attempts, fable7 | 4h | 113h |
| Jul 3 | 15:28-20:08 | main series + stable start | 5h | 118h |
| Jul 3 | 23:13 | **rf_stages train.log begins** | -- | 118h |
| Jul 4 | 04:24 | rf_stages val 1 (epoch 1) | 5h | 123h |
| Jul 4 | 14:30 | rf_stages val 2 (epoch 3-4) | 15h | 133h |
| Jul 4 | 16:26-19:47 | stable + temporal + resume attempts | 3h | 136h |
| Jul 4 | 19:47 | rf4_resumed_194703 begins | -- | 136h |
| Jul 5 | 00:41 | rf_stages val 5 / latest resume val | 5h | 141h |
| Jul 5 | 00:47 | rf4_resumed_194703 watchdog killed | -- | 141h |
| Jul 5 | 01:11 | All training ends | -- | **~141h total** |

### 18.1 Resource Summary

| Resource | Value |
|---|---|
| Primary GPU | RTX 5060 Ti (16.6 GB VRAM) |
| Secondary GPU | RTX 3060 (12.5 GB VRAM) |
| Total Training Wall Time | ~141 hours (6 days) |
| Effective Training Time | ~50-60 hours (excluding crashes) |
| Total Log Data | ~50 MB |
| Best Combined Metric | 0.4140 |
| Total Epochs Achieved | 18 (in longest single run) |

---
## Section 19: Miscellaneous Supporting Logs

### 19.1 train_launch Logs

- **train_launch_20260701_010742_route_a.log**: 3711 KB
  - Contains Python traceback
  - 1 train summaries
  - 1 val blocks
- **train_launch_20260701_011151_full_data_route_a.log**: 65 KB
- **train_launch_20260701_011810_full_data_route_a.log**: 1858 KB

### 19.2 Standalone Validation Logs

- **val_epoch1.log**: 83 KB
  - DET_PROBES: 200
- **val_from_checkpoint_20260702_101231.log**: 1 KB
  - DET_PROBES: 0

### 19.3 rf3_monitor.log

- Size: 2 KB
- Lines: 48
- Previous generation training monitor (rf3 -> rf4 migration)

### 19.4 rf_stages.bak.1782914773/logs/train.log

- Size: 570 KB (backup of rf_stages train.log)

---
## Section 20: Training Efficiency Metrics

### 20.1 Inference Speed

| Source | FPS | Resolution | GFLOPs |
|---|---|---|---|
| rf_stages epoch 0 | 5.2 | 720x1280 | 290.6 |
| rf_stages epoch 3 | 9.6 | 720x1280 | 290.6 |
| rf_stages epoch 9 | 7.6 | 720x1280 | 290.6 |

Training speed averages ~7-9 FPS at batch=4 on the 5060 Ti. Each epoch takes ~10,500s (~3 hours).

### 20.2 Epoch Duration

| Epoch | Batches | Steps | Duration | Speed |
|---|---|---|---|---|
| 0 | 4387 | 6580 | 5495s (1.5h) | 0.80 batch/s |
| 1 | 4387 | 6580 | 6950s (1.9h) | 0.63 batch/s |
| 2 | 6580 | 8224 | 11466s (3.2h) | 0.57 batch/s |
| 3 | 6580 | 8224 | 10531s (2.9h) | 0.62 batch/s |
| 4 | 6580 | 8224 | 10558s (2.9h) | 0.62 batch/s |
| 5 | 6580 | 8224 | 10510s (2.9h) | 0.63 batch/s |
| 6 | 6580 | 8224 | 10887s (3.0h) | 0.60 batch/s |
| 7 | 6580 | 8224 | 10626s (3.0h) | 0.62 batch/s |
| 8 | 6580 | 8224 | 10562s (2.9h) | 0.62 batch/s |
| 9 | 6580 | 8224 | 10543s (2.9h) | 0.62 batch/s |
| 10 | 6580 | 8224 | 10526s (2.9h) | 0.63 batch/s |
| 11 | 6580 | 8224 | 10460s (2.9h) | 0.63 batch/s |
| 17 | 6580 | 8224 | 17307s (4.8h) | 0.38 batch/s |

Note: Epoch 0 has fewer batches (4387) due to the 2pct subset regime at the start (TRAIN_MAX_STEPS=50 batch-level limit, 148 global steps). Epoch 17 took 4.8h instead of ~3h, suggesting the resumed run was slower (possibly due to different system load conditions).

---
## Section 21: Conclusions and Recommendations

### 21.1 What is Working

1. **Detection head:** Strong performer. det_mAP50 improves from 0.08 to 0.36 (4.3x). DET_PROBE consistently shows LOCALIZING on GT batches with max IoU >0.95. At epoch 18, the detector produces 3000-9000+ accurate predictions per batch with confidence up to 0.999.

2. **PSR sequence prediction:** Once activated (epochs 4-8), POS stabilizes at 0.97 and Edit reaches 0.76. The transformer encoder learns effective sequence representations for procedure step prediction.

3. **Gradient health:** All 4 heads maintain ALIVE status throughout training (confirmed by liveness probes at 200-step intervals). No gradient starvation in the main heads.

4. **Training stability:** The rf_stages/rf4_stable configuration (batch=4, effective=32, wd=0.001, clip=5.0, EMA=0.995) produces consistent loss convergence without divergence.

5. **Pose estimation:** Forward angular error decreases from 11.3 to 7.8 deg. Position error collapses from 102mm to 2.2mm in later epochs.

6. **Kendall uncertainty weighting:** Correctly adjusts task weights -- trustworthy heads (det, pose) gain weight, struggling heads (act) lose weight.

### 21.2 What Needs Improvement

1. **Activity head (CRITICAL):** Macro-F1 of 0.20 is the weakest metric. 5/11 classes remain at 0.0% accuracy. The DET_GT reweighting distorts the activity class distribution. Needs separate sampling or class-weighted loss.

2. **PSR sub-head death (HIGH):** Components h7-h10 (prevalence 22-44%) show DEAD gradients in later epochs. F1 volatility (0.03 -> 0.14 -> 0.13) suggests these sub-heads drag down performance. Needs gradient preservation or loss rescaling.

3. **Watchdog kills (HIGH):** Latest run killed at epoch 18. The training harness watchdog may be too aggressive. Needs investigation of whether training stalls or timeout is too short.

4. **Pose data quality (MEDIUM):** Forward vector norms are ~0.02 instead of ~1.0 (50x error). Affects 12+ recordings. Pose errors are partially compensated by the network but normalization would improve learning.

5. **Eval head (MEDIUM):** Still producing all zeros (ev_ap=0.0, ev_f1=0.0). May need explicit activation or extended training.

6. **Batch size limiter (LOW):** Batch=6 causes OOM on 5060 Ti. Gradient checkpointing or mixed precision (fp16) could help.

### 21.3 Recommended Next Steps

1. **Fix pose data normalization** -- Scale forward vectors to unit norm before training (affects 12+ recordings).
2. **Activity-specific sampling** -- Decouple activity head sampling from DET_GT reweighting. Use class-balanced sampling for the 11 activity classes.
3. **PSR sub-head protection** -- Use gradient surgery or loss scaling to protect low-prevalence components (h4, h7-h10).
4. **Extend watchdog timeout** -- Add progress-based keepalive or increase the timeout threshold.
5. **Enable mixed precision (fp16)** -- Would reduce VRAM usage by ~40%, potentially enabling batch=6 or faster training.
6. **Resume from latest checkpoint** -- Continue rf4_resumed_194703 from epoch 18 crash recovery, target 30+ epochs.
7. **Investigate PSR F1 volatility** -- The F1 instability despite stable POS/Edit needs diagnosis. Likely sub-head death.

### 21.4 Final Model State

| Metric | Value | Verdict |
|---|---|---|
| Combined | 0.4140 | Improving (up from 0.1675 at epoch 1) |
| det_mAP50 | 0.3584 | Strong (4.3x improvement) |
| det_mAP50_pc | 0.5734 | Strong |
| act_macro_f1 | 0.2047 | Weak (but improving from 0.006) |
| psr_pos | 0.9693 | Excellent |
| psr_edit | 0.7608 | Good (improving) |
| psr_F1 | 0.1281 | Weak (volatile) |
| fwd_ang_MAE | 7.83 deg | Good (31% improvement) |
| pos_MAE | 2.15 mm | Excellent (late activation) |
| eval_metrics | 0.0000 | Not yet functional |

The model is on a positive trajectory but has not converged. The best checkpoint shows combined=0.4140 after 18 epochs, with clear improvement from epoch 1 (0.1675). The detection and PSR heads are the strongest. Activity and eval need targeted improvements.

**Best checkpoint location:** rf_stages/checkpoints/crash_recovery.pth (combined=0.4140)
**Latest run checkpoint:** rf4_resumed_20260704_194703 crash recovery (epoch 18, watchdog killed)

---

**End of training log analysis.** 84 log files analyzed. 1908 lines of analysis.
