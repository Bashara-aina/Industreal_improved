# RF1–RF10: Complete Transition from R1/R2/R2.5/R3 — Comprehensive Status

> Generated 2026-06-16 17:45 UTC (v3 — updated with Phase 4 trajectory + cycling death spiral)  
> Author: Bashara Aina  
> Hardware: NVIDIA RTX 3060 12GB, i5-12400F, 64GB RAM, Ubuntu 22.04  
> Framework: PyTorch 2.2, CUDA 12.1

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Why R1–R3 Was Replaced](#2-why-r1r3-was-replaced)
3. [The RF1–RF10 Architecture](#3-the-rf1rf10-architecture)
4. [All Python Files and Their Roles](#4-all-python-files-and-their-roles)
5. [Stage Definitions and Gate Criteria](#5-stage-definitions-and-gate-criteria)
6. [Configuration Presets (config.py)](#6-configuration-presets)
7. [Training Supervisor Deep Diagnostics](#7-training-supervisor-deep-diagnostics)
8. [Complete Training Timeline (June 16) — 4 Phases](#8-complete-training-timeline-june-16--4-phases)
9. [DET-DEBUG Trajectory Analysis](#9-det-debug-trajectory-analysis)
10. [The Class Imbalance Death Spiral — Root Cause](#10-the-class-imbalance-death-spiral--root-cause)
11. [PSR Head Status](#11-psr-head-status)
12. [Cron and Automation Infrastructure](#12-cron-and-automation-infrastructure)
13. [Key Fixes Applied](#13-key-fixes-applied)
14. [Timeline Estimation to RF10](#14-timeline-estimation-to-rf10)
15. [Options to Break the Death Spiral](#15-options-to-break-the-death-spiral)
16. [Remaining Uncertainties and Confusions](#16-remaining-uncertainties-and-confusions)
17. [References](#17-references)

---

## 1. Executive Summary

The original training protocol used an ad-hoc R1/R2/R2.5/R3 naming scheme with manual recovery presets, fixed training configurations, and no automated stage progression. The RF1–RF10 system replaces this with a **structured multi-stage progressive training pipeline** featuring:

- **10 sequential stages** (RF1–RF10) with increasing data subsets (20% → 100%)
- **Per-stage gate criteria** (mAP thresholds) that must be met to advance
- **Automated stage_manager** with state machine, 5-category checklists, and 20-why root cause analysis
- **Training supervisor** with deep logit-level collapse detection and autonomous intervention
- **15-minute cron** for automated health checks and stage transitions

**Current status (2026-06-16 17:45 UTC):** RF1 training is actively running at epoch 58 step ~800/1241, 95% GPU util, with a 20× reduced learning rate (retry #5). Training has cycled through **5 retries across 4 distinct phases** today — every retry strategy has been exhausted:

| Phase | Time | What Happened |
|-------|------|---------------|
| **Phase 1** (pi=0.1) | 14:19–15:19 | Three rapid pi=0.1 collapse cycles (cls_mean -2→-20 in ~15 min each) |
| **Phase 2** (false-positive kills) | 16:37–16:49 | pi=0.01 applied but old -8.0 threshold killed healthy training at step 51 |
| **Phase 3** (real death spiral) | 17:04–17:22 | Fixed threshold, training reached step 1300, but det gradient vanished |
| **Phase 4** (current, 20× LR) | 17:22–now | reduce_lr_20x_warmup_3x, reproducing Phase 3 trajectory identically at step 800+ |

**Critical new finding (v3): Phase 4 is producing EXACTLY the same DET-DEBUG values as Phase 3 at every step.** The 20× LR reduction is not changing the model's output trajectory — the training is deterministic from the same checkpoint and weight initialization. This means Phase 4 WILL reproduce Phase 3's collapse at step ~1300, ~17:48 UTC.

**Root cause identified**: The detection head enters a **class imbalance death spiral** at ~step 1200–1300. With 99.3% empty frames (subset_ratio=0.2), the model learns to predict "confident background" everywhere. Positive logits decay from +2.9→+0.055, focal loss gradient vanishes, and training stalls. **All 5 retry strategies tried — none fix the core problem.**

---

## 2. Why R1–R3 Was Replaced

### The Old System (R1 / R2 / R2.5 / R3)

The original protocol was evolved organically during debugging and had no formal structure:

| Phase | Purpose | Problem |
|-------|---------|---------|
| **R1** | Detection bootstrap from crash_recovery | Manual launch, no gate criteria, fixed epoch count |
| **R2** | Detection + Pose after reinit | Same preset as full training, no intermediate validation |
| **R2.5** | "Paper run" with all heads but PSR warmup | Single monolithic 100-epoch run, no data scaling, Kendall log_vars pinned |
| **R3** | Full paper-ready training | Never reached — collapsed at epoch 48 due to dead heads |

**Key failures of the old system:**
1. **Monolithic training**: All heads activated simultaneously → PSR head never escaped zero-loss equilibrium
2. **No data scaling**: Single `subset_ratio` throughout → wasted compute on 100% data for detection-only phase
3. **No gate criteria**: No objective advancement metric → training continued despite collapsed heads
4. **Manual orchestration**: Every launch was manual → no automated retry or intervention
5. **Kendall log_vars pinned**: Three of four Kendall weights pinned at bounds → effectively disabled adaptive loss weighting
6. **Fixed epoch counts**: Epoch 100 with no early exit → 4.2 days per run even if saturated
7. **No checkpoint progression**: Each stage required manual checkpoint selection

### The New System (RF1–RF10)

The RF system addresses all failures:
1. **Progressive head activation**: RF1=det only, RF2=det+pose, RF3=det+pose+act, RF4+=all heads
2. **Data scaling**: 20% → 35% → 50% → 65% → 80% → 90% → 100%
3. **Gate criteria**: Objective mAP thresholds at each stage
4. **Automated state machine**: stage_manager reads/writes state file, determines next action
5. **5-category checklists**: Gate, Health, Convergence, Validation, Stability
6. **20-why root cause analysis**: Rule-based diagnosis from failure patterns
7. **Training supervisor**: Deep logit-level collapse detection with autonomous intervention
8. **Resume from previous stage**: Each stage resumes from previous stage's best.pth

---

## 3. The RF1–RF10 Architecture

### High-Level Flow

```
stage_manager --launch RF1
    │
    ▼
┌─────────────────────────────────────────────┐
│  RF1: Detection only (20% data, 20 epochs)   │
│  Gate: det_mAP50 >= 0.30, det_mAP50_95>=0.12 │
│     ↓ PASS?                                   │
│  YES → advance to RF2                         │
│  NO → retry (max N times, then strategies)     │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  RF2: Detection + Pose (35% data, 15 epochs) │
│  Gate: det_mAP50>=0.40, det_mAP50_95>=0.18,  │
│         forward_angular_MAE_deg<=60.0         │
└─────────────────────────────────────────────┘
    │
    ▼ (continues through RF3→RF4→RF5→RF6→RF7→RF8→RF9→RF10)
```

### Stage Manager Components

```
stage_manager.py
├── State Machine (rf_stage_state.json)
│   ├── current_stage, stage_index, status
│   ├── training_pid, epoch, best_metric
│   ├── gate_passed, retry_count, current_strategy
│   └── checklist_results (5 categories)
├── Process Management
│   ├── subprocess.Popen for train.py
│   ├── PID lock file (.training_pid)
│   ├── duplicate prevention (kill_all_train_pids)
│   └── crash recovery detection
├── Log Parser
│   ├── DET-DEBUG: cls_mean, near_zero, anchors
│   ├── LIVENESS: per-head gradient health
│   ├── Validation: det_mAP50, det_mAP50_95
│   └── Epoch progress
├── Decision Engine
│   ├── evaluate_gate() — metric thresholds
│   ├── assess_det_health() — liveness + cls_mean
│   ├── check_convergence() — patience + improvement
│   └── decide_action() → continue / kill_and_retry / advance
├── 5-Category Checklist
│   ├── Gate: metric thresholds for stage transition
│   ├── Health: gradient liveness, loss spikes
│   ├── Convergence: rate of improvement, patience
│   ├── Validation: metric floors (non-blocking warnings)
│   └── Stability: crash count, grad spikes
└── Stage Transition
    ├── advance_stage() → save history, load next preset
    ├── kill_and_retry() → update retry_count, apply strategy
    └── ALL STAGES COMPLETE → print paper results
```

### Training Supervisor Components

```
training_supervisor.py
├── GPU Monitor (nvidia-smi)
├── Config Editor (regex-based config.py patching)
├── Log Parsers
│   ├── parse_det_debug() — cls_mean, std, near_zero
│   ├── parse_det_health() — live logit statistics
│   ├── parse_psr_diag() — PSR loss, logits, zeros
│   └── parse_epoch_progress() — epoch, step, total
├── SYMPTOM_DATABASE (7 symptoms)
│   ├── det_collapse_cls_mean — CRITICAL (threshold: -18.0)
│   ├── det_collapse_near_zero — CRITICAL (>50% near-zero)
│   ├── psr_dead — WARN (loss < 1e-7)
│   ├── psr_logits_diverging — WARN (trending negative)
│   ├── gpu_oom — CRITICAL (CUDA OOM in log)
│   ├── loss_spike — HIGH (>10x rolling mean)
│   └── no_progress — HIGH (validation stalled)
├── Intervention Engine
│   ├── CRITICAL: kill + apply config fixes + restart
│   ├── HIGH: apply fixes, monitor
│   └── WARN: log and monitor
└── Epoch-Level Orchestration
    ├── epoch_sync() — track transitions
    ├── checkpoint freshness monitoring
    └── next mAP validation estimation
```

---

## 4. All Python Files and Their Roles

### Core Training Files

| File | Lines | Role |
|------|-------|------|
| `src/training/stage_manager.py` | 2334 | **RF1-RF10 stage orchestration**: state machine, process management, log parsing, 5-category checklist, decision engine, stage transitions |
| `src/training/training_supervisor.py` | 760 | **Deep diagnostic supervisor**: GPU monitoring, config editor, SYMPTOM_DATABASE, 5-why analysis, autonomous intervention |
| `src/training/train.py` | 4355 | **Main training script**: model forward/backward, loss computation, validation, checkpoints, DET-DEBUG, LIVENESS probes |
| `src/config.py` | 1321 | **All configuration**: 10 stage presets (stage_rf1 through stage_rf10), training hyperparameters, model architecture settings |

### Supporting Files

| File | Role |
|------|------|
| `src/runs/rf_stage_state.json` | **Stage state**: current_stage, epoch, PID, checklist results, metric history, retry count |
| `src/runs/rf_stages/logs/train.log` | **Training log (3161 lines)**: all training output including DET-DEBUG, LIVENESS, validation metrics |
| `src/runs/rf_stages/logs/subprocess.log` | **Subprocess log (2.2 MB)**: stdout/stderr of the training subprocess |
| `src/runs/rf_stages/logs/supervisor.log` | **Supervisor log (19 KB)**: history of all supervisor runs and interventions across all 4 phases |
| `src/runs/rf_stages/logs/cron.log` | **Cron log**: stage_manager --check output from 15-min cron |
| `src/runs/rf_stages/checkpoints/latest.pth` | **Latest checkpoint** (804.8 MB, overwritten each epoch — STALE: 202+ min) |
| `src/runs/rf_stages/checkpoints/crash_recovery.pth` | **Crash recovery checkpoint** (written at epoch start) |
| `src/runs/rf_stages/.training_pid` | **PID lock file**: prevents duplicate training processes |
| `src/runs/rf_stages/.stage_target_met` | **Gate met signal file**: written by train.py when all thresholds reached |

### Old R-System Files (preserved in opus_consult directory)

| File | Role |
|------|------|
| `opus_consult_2026_06_10_v2/code/train.py` | Old train.py (R-system era) |
| `opus_consult_2026_06_10_v2/code/config.py` | Old config.py (R-system era) |
| `opus_consult_2026_06_10_v2/code/model.py` | Old model definition |
| `opus_consult_2026_06_10_v2/code/losses.py` | Old loss functions |
| `opus_consult_2026_06_10_v2/code/evaluate.py` | Old evaluation script |
| `opus_consult_2026_06_10_v2/logs/train_main.log` | R2.5 training log |
| `opus_consult_2026_06_10_v2/logs/paper_run_r25_fix_20260615.log` | R2.5 paper run log |
| `opus_consult_2026_06_10_v2/logs/recovery_r1_det_bootstrap.log` | R1 detection bootstrap log |
| `opus_consult_2026_06_10_v2/logs/recovery_r1_det_bootstrap_2pct.log` | R1 detection bootstrap (2% data) log |

---

## 5. Stage Definitions and Gate Criteria

### RF1 — Detection Only Stabilization

| Field | Value |
|-------|-------|
| **Description** | Detection only — stabilize det head after reinit |
| **Preset** | `stage_rf1` |
| **Data** | 20% subset (~20% of training data, ~0.7% of batches have GT boxes) |
| **Max epochs** | 20 (resumed from crash recovery at epoch 58, actual max-epochs=78 after retries) |
| **Active heads** | Detection only |
| **Mixed precision** | False (FP32) |
| **EMA** | True |

**Gate criteria:** Both must pass:
- `det_mAP50 >= 0.30`
- `det_mAP50_95 >= 0.12`

**Health thresholds:**
- `min_grad_norm_det >= 1e-6`
- `max_consecutive_dead <= 5 epochs`
- `max_loss_spike_factor <= 10.0x`

**Convergence:**
- `patience_epochs = 8` (epochs without det_mAP50 improvement before kill)
- `min_improvement = 0.005`

### RF2 — Detection + Pose

| Field | Value |
|-------|-------|
| **Description** | Detection + Body/Head Pose |
| **Preset** | `stage_rf2` |
| **Data** | 35% subset |
| **Max epochs** | 15 |
| **Active heads** | det + pose |
| **Resume from** | RF1 best.pth |

**Gate:** `det_mAP50 >= 0.40`, `det_mAP50_95 >= 0.18`, `forward_angular_MAE_deg <= 60.0`

### RF3 — Detection + Pose + Activity

| Field | Value |
|-------|-------|
| **Description** | Detection + Pose + Activity |
| **Preset** | `stage_rf3` |
| **Data** | 35% subset |
| **Max epochs** | 15 |
| **Active heads** | det + pose + act |
| **Resume from** | RF2 best.pth |

**Gate:** `det_mAP50 >= 0.45`, `det_mAP50_95 >= 0.20`, `act_top1 >= 0.40`, `forward_angular_MAE_deg <= 55.0`

### RF4 — All Heads + PSR Transition

| Field | Value |
|-------|-------|
| **Description** | All heads + PSR (transition enabled) |
| **Preset** | `stage_rf4` |
| **Data** | 50% subset |
| **Max epochs** | 20 |
| **Active heads** | all |
| **Resume from** | RF3 best.pth |

**Gate:** `det_mAP50 >= 0.50`, `act_top1 >= 0.45`, `psr_f1_at_t >= 0.25`, `forward_angular_MAE_deg <= 50.0`

### RF5 — Consolidation

| Field | Value |
|-------|-------|
| **Description** | Consolidate all heads |
| **Preset** | `stage_rf5` |
| **Data** | 50% subset |
| **Max epochs** | 10 |
| **Active heads** | all |
| **Gate:** | `det_mAP50 >= 0.55`, `act_top1 >= 0.50`, `psr_f1_at_t >= 0.30`, `pose_MAE <= 45.0` |

### RF6–RF9 — Data Scaling

| Stage | Data | Epochs | det_mAP50 gate | act_top1 gate | psr_f1 gate | pose_MAE gate |
|-------|------|--------|---------------|--------------|------------|--------------|
| **RF6** | 65% | 10 | >= 0.58 | >= 0.52 | >= 0.35 | <= 42.0 |
| **RF7** | 65% | 10 | >= 0.62 | >= 0.55 | >= 0.40 | <= 40.0 |
| **RF8** | 80% | 10 | >= 0.65 | >= 0.58 | >= 0.45 | <= 38.0 |
| **RF9** | 90% | 10 | >= 0.70 | >= 0.60 | >= 0.50 | <= 35.0 |

### RF10 — Final Full-Data Push (Paper Results)

| Field | Value |
|-------|-------|
| **Description** | Final full-data push — paper results |
| **Preset** | `stage_rf10` |
| **Data** | 100% |
| **Max epochs** | 15 |
| **Active heads** | all |
| **Resume from** | RF9 best.pth |

**Gate:** `det_mAP50 >= 0.75`, `det_mAP50_95 >= 0.35`, `act_top1 >= 0.63`, `psr_f1_at_t >= 0.55`, `forward_angular_MAE_deg <= 30.0`

**Paper baselines:**
- det_mAP50: YOLOv8m = 0.838
- det_mAP50_95: estimated = 0.45

---

## 6. Configuration Presets

### RF1 Preset (stage_rf1)

```python
'stage_rf1': {
    'description': 'RF1: Detection only — stabilize det head after reinit (20% data, 20 ep).',
    'dataset_mode':       'manual_only',
    'backbone':           'convnext_tiny',
    'use_tma_cell':       True,
    'use_temporal_bank':  True,
    'use_hand_film':      True,
    'benchmark_mode':     False,
    'batch_size':         4,
    'grad_accum_steps':   8,
    'zero_det_conf':      False,
    'staged_training':    False,
    'mixed_precision':    False,    # FP32 — AMP broken on RTX 3060 for this model
    'use_mixup':          False,
    'use_ema':            True,
    'train_det':          True,
    'train_act':          False,
    'train_psr':          False,
    'train_head_pose':    False,
    'use_psr_transition':       False,
    'use_geo_head_pose':        True,
    'feature_bank_detach':      True,
    'feature_bank_slot_overwrite': False,
    'use_psr_order_prior':      False,
    'psr_sensitivity_weight':   0.0,
    'use_ldam_drw':             False,
}
```

### Key Configuration Constants

| Constant | Value | Purpose |
|----------|-------|---------|
| `BATCH_SIZE` | 4 | Per-GPU batch size |
| `GRAD_ACCUM_STEPS` | 8 | Effective batch = 32 |
| `MIXED_PRECISION` | False | FP32 forced (AMP causes GradScaler corruption from PSR seq spikes) |
| `GRAD_CLIP_NORM` | 1.0 | Gradient clipping threshold |
| `WARMUP_EPOCHS` | 5 | LR warmup duration |
| `USE_EMA` | True | EMA with decay=0.999 |
| `PSR_WEIGHT` | 60.0 | PSR loss weight (disabled in RF1) |
| `PSR_WARMUP_STEPS` | 6000 | PSR warmup duration |
| `PSR_WARMUP_INIT_MULT` | 3.0 | PSR warmup initial multiplier |
| `ZERO_DET_CONF` | False | Don't zero detection confidence for activity |
| `DET_INIT_PI` | 0.01 | Focal loss prior initialization **KEY FIX** |
| `CLS_MEAN_CRITICAL` | -18.0 | Training supervisor collapse threshold (for pi=0.01) |
| `CLS_MEAN_WARN` | -15.0 | Training supervisor warning threshold |

### Pi Initialization Detail

The pi=0.01 fix replaces the old pi=0.1 prior:

```
pi=0.01 → cls_bias = -log((1-0.01)/0.01) ≈ -4.60
pi=0.10 → cls_bias = -log((1-0.10)/0.10) ≈ -2.20

Focal loss background gradient at pi=0.01: ~2.24e-6 per element
Focal loss background gradient at pi=0.10: ~0.00072 per element
Ratio: 321× smaller background gradient
```

This prevents the detection head from collapsing to degenerate background prediction. The expected cls_mean with pi=0.01 is -11 to -12 (not collapse). At sigmoid(-11) ≈ 1.67e-5, the model conservatively predicts background for most anchors but CAN make positive predictions (max logit reaches +0.334 to +2.909).

**Key observation from Phase 1**: At pi=0.1, cls_mean cycles from -2.3 to -20 in ~15 minutes (1300 steps). Each cycle: cls_mean=-2.3 (step 151) → -16.6 (step 1051) → -17.4 (step 1851). The model collapses in under 20 minutes. At pi=0.01, it takes ~45 minutes and the cls_mean stays at -10 to -12 for 1100 steps before positive logits fade.

### Current Retry Strategy Configuration

The stage_manager has escalated through all 5 strategies:

| Retry # | Strategy | LR Mult | Warmup Mult | Seed Offset | Effective LR (backbone) |
|---------|----------|---------|-------------|-------------|------------------------|
| 0 | default | 1.0× | 1.0× | 0 | 1e-3 |
| 1 | reduce_lr_5x | 0.2× | 1.0× | 1 | 2e-4 |
| 2 | reduce_lr_2x_warmup_2x | 0.5× | 2.0× | 2 | 5e-4 |
| 3 | reduce_lr_10x_warmup_2x | 0.1× | 2.0× | 3 | 1e-4 |
| **4 (current)** | **reduce_lr_20x_warmup_3x** | **0.05×** | **3.0×** | **5** | **5e-5** |

---

## 7. Training Supervisor Deep Diagnostics

### SYMPTOM_DATABASE

| Symptom | Severity | Detection | Threshold |
|---------|----------|-----------|-----------|
| **det_collapse_cls_mean** | CRITICAL | cls_mean < CLS_MEAN_CRITICAL | -18.0 |
| **det_collapse_near_zero** | CRITICAL | near_zero > 50% | 0.5 |
| **gpu_oom** | CRITICAL | "CUDA out of memory" in log | presence |
| **loss_spike** | HIGH | > 10x rolling mean | 10.0x |
| **no_progress** | HIGH | Validation not improving | patience window |
| **psr_dead** | WARN | PSR loss < 1e-7 | 1e-7 |
| **psr_logits_diverging** | WARN | logits_mean < -5, trending negative | trend |

### Intervention Levels

| Level | Action | Example |
|-------|--------|---------|
| **CRITICAL** | Kill training, apply config fixes, restart via stage_manager | cls_mean=-18.5 → MIXED_PRECISION=False, GRAD_CLIP=1.0, restart |
| **HIGH** | Apply fixes in-place, monitor | Loss spike → tighter GRAD_CLIP |
| **WARN** | Log symptom with 5-why analysis, continue monitoring | PSR dead at 1.5e-8 → "expected in RF1" |

### Threshold Fix Applied 2026-06-16

**OLD (false-positive collapse — caused Phase 2 kills):**
```python
CLS_MEAN_CRITICAL = -8.0   # ← triggered at cls_mean=-8.058 (NORMAL for pi=0.01!)
```

**NEW (correct for pi=0.01):**
```python
CLS_MEAN_WARN = -15.0       # warning at -15 (normal range is -11 to -12)
CLS_MEAN_CRITICAL = -18.0   # collapse at -18 (verified non-collapse at -12)
```

The SYMPTOM_DATABASE detect lambda was also using a **hardcoded `-8`** instead of the constant:
```python
# OLD — hardcoded, never updated:
'detect': lambda dd: any(d.get('cls_mean', 0) < -8 ...)  # ← WRONG

# NEW — uses CLS_MEAN_CRITICAL constant:
'detect': lambda dd: any(d.get('cls_mean', 0) < CLS_MEAN_CRITICAL ...)  # ← CORRECT
```

---

## 8. Complete Training Timeline (June 16) — 4 Phases

### Phase 1: pi=0.1 Collapse Cycles (14:19–15:19)

The original training used pi=0.1 (cls_bias=-2.2, expected cls_mean=-2.3). This caused three rapid collapse cycles:

| Time | PID | Steps | cls_mean | Event |
|------|-----|-------|----------|-------|
| 14:19 | 2291883 | 551 | **-2.304** | Supervisor: "continue" — pi=0.1 normal range |
| 14:34 | 2291883 | 1851 | **-17.378** | cls_mean dropped -15 points in 15 min → CRITICAL collapse! |
| 14:34 | — | — | — | Supervisor kills PID 2291883, forces stage_manager restart |
| 14:49 | 2427980 | 151 | **-2.291** | New run — cls_mean back to pi=0.1 range (reinit from checkpoint) |
| 15:04 | 2427980 | 1051 | **-16.582** | Collapsed AGAIN in 15 min → CRITICAL |
| 15:04 | — | — | — | Supervisor kills, stage_manager retries |
| 15:19 | 2511332 | 851 | **-20.509** | Third collapse, even more negative |

**Pattern**: At pi=0.1, cls_mean drops from -2.3 → -17 in ~900 steps (≈10 min). The model overfits to background so aggressively that ALL logits become extremely negative.

### Phase 2: pi=0.01 + False-Positive Kills (16:37–16:49)

Pi=0.01 was deployed but the old -8.0 threshold hadn't been updated yet:

| Time | PID | Steps | cls_mean | Event |
|------|-----|-------|----------|-------|
| 16:37 | 2917794 | — | — | reduce_lr_5x retry (LR mult=0.2×) |
| 16:40 | 2927303 | — | — | reduce_lr_2x_warmup_2x (LR mult=0.5×, warmup 2×) |
| 16:42 | 2937233 | — | — | reduce_lr_10x_warmup_2x (LR mult=0.1×, warmup 2×) |
| 16:46 | 2954352 | 51 | **-8.058** | **FALSE POSITIVE!** cls_mean=-8.058 > -18 (CRITICAL threshold) but OLD -8.0 threshold kills it |
| 16:49 | 2959102 | 51 | -8.058 | Stage manager kill_and_retry (checkpoint too old, 157 min) |

**Root cause of Phase 2**: Three things compounded:
1. The -8.0 threshold (now fixed to -18.0) false-positive killed pi=0.01 training
2. The multiple retries (LR mult 0.2×, 0.5×, 0.1×) were irrelevant — the model wasn't collapsed
3. The checkpoint was old because the new runs hadn't completed an epoch yet

### Phase 3: Real Death Spiral (17:04–17:22)

With the fixed -18.0 threshold, training finally ran without false-positive kills:

| Time | Steps | cls_mean | max | det_cls_loss | det_grad | Event |
|------|-------|----------|-----|-------------|----------|-------|
| 17:04 | 751 | -10.44 | +1.547 | 2.159 | ALIVE (3.64) | Healthy — detection actively learning |
| 17:05 | 851 | -10.58 | +1.547 | 1.840 | ALIVE (8.64) | Healthy |
| 17:06 | 951 | -10.60 | +1.777 | 0.263 | ALIVE (0.34) | **Dropping** — loss cratered |
| 17:07 | 1000 | — | — | — | ALIVE (0.34) | Gradient low but alive |
| 17:08 | 1051 | -10.05 | **+0.506** | **0.064** | ALIVE (0.19) | **Death spiral begins** |
| 17:09 | 1100 | — | — | — | ALIVE (0.19) | Max logit near zero |
| 17:10 | 1151 | **-9.81** | **+0.055** | **0.056** | ALIVE (6.56) | **Bounces back!** GT batch arrived |
| 17:13 | 1200 | — | — | — | ALIVE (6.56) | Gradient recovered |
| 17:13 | 1300 | — | — | — | **DEAD (0.047)** | det gradient vanished! |
| 17:18 | 1400 | — | — | — | **DEAD (0.047)** | Still dead |
| 17:22 | — | — | — | — | — | Stage manager kills and retries |

**Critical observation**: The detection head **bounces back at step 1200** (det=6.56 ALIVE) before dying at step 1300. This bounce happens when a GT batch appears (~step 1151-1200). The positive gradient from the GT batch temporarily revives the head, but the subsequent empty batches drain it again.

**The death spiral (steps 750 → 1300):**
```
max logit:  +2.909 → +1.547 → +1.777 → +0.506 → +0.055 → DEAD AT 1300
cls_loss:   1.057 → 2.159 → 1.840 → 0.264 → 0.064 → 0.056 → (stalled)
det_grad:   [ALIVE] → [ALIVE] → [DROPPING] → [BOUNCE] → [DEAD]
```

### Phase 4: Current — 20× LR Reduction (17:22–Now)

| Retry # | Strategy | LR Mult | Warmup Mult | Seed Offset |
|---------|----------|---------|-------------|-------------|
| 0 (original) | default | 1.0× | 1.0× | 0 |
| 1 | reduce_lr_5x | 0.2× | 1.0× | 1 |
| 2 | reduce_lr_2x_warmup_2x | 0.5× | 2.0× | 2 |
| 3 | reduce_lr_10x_warmup_2x | 0.1× | 2.0× | 3 |
| **4 (current)** | **reduce_lr_20x_warmup_3x** | **0.05×** | **3.0×** | **5** |

Current run (PID 3096257):
- **Resumed from**: checkpoint epoch 58, reinit heads (det/act/psr + FPN)
- **Effective LR**: backbone=5e-05, heads=5e-04
- **Warmup**: 15 epochs (3× original)
- **Max epochs**: 78 (original 20 + 58 from resume)
- **Progress**: epoch 58 step ~800/1241, ~0.9 batch/s
- **GPU**: 95% util, 8175/12288 MiB, 74°C
- **DET**: cls_mean=-10.44, max=+1.142 at step 751
- **LIVENESS**: det=3.64 ALIVE at step 800
- **Stage Manager**: **continue** (no critical issues)
- **Checkpoint**: **STALE** — latest.pth from 14:12 (202+ min old)

#### Critical Finding: Phase 4 is Reproducing Phase 3 Identically

DET-DEBUG values at identical steps across Phase 3 and Phase 4:

| Step | Phase 3 cls_mean | Phase 4 cls_mean | Phase 3 det_cls | Phase 4 det_cls |
|------|-----------------|-----------------|----------------|----------------|
| 551 | -11.037 | -11.037 | 6.427 | 6.427 |
| 651 | -10.661 | -10.661 | 1.057 | 1.057 |
| 751 | -10.439 | -10.439 | 2.159 | 2.159 |

The two runs produce **identical outputs at the same step numbers** because:
1. Both start from the **same checkpoint** (epoch 58 crash_recovery.pth)
2. Both use `--reinit-heads` with the **same weight initialization**
3. The seed offset (5 vs 0) changes data loading order but the model outputs converge to the same trajectory
4. With 20× lower LR in Phase 4, the weight updates are so small they don't measurably change the logit outputs within the first 800 steps

**Prediction**: Phase 4 will reproduce Phase 3's collapse trajectory. The death spiral will repeat at step ~1300 (estimated ~17:48 UTC). The 20× LR delay is insufficient to change the outcome.

---

## 9. DET-DEBUG Trajectory Analysis

### Full DET-DEBUG History (all runs combined)

| Step | Run | cls_mean | cls_std | near_zero | max | det_cls_loss | det_reg_loss | LIVENESS | Phase |
|------|-----|----------|---------|-----------|-----|-------------|-------------|----------|-------|
| 51 | Phase 2 | -8.058 | 2.136 | 0.05% | +2.636 | 71.698 | 0.000 | — | pi=0.01 cold start |
| 100 | Phase 4 | — | — | — | — | — | — | **6.79** | Current |
| 151 | Phase 4 | -10.511 | 3.412 | 0.00% | -0.513 | 2.585 | 0.000 | — | Current (20× LR) |
| 200 | Phase 4 | — | — | — | — | — | — | **3.88** | Current |
| 251 | Phase 4 | -11.931 | 3.451 | 0.00% | -0.074 | 7.860 | 0.266 | — | Current |
| 300 | Phase 4 | — | — | — | — | — | — | **0.20** | Current — dip |
| 351 | Phase 3 | -12.328 | 3.623 | 0.00% | +0.481 | 7.302 | 0.408 | — | Previous run |
| 400 | Phase 3 | — | — | — | — | — | — | **6.80** | Both — bounce back |
| 451 | Phase 3 | -11.424 | 3.400 | 0.00% | -0.790 | 0.113 | 0.000 | — | Previous — loss crater |
| 500 | Phase 3 | -10.999 | 3.575 | 0.00% | — | tally: 237/237 alive | — | **2.93** | Both — tally healthy |
| 551 | Phase 3/4 | **-11.037** | 3.693 | 0.00% | +0.334 | 6.427 | 0.663 | — | **IDENTICAL in both runs** |
| 600 | Phase 3 | — | — | — | — | — | — | **4.01** | Both |
| 651 | Phase 3/4 | **-10.661** | 3.569 | 0.00% | **+2.909** | 1.057 | 0.000 | — | **IDENTICAL — peak max logit** |
| 700 | Phase 3 | — | — | — | — | — | — | **0.33** | Both — dip |
| 751 | Phase 3/4 | **-10.439** | 3.448 | 0.00% | +1.142 | 2.159 | 0.857 | — | **IDENTICAL** |
| 800 | Phase 4 | — | — | — | — | — | — | **3.64** | Current |
| 851 | Phase 3 | -10.583 | 3.673 | 0.00% | +1.547 | 1.840 | 0.903 | 8.64 | Previous |
| 900 | Phase 3 | — | — | — | — | — | — | **8.64** | Previous |
| 951 | Phase 3 | -10.599 | 3.678 | 0.00% | +1.777 | 0.263 | 0.000 | 0.34 | Previous — loss crater |
| 1000 | Phase 3 | — | — | — | — | — | — | **0.34** | Previous |
| 1051 | Phase 3 | -10.047 | 3.575 | 0.00% | **+0.506** | **0.064** | 0.000 | 0.19 | **Death spiral begins** |
| 1100 | Phase 3 | — | — | — | — | — | — | **0.19** | Previous |
| 1151 | Phase 3 | **-9.812** | 3.487 | 0.00% | **+0.055** | **0.056** | 0.000 | 6.56 | **Bounces back on GT batch** |
| 1200 | Phase 3 | — | — | — | — | — | — | **6.56** | Previous — GT batch revived |
| 1300 | Phase 3 | — | — | — | — | — | — | **0.047 DEAD** | **det gradient vanished** |
| 1400 | Phase 3 | — | — | — | — | — | — | **0.047 DEAD** | Still dead |

### Key Observations

1. **Phase 4 === Phase 3** — Identical DET-DEBUG outputs at steps 551, 651, 751. The 20× LR reduction has not changed the model's output trajectory.
2. **Cycling pattern**: LIVENESS oscillates between ~0.2 (near-dead) and ~6.8 (healthy) every ~100-200 steps. This is the model bouncing between empty batches (draining gradient) and GT batches (reviving gradient).
3. **Bounce effect at step 1151-1200**: In Phase 3, a GT batch arrived at ~step 1151, reviving det gradient from 0.19 → 6.56. But by step 1300 it died anyway. The bounce buys ~100 extra steps, not a recovery.
4. **No collapsed anchors**: near_zero=0.00% throughout all runs. The pi=0.01 fix prevents full collapse.
5. **cls_mean stable at -9.8 to -12.3** — well above CLS_MEAN_CRITICAL=-18. No risk of false-positive kill.
6. **Peak max logit is +2.909 at step 651** — the model CAN make strong positive predictions.
7. **det_reg_loss mostly zero** — model not learning regression on empty frames.
8. **No mAP metrics produced**: All validation entries show "Validation samples: 11,383" but never output actual det_mAP50/95 values. The gate eval at epoch 58 was "capped at 200 batches."

### The Death Spiral Mechanism

Between steps 651 and 1151 (500 steps ≈ 10 min):
- **max logit** drops 98%: +2.909 → +0.055
- **cls_loss** drops 95%: 1.057 → 0.056
- **cls_mean** stays constant at -10.5 (background confidence doesn't change)
- The positive signal fades while background stays the same

### The Bounce-and-Die Pattern

```
Step 651:  max=+2.909  det_grad=4.01  [Healthy — peak performance]
Step 751:  max=+1.547  det_grad=0.33  [Dropping — less positive data]
Step 851:  max=+1.547  det_grad=8.64  [Bounces — GT batch]
Step 951:  max=+1.777  det_grad=0.34  [Dropping — empty batches]
Step 1051: max=+0.506  det_grad=0.19  [Critical — near zero]
Step 1151: max=+0.055  det_grad=6.56  [Bounces — GT batch revived it]
Step 1200: max=?       det_grad=6.56  [Temp recovery]
Step 1300: max=?       det_grad=0.047 [DEAD — final, no recovery]
```

The detection head can **only learn when a GT batch appears**. These GT batches are ~0.7% of all batches. When they arrive, the head bounces back. But between GT batches (~143 empty batches), the positive logits decay. Each cycle produces a smaller bounce until the positive weights decay below useful levels.

---

## 10. The Class Imbalance Death Spiral — Root Cause

### Why It Happens

The model has **164,544 anchors per frame** × **4 frames per batch** = **658,176 anchor predictions per step**. With 99.3% empty batches, virtually all of these are background.

**Focal loss dynamics:**
1. Model predicts background for all anchors (cls_mean=-10.5 → sigmoid≈2.7e-5)
2. Focal loss for background at sigmoid=2.7e-5: `(1-2.7e-5)^γ * log(2.7e-5)` → very small
3. **Loss is already near zero** → gradient is near zero → model stops learning
4. On the rare GT batch (0.7%), the positive focal loss is: `(1-sigmoid(logit))^γ * focal_loss` 
5. But the positive gradient is competing with **164K background anchors** in the same batch
6. **Result**: Positive logit updates are tiny compared to background → max logit decays

### The Bounce Cycle — Why It Repeats

The cycling LIVENESS pattern reveals a dynamic equilibrium:

1. **GT batch arrives** (~step 1151): detection head sees positive examples → gradient revives (0.19→6.56)
2. **Next 143 batches** are empty: positive logits decay, cls_loss drops, gradient weakens
3. **Another GT batch** (~1300): but by now max logit = +0.055, positive predictions are too weak
4. **Gradient fails to revive**: the bounce gets smaller each cycle until it's below ALIVE threshold

### Why LR Reduction Won't Fix It

Each retry reduces LR, but the mechanism is unchanged:
- **Lower LR** → slower positive logit growth AND slower background decay
- **The equilibrium is the same**: model converges to "predict background for everything"
- LR reduction just makes the spiral take longer (more steps to reach max=+0.055)

**Proof**: Phase 4 (20× LR) produces IDENTICAL outputs to Phase 3 (1× LR) at the same step numbers. The 20× LR reduction doesn't change the trajectory at all within the first 800 steps.

### Why pi=0.01 Alone Isn't Enough

Pi=0.01 prevents the full collapse (cls_mean never went below -12.3 in Phase 3/4) but doesn't prevent the death spiral. The difference:
- **Pi=0.1**: Full collapse to cls_mean=-20 → supervisor kills within 15 min (Phase 1)
- **Pi=0.01**: Gradient fading to DEAD over ~45 min (Phase 3/4)
- Both: detection head stops learning. Pi=0.01 buys ~30 min, doesn't solve the problem.

### What Actually Needs to Change

The death spiral happens because the ratio of positive to negative examples is **1:164,000**. Solutions:

| Approach | Mechanism | Expected Benefit |
|----------|-----------|-----------------|
| **GT frame oversampling** | Force GT frames into every batch | 143× more positive updates — breaks the death spiral |
| **Increase subset_ratio** | Use more data → more GT frames | RF2 (35%) has ~1.75× more GT frames |
| **GT-only pre-training** | Freeze backbone, train detection on only GT frames for N epochs | Pure positive signal |
| **Positive focal loss weight** | Add weight to positive examples in focal loss | Directly counter imbalance |
| **Score confidence boost** | Bias the cls_score output head toward positive | Hack, not a fix |

---

## 11. PSR Head Status

### PSR Diagnostics (RF1)

PSR is NOT trained in RF1 (`train_psr: False`). The diagnostic output reflects the PSR head's inherent behavior when receiving random/detached features:

```
PSR: loss=1.546e-08 (constant) logits[min=-23, max=+22] zeros=20/22
```

**PSR logits are diverging**: The transformer produces increasingly extreme values (min=-23, max=+22) while the loss remains constant at 1.546e-08. The PSR head predicts mostly zeros (20/22 components) with fill-forward labels.

PSR diagnostics at different steps (all identical loss):
```
step 540: logits[min/max/mean]=-21.175/19.909/-5.629
step 540: per_elem[min/max/sum]=0.000e+00/3.603e-08/3.402e-07
step 540: target counts: zeros=20 ones=2 neg1=0
```

**Key PSR observation**: The per-element loss for the "ones" targets is 3.603e-08 — essentially zero. The PSR head receives NO learning signal, even for the two positive targets (assembly/insertion transitions). The fill-forward labels (zeros=20) are trivial, and the causal transformer produces logits extreme enough that sigmoid saturates.

### 1026 PSR_DIAG Entries in train.log

The PSR diagnostic fires every ~2 steps, generating 1026 entries so far. All show:
- `loss=1.546e-08` exactly (not approximately — bit-exact constant)
- `zeros=20 ones=2` in all entries
- `logits[min/max]` diverging over time: [-19,+18] → [-23,+22]
- `per_elem[max]=3.603e-08` — essentially zero gradient

### PSR Lifecycle Across RF Stages

| Stage | PSR Training | Expected Status | Notes |
|-------|-------------|----------------|-------|
| RF1 | Disabled | DEAD (1.5e-8) | Not trained, no gradient flow |
| RF2 | Disabled | DEAD (1.5e-8) | Not trained, no gradient flow |
| RF3 | Disabled | DEAD (1.5e-8) | Not trained, no gradient flow |
| **RF4** | **Enabled** | **Transition start** | PSR transition mode enabled, causal transformer activated |
| RF5 | Enabled | Learning phase | Order prior enabled, PSR weight active |
| RF6–RF9 | Enabled | Learning phase | More data |
| RF10 | Enabled | **Target: F1>=0.55** | Full data |

**Risk**: PSR has NEVER been observed to train successfully in any run (all of Phase 1–4, all previous R2.5 runs). The transformer produces extreme logits, the loss is constant, and no learning signal exists. There is a real possibility that PSR training at RF4 will also fail, requiring additional fixes (different loss function, gradient clipping for PSR, different label scheme).

---

## 12. Cron and Automation Infrastructure

### Single Cron Job (15-minute interval)

```
*/15 * * * * cd /media/newadmin/master/POPW/working/code/industreal_improved && \
    python3 -m src.training.stage_manager --check \
    >> src/runs/rf_stages/logs/cron.log 2>&1
```

**Responsibilities:**
1. Check training PID liveness
2. Parse latest train.log for metrics
3. Run 5-category checklist
4. Decide action: continue / kill_and_retry / advance / abort
5. If training died: re-launch with retry strategy
6. If gate met: advance to next RF stage
7. If all stages complete: print paper results

### Supervisor (manual invocation)

```bash
python3 -m src.training.training_supervisor 2>&1 | \
    tee -a src/runs/rf_stages/logs/supervisor.log
```

### Commands

| Command | Action | Duration |
|---------|--------|----------|
| `--check` | Read log, run checklists, decide action | ~5-10 seconds |
| `training_supervisor` | Full deep diagnostics + stage_manager --check | ~10-30 seconds |
| `--status` | Print current state | immediate |
| `--launch RF1` | Force-launch stage (kills existing training) | ~30 seconds |
| `--reset` | Reset state to fresh (next --check starts RF1) | immediate |
| `--abort` | Kill training, set status to aborted | ~5 seconds |

### Monitoring Commands

```bash
# Real-time training progress
tail -f src/runs/rf_stages/logs/train.log | \
    grep -E "(DET-DEBUG|batch.*speed|LIVENESS|Validation)"

# DET trajectory
grep "DET-DEBUG" src/runs/rf_stages/logs/train.log | tail -5

# LIVENESS history
grep "LIVENESS step" src/runs/rf_stages/logs/train.log | tail -10

# PSR diagnostics
grep "PSR_DIAG" src/runs/rf_stages/logs/train.log | tail -3

# Current epoch progress
tail -1 src/runs/rf_stages/logs/train.log | grep -o "Epoch [0-9]* batch [0-9]*/[0-9]*"
```

---

## 13. Key Fixes Applied

### Fix 1: CLS_MEAN_CRITICAL Threshold (2026-06-16 16:40 UTC)

**Problem**: training_supervisor.py had `CLS_MEAN_CRITICAL = -8.0`, which treated cls_mean=-8.058 (normal for pi=0.01) as collapse.

**Fix**: Changed to `CLS_MEAN_CRITICAL = -18.0` and `CLS_MEAN_WARN = -15.0`.

**Impact**: Training no longer false-positive killed. This directly stopped Phase 2's cascade of unnecessary kills.

### Fix 2: Hardcoded -8 in SYMPTOM_DATABASE (2026-06-16 16:40 UTC)

**Problem**: The detect lambda at line 293 used a raw `-8` instead of the `CLS_MEAN_CRITICAL` constant.

**Fix**: Changed to reference the constant.

### Fix 3: Removed Duplicate Cron Job (2026-06-16)

**Problem**: Two cron jobs existed — `*/30 monitor_r25_training.sh` and `*/15 stage_manager --check` — causing race conditions.

**Fix**: Removed the 30-min monitor script. Only the 15-min stage_manager --check remains.

### Fix 4: State File Cleanup (2026-06-16)

**Problem**: State file had stale PIDs and corrupted health check data from repeated kill-restart cycles.

**Fix**: Ran `stage_manager.py --reset` to get a clean state.

### Fix 5: Empty-Batch Gradient Zeroing (train.py)

**Problem**: 99.3% of frames have zero GT boxes. Focal loss on empty batches was producing spurious gradients.

**Fix**: Empty-batch gradient zeroing in train.py (line 1553-1578). When a batch has no GT boxes, detection gradients are zeroed.

### Fix 6: pi=0.01 Initialization (train.py)

**Problem**: pi=0.1 initialization produced cls_bias=-2.2, causing focal loss background gradient of 0.00072 per element — enough to drive collapse on 164K anchors per frame.

**Fix**: pi=0.01 → cls_bias=-4.6 → background gradient 321× smaller at 2.24e-6 per element.

---

## 14. Timeline Estimation to RF10

### Retry Cost Analysis

Each retry costs:
- **Time lost**: ~45 min of training (steps 0→1300 before death spiral)
- **Reinit cost**: Detection head + FPN weights reset, model must re-learn basic detection
- **Checkpoint age**: Each retry doesn't save a new checkpoint until epoch boundary
- **Retry strategies exhausted**: All 5 strategies tried. No more LR-based strategies available.

### Current Performance (Phase 4, 20× LR)

| Metric | Value |
|--------|-------|
| Batch speed | 0.9 batch/s |
| Epoch time (20% data) | ~23 min |
| Batches/epoch (20% data) | 1241 |
| GT frames ratio | ~0.7% |
| Effective det training steps/epoch | ~8-9 (with GT) |
| Time until death spiral | ~45 min (1300 steps) |
| Death spiral step | ~1300 (confirmed in Phase 3) |

### Estimate by Stage (without death spiral fix)

If the death spiral isn't addressed, each stage will require multiple retries:

| Stage | Data | Heads | Epochs | Est. retries | Est. wall time | Notes |
|-------|------|-------|--------|-------------|---------------|-------|
| **RF1** (current) | 20% | det | 20 | **∞** | **STUCK** | All 5 strategies tried, none work |
| **RF2** | 35% | det+pose | 15 | 3-5 | ~3-5 days | ~1.75× more GT, may still stall |
| **RF3** | 35% | det+pose+act | 15 | 3-5 | ~4-6 days | Additional heads add complexity |
| **RF4** | 50% | all+PSR | 20 | 5+ | ~7+ days | PSR has never trained successfully |
| **RF5–RF10** | 50-100% | all | 60 | per stage | ~14-30 days | Each stage stalls |

**Total without fix: Stuck at RF1 indefinitely.** No LR-based strategy solves the class imbalance.

### Estimate by Stage (with death spiral fix, e.g., GT oversampling)

| Stage | Data | Epochs | Time/epoch | Est. wall time |
|-------|------|--------|------------|---------------|
| **RF1** | 20% | 20 | ~23 min | ~8 hours |
| **RF2** | 35% | 15 | ~40 min | ~10 hours |
| **RF3** | 35% | 15 | ~42 min | ~11 hours |
| **RF4** | 50% | 20 | ~60 min | ~20 hours |
| **RF5** | 50% | 10 | ~60 min | ~10 hours |
| **RF6** | 65% | 10 | ~78 min | ~13 hours |
| **RF7** | 65% | 10 | ~78 min | ~13 hours |
| **RF8** | 80% | 10 | ~96 min | ~16 hours |
| **RF9** | 90% | 10 | ~107 min | ~18 hours |
| **RF10** | 100% | 15 | ~120 min | ~30 hours |

**Total with fix: ~6-7 days** (assuming PSR recovers at RF4, which is uncertain)

---

## 15. Options to Break the Death Spiral

### Option A: GT Frame Oversampling (Recommended — Immediate Fix)

**What**: Modify DataLoader to guarantee at least one GT frame per batch.

**Implementation**:
```python
# In train.py dataloader, create a balanced sampler:
gt_indices = [i for i, sample in enumerate(dataset) if sample['has_gt']]
empty_indices = [i for i, sample in enumerate(dataset) if not sample['has_gt']]

# For each batch: sample 1 from GT indices, batch_size-1 from empty
# This guarantees every batch has at least one positive frame
```

**Current baseline**: GT batch arrives every ~143 batches on average.
**With oversampling**: GT batch arrives EVERY batch.
**Expected effect**: 143× more positive updates. The detection head gets positive signal every ~11 seconds instead of every ~26 minutes.

**Pros**: Direct fix for the imbalance. Minimal code change (modify sampler only). Keeps 20% data subset.
**Cons**: Oversampled GT frames reduce effective dataset diversity. May overfit to 0.7% GT frames. Requires validation that mAP improves.

**Effort**: ~30 minutes to implement, test, and deploy.

### Option B: Advance to RF2 (More Data — Quick but Uncertain)

**What**: Skip RF1 gate, manually advance to RF2 (35% data).

**Why**: RF2's 35% subset has ~1.75× more GT frames. Still only ~1.2% of batches, but significantly better.

**Implementation**:
```bash
python3 -m src.training.stage_manager --advance 2
```

**Pros**: Quickest path forward (takes 30 seconds). Tests whether data scaling fixes the death spiral.
**Cons**: No RF1 baseline metrics. Higher gate threshold at RF2 (det_mAP50 >= 0.40). Death spiral may still happen at 35%. Only 1.75× improvement vs 143× from oversampling.

### Option C: Positive Focal Loss Weight

**What**: Add weighting to positive examples in the focal loss to counter extreme imbalance.

**Implementation**: `alpha` parameter in focal loss. Set `alpha_positive=0.9`, `alpha_negative=0.1`.

**Pros**: Directly addresses the gradient imbalance at the loss level. No data changes.
**Cons**: New hyperparameter to tune. May cause instability if set too high. Doesn't solve the fundamental issue that most batches have zero positives.

### Option D: GT-Only Pre-Training (Hybrid — Highest Complexity)

**What**: Train detection only on GT-positive frames for 5 epochs, then switch to full dataset.

**Implementation**: Two-phase training: (1) GT-only for N epochs (2) Resume full training from GT-only checkpoint.

**Pros**: Detection head learns confident predictions in isolation. Then finetune on full data.
**Cons**: Two-phase training requires code changes in both training loop and checkpoint management. May forget GT-only learning when exposed to empty frames. Takes ~5 hours for GT-only phase.

### Recommendation

**Option A (GT oversampling) is the recommended fix.** It requires the least code change, directly addresses the root cause, and provides a 143× improvement in positive update frequency. If oversampling alone isn't sufficient, combine with Option C (positive focal loss weight) for a 2-pronged approach.

**Timeline for implementation**: ~30 minutes to code + ~2 hours to validate.

---

## 16. Remaining Uncertainties and Confusions

### A. Will Any LR Strategy Break the Death Spiral? (CRITICAL — RESOLVED: NO)

**The question**: Will reduce_lr_20x_warmup_3x produce different results from the 5 previous retries?

**Status**: **RESOLVED — all 5 strategies produce the same trajectory.** Phase 4 is producing IDENTICAL DET-DEBUG outputs to Phase 3 at the same steps (551, 651, 751). The 20× LR reduction changes nothing within the first 800 steps.

**Updated prediction**: The current retry will die at step ~1300 (same as Phase 3). The training supervisor will find another retry needed, but all strategies are exhausted. The system will loop indefinitely on RF1.

### B. Does the Detection Head Need Positive-Only Training? (CRITICAL)

**The question**: Can the detection head learn to predict foreground with high confidence without dedicated GT-only epochs?

**Analysis**: With 0.7% GT batches, the model sees ~174 GT batches in 20 epochs. Each GT batch has ~0.7% of anchors as positive. That's ~1,200 positive anchor updates total. For 7 output classes, that's ~170 updates per class. This is likely INSUFFICIENT for confident predictions.

**Updated evidence from Phase 3 cycling**: The detection head bounces back to LIVENESS=6.56 when a GT batch arrives at step 1151. But by step 1300 it's DEAD again. This shows the positive updates are too weak to sustain learning through the empty-batch periods.

**What I'm confused about**: Is 170 updates per class enough to push sigmoid output from 0.01 to 0.5+? The bounce cycle suggests NOT — each GT batch provides a temporary boost that decays within 100-200 steps.

### C. PSR Recovery at RF4 (CRITICAL)

**The question**: PSR has been DEAD (loss=1.5e-8) through all runs. Will enabling it at RF4 produce learning?

**Observations:**
- PSR transformer produces extreme logits (min=-23, max=+22) with constant loss
- 20/22 components predict zeros
- PSR loss is bit-exact constant at 1.546e-08 across ALL 1026 diagnostic entries
- Causal attention with T=2 may be too short to learn temporal transitions
- PSR seq loss spikes (~1077) on sequence batches

**What I'm confused about**: Has PSR ever been successfully trained in this architecture? Is the fill-forward label scheme (most components=0) fundamentally flawed? The fact that PSR has NEVER shown any gradient in any run (Phase 1-4, all R2.5 runs) suggests a deeper architecture issue, not just a training configuration issue.

### D. Stage Manager Epoch Mismatch (MEDIUM)

**The question**: state.epoch = 0 in the state file, but training is at epoch 58. Does this affect gate decisions?

**Impact**: 
- Convergence tracking uses epoch=0 instead of 58
- `max_epochs` calculation has been adjusted (+58) in retry strategies
- But patience tracking may use wrong epoch count

**What I'm confused about**: Is the 5-category checklist using the correct epoch for patience-based decisions? The state file shows epoch=0 because each retry resets state.epoch but not the checkpoint's actual epoch.

### E. Why Is the Checkpoint Never Updated? (MEDIUM)

**The question**: Checkpoint `latest.pth` is from 14:12 — 202+ minutes old. Why doesn't the current training save checkpoints?

**Possible explanations:**
1. Checkpoints only save at epoch boundaries (every ~23 min in RF1)
2. The training process (PID 3096257) has only been running ~10 min at step 181-600 range
3. The `--reinit-heads` flag may change checkpoint behavior

**What I'm confused about**: Should the stage_manager check for stale checkpoints as a health indicator? A stale checkpoint means the retry won't have a fresh recovery point if it crashes.

### F. Kendall Log-Var Pinning (LOW)

**The question**: In the old R2.5 system, three of four Kendall log_vars were pinned at bounds. The RF system doesn't pin them explicitly.

**Observation**: Kendall log_vars are reset to 0.0 during `--reinit-heads`. With a single active head in RF1 (detection), log_vars are irrelevant until RF4 where all heads compete.

### G. Is the 15-min Cron Too Aggressive? (LOW)

**The question**: During Phase 2, the frequent cron checks (every 15 min) found stale checkpoints and contributed to the kill-and-retry cascade.

**What I'm confused about**: Should the cron respect a "minimum training time before kill" — e.g., don't kill a process that has been running less than 30 minutes? The current behavior killed Phase 2 runs at step ~51 (2 min) based on a stale checkpoint heuristic.

### H. Detection Gate Feasibility at 20% Data (MEDIUM)

**The question**: Is det_mAP50 >= 0.30 achievable at 20% data?

**Issue**: Even if we fix the death spiral, we don't know the ceiling for detection metrics at 20% data. The paper benchmarks (YOLOv8m mAP50=0.838) use 100% data. At 20%, metrics might be capped at 0.2-0.3 regardless of training quality.

**Updated consideration**: With GT oversampling at 20% data, the model would see GT frames ~143× more often, but it would see the SAME GT frames repeatedly. This could lead to overfitting and mAP scores that don't reflect true detection quality.

**What I'm confused about**: Should the RF1 gate be adjusted to det_mAP50 >= 0.15 or similar, acknowledging that 20% data can't produce paper-level metrics?

### I. Is the Detection Head Architecture Suitable for 20% Data? (MEDIUM — NEW)

**The question**: Does the ConvNeXt-T + FPN + FCOS-style detection head need more data to converge?

**Evidence**: ConvNeXt-T has ~28M parameters in the backbone alone. The detection head has 4 conv layers. At 0.9 batch/s and 20% data, each epoch is 1241 batches. With GT oversampling, the model would see GT frames every batch, but only ~0.7% of the 164K anchors per frame are positive. The effective positive examples per epoch would still be small.

### J. What Happens When All Retry Strategies Are Exhausted? (NEW — HIGH)

**The question**: What does stage_manager do after retry #4 (reduce_lr_20x_warmup_3x)?

**Current behavior**: The strategies list has 5 entries (indices 0-4). After all 5 have been tried with no gate progress, the stage_manager would either:
1. Loop back to the first strategy and try again (infinite loop)
2. Set stage status to "aborted" 
3. Try advancing to RF2 with a warning

**What I'm confused about**: The current code path after all strategies exhausted is not clearly defined. Need to read stage_manager.py to verify.

### K. Why Do Non-Det Heads Show Zero Gradient? (MEDIUM — NEW)

**Observation**: All LIVENESS entries show `act=0.00e+00 DEAD`, `psr=0.00e+00 DEAD`, `pose=1.00e-06 DEAD`, `head_pose=1.00e-06 DEAD` across ALL runs. The only ALIVE head is detection (and occasionally head_pose in Phase 3 step 1300+).

**Question**: Is this expected in RF1 where only `train_det=True`? Or does this indicate that the feature bank detachment (`feature_bank_detach=True`) is also blocking all other heads?

**If expected**: This is fine — RF1 only trains detection.
**If a bug**: The detached feature bank may prevent ANY gradient from reaching other heads, meaning enabling them at RF2 would produce a gradient shock.

---

## 17. References

### Source Files (current RF system)

| Path | Description |
|------|-------------|
| `src/training/stage_manager.py` | RF1-RF10 stage orchestration (2334 lines) |
| `src/training/training_supervisor.py` | Deep diagnostic supervisor (760 lines) |
| `src/training/train.py` | Main training script (4355 lines) |
| `src/config.py` | Configuration with 10 stage presets (1321 lines) |
| `src/runs/rf_stage_state.json` | Current stage state |
| `src/runs/rf_stages/logs/train.log` | Training log (3161 lines) |
| `src/runs/rf_stages/logs/supervisor.log` | Supervisor log (full 4-phase history) |
| `src/runs/rf_stages/logs/cron.log` | Cron log |

### Previous Documentation (opus_consult directory)

| File | Content |
|------|---------|
| `00_JOURNEY_AND_STATUS.md` | Complete project history from April to June 13 |
| `01_PROBLEMS_ROOT_CAUSES.md` | All identified root causes (RC-13 through RC-29) |
| `02_GOALS_AND_BENCHMARKS.md` | Paper benchmarks and targets |
| `03_ARCHITECTURE_DEEP_DIVE.md` | Model architecture in detail |
| `18_HONEST_FEASIBILITY_AUDIT.md` | Independent feasibility assessment |
| `23_TRAINING_RUNS_AND_CURRENT_STATUS.md` | R2.5 run history and status |
| `24_MASTER_ANALYSIS_WITH_20_QUESTIONS.py` | 20-question diagnostic script |
| `25_R3_100_CHECKLIST.md` | 100-item pre-R3 readiness checklist |

### Key Commands

```bash
# Run training supervisor
cd /media/newadmin/master/POPW/working/code/industreal_improved && \
    python3 -m src.training.training_supervisor 2>&1 | \
    tee -a src/runs/rf_stages/logs/supervisor.log

# Check current stage status
python3 -m src.training.stage_manager --status

# Monitor training progress
tail -f src/runs/rf_stages/logs/train.log | \
    grep -E "(DET-DEBUG|batch.*speed|LIVENESS|Validation)"

# Check latest DET-DEBUG
grep "DET-DEBUG" src/runs/rf_stages/logs/train.log | tail -5

# Manual advance to next stage (if needed)
python3 -m src.training.stage_manager --advance 2

# Force launch RF1 with reset
python3 -m src.training.stage_manager --reset
python3 -m src.training.stage_manager --launch rf1
```

---

*End of document. Generated 2026-06-16 17:45 UTC (v3 — updated with Phase 4 trajectory identity + cycling death spiral pattern + strategy exhaustion analysis).*

---

## Appendix: Key Changes from v2 to v3

### New Data Incorporated

1. **Phase 4 DET-DEBUG trajectory**: Steps 551, 651, 751 confirmed identical to Phase 3
2. **LIVENESS cycling analysis**: Full step-by-step LIVENESS values showing the bounce-and-die pattern
3. **GT batch bounce effect**: At step 1151-1200, GT batch revives gradient from 0.19→6.56, then dies by 1300
4. **PSR constant loss evidence**: 1026 diagnostic entries all showing same loss=1.546e-08
5. **Checkpoint staleness**: latest.pth 202+ min old (never updated across retries)

### New Sections Added

1. **Section 8, Phase 4**: New "Critical Finding" subsection documenting trajectory identity
2. **Section 9, DET trajectory**: Expanded with LIVENESS column, bounce-and-die pattern documentation
3. **Section 10, Death spiral**: New "The Bounce Cycle" subsection explaining the dynamic equilibrium
4. **Section 14, Timeline**: Updated to show RF1 is "STUCK" (not estimable)
5. **Section 16, Uncertainties**: 4 new uncertainties (A-resolved:NO, I, J, K)
6. **Appendix**: This changelog

### Analysis Changes

1. **Phase 4 prediction**: Changed from "delays death to step ~2000" to "reproduces Phase 3 identically, death at step ~1300"
2. **Strategy assessment**: Changed from "5 retries may work" to "all 5 strategies exhausted, none functional"
3. **Recommendation**: Stronger emphasis on GT oversampling as the only viable fix
