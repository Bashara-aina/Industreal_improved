# 81 — Opus Master Prompt: Final Verification Round [2026-07-01]

**Goal:** Confirm the training pipeline is ready for a 100-epoch run with meaningful metrics across all 4 heads. The previous rounds diagnosed and fixed specific bugs (gradient starvation, class collapse, detection death spiral). This round is the final verification — you need to confirm that the fixes are complete, the config is coherent, and no hidden issues remain.

---

## How to Read This Directory

This directory (`analyses/consult_2026_06_10/`) contains the complete audit trail. For this final round, focus on:

### 4 New Analysis Files (this round)

| File | Focus | Key Sections for You |
|------|-------|----------------------|
| **77_ACTIVITY_HEAD_FINAL_VERIFICATION.md** | Verb-grouping wiring, sampler, simple MLP | §1 Grouping, §2 Sampler, §4 Simple MLP, §7 Go/No-go |
| **78_DETECTION_HEAD_FINAL_VERIFICATION.md** | OHEM, GT fraction, regression warmup | §1 Four collapse mechanisms, §6 Expected mAP trajectory |
| **79_PSR_HEAD_FINAL_VERIFICATION.md** | Causal transformer, per-component balance | §2 Loss config (focal is active, gamma=0.5), §5 Sequence mode, §7 Go/No-go |
| **80_HEAD_POSE_MULTI_TASK_FINAL_VERIFICATION.md** | Kendall weighting, gradient blend, staged training | §1 Kendall bounds, §6 Log-var device (SAFE — corrected), §9 Risk table, §10 Go/No-go |
| **81_OPUS_MASTER_PROMPT_FINAL_ROUND.md** | This file — master index and summary | Everything below |

### Source Files Referenced (all under `code/industreal_improved/src/`)

| File | Lines of Interest | Role |
|------|-------------------|------|
| `config.py` | Full file (~1950 lines) | All hyperparameters, activity grouping, sampler mode, loss caps, staged training presets |
| `models/model.py` | ActivityHead: 1262-1478, HeadPoseHead: 1482-1531, PSRHead: 1535-1742, DetectionHead: 1804-1810, Full model: 1745-2210 | Architecture definitions |
| `training/losses.py` | MultiTaskLoss: 972-1800, FocalLoss: ~500-690, CB-Focal: ~690-790, GIoU floor: 1247-1252 | All 4 loss functions with Kendall uncertainty weighting |
| `training/train.py` | Training loop: dual paths (seq + non-seq), optimizer build: ~3592, class_counts: 3365-3386 | Training orchestration |
| `data/industreal_dataset.py` | Sampler: 1412-1520, Label remap: 789-805, 897-908, 1020-1028 | Data loading and activity label mapping |
| `evaluation/evaluate.py` | Activity metrics: 886-1018, Diversity monitor: 3440-3495, Segment eval: 851-884 (has label-mismatch bug — see below) | Evaluation |

---

## Executive Summary of All Previous Work

### What Was Diagnosed (Verified in Code)

1. **Activity collapse** (files 70-76): 74-class per-frame AR is near-zero bounded by data constraints (48/74 classes <10 frames). Solved via **verb-grouping** (Route A): collapse to ~47 hybrid groups (fine-grained for common classes, verb-grouped for tail). Head architecture simplified from 8.2M-param TCN+ViT to 150K-param simple MLP.

2. **Detection death spiral** (files 56, 62, 67-68): Cumulative negative gradient from 173K:1 positive-negative anchor imbalance. Solved via **4 independent mechanisms**: OHEM (2:1 ratio), asymmetric gamma (pos=0.0, neg=1.5), DET_GT_FRAME_FRACTION (abs% of GT-bearing batch), and empty-frame background loss.

3. **PSR gradient starvation** (files 64, 67, 69): Causal transformer was processing T=1 shuffled frames. Solved via **sequence mode** (T=8, every 2 batches) and per-component prevalence weighting.

4. **Multi-task gradient imbalance** (files 57, 69): Kendall weighting with per-task bounds, head-pose precision cap, activity gradient blend, PSR/reg detach from FPN.

### What Was Fixed This Round (7 code changes, commit `cb18506`)

| # | Fix | File | Why |
|---|-----|------|-----|
| 1 | **Double-remap** — class_counts no longer re-remapped through `remap_activity_label` | `train.py:3365-3386` | Dataset already produces group-space counts; re-remapping corrupted loss weights |
| 2 | **Weight decay** 5e-2 → 1e-3 | `config.py:512` | 50-500x standard; with clip=1.0, decay dominated gradient 5:1 for params with norm >4 |
| 3 | **Grad clip** 1.0 → 5.0 | `config.py:518` | 5-head combined norm ~7.3; 1.0 was damping backbone 86% |
| 4 | **GT fraction** default 0.90 → 0.40 | `config.py:801` | 0.90 starved activity to ~0.14 frames/class/batch with 47 groups |
| 5 | **TCN+ViT** conditionally allocated | `model.py:1319-1364` | 7.66M dead params when ACTIVITY_HEAD_SIMPLE=True (~92 MB GPU waste) |
| 6 | **GIoU floor** NEG_SLOPE 0.0 → 0.01 | `losses.py:1247` | Zero floor killed regression gradient for non-overlapping boxes |
| 7 | **PSR warmup**: STEPS=500, EPOCHS=3 | `config.py` | PSR had no warmup when STAGED_TRAINING=False |

### Corrections to Previous MD Files (commit `f95a1aa`)

- **File 78 §1.3 Fix 2**: `DETACH_REG_FPN=True` corrected — ALL RF presets override to False. The regression warmup is the gradient shock guard, not the detach.
- **File 79 §2.1**: "PSR uses plain BCE" corrected — focal loss IS active (PSR_FOCAL_GAMMA=0.5). Both focal alpha and comp_weights are simultaneously active.
- **File 80 §1.6**: Log-var device "bug" retracted — `criterion.to(device)` at `train.py:3469` moves all params to GPU before optimizer construction. The `.data` guard never fires. No bug.

---

## Current Configuration Summary

### Activity Head
```
ACT_CLASS_GROUPING: 'hybrid'  (~47 outputs: standalone for ≥100-frame, verb-group for tail)
ACT_SAMPLER_MODE: 'balanced'  (equal mass per class with ≥ACT_SAMPLER_COUNT_FLOOR=15)
ACTIVITY_HEAD_SIMPLE: True    (150K MLP, no TCN/ViT)
ACTIVITY_GRAD_BLEND_RATIO: 1.0 (full gradient into backbone for activity)
ACTIVITY_LOSS_WEIGHT: 1.0     (no fixed multiplier; Kendall handles weighting)
ACTIVITY_LOSS_CAP: 80.0       (safety net — CE init ≈4.3, never binds)
ACTIVITY_HEAD_GRAD_CLIP: 1.0  (per-head safety ceiling)
USE_CB_FOCAL_ACT: False       (plain CE + label_smoothing 0.1 + CB weights)
```

### Detection Head
```
DET_OHEM_ENABLED: True, RATIO=2.0, MIN_NEG=32
DET_ASYMMETRIC_GAMMA: True (pos=0.0, neg=1.5)
DET_GT_FRAME_FRACTION: default 0.40 (RF1 preset overrides to 0.90)
DET_EMPTY_SAMPLE: 2048, BG_SCALE=0.05
DETACH_REG_FPN: global default True but ALL RF presets override to False
REINIT_REG_WARMUP_STEPS: 1000, INIT_MULT=0.01
GIOU_WEIGHT: 2.0, NEG_SLOPE: 0.01
```

### PSR Head
```
USE_PSR_SEQUENCE_MODE: True, LENGTH=8, EVERY=2
PSR_FOCAL_GAMMA: 0.5 (focal NOT plain BCE)
PSR_WEIGHT: 10.0 (before Kendall)
PSR_WARMUP_STEPS: 500 (step-based, decays from 2.0→1.0)
PSR_WARMUP_EPOCHS: 3 (epoch-based for staged path)
DETACH_PSR_FPN: True (gradients don't corrupt FPN)
```

### Multi-Task (Kendall)
```
KENDALL_HP_PREC_CAP: True  (head-pose prec ≤ detection prec)
KENDALL_LOG_VAR_MIN_ACT: -0.5  (activity prec max 1.65×)
KENDALL_LOG_VAR_MAX_PSR: 0.0   (PSR prec min 1.0×)
KENDALL_LOG_VAR_MAX_POSE: 3.0  (pose prec min 0.05×)
STAGED_TRAINING: False  (all heads active from epoch 0)
WEIGHT_DECAY: 1e-3  (was 5e-2)
GRAD_CLIP_NORM: 5.0  (was 1.0)
```

---

## Questions for Opus to Verify

### 1. Sampling Distribution Distortion (highest priority)
With `DET_GT_FRAME_FRACTION=0.40`, 40% of batch mass is forced onto detection GT-bearing frames. Since only ~0.7% of all frames carry detection GT, this means heavy oversampling of the tiny subset of frames that have BOTH activity labels and detection boxes. Does this distort the activity label distribution? The activity-balanced weights are preserved *within* the GT and non-GT pools, but the relative sampling of a given action class depends on how many of its frames also have detection boxes. A class that appears predominantly on frames without detection boxes would be relegated to the 60% non-GT pool, while a class that co-occurs with detection boxes gets sampled from both pools. Opus should confirm this is acceptable or suggest logging effective per-class sampling rates.

### 2. GIoU Range (confirmatory check)
The `generalized_box_iou_loss` from torchvision returns `1 - GIoU` which is always in `[0, 2]`. Since GIoU loss is non-negative, `loss_det = cls_loss + 2.0 * reg_loss` can never be negative under normal conditions. This means the NEG_SLOPE=0.01 floor at losses.py:1247 is pure defensive coding — it never activates during healthy training. Opus should confirm this understanding is correct and that no other mechanism can produce negative loss_det.

### 3. PSR Double-Weighting
PSR has TWO active weighting mechanisms: focal alpha (from `set_psr_class_counts`, clamped max=1.0 in `binary_focal_loss`) and `_psr_comp_weights` (uniform per-component scaling). Verify these don't over-emphasize rare components to the point of instability. The sensitivity penalty `-log(mean(per_component_std))` at losses.py:1471-1478 provides a backstop against collapsed logits.

### 4. LR Schedule vs Detection Convergence
With `WARMUP_EPOCHS=2`, `BASE_LR=5e-4`, and 100 total epochs, OneCycleLR peaks at epoch 2 then cosine-decays. Detection has a hardcoded 250-step warmup (50 zero-grad + 200 linear ramp) which takes ~1.1 epochs at batch=2, accum=8. This means detection is at full strength for only ~0.9 epochs before LR begins decaying. Is this sufficient for detection to escape the all-background equilibrium? Consider raising WARMUP_EPOCHS to 5.

### 5. Segment Eval Protocol Bug
`compute_activity_segment_metrics` at evaluate.py:851-884 compares model predictions (in group space, from NUM_ACT_OUTPUTS channels) against raw annotation IDs (0-75 from CSV). This label mismatch makes `act_seg_top1` and `act_seg_top5` unreliable when `ACT_CLASS_GROUPING='hybrid'`. Opus should confirm the fix: apply `remap_activity_label()` to the `aid` returned by `_parse_ar_segments`.

### 6. Any Remaining Issues?
Is there any mechanism we haven't considered that could prevent a specific head from converging? The most plausible failure path is the sampling distortion from #1 above. Is there anything else?

---

## Go/No-Go Criteria

**If Opus confirms all 6 questions above are addressed** — proceed with a 50-step training probe (`SIMPLIFY_LOSS=True`, all 4 heads enabled) and check these 6 numbers at step 50:

| Signal | Pass | Fail |
|--------|------|------|
| `det_cls_mean` | -3.0 to -1.0 | < -5.0 (collapse) |
| `det_num_pos` per batch | ≥ 2 | 0 (no positives) |
| `act_pred_distinct` | ≥ 10 groups | < 5 (collapse) |
| `psr_loss` (raw BCE) | 0.2-1.5 | > 5.0 or NaN |
| `head_pose_mae` (angular) | < 40° | > 60° |
| `log_var_det.device` | cuda:0 | cpu (would mean device bug) |

**If all 6 pass at step 50** — launch the 100-epoch run. Expected epoch-100 targets:
- Detection mAP50: 0.50–0.65
- Activity clip-acc (grouped): 0.40–0.60, macro-F1: 0.30–0.50
- PSR binary accuracy: 0.75–0.85
- Head pose forward-gaze MAE: 8.71° (or better)

**Overall confidence: 78%** (22% uncertainty: 10% sampling distortion, 8% detection plateau, 4% systems failure)
