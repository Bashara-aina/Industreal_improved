# 00 — Master Index: Complete File Map for Opus Consult [2026-07-01]

**This is the single-entry index for the entire consultation.** All 10 MD analysis files + all 6 Python source files + 1 verification script. Every Opus prompt in this directory is designed to be read as a standalone file, but this index explains how they connect.

**Repository root:** `/media/newadmin/master/POPW/working/code/industreal_improved/`
**Code root:** `code/industreal_improved/src/`
**Analysis root:** `code/industreal_improved/analyses/consult_2026_06_10/`

**Current HEAD:** `4369622` (main) — 18 fixes applied across 9 commits since file 75.

---

## Part 1: Analysis Files (10 MD Files)

These are the analysis documents. Read them in this order for the full story, or individually for specific topics.

```
00_MASTER_INDEX.md          ← YOU ARE HERE
77_ACTIVITY_HEAD_FINAL_VERIFICATION.md
78_DETECTION_HEAD_FINAL_VERIFICATION.md
79_PSR_HEAD_FINAL_VERIFICATION.md
80_HEAD_POSE_MULTI_TASK_FINAL_VERIFICATION.md
81_OPUS_MASTER_PROMPT_FINAL_ROUND.md
82_OPUS_FINAL_VERIFICATION_RESPONSE.md
83_CRITICAL_FIXES_SCHEDULER_WEIGHT_DECAY_METRICS.md
84_FULL_PIPELINE_VERIFICATION.md
85_RF4_RF10_GATE_CRITERIA.md
86_OPUS_FINAL_CONFIRMATION.md
```

### File-by-File Summary

| File | What It Covers | Key Reader |
|------|---------------|------------|
| **77** | Activity head: verb-grouping (Route A → hybrid), balanced sampler, simple MLP, CE+CB loss, eval metrics in group space, epoch-2 go/no-go | Activity specialist |
| **78** | Detection head: 4 collapse mechanisms (OHEM, asymmetric gamma, GT fraction, empty-frame), detach_reg_fpn corrected (ALL presets False), mAP trajectory, epoch-5 go/no-go | Detection specialist |
| **79** | PSR head: binary focal loss (corrected — not BCE), per-component weighting, sequence mode, per-frame caveat, epoch-5 go/no-go | PSR specialist |
| **80** | Head pose + multi-task: Kendall weighting, gradient blend, smooth caps, log-var device confirmed SAFE (not a bug), stage-by-stage orchestration | Multi-task specialist |
| **81** | Master prompt with 6 verification questions for Opus (round 4) | Opus (first read) |
| **82** | Opus's answers to Q1-Q6: sampling distortion (Q1, DIAG), GIoU floor dead code (Q2), PSR per-frame (Q3), LR schedule (Q4), segment-eval bug fix (Q5), 3 remaining items (Q6) | Opus (confirms answers) |
| **83** | 3 CRITICAL fixes from 20-agent audit: OneCycleLR scheduler (`steps_per_epoch=1`), bias/norm WD=0, GT fraction + PSR accuracy metrics | Anyone launching training |
| **84** | Complete end-to-end pipeline: every config value, model arch, loss path, eval path for ALL 4 heads with file:line references | Full-stack reviewer |
| **85** | Stage-by-stage gate criteria for RF4-RF10: what to check at each epoch, pass/fail thresholds, crash recovery procedures | Training operator |
| **86** | Master confirmation with 5 yes/no sign-off questions. Verification map (30+ claims → code), final confidence assessment | Final sign-off |

---

## Part 2: Python Source Files (6 Files + 1 Script)

These are the actual code files referenced throughout the analysis documents. Each file:line reference in the MD files points here.

```
code/industreal_improved/
├── src/
│   ├── config.py                          # ~1967 lines — ALL hyperparameters
│   ├── models/
│   │   └── model.py                       # ~2360 lines — ALL architectures
│   ├── data/
│   │   └── industreal_dataset.py          # ~1720 lines — dataset + sampler
│   ├── training/
│   │   ├── train.py                       # ~5320 lines — training loop
│   │   └── losses.py                      # ~1900 lines — ALL loss functions
│   └── evaluation/
│       └── evaluate.py                    # ~4500 lines — ALL metrics
└── scripts/
    └── verify_act_grouping.py             # 53 lines — pre-flight grouping inspector
```

### File-by-File Summary

| File | Lines | Role | Key Content |
|------|-------|------|-------------|
| `config.py` | ~1967 | Central configuration hub | All hyperparameters, activity grouping, sampler mode, loss caps, staged training presets (RF1-RF10), Kendall bounds, gradient blend ratio, DET_CLASS_ALPHAS per-class alphas, PSR warmup, WD/clip values |
| `models/model.py` | ~2360 | Neural network architecture | ActivityHead (simple MLP or TCN+ViT), HeadPoseHead (9-DoF), PSRHead (causal transformer + per-component heads), DetectionHead (RetinaNet-style), PoseFiLM, HeadPoseFiLM, POPWMultiTaskModel |
| `industreal_dataset.py` | ~1720 | Data loading and sampling | _parse_ar_labels (action recognition), _parse_pose (head pose 9-DoF), _parse_psr, get_sampler (3-layer reweighting), collate_fn, collate_fn_sequences, RAM cache |
| `train.py` | ~5320 | Training orchestration | Optimizer construction (AdamW with bias/norm WD=0), OneCycleLR scheduler (steps_per_epoch=1), gradient clipping (5.0), checkpoint save/load, OOM recovery, apply_preset for stage transitions, all logging |
| `losses.py` | ~1900 | All loss functions | FocalLoss (OHEM + asymmetric gamma + GIoU), ClassBalancedFocalLoss, MultiTaskLoss (Kendall with bounds/smooth caps), binary focal loss (PSR), head_pose_split_loss (9-DoF MSE), LDAMLoss |
| `evaluate.py` | ~4500 | Evaluation metrics | compute_activity_metrics (macro-F1 with present_labels, clip-level accuracy, segment eval with Q5 fix), detection mAP (COCO-style), compute_psr_metrics (per-component F1, comp-acc new), compute_head_pose_metrics (angular MAE, position MAE), diversity monitor |
| `verify_act_grouping.py` | 53 | Pre-flight inspection | Prints current grouping mode, output count, group names, raw-id mapping, per-group TRAIN frame counts |

---

## Part 3: Key Line Number Ranges (for Quick Reference)

### `config.py`

| Lines | Content |
|-------|---------|
| 38-45 | Training flags (TRAIN_DET, TRAIN_HEAD_POSE, TRAIN_ACT, TRAIN_PSR) |
| 54-66 | Detection LR, DET_LR_MULTIPLIER |
| 68-95 | Kendall flags, KENDALL_HP_PREC_CAP, fixed weights |
| 97-99 | Hand-FiLM conditioning (NOT a task) |
| 103-145 | Backbone (ConvNeXt-Tiny), gradient checkpointing |
| 178-204 | DET_CLASS_NAMES (24 ASD classes) |
| 270-330 | Activity class grouping (ACT_CLASS_GROUPING, _build_act_grouping) |
| 393-412 | NUM_KEYPOINTS (17 COCO pseudo-keypoints — NO real GT) |
| 408-409 | NUM_HEAD_POSE_DOF=9 (real GT from pose.csv) |
| 500-560 | BATCH_SIZE=2, GRAD_ACCUM_STEPS=8, EPOCHS=100, BASE_LR=5e-4 |
| **512/528** | **WEIGHT_DECAY=1e-3** (was 5e-2) |
| **518/537** | **GRAD_CLIP_NORM=5.0** (was 1.0) |
| 568-586 | DET_EVAL score threshold, NMS, max per image |
| 605-676 | FOCAL_ALPHA/GAMMA, DET_CLASS_ALPHAS, OHEM, asymmetric gamma |
| 680-703 | Activity sampler mode, CB loss params |
| 704-800 | PSR temporal smooth, task-aware sampling, DET_GT_FRAME_FRACTION |
| 720-735 | Staged training flags |
| 800-860 | Activity head config: simple MLP, dropout, grad blend, LR multiplier |
| 854-900 | Kendall bounds, PSR warmup, REINIT_REG_WARMUP |
| 915-990 | PSR config: focal alpha/gamma, PSR_WARMUP_STEPS/EPOCHS, sequence mode |
| 1920-1967 | Module-level grouping attrs: NUM_ACT_OUTPUTS, ACT_ID_TO_GROUP |

### `models/model.py`

| Lines | Content |
|-------|---------|
| 540-570 | DetectionHead (RetinaNet cls/reg subnets) |
| 571-620 | PoseHead (heatmaps → soft-argmax → keypoints) |
| 622-716 | PoseFiLMModule (keypoint → FiLM on C5) |
| 718-792 | HeadPoseFiLMModule (9-DoF → FiLM on c5_mod) |
| 1262-1478 | ActivityHead (simple MLP at 1319-1422, TCN+ViT conditional) |
| 1482-1534 | HeadPoseHead (C4 GAP + C5 GAP → MLP → 9-DoF) |
| 1535-1743 | PSRHead (causal transformer + 11 per-component heads) |
| 1745-2210 | POPWMultiTaskModel (full architecture assembly) |
| 1804-1810 | DetectionHead construction |
| 1854-1867 | ActivityHead construction (num_classes=NUM_ACT_OUTPUTS) |
| 2150-2210 | Forward pass with gradient path (c5_mod_blend, bank_output=None) |

### `training/losses.py`

| Lines | Content |
|-------|---------|
| 200-420 | FocalLoss.forward (OHEM, asymmetric gamma, GIoU, empty-frame) |
| 260-270 | num_pos clamp (div-by-zero protection) |
| 310-335 | OHEM logic (RATIO=2.0, MIN_NEG=32) |
| 344-349 | Asymmetric gamma (pos=0, neg=1.5) |
| 679-790 | ClassBalancedFocalLoss (unused — USE_CB_FOCAL_ACT=False) |
| 829-935 | binary_focal_loss (PSR: per-component alpha + comp_weights) |
| 938-965 | head_pose_split_loss (9-DoF MSE with unit norm regularization) |
| 972-1845 | MultiTaskLoss (Kendall weighting, bounds, smooth caps, NaN guards) |
| 1044-1068 | Activity loss selection (CE+label_smooth, not CB-Focal) |
| 1125-1135 | CB weight injection into CE loss |
| 1170-1187 | Activity ramp with stage-local epoch counter |
| 1206-1210 | Log-var device management (SAFE — dead code, params on GPU) |
| **1220-1252** | GIoU warmup, NEG_SLOPE=0.01 (dead code, loss_det always ≥0) |
| 1272-1296 | NaN guard before Kendall (triple layer) |
| 1309-1314 | _smooth_cap function (never-zero gradient) |
| 1386-1390 | Activity ramp: (epoch+1)/5 → [0.2, 0.4, 0.6, 0.8, 1.0] |
| 1458-1464 | PSR focal loss path (binary_focal_loss, not BCE) |
| 1676-1682 | Kendall log_var bounds (per-task overrides) |
| 1689-1690 | KENDALL_HP_PREC_CAP (lv_hp >= lv_det) |
| 1766-1777 | Head pose included in Kendall (even when body pose loss=0) |
| 1796-1800 | PSR step-based warmup (500 steps, 2.0→1.0) |

### `training/train.py`

| Lines | Content |
|-------|---------|
| 1126 | Sequence batch detection (`is_seq_batch`) |
| 1291-1294 | Gradient clipping (model + criterion, max_norm=5.0) |
| 1450-1475 | DET-HEALTH logging (cls_mean, std, near_zero, det_gt_fraction) |
| 1773-1776 | Gradient clipping (non-seq path, model + criterion, max_norm=5.0) |
| 1917-1924 | GRAD-NORM logging (backbone, det, hp, act, psr) |
| 3341-3363 | Sequence dataloader construction |
| **3365-3386** | Class_counts handling (NO double-remap, pad/truncate guard) |
| 3461-3470 | Criterion init (to(device), num_classes_act=NUM_ACT_OUTPUTS) |
| 3487-3521 | Parameter group construction (backbone, det, head, bias, activity, psr) |
| **3579-3593** | Optimizer: AdamW with per-group weight_decay (bias/norm WD=0) |
| **3607-3644** | Scheduler: SequentialLR(LinearLR(warmup, iters=2) + OneCycleLR(steps_per_epoch=1)) |
| 3659 | GradScaler (enabled=MIXED_PRECISION=False → pass-through) |
| 4245-4279 | OOM recovery (halve batch, rebuild dataloader, retry epoch) |
| 4290 | scheduler.step() — once per epoch (matches steps_per_epoch=1) |

### `evaluate/evaluate.py`

| Lines | Content |
|-------|---------|
| 730-847 | _compute_clip_level_accuracy (16 uniform frames, majority vote) |
| 851-884 | compute_activity_segment_metrics (Q5 fix: label remapped to group space) |
| 897-941 | compute_activity_metrics (macro-F1 with present_labels, clip accuracy) |
| 956-958 | present_labels filter for macro-F1 |
| 1005-1008 | Clip-level accuracy computation |
| 1607-1666 | Detection mAP (COCO-style 101-point interpolation) |
| 1840-1875 | Head pose metrics (angular MAE, position MAE — **position mm unverified**) |
| 2434-2550 | PSR metrics (per-component F1, overall F1, POS, edit score) |
| 3302-3310 | Detection eval: score_thresh, max_per_image, NMS |
| **3440-3495** | Diversity monitor (pred_distinct, entropy, collapse warning) |
| **3755-3766** | PSR component binary accuracy (NEW) |

### `data/industreal_dataset.py`

| Lines | Content |
|-------|---------|
| 447-493 | _parse_ar_labels (action_id 0 is real, -1 sentinel unlabeled) |
| 570-620 | _parse_pose (9-DoF from pose.csv, HEAD_POSE_POS_SCALE applied) |
| 789-805 | Label remap to group space (activity_ids, class_counts with minlength) |
| 897-908 | Per-frame __getitem__ label remap |
| 1021-1023 | Sequence path label remap |
| 1240-1265 | _build_seq_sample_index (consecutive T=8 windows) |
| **1412-1511** | get_sampler (3-layer: balanced → task-aware → DET_GT_FRAME_FRACTION) |
| **1519-1538** | Per-class sampling mass diagnostic (Q1 fix) |
| 1552-1659 | collate_fn |
| 1662-1747 | collate_fn_sequences |

---

## Part 4: Fix History (18 Fixes Across 9 Commits)

| # | Commit | What | Why |
|---|--------|------|-----|
| 1 | `c27476f` | Route A verb-grouping wiring | Files 75-76 implementation |
| 2 | `cb18506` | Double-remap: class_counts no longer re-remapped | Corrupted loss weights in group space |
| 3 | `cb18506` | WEIGHT_DECAY 5e-2 → 1e-3 | 50-500x standard; with clip=1.0, decay dominated gradient |
| 4 | `cb18506` | GRAD_CLIP_NORM 1.0 → 5.0 | 5-head combined norm ~7.3; 1.0 was damping 86% |
| 5 | `cb18506` | DET_GT_FRAME_FRACTION 0.90 → 0.40 | Was starving activity (0.14 frames/class/batch) |
| 6 | `cb18506` | TCN+ViT conditionally allocated | Saved 7.66M dead params (~92 MB) |
| 7 | `cb18506` | GIoU NEG_SLOPE 0.0 → 0.01 | Was killing regression gradient (now known dead code — loss_det ≥ 0 always) |
| 8 | `cb18506` | PSR warmup: STEPS=500, EPOCHS=3 | PSR had no warmup when STAGED_TRAINING=False |
| 9 | `f95a1aa` | File 78 §1.3 correction: DETACH_REG_FPN | All RF presets override to False |
| 10 | `f95a1aa` | File 79 §2.1 correction: PSR focal | PSR uses focal loss (gamma=0.5), not BCE |
| 11 | `f95a1aa` | File 80 §1.6 correction: log-var device | SAFE — criterion.to(device) before optimizer |
| 12 | `f95a1aa` | Config comment fixes (USE_AMP, LDAM, LR_MULTI) | 3 stale comments updated |
| 13 | `832259f` | Pose confusion clarified | Head pose vs body keypoints vs hand-film |
| 14 | `b6d4cce` | Segment-label remap (Q5) | Makes act_seg_top1 valid under grouping |
| 15 | `b6d4cce` | Per-class sampling mass log (Q1) | Monitors DET_GT distortion |
| 16 | **`2e69b1e`** | **OneCycleLR scheduler: steps_per_epoch=1** | **CRITICAL — was stuck in warmup forever** |
| 17 | **`2e69b1e`** | **Bias/norm weight decay = 0.0** | Standard AdamW practice |
| 18 | **`2e69b1e`** | **GT fraction + PSR accuracy metrics** | Now logged in standard output |

---

## Part 5: Quick Opus Navigation

### For Activity Review → Read
- `77` (all sections)
- `84 §2` (pipeline)
- `85 §RF4` (gate criteria)

### For Detection Review → Read
- `78` (all sections, especially §1 with corrected DETACH_REG_FPN)
- `84 §1` (pipeline)
- `83 §Fix 3` (new GT fraction metric)

### For PSR Review → Read
- `79` (all sections, especially §2.1 with corrected focal loss)
- `84 §3` (pipeline)
- `83 §Fix 3` (new PSR comp-acc metric)

### For Multi-Task / Kendall / Head Pose → Read
- `80` (all sections, especially §1.6 corrected)
- `84 §4-5` (pipeline)
- `83 §Fix 1-2` (scheduler + weight decay)

### For Final Sign-Off → Read
- `86` (5 yes/no questions, verification map)
- `85` (gate criteria)
- `83` (critical fixes)

---

## Part 6: Running the Code

```bash
# 1. Verify activity grouping (pre-flight, ~1 min)
python scripts/verify_act_grouping.py

# 2. Run 50-step probe (diagnostic, ~5 min)
python src/training/train.py --preset stage_rf4 --reinit-heads --max-epochs 1 --max-steps 50
# Check: [DET-HEALTH] cls_mean, det_gt_fraction
# Check: [GRAD-NORM] all 4 heads > 0
# Check: [Kendall log_sigma] lv values in [-1, +1]

# 3. Launch RF4 (production, ~24 hours)
python src/training/train.py --preset stage_rf4 --reinit-heads

# 4. Monitor gates at epoch 2, 5, 10, 20, 40...
# epoch 2: check [DIVERSITY] pred_distinct >= 10
# epoch 5: check mAP@0.5 >= 0.005
# epoch 20: check all 4 metrics improving

# 5. Continue through RF5-RF10 (no --reinit-heads needed)
python src/training/train.py --preset stage_rf5 --resume
python src/training/train.py --preset stage_rf6 --resume
# ... RF7-RF10 same pattern
```
