# 84 — Complete Pipeline Verification: All 4 Heads End-to-End [2026-07-01]

**Goal:** Trace the complete training pipeline for each head from input to metric, verifying every component is correctly wired. Opus must confirm no broken links.

**Source files (all under `src/`):**
- `config.py` — ~1967 lines, all hyperparameters and feature flags
- `data/industreal_dataset.py` — ~1720 lines, dataset loading, label remap, sampler
- `models/model.py` — ~2360 lines, all 5 head architectures
- `training/losses.py` — ~1900 lines, all loss functions + MultiTaskLoss with Kendall
- `training/train.py` — ~5320 lines, training loop, optimizer, scheduler, checkpoint
- `evaluation/evaluate.py` — ~4500 lines, all evaluation metrics

---

## 1. DETECTION PIPELINE (24 ASD classes, COCO-style mAP50)

### Config Path (`config.py`)
| Setting | Line | Value |
|---------|------|-------|
| `NUM_DET_CLASSES` | 188 | 24 (background + 22 states + error) |
| `ANCHOR_SIZES` | 448 | (96, 160, 256, 384, 512) |
| `DET_POS_IOU_THRESH` | 450 | 0.4 |
| `DET_POS_IOU_TOP_K` | 455 | 9 |
| `FOCAL_ALPHA/FOCAL_GAMMA` | 625 | 0.25 / 2.0 |
| `DET_CLASS_ALPHAS` | 633 | Per-class overrides for stuck classes |
| `DET_OHEM_ENABLED/RATIO/MIN_NEG` | 677-681 | True / 2.0 / 32 |
| `DET_ASYMMETRIC_GAMMA` | 691 | True (pos=0.0, neg=1.5) |
| `DET_GT_FRAME_FRACTION` | 828 | 0.40 (0.90 for det-only presets) |
| `DET_EMPTY_SAMPLE/BG_SCALE` | 791-792 | 2048 / 0.05 |
| `DETACH_REG_FPN` | 913 | True (global) → **False in ALL RF presets** |
| `REINIT_REG_WARMUP_STEPS` | 905 | 1000 (1%→100% linear ramp) |
| `DET_EVAL_SCORE_THRESH` | 606 | 0.001 (YOLOv8 comparability) |
| `GIOU_WEIGHT` | 669 | 2.0 |
| `GRAD_CLIP_NORM` | 537 | **5.0** (was 1.0) |
| `WEIGHT_DECAY` | 528 | **1e-3** (was 5e-2) |

### Model Path (`models/model.py:1807-1810`)
```
DetectionHead(in_channels=256, num_classes=24, detach_reg_fpn=False in RF presets)
  ├── cls_subnet (4 conv layers) → cls_score → cls_preds [B, 24×9, H', W']
  └── reg_subnet (4 conv layers) → reg_pred → reg_preds [B, 4×9, H', W']
```

### Loss Path (`training/losses.py:260-420, 1220-1252`)
```
losses.py ~270:  num_pos = max(pos_mask.sum().item(), 1)  ← protects div-by-zero
losses.py ~310:  OHEM keeps hardest negatives at RATIO=2.0, MIN_NEG=32
losses.py ~344:  asymmetric gamma: pos=0.0, neg=1.5
losses.py ~361:  focal loss = alpha_t * (1-p_t)^gamma * CE
losses.py ~391:  GIoU loss = generalized_box_iou_loss (1-GIoU ∈ [0,2])
losses.py ~1235: loss_det = cls_loss + 2.0 * reg_loss  ← always ≥ 0 (NEG_SLOPE=0.01 dead code)
losses.py ~1247: NEG_SLOPE=0.01 floor — never fires (loss_det ≥ 0 always)
```

### Eval Path (`evaluation/evaluate.py:~1607-1666, 3302-3310`)
```
COCO-style mAP@0.5 with 101-point PR interpolation
DET_EVAL_SCORE_THRESH=0.001, MAX_PER_IMAGE=300, NMS_IOU=0.5
```

### 4 Collapse Mechanisms (ALL Active)
| Mechanism | Config | Why |
|-----------|--------|-----|
| OHEM | ratio=2.0, min_neg=32 | Breaks cumulative negative gradient from 173K:1 imbalance |
| Asymmetric gamma | pos=0.0, neg=1.5 | Positives get full gradient, negatives moderated |
| DET_GT_FRAME_FRACTION | 0.40 (0.90 det-only) | Guarantees GT-bearing batch fraction |
| Empty-frame bg loss | 2048 samples, scale=0.05 | Keeps head alive between GT batches |

### Opus Verification Checklist
- [ ] Run `--preset stage_rf4` and confirm `DETACH_REG_FPN=False` (regression gradient reaches FPN)
- [ ] At step 500: `[DET-HEALTH] det_gt_fraction: ~X/4=~0.35-0.45` matches DET_GT_FRAME_FRACTION=0.40
- [ ] At step 500: `cls_preds mean` should be between -3.0 and -1.0 (not -16 collapse)
- [ ] At epoch 2: `mAP@0.5` should be ≥ 0.005 (not 0.0)

---

## 2. ACTIVITY PIPELINE (47 hybrid groups, CE + CB weights)

### Config Path (`config.py`)
| Setting | Line | Value |
|---------|------|-------|
| `ACT_CLASS_GROUPING` | 303 | 'hybrid' (~47 groups) |
| `ACT_HYBRID_THRESHOLD` | 304 | 100 frames |
| `NUM_ACT_OUTPUTS` | 1963 | ~47 (derived from grouping) |
| `ACT_SAMPLER_MODE` | 714 | 'balanced' |
| `ACT_SAMPLER_COUNT_FLOOR` | 715 | 15.0 |
| `ACTIVITY_HEAD_SIMPLE` | 861 | True (150K MLP, no TCN/ViT) |
| `ACTIVITY_HEAD_SIMPLE_HIDDEN` | 862 | 256 |
| `ACTIVITY_HEAD_DROPOUT` | 828 | 0.3 |
| `ACTIVITY_GRAD_BLEND_RATIO` | 882 | 1.00 |
| `ACTIVITY_LOSS_WEIGHT` | 839 | 0.8 |
| `ACTIVITY_LOSS_CAP` | 752 | 80.0 (safety net, CE init ~4.3) |
| `ACT_RAMP_EPOCHS` | 751 | 5 |
| `USE_CB_FOCAL_ACT` | 719 | False (plain CE + CB weights) |
| `CB_LABEL_SMOOTHING` | 719 | 0.1 |
| `CB_BETA` | 705 | 0.99 (for CB weight computation) |

### Grouping Path (`config.py:344-403`)
```python
_build_act_grouping('hybrid'):
  - classes with ≥100 frames: standalone identity (fine-grained)
  - classes with <100 frames: verb-grouped by first underscore token
  - unknown verbs folded into 'other' (index 0)
  → returns (id_to_group[75], group_names, num_groups≈47)
```

### Label Remap Path (4 production sites)
| Site | File:Line | Mechanism |
|------|-----------|-----------|
| Dataset init (activity_ids) | `industreal_dataset.py:796-804` | remap_activity_label on all action_labels |
| Per-frame __getitem__ | `industreal_dataset.py:904-908` | remap_activity_label on single label |
| Sequence path | `industreal_dataset.py:1021-1023` | remap_activity_label on window majority vote |
| Training class_counts | `train.py:3365-3386` | **No remap** — already in group space (FIX: no double-remap) |

### Sampler Path (`industreal_dataset.py:1427-1511`)
```
Layer 1: Activity balance (line 1428)
  class_weights = 1/max(counts, 15.0)  — equal mass for ≥15-frame classes

Layer 2: Task-aware boost (line 1450)
  det/psr-bearing frames get ×2.0/×1.5 within-pool boost

Layer 3: DET_GT_FRAME_FRACTION (line 1475)
  Redistributes: 40% mass → GT pool, 60% → non-GT pool
  Within-pool relative weights preserved

Diagnostic: Per-class sampling mass logged at line 1519
  → check max/min ratio at epoch 0, should be <10x
```

### Model Path (`models/model.py:1319-1422`)
```
ActivityHead: ACTIVITY_HEAD_SIMPLE=True
  TCN=None, ViT=None, CLS_token=None, activity_classifier=None
  simple_classifier = Sequential(
    LayerNorm(512), Linear(512→256), GELU, Dropout(0.3), Linear(256→47)
  )
  forward: early return at line 1421 — bypasses all temporal modules
```

### Loss Path (`training/losses.py:1063-1066, 1125-1135, 1333-1417`)
```
act_loss_fn = CrossEntropyLoss(label_smoothing=0.1)  # NOT CB-Focal
set_class_counts injects CB effective-number weights (beta=0.99)
  weights = 1 / ((1-0.99^counts) / 0.01), normalized to mean=1.0
Activity ramp: act_ramp = min(1, (epoch+1)/5) → [0.2, 0.4, 0.6, 0.8, 1.0]
```

### Eval Path (`evaluation/evaluate.py:956-958, 3491-3495, 1005-1008`)
```
macro-F1: present_labels filter → averages only GT-present groups → correct
pred_distinct: logged every epoch in [DIVERSITY] → collapse warning at <5
clip-level accuracy: 16 uniform frames per clip, majority vote → the benchmark number
segment-label remap: Q5 fix applied (evaluate.py:875-877)
```

### Opus Verification Checklist
- [ ] At epoch 1: `pred_distinct` ≥ 10 groups (not <5 collapse)
- [ ] At epoch 1: entropy ≥ 1.5 nats (not < 1.0 collapse)
- [ ] At epoch 1: `act_macro_f1` ≥ 0.05
- [ ] Per-class sampling mass max/min ratio < 10x at epoch 0
- [ ] Segment eval `act_seg_top1` is valid (label remapped to group space)

---

## 3. PSR PIPELINE (11 components, binary focal loss)

### Config Path (`config.py`)
| Setting | Line | Value |
|---------|------|-------|
| `NUM_PSR_COMPONENTS` | 432 | 11 |
| `PSR_FOCAL_GAMMA` | 961 | **0.5** (NOT plain BCE) |
| `PSR_FOCAL_ALPHA` | 958 | 0.25 |
| `PSR_WEIGHT` | 768 | 10.0 (before Kendall) |
| `PSR_WARMUP_STEPS` | 896 | **500** (step-based, 2.0→1.0) |
| `PSR_WARMUP_EPOCHS` | 925 | **3** (epoch-based for staged path) |
| `PSR_WARMUP_INIT_MULT` | 895 | 2.0 |
| `USE_PSR_SEQUENCE_MODE` | 973 | True |
| `PSR_SEQUENCE_LENGTH` | 974 | 8 |
| `PSR_SEQ_EVERY_N_BATCHES` | 977 | 2 |
| `DETACH_PSR_FPN` | 919 | True |
| `PSR_COMP_WEIGHTS` | 970 | [1.0, 1.21, ..., 4.61] |

### Model Path (`models/model.py:1535-1743`)
```
PSRHead: CausalTransformer(3 layers, 4 heads, d_model=256)
  ├── per_frame_mlp: multi-scale FPN GAP → 256-D feature
  ├── transformer: 3-layer causal encoder with upper-triangular mask
  └── output_heads: 11 per-component MLPs (256→64→1)
```

### Loss Path (`training/losses.py:1071-1075, 1137-1168, 1418-1492`)
```
binary_focal_loss (gamma=0.5, alpha=0.25) — NOT BCEWithLogitsLoss
per-component alpha from set_psr_class_counts: 2*(1-prevalence), clamp min=0.1
  → in focal loss, clamped max=1.0 (all components with prevalence <0.5 get alpha=1.0)
comp_weights: multiplicative per-component scaling (normalized by mean)
Both weighting mechanisms active simultaneously — intentional design
```

### Eval Path (`evaluation/evaluate.py:2434-2550, 3755-3766`)
```
psr_overall_f1: macro-F1 across 11 components (from decoded transition alignments)
psr_comp_acc: NEW — binary sigmoid threshold accuracy (for quick go/no-go)
```

### Opus Verification Checklist
- [ ] At epoch 1: `psr_loss` (Kendall-weighted) should be in 0.2-1.5 range
- [ ] At epoch 5: `psr_comp_acc` ≥ 0.50 (above chance)
- [ ] At epoch 5: `psr_overall_f1` ≥ 0.20
- [ ] **Note PSR is per-frame (T=1) for non-sequence batches** — paper must call it "per-frame component recognition," NOT "transition detection"

---

## 4. HEAD POSE PIPELINE (9-DoF, real GT from pose.csv)

### Config Path (`config.py`)
| Setting | Line | Value |
|---------|------|-------|
| `TRAIN_HEAD_POSE` | 41 | True |
| `HEAD_POSE_POS_SCALE` | 761 | 100.0 (applied at dataset load, before MSE) |
| `HEAD_POSE_LOSS_WEIGHT` | 855 | 5.0 |
| `HEAD_POSE_LOSS_CAP` | 760 | 30.0 |
| `KENDALL_HP_PREC_CAP` | 74 | True (head pose precision ≤ detection precision) |
| `NUM_KEYPOINTS` | 413 | 17 COCO (pseudo, no real GT — NOT a reportable task) |

### Data Path (`industreal_dataset.py:570-620`)
```python
_parse_pose: reads pose.csv → [num_frames, 9] float array
  columns: forward[3], position[3], up[3]
  position /= HEAD_POSE_POS_SCALE=100 at load time → O(1) range
  forward/up vectors sanity-checked for unit norm
```

### Model Path (`models/model.py:1484-1533`)
```
HeadPoseHead:
  GAP(C4=384ch) + GAP(C5=768ch) → concat [1152]
  → Linear(1152→512) → LayerNorm → GELU → Dropout(0.15)
  → Linear(512→256) → LayerNorm → GELU → Dropout(0.1)
  → Linear(256→9) → [B, 9]
```
HeadPoseFiLM (`models/model.py:1832-1836`): second-stage FiLM from 9-DoF → modulates c5_mod

### Loss Path (`training/losses.py:938-965, 1535-1565`)
```python
head_pose_loss = split_MSE + norm_regularizer
  fwd_err = MSE(fwd_norm, fwd_gt)    # angular via normalized vectors
  pos_err = MSE(pos_pred, pos_gt)     # positional (O(1) scaled)
  up_err  = MSE(up_norm, up_gt)       # angular via normalized vectors
  norm_reg = (||fwd||-1)² + (||up||-1)²  # encourages unit vectors
```

### Kendall HP Precision Cap (`training/losses.py:1689-1690`)
```python
if KENDALL_HP_PREC_CAP:
    lv_hp = torch.maximum(lv_hp, lv_det.detach())  # prec_hp ≤ prec_det
```

### Eval Path (`evaluation/evaluate.py:1840-1875`)
```
forward_angular_MAE_deg: proper unit-vector arccos in degrees ✓
position_MAE_mm: multiplied by 1000 — **UNIT UNVERIFIED, DO NOT REPORT**
  (code comment at evaluate.py:1861-1866 explicitly warns)
```

### Body Pose Clarification (`config.py:406-412`)
```python
# These 17 COCO-style keypoints are NOT annotated in IndustReal.
# Pseudo-generated from detection boxes (model.py:1972-1976).
# Wing Loss trains against pseudo-targets — no ground truth.
# DO NOT report body pose metrics.
```

### Opus Verification Checklist
- [ ] At epoch 1: `forward_angular_MAE_deg` should be < 60° (not random)
- [ ] At epoch 1: `log_var_det.device` is `cuda:0` (not cpu — confirmed safe, no device bug)
- [ ] Confirm `log_var_pose` is **shared** between body pose (dead) and head pose (real) — intentional
- [ ] **NOTE**: No IndustReal benchmark exists for head pose — this is a novel contribution. Report forward-gaze only (up-vector ~95° is unlearned).

---

## 5. MULTI-TASK KENDALL ORCHESTRATION

### Task Groups
| Task | Loss | Kendall log_var | Benchmark? |
|------|------|----------------|------------|
| Detection | Focal + GIoU (2:1 weight) | `log_var_det` | ✅ vs YOLOv8m 0.838 |
| Activity | CE + CB weights | `log_var_act` | ✅ vs MViTv2 (but grouped — different task) |
| PSR | Binary focal + comp_weights | `log_var_psr` | ⚠️ per-frame, not transition order |
| Head pose | 9-DoF split MSE | `log_var_pose` (shared with body) | ❌ No existing baseline — first reported |

### Kendall Bounds (`config.py:888-890`)
```
KENDALL_LOG_VAR_MIN_ACT: -0.5   activity prec max 1.65×
KENDALL_LOG_VAR_MAX_PSR: 0.0    PSR prec min 1.0× (cannot be suppressed)
KENDALL_LOG_VAR_MAX_POSE: 3.0   pose prec min 0.05×
```

### LR Schedule (`train.py:~3607-3644`)
```
SequentialLR([
  LinearLR(warmup, start_factor=0.1, total_iters=2),       # epochs 0-1: 0→~5e-5
  OneCycleLR(epochs=100, steps_per_epoch=1, pct_start=0.1) # epochs 2-99: rise 10 epochs, decay 90
], milestones=[2])
→ Effective: epochs 0-1 warmup → 2-9 rise to peak 5e-4 → 10-99 cosine decay to ~0
```

### Smooth Loss Caps (`training/losses.py:1309-1314`)
```python
_smooth_cap(x, cap) = x if x <= cap else cap * (1 + log(x/cap))  # gradient never zero
```
All 5 caps: DET=50, POSE=30, ACT=80 (safety net, never binds for CE), PSR=20, HEAD_POSE=30

### Opus Verification Checklist
- [ ] At runtime, log `[Kendall log_sigma]` shows all 4 log_vars in [-1, +1] at epoch 1
- [ ] At runtime, `[GRAD-NORM step=N]` shows all 4 heads with non-zero gradient
- [ ] At runtime, LR scheduler log shows proper warmup → OneCycleLR schedule
