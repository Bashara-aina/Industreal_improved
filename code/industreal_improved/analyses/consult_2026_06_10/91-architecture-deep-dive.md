# 02 — Architecture Deep-Dive

## Overview

POPW is a multi-task architecture based on popw_paper.tex that simultaneously predicts detection (ASD), activity classification, 9-DoF head pose, and part state recognition (PSR) from a single RGB frame. Architecture verification reports claim **95%+ alignment** with the paper.

## Backbone

- **ConvNeXt-Tiny** (28M params) — paper mandates ConvNeXt-Tiny, NOT ResNet-50
- FPN (Feature Pyramid Network): 4,474,880 params
- Input: 1280×720 RGB frames (full resolution)
- IMAGENET_MEAN/STD normalization
- CUDNN_DETERMINISTIC=False, CUDNN_BENCHMARK=True, ALLOW_TF32=True

**CORRECTION — TMA Cell parameter count unclear:** The original file stated "0 params" for the GRU-based TMA Cell. A GRU cell inherently has gate weight matrices (update, reset, new gates with weight + bias). If truly 0 params, it would be an identity no-op. The parameter count may be shared with backbone conv features, or the GRU may not be properly instantiated. This needs verification.

## Auxiliary Modules

### TMA Cell (GRU-based Temporal Masked Attention)
- `USE_TMA_CELL=True` — GRU-based temporal conditioning
- Processes frame-level features with temporal context
- **Parameter count unclear** — see correction above. A GRU inherently has ~3*hidden_dim*(input_dim+hidden_dim+1) params.

### Temporal Feature Bank
- `USE_TEMPORAL_BANK=True` — stores T=16 frame history
- `feature_bank_detach=True` — gradient detached through bank to prevent double-backward crash
- `feature_bank_slot_overwrite=True` — overwrite oldest slot rather than appending

### Hand-FiLM
- `use_hand_film=True` — FiLM conditioning on activity using hand features
- Requires `use_hand_crops` in dataset config

## Heads (5 total, all active in RF4)

### 1. Detection Head (ASD)
- **Params:** 5,305,596
- Predicts: class logits + bounding box regression for **24 detection classes** (background + 22 assembly states + error_state, per NUM_DET_CLASSES=24). **CORRECTION:** Was previously listed as 63 action classes — that's the activity head count.
- Loss: FocalLoss(α=0.25, γ=2.0) for class + IoU/GIoU for box. **NOTE:** DET_ASYMMETRIC_GAMMA=True overrides effective gamma to gamma_pos=0.0, gamma_neg=1.5 (config.py lines 732-734), NOT FOCAL_GAMMA=2.0. Alpha=0.25 with gamma_pos=0 means positive loss = 0.25*CE — potentially suppressing positive gradient.
- Architecture: Anchor-free per paper, DET_EVAL_SCORE_THRESH=0.001 for evaluation
- DET_EVAL_NMS_IOU_THRESH=0.5, DET_EVAL_MAX_PER_IMAGE=300
- DET_POS_IOU_THRESH=0.4, DET_POS_IOU_TOP_K=9, DET_POS_IOU_IOU_FLOOR=0.2
- DET_OHEM_ENABLED=True, DET_ASYMMETRIC_GAMMA=True
- Target: ASD mAP@0.5 of 70-78% (paper projection)

### 2. Activity Head
- **Params:** 687,173
- Predicts: activity class logits (**75 raw classes → ~41-47 hybrid grouped outputs** per ACT_CLASS_GROUPING='hybrid', ACT_HYBRID_THRESHOLD=100). **CORRECTION:** Raw count is 75 (IDs 0-74), not 69. Grouped count depends on dataset frame counts (config comment says "47 hybrid groups" at line 869).
- Architecture: ViT temporal block + TCN (depthwise Conv1d with groups=embed_dim). ACTIVITY_HEAD_SIMPLE=True bypasses this with a 150K MLP — the root cause fix for activity collapse.
- ViT attn_dropout: paper says 0.1, was 0.3 — **FIXED to 0.1**
- Activity ramp-up: ACT_RAMP_EPOCHS=5 epochs
- Sampler: ACT_SAMPLER_MODE='balanced' with ACT_SAMPLER_COUNT_FLOOR=15.0
- **CRITICAL NOTE:** Activity uses hybrid grouping. The activity collapse root cause was the FeatureBank gradient being severed by in-place tensor assignments (config.py lines 883-885). This has been FIXED with ACTIVITY_HEAD_SIMPLE=True + ACTIVITY_GRAD_BLEND_RATIO=1.0. Verb-grouping questions are partially moot since the dominant factor was gradient flow.
- Target: 55-63% Top-1 (RGB-only), 62-68% with VideoMAE

### 3. PSR Head (Part State Recognition)
- **Params:** 3,077,515
- Predicts: 11 PSR states (pre/post conditions for each part)
- Architecture: Transformer with d_model=128→256 (FIXED from half paper spec), GRU hidden=256
- **Sequence mode:** USE_PSR_SEQUENCE_MODE=True, PSR_SEQUENCE_LENGTH=8 frames
- **Seq cadence:** PSR_SEQ_EVERY_N_BATCHES=2 — every 2nd batch is PSR-only sequence
- USE_PSR_TRANSITION=True — transition-aware loss with PSR_SENSITIVITY_WEIGHT=0.50
- `detach_psr_fpn=True` — PSR gradient detached from FPN
- PSR_WARMUP_EPOCHS=3, PSR loss cap=20.0
- Target: PSR F1@±3 of 0.50-0.65

### 4. Pose Head (9-DoF Head Pose)
- **Params:** 1,643,793
- Predicts: 9-DoF head pose (forward, up, position vectors)
- Loss: WingLoss with weight=target_confidence, loss_pose × 0.001 per paper
- HeadPoseFiLM: 400,896 params — conditions activity on head pose
- Pose FiLM: 841,216 params — conditions something on pose
- KENDALL_HP_PREC_CAP=True — head pose precision can never exceed detection precision
- **Pose converged extremely fast** — loss dropped from 8.38 to 0.69 within 500 batches of epoch 1

### 5. Detection Head Subcomponents
- The detection head outputs: **cls_score** (classification logits, 24 classes) and **reg_pred** (box deltas). **CORRECTION:** There is NO objectness branch — this is a standard anchor-free RetinaNet head.
- The bias init for cls uses config-driven REINIT_PI. **Note:** model.py DetectionHead._init_weights() hardcodes pi=0.03 (bias=-3.48) while the reinit function uses REINIT_PI=0.01 (bias=-4.60) — a 3x discrepancy producing different bias values.

## Parameter Count

Total trainable: ~53M (without VideoMAE), ~75M (with VideoMAE frozen)
Paper target: <50M (close, within margin)

Breakdown:
- Backbone (ConvNeXt-Tiny): ~28M
- FPN: 4.47M
- Detection: 5.31M
- Pose: 1.64M + 0.84M (FiLM) + 0.40M (HeadPoseFiLM) = 2.88M
- Activity: 0.69M
- PSR: 3.08M
- Feature Bank: 0 (parameterless buffer)

## Multi-Task Architecture

### Kendall Loss Weighting
- `USE_KENDALL=True` — adaptive loss weighting via learned log_var parameters
- `KENDALL_STAGED_TRAINING=False` — single curriculum, not staged
- `KENDALL_FIXED_WEIGHTS=False` — learned, not fixed
- `KENDALL_HP_PREC_CAP=True` — head pose capped
- `KENDALL_LOG_VAR_MIN_ACT=-0.5` — activity allowed moderate precision boost
- `KENDALL_LOG_VAR_MAX_PSR=0.0` — PSR can't be suppressed
- `KENDALL_LOG_VAR_MAX_POSE=3.0` — pose can be suppressed (it dominates)

### Gradient Flow
- `detach_reg_fpn=False` — regression gradient flows to FPN (FIXED from old detach)
- `detach_psr_fpn=True` — PSR detached from FPN. **CORRECTION:** This means PSR contributes ZERO gradient to the backbone. Combined with backbone grads zeroed on PSR seq batches, PSR is entirely isolated from shared feature learning. This fundamentally undermines the multi-task thesis for the PSR head.
- `feature_bank_detach=True` — bank detached to prevent double-backward
- Backbone grads zeroed on PSR seq batches to prevent PSR from pulling backbone features away from detection

### EMA
- `USE_EMA=True` with EMA_DECAY=0.995 (paper specifies 0.999)
- EMA weights used for best checkpoint

## Training Setup

- Optimizer: AdamW with differential LR
  - Backbone: 0.1x BASE_LR (5e-5)
  - Heads: 1.0x BASE_LR (5e-4)
  - Bias: 0.3x
- Scheduler: OneCycleLR (pct_start=0.1, steps_per_epoch=1, called once per epoch)
  - **FIXED:** Was incorrectly built with steps_per_epoch=len(train_loader)//accum_steps but called once per epoch
  - **CORRECTION:** pct_start=0.3 may be active (optimizer.py line 58). If so, peak LR at epoch 31, not 10.
- Warmup: 2 epochs (WARMUP_EPOCHS=2)
- Weight decay: 1e-3 (FIXED from 5e-2 which dominated gradients)
- Gradient clip: 5.0 (FIXED from 1.0 which was too tight)
- GRAD_ACCUM_STEPS=8, EFFECTIVE_BATCH=48
- MIXED_PRECISION=False (full FP32 — PSR seq loss spikes corrupt GradScaler)

### Effective Batch Mismatch with Paper

**CORRECTION:** Current EFFECTIVE_BATCH=48 vs paper specification of 16. This is a 3x increase without corresponding LR adjustment. Per the linear scaling rule, BASE_LR should be ~1.5e-3 (3x current 5e-4) for batch 48. Without this, the model receives 1/3 the parameter updates per epoch relative to the reference — slowing all head convergence.

Options:
1. Scale LR to ~1.5e-3 for heads, backbone to ~1.5e-4
2. Reduce GRAD_ACCUM_STEPS from 8 to 2 (effective batch = 12, close to paper's 16)
3. Keep current setup and accept slower convergence (valid if convergence quality improves)

### Backbone LR Differential Implication

The 10x LR differential (backbone: 5e-5, heads: 5e-4) means backbone representation learning is an order of magnitude slower than head adaptation. For early training this preserves ImageNet features, but after extended training, heads may saturate while the backbone is still adapting. Starting from epoch 3+, consideration should be given to whether the backbone LR can be increased.
