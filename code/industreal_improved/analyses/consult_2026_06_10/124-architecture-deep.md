# 124 — Multi-Task IndustReal Architecture: Complete Deep Dive for Opus

## Table of Contents

1. System Overview
2. ConvNeXt-Tiny Backbone (28.6M params, 71 GFLOPs)
3. FPN Neck (4.5M, 5.5 GFLOPs)
4. Detection Head (5.3M, RetinaNet-style)
5. Head Pose Head (0.8M, MLP)
6. Activity Head (0.7M, per-frame MLP)
7. PSR Head (3.1M, 3-layer Causal Transformer)
8. Body Pose Head (1.6M, aux FiLM)
9. Kendall Uncertainty Weighting
10. HP_PREC_CAP (Head-Pose Precision Cap)
11. Combined Metric Weights
12. Training Loop (train.py: epoch loop, batch loop, seq batches, validation)
13. Evaluation Pipeline (evaluate.py: evaluate_all, detection, activity, PSR, head pose)
14. Subprocess Evaluation (subprocess_eval.py)
15. TTA + Soft-NMS (eval_tta.py)
16. eval_post_reinit.py
17. F1-F22b Complete Training Fixes
18. 5-Bug-Fix History
19. 10 NaN to 0.0 Fix Locations
20. det_mAP50 epoch=-1 Fix
21. Per-File Line Reference Index

---

## 1. System Overview

The IndustReal multi-task model jointly learns 4 tasks from a single egocentric RGB camera:
- Assembly State Detection (ASD): 24-class object detection
- Action Recognition (AR): 75-class per-frame activity classification (grouped to ~13-47 verb groups)
- Head Pose: 9-DoF regression (forward vector + position + up vector)
- Procedure Step Recognition (PSR): 11-component fill-forward assembly state tracking

Architecture: ConvNeXt-Tiny backbone -> FPN neck -> 5 task heads + 2 FiLM conditioning modules. Loss: Kendall homoscedastic uncertainty with 4 learnable log-vars. Single GPU training (RTX 5060 Ti 16GB), batch size 6, grad accumulation 8.

Total parameters: 46.47M. Total GFLOPs: 245.3. Single GPU inference FPS: 11.02.

---

## 2. ConvNeXt-Tiny Backbone (28.6M params, 71 GFLOPs)

**File:** `/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/models/model.py`
**Lines:** 175-262 (class ConvNeXtBackbone)
**Factory:** 373-384 (build_backbone)

### Architecture

ImageNet-pretrained ConvNeXt-Tiny, 28.6M parameters, ~71 GFLOPs at 1280x720 input. Replaces ResNet-50 (25M params) for improved ImageNet performance (+1.5%).

Four stages producing multi-scale feature maps:

| Stage | Feature | Stride | Channels | Module | Line |
|---|---|---|---|---|---|
| Stage 0 (features[0]+features[1]) | C2 | 4 | 96 | stem + stage1 | 228-230 |
| Stage 1 (features[2]+features[3]) | C3 | 8 | 192 | downsample2 + stage2 | 232-234 |
| Stage 2 (features[4]+features[5]) | C4 | 16 | 384 | downsample3 + stage3 | 236-238 |
| Stage 3 (features[6]+features[7]) | C5 | 32 | 768 | downsample4 + stage4 | 240-242 |

### Gradient Checkpointing

When USE_BACKBONE_CHECKPOINT=True (config.py:167), each of the 4 stages is wrapped with torch.utils.checkpoint.checkpoint() (model.py:251-255). This trades ~20% compute for ~50% activation memory reduction during backprop. Required to prevent OOM when USE_PSR_SEQUENCE_MODE=True on RTX 3060 12GB.

### ConvNeXt vs ResNet-50 Channel Map

| Level | ResNet-50 | ConvNeXt-Tiny |
|---|---|---|
| C2 | 256 | 96 |
| C3 | 512 | 192 |
| C4 | 1024 | 384 |
| C5 | 2048 | 768 |

The ConvNeXt uses LayerNorm internally (no frozen BN like ResNet-50). The lower C5 channel count (768 vs 2048) impacts PoseFiLM (model.py:710-715): when C5 != FILM_DIM (768), a 1x1 conv projects C5 to 768 channels.

### Stage Freezing

Controlled by set_backbone_stage_requires_grad() at model.py:265-308. During staged training:

- Stage 1 (epochs 1-5): stages[0-1] frozen (C2/C3 not trainable)
- Stage 2 (epochs 6-15): stage[0] frozen (C2 not trainable)
- Stage 3 (epochs 16+): all trainable

### Forward Path (model.py:216-262)

```
Input: [B, 3, 1280, 720]
-> stage0 -> C2 [B, 96, 320, 180]
-> stage1 -> C3 [B, 192, 160, 90]
-> stage2 -> C4 [B, 384, 80, 45]
-> stage3 -> C5 [B, 768, 40, 23]
```

### Key Fix History

| Date | Fix | File:Line |
|---|---|---|
| 2026-06-15 | Gradient checkpointing: USE_BACKBONE_CHECKPOINT | model.py:1777-1778 |
| 2026-07-04 | NaN guard: _sanitize() after backbone forward | model.py:1953-1958 |

---

## 3. FPN Neck (4.5M params, 5.5 GFLOPs)

**File:** `src/models/model.py`
**Lines:** 390-440 (class FPN)
**Construction:** 1804 (self.fpn)

### Architecture

Takes [C3, C4, C5] from backbone, produces [P3, P4, P5, P6, P7]. All pyramid levels have 256 channels. Standard top-down FPN with lateral 1x1 convs and 3x3 smoothing.

### Level Map

| Level | Input | Stride | Channels | Construction | Line |
|---|---|---|---|---|---|
| P3 | C3 (lateral 1x1) + P4 up | 8 | 256 | 429-433 |
| P4 | C4 (lateral 1x1) + P5 up | 16 | 256 | 430 |
| P5 | C5 (lateral 1x1) | 32 | 256 | 429 |
| P6 | C5 (stride-2 conv 3x3) | 64 | 256 | 437 |
| P7 | P6 (stride-2 conv 3x3) | 128 | 256 | 438 |

### Channel Configuration

P6 derived directly from C5 (768ch) via stride-2 conv. P7 from ReLU(P6) via stride-2 conv. Both skip the lateral/top-down path.

### Consumer Heads

| Head | FPN Level(s) | Reason |
|---|---|---|
| Detection (RetinaNet) | P3-P7 all levels | Multi-scale object detection |
| Body Pose | P3 only | Highest resolution for keypoint localization |
| Activity | P4 (via GAP) | Mid-level semantic features |
| PSR | P3+P4+P5 (via GAP + concat) | Multi-scale assembly state features |
| PoseFiLM | C5 direct (bypasses FPN) | Pose conditioning at highest semantic level |

### Gradient Isolation

Two independent gradient isolation mechanisms protect shared FPN features:

1. **Detection regression isolation** (config.py:1002, model.py:562): When DETACH_REG_FPN=True, reg_subnet receives feat.detach() so regression gradients don't corrupt shared FPN.

2. **PSR isolation** (config.py:1008, model.py:2091-2094): When DETACH_PSR_FPN=True, PSR receives detached p3/p4/p5 so PSR loss spikes (~23.9 at step 850) don't corrupt detection features.

---

## 4. Detection Head (5.3M params, RetinaNet-style)

**File:** `src/models/model.py`
**Lines:** 500-567 (class DetectionHead)
**Construction:** 1807-1810

### Architecture

RetinaNet-style with shared cls/reg subnets across all FPN levels (P3-P7). Each subnet: 4x Conv3x3+ReLU, followed by output conv. GroupNorm(8) after each conv layer (not BatchNorm).

```
cls_subnet: Conv3x3 -> GroupNorm(8) -> ReLU -> (x4) -> cls_score: Conv(256, 9*24)  [B, 216, H, W]
reg_subnet: Conv3x3 -> GroupNorm(8) -> ReLU -> (x4) -> reg_pred: Conv(256, 9*4)    [B, 36, H, W]
```

### Anchor Configuration

| Parameter | Value | Config Key | Line |
|---|---|---|---|
| Anchor sizes | (96, 160, 256, 384, 512) | ANCHOR_SIZES | config.py:499 |
| Ratios | (0.5, 1.0, 2.0) | — | model.py:454 |
| Scales | (1.0, 2^(1/3), 2^(2/3)) | — | model.py:455 |
| Anchors per location | 9 (3 ratios x 3 scales) | — | model.py:461 |
| Strides | [8, 16, 32, 64, 128] | — | model.py:468 |

Anchor generator at model.py:446-494. Total anchors per image: sum(P3 + P4 + P5 + P6 + P7 grid cells x 9) ≈ 173K.

### Bias Initialization

```python
# model.py:542-543
pi = 0.03
nn.init.constant_(self.cls_score.bias, -math.log((1 - pi) / pi))
```

Prior probability pi=0.03 (bias=-3.49). Changed from 0.01 (bias=-4.60) in RF1 fix. The default RetinaNet pi=0.01 provides too aggressive a background prior when anchors poorly fit small assembly parts.

### Anchor Matching

In losses.py (class FocalLoss, lines 74-200):

| Parameter | Value | Config Key | Line |
|---|---|---|---|
| Positive IoU threshold | 0.4 | DET_POS_IOU_THRESH | config.py:500 |
| Negative IoU threshold | 0.4 | DET_NEG_IOU_THRESH | config.py:504 |
| Top-k force-match | 9 | DET_POS_IOU_TOP_K | config.py:505 |
| IoU floor for force-match | 0.2 | DET_POS_IOU_IOU_FLOOR | config.py:506 |

Key change: IoU threshold lowered from 0.5 to 0.4 (config.py:500). At 0.5, small assembly parts (h≈156px) match only ~1 anchor/GT. At 0.4, ~3-5x more positive anchors fire.

### Asymmetric Focal Loss

| Parameter | Value | Config Key | Line |
|---|---|---|---|
| Focal alpha | 0.50 | FOCAL_ALPHA | config.py:704 |
| Focal gamma | 2.0 | FOCAL_GAMMA | config.py:705 |
| Gamma positive | 0.0 | DET_GAMMA_POS | config.py:770 |
| Gamma negative | 1.5 | DET_GAMMA_NEG | config.py:771 |

Asymmetric gamma: gamma_pos=0 (no easy-positive suppression) prevents positive gradient starvation. Without this, well-classified positives (p≈0.9) have near-zero gradient under gamma=2 on both sides.

### OHEM (Hard Negative Mining)

| Parameter | Value | Config Key | Line |
|---|---|---|---|
| Enabled | True | DET_OHEM_ENABLED | config.py:756 |
| Ratio | 2.0 (negatives per positive) | DET_OHEM_RATIO | config.py:757 |
| Min negatives | 32 | DET_OHEM_MIN_NEG | config.py:760 |

### Loss Weighting

- GIoU weight: 2.0 (config.py:749)
- GIoU loss replaces SmoothL1 for box regression
- Focal + GIoU combined: loss_det = cls_loss + 2.0 * giou_loss (losses.py:1235)

### Key Fix History

| Date | Fix | File:Line |
|---|---|---|
| 2026-06-20 | Top-k force-match per GT (DET_POS_IOU_TOP_K=9) | losses.py:138-153 |
| 2026-06-20 | Per-class alpha config (DET_CLASS_ALPHAS) | config.py:712-748 |
| 2026-06-21 | IoU floor for force-match (IOU_FLOOR=0.2) | config.py:506 |
| 2026-06-12 | Empty-frame background loss (DET_EMPTY_SAMPLE=2048) | config.py:873-874 |
| 2026-06-16 | Regression gradient warmup for reinit | losses.py:1222-1234 |
| 2026-07-02 | Focal alpha raised 0.25 -> 0.50 | config.py:704 |

---

## 5. Head Pose Head (0.8M params, MLP)

**File:** `src/models/model.py`
**Lines:** 1484-1533 (class HeadPoseHead)
**Construction:** 1848-1860 (gated by USE_GEO_HEAD_POSE)

### Architecture

Multi-scale fusion: GAP(C4) + GAP(C5) -> concat -> MLP 1152->512->256->9.

```
C4 [B, 384, H/16, W/16] -> GAP -> [B, 384]
C5 [B, 768, H/32, W/32] -> GAP -> [B, 768]
Concat -> [B, 1152]
FC(1152->512) + LayerNorm + GELU + Dropout(0.15)
FC(512->256) + LayerNorm + GELU + Dropout(0.1)
FC(256->9) -> head_pose [B, 9]
```

### Output: 9-DoF

Position (3): x, y, z world coordinates
Forward vector (3): gaze direction
Up vector (3): head orientation

### Loss

Split loss: position MSE + direction MSE (losses.py:1559-1565). The raw position is divided by HEAD_POSE_POS_SCALE=100.0 (config.py:843) to bring coordinates to O(1).

Weight multiplier: HEAD_POSE_LOSS_WEIGHT=5.0 (config.py:954, per paper "L_hp = MSE x 5.0"). Applied at losses.py:1574.

### Geometry-Aware Variant (USE_GEO_HEAD_POSE=True)

When USE_GEO_HEAD_POSE=True (config.py:1100), replaces HeadPoseHead with GeometryAwareHeadPose from src/models/head_pose_geo.py. Uses 6D continuous rotation (Zhou et al., CVPR 2019) rather than raw 9-number MSE. Output is reconstructed to [B,9] for downstream compatibility (model.py:2161-2166).

### Key Fix History

| Date | Fix | File:Line |
|---|---|---|
| 2026-06-17 | Custom weight init (was missing _init_weights) | model.py:1515-1527 |
| 2026-06-14 | Separate head_pose from body-pose in the Kendall grouping | — |

---

## 6. Activity Head (0.7M params, per-frame MLP)

**File:** `src/models/model.py`
**Lines:** 1262-1478 (class ActivityHead)
**Construction:** 1863-1873

### Architecture (Simple Mode)

When ACTIVITY_HEAD_SIMPLE=True (config.py:950):

```
det_conf -> sigmoid(cls_preds.max(dim=1)) -> [B, 24]  (stop_grad)
c5_mod -> GAP -> [B, 768] (ConvNeXt-Tiny) or [B, 2048] (ResNet-50)
p4 -> GAP -> [B, 256]
Concat -> [B, 24+768+256] = [B, 1048]
proj_features: Linear(1048->512) -> proj_feat [B, 512]
simple_classifier: LayerNorm -> Linear(512->256) -> GELU -> Dropout(0.3) -> Linear(256->75)
```

Total: ~150K params (vs 8.2M for TCN+2xViT).

### Full Temporal Mode (ACTIVITY_HEAD_SIMPLE=False)

When ACTIVITY_HEAD_SIMPLE=False, the full temporal stack is used:

```
proj_feat [B, 512]
FeatureBank: ring buffer T=16, stores f̃_t-t+1 ... f̃_t
TCN: TemporalConvBlock (depthwise conv k=5 + pointwise conv 1x1 + residual)
2x ViT blocks (8 heads, d_k=64, FFN 512->2048->512, DropPath 0.1/0.15)
CLS token cross-attention pooling
Dropout(0.1) -> Linear(512->75)
```

### Feature Bank Gradient Path

**ROOT CAUSE FIX (2026-06-30 v4, model.py:1239-1244):** The old code used in-place assignment (bank_i[-1] = feat_i). This does NOT propagate gradients because bank_i (from torch.stack of detached tensors) has requires_grad=False. Fixed to use torch.cat([bank_i[:-1].detach(), feat_i.unsqueeze(0)], dim=0), which creates a new tensor where the last position carries feat_i's gradient.

### Gradient Blending

Per paper section 5.4: "blend_ratio * C5_mod2 + (1 - blend_ratio) * detach(C5_mod2)"

```python
# model.py:2175-2176
_blend = float(getattr(C, 'ACTIVITY_GRAD_BLEND_RATIO', 0.05))
c5_mod_blend = _blend * c5_mod + (1 - _blend) * c5_mod.detach()
```

Progression: 0.05 -> 0.10 -> 0.30 -> 0.50 -> 0.70 -> 1.00 (current in config.py:971).

### Activity Loss

Per paper section 3.7.1: CrossEntropyLoss with label_smoothing=0.1 (config.py:796). CB_FOCAL_BETA=0.999, CB_FOCAL_GAMMA=2.0 used when USE_CB_FOCAL_ACT=True (config.py:800-802). LDAM-DRW available as ablation (config.py:1025-1038).

Activity warmup ramp: ACT_RAMP_EPOCHS=3 (config.py:833). Ramps from 0 to 1 over 3 epochs using stage-local epoch counter (losses.py:1385-1389).

### Class Grouping

ACT_CLASS_GROUPING='hybrid' (config.py:315): classes with >= ACT_HYBRID_THRESHOLD=100 frames (config.py:316) stay standalone; the rest are verb-grouped by first underscore token. Reduces output dimension from 75 to ~13-47 groups.

---

## 7. PSR Head (3.1M params, 3-layer Causal Transformer)

**File:** `src/models/model.py`
**Lines:** 1536-1736 (class PSRHead)
**Construction:** 1876-1881

### Architecture

```
Per-frame: GAP(P3) + GAP(P4) + GAP(P5) -> concat [B, 768]
-> per_frame_mlp: Linear(768->512) -> LayerNorm -> GELU -> Dropout -> Linear(512->256) -> LayerNorm -> [B, 256]

Causal Transformer:
  3 layers, 4 heads, d_model=256, dim_feedforward=1024
  batch_first=True, norm_first=True (pre-norm)
  Upper-triangular causal mask: position t attends to positions 0..t only

Per-component output heads:
  11 separate tiny MLPs: Linear(256->64) -> GELU -> Dropout -> Linear(64->1)
  Each MLP outputs the binary transition logit for one assembly component
```

### Causal Mask (model.py:1624-1633)

```python
mask = torch.triu(ones, diagonal=1).bool()
# True = ignore (cannot attend to future)
# False = attend (past and current)
```

Generated on-demand, cached for reuse. At inference, per-recording cache stores up to 32 frames and processes incrementally (model.py:1682-1716).

### PSR Transition Objective (psr_transition.py)

**File:** `/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/models/psr_transition.py`
**Lines:** 1-319

The key insight: per-frame BCE on 95%-static labels collapses to constant pattern. Instead, predict transition EVENTS using Gaussian-smeared targets.

#### build_transition_targets (lines 31-70)

```
Input: [B, T, 11] binary fill-forward labels
For each component c:
  - Find 0->1 transition frames
  - Build Gaussian kernel: exp(-0.5*(t/sigma)^2), sigma=3.0
  - Convolve: targets[b, frame, c] = max over transitions
Output: [B, T, 11] Gaussian-smoothed transition indicators
```

#### MonotonicDecoder (lines 76-163)

Viterbi-like forward pass with procedure-order constraints:

```
For each frame t:
  can_transition = (current_state == 0)     # not yet placed
  predecessors_placed = all predecessors are 1
  can_transition = can_transition & predecessors_placed
  transition = (trans_prob > threshold) & can_transition
  current_state = clamp(current_state + transition.float(), max=1.0)
```

Key fix at lines 117-131 (F22b, 2026-07-03): Explicit dim handling. The old blanket .squeeze() collapsed a batch of one recording [1,T,C] to [T,C], then unsqueeze(1) produced [T,1,C] -- T independent length-1 sequences. The monotone constraint never applied across time.

#### Procedure Order Prior (model.py lines 92-96)

Default: comp0->comp1->comp2->... (sequential assembly). Used by MonotonicDecoder to enforce that certain components must be placed before others.

### PSR Sequence Mode

When USE_PSR_SEQUENCE_MODE=True (config.py:1062), every PSR_SEQ_EVERY_N_BATCHES batches (config.py:1078, default 4) uses a contiguous T-frame sequence instead of a single frame. This enables the causal Transformer to actually process temporal patterns.

### PSR Loss

Binary focal loss (config.py:1165-1166: PSR_FOCAL_GAMMA=0.5). Per-component weights from PSR_COMP_WEIGHTS (config.py:1055) for rare components.

PSR sensitivity penalty (losses.py:1482-1507): -log(std(per-component logits)) encourages per-component logits to separate rather than all sitting at the same value.

### Key Fix History

| Date | Fix | File:Line |
|---|---|---|
| 2026-06-15 | PSR_WARMUP_STEPS=500 | config.py:985 |
| 2026-06-30 | PSR_FOCAL_GAMMA reduced 2.0 -> 0.5 | config.py:1050 |
| 2026-07-02 | PSR_SEQ_EVERY_N_BATCHES 2->4 | config.py:1078 |
| 2026-07-02 | Gradient snapshot/restore for FPN isolation | train.py:1305-1339 |
| 2026-07-03 | MonotonicDecoder dim handling (F22b) | psr_transition.py:117-131 |

---

## 8. Body Pose Head (1.6M params, aux FiLM)

**File:** `src/models/model.py`
**Lines:** 573-620 (class PoseHead)
**Construction:** 1815-1819

### Architecture

Takes P3 (stride 8, 256ch) from FPN:

```
ConvTranspose2d(k=4, s=2, p=1) -> [B, 256, H/4, W/4]
GroupNorm(32) + ReLU
Heatmap head: Conv(256->256, 3x3) -> ReLU -> Conv(256->17, 1x1) -> [B, 17, H/4, W/4]
Soft-argmax(T=0.07) -> keypoints [B, 17, 2] + confidence [B, 17]
```

### Soft-Argmax (model.py:89-140)

Differentiable keypoint extraction from heatmaps:

```
weights = softmax(heatmap / temperature)   # [B, K, HW]
coords = sum(positions * weights)           # [B, K, 2]
confidence = sigmoid(max_heatmap)           # [B, K]
```

Training temperature: 1.0 (config.py:863) for gradient flow. Eval temperature: 0.07 (config.py:862) for coordinate precision.

### Body Pose: Dead Code (No Real Annotations)

IndustReal has NO COCO keypoint annotations. The 17 keypoints are pseudo-generated from the detection head's bounding boxes (model.py:1981-2037). The Wing Loss trains against these pseudo-keypoints. Configuration comment (config.py:453-458) explicitly states this.

When FREEZE_BODY_POSE_BRANCH=True (config.py:54, model.py:1822-1825), the body-pose sub-head is frozen and its loss is zeroed (losses.py:1335-1336). The TRUE pose task is head pose (9-DoF from pose.csv).

### PoseFiLM Module (model.py:626-716)

Keypoint-conditioned FiLM modulation on C5:

```
keypoints [B, 17, 2] + confidence [B, 17] -> flatten -> [B, 51]
gamma_net: Linear(51->512) -> ReLU -> Linear(512->768) -> 1+tanh -> (0,2)
beta_net: Linear(51->512) -> ReLU -> Linear(512->768) -> unbounded
C5_mod = gamma * C5_768 + beta   # [B, 768, H/32, W/32]
```

### HeadPoseFiLM Module (model.py:722-792)

Second-stage FiLM from 9-DoF head pose:

```
head_pose [B, 9] (stop_grad - CRITICAL per paper)
gamma_net: Linear(9->256) -> LayerNorm -> GELU -> Linear(256->768) -> 1+tanh
beta_net: Linear(9->256) -> LayerNorm -> GELU -> Linear(256->768) -> unbounded
C5_mod2 = gamma_hp * C5_mod + beta_hp
```

Note: head_pose is detached before HeadPoseFiLM (model.py:2168). This prevents second-stage FiLM gradients from corrupting head_pose head training.

---

## 9. Kendall Uncertainty Weighting (4 learnable log-vars)

**File:** `src/training/losses.py`
**Lines:** 1000-1790 (class MultiTaskLoss)

### Mathematical Form

Per paper section 3.7:

```
L_total = sum_t [ exp(-s_t) * L_t * ramp_t + s_t ]
```

Where s_t = log(sigma^2_t) is the learnable log-variance for task t. The exp(-s_t) term is the task precision. Tasks with higher uncertainty (larger variance) automatically get lower weight.

### Initialization

| Log-Var | Init Value | Physical Meaning | Line |
|---|---|---|---|
| log_var_det | 0.0 | precision = exp(0) = 1.0 | 1027 |
| log_var_pose | 0.0 | precision = exp(0) = 1.0 | 1033 |
| log_var_act | 0.0 | precision = exp(0) = 1.0 | 1034 |
| log_var_psr | 0.0 | precision = exp(0) = 1.0 | 1035 |

Note: log_var_pose is shared for BOTH body pose and head pose (paper spec: t in {det, pose+head_pose, act, psr}). This is intentional per the Kendall grouping.

### Clamp Bounds (losses.py:1702-1705)

| Log-Var | Min | Max | Rationale |
|---|---|---|---|
| log_var_det | -4.0 | 2.0 | Standard | 
| log_var_pose (lv_hp) | -4.0 | KENDALL_LOG_VAR_MAX_POSE | 3.0 (config.py:979) - allows suppression |
| log_var_act | KENDALL_LOG_VAR_MIN_ACT | 2.0 | -0.5 (config.py:977) - allows moderate boost |
| log_var_psr | -4.0 | KENDALL_LOG_VAR_MAX_PSR | 0.0 (config.py:978) - PSR can't be suppressed |

### Fixed-Weight Path (losses.py:1666-1689)

When KENDALL_FIXED_WEIGHTS=True (config.py:96), bypasses learned Kendall log_vars entirely:

- Detection: weight 1.0
- Head pose: weight KENDALL_HP_FIXED_LAMBDA (config.py:108, default 0.2)
- Activity: weight ACTIVITY_LOSS_WEIGHT (config.py:938, default 0.8)
- PSR: weight PSR_WEIGHT (config.py:850, default 10.0)

This is the bootstrap mode for RF1-RF2 where detection must drive the backbone.

### Log-Var Gradient Logging (train.py:66)

LOG_KENDALL_GRAD_EVERY=500 (config.py:66). Logs log_var values, effective precisions, and log_var gradients for observability.

### Key Fix History

| Date | Fix | File:Line |
|---|---|---|
| 2026-06-15 | Per-task Kendall bounds (min_act, max_psr, max_pose) | config.py:977-979 |
| 2026-06-20 | HP_PREC_CAP (head-pose precision cap) | losses.py:1707-1713 |
| 2026-06-20 | KENDALL_FIXED_WEIGHTS path | losses.py:1666-1689 |
| 2026-07-02 | Double-ramp fix (removed precision-side activity ramp) | losses.py:1734-1743 |

---

## 10. HP_PREC_CAP (Head-Pose Precision Cap)

**File:** `src/training/losses.py`
**Lines:** 1707-1713

### Mechanism

```python
if bool(getattr(C, 'KENDALL_HP_PREC_CAP', True)):
    lv_hp = torch.maximum(lv_hp, lv_det.detach())
```

This ensures head-pose precision (exp(-lv_hp)) can never exceed detection precision (exp(-lv_det)).

### Rationale (Opus v8 section 1.1)

Without HP_PREC_CAP: head_pose (loss ~0.01) gets Kendall-optimal precision ~54.6x while detection (loss ~0.5) gets ~1.4x. The shared backbone is optimized for head_pose, losing object-discriminative features needed by detection.

With HP_PREC_CAP: lv_hp >= lv_det, so exp(-lv_hp) <= exp(-lv_det). Detection always drives backbone at least as strongly as head pose.

### Example Dynamics

| Metric | Without HP_PREC_CAP | With HP_PREC_CAP |
|---|---|---|
| det loss | 1.05 | 1.38 |
| head pose precision | 54.6x (very aggressive) | max 1.4x (capped) |
| backbone gradient share | ~2% head pose dominates | ~70% detection healthy |

The `.detach()` on lv_det ensures detection's log_var parameter is not affected by the comparison operation.

---

## 11. Combined Metric Weights

**File:** `src/training/train.py`
**Lines:** 168-171

```python
_W_DET  = 0.30
_W_ACT  = 0.35
_W_POSE = 0.15
_W_PSR  = 0.20
```

These weights are used to compute the combined validation metric (used for best-model selection):

```python
combined = (_W_DET * det_mAP50 +
            _W_ACT * act_macro_f1 +
            _W_POSE * (1.0 - normalized_pose_MAE) +
            _W_PSR * psr_f1)
```

The combined metric is used for:
- Best checkpoint selection (highest combined = best.pth)
- Early stopping decisions
- Stage-gate transitions in stage_manager.py

### Weight Rationale

| Task | Weight | Rationale |
|---|---|---|
| Activity | 0.35 | Highest weight - flagship metric for paper, hardest task (75 classes) |
| Detection | 0.30 | Core task, drives backbone features |
| PSR | 0.20 | Novel task (state change detection), secondary paper claim |
| Pose | 0.15 | Head pose is well-conditioned (regression, MSE~0.08), needs less weight |

---

## 12. Training Loop (train.py)

**File:** `src/training/train.py`

### Main Entry Point (main function)

**Lines:** approximately 1000-2000 for main(). Key responsibilities:

1. Parse args, seed everything
2. Build model, criterion, optimizer, scheduler
3. Build training + validation loaders
4. Load checkpoint (resume, reinit-heads, crash recovery)
5. Run epoch loop with train_one_epoch + validation + checkpoint saving

### train_one_epoch() (line 987+)

**Signature:** lines 987-1004

```
train_one_epoch(model, criterion, loader, optimizer, scaler, device, epoch, 
                 ckpt_dir, accum_steps, ema, seq_loader, resume_batch, best_metric,
                 val_ds, val_every_n_steps, distill_loss_fn)
```

#### Per-Batch Loop (starting at line 1163)

**For each batch:**

1. **Clamp Kendall log_vars** (line 1169): `_clamp_kendall_log_vars(criterion)` ensures all 4 log_var params are within allowed bounds before forward pass.

2. **Data integrity check** (lines 1192-1208): Checks images for NaN/Inf. Skips batch if corrupted.

3. **Sequence batch alternation** (lines 1210-1447): Every PSR_SEQ_EVERY_N_BATCHES steps, runs a PSR-only sequence batch:
   - PSR-only forward (other heads zeroed in criterion)
   - Loss is PSR_SEQ_LOSS_SCALE * PSR_loss
   - Gradient snapshot/restore for backbone/FPN (when DETACH_PSR_FPN=False)
   - Criterion flags saved and restored

4. **Standard training batch** (lines 1449+):
   - Move targets to device
   - Forward pass with amp.autocast
   - Mixup/CutMix (if enabled, gated by epoch)
   - Criterion set_epoch + forward -> loss + loss_dict
   - Distillation loss (if enabled)
   - loss / accum_steps -> backward

5. **Optimizer step** (every GRAD_ACCUM_STEPS batches or end of loader):
   - Per-head gradient clip for activity (ACTIVITY_HEAD_GRAD_CLIP)
   - Gradient centralization (ACTIVITY_GRAD_CENTRALIZATION, default off)
   - Global gradient clip (GRAD_CLIP_NORM)
   - PSR output head warmup multiplier (if reinit active)
   - scaler.step(optimizer)
   - scaler.update() (only on non-seq steps)
   - EMA update

6. **Logging every 10/50/200 steps**: GPU memory, heartbeat, grad norms, Kendall values

### Validation Loop (after train_one_epoch)

At epoch end (and at step boundary when VAL_EVERY_N_STEPS > 0):

1. Flush memory (`_flush_before_val`)
2. Set IN_EVALUATION_PHASE=True
3. If USE_SUBPROCESS_EVAL: launch subprocess evaluation on GPU 1
4. Else: run evaluate_all(model, criterion, val_loader, device, ...)
5. Compute combined metric
6. Save best checkpoint if improved
7. Save epoch checkpoint
8. Save stage heartbeat

### Sequence Batch Alternation Detail

Every `PSR_SEQ_EVERY_N_BATCHES` = 4 steps (config.py:1078):

1. Get a contiguous T=8 frame batch from seq_loader (model.py:1063)
2. Model forward on [B, T, 3, H, W] tensor
3. PSR-only criterion (all other heads zeroed)
4. Loss scaled by PSR_SEQ_LOSS_SCALE=1.5
5. Backward with gradient snapshot/restore for backbone/FPN
6. Criterion flags restored
7. Optimizer step (every accum_steps)

This design: with GRAD_ACCUM=8 and seq_every=4, each accumulation window of 8 standard batches contains exactly 2 seq batches. Each seq batch contributes only PSR gradient to the optimizer.

### Crash Recovery System

**Lines:** 752-958, 812-958

Modular, thread-safe, with disk-space check:

- _cr_set_state() updates module-level globals
- _save_crash_recovery(tag) saves in background thread (30s timeout)
- _checkpoint_has_nan() prevents saving NaN-weighed checkpoints
- _cuda_is_healthy() lightweight probe (no synchronize)
- Signal handlers capture SIGSEGV, SIGABRT, SIGBUS, SIGFPE, SIGTERM, SIGINT, SIGHUP

### Scheduler

OneCycleLR with warmup: WARMUP_EPOCHS=2 (config.py:583). Peak factor ONE_CYCLE_PEAK_FACTOR='auto' (config.py:1164), resolved as EFFECTIVE_BATCH/32 for Goyal linear scaling.

---

## 13. Evaluation Pipeline (evaluate.py)

**File:** `/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/evaluation/evaluate.py`

### evaluate_all() (line 3340+)

**Signature:** lines 3340-3344

```
evaluate_all(model, criterion, loader, device, max_batches, save_dir,
              use_flip_tta, use_crop_tta, epoch, predictions_path)
```

#### Per-Batch Loop (starting at line 3456)

For each batch:

1. **CUDA health + memory snapshot** (lines 3460-3464)
2. **Image prep** (line 3466): `_prepare_images(images, device)`
3. **PSR cache management** (lines 3468-3483): Detect recording boundaries and reset PSR causal cache to prevent cross-recording contamination
4. **Targets to device** (lines 3485-3501)
5. **Forward pass** (line 3515): `model(images, video_ids=batch_recording_ids, clip_rgb=clip_rgb)`
6. **TTA branches** (lines 3517-3545): Optional horizontal flip TTA and 5-crop TTA, averaged into outputs
7. **Loss computation** (line 3555): `loss, _loss_dict = (None, {}) if criterion is None else criterion(outputs, targets)`
8. **Activity collection** (lines 3569-3624): Act logits, predictions, clip-level IDs with act_valid mask filtering
9. **Head pose collection** (lines 3626-3637): Guard against None if model.train_pose=False
10. **PSR collection** (lines 3639-3654): Logits + labels + recording IDs + frame numbers for per-recording decode
11. **Detection** (lines 3656-3731):
    - Cached anchors
    - Sigmoid scores, score threshold filtering
    - Top-k per image cap (max 300)
    - Decode boxes, NMS per class
    - GT boxes/labels collected
    - Detection probe every batch (first 5 only)
12. **Crash checkpoint every 5 batches** (line 3737)

### PSR Transition Scoring (lines 320-421)

When USE_PSR_TRANSITION=True, uses MonotonicDecoder from psr_transition.py:

1. Group predictions by recording ID with temporal sorting (`_group_psr_by_recording`, lines 324-376)
2. Decode to monotone states (`decode_and_score_psr`, lines 379-421)
3. F1 on transition events within +/-tol_frames (bi-directional greedy match)
4. PSR-POS: ordered pair fraction
5. PSR-Edit: Damerau-Levenshtein on state-change event strings

### Detection mAP (lines 158-275)

`compute_detection_map()`:

1. Decode boxes from anchor deltas
2. Sigmoid class scores
3. Score threshold filtering (DET_EVAL_SCORE_THRESH=0.001)
4. Per-class NMS (IoU threshold 0.5)
5. COCO interpolation mode for AP
6. Returns per-class AP + mean mAP

### Detection Probe (lines 68-152)

`probe_detection_batch()`: drop-in diagnostic that logs per-batch score distributions, IoU statistics, and collapse verdicts. Self-throttling (first 5 batches only).

### Metrics Dictionary (after all batches)

The final metrics dict includes:

| Task | Metrics |
|---|---|
| Detection | det_mAP50, det_mAP_50_95, det_mAP50_pc, per-class AP |
| Activity | act_top1, act_top5, act_macro_f1, act_clip_accuracy, act_frame_accuracy |
| Head Pose | forward_angular_MAE_deg, up_angular_MAE_deg, position_MAE_mm, head_pose_MAE |
| PSR | psr_f1 (transition), psr_pos, psr_edit, psr_step_acc, psr_comp_acc |
| Efficiency | eff_fps, pipeline_params_m, pipeline_gflops |

---

## 14. Subprocess Evaluation (subprocess_eval.py)

**File:** `/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/evaluation/subprocess_eval.py`
**Lines:** 1-314

### Purpose

SIGKILL-safe validation on isolated CUDA context. When USE_SUBPROCESS_EVAL=True (config.py:1212), training forks a `spawn` child on the idle GPU (RTX 3060, indexed as GPU 1 after CUDA reorder) that loads the latest checkpoint and runs evaluate_all.

The parent can SIGKILL the child on timeout (SUBPROCESS_EVAL_TIMEOUT=900s, config.py:1213) without corrupting the training CUDA context.

### Architecture

```
run_val_subprocess(ckpt_path, out_path, overrides, timeout, predictions_path)
  -> _CTX.Process(target=_val_worker, ...)
     -> _val_worker():
        -> os.environ['CUDA_VISIBLE_DEVICES'] = '1'
        -> torch.load(ckpt_path)
        -> build POPWMultiTaskModel
        -> load state_dict (strict=False)
        -> build val DataLoader (num_workers=0)
        -> evaluate_all(model, criterion=None, loader, 'cuda', max_batches, epoch)
        -> json.dump(clean_metrics, out_path)
```

### Key Details

- Uses `spawn` context (line 35: `mp.get_context('spawn')`) for clean CUDA isolation
- Criterion=None for inference-only eval (no loss needed)
- All non-serializable values converted to float/str
- Timeout monitoring with 15s check intervals, mid-timeout warning log
- Returns {} on timeout/error for graceful degradation

### Crash Handling

The eval loop (evaluate.py line 3737) saves crash checkpoints every 5 batches to `eval_crash_recovery.pth`. If CUDA is unhealthy (OOM), it saves to CPU-only state.

---

## 15. TTA + Soft-NMS (eval_tta.py)

**File:** `/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/evaluation/eval_tta.py`
**Lines:** 1-605

### Architecture

Test-Time Augmentation with multi-scale + horizontal flip + Soft-NMS merging.

Scales: [0.8, 1.0, 1.2] (line 56)
Flips: [False, True] (line 57)
Total augmentations: 3 x 2 = 6 per image
Soft-NMS sigma: 0.5 (line 472)

### Pipeline

```python
run_tta_eval(ckpt_path, batch_size, max_batches):
    1. Load model and val loader
    2. For each batch:
       a. Save GT boxes/labels
       b. For each scale x flip:
          - Resize image, flip if needed
          - Forward pass
          - Decode boxes with Soft-NMS (per class)
          - Rescale boxes to original coordinates
       c. Merge all TTA predictions:
          - Concatenate all 6 per-image predictions
          - Apply Soft-NMS per class
          - Cap at max_per_image
       d. Accumulate merged predictions
    3. compute_det_metrics_extended on merged predictions
```

### Soft-NMS (soft_nms.py)

Implements the Gaussian Soft-NMS algorithm (Bodla et al., 2017):
- Instead of hard-NMS (remove overlapping boxes), decay their scores by:
  score_i = score_i * exp(-IoU^2 / sigma)
- Effectively Gaussian re-weighting: boxes that strongly overlap a detected box get their scores suppressed, not eliminated.
- Prevents hard cutoffs that miss genuinely overlapping objects.

### Expected Gain

Expected +0.02-0.07 mAP over single-pass inference (Opus Q50). ~2-3 hours on one GPU.

---

## 16. eval_post_reinit.py

**File:** `/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/evaluation/eval_post_reinit.py`
**Lines:** 1-147

### Purpose

Post-reinit evaluation: load a crash recovery checkpoint, reinitialize the 3 dead heads (detection, activity, PSR), and run evaluate_all. Used to verify that backbone features are alive even when heads have collapsed.

### Pipeline

1. Load checkpoint from file (default: crash_recovery.pth)
2. Build model with same constructor args as training
3. Load state_dict (strict=False) to tolerate missing/unexpected keys
4. Check for NaN/Inf in loaded params
5. Call `_reinit_dead_heads(model)` from train.py (re-inits det/act/psr heads)
6. Build criterion (MultiTaskLoss with correct num_classes)
7. Run evaluate_all with capped max_batches
8. Save sanitized metrics to JSON

### Key Features

- Env-overridable: EVAL_CKPT, EVAL_SPLIT, MAX_BATCHES, EVAL_BS, RUN_NAME
- EVAL_SKIP_REINIT=1 to evaluate trained checkpoint WITHOUT reinit
- All NaN/Inf values sanitized to None before JSON serialization
- Explicit `_print_single_run_results` for human-readable output

### Head Reinitialization

The `_reinit_dead_heads` function in train.py re-initializes:
1. Detection head (cls_score, reg_pred weights + biases)
2. Activity head (simple_classifier or full TCN+ViT classifier)
3. PSR head (all per-component output heads)

Backbone, FPN, pose head, head pose head, and FiLM modules are preserved.

---

## 17. F1-F22b Complete Training Fixes

Organized by Fable consultation round, with file:line references.

### F1-RF4 Consult (2026-07-02)

| Fix | Description | Config Key/File:Line |
|---|---|---|
| F1 | Seq batch gradient wipe fix: snapshot/restore backbone/FPN grads | train.py:1305-1339 |
| F2 | Kendall log_var VALUE logging cadence (was biggest observability gap) | config.py:62-66 |
| F3 | PSR structurally zero flag: skip per-frame +lv_psr in Kendall | losses.py:1460-1469 |
| F4 | OneCycleLR peak factor 'auto' (EFFECTIVE_BATCH/32) | config.py:1163-1164 |
| F5 | Activity gradient centralization disabled (was collapse-era hack) | config.py:928-930 |
| F6 | AMP dtype bf16 (no GradScaler, PSR spikes representable) | config.py:618-626 |
| F7 | PSR_SEQ_EVERY_N_BATCHES 2 -> 4 (50% seq batches was too high) | config.py:1078 |
| F8 | Focal alpha raised 0.25->0.50 (asymmetric gamma already handles neg) | config.py:696-704 |
| F9 | ACT_RAMP_EPOCHS 5 -> 3 (gradient path fixed, no need for long ramp) | config.py:833 |
| F10 | ACTIVITY_HEAD_GRAD_CLIP 1.0 -> 5.0 (was 5x tighter than global) | config.py:921 |
| F11 | GATE_EVAL_MAX_BATCHES 200 -> 250 (full val coverage) | config.py:1205 |

### F12-F21 Consult Round 5 (2026-07-03)

| Fix | Description | File:Line |
|---|---|---|
| F12 | Empty list accumulators for psr_preds_logits/psr_labels | evaluate.py:3446 |
| F13 | act_clip_ids / act_clip_frame_nums filtered by valid mask | evaluate.py:3593-3624 |
| F14 | nan-replacement for act_clip_id generation | evaluate.py:3612-3614 |
| F15 | KENDALL_FIXED_WEIGHTS env-overridable (ablation without code edit) | config.py:95 |
| F16 | _s() helper accepts int (Anomaly 2 root cause fix) | train.py:5035 |
| F17 | det_loss > 0.1 check before counting alive | train.py (unnumbered) |
| F18 | Double-ramp fix: removed precision-side activity ramp | losses.py:1734-1743 |
| F19 | Consistency rewards removed during non-seq batches | losses.py (unnumbered) |
| F20 | fixed-weights ramp matched to standard Kendall ramp | losses.py (unnumbered) |
| F21 | ONE_CYCLE_PEAK_FACTOR default 'auto' | config.py:1164 |

### F22/F22b Consult Round 6 (2026-07-03)

| Fix | Description | File:Line |
|---|---|---|
| F22 | _group_psr_by_recording + per-recording sort for transition F1 | evaluate.py:324-376 |
| F22b | MonotonicDecoder explicit dim handling (squeeze bug) | psr_transition.py:117-131 |

---

## 18. 5-Bug-Fix History

Documented in AAIML 119-progress-log.md, discovered during D3 full eval run (2026-07-04 19:43-19:46):

### Bug 1: criterion=None crash (evaluate.py:3365)

**Symptom:** `evaluate.py:3365` called `criterion.to(device_obj)` but criterion=None in inference-only mode.

**Fix:** Added `if criterion is not None:` guard before `criterion.to(device_obj)`.

### Bug 2: tuple unpacking crash (evaluate.py:3454)

**Symptom:** `evaluate.py:3454` unpacked `(images, targets)` from loader, but loader without `collate_fn` returned dicts.

**Fix:** Added explicit `collate_fn=collate_fn` in subprocess_eval.py loader construction.

### Bug 3: max_batches=None (evaluate.py:3454)

**Symptom:** `evaluate.py:3454` checked `if max_batches > 0` but `max_batches` could be `None`.

**Fix:** Changed to `if max_batches is not None and max_batches > 0`.

### Bug 4: criterion=None in loss computation (evaluate.py:3553)

**Symptom:** `evaluate.py:3553` called `criterion(outputs, targets)` with criterion=None.

**Fix:** Changed to `loss, _loss_dict = (None, {}) if criterion is None else criterion(outputs, targets)`.

### Bug 5: subprocess_eval.py IndustRealDataset kwargs

**Symptom:** `subprocess_eval.py` called `IndustRealDataset(root=val_root, cache_max_images=...)` but class doesn't accept `root` or `cache_max_images` kwargs.

**Fix:** Changed to `IndustRealDataset(split='val', img_size=(C.IMG_HEIGHT, C.IMG_WIDTH))`.

These 5 bugs were all found and fixed within 3 minutes (19:43-19:46 JST on 2026-07-04).

---

## 19. 10 NaN to 0.0 Fix Locations

All locations where NaN/inf tensors are explicitly replaced with 0.0 (or a small fallback like 1e-4) to prevent NaN cascade:

| # | Location | File | Line(s) | Replacement |
|---|---|---|---|---|
| 1 | Backbone feature sanitizer | model.py | 1953-1958 | `torch.where(torch.isfinite(x), x, torch.zeros_like(x))` |
| 2 | FPN pyramid sanitizer | model.py | 1963-1969 | `torch.where(torch.isfinite(pyramid[k]), pyramid[k], torch.zeros_like(pyramid[k]))` |
| 3 | FeatureBank NaN guard (per-frame) | model.py | 1196-1207 | Pad with last valid or zeros |
| 4 | FeatureBank batch mode | model.py | 1169-1170 | Return zeros_like |
| 5 | proj_feat NaN guard | model.py | 2187-2188 | `torch.zeros_like(proj_feat)` |
| 6 | Per-task NaN guard before Kendall assembly | losses.py | 1273-1296 | `torch.where(torch.isfinite(loss), loss, torch.tensor(1e-4))` |
| 7 | Activity loss NaN guard | losses.py | 1371-1379 | `torch.where(torch.isfinite(loss_act), loss_act, zero)` |
| 8 | PSR NaN guard before smooth_cap | losses.py | 1539-1554 | `torch.where(torch.isfinite(loss_psr), loss_psr, torch.tensor(1e-4))` |
| 9 | Final Kendall NaN guard (_safe lambda) | losses.py | 1602-1617 | `torch.where(torch.isfinite(l), ..., torch.tensor(1e-4))` |
| 10 | PSR sensitivity penalty NaN guard | losses.py | 1503-1504 | `torch.where(torch.isfinite(_sens), _sens, torch.tensor(0.0))` |

Additionally, the NaN guard in train.py's batch loop (lines 1192-1208) skips the entire batch when input images contain NaN/inf, preventing the corrupted data from ever reaching the model.

---

## 20. det_mAP50 epoch=-1 Fix

**File:** `/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/evaluation/evaluate.py`
**Line:** 3342

The `evaluate_all()` function signature changed: `epoch` parameter now defaults to -1 instead of 0:

```python
def evaluate_all(
    ...
    epoch: int = -1,  # [FIX 2026-07-05] Default -1 = post-hoc eval, computes everything
    ...
)
```

### Why epoch=-1

Validation during training passes the real epoch (0, 1, 2, ...). Post-hoc evaluation (via subprocess_eval.py, eval_tta.py, eval_post_reinit.py) passes -1.

The epoch value gates expensive metrics. With epoch=-1:
- All metrics are computed (no gating)
- `C._CURRENT_EPOCH = -1` means efficiency gate `SKIP_EFFICIENCY_METRICS=False` computes everything
- `DET_METRICS_EVERY_N` checks: epoch=-1 always triggers full eval

Previously, epoch=0 from post-hoc calls would gate out detection mAP if DET_METRICS_EVERY_N > 1 (det_mAP50 only computed every N epochs), causing silent zero det_mAP50 in results.

### Upstream Impact

- subprocess_eval.py passes `epoch=int(overrides.get('epoch', 0))` (line 137)
- train.py passes the real epoch (0-indexed) during validation call
- eval_post_reinit.py doesn't pass epoch (gets default -1)
- eval_tta.py doesn't pass epoch (gets default -1)

---

## 21. Per-File Line Reference Index

### src/config.py
- Line 27: BENCHMARK_MODE
- Line 34: TRAIN_FRAME_STRIDE = 3
- Line 54: FREEZE_BODY_POSE_BRANCH
- Line 62-66: LOG_KENDALL_GRAD_EVERY
- Line 89-96: KENDALL_HP_PREC_CAP, KENDALL_FIXED_WEIGHTS
- Line 123-125: BACKBONE = 'convnext_tiny'
- Line 128-133: CONVNEXT_CHANNELS
- Line 167: USE_BACKBONE_CHECKPOINT
- Line 200-227: NUM_DET_CLASSES=24, DET_CLASS_NAMES
- Line 243-244: NUM_CLASSES_ACT=75
- Line 315-316: ACT_CLASS_GROUPING='hybrid', ACT_HYBRID_THRESHOLD=100
- Line 459-472: NUM_KEYPOINTS=17, NUM_HEAD_POSE_DOF=9
- Line 478-484: NUM_PSR_STEPS=36, NUM_PSR_COMPONENTS=11
- Line 499: ANCHOR_SIZES=(96, 160, 256, 384, 512)
- Line 500-507: DET_POS_IOU_THRESH=0.4, DET_POS_IOU_TOP_K=9
- Line 569: BATCH_SIZE=6 (RTX 5060 Ti)
- Line 570: GRAD_ACCUM_STEPS=8
- Line 577: EPOCHS=100, BASE_LR=5e-4
- Line 597: VAL_EVERY=1
- Line 600: EVAL_MAX_BATCHES=250
- Line 604: NUM_WORKERS=0
- Line 608: RAM_CACHE_MAX_IMAGES=8000
- Line 618-626: AMP_DTYPE='bf16'
- Line 630-631: USE_EMA=True, EMA_DECAY=0.995
- Line 672: DET_EVAL_SCORE_THRESH=0.001
- Line 696-704: FOCAL_ALPHA=0.50, FOCAL_GAMMA=2.0
- Line 749: GIOU_WEIGHT=2.0
- Line 756-761: DET_OHEM_ENABLED, RATIO=2.0, MIN_NEG=32
- Line 769-775: DET_ASYMMETRIC_GAMMA, GAMMA_POS=0.0, GAMMA_NEG=1.5
- Line 796: CB_LABEL_SMOOTHING=0.1
- Line 833: ACT_RAMP_EPOCHS=3
- Line 834: ACTIVITY_LOSS_CAP=80.0
- Line 843: HEAD_POSE_POS_SCALE=100.0
- Line 850: PSR_WEIGHT=10.0
- Line 862-863: SOFT_ARGMAX_TEMPERATURE=0.07, TEMP_TRAIN=1.0
- Line 873-874: DET_EMPTY_SAMPLE=2048, DET_EMPTY_BG_SCALE=0.05
- Line 910: DET_GT_FRAME_FRACTION=0.40
- Line 921: ACTIVITY_HEAD_GRAD_CLIP=5.0
- Line 928-930: ACTIVITY_GRAD_CENTRALIZATION=False
- Line 950: ACTIVITY_HEAD_SIMPLE=True
- Line 971: ACTIVITY_GRAD_BLEND_RATIO=1.00
- Line 977-979: Kendall log_var bounds
- Line 985: PSR_WARMUP_STEPS=500
- Line 1002: DETACH_REG_FPN=True
- Line 1008: DETACH_PSR_FPN=True
- Line 1055: PSR_COMP_WEIGHTS
- Line 1062-1063: USE_PSR_SEQUENCE_MODE=True, PSR_SEQUENCE_LENGTH=8
- Line 1078: PSR_SEQ_EVERY_N_BATCHES=4
- Line 1079: PSR_SEQ_LOSS_SCALE=1.5
- Line 1085: USE_PSR_TRANSITION=True
- Line 1100: USE_GEO_HEAD_POSE=True
- Line 1131: PSR_SENSITIVITY_WEIGHT=0.50
- Line 1164: ONE_CYCLE_PEAK_FACTOR='auto'
- Line 1212: USE_SUBPROCESS_EVAL (env-overridable)
- Line 1200: DET_METRICS_EVERY_N=3

### src/models/model.py
- Line 89-140: SoftArgmax
- Line 146-169: WingLoss
- Line 175-262: ConvNeXtBackbone
- Line 265-308: set_backbone_stage_requires_grad
- Line 311-370: ResNet50Backbone
- Line 373-384: build_backbone factory
- Line 390-440: FPN
- Line 446-494: AnchorGenerator
- Line 500-567: DetectionHead
- Line 573-620: PoseHead
- Line 626-716: PoseFiLMModule
- Line 722-792: HeadPoseFiLMModule
- Line 798-973: VideoMAEStream
- Line 992-1040: TemporalConvBlock
- Line 1043-1133: ViTTemporalBlock
- Line 1139-1256: FeatureBank (ring buffer T=16)
- Line 1262-1478: ActivityHead
- Line 1484-1533: HeadPoseHead
- Line 1539-1736: PSRHead (Causal Transformer)
- Line 1749-2200: POPWMultiTaskModel (main model)
- Line 1891-1913: _decode_boxes static method
- Line 1922-2200: forward method (full pipeline)
- Line 1952-1958: _sanitize NaN guard
- Line 1976-2037: Body pose pseudo-keypoints generation
- Line 2073-2166: Activity, PSR, head pose forward branching
- Line 2175-2181: Activity gradient blending

### src/training/losses.py
- Line 74-200: FocalLoss (detection)
- Line 416-427: GIoULoss
- Line 461-487: PoseLoss (Wing)
- Line 494-600: LDAMLoss
- Line 650-790: ClassBalancedFocalLoss
- Line 793-826: PSRFocalLoss
- Line 829-900: binary_focal_loss
- Line 980-1790: MultiTaskLoss (main loss class)
- Line 1000-1011: __init__ signature
- Line 1014-1035: 4 log_var parameters
- Line 1037-1081: Sub-loss functions
- Line 1170-1187: set_epoch
- Line 1189-1656: forward (loss computation)
- Line 1273-1296: NaN guard on per-task losses
- Line 1298-1311: smooth cap
- Line 1385-1389: Activity warmup ramp
- Line 1417-1555: PSR loss (transition, focal, sensitivity, temporal smooth)
- Line 1559-1579: Head pose loss + cap
- Line 1602-1617: Final _safe NaN guard
- Line 1658-1790: Kendall weighting
- Line 1707-1713: HP_PREC_CAP
- Line 1714-1718: Precision computation
- Line 1720-1743: Stage-aware precision zeroing
- Line 1744-1790: Task-specific ramp logic

### src/training/train.py
- Line 1-30: CUDA environment setup
- Line 80-158: Imports
- Line 167-171: Combined metric weights
- Line 188: IN_EVALUATION_PHASE flag
- Line 220-263: _write_stage_heartbeat
- Line 284-308: _atomic_save
- Line 311-328: seed_everything
- Line 331-362: _prepare_images
- Line 373-419: _build_loader
- Line 465-514: _shutdown_loader_workers
- Line 517-538: _flush_before_val
- Line 653-676: get_stage
- Line 679-749: _set_stage_requires_grad
- Line 752-958: Crash recovery system
- Line 987-1447: train_one_epoch
- Line 1163-1447: Per-batch loop
- Line 1192-1208: Data integrity check
- Line 1210-1447: Sequence batch alternation
- Line 1449+: Standard training batch
- Line 1360-1441: Optimizer step
- Line 1405-1408: Global gradient clipping
- Line 1381-1404: Per-head activity gradient clip + centralization
- Line 1450-1790: Main training loop

### src/evaluation/evaluate.py
- Line 68-152: probe_detection_batch
- Line 158-275: compute_detection_map
- Line 288-314: compute_activity_accuracy
- Line 324-376: _group_psr_by_recording
- Line 379-421: decode_and_score_psr
- Line 424-448: _event_f1
- Line 451-456: _ordered_pair_fraction
- Line 458-475: _psr_edit_score
- Line 482-511: compute_psr_accuracy
- Line 518-587: EvaluationMetrics class
- Line 3340-3738: evaluate_all
- Line 3456-3738: Per-batch loop
- Line 3569-3624: Activity collection
- Line 3626-3637: Head pose collection
- Line 3639-3654: PSR collection
- Line 3656-3731: Detection processing

### src/models/psr_transition.py
- Line 31-70: build_transition_targets
- Line 76-163: MonotonicDecoder
- Line 117-131: Dim handling fix (F22b)
- Line 142-153: Procedure-order constraint
- Line 169-257: PSRTransitionPredictor (full module)
- Line 259-318: compute_loss

### src/evaluation/subprocess_eval.py
- Line 38-153: _val_worker
- Line 155-236: run_val_subprocess
- Line 239-315: main CLI

### src/evaluation/eval_tta.py
- Line 56-57: TTA config
- Line 63-97: _build_model
- Line 119-138: TTA resize/flip
- Line 141-219: _decode_batch_predictions (Soft-NMS enabled)
- Line 222-261: _rescale_boxes_to_original
- Line 264-357: _merge_tta_predictions
- Line 360-524: run_tta_eval

### src/evaluation/eval_post_reinit.py
- Line 1-147: Full pipeline

### analyses/consult_2026_06_10/AAIML/119-progress-log.md
- Line 47-52: 5-bug-fix history (D3 run)
- Line 58-67: Q43 result (GATE G4 STRONG_PASS)
- Line 105-104: Epoch metrics (det_mAP50=0.3584, etc.)
- Line 185-203: Epoch 17 breakthrough table

---

## Architecture Data Flow Diagram (Text)

```
Input [B, 3, 1280, 720]
    |
    v
ConvNeXt-Tiny Backbone (28.6M)
    |-- C2 [B, 96, H/4, W/4]
    |-- C3 [B, 192, H/8, W/8]
    |-- C4 [B, 384, H/16, W/16]
    |-- C5 [B, 768, H/32, W/32]
    |
    v
FPN Neck (4.5M)
    |-- P3 [B, 256, H/8] --> Pose Head (1.6M) --> keypoints [B, 17, 2]
    |-- P3-P7 [B, 256]   --> Detection Head (5.3M) --> cls [B, N, 24] + reg [B, N, 4]
    |-- P4 [B, 256, H/16] (-> GAP for activity)
    |-- P5 [B, 256, H/32] (-> GAP for PSR)
    |
C5 --> PoseFiLM (keypoints modulate)
    |-- C5_mod [B, 768, H/32] --> HeadPoseFiLM (head_pose modulate)
            |-- C5_mod2 [B, 768, H/32]
            |
            |-- Head Pose Head (0.8M) --> [B, 9] (9-DoF)
            |
            |-- GAP + concat(p4) + det_conf
            |       |-- proj_feat [B, 512]
            |       |-- FeatureBank T=16
            |       |-- Activity Head (0.7M) --> [B, 75]
            |
            v-- PSR Head (3.1M): Multi-scale GAP(P3+P4+P5)
                    |-- per_frame_mlp -> [B, 256]
                    |-- Causal Transformer (3 layers)
                    |-- Per-component heads (11 x MLP)
                    |-- psr_logits [B, 11]
```

## Parameter Count Summary

| Component | Params (M) | GFLOPs | 
|---|---|---|
| ConvNeXt-Tiny Backbone | 28.6 | 71 |
| FPN Neck | 4.5 | 5.5 |
| Detection Head | 5.3 | 98 |
| Body Pose Head | 1.6 | 47 |
| PoseFiLM + HeadPoseFiLM | 2.2 | 13 |
| Head Pose Head | 0.8 | 3 |
| Activity Head (simple) | 0.7 | 1.2 |
| PSR Head | 3.1 | 6 |
| EMA + Scheduler + Misc | 0.07 | — |
| **Total** | **46.47** | **245.3** |

Note: GFLOPs measured at 1280x720 input resolution. The detection head dominates GFLOPs because it processes 5 FPN levels at full resolution (P3-P7), each with 9 anchors per location and 4 conv layers per subnet plus output convolutions.

---

## 22. Complete Inference Code Path

Tracing the full forward pass for a single 1280x720 RGB frame:

### Step 1: Image Preparation (train.py:331-362)

```python
images = _prepare_images(images, device)
# uint8 -> float32 / 255.0
# Normalize by ImageNet mean/std
# If 5-D [B,T,C,H,W], flatten to [B*T, C, H, W]
```

### Step 2: Backbone (model.py:1948)

```python
# model.py:1948
c2, c3, c4, c5 = self.backbone(images)
# C2 [B, 96, 320, 180]  stride 4
# C3 [B, 192, 160, 90]  stride 8
# C4 [B, 384, 80, 45]   stride 16
# C5 [B, 768, 40, 23]   stride 32
```

NaN guard applied: _sanitize(c3), _sanitize(c4), _sanitize(c5) (model.py:1958).

### Step 3: FPN (model.py:1959)

```python
pyramid = self.fpn(_c3, _c4, _c5)
# Returns {'p3', 'p4', 'p5', 'p6', 'p7'} all 256ch
```

Pyramid NaN guard (model.py:1963-1969).

### Step 4: Detection Head (model.py:1973-1974)

```python
cls_preds, reg_preds = self.detection_head(pyramid)
anchors = self.anchor_gen(pyramid)
# cls_preds: [B, ~173K, 24]
# reg_preds: [B, ~173K, 4]
# anchors: [~173K, 4]
```

### Step 5: Body Pose + PoseFiLM (model.py:1976-2046)

```python
# Pose head on P3
heatmaps, keypoints, pose_confidence = self.pose_head(pyramid['p3'])
# heatmaps: [B, 17, H/4, W/4]
# keypoints: [B, 17, 2] (spatial coords)
# confidence: [B, 17]

# If train_pose=False: generate pseudo-keypoints from detection boxes
# (model.py:1982-2037)

# PoseFiLM: keypoints -> gamma/beta -> modulate C5
c5_mod = self.pose_film(c5, keypoints.detach(), pose_confidence)
# c5_mod: [B, 768, H/32, W/32]
```

### Step 6: Head Pose + HeadPoseFiLM (model.py:2057-2168)

```python
head_pose = self.head_pose_head(c4, c5)
# head_pose: [B, 9] (forward[3] + position[3] + up[3])

# If use_headpose_film: second FiLM on c5_mod
c5_mod = self.headpose_film(c5_mod, head_pose.detach())
# c5_mod: [B, 768, H/32, W/32], double-modulated
```

The head_pose.detach() is CRITICAL: prevents HeadPoseFiLM gradients from flowing back through the head pose head.

### Step 7: Activity Head (model.py:2073-2188)

```python
# Gradient blending
c5_mod_blend = blend * c5_mod + (1 - blend) * c5_mod.detach()

# Joint feature construction
activity_proj = torch.cat([
    det_conf,                                          # [B, 24]
    GAP(c5_mod_blend).flatten(1),                     # [B, 768]
    GAP(pyramid['p4'].detach()).flatten(1)            # [B, 256]
], dim=1)  # [B, 1048]

# Projection
proj_feat = self.activity_head.proj_features(activity_proj)
# proj_feat: [B, 512]

# NaN guard
if not torch.isfinite(proj_feat).all():
    proj_feat = torch.zeros_like(proj_feat)

# Feature Bank + Activity Head
temporal_bank = self.feature_bank(proj_feat, video_ids=video_ids)
act_logits = self.activity_head(proj_feat, temporal_bank=temporal_bank)
# act_logits: [B, 75] (or grouped output dimension)
```

### Step 8: PSR Head (model.py:2073-2141)

Two paths: sequence mode (model.py:2077-2141) and per-frame mode (model.py:2143-2151).

For sequence mode:
```python
pyramid_seq = {k: v.reshape(B, T, C, H, W) for k, v in pyramid.items()}
# Per-frame: GAP(p3) + GAP(p4) + GAP(p5) -> per_frame_mlp -> [B, T, 256]
# Causal Transformer -> [B, T, 256]
# Per-component heads -> [B*T, 11]
# psr_logits: [B*T, 12] (11 + confidence)
```

For per-frame mode:
```python
psr_logits = self.psr_head(pyramid, video_ids=video_ids)
# [B, 12] (11 component logits + 1 confidence)
```

### Step 9: Output Dict Assembly

```python
outputs = {
    'cls_preds': cls_preds,       # [B, N, 24] detection logits
    'reg_preds': reg_preds,       # [B, N, 4] box deltas
    'anchors': anchors,           # [N, 4] anchor boxes
    'act_logits': act_logits,     # [B, num_groups] activity logits
    'psr_logits': psr_logits,     # [B, 12] or [B*T, 12] PSR logits
    'head_pose': head_pose,       # [B, 9] 9-DoF head pose
    'heatmaps': heatmaps,         # [B, 17, H/4, W/4] keypoint heatmaps
    'keypoints': keypoints,       # [B, 17, 2] body keypoints
    'pose_confidence': pose_confidence,  # [B, 17]
}
```

---

## 23. Optimizer and Scheduler Configuration

### Optimizer

**File:** `src/training/train.py` (main function)

- Type: AdamW
- Base LR: 5e-4 (config.py:578)
- Weight decay: 1e-3 (config.py:579)
- Parameter groups:
  - Majority of params: backbone + heads at lr=BASE_LR
  - VideoMAE stream (if enabled): separate group at VIDEOMAE_UNFREEZE_LR=1e-5 (config.py:155)
  - Head-pose head: independent group (maintained by train.py)

### Scheduler OneCycleLR

- Enabled: ONE_CYCLE_LR=True (config.py:1149)
- Warmup: 2 epochs (WARMUP_EPOCHS=2, config.py:583)
- Peak factor: 'auto' -> EFFECTIVE_BATCH/32 (config.py:1164)
  - At batch=6, accum=8: effective batch=48 -> factor=1.5
  - At batch=4, accum=4: effective batch=16 -> factor=0.5
- Cosine annealing after warmup to near-zero LR

### Hyperparameter Validation

| Parameter | Value | Standard Range | Assessment |
|---|---|---|---|
| BASE_LR | 5e-4 | 1e-4 to 1e-3 | Matches paper spec |
| WEIGHT_DECAY | 1e-3 | 1e-4 to 1e-2 | Standard AdamW (was 5e-2, fixed) |
| GRAD_CLIP_NORM | 5.0 | 1.0 to 10.0 | Standard multi-task (was 1.0, fixed) |
| WARMUP_EPOCHS | 2 | 1-5 | Matches paper |
| EMA_DECAY | 0.995 | 0.99-0.999 | Standard |
| BATCH_SIZE | 6 | GPU-dependent | Safe for RTX 5060 Ti 16GB |
| GRAD_ACCUM_STEPS | 8 | — | Effective batch = 48 |

### GRAD_CLIP_NORM Fix History

| Date | Value | Reason | Line |
|---|---|---|---|
| 2026-06-11 (legacy) | 1.0 | Original | config.py:594 |
| 2026-07-01 (agent audit) | 5.0 | 5-head multi-task combined gradient norm easily exceeds 5.0. At 1.0, every head's gradient was clipped 80-90% | config.py:597 |

---

## 24. Checkpoint System

### Checkpoint Files

| File | When Saved | Content |
|---|---|---|
| best.pth | When combined metric improves | Full state dict, optimizer, criterion, scheduler, EMA |
| latest.pth | Every epoch end | Full state dict, optimizer, criterion, scheduler, EMA |
| crash_recovery.pth | Every 50 batches, signal handlers, epoch start | model, optimizer, scaler, criterion, EMA (CPU-only tensors) |
| epoch_N.pth | Every epoch end (kept for all N) | Same as latest.pth |
| eval_crash_recovery.pth | Every 5 eval batches (in evaluate_all) | model state, batch index |

### Checkpoint Save Safety

The `_atomic_save` function (train.py:284-308):
1. Writes to .tmp file
2. Renames atomically on POSIX
3. Cleans up temp file on failure
4. Checks disk space before saving (warns if <1GB free)

The `_save_crash_recovery` function (train.py:812-958):
1. Runs in daemon thread with 30s timeout
2. Checks for NaN/Inf params before saving
3. CPU-fallback if CUDA is unhealthy
4. Double disk-space check at 2GB threshold
5. Never blocks the main training loop

### Checkpoint State Dict Keys

```python
save_dict = {
    'tag': tag,                    # event identifier
    'epoch': epoch,                # current epoch
    'model': model_state_dict,     # all model parameters
    'optimizer': optimizer_state,  # AdamW state
    'scaler': scaler_state,        # GradScaler state
    'scheduler': scheduler_state,  # OneCycleLR state
    'criterion': {                 # Kendall log_vars
        'log_var_det': ..., 'log_var_pose': ..., 
        'log_var_act': ..., 'log_var_psr': ...
    },
    'ema_shadow': {...},           # EMA shadow weights
    'best_metric': float,          # best combined metric
    'global_step': int,            # step counter
    'timestamp': time.time(),
}
```

---

## 25. Complete Loss Computation Flow

### Step-by-Step Loss Assembly (losses.py:1189-1656)

The MultiTaskLoss.forward() method assembles the total loss in this order:

1. **Detection** (lines 1215-1248): Focal + GIoU -> loss_det
   - reg_loss warmup ramp for reinit heads (lines 1222-1234)
   - GIoU weight = 2.0 applied (line 1235)
   - Negative-slope guard for GIoU < 0 (lines 1241-1246)

2. **NaN guard on individual losses** (lines 1250-1296): `torch.where(torch.isfinite(loss), loss, 1e-4)`
   - Separate guard for each of 5 task losses
   - Uses locals() instead of dir() to correctly capture loss_pose/loss_head_pose

3. **Smooth loss caps** (lines 1298-1311): `_smooth_cap(x, cap)`
   - Detection: DET_LOSS_CAP=50.0
   - Pose: POSE_LOSS_CAP=30.0

4. **Body Pose** (lines 1319-1336): Wing Loss on keypoints
   - POSE_LOSS_WEIGHT=5.0 applied
   - Zeroed when FREEZE_BODY_POSE_BRANCH=True

5. **Activity** (lines 1338-1415): CE + label_smooth(0.1)
   - Valid mask applied (-1 sentinel excluded)
   - LDAM path when USE_LDAM_DRW=True
   - Activity warmup ramp (stage-local epoch counter)
   - ACTIVITY_LOSS_CAP=80.0 smooth cap

6. **PSR** (lines 1417-1556): 
   - Transition objective (dim==3): Gaussian-smeared transition targets
   - Per-frame (dim==2): structurally zero or binary focal
   - Sensitivity penalty: -log(std(per-component logits))
   - Temporal smoothness loss on seq batches
   - PSR_LOSS_CAP=20.0

7. **Head Pose** (lines 1558-1579): Split MSE (position + direction)
   - L_hp weight = 5.0 applied
   - HP_CAP=30.0 smooth cap

8. **Final NaN guard** (lines 1594-1617): `_safe(l, z)` - replaces all non-finite with 1e-4

9. **Kendall weighting** (lines 1658-1790):
   - Two paths: FIXED_WEIGHTS or standard precision-weighted
   - HP_PREC_CAP: lv_hp >= lv_det (head pose precision never exceeds detection)
   - Stage-aware precision zeroing for frozen tasks
   - Fixed-weight path: det=1.0, hp=0.2, activity=0.8, psr=10.0

10. **Head pose within Kendall** (lines 1744-1790): 
    - Head pose is grouped under log_var_pose (shared with body pose)
    - Kendall total = sum(prec_t * loss_t + lv_t) for active tasks

### Unweighted Loss Accumulation (evaluate.py:3560-3567)

During validation, per-head unweighted losses are accumulated separately to avoid Kendall confounding:

```python
if lc == 1:
    _per_head_sums = {k: 0.0 for k in ('det', 'pose', 'head_pose', 'activity', 'psr')}
for _k in _per_head_sums:
    _per_head_sums[_k] += float(_loss_dict.get(_k, 0.0))
```

These are logged on the Val: line as vl_det, vl_hp, vl_act, vl_psr.

---

## 26. Per-Head Loss Scale Analysis

Understanding how each head's loss contributes to the total gradient:

| Head | Raw Loss Range | Multiplier | After Multiplier | Kendall Precision | Effective Contribution |
|---|---|---|---|---|---|
| Detection | 0.5-3.0 | GIoU x2 | 1.0-6.0 | exp(-lv_det) ~1.4 | 1.4-8.4 (backbone driver) |
| Head Pose | 0.01-0.08 | x5.0 (config) | 0.05-0.4 | exp(-lv_hp) ~1.0-1.4 (capped) | 0.05-0.56 (stabilizer) |
| Activity | 1.0-5.0 | x0.8 (weight) | 0.8-4.0 | exp(-lv_act) ~1.0-1.65 | 0.8-6.6 (major contributor) |
| PSR | 0.01-0.5 | x5.0 (config) | 0.05-2.5 | exp(-lv_psr) ~1.0 | 0.05-2.5 (varies) |

Note: The raw loss ranges are approximate from actual training runs. Kendall precision adjusts dynamically based on task noise. When detection is healthy, it dominates the backbone gradient (desired). When detection is collapsed (loss near zero), head pose would dominate without HP_PREC_CAP.

---

## 27. All Training Presets Registry

Defined in config.py, lines 1246-1500:

### RF Stage Presets (lines 1386-1500)

| Stage | Description | Data% | Epochs | Tasks Active | Key Overrides |
|---|---|---|---|---|---|
| rf1 | Detection bootstrap | 20% | 20 | det + head_pose | reinit_pi=0.05, spatial_aug=False, randaugment=False, detach_reg_fpn=False |
| rf2 | + Pose | 35% | 15 | det + head_pose | randaugment=True, spatial_aug=True |
| rf3 | + Activity | 35% | 15 | det + head_pose + activity | geo_head_pose=True, psr_sensitivity=0.0 |
| rf4 | + PSR (transition) | 50% | 20 | all 4 heads | use_psr_transition=True, psr_sensitivity=0.50 |
| rf5 | Consolidate | 50% | 10 | all heads | — |
| rf6 | Scale data | 65% | 10 | all heads | — |
| rf7 | Scale data | 65% | 10 | all heads | — |
| rf8 | Scale data | 80% | 10 | all heads | — |
| rf9 | Scale data | 90% | 10 | all heads | — |
| rf10 | Full data | 100% | 15 | all heads | — |

### Other Presets

| Preset | Description | Key Config |
|---|---|---|
| recovery | Joint recovery (FP32, no staging) | batch=1, accum=8, zero_det_conf=False |
| recovery_det_only | Detection bootstrap only | det + head_pose only, activity/PSR off |
| paper_run | Final paper preset | batch=2, accum=16, PSR transition, geo head pose |
| benchmark_full | Full benchmark | VideoMAE enabled, batch=1, accum=32 |
| benchmark_quick | Quick baseline | No temporal, no Hand-FiLM, batch=4 |

---

## 28. Dataset and DataLoader Architecture

### Dataset Class: IndustRealMultiTaskDataset

**File:** `src/data/industreal_dataset.py`

Key features:
- Loads from IndustReal dataset: recordings/ split across train/val/test
- Each recording: rgb frames + PSR_labels_raw.csv + AR_labels.csv + pose.csv + hands.csv
- Multi-task output dictionary with keys: images, detection (COCO-format boxes), activity, psr_labels, head_pose, hand_joints, keypoints, metadata

### Collate Functions

| Collate | Used By | Behavior |
|---|---|---|
| collate_fn | Standard batches | Groups per-frame samples, stacks images |
| collate_fn_sequences | PSR sequence batches | Returns [B, T, C, H, W] contiguous frame sequences |

### Sampler Strategy

Default: weighted sampler with ACT_SAMPLER_MODE='balanced' (config.py:791)
- Classes with >= ACT_SAMPLER_COUNT_FLOOR=15 frames get true class balance
- Sub-floor classes scaled by count
- DET_GT_FRAME_FRACTION=0.40: 40% of batch mass goes to GT-bearing frames

### Task-Aware Sampling

| Parameter | Value | Effect | Line |
|---|---|---|---|
| USE_TASK_AWARE_SAMPLING | True | Upweights GT-bearing frames | config.py:882 |
| TASK_AWARE_DET_BOOST | 2.0 | 2x weight for frames with GT boxes | config.py:883 |
| TASK_AWARE_PSR_BOOST | 1.5 | 1.5x weight for frames with PSR labels | config.py:884 |
| DET_GT_FRAME_FRACTION | 0.40 | Absolute per-batch GT frame fraction | config.py:910 |

### Data Integrity

- NUM_WORKERS=0 (config.py:604) eliminates DataLoader worker deadlocks
- RAM_CACHE_MAX_IMAGES=8000 (config.py:608) caches full dataset in RAM (~2.2 GB)
- DataLoader auto-fallback when /dev/shm space is low (train.py:422-462)
- Worker shutdown with timeout (train.py:465-514) prevents zombie processes

### Fix History

| Date | Fix | Line |
|---|---|---|
| 2026-06-30 | NUM_WORKERS 4->0 (eliminate deadlocks) | config.py:604 |
| 2026-06-30 | VAL_NUM_WORKERS 0->0 (match NUM_WORKERS) | config.py:574 |
| 2026-06-30 | RAM_CACHE 5000->8000 (full dataset) | config.py:608 |

---

## 29. Augmentation Pipeline

### Training Augmentations

| Augmentation | Probability/Epoch | Config | Line |
|---|---|---|---|
| RandAugment | On when USE_RANDAUGMENT=True | num_ops=2, magnitude=9 | config.py:1136 |
| Horizontal flip | USE_SPATIAL_AUG=True | 50% | config.py:36 |
| Random crop+resize | USE_SPATIAL_AUG=True | — | dataset.py |
| Random temporal stride | RANDOM_TEMPORAL_STRIDE=True | stride in {1,2,3} | config.py:1143 |
| Mixup (activity) | DISABLED (config.py:642) | alpha=0.4 | config.py:643 |
| CutMix (activity) | DISABLED (alpha=0.0) | alpha=1.0 | config.py:1138 |
| Spatial aug (RF1) | DISABLED for RF1 bootstrap | — | config.py:1463 |

### Validation Augmentations

No augmentations. Images are loaded at original 1280x720, normalized by ImageNet stats.

### Test-Time Augmentations (eval_tta.py)

- Multi-scale: {0.8, 1.0, 1.2}
- Horizontal flip
- Soft-NMS merge (sigma=0.5)
- 6 total augmentations per image

### Augmentation Design Notes

1. Mixup and CutMix are DISABLED (config.py:642, 1138) because their implementations mix logits (after forward pass) instead of images (before forward pass), causing label corruption.

2. RandAugment is disabled for RF1 bootstrap (config.py:1455) because training with augmentation but evaluating on clean images causes a distribution mismatch that suppresses det_mAP50.

---

## 30. Feature Bank Gradient Flow (Root Cause Fix)

### The Bug (2026-06-30 v4)

The FeatureBank stores projected features in a ring buffer [B, T=16, 512]. The original code:

```python
# model.py (old code, removed 2026-06-30)
bank_i = torch.stack(seq)  # [T, 512], all detached
bank_i[-1] = feat_i  # in-place assignment
```

Problem: `torch.stack` of detached tensors produces a tensor with `requires_grad=False`. In-place assignment (`bank_i[-1] = feat_i`) does NOT retroactively enable gradient flow because PyTorch's in-place assignment preserves the storage's `requires_grad` flag. Result: activity gradient norm was ~0.010 instead of expected ~0.48.

### The Fix

```python
# model.py:1244 (post-fix)
bank_i = torch.cat([bank_i[:-1].detach(), feat_i.unsqueeze(0)], dim=0)
```

This creates a NEW tensor via `torch.cat` where the last position carries feat_i's gradient. The gradient now flows through:
- feat_i -> proj_features (has grad) -> concat -> bank_i (has grad at last position)
- bank_i[0:-1] are still detached (history cached, no backward-through-time needed)
- The TCN receives [detached_history, grad_carrying_current] -> processes all T positions

After the fix: activity grad norm recovered from 0.010 to ~0.48 (48x improvement).

### Slot Overwrite Gate

Controlled by FEATURE_BANK_SLOT_OVERWRITE (config.py:1115). When True (current default), the live frame overwrites position -1 of the bank, preserving gradient flow through the current frame. When False, the entire bank output is detached - used for ablation experiments.

---

## 31. Soft-NMS Integration

**File:** `src/evaluation/soft_nms.py` (lines 1-114, referenced but not shown in full)

### Algorithm

Gaussian Soft-NMS (Bodla et al., 2017):
```
For each detected box with max score:
  For every other box:
    iou = IoU(detected_box, other_box)
    score[other] *= exp(-iou^2 / sigma)
```

### Integration Points

1. **eval_tta.py per-class**: `soft_nms(pb[cm], ms[cm], sigma=soft_nms_sigma, score_thresh=0.001)` at line 203
2. **eval_tta.py merge**: `soft_nms(cat_boxes[cm], cat_scores[cm], sigma=0.5)` at line 327
3. **Standard eval**: Uses `nms_numpy` (standard NMS) - NO Soft-NMS in evaluate.py

The distinction: TTA eval uses Soft-NMS throughout (both per-augmentation decoding and final merging). Standard eval uses hard NMS (DET_EVAL_NMS_IOU_THRESH=0.5).

---

## 32. Dataset Constants Reference

| Constant | Value | Meaning | File:Line |
|---|---|---|---|
| NUM_DET_CLASSES | 24 | ASD classes (22 states + background + error) | config.py:200 |
| NUM_CLASSES_ACT | 75 | Action IDs 0-74 (fixed, not data-derived) | config.py:244 |
| NUM_PSR_STEPS | 36 | Procedure step types | config.py:478 |
| NUM_PSR_COMPONENTS | 11 | Assembly components (comp0-comp10) | config.py:479 |
| NUM_KEYPOINTS | 17 | COCO-style keypoints (pseudo) | config.py:459 |
| NUM_HEAD_POSE_DOF | 9 | [forward, position, up] each 3-DoF | config.py:472 |
| NUM_HAND_JOINTS | 26 | MediaPipe-style per hand | config.py:475 |
| IMG_WIDTH | 1280 | Native resolution | config.py:527 |
| IMG_HEIGHT | 720 | Native resolution | config.py:528 |

### Activity Class Names

Action IDs 1-74 are loaded from AR_labels.csv on disk (config.py:247-281). IDs 37 and 64 are absent in stock IndustReal, leaving two permanently cold channels. Slot 0 is "take_short_brace" (a real action with 63 frames), NOT background/NA.

### Detection Class Names

24 assembly state names encoded as 11-bit binary strings (config.py:203-227), representing the fill-forward state of 11 assembly components at that step.

---

## 33. Ablation Configuration Summary

Key ablation flags for paper experiments (all env-overridable):

| Ablation | Env Var | Default | Effect |
|---|---|---|---|
| Fixed vs learned Kendall | KENDALL_FIXED_WEIGHTS | 0 | Bypass learned log_vars, use fixed lambdas |
| Activity head type | ACTIVITY_HEAD_SIMPLE | True | Simple MLP (per-frame) vs TCN+ViT (temporal) |
| PSR transition objective | USE_PSR_TRANSITION | True | Transition targets vs per-frame BCE |
| Geometry-aware head pose | USE_GEO_HEAD_POSE | True | 6D rotation vs raw 9-DoF MSE |
| Feature bank gradient | FEATURE_BANK_DETACH_GRAD_ENTRIES_ONLY | True | Grad on current frame only vs no grad |
| Hand-FiLM (PoseFiLM) | USE_HAND_FILM | True | Keypoint conditioning on C5 |
| HeadPoseFiLM | USE_HEADPOSE_FILM | True | Second FiLM from 9-DoF head pose |
| VideoMAE stream | USE_VIDEOMAE | False | 2-stream activity with VideoMAE |
| AMP dtype | AMP_DTYPE | bf16 | bf16 or fp16 for mixed precision |
| Body pose freeze | FREEZE_BODY_POSE_BRANCH | False | Freeze body-pose sub-head |
| Activity class grouping | ACT_CLASS_GROUPING | hybrid | none, verb, or hybrid grouping |
| OneCycleLR peak factor | ONE_CYCLE_PEAK_FACTOR | auto | Goyal linear scaling vs manual |

---

## 34. Performance Budgets

### Memory

| Component | VRAM | Notes |
|---|---|---|
| ConvNeXt-Tiny backbone | ~2.1 GB | FP32, at 1280x720 |
| FPN + Detection head | ~1.8 GB | 5-level pyramid at full resolution |
| Feature Bank + Activity | ~0.6 GB | T=16 x 512-dim ring buffer |
| PSR + Transformer | ~0.4 GB | 3-layer causal transformer |
| Head pose + FiLM modules | ~0.3 GB | Small MLPs |
| Data + misc | ~1.0 GB | Images, targets, optimizer states |
| **Total (batch=1)** | **~6.2 GB** | |
| **Total (batch=6)** | **~10-12 GB** | RTX 5060 Ti 16GB - safe |

### Throughput

| Mode | Batch Size | FPS | Notes |
|---|---|---|---|
| Training (FP32) | 6 | ~7-9 | 8 accum steps, RTX 5060 Ti |
| Training (BF16) | 6 | ~12-15 | Estimated 1.5-2x speedup |
| Inference (FP32) | 1 | ~11 | Single frame, all heads |
| Inference with VideoMAE | 1 | ~8 | ~25% FPS drop |

### DET GT Frame Statistics

| Metric | Value | Notes |
|---|---|---|
| Training frames | ~3,667 | Full dataset |
| Validation frames | ~1,928 | |
| GT-bearing frames | ~24% | Activity-balanced sampler default |
| GT pixels vs total | ~0.3-0.5% | Small assembly parts in 720p |
| Positive anchors | ~20/173K per GT frame | ~0.01% of anchors |
| Max IoU for best anchor | ~0.45-0.55 | Typical at DET_POS_IOU_THRESH=0.4 |

---

## 35. Key Numerical Constants Table

| Constant | Value | Used In | File:Line |
|---|---|---|---|
| Soft-argmax train temp | 1.0 | gradient flow through heatmaps | config.py:863 |
| Soft-argmax eval temp | 0.07 | coordinate precision | config.py:862 |
| Focal alpha (det) | 0.50 | detection focal loss | config.py:704 |
| Focal gamma (det) | 2.0 | detection focal loss | config.py:705 |
| Gamma positive | 0.0 | no easy-pos suppression | config.py:770 |
| Gamma negative | 1.5 | moderate hard-neg | config.py:771 |
| Label smoothing (act) | 0.1 | activity CE loss | config.py:796 |
| PSR focal gamma | 0.5 | PSR binary focal | config.py:1050 |
| PSR focal alpha | 0.25 | PSR binary focal | config.py:1043 |
| GIoU weight | 2.0 | detection regression | config.py:749 |
| Wing epsilon | 0.005 | keypoint regression | config.py:778 |
| Wing omega | 0.05 | keypoint regression | config.py:777 |
| Kendall log_var clamp | [-4, 2] | all log_vars | losses.py:1702 |
| Anchor sizes | (96,160,256,384,512) | P3-P7 RetinaNet | config.py:499 |
| EMA decay | 0.995 | EMA averaging | config.py:631 |
| Gradient clip norm | 5.0 | global grad clipping | config.py:597 |
| Watchdog timeout | 1800s | GPU heartbeat | config.py:592 |
| Weight decay | 1e-3 | AdamW | config.py:579 |
| Feature bank T | 16 | temporal window | config.py:159 |
| PSR sequence length | 8 | contiguous frames | config.py:1063 |

---

## 36. Error States and Guard Mechanisms

### NaN Propagation Prevention

Ten explicit NaN-to-fallback locations (documented in section 19) plus:

1. `_checkpoint_has_nan()` - prevents saving corrupt checkpoints (train.py:779-789)
2. Step-0 assertion - detects collapsed detection head (train.py:1607-1627)
3. CUDA health check before eval crash save (evaluate.py:3377-3385)
4. Pre-backward CUDA sync (train.py:1817-1821)
5. Gradient NaN guard before optimizer step (train.py:1920-1932)
6. PSR-NEG1 diagnostic - logs fraction of ignored PSR targets (train.py:1288-1295)

### CUDA Crash Hardening

| Mechanism | Severity | File:Line |
|---|---|---|
| CUDA_LAUNCH_BLOCKING=1 | Always on (env var) | train.py:20 |
| expandable_segments:True | Prevents frag OOM | train.py:6 |
| NVIDIA_TF32_OVERRIDE=0 | Determinism | train.py:25 |
| CUDA_MODULE_LOADING=LAZY | Reduce context size | train.py:29 |
| CUBLAS_WORKSPACE_CONFIG | Context stability | train.py:11 |
| faulthandler.register(SIGUSR1) | Signal stack trace | train.py:36 |
| Signal handlers (SIGSEGV etc.) | Crash recovery | train.py:1087-1122 |
| SIGHUP handler | Terminal death | train.py:1128-1138 |
| OMP_NUM_THREADS=4 | Thread convoy fix | train.py:112 |

### Retry/Recovery Paths

1. Outer retry loop in main(): restarts from crash_recovery.pth on CUDA errors
2. Epoch-level retry: resumes at the epoch boundary
3. Batch-level retry: skips bad batch, continues training
4. AMP fallback: scalar detects inf grads, skips optimizer step

---

## 42. Validation Loop Detail

The validation loop at the end of each epoch (train.py main function, after train_one_epoch returns):

```
1. memory flush (_flush_before_val -> clear COCO cache, gc collect, empty_cache)
2. IN_EVALUATION_PHASE = True
3. Set model to eval mode
4. Heartbeat write (status='validating')
5. Choose eval path:
   a. Subprocess (USE_SUBPROCESS_EVAL=True):
      - Launch subprocess on GPU 1 (isolated CUDA context)
      - Load latest checkpoint, run evaluate_all with EVAL_MAX_BATCHES cap
      - Read results from JSON file
      - SIGKILL after SUBPROCESS_EVAL_TIMEOUT (900s)
   b. In-process (default):
      - Run evaluate_all on current model
      - Determine max_batches:
        - Full eval every DET_METRICS_EVERY_N epochs: max_batches=0 (all)
        - Gated eval other epochs: max_batches=GATE_EVAL_MAX_BATCHES (250)
6. Restore model to train mode
7. IN_EVALUATION_PHASE = False
8. Compute combined metric from results:
   combined = 0.30 * det_mAP50 + 0.35 * act_macro_f1 + 0.15 * pose_metric + 0.20 * psr_f1
9. Check for best metric: save best.pth if improved
10. Save latest.pth
11. Save epoch_N.pth
12. Update stage_state.json heartbeat
13. Early stopping check: if combined not improved for PATIENCE (10) epochs -> stop
```

### Eval Gating for Speed

Not every epoch runs a full 38K-frame evaluation. The gate system trades speed for coverage:

| Eval Type | When | Max Batches | Coverage |
|---|---|---|---|
| Full det eval | Every DET_METRICS_EVERY_N (3) epochs | unlimited | All ~1928 val frames |
| Gated val | Every other epoch | GATE_EVAL_MAX_BATCHES (250) | ~2000 frames at batch=8 |
| Step val | Every VAL_EVERY_N_STEPS intra-epoch | 200 | Quick health check |

This means: most epochs run ~250-batch gated eval (~15 min), while every 3rd epoch runs full eval (~90 min).

### Validation Results Logging

The `Val:` line is the primary results output, containing:

```
det_mAP50=0.3584 det_mAP50_pc=0.5734 n_present=15/24
act_clip=0.0625 act_frame=0.1770 act_macro_f1=0.2047 act_top1=0.3110 act_top5=0.5420
forward_angular_MAE_deg=7.83
psr_f1=0.1281 psr_edit=0.7520 psr_pos=0.9693
combined=0.3058->0.4140 (NEW BEST)
vl_det=1.71 vl_hp=0.08 vl_act=1.40 vl_psr=0.00
```

The `vl_*` metrics are unweighted loss values (accumulated from criterion output before Kendall weighting) - critical for distinguishing task health from Kendall-learned suppression.

---

## 43. Complete Metric Registry

### Detection Metrics

| Metric | Type | Source | Description |
|---|---|---|---|
| det_mAP50 | float | compute_det_metrics_extended | mAP at IoU=0.5, COCO interpolation |
| det_mAP_50_95 | float | compute_det_metrics_extended | mAP averaged over 0.5:0.05:0.95 |
| det_mAP50_pc | float | compute_det_metrics_extended | mAP at 0.5 over present-classes only (un-diluted) |
| det_n_present_classes | int | compute_det_metrics_extended | How many of 24 classes have any GT in subset |
| per_class_ap | dict | compute_detection_map | Per-class AP at 0.5 |
| det_precision | float | derived | Macro detection precision |
| det_recall | float | derived | Macro detection recall |
| probe_bestIoU_max | float | probe_detection_batch | Max IoU for best-matching pred (diagnostic) |
| probe_score_p99 | float | probe_detection_batch | 99th percentile cls score (diagnostic) |

### Activity Metrics

| Metric | Type | Source | Description |
|---|---|---|---|
| act_top1 | float | compute_activity_accuracy | Top-1 accuracy (activity_mask applied) |
| act_top5 | float | compute_activity_accuracy | Top-5 accuracy |
| act_macro_f1 | float | compute_activity_metrics | Macro F1 over present classes |
| act_clip_accuracy | float | _compute_clip_level_accuracy | Clip-level (16 uniform frames per recording) |
| act_frame_accuracy | float | compute_activity_metrics | Per-frame accuracy |
| act_top1_correct | int | compute_activity_accuracy | Raw correct count |
| act_top5_correct | int | compute_activity_accuracy | Raw correct count |
| act_total | int | compute_activity_accuracy | Total valid frames |

### Head Pose Metrics

| Metric | Type | Source | Description |
|---|---|---|---|
| forward_angular_MAE_deg | float | compute_head_pose_metrics | Forward direction angular error (degrees) |
| up_angular_MAE_deg | float | compute_head_pose_metrics | Up direction angular error (degrees) |
| position_MAE_mm | float | compute_head_pose_metrics | Position L2 error (mm, assumed) |
| head_pose_MAE | float | compute_head_pose_metrics | Aggregate head pose error |

### PSR Metrics

| Metric | Type | Source | Description |
|---|---|---|---|
| psr_f1 | float | decode_and_score_psr | Transition F1 at +/-3 frame tolerance |
| psr_pos | float | decode_and_score_psr | Ordered pair fraction (correct adjacent pairs) |
| psr_edit | float | decode_and_score_psr | Damerau-Levenshtein edit score |
| psr_pos_blind | float | decode_and_score_psr | POS with canonical order disclosure (Q43) |
| psr_tau | float | decode_and_score_psr | Kendall tau correlation (Q17) |
| psr_f1_calibrated | float | decode_and_score_psr | F1 with per-component thresholds (Q18) |
| psr_step_acc | float | compute_psr_accuracy | 36-class step prediction accuracy |
| psr_comp_acc | float | compute_psr_accuracy | 11-component binary prediction accuracy |

### Efficiency Metrics

| Metric | Type | Source | Description |
|---|---|---|---|
| eff_fps | float | compute_efficiency_metrics | Single-GPU inference FPS (batched) |
| eff_fps_streaming | float | compute_efficiency_metrics | Streaming FPS (single-frame, no batching) |
| pipeline_params_m | float | compute_efficiency_metrics | Total trainable parameters (millions) |
| pipeline_gflops | float | compute_efficiency_metrics | Total forward-pass GFLOPs |

### Combined Metric Formula

```
combined = 0.30 * det_mAP50 + 0.35 * act_macro_f1 + 0.15 * (1 - forward_MAE_normalized) + 0.20 * psr_f1
```

Used for best-model selection and early stopping.

---

## 44. Stage Manager Integration

**File:** `src/training/stage_manager.py` (external process, referenced by train.py via JSON state file)

The stage manager runs as an external process that:
1. Writes the next-stage preset key to `rf_stage_state.json`
2. Launches train.py with the appropriate preset and --resume from latest.pth
3. Monitors training via heartbeat updates in the state file

The train.py writes to `rf_stage_state.json` (via _write_stage_heartbeat, train.py:220-263):
- epoch: current epoch number
- status: running/validating/done/crashed
- best_metric: best combined metric value
- best_metrics: detailed per-task breakdows
- batch: (current_batch, total_batches) within epoch
- training_pid: OS process ID for kill/monitor
- last_heartbeat: ISO 8601 timestamp

The stage manager decides when to switch stages based on:
- Convergence on the combined metric (no improvement for N epochs)
- Minimum epoch count for current stage
- Maximum epoch count reached
- Manual override via CLI

---

## 45. Paper Claims Architecture Support

### Primary Claims

| Paper Claim | Architectural Component | Evidence in Code |
|---|---|---|
| Single-GPU 4-task system (46.5M params) | ConvNeXt-Tiny + 4 heads + Kendall | model.py:1749-1888 (POPWMultiTaskModel) |
| First ego-pose baseline (8.14 deg forward MAE) | HeadPoseHead + HeadPoseFiLM | model.py:1484-1533, config.py:954 |
| PSR POS 0.968 beats SOTA 0.812 | MonotonicDecoder + Transition Loss + Q43 verified | psr_transition.py:76-163, 119-progress-log.md:58-67 |
| Detection mAP50_pc 0.506 (present-class only) | RetinaNet + asymmetric focal + OHEM | model.py:500-567, losses.py:74-200 |
| Per-frame activity macro-F1 0.110 (first baseline) | ActivityHead simple MLP | model.py:1377-1385, config.py:950 |

### Supporting Architecture Features

| Feature | Paper Section | Implementation File:Line |
|---|---|---|
| Kendall uncertainty weighting | Section 3.7 | losses.py:1658-1790 |
| HP_PREC_CAP | Novel (Opus v8 addition) | losses.py:1707-1713 |
| Activity warmup ramp | Section 3.7.1 | losses.py:1385-1389 |
| Gradient blending (c5_mod blend) | Section 5.4 | model.py:2175-2176 |
| PSR transition objective | Novel (Opus v5 addition) | psr_transition.py:31-70 |
| Feature Bank gradient flow fix | Bug fix (Opus v11) | model.py:1239-1244 |
| Geometry-aware head pose (6D rotation) | Novel improvement | model.py:1848-1860, config.py:1100 |
| Soft-NMS merge for TTA | Evaluation improvement | eval_tta.py:264-357 |

### Honest Disclosures for Paper (from 119-progress-log.md)

1. POS paradigm: SOTA uses per-recording transition detection F1, this work measures per-frame state ordering accuracy
2. n_present=15/24: validation subset evaluation (not full 24-class mean)
3. Per-frame vs temporal activity: per-frame MLP is a baseline, not comparable to temporal models
4. GPU pricing: sub-$450 consumer GPU uses promotional pricing ($299), MSRP is $429
5. COCO-pretrained YOLOv8m achieves 0 mAP on IndustReal classes -- domain-specific pretraining is required

---

## 46. Per-File Line Reference: config.py Constants Table

All critical configurable constants with file:line:

| Constant | Value | File:Line | Notes |
|---|---|---|---|
| ANCHOR_SIZES | (96,160,256,384,512) | config.py:499 | Absolute pixels for 1280x720 |
| BACKBONE | 'convnext_tiny' | config.py:125 | Also supports 'resnet50' |
| BATCH_SIZE | 6 | config.py:569 | RTX 5060 Ti 16GB safe |
| NUM_DET_CLASSES | 24 | config.py:200 | 22 ASD + background + error |
| NUM_CLASSES_ACT | 75 | config.py:244 | Fixed, not data-derived |
| NUM_KEYPOINTS | 17 | config.py:459 | COCO-style, pseudo-generated |
| NUM_HEAD_POSE_DOF | 9 | config.py:472 | Forward + position + up |
| NUM_PSR_COMPONENTS | 11 | config.py:479 | Assembly component count |
| NUM_PSR_STEPS | 36 | config.py:478 | Procedure step types |
| GRAD_ACCUM_STEPS | 8 | config.py:570 | Effective batch = 48 |
| EPOCHS | 100 | config.py:577 | Total training epochs |
| BASE_LR | 5e-4 | config.py:578 | Paper spec |
| WEIGHT_DECAY | 1e-3 | config.py:579 | Fixed from 5e-2 |
| WARMUP_EPOCHS | 2 | config.py:583 | Paper spec |
| VAL_EVERY | 1 | config.py:598 | Validate every epoch |
| EVAL_MAX_BATCHES | 250 | config.py:600 | Cap for training val |
| MIXED_PRECISION | False | config.py:614 | FP32 only (AMP disabled) |
| AMP_DTYPE | 'bf16' | config.py:626 | For future AMP enable |
| USE_EMA | True | config.py:630 | EMA enabled |
| EMA_DECAY | 0.995 | config.py:631 | Stage 3 value |
| FOCAL_ALPHA | 0.50 | config.py:704 | Raised from 0.25 |
| FOCAL_GAMMA | 2.0 | config.py:705 | Paper spec |
| GIOU_WEIGHT | 2.0 | config.py:749 | Reg loss weight |
| KENDALL_HP_PREC_CAP | True | config.py:89 | Head pose precision cap |
| ACTIVITY_HEAD_SIMPLE | True | config.py:950 | Per-frame MLP (not temporal) |
| USE_PSR_SEQUENCE_MODE | True | config.py:1062 | Contiguous sequences for PSR |
| PSR_SEQUENCE_LENGTH | 8 | config.py:1063 | T=8 windows |
| PSR_SEQ_EVERY_N_BATCHES | 4 | config.py:1078 | 25% seq rate |
| PSR_SEQ_LOSS_SCALE | 1.5 | config.py:1079 | Seq batch loss scale |
| USE_PSR_TRANSITION | True | config.py:1085 | Transition objective |
| USE_GEO_HEAD_POSE | True | config.py:1100 | 6D rotation head pose |
| DET_EVAL_SCORE_THRESH | 0.001 | config.py:672 | COCO-comparable low threshold |
| DET_EVAL_NMS_IOU_THRESH | 0.5 | config.py:674 | Standard NMS threshold |
| ONE_CYCLE_PEAK_FACTOR | 'auto' | config.py:1164 | Goyal linear scaling |
| SUBPROCESS_EVAL_TIMEOUT | 900 | config.py:1213 | Seconds before SIGKILL |
| GATE_EVAL_MAX_BATCHES | 250 | config.py:1205 | Full val coverage |
| DET_METRICS_EVERY_N | 3 | config.py:1200 | Full mAP every 3 epochs |
| ACTIVITY_HEAD_DROPOUT | 0.3 | config.py:920 | Simple classifier dropout |
| PSR_SENSITIVITY_WEIGHT | 0.50 | config.py:1131 | Per-component logit separation |
| CUDNN_BENCHMARK | False | config.py:684 | Stability (was True) |
| NUM_WORKERS | 0 | config.py:604 | Eliminate deadlocks |
| RAM_CACHE_MAX_IMAGES | 8000 | config.py:608 | Full dataset in RAM |
