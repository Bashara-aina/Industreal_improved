# MTL Architecture Research: Backbone, Neck, and Head Design for 4-Task MTL

**Generated:** 2026-07-23
**Context:** POPW 4-task MTL (Detection + Activity + Pose + PSR) on IndustReal egocentric assembly video.
**NOTE (CORRECTION):** The active training script `scripts/train/train_mtl_v3_yolov8_head.py` uses `MTLMViTModel` (MViTv2-S backbone, K400 pretrain) from `src/models/mvit_mtl_model.py`. The `POPWMultiTaskModel` (ConvNeXt-Tiny backbone) in `src/models/model.py` is a secondary/research model used in older scripts and smoke tests but is NOT the current training target. This report covers both architectures — MViTv2-S as the primary active model, ConvNeXt-Tiny as the secondary research reference. All metrics in the companion reports (loss balancing, PSR/activity, training recipe) refer to the MViTv2-S model unless stated otherwise.

---

## Table of Contents

1. [Current Architecture Summary](#1-current-architecture-summary)
2. [Backbone Design and Capacity](#2-backbone-design-and-capacity)
3. [Neck / Feature Pyramid Design](#3-neck--feature-pyramid-design)
4. [Detection Head Design](#4-detection-head-design)
5. [Feature Routing for Multi-Task Learning](#5-feature-routing-for-multi-task-learning)
6. [Task-Head Architecture Comparison](#6-task-head-architecture-comparison)
7. [Temporal Modeling Survey](#7-temporal-modeling-survey)
8. [Parameter Efficiency and Bottlenecks](#8-parameter-efficiency-and-bottlenecks)
9. [Recommendations and Migration Path](#9-recommendations-and-migration-path)
10. [References](#10-references)

---

## 1. Current Architecture Summary

### Active Model: MViTv2-S (MTLMViTModel, mvit_mtl_model.py)

| Component | Details | Params |
|-----------|---------|--------|
| **Backbone** | MViTv2-S (K400 pretrain, T=16, 224px) | 34.54M |
| **BiFPN Neck** | LightweightFPN (P2-P5, 256ch, 3D convs) | ~14.53M |
| **Detection Head** | YOLOv8-style DFL head (decoupled cls + reg), 3 levels (P3-P5) | ~1.20M |
| **Activity Head** | 3-layer MLP (768->2048->1024->75) | ~3.75M |
| **PSR Head** | P5 -> spatial pool -> causal Transformer (d=256) | ~1.78M |
| **Pose Head** | 2-layer MLP (768->256->6) | ~0.20M |
| **Total** | | **~55.69M** |

### Secondary / Research Model: ConvNeXt-Tiny (POPWMultiTaskModel, model.py)

| Component | Details | Params |
|-----------|---------|--------|
| **Backbone** | ConvNeXt-Tiny (ImageNet-1K pretrain) | 28.59M |
| **FPN Neck** | Standard P3-P7 (256ch), top-down only, additive fusion | ~4.48M |
| **Detection Head** | RetinaNet-style, 9 anchors, 5 levels (P3-P7) | ~5.31M |
| **Activity Head** | FeatureBank+TCN+2xViT + optional VideoMAE V2 stream | ~0.69M |
| **PSR Head** | PSRHead (hidden_dim=128) | ~3.08M |
| **Pose Body** | ConvT+soft-argmax pipeline | ~1.64M |
| **Pose Head** | 2x FiLM (PoseFiLM, HeadPoseFiLM) + final projection | ~1.45M + ~1.24M |
| **Total** | | **~46.47M** |

**Key Difference:** The active MViTv2-S model is 9.22M params larger with a stronger video-pretrained backbone (K400) and BiFPN neck, but simpler task heads (basic MLPs). The ConvNeXt model has complex FiLM-based conditioning for pose and a sophisticated FeatureBank+TCN+ViT activity pipeline — these head innovations could potentially be ported to the MViTv2-S model.

---

## 2. Backbone Design and Capacity

### 2.1 ConvNeXt-Tiny (Secondary / Research Reference)

ConvNeXt-Tiny is a ResNet-50-sized pure-conv architecture with modern design:
- 4 stages: C2 (56x56, 96ch), C3 (28x28, 192ch), C4 (14x14, 384ch), C5 (7x7, 768ch)
- Large-kernel depthwise conv (7x7), LayerNorm, GELU
- ImageNet-1K pretrained (81.5% top-1)
- **28.59M params** (vs 25.56M for ResNet-50)

**Capacity assessment for 4-task MTL:** At 28.59M for 4 tasks, the backbone provides ~7.15M params per task on average. The C4 stage (384ch, 14x14) is the narrowest shared representational bottleneck with 10+ residual blocks competing for capacity.

**Gradient checkpointing:** Available via `use_backbone_checkpoint=True`, reduces VRAM ~30-40%.

### 2.2 MViTv2-S (Active)

MViTv2-S is a transformer-based video backbone with:
- 16 blocks, 3D pooling attention (T=8 after conv_proj stride 2)
- Channel progression: 96->96->192->192->384->384->...->384->768
- 4 heads (head_dim=96) for blocks 0-13, 8 heads for blocks 14-15
- K400 pretrained (80.8% top-1)
- **34.54M params** (6M more than ConvNeXt-Tiny)

**Critical temporal limitation:** MViTv2-S has NO hierarchical temporal abstraction. All `stride_q[0] = 1` and `stride_kv[0] = 1` throughout all 16 blocks. The temporal dimension stays at T=8 (16 input frames pooled by conv_proj stride 2). Assembly actions spanning 1-4 seconds (10-40 frames) rely entirely on late fusion/clip aggregation.

**384-dim bottleneck:** Blocks 4-13 (10 out of 16 blocks) operate at 384 hidden dimension. This is the narrowest point for 4-task gradient competition. The MLP expansion (4x -> 1536 intermediate) is adequate for single-task K400 but potentially limiting for 4-task MTL.

### 2.3 Larger Backbone Feasibility

| Backbone | Params | GFLOPs | K400 Top-1 | VRAM @224 B=1 (FP16) | Fits 16GB? |
|----------|--------|--------|------------|---------------------|------------|
| ConvNeXt-Tiny | 28.6M | 4.5 | 81.5%* | ~2 GB | YES |
| MViTv2-S | 34.5M | 64 | 80.8% | ~3.5 GB (ckpt) | YES |
| MViTv2-B | ~52M | ~112 | ~82.0% | ~5.5 GB (ckpt) | YES |
| VideoMAE-base | ~87M | ~180 | ~81.0% | ~8 GB (ckpt) | YES (boundary) |
| MViTv2-L | ~97M | ~282 | ~83.5% | ~9 GB | NO (train) |

*ConvNeXt ImageNet-1K top-1; K400 not directly comparable

**Verdict:** MViTv2-B (~52M) is the largest feasible upgrade on 16GB with gradient checkpointing. The incremental +1.2% K400 top-1 over MViTv2-S may not justify the 50% param increase for MTL.

---

## 3. Neck / Feature Pyramid Design

### 3.1 ConvNeXt FPN (model.py, secondary model)

Standard top-down FPN with additive fusion:
- Inputs: C3 (192ch), C4 (384ch), C5 (768ch) -> lateral 1x1 -> 256ch
- Top-down: P5 -> upsample + P4 -> P4 -> upsample + C3 -> P3
- Extra levels: P6 (stride 64 via Conv), P7 (stride 128 via Conv+ReLU)
- **Outputs:** P3 (28x28), P4 (14x14), P5 (7x7), P6 (4x4), P7 (2x2) at 256ch
- **Params:** ~4.48M (4 lateral + 3 smooth + 2 extra convs)

### 3.2 BiFPN (mvit_mtl_model.py, active model)

Weighted Bidirectional FPN (EfficientDet-style):
- Inputs: P2-P5 (96/192/384/768ch) -> lateral 1x1 -> 256ch
- Top-down pathway + bottom-up pathway
- Learnable fusion weights (`_fast_weightsum`: ReLU + normalize)
- **3D convolutions** (Conv3d, kernel=3, preserving T=8)
- **Params:** ~14.53M (8x Conv3d(256,256,3) dominate at 1.77M each)
- **Detection uses P3/P4/P5 only** (P2 excluded despite being computed)

### 3.3 Comparison and Findings

| Feature | FPN (model.py) | BiFPN (mvit_mtl) | Benefit |
|---------|----------------|-------------------|---------|
| Fusion | Additive (fixed) | Learned weighted | +3-7% AP on COCO |
| Direction | Top-down only | Bidirectional | +1-3% AP |
| Levels | P3-P7 (5) | P2-P5 (4) | P2 helps small objects |
| Temporal | 2D only | 3D (T=8) | Temporal context |
| Params | ~4.48M | ~14.53M | 3.2x larger |

**Key finding:** The BiFPN at 14.5M is ~12x larger than the detection head (1.2M), an inverted ratio vs typical architectures (neck 2-4x smaller than detection head). The 8x Conv3d(256,256,3) are the primary cost at ~1.77M each.

**P2 exclusion paradox:** Detection skips P2 (line 575-577: "Skip P2 (raw conv_proj features, no semantics)") but the BiFPN still computes P2 for its contribution to the bottom-up pathway. P2's processing (~2M params for td_conv + bu_conv) is auxiliary—it contributes only indirectly.

**Feasibility of BiFPN in ConvNeXt model:** The active MViTv2-S model already has BiFPN enabled. The ConvNeXt model does not, but `config.py` has `USE_BIFPN = False` with a docstring claiming +0.4-0.7 mAP expected. If migrating ConvNeXt-based training in the future, the FPN would need upgrading to a bidirectional weighted version.

### 3.4 FPN Improvements (from literature survey)

| Technique | Expected AP gain | Effort | Papers |
|-----------|-----------------|--------|--------|
| Switch to BiFPN (weighted fusion) | +3-7% | Trivial (config toggle) | EfficientDet, CVPR 2020 |
| Add P2 level (stride 4) | +3-5% AP_small | Medium (1 day) | RedMask ICCV 2023 |
| CARAFE upsampling | +2-3% | Medium (2 days) | ICCV 2019 |
| ASPP in P5 (dilated encoder) | +2-4% | Small (1 day) | DeepLabv3, TPAMI 2018 |
| PAFPN bottom-up path | +1-3% | Medium (2 days) | PAFPN, CVPR 2018 |
| **Cumulative** | **+11-22% AP** | **1-2 weeks** | |

---

## 4. Detection Head Design

### 4.1 ConvNeXt Detection Head (model.py, secondary model)

RetinaNet-style with 9 anchors per location (3 sizes x 3 ratios):
- Shared 4-conv tower (Conv2d + BN + ReLU) for cls, shared 4-conv tower for reg
- Cls output: [B, 24*num_anchors, H, W] = [B, 216, H, W]
- Reg output: [B, 4*num_anchors, H, W] = [B, 36, H, W]
- **Params:** ~5.31M (shared tower + output convs across 5 levels)
- **Loss:** Focal Loss (cls) + GIoU Loss (reg)
- **Matching:** IoU-based anchor matching (>0.5 positive)

### 4.2 Active Detection Head (mvit_mtl_model.py, MViTv2-S)

YOLO-style decoupled head with DFL:
- Single Conv3x3+GN+ReLU per branch
- Cls output: [B, 24, H, W] (per-location, no anchors)
- Reg output: [B, 64, H, W] (16 bins x 4 coords for DFL)
- **Params:** ~1.20M (cls head + reg head, shared across 3 levels)
- **Loss:** Varifocal Loss + DFL + CIoU
- **Matching:** TAL (Task-Aligned Learning, TOOD-style)

### 4.3 Comparison

| Feature | ConvNeXt (model.py, secondary) | MViTv2-S (active) |
|---------|-------------------|-------------------|
| Style | Anchor-based (9 anchors) | Anchor-free (per-location) |
| Tower depth | 4 convs per branch | 1 conv per branch |
| Normalization | BatchNorm | GroupNorm |
| Activation | ReLU | ReLU |
| FPN levels | P3-P7 (5 levels) | P3-P5 (3 levels) |
| Params | ~5.31M | ~1.20M |
| Loss | Focal + GIoU | Varifocal + DFL + CIoU |
| Matcher | IoU > 0.5 | TAL (top-K alignment) |
| DFL | None | reg_max=16 |

### 4.4 Modern Detection Techniques (Literature Survey)

**Generalized Focal Loss (GFL) V1/V2** (NeurIPS 2020 / TPAMI 2022):
- QFL: merges cls score with IoU quality estimation. Classification learns `IoU(box, GT)` rather than binary objectness.
- DFL: replaces single-value regression with discrete probability distribution. `pred = sum(softmax(logits) * bin_centers)`.
- GFLV2 adds DGQP: distribution statistics as features to predict IoU quality.
- **Expected AP gain:** +2.6-3.0 AP over ATSS baseline on COCO
- **Status in codebase:** QFL implemented in `src/losses/qfl.py`, integrated via `--loss qfl` flag

**VarifocalNet (VFNet)** (CVPR 2021):
- IACS: cls branch outputs IoU rather than binary sigmoid -> direct ranking by mAP metric
- Varifocal Loss: asymmetric loss weighting, down-weights negatives via `alpha * p^gamma`, preserves gradient from rare positives
- Star-shaped deformable feature extraction
- **Expected AP gain:** +2.0 AP over FCOS+ATSS
- **Status in codebase:** Varifocal Loss partially integrated (training script)

**ATSS Adaptive Matching** (CVPR 2020):
- Per-GT dynamic threshold = mean(top-K IoU) + std(top-K IoU)
- Replaces fixed IoU=0.5 with statistical per-GT threshold
- **Expected AP gain:** +1.4 AP over FCOS
- **Status in codebase:** Implemented in `src/losses/at_matcher.py`, integrated via `--matcher atss`

**TAL (Task-Aligned Learning)** (AAAI 2021):
- Jointly optimizes cls and reg through alignment weighting: `cls_weight = IoU(pred, GT)^alpha`
- Background weights = 1.0 (standard focal loss)
- Boosts detection head LR by 2x when active
- **Expected AP gain:** +3.2 AP over ATSS (per TOOD paper)
- **Status in codebase:** Implemented via `--use-tal` flag

**RTMDet Training Recipe** (arXiv 2022):
- Soft labels weighted by IoU (not hard 0/1)
- Cached Mosaic + MixUp augmentation
- Top-K dynamic assignment (K=13)
- Large-batch EMA
- **Expected AP gain:** +0.5-1.0 AP from EMA, +1-3 AP from recipe

---

## 5. Feature Routing for Multi-Task Learning

### 5.1 Secondary Model Routing (model.py, ConvNeXt)

The ConvNeXt model (`POPWMultiTaskModel`) has **significantly more complex** feature routing than the MViTv2-S model:

```
Input [B, 3, H, W]
  |
ConvNeXt-Tiny
  +---> C2 (56x56, 96ch) -- not used directly
  +---> C3 (28x28, 192ch) -- FPN lateral -> P3
  +---> C4 (14x14, 384ch) -- FPN lateral -> P4 -> GAP -> activity
  +---> C5 (7x7, 768ch)   -- FPN lateral -> P5 -> P6 -> P7
        |                    + det_head (all 5 levels)
        |                    + GAP -> activity (concat with P4 GAP)
        |                    + mod_FiLM1 (keypoints -> gamma/beta -> C5)
        |                    + mod_FiLM2 (headpose -> gamma/beta -> C5_mod)
        |                    + PSR head (multi-scale P3+P4+P5 GAP)
        |
  +---> Pose Body: P3 -> ConvT -> GN -> ReLU -> heatmaps + soft-argmax -> keypoints + confidence
  +---> PoseFiLM: keypoints + conf -> gamma/beta -> C5_mod
  +---> HeadPoseFiLM: 9-DoF head pose -> second-stage gamma/beta -> C5_mod2
  +---> Activity: det_conf + GAP(C5_mod2) + GAP(P4) -> proj -> FeatureBank -> TCN -> 2xViT -> CLS
```

**FiLM conditioning:** PoseFiLM applies spatial feature modulation via learned gamma/beta from keypoint detections onto the C5 feature map. HeadPoseFiLM applies a second-stage modulation from 9-DoF head pose. This creates a task-conditional pathway where pose knowledge modulates detection features.

**Gradient isolation:** `DETACH_REG_FPN`, `DETACH_PSR_FPN`, and `detach keypoints` flags control which gradients flow back to shared features.

### 5.2 Active Model Routing (mvit_mtl_model.py, MViTv2-S)

The MViTv2-S model has simpler routing:

```
MViTv2-S (all 16 blocks, fully shared)
  +---> FPN (P3/P4/P5) ---> Detection head
  +---> cls_token ---> Activity head (3-layer MLP)
  +---> cls_token ---> Pose head (2-layer MLP)
  +---> P5 conv ---> PSR head (spatial pool -> causal Transformer)
```

No FiLM conditioning, no gradient isolation, no RotoGrad integration.

### 5.3 Literature on Feature Routing

**Key papers reviewed** (from Agent 03 Architecture Routing analysis):

| Paper | Venue | Mechanism | Key Finding |
|-------|-------|-----------|-------------|
| Cross-Stitch Networks | CVPR 2016 | Learnable linear combination of task features | Foundational soft-parameter-sharing |
| NDDR-CNN | CVPR 2019 | 1x1 Conv fusion per layer | Layerwise fusion beats late fusion |
| MTAN | CVPR 2019 | Task-specific soft-attention masks | ~2k params/task, outperforms Cross-Stitch |
| Task Routing | ICCV 2019 | FiLM modulation, ~50% sharing optimal | For 10+ heterogeneous tasks |
| MTI-Net | ECCV 2020 | Multi-scale cross-task distillation | Task affinity varies by scale (critical for FPN) |
| Routing Networks | ICLR 2018 | Dynamic per-input router | Per-input dynamic routing |
| AdaShare | NeurIPS 2020 | Learned skip/execute policy per layer | Adaptive layer allocation |
| ETR-NLP | CVPR 2023 | Non-learnable primitives + explicit routing | State-of-the-art routing |
| Sluice Networks | AAAI 2019 | Subspace-level gating | Subspace sharing > layer-level |

**MTI-Net's critical finding for our setting:** "Tasks with high affinity at a certain scale are not guaranteed to retain this behaviour at other scales." For our FPN-based architecture, this means the sharing pattern between detection and PSR at P5 (coarse, semantic) is different than at P3 (fine, spatial). Each FPN level should have its own routing parameters.

**Debate finding** (Agent 13): The Vandenhende MTL survey (arXiv:2004.13379) concludes encoder-focused routing methods provide **"moderate only"** performance improvements. For MViTv2-S specifically, the built-in architectural sophistication (pooling attention, FPN, cls_token) already subsumes much of the benefit that routing methods provide for weaker backbones. The debate recommends:
1. Start with simple shared backbone (all blocks shared)
2. Add gradient surgery (PCGrad/FAMO) if interference detected
3. Only then consider minimal routing (Cross-Stitch at blocks 12 and 16 only)

### 5.4 Recommended Routing Architecture

Based on literature synthesis, the optimal routing for 4 heterogeneous tasks:

```
Blocks 1-8 (low-level): FULLY SHARED across all tasks
Blocks 9-12 (mid-level): GROUPED -- detection path + shared activity/pose/PSR path
Blocks 13-16 (high-level): TASK-SPECIFIC -- detection: FPN; activity/pose: cls_token; PSR: conv features
NDDR/cross-stitch at blocks 9, 12: allow cross-talk between groups
```

This aligns with the ~50% sharing finding from Task Routing (ICCV 2019) while providing task-specific capacity in late blocks.

---

## 6. Task-Head Architecture Comparison

### 6.1 Activity Head

| Feature | ConvNeXt (model.py, secondary) | MViTv2-S (active) | SOTA Reference |
|---------|-------------------|-------------------|----------------|
| Input | GAP(C5_mod2) + GAP(P4) + det_conf(24) -> concat (1017-dim) | cls_token (768-dim) | — |
| Feature extraction | FeatureBank (T=16) + TCN + 2xViT blocks | 3-layer MLP (768->2048->1024->75) | — |
| Temporal modeling | FeatureBank (T=16 history) + TCN causal conv + ViT self-attn | None (clip-level only) | MS-TCN, ASRF |
| Params | ~0.69M | ~3.75M | — |
| Output | 75-class logits | 75-class logits | — |

**The ConvNeXt activity head is significantly more sophisticated** with dedicated temporal modeling (FeatureBank + TCN + ViT), while the MViTv2-S uses a simple MLP on cls_token. The ConvNeXt design has less capacity (0.69M vs 3.75M) but more temporal structure.

### 6.2 PSR Head

| Feature | ConvNeXt (model.py, secondary) | MViTv2-S (active) |
|---------|-------------------|-------------------|
| Input | Multi-scale: GAP(P3) + GAP(P4) + GAP(P5) concat (768-dim) | P5 conv features (768-dim) -> spatial pool |
| Hidden | PSRHead hidden_dim=128 | Causal Transformer d=256 |
| Temporal | Not specified (likely per-frame MLP) | Causal Transformer (2 layers, 4 heads) |
| Output | 11 binary logits (per frame) | 11 binary logits (T=8 sequence) |
| Params | ~3.08M | ~1.78M |

**Key issue (both models):** PSR produces flat ~0.69-0.71 output for all 11 components with frame-to-frame stddev of 0.02 (documented in consult_v2 synthesis). Temporal transition detection is non-functional. The model predicts marginal probability rather than detecting state changes.

### 6.3 Pose Head

| Feature | ConvNeXt (model.py, secondary) | MViTv2-S (active) |
|---------|-------------------|-------------------|
| Body | ConvT decoder (P3 -> heatmaps) + soft-argmax | None (uses cls_token directly) |
| Head | 2-layer MLP on keypoints + 2x FiLM (keypoint + head pose) | 2-layer MLP (768->256->6) |
| Output | 6-DoF (quaternion + 3D translation) | 6-DoF |
| Params | ~4.33M (body 1.64M + head 1.45M + 2 FiLMs 1.24M) | ~0.20M |

**The ConvNeXt model has a much more sophisticated pose pipeline** with:
1. ConvTranspose2d decoder on P3 features -> heatmaps -> soft-argmax -> keypoints
2. PoseFiLM: keypoints + confidence modulate C5 features (gamma/beta)
3. HeadPoseFiLM: 9-DoF head pose -> second-stage modulation on C5_mod

The MViTv2-S pose head is minimal (2-layer MLP on cls_token) but achieves 7.48 deg best forward angular MAE.

---

## 7. Temporal Modeling Survey

### 7.1 Current Temporal Capabilities

| Component | Temporal Resolution | Aggregation Method | Effective Context |
|-----------|-------------------|-------------------|-------------------|
| MViTv2-S backbone | T=8 (16 frames, stride 2) | 3D pooling attention | ~0.8s at 10fps |
| ConvNeXt backbone | Single frame | None (backbone is 2D) | Instantaneous |
| FeatureBank (activity) | T=16 history buffer | Causal conv (TCN) + ViT | ~1.6s at 10fps |
| Detection | Per-frame (temporal pool) | Mean over T=8 (MViT) or none (ConvNeXt) | Single frame |
| PSR (MViT model) | T=8 sequence | Causal Transformer (2 layers) | ~0.8s |
| PSR (ConvNeXt model) | Per-frame | MLP (no temporal) | Single frame |

### 7.2 Key Temporal Gaps

1. **No hierarchical temporal abstraction in MViTv2-S:** The temporal dimension stays flat at T=8. No block ever pools temporally (`stride_q[0] = 1` throughout). Assembly actions spanning 1-4 seconds (10-40 frames) require late-fusion.

2. **Detection has no temporal modeling in either model:** Both models collapse or process per-frame without temporal context. Video detection literature (e.g., YOWO, TubeR) shows consistent gains from temporal aggregation.

3. **PSR temporal modeling is weak:** The MViTv2-S has a causal transformer (T=8 window) but produces flat output. The ConvNeXt model has no PSR temporal modeling at all.

4. **Activity temporal modeling in ConvNeXt model is the strongest:** FeatureBank (T=16) + TCN + 2xViT provides the most temporal context. However, 16 frames at 10fps is only 1.6s, while many assembly actions span 3-10s.

### 7.3 Temporal Modeling Literature

| Method | Venue | Temporal Scope | Applicability |
|--------|-------|---------------|---------------|
| MS-TCN | CVPR 2020 | Full sequence | PSR refinement |
| ASRF | CVPR 2021 | Full sequence | PSR boundary detection |
| Bridge-Prompt | CVPR 2024 | Full sequence | Activity segmentation |
| STORM-PSR | WACV 2024 | Clip-level window | PSR on IndustReal (same dataset!) |
| VideoMamba | ICML 2024 | Full sequence | Efficient video backbone |
| YOWO | CVPR 2019 | Clip-level (T=16) | Video detection |
| TubeR | CVPR 2022 | Clip-level | End-to-end video detection |

**STORM-PSR relevance:** Uses the IndustReal dataset directly, achieving 26.1% delay reduction for PSR state change detection. Uses a temporal stream on top of per-frame features. Most directly applicable improvement for PSR.

---

## 8. Parameter Efficiency and Bottlenecks

### 8.1 Parameter Distribution Comparison

| Component | ConvNeXt (model.py, secondary) | MViTv2-S (active) |
|-----------|-------------------|-------------------|
| Backbone | 28.59M (61.5%) | 34.54M (62.0%) |
| Neck (FPN) | ~4.48M (9.6%) | ~14.53M (26.1%) |
| Detection Head | ~5.31M (11.4%) | ~1.20M (2.2%) |
| Activity Head | ~0.69M (1.5%) | ~3.75M (6.7%) |
| PSR Head | ~3.08M (6.6%) | ~1.78M (3.2%) |
| Pose Body + Head | ~4.33M (9.3%) | ~0.20M (0.4%) |
| **Total** | **~46.47M** | **~55.69M** |

### 8.2 Identified Bottlenecks

1. **FPN-to-detection-head ratio in MViT model:** BiFPN (14.5M) is ~12x larger than detection head (1.2M). Inverted vs typical architectures where neck is 2-4x smaller than head. This is likely wasteful—the bidirectional weighted fusion provides benefit but 12:1 ratio is extreme.

2. **C4 bottleneck in ConvNeXt:** The C4 stage (384ch, 14x14) processes all task features through ~10 residual blocks. For 4-task MTL, this 384-dim shared space is the narrowest representational point.

3. **PSR head oversizing in ConvNeXt model:** At 3.08M with hidden_dim=128 but producing flat output, the PSR head capacity is not being used effectively. The signal from PSR supervision is weak (0.31% transition rate with 54.88% positive frame rate due to fill-forward).

4. **MViT model idle compute:** P2 is computed in the BiFPN (~2M params) but excluded from detection. This is auxiliary computation that benefits detection only indirectly through P3 fusion.

### 8.3 VRAM Analysis

| Resolution | Attention Memory (MViT) | Total Est VRAM (B=1, FP16) | Feasible on 16GB? |
|------------|------------------------|---------------------------|-------------------|
| 224x224 | 0.14 GB | 3-4 GB | YES |
| 320x320 | 0.58 GB | 5-7 GB | YES |
| 480x480 | 2.95 GB | 10-13 GB | YES (w/ ckpt) |
| 640x640 | 9.31 GB | 16+ GB | NO |

**ConvNeXt is more VRAM-efficient** (2D backbone, no attention matrices). At 480px with batch=6 and grad_accum=8 (effective batch=48), the ConvNeXt model fits on 16GB.

---

## 9. Recommendations and Migration Path

### 9.1 Immediate Fixes (Architecture)

1. **[CRITICAL] Fix FPN prefix bug:** The BiFPN's 14.5M params are frozen due to `feature_pyramid.fpn` vs actual `fpn` prefix mismatch. Fix: change prefix to `"fpn"` or register under `self.feature_pyramid`.

2. **[CRITICAL] Integrate RotoGrad properly:** RotoGradRotation is instantiated after optimizer creation. Add `add_param_group()` for RotoGrad params. Implement `rotation_loss()` optimization loop.

3. **[HIGH] Warm-start all task heads:** Generate ST checkpoints for detection, activity, and PSR. Fix `load_state_dict_with_prefix` for MTL head structure vs ST checkpoint structure.

### 9.2 Short-Term (1-2 Weeks)

4. **[HIGH] Anchor-free detection migration (ConvNeXt model only):** The active MViTv2-S model already uses anchor-free YOLOv8-style DFL head. If the ConvNeXt model is revived for production training, migrate from 9-anchor RetinaNet to per-location anchor-free with QFL + DFL. Expected: +3-5% mAP.

5. **[MEDIUM] Switch to ATSS matching:** Replace fixed IoU=0.5 with per-GT adaptive threshold. Already implemented in `src/losses/at_matcher.py` — enable via `--matcher atss`.

6. **[MEDIUM] Add TAL (Task-Aligned Learning):** Already implemented via `--use-tal`. The alignment weighting jointly optimizes cls+reg. Expected: +3% AP.

### 9.3 Medium-Term (2-4 Weeks)

7. **[HIGH] Upgrade FPN to bidirectional weighted fusion (ConvNeXt model only):** The active MViTv2-S model already has BiFPN with learnable fusion weights (14.53M params). If the ConvNeXt model is revived for production training, convert standard FPN to BiFPN — the config toggle `USE_BIFPN` exists but the actual module needs porting from the MViT model. Expected: +3-7% mAP.

8. **[MEDIUM] Evaluate P2 level for small object detection:** The active MViTv2-S model already computes P2 in the BiFPN but excludes it from detection (flagged as "raw conv_proj features, no semantics"). Evaluate whether including P2 via `--use-p2-level` improves AP_small. Expected: +3-5% AP_small.

9. **[MEDIUM] Implement CARAFE upsampling:** Content-aware reassembly replaces nearest-neighbor in FPN upsampling. Expected: +2-3% mAP.

### 9.4 Long-Term (4-8 Weeks)

10. **[HIGH] Temporal modeling for PSR:** Integrate STORM-PSR temporal stream (same dataset, WACV 2024). Add temporal convolution or attention on T=8 window. Addresses the flat PSR output problem. Expected: PSR F1 0.556 -> ~0.70+.

11. **[MEDIUM] Task-specific feature routing:** Implement Cross-Stitch units at FPN levels P3/P4/P5 to learn inter-task feature flow. Follow MTI-Net principle: task affinity varies by scale.

12. **[LOW] Backbone upgrade to MViTv2-B:** The active model uses MViTv2-S (34.5M). Upgrade to MViTv2-B (~52M) for +1.2% K400 top-1. Feasible on 16GB with gradient checkpointing at 224px. Not recommended unless all other improvements fail to close the gap.

13. **[LOW] Tune EMA hyperparameters:** EMA is already active (EMA_START_EPOCH=5, EMA_DECAY=0.995). Tune start epoch and decay rate for optimal detection performance. Consider large-batch EMA (RTMDet recipe: 0.9999 decay with 2000-step burn-in). Expected: +0.5-1% AP from optimization.

---

## 10. References

1. Tan et al., "EfficientDet: Scalable and Efficient Object Detection," CVPR 2020. arXiv:1911.09070. [BiFPN, weighted bidirectional FPN]
2. Li et al., "Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes," NeurIPS 2020 / TPAMI 2022. arXiv:2006.04388. [QFL, DFL, GFLv1/v2]
3. Zhang et al., "Bridging the Gap Between Anchor-based and Anchor-free Detection via ATSS," CVPR 2020. arXiv:1912.02424. [Adaptive threshold matching]
4. Feng et al., "TOOD: Task-aligned One-stage Object Detection," AAAI 2021. arXiv:2108.07755. [TAL, alignment weighting]
5. Zhang et al., "VarifocalNet: An IoU-aware Dense Object Detector," CVPR 2021. arXiv:2008.13367. [IACS, Varifocal Loss]
6. Lyu et al., "RTMDet: An Empirical Study of Designing Real-Time Detectors," 2022. arXiv:2212.07784. [Training recipe, soft labels, mosaic]
7. Wang et al., "CARAFE: Content-Aware ReAssembly of FEatures," ICCV 2019. arXiv:1905.02188. [Content-aware FPN upsampling]
8. Misra et al., "Cross-Stitch Networks for Multi-Task Learning," CVPR 2016. arXiv:1604.03539. [Soft parameter sharing]
9. Gao et al., "NDDR-CNN: Layerwise Feature Fusing in Multi-Task CNNs," CVPR 2019. [1x1 Conv fusion]
10. Liu et al., "MTAN: Multi-Task Attention Network," CVPR 2019. arXiv:1803.10704. [Task-specific attention masks]
11. Strezoski et al., "Many Task Learning With Task Routing," ICCV 2019. [FiLM-based routing, ~50% sharing]
12. Vandenhende et al., "MTI-Net: Multi-Scale Task Interaction Networks," ECCV 2020. [Scale-dependent task affinity]
13. Vandenhende et al., "Multi-Task Learning for Dense Prediction Tasks: A Survey," arXiv:2004.13379, 2020. [Comprehensive MTL survey]
14. Sun et al., "AdaShare: Learning What To Share For Efficient Deep MTL," NeurIPS 2020. [Adaptive layer allocation]
15. Ding et al., "ETR-NLP: Explicit Task Routing with Non-Learnable Primitives," CVPR 2023. [Explicit shared/private decoupling]
16. Liu et al., "PAFPN: Path Aggregation Network for Instance Segmentation," CVPR 2018. [Bottom-up FPN path]
17. Chen et al., "DeepLabv3: Rethinking Atrous Convolution for Semantic Image Segmentation," TPAMI 2018. [ASPP, dilated encoder]
18. Schoonbeek et al., "IndustReal: A Dataset for Assembly State Detection," WACV 2024. [Same dataset, PSR baseline]
19. Fan et al., "STORM-PSR: Spatio-Temporal Obfuscated Masking for PSR," WACV 2024. [Temporal PSR on IndustReal]
20. Li et al., "MViTv2: Improved Multiscale Vision Transformers," CVPR 2022. [MViTv2-S backbone architecture]
21. Kendall et al., "Multi-Task Learning Using Uncertainty to Weigh Losses," CVPR 2018. arXiv:1705.07115. [Uncertainty weighting for routing decisions]
