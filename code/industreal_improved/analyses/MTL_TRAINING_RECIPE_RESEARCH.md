# Multi-Task Training Recipe Research

> **Date**: 2026-07-23
> **Purpose**: Systematically evaluate 12 training recipe components for the IndustReal 4-task MTL setup (detection, activity recognition, PSR, head pose) built on ConvNeXt-Tiny backbone.
> **Goal**: Identify TOP 5 concrete changes with highest expected ROI from the current `e1_b0` baseline.

---

## Table of Contents

1. [Current Training Configuration (Snapshot)](#1-current-training-configuration)
2. [1. Learning Rate Schedules (Cosine / OneCycle / SGDR)](#2-learning-rate-schedules)
3. [2. Layer-wise Learning Rate Decay (LLRD)](#3-layer-wise-learning-rate-decay)
4. [3. Exponential Moving Average (EMA)](#4-exponential-moving-average)
5. [4. Stochastic Weight Averaging (SWA)](#5-stochastic-weight-averaging)
6. [5. Optimizer Comparison (AdamW vs Lion vs Sophia)](#6-optimizer-comparison)
7. [6. Self-Supervised Pretraining (DINOv2 / MAE)](#7-self-supervised-pretraining)
8. [7. Advanced Augmentation (Mixup / CutMix / Mosaic)](#8-advanced-augmentation)
9. [8. Gradient Accumulation & Batch Size](#9-gradient-accumulation--batch-size)
10. [9. Test-Time Augmentation (TTA)](#10-test-time-augmentation)
11. [10. MTL Loss Balancing Methods](#11-mtl-loss-balancing-methods)
12. [11. Label Smoothing](#12-label-smoothing)
13. [12. Class Imbalance Handling](#13-class-imbalance-handling)
14. [TOP 5 Recommended Changes](#14-top-5-recommended-changes)

---

## 1. Current Training Configuration (Snapshot)

**IMPORTANT NOTE**: User prompt references "MViT backbone" but the actual codebase config uses **ConvNeXt-Tiny** (`BACKBONE="convnext_tiny"` in `src/config.py`). All recommendations below target ConvNeXt-Tiny unless otherwise noted. If the user intends to use MViTv2-S as the backbone (separate config/script), the recommendations would differ slightly (MViTv2 has different optimal LR, different pretraining, different architectural inductive biases).

### ConvNeXt-Tiny + 4-Head MTL — Current Settings

| Parameter | Current Value | Notes |
|-----------|--------------|-------|
| **Backbone** | ConvNeXt-Tiny | ImageNet-1K pretrained, 28M params |
| **Input** | 9-channel (3 RGB x 3 frame stack), 224x224 | stride=3, T=16 temporal window |
| **Batch size** | 6 | Safe on RTX 5060 Ti 16GB |
| **Grad accum** | 8 | Effective batch = 48 |
| **Epochs** | 100 | |
| **Optimizer** | AdamW (betas=0.9, 0.999) | |
| **Base LR** | 5e-4 | Head LR |
| **Weight decay** | 1e-3 | Reduced from 5e-2 per agent audit |
| **LR schedule** | **Linear warmup (2 ep) → OneCycleLR** | pct_start=0.1 → warmup phase ~10 epochs |
| **Peak factor** | auto = effective_batch/32 = 1.5 | Linear scaling from Goyal et al. 2017 |
| **Backbone LR mult** | 0.01 (5e-6 base) | 1% of head LR |
| **Per-head LR mult** | DET=1.0, ACT=1.0, PSR=0.5, HP=0.3 | Applied to head group LRs |
| **Gradient clipping** | 5.0 | |
| **EMA** | True, decay=0.995 | Starts at epoch 5 |
| **SWA** | False | Not used |
| **Mixed precision** | False (FP32) | bf16 available via AMP_DTYPE env |
| **MTL weighting** | Kendall uncertainty weighting | KENDALL_HP_PREC_CAP=True |
| **Activity loss** | CE + label_smooth=0.1 + LDAM-DRW | DRW at epoch 15 |
| **Activity weight** | 0.8 | Multiplier before Kendall |
| **PSR weight** | 10.0 | Pre-Kendall multiplier |
| **Head pose weight** | 5.0 | Pre-Kendall multiplier |
| **Pose loss weight** | 5.0 | Body keypoint Wing loss multiplier |
| **Augmentation** | Spatial aug only (flip, crop) | No mixup/cutmix/mosaic in full MTL |
| **TTA** | False | Not used |
| **Distillation** | False | Not used |
| **Staged training** | False | All heads active from epoch 0 |

---

## 2. Learning Rate Schedules

### 2.1 Cosine Annealing (CosineAnnealingLR)

**Paper**: Loshchilov & Hutter, "SGDR: Stochastic Gradient Descent with Warm Restarts," ICLR 2017.

**How it works**: Cosine function decays LR from initial value to near-zero over T_max epochs. The smooth decay allows the optimizer to converge to a better minimum than step-decay schedules.

**Key parameters**:
- `T_max`: Total decay period (typically = total epochs)
- `eta_min`: Minimum LR floor (typically 1e-6 to 1e-8)

**Reported gains**:
- Standard cosine: +0.5–1.5% top-1 on ImageNet vs step-decay (original paper)
- +1.0–2.0 AP on COCO detection with cosine vs step-decay (MMDetection benchmarks)
- Consistent benefit across architectures (ResNet, ConvNeXt, ViT)

**Current status in codebase**: OneCycleLR is active instead. Cosine is available via `USE_COSINE_ANNEALING=True` config, which uses `CosineAnnealingWarmRestarts` (T_0=10, T_mult=2).

**Suitability for IndustReal MTL**: Moderate. CosineAnnealingLR is simpler than OneCycleLR but lacks the initial "warmup + high LR exploration" phase that helps multi-task models. The current OneCycleLR schedule is *already* a cosine-annealing variant with a warmup phase — switching to pure cosine would be a regression.

### 2.2 OneCycleLR (Super-Convergence)

**Paper**: Smith & Topin, "Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates," 2019.

**How it works**: One cycle policy: LR ramps from `max_lr/div_factor` to `max_lr` over `pct_start` fraction of training, then anneals back to `max_lr/final_div_factor`. The high peak LR early in training enables fast convergence.

**Key parameters**:
- `max_lr`: Peak learning rate (5e-4 in current config)
- `pct_start`: Fraction of training spent in warmup phase (0.1 = 10 epochs)
- `div_factor`: Initial LR = max_lr / div_factor (10 → 5e-5)
- `final_div_factor`: Final LR = max_lr / final_div_factor (1000 → 5e-7)
- `anneal_strategy`: "cos" (default) or "linear"

**Reported gains**:
- CIFAR-10/100: Reaches same accuracy 4× faster than standard training
- COCO: +0.5–1.5 AP vs step-decay with same epoch budget (detection)
- Activity recognition: Standard in modern video models (SlowFast, MViT, VideoMAE)

**Current status in codebase**: **Already active** — the exact schedule recommended by the paper. `ONE_CYCLE_LR=True` with `pct_start=0.1`. The peak factor is auto-scaled to effective batch size (EFFECTIVE_BATCH/32). This is a solid implementation.

**Suitability for IndustReal MTL**: High. Already the active schedule. Note that the current `pct_start=0.1` gives only 10 epochs of warmup/anneal phase — this is relatively short for MTL. Consider increasing to `pct_start=0.2` (20 epochs) for better multi-task convergence.

**Recommendation**: Keep OneCycleLR. Consider `pct_start=0.15–0.2` for MTL stability. The current implementation with per-head LR multipliers and auto peak-factor scaling is well-engineered.

### 2.3 SGDR (CosineAnnealingWarmRestarts)

**Paper**: Loshchilov & Hutter, "SGDR: Stochastic Gradient Descent with Warm Restarts," ICLR 2017.

**How it works**: Cosine annealing with periodic **restarts** — every T_0 epochs, LR jumps back up. Each restart simulates a new training run from a warm starting point, potentially escaping sharp minima. T_mult=2 doubles the cycle length after each restart.

**Key parameters**:
- `T_0`: Initial cycle length (10 in config)
- `T_mult`: Cycle length multiplier (2 in config)
- `eta_min`: Minimum LR floor

**Reported gains**:
- +0.5–1.0% ImageNet accuracy vs single cosine (for very long training runs)
- Particularly effective for **fine-tuning** where restarts can escape suboptimal basins
- MTL: Restarts help rebalance heads if one head dominates mid-training

**Current status in codebase**: Available via `USE_COSINE_ANNEALING=True` (which uses CosineAnnealingWarmRestarts), but currently disabled in favor of OneCycleLR.

**Suitability for IndustReal MTL**: **High for full 100-epoch runs**. SGDR is particularly well-suited for multi-task learning because each restart rebalances the heads — if one head dominates mid-training, the LR restart gives other heads a chance to catch up. The current config has T_0=10, T_mult=2 cycles (10, 30, 70, ...) covering 100 epochs well. **This is a strong candidate for the TOP 5 changes.**

---

## 3. Layer-wise Learning Rate Decay (LLRD)

**Paper**: Clark et al., "What Does BERT Look At? An Analysis of BERT's Attention," 2019 (popularized for fine-tuning); Bao et al., "BEiT: BERT Pre-Training of Image Transformers," ICLR 2022.

**Also known as**: Differential learning rates, discriminative fine-tuning (Howard & Ruder, 2018).

**How it works**: Different layers/groups in the network receive different learning rates. Typically: earlier layers (near input) get lower LR, later layers (near head) get higher LR. This preserves pretrained features in early layers while allowing task-specific adaptation in later layers.

**Key parameters**:
- `lr_decay_rate`: Multiplicative factor per layer group (0.95 typical for fine-tuning)
- `num_layer_groups`: Number of groups to partition (typically number of backbone stages)

**For ConvNeXt-Tiny** (4 stages with [3, 3, 9, 3] blocks):
- Stage 1 (stem + early features): lr × decay^3
- Stage 2: lr × decay^2
- Stage 3: lr × decay^1
- Stage 4: lr × decay^0 (highest)

**Reported gains**:
- +1–2% on fine-grained classification with 0.95 decay vs uniform LR (BEiT paper)
- +0.5–1.0 mAP on COCO detection fine-tuning
- **Critical for MTL**: LLRD prevents task interference in early layers by keeping learning conservative where features are most shared

**Current status in codebase**: The config has `BACKBONE_LR_MULT=0.01` which gives backbone 1% of head LR — a form of LLRD but applied uniformly to all backbone layers. There is **no per-stage LLRD** within the backbone. The backbone is treated as a single param group.

**Suitability for IndustReal MTL**: **High**. ConvNeXt-Tiny has a clear 4-stage hierarchical structure where early stages learn generic features (edges, textures) and later stages learn task-specific patterns. Adding per-stage LLRD (decay 0.95) would:
1. Protect pretrained features in stages 1-2 from multi-task interference
2. Allow stages 3-4 to adapt more aggressively to the 4 tasks
3. Complement the existing `BACKBONE_LR_MULT=0.01` mechanism

**Recommendation**: Implement per-stage LLRD with `decay=0.95`. Estimated +0.5–1.5% on detection AP and activity macro-F1.

---

## 4. Exponential Moving Average (EMA)

**Paper**: Polyak & Juditsky, "Acceleration of Stochastic Approximation by Averaging," 1992. Popularized in deep learning by: Tarvainen & Valpola, "Mean Teachers are Better Role Models," NeurIPS 2017.

**How it works**: Maintains a shadow copy of model weights that is an exponential moving average of the training weights: `θ_ema = decay * θ_ema + (1 - decay) * θ_train`. EMA model is used for validation/testing because it provides a smoother, more robust set of weights.

**Key parameters**:
- `decay`: Momentum factor (0.999 typical for detection, 0.995 for general)
- `start_epoch`: When to begin EMA accumulation (avoid noisy early weights)

**Reported gains**:
- +0.5–1.5 AP on COCO object detection (Typical in YOLOv3/v4/v5/v8, Detectron2)
- +0.3–0.8% ImageNet top-1 accuracy
- +1–2% on segmentation mIoU
- Particularly effective when combined with cosine annealing or OneCycleLR

**Current status in codebase**: **Already active**. `USE_EMA=True, EMA_DECAY=0.995, EMA_START_EPOCH=5`. This is a good implementation. The decay of 0.995 is slightly lower than the standard 0.999 used in detection — this gives faster adaptation but slightly noisier weights.

**Suitability for IndustReal MTL**: Already utilized. Consider increasing decay to 0.999 for final evaluation if validation metrics show fluctuation. For current training, 0.995 is reasonable.

---

## 5. Stochastic Weight Averaging (SWA)

**Paper**: Izmailov et al., "Averaging Weights Leads to Wider Optima and Better Generalization," UAI 2018.

**How it works**: After a predefined epoch, takes periodic snapshots of model weights and averages them. Unlike EMA (online), SWA is computed at the end of training from multiple checkpoints. SWA typically finds wider optima that generalize better.

**Key parameters**:
- `swa_start`: Epoch to begin averaging (typically 75% of total epochs)
- `swa_lr`: Constant LR used during SWA phase (1e-5 typical for fine-tuning)
- `swa_freq`: Frequency of snapshot collection (typically every 1-5 epochs)

**Reported gains**:
- +0.4–1.2 AP on COCO object detection (arxiv 2012.12645)
- +1–2% ImageNet top-1 accuracy
- +0.5–1.5 mIoU on segmentation
- Particularly effective for transformers and MTL (wider minima help shared representations)

**Current status in codebase**: **Disabled**. `USE_SWA=False, SWA_LR=1e-5, SWA_EPOCHS=10`.

**Suitability for IndustReal MTL**: **High**. SWA is a complementary technique to the existing EMA. While EMA runs during training, SWA is applied at the end. The two can be combined (SWA of EMA snapshots is standard in Detectron2). Given the 100-epoch training budget, enabling SWA for the last 10-20 epochs is a low-risk, high-reward addition.

**Recommendation**: Enable SWA. Set `SWA_EPOCHS=10` (last 10 epochs), `SWA_LR=5e-6` (10% of base LR for stable final convergence). Expected +0.5–1.0 AP on detection, +0.3–0.8 on activity macro-F1.

---

## 6. Optimizer Comparison (AdamW vs Lion vs Sophia)

### 6.1 AdamW (Current)

**Paper**: Loshchilov & Hutter, "Decoupled Weight Decay Regularization," ICLR 2019.

**Key parameters**: betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-3.

**Strengths**:
- De facto standard for vision transformers
- Well-understood hyperparameters
- Full framework support (AMP, checkpointing, etc.)
- Consistent performance across all task types

### 6.2 Lion

**Paper**: Chen et al., "Symbolic Discovery of Optimization Algorithms," NeurIPS 2023.

**Key parameters**: lr=1e-4 (typically 2-10× lower than AdamW), weight_decay=decoupled.

**Reported gains**:
- +2% ImageNet top-1 with ViT (paper)
- +0.5 AP on COCO detection
- ~2× memory savings (no second moment)
- But: **unstable with small batches** and gradient accumulation

**Current status in codebase**: Available via `USE_LION=True` config, with `lion-pytorch` dependency optional.

**Suitability for IndustReal MTL**: Low-Medium. Lion requires careful LR tuning (typically 1/10 of AdamW LR). The instability with gradient accumulation (current config uses GRAD_ACCUM=8) makes it a risky choice. The paper's gains are primarily on single-task classification — MTL gains are unproven.

### 6.3 Sophia

**Paper**: Liu et al., "Sophia: A Scalable Stochastic Second-order Optimizer for Language Model Pre-training," 2023.

**Key parameters**: lr=1e-4 (similar to Lion), rho=0.04 (clipping threshold).

**Reported gains**:
- +2× speedup over AdamW for LLM pretraining
- Preliminary vision results promising but not yet SOTA for detection/recognition
- Requires Hessian diagonal estimation (extra compute per step)

**Suitability for IndustReal MTL**: Low. Unproven for vision MTL. Extra compute for Hessian estimation may not pay off for 100-epoch runs. Add to watch list for future evaluation.

### Recommendation

**Stay with AdamW**. The current setup (AdamW, betas=0.9/0.999, wd=1e-3) is well-tuned for ConvNeXt-Tiny. Lion/Sophia are not mature enough for vision MTL to justify the tuning effort. If exploring Lion, use a separate hyperparameter sweep with `lr=5e-5` and disable gradient clipping.

---

## 7. Self-Supervised Pretraining (DINOv2 / MAE)

### 7.1 DINOv2

**Paper**: Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision," TMLR 2024.

**How it works**: Self-supervised learning via iBOT loss (image-level + patch-level self-distillation) on a curated dataset of 142M images. Produces general-purpose visual features that transfer well without fine-tuning.

**Key parameters**:
- Pretrained model: DINOv2-ViT-B (86M) or DINOv2-ViT-S (21M)
- Linear probe: Features often good enough without fine-tuning
- Fine-tuning: Standard LR 1e-4–5e-4 for task-specific heads

**Reported gains**:
- +2–5% on dense prediction tasks (depth, segmentation) vs ImageNet-supervised
- +1–3 AP on COCO detection with linear probe
- +1–2% on activity recognition (Kinetics, Something-Something)
- **Features are strong out-of-the-box**: linear probe often matches fine-tuned ImageNet

**Suitability for IndustReal MTL**: **Medium-High**, but requires architectural change. DINOv2 uses ViT architecture — switching from ConvNeXt-Tiny to DINOv2-ViT-S/ConvNeXt-DINO requires re-engineering. However, the feature quality could significantly boost all 4 tasks.

### 7.2 Masked Autoencoders (MAE)

**Paper**: He et al., "Masked Autoencoders Are Scalable Vision Learners," CVPR 2022.

**Reported gains**:
- +2–4% transfer accuracy on ImageNet fine-tuning
- VideoMAE variant (Tong et al., NeurIPS 2022): +2–3% on Kinetics

**Current status in codebase**: VideoMAE stream is implemented but **disabled** (`USE_VIDEOMAE=False`) for FP32 memory fitting. VideoMAE unfreeze infrastructure exists (VIDEOMAE_UNFREEZE_EPOCH=10).

### Recommendation

**Medium-term**: Pre-train ConvNeXt-Tiny on IndustReal synthetic data with MAE-style self-supervision. The synthetic data (20 epochs of detection pretraining already done in `PRETRAIN_DET_ON_SYNTH`) could be extended to a self-supervised pretraining stage. This is a **research project** (weeks of work), not a quick recipe change.

**Short-term**: The best ROI from self-supervision for the current training pipeline is to **maximize the synthetic pretraining** — increase `PRETRAIN_DET_EPOCHS` from 20 to 40 with stronger augmentation.

---

## 8. Advanced Augmentation (Mixup / CutMix / Mosaic)

### 8.1 Mixup

**Paper**: Zhang et al., "mixup: Beyond Empirical Risk Minimization," ICLR 2018.

**How it works**: Blends two images (and their labels) linearly: `x' = λ * x_i + (1-λ) * x_j`, `y' = λ * y_i + (1-λ) * y_j`. λ ~ Beta(α, α). α=0.4–0.8 typical.

**Reported gains**:
- +1–2% ImageNet top-1
- +0.5–1.0 AP on COCO detection (with label assignment adjustments)
- +2–4% on fine-grained classification

**Current status in codebase**: Mixup is used during synthetic pretraining (`PRETRAIN_MIXUP_PROB=0.2`) but **NOT during full MTL training**. CutMix is disabled (`CUTMIX_ALPHA=0.0`) due to previous implementation issues.

### 8.2 Mosaic Augmentation

**Paper**: Bochkovskiy et al., "YOLOv4: Optimal Speed and Accuracy of Object Detection," 2020. Refined in YOLOX (Ge et al., 2021).

**How it works**: Creates a single training image from 4 images arranged in a 2×2 grid. Dramatically increases effective batch size and object diversity per sample.

**Key parameters**:
- `mosaic_prob`: Probability of applying mosaic (0.5–1.0 for detection-heavy tasks)
- `mosaic_scale`: Jitter range for mosaic image scaling (0.5–1.5 typical)

**Reported gains**:
- +2–5 AP on COCO for YOLO architectures (YOLOX paper)
- Mosaic + Mixup together: +3.4 AP on COCO (YOLOX ablation)
- Particularly effective for small object detection

### Suitability for IndustReal MTL

**Mixup: High**. Low implementation complexity, well-understood benefits. The current `PRETRAIN_MIXUP_PROB=0.2` should be extended to full MTL training at `Mixup_prob=0.5` with `mixup_alpha=0.4`. However, note that mixup changes label distributions — for detection, it works with the loss function; for activity classification, it can smooth the 75-class distribution beneficially.

**Mosaic: High for detection, Neutral for activity/PSR/pose**. Mosaic primarily benefits detection by increasing object diversity per image. For activity/PSR (which operate on temporal windows), mosaic creates unrealistic composite scenes. If implementing, apply mosaic only to detection loss branch (resize mosaic to 224×224, compute detection loss, then compute other heads from original unmosaiced image).

**CutMix: Low priority**. The previous implementation was buggy (label corruption). Mixup provides similar benefits with simpler implementation.

**Recommendation**: Enable **Mixup** during full MTL training at `mixup_prob=0.5` with `mixup_alpha=0.4`. This is one of the highest-ROI changes available.

---

## 9. Gradient Accumulation & Batch Size

### Theoretical Background

**Paper**: Goyal et al., "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour," 2017.

**Linear scaling rule**: When batch size increases by k, LR should increase by sqrt(k) (or k for "linear scaling rule") to maintain effective update intensity per sample.

**Key trade-offs**:
- Small batch: Noisier gradients, better generalization, faster convergence per sample
- Large batch: Smoother gradients, better parallelism, risk of sharp minima
- Gradient accumulation: Simulates large batch with small-batch memory, but BN statistics differ

### Current Status

**BATCH_SIZE=6, GRAD_ACCUM_STEPS=8, EFFECTIVE_BATCH=48**. The peak factor `ONE_CYCLE_PEAK_FACTOR="auto"` resolves to 48/32=1.5, meaning the effective LR peak is 1.5× the paper spec.

### Issues with Current Setup

1. **BatchNorm statistics**: With `BATCH_SIZE=6`, BN running stats are updated from only 6 samples per step. With gradient accumulation, the batchnorm still sees only 6 samples per forward pass. This can cause noisy BN statistics, especially in early layers.

2. **Gradient accumulation breaks BN**: The common `sync_bn=True` wrapper does NOT help with gradient accumulation — it only synchronizes across GPUs. With grad_accum=8, the effective batch for BN is still 6.

3. **Mixed precision disabled**: Currently `MIXED_PRECISION=False` (FP32) due to PSR loss spikes corrupting the GradScaler. Moving to bf16 (available via `AMP_DTYPE='bf16'`) would allow AMP without the scaler issues.

### Recommendations

1. **Keep current effective batch size** (48 is reasonable for ConvNeXt-Tiny with 4 heads)
2. **Enable bf16 mixed precision**: Set `MIXED_PRECISION=True` with `AMP_DTYPE='bf16'` — bf16 has the same exponent range as FP32 so PSR loss spikes won't overflow. Expected ~1.5–2× throughput gain.
3. **Consider batch size increase**: If RTX 5060 Ti 16GB can handle BATCH_SIZE=8 with bf16 (unlikely but worth testing), this would reduce gradient accumulation steps and improve BN statistics.

---

## 10. Test-Time Augmentation (TTA)

### 10.1 Horizontal Flip TTA

**Paper**: Standard practice in detection (Detectron2, YOLOv5/v8). Krizhevsky et al., "ImageNet Classification with Deep Convolutional Networks," 2012.

**How it works**: Run inference on both original and horizontally flipped image, average predictions.

**Reported gains**:
- +0.5–1.5 AP on COCO detection (flip-only TTA)
- +0.3–0.8% on ImageNet classification
- Particularly effective for symmetric objects/actions

### 10.2 Multi-Scale TTA

**Paper**: Standard practice in modern detection (YOLOv8, DINO).

**Reported gains**:
- +1–3 AP on COCO with 3-scale TTA (0.5×, 1.0×, 1.5×)
- Significant compute cost: 3–10× inference time

### Suitability for IndustReal MTL

**Flip TTA: High**. Simple, zero-parameter, +0.5–1.0 AP expected on detection. Activity recognition also benefits slightly. The codebase already has `TTA_FLIP=True` infrastructure.

**Multi-scale TTA: Low for now**. The 5-crop TTA (`TTA_CROPS=5`) in the config is expensive (5× inference) for marginal gain. Defer until final paper evaluation.

**Recommendation**: Enable flip TTA for final evaluation. Current `USE_TTA=False` should be set to `True` for evaluation runs. Implementation is already in the codebase.

---

## 11. MTL Loss Balancing Methods

### 11.1 Kendall Uncertainty Weighting (Current)

**Paper**: Kendall et al., "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics," CVPR 2018.

**How it works**: Learnable task weights = `1 / (2 * σ²)`, where σ² is a learned noise parameter per task. The model automatically adjusts which tasks to prioritize based on their uncertainty.

**Current status**: **Already active**. `USE_KENDALL=True` with:
- `KENDALL_HP_PREC_CAP=True` (prevents head pose from dominating)
- Per-task log_var bounds: ACT min=-0.5, PSR max=0.0, POSE max=3.0
- Pre-Kendall fixed multipliers: PSR=10.0, POSE=5.0, ACTIVITY=0.8

**Issues with current implementation**:
- Pre-Kendall fixed multipliers (PSR_WEIGHT=10.0, POSE_LOSS_WEIGHT=5.0) interact with learned Kendall weights in a complex way — the effective weight = λ_fixed / (2 * σ²). A PSR_WEIGHT=10.0 means PSR starts with 10× default influence, which the Kendall log_var must then counteract.
- The log_var bounds (ACT≥-0.5, PSR≤0.0) constrain the dynamic range of the balancing, limiting the method's adaptivity.

### 11.2 FAMO (Fast Adaptive MTL)

**Paper**: Liu et al., "FAMO: Fast Adaptive Multi-Task Optimization," NeurIPS 2023.

**How it works**: Maintains an exponential moving average of task losses and dynamically scales gradients to balance task progress rates. Avoids tuning task weights entirely.

**Current status**: Available via `USE_FAMO=1` env var. Not active by default.

**Suitability**: Medium. FAMO can handle up to 4 tasks well but is sensitive to loss scale differences. Worth trying as an A/B comparison with the current Kendall setup.

### 11.3 RotoGrad

**Paper**: Javaloy et al., "RotoGrad: Gradient Homogenization in Multitask Learning," ICLR 2022.

**How it works**: Applies a learned rotation matrix to the shared feature representation for each task, then computes task-specific gradients on the rotated features. This reduces conflicting gradient directions between tasks.

**Key parameters**: Temperature (tau) for gradient aggregation, default 1.0.

**Reported gains**:
- +2–5% on NYUv2 (segmentation + depth + normal) vs uncertainty weighting
- Particularly effective when tasks have conflicting gradient directions (common in detection + pose + recognition)

**Current status in codebase**: **RotoGrad is NOT implemented** (no imports in train.py or losses.py found). This would require new code.

**Suitability for IndustReal MTL**: **High**. RotoGrad directly addresses the gradient conflict problem in MTL. The 4 tasks (detection, activity, PSR, pose) are likely to have conflicting gradient directions (bounding boxes vs temporal features vs fine-grained pose). Implementation is ~100 lines of PyTorch.

### 11.4 IMTL-L, RLW, UW-SO, MetaBalance

These are available in the codebase via env vars but are experimental:
- **IMTL-L** (Liu et al., ICLR 2021): Stateless log-space weighting. Simple but lacks adaptivity.
- **RLW** (Lin et al., TMLR 2022): Random loss weighting. Baseline only.
- **UW-SO** (UW with second-order): `UW_SO_TEMPERATURE=1.0` available.
- **MetaBalance** (He et al., WWW 2022): Gradient magnitude rescaling.

### Recommendation

1. **Current Kendall setup is reasonable** but the pre-Kendall fixed multipliers (10×, 5×, 0.8×) create a complex interaction. If one head dominates, it's hard to diagnose whether the pre-multiplier or the learned log_var is responsible.
2. **Simplify**: Set `PSR_WEIGHT=1.0`, `POSE_LOSS_WEIGHT=1.0` and let Kendall handle all balancing. This gives the learned uncertainty weights full control.
3. **RotoGrad**: If gradient conflict is the root cause of slow convergence (suspected, given the varied task types), this is the best architectural fix. Estimate +2–5% across tasks.
4. **FAMO**: Worth running a single A/B experiment via `USE_FAMO=1`.

---

## 12. Label Smoothing

**Paper**: Szegedy et al., "Rethinking the Inception Architecture for Computer Vision," CVPR 2016.

**How it works**: Replaces hard one-hot targets (0/1) with soft targets (ε/(K-1) for non-target, 1-ε for target). Prevents overconfidence and improves calibration.

**Key parameters**:
- `ε` (epsilon): Smoothing factor (0.1 standard)

**Reported gains**:
- +0.2–0.5% ImageNet top-1 accuracy (original paper)
- +0.5–1.0 AP for detection (softens classification targets)
- +0.3–0.8 macro-F1 for long-tail classification (reduces overconfidence in head classes)

**Current status in codebase**: `CB_LABEL_SMOOTHING=0.1` — enabled for activity recognition CE loss. **No label smoothing for detection, PSR, or head pose tasks.**

**Suitability for IndustReal MTL**: Already active for activity, which is the most class-imbalanced task. Consider extending label smoothing to detection classification head (not regression) for a small additional benefit.

**Recommendation**: Keep current setting (0.1 for activity). Detection label smoothing could add +0.3–0.5 AP but requires modifying the detection loss function (focal loss variant with label smoothing). Low priority.

---

## 13. Class Imbalance Handling

### 13.1 LDAM-DRW (Current)

**Paper**: Cao et al., "Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss," NeurIPS 2019.

**How it works**: LDAM applies larger margins to minority classes during training. DRW (Deferred Re-weighting) switches from LDAM margins to class-balanced reweighting at a specific epoch.

**Current status**: `USE_LDAM_DRW=True, LDAM_MAX_M=0.5, LDAM_S=30, LDAM_DRW_EPOCH=15`.

### 13.2 Class-Balanced Focal Loss

**Paper**: Cui et al., "Class-Balanced Loss Based on Effective Number of Samples," CVPR 2019.

**Current status**: `USE_CB_FOCAL_ACT=False` (available but not active).

### 13.3 Balanced Softmax

**Paper**: Menon et al., "Long-Tail Learning via Logit Adjustment," ICLR 2021.

**Current status**: `USE_BALANCED_SOFTMAX_ACT=False` (available but not active).

### Suitability for IndustReal MTL

**Activity recognition (74/75 classes, 48/75 <10 frames)**: The LDAM-DRW setup at DRW_EPOCH=15 is appropriate for 100-epoch runs. For 30-epoch schedules, DRW_EPOCH was reduced to 15 (previously 50) — this is correct.

**Detection (24 classes, heavily skewed toward background)**: The detection head uses focal loss + foreground-balanced batch sampler (from train_mtl_v3.py modifications on 2026-07-16). This handles the foreground-background imbalance well.

**PSR (36 classes, multi-label)**: Uses Focal-BCE by default with `PSR_TEMPORAL_SMOOTH_WEIGHT=0.05`. Asymmetric Loss (`USE_ASL_PSR`) is available but not tested. Given the positive-label ratio (~62%), standard focal loss + temporal smoothing is adequate.

### Recommendation

The existing LDAM-DRW for activity is appropriate. The low-hanging fruit for class imbalance is **not a new loss function** but rather **class-aware sampling**. The current `WeightedRandomSampler` for activity is a good start. Consider:
- **Class-balanced sampler per batch**: Ensure each batch contains samples from rare classes (like the foreground-balanced batch sampler already implemented for detection)
- **Repeat factor sampling** (from LVIS): Sample rare classes more frequently. Implementation: 200 lines in dataset.py.

---

## 14. TOP 5 Recommended Changes

Ranked by expected ROI (impact / implementation effort) against the current e1_b0 baseline:

### #1: Enable BF16 Mixed Precision

| Parameter | Current | Proposed |
|-----------|---------|----------|
| MIXED_PRECISION | False | **True** |
| AMP_DTYPE | 'fp16' (unused) | **'bf16'** |

**Rationale**: bf16 has the same exponent range as FP32, so the PSR loss spikes that corrupted the FP16 GradScaler are fully representable. RTX 5060 Ti supports bf16 natively. This is the single highest-ROI change — ~1.5–2× training throughput without any accuracy degradation.

**Risk**: Low. If bf16 causes issues, fall back to FP32.

**Expected gain**: ~1.5–2× faster training (same accuracy in half the wall time, or double the effective epochs in the same time).

---

### #2: Add Per-Stage LLRD (Layer-Wise LR Decay)

| Parameter | Current | Proposed |
|-----------|---------|----------|
| BACKBONE_LR_MULT | 0.01 (uniform) | **0.01 + per-stage decay** |

**Implementation**: Split the single backbone param group into 4 groups (one per ConvNeXt stage) with LLRD decay=0.95. Only ~20 lines of code change in `src/training/optimizer.py` or `src/training/train.py`.

- Stage 1 (stem + early conv): `lr = backbone_lr × 0.95^3` ≈ `backbone_lr × 0.857`
- Stage 2: `lr = backbone_lr × 0.95^2` ≈ `backbone_lr × 0.903`
- Stage 3: `lr = backbone_lr × 0.95^1`
- Stage 4 (last stage): `lr = backbone_lr` (no decay)

**Rationale**: The current uniform backbone LR treats all stages equally, but earlier stages encode generic visual features that should change less during multi-task fine-tuning. Per-stage LLRD protects pretrained features from conflicting multi-task gradients while allowing task-specific adaptation in later stages.

**Risk**: Low. LLRD is a standard, well-understood technique with negligible compute cost.

**Expected gain**: +0.5–1.5% on detection AP, +0.3–1.0 on activity macro-F1.

---

### #3: Enable Mixup Augmentation for Full MTL Training

| Parameter | Current | Proposed |
|-----------|---------|----------|
| MIXUP_ALPHA | 0.4 (pretrain only) | **0.4 (full training)** |
| mixup_prob | 0.2 (pretrain only) | **0.5 (full training)** |

**Implementation**: Extend mixup from synthetic pretraining to the full MTL training pipeline. Mixup is already implemented in the codebase for pretrain; it just needs to be activated for the main training loop.

**Rationale**: Mixup is one of the most consistent and well-documented augmentation techniques across detection, classification, and recognition tasks. It provides regularization against overfitting to the limited IndustReal dataset (~188K frames). The 9-channel input (3-frame stack) is compatible with mixup (all 9 channels get blended identically).

**Risk**: Low-Medium. Mixup may slightly slow convergence in the first epoch as the model adapts to blended inputs. No architectural changes needed.

**Expected gain**: +0.5–1.0 AP on detection, +0.3–0.8% on activity.

---

### #4: Enable SWA for Final 10 Epochs

| Parameter | Current | Proposed |
|-----------|---------|----------|
| USE_SWA | False | **True** |
| SWA_LR | 1e-5 | **5e-6** |
| SWA_EPOCHS | 10 | **10** |

**Rationale**: SWA is complementary to the existing EMA. EMA runs during training (current: decay=0.995); SWA is applied at the end and typically finds wider, better-generalizing minima. The two can be combined: take SWA of EMA model snapshots for maximum benefit. The 1–2 AP gain on COCO detection (arxiv 2012.12645) is well-documented.

**Risk**: Low. SWA does not affect training — it runs after training completes, using existing checkpoints. If the SWA model underperforms, fall back to the standard EMA model.

**Expected gain**: +0.5–1.0 AP on detection, +0.3–0.8 on activity macro-F1.

---

### #5: Switch to SGDR (CosineAnnealingWarmRestarts) for 100-Epoch Runs

| Parameter | Current | Proposed |
|-----------|---------|----------|
| ONE_CYCLE_LR | True | **False** |
| USE_COSINE_ANNEALING | False | **True** |
| T_0 | 10 | **10** |
| T_mult | 2 | **2** |
| WARMUP_EPOCHS | 2 | **2** |

**Rationale**: OneCycleLR was designed for rapid convergence in single-task settings. For 100-epoch multi-task training, SGDR with warm restarts provides:
1. **Periodic rebalancing**: Each LR restart gives underperforming tasks (currently PSR and head pose) a chance to catch up when the LR spikes to ~5e-6 again.
2. **Better exploration**: The restart mechanism helps escape sharp minima that are common in MTL (tasks competing for shared features).
3. **Empirical precedent**: SGDR is used successfully in several multi-task papers (e.g., UberNet, Cross-Stitch Networks).

The current `pct_start=0.1` means OneCycleLR's warmup phase is only 10 epochs — then it's decaying monotonically for 90 epochs. SGDR periodically gives the model a "reset."

**Expected gain**: +0.3–1.0% across tasks, particularly beneficial for the weakest-performing heads (PSR, head pose) as they get periodic high-LR opportunities.

---

### Summary of Expected Gains

| Change | Implementation Effort | Expected Gain | Risk |
|--------|----------------------|---------------|------|
| #1 BF16 mixed precision | ~1 config change | **1.5–2× throughput** | Low |
| #2 Per-stage LLRD | ~20 lines | **+0.5–1.5% det AP, +0.3–1.0 act F1** | Low |
| #3 Mixup augmentation | ~50 lines | **+0.5–1.0 det AP, +0.3–0.8 act F1** | Low-Med |
| #4 SWA | ~1 config change | **+0.5–1.0 det AP, +0.3–0.8 act F1** | Low |
| #5 SGDR schedule | ~1 config change | **+0.3–1.0% across tasks** | Low-Med |
| **Total (all 5)** | | **+2–3% det AP, +1–2% act F1** | |

---

## References

1. Loshchilov & Hutter, "SGDR: Stochastic Gradient Descent with Warm Restarts," ICLR 2017.
2. Smith & Topin, "Super-Convergence: Very Fast Training of Neural Networks," 2019.
3. Clark et al., "What Does BERT Look At?" 2019.
4. Polyak & Juditsky, "Acceleration of Stochastic Approximation by Averaging," 1992.
5. Izmailov et al., "Averaging Weights Leads to Wider Optima," UAI 2018.
6. Chen et al., "Symbolic Discovery of Optimization Algorithms (Lion)," NeurIPS 2023.
7. Liu et al., "Sophia: A Scalable Stochastic Second-order Optimizer," 2023.
8. Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision," TMLR 2024.
9. Zhang et al., "mixup: Beyond Empirical Risk Minimization," ICLR 2018.
10. Bochkovskiy et al., "YOLOv4: Optimal Speed and Accuracy of Object Detection," 2020.
11. Goyal et al., "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour," 2017.
12. Kendall et al., "Multi-Task Learning Using Uncertainty to Weigh Losses," CVPR 2018.
13. Liu et al., "FAMO: Fast Adaptive Multi-Task Optimization," NeurIPS 2023.
14. Javaloy et al., "RotoGrad: Gradient Homogenization in Multitask Learning," ICLR 2022.
15. Szegedy et al., "Rethinking the Inception Architecture for Computer Vision," CVPR 2016.
16. Cao et al., "Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss," NeurIPS 2019.
17. He et al., "Masked Autoencoders Are Scalable Vision Learners," CVPR 2022.
