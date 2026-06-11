# POPW Project Journey — Complete Status Report
## For Opus Consultation v2 (2026-06-11)

---

## 1. Project Identity

**Project**: POPW — Pose-Conditioned Multi-Task Architecture for Assembly Understanding  
**Author**: Bashara Aina  
**Target Paper**: `popw_paper_improved.tex` — "POPW: A Unified Multi-Task Architecture for Egocentric Assembly Understanding"  
**Hardware**: Single NVIDIA RTX 3060 (12 GB VRAM), Intel i5-12400F, 32 GB RAM, Ubuntu 22.04  
**Framework**: PyTorch 2.2, CUDA 12.1, cuDNN 8.9  
**Primary Dataset**: IndustReal (egocentric RGB, 1280×720, 10 FPS)  
**Secondary Dataset**: IKEA ASM (third-person, 640×480, 3 RGB views)

---

## 2. What We Built (Architecture Overview)

### 2.1 Core Architecture
- **Backbone**: ConvNeXt-Tiny (ImageNet pretrained) → C2(96ch), C3(192ch), C4(384ch), C5(768ch)
- **FPN Neck**: P3–P7, 256 channels each, lateral 1×1 + top-down upsample + 3×3 smooth
- **Total Parameters**: 76.16M (53.42M trainable)
- **Input**: Single RGB frame [B, 3, 720, 1280]
- **Output**: 5 simultaneous task predictions in a single forward pass

### 2.2 Five Task Heads

| Head | Architecture | Output | Loss |
|------|-------------|--------|------|
| **Detection (ASD)** | RetinaNet-style, P3–P7, 4×Conv3×3+ReLU subnets | 24 classes × 9 anchors/location | Focal(α=0.25,γ=2) + GIoU |
| **Body Pose** | ConvTranspose2d + GroupNorm + soft-argmax | 17 keypoints + confidence | Wing Loss(ω=0.05,ε=0.005) |
| **Head Pose** | GAP(C4)‖GAP(C5) → MLP(1152→512→256→9) | 9-DoF (forward, position, up) | MSE × 0.001 |
| **Activity** | Feature Bank(T=16) + TCN + 2×ViT + CLS token | 75 classes (NA + 74 actions) | LDAM-DRW(s=30) |
| **PSR** | Multi-scale GAP → Causal Transformer(3L,4H) → 11 per-component MLPs | 11 binary components | Binary Focal(α=0.25,γ=1.0) + temporal smooth |

### 2.3 Cross-Task Conditioning (Key Innovation)
- **PoseFiLM**: Body keypoints → γ,β modulation of C5 features (bypasses FPN)
- **HeadPoseFiLM**: Head pose (stop_grad) → second-stage γ,β modulation
- **det_conf**: MaxPool(cls_preds) → [B,24] stop_grad → concatenated into activity input
- **FiLM γ constraint**: `1 + tanh(·)` ∈ (0,2) prevents feature inversion

### 2.4 Training Strategy
- **Kendall Homoscedastic Uncertainty**: L = Σ_t exp(-s_t)·L_t + s_t, init s_det=0, s_pose=-1, s_act=0, s_psr=0
- **Staged Training**: Stage 1 (det only, ep 1–5) → Stage 2 (+pose+headpose, ep 6–15) → Stage 3 (all heads, ep 16–100)
- **EMA**: decay=0.999, active from epoch 16
- **Optimizer**: AdamW, differential LR (backbone 0.1×, heads 1×, bias 0.3×)
- **Scheduler**: Warmup(5ep) → CosineAnnealing(T₀=10, T_mult=2)
- **Batch**: Physical=1, GradAccum=32 → Effective=32 (VRAM constraint with VideoMAE)

---

## 3. The Journey — Timeline of Events

### Phase 1: Initial Build (April–May 2026)
- Built the full POPW architecture from the paper specification
- Implemented all 5 heads, FPN, FiLM conditioning, Kendall weighting
- Implemented IndustReal dataset loader with all modalities (AR, ASD, PSR, head pose, hand joints)
- Frame cache system for RAM-based data loading (~5-7GB for full train set)
- VideoMAE-Small integration as optional second stream for activity

### Phase 2: First Training Attempts (May 2026)
- Initial runs showed immediate problems: detection mAP = 0.0, activity collapsed to single class
- PSR head frozen in stages 1-2 as designed, but never recovered in stage 3
- Multiple NaN crashes, gradient explosions, and silent failures
- Discovered Kendall log_var clamp was running AFTER backward (Bug #1)
- Discovered Stage 3 transition was resetting log_var to 0 (Bug #2)

### Phase 3: Debugging Marathon (Late May–Early June 2026)
- Identified 3-head collapse: det_mAP50=0.0, act_top1=0.0, psr_mAP50=0.0001
- Only pose head was producing non-zero metrics
- Extensive diagnostic scripts written (6+ diagnostic tools)
- Multiple "fix" attempts that masked symptoms rather than solving root causes
- NaN guards and loss caps added (which later proved to hide bugs)

### Phase 4: First Opus Consultation (June 8, 2026)
- Packaged code for Opus analysis
- Opus identified the superpowers redesign plan with 20 tasks
- CrossHeadCrossAttn module designed (det→act, pose→act attention)
- Bug catalogue expanded to 10 confirmed bugs

### Phase 5: The Collapse Crisis (June 9–10, 2026)
- Retrain from crash_recovery.pth (epoch 43) with --reinit-heads
- **Critical discovery**: best.pth contained EMA weights, not trained weights
- EMA shadow "reset" was a no-op: `ema.shadow[_n] = ema.shadow[_n].clone().detach()` (copies old shadow into itself)
- Collapsed checkpoint's EMA shadow was restored AFTER the reinit, overwriting fresh weights
- Detection trunk was NEVER re-initialized: code looked for `cls_tower/reg_tower` but model uses `cls_subnet/reg_subnet`
- Mixup/CutMix were mixing OUTPUT LOGITS, not inputs — coin-flip label corruption
- Eval collate dropped `clip_rgb` — half the activity input zeroed at eval
- `det_conf` was raw unbounded logits — dominated activity head input (L2 = 243.39 ± 0.001)
- ViT attention scaling was INVERTED: dividing by d^-0.5 multiplies by √d = 8 (64× too large)
- FeatureBank was dead: always returned current frame replicated 16× (video_ids=None)

### Phase 6: Current State (June 11, 2026)
- All root causes identified (RC-13 through RC-24, 12 new findings)
- Patches P1–P11 designed but NOT YET APPLIED to a clean retrain
- Config updated with Tier 1-3 improvements (simplified loss, ROI detector, PSR transition, K400 video stream, geometry-aware head pose, knowledge distillation, task-aware sampling)
- **The model has NEVER produced real, trustworthy multi-task metrics**
- We are at the "fix the measurement chain" stage — zero GPU-cost experiments remain

---

## 4. What Works

| Component | Status | Evidence |
|-----------|--------|----------|
| ConvNeXt-Tiny backbone | ✅ Working | ImageNet features load, FPN produces multi-scale features |
| FPN neck | ✅ Working | P3–P7 produce expected spatial dimensions |
| Head Pose head | ✅ Partially working | MAE ~0.344 (only living term in combined metric) |
| Body Pose head | ✅ Working | Soft-argmax produces keypoints, Wing Loss computes |
| Dataset loader | ✅ Working | All modalities load correctly, frame cache functional |
| Anchor generation | ✅ Working | Pixel-space xyxy, matched with GT format |
| Box encode/decode | ✅ Working | decode(encode(gt)) == gt verified |
| VideoMAE stream | ✅ Working | Loads VideoMAE-Small, produces 384-D features |

---

## 5. What Does NOT Work

| Component | Status | Root Cause |
|-----------|--------|------------|
| Detection head | ❌ Collapsed | EMA contamination (RC-13), trunk not reinit'd (RC-14), anchor/GT mismatch (RC-22) |
| Activity head | ❌ Collapsed | Mixup/CutMix label corruption (RC-15), inverted ViT scaling (RC-16), eval input mismatch (RC-17), dead FeatureBank (RC-18), det_conf domination (RC-19) |
| PSR head | ❌ Collapsed | Constant output (1 unique pattern), near-constant labels on small subset (RC-24) |
| Combined metric | ❌ Broken | Mathematically pose-only: 0.15/(1+0.344) = 0.1116 (RC-20) |
| EMA system | ❌ Broken | No-op reset, collapsed shadow restore, best.pth = EMA blend (RC-13) |
| Checkpoint selection | ❌ Broken | Driven by contaminated EMA metrics (RC-13) |
| Training subset | ❌ Insufficient | 5% = 4 recordings, 12/75 classes present (RC-24) |

---

## 6. Current Configuration State

The config.py has been updated with extensive Tier 1-3 improvements:
- `USE_SIMPLIFIED_LOSS = True` — replaces LDAM with CE+label_smoothing, fixed per-task weights
- `ASSERT_AND_CRASH = True` — no more silent NaN guards
- `USE_EMA = False` — disabled for recovery run
- `USE_MIXUP = False`, `CUTMIX_ALPHA = 0.0` — disabled until implementation fixed
- `USE_DET_SIGMOID_CONDITIONING = True` — sigmoid-bounded det_conf
- Tier 2.4: Embedding cache pipeline (configured but not yet used)
- Tier 2.5: ROI-centric detection (configured, `USE_ROI_DETECTOR = False`)
- Tier 2.7: PSR transition prediction (configured, `USE_PSR_TRANSITION = False`)
- Tier 2.8: K400 video stream (configured, `USE_K400_VIDEO_STREAM = False`)
- Tier 3.9: Knowledge distillation (configured, `USE_DISTILLATION = False`)
- Tier 3.11: Geometry-aware head pose (configured, `USE_GEO_HEAD_POSE = False`)
- Tier 3.12: Task-aware sampling (configured, `USE_TASK_AWARE_SAMPLING = False`)

---

## 7. What We Need From Opus

We need Opus to analyze our entire codebase and produce **at least 5 detailed implementation guides** (MD files) that we can directly implement. The guides should cover:

1. **How to fix the measurement chain** — the zero-GPU-cost experiments that tell us the truth
2. **How to redesign the architecture** for maximum multi-task learning (we are open to changing backbone, heads, training flow — anything)
3. **How to make each head actually learn** — specific loss functions, training strategies, data sampling
4. **How to beat the paper baselines** — concrete strategies for each metric
5. **How to make the unified model genuinely better than separate specialists** — the core thesis

We are NOT looking for incremental fixes. We want a fundamental redesign if needed. The goal is a model that LEARNS, not one that catastrophically fails.
