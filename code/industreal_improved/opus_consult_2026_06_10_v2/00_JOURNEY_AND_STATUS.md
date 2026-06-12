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

### Phase 6: Opus v2 Consultation (June 11, 2026)
- All root causes identified (RC-13 through RC-24, 12 new findings)
- Opus v2 answer received: identified RC-25 (feature-magnitude explosion in trunk) as the dominant failure mode
- Prescribed RC-25 (FPN reinit), RC-27 (GroupNorm in detection subnets), step-0 assertion, det_conf zeroing during recovery, EMA re-anchor, and D7-D9 diagnostics
- All Opus v2 prescriptions IMPLEMENTED in the codebase by June 11 evening

### Phase 7: RC-25 Recovery Attempts (June 11–12, 2026) — DEADLOCK DISCOVERED

**Run 1** (June 11, ~14:00–03:00): `--preset recovery --reinit-heads --subset-ratio 0.25 --max-epochs 45`
- Resumed from epoch-43 `latest.pth` (948MB), reinit FPN+heads
- Step-0 assertion PASSED: median |z| = 2.95 < 8
- Train loss non-zero, non-NaN (25–160 per batch)
- 2 epochs completed (44–45): **identical** validation results
- det_mAP50=0.0000, act_macro_f1=0.0007, psr_f1=0.0000, combined=0.1067
- Crash checkpoint saved at `crash_recovery.pth` (301MB)

**Run 2** (June 12, 07:21–10:56): `--preset recovery --resume crash_recovery.pth --subset-ratio 0.25 --max-epochs 55 --num-workers 0`
- NO --reinit-heads (continuing from 2-epoch trained state)
- 2 more epochs completed (46–47): **STILL identical** validation results
- **KILLED at epoch 47/55** — zero progress after 4 total epochs across both runs
- 4 validation cycles: identical to 4 decimal places (combined=0.1067)

**The 3-Way Deadlock (RC-28)**

Architecture coupling: Backbone+FPN → detection_head → det_conf → activity_head input

| Head | Status | Why Stuck |
|------|--------|-----------|
| Detection | Scores flat at 0.154 (std=0.0095) | Focal Loss at pi=0.05 with ~2.76M neg anchors per batch. Only 10–15% of batches have GT. Negative Focal Loss dominates gradient. |
| Activity | 1/75 classes (class 20, 100% of frames) | `ZERO_DET_CONF_FOR_RECOVERY=True` zeros det_conf input into activity head. Activity gets NO detection signal — can't differentiate frames. |
| PSR | 1 unique binary pattern | Depends on learned FPN features, but backbone gradient is dominated by detection's negative Focal Loss signal. |

**The recovery flag IS the deadlock**: `ZERO_DET_CONF_FOR_RECOVERY` was designed for the original collapse (saturated det_conf O(10-100) poisoning activity). But at pi=0.05 with healthy logits (Step-0 PASSED), it's **starving** the activity head. Without detection signal, activity can't learn. Without activity gradient, the shared backbone gets no useful signal for 2 of 3 tasks.

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

## 5. What Does NOT Work (Updated June 12)

| Component | Status | Root Cause |
|-----------|--------|------------|
| Detection head | ❌ Collapsed | **RC-28**: Focal Loss at pi=0.05 + 2.76M neg anchors per batch → equilibrium at score=0.154, no escape |
| Activity head | ❌ Collapsed | **RC-28**: ZERO_DET_CONF_FOR_RECOVERY=True starves activity of detection signal → 1/75 classes |
| PSR head | ❌ Collapsed | **RC-28**: Backbone gradient dominated by detection's negative Focal Loss → 1 binary pattern |
| Combined metric | ❌ Broken | act_top5=0.2425 + psr_edit=0.4773 only → combined=0.1067 (pose dead at MAE=66°) |
| RC-25 recovery strategy | ❌ Failed | FPN reinit + prior-based heads necessary but INSUFFICIENT — 3-way coupling deadlock prevents any head from escaping |
| EMA system | ✅ Fixed | USE_EMA=False, re-anchor after reinit implemented |
| Checkpoint selection | ✅ Fixed | Best measured from RAW model, not EMA |

---

## 6. Current Configuration State (As Tested June 12)

**Recovery preset used in both runs:**
```python
'recovery': {
    'dataset_mode': 'manual_only', 'backbone': 'convnext_tiny',
    'use_tma_cell': True, 'use_temporal_bank': True,
    'use_hand_film': True, 'benchmark_mode': False,
    'batch_size': 1, 'grad_accum_steps': 4,
    'zero_det_conf': True,     # ← THE DEADLOCK
    'staged_training': False,
    'mixed_precision': True,   # AMP for 12GB VRAM
    'use_mixup': False, 'use_ema': False,
},
```
- `EVAL_MAX_BATCHES = 75` (temp: limit val runtime)
- `CUDA_MEMORY_FRACTION = 0.80` (temp: 12GB VRAM)
- `ZERO_DET_CONF_FOR_RECOVERY = True` ← starves activity head
- `--subset-ratio 0.25` (25% of training data)
- `--reinit-heads` on Run 1 only

**Evidence in logs/ directory:**
- `recovery_train1_run1.log` — Run 1 (epochs 44–45, with --reinit-heads)
- `recovery_train2_run2.log` — Run 2 (epochs 46–47, without --reinit-heads)

---

## 7. What We Need From Opus v3

**The core question has changed.** We now know RC-25 was real (FPN reinit fixed step-0 saturation), but the recovery strategy has a **3-way coupling deadlock** that prevents any head from escaping zero.

We need Opus to:

1. **Verify the deadlock diagnosis (RC-28)** — is the 3-way coupling (det→act via ZERO_DET_CONF, det→backbone via Focal Loss mass, backbone→PSR) the correct explanation?

2. **Design an ESCAPE strategy** — how to break the deadlock so at least ONE head can reach non-zero metrics:
   - Should we disable ZERO_DET_CONF_FOR_RECOVERY immediately?
   - Should we train detection alone first (staged) before unfreezing activity/PSR?
   - Should we use a higher LR for detection head (1e-3 vs 1e-4) to escape Focal Loss equilibrium faster?
   - Should we reduce the negative anchor mass (e.g., use subsampling of negatives)?
   - Should we use a different prior (pi=0.01 instead of pi=0.05)?

3. **Prescribe concrete changes** — exact config values, training stages, LR schedule, number of epochs per stage

4. **Is a fresh ImageNet-init retrain better?** — Or can we salvage the epoch-43 lineage with the right escape strategy?

5. **Design a staged recovery protocol** — if coupled training can't work, what's the right sequence to bootstrap each head independently before joint training?
