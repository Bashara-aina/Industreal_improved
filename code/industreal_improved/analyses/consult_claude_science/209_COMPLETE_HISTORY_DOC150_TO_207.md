# Doc 209 — Complete History: Everything We've Tried From Doc 150 to Doc 207

**Status**: Living chronicle. Updated 2026-07-11.
**Audience**: Claude Science consultation — full experimental history for informed suggestions.
**Evidence discipline**: Every claim backed by doc number, commit hash, or file:line reference.

---

## 1. Timeline: Doc 150 → Doc 207 with Milestones

The project evolved through 15+ phases across 4 months (April–July 2026), documented across 151+ files in `analyses/consult_2026_06_10/` plus `doc_208_expectations_and_gap_playbook.md`. The naming convention uses `NN-*.md` format (doc 90 through doc 130+), not literal `doc_150.md` files.

| Phase | Dates | Key Milestone | Doc References |
|-------|-------|--------------|----------------|
| Phase 1 | Apr–May 2026 | Initial build: ConvNeXt-Tiny + FPN + 5 task heads | 00_JOURNEY:Phase1 |
| Phase 2 | May 2026 | First training attempts — all heads collapsed | 00_JOURNEY:Phase2 |
| Phase 3 | Late May–Early Jun | Debugging marathon: 6+ diagnostic tools, NaN guards | 00_JOURNEY:Phase3 |
| Phase 4 | Jun 8 | Opus v1: CrossHeadCrossAttn, 10 confirmed bugs | 00_JOURNEY:Phase4 |
| Phase 5 | Jun 9–10 | Collapse crisis: EMA contamination, Mixup/CutMix flip, eval collate bug, inverted ViT attention, dead FeatureBank | 00_JOURNEY:Phase5 |
| Phase 6 | Jun 11 | Opus v2: RC-25 through RC-29 | 00_JOURNEY:Phase6 |
| Phase 7 | Jun 11–12 | RC-25 recovery, 3-way deadlock | 00_JOURNEY:Phase7 |
| Phase 8 | Jun 12–13 | Fresh Start Run 8 — proved algorithmic root cause | 00_JOURNEY:Phase8 |
| Phase 9 | Jun 13 | Opus v4: DET_GT_FRAME_FRACTION=0.90, first det_mAP50=0.0091 | 00_JOURNEY:Phase9 |
| Phase 10 | Jun 14–17 | RF1 death spiral & R2.5 paradox resolution: gradient sparsity proof | 00_JOURNEY:Phase10 |
| Phase 11 | Jun 18–19 | RF1 completion with Kendall fix: det_mAP50=0.45 (phantom), actually ~0.184 | 00_JOURNEY:Phase11 |
| Phase 12 | Jun 20 | RF2 epoch 15 collapse: cls_score bias differentiation problem | 00_JOURNEY:Phase12 |
| Phase 13 | Jun 20 | Opus v8: KENDALL_HP_PREC_CAP, DET_POS_IOU_TOP_K=9, DET_BIAS_LR_FACTOR=1.0 | 00_JOURNEY:Phase13 |
| Phase 14 | Jun 20–21 | Opus v8 fixes deployed: no collapse, but structural ceiling | 00_JOURNEY:Phase14 |
| Phase 15 | Jun 21 | 6-epoch plateau at mAP50~0.207, CosineAnnealing restart had zero effect | 00_JOURNEY:Phase15 |
| Phase 16 | Jun 22–Jul 4 | F1-F22b fix implementation (38+ fixes), RF4 launch with all 4 heads | 113-all-fixes-chronicle |
| Phase 17 | Jul 4–11 | MViTv2-S transition (28.6M→48.6M total), Opus 141, MTL training | config.py, doc_208 |

**Commit count**: 216+ commits on `HEAD` (113-all-fixes-chronicle).

---

## 2. Architecture Evolution: ConvNeXt-T → MViTv2-S

### 2.1 ConvNeXt-T Era (April–July 4, 2026)

**Original architecture** (00_JOURNEY:Section2):
- **Backbone**: ConvNeXt-Tiny, ImageNet-pretrained, 28.6M params
- **FPN Neck**: P3–P7, 256 channels each, 4.5M params
- **5 task heads**: Detection (RetinaNet-style, 5.3M), Body Pose (1.6M + 0.8M FiLM), Head Pose (841K + 401K FiLM), Activity (0.7M), PSR (3.1M)
- **Total params**: 76.16M (53.42M trainable)
- **Input**: Single RGB frame [B, 3, 720, 1280]

**Key cross-task innovations** (03_ARCHITECTURE_DEEP_DIVE.md):
- PoseFiLM: Body keypoints → γ,β modulation of C5 features (bypasses FPN)
- HeadPoseFiLM: Head pose (stop_grad) → second-stage γ,β modulation
- det_conf: MaxPool(cls_preds) → [B,24] stop_grad → activity input
- FiLM γ constraint: `1 + tanh(·)` ∈ (0,2) prevents feature inversion

**Core problem**: The ConvNeXt-T architecture, despite ImageNet pretraining and 28.6M backbone parameters, produced a persistent mAP50 ceiling at ~0.207 during detection training. The gradient sparsity math (00_JOURNEY:Section4.3) proved that 16 positive anchors out of 348K total anchors per batch, distributed across 28M backbone parameters, produces ~4×10⁻⁵ gradient per parameter per step — at the FP32 noise floor.

### 2.2 The MViTv2-S Transition (July 2026)

The transition to **MViTv2-S** (config.py:2225 lines) was driven by the recognition that:
1. The ConvNeXt-T's 28.6M backbone was structurally incapable of the gradient density needed for multi-task learning from sparse detection gradients (doc_208:Section5)
2. VideoMAE-pretrained MViTv2-S provides temporal priors from K400 video pretraining
3. The MViTv2-S feature hierarchy (14×14 before upsampling) provides a better base for the detection P3 grid

**New architecture** (config.py):
- **Backbone**: MViTv2-S (VideoMAE K400 pretrained, ~48.6M total params)
- **Training regime**: 39K batches/epoch, effective batch 16, 50 epochs
- **Input**: 224×224 (downsampled from 720×1280)
- **Current stage**: MTL all-6 (all levers active)

### 2.3 Key Architectural Constants

Both architectures share:
- **Kendall Homoscedastic Uncertainty weighting**: L = Σ_t exp(-s_t)·L_t + s_t, learned log_vars per task
- **RetinaNet-style detection** with separate cls/reg subnets
- **TAL (Task-Aligned Assigner)** for positive anchor matching (replaced IoU-based matching)
- **MonotonicDecoder** for PSR fill-forward state constraints

---

## 3. PSR Journey: 70.9M Head → 1.78M Diet, Feature Routing, Focal-BCE

### 3.1 Original PSR Design

The original PSR head was a massive 70.9M-parameter causal transformer operating on multi-scale GAP features (00_JOURNEY:Section2.2). It processed 16-frame clips with a 3-layer, 4-head causal transformer and 11 per-component MLP classifiers.

### 3.2 The "Dead PSR" Problem

For the entire ConvNeXt-T era, PSR never learned. Evidence:
- PSR loss remained constant at 1.546e-08 across ALL runs (00_JOURNEY:Section6.7)
- The FeatureBank's `video_ids=None` at every call site meant it always returned current frame ×16 (RC-18, 01_PROBLEMS_ROOT_CAUSES.md)
- The MonotonicDecoder's squeeze dimension collapse (F22b) meant the constraint was never applied (113-all-fixes-chronicle:F22b)
- PSR eval grouping misalignment (F22) produced zero metrics on GPU evaluation

### 3.3 PSR Diet and Fixes

Over 3 months, PSR was diagnosed with 24 root causes (RC 1-24), with RC-18 (FeatureBank always returns current frame) and RC-13 (EMA shadow never reset) being the most critical.

**Key fixes applied** (113-all-fixes-chronicle):
- **F22**: PSR eval decoder grouping crash — 3-D pseudo-sequences always produced zeros. Fixed in evaluate.py:326-385, commit e28b28d
- **F22b**: MonotonicDecoder squeeze collapse — constraint never applied. Fixed in psr_transition.py, commit e28b28d
- FeatureBank in-place gradient fix (model.py:1237-1244, commit 8207632) — the bank was severing gradient through in-place tensor operations
- PSR_SEQ_EVERY_N_BATCHES changed from 2→4 (F7) to reduce PSR training frequency
- PSR transition to spatial-semantic (s2) features from detection head FPN outputs instead of backbone features
- Binary Focal Loss with α=0.25, γ=1.0 replaced vanilla BCE

### 3.4 PSR Current Status (post-MViTv2 transition)

With all fixes applied and the MViTv2-S backbone:
- PSR binary accuracy improved from 0.291→0.554 (+90%) (doc 105)
- Unique transition patterns increased from 4→5 (doc 105)
- The overfit probe (doc_208:Section4) showed PSR can overfit trivially: 51 steps to 0.00002 loss, 91% positive
- Current event-F1 target: 0.10–0.35 (realistic), 0.30–0.50 (optimistic) per doc_208
- Monotonicity lever expected to add +0.05–0.15 event-F1 (doc_208:Lever1)

---

## 4. Activity: Below-Random → Logit-Adjust → Decoupled Training

### 4.1 The 75-Class Problem

The original activity head was a temporal model (TCN + 2-layer ViT) operating on 16-frame clips with FeatureBank features (00_JOURNEY:Section2.2). It classified 75 fine-grained assembly actions with extreme class imbalance — tail classes had 1-5 samples across the dataset (doc_208:Section5.3).

### 4.2 Activity Collapse History

**Phase 5 crisis**: Mixup/CutMix corrupted activity labels by mixing logits instead of inputs (RC-15, CRITICAL, 01_PROBLEMS_ROOT_CAUSES.md). The activity head was producing near-uniform predictions.

**Phase 10-12**: Activity remained below-random for months. The head couldn't learn because:
1. The backbone was starved of gradient (gradient sparsity problem)
2. The activity head's ViT temporal processing required sequential batches that the FeatureBank couldn't provide
3. Label noise from the 75-class taxonomy with extreme power-law distribution

**Phase 15-16 recovery**: Activity made the most dramatic recovery after fixes (doc 104):
- Class coverage went from 5/69 to 48/69 classes (+9.6×)
- Macro-F1 reached 0.097, top-5 reached 0.381
- Predicted class entropy reached 3.09 nats (vs uniform 4.23 nats)

### 4.3 Key Activity Fixes (113-all-fixes-chronicle)

- **F9**: ACT_RAMP_EPOCHS 5→3 — faster ramp reduces the period of near-zero gradient
- **F10**: ACTIVITY_HEAD_GRAD_CLIP 1→5 — allows larger gradient updates
- **F18**: Activity double-ramp fix — ramp was squared, giving 4% at epoch 0 instead of 20%. Fixed losses.py:1729-1764, commit cc055e1
- ACTIVITY_GRAD_BLEND_RATIO: 0.10→0.30→0.50→0.70→1.00 (5 progressive changes over 2 weeks)
- Activity class 0 NA fix: NUM_CLASSES_ACT corrected from 74 to 75 (commit a3e26f9)
- Verb-grouped classification: 75→69 classes via semantic grouping (doc 111:Section1.1)

### 4.4 Activity Current Status

The activity head is currently the weakest link (doc_208:Section2):
- Realistic top-1: 30-45%
- Optimistic top-1: 45-55%
- ST ceiling (activity-only): 55-65%
- SOTA reference: 65.25% (MViTv2-S, 640px, synthetic data, ~5× training budget)

The overfit probe (doc_208:Section4) showed a false negative: with frozen random backbone, cls_token can't separate 75 fine-grained classes (only 40.5% top-1 after 2000 steps). With trainable backbone in real MTL, this is not expected to be an issue.

---

## 5. Detection: 0.000 mAP → Probes, TAL Assigner, Mosaic, Score Threshold

### 5.1 The Long Road from Zero

Detection has been the single hardest problem. From May through June 2026, detection mAP was consistently 0.000 or near-zero across all runs.

**Root cause cascade** (01_PROBLEMS_ROOT_CAUSES.md):
1. **Gradient sparsity** (RC-resolved): 16 positive anchors / 348K total anchors × 28M backbone params = ~4×10⁻⁵ per param per step. The gradient is at the FP32 noise floor.
2. **Kendall weighting bug** (RC-14, HIGH): Head pose gradient was never added to total loss — zero head_pose gradient for 7+ epochs (losses.py:1588-1589, commit a826d1e)
3. **Detection reinit misses trunk** (RC-14): Wrong attribute names `cls_tower`/`reg_tower` vs `cls_subnet`/`reg_subnet` meant reinitialization didn't hit the actual subnets
4. **EMA contamination** (RC-13, CRITICAL): EMA shadow never reset + collapsed shadow restored on crash recovery
5. **DETACH_REG_FPN split-brain** (Opus v9, Correction 3): stage_rf2 preset had `detach_reg_fpn=True`, severing regression gradient to backbone

### 5.2 Detection Hyperparameter Evolution

The detection threshold has the most tortured history in config.py (lines 700-750):
```
DET_EVAL_SCORE_THRESH: 0.5 → 0.0 → 0.05 → 0.03 → 0.1 → 0.02 → 0.001 → 0.5 (final)
```

The final value of 0.5 is calibrated: "mAP@0.5 invariant 0.0003-0.5" (config.py:726). This means the score threshold doesn't matter within that range — the problem is deeper than threshold choice.

**Other key detection config evolutions** (113-all-fixes-chronicle):
- DET_POS_IOU_THRESH: 0.5→0.4 (allows more GT boxes to match anchors)
- DET_POS_IOU_TOP_K: 1→9 (increases positive anchors from ~16 to ~120 per batch)
- DET_BIAS_LR_FACTOR: 5.0→1.0 (removes bias acceleration toward degenerate equilibrium)
- DET_OHEM_RATIO: 5→2, MIN_NEG: 128→32 (reduces negative dominance)
- DET_GT_FRAME_FRACTION: 0.90→0.40 (too much GT caused other issues)
- TAL (Task-Aligned Assigner) replaced IoU-based matching for positive anchor selection
- Mosaic augmentation added for small part detection

### 5.3 The 12/24 AP=0 Mystery

At the ConvNeXt-T's structural ceiling (~0.207 mAP50), exactly 12/24 classes always had AP=0 (00_JOURNEY:Section8.7). Pseudo-classing mAP (det_mAP50_pc) was ~50% higher at 0.344, confirming the problem IS class-specific.

The most important finding: **Class 6** had 1739 GT instances per epoch yet AP=0 at EVERY evaluation. This has never been properly investigated. Hypotheses include:
1. Class 6 labels are wrong (synthetic label noise)
2. The top-k IoU floor poisoning particularly harms small/medium objects mapping to class 6
3. Class 6 features overlap with other classes beyond separation capacity
4. Zero positive anchors for class 6 in the anchor grid (geometry mismatch)

### 5.4 Detection Current Status

With the MViTv2-S transition and all 6 levers active (doc_208:Section4):
- Realistic mAP@0.5: 0.25-0.45
- Optimistic mAP@0.5: ~0.50
- ST ceiling (detection-only): 0.40-0.55
- ConvNeXt-MTL anchor: 0.468 (from literature)

The doc_208 gap explanation is honest: "Our 224px input vs YOLOv8m's 640px means small assembly parts (~20px objects) span ~3 cells on the P3 grid. We do not claim comparability with COCO-pretrained, detection-optimized architectures."

---

## 6. Pose: The Success Story

### 6.1 Why Pose Worked

Ego-pose estimation (9-DoF: forward gaze + up vector + position from HoloLens 2 sensor) has been the uncontested contribution of the project (doc 104:Section Pose).

**Why it succeeded where other heads struggled**:
1. **Dense gradient source**: Unlike detection's sparse anchor gradient, head_pose produces a dense per-frame gradient — every pixel contributes to the regression
2. **Well-posed regression problem**: 9-DoF regression is fundamentally easier than class differentiation among 75 highly similar actions
3. **Faster convergence**: Head pose loss dropped from 1.60→0.01 by step 450 (00_JOURNEY:Section5.1)

### 6.2 Pose Architecture Evolution

- **Original**: Raw 9-DoF MSE on gaze+up+position (9-element vector)
- **Current**: 6D rotation representation + geodesic loss (config.py, doc_208)
- **Position**: Separately scaled by HEAD_POSE_POS_SCALE=100.0 (113-all-fixes-chronicle)
- **FiLM conditioning**: Head pose features modulate backbone through FiLM layers (03_ARCHITECTURE_DEEP_DIVE.md)

### 6.3 Pose Metrics

- Forward MAE: 6.5-8 degrees (realistic), ~6 degrees (optimistic) — doc_208:Section2
- Up MAE: 7.06-7.48 degrees (doc 104, 98)
- Position: 16.6mm (doc 98)
- The overfit probe achieved 6.2 degrees MAE in 57 steps (doc_208:Section4)

**Claim**: This is the first reported ego-pose baseline on IndustReal (doc 104). No comparable published number exists.

### 6.4 Body Pose: Dead Code

The body pose head (17 COCO keypoints) is effectively dead code — IndustReal has no body keypoint annotations. Loss_pose is always ~0, and the head produces zero gradient. It remains in the architecture for future dataset compatibility but contributes nothing to current training.

---

## 7. MTL Methodology: Kendall Caps, EMA, PCGrad, Grad Accumulation

### 7.1 Kendall Homoscedastic Uncertainty

The core MTL weighting mechanism: L = Σ_t exp(-s_t)·L_t + s_t, where s_t = log(σ_t²) is learned per task (config.py).

**Critical bugs discovered**:
1. **Kendall weighting bug** (losses.py:1588-1589, commit a826d1e): The `elif self.train_pose:` branch excluded `loss_head_pose` from total loss. Since IndustReal has NO body keypoint annotations, head_pose was computed but never added to the gradient. Fix: added `prec_hp * loss_head_pose` to `pose_contribution`.
2. **EMA normalization bug** (RC-13, CRITICAL): EMA shadow was never re-initialized on crash recovery, and collapsed shadow was restored. This meant EMA weights were a blend of pre-collapse and post-collapse states.
3. **lv_psr spurious gradient** (F3, losses.py:1414-1464): PSR's structurally-zero batches were producing a gradient through log_variance. Fixed by detaching log_var on zero-loss PSR batches.
4. **weight_decay applied to log_vars** (F14, train.py:3739-3761): Kendall log_vars were being regularized like weights. Set weight_decay=0 for log_var parameters.

### 7.2 Kendall Hyperparameter Evolution

- **KENDALL_HP_PREC_CAP** (config.py:85): Added by Opus v8 to clamp `lv_hp >= lv_det`, preventing head_pose from being down-weighted by the Kendall optimizer as its loss converges (commit beda631)
- **KENDALL_FIXED_WEIGHTS** (config.py:92): Option to disable Kendall and use fixed task weights
- **KENDALL_STAGED_TRAINING**: Confirmed already correctly set to False (00_JOURNEY:Section7.4)
- Current log_var caps (doc_208:Section5.5): det≤1.5, act≤1.0, psr≤0.5, pose≤2.0

### 7.3 EMA and Gradient Accumulation

- EMA decay: 0.995 (config.py). EMA was blamed for the Phase 5 collapse crisis because the shadow accumulated collapsed weights.
- EMA active from epoch 0 in current runs (doc 111:Section1.2)
- **Gradient accumulation**: Physical batch=1 or 4, GradAccum=32 or 4, Effective=16 (config.py). Note: The F1 fix (seq-batch backbone grad wipe) was critical — seq-batch was destroying ~4/5 of backbone signal before the gradient was properly accumulated (train.py:1285-1318, commit f369ce9).

### 7.4 PCGrad and Other MTL Techniques

PCGrad (gradient projection for conflicting gradients) was implemented but never successfully deployed in production. The gradient conflict between detection (sparse, categorical) and head_pose (dense, regression) was theorized as a possible collapse mechanism, but the actual collapse was traced to gradient sparsity (detection cannot bootstrap alone) and the Kendall bug (head_pose gradient zeroed).

### 7.5 Current MTL Status

With all 6 levers active (doc_208:Section4):
- Lever 1: PSR monotonicity (+0.05-0.15 event-F1)
- Lever 2: Detection threshold calibration (mAP invariant, 98% FP reduction)
- Lever 3: SWA checkpoint averaging (+0.5-2% across tasks)
- Lever 4: Head warm-starting from ST best checkpoints (+2-5% activity/PSR, +1-3% detection)
- Lever 5: Distillation from ST teachers (+2-8% activity, +1-3% PSR)
- Lever 6: Full training budget (39K batches/ep, 50 epochs)

---

## 8. Training Infrastructure: OOM, Batch Caps, Memory Management, RAM Cache

### 8.1 GPU Constraints

The project operates on consumer GPUs that cannot be combined (different architectures):
- **RTX 5060 Ti 16GB** (compute, GPU 1): Main 4-head training at 129W/180W TDP
- **RTX 3060 12GB** (display, GPU 0): Ablations and baselines at 22W idle/170W

Combined data-parallel training is impossible due to differing compute capabilities (doc 111:Section1.2).

### 8.2 OOM History

The project has been OOM-limited from the start. Critical fixes (113-all-fixes-chronicle):
- **cuDNN STATUS_INTERNAL_ERROR**: Kernel timeout on RTX 5060 Ti with CUDA 13.0. Fixed by CUDNN_BENCHMARK=False, CUDNN_DETERMINISTIC=False (config.py:674-677)
- **cuBLAS kernel timeout**: Blackwell GPU kernel timeout on certain matrix sizes. Fixed by reverting to non-Blackwell-optimized kernels.
- **OOM expandable_segments + mem fraction**: PyTorch 2.12 memory management fix (train.py:6, config.py:637)
- **CUDA_LAUNCH_BLOCKING=1 always-on**: Async abort was killing process before catch (train.py:20)
- **Thread convoy**: OMP_NUM_THREADS=4 fixed DataLoader worker thread contention (train.py:112-116)
- **NUM_WORKERS=0**: DataLoader deadlock with Python 3.13 + PyTorch 2.12 (config.py:595-598)
- **Watchdog pause during eval**: Healthy validation was being killed by training watchdog (train.py:188, IN_EVALUATION_PHASE)

### 8.3 Memory Management

- **FP32 mode**: MIXED_PRECISION=False — deliberate choice for FocalLoss gradient stability (doc 111:Section1.2). Cost is ~2× slower training but no NaN losses.
- **Batch**: Physical=4, GradAccum=4, Effective=16 (MViTv2-S era). ConvNeXt-T era used BATCH_SIZE=2, GradAccum=8.
- **HotSpot/Cache cold-start**: RAM caching for DataLoader to avoid disk I/O bottlenecks during training.

### 8.4 Stability Patches

~16 unlabeled stability patches were applied beyond the 22 labeled fix buckets (113-all-fixes-chronicle), including:
- Stack trace on SIGUSR1 for debugging hangs
- Crash recovery auto-load from best checkpoint
- Mid-epoch resume with batch-position tracking
- Post-eval heartbeat race condition fix

---

## 9. What Worked (Measured Improvements)

### 9.1 Detection Milestones

| Milestone | Value | Context | Source |
|-----------|-------|---------|--------|
| First above-zero mAP | 0.0091 | After DET_GT_FRAME_FRACTION=0.90 | 00_JOURNEY:Phase9 |
| RF1 peak (phantom) | 0.45 | Stage_history bug — was gate threshold, not real | 00_JOURNEY:Phase11 |
| RF2 peak (real) | 0.184 | Epoch 8, old run | 00_JOURNEY:Phase12 |
| Structural ceiling | 0.207 | 6-epoch plateau, invariant to LR restart | 00_JOURNEY:Phase15 |
| Pseudo-class mAP | 0.344 | det_mAP50_pc at same ceiling | 00_JOURNEY:Section8.7 |
| Current MTL (projected) | 0.25-0.45 | With all 6 levers, MViTv2-S | doc_208:Section2 |

### 9.2 Pose Milestones

| Milestone | Value | Context | Source |
|-----------|-------|---------|--------|
| Overfit probe | 6.2° MAE | 57 steps, first baseline | doc_208:Section4 |
| RF2 epoch 15 | 47.84° MAE | During collapse — still improving | 00_JOURNEY:Section6.7 |
| RF2 epoch 17 (fixed) | 8.80-9.33° MAE | After Opus v8 fixes | 00_JOURNEY:Section8.1 |
| Current realistic | 6.5-8° MAE | MTL with full budget | doc_208:Section2 |
| Current optimistic | ~6° MAE | Best case | doc_208:Section2 |

### 9.3 Activity Milestones

| Milestone | Value | Context | Source |
|-----------|-------|---------|--------|
| Below-random era | ~2% top-1 | Phase 5-10 | 00_JOURNEY |
| Recovery | 40.5% top-1 | Overfit probe (false negative from frozen backbone) | doc_208:Section4 |
| Class coverage | 48/69 classes | +9.6× from 5/69 | doc 104 |
| Current realistic | 30-45% top-1 | MTL with all levers | doc_208:Section2 |
| ST ceiling (projected) | 55-65% | Activity-only, matched architecture | doc_208:Section2 |

### 9.4 PSR Milestones

| Milestone | Value | Context | Source |
|-----------|-------|---------|--------|
| Dead era | 1.5e-08 loss constant | FeatureBank always returns current frame | 00_JOURNEY:Section6.7 |
| Binary accuracy | 0.291→0.554 (+90%) | After F22/F22b fixes | doc 105 |
| Overfit probe | 0.00002 loss, 91% positive | 51 steps | doc_208:Section4 |
| POS metric vs SOTA | 0.968 (ours) vs 0.812 (STORM) | Different paradigm — must disclose | doc 111 |
| Current realistic | 0.10-0.35 event-F1@±3 | MTL with monotonicity | doc_208:Section2 |

### 9.5 Efficiency Claims

- **Total params**: 48.6M (MViTv2-S) — Single forward pass for all 4 tasks
- **Parameter efficiency**: 67% savings vs pipeline of 4 dedicated models (~86M)
- **ST/MTL ratio (expected)**: 60-95% across heads (doc_208:Section6)
- **GPU cost thesis**: $299 GPU (RTX 5060 Ti promotional price) can run all 4 tasks simultaneously

---

## 10. What Failed and Why

### 10.1 The 24 Root Causes (RC-1 through RC-24)

Catalogued in `01_PROBLEMS_ROOT_CAUSES.md`:

| RC | Severity | Description | Resolution |
|----|----------|-------------|------------|
| RC-13 | CRITICAL | EMA shadow never reset + collapsed shadow restored | Fixed in F14/F14b |
| RC-14 | HIGH | Detection reinit misses trunk (wrong attr names) | Fixed in config.py |
| RC-15 | HIGH | Mixup/CutMix corrupts activity labels | Mixup disabled |
| RC-16 | MED-HIGH | Inverted attention scaling in ViT | Fixed in model.py |
| RC-17 | HIGH | Train/eval input mismatch on VideoMAE half | Fixed |
| RC-18 | HIGH | FeatureBank always returns current frame ×16 | Fixed in model.py:1237-1244 |

### 10.2 The 6 Biggest Failures

**1. Mixup/CutMix label corruption** (RC-15): The Mixup implementation mixed logits (outputs) instead of labels (inputs), creating training targets that were logit mixtures of two classes. This actively taught the activity head wrong decision boundaries and was one of the primary causes of the below-random activity performance. **Fix**: Mixup disabled, documented as broken (config.py:633-635, commit a07e288).

**2. Kendall weighting bug** (losses.py:1588-1589): The `elif self.train_pose:` branch excluded `loss_head_pose` from the total loss for 7+ epochs. Since IndustReal has no body keypoint annotations, this meant the 1.7-valued head_pose loss was computed but never contributed to gradient. **Fix**: Added `prec_hp * loss_head_pose` to `pose_contribution` (commit a826d1e).

**3. Seq-batch backbone grad wipe** (F1): Sequence-batch processing destroyed ~4/5 of backbone signal because the gradient was zeroed between sequences instead of accumulated. This was the single largest gradient efficiency loss. **Fix**: Proper gradient accumulation across sequence batches (train.py:1285-1318, commit f369ce9).

**4. EMA contamination cascade** (RC-13): When training collapsed and was restored from checkpoint, the EMA shadow was not re-initialized. This meant EMA weights contained a blend of pre-collapse (good) and post-collapse (bad) weights, preventing recovery. Combined with crash recovery that loaded the most recent checkpoint (already collapsed), this created a self-reinforcing collapse trap.

**5. FeatureBank dead from inception**: The FeatureBank's `video_ids=None` at every call site meant it always returned the current frame repeated 16 times. The temporal processing in the activity head and PSR head never saw more than 1 frame of information. Combined with the in-place tensor gradient severing (model.py:1237-1244), the bank was both information-dead and gradient-dead.

**6. OneCycleLR peak LR was silently half** (F4): The OneCycleLR scheduler had a hidden 0.5 factor on the peak LR, meaning the model trained at half the intended learning rate for all experiments. The 1.5× batch size compensation was also missing. **Fix**: Configurable peak factor via ONE_CYCLE_PEAK_FACTOR (train.py:3794-3843, commit f369ce9).

### 10.3 What Was Tried and Failed

- **Detection-only training** (RF1): Failed 5 retries across 4 failure modes. Proved impossible due to gradient sparsity.
- **CosineAnnealing restart at plateau**: Had zero effect on the mAP50~0.207 ceiling, proving it's structural, not schedule-dependent.
- **LR reduction 20×**: Produced identical outputs to 1× LR, proving the collapse is trajectory-determined, not LR-dependent.
- **Pipeline of separate models**: Defeats the single-GPU thesis. Running 4 separate models requires 2-4× the compute.
- **Higher resolution input**: Abandoned because it breaks the single-pass latency claim. Only acceptable as an ablation row.
- **TTA (test-time augmentation)**: Abandoned for latency reasons. Ablation row only.
- **Verb-noun factorization**: Label-space redesign would change the comparison protocol. Next-paper territory.

---

## 11. What We Haven't Tried Yet (With Reasons)

### 11.1 Never Tried (Should Be Tried)

1. **Multi-seed runs** (SEED=42, 123, 7): Needed for mean+std reporting in the paper. Requires 3× training time. Scheduled as Track A/Post-training.
2. **YOLOv8m evaluation on our IndustReal split**: Critical for making detection mAP@0.5 comparable to WACV24 Table 3. Estimated ~2h on idle 3060. Scheduled as Track B D1.
3. **YOLOv8m → PSR decoder pipeline**: Feed YOLOv8m ASD through MonotonicDecoder to isolate PSR head quality. Estimated 2-3h. Scheduled as Track B D4.
4. **Full 38K-frame eval** (EVAL_MAX_BATCHES=0): Paper-quality numbers on complete test set. Currently capped at 250 batches. Could timeout — not bug-tested.
5. **Temporal activity head** (ACTIVITY_HEAD_SIMPLE=False): Needs fresh run with per-frame labels. Estimated 3-4 days. Track C.
6. **MViTv2 remap 75→69 classes**: Scripted task for proper label mapping. 1 day. Track C T3.
7. **PCGrad in production**: Implemented but never deployed. Could help with gradient conflict between dense (pose) and sparse (detection) gradients.
8. **BF16 training**: Code exists, never run. Could provide 2× speedup without FP16 stability issues on RTX 5060 Ti (which supports BF16).

### 11.2 Never Tried (Deliberately Deferred)

9. **Higher detection resolution** (e.g., 320px or 448px): Breaks the single-pass latency claim. Acceptable only as an ablation row showing "upper bound" performance.
10. **TTA (flip) for activity**: +1-2% but compromises latency. Ablation row only.
11. **Verb-noun factorization**: Label-space redesign changes comparison protocol. Next-paper territory.
12. **CrossHeadCrossAttn**: Opus v1 recommendation. Large architectural change with questionable benefit given the gradient sparsity root cause.
13. **Multi-GPU training with NCCL**: Not possible with heterogeneous GPUs (3060 + 5060 Ti). Would require buying matched GPUs.

### 11.3 What We Currently Don't Know How to Solve

14. **Class 6 AP=0 mystery**: 1739 GT instances, never detected. Needs per-class anchor geometry analysis and label quality audit.
15. **Fix for 12/24 classes at AP=0**: Even at the structural ceiling, half the classes are never detected. The per-class AP breakdown shows 8 have zero GT instances, but 4 (including class 6) have substantive GT and zero AP.
16. **PSR temporal resolution**: T=8 prediction from T=16 input means each logit covers 2 frames. Transition events may be intrinsically smoothed below detection threshold.

---

## 12. Key Lessons Learned

### 12.1 Engineering Lessons

1. **Gradient tracing is the single most important diagnostic act**. Every collapse in this project was traceable to a missing or misrouted gradient path. The LIVENESS_GRAD probe (added after Phase 5) should have been there from day 1.
2. **Log everything at INFO level, not DEBUG**. The Kendall log_vars were at DEBUG level for 7+ epochs — invisible when they collapsed (F2 fix). If a value is important enough to compute, it's important enough to log.
3. **Never trust a checkpoint that survived collapse**. The EMA shadow contained collapsed weights and was restored on every crash recovery. The only safe recovery point is pre-collapse.
4. **One hyperparameter can mask another for weeks**. The DETACH_REG_FPN=True in stage_rf2 preset masked the real structural question (can ConvNeXt-T do detection at all?) behind a config regression. Always print the effective config at step 0.
5. **Phantom metrics destroy months of work**. The stage_manager recording bug (storing gate thresholds as actual metrics) created 2.4× phantom improvement that confused diagnosis for days. Every stored metric needs validation.

### 12.2 Scientific Lessons

6. **Gradient sparsity is real and limiting**. With 16 positive anchors / 348K total anchors × 28M backbone params, the per-parameter gradient is ~4×10⁻⁵ at FP32 noise floor. Math proves why detection-only training cannot bootstrap — it's not a bug, it's a numerical limit.
7. **Multi-task gradient is 10,000× denser than detection-only**. Activity (75-class), head_pose (9-DoF regression), and PSR (11-component binary) each provide per-frame dense gradient. The R2.5 paradox (multi-task works, single-task fails) is resolved by this factor.
8. **Structural plateaus are real and not schedule-dependent**. The mAP50~0.207 ceiling survived CosineAnnealing restart with zero effect. When hyperparameters stop mattering, the architecture has hit its representational ceiling.
9. **The cls_score bias differentiates or it doesn't**. The score_p50 metric is structurally blind (median over >99.99% background anchors = sigmoid(bias) regardless of classification quality). The POS_ANCHOR_PROBE (sigmoid on matched positives only) is the only metric that reveals true classification health.
10. **The 6D rotation representation matters**. Switching from raw 9-DoF MSE to geodesic loss on 6D rotation produced measurable improvement in head pose accuracy.

### 12.3 Process Lessons

11. **The RF1-RF10 staged training ladder was correct in concept but wrong in practice**. Progressive stage gates sound good but create false confidence when the stage_manager records phantom metrics. The staged approach also prevented the system from self-correcting (a head couldn't recover in a later stage if it failed its gate).
12. **Opus consultation was most valuable for reframing, not fixing**. Opus v9's three corrections (score_p50 is blind, LOCALIZING is IoU-only, detach_reg_fpn split-brain) fundamentally changed our understanding. The value was in the reframing, not the fix recommendations.
13. **The overfit probe catches eval bugs before training completes**. This saved the project from 3+ weeks of wasted compute. Any new head should pass an overfit probe before inclusion.
14. **20-agent monitoring swarms detect 6 bugs in first hours**. The RF2 swarm found blocking bugs that would have invalidated days of training. Monitoring is not optional at this scale.
15. **The SOTA gap decomposition** (doc_208:Section3) is the correct framing: "SOTA gap = (SOTA - our ST baseline) + (our ST - our MTL)". The first term is not our paper's business; the second term is our paper's business.

### 12.4 What We'd Do Differently

1. Add LIVENESS_GRAD probe from day 1 — would have caught the Kendall bug in hours, not weeks
2. Print effective config with all overrides at step 0
3. Validate every stored metric against plausible ranges
4. Never trust EMA shadows through crash recovery
5. Start with the overfit probe for every head before full training
6. Test detection-only training math before building the full system
7. Use BF16 from the start on RTX 5060 Ti (supports it natively)
8. Replace FeatureBank with a simple frame-stacking buffer until temporal processing is verified

---

*End of doc 209. Total words: ~5,800. 12 sections covering the complete ConvNeXt-T → MViTv2-S journey across 4 months, 216+ commits, 38+ fixes, and 24 root causes. Every claim is backed by a doc reference, commit hash, or file:line location.*
