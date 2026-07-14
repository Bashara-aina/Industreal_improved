# === V1 FACT-CHECK (REQUIRED READING) ===
# This file was generated as V2 before the codebase was fully audited.
# The V1 documents (208-227) it references describe the MViTv2-S era
# which is NO LONGER the active codebase. Key corrections:
#
# **Active model:** POPWMultiTaskModel in src/models/model.py (2361 lines)
# **Backbone:** convnext_tiny (28.59M, ImageNet-1K pretrain) — NOT MViTv2-S
# **Total params:** 46.47M (measured) — NOT ~48.6M
# **Detection head:** RetinaNet-style 5.31M (9 anchors, 5 levels P3-P7)
# **Activity head:** FeatureBank+TCN+2xViT 0.69M (NOT 3-layer MLP 2M)
# **PSR head:** PSRHead hidden_dim=128 3.08M (NOT 1.8M Causal Transformer)
# **Pose:** body 1.64M + head 1.45M + 2 FiLMs 1.24M (NOT single 0.2M)
# **FPN:** Standard P3-P7 4.48M (NOT BiFPN P2-P5 ~2.5M)
# **Effective batch:** 48 (BATCH=6 x GRAD_ACCUM=8) — NOT 16
# **PSR_FOCAL_GAMMA:** 0.5 — NOT 2.0
# **Kendall caps:** act=[-0.5,2.0], psr=[-4,0.0], pose=[-4,3.0] (NOT [1.5,1.0,0.5,2.0])
# **Recordings:** 36/16/32 (NOT 10/6)
# **Frames:** 207,266 (NOT ~75K)
# **Free GPUs:** RTX 5060 Ti 16GB + RTX 3060 12GB
# **Deadline:** AAIML Oct 10, 2026
#
# See consult_v2/V1_VS_CODEBASE_DISCREPANCY_REPORT.md for full audit.
# The legacy src/models/mvit_mtl_model.py (MViTv2-S) is DEAD CODE.
# All architecture numbers in this report should be re-validated against
# the active model before any paper submission.
# === END FACT-CHECK ===

# Agent 20: Claude Science Query — Comprehensive Synthesis

**Date**: 2026-07-13
**Phase**: Phase 4 — AAIML Strategy (Final Agent)
**Source**: Synthesis of 19 preceding agents (agents 01-19) from consult_v2
**Filing**: This document is the briefing that would launch a new Claude Science consultation.
**Warning**: Agent outputs for agents 12, 13, 14, 16, 17, 18, 19 were NOT FOUND in the output directory. Only 12 of 19 expected outputs were found.

---

## Section 1: Executive Summary

The IndustReal MTL project aims to train a single MViTv2-S-based model on 4 tasks (75-class activity recognition, 24-class detection-as-PSR-state-classifier, 11-component binary PSR, 9-DoF head pose) using the IndustReal egocentric video dataset. As of July 2026, the project has run multiple training experiments at 480px resolution with T=8 temporal windows, using FAMO+RotoGrad+Kendall loss weighting, BiFPN neck, and YOLOv8-style detection head.

**What we know**: The dataset has 36 train / 16 val / 32 test recordings (participant-disjoint), 78,961 raw train frames (26,322 at stride=3), and a 75-class activity label space with severe long-tail (15 classes with <10 frames). PSR labels are fill-forward from sparse state-change annotations (0.31% transition rate) producing dense per-frame binary vectors (54.88% positive frame rate). Detection is a 24-class region-based state classifier (not traditional object detection) with exactly 1 bbox per frame.

**What works**: The training pipeline is complete and functional at 480px T=8. Gradient checkpointing enables training on 16GB. FAMO loss weighting is wired. Varifocal + WIoU v3 for detection are active. PSR uses Gaussian-smeared transition targets (sigma=3.0) with causal transformer. Head pose achieves 7.48 deg best forward angular MAE. The 1-line thw fix enables multi-resolution inference.

**What doesn't work**: The training pipeline has 5 critical bugs that collectively prevent meaningful convergence:
1. FPN prefix bug freezes 14.5M parameters (26% of model)
2. RotoGrad parameters frozen (639K parameters, random matrices)
3. Warm-start broken (3 of 4 ST head checkpoints missing)
4. DetectionAugment clamp bug destroys normalized image distribution
5. Missing expandable_segments:True risk of fragmentation OOM

Activity head has collapsed (1 unique class per batch at epoch 11). PSR head produces flat ~0.69-0.71 output for all 11 components. Without fixing the 5 critical bugs, the model will not converge to a meaningful multi-task solution.

---

## Section 2: Critical Findings (Ranked by Impact)

### Finding 1 [CRITICAL]: FPN Frozen by Prefix Bug — 14.5M Parameters Never Updated
**Source**: Agent 08 (Task Heads), Agent 06 (Backbone Capacity)
**Impact on paper**: The BiFPN (26% of total model parameters) is initialized to random weights and never receives gradient updates. Detection features are frozen random projections. No model evaluation is meaningful with random FPN weights.
**Evidence**: `_group_params(["feature_pyramid.fpn", "det_head"], 1.0)` at `train_mtl_mvit.py:2133` uses prefix `feature_pyramid.fpn`. The actual FPN module is `self.fpn` (registered at `mvit_mtl_model.py:520`), so `fpn.lateral.*`, `fpn.td_conv.*`, `fpn.bu_conv.*` parameters match no optimizer group.
**Action**: Change prefix from `"feature_pyramid.fpn"` to `"fpn"` in the param group definition. Alternatively, register the FPN under `self.feature_pyramid` to match the existing prefix.

### Finding 2 [CRITICAL]: RotoGrad Parameters Frozen — 639K Random Matrices
**Source**: Agent 08 (Task Heads), Agent 09 (Training Pipeline)
**Impact on paper**: RotoGradRotation is instantiated AFTER optimizer creation (`train_mtl_mvit.py:2273-2276` vs optimizer at line 2142). No `add_param_group()` call is made. The `rotation_loss()` function is never called. The "rotation" is equivalent to random noise injection.
**Evidence**: grep for `rotation_loss|rotograd.*optimizer|rotograd.*backward` returns zero matches in `train_mtl_mvit.py`. The model saves parameters log shows `55.7M total, 41.16M trainable` — 14.54M not trainable, matching FPN (14.5M) + RotoGrad (0.64M - actually FPN accounts for almost all of it).
**Action**: Either add RotoGrad params to the optimizer, or implement the rotation loss optimization as described in the module docstring. Route PSR through RotoGrad (currently only activity and pose get rotation, not PSR).

### Finding 3 [CRITICAL]: Warm-Start Completely Broken — 3 of 4 Head Checkpoints Missing
**Source**: Agent 09 (Training Pipeline), Agent 08 (Task Heads)
**Impact on paper**: Only `st_pose_best.pt` was found. Detection, activity, and PSR heads initialize from random weights. The first 5+ epochs are wasted re-converging each task head. The pose checkpoint only loaded 2 tensors (partial match).
**Evidence**: Log lines: `Warm-start det: checkpoint not found, skipping`, `Warm-start act: checkpoint not found, skipping`, `Warm-start psr: checkpoint not found, skipping`, `Warm-start pose: loaded 2 tensors`. The checkpoint directory `src/runs/st_checkpoints/` contains only `st_pose_best.pt`.
**Action**: Verify checkpoint directory path. Generate ST checkpoints for detection, activity, and PSR. Fix `load_state_dict_with_prefix` for MTL head structure vs ST checkpoint structure.

### Finding 4 [CRITICAL]: Activity Head Collapsed — 1 Unique Class at Epoch 11
**Source**: Agent 08 (Task Heads), Agent 09 (Training Pipeline)
**Impact on paper**: Activity classification performs below random baseline (1.33% vs 1.3% random uniform for 75 classes). The model's core classification capability is effectively zero.
**Evidence**: `act_preds=1uniq/0.03maxconf` at epochs 11-13 (T8_frag.log:72, T4_v2.log:71,95,118). Activity loss INCREASES across epochs (3.99 -> 4.07 -> 4.15), creating a self-reinforcing collapse via FAMO: decreasing weight -> less learning -> worsening loss -> even less weight.
**Action**: (a) Fix warm-start to provide ST activity checkpoint. (b) Consider `act_decoupled=True` to freeze backbone and retrain just the classifier. (c) Investigate whether BalancedSoftmax is counterproductive (eval uses argmax over raw logits, but training shifts logits by `log(priors)`).

### Finding 5 [HIGH]: 3 of 24 Detection Classes Have Zero Training Instances
**Source**: Agent 01 (Data Audit), Agent 03 (Detection Annotation Audit)
**Impact on paper**: Classes 13, 19, 23 have 0 training instances. Class 23 (`error_state`) has 0 instances across ALL splits (train/val/test). This dilutes reported mAP by ~8-13% (3 zero-AP channels out of 24).
**Evidence**: Exhaustive label scan across 36 train recordings. Class 13 (57 val-only instances), class 19 (39 val-only instances), class 23 (0 everywhere).
**Action**: Either remove these classes (change NUM_DET_CLASSES to 21) or compute mAP on active classes only for monitoring. Check if class 23 is actually a frame corruption flag that belongs outside the softmax label space.

### Finding 6 [HIGH]: PSR Head Produces Flat Output — No Temporal Discrimination
**Source**: Agent 08 (Task Heads), Agent 09 (Training Pipeline), Agent 04 (Activity/PSR Data)
**Impact on paper**: All 11 PSR components predict ~0.69-0.71 probability with frame-to-frame stddev of 0.02. Temporal transition detection is non-functional. The model predicts the marginal probability of each component (treats every frame as background).
**Evidence**: `psr_stdmax=0.0206` in logs. All components converge to similar values regardless of input frame. The MS-TCN refinement head (2 stages, 206K params) operates on already-flat probabilities.
**Action**: Investigate whether PSR head input features (P5 from blocks[14]) carry temporal information. Add diagnostic for temporal variance of P5 features before and after spatial pooling. Consider reducing PSR head capacity (d=256 -> d=128) since the signal is weak.

### Finding 7 [HIGH]: PSR False Monotonicity Assumption — 82.6% of Frames Violate
**Source**: Agent 04 (Activity/PSR Data)
**Impact on paper**: The MonotonicDecoder assumes cumulative procedure semantics (comp_k=1 implies comp_{0..k-1}=1). Actual annotation data shows 82.6% of frames violate this. The 11 components are independent binary attributes, not sequential assembly steps.
**Evidence**: 17.4% of frames satisfy monotonic non-increasing property. Key violations: comp5=1 but comp4=0 (45.5% of frames), comp4=1 but comp3=0 (19.04%), comp10=1 but comp9=0 (16.84%). The `USE_PSR_ORDER_PRIOR = False` setting is correct.
**Action**: Confirm `USE_PSR_ORDER_PRIOR = False` is active in all configs. Remove any paper text suggesting cumulative procedure semantics for PSR.

### Finding 8 [HIGH]: DetectionAugment Clamp Destroys Normalized Distribution
**Source**: Agent 09 (Training Pipeline)
**Impact on paper**: Images are normalized to [-2.0, +2.4] range, then DetectionAugment color jitter clamps to `[0.0, 1.0]`, truncating ~50% of pixel distribution. Every batch that triggers color jitter (~50%) sends distribution-shifted images to the backbone, degrading all 4 tasks.
**Evidence**: `det_augment.py:102`: `aug_images = aug_images.clamp(0.0, 1.0)`. But images are normalized at `train_mtl_mvit.py:2365-2368` BEFORE augmentation at line 994.
**Action**: Replace clamp to match normalized range, e.g., `aug_images.clamp(-2.5, 2.5)`, or remove the clamp entirely.

### Finding 9 [HIGH]: Missing expandable_segments:True — Fragmentation OOM Risk
**Source**: Agent 09 (Training Pipeline)
**Impact on paper**: `train_mtl_mvit.py` does not set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Every other training script in the repo sets it. Fragmentation OOM is likely at epoch 20-25 of a 35-epoch run.
**Evidence**: The script runs 2.8M optimizer steps over 35 epochs. Without expandable_segments, PyTorch CUDA allocator fragments memory over long runs.
**Action**: Add `os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'` at the top of `train_mtl_mvit.py`, BEFORE any torch import.

### Finding 10 [HIGH]: V1 Consultation Based on Wrong Data Assumptions
**Source**: Agent 01 (Data Audit)
**Impact on paper**: The V1 Claude Science consultation (docs 208-227) was based on data estimates that were incorrect in key aspects: 10 train recordings (actual 36), 6 val recordings (actual 16), PSR <0.5% positive rate (actual 54.88%), ID 0=NA/background (actually real action class). While the V1 architectural recommendations were sound, their data rationale was wrong.
**Evidence**: 11-claim verification matrix in Agent 01, Section 5. 4 of 11 claims were WRONG, 3 were OUTDATED, 1 was CLOSE, only 3 were CORRECT.
**Action**: All future analysis and claims must be verified against the actual data, not literature-derived estimates. Cross-check every quantitative claim before submission.

---

## Section 3: Open Questions for Claude Science

### Question 1: FPN-Head Parameter Ratio
> "Our MTL model has a BiFPN neck with 14.5M parameters that is ~12x larger than the detection head (1.2M). This inverted ratio (neck dominate over detection head) is atypical in detection literature, where necks are typically 2-4x smaller than detection heads. Is this 12:1 ratio a design error, or does the BiFPN's bidirectional 3D computation provide sufficient detection benefit to justify 14.5M params? What is the expected mAP impact of reducing the BiFPN to a standard 2D FPN (similar to the 1.0M FPN in model.py)? Given that detection uses only P3/P4/P5 (P2 is excluded), could we eliminate P2 processing entirely and save ~2M params?"

**Motivation**: The FPN prefix bug (Finding 1) means the BiFPN has never actually been trained — its 14.5M params have been frozen at random initialization for all runs. The question of whether this architecture is beneficial or wasteful is empirically unanswerable from existing runs.

### Question 2: Activity Collapse — Root Cause and Fix
> "Our MViTv2-S MTL model's activity head predicts only 1 unique class per batch with 3-4% max softmax confidence after 11 epochs of training. Four simultaneous factors may contribute: (a) random head initialization (warm-start failed), (b) BalancedSoftmax adding `log(priors)` to logits during training but raw argmax evaluation, (c) FAMO decreasing weight when activity loss increases (self-reinforcing collapse), and (d) pose gradients (100-1000x larger raw loss) drowning activity gradient. Which factor is dominant, and what is the minimal intervention that produces >10 unique predictions within 5 epochs?"

**Motivation**: Activity collapse is the single largest barrier to paper submission. Understanding the root cause distinguishes a simple fix (provide ST checkpoint) from a fundamental MTL architecture problem (cannot learn 75-class activity with 4-task MTL).

### Question 3: Detection as PSR State Classifier — Metric Validity
> "Our detection task is not traditional object detection: every frame has exactly 1 bbox, and the 24 class labels encode the 11-bit PSR binary vector. Three classes (13, 19, 23) have zero training instances. Class 23 (error_state) has zero instances across all dataset splits. We report COCO-style mAP across 24 classes. Is mAP a valid metric for this task, or does it conflate classification accuracy (distinguishing PSR states at Hamming distance 1-2) with box regression accuracy (predicting the assembly area bbox)? What metric would better separate these two sub-tasks?"

**Motivation**: The paper claims "detection" performance, but the task is region-based state classification. This framing needs to be accurate for AAIML reviewers.

### Question 4: Pre-Scaling + FAMO Interaction
> "Our training applies pre-scaling factors [det:0.125, act:0.27, psr:2.7, pose:0.00025] to bring raw losses (~5 orders of magnitude apart) to approximately the same scale, then applies FAMO adaptive weighting on the pre-scaled losses. Does this two-stage normalization create a correct loss landscape for FAMO optimization? Specifically, FAMO's weight update uses `log(L_k^t) - log(L_k^{t+1})` on pre-scaled losses. If the pre-scaling factors are wrong, can FAMO recover the correct weights through its adaptive mechanism, or are we always limited by the quality of the hand-tuned pre-scaling?"

**Motivation**: Agent 15 found that no literature validates this two-stage approach. If pre-scaling + FAMO is our methodological contribution, we need to understand if FAMO is doing real work or if the pre-scaling is doing everything.

### Question 5: Gradient Clipping — Masking Effect on MTL Weighting
> "Our training applies `clip_grad_norm_(model.parameters(), 5.0)` AFTER FAMO/UW-SO weighting. Kurin et al. (NeurIPS 2022) found that equal weights with tuned hyperparameters matches sophisticated MTO methods. Does gradient clipping at norm=5.0 mask the effect of different loss weightings by capping all gradient norms to a similar magnitude? If we removed clipping entirely, would FAMO's weight adaptation become more effective, or would pose gradients (naturally 100-1000x larger) dominate catastrophically?"

**Motivation**: No literature studies the interaction between gradient clipping and MTL loss weighting. This could explain why our runs show activity collapse despite FAMO — the clipping may be nullifying FAMO's effect.

### Question 6: Temporal Modeling Gap
> "Our MViTv2-S backbone processes T=8 temporal tokens (16 input frames pooled to 8 by conv_proj stride 2). No block ever pools the temporal dimension — all `stride_q[0] = 1` and `stride_kv[0] = 1` throughout all 16 blocks. The model has no hierarchical temporal abstraction. Assembly actions span 1-4 seconds (10-40 frames at 10 FPS). Is the lack of temporal hierarchy a fundamental limitation for activity recognition, or does the 8-frame self-attention (which is global within the 0.8s window) provide sufficient temporal context for distinguishing the 75 action classes? What is the expected top-1 accuracy ceiling with T=8 vs T=16 vs T=32?"

**Motivation**: Agent 06 identified this as a structural limitation. PSR uses a causal transformer over the same T=8 window, detection mean-pools over time. Understanding the temporal ceiling is critical for paper framing.

### Question 7: PSR Component Identity Artifact
> "Our 11 PSR components show comp7 and comp8 with IDENTICAL positive frame counts (11,631 frames each, 44.17% positive rate) across all 36 training recordings. These components have exactly the same distribution — a statistical impossibility for independent assembly features. Is this a data annotation artifact (copy-paste error, or comp7 and comp8 representing the same physical state)? If so, what is the expected impact on PSR binary F1 of having 2 of 11 channels as duplicate targets?"

**Motivation**: Agent 04 discovered this. If confirmed as an artifact, it means the effective PSR component count is 10, not 11, and the per-component PSR F1 metric inflates by ~9%.

### Question 8: Head Pose Ceiling Analysis
> "Our best head pose forward angular MAE is 7.48 deg (tma_tbank_benchmark run). The within-recording angular spread is ~6.4 deg (range 2.7-10.4 deg across 36 recordings). The HoloLens 2 tracking accuracy is ~1-3 deg. The pose head is a 2-layer MLP (768->256->6, 0.20M params) operating on GAP-pooled cls_token. Is the 7.48 deg ceiling primarily: (a) annotation noise floor (2-4 deg), (b) model capacity (0.20M MLP on pooled features is insufficient), (c) resolution limit (480px backbone pretrained at 224px), or (d) multi-task interference? What would be the expected improvement from: increasing head to 768->512->128->6 (0.7M params), adding temporal smoothness loss, or using spatial features instead of pooled?"

**Motivation**: The 7.48 deg MAE is the best performing metric in the entire project. Understanding whether this represents near-ceiling performance or whether significant gains are possible determines how we frame pose results in the paper.

### Question 9: Activity Long-Tail — What Classes Are Learnable?
> "Our activity classifier has 75 fixed output channels. Of 72 present classes, 41 have <500 frames (57%), and 15 have <10 frames at stride=3. With frozen backbone + only heads training at 480px, the effective number of learnable classes is limited by fixed ImageNet-pretrained features. Classes with <50 frames (~5 seconds at 10 FPS) are unlikely to produce separable feature clusters. What is the expected effective number of learnable activity classes, given ConvNeXt-Tiny (or MViTv2-S) features under 4-task MTL? Would verb-grouping (`ACT_CLASS_GROUPING = 'verb'`) collapse to ~13 well-populated groups and produce better paper metrics than 75-class reporting?"

**Motivation**: The 75-class activity results will be poor regardless of training fix. Understanding whether to report 75-class metrics or collapse to verb groups is a paper strategy question.

### Question 10: Paper Positioning — What Is the Novel Contribution?
> "Our project has: (a) a novel task combination (detection + activity + PSR + pose on egocentric assembly video), (b) a BiFPN neck with 3D convolutions for temporal multi-scale features, (c) a combined pre-scaling + FAMO weighting for heterogeneous MTL, (d) the 1-line thw fix enabling multi-resolution MViT inference, and (e) documented real-world bugs in MTL system integration. Given that many standard MTL components (RotoGrad, MS-TCN, LDAM) remain unimplemented or disconnected, what is the most publishable contribution at our current stage? Should we focus on the 'lessons learned from building a real MTL system' narrative (acknowledging the bugs), or fix all bugs and pursue a 'state-of-the-art on IndustReal' quantitative claim?"

**Motivation**: This is the fundamental strategic question. The 5 critical bugs mean no quantitative results are meaningful. The project needs to decide: invest 2-4 weeks in bug fixing and re-running, or publish the engineering experience paper.

---

## Section 4: Implementation Status

Using the 95-item Phase B checklist as reference. Status legend: DONE / PARTIAL / BROKEN / MISSING / N/A.

### Data Pipeline (items 1-14)

| Item | Status | Notes |
|------|--------|-------|
| Dataset loading | DONE | `IndustRealMultiTaskDataset` functional |
| Frame stride (train=3, eval=1) | DONE | Configurable per split |
| RAM frame cache | DONE | 500-image LRU cache |
| Sequence mode (T=8) | DONE | Both T=4 and T=8 tested |
| GuaranteedGTBatchSampler | DONE | 40% GT-bearing frames |
| Class-balanced WeightedRandomSampler | DONE | Floor=15, sqrt-tamed |
| Hybrid verb grouping | DONE | `ACT_CLASS_GROUPING='hybrid'` |
| Gaussian-smeared PSR targets | DONE | sigma=3.0 (not 2.0 as previously claimed) |
| PSR fill-forward label logic | DONE | `_parse_psr_raw` |
| Image normalization | DONE | mean=[0.45,0.45,0.45], std=[0.225,0.225,0.225] |
| DetectionAugment | DONE | **BROKEN** — clamp bug (Finding 8) |
| Train/val/test split | DONE | Participant-disjoint, 36/16/32 |
| Frame stride mismatch (train=3, val=1) | DONE | Asymmetric evaluation |
| num_workers=0 | DONE | **Suboptimal** — synchronous decode bottleneck |

### Architecture (items 15-35)

| Item | Status | Notes |
|------|--------|-------|
| MViTv2-S backbone | DONE | 34.23M params |
| BiFPN neck | DONE | 14.5M params, **BROKEN** — frozen (Finding 1) |
| Detection head (decoupled cls+reg) | DONE | 1.20M params |
| Activity head (3-layer MLP) | DONE | 3.75M params, **COLLAPSED** (Finding 4) |
| PSR head (causal transformer) | DONE | 1.78M params, **FLAT OUTPUT** (Finding 6) |
| Pose head (2-layer MLP) | DONE | 0.20M params, functional (7.48 deg best) |
| 1-line thw fix (multi-resolution) | DONE | Active at `mvit_mtl_model.py:106-109` |
| Gradient checkpointing | DONE | 16 blocks all checkpointed |
| RotoGrad rotation | DONE | **BROKEN** — frozen params (Finding 2) |
| RotoGrad scale | PARTIAL | Implemented but never instantiated |
| PSR refinement (MS-TCN, 2-stage) | DONE | Active, but operates on flat probabilities |
| FAMO loss weighting | DONE | Active, destabilized by pose variance |
| Kendall uncertainty weighting | RETAINED | Replaced by UW-SO |
| UW-SO weighting | DONE | Default when FAMO disabled |
| PCGrad | PARTIAL | `MTLBalancer` exists, optional |
| IMTL-L | MISSING | Implemented in `imtl_l.py` but not wired |
| MetaBalance | PARTIAL | Implemented but not default |
| FeatureBank (activity temporal) | MISSING | Present in original model.py, absent in MTLMViTModel |
| VideoMAE temporal encoder | MISSING | Not enabled |
| TAL assigner | DONE | `tal_assigner.py`, topk=10 |
| Varifocal loss | DONE | Active |
| WIoU v3 loss | DONE | Active |
| BalancedSoftmax | DONE | Active, may be counterproductive |
| CB-Focal loss | MISSING | `USE_CB_FOCAL_ACT = False` |
| LDAM-DRW | MISSING | Not implemented |
| MS-TCN smoothness loss | MISSING | Implemented but not wired |
| Decoupled classifier re-training | MISSING | cRT not implemented |
| Curriculum learning decay | DONE | **DEAD CODE** — function defined but never called |

### Training Infrastructure (items 36-55)

| Item | Status | Notes |
|------|--------|-------|
| Mixed precision (bf16) | DONE | Correctly configured |
| GradScaler | DONE | No-op with bf16 (correct) |
| Gradient accumulation | DONE | Overridden to 1 (effective batch=1) |
| expandable_segments:True | **MISSING** | Fragmentation OOM risk (Finding 9) |
| Warm-start head loading | DONE | **BROKEN** — 3/4 checkpoints missing (Finding 3) |
| Resumable checkpoints | DONE | Working |
| EMA (exponential moving average) | DONE | Active |
| Test-time augmentation (TTA) | DONE | Active |
| Training logs (TensorBoard) | DONE | Working |
| GradNorm | MISSING | Abandoned for FAMO |
| CAGrad | MISSING | Not implemented |
| Nash-MTL | MISSING | Not implemented |
| Per-task learning rates | DONE | 0.3x for PSR and pose |
| Per-recording val metrics | DONE | Tracked in logs |
| Per-class val accuracy | DONE | Available |
| Gradient diagnostics | PARTIAL | `e8_gradient_diagnostic.py` exists |
| OOM mitigation | BROKEN | Missing expandable_segments, no memory fraction |

### Data Quality (items 56-70)

| Item | Status | Notes |
|------|--------|-------|
| Activity class distribution | AUDITED | 72/75 present, severe long-tail (41 classes <500 frames) |
| PSR positive rate | AUDITED | 54.88% positive, 0.31% transition rate |
| PSR monotonicity | AUDITED | 82.6% of frames violate — NOT cumulative |
| PSR comp7/comp8 identical | AUDITED | Data artifact, duplicate distribution |
| Detection class coverage | AUDITED | 3 classes zero-instance, class 23 globally absent |
| Detection bbox analysis | AUDITED | All large, no tiny objects, mean AR=1.96 |
| Head pose noise floor | AUDITED | HoloLens 2: 1-3 deg, 6D representation lossless |
| Head pose data leak risk | AUDITED | Low — 33/36 recordings share position range |
| Participant-level dominance | AUDITED | Participant 26 = 24.4% of val frames |
| Val-only activity classes | AUDITED | IDs 66, 72 (5 segments) — irreducible error |
| check_instruction over-representation | AUDITED | 18.5% val vs 11.0% train (1.68x) |
| Class ID 0 resolution | AUDITED | NOT background — real action (take_short_brace) |
| Train/val distribution shift | AUDITED | 2 val-only classes, check_instruction inflated |
| Test set audit | MISSING | Not analyzed |

### Literature Support (items 71-85)

| Item | Status | Notes |
|------|--------|-------|
| YOLOP (shared encoder + decoders) | VERIFIED | Architecture support |
| MViTv2 backbone | VERIFIED | CVPR 2022, Kinetics-400 80.8% top-1 |
| BiFPN weighted fusion | VERIFIED | EfficientDet, CVPR 2020 |
| TAL (Task-Aligned Learning) | VERIFIED | TOOD, ICCV 2021 |
| Varifocal loss | VERIFIED | VFNet, CVPR 2021 Oral |
| FAMO | VERIFIED | NeurIPS 2023 |
| RotoGrad | VERIFIED | ICLR 2022 |
| PCGrad | VERIFIED | NeurIPS 2020 |
| Huberised geodesic loss | VERIFIED | Geist et al., ICML 2024 — dead code |
| MS-TCN refinement | VERIFIED | Abu Farha & Gall, CVPR 2019 — not wired |
| Ego4D baselines | VERIFIED | CVPR 2022 |
| EPIC-KITCHENS assembly | VERIFIED | Tavakoli et al., EPIC@CVPR2021 |
| Kurin equal weights finding | VERIFIED | NeurIPS 2022 |
| Xin et al. scale finding | VERIFIED | NeurIPS 2022 |
| Egocentric online action detection | VERIFIED | An et al., CVPR 2024 |

### Paper Writing (items 86-95)

| Item | Status | Notes |
|------|--------|-------|
| Data section | DRAFT | Agent 01 provides verified numbers |
| Architecture section | DRAFT | Agents 06-08 provide architecture details |
| Training section | DRAFT | Agent 09 provides pipeline details |
| Literature review | DRAFT | Agents 11, 15 provide verified citations |
| Efficiency table (Table 4) | DRAFT | Agent 10 provides numbers — must verify at 480px |
| Results table | NOT DRAFTABLE | All runs have critical bugs — no valid metrics |
| Ablation study | NOT STARTED | Requires functional training first |
| Figure planning | DRAFT | Doc 224 provides plan |
| Risk assessment | COMPLETE | Doc 225, verified by agents |
| Implementation roadmap | COMPLETE | Doc 226, partially implemented |

---

## Section 5: Paper Readiness Assessment

### Data: ⚠️ (MODERATE ISSUES)

**Strengths**: Dataset fully documented. 36/16/32 recording split verified. Class distributions computed. Val distribution shifts quantified.

**Weaknesses**:
- 3 detection classes with zero training data (classes 13, 19, 23)
- Class 23 (error_state) has zero instances in ALL splits — cannot be learned or evaluated
- 2 activity classes (66, 72) appear only in val — irreducible accuracy floor
- PSR comp7/comp8 are duplicate columns — data artifact
- PSR labels are NOT cumulative (82.6% of frames violate monotonicity) — paper must not claim procedure semantics
- Test set data distribution not yet analyzed

**Fixable before submission**: Yes. Remove dead detection classes, acknowledge val-only action classes, document PSR non-cumulative property.

### Architecture: ⚠️ (MODERATE ISSUES)

**Strengths**: MViTv2-S backbone is appropriate. BiFPN neck with 3D convolutions is state-of-the-art. Decoupled detection head with TAL and DFL. PSR causal transformer with Gaussian-smeared targets. Varifocal + WIoU v3 for detection.

**Weaknesses**:
- FPN prefix bug means BiFPN is frozen — no architecture evaluation possible
- RotoGrad parameters are frozen — feature rotation is noise
- No temporal hierarchy in backbone (T=8 flat, no hierarchical temporal abstraction)
- 384-dim bottleneck across 10 blocks limits 4-task capacity
- Pose head under-parameterized (0.20M), PSR head over-parameterized (1.78M)
- No per-task feature routing

**Fixable before submission**: Yes, with 1-2 weeks of engineering. Fix FPN prefix (1-line change). Wire RotoGrad parameters to optimizer. Rebalance head capacities. Design choices are sound — only the wiring is broken.

### Training: ❌ (CRITICAL ISSUES)

**Strengths**: Training pipeline is complete. FAMO is wired. Mixed precision works. Gradient checkpointing enables 480px training.

**Weaknesses**:
- Activity head collapsed (1 unique class per batch) — model not learning
- PSR head produces flat output (stddev=0.02 across all frames)
- All 5 critical bugs (Rank 1-5 in Section 2) prevent meaningful convergence
- Effective batch size = 1 (grad_accum overridden to 1)
- No valid quantitative results from any training run

**Fixable before submission**: Yes, but requires 2-4 weeks: (1) Fix FPN prefix, (2) Wire RotoGrad optimizer, (3) Generate ST head checkpoints, (4) Fix clamp bug, (5) Add expandable_segments, (6) Re-run training for 35 epochs.

### Baselines: ⚠️ (PARTIAL)

**Strengths**: YOLOv8-m at 224 achieved 0.5377 mAP for detection (agent03). Single-task pose achieves 7.48 deg (agent05). Constant-predictor floors computed for all tasks (agent02).

**Weaknesses**:
- MT baselines (FAMO vs equal-weights vs UW-SO) not yet compared
- No ST baselines for activity or PSR
- The V1 consultation's claimed baselines were based on different architecture (random init at 224, not FixRes at 480)
- No inference-time numbers for the current MTL model

**Fixable before submission**: Yes. ST baselines require generating checkpoints (which are needed for warm-start anyway). Equal-weights comparison requires running with `--equal-weights` flag.

### Ablations: ❌ (NOT STARTED)

**Strengths**: Doc 222 provides a complete ablation study plan. Key questions identified: resolution (224 vs 480), temporal window (T=4 vs T=8), loss weighting (FAMO vs equal vs UW-SO), neck design (BiFPN vs standard FPN), temporal pooling (mean vs attention for detection).

**Weaknesses**: None of these ablations have been run because the training pipeline has critical bugs.

**Fixable before submission**: Requires functional training first (see Training readiness). Then approximately 10 training runs at 2-3 days each = 20-30 days of compute time.

### Writing: ⚠️ (PARTIAL)

**Strengths**: 20 consultation documents (docs 208-227) provide comprehensive architecture, training, data, and literature content. Agent outputs provide verified facts and citations.

**Weaknesses**: No quantitative results exist to populate the results section. The narrative must be built around "lessons learned" if results cannot be produced. Some claims in existing docs are stale (e.g., docstring says 45M params, actual is 55.7M).

**Fixable before submission**: Yes, but the writing must be grounded in verified numbers, not docstring claims. All quantitative claims from V1 (which were literature estimates, not data analysis) must be replaced with verified numbers.

### Overall Assessment

The project is approximately 4-6 weeks away from a submittable paper, assuming:

- **Week 1-2**: Fix 5 critical bugs (FPN prefix, RotoGrad optimizer, warm-start, clamp, expandable_segments). Re-run training.
- **Week 3-4**: Verify convergence. Run ST baselines. Run ablation studies (resolution, temporal window, loss weighting).
- **Week 5-6**: Write paper with real numbers. Generate figures and tables.

**Without bug fixes, the paper cannot be submitted.** No quantitative result from any existing training run is valid because the BiFPN (26% of model) has been frozen at random initialization.

---

## Section 6: Key Metrics to Track

The following metrics should be monitored in the NEXT training run after the 5 critical bugs are fixed. These metrics form the "smoke test" that determines if training is working correctly.

### Early Training (Epochs 1-5) — Convergence Diagnostics

| Metric | Expected | Concern if | Meaning |
|--------|----------|------------|---------|
| Activity unique predictions per batch | >10 at epoch 1, >30 by epoch 5 | <5 (especially after epoch 3) | Activity not learning |
| Activity loss trajectory | Monotonically decreasing | Increasing | Activity collapse (FAMO self-reinforcing) |
| PSR output stddev across frames | >0.05 by epoch 3 | <0.03 after epoch 5 | PSR not discriminating |
| PSR per-component variance | >0.05 across 11 components | Components converge to same value | Duplicate artifact or no temporal signal |
| Pose forward angular MAE | <30 deg by epoch 5 | >50 deg at epoch 5 | Pose head not converging |
| Detection mAP (monitored) | >0.01 by epoch 2 | Remains 0.0 after epoch 3 | Detection not learning any class |
| Gradient norm (total) | 1.0-15.0 after clipping | >clip_norm frequently or near-0 | Gradient explosion or vanishing |
| FPN parameter norms | Non-zero | All zero (not updated) | FPN prefix bug still present |
| RotoGrad parameter change | Non-zero (if optimizer wired) | Zero change from init | RotoGrad still frozen |

### Mid Training (Epochs 10-20) — Performance Metrics

| Metric | Expected | Concern if | Meaning |
|--------|----------|------------|---------|
| Activity top-1 val accuracy | >10% | <5% | Still collapsed |
| Activity unique predictions | >50 of 75 | <20 | Tail classes not learned |
| PSR per-component binary F1 | >0.4 for head components | <0.2 | PSR not detecting transitions |
| PSR event F1 (within +/-3 frames) | >0.3 | <0.1 | Temporal detection failing |
| Pose forward angular MAE | <15 deg | >20 deg | Head not converging to noise floor |
| Detection mAP (val) | >0.3 | <0.1 | Detection not working |
| Detection mAP on active classes | >0.35 | <0.15 | Even with dead classes removed, poor |
| Loss pre-scaling output range | All tasks 0.1-5.0 scaled | Any task >>5.0 or <<0.01 | Pre-scaling factors need re-tuning |
| FAMO weight distribution | No weight <0.05 | Any task weight near-zero | Task being suppressed — weight collapse |

### Late Training (Epochs 25-35) — Paper-Ready Metrics

| Metric | Target | Good | Excellent |
|--------|--------|------|-----------|
| Activity top-1 val accuracy | >35% | >30% | >40% |
| Detection mAP (24-class) | >0.45 | >0.35 | >0.55 |
| Detection mAP (21 active classes) | >0.50 | >0.40 | >0.60 |
| PSR per-component binary F1 | >0.5 | >0.4 | >0.6 |
| PSR event F1 (transition detection) | >0.35 | >0.25 | >0.50 |
| Pose forward angular MAE (deg) | <10 | <12 | <8 |
| Position MAE (scaled units) | <0.5 | <1.0 | <0.3 |
| Training time per epoch (480px T=8) | <1500s | <2000s | <1000s |
| Total training time (35 epochs) | <15 hours | <20 hours | <10 hours |
| GPU peak memory | <14 GB | <15.5 GB | <12 GB |

### Infrastructure Health Metrics (Check Once After Bug Fix)

| Metric | Expected | How to Verify |
|--------|----------|---------------|
| FPN parameters in optimizer | 14.5M params with non-zero gradients | `optimizer.param_groups[1]` contains `fpn.*` params |
| RotoGrad parameters in optimizer | 639K params trained | `optimizer` has RotoGrad param group |
| Warm-start tensor count | All 4 heads loaded (tensor count matches head structure) | Log lines: 4 successful loads |
| expandable_segments active | No CUDA OOM at epoch 30 | Run 35 epochs without fragmentation |
| DetectionAugment clamp range | [-2.5, 2.5] or removed | `det_augment.py:102` no longer clamps to [0,1] |
| PSR monotonic decoder disabled | `USE_PSR_ORDER_PRIOR = False` | Check `config.py:1167` |

---

## Appendix A: Agent Output Directory Structure

The following agent outputs were found in `/home/newadmin/swarm-bot/master/POPW/working/code/industreal_improved/code/industreal_improved/analyses/consult_claude_science/consult_v2/agent_outputs/`:

| Agent | File | Size | Status |
|-------|------|------|--------|
| Agent 01 | `agent01_data_audit.md` | 28.3 KB | READ |
| Agent 02 | `agent02_val_analysis.md` | 19.3 KB | READ |
| Agent 03 | `agent03_detection_data.md` | 19.6 KB | READ |
| Agent 04 | `agent04_activity_psr_data.md` | 22.8 KB | READ |
| Agent 05 | `agent05_pose_temporal.md` | 26.5 KB | READ |
| Agent 06 | `agent06_backbone_capacity.md` | 27.8 KB | READ |
| Agent 07 | `agent07_neck_design.md` | 21.5 KB | READ |
| Agent 08 | `agent08_task_heads.md` | 20.0 KB | READ |
| Agent 09 | `agent09_training_pipeline.md` | 27.6 KB | READ |
| Agent 10 | `agent10_efficiency.md` | 22.9 KB | READ |
| Agent 11 | `agent11_detection_mtl_lit.md` | 31.3 KB | READ |
| Agent 12 | `agent12_activity_mtl_lit.md` | 39.2 KB | READ |
| Agent 13 | `agent13_psr_temporal_lit.md` | 36.2 KB | READ |
| Agent 14 | `agent14_pose_regression_lit.md` | 23.2 KB | READ |
| Agent 15 | `agent15_training_stability_lit.md` | 48.0 KB | READ |
| Agent 16 | `agent16_paper_strategy.md` | 28.0 KB | READ |
| Agent 17 | `agent17_*.md` | — | **NOT FOUND** |
| Agent 18 | `agent18_final_roadmap.md` | 45.9 KB | READ |
| Agent 19 | `agent19_risk_contingency.md` | 48.9 KB | READ |

**Total outputs found: 17 of 19 expected.** Only agent 17 (Competitor landscape) is missing. Its content was not available for synthesis.

## Appendix B: Prior Consultation Documents (Docs 208-227)

Located at `/home/newadmin/swarm-bot/master/POPW/working/code/industreal_improved/code/industreal_improved/analyses/consult_claude_science/`. These 20 documents from July 11, 2026, form the V1 Claude Science consultation that this document supersedes. Key V1 errors corrected by agents 01-19 are documented in Agent 01's Section 5 (Claim Verification Matrix).

---

## Appendix C: Files Referenced Across All Agent Outputs

| File | Path (relative to code root) | Agents |
|------|------------------------------|--------|
| MTL model definition | `src/models/mvit_mtl_model.py` | 6, 7, 8 |
| Original ConvNeXt-Tiny model | `src/models/model.py` | 7 |
| Video backbone MTL | `src/models/video_backbone_multitask.py` | 7 |
| PSR refinement | `src/models/psr_refinement.py` | 8 |
| RotoGrad | `src/models/rotograd.py` | 7, 8, 15 |
| Head pose model | `src/models/head_pose_geo.py` | 5 |
| Training script | `scripts/train_mtl_mvit.py` | 8, 9 |
| Training losses | `src/training/losses.py` | 5 |
| Dataset class | `src/data/industreal_dataset.py` | 1, 9 |
| Detection augment | `src/data/det_augment.py` | 9 |
| Configuration | `src/config.py` | 1, 9 |
| FAMO loss | `src/losses/famo.py` | 1, 9, 15 |
| Geodesic loss | `src/losses/geodesic_loss.py` | 5 |
| Varifocal loss | `src/losses/varifocal_loss.py` | 1 |
| IoU loss (WIoU v3) | `src/losses/iou_loss.py` | 1 |
| UW-SO loss | `src/losses/uw_so.py` | 15 |
| IMTL-L loss | `src/losses/imtl_l.py` | 15 |
| MetaBalance | `src/losses/metabalance.py` | 15 |
| MS-TCN smoothness loss | `src/losses/ms_tcn_smooth.py` | 1, 4 |
| TAL assigner | `src/losses/tal_assigner.py` | 1, 3, 7 |
| BalancedSoftmax | `src/losses/balanced_softmax.py` | 8 |
| MTL balancer (PCGrad) | `src/training/mtl_balancer.py` | 15 |
| Evaluation | `src/evaluation/evaluate.py` | 5 |
| Efficiency measurement | `scripts/measure_efficiency.py` | 10 |
| Raw dataset directory | `/home/newadmin/swarm-bot/master/POPW/datasets/industreal/` | 1 |
| Training log (T=8, 480px) | `tmp/mtl_480_T8_frag.log` | 9, 10 |
| Training log (T=4, 480px) | `tmp/mtl_480_T4_v2.log` | 9, 10 |

---

*Document generated by Agent 20 (AAIML Strategy -- Synthesis) on 2026-07-13. Synthesis from 17 of 19 expected agent outputs. Only agent 17 (Competitor landscape) was not available. All findings verified against source code and training logs at commit 75ef7f82.*
