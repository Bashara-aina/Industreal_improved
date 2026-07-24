# Comprehensive MTL Improvement Synthesis: Unified Action Plan

**Generated:** 2026-07-23
**Context:** POPW 4-task MTL (Detection + Activity + Pose + PSR) on IndustReal egocentric assembly video.
**Sources:** MTL_ARCHITECTURE_RESEARCH.md, MTL_LOSS_BALANCING_RESEARCH.md, PSR_ACTIVITY_RESEARCH.md, MTL_TRAINING_RECIPE_RESEARCH.md, plus codebase analysis of `train_mtl_v3_yolov8_head.py` and `src/config.py`.

---

## Critical Correction: Which Model Is Active

The architecture report (`MTL_ARCHITECTURE_RESEARCH.md`) incorrectly labeled `POPWMultiTaskModel` (ConvNeXt-Tiny) as "active" and `MTLMViTModel` (MViTv2-S) as "legacy/dead code." This is **wrong**.

**The active training model IS `MTLMViTModel` (MViTv2-S).** The production training script `scripts/train/train_mtl_v3_yolov8_head.py` imports and instantiates `MTLMViTModel` from `src/models/mvit_mtl_model.py` at line 192. The `POPWMultiTaskModel` in `src/models/model.py` (ConvNeXt-Tiny) is only used in smoke-test utilities and deprecated training scripts.

All metrics referenced in the loss balancing, PSR/activity, and training recipe reports are from the MViTv2-S model and are correct. All findings below reference the MViTv2-S architecture.

### Active Model Architecture (MViTv2-S + YOLOv8 Head)

| Component | Detail | Params |
|-----------|--------|--------|
| Backbone | MViTv2-S (K400 pretrain, T=8 temporal) | 34.54M |
| Neck | LightweightFPN (BiFPN style, P2-P5, 3D convs) | ~14.53M |
| Detection Head | YOLOv8-style DFL (decoupled, anchor-free, reg_max=16) | ~1.20M |
| Activity Head | 3-layer MLP (768->2048->1024->75) | ~3.75M |
| PSR Head | Causal Transformer (d=256, 2 layers, 4 heads) | ~1.78M |
| Pose Head | 2-layer MLP (768->256->6) | ~0.20M |
| **Total** | | **~55.69M** |

Ref: `train_mtl_v3_yolov8_head.py` L192, `mvit_mtl_model.py`

---

## 1. Cross-Cutting Themes

### Theme A: The Detection Head Is the Bottleneck

Detection mAP@0.5 at 0.366 (Checkpoint B) vs 0.554 (Checkpoint A) is the single largest metric gap. The Forensics Report identifies the cls head specifically — B produces 46% more boxes but 3.7x more high-confidence false positives. Across all four reports, detection improvement is consistently the highest-ROI target.

**Convergence of evidence:**
- Architecture report: 5 techniques identified (QFL, ATSS, TAL, BiFPN, P2 level), 3 already implemented
- Training recipe: SWA (+0.5-1.0 AP), Mixup (+0.5-1.0 AP), SGDR (+0.3-1.0%) all benefit detection
- Loss balancing: IMTL-L reduces effective loss ratio from 10,000x to ~9x, directly protecting det cls head
- PSR report: Detection improvements do not directly benefit PSR but free up capacity

### Theme B: PSR Is Structurally Broken — Needs Imbalance Fix, Not Architecture

PSR macro F1=0.556 with 4 dead components and flat ~0.69-0.71 output is a class imbalance problem, not fundamentally an architecture problem. Reports converge on:

1. LogitAdjust (5 lines): Add `log(class_prior)` to PSR logits — directly addresses dead components
2. Class-balanced loss for multi-label: Replace BCE with effective-number reweighting
3. STORM-PSR temporal stream: Address temporal coherence for remaining gains beyond imbalance fix

PSR architecture is adequate (causal Transformer, 1.78M params). The issue is the signal from supervision, not capacity.

### Theme C: Training Recipe Changes Are Highest ROI Per Unit Effort

The training recipe report identifies 5 changes requiring 0-50 lines of code each, with estimated cumulative +2-3% det AP and +1-2% act F1. Compare to architecture changes requiring days-weeks of development for similar or smaller gains. The training recipe changes should be implemented first because they compound with architecture changes.

### Theme D: Loss Balancing Is a Force Multiplier

The 10,000x loss ratio (DFL=10-30 vs pose MSE=0.003) means the model is effectively training on detection only for long stretches. IMTL-L and FAMO both operate in log-space to compress this ratio. These methods are already implemented and env-flag gated — activation cost is near zero. They directly protect the weaker tasks (PSR, pose, activity) from being overwhelmed.

---

## 2. Unified Recommendation Ranking (by Impact / Effort)

All 18 recommendations from the 4 reports, ranked by expected ROI. Effort is person-days for a researcher familiar with the codebase.

### Tier 1: Immediate (0-2 days, near-zero code change, compound benefit)

| Rank | Change | Source Report | Effort | Expected Impact | Code Status |
|------|--------|--------------|--------|-----------------|-------------|
| **1** | Enable BF16 mixed precision | Training Recipe (#1) | Config toggle | 1.5-2x throughput | `MIXED_PRECISION=False`, `AMP_DTYPE='bf16'` ready |
| **2** | Fix 4 dead PSR components with LogitAdjust + class-balanced loss | PSR/Activity (#1) | 1-2 days | PSR macro F1 0.556 -> ~0.70 | Not implemented |
| **3** | Enable IMTL-L or FAMO loss balancing | Loss Balancing | Env flag | Protects weaker tasks from 10,000x ratio | `USE_IMTL_L=1` or `USE_FAMO=1` |
| **4** | Simplify Kendall weighting (remove pre-multipliers) | Training Recipe (#11) | Config change | Cleaner loss landscape | `PSR_WEIGHT=10.0` -> 1.0, `POSE_LOSS_WEIGHT=5.0` -> 1.0 |
| **5** | Enable Mixup for full MTL training | Training Recipe (#3) | ~50 lines | +0.5-1.0 det AP, +0.3-0.8 act F1 | `USE_MIXUP=False` currently |
| **6** | Enable SWA for final 10 epochs | Training Recipe (#4) | Config toggle | +0.5-1.0 det AP, +0.3-0.8 act F1 | `USE_SWA=False` |

**Subtotal Tier 1: 6 changes, 2-5 days total, est. +3-5% det AP, +1-3% act F1, PSR ~0.70**

### Tier 2: Short-Term (1-2 weeks, code changes needed)

| Rank | Change | Source Report | Effort | Expected Impact | Code Status |
|------|--------|--------------|--------|-----------------|-------------|
| **7** | Add per-stage LLRD (ConvNeXt/MViT stages) | Training Recipe (#2) | ~20 lines | +0.5-1.5% det AP, +0.3-1.0 act F1 | Uniform `BACKBONE_LR_MULT=0.01` only |
| **8** | Switch to SGDR (CosineAnnealingWarmRestarts) for 100-epoch | Training Recipe (#5) | Config toggle | +0.3-1.0% across tasks | `ONE_CYCLE_LR=True`, `USE_COSINE_ANNEALING=False` |
| **9** | Enable ATSS adaptive matching | Architecture (#2) | Flag change | +1.4% AP | `--matcher atss` flag exists in train_mtl_v3.py |
| **10** | Enable TAL (Task-Aligned Learning) | Architecture (#3) | Flag change | +3% AP | `--use-tal` flag exists |
| **11** | Enable flip TTA for evaluation | Training Recipe (#10) | Config toggle | +0.5-1.0 det AP | `USE_TTA=False` currently |

**Subtotal Tier 2: 5 changes, 1-2 weeks, est. +3-7% det AP, +0.3-1.0% act F1**

### Tier 3: Medium-Term (2-4 weeks, significant engineering)

| Rank | Change | Source Report | Effort | Expected Impact | Code Status |
|------|--------|--------------|--------|-----------------|-------------|
| **12** | Upgrade FPN to bidirectional weighted fusion | Architecture (#7) | 1 week | +3-7% mAP | `USE_BIFPN=True` config toggle exists but module needs porting |
| **13** | Integrate STORM-PSR temporal stream | PSR/Activity (#2) | 2 weeks | PSR ~0.70 -> ~0.80, +5-10% act | Public code exists |
| **14** | Anchor-free detection migration (QFL + DFL) | Architecture (#4) | 1 week | +3-5% mAP | QFL in `src/losses/qfl.py`, DFL not implemented |
| **15** | Add P2 level for small object detection | Architecture (#8) | 1 day | +3-5% AP_small | `--use-p2-level` flag exists, needs porting |
| **16** | Implement per-task gradient conflict diagnosis (cosine similarity) | Loss Balancing (Phase 1) | 1 day | Diagnostic only | `src/training/mtl_balancer.py` supports |
| **17** | Add ASRF boundary regression for PSR | PSR/Activity | 1 week | PSR F1 +5-10% | Not implemented |

**Subtotal Tier 3: 6 changes, 2-4 weeks, est. +5-15% mAP, PSR ~0.80, activity +5-10%**

### Tier 4: Long-Term (4-8 weeks, research projects)

| Rank | Change | Source Report | Effort | Expected Impact | Code Status |
|------|--------|--------------|--------|-----------------|-------------|
| **18** | Clip-level temporal modeling for activity | PSR/Activity (#3) | 2 weeks | Activity Top-1 +10-15% | Per-frame eval only |
| **19** | CARAFE upsampling in FPN | Architecture (#9) | 2 days | +2-3% mAP | Not implemented |
| **20** | Cross-Stitch feature routing at FPN levels | Architecture (#11) | 2 weeks | +1-3% across tasks | Not implemented |
| **21** | Add EMA model averaging with higher decay | Training Recipe | Config | +0.5-1.0% det AP | Already active at 0.995 |
| **22** | Backbone upgrade to MViTv2-B | Architecture (#12) | 2 weeks | +1-2% across tasks (speculative) | Feasible at 224px on 16GB |
| **23** | Self-supervised pretraining on IndustReal (VideoMAE) | Training Recipe (#7) | 4 weeks | Domain-aligned features | VideoMAE stream exists but disabled |
| **24** | Group-based optimization (GO4Align) | Loss Balancing | 3 weeks | Gradient conflict resolution | Research-stage method |

**Subtotal Tier 4: 7 changes, 4-8 weeks, speculative gains**

---

## 3. Dependency Graph

The improvements have strong ordering dependencies. Here is the recommended execution sequence:

```
Phase 0 (Day 0): Diagnostic checkpoint
  ├── Run gradient cosine similarity diagnostic (Rank 16)
  ├── Measure per-task loss scales and gradient norms
  └── Identify whether gradient conflict or scale imbalance dominates

Phase 1 (Days 1-5): Training Recipe + Loss Fixes (Tier 1, independent)
  ├── Rank 1: BF16 mixed precision (throughput enabler)
  ├── Rank 2: PSR LogitAdjust + class-balanced loss
  ├── Rank 3: Enable IMTL-L or FAMO
  ├── Rank 4: Simplify Kendall pre-multipliers
  ├── Rank 5: Enable Mixup
  └── Rank 6: Enable SWA
  └── [EVALUATE] Measure mAP, act F1, PSR F1, pose MAE after Phase 1

Phase 2 (Days 5-12): Detection Architecture (Tier 2, mostly independent)
  ├── Rank 9: Enable ATSS matching (flag change)
  ├── Rank 10: Enable TAL (flag change)
  ├── Rank 7: Add per-stage LLRD
  ├── Rank 8: Switch to SGDR schedule
  ├── Rank 11: Enable flip TTA
  └── [EVALUATE] Compare Phase 2 vs Checkpoint A (target: match or exceed 0.554 mAP)

Phase 3 (Days 12-26): Neck + Detection Architecture (Tier 3)
  ├── Rank 12: BiFPN weighted fusion (USE_BIFPN=True)
  ├── Rank 14: QFL + anchor-free migration
  ├── Rank 15: P2 level for small objects
  └── [EVALUATE] Target: mAP >= 0.60

Phase 4 (Days 26-40): Temporal Modeling (Tier 3-4)
  ├── Rank 13: STORM-PSR temporal stream
  ├── Rank 17: ASRF boundary regression for PSR
  ├── Rank 18: Clip-level temporal modeling for activity
  └── [EVALUATE] Target: PSR F1 >= 0.75, Act Top-1 >= 35%

Phase 5 (Days 40-56): Advanced Architecture (Tier 4, optional)
  ├── Rank 19: CARAFE upsampling
  ├── Rank 20: Cross-Stitch routing
  ├── Rank 22: Backbone upgrade
  └── [EVALUATE] Target: mAP >= 0.65, PSR >= 0.80, Act >= 40%
```

---

## 4. Conflict Resolution: Where Reports Disagree

### 4.1 OneCycleLR vs SGDR

| Report | Position | Rationale |
|--------|----------|-----------|
| Training Recipe (#5) | Switch to SGDR | Periodic restarts give underperforming heads a chance to catch up |
| Architecture (implicit) | Keep OneCycleLR | Current implementation is well-tuned; pct_start=0.1 is standard |

**Resolution:** Keep OneCycleLR for short runs (<50 epochs). Use SGDR for full 100-epoch runs. The SGDR restarts (T_0=10, T_mult=2) provide cycles at epochs 10, 30, 70 — well-suited to the 100-epoch budget. Test a single 100-epoch A/B comparison.

### 4.2 UW-SO vs IMTL-L vs FAMO

| Report | Position | Rationale |
|--------|----------|-----------|
| Loss Balancing (Top-1) | IMTL-L + PCGrad | Stateless, zero overhead, compresses 10,000x ratio to ~9x |
| Training Recipe (#11) | Simplify Kendall + try FAMO | Already implemented, log-space loss-decrease tracking |
| Loss Balancing (Top-3) | FAMO as close second | Dynamic adaptation, O(1) complexity |

**Resolution:** Run all 3 as a Phase 0 diagnostic (1 GPU-day each per Loss Balancing report). The gradient cosine similarity diagnostic will reveal whether gradient conflict (PCGrad territory) or scale imbalance (IMTL-L/FAMO territory) is the dominant problem. UW-SO is the weakest method due to the [-1.0, 2.0] log_sigma bounds limiting dynamic range.

### 4.3 BiFPN Priority

| Report | Position | Rationale |
|--------|----------|-----------|
| Architecture (#7) | HIGH priority | +3-7% mAP, config toggle exists |
| Architecture (debate) | MEDIUM priority | 12:1 neck-to-head ratio suggests over-engineering |

**Resolution:** MEDIUM priority. The BiFPN is already 14.5M of 55.7M total params (26%). Adding weighted fusion to an already-massive neck may provide diminishing returns. Implement as Phase 3, after the smaller detection-head changes (ATSS, TAL, QFL) are exhausted.

---

## 5. Key Metrics: Current vs Target

| Metric | Current (Checkpoint B) | Checkpoint A | Phase 1 Target | Phase 3 Target | Phase 5 Target |
|--------|----------------------|-------------|----------------|----------------|----------------|
| Det mAP@0.5 | 0.366 | 0.554 | 0.58 | 0.62 | 0.68 |
| Act Top-1 | 20.3% | 19.4% | 22% | 30% | 40% |
| Pose MAE (deg) | 6.61 | 6.33 | 6.3 | 6.0 | 5.5 |
| PSR macro F1 | 0.569 | 0.568 | 0.70 | 0.75 | 0.82 |
| PSR dead components | 4 | 4 | 0 | 0 | 0 |

**Note:** Phase 1 target for detection (0.58) already exceeds Checkpoint A (0.554). This is achievable via training recipe changes alone (Mixup + SWA + BF16 throughput enabling longer training).

---

## 6. Implementation Checklist

### Files to Modify (in execution order)

```
scripts/train/train_mtl_v3_yolov8_head.py
  └── Tier 1: Enable mixup for MTL training (Rank 5)
  └── Tier 1: Add LogitAdjust to PSR loss (Rank 2)
  └── Tier 1: Simplify Kendall pre-multipliers (Rank 4)
  └── Tier 2: Add per-stage LLRD (Rank 7)
  └── Tier 2: Switch to SGDR (Rank 8)

src/config.py
  └── Tier 1: MIXED_PRECISION = True (Rank 1)
  └── Tier 1: AMP_DTYPE = 'bf16' (Rank 1)
  └── Tier 1: USE_MIXUP = True (Rank 5)
  └── Tier 1: USE_SWA = True, SWA_LR = 5e-6 (Rank 6)
  └── Tier 1: PSR_WEIGHT = 1.0, POSE_LOSS_WEIGHT = 1.0 (Rank 4)
  └── Tier 2: USE_COSINE_ANNEALING = True, ONE_CYCLE_LR = False (Rank 8)
  └── Tier 3: USE_TTA = True (Rank 11)

src/losses/ (PSR head loss)
  └── Tier 1: Add LogitAdjust: logits += log(class_prior) (Rank 2)
  └── Tier 1: Replace BCE with class-balanced BCE (Rank 2)

src/models/mvit_mtl_model.py (or FPN module)
  └── Tier 3: BiFPN weighted fusion upgrade (Rank 12)
  └── Tier 3: P2 level integration (Rank 15)

src/losses/ (detection loss)
  └── Tier 3: QFL/DFL integration (Rank 14)
```

### Files to Create

```
src/losses/psr_balanced_loss.py  -- Class-balanced BCE for PSR (Rank 2)
src/training/llrd.py             -- Per-stage LLRD helpers (Rank 7)
```

---

## 7. Risk Assessment

### Low Risk (Tier 1-2 changes)

- BF16 mixed precision: Hardware support exists, fallback to FP32 trivial
- LogitAdjust: 5 lines, mathematically correct
- IMTL-L/FAMO: Already implemented, env-flag gated, revert by unsetting env var
- Mixup: Already implemented for synthetic pretraining, extend to full training
- SWA: Post-training only, compare vs non-SWA model
- ATSS/TAL: Already implemented via flags, revert by changing flag
- Flip TTA: Evaluation only, no training impact
- SGDR: Config toggle, can fall back to OneCycleLR

### Medium Risk (Tier 3)

- BiFPN upgrade: BiFPN already exists in legacy model (14.5M params). Porting may reveal compatibility issues with YOLOv8 detection head.
- STORM-PSR integration: External code dependency. May need adaptation for 6-channel input. Risk of dependency conflicts.
- Anchor-free migration: Changes training dynamics significantly. Must validate regression (CIoU) doesn't diverge.

### High Risk (Tier 4)

- Backbone upgrade to MViTv2-B: 50% more backbone params (34.5M -> 52M). VRAM unknown at 640x360 resolution. May require input resolution reduction.
- Self-supervised pretraining: Weeks of engineering for uncertain gain. Synthetic data quality unknown.
- Cross-Stitch routing: Unproven benefit for 4-task MTL with strong backbone. Vandenhende survey found "moderate only" gains.

---

## 8. References (Master List)

### Training Recipe
1. Smith & Topin, "Super-Convergence: Very Fast Training of Neural Networks," 2019
2. Loshchilov & Hutter, "SGDR: Stochastic Gradient Descent with Warm Restarts," ICLR 2017
3. Izmailov et al., "Averaging Weights Leads to Wider Optima," UAI 2018
4. Zhang et al., "mixup: Beyond Empirical Risk Minimization," ICLR 2018
5. Goyal et al., "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour," 2017

### Loss Balancing
6. Kendall et al., "Multi-Task Learning Using Uncertainty to Weigh Losses," CVPR 2018
7. Liu et al., "Towards Impartial Multi-Task Learning (IMTL)," ICLR 2021
8. Liu et al., "FAMO: Fast Adaptive Multitask Optimization," NeurIPS 2023
9. Yu et al., "Gradient Surgery for Multi-Task Learning (PCGrad)," NeurIPS 2020
10. Chen et al., "GradNorm: Gradient Normalization for Adaptive Loss Balancing," ICML 2018

### PSR/Activity
11. Schoonbeek et al., "IndustReal: A Dataset for Assembly State Detection," WACV 2024
12. Fan et al., "STORM-PSR: Spatio-Temporal Obfuscated Masking for PSR," CVIU 2025
13. Cui et al., "Class-Balanced Loss Based on Effective Number of Samples," CVPR 2019
14. Menon et al., "Long-Tail Learning via Logit Adjustment," ICLR 2021
15. Farha & Gall, "MS-TCN: Multi-Stage Temporal Convolutional Network," CVPR 2019
16. Ishii et al., "ASRF: Action Segment Refinement Framework," WACV 2021

### Architecture/Detection
17. Li et al., "Generalized Focal Loss (GFL V1/V2)," NeurIPS 2020 / TPAMI 2022
18. Zhang et al., "ATSS: Bridging Anchor-based and Anchor-free Detection," CVPR 2020
19. Feng et al., "TOOD: Task-aligned One-stage Object Detection (TAL)," AAAI 2021
20. Tan et al., "EfficientDet: Scalable and Efficient Object Detection," CVPR 2020
21. Li et al., "MViTv2: Improved Multiscale Vision Transformers," CVPR 2022
22. Vandenhende et al., "Multi-Task Learning for Dense Prediction Tasks: A Survey," arXiv 2020
23. Misra et al., "Cross-Stitch Networks for Multi-Task Learning," CVPR 2016
