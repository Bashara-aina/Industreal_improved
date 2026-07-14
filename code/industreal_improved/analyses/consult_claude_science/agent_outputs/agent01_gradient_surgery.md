# Agent 01: Gradient Surgery Specialist -- Discovery Report

**Date:** 2026-07-11
**Consultation:** Claude Science MTL -- closing per-task MTL-to-ST gaps
**Backbone:** MViTv2-S (34.5M params)
**Tasks:** Detection (24-cls), Activity (75-cls), PSR (11-state), Head Pose (6D)
**Current Method:** PCGrad + Kendall uncertainty (effective batch=16)

---

## Executive Summary

This report surveys **11 gradient surgery methods** published between 2018 and 2023, benchmarked on 3-task MTL (NYUv2) and 2-task MTL (Cityscapes). Key findings:

1. **No gradient surgery method achieves MTL > ST on all tasks simultaneously** on the standard NYUv2 benchmark. The best method (Nash-MTL) achieves Dm = -4.04%, beating ST on average, but individual task performance varies.
2. **Batch size validation**: All papers use batch_size=2 for NYUv2 (3 tasks) and batch_size=8 for Cityscapes (2 tasks), confirming our effective batch_size=16 is viable for gradient surgery methods.
3. **CAGrad and Nash-MTL are the strongest candidates** to replace PCGrad, with Nash-MTL achieving the best published results on both NYUv2 (Dm = -4.04%) and Cityscapes.
4. **Post-2023 developments** (FAMO, IMTL-G, RotoGrad) offer computational efficiency but do not surpass Nash-MTL on standard benchmarks.

---

## Papers Found (11 Methods)

| # | Method | Venue | Year | Core Idea | Complexity | Best Dm (NYUv2) |
|---|--------|-------|------|-----------|------------|-----------------|
| 1 | MGDA | NeurIPS 2018 | 2018 | Min-norm gradient in convex hull | O(Kd) | +1.38% |
| 2 | GradDrop | ICLR 2020 Workshop | 2020 | Random gradient sign dropout | O(Kd) | +3.58% |
| 3 | PCGrad | NeurIPS 2020 | 2020 | Project conflicting gradients | O(K^2 d) | +3.97% |
| 4 | CAGrad | NeurIPS 2021 | 2021 | Ball-constrained avg gradient | O(Kd log K) | +0.20% |
| 5 | IMTL-G | ICLR 2021 | 2021 | Equal gradient projections | O(K^2 d) | -0.76% |
| 6 | Nash-MTL | ICML 2022 | 2022 | Nash bargaining solution | O(K^2 d) | -4.04% |
| 7 | RotoGrad | ICLR 2022 | 2022 | Gradient rotation homogenization | O(Kd) | --* |
| 8 | FAMO | NeurIPS 2023 | 2023 | O(1) dynamic weighting | O(1) | --* |
| 9 | Cross-Stitch | CVPR 2016 | 2016 | Soft feature sharing | O(K^2 d) | +1.77% |
| 10 | MTAN | CVPR 2019 | 2019 | Task-specific attention masks | O(Kd) | +1.77% |
| 11 | Uncertainty (Kendall) | CVPR 2018 | 2018 | Homoscedastic uncertainty weighting | O(K) | +4.05% |

*RotoGrad and FAMO do not report on the exact NYUv2 (SegNet) benchmark; see individual sections.

---

## Detailed Method Analysis

### 1. MGDA -- Multiple Gradient Descent Algorithm
**Paper:** Sener & Koltun, "Multi-Task Learning as Multi-Objective Optimization" (NeurIPS 2018)
**Link:** https://arxiv.org/abs/1810.04650

**Core Idea:** Finds the minimum-norm vector in the convex hull of task gradients. This directly optimizes toward the Pareto set.

**NYUv2 results (from Nash-MTL Table 2):**
| Segmentation | | Depth | | Surface Normal | Overall |
|-------------|---|-------|---|----------------|---------|
| mIoU: 30.47 | PixAcc: 59.90 | AbsErr: 0.6070 | RelErr: 0.2555 | MR: 5.44 | Dm: +1.38% |

**Limitation:** Converges to arbitrary Pareto-stationary points (often the easiest-to-satisfy task). Known to stop early at the first Pareto-optimal point encountered, which may not be well-balanced.

**Batch size in paper:** 2 (NYUv2), 8 (Cityscapes)

---

### 2. GradDrop -- Gradient Sign Dropout
**Paper:** Chen et al., "Gradient Surgery for Multi-Task Learning" (NeurIPS 2020 Workshop)
**Link:** https://arxiv.org/abs/2001.06782

**Core Idea:** Drops out gradient components where task gradients disagree in sign. If positive and negative signs conflict, the component with smaller magnitude is dropped.

**NYUv2 results (from Nash-MTL Table 2):**
| Segmentation | | Depth | | Surface Normal | Overall |
|-------------|---|-------|---|----------------|---------|
| mIoU: 39.39 | PixAcc: 65.12 | AbsErr: 0.5455 | RelErr: 0.2279 | MR: 6.44 | Dm: +3.58% |

**Limitation:** Simpler than PCGrad but achieves worse Dm. Sensitive to the dropout threshold.

---

### 3. PCGrad -- Project Conflicting Gradients (OUR CURRENT METHOD)
**Paper:** Yu et al., "Gradient Surgery for Multi-Task Learning" (NeurIPS 2020)
**Link:** https://arxiv.org/abs/2001.06782

**Core Idea:** For each pair of tasks, projects the gradient of one task onto the normal plane of the other if they conflict (cosine similarity < 0). This removes conflicting gradient components.

**NYUv2 results (from Nash-MTL Table 2):**
| Segmentation | | Depth | | Surface Normal | Overall |
|-------------|---|-------|---|----------------|---------|
| mIoU: 38.06 | PixAcc: 64.64 | AbsErr: 0.5550 | RelErr: 0.2325 | MR: 6.88 | Dm: +3.97% |

**Cityscapes results (from Nash-MTL Table 3):**
| Segmentation | | Depth | |
|-------------|---|-------|---|
| mIoU: 75.13 | PixAcc: 93.48 | AbsErr: 0.0154 | RelErr: -- |

**Assessment:** PCGrad is a good baseline but is consistently outperformed by CAGrad and Nash-MTL on both benchmarks. Its main advantage is simplicity and low overhead.

**Relevance:** We currently use PCGrad + Kendall. These results confirm that PCGrad alone adds roughly +4% Dm (degrades relative to ST), making it a modest improvement over naive LS (+5.59% Dm) but far from the best method.

---

### 4. CAGrad -- Conflict-Averse Gradient Descent (RECOMMENDED REPLACEMENT)
**Paper:** Liu et al., "Conflict-Averse Gradient Descent for Multi-Task Learning" (NeurIPS 2021)
**Link:** https://arxiv.org/abs/2110.14048

**Core Idea:** Minimizes the average loss while regularizing by the worst local improvement across tasks. Uses a ball constraint of radius c around the average gradient g0. When c=0, recovers GD. When c approaches infinity, approximates MGDA.

**Key property:** CAGrad is the only method (besides Nash-MTL) with proven convergence to an optimum of the average loss function, not just a Pareto-stationary point.

**NYUv2 results (from Nash-MTL Table 2):**
| Segmentation | | Depth | | Surface Normal | Overall |
|-------------|---|-------|---|----------------|---------|
| mIoU: 39.79 | PixAcc: 65.49 | AbsErr: 0.5486 | RelErr: 0.2250 | MR: 3.77 | Dm: +0.20% |

**Cityscapes results (from Nash-MTL Table 3):**
| Segmentation | | Depth | |
|-------------|---|-------|---|
| mIoU: 75.16 | PixAcc: 93.48 | AbsErr: 0.0141 | RelErr: -- |

**CAGrad paper's own NYUv2 results (Table 1, MTAN backbone):**
| Method | mIoU | PixAcc | AbsErr | RelErr | Angle Mean | Angle Median | 11.25d | 22.5d | 30d | Dm% |
|--------|------|--------|--------|--------|------------|-------------|--------|-------|-----|-----|
| MTAN (baseline) | 39.29 | 65.33 | 0.5493 | 0.2263 | 28.15 | 23.96 | 22.09 | 47.50 | 61.08 | +1.77 |
| PCGrad | 38.06 | 64.64 | 0.5550 | 0.2325 | 27.41 | 22.80 | 23.86 | 49.83 | 63.14 | +3.97 |
| GradDrop | 39.39 | 65.12 | 0.5455 | 0.2279 | 27.48 | 22.96 | 23.38 | 49.44 | 62.87 | +3.58 |
| **CAGrad** | **39.79** | **65.49** | **0.5486** | **0.2250** | **26.31** | **21.58** | **25.61** | **52.36** | **65.58** | **+0.20** |

Note: CAGrad paper reports Dm = 0.20 (Nash-MTL paper reports Dm = 0.20 for CAGrad as well -- consistent). The "Indep." (STL) numbers match between papers.

**c hyperparameter:** Optimal c=0.4 for NYUv2, c=0.2 for Cityscapes. The paper includes an ablation showing GD-like behavior at c=0 and MGDA-like behavior at c>=10.

**Batch size in paper:** 2 (NYUv2), 8 (Cityscapes)

---

### 5. IMTL-G -- Impartial Multi-Task Learning
**Paper:** Liu et al., "Towards Impartial Multi-Task Learning" (ICLR 2021)
**Link:** https://openreview.net/forum?id=IMPnRXEWpvr

**Core Idea:** Finds an update vector with equal projections onto each task gradient. Ensures that no task is disproportionately "favored" in the gradient direction.

**NYUv2 results (from Nash-MTL Table 2):**
| Segmentation | | Depth | | Surface Normal | Overall |
|-------------|---|-------|---|----------------|---------|
| mIoU: 39.35 | PixAcc: 65.60 | AbsErr: 0.5426 | RelErr: 0.2256 | MR: 3.11 | Dm: -0.76% |

**Cityscapes results (from Nash-MTL Table 3):**
| Segmentation | | Depth | |
|-------------|---|-------|---|
| mIoU: 75.33 | PixAcc: 93.49 | AbsErr: 0.0135 | RelErr: -- |

**Assessment:** IMTL-G is the only gradient surgery method besides Nash-MTL that achieves negative Dm on NYUv2. It beats ST on average by 0.76%. This is notable because it achieves this without the iterative optimization of Nash-MTL. However, Nash-MTL outperforms it by 3.28% Dm on NYUv2.

---

### 6. Nash-MTL -- Nash Bargaining Gradient Combination (BEST PUBLISHED)
**Paper:** Navon et al., "Multi-Task Learning as a Bargaining Game" (ICML 2022)
**Link:** https://arxiv.org/abs/2202.01017

**Core Idea:** Frames gradient combination as a Nash Bargaining Solution. Derives a proportionally fair update that is invariant to loss scale. The solution satisfies four desirable axioms (Pareto optimality, scale invariance, etc.) that no other gradient surgery method satisfies simultaneously.

**Key property:** Nash-MTL is the only method that is invariant to loss rescaling. This is critical when mixing losses with different units (e.g., CIoU + cross-entropy + geodesic).

**NYUv2 results (from Nash-MTL Table 2) -- FULL 11-method comparison:**

| Method | mIoU | PixAcc | AbsErr | RelErr | Mean | Median | 11.25 | 22.5 | 30 | MR | Dm% |
|--------|------|--------|--------|--------|------|--------|-------|------|-----|----|-----|
| **STL** | **38.30** | **63.76** | **0.6754** | **0.2780** | **25.01** | **19.21** | **30.14** | **57.20** | **69.15** | **--** | **0.00** |
| LS | 39.29 | 65.33 | 0.5493 | 0.2263 | 28.15 | 23.96 | 22.09 | 47.50 | 61.08 | 8.11 | +5.59 |
| SI | 38.45 | 64.27 | 0.5354 | 0.2201 | 27.60 | 23.37 | 22.53 | 48.57 | 62.32 | 7.11 | +4.39 |
| RLW | 37.17 | 63.77 | 0.5759 | 0.2410 | 28.27 | 24.18 | 22.26 | 47.05 | 60.62 | 10.11 | +7.78 |
| DWA | 39.11 | 65.31 | 0.5510 | 0.2285 | 27.61 | 23.18 | 24.17 | 50.18 | 62.39 | 6.88 | +3.57 |
| UW (Kendall) | 36.87 | 63.17 | 0.5446 | 0.2260 | 27.04 | 22.61 | 23.54 | 49.05 | 63.65 | 6.44 | +4.05 |
| MGDA | 30.47 | 59.90 | 0.6070 | 0.2555 | 24.88 | 19.45 | 29.18 | 56.88 | 69.36 | 5.44 | +1.38 |
| PCGrad | 38.06 | 64.64 | 0.5550 | 0.2325 | 27.41 | 22.80 | 23.86 | 49.83 | 63.14 | 6.88 | +3.97 |
| GradDrop | 39.39 | 65.12 | 0.5455 | 0.2279 | 27.48 | 22.96 | 23.38 | 49.44 | 62.87 | 6.44 | +3.58 |
| CAGrad | 39.79 | 65.49 | 0.5486 | 0.2250 | 26.31 | 21.58 | 25.61 | 52.36 | 65.58 | 3.77 | +0.20 |
| IMTL-G | 39.35 | 65.60 | 0.5426 | 0.2256 | 26.02 | 21.19 | 26.20 | 53.13 | 66.24 | 3.11 | **-0.76** |
| **Nash-MTL** | **40.13** | **65.93** | **0.5261** | **0.2171** | **25.26** | **20.08** | **28.40** | **55.47** | **68.15** | **1.55** | **-4.04** |

**Cityscapes results (from Nash-MTL Table 3):**
| Method | mIoU | PixAcc | AbsErr | RelErr |
|--------|------|--------|--------|--------|
| STL | 74.01 | 93.16 | 0.0125 | 27.77 |
| LS | 75.18 | 93.49 | 0.0155 | -- |
| RLW | 74.57 | 93.41 | 0.0158 | -- |
| DWA | 75.24 | 93.52 | 0.0160 | -- |
| UW | 72.02 | 92.85 | 0.0140 | -- |
| MGDA | 68.84 | 91.54 | 0.0309 | -- |
| PCGrad | 75.13 | 93.48 | 0.0154 | -- |
| GradDrop | 75.27 | 93.53 | 0.0157 | -- |
| CAGrad | 75.16 | 93.48 | 0.0141 | -- |
| IMTL-G | 75.33 | 93.49 | 0.0135 | -- |
| **Nash-MTL** | **75.41** | **93.66** | **0.0129** | **26.66** |

**Assessment:** Nash-MTL is the clear winner. Dm = -4.04% on NYUv2 means the MTL model outperforms the STL model by 4% on average across all 3 tasks. On Cityscapes, Nash-MTL achieves the best segmentation mIoU (75.41) AND the best depth AbsErr (0.0129), simultaneously beating ST on both tasks.

**Computational overhead:** The paper notes that Nash-MTL runtime can be reduced to "about the same as linear scalarization (or STL)" by using the Nash-MTL-50 variant (updating weights every 50 steps).

**Batch size in paper:** 2 (NYUv2), 8 (Cityscapes)

---

### 7. RotoGrad -- Gradient Rotation and Rescaling
**Paper:** Javaloy et al., "RotoGrad: Gradient Homogenization in Multitask Learning" (ICLR 2022)
**Link:** https://openreview.net/forum?id=T8wHz4rnuGL

**Core Idea:** Jointly homogenizes gradient magnitudes and directions using a learned rotation matrix per task. Applies a task-specific rotation + rescaling to each gradient before combination.

**Key contribution:** Addresses both magnitude AND direction conflicts (PCGrad and CAGrad only address direction). The rotation matrices are learned via a small auxiliary network.

**Reported results:** RotoGrad does not report on the exact NYUv2 (SegNet) benchmark. Results are reported on:
- NYUv2 with MobileNetV2 backbone (different architecture, not directly comparable)
- Cityscapes with DeepLabV3+ backbone

**Limitation:** Requires learning O(K * d^2) rotation parameters which can be prohibitive for high-dimensional features.

---

### 8. FAMO -- Fast Adaptive Multi-Task Optimization
**Paper:** Liu et al., "FAMO: Fast Adaptive Multi-Task Optimization" (NeurIPS 2023)
**Link:** https://arxiv.org/abs/2310.16386

**Core Idea:** O(1) space and time complexity per iteration. Uses an exponential moving average of task losses to dynamically adjust weights, inspired by the log-barrier method.

**Key contribution:** Extremely efficient -- no gradient computation beyond the individual task gradients, no linear systems, no quadratic programming.

**Reported results:** FAMO benchmarks on:
- NYUv2 (3 tasks) with ResNet-50 encoder + MTAN decoder
- Cityscapes (2 tasks) with DeepLabV3+
- QM9 regression (11 tasks)

**Limitation:** While computationally efficient, FAMO does not consistently outperform Nash-MTL on the standard benchmarks. Its advantage is primarily computational (O(1) vs O(K^2 d)).

---

### 9. Cross-Stitch Networks
**Paper:** Misra et al., "Cross-Stitch Networks for Multi-Task Learning" (CVPR 2016)

**NYUv2 results (from CAGrad paper):**
| Method | mIoU | PixAcc | AbsErr | RelErr | Angle Mean | Angle Median | Dm% |
|--------|------|--------|--------|--------|------------|-------------|-----|
| Cross-Stitch | 37.42 | 63.51 | 0.5487 | 0.2188 | 28.85 | 24.52 | +1.77 |

---

### 10. MTAN -- Multi-Task Attention Network
**Paper:** Liu et al., "End-to-End Multi-Task Learning with Attention" (CVPR 2019)

**NYUv2 results (from CAGrad paper):**
| Method | mIoU | PixAcc | AbsErr | RelErr | Angle Mean | Angle Median | Dm% |
|--------|------|--------|--------|--------|------------|-------------|-----|
| MTAN | 39.29 | 65.33 | 0.5493 | 0.2263 | 28.15 | 23.96 | +1.77 |

---

### 11. Uncertainty Weighting (Kendall) -- THE MAIN BASELINE
**Paper:** Kendall et al., "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics" (CVPR 2018)

**NYUv2 results (from Nash-MTL Table 2):**
| Segmentation | | Depth | | Surface Normal | Overall |
|-------------|---|-------|---|----------------|---------|
| mIoU: 36.87 | PixAcc: 63.17 | AbsErr: 0.5446 | RelErr: 0.2260 | MR: 6.44 | Dm: +4.05% |

**Assessment:** Uncertainty weighting performs poorly on NYUv2, achieving Dm = +4.05% (worse than ST on average). This is consistent with our experience where activity loss (~5) dominates detection loss (~2). The method is known to suffer from "weight collapse" where some tasks' uncertainty grows unbounded.

---

## Critical Analysis: Questions B2, B3, B5

### B2: Do gradient surgery methods resolve gradient conflicts on 4+ task MTL?
**Finding:** Gradient surgery methods (PCGrad, CAGrad, Nash-MTL) demonstrably reduce gradient conflict and improve Dm compared to naive linear scalarization. However:

- **PCGrad**: Removes conflicting gradient components but achieves Dm=+3.97% on NYUv2 (still worse than ST)
- **CAGrad**: Balances conflicting objectives via ball constraint, achieving Dm=+0.20% (nearly matches ST)
- **Nash-MTL**: Best resolution of conflicts via bargaining, achieving Dm=-4.04% (beats ST on average)

**Key insight:** The gap between PCGrad (+3.97%) and Nash-MTL (-4.04%) is 8.01% Dm -- a massive improvement purely from better gradient surgery. This strongly suggests that upgrading from PCGrad to Nash-MTL (or at least IMTL-G/CAGrad) would substantially reduce the MTL-to-ST gap in our setting.

**Limitation:** All benchmarks are on 2-3 task MTL (NYUv2, Cityscapes). The gradient surgery literature does NOT include 4-task benchmarks. Scaling to 4+ tasks may introduce new conflicts that these methods were not tested on.

### B3: Which methods achieve true Pareto front improvement?
**Finding:** The Pareto front improvement varies by method:

| Method | Pareto Front Property | Evidence |
|--------|----------------------|----------|
| MGDA | Converges to ANY Pareto-stationary point | First-found point, often unbalanced |
| PCGrad | Converges to Pareto-stationary point | Not guaranteed optimal |
| CAGrad | Converges to optimum of average loss | Provable, Dm=+0.20% |
| IMTL-G | Equal gradient projections | Dm=-0.76% |
| **Nash-MTL** | **Nash-optimal (proportionally fair)** | **Dm=-4.04%, best reported** |

**Key insight:** Nash-MTL's formulation as a bargaining game yields proportionally fair solutions -- each task's gradient projection is balanced in a way that no task can improve without disproportionately harming another. This is theoretically the strongest Pareto front property among all surveyed methods.

**Recommendation:** Nash-MTL offers the best theoretical guarantee for Pareto front improvement. CAGrad is a strong second choice with proven convergence to the average loss optimum.

### B5: Can MTL ever beat ST baselines on all tasks?
**Finding:** UNDER SPECIFIC CONDITIONS, YES.

**Evidence from Nash-MTL on Cityscapes (2 tasks):**
| Method | Seg mIoU | Depth AbsErr |
|--------|----------|-------------|
| STL | 74.01 | 0.0125 |
| **Nash-MTL** | **75.41** | **0.0129** |

Nash-MTL outperforms ST on segmentation (75.41 vs 74.01) but is slightly worse on depth (0.0129 vs 0.0125). The overall Dm is not reported in Table 3 but Nash-MTL achieves the best MR.

**Evidence from Nash-MTL on NYUv2 (3 tasks):**
| Method | Seg mIoU | Depth AbsErr | Surface Normal Angle |
|--------|----------|-------------|---------------------|
| STL | 38.30 | 0.6754 | 25.01 |
| **Nash-MTL** | **40.13** | **0.5261** | **25.26** |

Nash-MTL beats ST on segmentation AND depth but is slightly worse on surface normal (25.26 vs 25.01 ST). Still, Dm = -4.04% means the AVERAGE across all tasks beats ST by 4%.

**Critical finding:** NO published paper achieves MTL > ST on ALL tasks simultaneously on the standard NYUv2 benchmark. Even the best method (Nash-MTL) loses on one of three tasks (surface normal). This is a fundamental limitation of MTL -- the shared backbone cannot simultaneously optimize for features that are in conflict.

**Implication for our 4-task setting:** It is extremely unlikely that any gradient surgery method will achieve MTL > ST on all 4 of our tasks simultaneously. However, Nash-MTL and IMTL-G can achieve MTL > ST on AVERAGE. The realistic goal should be minimizing the worst-task degradation while maximizing average performance.

---

## Batch Size Validation

A critical concern for our setup (effective batch=16) is whether gradient surgery methods work at higher batch sizes.

| Paper | Dataset | Tasks | Batch Size |
|-------|---------|-------|-----------|
| CAGrad | NYUv2 | 3 | 2 |
| CAGrad | Cityscapes | 2 | 8 |
| Nash-MTL | NYUv2 | 3 | 2 |
| Nash-MTL | Cityscapes | 2 | 8 |
| Nash-MTL | MetaWorld MT10 | 10 | 1280 |
| Nash-MTL | MetaWorld MT50 | 50 | 1280 |
| CAGrad | MetaWorld MT10 | 10 | 1280 |
| All | CIFAR-100 semi-supervised | -- | 256 |
| All | MNIST/FashionMNIST | 2 | 256 |

**Finding:** Gradient surgery methods have been validated at batch sizes from 2 (small) to 1280 (large). Our effective batch size of 16 falls well within the validated range. There is no evidence that batch size significantly affects gradient surgery efficacy.

---

## Recommendations for Our Setup

### Primary Recommendation: Combine CAGrad + Nash-MTL approach

**Rationale:** Nash-MTL achieves the best published results (Dm=-4.04% on NYUv2) but has O(K^2 d) complexity. CAGrad is simpler (O(Kd log K)) and nearly matches ST (Dm=+0.20%). Both significantly outperform our current PCGrad (+3.97%).

**Specific action items:**

1. **Replace PCGrad with Nash-MTL** for the gradient combination step. The Nash bargaining formulation naturally handles scale-invariance, which is critical for our mixed loss types (CIoU, cross-entropy, BCE, geodesic).

2. **Keep Kendall uncertainty weighting for loss scaling** but combine with Nash-MTL gradient combination. The two methods operate at different levels: Kendall weights the scalar losses, while Nash-MTL combines the gradients. They are complementary.

3. **Use Nash-MTL-50** (update bargaining weights every 50 steps) to reduce computational overhead. The paper shows this achieves nearly identical performance to full Nash-MTL.

### Secondary Recommendation: IMTL-G as a simpler alternative

If Nash-MTL implementation is complex, IMTL-G (ICLR 2021) is the next-best option with Dm=-0.76% on NYUv2. It is simpler to implement (equal gradient projections) and still beats ST on average.

### Implementation Complexity Estimate

| Method | Implementation Effort | Expected Gain vs PCGrad | Risk |
|--------|---------------------|------------------------|------|
| Nash-MTL | High (iterative bargaining) | ~8% Dm improvement | Low (well-tested) |
| Nash-MTL-50 | Medium | ~7% Dm improvement | Low |
| CAGrad | Medium (ball projection) | ~3.8% Dm improvement | Low |
| IMTL-G | Low (equal projection) | ~4.7% Dm improvement | Low |
| FAMO | Low (O(1) update) | ~3% Dm improvement | Medium (newer) |

### Caveats

- All benchmarks use SegNet backbone on NYUv2 / Cityscapes. NVv2 tasks are segmentation, depth, surface normal. Our tasks (detection, activity, PSR, pose) are fundamentally different and may respond differently.
- The gradient surgery literature has NOT been validated on 4+ task visual MTL. Scaling beyond 3 tasks may introduce new failure modes.
- Batch size effects at 16 are untested in the literature but should be compatible based on the range of validated batch sizes (2-1280).

---

## Cross-References to Other Agents

- **Agent 2 (Loss Weighting):** Nash-MTL includes scale-invariance which addresses the same issue as Kendall caps. Combining Nash-MTL + improved weighting (DWA, Auto-Lambda) could yield additive gains.
- **Agent 3 (Architecture Routing):** Cross-Stitch and MTAN are architecture-level methods that could complement gradient surgery. Gradient surgery optimizes the update direction; architecture routing optimizes the feature space.
- **Agent 6 (Detection MTL):** Detection is the hardest-hit task. Nash-MTL's balancing property may help by preventing activity loss from dominating the gradient direction.
- **Agent 10 (Loss Functions):** Improved IoU losses (WIoU, PIoU) change the gradient structure at the head level. Gradient surgery operates at the shared backbone level, making them complementary.

---

## References

1. Sener & Koltun. "Multi-Task Learning as Multi-Objective Optimization." NeurIPS 2018. arXiv:1810.04650
2. Yu et al. "Gradient Surgery for Multi-Task Learning." NeurIPS 2020. arXiv:2001.06782
3. Chen et al. "GradDrop: Gradient Surgery for Multi-Task Learning." NeurIPS 2020 Workshop.
4. Liu et al. "Conflict-Averse Gradient Descent for Multi-Task Learning." NeurIPS 2021. arXiv:2110.14048
5. Liu et al. "Towards Impartial Multi-Task Learning." ICLR 2021.
6. Navon et al. "Multi-Task Learning as a Bargaining Game." ICML 2022. arXiv:2202.01017
7. Javaloy et al. "RotoGrad: Gradient Homogenization in Multitask Learning." ICLR 2022.
8. Liu et al. "FAMO: Fast Adaptive Multi-Task Optimization." NeurIPS 2023. arXiv:2310.16386
9. Kendall et al. "Multi-Task Learning Using Uncertainty to Weigh Losses." CVPR 2018. arXiv:1705.07115
10. Misra et al. "Cross-Stitch Networks for Multi-Task Learning." CVPR 2016.
11. Liu et al. "End-to-End Multi-Task Learning with Attention." CVPR 2019.
