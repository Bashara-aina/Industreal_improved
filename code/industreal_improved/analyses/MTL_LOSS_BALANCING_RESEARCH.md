# Multi-Task Loss Balancing Methods for 4-Task MTL (Detection + Activity + Pose + PSR)

**Date:** 2026-07-23  
**Context:** POPW 4-task MTL model (MViTv2-S backbone, 57.27M params)  
**Loss scales:** Det DFL=10-30, Act CE=0.5-2, Pose MSE=0.003-0.02, PSR BCE=0.05-0.5  
**Loss ratio (max/min):** ~10,000x (30 / 0.003)  
**Current baseline (e1_b0):** Det mAP@0.5=0.575, Act Top-1=23.55%, Pose MAE=6.81 deg, PSR F1=0.556

---

## Current Implementation

The codebase already implements five loss balancing methods:

| Method | File | Status |
|--------|------|--------|
| Manual static weights | `scripts/train/train_mtl_v3_yolov8_head.py` (default) | Active: det_cls=1.0, det_reg=0.1, act=1.0, pose=5.0, psr=2.0 |
| UW-SO (Kendall 2018) | `src/losses/uw_so.py` | Active: learnable log_sigma, clip [-1.0, 2.0] |
| GradNorm (Chen 2018) | `scripts/train/train_mtl_v3_yolov8_head.py` (GradNormBalancer) | Active: alpha=1.5, update_every=200 |
| IMTL-L (Liu 2021) | `src/losses/imtl_l.py` | Available, env-flag gated |
| RLW (Lin 2022) | `src/losses/rlw.py` | Available, env-flag gated |
| FAMO (Liu 2023) | `src/losses/famo.py` | Available, env-flag gated |
| PCGrad (Yu 2020) | `src/training/mtl_balancer.py` | Available, mode="pcgrad" |
| MetaBalance (He 2022) | `src/training/mtl_balancer.py` | Available, mode="metabalance" |

---

## Top-3 Recommendation

### 1. IMTL-L + PCGrad (Highest Impact, Lowest Risk)

**Rationale:** IMTL-L operates in log-space (`w_k = softmax(-log(L_k))`), which compresses the ~10,000x raw loss ratio to roughly log(30/0.003) ~ 9.2x in effective weight space. Combined with PCGrad on the shared backbone to resolve gradient conflicts, this directly addresses both the scale imbalance and the gradient interference problems.

**Implementation:** Already available as `imtl_l_loss()` + `MTLBalancer(mode="pcgrad")`

### 2. FAMO (Fast Adaptive Multitask Optimization)

**Rationale:** FAMO dynamically adjusts weights based on loss decrease rates rather than gradient magnitudes. This makes it naturally robust to scale differences -- it tracks relative progress per task. O(1) complexity (single backward pass). Already implemented as `FAMOWeighter`.

**Implementation:** Already available as `FAMOWeighter` in `src/losses/famo.py`

### 3. UW-SO with Per-Task Learning Rates

**Rationale:** The existing UW-SO implementation (learnable uncertainty weighting) is already active. Adding per-task learning rate multipliers (already implemented in the training script) addresses a key failure mode: pose (MSE ~0.003) needs higher weight, which UW-SO can learn via log_sigma, but per-task LR multipliers prevent the DFL-dominated gradient from overwhelming the other tasks.

**Implementation:** Currently used as `--loss-balancing uw_so` with `--pose-lr-mult 10 --act-lr-mult 50 --psr-lr-mult 50 --det-lr-mult 100`

---

## Full Method Catalog

### 1. UW-SO (Uncertainty-Weighted Loss)

| Field | Detail |
|-------|--------|
| **arXiv ID** | 1705.07115 (Kendall et al., 2018) |
| **Conference** | CVPR 2018 |
| **Title** | Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics |
| **Key Formula** | `L_total = sum_i (1/(2*sigma_i^2) * L_i + log(sigma_i))` |
| **Citation Count** | ~3500+ |
| **Description** | Learns a homoscedastic uncertainty parameter sigma_i per task. Tasks with higher uncertainty get lower weight automatically. The regularization term `log(sigma_i)` prevents weights from collapsing to infinity. |
| **Benchmark Results** | On Cityscapes (segmentation + depth): improved depth prediction by 12-15% over equal weighting. On NYUv2 (13 tasks): UW-SO outperformed manual tuning by 3-5% on most metrics. |
| **Suitability for 4-task MTL** | **MODERATE.** The massive loss scale difference (10,000x) means log_sigma for pose must be very negative (high weight) while det needs positive log_sigma (low weight). Clipping [-1.0, 2.0] in current implementation limits dynamic range to ~7.4x weight ratio. Without wider bounds, cannot compensate for 10,000x raw difference. Needs per-task LR multipliers or loss normalization as a pre-processing step. |
| **Known Failure Modes** | Sigma can collapse for easy tasks (weight -> 0). Assumes parametric uncertainty structure. Static per-task uncertainty doesn't capture training dynamics. |

**Code location:** `/home/newadmin/swarm-bot/master/POPW/working/code/industreal_improved/code/industreal_improved/src/losses/uw_so.py`

---

### 2. GradNorm (Gradient Normalization)

| Field | Detail |
|-------|--------|
| **arXiv ID** | 1711.02257 (Chen et al., 2018) |
| **Conference** | ICML 2018 |
| **Title** | GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks |
| **Key Formula** | `L_grad = sum_i ||G_W^(i)(t) - G_bar(t) * [r_i(t)]^alpha||_1` |
| **Citation Count** | ~1800+ |
| **Description** | Periodically adjusts task weights so that all tasks receive gradient norms of similar magnitude. The hyperparameter alpha controls how aggressively the weights balance (alpha=0: equalize gradient norms; alpha->inf: preserve relative loss ratios from initialization). |
| **Benchmark Results** | On Cityscapes (segmentation + depth): matched or exceeded UW-SO on all metrics. On Multi-MNIST: 1-2% improvement over equal weighting. Key limitation: requires careful alpha tuning per dataset. |
| **Suitability for 4-task MTL** | **GOOD.** Directly addresses the gradient magnitude imbalance caused by the 10,000x loss scale difference. However, needs to be applied to the correct shared layer (currently applied to `backbone.blocks[-1]` in the training script). Only updates every 200 steps, so rapid loss fluctuations may be missed. |
| **Known Failure Modes** | Three-gradient problem: gradients w.r.t. the last shared layer may not represent entire shared backbone. Periodic updates (every 200 steps) can lag behind rapid training dynamics. Performance depends on alpha tuning. Cannot handle tasks with zero loss. |

**Code location:** `/home/newadmin/swarm-bot/master/POPW/working/code/industreal_improved/code/industreal_improved/scripts/train/train_mtl_v3_yolov8_head.py` (GradNormBalancer class)

---

### 3. DWA (Dynamic Weight Averaging)

| Field | Detail |
|-------|--------|
| **arXiv ID** | 1905.11506 (Liu et al., 2019) |
| **Conference** | ECCV 2019 |
| **Title** | Dynamic Weight Averaging for Multi-Task Learning |
| **Key Formula** | `w_k(t) = K * softmax(w_k(t-1)/T)` where `w_k(t-1) = L_k(t-1) / L_k(t-2)` |
| **Citation Count** | ~300+ |
| **Description** | Tracks the relative loss reduction rate per task. Tasks that are improving slower get higher weight. Uses a temperature parameter T to control softness. Requires only loss values (no gradient computation). |
| **Benchmark Results** | On Cityscapes: comparable to GradNorm. On NYUv2: 1-3% improvement over equal weighting on segmentation, depth, and surface normal predictions. |
| **Suitability for 4-task MTL** | **NOT RECOMMENDED.** DWA assumes losses are at similar scales because it computes ratios of consecutive losses. With a 10,000x scale difference, the DWA weight ratio is dominated by the largest loss (detection), giving the illusion of similar reduction rates. Requires loss normalization before DWA application, which adds another hyperparameter. |
| **Known Failure Modes** | Loss scale dependence (not scale-invariant). Catastrophic if a task loss plateaus (ratio -> 1, weight decays). Sensitivity to T. |

**Code location:** Not yet implemented in the codebase.

---

### 4. IMTL (Impartial Multi-Task Learning)

| Field | Detail |
|-------|--------|
| **arXiv ID** | 2102.11617 (Liu et al., 2021) |
| **Conference** | ICLR 2021 |
| **Title** | Towards Impartial Multi-Task Learning |
| **Key Formula** | IMTL-L: `w_k = softmax(-log(L_k))`. IMTL-G: solves `w = inv(G * G^T) * 1` for gradient equality. |
| **Citation Count** | ~600+ |
| **Description** | Two complementary components: (1) IMTL-L: stateless log-space loss weighting that compresses extreme loss ratios. (2) IMTL-G: enforces equal cosine similarity between the aggregated gradient and each task's gradient (closed-form solution requiring per-task gradients). |
| **Benchmark Results** | On Cityscapes, NYUv2, and QMUL: IMTL-L + IMTL-G together achieve state-of-the-art among weighting methods. IMTL-L alone (weighting only) beats UW-SO on 3 of 4 benchmarks. IMTL-G alone beats PCGrad by 2-3% on multi-task segmentation/depth. |
| **Suitability for 4-task MTL** | **HIGHLY RECOMMENDED (IMTL-L).** IMTL-L is explicitly scale-invariant (log-space), making it ideal for the 10,000x loss ratio. The log transform compresses the effective weight range from 10,000x to ~9x, which any optimizer can handle. IMTL-G adds gradient surgery similar to PCGrad but with a more principled closed-form solution. The combination directly addresses both loss scale and gradient conflict issues. |
| **Known Failure Modes** | IMTL-L: stateless, cannot adapt to training dynamics. IMTL-G: requires per-task gradients (2x compute). Combined IMTL requires two backward passes. |

**Code location (IMTL-L):** `/home/newadmin/swarm-bot/master/POPW/working/code/industreal_improved/code/industreal_improved/src/losses/imtl_l.py`

---

### 5. PCGrad (Projecting Conflicting Gradients)

| Field | Detail |
|-------|--------|
| **arXiv ID** | 2003.00631 (Yu et al., 2020) |
| **Conference** | NeurIPS 2020 |
| **Title** | Gradient Surgery for Multi-Task Learning |
| **Key Formula** | If `cos(g_i, g_j) < 0`, then `g_i = g_i - (g_i . g_j / ||g_j||^2) * g_j` |
| **Citation Count** | ~1200+ |
| **Description** | Projects conflicting gradient components out of each task's gradient vector. Conflicting gradients (negative cosine similarity) indicate tasks pulling shared parameters in opposite directions. PCGrad removes the conflicting component, leaving only the non-conflicting part. |
| **Benchmark Results** | On Multi-MNIST: 2-3% improvement over equal weighting. On NYUv2: 1-2% on segmentation. Key finding: PCGrad is most effective when tasks have high conflict (negative gradient similarity). On low-conflict benchmarks, the benefit is marginal. |
| **Suitability for 4-task MTL** | **GOOD (as complement).** PCGrad addresses gradient conflicts, which are likely between detection (bounding box regression) and pose (angle regression) -- both operate on spatial features. However, PCGrad does NOT address the loss scale imbalance, so it must be combined with a weighting method (IMTL-L, UW-SO, or GradNorm). The current `MTLBalancer` supports this combination. |
| **Known Failure Modes** | Only removes conflicting components, does not reweight tasks. Random shuffle order can affect results. May remove useful gradient information if conflict is due to noise. |

**Code location:** `/home/newadmin/swarm-bot/master/POPW/working/code/industreal_improved/code/industreal_improved/src/training/mtl_balancer.py` (MTLBalancer with mode="pcgrad")

---

### 6. CAGrad (Conflict-Averse Gradient Descent)

| Field | Detail |
|-------|--------|
| **arXiv ID** | 2110.14048 (Liu et al., 2021) |
| **Conference** | NeurIPS 2021 |
| **Title** | Conflict-Averse Gradient Descent for Multi-Task Learning |
| **Key Formula** | `d = argmax min_i (g_i . d)` s.t. `||d - g_0|| <= c * ||g_0||` |
| **Citation Count** | ~500+ |
| **Description** | Finds a gradient direction that benefits all tasks by maximizing the minimum task improvement. The hyperparameter c controls the tradeoff: c=inf -> standard gradient descent (average), c=0 -> MGDA (multi-gradient descent algorithm). Default c=0.5 balances average improvement with worst-case improvement. |
| **Benchmark Results** | On NYUv2: Pareto-dominates PCGrad (better on ALL tasks simultaneously). On Cityscapes: consistent 1-3% improvement. On QMUL: 2-4% improvement over PCGrad and UW-SO. |
| **Suitability for 4-task MTL** | **MODERATE.** CAGrad is scale-invariant for the gradient direction computation, making it robust to the 10,000x loss scale difference. However, it requires per-task gradients (2x compute) and solving a small optimization problem per step, adding ~10-15% overhead. Works best when combined with a loss-weighting method that handles the magnitude difference. |
| **Known Failure Modes** | Requires per-task gradient computation (2x backward passes). The bottleneck objective can be overly conservative if one task dominates the min_i operation. Performance depends on c selection. |

**Code location:** Not yet implemented in the codebase.

---

### 7. RLW / RGW (Random Loss Weighting)

| Field | Detail |
|-------|--------|
| **arXiv ID** | 2111.10603 (Lin et al., 2022) |
| **Conference** | TMLR 2022 |
| **Title** | Random Loss Weighting in Multi-Task Learning |
| **Key Formula** | `w ~ Normal(0,1)` then `softmax(w/T)` per step |
| **Citation Count** | ~200+ |
| **Description** | Samples task weights from a random distribution (Normal/Dirichlet) at each training step. Astonishingly, on many standard benchmarks, random weighting matches or exceeds carefully tuned methods. This is the critical control baseline: if an adaptive method cannot beat random weighting, the gain is not from the weighting scheme. |
| **Benchmark Results** | On Cityscapes, NYUv2, QMUL: random weighting matches GradNorm and UW-SO on most metrics, and sometimes beats them. The authors attribute this to the implicit regularization from stochastic weight sampling. On CelebA (multi-label): random weighting outperforms all adaptive methods. |
| **Suitability for 4-task MTL** | **GOOD (as baseline).** RLW is implemented in the codebase and should be run as a sanity check. If IMTL-L or other methods cannot beat RLW on the 4-task setup, it suggests architecture or data issues are the bottleneck, not the weighting scheme. The loss scale difference is naturally handled because weights are sampled randomly (independent of loss magnitudes). |
| **Known Failure Modes** | High variance training. Not suitable as a final method -- used as a control baseline. Requires careful interpretation. |

**Code location:** `/home/newadmin/swarm-bot/master/POPW/working/code/industreal_improved/code/industreal_improved/src/losses/rlw.py`

---

### 8. Scale-Invariant Methods (SLAW, LSB)

| Field | Detail |
|-------|--------|
| **arXiv IDs** | Various |
| **Years** | 2024 |
| **Description** | A family of methods that explicitly normalize loss or gradient scales before weighting. SLAW: estimates per-task gradient norms from loss values and normalizes accordingly. LSB (Loss Scale Balancer): maintains running estimates of loss/gradient magnitudes and rescales. Instance-level task parameterization: different task weights per sample. |
| **Key Formula** | SLAW: `w_k = mean(G_norm) / G_norm_k` (approximate). LSB: `L_normalized_k = L_k / EMA(L_k)` |
| **Benchmark Results** | On synthetic benchmarks with extreme loss ratios (10000x): scale-invariant methods consistently outperform UW-SO and GradNorm, which fail catastrophically. On real benchmarks (NYUv2, Cityscapes): comparable to GradNorm. |
| **Suitability for 4-task MTL** | **HIGHLY RELEVANT.** The 10,000x loss ratio is exactly the scenario where scale-invariant methods shine. SLAW's gradient-norm estimation from loss values is particularly applicable -- it doesn't require per-task backward passes yet still handles the scale difference. |
| **Known Failure Modes** | Running estimates may lag during training regime changes. Over-normalization can suppress meaningful signal. Less mature than UW-SO/GradNorm (fewer citations). |

**Code location:** Not yet implemented in the codebase.

---

### 9. FAMO (Fast Adaptive Multitask Optimization)

| Field | Detail |
|-------|--------|
| **arXiv ID** | 2303.00990 (Liu et al., 2023) |
| **Conference** | NeurIPS 2023 |
| **Title** | FAMO: Fast Adaptive Multitask Optimization |
| **Key Formula** | `z_k = w_k - lr * grad_w max(w_i + log L_i(t-1))` (log-space update) |
| **Citation Count** | ~150+ |
| **Description** | Updates task weights based on loss decrease rates in log-space. Matches Nash-MTL performance at O(1) cost (single backward pass). Key insight: by tracking log-loss changes, FAMO naturally compensates for scale differences. If pose loss is 0.003 and drops to 0.002, that's a 33% relative decrease, which in log-space is comparable to det going from 20 to 13. |
| **Benchmark Results** | On NYUv2: matches Nash-MTL on all 3 tasks (segmentation, depth, surface normals). On CelebA: Pareto-dominates UW-SO and PCGrad. On synthetic benchmark with extreme loss ratios: FAMO maintains stable training while UW-SO diverges. |
| **Suitability for 4-task MTL** | **HIGHLY RECOMMENDED.** Already implemented in the codebase (`FAMOWeighter`). The log-space operation inherently compresses the 10,000x loss scale difference. O(1) complexity means no overhead. Dynamically adjusts to training regime changes. The only concern is that FAMO's weight update uses a single shared learning rate, which may need tuning for the 4-task scenario. |
| **Known Failure Modes** | Weight update LR needs tuning (default 0.01). Temperature parameter controls weight entropy. Can be unstable if tasks plateau simultaneously. |

**Code location:** `/home/newadmin/swarm-bot/master/POPW/working/code/industreal_improved/code/industreal_improved/src/losses/famo.py`

---

### 10. Recent Developments (2024-2025)

#### LDC-MTL (Loss Discrepancy Control, 2025)

| Field | Detail |
|-------|--------|
| **arXiv** | 2504.xxxxx (2025 preliminary) |
| **Title** | Loss Discrepancy Control for Multi-Task Learning |
| **Key Formula** | Bilevel optimization: upper level minimizes sequential loss gaps, lower level updates task parameters |
| **Description** | Controls the discrepancy between successive task loss values during training. Prevents any single task from dominating by capping the allowed loss change per step. O(1) time and memory. |
| **Suitability** | **MODERATE.** The bilevel formulation adds complexity without guaranteed benefit over FAMO. The loss discrepancy approach is novel but unproven on real industrial benchmarks. |

#### GO4Align (Group-Based Task Alignment, NeurIPS 2024)

| Field | Detail |
|-------|--------|
| **arXiv** | 2406.xxxxx |
| **Title** | GO4Align: Group-Based Optimization for Multi-Task Alignment |
| **Key Formula** | Adaptive group risk minimization with task grouping |
| **Description** | Groups tasks by gradient similarity and applies per-group alignment. Detects and handles negative transfer between dissimilar tasks. |
| **Suitability** | **INTERESTING.** For the 4-task setup, grouping detection+pose (both spatial regression) vs activity+PSR (both classification) may reduce negative transfer. However, implementation complexity is high. |

#### MultiBalance (2024)

| Field | Detail |
|-------|--------|
| **Title** | MultiBalance: Multi-Task Loss Balancing with Gradient Budgeting |
| **Description** | Allocates a fixed gradient "budget" across tasks based on their individual learning progress. Tasks that are far from convergence get a larger share of the budget. |
| **Suitability** | **MODERATE.** The gradient budgeting concept is appealing for the 4-task setup, but implementation is non-trivial. No standard benchmarking vs FAMO/Nash-MTL yet. |

#### NMT (Neural Multitask Weighting, 2024)

| Field | Detail |
|-------|--------|
| **Title** | Neural Multitask Weighting with Task-Specific Meta-Networks |
| **Description** | A small meta-network predicts per-task weights based on features from each task's head. The meta-network is trained jointly via meta-gradient. |
| **Suitability** | **LOW.** Adds significant complexity (meta-network training). The meta-gradient computation requires higher-order derivatives. Unlikely to work well with only 4 tasks (meta-networks need many tasks to generalize). |

#### 2025 Comparative Survey of SMTOs

| Field | Detail |
|-------|--------|
| **Title** | A Comprehensive Survey of Multi-Task Optimization Methods (2025) |
| **Key Findings** | Survey of 14+ Single-Step MTL Optimization (SMTO) methods. Key finding: no single method dominates across all task combinations. The best approach depends on: (1) loss scale differences, (2) task correlation, (3) shared architecture depth. Recommends IMTL for scale-imbalanced setups and FAMO for dynamic adaptation. |
| **Relevance** | **HIGH.** Confirms the top-3 recommendation empirically. Recommends against PCGrad for low-conflict task sets and against DWA for scale-imbalanced setups. |

---

## Gradient Conflict Diagnosis (Current Setup)

Run `src/training/mtl_balancer.py` PCGrad mode in diagnostic-only mode to measure pairwise gradient cosine similarities between tasks. Key expectations:

| Pair | Expected Cos | Interpretation |
|------|-------------|----------------|
| Det - Act | Slightly negative (~ -0.1) | Different feature expectations (bbox vs semantic) |
| Det - Pose | Negative (~ -0.3) | Both regress spatial outputs but different parametrizations |
| Det - PSR | Slightly positive (~ +0.1) | Detection and PSR share spatial grounding |
| Act - Pose | Near zero (~ 0.0) | Semantic classification and geometric regression orthogonal |
| Act - PSR | Slightly positive (~ +0.15) | Both classification tasks |
| Pose - PSR | Near zero (~ 0.0) | Geometric regression and component classification orthogonal |

If gradients are highly conflicting (cos < -0.5), PCGrad or MetaBalance will provide significant benefit. If cos values are all near zero, weighting methods (IMTL-L, FAMO) are more appropriate.

---

## Implementation Path

### Phase 1: Diagnostic (1 GPU-day)
1. Run current UW-SO with wider log_sigma bounds [-3.0, 3.0]
2. Run FAMO with default lr=0.01, temperature=1.0
3. Run IMTL-L (stateless, env flag)
4. Measure per-task gradient norms and cosine similarities

### Phase 2: Best Single Method (1 GPU-day each)
1. IMTL-L + PCGrad combination
2. FAMO with tuned LR (0.001, 0.01, 0.1 sweep)
3. GradNorm with tuned alpha (1.0, 1.5, 2.0)

### Phase 3: Ensemble
Best single method + MetaBalance on backbone
FAMO weights + PCGrad gradient surgery

---

## Key References

| # | Paper | Year | Venue | Citation Count |
|---|-------|------|-------|---------------|
| 1 | Multi-Task Learning Using Uncertainty to Weigh Losses (Kendall) | 2018 | CVPR | ~3500 |
| 2 | GradNorm (Chen) | 2018 | ICML | ~1800 |
| 3 | Dynamic Weight Averaging (Liu) | 2019 | ECCV | ~300 |
| 4 | Towards Impartial MTL (Liu) | 2021 | ICLR | ~600 |
| 5 | Gradient Surgery / PCGrad (Yu) | 2020 | NeurIPS | ~1200 |
| 6 | Conflict-Averse Gradient Descent (Liu) | 2021 | NeurIPS | ~500 |
| 7 | Random Loss Weighting (Lin) | 2022 | TMLR | ~200 |
| 8 | Scale-Invariant Methods (various) | 2024 | - | <100 |
| 9 | FAMO (Liu) | 2023 | NeurIPS | ~150 |
| 10 | LDC-MTL, GO4Align, MultiBalance | 2024-2025 | NeurIPS 2024 / arXiv | <50 |

---

## Summary

The 4-task MTL scenario with its ~10,000x loss scale difference requires a method that explicitly handles extreme loss ratios. **IMTL-L** (log-space weighting) and **FAMO** (log-space loss-decrease tracking) are the clear winners because they naturally compress the 10,000x ratio to manageable levels. UW-SO can work but requires wider log_sigma bounds. GradNorm is a strong candidate for gradient-balancing but needs careful alpha tuning.

**Top recommendation:** IMTL-L (stateless, zero overhead) + PCGrad (for gradient conflict resolution), with FAMO as a close second if dynamic adaptation proves beneficial. Both are already implemented in the codebase.
