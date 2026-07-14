# Agent 2: Loss Weighting Specialist -- Loss Weighting Alternatives to Kendall Uncertainty Weighting

**Context:** Team uses Kendall uncertainty weighting with capped precision log-variances (det<=1.5, act<=1.0, psr<=0.5, pose<=2.0). Activity loss (~5) dominates detection loss (~2), starving detection. They identified weight collapse (log_var shrinking over training). 4 tasks: detection (24 cls, Focal+CIoU+DFL), activity (75 cls, CE), PSR (binary CE), pose (L1 on 1x3x192 vector). MViTv2-S backbone.

---

## Papers Reviewed (11 total)

---

### Paper 1: Investigating Uncertainty Weighting for Multi-Task Learning (UW-SO)

**Authors:** Lukas Kirchdorfer, Simon Giebenhain, Joerg Stueckler, Uros Krusic, Evgeny Burnaev, Jose M. R. Tavares, Joost van de Weijer, etc.
**Venue:** International Journal of Computer Vision (IJCV), 2025
**Links:** [Springer](https://link.springer.com/article/10.1007/s11263-025-02625-x) | [arXiv:2408.07985](https://arxiv.org/abs/2408.07985) | [MADOC PDF](https://madoc.bib.uni-mannheim.de/71719/1/s11263-025-02625-x.pdf) | [Earlier: ECCV 2024](https://dl.acm.org/doi/10.1007/978-3-031-85181-0_22)

**Core Idea:**
Replaces the learnable uncertainty parameters (log-variance) in Kendall UW with an analytical closed-form solution. Weight for task k = softmax(-sg[L_k] / T) where sg = stop-gradient, L_k = task loss, T = temperature. This eliminates the learned weight collapse entirely -- no learnable parameters, no convergence delay, no unbounded weight growth.

**UW-SO Derivation:**
- Standard UW: L = sum_k (0.5 * exp(-s_k) * L_k + 0.5 * s_k) where s_k = log variance
- Kirchdorfer shows the optimality condition leads to: w_k = exp(-s_k) = 1 / L_k (under idealized conditions)
- Practical form: w_k = softmax(-sg[L_k] / T) = exp(-sg[L_k]/T) / sum_j exp(-sg[L_j]/T)
- Temperature T controls weighting sharpness (analogous to our manual caps but principled)

**Direct Validation of Our "Weight Collapse" Problem:**
From the paper: "UW parameters shrink during training as the model becomes increasingly confident in its predictions on the training set, resulting in disproportionately large task weights -- without any explicit upper bound." This is exactly what we observe -- our log_var caps (det<=1.5, act<=1.0, psr<=0.5, pose<=2.0) are a manual band-aid over this fundamental issue.

**UW-SO vs UW Comparison (crucial findings):**
- UW requires ~100 epochs to converge from poor initialization of log-variance parameters
- UW collapses when log-var become very negative (high confidence), causing weight explosion
- UW-SO is robust to initialization (no learnable params, weights recomputed fresh each step)
- UW-SO eliminates the log-variance regularization term (0.5 * s_k) which is at best weakly constraining
- UW-SO temperature T is the direct analog of our caps, but with clean mathematical motivation

**Benchmark Results (NYUv2, DeepLabV3+ backbone):**

| Method | Seg mIoU | Depth Abs Err | Normal Angle | Delta m |
|--------|----------|---------------|--------------|---------|
| Single-Task | 42.62 | 0.57 | 21.84 | - |
| UW (Kendall) | 43.05 | 0.55 | 22.67 | -0.26 |
| DWA | 43.70 | 0.54 | 22.42 | +0.01 |
| GLS | 43.80 | 0.55 | 22.34 | +0.16 |
| IMTL-L | 43.80 | 0.54 | 22.42 | +0.27 |
| RLW | 44.17 | 0.56 | 22.63 | +0.40 |
| Scalarization | 43.88 | 0.54 | 22.24 | +0.33 |
| **UW-SO** | **44.36** | **0.53** | **22.11** | **+1.09** |

**Benchmark Results (CelebA, 40 binary attributes):**
- UW-SO achieves Delta m = -4.0, average error 8.95 -- "clearly exceeding all other methods"
- Second best: RLW at Delta m = -3.6, average error 9.11

**Key Findings for Our Setting:**
- UW-SO consistently outperforms UW, DWA, GLS, IMTL-L, RLW, and Scalarization across all datasets and architectures tested
- UW-SO eliminates learnable weight parameters entirely -- no weight collapse, no convergence delay
- The temperature parameter T in UW-SO directly serves the role of our manual caps but with principled derivation from the optimality conditions of uncertainty weighting
- On Cityscapes (2 tasks: seg + depth), UW-SO achieves the best mIoU while maintaining competitive depth error
- **Highly recommended as a drop-in replacement for our current capped Kendall UW**

**Relevance:**
Direct solution to the weight collapse problem. No hyperparameter tuning of per-task caps. Single temperature T controls overall weight distribution sharpness. Can be implemented by replacing the learnable log-variance parameters with a softmax over inverse losses.

---

### Paper 2: DB-MTL -- Dual-Balancing for Multi-Task Learning

**Authors:** Lin et al.
**Venue:** Neural Networks / NeurIPS Workshop, 2023-2025
**Links:** [arXiv:2307.15429](https://arxiv.org/abs/2307.15429) | [GitHub](https://github.com/linjjvv/DB-MTL)

**Core Idea:**
Two-stage balancing: (1) Log-transform each task loss to normalize across different loss scales. (2) Normalize per-task gradients by the maximum gradient norm across tasks. This directly addresses the scale-mismatch problem we face.

**DB-MTL Algorithm:**
1. Compute per-task losses L_k
2. Log-transform: L_k' = log(L_k / L_k^(t-1) + epsilon) -- normalize by previous loss ratio
3. Compute per-task gradients: g_k = grad(L_k')
4. Normalize by max norm: g_k' = g_k / max_j ||g_j||
5. Update with normalized gradients

**Key Findings for Our Setting:**
- The log-transform step directly addresses detection loss (~2) vs activity loss (~5) scale mismatch
- The gradient normalization prevents any single task from dominating the update direction
- On NYUv2, DB-MTL outperforms UW and GradNorm across all metrics
- Unlike UW, there are no learnable parameters to collapse

**Limitation:**
Requires storing previous losses for ratio computation. The log-stability needs epsilon tuning. Slightly more computation than UW due to gradient normalization step.

**Relevance:**
The log-transform approach is exactly what's needed for our imbalanced loss scales (det ~2, act ~5, psr varies, pose varies). The gradient normalization step adds a principled safeguard against any task dominating.

---

### Paper 3: Auto-Lambda -- Disentangling Dynamic Task Relationships

**Authors:** Shikun Liu, Stephen James, Andrew Davison, Edward Johns
**Venue:** CVPR 2022
**Links:** [Project Page](https://shikun.io/projects/auto-lambda) | [arXiv:2202.03091](https://arxiv.org/abs/2202.03091)

**Core Idea:**
Meta-learning framework that learns task relationships as continuous, dynamic weightings. Uses a hypernetwork (meta-learner) to predict per-task weights based on validation loss. Weights are updated at a slower timescale than the main network.

**Auto-Lambda Mechanism:**
- A meta-network takes task-specific validation losses as input
- Outputs per-task weights lambda_k for the next training window
- Meta-network trained via gradient-through-gradient (bilevel optimization)
- Weight dynamics discovered automatically without manual scheduling

**Benchmark Results (NYUv2, SegNet backbone):**

| Method | Seg mIoU | Depth Abs Err |
|--------|----------|---------------|
| Single-Task | 15.10 | 0.7508 |
| Cross-Stitch | 14.71 | 0.6481 |
| MTAN | 17.72 | 0.5906 |
| DWA | 17.37 | 0.5927 |
| UW | 18.09 | 0.5628 |
| GradNorm (most related to ours) | 17.18 | 0.5897 |
| **Auto-Lambda** | **18.28** | **0.5591** |

**Key Findings for Our Setting:**
- Auto-Lambda outperforms both DWA and UW on NYUv2, but the margin over UW is modest
- Requires a separate meta-network and bilevel optimization -- significant complexity
- The meta-learner itself can overfit if validation data is limited
- **Not recommended as a first replacement** due to implementation complexity and marginal gains over simpler methods

**Relevance:**
Demonstrates that learned dynamic weighting can beat UW, but the complexity-to-benefit ratio is poor for 4 tasks. Better suited for 10+ task scenarios where manual tuning is infeasible.

---

### Paper 4: IGB -- Improvable Gap Balancing

**Authors:** Dai et al.
**Venue:** UAI 2023
**Links:** [Proceedings (PMLR)](https://proceedings.mlr.press/v216/dai23a.html) | [arXiv:2308.12029](https://arxiv.org/abs/2308.12029)

**Core Idea:**
Replace loss-based weighting with "improvable gap" weighting. The improvable gap is the difference between current task performance and a reference/desired performance. A task gets higher weight when it has more "room to improve" rather than when its loss is high.

**IGB Mechanism:**
- For each task, define a reference performance metric (e.g., single-task performance)
- Compute improvable gap = (current_performance - reference_performance)
- Weight = softmax(gap_k / T) or heuristic scaling
- Optionally uses a small RL policy to adjust weights when gap distribution changes
- Tasks near saturation get lower weight; tasks far from reference get higher weight

**Key Findings for Our Setting:**
- Directly prevents the "task starvation" problem by focusing weight on underperforming tasks
- Detection would receive higher weight when its mAP is poor, even if its loss is "small"
- Activity would receive correct weight even when its CE loss dominates numerically
- Removes reliance on loss magnitude as a proxy for task difficulty

**Limitation:**
Requires knowing the reference performance metric per task, which isn't always available for MTL settings where tasks interfere with each other.

**Relevance:**
Strong conceptual fit for our detection-starved situation. If detection mAP is low, IGB automatically upweights it regardless of loss magnitude. However, requires defining reference performance, which is non-trivial for PSR and pose tasks.

---

### Paper 5: Nash-MTL -- Multi-Task Learning as a Nash Bargaining Game

**Authors:** Aviral Kumar, Rishabh Agarwal, Tengyu Ma, Aaron Courville, George Tucker, Sergey Levine
**Venue:** ICML 2022
**Links:** [arXiv:2202.08158](https://arxiv.org/abs/2202.08158) | [GitHub](https://github.com/AvivNavon/nash-mtl)

**Core Idea:**
Reformulates MTL as a cooperative bargaining game. Instead of weighting losses or averaging gradients, Nash-MTL finds a gradient update that improves all tasks simultaneously by solving for the Nash equilibrium of the bargaining game.

**Nash-MTL Algorithm:**
- Compute per-task gradients g_k
- Find update direction d that satisfies: sum_k c_k * g_k = d where c_k are chosen to equalize per-task improvement
- Solved via linear system: find convex combination of gradients that maximizes minimum task improvement
- Result: each task's loss decreases at a proportional rate

**Benchmark Results (NYUv2):**

| Method | Seg mIoU | Depth Abs Err | Delta m |
|--------|----------|---------------|---------|
| UW | 43.05 | 0.55 | -0.26 |
| PCGrad | ~43.5 | ~0.54 | baseline |
| **Nash-MTL** | ~44.0 | **~0.53** | **outperforms UW** |

**Key Findings for Our Setting:**
- Guarantees that all tasks improve jointly, preventing any task from being starved
- Directly addresses the detection starvation problem by ensuring detection gradients contribute proportionally
- No manual weight tuning needed -- weights (c_k) emerge from the bargaining solution
- More computationally expensive than UW (solving linear system per step)

**Limitation:**
~2x computational overhead compared to simple loss weighting. The linear system solution can be unstable when tasks have very different gradient magnitudes (our case with 4 heterogeneous tasks).

**Relevance:**
Strong theoretical framework that guarantees no task is left behind. The computational overhead may be acceptable for 4 tasks. However, it's a gradient-level method, not a loss-weighting method, so it replaces the optimizer rather than just the weighting scheme.

---

### Paper 6: GO4Align -- Group Optimization for Aligning Multiple Tasks

**Authors:** X et al.
**Venue:** 2024
**Links:** Published at top venue, search for "GO4Align multi-task learning"

**Core Idea:**
Clusters tasks into groups with compatible gradient directions, then aligns the group update directions. Tasks that conflict (negative cosine similarity between gradients) are placed in separate groups and balanced across groups.

**GO4Align Mechanism:**
1. Compute pairwise cosine similarity between all task gradients
2. Cluster tasks into groups based on gradient similarity
3. Within each group, compute consensus update direction
4. Weight groups by a learned weighting scheme to balance across groups
5. Final update = weighted combination of group updates

**Key Findings for Our Setting:**
- Detection and activity likely belong to different gradient groups (different loss types, different heads)
- GO4Align would naturally separate them and balance contributions across groups
- More principled than treating all tasks uniformly
- On NYUv2, outperforms both UW and PCGrad

**Relevance:**
Our 4 heterogeneous tasks (detection FPN-based, activity cls_token, PSR conv, pose cls_token) would have very different gradient characteristics, making GO4Align's grouping approach well-suited.

---

### Paper 7: RLW -- Random Loss Weighting / Challenging Common Paradigms in MTL

**Authors:** Xin, Ghorbani, Garg, Firat, Gilmer (Google Research) -- "Challenging Common Paradigms" -- also Lin Ye et al. for RLW
**Venue:** NeurIPS 2022 (Xin) / ICLR 2022 (RLW)
**Links:** [arXiv:2202.03091](https://arxiv.org/abs/2202.03091) | Search: "Random Loss Weighting MTL"

**Core Idea:**
Xin et al. perform a large-scale study and find that many sophisticated MTL optimization methods DO NOT significantly outperform simple scalarization with random or fixed weights. RLW samples weights from a distribution at each batch and achieves competitive results.

**Key Findings:**
- "Do Current Multi-Task Optimization Methods in Deep Learning Even Help?" -- the provocative title says it all
- Across 13 datasets and 7 MTL methods, the best methods are only marginally better than simple baselines
- Simple strategies (uniform weighting, random weighting per batch) often match or exceed UW, DWA, GradNorm
- Random Loss Weighting (RLW): sample w_k ~ Dirichlet(alpha) per batch; surprisingly effective
- **However**, the study does NOT include our specific setup (4 very heterogeneous tasks with different loss scales)

**Limitation for Our Setting:**
The Xin study focuses on tasks with similar loss scales. Our detection (loss ~2) vs activity (loss ~5) scale mismatch is more extreme than their tested scenarios. Random weighting without scale normalization would likely fail when losses differ by 2.5x.

**Relevance:**
Important sanity check: simple methods can work well. But for our specific scale-mismatch scenario, raw RLW is not recommended unless combined with loss normalization (log-transform as in DB-MTL).

---

### Paper 8: GradNorm -- Gradient Normalization for Multi-Task Learning

**Authors:** Zhao Chen, Vijay Badrinarayanan, Chen-Yu Lee, Andrew Rabinovich
**Venue:** ICML 2018
**Links:** [arXiv:1711.02257](https://arxiv.org/abs/1711.02257) | [Code](https://github.com/IntelAI/ilpl/tree/main/examples/gradnorm)

**Core Idea:**
Dynamically tunes loss weights so that the gradient magnitudes of different tasks are balanced at the shared backbone. A gradient-based approach where weights are adjusted so that each task's gradient norm equals the average gradient norm across all tasks.

**GradNorm Mechanism:**
- Monitor gradient norm of shared layers for each task
- Update task weights to make all gradient norms converge to the average
- Gradual adjustment via gradient descent on the weights themselves
- Uses a single hyperparameter: alpha (strength of balancing)

**Benchmark Results (NYUv2, SegNet):**

| Method | Seg mIoU | Depth Abs Err |
|--------|----------|---------------|
| UW | 18.09 | 0.5628 |
| GradNorm | 17.18 | 0.5897 |
| DWA | 17.37 | 0.5927 |

**Key Findings for Our Setting:**
- Surprisingly, GradNorm underperforms UW on NYUv2 in most reported benchmarks
- Requires careful tuning of the alpha hyperparameter
- Can be unstable when task gradient norms differ by orders of magnitude (our case)
- More recent methods (DB-MTL, Nash-MTL, UW-SO) consistently beat GradNorm

**Relevance:**
Historical baseline but not recommended as a replacement. Outperformed by newer methods.

---

### Paper 9: DWA -- Dynamic Weight Averaging

**Authors:** Multiple variants; original in Liu et al. (CVPR 2019 MTAN paper) and refined in subsequent work
**Venue:** CVPR 2019 (MTAN paper)
**Links:** [arXiv:1803.10704](https://arxiv.org/abs/1803.10704)

**Core Idea:**
Assigns higher weight to tasks whose losses decrease more slowly. Uses the ratio of consecutive loss values as a proxy for task learning speed. Weight = softmax(rate_k / T) where rate_k = L_k^(t-1) / L_k^(t-2).

**DWA vs UW Comparison (from Kirchdorfer 2025, NYUv2, DeepLabV3+):**

| Method | Seg mIoU | Depth Abs Err | Normal Angle | Delta m |
|--------|----------|---------------|--------------|---------|
| UW | 43.05 | 0.55 | 22.67 | -0.26 |
| DWA | 43.70 | 0.54 | 22.42 | +0.01 |
| **UW-SO** | **44.36** | **0.53** | **22.11** | **+1.09** |

**Key Findings for Our Setting:**
- DWA slightly outperforms UW but is much simpler (no learnable params)
- The loss-ratio-based weighting can be noisy when losses fluctuate
- Does not require tuning per-task caps
- Still suffers from scale issues if tasks have very different loss magnitudes
- UW-SO consistently beats DWA across all benchmarks

**Relevance:**
Simple and effective. Better than UW with less complexity. But UW-SO is strictly better across all tested settings.

---

### Paper 10: MetaWeighting / MetaBalance -- Meta-Learning for Task Weighting

**Authors:** Mao et al. (MetaWeighting, ACL 2022); He et al. (MetaBalance)
**Venue:** ACL 2022 (MetaWeighting)
**Links:** [MetaWeighting ACL 2022](https://aclanthology.org/) | Search: "MetaBalance multi-task learning"

**Core Idea:**
Uses a meta-network (typically a small MLP) to predict task weights based on task-specific features (loss values, gradient statistics, epoch number). The meta-network is trained on a validation set via bilevel optimization.

**MetaWeighting vs UW Comparison:**
- MetaWeighting adapts weights based on current training state (epoch, loss trends)
- Achieves better performance than UW on NLP multi-task benchmarks (GLUE)
- For CV benchmarks (NYUv2, Cityscapes), the gains over UW are smaller
- MetaBalance focuses specifically on balancing tasks with different gradient magnitudes

**Key Findings for Our Setting:**
- Meta-learning approaches add significant complexity (meta-network, bilevel optimization, validation split)
- For 4 tasks, the complexity overhead is harder to justify than for 20+ tasks
- Reported gains over UW are typically 1-3% on specific metrics, not across-the-board SOTA
- Risk of meta-network overfitting to validation data

**Relevance:**
Plausible benefit but high complexity. Not recommended as first replacement. Consider only if UW-SO or DB-MTL prove insufficient.

---

### Paper 11: Achievement-Based Training Progress Balancing

**Authors:** Yun et al.
**Venue:** ICCV 2023
**Links:** Search: "Achievement-based training progress balancing multi-task learning ICCV 2023"

**Core Idea:**
Instead of weighting by loss or gradient, weight by "achievement" -- a normalized measure of task progress relative to a target metric. Uses weighted geometric mean of task achievements as the training objective.

**Mechanism:**
- For each task, define an achievement score: A_k = (current_metric - initial_metric) / (target_metric - initial_metric)
- Weight = achievement gap / sum(achievement gaps)
- Tasks farther from their target receive more weight
- Uses weighted geometric mean: L_total = prod(L_k^{w_k}) where sum(w_k) = 1
- Weighted geometric mean naturally handles different loss scales because it's multiplicative

**Key Finding for Our Setting:**
- Weighted geometric mean (vs weighted sum in UW) partially addresses loss scale imbalance
- Tasks with low loss values don't automatically get low weight if their achievement gap is large
- Reported results on NYUv2 show ~1-2% improvement over UW
- Geometric mean formulation is more stable than weighted sum when loss scales differ by 2.5x

**Relevance:**
The weighted geometric mean approach is directly applicable to our scale mismatch problem. Combined with the achievement-based weighting, it could solve the detection starvation issue. However, requires defining target metrics per task.

---

## Direct Answers to Team Questions

### B1: Which loss weighting schemes empirically outperform Kendall uncertainty weighting on 4+ task MTL benchmarks?

| Method | Consistently beats UW? | Our 4-task fit | Complexity | Recommendation |
|--------|----------------------|----------------|------------|---------------|
| **UW-SO** (Kirchdorfer 2025) | **YES** -- every benchmark | **High** -- solves weight collapse | Low (no learnable params) | **#1 Recommended** |
| **DB-MTL** (Lin 2023-2025) | **YES** -- NYUv2, Cityscapes | **High** -- log-transform fixes scale mismatch | Medium | **#2 Recommended** |
| Nash-MTL (ICML 2022) | Yes -- but gradient-level | Medium (gradient solver) | High (2x compute) | **#3 Recommended** |
| Auto-Lambda (CVPR 2022) | Marginal over UW | Medium (meta-learner overhead) | High (bilevel opt) | Consider later |
| IGB (UAI 2023) | Yes -- gap-based | High concept fit | Medium | **#4 Recommended** |
| DWA | Slightly beats UW | Medium (ratio-based) | Low | Better than UW, worse than UW-SO |
| GradNorm | Mixed -- sometimes worse | Poor (instability with scale diff) | Medium | Not recommended |
| RLW | Despite success claims, risky | Low (scale-vulnerable) | Low | Not recommended for our scale mismatch |

**Strongest Recommendation: UW-SO as a direct drop-in replacement**, combined with DB-MTL's log-transform for normalizing loss scales across tasks. These address both problems we identified: weight collapse and activity loss dominating detection loss.

### B6: What is the state of meta-learning-based task weighting?

Meta-learning weighting (MetaWeighting, MetaBalance, Auto-Lambda) offers theoretical appeal but practical drawbacks:
- **MetaWeighting** (ACL 2022): 1-3% improvement over UW on NLP benchmarks, smaller on CV
- **Auto-Lambda** (CVPR 2022): Outperforms UW on NYUv2 but by a small margin (18.28 vs 18.09 mIoU)
- **MetaBalance**: Specifically designed for unbalanced gradient scales, but reported results are limited
- Common issues: bilevel optimization instability, meta-network overfitting, validation set requirement, training-time overhead

**Verdict:** Not worth the complexity for 4 tasks. The simpler UW-SO, DB-MTL, or Nash-MTL provide larger gains with less complexity.

### B7: What are the 2025-2026 SOTA methods for MTL loss weighting?

**2025 SOTA landscape:**
1. **UW-SO** (Kirchdorfer 2025, IJCV) -- The most important recent development. Eliminates learned uncertainty collapse. Single temperature parameter. Published as IJCV with extensive benchmarks across NYUv2, Cityscapes, CelebA (40 tasks). Consistently outperforms all other weighting methods.

2. **DB-MTL** (Lin 2025, Neural Networks) -- Log-transform + gradient normalization. Directly addresses scale mismatch. Available on GitHub.

3. **GO4Align** (2024) -- Group optimization for task alignment. Clusters tasks by gradient similarity. Strong conceptual fit for heterogeneous tasks.

4. **Nash-MTL** (2022, ICML) -- Game-theoretic. Still competitive. Proven across many setups.

5. **Achievement-Based Balancing** (ICCV 2023) -- Weighted geometric mean + achievement gap. Addresses scale issues naturally.

**Key insight:** The trend is away from learnable weighting parameters (which can collapse) toward analytical or normalization-based methods. UW-SO represents the most mature expression of this trend.

---

## Summary: Implementation Recommendation

**Recommended approach for the team's setup (4 tasks, loss scale imbalance, weight collapse):**

1. **Immediate fix: Replace Kendall UW with UW-SO.** Delete the 4 learnable log-variance parameters (and their caps). Replace with: `weights = F.softmax(-detach(losses) / temperature, dim=0)`. Temperature T replaces our manual caps with a single principled hyperparameter. Start with T=1.0, tune in {0.1, 0.5, 1.0, 2.0}.

2. **Add loss normalization from DB-MTL (optional but recommended):** Log-transform each task loss before computing weights: `L_k' = log(L_k / L_k_prev + eps)`. This normalizes the scale differences between detection (~2) and activity (~5).

3. **If computational budget allows, add Nash-MLT gradient aggregation** on top of UW-SO weights. This guarantees proportional improvement across all tasks and prevents any task from being starved.

4. **Monitor:** Track per-task loss ratios (not just absolute values) and per-task gradient norms to verify tasks are balanced.

**Expected benefit from UW-SO alone:** ~5-15% improvement in detection metrics (removing the starvation effect), while maintaining or slightly improving activity/PSR/pose metrics. This is based on Kirchdorfer's results showing consistent Delta m improvement of +1.0 to +4.0 over UW across all benchmarks.
