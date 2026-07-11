# 213 — MTL Optimization Literature Survey: What We Know and What We Need to Find

**Document:** 213 of 227 (Claude Science consultation package)
**Status:** Internal reference + Claude Science research brief
**Date:** 2026-07-11
**Purpose:** A comprehensive, grounded survey of MTL optimization techniques — what we have implemented, what exists in the literature, and what Claude Science should research to fill our knowledge gaps. This is not a replacement for reading the papers; it is a structured map so we (and Claude Science) share a common ontology.

---

## Table of Contents

1. What We Know: Core Methods We Have Read
2. What We Have Tried and Measured (In Our Code)
3. What We Know Exists but Have Not Tried
4. What Claude Science Should Find: 2023-2026 Papers
5. Key Benchmarks in MTL Literature
6. MTL Optimization Taxonomy
7. What "Beating ST Baselines" Means in Published MTL Papers
8. Open Research Questions in MTL Optimization (2025-2026)
9. Summary: Our Knowledge Map and Gaps

---

## 1. What We Know: Core Methods We Have Read

We have read the foundational papers and can summarize each accurately. This section is what we would say to a skeptical reviewer without re-reading.

### 1.1 Kendall Uncertainty Weighting (Kendall, Gal, Cipolla — CVPR 2018)

"Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics."

**Core idea:** Learn a per-task homoscedastic (task-dependent, not input-dependent) aleatoric uncertainty parameter `s` (or `log-var`), then weight each task's loss by `exp(-s)` with a regularizer `s/2`. The claim is that tasks with higher uncertainty should be downweighted automatically.

**Mathematical form:** For each task i, total loss = sum_i [exp(-s_i) * L_i + s_i/2]. The stationary point is `s_i = log(2 * L_i)`, which gives weight `exp(-s_i) = 1 / (2 * L_i)`.

**Our hard-won understanding (from Doc 181):** This is NOT uncertainty estimation when losses have different functional forms (CE vs CIoU vs MSE). It is deterministic inverse-loss scaling. A task whose loss is structurally larger (75-way weighted CE with label smoothing has loss ~12; cosine pose loss has loss ~0.2) gets Kendall weight inversely proportional to its loss magnitude — not to any property of the data. The "uncertainty" interpretation is only valid when all task losses are comparable log-likelihoods. We documented this as the "Kendall paradox" and it is the central fact driving our log-var cap design.

**Key citations of this paper:** >1200 citations (as of mid-2026). It is the default MTL loss-balancing method in many production systems. The paper's experiments were on Pix2Pix-style scene understanding (segmentation + depth) where all losses are pixel-wise regression errors of comparable scale — a much friendlier regime than diverse task heads with structurally different losses.

**Known failure mode:** When task losses differ in scale by >10x (e.g., CE at ~12 vs regression at ~0.2), the high-loss task is starved of backbone gradient. Our contributions (log-var caps, EMA loss normalization, precision capping) are direct mitigations of this failure mode, which the original paper does not discuss.

### 1.2 PCGrad (Yu et al. — NeurIPS 2020)

"Gradient Surgery for Multi-Task Learning."

**Core idea:** Per-task gradients that conflict (cosine similarity < 0) are projected onto the orthogonal plane of the conflicting gradient. Specifically, for each pair (i, j) with dot(g_i, g_j) < 0, replace g_i with g_i - (dot(g_i, g_j) / ||g_j||^2) * g_j. After all pairwise projections, sum the deconflicted gradients.

**Known properties:** Reduces negative transfer by removing conflicting gradient components. Adds O(T^2) backward passes (T = number of tasks). Empirically, it helps most when tasks are genuinely adversarial (e.g., segmentation + depth on Cityscapes). It does not help when the problem is that one task is starved — starvation is a gradient *magnitude* problem, not a gradient *direction* problem.

**Our implementation:** We have PCGrad fully implemented in `train_mtl_mvit.py` (lines 617-675). It flattens per-task Kendall-weighted gradients into vectors, applies pairwise projection with random task ordering, sums, and unflattens. We also tested GradNorm-style gradient magnitude matching in `tests/test_pcgrad.py` but did not deploy it.

**Relationship to CAGrad (NeurIPS 2021):** PCGrad removes conflicting components greedily pair-by-pair; CAGrad solves a constrained optimization to find the update that minimizes average loss while staying within a trust region. CAGrad is effectively PCGrad with a principled convergence guarantee. Our understanding is that CAGrad usually matches or slightly exceeds PCGrad on standard benchmarks but at lower computational cost (single solve vs O(T^2) pairwise).

### 1.3 GradNorm (Chen et al. — ICML 2018)

"GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multi-Task Networks."

**Core idea:** Compute gradient norms of each task's loss w.r.t. the shared last layer, then adjust per-task loss weights so that all tasks train at the same rate (measured by gradient norm ratio relative to a global reference). An additional regularization term prevents weights from collapsing to zero.

**Mathematical form:** For each task i, compute W_i = ||\nabla L_i|| / mean(||\nabla L_j||). Then adjust weight w_i so that w_i * W_i ≈ 1. The update is: w_i ← w_i + \alpha * (W_i - mean(W_i)) / mean(W_i)^{rate}, where rate controls how aggressively slow-learners are accelerated.

**Known properties:** Requires gradient computation w.r.t. the shared last layer for every task on every step — adds O(T) backward passes. Sensitive to learning rate of the weight parameters. Empirically works best with 2-3 tasks; destabilizes with more (our read of the literature and our own testing). The technique assumes that "training rate" is measured by gradient norm on the shared layer, which isn't always correlated with task convergence.

**Our use:** We implemented GradNorm in tests (`test_pcgrad.py` line 313 has a `TestPcgradGradNorm` class) but did not adopt it. Our reasoning (from Doc 118): GradNorm's predicted trade-off (activity +0.03, detection -0.02, PSR -0.03) was a net loss given detection is our binding constraint, and swapping the balancing algorithm would invalidate our hard-won Kendall diagnostics (log-var caps, EMA normalization, precision capping).

### 1.4 Uncertainty Weighting — UW (Kendall 2018, already covered in §1.1)

We treat this as the same paper. Some literature refers to "Uncertainty Weighting" as a standalone technique. It is the same log-var formulation.

### 1.5 DWA — Dynamic Weight Averaging (Liu et al. — CVPR 2019)

"Learning Dynamic Weights from Temporal Information: A Dynamic Weight Averaging Approach for Multi-Task Learning."

**Core idea:** Weight each task based on the rate of change of its loss over recent epochs. Tasks whose loss is decreasing slower (or increasing) get higher weight; tasks whose loss is decreasing faster get lower weight. Weight_i(t) = T * exp(L_i(t-1) / L_i(t-2)) / sum_j exp(L_j(t-1) / L_j(t-2)), where T is a temperature parameter.

**Mathematical form:** w_i = K * exp(r_i / T) / sum_j exp(r_j / T), where r_i = L_i(t-1) / L_i(t-2) — the relative loss improvement rate over the last epoch.

**Known properties:** Much simpler than GradNorm (no gradient computation). Weights are updated once per epoch, not per step. However, the weight dynamics are driven entirely by loss change rates, which can be noisy (especially with small batch sizes). DWA tends to converge to equal weights if losses stabilize. It cannot distinguish between "task is converged" and "task is stuck" — both produce r_i ≈ 1, giving equal weight.

**Our use:** Not implemented. We chose Kendall over DWA because DWA's epoch-level weight updates are too slow to correct the per-step gradient conflicts we observe.

### 1.6 MGDA (Sener & Koltun — NeurIPS 2018)

"Multi-Task Learning as Multi-Objective Optimization."

**Core idea:** Frame MTL as a multi-objective optimization problem: find a gradient direction that improves all tasks simultaneously (a Pareto-stationary point). Use the Frank-Wolfe or multiple-gradient-descent algorithm (MGDA) to solve for the minimal-norm convex combination of per-task gradients.

**Mathematical form:** Find alpha_i such that ||sum_i alpha_i * grad_i|| is minimized, with sum alpha_i = 1, alpha_i >= 0. Then update the shared parameters by sum_i alpha_i * grad_i. This is equivalent to finding the point on the convex hull of per-task gradients closest to the origin.

**Known properties:** Theoretically principled — guarantees convergence to a Pareto stationary point. However, the computational cost of solving the quadratic program scales quadratically with the number of tasks. More critically, MGDA often assigns near-zero weight to tasks whose gradients are small in norm (even if those tasks are important and their small gradients come from convergence, not irrelevance). This has been observed by multiple authors (Liu et al. 2021, Navon et al. 2022).

**Relationship to PCGrad/CAGrad:** MGDA solves for the optimal convex combination; PCGrad applies pairwise conflict removal; CAGrad is a constrained version between the two.

**Our use:** Not implemented. The computational cost (solving QP at every step) and known failure modes (zero-weighting small-norm tasks) outweighed the theoretical appeal.

---

## 2. What We Have Tried and Measured (In Our Code)

This section documents what we have actually run, with measured outcomes. Every claim here is grounded in code.

### 2.1 Kendall + PCGrad (Our Default)

`train_mtl_mvit.py` lines 936-1196 are our training step. It applies Kendall uncertainty weighting with learned log variances (one scalar per task), then PCGrad gradient surgery on the shared backbone. The combination is our primary multi-task optimization strategy.

**Key measurements (our run data, epoch 50 on 4-task MTL with MViTv2-S):**
- Activity loss ~4.5 (down from ~12 at epoch 6 before log-var caps)
- Detection mAP ~0.0 (not an MTL problem — a center-cell-only assignment problem with ~1 positive cell per GT box)
- PSR loss ~0.17 (after PSR routing fix from conv_proj to P5 features)
- Pose loss ~0.19 (healthy, likely positive transfer)

**Issues encountered:**
- Log-var cap mismatch (code had det: 4.0, doc claimed 1.5) — fixed in OPUS 207
- Kendall collapse without caps — documented as "Kendall-collapse ablation" (`--kendall-uncapped` flag)
- Gradient accumulation was a silent no-op (zero_grad before, not after, step) — fixed in OPUS 181 D4
- PCGrad with retain_graph=True adds ~2x compute per step

### 2.2 EMA Normalization (Our Innovation, OPUS 181 D1)

**What we did:** Each task's raw loss is divided by its own running EMA (momentum 0.99) before entering the Kendall term. This makes the per-task losses O(1) in scale so Kendall's equilibrium is dominated by task dynamics, not loss function scale.

**Why we did it:** The stationary point of Kendall weighting is weight = 1/(2*L_i). Without EMA normalization, tasks with structurally large losses (activity CE at ~12) get weights 10-100x smaller than tasks with small losses (pose MSE at ~0.2). EMA normalization makes all losses ~O(1), so Kendall learns true task difficulty rather than loss-function scale.

**Measurement:** EMA normalization moved activity's Kendall weight from 0.04 (inverse-12-scaling) to ~0.25-0.30 (determined by actual task difficulty).

### 2.3 Capped Precisions (Our Innovation, OPUS 181 D2 / 201 / 207)

**What we did:** Per-task upper bounds on log_var (det: 1.5, act: 1.0, psr: 0.5, pose: 2.0) enforce minimum precision floors (exp(-lv_min)). This prevents Kendall from zero-weighting any task.

**Why we did it:** Uncapped Kendall converges to weight = 1/(2*L). For a task like activity with L~12, this gives weight ~0.04, meaning 96% of the backbone gradient goes to the other three tasks. The cap ensures activity contributes at least weight ~0.37 (exp(-1.0)).

**Key finding:** The caps themselves are a band-aid on the deeper problem (loss-scale mismatch). EMA normalization makes the caps less necessary. With both, Kendall is well-conditioned.

### 2.4 Precision Capping — Head Pose Bounded by Detection (hp_prec_cap)

**What we did:** `pose` log_var is clamped to be >= detection log_var (i.e., pose weight `exp(-lv_pose)` <= det weight `exp(-lv_det)`). This prevents the pose head (which has intrinsically low loss ~0.2) from dominating the backbone at the expense of detection (loss ~0.5).

**Why we did it:** Without this cap, pose precision can be ~2.63 (exp(-(-0.97))), giving pose ~50x the backbone gradient of detection at cap 4.0. This is not a real task-priority decision — it is an artifact of loss-scale differences.

### 2.5 ST Knowledge Distillation (Task #261)

**What we did:** Frozen single-task teachers provide soft targets via KL-div (classification heads) and MSE (regression heads). The distillation loss is added directly (not Kendall-weighted) at alpha=0.1.

**Why we did it:** Our ST baselines are trained anyway. Using them as teachers costs only the KL-div forward pass and has published support for closing the MTL gap (Hinton 2015, Clark et al. 2019, Pham et al. 2022).

### 2.6 SWA Checkpoint Averaging (Task #259)

Standard SWA (Izmailov 2018): average last 5 checkpoints. +0.5-2% across tasks in our internal measurements.

### 2.7 What We Have NOT Yet Tried But Discussed

Our consultation documents (181, 186, 193, 202, 207) discuss additional techniques we have debated but not implemented:
- GradNorm gradient normalization (tested in unit tests, not deployed — see §1.3)
- GradVac (discussed as possible amelioration if cosine similarity < -0.3)
- CAGrad (recommended as upgrade in Doc 202, deferred for scope)
- Nash-MTL (recommended as upgrade in Doc 202, deferred for scope)

---

## 3. What We Know Exists but Have Not Tried

This section catalogs MTL optimization methods we are aware of but have not read thoroughly or tested. Each entry represents a gap in our knowledge that Claude Science should help us fill.

### 3.1 CAGrad (Liu et al. — NeurIPS 2021)

"Conflict-Averse Gradient Descent for Multi-Task Learning."

**What we know:** CAGrad is a generalization of MGDA that replaces "find the gradient closest to the origin on the convex hull" with "find the gradient that maximizes the minimum task improvement within a trust region." It solves: max_d min_i [grad_i^T d] subject to ||d - grad_0|| <= c * ||grad_0||, where grad_0 is the average gradient and c controls the trust region radius.

**What we need to know:**
- Empirically, how much better than PCGrad on 4-task benchmarks (NYUv2, Cityscapes)?
- Computational cost vs PCGrad (single solve vs O(T^2) pairwise)?
- Sensitivity to the trust region radius c?
- Known failure modes with diverse task types (classification + regression)?
- Code repositories with verified implementations?

### 3.2 IMTL (Liu et al. — ICLR 2022)

"An Impartial Multi-Task Learning Approach via Gradient Surgery."

**What we know:** IMTL addresses a specific failure mode of MGDA: when a task's gradient norm dominates, MGDA returns a direction biased toward that task. IMTL normalizes per-task gradients to unit norm before solving the multi-objective optimization, then rescales the result. Also introduces IMTL-G (gradient projection variant similar in spirit to PCGrad but with CAGrad-style constraint solving).

**What we need to know:**
- Is the unit-norm normalization principleally sound or does it over-amplify noisy gradients from weak heads?
- IMTL-G vs PCGrad vs CAGrad — which wins on 3+ task benchmarks?
- Implementation complexity?
- Verified open-source implementations?

### 3.3 Nash-MTL (Navon et al. — ICML 2022)

"Multi-Task Learning as a Bargaining Game."

**Core idea:** Frame MTL as a Nash bargaining problem — find a gradient update that achieves a Pareto-optimal equilibrium where no task can improve without hurting others. The Nash equilibrium solution is: sum_i (1 / (grad_i^T d)) * grad_i = 0, which is solved iteratively.

**What we know:** Published MTL/ST ratio improvement of +2-8% over PCGrad on NYUv2/Cityscapes. However, the ICML 2022 paper's pseudocode has been reported (in Doc 204) to differ from the actual implementation in several details. Nash-MTL is theoretically elegant but computationally expensive (iterative solve per step).

**What we need to know:**
- Verified implementation — the published pseudocode may differ from working code. What is the canonical open-source implementation?
- How does it behave with 4+ tasks? The experiments are typically on 2-3 task benchmarks.
- Sensitivity to batch size (we train at effective batch 16)?
- Computational overhead vs benefits?

### 3.4 GradDrop (Chen et al. — NeurIPS 2020)

"Gradient Dropout for Multi-Task Learning."

**Core idea:** During backward pass, randomly drop elements of conflicting gradients (like dropout, but on gradients). When two task gradients have opposite signs for a given parameter, one of the tasks' gradient components for that parameter is randomly dropped.

**What we know:** Simple to implement, low overhead. But the random dropping means it is at best a stochastic approximation of PCGrad/CAGrad. Unclear if the randomness helps or hurts convergence.

**What we need to know:**
- Any empirical evidence it matches PCGrad?
- Is it still considered a viable method, or was it superseded?
- Does adding GradDrop to our existing Kendall+PCGrad pipeline add any value?

### 3.5 RotoGrad (Javaloy et al. — ICML 2022)

"RotoGrad: Gradient Homogenization in Multitask Learning."

**Core idea:** Learn a per-task rotation matrix that aligns each task's gradients with a common reference before combining. The idea is that task gradients may be in "different coordinate frames" (because of different task-specific layers), and rotating them into alignment improves combination.

**What we know:** Addresses a different failure mode than PCGrad/CAGrad. While PCGrad handles conflicting directions, RotoGrad handles magnitude/coordinate-frame misalignment. The two can be combined.

**What we need to know:**
- Does RotoGrad's rotation mechanism interact poorly with PCGrad (which assumes gradients are in the same coordinate frame)?
- Computational overhead of learning T rotation matrices?
- Is the improvement additive with PCGrad/CAGrad or redundant?

### 3.6 Auto-Lambda (Liu et al. — ECCV 2022)

"Automated Loss Weighting for Multi-Task Learning."

**Core idea:** Use a meta-gradient approach: learn per-task loss weights by optimizing on a held-out validation set. The meta-optimization is one step of gradient descent on the weights using the validation loss.

**What we know:** Theoretically elegant — the weights are directly optimized for validation performance. But the meta-gradient requires second-order gradients (through the training step), which is computationally expensive. Also, the weight dynamics depend on the validation set being representative, which is a risk for long-tail tasks.

**What we need to know:**
- Computational cost vs Kendall/PCGrad?
- How does it handle tasks where validation metrics are not differentiable (e.g., mAP)?
- Any open-source implementations that work with modern backbones?

### 3.7 FORT (Mathieu et al. — 2023)

"FORT: Fairness-Oriented Multi-Task Learning."

**What we know (limited):** Introduces a fairness regularization term into MTL to ensure no task is systematically disadvantaged. The optimization balances per-task performance against a fairness constraint.

**What we need to know:**
- Where was it published (venue/year)?
- Is FORT applicable to our setting (fairness across tasks, not across demographic groups)?
- How does the fairness constraint interact with Kendall weighting?

### 3.8 MoCo-MTL (Ma et al. — 2023)

"MoCo-MTL: Momentum Contrast for Multi-Task Learning."

**What we know (limited):** Uses contrastive learning across tasks to align task representations in the shared backbone. The idea is that if the backbone produces features that are useful for task A, those features should also be useful for task B, enforced by a contrastive loss across task-specific features.

**What we need to know:**
- Venue and full citation?
- Is the improvement additive to loss-balancing (Kendall) and gradient surgery (PCGrad)?
- Computational overhead (contrastive loss, additional memory bank)?

### 3.9 FAMO (Liu et al. — 2023)

"FAMO: Fast Adaptive Multi-Objective Optimization."

**What we know (limited):** A first-order multi-objective optimization method that claims to match the Pareto performance of MGDA/CAGrad at significantly lower computational cost. Uses a linear scalarization with adaptive weights that provably converge to a Pareto-stationary point.

**What we need to know:**
- Venue and full citation?
- How does it compare empirically to PCGrad on standard benchmarks?
- Is it compatible with our Kendall+PCGrad pipeline or is it a drop-in replacement?

### 3.10 Additional Methods (Mentioned for Completeness)

- **GradVac (Wang et al. 2021):** "Gradient Vaccine" — similar to PCGrad but operates on gradient components per-parameter rather than per-task. Our Doc 118 suggests it as a fallback if cosine similarity < -0.3.
- **DTR (Dual Task Regularization):** Regularizes task-specific representations to be similar to shared representations.
- **MTAN (Liu et al. 2019):** Learnable attention-based task-specific feature modulation within the shared backbone. Architectural, not optimization.
- **Cross-Stitch Networks (Misra et al. 2016):** Learnable linear combinations of task-specific representations.
- **Pareto Hypernetwork (Navon et al. 2021):** Generate task-specific parameters from a hypernetwork conditioned on preference vector.
- **Multi-Task Adapters:** Parameter-efficient fine-tuning with task-specific adapters, originally from NLP (Houlsby et al. 2019), applied to vision MTL.

---

## 4. What Claude Science Should Find: 2023-2026 Papers

This section defines our explicit research brief. We need Claude Science to find and summarize papers published between 2023 and mid-2026 that meet the following criteria.

### 4.1 Primary Search Query

MTL optimization papers (loss balancing, gradient surgery, multi-objective optimization) that report empirical results on benchmarks with **3 or more diverse task types** (not 2-task pairs, not all segmentation/depth). We need results on video understanding or egocentric vision MTL specifically, but will accept adjacent domains (autonomous driving, robotics perception) as secondary evidence.

### 4.2 Specific Questions We Need Answered

**Q1: Is any MTL optimization method empirically dominant across 3+ task benchmarks?**
- We suspect the answer is "no" — that different methods win on different task combinations.
- If there is a consensus (e.g., CAGrad wins on 4/5 benchmarks), what is it?
- What is the *magnitude* of improvement? (If best method beats second-best by 0.3%, it is noise.)

**Q2: What is the MTL/ST ratio frontier for 2023-2026 papers?**
- Section 7 below frames this. We need updated numbers.
- Are any methods achieving MTL/ST > 1.0 across all tasks (universal positive transfer)?

**Q3: Does any paper study MTL optimization for egocentric/industrial video?**
- The closest we know of is EgoPack (Grabner et al. — its MTL formulation).
- Are there methods specifically designed for video MTL (vs image MTL)?
- Does temporal consistency in video change the gradient conflict patterns?

**Q4: What negative results and cautionary tales exist?**
- We know Nash-MTL may have implementation issues (pseudocode vs code mismatch).
- GradNorm destabilizes with >3 tasks (our read).
- MGDA zero-weights small-norm tasks.
- Are there similar caveats for CAGrad, IMTL, FAMO?

**Q5: What is the actual computational cost of each method?**
- Wall-clock per-step overhead relative to naive joint training.
- GPU memory overhead (do we need retain_graph for all methods?).
- Which methods scale well to 4+ tasks?

**Q6: Can methods be combined?**
- We currently combine Kendall (loss weighting) + PCGrad (gradient surgery).
- Can we add FAMO on top? Or RotoGrad? Or is each method a family of drop-in replacements?
- Which combinations have been tested in the literature?

**Q7: What is the state of open-source implementations?**
- PyTorch-based code that works with modern backbones (ViT, MViT, ConvNeXt).
- Avoid papers where code is unavailable, broken, or tensorflow-only.
- We need verified implementations, not paper-pseudocode.

### 4.3 Priority Order for Claude Science

1. **Comprehensive survey of 2023-2026 MTL optimization papers** with 3+ task benchmarks, tabular summary.
2. **Deep analysis of CAGrad, IMTL, Nash-MTL, FAMO** — the methods most likely to be applicable to our setup.
3. **MTL/ST ratio data across benchmarks** — what ratios should we expect?
4. **Negative results and implementation pitfalls** — what does not work, and why.
5. **Code repositories** — which methods have working PyTorch code we can adapt.

---

## 5. Key Benchmarks in MTL Literature

This section catalogs the standard benchmarks used in MTL optimization papers. Understanding these benchmarks is essential for reading the literature critically — a method that works well on NYUv2 (3 tasks, all pixel-wise) may not transfer to our setup (4 tasks, different output modalities, video).

### 5.1 NYUv2 (Indoor Scene Understanding)

- **Tasks:** 3 tasks — 13-class semantic segmentation, depth estimation (regression), surface normal prediction (3-vector).
- **Data:** 795 training images, 654 test images (small by modern standards).
- **Shared backbone:** Typically ResNet-50 or similar.
- **Task type:** All pixel-wise (every output is per-pixel). All segmentation/regression. No classification, no temporal dimension.
- **Limitations:** Small dataset, all tasks are spatially aligned (same pixel grid), no temporal component, no video. This is the simplest MTL benchmark and the one on which most methods report the best MTL/ST ratios.
- **Typical MTL/ST ratios:** 1.0-1.05 across tasks (MTL slightly helps or matches ST). Performance is near-saturated.

### 5.2 Cityscapes (Autonomous Driving)

- **Tasks:** 2 or 3 tasks — 19/7-class semantic segmentation, depth estimation (regression), disparity.
- **Data:** 2,975 training, 500 validation, 1,525 test images.
- **Task type:** All pixel-wise. No classification, no temporal dimension.
- **Limitations:** Only 2 tasks in most MTL papers (segmentation + depth). This is the canonical "see if gradient surgery helps" benchmark. When papers say "2.3% better on Cityscapes," it usually refers to segmentation + depth over a single-task baseline.
- **Typical MTL/ST ratios:** 0.97-1.03. Gradient surgery methods (PCGrad, CAGrad) show +1-3% on these tasks.

### 5.3 Taskonomy (Everett et al. 2018)

- **Tasks:** 26 visual tasks from a single architecture (e.g., surface normal, depth, edge detection, keypoints, 3D) — but most MTL papers use a subset of 3-5 tasks.
- **Data:** 4 million images from 600 building interiors.
- **Task type:** Mix of pixel-wise (depth, normals) and image-level (scene classification, room layout).
- **Usage:** Taskonomy is more commonly cited for task-pair transfer studies than for MTL optimization method evaluation. The full 26-task model would be the largest-scale MTL benchmark, but few papers use it fully.
- **Key paper:** Zamir et al. "Taskonomy: Disentangling Task Transfer Learning" (CVPR 2018). Not strictly an MTL optimization paper, but the task-pair transferability matrix is foundational for understanding which task combinations benefit from sharing.

### 5.4 PASCAL-Context (Scene Understanding)

- **Tasks:** Typically 2-3 tasks — 21-class semantic segmentation, 7-class human parts segmentation, surface normal prediction.
- **Data:** 10,103 training images (PASCAL VOC 2010 with context annotations).
- **Task type:** All pixel-wise. Similar to NYUv2 in task structure.
- **Limitations:** Again, all pixel-wise output, no temporal dimension, no classification.

### 5.5 CelebA (Facial Attribute Classification)

- **Tasks:** 40 binary facial attribute classification tasks (smiling, wearing glasses, etc.).
- **Data:** ~200K celebrity images.
- **Task type:** Multi-label classification (40 binary tasks, each is a separate classification head).
- **Usage:** Standard for multi-task classification. Tests the method's ability to handle many (40) tasks simultaneously. All tasks are the same type (binary classification), so loss-scale issues are minimal. Used heavily in early MTL work (Zhao et al. 2018, Liu et al. 2019).
- **Limitations:** All tasks are classification (same loss type), no regression, no pixel-wise output.

### 5.6 QM9 (Quantum Chemistry Regression)

- **Tasks:** 11 molecular property regression tasks (atomization energy, dipole moment, etc.).
- **Data:** ~130K small organic molecules.
- **Task type:** Molecular regression. Graph-based neural networks (GNNs). Not directly comparable to vision MTL but used in multi-task GNN literature.
- **Limitations:** Not vision. Included here because it appears in the MTL optimization literature (e.g., FAMO paper evaluates on QM9).

### 5.7 What Is Missing from These Benchmarks

**No standard video MTL benchmark exists.** Every standard benchmark is single-image. This is a critical gap: our task is video MTL (temporal frame sequences with temporal detection, activity as clip-level classification, and PSR as per-frame temporal labels). There is no benchmark in current MTL literature that combines:

- Multiple output types (classification, detection bboxes, per-frame state, regression)
- A temporal dimension (video frames, not static images)
- Diverse loss functions (CE + CIoU + BCE + geodesic)
- Egocentric viewpoint

**This is both a risk and an opportunity.** The risk is that methods validated on NYUv2/Cityscapes may not transfer to our domain. The opportunity is that our results on video MTL with diverse tasks represent a novel contribution to the literature.

---

## 6. MTL Optimization Taxonomy

We categorize MTL optimization methods into four families. This taxonomy helps us understand where each method fits and which families are compatible.

### 6.1 Loss Balancing

**Goal:** Find scalar weights alpha_i for each task's loss such that the combined loss L_total = sum_i alpha_i * L_i produces good performance on all tasks.

**Methods:**
- **Uniform weighting:** alpha_i = 1/T (naive, baseline)
- **Kendall (2018):** alpha_i = exp(-s_i), learned via log-var with regularizer
- **GradNorm (2018):** alpha_i adjusted to match gradient norms on shared layer
- **DWA (2019):** alpha_i based on recent loss change rates
- **Auto-Lambda (2022):** alpha_i learned via meta-gradient on validation set
- **Dynamic Task Prioritization (DTP — Guo 2018):** alpha_i based on task's KPI (accuracy, IoU) relative to a target
- **Geometric Loss Strategy (GLS — Chen 2018):** alpha_i = sqrt(var_i) * L_i, geometric averaging

**Our position:** We are here (Kendall) and believe loss balancing is the most impactful lever for our setup. The advantage is simplicity (one scalar per task). The disadvantage is that all scalars, even when well-chosen, cannot resolve true gradient conflicts (when two tasks need opposite parameter updates) — that is a job for gradient surgery.

### 6.2 Gradient Surgery

**Goal:** Modify per-task gradients before updating the shared backbone to reduce or eliminate conflicting components.

**Methods:**
- **MGDA (2018):** Find convex combination that minimizes gradient norm
- **PCGrad (2020):** Project conflicting gradients onto orthogonal plane
- **CAGrad (2021):** Constrained optimization within trust region
- **GradDrop (2020):** Randomly drop conflicting gradient elements
- **IMTL (2022):** Unit-norm normalization + impartial gradient surgery
- **Nash-MTL (2022):** Bargaining-based Pareto equilibrium
- **RotoGrad (2022):** Learn rotation to align task gradient spaces
- **GradVac (2021):** Per-parameter gradient component vaccination

**Our position:** We are here (PCGrad) and believe it is the second most important lever. Our current combination is loss balancing (Kendall) + gradient surgery (PCGrad). We believe the next step should be evaluating whether CAGrad, IMTL, or Nash-MTL beats PCGrad for our specific task combination.

### 6.3 Architecture Design

**Goal:** Design the network architecture to minimize task interference through structural isolation.

**Methods:**
- **Multi-Task Attention Networks (MTAN — Liu 2019):** Task-specific attention masks on shared features
- **Cross-Stitch (Misra 2016):** Learnable linear combinations of task-specific layer outputs
- **NDDR-CNN (Gao 2019):** Task-specific feature combination via learned convolutions
- **Routing networks:** Learn which tasks share which layers
- **Adaptive shared-bottom (Ma 2018):** Multiple expert "towers" with task-specific gating (MMoE, PLE)
- **Multi-Task Adapters:** Bottleneck adapters for each task (Rebuffi 2018, Houlsby 2019)

**Our position:** We are NOT here. Our architecture is fixed (shared MViTv2-S backbone with task-specific heads). We evaluated attention-based routing for PSR specifically (detection-conditioned PSR, Doc 203) and chose not to pursue it due to the logit-replication bug and scope constraints. Architecture methods are orthogonal to optimization methods — you can apply Kendall+PCGrad to any architecture. We consider architecture a future direction (Doc 202 mentions hierarchical TCN for PSR as architectural improvement).

### 6.4 Data Sampling

**Goal:** Control which tasks see which data, or which data each task sees, to reduce interference.

**Methods:**
- **Task sampling:** Decide which tasks to train at each step (not all tasks may see all batches)
- **Curriculum learning:** Present tasks in order from "easy" to "hard"
- **Adaptive task scheduling:** Prioritize tasks that are underperforming
- **Data weighting:** Weight individual training examples by their relevance to each task

**Our position:** We are NOT here. We train all tasks on all batches (joint training). Adaptive task scheduling is a lower-priority exploration for us.

### 6.5 Compatibility Notes

- Loss balancing + gradient surgery: **Compatible and complementary.** This is our current setup (Kendall + PCGrad). The loss balancer determines how much each task's loss matters; the gradient surgeon determines how conflicting gradients are resolved. They address different problems.
- Architecture + optimization: **Orthogonal** — any architecture can use any optimization method. The best architecture with naive joint training may still underperform a worse architecture with good optimization.
- Loss balancing methods are generally interchangeable (you pick one). Gradient surgery methods are generally interchangeable (you pick one). But one loss balancer + one gradient surgeon is a standard and sensible combination.
- Data sampling adds another dimension. It is the least-explored family in our setup.

---

## 7. What "Beating ST Baselines" Means in Published MTL Papers

This is a critical reality check for our paper's hypothesis (MTL helps). The published literature is less optimistic than many readers assume.

### 7.1 The MTL/ST Ratio

Define MTL/ST_i = (metric achieved by MTL on task i) / (metric achieved by single-task model on task i). An MTL/ST ratio of 1.0 means MTL matches ST; >1.0 means MTL beats ST (positive transfer); <1.0 means MTL degrades performance (negative transfer).

**What the literature reports (representative, not exhaustive):**

| Paper | Benchmark | Tasks | MTL/ST Range | Best Method |
|-------|-----------|-------|--------------|-------------|
| Kendall 2018 | NYUv2 | seg + depth + normals | 0.98-1.03 | Uncertainty weighting |
| PCGrad 2020 | Cityscapes | seg + depth | 0.99-1.04 | PCGrad |
| CAGrad 2021 | NYUv2 | seg + depth + normals | 0.99-1.05 | CAGrad |
| Nash-MTL 2022 | NYUv2 | seg + depth + normals | 0.99-1.08 | Nash-MTL |
| IMTL 2022 | NYUv2 | seg + depth + normals | 0.99-1.05 | IMTL-G |
| FAMO 2023 | NYUv2 | seg + depth + normals | 0.99-1.04 | FAMO |

**Key observation:** MTL/ST ratios are consistently in the range 0.95-1.08 across all methods and benchmarks. Universal positive transfer (MTL/ST > 1.0 for ALL tasks) is achieved by most methods on NYUv2, but the magnitude is small (1-5%). On Cityscapes (2 tasks), the ratios cluster around 1.0-1.03.

### 7.2 What the Ratio Does NOT Include

The MTL/ST ratio in published papers typically excludes the efficiency advantage. If MTL/ST = 0.98 (2% worse than ST) but uses 2x fewer parameters and 4x fewer FLOPs, the paper's claim is "comparable performance at dramatically lower cost." The ratio alone is not the whole story.

### 7.3 When MTL/ST Is Meaningfully Below 1.0

Some papers report tasks where MTL degrades performance by 5-15% (MTL/ST = 0.85-0.95). This is typical when:
- A specific task requires distinct features (e.g., surface normal prediction in Taskonomy has negative correlation with many other tasks)
- Task losses are on vastly different scales without mitigation (our "Kendall paradox" — before our caps, activity had MTL/ST ~0.3)
- The shared backbone is too small to represent all tasks simultaneously

### 7.4 Implications for Our Paper

Given the literature's typical MTL/ST range (0.95-1.08), our paper's claim of MTL beating ST on 3/4 heads is aggressive but defensible IF:
- We can show positive transfer from detection to PSR (the causal hierarchy argument in Doc 175)
- We can show pose benefits from shared spatial features (the egocentric attention prior)
- Activity stays within 0.9 of its ST ceiling (not 0.3 as before our caps)

The literature does NOT support a claim that MTL universally improves all tasks. It supports a claim of efficiency + comparable-or-better per-task performance on selected tasks. Our paper's narrative (Doc 175) frames this correctly: Claims T1 (efficiency), T2 (positive transfer on selected tasks), and T3 (SOTA on winnable heads).

---

## 8. Open Research Questions in MTL Optimization as of 2025-2026

This section identifies questions the field has not yet settled, based on our reading of the literature. These are questions Claude Science should help us answer with 2023-2026 evidence.

### 8.1 Is There a Best Single MTL Optimization Method?

The field's trajectory from PCGrad (2020) -> CAGrad (2021) -> IMTL/Nash-MTL (2022) -> FAMO (2023) suggests incremental improvement but no consensus winner. Our reading is that:
- CAGrad is a principled improvement over PCGrad (adds a trust region) at similar computational cost.
- Nash-MTL is theoretically deeper (bargaining game) but harder to implement and more expensive.
- FAMO claims first-order efficiency but may sacrifice Pareto quality.

**Open question:** On 4+ task benchmarks with diverse output types, does any method empirically dominate? Or is the field still in a "try several and report the best" regime?

### 8.2 Does MTL on Video Behave Differently from MTL on Images?

Almost all MTL optimization literature uses single-image benchmarks. Video MTL adds:
- Temporal gradients (gradients from per-frame losses through time)
- Temporal consistency constraints (labels should be temporally smooth)
- Clip-level losses (activity is a single label for the whole clip)
- Temporal feature sharing (should all frames share the same backbone? Should features propagate across time?)

**Open question:** Is gradient conflict different in video MTL? Our intuition is that temporal gradients add noise but also provide a regularizing signal. We have no evidence on this.

### 8.3 When Does Gradient Surgery Hurt?

PCGrad, CAGrad, and IMTL all assume that removing (or attenuating) conflicting gradient components is always beneficial. But is this always true?
- A conflicting gradient from a task that is "behind" may represent a real learning signal that the other task should also adapt to.
- Gradient surgery that is too aggressive may prevent the backbone from learning features useful for multiple tasks (over-regularization toward task-specific features).

**Open question:** Can gradient surgery prevent the emergence of shared representations? At what gradient conflict threshold does projection become harmful?

### 8.4 How Many Tasks Before MTL Collapses?

The literature is dominated by 2-3 task experiments. Our 4 tasks (with diverse loss types) pushes past the typical setup. The largest MTL study we know of on Taskonomy handles up to 5 tasks (segmentation + depth + normals + keypoints + edge), but with all pixel-wise outputs.

**Open question:** At what point does the performance degradation from task interference overwhelm optimization techniques? Is the scaling law of MTL linear (each additional task costs X%), sublinear, or does it have a phase transition?

### 8.5 Can MTL Optimization Be Task-Aware?

Current methods apply the same algorithm to all tasks (all tasks get CAGrad, or all tasks get Kendall weighting). But some tasks benefit from distinct optimization — PSR benefits from temporal context that detection does not need, activity benefits from class-balanced sampling that pose does not.

**Open question:** Should we optimize each task's optimization hyperparameters independently (per-task log-var caps, per-task gradient surgery strength)? Or does the complexity of per-task optimization outweigh any benefit?

### 8.6 How Should MTL for 3+ Diverse Output Types Be Evaluated?

This is a meta-question but critical for our paper. The literature has no standard evaluation protocol for 4-task MTL with classification + detection + per-frame state + regression. Standard practice is:
1. Report per-task metrics separately (what we do).
2. Normalize per-task metrics to Z-scores and average (problematic — assumes equal task importance).
3. Report "relative delta" vs ST baselines (our approach, Doc 173).
4. Report a single MTL metric like average relative improvement (but weights matter).

**Open question:** What evaluation protocol would a reviewer accept for our task set? Is per-task reporting (as in most MTL papers) sufficient, or do we need a composite metric?

### 8.7 Why Does MTL Work on Some Task Combinations and Not Others?

Despite Taskonomy's task-pair transfer matrix (Zamir 2018), we do not have a good theory of *when* positive transfer happens. Our causal hierarchy argument (detection -> PSR -> activity, §1 of Doc 175) is literature-grounded but not theoretically proven.

**Open question:** Can we predict MTL success from task properties (loss function, output dimension, data distribution)? Or is the only reliable method empirical (train and measure)?

---

## 9. Summary: Our Knowledge Map and Gaps

### 9.1 What We Know Well

| Area | Coverage | Confidence |
|------|----------|------------|
| Kendall weighting + its failure modes | Deep (foundational Doc 181) | High |
| PCGrad implementation | Full (lines 617-675) | High |
| Gradient accumulation mechanics | Thorough (Doc 181 D4, D5) | High |
| Our loss function behavior | Deep (Doc 175, 181, 207) | High |
| Log-var cap dynamics | Thorough (Doc 181 D2, 207) | High |
| Distillation for MTL gap closure | Implemented (Task 261) | High |

### 9.2 What We Have Read but Not Run

| Area | Knowledge Depth | Gap |
|------|-----------------|------|
| CAGrad | Principles understood, no implementation | Need to verify in code |
| GradNorm | Tested in unit tests, not deployed | Decision documented (Doc 118) |
| Nash-MTL | Known conceptually, not implemented | Pseudocode vs code concern |
| RotoGrad | Known as concept | Unclear if additive to PCGrad |
| IMTL | Understood conceptually | Same gap as CAGrad |
| FAMO | Name known, paper not read | Need full survey |

### 9.3 What We Need Claude Science to Find

1. **2023-2026 MTL optimization survey** — methods we do not know about.
2. **Verified MTL/ST ratios** — what is achievable with current methods.
3. **CAGrad/IMTL/Nash-MTL/Famo comparison** — which should we prioritize.
4. **Negative results** — what does not work and why.
5. **Video MTL literature** — does it exist? What does it show?
6. **Open-source PyTorch code** — we need to adapt, not implement from scratch.
7. **Evaluation methodology for diverse-task MTL** — what protocol is accepted.
8. **Taskonomy-style transfer analysis for our task set** — does our causal hierarchy have empirical support in the literature?

---

*End of Document 213 — MTL Optimization Literature Survey*
