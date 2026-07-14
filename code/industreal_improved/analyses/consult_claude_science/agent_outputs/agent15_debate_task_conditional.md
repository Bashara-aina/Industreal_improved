# Agent 15: Debate -- Failure Cases for Task-Conditional Modulation in MTL

## Challenging Agent 5's Claim That TSBN/TS-sigma-BN "Essentially Matches ST Performance"

---

## Executive Summary

Agent 5 recommends TSBN/TS-sigma-BN as closing the MTL-to-ST gap to "98-100%."
The evidence from the papers Agent 5 itself cites tells a different story:

1. **TSBN actually HURTS segmentation performance** on NYUv2 (53.93 -> 53.44 mIoU). Gains are concentrated in depth estimation, not uniform.
2. **Standard adapters, LoRA, BitFit, and VPT-deep all UNDERPERFORM single-task learning** on PascalContext with pretrained ViT-S (negative delta-m in every case).
3. **Most soft-sharing methods underperform STL on the CelebA 40-task benchmark** -- the TSBN authors' own admission.
4. **Task-conditional modulation cannot resolve gradient interference** that arises from fundamentally competing task objectives.
5. **The methods add meaningful complexity** (differential learning rates, sigmoid gating, per-task BN statistics) that the parameter-overhead comparison obscures.

---

## 1. FiLM Failure Cases

### 1.1 FiLM Was Designed for Visual Reasoning, Not Multi-Task Learning

The original FiLM paper (Perez et al., 2018) addresses visual reasoning: a single model conditioned on a language query. It was never evaluated on multi-task benchmarks like NYUv2, Cityscapes, or PascalContext. The FiLM-Ensemble paper (Turkoglu et al., 2022) extends FiLM to uncertainty quantification (ensembles), not to resolving task interference in MTL.

**No FiLM paper demonstrates that FiLM alone closes the MTL-to-ST gap across multiple dense prediction tasks.** Agent 5's claim of "95-98% ST retention" for FiLM is extrapolated from the ensemble/UQ literature, not from direct MTL benchmarks.

### 1.2 FiLM-Ensemble's Own Failure Examples

The FiLIM-Ensemble paper reports that its implicit ensemble sometimes fails due to:
- **Off-by-one counting errors** (96.1% of counting mistakes are off-by-one), revealing that FiLM learns approximate but imprecise relationships.
- **Logical consistency failures**: The model gives contradictory answers to related questions (e.g., correctly counting 1 gray object and 2 cyan objects while simultaneously answering "there are the same number").
- **Occlusion sensitivity**: Many errors stem from partial occlusion, which the authors note "may likely be fixed using a CNN that operates at a higher resolution."

These failure modes in the ensemble context suggest that FiLM modulation can produce internally inconsistent feature representations.

### 1.3 FiLM Does Not Address Competing Gradients

PCGrad (Yu et al., 2020) and MGDA (Sener & Koltun, 2018) demonstrate that gradient interference is a fundamental optimization challenge in MTL. FiLM modulates intermediate features but does **nothing** to resolve conflicting gradient directions during training. A shared backbone with FiLM layers still receives contradictory gradient signals from different tasks -- the FiLM parameters just learn to mask the interference, not resolve it. When task gradients strongly conflict (e.g., segmentation vs. surface normals at boundaries), FiLM layers cannot reconcile them.

---

## 2. Task-Specific BN Failure Cases

### 2.1 The TSBN Paper's Own Data Shows TSBN HURTS Some Tasks

From **Table 1** of the TSBN paper (arXiv:2512.20420) -- the primary source Agent 5 relies upon:

| Method | Segmentation (mIoU) | Depth (RMSE) | Normal (mErr) | Delta-m |
|--------|-------|-------|-------|-------|
| HPS (fully shared baseline) | **53.93** | 0.3825 | 23.57 | 0.00 |
| TSBN (Agent 5's #1 recommendation) | 53.44 (-0.49) | 0.3812 | 23.01 | +1.04 |
| TS-sigma-BN (Agent 5's variant) | 53.78 (-0.15) | 0.3735 | **22.31** | **+2.48** |

**On NYUv2, TSBN makes segmentation WORSE than the fully shared baseline** (53.93 -> 53.44 mIoU). The overall positive delta-m comes entirely from depth estimation and surface normal improvements compensating for the segmentation degradation. This is **not** "essentially matching ST performance" -- it is a trade-off.

### 2.2 Marginal Gains on Cityscapes

On **Cityscapes** (3 tasks):

| Method | Segmentation (mIoU) | Depth (RMSE) | Delta-m |
|--------|-------|-------|-------|
| HPS | 69.81 | 0.0125 | 0.00 |
| TSBN | 69.89 (+0.12) | 0.0124 | +0.38 |
| TS-sigma-BN | 70.17 (+0.36) | 0.0123 | +0.85 |

Segmentation gains: **+0.12% (TSBN) and +0.36% (TS-sigma-BN)**. These are marginal. Meanwhile, MTAN achieves +0.49% segmentation gain and +0.49 delta-m with comparable efficiency. The TSBN paper's own data shows their method is not clearly superior to older approaches.

### 2.3 TSBN Gains Are Brittle to Hyperparameters

From the TSBN paper's own ablation (Section 7.1):

> "For TSBN, moderate multipliers yield small gains, but performance collapses at high rates."

TSBN is **brittle**: its gains are small at moderate learning-rate multipliers, and performance collapses when the multiplier is too high. The paper requires a special "discriminative learning rate" (multiplier of 100x for sigma-BN parameters, 10x for LN) that adds optimization complexity. This differential LR scheme is **not included in Agent 5's "~0.06% parameter overhead"** claim -- the optimization overhead is real (separate LR schedules for BN vs. backbone parameters).

### 2.4 TSBN Requires BatchNorm -- Not Applicable to All Architectures

Agent 5 notes this as a "risk" but understates it: **TSBN fundamentally requires BatchNorm**. For LayerNorm-based architectures (ViT, Swin, GPT-style transformers), TSBN does not apply. The paper's extension to LayerNorm is TS-sigma-LN, which uses 2x per-task LN affine parameters, but:
- LayerNorm has far fewer parameters than BatchNorm per layer (2 per channel vs. 4 per channel with running stats), so the relative parameter overhead is LARGER
- LayerNorm's effect is different from BN (no batch statistics), so the TSBN argument about "mismatched population statistics" does not carry over

**If the backbone uses LayerNorm (common in modern vision transformers), TSBN provides no mechanism for task-specific normalization of batch statistics.**

### 2.5 Soft-Sharing Methods FAIL on Many-Task Regimes

From the TSBN paper itself, regarding the **CelebA** benchmark (40 tasks):

> "Notably, soft parameter sharing methods underperform the STL baseline on CelebA, highlighting their poor scalability to many tasks, whereas TSBN remains robust."

The authors **admit** that soft-sharing methods (CS, MMOE, MTAN, etc.) underperform single-task learning on the 40-task CelebA benchmark. TSBN remains robust, but this is relative to other failing soft-sharing methods, not relative to an oracle. The absolute performance on CelebA is:

| Method | Delta-m (%) |
|--------|-------|
| HPS | -1.69 |
| CS | -3.86 |
| MMOE | -12.78 |
| MTAN | -1.52 |
| TSBN | -1.52 |
| TS-sigma-BN | +1.81 |

Even TS-sigma-BN achieves only +1.81% delta-m on 40 tasks. The gains are modest for the complexity introduced.

---

## 3. Adapter-Based MTL Failure Cases

### 3.1 Adapters Can UNDERPERFORM Single-Task Learning

From **Table 4** of the TSBN paper (PascalContext with pretrained ViT-S):

| Method | Delta-m (%) | Trainable Params (M) |
|--------|-------|-------|
| STL (separate models) | 0.00 | 112.62 |
| **Standard Adapter** | **-2.71** | 11.24 |
| VL-Adapter | **-1.83** | 4.74 |
| MTLoRA (r=16) | +1.35 | 4.95 |
| **LoRA (r=8)** | **-0.54** | 3.08 |
| **BitFit** | **-4.60** | 2.85 |
| **Compacter** | **-2.17** | 2.87 |
| **VPT-deep** | **-10.85** | 3.43 |
| Polyhistor | +2.34 | 8.96 |
| TSBN | +0.91 | 3.08 |
| TS-sigma-BN | +1.63 | 4.25 |

**Standard adapters, LoRA, BitFit, Compacter, and VPT-deep all produce NEGATIVE gains over single-task learning.** Adapting just 3% of backbone parameters is not sufficient to resolve task interference. This directly contradicts Agent 5's secondary recommendation of TCA (Task-Conditional Adapters) at "3-8% parameter overhead."

### 3.2 Adapter Gains Are Task-Dependent

The TSBN paper's Table 4 also reveals that adapters (LoRA, standard)
sharply degrade parts segmentation (Parts mIoU drops from 67.21 STL to 57.38 for Adapter, 58.99 for MTLoRA). This suggests that adapter-based methods systematically **sacrifice performance on fine-grained tasks** for modest gains on simpler tasks.

### 3.3 Adapters Add Latency Despite Small Parameter Count

Agent 5 claims adapters "add no inference latency" (for LoRA) or have "negligible overhead." This is misleading:

- **LoRA** can be merged into weights, but **TCA and standard adapters cannot** -- they are parallel pathways that require separate computation during inference.
- Even for merged LoRA, the rank decomposition means each forward pass does matrix multiplies of shape (d x r) and (r x d) instead of the single (d x d) multiply, which is actually **slower per parameter** for small batch sizes (where compute is bandwidth-bound).
- The CoDA paper uses sparse conditional computation specifically to address the latency problem that standard adapters create. If adapters had "no inference latency," CoDA would not need to exist.

### 3.4 TSBN Paper Results Show Most Methods Fail

Of the 13 methods compared in Table 4 (PascalContext, pretrained ViT-S):

| Outcome | Count | Methods |
|---------|-------|---------|
| **Negative delta-m** | **7** | Adapter, VL-Adapter, LoRA, BitFit, Compacter, VPT-deep, TSBN base |
| **Positive but <2%** | **3** | TSBN (+0.91), TS-sigma-BN (+1.63), MTLoRA(r=16) (+1.35) |
| **Positive >2%** | 2 | HyperFormer (+2.23), Polyhistor (+2.34) |

54% of methods tested produced **negative results**. The claim that "task-conditional modulation works" is only true for a minority of methods, and even the best methods achieve only modest improvements.

---

## 4. Task-Conditional Modulation Overfitting

### 4.1 Overfitting with Few Tasks

When there are few tasks (2-4), task-conditional modulation has limited data to learn meaningful conditioning vectors. Consider the evidence:

- **TSBN requires differential learning rates specifically to prevent the BN parameters from collapsing** -- the authors note that "lower learning rates result in minimal divergence, with alpha=1 being excluded as it shows almost no differentiation between tasks." This means the task-conditional signal CAN disappear if not carefully tuned.
- **Standley et al. (2020)**: "Which Tasks Should Be Learned Together in Multi-Task Learning" demonstrates that the optimal grouping of tasks depends on their relatedness. When related tasks are forced to share a backbone with conditioning, the conditioning vectors can learn trivial distinctions (e.g., always-on for one task, always-off for another) that add parameters without meaningful modulation.
- **The Sener & Koltun (2018) multi-objective formulation** demonstrates that the Pareto front of multi-task solutions is non-trivial. Task-conditional modulation only adjusts the representation space; it cannot resolve the fundamental optimization conflict when tasks have opposing gradient directions.

### 4.2 Gradient Interference Persists

The PCGrad paper (Yu et al., 2020) identifies three conditions that cause detrimental gradient interference in MTL:
1. **Conflicting gradient directions**: tasks pull the shared parameters in opposite directions
2. **Dominant gradient magnitudes**: one task's loss dominates the update
3. **Covariate shift**: task-specific input distributions cause the backbone to oscillate

Task-conditional modulation addresses NONE of these. TSBN adjusts BN statistics (condition 3 partially), but FiLM, adapters, and gating methods only modulate features after the shared backbone has already been updated with conflicting gradients.

The GradNorm paper (Chen et al., 2018) and uncertainty weighting (Kendall et al., 2018) address gradient magnitude imbalance, but these are **orthogonal** to task-conditional modulation. Agent 5 does not mention gradient balancing at all, yet it is a necessary complement to any architectural method.

### 4.3 Adding Parameters Can Make Things Worse

The TSBN paper's Table 4 shows a clear pattern: **methods that add too few trainable parameters underperform**, but **methods that add too many also underperform**:

- ViT-B full fine-tuning (72.77M params): +2.64% delta-m (best)
- BitFit (2.85M params, 1% of backbone): **-4.60% delta-m** (catastrophic)
- VPT-deep (3.43M params): **-10.85% delta-m** (worst result)
- Standard adapter (11.24M params): **-2.71% delta-m**

More parameters does not guarantee better results, and fewer parameters does not guarantee worse results. The relationship between task-conditional capacity and performance is **non-monotonic** and **task-dependent**. Agent 5's linear ranking of methods by parameter overhead is misleading.

### 4.4 Interpretability Claims Overstated

Agent 5 and the TSBN paper claim that "learned gates provide interpretable insights into capacity allocation." However:

- The paper's own analysis (Appendix D) shows that the learned gating distributions depend heavily on the learning-rate multiplier. Different hyperparameters produce different "interpretable" patterns, undermining the claim of stable task relationship discovery.
- Gate-based interpretability has been criticized in the broader literature (e.g., Jain & Wallace, 2019 on attention interpretability) -- sparsity does not equal interpretability.

---

## 5. Recommendations: What Agent 5 Missed

### 5.1 Gradient Balancing Is Essential

Agent 5 recommends only architectural methods (TSBN, adapters). It omits gradient-based methods (PCGrad, CAGrad, Nash-MTL) that directly address gradient interference. The best results in the TSBN paper (Polyhistor: +2.34%, HyperFormer: +2.23%) combine architecture AND optimization modifications. Pure architectural fixes leave the core optimization problem unsolved.

### 5.2 Task Competition Dictates Whether MTL Helps

The Standley et al. (2020) framework recommends **not** forcing all tasks through one backbone when tasks compete. Task-conditional modulation cannot fix fundamentally incompatible task objectives. The decision of **which tasks to group** should precede the decision of **how to modulate**.

### 5.3 No Single Method Dominates Across All Benchmarks

| Method | NYUv2 Delta | Cityscapes Delta | CelebA Delta | PascalContext Delta |
|--------|-------|-------|-------|-------|
| TSBN | +1.04 | +0.38 | -1.52 | +0.91 |
| TS-sigma-BN | **+2.48** | +0.85 | **+1.81** | +1.63 |
| MTAN | +1.55 | +0.49 | -1.52 | N/A |
| Adapter | N/A | N/A | N/A | **-2.71** |
| LoRA | N/A | N/A | N/A | -0.54 |
| Polyhistor | N/A | N/A | N/A | +2.34 |

No method dominates. TS-sigma-BN wins on NYUv2 but is marginal on Cityscapes and modest on PascalContext. The choice of method should be benchmark-specific, not one-size-fits-all.

### 5.4 The Hybrid Recommendation Is Untested

Agent 5's primary recommendation is to combine TSBN + TCA: "this hybrid should close >99% of the MTL-to-ST gap at under 5% parameter overhead." There is **zero published evidence** that this combination works. The TSBN paper does not test this combination. The TCA paper does not use TSBN. This is an untested hypothesis presented as a concrete recommendation.

---

## 6. Summary of Failure Cases

| Method | Failure Case | Evidence Source |
|--------|-------------|-----------------|
| TSBN | Hurts segmentation on NYUv2 (53.93 -> 53.44) | TSBN paper Table 1 |
| TSBN | Only +0.12% segmentation gain on Cityscapes | TSBN paper Table 1 |
| TSBN | Requires special LR scheme; performance collapses without it | TSBN paper Section 7.1 |
| TSBN | Not applicable to LayerNorm architectures | TSBN paper (explicit) |
| TS-sigma-BN | Hurts segmentation on NYUv2 (53.93 -> 53.78) | TSBN paper Table 1 |
| Adapter | -2.71% delta-m (WORSE than STL) | TSBN paper Table 4 |
| LoRA | -0.54% delta-m (WORSE than STL) | TSBN paper Table 4 |
| BitFit | -4.60% delta-m (catastrophic) | TSBN paper Table 4 |
| VPT-deep | -10.85% delta-m (catastrophic) | TSBN paper Table 4 |
| FiLM | Logical inconsistency; not validated on MTL benchmarks | FiLM-Ensemble paper |
| Soft-sharing | All methods underperform STL on CelebA (40 tasks) | TSBN paper Section 6.1 |
| Task-conditional | Cannot resolve gradient interference from competing tasks | PCGrad, Sener & Koltun |

---

## References

1. Perez et al. "FiLM: Visual Reasoning with a General Conditioning Layer." NeurIPS 2018. arXiv:1709.07871
2. Turkoglu et al. "FiLM-Ensemble: Probabilistic Deep Learning via Feature-wise Linear Modulation." NeurIPS 2022. arXiv:2206.00050
3. [TSBN] "Simplifying Multi-Task Architectures Through Task-Specific Normalization." arXiv:2512.20420, 2024.
4. Sener & Koltun. "Multi-Task Learning as Multi-Objective Optimization." NeurIPS 2018. arXiv:1810.04650
5. Yu et al. "Gradient Surgery for Multi-Task Learning." (PCGrad) NeurIPS 2020. arXiv:2001.06782
6. Chen et al. "GradNorm: Gradient Normalization for Adaptive Multi-Task Loss Balancing." CVPR 2018. arXiv:1711.02257
7. Kendall et al. "Multi-Task Learning Using Uncertainty to Weigh Losses." CVPR 2018. arXiv:1705.07115
8. Standley et al. "Which Tasks Should Be Learned Together in Multi-Task Learning?" ICML 2020. arXiv:1905.07553
9. Wallingford et al. "Task Adaptive Parameter Sharing for Multi-Task Learning." CVPR 2022. arXiv:2201.12999
10. Jiang et al. "Task-Conditional Adapter for Multi-Task Dense Prediction." ACM MM 2024.
11. Lei et al. "Conditional Adapters: Parameter-efficient Transfer Learning with Fast Inference." (CoDA) NeurIPS 2023. arXiv:2304.08268
12. Hu et al. "LoRA: Low-Rank Adaptation of Large Language Models." ICLR 2022. arXiv:2106.09685
13. Vandenhende et al. "Multi-Task Learning for Dense Prediction Tasks: A Survey." TPAMI 2021. arXiv:2004.13379
