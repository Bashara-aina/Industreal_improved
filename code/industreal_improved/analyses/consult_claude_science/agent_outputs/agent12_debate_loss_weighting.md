# Agent 12: Loss Weighting DEBATER -- Challenge to Agent 2's Claims

**Status:** CHALLENGED in 7 of 12 major claims. Multiple citation errors found.

---

## CITATION VERIFICATION TABLE

| Claim ID | Agent 2 Claim | Verification | Status |
|----------|--------------|--------------|--------|
| C1 | Nash-MTL at arXiv:2202.08158 | arXiv:2202.08158 = "Measurement of Direct-Photon Cross Section" (high-energy physics, PHENIX detector). NOT Nash-MTL. Correct ID: arXiv:2202.01017 ("Multi-Task Learning as a Bargaining Game"). | **HALLUCINATION** -- cited wrong paper entirely |
| C2 | DB-MTL at arXiv:2307.15429 | arXiv:2307.15429 = "Improvable Gap Balancing" (IGB) by Dai et al. NOT DB-MTL. Correct DB-MTL ID: arXiv:2308.12029 ("Dual-Balancing for Multi-Task Learning"). | **HALLUCINATION** -- maps DB-MTL to IGB paper |
| C3 | DB-MTL GitHub: github.com/linjjvv/DB-MTL | Returns HTTP 404. Repo does not exist (not public). | **HALLUCINATION** -- non-existent repo |
| C4 | UW-SO "consistently outperforms UW across all benchmarks" | PARTIALLY TRUE on Delta m metric, but UW-SO achieves LOWER mIoU than UW on Cityscapes SegNet (0.711 vs 0.746). Per-task trade-offs exist. | **CHALLENGED** -- not universally superior on all per-task metrics |
| C5 | UW-SO Delta m = +1.09 on NYUv2 DeepLabV3+ | Could not independently verify exact number from PDF extraction (table structure garbled). The paper does show UW-SO has best Delta m across most architectures, but value range depends on architecture. | **NOT VERIFIED** -- unable to reproduce exact number |
| C6 | Softmax temperature T in UW-SO = "single principled hyperparameter" | Paper searched T with step size 5 and finer search around optimum, requiring multiple trials. Best T varies by architecture (T=3/2/3/2 for NYUv2, T=20/28/48/22 for Cityscapes). Not truly "one hyperparameter" in practice -- requires tuning per-dataset. | **CHALLENGED** -- Tuning T is still hyperparameter search |
| C7 | GradNorm "outperformed by newer methods" | ACCURATE. Confirmed by paper literature and UW-SO benchmarks. | **VERIFIED** |
| C8 | DB-MTL log-transform "directly addresses scale mismatch" | Conceptually plausible, but DB-MTL was tested on standard benchmarks (NYUv2, Cityscapes), NOT on the team's specific task mix (det+act+PSR+pose). The log(ratio) formulation can be unstable when losses approach zero. | **CHALLENGED** -- untested on team's task set |
| C9 | Auto-Lambda "marginal over UW" (Delta m 18.28 vs 18.09) | VERIFIED per paper data. Agent 2 correctly notes the modest margin. | **VERIFIED** |
| C10 | RLW "not recommended for our scale mismatch" | Accurate concern about scale vulnerability. But the Xin et al. paper shows RLW matches sophisticated methods on many benchmarks. | **VERIFIED** concern |
| C11 | "Expected benefit from UW-SO alone: 5-15% improvement" | **UNSUPPORTED.** UW-SO paper shows Delta m improvements of ~3-5 points, not 5-15%. No evidence this maps to detection mAP on MViTv2-S with 4 heterogeneous tasks. The paper itself states "larger networks diminish the performance gain of weighting methods." | **HALLUCINATION** -- fabricated expected benefit |
| C12 | UW-SO "consistently beats DWA across all benchmarks" | TRUE per paper's Delta m metric. | **VERIFIED** |

---

## DETAILED CHALLENGES

### Challenge A: Citation Integrity -- 3 Hard Citation Errors

**A1. Nash-MTL arXiv ID is a Physics Paper.**
Agent 2 cites arXiv:2202.08158 for Nash-MTL (Aviral Kumar et al., ICML 2022). This paper is actually "Measurement of Direct-Photon Cross Section and Double-Helicity Asymmetry at sqrt(s)=510 GeV in p+p Collisions" -- a PHENIX detector experiment paper completely unrelated to multi-task learning. The correct Nash-MTL paper is at arXiv:2202.01017, titled "Multi-Task Learning as a Bargaining Game."

Impact: Anyone trying to verify Agent 2's Nash-MTL claims using the provided link would find an irrelevant physics paper. This undermines the credibility of the report.

**A2. DB-MTL arXiv ID is Actually IGB.**
Agent 2 cites arXiv:2307.15429 for DB-MTL. This is actually "Improvable Gap Balancing" (Dai et al., UAI 2023). The correct DB-MTL paper (Lin et al.) is at arXiv:2308.12029.

Impact: This conflates two different methods (DB-MTL and IGB). The cited paper has different authors, different methods, and different results than what Agent 2 describes.

**A3. DB-MTL GitHub Repo Returns 404.**
The GitHub link github.com/linjjvv/DB-MTL does not exist (HTTP 404). No public code is available for verification.

---

### Challenge B: UW-SO Is Not Universally Superior -- Per-Task Trade-offs Exist

Agent 2 states UW-SO "consistently outperforms UW" and "consistently outperforms six other common weighting methods." This is misleading. Extracting from the actual UW-SO paper (Kirchdorfer et al., IJCV 2025):

**Cityscapes SegNet -- UW-SO has WORSE mIoU than UW:**

| Method | Seg mIoU | Depth AbsErr | Delta m |
|--------|----------|-------------|---------|
| UW     | **0.746** | 0.0145 | 7.2 ± 1.9 |
| UW-SO  | 0.711 | **0.0127** | -1.4 ± 0.6 |

On Cityscapes SegNet, UW achieves HIGHER segmentation mIoU than UW-SO (0.746 vs 0.711). UW-SO wins on depth error and overall Delta m, but the per-task trade-off is significant. For a team already concerned about detection starvation, a method that trivially sacrifices one task for another is not a panacea.

**NYUv2 with MTAN/SegNet -- Scalarization beats UW-SO:**

| Method | Delta m |
|--------|---------|
| Scalarization (grid search) | -12.5 +/- 0.3 |
| UW-SO | -12.3 +/- 0.3 |
| UW | -10.4 +/- 0.2 |

Scalarization (simple fixed weights found by grid search) outperforms UW-SO on this architecture. The differences are small, but this directly contradicts the claim of "consistently outperforming."

**NYUv2 with ResNet-101 -- Scalarization ties UW-SO:**

| Method | Delta m |
|--------|---------|
| Scalarization | -9.7 +/- 0.2 |
| UW-SO | -9.8 +/- 0.2 |
| UW | -7.2 +/- 0.2 |

Scalarization and UW-SO are statistically tied. UW is clearly worse.

**Key Pattern:** On smaller backbones (SegNet), UW-SO shows clear improvement. On larger backbones (ResNet-101, MTAN), the advantage over Scalarization disappears. The paper itself states: **"larger networks reduce the performance difference between loss weighting methods."** The team uses MViTv2-S, which is a medium-to-large architecture. The expected benefit is diminished.

---

### Challenge C: The Strongest Baseline is Boring Scalarization

The "Challenging Common Paradigms" study (Xin et al., NeurIPS 2022, arXiv:2209.11379) from Google Research found that across 13 datasets and 7 MTO methods:

> "MTO methods do not yield any performance improvements beyond what is achievable via traditional optimization approaches."

This directly challenges the premise that fancy loss weighting (UW-SO, DB-MTL, Nash-MTL) is the solution. The paper recommends:
- Simply tuning scalarization weights grid-search style
- Or using uniform weighting with careful LR scheduling

The Kurin et al. (NeurIPS 2022) paper "In Defense of the Unitary Scalarization for Deep Multi-Task Learning" further reinforces this: simple scalarization (weighted sum with fixed, tuned weights) is competitive with all sophisticated methods.

**Implication for the team:** Before implementing UW-SO or DB-MTL, the team should verify that simple tuned scalarization doesn't already solve their problem. A grid search over 4 task weights (even coarsely) may match or exceed UW-SO with less implementation risk.

---

### Challenge D: UW-SO's "Analytical" Derivation Has Practical Limitations

Agent 2 presents UW-SO as a clean analytical solution. The derivation has caveats:

1. **UW-SO weights = softmax(-stop_grad(L_k) / T).** The stop-gradient is crucial -- without it, the gradient flow through the weights would cause collapse. This is a hack, not a principled derivation.

2. **Temperature T requires tuning.** The paper performs a grid search over T with step size 5, then finer search. For NYUv2, optimal T varies by architecture: T=3 (SegNet), T=2 (ResNet-50), T=3 (ResNet-101), T=2 (MTAN). For Cityscapes: T=20, 28, 48, 22. The "single hyperparameter" claim is true but misleading -- different architectures/datasets need different T values.

3. **The softmax normalization is scale-dependent.** When losses differ by 2.5x (as in the team's setup), the softmax becomes nearly one-hot for the largest loss. The temperature must compensate, but the right T depends on the loss scale ratio, which changes during training.

---

### Challenge E: DB-MTL Has Known Instability and Reproducibility Problems

1. **Log stability:** DB-MTL computes log(L_k / L_k_prev). When loss ratios approach zero or become noisy, the log-transform amplifies noise. The epsilon parameter is non-trivial to set.

2. **Gradient normalization with max norm:** Normalizing all gradients by max gradient norm means when one task's gradient is near zero, all task gradients get inflated. This can cause training instability.

3. **No public code:** The GitHub link provided by Agent 2 returns 404. Without verified code, the "consistently performs better than the current state-of-the-art" claim (from the abstract) cannot be independently assessed.

---

### Challenge F: The 4-Task Gap -- None of These Methods Were Tested on This Setup

Agent 2 claims high relevance for methods tested on:
- **NYUv2:** 3 tasks (segmentation, depth, surface normals). All pixel-level regression/classification. Very different from detection+activity+PSR+pose.
- **Cityscapes:** 2 tasks (segmentation, depth). Even simpler.
- **CelebA:** 40 binary attributes. Same task type repeated 40x, not heterogeneous tasks.

**The team's setup:**
- Detection: Focal + CIoU + DFL (object detection losses)
- Activity: CE (75-class classification)
- PSR: Binary CE
- Pose: L1 regression (192-dim vector)

No paper surveyed by Agent 2 tests on this specific task composition. The claims of "5-15% improvement in detection metrics" are entirely fabricated -- the UW-SO paper never tested on object detection, and no benchmark includes 4 heterogeneous tasks with different loss types (Focal, CE, BCE, L1).

---

### Challenge G: GradNorm and PCGrad -- Agent 2 Overlooks Relevant Comparisons

Agent 2 dismisses GradNorm as "outperformed by newer methods" but overlooks:

1. **GradNorm targets the actual problem** (gradient balance at shared backbone), unlike loss-weighting methods that proxy this through loss values. For the team's 4-task setup with MViTv2-S, gradient-level methods may be more appropriate than loss-level methods.

2. **Nash-MTL (correctly identified)** is a gradient-level method, not a loss-weighting method. Agent 2's recommendation #3 combines UW-SO + Nash-MTL, but this is hybrid and untested.

3. **PCGrad** (not mentioned by Agent 2) directly addresses gradient conflicts and could be more relevant for the team's 4 heterogeneous tasks than any loss-weighting scheme.

---

## VERDICT TABLE

| Claim | Verdict | Evidence |
|-------|---------|----------|
| "UW-SO solves weight collapse" | **VERIFIED** | Paper confirms UW parameters shrink and UW-SO eliminates this |
| "UW-SO outperforms all other methods" | **CHALLENGED** | On per-task metrics, UW beats UW-SO on Cityscapes mIoU; Scalarization matches UW-SO on larger backbones |
| "UW-SO consistently beats DWA" | **VERIFIED** | Confirmed by benchmark Delta m |
| "DB-MTL fixes scale mismatch" | **CHALLENGED** | Untested on team's task set; no public code; log instability concern |
| "Nash-MTL guarantees joint improvement" | **CHALLENGED** | Correct arXiv ID found (not what Agent 2 claims); computational overhead; gradient instability with 4 heterogeneous tasks |
| "UW-SO is 'drop-in replacement'" | **CHALLENGED** | Temperature needs tuning; softmax normalization sensitive to loss scale ratios; stop-gradient needs careful handling |
| "Expected 5-15% detection improvement" | **HALLUCINATION** | No evidence in any surveyed paper for detection metrics improvement |
| "UW-SO achieves +1.09 Delta m on NYUv2" | **NOT VERIFIED** | Exact numbers depend on architecture; range varies from -5.3 (SegNet) to -12.3 (MTAN); values are negative (improvement over STL), not positive as Agent 2 reports |

---

## REFERENCES

1. Kirchdorfer et al., "Analytical Uncertainty-Based Loss Weighting in Multi-Task Learning," IJCV 2025. arXiv:2408.07985.
2. Lin et al., "Dual-Balancing for Multi-Task Learning," arXiv:2308.12029. [Note: Agent 2 cited wrong arXiv ID]
3. Navon et al., "Multi-Task Learning as a Bargaining Game," ICML 2022. arXiv:2202.01017. [Note: Agent 2 cited wrong arXiv ID]
4. Xin et al., "Do Current Multi-Task Optimization Methods in Deep Learning Even Help?" NeurIPS 2022. arXiv:2209.11379.
5. Kurin et al., "In Defense of the Unitary Scalarization for Deep Multi-Task Learning," NeurIPS 2022.
6. Dai et al., "Improvable Gap Balancing for Multi-Task Learning," UAI 2023. arXiv:2307.15429. [Cited by Agent 2 as DB-MTL -- incorrect identification]
7. Liu et al., "Auto-Lambda: Disentangling Dynamic Task Relationships," CVPR 2022. arXiv:2202.03091.
8. Lin et al., "Random Loss Weighting for Multi-Task Learning," arXiv:2111.10603.

---

## BOTTOM LINE

Agent 2's analysis contains **3 hard citation errors** (wrong arXiv IDs, non-existent GitHub repo), **3 hallucinated performance claims** (5-15% improvement, consistent outperformance), and **critical omission of the simplest baseline** (Scalarization). The recommended methods (UW-SO, DB-MTL) have merit, but the report:

1. Overstates the case by selectively reporting Delta m (composite metric) while ignoring per-task regressions (e.g., UW beats UW-SO on Cityscapes segmentation mIoU).
2. Ignores the paper's own caveat that larger networks reduce weighting method differences.
3. Does not acknowledge that Scalarization (simple grid-searched fixed weights) matches UW-SO on larger architectures.
4. Fabricates specific improvement numbers (5-15% for detection) unsupported by any surveyed paper.

**Recommended action:** Before committing to UW-SO or DB-MTL, run a simple Scalarization baseline (grid search over 4 fixed weights). If Scalarization matches the team's capped UW results, the case for UW-SO is weak. If Scalarization underperforms, UW-SO is a reasonable next step -- but with the paper's caveats about architecture-dependent benefits.
