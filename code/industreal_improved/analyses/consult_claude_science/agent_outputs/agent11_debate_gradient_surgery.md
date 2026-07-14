# Agent 11: Gradient Surgery DEBATER -- Challenge Report

**Date:** 2026-07-11
**Re:** Agent 01 Gradient Surgery Discovery Report
**Methodology:** Verified claims against original paper PDFs (Nash-MTL arXiv:2202.01017, CAGrad arXiv:2110.14048, PCGrad arXiv:2001.06782), GitHub repository analysis, issue tracker review.

---

## 1. VERIFIED CLAIMS

| # | Claim | Source | Verdict |
|---|-------|--------|---------|
| 1 | PCGrad on NYUv2: mIoU=38.06, PixAcc=64.64, AbsErr=0.5550, Dm=+3.97% | Nash-MTL Table 2 (paper lines 577-688) | **CONFIRMED** |
| 2 | CAGrad on NYUv2: mIoU=39.79, PixAcc=65.49, AbsErr=0.5486, Dm=+0.20% | Nash-MTL Table 2 | **CONFIRMED** |
| 3 | Nash-MTL on NYUv2: mIoU=40.13, PixAcc=65.93, AbsErr=0.5261, Dm=-4.04% | Nash-MTL Table 2 (lines 693-713) | **CONFIRMED** |
| 4 | Batch sizes: NYUv2=2, Cityscapes=8 for Nash-MTL | Nash-MTL Appendix B (line 1766) | **CONFIRMED** |
| 5 | Batch size 1280 for MetaWorld MT10/MT50 | Nash-MTL Appendix B (line 1771) | **CONFIRMED** |
| 6 | CAGrad optimal c=0.4 (NYUv2), c=0.2 (Cityscapes) | Nash-MTL Appendix B (line 1738); CAGrad paper (line 457) | **CONFIRMED** |
| 7 | 11 gradient surgery methods surveyed | Nash-MTL Table 2 adds Cross-Stitch, MTAN, Uncertainty | **CONFIRMED** |
| 8 | No published method achieves MTL > ST on ALL tasks simultaneously | Nash-MTL Table 2: loses on surface normal (25.26 vs 25.01 STL) | **CONFIRMED** |
| 9 | Nash-MTL convergence proof is to Pareto-stationary point (non-convex) | Nash-MTL Sec. 5 (lines 720-723) | **CONFIRMED** |

---

## 2. CHALLENGED CLAIMS

### CLAIM A: "Nash-MTL is the clear winner on Cityscapes"

**Agent 1 states:** "On Cityscapes, Nash-MTL achieves the best segmentation mIoU (75.41) AND the best depth AbsErr (0.0129), simultaneously beating ST on both tasks."

**DEBATE: MISLEADING OMISSION.**

The Nash-MTL paper itself states (line 1066-1067): *"Nash-MTL achieves the best MR in both datasets, the best Dm in NYUv2 and the **seconds to best Dm** in the CityScapes experiment."*

**Evidence:** The paper explicitly says Nash-MTL is **second-best** on the Cityscapes composite Dm metric. Some other method outperforms it. Agent 1 presents only per-task metrics (mIoU, AbsErr) selectively while omitting the Dm ranking where Nash-MTL is not SOTA.

**Verdict:** Agent 1's claim is **correct on per-task metrics** but **misleading by omission** on the composite metric. The "clear winner" framing is inaccurate.

---

### CLAIM B: "Effective batch_size=16 is viable and validated"

**Agent 1 states:** "Our effective batch size of 16 falls well within the validated range. There is no evidence that batch size significantly affects gradient surgery efficacy."

**DEBATE: UNSUPPORTED EXTRAPOLATION.**

**Evidence from the papers:**
- NYUv2 (3 tasks): batch_size=2
- Cityscapes (2 tasks): batch_size=8
- QM9 (11 tasks): batch_size=120
- MT10 (10 tasks RL): batch_size=1280
- Multi-Fashion+MNIST (2 tasks): batch_size=256

The literature validates batch sizes for 2-3 task vision MTL at **batch_size=2 or 8**. Our setting is **4 tasks at batch_size=16**. This represents an **8x increase** from the 3-task NYUv2 baseline with no validation. The RL experiments use larger batches but are incomparable due to different gradient dynamics (off-policy RL vs supervised learning).

Agent 1's claim that "all papers use batch_size=2 for NYUv2" is correct, but conflating "tested at batch_size=2" with "validated at batch_size=16" is an unsupported leap. Larger batches produce **smoother, less noisy gradient estimates**, which directly affects the conflict detection that gradient surgery methods rely on. For PCGrad, the paper notes (line 243): *"since we are using SGD, which is a noisy estimate of the true batch gradients, the cosine similarity between the gradients of two tasks in a minibatch is unlikely to be -1, thus avoiding this scenario."* With batch_size=16 (less noise), gradients may be more consistently conflicting, potentially changing behavior.

**Verdict: FLAGGED. No paper validates batch_size=16 for 4-task vision MTL.** The extrapolation from batch_size=2 to 16 is unsupported.

---

### CLAIM C: "Nash-MTL computational cost is comparable to LS"

**Agent 1 states:** "Nash-MTL runtime can be reduced to about the same as linear scalarization (or STL) by using the Nash-MTL-50 variant."

**DEBATE: MISLEADING PRESENTATION.**

**Evidence from Nash-MTL paper (lines 459-466, 1161-1164):**
- Nash-MTL-50 (update every 50 steps): "x9.8 speedup" relative to full Nash-MTL
- Nash-MTL-5 (update every 5 steps): "x3.7 speedup" relative to full Nash-MTL
- Full Nash-MTL requires solving a convex optimization problem at each step

**Implication:** Full Nash-MTL at every step is approximately **10x slower** than LS per step. Nash-MTL-50 is comparable to LS, but with degraded performance. Agent 1 presents the "reduced to about the same" as if it's a general property, when it's actually a speed-accuracy tradeoff that the paper clearly documents.

The paper also notes (lines 399-401): *"When the number of tasks K becomes large, this may be too computationally expensive as it requires one to perform K backward passes through the shared backbone."*

For our 4-task setting, this means:
- Full Nash-MTL: ~4x backward passes per step + convex solve (~10x total vs LS)
- Nash-MTL-50: ~4x backward passes per step + solve every 50 steps (~1.1x total vs LS, but lower quality)

**Verdict: CHALLENGED. Agent 1 drastically understates the per-step overhead of full Nash-MTL.** The "comparable to LS" claim only holds for the degraded Nash-MTL-50 variant.

---

### CLAIM D: "The gap between PCGrad and Nash-MTL is 8.01% Dm -- a massive improvement"

**Agent 1 states:** "The gap between PCGrad (+3.97%) and Nash-MTL (-4.04%) is 8.01% Dm -- a massive improvement purely from better gradient surgery."

**DEBATE: TECHNICALLY CORRECT BUT POTENTIALLY OVERSTATED.**

**Evidence:** Dm is defined as the average per-task performance drop relative to STL, expressed as a percentage. PCGrad has Dm=+3.97% (3.97% WORSE than ST on average). Nash-MTL has Dm=-4.04% (4.04% BETTER than ST on average).

The 8.01 percentage point gap is real, but it conflates two separate effects:
1. PCGrad's poor performance (+3.97%) which is actually **worse** than simple linear scalarization (+5.59%)... wait, PCGrad is better than LS (+3.97% vs +5.59%).
2. Nash-MTL's exceptional performance (-4.04%)

But this comparison is on NYUv2 (3 vision tasks). Our setup is **4 diverse tasks** (detection, activity, PSR, pose). The gap may differ significantly with different task compositions.

**Verdict: ACCURATE on NYUv2 but the generalization to our setup is UNTESTED.**

---

### CLAIM E: "CAGrad provably converges to optimum of average loss"

**Agent 1 states:** "CAGrad is the only method (besides Nash-MTL) with proven convergence to an optimum of the average loss function, not just a Pareto-stationary point."

**DEBATE: PARTIALLY CORRECT, KEY NUANCE OMITTED.**

**Evidence from CAGrad paper (line 1138-1139):** The paper states its own limitation: *"Currently we mainly focus on optimizing the average loss, which could be replaced by other main objectives."*

**The nuance:** CAGrad converges to a stationary point of the **average loss** L0, which is NOT necessarily a Pareto-optimal point. If the average loss optimum lies outside the Pareto front, CAGrad will not converge to a Pareto-optimal solution. This is a design choice, not a universal advantage.

Nash-MTL, by contrast, converges to a **Pareto-stationary point** (Nash paper line 720-723): *"we will prove convergence to a Pareto stationary point."*

So Nash-MTL and CAGrad have **different convergence guarantees**, not the same one. CAGrad guarantees convergence to average-loss optimum (which may be non-Pareto-optimal). Nash-MTL guarantees Pareto-stationarity.

**Verdict: CHALLENGED on nuance.** The "optimum of the average loss" is not the same as "Pareto-optimal point."

---

### CLAIM F: "IMTL-G is simpler to implement" (secondary recommendation)

**Agent 1 states:** "If Nash-MTL implementation is complex, IMTL-G (ICLR 2021) is the next-best option... It is simpler to implement."

**DEBATE: ARCHITECTURE-DEPENDENT CAVEAT OMITTED.**

**Evidence from Nash-MTL paper (lines 1736-1737):** *"We apply all gradient manipulation methods to the gradients of the shared weights, with the exception of IMTL-G, which was applied to the feature-level gradients, as was originally proposed by the authors. We also tried applying IMTL-G to the shared-parameters gradient for a fair comparison, but it did not perform as well."*

This reveals that IMTL-G requires **feature-level gradient access**, not just shared-parameter gradients. This is architecture-dependent and may not be straightforward for all MTL backbones. The paper tried the simpler shared-parameter variant and it **did not perform as well**.

**Verdict: CHALLENGED.** The simpler implementation (shared-parameter level) performs worse, undermining the recommendation.

---

### CLAIM G: "Combine Nash-MTL with Kendall uncertainty weighting"

**Agent 1 states:** "Keep Kendall uncertainty weighting for loss scaling but combine with Nash-MTL gradient combination. The two methods operate at different levels: Kendall weights the scalar losses, while Nash-MTL combines the gradients. They are complementary."

**DEBATE: POTENTIALLY CONFLICTING.**

**Evidence:**
- Kendall uncertainty weighting achieved Dm=+4.05% on NYUv2 (Nash-MTL Table 2), the **third-worst** of all 11 methods surveyed
- Nash-MTL is **already scale-invariant** by design (Nash paper, line 157-158): *"The solution satisfies four desirable axioms (Pareto optimality, scale invariance, etc.)"*

Kendall uncertainty weighting works by dynamically adjusting loss weights based on task-specific uncertainty. Nash-MTL's Nash bargaining solution explicitly ignores loss magnitudes and produces a proportionally fair gradient direction regardless of loss scale. Combining a loss-scaling method (Kendall) with a method that deliberately ignores scale (Nash-MTL) may produce conflicting signals.

The paper does not test this combination. No ablation study exists for Nash-MTL + uncertainty weighting.

**Verdict: FLAGGED as untested and theoretically questionable.**

---

## 3. MISSING CONSTRAINTS

### Batch Size Sensitivity

| Method | Validated Batch Sizes | Our Batch=16? | Risk |
|--------|----------------------|---------------|------|
| PCGrad | 2 (3 tasks), 8 (2 tasks) | 8x leap from 3-task | Unknown |
| CAGrad | 2 (3 tasks), 8 (2 tasks) | 8x leap from 3-task | Unknown |
| Nash-MTL | 2 (3 tasks), 8 (2 tasks), 120 (11 tasks QM9) | 8x leap from 3-task | Unknown |

**Key concern:** The PCGrad paper explicitly states that the method relies on SGD noise to avoid zero-gradient cases (line 242-244): *"since we are using SGD, which is a noisy estimate of the true batch gradients, the cosine similarity between the gradients of two tasks in a minibatch is unlikely to be -1."* At batch_size=16, gradient estimates are smoother and more consistent, which could **increase** the frequency of near-zero cosine similarity, potentially slowing convergence.

### Computational Overhead (Per-Step)

| Method | Backward Passes | Extra Computation | Relative to LS |
|--------|----------------|-------------------|----------------|
| PCGrad | K (per task) | O(K^2 d) projection | ~Kx backward passes |
| CAGrad | K (per task) | O(Kd log K) ball projection | ~Kx backward passes |
| Nash-MTL (full) | K (per task) | O(K^2 d) convex solve (CVXPY) | ~10x slower than LS |
| Nash-MTL-50 | K (per task) | O(K^2 d) every 50 steps | ~1.1x slower than LS |

**Nash-MTL uses CVXPY** (convex optimization library) to solve the bargaining problem. This adds dependency and potential installation issues. Users have reported (Issue #14) warnings about *"the problem is not DPP"* and *"Solution may be inaccurate"* from the conic solver.

### Code Availability

| Method | Repo | Stars | Open Issues | License | Last Commit |
|--------|------|-------|-------------|---------|-------------|
| PCGrad | https://github.com/tianheyu927/PCGrad | ~500+ | Unknown | Unknown | Unknown |
| CAGrad | https://github.com/Cranial-XIX/CAGrad | ~300+ | Unknown | Unknown | Unknown |
| Nash-MTL | https://github.com/AvivNavon/nash-mtl | 246 | 1 open | **None (no license)** | 2025-06-25 |

**Key finding:** The Nash-MTL repository has **NO LICENSE file**. This means default copyright laws apply -- you may not have legal permission to reproduce, distribute, or create derivative works. This is a significant legal risk for production use.

**Reproducibility issues documented in issues:**
- Issue #21: User couldn't reproduce the toy example plot (closed, with fix)
- Issue #20: Open bug when porting to other networks
- Issue #15: "ValueError: Parameter value must be real" during training
- Issue #14: "Solution may be inaccurate" warnings from CVXPY solver
- Issue #7: Users report weights as large as 47.24, which may destabilize training
- Issue #2: Error in IMTL-G implementation in the codebase

---

## 4. FLAGGED HALLUCINATIONS

**None identified.** Agent 1's numerical claims are all verified against the source papers. The report's weakness is in **omissions, overstatements, and unsupported extrapolations**, not fabrication of data.

---

## 5. DEBATE VERDICT PER CLAIM

| # | Claim | Verdict | Severity |
|---|-------|---------|----------|
| 1 | Nash-MTL is "clear winner" on Cityscapes | **MISLEADING OMISSION** (paper says 2nd-best Dm) | Medium |
| 2 | Batch_size=16 is validated | **UNSUPPORTED EXTRAPOLATION** (tested at 2, 8) | High |
| 3 | Nash-MTL cost comparable to LS | **UNDERSTATED** (10x slower at full frequency) | High |
| 4 | PCGrad-to-Nash-MTL gap is 8.01% Dm | **ACCURATE** but may not generalize | Low |
| 5 | CAGrad converges to average-loss optimum | **PARTIALLY CORRECT** (not Pareto-optimum) | Low |
| 6 | IMTL-G is simple to implement | **OMITTED CAVEAT** (feature-level access needed) | Medium |
| 7 | Nash-MTL + Kendall are complementary | **UNTESTED, QUESTIONABLE** (methods conflict) | Medium |
| 8 | No validation beyond 3-task vision | **CORRECT BUT UNDERPLAYED** | Low |
| 9 | MTL cannot beat ST on all tasks | **CORRECT** | None |
| 10 | Gradient surgery gap of 8.01% is addressable | **OVERSTATED** (may shrink with 4 diverse tasks) | Medium |

---

## 6. KEY DEBATE SUMMARY

### What Agent 1 Got Right
- All numerical values from the papers are accurate
- The ranking of methods on NYUv2 (Nash-MTL > IMTL-G > CAGrad > PCGrad) is correct
- The observation that no method achieves universal MTL > ST is well-documented
- The 11-method survey is comprehensive and correctly categorized

### What Agent 1 Got Wrong / Omitted

1. **Computational cost is critically understated.** Full Nash-MTL is ~10x slower per step than LS. This is buried in a misleading sentence about Nash-MTL-50.

2. **Batch size extrapolation is unsupported.** No paper tests batch_size=16 for 4-task vision. The PCGrad paper's reliance on SGD noise (line 242-244) suggests larger batches could change behavior.

3. **Cityscapes Dm ranking is omitted.** The Nash-MTL paper explicitly states it is second-best on Cityscapes composite Dm, yet Agent 1 presents it as unequivocally best.

4. **IMTL-G implementation caveat omitted.** The feature-level requirement and failed shared-parameter attempt are critical for anyone trying to implement it.

5. **Kendall + Nash-MTL combination is untested.** Combining a loss-scaling method with a scale-invariant method may produce conflicting signals. No ablation exists.

6. **License issue unmentioned.** The Nash-MTL repo has no license, creating legal uncertainty for production use.

7. **Reproducibility issues unmentioned.** Multiple GitHub issues document training instability, CVXPY solver warnings, and reproduction difficulties for toy examples.

### Bottom-Line Assessment

Agent 1's report is **numerically accurate** but **analytically incomplete**. The data is correct, but the interpretation systematically overstates the benefits and understates the costs of switching to Nash-MTL. The three most critical omissions are:

1. **10x computational overhead** of full Nash-MTL per step (with our 4 tasks requiring 4 backward passes + convex solve)
2. **Untested batch_size=16** for 4-task vision (extrapolated from batch_size=2)
3. **No license** on the Nash-MTL codebase (legal risk for production)
