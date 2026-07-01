# AAIML 2027 — 20 IEEE Reviewer Synthesis: Complete Audit & Restructure Plan

**Date:** 2026-07-01 · **Deadline:** October 10, 2026 (101 days)

## The Core Problem

The paper has a **split identity**: it is simultaneously an MTL training-pathologies paper and a blockchain/pilot deployment paper. For AAIML (broad AI/ML conference), the training pathologies are the relevant contribution. The blockchain/pilot content consumes ~25% of the paper but belongs at a systems/HCI venue. **The paper must choose one identity.** Recommendation: lead with "Three Training Pathologies in Multi-Task Learning" and reduce deployment to 0.5 pages.

## Critical Issues (Must Fix Before Submission)

| # | Issue | Source | Blocker? |
|---|-------|--------|----------|
| 1 | **All 28 results are \inprogress placeholders** — Table 2, every ablation, all figures, compute estimates, GPU-hours. Paper has zero completed empirical results. | Reviewer: figures-format, results | **YES** |
| 2 | **Missing IRB approval** — N=20 human subjects, surveys, interviews, video monitoring, but no IRB protocol number or ethics exemption stated. Desk-reject risk at IEEE. | Reviewer: ethics-safety | **YES** |
| 3 | **Paper has two identities** — training pathologies paper + blockchain/pilot deployment paper. Neither is developed deeply enough. | Reviewer: paper-structure, weaknesses | **YES** |
| 4 | **Pathology 1 uses R=12 (wrong)** — Eq 1 should use R=58 training recordings, giving P=1.72% not 8.3%. The conclusion strengthens (98.3% non-temporal), but the specific number is incorrect. | Reviewer: pathology1 | **YES** |
| 5 | **Pathology 2 claims "majority-class >98% correct" but balanced sampler makes this impossible** — With ACT_SAMPLER_MODE='balanced', every class appears equally. L_act ≈ log(47) ≈ 3.85 at init. The pathology mechanism described cannot occur with the current sampler. Fix description also doesn't match code (bounds [-0.5,2.0], not [-2,2]). | Reviewer: pathology2 | **YES** |
| 6 | **Pathology 3 survey has no methodology** — No search date, query, inclusion criteria, or list of 20 repos. The 70% claim is unverifiable. | Reviewer: pathology3, citations | **YES** |
| 7 | **Code repo URL returns 404** — github.com/bashara-aina/popw does not exist. Foundational reproducibility requirement unmet. | Reviewer: reproducibility | **YES** |
| 8 | **Test set frame count is wrong** — Paper says "test set contains 5,595 frames" but this is actually train+val combined (3,667+1,928). Actual test set count unknown. | Reviewer: reproducibility | **YES** |
| 9 | **Seeds claimed (73, 128) don't match code** — Code documents seeds 42, 123, 7. The paper and code must agree. | Reviewer: reproducibility | **NO** |
| 10 | **Body pose has no real annotations** — Paper claims "5 tasks" but body pose uses pseudo-keypoints from detection boxes. Only 4 tasks have real supervision. | Reviewer: architecture, weaknesses | **NO** |

## Critical Omissions (Missing from Paper Entirely)

| Missing Content | Source | Estimated Lines |
|----------------|--------|----------------|
| **OneCycleLR scheduler bug** — steps_per_epoch mismatch, LR stayed in warmup forever. 4th pathology or merge into Pathology 1. | Reviewer: fixes-integration | 15-20 |
| **DET_GT_FRAME_FRACTION** — 3-layer sampling design, most impactful sampling fix. | Reviewer: missing-content | 15-20 |
| **Weight decay 1e-3 + grad clip 5.0** — Critical hyperparameters, were 50-500x off from standard. | Reviewer: hyperparameters | 5-8 |
| **Smooth loss caps** — Differentiable caps on all 5 losses, gradient never zeroed. | Reviewer: missing-content | 10-15 |
| **NaN guard architecture** — Triple-layer NaN defense across all losses. | Reviewer: missing-content | 8-12 |
| **Per-class sampling diagnostic** — Monitoring tool for Pathology 2's input conditions. | Reviewer: missing-content | 5-8 |
| **Activity is 47 hybrid groups, not 74 classes** — Fundamental task definition change. | Reviewer: missing-content | 12-18 |
| **Limitations section** — 10+ specific limitations identified across all reviewers. | Reviewer: limitations | 20-30 |
| **Hyperparameter training config table** — 15 critical hyperparameters. | Reviewer: hyperparameters | Full table |
| **EgoPack comparison** — Closest competitor, 1-sentence dismissal is insufficient. | Reviewer: competitor-positioning | 5-8 |
| **Diversity monitor** — The actual detection tool for activity collapse. | Reviewer: missing-content | 8-12 |
| **18 infrastructure fixes catalog** — 4 tiers by impact, with ablation. | Reviewer: fixes-integration | 20-30 |
| **Survey supplementary** — List of 20 repos, methodology, classification criteria. | Reviewer: pathology3 | 10-15 |
| **IRB statement** — Protocol number or exemption category. | Reviewer: ethics-safety | 2-3 |

## Pathologies: Corrections Required

### Pathology 1: Temporal-Head/Sampler Mismatch
- **Fix R=12 to R=58**: P(same recording) = sum_r(f_r/total)² ≈ 1.72%, not 8.3%. Non-temporal = 98.3%, not 91.7%.
- **Acknowledge the fix side-steps the problem**: Switching to a per-frame MLP eliminates the temporal encoder entirely. A root-cause fix would use sequence-level sampling.
- **Merge in the OneCycleLR scheduler bug**: Same class of infrastructure component interface mismatch. Both involve components designed for different cadences.
- **Merge in the double-remap bug**: Data pipeline interface mismatch.

### Pathology 2: Loss Scale Suppression Under Label Sparsity
- **Remove "majority-class >98%" claim**: Falsified by the balanced sampler. With ~47 groups and equal per-class sampling, every class appears equally.
- **Correct the fix description**: Actual bounds are KENDALL_LOG_VAR_MIN_ACT=-0.5, init s_act=0. NOT [-2,2] with init=-1 as stated.
- **Note balanced sampler prevents the pathology**: The paper's own sampler fix eliminates the root cause. Describe it as a preemptive fix, not an observed failure.
- **Merge DET_GT_FRAME_FRACTION**: At 0.90, activity got only 0.14 frames/class/batch, artificially suppressing activity loss and compounding the Kendall spiral.

### Pathology 3: Gradient Measurement Artifacts
- **Downgrade from "training pathology" to "diagnostic lesson"**: It's about debugging methodology, not training dynamics.
- **Provide survey supplementary**: List of 20 repos, search methodology, classification criteria.
- **Clarify the "10 days" attribution**: The 10 days included debugging Pathologies 1 and 2, not just this measurement error. The precise attribution is not quantifiable.

## Structural Restructure Plan

### Proposed Outline (8 pages)

| Section | Pages | Changes |
|---------|-------|---------|
| Abstract | 0.2 | Focus on 3 pathologies. Blockchain/pilot: 1 sentence. |
| 1. Introduction | 1.0 | Open with concrete anomaly. 4 contributions reordered. |
| 2. Related Work | 0.8 | Add EgoPack comparison. Add ConsMTL. Expand to 1 paragraph per subtopic. |
| 3. System Architecture | 1.0 | Body pose de-emphasized. Add hyperparameter table. Training protocol condensed. |
| 4. Three Training Pathologies | 2.5 | P1 broadened (scheduler merged). P2 corrected. P3 shortened. Add 18-fixes catalog. |
| 5. Empirical Results | 1.0 | Real numbers only. No \inprogress. Sequential baseline measured, not estimated. |
| 6. Deployment + Pilot | 0.5 | Trimmed from 2 pages. One paragraph each. Full details in supplementary. |
| 7. Limitations | 0.5 | NEW — 10+ limitations honest and structured. |
| 8. Conclusion | 0.3 | Mention all 4 contributions. No new claims. |
| References | 0.5 | ~25 refs. Add FPN, EgoPack, Cui 2019, ConsMTL, IEEE 7005. |

### Pre-Flight Checklist (Final 5 Actions Before Submission)

1. **Generate all figures**: At minimum 3 (Pathology 1 mechanism + ablation heatmap, LR schedule trajectory, counterfactual vs actual).
2. **Complete 3-seed training**: RF4-RF10, all metrics, bootstrap CIs.
3. **Factory pilot**: July 15 - August 1 window. Must have SUS, NASA-TLX, Trust scores.
4. **Code release**: Upload to public GitHub, verify URL works.
5. **IRB approval**: Obtain and document protocol number in paper.
