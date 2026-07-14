# D8 — Literature Detailed Adversarial Debate

**Phase:** ULTIMATE Consultation V2 — Phase 2 Adversarial Debate
**Date:** 2026-07-14
**Debater:** D8 (continues D3 with deeper literature challenges)

---

## 0a. Update Log (2026-07-14 — Batch 1 Agent Findings)

| Finding | Detail |
|---|---|
| GeoHeadPose bug | model.py:2177-2178 column-swap bug in head pose. Fix via `to_legacy_9dof()`. |
| LDAM-DRW wiring | Already wired — flip `USE_LDAM_DRW=True` at config.py:1098. |
| Distillation stub | train.py:1567 has `pass`. Needs ~50-100 lines. |
| Module sweep | Only `ldam_drw` wired; 10 others NOT_FOUND in train.py. |
| **New gradient norms** | pose=3278.0, act=13.80, det=1.86, psr=0.16. **20,245x ratio** (pose vs psr). V1 312x is **reversed**: now act>psr. |

**Confidence updates from Batch 1:**
- 312x gradient ratio (Section 3.1): resolved — now 20,245x with different dominance (pose >> act > det > psr). 312x claim is **superseded**.

---

## 1. Methodology

D8 continues D3 with deeper investigation into:
- Failure modes of cited methods
- Specific numerical claims cross-verification
- Missing 2025-2026 papers
- Method limitations not in R3

---

## 2. Failure Modes of Cited Methods

### 2.1 Kendall Collapse (CVPR 2018)

**R3 finding:** Kendall et al. CVPR 2018 is our loss weighting method.

**D8 challenge:** Kendall's own paper tested on 2 tasks (depth + segmentation) with similar loss scales. Our 4-task setup has 100x+ loss scale differences (CE ~12, MSE ~0.01). Kendall's original formulation likely fails for us.

**Verification:** Our codebase has `KENDALL_HP_PREC_CAP=True` precisely because vanilla Kendall fails (head pose precision 54.6x). This is a known limitation of the original method, not our innovation.

**Implication:** Cite Kendall as inspiration; cite our caps as the actual contribution.

### 2.2 PCGrad Projection Failure Modes

**R3 finding:** Yu et al. NeurIPS 2020 — projected gradient conflicts.

**D8 challenge:** PCGrad assumes gradient conflicts can be resolved by projection. When:
- Two tasks have IDENTICAL gradient directions, PCGrad does nothing (correct).
- Two tasks have ORTHOGONAL gradients, PCGrad does nothing (correct).
- Two tasks have OPPOSING gradients, PCGrad projects one onto the other's perpendicular component.

**Problem:** PCGrad treats all conflicts equally. In MTL with very different loss magnitudes, the "opposing" task might be the only one providing useful signal.

**Counter-evidence:** Liu et al. (CAGrad, NeurIPS 2021, arxiv 2110.14048) explicitly improves on this by considering loss magnitudes.

**Implication:** PCGrad has known limitations. CAGrad or Nash-MTL may be better. Module exists for CAGrad? Check `src/losses/`.

### 2.3 RotoGrad's Stackelberg Game

**R3 finding:** Javaloy, Valera, ICML 2022 — rotate features to align gradients.

**D8 challenge:** Original paper uses Stackelberg game (rotations learn slowly). Our codebase's simplified version uses SGD on cosine similarity. The simplification might lose the convergence guarantees.

**Verification needed:** Does our simplification actually align gradients as the original Stackelberg version would?

### 2.4 FAMO Initialization Sensitivity

**R3 finding:** Liu et al. CVPR 2023 — fast adaptive MTL optimization.

**D8 challenge:** FAMO is sensitive to initialization. Original paper uses 0.01 init for weight parameters. Our codebase's default init might be different.

**Verification:** Grep `src/losses/famo.py` for default init.

### 2.5 MetaBalance Rescaling Cap

**R3 finding:** He et al. WWW 2022 — gradient magnitude rescaling per parameter.

**D8 challenge:** Original paper has scale cap [0.1, 10.0] for stability. Our codebase has the same cap (per R2 reading). When scale > 10, we cap. But this means tasks with >10x gradient ratio are clipped to 10x.

**Implication:** MetaBalance doesn't fully eliminate gradient starvation; it just bounds it. For our 312x ratio (PSR 3.18 vs activity 0.010), MetaBalance can only compress 312x → 10x (a 31x improvement, but still 31x starved).

---

## 3. Numerical Claim Verification

### 3.1 20,245x Gradient Ratio (was 312x)

**R3 finding (via V1 doc 210):** PSR gradient norm 3.18, activity 0.010. Ratio 312x.

**D8 challenge:** When was this measured? At what epoch? On which task head?

**Batch 1 resolution (2026-07-14):** Re-measured on current V2 codebase. **New norms: pose=3278.0, act=13.80, det=1.86, psr=0.16.** The dominant task is now **pose** (not PSR). Ratio pose/psr = **20,245x**. Additionally, the V1 ordering is reversed: act > psr (was psr > act).

**Interpretation:** The old 312x ratio from V1 (RF4 epoch 5) is no longer representative. The addition of PoseFiLM and HeadPoseFiLM modules dramatically increased pose gradients. Act also overtook PSR, likely due to the FeatureBank+TCN+ViT architecture update.

**Risk:** Updated — 20,245x ratio makes the gradient imbalance problem far worse than V1 reported. Pose completely dominates. Kendall capping and PCGrad may be insufficient.

### 3.2 Frozen ConvNeXt Probe = 0.2169

**R3 finding (via V1 doc 220):** 21.69% top-1 with frozen backbone.

**D8 challenge:** "Frozen" implies no gradient flow to backbone. Did the experiment use:
- Truly frozen (requires_grad=False)?
- Or just no learning rate applied?

**Verification needed:** Read the actual probe code (`scripts/overfit_probe.py`).

### 3.3 Kendall Collapse Reproducibility

**R3 finding:** Capping log_var prevents collapse. Uncapped ablation reproduces collapse.

**D8 challenge:** Has the uncapped ablation been run on the current codebase? The cap values changed (V1 said 1.5/1.0/0.5/2.0; V2 has -0.5/2.0, 0.0, 3.0).

**Implication:** The "Kendall collapse" finding from V1 may not apply to V2 codebase.

---

## 4. Missing 2025-2026 Papers (Systematic Search Needed)

### 4.1 Industrial MTL (2025-2026)

Searches to run:
- "industrial multi-task 2025"
- "factory MTL detection activity 2025"
- "MTL assembly egocentric 2026"

### 4.2 Head Pose MTL (2025-2026)

Searches to run:
- "head pose MTL 2025"
- "6D rotation multi-task 2026"
- "egocentric pose estimation 2025"

### 4.3 Long-Tail MTL (2025-2026)

Searches to run:
- "long-tail multi-task learning 2025"
- "imbalanced MTL classification 2026"

### 4.4 PSR / Action Detection (2025-2026)

Searches to run:
- "procedure step recognition 2025"
- "temporal action detection long-tail 2026"

---

## 5. Method Limitations Not in R3

### 5.1 MTL Sample Efficiency

**D8 finding:** Multi-task learning doesn't always improve sample efficiency. Some papers show MTL helps only when:
- Tasks share low-level features (CNN early layers)
- Tasks have similar data distributions
- Joint training data > ST training data per task

**Implication:** Our setup (egocentric assembly, all tasks share spatial features) is the GOOD case. But our sparse data per task may not benefit from MTL sample efficiency.

### 5.2 Catastrophic Forgetting in MTL

**D8 finding:** MTL models can catastrophically forget earlier tasks as training progresses. PCGrad doesn't prevent this.

**Counter-evidence:** SWA (Stochastic Weight Averaging) does help. Our codebase has SWA (`ema.py`, `pretrain_mae.py`).

**Implication:** We have SWA but should verify it's active in the main MTL run.

### 5.3 Negative Transfer Quantification

**D8 finding:** MTL papers often report "MTL beats ST on N/M tasks." But the magnitude of positive vs negative transfer matters.

**Counter-evidence:** Some papers show negative transfer can be -50% (catastrophic). Our targets (V1 doc 208: MTL/ST ≥ 0.60 for det, ≥ 0.30 for act) suggest we're OK with significant negative transfer.

---

## 6. Survived Findings

| Claim | Status |
|---|---|
| All 23 citations are real | HIGH |
| WACV 2024 anchors are accurate | HIGH |
| Kendall+PCGrad is our core method | HIGH |

---

## 7. Refined Findings

| Finding | Refinement (Batch 1 update) |
|---|---|
| 312x → **20,245x** gradient ratio | **Re-measured 2026-07-14**: pose=3278, act=13.80, det=1.86, psr=0.16. Pose dominates; act > psr (reversed from V1) |
| Kendall collapse reproducibility | Still needs ablation on current cap values |
| PCGrad limitations | Document; consider CAGrad as alternative |
| MetaBalance 10x cap | More insufficient than ever at 20,245x ratio |
| FAMO initialization | Verify our default matches paper |

---

## 8. Output

D8 reveals:
1. **Gradient ratio re-measured: 20,245x** (pose=3278, act=13.80, det=1.86, psr=0.16). 312x claim is **superseded**. Dominance pattern reversed: act > psr (was psr > act).
2. **Kendall collapse** reproducibility still needs verification on V2 codebase
3. **CAGrad** is a documented improvement over PCGrad — consider wiring
4. **2025-2026 systematic search** needed for missing papers

Most actionable: re-measure gradient norms on current codebase and run uncapped ablation. GeoHeadPose bug also needs fix (column-swap at model.py:2177-2178).
