# 180 — Opus Consultation Prompt: MTL Path Decision for MViTv2-S Multi-Task Learning

**Date:** 2026-07-09
**Purpose:** This file is the primary input for an Opus consultation. It summarizes the complete situation (epochs 1-6 of `mtl_mvit_run3`), presents the MTL paradox, and asks for a path decision.
**Accompanying files:** 177 (status + path analysis), 178 (50 deep questions), 179 (experimental design)
**Status:** Training is PAUSED pending this decision. We need Opus to read this file first, then refer to 177-179 as needed.

---

## SECTION 1: EXECUTIVE SUMMARY — THE PROBLEM

We are training a **4-task MTL model** (Detection, Activity, PSR, Pose) with a shared **MViTv2-S backbone** on the IndustReal assembly dataset.

**After 6 epochs (24K batches, ~4.4 hours):**
- **Detection:** Learning well spatially (DFL decode produces valid boxes, CIoU decreasing) BUT class-0 collapse (all cells predict class 0 at 0.9999). This may be normal <10 epochs.
- **Activity:** **STARVED.** Kendall weight = 0.04 (exp(-3.20)). Activity receives 4% of backbone gradient. Top-1 ≈ 0.8% (below random 1.33%).
- **PSR:** **FLAT.** Loss stuck at 1.30 across all 6 epochs. No learning signal reaches the PSR head. Event F1 = 0.0.
- **Pose:** **HEALTHY.** Loss 0.19, stable. Tanh-bounded outputs.

**The Kendall Paradox:** Kendall uncertainty weighting is designed to balance tasks automatically. But when activity CE loss (12.31) is 40× larger than detection loss (0.31), Kendall responds by increasing log_var_act (from -0.5→3.20), which reduces activity's backbone gradient to near zero. The mechanism meant to prevent negative transfer is also preventing positive transfer.

**We need Opus to decide: Path A (fix Kendall with per-task caps), Path B (accept current), or Path C (fixed weights)?**

---

## SECTION 2: WHAT WE HAVE — CURRENT METRICS (EPOCH 6)

### 2.1 Loss & Weight Table

| Task | Loss (Ep1) | Loss (Ep6) | Trend | log_var(Ep6) | Kendall Weight | Backbone Gradient % |
|------|-----------|-----------|-------|-------------|---------------|-------------------|
| Detection | 0.19 | 0.31 | Stable | -0.41 | 1.51 | ~30% |
| Activity | 10.89 | 12.31 | **Diverging** | +3.20 | **0.04** | **~4%** |
| PSR | 1.31 | 1.30 | **Flat** | +0.94 | 0.39 | ~18% |
| Pose | 0.14 | 0.19 | Stable | -0.49 | 1.63 | ~48% |

### 2.2 Eval Metrics (Epoch 5 — Only Eval So Far)

| Metric | Value | SOTA | Interpretation |
|--------|-------|------|---------------|
| Activity top-1 | 0.008 | 0.652 | Below random (0.0133) |
| Detection mAP@0.5 | 0.0 | 0.838 | Class collapse, all predict class 0 |
| PSR event F1 | 0.0 | 0.901 | Head not learning |
| Pose fwd MAE | ~10° | ~15° (unsourced) | Functional |

### 2.3 Model Efficiency

| Metric | Current (MTL) | 4× Single-Task (Estimated) | Savings |
|--------|--------------|---------------------------|---------|
| Params | 43.5M | ~138M (4×34.5M) | **68% fewer** |
| GFLOPs | 129.6 | ~200 (per-task × heads) | ~35% less |
| Training time | ~73h (100 ep) | ~72h (4×18h, parallel) | **Same** (but one model) |
| Inference | 1 forward pass | 4 forward passes | **4× faster** |

### 2.4 Training Health (20-Point Check)

- ✅ ZERO NaN events across 24K batches
- ✅ ZERO gradient warnings
- ✅ ZERO "skipping optimizer step"
- ✅ ZERO detection decode errors (DFL produces valid boxes)
- ✅ NMS pipeline verified working (4165→~400 boxes/image)
- ⚠️ log_var_act approaching +4 cap (spiral confirmed)
- ⚠️ PSR loss completely flat (no improvement, same as ConvNeXt collapse)
- ⚠️ Activity head predicts single class (class 11) at ~7% confidence

---

## SECTION 3: THE THREE PATHS

### 3.1 Path A — Per-Task Log_Var Caps (Recommended by analysis)

**Code change:** Add `.clamp(max=1.0)` for log_var_act, `.clamp(max=0.5)` for log_var_psr.

**Expected effect on weights:**

| Task | Current Weight | After Cap | Gradient Increase |
|------|---------------|-----------|-------------------|
| Activity | 0.04 | **0.37** | **9.25×** |
| PSR | 0.39 | **0.61** | **1.56×** |
| Detection | 1.51 | 1.51 | No change |
| Pose | 1.63 | 1.63 | No change |

**Expected metrics (Path A):**

| Metric | Ep10 (no fix) | Ep10 (Path A) | Ep50 (Path A) |
|--------|--------------|--------------|--------------|
| Act top-1 | 0.01 | 0.03-0.05 | 0.25-0.35 |
| Det mAP | 0.05-0.10 | 0.05-0.10 | 0.50-0.65 |
| PSR F1 | 0.0 | 0.05-0.15 | 0.50-0.70 |
| Pose MAE | 10° | 8-10° | 4-5° |

**Risks:**
1. Activity with 9.25× more gradient may corrupt detection/PSR features (PCGrad should mitigate)
2. Activity may hit the cap and stay there — cap becomes a hard constraint, not adaptive
3. Still may not reach SOTA activity (head is just LayerNorm→Linear)

**Code change details:**
```python
# In train_step(), before Kendall weighting:
log_var_det = log_vars[0].clamp(-4, 4)
log_var_act = log_vars[1].clamp(-4, 1.0)   # NEW: max weight = exp(-1.0) = 0.368
log_var_psr = log_vars[2].clamp(-4, 0.5)   # NEW: max weight = exp(-0.5) = 0.607
log_var_pose = log_vars[3].clamp(-4, 4)

total_loss = (
    0.5 * torch.exp(-log_var_det) * losses_safe[0] + log_var_det +
    0.5 * torch.exp(-log_var_act) * losses_safe[1] + log_var_act +
    0.5 * torch.exp(-log_var_psr) * losses_safe[2] + log_var_psr +
    0.5 * torch.exp(-log_var_pose) * losses_safe[3] + log_var_pose
)
```

### 3.2 Path B — Accept Current Formulation (Status Quo)

**Expected metrics (Path B, no changes):**

| Metric | Ep10 | Ep20 | Ep50 | Ep100 |
|--------|------|------|------|-------|
| Act top-1 | 0.01 | 0.02 | 0.03 | 0.05 |
| Det mAP | 0.05-0.15 | 0.20-0.40 | 0.40-0.55 | 0.45-0.60 |
| PSR F1 | 0.0 | 0.02 | 0.05 | 0.10 |
| Pose MAE | 8-10° | 6-8° | 4-6° | 3-5° |

**Verdict:** Activity and PSR never converge. MTL appears to "fail" because the optimization framework prevents learning. This is the worst path for the hypothesis that "MTL helps."

### 3.3 Path C — Fixed Weights (Remove Kendall)

**Configuration:** det=1.0, act=0.025, psr=0.24, pose=1.63 (equalized gradient contribution)

**Pros:**
- Predictable, controllable weights
- No learned parameter divergence
- Simpler optimization

**Cons:**
- Loses "automatic balancing" contribution claim in paper
- No adaptation as task difficulty changes
- Must hand-tune weights

**Expected metrics (Path C):**

Similar to Path A but with more predictable weight dynamics. The main risk is that 0.025 for activity may still be too low (Kendall arrived at 0.04 independently — similar magnitude).

---

## SECTION 4: KEY UNCERTAINTIES & QUESTIONS FOR OPUS

### 4.1 The Kendall Paradox (Must answer first)

**Q1:** Is the log_var_act spiral from -0.5→3.20 inevitable when one task's loss is 40× larger than others, or can it be fixed with better initialization, learning rates, or clamping?

**Q2:** Does Kendall's mechanism measure "aleatoric uncertainty" (as claimed) or is it just doing loss-scale normalization in disguise? If the latter, should we replace Kendall with explicit loss balancing?

**Q3:** What is the theoretical justification for Kendall with 4 tasks? Is there any guarantee of convergence to a stable equilibrium?

### 4.2 The Activity Bottleneck

**Q4:** Can a LayerNorm→Linear head on top of MViTv2-S class token achieve >30% top-1 on IndustReal 75-class activity with adequate backbone gradient? Or is the head architecture fundamentally insufficient?

**Q5:** If the head architecture IS sufficient, what weight does activity need to learn? Is Kendall's 0.04 catastrophically low, or is 0.37 (Path A cap) enough?

### 4.3 The PSR Collapse

**Q6:** Is PSR flat (1.30 loss across 6 epochs) because of:
(a) Kendall weight (0.39 — moderate)?
(b) DETACH_PSR_FPN (no gradient to backbone)?
(c) Conv_proj features (too early, no temporal info)?
(d) Causal Transformer capacity (3 layers, d_model=96)?

**Q7:** Should we remove DETACH_PSR_FPN and accept potential detection degradation, or keep it and give PSR an alternative gradient path?

### 4.4 Detection Class Collapse

**Q8:** Is detection class-0 collapse at epoch 5 a normal precursor to class discrimination (expected <15 epochs in YOLOv8), or is MTL specifically preventing class discrimination from emerging?

### 4.5 Experimental Design

**Q9:** Given limited compute (2 GPUs, 1 primary), what is the MINIMUM set of experiments to prove whether MTL helps or hurts? Is the 3-experiment set (E6b caps + E2a single-task activity + E1 single-task detection) sufficient?

**Q10:** What is the "MTL proof" bar for an AAIML paper? Does the paper need L1 (MTL beats single-task on all tasks), L2 (MTL beats single-task on one task, others within 90%), L3 (efficiency advantage), or L4 (multi-task inference use case)?

### 4.6 The Paper Story

**Q11:** What is the best narrative for the paper if Path A is chosen?

- Option 1: "MTL Provides Positive Transfer" — emphasize that shared features help all tasks, with evidence from gradient analysis and single-task comparison
- Option 2: "Efficient Multi-Task Video Understanding" — emphasize one model for all tasks at competitive accuracy, accept lower per-task metrics
- Option 3: "Diagnosing and Fixing MTL Optimization for Video Transformers" — make the Kendall paradox the methodological contribution

---

## SECTION 5: COMPLETE EXPERIMENTAL ROADMAP (30-SECOND VERSION)

```
CURRENT STATE:
├─ MTL-MViTv2-S, 6 epochs complete
├─ Detection: spatial OK, class collapse (normal <10 ep)
├─ Activity: STARVED (Kendall weight=0.04)
├─ PSR: FLAT (loss 1.30, no improvement)
└─ Pose: HEALTHY

CRITICAL QUESTION:
└─ Does MTL help or hurt? (Current evidence: hurts 3/4 heads)

TOP 3 EXPERIMENTS:
1. E6b: Apply log_var caps (act max=1.0, psr max=0.5) — is it fixed? (14.7h)
2. E2a: Single-task activity — what's the ceiling? (3.1 days)
3. E1: Single-task detection — what's the MTL cost? (3.1 days)

DECISION NEEDED:
└─ Path A (caps), Path B (accept), or Path C (fixed weights)?
```

---

## SECTION 6: DATA DIAGNOSTIC SUMMARY

### 6.1 Detection DFL Decode Verification

```
P2 (56×56, 3136 cells): >0.001=3128 >0.5=8  boxes=[-5.3, 331.1]
P3 (28×28, 784 cells):  >0.001=748  >0.5=22 boxes=[-4.2, 239.0]
P4 (14×14, 196 cells):  >0.001=185  >0.5=20 boxes=[-1.5, 170.5]
P5 (7×7, 49 cells):     >0.001=41   >0.5=2  boxes=[-0.4, 119.1]
```
All boxes are valid. NMS reduces 4165→~400 boxes per image. Top boxes are ~71×117px (person-sized).

### 6.2 Activity Diagnostic

- All samples predict class 11 at ~7% confidence
- Logit range: [-3.96, 4.07] (meaningful range, wrong class)
- Per-class max sigmoid: class 11 dominates
- Activity loss: 12.31 (epoch 6), up from 3.98 (epoch 5)

### 6.3 PSR Per-Component Predictions

```
Comp  0: mean=0.008 (never positive)
Comp  1: mean=0.465
Comp  2: mean=0.453
Comp  3: mean=0.692
Comp  4: mean=0.481
Comp  5: mean=0.662
Comp  6: mean=0.670
Comp  7: mean=0.684
Comp  8: mean=0.684
Comp  9: mean=0.638
Comp 10: mean=0.521
```
Components 3-10 hover near 0.5-0.7 (uncertain). Component 0 is almost never predicted. This pattern is consistent with the head learning per-component biases and not reacting to input features.

### 6.4 Pose Diagnostic

- Tanh-bounded mean = -0.383 (within [-1, 1])
- Loss = 0.19 (stable, moderate)
- Forward MAE ~10°, Up MAE ~18° (epoch 5 eval)

---

## SECTION 7: THE CORE QUESTION (FOR OPUS TO ANSWER)

After reading files 177-180, Opus should answer:

**1. Which path should we take?** Path A (caps), Path B (accept), or Path C (fixed weights)?

**2. What is the minimal experimental protocol to prove "MTL helps"?** Which of the 10 experiments in file 179 are essential?

**3. What is the paper narrative if Path A results are mixed (activity improves to 20%, detection to 50%, PSR to 0.6)?** Is that publishable as an efficiency story?

**4. Is there a Path D that we haven't considered?** (e.g., staged training, task-specific backbone branches, gradient surgery variants)

**5. What is the single most important change to make right now?** (One action, one command, one code change.)

---

## SECTION 8: REFERENCE FILES

| File | Content |
|------|---------|
| `177_MTL_CRITICAL_STATUS_AND_PATH_DECISION.md` | Full training status, 20-point checklist, Path A/B/C analysis |
| `178_50_DEEP_QUESTIONS_FOR_PATH_DECISION.md` | 50 deep questions across 7 sections: Kendall, PCGrad, Activity, PSR, Detection, Architecture, Hypothesis |
| `179_EXPERIMENTAL_DESIGN_FOR_PROVING_MTL.md` | 10 experiments with compute budget, expected outcomes, decision tree |

---

## SECTION 9: FILE SYSTEM PATHS (FOR DEBUGGING)

```
Checkpoint:     /media/.../industreal_improved/src/runs/rf_stages/checkpoints/mtl_mvit_run/best.pt
Training log:   /tmp/mtl_mvit_run3.log
Config:         /media/.../industreal_improved/src/config.py
Model:          /media/.../industreal_improved/src/models/mvit_mtl_model.py
Training loop:  /media/.../industreal_improved/scripts/train_mtl_mvit.py
Metrics:        /media/.../industreal_improved/src/runs/rf_stages/checkpoints/mtl_mvit_run/metrics.json
Diag script:    /tmp/diagnose_training.py
Deep diag:      /tmp/deep_diagnose.py
```

---

*Files 177-180 together form the complete consultation package for Opus to decide the MTL path. Read this file first (180), then dive into 177 for details, 178 for questions, and 179 for experimental design.*
