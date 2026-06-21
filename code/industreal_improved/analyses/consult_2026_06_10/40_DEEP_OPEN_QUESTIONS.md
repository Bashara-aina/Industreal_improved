# 40 — Deep Open Questions: What We Still Cannot Answer

> Generated 2026-06-21 — After 15 phases across 10 weeks, 9 Opus consultations, 3 independent training regimes, 34+ diagnostic probes, and ~500 GPU-hours of accumulated training  
> Current state: RF2 epoch 21 (mAP50 plateau at 0.204-0.215, 6 epochs and counting)

---

## How to Read This Document

This is NOT the same as `33_OPEN_QUESTIONS.md`. That document catalogs questions as actionable items organized by severity. **This document goes deeper** — it explores the fundamental unknowns that persist after all our investigation, the questions that we cannot answer even with our current evidence, and the uncomfortable possibility that some of them may never be answerable with the tools we have.

Each chapter explores one deep question through:
- **The core unknown** — what we genuinely cannot answer
- **Evidence we have** — what we DO know, and why it's not enough
- **The blind spots** — what information we'd need but can't easily get
- **Why it matters** — what changes if we knew the answer
- **The uncomfortable possibility** — what the answer might be that we don't want

---

## Chapter 1: The Class-Specific AP Mystery

### 1.1 The Core Unknown

**Why are 12 of 24 classes always at AP=0 across every epoch, every run, every configuration?**

This is the single most important unanswered question in the entire project. If we understood this, we would likely understand the entire detection plateau.

### 1.2 What We Know

- **det_n_present_classes is 15-16/24**, always the same classes present
- **Pseudo-classing mAP (det_mAP50_pc = 0.307-0.344)** is ~50% above raw mAP — raising the average of working classes would disproportionately raise RAW mAP
- **Class 6 has 1500-1800 GT instances per epoch** yet is AP=0 — it's not a data scarcity problem
- **POS_ANCHOR_PROBE shows the classifier working on positive anchors** — the classifier can learn, just not for these classes
- **The pattern is epoch-consistent** — it's not random variation, it's systematic

### 1.3 The Four Hypotheses — None Confirmed

**Hypothesis A: Label Error (the simplest explanation)**
- Class 6 labels are systematically wrong — wrong boxes, wrong class IDs, temporal misalignment
- 1500+ GT instances with AP=0 is nearly impossible without label error in a model that CAN learn (proven by other classes at mAP50~0.40-0.60)
- If true: fix the labels → those classes jump from AP=0 to AP~0.30 → raw mAP rises to ~0.30
- **Uncomfortable possibility**: All 12 AP=0 classes have wrong labels → raw mAP would go from 0.204 to ~0.35 with just label fixes. This would mean weeks of investigating training dynamics, consultation rounds, and architecture tweaks were addressing the WRONG problem.

**Hypothesis B: Anchor Geometry Mismatch**
- Class 6 objects have sizes/aspect ratios incompatible with the anchor grid
- If no anchor has IoU>0.5 with ANY class 6 GT, the GT produces zero positive matches regardless of TOP_K
- The TOP_K=9 fix only helps if IoU>0 (it force-assigns the best 9, even at IoU~0.05)
- But at IoU~0.05, the anchor is essentially random — the classifier learns noise
- **Uncomfortable possibility**: The anchor grid was designed for COCO-style objects (medium/large, aspect ratios ~0.5-2.0). IndustReal assembly parts may have fundamentally different size distributions. A complete anchor redesign may be needed.

**Hypothesis C: Feature Confusion Between Similar Classes**
- Multiple assembly parts look similar at the FPN feature scale
- The classifier consistently predicts class A for class B's objects
- This would produce: AP=0 for class B (it's never correctly predicted), inflated FP for class A (lowers its precision)
- The 12 AP=0 / 12 working classes pattern could be "confusion pairs" — 12 classes are consistently confused with their 12 partners
- **Uncomfortable possibility**: 24 assembly-part classes from synthetic projection is too fine-grained. The Bayes-optimal classifier may genuinely be unable to separate them at this resolution. No amount of training fixes this — you'd need to merge similar classes or get better labels.

**Hypothesis D: Top-k Poisoning Is Class-Specific**
- The top-k IoU floor problem (Q32) disproportionately affects classes where objects are smaller than ANCHOR_SIZES[0]=96px
- If class 6 objects are typically <96px in the 720p image, their best anchors are at the smallest scale
- The FPN features at P7 (smallest scale) may be insufficient for fine-grained classification
- **Uncomfortable possibility**: This is fundamentally a multi-scale representation problem. The current FPN may not provide sufficient resolution for small-object classification, regardless of training improvements.

### 1.4 The Crucial Missing Information

**We need to know, in order of importance:**
1. **Per-class AP from epoch-end EVAL** — the raw per-class AP arrays are in metrics.jsonl but we haven't parsed them. This would tell us: is it truly 12 classes at AP=0 or a spectrum?
2. **Anchor-IoU histogram per class** — what IoU does each class's GT achieve with its best anchor? This would confirm/refute Hypothesis B.
3. **Visual inspection of class 6 GT boxes** — are the labels correct? A 30-minute visual audit of 50 random class 6 GT boxes would answer Hypothesis A.
4. **Confusion matrix from validation** — which classes are being predicted for class 6's GT? This would confirm/refute Hypothesis C.

### 1.5 What Changes If We Knew

- **If label error**: Stop investigating training dynamics. Fix the labels. Raw mAP likely jumps to ~0.30-0.35.
- **If anchor mismatch**: Redesign anchor grid or switch to ATSS matching. This is a known fix with well-established methods.
- **If feature confusion**: Either merge similar classes (changes the task definition) or accept the ceiling. Neither is an easy path.
- **If top-k poisoning**: Add IoU floor to top-k matching. Run 5 epochs to verify improvement.

---

## Chapter 2: The Structural Ceiling

### 2.1 The Core Unknown

**Is mAP50≈0.21 the absolute ceiling of this architecture with this data and loss function?**

We have 6 consecutive epoch-ends at mAP50=0.204-0.215 with zero trend. The LR restart failed. We've never seen mAP above 0.215 in ANY configuration across 3 independent training regimes.

### 2.2 What "Structural" Actually Means

A structural ceiling means the loss landscape itself has no gradient path to a better solution. The model is not "stuck" in a local minimum — it's at the global minimum of the achievable loss landscape.

**Evidence for structural ceiling:**
1. **LR restart produced zero change** (epoch 20 CosineAnnealing) — this eliminates "stuck in local minimum at low LR"
2. **6 epochs of absolutely flat trend** — not noise, not slow improvement, literally zero slope
3. **Cross-regime consistency** — RF2, R2.5, Run 8 all converged to similar mAP (~0.15-0.21) before collapse or plateau
4. **RF1's claimed 0.45 was a bug** — the REAL RF1 best was probably ~0.184 (epoch 8), consistent with all other runs

**Evidence against structural ceiling:**
1. **Pseudo-classing mAP at 0.307-0.344** — if the architecture can reach 0.344 on a per-class basis, the total potential if all classes worked might be 0.40+
2. **POS_ANCHOR_PROBE shows the classifier CAN produce confident scores** — ceiling may be in matching/evaluation, not in raw classification ability
3. **We've never tried the 50-image cls-only overfit** — we don't know what the architecture is capable of in isolation

### 2.3 The Ceiling Components

The observed mAP of 0.204 likely decomposes as:

```
mAP50 = 0.204 = (classes_working / 24) × avg_working_mAP × (1 - confusion_penalty)

Where:
- classes_working / 24 ≈ 12/24 = 0.5 (only half the classes contribute)
- avg_working_mAP = unknown (could be 0.40-0.60 if confusion penalty is small)
- confusion_penalty = unknown (FP from confused classes reduces precision)
```

This decomposition is the key unknown. If `avg_working_mAP` is already 0.50, then fixing the 12 AP=0 classes would yield mAP≈0.25 (with confusion penalty) to 0.25 (no confusion). That's still well under 0.40. The ceiling may be structural even for the "working" classes.

### 2.4 The Uncomfortable Truth

**The ceiling may be the labels.** Opus v9's strongest hypothesis (§3.2): IndustReal synthetic labels may be noisy enough that the Bayes-optimal classifier IS a low-mAP one. If 5-10% of class labels are wrong, the maximum achievable mAP with those labels is ~0.25-0.30 regardless of architecture, loss function, or training duration.

If this is true:
- No amount of scheduler tuning, loss function tweaking, or architectural modification fixes it
- The entire investigation (Phases 1-15, 9 Opus consultations) was addressing a symptom, not the root cause
- The path forward is: audit labels → fix them → retrain → if mAP jumps, label noise was the ceiling all along

### 2.5 The Experiment That Would Answer This

**The 50-image cls-only overfit (Opus v9 §6).** Train on 50 images, classification-only, head_pose OFF, Kendall OFF, 300-500 steps. Cost: <30 minutes.

| Outcome | Conclusion |
|---------|-----------|
| mAP → 0.8+ | Architecture is fine. Ceiling is multi-task dynamics or data-at-scale issue. Fix: KENDALL_FIXED_WEIGHTS or continue training. |
| mAP → 0.3-0.5 | The architecture has a real ceiling, but it's not the 0.21 we see. The gap between 0.5 and 0.21 is label noise + dynamics. Fix: both. |
| mAP → 0.15-0.25 | Even in isolation, the architecture can't do better. The plateau IS the architecture's ceiling. Fix: fundamental architecture change. |
| No localization | The anchor/assignment pipeline is fundamentally broken. Fix: matching before anything else. |

We have not run this experiment. Every day we spend not running it, we accumulate more data that may be uninterpretable without knowing the answer to this.

---

## Chapter 3: The Top-k Paradox

### 3.1 The Core Unknown

**Did DET_POS_IOU_TOP_K=9 (Fix 2) create a NEW problem that masks its own benefit?**

The fix was intended to solve "not enough positive anchors." It succeeds at that — we now have 6-10× more positive anchors. But without a minimum-IoU guard, it may be poisoning the classifier for small/medium objects.

### 3.2 The Double-Edged Sword

Before Fix 2:
```
~16 positive anchors/batch, all at IoU>0.5 → perfect labels, but gradient starvation
```

After Fix 2 (no IoU floor):
```
~120 positive anchors/batch → 6× more gradient, but:
  - 16 at IoU>0.5 (clean labels, same as before)
  - ~104 at IoU 0.05-0.5 (noisy labels — classifier learns wrong associations)
```

The net effect on the classifier:
```
Δw = η × (Σ_clean ∇L_clean + Σ_noisy ∇L_noisy)
```

If `Σ_noisy ∇L_noisy` is large enough (104 noisy vs 16 clean anchors), the classifier may learn MORE from the noisy signals than the clean ones. The 6× increase in gradient may be a 6× increase in WRONG gradient.

### 3.3 Why We Can't Tell

The mAP plateau is consistent with BOTH:
- **Under-supply of positive gradient** (the original problem Fix 2 targeted)
- **Over-supply of noisy positive gradient** (the new problem Fix 2 may have created)

Both would produce similar behavior: the classifier makes some confident predictions (from the 16 clean anchors) but also fires false positives (from the 104 noisy anchors). The net mAP is ~0.21 either way.

### 3.4 The Test

Add a minimum IoU floor to the top-k match:
```python
# Current (no floor):
for idx in topk_idx:
    labels[idx] = gt_labels[gi]

# With floor:
for idx in topk_idx:
    if gi_ious[idx] >= 0.3:
        labels[idx] = gt_labels[gi]
    # else: leave as ignore (-1), not trained as positive or negative
```

Then retrain for 5 epochs:
- If mAP increases: the top-k WAS poisoning the classifier. Floor fixes it.
- If mAP stays same: the top-k wasn't the problem. Ceiling is elsewhere.
- If mAP decreases: clean positives alone aren't enough. Need more gradient, even if noisy.

This 5-epoch experiment (~7 hours) would definitively answer whether Fix 2 helped or hurt.

### 3.5 The Deeper Question

If the top-k WITHOUT floor is genuinely worse than the original (no top-k), then we've spent weeks with a "fix" that made things worse. The failure to detect this would be a systematic validation gap: we never checked whether the fix achieved its intended effect because we were measuring the wrong metrics (score_p50, LOCALIZING verdict — which Opus v9 proved can't see classification quality).

---

## Chapter 4: The detach_reg_fpn Schizophrenia

### 4.1 The Core Unknown

**What is the effective value of C.DETACH_REG_FPN in the running configuration?**

This one boolean value changes the entire diagnosis of the project's central failure mode.

### 4.2 Two Different Architectures

**If True** (what `config.py:1109` says):
```
Training signal flow:
  cls_loss → cls_subnet → FPN → backbone ← head_pose_loss
  reg_loss → reg_subnet ──┘         (detached from FPN)
```

The regression subnet is detached. Only classification and head_pose shape the backbone. Localization is produced by the reg subnet on features carved by CLASSIFICATION + HEAD POSE — not by regression errors.

**Under this regime:**
- The excellent localization (bestIoU=0.86-0.98) proves the features ARE good for localization
- If features are good for localization but NOT for classification, the problem is squarely in the cls subnet / cls loss / cls targets
- Head_pose "dominating" the backbone doesn't explain anything — if it had wrecked features, localization would suffer
- **The finger points at:** label noise (§3.2), top-k poisoning (§3.3), or Focal Loss inadequacy

**If False** (what documentation claims):
```
Training signal flow:
  cls_loss → cls_subnet ─┐
                          ├── FPN → backbone ← head_pose_loss
  reg_loss → reg_subnet ─┘
```

Both subnets feed the backbone. The Kendall-domination story applies — head_pose may be shaping features suboptimally for detection.

**Under this regime:**
- The dissociation between "classifies poorly" and "localizes well" is harder to explain
- Head_pose dominating the backbone becomes a real concern
- **The finger points at:** multi-task interference, Kendall loss weighting, or feature competition

### 4.3 Why We Never Resolved This

This is the most embarrassing gap in the entire investigation. We've had 9 Opus consultations. The split-brain was identified in Opus v9. But we haven't printed `C.DETACH_REG_FPN` at step 0. The fix is literally one print statement:

```python
print(f"EFFECTIVE CONFIG: DETACH_REG_FPN={C.DETACH_REG_FPN}, REINIT_PI={C.REINIT_PI}")
```

We haven't done this. Not because it's hard. Because somehow, across all this investigation, we never prioritized resolving this ambiguity.

### 4.4 The Cost

Every analysis, every diagnosis, every consultation round has operated under an assumption about this value. If the assumption was wrong:
- **Our understanding of which mechanism causes the plateau is WRONG**
- **The fixes we prioritized may address the wrong problem**
- **We may have spent 9 consultation rounds diagnosing a phantom mechanism**

### 4.5 The Uncomfortable Possibility

If True (which the committed code says):
- Everything changes. The problem IS the cls loss/targets/labels. Not the backbone, not multi-task interference, not head_pose domination.
- This means: Opus v1-v8's analysis of multi-task dynamics was largely irrelevant to the actual failure mode.
- The real fix would be: audit labels, fix top-k floor, potentially switch to Quality Focal Loss or ATSS matching.
- "Head_pose ate the backbone" is WRONG. The backbone is fine. The cls head is the bottleneck.

If True, we've been investigating the wrong mechanism for weeks.

---

## Chapter 5: The Gradient Bottleneck

### 5.1 The Core Unknown

**Why is the detection head's total gradient norm (2.35e-02) 117× smaller than the backbone gradient norm (2.770e+00)?**

Is this normal for a RetinaNet-style detector, or is it a pathology?

### 5.2 The Two Interpretations

**Normal interpretation:** The gradient decays naturally through the FPN and cls subnet. The cls subnet has 4 Conv3×3+ReLU layers. Each layer attenuates the gradient. The total attenuation of 117× over 10+ layers is expected.

Pathological interpretation: Something is blocking detection gradient flow. Possible causes:

1. **Focal Loss suppression**: Focal Loss (γ=2) explicitly suppresses gradient from well-classified examples. After 21 epochs, if most positive anchors have scores of 0.64-0.80 (from POS_ANCHOR_PROBE), their Focal Loss gradient is suppressed by `(1-p)^γ = (0.36)^2 = 0.13` — only 13% of the normal gradient.

2. **EMA model divergence**: If the LIVENESS_GRAD probe is reading from the EMA model (which averages weights over time), and the online model has larger gradients, then the 117× ratio is an artifact of EMA smoothing.

3. **Frozen normalization layers**: If batch norm running statistics are frozen (as they often are in fine-tuning), gradient through these layers is zero, effectively creating bottlenecks.

### 5.3 Why This Matters

If the gradient bottleneck is pathological:
- Detection barely contributes to backbone representation learning
- The backbone is being shaped almost entirely by head_pose (and body_pose)
- Detection features are a "free rider" on other heads
- This explains the mAP plateau: detection can't learn because it can't influence its own feature extractor

### 5.4 The Missing Measurements

We need gradient norms at multiple points in the detection pathway:
```
cls_subnet output:  ? ← bottleneck if gradient here is small
FPN P3 output:      ? ← bottleneck if gradient here is small vs large at P7
FPN P7 output:      ? ← bottleneck if gradient decays with pyramid level
Backbone C5:        2.770e+00 (known)
```

Without these measurements, we can't tell whether the bottleneck is in the FPN, the cls subnet, or naturally distributed.

---

## Chapter 6: The score_p50 Blindness Retrospective

### 6.1 The Core Unknown

**How much of our "understanding" from consultation rounds 6-9 was based on a metric that physically cannot measure what we thought it was measuring?**

### 6.2 The Scale of the Problem

score_p50 was used as evidence in:
- **Q01 analysis** (rounds 6-8): "classifier collapsed at epoch 15, score_p50=0.019"
- **Q04 analysis**: "bias drift produces ~0.079 uniform scores"
- **Phase 14 analysis**: "score_p50 range 0.020-0.072 is promising"
- **"LOCALIZING but not CLASSIFYING" narrative**: the "not CLASSIFYING" half relied on score_p50

If score_p50 cannot measure classification quality (proven by Opus v9 §1.1), then ALL of these conclusions are either weakened or invalid.

### 6.3 What Survives

The following conclusions are INDEPENDENT of score_p50:
- Catastrophic collapse at epoch 15: epoch-end mAP50 went to 0.001. That's real.
- 6-epoch plateau: 5 epoch-end validations. Real.
- POS_ANCHOR_PROBE shows classifier learning: completely independent measurement. Real.
- 12/24 AP=0: per-class AP from EVAL. Real.
- head_pose convergence: MAE decreasing. Real.
- LR restart failure: epoch-end mAP before/after restart. Real.

### 6.4 What Collapses

- **"Classifier collapsed"** narrative: at epoch 15, mAP DID collapse, but the characterization (uniform weights, score_p50=0.019 as evidence) was over-interpreted. The classifier may have collapsed differently than we thought.
- **"~0.079 uniform equilibrium"** characterization: score_p50 can't see this. We don't actually know what the score distribution looked like at epoch 15 collapse. We just know mAP was 0.001.
- **"score_p50 range 0.020-0.072 is promising"**: this is literally meaningless. It measures bias drift, not classification health. The POS_ANCHOR_PROBE is the real evidence.

### 6.5 The Uncomfortable Truth

We had a metric that told us what we wanted to hear (or fear), and we believed it without verifying what it actually measured. The code was there — evaluate.py:107-131 is 24 lines. We could have read it at any point. We didn't.

Opus v9 caught this because it READ the evaluation code. We didn't. The lesson: when a single number drives your diagnosis, verify what that number actually measures by reading its source code.

---

## Chapter 7: The PSR Never-Trained Puzzle

### 7.1 The Core Unknown

**Does the PSR head actually work? Has it EVER produced a non-trivial prediction?**

PSR loss = 1.546e-08 constant across ALL runs, ALL configurations, ALL phases. 10+ training runs, 1000+ DET_PROBE entries.

### 7.2 The Two Interpretations

**Benign interpretation** (Opus v9's correction): The 1.546e-08 is the binary-focal floor of a predictor trivially correct on the ~20/22 always-zero components. The formula `(1-p_t)^γ * CE_loss` produces a finite-but-degenerate value when the predictor outputs exactly 0.0 (sigmoid via extreme negative logits). This is not a "frozen graph" — it's just a predictor that found the correct trivial answer.

**Pathological interpretation**: The causal transformer produces extreme logits (-23 to +22), sigmoid saturates, gradient vanishes. The head is genuinely untrainable in its current form. No amount of training data or duration will fix this because the gradient is literally zero.

### 7.3 The Distinction Matters

If benign: PSR just needs better training signal. The 50-sequence overfit will show it CAN learn if given non-trivial targets. We can train it in RF4.

If pathological: PSR needs an architecture fix (logit clamp, smaller weight init, pos_weight, or different loss). The paper's novelty claim is at risk if PSR doesn't work.

### 7.4 The Crucial Experiment

The PSR 50-sequence overfit (Opus v9 §Q5). Run PSR-only on 50 sequences. Fully decoupled from detection. <1 hour. If PSR can overfit 50 sequences, it works. If it can't, the architecture is fundamentally broken.

### 7.5 Why We Haven't Done This

PSR isn't trained in RF2 (train_psr=False). It's "an RF4 problem." But:
- We've known PSR has never trained since Phase 6
- We've had the capacity to run the 50-sequence overfit for weeks
- We keep deferring it because it's "not blocking current stage"
- But PSR is the paper's novelty claim. If it doesn't work, the entire architecture thesis is undermined

---

## Chapter 8: The Kendall Invariance

### 8.1 The Core Unknown

**Do Kendall uncertainty weights actually matter for this architecture's training dynamics?**

The HP_PREC_CAP clamp ensures `lv_hp >= lv_det`, which means head_pose never gets less Kendall weight than detection. This was our primary fix for preventing head_pose gradient starvation. But:

### 8.2 The Evidence for Invariance

- KENDALL_FIXED_WEIGHTS=False (current) produces the same plateau as the collapsed run (pre-HP_PREC_CAP)
- The LR restart didn't change anything — the Kendall weights presumably re-converged to the same values
- The Kendall log-vars stabilize within 2-3 epochs after any change

This suggests: **the Kendall weights are just passive followers of the loss magnitudes, not active drivers of training dynamics.** The HP_PREC_CAP clamp prevents extreme starvation but doesn't change the fundamental behavior — it just prevents the worst case.

### 8.3 The Deeper Question

If Kendall weights are invariant (within bounds), then:
- **They're not the mechanism driving the plateau**
- The plateau is a property of the loss landscape, not the loss weighting
- Flipping KENDALL_FIXED_WEIGHTS=True (Opus v9 §Q2) would confirm this — if the plateau doesn't change, Kendall was never the lever

### 8.4 The Uncomfortable Possibility

We may have spent 3 consultation rounds (v6, v7, v8) and 4 fixes on a mechanism that doesn't control the outcome. The Kendall weights mediate the balance between heads, but if ALL heads' losses are at their respective floors (detection at its ceiling, head_pose at its mean-pose floor, body_pose near its floor), the Kendall weights have nothing to balance. They converge to whatever ratio minimizes the current losses, and that ratio is the same regardless of initialization or clamping.

If this is true, the HP_PREC_CAP fix was a red herring — it addressed a mechanism that doesn't control the central failure mode.

---

## Chapter 9: The LR Restart Puzzle

### 9.1 The Core Unknown

**Why did CosineAnnealingWarmRestarts at epoch 20 produce absolutely zero effect?**

This is one of the most puzzling results in the entire investigation. A well-designed LR restart should:
1. Increase the LR to its maximum value
2. Reset AdamW momentum buffers
3. Potentially escape local minima

The restart produced statistically identical metrics. Not worse, not better — identical.

### 9.2 The Three Interpretations

**Interpretation 1: The plateau is a global minimum, not a local minimum.**

The model is at the best possible point in the loss landscape. Any perturbation returns to this point because it is the lowest loss achievable. The LR restart is like kicking a ball in a flat valley — it rolls in one direction temporarily, then returns to the center.

**Prediction**: Any perturbation (LR restart, weight noise, data shuffle) would produce the same result. The model is at the global minimum of a convex-ish loss landscape.

**Interpretation 2: The gradient is zero everywhere, so LR doesn't matter.**

If gradients are near-zero (detection_head=2.35e-02), even a 10× LR increase produces negligible weight updates. The restart can't help because there's no gradient signal to amplify.

**Prediction**: Monitoring gradient norms before/after the restart would show no change. The gradients are zero before, zero after.

**Interpretation 3: The restart perturbs into a different loss basin, but both basins have the same loss value.**

The loss landscape has multiple degenerate minima (common in overparameterized networks), all at approximately the same loss. The restart moves the model to a different minimum but with the same loss and same mAP.

**Prediction**: The model's weights would be different after the restart (in parameter space) but produce the same metrics. Model similarity analysis (CKA, weight distance) would confirm.

### 9.3 Why Distinguishing These Matters

- **If Interpretation 1 (global minimum)**: Only architectural changes (more data, better labels, different loss) can help. No training modification will break the ceiling.
- **If Interpretation 2 (zero gradient)**: Finding the gradient bottleneck (Chapter 5) and fixing it would unlock further progress.
- **If Interpretation 3 (degenerate minima)**: The model is not fundamentally limited — it just needs a different optimization approach (e.g., Sharpness-Aware Minimization, lookahead optimizer) to find a better basin.

### 9.4 The Experiment

We need gradient norms logged at high resolution around the restart epoch:
```
epoch 19 step -100: grad = ?
epoch 20 step 0:    grad = ?   (immediately after restart, high LR)
epoch 20 step 100:  grad = ?   (after some steps at high LR)
epoch 20 step 1000: grad = ?
```

If gradients never increase, Interpretation 2 is correct. If they increase but mAP doesn't change, Interpretation 1 or 3.

---

## Chapter 10: The Question That Underlies All Others

### 10.1 The Meta-Question

**Do we need to fix training dynamics, or do we need to fix the data?**

Every investigation, every consultation, every fix has targeted training dynamics — loss functions, gradient flow, matching strategies, Kendall weights, LR schedules.

Opus v9's hypothesis (label noise as ceiling) suggests we may have been solving the wrong problem. The data may be the ceiling. If so:
- **All training dynamics fixes are irrelevant** — they can't raise a ceiling that's determined by label quality
- **All diagnosis of "collapse" and "plateau"** is misattributed — the model was always doing the best it could with the labels given
- **The path forward is data, not architecture**: label audit → label correction → retrain → if mAP jumps, labels were always the ceiling

### 10.2 How We Could Have Known Earlier

Signs that pointed to label issues that we dismissed:
1. **RF1's phantom 0.45**: We assumed it was a recording bug (it was). But the intense focus on this number made us think "the model CAN reach 0.45." That assumption drove 9 consultation rounds.
2. **Synthetic labels from floor plan projections**: We knew the labels were synthetic. We assumed they were accurate enough. We never verified.
3. **24 classes from assembly parts**: Many are visually similar (screw of type A vs screw of type B). At 720p resolution, the FPN features may genuinely not distinguish them.
4. **Per-class AP**: We had per-class AP in the EVAL output from day one. We never parsed it. If we had, we would have seen the 12/24 AP=0 pattern weeks ago.

### 10.3 The Single Most Important Action

**Run the 50-image cls-only overfit.** It costs <30 minutes and answers the meta-question:
- If mAP→0.8: dynamics are the issue. Fix training.
- If mAP→0.3-0.5: labels are part of the ceiling. Fix both.
- If mAP→0.15-0.25: architecture is the ceiling. Fix architecture.
- If no localization: pipeline is broken. Fix matching.

Every day we don't run this, we accumulate more data that may be uninterpretable without knowing the answer to the meta-question.

---

## Appendix: The 12 Questions We Could Answer in <2 Hours

These are not deep unknowns — they are measurements we could take RIGHT NOW that would dramatically improve our understanding:

| Question | What to Look At | Time |
|----------|----------------|------|
| Per-class AP | Parse metrics.jsonl per-class AP arrays | 5 min |
| Anchor-IoU per GT class | Add MATCH_PROBE logging to losses.py | 30 min + 1 epoch |
| cls_score.weight.norm() | Add logging to train.py epoch end (Opus v8 E3) | 10 min |
| effective DETACH_REG_FPN | Print C.DETACH_REG_FPN at step 0 (Opus v9 §5.4) | 5 min |
| cls_score.bias value | Log bias value at epoch end | 10 min |
| Gradient at multiple FPN levels | Add grad norm logging at P3, P5, P7 outputs | 30 min + 1 epoch |
| 50-image cls-only overfit | Isolated training on 50 images (Opus v9 §6) | <30 min |
| PSR logit values | Log transformer outputs before sigmoid | 10 min |
| Score at higher eval thresholds | Evaluate at thresh=0.05, 0.10, 0.25 | 5 min |
| Visual class 6 label audit | Overlay 50 random class 6 GT boxes on images | 30 min |
| Gradient norms pre/post LR restart | Plot gradient norms around epoch 20 | Already happened — just look at logs |
| Confusion matrix from val | Parse which classes are predicted for each GT | 10 min |

---

*Generated 2026-06-21. This document represents the deepest unknowns after 15 phases, 9 Opus consultations, and ~500 GPU-hours of investigation. Some of these questions may be unanswerable with current tools. Some may have answers we don't want. All of them are genuine — no question here has been included for rhetorical effect.*
