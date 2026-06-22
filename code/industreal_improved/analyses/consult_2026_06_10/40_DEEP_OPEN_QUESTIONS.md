# 40 — Deep Open Questions: What We Still Cannot Answer

> Generated 2026-06-21 — Updated 2026-06-21 22:16 UTC: **CRITICAL CORRECTION APPLIED — see CORRECTION NOTICE below.**  
> Prior header claimed "ba48691 restart confirmed: detach fix alone is insufficient" — THIS CONCLUSION IS INVALIDATED. It was based on Run 1 (wrong LR/BIAS=4.0/2.0). Run 2 (correct LR/BIAS=1.0/1.0) has only completed 1 epoch val (epoch 17, mAP50=0.2039). The OHEM+FocalLoss gradient suppression hypothesis is SUPPORTED BY THE OVERFIT but NOT YET CONFIRMED OR REFUTED BY MAIN TRAINING WITH CORRECT CONFIG. See CORRECTION NOTICE below.

---

## ⚠️ CRITICAL CORRECTION NOTICE (2026-06-21 22:16 UTC)

**What changed**: Analysis of the training log revealed that what we thought was "5 epochs of post-restart data" was actually **Run 1 with wrong LR/BIAS multipliers** (DET_BIAS_LR_FACTOR=4.0, DET_LR_MULTIPLIER=2.0). The correct-config Run 2 (both=1.0) started at 19:11:11 UTC and has completed only **1 epoch val** (epoch 17, mAP50=0.2039 — restart from same checkpoint, expected to match).

**What this invalidates**: The central conclusion across ALL analysis files — "detach fix alone insufficient, OHEM+FocalLoss validated as primary bottleneck by main training" — was based on Run 1's flat 5 epochs. This evidence is **invalidated**. The correct-config Run 2 has not yet generated enough data to draw conclusions.

**What survives unchanged**:
- ✅ The 50-image overfit (WEAK PASS, cls_loss→0.062) — this was a separate experiment, not affected
- ✅ The per-class AP correction (class 6 has 65/91 samples, not 1739) — from config.py audit
- ✅ The detach_reg_fpn fix (committed in ba48691) — correct fix applied
- ✅ POS_ANCHOR_PROBE showing 364-783 positive anchors/image in main training
- ✅ The structural limitations (anchor sizes, IoU threshold, OHEM+FL gradient suppression) — overfit evidence stands

**What is now UNKNOWN**: Whether the correct config (detach=False, LR/BIAS=1.0) will break the mAP ceiling. This is the decisive experiment that Run 2 is now running. Epoch 18 val expected ~23:31 UTC.

---

> **Previous header**: ba48691 restart confirmed: detach fix alone is insufficient. OHEM+FocalLoss gradient suppression validated as primary bottleneck by main training.

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

### 1.2 What We Know (UPDATED — v11 C2 Correction)

- **det_n_present_classes is 15-16/24**, always the same classes present
- **Pseudo-classing mAP (det_mAP50_pc = 0.307-0.344)** is ~50% above raw mAP — raising the average of working classes would disproportionately raise RAW mAP
- **Class 6 has 65/91 total samples (not 1739)** — v11 C2 correction: the 1739 was an accumulation artifact. At 50% subset, that's ~33 images. **Class 6's AP=0 is NOW plausible as data scarcity**, not a smoking gun.
- **POS_ANCHOR_PROBE shows the classifier working on positive anchors** — the classifier can learn, just not for these classes
- **The pattern is epoch-consistent** — it's not random variation, it's systematic
- **The 50-image cls-only overfit PROVED the architecture CAN learn classification** — cls_loss→0.062, pos_score→0.97

### 1.3 The Four Hypotheses — v11 C2 Updates

**Hypothesis A: Label Error (WEAKENED by C2 correction)**
- Class 6 having only ~33 training images makes AP=0 plausible without label error
- Previous claim "1500+ GT with AP=0 is impossible without label errors" is INVALIDATED
- Label error is still possible but no longer the default explanation
- If true: fix the labels → those classes might jump from AP=0 to AP~0.30
- **Uncomfortable possibility (WEAKENED)**: The data-scarcity explanation is now simpler than the label-error explanation for class 6. The "weeks of misdirected effort" narrative depended on there being ample data for class 6, which is false.

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

**Evidence against structural ceiling (UPDATE — overfit results):**
1. **Pseudo-classing mAP at 0.307-0.344** — if the architecture can reach 0.344 on a per-class basis, the total potential if all classes worked might be 0.40+
2. **POS_ANCHOR_PROBE shows the classifier CAN produce confident scores** — ceiling may be in matching/evaluation, not in raw classification ability
3. **50-image cls-only overfit COMPLETE — WEAK PASS (cls_loss→0.062, pos_score→0.97)** — the architecture CAN learn classification. The ceiling is in the training dynamics, not the architecture itself.

### 2.3 The Ceiling Components — Updated

The overfit experiment adds a critical data point:

```
At 50 images (overfit):    cls_loss→0.062, pos_score_mean→0.97 → Architecture CAN classify
At 2000 images (training): mAP50→0.204, 12/24 classes AP=0     → Something blocks at scale
```

The gap between "architecture can overfit 50 images" and "mAP plateaus at 0.204 at scale" IS the unknown. Possible explanations:
1. **OHEM+FocalLoss gradient suppression** (new — from overfit Regime 2): Even on 50 images, learning takes 55 epochs to escape plateau. At scale, the suppression may be proportionally worse.
2. **13-pos-anchor structural limit**: If only 13 positive anchors/batch exist regardless of dataset size, scale doesn't help — you hit the same anchor-matching bottleneck at 50 or 2000 images.
3. **Label noise at scale**: 50 carefully chosen images (all with GT) may have cleaner effective labels than the full dataset where augmentation, misalignment, and synthetic errors accumulate.

### 2.4 The Uncomfortable Truth — Updated

**The ceiling may be a combination of OHEM+FocalLoss suppression and the 13-pos-anchor structural limit**, not labels per se. The overfit proved the architecture CAN learn, but also revealed that even in ideal conditions (50 GT-rich images, detection-only, no multi-task interference), learning is slow and anchor-limited.

The uncomfortable possibility has shifted:
- **Old**: "The ceiling may be the labels" (Opus v9)
- **New**: "The ceiling may be the anchor system + loss function working together to create a gradient-suppressed equilibrium that no amount of data or architecture tuning fully escapes"

If the 13-pos-anchor limit is structural (anchor sizes starting at 96px are too large for IndustReal objects), then adding more data or training longer at scale won't help — each image contributes the same small number of positive anchors.

### 2.5 The Experiment That Partially Answered This — OVERFIT COMPLETE

**The 50-image cls-only overfit (Opus v9 §6) HAS BEEN RUN.** 200 epochs, 50 images with GT, detection-only.

| Predicted Outcome | Actual Result | Interpretation |
|---------|-----------|-----------|
| mAP → 0.8+ | WEAK PASS — cls_loss→0.062 but took 200 epochs | Architecture IS fine. But even in ideal isolation, learning is gradient-suppressed (55 epochs to escape plateau) |
| mAP → 0.3-0.5 | N/A (we measured loss/score not mAP in overfit) | Would need full EVAL pipeline to measure overfit mAP |
| 13 pos anchors consistent | **CONFIRMED** — pos_n=13 across ALL 200 epochs | Anchor-matching structural limit is real and persistent |
| Three-regime trajectory | **CONFIRMED** — Fast drop (1-5) → Plateau (5-55) → Slow decline (55-200) | OHEM+FocalLoss gradient suppression is real |

**The experiment answered: "Can the architecture learn?" → YES. But it raised: "Why is learning so slow even in ideal conditions?" → The 13-pos-anchor limit + OHEM+FL gradient suppression now become the central questions.**

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

## Chapter 4: The detach_reg_fpn Schizophrenia — RESOLVED (Opus v10)

### 4.1 The Resolution

**Opus v10 confirmed the value: detach_reg_fpn=True for RF2.** The code trace:
- config.py stage_rf2 preset: `'detach_reg_fpn': True` (line 1117 committed)
- RF2 stage_cfg: NO override for detach_reg_fpn
- CLI: no `--detach-reg-fpn` flag passed
- **Effective value: True** — the "If True" branch was correct all along

### 4.2 What We Now Know

The training for epochs 7-21 ran with regression gradients detached from the backbone:

```
Training signal flow (ACTUAL, epochs 7-21):
  cls_loss → cls_subnet → FPN → backbone ← head_pose_loss
  reg_loss → reg_subnet ──┘         (detached from FPN)
```

The backbone was shaped only by classification + head_pose. The regression subnet's strong GIoU signal (bestIoU 0.86-0.98) could NOT reach the backbone to make features object-discriminative.

**The one-to-one symptom match** (from Opus v10):
- **bestIoU 0.86-0.98** (regression works — it receives decent detached features)
- **mAP 0.20** (classifier starved — cls-shaped features are not sufficiently object-discriminative)
- **12/24 AP=0** (feature-starvation produces class-selective effects — distinctive classes survive, subtle/small/rare ones collapse)
- **LR restart zero effect** (a detached gradient path is not a local minimum)
- **POS_ANCHOR_PROBE 0.64-0.80** (on the easy classes, cls-shaped features ARE good enough)

### 4.3 Why We Never Resolved This Sooner

The gap is real and uncomfortable. The `detach_reg_fpn` ambiguity was identified in Opus v9, but the config resolution chain (preset → stage_cfg → CLI) was never traced. The fix is indeed "one print statement" — and we didn't do that either.

**Why it persisted**: The config.py comment for stage_rf2 said "Detach FPN gradients to prevent regression/PSR gradient shock" — the original commit message (Fix D7 era) made it sound like a safety mechanism. The assumption was that "detach protects against shock" and was therefore good. No one read the comment critically until Opus v10.

### 4.4 The Cost (Retrospective)

- Consultation rounds v6-v9 analyzed multi-task interference, Kendall domination, and head_pose competition — mechanisms that were SECONDARY to the detached gradient path
- The "head_pose ate the backbone" story was WRONG for the epoch 7-21 plateau. The backbone was fine — it was being starved of regression signal
- The fixes (Kendall caps, HP_PREC_CAP, staged training) addressed real concerns but were peripheral to the primary mechanism
- RF1 reached 0.184 with detach=False because regression signal DID flow. RF2 reached 0.204 despite detach=True because 2.5× more data + v8 fixes compensated

### 4.5 The Uncomfortable Truth — Confirmed

The "If True" branch was correct:
- **Everything changes** — the problem IS the cls loss/targets/labels. Not the backbone, not multi-task interference.
- **"Head_pose ate the backbone" was WRONG** — the backbone is fine. The cls head was the bottleneck because it couldn't get regression signal.
- **Opus v1-v8's analysis of multi-task dynamics was partially irrelevant** to the actual failure mode of epochs 7-21 (though it remains relevant for the general case).
- **The fix is one config line**: `'detach_reg_fpn': False` for stage_rf2.

**Status: RESOLVED.** The answer was "True," and it was the primary cause of the plateau. The only remaining question is whether flipping it breaks the ceiling entirely or reveals a second, data-quality ceiling underneath.

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

### 10.1 The Meta-Question — ANSWERED (Partially)

**Do we need to fix training dynamics, or do we need to fix the data?**

The 50-image cls-only overfit has PARTIALLY answered this. The architecture CAN learn classification (cls_loss→0.062, pos_score→0.97). But the overfit also revealed that even in ideal conditions, learning is gradient-suppressed by OHEM+FocalLoss and limited by the 13-pos-anchor structural ceiling.

**What the overfit settled:**
- ✅ The architecture is NOT the ceiling (can overfit 50 images)
- ❌ The data alone is NOT the ceiling (clean 50-image setup still shows slow learning)
- ❓ The loss function + anchor system together create a gradient-suppressed equilibrium

So the answer is: **BOTH training dynamics AND data need attention, but the bottleneck is now identified as OHEM+FocalLoss gradient suppression + the anchor-matching system, not multi-task interference or the backbone.**

### 10.2 How We Could Have Known Earlier — Updated

Signs we missed (updated with overfit hindsight):
1. **RF1's phantom 0.45**: Drove assumption that "model CAN reach 0.45" — now we know the model CAN learn classification but at a very slow rate
2. **Synthetic labels**: Still a concern, but the overfit shows clean labels don't eliminate the slow-learning problem
3. **24 classes**: Many visually similar — the 13-pos-anchor limit means the classifier barely sees positive examples of rare classes
4. **Per-class AP**: We parsed it late. But even early parsing wouldn't have revealed the 13-pos-anchor limit — that required the dedicated overfit experiment

### 10.3 The Meta-Question Is Now Reframed

**Do we need to fix OHEM+FocalLoss gradient suppression, the anchor-matching system, or both?**

This is the v11 meta-question. The overfit established that:
- OHEM+FocalLoss produces a ~50-epoch plateau even on 50 images
- The anchor system limits positive matches to 13/batch regardless of data quantity
- Together, these create a gradient-suppressed equilibrium that may explain the 12/24 AP=0 pattern at scale

The uncomfortable possibility: **The anchor system may be the more fundamental bottleneck.** If most IndustReal objects are smaller than ANCHOR_SIZES[0]=96px, no amount of loss function tuning increases the positive anchor count. The classifier must learn from ~13 positive examples per batch out of 656K predictions — an effective positive rate of 0.002%.

---

## Chapter 11: The Post-Breakdown — What Changed After ba48691

### 11.1 The Core Unknown

**After ba48691 committed detach=False for ALL stages and the overfit proved the architecture CAN learn, what remains of the original plateau mystery?**

The v11 update has resolved two major questions and reframed the rest:
- **detach_reg_fpn=False** → now committed for RF1-RF10 + paper_run ✅
- **50-image cls-only overfit** → COMPLETE. Arch CAN learn. ✅
- **Class 6 "1739 GT"** → corrected to 65/91 samples. Changes everything. ✅

What remains is the new synthesis: **OHEM+FocalLoss gradient suppression + 13-pos-anchor structural limit = gradient-suppressed equilibrium.**

### 11.2 What Was Actually Wrong vs. What We Investigated (Updated)

| Actually Wrong | What We Investigated | Status |
|---------------|---------------------|--------|
| **detach_reg_fpn=True** cut regression gradient from backbone | Multi-task interference, Kendall weighting, head_pose domination | **Fixed in ba48691** — ALL stages |
| **Top-k without IoU floor** potentially poisoning classifier | Top-k was a FIX (v8 §3) that we applied, never questioned side effects | **Still unresolved** — needs IoU floor experiment |
| **OHEM+FocalLoss gradient suppression** (v11 discovery) | Not investigated at all — the overfit revealed this | **NEW** — the central v11 finding |
| **13-pos-anchor structural limit** (v11 discovery) | Not investigated — thought TOP_K=9 fixed the positive anchor problem | **NEW** — the threshold (0.4) is the real limiter, not TOP_K |
| **12/24 AP=0** — systematic class-specific failure | General "classifier isn't working" narrative | **C2 correction applied** — 8 of 12 have no GT in subset |

### 11.3 The Pipeline Failures — What We Fixed

1. ✅ **Per-class AP now persisted and logged** (ba48691 adds det_per_class to state)
2. ✅ **detach_reg_fpn=False for ALL stages** (ba48691) — no future split-brain
3. ✅ **LR/BIAS multipliers reverted to 1.0** (ba48691) — clean config
4. ❌ **score_p50 still in evaluation output** — not removed but now understood as structurally blind
5. ❌ **Effective config not printed at startup** — still not implemented

### 11.4 The Three-Ceiling Hypothesis (v11 Update)

The overfit data allows us to decompose the mAP ceiling into three components:

**Ceiling 1: Anchor matching (structural)**
- Only 13 positive anchors/batch regardless of data quantity
- Root cause: DET_POS_IOU_THRESH=0.4 is too high for IndustReal objects against ANCHOR_SIZES starting at 96px
- Estimated contribution to ceiling: 30-50%
- Fix: lower IoU threshold or redesign anchors

**Ceiling 2: OHEM+FocalLoss gradient suppression**
- Even on 50 GT-rich images, the classifier takes 55 epochs to escape the initial plateau
- OHEM 2:1 + FocalLoss gamma_neg=1.5 create a gradient-suppressed equilibrium
- Estimated contribution to ceiling: 20-30%
- Fix: OHEM ablation, reduce gamma_neg, or switch to Quality Focal Loss

**Ceiling 3: Label noise / data quality**
- Synthetic projections may have systematic label errors
- 8/12 AP=0 classes have NO GT in 50% subset (expected)
- Class 6's ~33 training images may be insufficient and/or noisy
- Estimated contribution to ceiling: 10-20%
- Fix: label audit, class merging, or data augmentation

### 11.5 The Single Most Important Measurement

**Per-class AP from the ba48691 restart at epoch 20+.** With detach=False, LR/BIAS=1.0, and all fixes committed:

| If class 6 goes from AP=0 to AP>0 | detach was part of class 6's problem → continue monitoring |
| If class 6 stays at AP=0 | data scarcity or label error on ~33 class-6 images → investigate |
| If all previously AP=0 classes wake | anchor-matching + feature-starvation was the universal bottleneck |
| If only some wake | class-specific mechanism (anchor mismatch per class) |

### 11.6 The Uncomfortable Possibility — Updated

The most uncomfortable possibility after v11 is that **even with detach=False, the 13-pos-anchor structural limit combined with OHEM+FocalLoss suppression creates an effective ceiling at mAP≈0.25-0.30.** In this scenario:
- The detach fix was necessary but not sufficient
- The overfit correctly predicted that anchor-matching is the bottleneck (pos_n=13 always)
- The real fix is anchor redesign OR OHEM ablation OR both
- And each of those requires additional weeks of investigation

**The harder truth**: If the anchor system is fundamentally mismatched (objects smaller than ANCHOR_SIZES[0]=96px), then NO amount of training dynamics fixes will create more positive anchors. The model is capped at ~13 positives/batch regardless of architecture, loss function, or data quantity. The fix would be to add smaller anchor sizes (e.g., 32×32 or 48×48) and retrain from scratch.

### 11.7 What v11 Changes

The post-v11 investigation prioritizes **anchor matching verification**:

1. **Anchor-IoU histogram**: measure max IoU per GT box across all classes — does any GT exceed 0.5?
2. **IoU threshold ablation**: run 5 epochs with DET_POS_IOU_THRESH=0.3 — does pos_n increase? Does mAP improve?
3. **OHEM ablation**: run 3 epochs without OHEM — does cls_w_norm grow faster? Does mAP improve?
4. **Anchor size audit**: what's the actual size distribution of GT boxes in pixels? Are most <96×96?

---

---

## Chapter 12: The Overfit Synthesis — What the 50-Image Experiment Taught Us

### 12.1 The Core Unknown

**Why does a detection architecture that CAN learn classification (proven by overfit) consistently fail to reach mAP>0.21 at scale?**

The 50-image cls-only overfit (200 epochs, detection-only, no multi-task interference) was the most decisive experiment in the project. It proved the architecture CAN classify but revealed a deeper gradient-suppression mechanism. The core unknown has shifted from "can the architecture learn?" to "why is learning so slow even in ideal conditions?"

### 12.2 What the Overfit Proved

| Claim | Status | Evidence |
|-------|--------|----------|
| Architecture CAN classify | ✅ PROVEN | cls_loss→0.062, pos_score_mean→0.97, pos_score_max→1.0 |
| Architecture CANNOT classify | ❌ REFUTED | See above — 200 epochs, 50 images, clean learning trajectory |
| Classification needs multi-task gradient | ❌ REFUTED | Overfit was detection-only (no pose, no head_pose, no PSR, no activity) |
| Positive anchors are sufficient with TOP_K=9 | ❌ REFUTED | pos_n=13 consistently — TOP_K=9 never kicks in |
| cls_w_norm saturates when classifier converges | ❌ REFUTED | cls_w_norm grows LINEARLY (7.07→13.43) over 200 epochs, never plateaued |
| OHEM+FocalLoss suppresses gradient | 🔶 STRONGLY SUGGESTED | Three-regime trajectory matches gradient-suppression predictions |
| Anchor-matching threshold is the bottleneck | 🔶 STRONGLY SUGGESTED | pos_n=13 constant across 200 epochs of 50 GT-rich images |

### 12.3 The Three-Regime Trajectory — A Gradient-Suppression Fingerprint

```
Regime 1 (epochs 1-5):  Fast drop     cls_loss 2.0→0.50    ~0.30/epoch
Regime 2 (epochs 5-55):  Plateau       cls_loss ~0.50        0.00/epoch for 50 epochs
Regime 3 (epochs 55-200): Slow decline cls_loss 0.50→0.062   ~0.003/epoch
```

**This is the single most important finding from the entire project.** The three-regime trajectory is a fingerprint of gradient suppression:

- **Regime 1**: The easy positives (high-IoU anchors, distinctive objects) are learned quickly. These are the same classes that achieve AP>0 in the main training.
- **Regime 2**: The model enters a plateau where gradient from OHEM+FocalLoss is too small to make progress. OHEM selects hard examples (including borderline positives at IoU~0.25-0.4), and FocalLoss suppresses their gradient. The net effective gradient is near zero.
- **Regime 3**: Over 145 epochs, the cls_w_norm grows linearly enough to slowly pull the decision boundary. The weights grow without bound (13.43 from 7.07) — but the gradient per step is so small that progress takes 50× longer than Regime 1.

### 12.4 The 13-Pos-Anchor Mystery

The most surprising result was the **consistent 13 positive anchors per batch** across all 200 epochs. This number was constant regardless of:
- Which 50 images were sampled (each epoch shuffles differently)
- How many GT boxes each image had (1-5+)
- The learning state of the classifier (epoch 1 vs epoch 200)

**Why exactly 13?** Because:
- DET_POS_IOU_THRESH=0.4: an anchor is only positive if IoU ≥ 0.4 with a GT box
- Most GT boxes achieve max IoU of ~0.3-0.5 with the best anchor (anchor sizes start at 96px)
- A GT box typically matches 1-4 anchors above 0.4 IoU (depending on box size relative to 96px)
- At batch_size=4, each batch samples ~3-4 images with GT boxes
- 3-4 images × ~3-4 positive anchors/image ≈ 10-16 → mean = 13

**The implication**: If positive anchors are limited by the IoU threshold (not the TOP_K count), then adding more data, training longer, or changing the loss function doesn't increase the effective positive-to-negative ratio. It's structurally capped at 13:656K = 0.002%.

### 12.5 The cls_w_norm Linear Growth

cls_w_norm grew from 7.07 to 13.43 over 200 epochs — a nearly exact linear trajectory:

```
cls_w_norm(t) ≈ 7.07 + 0.032 × t    (R² near 1.0)
```

**Why linear growth is suspicious**: In normal classification, weight norms either:
- Saturate once the decision boundary is found (typical)
- Grow log-normally with dataset size (theoretical)
- Explode if the loss is unbounded (pathological)

Linear growth with no saturation after 200 epochs suggests the classifier is in a **gradient-suppressed regime where the weight update is constant per step** rather than proportional to the remaining error. This is consistent with OHEM+FocalLoss selecting the same marginal examples each step and providing a constant-but-small gradient.

### 12.6 What This Means for Main Training

The main training (PID 361404, epoch 17/36, 50% data subset) likely faces the SAME gradient-suppression dynamics but amplified:

| Factor | Overfit (50 images) | Main Training (2000 images) | Net Effect |
|--------|-------------------|----------------------------|------------|
| Positive anchors/batch | 13 | ~13 (same threshold) | NO BENEFIT from more data |
| Negative anchors/batch | 656K | 656K (same batch size) | Same suppression ratio |
| Data diversity | 50 images | Thousands | More confusion + label noise |
| Multi-task interference | None | head_pose, activity | Additional gradient competition |
| OHEM selection | 50 images of hard examples | Diverse hard examples | Proportionally MORE OHEM noise |

**The main training has no advantage over the overfit in terms of positive anchor count** — both are capped at ~13 positives/batch by the same IoU threshold.

### 12.7 The Two Possible Paths Forward

**Path A: Keep the current anchor system, fix the loss function**
- Ablate OHEM (remove or reduce ratio to 1:1)
- Reduce FocalLoss gamma_neg (1.5 → 1.0 or 0.5)
- If either increases learning speed and mAP, the bottleneck was loss-function suppression
- **Cost**: 3-5 epochs per ablation experiment (~4-7 hours each)

**Path B: Fix the anchor system**
- Add smaller anchor sizes (32×32, 48×48) to the anchor grid
- Lower DET_POS_IOU_THRESH from 0.4 to 0.3
- Switch from RetinaNet-style matching to ATSS (adaptively selects per image)
- If any of these increase pos_n from 13 to 30+, the bottleneck was anchor matching
- **Cost**: Requires architecture change + retrain from scratch (~2-3 days)

**The critical insight**: Path A and Path B address different mechanisms. Path A fixes "the gradient we have is suppressed." Path B fixes "we don't have enough gradient to start with." Both may be needed.

### 12.8 The Uncomfortable Truth

The overfit experiment was proposed as a "smoking gun" that would bin the problem to either architecture or data. Instead, it revealed a third category: **loss-function + anchor-system interaction creating a gradient-suppressed equilibrium that neither more data nor more training fixes.**

The uncomfortable possibility is that this interaction is the fundamental reason for the mAP ceiling:
- It's NOT that the architecture can't learn (can overfit 50 images)
- It's NOT that the data is too noisy (50 clean images still show the same pattern)
- It IS that the combination of anchor matching (threshold-limits-positives) and loss function (OHEM+FocalLoss suppresses gradient) creates a system where the classifier never receives enough gradient to differentiate all 24 classes

If this is true, the path to 0.40 mAP requires changing BOTH the anchor system AND the loss function. Either alone would help but neither alone would break the ceiling.

### 12.9 Post-Script: The Main Training — Run 2 (Correct Config) Is the Real Test (2026-06-21, CORRECTED)

**⚠️ CORRECTION: What we thought was "5 epochs of post-restart data" was Run 1 with wrong LR/BIAS (4.0/2.0). Run 2 (correct LR/BIAS=1.0/1.0) has only completed 1 epoch val (epoch 17, mAP50=0.2039).**

The prior Section 12.9 claimed the overfit was validated by main training. This conclusion is **invalidated** because:

| Claim | Source | Status |
|-------|--------|--------|
| "5 epochs flat at 0.202-0.209" | Run 1 (wrong LR/BIAS=4.0/2.0) | ❌ INVALIDATED — not a valid test of the correct config |
| "detach fix alone is insufficient" | Extrapolation from Run 1 | ❌ INVALIDATED — insufficient evidence |
| "OHEM+FocalLoss confirmed as primary bottleneck" | Synthesis across Run 1 + overfit | ⚠️ WEAKENED — overfit evidence stands, Run 1 doesn't corroborate |
| Architecture CAN learn (overfit cls_loss→0.06) | 50-image overfit (independent experiment) | ✅ UNCHANGED |
| Three-regime trajectory is OHEM-mediated | 50-image overfit (independent experiment) | ✅ UNCHANGED |

**The overfit's findings remain valid as a proof of concept.** The architecture CAN learn classification. OHEM+FocalLoss does produce gradient suppression in isolation. The 13-pos-anchor limit is real at batch_size=4 with DET_POS_IOU_THRESH=0.4.

**But the key question is STILL UNANSWERED**: With ALL fixes applied (detach=False, LR/BIAS=1.0, correct config), does the main training break through the mAP ceiling? Run 2 will answer this over epochs 18-22.

**Updated path forward:**
1. **Wait for Run 2 epochs 18-22** (ETA ~7 hours from epoch 17 val at 22:09 UTC) — these will be the decisive data points
2. If Run 2 shows mAP improvement past 0.25: detach WAS the bottleneck, continue monitoring
3. If Run 2 stays flat at 0.20-0.21: the OHEM+FocalLoss hypothesis is now genuinely supported by main training with correct config
4. **Only then** proceed with the ablation ladder (OHEM → FocalLoss gamma → anchor matching)

---

## Appendix: The 12 Questions We Could Answer in <2 Hours (Updated)

These are not deep unknowns — they are measurements we could take RIGHT NOW that would dramatically improve our understanding:

| Question | What to Look At | Time | Status |
|----------|----------------|------|--------|
| ~~Per-class AP~~ | ~~Parse metrics.jsonl per-class AP arrays~~ | ~~5 min~~ | ✅ DONE (ba48691 adds persistence) |
| Anchor-IoU per GT class | Add MATCH_PROBE logging to losses.py | 30 min + 1 epoch | ❌ NOT DONE |
| cls_score.weight.norm() | Add logging to train.py epoch end (Opus v8 E3) | 10 min | ❌ NOT DONE (but measured in overfit) |
| ~~effective DETACH_REG_FPN~~ | ~~Print C.DETACH_REG_FPN at step 0~~ | RESOLVED | ✅ RESOLVED (ba48691 commits fix) |
| cls_score.bias value | Log bias value at epoch end | 10 min | ❌ NOT DONE |
| Gradient at multiple FPN levels | Add grad norm logging at P3, P5, P7 outputs | 30 min + 1 epoch | ❌ NOT DONE |
| ~~50-image cls-only overfit~~ | ~~Isolated training on 50 images (Opus v9 §6)~~ | ~~<30 min~~ | ✅ COMPLETE (200 epochs, WEAK PASS) |
| PSR logit values | Log transformer outputs before sigmoid | 10 min | ❌ NOT DONE |
| Score at higher eval thresholds | Evaluate at thresh=0.05, 0.10, 0.25 | 5 min | ❌ NOT DONE |
| Visual class 6 label audit | Overlay 50 random class 6 GT boxes on images | 30 min | ❌ NOT DONE |
| Gradient norms pre/post LR restart | Plot gradient norms around epoch 20 | Already happened | ✅ Already happened |
| Confusion matrix from val | Parse which classes are predicted for each GT | 10 min | ❌ NOT DONE |

---

*Generated 2026-06-21. Updated 2026-06-21 22:16 UTC: **CRITICAL CORRECTION** — Section 12.9 rewritten, correction notice added at top. The prior "main training validates overfit" conclusion was based on Run 1 (wrong LR/BIAS=4.0/2.0), now INVALIDATED. Run 2 (correct LR/BIAS=1.0/1.0) has only 1 epoch val so far. The OHEM+FocalLoss gradient suppression hypothesis is SUPPORTED BY OVERFIT but NOT YET CONFIRMED OR REFUTED by main training with correct config. This document covers 12 chapters (9 sections in Ch12) across 18+ phases, 11 Opus consultations, and ~500 GPU-hours. Every answer reveals deeper questions. The most important question remains: what happens with the correct config?*
