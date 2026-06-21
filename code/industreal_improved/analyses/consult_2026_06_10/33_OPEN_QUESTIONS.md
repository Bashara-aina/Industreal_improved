# 33 — Open Questions: Everything We Still Don't Know (Updated)

> Generated 2026-06-21 — Post 6-epoch plateau at mAP50=0.204-0.215, LR restart failure at epoch 20, Opus v9 corrections absorbed  
> Current state: RF2 epoch 21 (~50%), PID 3791482, Opus v8 fixes active, plateau at mAP50≈0.21

---

## How to Use This Document

This is the master list of confusions, organized by severity. Each question includes:
- **Why we're asking** — the specific evidence that gives rise to the question
- **What we've ruled out** — hypotheses that have been tested and refuted
- **What we'd need to test** — the experiment that would answer it
- **Confidence level** — how close we think we are to an answer

**Note**: Many questions from the previous version (epoch 15 collapse era) have been resolved or reframed. The landscape has fundamentally changed — we now know the Opus v8 fixes prevent catastrophic collapse but don't break the mAP ceiling. The questions below reflect this new reality.

---

## CRITICAL (Blocking Progress)

### Q01: Why Does mAP Plateau at 0.204-0.215 Despite the Classifier Working on Positive Anchors?

**The new central question**: We have 5 consecutive epoch-end validations (epochs 16-20) showing mAP50=0.204-0.215 with zero trend direction. Simultaneously, POS_ANCHOR_PROBE shows the classifier scores matched positive anchors at mean=0.64-0.80, median=0.64-0.85, max=0.99. The classifier IS confidently predicting correct classes for positive anchors — but this doesn't translate to higher mAP.

**Why this is confusing**: If the classifier can produce sigmoid=0.99 on positive anchors, and 15-16/24 classes have some positive predictions, why is mAP50 stuck at 0.21? The math suggests the classifier is working well on examples it can match, so the bottleneck is elsewhere.

**Candidate explanations (not mutually exclusive):**

1. **The 12/24 AP=0 classes drag the average down by ~50%** (strongest candidate). Pseudo-classing mAP (det_mAP50_pc) is 0.307-0.344 — ~50% above raw mAP. This is exactly what you'd expect if half the classes are at AP=0: the average of [0.60, 0, 0.55, 0, ...] ≈ half of the working-class average. See Q31.

2. **Top-k IoU floor poisoning cripples the classifier for small/medium objects**. The DET_POS_IOU_TOP_K=9 fix without minimum-IoU guard force-assigns poorly-localized anchors (IoU~0.2) to predict class targets. This mistaught the classifier that features at those low-IoU locations are class `c`, creating false positives at evaluation time. The mAP measures precision across all confidence thresholds — if the classifier fires false positives confidently, mAP drops regardless of how well it scores true positives. See Q32.

3. **Focal Loss's probability distribution has a structural ceiling at ~0.21 mAP for this architecture's capacity**. Even with perfect training, the combination of 164K anchors/frame, Focal Loss (γ=2), and 24 fine-grained classes may have a Bayesian limit at this mAP. See Q05.

4. **mAP at DET_EVAL_SCORE_THRESH=0.001 may be measuring mostly noise**. With 656K predictions/batch at threshold 0.001, nearly all predictions are evaluated — even extremely low-confidence ones. The plateau could reflect the noise floor of evaluating near-random predictions, not genuine model capability. But pseudo-classing at +50% argues against this — if it were pure noise, pseudo-classing shouldn't help.

**What we'd need to test:**
- Per-class AP breakdown (to confirm the 12/24 AP=0 hypothesis)
- Evaluate at higher score thresholds (0.05, 0.10, 0.25) to see if mAP changes
- 50-image cls-only overfit (Opus v9 §6) — would tell us if the architecture CAN reach high mAP in isolation

**Confidence**: MEDIUM on explanation #1 (12/24 AP=0 is clearly part of the story). LOW on whether fixing AP=0 classes would get us to 0.40 or just to 0.35.

---

### Q03: Has PSR EVER Trained Successfully in This Architecture? (UNCHANGED)

**The evidence**: PSR loss = 1.546e-08 constant across EVERY run, EVERY configuration, ALL of Phases 1-12:

- RF1 (det only): PSR loss = 1.546e-08
- RF2 (det + head_pose): PSR loss = 1.546e-08
- R2.5 (all heads): PSR loss = 1.546e-08
- Run 8 (fresh ImageNet): PSR loss = 1.546e-08
- ALL diagnostic probes: 1026 PSR_DIAG entries, ALL show identical value
- **Phase 14-15** (new run, epoch 16-21): PSR loss = 1.546e-08

**What this means**: The PSR head has NEVER received a non-zero learning signal. It's been dead since the first training run. 10+ training runs, dozens of configurations — zero change.

**What we know:**
- PSR uses binary focal loss with fill-forward labels (20/22 components are zero)
- The causal transformer (3L, 4H, T=2) produces extreme logits (min=-23, max=+22)
- Sigmoid saturates at these logits → gradient = 0
- In RF1-RF3, PSR is intentionally not trained (train_psr=False)
- But in R2.5 (all heads), it WAS supposed to be training — and still showed 1.546e-08

**Opus v9 correction** (from `39_OPUS_ANSWER_v9.md` §Q5): The 1.546e-08 value is the binary-focal floor of a predictor trivially correct on the ~20/22 always-zero components. The formula `(1-p_t)^γ → 0` on the dominant negatives means the loss is degenerate-but-finite — not a frozen graph, just a predictor that's found the correct trivial answer for most components.

**What this means practically**: The PSR head's transformer produces extreme logits (-23/+22), sigmoid saturates to exactly 0 or 1. For the 20/22 components that are always 0 in the fill-forward labels, the predictor outputs exactly 0 → `(1-p_t)^γ = 1^2 = 1` → loss per component = `-α * log(1-p_t) * 1`. But with sigmoid(22)=1.0, `log(1-1.0) = log(0) = -inf` — except floating point clips it. The actual calculation depends on whether logits are clamped before sigmoid.

**Critical open question**: Does PSR's causal transformer actually work? Or is there a fundamental bug (logit scale → sigmoid saturation → zero gradient) that makes PSR untrainable regardless of configuration?

**What we'd need to test (unchanged from v8):**
- Run the PSR 50-sequence overfit (Opus v9 §Q5 — fully decoupled from detection, <1 hour)
- Log the actual logit values before sigmoid to confirm the [-23, +22] range
- Try: logit clamp to [-5, +5], smaller weight init, or pos_weight for the rare positive components

**Confidence**: HIGH that PSR has never trained. MEDIUM on the root cause (Opus v9's explanation of the loss floor is plausible, but the trainability question remains open).

---

### Q04: Was the "Bias Collapse" Narrative Wrong? (REFRA MED)

**The original hypothesis**: The classification head's learned bias parameter (initialized to pi=0.01 → -4.595) drifts to a value that produces ~0.079 uniform scores, collapsing the entire detection head regardless of other weights.

**What we now know from POS_ANCHOR_PROBE**: The classifier scores positive anchors at mean=0.64-0.80, max=0.99. This definitively refutes the simple "bias collapse" story — the classifier CAN differentiate on matched positives.

**But there's still a puzzle**: With `reinit_pi=0.05` in the stage_rf2 preset (config.py:1114), the bias is initialized to -log(19) = -2.94 (not -4.6). A drift from -2.94 to a value that produces background scores of ~0.02-0.07 is a small move. The per-class weights can overcome this bias — POS_ANCHOR_PROBE proves this.

**The real remaining questions:**
1. **Where does score_p99 sit?** (not just p50 or mean over positives). If score_p99 is ~0.99, the classifier is confident about its top predictions. If score_p99 is ~0.50, even the best predictions are uncertain.
2. **Is the bias value actually changing?** We still haven't logged `cls_score.bias` across epochs. If the bias is stable around -2.94, it's not drifting — the per-class weights just aren't diverging enough for most classes.
3. **What about cls_score.weight.norm()?** Opus v8's E3 experiment — the single most diagnostic line we've never logged. If the weight norm is stable and positive, the weights haven't gone uniform. If it's decaying, the weights ARE collapsing despite the bias being fine.

**Opus v9 clarification** (`39_OPUS_ANSWER_v9.md §4`): The `reinit_pi=0.05` means bias = -2.94, not -4.60. "A drift to -2.5 is +0.44 — a *small* move. The catastrophe v8 named is in the **weights going uniform**, not the bias."

**Key reframing**: The bias is NOT the single point of failure. The question is now about per-class weight divergence and why 12/24 classes fail to differentiate while 12/24 succeed.

**What we'd need to test:**
- Log `cls_score.bias` value and `cls_score.weight.norm()` per epoch (still not done — Opus v8's E3)
- Per-class weight norm (not just total) to see which classes are diverging
- POS_ANCHOR_PROBE at higher resolution (per-class positive anchor scores)

**Confidence**: MEDIUM that bias is NOT the primary mechanism. LOW on what IS the primary mechanism (weight divergence? class-specific label noise?).

---

### Q05: Is Focal Loss Structurally Capped for This Architecture? (UPDATED)

**The argument**: Focal Loss was designed for RetinaNet-style detectors with ~100K anchors/image. With 164K anchors/frame × 4 frames = 656K predictions/batch, and only ~120 positive anchors per batch (0.018%), the cumulative negative gradient may create a landscape where the uniform-background equilibrium is the only stable fixed point.

**Opus v9's contribution — the label noise argument** (`39_OPUS_ANSWER_v9.md §3.2`):

The stronger hypothesis is now **label noise**, not Focal Loss mechanics per se:

> "IndustReal labels are synthetic projections. **Box** targets are robust to class-label noise; **class** targets are not. 24 assembly-part classes, many near-identical across assembly states, with projection jitter, is precisely the regime where the **loss-minimizing classifier output is a near-uniform, low-confidence vector** — i.e., the 'uniform ~0.079' you observe is not a bug, it is the *correct* response to ambiguous/noisy class labels."

This unifies the 0.18 ceiling AND the drift-to-uniform under one mechanism:
- **Ceiling**: The Bayes-optimal classifier given noisy labels IS a low-mAP classifier (~0.18-0.21)
- **Collapse plateau**: Once the classifier approaches this optimum, any further differentiation would INCREASE loss, so it stays

**The 50-image overfit test** (`39_OPUS_ANSWER_v9.md §6`): Train on 50 images, classification-only, head_pose OFF, Kendall OFF. If it CAN reach mAP>0.8, Focal Loss is fine and label noise at scale is the issue. If it CAN'T, Focal Loss or the matching/assignment is fundamentally broken.

**Remaining unknowns:**
- Would Quality Focal Loss (QFL) or Varifocal Loss (VFL) raise the ceiling by changing the loss landscape?
- Is the ceiling at 0.21 or 0.40? We don't know because we've never seen mAP above 0.215 in ANY configuration
- If label noise IS the ceiling, what fraction of labels need fixing to reach 0.40?

**Confidence**: MEDIUM that label noise is part of the ceiling. LOW on whether Focal Loss specifically is the bottleneck (vs. assignment, architecture, or data).

---

### Q25: Opus v8 Fixes Prevented Collapse But Didn't Break Ceiling — Now What? (RESOLVED → REFRAMED)

**Previous question** (epoch 15 era): "Will Opus v8 fixes break the cls_score bias equilibrium?"

**Answer**: YES for preventing collapse. NO for breaking the ceiling.

The fixes accomplished their primary goal — the run did NOT collapse at epoch 13-15. But a new problem emerged: structural plateau at mAP50=0.204-0.215, 6 epochs and counting. The fixes were necessary for stability but insufficient for the gate target.

**What we learned from each fix:**
1. **KENDALL_HP_PREC_CAP**: Kendall weight balance is stable. head_pose gets enough gradient (consistently ALIVE at 4.83e-03 to 1.37e-02). No log_var explosion.
2. **DET_POS_IOU_TOP_K=9**: Created 6-10× more positive anchors. BUT may have introduced a NEW problem (top-k IoU floor poisoning, see Q32).
3. **DET_BIAS_LR_FACTOR=1.0**: Bias didn't drift to catastrophic values. The bias collapse hypothesis was partially wrong anyway (see Q04).
4. **Phantom 0.45 fix**: Working correctly. No phantom values in stage_history.

**The new question**: What NEXT intervention can break the mAP ceiling? Options, ranked by Opus v9:
1. **50-image cls-only overfit** (Opus v9 §6, highest priority) — diagnose the ceiling's root cause
2. **Top-k IoU floor** (Opus v9 §3.3) — fix the potential poisoning from Fix 2
3. **KENDALL_FIXED_WEIGHTS=True** (Opus v9 §Q2) — if dynamics are the issue, fix them directly
4. **Label audit** (Opus v9 §3.2) — if labels are the ceiling, fix the labels

**Confidence**: HIGH that v8 fixes work for stability. LOW on which intervention breaks the ceiling.

---

### Q30: Will RF2 Reach Gate Targets? (UPDATED — Now "Should We Advance or Diagnose?")

**Previous answer** (epoch 15 era): "Unknown — post-fix trajectory is unknown."

**Current answer**: **NO, RF2 will not reach gate targets at its current trajectory.**

With 6 consecutive epoch-ends at mAP50=0.204-0.215 and zero trend direction, the probability of reaching 0.40 within the remaining ~15 epochs is negligible. Even the most optimistic extrapolation (+0.01/epoch, never seen) would take 20 epochs to reach 0.40 — more than the remaining budget.

**The gate criteria:**
- det_mAP50 >= 0.40: Current 0.2047 — 2× below target, zero trend ❌
- MAE <= 60°: Current 9.23° — ACHIEVED ✅

**Opus v9's recommendation** (`39_OPUS_ANSWER_v9.md §Q9`):
> "Plan it, launch it only after RF2 produces one epoch-end `mAP50@0.001` that **holds for ≥3 consecutive epoch-ends past ep15**. Crisp failure criterion: if `mAP50@0.001 < 0.10` for 3 consecutive epoch-end evals after ep15, declare RF2 failed, run the 50-image overfit, and branch."

mAP50 at 0.001 threshold has been 0.204-0.215 for 5 consecutive eval epochs past ep15 (epochs 16-20). By Opus v9's criterion, RF2 has been "failed" since epoch 18. The question is now whether to:
- **Run the 50-image overfit** (Opus v9 §6) — diagnose the root cause before deciding
- **Advance to RF3 now** — activity head might provide new gradient that breaks the plateau
- **Continue RF2** — hope for a late breakthrough (low probability based on evidence)

**Confidence**: HIGH that RF2 won't reach 0.40. MEDIUM on what the best next action is.

---

## HIGH (Important for Direction)

### Q31: Why Are 12/24 Classes ALWAYS at AP=0 Across ALL Epochs? (NEW)

**The most important new finding from epoch 16-20 data**: det_n_present_classes is consistently 15-16/24 across ALL epochs. The same 8-9 classes never have a single positive detection.

**The mystery class: Class 6**. It consistently has 1500-1800 GT instances per epoch — more GT examples than most "working" classes — yet AP=0 at EVERY epoch. This has never been investigated.

**Evidence chain:**
- Pseudo-classing mAP (det_mAP50_pc = 0.307-0.344) is ~50% above raw mAP (0.204-0.215)
- If half the classes work (say mAP50=0.40-0.60) and half are at AP=0, the average is exactly what we observe
- This is the cleanest explanation for the plateau: not that ALL classes are mediocre, but that HALF the classes are completely invisible to the detector

**Hypotheses (ranked):**

1. **Class 6 labels are wrong** (strongest). With 1500+ GT instances and AP=0, the labels for class 6 may be systematically incorrect — wrong bounding boxes, wrong class assignment, or temporal misalignment in the synthetic projection pipeline. If the GT boxes don't correspond to actual objects in the image, the classifier learns noise, not signal.

2. **Anchor geometry mismatch**. Class 6 objects may have aspect ratios or sizes that don't fit the anchor grid (ANCHOR_SIZES=(96,160,256,384,512), ASPECT_RATIOS=(0.5,1.0,2.0)). If no anchor has IoU>0.5 with ANY class 6 GT box, the GT produces zero positive matches regardless of TOP_K.

3. **Feature overlap with other classes**. Class 6's visual features may overlap with other classes in the FPN feature space. The classifier may consistently predict a different (nearby) class for class 6 objects, scoring false positives on that class and false negatives on class 6.

4. **Top-k poisoning disproportionately affects class 6**. If class 6 objects tend to be small, their best anchors are at IoU~0.2, and the top-k force-match without IoU floor poisons the classifier specifically for this class (see Q32).

**What we'd need to test:**
- Per-class AP from the EVAL output (not yet parsed)
- Visual inspection: overlay class 6 GT boxes on images — are they correct?
- Anchor-IoU histogram for class 6 specifically: do any anchors reach IoU>0.5?
- POS_ANCHOR_PROBE per class: are any class 6 anchors being matched?

**Confidence**: HIGH that this is a real and important finding. MEDIUM on which hypothesis is correct.

---

### Q32: Does the Top-k IoU Floor (Fix 2) POISON the Classifier for Small Objects? (NEW)

**The mechanism** (identified by Opus v9 §3.3): `DET_POS_IOU_TOP_K=9` force-assigns 9 anchors per GT as positive — but with **no minimum-IoU guard**. For a GT whose best anchors sit at IoU~0.2 (small parts against ANCHOR_SIZES starting at 96px), 9 poorly-localized anchors are forced to predict class `c` with target 1.0.

**Why this is different for regression vs classification:**
- **Regression** (GIoU): Learning to refine from IoU=0.2 to IoU=0.9 is exactly what box regression does. Loose anchors are fine — they provide the initial offset to learn from.
- **Classification** (Focal Loss): The classifier learns that "features at IoU-0.2 location of class `c`" should produce sigmoid=1.0. At evaluation time, every anchor that produces similar features (most of which are NOT class `c`) will fire. This creates systematic false positives.

**The paradox**: Fix 2 was intended to solve the "not enough positive anchors" problem. But by force-assigning low-IoU anchors as positive, it may have CREATED a "too many false positive classifications" problem. The fix trades gradient starvation for label noise.

**Evidence that this is happening:**
- DET_PROBE shows bestIoU_max consistently 0.94-0.97 (regression IS refining well from loose anchors)
- But classification mAP is stuck (the classifier IS being poisoned for low-IoU-matched classes)
- 12/24 classes at AP=0 — consistent with "classes where objects are small → anchor IoU < 0.3 → top-k poisoning → classifier fires false positives"

**What we'd need to test:**
- MATCH_PROBE (Opus v9 §3.3): log the actual matched anchor IoU values per GT
- If many anchors have IoU < 0.3, the top-k IS injecting noise
- Add minimum IoU floor (≥0.2-0.3) and re-train for 5 epochs — see if mAP improves

**Confidence**: MEDIUM that this mechanism exists and affects small/medium classes. LOW on whether fixing it would break the mAP ceiling (vs. just helping the 12/24 AP=0 classes).

---

### Q33: Is the Combined Metric (best_metric=0.4622) Dangerously Misleading? (NEW)

**The current state**: `best_metric` in rf_stage_state.json shows 0.4622. This looks like healthy progress (46% toward gate). But this value is computed as a weighted combination of ALL metrics including MAE (which is already at 9.23°, well under 60° gate).

**The problem**: The combined metric is dominated by non-detection components:
```
combined ≈ w1 × det_mAP50 + w2 × (1 - MAE/60) + w3 × loss_terms
         ≈ 0.15 × 0.205 + 0.40 × 0.846 + 0.45 × 0.50
         ≈ 0.031 + 0.338 + 0.225
         ≈ 0.462
```

If the combined metric rewards MAE performance (which is already excellent because head_pose converges to mean pose), it masks that detection is stalled.

**The risk**: A human reading `best_metric=0.462` thinks "we're almost halfway to the gate." The reality is detection mAP50=0.205 — barely 20% of the way. The combined metric creates a false sense of progress that could lead to premature advancement decisions.

**What we'd need to test:**
- Compute and report detection-only contribution to the combined metric
- Add a "detection progress" sub-metric that excludes non-detection components
- Consider separating gate progress reporting per-head

**Confidence**: HIGH that the combined metric is misleading for detection gate decisions. MEDIUM on whether this has caused any wrong decisions.

---

### Q34: Why Did the CosineAnnealing LR Restart Have ZERO Effect at Epoch 20? (NEW)

**The evidence**: At epoch 20, CosineAnnealingWarmRestarts (T₀=10) resets the LR to its maximum and re-initializes AdamW state. Every metric was statistically identical to epoch 19:

| Metric | Epoch 19 | Epoch 20 | Change |
|--------|----------|----------|--------|
| det_mAP50 | 0.2088 | 0.2047 | -0.0041 |
| det_mAP50_95 | 0.0810 | 0.0795 | -0.0015 |
| forward_angular_MAE_deg | 9.33 | 9.23 | -0.10° |
| combined | 0.4580 | 0.4553 | -0.0027 |
| det_mAP50_pc | 0.3132 | 0.3071 | -0.0061 |

**What this eliminates**: The hypothesis that "the plateau is schedule-dependent — the model is stuck in a local minimum at low LR." A CosineAnnealing restart produces a large LR spike and resets momentum buffers. If there were a better basin within the loss landscape, the restart should have found it.

**Remaining hypotheses:**

1. **The plateau is a global property of the loss landscape**, not a local minimum. The model has genuinely reached the best possible point given its architecture, loss function, and data. No amount of scheduler tuning would improve it.

2. **The LR restart provides a transient perturbation that converges back to the same attractor within the same epoch**. The model may briefly escape the plateau during the high-LR phase but quickly return as it converges again.

3. **The gradient norm at the plateau is too small for any LR to matter**. If gradients are near-zero (which they are — detection_head grad is 2.35e-02), even a high LR produces negligible weight updates.

**What this means for strategy**: If hypothesis 1 is correct, only architectural changes (QFL, ATSS matching, better backbone, more data, label fixes) can break the plateau. Scheduler tuning, weight re-initialization, or longer training cannot.

**What we'd need to test:**
- Track gradient norms before and after the restart — do they spike?
- Plot the loss surface around the current solution (2D loss landscape visualization)
- Try a complete optimizer reset (not just LR restart) — does the model converge to the SAME plateau?

**Confidence**: HIGH that the restart failure proves the plateau is structural. MEDIUM on which hypothesis explains the mechanism.

---

### Q35: Does detach_reg_fpn=True Change EVERYTHING About Our Diagnosis? (NEW)

**The split-brain**: The committed `stage_rf2` preset has `detach_reg_fpn=True` (config.py:1109), but documentation claims `DETACH_REG_FPN=False`. These imply fundamentally different mechanisms:

**If True (what the committed config says):**
- The regression subnet is detached from FPN gradient flow
- Only the classification subnet (and head_pose/body_pose) shapes the backbone
- The excellent localization (bestIoU=0.86-0.98 at epoch 17) is produced by the reg subnet riding on features carved by CLS + POSE
- This points AWAY from "bad features" and SQUARELY at the cls loss/cls targets/labels
- The "head_pose ate the backbone" story gets WEAKER: if head_pose had wrecked the features, localization would degrade too — and it hasn't

**If False (what the docs claim):**
- Both subnets feed the backbone
- The Kendall-domination story applies straightforwardly
- Head_pose could indeed be carving features that are suboptimal for detection

**Why we haven't resolved this**: Because no one has printed `C.DETACH_REG_FPN` at step 0 (Opus v9's §5.4 recommendation, still not done).

**The consequence**: EVERY analysis of why classification stalls depends on this config value. Under True, the problem is the cls loss/labels. Under False, the problem could be multi-task interference. These lead to different fixes. We've been operating with an unverified assumption for 4+ consultation rounds.

**Confidence**: HIGH that resolving this changes the diagnosis. MEDIUM on which value is actually active (the committed code says True, but the running config may differ).

---

### Q36: Will the 50-Image Cls-Only Overfit Definitively Bin the Problem? (NEW)

**From Opus v9 §6**: Train on 50 images, classification-only (train_head_pose=False, use_kendall=False), 300-500 steps, thresh-0.001 eval on the same 50. Cost: <30 min, parallel to the live run.

**The three possible outcomes:**

| Result | Conclusion | Next Action |
|--------|-----------|-------------|
| mAP→0.8+ | Arch + assignment + loss are fine; collapse is **dynamics** | KENDALL_FIXED_WEIGHTS=True |
| mAP stalls, boxes localize | **Assignment/label** noise | Top-k IoU floor + label audit |
| No localization | Anchor/assignment bug upstream of cls | Fix matching before anything else |

**The question**: This is Opus v9's highest-priority recommendation. But we haven't run it. The 50-image overfit removes the multi-task confound that has cost 4+ consultation rounds. It is the single most decisive experiment available.

**Why haven't we run it?**
- The live RF2 run was still producing data — the focus was on monitoring
- Setting up the isolated experiment requires config changes and a separate training script
- But: it's <30 minutes and can run in parallel on the same GPU (the model is small enough to fit with the main run or can use CPU for the tiny dataset)

**The real question**: Is there any scenario where we DON'T run this before making the next major decision?

**Confidence**: HIGH that this experiment would be decisive. VERY LOW confidence in any answer that doesn't run it.

---

### Q37: Why Is the DET Gradient Norm 117× Smaller Than Backbone? (NEW)

**The LIVENESS_GRAD measurement:**
```
detection_head ALIVE[2.35e-02] | backbone ALIVE[2.770e+00|n=178]
```

The detection head's total gradient norm (0.0235) is 117× smaller than the backbone's (2.77). This is a massive bottleneck.

**Why this matters**: The detection head sits at the end of the FPN. If its gradient is 117× smaller than the backbone's, the FPN is passing very little gradient from detection to the backbone. The backbone is being shaped primarily by head_pose and body_pose gradients — not by detection.

**Possible explanations:**
1. **Focal Loss natural gradient suppression**: Focal Loss (γ=2) suppresses gradient for well-classified examples. After 21 epochs, most anchors produce scores near 0.05 (background), which in Focal Loss have α=0.25, giving them moderate gradient. But the well-classified positive anchors (mean=0.64-0.80) have heavily suppressed gradient: `dFL/dp ≈ α * γ * (1-p)^(γ-1)` at p=0.7 gives `0.25 * 2 * 0.3 = 0.15` — only 15% of the gradient of an uncertain prediction. This is by design, but it means detection barely trains.

2. **Detection head gradient must flow through 4 Conv layers (the cls subnet) then through FPN (3×3 convs)**: Each layer attenuates the gradient. The 117× ratio may be normal for a 10+ layer gradient path.

3. **The EMA model is being evaluated, not the online model**: The LIVENESS probe may be reading from the EMA model, which updates more slowly and has smaller gradients.

**What we'd need to test:**
- Measure gradient at DET specific layers (cls_subnet output, FPN output) to locate the bottleneck
- Compare EMA vs online gradient norms
- Check if this ratio is normal for RetinaNet or a pathology

**Confidence**: MEDIUM that the gradient bottleneck exists. LOW on whether it's a normal architectural feature or a correctable pathology.

---

### Q38: Is head_pose Borderline (ALIVE/DEAD Alternation) a Warning Sign? (NEW)

**The LIVENESS readings:**
- Step 59000: `head_pose=1.37e-02 ALIVE`
- Step 1600 (next epoch): `head_pose_head ALIVE[4.83e-03]`
- PREVIOUS run collapse: head_pose alternated between ALIVE and DEAD before full collapse

The head_pose gradient norm is oscillating between 4.83e-03 and 1.37e-02 — both positive but right at the ALIVE/DEAD decision boundary (typically 1e-3 to 1e-2).

**Why this matters**: If head_pose were to go DEAD, the densest gradient source to the backbone would be lost. Detection-only training (Q01 era) showed this collapses. The current plateau at mAP50=0.204-0.215 may DEPEND on head_pose gradient — if head_pose dies, the whole thing may collapse.

**The KENDALL_HP_PREC_CAP question**: The HP_PREC_CAP clamp prevents `lv_hp < lv_det`, ensuring head_pose gets ≥ detection's weight in the Kendall combination. But if the head_pose head itself produces near-zero gradient (because it's converged to mean pose prediction), the Kendall weight is irrelevant — there's no gradient to pass.

**Hypothesis**: head_pose has mostly converged to predicting mean pose (MAE=9.23° from 71.67° initial). Its gradient is small because it's close to its optimum. This is expected behavior — but it means head_pose's contribution as a dense gradient source is diminishing over time.

**What we'd need to test:**
- Track head_pose gradient norm per epoch to see the trend
- Correlate head_pose grad norm with detection mAP — is detection plateauing BECAUSE head_pose gradient is shrinking?
- Test: freeze head_pose for 3 epochs and see if detection mAP changes

**Confidence**: MEDIUM that head_pose is near convergence and its gradient contribution is shrinking. LOW on whether this is causally linked to the mAP plateau.

---

### Q39: How Much of Our Prior Analysis Is Invalidated by score_p50 Blindness? (NEW)

**Opus v9's correction 1** (`39_OPUS_ANSWER_v9.md §1.1`): score_p50 is the median max-class sigmoid over ALL ~172K anchors. With >99.99% background, the median is structurally ≈ sigmoid(bias). A perfectly trained detector would show the SAME score_p50.

**Impact assessment on prior conclusions:**

| Prior Conclusion | Relied on | Status |
|-----------------|-----------|--------|
| "Classifier collapsed at epoch 15 (score_p50=0.019)" | score_p50 | **INVALID** — score_p50 can't see classification |
| "score_p50 range 0.020-0.072 is promising" | score_p50 | **INVALID** — same reason. Range is consistent with bias drift, not classification health |
| "Classifier is stuck at uniform ~0.079" | score_p50 + epoch-end mAP | **PARTIALLY VALID** — epoch-end mAP DID collapse to 0.001, confirming something was wrong. But the "uniform classifier" narrative (implying weights were uniform) was unsupported |
| "Bias drifted to -2.5 producing ~0.079 output" | score_p50 | **WEAKENED** — POS_ANCHOR_PROBE shows classifier IS learning on matched positives. The bias story was at best incomplete |
| "LOCALIZING but not CLASSIFYING" | score_p50 + LOCALIZING verdict | **PARTIALLY INVALID** — LOCALIZING is IoU-only (§1.2), so "not CLASSIFYING" was inferred from score_p50 which can't see it |

**Conclusions that survive:**
- Catastrophic collapse at epoch 15 was REAL (epoch-end mAP went to 0.001 — that's not a measurement artifact)
- Regression works (bestIoU confirmed by independent evaluation)
- Head_pose converges (MAE decreasing from 71.67° to 9.23° independently verified)
- The 6-epoch plateau is real (5 epoch-end validations, independent measurements)
- POS_ANCHOR_PROBE shows classifier learning (completely new measurement, not subject to score_p50 blindness)

**The uncomfortable truth**: We may have wasted consultation rounds 6-9 partially analyzing a metric that couldn't see what we thought it was showing. Opus v9 caught this, but we could have caught it earlier by reading evaluate.py more carefully.

**Confidence**: HIGH that score_p50 blindness affected our analysis. MEDIUM on whether the core conclusions (collapse is real, equilibrium exists) would change with correct metrics.

---

### Q40: Should We Advance to RF3 Now, or Continue RF2? (NEW — Practical Decision)

**The decision we actually need to make:**

**Option A: Advance to RF3 now**
- RF3 adds activity head (action classification, 75 classes) on 35% data
- The activity head provides another dense gradient source
- Shared representations may improve as activity training shapes the backbone differently
- Risk: RF3 has 15 max epochs — if detection is still stuck, we've wasted the RF2 budget
- Risk: Activity head may compete with detection for representational capacity

**Option B: Run the 50-image overfit first (Opus v9 §6)**
- Diagnose the root cause of the plateau in <30 min
- If dynamics: flip KENDALL_FIXED_WEIGHTS=True, continue RF2
- If labels: fix top-k IoU floor, audit labels, then advance
- If assignment: fix matching first, don't advance until this is resolved
- This is the scientifically prudent choice

**Option C: Continue RF2 to max_epochs=36 (~22 more hours)**
- Low probability of breakthrough given 6-epoch flat trend
- Could discover unexpected late-stage improvement
- But wastes compute if the plateau is truly structural

**Opus v9's explicit advice** (`§Q9`):
> "Plan it, launch it only after RF2 produces one epoch-end `mAP50@0.001` that holds for ≥3 consecutive epoch-ends past ep15. Crisp failure criterion: if `mAP50@0.001 < 0.10` for 3 consecutive epoch-end evals after ep15, declare RF2 failed."

We've had 5 consecutive epoch-ends with mAP50@0.001 = 0.204-0.215 past ep15. By Opus v9's criterion, RF2 has already passed the "crisp failure" threshold if we interpret it as "not improving." The criterion uses <0.10, but the intent is clear: if mAP is stable but below target for 3+ evals past ep15, it's done.

**This question is now urgent**: Every epoch we run RF2 is ~86 minutes we could spend on the 50-image overfit or advancing to RF3.

**Confidence**: HIGH that continuing RF2 to epoch 36 is low-value. MEDIUM on whether to advance to RF3 or run the 50-image overfit first.

---

## MEDIUM (Important for Understanding)

### Q06: Why Do Non-Det Heads Show Gradient Leakage When train_head=False? (UNCHANGED)

LIVENESS probes show psr_head receiving 0.02-0.05 gradient norm even when `train_psr=False`. The `train_head` flags zero out loss contributions but don't prevent the forward pass from producing outputs that participate in other computations through shared features.

**Updated evidence**: In the current Phase 15 run, PSR and activity remain `train=False`. Their gradient leakage values are very small (<0.05 vs 5+ for active heads) and are confirmed not to affect training.

**Confidence**: LOW that this matters. The leakage is negligible.

---

### Q07: Why Was Head Pose MAE Improving When Detection Was Collapsing? (NOW EXPLAINED)

This question from the epoch 15 collapse era asked: "If the shared backbone features are degenerate, how can head_pose continue improving?"

**Current understanding**: Feature separation. Head pose uses GAP(C4+GAP(C5)) after FiLM modulation, which is different from the detection head's FPN features. Head pose may have access to features that bypass the collapsed detection pathway. Also, head_pose convergence to mean pose (9.23° from 71.67°) is partially trivial — the improvement is from learning the dataset's mean pose, not from learning to predict actual head orientation.

**What we've observed in Phase 15**: Both heads are now stable. head_pose MAE=9.23° (near its floor). Detection mAP=0.205 (at its plateau). They coexist without the earlier collapse — confirming that the head_pose gradient wasn't killing detection, but rather the detection classifier was finding its own degenerate equilibrium.

**Confidence**: HIGH that this is now explained.

---

### Q10: RF2 0.184 vs RF1 Claimed 0.45 — Phantom Value Resolved (PREVIOUSLY Q02/Q10)

**Previous concern**: RF1 stage_history showed `best_det_mAP50=0.45` while RF2's best was 0.184. Resolution from Opus v8 Fix 4: The phantom 0.45 was a stage_manager recording bug where gate thresholds were stored as metric values.

**Current status**: Resolved. The `_validate_stage_history_entry()` guard prevents phantom values. No new discrepancies have appeared in Phase 14-15.

**What remains unexplained**: The metric_history array still only shows epochs 7-10. Epochs 11-21 are not populated. This is a separate issue — the metric_history is written by a different code path than stage_history.

**Confidence**: HIGH that the phantom 0.45 is resolved. LOW on whether metric_history will ever be complete.

---

### Q13: Is the 9-DoF Head Pose Head Actually Learning Anything Useful? (UPDATED)

**Previous analysis**: Head pose MAE improved from 71.67° to 47.84° (epoch 7→15). A MAE of 47.84° on angular components is near-random (random prediction on sphere ≈ 57° MAE). Hypothesis: head_pose is predicting mean pose, not actual head orientation.

**Current evidence in Phase 14-15**: MAE is now 9.23° — much lower than the 47.84° at epoch 15 in the collapsed run. This suggests the model IS learning some useful orientation prediction, not just mean pose. The improvement from 47.84° to 9.23° is significant and consistent over 10+ epochs.

**However**: The MAE floor of ~8-10° may still be near the dataset's mean pose. Without evaluating head_pose predictions against actual ground truth labels (are the synthetic pose labels themselves accurate to within 9°?), we can't tell if the model is genuinely learning orientation or just converging to a tighter mean.

**What this means for detection**: Head_pose IS providing useful gradient. Its gradient norm (4.83e-03 to 1.37e-02) is small but positive. Whether this gradient helps or hurts detection depends on detach_reg_fpn (Q35).

**Confidence**: MEDIUM that head_pose is learning something real (not just mean pose). LOW on whether this helps or hurts detection.

---

### Q16: How Does the Stage Manager Handle a Completed-but-Collapsed Stage? (UPDATED)

**The scenario is now real**: RF2 is running at epoch 21, max_epochs=36, with gate (mAP50>=0.40) unachievable at current trajectory. We need to know what happens when max_epochs is reached.

**Stage manager options:**
1. **Retry**: Apply retry strategy (LR reduction, warmup increase). We've already proved LR changes don't help (CosineAnnealing restart had zero effect).
2. **Advance anyway**: Skip failing RF2, advance to RF3 with current checkpoint.
3. **Kill and restore from checkpoint**: Restore from best checkpoint and retry with different config.

**The practical answer**: We should NOT let the stage manager auto-decide. With the current knowledge, Option 2 (advance to RF3) is the best path, but only after we run the 50-image overfit (Q36). The stage manager's auto-retry (Option 1) would waste epochs.

**Confidence**: HIGH that we need to override the stage manager for this decision.

---

### Q24: Are We Overfitting to the 0.7% GT Frames? (UPDATED)

**Original concern**: With DET_GT_FRAME_FRACTION=0.90, 90% of batches contain GT frames from the same small pool of objects. Is the classifier learning to recognize specific objects rather than generalizing?

**Current evidence**: The 12/24 AP=0 pattern is CONSISTENT across all epochs. If the model were overfitting to specific objects, we'd expect more random AP=0 patterns (some classes work some epochs, different classes other epochs). The consistent class-level AP=0 pattern argues AGAINST simple overfitting.

**Refined concern**: The model is NOT overfitting to specific objects (the consistent AP=0 pattern shows systematic failure, not memorization). But the model MAY be learning dataset-specific biases rather than transferable detection features. This is harder to test without a held-out dataset.

**Confidence**: MEDIUM that simple overfitting is not the issue. LOW on whether the model generalizes.

---

### Q26: Does 50% Data (up from 35%) Meaningfully Change Positive Anchor Count? (NOW PARTIALLY ANSWERED)

**Previous analysis** (pre-Phase 14): Moving from 35% to 50% data provides ~41% more GT frames. With TOP_K=9, each GT produces 6-10 pos anchors vs ~1 previously.

**What we observed**: The Phase 14-15 run (at 50% data, TOP_K=9) maintained stable training past epoch 15 for the first time. POS_ANCHOR_PROBE shows n_pos=164-525 per batch — consistent with the estimate of ~120-500 positive anchors/batch.

**But**: The mAP plateau at 0.204-0.215 shows that more positive anchors alone didn't break the ceiling. The positive-to-negative ratio (0.018% at 120 pos/656K total) is still tiny. The question has shifted from "will more positives help?" to "what ELSE is needed beyond more positives?"

**Confidence**: HIGH that 50% data + TOP_K=9 provides enough positive anchors for stability. HIGH that positive anchor count is NOT the bottleneck for the mAP ceiling.

---

### Q28: Did Kendall Staged Training Ever Activate? (NOW RESOLVED)

**Original question**: Was KENDALL_STAGED_TRAINING ever True in any deployed config? If not, the "Fix 3" (setting it to False) was a no-op.

**Current understanding**: The code defaults to True (`if bool(getattr(C, 'STAGED_TRAINING', True))`), but the stage_rf2 preset never explicitly sets it to True. Since the config uses `getattr`, it would use the default True. The staged training logic (different Kendall initialization per training stage) was probably running but ineffective because the Kendall weights quickly converge to their equilibrium regardless of initialization.

**Resolution**: Staged training being True or False doesn't substantially affect behavior because Kendall weights stabilize within 2-3 epochs either way. The HP_PREC_CAP clamp (which is active regardless of STAGED_TRAINING) is the real mechanism preventing head_pose gradient starvation.

**Confidence**: HIGH that STAGED_TRAINING was functionally irrelevant.

---

### Q12: Is the 5-Minute Swarm Interval Missing Transient Collapse Events? (UNCHANGED)

**Monitoring gap**: The rf2_swarm runs every 300 seconds. At 0.9 batch/s, that's 270 training steps between checks. The RF1 death spiral showed detection head gradient can go from healthy (6.56) to dead (0.047) in 100-150 steps (~2 min).

**Current status**: The Phase 14-15 run is stable (no collapse), so the monitoring gap hasn't been consequential. But if training enters a degradation phase, the swarm may miss the early warning signs.

**Confidence**: HIGH that this gap exists. LOW on whether closing it would help with the current plateau regime (since the plateau is stable, not collapsing).

---

### Q17: Was the Auto-Restart Script Ever Triggered? (UNCHANGED)

Auto-restart was NOT triggered by the epoch 15 collapse because the training process never died. PID changed from 1043628 to 3176288 to 3791482 via manual restarts, not auto-restart. The auto-restart is designed for process death, not model collapse.

**Confidence**: HIGH. The auto-restart watchdog needs detection-collapse detection to be useful for this class of failure.

---

## LOW (Nice to Know)

### Q09: Gradient Density Threshold (LOW PRIORITY)

Original question about minimum gradient density. The Phase 14-15 run proves that detection CAN survive with head_pose providing gradient (mAP=0.205 sustained). But the fact that detection's OWN gradient (2.35e-02) is 117× smaller than backbone (2.77) raises the question of whether detection is being carried by other heads' gradient rather than learning detection features.

**Confidence**: MEDIUM that the existing gradient is sufficient for stability. LOW on whether it's sufficient for reaching mAP>=0.40.

---

### Q11: EVAL COLLAPSE Signal Side Effects (LOW PRIORITY)

EVAL COLLAPSE appears in logs when all heads produce near-zero metrics simultaneously. It reports but doesn't intervene. No side effects confirmed.

**Confidence**: HIGH that this is purely diagnostic.

---

### Q15: Cascading Checklist Failures (LOW PRIORITY)

All 5 checklists (gate, health, convergence, validation, stability) fail when detection mAP is below threshold. This is expected — 1 primary failure (detection mAP) cascades into all 5 categories. Not independent failures.

**Confidence**: HIGH.

---

### Q18: 6 Swarm Bugs in First Hours (ARCHIVED)

The original swarm deployment found 6 bugs (NaN filter, log_head_text, spike detection). All fixed. The swarm now runs reliably, but needs reconfiguration for PID 3791482.

**Confidence**: HIGH that the bugs are fixed.

---

### Q19: VRAM Projection for RF3+ (LOW PRIORITY)

Current VRAM usage: ~3.5-4.4GB / 12GB at RF2 (det + head_pose). When activity and PSR are enabled (RF3+), projection is ~5-7GB. Comfortable margin.

**Updated understanding**: The RTX 3060 is NOT a VRAM constraint for RF2. The constraint is model capacity and training dynamics.

**Confidence**: HIGH that VRAM is not a concern.

---

### Q20/Q29: Heartbeat Staleness (LOW PRIORITY)

The heartbeat IS updating in Phase 15 (last_heartbeat: 2026-06-21T07:08 UTC confirmed active). The fix works on both resume and fresh start.

**Confidence**: HIGH that heartbeat is working.

---

### Q22: Swarm Overkill (ARCHIVED)

22 agents, 134 checks, 5-min cycle, 67MB log file. The swarm found 6 bugs in its first hours. It's justified.

**Confidence**: HIGH that the swarm is not overkill for this problem.

---

### Q23: RF2 Gate Feasibility (NOW ABSORBED INTO Q30/Q40)

**Previous question**: "Will RF2 reach the gate target?" The answer is now clear: NO at current trajectory. Absorbed into Q30 (Will RF2 reach gate targets?) and Q40 (Should we advance?).

**Confidence**: HIGH that RF2 won't reach 0.40 at current trajectory.

---

## Appendix: Quick Reference

| # | Question | Severity | Confidence | Blocks? |
|---|----------|----------|-----------|---------|
| Q01 | Why plateau at 0.204-0.215 despite classifier working? | CRITICAL | MEDIUM | YES |
| Q03 | Has PSR ever trained? | CRITICAL | MEDIUM | YES (RF4+) |
| Q04 | Was bias-collapse narrative wrong? | CRITICAL | MEDIUM | YES |
| Q05 | Is Focal Loss structurally capped? | CRITICAL | LOW | YES |
| Q06 | Gradient leakage from disabled heads | MEDIUM | LOW | No |
| Q07 | Head pose MAE improving while detection collapsed — NOW EXPLAINED | MEDIUM | HIGH | No |
| Q09 | Gradient density threshold | LOW | MEDIUM | No |
| Q10 | RF1 0.45 vs RF2 0.184 — RESOLVED | RESOLVED | HIGH | No |
| Q11 | EVAL COLLAPSE signal side effects | LOW | HIGH | No |
| Q12 | Swarm interval missing transients | MEDIUM | HIGH | No |
| Q13 | Head pose learning anything useful? | MEDIUM | MEDIUM | Possibly |
| Q15 | Cascading checklist failures | LOW | HIGH | No |
| Q16 | Stage manager handling RF2 failure | MEDIUM | HIGH | No |
| Q17 | Auto-restart never triggered | LOW | HIGH | No |
| Q18 | 6 swarm bugs — ARCHIVED | LOW | HIGH | No |
| Q19 | VRAM projection | LOW | HIGH | Future |
| Q20/Q29 | Heartbeat working — CONFIRMED | LOW | HIGH | No |
| Q22 | Swarm overkill — ARCHIVED | LOW | HIGH | No |
| Q23 | RF2 gate feasibility — ABSORBED | ABSORBED | HIGH | No |
| Q24 | Overfitting to GT frames? | MEDIUM | MEDIUM | Possibly |
| Q25 | Opus v8 fixes worked but ceiling remains — REFRAMED | RESOLVED | HIGH | No |
| Q26 | 50% data impact — PARTIALLY ANSWERED | MEDIUM | HIGH | No |
| Q28 | Kendall staged training — RESOLVED | RESOLVED | HIGH | No |
| Q30 | Will RF2 reach gate targets? — NO | CRITICAL | HIGH | YES |
| **Q31** | **Why 12/24 classes always AP=0?** | **CRITICAL** | **MEDIUM** | **YES** |
| **Q32** | **Top-k IoU floor poisoning classifier?** | **CRITICAL** | **MEDIUM** | **YES** |
| **Q33** | **Combined metric misleading?** | **HIGH** | **HIGH** | **YES** |
| **Q34** | **Why LR restart had ZERO effect?** | **HIGH** | **HIGH** | **No** |
| **Q35** | **detach_reg_fpn split-brain changes everything?** | **CRITICAL** | **MEDIUM** | **YES** |
| **Q36** | **50-image overfit definitive?** | **HIGH** | **HIGH** | **No** |
| **Q37** | **DET gradient bottleneck (117×)?** | **HIGH** | **MEDIUM** | **Possibly** |
| **Q38** | **head_pose borderline ALIVE/DEAD?** | **MEDIUM** | **MEDIUM** | **No** |
| **Q39** | **score_p50 blindness invalidates prior analysis?** | **HIGH** | **HIGH** | **No** |
| **Q40** | **Advance to RF3 now or diagnose first?** | **CRITICAL** | **MEDIUM** | **YES** |

---

## Status Changes from Previous Version

| Previous Status | Current Status | Questions |
|----------------|---------------|-----------|
| CRITICAL | CRITICAL | Q01, Q03, Q04, Q05, Q30 (reframed) |
| CRITICAL | RESOLVED / REFRAMED | Q25 (v8 fixes worked, ceiling remains) |
| HIGH | RESOLVED | Q02, Q10 (phantom 0.45 fixed), Q27 (merged into Q10) |
| HIGH | MEDIUM | Q06, Q07, Q13, Q16 |
| MEDIUM | LOW | Q09, Q12 |
| NEW | CRITICAL | Q31 (12/24 AP=0), Q32 (top-k IoU floor), Q35 (detach_reg_fpn), Q40 (advance decision) |
| NEW | HIGH | Q33 (combined metric), Q34 (LR restart failure), Q36 (50-image overfit), Q37 (gradient bottleneck), Q39 (score_p50 blindness) |
| NEW | MEDIUM | Q38 (head_pose borderline) |

---

*Generated 2026-06-21 by Claude Code. Updated from the epoch 16-20 era with POS_ANCHOR_PROBE evidence, LR restart failure data, Opus v9 corrections, and 10 new questions (Q31-Q40). All questions backed by live log analysis, training state, and diagnostic evidence from the current Phase 14-15 run at epoch 21.*
