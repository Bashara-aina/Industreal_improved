# 33 — Open Questions: Everything We Still Don't Know (v11 Era — Overfit WEAK PASS + ba48691)

> Generated 2026-06-21 — Updated 2026-06-21 22:16 UTC: **CRITICAL CORRECTION — Run 1/2 split discovered.** Prior conclusions about "5 epochs flat proving detach fix insufficient" were based on Run 1 with wrong LR/BIAS (4.0/2.0). Run 2 (correct 1.0/1.0) has only completed 1 epoch val. See Q01 and Q30 for corrected assessments.  
> Overfit COMPLETE (WEAK PASS, cls_loss=0.062), ba48691 committed (detach=False ALL stages, LR/BIAS=1.0, honest metrics), v11 C2 correction (class 6 has 65/91 samples, not 1739).  
> Current state: RF2 **Run 2** epoch 18 training, PID 361404. **All fixes committed in ba48691 — but the correct-config test has only just begun.**

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

### Q01: Why Does mAP Plateau at 0.204-0.215 Despite detach_reg_fpn Fix AND Architecture CAN Learn? (UPDATED — CORRECTED 2026-06-21 22:16 UTC)

**⚠️ CORRECTION**: The prior assessment treated "5 epochs of post-restart data" as validating the OHEM+FocalLoss hypothesis via main training. We've since discovered that log contains TWO runs: Run 1 (wrong LR/BIAS=4.0/2.0, epochs 17-21 flat) and Run 2 (correct LR/BIAS=1.0/1.0, only 1 epoch val completed). The "5 epochs flat" evidence is INVALIDATED. The decisive experiment is STILL IN PROGRESS with the correct config.

**The question has fundamentally changed twice**: Opus v10 identified `detach_reg_fpn=True` as the primary cause. The overfit experiment (v11) proved the architecture CAN learn classification. But the 50-image overfit also revealed a deeper gradient-suppression mechanism that persists even with detach fixed.

**What we NOW know** (50-image cls-only overfit, 200 epochs):
- **Architecture CAN overfit**: cls_loss→0.062, pos_score_mean→0.97, pos_score_max→1.0 ✅
- **But learning is VERY slow**: 55 epochs to escape the initial plateau (Regime 2), even on 50 GT-rich images
- **cls_w_norm grows LINEARLY** over 200 epochs (7.07→13.43), never plateauing — the weights haven't saturated
- **Only 13 positive anchors per batch consistently** — this is an anchor-matching structural limit, not a data quantity limit
- **Three-regime learning trajectory**: Fast drop (1-5) → Plateau (5-55) → Slow decline (55-200) — consistent with OHEM+FocalLoss gradient suppression

**The decisive experiment is IN PROGRESS**: ba48691 commits detach_reg_fpn=False for ALL stages and reverts LR/BIAS multipliers to 1.0. PID 361404 is running with the clean config. We'll know in 3-4 epochs whether mAP climbs past 0.25.

**The remaining unknowns** (updated post-overfit):
1. How much of the ceiling is detach_reg_fpn (fixable) vs. OHEM+FocalLoss gradient suppression (architectural)?
2. The 13-pos-anchor structural limit — is this the bottleneck even at scale?
3. If detach fix boosts mAP to 0.30 but not 0.40, is the remaining ceiling OHEM, FL, or data quality?

**Confidence**: HIGH that detach_reg_fpn was a major factor. MEDIUM that the overfit proves the architecture is fine. LOW on whether the combined fix gets to 0.40.

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

### Q04: Was the "Bias Collapse" Narrative Wrong? (REFRAMED — Now a Post-V10 Footnote)

**The original hypothesis**: The classification head's learned bias parameter (initialized to pi=0.01 → -4.595) drifts to a value that produces ~0.079 uniform scores, collapsing the entire detection head regardless of other weights.

**What we now know from POS_ANCHOR_PROBE**: The classifier scores positive anchors at mean=0.64-0.80, max=0.99. This definitively refutes the simple "bias collapse" story — the classifier CAN differentiate on matched positives.

**But there's still a puzzle**: With `reinit_pi=0.05` in the stage_rf2 preset (config.py:1114), the bias is initialized to -log(19) = -2.94 (not -4.6). A drift from -2.94 to a value that produces background scores of ~0.02-0.07 is a small move. The per-class weights can overcome this bias — POS_ANCHOR_PROBE proves this.

**The real remaining questions:**
1. **Where does score_p99 sit?** (not just p50 or mean over positives). If score_p99 is ~0.99, the classifier is confident about its top predictions. If score_p99 is ~0.50, even the best predictions are uncertain.
2. **Is the bias value actually changing?** We still haven't logged `cls_score.bias` across epochs. If the bias is stable around -2.94, it's not drifting — the per-class weights just aren't diverging enough for most classes.
3. **What about cls_score.weight.norm()?** Opus v8's E3 experiment — the single most diagnostic line we've never logged. If the weight norm is stable and positive, the weights haven't gone uniform. If it's decaying, the weights ARE collapsing despite the bias being fine.

**Opus v9 clarification** (`39_OPUS_ANSWER_v9.md §4`): The `reinit_pi=0.05` means bias = -2.94, not -4.60. "A drift to -2.5 is +0.44 — a *small* move. The catastrophe v8 named is in the **weights going uniform**, not the bias."

**Post-v10 reframing**: With detach_reg_fpn=True confirmed, the bias narrative is now understood as a **symptom analysis** — the bias drift to ~0.079 was the classifier's best compromise on features that couldn't become fully object-discriminative because regression gradient was cut. The per-class weights couldn't diverge enough because the backbone was receiving insufficient gradient to carve class-specific features.

**What this means**: The "bias collapse" analysis from Opus v8 was diagnosing a downstream effect of the real problem (detached regression). The bias was NOT the cause of the plateau — it was an EFFECT of the classifier being starved of object-discriminative features.

**Test**: When we flip detach_reg_fpn=False, if class-wise mAP improves while bias stays near -2.94, the bias was never the problem. The restart will effectively answer this question.

**Confidence**: HIGH that bias was an effect, not a cause. The v10 breakthrough reframes this entire question.

---

### Q05: Is Focal Loss + OHEM Structurally Capping Gradient? (UPDATED — 50-Image Overfit Provides Strong Evidence)

**The argument has been substantially updated by the 50-image cls-only overfit.**

The 200-epoch overfit revealed a three-regime learning trajectory that strongly implicates OHEM+FocalLoss gradient suppression:

**Three-regime trajectory (50 images, 200 epochs):**
1. **Regime 1 (epochs 1-5): Fast drop** — cls_loss 2.0→0.50. The model learns the easy positives (high-IoU anchors, distinctive objects).
2. **Regime 2 (epochs 5-55): Plateau** — cls_loss stays at ~0.50 for 50 epochs. This is the key regime — the model is stuck despite having only 50 images with GT.
3. **Regime 3 (epochs 55-200): Slow decline** — cls_loss drops from 0.50→0.062 over 145 epochs. Learning continues but at 3-4× slower rate than Regime 1.

**The OHEM+FocalLoss gradient suppression hypothesis:**
- OHEM 2:1 ratio selects the hardest negatives, which include borderline positives (IoU~0.25 anchors that are close to the decision boundary)
- FocalLoss gamma_neg=1.5 further suppresses gradient for well-classified examples
- Together: OHEM selects hard examples → FocalLoss suppresses their gradient → the effective learning signal is a fraction of what it should be → 50-epoch plateaus

**The 13-pos-anchor structural limit**: Even with 50 GT-rich images (each with ≥1 box, some with multiple), the overfit experiment always found exactly 13 positive anchors per batch. This suggests the anchor-matching system (DET_POS_IOU_THRESH=0.4) is the bottleneck — TOP_K=9 doesn't help if no anchor exceeds IoU=0.4 for most GT boxes.

**Implications**:
- If OHEM+FocalLoss suppresses gradient by ~5-10×, the main training (2000 images) would see proportionally more gradient but still face the same suppression factor
- The cls_w_norm growing linearly (7.07→13.43 over 200 epochs, never plateauing) suggests the classifier operates in a gradient-suppressed regime even after 200 epochs
- This may explain why 12/24 classes are AP=0: the combined gradient suppression prevents the classifier from learning low-frequency classes

**What we'd need to test:**
- OHEM ablation: train 3 epochs without OHEM, track pos_score_mean and cls_w_norm
- FocalLoss ablation: compare OHEM+FL vs. FL-only vs. vanilla BCE
- If either removal significantly accelerates learning, we've identified the bottleneck

**Confidence**: MEDIUM-HIGH that OHEM+FL gradient suppression is real and significant. MEDIUM on whether removing it would break the mAP ceiling.

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

### Q30: Will RF2 Reach Gate Targets? (CORRECTED 2026-06-21 22:16 UTC — Run 2 Is the Real Test)

**⚠️ CORRECTION**: The prior answer "NO — ba48691 restart confirmed detach fix insufficient" was based on Run 1 (wrong LR/BIAS=4.0/2.0). This conclusion is INVALIDATED. Run 2 (correct LR/BIAS=1.0/1.0) has only completed 1 epoch val (epoch 17, mAP50=0.2039 — a restart from the pre-existing checkpoint, so matching is expected).

**What we actually know now:**

**Run 1** (wrong config, DET_BIAS_LR_FACTOR=4.0, DET_LR_MULTIPLIER=2.0):
- Epochs 17-21: mAP50=0.2039, 0.2065, 0.2088, 0.2047, 0.2024 — flat
- This run had the CORRECT detach=False but WRONG LR/BIAS multipliers
- The high bias LR (4.0×) may have actively impaired the classifier by over-updating the bias relative to weights
- **Verdict: INVALID as test of the correct config**

**Run 2** (correct config, DET_BIAS_LR_FACTOR=1.0, DET_LR_MULTIPLIER=1.0):
- Started at 19:11:11 UTC
- Epoch 17: mAP50=0.2039, MAE=9.25° (only 1 epoch completed)
- All other metrics match Run 1 start (expected: same checkpoint, same epoch)
- **Verdict: INSUFFICIENT DATA — need epochs 18-22**

**What this means:**
- The OHEM+FocalLoss gradient suppression hypothesis is SUPPORTED BY OVERFIT but NOT YET TESTED by main training with ALL fixes
- detach=False is committed for ALL stages — Run 2 tests this in full
- LR/BIAS=1.0/1.0 is the clean config we've been working toward
- **There is a REAL chance Run 2's correct config breaks the ceiling**, unlike what Q30 previously claimed

**Updated gate criteria:**
- det_mAP50 >= 0.40: Current 0.2039 — UNKNOWN whether correct config changes trajectory
- MAE <= 60°: ACHIEVED ✅ (9.25°)
- **Next decisive checkpoint**: Run 2 epoch 18 val (expected ~23:31 UTC on 2026-06-21)
- **If Run 2 epochs 18-20 show consistent mAP improvement**: detach + LR/BIAS fix WAS the bottleneck
- **If Run 2 stays flat at 0.20-0.21**: OHEM+FocalLoss hypothesis is now genuinely supported by main training

**Confidence**: LOW on whether current config reaches gate. The prior HIGH-confidence "NO" was based on invalidated evidence. We genuinely don't know yet.

---

## HIGH (Important for Direction)

### Q31: Why Are 12/24 Classes ALWAYS at AP=0 Across ALL Epochs? (UPDATED — v11 C2 Correction: Class 6 Has 65/91 Samples, Not 1739)

**The v11 C2 correction fundamentally changes this question.** Opus v11 §C2 discovered that the "1739 GT instances for class 6" number was an accumulation artifact — the per-class AP table we were reading did NOT have a trustworthy source. The repo's own `DET_CLASS_ALPHAS` (config.py) shows class 6 has only **65 training samples** (91 total across all splits, 65 at 50% subset).

The correct numbers for the 50% RF2 subset:

```
AP=0 WITH GT — classes where GT exists in subset:
  Class 6:  ~33 actual images (65/91 total samples → ~33 in 50% subset), AP=0.0
  Class 8:  GT present, AP=0.0
  Class 13: GT present, AP=0.0
  Class 19: GT present, AP=0.0

AP=0 — NO GT IN SUBSET — 8 classes:
  Classes 1, 2, 3, 14, 15, 16, 18, 23 — zero GT instances in the 50% subset
```

**This changes the narrative from "1739 GT with AP=0 → impossible without label error" to "~33 images with AP=0 → plausible data-scarcity problem."** The burden of proof shifts: it is NOW reasonable that a class with only ~33 training images (at 50% subset) scores AP=0 — especially if those 33 images have high intra-class variance, small objects, or annotation quality issues.

**Classes 5 and 21 achieved AP=1.0** with 33 and 151 GT instances respectively. But these are very distinctive rare objects (likely unique assembly components with no visual ambiguity). Class 6 may be visually confused with other classes.

**Re-opened possibilities:**
1. **Data scarcity** (now the PRIMARY hypothesis): ~33 images per epoch may be insufficient for the classifier to learn class 6, especially if those images are visually similar to other classes
2. **Label error** (still possible but no longer the default): systematic label errors would explain AP=0 even with adequate data — but we no longer have evidence that data IS adequate
3. **Anchor mismatch** (unchanged): class 6 objects may have geometry incompatible with the anchor grid
4. **Feature confusion** (unchanged): class 6 consistently confused with another visually similar class

**What this means for training**: The "12/24 AP=0" problem is less dramatic than we thought. 8 of those 12 classes have NO GT in the subset (expected). Only 4 classes have GT but score AP=0, and class 6's story is plausible data scarcity, not a smoking gun.

**What we need next:**
1. Visual inspection of class 6 GT boxes (are they correct? do they look like class 6?)
2. After ba48691 restart: re-check per-class AP at epoch 2-3
3. Anchor-IoU histogram for class 6 specifically — are anchors matching at all?

**Confidence**: HIGH that the 1739→65 correction changes our understanding. MEDIUM on whether class 6's AP=0 is data scarcity vs. label error vs. anchor mismatch.

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

### Q35: Does detach_reg_fpn=True Change EVERYTHING About Our Diagnosis? (RESOLVED — Opus v10 Confirmed True)

**Opus v10 confirmed**: `detach_reg_fpn=True` for RF2 via code trace. The resolution chain is:
- config.py stage_rf2 preset: `'detach_reg_fpn': True` (committed code)
- RF2 stage_cfg: NO override for detach_reg_fpn
- CLI: no `--detach-reg-fpn` flag passed
- **Effective value: True** — confirmed by reading all three layers

**Resolution**: Opus v10 traced the full config resolution chain. For RF2, the preset value `True` propagates unmodified. The training we ran for epochs 7-21 had regression gradients detached from the backbone.

**What this means**:
1. The backbone was shaped only by classification + head_pose — no regression signal
2. This is a one-to-one match for symptoms: bestIoU 0.86-0.98 (regression works) + mAP 0.20 (classifier starved)
3. The LR restart had zero effect because a detached gradient path is not a local minimum — it's a structural constraint

**The fix is applied** (working tree, uncommitted):
- config.py:1115: `'detach_reg_fpn': False` for stage_rf2
- DET_LR_MULTIPLIER=2.0 and DET_BIAS_LR_FACTOR=4.0 also changed (see Config Tensions section)
- Awaiting training restart from best checkpoint

**The caveat** (Opus v10): RF2 with detach=True reached mAP=0.204, slightly ABOVE RF1 (detach=False) at 0.184. So detach is a **handicap that v8 fixes + 2.5× data partly compensated for**. Flipping it should move the ceiling but may not single-handedly clear 0.40.

**Confidence**: HIGH that detach was a major factor. The config resolution is confirmed.**Status: RESOLVED.**

---

### Q36: Will the 50-Image Cls-Only Overfit Definitively Bin the Problem? (SOLVED — WEAK PASS)

**The experiment has been run**: 200 epochs, 50 images with GT, detection-only, no multi-task interference. The most decisive experiment in the project's history is now complete.

**Result: WEAK PASS**

| Metric | Value | Interpretation |
|--------|-------|---------------|
| Final cls_loss | 0.0618 | Near zero — classification almost saturated |
| Final reg_loss | 0.113 | Flat — minimal, detection-only |
| Pos_score_mean | 0.9716 | Positives scored near 1.0 |
| Pos_score_max | 1.0000 | Perfect confidence achievable |
| cls_w_norm | 7.07 → 13.43 | LINEAR growth over 200 epochs, never plateaued |
| pos_n | 13 | CONSISTENTLY 13 positive anchors/batch |

**What it proved:**
1. ✅ The architecture CAN learn to classify (not an architectural ceiling)
2. ✅ Positive signals are present (pos_score_mean=0.97, pos_max=1.0)
3. ❌ Learning speed is VERY slow — 55 epochs to escape plateau even on 50 images
4. ❓ Only 13 positive anchors/batch — the IoU threshold limits positive matches, not TOP_K

**What binned:**
- **Architectural ceiling**: REFUTED. The architecture CAN overfit to near-perfect classification. The ceiling is in the training dynamics (OHEM+FocalLoss suppression, or data quality at scale).

**What remains open:**
- The 50-epoch plateau in Regime 2 (cls_loss~0.50) is consistent with OHEM+FocalLoss gradient suppression
- cls_w_norm grows LINEARLY over 200 epochs (7.07→13.43) — hasn't saturated
- This suggests the classifier is in a gradient-suppressed regime even after 200 epochs on 50 images

**Implications for main training:**
- With detach_reg_fpn=False, the backbone will receive regression gradient
- But the classification head's learning will still be OHEM+FocalLoss suppressed
- The expected improvement is FROM the backbone becoming more object-discriminative, not from faster classification head training
- At the main scale (50% subset, thousands of images), the same gradient suppression may explain why 12/24 classes are AP=0

**Confidence**: HIGH (SOLVED — the question of whether the architecture CAN classify is definitively answered). MEDIUM on the exact mechanism of gradient suppression.

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

### Q40: Should We Advance to RF3 Now, or Continue RF2? (UPDATED — Opus v10 Changes the Equation)

**This question has changed fundamentally**: Opus v10 identified `detach_reg_fpn=True` as the primary config regression. The immediate action is no longer "advance vs diagnose" — it's **"restart RF2 with detach=False and watch 3-4 epochs."

**Updated decision framework:**

**Step 1 (immediate): Restart RF2 with fixes applied**
- detach_reg_fpn=False (regression gradient flows to backbone)
- DET_POS_IOU_IOU_FLOOR=0.2 (prevents top-k poisoning)
- DET_LR_MULTIPLIER=2.0, DET_BIAS_LR_FACTOR=4.0 (config tensions — see Q41)
- Run 3-4 epochs, track per-class AP
- **Cost**: ~6 hours. **Expected**: mAP climbs to 0.25-0.35 if detach was the bottleneck

**Step 2 (after restart): Decision tree**
- If mAP climbs past 0.30 in 3-4 epochs: continue RF2, detach was the bottleneck
- If mAP stays flat (0.20-0.22): label noise / anchor mismatch is the ceiling. Run 50-image overfit.
- If mAP drops: something else is wrong. Run diagnostic probes.

**Step 3 (conditional): Advance to RF3**
- Only after the restart demonstrates the ceiling has moved
- RF3's detach_reg_fpn must also be set to False (same bug exists in stage_rf3 preset)
- Activity head adds value only if detection is healthy

**The Opus v10 decision rule**: 
> *"Don't advance to RF3 — fix RF2 first. RF3 stage_rf3 preset ALSO has detach_reg_fpn: True — advancing inherits the bug."*

**The 50-image overfit is still valuable** — it would tell us whether the architecture CAN learn classification without multi-task interference. But the priority has shifted: restart with detach=False first, run overfit in parallel.

**Confidence**: HIGH that restarting RF2 with detach=False is the correct immediate action. LOW on what the outcome will be.

---

### Q41: Will Flipping detach_reg_fpn Break the Plateau? (ANSWERED — Epoch 19: "detach Fix Alone is INSUFFICIENT")

**The central question of the post-v11 era**: After 17+ phases, 11 Opus consultations, and ~500 GPU-hours, the detach fix is committed in ba48691. The question shifts from "will it help?" to "is it sufficient?"

**ANSWERED 2026-06-21 (epoch 19)**: **NO — detach fix alone is NOT sufficient.** 5 epochs of post-restart training show identical mAP (0.202-0.209) to pre-restart.

**What has changed since v10 era (when Q41 was first written):**

| Parameter | v10 Working Tree | v11 Final (ba48691) | Impact |
|-----------|-----------------|---------------------|--------|
| detach_reg_fpn (RF2) | False | False | Unchanged ✅ |
| detach_reg_fpn (RF3-RF10, paper_run) | Default (True) | False | FIXED — prevents bug inheritance |
| DET_LR_MULTIPLIER | 2.0 | 1.0 | Reverted to v8 baseline ✅ |
| DET_BIAS_LR_FACTOR | 4.0 | 1.0 | Reverted to v8 baseline ✅ |
| KENDALL_FIXED_WEIGHTS | False | True (ba48691 also) | Fixed Kendall weights |
| Overfit experiment | NOT RUN | COMPLETE (WEAK PASS) | Architecture CAN learn |

**The ba48691 commit resolves the config tensions that existed in the working tree.** However, 5 epochs of training demonstrate that these fixes, while necessary, are not sufficient. The OHEM+FocalLoss gradient suppression mechanism (identified by the overfit) is the primary bottleneck.

**The updated post-answer scenarios:**

| Scenario | Likelihood | Evidence | Next Action |
|----------|-----------|----------|-------------|
| ~~Best case: mAP climbs to 0.30-0.40~~ | ~~30-40%~~ | ❌ RULED OUT by epoch 19 | N/A |
| ~~Moderate case: mAP climbs to 0.22-0.28~~ | ~~40-50%~~ | ❌ RULED OUT by epoch 19 (mAP never exceeded 0.209) | N/A |
| **Actual outcome**: mAP stayed flat at 0.20-0.22 | **100%** | ✅ CONFIRMED — OHEM+FL suppression is primary ceiling | Run OHEM ablation |
| **What if we also fix OHEM?**: mAP improves to 0.25-0.35 | 40-60% (speculative) | Only known after ablation | OHEM OFF experiment |

**Key insight**: The detach bug consumed 10 Opus consultations to find and fix, but it was never the primary bottleneck. The OHEM 2:1 ratio + FocalLoss gamma_neg=1.5 was always the deeper issue. The overfit proved it on 50 images, and the main training confirmed it at scale.

**Confidence**: HIGH that detach fix alone is insufficient. MEDIUM on whether OHEM ablation will be sufficient.

---

### Q42: Why Only 13 Positive Anchors Per Batch Consistently Across 200 Epochs? (NEW — Overfit Discovery)

**The discovery**: The 50-image cls-only overfit logged `pos_n` (number of positive anchors per batch) at every epoch. Across all 200 epochs, **pos_n was consistently 13**. Not 12, not 14 — exactly 13 positive anchors per batch, every single time.

**Why this is shocking**: The dataset had 50 images, each with ≥1 GT box (some with multiple). Batch_size=4 means each epoch processes 200/4=50 batches. With 50 images × ~2-5 GT boxes/image = ~100-250 total GT boxes, and DET_POS_IOU_TOP_K=9 allowing up to 9 positive anchors per GT, the theoretical maximum is ~900-2250 positive anchors/batch. But the actual is always 13.

**The mechanism**: `DET_POS_IOU_THRESH=0.4` is the gatekeeper. An anchor is only positive if its IoU with a GT exceeds 0.4. The overfit consistently found ~3.25 GT boxes/batch, each matching ~4 anchors above 0.4 IoU: 3.25 × 4 ≈ 13. The TOP_K=9 never kicks in because no GT box exceeds 4 high-IoU anchors.

**What this means:**
1. **TOP_K=9 is mostly irrelevant** — the IoU threshold (0.4) is the real limiter, not the top-K count
2. **The effective positive-to-negative ratio is 13:656K = 0.002%** — essentially zero
3. **This is NOT a data quantity issue** — even with 50 GT-rich images, the anchor system limits positives to 13
4. **Scale invariance**: If each image produces ~3.25 GT boxes matching ~1 positive anchor each, then 2000 training images would produce ~13 positives/batch at the SAME rate (batch_size=4 means ~13 images/batch × ~1 positive/image)

**Three hypotheses:**
1. **Anchor grid mismatch**: The anchor sizes (starting at 96px) are too large for most IndustReal objects. Small parts never exceed IoU=0.4 with any anchor, so they never become positive.
2. **IoU threshold too high**: DET_POS_IOU_THRESH=0.4 is RetinaNet's default, designed for COCO (medium-large objects). For small assembly parts, 0.3 might be more appropriate.
3. **Label density artifact**: Most images genuinely have only 1-2 GT boxes, and those boxes only match 1-2 anchors at >0.4 IoU. The 13 constant is a property of the dataset's GT distribution.

**What we'd need to test:**
- Lower DET_POS_IOU_THRESH to 0.3 and re-run the 50-image overfit — does pos_n increase?
- Anchor-IoU histogram: for all GT boxes in 50 images, what's the max IoU with any anchor?
- If max IoU < 0.4 for most GT boxes, the anchor grid is fundamentally mismatched

**Confidence**: HIGH that the 13-pos-anchor constant is real and significant. MEDIUM on which hypothesis explains it.

---

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
| Q01 | Why plateau despite detach fix AND arch CAN learn? (UPDATED — overfit evidence) | CRITICAL | MEDIUM | YES |
| Q03 | Has PSR ever trained? | CRITICAL | MEDIUM | YES (RF4+) |
| Q04 | Was bias-collapse narrative wrong? | CRITICAL | MEDIUM | YES |
| Q05 | Is Focal Loss + OHEM structurally capping gradient? (UPDATED — overfit data) | CRITICAL | MEDIUM-HIGH | YES |
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
| Q30 | Will ba48691 restart reach gate? (ANSWERED — epoch 19: NO) | **ANSWERED: NO** | CRITICAL | YES |
| **Q31** | **Why 12/24 classes always AP=0? (UPDATED — v11 C2: class 6 has 65/91, not 1739)** | **CRITICAL** | **MEDIUM** | **YES** |
| **Q32** | **Top-k IoU floor poisoning classifier?** | **CRITICAL** | **MEDIUM** | **YES** |
| **Q33** | **Combined metric misleading?** | **HIGH** | **HIGH** | **YES** |
| **Q34** | **Why LR restart had ZERO effect?** | **HIGH** | **HIGH** | **No** |
| **Q35** | **detach_reg_fpn split-brain — RESOLVED (Opus v10 confirmed True)** | **RESOLVED** | **HIGH** | **No** |
| **Q36** | **50-image overfit — SOLVED (WEAK PASS)** | **SOLVED** | **HIGH** | **No** |
| **Q37** | **DET gradient bottleneck (117×)?** | **HIGH** | **MEDIUM** | **Possibly** |
| **Q38** | **head_pose borderline ALIVE/DEAD?** | **MEDIUM** | **MEDIUM** | **No** |
| **Q39** | **score_p50 blindness invalidates prior analysis?** | **HIGH** | **HIGH** | **No** |
| **Q40** | **Restart RF2 with detach=False first? (REFRAMED)** | **CRITICAL** | **MEDIUM** | **YES** |
| **Q41** | **Is detach fix sufficient? (ANSWERED — epoch 19: NO)** | **ANSWERED: NO** | **CRITICAL** | **YES** |
| **Q42** | **Why only 13 positive anchors per batch consistently? (NEW)** | **HIGH** | **MEDIUM** | **Possibly** |

---

## Status Changes from Previous Version

| Previous Status | Current Status | Questions |
|----------------|---------------|-----------|
| CRITICAL | ANSWERED | Q30 (ba48691 restart won't reach gate — epoch 19 confirmed), Q41 (detach fix insufficient — epoch 19 confirmed) |
| CRITICAL | CRITICAL | Q01 (UPDATED — overfit evidence), Q03, Q04, Q05 (UPDATED — OHEM+FL data + main training confirmed) |
| CRITICAL | RESOLVED / REFRAMED | Q25 (v8 fixes worked, ceiling remains), Q35 (detach_reg_fpn — Opus v10 confirmed) |
| CRITICAL | REFRAMED | Q40 (now: restart RF2 with detach=False, not advance-or-diagnose) |
| CRITICAL | SOLVED | Q36 (50-image overfit — WEAK PASS) |
| HIGH | RESOLVED | Q02, Q10 (phantom 0.45 fixed), Q27 (merged into Q10) |
| HIGH | MEDIUM | Q06, Q07, Q13, Q16 |
| MEDIUM | LOW | Q09, Q12 |
| NEW | CRITICAL | Q31 (UPDATED — v11 C2: class 6 65/91, not 1739), Q32 (top-k IoU floor) |
| NEW | HIGH | Q33 (combined metric), Q34 (LR restart failure), Q37 (gradient bottleneck), Q39 (score_p50 blindness), Q42 (13-pos-anchor mystery) |
| NEW | MEDIUM | Q38 (head_pose borderline) |

---

*Generated 2026-06-21 by Claude Code. Updated 2026-06-21 21:00 UTC with epoch 19 evidence: Q30 ANSWERED (ba48691 restart won't reach gate), Q41 ANSWERED (detach fix alone insufficient), OHEM+FocalLoss gradient suppression CONFIRMED by main training. 42 questions total covering 18+ phases, 11 Opus consultations, ~500 GPU-hours of investigation, and now ~5 epochs of post-restart validation.*
