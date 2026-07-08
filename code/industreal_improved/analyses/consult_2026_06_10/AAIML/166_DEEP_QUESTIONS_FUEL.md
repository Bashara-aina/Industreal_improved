# 166 — 50+ Deep Questions Driving the AAIML Paper

**Date:** 2026-07-08
**Purpose:** These questions are the FUEL for the paper. Each one forces a specific measurement, a specific comparison, or a specific ablation. Answering all of them is the path to the best AAIML paper.

---

## Section A: Multi-Task vs Single-Task (Core Hypothesis)

**A1.** Does multi-task training on IndustReal actually HURT or HELP each head compared to single-task training on the same architecture? Give per-head numbers with confidence intervals.

**A2.** If multi-task hurts some heads, what is the magnitude of the hurt? For each head, give the ratio (multi-task performance / single-task performance) with 95% CI.

**A3.** Which head is MOST sensitive to multi-task interference, and which is MOST robust? Rank the 4 heads by sensitivity.

**A4.** Does the choice of multi-task architecture (V5 ConvNeXt vs V6/V8 MViTv2-S+ConvNeXt) change the multi-task vs single-task comparison? Show per-architecture numbers.

**A5.** Is there a sweet spot for multi-task weight balancing (Kendall log_var) that maximizes all 4 heads simultaneously, or are the optima for different heads incompatible?

**A6.** What is the CORRELATION between Kendall log_var values for the 4 heads during training? If pose log_var grows while activity shrinks, that's evidence of multi-task interference.

**A7.** Does freezing the backbone for some heads and fine-tuning for others (e.g., YOLOv8m frozen, MViTv2-S frozen, pose/PSR trained) reduce interference? Show ablation.

**A8.** At what point during training does multi-task interference emerge? Is it present at epoch 1, or only after some heads converge?

**A9.** Does the KENDALL_FIXED_WEIGHTS=0 (let Kendall learn) vs =1 (frozen weights) change the multi-task dynamics? Show the log_var trajectory over training.

**A10.** For each head, what is the OPTIMAL Kendall weight (from a hyperparameter sweep)? If the optimum differs from the balanced 1/N, that's evidence of asymmetric interference.

---

## Section B: Architecture and Backbone (Detection + Activity)

**B1.** The detection backbone is currently YOLOv8m. Is the SOTA-comparable 0.995 mAP50 from a SINGLE-TASK YOLOv8m run, or can a multi-task run with shared YOLOv8m features also hit 0.995?

**B2.** Does using YOLOv8m features for multiple heads (det + pose + PSR) reduce the detection mAP50 compared to YOLOv8m dedicated features? If yes, by how much?

**B3.** The activity backbone is MViTv2-S. The frozen probe is 0.3810. What's the EXPECTED improvement from full fine-tuning? Cite 150's estimate: 0.45-0.55.

**B4.** Can MViTv2-S features be used for ALL 4 heads (det + pose + activity + PSR)? The temporal features might help pose (which involves rotation estimation) and PSR (which involves temporal transitions).

**B5.** For the detection head, what's the best architecture: YOLOv8m dedicated, or MViTv2-S features + FPN? Compare both.

**B6.** For the pose head, what's the best regression target: fwd+up as 6D vector, or quaternion (4D) + up vector (3D) = 7D? Compare both.

**B7.** For the PSR head, is per-component binary classification the best formulation, or would multi-label classification (allowing multiple components to be positive in the same frame) be better?

**B8.** For the activity head, is the 69-class multi-class formulation the best, or would a hierarchical formulation (top-level assembly stage → fine-grained action) be better?

**B9.** What's the computational cost of V8 (MViTv2-S + YOLOv8m) vs V5 (ConvNeXt)? FLOPs, parameters, training time. V8 should be more efficient per task, or it loses the multi-task advantage.

**B10.** Does the choice of backbone (ConvNeXt vs MViTv2-S vs YOLOv8m) affect which head is most sensitive to multi-task interference?

---

## Section C: SOTA Comparison and Benchmarks

**C1.** For each of the 4 heads, list ALL SOTA references in the IndustReal paper family. Cite paper, number, and what "comparable" means in each case.

**C2.** For detection mAP50: WACV reports 0.95, 0.838, 0.641. Which is the "fair" comparison for our multi-task result? (Probably the WACV full-system number, not the per-component breakdown.)

**C3.** For activity top-1: WACV reports 0.6223 (RGB-only) and 0.6645 (RGB+VL+stereo). Our frozen probe is 0.3810 with same MViTv2-S backbone. What's the expected fine-tuned number?

**C4.** For PSR F1: STORM reports 0.506, B2 baseline reports 0.731, B3 reports 0.883. How does each compare to our V5b/V8 paradigm? (Note: STORM/B use different prediction schemes.)

**C5.** For head pose: there's NO published SOTA on IndustReal. Cite this as a "first baseline" claim. What does the literature say about similar industrial assembly pose estimation?

**C6.** Cross-architecture comparison: our D1R YOLOv8m 0.995 vs WACV 0.838. The architecture is the same (YOLOv8), so the comparison is FAIR. But the training data and recipe might differ.

**C7.** Cross-paradigm comparison: our V5b multi-task vs single-task runs. The paradigm is the same (multi-head) but the model class is different (ConvNeXt vs YOLOv8m). This is a confounded comparison.

**C8.** Are the SOTA numbers in the published papers (WACV, STORM, B3 baseline) from the same train/val split as ours? If not, the comparison is unfair.

**C9.** What is the "honest brief" we can present? I.e., which numbers can be presented side-by-side with what caveats?

**C10.** If V5b (multi-task ConvNeXt) gives detection mAP50 = 0.01, can we honestly present that as "vs WACV 0.838"? The architecture difference (ConvNeXt vs YOLOv8) is the dominant factor.

---

## Section D: Implementation Fixes (Code-Level)

**D1.** Were all 9 file-152 fixes correctly applied to V5b/V8? Audit them one by one with file:line evidence.

**D2.** What is the LeakyReLU activation doing for the PSR head? It's supposed to fix the dead GELU. Is PSR actually learning now (non-zero F1)?

**D3.** What is the GT-balanced detection sampler doing? Does it actually help multi-task training, or just single-task detection?

**D4.** Is the DETACH_PSR_FPN=False fix actually allowing gradient flow? The LIVENESS probe showed PSR is ALIVE (gradient norm 0.13-2.12), but is this significant or just noise?

**D5.** The F-1 Fix 1 (psr_head freeze bypass) and F-1 Fix 2 (Kendall staging guard) are in the code. Are they being applied correctly in V5b? V5b uses KENDALL_FIXED_WEIGHTS=0, so Fix 2 is bypassed (no staging). Fix 1 should be active.

**D6.** The MIXED_PRECISION=True (bf16) is applied. Is the training stable under bf16? Are there NaN issues?

**D7.** For V8: are the YOLOv8m weights loaded correctly? Is the Multi-scale FPN integration working?

**D8.** For V8: is the KENDALL_FIXED_WEIGHTS=0 causing any instability? Should we use a learning rate warmup for the log_var parameters?

---

## Section E: Training Dynamics and Failure Modes

**E1.** Why did V5b (with KENDALL_FIXED_WEIGHTS=1) collapse on detection/PSR/activity? Is the root cause pose being over-weighted (precision 2.68 vs 0.58 for det)?

**E2.** With KENDALL_FIXED_WEIGHTS=0, will V5b recover? Or will the model just learn different collapse dynamics?

**E3.** V8 shows similar collapse at epoch 0 step 700 (all-zero predictions on classification heads). Is this the same root cause?

**E4.** What's the difference between "model collapse" (always predicts 0) and "model converged to trivial solution" (always predicts the same class)? The metrics may not distinguish.

**E5.** Is the "rebuild from epoch 1 with KENDALL_FIXED_WEIGHTS=0" strategy correct? Or should we instead keep training and just rebalance the log_var values mid-training?

**E6.** What is the impact of the "pose overweight" in V5b's pre-fix checkpoint? If the model's already learned that pose=0.001° is the safe answer, can it unlearn this?

**E7.** How long does V5b's KENDALL rebalance take to converge? Should I check log_var trajectories at epoch 1, 5, 10, 25, 50?

**E8.** For V8, the model was initialized with small-normal weights. Did this help with the early collapse? Or does the loss curve still go to 0?

**E9.** What does the training loss look like at epoch 1 vs epoch 5 for V8? Is there a turning point where the model starts learning, or is it just slow convergence?

**E10.** For V5b, the val loss is NaN on detection. Is the model producing inf/nan values, or is the metric calculation broken?

---

## Section F: Multi-Task Efficiency (vs Single-Task)

**F1.** Total FLOPs for V8 multi-task vs V5 multi-task vs 4 separate single-task runs. V8 should be more efficient if the multi-task architecture shares computation.

**F2.** For a fixed compute budget, is V8 multi-task or 4 single-task runs better? In theory, multi-task should win (shared representation).

**F3.** For a fixed dataset (one pass through data), does V8 learn more than V5 because of the better backbone (MViTv2-S vs ConvNeXt)? Or is the multi-task overhead too high?

**F4.** Parameter count: V8 (MViTv2-S + YOLOv8m + 3 heads) vs 4 single-task. Is V8 fewer parameters? (Should be — shared backbone.)

**F5.** Training time per epoch: V8 vs 4 single-task. V8 should be faster (single multi-task forward pass) unless the YOLOv8m and MViTv2-S backbones are too slow together.

**F6.** Inference time per sample: V8 multi-task vs V5 vs single-task. V8 should be similar (one forward pass through each backbone).

**F7.** Memory usage: V8 should use less GPU memory than 4 separate single-task models (shared backbones).

**F8.** Data efficiency: V8 multi-task should achieve same performance with less data (shared representations learn from multiple tasks).

**F9.** What is the "winning" architecture? A 2-backbone multi-task (V8: MViTv2-S for activity, ConvNeXt for others) vs 1-backbone multi-task (V5: ConvNeXt for all) vs 4 single-task.

**F10.** For the AAIML paper, what is the fair efficiency comparison? V8 multi-task on 1 GPU vs 4 single-task on 4 GPUs?

---

## Section G: AAIML Paper Story (How to Write It)

**G1.** What is the ONE sentence that captures the paper's contribution? (See 165 master blueprint.)

**G2.** What is the most compelling single experiment to put in the main paper? (The decision-of-decisions result.)

**G3.** What experiments go in supplementary vs main paper? (5 main, 8 supplementary?)

**G4.** How to structure the related work? (Multi-task learning, action recognition, detection, pose estimation, IndustReal benchmark.)

**G5.** How to frame the "implementation matters" message without making it sound like the multi-task is the issue (which is the user's bias)?

**G6.** What's the title? "What Four Tasks Cost One Backbone: A Pathology Analysis of Multi-Task Training on IndustReal" (per 150) vs "Multi-Task Training with Kendall Rebalancing on IndustReal" vs something else.

**G7.** What's the single best ablation to show? (e.g., V5 with KENDALL_FIXED_WEIGHTS=0 vs =1 vs single-task baseline.)

**G8.** How to present negative results (collapsed heads, pose overweight)? "Caveats" section or "limitations" section.

**G9.** How to handle the "honest brief" — present only what V5b/V8 can give now, or wait for full results?

**G10.** What's the reviewer defense? Reviewers will challenge: (a) why not YOLOv8m for everything, (b) why no fresh-from-scratch baseline, (c) why V8 results aren't in the brief yet.

---

## Section H: Specific Measurable Targets

For each head, what's the target number we need to hit to claim "near SOTA"?

**H1.** Detection: target mAP50 = 0.5+ (V5b post-fix), 0.99+ (D1R YOLOv8m) — what is the V8 multi-task expectation?

**H2.** Activity: target top-1 = 0.45+ (V8 multi-task), 0.45+ (V6 fine-tune) — what is realistic for 5 epochs?

**H3.** PSR: target F1 = 0.5+ (V5b/V8 multi-task) — what is the realistic ceiling?

**H4.** Pose: target fwd MAE = 8.0-8.5° (V5b/V8 multi-task) — what is the converged value?

**H5.** Activity: target efficiency (samples/hour on RTX 3060/5060) — V8 vs V5 vs single-task.

**H6.** All 4 heads in single training run — V8 needed (V8 not yet validated).

**H7.** Honest brief: detection=0.995, activity=0.3810 (frozen) or 0.45+ (V8), PSR=0.5+, pose=7.5-8.5°.

**H8.** Architectural change: V8 (MViTv2-S + YOLOv8m) — the user's directive. Is it realistic in 20h?

**H9.** Detection SOTA: 0.995 from D1R YOLOv8m. Is this the headline, or is it V5b's multi-task result?

**H10.** Multi-task efficiency claim: V8 should be faster than 4 single-task — can we prove this?

---

## Section I: How to Use These Questions

Each question is a SPECIFIC measurement or ablation. Answering all of them takes the paper from "general" to "deep and comprehensive."

The 10 specialized agents will be deployed to debate and refine these. The output of each agent's debate will go into 167-170.

The remaining time is 18h. The training will give partial data. The paper will need to be honest about what we have.
</content>
