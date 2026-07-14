# === V1 FACT-CHECK (REQUIRED READING) ===
# This file was generated as V2 before the codebase was fully audited.
# The V1 documents (208-227) it references describe the MViTv2-S era
# which is NO LONGER the active codebase. Key corrections:
#
# **Active model:** POPWMultiTaskModel in src/models/model.py (2361 lines)
# **Backbone:** convnext_tiny (28.59M, ImageNet-1K pretrain) — NOT MViTv2-S
# **Total params:** 46.47M (measured) — NOT ~48.6M
# **Detection head:** RetinaNet-style 5.31M (9 anchors, 5 levels P3-P7)
# **Activity head:** FeatureBank+TCN+2xViT 0.69M (NOT 3-layer MLP 2M)
# **PSR head:** PSRHead hidden_dim=128 3.08M (NOT 1.8M Causal Transformer)
# **Pose:** body 1.64M + head 1.45M + 2 FiLMs 1.24M (NOT single 0.2M)
# **FPN:** Standard P3-P7 4.48M (NOT BiFPN P2-P5 ~2.5M)
# **Effective batch:** 48 (BATCH=6 x GRAD_ACCUM=8) — NOT 16
# **PSR_FOCAL_GAMMA:** 0.5 — NOT 2.0
# **Kendall caps:** act=[-0.5,2.0], psr=[-4,0.0], pose=[-4,3.0] (NOT [1.5,1.0,0.5,2.0])
# **Recordings:** 36/16/32 (NOT 10/6)
# **Frames:** 207,266 (NOT ~75K)
# **Free GPUs:** RTX 5060 Ti 16GB + RTX 3060 12GB
# **Deadline:** AAIML Oct 10, 2026
#
# See consult_v2/V1_VS_CODEBASE_DISCREPANCY_REPORT.md for full audit.
# The legacy src/models/mvit_mtl_model.py (MViTv2-S) is DEAD CODE.
# All architecture numbers in this report should be re-validated against
# the active model before any paper submission.
# === END FACT-CHECK ===

# Agent 19: Risk & Contingency Analysis

**Date:** 2026-07-13
**Context:** Phase 4 of Claude Science V2 — AAIML Strategy
**Inputs:** Agent 08 (Task Heads), Agent 09 (Training Pipeline), Agent 15 (MTL Stability Lit), Agent 06 (Backbone), Agent 11 (Detection MTL Lit), original 225_RISK_ASSESSMENT.md, 216_AAIML_STRATEGY.md
**Note:** The project has shifted from ConvNeXt-Tiny + Kendall (original analysis) to MViTv2-S + FAMO + RotoGrad. The one constant is that both architectures have critical bugs preventing any meaningful training. Agent 08 and Agent 09 independently verified that the current codebase has at least 8 bugs that made all training runs invalid. This document supersedes the original 225_RISK_ASSESSMENT.md.

---

## Pre-Read: The Honest Bottom Line

As of July 13, 2026, the project has never produced a valid training result. Every training run to date is compromised by at least one of: frozen FPN (14.5M frozen parameters, 26% of total model), broken RotoGrad (random matrices, never updated), frozen detection augmentation destroying image normalization, collapsed activity head (1/75 unique predictions), or flat PSR (all 11 components at ~0.7). The activity head collapse and PSR uniformity observed in logs are the CONSEQUENCE of these bugs, not evidence of fundamental architecture problems. We simply do not know what the architecture can do because it has never been correctly trained.

This is simultaneously good news and bad news. Good: the fixes are well-documented and actionable. Bad: every claimed metric, every observed trajectory, every "insight" from the training logs is suspect. The project must essentially restart training from a clean baseline. The timeline to WACV/AAIML submission may be salvageable, but only if the 8 bugs are fixed without introducing new ones and a complete run succeeds on the first attempt.

---

## 1. Technical Risks

### TECH-1 (was CRIT-1): FPN has never been trained — 14.5M frozen parameters

**Probability: CERTAIN (100%)** — verified by code audit (Agent 08 Finding 1)
**Impact if it materializes: CRITICAL** — the FPN is 26% of total model parameters, initialized to random weights, and never updated. The detection head (1.20M) has been trying to learn from random features. All detection metrics from all runs are meaningless.

**Mitigation:**
- Fix `_group_params(["feature_pyramid.fpn", "det_head"], 1.0)` to `_group_params(["fpn", "det_head"], 1.0)` in `train_mtl_mvit.py` line 2133. This is a one-character fix (remove "feature_pyramid.").
- Verify after fix: `len([n for n, _ in model.named_parameters() if "fpn" in n])` should match `len(optimizer.param_groups[1]["params"]) / 2` (approx, since FPN params appear in the group).

**Contingency if fix fails:** The FPN layer name structure may differ from expected. Surround the param group check with a `named_parameters()` dump to verify matching. If the FPN module is nested differently, adjust the prefix accordingly.

**Decision trigger:** FPN should appear in optimizer param_groups immediately after the fix. If post-fix training shows detection loss not decreasing within 500 steps, there may be a deeper wiring issue.

---

### TECH-2 (was CRIT-2): RotoGrad parameters are frozen — 639K never updated

**Probability: CERTAIN (100%)** — RotoGradRotation is instantiated AFTER optimizer creation (verified Agent 08 Finding 2). No `add_param_group()` call. `rotation_loss()` never called. grep returns zero matches.

**Impact: CRITICAL** — RotoGrad adds random fixed rotation matrices to the cls_token before activity and pose heads. The "feature rotation for gradient alignment" is actually noise injection. RotoGradScale is also never instantiated. All RotoGrad-related claims in the paper are unsupported.

**Mitigation:**
- Option A (clean): Move RotoGrad instantiation BEFORE optimizer creation (before line 2142). Add its params to an existing param group or create a dedicated group.
- Option B (minimal): Call `optimizer.add_param_group({"params": rotograd_model.parameters(), "lr": 1e-4})` after line 2279. Wire `rotation_loss()` into the training loop (compute after forward pass, backprop separately with `retain_graph=True` or alternate steps).
- Option C (honest): Remove RotoGrad entirely from the paper. Frame the gradient alignment as future work. FAMO alone is sufficient for the contribution.

**Contingency:** Option C is the safest. RotoGrad adds complexity and another potential failure mode. The paper's core narrative (pathology framework + FAMO for heterogeneous MTL) does not require RotoGrad. Removing it simplifies the codebase and removes the "frozen RotoGrad" embarrassment.

**Decision trigger:** If Option A or B is chosen, verify within 50 training steps: `rotograd_model.rotation.weight.grad` should be non-None. If gradients are zero after 50 steps, the rotation_loss optimization path is not correctly wired.

---

### TECH-3 (was CRIT-3): Activity head collapse is a symptom, not a root cause

**Probability: HIGH (80%)** — The collapse is observed in all training logs (act_preds=1uniq/0.03maxconf). However, it is likely caused by the interaction of frozen FPN + broken RotoGrad + DetectionAugment clamp bug + missing warm-start, not by fundamental architecture limitations.

**Impact: CRITICAL** if it persists after fixes. Activity is the primary metric the paper is organized around.

**Mitigation (four-pronged, ordered by likelihood of fixing):**
1. **Fix warm-start path** (Agent 09 Finding 2). Verify `st_act_best.pt` exists in `src/runs/st_checkpoints/`. If not, train an ST activity baseline first (1-2 GPU days). The ST checkpoint provides a non-random initialization for the activity head.
2. **Fix the DetectionAugment clamp bug** (Agent 09 Finding 5). Change `det_augment.py:102` from `clamp(0.0, 1.0)` to `clamp(-2.5, 2.5)` or remove the clamp entirely. This prevents 50% of batches from having their pixel distribution truncated.
3. **Remove or fix RotoGrad** (TECH-2). If RotoGrad is adding random noise to the activity head's input, removing it eliminates the noise.
4. **Increase activity head capacity** only if the above fail. The current 3.75M param head (768->2048->1024->75) is adequate for 75-class classification. The bottleneck is gradient starvation, not capacity.

**Contingency if mitigation fails:** If activity still collapses after all 4 fixes:
1. Enable `act_decoupled=True` — freeze backbone, train activity classifier alone for 5 epochs, then unfreeze and continue.
2. Class-balanced focal loss (USE_CB_FOCAL_ACT=True) — replaces CE with a principled long-tail loss.
3. Reduce class count: collapse 75 fine-grained classes into 20-30 coarse groups. The paper can report both fine-grained (as is) and coarse-grained (for comparison).
4. Report the collapse as a finding: "Activity collapse under 5-order loss scale disparity is a diagnosed pathology" — this turns a weakness into the paper's core contribution.

**Decision trigger:** Run 10 epochs after all fixes. If activity still produces 1 unique class, pivot to contingency 4 (publish collapse as a finding).

---

### TECH-4 (was CRIT-3): PSR uniform output is also a symptom

**Probability: HIGH (70%)** — All 11 PSR components predict ~0.69-0.71 with stddev 0.02. Like the activity collapse, this is likely caused by the known bugs, not fundamental architecture problems. But the causal Transformer on T=8 frames with causal masking may genuinely struggle to detect transitions in a <1% event-rate regime.

**Impact: HIGH** — PSR is the paper's second contribution (temporal state transition detection). If PSR cannot detect transitions, the 4-task narrative weakens.

**Mitigation:**
1. Fix the known bugs first (frozen FPN, RotoGrad, clamp, warm-start). PSR may start working once the backbone gets meaningful gradients.
2. If still uniform after fixes: replace causal masking with non-causal masking. The original justification for causal masking (online/streaming) does not apply to offline training. Non-causal attention gives each frame access to future context, improving transition detection.
3. If still uniform: reduce T from 8 to 4 or 2. Shorter windows have higher event density per window. This trades temporal precision for detection power.
4. Switch from Focal-BCE to Asymmetric Loss (ASL). Hard-threshold negative gradients instead of soft focal weighting. This directly targets the "predict all zeros" failure mode.

**Contingency:** If PSR still produces uniform output after all mitigations, the paper must pivot. Three options:
1. Report PSR as a documented failure case: "PSR: Negative Result Under 0.5% Event Rate" — a useful contribution given how few papers publish negative results.
2. Frame PSR as binary per-frame state classification (which works: comp_acc=0.567) rather than transition detection (which fails: F1=0).
3. Remove PSR from the core 4-task narrative and report a 3-task MTL (detection + activity + pose) with PSR in supplementary.

**Decision trigger:** If after all fixes, PSR event-F1 < 0.10 at epoch 20, begin contingency planning. If < 0.05 at epoch 30, implement contingency.

---

### TECH-5 (was MED-1): DetectionAugment clamp bug destroys training

**Probability: CERTAIN (100%)** — Verified by Agent 09 Finding 5. `det_augment.py:102` clamps to [0.0, 1.0] after color jitter, but images are normalized to [-2.0, +2.4] range at that point. 50% of batches (p_color=0.5) have their pixel distribution truncated.

**Impact: HIGH** — Every batch that triggers color jitter sends distribution-shifted images to the backbone. This degrades feature quality for ALL four tasks, not just detection. The backbone learns to interpret truncated features, which may explain some of the activity/PSR failures.

**Mitigation:** One-line fix: Change `aug_images.clamp(0.0, 1.0)` to `aug_images.clamp(-2.5, 2.5)` or remove the clamp entirely (check that no downstream code assumes [0,1] range). Verification: after fix, run a single batch and check min/max of augmented images are within [-2.5, 2.5].

**Decision trigger:** Fix immediately. It is a one-line change with no downside.

---

### TECH-6 (was MED-2): Warm-start is completely broken

**Probability: CERTAIN (100%)** — Verified by Agent 09 Finding 2. Only `st_pose_best.pt` was found (loaded 2/4 tensors). Detection, activity, and PSR checkpoints are missing. Even the pose checkpoint only partially loaded (2 tensors vs expected ~4-6).

**Impact: HIGH** — All task heads except pose (partial) were randomly initialized. This means 3/4 heads spent the first 5+ epochs finding their feet from scratch. This directly contributes to the activity collapse and PSR uniformity (TECH-3, TECH-4).

**Mitigation:**
1. Verify checkpoint path: check `src/runs/st_checkpoints/` exists and contains the expected files.
2. If missing: train ST baselines first. ST-det (~55h), ST-act (~55h), ST-psr (~55h) — these are necessary anyway for the MTL/ST ratio comparison in the paper.
3. Fix the partial load for pose: the checkpoint structure may not match the MTL head structure. Run `load_state_dict_with_prefix` debug mode to see which keys match and which don't.

**Decision trigger:** Checkpoint directory inspection. If ST checkpoints exist but aren't being found, fix the path. If they don't exist, ST baselines become the critical path item.

---

### TECH-7 (was MED-3): 480px resolution may be insufficient for small objects

**Probability: MEDIUM (40%)** — Screws and bolts in the assembly video may be <20px even at 480px. At 480px, P3 feature resolution is 60x60, giving ~8px per grid cell. Objects smaller than 8px cannot be resolved at P3 (the finest FPN level). Analysis from Agent 06 shows the attention matrix at 480px is 2.95 GB, so 640px is infeasible on 16GB.

**Impact: HIGH** — If detection cannot detect small objects because resolution is too low, no amount of architecture optimization fixes it. The detection ceiling is imposed by input resolution.

**Mitigation:**
1. Measure actual small-object sizes in the dataset. Extract bounding box dimensions and compute the fraction of objects below (480 / grid_cell_size) pixels in either dimension.
2. If confirmed: ensure P2 features (which are avoided by current code: "Skip P2 (raw conv_proj features, no semantics)") are used for small object detection. P2 at 480px has 120x120 resolution, giving ~4px per cell — still marginal but better.
3. Implement test-time augmentation (TTA) with multi-scale inference (e.g., run detection at 480px and 320px and ensemble). The THW fix (Agent 06 Finding 5) already enables multi-resolution inference.
4. Accept the resolution constraint and frame detection as "operating at 480px for efficiency" with honest disclosure.

**Decision trigger:** Compute bounding box size distribution first. If >20% of objects are below 16px, this risk is HIGH and mitigation 2 (P2 features) should be prioritized.

---

### TECH-8 (was CRIT-4): OOM due to missing expandable_segments

**Probability: HIGH (60%)** — Verified by Agent 09 Finding 4. `train_mtl_mvit.py` does not set `expandable_segments:True` unlike every other training script in the repo.

**Impact: MEDIUM** — Fragmentation OOM typically occurs after 20-25 epochs, not immediately. By the time OOM hits, the team would have days of training to lose. Recovery requires restarting from the last checkpoint.

**Mitigation:** Add `os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'` at the top of `train_mtl_mvit.py`, before any torch import. One-line fix.

**Contingency:** If OOM still occurs (possible with other fragmentation sources), reduce T from 8 to 4, or reduce batch size, or enable gradient checkpointing if not already enabled.

**Decision trigger:** Fix immediately. No experiment should be started without this environment variable. Post-fix: verify per-epoch memory usage is stable (not increasing).

---

### TECH-9 (was HIGH-3): Gradient starvation — activity and PSR may never compete with pose

**Probability: MEDIUM (30%)** after fixes — Pose loss (raw ~700-4000) is 1000x larger than PSR loss (~0.3) and 100x larger than activity (~4). Even with pre-scaling (pose: 0.00025, act: 0.27, psr: 2.7), the effective gradient magnitudes may still be unbalanced because pre-scaling is static and the loss landscape shifts during training.

**Impact: HIGH** — If activity or PSR are gradient-starved, they never learn regardless of architecture.

**Mitigation:**
1. The pre-scaling factors are the first line of defense. Verify they are still appropriate for the post-fix loss landscape (they were tuned on buggy training runs).
2. FAMO's adaptive weighting is the second line. Monitor the FAMO weights during training: if any task's weight drops below 0.05, investigate.
3. Enable IMTL-L as a drop-in alternative (already implemented but not wired). IMTL-L uses log-space weighting which is more robust to scale differences.
4. If all else fails: staged training. Train detection-only for 5 epochs first (since detection establishes backbone features), then introduce PSR + pose, then introduce activity last.

**Decision trigger:** Monitor task gradient norms on a held-out batch at the start of training. If any task's gradient norm is <5% of the total, that task will likely be starved. Implement mitigation (4) in that case.

---

### TECH-10 (was MED-4): num_workers=0 creates data loading bottleneck

**Probability: HIGH (70%)** — Verified by Agent 09 Finding 1. `num_workers=0` means all data loading (decode, crop, augment, normalize) happens in the main process. For T=8 at 480px, this is 240-400ms per batch just for image decode, out of ~760ms total batch time.

**Impact: MEDIUM** — Training is 30-50% slower than necessary. This adds ~2-3 days to a 10-day training run. Not a fatal issue but a schedulability risk.

**Mitigation:** Increase `num_workers=2` or `num_workers=4`. The dataset seed is intentionally shared across all workers (line 120), which means all workers use the same random seed for augmentations. This is acceptable for reproducibility but means augmentations are correlated across workers. If this is a concern, each worker needs a unique seed derived from worker_id.

**Contingency:** If `num_workers>0` causes CUDA errors (shared memory contention), keep `num_workers=0` and accept the speed penalty. The training will complete, just slower.

**Decision trigger:** Test with `num_workers=2` for 100 batches. If no errors and batch time drops to <500ms, adopt. If errors, revert to 0.

---

### TECH-11 (was MED-5): Non-stationary loss landscape under MViTv2-S fine-tuning

**Probability: MEDIUM (35%)** — MViTv2-S is Kinetics-400 pretrained. Fine-tuning on IndustReal (4 tasks) creates a non-stationary loss landscape where task relationships shift dramatically in the first 5-10 epochs as the backbone adapts from action recognition to assembly perception. FAMO's adaptive weights may oscillate during this critical period.

**Impact: MEDIUM** — Weight oscillations during the first 5 epochs could cause a temporary collapse that never recovers (like the activity collapse observed in logs).

**Mitigation:**
1. Warm-start all heads from ST checkpoints (see TECH-6). This reduces the initial chaos because heads start from reasonable initializations rather than random.
2. Set a minimum FAMO weight floor (e.g., min_weight=0.05) to prevent any task from being completely deweighted during the unstable early phase.
3. Use EMA-smoothed losses for FAMO's weight update instead of raw per-step values. This damps oscillations.

**Decision trigger:** Monitor FAMO weights every 500 steps for the first 10 epochs. If any weight oscillates by >0.3 between consecutive steps, implement mitigation 2.

---

## 2. Baseline Risks

### BASE-1: ST baselines are missing — no MTL/ST ratio exists

**Probability: CERTAIN (100%)** — Verified by Agent 09 Finding 2. Only ST-pose checkpoint partially exists. ST-det, ST-act, ST-psr have never been trained.

**Impact: CRITICAL** — The paper cannot claim any MTL/ST retention ratio without ST baselines. The central quantitative claim ("MTL achieves X% of ST performance at 2x parameter savings") requires all 4 ST baselines. Without them, the paper has no comparative MTL analysis.

**Mitigation:**
1. Start ST baselines immediately. They are independent (4 separate training scripts) and can run in parallel if GPU memory permits.
2. ST-det: ~55h (2.3 days). ST-act: ~55h. ST-psr: ~55h. ST-pose: ~12h (already exists). Total: ~6.9 days of continuous GPU time if run sequentially.
3. Use the fixed FPN + fixed augmentation + fixed everything for ST baselines. A baseline trained with the same bugs as MTL is useless.

**Contingency:** If GPU time is insufficient:
1. Run ST-det and ST-act as top priority (the two most critical baselines for the paper narrative).
2. Report ST-pose as "available from prior work" (the existing checkpoint).
3. Publish ST-psr as "budget-limited" and note that the MTL/PSR ratio is computed against a per-frame binary accuracy baseline instead.

**Decision trigger:** Start ST baselines before any MTL training. The ST baseline run is the critical path item. Every day of delay pushes the submission date back.

---

### BASE-2: ST baselines may themselves perform poorly

**Probability: MEDIUM (35%)** — Verified by Agent 09 analysis. MViTv2-S is Kinetics-400 pretrained (action recognition), not detection pretrained. The backbone has never seen bounding box supervision. The ST-det ceiling is fundamentally lower than a COCO-pretrained backbone. Additionally, the ST scripts use the same hyperparameters as MTL (batch size, LR, augmentation) which may be suboptimal for single-task training.

**Impact: HIGH** — If ST baselines are weak, the MTL/ST ratio narrative collapses ("we retain 80% of a weak ceiling" is not publishable). The absolute metrics also matter for the paper's credibility.

**Mitigation:**
1. Tune ST baseline hyperparameters separately. The MTL hyperparameters (designed for balancing 4 tasks) are likely suboptimal for single-task training. Use a broader LR search: [1e-5, 3e-5, 1e-4].
2. If ST-det is weak (<30 mAP50), use a detection-specific pretrained backbone (e.g., DINOv2, MAE) for the ST baseline only. This is defensible: the ST baseline should represent the best achievable single-task performance, not the performance of a suboptimal backbone.
3. Report the comparison transparently: "ST baselines use the same backbone as MTL for fair comparison. Detection-specific pretraining would improve ST by X% but is excluded to isolate the MTL effect."

**Contingency:** If ST baselines are irrecoverably weak (<50% of projected targets), the paper must shift its narrative from "MTL retains ST performance" to "MTL pathology characterization" (which is the stronger contribution anyway, per doc 216). The absolute numbers become less important than the diagnosis and fix.

**Decision trigger:** ST-det at epoch 30 (of 99). If mAP50 < 0.20, begin contingency. If < 0.30, consider detection-specific pretraining for ST.

---

### BASE-3: Equal weights (Kurin) baseline is untuned

**Probability: HIGH (70%)** — The `--equal-weights` flag exists but has never been run with the correct architecture (no frozen FPN, no broken RotoGrad, no clamp bug). The Kurin et al. (NeurIPS 2022) finding requires a WELL-TUNED equal-weight baseline to be meaningful.

**Impact: MEDIUM** — The paper needs to show FAMO beats equal weights to justify the complexity of FAMO. If equal weight with tuned hyperparameters matches FAMO within 1%, the paper's claim weakens significantly. If FAMO is worse than equal weights, the paper's central methodology is invalidated.

**Mitigation:**
1. Run the equal-weight baseline AFTER all bugs are fixed. Using a buggy equal-weight baseline and comparing it to a buggy FAMO baseline tells us nothing.
2. Tune the equal-weight baseline separately: search over LR [1e-5, 5e-5, 1e-4], pre-scaling factors, and gradient clip norm.
3. Plan to report the comparison honestly: "FAMO outperforms equal weights by X% on activity and Y% on PSR. Consistent with Kurin et al., equal weights is competitive on detection (difference <1%)."

**Contingency:** If FAMO does not beat equal weights statistically significantly, the paper's methodology contribution shifts: "We show that with proper pre-scaling, FAMO's adaptive weighting provides marginal benefit over equal weights for detection and activity, but significant benefit for PSR (which has the most dynamic loss landscape)." This is still publication-worthy as a negative result.

**Decision trigger:** After MTL run 1 (post-fix, FAMO), compare with equal-weights run. If combined metric difference < 0.02, run FAMO vs equal weights significance test with 3 seeds each.

---

## 3. Reviewer Risks

### REVIEW-1: "Why MViTv2-S when ViT-L/VideoMAE-L exists?"

**Probability: HIGH (70%)** — Reviewers familiar with video understanding will ask why the paper uses a smaller backbone when stronger alternatives are available. This is the most predictable reviewer objection.

**Impact: MEDIUM** — Strongly affects the "method novelty" dimension of the review. If not addressed, the paper may be desk-rejected as incremental.

**Mitigation:**
1. Document this explicitly in the paper: MViTv2-S is a deliberate choice for consumer GPU deployment (RTX 3060, 11 FPS). Larger models do not fit on 16GB VRAM (see Agent 06: MViTv2-L is 97M params, ~9 GB at 224px alone; total would exceed 16GB).
2. Frame as: "We prioritize realistic deployment constraints (consumer GPU, real-time or near-real-time inference). Scaling to larger backbones is future work."
3. Reference the efficiency audit: 46.47M total params (with FPN unfrozen: ~55.7M), 1.5GB VRAM, 11.02 FPS. "A 2x larger backbone would halve the frame rate and double VRAM — unacceptable for deployment."

**Contingency:** If a reviewer insists on larger backbone comparison, acknowledge the limitation and provide estimates from the literature. The MViTv2-B paper reports ~82.0% Kinetics-400 top-1 vs 80.8% for MViTv2-S. The marginal gain does not justify the 2x compute cost for this paper's purposes.

---

### REVIEW-2: "Why not just use 4 specialists?"

**Probability: HIGH (70%)** — This is the fundamental MTL question. Every reviewer will ask whether 4 separate single-task models are simpler and better.

**Impact: HIGH** — The paper's primary claim is that MTL saves parameters and latency. If the savings are marginal (~2x) and the cost is significant (collapsed heads, complex tuning), reviewers will prefer 4 specialists.

**Mitigation:**
1. The 2x parameter savings and 1.5 GB total VRAM (vs ~3 GB for 4 ST models) must be precisely quantified pre-submission. No fabricated numbers.
2. Emphasize the latency advantage: 1 forward pass vs 4, which is critical for real-time assembly feedback (11 FPS vs 2.75 FPS).
3. Document the factory pilot: the deployed system uses MTL, not 4 ST models. "In practice, running 4 forward passes introduces unacceptable latency for worker feedback."
4. The parameter efficiency argument is strongest when combined with the deployment evidence. A standalone parameter comparison is weak.

**Contingency:** If the 2x parameter savings are questioned as insufficient, highlight the PILOT deployment as the deciding factor: "The real-world deployment validated that MTL's single-pass architecture met the sub-100ms latency requirement; 4 specialists could not." This is an operational requirement, not a convenience.

---

### REVIEW-3: "Equal weights probably works just as well"

**Probability: MEDIUM (40%)** — The Kurin finding is well-known in the MTL community. A knowledgeable reviewer will ask about equal-weight baseline comparison.

**Impact: MEDIUM** — Undermines the FAMO methodology contribution but not the overall paper. The paper's core contribution (pathology framework) is independent of FAMO's superiority.

**Mitigation:**
1. Run and report the equal-weight baseline (BASE-3).
2. Frame the FAMO comparison honestly: "FAMO shows consistent improvement over equal weights for the two hardest tasks (activity: +X%, PSR: +Y%) while matching or near-matching on detection and pose."
3. If the difference is small: "We attribute FAMO's marginal benefit to the well-tuned pre-scaling factors, which already address the bulk of the scale disparity. FAMO provides robustness to loss landscape shifts during training."

**Contingency:** If FAMO equals or underperforms equal weights, the paper becomes: "Pre-scaling + equal weights is sufficient for this task set; FAMO provides a marginal safety margin." The methodology contribution shifts to the pre-scaling analysis (which factors to choose and why), not the adaptive weighting.

---

### REVIEW-4: "Your dataset is too small (36 subjects)"

**Probability: HIGH (70%)** — 36 subjects across 4 assemblies with ~26K frames is small by modern video understanding standards. Reviewers from large-dataset communities (Ego4D: 3,670 hrs, EPIC-Kitchens: 100 hrs) will flag this.

**Impact: MEDIUM** — Affects the "generalizability" dimension. Can be mitigated with proper framing.

**Mitigation:**
1. Frame the dataset contribution as "IndustReal: a challenging real-world industrial assembly dataset with dense multi-task annotations." Small datasets with high-quality annotations are valuable precisely because they are realistic (real factories are not YouTube).
2. Emphasize the annotation density: per-frame detection (24 classes), per-frame activity (75 classes), PSR (11 binary), pose (6-DoF). Few datasets have this annotation density across 4 task types.
3. Report per-recording metrics with bootstrap CIs to show variance across subjects. If metrics are consistent across subjects (low variance), the 36-subject dataset is sufficient.
4. Reference the factory pilot as external validation.

**Contingency:** If per-subject variance is high (>20% relative), acknowledge openly: "Subject-specific variation is substantial, suggesting need for more diverse training data. This is consistent with the egocentric vision literature where head-mounted cameras produce subject-dependent viewpoints."

---

### REVIEW-5: "Industrial assembly is too niche"

**Probability: MEDIUM (40%)** — A reviewer from a general vision venue (WACV, CVPR) may consider the domain too narrow.

**Impact: LOW-MEDIUM** — This is an AAIML paper. AAIML values industrial relevance. If the paper is submitted to WACV instead, this becomes a HIGH risk.

**Mitigation:**
1. Frame contributions as generalizable MTL pathology findings, not assembly-specific results. "The three training pathologies are dataset-independent: any MTL system with heterogeneous tasks and sparse labels is susceptible."
2. The pathology framework (Kendall collapse under label sparsity, FPN prefix bug as representative misconfiguration) is the general contribution. The assembly domain is the testbed.
3. For AAIML specifically: the industrial application IS the value proposition. "Validated in a real factory" is the headline for AAIML reviewers.

**Contingency:** If a reviewer still considers the domain too niche, the paper is submitted to the wrong venue. This risk is about venue selection, not paper quality. Do not change the paper's core framing.

---

## 4. Timeline Risks

### TIME-1: Fixing all bugs takes longer than expected

**Probability: HIGH (70%)** — There are 8 confirmed bugs (Frozen FPN, Frozen RotoGrad, DetectionAugment clamp, Warm-start missing, expandable_segments missing, num_workers=0, Curriculum decay dead code, Activity BalancedSoftmax double-counting). Even simple fixes can have unexpected interactions.

**Impact: HIGH** — Each bug fix potentially introduces new bugs. The testing cycle (fix -> train -> verify -> find new issue) takes 1-3 days per cycle. If 3 cycles are needed, that's 9 days before a single valid training run starts.

**Mitigation:**
1. Fix in dependency order: (1) expandable_segments, (2) DetectionAugment clamp, (3) FPN prefix, (4) RotoGrad (or remove), (5) curriculum decay, (6) num_workers. Fix warm-start and BalancedSoftmax last (they are enhancements, not blockers).
2. Write a verification script for each fix: e.g., for FPN fix, verify `"fpn"` appears in optimizer param_groups. For clamp fix, verify min/max of augmented images.
3. Run a 100-batch smoke test after ALL fixes before starting a full training run. Check: loss decreases, all 4 heads produce diverse outputs, no NaN, GPU memory stable.

**Contingency:** If bug-fixing takes >5 days, skip RotoGrad entirely (remove from codebase), skip curriculum decay (static 0.40 is acceptable), skip warm-start (train from scratch). The remaining 5 fixes (expandable_segments, clamp, FPN, num_workers, BalancedSoftmax) are the minimum viable set.

**Decision trigger:** Day 2 of bug-fixing: if not all critical fixes (1-3) are verified, escalate to contingency (drop non-critical items).

---

### TIME-2: ST baselines + MTL run + equal weights = ~30 days

**Probability: HIGH (70%)** — Sequential computation estimate: ST baselines (~7 days) + MTL FAMO (~7 days) + MTL equal weights (~7 days) + ablation runs (~7 days) = 28 days. This is tight for a submission in late August / early October.

**Impact: CRITICAL** — If training cannot complete before the submission deadline, the paper has incomplete results.

**Mitigation:**
1. Parallelize ST baselines. ST-det, ST-act, ST-psr, and ST-pose can run simultaneously if GPU memory permits (~8 GB each at 480px B=1, so 4 x 8 = 32 GB > 16 GB limit). Run ST-det and ST-act sequentially, ST-psr and ST-pose on the same GPU in alternating configurations.
2. Reduce epoch count. The current 99-epoch schedule is from an older codebase. With proper initialization and warm-start, 50 epochs may suffice. Run a validation set at epoch 30, 40, 50. If metrics plateau, stop early.
3. Skip unnecessary ablations. The ablation suite in doc 216 has 7 experiments (A1-A7). Tier 1 (A1-A3) is required. Tier 2 (A4-A7) is negotiable. A4-A7 add ~14 days.
4. Use cloud GPU as overflow. Lambda Labs RTX 3090 (~$0.50/hr) for ST baselines while the local GPU runs MTL.

**Contingency:** If total time exceeds available window:
1. Report 2-task MTL (detection + pose) with activity and PSR as "ongoing work." This dramatically reduces training time (3 heads fewer).
2. Submit to a later deadline (NeurIPS 2027 workshop track, or even AAAI 2027).
3. Accept that the paper will be an arXiv preprint with incomplete results, filed as "work in progress."

**Decision trigger:** If ST baselines are not started by July 16 (3 days from now), the timeline is in critical jeopardy. If by July 20 the first MTL run hasn't started, seriously consider contingency 1 or 2.

---

### TIME-3: Each failed training run costs 1-7 days

**Probability: HIGH (60%)** — Given the bug history, it is likely that post-fix training will reveal new issues. A single 7-day MTL run that fails at epoch 30 due to OOM or NaN costs 3 days of GPU time wasted.

**Impact: HIGH** — Wasted GPU time directly delays the submission. Each failed run adds 7 days to the timeline.

**Mitigation:**
1. Run a "mini-MTL" sanity test before full runs: 5 epochs, 10% of data, T=4, 320px. Verify all 4 heads show learning, loss curves are correct, eval metrics make sense. This takes ~2 hours instead of 7 days.
2. After mini-MLT passes, run the full 50-epoch training.
3. Implement robust checkpointing: save every 5 epochs, keep last 3 checkpoints. Verify checkpoint loading works (restart from epoch 20 and confirm metrics match).

**Contingency:** After 2 failed full runs, stop and re-audit the codebase. Every failed run after the first is a sign of undiagnosed bugs, not bad luck.

**Decision trigger:** Mini-MLT at 5 epochs is the gate. If mini-MLT fails, do not start full training until it passes.

---

## 5. Novelty Risks

### NOVEL-1: Most components are from literature — what is truly new?

**Probability: HIGH (70%)** — FAMO (NeurIPS 2023), RotoGrad (ICLR 2022), MViTv2-S (CVPR 2022), MS-TCN (ICCV 2019), Varifocal (CVPR 2021), WIoU (AAAI 2022), TAL (ICCV 2021). Every major component is from existing literature. The paper's contribution is the COMBINATION, not the individual components.

**Impact: HIGH** — A reviewer's first reaction to "we combined FAMO + RotoGrad on MViTv2-S" is "what did you actually contribute?" If the combination is straightforward, the paper is an engineering exercise.

**Mitigation:**
1. The paper's primary novelty MUST be the pathology framework (three infrastructure-level training pathologies distinct from gradient conflict). This is documented in doc 216 as the core contribution. The FAMO + MViTv2-S combination is the testbed, not the contribution.
2. The novelty is the DIAGNOSIS, not the CURE. "We found three pathologies that silently degrade MTL systems, undetectable by standard monitoring." This is the unique contribution.
3. Supporting evidence for pathology generalizability: the 70% repository prevalence rate (Pathology 3: head-level gradient aggregation is missing from 70% of open-source MTL repos). This is the strongest generalizability claim.
4. The deployment validation (factory pilot, NASA-TLX, SUS) is a rare contribution that most MTL papers cannot offer.

**Contingency:** If the pathology framework is rejected as "not novel enough," the paper falls back to:
1. "First MTL benchmark for industrial assembly with 4 heterogeneous tasks" (a dataset/task contribution).
2. "First application of FAMO to 4-task heterogeneous MTL" (an application contribution).
3. The honesty/comparability framework (8 disclosures, comparability matrix) as a methodological contribution.

These are weaker but still publishable at AAIML (which values application over method).

---

### NOVEL-2: The 8-disclosure framework is a strength, but reads as defensive

**Probability: MEDIUM (30%)** — As noted in doc 216: "Our 8 disclosures currently read as excuses for weak numbers." The honest comparability framework is a unique contribution, but it must be framed positively, not defensively.

**Impact: LOW** — Frame-dependent. If presented as "We are the only paper honest enough to disclose X," it is a strength. If presented as "Our numbers are weak because of X," it is a weakness.

**Mitigation:**
1. Move the comparability table to the first results position (as recommended by doc 216). This sets the tone: "We care about methodological rigor, not about hiding weaknesses."
2. Frame each disclosure as a positive contribution: "First measured MTL detection cost on IndustReal" not "Our detection is only 64% of ST."
3. Lead the abstract with the pathology narrative, not the metric claims.

**Contingency:** If a reviewer dismisses the disclosures as "excuse-making," the paper may still be saved by the strength of the deployment evidence. A deployed system with careful documentation is harder to dismiss than a purely academic benchmark.

---

### NOVEL-3: FAMO is from 2023 — not novel for a 2027 paper

**Probability: MEDIUM (40%)** — FAMO was published at NeurIPS 2023. By AAIML 2027 (submission ~Oct 2026), FAMO will be 3 years old. Using a 3-year-old method without modification may seem stale.

**Impact: LOW-MEDIUM** — FAMO is not claimed as a novel contribution. The paper's primary contribution is the pathology framework. However, if the reviewer perceives the paper as "FAMO applied to assembly data," the novelty is weak.

**Mitigation:**
1. The paper must NOT claim "we use FAMO" as a contribution. It must claim "we characterize MTL failure modes" with FAMO as the training tool.
2. Frame FAMO critically: "FAMO, despite being state-of-the-art, fails to prevent activity collapse when the FPN is frozen (Pathology 2)." This positions the paper's contribution as BEYOND FAMO, not USING FAMO.
3. Reference FAMO's limitations (weight collapse with non-stationary losses, gradient direction conflicts) and show how the paper's findings explain why FAMO fails in certain configurations. The pathology framework is a META-contribution that explains WHY MTO methods fail.

**Contingency:** If compared unfavorably to "yet another FAMO application paper," redirect to the pathology framework and deployment evidence. "Our paper is not about FAMO. It is about why MTL fails in practice and how we fixed it."

---

## 6. Summary Risk Register

| ID | Risk | Prob | Impact | Mitigation | Contingency | Triggers |
|----|------|------|--------|------------|-------------|----------|
| **TECH-1** | FPN frozen (14.5M params) | 100% | CRITICAL | Fix prefix in param group | N/A (must fix) | Immediate |
| **TECH-2** | RotoGrad frozen (639K params) | 100% | CRITICAL | Add to optimizer or remove | Remove RotoGrad entirely | Immediate |
| **TECH-3** | Activity head collapse | 80% | CRITICAL | Fix FPN+RotoGrad+warm-start+clamp | Decoupled training, focal loss | Act uniq < 10 at epoch 10 |
| **TECH-4** | PSR uniform output | 70% | HIGH | Fix backbone gradients first | Publish as negative result | PSR event-F1 < 0.10 at epoch 20 |
| **TECH-5** | DetectionAugment clamp bug | 100% | HIGH | One-line clamp range fix | N/A (must fix) | Immediate |
| **TECH-6** | Warm-start broken | 100% | HIGH | Verify path or train ST | Train from scratch (weaker) | Immediate |
| **TECH-7** | 480px too low for small objects | 40% | HIGH | Use P2 features, TTA | Accept limitation | Compute bbox size distribution |
| **TECH-8** | OOM from fragmentation | 60% | MEDIUM | Set expandable_segments:True | Reduce T/batch | Immediate |
| **TECH-9** | Gradient starvation persists | 30% | HIGH | Monitor FAMO weights | Staged training | Any task weight < 0.05 |
| **TECH-10** | num_workers=0 bottleneck | 70% | MEDIUM | Increase to 2 or 4 | Accept slower training | Test 100 batches |
| **TECH-11** | FAMO weight oscillation | 35% | MEDIUM | Weight floor, EMA smoothing | Tune temperature | Weight delta > 0.3/step |
| **BASE-1** | Missing ST baselines | 100% | CRITICAL | Start ST runs immediately | Report partial baselines | Not started by Jul 16 |
| **BASE-2** | ST baselines are weak | 35% | HIGH | Tune ST hyperparameters | Shift to pathology narrative | ST-det mAP50 < 0.20 at epoch 30 |
| **BASE-3** | Equal weights matches FAMO | 70% | MEDIUM | Run both, report honestly | Shift narrative to pre-scaling | Combined diff < 0.02 |
| **REVIEW-1** | "Why not larger backbone?" | 70% | MEDIUM | Consumer GPU constraint | Scale estimates from literature | Not applicable |
| **REVIEW-2** | "Why not 4 specialists?" | 70% | HIGH | Parameter savings + latency + deployment | Pilot evidence | Not applicable |
| **REVIEW-3** | "Equal weights works fine" | 40% | MEDIUM | Run and report honestly | Shift to pre-scaling contribution | Not applicable |
| **REVIEW-4** | "Dataset too small" | 70% | MEDIUM | Annotation density + bootstrap CIs | Acknowledge openly | Not applicable |
| **REVIEW-5** | "Industrial assembly too niche" | 40% | LOW | General pathology framework | Venue is AAIML (niche = value) | Not applicable |
| **TIME-1** | Bug-fixing takes too long | 70% | HIGH | Fix in dependency order | Drop non-critical items | Day 2 not complete |
| **TIME-2** | Training takes 30+ days | 70% | CRITICAL | Parallelize, reduce epochs, cloud GPU | Report partial results, later venue | ST not started by Jul 16 |
| **TIME-3** | Failed runs waste time | 60% | HIGH | Mini-MLT gate before full runs | Stop-and-reaudit after 2 failures | Mini-MLT fails |
| **NOVEL-1** | All components from literature | 70% | HIGH | Pathology framework is primary novelty | Dataset + application contribution | Not applicable |
| **NOVEL-2** | Disclosures read as defensive | 30% | LOW | Frame positively, lead with comparability | Reviewer-dependent | Not applicable |
| **NOVEL-3** | FAMO is 3 years old | 40% | LOW | Pathology is beyond FAMO | Redirect to primary contribution | Not applicable |

---

## 7. Minimum Viable Path to Submission

Given the current state (8+ bugs, no valid training run, no ST baselines), the minimum viable path is:

### Stage 1: Emergency Bug Fix (Days 1-3)
1. Fix expandable_segments:True (1 line)
2. Fix DetectionAugment clamp (1 line)
3. Fix FPN prefix (1 character)
4. Remove RotoGrad entirely (delete or comment out — reduces risk)
5. Fix num_workers=2 (1 line)
6. Write verification script for all fixes
7. Run mini-MLT test (5 epochs, 10% data, T=4, 320px)

**Gate:** Mini-MLT must show all 4 heads learning, all loss curves decreasing, act_preds > 10 unique, PSR components varying across frames.

### Stage 2: ST Baselines (Days 3-12)
8. Start ST-det and ST-act in parallel (sequential on single GPU: ~4.6 days)
9. Start ST-psr and ST-pose (ST-pose checkpoint may already exist)
10. At ST epoch 30, check if early stopping is possible

**Gate:** ST-det mAP50 > 0.20, ST-act top-1 > 30% at epoch 30. If not, tune hyperparameters.

### Stage 3: MTL Runs (Days 12-26)
11. MTL FAMO run (7 days, 50 epochs, 480px, T=8)
12. MTL equal-weights run (7 days, 50 epochs)
13. Ablation A2 (capped vs uncapped Kendall — already obsolete but for the record, run 2 days)
14. Ablation A3 (fixed-weight MTL — 7 days)

**Gate:** MTL combined metric > 0.30 at epoch 30. If not, initiate contingency paper framing.

### Stage 4: Ablations + Writing (Days 26-45)
15. Remaining Tier 2 ablations (A4-A7) if time permits
16. Test split evaluation
17. Figures, tables, writing
18. Internal review and claim verification

### Stage 5: Submission (Day 45)
19. Camera-ready freeze
20. Submit to AAIML or WACV depending on result quality

**Total optimistic timeline:** 45 days from now = August 27. This assumes zero failures, zero re-runs, and perfectly parallel execution. Realistic: 60-70 days = September 11-21. This fits the WACV/AAIML submission window (usually late August to early October) but leaves zero margin.

---

## 8. Go/No-Go Decision Points

### DP1 (Day 3): Do ST baselines show promise?
- **Go:** ST-det mAP50 > 0.15, ST-act top-1 > 25% at epoch 20 (10% of 200 epochs).
- **No-go:** ST baselines are themselves collapsing or not learning. Root cause may be backbone pretraining, data quality, or hyperparameters. Pivot: diagnose backbone issue, consider ConvNeXt-Tiny fallback (which at least has documented working results).

### DP2 (Day 12): Does MTL training start showing non-collapsed behavior?
- **Go:** After 10 epochs (2% of 500), act_preds > 30 unique, PSR comps spanning > 0.2 range, detection mAP50 > 0.05, pose loss < 200 (raw).
- **No-go:** Any head still collapsed. Pivot: re-verify bug fixes, add staged training, consider 3-task MTL.

### DP3 (Day 22): Are MTL metrics approaching ST baselines?
- **Go:** Combined MTL/ST ratio > 0.60 at epoch 30.
- **No-go:** MTL is substantially worse than ST on 2+ heads. Pivot: shift paper narrative to "Understanding MTL Failure" rather than "Successful MTL."

### DP4 (Day 40): Are final metrics strong enough for the target venue?
- **Go (AAIML):** Combined metric > 0.35 with at least 2 of 4 heads showing MTL/ST > 0.70. Honest disclosure framework intact.
- **Go (WACV):** Combined metric > 0.45 with 3 of 4 heads showing MTL/ST > 0.75. Stronger results needed for general venue.
- **No-go (any venue):** Publish as arXiv preprint with the pathology framework as the primary contribution. The negative training results (multiple heads collapsing, FPN bug) are themselves publishable as a cautionary study.

---

## 9. The Most Dangerous Scenario

The most dangerous scenario is not that training fails — it is that training partially succeeds but produces plausible-looking but wrong metrics due to an undiagnosed eval bug. The project has already experienced this (the activity false negative in the overfit probe that required careful interpretation to understand it was a probe design issue, not an eval bug).

If the paper submits with buggy results and the bug is discovered after acceptance (or worse, after publication), the reputational damage outweighs the benefit of the publication.

**Mitigation:**
1. Run overfit probes on the FIXED codebase. Every head should be able to memorize a single training sample if given enough capacity.
2. Write eval unit tests: synthetic data with known ground truth, compare computed metrics against hand-calculated values.
3. For detection: generate 5 images with known bounding boxes at known positions, run through eval pipeline, compare mAP against pycocotools reference.
4. For PSR: create a 100-frame sequence with known transitions at known timestamps, verify event-F1 computation.
5. For activity: create a 100-sample dataset with known class labels, verify top-1, top-5, macro-F1.
6. For pose: numerically verify the geodesic loss against a known formula reference (published literature has closed-form expressions for geodesic distance between rotation matrices).

**This is non-negotiable.** A retracted paper would be worse than no paper. Run the eval verification before any experiment that generates publishable numbers.

---

## 10. Final Honest Assessment

**Current state:** The project has 0 valid training results. Every observed metric is contaminated by at least one confirmed bug. The codebase audit (Agents 08, 09) has identified and verified the bugs with source code line references. None of the bugs are unfixable.

**Outlook:** The project is recoverable. The fixes are well-understood and mostly one-line changes. The activity collapse and PSR uniformity observed in logs are likely consequences of the bugs, not fundamental architecture limitations. The paper's core contribution (pathology framework) is interesting, timely, and does not depend on any particular metric value.

**But the timeline is the real enemy.** Every day of bug-fixing delays the ST baseline runs, which delays the MTL runs, which delays the paper. If the first MTL run fails, there may not be time for a second attempt before the submission deadline.

**The recommendation:** Fix the bugs aggressively (3 days), run mini-MLT to validate (6 hours), start ST baselines immediately (parallel with writing). The paper should be drafted with the pathology framework as the primary contribution, metrics as supporting evidence. If metrics are strong at submission time, great. If not, the paper still stands on the pathology analysis — which is the stronger contribution anyway.

**The paper will succeed or fail on the pathology framework, not on the absolute metrics.** The metrics are evidence for the pathology diagnosis, not the paper's result. Internalize this and the project's risk profile changes from "metrics might be too low" to "the pathology story might not convince reviewers." The latter is a risk that can be managed with clear writing, compelling figures, and honest framing. The former is a risk that depends on GPU uptime and the reliability of CUDA memory management — things the team cannot fully control.
