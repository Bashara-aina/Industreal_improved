# 137 — Head Pose: Adversarial Debate

**Date:** 2026-07-06
**Target:** `137_HEAD_POSE_DEEP_QUESTIONS.md` — 50 deep questions for Opus on head pose evaluation
**Role:** Adversarial debate agent — challenge the narrative, identify gaps, surface alternative interpretations, and propose new questions not yet asked.

---

## §A. Five Strongest Challenges to the Head Pose Narrative

### Challenge 1. The up-vector "advantage" over forward is mechanically guaranteed, not a model achievement

**Claim:** Up-vector achieves 7.78deg vs forward 9.14deg, suggesting the model is "better" at predicting up.

**Challenge:** In a typical assembly posture, the operator's head is pitched ~30deg downward toward the workbench. The up-vector in this configuration is dominated by the gravity vector (world-up rotated by pitch), which varies primarily with pitch and roll. The forward vector encodes both yaw and pitch — two DoF that vary widely as the operator looks around the workspace. The up-vector is physically more constrained: in seated assembly, the head's up-direction has a much smaller angular range than the forward direction. You are comparing two quantities with different intrinsic ranges and calling the easier one "better." The headline "up-vector 7.78deg vs forward 9.14deg" should be reframed as: "up-vector error is 15% lower, consistent with its smaller angular variance in the dataset."

**Counter:** The angular range difference is testable. Compute the standard deviation of per-frame GT up-vector and forward-vector angular displacements. If up-vector's angular std is 10deg and forward's is 25deg, then normalized by range, both tasks are equally hard (error/range ratio ~0.78 for both). We should report this ratio, not raw MAE. If we do not, a sharp reviewer at AAIML will compute it from their own understanding of HoloLens kinematics and say the emperor has no clothes.

---

### Challenge 2. The single-task ablation gap invalidates any multi-task attribution

**Claim:** Head pose is presented as a success of the multi-task architecture, alongside detection, PSR, and activity.

**Challenge:** There is no single-task pose baseline anywhere in the evidence inventory. The file's own A-4 acknowledges this as "a gap in the ablation study" and moves on. But this is not a gap — it is a foundational omission. Without a pose-only model (same ConvNeXt-Tiny backbone, pose loss only, no detection/activity/PSR heads), we cannot attribute a single degree of the 7.78deg accuracy to the multi-task design. For all we know, the detection head hurts pose through gradient conflict, and a single-task model would achieve 6.5deg. Or the multi-task design helps, and a single-task model would achieve 9.0deg. Either outcome changes the paper narrative dramatically. Running this experiment costs 2-3 GPU-hours (train pose-only from scratch for 18 epochs on the 130k-frame training set). The absence of this result is a reviewer grenade with the pin pulled.

**Counter:** Any sharp reviewer will ask "how does multi-task affect pose?" The answer "we didn't measure it" is fatal. The counter-strategy is to scope the claim: "We present orientation accuracy under the full multi-task model. The effect of individual task branches on pose accuracy is studied in our ablation analysis (Table X)." But Table X must exist. If the experiment cannot be run before freeze, remove all multi-task attribution language from the pose section.

---

### Challenge 3. The 3.5-month index bug is a quality-assurance catastrophe dressed as a "finding"

**Claim:** The index bug narrative is presented as a discovery — "we found and fixed a 3.5-month bug."

**Challenge:** A 26.20deg up-vector MAE is worse than random. A random unit vector on the sphere has expected angular error ~90deg, so 26.20deg is not catastrophic per se — but for a model already producing 9deg forward, seeing 26deg on the other output should have triggered an investigation immediately. The file's own Q7 asks "why did 26.20deg survive plausibility checks?" but the answer is tepid ("lack of reference for what good looks like"). The adversarial interpretation is sharper: the team had no systematic review process, no sanity bounds, and no cross-validation between independent metrics for 3.5 months. The "finding" is not that the bug existed, but that the team's infrastructure could not detect a 3x discrepancy for over a quarter. And the only reason it was caught was the Kalman eval accidentally used correct indices, revealing inconsistency. If the Kalman eval had also been written with the same bug (as head_pose_diag.py still is), the bug would still be alive today.

**Counter:** The honest defense is: "We caught it, we fixed it, and we are disclosing it transparently — most papers would never admit such a bug existed." This is actually a stronger position if framed correctly. But it requires not soft-pedaling the severity. The file says the bug "went unnoticed across 3 eval script generations." That is not a description of a process that inspires confidence in any other number in the paper.

---

### Challenge 4. The GT noise floor is hand-waived, and it could invalidate the entire comparison

**Claim:** The model achieves 7.78deg on up-vector. The GT accuracy of HoloLens 2 in optimal conditions is 1-3deg. In industrial settings with vibration, rapid motion, and occlusion, GT noise could be 3-5deg.

**Challenge:** The file speculates GT noise at 3-5deg but has zero evidence. More critically: if GT noise is 5deg, the model's true accuracy (against an infinite-precision reference) is sqrt(7.78^2 - 5^2) = 5.96deg. If GT noise is 7deg (not unlikely for rapid assembly motion with frequent HoloLens tracking loss), the model's true accuracy would be sqrt(7.78^2 - 7^2) = 3.40deg — an extraordinary result that would be a paper highlight. But we do not know because we never characterized the GT. The paper cannot claim "7.78deg up-vector MAE" as a model accuracy without reporting the reference accuracy and deconvolving it. Every published ego-pose paper on HoloLens data has this limitation, but that does not make it acceptable to ignore.

**Counter:** The standard dodge is "we report against the same GT that any competing method would use, making the comparison fair." This is defensible for a benchmark paper. But it becomes indefensible if the paper claims to "beat SOTA" or "establish a baseline" — because the GT noise means the true margin may be much larger or smaller than reported. The fix: measure HoloLens tracking confidence per frame (if available in pose.csv), compute MAE on high-confidence frames (confidence > 0.9), and report both numbers. If high-confidence MAE is significantly lower, the lower number becomes the truthful headline.

---

### Challenge 5. The forward MAE of 9.14deg may be inflated by recordings where the model collapsed to a mean prediction

**Claim:** Per-recording forward MAE ranges from 6.07deg (24_assy_2_4) to 17.05deg (14_assy_0_1). The weighted mean is 9.14deg.

**Challenge:** A range of 11deg between best and worst recording on a supposedly homogeneous task is enormous. It suggests the model is not learning a generalizable forward vector predictor, but rather memorizing recording-specific cues. The worst recording (14_assy_0_1 at 17.05deg) is more than 2.8x worse than the best recording (6.07deg). For the up-vector, the ratio is 2.2x (5.71deg to 12.32deg). If the model were genuinely learning head pose, the per-recording variance would be much smaller — typical inter-recording variance in published HoloLens pose papers is 15-25% of the mean, not 190% (for forward). This pattern is consistent with: (a) the model overfitting to recording-specific visual backgrounds (workbench color, lighting, tool presence), or (b) the model using temporal shortcuts (adjacent frames are similar, so the MLP head predicts similar outputs) rather than true visual understanding.

**Counter:** A strong rebuttal requires computing per-recording forward MAE systematically (which the file notes is not yet done — only up-vector has per-recording analysis). If the forward per-recording analysis reveals that 3-4 recordings drive the mean, the paper should report median-of-per-recording-forward-MAE (like they do for up-vector) and see if the number is closer to 7deg. If so, the 9.14deg is an artifact of a few hard recordings, not the model's typical performance. The honest headline would be: "Median per-recording forward MAE: ~7.5deg, with a weighted mean of 9.14deg due to 3 outlier recordings."

---

## §B. Five Evidence Gaps

### Gap 1. The training loss function's index slicing is unverified

The file's Q3 asks whether the training loss uses [3:6] or [6:9] for up-vector, but the answer is not provided. The file references evaluate.py line 1280 as the location of the head pose loss. This is the single most important fact in the entire head pose analysis: if the training loss used [3:6] (position data) as the up-vector target, then the model was trained to regress position coordinates into the up-vector output channels (index 6:9). The corrected eval would then show 7.78deg not because the model is good, but because the eval now reads from the correct channels — channels that were trained on position data. This would mean the model is not predicting orientation at all in the [6:9] channels, and the 7.78deg number is meaningless. **This must be confirmed before any paper submission.**

### Gap 2. No per-recording forward MAE exists anywhere in the evidence

The file is meticulous about per-recording up-vector breakdown (Q21-Q30) but notes that per-recording forward MAE has never been computed. The pose_kalman_results.json apparently contains per-recording forward data (the file quotes forward per-recording numbers like 6.07deg and 17.05deg from it), but a formal per-recording table or analysis file does not exist. Without this, the 9.14deg headline is an opaque aggregate that could hide bimodal performance.

### Gap 3. The frame-level error distribution is unknown

Q13 asks for the frame-level error histogram. The only reported std is 1.74deg, which is over per-recording MEANS, not frames. The frame-level dispersion could be 5-10deg. If the 90th percentile of frame-level error is 25deg, the model produces unusable predictions on 10% of frames. The mean MAE of 7.78deg hides this entirely. A claim like "our model achieves 7.78deg MAE" implies all frames perform near this level — almost certainly false if the distribution is heavy-tailed.

### Gap 4. The correlation between forward and up errors has not been computed

Q15 asks for this analysis, noting that if errors are uncorrelated, the model is more robust. This is trivially computable from cached inference outputs (38,036 frames, each with predicted and GT forward+up). A single Python script, 20 lines, 5 minutes to run. Its absence means a key robustness claim (or vulnerability) is undocumented.

### Gap 5. The linear probe experiment — the highest-value cheap experiment — is not run

Q16 identifies this: train a linear classifier (Linear(768, 6)) on frozen ConvNeXt-Tiny features using angular loss. If the linear probe matches or exceeds 7.78deg/9.14deg, the backbone features are already optimal and the MLP head is adequate. If the linear probe is significantly worse, the backbone is the bottleneck. If the linear probe is significantly better, the current head is under-trained or poorly designed. This experiment costs 1 GPU-hour and directly answers whether model architecture improvements can reduce head pose error by 0.5deg, 2deg, or 5deg. Without it, any discussion of "headroom" is speculative.

---

## §C. Five Alternative Interpretations

### Interpretation 1. The HoloLens GT itself may have systematic bias on the up axis in assembly postures

The inside-out tracking of HoloLens 2 relies on visual features in the environment. In an industrial assembly setting with metallic, reflective, or textureless surfaces (common on workbenches and parts), the tracking system may lose lock or drift. The up-vector (gravity-aligned axis in world coordinates) is typically the most robust axis for IMU-based systems because gravity provides an absolute reference. If the IMU's gravity estimate drifts due to linear accelerations during assembly motion, the GT up-vector could be biased by 2-4deg. The model, predicting from visual input alone, would match this biased GT at 7.78deg — meaning the model's true accuracy could be higher than reported. Alternatively, if the model learned a different gravity estimate than the HoloLens IMU, the 7.78deg could be the disagreement between two estimators, both of which are partially wrong.

### Interpretation 2. The 14_assy_0_1 outlier may not be a hard case — it may be a dataset artifact

The file proposes four hypotheses for the outlier (different motion, corrupted GT, different camera mount, unusual visual conditions). But a simpler interpretation: recording 14_assy_0_1 may have been annotated by a different MTurk worker who systematically marked assembly actions at slightly different head orientations. The annotation instructions for "assembly" frames may have been interpreted differently by this worker. If the model predicts the true head orientation (per HoloLens) but the frame labels imply a different segment of the recording, the evaluation (which only evaluates on frames labeled as that action) could sample systematically different head poses. The fix: load the video for 14_assy_0_1, check if the action boundaries (assembly vs non-assembly) are consistent with other recordings from series 14.

### Interpretation 3. The forward vector's higher MAE (9.14deg vs 7.78deg) might reflect that forward is the MLP's "spare capacity" channel

In a multi-task model, the pose head is a single linear layer on GAP-pooled features. If the backbone allocates representation capacity to the detection and PSR tasks (which have higher gradient magnitude due to their loss weights), the pose head may get residual features. Within the pose head, the network may implicitly prioritize the forward vector (which affects the gaze direction most directly) at the expense of the up-vector, or vice versa. The up-vector's lower error could be because it's easier to predict from the same features (gravity direction from ceiling/wall lines), not because the model is better at it. The forward vector requires more subtle visual cues (the direction the person is facing relative to the environment), which the features may not encode well.

### Interpretation 4. The position channels [3:6] may actually be correctly calibrated — config.py:853 may be stale

The file says config.py:853 states "DO NOT REPORT mm/cm — unit uncertain." This could be a stale comment from an earlier version of the codebase when the HoloLens coordinate system was not yet confirmed. If the training pipeline has been successfully using these values for regression, and the loss function produces reasonable values (single-digit position error in plausible units), the "unit uncertain" warning may be overcautious. The adversarial interpretation: the team accepted the warning without re-verifying whether the units are known by now. If they are indeed meters, position error of X cm is a publishable result. If they are not meters, the paper should explain why the model was trained on an uncalibrated target.

### Interpretation 5. The Kalman smoother's marginal improvement (0.21deg) could be selecting for easy frames, not smoothing

The RTS smoother uses per-channel independent Kalman filters with Gaussian noise assumptions. If the measurement noise (R = 0.200) is too high relative to the prediction noise, the smoother will trust the predictions more than the measurements for high-innovation frames — effectively ignoring frames where the model disagrees with GT. The 0.21deg improvement might come entirely from the smoother attenuating high-error predictions rather than actually smoothing any temporal structure. The per-recording breakdown shows some recordings get 0.8deg improvement (05_assy_2_2) while others get negative improvement — consistent with the smoother being a sophisticated way to suppress outliers. If this is the case, the claim that "predictions are already temporally smooth" is wrong; the correct statement is "the smoother filters out high-error frames, which happen to cluster temporally."

---

## §D. Five New Questions Not Asked in the Original

### NQ-1. What is the per-recording breakdown for forward MAE?

The original provides meticulous per-recording analysis for up-vector (Q21-Q30) but notes that per-recording forward MAE has never been computed or reported as a standalone analysis. The Kalman data appears to contain it (forward numbers are quoted per-recording), but there is no dedicated analysis file, table, or figure. This is the single most important missing analysis: if the forward per-recording median is 7.5deg but the mean is 9.14deg, the headline should change. And if the worst-recording forward MAE (14_assy_0_1 at 17deg) is 3x the outlier-free median, the paper has an outlier problem on both vectors, not just up-vector.

### NQ-2. How does the 7.78deg up-vector compare to OpenFace or 6DRepNet on the same frames?

The file correctly notes (Q47) that OpenFace requires a face crop and fails on egocentric video. But the frames DO contain faces — the operator's own face is occasionally visible in reflections (HoloLens display reflection, metal surfaces, monitor screens). Additionally, the question should be: "Can we crop the lower portion of the egocentric frame (where the operator's body/chest is visible) and run a body-pose estimator that also predicts head orientation from body configuration?" This is a different question from face-pose estimation. The comparison would establish whether the head pose signal can be recovered from the body proxy (arms, shoulders visible in the frame) rather than from the scene content.

### NQ-3. Is there a domain shift between the 9-recording v3 subset and the full 16-recording set?

The file notes that up_vector_v3 covers only 9 of 16 recordings and its median-of-medians (5.82deg) is substantially better than the full weighted mean (7.78deg). The file recommends recomputing on all 16, but does not ask the follow-up: "Is the 9-recording subset biased toward easier recordings because the v3 script was only run on recordings that already had passing quality checks?" If the 7 excluded recordings were systematically harder (14_assy_0_1, 05_assy_2_2, 26_assy_0_1, 26_main_0_1 are all in the lower half), then the v3 analysis underrepresents the hardest cases by 50%-plus. The v3 5.82deg number should not appear anywhere without a caveat that it covers the easier 56% of recordings.

### NQ-4. What happens if we evaluate on the training set? Is the model overfitting to recording-specific backgrounds?

If the model achieves 6deg on the training set (130k frames) but 7.78deg on the validation set (38k frames), the 1.78deg gap is modest overfitting. But if the training set MAE is 4deg, the model is severely overfit to recording-specific backgrounds (workbench color, lighting, tool arrangement) and will not generalize to new environments. This experiment costs nothing — run inference on the training set from cached features. The training/validation gap for head pose has never been reported.

### NQ-5. What is the angular error conditioned on the other task predictions?

Does head pose error correlate with detection failure (missing object detections due to occlusion)? Does it correlate with activity misclassification (angry vs happy vs neutral)? The multi-task architecture means all heads share the same backbone. If head pose error is elevated on frames where detection also fails, the shared features are the bottleneck (occlusion, motion blur affects all tasks). If head pose error is independent of other task errors, the pose head has unique failure modes. This analysis costs 1 hour and directly informs whether head pose improvements require backbone improvements (detection success) or head-specific improvements (pose failure despite good detection).

---

## §E. Overall Verdict

The 137 analysis is thorough — 50 well-structured questions that catch most of the important issues. The strengths are: the index-bug forensic reconstruction (Q1-Q10), the per-recording outlier analysis (Q21-Q30), and the honest presentation of the SOTA comparison dilemma (Q41-Q50). The weaknesses that this adversarial brief targets are:

1. **Missing empirical anchors.** The linear probe (Q16, cost 1 GPU-hour), per-recording forward MAE (cost 10 minutes), frame-level error histogram (cost 5 minutes), and training/validation gap (cost 30 minutes) are all computable from cached data or short experiments. Their absence means the paper makes claims about model quality without measuring the model's actual capabilities.

2. **Over-attribution to multi-task design.** Without a single-task ablation, every claim that head pose benefits from the multi-task architecture is speculation. The A-4 response treats this as a minor gap; this adversarial review treats it as a potentially paper-threatening omission.

3. **Under-characterization of the reference standard.** The HoloLens GT noise analysis (Q14) is noted as important but not acted upon. The position of GT noise in the debate — it could make the model look worse (7.78deg against noisy GT) or better (3.4deg against de-noised GT) — means the paper has an unquantified 2x range in its fundamental result.

**Recommendation for Opus:** Before writing the head pose section, run the following experiments in order:
1. Verify training loss slices (Q3): check evaluate.py line 1280 for head pose index usage.
2. Linear probe (Q16): does backbone already encode pose at 7deg or 12deg?
3. Per-recording forward MAE table: is 9.14deg a true mean or outlier-driven?
4. Frame-level error histogram: what is the 90th/95th percentile up-vector error?
5. High-confidence GT subset analysis: if HoloLens tracking confidence is available, what is the MAE on frames with confidence > 0.9?

These five experiments, costing ~3 GPU-hours total, would transform the head pose section from "well-analyzed uncertainties" to "data-driven claims with quantified bounds."

---

**End of 137_HEAD_POSE_DEBATE.md. Reviewed against 137_HEAD_POSE_DEEP_QUESTIONS.md.**
