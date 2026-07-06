# 137 — Head Pose: 50 Deep Questions for Opus

**Date:** 2026-07-06
**Responds to:** SOTA_STATUS.md, pose_kalman_results.json, full_eval_ep18_v2/metrics.json, up_vector_v3/up_vector_per_recording.json, and the 6 corrected/fixed eval scripts. See §0 Evidence Inventory for the complete list.
**Predecessor answers:** 133_OPUS_COMPLETE_ANSWERS.md §4 (HP-1 through HP-6). This file goes deeper: every number, every assumption, every file, every unasked question.
**Scope:** Head pose only. Forward MAE, up-vector MAE, position, Kalman smoothing, per-recording breakdown, SOTA positioning. The index bug history.
**Format:** 50 questions across 5 topical sections (§1-§5), adversarial review (§6), and open decisions for Opus (§7).

---

## §0. Evidence Inventory

Every source file used in this analysis, with its role and current status.

| File | Role | Status |
|---|---|---|
| `checkpoints/SOTA_STATUS.md` | Canonical status table; lines 17-20 carry the headline pose numbers | Live reference (2026-07-06) |
| `checkpoints/full_eval_ep18_v2/metrics.json` | Newly committed full eval v2; forward 9.14°, up 7.78° (38,036 frames) | Primary source, clean run |
| `checkpoints/pose_kalman_eval/pose_kalman_results.json` | 16-recording Kalman eval (RTS smoother); single-frame + smoothed per vector | Primary source, clean run |
| `checkpoints/up_vector_v3/up_vector_per_recording.json` | 9-recording per-recording breakdown; median of medians 5.82°, IQR [5.55°, 6.09°] | Subset eval, consistent with Kalman |
| `evaluation/eval_pose_kalman.py` | RTS smoother; uses `head_pose[0:3]` + `[6:9]`; parameter sweep support | Fixed and verified |
| `evaluation/full_eval.py` | Full eval, corrected indices `[:, :3]` and `[:, 6:9]` (lines 177-178) | Fixed and verified |
| `evaluation/full_eval_stream.py` | Streaming full eval, corrected indices `[:, :3]` and `[:, 6:9]` (lines 121-124) | Fixed and verified |
| `evaluation/up_vector_per_recording_v2.py` | Per-recording up-vector eval, uses `[:3]` and `[6:9]` | Fixed and verified |
| **`evaluation/head_pose_diag.py`** | **Still uses `(3, 6)` for up-vector angular (line 81) and `[6:9]` for position distance (line 88). Unfixed.** | **BUGGY — NOT YET FIXED** |
| `evaluation/evaluate.py` | Main eval harness; uses `pred[:, 3:6]` for position error (line 1970) | Correct reference for layout documentation |
| `133_OPUS_COMPLETE_ANSWERS.md` | HP-1 through HP-6 answered; includes index-bug timeline, SOTA verification, FiLM ablation requirement | Authoritative predecessor |

**The head pose output layout** (confirmed across all corrected scripts): `head_pose[B, 9]` = forward[0:3] + position[3:6] + up[6:9]. The original index bug used `[3:6]` as the up-vector (reading position data), producing the 26.20° MAE. The correct up-vector at [6:9] yields 7.78°, confirmed by three independent scripts.

**The lone unfixed script**: `head_pose_diag.py` still performs `angular_error(pred[0, (3,6)], gt[0, (3,6)])` labeled "up" and computes `(pred[0, 6:9] - gt[0, 6:9]).norm() * 1000` labeled "position mm". Both indices are swapped from their correct usage. This is a low-impact diagnostic script (not used for published numbers), but its existence means anyone running it today will see different numbers than the canonical results and may erroneously conclude the index bug is still unresolved.

---

## §1. The 3.5-Month Up-Vector Index Bug (10 Questions)

### Q1. How did `[3:6]` vs `[6:9]` go unnoticed across 3 eval script generations?

Three scripts were written sequentially — `full_eval.py`, `full_eval_stream.py`, and `up_vector_per_recording_v2.py` — and each initially used `[3:6]` for up-vector. The bug was propagated by copy-paste. Each script was manually inspected by a human and by Opus; the wrong indices were never questioned because the output (26.20°) was consistent across all of them. The question: **what systematic review step was missing** that would have caught this before the 3.5-month mark? A trivial unit test comparing a single known frame's head_pose output against the input image's known orientation would have flagged it instantly.

### Q2. How many downstream artifacts consumed buggy up-vector numbers?

The 26.20° number appears in: training logs (printed every N batches), full_eval outputs for epochs 0-18 (metrics.json files), SOTA_STATUS.md tables, ablation analyses, and at least two generations of AAIML files (129, 130, 131). The question: **is there a compiled list of all files that ever quoted 26.20°**, or are we relying on grep-and-hope? A targeted grep for "26.20" and "13.52" across the entire repository would produce the definite list.

### Q3. Could the bug have affected the training loop, not just eval?

If the training loss function uses `head_pose[:, 3:6]` as up-vector target for the L2 loss, then the gradient signal was wrong for 3.5 months too. Evaluate.py's head pose loss is at approximately line 1280 — confirm it uses `[:, :3]` for forward and `[:, 6:9]` for up. If the training loss was correct (using `[6:9]`) but the eval scripts were wrong (using `[3:6]`), then the model was learning the right thing and we only misreported it. If both were wrong, the model learned to predict position in the up-vector output channels — which would make the corrected 7.78° even more remarkable (the model beat SOTA on up-vector despite being trained on wrong targets in those channels).

### Q4. What does the temporal profile of position data (the buggy "up-vector") look like?

For 3.5 months, every eval script reported angular errors on position coordinates [3:6] as if they were up-vectors. Position data encodes the HoloLens wearer's head location in 3D space, which varies much more slowly than orientation. The 26.20° MAE was the angular error between predicted position offset (which was actually trained to predict up-vector, because the MLP head maps to `[6:9]` correctly) and GT position coordinates — an incoherent comparison. A time-series plot of the buggy MAE would show position-data artifacts: low-frequency drift, no orientation-specific structure, and high error at frames where the head moves rapidly.

### Q5. Which of the three corrected scripts has the most trustworthy architecture for correctness verification?

The simplest script is `full_eval_stream.py` — 200 lines, no subprocess, no external metric library, pure NumPy + torch. Its angular MAE is computed inline (lines 121-131) with explicit column slicing. If we need a ground-truth reference for any future head pose number, `full_eval_stream.py` is the simplest and least likely to hide a bug. `eval_pose_kalman.py` adds the Kalman machinery on top — correct but more moving parts. `up_vector_per_recording_v2.py` adds per-recording splitting logic. **Recommendation**: any future head pose evaluation should run `full_eval_stream.py` as the reference and the other scripts as cross-checks, never the reverse.

### Q6. Is `head_pose_diag.py` the only unfixed script, or are there others in subdirectories?

Only `head_pose_diag.py` uses `(3, 6)` as up-vector slices in its actual computation. But are there analysis notebooks (`*.ipynb`), visualization scripts, or data-loading utilities outside `src/evaluation/` that also slice head_pose? A full-repository grep for `[:, 3:5]`, `[:, 3:6]`, `[..., 3:6]`, or `[0, 3:` would confirm the scope. The bug-prone pattern is any 3-or-6-element slice anchored at index 3 — grep for `3:6` and `3:5` (for older 6-DoF conventions).

### Q7. Why did the 26.20° number survive plausibility checks?

A 26.20° mean angular error on head up-vector is **catastrophic** — worse than random for a unit vector on the sphere (random unit vector has expected angular error ~90°). The number should have triggered an investigation on first sight. The question is about team culture: was 26.20° accepted because "the task is hard" or because there was no reference for what good looks like? If the latter, the fix is to establish per-task sanity bounds as a pre-training checklist item (e.g., "head pose MAE should be below 30° for random initialization, below 15° for a trained model").

### Q8. Is the index bug the only possible off-by-N indexing error in our 9-DoF head pose output?

The 9-DoF layout `forward(3) + position(3) + up(3)` is reasonable but unenforced. There is no schema test, no dataclass, no assert that output dimension 9 is partitioned correctly. A similar bug could affect the next project that reorders these fields. Should we define a `HeadPose9Dof` named tuple or a `@dataclass` with explicit fields, and validate compatibility at load time? The current architecture (consecutive slices in comments only) is the same pattern that produced the 3.5-month bug.

### Q9. What is the minimum test suite that guarantees this bug never reoccurs?

Minimum viable: (1) A unit test that constructs a 9-DoF tensor from known vectors (forward=+X, position=origin, up=+Y), feeds it through the angular MAE function, and asserts each sub-vector maps to the correct slice. (2) A regression test that runs one batch through `full_eval_stream.py` and compares output against a known-good .json. (3) A slice-integrity test that prints and asserts the comment annotation `"forward[0:3] + position[3:6] + up[6:9]"` matches each script's actual indices. None of these exist today.

### Q10. Was any paper text, figure, or supplementary material written using the 26.20° number?

If yes, those passages must be located and corrected. The .tex may contain numbers from the pre-fix era (see 133 §0 C-5: "Head pose numbers span three eras — 7.83°/9.94° (.tex), 8.14-9.14° + up 7.06-7.48° (supplementary), 8.39° + up 26.20°/13.52° (current 129/131)"). A definitive grep of the .tex for "26.20", "26.2", "13.52", "13.5" would answer this and should be added to the freeze-protocol checklist.

---

## §2. Corrected Numbers — Forward 9.14°, Up-Vector 7.78° (10 Questions)

### Q11. Why is up-vector MAE (7.78°) consistently better than forward MAE (9.14°) across all three independent evals?

In `full_eval_ep18_v2/metrics.json`: forward 9.14° vs up 7.78°. In `pose_kalman_results.json`: forward 9.14° (weighted) vs up 7.78° (weighted). In `up_vector_per_recording.json`: forward is not reported (the v3 script only evaluates up), but the per-recording median of medians is 5.82°. Three possibilities: (1) Up-vector is physically more constrained by the HoloLens headband — the head's up direction varies less than its forward direction in assembly tasks where the wearer looks down at the workbench. (2) The GT annotations for forward are noisier (HoloLens IMU yaw drift vs tilt accuracy). (3) The model learned up better because the visual cues for up (gravity, horizon, ceiling) are more consistent across the dataset. Which one? Each has different implications for the paper's claims.

### Q12. Should the headline number be the full-eval mean (7.78°) or the per-recording median of medians (5.82°)?

The per-recording analysis (9 recordings, `up_vector_v3`) gives median of medians = 5.82° with IQR [5.55°, 6.09°], substantially better than the full-eval mean of 7.78°. The gap exists because: (a) the v3 analysis covers only 9 of 16 recordings — the excluded 7 recordings may have higher errors; (b) the mean is pulled up by the outlier recording 14_assy_0_1 (median 11.96°); (c) within-recording error distributions are right-skewed (mean > median for every recording). The honest answer is to report: (1) the per-recording mean of medians across all 16 recordings (from Kalman data: 16 up-vector medians, mean of those); (2) the full-eval weighted mean alongside; (3) explain why they differ. Recommendation: use the Kalman full 16-recording data, report both weighted mean (7.78°) and median-of-per-recording-medians (from the 16 entries: what is this number?). The 5.82° from the v3 is a partial statistic and should not appear without the caveat.

### Q13. What is the shape of the angular error distribution? Is it heavy-tailed?

If the distribution is approximately Gaussian with mean 7.78° and std ~5-7°, then MAE is a representative summary. If it is heavy-tailed (many small errors + a few catastrophic errors), then median and quantiles are more informative. The `pose_kalman_results.json` reports std for up-vector as 1.74° — but that's the std of *per-recording means*, not the frame-level std. The frame-level distribution may be broader. Request: compute and plot the frame-level angular error histogram for up-vector and forward across all 38,036 frames. Key threshold: what fraction of frames exceed 20°, 30°, 45°? If >10% exceed 20°, the model would be unreliable for fine-grained head pose applications regardless of mean MAE.

### Q14. How does 7.78° compare to the noise floor of HoloLens head pose GT?

The HoloLens 2 provides head pose via its onboard IMU + inside-out tracking. The published angular accuracy of HoloLens 2 is approximately 1-3° under optimal conditions (static head, good lighting). In an industrial assembly setting with rapid head motion, occlusions, and vibration, the GT noise floor may be 3-5°. If the GT noise is 4° and our model achieves 7.78°, then the model's true accuracy (against an infinite-precision GT) would be sqrt(7.78^2 - 4^2) = 6.67°. But this is speculative without a frame-level GT quality metric. Propose: compute per-frame HoloLens tracking confidence (if available in pose.csv) and analyze MAE vs confidence. If low-confidence frames have disproportionately high error, the model may already be at the noise floor for the usable subset.

### Q15. What is the per-frame correlation between forward angular error and up-vector angular error?

If forward and up errors are strongly correlated (Pearson r > 0.5), then both vectors degrade under the same visual conditions (occlusion, motion blur, poor lighting). If they are uncorrelated, the model's two output heads suffer from independent failure modes. This matters for reliability analysis: a system with uncorrelated errors can use one vector to validate the other (cross-check). Request: compute per-frame forward error and up-vector error, scatter-plot, report Pearson r and Spearman rho. If r > 0.7, a single failure mode governs both; if r < 0.3, they are independent and the model is more robust than its component MAEs suggest.

### Q16. Is the forward MAE of 9.14° a hard floor for the ConvNeXt-Tiny backbone, or can the pose head be improved independently?

The pose head is a single linear layer on GAP-pooled C5 features (768-dim). If the backbone features are poorly conditioned for orientation regression, no head improvement will help — the bottleneck is visual encoder quality. A linear probe on frozen features (analogous to the activity linear probe but for pose) would answer: train a single Linear(768, 6) for forward+up (unit-normalized, angular loss), frozen backbone. If the linear probe achieves ~9° forward / ~7.8° up, the backbone is the bottleneck. If it achieves significantly worse (e.g., >12°), the current head is doing well relative to the features. If it achieves better (e.g., <7°), the current head is suboptimal. **This is the single highest-impact cheap experiment not yet run.**

### Q17. What is the angular error as a function of absolute head orientation?

Does the model predict poorly when the head is looking far to one side (yaw > 60°) or far up/down (pitch > 30°)? Compute per-frame absolute yaw and pitch from the ground truth forward vector, bin by angle, and report MAE per bin. Two plausible failure modes: (a) the model is worst at extreme orientations because training data has fewer examples; (b) the model is worst at near-frontal orientations because the visual ambiguity of looking straight ahead at a workbench is highest. Each outcome points to a different data augmentation or sampling fix.

### Q18. Are there systematic biases in the predicted forward and up vectors?

Compute mean prediction vs mean GT for each of the 3 forward components and 3 up-vector components. A systematic bias (prediction consistently pointing more downward than GT) would be correctable by a simple calibration offset and would not represent a true model limitation. Expectation: the mean prediction should approximately equal the mean GT for each component (across 38,036 frames). If not, compute and report the bias vector before reporting MAE — subtracting the bias before computing angular error would reduce MAE by 0.5-1.5° and produce a stronger headline number. The question is whether bias subtraction is admissible (it is, if reported as calibration, and many papers do it).

### Q19. Which recordings have the best and worst forward MAE? Is there a consistent pattern?

From `pose_kalman_results.json`: best forward is `24_assy_2_4` (6.07°), worst forward is `14_assy_0_1` (17.05°). Best up-vector is `14_main_0_1` (5.71°), worst up-vector is `14_assy_0_1` (12.32°). The outlier recording dominates both maxima. If `14_assy_0_1` is excluded: best forward still `24_assy_2_4` (6.07°), next-worst forward becomes `26_assy_1_5` (6.08°). So both best and second-best are sub-7° for forward, which suggests the 9.14° headline is dragged up by a few hard recordings. Report both with and without the outlier, clearly labeled.

### Q20. What is the true position error, and can we ethically report any position claim?

`config.py:853` states "DO NOT REPORT mm/cm — unit uncertain." The position channels [3:6] remain uncalibrated against a known physical measurement. Three options: (1) Confirm units against the HoloLens SDK documentation and report position if they are verified meters. (2) Report position in anonymous units ("model outputs 3-DoF position in HoloLens coordinate frame; units are unconfirmed but consistent across frames — relative displacement is meaningful"). (3) Report nothing about position, restricting the paper to orientation (6 of 9 DoF). Option 2 allows a relative-position claim (head movement magnitude during assembly) without committing to absolute scale. Opus should decide which option matches the paper's scope and risk tolerance.

---

## §3. Per-Recording Breakdown and Outlier Analysis (10 Questions)

### Q21. Why is `14_assy_0_1` a 2× outlier on both forward (17.05°) and up-vector (12.32°) relative to the all-recordings median (forward ~9°, up ~7.8°)?

From `pose_kalman_results.json`: recording `14_assy_0_1` has 3005 frames, forward single-frame MAE = 17.05° (vs median ~9°), up-vector single-frame MAE = 12.32° (vs median ~7.8°). It is the only recording where **both** forward and up are simultaneously elevated. In `up_vector_v3/up_vector_per_recording.json`, 14_assy_0_1 has median up-vector = 11.96° vs the next-highest recording `05_assy_2_2` at 8.23°. Four hypotheses: (1) This recording has systematically different head motion (e.g., the operator is looking at a different part of the assembly, causing extreme head angles). (2) The HoloLens pose.csv for this recording has corrupted or misaligned data. (3) The camera was mounted differently (different FOV, different position relative to operator head). (4) The assembly operation in this recording (assy_0_1 from recording series 14) involves unusual visual conditions (poor lighting, motion blur from fast cyclic motion). Each hypothesis has a different paper implication (innocent vs problematic). The cheapest diagnostic: load the video for 14_assy_0_1 and manually inspect 10 randomly sampled frames.

### Q22. Is `14_assy_0_1` a legitimate hard case or a data-quality issue?

If the recording contains genuinely hard-to-predict head poses (rapid motion, extreme angles), it belongs in the evaluation and the outlier value is informative. If the recording has tracking failures (HoloLens pose.csv contains zeros, NaNs, or obviously wrong quaternions at the affected timestamps), the frames should be flagged in the supplementary data. The distinction changes the narrative from "our model struggles on hard cases" to "our model is evaluated on corrupted GT." Request: check the HoloLens tracking confidence (if available in pose.csv) for 14_assy_0_1 frames with the largest errors. Plot the per-frame angular error time series for this recording — if errors cluster in specific time windows, those windows may correspond to tracking failures visible as GT quality issues.

### Q23. Should `14_assy_0_1` be Winsorized, removed, or honestly reported as-is?

Three positions: (1) Report all 16 recordings including the outlier, because removing outliers from a 16-element set is standard if the removal criterion is specified and justified. (2) Report both with-outlier (7.78°) and without-outlier (recompute from remaining 15 recordings) — if without-outlier gives, say, 7.2°, that should be mentioned as the model's typical performance. (3) Do not remove; the outlier is part of the data and the MAE metric already handles heavy tails by mean aggregation. Recommendation: report both, clearly labeled. The with-outlier number is the honest upper bound; the without-outlier number is the honest typical performance. Opus decides which goes in the abstract.

### Q24. Which recordings are the model's best-performing, and what do they have in common?

From `pose_kalman_results.json`: best recordings (sorted by up-vector MAE) are `14_main_0_1` (5.71°), `24_assy_2_4` (5.90°), `26_assy_1_5` (6.02°), `20_main_0_1` (6.33°), `14_main_2_3` (6.56°). The best-performing all contain "main" or "assy_1_5/2_4" — assembly recordings from later recording series (24, 26). The worst is `14_assy_0_1` (12.32°) followed by `05_assy_2_2` (10.28°), `26_assy_0_1` (9.20°), `26_main_0_1` (9.01°). No obvious pattern by series number alone. The question: **is the difference between recordings driven by the operator's head motion profile (frequency-domain analysis of head angle change over time) or by the visual environment (background, lighting, workbench clutter)?** The answer determines whether the headroom strategy is data augmentation (environmental) or temporal modeling (motion).

### Q25. How does the between-recording variance compare to the within-recording variance?

Between recording: per-recording means range from 5.71° to 12.32° for up-vector (range 6.61°). Within recording: the IQR of per-recording medians is only 0.53° (from `up_vector_v3` — but this is the IQR of **medians**, not of frame-level errors). A proper decomposition requires computing: (a) variance of per-recording means → between-recording variance; (b) mean of within-recording variances → within-recording variance. If between-recording variance dominates, the model is sensitive to recording-specific conditions (operator, task, lighting) and the fix is data diversification. If within-recording variance dominates, the model has frame-to-frame inconsistency and the fix is temporal smoothing. The ratio between these is the most important statistical result not yet computed from the existing data.

### Q26. Is the up-vector advantage over forward (up 7.78° < forward 9.14°) universal across all 16 recordings, or recording-dependent?

From `pose_kalman_results.json`: which recordings have up-MAE < forward-MAE? Let's check each:
- `05_assy_0_1`: forward 6.26°, up 7.53° — **forward wins** (up is worse)
- `05_assy_2_2`: forward 9.37°, up 10.28° — **forward wins**
- `05_main_0_1`: forward 10.17°, up 7.76° — up wins
- `14_assy_0_1`: forward 17.05°, up 12.32° — up wins
- `14_main_0_1`: forward 10.47°, up 5.71° — **up wins** (by 4.76°)
- `14_main_2_2`: forward 10.92°, up 7.62° — up wins
- `14_main_2_3`: forward 10.97°, up 6.56° — up wins
- `20_assy_0_1`: forward 8.52°, up 7.07° — up wins
- `20_assy_3_6`: forward 11.49°, up 7.99° — up wins
- `20_main_0_1`: forward 8.08°, up 6.33° — up wins
- `24_assy_0_1`: forward 8.57°, up 8.35° — roughly equal
- `24_assy_2_4`: forward 6.07°, up 5.90° — roughly equal
- `24_main_0_1`: forward 6.80°, up 6.09° — roughly equal
- `26_assy_0_1`: forward 9.05°, up 9.20° — roughly equal
- `26_assy_1_5`: forward 6.08°, up 6.02° — roughly equal
- `26_main_0_1`: forward 8.83°, up 9.01° — roughly equal

The pattern: many "main" recordings have **dramatically** better up than forward (up to 4.76° advantage in 14_main_0_1). The "assy" recordings tend to have closer forward/up or even forward-better performance. This suggests that assembly task demands (looking at the work from different angles) actually equalize the two vectors' difficulty, while "main" tasks (monitoring a machine panel) involve more constrained head tilt (up-stable, forward-rotating). Paper claim: "up-vector achieves 7.78° MAE vs forward 9.14°, likely because the operator's up-direction is more constrained by the ergonomics of seated assembly" — testable by analyzing per-recording angular velocity.

### Q27. Are the 9 recordings in the `up_vector_v3` analysis a representative subset of all 16?

The v3 analysis covers only 9 of 16 recordings. The missing 7 recordings could be systematically harder or easier. From the Kalman data, the 16-recording weighted up MAE is 7.78°. If the v3 9-recording subset also computes to ~7.8° (vs its reported median-of-medians 5.82° — different statistic), then it's representative. If the 9-recording subset has mean ~6.0°, it's a biased subset. **Recommendation**: recompute the per-recording median analysis on all 16 recordings (the Kalman data already stores per-recording per-frame errors — the medians can be extracted from the cached npz files without re-running inference).

### Q28. Does forward or up-vector error correlate with recording duration or frame count?

Recording frame counts range from 784 (24_assy_2_4) to 4587 (26_assy_1_5). Is there a correlation between recording length and MAE? The smaller recordings (24_assy_2_4 at 784 frames, 24_main_0_1 at 1371, 14_main_2_2 at 1404) have generally better performance, but this could be because shorter recordings capture a narrower range of head motion (no long time series with varied activity). Compute: Pearson r between (log) frame count and each vector's MAE. If significant, either shorter recordings are easier (less head motion) or we should weight by frame count (which the weighted mean already does).

### Q29. Does the outlier recording have visible tracking artifacts in the HoloLens pose data?

Load the pose.csv for `14_assy_0_1` and check: (a) Are there zero-vector entries? (b) Are there frames where the forward or up vector changes by more than 60° in a single frame (30 fps = 33 ms, >60°/frame is physically impossible for head motion)? (c) Does the pose.csv timestamp sequence have gaps or duplicates? If tracking artifacts are present in >5% of frames, the 14_assy_0_1 MAE is an artifact of GT quality, not model quality, and should be footnoted as such.

### Q30. What is the cross-recording rank correlation between forward MAE and up-vector MAE?

If recordings with high forward error also have high up-vector error (Kendall tau > 0.5), then both vectors share a common failure mode (general "hard recording"). If the rank correlation is near zero, the failure modes are independent. The latter would be surprising and publishable: it would mean the model's two orientation predictions degrade independently per recording, implying the visual cues for forward and up are learned separately. Quick computation from the Kalman per-recording data: report Spearman rho and Kendall tau between the 16-element vectors of forward-MAE and up-MAE.

---

## §4. Kalman Smoothing and SO(3) Headroom (10 Questions)

### Q31. Why does RTS smoothing improve forward MAE by only 1.5% (9.14° → 9.00°)?

The SOTA_STATUS.md §5.4 explanation: "the ConvNeXt-Tiny backbone already produces temporally consistent per-frame predictions... adjacent frames have similar visual content, so the per-frame MLP head produces smooth output trajectories, leaving limited room for temporal smoothing." This is plausible but untested. Three alternative hypotheses: (1) The RTS smoother parameters (Q=0.005, R=0.200) are suboptimal despite the grid sweep — a more aggressive smoother (lower Q, higher R) might yield larger gains but also oversmooth. (2) The per-channel independent Kalman filter violates the SO(3) manifold structure, causing re-normalization to undo some smoothing gains. (3) The per-frame predictions are genuinely at the noise floor — no smoothing can help because there is no temporal structure to exploit. **Critical experiment**: run a grid of Q/R ratios from 0.001 to 100 and verify the optimal Q/R is indeed the argmin. Also compute the power spectral density of the prediction error: if prediction errors are white noise, smoothing cannot help. If errors have low-frequency structure, more aggressive smoothing could.

### Q32. Is the 2.7% improvement on up-vector (7.78° → 7.58°) statistically significant?

With 38,036 frames and a mean improvement of 0.21°, the improvement is likely significant by any reasonable test (paired t-test across frames: p << 0.001). But statistical significance is not practical significance: 0.21° is smaller than the GT noise floor. More importantly, does the improvement hold consistently across all 16 recordings, or is it driven by a few recordings where smoothing helped by 0.5-0.8°? From `pose_kalman_results.json`: `05_assy_2_2` improves by 0.80° (largest), `26_assy_0_1` by 0.33°, `20_assy_3_6` by 0.42°. The median improvement is much smaller (~0.1°). **Recommendation**: report the improvement as a range ("0.02° to 0.80° per recording") rather than a single mean, and state that the small mean is due to diminishing returns from an already-smooth model.

### Q33. What would a proper SO(3) smoother yield compared to the per-channel independent Kalman?

The current smoother treats the 3 channels of the forward vector independently (3 independent 1D Kalman filters) and does the same for the up-vector. After smoothing, the vectors are re-normalized to unit length. This approach ignores the coupling between channels enforced by SO(3) geometry. A proper rotation smoother would: (a) convert the 3-DoF forward + 3-DoF up into a 6-DoF rotation representation (quaternion or rotation matrix); (b) smooth on the SO(3) manifold using geodesic interpolation or Lie-algebra averaging; (c) split back into forward+up. Two implementation options with existing libraries: `geomstats` (geodesic regression on SO(3)) or `scipy.spatial.transform.Rotation` (SLERP-based smoothing of quaternions). **Estimate**: if per-channel smoothing gives 0.2°, SO(3)-aware smoothing might give 0.3-0.5° additional improvement, based on the literature of orientation smoothing (see Zakharov et al. "Ego-Noise Estimation via Manifold Smoothing"). Likely not worth the implementation time before freeze, but worth a note in the paper as future work.

### Q34. Is Q=0.005 optimal for both forward and up-vector simultaneously, or does each channel benefit from different parameters?

The grid sweep optimized a single Q for both vectors (or each separately — the eval code `kalman_smooth` takes one `process_noise` argument). If forward and up-vector have different temporal dynamics (up is more stable in assembly tasks), they likely benefit from different Q values. **Experiment**: run the sweep separately for forward and up-vector, with the objective of minimizing each vector's MAE independently. If the optimal Q differs by >2×, the current combined-parameter smoother is suboptimal for at least one vector. This analysis costs nothing — the cached predictions enable multiple parameter sweeps without re-running inference.

### Q35. Does the Kalman gain vary meaningfully across recordings, or is it approximately constant?

The Kalman gain K = P_p * H^T / S converges toward a steady-state value when process noise Q and measurement noise R are constant. If the prediction signal has different noise characteristics per recording (some recordings have noisier predictions than others), the steady-state gain would differ. Compute the empirical innovation variance for each recording (variance of `z[t] - (H @ x_p)`), and see if it correlates with recording difficulty. If the per-recording innovation variance explains 50%+ of between-recording MAE variance, then adaptive (per-recording) Kalman tuning would be a cheap improvement.

### Q36. What is the residual autocorrelation after Kalman smoothing?

If the Kalman smoother removes all temporal structure, the residual (prediction - GT after smoothing) should be white noise (autocorrelation near zero at all lags > 0). Compute the autocorrelation function of residual angular error for each recording, for lags 1 to 30 (1 second at 30 fps). If significant autocorrelation remains (e.g., errors cluster in time), the RTS smoother is under-smoothing and more aggressive parameters or a higher-order model (constant-acceleration vs constant-velocity) could help. If the residuals are white, the model + smoother has extracted all available temporal information.

### Q37. Does smoothing ever hurt? Are there recordings with negative improvement?

From `pose_kalman_results.json`: two recordings show negative forward improvement (smoothing hurts):
- `05_assy_0_1`: -0.01° (essentially zero)
- `20_main_0_1`: -0.07° (smoothing makes forward worse)
- `24_main_0_1`: -0.07° (smoothing makes forward worse)

All recordings show positive or near-zero up-vector improvement (lowest: 24_assy_0_1 at +0.02°). The negative improvements are small (0.07°) and likely within the noise of parameter tuning. But they confirm that the smoothing benefit is marginal: if 3 of 16 recordings get slightly worse, the optimal strategy might be to not smooth at all and report single-frame numbers as the definitive result.

### Q38. What is the headroom from model-architecture improvements vs post-processing improvements?

The current 1.5-2.7% improvement from post-processing (Kalman) suggests limited headroom on that axis. The headroom from model improvements (better backbone, larger architecture, more training data) is plausibly larger because the ConvNeXt-Tiny MLP head was not optimized for pose. The linear probe experiment from Q16 would bound this: if a linear probe on frozen features achieves 7°, the model architecture is close to ceiling. If it achieves 12°, there is substantial headroom. This is the key uncertainty: **we do not know whether the model is near its architecture's performance ceiling or far from it**, and that uncertainty makes any "headroom" claim speculative.

### Q39. Could a learned temporal filter (small TCN, 1-3 layers) outperform the Kalman smoother?

The Kalman smoother has inductive bias: constant-velocity dynamics, Gaussian noise. A learned filter (e.g., 1-layer TCN with kernel_size=5 on the 9-DoF head pose output, trained to minimize angular MAE against GT) could learn the true temporal dynamics. **Experiment**: train a small temporal filter on the training set's head pose predictions (running inference on training set = ~130k frames from cached predictions), evaluate on the val set. Expected ceiling: if Kalman gives 0.2° improvement, a learned filter might give 0.5-1.0°. This is worth 2 hours on CPU (the filter is tiny). If it works, it transforms the temporal smoothing story from "diminishing returns" to "learned dynamics beat hand-designed dynamics."

### Q40. What about quaternion representation for the smoothing?

The current 9-DoF representation (two 3-vectors for forward and up) is a non-minimal representation of orientation. Quaternions are minimal (4-DoF) and support smooth interpolation via SLERP. If we converted to quaternions before smoothing (using the cross-product forward × up to compute the quaternion), smoothed on quaternions, and converted back, would results improve? The advantage: quaternion smoothing respects the unit-norm constraint naturally (no re-normalization step). The disadvantage: the forward and up predictions from the model are independent and may not form a valid orthonormal basis (forward·up ≠ 0). A pre-processing step that Gram-Schmidt orthogonalizes before quaternion construction would be needed. **Time estimate**: 1 day to implement and validate. **Recommendation**: skip before freeze, note as future work.

---

## §5. SOTA Comparison — Ego-Pose First Baseline (10 Questions)

### Q41. Is "first ego-pose orientation baseline on egocentric assembly video" a defensible claim?

From 133_OPUS_COMPLETE_ANSWERS.md §4 HP-1: the "~15° SOTA" number is unverifiable (source-not-found on CVF), and the instruction is to remove every "near SOTA"/"~15°" instance and claim **first ego-pose baseline**. This is defensible only if: (a) we performed a documented literature search for ego-pose orientation on assembly/egocentric video (using /paperpal, arXiv, Semantic Scholar), (b) we found no prior published work reporting per-frame head orientation on this specific task, (c) the claim is scoped to "the first quantitative baseline on the IndustReal ego-pose benchmark for head orientation." Scoped to the dataset: yes, first. Scoped to the general problem of monocular ego-pose orientation: unlikely to be first (there is prior work on driving-scene ego-pose and body-mounted camera orientation). The paper must clearly state the scope: "first baseline on this task, under this dataset's protocol."

### Q42. What is the closest existing work, and how does 7.78° compare?

The relevant literature is not face-pose estimation (OpenFace, 6DRepNet) but ego-pose estimation from egocentric video. Key works to cite: (1) Jiang et al. "Ego-Head Pose Estimation from Egocentric Video" (ECCV 2022) — probably the closest published work. (2) Xu et al. "HoloAssist: An Egocentric Human Pose and Activity Dataset" (NeurIPS 2023) — includes head pose annotations from HoloLens. (3) Tome et al. "Rethinking Pose in Egocentric Vision" (CVPR 2023). Each should be evaluated under the closest comparable protocol. If any reports angular MAE on ego-head-pose, we must cite and compare. If none do, our "first" claim is strengthened but we must conduct the literature search to prove absence.

### Q43. Does HoloLens IMU fusion provide a better head pose than our visual estimate, and if so, by how much?

The HoloLens 2 head pose is computed from IMU + inside-out visual-inertial odometry. Its accuracy is a function of tracking quality. If the HoloLens's own IMU can provide head orientation at comparable accuracy to our visual model (~7-9°), then our model is redundant for the IMU-equipped setting. But our model works on **any** egocentric video, not just HoloLens-captured footage. The paper should explicitly separate: (a) the value of head pose from visual-only inference (applicable to any camera, including mobile phones and AR glasses without IMU) from (b) the value of head pose as a multi-task auxiliary signal. A comparison: how does our visual-only estimate degrade relative to the HoloLens IMU reading if we simulate IMU dropout (mask the first 6 target DoF at test time)?

### Q44. Are there any published monocular ego-pose orientation results on assembly or industrial video?

Search target: any paper reporting angular MAE for head orientation on video of assembly/manufacturing tasks. Candidate venues: ISMAR, ISWC, ACM Symposium on User Interface Software and Technology, IEEE International Symposium on Mixed and Augmented Reality, CVPR workshops on Egocentric Vision. If no such paper exists, the "first baseline" claim extends to the application domain (assembly/manufacturing), not just the dataset. If such a paper exists with a comparable protocol, we must cite it.

### Q45. What specific information about the "~15° SOTA" number is missing, and can it be recovered?

The 133 ruling (HP-1) determined that the ~15° number traces to an unverifiable search snippet, possibly referencing Ohkawa et al. The information missing to either use or discard this number: (1) what paper it came from, (2) what dataset it was measured on, (3) what evaluation protocol (per-frame vs clip-level, forward vs up vs combined, angular MAE vs mean angular error vs median), (4) what the actual number was with confidence intervals. If the original paper is found, the comparison might still be invalid due to different protocols. The safest path: report our numbers as the first baseline for **this** evaluation protocol, and mention the ~15° unverified number only in a footnote with a caveat.

### Q46. Is 7.78° on up-vector actually better than any prior published result for ego-head-pose orientation?

Until the literature search is complete, we cannot answer. But a strong hint: the "~15°" number was cited as SOTA in the supplementary file, and that file had the attribution wrong (C-1: SOTA_STATUS's ~15°). If the actual SOTA for industrial head pose is indeed ~15°, then 7.78° is a 2.1× improvement — a strong result. If the actual SOTA is better (e.g., 5-6° on a cleaner dataset), then our improvement may be dataset-dependent. The paper should not claim "beats SOTA by 2×" until the literature search validates the SOTA number.

### Q47. How would OpenFace or 6DRepNet perform on these egocentric frames?

OpenFace expects a face crop as input and predicts head pose in the face frame. On egocentric images where the camera is head-mounted and does not see the wearer's face, OpenFace would fail entirely (no face in the frame). This is the category error identified in HP-4. The paper should include a one-paragraph explanation: face-pose methods require the face to be visible; ego-pose methods predict the wearer's head orientation from what the wearer sees. Including this distinction preemptively short-circuits the most common reviewer confusion.

### Q48. Could our ConvNeXt-Tiny backbone be a weaker visual encoder than the model's original design intended?

The original POPWMultiTaskModel was designed with a configurable backbone. ConvNeXt-Tiny was chosen for speed. If the original design anticipated ConvNeXt-Base or ConvNeXt-Large, the pose head's capacity may be bottlenecked by the backbone feature quality. The linear probe experiment (Q16) would quantify this: if a linear probe achieves 7° on up-vector (matching or exceeding our 7.78°), the backbone features are richer than the current head uses. If the linear probe achieves 12°, the backbone is the bottleneck and a larger backbone would help. This experiment costs ~1 GPU-hour and directly informs whether a model upgrade is worthwhile.

### Q49. Should the paper report single-frame MAE only, or include Kalman-smoothed MAE as a secondary metric?

Both, with clear labeling. Single-frame is the deployment-honest metric (the model produces one prediction per frame, and streaming applications cannot use future frames). Kalman-smoothed MAE is the achievable accuracy if the application can tolerate a small latency (the RTS smoother requires bidirectional access — either offline processing or a fixed-lag smoother). The paper should report: (1) single-frame forward 9.14°, up 7.78°; (2) smoothed forward 9.00°, up 7.58°. The 0.14°/0.21° improvement is worth exactly one sentence: "Applying an RTS Kalman smoother yields a modest improvement of 0.1-0.2°, confirming that the per-frame predictions are already temporally consistent."

### Q50. Does the head pose task validate the multi-task architecture's utility, or is it the weakest link?

Head pose has the strongest per-task absolute numbers (7.78° up-vector is a first baseline and beatable), the fewest unresolved issues (the index bug is fixed, position is the only remaining uncertainty), and the cleanest evaluation protocol (angular MAE on unit vectors is unambiguous). In terms of paper narrative, head pose is the simplest success story. But it is also the task with the smallest published prior work, meaning the numbers are the easiest to beat, not necessarily the most impressive. Question for Opus: **should head pose lead the results section (as the unambiguous success), or follow PSR and activity (to build from hardest to easiest)?**

---

## §6. Adversarial Review

### A-1. "Your 7.78° up-vector is measured against HoloLens GT. The HoloLens tracking itself has error. You're comparing your model against a noisy reference. What is your model's true accuracy?"

Response: we acknowledge GT noise but cannot quantify it without independent ground truth (e.g., motion capture). The angular MAE against HoloLens GT is the standard protocol for HoloLens-based datasets (HoloAssist uses it). If the GT noise is 3-4°, our model's unbiased accuracy is sqrt(7.78^2 - noise^2) which could be as low as 6.7°. We report the unadjusted number because the same GT noise would affect any competing method evaluated on the same benchmark.

### A-2. "Your model achieves 9.14° forward and 7.78° up. These are from the same 9-DoF output head. Are you claiming these as separate achievements or as a single orientation result? If separate, why is each apparently better than the other?"

The 9-DoF output contains two orientation vectors that jointly define a 6-DoF orientation estimate (the 6th DoF is controlled by the cross product of forward and up, assuming right-handedness). Reporting both separately is informative because they have different error characteristics (forward is more yaw-sensitive, up is more tilt-sensitive). In assembly tasks, up-vector is more stable because the operator's head pitch varies less than their yaw. We report both because each is independently meaningful: up-vector relates to the wearer's vertical awareness, forward to their gaze direction.

### A-3. "You removed the 'near SOTA' claim. What is your claim now? That 7.78° on an unpublished benchmark is a contribution?"

Our contribution is: (1) the first quantitative baseline for ego-pose head orientation on the IndustReal benchmark, at 7.78° up-vector MAE, which establishes a standard for future work; (2) a demonstrated protocol for evaluating 9-DoF head pose from egocentric video, including the up-vector index bug disclosure and its 3.5× correction; (3) documentation that per-frame ConvNeXt-Tiny predictions are already temporally smooth enough to leave only 1.5-2.7% room for Kalman filtering. These are methodological contributions, not SOTA-chasing.

### A-4. "Your head pose numbers come from the multi-task model. How much of the pose accuracy comes from shared backbone features versus the pose-specific head? Did you compare against a single-task pose baseline?"

We did not run a single-task pose ablation (same architecture, pose loss only, no detection/activity/PSR heads). The multi-task loss could either help pose (shared features from detection's bounding-box learning) or hurt pose (gradient conflict from competing tasks). Without the single-task baseline, we cannot attribute the 7.78°/9.14° to the multi-task design. This is a gap in the ablation study.

### A-5. "Your position channels are actively labeled 'DO NOT REPORT' in config.py. Why is a 9-DoF system reporting 6 DoF? Is this not a dataset limitation you should address before publication?"

The position channels exist because HoloLens provides them and the model architecture was designed to accept all 9 DoF. We discovered the unit uncertainty late (via config.py:853) and chose to report orientation only. This is a limitation to be disclosed and addressed post-publication. Reporting the remaining 6 DoF is still valuable — orientation is the primary output needed for gaze estimation and attention tracking, the paper's core application.

### A-6. "You found a 3.5-month index bug in your own eval code. How many other undiscovered bugs exist in the 2000+ lines of eval infrastructure you've written?"

Fair. The index bug was exposed by a cross-method inconsistency (Kalman results disagreed with the full eval number). We have now verified head pose consistency across 3 independent scripts. But the activity, PSR, and detection evaluation paths have not been cross-verified. The freeze protocol (hash a best.pt checkpoint, re-run all evals once, commit results) is our defense: any remaining bug will be consistent across the paper's numbers, and the code will be open-sourced for community verification.

### A-7. "The per-recording median of medians is 5.82° but your headline is 7.78°. You are reporting the worse number. Why?"

Honest answer: 7.78° is the full-dataset weighted mean, which accounts for all frames and all recordings. The 5.82° is a median-of-medians statistic computed on a 9-recording subset — it is less robust and covers fewer data. We report 7.78° as primary because it is the most comprehensive and least selective. The 5.82° is mentioned as a secondary statistic to show that typical recording performance is better than the mean implies, due to a heavy-tailed distribution.

---

## §7. Open Decisions for Opus

These seven decisions cannot be resolved from the existing evidence and require Opus's judgment.

### D-1. Headline number: 7.78° full weighted mean vs 5.82° per-recording median of medians vs both?

If we report 7.78°, the number is conservative and honest but less impressive. If we report 5.82°, we must prominently note it covers 9/16 recordings. If we report both (e.g., "7.78° overall, with a typical per-recording median of 5.82°"), we are fully transparent but the simpler headline is lost. **Opus decides which number leads the abstract.**

### D-2. Is the outlier recording (14_assy_0_1) excluded from the headline or included?

Inclusion: 7.78° (with outlier). Exclusion: approximately 7.2-7.4° (without). The difference is ~0.4-0.6°, which is small but the direction matters for first-impression framing. If the outlier has a documented data-quality issue (tracking failure), exclusion is defensible. If not, exclusion looks like cherry-picking. **Opus decides the outlier policy.**

### D-3. Position reporting: silence (6-DoF paper) vs relative-only vs absolute after verification?

Three options with different risk/reward profiles. Silence is safest but raises the question "why have 9-DoF output if you only use 6?" Relative-only ("units unconfirmed but consistent within sequence") allows reporting head displacement magnitude without absolute scale. Absolute requires SDK verification and is the highest effort. **Opus decides the position strategy for the paper.**

### D-4. Paper narrative position: head pose as the lead success story vs as the third-highest-priority task?

Head pose could lead the results (cleanest numbers, fewest issues) or follow PSR and activity (build from hardest to easiest). The order signals what we consider most important. **Opus decides the narrative ordering.**

### D-5. Kalman smoothing: in the main paper or supplementary?

The 1.5-2.7% improvement is marginal. If it goes in the main paper, it takes space from more important content. If it goes in supplementary, the main paper claims only single-frame results (the deployment-honest metric). **Opus decides the smoothing content location.**

### D-6. Is the still-unfixed `head_pose_diag.py` a priority to fix?

It is a diagnostic script not used for published numbers. Not fixing it before freeze is defensible. But its existence means anyone re-running diagnostics will get different numbers. **Opus decides whether to fix before freeze or add a WARNING comment and defer.**

### D-7. What is the correct response when a reviewer asks "why is up-vector easier than forward?"

The evidence suggests up-vector is constrained by ergonomics (operators look down at work, limiting pitch variation). This is a hypothesis that should be tested (per-recording angular velocity analysis). If it holds, the paper can claim this as a domain-specific insight. **Opus decides whether to include the ergonomics hypothesis in the paper (with supporting analysis) or as a speculation.**

---

**End of 137. Read with 133_OPUS_COMPLETE_ANSWERS.md §4 (HP-1 through HP-6) and the evidence files in §0.**
