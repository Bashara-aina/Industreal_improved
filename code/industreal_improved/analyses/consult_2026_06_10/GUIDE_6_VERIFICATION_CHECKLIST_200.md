# GUIDE 6 — THE 200-POINT VERIFICATION CHECKLIST

*Re-evaluated with maximum rigor to guarantee that, by following GUIDES 1–5, you can
**prove** a unified multi-head model with benchmarkable results. Every item is a checkable
box. Citations `[n]` point to the References at the end (all real, venue-verified).*

> **Verdict of the re-evaluation:** the plan proves the idea **if** you (a) report honest,
> baseline-matched metrics on the full test set, (b) run the 3-arm experimental matrix
> (single-task / frozen-MTL / joint-MTL) so the multi-task claim survives even *negative
> transfer* [9,10], and (c) finish on a joint fine-tune so the final model is a genuine
> unified MTL model [25,26]. This checklist enforces all three.

**The claim you are proving:** *A single shared-backbone model performs egocentric
assembly understanding — assembly-state detection, body pose, head pose, activity
recognition, and procedure-step recognition — in one forward pass, competitively and far
more efficiently than separate specialists, with cross-task FiLM conditioning.* [1,6,7,8]

Legend: 🔴 = blocks the claim if failed · 🟡 = weakens · 🟢 = polish.

---

## A. Thesis & claim integrity (1–12)
- [ ] 1. 🔴 Write the one-sentence thesis (above) verbatim in the paper intro and verify every experiment maps to C1–C5.
- [ ] 2. 🔴 Confirm the contribution is the *architecture + training method + efficiency*, NOT beating any single specialist [8,14].
- [ ] 3. 🔴 State explicitly that "unified" = one shared backbone + one forward pass at inference; verify this in code (single `forward()` returns all heads).
- [ ] 4. 🔴 Decompose the claim into C1–C5 and assign each a table/figure (traceability matrix).
- [ ] 5. 🟡 Define "benchmarkable" precisely: same dataset (IndustReal [1]), same split, same metric, same protocol as each cited baseline.
- [ ] 6. 🔴 Pre-register the success criteria (GUIDE_3 targets) *before* the final runs so you can't move goalposts.
- [ ] 7. 🟡 Decide the fallback claim in advance: if synergy is negative, the claim is "accuracy–efficiency Pareto" [9]. Write both versions of the abstract.
- [ ] 8. 🔴 Confirm none of C1–C5 depends on detection reaching SOTA (it won't; real-data-only).
- [ ] 9. 🟢 Identify the two uncontested wins (head-pose: no baseline; efficiency: by construction) and make them prominent.
- [ ] 10. 🟡 List the exact baselines: YOLOv8m (ASD) [32], MViTv2 (AR) [15], B2 + STORM-PSR (PSR) [1,2]; record their reported numbers and *their* protocols.
- [ ] 11. 🔴 Verify the IndustReal task definitions you target match the paper: AR (action recognition), ASD (assembly-state detection), PSR (procedure-step recognition) [1].
- [ ] 12. 🟢 Confirm novelty vs prior unified video models / Taskonomy-style transfer [14]; articulate what is new (this task combination + two-stage FiLM).

## B. Dataset integrity, splits, leakage (13–28)
- [ ] 13. 🔴 Use the **official** IndustReal train/val/test splits [1]; never the 50% dev subset for final numbers.
- [ ] 14. 🔴 Verify splits are **recording-level** (no frames from one recording in both train and test) — frame-level splits leak and inflate every metric.
- [ ] 15. 🔴 Confirm IndustReal's design: procedural/execution errors appear *mainly in val/test* [1]; ensure your model never trains on test-only error patterns.
- [ ] 16. 🔴 Verify class counts against the dataset, not your docs: NUM_DET_CLASSES, NUM_ACT raw IDs, NUM_PSR components/steps (config says 24 / 74 / 11 / 36).
- [ ] 17. 🟡 Audit the ASD label space: confirm the 24 classes are 11-bit assembly-state codes and the "background" channel handling (channel 0 = category 1).
- [ ] 18. 🔴 Verify COCO category_id→model-channel mapping (`idx = cat-1`) once, with a printed table; this single off-by-one has bitten you before.
- [ ] 19. 🟡 Compute per-class GT counts (train/val/test) from the data; record which classes are <100 instances (expected AP≈0) for the limitations table.
- [ ] 20. 🟢 Plot the activity-class long-tail and PSR-component prevalence; justify class-balanced choices [20,21].
- [ ] 21. 🔴 Confirm no augmentation leaks across train/val (val/test must be augment-free) — you already found RF1 train/val aug mismatch; verify it's gone.
- [ ] 22. 🟡 Verify frame-rate / stride consistency between training clips and the baseline's clip protocol [15].
- [ ] 23. 🟢 Checksum the dataset (frame counts per recording) and log it, so a re-clone is verifiably identical.
- [ ] 24. 🔴 Confirm GT boxes are correctly rescaled to IMG_SIZE (the `_sx/_sy` path) and visualize 20 random GT overlays to catch coordinate bugs.
- [ ] 25. 🟡 Verify head-pose/body-pose label units (degrees, mm) and the position normalization (`HEAD_POSE_POS_SCALE`).
- [ ] 26. 🟢 Confirm activity segment boundaries used for clip-level eval match the dataset's action annotations.
- [ ] 27. 🔴 Hold out the test set untouched until final eval; do all tuning on val (no test-set peeking — a publication-integrity requirement).
- [ ] 28. 🟢 Document dataset version/commit of the IndustReal release used [1].

## C. Reproducibility & experimental hygiene (29–42)
- [ ] 29. 🔴 Fix and log seeds (you use SEED=42); run ≥3 seeds {42,123,7} for headline tasks [38].
- [ ] 30. 🟡 Set `CUDNN_DETERMINISTIC=True`, `CUDNN_BENCHMARK=False` (already set) and record any remaining nondeterminism.
- [ ] 31. 🔴 Log the exact git commit + config hash into every run's output dir.
- [ ] 32. 🔴 Snapshot the resolved config (after preset application) per run — your presets mutate globals; capture the *effective* values.
- [ ] 33. 🟡 Record library versions (torch, torchvision, CUDA) in each run log.
- [ ] 34. 🟢 Save the full command line + env vars (DET_GT_FRAME_FRACTION etc.) per run.
- [ ] 35. 🔴 Verify the runtime config equals the committed config (your Run1/Run2 disaster: 4.0/2.0 vs 1.0/1.0). Add a startup assertion that logs every LR/loss hyperparam.
- [ ] 36. 🟡 Confirm `rf_stage_state.json` is actually being written (doc 45 says it wasn't) if you use the orchestrator.
- [ ] 37. 🟢 Keep one experiment = one output dir = one log; no two runs sharing a log file ever again.
- [ ] 38. 🔴 Verify checkpoint save/load round-trips (weights, optimizer, Kendall log_vars, EMA shadow) with a unit test [tests/test_checkpointing.py exists — run it].
- [ ] 39. 🟡 Confirm EMA [27] is applied at eval and that best.pth stores the intended (EMA vs raw) weights.
- [ ] 40. 🟢 Write a `make reproduce` / shell script that regenerates every paper number from checkpoints.
- [ ] 41. 🔴 Verify `det_mAP50_pc` now drives best.pth + gates (the fix I applied); grep for "HONEST METRIC" in train.py.
- [ ] 42. 🟢 Archive checkpoints + cache + logs off the ephemeral box (you lost state before).

## D. Phase A — backbone + detection training (43–60)
- [ ] 43. 🔴 Train Phase A with `recovery_det_only` (det + head-pose; act/PSR off) per GUIDE_2.
- [ ] 44. 🔴 Verify `detach_reg_fpn=False` so GIoU regression gradient shapes the FPN [4,22] (you proved detach starves the trunk).
- [ ] 45. 🟡 Keep RetinaNet prior init (`reinit_pi`) so cls bias starts at the focal prior [3].
- [ ] 46. 🔴 Confirm focal config is RetinaNet-sane: it's fine as-is (α=0.90, γ_pos=0, γ_neg=1.5); do NOT run more OHEM/γ sweeps [3,19].
- [ ] 47. 🟢 Confirm GT-frame oversampling (`DET_GT_FRAME_FRACTION≈0.9`) is active for the detection-dominant phase.
- [ ] 48. 🔴 Watch `POS_ANCHOR_PROBE` = 400–800 pos/img (anchor coverage is settled — do not touch anchors).
- [ ] 49. 🔴 Track `det_mAP50_pc` (not diluted `det_mAP50`) as the Phase-A objective.
- [ ] 50. 🟡 Confirm the loss is finite throughout (NaN guards) and grad-clip (5.0) is active.
- [ ] 51. 🟢 Plot per-class AP over epochs: frequent states rise first, rare ones lag (expected, not a bug).
- [ ] 52. 🔴 **Synthetic pretrain decision:** if synthetic data exists, run `pretrain_synthetic.py` [1] first (the YOLOv8m gap is mostly synthetic data [32]); else record it as a stated limitation.
- [ ] 53. 🟡 If using synthetic pretrain, verify domain-gap handling (don't eval on synthetic; real fine-tune after).
- [ ] 54. 🟢 Add class-balanced sampling for <100-instance states [20] and measure the lift (don't expect miracles on 26-instance classes).
- [ ] 55. 🔴 Exit Phase A on plateau (3 epochs <+0.005 `det_mAP50_pc`); record that value as the detection result — do not grind.
- [ ] 56. 🟡 Save the Phase-A backbone+FPN+det+head-pose `best.pth` as the shared trunk for all later phases.
- [ ] 57. 🟢 Confirm ConvNeXt-T backbone choice is justified (accuracy/efficiency, transfer) [5,30].
- [ ] 58. 🟡 Verify gradient-checkpointing on/off doesn't change results (only memory).
- [ ] 59. 🟢 Confirm AMP is off (you found AMP-incompatible ops) or fixed; record FP32 as the setting.
- [ ] 60. 🔴 Visualize 20 detection predictions vs GT to confirm localization is good (it is — the error is *state class*, not box).

## E. Phase B — decoupled cache + temporal heads (61–75)
- [ ] 61. 🔴 Fix `embedding_cache.py` line ~489 (`batch_idx := 1` bug) before caching.
- [ ] 62. 🔴 Replace `CacheDataset`'s "first 80% recordings" split with the **official** split (else numbers aren't comparable).
- [ ] 63. 🔴 Verify cached output keys match `model.py` forward dict (`activity_proj`/`proj_feat`, `det_conf`, `c5_mod`, `pyramid.p4`) — print `outputs.keys()` once.
- [ ] 64. 🟡 Cache train, val, AND test splits separately; confirm frame counts equal dataset counts.
- [ ] 65. 🔴 Confirm the backbone is frozen during caching (no grad) — this is the interference cure [9,10] and the VRAM unlock.
- [ ] 66. 🟢 Verify cache dtype (float16) doesn't degrade head accuracy vs float32 (spot-check one head).
- [ ] 67. 🟡 Confirm `embedding_cache` trains ONLY activity + PSR; pose/head-pose stay in Phase A (spatial heads). Document this division.
- [ ] 68. 🔴 Sanity: a head trained on cache must match the same head's frozen-backbone accuracy when run live (no cache/live mismatch — your RC-17 class of bug).
- [ ] 69. 🟢 Tune `seq_len` (T=64 default) for activity/PSR temporal context; report the choice.
- [ ] 70. 🟡 Confirm the temporal models (TMA cell, feature bank) are actually engaged on long sequences (they were dormant in per-frame mode).
- [ ] 71. 🟢 Measure Phase-B throughput (epochs/hour) to support the "100× faster head training" efficiency narrative.
- [ ] 72. 🔴 Frame decoupling as a *curriculum / gradual-unfreezing* training method [25,26], not an ad-hoc hack, in the paper.
- [ ] 73. 🟡 Save per-head best weights from cache (`best_activity_head.pth`, `best_psr_head.pth`).
- [ ] 74. 🟢 Confirm reproducibility of cache (same checkpoint → same embeddings).
- [ ] 75. 🔴 Verify the assembled inference model (frozen trunk + cached-trained heads) runs end-to-end in ONE forward pass (C1).

## F. Detection head correctness & honest metric (76–90)
- [ ] 76. 🔴 Report `det_mAP50_pc` (present-class) as primary; always show `det_mAP50` (COCO-24) WITH `n_present/24` [23].
- [ ] 77. 🔴 Run final detection eval on the **full test set** (more classes present → less dilution).
- [ ] 78. 🟡 Report mAP@[0.5:0.95] too; expect it lower (fine-grained boxes).
- [ ] 79. 🔴 Produce the **24×24 confusion matrix**; verify errors concentrate on 1-bit-Hamming-neighbor states (reframes "low mAP" as fine-grained state ID).
- [ ] 80. 🟡 Report localization recall separately (boxes are good even when state class is wrong) — supports the reframe.
- [ ] 81. 🔴 Use the same score threshold / NMS / max-dets as a COCO-style protocol [23]; document them.
- [ ] 82. 🟢 Run `diag_per_class_truth.py` for the authoritative per-class AP/GT table (kills index ambiguity).
- [ ] 83. 🟡 Confirm AP=1.0 artifacts (no-GT classes) are excluded/flagged, not reported as "perfect."
- [ ] 84. 🟢 Verify GIoU box decoding clamps are correct (no degenerate boxes) [22].
- [ ] 85. 🔴 Do NOT re-open OHEM/γ/bias-LR/anchor tuning — settled (GUIDE_1 §1.2) [3,19].
- [ ] 86. 🟡 Compare against YOLOv8m honestly: note its COCO+260k-synthetic+real budget [32] in the same table caption.
- [ ] 87. 🟢 If synthetic pretrain was run, show the with/without-synth delta (a clean ablation).
- [ ] 88. 🟡 Report detection FPS as part of the unified forward pass, not standalone.
- [ ] 89. 🟢 Provide qualitative detection figures (success + the fine-grained failure mode).
- [ ] 90. 🔴 Lock the detection number after Phase A; it is one row, not the paper.

## G. Activity head (91–104)
- [ ] 91. 🔴 Evaluate **clip-level** (16 uniform frames → one prediction), matching MViTv2 [15]; per-frame is a protocol mismatch costing double digits.
- [ ] 92. 🔴 Report clip Top-1 AND Top-5 (baseline reports both) [15].
- [ ] 93. 🟡 Keep CB-Focal + label smoothing (`CB_GAMMA=1.0`) [20]; keep `USE_LDAM_DRW=False` for the first clean run (s=30 collapses) [21].
- [ ] 94. 🟢 Add class-balanced sampling; track `pred_seen` (>20/75 classes = exploring, not collapsed).
- [ ] 95. 🟡 Verify the 75-channel head (raw action_id 0..74) is correct; cold IDs 37/64 are harmless.
- [ ] 96. 🔴 Confirm no per-frame label leakage into clip aggregation (aggregate logits, then argmax once).
- [ ] 97. 🟡 If re-enabling a K400 video stream [16], integrate it as a separate stream on the frozen trunk; measure VRAM and the accuracy delta.
- [ ] 98. 🟢 Report a confusion matrix / top-confused action pairs for interpretability.
- [ ] 99. 🟡 Verify temporal context (feature bank T=16, TMA) actually contributes (ablate T).
- [ ] 100. 🔴 Target clip Top-1 0.35–0.45, Top-5 0.70+; frame as "RGB-only, no video encoder, multi-task, fraction of compute" vs MViTv2 0.6525 [15].
- [ ] 101. 🟢 Confirm mixup/cutmix stay OFF (your impl mixes logits = label corruption) until fixed.
- [ ] 102. 🟡 Check activity does not dominate the backbone in any joint phase [6,12].
- [ ] 103. 🟢 Report per-class activity accuracy for the long tail (honesty).
- [ ] 104. 🔴 Lock the activity number on the full test set, clip-level.

## H. PSR head (105–118)
- [ ] 105. 🔴 Report F1(±3-frame) and POS exactly as IndustReal/STORM-PSR define them [1,2]; use `psr_f1_at_t`, NOT `psr_overall_f1` (≈0).
- [ ] 106. 🔴 Train PSR as a **transition predictor** (Gaussian-smeared transition targets), not per-frame BCE on ~95%-static labels (constant output trap) [1].
- [ ] 107. 🟡 Apply the monotonic decoder + procedure-order prior (your `MonotonicDecoder`, `USE_PSR_ORDER_PRIOR`) — this is B2's strength, learned [1].
- [ ] 108. 🟡 Feed cached detection state (`det_conf`) into PSR — PSR is the natural consumer of ASD (mirrors B2's ASD-accumulation) [1].
- [ ] 109. 🟢 Track unique predicted patterns (>10 = not constant output).
- [ ] 110. 🔴 Target F1 0.50–0.62 → beats STORM-PSR neural F1 0.506 [2]; approaches B2 heuristic 0.731 [1] with a *learned* model.
- [ ] 111. 🟡 Report the delay metric if you claim PSR competitiveness (STORM-PSR optimizes delay) [2].
- [ ] 112. 🟢 Verify sequence-mode training engages the temporal transformer (per-frame leaves it dormant).
- [ ] 113. 🟡 Handle execution-error steps (IndustReal's novelty) — report robustness to unseen errors [1].
- [ ] 114. 🟢 Confirm PSR component weights / prevalence handling are correct.
- [ ] 115. 🔴 Evaluate PSR on full test set with the official tolerance window [1].
- [ ] 116. 🟢 Provide a qualitative PSR timeline figure (predicted vs GT step completions).
- [ ] 117. 🟡 Contrast learned-PSR vs the B2 heuristic explicitly (your claim is "learned superset of B2") [1].
- [ ] 118. 🔴 Lock the PSR number.

## I. Head pose + body pose (119–128)
- [ ] 119. 🟢 Report head-pose forward/up angular MAE + position MAE; you're at ~9° (excellent).
- [ ] 120. 🟡 Justify 6D continuous rotation representation [17] vs raw-9-number MSE (continuity → lower MAE).
- [ ] 121. 🔴 Note: head pose has **no published supervised baseline** [1] → uncontested contribution; make it a clean table row.
- [ ] 122. 🟢 Verify stop-gradient on the head-pose conditioning signal (no feedback loop) [7].
- [ ] 123. 🟡 Report body-pose PCK@0.2 + per-keypoint MAE.
- [ ] 124. 🟡 Confirm Wing loss [18] + soft-argmax with train τ=1.0 / eval τ=0.1 [35] (you fixed the gradient-killing low-τ).
- [ ] 125. 🟢 Verify keypoint flip pairs are correct under horizontal-flip augmentation.
- [ ] 126. 🟢 Provide qualitative pose overlays.
- [ ] 127. 🟡 Confirm pose heads are trained in Phase A (spatial) and don't degrade detection.
- [ ] 128. 🔴 Lock both pose numbers.

## J. Multi-task integration: FiLM, Kendall, joint training (129–146)
- [ ] 129. 🔴 Implement the 3-arm experimental matrix: (i) single-task per head, (ii) frozen-shared-backbone MTL, (iii) jointly-trained MTL. This is what proves C4 [8,9].
- [ ] 130. 🔴 For single-task baselines, use the SAME backbone/data/eval so the only variable is task-sharing [9,14].
- [ ] 131. 🔴 Run Phase C joint fine-tune (low LR 1e-5, short) so the FINAL model is genuinely jointly optimized [25,26]; if it destabilizes, report (ii) as the model and (iii) as an attempted-joint ablation.
- [ ] 132. 🟡 Verify Kendall uncertainty weighting [6] is correctly implemented (the log_var bug you fixed — head_pose was excluded from total loss); add a unit test [tests/test_loss_kendall.py].
- [ ] 133. 🟡 With `KENDALL_FIXED_WEIGHTS`, report the fixed λ's; with learned Kendall, report the learned σ's per task [6].
- [ ] 134. 🔴 FiLM ablation: no-cond / +PoseFiLM / +PoseFiLM+HeadPoseFiLM; report activity (and any) deltas [7].
- [ ] 135. 🟡 Verify FiLM γ∈(0,2) via 1+tanh (no feature inversion) and stop-grad on conditioning [7].
- [ ] 136. 🟢 If joint training shows interference, try/cite a gradient-conflict remedy (PCGrad [10], CAGrad [28], GradNorm [12], or MGDA [11]) — even citing it as future work strengthens the paper.
- [ ] 137. 🔴 Compute the MTL relative-performance Δ per task vs single-task (the standard MTL metric) [8].
- [ ] 138. 🟡 Report total params + per-head params; confirm shared-backbone param savings [8].
- [ ] 139. 🟢 Verify the order of FiLM stages (Pose → HeadPose) is justified and ablated if claimed novel.
- [ ] 140. 🔴 Confirm "one forward pass" empirically: profile that all heads come from a single backbone pass (C1).
- [ ] 141. 🟡 Document which heads are spatial (Phase A) vs temporal (Phase B) and how they unify at inference.
- [ ] 142. 🟢 If negative transfer appears, characterize *which* task pairs conflict (task-affinity analysis) [9].
- [ ] 143. 🟡 Verify no task's loss is NaN/zeroed in the joint phase (your loss-cap / log_var floor history).
- [ ] 144. 🟢 Report training stability (loss curves) for joint vs decoupled — evidence for the curriculum claim.
- [ ] 145. 🔴 Make the multi-task conclusion explicit and honest: synergy / neutral / efficiency-Pareto — whichever the data shows [9].
- [ ] 146. 🟡 Cross-check that the unified model's per-task numbers (final) are the ones reported in the headline table.

## K. Evaluation protocol & baseline-matched metrics (147–162)
- [ ] 147. 🔴 For EACH task, match the baseline's metric AND protocol exactly (GUIDE_3) [1,2,15,32].
- [ ] 148. 🔴 Detection: COCO-style mAP [23]; report present-class + diluted + n_present.
- [ ] 149. 🔴 Activity: clip-level Top-1/Top-5 [15].
- [ ] 150. 🔴 PSR: F1(±3)/POS with the official tolerance [1,2].
- [ ] 151. 🟡 Head/body pose: angular MAE / PCK [17,35].
- [ ] 152. 🔴 All headline numbers on the held-out TEST set (val only for tuning).
- [ ] 153. 🟡 State the eval protocol in the paper for each task (so comparisons are defensible).
- [ ] 154. 🟢 Verify the combined metric is deprioritized (it's MAE-dominated; judge per-task) — you've been misled by it before.
- [ ] 155. 🔴 Re-confirm the honest-metric code path drives reported numbers (grep "det_mAP50_pc").
- [ ] 156. 🟡 Verify evaluation uses EMA weights consistently if that's your deployment model [27].
- [ ] 157. 🟢 Report inference at the native 1280×720 (anchors are calibrated for it).
- [ ] 158. 🟡 Confirm no test-time augmentation unless you report it as a separate row [USE_TTA].
- [ ] 159. 🟢 Sanity-check metric implementations against a tiny known example (e.g., hand-computed AP/F1).
- [ ] 160. 🔴 Ensure the same checkpoint produces the reported numbers across re-runs (determinism).
- [ ] 161. 🟡 Report whether numbers are single-model or ensemble (single, for the efficiency claim).
- [ ] 162. 🟢 Keep a CSV of every (task, metric, value, seed, checkpoint, split) for the tables.

## L. The proving ablations / experimental matrix (163–178)
- [ ] 163. 🔴 Ablation A (the core): single-task vs frozen-MTL vs joint-MTL, per task, same backbone [8,9].
- [ ] 164. 🔴 Ablation B: FiLM conditioning ladder (none/Pose/Pose+HeadPose) [7].
- [ ] 165. 🟡 Ablation C: Kendall vs fixed vs uniform loss weights [6].
- [ ] 166. 🟡 Ablation D: decoupled curriculum vs naive joint-from-scratch (shows your method's value) [25,26].
- [ ] 167. 🟢 Ablation E: with/without synthetic detection pretrain (if data available) [1,32].
- [ ] 168. 🟢 Ablation F: temporal context length for activity/PSR (T sweep).
- [ ] 169. 🔴 Each ablation changes ONE variable (you have a documented history of confounded multi-variable runs).
- [ ] 170. 🟡 Report ablations on val (or a fixed ablation split), final numbers on test.
- [ ] 171. 🟢 Include a "naive joint training fails / interferes" result as evidence for the method [9,10].
- [ ] 172. 🔴 Ensure Ablation A is decisive: it is the experiment that proves/qualifies C4 — budget time for it.
- [ ] 173. 🟡 If multi-task ≥ single-task on any head → emphasize positive transfer [8].
- [ ] 174. 🟡 If multi-task < single-task → emphasize efficiency-Pareto + characterize interference [9].
- [ ] 175. 🟢 Tabulate compute/params for each arm so the trade-off is explicit.
- [ ] 176. 🟢 Cross-task FiLM: show *which* task benefits (pose→activity) for the mechanism story [7].
- [ ] 177. 🟡 Verify ablation deltas exceed seed noise (significance) [37,38].
- [ ] 178. 🔴 Write each ablation's conclusion as one sentence tied to a claim (C1–C5).

## M. Statistical rigor (179–186)
- [ ] 179. 🔴 ≥3 seeds for headline tasks; report mean ± std [38].
- [ ] 180. 🟡 Significance test for key comparisons (paired test across seeds / classes) [37].
- [ ] 181. 🟢 Report variance sources (seed, data order) honestly [38].
- [ ] 182. 🟡 Avoid claiming a win inside the noise band (e.g., +0.003 mAP is noise).
- [ ] 183. 🟢 Confidence intervals on the headline numbers where feasible.
- [ ] 184. 🟡 For per-class metrics, note small-sample instability (rare classes).
- [ ] 185. 🟢 Keep raw per-seed numbers in an appendix/CSV.
- [ ] 186. 🔴 Ensure the "competitive" claims are statistically defensible, not single-run.

## N. Efficiency claims (187–192)
- [ ] 187. 🔴 Measure params (POPW ~53M) vs sum of specialists (~81M: YOLOv8m 26M [32] + MViTv2 35M [15] + STORM-PSR ~20M [2]).
- [ ] 188. 🔴 Measure GFLOPs at 1280×720 and FPS on the RTX 3060 (`efficiency_report.py`).
- [ ] 189. 🔴 Quantify "1 forward pass vs 3" precisely (shared backbone pass + light heads).
- [ ] 190. 🟡 Report memory footprint (single model vs three loaded specialists).
- [ ] 191. 🟢 Note commodity-hardware accessibility (single 12GB GPU) as a practical contribution.
- [ ] 192. 🔴 This table needs no accuracy to be true — make it prominent; it is your most bulletproof claim.

## O. Paper, figures, limitations, reviewer-proofing (193–200)
- [ ] 193. 🔴 Fill every `\todo`/`\popwres` in `popw_paper_improved.tex` (~L539–625) from the results CSV.
- [ ] 194. 🔴 Write the limitations section proactively: no synthetic pretrain for detection [1,32]; single-GPU constraints; fine-grained ASD ceiling; subset-dev/full-test.
- [ ] 195. 🔴 Include the 4 figures: architecture (1 pass), two-stage FiLM, ASD confusion matrix, MTL vs single-task bar chart.
- [ ] 196. 🟡 Prepare rebuttals (GUIDE_4 §7) for: "below YOLOv8m/MViTv2", "is MTL helping?", "why not 3 specialists?".
- [ ] 197. 🟡 State the negative-transfer finding (if any) as a contribution, citing [9,10] — reviewers respect honesty.
- [ ] 198. 🟢 Ensure related-work positions you vs Taskonomy [14], MTL surveys [8,29,33,34], IKEA-ASM [31], MECCANO, IndustReal [1], STORM-PSR [2].
- [ ] 199. 🟢 Release code + configs + checkpoints for reproducibility (matches IndustReal's open release) [1].
- [ ] 200. 🔴 Final gate: every claim C1–C5 has ≥1 test-set table/figure with seeds; every `\todo` filled; limitations written. **When all 200 are checked, the idea is proven and the paper is done.**

---

## References (venue-verified)

1. Schoonbeek et al. *IndustReal: A Dataset for Procedure Step Recognition Handling Execution Errors in Egocentric Videos in an Industrial-Like Setting.* WACV 2024. arXiv:2310.17323.
2. Schoonbeek et al. *Learning to Recognize Correctly Completed Procedure Steps... (STORM-PSR).* CVIU 2025. arXiv:2510.12385. (IndustReal: F1 0.506, POS 0.497.)
3. Lin et al. *Focal Loss for Dense Object Detection (RetinaNet).* ICCV 2017.
4. Lin et al. *Feature Pyramid Networks for Object Detection.* CVPR 2017.
5. Liu et al. *A ConvNet for the 2020s (ConvNeXt).* CVPR 2022.
6. Kendall, Gal, Cipolla. *Multi-Task Learning Using Uncertainty to Weigh Losses...* CVPR 2018.
7. Perez et al. *FiLM: Visual Reasoning with a General Conditioning Layer.* AAAI 2018.
8. Vandenhende et al. *Multi-Task Learning for Dense Prediction Tasks: A Survey.* TPAMI 2021.
9. Standley et al. *Which Tasks Should Be Learned Together in Multi-Task Learning?* ICML 2020.
10. Yu et al. *Gradient Surgery for Multi-Task Learning (PCGrad).* NeurIPS 2020.
11. Sener & Koltun. *Multi-Task Learning as Multi-Objective Optimization (MGDA).* NeurIPS 2018.
12. Chen et al. *GradNorm.* ICML 2018.
13. Liu et al. *End-to-End Multi-Task Learning with Attention (MTAN).* CVPR 2019.
14. Zamir et al. *Taskonomy: Disentangling Task Transfer Learning.* CVPR 2018.
15. Li et al. *MViTv2: Improved Multiscale Vision Transformers...* CVPR 2022.
16. Tong et al. *VideoMAE.* NeurIPS 2022.
17. Zhou et al. *On the Continuity of Rotation Representations in Neural Networks (6D rotation).* CVPR 2019.
18. Feng et al. *Wing Loss for Robust Facial Landmark Localisation.* CVPR 2018.
19. Shrivastava et al. *Training Region-based Object Detectors with Online Hard Example Mining (OHEM).* CVPR 2016.
20. Cui et al. *Class-Balanced Loss Based on Effective Number of Samples.* CVPR 2019.
21. Cao et al. *Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss (LDAM-DRW).* NeurIPS 2019.
22. Rezatofighi et al. *Generalized Intersection over Union (GIoU).* CVPR 2019.
23. Lin et al. *Microsoft COCO: Common Objects in Context.* ECCV 2014.
24. He et al. *Masked Autoencoders Are Scalable Vision Learners (MAE; linear-probe paradigm).* CVPR 2022.
25. Howard & Ruder. *Universal Language Model Fine-tuning (ULMFiT; gradual unfreezing).* ACL 2018.
26. Bengio et al. *Curriculum Learning.* ICML 2009.
27. Tarvainen & Valpola. *Mean teachers... (EMA).* NeurIPS 2017.
28. Liu et al. *Conflict-Averse Gradient Descent (CAGrad).* NeurIPS 2021.
29. Crawshaw. *Multi-Task Learning with Deep Neural Networks: A Survey.* arXiv 2020.
30. Kornblith et al. *Do Better ImageNet Models Transfer Better?* CVPR 2019.
31. Ben-Shabat et al. *The IKEA ASM Dataset.* WACV 2021.
32. Jocher et al. *YOLOv8 (Ultralytics).* 2023.
33. Ruder. *An Overview of Multi-Task Learning in Deep Neural Networks.* arXiv 2017.
34. Caruana. *Multitask Learning.* Machine Learning, 1997.
35. Sun et al. *Integral Human Pose Regression (soft-argmax).* ECCV 2018.
37. Demšar. *Statistical Comparisons of Classifiers over Multiple Data Sets.* JMLR 2006.
38. Bouthillier et al. *Accounting for Variance in Machine Learning Benchmarks.* MLSys 2021.

*Citations grounded in literature verified up to the knowledge cutoff (and live-checked for
[1], [2], [9], [10]). Verify exact page/number details against the published PDFs before
camera-ready.*
