# 135 — Adversarial Review of 50 PSR Questions

**Reviewer:** Agent 7 (Adversarial Debate)
**Date:** 2026-07-06
**Status:** Review of 135_PSR_DEEP_QUESTIONS.md — 50 questions prepared for Opus
**File under review:** `135_PSR_DEEP_QUESTIONS.md`

---

## Section A. Five Strongest Challenges to the PSR Narrative

**Challenge 1: The input_dim=512 vs 768 mismatch (Q8) is a blocking issue that, if confirmed, invalidates every PSR result derived from head outputs.**

The document flags this as a "blocking diagnostic" at line 86 but provides no evidence that it was ever checked. ConvNeXt C5 features after GAP are 768-dimensional. The PSR head constructor defaults to `input_dim=512`. If the forward pass feeds 768-dim tensors into `nn.Linear(512, 256)`, PyTorch raises a runtime error on the first call. The fact that no error was raised means either (a) the actual input is somehow 512-dim (through a different projection path not documented in `psr_transition.py`), or (b) there is silent dimension truncation or broadcasting somewhere in the call chain. This must be resolved by tracing tensor shapes at runtime, not by reading constructor defaults. If the mismatch exists and no error occurs, the projection layer produces garbage for all 11 heads in every checkpoint, and the head repair, LOO-CV, and null-delta are all built on corrupt features.

**Counter-argument:** The document correctly labels this as "blocking" and lists it as a diagnostic to verify. The PSR head is not called directly on ConvNeXt features — there is an intermediate feature extraction path documented in `model.py` that may project to 512-dim before reaching the PSR head. The question identifies uncertainty correctly rather than asserting a bug. No conclusion in the document relies on assuming correct input dimensions.

**Challenge 2: The POS paradox is mathematically trivial, and presenting it as an empirical "finding" wastes reviewer goodwill.**

POS(constant) = 1 - N_transitions / (T-1) for any constant predictor on monotonic data. This is first-semester algebra. The null-model experiment "confirming" what algebra already mandates is equivalent to running an experiment to prove the sky is blue. The document's Debate 5 prosecutes this position effectively but then walks it back in the "Defense" paragraph, arguing the community might not know. The stronger position: the contribution is not the empirical finding that POS is bounded — it is the POS@tolerance salvage proposal, the Edit score inflation parallel, and the methodological template (null-model delta as metric health check). These are the deliverable insights. The empirical null-model POS table is supporting evidence, not a primary finding.

**Counter-argument:** The reviewer's counter in Debate 5 is actually correct here. The community does not systematically check metric bounds on sparse-event data before reporting numbers. The STORM and B3 papers both report POS without bounding it. An explicit demonstration with concrete numbers makes the issue impossible to ignore in a way that a theoretical bound does not. The algebraic bound is necessary but not sufficient for reviewer impact.

**Challenge 3: The LOO-CV (+0.0358) is unreliable because it confounds train/val splits, component-level degradation, and recording-level variation.**

Three separate issues converge on this number. First (Q20), some LOO recordings were in the model's training set, others were held out — yet the improvement is averaged across all 16 without split reporting. Second (Q18), the macro-mean improvement masks potential per-component degradation: a component losing -0.10 could be hidden by four components gaining +0.035 each. The cross-component variance is unreported. Third (Q12), the std (60% of mean) across recordings suggests the gain is not uniform — some recordings drive the entire signal while others contribute nothing. Without per-component, per-recording, by-split reporting, the +0.0358 number carries less weight than the narrative around it suggests.

**Counter-argument:** The document raises all three concerns in the questions themselves (Q12, Q18, Q20). The LOO-CV script can produce per-recording results; they simply haven't been extracted into the Evidence Inventory table yet. The mean is reported transparently with its std. The question format invites Opus to examine these distributions before using the number. The document doesn't claim the +0.0358 is the final answer — it asks whether it holds up under scrutiny.

**Challenge 4: The headline F1 (originally 0.7499 on 10k subset) is an upper bound, not a representative number, and the gap to STORM widens when using the honest 38k eval (corrected to 0.7018).**

The document admits at Q49 that "the 0.7499 is an upper bound, not a representative number, corrected to 0.7018 on full 38k." The full 38k eval at global 0.10 gives macro F1 = 0.677 (from `full_eval_ep18_stream/metrics.json`). Per-comp optimal thresholds have never been applied to all 38k frames. If the 10k optimal thresholds overfit to the subset — yielding only 0.70-0.72 on the full 38k — the honest gap to STORM's 0.901 widens from 0.151 to 0.181-0.201. The "competitive" label in SOTA_STATUS.md depends on which number is published. This decision (D3) is listed as zero-cost compute and should have been run before writing the 50 questions, not scheduled as a parallel task during training.

**Counter-argument:** The D3 item explicitly schedules this computation "immediately on the RTX 3060 while the RTX 5060 Ti trains." The 50 questions were prepared for Opus to make these decisions, not as a final publication draft. The document correctly flags the issue rather than hiding it. The 10k vs 38k gap exists, but its magnitude is unknown, which is precisely why D3 exists.

**Challenge 5: The bundled intervention (head repair + Kendall fixed weights) means no attribution is possible after the in-flight training converges.**

The in-flight training applies both PSR_HEAD_REPAIR=1 and KENDALL_FIXED_WEIGHTS=1 simultaneously. If PSR F1 improves from 0.7018 to 0.8000, the narrative will claim "head repair success" but the Kendall fix could account for 20-60% of the gain. The factorial ablation (D10) is listed as 60-90 hours of compute and tagged as "high-effort" with the recommendation to skip it. The document reasons that the head repair is "the primary cause" based on 133 PSR-3, but PSR-3's evidence was gathered before Kendall was known to be broken. After Q48's own admission, the Kendall fix is expected to contribute +0.01-0.03. If the total improvement is +0.05 and Kendall accounts for +0.03 (60%), the bundled intervention's attribution framework is misleading.

**Counter-argument:** Science advances on bundled interventions. The combined improvement is what matters for the final model. The attribution question is a secondary concern for the ablation section of the paper, not a prerequisite for the primary result. If the combined F1 reaches 0.80+, the narrative "fixing two bugs gave +0.05" is honest and doesn't require fine-grained attribution. The D10 ablation is postponed, not rejected — it can be run after the deadline if time permits.

---

## Section B. Five Evidence Gaps

**Gap 1: No runtime tensor shape verification for PSR head input (Q8).**

The document flags `input_dim=512` default against `768-dim ConvNeXt C5 features` as a blocking diagnostic, then never reports whether it was checked. The gap is that all PSR analysis assumes correct tensor shapes without runtime confirmation. Verifying this requires one `print(x.shape)` call in the forward pass — estimated effort 5 minutes — yet it is not listed as an Open Decision or diagnostic to run. Adding it as D11 would cost less than any other diagnostic.

**Gap 2: No per-component transition F1 extracted (Q45).**

STORM's per-component F1 breakdown (Table 2) is the only direct comparison point with our work. The document acknowledges the gap (Q45: "We need to extract per-component transition F1") but doesn't demonstrate the data exists to extract it. Does `psr_transition_f1.py` already store per-component per-recording results, or would this require code changes? The gap is not just "the data hasn't been formatted" — it's "we don't know if the data structure supports per-component extraction without rewriting the eval script."

**Gap 3: No empirical verification that the 10k optimal thresholds generalize to 38k (Q49/D3).**

D3 estimates 30 minutes to run this. The document was written before D3 was executed. The 0.7018 number appears in SOTA_STATUS.md and the Evidence Inventory without a caveat that it's possibly overfit to a subset. The gap between the reported number and the honest number could be 0.03-0.08 F1. Running D3 before circulating these questions would have anchored the discussion on the stronger or weaker number.

**Gap 4: No train/val prevalence comparison (Q27).**

The null-delta analysis uses validation-set prevalence to compute F1_null. If training-set prevalence differs for any component, the null overestimates the baseline. D6 estimates 10 minutes on CPU. This is a trivial diagnostic that should have been run before the null-delta table was finalized.

**Gap 5: No empirical evidence that the LOO-CV recordings include both train and val splits (Q20).**

The document asks "which recordings are in train vs val?" but doesn't report finding the split definition. This is a data-loading question answerable by reading `dataset.py` or the recording split file. The LOO-CV results cannot be correctly interpreted until this is known. If all 16 recordings were in the training set, the LOO-CV measures within-training-distribution stability, not held-out generalization. If some were in validation, the improvement should be reported separately for each split.

---

## Section C. Five Alternative Interpretations

**Interpretation 1: The head repair may reduce F1 because the loss landscape has shifted away from the dead-head convergence point.**

The dead heads reached F1=0.7018 by fitting prevalence priors with near-constant sigmoid outputs. The transformer features adapted to produce features that, when run through the dead heads, give the optimal constant prediction. After head repair with Xavier initialization (fresh weights), the repaired heads must learn from scratch while the transformer features are already tuned for a different output distribution. There is a risk of negative interaction: the repaired heads produce non-constant outputs, the transformer sees new gradient signals, and the combined system must re-converge to a point that may be no better (or worse) than the dead-head equilibrium. The in-flight training's first 5 epochs will show whether the repaired heads quickly find signal (crossing baseline at epoch 27-28) or struggle (crossing at epoch 35+ or never).

**Interpretation 2: The Kendall fixed weights may account for most of the improvement, not the head repair.**

The document estimates Kendall delta at +0.01-0.03 and head repair delta at +0.03-0.08, but these are guesses. If the Kendall loss weighting was incorrectly zeroing out PSR gradients in the multi-task objective, then fixing it restores gradient flow to the PSR head regardless of whether the head is dead or alive. The original head may have been receiving near-zero gradients from the multi-task loss, not from its own ReLU non-linearities. Under this interpretation, the dead heads are a symptom of a multi-task training bug, not the primary cause. The bundled in-flight training cannot distinguish these. A `KENDALL_FIXED_WEIGHTS=1` only run (without head repair) would resolve this but is not scheduled.

**Interpretation 3: The Gaussian-smeared loss (sigma=3) actively prevents the repaired heads from learning the sharp transitions the decoder needs.**

With sigma=3, the optimal head output is a soft Gaussian ramp spanning 19 frames around each transition. The MonotonicDecoder expects sharp binary state changes. The repaired heads, now learning effectively for the first time, will optimize for the loss they receive (Gaussian targets producing smooth ramps). This produces sigmoid outputs that look like "mountain ranges" — slow ascents and descents around transitions — which the decoder's hysteresis must then threshold. The decoder was designed for sharp ConvNeXt features (every frame has a prediction), not for soft transition ramps. Under this interpretation, the head repair changes the output distribution, but the decoder is optimized for a different output distribution, and the combined F1 may not improve regardless of head health.

**Interpretation 4: Comp9's at-null delta (Q17) is not a bug but a feature bound: PSR is structurally unsolvable for some components.**

Comp9 (mid-placement hand posture) has null-delta = -0.000. The model cannot distinguish comp9's active state from its inactive state using visual features. This may be because comp9's posture is visually identical to an intermediate stage of another component — the hand position is the same, only the assembly context differs. If this is a feature bound (the visual information is not in the 2D ConvNeXt features, regardless of head quality), then the 0.7018 ceiling is an average of 10 solvable components (mean F1 approx. 0.80) and 1 unsolvable component (F1 stuck at 0.69). Under this interpretation, the "gap to STORM" narrows to (STORM's comp9 mean - our comp9 mean), which may be smaller than the overall gap. The appropriate comparison is per-component, not macro-mean.

**Interpretation 5: The per-comp optimal thresholding success is entirely driven by two components (comp4 and comp10), not a general threshold improvement.**

Q15 estimates that the +0.028 improvement from global 0.10 to per-comp optimal is concentrated in comp4 and comp10. The supporting computation is not presented as a completed analysis ("Compute macro F1 with per-comp optimal thresholds EXCEPT comp4 and comp10..." is a future action). If confirmed, the narrative shifts: threshold tuning is not a "general technique that improves all components" but rather "targeted fix for two low-prevalence components where the default threshold was catastrophically wrong." The general applicability claim should be downgraded. Under this interpretation, the threshold improvement on high-prevalence components (comp0-3, 7-8) is noise-level (0-0.005).

---

## Section D. Five New Questions for the PSR Analysis

**Question 1: What is the rollback plan if the in-flight training produces PSR F1 lower than the epoch 18 baseline (0.7018 or 0.677 at global 0.10)?**

The document's D1 says: "If no improvement by epoch 30, stop and diagnose transformer health (Q4)." This plan assumes the repair will either help or be neutral. If F1 drops below baseline (e.g., to 0.65 or lower), the repair is actively harmful. In that case, stopping the training and restoring the dead-head checkpoint is the safe option. But the restored checkpoint still has dead heads and the same ceiling. The question is: is there a third path? Options include (a) re-initialize heads with bias=-0.5 instead of bias=0.0 to provide a moderate prevalence prior, (b) keep the old head weights frozen and add a small residual MLP on top, or (c) train only the heads (freeze transformer) for the first 5 epochs to let them adapt without disrupting the backbone. This contingency plan should be established before epoch 30 arrives.

**Question 2: What is the 95% confidence interval on the LOO-CV +0.0358 improvement, including both cross-recording AND cross-component variance?**

The document reports std = +-0.0216 across 16 recordings. With n=16 and a paired t-test (t_{0.025, 15} = 2.13), the 95% CI from recording variance alone is approximately [0.024, 0.047]. But the cross-component variance is not reported (Q18). If components vary independently, the effective sample size is 16 recordings * 11 components = 176, which would narrow the CI substantially. If components are correlated within recordings (likely — all 11 components share the same lighting, occlusion, worker), the effective sample size is closer to 16. The CI should be computed using a hierarchical bootstrap (resample recordings, then components within recordings) to account for the nested structure. The reported +-0.0216 is likely an underestimate because (a) it ignores component-level variance, and (b) it assumes normality of a small sample.

**Question 3: Does the procedure_order constraint (0->1->...->10) suppress correct D4 detections when YOLOv8m sees components out of sequential order?**

D4's F1=0.000 at default thresholds and 0.347 after retuning. The decoder enforces sequential component placement. But in the YOLOv8m detection pipeline, components may be detected out of sequence due to occlusion (component 5 visible before component 4), simultaneous placement (two components placed at once), or detection noise (component 3 detected as component 4 due to visual similarity). If the decoder encounters a detection for component 5 before component 4 has been placed, it may suppress the detection, reset the state, or produce a false negative. The F1=0.347 may be a function of how often YOLOv8m detections arrive in sequential order, not of detection quality. The per-video breakdown (D4) should include the fraction of detections that arrive in sequential order.

**Question 4: What is the mutual information between ConvNeXt features and comp4/comp10 transition timing?**

The central debate (Debate 1) hinges on whether the comp4 delta (+0.097) comes from per-frame discrimination (backbone sees comp4 features) or calibration (threshold adjustment alone). Mutual information (MI) between the ConvNeXt feature vector and the binary comp4 transition label, computed on held-out frames, answers this directly. If MI > 0, the backbone carries transition information for comp4. If MI ≈ 0, the delta is purely calibration. MI is easy to compute: bin feature activations, compute H(features) + H(labels) - H(features, labels) from the empirical joint distribution. Or use a learned MI estimator (MINE, InfoNCE). This one number would resolve the calibration-vs-detection debate decisively.

**Question 5: At what epoch does the in-flight training cross the epoch 18 baseline, and what does the crossing time reveal about whether the head repair or the Kendall fix is responsible?**

If crossing happens at epoch 27 (1 epoch after resume), the repair was immediately effective — the new heads found signal on the first gradient step. This strongly suggests the heads were the primary bottleneck (the transformer features were healthy, just waiting for heads to use them). If crossing happens at epoch 35+ (9+ epochs), the repair needed significant re-convergence, implying the transformer features also had to adapt to the new output distribution. This would be consistent with either (a) the Kendall fix being the primary driver (the multi-task re-weighting takes time to propagate through the transformer), or (b) the head repair disrupting previously adapted features (heads fixed, but transformer needs to re-learn its feature distribution to match). The crossing epoch is a free diagnostic — just track the eval PSR F1 at each epoch and note when it surpasses 0.7018 (or 0.677 at global 0.10).

---

## Summary for Opus

**Most actionable challenge:** The input_dim mismatch (Q8) must be resolved immediately. One `print(x.shape)` call in the forward pass determines whether the entire PSR analysis stack is built on valid features or silent corruption. D3 should be elevated to blocking priority.

**Least defensible number:** The 0.7499 per-comp optimal F1 on 10k frames was an upper bound, not a representative result (corrected to 0.7018 on 38k). Publish the full 38k per-comp optimal alongside it, or caveat aggressively.

**Most interesting alternative interpretation:** The Gaussian-smeared loss (sigma=3) and the MonotonicDecoder hysteresis may be actively working against each other — the loss rewards smooth ramps, the decoder rewards sharp steps. If confirmed, this is a design-level flaw that no amount of head repair can fix.

**Most useful new question:** The crossing epoch of the in-flight training (the epoch at which F1 surpasses the epoch 18 baseline) is a free diagnostic that directly reveals whether head repair alone was sufficient or whether re-convergence is needed. Track it from epoch 27 onward.

---

**End of 135_PSR_DEBATE.md. Prepared for Opus alongside 135 (questions) and 136 (answers).**
