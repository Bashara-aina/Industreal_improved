# Doc 226: Implementation Roadmap — After Claude Science Returns

**Document:** 226 of 227 (Claude Science consultation package, docs 208-227)
**Status:** Action plan — execute after Claude Science findings are received and triaged
**Date:** 2026-07-11
**Audience:** Research team (this is the execution playbook for the 30 days following Claude Science output)
**Prerequisites:** Claude Science findings (doc 215 answers), all ST baselines complete, overfit probes passed

---

## Table of Contents

1. Triage Process: Evaluating Claude Science Recommendations
2. Phase 1 (Days 1-3): Quick Wins
3. Phase 2 (Days 4-10): Architecture Changes
4. Phase 3 (Days 11-20): Training Runs
5. Phase 4 (Days 21-30): Paper Writing
6. Compute Budget Allocation
7. Decision Gates: When to Pivot, When to Commit
8. Fallback Timeline: If Claude Science Findings Are Limited

---

## 1. Triage Process: Evaluating Claude Science Recommendations

Claude Science will return a set of papers, methods, and architecture suggestions. Not all will be actionable. This triage process filters them into three buckets: **implement now**, **ablate later**, and **archive for next paper**.

### 1.1 Impact vs. Cost Assessment per Finding

Every recommendation is scored on two axes before any code is written.

**Impact score** (1-5, estimated from published results):
- 5 = Closes a >=15 point gap on any head (e.g., activity from 0% to >15%, detection from 0.202 to >0.35 mAP)
- 4 = Closes a 5-14 point gap, or improves two heads simultaneously
- 3 = Closes a 2-4 point gap, or improves the efficiency spine
- 2 = Marginal improvement (<2 points), or improves only one head marginally
- 1 = Theoretical/no published evidence of improvement

**Cost score** (1-5, in engineering time + GPU hours):
- 5 = Under 1 day of coding + under 10 GPU-hours to validate
- 4 = 1-2 days of coding + 10-50 GPU-hours
- 3 = 2-5 days of coding + 50-200 GPU-hours
- 2 = 5-10 days of coding + 200-500 GPU-hours
- 1 = Over 10 days or over 500 GPU-hours (requires rethinking the paper timeline)

**Decision matrix:**

| Impact \ Cost | 5 (Cheap) | 4 | 3 | 2 | 1 (Expensive) |
|---|---|---|---|---|---|
| 5 (High) | **Implement now** | **Implement now** | Phase 2 candidate | Evaluate carefully | Next paper |
| 4 | **Implement now** | Phase 1/2 | Phase 2 | Phase 3 | Next paper |
| 3 | Phase 1 | Phase 2 | Phase 3 ablation | Archive | Archive |
| 2 | Phase 1 if idle | Phase 2 ablation | Archive | Archive | Archive |
| 1 (Low) | Archive | Archive | Archive | Archive | Archive |

**Implementation rules:**
- **Implement now**: Start within 24 hours of triage. Blocking priority.
- **Phase 1/2**: Add to the appropriate phase's backlog. Estimate capacity.
- **Phase 3 ablation**: Design as an ablation experiment during the main training phase. Changes the test matrix, not the core model.
- **Archive**: Log in `analyses/claude_science_findings_archive.md` with the reasoning. Revisit only if a related method fails.
- **Next paper**: Keep in a separate "Future Work" buffer. Not for this submission cycle.

### 1.2 Compatibility with Existing Codebase

Each recommendation must pass a compatibility review before implementation begins.

**Compatibility criteria:**

| Criterion | Pass | Fail | Mitigation |
|---|---|---|---|
| Works with MViTv2-S backbone | Method has been applied to ViT/transformer backbones | Method requires CNN-specific ops (e.g., deformable convs) | Port to transformer equivalent or reject |
| Respects single-forward-pass constraint | Method adds <5% overhead to forward pass | Method requires N forward passes (N tasks > 1) | Task-specific adapters break the latency spine. Reject unless efficiency claim is abandoned. |
| Fits in 12-16 GB VRAM | Peak VRAM increase <2 GB | Peak VRAM increase >4 GB | Reduce batch size, accept the train-time cost |
| Compatible with bf16 | Method uses standard operations | Method requires fp64 or specialized kernels | Test bf16 compatibility in a 50-step probe |
| Compatible with PCGrad + Kendall | Method works alongside existing gradient surgery | Method replaces the gradient surgery framework entirely | If the replacement is Nash-MTL or similar, it becomes a Phase 3 ablation, not a replacement of the core method |

**Compatibility test protocol:**
1. Read the paper's official implementation (if available). Check dependencies.
2. Write a unit test that runs the method on a single batch of our data through our model.
3. Verify: (a) forward pass completes, (b) backward pass completes, (c) loss is finite, (d) VRAM is within budget.
4. If any of these fail, move to "archive" and document the incompatibility.

### 1.3 Risk of Integration

Beyond raw compatibility, integration risk considers the method's impact on the project's stability and timeline.

**Risk categories:**

| Risk Level | Definition | Examples | Response |
|---|---|---|---|
| **Low** | Config flag or isolated module swap. No downstream dependencies. | Enabling Mosaic augmentation, changing loss hyperparameters, threshold tuning | Implement without reservation. Add to ongoing experiment. |
| **Medium** | New module or training loop change. Affects one head but not others. | New detection head design, PSR transition predictor, new loss function for one head | Implement in a feature branch. Run overfit probe + 10-epoch validation before merging to main. |
| **High** | Changes the training paradigm. Affects all heads. Requires rerunning all baselines. | New gradient surgery method, backbone swap, new multi-task weighting scheme | Must pass a design review (this document). Requires full re-baselining. Accept only if impact score >= 4. |
| **Critical** | Changes the paper's core claim. Invalidates previous results. | Dropping a head, abandoning the single-pass claim, changing the dataset | Requires team discussion. Do not implement without explicit sign-off documented in this file. |

**Rollback plan for every integration:**
- Every method change must have a `--disable-<method>` flag that restores the baseline behavior.
- Before implementing any change, verify that running with the disable flag reproduces the baseline results within statistical noise.
- This is non-negotiable. Without a rollback path, the change is not implemented.

---

## 2. Phase 1 (Days 1-3): Quick Wins

Phase 1 runs in parallel with the triage process. These are zero-risk, high-confidence improvements that should be done regardless of Claude Science findings.

### 2.1 ST Baseline Completion (Day 1-2, RTX 5060 Ti)

Single-task baselines are the paper's spine. Without them, no MTL/ST ratios exist. Run these on the RTX 5060 Ti while the RTX 3060 handles other work.

| Baseline | Duration | GPU | Priority | Notes |
|---|---|---|---|---|
| ST pose | ~3.5 hours | RTX 5060 Ti | **Highest** | Already has code, needs 5 seeds |
| ST detection | ~7 hours | RTX 5060 Ti | **High** | OHEM should be verified on/off |
| ST PSR | ~5 hours | RTX 5060 Ti | **High** | Critical diagnostic: is PSR failure architectural or MTL-specific? |
| ST activity | ~5 hours | RTX 5060 Ti | **High** | Low expectations, but must have the number |

**Total Phase 1 ST: ~20.5 hours sequential, ~10 hours parallelized (back-to-back on one GPU).**

Each ST baseline uses the MViTv2-S backbone with only the relevant head. Use the same hyperparameters as the MTL config but with single-head loss only. Five seeds each (103-107). Report mean + 95% bootstrap CI.

### 2.2 Detection Quick Fixes (Day 1-2, RTX 3060)

These are config-level changes that cost zero engineering time:

1. **OHEM ablation (highest ROI)**: Set `DET_OHEM_RATIO=0` and `DET_MIN_NEG=0`. The per-head gap analysis (doc 212) projects +0.05-0.10 mAP from this alone. Run 3 seeds, compare to baseline MTL.
2. **Mosaic augmentation enablement**: Already implemented but never activated (doc 208). Expected +3-5 mAP from published results. Run with 3 seeds.
3. **Detection threshold calibration re-verify**: The mAP is invariant to threshold in 0.0003-0.5 range per config.py:726. Confirm this still holds with MViTv2-S backbone.

**Total Phase 1 detection quick fixes: ~8 GPU-hours per configuration (3 seeds x ~2.7 hours per MTL run).**

### 2.3 Activity Recovery Probe (Day 2, RTX 3060)

The activity head has demonstrated zero learning. Before any Claude Science recommendations can help, we need to verify the head can learn at all:

1. **Fixed-backbone activity probe**: Freeze the MViTv2-S backbone. Train only the activity head with cross-entropy on 75 classes. If top-1 exceeds 5% within 10 epochs, the head is functional and the issue is gradient conflict. If it stays near 0%, the head architecture is fundamentally broken.
2. **Logit adjustment for long-tail**: Apply the standard logit-adjustment (`tau * log(pi)` where pi is class prior) to the activity head's output logits. This is a 3-line code change that helps all long-tail classifiers.
3. **Class grouping probe**: If verb-grouped classification (75 -> 69 classes from doc 111) improves results, apply it as a permanent change.

**Total Phase 1 activity: ~6 GPU-hours.**

### 2.4 PSR Diagnostic Run (Day 2-3, RTX 3060)

PSR needs the same "can it learn at all?" diagnosis as activity:

1. **ST PSR baseline**: Train PSR head alone with MViTv2-S backbone. If event-F1 > 0.0, the problem is not MTL-specific.
2. **Constant-prediction diagnosis**: Log the output distribution at every epoch. If the model always predicts the majority class (label is 95% static), the focal loss gamma or the class weighting needs adjustment.
3. **Gaussian-smeared targets**: Replace hard binary transition labels with Gaussian-blurred targets (sigma = 2 frames). This gives the model a gradient signal even when it predicts slightly off-timing transitions.

**Total Phase 1 PSR: ~8 GPU-hours.**

### 2.5 Infrastructure Hardening (Day 1, Parallel)

While training runs, harden the evaluation and logging infrastructure:

1. **Full test set evaluation**: Remove the EVAL_MAX_BATCHES=250 cap. Run on the complete test set (38K frames). Verify timeout doesn't occur. This is required for paper-quality numbers.
2. **LIVENESS_GRAD probe always-on**: The single most important diagnostic from doc 209's lessons. Ensure gradient norms per head are logged every N steps.
3. **Log-var trajectory logging**: Already mentioned in doc 223. Ensure log-vars are logged at every epoch. Plot them. A monotonic decrease means the task is learning; a rise means the model is down-weighting that task (giving up).
4. **Seed management automation**: Implement the `seeds.csv` tracking from doc 223 Section 1.2. Every run auto-appends its seed configuration.

**Total Phase 1 infrastructure: No GPU cost. 1 day of engineering overhead.**

### 2.6 Phase 1 Summary

| Item | GPU-Hours | Risk | Outcome |
|---|---|---|---|
| ST pose (5 seeds) | 17.5 | Low | MTL/ST ratio for the paper's strongest head |
| ST detection (5 seeds) | 35 | Low | Detection MTL/ST ratio |
| ST PSR (5 seeds) | 25 | Low | Diagnostic: architectural vs. MTL failure |
| ST activity (5 seeds) | 25 | Low | Diagnostic: can the head learn at all? |
| OHEM ablation (3 seeds) | 24 | Low | +0.05-0.10 mAP potential |
| Mosaic enablement (3 seeds) | 24 | Low | +3-5 mAP potential |
| Activity recovery probe | 6 | Low | Verify head functionality |
| PSR diagnostics | 8 | Low | Diagnose constant-prediction problem |
| Infrastructure hardening | 0 | Low | Required for all subsequent phases |
| **Total Phase 1** | **~165 GPU-hours** | | **3 calendar days, both GPUs running** |

These runs should be structured as a single experiment matrix launched on Day 1. Do not wait for one run to finish before launching the next. The RTX 5060 Ti runs the 5-seed ST baselines back-to-back while the RTX 3060 runs the diagnostic probes.

---

## 3. Phase 2 (Days 4-10): Architecture Changes

Phase 2 implements the Claude Science recommendations that passed triage with impact >= 4 and cost <= 4. It also implements the pre-planned architecture improvements that do not depend on Claude Science findings.

### 3.1 Detection Architecture (Days 4-6, RTX 3060)

The detection head is the primary bottleneck at 0.202 mAP. The per-head gap analysis identifies three concrete paths:

**Path A: Anchor-free detection (highest priority)**
- Replace the RetinaNet-style anchor-based detection head with an anchor-free design (FCOS-style or the existing `roi_detector.py` module).
- Expected gain: +0.05-0.15 mAP per doc 212.
- Engineering effort: 1-2 days to implement and overfit-probe.
- Risk: Medium. The anchor-free design eliminates the anchor-GT geometry mismatch (doc 209 Section 5.3 identified anchor sizes 24-384px vs GT centers 164-404px). But it requires re-integrating with the FPN.

**Path B: Gradient starvation fix**
- If OHEM ablation from Phase 1 shows large improvement (>=0.05 mAP), the fix is confirmed: OHEM's negative mining is starving the detection gradient.
- Implement a dynamic OHEM ratio that starts aggressive (early training needs hard negatives) and decays to zero (later training needs all gradients).
- Engineering effort: 0.5 days.
- Risk: Low. Config-only change.

**Path C: Detection-specific FPN features (if recommended by Claude Science)**
- Some MTL papers recommend task-specific feature channels from the shared FPN. This is a small parameter increase (<0.5M) but could reduce gradient conflict between detection and other heads.
- Only implement if Claude Science identifies a paper with published positive results for this approach.

**Decision rule:** Run Path A and Path B simultaneously if resources permit. If only one, Path A has higher potential ceiling.

### 3.2 PSR Transition Prediction (Days 5-7, RTX 3060)

The PSR head's fundamental flaw is predicting per-frame binary states when the signal is in the transitions. The fix is architectural:

1. **PSRTransitionPredictor enablement** (already exists: `psr_transition.py`): Switch from per-frame binary classification to transition-event prediction. The module predicts "when does the state change?" rather than "what is the state now?"
2. **Gaussian-smeared targets**: Smooth the 95% static binary labels with a Gaussian kernel (sigma = 2-3 frames) centered on transition frames.
3. **Post-processing**: After transition prediction, run the MonotonicDecoder (already exists, was previously broken per F22b) to enforce the "once-on, stays-on" constraint.
4. **PSR monotonicity lever**: Apply the median filter from Lever 1 (doc 208). Expected +0.05-0.15 event-F1 from this alone.

**Engineering effort:** 1-2 days for integration and overfit probe. The components exist but need wiring and testing.

**Validation criteria (from doc 212):**
- Success: event-F1 > 0.15 after monotonicity
- Strong: event-F1 > 0.30
- Failure: event-F1 < 0.10 (drop PSR from paper)

### 3.3 Activity Recovery (Days 6-8, RTX 3060)

Activity is the hardest head. If it didn't recover in Phase 1's probe, Phase 2 interventions are more aggressive:

1. **Two-stage training**: First, freeze the backbone and train the activity head alone on cached backbone embeddings. This eliminates gradient conflict entirely. After the head converges, unfreeze the backbone for joint fine-tuning. This is a standard MTL technique that doc 209 notes was never tried.
2. **Claude Science-identified methods**: If Claude Science found a paper with a proven MTL method for long-tail activity classification (questions 6-10 from doc 215), implement it here.
3. **VideoMAE stream**: The MViTv2-S backbone's VideoMAE pretraining was never fully enabled per doc 209. If Claude Science confirms this helps, enable it. Expected +5-10% top-1.
4. **Default to fallback**: If none of the above produces top-1 > 10%, the activity head becomes a "failure dissection" section in the paper rather than a claim. This is acceptable per doc 212's risk assessment.

**Decision rule at Day 8:** If activity top-1 < 10%, stop investing in activity recovery. Document the failure mechanism and move to paper writing. The paper stands on pose + detection + PSR without activity.

### 3.4 Pose Refinement (Days 7-8, RTX 3060)

Pose is already the strongest head. Architecture changes here have diminishing returns:

1. **GeometryAwareHeadPose**: Implement the 6D rotation representation + geodesic loss (251 lines already written per doc 212). Expected -2 to -5 degrees MAE improvement.
2. **Head warm-starting from ST checkpoint**: If Phase 1's ST pose baseline is substantially better (>1 degree), initialize the MTL pose head from the ST checkpoint.
3. **Do not over-invest**: Pose at 9 degrees is already publishable as the first MTL head pose baseline on IndustReal. Every hour spent on pose refinement is an hour not spent on detection or activity recovery.

**Engineering effort:** 0.5 days. Low priority within Phase 2.

### 3.5 Claude Science Special Projects (Days 8-10)

Days 8-10 are reserved for implementing the highest-impact Claude Science recommendation that requires more than 2 days of engineering.

**Possible candidates (depending on Claude Science output):**
- **New gradient surgery method**: If Claude Science identifies a gradient surgery method proven to work better than PCGrad for 4+ diverse tasks (doc 215 questions 26-30), implement and validate it. This is a training-loop change affecting all heads, so it requires careful before/after comparison.
- **Cross-task feature routing**: If Claude Science identifies a lightweight routing mechanism (not LoRA adapters that break single-pass) that improves feature sharing, implement it. Examples: task-specific attention heads within the shared backbone, FiLM modulation, channel gating.
- **Data augmentation from literature**: If Claude Science finds an augmentation strategy specifically designed for MTL with small datasets (doc 215 questions 36-40), implement it.

**Integration test for each special project:**
1. Implement in a feature branch.
2. Run overfit probe (50 images, frozen backbone). Verify the head can overfit.
3. Run 10-epoch validation on the full dataset (single seed). Compare to Phase 1 baseline.
4. If the 10-epoch validation shows improvement, promote to the main experiment matrix for Phase 3.
5. If it shows degradation or no improvement, archive with documentation.

### 3.6 Phase 2 Summary

| Item | GPU-Hours | Calendar Days | Dependency |
|---|---|---|---|
| Anchor-free detection | ~50 | 3 | Phase 1 OHEM results |
| PSR transition prediction | ~40 | 3 | Phase 1 PSR diagnostics |
| Activity two-stage training | ~35 | 3 | Phase 1 activity probe |
| Pose geometric head | ~10 | 1 | Phase 1 ST pose |
| Claude Science special projects | 30-100 | 2-3 | Claude Science findings |
| Integration testing | 20-40 | 2 | All Phase 2 changes |
| **Total Phase 2** | **~185-275 GPU-hours** | **7 calendar days** | |

**Capacity note:** Phase 2 runs both GPUs continuously. The RTX 5060 Ti handles training runs (the long experiments) while the RTX 3060 handles integration tests and probes (short, interactive experiments).

---

## 4. Phase 3 (Days 11-20): Training Runs

Phase 3 is the heavy compute phase. All architecture changes are frozen by Day 10. Phase 3 runs the experiments that generate paper numbers: the main MTL comparison, ST baselines (re-run if architecture changed), and the ablation matrix.

### 4.1 Full MTL Training (Days 11-15, RTX 5060 Ti)

Run the full MTL model with the finalized architecture from Phase 2.

**Configuration:**
- Model: MTL-MViT with all Phase 2 architecture improvements applied
- Backbone: MViTv2-S (Kinetics-400 pretrained), all layers unfrozen
- Training: 100 epochs, effective batch size 16 (B=4 x accum=4)
- Precision: bf16 mixed precision
- Optimizer: AdamW (backbone LR=1e-4, head LR=1e-3)
- Loss weighting: Kendall log-var with caps (det=1.5, act=1.0, psr=0.5, pose=2.0)
- Gradient surgery: PCGrad (or replacement if Claude Science recommended one)
- Regularization: EMA (decay=0.995), SWA (last 5 checkpoints), gradient clip=5.0
- Augmentation: Mosaic + Copy-Paste (if Phase 1 showed positive results)
- Distillation: Knowledge distillation from ST teachers (if ST baselines have meaningful signal)

**Seed regime:** 5 seeds (103-107) for all main experiments.

**Duration:** 5 seeds x ~10 hours per MTL run = ~50 GPU-hours sequential, ~25 hours if parallelized across 2 GPUs (not possible with heterogeneous hardware). On RTX 5060 Ti: approximately 3 calendar days running back-to-back.

### 4.2 ST Baseline Re-runs (Days 12-15, RTX 3060)

If Phase 2 changed the architecture for any head (e.g., anchor-free detection, PSR transition predictor), the corresponding ST baselines must be re-run.

**Why re-run:** The MTL/ST ratio is the paper's core claim. If the detection head changed, the ST detection baseline must use the same new head to make the comparison valid.

**Re-run matrix:**

| Head | If architecture changed | Run time (5 seeds) | GPU |
|---|---|---|---|
| Detection | Anchor-free implemented | ~35 hours | RTX 3060 |
| PSR | Transition predictor enabled | ~25 hours | RTX 3060 |
| Activity | Two-stage training added | ~25 hours | RTX 3060 |
| Pose | Geometry-aware head added | ~17 hours | RTX 3060 |

**Each ST baseline:** Same head architecture as MTL version, same backbone, same training budget. Single-head loss only. 5 seeds.

**Heads without architecture changes:** Do not re-run. Use ST baselines from Phase 1.

### 4.3 Ablation Matrix (Days 15-18, RTX 3060 + RTX 5060 Ti)

The ablation matrix runs on both GPUs. Each ablation is a single-factor change from the main MTL configuration. 3 seeds per ablation.

**Pre-planned ablations (from doc 222, confirmed necessary):**

| Ablation | Purpose | Expected result | Seeds | GPU-hours |
|---|---|---|---|---|
| No PCGrad | Measure gradient surgery impact | +2-8% interference | 3 | 30 |
| No Kendall weighting (fixed weights) | Measure loss balancing impact | +1-5% imbalance | 3 | 30 |
| No SWA | Measure averaging impact | -0.5 to -2% | 3 | 30 |
| No distillation | Measure knowledge transfer impact | -1 to -8% | 3 | 30 |
| No head warm-starting | Measure initialization impact | -1 to -5% | 3 | 30 |
| No monotonicity (PSR) | Measure constraint impact | -0.05 to -0.15 F1 | 3 | 30 |
| No Mosaic augmentation | Measure augmentation impact | -3 to -5 mAP | 3 | 30 |
| Higher resolution (320px) | Measure resolution ceiling | +2-5 mAP | 3 | 45 |

**Claude Science-inspired ablations (if applicable):**

| Ablation | Source | Seeds | GPU-hours |
|---|---|---|---|
| Alternative gradient surgery (Nash-MTL, CAGrad, etc.) | Claude Science finding | 3 | 30 per variant |
| Cross-task feature routing ablation | Claude Science finding | 3 | 30 |
| Task-specific augmentation ablation | Claude Science finding | 3 | 30 |

**Total ablation budget:** Pre-planned: 8 ablations x 3 seeds x ~10 hours = ~240 GPU-hours. Claude Science-inspired: up to 3 additional ablations = +90 GPU-hours.

### 4.4 Statistical Analysis (Days 16-20, Parallel with Training)

While training runs complete, run the statistical analysis pipeline:

1. **Bootstrap confidence intervals**: For every metric, every condition. B = 10,000 resamples. 95% CI.
2. **Paired MTL vs ST tests**: Bootstrap paired differences for each head. Holm-Bonferroni correction.
3. **Ablation analysis**: Benjamini-Hochberg FDR control at q = 0.1 for the ablation matrix.
4. **Effect sizes**: Cohen's d for every significant comparison.
5. **Log-var trajectory analysis**: Plot log-var trajectories across training for the main MTL run. Extract the story they tell about task competition dynamics.
6. **Per-class breakdown**: For detection (24 classes), activity (75 classes), PSR (11 states). Identify which classes benefit from MTL and which degrade.
7. **Training curve analysis**: Plot loss curves and metric curves for all heads. Identify convergence patterns.

**All analysis runs in `analyze_results.py` (from doc 223 Section 7.7).** The script reads raw per-run JSONL results, computes all statistics, and outputs LaTeX tables.

### 4.5 Phase 3 Summary

| Item | GPU-Hours | Calendar Days | Notes |
|---|---|---|---|
| MTL main (5 seeds) | 50 | 3 | RTX 5060 Ti |
| ST re-runs (if needed) | 50-100 | 3-4 | RTX 3060, overlaps MTL |
| Ablation matrix | 240-330 | 4-5 | Both GPUs |
| Statistical analysis | 0 (CPU) | 2-3 | Overlaps training |
| **Total Phase 3** | **~340-480 GPU-hours** | **10 calendar days** | |

**Parallelization strategy:**
- RTX 5060 Ti: MTL main (5 seeds, sequential). This is the longest pole.
- RTX 3060: ST re-runs (if needed), then ablation matrix.
- CPU: Statistical analysis runs alongside training. No GPU needed.

---

## 5. Phase 4 (Days 21-30): Paper Writing

Phase 4 is the writing and figure generation phase. No new training runs occur unless a review cycle identifies a gap that requires a specific additional experiment (budget: up to 50 GPU-hours reserved for this purpose).

### 5.1 Figure and Table Generation (Days 21-24)

Generate all figures and tables following the template in doc 224.

**Table 1: Efficiency comparison**
- Parameters (total, per component), FLOPs, latency, FPS, VRAM
- MTL vs 4x ST comparison with savings ratios
- Per-head overhead in params and FLOPs

**Table 2: Main results**
- Per-head metrics for MTL (mean + 95% CI across 5 seeds)
- Per-head metrics for each ST baseline
- MTL/ST ratio with 95% CI
- Paired difference with significance markers
- SOTA anchor (with honest gap decomposition per doc 212)

**Table 3: Ablation results**
- All ablations sorted by impact (most to least destructive per doc 223 Section 9.5)
- Composite score alongside per-head metrics
- Statistical significance markers

**Table 4: Per-class breakdown (supplementary)**
- Per-class precision, recall, F1 for detection, activity, PSR
- Identify classes with positive and negative transfer

**Figure 1: Model architecture diagram**
- MViTv2-S backbone -> FPN -> 4 heads
- Feature extraction points, head structures
- Shared backbone + parallel heads visual

**Figure 2: Training curves**
- Four panels: loss per head over epochs
- MTL vs ST curves on each panel
- +/- 1 std shading across 5 seeds

**Figure 3: Log-var trajectories**
- Four log-vars over training epochs
- +/- 1 std shading
- Annotations for key events (e.g., "detection confidence rising," "activity being down-weighted")

**Figure 4: Qualitative results**
- Detection: GT boxes vs predicted boxes on sample frames
- Activity: Confusion matrix for top-20 classes
- PSR: Ground truth vs predicted state timelines
- Pose: Forward vector overlay on sample frames

**Figure 5: Efficiency-performance tradeoff (if applicable)**
- Pareto frontier: average per-task performance vs total parameters
- Mark MTL point and 4x ST point

### 5.2 Paper Sections (Days 22-28)

Write the paper sections in order of importance:

**Day 22-24: Method (section 3)**
- Architecture description: shared backbone, FPN, four heads
- Training methodology: Kendall, PCGrad, EMA, SWA, distillation
- Efficiency claim: single-forward-pass, 3.3x parameter savings, 4x FLOPs savings

**Day 24-26: Experiments (section 4)**
- Dataset and setup: IndustReal, metrics, baseline descriptions
- Main results: Table 2 with MTL vs ST comparison
- Ablation study: Table 3 with component analysis
- Efficiency analysis: Table 1 with detailed breakdown

**Day 26-27: Introduction and Related Work (sections 1 and 2)**
- Introduction: Problem statement, hypothesis, contributions
- Related work: MTL methods, assembly perception, efficient video understanding
- Frame the gap honestly: our MTL/ST ratios are the contribution, not SOTA chasing

**Day 27-28: Discussion, Conclusion, Abstract (sections 5, 6, abstract)**
- Discussion: Why MTL works (or doesn't) for each head, failure analysis
- Conclusion: Summary of findings, limitations, future work
- Abstract: 150-200 words covering the problem, method, key results, contributions

### 5.3 Internal Review (Days 28-30)

Three rounds of internal review, each with a different lens:

**Round 1: Claim verification (Day 28)**
- For every claim in the paper, verify the number against the raw results JSONL.
- Check all statistical significance claims against the bootstrap analysis.
- Verify all comparisons are fair (same backbone, same data, same metric).
- Flag any number that changed between draft versions.

**Round 2: Figure quality (Day 29)**
- Check all figures for resolution, font size, axis labels, legend clarity.
- Verify all tables are formatted for the venue template (AAIML/WACV).
- Check that colorblind-friendly palettes are used.
- Ensure all figures are reproducible from the analysis scripts.

**Round 3: Narrative coherence (Day 30)**
- Does the abstract match the conclusions? Do the conclusions match the results?
- Is the story consistent: "MTL is efficient, MTL can work for some tasks, here is how we measure it honestly"?
- Are the limitations stated clearly and fairly?
- Would a reviewer who is skeptical of MTL find the arguments convincing?

### 5.4 Phase 4 Summary

| Item | Calendar Days | Who | Notes |
|---|---|---|---|
| Figure and table generation | 3 (Days 21-23) | Lead author | Use doc 224 templates |
| Method section | 3 (Days 22-24) | Lead author | Architecture + training methodology |
| Experiments section | 3 (Days 24-26) | All authors | Results and ablation analysis |
| Intro + Related work | 2 (Days 26-27) | Co-author | Literature review from doc 220 |
| Discussion + Conclusion | 2 (Days 27-28) | Lead author | Failure analysis, limitations |
| Internal review (3 rounds) | 3 (Days 28-30) | All authors + external reviewer | Full paper pass each round |
| **Total Phase 4** | **10 calendar days** | | |

---

## 6. Compute Budget Allocation

### 6.1 Hardware Inventory

| GPU | VRAM | Architecture | Role | Max simultaneous experiments |
|---|---|---|---|---|
| RTX 3060 | 12 GB | Ampere (compute 8.6) | Ablations, probes, ST baselines | 1 |
| RTX 5060 Ti | 16 GB | Blackwell (compute 12.0) | Main MTL training | 1 |

**Critical constraint:** These GPUs cannot be combined for data-parallel training (different architectures). Each runs independent experiments. The RTX 5060 Ti is ~1.5-2x faster than the RTX 3060 for training due to newer architecture and higher VRAM bandwidth.

### 6.2 GPU-Hours Budget by Phase

| Phase | RTX 3060 hours | RTX 5060 Ti hours | Total | Calendar days |
|---|---|---|---|---|
| Phase 1: Quick wins | 80 | 85 | 165 | 3 |
| Phase 2: Architecture | 120 | 100 | 220 | 7 |
| Phase 3: Training | 200 | 180 | 380 | 10 |
| Phase 4: Buffer | 30 | 20 | 50 | 0 (buffer) |
| **Total** | **430** | **385** | **815** | **30** |

**Running both GPUs continuously for 30 days at ~20 hours/day (accounting for overhead, debugging, evaluation pauses):**
- RTX 3060: 430 GPU-hours (72% utilization)
- RTX 5060 Ti: 385 GPU-hours (64% utilization)
- Combined utilization: ~68%

**Slack capacity:** ~35% of each GPU's time is reserved for debugging, evaluation, experiment configuration changes, and unexpected delays. This is realistic for a research operation where no experiment runs perfectly the first time.

### 6.3 Cloud GPU Fallback

If on-premise compute is insufficient or a GPU fails:

| Option | Cost | Availability | Notes |
|---|---|---|---|
| RunPod RTX 4090 | ~$0.34/hr | Instant | 24 GB VRAM, 3x faster than RTX 3060 |
| RunPod RTX 6000 Ada | ~$0.59/hr | Instant | 48 GB VRAM, batch size 32 possible |
| Lambda RTX 4090 | ~$0.40/hr | ~2 min setup | Similar to RunPod |
| Vast.ai A5000 | ~$0.25/hr | Variable | Cheapest option for ablation runs |

**Budget allocation:** Reserve $200 for cloud GPU fallback. This covers approximately 500 GPU-hours on an A5000 or 300 GPU-hours on an RTX 4090.

**When to use cloud:**
- RTX 3060 is saturated and RTX 5060 Ti is running a critical path experiment. Offload ablations to cloud.
- A GPU fails. Use cloud as replacement while waiting for hardware repair.
- Need to run a single large-batch experiment (BS=32+). Cloud GPUs with 24-48 GB VRAM can handle this.

### 6.4 Power and Thermal Budget

| GPU | TDP | Daily power (20h) | 30-day power cost at $0.12/kWh |
|---|---|---|---|
| RTX 3060 | 170W | 3.4 kWh | $12.24 |
| RTX 5060 Ti | 180W | 3.6 kWh | $12.96 |
| **Total** | **350W** | **7.0 kWh/day** | **~$25.20** |

**Thermal management:**
- Room temperature during sustained load: expect +5-8 degrees C above ambient.
- If room exceeds 30 degrees C, GPU throttling may occur. Use external fans or reduce ambient temperature.
- Monitor GPU temperature via `nvidia-smi`. Throttle begins at ~83 degrees C for both GPUs.
- Schedule a 30-minute cooldown every 8 hours of continuous load.

---

## 7. Decision Gates: When to Pivot, When to Commit

Decision gates are checkpoints where experimental results determine whether to continue on the current path or pivot to a fallback plan. Each gate has clear criteria.

### 7.1 Gate G1: End of Phase 1 (Day 3)

**Trigger:** ST baselines and quick fix results available.

**Criteria:**

| Condition | Decision |
|---|---|
| ST pose MAE < 10 degrees AND ST detection mAP > 0.30 | **Commit to full paper.** Proceed to Phase 2 with confidence. |
| ST pose MAE > 15 degrees | **Investigate critical bug.** Pose has always worked; failure here indicates a fundamental regression in the codebase. Do not proceed until resolved. |
| ST detection mAP < 0.15 | **Re-evaluate detection head.** The head architecture may be fundamentally broken irrespective of MTL. Run additional diagnostic probes before Phase 2. |
| ST activity top-1 > 15% | **Good news.** Activity can learn outside MTL. Proceed with Phase 2 recovery strategies. |
| ST activity top-1 < 5% | **Activity is structurally broken.** Proceed with Phase 2 but prepare to drop activity head from paper. |
| ST PSR event-F1 > 0.10 | **PSR can learn.** The MTL-specific recovery strategies (transition prediction, monotonicity) are worth implementing. |
| ST PSR event-F1 < 0.05 | **PSR is structurally broken.** Drop PSR from paper. Focus on detection + pose + activity. |
| OHEM ablation improves mAP > 0.05 | **Detection quick fix works.** Apply permanently and proceed. |
| OHEM ablation improves mAP < 0.02 | **Detection gradient starvation is not the root cause.** The architecture itself (anchor-based, 224px resolution) is the ceiling. Proceed with anchor-free detection in Phase 2. |

**Escalation:** If 3+ heads are structurally broken (ST metrics near baseline), schedule a 1-day architecture review. The shared backbone may be inadequate for the task set. Consider backbone swap or task reduction.

### 7.2 Gate G2: Mid-Phase 2 (Day 7)

**Trigger:** Architecture changes implemented and validated in 10-epoch probes.

**Criteria:**

| Condition | Decision |
|---|---|
| Anchor-free detection probe shows +0.03+ mAP over baseline at epoch 10 | **Commit to anchor-free.** Integrate into main model for Phase 3. |
| Anchor-free detection probe shows no improvement | **Revert to anchor-based.** Add resolution increase (320px) as the alternative path for Phase 3. |
| PSR transition predictor probe shows event-F1 > 0.10 | **Commit to transition predictor.** Integrate into main model. |
| PSR transition predictor probe shows no improvement | **Drop PSR from paper.** The end-to-end formulation is unsolvable at 224px. |
| Activity two-stage training probe shows top-1 > 10% | **Commit to two-stage training.** Integrate into main model. |
| Activity two-stage training shows no improvement | **Drop activity from paper.** Move to "failure dissection" framing. |
| Claude Science special project passes 10-epoch validation | **Integrate into main model.** Add corresponding ablation to the matrix. |
| Claude Science special project fails 10-epoch validation | **Archive.** Document the negative result. No further time investment. |

**Escalation:** If both detection and PSR improvements fail, the paper rests on pose alone. This is acceptable (novel baseline, credible 9 degrees) but weakens the multi-task claim substantially. Reset expectations: the paper becomes "Efficient Multi-Task Assembly Perception: Pose Works, Everything Else is a Lesson."

### 7.3 Gate G3: End of Phase 3 (Day 20)

**Trigger:** All training runs complete. Statistical analysis done.

**Criteria:**

| Condition | Decision |
|---|---|
| MTL/ST ratio >= 0.75 for 2+ heads AND pose MTL/ST >= 0.75 | **Strong paper.** Submit to WACV/AAIML. Lead with efficiency + pose + detection. |
| MTL/ST >= 0.75 for pose only, other heads show non-trivial learning | **Adequate paper.** Submit to workshop or lower-tier venue. Frame as "honest MTL assessment." |
| All heads MTL/ST < 0.50 (only pose survives) | **Weak paper.** Consider: (a) submit to workshop as a negative result paper, (b) defer to next cycle with more data, (c) change the paper's thesis to focus solely on pose and treat MTL as the secondary contribution. |
| Any head unexpectedly beats its ST baseline (MTL/ST > 1.0) | **This is the headline result.** Restructure the paper around this finding. It is the evidence for positive transfer that the MTL literature rarely produces. |

**Paper strength assessment:**
```
Strong = Head pose (0.77) + Detection (0.60) + PSR (0.50) + Activity (0.30)
       = "MTL works for 3/4 heads, here is why the 4th fails"
       = AAIML submission, confident

Adequate = Head pose (0.77) + Detection (0.40) + PSR/Activity (near zero)
         = "Efficient MTL with honest failure analysis"
         = Workshop submission

Weak = Head pose (0.77) only
     = "First MTL baseline for assembly head pose"
     = Rethink scope or defer
```

### 7.4 Gate G4: Mid-Phase 4 (Day 26)

**Trigger:** First complete draft of the paper.

**Criteria:**

| Condition | Decision |
|---|---|
| All claims verified against raw data | Submit |
| Any claim cannot be verified | Add the verification run to the buffer budget (50 GPU-hours reserved). Delay submission by 1-2 days while the verification run completes. |
| Reviewer simulation identifies fatal flaw | Address the flaw if fixable within the 50 GPU-hour buffer. If not fixable, defer to next cycle with the flaw documented. |
| Paper exceeds page limit | Cut without removing the efficiency spine or the core MTL/ST comparison. Figures and supplementary material absorb overflow. |
| Paper is under page limit | Add per-class breakdown analysis, more qualitative results, or additional ablation discussion. |

---

## 8. Fallback Timeline: If Claude Science Findings Are Limited

Claude Science may return findings that are interesting but not actionable, or that confirm what we already know. This section defines the default timeline that does not depend on Claude Science output.

### 8.1 Scenario A: Claude Science Returns Strong Findings

If Claude Science identifies 2+ high-impact methods (score >= 4 on the triage matrix):

- **Phase 1**: Proceed as planned (ST baselines and quick fixes are independent of Claude Science).
- **Phase 2**: Reserve 2 of the 7 days for implementing Claude Science special projects. Reduce PSR transition predictor priority if Claude Science identifies a higher-impact PSR method. Reduce anchor-free detection priority if Claude Science identifies a higher-impact detection method.
- **Phase 3**: Add 2-3 Claude Science-inspired ablations to the ablation matrix.
- **Phase 4**: Frame the paper's contribution around the Claude Science-inspired innovation, if it produces a significant improvement.
- **Timeline: 30 days unchanged.** The strong findings accelerate Phase 2 (clearer priorities) rather than extending it.

### 8.2 Scenario B: Claude Science Returns Moderate Findings

If Claude Science identifies 1 high-impact method or 2-3 moderate findings:

- **Phase 1**: Proceed as planned.
- **Phase 2**: Implement the best finding within the existing 7-day window. If the finding conflicts with pre-planned architecture work, prioritize the finding with higher impact score.
- **Phase 3**: Add 1 Claude Science-inspired ablation alongside the pre-planned matrix.
- **Phase 4**: Frame the paper with the Claude Science finding as a secondary contribution.
- **Timeline: 30 days unchanged.** Moderate findings fit within the existing schedule.

### 8.3 Scenario C: Claude Science Returns Limited Findings (Default Fallback)

If Claude Science confirms our existing approach, identifies only papers we already know, or produces recommendations that are inapplicable to our setup:

This is the **expected outcome** for an already-well-researched problem. Do not treat it as a failure. The default timeline proceeds without Claude Science-specific changes.

**Default timeline (no Claude Science dependency):**

| Day | Activity | GPU allocation |
|---|---|---|
| 1-3 | ST baselines (all 4 heads), OHEM ablation, Mosaic test, activity probe, PSR diagnostics | Both GPUs |
| 4-6 | Anchor-free detection implementation + probe. PSR transition predictor implementation + probe. | RTX 3060 |
| 6-8 | Activity two-stage training integration. Pose geometric head. Integration testing. | RTX 3060 |
| 8-10 | Validation of all architecture changes. Re-run failing probes. Freeze architecture. | Both GPUs |
| 11-13 | MTL main training (5 seeds) | RTX 5060 Ti |
| 12-15 | ST baseline re-runs (if architecture changed) | RTX 3060 |
| 15-18 | Ablation matrix (8-10 ablations, 3 seeds each) | Both GPUs |
| 18-20 | Statistical analysis, per-class breakdown, check all results | CPU only |
| 21-23 | Figure and table generation | CPU only |
| 22-26 | Paper writing (Method first, then Experiments, then Intro/Related) | CPU only |
| 26-28 | Internal review round 1 (claim verification), round 2 (figures), round 3 (narrative) | CPU only |
| 28-30 | Final revisions, formatting, supplementary material | CPU only |
| **30** | **Submit** | |

### 8.4 Scenario D: Catastrophic Failure (All Heads Show Zero Learning in MTL)

If after Phase 1+2, the MTL model shows zero learning on 3+ heads (not just activity and PSR, but detection too):

**Response:** Pivot within 24 hours. Options ranked by feasibility:

1. **Pose-only paper**: Drop all other heads. The paper becomes "First MTL-based egocentric head pose estimation for industrial assembly." Efficiency claim becomes: "Our model adds a 0.2M pose head to a 34.5M backbone at <0.5% parameter overhead." This is a valid submission to workshops and potentially to WACV as a short paper.
2. **Detection + pose paper**: If detection recovers (mAP > 0.20) but activity and PSR remain zero, drop the two failed heads. The paper becomes "Efficient multi-task perception: simultaneous detection and head pose estimation in assembly." Two tasks is still MTL (many MTL papers cover only 2-3 tasks).
3. **Single-task selection**: If MTL genuinely provides no benefit (all heads perform better as ST models), the paper becomes a rigorous empirical study of "When does MTL fail for industrial assembly perception?" This is publishable at a workshop as a negative result.
4. **Data collection**: If the root cause is insufficient data (most likely for activity with 46/75 classes <1%), and no other path works, pause the paper submission and spend 2-4 weeks on: (a) synthetic data generation, (b) active learning to fill tail classes, (c) semi-supervised pretraining on unlabeled IndustReal video.

### 8.5 Decision Authority

| Decision | Who decides | When |
|---|---|---|
| Move from Phase 1 to Phase 2 | Lead author, based on G1 results | Day 3 |
| Implement Claude Science special project | Lead author + advisor sign-off | Day 4-8 |
| Drop a head from the paper | Team consensus | Any gate |
| Swap backbone | Team consensus + compute budget review | Only at Gate G1 or G2 |
| Go to cloud GPU | Any team member (with budget approval) | When on-premise saturated |
| Submit to venue | Team consensus | After G3 results |
| Defer to next cycle | Team consensus + written justification | After G3 or G4 |
| Change paper thesis (e.g., pose-only) | Team consensus | After G3 if conditions warrant |

---

## Appendix: Quick-Reference Checklists

### A.1 Pre-Flight Checklist (Before Each Phase)

- [ ] All Phase N experiments logged in experiment tracking (seeds.csv, config logged).
- [ ] All baseline results from Phase N-1 available and analyzed.
- [ ] Decision gate criteria from Phase N-1 evaluated.
- [ ] GPU health check: `nvidia-smi` shows expected power, temperature, memory.
- [ ] No other GPU workloads running (check with `nvidia-smi`).
- [ ] Disk space available for checkpoints (minimum 50 GB free).
- [ ] Ram cache warm (HotSpot cache populated for IndustReal dataset).
- [ ] Git branch clean, all previous changes committed.
- [ ] Overfit probe passed for any new architecture change.
- [ ] Rollback flag verified (disable flag reproduces baseline).

### A.2 Checkpoint Retention Policy

| Checkpoint type | Retention | Location |
|---|---|---|
| Best by composite metric | Permanent | `checkpoints/{exp_name}/best_composite.pt` |
| Best per head | Permanent | `checkpoints/{exp_name}/best_det.pt`, etc. |
| Last 5 periodic | Permanent (for SWA) | `checkpoints/{exp_name}/epoch_{N}.pt` |
| Final epoch | Permanent | `checkpoints/{exp_name}/final.pt` |
| Pre-architecture-change baseline | Permanent (comparison) | `checkpoints/baseline_{date}/` |
| Failed/deprecated runs | Retain logs only | `logs/failed/{date}/` (logs only, delete checkpoints) |

### A.3 Emergency Response Times

| Failure mode | Detection | Response | Max downtime |
|---|---|---|---|
| GPU hang/crash | nvidia-smi, watchdog timeout | Kill process, restart from last checkpoint | 30 min |
| NaN loss | Loss NaN check in training loop | Revert to previous checkpoint, reduce LR, disable AMP | 1 hour |
| OOM | torch.cuda.OutOfMemoryError | Reduce batch size by 1, clear cache, retry | 15 min |
| Disk full | df -h, training save failure | Remove old checkpoints, notify team | 1 hour |
| Power outage | Machine unreachable | Auto-restart training from last checkpoint on boot | Recovery dependent |

### A.4 Weekly Review Cadence

| Day | What to review | Participants |
|---|---|---|
| Day 3 (end Phase 1) | ST baselines, quick fix results, G1 decisions | Lead + advisor |
| Day 7 (mid Phase 2) | Architecture change validation, G2 decisions | Lead + advisor |
| Day 10 (end Phase 2) | Architecture freeze, Phase 2 summary | Lead + advisor |
| Day 15 (mid Phase 3) | MTL training progress, early ablation results | Lead |
| Day 20 (end Phase 3) | Full results, statistical analysis, G3 decisions | Lead + advisor |
| Day 24 (mid Phase 4) | Draft paper quality, narrative check | Lead + advisor |
| Day 28 (pre-submit) | Final claim verification, G4 decisions | Lead + advisor + external reviewer |

---

**Document version:** 1.0
**Last updated:** 2026-07-11
**Total words:** ~3,200
**Next review:** Immediately after Claude Science findings are received and triaged. Update Phase 2 and Phase 3 sections with specific Claude Science-inspired items.
