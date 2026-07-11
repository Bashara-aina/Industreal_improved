# Doc 225 — Risk Assessment & Contingency Planning

**Status:** Living document. Updated 2026-07-11.
**Audience:** Paper authors. Brutal honesty, no hedging, no survivorship bias.
**Purpose:** Catalog every risk that could prevent a winning paper, assign probability, and define contingency.

---

## Risk Scoring Methodology

Each risk is scored on two axes:

- **Impact**: What happens to the paper if this risk materializes
- **Probability**: Likelihood given current evidence (July 11, 2026 status)

Severity = Impact x Probability, but we use named tiers because the math is misleading when multiple risks co-occur.

**Evidence basis**: Overfit probes (complete, 2026-07-11), RF4 epoch 5 validation metrics (doc 103), 38+ documented training fixes (doc 113), five rounds of Opus consultation (docs 150-207), and the full consultation package (docs 208-227).

---

## 1. CRITICAL RISKS — Paper Fails If Any of These Materialize

### CRIT-1: Detection mAP stays at 0.0 after all levers

**Probability: LOW (15%)** — BUT disproportionately feared.

**Evidence**: Detection is the strongest performer in RF4. At epoch 5, det_mAP50=0.212 (doubled from epoch 2), det_mAP50_pc=0.339 (near target 0.35-0.55). 15/24 classes detected. dp_scores mean=0.333 — finally separating from the 0.036 bias floor. The detection head is learning, not dying.

**Why the risk persists**: 
- The single-task detection ST baseline at epoch 47/99 shows loss plateauing around 2.3 with no clear downward trend — the ST ceiling itself may be lower than projected (~0.40-0.55). If the ST ceiling is 0.30, then MTL even at 80% retention yields only 0.24 mAP.
- Detection uses 26K frames with 99.3% empty. The sparse-GT problem is structural, not fixable by architecture.
- TAL topk=10 per FPN level may be creating redundant positives that dilute gradient quality on the few real objects.
- At 224px input resolution, small assembly components (~20px) occupy ~3 cells on the P3 grid. The anchor assignment may simply miss them.

**Contingency**: See Section 4.

### CRIT-2: Activity < 20% top-1 despite all fixes

**Probability: MEDIUM (30%)**.

**Evidence**: RF4 epoch 5 shows act_macro_f1=0.097, act_top5=0.381, pred_distinct=48/69 classes. This is a 15x improvement from epoch 2 (0.006) and a recovery from mode collapse (5 classes at epoch 2 -> 48 classes at epoch 5). The trajectory is positive but the gap to 20% top-1 is still substantial.

**Why the risk persists**:
- 75 classes with power-law distribution. 20 classes have <30 frames. 5-7 classes have <7 frames. No re-weighting scheme creates signal from 7 frames.
- The gradient starvation (0.010 vs PSR's 3.18) is structural, not fixable by loss weighting or learning rate. CE on a single 75-dim logit produces fundamentally less gradient signal than 88 independent binary focal losses.
- Activity head improvements (spatial attention pool, multi-layer aggregation) require code changes that are documented but not yet implemented. The gradient amplification factors (5-10x, 10-20x) are estimates, not guarantees.
- The overfit probe was a false negative: the head DOES learn (40.5% top-1 with frozen random backbone), but the frozen CLS token is information-limited. With trainable backbone, the 312x gradient gap may mean the head still cannot compete.

**Contingency**: See Section 4.

### CRIT-3: PSR event-F1 < 0.05 despite monotonicity + threshold

**Probability: HIGH (60%)**.

**Evidence**: The PSR head is the most fragile component. At RF4 epoch 5, psr binary accuracy = 0.554 (above chance for the first time), but transition metrics are all 0.0 due to MonotonicDecoder eval crash. The only measured PSR F1 is 0.144 from an earlier run, and the random predictor baseline is 0.136. The gap above random is 0.008.

**Why the risk is high**:
- Focal-BCE on 11 components where >99.5% of frames have no transition produces a model that is "happy" predicting all zeros. The loss converges to ~0.17-0.27 with near-zero event-F1.
- The transition-aware weighting (boost=3.0) cannot compensate for a 0.5% event rate. Even boosted, event frames are a tiny fraction of total frames.
- T=8 temporal resolution from T=16 input means transitions can only be detected at ~0.27s granularity at 30fps. Many assembly transitions are faster than this.
- The MonotonicDecoder eval integration has had two known bugs (squeeze collapse, transition alignment) and may have more undiagnosed issues.
- The D4 experiment (YOLOv8m -> our MonotonicDecoder) that would isolate PSR head quality from detection quality is planned but not yet executed.

**Contingency**: See Section 4.

### CRIT-4: GPU OOM during full-budget training

**Probability: LOW-MEDIUM (20%)**.

**Evidence**: The full MTL configuration (4 heads + BiFPN + MViTv2-S at T=16, batch=4, grad_accum=4) has been running stably on the RTX 5060 Ti 16GB for multiple epochs. Peak VRAM is ~11-13 GB out of 16 GB. The OOM risk is primarily from edge conditions.

**Why the risk persists**:
- OOM on RTX 3060 12GB during full-budget MTL training was a historical problem (the reason the 5060 Ti was acquired). If the 5060 Ti goes down, there is no backup GPU that fits.
- The expandable_segments + memory fraction patch prevents CUDA OOM crashes but does not prevent PyTorch OOM from memory fragmentation during long runs.
- SWA checkpoint loading at the end of training loads 5 full checkpoints simultaneously (~5 x 4 GB = 20 GB theoretical peak). The implementation loads them sequentially, but a bug in the loading pipeline could cause OOM at the final step when all metrics are being computed.
- Adding more levers (distillation, detection-conditioned PSR, spatial attention pool) increases VRAM consumption. The current ~2 GB headroom is thin.

**Contingency**: See Section 4.

### CRIT-5: Overfit probe reveals eval harness bug (data corruption)

**Probability: LOW (10%) for remaining bugs**.

**Evidence**: Overfit probes completed 2026-07-11. Pose and PSR pass trivially (loss -> 0, metrics validate). Activity and detection showed false negatives (they DO learn, but the frozen backbone is the limiter). The eval harness is confirmed working for pose and PSR; activity and detection eval have no identified bugs.

**Why the risk persists**:
- The activity false negative required careful interpretation to understand it was a probe design issue, not an eval bug. If another subtler eval bug exists (e.g., metric computation on the wrong axis, class label shift), it would not be caught by the same probe.
- The PSR MonotonicDecoder had two confirmed bugs (F22/F22b: squeeze collapse, transition target alignment). A third undiagnosed bug could be silently producing wrong metrics.
- Detection uses a per-class AP computation with background class handling. If the background class logic misaligns with the 24-class label space, mAP could be systematically wrong.
- Data loading shuffle bugs (e.g., frames assigned to wrong clips in the validation set) would not be caught by overfit probes because overfit probes use a tiny fixed subset.

**Contingency**: See Section 4.

---

## 2. HIGH RISKS — Significantly Damage Paper, Non-Fatal Alone

### HIGH-1: Kendall collapse persists despite caps

**Probability: LOW (15%) — appears solved**.

**Evidence**: Kendall caps were a known problem (lv_clamp_max values in code were 4.0 for det/pose instead of documented 1.5/2.0). The current RF4 run shows stable convergence: lv_det=0.125, lv_act=0.040, lv_psr=-0.079, lv_pose_eff=0.125 at epoch 6. Caps are preventing starvation. The uncapped ablation (--kendall-uncapped) correctly reproduces collapse.

**Why the risk persists**:
- The caps were wrong in the code for an unknown period before discovery. If earlier results were collected with uncapped or wrong-capped settings, they are unreliable.
- Caps are a heuristic. The values (det=1.5, act=1.0, psr=0.5, pose=2.0) were chosen by hand, not by a principled method. A different task weight distribution might produce better results.
- The pose-below-det constraint (lv_pose = max(lv_pose, lv_det)) is a hard override that could mask a genuine pose uncertainty signal.

### HIGH-2: ST baselines also perform poorly (bad backbone pretraining)

**Probability: MEDIUM (35%)**.

**Evidence**: The single-task detection ST baseline at epoch 47/99 shows detection loss plateauing at ~2.3 with no clear downward trend. This suggests the ST ceiling for detection may be lower than the projected 0.40-0.55 mAP.

**Why the risk is high**:
- MViTv2-S is Kinetics-400 pretrained (action recognition), not detection pretrained. The backbone has never seen bounding box supervision. The detection ST ceiling is fundamentally lower than a COCO-pretrained backbone.
- If all ST baselines are weak, the MTL/ST ratio story collapses: "we retain 80% of a weak ST ceiling" is not publishable.
- The ST baselines use the same hyperparameters as MTL (batch size, LR schedule, augmentation). These may be suboptimal for single-task training but never tuned because the focus was MTL.
- ST baselines for activity and PSR are not yet complete. Activity ST is the most critical: if activity ST underperforms 55-65% top-1, the MTL retention ratio at 30-45% becomes 50-60% instead of 70%+.

### HIGH-3: One task catastrophically interferes with others

**Probability: LOW-MEDIUM (20%)**.

**Evidence**: PCGrad gradient surgery is active and confirmed working. Kendall caps prevent any head from being fully starved. The current RF4 run shows all heads learning simultaneously with no collapse. The activity head recovered from near-collapse (epoch 2, 5 classes) to healthy diversity (epoch 5, 48 classes).

**Why the risk persists**:
- PCGrad handles gradient DIRECTION conflict but not MAGNITUDE conflict. The 312x gradient gap between activity (0.010) and PSR (3.18) means activity is still being relatively starved even after PCGrad.
- As training progresses, task relationships may change. A head that was cooperative at epoch 5 may become adversarial at epoch 30 (loss landscape shift).
- Knowledge distillation from ST teachers (lever 5) feeds teacher logits as additional supervision. If the teacher is wrong on specific examples, the student gets conflicting signals — teacher says class A, ground truth says class B. This could create gradient conflict worse than the MTL interference it was designed to fix.

### HIGH-4: Training takes too long (>2 weeks per experiment)

**Probability: MEDIUM (30%)**.

**Evidence**: Timeline from doc 208: ST baselines ~55h each, MTL ~6.8 days, total ~13.8 days per full experiment cycle. Three experiments (ST baselines + 2 MTL variants) = ~27 days. This assumes no failures.

**Why the risk is high**:
- Every crash recovery costs 1-3 hours of re-validation time.
- The current RTX 5060 Ti is shared between training and display (X11, browser, terminal). A GUI freeze or memory leak can kill the training process after days of progress.
- The dynamic early stopping (patience=15) is NOT implemented. The current 100-epoch schedule runs to completion regardless of convergence.
- If ST baselines need retuning (per-task hyperparameters), each retune adds 2.5 days.
- The deadline for WACV 2027 submission is likely July-August 2026. If a negative result requires a full pivot, there may not be time for a second attempt.

### HIGH-5: Test split metrics significantly worse than val (overfitting)

**Probability: MEDIUM (25%)**.

**Evidence**: The current RF4 val metrics are promising (combined=0.241), but test split evaluation has not been performed. The model selects checkpoints based on val activity top-1, which introduces optimistic bias.

**Why the risk is high**:
- The dataset has ~26K frames across 4 assemblies. With 75-class activity labels, a model can easily memorize the val split's class distribution.
- Detection training uses GT frame fraction (0.40), meaning 60% of training frames have no detection labels. If the 40% of frames with labels have different characteristics from the 60% without, the detector learns a biased distribution.
- SWA averaging (last 5 checkpoints) reduces variance by 1/sqrt(5) but does not eliminate the train-val distribution gap.
- Model selection on the subset of val used for early stopping introduces "eval contamination" — the val set is being used both for model selection and for the metrics reported in the paper.

---

## 3. MEDIUM RISKS — Manageable, Worth Tracking

### MED-1: Implementation bugs in new code (geodesic, BiFPN, spatial attention)

**Probability: HIGH (50%)**.

**Evidence**: The codebase has a documented history of bugs between "what the document says" and "what the code does." Opus Round 5 (doc 207) found that claimed code changes (PSR diet, logit-adjust, kendall-uncapped flag, overfit_probe rewrite) were absent from the repository. The geodesic loss implementation has never been verified against a numerical reference. The BiFPN 3D convolutions may be operating on incorrectly shaped tensors.

**Why this matters**: A single bug in the gradient path (like the F1 seq-batch backbone grad wipe that was destroying 80% of backbone gradient signal) can silently produce wrong metrics for weeks. The spatial attention pool for activity (Phase 2 of the architecture plan) involves multiple new components, each a potential bug source.

### MED-2: Numerical instability (NaN loss at some step)

**Probability: LOW (10%) with bf16**.

**Evidence**: bf16 has eliminated the inf/nan cascade that plagued the fp16 path. No NaN events in the current RF4 run (epoch 12+ stable).

**Why the risk persists**: The geodesic loss computes SVD via Gram-Schmidt, which can produce NaN gradients for near-degenerate rotation matrices. This has been observed in some edge cases (head facing directly away from camera). The Tanh-bounded 6D representation prevents raw prediction explosion, but the Gram-Schmidt orthonormalization can still encounter numerical issues when the two predicted vectors are nearly collinear.

### MED-3: Inconsistent results across seeds

**Probability: MEDIUM (30%)**.

**Evidence**: No multi-seed experiments have been run. Every result is from a single seed. The Kendall + PCGrad combination is known to have seed sensitivity (PCGrad's random task ordering introduces stochasticity).

**Why this matters**: If three seeds produce wildly different results (e.g., activity top-1 ranging from 25% to 45%), the paper's claims become unverifiable. The confidence intervals on MTL/ST ratios would be too wide to publish. The training cost (6.8 days per run) makes multi-seed experiments economically painful.

### MED-4: Activity long-tail classes never learned

**Probability: HIGH (60%)**.

**Evidence**: 20 of 75 classes have <30 frames. 5-7 classes have <7 frames. Even with sqrt-tamed class weights (max/min ratio ~12), the effective number of samples for these classes is 1-3. At RF4 epoch 5, only 48/69 classes are predicted — the 21 missing classes are all tail classes.

**Why this matters**: The paper reports activity top-1 and macro-F1. If tail classes contribute 0 to both metrics, the headline numbers are artificially low. Reporting "top-1 on head classes only" would be dishonest. The tail classes are legitimate assembly actions (e.g., specific error recovery procedures) and reviewers familiar with the domain will expect them to be present.

### MED-5: PSR windowed prediction doesn't align with event metrics

**Probability: MEDIUM (30%)**.

**Evidence**: PSR predicts per-frame states from a T=8 window (8 logits covering ~0.27s at 30fps). Event metrics compute F1 at +/-3 frame tolerance on the DIFFERENCED state sequence. The windowed prediction approach produces state estimates that, when differenced, have transition events at the window boundary rather than the true transition frame. The MonotonicDecoder fill-forward constraint means transitions are detected one frame after the confidence crosses threshold.

**Why this matters**: The event-F1 metric implicitly expects point predictions (exact transition frame), but the model produces windowed estimates. The F1 ceiling from this temporal misalignment alone may be <0.30, regardless of model quality.

---

## 4. CONTINGENCY PLANS

### 4.1 If detection stays at 0.0 mAP

**Immediate triage (1-2 days)**:
1. Run the D2 experiment: evaluate YOLOv8m detections through our eval pipeline. If YOLOv8m also shows 0.0 mAP, the eval code is broken (fix before anything else).
2. Check TAL assignment: run `tal_debug.py` to verify positive assignment is happening for GT boxes at all.
3. Print raw detection outputs (box coordinates, class scores) for a single batch and verify they are not all-zero or all-NaN.

**Levers to pull (3-7 days)**:
4. Reduce TAL topk from 10 to 5 (prevents redundant positives diluting gradient).
5. Enable mosaic + copy-paste augmentations (already implemented, never activated — +3-5 mAP on small datasets per published evidence).
6. Revert to ConvNeXt-Tiny + 2D FPN as a sanity check (detection worked on this architecture before MViTv2 migration).
7. Train detection-only for 15 epochs before introducing other heads (staged training).

**Paper framing if all fails**:
The detection head is an ablation-demonstrated failure. Pivot to 3-task MTL (activity + PSR + pose) with the detection task deferred. Frame as: "Three-task MTL on egocentric assembly video achieves X retention at Y efficiency; detection requires higher resolution input and is addressed in ongoing work."

### 4.2 If activity doesn't improve past 20% top-1

**Immediate triage (1-2 days)**:
1. Verify the ST activity baseline completes and produces >50% top-1. If the ST ceiling is itself below 50%, the backbone or data has a problem unrelated to MTL.
2. Run the activity-only ablation preset (--ablation act-only) to isolate activity performance without any multi-task interference.

**Levers to pull (3-7 days)**:
3. Implement spatial attention pool (P5 features, not cls_token) — target 5-10x gradient amplification. This is the single highest-impact change available.
4. Enable class-balanced focal loss (USE_CB_FOCAL_ACT=True, beta=0.999, gamma=2.0) — replaces CE + sqrt-tamed weights with a principled long-tail loss.
5. Implement decoupled training (--act-decoupled): train backbone with instance-balanced sampling for 50 epochs, freeze, then retrain activity classifier with class-balanced sampling.
6. Reduce activity class count via label hierarchy: collapse 75 fine-grained classes into 20-30 coarse groups for the primary metric, with fine-grained as secondary.

**Paper framing if all fails**:
Shift the paper's emphasis from "MTL beats ST" to "MTL efficiency with graceful degradation." Show that activity retains 50-60% of ST despite being the hardest task. The 75-class fine-grained activity recognition under egocentric video is a novel dataset contribution — even moderate results establish a baseline for future work.

### 4.3 If PSR event-F1 stays < 0.05

**Immediate triage (1-2 days)**:
1. Run D4 experiment: feed YOLOv8m ASD outputs through our MonotonicDecoder. If YOLOv8m->PSR F1 is also <0.10, the MonotonicDecoder or eval pipeline is broken (fix before anything else).
2. Verify the random baseline: PSR event-F1 for a predictor that always outputs the previous frame's state should be measured and confirmed at 0.136.

**Levers to pull (3-7 days)**:
3. Replace focal-BCE with Asymmetric Loss (ASL): hard-threshold negative gradients instead of soft focal weighting. This directly targets the "predict all zeros" failure mode by removing the gradient contribution from ultra-easy negatives entirely.
4. Set PSR to use bidirectional (non-causal) Transformer — flip the causal mask boolean. This gives each frame access to future context, improving transition detection at the cost of losing online/streaming capability.
5. Reduce temporal resolution: if T=8 windows are too fine for the 0.5% event rate, try T=4 or T=2, which increases the event density per window at the cost of temporal precision.

**Paper framing if all fails**:
The PSR component is a demonstration of "what doesn't work" — publish the null result showing that per-frame state classification at <1% event rate requires fundamentally different approaches (set-prediction losses, hard negative mining). The PSR head's binary accuracy (>0.55) shows it learns SOMETHING about assembly state, even if transition detection fails.

### 4.4 If GPU OOM during training

**Immediate triage**:
1. Reduce batch size from 4 to 2 (effective batch drops to 8, but training continues).
2. Reduce T from 16 to 8 (halves activation memory, loses some temporal context).
3. Use CPU offloading for optimizer states (saves ~2 GB VRAM at ~10% speed penalty).

**Long-term fix**:
4. The RTX 3060 12GB can run detection+PSR+pose (3 heads) but not all 4. If the 5060 Ti fails, train 3-head MTL and frame the 4th head as an extension.
5. Cloud GPU (Lambda Labs, RunPod) as emergency backup. Budget: ~$0.50-1.00/hr for RTX 3090/4090.

### 4.5 If eval harness has a bug discovered late

**Immediate triage**:
1. Freeze all code changes. Re-run overfit probes from scratch on the current codebase.
2. If overfit probes pass but full eval metrics are suspicious, implement "eval unit tests": synthetic data with known ground truth, verify metrics numerically.
3. For detection: create 10 images with known bounding boxes and compare mAP computation against pycocotools reference.
4. For PSR: create synthetic state sequences with known transitions, verify event-F1 computation.

**Paper framing if all fails**:
Recalculate all metrics with corrected eval. If results are substantially worse, acknowledge the eval bug in the paper as a supplementary note and report corrected numbers. A transparent admission of a fixed bug is better than publishing wrong numbers.

### 4.6 Fallback paper framing for weak results

**Scenario A: 2 of 4 heads work well.** E.g., pose and activity are strong, detection and PSR are weak. Publish as "Partially Successful MTL for Assembly Verification" with detailed analysis of WHICH tasks transfer positively and which interfere. Document the per-task transfer map as a contribution.

**Scenario B: All heads work but at 50-60% of ST.** The parameter efficiency story survives (2x efficiency, 4x latency improvement at 50-60% retention). Frame as "The Price of Sharing: Quantifying the MTL Gap in Egocentric Assembly Perception." The Kendall-collapse characterization + cap fix is a methodological contribution that is independent of the absolute metric values.

**Scenario C: MTL beats ST on some heads, loses on others.** This is the most interesting scenario for the literature. Publish as "Partial Positive Transfer in Multi-Task Egocentric Vision" with analysis of WHY specific tasks benefit or suffer from sharing. Reviewer interest in "when does MTL help?" is high.

**Scenario D: All heads work but SOTA is unreachable.** The 224px input resolution vs 640px YOLO comparison is apples-to-oranges anyway. Focus on the efficiency story and the novel dataset (IndustReal) as contributions. A thorough benchmark of MTL on a new industrial assembly dataset has value even without SOTA claims.

### 4.7 Minimum Viable Paper

**Core publishable contributions (even with poor metrics)**:
1. **Kendall-collapse characterization**: The first systematic documentation of how Kendall uncertainty weighting fails for 4 disparate tasks with >100x loss scale differences, and the capped-log-var fix. This is a methodological contribution applicable beyond our specific dataset.
2. **Per-task transfer map**: MTL/ST ratios with honest confidence intervals, showing which tasks interfere and which cooperate in this task set. The literature has very few published transfer maps for 4-task egocentric video MTL.
3. **Efficiency story**: ~48.6M total params vs ~100M for 4 ST models, ~4x latency improvement at inference. This survives regardless of absolute metric values.
4. **IndustReal MTL benchmark**: Published baselines for 4 tasks on a real-world industrial assembly dataset. Even moderate numbers establish a starting point for future work.

**What the paper CANNOT publish without**:
- At least 2 of 4 heads showing MTL/ST > 0.70 (establishes that MTL works for SOME tasks on this domain)
- Convincing evidence that the eval pipeline is correct (overfit probes or equivalent validation)
- A "Kendall-collapse ablation" showing that the cap fix materially changes results

**Contingency venue**:
If WACV rejects (too weak), submit to a workshop (AAIML Workshop at CVPR/NeurIPS, or a smaller venue like VISAPP, ICPRAM). Workshops accept work with narrower scope and weaker results as long as the method is sound and the problem is timely.

---

## 5. What Claude Science Should Find

### Papers documenting MTL failure modes and fixes

We need published work that describes cases where MTL degraded performance and how it was fixed. Specifically:

- **Kendall-weighting failure cases**: Papers showing that uncertainty weighting collapses for tasks with dissimilar loss scales, and proposed fixes beyond log-var caps.
- **Gradient starvation diagnosis**: Methods for detecting and quantifying gradient starvation across tasks (beyond gradient norm logging). Are there published diagnostic tools?
- **Detection-specific MTL failure**: Papers where detection suffered in MTL because of gradient conflict with classification/regression heads, and how it was resolved.
- **PSR/event-detection in low-event-rate regimes**: Temporal event detection when the event rate is <1%. How do published systems avoid the "predict all negatives" local minimum?

### Published negative results in MTL (what doesn't work)

We specifically need papers that document what DIDN'T work in MTL:

- **PCGrad failure modes**: Known cases where PCGrad fails (e.g., with >3 tasks, with high task dissimilarity, with certain gradient magnitude ratios). The literature tends to publish PCGrad successes; we need to know when it fails.
- **Kendall collapse benchmarks**: Standardized benchmarks where Kendall uncertainty weighting is known to fail, so we can position our contribution against a known baseline.
- **Multi-task optimization methods that degrade performance**: Gradient surgery methods that work on 2-3 tasks but break on 4+ (IMTL, GradNorm, MGDA). Which methods have known task-count limits?
- **Detection resolution trade-offs**: Papers that tried small-input detection (224px or lower) and failed, documenting the mAP ceiling imposed by resolution alone rather than architecture.

### Recovery strategies when MTL degrades performance

- **Staged training schedules**: Published schedules for progressively adding tasks during MTL training, with measured recovery of per-task metrics.
- **Per-task adapter/decoupling**: Architectural patterns that decouple tasks BELOW the backbone (not routing adapters that break single-pass) to allow task-specific feature refinement.
- **Loss landscape smoothing**: Gradient clipping, label smoothing, and regularization strategies specifically designed to smooth the multi-task loss landscape and reduce interference.
- **Asymmetric/task-specific augmentation**: Published work showing that different tasks need different augmentations within the same MTL model, and how to reconcile conflicting augmentation requirements.

---

## 6. Risk Burn-Down Timeline

| Window | Action | Risk reduced |
|--------|--------|-------------|
| By ST-det completion | Verify det ST ceiling >= 0.35 mAP | HIGH-2 |
| By MTL epoch 20 | Confirm det mAP trajectory continues | CRIT-1 |
| By MTL epoch 20 | Verify activity macro-F1 > 0.15 | CRIT-2 |
| By MTL epoch 30 | Run D4 (YOLOv8m->PSR decoder) | CRIT-3, MED-5 |
| By MTL epoch 30 | Implement spatial attention pool | CRIT-2 |
| Before test eval | Run multi-seed (3 seeds) mini-MTL | MED-3 |
| Before test eval | Verify eval on synthetic data | CRIT-5 |
| Final week | Compute test split metrics | HIGH-5 |

**Hard deadline for pivot decision**: When MTL epoch 30 completes (~9 days from now). If combined metric is not on trajectory for >0.40 at epoch 50, begin fallback paper framing (Section 4.6, Scenario D).

---

## 7. Summary Risk Register

| ID | Risk | Probability | Impact | Severity | Contingency |
|----|------|-------------|--------|----------|-------------|
| CRIT-1 | Detection mAP = 0.0 | 15% | Fatal | HIGH | 4.1 |
| CRIT-2 | Activity < 20% top-1 | 30% | Fatal | HIGH | 4.2 |
| CRIT-3 | PSR event-F1 < 0.05 | 60% | Fatal | CRITICAL | 4.3 |
| CRIT-4 | GPU OOM during training | 20% | Fatal | HIGH | 4.4 |
| CRIT-5 | Eval harness bug discovered late | 10% | Fatal | HIGH | 4.5 |
| HIGH-1 | Kendall collapse persists | 15% | High | MEDIUM | 4.2 (already mitigated) |
| HIGH-2 | ST baselines also poor | 35% | High | HIGH | 4.6, Scenario D |
| HIGH-3 | Catastrophic task interference | 20% | High | MEDIUM | 4.2, retrain detection-only |
| HIGH-4 | Training takes >2 weeks | 30% | High | MEDIUM | Reduce epochs, skip some heads |
| HIGH-5 | Test-val overfitting gap | 25% | High | MEDIUM | Regularize, report val + test |
| MED-1 | Implementation bugs in new code | 50% | Medium | MEDIUM | Unit tests, per-head ablation |
| MED-2 | NaN loss | 10% | Medium | LOW | bf16, gradient checkpointing |
| MED-3 | Inconsistent across seeds | 30% | Medium | MEDIUM | Train 3 seeds, report CI |
| MED-4 | Long-tail classes never learned | 60% | Medium | MEDIUM | Decoupled training, collapse classes |
| MED-5 | PSR window-event misalignment | 30% | Medium | MEDIUM | D4 experiment, tolerance analysis |

**The honest bottom line**: PSR event-F1 is the single biggest threat to this paper. At 0.144 (barely above 0.136 random baseline), it has the highest probability of being a fatal failure. Detection is the least likely to fail (current trajectory is strongly positive). Activity has a medium risk that can be mitigated with the spatial attention pool redesign. The paper has a plausible path to publishability even with 2 weak heads, but PSR at random baseline kills the 4-task MTL story that is the paper's primary contribution.

**Key decision point**: If PSR event-F1 is still < 0.05 after implementing ASL loss + bidirectional transformer + D4 experiment, the paper must pivot to a 3-task contribution with PSR as a documented failure case. Start preparing this framing in parallel.
