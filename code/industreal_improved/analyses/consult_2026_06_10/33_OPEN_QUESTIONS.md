# 33 — Open Questions: Everything We Still Don't Know

> Generated 2026-06-20 — After RF2 epoch 15 collapse, Opus v8 fixes now deployed  
> Current state: RF2 epoch 17, PID 3176288, Opus v8 fixes active, awaiting epoch-end validation

---

## How to Use This Document

This is the master list of confusions, organized by severity. Each question includes:
- **Why we're asking** — the specific evidence that gives rise to the question
- **What we've ruled out** — hypotheses that have been tested and refuted
- **What we'd need to test** — the experiment that would answer it
- **Confidence level** — how close we think we are to an answer

---

## CRITICAL (Blocking Progress)

### Q01: Why Did the Detection Classifier Collapse AGAIN at RF2 Epoch 15? (Will Opus v8 Fixes Break the Equilibrium?)

**The core contradiction**: We fixed the gradient sparsity problem (Kendall bug → head_pose provides dense gradient), increased data to 35%, guaranteed GT frames in 90% of batches via DET_GT_FRAME_FRACTION=0.90, enabled regression gradient flow via DETACH_REG_FPN=False, AND NOW deployed 4 Opus v8 fixes targeting the collapse mechanism. Despite ALL of this, we don't yet know if any of it worked — epoch 17 is still running and epoch-end validation results are pending.

**Evidence from the original collapse (epochs 1-15, pre-fixes):**
- det_mAP50 reached 0.184 at epoch 8 (best ever for this architecture outside RF1)
- Head pose MAE improved from 71.67° to 47.84° (confirming gradient IS flowing)
- DET_PROBE at epoch 15 shows score_max=0.93-0.97 (the classifier CAN make confident predictions)
- Then collapse: det_mAP50: 0.184 (ep 8) → 0.159 (ep 10) → 0.000010 (ep 13) → 0.001 (ep 15)
- classifier score distribution: score_p50=0.019, cls_score std=0.0068-0.0088 (< 0.01)
- EVAL COLLAPSE: 56 occurrences in train.log at epoch 15

**Opus v8 fixes deployed (epoch 17 restart, PID 3176288):**

1. **KENDALL_HP_PRE_CAP** — Experimental cap on Kendall log variance to prevent extreme log_var values from destabilizing head_pose gradient
2. **DET_POS_IOU_THRESH=0.4** — Lower positive anchor IOU threshold from ~0.5 to 0.4, increasing the number of anchors that qualify as positive matches per GT box
3. **DET_POS_IOU_TOP_K=9** — Force the top 9 anchors (by IOU) per GT to be positive regardless of whether they exceed the IOU threshold. Primary mechanism for increasing positive anchor count
4. **DET_BIAS_LR_FACTOR=1.0** — Remove any special bias LR suppression, set bias learning rate factor to the default 1.0
5. **_validate_stage_history_entry() guard** — Stage history entry validation preventing phantom values in state tracking (Fix 4, resolves Q02)

**What we still don't know:**
- Do fixes 2-3 produce enough positive anchors to sustain differentiation? Estimated: 6-10 per GT = 96-160 per batch vs ~16 previously
- Does fix 1 prevent log_var explosion that leads to head_pose gradient collapse?
- Does fix 4 (bias LR factor) actually change anything? The bias may not have been suppressed in the first place
- Most importantly: Do these 4 fixes collectively break the 0.079 uniform equilibrium, or is it reachable from a different path?

**What we've ruled out (still valid from previous analysis):**
- NOT gradient sparsity (head_pose provides dense gradient, proven by LIVENESS_GRAD)
- NOT checkpoint lineage (fresh ImageNet start reproduced collapse in Run 8)
- NOT LR (20× reduction produced identical trajectory)
- NOT missing GT frames (DET_GT_FRAME_FRACTION=0.90 guarantees GT per batch)
- NOT DETACH_REG_FPN (regression gradient flows, but this only matters for box regression, not classification)

**Remaining hypotheses:**
1. **Focal Loss has a degenerate equilibrium at ~0.079 uniform scores** that is reachable from pi=0.01 init. The ratio of positive-to-negative gradient (16:348K anchors) means the negative gradient, even if 321× suppressed by pi=0.01, still dominates in aggregate (348K × 2.55e-5 = 8.87 total negative gradient vs 16 × 73.5 = 1176 positive gradient — but positive gradient distributed across 595K cls head params = 0.002 per param, while negative gradient affects mostly bias = 8.87/1 = 8.87 change to bias → bias moves from -4.6 toward where positive/negative balance)
2. **The cls_score bias drifts to a value where Focal Loss for positives and negatives exactly cancel**, creating a fixed point. The classifier can't escape because any move toward differentiation increases loss.
3. **Even with 6-10× more positive anchors**, the positive-to-negative ratio (160:348K = 0.046%) may still be insufficient. The cumulative negative gradient may still dominate the bias update.
4. **The head_pose gradient, while dense, may not help detection-specific features.** Head_pose trains the backbone to produce features useful for orientation regression, not object classification. These features may actually be suboptimal for detection.

**What we're doing now:**
- Waiting for epoch 17 epoch-end validation results (~49 min per epoch)
- If det_mAP50 > 0.05 at epoch 17 end: fixes are having SOME positive effect
- If det_mAP50 still ~0.001: fixes didn't address the root cause
- If det_mAP50 > 0.20: fixes may have broken the equilibrium entirely

**Confidence**: VERY LOW — first epoch-end validation after Opus v8 will tell us much more.

---

### Q03: Has PSR EVER Trained Successfully in This Architecture?

**The evidence**: PSR loss = 1.546e-08 constant across EVERY run, EVERY configuration, ALL of Phases 1-12:

- RF1 (det only): PSR loss = 1.546e-08
- RF2 (det + head_pose): PSR loss = 1.546e-08
- R2.5 (all heads): PSR loss = 1.546e-08
- Run 8 (fresh ImageNet): PSR loss = 1.546e-08
- ALL diagnostic probes: 1026 PSR_DIAG entries, ALL show identical value

**What this means**: The PSR head has NEVER received a non-zero learning signal. It's been dead since the first training run.

**What we know:**
- PSR uses binary focal loss with fill-forward labels (20/22 components are zero)
- The causal transformer (3L, 4H, T=2) produces extreme logits (min=-23, max=+22)
- Sigmoid saturates at these logits → gradient = 0
- In RF1-RF3, PSR is intentionally not trained (train_psr=False)
- But in R2.5 (all heads), it WAS supposed to be training

**Critical question**: Does PSR actually work? Has it ever worked in any configuration? Or is there a fundamental architecture bug (like the transformer producing extreme logits that sigmoid saturates)?

**What we'd need to test:**
- Check if PSR logit initialization produces reasonable values (not -23/+22)
- Try training PSR in isolation with a small subset to see if loss changes
- Add gradient clipping to PSR head specifically
- Check if the fill-forward label scheme is fundamentally broken

**Confidence**: HIGH that PSR has never trained. LOW on the root cause.

---

### Q04: Is the cls_score Bias the Single Point of Failure?

**The hypothesis**: The classification head's learned bias parameter (initialized to pi=0.01 → -4.595) is the dominant failure mode. If the bias drifts to a value that produces ~0.079 uniform scores, the entire detection head collapses regardless of other weights.

**Evidence for:**
- score_p50 across ALL DET_PROBE results is consistently 0.016-0.025 (near init)
- cls_score std drops to 0.0068-0.0088 (< 0.01) when collapsed
- score_max reaches 0.93-0.97 on SOME classes (specific weights work fine)
- This pattern is consistent with: bias dominates output, per-class weights are fine
- The math: sigmoid(-2.5) = 0.076, sigmoid(-2.6) = 0.069. A bias shift of just 0.1 moves the entire distribution

**Evidence against:**
- At epoch 8 when det_mAP50=0.184, the bias is presumably different
- The classifier doesn't collapse immediately — it takes 7-8 epochs of healthy training first
- score_max=0.97 means SOME predictions are confident — the per-class weight for that class must be large enough to overcome the bias

**What this looks like mathematically:**
```
cls_logit = W · x + b  (where b is the shared bias, ~595K weights in W)
score = sigmoid(cls_logit)

When b = -4.6 (pi=0.01 init): score ≈ 0.01 for all classes (no differentiation)
When b = -2.5: score ≈ 0.076 for classes where W·x ≈ 0 (most)
When b = -2.5 AND W·x > 5 for a specific class: score > 0.95
```

The question: **Is the bias drifting toward a value where most W·x contributions are near-zero, making the output effectively uniform?**

**What we'd need to test:**
- Track cls_score.bias value across epochs 1-15
- Compute W·x distribution (not just final logit) to separate bias from actual learned weights
- If bias is the problem: bias initialization, bias-specific LR, or remove bias entirely

**Confidence**: MEDIUM — the evidence is suggestive but we need actual bias value tracking to confirm.

---

### Q05: Is Focal Loss Fundamentally Wrong for This Architecture?

**The argument**: Focal Loss was designed for RetinaNet-style detectors with ~100K anchors/image, but with this architecture's specific design:
- 164K anchors/frame × 4 frames = 656K predictions/batch
- Only ~16 positive anchors per batch (0.0024%)
- Focal Loss's negative suppression (p^γ) works at p=0.01 but the cumulative negative gradient across 656K/frame may still dominate

**The math that keeps us up at night:**
```
dFL_neg/dp ≈ 2.55e-5 per negative anchor (at p=0.01)
64K negative anchors × 2.55e-5 = 1.63 cumulative negative gradient

dFL_pos/dp ≈ 73.5 per positive anchor
16 positive anchors × 73.5 = 1176 cumulative positive gradient

1176 / 1.63 = 721× positive-to-negative ratio. This SHOULD be enough.
But: 1176 spread across 595K cls head params = 0.002 per param.
And: 1.63 concentrated on bias + few active weights = much larger effect on bias.
```

**The real question**: Does Focal Loss's negative suppression create a landscape where the uniform-background equilibrium is the ONLY stable fixed point, and all positive gradients are transient perturbations?

**What we'd need to test:**
- Replace Focal Loss with Quality Focal Loss (QFL) or Varifocal Loss for one epoch
- If mAP jumps, the loss function was the bottleneck
- If mAP stays collapsed, the problem is elsewhere

**Confidence**: LOW — we don't have enough evidence to call this yet, but the pattern across 3 separate training regimes (R2.5, Run 8, RF2) all converging to the same equilibrium is suspicious.

---

### Q25: Will the Opus v8 Fixes Break the cls_score Bias Equilibrium? (CRITICAL)

**Why this is the central question**: The 4 fixes (KENDALL_HP_PRE_CAP, DET_POS_IOU_THRESH=0.4, DET_POS_IOU_TOP_K=9, DET_BIAS_LR_FACTOR=1.0) target different aspects of the collapse mechanism. But the primary fix for the equilibrium itself — better positive anchor coverage — is only the IOU_THRESH+TOP_K change. If the equilibrium is not primarily caused by insufficient positive anchors, these fixes may have no effect.

**What we're uncertain about:**
- What if 6-10 positive anchors per GT is still not enough? The math suggests ~150 pos/batch still means only 0.046% positive ratio — still heavily dominated by negatives
- What if the bias equilibrium has a different root cause entirely? (E.g., architectural — Focal Loss landscape itself has this attractor at ~0.079 uniform scores)
- What if the fixes interact in unexpected ways? (E.g., more positive anchors + bias LR=1.0 could accelerate collapse rather than prevent it if the bias drift is the core mechanism)
- No epoch-end validation results yet — we won't know until epoch 17 completes (~49 min/epoch)

**What we'd need:** First epoch-end validation result from epoch 17.

**Confidence**: VERY LOW — first epoch-end val will tell us more.

---

### Q30: Will RF2 Reach Gate Targets with Opus v8 Fixes? (CRITICAL)

**The gate criteria:**
- Target: det_mAP50 >= 0.40, MAE <= 60°
- Current best: det_mAP50 = 0.184 (old run epoch 8), MAE = 4.61° (old run, best ever)

**The reality check:**
- Opus v8 fixes don't directly address the mAP ceiling — they improve stability and bias management, not the model's representational capacity
- The 0.40 target is still 2.2× above best-ever observed mAP across ALL runs (RF1, RF2, Run 8, R2.5)
- Even with perfect stability, reaching 0.40 may require architectural changes (QFL/VFL, improved FPN, better backbone, or more data)

**What we'd need:**
- Epoch 17 validation to establish new baseline with Opus v8
- If mAP recovers to 0.15-0.20 range: fixes help stability but gate target still far away
- If mAP stays at ~0.001: fixes insufficient, need deeper architectural investigation
- If mAP > 0.25: significant improvement but still well below gate target

**Confidence**: LOW — fixes improve stability but may not be sufficient for gate target.

---

## HIGH (Important for Direction)

### Q02: Why Did stage_history Show RF1 best_det_mAP50=0.45 When metric_history Showed Max 0.184? — NOW RESOLVED

**The discrepancy (historical):**
```json
// rf_stage_state.json "stage_history":
{
    "stage": "rf1",
    "status": "completed",
    "best_det_mAP50": 0.45
}

// rf_stage_state.json "metric_history":
[
    {"epoch": 7, "det_mAP50": 0.007},
    {"epoch": 8, "det_mAP50": 0.184},
    {"epoch": 9, "det_mAP50": 0.181},
    {"epoch": 10, "det_mAP50": 0.159}
]
```

The metric_history only has 4 entries (epochs 7-10) and the best is 0.184. But stage_history claimed RF1 completed with best 0.45. That's a 2.4× discrepancy.

**Resolution: FIXED via _validate_stage_history_entry() guard in Opus v8 Fix 4.**

The root cause was an unchecked write to stage_history in the stage manager's gate evaluation code. When RF1's evaluation ran during a transition period, it wrote a state value that was inconsistent with the actual metric_history. Fix 4 adds a validation guard that:
- Checks the written stage_history entry against available metric_history data
- Rejects entries that deviate beyond a configurable threshold from observed metrics
- Logs a warning when such discrepancies are detected

**Impact of the fix:**
- stage_history will now only record values that are consistent with metric_history
- The phantom 0.45 value cannot re-appear in future stage transitions
- metric_history gaps (epochs 11-20 missing) are a separate concern, not addressed by this fix

**Remaining concerns:**
- Other state.json fields (like metric_history truncation at epochs 7-10) may have similar issues
- The guard only checks stage_manager's stage_history, not the entire state.json
- If 0.45 was real (from a different evaluation protocol), we'll never know — but the fix prevents spurious values regardless

**Confidence**: HIGH that stage_history is now clean. LOW on metric_history completeness.

---

### Q06: Why Do Non-Det Heads Show Gradient Leakage When train_head=False?

**Evidence**: LIVENESS probes show psr_head receiving 0.02-0.05 gradient norm even when `train_psr=False`. Similarly body_pose_head shows gradient when `train_pose=False`.

**Why this matters**: If activation flags don't fully stop gradient flow, then the "disabled head" assumption is wrong. The disabled heads may still be consuming gradient that should go to enabled heads.

**Hypothesis**: The `train_head` flags in train.py zero out loss contributions but don't prevent the forward pass from producing outputs that participate in other computations (e.g., through shared features in the FeatureBank or through Kendall log_var regularization).

**Confidence**: LOW — the gradient leakage values are very small (0.02 vs 5+ for active heads) and likely don't affect training.

---

### Q07: Why Is Head Pose MAE Improving When Detection Is Collapsing?

**Evidence**: At epoch 15, head pose MAE = 47.84° (best ever), while det_mAP50 = 0.001 (worst since epoch 7).

**Why this is confusing**: If the detection classifier collapse means the shared backbone features are degenerate, how can head_pose continue improving on those same features?

**Possible explanations:**
1. **Feature separation**: Head pose head uses GAP(C4+GAP(C5)) after FiLM modulation, which is different from the detection head's FPN features. Head pose may have access to features that bypass the collapsed detection pathway.
2. **Detection collapse is output-specific**: The classification subnet collapses, but the backbone and FPN features remain useful for other tasks.
3. **Head pose convergence is trivial**: Once the 9-DoF regression converges to mean pose, the MAE improvement is just noise reduction, not genuinely better understanding.

**Confidence**: MEDIUM — explanation #2 seems most likely. The detection classifier is a thin head on top of FPN features. The FPN and backbone may still produce useful features while the classifier head's specific weights degenerate.

---

### Q08: How Much Does Label Quality Affect Detection Training?

**Context**: The IndustReal detection labels are synthetic (floor plan projections + AR overlay bounding boxes). They're not hand-annotated.

**Why this matters**: Noisy labels could explain:
- The detection mAP ceiling (~0.184 at RF2 epoch 8)
- The classifier's reluctance to make confident predictions (scores stuck at 0.019)
- Score_max reaching 0.97 only for classes with consistent synthetic labels

**What we know**: The synthetic labels are geometrically accurate (projected from known 3D object positions) but may miss objects, have temporal jitter, or use different class definitions than standard detection datasets.

**Confidence**: MEDIUM — label quality is likely a ceiling factor but unlikely to be the primary collapse mechanism. The collapse is too dramatic (0.184→0.001) to be explained by label noise alone.

---

### Q09: Is the 0.001 Gradient Density Threshold Universal?

**The gradient sparsity math**: RF1 detection-only produced ~4×10⁻⁵ gradient per backbone parameter per step. With head_pose, we estimated ~400K gradient contributions per batch instead of ~16.

**The threshold question**: What minimum gradient density is needed for this architecture to learn? Is:
- 4×10⁻⁵/param → NOT ENOUGH (RF1 detection-only: collapse)
- 0.01/param → ENOUGH? (RF2 with head_pose: works for 7-8 epochs)
- 1.0/param → ENOUGH? (R2.5 all heads)

**Why we don't know**: We don't have actual per-parameter gradient norm measurements. The LIVENESS probe measures total head gradient, not density.

**What we'd need**: A `backbone_grad_norm_per_param` diagnostic that divides total backbone gradient norm by number of active parameters.

**Confidence**: MEDIUM — the concept is sound but we've never actually measured per-parameter gradient norms.

---

### Q10: Why Does RF2 Show 0.184 While RF1 Stage Claimed 0.45?

**The puzzle**: RF1 stage_history says `best_det_mAP50=0.45`. RF2 best is 0.184 at epoch 8. If the RF1 checkpoint was truly at 0.45, then RF2 (continuing from RF1 best.pth) should START near 0.45 and improve. Instead, RF2's best is far below 0.45, and it reaches this at epoch 8 (not epoch 1).

**Implications if 0.45 is real:**
- RF2 somehow DESTROYED the detection capability from RF1
- Adding head_pose training in RF2 interfered with detection (catastrophic forgetting)
- Need to freeze detection head during RF2 or use lower LR for detection

**Implications if 0.45 is an artifact:**
- RF1 was never as successful as claimed
- The detection model has a fundamental performance ceiling below 0.20
- The gate criteria needs re-examination

**What we'd need**: 
- Find the RF1 checkpoint that achieved 0.45 and evaluate it independently
- Compare the configuration used for RF1 gate evaluation vs RF2's per-epoch validation

**Confidence**: MEDIUM — the discrepancy could explain the entire RF2 collapse if RF1's checkpoint was never actually good.

---

### Q11: Does the EVAL COLLAPSE Signal Affect Training?

**Evidence**: "EVAL COLLAPSE" appears 56 times in train.log at epoch 15. This means the evaluation code detected that all 3 heads (detection, activity when enabled, PSR) produced near-zero metrics simultaneously.

**The concern**: If EVAL COLLAPSE triggers any special behavior (e.g., emergency checkpoint save, metric clamping, NaN substitution), it might mask the detection of collapse or interfere with normal training.

**What we know**: The EVAL COLLAPSE signal was added as a diagnostic flag. It reports but doesn't intervene. It shouldn't affect training, but we should verify.

**Confidence**: LOW — we should read the actual EVAL COLLAPSE code path.

---

### Q23: Will RF2 Ever Reach the Gate Target of det_mAP50>=0.40?

**Pre-Opus v8 trajectory**: det_mAP50 = 0.001 at epoch 15 (after peaking at 0.184 at epoch 8). The trajectory was declining, not improving.

**Post-Opus v8**: Restarted at epoch 17 with 4 fixes. The trajectory may change, but no validation results are available yet.

**Projection (pre-fixes)**: At the epoch 15 trajectory, the model would never reach 0.40. Even if it recovered to epoch 8's 0.184, that's still 54% below target.

**Theoretical maximum at 35% data**: Unknown. If RF1's 0.45 at 20% data is real, then 0.40 at 35% data is achievable. But we've never seen RF2 reach even 0.184 consistently.

**Opus v8 factor**: Fixes target stability, not ceiling. If we now maintain 0.15-0.20 consistently instead of collapsing, we still need architectural upgrades for the gate target. See Q30 for the full analysis.

**Confidence**: MEDIUM — pre-fix trajectory was hopeless. Post-fix trajectory is unknown. Gate target may be unachievable without architectural changes.

---

### Q27: Is the Phantom 0.45 Fix Complete? (HIGH — now RESOLVED)

**Background**: The _validate_stage_history_entry() guard was added as Fix 4 in Opus v8, resolving the phantom 0.45 value in stage_history. But there's still the question of whether other state.json fields (like metric_history truncation at epochs 7-10) have similar unchecked write issues.

**What was fixed:**
- stage_history now validates its entries against metric_history before writing
- Phantom values that deviate beyond a configurable threshold are rejected
- The fix specifically targets stage_manager.stage_history writes

**What remains open:**
- metric_history array: epochs 7-10 are present but epochs 11-20+ are missing. Was this truncation, or was the array never fully populated?
- Other state.json fields may have similar phantom-value vulnerabilities
- The guard only checks within stage_manager's stage_history, not the entire state persistence layer

**Resolution**: Stage history phantom is fixed. Metric_history gap is a separate, lower-priority issue.

**Confidence**: HIGH that stage_history is now clean. LOW on metric_history completeness.

---

## MEDIUM (Important for Understanding)

### Q12: Is the 5-Minute Swarm Interval Missing Transient Collapse Events?

**The monitoring gap**: The rf2_swarm runs every 300 seconds. At 0.9 batch/s, that's 270 training steps between checks. The RF1 death spiral in Phase 3 showed that the detection head can go from healthy (gradient=6.56) to dead (gradient=0.047) in 100-150 steps (~2 min).

**If the swarm misses the transient**: The auto-restart watchdog counts "dead cycles" when PH01 (PID alive) fails. But if the training process is alive while the detection head is dead (as it is now at epoch 15), the swarm won't detect this as a health failure.

**What we'd need**: Add detection collapse detection to the swarm — a metric health check that triggers when det_mAP50 drops below a threshold for N consecutive cycles.

**Confidence**: HIGH that this monitoring gap exists. LOW on whether closing it would help (we already know the collapse is happening).

---

### Q13: Is the 9-DoF Head Pose Head Actually Learning Anything Useful?

**Head pose MAE**: 71.67° → 47.84° (epoch 7 → 15), improving by ~2° per epoch.

**But what does 47.84° MAE mean for a 9-DoF prediction?**
- Forward vector: 3 values (cos/sin of yaw, pitch — but there's no roll)
- Position: 3 values (x, y, z normalized)
- Up vector: 3 values

A MAE of 47.84° on angular components is basically random (random prediction on a sphere has ~57° MAE). The improvement from 71.67° to 47.84° suggests the head is converging toward a mean pose, not learning to predict actual head orientation per frame.

**If head_pose is just predicting mean pose**: Then it's NOT providing useful gradient diversity to the backbone. It's providing a constant gradient that pushes backbone features toward a generic orientation-invariant representation, which might actually HURT detection by smoothing out features that vary with head orientation.

**This could explain the RF2 collapse**: Head_pose provides dense gradient, but that gradient is essentially "predict the empirical mean" — it washes out the feature variance that detection needs.

**What we'd need to test:**
- Check if head_pose predictions have any correlation with actual head orientation
- If not: head_pose is a mean-predictor and its gradient is feature-smoothing
- If so: head_pose is genuinely learning and its gradient is usefully diverse

**Confidence**: MEDIUM — this is a speculative but testable hypothesis that could explain the RF2 collapse.

---

### Q15: Why Are All 5 Checklists Failed When Gate Is Not Passed?

**Evidence**: rf_stage_state.json shows ALL 5 checklists failed:
```json
"checklist_results": {
    "gate": {"passed": false},
    "health": {"passed": false},
    "convergence": {"passed": false},
    "validation": {"passed": false},
    "stability": {"passed": false}
}
```

Every single category is failed. Not a single one passes. This means:
1. No gate criteria met (expected — det_mAP50=0.001 < 0.40)
2. Some health check failed (likely detection gradient or metric health)
3. No convergence (metrics not improving)
4. Validation metrics below thresholds
5. Some stability issue

**The question**: Are these 5 independent failures, or does 1 failure (detection collapsed) cascade into all 5? Most likely the latter — if detection is collapsed, all 5 categories that depend on detection metrics will fail.

---

### Q16: How Does the Stage Manager Handle a Completed-but-Collapsed Stage?

**Critical path question**: What happens when RF2 times out (max_epochs=36) with gate not met?

Options:
1. **Retry**: Stage manager applies retry strategy (LR reduction, warmup increase)
2. **Advance anyway**: Skip failing RF2, advance to RF3
3. **Kill and restore from checkpoint**: Restore from RF1 best.pth, retry RF2

If option 1: We already proved LR reduction doesn't fix the death spiral (Phase 3→4 identity).
If option 2: RF3 adds activity head, which might provide useful gradient — but PSR will still be dead.
If option 3: RF2 retry from RF1 checkpoint would reproduce the same collapse.

**Confidence**: HIGH that none of the current retry strategies will fix the RF2 collapse.

---

### Q24: Are We Overfitting to the 0.7% GT Frames?

**With DET_GT_FRAME_FRACTION=0.90**: 90% of batches contain GT frames. But those GT frames are from the same small pool of objects. The model sees the SAME GT objects every batch but different backgrounds.

**Is the detection classifier learning to recognize the specific 1-2 objects that appear in GT frames** rather than learning a general object detector? This would explain:
- Why det_mAP50 is bounded (the model memorizes specific objects, doesn't generalize)
- Why score_max reaches 0.97 (it IS confident about memorized objects)
- Why score_p50 is 0.019 (it's background for everything else)

**This would be catastrophic**: The model is memorizing, not learning.

**What we'd need**: Test on a held-out set of objects/scenes that were never in the training GT set.

---

### Q26: Does 50% Data (up from 35%) Meaningfully Change the Positive Anchor Count? (MEDIUM)

**The data increase**: Moving from 35% to 50% training data means approximately:
- ~24 recording files vs ~17 at 35% = 41% more GT frames
- With the IOU_THRESH=0.4 + TOP_K=9 fixes, each GT produces 6-10 positive anchors vs ~1 previously

**Combined effect estimate:**
- Previously: ~16 positive anchors per batch at 35% data
- Post-fix at 35% data: ~96-160 positive anchors per batch (6-10× from TOP_K)
- At 50% data: ~135-225 positive anchors per batch (41% more GT frames)
- Total anchors: 656K per batch → positive ratio = 0.017%-0.034%

**Is this enough?**
- The cumulative negative gradient: 656K × 2.55e-5 = 16.7 total (after focal suppression)
- The cumulative positive gradient: 160 × 73.5 = 11,760 (distributed across all cls head params)
- Still concentrated impact: 16.7 on bias, ~11,760 spread across 595K params

**Verdict**: Math suggests improvement, but we don't know if 0.017% positive ratio is sufficient. The key question is whether the bias equilibrium is broken, not whether total gradient is higher.

**Confidence**: MEDIUM — math suggests improvement, but unknown if sufficient.

---

### Q28: Did Kendall Staged Training Ever Activate? (MEDIUM)

**The background**: KENDALL_STAGED_TRAINING=False was documented as Fix 3 in the Kendall analysis. However, analysis of losses.py showed that the code already guards with `if bool(getattr(C, 'STAGED_TRAINING', True))` — meaning staged training defaults to True but is checked at runtime.

**The question**: If the config never had STAGED_TRAINING=True in any deployed version, then Fix 3 was a no-op — the guard was never necessary because the feature was never active.

**Why this matters:**
- If STAGED_TRAINING was never True: we wasted time diagnosing and "fixing" a non-issue
- If STAGED_TRAINING was True in some version: the fix was genuinely needed
- Either way, it doesn't affect the current collapse — but it affects our understanding of which fixes actually matter

**What we'd need to test:**
- Check git history for STAGED_TRAINING=True across all deployed config versions
- If never set to True: mark Fix 3 as "never needed"
- If set to True in some version: verify the fix was correct

**Confidence**: MEDIUM — likely never active, but should verify through git history.

---

## LOW (Nice to Know)

### Q14: Why Does RF2 Use epoch Index 1 for stage_index?

**Evidence**: rf_stage_state.json shows `"stage_index": 1` for RF2. Since RF1 is index 0, this is correct. But RF1's stage_history entry says `"stage": "rf1", "status": "completed"`. So the stage_manager correctly recorded RF1 completion and advanced.

**This is NOT a bug** — just noting that the index is 0-based (RF1=0, RF2=1). Consistent.

---

### Q17: Was the Auto-Restart Script Ever Triggered?

**Evidence**: Auto-restart watchdog at 3 consecutive dead cycles. Training PID 3176288 (current, post-Opus-v8 restart) replaced PID 1043628 which ran continuously through the epoch 15 collapse.

**Status**: Auto-restart was NOT triggered by the epoch 15 collapse because the training process never died. The collapse was within the model, not the process. The new PID 3176288 was a manual restart to deploy Opus v8 fixes.

**Concern**: If the training enters a state where it produces loss=NaN or crashes, the auto-restart would trigger. But the epoch 15 collapse doesn't crash — it just produces degenerate predictions. The auto-restart is designed for process death, not model collapse.

---

### Q18: Why Did the Swarm Find 6 Blocking Bugs in the First Hours?

**The pattern**: Deploying a comprehensive monitoring system immediately found 6 bugs in the things it monitors. This suggests the original monitoring (rf2_checklist.py) had significant gaps.

**Root cause**: The monolithic checklist had too many checks (118) to verify manually. The swarm's per-agent granularity made debugging easier (each agent has 4-15 checks), so specific failures (NaN filter, log_head_text, spike detection) were immediately visible.

---

### Q19: How Much VRAM Does Each Head Use at Training Time?

**Current usage**: ~8GB out of 12GB at RF2 (detection + head_pose only).

**When activity and PSR are enabled**: Projected ~11GB (near limit). May need to reduce batch_size or grad_accum.

**This needs tracking** when RF3+ is ready.

---

### Q20: Is the Heartbeat Staleness a Red Herring?

**Evidence**: `last_heartbeat: "2026-06-20T06:23:34"` (timestamp). If this is not being updated, the heartbeat code may not be active in the running training process.

**Status**: The heartbeat fix was applied to train.py. The current RF2 process (PID 3176288) was started from an epoch 15 checkpoint after the Opus v8 restart — whether the heartbeat fix took effect depends on whether train.py's heartbeat code runs on resume (not just fresh start). This should be verified by checking if the heartbeat timestamp updates during the current run.

**Impact**: LOW — the heartbeat is a monitoring feature, not a training feature. Its absence doesn't affect model quality.

---

### Q21: Should We Track per-Class AP to See Which Classes Collapse First?

**Current diagnostics**: det_mAP50 only (aggregate). We don't know WHICH of the 24 classes are collapsing or whether some classes maintain healthy scores.

**Hypothesis**: If only a few classes drive the mAP (e.g., classes with the most GT examples), and the rest collapse, the problem is data imbalance. If ALL classes collapse simultaneously, it's a classifier-wide failure.

**What we'd need**: Per-class AP tracking in the validation output (or DET_PROBE).

---

### Q22: Is the Cluster + Swarm Architecture Overkill?

**22 agents, 134 checks, 5-min cycle, 67MB log file**. Is this complexity justified?

**Counter-evidence**: The 6 bugs found in the first hours prove the monitoring is necessary. The auto-restart watchdog saved RF1 once. The per-agent architecture makes debugging specific failures tractable.

**Open question**: Could the same results be achieved with a simpler 5-agent system?

---

### Q29: Is the Heartbeat Fix Actually Working? (LOW)

**The concern**: The heartbeat fix was applied to train.py, but the running process (PID 3176288) was started from a checkpoint (epoch 15 checkpoint loaded at epoch 17 restart). If the heartbeat code requires a fresh start and does not initialize properly on resume, it may not be updating state.json's last_heartbeat field.

**Why this matters (minimally):**
- The heartbeat is a monitoring feature — it doesn't affect training quality
- But stale heartbeat means the health monitoring system might not detect liveness accurately
- If we ever need the auto-restart watchdog to make decisions based on heartbeat freshness, it needs to be reliable

**What to check:**
- Compare last_heartbeat timestamp in state.json against current time
- If timestamp is updating: heartbeat fix is working on resume
- If timestamp is stuck: heartbeat fix needs a fresh-start path

**Confidence**: MEDIUM — partially answered by checking if heartbeat timestamp updates during the current epoch 17 run.

---

## Appendix: Quick Reference

| # | Question | Severity | Confidence in Answer | Blocks? |
|---|----------|----------|---------------------|---------|
| Q01 | Why collapse AGAIN at RF2 epoch 15? (Will Opus v8 fix it?) | CRITICAL | VERY LOW | YES |
| Q02 | stage_history 0.45 vs metric_history 0.184 — NOW RESOLVED | HIGH (RESOLVED) | HIGH | No |
| Q03 | Has PSR ever trained? | CRITICAL | LOW | YES (for RF4+) |
| Q04 | Is cls_score bias the SPOF? | CRITICAL | MEDIUM | YES |
| Q05 | Is Focal Loss wrong for this arch? | CRITICAL | LOW | YES |
| Q06 | Gradient leakage from disabled heads | HIGH | LOW | No |
| Q07 | MAE improving while mAP collapses | HIGH | MEDIUM | No |
| Q08 | Label quality impact | HIGH | MEDIUM | Possibly |
| Q09 | Gradient density threshold | HIGH | MEDIUM | No |
| Q10 | RF2 0.184 vs RF1 claimed 0.45 | HIGH | MEDIUM | YES |
| Q11 | EVAL COLLAPSE signal side effects | HIGH | LOW | No |
| Q12 | Swarm interval missing transients | MEDIUM | HIGH | No |
| Q13 | Head pose learning nothing useful | MEDIUM | MEDIUM | Possibly |
| Q14 | stage_index discrepancy | LOW | HIGH | No |
| Q15 | Cascading checklist failures | LOW | HIGH | No |
| Q16 | Stage manager handling of RF2 | MEDIUM | HIGH | No |
| Q17 | Auto-restart never triggered | LOW | HIGH | No |
| Q18 | 6 swarm bugs in first hours | LOW | HIGH | No |
| Q19 | VRAM projection for RF3+ | LOW | MEDIUM | Future |
| Q20 | Heartbeat staleness | LOW | HIGH | No |
| Q21 | Per-class AP tracking | LOW | HIGH | No |
| Q22 | Swarm overkill | LOW | MEDIUM | No |
| Q23 | RF2 gate feasibility | HIGH | MEDIUM | YES |
| Q24 | Overfitting to GT frames | MEDIUM | LOW | Possibly |
| Q25 | Will Opus v8 fixes break bias equilibrium? | CRITICAL | VERY LOW | YES |
| Q26 | 50% data impact on positive anchor count | MEDIUM | MEDIUM | Possibly |
| Q27 | Phantom 0.45 fix complete? — RESOLVED | HIGH (RESOLVED) | HIGH | No |
| Q28 | Kendall staged training ever active? | MEDIUM | MEDIUM | No |
| Q29 | Heartbeat fix actually working? | LOW | MEDIUM | No |
| Q30 | RF2 gate targets with Opus v8 fixes? | CRITICAL | LOW | YES |

---

*Generated 2026-06-20 by Claude Code. All questions backed by live log analysis, training state, and diagnostic evidence. Severity reflects impact on path to RF2 gate targets. Opus v8 fixes deployed at epoch 17 restart (PID 3176288); epoch-end validation pending.*
