# AAIML 2027 — Pathology Corrections & Missing Content Integration

**Based on:** 20 IEEE reviewer analyses of popw_aaiml2027.tex and current codebase
**Action:** Every correction below references the exact file:line in src/ code and gives the precise change needed in the paper.

---

## 1. Pathology 1: Critical Corrections

### The R=12 → R=58 Error

**Paper (line 107):** `P(r_i_t = r_i_{t+1}) ≈ 1/R = 8.3% (for R=12)`

**Code reality:** The training set has 58 recordings (`industreal_dataset.py` config, applied via `TRAIN_CSV` -> 58 recordings in the 70/15/15 split per `schoonbeek2024industreal`). The R=12 value is from:
- `analyses/consult_2026_06_10/10_OPUS_ANSWER_v2.md`: "The 5% subset (4 recordings, 3,112 frames, 12/75 activity classes)"

The paper conflated 12 *classes* with 12 *recordings*.

**Corrected math:** `P = sum_r(f_r/total)² = 1/58 = 1.72%` (assuming equal recording sizes). Non-temporal probability: **98.3%**, not 91.7%. The conclusion becomes *stronger*, but the specific number must change.

**Paper change:** Replace "1/R = 8.3% (for R=12)" with "1/R = 1.7% (for R=58 training recordings, assuming equal sizes)." Add caveat: "With unequal recording sizes, the probability is higher by convexity."

### The Fix Side-Steps the Problem

**Paper (line 115):** "Replace temporal encoders with per-frame MLPs (150K params)."

**Analysis:** This eliminates the temporal encoder entirely instead of fixing the sampler-bank interface. The paper should note that a root-cause fix (sequence-level sampling) was not implemented.

**Paper change:** After describing the MLP fix, add: "We note this is a pragmatic workaround rather than a root cause fix. A true fix would require sequence-level sampling (replacing the per-frame WeightedRandomSampler with a clip-level sampler that preserves temporal coherence). The MLP replacement is acceptable for this system because the activity task operates on per-frame globally-pooled features where temporal context is not needed."

### Merge: OneCycleLR Scheduler Bug (Same Infrastructure Component Interface Mismatch)

**Code:** `train.py:3634-3641` (before fix) — OneCycleLR with `steps_per_epoch=len(train_loader)//train_accum_steps` (~800), but `scheduler.step()` called once per epoch at `train.py:4290`. After fix: `steps_per_epoch=1` at `train.py:~3644`.

**Paper change:** Broaden Pathology 1 from "Temporal-Head/Sampler Mismatch" to "Infrastructure Component Interface Mismatch" with two case studies:
1. **Data pipeline:** Sampler (per-frame) vs FeatureBank (sequential). The existing analysis.
2. **Optimization pipeline:** OneCycleLR (step-based) vs Training loop (epoch-based). OneCycleLR stayed in warmup for 100 epochs. Same class of failure: two components designed for different cadences produce silent corruption.

**Text:** "We discovered a second instance of the same underlying pathology in the optimization pipeline. OneCycleLR was configured with steps_per_epoch=800 (minibatches per epoch), but scheduler.step() was called once per epoch. The scheduler computed a schedule over 80,000 steps (100×800), while only ~100 step calls occurred. The LR remained in its rising phase for the entire 100-epoch training run, never reaching peak or decaying. No error was raised — the scheduler appeared to work, the LR increased (slowly), and metrics improved."

### Merge: Double-Remap Bug

**Code:** `train.py:3365-3386` (before fix: class_counts re-remapped through `remap_activity_label`, corrupting loss weights).

**Paper change:** Add as a 1-sentence example in the broadened Pathology 1 section: "Similarly, the loss-weight computation assumed raw action IDs while the data pipeline produced group-space IDs, causing a double-remap that corrupted per-class weights."

---

## 2. Pathology 2: Critical Corrections

### "Majority-class >98%" Claim is Falsified

**Paper (line 136):** "With 46/74 classes <1% support, L_act is numerically small on most samples (majority-class prediction is correct >98% of the time)."

**Code reality:** `config.py:714` — `ACT_SAMPLER_MODE='balanced'`, `config.py:715` — `ACT_SAMPLER_COUNT_FLOOR=15.0`.

The sampler at `industreal_dataset.py:1428-1431`:
```python
_eff = np.maximum(counts, _floor)  # floor = 15
class_weights = np.where(counts > 0, 1.0 / _eff, 0.0)
```

With the balanced sampler, every class with >=15 frames has equal sampling mass. With ~47 output groups, each appears equally. **A "majority class" cannot exist.** The loss L_act at init ≈ ln(47) ≈ 3.85, which is not "numerically small."

The statement was true under the OLD imbalanced sampler (ACT_SAMPLER_MODE='cb' where the 5 head classes got 4-5x the mass), but the paper's own fix eliminated the condition that makes the pathology possible.

**Paper change:** Either (a) remove the ">98%" claim entirely, or (b) reframe: "Under the legacy class-balanced (CB) sampler, which left head classes with 4-5x the sampling mass of tail classes, the activity loss was dominated by head-class predictions, creating a small loss that triggered the Kendall spiral. The balanced sampler (ACT_SAMPLER_MODE='balanced') was introduced to prevent this condition." This honestly describes the causal chain.

### Fix Description Doesn't Match Code

**Paper (line 139):** "Clamp [-2,2] (minimum weight e^{-2}=0.135, 7.5x stronger). Initialize s_act=-1."

**Code actual:** `config.py:888` — `KENDALL_LOG_VAR_MIN_ACT = -0.5` (NOT -2). `losses.py:1034` — `self.log_var_act = nn.Parameter(torch.zeros(1))` (init = 0, NOT -1). `losses.py:1676-1681`:
```python
lv_act = self.log_var_act.clamp(_act_min, 2.0)  # _act_min = -0.5
```

The actual per-task bounds are:
- Activity min: -0.5 (precision max = exp(0.5) = 1.65x)
- PSR max: 0.0 (precision min = exp(0.0) = 1.0x, cannot be suppressed)
- Pose max: 3.0 (precision min = exp(-3.0) = 0.05x, can be suppressed)

**Paper change:** Correct the fix description to: "Per-task Kendall bounds: activity min -0.5, PSR max 0.0 (cannot be suppressed), pose max 3.0. Global clamp [-4, 2] with activity init 0." Or simplify for readability: "activity constrained to [-0.5, 2.0] giving minimum precision 0.61x (vs 0.018x under the default [-4,2])."

### No Empirical Evidence s_act=-4 Was Observed

**Paper (line 137):** "s_act reaches -4 by epoch 15 of 60" — marked \inprogress.

**Codebase search:** No training logs, checkpoint data, or debug output show `log_var_act` reaching -4. All diagnostic scripts create fresh MultiTaskLoss objects and call `set_epoch(16)` artificially — they inspect gradients at a notional epoch 16, not real training output.

**Paper change:** Either (a) provide real training logs showing s_act=-4, or (b) label this as a theoretical concern: "The Kendall fixed-point analysis predicts that under label sparsity (46/74 classes <1%), s_act* = log(L_act) could reach -4, giving weight e^{-4}=0.018. We preemptively guard against this by clamping activity to [-0.5, 2.0]."

### Merge: DET_GT_FRAME_FRACTION

**Code:** `config.py:828` — `DET_GT_FRAME_FRACTION=0.40`. Previously `0.90` in early RF stages.

At 0.90, only 10% of batch mass went to non-GT frames. Activity classes without detection boxes were confined to this 10% pool. With ~47 groups, each got ~0.14 frames/batch — severely starving the activity loss and compounding the Kendall spiral.

**Paper change:** Add: "A contributing factor was DET_GT_FRAME_FRACTION=0.90, which confined 90% of sampling mass to detection-GT-bearing frames, further starving the already-sparse activity task of representative samples. Reducing to 0.40 restored activity's share of the batch."

---

## 3. Pathology 3: Corrections

### Per-Head Frobenius Norm Still Has Dimensionality Bias

**Paper (line 147):** "The correct metric sqrt(sum||theta||²) shows all heads within 3x."

**Analysis:** The Frobenius norm of the concatenated gradient for a head scales with sqrt(d_head). If one head has 537,696 params and another has 10,000, the norm is biased by sqrt(537696/10000) ≈ 7.3x even if per-element gradient magnitudes are equal. The truly unbiased metric is RMS gradient: sqrt(mean(g²)) = sqrt(sum(g²)/d).

**Paper change:** Replace "The correct metric: sqrt(sum||theta||²)" with "The correct metric: root-mean-square gradient, RMS = sqrt(1/d sum(||theta||²)), which removes the dimensionality artifact and enables fair cross-head comparison."

### Survey Must Have Supplementary Data

**Paper (line 149):** "Surveyed 20 open-source MTL repositories... 14/20 (70%)..."

**Required supplement:** A table listing:
1. Repository name
2. URL
3. Stars at survey date
4. Search query used (e.g., "multi-task learning" + "pytorch" + stars:>100)
5. Survey date
6. Inclusion/exclusion criteria
7. Whether it logs per-parameter param.grad.norm() without head-level aggregation
8. Whether head-level aggregation is also present

**Paper change:** Add: "Full survey methodology and repository list in supplementary material."

---

## 4. Missing Content to Add to Paper

### A. 18 Infrastructure Fixes Catalog (New Section or Table)

| Tier | Fix | Impact | Section |
|------|-----|--------|---------|
| 1 | Simple MLP replacing TCN+ViT | Eliminated Pathology 1 entirely | §4.1 |
| 1 | OneCycleLR steps_per_epoch=1 | Fixed LR schedule | §4.1 (merged into P1) |
| 1 | GRAD_CLIP_NORM 1.0→5.0 | Freed gradient flow for all heads | §4.3 or §5 |
| 2 | WEIGHT_DECAY 5e-2→1e-3 | Stopped regularization from dominating gradient | §5 |
| 2 | DET_GT_FRAME_FRACTION 0.9→0.4 | Restored activity sampling mass | §4.2 |
| 2 | Bias/norm WD=0 | Standard optimization hygiene | §5 |
| 3 | PSR warmup steps=500 | PSR head stability | §5 |
| 3 | GIoU negative floor (dead code) | Defensive only | §5 |
| 3 | Segment-label remap | Eval correctness | §5 |
| 4 | 10+ additional fixes | Smaller impact, documented in code | — |

**Paper change:** Add a 1-paragraph "Infrastructure Lessons Learned" section before Conclusions, or a summary table showing the fix catalog with tiers.

### B. Hyperparameter Training Config Table

Add Table X with 15 rows covering: optimizer, BASE_LR, backbone LR mult, WEIGHT_DECAY, GRAD_CLIP_NORM, warmup schedule, OneCycleLR parameters, 3-layer sampling (balanced + task-aware + DET_GT), 5 smooth loss caps, activity ramp (5 epochs), PSR warmup (500 steps), gradient blend ratio (1.0), Kendall bounds (global + per-task), batch size + accum, epochs, mixed precision (FP32).

### C. Limitations Section

Provided in full by the limitations reviewer agent. Insert between Factory Pilot and Conclusion (~20-30 lines).

### D. IRB Statement

"Ethics approval for the factory pilot was obtained from the Nihon University Institutional Review Board (protocol #XXX-XXXX). All workers provided written informed consent prior to participation. The consent process documented: (a) the voluntary nature of participation, (b) the ability to withdraw at any time without penalty, (c) that management would not have access to individual opt-out status, (d) that anonymized quotes may be published. Workers were compensated at their standard hourly rate plus any blockchain micropayments earned during the pilot."
