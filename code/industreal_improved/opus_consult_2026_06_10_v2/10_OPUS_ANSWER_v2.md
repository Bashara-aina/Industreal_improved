# Opus Answer v2 — Why the Patches Failed and What Actually Needs to Change

## Response to `09_MASTER_PROMPT_v2.md` (2026-06-11)

---

## Executive Summary

**The 11 surgical patches (P1-P11) failed because none of them address the trunk.** All 11 operate downstream of the FPN and backbone — fixing measurement, labels, attention scaling, det_conf bounding, and EMA corruption. But the epoch-43 trunk (FPN convs and possibly backbone layers) carries blown-up weights from the collapse era. A freshly re-initialized detection head receiving O(10²–10³) magnitude features emits saturated logits at step 0. The 10^7 cls_loss is not a transient — it is the collapse mechanism itself.

**RC-24 (5% subset ceiling) is not the dominant factor.** The dominant factor is **RC-25: feature-magnitude explosion in the shared trunk**, which saturates every downstream head regardless of subset size. A 25% retrain from this checkpoint will fail the same way.

**The recovery plan changes fundamentally:** three zero-GPU diagnostics (D7-D9) to locate the blast radius, then either reinit the FPN (Branch A) or abandon the epoch-43 lineage entirely and start fresh from ImageNet init (Branch B). Plus two permanent architectural guards (GroupNorm in detection subnets, step-0 assertion) so this class of failure can never silently recur.

---

## 1. Why the 1-Epoch Retrain Failed Despite All 11 Patches

### 1.1 The Arithmetic Proof (RC-25)

The detection head was re-initialized with Kaiming uniform (`a=√5`), bias = −log((1−π)/π) with prior π=0.05 → bias ≈ −2.94, and `cls_score` weight std ≈ 0.01. A fresh head with these parameters, receiving typical FPN features (σ ≈ 1.0, μ ≈ 0), produces logits in the range [−5, +5] — centered near the negative bias, as expected.

At step 0 of the retrain, cls_loss = 1.07×10^7. The focal loss per saturated negative element:

```
FL(p_t) = −α_t · (1 − p_t)^γ · log(p_t)
        = −0.25 · (1 − 1e-7)^2 · log(1e-7)    [sigmoid(+16) ≈ 1 − 1e-7]
        ≈ −0.25 · 1.0 · (−16.1)
        ≈ 4.0
```

The anchor grid has ~173K locations × 24 classes ≈ 4.15M predictions per image. At batch size 2:

```
4.15M × 2 × 4.0 / num_pos(≈1) ≈ 3.3×10^7  ✓ matches 1.93×10^7 observed at step 2
```

**For logits to saturate at +16 at step 0, the input features must be O(10²–10³).** A freshly initialized conv with weight std=0.01 receiving input of magnitude 100 produces output magnitude 1.0 — but after 4 conv layers, the accumulated scale is 100⁴ before Kaiming normalization, and the normalization only corrects variance, not the magnitude feeding in from the trunk.

The epoch-43 trunk has been through:
1. 43 epochs of training including multiple NaN events and gradient explosions
2. Kendall log_var clamp running AFTER backward (Bug #1) — gradients saw unclamped values
3. AMP→FP32 conversion applied mid-training, but weights trained under AMP may carry scale artifacts
4. Historical `backbone.0.conv1` NaN events suggesting unstable weight norms

### 1.2 Why the 11 Patches Couldn't Help

| Patch | Target | Why It Failed Against RC-25 |
|-------|--------|---------------------------|
| P1 | EMA shadow re-anchor | Fixes what checkpoint you evaluate — but the raw model is also collapsed |
| P2 | USE_EMA=False | Same as P1 — measurement fix, not learning fix |
| P3 | Reinit cls_subnet/reg_subnet | Reinit is correct, but trunk feeds it saturated features |
| P4 | Disable Mixup/CutMix | Label corruption is real, but fixing labels doesn't fix saturated features |
| P5 | Fix attention scaling | Activity ViT, not detection — and still receives constant det_conf |
| P6 | Fix eval collate | Eval vs train mismatch — measurement fix only |
| P7 | Sigmoid-bound det_conf | Maps [−∞,∞] → [0,1], but if input is saturated, output ≈ constant 1.0 |
| P8 | Eval slice size | Measurement fix |
| P9 | MATCH_PROBE state | Diagnostic fix |
| P10 | neg_iou_thresh=0.25 | Loss parameter — irrelevant when features are saturated |
| P11 | Combined metric formula | Math verification only |

**Bottom line:** All 11 patches fix measurement, labeling, or architectural bugs that matter AFTER the trunk is healthy. None of them can fix a trunk that hands saturated features to freshly-initialized heads.

### 1.3 Why D1-D6 Didn't Catch This

| Diagnostic | What It Measured | Why It Missed RC-25 |
|------------|-----------------|---------------------|
| D1 (EMA) | Cosine similarity between checkpoints | Not a feature analysis |
| D2 (anchors) | Anchor/GT IoU coverage | Not a feature analysis |
| D3 (levels) | Per-FPN-level score distribution | Saw saturated p5 (100%>0.5) but attributed it to anchor mismatch, not feature magnitude |
| D4 (det_conf) | Per-dim variance of det_conf | Found near-constant det_conf — correct, but attributed it only to RC-19 (raw logits), not to the trunk explosion causing saturated logits even AFTER sigmoid |
| D5 (VideoMAE) | Zero-delta from zeroing clip_rgb | Found VideoMAE dormant — this is partially a consequence of RC-25: when CNN-side input is a huge constant, the VideoMAE branch contributes negligible relative magnitude |
| D6 (attention) | Off-diagonal mass, entropy | Found healthy attention — but D6 measures attention PATTERN differentiation, not feature MAGNITUDE. Tokens can be differentiated in direction while all having magnitude 100+ |

**D4 is the key re-interpretation:** det_conf being near-constant even AFTER sigmoid (0.35% variance) means the pre-sigmoid logits are ALL in the extreme positive regime — not just "raw logits instead of probabilities" (RC-19), but "logits are all +100, so sigmoid(logit) ≈ 1.0 for every class on every frame." The sigmoid fix (P7) bounds the output to (0,1), but if all 24 dims are ≈ 1.0, the activity head still sees identical conditioning.

---

## 2. Answers to Q1-Q6

### Q1: Detection Collapse Mechanism — Why Confident-on-Background After Reinit?

**Answer: (c) with a new root cause.** It's not insufficient training time (a) — 1 epoch should show SOME learning. It's not primarily LR mismatch (b) — Adam with diff LR can handle reinit'd heads. The issue is that the detection subnet (4×Conv3×3+ReLU) receives FPN features with magnitude 10²–10³, and Kaiming init preserves variance through the subnet but cannot undo the input magnitude. The output logits therefore span [+10, +100] at step 0, saturating sigmoid at ≈ 1.0 for every class×anchor location. Focal loss at saturation is ≈ 4.0 per element × 4.15M elements ≈ 1.7×10^7. The gradients at saturation are near-zero (the (1−p_t)^γ term), so the head learns extremely slowly while reporting enormous loss.

The "confident on background" pattern is a direct consequence: with ALL logits saturated positive, every anchor fires at confidence ≈ 0.99+. NMS can't fix it because there's no differentiation — every location on every level fires.

**The evidence:** `eval.log:22-31` — `score_p50 ≈ 0.0 / 1e-39` in the EMA checkpoint, but step-0 training log shows cls_loss=1.07×10^7 (not −89 median logit). The raw model at step 0 IS saturated positive, not collapsed-to-zero. The collapse-to-zero happens later as the head learns to output −∞ everywhere to escape the enormous loss — which is the "recovery" that R9 misinterpreted as a transient spike.

### Q2: Activity Collapse After RC-19 Fix — Is Sigmoid Enough?

**Answer: (a) Zero det_conf during recovery, combined with trunk fix.** Sigmoid bounding (P7) maps [−∞,∞] → [0,1], but when the pre-sigmoid logits are all +50 to +100, sigmoid(+50) ≈ 1 − 2×10^−22, which is 1.0 in float32. The 24 det_conf dims are still near-identical (all ≈ 1.0). D4's finding of 0.35% per-dim variance confirms this — that variance is just float32 rounding noise.

**Recommendation:** For the recovery run, set `det_conf = torch.zeros(B, 24, device=...)` or gate it: `if not recovery_phase: det_conf = ...`. Once the detection head produces meaningful (differentiated) sigmoid outputs on actual frame content, re-enable det_conf conditioning. Option (b) (max score + which class) adds complexity without solving the root cause. Option (c) (threshold gate) is a band-aid.

### Q3: FeatureBank / RC-18 — Where Does D6's Token Differentiation Come From?

**Answer: GAP features (c5_mod + p4) provide the differentiation; VideoMAE contributes nothing.**

D5 proved VideoMAE features are dormant (zero delta). D6 proved tokens ARE differentiated (off-diag mass=0.925, entropy=1.992). The resolution: the activity head's input is `[det_conf(24d, constant) ‖ GAP(C5_mod2)(768d) ‖ GAP(P4)(256d)]`. The 768+256=1024 dims from the CNN pathway differ per-frame because the backbone IS alive (pose head works). The VideoMAE stream (384d, zeroed at eval, dead at train) adds another 384 constant dims.

With RC-25, the GAP features also have elevated magnitude, but they at least vary across frames. The 24 det_conf dims are the problematic ones — constant AND huge. FeatureBank returning the current frame 16× means the TCN+ViT see 17 near-identical tokens, but the tokens are near-identical because (a) FeatureBank returns the same frame and (b) even across different frames, the det_conf component is constant.

**Action:** Fix trunk first, then pass real video_ids to engage FeatureBank, then assess whether FeatureBank + TCN + ViT architecture is worth keeping vs. replacing with a K400-fine-tuned video encoder.

### Q4: 5% Subset Ceiling (RC-24) — Dominant Factor or Not?

**Answer: RC-24 is real but NOT the dominant factor. RC-25 is.**

The 5% subset (4 recordings, 3,112 frames, 12/75 activity classes) IS structurally insufficient for final metrics. But it is NOT the reason the model produces zero on all three functional heads. Evidence:

- A healthy detection head on 4 recordings should still learn SOMETHING — mAP ≥ 0.05, not 0.0000
- A healthy activity head should predict more than 1 class — even 4/12 seen classes would be progress
- The 10^7 cls_loss at step 0 is independent of subset size — it's a feature-magnitude problem
- The same trunk explosion would produce the same saturated logits on 25%, 50%, or 100% subsets

**RC-24 is a ceiling problem, not a floor problem.** RC-25 is the floor problem that prevents ANY learning. Fix RC-25 first, then scale up the subset.

### Q5: Training Dynamics After Reinit — Is the Setup Viable?

**Answer: Not from this checkpoint.** The 10^7 initial cls_loss is NOT a "transient Adam spike" — it's a structural incompatibility between the blown-up trunk and the freshly-initialized head. R9 ("Adam resume spike is transient, don't kill it") must be rejected.

Evidence against the "transient" interpretation:
1. At step 100, cls_loss is still 8.94×10^6 (log line) — not recovering
2. At step 500, cls_loss = 302.98 — the head has "recovered" by learning to output −∞ everywhere, which IS the collapse
3. The "EMA vs Raw" comparison at train.log:649 shows the raw model had psr_f1=0.0909 which EMA destroyed — but 0.0909 is still collapsed (1 pattern, wrong pattern)
4. Activity head REGRESSED from 3→1 classes over the epoch — the head is not learning, it's being crushed

**The frozen backbone at epoch 43 has weights optimized to produce features for collapsed heads.** Those features are pathologically scaled. A fresh head cannot learn from them because (a) the magnitude saturates the loss surface flat and (b) the features encode no useful signal about the collapsed tasks.

### Q6: Priority for Second Recovery Attempt — Zero-GPU Fixes First?

**Answer: Yes. Three zero-GPU diagnostics (D7-D9) before any GPU time.**

Before committing to a 25% subset retrain:
1. **D7: Feature-magnitude probe** — measure per-channel μ, σ, max for C2-C5, P3-P7 from latest.pth vs an ImageNet-init control. This confirms RC-25 and locates the blast radius (FPN only vs backbone).
2. **D8: Step-0 logit percentiles** — run one forward pass with reinit'd head, record p50/p90/p99/p100 of cls_logits before sigmoid. If p99 > 50, RC-25 is confirmed.
3. **D9: Per-layer weight-norm ratios** — compare ‖W‖_F for each conv in fpn.* and backbone.* between latest.pth and ImageNet init. Ratios > 5× indicate blown-up layers.

These take minutes of CPU time and determine whether we go Branch A (FPN-only reinit) or Branch B (fresh start).

**Additional zero-GPU fix:** Zero det_conf during recovery (see Q2).

---

## 3. Revised Recovery Plan

### Phase 0: Zero-GPU Diagnostics (D7-D9) — EST. 10 MINUTES CPU

All three scripts load `latest.pth` vs ImageNet-init control, no GPU needed.

**D7 — `diag_feature_magnitude.py`:**
```python
# For each level C2-C5, P3-P7, compute:
#   per-channel mean, std, max across spatial dims
#   Compare latest.pth vs torchvision convnext_tiny(pretrained=True)
# Flag: any level where |μ| > 5 or σ > 10 or max > 50
```

**D8 — `diag_step0_logits.py`:**
```python
# Load latest.pth, reinit detection head, run ONE forward pass
# Print p50, p90, p95, p99, p100 of cls_logits (pre-sigmoid)
# Verdict: p99 > 50 → RC-25 CONFIRMED
```

**D9 — `diag_weight_norms.py`:**
```python
# For every conv weight in fpn.* and backbone.*:
#   Compute ||W||_F / ||W_init||_F
# Print top 20 layers by ratio
# Verdict: any ratio > 5 → that layer is blown up
```

### Phase 1: Branch A — FPN-Only Blowup (IF D9 shows backbone healthy)

1. Add `fpn.lateral_convs.*.conv`, `fpn.fpn_convs.*.conv`, `fpn.smooth_convs.*.conv` to the reinit list in `_reinit_dead_heads`
2. Add GroupNorm after each detection subnet conv block (model.py:499-506):
   ```python
   # After each Conv3x3+ReLU in cls_subnet and reg_subnet:
   nn.GroupNorm(num_groups=8, num_channels=256)
   ```
3. Zero det_conf for activity head input during recovery
4. Add step-0 assertion:
   ```python
   assert cls_loss.item() < 1e4, f"RC-25: cls_loss={cls_loss.item():.1e} — trunk features exploded"
   assert cls_logits.abs().median() < 8, f"RC-25: median |logit|={cls_logits.abs().median():.1f}"
   ```
5. 1 epoch, 5% subset, frozen backbone, reinit'd FPN+heads
6. **Gate:** det_mAP50 ≥ 0.05 after 1 epoch → proceed. Otherwise → Branch B.

### Phase 2: Branch B — Fresh Start (IF D9 shows backbone blown, OR Branch A gate fails)

1. Abandon the epoch-43 lineage. Start from `convnext_tiny(pretrained=True)`.
2. Keep ALL 11 patches (P1-P11) in the codebase — they fix real bugs.
3. Add GroupNorm to detection subnets (same as Branch A).
4. Add step-0 assertion (same as Branch A).
5. Zero det_conf for activity head during early training.
6. **First run: detection-only on annotated + synthetic frames, 3-5 epochs.**
   - `PRETRAIN_DET_ON_SYNTH=True`, stage 1 only, backbone frozen first 2 epochs
   - **Gate: mAP50 ≥ 0.3 on b-boxed val** — proves the detection pipeline works
7. Once detection gate passes: add pose + headpose (stage 2, 5 epochs)
8. Once spatial heads are stable: add activity + PSR with embedding cache (stage 3)
9. Scale subset: 5% → 25% → 100% as metrics improve

### Phase 3: Permanent Architectural Guards

These go into the codebase regardless of which branch succeeds:

1. **GroupNorm in detection subnets** (model.py:499-506) — prevents trunk excursions from silently saturating the head. GroupNorm normalizes within each sample, so even if a particular frame's features are 10× normal, the head sees normalized input.

2. **Step-0 assertion** (training/train.py, after first forward pass):
   ```python
   if step == 0 and cls_loss is not None:
       assert cls_loss.item() < 1e4, \
           f"FATAL: det cls_loss={cls_loss.item():.1e} at step 0. " \
           f"Trunk features are likely exploded (RC-25). " \
           f"Run D7-D9 diagnostics before retraining."
       median_logit = cls_logits.detach().abs().median().item()
       assert median_logit < 8, \
           f"FATAL: median |logit|={median_logit:.1f} at step 0. " \
           f"Detection head receiving saturated features."
   ```

3. **Remove or reduce defensive machinery:**
   - Keep `ASSERT_AND_CRASH=True` (already enabled)
   - Remove loss caps, NaN guards, sensitivity penalty caps that hide problems
   - `SIMPLIFIED_LOSS=True` with fixed weights (already configured)

---

## 4. Re-Interpreted Diagnostic Results Under RC-25

| Original Finding | RC-25 Re-Interpretation |
|-----------------|------------------------|
| D4: det_conf near-constant (0.35% variance) | Not just RC-19 (raw logits). Even after sigmoid, logits are all +50→+100, so sigmoid ≈ 1.0 for all. Fix: solve trunk, not just sigmoid. |
| D5: VideoMAE dormant (zero delta) | CNN features are O(10²–10³); VideoMAE features are O(1). Delta is zero because CNN term dominates numerically. Re-test D5 after trunk fix. |
| D3: p5 fires 100%, p6 silent | p5 features have largest spatial extent and highest magnitude. With saturated logits, p5 dominates simply because it has the most anchors. Not an anchor-calibration issue. |
| Retrain log: cls_loss 10^7 → 300 | The "recovery" is the head learning to output −∞ everywhere — that IS the collapse, not the fix. |
| R9: "Adam resume spike is transient" | REJECTED. The spike is structural. The head can only reduce loss by collapsing to all-negative. |
| Activity: 3→1 classes over epoch | The head starts with slight differentiation (3 classes) from random init, then is crushed by the constant det_conf signal + dead FeatureBank + label noise. |

---

## 5. Priority-Ranked Next Actions

### TODAY (Zero GPU)

| # | Action | Time | Blocks |
|---|--------|------|--------|
| 1 | Write and run D7 (feature magnitude probe) | 15 min | Decision A vs B |
| 2 | Write and run D8 (step-0 logit percentiles) | 10 min | Confirms RC-25 |
| 3 | Write and run D9 (weight-norm ratios) | 10 min | Locates blast radius |
| 4 | Add GroupNorm to detection subnets | 5 min | Permanent guard |
| 5 | Add step-0 assertion to train.py | 5 min | Permanent guard |
| 6 | Zero det_conf during recovery (config flag) | 5 min | Activity head isolation |

### TOMORROW (GPU, based on D7-D9 results)

**If D9 shows FPN-only blowup → Branch A:**
| # | Action | GPU Time | Gate |
|---|--------|----------|------|
| 7 | Reinit FPN + heads, 1 epoch, 5%, frozen backbone | ~30 min | mAP50 ≥ 0.05 |
| 8 | If gate passes: 25% subset, 3 epochs, all heads | ~3 hr | mAP50 ≥ 0.10, act ≥ 10 classes |

**If D9 shows backbone blowup → Branch B:**
| # | Action | GPU Time | Gate |
|---|--------|----------|------|
| 7 | Fresh ImageNet init, det-only synthetic pretrain, 3 ep | ~1 hr | mAP50 ≥ 0.3 |
| 8 | Real fine-tune, det + pose, 5 ep | ~1.5 hr | mAP50 ≥ 0.4 |
| 9 | Add activity + PSR from embedding cache, 20 ep | ~2 hr | act ≥ 20 classes, psr ≥ 5 patterns |

### THIS WEEK

| # | Action |
|---|--------|
| 10 | 25% subset, 15 epochs, all patches + guards, monitor step-0 assertion |
| 11 | Re-run D2-D6 on healthy checkpoint |
| 12 | If metrics are alive: scale to 100% subset, 100 epochs |

---

## 6. Additional Root Causes Discovered

### RC-25: Feature-Magnitude Explosion in Shared Trunk (NEW — CRITICAL)

**Location:** FPN lateral/smooth convs and possibly backbone stage 3-4 convs  
**Evidence:** cls_loss = 1.07×10^7 at step 0 after reinit (requires input magnitude 10²–10³)  
**Mechanism:** 43 epochs of training under AMP, NaN events, Kendall clamp-after-backward, and collapsed-head gradients produced pathologically scaled weights in the shared trunk  
**Fix:** D7-D9 locate blast radius → reinit affected layers or start fresh

### RC-26: R9 "Adam Spike Is Transient" Is Wrong (REJECTION)

**Location:** Memory: `feedback_optimizer_state_resume_spike.md`  
**Mechanism:** The "recovery" at step ~100 (cls_loss 10^7 → 300) is the head learning to output all-negative logits to escape the saturated regime. This IS the collapse, not its resolution.  
**Fix:** Delete or correct the memory entry. The spike is diagnostic — if cls_loss > 1e4 at step 0, the run is structurally broken, not transient.

### RC-27: GroupNorm Absent from Detection Subnets (DESIGN GAP)

**Location:** model.py:499-506 (cls_subnet), model.py:526-532 (reg_subnet)  
**Mechanism:** 4× Conv3×3+ReLU with no normalization. Kaiming init preserves variance through the stack, but any upstream magnitude excursion passes through unattenuated. BatchNorm would be wrong (batch=1), but GroupNorm provides per-sample normalization — exactly what's needed for a multi-task model where one head's gradient spike could perturb shared features.  
**Fix:** Add GroupNorm(8, 256) after each conv in cls_subnet and reg_subnet.

---

## 7. What the 25% Retrain Would Have Shown

If we had run the 25% subset, 3-epoch retrain from the epoch-43 checkpoint WITHOUT fixing RC-25:

- **Detection:** Same saturated logits at step 0 (cls_loss ~10^7), same collapse to all-negative over 1-2 epochs. More GT boxes on 25% might produce a few positive gradients, but the flat loss surface from saturated negatives would prevent meaningful learning.
- **Activity:** Same regression to 1 class. More activity classes present (35 vs 12) but the constant det_conf + dead FeatureBank prevent differentiation.
- **PSR:** Same 1-pattern output. More recordings but same fill-forward label structure.
- **Combined:** Marginally higher (maybe 0.12-0.15) from pose only, with more frames.
- **The run would be ~3 hours of GPU wasted proving what D7-D9 can prove in 10 minutes of CPU.**

This is why the zero-GPU diagnostics come first.

---

## 8. Summary

1. **RC-25 is the dominant factor** — feature-magnitude explosion in the shared trunk saturates every downstream head
2. **All 11 patches are correct and necessary** — they fix real bugs that matter once the trunk is healthy
3. **The epoch-43 checkpoint may be salvageable** if D9 shows blowup is FPN-only (Branch A), but backbone blowup means fresh start (Branch B)
4. **Three zero-GPU diagnostics (D7-D9)** determine the path in 10 minutes
5. **Two permanent guards** (GroupNorm + step-0 assertion) prevent this class of failure from recurring
6. **RC-24 (subset size) is not the blocker** — fix RC-25 first, then scale data
7. **R9 (Adam spike is transient) is wrong** — delete that memory

**The path to a model that learns: D7-D9 → Branch A or B → detection gate → multi-task → scale data. Start with D7.**
