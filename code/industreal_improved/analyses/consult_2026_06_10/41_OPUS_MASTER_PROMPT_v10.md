# 41 — OPUS MASTER PROMPT v10: The Complete Overview (2026-06-21)

> **Purpose**: Single self-contained file to upload to Opus. Everything Opus needs to understand the project, the current 6-epoch plateau, what we've tried, what we know, and what we need. No other files required.
>
> **Current state**: RF2 epoch 21 (~50%), PID 3791482, mAP50 plateau at 0.204-0.215 for 6 consecutive epochs (15-20). CosineAnnealing LR restart at epoch 20 produced ZERO effect. Training continues but progress is flat.
>
> **The meta-question after 15 phases, 9 Opus consultations, ~500 GPU-hours**: Is the ceiling structural (data quality, label noise, architectural limit) or dynamic (training configuration, loss weighting, optimization)?

---

## 1. Project Identity (30-Second Summary)

**POPW** = Pose-Conditioned Multi-Task Architecture for Egocentric Assembly Understanding.
Single RGB frame → 5 simultaneous predictions on one RTX 3060 (12 GB).

| Component | Details |
|-----------|---------|
| Backbone | ConvNeXt-Tiny (28M params), ImageNet-pretrained |
| Neck | FPN P3-P7, 256ch |
| Detection | RetinaNet-style, 24 classes × 9 anchors/location, Focal(α=0.25,γ=2) + GIoU |
| Body Pose | 17 keypoints, Wing loss — **NO annotations available, loss=0 always** |
| Head Pose | 9-DoF (forward/position/up), MSE × 0.001 |
| Activity | Feature Bank(T=16) + TCN + 2×ViT + CLS token, 75 classes LDAM-DRW |
| PSR | Causal Transformer(3L,4H), 11 binary components, Binary Focal — **never trained** |
| Total params | 76.16M (53.42M trainable) |
| Conditioning | PoseFiLM + HeadPoseFiLM (stop_grad) modulate backbone C5 |

**Training**: AdamW (differential LR: backbone 0.1×, heads 1×, bias 0.3×), Warmup(5ep) → CosineAnnealingWarmRestarts(T₀=10, T_mult=2), EMA(decay=0.999), effective batch=32 (physical=1, grad_accum=32).

---

## 2. The Journey (Ultra-Condensed)

### Phase 1-8 (April–June 13): Everything Broke
- **All heads collapsed** in first training runs (NaN, frozen outputs, dead FeatureBank)
- **RC-25→RC-29 fixes**: AMP→FP32, empty-frame gradient fix, DET_PROBE/LIVENESS diagnostics
- **3-way deadlock discovered**: detection-only training produces ~4×10⁻⁵ gradient per backbone param per step — below FP32 noise floor
- **R2.5 paradox**: R2.5 (all heads) trained visibly well but RF1 (detection-only) died immediately. Resolution: R2.5 had 10,000× denser gradient from 4 active heads

### Phase 9-10 (June 13-17): Kendall Bug Discovery
- **Kendall bug** (`losses.py:1589`): `elif self.train_pose:` branch excluded `loss_head_pose` from total loss. Head pose computed (~1.7 loss) but added ZERO gradient for 7+ epochs.
- **Fix confirmed**: head_pose gradient → ALIVE, cls_std 1.6× broader, cls_max 5.9× higher
- **RF1 completed** with `best_det_mAP50=0.184` (genuine metric after phantom-0.45 correction)

### Phase 11-12 (June 18-20): RF2 Epoch 15 Collapse
- RF2 launched with head_pose ALIVE + Kendall fix + DET_GT_FRAME_FRACTION=0.90
- mAP50 peaked at 0.184 (epoch 8) → declined → collapsed to ~0.001 by epoch 15
- **cls_score bias equilibrium**: classifier converged to uniform ~0.079 sigmoid scores
- **22-agent monitoring swarm deployed**: 134 checks/5-min cycle, 6 bugs found in first hours

### Phase 13 (June 20): Opus v8 — 4 Fixes Prescribed & Implemented
| Fix | Problem | Change |
|-----|---------|--------|
| KENDALL_HP_PREC_CAP | Kendall could zero head_pose weight | Clamp `lv_hp >= lv_det` |
| More positives | ~16 positive anchors/batch | IOU_THRESH=0.4, TOP_K=9 |
| BIAS_LR_FACTOR | 5× bias acceleration to collapse | 5.0 → 1.0 |
| Phantom 0.45 | Gate threshold stored as metric | Validation guard added |

### Phase 14-15 (June 20-21, NOW): The 6-Epoch Plateau
**The v8 fixes prevented collapse but created a new regime**: structural plateau.

```
Epoch  7: mAP50=0.007  MAE=71.67°  (run 2 start)
Epoch  8: mAP50=0.184  MAE=?       (all-time peak epoch)
Epoch  9: mAP50=0.181
Epoch 10: mAP50=0.159  MAE=20.73°
... old run collapsed epochs 11-15 ...
Epoch 16: mAP50=0.215  mAP50_95=0.081  MAE=8.80°  combined=0.462
Epoch 17: mAP50=0.204  mAP50_95=0.077  MAE=9.25°  combined=0.455
Epoch 18: mAP50=0.207  mAP50_95=0.078  MAE=9.27°  combined=0.456
Epoch 19: mAP50=0.209  mAP50_95=0.081  MAE=9.33°  combined=0.458
Epoch 20: mAP50=0.205  mAP50_95=0.080  MAE=9.23°  combined=0.455  ← LR RESTART (zero effect)
```

**Range**: 0.2039-0.2151 (1.1pp range over 6 epoch-ends). **Trend**: zero slope.

---

## 3. Current Training State

**Live status** (from `rf_stage_state.json` at epoch 21, batch 2300/3302):
```
PID:          3791482 (restarted — new PID from previous)
Stage:        rf2 (stage_index=1)
Status:       RUNNING
Epoch:        21/36 (~58%)
best_metric:  0.462 (combined: det_mAP50 + MAE + loss components)
best_metrics: det_mAP50=0.2047, forward_angular_MAE_deg=9.23°
gate_passed:  false
Heartbeat:    2026-06-21T07:23 UTC (updating — training active)
LIVENESS:     det ALIVE[2.35e-02], backbone ALIVE[2.770e+00], head_pose ALIVE[4.83e-03]
```

**DET gradient bottleneck**: detection_head grad 2.35e-02 vs backbone 2.770e+00 — **117× ratio**. Classification gradient is a tiny fraction of total backbone gradient.

**head_pose borderline**: Alternates ALIVE (4.83e-03) and DEAD (5.34e-04) at different probes. The HP_PREC_CAP clamp may be barely holding.

---

## 4. The Decisive Evidence (What Changed Our Understanding)

### 4.1 POS_ANCHOR_PROBE — Classifier IS Working on Matched Positives

At epoch 21 step 1600-1700:
```
POS_ANCHOR_PROBE img=0 call=204800: n_pos=525 mean=0.646 med=0.638 max=0.994
POS_ANCHOR_PROBE img=1 call=205000: n_pos=346 mean=0.732 med=0.757 max=0.998
POS_ANCHOR_PROBE img=0 call=205200: n_pos=476 mean=0.799 med=0.851 max=0.993
POS_ANCHOR_PROBE img=0 call=205400: n_pos=164 mean=0.754 med=0.754 max=0.991
```

**The classifier scores positive anchors at mean=0.64-0.80, median=0.64-0.85, max=0.99.** This refutes the "classifier is collapsed" narrative. The classifier CAN confidently predict correct classes for anchors it matches to GT.

### 4.2 Pseudo-Classing mAP = ~50% Above Raw mAP

`det_mAP50_pc` (pseudo-classing) sits at 0.307-0.344 versus raw mAP50 at 0.204-0.215. Pseudo-classing averages per-class AP treating each class independently. That it's +50% above raw mAP confirms **the problem is class-specific, not uniform**.

### 4.3 LR Restart Had ZERO Effect

CosineAnnealingWarmRestarts at epoch 20 (T₀=10 restart, LR reset to max, optimizer state re-init):
```
Epoch 19: mAP50=0.2088  mAP50_95=0.0810
Epoch 20: mAP50=0.2047  mAP50_95=0.0795
```
Change: −0.0041 mAP50, −0.0015 mAP50_95. Statistically zero. This eliminates "stuck in local minimum at low LR." The plateau is **structural**, not schedule-dependent.

### 4.4 Opus v9 Corrections (3 Critical Diagnoses)

1. **score_p50 is structurally blind**: With ~172K anchors/image and a handful of GT, >99.99% are background. The median anchor's score ≈ sigmoid(bias) regardless of classification quality. A perfectly trained detector shows the same score_p50. **All prior analysis using score_p50 as a classification metric was invalid.**

2. **LOCALIZING verdict is IoU-only**: DET_PROBE checks box overlap at score>0.01, never evaluates predicted class. A model putting wrong-class predictions on correct boxes at IoU 0.9 scores LOCALIZING and mAP=0 simultaneously. Consistent with "classification is fine" AND "classification is dead."

3. **detach_reg_fpn split-brain (UNRESOLVED)**: The committed `stage_rf2` preset has `detach_reg_fpn=True` (config.py:1109) but documentation claims False. If True, only classification subnet shapes the backbone → excellent localization produced by reg subnet riding on CLS-carved features → points AWAY from "bad features" and AT the cls loss/cls targets/labels. **This must be verified by printing effective value at step 0.**

### 4.5 The Top-k IoU Floor Problem

`DET_POS_IOU_TOP_K=9` has **no minimum-IoU guard**. For GT boxes whose best anchors sit at IoU~0.2 (small parts vs ANCHOR_SIZES starting at 96px), the code force-assigns 9 poorly-localized anchors to predict class c with target 1.0. Classification is **actively mistaught** that features at 0.2-IoU locations are class c. This may CREATE uniform-output pathology for small/medium objects — trading gradient starvation for label noise.

### 4.6 The 12/24 AP=0 Classes

**det_n_present_classes = 15-16/24** every epoch. The same 8-9 classes never register a single correct detection. Class 6 has 1500-1800 GT instances per epoch yet AP=0 everywhere. This has never been investigated.

---

## 5. The 5 Most Important Open Questions

### Q1: Why Are 12/24 Classes Always at AP=0?

**Evidence**: Systematic across all epochs, all runs. Pseudo-classing +50% confirms class-specific. Class 6 has abundant GT (1500+/epoch) yet AP=0.

**Hypotheses** (mutually testable, not exclusive):
1. **Label error**: Class 6 (and similar) labels are systematically wrong. If true: fix labels → mAP jumps ~0.30-0.35.
2. **Anchor mismatch**: No anchor achieves IoU>0 with class 6 GT boxes. ANCHOR_SIZES start at 96px — assembly parts may be smaller.
3. **Feature confusion**: Similar-looking classes are consistently confused. 12 AP=0 + 12 working classes = possible confusion pairs.
4. **Top-k poisoning**: Class-specific effect where small objects get force-assigned near-random anchors (IoU~0.2), poisoning classification targets.

**What we need**: Per-class AP from metrics.jsonl, anchor-IoU histogram per class, visual audit of class 6 labels.

### Q2: Is DETACH_REG_FPN True or False in the Running Config?

Committed config says `True`. Documentation says `False`. These imply completely different diagnoses:
- If **True**: Only classification (and head_pose) shape backbone. Excellent localization (bestIoU=0.86-0.98) comes from reg subnet riding on features carved by classification. Points to label/target problem.
- If **False**: Classification AND regression both shape backbone. Regression prefers tight localization even at cost of class features → may explain classifier plateau.

**Fix**: Print `C.DETACH_REG_FPN` at step 0. This is a 1-line change.

### Q3: Is the Ceiling Structural (Data/Architecture) or Dynamic (Training Config)?

**Evidence for structural**: LR restart had zero effect (eliminates "stuck in local min"). 6 epochs of flat trend. Three independent training regimes converged to similar mAP (~0.15-0.21). RF1's real best (0.184) matches RF2 peak.

**Evidence against structural**: POS_ANCHOR_PROBE shows classifier CAN learn on positive anchors. Pseudo-classing mAP=0.344 suggests the "ceiling" may be mostly 12 AP=0 classes dragging average down.

**The decisive experiment**: 50-image cls-only overfit (Opus v9 §6, <30 min). If mAP→0.8+, the architecture CAN work — plateau is dynamic. If mAP stalls, the ceiling is in the assignment/loss/target pipeline.

### Q4: Does the Top-k IoU Floor Explain the Class-Specific Failure?

**The mechanism**: `DET_POS_IOU_TOP_K=9` without IoU floor force-assigns noisy targets. For classes where objects are smaller than ANCHOR_SIZES[0]=96px, ALL 9 best anchors may be at IoU<0.3 → classifier trained on near-random features.

**Testable**: Add `topk_min_iou=0.2` guard to TOP_K matching. Rerun for 5 epochs. If AP=0 classes recover, this was the issue.

### Q5: Should We Advance to RF3 Now or Fix RF2 First?

**Arguments for advance**: RF3 adds activity head (dense 75-class gradient), which may improve shared representations. Activity head's ViT attends to spatial features — may break the classifier equilibrium naturally. Paper needs activity results. We have ~15 remaining RF2 epochs (~22h) producing nothing new.

**Arguments against**: RF2 hasn't reached its gate (mAP50>=0.40). If RF2's classifier plateau is structural, RF3 inherits the same broken detector. RF3 adds complexity (activity head, 35% subset) without fixing the root cause. If the ceiling is label noise, RF3 won't help.

**Recommendation needed**: Decision matrix based on risk/upside.

---

## 6. Experiments Recommended (In Priority Order)

| # | Experiment | Time | What It Bins | Risk |
|---|-----------|------|-------------|------|
| 1 | **Print DETACH_REG_FPN at step 0** | 5 min | Entire diagnosis changes if True vs False | None |
| 2 | **50-image cls-only overfit** | <30 min | If mAP≥0.8 → dynamics issue; if stalls → data/assignment | None (separate script) |
| 3 | **Per-class AP parsing** | 1 hour | Confirms/refutes 12/24 AP=0 hypothesis definitively | None (read metrics.jsonl) |
| 4 | **Anchor-IoU histogram per class** | 1 hour | Confirms/refutes anchor mismatch hypothesis | None |
| 5 | **Top-k IoU floor (min_iou=0.2)** | ~7h (5 epochs) | If AP=0 classes recover → poisoning confirmed | May reduce total positives |
| 6 | **Log cls_score.weight.norm()** | 1 line code | Detects weight collapse vs healthy divergence | None |
| 7 | **Visual audit of class 6 labels** | 30 min | Answers label noise question definitively | None |
| 8 | **PSR 50-sequence overfit** | <1 hour | De-risks paper's novelty claim before RF4 | None |

---

## 7. What We Need From Opus

**Please provide**:

1. **Your unified diagnosis** of the 6-epoch plateau given ALL the evidence above (POS_ANCHOR_PROBE showing classifier works on positives, LR restart failure, 12/24 AP=0 classes, pseudo-classing +50%, DET gradient bottleneck 117×, head_pose borderline ALIVE/DEAD, score_p50 blindness, potential detach_reg_fpn=True).

2. **Decision: advance to RF3 or fix RF2 first?** With explicit risk/benefit analysis. Consider that RF3 adds activity head (75-class, dense gradient, VIT attention) but inherits the same detection head. Consider remaining budget (~15 epochs at 86min each = ~22h).

3. **Which experiments to run and in what order**, given that experiments 1-4 (from the table above) can run in parallel to the live training, while experiment 5 would require stopping the current run.

4. **If label noise IS the ceiling** (the uncomfortable meta-question): What does "fix the labels" actually mean for this project? How much effort, what process, what class of improvement to expect? Is there a way to confirm this without a full re-labeling effort?

5. **Is the combined metric (0.462) dangerously misleading?** It shows "improvement" from epoch 16-21 that is entirely driven by MAE and loss components, not detection. Should we stop reporting it as the primary best_metric?

6. **What specific missing measurement would be decisive right now** — the one thing we haven't measured that would break the logjam if we knew it?

---

## 8. Appendix: Raw Data Snapshot

### 8.1 Current RF2 Config (from stage_rf2 preset + overrides)

| Parameter | Value | Parameter | Value |
|-----------|-------|-----------|-------|
| subset_ratio | 0.50 | DET_POS_IOU_THRESH | 0.4 |
| train_det | True | DET_POS_IOU_TOP_K | 9 |
| train_head_pose | True | DET_BIAS_LR_FACTOR | 1.0 |
| train_act | False | DETACH_REG_FPN | **???** (commit=True, doc=False) |
| train_psr | False | KENDALL_HP_PREC_CAP | True |
| DET_GT_FRAME_FRACTION | 0.90 | EMA decay | 0.9999 |
| reinit_pi | 0.05 | DET_EVAL_SCORE_THRESH | 0.001 |

### 8.2 Key Metrics Over Last 6 Epochs

| Epoch | mAP50 | mAP50_95 | mAP50_pc | MAE° | Combined | n_classes |
|-------|-------|----------|----------|------|----------|-----------|
| 15 | — | — | — | — | — | — |
| 16 | 0.2151 | 0.0810 | 0.3442 | 8.80 | 0.4622 | 15-16 |
| 17 | 0.2039 | 0.0770 | 0.3071 | 9.25 | 0.4547 | 15-16 |
| 18 | 0.2072 | 0.0782 | 0.3095 | 9.27 | 0.4564 | 15-16 |
| 19 | 0.2088 | 0.0810 | 0.3132 | 9.33 | 0.4580 | 15-16 |
| 20 | 0.2047 | 0.0795 | 0.3071 | 9.23 | 0.4553 | 15-16 |

### 8.3 Gradient/Liveness at Epoch 21

| Measurement | Value | Status |
|------------|-------|--------|
| det LIVENESS_GRAD | 2.35e-02 | ALIVE (but low — was 0.92-1.57 at epoch 17) |
| backbone LIVENESS_GRAD | 2.770e+00 | ALIVE (healthy) |
| head_pose LIVENESS_GRAD | 4.83e-03 | ALIVE borderline (alternates with DEAD 5.34e-04) |
| DET gradient ratio (head/backbone) | 0.0235 / 2.770 = **0.0085 (0.85%)** | Detection contributes <1% of backbone gradient |
| POS_ANCHOR_PROBE n_pos | 164-525 per probe | OK |
| POS_ANCHOR_PROBE mean | 0.64-0.80 | OK — classifier working on matched positives |
| POS_ANCHOR_PROBE max | 0.99 | OK |

### 8.4 Known Open Config Issues

- **detach_reg_fpn**: Committed True, doc says False. **UNRESOLVED.**
- **metric_history not updating**: state.json only shows epochs 7-10 (from previous run). Epochs 16-21 never recorded despite heartbeat updating.
- **DET_EVAL_SCORE_THRESH=0.001**: Extremely low threshold means mAP includes near-random predictions. Effect unknown.
- **best_metric=0.462**: Combined metric mixes detection + MAE + losses. Misleading as "progress."

---

*End of overview. Ready for Opus consultation round 10.*
