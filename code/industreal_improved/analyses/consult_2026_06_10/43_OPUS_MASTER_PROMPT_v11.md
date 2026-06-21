# 43 — OPUS MASTER PROMPT v11: Post-Breakthrough Status (2026-06-21)

> **Purpose**: Single self-contained file to upload to Opus. Everything Opus needs to understand the project after the v10 breakthrough (`detach_reg_fpn=True` confirmed as primary cause of the 6-epoch plateau).
>
> **Current state**: RF2 epoch 21 (97%, batch 3200/3302), PID 3791482, mAP50=0.2047 (epoch 20). **detach_reg_fpn=False applied in working tree but NOT YET RESTARTED.** The config fix is ready — the restart decision is the next action.
>
> **The v11 meta-question**: How much will flipping `detach_reg_fpn=False` break the ceiling? Is the remaining gap fixable with training alone, or does class 6 (1739 GT, AP=0.0) reveal a data-quality ceiling beneath the config regression?

---

## 1. Project Identity (30-Second Summary)

**POPW** = Pose-Conditioned Multi-Task Architecture for Egocentric Assembly Understanding.
Single RGB frame → 5 simultaneous predictions on one RTX 3060 (12 GB).

| Component | Details |
|-----------|---------|
| Backbone | ConvNeXt-Tiny (28M params), ImageNet-pretrained |
| Neck | FPN P3-P7, 256ch |
| Detection | RetinaNet-style, 24 classes × 9 anchors/location, Focal(α=0.25,γ=2) + GIoU |
| Body Pose | 17 keypoints, Wing loss — **NO annotations, loss=0 always** |
| Head Pose | 9-DoF (forward/position/up), MSE × 0.001, Kendall-weighted |
| Activity | Feature Bank(T=16) + TCN + 2×ViT + CLS token, 75 classes LDAM-DRW |
| PSR | Causal Transformer(3L,4H), 11 binary components, Binary Focal — **never trained** |
| Total params | 76.16M (53.42M trainable) |
| Conditioning | PoseFiLM + HeadPoseFiLM (stop_grad) modulate backbone C5 |

**Training**: AdamW (differential LR: backbone 0.1×, heads 1×, bias 0.3×), Warmup(5ep) → CosineAnnealingWarmRestarts(T₀=10, T_mult=2), EMA(decay=0.999), effective batch=32 (physical=1, grad_accum=32). RTX 3060 12GB, ~86 min/epoch, ~0.9 batch/s.

---

## 2. The Journey (Ultra-Condensed)

### Phase 1-8 (April–June 13): Everything Broke
- All heads collapsed in first runs (NaN, frozen outputs, dead FeatureBank)
- RC-25→RC-29 fixes: AMP→FP32, empty-frame gradient fix, DET_PROBE/LIVENESS diagnostics
- 3-way deadlock: detection-only training produces ~4×10⁻⁵ gradient per backbone param per step
- R2.5 paradox resolved: 10,000× denser gradient from 4 active heads explained stable multi-head training

### Phase 9-12 (June 13-20): Kendall Bug + RF2 Collapse
- Kendall bug (losses.py:1589): head_pose computed but excluded from total loss for 7+ epochs
- Fix confirmed: head_pose gradient ALIVE, classifier statistics improved 1.6-5.9×
- RF1 completed at best_det_mAP50=0.184 (genuine, after phantom-0.45 correction)
- RF2 launched → epoch 8 peak (0.184) → collapse to 0.001 by epoch 15
- cls_score bias equilibrium: classifier at uniform ~0.079
- 22-agent monitoring swarm deployed

### Phase 13 (June 20): Opus v8 — 4 Fixes
| Fix | Change |
|-----|--------|
| KENDALL_HP_PREC_CAP | Clamp `lv_hp >= lv_det` — prevents head_pose Kendall from zeroing detection weight |
| DET_POS_IOU_THRESH=0.4, TOP_K=9 | More positive anchors (~16→~120/batch) |
| DET_BIAS_LR_FACTOR 5.0→1.0 | Remove bias acceleration toward equilibrium |
| Stage history validation guard | Phantom 0.45 gate bug fixed |

### Phase 14-15 (June 20-21): The 6-Epoch Plateau
**v8 fixes prevented collapse but created structural plateau**:
```
Epoch  7: mAP50=0.007  MAE=71.67°
Epoch  8: mAP50=0.184  MAE=?
Epoch  9: mAP50=0.181
Epoch 10: mAP50=0.159  MAE=20.73°
... old run collapsed epochs 11-15 ...
Epoch 16: mAP50=0.215  mAP50_95=0.081  MAE=8.80°  combined=0.462
Epoch 17: mAP50=0.204  mAP50_95=0.077  MAE=9.25°  combined=0.455
Epoch 18: mAP50=0.207  mAP50_95=0.078  MAE=9.27°  combined=0.456
Epoch 19: mAP50=0.209  mAP50_95=0.081  MAE=9.33°  combined=0.458
Epoch 20: mAP50=0.205  mAP50_95=0.080  MAE=9.23°  combined=0.455 ← LR RESTART: ZERO EFFECT
```
**Range**: 0.2039-0.2151 (1.1pp over 6 epochs). **Trend**: zero slope.

### Phase 16 (June 21, NOW): Opus v10 Breakthrough

**`detach_reg_fpn=True` confirmed for RF2.** The code trace:

| Layer | Value | Source |
|-------|-------|--------|
| stage_rf2 preset | `detach_reg_fpn: True` | config.py:1117 (committed) |
| RF2 stage_cfg override | NONE — no key in config | stage_manager applies defaults |
| CLI flag | NOT PASSED | launch log confirmed |
| **Effective** | **True** | Opus v10 verification |

**The smoking gun**: The backbone was shaped by classification + head_pose only — regression gradient was cut (`model.py:561`, `reg_feat = feat.detach()`). This produces a one-to-one symptom match:

| Symptom | detach=True Explains |
|---------|---------------------|
| bestIoU 0.86-0.98 (localizes well) | Regression subnet gets decent detached features |
| mAP 0.20 (classifier stuck) | Cls-shaped features not object-discriminative enough |
| 12/24 AP=0 | Feature-starvation produces class-selective collapse |
| LR restart zero effect | Detached gradient path is not a local minimum |
| POS_ANCHOR_PROBE 0.64-0.80 | Easy classes survive on cls-shaped features alone |

**Opus v10 caveat**: RF2 (detach=True, 2.5× data) reached mAP=0.204, **above** RF1 (detach=False) at 0.184. So detach is a **handicap partly compensated by v8 fixes + more data** — flipping it should move the ceiling but may not single-handedly clear 0.40.

---

## 3. Current Training State

**Live status** (from rf_stage_state.json at epoch 21, batch 3200/3302 — 97%):
```
PID:          3791482
Stage:        rf2 (stage_index=1)
Status:       RUNNING
Epoch:        21/36
best_metric:  0.462 (combined: 0.667·mAP50 + 0.333·(1/(1+MAE)))
best_metrics: det_mAP50=0.2047, forward_angular_MAE_deg=9.23°
gate_passed:  false
Heartbeat:    2026-06-21T07:50 UTC (updating)
```

**Gradient health** at epoch 21:
- detection_head: 2.35e-02 (ALIVE but low — 117× below backbone)
- backbone: 2.770e+00 (healthy)
- head_pose: 1.37e-02 (ALIVE — stable with HP_PREC_CAP clamp)
- psr/act: correctly frozen (grad_norm ≈ 0)

**The 6-epoch plateau** is the defining problem. 6 consecutive epoch-ends at mAP50=0.204-0.215 with zero trend. LR restart produced no change.

---

## 4. The Config Delta — What's in the Working Tree

**These changes are applied but NOT COMMITTED and NOT YET RESTARTED.** The current epoch-21 run was started before these fixes.

| Parameter | Old Value | New Value | Source | Opus Rec? |
|-----------|-----------|-----------|--------|-----------|
| detach_reg_fpn (stage_rf2) | True | **False** | config.py:1115 | ✅ Tier 1 (v10) |
| DET_POS_IOU_IOU_FLOOR | (not set) | **0.2** | config.py:304 | ✅ Opus v9 §R2 |
| DET_LR_MULTIPLIER | 1.0 | **2.0** | config.py:55 | ❌ NOT recommended by Opus |
| DET_BIAS_LR_FACTOR | 1.0 | **4.0** | config.py:56 | ❌ CONFLICTS with v8 (called 5× "own-goal") |

**⚠️ The LR_MULTIPLIER and BIAS_FACTOR changes are OUR additions** — not part of any Opus recommendation. Their rationale:
- LR_MULTIPLIER=2.0: "Head needs enough gradient to shift FPN toward detection. 1.0 was stagnant." Plausible but untested.
- BIAS_FACTOR=4.0: "IOU_FLOOR=0.2 prevents false-positive labels so bias can't cheat into equilibrium. 1.0 was too conservative."

**These may need to be reverted to 1.0** before restart if Opus advises it. They introduce an uncontrolled variable.

---

## 5. The Per-Class AP Findings (Parsed from metrics.jsonl)

**This is the first time we've read this data** — it was always being written, we never parsed it until today.

### Epoch 18 Per-Class AP:
```
AP>0 — 12 WORKING classes:
  Classes 4 (0.370), 5 (1.0), 7 (0.719), 10 (0.477), 12 (0.559), 17 (0.396), 21 (1.0)
  Classes 0 (0.020), 9 (0.074), 11 (0.135), 20 (0.126), 22 (0.079)

AP=0 WITH GT — 4 classes (MOST IMPORTANT):
  Class 6:  1739 GT instances, AP=0.0 — THE mystery class
  Class 8:  GT present, AP=0.0
  Class 13: GT present, AP=0.0
  Class 19: GT present, AP=0.0

AP=0 — NO GT IN 50% SUBSET — 8 classes:
  Classes 1, 2, 3, 14, 15, 16, 18, 23
```

**Key insight**: Classes 5 and 21 hit AP=1.0 with only 33 and 151 GT instances — distinctive rare objects. This proves the architecture CAN learn perfect classification given clean, distinctive labels.

**Class 6 is the litmus test**: 1739 GT instances (more than most working classes) yet AP=0.0 everywhere. If detach flip fixes class 6, the bottleneck was gradient starvation. If class 6 stays AP=0, it's a label/assignment problem independent of training dynamics.

---

## 6. What We've Decided (Post-v10 Updates)

| Question | Previous Status | Current Status |
|----------|---------------|----------------|
| Q35: detach_reg_fpn? | UNRESOLVED — split-brain | **RESOLVED** — True confirmed, fix applied |
| Q04: Bias collapse? | MEDIUM confidence | **REFRAMED** — bias was effect, not cause |
| Q31: 12/24 AP=0? | Not parsed | **PARSED** — class 6 (1739 GT, AP=0) is top mystery |
| Q40: Advance to RF3? | 3 options debated | **DECIDED** — restart RF2 with detach=False first |
| Q41: Will detach flip work? | Not asked yet | **NEW** — the central question of v11 |

---

## 7. The 8 Most Important Open Questions for Opus

### Q1: How Much Will detach_reg_fpn=False Move the Ceiling?

**The evidence**: Our range is 0.184 (RF1, detach=False) to 0.204 (RF2, detach=True with 2.5× data + v8 fixes). With detach flipped AND the v8 fixes AND 2.5× data AND IoU floor, what's the expected mAP?

**Our three scenario estimates**:
- **Best case** (30-40%): 0.35-0.45 — detach was the primary bottleneck
- **Moderate case** (40-50%): 0.25-0.30 — detach helped but labels/anchor matching are the new ceiling
- **Worst case** (10-20%): 0.20-0.22 — detach wasn't the bottleneck, or LR/BIAS multipliers counteract benefit

**What we need**: Opus's expected magnitude. Should we see improvement in 1 epoch or 3-4?

### Q2: Should We Revert DET_LR_MULTIPLIER and DET_BIAS_LR_FACTOR to 1.0 Before Restart?

**The tension**: Opus v8 called DET_BIAS_LR_FACTOR=5.0 "an own-goal — the bias momentum heading toward equilibrium was the collapse mechanism." We set it to 4.0 with the IOU_FLOOR justification. But:

- Is 4× still accelerating bias toward the same equilibrium even with the floor?
- Does DET_LR_MULTIPLIER=2.0 help (more gradient to head) or hurt (Focal Loss makes noisy top-k assignments worse at high LR)?
- Should we restart with BOTH at 1.0, observe for 2 epochs, then tune up?

### Q3: Why Is Class 6 (1739 GT, AP=0) Completely Broken While Classes 5 and 21 Hit AP=1.0?

**What we know**: Class 6 has 1739 GT instances per epoch — more than most working classes. Classes 5 (33 GT) and 21 (151 GT) hit AP=1.0. The split cannot be explained by data scarcity.

**Hypotheses in order of plausibility**:
1. **Label error**: Class 6 labels are systematically wrong. 30-minute visual audit needed.
2. **Anchor mismatch**: Class 6 objects have geometry incompatible with the anchor grid (ANCHOR_SIZES start at 96px).
3. **Feature confusion**: Class 6 is visually similar to another working class → consistent wrong predictions.
4. **Top-k poisoning**: Small objects get force-assigned near-random anchors (IoU~0.2).

**What Opus should weigh**: If detach flip fixes class 6, the answer is gradient starvation. If it doesn't, the answer is labels. How many epochs after restart should we wait before concluding?

### Q4: Should the 50-Image Cls-Only Overfit Still Be the First Diagnostic Step?

**Before v10**: This was the top priority (<30 min, removes multi-task confound). But:
- The problem is now diagnosed (detach_reg_fpn=True)
- The fix is applied but not restarted
- Running the overfit now would delay the restart by ~30 min
- The overfit can run IN PARALLEL (separate GPU process, tiny dataset)

**Decision needed**: Restart NOW (<5 min config change) or run overfit FIRST (<30 min), then restart?

### Q5: How Does detach_reg_fpn=False Affect RF3's Stage Config?

**Known**: RF3's stage_rf3 preset ALSO has `detach_reg_fpn: True` (config.py:1124+). If we advance to RF3, it inherits the same bug.

**Question**: Should we set detach_reg_fpn=False for ALL stages (RF2, RF3, RF4) globally, or per-stage? Is there any stage where detach=True is correct (reinit_heads=True stages)?

### Q6: Is the Combined Metric (best_metric=0.462) Actively Harmful?

The formula `0.667·mAP50 + 0.333·(1/(1+MAE))` means 1/3 of "progress" is head-pose. With MAE at 9.23° (near floor), the combined metric shows 0.462 — suggesting the model is "close" to the 0.50 gate. This is misleading.

**We've stopped treating it as primary.** Should we remove it entirely or just deprioritize?

### Q7: What's the Complete Config for the Restart?

Please provide the exact config values for the restart. Specifically:
1. detach_reg_fpn → False (confirmed)
2. DET_POS_IOU_IOU_FLOOR → 0.2 (confirmed)
3. DET_LR_MULTIPLIER → 1.0 or 2.0?
4. DET_BIAS_LR_FACTOR → 1.0 or 4.0?
5. Any other changes needed before restart?

### Q8: Per-Class AP After Restart — What's the Decision Rule?

**We propose**: 3-4 epochs post-restart:
- If class 6 wakes (AP>0) AND mAP climbs past 0.25 → detach was bottleneck. Continue to 0.40.
- If class 6 stays AP=0 AND mAP stays <0.25 → labels/anchor mismatch. Run 50-image overfit + label audit.
- If some dead classes wake but not class 6 → class-specific mechanism. Anchor-IoU histogram needed.

**Does this decision rule make sense?** What thresholds would Opus use?

---

## 8. Experiments & Priority

| # | Experiment | Time | What It Answers | When |
|---|-----------|------|-----------------|------|
| 1 | **Restart training with detach=False** | ~8h (3-4 ep) | Does detach flip break the ceiling? | NOW |
| 2 | **50-image cls-only overfit** | <30 min | Architecture's ceiling without multi-task | PARALLEL to #1 |
| 3 | **Visual class 6 label audit** | 30 min | Are class 6 labels wrong? | AFTER #1 if class 6 still AP=0 |
| 4 | **Anchor-IoU histogram per class** | 1h + 1 epoch | Anchor geometry mismatch? | AFTER #1 if dead classes persist |
| 5 | **Revert LR/BIAS to 1.0** | ~6h (3 ep) | Are our config changes helping or hurting? | IF #1 improves but stalls under 0.25 |

---

## 9. What We Need From Opus

**Please provide**:

1. **Your expected mAP and timeline** from flipping detach_reg_fpn. How much, how fast?

2. **Should we revert DET_LR_MULTIPLIER and DET_BIAS_LR_FACTOR to 1.0 before restart**? Or is 2.0/4.0 safe given the IOU_FLOOR guard?

3. **The exact restart config** — a complete list of every non-default setting for the RF2 restart.

4. **Class 6 diagnosis** — given 1739 GT at AP=0 alongside classes 5/21 at AP=1.0, what's the most likely cause? Should we visually audit labels before or after restart?

5. **The 50-image overfit priority** — restart first (and risk wasting 6h if architecture can't learn) or overfit first (and delay restart by 30 min)?

6. **RF3 config** — should detach_reg_fpn be False in ALL stage presets, or are there stages where True is correct?

7. **Decision rule for "was the fix enough?"** — what specific metric threshold and epoch count confirms success vs. reveals a second ceiling?

---

*End of overview. Ready for Opus consultation round 11. 43 files in the consultation folder, 15 phases, 10 prior Opus consultations, ~500 GPU-hours of investigation. The fix is identified and applied — the question is whether it's sufficient.*
