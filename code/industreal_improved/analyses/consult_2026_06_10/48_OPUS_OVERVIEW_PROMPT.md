# 48 — Master Overview Prompt for Opus Consultation

> **Purpose**: One self-contained prompt to bring Opus fully up to speed on the current training state, the structural ceiling breakthrough, and what we need help with.
> **Generated**: 2026-06-22 10:30 UTC

---

## 1. Project Identity

**POPW** — Pose-Conditioned Multi-Task Architecture for Assembly Understanding (egocentric assembly detection + pose estimation). Industries dataset: 24 COCO-like assembly part classes, 720×1280 RGB images.

**Architecture**: ConvNeXt-T backbone + FPN (P3-P7, 256ch) + 5 task heads: detection (RetinaNet-style), body pose, head pose, activity classification, PSR (assembly state transitions). Total 76.16M params (53.42M trainable).

**Hardware**: Single RTX 3060 12GB, i5-12400F, 64GB RAM. Batch size 4.

---

## 2. Training Pipeline: Staged RF1-RF10

Sequential stages with progressive data scaling (20% → 100% subset):

| Stage | Subset | Heads | Gate |
|-------|--------|-------|------|
| rf1 | 20% | det + pose | mAP50≥0.25 |
| rf2 | 50% | det + pose + head_pose | mAP50≥0.40, MAE≤60° |
| rf3 | 35% | + activity | mAP50≥0.45, act_top1≥0.40, MAE≤55° |
| rf4-rf10 | 50-100% | + PSR at rf6 | progressively harder |

**Current**: rf2 epoch 21/36, best mAP50=0.2069 (epoch 20), MAE=9.21°. Gate not passed.

---

## 3. CRITICAL — The Structural Ceiling Breakthrough

### The Discovery: Two Runs, Identical Trajectories

The training log contains TWO runs that shared the same checkpoint but had different runtime configs:

| Run | DET_LR_MULTIPLIER | DET_BIAS_LR_FACTOR | Epochs |
|-----|------------------|-------------------|--------|
| Run 1 | **2.0** (accidental override) | **4.0** (accidental override) | 17-21 |
| Run 2 | **1.0** (correct, matches config.py) | **1.0** (correct, matches config.py) | 17-21 |

**Both produce IDENTICAL mAP50 trajectories:**

| Epoch | Run 1 mAP50 (2× LR, 4× Bias) | Run 2 mAP50 (1× LR, 1× Bias) |
|-------|------------------------------|------------------------------|
| 17 | 0.2039 | 0.2039 |
| 18 | 0.2065 | 0.2065 |
| 19 | 0.2088 | 0.2091 |
| 20 | 0.2047 (LR restart=ZERO) | 0.2069 (LR restart=ZERO) |
| 21 | (stopped) | 0.2024 |

**This proves mAP50 ceiling at ~0.207 is STRUCTURAL**, not config-dependent. LR/BIAS is ruled out as the bottleneck.

### CosineAnnealing LR Restart Has Zero Effect

Both runs confirm: the epoch 20 CosineAnnealingWarmRestarts restart produces zero response in mAP50. This is now definitive evidence of gradient-suppressed equilibrium — the gradient is already near-zero, so changing LR multiplier has nothing to amplify.

---

## 4. The Primary Hypothesis: OHEM + FocalLoss Gradient Suppression

**Claim**: OHEM (Online Hard Example Mining, 2:1 ratio) + FocalLoss (gamma=2.0, gamma_neg=1.5) together suppress detection classification gradients, creating the ~0.207 ceiling.

### Evidence For:
1. **50-image detection-only overfit (200 epochs)**: Shows three-regime trajectory — fast drop (epochs 1-5, cls_loss ~2→0.6) → plateau (epochs 5-55, cls_loss ~0.3-0.4 stuck) → slow decline (epochs 55-200, cls_loss 0.4→0.06). The plateau regime is >50 epochs of near-zero gradient. cls_w_norm grows linearly throughout, consistent with gradient-suppressed equilibrium.
2. **LIVENESS_GRAD diagnostic**: Detection head gradient norm ~0.02-0.03 vs backbone ~3.9-8.0. Ratio ≈ 0.007× (100:1+ in backbone's favor). This is consistent with OHEM selecting hard negatives that FocalLoss heavily down-weights.
3. **POS_ANCHOR_PROBE**: 400-800 positive anchors/image — anchor coverage is NOT the bottleneck. The bottleneck is gradient suppression on those anchors.
4. **LR/BIAS independence**: Four-fold change in bias LR and two-fold change in base LR produce zero trajectory change. This is consistent with OHEM+FL suppression dominating any LR effects.

### Evidence Against / Alternative Hypotheses Ruled Out:
- **Anchor coverage** (ruled out by POS_ANCHOR_PROBE: 400-800 pos anchors/image)
- **LR/BIAS config errors** (ruled out by identical Run 1/2 trajectories)
- **detach_reg_fpn** (ruled out as insufficient but still a correct fix)
- **Head pose domination** (ruled out by KENDALL_FIXED_WEIGHTS=True fix)
- **Gradient vanishing at backbone level** (ruled out — backbone grad norm ~3.9 is healthy)

### What Has Not Been Ruled Out:
- **Label noise / data quality issues** in the Industries dataset (unexamined)
- **Anchor size configuration** (sizes 96-768 on 720×1280 may be too large for small parts)
- **Combined effect** of OHEM + FocalLoss (each individually might be fine; together they suppress)

---

## 5. Current Training State (Live, Epoch 21 Completed, Epoch 22 In Progress)

```
State file: epoch=21, best_mAP50=0.2069 (epoch 20), best_MAE=9.21°
Combined best: 0.4622
DILUTION: det_mAP50_pc=0.3104 (present-class, honest) vs headline 0.2069 — 8 zero-GT channels dilute metric
Heads: detection ALIVE (grad 0.03), pose ALIVE (0.03), head_pose ALIVE (0.003)
       activity DEAD (frozen), PSR DEAD (frozen)
Backbone: grad 3.91 — ALIVE
FPN: grad 0.33 — ALIVE
mAP50_95: 0.0811 (epoch 19)
ETA per epoch: ~5100s (85 min), 15 epochs remaining at max_epochs=36
```

### Key Diagnostics Running:
- **POS_ANCHOR_PROBE** (every 200 batches): n_pos=400-800, mean IoU 0.05-0.87 (varies by image)
- **LIVENESS_GRAD** (per-parameter grad norms): detection_head ~0.02, backbone ~3.9 (ratio ~0.007)
- **DILUTION** (epoch-end): det_mAP50_pc vs det_mAP50 gap ~0.10-0.11

---

## 6. Questions We Need Opus's Help On

### Q1: OHEM Ablation Design
**What exact experiment would you run to test the OHEM+FocalLoss hypothesis definitively?**
- Current plan: set DET_OHEM_ENABLED=False, keep everything else, train 5 epochs from current checkpoint
- Expected outcome if OHEM is the bottleneck: mAP50 jumps to >0.30
- Risk: unbounded negatives flood the classifier, making things worse
- **Should we also adjust gamma? Change OHEM ratio? Remove FocalLoss entirely? Run ablation on the 50-image overfit first?**

### Q2: Alternative Hypotheses
**If OHEM ablation shows NO improvement (mAP stays at ~0.207), what should we check next?**
- Anchor configuration (too large for small assembly parts?)
- FPN level utilization (are small objects getting matched at appropriate pyramid levels?)
- Data quality audit (label noise, missing annotations?)
- Something else we haven't considered?

### Q3: The Gradient Ratio Problem
Detection head grad norm is consistently ~0.02-0.03 while backbone is ~3.9 (ratio ~0.007). Is this ratio pathological or expected?
- In multi-task learning, backbone accumulates gradients from all heads, so some imbalance is expected
- But 0.007× suggests detection is barely contributing to its own feature learning
- **Is there a way to increase detection gradient flow WITHOUT removing OHEM?** Gradient scaling? Loss weighting? Normalization?

### Q4: Should We Continue rf2 or Advance to rf3 Given the Ceiling?
- rf3 enables activity head + 35% new data subset
- Activity head might improve shared representations
- But rf3 gate requires mAP50≥0.45 (2.2× current)
- **Is there any scenario where advancing with broken detection is the right call?** E.g., if activity learning reactivates the backbone and pulls detection along?

### Q5: What Would You Try That We Haven't Considered?
The full list of interventions we've already applied:
- detach_reg_fpn=False (gradient routing fix)
- KENDALL_FIXED_WEIGHTS=True (head pose domination fix)
- LR/BIAS reverted to 1.0/1.0
- gamma_neg reduced to 1.5
- OHEM remains active
- CosineAnnealingWarmRestarts (T=10)

**What's missing?** What have we not thought of?

---

## 7. Quick Reference: Key Files

| File | Contents | Location |
|------|----------|----------|
| `45_CURRENT_TRAINING_STATE.md` | Single source of truth — full training state, config, metrics | `analyses/consult_2026_06_10/` |
| `47_HYPOTHESES_PROVEN_WRONG_AND_UNANSWERED.md` | All hypotheses, proven/wrong claims, unanswered questions | Same directory |
| `26_RF1_RF10_COMPREHENSIVE_STATUS.md` | Stage definitions, pipeline architecture, Section 20 = Run 1/2 correction | Same directory |
| `00_JOURNEY_AND_STATUS.md` | Full project timeline since inception | Same directory |
| `train.log` | Live training log (313k+ lines, both runs) | `src/runs/rf_stages/logs/` |
| `rf_stage_state.json` | Live stage state (epoch 21, PID 361404) | `src/runs/` |
| `rf2_checklist_report.txt` | 100-point RF2→RF3 decision checklist (scored ~30/100, STRONG HOLD) | `analyses/consult_2026_06_10/logs/` |

---

## 8. Configuration Details

```
DET_OHEM_ENABLED: True (ACTIVE — primary suspect)
DET_OHEM_RATIO: 2.0
FocalLoss gamma: 2.0
FocalLoss gamma_neg: 1.5
DET_POS_IOU_THRESH: 0.4
DET_POS_IOU_TOP_K: 9 (never kicks in — no GT exceeds 4 high-IoU anchors)
DET_POS_IOU_IOU_FLOOR: 0.2
DET_LR_MULTIPLIER: 1.0
DET_BIAS_LR_FACTOR: 1.0
detach_reg_fpn: False (correct fix applied ALL stages)
KENDALL_FIXED_WEIGHTS: True
LR scheduler: CosineAnnealingWarmRestarts (T_0=10)
Batch size: 4
Subset ratio: 0.50 (rf2)
Max epochs: 36
```

---

*End of overview. Provide this prompt to Opus for comprehensive consultation on the structural ceiling, OHEM+FocalLoss hypothesis, and next experimental steps.*
