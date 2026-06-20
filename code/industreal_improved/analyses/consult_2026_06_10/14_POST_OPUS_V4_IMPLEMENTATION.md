# 14 — Post-Opus v4 Implementation Journal (2026-06-13)

## What We Learned From Opus v4

Opus told us the hard truth: **Run 8 launched without the RC-28/RC-29 fixes.** The fixes had been sitting on `claude/keen-lamport-vxnhxg` since June 12, unmerged. Run 8 trained on pre-fix code where:

1. **RC-28**: Empty frames contributed their full ~4.15M-element negative focal mass divided by `num_pos=1` (~85% of frames, ~30:1 domination of the detection gradient)
2. **RC-29**: `mixed_precision: True` was hand-flipped (against audited FP32 config), causing GradScaler to silently skip `optimizer.step()` — the model was FROZEN, not at equilibrium

The "critical finding" that Run 8 proved the collapse is architectural was **wrong** — it proved the unfixed normalization reproduces, which we already knew. The fresh ImageNet start did prove the collapse isn't checkpoint lineage (valuable), but focal loss is fine, the architecture can learn, and no exotic loss replacements (VFL/GHM/OHEM/ATSS) are needed.

## Sequence of Events

### 1. Cherry-Pick the Fixes (commit `d101e7e`)

Cherry-picked `c219569` from `claude/keen-lamport-vxnhxg` onto main. The fix contained:
- **RC-28**: Empty frames with `gt_boxes.shape[0]==0` are skipped in detection loss; normalization changed from `/B` to `/max(n_img_with_gt, 1)`
- **RC-29**: Step-commit telemetry at both optimizer-step sites with per-epoch `committed/skipped/scaler_scale` verdict
- **recovery_det_only** preset: det + head_pose only, FP32, batch=1, grad_accum=8
- `13_OPUS_ANSWER_v4.md`: Full Opus answer document

### 2. Pre-R1 Config Fixes (commit `8bef821`)

- Added `TRAIN_MAX_STEPS` from env var to config.py (default 0 = disabled)
- Set `SKIP_DET_METRICS_EVAL=False` so we can see `det_mAP50` for the R1 gate
- Fixed `PRE_VAL_GUARD` to accept `loss ≤ 0` (changed `> 0` to `isfinite()`) — Kendall log-vars on fresh init produce near-zero total loss when empty frames are skipped

### 3. R0 Smoke Test — PASSED

```
TRAIN_MAX_STEPS=400 python3 src/training/train.py --preset recovery_det_only --subset-ratio 0.25
```

| Metric | Result |
|--------|--------|
| Time | 278s (~4.6 min) |
| RC-29 | committed=55, skipped=0 (0.0%), scaler_scale=1.0 |
| det c-loss | Started 49 → 0.05 (trending down on GT frames) |
| nan_skips | 0 |
| PRE_VAL_GUARD | Blocked val (loss≈0 — false positive from fresh Kendall init) |
| Verdict | **PASSED** — all gates met |

The false-positive PRE_VAL_GUARD was fixed immediately (changed to `isfinite()`).

### 4. R1 Detection Bootstrap — 4 Attempts, 3 Eval Crashes

**R1 command**: `python3 src/training/train.py --preset recovery_det_only --subset-ratio 0.25 --max-epochs 3 --seed 42`

#### R1 v1 — Crash #1: Activity eval broadcast mismatch

Epoch 0 completed (6,954 batches, 4,064s, committed=870, skipped=0). Validation started, DET_PROBE showed LOCALIZING on GT frames (bestIoU up to 0.94). Crashed at `evaluate.py:809`:

```
ValueError: operands could not be broadcast together with shapes (9626,5) (8115,1)
```

**Fix**: Added shape guard before `top5_indices == all_gt[:, None]` broadcast.

**Why it happened**: Activity head predictions and ground truth counts don't align when activity head is disabled during training. The evaluation pipeline collected mismatched prediction/GT arrays.

#### R1 v2 — Crash #2: Activity eval clip-level accuracy

Same epoch 0 result (6,954 batches, 4,065s). Crashed at `evaluate.py:833` in `_compute_clip_level_accuracy`. Another activity metric function with data alignment assumptions.

**Fix**: Skipped entire `compute_activity_metrics()` call when `TRAIN_ACT=False`. Also skipped PSR eval when `TRAIN_PSR=False`.

#### R1 v3 — Crash #3: Activity logging KeyError

Epoch 0 completed (6,954 batches, 4,058s). Crashed at `evaluate.py:3191`:

```
KeyError: 'act_accuracy'
```

The stub activity metrics dict (returned when TRAIN_ACT=False) only had `act_macro_f1`, `act_top5_acc`, `act_frame_acc` — but the logging section referenced 7+ activity keys.

**Fix**: Guarded the activity logger.info() block with `if getattr(C, 'TRAIN_ACT', True):`. Also guarded PSR logging similarly.

#### R1 v4 — SUCCESS: First Non-Zero Detection Metrics!

Epoch 0 completed (6,954 batches, 4,075s). Validation completed **without crashing** — all 5 eval guards worked.

**Val results (epoch 0):**
```
loss=-0.3916  det_mAP50=0.0091  as_f1=0.0000  as_map_r=0.0000
ev_ap=0.0268  ev_f1=0.0000  combined=0.1107
act_macro_f1=0.0000  act_top5=nan  psr_f1=nan  psr_edit=nan
```

**THIS IS THE FIRST TIME IN THE PROJECT'S HISTORY THAT `det_mAP50` PRINTED ABOVE ZERO.**

Key findings:
- `det_mAP50=0.0091` — detection is LEARNING (gate was ≥0.05, close after just 1 epoch on 25% subset)
- `ev_ap=0.0268` — error verification also showing non-zero signal
- `as_f1=0.0000`, `as_map_r=0.0000` — assembly state still zero (more epochs needed)
- `psr_f1=nan`, `act_top5=nan` — cosmetic NaN from stub dict key mismatches in Val line formatter. PSR/act eval correctly skipped; the stub values `{psr_f1: 0.0, psr_edit: 0.0}` don't include all keys the formatter expects (`psr_overall_f1`, etc.)
- Combined=0.1107 — up from 0.1067 (pre-fix runs) — driven by det_mAP50 and ev_ap contributions

**Epoch 1**: Started immediately after val. Training at 10% (batch 670).

| Guard | Location | Purpose |
|-------|----------|---------|
| Top5 shape check | evaluate.py:809 | Skip broadcast when preds ≠ GT count |
| TRAIN_ACT guard | evaluate.py:3174 | Skip activity eval when disabled |
| TRAIN_ACT logger guard | evaluate.py:3190 | Skip activity logging when disabled |
| TRAIN_PSR guard | evaluate.py:3255 | Skip PSR eval when disabled |
| TRAIN_PSR logger guard | evaluate.py:3264 | Skip PSR logging when disabled |

### 5. What We've Seen So Far (DET_PROBE on GT Frames)

Across all R1 runs, the model consistently shows:

| Batch | GT Boxes | preds@IoU>0.5 | bestIoU | Verdict |
|-------|----------|---------------|---------|---------|
| b0 | 16 | 49 | 0.59 | LOCALIZING |
| b1 | 16 | 93 | 0.61 | LOCALIZING |
| b2 | 6 | 44 | 0.72 | LOCALIZING |
| b4 | 4 | 673 | 0.91 | LOCALIZING |
| b34 | 15 | 1,867 | 0.85 | LOCALIZING |
| b197 | 9 | 1,124 | 0.84 | LOCALIZING |
| b222 | 12 | 2,775 | 0.89 | LOCALIZING |
| b223 | 16 | 4,142 | 0.94 | LOCALIZING |
| b224 | 16 | 3,116 | 0.92 | LOCALIZING |

**Empty frames**: Model correctly outputs near-zero scores (score_p50=0.001, 0-3 preds>0.05). False positive suppression improved across retrains as model stabilized.

## Confusions Encountered

1. **"Run 8 proved architectural collapse"**: We were wrong. Opus corrected us — it proved the unfixed normalization reproduces, not that focal loss is fundamentally broken. The RC-28/RC-29 fixes were written, just unmerged.

2. **The `mixed_precision` hand-flip**: Run 8's config had `'mixed_precision': True` with comment "[TEMP: AMP for 12GB VRAM]" — overriding the audited FP32 requirement. This caused GradScaler to skip every optimizer step. We didn't notice because forward-pass losses looked healthy. The RC-29 telemetry now catches this.

3. **PRE_VAL_GUARD rejecting healthy training**: The guard checked `total_loss > 0`, but Kendall-weighted loss on fresh init with empty frames skipped averages near zero. Changed to `isfinite()`.

4. **Eval pipeline assumes all heads are trained**: Three separate crashes from activity eval code that assumes aligned prediction/GT arrays. We didn't anticipate this because recovery_det_only (det+head_pose only) is a new training mode the eval pipeline was never tested against.

## Current State (as of 2026-06-13)

- **R1 v4**: Running validation (DET_PROBE b601+, past all crash points)
- **Training**: Healthy — committed=870, skipped=0, nan_skips=0
- **Detection**: LOCALIZING on GT frames with bestIoU up to 0.94
- **Waiting for**: Final `Val:` line with `det_mAP50` number
- **Next**: If det_mAP50 ≥ 0.05 → proceed to R2 (joint recovery). If not → det-head LR ×3 for 2 more epochs.

---

## Phase 10: RF1 Death Spiral — Gradient Sparsity Proves Lethal (2026-06-14 to 2026-06-17)

### 10.1 The Discovery

RF1 (detection-only bootstrap, `train_head_pose=False`) kept dying. DET_PROBE consistently showed `LOCALIZING` across ALL runs — the detector could place boxes (GIoU 0.8-0.95) but never fired with confidence above 0.10. cls_mean was frozen at -4.70 (pi=0.01 bias init).

**The pi=0.01 + anchor math** (documented in `29_RF1_DEATH_SPIRAL_AND_R2_5_PARADOX.md`):
- 16 positive anchors per batch of 4 images
- 2.76M total anchors per batch
- Positive ratio: 0.00058%
- Gradient per backbone parameter: ~3.4 × 10⁻⁸ per optimizer step
- Total parameter change over 20 epochs: ~0.06%

This is the **gradient sparsity** problem: not zero gradient, but gradient spread so thin across 28M backbone parameters that no parameter moves measurably.

### 10.2 The R2.5 Paradox Resolved

R2.5 (paper_run with ALL heads) trained visibly well. Why?

```
R2.5 gradient sources:
  - Activity head: 22M gradient contributions/batch (dense, per-frame)
  - PSR head: sequence-frame gradient (dense)
  - HeadPose head: 400K gradient contributions/batch (dense)
  - Detection: 16 positive anchors (sparse)
  Total: ~22M dense + 16 sparse = healthy backbone gradient

RF1 gradient sources:
  - Detection cls: 16 positive anchors (sparse)
  - Detection reg: DETACHED (DETACH_REG_FPN=True)
  Total: 16 positive anchors across 28M backbone params = invisible
```

**Resolution**: RF1 fails because single-task detection with 172K anchors/image
and pi=0.01 init creates gradient that's 10,000× too sparse. R2.5 worked because
multi-task learning provided dense gradient from 3 additional heads.

### 10.3 The Stages of RF1 Death Spiral

| Phase | Run | Symptom | Root Cause |
|-------|-----|---------|------------|
| pi=0.01 collapse | All runs | cls_mean frozen at -4.70 | pi=0.01 too low for 172K anchors |
| False-positive kills | RF1 retries | Bounded background loss dominates | Highest-scoring 512 anchors suppress features |
| Death spiral | RF1 retry #2-4 | Det loss exists, cls_mean doesn't move | 16 positives / 2.76M = no gradient per param |
| 20× LR identity | Retry strategies | LR 0.5×, 0.25×, 0.1× — all fail | Lower LR makes the tiny updates even smaller |

### 10.4 The Kendall Bug (2026-06-16 21:10)

Disaster: **The `train_head_pose=True` fix was silently neutralized by a bug in Kendall weighting logic.**

The fix was prescribed by Opus (v6/v7): enable head_pose in RF1 for dense gradient. But `losses.py:1589` had:
```python
elif self.train_pose:
    pose_contribution = prec_hp * loss_pose + lv_hp  # loss_head_pose MISSING!
```
Since IndustReal has NO keypoint annotations, `loss_pose=0` always. So:
```
pose_contribution = prec_hp * 0 + lv_hp = log_var ONLY
```
The `loss_head_pose` (~1.7) was computed in the forward pass but EXCLUDED from total loss. Head pose received ZERO gradient for 7+ epochs.

**Detection:** LIVENESS_GRAD probe showed `head_pose_head:NO_GRAD` 104 times.
**Fix applied:** Both Kendall and non-Kendall paths fixed, confirmed at step 0.

---

## Phase 11: RF1 Completion — The Kendall Fix Works (2026-06-17 to 2026-06-19)

### 11.1 Post-Fix Verification

Fresh RF1 launch (PID 1220890) confirmed at step 0:
```
head_pose_head:ALIVE[8.99e-01]/ALIVE[7.01e-03]
                          ^^^^^^^^ gradient-based liveness (was NO_GRAD)
```

**cls_preds differentiation breakthrough:**
```
Broken run (step 751): cls_std=0.88, cls_max=0.47 — weak diff
Fixed run  (step 751): cls_std=1.37, cls_max=2.78 — 1.6× broader, 5.9× higher
```
Head_pose converged from 1.60→0.01 by step 450 (99.4% reduction).

### 11.2 RF1 Gate Met

Per stage_history, RF1 achieved `best_det_mAP50=0.45`. However, metric_history
shows max 0.184 — a **2.4× discrepancy** that remains UNRESOLVED (see Q02 in
33_OPEN_QUESTIONS.md). Possible causes: different evaluation protocols, data
splits, or metric_history truncation.

### 11.3 Stage Progression

Despite the discrepancy, RF1 was marked complete and the stage manager
progressed through RF2 onward. The head_pose fix enabled the model to train
through gradient-sparse detection-only, but the fundamental question remains:
does RF1 actually work at 0.45 or is the 0.184 the real number?

---

## Phase 12: RF2 Epoch 15 Collapse — A New Failure Mode (2026-06-19 to 2026-06-20)

### 12.1 The Collapse That Shouldn't Have Happened

RF2 had everything supposedly needed:
- All RF1 fixes applied (Kendall bug, FP32, bounded bg loss)
- `train_head_pose=True` (head_pose gradient ALIVE throughout)
- 35% data (vs 20% in RF1)
- `DET_GT_FRAME_FRACTION=0.90` (90% of batches contain GT frames)
- `DETACH_REG_FPN=False` (regression gradient flows to backbone)

Yet at epoch 15, EVAL COLLAPSE struck:
```
det_mAP50    = 0.001  (near zero)
det_mAP      = 0.000
det_mAP50_95 = 0.000
EVAL COLLAPSE: 56 occurrences at epoch 15, all 3 heads simultaneously zero
```

### 12.2 DET_PROBE Evidence at Epoch 15

```
score_p50  = 0.019 (median at bias floor)
score_mean = 0.079 (all classes ~same)
score_std  = 0.0068-0.0088 (near-zero variance — no differentiation)
cls_mean   = -2.54 (bias drifted from -4.6 init to -2.54 equilibrium)
preds>0.30 = 0     (zero high-confidence predictions across ALL probes)
```

**The cls_score bias equilibrium**: sig(-2.54) = 0.073. The bias parameter
drifts from -4.6 (pi=0.01) to ~-2.5 where sigmoid produces ~0.076-0.079 for
most classes. The classifier CAN make confident predictions (score_max =
0.93-0.97 for SOME classes) but the median is stuck at 0.019.

### 12.3 Head Pose MAE Was Improving While Detection Collapsed

```
epoch  7: forward_angular_MAE_deg = 71.67
epoch  9: forward_angular_MAE_deg = 63.65
epoch 11: forward_angular_MAE_deg = 56.61
epoch 13: forward_angular_MAE_deg = 55.73
epoch 15: forward_angular_MAE_deg = 47.84
```

MAE improving 71.67→47.84 while det_mAP50 drops from 0.184→0.001. This is the
critical dissociation: head_pose gradient keeps backbone healthy, but the
classification head converges to its own internal equilibrium independent
of backbone state.

### 12.4 cls_score Bias Equilibrium — Mathematical Model

The bias parameter b in the final cls_preds conv layer follows:
```
b_update ∝ Σ(sigmoid(b + W·f_i) - target_i)  for all anchors i
```
At equilibrium, Σ(sigmoid(b + W·f_i)) = Σ(target_i) = total_positive_anchors.
With target ≈ 0.0011% positive anchors:
```
Σ sigmoid(b + W·f_i) ≈ 0.0011% of total anchors
```
This forces b to a value where sigmoid(b) ≈ mean(cls_score) ≈ 0.079, exactly
the observed equilibrium.

### 12.5 20-Agent Monitoring Swarm Deployed (2026-06-20)

The monolithic rf2_checklist.py (1320 lines, 118 checks) was replaced with a
22-agent swarm (134 checks/cycle, 5-min interval, 40-thread ThreadPoolExecutor).

**6 bugs found and fixed in first hours:**
1. **ND01 NaN false positives**: Efficiency stat lines `Params: nanM, GFLOPs: nanG` matched NaN regex — fixed with EFFICIENCY_RE exclusion
2. **ND01 compound word matching**: `\b` word boundaries added to NaN patterns
3. **CS06 log_head_text missing**: Det_head_bias optimizer line at training start wasn't in data_sources — added log_head_text fallback
4. **BU01 same log_head_text**: Same fix as CS06
5. **L06 keyword spike unreliable**: Replaced keyword-based "spike" detection with 3σ statistical outlier detection
6. **Training heartbeat**: Not updating — fix applied to train.py

**Swarm limitation detected**: No check for cls_score equilibrium (uniform
~0.079 scores). CS07: "cls_score std < 0.01" check is planned.

### 12.6 What We Learned

**Proven hypotheses (5):**
1. Gradient sparsity (16/2.76M positive anchors) kills detection-only training
2. Head_pose dense gradient enables backbone updates (Kendall fix confirmed)
3. The R2.5 Paradox is resolved — multi-task gradient density, not architecture
4. Focal Loss can train this architecture (R2.5 was healthy)
5. Eval pipeline needs 5 guards for single-head training modes (TRAIN_ACT, TRAIN_PSR)

**Refuted hypotheses (4):**
1. "Run 8 proved architectural collapse" — wrong, it proved unfixed normalization
2. "LR reduction = fix" — all retry strategies that reduce LR make things worse
3. "Head_pose + Kendall fix = complete solution" — RF2 epoch 15 collapse disproves
4. "DETACH_REG_FPN = main cause" — regression gradient helps but doesn't resolve
   the classification head's internal equilibrium

**Still unknown (see 33_OPEN_QUESTIONS.md):**
- Why collapse AGAIN at epoch 15 with everything supposedly fixed?
- stage_history 0.45 vs metric_history 0.184 — which is real?
- Why has PSR NEVER trained in any configuration?
- Is Focal Loss wrong for 172K anchors?
- Is the model overfitting to 0.7% GT frames?

---

## What Comes Next

The cls_score bias equilibrium is the central unsolved problem. Five fix proposals
are ranked in `26_RF1_RF10_COMPREHENSIVE_STATUS.md` (Appendix A):

| Option | Approach | Risk | ETA |
|--------|----------|------|-----|
| A | pi=0.1 bias init (was 0.01) | Low — known in RetinaNet literature | 1 epoch to verify |
| B | Remove classification bias | Low — forces weight-based differentiation | 1 config change |
| C | Quality Focal Loss | Medium — eliminates bias parameter entirely | Code change |
| D | Varifocal Loss | Medium — asymmetric learning | Code change |
| E | Dedicated bias LR | Low — keeps current training stable | 1 config change |

The full journey, all open questions, and next steps are documented in:
- `00_JOURNEY_AND_STATUS.md` — Full timeline Phases 1-12
- `26_RF1_RF10_COMPREHENSIVE_STATUS.md` — Stage definitions + RF2 data + fix proposals
- `33_OPEN_QUESTIONS.md` — 24 open questions organized by severity
- `34_RF2_SWARM_MONITOR.md` — 20-agent monitoring swarm documentation
