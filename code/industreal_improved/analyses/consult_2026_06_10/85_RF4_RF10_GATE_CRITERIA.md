# 85 — RF4 Through RF10 Gate Criteria: Stage-by-Stage Go/No-Go [2026-07-01]

**Goal:** Define clear pass/fail criteria for each RF stage so training can be stopped early if something goes wrong, saving GPU time. Every criterion below is **logged in standard training output** — no offline analysis needed.

**Source files:**
- `src/training/train.py` — DET-HEALTH logging (line 1455), GRAD-NORM logging (line 1917), LR scheduler (line 3644)
- `src/evaluation/evaluate.py` — Activity diversity (line 3491), PSR comp acc (line 3764), detection mAP
- `src/config.py` — apply_preset (line ~1827), DET_GT_FRAME_FRACTION (line 828)

---

## Stage Overview

| Stage | Epochs | What Happens | Config |
|-------|--------|-------------|--------|
| **RF1** | 0-5 | Detection bootstrap (reinit heads) | det-only, WD=1e-3, clip=5.0 |
| **RF2** | 5-15 | Detection + head pose | add head pose |
| **RF3** | 15-20 | Add activity (ramp epochs 0-4) | activity ramp, CB weights |
| **RF4** | 20-40 | **First full 4-task** | ALL heads, DET_GT_FRAC=0.40 |
| **RF5** | 40-55 | Continued training | Same as RF4 |
| **RF6** | 55-70 | Continued training | Same as RF4 |
| **RF7** | 70-80 | Continued training | Same as RF4 |
| **RF8** | 80-90 | Continued training | Same as RF4 |
| **RF9** | 90-95 | Continued training | Same as RF4 |
| **RF10** | 95-100 | **Final convergence** | Same as RF4 |

---

## Critical Runtime Diagnostics (Every 500 Steps)

These are logged automatically during training and should be monitored in real-time:

| Log Line | What to Look For | Failure Signal |
|----------|-----------------|----------------|
| `[DET-HEALTH step=N] cls_preds: mean=... std=... near_zero=...` | mean between -3 and -1, std > 0.5 | mean < -5 (collapse toward -16), near_zero > 0.8 |
| `[DET-HEALTH step=N] det_gt_fraction: X/Y=0.XX` | 0.30-0.50 (target DET_GT_FRAME_FRACTION=0.40) | < 0.15 (sampler not distributing correctly) |
| `[GRAD-NORM step=N] backbone=... det=... hp=... act=... psr=...` | ALL 4 heads non-zero (> 1e-8) | Any head at 0.00e+00 (dead gradient) |
| `[DIVERSITY epoch=N] pred_distinct=X/Y entropy=Z.ZZZ` | pred_distinct ≥ 10, entropy ≥ 1.5 | pred_distinct < 5 (activity collapse) |

---

## RF4 Gate Criteria (Epoch 2 — First Full 4-Task Check)

**Current config at RF4 start:** `DET_GT_FRAME_FRACTION=0.40`, `ACT_SAMPLER_MODE='balanced'`, `ACTIVITY_HEAD_SIMPLE=True`

| Signal | Where to Find It | Pass | Borderline | Fail |
|--------|-----------------|------|-----------|------|
| `det_cls_mean` | `[DET-HEALTH step=N]` | -3.0 to -1.0 | -5.0 to -3.0 | < -5.0 or > 0.0 |
| `det_gt_fraction` | `[DET-HEALTH step=N]` | 0.30-0.50 | 0.20-0.30 | < 0.20 |
| `act_pred_distinct` | `[DIVERSITY epoch=N]` | ≥ 10 | 6-9 | < 6 |
| `act_macro_f1` | Validation `act_macro_f1=...` | ≥ 0.10 | 0.05-0.10 | < 0.05 |
| `act_clip_accuracy` | Validation `act_clip=...` | ≥ 0.20 | 0.10-0.20 | < 0.10 |
| `act_entropy` | `[DIVERSITY] entropy=...` | ≥ 1.5 nats | 1.0-1.5 | < 1.0 |
| `psr_loss` (Kendall-weighted) | `psr=...` in training log | 0.2-1.5 | 0.1-2.0 | > 5.0 or NaN |
| `psr_comp_acc` | Validation `PSR — Component Binary Accuracy: ...` | ≥ 0.55 | 0.45-0.55 | < 0.45 |
| `forward_angular_MAE_deg` | Validation `forward_angular_MAE_deg=...` | < 40° | 40-60° | > 60° |
| `log_var_det` | `[Kendall log_sigma] lv_det=...` | -1.0 to +1.0 | -2.0 to +2.0 | Outside [-4, 2] |
| `[GRAD-NORM]` all 4 > 0 | `[GRAD-NORM step=N]` | ALL > 1e-8 | — | Any head = 0.0 |
| Per-class sampling mass ratio | `[get_sampler]` at epoch 0 | < 10x | 10-20x | > 20x |

**RF4 Fail Action:** If 3+ signals are in Fail at epoch 2 → STOP, diagnose. If only 1-2 borderline → continue to epoch 5 and re-check.

**RF4 Expected Final (epoch 40):**
- Detection mAP50: 0.20-0.40
- Activity clip-acc: 0.25-0.45 (grouped)
- PSR comp-acc: 0.65-0.75
- Head pose forward-gaze: 15-30°

---

## RF5 Gate Criteria (Epoch 45)

| Signal | Pass | Fail |
|--------|------|------|
| Detection mAP50 | ≥ 0.25 | < 0.10 |
| Activity macro-F1 | ≥ 0.15 | < 0.05 |
| PSR comp-acc | ≥ 0.70 | < 0.55 |
| Head pose MAE | < 25° | > 40° |
| Activity pred_distinct | ≥ 15 | < 8 |

**RF5 Fail Action:** If detection mAP50 < 0.10 at epoch 45 → detection is stuck. Verify `DET_GT_FRAME_FRACTION=0.40` is being applied (check `[DET-HEALTH] det_gt_fraction`).

---

## RF6-RF9 Continuous Monitoring

| Signal | Healthy Range | Watch | Critical |
|--------|-------------|-------|----------|
| Detection mAP50 | Increasing trend | Plateaued 3 epochs | Decreasing 3 epochs |
| Activity macro-F1 | Increasing trend | Plateaued 3 epochs | Decreasing 3 epochs |
| PSR comp-acc | Increasing trend | Plateaued 3 epochs | Decreasing 3 epochs |
| Head pose MAE | Decreasing trend | Plateaued 3 epochs | Increasing 3 epochs |
| `[DIVERSITY] pred_distinct` | ≥ 10 | 6-9 | < 6 |
| `[GRAD-NORM]` all heads | ALL > 1e-8 | — | Any = 0 |

**Any signal in Critical for 3+ consecutive epochs** → STOP and diagnose.

---

## RF10 Final Gate Criteria (Epoch 100 — Paper Metrics)

| Metric | Target Range | Comparison | Notes |
|--------|-------------|-----------|-------|
| **Detection mAP50** | 0.50-0.65 | vs YOLOv8m 0.838 | Same ASD protocol/classes expected |
| **Activity clip-acc** (grouped) | 0.40-0.60 | vs MViTv2 0.653 (74-class) | **Different task** — must state "action-group recognition" |
| **Activity macro-F1** (grouped) | 0.30-0.50 | Own baseline | present_labels filter, ~47 groups |
| **PSR comp-acc** | 0.75-0.85 | Own baseline | Per-frame component recognition, NOT transition |
| **PSR overall F1** | 0.60-0.80 | Own baseline | Macro-F1 across 11 components |
| **Head pose forward-gaze MAE** | < 15° (target 8.71°) | **First reported** — no IndustReal baseline | Report forward only (up ~95° unlearned) |
| **Head pose position MAE** | **DO NOT REPORT** | — | Unit unverified (evaluate.py:1861 warns) |
| **Activity pred_distinct** | ≥ 20 groups | — | Diversity indicator |
| **All 4 gradient norms** | ALL > 1e-8 | — | Gradient health |

### Paper Claim Honesty Check
| Claim | Honest? | Required Phrasing |
|-------|---------|-------------------|
| "Action recognition: X%" | ❌ | Must say "action-group recognition" (74→47 classes) |
| "Benchmarked against MViTv2" | ❌ | Different task (grouped vs 74-class). Re-evaluate MViTv2 under same grouping or establish own baseline |
| "Detection benchmarked against YOLOv8m" | ✅ | Same 24-class ASD protocol, COCO mAP |
| "PSR results" | ⚠️ | Must say "per-frame component recognition," not "transition detection" |
| "Head pose forward-gaze MAE" | ✅ | First reported result on IndustReal — say so explicitly |
| "Body pose/keypoints" | ❌ | Pseudo-generated from detection boxes. No real GT. Do not report. |

---

## Crash Protection: What Happens on Failure

| Failure | Recovery | Impact |
|---------|----------|--------|
| **CUDA OOM** | Halve batch_size, rebuild dataloader, retry epoch (train.py:4252-4279) | 500-1000 steps lost per OOM (crash checkpoint every 1000 steps) |
| **NaN loss** | Replaced with 1e-4 fallback, gradient continues (losses.py:1272-1296) | One dead batch, no crash |
| **Zero positives** | num_pos clamped to 1, OHEM keeps MIN_NEG=32 negatives (losses.py:270, 321) | No crash, slower convergence |
| **All labels = -1** | loss_act = 0 (losses.py:1363) | One dead batch, no crash |
| **CUDA kernel hang** | ThreadPoolExecutor timeout, retry (train.py:4610-4622) | 1-2 minutes lost, retry |
| **Scheduler restart on crash** | crash_recovery.pth doesn't save scheduler state → LR restarts from warmup | Visible LR spike, ~2 epochs recovery |
| **Process death** | `--resume` from latest checkpoint | Model/optimizer/scheduler state fully restored from latest/epoch_N.pth |

---

## Pre-Flight Checklist (Run Before Launching RF4)

1. **Verify scheduler:** Run `python -c "from src.training.train import main; main(['--preset', 'stage_rf4', '--max-epochs', '2'])"` and check `lr=...` in training log rises from ~1e-5 to ~5e-4 over epochs 0-9
2. **Verify DET_GT_FRAME_FRACTION:** Check apply_preset() log at startup: `[config] DET_GT_FRAME_FRACTION = 0.40 (train_det=True, train_act=True, train_psr=True)`
3. **Verify no DETACH_REG_FPN:** Check model log — should say `detach_reg_fpn: False`
4. **Verify activity grouping:** Run `python scripts/verify_act_grouping.py` and confirm NUM_ACT_OUTPUTS matches ~47 and no group has <15 frames
5. **Run 50-step probe:** Launch with `SIMPLIFY_LOSS=True` and check: `det_cls_mean` (-3 to -1), `psr_loss` (0.2-1.5), `log_var_det.device` (cuda:0), per-class sampling mass ratio (<10x)
