# 67: RF10 Training Optimization Roadmap — Final Push [2026-06-30]

## Current State

Training is running PID 3618126 (RF4, simple head, act=1x, LR=5e-4, 50% data).
Epoch 3 batch ~150/3469. First validation with simple head in ~40 min.

### Key Variables After All Fixes

| Variable | Value | Status |
|----------|-------|--------|
| ACTIVITY_HEAD_SIMPLE | True | DONE |
| ACTIVITY_LR_MULTIPLIER | 1.0 | DONE |
| ACTIVITY_GRAD_BLEND_RATIO | 1.0 | DONE |
| ACTIVITY_HEAD_GRAD_CLIP | 1.0 (no-op) | DONE |
| NUM_WORKERS | 0 | DONE |
| RAM_CACHE_MAX_IMAGES | 8000 | DONE |
| STAGED_TRAINING | False (all heads active) | DONE |
| FEATURE_BANK bypass | True (non-staged mode) | DONE |
| Pre-val checkpoint | Active | DONE |
| Watchdog | Active (PID-verified, 600s) | DONE |
| Gradient centralization | Active (AMP + FP32) | DONE |
| Subprocess eval | NOT YET | **TODO** |
| Head pose normalization | NOT YET | **TODO** |
| Auto-load crash_recovery | NOT YET | **TODO** |

## RF4-RF10 Schedule (Projected)

With NUM_WORKERS=0 and RAM_CACHE_MAX_IMAGES=8000:
- Speed: 1.2-1.3 batch/s, ~48 min/epoch at 50% data → ~24 min/epoch at 100% data
- Each epoch processes: 3,469 batches × 4 batch_size × 8 grad_accum = ~111K frames/epoch
- Each stage has 50% subset → 18 train recordings, ~1,834 frames

| Stage | Data | Epochs | Est. Time | Cumulative | Gate Target |
|-------|------|-------|-----------|------------|-------------|
| **RF4 (CURRENT)** | 50% | 23 max | ~18h | ~18h | det≥0.20, act≥0.06, psr≥0.05 |
| RF5 | 50% | 10 | ~8h | ~26h | det≥0.22, act≥0.08, psr≥0.06 |
| RF6 | 65% | 10 | ~8h | ~34h | det≥0.24, act≥0.10, psr≥0.08 |
| RF7 | 65% | 10 | ~8h | ~42h | det≥0.24, act≥0.12, psr≥0.10 |
| RF8 | 80% | 10 | ~8h | ~50h | det≥0.26, act≥0.14, psr≥0.12 |
| RF9 | 90% | 10 | ~8h | ~58h | det≥0.28, act≥0.16, psr≥0.14 |
| RF10 | 100% | 15 | ~12h | ~70h | det≥0.30, act≥0.18, psr≥0.16 |

**Total: ~70 hours of continuous training (3 days) assuming no crashes.**

With subprocess eval eliminating CUDA hangs AND RAM cache eliminating HDD bottleneck,
we can reasonably expect crash-free runs. Without subprocess eval, expect 1-2 crashes
per day → 2-4 epoch losses per crash → ~4-8 extra hours.

## Opus's Honest Feasibility Assessment (63 §4)

| Task | Current | Stage Gate (RF10) | Opus's Realistic Ceiling | Verdict |
|------|---------|:-:|:-:|---------|
| head_pose MAE | 8.71° | ≤35° | **Already met** | CERTAIN PASS |
| det_mAP50 | 0.053 | 0.30 | 0.20-0.30 | PLAUSIBLE |
| act_top1 | ~0 | 0.18 | 0.10-0.20 with simple head | **UNCERTAIN** |
| psr_f1_at_t | ~0 | 0.16 | needs transition obj + seq batches | UNLIKELY |

**Key judgment from Opus (63 §4):** "The stage gates (act_top1 ≥0.18, det ≥0.30)
are reachable; the paper numbers (act=0.375, det=0.838) are not."

This means our RF1-RF10 stage schedule is REASONABLE if the simple head works.
The critical variable is the simple head's first validation — if it produces
diverse predictions (>10 classes), we have a path. If it collapses to 1 class,
activity is dead regardless of architecture.

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|:-:|:-:|-----------|
| Simple head still collapses to 1 class | 30% | **CRITICAL** — activity dead | Add focal loss for activity (was CE+LS) |
| Subprocess eval not implemented in time | 60% | **HIGH** — 1-2 crashes/day | Temporary: VAL_EVERY=3, reduce exposure |
| Head pose normalization changes 8.71° | 40% | **MEDIUM** — best number degrades | Redo validation, adjust paper claim |
| Detection stalls at 0.15 | 30% | **MEDIUM** — misses RF10 0.30 | Need more data (100%), better aug |
| PSR never produces F1 > 0 | 50% | **MEDIUM** — must skip PSR claim | Reduce scope to 4 tasks |
| CUDA hang during full eval (not gate) | 40% | **LOW** — watchdog catches it | Every-N-epochs full eval |
| Disk space runs out (checkpoints) | 10% | **LOW** | Auto-delete old checkpoints |

## Detection Optimization Path

Detection currently at det_mAP50=0.053 at epoch 2. At 0.025/epoch improvement:
- Epoch 10: 0.25 (passes RF4 gate)
- Epoch 20: 0.50 (passes RF10 gate)
- Epoch 30: 0.75

This trajectory assumes the seq/det batch alternation doesn't disrupt detection
training. With simple head (no more TCN+ViT gradient), less gradient competition.

### Detection-Specific Tuning Options

1. **DET_POS_IOU_THRESH = 0.4** (current, was 0.5) — lowered for small assembly parts
2. **DET_EVAL_SCORE_THRESH = 0.001** (current) — captures low-confidence preds
3. **DET_LR_MULTIPLIER = 1.0** (current) — could increase to 3.0 if det stalls
4. **DET_GT_FRAME_FRACTION = 0.4** (current for RF4) — 40% of batch has GT frames
5. **GRAD_CLIP_NORM = 1.0** (current) — adequate for detection

## PSR Optimization Path

PSR has never produced F1 > 0 because:
1. Per-component focal loss with sparse targets (most frames have 0/11 components active)
2. Transition objective (detect STATE CHANGES, not frame labels) requires sequence mode
3. The oscillation is from seq/det batch alternation — PSR head goes DEAD on det-only steps

### PSR Fixes to Consider

1. **Enable sequence mode for PSR** — already partially done (seq_every=2)
2. **Raise PSR sensitivity weight** — config.py: `psr_sensitivity_weight: 0.50` in RF4
3. **Increase PSR_SEQ_LOSS_SCALE** — currently ~1.0 (hardcoded in loss config)
4. **Lower PSR_FOCAL_GAMMA** from 0.5 to 0.25 — easier positive gradient for rare steps

Opus (63 Q13): "The PSR ALIVE/DEAD oscillation is the seq/det batch alternation.
Don't chase it."

## Questions for Opus

1. **Simple head validation threshold:** What specific metric should we use to decide
   "simple head works" vs "simple head also collapses"? If act_macro_f1 > 0.01 with
   >10 diverse predicted classes, is that sufficient? Or do we need act_top1 > 0.05?

2. **Detection ceiling:** Opus estimates 0.20-0.30 mAP50 for our joint model. Is this
   on 50% data (RF4) or 100% data (RF10)? If 0.30 requires 100% data, the RF4 gate
   of 0.20 mAP50_pc at 50% data may need a lower threshold.

3. **Should we reduce scope to skip PSR entirely?** If PSR F1 stays at 0.0, we have
   4 working tasks (det + pose + head_pose + act with simple head). Is that sufficient
   for the paper? Or does PSR add critical value for the assembly monitoring claim?

4. **What LR schedule is optimal for the simple head?** Currently OneCycleLR with
   pct_start=0.1, max_lr=5e-5/5e-4. For a 150K-param MLP with Xavier init and
   bias=-0.5, should we use a different LR or schedule?

5. **Data augmentation strategy:** We have USE_SPATIAL_AUG=True but no RandAugment,
   MixUp, or CutMix for activity. With 3.7k frames and 72 classes, data augmentation
   could be more valuable than architecture changes. What augmentation helps most
   for this specific long-tail setup?

6. **When to enable sequence_mode for all heads?** RF4-RF10 currently use
   STAGED_TRAINING=False, which means per-frame mode. At what data scale
   (65%? 80%? 100%?) should we switch to true sequence batches where the
   temporal head becomes useful? Or does the simple head outperform the
   temporal head even with sequence data?

7. **GPU 0 (RTX 3060) is still idle.** Opus says skip DDP. But with the simple
   head reducing activity params from 8.2M to 0.5M, the total model is now
   ~45M params instead of ~53M. Could we fit TWO batches per step on GPU 0
   using a naive split (GPU 0 handles half the batch)? Not DDP — just
   `torch.cuda.device` in the forward loop. Worth 30 minutes of engineering?

8. **Full dataset eval before paper:** At 100% data (RF10), the full val set is
   38K frames. At 1.2 batch/s with 0 workers, a full eval takes ~9 hours.
   Should we plan this as the FINAL step before paper writing, or can we get
   away with 500-batch gate evals for all results?

9. **Should we checkpoint EVERY stage's best model?** Currently RF stages share
   the same best.pth (overwritten each stage). If RF5's best is worse than
   RF4's best, we can't go back. Should we save stage_named checkpoints
   (best_rf4.pth, best_rf5.pth) for ablation comparison in the paper?

10. **Seed stability:** With SEED=42 fixed, are results deterministic? If we resume
    from an RF4 checkpoint and run RF5 three times with different seeds, do we
    get consistent results? For the paper, we need variance estimates.
