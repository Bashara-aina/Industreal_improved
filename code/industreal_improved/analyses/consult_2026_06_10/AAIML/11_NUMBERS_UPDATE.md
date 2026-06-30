# AAIML 2027 — 11: Performance Number Update After Training Reboot [2026-06-30]

## Critical: All Paper Numbers Must Be Re-Established

The AAIML paper currently references performance metrics (det mAP50=0.34, act_top1=18.3%,
head_pose=9.1°) from training runs BEFORE the Opus-verified fixes. EVERY number changed
in the last 12 hours. We must re-establish all metrics before writing.

## What Changed (and Why Numbers Will Differ)

| Old Architecture | New Architecture (Active Now) | Impact |
|:----------------|:-----------------------------|:-------|
| Activity: TCN+ViT (8.2M params), temporal bank with shuffled frames | Activity: Simple MLP (150K params): LayerNorm→Linear(512→256)→GELU→Dropout→Linear(256→75) | act_top1 **will change** — unknown direction. The old 18.3% may have been overfitting noise. New head has less capacity but cleaner gradient. |
| Detection trained with seq/det batch alternation competing with TCN+ViT gradient | Detection no longer competes with TCN+ViT (simple head uses far less gradient) | det_mAP50 **may improve** — less gradient contention in shared backbone |
| Head pose GT forward vectors NOT normalized (norm 0.014-0.030 instead of 1.0) | Head pose GT needs normalization fix before paper numbers are valid | 9.1° forward MAE **may degrade** after proper normalization |
| Gradient probe showed "312x gap" (was a measurement artifact) | Probe now understood — per-parameter norms ≠ head totals | No change to model, but we can ADD a findings paragraph |

## Current Training State (PID 3618126)

```
Stage: RF4 (50% data, all 5 heads, simple head active)
Epoch: 3/23 (batch ~300/3469)
Speed: 1.2 batch/s → ~48 min/epoch
Optimizer: AdamW, act=1x (5e-4), OneCycleLR
First validation with simple head: ~30 min from now
```

## What the AAIML Paper NEEDS (and When We Can Get It)

| Metric | Paper Needs | Current | When We'll Have It | Confidence |
|--------|:-----------|:--------|:-------------------|:-----------|
| det_mAP50 (standard) | ≥ 0.10 | 0.053 (old, ep2) | RF4 epoch 5-10 | Medium |
| det_mAP50 (present-class) | ≥ 0.20 | 0.079 (old, ep2) | RF4 epoch 10-15 | Medium |
| act_top1 | ≥ 10% | ~0.0 | **THIS EPOCH (3)** | Low — first signal in 30 min |
| act_top5 | ≥ 30% | 27.2% (old, ep2) | RF4 epoch 3 | Medium |
| head_pose forward MAE | ≤ 12° | 8.71° (needs normalization) | After normalization fix | Uncertain |
| psr_f1_at_t | > 0 | 0.0 | Needs sequence batches | Low |
| 93 GFLOPs, 53M params | Stable | Unchanged | Already known | High |
| 4.8 FPS on RTX 3060 | Stable | Unchanged | Already known | High |

## Timeline for Paper-Ready Numbers

```
NOW        RF4 epoch 3 validation → first simple head signal
+2h        RF4 epoch 4 → activity trend visible
+TODAY     RF4 epoch 5-6 → detection trend
+2 days    RF4 complete → all 5-head metrics for ablation analysis
+3 days    RF5-RF6 → data scale 65%, metrics improve
+1 week    RF7-RF8 → data scale 80%, head pose normalized
+2 weeks   RF9-RF10 → final paper numbers at 100% data
```

## Recommendation for the Paper Draft

Until we have stable numbers from the simple head:

1. **Write the paper with PLACEHOLDER metrics** marked clearly with `\todo{...}`
2. **Do NOT fill numbers** until RF4 completes (2 days)
3. **Head pose MAE**: Do NOT report 9.1° until GT normalization is verified
4. **Activity Top-1**: Report ONLY after at least 2 epochs with the simple head
5. **Detection**: Current det_mAP50=0.053 should be replaced — it's from RF3 where only
   3 heads were active, not all 5

## Old vs New: What the Paper Currently Says

| Section | Current Paper Text | Needs Update To |
|---------|-------------------|-----------------|
| Abstract | "activity Top-1 of 18.3%" | Simple head result (TBD) |
| Abstract | "head pose angular error of 9.1°" | Normalized GT result (TBD) |
| §4.2 Primary Results | det=0.34 pc, act=18.3%, pose=9.1° | All numbers from RF4+ run |
| §4.4 Ablation A: single vs multi | Δ = −0.03 mAP50_pc | Needs re-run with simple head |
| §4.5 Ablation B: FiLM | p = 0.032 for activity +2.2pp | Needs re-run with simple head |
| §3.3 Activity Head | "TCN + ViT temporal encoder" | "Per-frame MLP with LayerNorm" |
