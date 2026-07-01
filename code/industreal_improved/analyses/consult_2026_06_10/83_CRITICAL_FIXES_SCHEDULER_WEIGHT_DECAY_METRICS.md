# 83 — Critical Fixes from 20-Agent Audit: Scheduler, Weight Decay, Missing Metrics [2026-07-01]

**Goal:** Document the 3 CRITICAL issues found by the 20-agent validation swarm that were fixed in commit `2e69b1e`. Opus must verify these are correctly resolved before training.

**Source files:**
- `src/training/train.py` (lines 1459-1475, 3595-3645, 4290) — GT fraction logging, bias/norm weight decay, OneCycleLR scheduler
- `src/evaluation/evaluate.py` (lines 3755-3766) — PSR component binary accuracy

---

## Fix 1: OneCycleLR Scheduler Stepping Bug (P0 — CRITICAL)

### The Bug

OneCycleLR was constructed with:
```python
# train.py ~3634 (BROKEN — before fix)
steps_per_epoch=len(train_loader) // train_accum_steps
# ≈ 800 steps per epoch
```
But `scheduler.step()` is called exactly **once per epoch** at `train.py:4290` (in the outer epoch loop, after `train_one_epoch` returns). This means:
- OneCycleLR's internal math computed `total_steps = 100 epochs × 800 steps/epoch = 80,000`
- With `pct_start=0.1`, the rising phase was `0.1 × 80,000 = 8,000 steps`
- But the scheduler received only **~100 total calls** (98 post-warmup)
- **Result: OneCycleLR stayed in its rising (warmup) phase for the entire 100-epoch training run**
- The LR never reached its peak, never entered cosine decay, the entire super-convergence schedule was defeated

The LinearLR warmup (2 epochs) peaked at epoch 1, then OneCycleLR ran at near-minimum LR for epochs 2-99. The model trained on a flat, sub-optimal LR for the entire run.

### The Fix

```python
# train.py ~3634 (FIXED — current code at HEAD 2e69b1e)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=max_lr,
    epochs=C.EPOCHS,
    steps_per_epoch=1,          # ← matches epoch-level stepping cadence
    pct_start=0.1,              # 10 warmup epochs, 90 cosine-decay epochs
    anneal_strategy='cos',
)
```

With `steps_per_epoch=1`:
- `total_steps = 100 × 1 = 100` — matches the number of `scheduler.step()` calls
- Rising phase: `0.1 × 100 = 10 epochs` (epochs 0-9: LR rises from ~5e-5 to 5e-4)
- Decay phase: epochs 10-99: LR cosine-decays from 5e-4 to ~0
- Combined with `SequentialLR([LinearLR(warmup, iters=2), scheduler])`: epochs 0-1 are linear warmup (0→5e-5), epochs 2-9 OneCycleLR rising phase (5e-5→5e-4), epochs 10-99 cosine decay

### Opus Verification
- [ ] Confirm `steps_per_epoch=1` at `train.py:~3644`
- [ ] Confirm `scheduler.step()` is called once per epoch at `train.py:4290`
- [ ] Confirm the comment explaining the fix is present above the OneCycleLR constructor
- [ ] At runtime: LR should be ~1e-5 at epoch 0, ~5e-5 at epoch 2, ~5e-4 at epoch 10, then decay

---

## Fix 2: Bias/Norm Weight Decay Exclusion (P1 — Performance)

### The Bug

All AdamW param groups shared the global `weight_decay=1e-3`. Standard practice excludes bias parameters and normalization weights from weight decay because:
- **Biases**: Have small dimension, regularizing them can shift the decision boundary
- **LayerNorm/BatchNorm weights**: Scaling factors that need freedom to adjust per-channel — weight decay forces them toward zero

### The Fix

```python
# train.py ~3595 (FIXED)
{'params': det_head_bias_params, 'lr': det_head_bias_lr, 'weight_decay': 0.0},
{'params': bias_params,          'lr': bias_lr,           'weight_decay': 0.0},
```

Per-group `weight_decay: 0.0` overrides the global `weight_decay=1e-3` for bias parameters. The `_effective_wd` global keyword on `AdamW()` still applies to all other groups (backbone, detection, activity, PSR, head params).

### Impact
- LayerNorm weights (all 5 heads + backbone) are no longer decayed toward zero
- Bias terms are no longer regularized
- Expected: slightly faster convergence, less normalization drift
- Cosmetic: optimizer log now includes `bias/norm WD=0`

### Opus Verification
- [ ] Confirm `det_head_bias_params` has `weight_decay: 0.0` at `train.py:~3601`
- [ ] Confirm `bias_params` has `weight_decay: 0.0` at `train.py:~3602`
- [ ] Confirm the optimizer log line now says "bias/norm WD=0"

---

## Fix 3: Missing Go/No-Go Metrics (P2 — Monitoring)

### Metric A: GT-Bearing Batch Fraction

**Why it matters**: `DET_GT_FRAME_FRACTION=0.40` targets 40% GT-bearing frames per batch. If the actual fraction falls far below this (e.g., <0.15), detection is starving for positive gradient. Previously this was not logged anywhere — the config target was set but the achieved fraction was invisible.

**The fix** (`train.py:1459-1475`):
```python
# Logged every 500 steps as part of [DET-HEALTH]
_gt_frames = sum(
    1 for t in targets.get('detection', [])
    if t.get('boxes') is not None and t['boxes'].shape[0] > 0
)
_total_frames = len(targets.get('detection', []))
if _total_frames > 0:
    logger.info(
        f'  [DET-HEALTH step={step}] det_gt_fraction: '
        f'{_gt_frames}/{_total_frames}={_gt_frames/max(_total_frames,1):.2f}  '
        f'  (target DET_GT_FRAME_FRACTION={getattr(C, "DET_GT_FRAME_FRACTION", 0.40):.2f})'
    )
```

**Expected**: At epoch 0 with `DET_GT_FRAME_FRACTION=0.40`, the `det_gt_fraction` should be ~0.35-0.45. If it's below 0.20, the sampler is not redistributing correctly.

### Metric B: PSR Component Binary Accuracy

**Why it matters**: The MD files specified "PSR mean binary accuracy >= 0.70" as a go/no-go criterion, but `compute_psr_accuracy()` existed but was never called from `evaluate_all`. The F1-based metrics (`psr_overall_f1`) were computed but binary accuracy is a more intuitive measure for quick assessment.

**The fix** (`evaluate.py:3755-3766`):
```python
_psr_pred_bin = (1.0 / (1.0 + np.exp(-all_psr_logits[..., :11]))) > 0.5
_psr_comp_acc = (_psr_pred_bin == all_psr_labels).mean()
results['psr_comp_acc'] = float(_psr_comp_acc)
logger.info(f'  PSR — Component Binary Accuracy: {_psr_comp_acc:.4f}')
```

**Expected**: At epoch 2, `psr_comp_acc` should be above 0.50 (better than chance). At epoch 100, target 0.75-0.85.

### Opus Verification
- [ ] At step 500 training output, confirm `[DET-HEALTH] det_gt_fraction` is logged
- [ ] At first validation, confirm `PSR — Component Binary Accuracy` is logged
- [ ] Confirm `act_pred_distinct` is logged in `[DIVERSITY]` line (from evaluate.py:3449 — this was already wired)

---

## Summary of All Fixes to Date (commits `f95a1aa` → `2e69b1e`)

| Commit | Fixes |
|--------|-------|
| `cb18506` | 7 agent-audit fixes: double-remap, WD 5e-2→1e-3, grad clip 1→5, GT frac 0.9→0.4, conditional TCN/ViT, GIoU floor 0→0.01, PSR warmup |
| `b6d4cce` | 2 Opus fixes: segment-label remap (Q5), per-class sampling mass log (Q1) |
| `f95a1aa` | 3 MD corrections: DETACH_REG_FPN, PSR focal claim, log-var device bug retraction |
| `832259f` | Pose confusion clarified: head pose vs body keypoints vs hand-film |
| `2e69b1e` | **3 CRITICAL: OneCycleLR scheduler, bias/norm WD, GT fraction + PSR acc metrics** |

**Total: 18 fixes applied across 5 commits since file 75.**
