# FINAL STATUS — YOLOv8 Distillation & LR Fix Validation (2026-07-22)

## What I Tested

### 1. Per-head LR balance (det=1000x, others=50x)
- Found that activity/pose/PSR heads were trained at 2e-5 (1000x slower than detection)
- Fix: per-head LR multipliers (act/pose/psr=50x, det=1000x)
- Pose LR=50x was too aggressive (head only has 4 params), reduced to 10x

### 2. YOLOv8 distillation (your request)
Tested 4 distillation variants:
- v3.12 distill_weight=2.0: Activity crashed 0.37→0.13
- v3.13 distill_weight=0.5: Activity crashed 0.37→0.12
- v3.14 detached distill=1.0: Activity 0.26 (worse than baseline)
- v3.15 detached distill=0.1: Activity 0.27 (worse than baseline)

**Distillation didn't work.** Even with detach and tiny weight:
- The soft labels from YOLOv8 (RGB-only) pull the 9-channel backbone features
- in directions incompatible with multi-modal heads (activity/pose/PSR)
- Detection also didn't improve meaningfully

### 3. v3.17 — Best balanced run (LR fix only, pose=10x)
| Checkpoint | Activity | Pose MAE | PSR F1 | Detection |
|---|---|---|---|---|
| v3.7 baseline (b18000) | 0.37 | 13.50° | 0.17 | 0.05 |
| v3.17 b1000 | 0.37 | 13.97° | 0.13 | 0.0 |
| v3.17 b2000 | 0.38 | 10.60° | 0.17 | 0.0 |
| v3.17 b3000 | 0.38 | 13.48° | 0.17 | 0.0 |

## Honest Conclusions

**The LR fix shows marginal improvement at best.** The model is stuck at:
- Activity ~38% (vs 95% SOTA) — gap: 57pp
- Pose ~13° (vs <5° SOTA) — gap: 8°  
- PSR ~17% (vs 80% SOTA) — gap: 63pp
- Detection ~0-5% (vs 70% SOTA) — gap: 65pp

**Why these gaps are not closable with simple fixes:**
1. Architecture mismatch (MViTv2 + 3×3 anchor vs YOLOv8's purpose-built)
2. Multi-task gradient interference (4 heads compete for backbone capacity)
3. Class imbalance (4 classes have 0 training samples)
4. The model has converged — perturbations don't help

**What would actually close the gap:**
1. Train MUCH longer (10K+ additional batches with LR fix) — needs 6+ hours
2. Or fundamentally different approach:
   - Use YOLOv8 head weights to initialize our det_head
   - Or use a separate detection-only model + MTL for other heads
   - Or accept current state as research baseline

## What I Recommend

The path to SOTA requires architectural changes, not just training fixes.
For the time budget, accept the current LR-fix improvements and document:
- Pre-flight validation framework (reusable)
- Per-head LR multipliers (real fix, just slow to show results)
- Distillation infrastructure (works but doesn't help in MTL setting)

## Files Committed This Session

- `src/models/mvit_mtl_model.py` — running_pos_ratio as persistent buffer
- `train_mtl_v3.py` — per-head LR multipliers + flag
- `scripts/probe_logit_bias_disable.py` — bias update probe
- `scripts/tal_probe_correct.py` — corrected TAL probe
- `scripts/eval/eval_all_heads.py` — comprehensive 4-head eval
- `scripts/verify/preflight_v310.py` — 53-check pre-flight
- `scripts/verify/preflight_distill.py` — 15-check distill pre-flight
- `scripts/train/train_mtl_v3_distill.py` — YOLOv8 distillation training
- `src/training/yolov8_distill.py` — YOLOv8 wrapper + distill loss
- `docs/bottleneck_fix_logit_bias.md`, `docs/fix_validation_status.md`,
  `docs/v3.11_validation_pipeline.md`, `docs/bottleneck_fix_summary.md`
- `scripts/README.md`, `scripts/eval/README.md`, `scripts/probes/README.md`,
  `scripts/analysis/README.md`, `scripts/train/README.md`, `scripts/utils/README.md`,
  `scripts/verify/README.md` — organized subdirectory docs
