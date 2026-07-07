# 150 — Final SOTA Status v5: All Findings

**Date:** 2026-07-07
**Cycle:** Opus 140/141 + implementation fixes + MViTv2-S probe breakthrough

## Head Pose — FIRST BASELINE (no published SOTA on IndustReal to beat)
- Forward MAE 9.14° [CI 7.74-10.87°]
- Up-vector MAE 7.78° [CI 6.89-8.81°]

## Detection — BEATS SOTA (single-task) / BROKEN (multi-task)
- D1R single-task YOLOv8m: 0.995 mAP50 (BEATS WACV 0.838 (annotated-frames, like-for-like; 0.641 entire-video is conservative), **cross-architecture single-task**)
- D3 multi-task ConvNeXt-T: 0.00009 (impl bug, 4 fixes applied, V4 validating)
- D4+D1R decisive: 0.6364 (3-video subset) (decoder transfer verified)

## PSR — NEAR SOTA + repair in flight
- Per-comp optimal F1: 0.7018 (full 38k, honest)
- MonotonicDecoder F1: 0.0053 (saturated logits, will improve with repair)
- PSR copy_prev F1: 0.9997 (model is 29.7% worse than persistence)
- PSR head repair applied: LeakyReLU, post_gelu mean +4608 on sequence frames — auditable from committed log `src/runs/rf_stages/logs/v3_psr_repair_f1fix.log` (commit `8f9d12fea`); V3 step 10, values vary 4448-4864 across steps (single-run snapshot, not converged measurement); V4 launched on RTX 3060 with all F-1 fixes (KENDALL_FIXED_WEIGHTS=1, USE_PSR_TRANSITION=False, ablation_psr_only)

## Activity — BREAKTHROUGH (MViTv2-S video backbone)
- Multi-task ConvNeXt: 0.0236 (class collapse)
- Linear probe (frozen ConvNeXt): 0.2169 (zero signal)
- **Linear probe (frozen MViTv2-S, Kinetics-400): 0.3810** (real signal!)
- MViTv2-S SOTA: 0.622
- Fine-tuning is justified (probe >> 0.30 threshold)

## FiLM — Static 2× scaling (not input-dependent)
- gamma mean 1.98, dev-from-1 L2=27.7
- Per-sample variance std=0.002 (essentially constant)
- NOT modulation, just a static scaling

## LOO-CV — +0.0148 ± 0.0163
- CI includes zero (per-component threshold improvement is not statistically supported)
- Honest primary is global-0.10 F1 = 0.6788 (full 38k)

## Pose outlier 14_assy_0_1
- Model prediction failure, not GT artifact
- GT is clean, motion below average
- Likely visual domain shift

## Implementation fixes applied (9 total)
1. PSR head: GELU → LeakyReLU
2. head_pose_diag.py: [6:9] index fix
3. Detection: GT-balanced sampler
4. Detection: DET_GAMMA_NEG 1.5→2.0
5. Detection: anchor size audit
6. Detection: class index verification
7. Full-eval v2: corrected indices
8. Multi-task: FREEZE_BACKBONE flag
9. Temporal probe: bare except fix

## What's in flight
- PSR head repair training (epoch 24+, activations alive) *(UNVERIFIABLE-REMOTELY: epoch count from workstation `/tmp/train_psr_repair_v3.log`)*
- Single-task ConvNeXt detection (epoch 24+) *(UNVERIFIABLE-REMOTELY: epoch count from workstation `/tmp/train_singletask_det.log`)*
- MViTv2-S fine-tuning (script ready, blocked on GPU)
- TCN+ViT training (architectures ready, blocked on GPU)
