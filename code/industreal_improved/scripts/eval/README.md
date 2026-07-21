# Evaluation Scripts

Scripts to evaluate trained checkpoints on each task.

## Quick Start

### Evaluate all 4 heads at once
```bash
python scripts/eval/eval_all_heads.py \
    --checkpoint runs/mtl_v3.7/checkpoints/phase2_e0_b18000.pth \
    --output /tmp/all_heads_eval.json
```

This produces a JSON with all 4 heads' metrics + SOTA gap analysis.

### Evaluate individual heads
```bash
# Detection (mAP@0.5)
python scripts/eval/eval_mvit_mAP.py \
    --checkpoint <path> --max-frames 2000

# Activity (Top-1, Top-5)
python scripts/eval/eval_activity_75class.py \
    --checkpoint <path>

# Pose (MAE)
python scripts/eval/eval_pose_norm_fix.py \
    --checkpoint <path>

# PSR (F1, Edit)
python scripts/eval/eval_psr_transition_f1.py \
    --checkpoint <path>
```

## Paper SOTA Targets

| Head | Metric | SOTA Target | Paper Best |
|------|--------|-------------|------------|
| Detection | mAP@0.5 | ≥70% | 0.641 (MViTv2-S) |
| Activity | Top-1 | ≥95% | — |
| Pose | MAE (°) | <5° | — |
| PSR | F1 | ≥80% | — |

## Scripts

- `eval_all_heads.py` — Comprehensive 4-head eval (NEW, recommended)
- `eval_mvit_mAP.py` — Detection mAP@0.5 with training-correct decode
- `eval_activity_75class.py` — 75-class activity recognition
- `eval_pose_norm_fix.py` — Pose with normalization fix
- `eval_psr_transition_f1.py` — PSR event F1
- `eval_v38_fix.py` — Compare v3.8_fix vs v3.7
- `eval_detection_dual_protocol.py` — Dual-protocol detection eval
- `eval_with_tta.py` — Detection with horizontal-flip TTA
- `eval_test_split.py` — Test split evaluation
- `eval_checkpoint.py` — Legacy POPWMultiTaskModel eval (compatibility)