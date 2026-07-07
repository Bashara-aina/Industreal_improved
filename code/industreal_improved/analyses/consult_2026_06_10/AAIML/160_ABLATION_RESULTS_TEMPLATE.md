# 160 — Ablation Results: 2x2 Matrix Across All 4 Heads (Template)

**Date:** (when results come in)
**Source:** /media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/runs/rf_stages/checkpoints/

## §1. Per-Head Results (to fill in when ready)

### §1.1 Detection
- D3 multi-task (current): 0.00009 mAP
- D3 multi-task (with 4 fixes): ?
- Single-task ConvNeXt: ?
- D1R (YOLOv8m): 0.995 mAP
- WACV 2024: 0.641 (entire-video) / 0.838 (annotated)
- Verdict: ?

### §1.2 Activity
- Multi-task (current): 0.0236
- Multi-task (with all fixes): ?
- Single-task ConvNeXt: ?
- Single-task MViTv2-S frozen: 0.3810
- Single-task MViTv2-S fine-tune: ?
- WACV MViTv2-S: 0.622
- Verdict: ?

### §1.3 PSR
- Multi-task (current): 0.7018
- Multi-task V5 (with all fixes): ? (V5b in progress, GPU 0, resumed from epoch 33; ETA epoch 50: ~22h)
- Single-task ConvNeXt: ?
- Decoder (no learning): 0.0053 full-38k
- null_copy_prev: 0.9997
- STORM B3: 0.883 (transition, diff paradigm)
- Verdict: ?

### §1.4 Pose
- Multi-task: 9.14° fwd / 7.78° up
- Multi-task (Kalman): 9.00° / 7.58°
- Single-task: ? (target 5-7°)
- Verdict: ?

## §2. The 2x2 Verdict

| Head | Multi-Task | Single-Task | Ratio | Best |
|---|---|---|---|---|
| Detection mAP | 0.00009 (broken) | ? | ? | ? |
| Activity top-1 | 0.0236 | ? | ? | ? |
| PSR F1 | 0.7018 | ? | ? | ? |
| Pose MAE | 9.14° | ? | ? | ? |

## §3. The Final Paper Story

To be filled when results come in.
