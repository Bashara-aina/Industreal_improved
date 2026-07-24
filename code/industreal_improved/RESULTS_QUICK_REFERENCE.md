# Quick Reference: Eval Scripts (paper-protocol compatible)

## What was fixed

| Bug | Status | Effect |
|-----|--------|--------|
| Eval scripts used bare `MTLMViTModel` while training saves via `WrappedMTL` (m.-prefix) | ✅ FIXED | Trained weights not loaded previously |
| Detection eval used coarse sigmoid>0.3 proxy | ✅ REPLACED | Real COCO mAP@0.5 with bbox decode+NMS+IoU match |
| AR eval computed per-frame Top-1 only | ✅ ADDED per-segment | Per-segment Top-1/Top-5 matching paper Table 2 |
| PSR eval computed per-frame macro F1 | ✅ ADDED sequence F1 | Sequence-level F1, POS, τ matching paper Tables 4,6,7 |

## Eval Scripts Overview

| Script | Speed (5 recordings) | Output |
|--------|---------------------|--------|
| `eval_mtl_AR_segment.py` | ~10 min | Top-1/Top-5 per segment |
| `eval_real_mAP.py` | ~35 min | mAP@0.5 + per-class APs |
| `eval_mtl_PSR_event_f1.py` | ~5 min | Sequence F1, POS, delay τ |
| `eval_mtl_with_gt.py` | ~25 min | Per-frame all-head metrics |
| `quality_check_10.py` | ~3 min | 10/10 structural verification |
| `benchmarks/run_full_benchmark.sh` | ~75 min | Runs all + JSON report |

## Shared Utilities
- `src/eval_utils.py` — `load_mtl_checkpoint()` consistently loads WrappedMTL + handles m. prefix

## Quick Run

```bash
# All benchmarks on v3.5 final
bash benchmarks/run_full_benchmark.sh runs/mtl_v3.5/checkpoints/phase2_e5_b0.pth

# Specific eval
python3 eval_mtl_AR_segment.py \
  --checkpoint runs/mtl_v3.5/checkpoints/phase2_e5_b0.pth \
  --output runs/eval/AR_segment.json
```

## Key Results Summary

| Metric | Value | Paper SOTA | Status |
|--------|-------|-----------|--------|
| AR Top-1 (per-segment) | 35.30% | 66.45% | 53% of SOTA |
| AR Top-5 (per-segment) | 68.47% | 88.43% | 77% of SOTA |
| ASD mAP@0.5 (COCO) | 0.0146 | 0.641 | 2.3% of SOTA |
| PSR F1 (sequence) | 0.050 | 0.901 | 5.5% of SOTA |
| PSR POS | 0.450 | 0.812 | 55% of SOTA |
| PSR delay τ | 0.37s | 22.4s | (low TP, stat. weak) |
| Activity per-frame Top-1 | 35.46% | N/A | 26× random |
| Pose fwd MAE | 14.01° | N/A | (not benchmarked) |
| Pose up MAE | 13.10° | N/A | (not benchmarked) |
| Per-frame PSR macro F1 | 0.866 | N/A | (per-frame, not comparable) |
| Quality 10-check | 10/10 PASS | N/A | All heads active |
