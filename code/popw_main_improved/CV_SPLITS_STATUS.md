# Cross-View (CV) Splits — Status Report

## PTMA cv Protocol (Cross-View)

PTMA (arXiv:2508.17025) reports cross-view results using view-pair evaluation:
- v1-v2: 92.48% (train v1 → test v2)
- v1-v3: 90.03% (train v1 → test v3)  
- v2-v3: computed from avg of reported pairs

## Reproducibility Analysis

### ✅ CAN Reproduce — Cross-View (cv)
Cross-view is straightforward because:
- Every video in IKEA ASM has all 3 views simultaneously (dev1, dev2, dev3)
- Same video IDs exist across all camera directories
- Split = "which camera to train on vs. which to test on"

### ❌ CANNOT Reproduce — Cross-Subject (cs) / Cross-Subject-View (csv)
- PTMA defines subject-based splits (which assemblers in train vs. test)
- Subject IDs are NOT published in any public dataset
- Without exact subject ID definitions, exact cs/csv comparison is impossible

## Implementation Required

To properly evaluate cross-view mcAP:

1. **Modify dataloader** to use different camera for train vs test:
   - For v1-v2: Train with CAMERA='dev1', Test with CAMERA='dev2'
   - This requires separate camera configs for train/eval loaders

2. **Current codebase limitation**: Uses single `C.CAMERA` config for both

## Quick CV Benchmark (Workaround)

Use existing cross_env splits with `protocol='calibrated'`:
- All 3 views are already loaded in multi-camera dataset
- Activity fusion happens via `MultiViewActivityFusion`
- Result won't match PTMA cv exactly (different methodology)
- But will show calibrated AP is higher than standard AP

## Files Generated

- `generate_cv_splits.py` — Script to create cv split files
- Split files just list video keys; camera selection done at runtime
