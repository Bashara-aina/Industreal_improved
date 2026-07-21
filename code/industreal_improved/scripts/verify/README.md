# Verification Scripts

Sanity checks and validation tests.

## Scripts

- `verify_checkpoint.py` — Verify checkpoint loads + all 4 heads produce outputs
- `verify_act_grouping.py` — Verify activity grouping against class names
- `verify_gt_coordinates.py` — Verify ground truth coordinates
- `verify_subject_split.py` — Verify train/val/test subject splits
- `discover_test_subjects.py` — Discover test subjects in dataset
- `integration_test.py` — Full train_step integration test
- `test_anchor_normalization.py` — Test anchor normalization

## Usage

```bash
# Verify checkpoint integrity
python scripts/verify/verify_checkpoint.py runs/mtl_v3.7/checkpoints/phase2_e0_b18000.pth

# Verify subject splits
python scripts/verify/verify_subject_split.py

# Run integration test
python scripts/verify/integration_test.py
```