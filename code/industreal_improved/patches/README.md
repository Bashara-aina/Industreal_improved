# V3 Patches — README

## Status
**PRE-DRAFT** — DO NOT APPLY until V2 smoke test (PID 2272603) completes.

## Why V3?
V2 baseline (epoch 19) val metrics showed:
- `det_mAP50 = 0.00209` (paper benchmark: ~0.40-0.60 for ASD)
- `score_max ~0.07-0.09` (never >0.30) — **confidence collapse**
- `act_accuracy = 0.0` (collapsed at ep17, was 0.5 at ep16)
- `combined = 0.103` (plateauing)

Root cause: `FOCAL_ALPHA = 0.25` is too low. The model is rewarded for predicting
LOW confidence because positives only contribute 0.25× to loss. Raising to 0.5
should give equal weight to positives/negatives, allowing the model to learn
to push confidence up.

## What's in this directory

| File | Purpose |
|------|---------|
| `01_BACKUP.sh` | Snapshots config.py and evaluate.py before patching |
| `02_PATCH_FOCAL_ALPHA.sh` | FOCAL_ALPHA 0.25 → 0.5 in `src/config.py:345` |
| `03_PATCH_DET_PROBE_NOGT.sh` | Distinguishes "no GT in batch" from "model collapse" in DET_PROBE verdicts |
| `04_REVERT.sh` | Restores the snapshot taken by `01_BACKUP.sh` |
| `05_LAUNCH_V3.sh` | Resumes from V2's final ckpt with patches applied |

## V2 vs V3 expected comparison

| Metric | V2 (ep19) | V3 (expected after 1-3 epochs of patch effect) |
|--------|-----------|-------------------------------------------------|
| score_max | 0.07-0.09 | >0.30 |
| det_mAP50 | 0.0021 | >0.005 (10x easier target with stronger pos signal) |
| act_accuracy | 0.0 | >0.0 (recovery from ep16 spike) |
| combined | 0.103 | >0.12 |

## Launch sequence (manual)

```bash
# 1. Confirm V2 has finished (PID 2272603 not running)
ps -p 2272603 2>&1 | grep -q "python3" && echo "V2 still running — wait" || echo "V2 done"

# 2. Backup
cd /home/newadmin/swarm-bot/project/popw/working/code/industreal_improved/v3_patches
bash 01_BACKUP.sh

# 3. Apply patches
bash 02_PATCH_FOCAL_ALPHA.sh
bash 03_PATCH_DET_PROBE_NOGT.sh

# 4. Launch V3
bash 05_LAUNCH_V3.sh

# 5. Monitor
tail -f $(ls -t src/runs/full_multi_task_tma_tbank_v3_alpha05_*/logs/v3_smoke_test_*.log | head -1)
```

## Revert (if V3 fails)

```bash
cd /home/newadmin/swarm-bot/project/popw/working/code/industreal_improved/v3_patches
bash 04_REVERT.sh
# Re-launch V2 from the same checkpoint if needed
```

## Open questions to investigate
1. Why did `act_accuracy` spike to 0.5 at ep16 then collapse to 0.0 at ep17-19?
2. Why are 12 of 24 det classes completely absent from val GT? (cls 12 dominates with 98 GTs)
3. If V3 (alpha=0.5) also plateaus, escalate to:
   - Class-balanced sampling (oversample cls 12 etc.)
   - Label smoothing (CB_LABEL_SMOOTHING already at 0.1 — try 0.05 or 0.15)
   - CB_BETA / CB_GAMMA tuning
