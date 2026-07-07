# Final Session Status Report
**Date**: 2026-07-07 ~13:00 UTC

## GPU State
| GPU | Utilization | Memory Used |
|-----|-------------|-------------|
| 0   | 54%         | 3832 MiB    |
| 1   | 94%         | 5689 MiB    |

Both GPUs active. GPU 1 heavily loaded (94%), GPU 0 moderate (54%).

## Running Training Processes
1. **Single-task detection** (PID 811481, GPU 0):
   - `train_singletask_detection.py --batch-size 2 --no-staged-training`
   - Epoch 24, batch ~1770/13161 (~13.4% of epoch)
   - Running ~2118 seconds, speed ~0.8-0.9 batch/s
   - Log: `logs/train.log` (54,553 lines)

2. **PSR repair** (PID 813375, GPU 1):
   - `train_psr_repair_wrapper.py --preset stage_rf4 --batch-size 2`
   - Resumed from `crash_recovery.pth`
   - Active training in progress

## Commit Summary (This Session)
- **74 commits since yesterday**
- **321 total commits** in repository
- **HEAD**: `ef3497ccdf0d4885584ab2be53c5f044986fdcde`
- Branch: `auto/2pct-training-fix-20260520-202419`

### Recent Commits (tail of yesterday-forward)
- `bc7a03761` feat: comprehensive AAIML strategy files + eval pipeline + bug fixes
- `766a3099d` docs: 132 Opus answers — audit of 131 overview, top-10 verdicts
- `683919de8` Merge pull request #24
- `623c63fb2` docs: 133 Opus complete answers — all 66 questions, all 30 debates
- `6d8bc67bd` Merge pull request #25

### Latest Commits (top of log)
- `26ebc2e4a` feat: D3 detection 0.00009 root cause analysis
- `7e4a909b5` docs: opus_140_141_compliance/audit.md
- `590be5100` feat: PSR copy_prev deep analysis
- `f65fe6586` feat: activity MLP single-task training script
- `e8eca0de5` docs: video backbone work status check
- ... (74 total this session)

## Key Checkpoint Files
| File | Size |
|------|------|
| `best.pth` | 738 MB |
| `crash_recovery.pth` | 738 MB |
| `epoch_18.pth` | latest stable epoch |
| `epoch_1-11.pth` | epoch checkpoints |
| `yolov8m_industreal.pt` | backbone |
| `psr_data_cache_best.pth` | PSR cache |

## PSR Evaluation Results (completed earlier)
- Full-data global F1: **0.7013**
- Full-data optimal F1: **0.7810**
- LOO-CV global F1: **0.6827**
- LOO-CV optimal F1: **0.7186**
- LOO-CV improvement persists — threshold is real

## Other Artifacts
- 53 subdirectories in checkpoints/ (111 total items)
- Activity confusion matrix, linear probe results, CI bootstrap
- ConvNeXt PSR decoder, activity temporal probes, per-class metrics
- Multi-task training config, run logs for D1/D3 variants

## Status
- Training actively running on both GPUs
- 74 commits this session across feature, doc, and fix categories
- PSR repair training resumed from crash recovery
- Single-task detection at Epoch 24 progressing
