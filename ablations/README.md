# Ablation Experiments — Tier F Experiment Matrix (175 6, rows 7-10)

## Overview

Five ablation runs that prove the multi-task learning mechanism on IndustReal.
Each is a variant of the full MTL-All model (shared backbone + 4 heads) differing
in exactly one variable: which heads are present, whether the backbone is frozen,
or which backbone architecture is used.  Comparing each to the MTL-All reference
identifies the source of positive transfer.

Reference: `code/industreal_improved/analyses/consult_2026_06_10/AAIML/175_ULTIMATE_GUIDE_TIER_F.md` section 6.

## The Five Ablations

### 1. LOO-noDet (Leave-One-Out: remove detection)

MTL-All with the detection head removed.  If detection provides the hierarchical
features that PSR aggregates (the mechanism in 175 1), then removing detection
should cause a measurable drop in PSR event-F1.  This is the central attribution
experiment: a PSR drop here proves the detection-to-PSR hierarchy is real.

Expected outcome: PSR drops (event-F1 lower than MTL-All).  Activity and pose
are approximately unchanged.

### 2. LOO-noPose (Leave-One-Out: remove pose)

MTL-All with the pose head removed.  Pose is hypothesised to be an egocentric
attention prior (175 1) that biases the backbone toward the manipulated
workspace region.  If this mechanism is real, removing pose should degrade
detection and/or activity; if pose is dead weight, removing it leaves
everything unchanged or slightly improved.

Expected outcome: unclear -- either no change (dead weight) or a small
detection/activity drop (useful auxiliary).

### 3. MT-frozenBB (Frozen backbone)

MTL-All with all backbone parameters frozen.  Only the task heads train.
This replicates the condition that caused V5 activity to fail: a frozen ConvNeXt
probe cannot learn the activity task because activity requires temporal features
the backbone was not asked to develop.  The readout is per-head loss
trajectories: activity and PSR should stay flat or diverge.

Expected outcome: activity and PSR losses collapse (fail to decrease).
This explains V5's failure mode.  The fix is unfreezing the backbone.

### 4. Backbone-swap ConvNeXt

MTL-All using ConvNeXt-Tiny (ImageNet-pretrained, 2D spatial backbone) instead of
Hiera-B (MAE video-pretrained, hierarchical spatiotemporal backbone).  All heads,
training schedule, and data are identical.  The only difference is the backbone.

Expected outcome: Hiera outperforms ConvNeXt on activity and PSR because Hiera
processes temporal structure.  Interference ranking between heads may shift,
proving that interference is representation-mediated (not intrinsic to the
task set).  This is described as "the single best figure" in 172 G2.

### 5. Backbone-swap Hiera

MTL-All using Hiera-B (the primary Tier F backbone).  This is the reference
run that backbone-swap ConvNeXt is compared against.  Together they form a
paired experiment: same data, same schedule, same heads, different backbone.

Expected outcome: Hiera > ConvNeXt on activity and PSR.  Execution cost
(feature FLOPs, memory) is also measured as part of the efficiency analysis.

## How to Invoke

All runs use the same entrypoint:

```
python scripts/run_ablation.py --ablation <name> --epochs N [--output <dir>]
```

| Ablation | Flag | Default output dir | Expected runtime (5 epochs) |
|---|---|---|---|
| LOO-noDet | `--ablation loo-no-det` | `rf_stages/checkpoints/loo_no_det` | ~5-10 minutes on 1 GPU |
| LOO-noPose | `--ablation loo-no-pose` | `rf_stages/checkpoints/loo_no_pose` | ~5-10 minutes |
| MT-frozenBB | `--ablation mt-frozen-bb` | `rf_stages/checkpoints/mtl_frozen` | ~5-10 minutes |
| Backbone-swap ConvNeXt | `--ablation backbone-swap-convnext` | `rf_stages/checkpoints/bs_convnext` | ~5-10 minutes |
| Backbone-swap Hiera | `--ablation backbone-swap-hiera` | `rf_stages/checkpoints/bs_hiera` | ~5-10 minutes |

Each run creates:
- `latest.pth` -- final model checkpoint (state dict + metadata)
- `best.pth` -- best model checkpoint by mean validation loss
- `metrics.json` -- per-epoch train/val losses for every active head
- `summary.json` -- compact run summary for cross-run comparison

## Expected Runtimes

Plumbing verification (--epochs=1): under 2 minutes per run on a single GPU.
Full runs (--epochs=50-100): approximately 1-4 hours per run depending on
backbone.  ConvNeXt is faster per-step than Hiera-B due to lower parameter
count and 2D-only processing.

## Read-outs (per 175 6)

The metrics.json files from all five runs + MTL-All reference are consumed by
the analysis notebook (or `generate_ablation_table.py`) to produce the paper's
Table 3:

| Run | Det mAP@50 | Act top-1 | PSR event-F1 | Pose MAE |
|---|---|---|---|---|
| MTL-All | baseline | baseline | baseline | baseline |
| LOO-noDet | N/A | ... | expected lower | ... |
| LOO-noPose | ... | ... | ... | N/A |
| MT-frozenBB | ... | expected collapse | expected collapse | ... |
| BS-ConvNeXt | ... | expected lower | expected lower | ... |
| BS-Hiera | baseline | baseline | baseline | baseline |

The transfer column (MTL-All minus ST- baselines) is computed separately from
the single-task runs (rows 1-4 of the experiment matrix).
