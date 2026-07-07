# 152 — Implementation Bug Catalog: All 9 Fixes Applied

## §1. PSR Head (GELU Dead)

### §1.1 The Bug
- All 11 sub-heads had GELU zero-fraction > 0.97
- Pre-activations mean = -130
- +0.1 first-layer bias = 1300x too small
- Result: GELU was passing ~zero gradient through all training
- PSR F1 ≈ 0.9997 (null_copy_prev, persistence / copy-prev null) — model not learning

### §1.2 The Fix (e618d929a, 6defe1f5f)
```python
# src/models/model.py:1597-1604
nn.Sequential(
    nn.Linear(gru_hidden, 64),
    nn.LeakyReLU(negative_slope=0.01),  # was GELU()
    nn.Dropout(dropout * 0.3),
    nn.Linear(64, 1),
)
# And: nn.init.normal_(head[0].weight, std=0.01), zeros_(head[0].bias)
# And: nn.init.normal_(head[3].weight, std=0.01), zeros_(head[3].bias)
```

### §1.3 The Critical Additional Bug (Agent-75 found)
- DETACH_PSR_FPN = True default in config.py
- Detaches FPN features, breaking gradient flow from backbone to PSR head
- All 11 sub-heads gradient RMS = 0.00e+00 (DEAD)
- This is why the V1 PSR repair training didn't work

### §1.4 The V3 Fix (28bf668c)
```bash
# scripts/train_psr_repair_v3.sh
export DETACH_PSR_FPN=False
export KENDALL_FIXED_WEIGHTS=1
export AMP_DTYPE=bf16
# Result: post_gelu activations +4608 (was -1.0 to -1.4 dead) — auditable from committed log `src/runs/rf_stages/logs/v3_psr_repair_f1fix.log` (commit `8f9d12fea`)
```

### §1.5 File Paths
- Source: /media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/models/model.py:1597-1640
- Detection of bug: /media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/runs/rf_stages/checkpoints/psr_repair_training/
- V3 training: `src/runs/rf_stages/logs/v3_psr_repair_f1fix.log` (committed, commit `8f9d12fea`)
- V3 script: /media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/scripts/train_psr_repair_v3.sh

## §2. Detection (5 Classes Never Predicted)

### §2.1 The Bug
- Classes 1, 13, 16, 19, 23 have ZERO predictions across 38k frames
- Class 12 is "default catch-all" for 7 different states
- 91.9% of frames have zero GT (3,102 GT boxes / 38,036 frames)
- Positive gradient on only 8% of batches

### §2.2 The Fixes (8cef56fc2, cd901f655, 10d5ab596, a0ffb9aa8)

**Fix 1: GT-balanced sampler (8cef56fc2)**
- File: /media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/data/industreal_dataset.py
- New class: `GuaranteedGTBatchSampler`
- Replaces non-GT index with random GT index when batch has 0 GT
- Ensures 100% of batches have GT

**Fix 2: DET_GAMMA_NEG 1.5→2.0 (cd901f655)**
- File: /media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/config.py
- Harder negative mining via focal loss
- Suppresses 3.8M false positives

**Fix 3: Anchor size audit (10d5ab596)**
- File: /media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/config.py
- Re-ran k-means on 14,122 training GT boxes
- Current sizes adequate (>99% GT coverage at IoU>0.5)
- Not the root cause

**Fix 4: Class index verification (a0ffb9aa8)**
- File: /media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/config.py
- Added runtime assertions: DET_CLASS_NAMES has 24 entries, indices 1-24, mapping correct
- 5 never-predicted classes is a training convergence issue, not mapping bug

### §2.3 File Paths
- Source: /media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/models/model.py (DetectionHead)
- Loss: /media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/training/losses.py
- Root cause: /media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/runs/rf_stages/checkpoints/detection_root_cause/analysis.md
- Train script: /media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/scripts/train_singletask_convnext_det.sh

## §3. Activity (41/69 Zero-Accuracy Classes)

### §3.1 The Bug
- Per-frame MLP: 0.0236 (class collapse)
- 41 of 69 classes have zero accuracy
- Backbone: ImageNet ConvNeXt has no action semantics
- Linear probe (frozen): 0.2169 ≈ 0.2217 baseline (zero signal)

### §3.2 The Fix (NOT a code bug — a backbone type mismatch)
- Built MViTv2-S video backbone: src/models/video_backbones.py
- Built video backbone multi-task: src/models/video_backbone_multitask.py
- Built TCN/TCN+ViT: src/models/activity_tcn.py, activity_tcn_vit.py
- Linear probe MViTv2-S = 0.3810 (real signal)

### §3.3 File Paths
- Source: /media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/models/model.py (activity head)
- Video: /media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/models/video_backbones.py
- TCN: /media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/models/activity_tcn.py
- Probe: /media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/evaluation/activity_mvit_probe.py
- Per-class: /media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/runs/rf_stages/checkpoints/activity_per_class/

## §4. Pose Index Bug (3.5-Month Stale Number)

### §4.1 The Bug
- Eval scripts used `hp[:, 3:6]` (position data) as up-vector
- Up-vector is at `[:, 6:9]` (correct)
- Result: 26.20 reported as up-vector MAE
- Should have been 7.78

### §4.2 The Fix (bff38b790)
- File: /media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/evaluation/head_pose_diag.py
- Changed indices to [0:3] fwd, [6:9] up, [3:6] position
- Also deprecated the script (use eval_pose_kalman.py for reporting)

### §4.3 The Training Loss Verification
- File: /media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/training/losses.py:951-952
- Training loss was ALWAYS using correct [6:9] indices
- Only the eval scripts were buggy
- Model was always trained correctly

## §5. Multi-Task Training: FREEZE_BACKBONE Flag

### §5.1 The Bug (Discovery)
- Multi-task training had no flag to control backbone freezing
- Hardcoded backbone LR multiplier = 0.1 in train.py
- No way to do single-task training cleanly

### §5.2 The Fix (bc6bebdb7)
- File: /media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/config.py
- New: `FREEZE_BACKBONE = True` (default preserves existing behavior)
- New: `BACKBONE_LR_MULT = 0.01` (backbone LR = head LR * this)
- File: /media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/training/train.py
- Modified param-group creation: backbone included only if `FREEZE_BACKBONE=False`
- File: /media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/scripts/train_finetune_backbone.sh
- New launch script: `FREEZE_BACKBONE=False python src/training/train.py`

## §6. The Comprehensive Fix List (All 9)

| # | Fix | Commit | File | Effect |
|---|---|---|---|---|
| 1 | PSR head GELU->LeakyReLU | e618d929a | src/models/model.py:1597-1640 | Activations -130 -> +4608 — auditable from committed log `src/runs/rf_stages/logs/v3_psr_repair_f1fix.log` (commit `8f9d12fea`) |
| 2 | PSR head init index fix | 6defe1f5f | src/models/model.py | Sequential [3]=Linear, [2]=Dropout |
| 3 | Pose diag index fix | bff38b790 | src/evaluation/head_pose_diag.py | 26.20 -> 7.78 |
| 4 | Detection GT-balanced sampler | 8cef56fc2 | src/data/industreal_dataset.py | 100% batches have GT |
| 5 | Detection gamma_neg 1.5->2.0 | cd901f655 | src/config.py | Suppress false positives |
| 6 | Detection anchor audit | 10d5ab596 | src/config.py | Confirmed not root cause |
| 7 | Detection class index verify | a0ffb9aa8 | src/config.py | Mapping correct |
| 8 | Full-eval v2 corrected | 216566da0 | src/evaluation/full_eval_stream.py | 9.14/7.78 verified |
| 9 | Multi-task FREEZE_BACKBONE | bc6bebdb7 | src/config.py + train.py | Single-task enabled |
| 10 | PSR V3 DETACH fix | 28bf668c | scripts/train_psr_repair_v3.sh | Gradient flows |

## §7. The Decisive Question

**With all 9 fixes applied:**
- Detection mAP expected: 0.5-0.7 (from 0.00009)
- PSR F1 expected: 0.78+ (from 0.7018, if V3 works)
- Activity expected: 0.45-0.55 (with MViTv2-S fine-tune)
- Pose expected: 9.14 (no change)

**Can we prove multi-task HELPS after all 9 fixes?**
- The single-task vs multi-task ablation (when GPU free) is the definitive test
- If multi-task with all 9 fixes + V3 > single-task: multi-task theory is right
- If single-task > multi-task with all fixes: implementation was the killer

The user has green-lit "do the best". The path is clear: 4 weeks of focused work.
