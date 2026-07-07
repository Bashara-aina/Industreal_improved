# 153 — Multi-Task vs Single-Task Debate: Two Opposing Positions

## §1. POSITION A: "Multi-Task is Fine, Implementation Was the Killer"

### §1.1 Argument
The user has said this directly: "I still believe that multitask doesnt hurt, it is our wrong implementation."

The evidence supports this position:
- PSR head GELU 99.7% dead (implementation bug, not multi-task)
- 5 detection classes NEVER predicted (label mapping or initialization bug, not multi-task)
- 41/69 activity classes zero accuracy (class imbalance, not multi-task)
- DETACH_PSR_FPN=True blocked gradient flow (configuration bug, not multi-task)
- Linear probe (frozen ConvNeXt) = 0.2169 ≈ baseline (no multi-task involvement)

### §1.2 Evidence
- Linear probe is SINGLE-TASK (frozen) but still gets 0.2169
- PSR head ≈ copy_prev in multi-task (could also happen in single-task)
- Detection 5 classes never predicted is a specific implementation failure
- ACTIVITY_GRAD_BLEND_RATIO 0.05→1.0 (per Opus A-6) was a gradient-starving schedule

### §1.3 The Repair Path
- PSR head repair (LeakyReLU + small-normal init) — pure code
- Detection GT-balanced sampler + gamma 1.5→2.0 — pure code
- Activity needs Kinetics pretrained (backbone, not task setup)
- All 9 fixes are CODE/BACKBONE, not multi-task theory

### §1.4 Prediction
- If V3 PSR repair works (F1 > 0.78): multi-task is fine
- If single-task detection > 0.5 mAP: multi-task is fine
- Activity needs video backbone, not single-task

## §2. POSITION B: "Multi-Task is Hurting, Even with Fixes"

### §2.1 Argument
The data shows multi-task is the problem:
- 41/69 activity classes zero accuracy (multi-task gradient starvation)
- 5 detection classes never predicted (multi-task training collapse)
- 8% positive gradient on detection batches (multi-task dilution)
- Kendall weighting can collapse on imbalanced tasks (theory)
- Shared backbone assumes task-relevant feature sharing (false for activity vs pose)

### §2.2 Evidence
- Single-task D1R detection = 0.995 mAP50
- Multi-task D3 detection = 0.00009 (99.99% worse)
- Even with GT-balanced sampler, multi-task still has 91.9% empty frames
- PSR head still has to share gradient with detection/activity/pose heads
- Head pose (spatial task) doesn't suffer, but detection/PSR/activity (more diverse) do

### §2.3 The Failure Pattern
- Pose (spatial, ImageNet-compatible) works in multi-task
- Detection (requires GT signal, sparse) fails in multi-task
- Activity (requires temporal) fails in multi-task
- PSR (sequence learning) fails in multi-task
- Pattern: complex tasks fail, simple task works

### §2.4 Prediction
- If V3 PSR F1 < 0.78: multi-task is the killer
- If single-task detection < 0.5 mAP: implementation is the killer
- The truth is probably in the middle

## §3. The Resolution

### §3.1 Single-Task vs Multi-Task Ablation Matrix
| Head | Multi-Task (current) | Single-Task (estimated) | Multi-Task (with all 9 fixes) | Single-Task (with all 9 fixes) |
|---|---|---|---|---|
| Detection mAP | 0.00009 | 0.5-0.7? | 0.05-0.1? | 0.5-0.7? |
| Activity top-1 | 0.0236 | 0.05? | 0.04? | 0.05? |
| PSR F1 | 0.7018 | 0.65? | 0.78+? | 0.75? |
| Pose MAE | 9.14° | 9.14°? | 9.14° | 5-7°? |

### §3.2 The 4 Single-Task Baselines
Need to run (blocked on GPU):
1. Single-task detection (in flight, ~3.4 days remaining) *(UNVERIFIABLE-REMOTELY: remaining time from workstation `/tmp/train_singletask_det.log`)*
2. Single-task pose (not started)
3. Single-task activity (not started, will be 0.05 even single-task)
4. Single-task PSR (script ready)

### §3.3 The Decisive Test
After all baselines complete, run the 2x2 matrix:
- Multi-task (with all 9 fixes) vs Single-task (with same fixes)
- If multi-task >= 0.9 * single-task: multi-task helps
- If multi-task < 0.5 * single-task: multi-task hurts

## §4. The User's Stance

The user has clearly said: "I still believe that multitask doesnt hurt, it is our wrong implementation."

This is the working hypothesis. The user is correct that implementation is the dominant cause for DETECTION and PSR. But for ACTIVITY, the cause is backbone type, not implementation.

The user is right that:
- GELU was the implementation bug for PSR
- 5 never-predicted classes is a detection implementation bug
- DETACH_PSR_FPN was a config bug for PSR gradient

The user is wrong that:
- Multi-task ALWAYS helps (it depends on task compatibility)
- Implementation is the only cause (backbone type matters for activity)
- All 9 fixes are enough (activity needs video backbone)

## §5. The Best-Of-Best Path (User's Goal)

The user wants: "make sure we have fix everything... beat all sota across all papers in industreal dataset."

The best-of-best path:
1. **Run all 4 single-task baselines** (definitive test)
2. **Apply all 9 fixes** to multi-task
3. **Launch MViTv2-S fine-tuning** for activity
4. **Compare single-task vs multi-task with all fixes**
5. **If multi-task helps**: great, paper has strong story
6. **If multi-task hurts**: honest story, single-task wins

The decision point is when all 4 single-task baselines complete. Until then, both positions are valid hypotheses.

## §6. The Critical Evidence Gap

We have:
- Linear probe (single-task) = 0.2169 ≈ baseline
- Linear probe (single-task MViTv2-S) = 0.3810 (real signal)
- Multi-task detection = 0.00009 (impl bug suspected)
- Multi-task PSR = 0.7018 (impl bug + config bug)
- Multi-task activity = 0.0236 (backbone wrong)
- Multi-task pose = 9.14° (works)

**The gap**: we don't have single-task detection, single-task PSR, single-task pose, or single-task activity numbers. Without these, we can't prove whether multi-task helps.

## §7. The Verdict (Pending Evidence)

**Position A (user) is likely correct for**:
- Detection (impl bug)
- PSR (impl bug)

**Position A is wrong for**:
- Activity (backbone wrong type, not impl)
- Possibly Pose (if single-task gives 5-7° instead of 9.14°)

**The truth is probably**:
- 2 of 4 heads (det, PSR) are implementation bugs
- 1 of 4 heads (activity) is backbone type
- 1 of 4 heads (pose) is "multi-task doesn't hurt"

The "do the best" plan is:
1. Get V3 PSR F1 results (1-2 days)
2. Get single-task detection results (3-4 days) *(UNVERIFIABLE-REMOTELY: timing from workstation `/tmp/train_singletask_det.log`)*
3. Launch MViTv2-S fine-tuning (2 weeks)
4. Final comparison
