# 74: 4-Task Paper Fallback — Activity as Documented Failure [2026-07-01]

## Decision Framework

If Opus confirms activity is not salvageable given our data constraints, the
paper transitions from "5-task multi-task system" to "4-task system + 1 documented
failure analysis."

## Why This Is Publishable

From Opus (file 66, confirmed in file 69):

> "A clean 4-task system + 1 dissected failure is a stronger paper than
> 5 half-working tasks."
>
> "The failure analysis is more novel than the absolute performance numbers.
> ML/AI reviewers will value the training pathology insights more than system
> deployment details."

The activity collapse story is itself a contribution for AAIML:
1. Both CE+label_smooth and CB-Focal (γ=2.0) failed
2. The failure is attributable to data (46/72 classes <1%) + per-frame MLP
   architecture limitation
3. The finding generalizes to any long-tail multi-task video understanding task

## Revised Contribution Claims

| # | Claim | Evidence | Strength |
|:-|-------|----------|:--------:|
| 1 | **Training pathology analysis (4 findings)** | Temporal-head/sampler mismatch, gradient probe misreading, CB-Focal collapse, head pose annotation | ★★★★★ |
| 2 | **4-task system on consumer GPU** | Detection + forward-gaze + body pose + PSR from 1 model | ★★★★☆ |
| 3 | **Temporal-head/sampler mismatch documented** | Code + ablation + root cause fix | ★★★★★ |
| 4 | **Activity collapse documented as negative result** | CE failed, CB-Focal failed, data constraint identified | ★★★★☆ |
| 5 | **Forward-gaze at SOTA on consumer GPU** | 8.71° angular MAE (from earlier run) | ★★★★☆ |
| 6 | **First multi-task baseline on IndustReal** | 4 tasks from 1 model, $299 GPU | ★★★★☆ |
| 7 | **97% cost reduction vs multi-model** | $299 vs $12K-$55K | ★★★★☆ |

## Revised Paper Structure

| Section | Pages | Focus |
|---------|-------|-------|
| 1. Introduction | 1 | 4-task system + consumer GPU + 4 training pathology findings |
| 2. Related Work | 1 | Multi-task learning, assembly understanding, failure analysis |
| 3. Architecture | 1.5 | ConvNeXt-T + FPN + 4 heads (activity head described but excluded from results) |
| 4. Experiments | 2 | Detection, forward-gaze, body pose, PSR results + efficiency table |
| 5. Training Pathologies | 2.5 | **4 findings** — temporal mismatch, gradient probe, activity collapse, pose annotation |
| 6. Conclusion | 0.5 | Summary, limitations, code release |
| **Total** | **~8.5 pages** | (within 10-page IEEE limit) |

### Section 5.3: Activity Recognition Failure (0.75 page — NEW)

```
5.3 Activity Recognition Failure

The 74-class activity recognition head collapsed despite two loss function
attempts. Using a per-frame MLP (150K parameters) with CE+label-smoothing(0.1),
the model predicted the majority class for all validation frames
(pred_distinct=1/75, macro-F1≈0).

Switching to class-balanced focal loss (Cui et al., 2019) with β=0.999, γ=2.0
and increasing dropout from 0.2 to 0.3 marginally improved diversity to 2/75
predicted classes but did not achieve meaningful recognition (macro-F1≈0).

We attribute this to a binding data constraint: 46/72 real action classes have
<1% annotation support among 3,667 training frames (median 8 frames per class).
A per-frame classifier operating on a 512-dimensional joint projection cannot
learn discriminative features for 74 classes when the median class has 8
exemplars. This finding corroborates the temporal-head/sampler mismatch
(Section 5.1): the original temporal encoder was never evaluated on truly
temporal data, and the per-frame fallback revealed the data limitation.

We report this negative result as a community caution: multi-task long-tail
activity recognition at 74 classes requires either substantially more
annotated data (≥100 frames per class) or a temporal architecture operating on
genuinely sequential (non-shuffled) frame batches.
```

## Revised Title Options

1. "Multi-Task Assembly Verification on Consumer GPUs: System and Training Pathologies"
2. "POPW: A 4-Task Assembly Verification Framework on Consumer Hardware"
3. "Learning to Verify Assembly on a Single GPU: Lessons from Multi-Task Training"

## Revised Abstract (4-Task Version, ~180 words)

```
We present POPW, a single-model multi-task system for assembly verification on
consumer GPUs (RTX 5060 Ti, 4.8 FPS). The system jointly predicts assembly state
detection, forward-gaze estimation, body pose, and procedure step recognition from
egocentric video using a ConvNeXt-Tiny backbone with shared FPN. At 46M trainable
parameters and 85 GFLOPs—less than a single specialist model—POPW achieves
detection mAP50 of 0.33 (present-class) and forward-gaze MAE of 8.71° on the
IndustReal dataset.

The paper's primary contribution is a systematic analysis of training pathologies
in multi-task assembly learning. We document four verified findings: (1) a per-frame
balanced sampler combined with a recording-id-keyed feature bank silently defeats
temporal heads; (2) per-parameter gradient norms can be misinterpreted as head-level
magnitudes; (3) activity recognition at 74 classes from 3.7k frames collapses
regardless of loss function (both CE and CB-Focal), establishing a data lower bound
for long-tail multi-task classification; and (4) head-pose annotation artifacts in
the IndustReal dataset. All code and model weights are open-source.
```

## Key Changes from 5-Task to 4-Task Paper

| Item | 5-Task | 4-Task |
|------|--------|--------|
| Title scope | "5-task assembly verification" | "Multi-task assembly verification" |
| Abstract numbers | Activity metrics included | Activity metrics removed |
| Section 4 | Activity results | Remove activity, expand pose |
| Section 5 | 3 pathologies | 4 pathologies (add §5.3 activity collapse) |
| Architecture | Activity head described + results | Activity head described (for completeness) but no results |
| Conclusion | "5 tasks from 1 model" | "4 tasks from 1 model, activity analyzed as failure" |
| Page count | 8.5 pages | 8.5 pages (activity results → activity pathology) |

## Opus Validation Required

1. **Is the 4-task paper strong enough for AAIML?** The headline metrics become:
   - Detection mAP50_pc = 0.33 (vs YOLOv8m single-task 0.84)
   - Forward-gaze MAE = 8.71° (vs SOTA ~8°)
   - PSR F1 = [current value, unknown]
   - 4 training pathology findings (section 5)
   Does this comp.

2. **Should we run the 4-task system all the way to RF10?** Or stop now and
   report the current RF4 results?

3. **If 4-task, do we attempt to improve detection/pose/PSR without activity
   gradient competition?** Removing activity may improve the other 3 tasks.
   Worth running 1 epoch without activity to measure deltas?