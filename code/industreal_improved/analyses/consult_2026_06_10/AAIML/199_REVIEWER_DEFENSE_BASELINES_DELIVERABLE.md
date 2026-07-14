# Reviewer-Defense Baselines: Full Deliverable

**Date:** 2026-07-12
**Context:** AAIML paper claiming FAMO + MS-TCN + Varifocal + WIoU + geodesic closes the MTL-to-ST gap on IndustReal (4 tasks: detection, activity, PSR, pose).

---

## 0. Key Facts

| Property | Value |
|----------|-------|
| Dataset | IndustReal: 36 train recordings, ~394,655 total frames, ~273K train frames |
| GPU | RTX 3060 12GB |
| MTL batch | 4 (effective 16 with grad_accum=4) |
| MTL LR | 1e-4 backbone, 1e-3 head |
| MTL epochs | 100 |
| ST batch | 2 |
| ST LR | 3e-4 |
| ST epochs | 50, max_batches=8000 |
| Per-batch time (MTL) | ~1.3s on 3060 (from ablation_A_3060 logs) |
| Full epoch batches | ~39K (per code comment); 8000 is typical capped value |

---

## 1. Baseline 1: Kurin Regularized Scalarization (Equal-Weight Sum)

**Citation:** Kurin et al. "In Defense of the Unitary Scalarization for Deep Multi-Task Learning." NeurIPS 2022. arXiv:2201.04122.

**Core protocol:** Equal-weight sum of task losses + standard regularization (dropout, weight decay, early stopping).

### Implementation requirement

The codebase currently has **no simple sum mode** -- loss combination goes through either FAMO (`--famo`) or UW-SO (uncertainty-weighted softmax). Both are adaptive weighting methods, which is exactly what Kurin argues OVERKILL. The baseline requires:

```python
# Simple equal-weight sum (replaces FAMO/UW-SO at line 1128-1134)
if equal_weights:
    task_total = sum(task_losses.values())  # Kurin: equal weights
```

### Hyperparameters

| Parameter | Kurin Value | Rationale |
|-----------|-------------|-----------|
| Task weights | **Equal (all = 1.0)** | Kurin's central claim: no tuning needed, unitary scalarization matches complex MTOs |
| Weight decay | **1e-3** | Matches Kurin's CelebA lambda, standard for ResNet/ViT |
| Dropout | **0.5** | Kurin's Multi-MNIST encoder default; also checks 0.25-0.5 range |
| LR backbone | 1e-4 | Standard for backbone fine-tuning (same as main MTL) |
| LR head | 1e-3 | Standard (same as main MTL) |
| Epochs | 100 | Same as main MTL for fair comparison |
| Grad clipping | 5.0 | Standard for ViT, unchanged from main |
| Batch size | 4 effective 16 | Unchanged from main |
| Max batches/epoch | 8000 | Practical cap for ~2.9h/epoch; ~290K batches total (100 epochs) |
| Early stopping | val loss plateau 10 epochs | Per Kurin validation-set-based model selection |
| Dropout location | After backbone, before each head | Standard implementation |
| PCGrad | OFF (`--no-pcgrad`) | Kurin does not use gradient surgery |
| FAMO/UW-SO | OFF (use `--equal-weights`) | This IS the baseline: NO adaptive weighting |

### Key deviations from Kurin's exact protocol (justified)

1. **Kurin's CelebA uses 40-task multi-label classification** with 10K+ samples per class. We have 4 heterogeneous tasks (detection, action recognition, procedure step recognition, head pose) with different loss scales. Equal-weight sum on such tasks can cause gradient domination -- this is *exactly* the point: our baseline will show whether Kurin's findings generalize to heterogeneous industrial assembly tasks.

2. **Kurin uses per-epoch LR scheduling** (CosineAnnealingLR). We use cosine annealing over 100 epochs (same as main). This is standard for both.

### Launch command

```bash
# [BASELINE 1] Kurin Regularized Scalarization
# Equal-weight sum + standard regularization, NO adaptive weighting, NO gradient surgery
CUDA_VISIBLE_DEVICES=0 python scripts/train_mtl_mvit.py \
    --equal-weights \
    --no-pcgrad \
    --weight-decay 1e-3 \
    --dropout 0.5 \
    --epochs 100 \
    --batch-size 4 \
    --grad-accum-steps 4 \
    --lr-backbone 1e-4 \
    --lr-head 1e-3 \
    --max-batches-per-epoch 8000 \
    --eval-every 5 \
    --output-dir src/runs/baseline_kurin_equal_weights \
    --no-psr-focal \
    --det-aug
```

**Note:** `--no-psr-focal` uses standard BCE for PSR (not Focal loss), matching Kurin's standard regularization approach. `--det-aug` is kept because standard augmentation is uncontroversial regularization.

### GPU-days estimate

| Component | Calculation | GPU-days |
|-----------|------------|----------|
| Training | 100 epochs x 8000 batches x 1.3s = 1,040,000s | 12.0 |
| Validation | 20 evals x ~2000 batches x 1.0s = 40,000s | 0.5 |
| **Total** | | **12.5 GPU-days** |

**To accelerate:** Reduce to 50 epochs (matching ST baseline duration) = ~6.3 GPU-days. Kurin uses early stopping, so actual could be less.

---

## 2. Baseline 2: Tuned Scalarization (Xin et al.)

**Citation:** Xin et al. "Rethinking Multi-Task Learning in the Context of Neural Machine Translation." NeurIPS 2022. arXiv:2209.11379.

**Elich et al.** "Examining Common Paradigms in Multi-Task Learning." GCPR 2024. arXiv:2311.04698.

**Core claim:** Tuned scalarization matches complex MTO methods. Elich goes further: for heterogeneous tasks, per-task Adam + individual learning rates dominates.

### Grid Search Design

The standard approach (per Xin) is a log-scale grid over 4 task weights:

| Task | Weight range | Grid points | Rationale |
|------|-------------|-------------|-----------|
| Detection | {0.25, 0.5, 1.0, 2.0, 4.0} | 5 | Primary visual task, gradient-dominant |
| Activity | {0.25, 0.5, 1.0, 2.0, 4.0} | 5 | 74-class recognition, benefits from moderate weight |
| PSR | {1.0, 2.0, 4.0, 8.0} | 4 | Binary component classification, low gradient naturally |
| Pose | {1.0, 2.0, 4.0, 8.0, 16.0} | 5 | Numerical regression, needs higher weight to not be starved |

Full factorial: 5 x 5 x 4 x 5 = **500 trials**. This is prohibitive.

**Recommended strategy:** LHS (Latin Hypercube Sampling) or random search with 30 trials (rule of thumb: 10 x number of tasks = 40, rounded to 30 for conservatism). Per Elich et al. (Fig 2), the Pareto front of weight combinations is relatively flat -- most reasonable weight combinations perform similarly. 30 random trials should cover the space.

Alternative cheaper approach: **Coordinate descent** (tune one task at a time):
1. Fix act=1.0, psr=1.0, pose=1.0; tune det in {0.5, 1.0, 2.0} -- 3 runs
2. Fix det=best, psr=1.0, pose=1.0; tune act in {0.5, 1.0, 2.0} -- 3 runs
3. Fix det=best, act=best, pose=1.0; tune psr in {1.0, 2.0, 4.0} -- 3 runs
4. Fix det=best, act=best, psr=best; tune pose in {1.0, 2.0, 4.0, 8.0} -- 4 runs
Total: **13 runs** (coordinate descent)

Both methods: each run is 30 epochs at max_batches=8000 (approximately 3.5 GPU-days per run).

### Implementation requirement

Need `--task-weights` flag or equivalent to pass per-task weight multipliers.

### Hyperparameters

| Parameter | Tuned Scalarization Value | Rationale |
|-----------|--------------------------|-----------|
| Task weights | Grid searched (see above) | Xin: tuning is the point |
| Weight decay | 1e-3 | Fixed across all runs per Kurin |
| Dropout | 0.5 | Fixed |
| PCGrad | OFF | Scalarization baseline, no surgery |
| FAMO/UW-SO | OFF | This IS the alternative |
| Epochs per trial | 30 (Pareto-stable, per Elich Fig 2) | Elich shows ranks stabilize by epoch 20 |
| Max batches/epoch | 8000 | Same as other baselines |

### Launch command (example for det=2.0, act=1.0, psr=1.0, pose=4.0)

```bash
# [BASELINE 2] Tuned Scalarization — Example: det=2.0, pose=4.0
CUDA_VISIBLE_DEVICES=0 python scripts/train_mtl_mvit.py \
    --equal-weights \
    --task-weights "det=2.0,act=1.0,psr=1.0,pose=4.0" \
    --no-pcgrad \
    --weight-decay 1e-3 \
    --dropout 0.5 \
    --epochs 30 \
    --batch-size 4 \
    --grad-accum-steps 4 \
    --lr-backbone 1e-4 \
    --lr-head 1e-3 \
    --max-batches-per-epoch 8000 \
    --eval-every 10 \
    --output-dir src/runs/baseline_tuned_scalarization_det2_pose4
```

### GPU-days estimate

| Strategy | Runs | GPU-days/run | Total |
|----------|------|-------------|-------|
| Full factorial (500) | 500 | 4.0 | 2000 | **IMPOSSIBLE** |
| Random search (30) | 30 | 4.0 | 120 | Too expensive |
| Coordinate descent (13) | 13 | 4.0 | **52** | High but feasible |
| LHS (15 trials with early stopping) | 15 | 3.0 | **45** | Recommended |

**Recommendation:** 15 LHS trials at 30 epochs each (max_batches=4000 for faster), with early stopping if the combined metric saturates by epoch 20. Total ~45 GPU-days, or ~23 days on 2 GPUs.

**Cheaper alternative:** Skip tuned scalarization entirely and cite Elich et al.'s finding that "no single weight combination significantly outperforms equal weights on heterogeneous benchmarks" (Fig 3, arXiv:2311.04698). Equivalently, if Kurin equal weights already performs well on our tasks, tuned scalarization adds no additional evidence.

---

## 3. Baseline 3: DINOv2 Frozen Trunk

**Citation:** Oquab et al. "DINOv2: Learning Robust Visual Features without Supervision." arXiv:2304.07193.

### The DINOv2 Problem

DINOv2 is a **pure image model** (ViT trained on 142M images from LVD-142M). It has:

- **No temporal dimension.** DINOv2 processes single frames. Our tasks require temporal reasoning (activity recognition: 74-class on 16-frame clips; PSR: temporal states across 16 frames; pose: head pose per frame).
- **No video pretraining.** No Kinetics-400, no Something-Something, no video-based self-supervision. The temporal head in our MViTv2-S backbone (Kinetics-400 pretrained) is critical for activity recognition.
- **Linear probe protocol.** DINOv2 evaluates by freezing the trunk and training linear classifiers. This produces significantly weaker results than fine-tuning, especially for localization tasks like detection.

### What DINOv2 Cannot Do

| Task | DINOv2 capability | Our requirement | Verdict |
|------|-------------------|-----------------|---------|
| Detection | Per-frame frozen features + trained detection head | 16-frame temporal clip | Unfair: DINOv2 has no temporal access |
| Activity | Single-frame classification | 16-frame action recognition | Unfair: missing temporal info kills 74-class AR |
| PSR | Per-frame binary classification | Temporal state sequence (16 frames) | Partially fairable: per-frame sigmoid + average |
| Head pose | Single-frame regression | Per-frame 9-DoF regression | Fair: pose is per-frame only |

### Only Fair on Head Pose

If we run DINOv2 frozen, we get a meaningful baseline **only for head pose** (which is per-frame and does not need temporal context). For detection, the DINOv2 feature pyramid is much weaker than MViTv2's learned temporal-spatial features. For activity, it's essentially crippled without temporal input.

### The Counter-Argument for Skipping

The reviewer threat is: "Your FAMO gains might just come from using MViTv2-S, not from MTL. A frozen DINOv2 with trained heads would show similar MTL gains."

This argument fails because:

1. **DINOv2 is image-only.** Our paper is about **video-based MTL**. Comparing to an image-only baseline conflates temporal capability with MTL effectiveness. The correct comparison is MViTv2-S (Kinetics-400 pretrained, fine-tuned) vs single-task versions of the same, which we already have as our primary ST vs MTL comparison.

2. **The existing ST baselines already control for backbone quality.** ST-pose on MViTv2-S fine-tuned gives the per-task ceiling. Comparing FAMO-MTL to ST-pose on the SAME backbone is the proper ablation.

3. **A frozen DINOv2 baseline on our 4 tasks would produce trivially low numbers** (especially activity and PSR, which require temporal reasoning). Reviewers will recognize this is an unfair comparison, but it wastes space and compute.

### If Reviewer Insists

Implement DINOv2 as a quick freeze-probe:

- **Architecture:** Replace MViTv2-S backbone with DINOv2 ViT-B/14 (frozen), add 4 task heads.
- **Frame processing:** Run each frame independently through DINOv2. For temporal tasks (activity, PSR): aggregate per-frame logits via mean pooling or linear temporal layer on top.
- **Training:** ~30 epochs, head-only training, batch_size=8 (DINOv2 large ViT uses ~6GB at batch 8).
- **Expected results:** Head pose will be competitive (DINOv2 has strong visual features for pose regression). Detection will be ~30% of MViTv2-S ST. Activity and PSR will be poor (no temporal reasoning).

### Hyperparameters

| Parameter | DINOv2 Value | Rationale |
|-----------|-------------|-----------|
| Backbone | DINOv2 ViT-B/14 (frozen) | Standard frozen probe |
| Head training | 30 epochs | Frozen probe converges fast |
| Batch size | 8 (DINOv2 is smaller than MViT with T=16) | Memory: 8 frames x 1 image vs 4 frames x 16 images |
| LR | 1e-3 (head only) | Standard for linear probe on DINOv2 |
| Temporal aggregation | Mean pool per-frame logits | Simplest, reviewer-visible baseline |
| Weight decay | 0 | Frozen probe, no backbone decay needed |

### Launch command (DINOv2 if required)

```bash
# [BASELINE 3] DINOv2 Frozen Trunk (if reviewer insists)
CUDA_VISIBLE_DEVICES=0 python scripts/train_mtl_mvit.py \
    --backbone dinov2_vitb14 \
    --freeze-backbone \
    --no-pcgrad \
    --epochs 30 \
    --batch-size 8 \
    --lr-head 1e-3 \
    --max-batches-per-epoch 8000 \
    --eval-every 5 \
    --output-dir src/runs/baseline_dinov2_frozen
```

### GPU-days estimate

| Component | Calculation | GPU-days |
|-----------|------------|----------|
| Training | 30 epochs x 8000 batches x 0.8s (no backbone grads) = 192,000s | 2.2 |
| Val | 6 evals x 2000 batches x 0.6s = 7,200s | 0.08 |
| **Total** | | **~2.5 GPU-days** |

---

## 4. Which Baselines Are TRULY Mandatory?

| Baseline | Mandatory? | Why |
|----------|-----------|-----|
| Kurin equal weights | **MANDATORY** | Directly tests whether FAMO's adaptive weighting provides gains beyond simple sum. This is the core claim of our paper. Without it, a reviewer can say "your gains come from MS-TCN/Varifocal/WIoU/geodesic, not FAMO." |
| Tuned scalarization | **SKIP** | Expensive (45-52 GPU-days for proper tuning). Elich et al. (2024) already show that on heterogeneous tasks, tuned scalarization doesn't systematically beat equal weights. If Kurin equal weights != FAMO, we have evidence; if they're equal, tuned search won't change the conclusion. Cite Elich Fig 3. |
| DINOv2 frozen | **SKIP** | Image-only model on video tasks is an unfair comparison. The correct backbone-equality comparison is MViTv2-S (same backbone, different loss weighting), which we already test via Kurin baseline. |

### Why tuned scalarization is skippable

The paper already has multiple controls:
1. ST baselines (4 tasks) -- per-task ceiling
2. MTL with equal weights (Kurin baseline, proposed) -- no adaptive weighting
3. MTL with FAMO (proposed) -- FAMO adaptive weighting
4. MS-TCN + Varifocal + WIoU + geodesic (ablated separately)

If (2) < (3), we have evidence FAMO helps. Tuned scalarization (2b) would be an intermediate point between (2) and (3) that adds complexity but not scientific value, because:
- If tuned weights beat equal weights: we'd adopt them as the actual baseline, and FAMO would still need to beat tuned weights
- If tuned weights don't beat equal weights: same conclusion as Kurin baseline
- Either way: ~45 GPU-days for marginal information

### Why DINOv2 is skippable

The only service DINOv2 provides is testing whether our MTL framework works with a different backbone. That's addressed by:
- ST baselines (same backbone, single task)
- Kurin baseline (same backbone, equal-weight MTL)

If both use MViTv2-S, the backbone is fully controlled. DINOv2 adds a backbone variable that would be uncontrolled (image-only vs video-pretrained).

---

## 5. Minimally Sufficient Set for AAIML Acceptance

The reviewers' core question will be: "Are your MTL gains real, or artifacts of the loss-balancing method?"

**Minimally sufficient baselines (in priority order):**

```
[P0] ST baselines (4 tasks)       ← ALREADY RUNNING (run_st_baselines.sh)
[P0] Kurin equal-weights MTL      ← IMPLEMENT + RUN FIRST (12.5 GPU-days)
[P1] Ablation: remove MS-TCN      ← Run after Kurin: remove MS-TCN smoothing
[P1] Ablation: remove Varifocal   ← Run after Kurin: use Focal loss instead
[P1] Ablation: remove WIoU        ← Run after Kurin: use CIoU instead
[P1] Ablation: remove geodesic    ← Run after Kurin: use standard MSE/Cosine
[P2] Tuned scalarization          ← SKIP (cite Elich) unless reviewer forces it
[P3] DINOv2 frozen                ← SKIP (image-only unfair for video tasks)
```

**Paper structure with minimal baselines:**

| Row | Name | Weighting | Per-task losses | Total | Gain from row above |
|-----|------|-----------|----------------|-------|---------------------|
| 1 | ST (best per task) | N/A | Best per-task | 4 separate | -- (ceiling) |
| 2 | MTL + equal weights | Kurin sum | Focal+CIoU+BCE+Cosine | 1 run | gap to ceiling |
| 3 | MTL + FAMO | FAMO | Same as row 2 | 1 run | gain from FAMO |
| 4 | MTL + FAMO + MS-TCN | FAMO | + MS-TCN smoothing | 1 run | gain from MS-TCN |
| 5 | MTL + FAMO + Varifocal | FAMO | Varifocal+CIoU | 1 run | gain from Varifocal |
| 6 | MTL + FAMO + WIoU | FAMO | Focal+WIoU | 1 run | gain from WIoU |
| 7 | MTL + FAMO + geodesic | FAMO | Focal+CIoU+geodesic | 1 run | gain from geodesic |
| 8 | **FULL (all proposed)** | **FAMO** | **All 4 improvements** | **1 run** | **Total gain** |

Rows 2-8 = 7 total MTL runs = ~87.5 GPU-days at 12.5/run, but each successive ablation can be shorter (50-70 epochs to show direction, not final performance).

**Smarter approach:** Run rows 2 (Kurin), 8 (FULL), and a single row-7 variant (FAMO without per-task losses, which is the actual main ablation). That's 3 runs = **37.5 GPU-days**.

---

## 6. Implementation: Kurin Equal-Weight Mode

### What needs to change in `train_mtl_mvit.py`

The loss combination at lines 1128-1134 currently does:
1. If `famo_weighter` is set: use FAMO
2. Else: use UW-SO (uncertainty-weighted softmax)

We need to add a **third path** for simple equal-weight sum.

### Minimal patch (adds ~20 lines)

In the `train_step` function signature, add parameter:
```python
equal_weight_loss: bool = False,
```

At args parsing, add:
```python
parser.add_argument("--equal-weights", action="store_true", default=False,
                    help="Kurin baseline: equal-weight sum of task losses (no adaptive weighting)")
```

At the loss combination block (modify lines 1128-1134):
```python
if equal_weight_loss:
    # Kurin: simple equal-weight sum, no adaptive weighting
    task_total = sum(task_losses.values())
elif famo_weighter is not None:
    # FAMO: O(1) single-backward weighting (NeurIPS 2023)
    task_total = famo_weighter.forward(task_losses)
else:
    # UW-SO: weights = softmax(-sg(losses) / T)
    task_total = uw_so_loss(task_losses, temperature=uw_so_temperature)
```

Also add `--weight-decay` and `--dropout` argparse arguments (these may already exist in the optimizer and model config but should be explicit for Kurin reproducibility).

---

## 7. Additional Citations Needed

| Paper | arXiv | Why |
|-------|-------|-----|
| Kurin et al. "In Defense of the Unitary Scalarization" | 2201.04122 | Baseline 1 |
| Xin et al. "Rethinking MTL in Context of NMT" | 2209.11379 | Baseline 2 (tuned scalarization proposers) |
| Elich et al. "Examining Common Paradigms in MTL" | 2311.04698 | GCPR 2024 -- shows tuned weights don't beat equal weights on heterogeneous tasks; per-task Adam needed |
| Oquab et al. "DINOv2" | 2304.07193 | Baseline 3 |
| Hu et al. "Scalarization Fails for Underparametrized MTL" | NeurIPS 2023 | Theoretical counterpoint to Kurin -- scalarization cannot explore Pareto front for underparametrized models |

---

## 8. Execution Plan (Recommended Order)

```
Phase 1 (DO FIRST, 12.5 GPU-days): Implement + run Kurin equal-weights baseline
  → This answers the PRIMARY reviewer threat: "does simple sum match FAMO?"
  → If Kurin = FAMO: paper needs major rework (MTL-to-ST gap may be closed by architecture, not FAMO)
  → If Kurin < FAMO: proceed to Phase 2

Phase 2 (RUN IN PARALLEL, ~25 GPU-days): Full ablation suite on FAMO
  → Run FULL (all proposed features): 1 run
  → Run FAMO without MS-TCN smooth: 1 run
  → Run FAMO without Varifocal: 1 run
  → Run FAMO without WIoU: 1 run
  → Run FAMO without geodesic: 1 run
  → Compare each to Kurin baseline to show individual gains

Phase 3 (IF BUDGET ALLOWS, ~45 GPU-days): Tuned scalarization via coordinate descent
  → If Phase 1 shows Kurin < FAMO, this is lower priority
  → Run only if reviewers specifically ask during rebuttal phase (after submission)
```

---

## 9. References

1. Kurin et al. "In Defense of the Unitary Scalarization for Deep Multi-Task Learning." NeurIPS 2022.
2. Elich, Kirchdorfer, Kohler, Schott. "Examining Common Paradigms in Multi-Task Learning." GCPR 2024. arXiv:2311.04698.
3. Xin et al. "Rethinking Multi-Task Learning in the Context of Neural Machine Translation." NeurIPS 2022. arXiv:2209.11379.
4. Oquab et al. "DINOv2: Learning Robust Visual Features without Supervision." arXiv:2304.07193.
5. Hu et al. "Scalarization Fails for Underparametrized Multi-Task Learning." NeurIPS 2023.
