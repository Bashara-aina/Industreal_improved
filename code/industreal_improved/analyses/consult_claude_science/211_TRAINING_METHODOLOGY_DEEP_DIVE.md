# Training Methodology Deep Dive

**Doc 211** — Everything about how we train, why we chose what we chose, and
what could be better.

---

## 1. Current Training Setup

### 1.1 Batch Size and Gradient Accumulation

The effective batch size is the product of two knobs:

| Parameter | Value | Purpose |
|---|---|---|
| `BATCH_SIZE` | 6 (config default) / 4 (CLI default) | Per-micro-batch |
| `GRAD_ACCUM_STEPS` | 8 (config) / 4 (CLI) | Micro-batches per optimizer step |
| **Effective batch** | 16 (config) / 16 (CLI) | `BATCH_SIZE x GRAD_ACCUM_STEPS` |

The config default (`BATCH_SIZE=6, GRAD_ACCUM_STEPS=8`) targets RTX 5060 Ti
16GB VRAM. The per-micro-batch of 6 was calibrated by a VRAM probe: batch=2
used ~2 GB, so 6x gives ~6-8 GB peak with `T=16` sequence batches staying
under 12 GB out of 16 GB. The `train_mtl_mvit.py` CLI defaults to
`--batch-size 4 --grad-accum-steps 4`, also yielding effective 16.

**Critical detail [OPUS 186 S5.1]**: The loss is divided by `grad_accum_steps`
before `backward()`, so accumulation produces a MEAN of micro-batch gradients
rather than a SUM. Without this fix, the boundary step sees `grad_accum_steps`
times the intended magnitude, causing the gradient clip to fire on most of the
gradient.

### 1.2 Optimizer

Three separate AdamW param groups:

| Group | Parameters | LR | Weight Decay |
|---|---|---|---|
| Backbone | `feature_pyramid.backbone.*` | `1e-4` | `0.05` |
| Heads | everything except backbone + log_vars | `1e-3` | `0.05` |
| Log vars | `log_vars.*` | `1e-3` | `0` |

Three-group design is standard for MTL: backbone gets lower LR (transfers
general features slowly), heads get 10x higher LR (task-specific learning
needs faster adaptation), log vars get zero weight decay (they must stay free
to represent any precision).

**Weight decay of 0.05 for backbone and heads** was audited [2026-07-01 agent
audit] and lowered from a previous 0.05 (which was already correct — the
earlier 5e-2 is the same as 0.05). The original paper value of 5e-2 was
deemed appropriate for single-task but high for MTL; after auditing, 0.05 was
retained for backbone/heads but zeroed for log_vars.

### 1.3 LR Scheduler

CosineAnnealingLR with `T_max = epochs` (default 100):

```
lr(t) = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(t / T_max * pi))
```

The config also defines a `USE_COSINE_ANNEALING = False` flag alongside a
commented reference to OneCycleLR from the paper, but the actual code in
`train_mtl_mvit.py` uses hard-coded `CosineAnnealingLR`. This is a
documentation drift: the paper described OneCycleLR, the implementation
uses cosine. The difference matters:

- **Cosine**: smooth decay from epoch 0, full LR for ~first 10% of training
- **OneCycle**: warm-up phase (LR increases from near-zero to max), then
  cosine decay; better for very large batches and avoids early-training
  instability

The warm-up is instead handled by a `WARMUP_EPOCHS = 2` config value, but
the actual CosineAnnealingLR receives `T_max = args.epochs` with no warm-up
period. The true warm-up comes from the EMA loss tracker (2.3 below) and the
log_var initialization (2.2 below), which effectively cap early training
aggressiveness.

### 1.4 Precision

Current setup: `MIXED_PRECISION = True` with `AMP_DTYPE = 'bf16'`.

The bf16 path uses `torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)`
for the forward pass and a `GradScaler` for backward. Critically, with bf16
the GradScaler is a **no-op**: bf16 has the same exponent range as fp32 (8
bits), so PSR loss spikes that corrupted the previous fp16 GradScaler are
fully representable. This makes bf16 the safe default that was unavailable on
older GPUs (pre-Ampere).

The fp16 path is legacy: the PSR sequential BCE loss produces infrequent
large-magnitude gradients (>1000) that overflow fp16's 5-bit exponent.
GradScaler compensated but introduced `inf`/`nan` cascade failures over
1100+ steps. bf16 eliminates this failure mode.

### 1.5 Gradient Clipping

Global clip at `GRAD_CLIP_NORM = 5.0` (raised from 1.0 after agent audit).

At the original 1.0, a 4-head MTL model with backbone gradients from all
tasks combined to ~3-5x the norm of any single head, so the clip fired on
~80-90% of steps. The auditor found that this effectively starved the
backbone of gradient signal. 5.0 is the standard ViT multi-task value —
still safe against true explosion (loss > 10^6) but permissive enough to
let all heads contribute.

---

## 2. Kendall Uncertainty Weighting

### 2.1 The Algorithm

Kendall et al. (CVPR 2018) formulates multi-task loss as a homoscedastic
uncertainty-weighted combination:

```
total_loss = sum_t [ exp(-s_t) * L_t + s_t/2 ]
```

where `s_t = log(var_t)` is a learnable parameter per task. The `exp(-s_t)`
term is the **precision** (inverse variance): tasks with high uncertainty get
low weight, tasks the model is confident about get high weight. The `s_t/2`
regularisation term prevents precision from going to infinity.

In this codebase:

```
lv = log_vars[name]           # learned log-variance
prec = torch.exp(-lv)         # precision (weight)
weighted = prec * loss_k[name] + lv / 2
```

### 2.2 Precision Capping (The Critical Fix)

Without caps, Kendall naturally **starves the highest-loss task**. Consider
activity (CE loss ~12) and detection (CIoU + Focal loss ~2-5). The Kendall
equilibrium sets `exp(-s) ∝ 1/(2*L)`, so activity gets weight ~0.04 vs
detection ~0.10-0.25. After a few epochs, activity collapses to 3/75 classes.

**LV_CLAMP_MAX** per task [OPUS 207 S1b FIX]:

| Task | Max log_var | Min precision (exp(-lv)) | Rationale |
|---|---|---|---|
| det | 1.5 | 0.22 | Detection loss is moderate; floor 0.22 prevents full starvation |
| act | 1.0 | 0.37 | Activity is highest-loss; floor 0.37 guarantees backbone attention |
| psr | 0.5 | 0.61 | PSR is low-signal; floor 0.61 prevents backbone drift |
| pose | 2.0 | 0.14 | Pose loss is tiny (~0.01); wide ceiling acceptable |
| LV_CLAMP_MIN | -4.0 | 54.6 | All tasks share this floor; prevents exp(-lv) explosion |

The caps mean every task contributes at least its floor-level gradient to the
shared backbone, breaking the Kendall starvation cascade.

**Pose precision cap** (separate mechanism): `lv_pose = max(lv_pose, lv_det)`.
This ensures pose precision never exceeds detection precision, preventing pose
(~0.01 loss, naturally low variance) from dominating the backbone.

### 2.3 EMA Normalization

Raw task losses operate at very different scales (activity ~12, pose ~0.01).
Without normalization, Kendall's `prec * L` product for activity is ~5 and for
pose is ~0.001 — pose is invisible. Normalization fixes this.

The EMA tracker maintains a running mean of each raw task loss:

```
ema_losses[name].mul_(ema_momentum).add_(loss, alpha=1.0 - ema_momentum)
```

After warmup (threshold: `ema_v > 1e-3`), each task loss is divided by its
EMA before entering the Kendall term:

```
loss_for_kendall[name] = loss[name] / (ema_losses[name] + 1e-6)
```

Now every task's normalized loss is ~O(1), so Kendall balances on the
**relative uncertainty** rather than absolute magnitude. During warmup (first
~100 steps), EMA values are initialized to 1.0, so `loss / 1.0 = loss` — the
raw scale is used temporarily until the EMA converges.

**Important**: The EMA values are detached from autograd. They track the mean
for normalization only, not for gradient computation.

### 2.4 What Happens in Practice

Tracking log_vars reveals the equilibrium:

- **det**: log_var drifts toward +1.5 (the cap). Detection is intrinsically
  harder (CIoU + focal + DFL on sparse GT), so Kendall wants to suppress it.
  The cap at 1.5 (precision floor 0.22) prevents full suppression.
- **act**: log_var settles near +1.0 (cap). Activity CE loss is high, but
  the cap ensures minimum weight 0.37.
- **psr**: log_var near +0.5. PSR has the tightest cap (0.5, precision floor
  0.61), reflecting that PSR's gradient signal is weakest and needs enforced
  contribution.
- **pose**: log_var drifts between +0.5 and +2.0. Pose naturally gets low
  weight due to tiny loss magnitude; the cap at 2.0 allows this but the pose-
  below-det constraint prevents pose from stealing the backbone.

### 2.5 The Uncapped Ablation

OPUS 201 tests the "before" state: `--kendall-uncapped` sets all caps to 4.0
(allowing precision ~0.018, near-zero weight). In this regime:

1. Activity (highest raw loss) gets precision ~0.02 → effective weight ~0.24
2. Pose (lowest raw loss, ~0.01) gets precision ~0.8 → effective weight ~0.008
3. Detection gets precision ~0.1 → effective weight ~0.5
4. Activity collapses to ~3 classes by epoch 5

This replicates the canonical Kendall-collapse failure mode and validates the
cap design.

---

## 3. PCGrad Gradient Surgery

### 3.1 How It Works

PCGrad (Yu et al., NeurIPS 2020) projects conflicting gradients away from
each other before applying them to the shared backbone. The core step:

```
For each task pair (i, j) where dot(g_i, g_j) < 0:
    g_i = g_i - (dot(g_i, g_j) / ||g_j||^2) * g_j
```

This removes the component of g_i that conflicts with g_j. After processing
all pairs in a random order, the deconflicted gradients are summed and
applied to the backbone.

The codebase processes only the shared backbone parameters (not head-specific
params) with PCGrad. This is correct: head gradients are private and don't
need deconfliction.

### 3.2 When It Helps

PCGrad helps when there is **substantial inter-task gradient conflict**:
tasks whose gradient directions are often >90 degrees apart. In this codebase,
the conflict pattern is:

- **det vs act**: High conflict. Detection wants features at high resolution
  (P3/P4) that localize object boundaries; activity wants discriminative
  features that separate action classes. These are fundamentally different
  visual patterns.
- **det vs pose**: Moderate conflict. Both are geometric, but detection cares
  about bounding boxes (object extent) while pose cares about head angle.
- **psr vs everything**: Low conflict but also low signal. PSR gradients are
  small; PCGrad rarely triggers because the dot products are near-zero.
- **act vs pose**: Low conflict. Activity is semantic/appearance, pose is
  geometric — different feature subspaces.

### 3.3 Implementation Detail: PCGrad on Log-Var-Scaled Gradients

The PCGrad branch in `train_step()` computes per-task gradients w.r.t. the
shared backbone using the **same** Kendall-scaled losses as the forward pass:

```
weighted_loss = prec * losses_k_safe[name]
g = torch.autograd.grad(weighted_loss, shared_params, retain_graph=True)
```

This is important: if PCGrad used raw task losses instead, the gradient
magnitudes would be at wildly different scales, and the dot-product conflict
detection would be dominated by the highest-magnitude task (activity).
Using Kendall-scaled losses means all tasks contribute comparable gradient
magnitudes to the conflict detection.

### 3.4 What PCGrad Does NOT Fix

PCGrad addresses gradient **direction** conflict, not gradient **magnitude**
conflict. If one task's gradient is 100x larger than another's after Kendall
scaling (e.g., activity vs pose even after EMA normalization), PCGrad's
projection step still leaves the larger gradient dominating. The Kendall
precision capping (Section 2.2) handles magnitude balance; PCGrad handles
direction conflict.

PCGrad also cannot fix the problem of a task having **zero gradient** (dead
head). If PSR's gradient path is severed by an in-place operation, PCGrad
has nothing to project.

### 3.5 Alternatives

| Method | Description | Pros | Cons |
|---|---|---|---|
| **PCGrad** (current) | Project conflicting grads away | Simple, cheap, well-understood | Random order dependence, doesn't handle magnitude |
| **CAGrad** | Conflicting-gradient avoidance | Finds optimal trade-off on Pareto front | Hyperparameter (c=0.5 default), 2x compute |
| **GradDrop** | Stochastic mask on conflicting grads | Smooth, no projection | Hyperparameter (T), hard to debug |
| **RotoGrad** | Rotate feature representations per task | Addresses conflict at feature level | Complex, many params, unstable |
| **Nash-MTL** | Nash bargaining solution | Theoretically principled | Expensive, iterative inner loop |
| **IMTL** | Gradient normalization + angle | Simple, no random order | Can be unstable with >3 tasks |
| **MGDA** | Multi-gradient descent algorithm | Pareto optimality guaranteed | Computes weighting, expensive |

For this codebase, CAGrad is the most natural upgrade: it replaces the
random-order sequential projection with a principled convex combination that
guarantees Pareto improvement. The compute cost (~2x backward passes) is
acceptable at 4-head scale. Nash-MTL would be overkill — the iterative inner
loop costs ~10x per step.

---

## 4. Loss Functions Per Task

### 4.1 Detection (24 classes)

Three-component loss with Task-Aligned Learning (TAL) assignment:

| Component | Loss | Weight |
|---|---|---|
| Classification | Asymmetric Focal BCE | `1.0` |
| Box regression | CIoU loss | `GIOU_WEIGHT = 2.0` |
| Distribution | DFL loss (reg_max=16) | `1.0` |

Problems:
- **TAL topk=10** per FPN level. For small objects at P3 (stride 8, 28x28
  grid), 10 positive cells can be a significant fraction of the grid. This
  creates redundant positives (multiple cells predicting the same box). The
  CIoU loss then has to differentiate near-identical predictions. Reducing
  `tal_topk` to 5-7 may sharpen predictions.
- **Asymmetric focal gamma**: `gamma_pos=0.0, gamma_neg=2.0`. This keeps
  positive gradients alive (well-classified positives still contribute) while
  aggressively down-weighting easy negatives. The 0.0 positive gamma is the
  extreme end of the spectrum — even `gamma_pos=0.5` would provide some
  focusing on positives while retaining most of the benefit.
- **OHEM ratio 2:1**: keeps 2x hardest negatives per positive. This is
  quite aggressive and may be filtering out moderate negatives that would
  provide useful signal. Standard RetinaNet uses no OHEM (focal loss handles
  it). The OHEM was introduced during the collapse-fix era and may now be
  unnecessary.

Alternatives from literature:
- **Varifocal loss**: replaces asymmetric focal with a learned
  IoU-aware classification signal. Directly predicts whether a box will
  survive NMS. +1-2 AP on COCO.
- **GFL (Generalized Focal Loss)**: unifies classification quality
  (IoU score) with box distribution. Eliminates the separate IoU branch.
- **TOOD (Task-aligned One-stage Object Detection)**: adds a TAP (task-aligned
  predictor) that aligns classification and regression heads. Already
  partially implemented via TAL assignment.

### 4.2 Activity (75 classes)

Cross-entropy with:
- **Label smoothing**: 0.05 (lowered from 0.1 to reduce CE floor on 75 classes)
- **Class weights**: sqrt-tamed inverse frequency (max/min ratio ~12)
- **Logit adjustment**: optional per-class prior log-freq (Menon et al. 2020)
- **Ignore index**: -1 (unlabeled frames)

Problems:
- **75 classes with ~3.7K frames** means ~50 frames/class on average, with
  a long tail of classes having 1-7 frames. The sqrt-taming compresses the
  weight range but cannot create signal where none exists.
- **CE + label smoothing** with 75 classes and severe imbalance is suboptimal.
  The loss is dominated by the majority classes even after re-weighting,
  because CE's gradient magnitude depends on class count.
- **Logit adjustment** helps decision boundaries but doesn't change loss
  dynamics — it shifts the logits but the gradient magnitude from majority
  classes still dominates.

Alternatives from literature:
- **Class-balanced focal loss** (Cui et al. 2019): effective number weighting
  + focal loss. The codebase has `USE_CB_FOCAL_ACT` but defaults to False.
  CB-Focal directly addresses the class imbalance at the gradient level.
- **LDAM loss** (Cao et al. 2019): label-distribution-aware margin loss that
  explicitly pushes minority-class decision boundaries away from the majority.
  The codebase has LDAM-DRW compatibility but uses CE.
- **CRT (Classifier Re-training)**: train with instance-balanced sampling,
  then freeze backbone and re-train classifier with class-balanced sampling
  (Kang et al. ICLR 2020). The `--act-decoupled` flag implements this but
  is off by default.

### 4.3 PSR (11 binary components)

Focal Binary Cross-Entropy with:
- **Transition-aware weighting**: `transition_boost=3.0` on frames near 0→1
  transitions and their immediate neighbors.
- **Component weights**: inverse prevalence per PSR component.
- **Focal gamma=2.0, alpha=0.25** (standard Focal-BCE).
- **Temporal smoothing**: `PSR_TEMPORAL_SMOOTH_WEIGHT=0.05` for transition
  adjacency.

Problems:
- **Event F1 ~0.006** with loss ~0.17-0.27 indicates focal-collapse to
  negatives: the model predicts all zeros because 99.5%+ of frames have no
  transition, and focal loss on all-zeros is low. The transition-aware
  weighting boosts event frames but the baseline event rate is so low that
  even boosted frames are a tiny fraction of the total.
- **11 components at T=8 frames**: the temporal resolution after the PSR head
  downsamples T=16 to T=8 means transitions can only be detected at 8-frame
  granularity (~0.27s at 30fps). Many transitions are sub-frame.
- **Component independence**: the focal BCE treats each of the 11 components
  independently, but many are correlated (e.g., pick-and-place components).
  A structured prediction loss (CRF, GNN) could capture inter-component
  dependencies.

Alternatives from literature:
- **Asymmetric loss (ASL)**: removes the gradient contribution from easy
  negatives entirely (hard threshold instead of soft focal weighting). Would
  directly target the focal-collapse-to-negatives failure mode.
- **Set prediction loss**: treat the 11 components as a multi-label set and
  use Hungarian matching to make a set-level prediction. Overkill for 11
  components but would naturally handle correlation.
- **Temporal CRF as loss**: a conditional random field layer that penalises
  transition patterns inconsistent with the known component dynamics.
  Computationally heavy but principled.

### 4.4 Pose (6-DoF)

Combined cosine + geodesic loss on renormalized 6D vectors:

```
cosine_loss = (1 - cos(fwd_pred, fwd_gt)) + (1 - cos(up_pred, up_gt))
geodesic_loss = geodesic_angle(R_pred, R_gt)
pose_loss = cosine_loss + geodesic_loss
```

The Gram-Schmidt process converts 6D predictions to 3D rotations; the
geodesic loss measures the angular error on SO(3).

Problems:
- **Geodesic loss is symmetric but expensive**. For single-batch training on
  small heads it's fine, but it computes SVD (via `gram_schmidt_rotation`)
  which has non-trivial backward pass overhead.
- **No per-component weighting**: fwd (gaze direction) and up (head tilt)
  have equal weight, but gaze is generally more important for attention
  estimation in the assembly context.
- **Loss magnitude is ~0.01**: after 6D normalization and cosine similarity
  in [-1, 1], the loss is naturally small. The `HEAD_POSE_POS_SCALE=100.0`
  adjustment for position coordinates helps, but the angular component
  operates at a fundamentally different scale from detection or activity.

Alternatives:
- **6D rotation representation**: the current `pred_6d` + Gram-Schmidt is
  already the best continuous representation (Zhou et al. CVPR 2019).
- **Direct geodesic on quaternions**: skip the 6D representation and
  predict unit quaternions with geodesic loss directly. Works but quaternion
  discontinuity at the identity is problematic for gradient methods.

---

## 5. Learning Rate Strategies

### 5.1 Current Setup

CosineAnnealingLR with no warm-up, 100 epochs total. Three parameter groups
with separate base LRs:

- Backbone: 1e-4 (transfers pre-trained features, low LR prevents forgetting)
- Heads: 1e-3 (10x backbone — task-specific features need faster adaptation)
- Log vars: 1e-3 (separate group with weight_decay=0)

### 5.2 Problems

- **No warm-up**: CosineAnnealingLR starts at full LR from epoch 0. For a
  multi-task model with randomly initialized heads (detection TAL head, PSR
  head, pose head), the initial random loss can be 3-5x the post-warmup
  loss, causing the first few steps to have disproportionally large gradient
  updates. The log_var initialization (`-0.5`, giving precision ~1.65)
  partially mitigates this, but a formal linear warm-up of 2-5 epochs would
  be safer.
- **Single LR for all heads**: Activity (75-class CE) and detection (CIoU +
  focal BCE) have very different loss landscapes. A head-specific LR would
  allow activity (high loss, complex decision boundary) to learn faster while
  pose (simple loss, low magnitude) stays stable.

### 5.3 Per-Task Learning Rates

A natural extension: assign each head its own LR:

| Head | Suggested LR | Rationale |
|---|---|---|
| Backbone | 5e-5 | Lower than current; shared params need stability |
| Detection | 1e-3 | Current head LR; works well |
| Activity | 2e-3 | 2x standard — 75-class CE needs aggressive learning |
| PSR | 1e-3 | Same as current |
| Pose | 5e-4 | Lower — pose converges quickly and overfitting risk |
| Log vars | 1e-3 | Current; works well |

Implementation: add separate parameter groups for each head. The codebase
already has two groups (backbone + everything else); extending to 6 groups
(backbone + 4 heads + log_vars) is straightforward.

---

## 6. Gradient Accumulation and Effective Batch Size

### 6.1 Current Design

| Effective batch | Per-micro-batch | Accumulation steps | Per-iteration |
|---|---|---|---|
| 16 | 4 | 4 | 1 optimizer step = 16 frames |
| 16 | 6 | 8 | Same effective, different VRAM profile |

The accumulation boundary logic:

```
is_accum_boundary = ((batch_idx + 1) % grad_accum_steps == 0) or \
                    (batch_idx + 1 == len(train_loader))
```

zero_grad happens AFTER the optimizer step (not before), so gradients survive
across the accumulation window.

### 6.2 Batch Size Trade-offs

Smaller batch sizes have well-known effects:
- **Noisier gradients**: 16-frame effective batch is very small by modern
  standards (typical ViT training uses 1024+). The noise can help escape
  sharp minima but also slows convergence.
- **Batch normalization**: the codebase uses LayerNorm (MViTv2), not
  BatchNorm, so batch size doesn't affect normalization statistics.
- **TAL assignment**: TAL assigns positives independently per image, not
  per batch, so batch size doesn't affect positive coverage.

Increasing effective batch to 32-64 would:
- Reduce gradient variance, potentially stabilizing activity head training
- Require more VRAM or more accumulation steps
- Not help with the fundamental data sparsity issue

### 6.3 Accumulation as Implicit Regularization

Gradient accumulation with mean-averaging (the current design) is equivalent
to training with batch_size = effective_batch, assuming the DataLoader's
sampling is purely random. Since the codebase uses a WeightedRandomSampler,
accumulation smooths the per-batch class distribution — each micro-batch has
slightly different class composition, and accumulation averages these.

---

## 7. Mixed Precision: bf16 vs fp16

### 7.1 Current State

bf16 with GradScaler as a pass-through. The key insight is that bf16 has the
same exponent range as fp32 (8 bits), so loss spikes of any magnitude are
representable. The mantissa is truncated (7 bits vs 23 bits for fp32), but
empirically this does not affect convergence for this model.

### 7.2 Stability Issues (Historical)

The fp16 path failed because:
1. PSR focal BCE produces small gradients most steps but occasional large
   gradients on transition frames (loss ~5-10 vs usual ~0.2).
2. fp16's 5-bit exponent overflows at ~65,504 — a gradient of 10^4 overflows
   to inf.
3. GradScaler detects inf, but by the time it downscales, the optimizer has
   stepped with inf in some parameter groups, corrupting momentum buffers.
4. Once momentum is corrupted, it takes hundreds of steps to recover —
   typically the model never recovers and diverges.

bf16 eliminates steps 2-4 above. The trade-off is ~5-10% throughput loss vs
fp16 (larger data movement) vs ~2x gain over pure fp32.

### 7.3 Recommended Path

Stay on bf16. The only reason to consider fp16 is if inference hardware
requires it (T4, V100 do not support bf16 natively). For RTX 3060/5060 Ti,
bf16 is native and preferred.

---

## 8. Checkpoint Strategies

### 8.1 Current: Best-on-Val + Periodic + Latest

| Checkpoint type | Frequency | Contents | Used for |
|---|---|---|---|
| `best.pt` | When act_top1 improves | Full state + optimizer + EMA | Model selection |
| `epoch_NNNN.pt` | Every 10 epochs | Full state | SWA averaging |
| `latest.pt` | Every 10 epochs | Full state | Resume |
| `swa_averaged.pt` | End of training | SWA-averaged weights | Final eval |

Best model selection is based on **activity top-1 accuracy** on the val split.
This is a deliberate choice: activity has the highest combined metric weight
(0.35 in the paper's composite score) and is the hardest task, so val top-1
is a good proxy for overall model quality.

### 8.2 EMA (Exponential Moving Average)

Running EMA of model weights with `momentum = 0.999`:

```
ema_model_state[k] = momentum * ema_model_state[k]
                     + (1 - momentum) * model_state_dict[k]
```

The EMA state is updated only on **boundary steps** (after optimizer step),
so it tracks post-update weights rather than mid-accumulation weights.

At evaluation, EMA weights are swapped in:

```
raw_state = model.state_dict()
model.load_state_dict(ema_swap)
eval_metrics = evaluate(model, ...)
model.load_state_dict(raw_state)
```

This gives +1-2% across all metrics reliably, per Yakovlev et al. and
confirmed empirically.

### 8.3 SWA (Stochastic Weight Averaging)

Post-training SWA averages the last N periodic checkpoints:

```
avg_sd[k] = (1/N) * sum(checkpoints[k])
```

Default `--swa-checkpoints 5`. The averaging happens after training
completes, using saved `epoch_NNNN.pt` files. SWA and EMA are complementary:

- EMA: online, smooths training trajectory, used for val model selection
- SWA: offline, averages endpoint solutions, used for final test eval

### 8.4 Missing: Model Soup (Post-hoc)

Model soup (Wortsman et al. 2022) is used **pre-training** (backbone
initialization) but not **post-training** (averaging multiple MTL runs).
The `auto-soup init` mechanism averages ST specialist backbones to initialize
the MTL shared backbone, which is a different technique.

A proper post-hoc model soup would:
1. Run MTL training 3-5 times with different seeds
2. Average the best checkpoint from each run
3. Evaluate the averaged model

This typically gives +0.5-1% across tasks but requires 3-5x the compute
budget for training.

### 8.5 Best-of-N Evaluation

Not currently implemented. The idea: run validation evaluation with different
test-time augmentations (horizontal flips, multi-scale) and take the best
prediction per sample. For activity (75-class CE), flipping the image and
averaging logits before softmax would give a reliable ~1% boost.

---

## 9. Training Duration and Convergence

### 9.1 Per-Head Convergence Speed

Empirical observations:

| Head | Converges by epoch | Notes |
|---|---|---|
| Detection | 15-25 | TAL + focal converges quickly; class AP spread persists |
| Activity | 30-40 | 75-class CE converges slowly; long-tail classes never converge |
| PSR | 5-10 | Binary per-component converges fast; event F1 stays near-zero |
| Pose | 5-10 | 6-DoF regression is well-conditioned; converges to noise floor |
| Kendall log_vars | 5-15 | Relative precision stabilizes early |

### 9.2 50 vs 100 Epochs

The full pipeline uses 50 epochs (ST baselines) and 100 epochs (MTL).

At 50 epochs:
- Detection: fully converged (~epoch 20)
- Activity: 60-70% of asymptotic performance
- PSR: at noise floor by epoch 10
- Pose: at noise floor by epoch 10

At 100 epochs:
- Detection: overfits to dominant classes (AP variance increases)
- Activity: 80-90% of asymptotic — long-tail classes still not learned
- PSR/Pose: same as epoch 50 (already converged)

The 100-epoch schedule primarily helps activity. If activity converges by
epoch 40, epochs 40-100 are wasted for detection/PSR/pose.

### 9.3 Dynamic Stopping

Not implemented. A dynamic stopping rule:

- Track `act_top1` on val split with a patience of 15 epochs
- Stop when act_top1 plateaus (no improvement for 15 epochs)
- This would typically stop around epoch 50-60, saving 40-50% compute

---

## 10. Curriculum and Staged Training

### 10.1 Current State

Staged training is **disabled** by default (`STAGED_TRAINING = False`). The
config defines three stages but they are not used in production:

| Stage | Epochs | Active heads | Config |
|---|---|---|---|
| Stage 1 | 1-5 | Detection only | Detection-only warmup |
| Stage 2 | 6-15 | Detection + Pose | Add head pose + body keypoints |
| Stage 3 | 16-100 | All 4 heads | Full multi-task with EMA |

The staged approach was developed during the collapse-fix era (June 2026) but
abandoned once the FeatureBank gradient path was restored (root cause: in-place
tensor operations severed gradient flow from activity head to backbone).

### 10.2 Why Staging Was Dropped

1. **Gradient path fix**: The in-place tensor fix (2026-06-30) restored
   activity gradient to ~0.48, making staging unnecessary for preventing
   activity collapse.
2. **Activity ramp overhead**: The `ACT_RAMP_EPOCHS` ramp (epochs 1-3 with
   reduced activity weight) was the staging mechanism's last remnant; it was
   reduced from 5 to 3 epochs when the root cause was fixed.
3. **Head warm-starting**: `warm_start_heads_from_st()` (Task #260)
   initializes MTL heads from ST checkpoints, which achieves the "good init"
   goal of Stage 1 without needing architectural staging.

### 10.3 When Staging Would Help

Despite being disabled, staging has two genuine benefits:

1. **Detection warm-up**: Detection needs ~10-15 epochs to learn reasonable
   box predictions. During this time, detection gradients are essentially
   noise (random box proposals + focal loss on near-random classification).
   Other heads learning simultaneously waste capacity on this noise.
2. **PSR after detection**: PSR transition detection builds on object
   presence (you can't detect a "pick" transition without knowing where the
   object is). Training PSR only after detection has a basic object detector
   makes linguistic sense.

For a future curriculum:

| Phase | Epochs | Active heads | Purpose |
|---|---|---|---|
| 1 | 1-10 | Detection | Learn basic object localization |
| 2 | 11-25 | Detection + Pose | Add geometric head (uses detection features) |
| 3 | 26-50 | All 4 | Full MTL with learned detection backbone |
| 4 | 51-100 | Activity retrain | Freeze backbone, retrain activity classifier with balanced sampling |

Phase 4 mirrors the `--act-decoupled` flag (Kang et al. ICLR 2020) and is
the most promising curriculum improvement: train the backbone with
instance-balanced sampling for 50 epochs, then freeze backbone and retrain
the activity classifier with class-balanced sampling. This separates
representation learning (Phase 1-3) from classifier calibration (Phase 4).

### 10.4 The ACT_RAMP Mechanism

The current `ACT_RAMP_EPOCHS = 3` linearly increases activity loss weight
from 0 to full weight over epochs 1-3. This is a minimal curriculum that
prevents the randomly-initialized activity head from dominating the backbone
in the first few steps. After ramp is complete (epoch 3+), activity gets full
loss weight.

The mechanism works by multiplying `ACTIVITY_LOSS_WEIGHT` (0.8) by
`min(epoch / ACT_RAMP_EPOCHS, 1.0)`.

### 10.5 Detection GT Frame Fraction

The `DET_GT_FRAME_FRACTION` mechanism redistributes sampling mass so that a
target fraction of each batch is GT-bearing for detection. This is an
alternative to staging that addresses the detection sparse-GT problem at the
data level rather than the training-stage level.

Current setting: 0.40 (40% of batch frames have GT boxes). This means ~60%
of frames are activity/PSR/pose only, which is sufficient to keep detection
gradient alive (vs ~0.7% without this mechanism).

---

## 11. What Could Be Better: Consolidated Recommendations

### Priority 1: Staged Decoupled Activity Training

Replace the 100-epoch flat schedule with a 4-phase curriculum:

1. Detection only (epochs 1-10) — let the box head stabilize
2. Detection + Pose (epochs 11-25) — geometric supervision
3. Full MTL (epochs 26-50) — all heads, EMA tracking
4. Classifier retrain (epochs 51-60) — freeze backbone, retrain activity
   head with class-balanced sampling, no class weights

Expected gain: +3-5% activity top-1 without backbone forgetting.

### Priority 2: CAGrad Instead of PCGrad

Replace PCGrad's random-order sequential projection with CAGrad's convex
optimization:

```
min_d ||g - d||^2  s.t.  min_t <g_t, d> >= c * ||g||
```

where `g` is the average gradient, `g_t` are task gradients, and `c` is the
conservative rate (default 0.5). CAGrad:
- Has no random order dependence
- Guarantees Pareto descent (every task improves or stays same)
- Is only ~1.5x the compute cost of PCGrad

Expected gain: +1-2% on hardest classes per task.

### Priority 3: Class-Balanced Focal Loss for Activity

Enable `USE_CB_FOCAL_ACT = True` with beta=0.999, gamma=2.0. This replaces
the current CE + label smoothing + sqrt-tamed class weights with a principled
loss that:
- Uses effective number weighting (Cui et al.) to estimate marginal benefit
  of additional samples
- Applies focal modulation to down-weight easy majority classes
- Does not need `ACT_CLASS_WEIGHTS` or sqrt-taming heuristics

Expected gain: +2-4% on tail classes (activity top-1 improves even if top-5
stays flat).

### Priority 4: Per-Task Learning Rates

Add separate param groups for each head with task-specific LRs:

- Activity head: 2e-3 (needs aggressive learning for 75 classes)
- Detection head: 1e-3 (current rate works)
- PSR head: 1e-3 (current rate works)
- Pose head: 5e-4 (converges quickly, lower LR prevents overfitting)

Expected gain: +0.5-1% on activity and pose (detection flat).

### Priority 5: Direct Best-of-N Evaluation

Add test-time augmentation averaging for activity:

- Forward pass with original + horizontal flip
- Average logits before softmax
- Select argmax from averaged logits

Cost: 2x eval compute, ~0 parameter change. Expected gain: +0.5-1% activity
top-1 reliably.

### Priority 6: Dynamic Early Stopping

Track validation act_top1 with patience=15. Stop training when metric
plateaus. With 100-epoch schedule, typically stops at epoch 50-60, saving
40% compute.

### Priority 7: Remove OHEM Now That Focal Works

The OHEM ratio (2:1) was introduced during the collapse-fix era when
detection was dying. Now that TAL + asymmetric focal gamma + GT frame
fraction are stable, OHEM adds unnecessary complexity and may be filtering
useful moderate negatives. Removing it (or setting ratio to 0) would:
- Simplify the loss computation
- Let focal gamma handle hard-negative mining naturally
- Prevent over-filtering of informative negatives

Expected: neutral or +0.1-0.3 mAP.

---

## 12. Current Performance Boundaries

As of July 2026, the MTL training with all 6 levers (from `full_pipeline_v1.sh`)
hits these approximate limits:

| Task | Current | Estimated ceiling | Gap cause |
|---|---|---|---|
| Detection mAP@0.5 | 0.558 | 0.75-0.80 | Data sparsity (26K frames, 99.3% empty) |
| Activity top-1 | ~0.35 | 0.50-0.55 | 75 classes with 50 frames/class avg |
| PSR event F1 | ~0.006 | 0.15-0.25 | Transition rate <0.5%, 8-frame temporal resolution |
| Pose fwd MAE | ~5 deg | 2-3 deg | Joint geometry noise, 224px resolution |

The largest gap is PSR, where the loss is low (model is "happy" predicting
all zeros) but event F1 is near-zero. This is the hardest problem: BCE-based
losses for extremely rare events (<<1% positive) require fundamentally
different approaches — either set-prediction losses or asymmetric losses that
completely ignore negative gradient (ASL, see 4.3).

The second-largest gap is activity, where the long-tail class imbalance is
structural. 20 of 75 classes have <30 frames. No amount of re-weighting can
create generalization signal from 7 frames. Decoupled training (Priority 1)
is the most promising approach because it separates representation learning
from classifier calibration.

---

## References

- Kendall, Gal, Cipolla. "Multi-Task Learning Using Uncertainty to Weigh
  Losses for Scene Geometry and Semantics." CVPR 2018.
- Yu et al. "Gradient Surgery for Multi-Task Learning." NeurIPS 2020.
- Liu et al. "Just Pick a Sign: Optimizing Deep Multitask Models with
  Gradient Sign Dropout." NeurIPS 2022.
- Chen et al. "Gradient Normalization for Adaptive Multi-Task Loss Balancing."
  NeurIPS 2018.
- Navon et al. "Multi-Task Learning as a Bargaining Game." ICML 2022.
- Sener & Koltun. "Multi-Task Learning as Multi-Objective Optimization."
  NeurIPS 2018.
- Wang et al. "Rotograd: Gradient Homogenization in Multitask Learning." ICLR
  2022.
- Liu et al. "Towards Impartial Multi-task Learning." ICLR 2021.
- Wortsman et al. "Model Soups: Averaging Weights of Multiple Fine-Tuned
  Models Improves Accuracy Without Increasing Inference Time." ICML 2022.
- Izmailov et al. "Averaging Weights Leads to Wider Optima and Better
  Generalization." UAI 2018.
- Hinton, Vinyals, Dean. "Distilling the Knowledge in a Neural Network." 2015.
- Kang et al. "Decoupling Representation and Classifier for Long-Tailed
  Recognition." ICLR 2020.
- Menon et al. "Long-tail Learning via Logit Adjustment." ICLR 2021.
- Cui et al. "Class-Balanced Loss Based on Effective Number of Samples."
  CVPR 2019.
- Cao et al. "Learning Imbalanced Datasets with Label-Distribution-Aware
  Margin Loss." NeurIPS 2019.
- Zhou et al. "On the Continuity of Rotation Representations in Neural
  Networks." CVPR 2019.
- Feng et al. "Distilling the Knowledge in Multi-Task Learning." 2023.
