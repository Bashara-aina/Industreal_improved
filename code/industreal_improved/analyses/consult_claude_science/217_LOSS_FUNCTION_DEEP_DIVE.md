# 217: Loss Function Deep Dive — Per-Task Losses and Alternatives

> **Date:** 2026-07-11
> **Scope:** Detailed analysis of each task's loss function in the IndustReal MTL pipeline: current implementation, known failure modes, and principled alternatives from the literature. Intended as a decision-making reference for the consultation.
> **Ground truth:** Code in `scripts/train_mtl_mvit.py` (all four losses + Kendall weighting) and `src/losses/tal_assigner.py` (TaskAlignedAssigner). Training at 224 px, MViTv2-S backbone, 8 GB VRAM on RTX 3060.

---

## 0. Total Loss Composition

The total loss is a Kendall uncertainty-weighted sum of four task losses plus an optional distillation term:

```
L_total = sum_{t in {det, act, psr, pose}} [ exp(-lv_t) * L_t_norm + lv_t / 2 ] + L_distill
```

where `lv_t` is a learnable log-variance parameter per task and `L_t_norm` is the raw task loss optionally divided by its EMA value (running mean tracked at momentum 0.99). The EMA normalization (Opus 181 D1) ensures all task losses enter Kendall at approximately O(1) scale, preventing the natural inverse-loss-weighting collapse where high-loss tasks (activity: ~12.3) are driven to near-zero weight.

**Kendall clamping** (Doc 175, Opus 201/207):
- Per-task upper bounds on `lv_t`: det <= 1.5, act <= 1.0, psr <= 0.5, pose <= 2.0
- Pose precision capped by detection: `lv_pose = max(lv_pose, lv_det.detach())`
- Global floor: `lv >= -4.0` for all tasks

These caps prevent the Kendall mechanism from completely starving any task. The `kendall_uncapped` ablation (Opus 201) removes all caps and demonstrates collapse: activity weight goes to near-zero, detection under-clamps.

**PCGrad** optionally projects conflicting per-task gradients. Applied to shared backbone parameters only: for task pairs with cosine similarity < 0, the gradient of task i is projected away from the conflict direction. The random task ordering (a core PCGrad step) prevents bias toward any specific task order.

**Distillation loss** (Task 261): KL divergence between student and frozen single-task teacher logits, added directly (not Kendall-weighted) with a fixed `distill_alpha=0.1`.

---

## 1. Detection Loss

### 1.1 Current Implementation

The detection head uses a YOLOv8-style three-component loss with a TaskAlignedAssigner (TOOD, ICCV 2021):

```
L_det = L_cls / N + L_iou / N + L_dfl / N    (N = number of active FPN levels)
```

**Components:**

1. **Classification: Focal-BCE** — Binary cross-entropy with focal weighting on 24 classes. Uses one-hot targets (not soft labels). Focal parameters: gamma=2.0, alpha=0.5. The focal weight formula is standard: `w = alpha_t * (1 - pt)^gamma` where `pt = y * p + (1-y) * (1-p)`.

2. **Box regression: CIoU** — Complete IoU loss (Zheng et al. 2020) on decoded xyxy boxes. The loss is `1 - CIoU` where CIoU extends DIoU with aspect-ratio consistency. CIoU = IoU - rho^2/c^2 - alpha*v, where rho=center distance, c=enclosing box diagonal, v=arctan aspect-ratio mismatch, alpha=v/((1-IoU)+v). Only applied to positive cells (assigned by TAL).

3. **Distribution Focal Loss (DFL)** — Li et al. 2020. Replaces standard L1 regression with a discrete distribution over `reg_max=16` bins per coordinate (l, t, r, b offsets in grid units). Two nearest-bins are weighted by fractional distance. The predicted distribution is decoded to box coordinates as the expectation over bin centers. DFL gradient is denser than L1 because every bin gets some gradient signal.

**Assignment: TaskAlignedAssigner (TAL):**
- Alignment score per cell-GT pair: `s = cls_score^alpha * box_iou^beta` with alpha=1.0, beta=6.0
- Each GT assigned to top-k=10 cells per FPN level by alignment score
- Positive cells get both classification and regression targets; non-positive cells are ignored for regression
- Applied at P3 (stride 8), P4 (stride 16), P5 (stride 32). P2 (stride 4) is skipped because conv_proj features lack semantic content (Opus 192 FC-2).

**Fallback (use_tal=False):** Legacy 3x3 grid assignment around the GT center cell — gives sparse positives (9 per GT vs 30 with TAL across 3 levels). Known to produce 0.0 mAP (Probe EP10 evidence).

### 1.2 Known Problems

**Coarse box regression at 224 px.** At stride 8 (P3, the finest level), each cell covers 8 pixels. The DFL distribution spans reg_max=16 bins, giving a coverage of `16 * 8 = 128` pixels per side. For 224 px images where the assembly board occupies ~150-200 px, the box localisation precision is limited to approximately one cell width (8 px). This is acceptable for coarse localisation but limits fine-grained bounding.

**QFL may not help with 24 classes.** Detection uses Focal-BCE, not Quality Focal Loss (QFL, Li et al. 2020). QFL jointly estimates classification quality and IoU, typically beneficial when class count is small (COCO: 80 classes). With 24 classes where adjacent classes differ by one binary assembly component, the class separability problem may dominate over IoU quality estimation. The current focal-BCE with alpha=0.5 gives equal weight to positives and negatives at the loss level, which is reasonable for balanced detection.

**TAL top-k=10 may include too many negatives.** With 24 classes, many cells may have moderate alignment scores for the wrong class, and top-k=10 could assign positive targets to cells whose peak score is still very low. The alignment score `s = cls^1.0 * iou^6.0` heavily weights box quality — if a cell predicts the wrong class but high IoU, it still gets assigned. However, the classification loss then tries to push that cell's class score up, which may conflict with the truly correct cells.

### 1.3 Alternatives

#### IoU Variants

| Loss | Formula | Key Difference | Pros | Cons |
|------|---------|---------------|------|------|
| **GIoU** | IoU - (C\(AUB)/C) | Penalizes non-overlapping boxes | Gradient everywhere, even at IoU=0 | Slower convergence than CIoU |
| **DIoU** | IoU - rho^2/c^2 | Center distance only, no aspect ratio | Simpler, faster | Ignores shape mismatch |
| **EIoU** | DIoU - rho_w^2/c_w^2 - rho_h^2/c_h^2 | Width/height distance separately | Finer gradient for aspect ratio | Three distance terms, more hyperparameters |
| **SIoU** | IoU - (dist + shape) / 2 | Adds angle cost for vector alignment | Theoretically richer gradient | Complex, unproven for this domain |
| **α-IoU** | 1 - IoU^alpha | Power transform of IoU | Tunes sensitivity via alpha | Another hyperparameter |

**Recommendation:** CIoU is suitable. The main bottleneck for detection is not box regression quality (DET_PROBE bestIoU consistently > 0.90) but fine-grained class discrimination. IoU alternatives would address the wrong problem.

#### Classification Loss Variants

- **VarifocalLoss** (Zhang et al. 2021): Asymmetrically weights positives vs negatives using the predicted IoU as GT: `VFL = -y*(y*log(p)+(1-y)*log(1-p))` for positives, `-alpha*p^gamma*log(1-p)` for negatives. The GT score y is the predicted IoU, naturally coupling classification quality with IoU. **Quality Focal Loss** (Li et al. 2020) is the predecessor: `QFL = -|y-sigma|^beta*(y*log(sigma)+(1-y)*log(1-sigma))` with y as IoU-aware label. Either is a drop-in replacement for the current Focal-BCE.

#### Assigner Variants

- **TaskAlignedAssigner with dynamic top-k**: The current assigner uses static top-k=10. Alternatives include: (a) score-thresholding instead of top-k, filtering cells where alignment > threshold; (b) Hungarian matching (DETR-style) with bipartite matching loss; (c) SimOTA (YOLOX), which dynamically computes k per GT based on the top-k IoU value.

- **Hungarian matching** (Carion et al. 2020): Standard for DETR-based detectors. Assigns each GT to exactly one prediction via bipartite matching (cost = -log-prob + L1 + GIoU). Advantages: no anchor hyperparameters, one-to-one assignment (sparser than TAL's 10), natural handling of variable object counts. Disadvantages: O(n^3) per image, requires `n_queries >= max_objects` (not true for FPN anchor grids), harder training dynamics.

- **SimOTA** (Ge et al. 2021): Dynamic k per GT where k = sum of top-10 IoU values. Removes the static top-k hyperparameter.

---

## 2. Activity Loss

### 2.1 Current Implementation

```
L_act = CE(logits + tau * log(freq), targets, weight=class_weights, label_smoothing=0.05, ignore_index=-1)
```

**Components:**

1. **Cross-entropy** between logits and 75-class integer labels. Single-label setting (one activity per video clip).

2. **Logit adjustment** (Menon et al. 2020): Prior to softmax, `logits += tau * log(freq + 1e-9)` where `freq` is the normalized class frequency in the training set. This shifts the decision boundary toward rare classes by adding class-specific bias terms. At inference, raw logits (without adjustment) are used for argmax. The adjustment is INSIDE the loss (not in the model forward pass), following the Menon et al. additive formulation.

3. **Class weights**: Sqrt-tamed inverse-frequency weights. Raw inverse-frequency: `w_c = total / (N * count_c)`. These are then power-transformed: `w = w^0.5`. This reduces the max/min weight ratio from ~137 to ~12 (Opus 181 D1b). Applied as `F.cross_entropy(weight=class_weights)`.

4. **Label smoothing**: 0.05 (reduced from 0.1 in Opus 181 D1b). Adds a small uniform component to the target distribution: `y_smooth = (1 - smoothing) * y_onehot + smoothing / C`.

5. **Ignore index**: -1 for unlabeled sequences. Batch safety: if every label in a batch is -1, returns 0.0 loss with no gradient (P2 guard).

### 2.2 Known Problems

**Power-law distribution over 75 classes.** The activity labels follow a heavy-tailed distribution where a few classes dominate and many tail classes have < 100 examples. Despite logit adjustment + class weights + label smoothing, tail classes are almost never predicted. The logit adjustment and class weights interact: both push toward tail classes, but the net effect may overshoot or undershoot depending on the data distribution.

**Tail classes never predicted.** In eval, the activity confusion matrix shows concentration on the top ~15 classes. Mix of reasons: (a) the gradient signal for tail classes is diluted by the dominant CE loss on head classes; (b) the 3-layer MLP from cls_token (768-2048-1024-75) produces a single 75-d logit vector per clip, making the gradient path from CE to backbone long and sparse; (c) gradient norm for activity is 0.010 vs PSR at 3.180 (312x gap, Arch Doc 210).

**Logit adjustment interacts with class weights.** Both mechanisms increase the model's bias toward rare classes. Their interaction is: `logits' = logits + tau * log(freq)` then `CE(, weight=inv_freq_sqrt)`. The additive logit bias and multiplicative weight bias compound. A cleaner design would use only one (logit adjustment alone, with no class weights) and tune tau.

### 2.3 Alternatives

#### Focal Loss (Lin et al. 2017)

Down-weights easy examples: `FL = -alpha_t * (1-p_t)^gamma * log(p_t)`. **Pros:** well-studied, single gamma parameter. **Cons:** not designed for long-tail (down-weights ALL easy examples, including easy tail-class ones we want to emphasize); with 75 classes, per-class probability is ~1/75 so most examples are already "hard." Focal is unlikely to solve the activity tail-class problem — the core issue is gradient propagation, not within-loss weighting.

#### LDAM (Cao et al. 2019)

Label-Distribution-Aware Margin loss applies a class-dependent margin to the logits:

```
LDAM(x, y) = -log( exp(logit_y - delta_y) / (exp(logit_y - delta_y) + sum_{j != y} exp(logit_j)) )
```

where `delta_y = C / n_y^{1/4}` with `n_y` = class count and `C` = hyperparameter.

**Pros:** (a) Theoretically motivated by the generalization bound for long-tail learning. (b) Directly introduces larger margin for tail classes (margin inversely proportional to count^{1/4}). (c) Works well with deferred re-weighting (DRW) where uniform weights are used initially and class weights are introduced later in training. **Cons:** (a) Requires tuning C (default 0.5, but sensitive). (b) Margin-aware softmax is slightly more expensive than plain CE. (c) Typically combined with DRW schedule, adding complexity.

**Recommendation:** LDAM is the strongest alternative for the activity head. The theoretical foundation (margin proportional to generalization gap) directly addresses the tail-class problem. Combine with DRW: train with uniform class weights for the first ~80% of epochs, then switch to inverse-frequency weights.

#### Balanced Softmax (Ren et al. 2020)

Modifies the softmax denominator to include class priors:

```
BalancedSoftmax(x, y) = -log( exp(x_y) / (exp(x_y) + sum_{j != y} n_j/n_y * exp(x_j)) )
```

where `n_j` is the count of class j. This is equivalent to adding log(n_j/n_y) to the logit of class j before softmax.

**Pros:** (a) Minimal implementation change — just modify the denominator in softmax. (b) Theoretically correct for long-tail (derived from Bayesian prior adjustment). (c) No additional hyperparameters. **Cons:** (a) Can over-emphasize tail classes if the imbalance factor n_j/n_y is extreme. (b) Less well-studied than LDAM for >50 classes.

**Recommendation:** Balanced Softmax is simpler than LDAM and has no hyperparameters. Disadvantage: cannot be combined with label smoothing (the two modifications conflict).

#### Seesaw Loss (Wang et al. 2021)

Dynamically balances per-class gradients: the mitigator reduces penalty on tail class i when confused with head class j; the compensator increases it when tail class j is penalized by head class i. SOTA on LVIS (1230 classes). **Con:** complex (two hyperparameters), designed for multi-label detection.

#### Equalization Loss v2 (Tan et al. 2021)

Adapts per-class weight based on accumulated gradient ratios: `EQLv2 = -log(p_y) * gradient_balancing_weight`. Prevents head classes from dominating the classifier. **Con:** requires per-class gradient stats; designed for multi-label.

#### Multi-label vs Single-label

Reformulate activity as 11 binary assembly states (same components as PSR) instead of 75 single-label classes:

**Pros:** Dense gradient signal (11 binaries vs 1-of-75), directly aligned with PSR features, naturally balanced (~50% active per component), 2^11=2048 combinations — model could generalize to unseen states. **Cons:** Requires re-labeling from 75-class to 11-binary, changes eval metrics to mA, possible conflicting labels (impossible combinations).

---

## 3. PSR Loss

### 3.1 Current Implementation

```
L_psr = focal_weight * BCE(pred, target) * frame_weight * comp_weight  (mean over all elements)
```

**Components:**

1. **Focal-BCE**: Binary focal loss on 11 per-component transition logits across T=8 frames (downsampled from T=16 via adaptive max-pool). Focal parameters: alpha=0.25, gamma=2.0. Per-frame classification of whether each of the 11 assembly components is present.

2. **Per-component weights**: Inverse-prevalence weights normalized to component 0 (first component). Either from `C.PSR_COMP_WEIGHTS` (precomputed) or computed per-batch as fallback.

3. **Transition-aware frame weighting** (Opus 207): Frames near 0-to-1 transitions get a `transition_boost=3.0` multiplier. Detected as `(targets[:, 1:] - targets[:, :-1]).clamp(min=0)` — only 0->1 transitions (assembly events), not 1->0. The boost applies to the transition frame and its predecessor: `frame_weight = 1.0 + (boost - 1.0) * (has_transition + neighbor_boost)`.

4. **Temporal downsampling**: PSR head outputs T=8 (after spatial-temporal pooling in the head). Targets at T=16 are downsampled via adaptive max-pool1d, which preserves transition events (max over adjacent frames retains 1s).

### 3.2 Known Problems

**Extreme imbalance: < 1% positive.** The fundamental challenge: assembly components are present for a small fraction of total frames. Focal loss (alpha=0.25 on positives) partially addresses this, but the focal mechanism primarily down-weights easy negatives rather than boosting rare positives. With gamma=2.0, the positive focal weight is `alpha * (1-p)^2`; if p for positives starts low (~0.1), the weight is 0.25 * 0.81 = 0.20, which is still low.

**Transition-aware weighting improves recall but may clip precision.** The Opus 207 fix boosts frames near transitions by 3x. This increases gradient magnitude on the few positive frames. However, the boost is binary (on/off based on transition detection) and may create a sharp boundary where positive-frame gradients dominate, causing the model to over-predict transitions.

**Temporal structure is ignored at the loss level.** The current loss treats each frame independently. No temporal smoothness term, no ordering constraint (assembly events are monotonic in practice: components don't un-assemble), no consistency regularization. The causal Transformer head provides implicit temporal modeling, but the loss doesn't exploit the monotonicity of the assembly process.

### 3.3 Alternatives

#### Transition-Aware BCE Variants

- **Monotonicity regularization**: Add a penalty when a component decreases (1->0 transition). Monotonicity is a hard constraint in assembly (screws don't unscrew themselves). Loss: `L_mono = sum( clamp(targets[:, t+1] - targets[:, t], max=0) )` — penalizes predicted decreases. This is asymmetric: increases (assembly) are free, decreases (disassembly) are penalized. Expected to stabilize predictions at component boundaries.

- **Edge-aware focal** (custom): Continuous transition weight `exp(-d^2/sigma^2)` where d is frame distance from nearest transition. Avoids the hard binary boundary of the current boost.

- **Weighted BCE with dynamic alpha**: Adapt focal alpha per component via EMA of recall: `alpha_c = 0.5 + 0.5 * (1 - recall_ema_c)`. Directly addresses per-component imbalance variation.

#### Dice Loss (Milletari et al. 2016)

```
Dice = 1 - (2 * |pred ∩ target| + 1) / (|pred| + |target| + 1)
```

where `|pred ∩ target| = sum(pred * target)` and `|pred| = sum(pred)`.

Applied per component as a soft Sørensen-Dice coefficient.

**Pros:** Naturally handles imbalance via overlap ratio (segmentation standard where foreground < 10% of pixels — directly analogous to PSR's < 1% positives). Predictions are already sigmoid logits, so Dice is a drop-in replacement. **Cons:** requires smoothing term when prediction and target are both zero; combined Dice+BCE adds an alpha hyperparameter. **Recommendation:** Strongest alternative for PSR. Use Dice+BCE combo: `L = alpha*Dice + (1-alpha)*BCE` with alpha=0.7 standard starting point.

#### Tversky Loss (Salehi et al. 2017)

```
Tversky = 1 - (|pred ∩ target| + 1) / (|pred ∩ target| + beta*|pred \ target| + alpha*|target \ pred| + 1)
```

where alpha and beta control the penalty on false positives vs false negatives.

**Pros:** (a) More flexible than Dice (which is Tversky with alpha=beta=0.5). (b) Setting alpha > beta emphasizes recall — directly addresses the < 1% positive imbalance by penalizing false negatives more. **Cons:** (a) Two hyperparameters (alpha, beta). (b) Less established than Dice in non-segmentation literature.

#### Temporal Consistency Regularization

- **Temporal smoothness (L2 on frame differences)**: `L_smooth = sum_t ||pred_t - pred_{t-1}||^2`. Penalizes jittery predictions. Strength: the assembly is piecewise constant (components snap on/off), so predictions should be mostly constant with sharp transitions.

- **KL consistency (temporal ensembling)**: Weakly enforce temporal agreement via EMA of adjacent frame predictions (mean-teacher style).

#### CLIP-Style Contrastive Loss

Reformulate PSR as video-text matching: encode the assembly state sequence as a text prompt and learn visual-textual alignment. Contrastive formulation naturally handles imbalance (one positive vs N-1 negatives per clip). **Cons:** requires text encoder + alignment projection, formulaic assembly prompts unlikely to benefit from pre-trained CLIP, instability at small batch sizes.

---

## 4. Pose Loss

### 4.1 Current Implementation

```
L_pose = [ (1 - cos(fwd_pred, fwd_gt)) + (1 - cos(up_pred, up_gt)) ] + geodesic_angle(R_pred, R_gt)
       = cosine_loss + geodesic_loss
```

**Components:**

1. **Cosine similarity on fwd/up vectors**: The model predicts 6D pose (raw_6d), split into forward (first 3) and up (last 3) vectors. Each is L2-normalized independently: `fwd = F.normalize(raw_6d[:, :3])`, `up = F.normalize(raw_6d[:, 3:])`. The cosine loss is `1 - cos(fwd_pred, fwd_gt) + 1 - cos(up_pred, up_gt)`, equivalent to MSE on the unit sphere.

2. **Geodesic angular error**: Constructs SO(3) rotation matrices from the (fwd, up) pair via Gram-Schmidt orthonormalization:
   - `b1 = normalize(fwd)`
   - `b2 = normalize(up - proj(up, b1))`
   - `b3 = cross(b1, b2)`
   - `R = stack([b1, b2, b3], dim=2)` forming a valid 3x3 rotation matrix.
   The geodesic loss is the angular error: `arccos((trace(R_pred^T @ R_gt) - 1) / 2)` in degrees.

3. **Target vectors**: Ground truth is provided as 9D (3 fwd + 3 up + 3 right or unused). Only fwd and up (first 6D) are used. The target vectors are already unit-normalized in the dataset.

### 4.2 Known Problems

**Ground truth is vectors, not rotation matrices.** The GT annotation gives camera-relative forward and up vectors. These are inherently noisy (manual annotation) and may not be exactly orthogonal. The cosine loss on individual vectors is robust to this noise (it doesn't require orthogonality), but the geodesic loss assumes a valid SO(3) matrix. Gram-Schmidt constructs an orthonormal basis from the fwd/up pair, which effectively "repairs" near-orthogonal noisy vectors. If the GT vectors are far from orthogonal, the Gram-Schmidt step introduces a systematic bias.

**Orthonormalization happens in the loss, not in the model.** Both `renormalize_pose` (L2 norm) and `gram_schmidt_rotation` are applied as a post-processing step in the loss function. The model outputs raw_6d which is not explicitly constrained to be orthogonal. This means:
- The model gradient propagates through the Gram-Schmidt computation graph, which is numerically stable but complex
- The model weights must learn to produce consistent (fwd, up) pairs indirectly through the loss gradient
- During inference, the same orthonormalization pipeline is applied to get final rotation matrices

**Geodesic + cosine: redundant or complementary?** The cosine loss individually on fwd and up provides gradient even when the SO(3) geodesic angle is small (e.g., fwd vector is rotated around the up axis — the fwd cosine loss still fires while the geodesic angle may be dominated by the up component). However, for typical pose errors, the two are highly correlated. An analysis of gradient cosine similarity between the two losses during training would determine redundancy.

### 4.3 Alternatives

#### Pure Geodesic Loss

Remove the cosine terms and rely solely on the SO(3) geodesic angular error:

```
L_pose = geodesic_angle(GramSchmidt(renorm(pred_6d)), GramSchmidt(renorm(gt_6d)))
```

**Pros:** (a) Principled — geodesic is the true metric on SO(3). (b) Single loss, no weighting. (c) Directly optimizes the evaluation metric (angular MAE). **Cons:** (a) No gradient signal for the individual vector directions when the Gram-Schmidt compensated for them. (b) Geodesic alone can be slow to converge (the gradient near identity is small because arccos has near-infinite slope at 0 but the trace of R_rel is close to 3). (c) Loss is bounded [0, 180] degrees — early training sees ~90 deg loss regardless.

#### Quaternion Loss

Predict a unit quaternion: `L_quat = 1 - |q_pred · q_gt|` (absolute dot product handles antipodal symmetry). Compact 4D representation, no orthonormalization, well-studied in pose estimation. **Cons:** antipodal symmetry requires absolute dot product, unit-norm constraint requires L2 normalization in forward pass. Cleanest alternative to 6D+Gram-Schmidt, but requires changing model head from 6D to 4D and retraining from scratch.

#### Direct 6D MSE

Replace the cosine + geodesic combination with simple MSE on the raw 6D output:

```
L_pose = MSE(fwd_pred, fwd_gt) + MSE(up_pred, up_gt)
```

**Pros:** (a) Simplest possible loss. (b) Gradient is straightforward L2 — no Gram-Schmidt computation graph. (c) Faster backward pass. **Cons:** (a) Ignores the unit vector constraint — the model can output arbitrary magnitudes. (b) Requires normalization at inference time. (c) No rotational consistency — fwd and up are treated independently, so the predicted fwd and up might not correspond to any valid rotation. (d) Loss scale depends on vector magnitude, which is unbounded without normalization.

#### Direct 6D + Orthogonalization in the Model

Move Gram-Schmidt INTO the model forward pass so outputs are guaranteed valid SO(3) matrices. Loss becomes pure geodesic: `L_pose = geodesic(R_pred, R_gt)`. **Pros:** guaranteed valid rotations, loss directly optimizes the metric, simpler computation. **Cons:** 3x3=9 values instead of 6 (marginally larger head), prevents non-SO(3) uncertainty modeling.

**Recommendation:** This is the preferred change. Cleanly separates representation from supervision. Current 8.7 deg MAE is good, but baking orthonormalization into the model is architecturally cleaner.

---

## 5. Cross-Cutting Considerations

### 5.1 Loss Scale Interaction with Kendall Weighting

The Kendall uncertainty weighting treats all tasks as having the same functional form: `L_kendall = exp(-lv) * L_norm + lv/2`. However, the four losses have fundamentally different scales:

| Task | Raw Loss Scale | After EMA Normalization | Kendall lv Cap | Implied Weight Floor |
|------|---------------|------------------------|----------------|---------------------|
| Detection | ~0.5-1.0 | ~1.0 | 1.5 | exp(-1.5) = 0.22 |
| Activity | ~12.3 | ~1.0 | 1.0 | exp(-1.0) = 0.37 |
| PSR | ~1.3 | ~1.0 | 0.5 | exp(-0.5) = 0.61 |
| Pose | ~0.01-0.1 | ~1.0 | 2.0 | exp(-2.0) = 0.14 |

The per-task Kendall caps constrain weight floors differently. This was intentional (Opus 181 D2): high-loss tasks need higher weight floors to prevent starvation. Detection (cap 1.5, floor 0.22) has the most room to be down-weighted, while PSR (cap 0.5, floor 0.61) is almost never down-weighted.

### 5.2 Gradient Normalization Alternatives

The current system uses EMA-normalized Kendall (Opus 181 D1) to address loss scale mismatch. Alternatives:

- **GradNorm** (Chen et al. 2018): Dynamically adjusts task weights to equalize gradient norms across tasks. More aggressive than Kendall — can destabilize given the intrinsic 312x gradient gap between activity (0.01) and PSR (3.18).
- **Uncertainty weighting with per-task EMAs** (current): Simple, stable. EMA momentum (0.99) controls adaptation speed.
- **Fixed weights**: Requires grid search over 3 DOF; no starvation protection.
- **PCGrad-only**: Already active on the shared backbone; Kendall + PCGrad is a stacked approach.

### 5.3 Recommendations Summary

| Task | Current | Recommended Change | Priority | Expected Impact |
|------|---------|-------------------|----------|-----------------|
| Detection | CIoU + DFL + Focal-BCE | Keep current | Low | Small (bottleneck is class discrimination, not regression) |
| Activity | CE + logit-adjust + class weights | LDAM with DRW; or 11-binary multi-label | High | Medium-high (tail-class activation) |
| PSR | Focal-BCE + transition-boost | Add Dice component (Dice+BCE combo); optional monotonicity reg | High | Medium (better recall on rare transitions) |
| Pose | Cosine + geodesic | Move Gram-Schmidt into model; pure geodesic | Medium | Low-medium (8.7 deg MAE is already good) |
| Weighting | EMA-Kendall + PCGrad | Keep current | Low | Stability (already functional) |

---

## References

1. Zheng et al., "Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression", AAAI 2020. [CIoU]
2. Li et al., "Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection", NeurIPS 2020. [DFL, QFL]
3. Feng et al., "TOOD: Task-aligned One-stage Object Detection", ICCV 2021. [TAL assigner]
4. Menon et al., "Long-tail Learning via Logit Adjustment", ICLR 2021.
5. Cao et al., "Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss", NeurIPS 2019. [LDAM]
6. Ren et al., "Balanced Meta-Softmax for Long-Tailed Visual Recognition", NeurIPS 2020.
7. Wang et al., "Seesaw Loss for Long-Tailed Instance Segmentation", CVPR 2021.
8. Tan et al., "Equalization Loss v2: A New Gradient Balance Approach for Long-Tailed Object Detection", CVPR 2021.
9. Milletari et al., "V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation", 3DV 2016. [Dice loss]
10. Salehi et al., "Tversky loss function for image segmentation using 3D fully convolutional deep networks", MLMI 2017.
11. Kendall et al., "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics", CVPR 2018.
12. Chen et al., "GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks", ICML 2018.
13. Yu et al., "PCGrad: Gradient Surgery for Multi-Task Learning", NeurIPS 2020.
14. Carion et al., "End-to-End Object Detection with Transformers", ECCV 2020. [DETR, Hungarian matching]
15. Zhang et al., "VarifocalNet: An IoU-aware Dense Object Detector", CVPR 2021. [VarifocalLoss]
16. Ge et al., "YOLOX: Exceeding YOLO Series in 2021". [SimOTA]
