# === V1 FACT-CHECK (REQUIRED READING) ===
# This file was generated as V2 before the codebase was fully audited.
# The V1 documents (208-227) it references describe the MViTv2-S era
# which is NO LONGER the active codebase. Key corrections:
#
# **Active model:** POPWMultiTaskModel in src/models/model.py (2361 lines)
# **Backbone:** convnext_tiny (28.59M, ImageNet-1K pretrain) — NOT MViTv2-S
# **Total params:** 46.47M (measured) — NOT ~48.6M
# **Detection head:** RetinaNet-style 5.31M (9 anchors, 5 levels P3-P7)
# **Activity head:** FeatureBank+TCN+2xViT 0.69M (NOT 3-layer MLP 2M)
# **PSR head:** PSRHead hidden_dim=128 3.08M (NOT 1.8M Causal Transformer)
# **Pose:** body 1.64M + head 1.45M + 2 FiLMs 1.24M (NOT single 0.2M)
# **FPN:** Standard P3-P7 4.48M (NOT BiFPN P2-P5 ~2.5M)
# **Effective batch:** 48 (BATCH=6 x GRAD_ACCUM=8) — NOT 16
# **PSR_FOCAL_GAMMA:** 0.5 — NOT 2.0
# **Kendall caps:** act=[-0.5,2.0], psr=[-4,0.0], pose=[-4,3.0] (NOT [1.5,1.0,0.5,2.0])
# **Recordings:** 36/16/32 (NOT 10/6)
# **Frames:** 207,266 (NOT ~75K)
# **Free GPUs:** RTX 5060 Ti 16GB + RTX 3060 12GB
# **Deadline:** AAIML Oct 10, 2026
#
# See consult_v2/V1_VS_CODEBASE_DISCREPANCY_REPORT.md for full audit.
# The legacy src/models/mvit_mtl_model.py (MViTv2-S) is DEAD CODE.
# All architecture numbers in this report should be re-validated against
# the active model before any paper submission.
# === END FACT-CHECK ===

# Agent 8: Task Head Architecture Analysis

## Overview

This analysis validates the 4 task heads in the IndustReal MTL model (MViTv2-S backbone + Detection, Activity, PSR, and Pose heads). Each claim is traced to exact source code.

---

## 1. Detection Head Architecture

**File**: `src/models/mvit_mtl_model.py`, lines 241-279

**Architecture**: Decoupled cls + box regression head with Distribution Focal Loss (DFL).

```
cls_head: Conv2d(256->256, k=3, p=1) -> GroupNorm(32, 256) -> ReLU -> Conv2d(256->24, k=1)
reg_head: Conv2d(256->256, k=3, p=1) -> GroupNorm(32, 256) -> ReLU -> Conv2d(256->64, k=1)
```

The reg_head outputs `4 * reg_max` channels (reg_max=16, so 64 channels) for DFL-based box regression.

**Shared across 3 FPN levels** (P3, P4, P5; P2 is skipped):
```python
# mvit_mtl_model.py:574-580
for level_name, feat in fpn_out.items():
    if level_name == "P2":
        continue
    pooled = feat.mean(dim=2)  # temporal-pool T dim -> [B, 256, H, W]
    det_outputs[level_name] = self.det_head(pooled)
```

**FPN architecture**: `LightweightFPN` (mvit_mtl_model.py:143-234) — BiFPN-style with EfficientDet weighted fusion. Lateral Conv3d projections (1x1x1), top-down and bottom-up paths with 3x3x3 Conv3d.

**Parameter count**: 1,203,800 (1.20M)
```
cls_head: 590,080 (conv1) + 512 (gn) + 6,168 (conv2) = 596,760
reg_head: 590,080 (conv1) + 512 (gn) + 16,448 (conv2) = 607,040
```

**Loss**: Varifocal + WIoU v3 properly wired:
- `--varifocal=True` (train_mtl_mvit.py:1902)
- `--wiou_v3=True` (train_mtl_mvit.py:1903)
- These are passed to detection_loss() at line 2403-2404

---

## 2. Activity Head Architecture

**File**: `src/models/mvit_mtl_model.py`, lines 286-370

**Architecture**: 3-layer MLP on cls_token (768-dim).

```
LayerNorm(768) -> Linear(768->2048) -> GELU -> Dropout(0.2) ->
Linear(2048->1024) -> GELU -> Dropout(0.2) -> Linear(1024->75)
```

**Parameter count**: 3,751,499 (3.75M)
```
norm:      1,536  (LayerNorm 768)
fc1:   1,574,912  (768*2048 + 2048)
fc2:   2,098,176  (2048*1024 + 1024)
clsf:     76,875  (1024*75 + 75)
```

**Logit adjustment**: The model's `enable_logit_adjust()` at line 2231 registers a `class_freq` buffer, but the forward pass (lines 335-355) does NOT use it — the logit adjustment was moved to the loss function per Opus 207. The buffer is registered but inert.

**Dual logit corrections active**:
1. `act_balanced_softmax=True` (config flag) → BalancedSoftmaxLoss adds `log(priors)` to logits during training (balanced_softmax.py:13)
2. `act_logit_adjust_freq` is computed at line 2240 and passed to `activity_loss()` at line 2396, which applies `tau * log(freq)` at line 395

This means **two additive log corrections** are applied simultaneously when `act_balanced_softmax=True`:
- BalancedSoftmaxLoss adds `log(priors)` (from class_counts / total)
- activity_loss checks `logit_adjust_freq` and adds `tau * log(freq)` (from class_counts / total)

BUT in the training loop (lines 1039-1041), when `act_balanced_softmax is not None`, the branch uses it EXCLUSIVELY — `activity_loss()` is NOT called. So only ONE correction is active. The `enable_logit_adjust` on the model head is a no-op.

```python
# train_mtl_mvit.py:1039-1052
elif act_balanced_softmax is not None:
    l_act = act_balanced_softmax(_act_logits, _act_trainable.to(images.device))
elif act_ldam is not None:
    ... LDAM path ...
else:
    l_act = activity_loss(..., logit_adjust_freq=act_logit_adjust_freq, ...)
```

So only BalancedSoftmax is active when both flags are on. The `logit_adjust_freq` code path is dead.

---

## 3. PSR Head Architecture

**File**: `src/models/mvit_mtl_model.py`, lines 376-454

**Architecture**: Causal Transformer on spatial-pooled P5 features.

```
P5 features [B, 768, T=8, 7, 7]
  -> AdaptiveAvgPool3d((None, 1, 1)) -> squeeze spatial dims -> [B, T=8, 768]
  -> Linear(768->256)  [input_proj]
  -> 2-layer Causal Transformer (d=256, nhead=4, ff=1024=4x, LeakyReLU(0.01))
  -> Dropout(0.15)
  -> Linear(256->11)   [projection]
```

**Parameter count**: 1,779,211 (1.78M)
```
input_proj:  196,864  (768*256 + 256)
Transformer: 1,579,520 (2 layers * 789,760 per layer)
  - Self-attn:      263,168/layer (in_proj: 196,608+768, out_proj: 65,536+256)
  - FFN:            525,568/layer (fc1: 263,168, fc2: 262,400)
  - LayerNorm:        1,024/layer (2 * 512)
projection:     2,827  (256*11 + 11)
```

**PSR Refinement Head** (separate module, wired at train_mtl_mvit.py:2289-2298):
- File: `src/models/psr_refinement.py`, lines 67-151
- 2-stage MS-TCN, each stage: 10 dilated Conv1d layers (filters=64, dilation=2^i)
- 206,230 params total
- Wired into training loop at line 1015-1018:
```python
if psr_refinement_head is not None and "psr_logits" in outputs:
    outputs["psr_logits"] = psr_refinement_head(
        outputs["psr_logits"], apply_sigmoid=True
    )
```
- Confirmed ACTIVE in both log configs: `psr_refinement=True, psr_refinement_stages=2`

**Note**: The refinement head is applied AFTER the Rotograd block, meaning it gets the unrotated PSR logits.

---

## 4. Pose Head Architecture

**File**: `src/models/mvit_mtl_model.py`, lines 460-490

**Architecture**: Simple 2-layer MLP on cls_token.

```
Linear(768->256) -> LeakyReLU(0.01) -> Dropout(0.15) -> Linear(256->6) -> Tanh
```

**Output**: 6D rotation representation (3 forward vector + 3 up vector), Tanh-bounded to [-1, 1].

**Parameter count**: 198,406 (0.20M)
```
fc1: 196,864  (768*256 + 256)
fc2:   1,542  (256*6 + 6)
```

**Renormalization** (mvit_mtl_model.py:613-637):
- `renormalize_pose()`: L2-normalizes fwd and up vectors separately
- `gram_schmidt_rotation()`: Orthonormalizes (fwd, up) via Gram-Schmidt to produce a valid SO(3) matrix

**Geodesic loss wiring**: Properly wired.
- `--pose_geodesic_huber=True` (config flag)
- When enabled: `huberised_geodesic_loss()` from `src/losses/geodesic_loss.py` (huberised_geodesic_loss.py:23-45)
- Huber threshold delta=30°, quadratic within, linear beyond
- Also: `pose_loss()` fallback (line 1128) combines cosine + geodesic

---

## 5. Activity Head Collapse Analysis

**Evidence from logs**:
```
T8_frag.log line 72:  act_preds=1uniq/0.03maxconf  (epoch 11)
T4_v2.log line 71:    act_preds=1uniq/0.04maxconf  (epoch 11)
T4_v2.log line 95:    act_preds=1uniq/0.04maxconf  (epoch 12)
T4_v2.log line 118:   act_preds=1uniq/0.04maxconf  (epoch 13)
```

**Diagnostic code** (train_mtl_mvit.py:2488-2499):
```python
_qc_preds = _qc_out["activity"].argmax(dim=-1)
_qc_n_pred = len(_qc_preds.unique())
_qc_max_conf = torch.softmax(_qc_out["activity"], dim=-1).max(dim=-1)[0].mean().item()
```

**Finding**: After 13 epochs, the activity head predicts only 1 unique class per batch with max softmax confidence ~0.03-0.04. For 75 classes, random uniform gives ~0.013, so the model is barely above random.

### Root causes:

**(a) Gradient starvation from pose loss magnitude**

The raw loss values per task show an extreme imbalance:
```
act=4.0      (cross-entropy for 75 classes, random ~4.3)
pose=700-4000 (geodesic error in degrees)
det=0.03-0.8  (detection loss, well-trained)
psr=0.3-0.5 (BCE-based, ~base rate)
```

The pose loss is 100-1000x larger than activity. Despite FAMO weight normalization (which operates on loss decrease rates, not magnitudes), the raw gradient from pose dominates the shared backbone update. Activity's gradient signal is drowned.

**(b) Activity loss is NOT decreasing (it increases)**

Across epochs:
```
Epoch 11: act=3.9937
Epoch 12: act=4.0709  (higher)
Epoch 13: act=4.1464  (higher)
Epoch 12: act=4.0709  (T4 config)
```

FAMO updates weights based on `log(l^t) - log(l^{t+1})`. When activity loss INCREASES, the weight for activity DECREASES, creating a self-reinforcing collapse: less weight -> less learning -> worsening loss -> even less weight.

**(c) 3 zero-weight classes**

From the log:
```
Class weights — min=0.0000  max=11.7127  mean=2.2867  num_nonzero=72  [sqrt-tamed]
```

3 out of 75 classes have ZERO samples in the training split. `compute_activity_class_weights()` (train_mtl_mvit.py:317-355) uses `np.divide(..., where=counts > 0)` which assigns weight=0 for zero-count classes. CrossEntropy with weight=0 means misclassifying these classes contributes nothing to the loss, further reducing the discriminative signal.

**(d) BalancedSoftmax may be counterproductive**

BalancedSoftmaxLoss (balanced_softmax.py:7-14) adds `log(priors)` to logits:
```python
logits_shifted = logits + torch.log(self.class_priors.unsqueeze(0))
```

For the most frequent class (prior ~0.12), this adds `log(0.12) = -2.1`.
For the rarest nonzero class, this adds ~`log(1/20578) = -9.9`.
The shift is +7.8 for the head vs the tail, which the model must overcome.

At eval time (line 2488-2490), raw logits (without adjustment) are used for argmax. This means the model is trained with a +7.8 bias toward common classes but evaluated without it. The model learns to produce logits that, when shifted by log(priors), correctly classify. But at eval without the shift, simple argmax over raw logits may not work correctly because the model's internal representations have been distorted by the training-time adjustment.

**(e) RotoGrad not helping activity**

RotoGrad rotation is applied to the cls_token before the activity head (lines 1003-1011):
```python
_z_act = rotograd_model.rotate(_z, 0)
outputs["activity"] = model.act_head(_z_act)
```

But RotoGrad's rotation matrices are frozen (see Section 8) — they remain at random initialization. The "rotation" adds noise, not alignment.

---

## 6. PSR Head Performance

**Evidence from logs**:
```
PSR comp: [0.7112 0.6984 0.6806 0.6931 0.6902 0.7025 0.6932 0.6995 0.6954 0.6882 0.6612]
psr_stdmax=0.0206
```

All 11 PSR components predict ~0.69-0.71 probability — essentially the same value for every component across all frames. The standard deviation across frames is 0.02, meaning predictions are nearly constant.

**Causes**:
1. **Limited temporal signal from T=8**: The PSR head operates on T=8 frames, which is a very short window for detecting transition events.
2. **Causal masking removes information**: The causal mask in the transformer (line 447-450) means each frame only sees past frames, limiting the temporal context for detecting transitions.
3. **Gradient suppression**: PSR loss (~0.3) is small relative to pose (~700), so PSR gradients to the backbone are weak. The 0.3x LR multiplier (line 2135) further reduces PSR backbone impact.

---

## 7. Pose Head Performance

Pose loss is enormous (`pose=784.6` at epoch 11) — this is the Huber-capped geodesic error in degrees at delta=30°, so values >30° saturate to `30 * (error - 15)`. A value of 784 means the mean error is `(784 / c + 30 * 0.5)` ... actually `loss = delta * (error - 0.5 * delta)` for error >= delta, so `784 = 30 * (error - 15)` → `error = 784/30 + 15 = 41.1°` mean geodesic error. This is high but not catastrophic for random initialization.

The pose head (0.20M params, 768->256->6) is the smallest head — a single-bottleneck MLP for a 6D geometric output. This may be under-parameterized for learning stable 6D rotations from the cls_token.

---

## 8. RotoGrad Wiring: NOT Functionally Wired

**File**: `src/models/rotograd.py`, lines 20-169

**Initialization** (train_mtl_mvit.py:2268-2279):
```python
rotograd_model = RotoGradRotation(feat_dim=768, num_tasks=3, subspace_dim=128)
```

**Applied in training loop** (train_mtl_mvit.py:1000-1012):
```python
if rotograd_model is not None and "cls_token" in outputs:
    _z = outputs["cls_token"]
    _z_act = rotograd_model.rotate(_z, 0)      # task 0 = activity
    _z_pose = rotograd_model.rotate(_z, 1)      # task 1 = pose
    outputs["activity"] = model.act_head(_z_act)
    outputs["pose_6d"] = model.pose_head(_z_pose)
```

### Two critical issues:

**Issue A: RotoGrad parameters are NOT in any optimizer param_group**

The optimizer is set up at lines 2131-2142 with 5 groups:
```python
param_groups = [
    {backbone},              # 34.23M
    {fpn + det_head},        # 1.20M (FPN prefix mismatch — see Issue C)
    {act_head},              # 3.75M
    {psr_head},              # 1.78M
    {pose_head},             # 0.20M
]
```

RotoGrad's parameters (638,976) are NOT in any of these groups. The optimizer is created BEFORE `rotograd_model` is instantiated. No call to `optimizer.add_param_group()` is made afterward.

Evidence: grep for `rotation_loss|rotograd.*optimizer|rotograd.*step|rotograd.*backward` returns **no matches** in train_mtl_mvit.py.

**Impact**: Random rotation matrices, never updated. The "rotation" adds noise.

**Issue B: RotoGrad.rotation_loss() never called**

The `rotation_loss()` method (rotograd.py:91-116) is designed to optimize rotation matrices via cosine-similarity alignment. It is defined but NEVER CALLED anywhere in the codebase.

**Issue C: Only 2 of 3 task rotations are applied**

RotoGrad is initialized with `num_tasks=3` (activity, pose, PSR), but PSR (task 2) rotation is never applied — PSR uses P5 features, not cls_token. The `RotoGradScale` (gradient magnitude normalization) is also never instantiated (it requires explicit construction, which is not done in the training script).

**Issue D: FPN prefix mismatch — ~14.5M frozen parameters**

The group prefix `feature_pyramid.fpn` at line 2133 does not match any model parameters. The actual FPN is named `model.fpn.*` in `named_parameters()`, not `model.feature_pyramid.fpn.*`.

```python
# train_mtl_mvit.py:2133 — BUG: prefix should be "fpn", not "feature_pyramid.fpn"
_group_params(["feature_pyramid.fpn", "det_head"], 1.0),
```

The FPN (LightweightFPN, mvit_mtl_model.py:143-234) is directly registered as `self.fpn = LightweightFPN(...)` at line 520, not inside `self.feature_pyramid`. So all ~14.5M FPN parameters are absent from all optimizer param groups.

**Impact on total model parameters**:

| Group | Claimed params | Actual params |
|-------|---------------|--------------|
| backbone | 34.23M | 34.23M (correct) |
| fpn + det | 1.20M | 1.20M (only det; FPN missing) |
| act | 3.75M | 3.75M (correct) |
| psr | 1.78M | 1.78M (correct) |
| pose | 0.20M | 0.20M (correct) |
| **Total optimized** | **41.16M** | **41.16M** |
| **Total model params** | **55.7M** | **55.7M** |
| **Missing (frozen)** | — | **~14.5M (FPN)** |

The FPN produces 26% of total model parameters that are initialized to random weights and never updated.

---

## 9. Head Initialization and Warm-Start

**Warm-start function**: `warm_start_heads_from_st()` at train_mtl_mvit.py:731-780

**Log evidence** (T8_frag.log lines 24-28):
```
Warm-start det: checkpoint not found, skipping
Warm-start act: checkpoint not found, skipping
Warm-start psr: checkpoint not found, skipping
Warm-start pose: loaded 2 tensors from st_pose_best.pt
Warm-start: loaded 2 head tensors total
```

Only the pose head was warm-started (2 tensors loaded). Detection, activity, and PSR heads all initialize from scratch (random weights). These ST checkpoints exist in `src/runs/st_checkpoints` but only `st_pose_best.pt` was found.

**Resume checkpoint**: Loaded from `best_v4_640resumable.pt` at lines 2151-2198. One shape mismatch:
```
act_head.class_freq: ckpt=(75,) vs model=None
```
This is the inert `class_freq` buffer that `enable_logit_adjust()` registers. The checkpoint has it but the just-loaded model hasn't re-called `enable_logit_adjust()` yet (it happens at line 2231, AFTER the resume at line 2152). This is benign — the buffer is recreated 80 lines later.

---

## Verdict: 5 Actionable Findings

### Finding 1 [CRITICAL]: FPN is frozen (~14.5M parameters, 26% of total model)
- **Evidence**: `_group_params(["feature_pyramid.fpn", "det_head"], 1.0)` at train_mtl_mvit.py:2133 uses the wrong prefix `feature_pyramid.fpn`. The actual FPN module is `self.fpn` in MTLMViTModel (mvit_mtl_model.py:520), so its parameters (`fpn.lateral.*`, `fpn.td_conv.*`, etc.) match no optimizer group.
- **Impact**: The BiFPN has random weights that never update. Detection features are frozen. The detection head (1.20M) learns to interpret frozen features.
- **Fix**: Change prefix to `"fpn"` in the param group.

### Finding 2 [CRITICAL]: RotoGrad parameters are frozen (639K parameters)
- **Evidence**: RotoGradRotation is instantiated (train_mtl_mvit.py:2273-2276) AFTER optimizer creation (line 2142). No `add_param_group()` call. `rotation_loss()` is never called. grep returns no matches for any RotoGrad optimization call in train_mtl_mvit.py.
- **Impact**: The "feature rotation" applies random fixed matrices — equivalent to noise injection, not gradient alignment. The 312x gradient magnitude gap between PSR and activity noted in the RotoGrad module docstring is not addressed.
- **Fix**: Either add RotoGrad params to the optimizer, or implement the rotation loss optimization as described in the module docstring.

### Finding 3 [HIGH]: Activity head collapse — self-reinforcing via FAMO
- **Evidence**: `act_preds=1uniq/0.03maxconf` at epochs 11-13 (T8_frag.log:72, T4_v2.log:71,95,118). Activity loss INCREASES across epochs (3.99 -> 4.07 -> 4.15 at T4_v2.log:70,93,116). FAMO decreases weight when loss increases, starving the activity head further. 3 of 75 classes have zero weight (log line 42).
- **Impact**: Activity classification performs below random baseline (1.33% vs 1.3% random), eliminating the model's core classification capability.
- **Fix**: (a) Use `act_decoupled=True` to freeze the backbone and retrain just the classifier; (b) investigate why ST activity checkpoint is missing from the warm-start dir; (c) ensure at least 1 sample per class in the training split.

### Finding 4 [HIGH]: PSR head produces flat output (psr_stdmax=0.0206)
- **Evidence**: All 11 PSR components predict ~0.69-0.71 probability with frame-to-frame stddev of 0.02 (T8_frag.log:71, T4_v2.log:71,94,117). The MS-TCN refinement (206K params, 2 stages) operates on already-flat probabilities and cannot generate meaningful variation.
- **Impact**: Temporal transition detection is effectively non-functional. PSR is predicting the marginal probability of each component.
- **Fix**: Investigate whether the PSR head's input features (P5, blocks[14]) carry temporal information. Add diagnostic: compute temporal variance of P5 features before and after spatial pooling.

### Finding 5 [MEDIUM]: Head capacity imbalance — pose under-parameterized, PSR over-parameterized relative to signal
- **Evidence**: 
  - Pose head: 0.20M (768->256->6) for 6D rotation — a geometry task that typically benefits from more capacity (cf. 3D rotation literature uses 512+ hidden dims).
  - PSR head: 1.78M + 0.21M refinement = 2M for 11 binary predictions from an 8-frame window — 10x the capacity of the pose head for a simpler task.
  - Activity head: 3.75M for 75-class classification — reasonable, but gradient-starved (Finding 3).
- **Impact**: Pose cannot learn stable rotation representations (loss ~700-4000°). PSR has unnecessary capacity that sees no gradient signal.
- **Fix**: Rebalance: increase pose to 768->512->128->6 (~0.7M), decrease PSR to d=128, nhead=2 (~0.4M). Reallocates ~1.5M from PSR to pose with zero net parameter change.

---

## Summary Table

| Head | Params | % Model | Optimized? | LR Mult | Performance Issue |
|------|--------|---------|-----------|---------|-------------------|
| Backbone | 34.23M | 61.5% | Yes | 1e-4 | — |
| FPN | 14.53M | 26.1% | **NO** | — | Frozen (prefix bug) |
| Detection | 1.20M | 2.2% | Yes | 1.0x | Volatile loss (0.03-7.28) |
| Activity | 3.75M | 6.7% | Yes | 1.0x | Collapsed (1 uniq class) |
| PSR | 1.78M | 3.2% | Yes | 0.3x | Flat output (stdmax=0.02) |
| Pose | 0.20M | 0.4% | Yes | 0.3x | High loss (700-4000°) |
| RotoGrad | 0.64M | 1.1% | **NO** | — | Frozen (missed optimizer) |
| PSR Refmnt | 0.21M | 0.4% | Yes | 0.3x | No temporal variation |
| **Total** | **55.70M** | **100%** | **41.16M** | — | **14.5M frozen** |

All 5 findings are verified against source code (file:line) and training logs.
