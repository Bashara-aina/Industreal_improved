# 80 — Head Pose + Multi-Task Orchestration: Final Gradient Health Verification [2026-07-01]

**Goal:** Guarantee the Kendall uncertainty-weighted multi-task training is stable, no single task dominates or starves, and all tasks receive healthy gradient flow. This is the glue that determines whether the previous 3 files' individual fixes translate to a working joint model.

## Important: Understanding the "Pose" Confusion

The term "pose" in this codebase refers to THREE different things. Keeping them straight is critical:

| Term | What it is | Ground truth source | Has IndustReal benchmark? | What to report |
|------|-----------|-------------------|--------------------------|----------------|
| **Head pose** (ours) | 9-DoF: forward[3] + position[3] + up[3] | Real HL2 sensor (`pose.csv`, 10 FPS) | **No** — novel addition | forward-gaze MAE, position MAE |
| **Body keypoints** | 17 COCO keypoints via PoseHead | **Pseudo** — generated from detection boxes (model.py:1972-1976). No real annotations exist in IndustReal. | **No** — not a real task | Do NOT report as benchmark |
| **Hand-FiLM** | Hand tracking → FiLM modulation on C5 features | From `hands.csv` (HL2) | **No** — feature enhancer | Ablation only (with/without) |

**Fact**: IndustReal (WACV 2024) defines exactly **3 official benchmarks**: Action Recognition (AR), Assembly State Detection (ASD), and Procedure Step Recognition (PSR). Body pose and head pose are not benchmarked — they are raw sensor streams. Our `HeadPoseHead` adds 9-DoF head pose prediction as a **novel multi-task contribution**, establishing the first baseline on this data.

The 4 task groups for Kendall are: **detection, activity, PSR, head pose**. Body pose shares `log_var_pose` with head pose but `loss_pose ≈ 0` always (no real keypoint targets).

**Source files (all paths relative to `code/industreal_improved/src/`):**
- `config.py:40–96, 508–518, 720–894, 1450–1530` — training flags, staged training, Kendall bounds, gradient blend
- `models/model.py:1482–1531, 1830–1865` — HeadPoseHead, HeadPoseFiLM, model assembly
- `training/losses.py:972–1248, 1333–1480` — MultiTaskLoss, Kendall forward, smooth caps
- `training/train.py` — optimizer construction, gradient clipping, stage management
- `data/industreal_dataset.py` — head pose label loading, data augmentation

---

## 1. Kendall Uncertainty Weighting — The Critical Linchpin

### 1.1 How Kendall Works (losses.py:972-1035, forward)

```python
# Multi-task loss with Kendall homoscedastic uncertainty weighting:
# L_total = sum_t [ exp(-s_t) * L_t + s_t ]
# where s_t = log(σ²_t) is the learned log-variance for task t
# precision = exp(-s_t) — higher = more weight
```

The 4 task groups and their log-var initialization:
```python
self.log_var_det = nn.Parameter(torch.zeros(1))   # s=0, precision=1.0
self.log_var_pose = nn.Parameter(torch.tensor([0.0]))  # s=0, precision=1.0 (SHARED: body pose + head pose)
self.log_var_act = nn.Parameter(torch.zeros(1))   # s=0, precision=1.0
self.log_var_psr = nn.Parameter(torch.zeros(1))   # s=0, precision=1.0
```

**Key design choice** (losses.py:1019-1026): `log_var_pose` is shared for BOTH body pose (17-keypoint Wing Loss — pseudo, no real GT) AND head pose (9-DoF MSE — real GT from pose.csv). This is intentional per the paper spec: both are pose tasks of similar magnitude. **However**, since IndustReal has no real body keypoint annotations, `loss_pose ≈ 0` always (the Wing Loss branch at losses.py:1325-1333 finds no `'keypoints'` in targets). The loss code at losses.py:1766-1777 was specifically fixed (2026-06-17) to include `loss_head_pose` even when `loss_pose = 0`, so the head pose contribution is not zeroed by the dead body-pose branch.

### 1.2 Kendall Clamp Range (losses.py:988-991, config.py:854-860)
```python
# Clamp on log_vars: [-4, 2] → precision range [0.135, 54.6]
# Per-task overrides:
KENDALL_LOG_VAR_MIN_ACT = -0.5    # Activity precision can't exceed exp(0.5)=1.65×
KENDALL_LOG_VAR_MAX_PSR = 0.0     # PSR precision capped at 1.0× (can't be suppressed)
KENDALL_LOG_VAR_MAX_POSE = 3.0    # Pose can be suppressed down to exp(-3)=0.05×
```

These bounds prevent any single task from dominating or being starved:
- Activity can't precision-boost more than 1.65× (prevents dominance)
- PSR can't be precision-suppressed below 1.0× (protects from starvation)
- Pose can be suppressed to 0.05× (prevents body-pose from dominating as it did in RF1)

### 1.3 Precision Cap (config.py:74)
```python
KENDALL_HP_PREC_CAP = True  # Head pose precision can never exceed detection precision
```

This is implemented via a post-step clamp: after each optimizer step, if `exp(-s_head_pose) > exp(-s_det)`, then `s_head_pose` is clamped equal to `s_det`. This prevents head pose from dominating the backbone — without this, head pose (loss ≈ 0.01) could get Kendall-optimal precision ~54.6× while detection (loss ≈ 0.5) gets ~1.4×.

### 1.4 Fixed Weights Mode (config.py:78-91)
```python
KENDALL_FIXED_WEIGHTS = False   # Learned Kendall, not fixed
KENDALL_HP_FIXED_LAMBDA = 0.2   # Only used when FIXED_WEIGHTS=True
```

When `KENDALL_FIXED_WEIGHTS=True`, the 4 log_vars are frozen and replaced with fixed λ multipliers. This is useful for detection-bootstrap phases (RF1-RF2) where learned Kendall would let the small head_pose loss dominate. Currently False — learned Kendall is active.

### 1.5 Kendall Staging (config.py:82-86)
```python
KENDALL_STAGED_TRAINING = False  # Double curriculum disabled
```

When True, losses.py had its own epoch-based curriculum (STAGE1_EPOCHS=5, STAGE2_EPOCHS=10) that duplicated the RF stage manager. Setting False makes the loss's staged_training a no-op — the RF stage manager is the sole curriculum. **Critical**: verify `apply_preset()` correctly sets `train_act`, `train_det`, `train_pose`, `train_psr` for each stage.

### 1.6 Log-Var Device Management (losses.py:1206-1210) — SAFE (not a bug)
```python
if self.log_var_det.device != device:
    self.log_var_det.data = self.log_var_det.data.to(device)
    # ... same for log_var_pose, log_var_act, log_var_psr
```

**This is NOT a bug.** The optimizer construction order in train.py ensures all log-var parameters are on GPU before optimizer creation:

1. **train.py:3469**: `criterion = MultiTaskLoss(...).to(device)` — criterion AND all its parameters (including log_vars) moved to GPU inline at construction
2. **train.py:3489**: `loss_params = list(criterion.parameters())` — captured AFTER `.to(device)`, so GPU-based
3. **train.py:3592**: `optimizer = torch.optim.AdamW(param_groups, ...)` — optimizer built with already-on-GPU params

The `.data` guard in losses.py:1206-1210 is **dead code** — it never fires because parameters are already on GPU by the time `forward()` is called. Even if it did fire, `nn.Parameter.data.to(device)` correctly updates the Parameter's storage in-place; the optimizer holds a reference to the `nn.Parameter` object, not a snapshot of its data pointer.

**No fix needed. The log-var parameters learn correctly.**

---

## 2. Gradient Blend and Feature Flow

### 2.1 Activity Gradient Blend (config.py:837-852)
```python
ACTIVITY_GRAD_BLEND_RATIO = 1.00  # Full gradient from activity into backbone
```

The blend ratio controls how much activity gradient flows into `c5_mod_blend` (the backbone's C5 feature after FiLM modulation). At 1.0, 100% of the activity gradient propagates. At 0.0, activity is fully detached. The progression over fixes:
- RF1-RF3: 0.05 → 0.10 → 0.30 → 0.50 → 0.70 → **1.00** (current)

With `ACTIVITY_HEAD_SIMPLE=True` and the in-place assignment bug fixed, there's no longer a gradient path issue. The 1.0 ratio is safe.

### 2.2 Gradient Clipping (config.py:518)
```python
GRAD_CLIP_NORM = 1.0  # Global ℓ₂-norm clip at 1.0
```

Applied after all loss.backward() calls. This is the standard paper value.

### 2.3 Activity-Specific Gradient Clip (config.py:808-811)
```python
ACTIVITY_HEAD_GRAD_CLIP = 1.0  # Was 0.3 — raised when gradient was 0.012 (below clip)
```

With the simple MLP path, activity gradient should be healthy (~0.48). The 1.0 clip is now non-constraining (was raised from 0.3 when gradient was 0.012).

**Verification**: log `act_logits.grad.norm()` and `backbone_layer[-1].weight.grad.norm()` at step 200. Both should be non-zero and within 0.1x-10x of each other.

---

## 3. Head Pose Head (model.py:1482-1531)

### 3.1 Architecture
```python
class HeadPoseHead(nn.Module):
    def __init__(self, c4_channels=384, c5_channels=768, hidden_dim=128):
        # C4 (384) GAP + C5 (768) GAP → concat [1152]
        # Linear(1152 → 512) → LayerNorm → GELU → Dropout(0.15)
        # Linear(512 → 256) → LayerNorm → GELU → Dropout(0.1)
        # Linear(256 → 9)
```

9-DoF output: forward_vector(3) + position(3) + up_vector(3). Trained with MSE loss against raw GT from pose.csv.

### 3.2 HeadPoseFiLM (model.py:1830-1834)
```python
if use_headpose_film:
    self.headpose_film = HeadPoseFiLMModule(
        c5_channels=c5_ch,    # 768
        hidden_channels=256,
    )
```

Second-stage FiLM: the 9-DoF prediction modulates C5_mod features (already modulated by PoseFiLM hand keypoints). This creates a hierarchical conditioning: hand pose → C5_mod → head pose → C5_mod2. Each stage refines features for the next task.

### 3.3 Head Pose Position Scale (config.py:735)
```python
HEAD_POSE_POS_SCALE = 100.0  # Normalizes raw position (~110 in CSV) to O(1)
```

Without this, raw position values (~110 mm) dominate the forward/up vectors (~0.0-1.0) in the MSE loss. The position MAE target of 8.71° (angular) and ~10mm (positional) assumes this scaling is active.

---

## 4. Loss Smooth Caps (config.py:729-734)

All losses use the same smooth-cap formula to prevent NaN cascade while preserving gradient:
```python
def _smooth_cap(x, cap):
    if getattr(C, 'SIMPLIFY_LOSS', False):
        return x
    x_safe = x.clamp(min=1e-6, max=1e6)
    return torch.where(x > cap, cap * (1 + torch.log(x_safe / cap)), x.clamp(min=1e-6))
```

| Loss | Cap | Notes |
|------|-----|-------|
| Detection (cls + GIoU) | 50.0 | GIoU can be large at init |
| Pose (Wing Loss) | 30.0 | Body keypoints |
| Activity (CE + CB weights) | 80.0 | Raised from 40 to allow LDAM losses (~55) |
| PSR (BCE) | 20.0 | Per-component BCE rarely exceeds 10 |
| Head Pose (MSE) | 30.0 | With HEAD_POSE_POS_SCALE=100, raw position ~1.1 → MSE ~1.2 |

---

## 5. LR and Optimizer (config.py:510-514)

```python
BASE_LR = 5e-4
WARMUP_EPOCHS = 2
USE_COSINE_ANNEALING = False  # Uses OneCycleLR instead
WEIGHT_DECAY = 5e-2
```

OneCycleLR schedule with warmup: LR rises from 0 → 5e-4 over 2 epochs, then cosine-anneals to ~0 over the remaining 98 epochs. Weight decay 5e-2 applied only to weight parameters (not bias/norm, per standard practice).

**Per-task LR multipliers** are applied via `train.py` parameter group construction:
- Activity head: `ACTIVITY_LR_MULTIPLIER=1.0` (config.py:812-818)
- Detection head: `DET_LR_MULTIPLIER=1.0` (config.py:55-59)
- PSR head: ramped via `STAGE3_WARMUP_EPOCHS=3` (config.py:887)

---

## 6. Staged Training (config.py:720-727)

```python
STAGED_TRAINING = False  # All heads active from epoch 0
```

The current config has `STAGED_TRAINING=False`, meaning all 5 heads (detection, body pose, head pose, activity, PSR) train simultaneously from epoch 0. The legacy stage definitions (STAGE1/2/3) are unused.

**Risk**: Without staging, detection has no head-start before activity, PSR, and head pose begin competing for backbone features. (Body pose is negligible — `loss_pose ≈ 0` always, no real keypoint GT.) The fixes in files 77-79 (OHEM, DET_GT_FRAME_FRACTION, detached PSR/reg gradients) are designed to work WITHOUT staging — but Opus should verify this was intentional.

**For RF1 (--reinit-heads)**: the `apply_preset()` function in train.py may override to detection-only for the first N epochs. Check if RF1 preset sets `STAGED_TRAINING=True` internally.

---

## 7. Task-Aware Sampling (config.py:771-775)

```python
USE_TASK_AWARE_SAMPLING = True
TASK_AWARE_DET_BOOST = 2.0   # 2× weight for frames with GT boxes
TASK_AWARE_PSR_BOOST = 1.5   # 1.5× weight for frames with PSR labels
```

On top of the class-balanced sampler and the DET_GT_FRAME_FRACTION redistribution, task-aware sampling further upweights frames that carry rare task annotations (GT boxes, PSR labels). The interaction of all three mechanisms:

1. **ACT_SAMPLER_MODE='balanced'** → activity classes get equal sampling mass
2. **DET_GT_FRAME_FRACTION** → redistributes after step 1 so X% of batch is GT-bearing
3. **TASK_AWARE_DET_BOOST** → within the GT-bearing allocation, upweight frames with boxes by 2×

**Order: step 1 → step 2 → step 3**. If all three are active, activity balance is first set, then detection GT fraction overrides it, then per-frame boost further reweights. The end result should be a batch that has both activity class balance AND guaranteed detection GT. Opus should verify the `apply_preset()` logic preserves this ordering.

---

## 8. Gradient Health Monitoring (config.py:50-51)

```python
LIVENESS_EVERY = 500     # Output/norm telemetry
LIVENESS_GRAD_EVERY = 200  # Gradient norm telemetry (more frequent)
```

At training step 0, 200, 400, etc., the training loop logs:
- Per-head loss values (raw + Kendall-weighted)
- Per-head gradient norms (`param.grad.norm()` for each head's parameters)
- Backbone gradient norm
- Kendall log-var values and derived precisions

**Opus should check**: At step 200, ALL 4 heads should have non-zero gradient norms. If any head has grad_norm=0.0, its contribution is dead and the Kendall weight will drift toward suppression.

---

## 9. What Could Still Go Wrong

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| log-var device management (losses.py:1206-1210) — `.data` guard is dead code (params already on GPU via `criterion.to(device)` before optimizer build) | None | `criterion.to(device)` at train.py:3469 moves all params to GPU. The `.data` guard never fires. See §1.6 for full trace. |
| KENDALL_HP_PREC_CAP prevents head pose from contributing at all | Medium | If head pose grad norm is 0.0 at step 200, the precision cap may be too aggressive. Check `exp(-s_head_pose)` and `exp(-s_det)` ratio. |
| Without staged training, detection never gets a head start | Medium | Detection is designed for the fully multi-task setting (OHEM, detach, empty-frame sampling all assume other heads are active). If epoch-5 detection mAP is <0.01, consider 5-epoch detection-only warmup. |
| PSR detached from backbone (DETACH_PSR_FPN=True) means backbone never learns PSR-relevant features | Low | PSR on per-frame data doesn't need backbone gradient. If PSR accuracy plateaus early, consider enabling gradient for PSR mid-training. |
| OneCycleLR at 100 epochs with warmup=2 may not give enough time at high LR | Low | Standard. 5e-4 for ~10 epochs, then cosine decay. If activity/detection converge slowly, increase `WARMUP_EPOCHS` to 5. |
| ACTIVITY_GRAD_BLEND_RATIO=1.0 may cause backbone to drift toward activity at expense of detection | Low | Monitor detection mAP over epochs. If detection starts high then declines, reduce blend to 0.7. |

---

## 10. Final Go/No-Go Criteria (Epoch 2)

| Signal | Pass | Borderline | Fail |
|--------|------|-----------|------|
| All 4 head gradient norms > 0 | Yes | Temporary zero (<10 steps) | Any head has grad_norm=0.0 for >50 consecutive steps |
| Kendall log-var values (all 4) | -0.5 to +2.0 | -1.0 to +3.0 | Any outside [-2, +4] |
| Exp(-s_act) / exp(-s_det) ratio | 0.1x – 10x | — | > 10x (activity dominating) |
| Detection mAP50_probe | ≥ 0.005 | 0.001-0.005 | 0.0 (death spiraling) |
| Activity pred_distinct | ≥ 10 groups | 5-9 groups | < 5 groups |
| PSR loss (raw, before Kendall) | 0.2-1.5 | 0.1-2.0 | > 5.0 or NaN |
| Head pose MAE (angular) | < 40° | 40-60° | > 60° |
| Forward-gaze MAE (headline) | < 15° | 15-25° | > 25° |

**Epoch-100 target ranges** (paper headline metrics):
- **Detection mAP50**: 0.50–0.65
- **Activity clip-accuracy (grouped)**: 0.40–0.60
- **Activity macro-F1 (grouped)**: 0.30–0.50
- **PSR mean binary accuracy**: 0.75–0.85
- **Head pose forward-gaze MAE**: 8.71° (or better) — our own target (no IndustReal baseline exists for head pose)
- **Head pose position MAE**: < 20mm

**If multiple heads fail simultaneously**: the issue is likely a gradient flow problem in the shared backbone (not the log-var device — §1.6 confirmed safe). Run one diagnostic epoch with `SIMPLIFY_LOSS=True` and `ASSERT_AND_CRASH=True` to surface any NaN/inf issues early.
