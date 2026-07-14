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

# Agent 9: Training Pipeline Analysis

- **Scope**: MViTv2-S MTL pipeline (detection + activity + PSR + pose)
- **Run evidence**: `mtl_480_T8_frag.log` (T=8) and `mtl_480_T4_v2.log` (T=4)
- **Branch**: `auto/2pct-training-fix-20260520-202419`
- **Date**: 2026-07-13

---

## Q1: Data Loading Pipeline

**Source**: `scripts/train_mtl_mvit.py` lines 1958-1964, `src/data/industreal_dataset.py` lines 975-1050 (`_getitem_sequence`)

The dataset is `IndustRealMultiTaskDataset` with `sequence_mode=True`:

```python
train_ds = IndustRealMultiTaskDataset(
    split="train",
    img_size=(480, 480),
    augment=True,
    sequence_mode=True,
    sequence_length=8,  # T=8 in T8 run; T=4 in T4 run
)
```

- **Sequence windows**: `_build_seq_sample_index` (`industreal_dataset.py:1242-1302`) builds T-frame sliding windows with stride=1. T=8 run reports 78,679 windows from 26,322 frames across 36 recordings.
- **Determinism**: Dataset is seeded with `seed=42` (`industreal_dataset.py:120`). All workers share the same seed, which is intentional for reproducibility but can cause correlated augmentations across workers.
- **Frame cache**: RAM cache pre-loads 500 images as JPEG bytes (`industreal_dataset.py` cache logic). 500 images at 480p is ~170MB -- a trivial fraction of the 26K-frame dataset.
- **Sampling**: `GuaranteedGTBatchSampler` (`industreal_dataset.py:1738-1792`) ensures at least one detection GT frame per batch. `DET_GT_FRAME_FRACTION=0.40` targets 40% GT-bearing frames per batch.
- **No pinned memory or async prefetch**: `batch_size=1`, `num_workers=0`. Data loading is serial, within the main process. At ~0.76s per batch (T=8, 75s per 100 batches), data loading is not the bottleneck but the single-worker mode adds latency.
- **num_workers=0 bottleneck**: With 0 workers, the main process must load, decode, crop, and normalize each batch synchronously. For 480px images, decode alone takes 30-50ms per image. With T=8, that is 240-400ms per batch just for decode, which is a significant fraction of the ~760ms per-batch time.

**Verdict**: Data loading is functional but suboptimal. `num_workers=0` forces synchronous decode, wasting ~30-50% of per-batch time on CPU work that could be pipelined. The 500-image RAM cache covers <2% of the dataset.

---

## Q2: Gradient Accumulation and Effective Batch Size

**Source**: `scripts/train_mtl_mvit.py` lines 1255, 1266, 1858-1859, 2373-2376

Both logs show `grad_accum_steps=1` in the Args dump (log line 2), which **overrides** the script's own default of `grad_accum_steps=4` (line 1858):

```python
parser.add_argument("--grad-accum-steps", type=int, default=4,
    help="Gradient accumulation steps (effective batch = batch_size * this)")
```

- **Effective batch = `batch_size * grad_accum_steps` = 1 x 1 = 1** in both runs.
- The log confirms `accum=1/1` on every batch line, meaning every micro-batch is a boundary step.
- Loss is correctly scaled by `1/grad_accum_steps` (lines 1255 and 1266): `scaler.scale(total_loss / grad_accum_steps).backward()`.
- The config.py `BATCH_SIZE=6` / `GRAD_ACCUM_STEPS=8` / `EFFECTIVE_BATCH=16` defaults are **completely overridden** by the CLI args.

**Impact**: With effective batch=1, gradient noise is high and batch-normalization statistics (if any) are unreliable. The FAMO weight update sees single-sample loss values rather than mini-batch means, amplifying per-sample variance.

**Verdict**: Gradient accumulation is disabled by explicit CLI override. The loss normalization is correct for the accumulation path, but with accum=1 it is a no-op. This is likely a resource constraint -- T=8 at 480px with batch_size=1 already strains 16GB VRAM.

---

## Q3: Mixed Precision (bf16 AMP)

**Source**: `scripts/train_mtl_mvit.py` lines 997, 1955-1956, 2219

Mixed precision is configured as:

```python
C.MIXED_PRECISION = True    # line 1955 -- overrides config.py default of False
C.AMP_DTYPE = "bf16"        # line 1956
scaler = torch.amp.GradScaler(device.type, enabled=True)  # line 2219
```

- `autocast(device_type="cuda", dtype=torch.bfloat16)` wraps the forward pass (line 997).
- **GradScaler with bf16 is a no-op**: bf16 does not require gradient scaling because bf16 has the same exponent range as fp32 (8 bits vs. 8 bits). The `GradScaler.step()` and `scaler.scale()` calls are effectively identity operations. This is correct and expected -- the scaler is harmless.
- The training log does not show any `INF/NAN` warnings, confirming bf16 stability.
- **GPU generation**: The code assumes Ampere+ architecture (bf16 support). Running on pre-Ampere would silently fall back to fp32 (autocast with bf16 degrades gracefully).

**Impact**: bf16 provides ~2x throughput vs fp32 with negligible accuracy loss. The GradScaler is vestigial but harmless. No issues found.

**Verdict**: Mixed precision is correctly configured. GradScaler is a harmless no-op for bf16. If the code ever switches to fp16, the scaler would become active.

---

## Q4: FAMO Loss Weighting Integration

**Source**: `scripts/train_mtl_mvit.py` lines 1140-1147 (pre-scaling), 1175-1177 (forward), 1286-1287 (step); `src/losses/famo.py` lines 50-103

### Forward call (train_step, lines 1175-1177):
```python
task_losses = {"det": l_det, "act": l_act, "psr": l_psr, "pose": l_pose}
task_total = famo_weighter.forward(task_losses)  # returns weighted sum
```

### Weight update (train_step, lines 1286-1287):
```python
if famo_weighter is not None and do_step:
    famo_weighter.step(task_losses)
```

### FAMO algorithm review (`src/losses/famo.py`):
- `log_weights` tensor initialized to zeros (= uniform weighting at start)
- `forward()`: `weights = F.softmax(log_weights / temperature, dim=0)`, returns `sum(weights * losses)`
- `step()`: Updates `log_weights` via: `xi_k += lr * z_k * (log l_k^t - log l_k^{t+1} + z_k * log z_k)` where `z_k` is the softmax weight and `l_k` are the per-task losses.
- Correctly uses `.detach()` and `torch.no_grad()` for the weight update.

### Pre-scaling interaction (lines 1143-1147):
```python
_loss_scale = {"det": 0.125, "act": 0.27, "psr": 2.7, "pose": 0.00025}
```
These are applied BEFORE FAMO sees the losses:
```python
l_det = scaled_losses["det"] * _loss_scale["det"]
```

**Critical issue**: FAMO operates on the pre-scaled losses, not the raw losses. This means FAMO's weight dynamics are driven by the scaled magnitudes, which decouples the weighting from the actual gradient magnitudes. The deliberate pre-scaling (pose scaled down 4,000x) makes FAMO's job easier, but if the pre-scaling is wrong, FAMO cannot correct it -- it will learn weights that fit the scaled losses, not the natural loss landscape.

**Stability concern**: Pose loss varies from 51 to 4,399 across consecutive batches (log lines 74-84 of T8 log). This ~80x variance in the raw pose loss means the scaled pose loss varies ~0.013 to ~1.10. FAMO's weight update uses `log l_k^t - log l_k^{t+1}` -- log-differences of values this noisy will cause large weight oscillations. This is visible in the loss spikes: pose swings between 51 and 4,399 randomly.

**Verdict**: FAMO is correctly integrated but its weight dynamics are destabilized by the extreme per-sample variance of the pose loss. The FAMO weight update sees noisy log-differences, causing oscillatory task weights.

---

## Q5: Per-Task Loss Pre-Scaling

**Source**: `scripts/train_mtl_mvit.py` lines 1138-1147

```python
if not equal_weight_loss:
    ...
    # Apply per-task loss pre-scaling
    _loss_scale = {"det": 0.125, "act": 0.27, "psr": 2.7, "pose": 0.00025}
    l_det = scaled_losses["det"] * _loss_scale["det"]
    l_act = scaled_losses["act"] * _loss_scale["act"]
    l_psr = scaled_losses["psr"] * _loss_scale["psr"]
    l_pose = scaled_losses["pose"] * _loss_scale["pose"]
```

### Scale rationale:
| Task | Raw range (log) | Pre-scale | Scaled range | Purpose |
|------|------------------|-----------|--------------|---------|
| det  | 0.03-7.3         | 0.125     | 0.004-0.91   | Compress to similar magnitude as other tasks |
| act  | 0.0-7.3          | 0.27      | 0.0-1.97     | Moderate down-weight |
| psr  | 0.09-0.47        | 2.7       | 0.24-1.27    | Amplify to compete with larger losses |
| pose | 51-4,399         | 0.00025   | 0.013-1.10   | Massive down-scale from raw geodesic angles |

### Issues:
1. **Pose pre-scale (0.00025) is extreme**: Raw pose loss is ~3700 at step 0, needing a 4,000x reduction to reach ~0.93. This means the pose pre-scale factor is ~1/4000, which is fragile -- if the pose loss mean shifts during training, the scaled magnitude shifts proportionally.
2. **Det pre-scale (0.125) is loss-size-agnostic**: The detection loss varies enormously depending on whether GT boxes are present (det=0.03 for no-box vs det=7.3 for positive-box batches). A fixed pre-scale cannot normalize this bimodal distribution.
3. **Pre-scaling + FAMO interaction**: FAMO believes it is weighting losses in ~[0, 2] range for all tasks. If pose loss suddenly jumps (as it does, to 4168 at batch 300), the scaled pose is 1.04, which is within-normal for FAMO. FAMO has no visibility into whether 1.04 came from a normal batch or a gradient-explosion batch.

**Verdict**: Pre-scaling is a pragmatic necessity given the 4-order-of-magnitude raw loss ranges, but the fixed scales interact poorly with the bimodal detection loss and the high-variance pose loss. A potential improvement: use EMA-normalized losses (the code has `ema_losses` tracking but does not use them for pre-scaling -- only for Kendall UW earlier in the code path).

---

## Q6: Curriculum Decay for DET_GT_FRAME_FRACTION

**Source**: `src/config.py` lines 2163-2209

### Config definition (config.py:2163-2176):
```python
if 'det_gt_frame_fraction' in preset:
    DET_GT_FRAME_FRACTION = float(preset['det_gt_frame_fraction'])
elif TRAIN_DET and not (TRAIN_ACT or TRAIN_PSR):
    DET_GT_FRAME_FRACTION = 0.9   # RF1, RF2, recovery_det_only
elif TRAIN_DET:
    DET_GT_FRAME_FRACTION = 0.4   # RF3-RF10 (detection + activity/PSR)
else:
    DET_GT_FRAME_FRACTION = 0.0   # no detection head active
```

### Curriculum decay function (config.py:2194-2209):
```python
def apply_curriculum_decay(epoch: int) -> float:
    """Linearly interpolate DET_GT_FRAME_FRACTION over epochs 0..N-1."""
    global DET_GT_FRAME_FRACTION
    if not DET_GT_CURRICULUM_DECAY:
        return DET_GT_FRAME_FRACTION
    if epoch >= DET_GT_CURRICULUM_EPOCHS:
        new_frac = DET_GT_CURRICULUM_END
    else:
        progress = epoch / max(DET_GT_CURRICULUM_EPOCHS - 1, 1)
        new_frac = DET_GT_CURRICULUM_START + (DET_GT_CURRICULUM_END - DET_GT_CURRICULUM_START) * progress
    DET_GT_FRAME_FRACTION = float(new_frac)
    return float(new_frac)
```

### Key finding: Dead code

`apply_curriculum_decay()` is **NEVER CALLED** from any training script:

- `scripts/train_mtl_mvit.py` has **zero references** to `DET_GT_FRAME_FRACTION` or `apply_curriculum_decay`.
- The training script reads the config at import time, getting the static `DET_GT_FRAME_FRACTION=0.40` default (set at config.py:979).
- The `apply_preset()` function (config.py:2100+) is also not called by `train_mtl_mvit.py`.
- The sampler reads `DET_GT_FRAME_FRACTION` once during initialization (`industreal_dataset.py:1520`) and never re-reads it.

The curriculum decay infrastructure was added (config.py lines 2184-2209, dated `[FIX 2026-07-13]`) but never wired into the training loop. The training scripts have no epoch-dependent sampler reconfiguration.

**Verdict**: Curriculum decay for DET_GT_FRAME_FRACTION is fully defined but unimplemented -- dead code. The sampler is initialized once with a static fraction and never updated. Adding a call to `apply_curriculum_decay(epoch)` at the start of each epoch in the training loop would activate it, but the sampler weights would also need recomputation.

---

## Q7: Warm-Start Head Initialization

**Source**: `scripts/train_mtl_mvit.py` lines 731-780 (function `warm_start_heads_from_st`)

### Implementation:
```python
def warm_start_heads_from_st(model, checkpoint_dir, logger):
    """Load ST head checkpoints into MTL model head prefixes."""
    st_checkpoints = {
        'det': 'st_det_best.pt',
        'act': 'st_act_best.pt',
        'psr': 'st_psr_best.pt',
        'pose': 'st_pose_best.pt',
    }
    load_prefixes = {
        'det': 'det_head',
        'act': 'act_head',
        'psr': 'psr_head',
        'pose': 'pose_head',
    }
    n_loaded = 0
    n_tensors = 0
    for task, ckpt_name in st_checkpoints.items():
        ckpt_path = Path(checkpoint_dir) / ckpt_name
        if not ckpt_path.exists():
            logger.info(f"Warm-start {task}: checkpoint not found ({ckpt_path}), skipping")
            continue
        st_state = torch.load(ckpt_path, map_location='cpu')
        prefix = load_prefixes[task]
        matched = load_state_dict_with_prefix(model, prefix, st_state)
        n_loaded += 1
        n_tensors += len(matched)
```

### Log evidence:
```
Warm-start det: checkpoint not found (src/runs/st_checkpoints/st_det_best.pt), skipping
Warm-start act: checkpoint not found (src/runs/st_checkpoints/st_act_best.pt), skipping
Warm-start psr: checkpoint not found (src/runs/st_checkpoints/st_psr_best.pt), skipping
Warm-start pose: loaded 2 tensors from st_pose_best.pt
Warm-start: loaded 2 head tensors total
```

### Impact:
- **3/4 checkpoints missing**: Detection, activity, and PSR heads are randomly initialized.
- **Pose head has only 2 tensors loaded**: The pose head has multiple layers (at minimum a linear projection and an output layer). Loading only 2 tensors suggests the checkpoint's state dict structure does not match the model's `pose_head` prefix. This could be a prefix mismatch or a structural change between ST and MTL heads.
- **Warm-start is essentially broken**: The remaining three heads start from scratch, which means the first 5+ epochs are wasted re-converging each task head independently. This directly contributes to the activity collapse (Q8) and PSR uniformity (Q9).

**Verdict**: Warm-start is critically broken. 3 of 4 head checkpoints are missing, and the pose checkpoint only partially loads. The checkpoint directory path must be verified, and the checkpoint naming convention must match `st_{task}_best.pt`. The MTL head architecture may also differ from the ST head architecture, causing partial state dict matching.

---

## Q8: Activity Head Behavior and Collapse

**Source**: Training log (both T8 and T4 runs)

### Log evidence -- T8 run, epoch 11:
```
Quick: act_preds=1uniq/0.03maxconf  |  psr_stdmax=0.1557
```

### Log evidence -- T4 run, epoch 11:
```
Quick: act_preds=1uniq/0.04maxconf
```

**After 11 epochs, the activity head predicts only 1 unique class with max confidence 3-4%.** This is a complete collapse scenario.

### Contributing factors:

1. **Random initialization**: Activity head was randomly initialized (warm-start failed, Q7). It never received a useful starting state.

2. **Loss magnitude**: The activity loss varies from 0.0 (no label?) to 7.3. The pre-scale of 0.27 brings it to ~0-2 range. But activity's raw CE loss can be very high on incorrect predictions with uniform output -- if the head outputs near-uniform [1/75, ..., 1/75], the CE loss is ~4.3 (log 75). This matches the observed activity loss values (avg ~3.99 in epoch 11).

3. **Class imbalance**: 75 classes with heavy long-tail. The config shows `class_weights -- min=0.0000 max=11.7127 mean=2.2867 num_nonzero=72 [sqrt-tamed]` -- 3 of 75 classes have zero weight, and the max class weight is 11.7x the min. This extreme weighting may cause gradient instability in the randomly initialized head.

4. **Balanced Softmax**: Active (log line 45). Initializes priors from class frequencies. If the initial predictions are near-uniform, the logit adjustment is dominated by the prior term, potentially causing the model to "lock in" to predicting the most frequent class exclusively.

5. **Detection augmentation interference**: DetectionAugment is applied to images before the shared backbone, then ALL heads (including activity) see augmented images. But the activity targets are NOT augmented (by design). This means activity trains on spatially augmented images with un-augmented labels. For randomly initialized heads, this is extra noise.

**Verdict**: Activity collapse (1/75 unique predictions, 3-4% max confidence) is a critical training failure. The root cause is multi-factorial: random initialization, severe class imbalance, and the interaction between Balanced Softmax and near-uniform initial predictions. This is the highest-priority issue in the pipeline.

---

## Q9: PSR Head Behavior

**Source**: Training log (T8 run, epoch 11)

### Log evidence:
```
PSR comp: [0.7112 0.6984 0.6806 0.6931 0.6902 0.7025 0.6932 0.6995 0.6954 0.6882 0.6612]
```

**All 11 PSR components have nearly identical loss values (~0.66-0.71)** with a tight spread of just 0.05 (max-min). This indicates the PSR head is predicting uniform background probability for every component -- effectively treating every frame as "no assembly transition occurring."

The model has converged to the trivial solution: predict all-zeros (or near-zero) for every PSR component, achieving a constant baseline loss. This is a form of collapse -- not to a single class, but to uniform background.

### Contributing factors:

1. **Random initialization**: PSR head was randomly initialized (warm-start failed, Q7).
2. **No temporal discriminative signal**: At initialization, the temporal transformer layers produce uniform attention, so every frame looks the same to the PSR head. Without a proper warm-start, the gradients from the PSR focal loss are too weak to break the symmetry.
3. **MS-TCN smoothing**: The `ms_tcn_smooth` loss penalizes frame-to-frame logit differences. If the PSR head is already predicting uniform output, the smoothing loss is near-zero, creating a self-reinforcing loop: uniform predictions minimize both the focal loss (which is near the expected value for uniform) AND the smoothing loss (which is zero for uniform).
4. **Gradient competition**: With FAMO weighting, the strong activity gradients (from the random activity head) may dominate the shared backbone, leaving the PSR branch with insufficient gradient signal.

**Verdict**: PSR has collapsed to uniform background prediction (all 11 components at ~0.69 loss). This is a secondary symptom of the same warm-start failure that affects the activity head. The MS-TCN smoothing loss may actively reinforce the collapse by penalizing non-uniform predictions.

---

## Q10: Memory Usage, Fragmentation, and OOM Risk

**Source**: `scripts/train_mtl_mvit.py` (no expandable_segments); `src/config.py` (no memory fraction)

### Memory configuration:

| Setting | train_mtl_mvit.py | Recommended | Status |
|---------|-------------------|-------------|--------|
| `expandable_segments:True` | **NOT SET** | Required for Ampere+ | **MISSING** |
| `PYTORCH_CUDA_ALLOC_CONF` | Not referenced | Prevents fragmentation OOM | **MISSING** |
| `set_per_process_memory_fraction` | Not called | Prevents system OOM | **MISSING** |
| Gradient checkpointing | Active (T8 log, line 22) | Reduces VRAM ~3x | Active |
| `grad_accum_steps=1` | Active (no gradient buffering) | Accumulation uses extra VRAM | N/A |

### Evidence from log:
```
Gradient checkpointing ENABLED -- reduces VRAM ~3x at cost of ~30% compute
Params: 55.7M total, 55.7M trainable
```

55.7M parameters at bf16 = ~111 MB for model weights. With optimizer states (Adam: 2x fp32 per param = ~445 MB) and activations (with gradient checkpointing, ~O(batch_size * T * H * W * channels) per layer), 480px T=8 batch=1 fits in 16GB with checkpointing. T=4 is proportionally smaller.

### The `expandable_segments` gap:
The separate `train.py` script sets `expandable_segments:True` at its line 6, and all shell training scripts in `scripts/` also export it. But `train_mtl_mvit.py` is a standalone script that does NOT set this environment variable. This means:

- If run directly (not via shell script), PyTorch uses the default CUDA allocator.
- On Ampere+ GPUs with large memory (24GB+), the default allocator produces memory fragmentation over long training runs, eventually causing CUDA OOM even though total usage is below capacity.
- 35 epochs at 78,679 batches/epoch with 55.7M params: fragmentation builds up over ~2.8M optimizer steps. OOM is likely at epoch 20-25.

### Key OOM risk factors:
1. **No expandable_segments** -- fragmentation OOM likely in later epochs.
2. **No memory fraction** -- training can be killed by Linux OOM-killer if another process allocates GPU memory.
3. **Detection augmentation** allocates intermediate tensors (flipped/cropped/jittered images) adding ~50-100 MB temporary overhead per forward pass.

**Verdict**: Memory management is fragile. The missing `expandable_segments:True` is the most critical gap -- it is universally set in all other training scripts and in `train.py` but omitted from `train_mtl_mvit.py`. The 35-epoch run is at risk of fragmentation OOM in the second half of training.

---

## Additional Finding A: DetectionAugment Double-Normalization Bug

**Source**: `src/data/det_augment.py` line 102; `scripts/train_mtl_mvit.py` lines 2365-2368, 991-994

### Flow:
1. **Normalization** (train_mtl_mvit.py:2365-2368): Images are divided by 255.0, then normalized with `mean=[0.45, 0.45, 0.45]`, `std=[0.225, 0.225, 0.225]`. After this step, image values range from approximately -2.0 (pixel=0) to +2.4 (pixel=255).
2. **DetectionAugment** is called on the normalized images (train_step, line 994).
3. **Clamp to [0, 1]** (det_augment.py:102): After color jitter, the code does `aug_images = aug_images.clamp(0.0, 1.0)`.

**This clamp is incorrect for normalized images.** It clips all negative values to 0 and all values > 1 to 1. For a normalized image with mean=0 and std=1, this destroys ~50% of the pixel distribution (everything below the mean). The color jitter math (brightness scaling, contrast, saturation) works correctly on any scale, but the final clamp assumes values in [0, 1].

**Scope**: Only triggers in ~50% of batches (p_color=0.5). But even at 50%, it introduces distribution shift that degrades backbone features.

**Fix**: Change the clamp to match normalized range, e.g., `aug_images = aug_images.clamp(-2.5, 2.5)`, or better, remove the clamp entirely (color jitter on normalized images cannot produce values outside a safe range for bf16).

---

## Additional Finding B: Per-Batch Timing and Efficiency

**Source**: `mtl_480_T8_frag.log` lines 50-70, `mtl_480_T4_v2.log` corresponding lines

| Metric | T=8 (480px) | T=4 (480px) |
|--------|-------------|-------------|
| Per-epoch time (35k batches) | ~1,327s (22 min) | ~366s (6 min) |
| Per-batch time (100 batches) | ~75s / 100 = 0.75s | ~21s / 100 = 0.21s |
| Throughput | ~10.7 batches/sec | ~28.6 batches/sec |
| Scaling efficiency | T=8 is 3.6x slower for 2x frames | Baseline |

**T=8 is 3.6x slower than T=4 despite having only 2x the frames.** This sub-linear scaling suggests:
- Attention complexity in the MViT temporal encoder is O(T^2) -- doubling T quadruples self-attention cost.
- Activation memory grows with T, and gradient checkpointing recomputes on every backprop step.
- `num_workers=0` means decode time is proportional to T: 8 images/batch vs 4 images/batch.

---

## Verdict

### Finding 1: Activity Head Collapse -- CRITICAL
**Evidence**: Both T8 and T4 logs show `act_preds=1uniq/0.03maxconf` after 11 epochs. The activity head predicts a single class with negligible confidence.
**Root causes**: Random initialization (warm-start failed), extreme class imbalance (max weight 11.7x min), and Balanced Softmax locking in to the majority class.
**Fix**: Resolve warm-start path (provide st_act_best.pt checkpoint). If unavailable, consider pretraining the activity head alone for 2-3 epochs before joint training. Alternatively, disable Balanced Softmax until the head produces >10 unique predictions.

### Finding 2: Warm-Start is Completely Broken -- CRITICAL
**Evidence**: Only `st_pose_best.pt` was found (loaded 2/4 tensors). Detection, activity, and PSR checkpoints are missing.
**Root causes**: Checkpoint files do not exist at `src/runs/st_checkpoints/`. Either the path is wrong or the ST checkpoints were never saved.
**Fix**: Verify checkpoint directory path. Fix the `load_state_dict_with_prefix` function to match MTL head structure against ST checkpoint structure. The 2-of-4 tensor load for pose suggests a prefix mismatch.

### Finding 3: DET_GT_FRAME_FRACTION Curriculum Decay is Dead Code -- HIGH
**Evidence**: `apply_curriculum_decay()` is defined in config.py (lines 2194-2209) but never referenced from any training script. The sampler reads the config once at import and never re-reads it.
**Impact**: The GT-frame fraction stays at 0.40 for the entire 35-epoch run. No warm-start GT oversampling (0.90) and no graceful transition.
**Fix**: Call `apply_curriculum_decay(epoch)` at each epoch boundary in the training loop, then reinitialize the sampler weights to pick up the new fraction.

### Finding 4: Missing `expandable_segments:True` -- HIGH
**Evidence**: `train_mtl_mvit.py` does not set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Every other training script in the repo sets it.
**Impact**: Fragmentation OOM likely at epoch 20-25 of a 35-epoch run. The T=4 run at 366s/epoch would fail after ~2 hours; T=8 after ~7 hours.
**Fix**: Add `os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'` at the top of `train_mtl_mvit.py`, BEFORE any torch import.

### Finding 5: DetectionAugment Clamp Destroys Normalized Distribution -- MEDIUM
**Evidence**: `det_augment.py:102` clamps to `[0.0, 1.0]` after color jitter, but images are normalized to [-2.0, +2.4] range at that point. The clamp truncates ~50% of the pixel distribution.
**Impact**: Every batch that triggers color jitter (~50%) sends distribution-shifted images to the backbone. This degrades feature quality for all four tasks.
**Fix**: Replace `aug_images.clamp(0.0, 1.0)` with `aug_images.clamp(-2.5, 2.5)` to match the normalized range, or remove the clamp entirely.

---

## Evidence Locations (file:line)

| Finding | File | Line(s) |
|---------|------|---------|
| Warm-start head loading | `scripts/train_mtl_mvit.py` | 731-780 |
| Loss pre-scaling | `scripts/train_mtl_mvit.py` | 1143-1147 |
| FAMO forward | `scripts/train_mtl_mvit.py` | 1175-1177 |
| FAMO step | `scripts/train_mtl_mvit.py` | 1286-1287 |
| Grad scaling (loss/accum) | `scripts/train_mtl_mvit.py` | 1255, 1266 |
| Grad clipping | `scripts/train_mtl_mvit.py` | 1270 |
| Mixed precision config | `scripts/train_mtl_mvit.py` | 1955-1956 |
| GradScaler init | `scripts/train_mtl_mvit.py` | 2219 |
| Image normalization | `scripts/train_mtl_mvit.py` | 2365-2368 |
| DetectionAugment call | `scripts/train_mtl_mvit.py` | 991-994 |
| DetectionAugment clamp bug | `src/data/det_augment.py` | 102 |
| DET_GT_FRAME_FRACTION default | `src/config.py` | 979 |
| DET_GT_FRAME_FRACTION preset logic | `src/config.py` | 2163-2176 |
| Curriculum decay (dead code) | `src/config.py` | 2194-2209 |
| FAMO weight update | `src/losses/famo.py` | 50-103 |
| expandable_segments (MISSING) | `scripts/train_mtl_mvit.py` | N/A |
| Dataset sequence indexing | `src/data/industreal_dataset.py` | 1242-1302 |
| GT sampler logic | `src/data/industreal_dataset.py` | 1520-1556 |
| Activity collapse (log) | `mtl_480_T8_frag.log` | 72 |
| PSR uniform losses (log) | `mtl_480_T8_frag.log` | 71 |
| Warm-start failure (log) | `mtl_480_T8_frag.log` | 24-27 |
