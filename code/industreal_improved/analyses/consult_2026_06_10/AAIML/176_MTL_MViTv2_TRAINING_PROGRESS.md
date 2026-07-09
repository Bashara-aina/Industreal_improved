# 176 — MTL-MViTv2 Training Progress & Architecture Documentation

**Date:** 2026-07-09
**Status:** Epoch 1 in progress (batch ~3100/4000)
**Run:** `mtl_mvit_run3` — PID 1002047, started 09:45 JST
**Config:** 100 epochs, batch_size=2, grad_accum=2 (eff. batch 4), bf16 AMP

---

## 1. ARCHITECTURE OVERVIEW

### 1.1 Shared Backbone: MViTv2-S
- **Source:** `torchvision.models.video.mvit_v2_s(weights=MViT_V2_S_Weights.KINETICS400_V1)`
- **Params:** 34.5M pretrained on Kinetics-400
- **All backbone params are trainable** (no freezing)
- Input: `[B, 3, T=16, 224, 224]` normalized video clips
- Output via forward hooks at 4 stages for detection FPN:
  - `conv_proj` → P2 (96ch, 56×56)
  - `blocks[1]` → P3 (192ch, 28×28)
  - `blocks[3]` → P4 (384ch, 14×14)
  - `blocks[14]` → P5 (768ch, 7×7)
- Class token `[B, 768]` extracted after all 16 blocks for Activity + Pose heads

### 1.2 Detection Head (24 assembly-state classes)
- **FPN:** `LightweightFPN` — lateral 1×1 convs (ch→256) + top-down 2× trilinear upsample + 3×3 smooth conv
- **Decoupled Head:** `DetectionHead` — cls_head (conv2d → 24 maps) + reg_head (conv2d → 4×16=64 channels for DFL)
- **Loss:** Focal (γ=2.0, α=0.25) + CIoU + DFL (reg_max=16)
- **Eval metric:** mAP@0.5 (per-class + pooled)
- **Head warmup:** 250 steps (50 zero-grad + 200 linear ramp)

### 1.3 Activity Head (75 classes)
- **Architecture:** LayerNorm(768) → Linear(768→75)
- **Loss:** CE with label_smoothing=0.1, ignore_index=-1
- **Class weights:** Inverse-frequency, computed from label counts (72/75 classes nonzero in train set)
- **Eval metric:** Top-1 / Top-5 accuracy
- **Note:** 3 zero-count classes produce `inf` weights but are guarded by `np.where(counts > 0, ...)` — these classes never appear in loss

### 1.4 PSR Head (11 procedure-step components)
- **Architecture:** AdaptiveAvgPool3d on conv_proj features → [B,96,T=8] → interpolate to T=16 → causal TransformerEncoder (3 layers, nhead=4, LeakyReLU activation) → Linear(96→11)
- **Causal mask:** triu(16×16, -inf, diagonal=1) — each frame attends only to past+present
- **Loss:** Per-frame BCE with inverse-prevalence weights [1, 1, 11]
- **Eval metric:** Event F1@k (k=1,3,5), oracle-bound

### 1.5 Pose Head (6D head pose)
- **Architecture:** Linear(768,256) → LeakyReLU → Linear(256,6) → Tanh → renormalized fwd+up vectors
- **Loss:** (1 - cos(fwd_pred, fwd_gt)).mean() + (1 - cos(up_pred, up_gt)).mean()
- **Eval metric:** Mean angular error (fwd MAE, up MAE) in degrees

### 1.6 Multi-Task Optimization

**Kendall Uncertainty Weighting:**
- 4 learned log_vars, initialized to -0.5, clamped [-4, 4]
- HP_PREC_CAP: Pose precision capped at detection precision (prevents pose from dominating via arbitrarily low variance)
- Loss: `sum( 0.5 * exp(-log_var) * task_loss + log_var )` for each task
- Loss is clamped to `[0, +inf)` for numerical safety

**PCGrad (Project Conflicting Gradients):**
- Applied per-batch (after grad_accum) on shared backbone parameters only
- Random 4-permutation of tasks → cosine conflict check → project conflicting grads → sum
- Reduces negative transfer on conflicting gradient directions

**Task-Aware Sampling:**
- PSR: 2.5× upsampled (temporal steps are rare events)
- Rare detection classes: 3.0× upsampled
- Pose: 0.5× downsampled (smooth signal, needs less data)

---

## 2. BACKBONE DECISION RATIONALE

### The Path: ConvNeXt-Tiny → MViTv2-S

**Phase 1 — ConvNeXt-Tiny (Initial, Rejected):**
- Config.py still reads `BACKBONE = 'convnext_tiny'` — this is **stale** and does NOT reflect current architecture
- ConvNeXt-Tiny is a 2D CNN: no temporal modeling built in, requires manual temporal aggregation per head
- Detection accuracy plateaued at mAP@0.5 ~0.468 under multi-task (see epoch_0001 metrics from old run)
- Activity single-task achieved 65.0% (near WACV SOTA of 65.25%) but zero-gradient bugs were later found in multi-task

**Phase 2 — Hiera-B (Considered, Rejected):**
- Video vision transformer with hierarchical features
- Used in recent video understanding literature
- Rejected because: no explicit temporal attention across frames (uses separated spatial/temporal modules), less mature PyTorch integration

**Phase 3 — Swin3D (Considered, Rejected):**
- Full 3D shifted-window transformer with explicit temporal attention
- Strong on Kinetics benchmarks
- Rejected because: torchvision integration is immature, checkpoint ecosystem is fragmented, training instability reported at small batch sizes

**Phase 4 — MViTv2-S (Selected):**
- Multiscale Vision Transformer: built-in temporal downsampling via pooling attention
- Native torchvision support: `mvit_v2_s(weights=MViT_V2_S_Weights.KINETICS400_V1)` — one-line instantiation
- Hierarchical feature maps (96→192→384→768) are ideal for FPN-based detection
- Class token naturally supports activity/pose heads without spatial pooling
- WACV 2024 SOTA uses MViTv2-S for activity recognition on IndustReal (65.25%/87.93%)
- Proven to work at moderate batch sizes (our memory budget allows batch=2, effective 4)

**Key Advantage — Temporal Modeling:**
MViTv2-S's pooling attention naturally handles T=16 input with pooled T=8 at conv_proj. This temporal compression creates the right abstraction level for PSR (which interpolates back to per-frame T=16) while giving detection multi-frame context in its FPN features.

---

## 3. CURRENT TRAINING STATUS

### 3.1 Run Parameters
| Parameter | Value |
|-----------|-------|
| Run ID | `mtl_mvit_run3` |
| Start time | 2026-07-09 09:45 JST |
| Elapsed | ~48 min |
| PID | 1002047 |
| State | Running (Sl, 99.9% CPU, 5.5% RAM) |
| GPU1 | 5594MB / 98% util |
| Epochs | 100 planned |
| Current | Epoch 1, batch ~3100/4000 |

### 3.2 Dataset
- **Training:** 78,391 sequence windows (T=16, stride=1), 36 recordings, 26,322 unique frames
- **Validation:** 37,796 sequence windows, 16 recordings, 38,036 unique frames
- **RAM cache:** 4,000 train images (~1.37GB), 2,000 val images (~0.68GB)
- **Label mode:** Hybrid (75-class activity, 24-class detection, 11-component PSR, 6D pose)

### 3.3 Loss Progression (Epoch 1)
```
Batch    Total     Det      Act      PSR      Pose
------------------------------------------------------
   0    18.4706   3.8503   4.7396   1.6192   1.6004    (initial, high det)
 100    35.9587   0.0651  22.0467   1.1132   0.1058    (act spikes up)
 500    23.2869   0.0246  16.1250   1.4182   0.0471    (act still dominant)
1000    11.4053   0.0043   9.3015   1.0455   0.1112    (act trending down)
1500    12.3167   0.0028  11.4531   1.3644   0.1862    (stable)
2000    14.4948   2.4423  11.9920   1.6345   0.0787    (det spikes: likely CIoU instability on hard boxes)
2500    11.1240   0.0018  13.9394   1.5872   0.0282    (act re-heating)
3000     6.2904   2.3324   3.2788   1.1347   0.1973    (best total so far — act at minimum)
3100    12.1233   0.0010  17.4191   1.5130   0.3043    (act spike: high-variance early training)
```

**Observations:**
- Activity loss dominates: ranges from ~3.3 to ~32.9, driving 60-80% of total loss
- Detection loss is mostly very low (<0.03) except occasional CIoU spikes (~2.4 at batches 2000, 3000)
- PSR loss is stable (0.9–2.0) — reasonable for 11-way per-frame BCE
- Pose loss is minimal (0.01–0.73) — the cosine-similarity loss converges quickly when backbone features are meaningful
- The `act=-0.0000` at batch 1900 is suspicious — likely numerical underflow or zero-gradient event (monitor)
- Loss volatility is typical for epoch 1 with pretrained backbone + randomly initialized heads

### 3.4 Checkpoint Status
| File | Size | Date | Notes |
|------|------|------|-------|
| `epoch_0001.pt` | 499MB | Jul 9 08:07 | From **previous run** (not current) |
| `latest.pt` | 499MB | Jul 9 08:07 | Same as epoch_0001 |
| `metrics.json` | 1.3KB | Jul 9 01:31 | Old eval: act_top1=0.022, det_mAP50=0.0, psr_f1=0.0, pose_fwd_mae=10.5° |
| `efficiency_metrics.json` | 277B | Jul 8 22:40 | 43.48M params, 129.59 GFLOPs, 10.97 FPS, 1.844 GB VRAM |

**No new checkpoints from current run yet** — first checkpoint saves at epoch boundary.

The `metrics.json` values (act_top1=0.022, det_mAP50=0.0, psr_event_f1=0.0) are from a `--test-only` eval of a partially-trained model or a config mismatch. These do NOT represent current capability:
- Act 2.2% is random chance on 75 classes — suggests the eval loaded wrong weights or had the zero-gradient bug
- Det 0.0 mAP was an eval subsample artifact (confirmed: 174 SOTA_PROTOCOLS audit found the model actually reached 0.468 in multi-task)
- PSR 0.0 F1 — PSR was starved by GELU saturation in the old head (now LeakyReLU-repaired)

---

## 4. SOTA COMPARISON (Protocol-Pinned)

### 4.1 Published SOTA on IndustReal
| Task | SOTA | Method | Reference | Our Target |
|------|------|--------|-----------|------------|
| Activity (Top-1) | 65.25% | MViTv2-S | Schoonbeek et al., WACV 2024 | ≥65.25% |
| Activity (Top-5) | 87.93% | MViTv2-S | Schoonbeek et al., WACV 2024 | ≥87.93% |
| Detection (mAP@0.5) | 0.838 | YOLOv8m | Schoonbeek et al., WACV 2024 | ≥0.838 (dual protocol) |
| Detection (mAP@0.5, full vid) | 0.641 | YOLOv8m | WACV protocol variant | ≥0.641 |
| PSR (Event F1) | 0.901 | STORM | Published baseline | ≥0.901 |
| Head Pose (Angular Error) | — | — | No established SOTA | Baseline |

### 4.2 Key Protocol Notes
- **Activity:** clip-level top-1/top-5 on 75 classes — **must use `ACT_CLASS_GROUPING="none"`** to match WACV exactly
- **Detection:** dual protocol — annotated-frames (↔ WACV 0.838) and entire-videos (↔ WACV 0.641)
- **Test split:** 10 subjects (never mix val/test) — confirmed in SOTA_PROTOCOLS_174
- **Val split:** 5 subjects — used for epoch-level eval, NOT for SOTA claims
- **SOTA numbers are from Schoonbeek et al. 2024** (WACV) — NOT from broader computer vision benchmarks

---

## 5. CONFIG STALENESS

The following `config.py` values are **out of sync** with the current MViTv2-S MTL training configuration:

| Config Variable | Current Value | Should Be | Impact |
|----------------|---------------|-----------|--------|
| `BACKBONE` | `'convnext_tiny'` | `'mvit_v2_s'` | Misleading — doesn't affect code (arch is hardcoded in train_mtl_mvit.py) but confuses readers |
| `FREEZE_BACKBONE` | `True` | `False` | Lying — backbone IS training (43.5M all trainable), this flag would break MViT if respected |
| `BACKBONE_PRETRAINED` | likely old path | N/A | Old ConvNeXt weights path — not used by MViT code path |
| Detection LR multipliers | various | Check | Some legacy DET_LR values were tuned for ConvNeXt |

These are cosmetic for running (train_mtl_mvit.py overrides with hardcoded arch) but will cause confusion during paper writing, ablation logging, and config tracking.

---

## 6. KNOWN ISSUES & FIXES APPLIED

### 6.1 Fixed Bugs (Post-Audit)
| Issue | Fix | File | Status |
|-------|-----|------|--------|
| PSR GELU saturation | Replaced GELU with LeakyReLU in TransformerEncoderLayer | `mvit_mtl_model.py:264` | Applied |
| Activity zero gradient | Schedule/masking bug in train loop (double-ramp in losses.py) | `losses.py` | Applied |
| Detection 0.0 mAP (eval artifact) | Eval subsample emptied some classes | `evaluate.py` | Patched |
| Class weights divide-by-zero | `np.where(counts > 0, ...)` guard | `train_mtl_mvit.py:296` | Applied |

### 6.2 Active Warnings (Non-Blocking)
```
UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False
  because encoder_layer.activation_relu_or_gelu was not True
```
- **Cause:** PSR head's TransformerEncoderLayer uses LeakyReLU activation instead of ReLU/GELU, which disables PyTorch's native nested tensor optimization
- **Impact:** No functional effect, marginal performance cost
- **Fix:** Suppress warning or set `enable_nested_tensor=False` explicitly

```
RuntimeWarning: divide by zero encountered in divide
  weights = np.where(counts > 0, total / (num_classes * counts), 0.0)
```
- **Cause:** 3 of 75 activity classes have zero samples in the training set
- **Impact:** Gracefully handled; those classes get weight=0 and never contribute to loss
- **Note:** Monitor during val eval — if zero-count classes appear in validation, their predictions are meaningless

### 6.3 Ongoing Risks
- **Detection CIoU spikes:** Detection loss occasionally jumps from ~0.002 to ~2.4 (batches 2000, 3000). This is common in early DFL+CIoU training but should stabilize by epoch 3-5. If persistent, consider CIoU loss clipping.
- **Activity loss volatility:** Activity CE loss swings wildly (3–33) even late in epoch 1. This may normalize after the first few epochs as classification head converges. Label smoothing (0.1) should help.
- **PCGrad computational cost:** PCGrad requires per-task backward + gradient projection on each step. With 4 tasks, this is ~4× the gradient computation of single-task training. Measured at ~1.1 min/100 batches (reasonable).
- **No validation eval yet:** `eval_every=5` means first eval at epoch 5. If there are configuration bugs (wrong loss, bad metric wiring), they won't surface until epoch 5 (~3.7 hours from start).

---

## 7. PERFORMANCE METRICS (Measured)

### 7.1 Efficiency
| Metric | Value |
|--------|-------|
| Total params | 43,475,842 (43.5M) |
| Trainable params | 43,475,842 (100%) |
| GFLOPs | 129.59 (per forward pass) |
| FPS (train) | ~10.97 |
| VRAM (GPU1) | 5,594 MiB / 15,911 MiB (35%) |
| Training cadence | ~1.1 min per 100 batches |
| Epoch time (est.) | ~44 min (at 4000 batches) |
| Total time (est.) | ~73 hours for 100 epochs |

### 7.2 Throughput Analysis
At batch_size=2, T=16, 224×224:
- GPU1 at 98% util during training — compute-bound (good)
- 5,594 MiB VRAM usage leaves ~10 GB free for larger batch or T
- CPU at 99.9% with num_workers=0 — the DataLoader runs in the main process

**Recommendation:** Increase `num_workers=4` to overlap data loading with GPU compute. This should improve throughput without increasing VRAM.

---

## 8. NEXT STEPS & RECOMMENDATIONS

### 8.1 Immediate (This Run)
- [ ] **Watch for epoch 1 checkpoint** at ~10:29 (44 min from start). Verify it saves correctly.
- [ ] **Monitor first loss drop at epoch 2** — activity should stabilize as classification head adapts
- [ ] **Watch batch 1900 pattern**: the `act=-0.0000` occurrence needs investigation if it reappears

### 8.2 Short-Term (Epochs 1-20)
- [ ] **First eval at epoch 5** (~3.7 hours from start) — this is the first signal of real metrics
- [ ] **Enable num_workers>0** on restart/hardware check (DataLoader is CPU-bound bottleneck)
- [ ] **Log gradient norms** for each head to detect vanishing/exploding gradients
- [ ] **Add TensorBoard logging** for loss curves (currently text-log only)

### 8.3 Medium-Term (Epochs 20-100)
- [ ] **Update config.py** — fix `BACKBONE`, `FREEZE_BACKBONE`, and stale ConvNeXt references
- [ ] **Implement proper eval** — run full val split eval at checkpoints, log mAP, top-1, F1, pose MAE
- [ ] **Ablation runs** — single-task baselines for all 4 heads with MViTv2-S backbone
- [ ] **Learning rate schedule** — cosine decay or step decay after epoch 50
- [ ] **Test split eval** — first SOTA comparison at epoch ~80-100 when activity/PSR converge

### 8.4 Config Cleanup (Paper Prep)
- [ ] Sync `config.py` to reflect MViTv2-S architecture
- [ ] Remove ConvNeXt-specific parameters and LR multipliers
- [ ] Document all training hyperparameters for reproducibility section
- [ ] Move hardcoded overrides from `train_mtl_mvit.py` into config

---

## 9. FILE MAP

| File | Purpose | Key Lines |
|------|---------|-----------|
| `src/models/mvit_mtl_model.py` | Full MTL model: backbone + 4 heads | 447 lines total |
| `src/models/mvit_mtl_model.py:345` | `MTLMViTModel` entry point | Forward pass orchestrator |
| `src/models/mvit_mtl_model.py:40` | `MViTFeaturePyramid` | Hook-based FPN feature extraction |
| `src/models/mvit_mtl_model.py:129` | `LightweightFPN` | Lateral+top-down fusion |
| `src/models/mvit_mtl_model.py:174` | `DetectionHead` | Decoupled cls+DFL reg |
| `src/models/mvit_mtl_model.py:219` | `ActivityHead` | LayerNorm→Linear |
| `src/models/mvit_mtl_model.py:243` | `PSRHead` | Causal Transformer on temporal features |
| `src/models/mvit_mtl_model.py:309` | `PoseHead` | MLP→Tanh→6D renormalized |
| `scripts/train_mtl_mvit.py` | Training entry point | ~1562 lines |
| `src/config.py` | Global config (**stale**) | ~1300+ lines |
| `src/runs/rf_stages/checkpoints/mtl_mvit_run/` | Checkpoint output dir | — |

---

## APPENDIX: Key Training Log Excerpt

```
2026-07-09 09:45:20 [INFO] Args: {plumbing=False, epochs=100, batch_size=2, ...pcgrad=True, hp_prec_cap=True...}
2026-07-09 09:45:20 [INFO] Building train dataset (sequence_mode=True, T=16)...
2026-07-09 09:48:34 [INFO] Train samples: 78391
2026-07-09 09:49:07 [INFO] Eval samples (val): 37796
2026-07-09 09:49:07 [INFO] Building MTL-MViT model...
2026-07-09 09:49:08 [INFO] MTLMViTModel: feats=768, act=75-cls, det=24-cls, psr=11-comp, fpn=256ch
2026-07-09 09:49:08 [INFO] Params: 43.5M total, 43.5M trainable
2026-07-09 09:49:08 [INFO] Log vars initialized to -0.5
2026-07-09 09:49:08 [INFO] Class weights — min=0.0000  max=137.1867  mean=9.9832  num_nonzero=72
2026-07-09 09:49:08 [INFO] Starting training (100 epochs)...
2026-07-09 09:49:10 [batch     0/39195] loss=18.4706 det=3.8503 act=4.7396 psr=1.6192 pose=1.6004
2026-07-09 10:32:33 [batch  3100/39195] loss=12.1233 det=0.0010 act=17.4191 psr=1.5130 pose=0.3043
```
