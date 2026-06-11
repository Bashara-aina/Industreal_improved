# Architecture Deep Dive — What We Have, What Works, What Doesn't
## POPW Opus Consultation v2 (2026-06-11)

---

## 1. Current Architecture (End-to-End Data Flow)

```
Input: [B, 3, 720, 1280] RGB frame
         │
         ▼
┌─────────────────────────────────────────────────┐
│ ConvNeXt-Tiny (ImageNet pretrained, 28.6M params)│
│ C2: [B, 96, 180, 320]   stride 4               │
│ C3: [B, 192, 90, 160]   stride 8               │
│ C4: [B, 384, 45, 80]    stride 16              │
│ C5: [B, 768, 23, 40]    stride 32              │
└─────────────────────────────────────────────────┘
         │                    │
         ▼                    ▼ (C5 bypasses FPN → PoseFiLM)
┌─────────────────┐   ┌──────────────────┐
│ FPN Neck         │   │ PoseFiLM         │
│ P3: [B,256,90,160]│   │ γ₁,β₁ from       │
│ P4: [B,256,45,80] │   │ keypoints+conf   │
│ P5: [B,256,23,40] │   │ C5_mod = γ·C5+β │
│ P6: [B,256,12,20] │   └────────┬─────────┘
│ P7: [B,256,6,10]  │            │
└────────┬──────────┘            ▼
         │              ┌──────────────────┐
         │              │ HeadPoseFiLM     │
         │              │ γ₂,β₂ from       │
         │              │ head_pose (sg)   │
         │              │ C5_mod2=γ·C5_mod+β│
         │              └────────┬─────────┘
         │                       │
    ┌────┴───────────────────────┴────────────────────┐
    │                                                  │
    ▼                                                  ▼
┌──────────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐
│ Detection    │  │ Body Pose│  │ Head Pose│  │ Activity     │
│ (P3-P7)      │  │ (P3)     │  │ (C4‖C5)  │  │ (C5mod2+P4+det)│
│ RetinaNet    │  │ Heatmap  │  │ MLP      │  │ FB+TCN+ViT  │
│ 24 classes   │  │ 17 kpts  │  │ 9-DoF    │  │ 75 classes   │
└──────────────┘  └──────────┘  └──────────┘  └──────────────┘
                                                     │
                                                     ▼
                                              ┌──────────────┐
                                              │ PSR          │
                                              │ (P3+P4+P5)   │
                                              │ Causal Trans. │
                                              │ 11 components │
                                              └──────────────┘
```

---

## 2. Module-by-Module Analysis

### 2.1 Backbone: ConvNeXt-Tiny

**What it does**: Feature extraction from single RGB frame.
**Parameters**: 28.6M (ImageNet pretrained)
**Output**: 4 multi-scale feature maps C2–C5

**Assessment**: 
- ✅ Solid choice for multi-task: good parameter efficiency, no BatchNorm (LayerNorm internal)
- ✅ ImageNet pretraining provides strong spatial features
- ⚠️ Per-frame only — no temporal information at backbone level
- ⚠️ On RTX 3060, limits batch size to 1 with all heads + VideoMAE active

**Alternatives considered**:
- ResNet-50: More params, similar FLOPs, BatchNorm complications
- EfficientNet: Compound scaling constraints
- Video-specific (TimeSformer, VideoMAE backbone): Requires video pretraining data
- **Swin-Tiny**: Could be better for multi-scale detection, window attention more efficient

### 2.2 FPN Neck

**What it does**: Multi-scale feature pyramid from backbone features.
**Parameters**: 4.47M
**Output**: P3–P7, 256 channels each

**Assessment**:
- ✅ Standard, proven architecture
- ✅ P6/P7 via stride-2 conv on C5 (correct for RetinaNet)
- ✅ Lateral 1×1 + top-down upsample + 3×3 smooth (textbook)
- ⚠️ Only C3–C5 used (C2 skipped) — may lose fine-grained detail for small objects

### 2.3 Detection Head (RetinaNet-style)

**What it does**: 24-class assembly state detection on P3–P7.
**Parameters**: 5.30M
**Architecture**: 
- Cls subnet: 4×Conv3×3+ReLU → Conv(9×24)
- Reg subnet: 4×Conv3×3+ReLU → Conv(9×4)
- 9 anchors per location (3 ratios × 3 scales)
- Anchor sizes: (24, 48, 96, 192, 384)

**Assessment**:
- ✅ Focal loss handles class imbalance
- ✅ GIoU loss for box regression
- ❌ **Anchor sizes mismatched to GT**: k-means on GT shows centers at 164–404px, but smallest anchors are 24px
- ❌ **Trunk not re-initialized**: `cls_subnet`/`reg_subnet` conv layers still contain collapsed features
- ❌ **173K anchors per image** for 0–3 large objects — massive over-provisioning
- ❌ Only 1.6% of anchors (P6/P7) can reach IoU≥0.5 with typical GT

**Proposed redesign**: 
- Class-agnostic localizer (single-class, anchor-free, P5–P7 only)
- ROI-Align high-res crop → state classification head (24-way)
- This converts dense fine-grained detection into two easier problems

### 2.4 Body Pose Head

**What it does**: 17-keypoint body pose estimation from P3 features.
**Parameters**: 1.64M
**Architecture**: ConvTranspose2d → GroupNorm → ReLU → Conv1×1 → heatmaps → soft-argmax

**Assessment**:
- ✅ Soft-argmax is differentiable, no heuristic grouping
- ✅ Wing Loss is robust to outliers
- ✅ Confidence-weighted loss ignores low-confidence predictions
- ⚠️ Only applicable to IKEA ASM (not IndustReal)
- ✅ Working correctly in current codebase

### 2.5 Head Pose Head (9-DoF)

**What it does**: Predicts forward_vector[3] + position[3] + up_vector[3].
**Parameters**: ~0.5M
**Architecture**: GAP(C4)‖GAP(C5) → MLP(1152→512→256→9)

**Assessment**:
- ✅ Only living metric in combined score (MAE ~0.344)
- ❌ **9 raw numbers with MSE** → 60-70° angular MAE (barely better than chance)
- ❌ No geometric constraints (vectors not forced to be unit-length or orthogonal)
- **Proposed fix**: 6D continuous rotation representation + normalized position + geodesic loss
  - Module exists: `head_pose_geo.py` (251 lines)
  - Config flag: `USE_GEO_HEAD_POSE = False`

### 2.6 PoseFiLM + HeadPoseFiLM

**What they do**: Two-stage feature modulation of C5 using pose information.
**Parameters**: PoseFiLM 841K + HeadPoseFiLM 401K = 1.24M

**Assessment**:
- ✅ Key architectural innovation (paper's identity)
- ✅ `1+tanh(·)` constraint on γ ∈ (0,2) prevents feature inversion
- ✅ Stop-gradient prevents feedback loops
- ❌ **On IndustReal**: body keypoints are pseudo-labels with confidence=1 (no real hand data)
- ❌ **det_conf poisoning**: raw unbounded logits dominate activity input (RC-19)
- **Proposed fix**: Sigmoid-bound det_conf, gate PoseFiLM by actual hand confidence

### 2.7 Activity Head (Feature Bank + TCN + 2×ViT)

**What it does**: 75-class activity recognition with temporal context.
**Parameters**: 8.44M
**Architecture**:
1. det_conf = MaxPool(cls_preds) → [B,24] (stop_grad)
2. f_joint = [det_conf ‖ GAP(C5_mod2) ‖ GAP(P4)] → [B,1048]
3. W_proj: Linear(1048→512) → f̃_t
4. Feature Bank: T=16 sliding window → [B,16,512]
5. TCN: 1D Depthwise Conv(k=5) + pointwise
6. 2× ViT blocks: CLS token, MHSA(8 heads, d_k=64), FFN
7. CLS readout → Linear(512→75)

**Assessment**:
- ❌ **FeatureBank is DEAD**: always returns current frame replicated 16× (RC-18)
- ❌ **ViT attention INVERTED**: dividing by d^-0.5 multiplies by √d=8 (RC-16)
- ❌ **det_conf is raw logits**: L2=243.39 ± 0.001, dominates activity input (RC-19)
- ❌ **Mixup/CutMix corrupt labels**: blend logits not inputs (RC-15)
- ❌ **Eval drops VideoMAE**: half input zeroed (RC-17)
- ❌ **Per-frame training**: benchmark requires clip-level protocol
- ❌ **LDAM s=30**: 30× logit amplifier on top of class-balanced sampling (three imbalance mechanisms)

**Proposed redesign**:
- Two-stage: (A) cache backbone embeddings, (B) train temporal heads on long sequences from cache
- Fine-tuned K400 video encoder (VideoMAE-v2 or MViTv2-S) as primary path
- CNN stream as auxiliary conditioning
- Clip-level training aligned to evaluation protocol

### 2.8 PSR Head (Causal Transformer + Per-Component MLPs)

**What it does**: 11-component binary procedure step recognition.
**Parameters**: 3.73M
**Architecture**:
1. Multi-scale GAP(P3+P4+P5) → concat → MLP(768→256)
2. Causal Transformer: 3 layers, 4 heads, d_model=256
3. 11 per-component tiny MLPs (256→64→1)
4. Per-video cache (max 32 entries) for O(1) inference

**Assessment**:
- ✅ Causal masking is correct (upper-triangular)
- ✅ Per-component heads are better than shared head
- ❌ **Per-frame BCE on 95%-static labels** → learns constant pattern
- ❌ **Temporal smooth loss can't fire in T=1 mode** (requires dim==3)
- ❌ **Sensitivity penalty capped at 0.05** — too gentle
- ❌ **Fill-forward labels near-constant within recordings** on 4 recordings

**Proposed redesign**:
- Predict TRANSITIONS, not per-frame states
- Gaussian-smeared transition targets
- Monotonic constraint + procedure-order prior
- Module exists: `psr_transition.py` (301 lines)

### 2.9 VideoMAE Stream (Optional)

**What it does**: Second temporal stream using VideoMAE-Small pretrained on Kinetics.
**Parameters**: 22M (frozen), ~600 MB VRAM
**Output**: 384-D features concatenated with CNN features

**Assessment**:
- ✅ Loads correctly from local cache
- ✅ +5-7% Top-1 improvement when working
- ❌ **Frozen features**: never fine-tuned (unfreeze at epoch 10 configured but not tested)
- ❌ **Eval drops it**: collate_fn_sequences doesn't include clip_rgb
- **Module exists**: `video_stream.py` (361 lines)

---

## 3. Loss Architecture

### 3.1 Current Loss Assembly

```
L_total = Σ_t exp(-s_t) · L_t · ramp_t + s_t

Where:
  L_det  = Focal(α=0.75, γ=2) + GIoU × 2.0
  L_pose = Wing(ω=0.05, ε=0.005) × 0.001
  L_hp   = MSE × 0.001
  L_act  = LDAM-DRW(s=30, label_smooth=0.1)  [or CE+smooth if simplified]
  L_psr  = BinaryFocal(α=0.25, γ=1.0) + temporal_smooth(w=0.05)

Kendall: s_t = clamp(log_var_t, -4, 2), learned per-task
Init: s_det=0, s_pose=-1, s_act=0, s_psr=0
```

### 3.2 Loss Problems

| Problem | Impact | Fix |
|---------|--------|-----|
| Kendall clamp was AFTER backward | Gradients see unclamped values | Move to before forward |
| Stage 3 resets log_var | Destroys learned uncertainty | Inherit across stages |
| LDAM s=30 + CB sampling + label_smooth = 3 imbalance mechanisms | Over-correction | Drop LDAM, use CE+smooth |
| Activity loss cap at 80 | Still allows 4.0 contribution | Lower or remove with simplified loss |
| PSR focal γ=1.0 (was 2.0) | Better but still constant-output | Switch to transition loss |
| det_conf raw logits in activity | Domination | Sigmoid-bound |

### 3.3 Simplified Loss (Configured, Active)

```python
USE_SIMPLIFIED_LOSS = True
SIMPLIFIED_LOSS_WEIGHTS = {
    'det': 2.0,      # 173K location loss
    'pose': 0.1,     # small-scale MSE
    'act': 2.0,      # 75-class CE
    'psr': 2.0,      # 11-component focal
    'head_pose': 0.1, # 9-DoF MSE
}
SIMPLIFIED_CE_LABEL_SMOOTHING = 0.15
SIMPLIFIED_DROP_LDAM = True
ASSERT_AND_CRASH = True
```

---

## 4. Training Pipeline

### 4.1 Staged Training (Current)

| Stage | Epochs | Active Heads | Backbone | Notes |
|-------|--------|-------------|----------|-------|
| 1 | 1–5 | Det only | Stages 0-1 frozen | Detection warmup |
| 2 | 6–15 | Det + Pose + HeadPose | Stage 0 frozen | Add spatial heads |
| 3 | 16–100 | All 5 heads | All trainable | EMA from ep 16 |

### 4.2 Problems with Staging

1. **Activity head frozen for 15 epochs** — backbone features drift away from what activity head expects
2. **Stage 3 warmup** multiplies gradient suppression (loss-side × LR-side)
3. **EMA enabled at stage 3** — but if heads are collapsing, EMA blends collapse into saved weights
4. **No per-task dataloaders** — every batch must serve all heads, but most frames have no GT for most tasks

### 4.3 Proposed: Two-Stage with Embedding Cache

**Stage A** (backbone + spatial heads):
- Train backbone + detection + head pose on annotated frames
- Use synthetic data for detection pretraining
- 20 epochs on synthetic, then fine-tune on real

**Stage B** (temporal heads from cache):
- Freeze backbone, run once over full dataset
- Cache per-frame embeddings (512-d × ~2M frames ≈ 4GB)
- Train activity + PSR on long real sequences (T=64–256) from cache
- Hundreds of epochs per hour from cache

---

## 5. Existing Tier Modules (Ready to Enable)

| Module | File | Lines | Purpose | Config Flag |
|--------|------|-------|---------|-------------|
| GeometryAwareHeadPose | `head_pose_geo.py` | 251 | 6D rotation + geodesic loss | `USE_GEO_HEAD_POSE` |
| PSRTransitionPredictor | `psr_transition.py` | 301 | Event-based PSR with monotonic decoder | `USE_PSR_TRANSITION` |
| ROIDetector | `roi_detector.py` | 379 | Class-agnostic localizer + state classifier | `USE_ROI_DETECTOR` |
| K400VideoStream | `video_stream.py` | 361 | Fine-tuned Kinetics video encoder | `USE_K400_VIDEO_STREAM` |

All four modules are implemented and importable. They need Opus's guidance on:
1. The correct order to enable them
2. How to integrate them with the existing training pipeline
3. What hyperparameters to use
4. What interactions to watch for

---

## 6. Data Pipeline

### 6.1 IndustReal Dataset Structure

```
recordings/{train,val,test}/{recording_id}/
├── rgb/000000.jpg ...      (1280×720, 10 FPS)
├── AR_labels.csv           (sparse action spans: start, action_id, end)
├── OD_labels.json          (COCO format: boxes + 24 ASD classes)
├── PSR_labels_raw.csv      (sparse per-component state changes)
├── pose.csv                (dense 9-DoF head pose per frame)
└── hands.csv               (dense 52-D hand joints per frame)
```

### 6.2 Data Statistics

| Split | Recordings | Frames | AR Classes | ASD Boxes | PSR Components |
|-------|-----------|--------|-----------|-----------|----------------|
| Train (full) | 37 | ~48K | 74 | ~923 | 11 |
| Train (5%) | 4 | 3,112 | ~12 | ~100 | 11 |
| Train (25%) | ~9 | ~12K | ~35 | ~450 | 11 |
| Val | 17 | ~35K | 74 | ~500 | 11 |

### 6.3 Critical Data Issues

1. **5% subset = 12/75 classes** — structural ceiling for activity
2. **Most frames have no GT boxes** — `g=0.0000000` on most training steps
3. **PSR labels fill-forward** — 95% static within recordings
4. **NA class dominates activity** — most frames are "no action"
5. **Frame cache uses 5-7GB RAM** — limits batch size experimentation
