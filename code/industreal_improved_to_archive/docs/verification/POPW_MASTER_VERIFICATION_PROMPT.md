# POPW Architecture Master Verification Prompt

**Date:** May 6, 2026
**Purpose:** Final architecture verification before training launch
**Status:** READY FOR TRAINING — All 9 fixes verified, 12/12 tests passing

---

## WHAT TO DO

You are verifying the POPW (Procedural Operations & Work) multi-task vision model implementation against its paper specification (`popw_paper.tex`). Your job is to:

1. **Verify every architectural component** in `model.py` matches `popw_paper.tex` exactly
2. **Verify all losses** in `losses.py` match paper specification
3. **Verify the training pipeline** in `train.py` implements staged training, Kendall loss, EMA correctly
4. **Verify the evaluation pipeline** in `evaluate.py` produces the correct metrics
5. **Confirm the benchmark targets** are achievable given the architecture
6. **Confirm efficiency claims** (params, GFLOPs, FPS) match the paper

---

## FILE MANIFEST

All files are located at:
```
/media/newadmin/master/POPW/working/code/industreal_improved_to_archive/
Paper (reference):
/media/newadmin/master/POPW/working/code/popw_paper.tex
```

### Implementation Files (send all to Claude)

| File | Lines | Purpose |
|------|-------|---------|
| `model.py` | ~1744 | POPWMultiTaskModel — all 5 task heads |
| `losses.py` | ~712 | MultiTaskLoss, Kendall weighting, all loss functions |
| `config.py` | ~563 | Hyperparameters, flags, defaults |
| `train.py` | ~2000 | Training loop, staged training, EMA, schedulers |
| `evaluate.py` | ~2225 | Evaluation pipeline, all metric computation |
| `industreal_dataset.py` | ~900 | Dataset, clip loading, transforms |

### Support Files (reference only — don't resend)

| File | Purpose |
|------|---------|
| `POPW_DEEP_VERIFICATION.md` | 100% component-by-component paper vs code verification |
| `POPW_VERIFICATION_REPORT.md` | All 9 fixes applied, smoke test results |
| `01_HONEST_AUDIT.md` | Pre-fix audit of all discrepancies |
| `01_FINAL_VERDICT.md` | Smoke test results, pre-flight checklist |
| `02_BENCHMARK_FORECAST.md` | Per-benchmark expected scores |
| `03_ACTION_PLAN.md` | Phased training plan |
| `smoke_test.py` | 12/12 passing smoke tests |
| `test_e2e_training.py` | E2E training verification |

---

## BACKBONE: ConvNeXt-Tiny + FPN

**Paper §2.1 (lines 138–152):**

> *"The backbone is ConvNeXt-Tiny pretrained on ImageNet (LayerNorm internal, no frozen BatchNorm). Given input [B, 3, 720, 1280], the backbone produces four feature maps: C2: stride 4, 96×180×320; C3: stride 8, 192×90×160; C4: stride 16, 384×45×80; C5: stride 32, 768×23×40."*

Verify:
- [ ] `build_backbone('convnext_tiny')` is the default (was `resnet50` before FIX #8)
- [ ] Channel counts: C2=96, C3=192, C4=384, C5=768
- [ ] FPN takes C3/C4/C5, outputs all levels at 256 channels
- [ ] C5 goes directly to PoseFiLM (bypasses FPN) — NOT from FPN output

---

## DETECTION HEAD (24 ASD Classes)

**Paper §2.2.1 (lines 155–164):**

> *"A RetinaNet-style head operating on P3--P7 with shared classification and regression subnets: Cls subnet: 4× Conv 3×3 + ReLU → Conv(9×24) producing cls_preds [B, N, 24]; Reg subnet: 4× Conv 3×3 + ReLU → Conv(9×4) producing reg_preds [B, N, 4]; Loss: Focal loss (α=0.25, γ=2) + GIoU loss; Anchors: 3 ratios × 3 scales, sizes (24, 48, 96, 192, 384)"*

Verify:
- [ ] DetectionHead uses shared cls_subnet and reg_subnet (4-layer each)
- [ ] Anchor sizes: (24, 48, 96, 192, 384) — NOT the old (32, 64, 128, 256, 512) — FIX #8
- [ ] Anchor ratios: [0.5, 1.0, 2.0], scales: [1.0, 1.26, 1.59]
- [ ] FocalLoss(alpha=0.25, gamma=2.0)
- [ ] GIoU loss for regression
- [ ] cls_preds shape [B, N, 24], reg_preds shape [B, N, 4]

---

## BODY POSE HEAD (17 Keypoints)

**Paper §2.2.2 (lines 166–175):**

> *"Upsampling: ConvTranspose2d (k=4, s=2, p=1) + GroupNorm(32) + ReLU → [B, 256, 180, 320]; Heatmaps: Conv 1×1 → [B, 17, 180, 320]; Keypoints: Soft-argmax (T=0.1) → kpts [B, 17, 2] + conf [B, 17]; Loss: Wing Loss (ω=0.05, ε=0.005), confidence-weighted"*

Verify:
- [ ] ConvTranspose2d(k=4, s=2, p=1) + GroupNorm(32) + ReLU
- [ ] SoftArgmax temperature=0.1
- [ ] WingLoss(omega=0.05, epsilon=0.005) with **explicit ×0.001** — FIX #6
- [ ] Output: keypoints [B, 17, 2], confidence [B, 17]

---

## HEAD POSE HEAD (9-DoF)

**Paper §2.2.3 (lines 176–184):**

> *"Input: GAP(C4) | GAP(C5) → [B, 384+768=1152]; MLP: 1152 → 512 → 256 → 9 (LayerNorm + GELU + Dropout); Output: head_pose [B, 9] = forward[3] | position[3] | up[3]; Loss: MSE × 0.001 (meter-scale normalization)"*

Verify:
- [ ] GAP(C4) [B, 384] + GAP(C5) [B, 768] → concat [B, 1152]
- [ ] MLP: 1152 → 512 → 256 → 9 (with LayerNorm, GELU, Dropout)
- [ ] **MSE loss × 0.001** (not ×1.0)
- [ ] HeadPoseFiLM receives head_pose but with **`.detach()`** — FIX #1 CRITICAL

---

## HEADPOSEFiLM MODULE

**Paper §2.3 (lines 219–228):**

> *"Input: head_pose [B, 9] (stop_grad); γ_hp-net: 9 → 256 → 768, output 1 + tanh(·); β_hp-net: 9 → 256 → 768, output unbounded; Modulation: C5_mod2 = γ_hp · C5_mod + β_hp; GAP → activity: GAP(C5_mod2) feeds into activity head"*

Verify:
- [ ] `headpose_film(c5_mod, head_pose.detach())` — head_pose is detached at call site — FIX #1
- [ ] γ_hp-net: Linear(9, 256) → LayerNorm → GELU → Linear(256, 768); output: 1+tanh ∈ (0,2)
- [ ] β_hp-net: Linear(9, 256) → LayerNorm → GELU → Linear(256, 768); unbounded
- [ ] Second modulation: C5_mod_2 = γ_hp · C5_mod + β_hp
- [ ] GAP(C5_mod2) feeds into activity head

---

## POSEFiLM MODULE

**Paper §2.3 (lines 207–218):**

> *"Confidence extraction: heatmaps → max → sigmoid → nan_to_num(0.5); no gradient; Pose encoding: keypoints [B, 34] | confidence [B, 17] → pose_flat [B, 51]; γ-net: 51 → 512 → 768, output 1 + tanh(·) ∈ (0, 2); β-net: 51 → 512 → 768, output unbounded; C5 direct: direct from backbone (bypasses FPN) [B, 768, 23, 40]; Modulation: C5_mod = γ · C5_direct + β"*

Verify:
- [ ] Confidence from soft-argmax: max → sigmoid, no gradient
- [ ] pose_flat = keypoints [B, 34] concat confidence [B, 17] → [B, 51]
- [ ] γ-net: 51 → 512 → 768, output 1+tanh ∈ (0, 2)
- [ ] β-net: 51 → 512 → 768, unbounded
- [ ] γ initialized so that γ bias = 1.0 (starts as identity)
- [ ] C5_mod = γ · C5_direct + β (C5 from backbone, NOT FPN)

---

## ACTIVITY RECOGNITION HEAD (74 Classes)

**Paper §2.2.4 (lines 186–199):**

> *"Detection context: MaxPool(cls_preds) → f_det [B, 24], stop_grad (no gradient back to detection); Spatial features: GAP(C5_mod2) [B, 768] (after FiLM conditioning) | GAP(P4) [B, 256]; Joint feature: Concat [f_det, f_app, f_spatial] → f_joint [B, 1048]; Projection: W_proj (1048 → 512) → f_t [B, 512]; Feature Bank: sliding window B_t = [f_t-T+1, ..., f_t] [B, T=16, 512]; TCN Block: 1D Depthwise Conv (k=5, dilation=1) for short-range motion; LayerNorm → GELU → Linear; DropPath=0.1; ViT Temporal Blocks (2 layers): Prepend CLS token [1, 1, 512]; Learnable pos. embed. [1, T+1, 512]; MHSA (8 heads, d_k=64, attn_dropout=0.1); FFN (LayerNorm → Linear 512→2048 → GELU → Linear 2048→512); DropPath 0.10, 0.15; pre-norm; Output: cls_token readout → y_cls [B, 512]; Dropout(0.1) → act_logits [B, 74]; Loss: LDAM-DRW Loss (74 cls, label_smooth=0.1)"*

Verify:
- [ ] Detection context: `cls_preds.max(dim=1)` inside `torch.no_grad()`
- [ ] Feature concat: det_conf(24) + GAP(C5_mod2)(768) + GAP(P4)(256) = 1048
- [ ] Projection: Linear(1048, 512)
- [ ] FeatureBank: embed_dim=512, window_size=16
- [ ] **TCN is TRUE depthwise: groups=embed_dim** — FIX #5 (was not depthwise before)
- [ ] MHSA: 8 heads, d_k=64, **attn_dropout=0.1** — FIX #3 (was 0.3 before)
- [ ] DropPath: 0.10 (first ViT block), 0.15 (second)
- [ ] Pre-norm: norm1 before attention
- [ ] CLS token prepended to sequence
- [ ] Classifier: LayerNorm → Dropout(0.1) → Linear(512, 74)
- [ ] LDAM-DRW with label_smoothing=0.1, DRW at epoch 60

---

## PSR HEAD (11 Components)

**Paper §2.2.5 (lines 1288–1294):**

> *"Architecture: Per-frame feature: multi-scale P3+P4+P5 GAP → MLP → 256-D; Causal Transformer encoder (3 layers, 4 heads, d_model=256); Per-component output heads (11 separate tiny MLPs)"*

Verify:
- [ ] P3+P4+P5 GAP → concat → MLP → 256-D per frame
- [ ] **d_model=256** for transformer — FIX #2 (was 128 before)
- [ ] Causal mask on transformer (subsequent positions can't attend to earlier)
- [ ] 3 transformer layers, 4 heads
- [ ] 11 separate output heads (256→64→1 each)
- [ ] PSR loss: Binary Focal(α=0.25, γ=2.0) + temporal smoothness(w=0.05)

---

## KENDALL LOSS & STAGED TRAINING

**Paper §3 (lines 232–260):**

> *"Following Kendall et al. (2018), we weight the four task losses: L = Σ_t exp(-s_t) · L_t · ramp_t + s_t where t ∈ {det, pose+head_pose, act, psr}, s_t = clamp(log σ²_t, -4, 2). Initialization: s_det=0, s_pose=-1, s_act=0, s_psr=0. Activity ramp: min(1, epoch/5)."*

> *"Stage 1 (epochs 1-5): Detection only; backbone layer1-3 frozen. Stage 2 (epochs 6-15): + Pose + Head Pose; Activity and PSR heads frozen. Stage 3 (epoch 16+): All four task groups active."*

Verify:
- [ ] Kendall log_var init: s_det=0, s_pose=-1, s_act=0, s_psr=0
- [ ] Kendall clamping: clamp(-4, 2)
- [ ] Stage 1: det + pose + head_pose active; act + psr zeroed
- [ ] Stage 2: det + pose + head_pose active; act + psr zeroed
- [ ] Stage 3: all four active
- [ ] Backbone freezing: Stage 1 → ConvNeXt stages [0,1] frozen; Stage 2 → stage [0] frozen — FIX #7

---

## EMA

**Paper implies EMA in staged training section.**

Verify:
- [ ] **USE_EMA = True by default** — FIX #4 (was False before)
- [ ] EMA decay = 0.999
- [ ] EMA shadow updated after each epoch

---

## LOSS FORMULA SUMMARY

**Paper §3 (lines 241–248):**

> *"L_det = Focal(α=0.25, γ=2) + GIoU; L_pose = Wing Loss(ω=0.05, ε=0.005) × 0.001; L_hp = MSE × 0.001; L_act = LDAM-DRW; L_psr = Binary Focal(α=0.25, γ=2.0) + temporal smoothness(w=0.05)"*

Verify all five losses with correct hyperparameters.

---

## EFFICIENCY TARGETS (Paper §6)

**Paper claims:**
- POPW does 5 tasks in ONE forward pass
- Comparable or fewer parameters than 3 separate models (~77M combined)
- POPW ~52-75M params depending on VideoMAE enabled

Verify from code:
- [ ] Single backbone forward pass produces all 5 task outputs
- [ ] Parameter count: ~52.5M trainable (without VideoMAE) or ~75M (with VideoMAE)
- [ ] `efficiency_report.py` produces FPS, latency, GFLOPs, memory numbers

---

## BENCHMARK TARGETS (from Paper Table 3 + STORM-PSR)

| Metric | Target | Expected (Phase 1) | Expected (Full) |
|--------|--------|-------------------|-----------------|
| ASD mAP@0.5 | >83.8% (YOLOv8m) | 83-85% | 86-88% |
| Activity Top-1 | >66.45% (MViTv2) | 67-70% | 73-77% |
| Activity Top-5 | >88.43% | 90-92% | 92-94% |
| PSR F1 (±3 frames) | >0.901 (STORM-PSR) | 0.86-0.89 | 0.91-0.93 |
| PSR POS | >0.812 | 0.78-0.81 | 0.83-0.86 |
| Head Pose | Establish baseline | 7-10° MAE | 6-8° MAE |
| Assembly State F1 | ~0.85 (estimated) | 0.84-0.87 | 0.87-0.90 |
| Error Verification AP | ~0.58 (baseline) | 0.62-0.66 | 0.65-0.70 |

---

## ALL 9 FIXES APPLIED (from POPW_VERIFICATION_REPORT.md)

| # | Priority | Fix | Location | Status |
|---|----------|-----|----------|--------|
| 1 | CRITICAL | `headpose_film(c5_mod, head_pose.detach())` | model.py:1620 | ✅ |
| 2 | HIGH | PSR d_model=256, hidden_channels=256 | model.py:1505 | ✅ |
| 3 | HIGH | ViT attention dropout=0.1 | model.py:1522 | ✅ |
| 4 | HIGH | USE_EMA=True | config.py:275 | ✅ |
| 5 | MEDIUM | TCN true depthwise: groups=embed_dim | model.py:901 | ✅ |
| 6 | MEDIUM | Pose loss ×0.001 explicit | losses.py:565 | ✅ |
| 7 | LOW | Backbone freezing: S1→[0,1], S2→[0] | train.py:430,448 | ✅ |
| 8 | LOW | Model constructor default: convnext_tiny | model.py:1457 | ✅ |
| 9 | LOW | Docstrings updated to ConvNeXt-Tiny | model.py:1-69 | ✅ |

---

## TEST RESULTS (POPW_VERIFICATION_REPORT.md)

### smoke_test.py — 12/12 PASSING ✅
1. Imports — all modules import cleanly
2. Config values — 17/17 match spec
3. Model tensor shapes — 16/16 shape checks pass
4. Kendall logvar init — s_det=0, s_pose=-1, s_act=0, s_psr=0
5. Loss function sanity — all 4 task losses + Kendall weights finite
6. Backward pass + gradient flow — 348 params have grads
7. headpose_film gradient isolation — `.detach()` correctly isolates head_pose_head
8. FeatureBank round-trip — forward/backward/reset all correct
9. EMA functionality — shadow diverges correctly at decay=0.999
10. Staged Kendall masking — Stage 1 zeroes act/psr; Stage 2 zeroes act/psr
11. Individual loss functions — Wing, Focal, GIoU, LDAM, BinaryFocal all finite
12. Parameter counting — 53M total, 52.3M trainable

### test_e2e_training.py — PASSING ✅
- Model forward pass on CUDA
- MultiTaskLoss forward + backward on CUDA
- Gradient accumulation across 4 micro-steps
- AdamW optimizer step
- EMA shadow update
- Kendall nn.Parameter device sync (forward device move)

---

## PRE-FLIGHT CHECKLIST (from 01_FINAL_VERDICT.md)

Before launching training:

1. **Fix evaluate.py** (if not already fixed):
   - Replace `MultiTaskIndustReal` → `POPWMultiTaskModel` (2 places)
   - Dedent lines 1808–1810 from 8 spaces to 4 spaces

2. **Install dependencies**:
   ```bash
   pip install lion-pytorch transformers fvcore onnxruntime psutil scikit-learn tqdm
   ```

3. **Run synthetic detection pretraining** (overnight):
   ```bash
   python pretrain_synthetic.py
   ```

4. **Smoke test**:
   ```bash
   python train.py --debug --max-epochs 1
   ```

5. **Launch real run**:
   ```bash
   python train.py --resume runs/pretrain_synthetic/checkpoints/best.pth --max-epochs 60 --seed 42
   ```

6. **Evaluate**:
   ```bash
   python evaluate.py --checkpoint runs/.../checkpoints/best.pth --split test
   ```

---

## WHAT YOU MUST VERIFY

For each section above, your job is to:
1. Read the **paper specification** (quoted lines from `popw_paper.tex`)
2. Read the **implementation** (exact lines from `model.py`, `losses.py`, `train.py`, `evaluate.py`)
3. Compare them
4. Confirm: **MATCH ✅** or **MISMATCH ❌**

If you find a MISMATCH, report:
- What the paper says
- What the code actually does
- Whether the previously applied fix (from the table above) is correct

---

## FINAL ANSWER FORMAT

After reading all files and verifying all components, respond with:

```
## VERIFICATION RESULT

### Architecture Components
[For each component section, state: MATCH ✅ or MISMATCH ❌ + evidence]

### Loss Functions
[For each loss, state: MATCH ✅ or MISMATCH ❌ + evidence]

### Training Pipeline
[Staged training, Kendall, EMA, schedulers — all MATCH ✅ or MISMATCH ❌]

### Evaluation Pipeline
[All metrics computed correctly — YES/NO]

### Efficiency Claims
[Params, GFLOPs, FPS match paper — YES/NO]

### Benchmark Achievability
[Given the verified architecture, are the benchmark targets achievable?]

### READY FOR TRAINING
YES / NO

[If NO, list remaining critical issues that must be fixed first]
```
