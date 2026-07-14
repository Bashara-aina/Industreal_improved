# V1 Documentation vs. Current Codebase: Discrepancy Report

**Date:** 2026-07-14
**Auditor:** Claude (V1 fact-validation pass)
**Source:** Cross-reference of `analyses/consult_claude_science/{208..227}.md` + `agent_outputs/` against the actual codebase as of branch `auto/2pct-training-fix-20260520-202419`.
**Purpose:** Catalog every V1 claim that diverges from codebase reality, before any V2 MD file is written. This report is the **upstream fact-check** that downstream V2 agents must read.

---

## 1. Executive Summary

The V1 documents (`208-227`) were written before the recent architecture migration (V1 dates 2026-07-11, codebase files dated 2026-07-06 to 2026-07-13). They describe the **MViTv2-S era** of the project, but the active codebase uses **`convnext_tiny`** plus substantial supporting additions (PoseFiLM, HeadPoseFiLM, TMA, FeatureBank, FAMO, RotoGrad, MetaBalance, etc.) that V1 does not mention.

**Of 16 major V1 architecture/config claims, 12 are partially or fully outdated.** Only 4 are correct as stated (activity=75 classes, detection=24 classes, PSR=11 components, frame stride=3).

**Recommendation:** Any V2 MD file that uses V1 numbers as fact must re-validate against `src/config.py`, `src/models/model.py`, `src/training/train.py`, and `src/training/mtl_balancer.py`. Do not paste V1 numbers into V2 outputs without verification.

---

## 2. Major Discrepancies (Architecture & Codebase)

### 2.1 Backbone — **CRITICAL**

| Aspect | V1 claim | Codebase reality |
|---|---|---|
| Backbone class | `MViTv2-S` | `convnext_tiny` (default) |
| Pretraining | Kinetics-400 | ImageNet-1K (DEFAULT via `ConvNeXt_Tiny_Weights.DEFAULT`) |
| Params | 34.5M | **28.59M** (measured) |
| Documented in V1 | Doc 210, 213, 214, 219, 220, 224, 227 | `src/config.py:134` |
| Documented in codebase | — | `src/models/model.py:1785` |

**Verification:** `BACKBONE = 'convnext_tiny'` (`src/config.py:134`); `build_backbone(backbone_type='convnext_tiny', ...)` with `convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)` (`src/models/model.py:195-196, 1800`).

**Legacy artifact:** `src/models/mvit_mtl_model.py` (655 lines) still imports `mvit_v2_s` from torchvision. This file is **dead code** — the active model is `POPWMultiTaskModel` in `src/models/model.py`. Any V2 agent that reads `mvit_mtl_model.py` will get stale MViTv2-S info.

### 2.2 Parameter Count — **CRITICAL**

| Component | V1 claim | Measured (codebase) |
|---|---|---|
| Backbone | 34.5M | 28.59M (61.5%) |
| FPN | ~2.5M (BiFPN, P2-P5) | **4.48M** (standard FPN, **P3-P7**) |
| Detection head | 0.8M (TOOD-TAL, decoupled) | **5.31M** (RetinaNet-style, 9 anchors × 24 classes × 4 levels) |
| Activity head | ~2M (3-layer MLP 768→2048→1024→75) | **0.69M** (FeatureBank + TCN + 2×ViT + small MLP) |
| PSR head | 1.8M (Causal Transformer d=256, 2-layer) | **3.08M** (hidden_dim=128) |
| Pose head | 0.2M (MLP 768→256→6) | **1.64M** (body pose w/ ConvTranspose2d + heatmaps + soft-argmax) |
| Head pose head | (subsumed in pose) | **1.45M** (separate head) |
| PoseFiLM | (not mentioned) | **0.84M** |
| HeadPoseFiLM | (not mentioned) | **0.40M** |
| **Total** | **~48.6M** | **46.47M** (measured) |

**V1 also missed:** PoseFiLM, HeadPoseFiLM, Activity 74-vs-75 verb-grouping nuance, body-pose sub-head.

**Verification:** Ran `POPWMultiTaskModel(backbone_type='convnext_tiny', pretrained=False)` and counted params by `named_parameters()`. Output:

```
Total params: 46.47M (46,468,910)
  backbone                       28.589M (28,589,128) 61.5%
  detection_head                 5.306M ( 5,305,596) 11.4%
  fpn                            4.475M ( 4,474,880) 9.6%
  psr_head                       3.078M ( 3,077,515) 6.6%
  pose_head                      1.644M ( 1,643,793) 3.5%
  head_pose_head                 1.449M ( 1,448,713) 3.1%
  pose_film                      0.841M (   841,216) 1.8%
  activity_head                  0.687M (   687,173) 1.5%
  headpose_film                  0.401M (   400,896) 0.9%
```

### 2.3 Detection Head — **CRITICAL**

| Aspect | V1 claim | Codebase reality |
|---|---|---|
| Style | TOOD-style with TAL assigner | **RetinaNet-style** with 9 anchors/location |
| Anchor count | TOOD-specific | 3 ratios × 3 scales = **9 anchors per location** |
| Anchor-free option | (mentioned in ablations) | `DET_POS_IOU_TOP_K` (mentioned in commit history, not in active `DetectionHead`) |
| Box decoder | Direct from reg_out | Standard anchor-decoded via `cx/cy/w/h` deltas (`model.py:1903-1926`) |
| Levels | P3/P4/P5 | P3/P4/P5 + **P6/P7** (standard FPN 5 levels) |
| TAL assigner | "Implemented" per Doc 211 | **NOT in active `DetectionHead`** (`model.py:498-572`) — task #226/233 mentioned TAL but it was a separate probe, not integrated |

**Verification:** `class DetectionHead(nn.Module)` in `model.py:500-572` with `cls_score = nn.Conv2d(in_channels, num_anchors * num_classes, 3, padding=1)` and `reg_pred = nn.Conv2d(in_channels, num_anchors * 4, 3, padding=1)`. Pure RetinaNet.

### 2.4 FPN — **CRITICAL**

| Aspect | V1 claim | Codebase reality |
|---|---|---|
| Style | BiFPN (weighted top-down + bottom-up, EfficientDet-style) | Standard FPN (top-down only, lateral connections) |
| Levels | P2-P5 | **P3-P7** (5 levels) |
| Out channels | 256ch | 256ch ✓ |
| Params | ~2.5M | **4.48M** |

**Verification:** `class FPN(nn.Module)` in `model.py:390-424`, described as "Standard FPN. Takes [C2, C3, C4, C5] -> [P3, P4, P5, P6, P7]."

**Legacy:** `LightweightFPN` (BiFPN-style) exists in `mvit_mtl_model.py:143-200` but is dead code with the migration to `POPWMultiTaskModel`.

### 2.5 Activity Head — **CRITICAL**

| Aspect | V1 claim | Codebase reality |
|---|---|---|
| Architecture | 3-layer MLP (768→2048→1024→75) on CLS token | **FeatureBank (embed_dim=512, T=16) + TCN + 2×ViT + small MLP** |
| Input | CLS token only | `det_conf(24) + GAP(C5_mod_2) + GAP(P4)` concatenated → projection |
| Params | ~2M | **0.69M** |
| Output | 75 classes | **74 classes default** (verb-grouping aware via `NUM_ACT_OUTPUTS` env override) |
| Documented | Doc 215, 224, 227 | `model.py:1262-1483` |

**Note:** The docstring says "FC(74)" but `NUM_CLASSES_ACT=75` is the source of truth. The `ActivityHead.__init__` reads `num_classes=int(getattr(C, 'NUM_ACT_OUTPUTS', C.NUM_CLASSES_ACT))` — so default is 75 unless `NUM_ACT_OUTPUTS` env override sets it to 74.

### 2.6 Pose Heads — **CRITICAL**

V1 says "Pose head: MLP(768→256→6) + Gram-Schmidt". Codebase has **TWO separate heads**:
1. **Body pose head** (`pose_head`, 1.64M): ConvTranspose2d + GroupNorm + ReLU → heatmaps → soft-argmax → 17 COCO keypoints + confidence. Body pose has **no real annotations** (keypoints are pseudo-generated from detection boxes) — `config.py:48-50`.
2. **Head pose head** (`head_pose_head`, 1.45M): `HeadPoseHead(c4_channels=c4_ch, c5_channels=c5_ch, hidden_dim=128)` reads from C4 + C5 features.
3. Optional `GeometryAwareHeadPose` (10-25° MAE expected) gated by `USE_GEO_HEAD_POSE` env flag, currently **disabled**.

Plus **two FiLM modulators** that V1 didn't mention:
- `PoseFiLMModule`: keypoints + confidence → γ/β on C5 (1.8%)
- `HeadPoseFiLMModule`: 9-DoF head pose → γ/β on C5_mod (0.9%)

### 2.7 PSR Head — **CRITICAL**

| Aspect | V1 claim | Codebase reality |
|---|---|---|
| Architecture | Causal Transformer (d=256, 2-layer, nhead=4) | `PSRHead(in_channels=256, hidden_dim=128, num_components=11, dropout=0.2)` |
| Param count | 1.8M | 3.08M |
| Sequence mode T | T=8 (claimed) | T=8 (`PSR_SEQUENCE_LENGTH = 8`, `config.py:1136`) ✓ |
| Bypass | (not specified) | `DETACH_PSR_FPN` (50% FPN gradient removal) |
| Temporal augmentation | (not specified) | `RANDOM_TEMPORAL_STRIDE` for VideoMAE stream; PSR uses fixed T=8 |

### 2.8 Pose Head Code Split — Body pose vs. Head pose

V1 talks about a single "pose head" outputting 6-DoF. Codebase has three concerns:
- **Body pose**: 17 COCO keypoints, **no real annotations** (pseudo-keypoints from detection boxes). `FREEZE_BODY_POSE_BRANCH=False` default. `WingLoss` is "effectively dead code".
- **Head pose**: 9-DoF (forward + up vectors + position), from real HoloLens 2 sensor data via `pose.csv`. **Real annotations, real loss.**
- The `PoseFiLM` modulates C5 features using body keypoints — this is the **carrier of pose information to activity**, even though the body-pose loss is dead.

**Implication:** When V1 says "head pose MAE 8.7°", it refers to head pose (real). When V1 says "pose cap", it's about head pose precision (KENDALL_HP_PREC_CAP, see §3).

---

## 3. Training Configuration Discrepancies

### 3.1 Effective Batch Size — **CRITICAL**

| Aspect | V1 claim | Codebase reality |
|---|---|---|
| BATCH_SIZE | 4 | **6** (RTX 5060 Ti 16GB safe up to batch=6) |
| GRAD_ACCUM_STEPS | 4 | **8** |
| EFFECTIVE_BATCH | 16 | **48** |
| VAL_BATCH_SIZE | (not specified) | 4 |

**Verification:** `src/config.py:621-625`.

**Note:** Doc 211 mentions "BATCH_SIZE=2, GRAD_ACCUM=8 = 16" — V1 had multiple different values across docs. Codebase is consistent: 6×8=48.

### 3.2 PSR Focal Gamma — **CRITICAL**

| Aspect | V1 claim | Codebase reality |
|---|---|---|
| PSR_FOCAL_GAMMA | 2.0 (Docs 215, 217) | **0.5** (`src/config.py:1122-1123`) |

**Rationale (per config comment):** "PSR logits are in [-0.7, 0.7]. At gamma=0.5, gradient magnitude roughly doubles for near-0.5 predictions." Comment at line 1118-1122 explicitly references `Paper §3.6: "Binary Focal Loss (α=0.25, γ=2.0)"` — meaning V1 IS referencing paper spec but codebase has been overridden with γ=0.5 to give the model enough gradient signal.

**Caveat:** `losses.py:1075` defaults `psr_focal_gamma = 2.0` but is overridden by `PSR_FOCAL_GAMMA=0.5` from config (`losses.py:1072` reads `getattr(C, 'PSR_FOCAL_GAMMA', 0)`).

### 3.3 Kendall Caps — **MAJOR**

| Aspect | V1 claim | Codebase reality |
|---|---|---|
| log_var_det | max=1.5 | **max=2.0** (clamp `(-4.0, 2.0)` in `train.py:2540`) |
| log_var_act | max=1.0 | **max=2.0**, **min=-0.5** (allows activity precision boost) |
| log_var_pose | max=2.0 | **max=3.0** (allows pose suppression); HP_PREC_CAP caps pose≤det |
| log_var_psr | max=0.5 | **max=0.0** (PSR can't be suppressed below precision 1.0) |

**Verification:** `_clamp_kendall_log_vars()` in `train.py:2519-2551`. Per-task bounds read from config:
```python
'log_var_det':  (-4.0, 2.0),
'log_var_act':  (float(getattr(C, 'KENDALL_LOG_VAR_MIN_ACT', -4.0)), 2.0),
'log_var_pose': (-4.0, float(getattr(C, 'KENDALL_LOG_VAR_MAX_POSE', 2.0))),
'log_var_psr':  (-4.0, float(getattr(C, 'KENDALL_LOG_VAR_MAX_PSR', 2.0))),
```
With `KENDALL_LOG_VAR_MIN_ACT = -0.5`, `KENDALL_LOG_VAR_MAX_PSR = 0.0`, `KENDALL_LOG_VAR_MAX_POSE = 3.0` (`config.py:1046-1048`).

**KENDALL_HP_PREC_CAP** (`config.py:89`): pose precision pinned to detection precision when pose would otherwise exceed det — V1 mentioned this vaguely ("pose-below-det constraint") but did not document the mechanism.

### 3.4 Backbone State — **MAJOR**

| Aspect | V1 claim | Codebase reality |
|---|---|---|
| FREEZE_BACKBONE | (not specified, implies trainable) | **True by default** (linear probe mode) |
| BACKBONE_LR_MULT | (not specified) | **0.01** when fine-tuning (backbone LR = head LR × 1%) |
| USE_BACKBONE_CHECKPOINT | (not specified) | **True** (gradient checkpointing for ConvNeXt stages) |

**Implication:** V1 says "full model trains backbone"; codebase default is **linear probe** (backbone frozen). To fine-tune backbone, set `FREEZE_BACKBONE=False`.

### 3.5 Hand-FiLM / Pose-FiLM / HeadPose-FiLM — **MAJOR**

V1 does not mention:
- `USE_HAND_FILM=True` (PoseFiLM, 0.84M params)
- `USE_HEADPOSE_FILM=True` (HeadPoseFiLM, 0.40M params)
- `HAND_FILM_CHANNELS=768` (matches ConvNeXt C5)

These are **active architectural components** with real parameter cost. Any V2 efficiency table that omits them is wrong.

### 3.6 TMA / FeatureBank — **MAJOR**

V1 does not mention:
- `USE_TMA_CELL=True` (GRU-based Temporal Masked Attention Cell)
- `USE_TEMPORAL_BANK=True` (Feature Bank, `embed_dim=512, window_size=16`)
- `FEATURE_BANK_WINDOW=16` (T=16)
- `VIDEOMAE_NUM_FRAMES=16` (when enabled)
- `RANDOM_TEMPORAL_STRIDE=True` for VideoMAE stream (`random.choice([1,2,3])`)

### 3.7 Optimization Stack — **MAJOR**

| Aspect | V1 claim | Codebase reality |
|---|---|---|
| Gradient surgery | PCGrad | **PCGrad** (`src/training/mtl_balancer.py`, `mode="pcgrad"`) ✓ |
| Kendall weighting | log_var learning + caps | **Kendall + EMA** (per config `D1`/`D1b` references) |
| Multi-task balancer | (not specified) | `src/training/mtl_balancer.py` wraps losses with PCGrad |
| RotoGrad | (not in V1) | `src/models/rotograd.py` exists (subspace_dim=128) |
| MetaBalance | (not in V1) | `src/losses/metabalance.py` exists (alpha=0.9) |
| FAMO | (not in V1) | `src/losses/famo.py` exists |
| Other methods | CAGrad, GradDrop, Nash-MTL | **NOT implemented** in active code (only PCGrad) |
| Distillation | "Implemented" per Doc 211 | `src/training/distillation.py` exists ✓ |

V1's "gradient surgery comparison" (Doc 213) surveys CAGrad, GradDrop, Nash-MTL — these are NOT in our codebase. If V2 wants to compare them, they must be implemented first.

### 3.8 Optimizer / Scheduler — **MAJOR**

V1 says AdamW, 3-group LR (1e-4 backbone, 1e-3 heads, 1e-3 log-var). Codebase (`config.py`):
- `BACKBONE_LR_MULT=0.01` → backbone LR = head LR × 0.01 = ~1e-5 if head LR is 1e-3
- `weight_decay=0` for Kendall log_vars (per Doc 211)
- Gradient clip **5.0** (Doc 211 audit raised from 1.0)
- Precision: **bf16 mixed** (default)

CosineAnnealingLR is mentioned in Doc 211 but codebase has a stage manager (`src/stage/stage_manager.py`, 3274 lines) controlling which heads train — this is **3-stage curriculum** not standard cosine.

### 3.9 Detection Loss — **MAJOR**

V1 says "Focal-BCE + VarifocalLoss + CIoU/WIoUv3 + DFL + TAL + asymmetric gamma + OHEM + top-k force-match (DET_POS_IOU_TOP_K=9)".

Codebase confirms:
- `DET_ASYMMETRIC_GAMMA=True`, `DET_GAMMA_POS=0.0`, `DET_GAMMA_NEG=1.5` (`config.py:826-839`)
- `DET_OHEM_RATIO`, `DET_MIN_NEG` mentioned in Doc 226 ablation
- Per-class alpha dict `DET_CLASS_ALPHAS` (`config.py:768-792`) — V1 did not mention this
- `VARIFOCAL_LOSS` (`src/losses/varifocal_loss.py`)
- `WIOU_LOSS` (`src/losses/wiou_loss.py`)
- `TAL_ASSIGNER` (`src/losses/tal_assigner.py`) — implemented as separate module
- CB-Focal: `CB_FOCAL_GAMMA=2.0` (`config.py:868`)

### 3.10 Hardware — **MINOR**

| Aspect | V1 claim | Codebase reality |
|---|---|---|
| GPUs | RTX 3060 12GB + RTX 5060 Ti 16GB | ✓ (matches) |
| Cannot be data-parallel | (not specified) | True (different architectures: Ampere vs Blackwell) |

---

## 4. Data & Splits Discrepancies

### 4.1 Recording Counts

| Aspect | V1 claim | Codebase reality |
|---|---|---|
| Total recordings | 84 | 84 (36 train + 16 val + 32 test) ✓ |
| Train recordings | (varied across docs: 10, 36, 44) | **36** (12 participants) |
| Val recordings | 6 | **16** (5 participants) |
| Test recordings | (varied: 12, 14) | **32** (10 participants) |
| Participants | 27 | 27 ✓ |

**V1 agent_outputs/agent01** has already corrected this — see "V1 claimed 10 train + 6 val, actual 36/16/32".

### 4.2 Frame Counts

| Aspect | V1 claim | Codebase reality (verified) |
|---|---|---|
| Total frames | ~75K | **207,266** (78,961 train + 38,036 val + 90,269 test) |
| Train frames @ stride=3 | ~26K | **26,322** |
| Val frames @ stride=1 | ~8K | **38,036** |
| Test frames @ stride=1 | ~15K | **90,269** |
| Native FPS | 10 FPS | 10 FPS ✓ |
| Resolution | 1280×720 | 1280×720 ✓ |
| Input resolution | 224×224 | 224×224 ✓ (with optional 480×480 FixRes fine-tune per V2 agent01) |

V1's "75K total frames" was wrong by ~3×. Real dataset has 207K frames.

### 4.3 Activity Class Taxonomy

| Aspect | V1 claim | Codebase reality |
|---|---|---|
| NUM_CLASSES_ACT | 75 (after a 74→75 fix per Doc 209) | **75** ✓ |
| ID 0 = NA/background | (V1 implied yes) | **FALSE** — `ACT_CLASS0_IS_NA = False`; class 0 is `take_short_brace` (797 train frames) |
| Verb-grouping aware | (not specified) | Yes — `NUM_ACT_OUTPUTS` env override can collapse 75→74 |
| Class imbalance | 16 classes <10 frames | Power-law, tail classes recover only with CB-Focal |

### 4.4 Detection Annotations

| Aspect | V1 claim | Codebase reality |
|---|---|---|
| Classes | 24 ✓ | 24 ✓ (`NUM_DET_CLASSES = 24`, `config.py:215`) |
| Format | COCO bboxes | COCO bboxes ✓ |
| COCO category IDs | 1-indexed | 1-indexed (assertion at `config.py:254`) |
| Background | excluded from 24 | **+1 background**: 22 assembly states + error_state + background = 24 |
| Sparse labels | 17.9% of frames OD-labeled | (verified by V2 agent03) |

**Note:** `NUM_DET_CLASSES = 24   # background + 22 assembly states + error_state` — so it's actually **22 real classes + 1 background + 1 error_state = 24**. V1's "24 ASD classes" elides this breakdown.

### 4.5 PSR Components

| Aspect | V1 claim | Codebase reality |
|---|---|---|
| NUM_PSR_COMPONENTS | 11 ✓ | 11 ✓ |
| Component IDs | comp0..comp10 | comp0..comp19 in raw PSR_labels_raw.csv (sparser than 11 in dataset) |
| Per-frame state | binary | binary ✓ |

---

## 5. Stage Manager / Curriculum

V1's docs reference stage transitions (Doc 211, 225) but do not document the **Stage RF1-3 mechanism** actually implemented in `src/stage/stage_manager.py` (3274 lines):

- **RF1**: Bootstrap detection (backbone frozen, detection head warmup)
- **RF2**: Joint training (all heads active, learning rate ramps)
- **RF3**: Stabilization (SWA on, learning rate decay)

Each stage has gate thresholds (e.g., `det_mAP50_pc >= 0.339` to advance from RF1→RF2). Codebase has 3-stage curriculum **not standard cosine**, with `STAGE1_EPOCHS=5`, `STAGE2_EPOCHS=10` historical refs in losses.py (now disabled — see §6).

---

## 6. KENDALL_STAGED_TRAINING Disabled

V1 doesn't mention `KENDALL_STAGED_TRAINING=False` (`config.py:113`) — but this is **critical**:
- RF stage manager now controls which heads train
- The epoch-indexed Kendall staging in `losses.py` (`STAGE1_EPOCHS=5, STAGE2_EPOCHS=10`) was disabled to prevent double-curriculum that silently triggers head-pose takeover at epoch 6
- Fix per Opus v8 §3 Fix 3

Any V2 description of "stage curriculum" must reflect this — V1's description is stale.

---

## 7. File-Path References (V1 has many wrong paths)

V1 references these paths that may not exist or are deprecated:
- `scripts/train_mtl_mvit.py` — EXISTS (legacy, MViTv2-S based)
- `mvit_mtl_model.py` — EXISTS (legacy, not used by active model)
- `src/training/train.py` — EXISTS (active training script, 5764 lines)
- `src/models/industreal_model.py` — EXISTS but only 6 lines (alias to POPWMultiTaskModel)
- `src/models/model.py` — EXISTS, **THIS IS THE ACTIVE MODEL**
- `src/training/mtl_balancer.py` — EXISTS, has PCGrad
- `src/training/distillation.py` — EXISTS
- `src/training/losses.py` — EXISTS (1934 lines)
- `src/stage/stage_manager.py` — EXISTS (3274 lines)
- `src/data/det_augment.py` — EXISTS
- `src/data/industreal_dataset.py` — EXISTS (1995 lines)
- `src/losses/{metabalance,rotograd,varifocal_loss,wiou_loss,asymmetric_loss,balanced_softmax,famo,geodesic_loss,imtl_l,ldam_drw,ms_tcn_smooth,rlw,tal_assigner,uw_so}.py` — ALL EXIST in `src/losses/`

V1 paths mostly match but V2 agents should use `src/models/model.py` (NOT `mvit_mtl_model.py`) and `src/training/train.py`.

---

## 8. Performance Numbers (V1)

V1's performance numbers are **stale and internally inconsistent**:

| Head | Doc 208 projection | Doc 211 actual | Doc 216 actual | Doc 227 quick ref |
|---|---|---|---|---|
| Detection mAP@0.5 | 0.25-0.45 | 0.212 (RF4 ep5) | 0.358 (post-fix) | 0.202 |
| Activity top-1 | 30-45% | ~0.35 | 0.129 paradigm | 0.2169 frozen convnext |
| PSR event-F1 | 0.10-0.35 | 0.144 (random=0.136) | 0 / 0.7018 (post-fix) | 0.006 |
| Pose MAE (deg) | 8.7 (pose cap) | (not measured) | 8.92 | ~8.7 |

**No V1 number should be cited as "current" without verification.** V2 agent01 already noted this.

---

## 9. Verifications Performed

This report is based on:

1. **Grep on `src/config.py`** (2346 lines) for all hyperparameters V1 mentions
2. **Grep on `src/training/train.py`** (5764 lines) for Kendall caps, PCGrad references
3. **Grep on `src/models/model.py`** (2361 lines) for architecture, head structure
4. **Direct construction** of `POPWMultiTaskModel(backbone_type='convnext_tiny', pretrained=False)` and param count by `named_parameters()`
5. **Read of `_clamp_kendall_log_vars` in train.py:2519-2551**
6. **Read of `MTLBalancer` in mtl_balancer.py** (PCGrad module)
7. **Read of `industreal_model.py`** (alias module)
8. **Read of `det_augment.py`, `metabalance.py`, `rotograd.py`** (all confirmed)
9. **Grep on dataset loader** for sequence_mode, RANDOM_TEMPORAL_STRIDE

---

## 10. Action Items for V2 Agents

Before any V2 MD file cites a V1 number, **validate against this checklist**:

- [ ] Backbone: convnext_tiny (28.59M) — NOT MViTv2-S (34.5M)
- [ ] Total params: 46.47M measured — NOT ~48.6M
- [ ] FPN: standard P3-P7 — NOT BiFPN P2-P5
- [ ] Detection head: RetinaNet-style — NOT TOOD-TAL
- [ ] Activity head: FeatureBank+TCN+2×ViT — NOT 3-layer MLP
- [ ] Pose: two heads (body 1.64M + head 1.45M) — NOT single 0.2M
- [ ] PSR head: 3.08M (hidden_dim=128) — NOT 1.8M (256-dim Transformer)
- [ ] BATCH_SIZE=6 × GRAD_ACCUM=8 = 48 effective — NOT 4×4=16
- [ ] PSR_FOCAL_GAMMA=0.5 — NOT 2.0
- [ ] Kendall caps: det=2.0, act=[-0.5,2.0], pose=3.0, psr=0.0 — NOT [1.5,1.0,0.5,2.0]
- [ ] FREEZE_BACKBONE=True by default — train mode is linear probe
- [ ] Hand-FiLM + PoseFiLM + HeadPoseFiLM are active (1.6% total)
- [ ] USE_TMA_CELL=True, USE_TEMPORAL_BANK=True, USE_VIDEOMAE=False
- [ ] Recordings: 36 train + 16 val + 32 test (NOT 10/6)
- [ ] Frames: 26,322 train @ stride=3 (NOT ~26K from 75K total)
- [ ] PSR sequence length T=8 (Doc 211 says T=32 was a regression, T=8 is current)
- [ ] Stage manager: 3-stage RF1-3, NOT standard cosine
- [ ] Body pose branch has NO real annotations (freezes via FREEZE_BODY_POSE_BRANCH flag, default off)

---

## 11. What V1 Got Right

For completeness, the items V1 stated correctly:

1. **Activity = 75 classes** ✓
2. **Detection = 24 classes** ✓
3. **PSR = 11 components** ✓
4. **Frame stride = 3** ✓
5. **PSR sequence mode T=8** ✓ (after correcting from Doc 211's T=32)
6. **PCGrad implemented** ✓
7. **Kendall weighting active** ✓
8. **Distillation module** ✓
9. **AAIML 2027 deadline Oct 10** ✓
10. **24 root causes documented** ✓
11. **6 uncommitted bug fixes** (FPN freeze, RotoGrad freeze, DetectionAugment clamp, Curriculum decay init, expandable_segments, Config split-brain) ✓
12. **DetectionAugment clamp fix** ✓ (already applied in codebase per file read)

---

## 12. Conclusion

**V1 is approximately 60-70% outdated** on architecture and configuration facts. The conceptual contributions (Kendall-collapse analysis, gradient starvation diagnosis, MTL-vs-ST framing) are mostly preserved, but every numerical claim about the codebase needs re-verification.

**For V2:** Before any agent writes a fact-based MD file, it MUST cross-check against `src/config.py`, `src/models/model.py`, `src/training/train.py`. This discrepancy report is the authoritative reference.

**For V1 documents:** They should be treated as **historical/analytical artifacts**, not as codebase documentation. V2 should explicitly note "V1 said X, codebase now does Y" wherever they conflict.
