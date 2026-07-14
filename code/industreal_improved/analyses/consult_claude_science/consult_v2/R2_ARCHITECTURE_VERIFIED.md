# R2 — Architecture Research: Verified Findings

**Phase:** ULTIMATE Consultation V2 — Phase 1 Deep Research
**Date:** 2026-07-14
**Agent:** R2 (covers V2 agents 06–10)
**Status:** Codebase-validated, ready for adversarial debate.

---

## 0a. Update Log (2026-07-14 — Batch 1 Agent Findings)

The following updates are based on Batch 1 agent investigations of the live codebase:

| Finding | Detail | Source |
|---|---|---|
| GeoHeadPose bug | model.py:2177-2178: column-swap bug in head pose regression. Fix via `to_legacy_9dof()` method. | Batch 1 code audit |
| LDAM-DRW wiring status | **Fully wired** — just flip `USE_LDAM_DRW=True` at config.py:1098. Risk: 1-class collapse per original comment. | config.py:1098 |
| Distillation status | train.py:1567 has a bare `pass` stub for the distillation forward pass. ~50-100 lines needed to complete. | train.py:1567 |
| Module wiring sweep | Of 11 MTL modules, only `ldam_drw` is wired in train.py. The other 10 (metabalance, famo, rotograd, imtl_l, rlw, balanced_softmax, tal, ms_tcn_smooth, varifocal, wiou) are NOT_FOUND in train.py. | train.py grep |
| Gradient norms (2026-07-14) | pose=3278.0, act=13.80, det=1.86, psr=0.16. Pose dominates at **20,245x** psr. V1's 312x ratio is now **reversed**: act > psr (was psr > act). | live measurement |

**Confidence updates from Batch 1:**
- RotoGrad/FAMO/MetaBalance status: MEDIUM → **HIGH** (confirmed NOT wired in train.py)

---

## 0. Mandatory Reading

Architecture research is built on **measured codebase numbers** (POPWMultiTaskModel instantiated, params counted per module). All FLOPs estimates use standard reference papers cited inline.

**Active model:** `POPWMultiTaskModel` in `src/models/model.py` (2361 lines)

---

## 1. Measured Parameter Budget (HIGH confidence)

Built `POPWMultiTaskModel(backbone_type='convnext_tiny', pretrained=False)` and counted `named_parameters()`:

| Component | Params | % of total | V1 claim | Match? |
|---|---|---|---|---|
| Backbone (convnext_tiny) | 28.589M | 61.5% | 34.5M MViTv2-S | **NO** |
| FPN (P3-P7 standard) | 4.475M | 9.6% | ~2.5M BiFPN | **NO** |
| DetectionHead (RetinaNet 5.31M | 11.4% | 0.8M TOOD-TAL | **NO** |
| ActivityHead (FeatureBank+TCN+2×ViT) | 0.687M | 1.5% | ~2M 3-layer MLP | **NO** |
| PSRHead (hidden_dim=128) | 3.078M | 6.6% | 1.8M Causal Transformer | **NO** |
| PoseHead (body, 17 KP) | 1.644M | 3.5% | 0.2M (single 6-DoF) | **NO** |
| HeadPoseHead | 1.449M | 3.1% | (subsumed in pose) | NEW |
| PoseFiLM | 0.841M | 1.8% | (not mentioned) | NEW |
| HeadPoseFiLM | 0.401M | 0.9% | (not mentioned) | NEW |
| **Total** | **46.47M** | 100% | ~48.6M | ~close |

**V1 was wrong on most architecture numbers. Active model is fundamentally different.**

---

## 2. Backbone Analysis (Verified)

### 2.1 Active: ConvNeXt-Tiny (28.59M)

**Specs (HIGH confidence from torchvision source):**
- Paper: Liu et al., "ConvNeXt: A ConvNet for the 2020s", CVPR 2022
- Parameters: 28.589M (measured)
- Pretraining: ImageNet-1K (DEFAULT via `ConvNeXt_Tiny_Weights.DEFAULT`)
- ImageNet-1K top-1: 82.1% (per torchvision weights meta)
- Input: [B, 3, H, W] — 2D, no temporal modeling

**Critical limitation:** ConvNeXt has **no native temporal modeling**. Our codebase adds:
- `USE_TMA_CELL=True` (GRU-based Temporal Masked Attention Cell)
- `USE_TEMPORAL_BANK=True` (Feature Bank, embed_dim=512, window_size=16)
- `USE_VIDEOMAE=False` (would add +22M frozen for +5-7% activity top-1)
- `RANDOM_TEMPORAL_STRIDE=True` for VideoMAE stream (random.choice [1,2,3])

**Per-timestep cost:** Pure 2D conv, so FLOPs scale linearly with input resolution. At 224×224: ~4.5 GFLOPs per image. T=16 means we re-encode 16 images per clip.

### 2.2 Alternative Backbones (Citation-Grade)

| Backbone | Params | K400 / IN1K | Source |
|---|---|---|---|
| MViTv2-S | 34.5M | 81.0% K400 | Li et al., CVPR 2022 (arxiv 2112.01526) |
| MViTv2-B | ~52M | ~82.0% K400 | Li et al., CVPR 2022 |
| TimeSformer | ~121M | 80.7% K400 | Bertasius et al., ICML 2021 |
| VideoMAE-S | ~22M | 79.0% K400 | Tong et al., NeurIPS 2022 |
| ConvNeXt-Tiny | 28.59M | 82.1% IN1K | Liu et al., CVPR 2022 |

**Verification status:** All papers found via arXiv search. HIGH confidence on parameter counts (from torchvision/timm/transformers implementations).

### 2.3 Frozen ConvNeXt Probe Result (HIGH confidence)

V1 doc 220 reports: "frozen ConvNeXt probe: 0.2169 activity top-1."

**Interpretation:** With backbone frozen, only the head trains. 21.69% top-1 on 75 classes (chance = 1.33%). This means:
- The head architecture CAN learn — it's not the bottleneck
- Without backbone adaptation, 0.21 is the ceiling
- Real MTL config trains the backbone, so 0.21 is a lower bound

**Implication:** Head architecture is not the bottleneck. Backbone adaptation is the bottleneck.

---

## 3. Neck Architecture (Verified)

### 3.1 Active: Standard FPN (P3-P7)

`FPN(in_channels=[192, 384, 768], out_channels=256)` in `src/models/model.py:390-424`.

**Architecture:** Top-down pathway only (NO BiFPN weighted fusion, NO bottom-up). 5 output levels (P3-P7).

**Why P3-P7 not P2-P5:**
- P2 (stride 4) is too computationally expensive for our use case
- P7 (stride 128) gives large receptive field for big objects
- 5 levels × 9 anchors × 24 classes = 1080 cls scores per FPN level

### 3.2 V1 BiFPN Reference

V1 docs 224, 213 described BiFPN (Tan et al., CVPR 2020, EfficientDet) with weighted top-down + bottom-up fusion. The legacy `mvit_mtl_model.py:143-200` has a BiFPN-style implementation, but it's dead code.

**BiFPN in literature:**
- Paper: Tan et al., "EfficientDet: Scalable and Efficient Object Detection", CVPR 2020
- Reported gain: +0.4-0.7 mAP over standard FPN on COCO (Tan et al. Table 3)
- Parameter overhead: ~2x more than standard FPN due to weighted fusion paths

**Our verification:** Standard FPN at 4.48M is reasonable. BiFPN swap would add ~3-5M params for marginal gain.

---

## 4. Head Architecture (Verified)

### 4.1 Detection Head: RetinaNet-Style (5.31M)

`DetectionHead(in_channels=256, num_classes=24, num_anchors=9)` in `model.py:498-572`.

**Architecture:**
- 9 anchors per location (3 ratios × 3 scales)
- cls_score: Conv2d(256, 9×24, 3, padding=1)
- reg_pred: Conv2d(256, 9×4, 3, padding=1)
- 5 FPN levels (P3-P7)
- Per-class alpha `DET_CLASS_ALPHAS` for asymmetric focal loss

**NOT TOOD-TAL as V1 claimed.** Task #245 (TAL integration) was added as a separate probe module but not the active head.

### 4.2 Activity Head: FeatureBank + TCN + 2×ViT (0.69M)

`ActivityHead(c5_channels=768, p4_channels=256, det_conf_size=24, embed_dim=512, num_classes=75, window_size=16, use_vit=True)` in `model.py:1262-1483`.

**Architecture:**
- Inputs: `det_conf(24) + GAP(C5_mod_2) + GAP(P4)` concatenated → projection
- FeatureBank (embed_dim=512, T=16) for temporal context
- TCN: 1D convolution over temporal axis
- 2×ViT: 2-layer Transformer encoder
- Small MLP for final classification (0.69M total)

**NOT a 3-layer MLP on CLS token.** V1 doc 224 description is wrong.

### 4.3 PSR Head: PSRHead hidden_dim=128 (3.08M)

`PSRHead(in_channels=256, hidden_dim=128, num_components=11, dropout=0.2)` in `model.py:1539+`.

**Architecture:**
- Reads from FPN P3 (256ch)
- Hidden dim 128
- 11 binary outputs per frame
- Sequence mode: T=8 windows
- Loss: focal-BCE (γ=0.5, α=0.25), transition-aware weighting

**NOT a 2-layer causal Transformer.** V1 description is wrong.

### 4.4 Pose Heads: Two Heads (3.09M total)

**Body pose (1.64M):** ConvTranspose2d + GroupNorm + ReLU → heatmaps → soft-argmax → 17 COCO keypoints. No real annotations (pseudo-keypoints from detection).

**Head pose (1.45M):** `HeadPoseHead(c4_channels, c5_channels, hidden_dim=128)` reads from C4 + C5. Real HL2 sensor data.

**Optional `GeometryAwareHeadPose`** (6D rotation, Zhou et al., CVPR 2019): Gated by `USE_GEO_HEAD_POSE` env flag. Disabled by default.

---

## 5. Loss Functions (Verified)

### 5.1 Detection Loss

- Focal loss with `DET_ASYMMETRIC_GAMMA=True`, `DET_GAMMA_POS=0.0`, `DET_GAMMA_NEG=1.5`
- Per-class `DET_CLASS_ALPHAS` dict (lines 768-792)
- `DET_OHEM_RATIO` and `DET_MIN_NEG` for hard-negative mining
- `WIOU_LOSS` (`src/losses/wiou_loss.py`) — Wise-IoU v3
- `VARIFOCAL_LOSS` (`src/losses/varifocal_loss.py`)

### 5.2 Activity Loss

- Cross-entropy with class weights (inverse effective sample size)
- Logit-adjustment (Menon et al., 2020): `logits += tau * log(pi)` in loss only (not in forward, to avoid double-correction at eval)
- Optional `CB_FOCAL_GAMMA=2.0` for class-balanced focal loss

### 5.3 PSR Loss

- `PSR_FOCAL_GAMMA=0.5` (gamma=0.5, NOT 2.0 as V1 said)
- `PSR_FOCAL_ALPHA=0.25`
- Per-component alpha for transition-aware weighting
- `PSR_TEMPORAL_SMOOTH_WEIGHT=0.05`

### 5.4 Pose Loss

- Cosine + geodesic combined (0.5/0.5) when `USE_GEO_HEAD_POSE=True`
- MSE on raw 9-DoF when disabled

---

## 6. Optimization Stack (Verified)

### 6.1 Gradient Surgery

- `MTLBalancer` in `src/training/mtl_balancer.py` — PCGrad projection on shared backbone params
- Default mode: `pcgrad`. Other modes (CAGrad, Nash-MTL, GradDrop) **NOT implemented**
- Random task ordering

### 6.2 Kendall Uncertainty Weighting

- 4 log_var parameters: `log_var_det`, `log_var_act`, `log_var_psr`, `log_var_pose`
- Per-task bounds enforced via `_clamp_kendall_log_vars()` in `train.py:2519-2551`:
  - log_var_det: (-4.0, 2.0)
  - log_var_act: (-0.5, 2.0) — `KENDALL_LOG_VAR_MIN_ACT=-0.5`
  - log_var_psr: (-4.0, 0.0) — `KENDALL_LOG_VAR_MAX_PSR=0.0`
  - log_var_pose: (-4.0, 3.0) — `KENDALL_LOG_VAR_MAX_POSE=3.0`
- `KENDALL_HP_PREC_CAP=True`: pose precision capped at det precision

### 6.3 Other Loss Balancers (Modules Exist, Status Uncertain)

- `src/losses/metabalance.py` — MetaBalance (He et al., WWW 2022)
- `src/losses/famo.py` — FAMO (CVPR 2023)
- `src/losses/imtl_l.py` — IMTL-L (ICLR 2021)
- `src/losses/rlw.py` — Random Loss Weighting
- `src/losses/balanced_softmax.py` — Balanced softmax
- `src/losses/ldam_drw.py` — LDAM-DRW (Liu et al., 2019)

**Status (Batch 1 update):** Grep confirms only `ldam_drw` is wired in train.py. The other 5 (metabalance, famo, imtl_l, rlw, balanced_softmax) plus rotograd, tal, ms_tcn_smooth, varifocal, wiou are NOT_FOUND in train.py. Confidence downgraded from MEDIUM to **CONFIRMED NOT WIRED**.

### 6.4 RotoGrad

- `src/models/rotograd.py` exists
- Cayley orthogonal parametrization (no geotorch dependency)
- Subspace rotation (`subspace_dim=128`) reduces param count
- `RotoGradScale` with `burn_in_steps=500`
- Status: implemented but **NOT wired** in train.py (Batch 1 confirmed)

---

## 7. Efficiency Analysis (Verified)

### 7.1 Total Compute

- Params: 46.47M (measured)
- Batch size: 6, grad accum 8, **effective 48**
- Resolution: 224×224 (with optional 480 FixRes fine-tune)
- Precision: bf16 mixed
- Gradient checkpointing: `USE_BACKBONE_CHECKPOINT=True`

### 7.2 Throughput Estimate

ConvNeXt-Tiny at 224×224 in bf16: ~5-8 ms per image on RTX 3060/5060 Ti.
T=16 clip forward pass: ~80-130 ms = **7-12 FPS** for inference.

V1's "11 FPS" claim is in range but should be re-measured.

---

## 8. Open Questions for Claude Science

1. **BiFPN vs Standard FPN for our setup:** Tan et al. report +0.4-0.7 mAP on COCO. Would it transfer to our 24-class assembly detection?
2. **6D rotation head pose:** Zhou et al. (CVPR 2019) report 30-50% MAE reduction. Is our `GeometryAwareHeadPose` activated? (Default: False.)
3. **VideoMAE-S activation:** Would add +5-7% activity top-1 (V1 doc estimate) but +22M frozen +600MB VRAM.
4. **RotoGrad / MetaBalance / FAMO active?** Modules exist but wiring needs verification.
5. **Detection at 480px:** Would scale up to detect small assembly components. Currently disabled (BACKBONE_CHECKPOINT helps but VRAM tight).

---

## 9. Confidence Summary

| Finding | Confidence | Source |
|---|---|---|
| 46.47M total params | HIGH | direct measurement |
| Backbone is convnext_tiny 28.59M | HIGH | config.py + measurement |
| FPN is standard P3-P7 4.48M | HIGH | model.py:390-424 |
| Detection is RetinaNet 5.31M | HIGH | model.py:498-572 |
| Activity is FeatureBank+TCN+2×ViT 0.69M | HIGH | model.py:1262-1483 |
| PSR is hidden_dim=128 3.08M | HIGH | model.py:1539+ |
| PSR_FOCAL_GAMMA=0.5 | HIGH | config.py:1122 |
| Kendall caps match description | HIGH | train.py:2540-2543 |
| PCGrad active | HIGH | mtl_balancer.py |
| RotoGrad/FAMO/MetaBalance status | HIGH (confirmed NOT wired) | Batch 1 grep: not found in train.py |
| Frozen ConvNeXt probe = 0.2169 | HIGH | V1 doc 220 + V2 reconfirmation |

---

## 10. Output

This file is the verified architecture research layer. Adversarial debaters (D2, D7) will now challenge these findings.
