# D7 — Architecture Detailed Adversarial Debate

**Phase:** ULTIMATE Consultation V2 — Phase 2 Adversarial Debate
**Date:** 2026-07-14
**Debater:** D7 (continues D2 with deeper architecture challenges)

---

## 0a. Update Log (2026-07-14 — Batch 1 Agent Findings)

| Finding | Detail |
|---|---|
| GeoHeadPose bug | model.py:2177-2178 column-swap in head pose. Fix via `to_legacy_9dof()`. |
| LDAM-DRW wiring | **Fully wired** — flip `USE_LDAM_DRW=True` at config.py:1098. Risk: 1-class collapse. |
| Distillation stub | train.py:1567 has `pass`. Needs ~50-100 lines to complete. |
| Module sweep | Only `ldam_drw` wired; 10 others (metabalance, famo, rotograd, imtl_l, rlw, balanced_softmax, tal, ms_tcn_smooth, varifocal, wiou) NOT_FOUND in train.py. |
| Gradient norms | pose=3278, act=13.80, det=1.86, psr=0.16. Ratio now **20,245x** (was 312x). V1 reversal: act > psr. |

**Revised action items from Batch 1:**
- Item 3 (Wire LDAM-DRW): **Completed** — already wired. Flip flag only.
- Item 6 (Verify distillation): **Stub confirmed** — needs implementation, not just verification.

---

## 1. Methodology

D7 continues D2's challenges with deeper investigation into:
- Specific architecture alternatives not yet considered
- Implementation cost vs benefit
- Failure modes of our current architecture

---

## 2. Specific Challenges Continued

### 2.1 Detection Anchor-Free Alternative

**R2 claim:** RetinaNet-style 5.31M with 9 anchors.

**D7 challenge:** Ge et al. (YOLOX, arxiv 2107.08430) report +4.3 mAP on COCO by switching from anchor-based to anchor-free. Our 24-class setup at 224px with sparse labels might benefit MORE.

**Counter-evidence:** Anchor-free requires more careful label assignment. Our TAL assigner module exists but isn't wired. Implementing anchor-free from scratch is 3-5 days.

**Mitigation:** Wire TAL assigner (Task #226 done, just not in active head). Run anchor-free ablation.

### 2.2 Multi-Scale Detection Training

**R7 challenge:** Most SOTA detectors train at multiple resolutions (multi-scale training). We train at fixed 224px.

**Evidence:** Standard practice in YOLOv8, DETR, etc.

**Mitigation:** Implement multi-scale training at 224, 256, 288, 320 (random per batch). Add 1 day of work.

### 2.3 Activity Class-Balanced Sampling

**R1 finding:** Activity has 16 classes with <10 frames.

**D7 challenge:** Without class-balanced sampling (oversampling rare classes), model never learns these. With balanced sampling, model overfits rare classes.

**Solution:** LDAM-DRW (Liu et al., NeurIPS 2019, arxiv 1906.07413) — defer re-weighting to later epochs. Module exists in `src/losses/ldam_drw.py` but status uncertain.

**Batch 1 update:** LDAM-DRW is **already wired** in train.py. No wiring work needed. Simply flip `USE_LDAM_DRW=True` at config.py:1098.

**Mitigation:** Flip the config flag. Run 100 epochs with schedule. Monitor for 1-class collapse (documented risk in original LDAM-DRW comment).

### 2.4 Pose: 6D Rotation Implementation Status

**R2 finding:** `GeometryAwareHeadPose` module exists, gated by `USE_GEO_HEAD_POSE` env flag, default False.

**D7 challenge:** Why is it disabled? Is there a known bug? Or just not tested?

**Verification needed:** Read `USE_GEO_HEAD_POSE` references in `train.py`.

**Implication:** If 6D rotation is implemented but not enabled, we lose the published 30-50% MAE reduction (Zhou et al. CVPR 2019).

**Mitigation:** Enable `USE_GEO_HEAD_POSE=1` env. Run 50-epoch training. Compare MAE.

### 2.5 PSR Sequence Mode Implementation

**R2 finding:** `PSR_SEQUENCE_LENGTH = 8`, `USE_PSR_SEQUENCE_MODE = True`.

**D7 challenge:** PSR sequence mode requires special data loading. Is it wired correctly?

**Verification:** Grep `train.py` for `USE_PSR_SEQUENCE_MODE` usage.

**Implication:** If sequence mode isn't active, our PSR is per-frame only, missing the temporal context that's essential for transition detection.

### 2.6 MTL Architecture: Knowledge Distillation

**R2 finding:** `src/training/distillation.py` exists. Task #261 implemented.

**D7 challenge:** Is distillation actually active? If yes, with what teachers? If no, we're missing a Tier 1 lever.

**Verification:** Grep `train.py` for `--distill-teacher-dir`.

**Batch 1 update:** Distillation is a **bare `pass` stub** in train.py:1567. The distillation forward pass is not implemented. ~50-100 lines needed. This is a Tier 1 implementation task, not a configuration toggle.

---

## 3. Architecture Failure Modes Not Considered

### 3.1 Gradient Checkpointing Trade-offs

**R2 finding:** `USE_BACKBONE_CHECKPOINT=True` for ConvNeXt.

**D7 challenge:** Gradient checkpointing trades 20-30% compute for 50% activation memory. For ConvNeXt at 224px, is this trade-off worth it?

**Counter-evidence:** Without checkpoint, activation memory at batch=6 is ~12 GB. With checkpoint, ~6 GB. RTX 5060 Ti has 16 GB. We could afford batch=10 without checkpoint.

**Mitigation:** A/B test with checkpoint on/off. Measure memory + throughput + final metric.

### 3.2 Numerical Precision: bf16 vs fp16 vs fp32

**R2 finding:** bf16 mixed precision. Documented V1 doc 211 says fp16 caused PSR overflow.

**D7 challenge:** bf16 has 8-bit exponent (vs fp16's 5-bit), so overflow is fixed. But bf16 has 7-bit mantissa (vs fp16's 10-bit), so accumulation precision is lower. For PSR focal loss with small gradients, bf16 might underflow.

**Verification needed:** Profile PSR loss gradients under bf16 vs fp16.

### 3.3 Hardware: RTX 5060 Ti Blackwell vs Ampere

**R2 finding:** RTX 5060 Ti 16GB (Blackwell, compute 12.0), RTX 3060 12GB (Ampere, compute 8.6).

**D7 challenge:** Blackwell has tensor cores that Ampere lacks. Some bf16 ops are 2x faster on Blackwell. This affects timing estimates.

**Mitigation:** Use RTX 5060 Ti for main MTL runs. Use RTX 3060 for ablations. Don't cross-compare latency numbers.

---

## 4. Concrete Action Items

1. **Wire TAL assigner** into detection head (3-5 days)
2. **Implement multi-scale detection training** (1 day)
3. **Flip USE_LDAM_DRW=True** — already wired (0.1 days)
4. **Enable GeometryAwareHeadPose** (0.5 days + 50-epoch run)
5. **Verify PSR sequence mode wiring** (0.5 days)
6. **Implement distillation forward pass** — replace `pass` stub at train.py:1567 (1-2 days)
7. **A/B test gradient checkpointing** (1 day)
8. **A/B test bf16 vs fp32** for PSR (1 day)

---

## 5. Survived Findings

| Claim | Status |
|---|---|
| 46.47M total params | HIGH |
| ConvNeXt-Tiny = 28.59M | HIGH |
| Standard FPN = 4.48M | HIGH |
| Detection = RetinaNet 5.31M | HIGH |
| PSR focal gamma = 0.5 | HIGH |
| PCGrad active | HIGH |

---

## 6. Refined Findings

| Finding | Refinement |
|---|---|
| 6D rotation head pose | Module exists, NOT enabled; enable as Tier 1 |
| LDAM-DRW | **Already wired** — flip `USE_LDAM_DRW=True` at config.py:1098 |
| TAL assigner | Module exists, NOT wired into head; wire as Tier 2 |
| Distillation | **Stub confirmed** — `pass` at train.py:1567; ~50-100 lines needed |
| Module sweep (Batch 1) | 10 modules NOT_FOUND in train.py (metabalance, famo, rotograd, imtl_l, rlw, balanced_softmax, tal, ms_tcn_smooth, varifocal, wiou) |

---

## 7. Output

D7 reveals several Tier 1 opportunities that were "implemented but not used":
1. **GeometryAwareHeadPose** — 0.5 day to enable, 30-50% MAE reduction expected
2. **LDAM-DRW** — **already wired**, just flip `USE_LDAM_DRW=True`. +5-10% tail-class recall expected
3. **TAL assigner** — wire into detection, +3-5 mAP expected
4. **Distillation** — **stub confirmed** at train.py:1567; implement forward pass (~50-100 lines) for +1-3% expected

These are HIGH-VALUE ablations that should be in the paper.
