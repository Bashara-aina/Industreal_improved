# D2 — Architecture Adversarial Debate

**Phase:** ULTIMATE Consultation V2 — Phase 2 Adversarial Debate
**Date:** 2026-07-14
**Debater:** D2 (challenges R2 architecture findings)
**Target:** R2_ARCHITECTURE_VERIFIED.md

---

## 1. Methodology

Challenges R2's architecture claims by:
- Searching for papers where smaller backbones outperform larger ones
- Questioning whether our architecture changes are the bottleneck
- Profiling FLOPs/latency with different batch sizes

---

## 2. Specific Challenges

### 2.1 Is ConvNeXt-Tiny Truly Sufficient for Video MTL?

**R2 claim:** ConvNeXt-Tiny (28.59M, ImageNet) is the active backbone, and TMA + FeatureBank compensate for no temporal modeling.

**Challenge:** Wang et al. (MViTv2, CVPR 2022, arxiv 2112.01526) show that K400-pretrained video transformers dramatically outperform ImageNet-pretrained ConvNets on action recognition (81.0% vs ~75% K400). The 6% gap may translate to 10-15% on IndustReal activity.

**Counter-evidence:** Liu et al. (ConvNeXt, CVPR 2022, arxiv 2201.03545) show ConvNeXt can match Swin Transformer on image tasks. But video is different.

**Test:** Run frozen ConvNeXt vs frozen MViTv2-S on 100 IndustReal activity frames. If ConvNeXt is within 5%, our choice is justified.

**Status:** HIGH confidence in the existence of the gap. MEDIUM confidence in our choice being suboptimal.

### 2.2 BiFPN vs Standard FPN: Are We Leaving mAP on the Table?

**R2 claim:** Standard FPN at 4.48M. BiFPN would add ~3-5M params for ~+0.4-0.7 mAP per Tan et al.

**Challenge:** Tan et al. (EfficientDet, CVPR 2020, arxiv 1911.09070) report BiFPN gains on COCO. Our 24-class assembly detection with sparse labels (17.9%) might benefit MORE than COCO from multi-scale fusion.

**Mitigation:** Run ablation with BiFPN. If +1-2 mAP, swap. If <0.5 mAP, keep standard FPN.

**Status:** MEDIUM confidence. Runnable in 2 days.

### 2.3 Detection Head: RetinaNet is Outdated

**R2 claim:** RetinaNet-style 5.31M with 9 anchors.

**Challenge:** RetinaNet (Lin et al., ICCV 2017, arxiv 1708.02002) is 9 years old. Modern detectors (FCOS, ATSS, YOLOv8) are anchor-free or use better assignment.

**Counter-evidence:** Wang et al. (TOOD, ICCV 2021, arxiv 2108.07755) report +3-5 mAP over RetinaNet via TAL assigner. Ge et al. (YOLOX, arxiv 2107.08430) report +4 mAP via anchor-free + simOTA.

**Implication:** Our detection head is using 2017-era design. Modern alternatives could give +3-5 mAP for similar params.

**Status:** HIGH confidence in opportunity. MEDIUM confidence in cost/benefit.

### 2.4 Activity Head Complexity vs Effectiveness

**R2 claim:** FeatureBank + TCN + 2×ViT (0.69M).

**Challenge:** TMA cell + FeatureBank + TCN + 2×ViT is a lot of machinery for 0.69M params. A simpler architecture (e.g., TCN only or 1×ViT) might match with less compute.

**Test:** Ablate TMA cell, FeatureBank, ViT layers. Measure activity top-1 vs FLOPs.

**Status:** MEDIUM confidence.

### 2.5 Pose Heads: Body Pose is Dead Code, Why Keep It?

**R2 claim:** Body pose head is "effectively dead code" with pseudo-keypoints.

**Challenge:** Body pose feeds PoseFiLM, which modulates C5 features. Even with pseudo-keypoints, this might still help activity recognition.

**Test:** Run ablation with body pose head completely removed (not just frozen). Measure activity top-1 delta.

**Status:** MEDIUM confidence.

---

## 3. Counter-Evidence: Smaller is Sometimes Better

### 3.1 MobileNet-V4 outperforms larger CNNs in some domains

Wightman et al. (ResNet strikes back, 2021) and Howard et al. (MobileNet families) show that properly designed smaller models can match larger ones on specific tasks.

**Implication:** ConvNeXt-Tiny might be sufficient; bigger isn't always better.

### 3.2 ConvNeXt-Tiny on video tasks

Liu et al. (ConvNeXt, CVPR 2022) report 82.1% ImageNet-1K top-1. For video, some works use ConvNeXt as spatial encoder + temporal attention on top. This is essentially our TMA + FeatureBank setup.

**Implication:** Our architecture is well-motivated, but we lack published evidence for ConvNeXt-Tiny + TMA on action recognition.

---

## 4. Verifications Needed

1. **Profile FLOPs/latency** with different batch sizes (claim: 7-12 FPS at batch=1).
2. **Run TOOD-TAL swap** as ablation (Task #245 module exists but not wired).
3. **Run body pose removal** ablation.
4. **BiFPN swap** ablation.

---

## 5. Survived Findings

| Claim | Status |
|---|---|
| 46.47M total params | HIGH (direct measurement) |
| ConvNeXt-Tiny = 28.59M | HIGH |
| Standard FPN = 4.48M | HIGH |
| RetinaNet-style detection | HIGH |
| PSR focal gamma = 0.5 | HIGH |
| PCGrad active | HIGH |

---

## 6. Refined Findings

| Finding | Refinement |
|---|---|
| Activity head is FeatureBank+TCN+2×ViT (0.69M) | May be over-engineered; ablation recommended |
| Detection head is RetinaNet (5.31M) | Modern alternatives may give +3-5 mAP |
| PoseFiLM with pseudo body pose | Impact unclear; ablation recommended |

---

## 7. Output

D2 challenges architecture choices. The biggest opportunities:
1. **BiFPN swap** (+0.4-0.7 mAP est.)
2. **TOOD-TAL or YOLOX anchor-free** (+3-5 mAP est.)
3. **Activity head ablation** (clarify what's necessary)
4. **Body pose removal** (clarify noise impact)

Each is runnable in 2-5 days as Tier 2-3 ablations.
