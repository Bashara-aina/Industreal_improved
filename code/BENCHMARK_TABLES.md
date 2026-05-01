# POPW Multi-Task Model — Benchmark Tables

---

## Tables Overview

| Table # | Name | # Metrics | Purpose |
|---------|------|-----------|---------|
| **1** | Activity Recognition Comparison | 14 | Compare Top-1, Top-5, mcAP, Temporal Localization, Phase Classification across both datasets |
| **2** | Object Detection Comparison | 4 | Compare AP, mAP@0.5 for assembly object segmentation (IKEA ASM) and ASD (IndustReal) |
| **3** | Pose & Head Pose Comparison | 4 | Compare PCK metrics (IKEA ASM) and 9-DoF head pose ( IndustReal) |
| **4** | Procedure Step Recognition (PSR) Comparison | 7 | Compare F1, POS, Acc for procedure steps on IndustReal |
| **5** | Efficiency Comparison | 6 | Compare Params, GFLOPs, FPS across both datasets |

---

## TABLE 1: Activity Recognition Comparison

### IKEA ASM Benchmarks

| Dataset | Metric | Competitor | Score to Beat | Paper Link | POPW Target |
|---------|--------|-------------|---------------|------------|-------------|
| **IKEA ASM** | Top-1 (RGB front view) | P3D | 60.4% | [Ben-Shabat et al. (WACV 2021)](https://arxiv.org/abs/2007.00394) | >60.4% |
| **IKEA ASM** | Top-1 (RGB+pose front) | I3D RGB+pose | 64.15% | [Ben-Shabat et al. (WACV 2021)](https://arxiv.org/abs/2007.00394) | >64.15% |
| **IKEA ASM** | Top-1 (all views) | I3D | 47.0% | [Ben-Shabat et al. (WACV 2021)](https://arxiv.org/abs/2007.00394) | >47.0% |
| **IKEA ASM** | Temporal Localization mAP@0.5 | Gated SRM | 21.77% | [Preprints.org 202602.1564](https://www.preprints.org/manuscript/202602.1564) | >21.77% |
| **IKEA ASM** | mcAP (csv) | PTMA | 84.47% | [Xie et al. (IEEE TMM 2025)](https://arxiv.org/abs/2508.17025) | >84.47% |
| **IKEA ASM** | mcAP (cs) | PTMA | 86.99% | [Xie et al. (IEEE TMM 2025)](https://arxiv.org/abs/2508.17025) | >86.99% |
| **IKEA ASM** | Phase Classification Acc@1.0 | STEPs (self-supervised) | 37.02% | [Shah et al. (ICCV 2023)](https://arxiv.org/abs/2301.00794) | >37.02% |
| **IKEA ASM** | Temporal Order Kendall's Tau | STEPs | 0.91 | [Shah et al. (ICCV 2023)](https://arxiv.org/abs/2301.00794) | >0.91 |
| **IKEA ASM** | Action Recognition Top-1 (all views, most relevant) | PC3D | 80.2% | [Aganian et al. (IJCNN 2023)](https://arxiv.org/abs/2306.05844) | >80.2% |

### IndustReal Benchmarks

| Dataset | Metric | Competitor | Score to Beat | Paper Link | POPW Target |
|---------|--------|-------------|---------------|------------|-------------|
| **IndustReal** | Top-1 | MViTv2 Kinetics | 66.45% | [Schoonbeek et al. (WACV 2024)](https://arxiv.org/abs/2310.17323) | >66.45% |
| **IndustReal** | Top-5 | MViTv2 Kinetics | 88.43% | [Schoonbeek et al. (WACV 2024)](https://arxiv.org/abs/2310.17323) | >88.43% |
| **IndustReal** | Assembly State Recognition F1@1 | SupCon+ISIL (ResNet-34) | ~0.85 est. | [Schoonbeek et al. (IEEE RAL 2024)](https://arxiv.org/abs/2408.11700) | >baseline |
| **IndustReal** | Assembly State MAP@R(+) | SupCon+ISIL (ResNet-34) | baseline | [Schoonbeek et al. (IEEE RAL 2024)](https://arxiv.org/abs/2408.11700) | >baseline |
| **IndustReal** | Error Verification AP (ResNet-34) | GCA model | ~0.58 est. | [Lehman et al. (ECCV VISION 2024)](https://arxiv.org/abs/2408.12945) | >baseline |

---

## TABLE 2: Object Detection Comparison

### IKEA ASM Benchmarks

| Dataset | Metric | Competitor | Score to Beat | Paper Link | POPW Target |
|---------|--------|-------------|---------------|------------|-------------|
| **IKEA ASM** | Object Segmentation AP@0.5 | ResNeXt-101-FPN | 85.3% | [Ben-Shabat et al. (WACV 2021)](https://arxiv.org/abs/2007.00394) | >85.3% |
| **IKEA ASM** | Object Segmentation AP (COCO) | ResNeXt-101-FPN | 65.9% | [Ben-Shabat et al. (WACV 2021)](https://arxiv.org/abs/2007.00394) | >65.9% |
| **IKEA ASM** | Object Segmentation AP@0.5 | Mask R-CNN | 78.9% | [Ben-Shabat et al. (WACV 2021)](https://arxiv.org/abs/2007.00394) | >78.9% |

### IndustReal Benchmarks

| Dataset | Metric | Competitor | Score to Beat | Paper Link | POPW Target |
|---------|--------|-------------|---------------|------------|-------------|
| **IndustReal** | ASD Detection mAP@0.5 | YOLOv8m COCO+synth+real | 83.8% | [Schoonbeek et al. (WACV 2024)](https://arxiv.org/abs/2310.17323) | >83.8% |

---

## TABLE 3: Pose & Head Pose Estimation Comparison

### IKEA ASM Benchmarks

| Dataset | Metric | Competitor | Score to Beat | Paper Link | POPW Target |
|---------|--------|-------------|---------------|------------|-------------|
| **IKEA ASM** | 2D Pose PCK@10px | MaskRCNN-ft | 64.3% | [Ben-Shabat et al. (WACV 2021)](https://arxiv.org/abs/2007.00394) | >64.3% |
| **IKEA ASM** | 2D Pose PCK@0.2 | MaskRCNN-ft | 88.0% | [Ben-Shabat et al. (WACV 2021)](https://arxiv.org/abs/2007.00394) | >88.0% |

### IndustReal Benchmarks

| Dataset | Metric | Competitor | Score to Beat | Paper Link | POPW Target |
|---------|--------|-------------|---------------|------------|-------------|
| **IndustReal** | Head Pose (9-DoF) MAE | No supervised baseline | evaluate vs raw GT | [Schoonbeek et al. (WACV 2024)](https://arxiv.org/abs/2310.17323) | establish baseline |
| **IndustReal** | Head Pose Forward Vector Error | No baseline | evaluate vs raw GT | [Schoonbeek et al. (WACV 2024)](https://arxiv.org/abs/2310.17323) | establish baseline |

---

## TABLE 4: Procedure Step Recognition (PSR) Comparison

| Dataset | Metric | Competitor | Score to Beat | Paper Link | POPW Target |
|---------|--------|-------------|---------------|------------|-------------|
| **IndustReal** | PSR F1 | B3 rule-based | 0.883 | [Schoonbeek et al. (WACV 2024)](https://arxiv.org/abs/2310.17323) | >0.883 |
| **IndustReal** | PSR Precision | B3 rule-based | 0.885 | [Schoonbeek et al. (WACV 2024)](https://arxiv.org/abs/2310.17323) | >0.885 |
| **IndustReal** | PSR Recall | B3 rule-based | 0.880 | [Schoonbeek et al. (WACV 2024)](https://arxiv.org/abs/2310.17323) | >0.880 |
| **IndustReal-PSR** | PSR F1 | STORM-PSR | 0.901 | [Schoonbeek et al. (CVIU 2025)](https://arxiv.org/abs/2510.12385) | >0.901 |
| **IndustReal-PSR** | PSR POS | STORM-PSR | 0.812 | [Schoonbeek et al. (CVIU 2025)](https://arxiv.org/abs/2510.12385) | >0.812 |

---

## TABLE 5: Efficiency Comparison

### IKEA ASM Models

| Dataset | Model | Accuracy | Metric | Params (M) | GFLOPs | FPS | Hardware | Paper |
|---------|-------|---------|--------|-----------|--------|-----|---------|-------|
| **IKEA ASM** | PTMA | 84.47% | mcAP (csv) | 12.9M | 1.96 | 291 | — | [arXiv:2508.17025](https://arxiv.org/abs/2508.17025) |
| **IKEA ASM** | MiniROAD | 80.84% | mcAP (cs avg) | 10.5M | 1.08 | 325 | — | [arXiv:2508.17025](https://arxiv.org/abs/2508.17025) |
| **IKEA ASM** | ActionFormer (RGB-only) | 21.49% | mAP@0.5 | 27.70 | 83.28 | ~21 | GTX 1080 | [Preprints.org](https://www.preprints.org/manuscript/202602.1564) |
| **IKEA ASM** | Concat (naive fusion) | 19.29% | mAP@0.5 | 28.56 | 87.25 | ~21 | GTX 1080 | [Preprints.org](https://www.preprints.org/manuscript/202602.1564) |
| **IKEA ASM** | Gated SRM (proposed) | 21.77% | mAP@0.5 | 33.55 | 121.09 | ~16 | GTX 1080 | [Preprints.org](https://www.preprints.org/manuscript/202602.1564) |

### IndustReal Models

| Dataset | Model | Accuracy | Metric | Params (M) | GFLOPs | FPS | Hardware | Paper |
|---------|-------|---------|--------|-----------|--------|-----|---------|-------|
| **IndustReal** | MViTv2 Kinetics | 66.45% | Top-1 | — | — | — | — | [arXiv:2310.17323](https://arxiv.org/abs/2310.17323) |
| **IndustReal** | YOLOv8m COCO+synth+real | 83.8% | ASD mAP@0.5 | — | — | — | — | [arXiv:2310.17323](https://arxiv.org/abs/2310.17323) |
| **IndustReal** | B3 rule-based | 0.883 | PSR F1 | — | — | — | — | [arXiv:2310.17323](https://arxiv.org/abs/2310.17323) |
| **IndustReal** | STORM-PSR | 0.901 | PSR F1 | — | — | — | — | [arXiv:2510.12385](https://arxiv.org/abs/2510.12385) |

---

## POPW Target Summary

### IKEA ASM Targets

| Task | Metric | Beat | Baseline |
|------|--------|------|----------|
| Object Segmentation AP@0.5 | AP@0.5 | >85.3% | ResNeXt-101-FPN |
| Object Segmentation AP (COCO) | AP | >65.9% | ResNeXt-101-FPN |
| 2D Pose PCK@10px | PCK@10px | >64.3% | MaskRCNN-ft |
| 2D Pose PCK@0.2 | PCK@0.2 | >88.0% | MaskRCNN-ft |
| Activity Top-1 (RGB+pose) | Top-1 | >64.15% | I3D RGB+pose |
| Activity Top-1 (RGB front) | Top-1 | >60.4% | P3D |
| Activity Top-1 (all views) | Top-1 | >80.2% | PC3D (Aganian) |
| Temporal Localization | mAP@0.5 | >21.77% | Gated SRM |
| Activity mcAP (csv) | mcAP | >84.47% | PTMA |
| Activity mcAP (cs) | mcAP | >86.99% | PTMA |
| Phase Classification | Acc@1.0 | >37.02% | STEPs self-sup |
| Temporal Order | Kendall's Tau | >0.91 | STEPs |

### IndustReal Targets

| Task | Metric | Beat | Baseline |
|------|--------|------|----------|
| ASD Detection | mAP@0.5 | >83.8% | YOLOv8m COCO+synth+real |
| Activity Top-1 | Top-1 | >66.45% | MViTv2 Kinetics |
| Activity Top-5 | Top-5 | >88.43% | MViTv2 Kinetics |
| PSR F1 | F1 | >0.901 | STORM-PSR |
| PSR POS | POS | >0.812 | STORM-PSR |
| Assembly State Recognition | F1@1 | >baseline | SupCon+ISIL |
| Error Verification | AP | >baseline | GCA model |
| Head Pose (9-DoF) | MAE | vs raw GT | no supervised baseline |

---

## What POPW Needs to Report

### Accuracy Metrics
- [ ] Pose: **PCK@0.2**, **PCK@10px** (on IKEA ASM)
- [ ] Head Pose: **9-DoF MAE**, forward vector error (on IndustReal)
- [ ] Activity: **Top-1**, **Top-5**, **mcAP** (on both datasets)
- [ ] Detection: **AP@0.5**, **mAP@0.5** (on both datasets)
- [ ] PSR: **F1**, **Precision**, **Recall**, **POS** (on IndustReal)
- [ ] Assembly State: **F1@1**, **MAP@R(+)** (on IndustReal)
- [ ] Error Verification: **AP** (on IndustReal)
- [ ] Phase Classification: **Acc@1.0** (on IKEA ASM)
- [ ] Temporal Order: **Kendall's Tau** (on IKEA ASM)
- [ ] Temporal Localization: **mAP@0.5** (on IKEA ASM)

### Efficiency Metrics
- [ ] **Parameters (M)** — report for both IKEA ASM and IndustReal configs
- [ ] **GFLOPs** — at dataset native resolution (640x480 IKEA, 1280x720 IndustReal)
- [ ] **FPS** — on RTX 3060
- [ ] **Latency (ms)** — per-frame inference time

---

## Example Table Format for POPW Results

### IKEA ASM Results

| Task | Dataset | Metric | POPW | Best Competitor | Improvement |
|------|---------|--------|------|-----------------|-------------|
| Activity mcAP | IKEA ASM | mcAP (csv) | ?% | 84.47% (PTMA) | +?% |
| Activity mcAP | IKEA ASM | mcAP (cs) | ?% | 86.99% (PTMA) | +?% |
| Temporal Loc | IKEA ASM | mAP@0.5 | ?% | 21.77% (Gated SRM) | +?% |
| Activity | IKEA ASM | Top-1 | ?% | 64.15% (I3D RGB+pose) | +?% |
| Activity | IKEA ASM | Top-1 (all views) | ?% | 80.2% (PC3D) | +?% |
| Pose | IKEA ASM | PCK@0.2 | ?% | 88.0% (MaskRCNN-ft) | +?% |
| Detection | IKEA ASM | AP@0.5 | ?% | 85.3% (ResNeXt-101-FPN) | +?% |

### IndustReal Results

| Task | Dataset | Metric | POPW | Best Competitor | Improvement |
|------|---------|--------|------|-----------------|-------------|
| Activity | IndustReal | Top-1 | ?% | 66.45% (MViTv2) | +?% |
| Activity | IndustReal | Top-5 | ?% | 88.43% (MViTv2) | +?% |
| Detection | IndustReal | ASD mAP@0.5 | ?% | 83.8% (YOLOv8m) | +?% |
| PSR F1 | IndustReal | F1 | ? | 0.901 (STORM-PSR) | +? |
| PSR POS | IndustReal | POS | ? | 0.812 (STORM-PSR) | +? |
| Head Pose | IndustReal | 9-DoF MAE | ?° | raw GT (no baseline) | — |

### Efficiency Results

| Model | Dataset | Params | GFLOPs | FPS | Metric |
|-------|---------|--------|--------|-----|--------|
| POPW (IKEA) | IKEA ASM | ?M | ?G | ? | mcAP (csv) |
| POPW (Indust) | IndustReal | ?M | ?G | ? | Top-1 / ASD mAP |

---

## Paper Quick Links

| # | Paper | Dataset | Link |
|---|-------|---------|------|
| 1 | Ben-Shabat et al. — IKEA ASM Dataset (WACV 2021) | IKEA ASM | https://arxiv.org/abs/2007.00394 |
| 2 | Shah et al. — STEPs (ICCV 2023) | IKEA ASM | https://arxiv.org/abs/2301.00794 |
| 3 | Aganian et al. — Skeleton+Object fusion (IJCNN 2023) | IKEA ASM | https://arxiv.org/abs/2306.05844 |
| 4 | Xie et al. — PTMA/MiniROAD (IEEE TMM 2025) | IKEA ASM | https://arxiv.org/abs/2508.17025 |
| 5 | Confidence-Aware Gated Multimodal Fusion (Preprints.org) | IKEA ASM | https://www.preprints.org/manuscript/202602.1564 |
| 6 | Schoonbeek et al. — IndustReal (WACV 2024) | IndustReal | https://arxiv.org/abs/2310.17323 |
| 7 | Schoonbeek et al. — STORM-PSR (CVIU 2025) | IndustReal | https://arxiv.org/abs/2510.12385 |
| 8 | Schoonbeek et al. — SupCon Representation Learning (IEEE RAL 2024) | IndustReal | https://arxiv.org/abs/2408.11700 |
| 9 | Lehman et al. — StateDiffNet error segmentation (ECCV VISION 2024) | IndustReal | https://arxiv.org/abs/2408.12945 |

---

## Key Benchmark Findings

### Best Efficiency (Params vs Accuracy on IKEA ASM)

| Model | Params | GFLOPs | FPS | Accuracy | Metric |
|-------|--------|--------|-----|----------|--------|
| MiniROAD | 10.5M | 1.08 | 325 | 80.84% | mcAP (cs) |
| PTMA | 12.9M | 1.96 | 291 | 84.47% | mcAP (csv) |
| Gated SRM | 33.55M | 121.09 | ~16 | 21.77% | mAP@0.5 |
| ActionFormer | 27.70M | 83.28 | ~21 | 21.49% | mAP@0.5 |

**Interpretation**: PTMA/MiniROAD use a completely different task (online action detection mcAP) compared to ActionFormer/Gated SRM (temporal action localization mAP@0.5). These are not directly comparable — they measure different aspects of temporal understanding.

### POPW Positioning

POPW is a **single unified architecture** (ConvNeXt-Tiny + FPN + multi-task heads) that adapts to both datasets:

**Same architecture across IKEA ASM and IndustReal:**
- ConvNeXt-Tiny backbone (ImageNet pretrained)
- Feature Pyramid Network (FPN) for multi-scale detection
- Shared pose/keypoint head (COCO 17-keypoint topology)
- Activity head with GCN skeleton module + GRU temporal modeling
- Temporal ordering head (Kendall's Tau loss)
- Identical loss functions, optimizers, and training schedules

**Dataset-specific adaptations only:**
- IKEA ASM: body pose (17 keypoints), 33 action classes, 7 object classes, phase classification, temporal localization
- IndustReal: head pose (9-DoF forward+up vectors from 17 keypoints), 74 action classes, 24 ASD states, PSR task
- Resolution: 640×480 (IKEA ASM) vs 1280×720 (IndustReal)
- Single RGB camera (IndustReal) vs tri-camera dev3 (IKEA ASM)

No single benchmark paper covers all these tasks. The combined targets represent the state-of-the-art across each individual task.