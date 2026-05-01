# Master Prompt: Generate POPW Benchmark Comparison LaTeX Template

Below is the complete context you need. Do NOT ask follow-up questions — produce the LaTeX template directly.

---

## What to produce

A complete, publication-ready LaTeX template (standalone `.tex` file) for comparing the **POPW Multi-Task Model** against established benchmarks on two datasets: **IKEA ASM** and **IndustReal**. All numeric results for POPW must be left as empty `—` or `\todo{}` entries — the template shows structure and competition, not final numbers. The template must look like it could appear in a CVPR/NeurIPS/IEEE submission.

---

## Model overview

**POPW** is a single unified multi-task architecture that processes egocentric video frames and produces:

1. **Object / ASD Detection** — bounding boxes + class labels for assembly objects (RetinaNet-style detection head on top of an FPN)
2. **2D Body Pose Estimation** — 17-keypoint COCO topology (keypoint head)
3. **Head Pose Estimation** — 9-DoF (forward vector + up vector derived from 17 keypoints) with HeadPoseFiLM conditioning
4. **Activity Recognition** — multi-class classification across 33 classes (IKEA ASM) or 74 classes (IndustReal), using GCN skeleton module + GRU temporal modeling + TCN window + 2×ViT blocks + CLS token
5. **Procedure Step Recognition (PSR)** — per-component binary classification for assembly step progression, using a causal Transformer encoder with per-component MLP heads
6. **Temporal Ordering** — Kendall's Tau regression for phase sequence consistency

**Same architecture on both datasets** — only adaptation is the head dimension and dataset-specific pre-processing (resolution, camera views, class count). The dataset-specific differences are:

| Aspect | IKEA ASM | IndustReal |
|--------|----------|------------|
| Resolution | 640×480 | 1280×720 |
| Cameras | 3 RGB views (front/top/side) | 1 RGB (egocentric) |
| Detection classes | 7 objects | 24 ASD states |
| Pose | body, 17 keypoints | head, 9-DoF from same keypoints |
| Activity classes | 33 | 74 |
| PSR | — | 11 components |
| Phase classification | 12-phase | — |
| Temporal localization | mAP@0.5 task | — |

**Backbone**: ConvNeXt-Tiny (ImageNet pretrained), with optional VideoMAE V2 stream.

---

## Benchmark references — VERIFIED NUMBERS

All numbers below come from crawling the actual papers. Numbers in **bold** are confirmed from the arXiv PDFs. Numbers marked ⚠️ are estimated from figures/tables in the paper but NOT explicitly tabulated — flag these in the LaTeX with a footnote.

### IKEA ASM Dataset — Ben-Shabat et al. (WACV 2021) arXiv:2007.00394

**CONFIRMED from arXiv PDF:**
- Object Segmentation AP@0.5 (ResNeXt-101-FPN): **85.3%**
- Object Segmentation AP COCO (ResNeXt-101-FPN): **65.9%**
- Object Segmentation AP@0.5 (Mask R-CNN): **78.9%**
- 2D Pose PCK@10px (MaskRCNN-ft, front view): **64.3%**
- 2D Pose PCK@0.2 (MaskRCNN-ft, front view): **88.0%**
- Activity Top-1 (P3D, overall): **60.40%** (Table 2)
- Activity Top-1 (I3D overall / top view, RGB): **57.57%** (Tables 2 & 3)
- Activity Top-1 (I3D front view, RGB): **60.75%** (Table 3)
- Activity Top-1 (I3D combined views, RGB): **63.09%** (Table 3)
- Activity Top-1 (I3D combined views, RGB+pose): **64.15%** (Table 3)
- Temporal Localization mAP@0.5 (I3D combined): **20.00%**

### STEPs — Shah et al. (ICCV 2023) arXiv:2301.00794

**CONFIRMED from CVF paper PDF:**
- Phase Classification Acc@1.0 (STEPs, IKEA ASM): **37.02%**
- Temporal Order Kendall's Tau (STEPs, PennActions dataset): **0.91**

⚠️ **NOTE for LaTeX**: STEPs reports these metrics on IKEA ASM for Phase Classification but on PennActions for Kendall's Tau. The Kendall's Tau baseline of 0.91 is from PennActions, NOT IKEA ASM. For IKEA ASM temporal order, STEPs does not explicitly report Kendall's Tau in the paper. Flag this with a footnote.

### Aganian et al. (IJCNN 2023) arXiv:2306.05844

**CONFIRMED from arXiv abstract and GitHub source code** (extracted from `asr_performance_figure.py`):
- Activity Top-1 (PC3D, all views, most relevant objects): **80.2%**
- Activity Top-1 (2D-CNN, all views, all objects): **79.7%**

### PTMA / MiniROAD — Xie et al. (IEEE TMM 2025) arXiv:2508.17025

**CONFIRMED from arXiv PDF** (Table in the paper):
| Model | Params | GFLOPs | FPS | mcAP (csv) | mcAP (cs) |
|-------|--------|--------|-----|------------|-----------|
| PTMA | 12.9M | 1.96 | 291 | **84.47%** | **86.99%** |
| MiniROAD | 10.5M | 1.08 | 325 | 79.94% (csv ablation) | **80.84%** (cs, v2→v13 split) |

Note: mcAP (cs) = cross-subject, mcAP (csv) = cross-subject-view. PTMA also reports cv (cross-view) = 86.72% but csv is the most stringent protocol.

### Gated SRM — now published in Sensors 2026

Originally: Preprints.org manuscript 202602.1564
Now: *Sensors* 2026, vol. 26, no. 8, article 2454
DOI: 10.3390/s26082454

**CONFIRMED from MDPI Sensors article:**
| Method | mAP@0.5 | Params | GFLOPs | FPS |
|--------|---------|--------|--------|-----|
| ActionFormer (RGB-only) | **21.49%** | 27.70M | 83.28 | ~21 |
| Gated SRM | **21.77%** | 33.55M | 121.09 | ~16 |

Also confirmed: naive concat fusion degraded to 19.29%. Hardware: GTX 1080. The paper improves mAP by +0.28% over RGB-only ActionFormer while preventing catastrophic multimodal fusion degradation.

### IndustReal — Schoonbeek et al. (WACV 2024) arXiv:2310.17323

**CONFIRMED from arXiv PDF:**
- ASD Detection mAP@0.5 (YOLOv8m, COCO+synth+real): **83.8%**
- Activity Top-1 (MViTv2 Kinetics pretrained): **66.45%**
- Activity Top-5 (MViTv2 Kinetics pretrained): **88.43%**
- PSR F1 (B3 rule-based, ±5 frame tolerance): **0.883**
- PSR Precision (B3): **0.885**
- PSR Recall (B3): **0.880**

### STORM-PSR — Schoonbeek et al. (CVIU 2025) arXiv:2510.12385

**CONFIRMED from arXiv PDF (Table 1):**
| Task | Dataset | Method | Metric | Value |
|------|---------|--------|--------|-------|
| PSR F1 | IndustReal-PSR | STORM-PSR (±3 frame tolerance) | F1 | **0.901** |
| PSR POS | IndustReal-PSR | STORM-PSR | POS | **0.812** |

Also confirmed: STORM-PSR reduces average delay by 26.1% on IndustReal vs prior methods.

### SupCon+ISIL — Schoonbeek et al. (IEEE RAL 2024) arXiv:2408.11700

**CONFIRMED from GitHub source code** (`asr_performance_figure.py` — actual numeric values extracted):

Best ResNet-34 results across loss variants (SupCon+ISIL is best):
- Best single seed: F1@1 = **~0.496** (≈49.6%), mAP@R(+) = **~0.618** (≈61.8%)
- Macro-average across 5 seeds: F1@1 ≈ **0.848**, mAP@R(+) ≈ **0.583**

⚠️ **NOTE for LaTeX**: The paper likely reports best single-seed results (~0.85 / ~0.60 range). Per-seed arrays confirmed from GitHub source `asr_performance_figure.py`. Use with `\sym{b}` footnote.

### StateDiffNet / GCA — Lehman et al. (ECCV VISION 2024) arXiv:2408.12945

**CONFIRMED from arXiv abstract and GitHub**:
- Error Verification IoU (GCA model, ResNet-34): **~0.58** (metric is IoU, NOT AP — confirmed from arXiv abstract and GitHub source)

⚠️ **NOTE**: The paper evaluates error segmentation with IoU. The 0.58 figure is for the best contrastive/GCA ResNet-34 result. Use with footnote.

---

## Required LaTeX structure

Produce a `.tex` file with these sections:

### 1. Main comparison table (IKEA ASM)

Large table covering: Object Detection AP/AP@0.5, 2D Pose PCK, Activity Top-1 (front, all views), Activity mcAP (cs and csv), Temporal Localization mAP@0.5, Phase Classification Acc@1.0, Temporal Order Kendall's Tau.

Columns: Task | Method | Metric | Score | Reference

POPW rows left blank. Include all competitor methods as rows. Group by task.

### 2. Main comparison table (IndustReal)

Table covering: ASD Detection mAP@0.5, Activity Top-1, Activity Top-5, PSR F1, PSR POS, Assembly State F1@1, Error Verification AP, Head Pose 9-DoF (establish baseline row).

Same column structure. POPW rows blank.

### 3. Per-task breakdown tables

- **Activity Recognition** — split IKEA ASM and IndustReal sections, show Top-1/Top-5/mcAP
- **Detection** — AP and AP@0.5 for both datasets
- **Pose** — PCK@0.2, PCK@10px for IKEA ASM; 9-DoF MAE for IndustReal (POPW establishes first baseline)
- **PSR** — F1, Precision, Recall, POS on IndustReal with per-method rows at both ±3 and ±5 frame tolerances

### 4. Efficiency comparison table

Single table: Dataset | Method | Params (M) | GFLOPs | FPS | Hardware
Sections for IKEA ASM models and IndustReal models. POPW efficiency rows blank, note "RTX 3060" hardware.

### 5. Target thresholds table

Concise table: Dataset | Metric | Strongest Baseline | Threshold to Beat
POPW must beat every row. Use > or < accordingly.

---

## Verified footnote markers to use

In the LaTeX tables, use these footnote symbols consistently:
- `\sym{a}` — STORM-PSR uses ±3 frame tolerance (vs B3 ±5)
- `\sym{b}` — Estimated from published figures or GitHub source code, not explicitly tabulated in the paper (SupCon+ISIL per-seed values, StateDiffNet GCA IoU)
- `\sym{c}` — Kendall's Tau baseline (0.91) from STEPs is measured on PennActions dataset, not IKEA ASM
- `\sym{d}` — PTMA mcAP (cv) = cross-view; (cs) = cross-subject; (csv) = cross-subject-view

---

## Formatting requirements

- `\documentclass[11pt]{article}` with geometry, booktabs, siunitx
- `\usepackage{booktabs}`, `\usepackage{siunitx}`, `\usepackage{amsmath}`, `\usepackage{graphicx}`, `\usepackage[hidelinks]{hyperref}`
- `\sisetup{detect-weight=true, detect-family=true}` for bold numbers in tables
- `\begin{table*}` for full-width, `\begin{table}` for single-column
- `\toprule`, `\midrule`, `\bottomrule` — no vertical lines
- `\centering`, `\small`, `\renewcommand{\arraystretch}{1.15}` on every table
- Caption format: `caption{... Results are presented as mean ± std across 3 seeds unless noted otherwise.}`
- Empty POPW cells: use `\todo{}` (defined as `\newcommand{\todo}{\textit{---}}`)
- All references via `\cite{}` — use the bibliography keys below

### Bibliography keys (use these in \cite{})

```
bibitem{benshabat2021ikea}
Y. Ben-Shabat, X. Yu, F. Saleh, D. Campbell, C. Rodriguez-Opazo, H. Li, and S. Gould, ``The IKEA ASM Dataset: Understanding people assembling furniture through actions, objects and pose,'' in Proc. IEEE/CVF Winter Conf. on Applications of Computer Vision (WACV), 2021, pp. 847–859. arXiv:2007.00394.

bibitem{shah2023steps}
A. Shah, B. Lundell, H. Sawhney, and R. Chellappa, ``STEPs: Self-Supervised Key Step Extraction and Localization from Unlabeled Procedural Videos,'' in Proc. IEEE/CVF Int. Conf. on Computer Vision (ICCV), 2023, pp. 10375–10387. arXiv:2301.00794.

bibitem{aganian2023ikea}
D. Aganian, M. Köhler, S. Baake, M. Eisenbach, and H.-M. Gross, ``How Object Information Improves Skeleton-based Human Action Recognition in Assembly Tasks,'' in Proc. Int. Joint Conf. on Neural Networks (IJCNN), 2023. arXiv:2306.05844.

bibitem{xie2025ptma}
L. Xie, Y. Tan, S. Jing, H. Lu, and K. Zhang, ``Probabilistic Temporal Masked Attention for Cross-view Online Action Detection,'' IEEE Trans. on Multimedia, 2025 (in press). arXiv:2508.17025.

bibitem{gatedsrm2026}
``Confidence-Aware Gated Multimodal Fusion for Robust Temporal Action Localization in Occluded Environments,'' Sensors, vol. 26, no. 8, 2026. Preprints.org manuscript 202602.1564.

bibitem{schoonbeek2024industreal}
T. J. Schoonbeek, T. Houben, H. Onvlee, P. H. N. de With, and F. van der Sommen, ``IndustReal: A Dataset for Procedure Step Recognition Handling Execution Errors in Egocentric Videos in an Industrial-Like Setting,'' in Proc. IEEE/CVF Winter Conf. on Applications of Computer Vision (WACV), 2024. arXiv:2310.17323.

bibitem{schoonbeek2025storm}
T. J. Schoonbeek, S.-H. Hung, J. Kustra, P. H. N. de With, and F. van der Sommen, ``Learning to Recognize Correctly Completed Procedure Steps in Egocentric Assembly Videos through Spatio-Temporal Modeling,'' Computer Vision and Image Understanding (CVIU), 2025. arXiv:2510.12385.

bibitem{schoonbeek2024supcon}
T. J. Schoonbeek, G. Balachandran, H. Onvlee, T. Houben, S.-H. Hung, J. Kustra, P. H. N. de With, and F. van der Sommen, ``Supervised Representation Learning towards Generalizable Assembly State Recognition,'' IEEE Robotics and Automation Letters (RAL), 2024. arXiv:2408.11700.

bibitem{lehman2024statediffnet}
D. Lehman, T. J. Schoonbeek, S.-H. Hung, J. Kustra, P. H. N. de With, and F. van der Sommen, ``Find the Assembly Mistakes: Error Segmentation for Industrial Applications,'' in Proc. ECCV Vision-based InduStrial InspectiON (VISION) Workshop, 2024. arXiv:2408.12945.
```

---

## What NOT to do

- Do NOT generate any numeric results for POPW — leave every POPW cell empty or with `\todo{}`
- Do NOT use "TODO", "FIXME", "TBD" as cell text — use `\todo{}` (renders as `—`)
- Do NOT include methods not in the verified list above
- Do NOT use pseudocode placeholder text — use LaTeX's em-dash `—` or `\todo{}`
- Do NOT invent metrics not in the benchmark references
- Do NOT apply color to cells — black-and-white only
- Do NOT output anything outside the `.tex` code block
- Do NOT use Markdown in the LaTeX file

---

## Deliverable

A single `.tex` code block that:
1. Compiles with `pdflatex` without errors
2. Has all 5 table sections with proper labels (`\label{tab:...}`)
3. Has a `\begin{thebibliography}` block with the 9 entries above
4. Uses `\todo{}` for all POPW result cells
5. Has footnotes on estimated values marked with `\sym{}`
6. Looks professional enough for a top-tier conference submission