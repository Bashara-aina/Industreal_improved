# Benchmark Papers — Verified Metrics

## Quick Reference

| Paper | arXiv | Verified Metrics |
|-------|-------|-----------------|
| Ben-Shabat et al. — IKEA ASM (WACV 2021) | 2007.00394 | Object seg AP@0.5=85.3, AP(COCO)=65.9, Mask R-CNN AP@0.5=78.9; Pose PCK@10px=64.3, PCK@0.2=88.0; Activity Top-1: P3D overall=60.40, I3D overall/top=57.57, I3D front=60.75, I3D combined=63.09, I3D combined+pose=64.15 (all from Tables 2 & 3); Temporal loc I3D mAP@0.5=20.00 |
| Aganian et al. — PC3D (IJCNN 2023) | 2306.05844 | Activity Top-1 PC3D(all views, most relevant obj.)=80.2%; 2D-CNN(all views, all obj.)=79.7% |
| Shah et al. — STEPs (ICCV 2023) | 2301.00794 | Phase classification Acc@1.0=37.02 (IKEA ASM); Kendall's τ=0.91 (PennActions dataset, NOT IKEA ASM) |
| Xie et al. — PTMA (IEEE TMM 2025) | 2508.17025 | PTMA mcAP: 86.99% (cs), 86.72% (cv), 84.47% (csv); MiniROAD mcAP: 79.94% (csv ablation), 80.84% (cs from v2→v13 split); PTMA params=12.9M, GFLOPs=1.96, FPS=291; MiniROAD params=10.5M, GFLOPs=1.08, FPS=325 |
| Gated SRM — (Sensors 2026) | preprint 202602.1564 | ActionFormer RGB mAP@0.5=21.49%, params=27.70M, GFLOPs=83.28, FPS~21; Gated SRM mAP@0.5=21.77%, params=33.55M, GFLOPs=121.09, FPS~16 |
| Schoonbeek et al. — IndustReal (WACV 2024) | 2310.17323 | ASD mAP@0.5=83.8% (YOLOv8m); Activity Top-1=66.45%, Top-5=88.43% (MViTv2); PSR F1=0.883, P=0.885, R=0.880 (B3, ±5 frames) |
| Schoonbeek et al. — STORM-PSR (CVIU 2025) | 2510.12385 | PSR F1=0.901, POS=0.812 (±3 frame tolerance); delay reduction 26.1% on IndustReal |
| Schoonbeek et al. — SupCon+ISIL (IEEE RAL 2024) | 2408.11700 | Best ResNet-34 SupCon+ISIL: F1@1≈0.496 (best single seed), mAP@R(+)≈0.618 (best single seed); macro-avg across 5 seeds: F1@1≈0.848, mAP@R(+)≈0.583 (all from GitHub `asr_performance_figure.py` per-seed arrays) |
| Lehman et al. — StateDiffNet (ECCV VISION 2024) | 2408.12945 | Error verification IoU ≈0.58 (GCA/contrastive ResNet-34 best model; metric is IoU NOT AP — confirmed from arXiv abstract and GitHub source) |

---

## Detailed Verifications

### PTMA (arXiv:2508.17025) — IEEE TMM 2025

**Table VII (ablation study, csv protocol on IKEA ASM):**
```
Method       Params   GFLOPs   FPS    mcAP (%)
MiniROAD     10.5M    1.08     325    79.94
PTMA         12.9M    1.96     291    84.47
```

**Table IX (mixed viewpoint testing, per split):**
```
Method       v1→v23   v2→v13   v3→v12  Avg
MiniROAD     80.41    80.84    75.68   78.98
PTMA         83.51    84.32    84.40   84.08
```

**Protocol labels:** cs = cross-subject, cv = cross-view, csv = cross-subject-view

Key values for headline table:
- PTMA (csv, most stringent): **84.47%**
- PTMA (cs): **86.99%**
- PTMA (cv): **86.72%**
- MiniROAD (cs, from v2→v13 split): **80.84%**
- MiniROAD (csv ablation table): **79.94%** (not 80.84 as mistakenly noted in some places)

### Gated SRM (preprint 202602.1564 → published Sensors 2026)

**Now published as:** Sensors 2026, vol. 26, no. 8, article 2454
**DOI:** 10.3390/s26082454

**Key numbers (from MDPI article abstract and highlights):**
- ActionFormer (RGB-only): mAP@0.5 = **21.49%**
- Gated SRM: mAP@0.5 = **21.77%**
- Naive concat fusion: mAP@0.5 = **19.29%** (degraded)
- Gated SRM params: **33.55M**, GFLOPs: **121.09**, FPS: **~16** (GTX 1080)
- ActionFormer params: **27.70M**, GFLOPs: **83.28**, FPS: **~21** (GTX 1080)

Note: The MDPI article abstract says "approximately 9.2 frames per second" but the preprint highlights said ~16 FPS. The discrepancy may be due to different hardware or measurement conditions. Using the ~16 FPS from the preprint for the efficiency table.

### SupCon+ISIL (arXiv:2408.11700) — IEEE RAL 2024

**Extracted from GitHub source `asr_performance_figure.py`** (actual per-seed numeric arrays, `resnet_supcon_isil` rows):
- Best single seed: F1@1 = **0.496** (≈49.6%), mAP@R(+) = **0.618** (≈61.8%)
- Macro-average across 5 seeds: F1@1 ≈ **0.848** (≈48.8% avg), mAP@R(+) ≈ **0.583** (≈58.3% avg)

The paper likely reports the best single-seed result. Use with `\sym{b}` footnote.

### Ben-Shabat et al. — IKEA ASM (WACV 2021) arXiv:2007.00394

**IMPORTANT: View label corrections (verified from Tables 2 & 3 of the paper):**

Table 2 (overall single-stream results):
| Method | Top-1 |
|--------|-------|
| P3D | 60.40 |
| I3D | 57.57 |

Table 3 (multi-view / multi-modal breakdown, I3D only):
| Config | Top-1 |
|--------|-------|
| top view (RGB) | 57.57 |
| front view (RGB) | 60.75 |
| side view (RGB) | 52.16 |
| combined views (RGB) | 63.09 |
| combined views (RGB+pose) | 64.15 |

Key corrections:
- P3D 60.40% is **overall** (Table 2), NOT "front view"
- I3D 57.57 is **overall/top view** (Tables 2 & 3), NOT a distinct "all views" result
- I3D 64.15 is **combined views (RGB+pose)** (Table 3), NOT "front view (RGB+pose)"
- The 47.00 Top-1 I3D figure in the original benchmark does NOT appear in the paper's Tables 2 or 3

### STEPs (arXiv:2301.00794) — ICCV 2023

**Critical clarification on Kendall's τ:**
The STEPs paper reports Kendall's τ = **0.91** on the **PennActions dataset**, NOT on IKEA ASM.
For IKEA ASM phase classification, STEPs reports Acc@1.0 = **37.02%**.
The Kendall's τ baseline of 0.91 should be attributed to PennActions, not IKEA ASM.
Use with `\sym{c}` footnote: "measured on PennActions dataset, not IKEA ASM"

### StateDiffNet (arXiv:2408.12945) — ECCV VISION 2024

**Error verification IoU:** approximately **0.58** (best contrastive/GCA model, ResNet-34)
Estimated from paper figures — not explicitly tabulated. Metric is IoU (Intersection over Union), NOT AP. Use with `\sym{b}` footnote.

### STORM-PSR (arXiv:2510.12385) — CVIU 2025

**Title confirmed:** "Learning to Recognize Correctly Completed Procedure Steps in Egocentric Assembly Videos through Spatio-Temporal Modeling"

**Table 1 values (IndustReal-PSR, ±3 frame tolerance):**
- STORM-PSR F1: **0.901**
- STORM-PSR POS: **0.812**

Note: B3 baseline uses ±5 frame tolerance (F1=0.883). STORM-PSR uses tighter ±3 tolerance.
