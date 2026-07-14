# WACV 2024 Comparability Matrix

**Date:** 2026-07-14
**Purpose:** Protocol compatibility analysis between our MTL system and WACV 2024 IndustReal baselines [Schoonbeek et al.]. Informs paper positioning per forbidden-claims list (no SOTA, no head-to-head).

---

## Comparison Matrix

| Dimension | Our MTL | WACV 2024 Baseline | Compatible? |
|-----------|---------|-------------------|-------------|
| **Input resolution** | 224x224 (4.7x down from 1280x720 native) | 1280x720 native (detection), 224x224 (activity/PSR) | **No** for detection; yes for activity/PSR |
| **Input modality** | RGB only | RGB + optical flow (activity MViTv2-S); RGB only (det, PSR) | **No** for activity; yes for det/PSR |
| **Paradigm** | Multi-task (single backbone, 4 heads) | Single-task (independent models per task) | **No** — fundamentally different paradigm |
| **Backbone** | ConvNeXt-Tiny (28.59M) | MViTv2-S (activity), YOLOv8m (det), EfficientNet-B3 (PSR) | **No** — different architectures |
| **Training regime** | MTL with Kendall UW, LDAM-DRW, PCGrad, 3-stage curriculum | Single-task training, no MTL weighting | **No** |
| **GPU** | RTX 3060 12GB / RTX 5060 Ti 16GB | Not specified in WACV paper | N/A |

### Per-Task Protocol Detail

| Task | Our Protocol | WACV 2024 Protocol | Compatibility |
|------|-------------|-------------------|---------------|
| **Activity** | 16-frame clip, majority-vote top-1, 75-way | Clip-level top-1, 75-way (MViTv2-S = 65.25%) | **Partial** — same metric (top-1), but WACV uses multi-modal input (RGB+flow). RGB-only MViTv2-S not reported. |
| **Detection** | mAP50 present-class (report n_present of 24), 224px input | mAP@50, 1280px input (YOLOv8m = 0.838) | **No** — resolution gap is 5.7x linear / 32.7x area. 224px is a known structural ceiling for small-object detection. |
| **PSR** | F1@±3-frame tolerance, transition targets (sigma=3), 11 binary components | F1 (authors' scorer), transition paradigm (B3 = 0.883) | **Partial** — same metric (F1) and same scorer. WACV B3 uses EfficientNet-B3 with clip-level transition prediction; ours uses per-frame with ±3 tolerance. STORM (0.506) uses spatio-temporal tubes. |
| **Head pose** | Geodesic MAE deg + position MAE mm, 6D rotation | None published (novel baseline) | **No comparison exists** — we provide the first head-pose baseline on IndustReal. |

---

## Fair vs Contextual Comparisons

### Fair comparisons (same metric, similar protocol, same paradigm)
- **None.** Every WACV baseline differs from our setup on at least one non-trivial protocol dimension (paradigm, modality, or resolution).

### Contextual-only comparisons (same metric, different protocol — cite with caveats)
- **Activity top-1:** WACV MViTv2-S 65.25% uses RGB + optical flow. Cite as "multi-modal upper bound" not directly comparable to our RGB-only MTL.
- **Detection mAP50:** WACV YOLOv8m 0.838 at 1280px. Cite as "high-resolution single-task reference" noting the 32.7x pixel count difference. Relevant for the limitations section (224px ceiling).
- **PSR F1:** WACV B3 0.883 uses same scorer but clip-level transition paradigm with stronger backbone. STORM 0.506 uses spatio-temporal tube features. Both are single-task. Cite as "single-task PSR references" with paradigm note.

### Informative baselines (same dataset, useful for calibration)
- **STORM PSR 0.506:** A spatio-temporal tube method on the same dataset. Not a direct competitor (different paradigm entirely) but useful as a lower-anchor signal for PSR difficulty.
- **WACV B1-B2 PSR:** Simpler PSR baselines in the WACV paper (lower than B3). If our MTL PSR surpasses B1 or B2, this is worth noting as a calibration point — but still single-task, different protocol.

---

## Recommended Paper Phrasing

### Where to position
After the main results table (Table 1) and before ablations (Table 3), insert a short paragraph + a **protocol comparability note** as a table footnote or a dedicated paragraph in §4.2.

### Specific phrasing

**Option A — Paragraph in §4.2 (recommended for compactness):**

> *Table 1 reports our multi-task results. For context, WACV 2024 [ref] published single-task baselines on IndustReal using different protocols: MViTv2-S activity (65.25% top-1, RGB+optical flow), YOLOv8m detection (0.838 mAP50 at 1280px), and EfficientNet-B3 PSR (0.883 F1, clip-level transition paradigm). These results are cited as contextual references only; differences in input resolution (1280px vs. 224px), modality (RGB+flow vs. RGB-only), and paradigm (single-task vs. multi-task) preclude direct comparison. No head-to-head claim is made.*

**Option B — Table footnote on Table 1 (compact, for space-constrained 8-page limit):**

> *WACV 2024 baselines on IndustReal (activity 65.25%, detection 0.838 mAP50, PSR 0.883 F1) use different protocols: RGB+optical flow (activity), 1280px input (detection), and single-task clip-level PSR. Cited as context only; not directly comparable.*

### For the limitations section (§6)

> *Input resolution (224x224) caps detection performance relative to WACV 2024 baselines which use native 1280px resolution for detection. The 32.7x pixel reduction disproportionately affects small ASD objects, and anchor-free detection at higher resolution is deferred to future work.*

---

## Risk Assessment

| Risk | Description | Mitigation |
|------|-------------|------------|
| Reviewer asks "Why not compare to WACV baselines?" | Expects a direct comparison table | Pre-empt with the comparability paragraph in §4.2 and protocol caveats |
| Reviewer claims SOTA | If our numbers happen to exceed a WACV baseline on any metric | Include forbidden-claims reminder in the paper checklist; use "context only" language |
| Overclaiming on PSR | PSR protocol (per-frame ±3) is most compatible with WACV but still differs | Explicitly state the transition-vs-per-frame difference in §4.2 |
