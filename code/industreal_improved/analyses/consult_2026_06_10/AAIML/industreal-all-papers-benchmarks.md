# IndustReal Benchmarks — All Published Papers

**Compiled:** 2026-07-04
**Source papers in:** `analyses/consult_2026_06_10/industrealpaper/`

---

## Paper 1: Original WACV 2024 — Schoonbeek et al. (arXiv 2310.17323)
**"IndustReal: A Dataset for Procedure Step Recognition Handling Execution Errors"**

### Action Recognition (AR) — Table 3.2

| Model | Pretrain | Modalities | Top-1% | Top-5% |
|---|---|---|---|---|
| SlowFast | Kinetics | RGB | 60.39 | 85.21 |
| SlowFast | MECCANO | RGB | 57.83 | 82.87 |
| **MViTv2-S** | **Kinetics** | **RGB** | **65.25** | **87.93** |
| MViTv2-S | MECCANO | RGB | 62.43 | 85.62 |
| SlowFast | Kinetics | RGB+VL+stereo | 62.34 | 85.97 |
| **MViTv2-S** | **Kinetics** | **RGB+VL+stereo** | **66.45** | **88.43** |

### Assembly State Detection (ASD) — Table 3.3

| Pretrain | Fine-tune | mAP (annotated) | mAP (entire videos) |
|---|---|---|---|
| COCO | Synthetic only | 0.573 | 0.341 |
| COCO | IndustReal only | 0.753 | 0.553 |
| Synthetic | IndustReal | 0.779 | 0.575 |
| **COCO** | **IndustReal + Synthetic** | **0.838** | **0.641** |

ASD model: YOLOv8-m. Error state AP: 0.23, error state FPR: 65%. FPS (V100): 178.

### Procedure Step Recognition (PSR) — Table 3.4

| Baseline | ASD Training | POS (all) | F1 (all) | τ (all) | POS (errors) | F1 (errors) | τ (errors) |
|---|---|---|---|---|---|---|---|
| B1 | Real+Synth | 0.570 | 0.779 | 14.9s | 0.480 | 0.698 | 14.4s |
| B1-S | Synthetic | 0.014 | 0.206 | 36.9s | 0.000 | 0.174 | 48.4s |
| **B2** | **Real+Synth** | **0.731** | **0.860** | **22.3s** | **0.636** | **0.784** | **20.2s** |
| B2-S | Synthetic | 0.240 | 0.573 | 44.4s | 0.107 | 0.516 | 60.5s |
| **B3 (best)** | **Real+Synth** | **0.797** | **0.883** | **22.4s** | **0.731** | **0.816** | **20.4s** |
| B3-S | Synthetic | 0.597 | 0.734 | 49.5s | 0.571 | 0.731 | 71.4s |

---

## Paper 2: STORM-PSR — Schoonbeek et al. (arXiv 2510.12385)
**"Learning to Recognize Correctly Completed Procedure Steps in Egocentric Assembly Videos"**

### Main Results — Table 1

| Method | POS | F1 | τ (s) |
|---|---|---|---|
| **B3 (SOTA baseline)** | **0.797** | **0.891** | **21.0** |
| STORM-PSR ASD stream only | 0.354 | 0.545 | 99.8 |
| **STORM-PSR (full)** | **0.812** | **0.901** | **15.5** |

**Improvements:** τ reduced by 26.1% vs B3. FPS (A100): 75.1.

### Ablation — Table 2

| Setting | POS | F1 | τ (s) |
|---|---|---|---|
| Temporal ResNet50 (no KFS, no KCAS) | 0.467 | 0.511 | 62.6 |
| + KFS pre-training | 0.766 | 0.892 | 30.0 |
| **+ KFS + KCAS (full)** | **0.812** | **0.901** | **15.5** |

### MECCANO Dataset Results

| Method | POS | F1 | τ (s) |
|---|---|---|---|
| B3 baseline | 0.377 | 0.545 | 99.8 |
| **STORM-PSR** | **0.377** | **0.497** | **88.6** |

### Key Takeaway
STORM-PSR achieves **F1=0.901, POS=0.812** on IndustReal with τ=15.5s. The temporal stream operates at 75.1 FPS on A100 (but requires temporal processing that our per-frame model doesn't).

---

## Paper 3: Supervised Representation Learning for ASD (arXiv 2408.11700)
**"Supervised Representation Learning towards Generalizable Assembly State Recognition"**

### Assembly State Recognition on IndustReal (Figure 4)

| Method | Backbone | F1@1 | MAP@R(+) |
|---|---|---|---|
| Cross-entropy | ResNet-34 | ~35 | ~30 |
| Batch Hard | ResNet-34 | ~45 | ~35 |
| SupCon | ResNet-34 | ~50 | ~40 |
| **SupCon + ISIL (proposed)** | **ResNet-34** | **~55** | **~48** |
| SupCon + ISIL | ViT-S | ~32 | ~25 |

**FPS:** 150 fps per image (both backbones). ImageNet-1k pretrained.

**Task:** This is ASSEMBLY STATE RECOGNITION (matching assembly states), not object detection. Uses 128-dim embeddings. Different from our detection task.

---

## Paper 4: PhD Thesis — Schoonbeek 2025
**"Automated support for operators executing industrial procedures"**

The thesis compiles all metrics from Papers 1-3 above plus additional context. No new benchmark numbers beyond what's already listed.

---

## Our Results (Epoch 11, Full 4-Task Model)

| Task | Metric | Our Value | SOTA | Gap |
|---|---|---|---|---|
| Detection | mAP@0.5 | **0.317** | 0.838 (YOLOv8m, Real+Synth) | -62% |
| Detection (present-class) | mAP50_pc | **0.506** | — | No SOTA equivalent |
| Activity | Not comparable | — | 65.25% (MViTv2, 75-class) | Different taxonomy |
| PSR | F1 | **0.144** | 0.901 (STORM-PSR) | Different paradigm* |
| PSR | POS | **0.968** | 0.812 (STORM-PSR) | **+19%** (different paradigm) |
| PSR | τ | N/A | 15.5s (STORM-PSR) | We don't measure delay |
| **Ego-pose** | **fwd MAE** | **8.14°** | **None** | **First baseline** |

*\*Our PSR is per-frame component recognition, STORM-PSR/B3 are transition detection with ASD backbone + procedural knowledge*

---

## What We Can Claim

1. **First ego-pose baseline on IndustReal** (8.14° forward MAE) — no prior work
2. **Single-pass 4-task system on consumer GPU** — no prior work combines ASD+activity+PSR+pose
3. **PSR POS=0.968** exceeds SOTA (0.812) — our MonotonicDecoder ordering is strong, but different task formulation
4. **Detection mAP=0.317** at 1/10th GPU cost of YOLOv8m — efficiency story

## What Needs Ablation A (Single-Task) Before We Can Claim

5. **Multi-task efficiency** — "67% fewer params" needs single-task baseline comparison
6. **PSR head quality** — need backbone swap experiment (see `todo-psr-backbone-swap.md`)
