# Comparability Matrix: Our Metrics vs Published SOTA

**Date:** 2026-07-04
**Goal:** One definitive reference for which comparisons survive peer review and which don't.

---

## Detection (ASD) — ⚠️ PARTIALLY COMPARABLE

| Dimension | Our Value (Epoch 11) | SOTA (YOLOv8m, WACV 2024) |
|---|---|---|
| Metric | mAP@0.5 = **0.317** | mAP@0.5 = **0.838** |
| Present-class | mAP50_pc = **0.506** | — (not reported) |
| Backbone | ConvNeXt-Tiny (random init) | YOLOv8-m (COCO→Real+Synth) |
| Tasks | 4 simultaneous | 1 (detection only) |
| GPU cost | $429 | $2,500+ (V100) |
| Data | Real IndustReal frames only | Real + 100K synthetic Unity |

**Verdict:** Same metric, same dataset. But different pretraining, task count, and GPU. 
**→ Comparable after YOLOv8m eval on our split (Experiment D1).**
**→ Comparable after Ablation A (single-task on same backbone).**

---

## Activity (AR) — ❌ NOT COMPARABLE (Category Error)

| Dimension | Our Value | SOTA (MViTv2, WACV 2024) |
|---|---|---|
| Metric | macro-F1 = **0.110**, top-5 = **0.398** | Top-1 = **65.25%**, Top-5 = **87.93%** |
| Classes | 69 verb-grouped | 75 fine-grained |
| Temporal | Per-frame (no temporal context) | 16-frame clips (MViTv2-S) |
| Pretrain | None (random init) | Kinetics-400 |
| Modality | RGB only | RGB + VL + stereo (ensemble) |
| **Task name** | **Per-frame action classification** | **Video action recognition** |

**Verdict:** Fundamentally different tasks — different class counts, temporal processing, metrics, pretraining, and modalities. All comparisons removed from docs. Renamed to "per-frame action classification."

---

## PSR — ❌ NOT COMPARABLE (Paradigm Difference)

| Dimension | Our Value | SOTA (STORM-PSR, 2025) | SOTA (B3, WACV 2024) |
|---|---|---|---|
| F1@±3 | **0.144** | **0.901** | **0.883** |
| POS | **0.968** | **0.812** | **0.797** |
| τ (delay) | N/A | 15.5s | 22.4s |
| **Task** | **Per-frame component state** | **Transition detection** | **Transition detection** |
| Backbone mAP | 0.317 (ours) | 0.838 (YOLOv8m) | 0.838 (YOLOv8m) |
| Temporal | None | Transformer | Confidence accumulation |
| Proc. knowledge | None | Yes | Yes |

**Verdict:** Same metric names (F1, POS) but fundamentally different paradigm. Our POS is higher but it's a metric artifact from the MonotonicDecoder fill-forward constraint. We do not measure τ (delay). The SOTA pipelines use temporal + procedural knowledge + a 4× stronger detection backbone.
**→ Partially comparable after YOLOv8m backbone swap (Experiment D4).**

---

## Ego-Pose — ✅ FIRST BASELINE (Original Contribution)

| Dimension | Our Value (Epoch 11) | Prior Work |
|---|---|---|
| Forward MAE | **8.14°** | **None — first baseline** |
| Up MAE | **7.06°** | None — first baseline |
| Position | ⚠️ UNRELIABLE (see evaluate.py:1918) | N/A |
| Task type | Ego-pose (HoloLens wearer) | N/A |
| Comparison to OpenFace/6DRepNet | ❌ Category error — removed | ❌ Removed from all docs |

**Verdict:** First reported ego-pose baseline on IndustReal. Position values not reportable. Forward/up MAE alone is publishable. Do NOT compare to face-based head pose estimators.

---

## Efficiency — ✅ COMPARABLE (After Ablation A)

| Metric | 4× Single-Task (est.) | Our Multi-Task | Savings |
|---|---|---|---|
| Total params | ~112M (4 × 28M) | **~28M** | **~67% fewer** |
| Inference passes | 4 | **1** | **~75% fewer** |
| GPU cost | ~$1,716 (4×$429) | **$429** | **~75% cheaper** |
| Det mAP@0.5 | ~0.45 (est.) | **0.317** | -29% (multi-task cost) |
| Ego-pose MAE | ~7° (est.) | **8.14°** | +16% (multi-task cost) |

**Verdict:** Parameter savings are real (~67%, not 31% as previously claimed). But need Ablation A numbers for the single-task baseline.

---

## Summary Table

| Task | Our Metric | Comparable to SOTA? | What's Needed |
|---|---|---|---|
| **Detection mAP@0.5** | 0.317 | ⚠️ After D1 + Ablation A | YOLOv8m eval + single-task |
| **Detection mAP50_pc** | 0.506 | ✅ Use as honest metric | No SOTA equivalent exists |
| **Activity macro-F1** | 0.110 | ❌ NEVER | Renamed to per-frame classification |
| **PSR F1** | 0.144 | ⚠️ After D4 | Backbone swap w/ YOLOv8m |
| **PSR POS** | 0.968 | ⚠️ After D4 (disclose artifact) | Backbone swap + paradigm note |
| **Ego-pose fwd MAE** | **8.14°** | ✅ **FIRST BASELINE** | Nothing — publishable now |
| **Efficiency** | ~67% less params | ✅ After Ablation A | Single-task baseline comparison |
