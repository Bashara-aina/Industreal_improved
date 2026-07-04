# PLAN: Make Our Architecture Comparable to ASD Rep Learning & AR Top-1/Top-5

**Date:** 2026-07-04
**Goal:** Enable fair comparisons to Paper 3 (ASD Rep Learning, arXiv 2408.11700) and Paper 1 Table 2 (Action Recognition, MViTv2/SlowFast)

---

## Part A: ASD Representation Learning (Paper 3)

### What Paper 3 Does

| Dimension | Paper 3 | Us (current) |
|---|---|---|
| **Task** | Assembly state recognition via embedding retrieval | Object detection (bounding boxes) |
| **Output** | 128-dim embedding vector per image | Bounding boxes + class labels |
| **Metric** | F1@1, MAP@R (nearest-neighbor retrieval) | mAP@0.5 (intersection-over-union) |
| **Backbone** | ResNet-34, ViT-S (ImageNet pretrained) | ConvNeXt-Tiny (random init) |
| **Training** | Contrastive learning (SupCon + ISIL) | Supervised detection loss |
| **Data** | IndustReal frames (unlabeled+labeled) | IndustReal frames (labeled only) |

### They are DIFFERENT tasks — but we CAN compare embeddings

We can't compare detection mAP to retrieval F1@1 — they measure different things. But we CAN:
1. **Extract embeddings from our ConvNeXt backbone** (before the detection head)
2. **Run Paper 3's evaluation protocol** on those embeddings
3. **Report F1@1 and MAP@R** — now directly comparable

### Experiment R1: Embedding Extraction & Retrieval Eval (2-3 days)

**Steps:**

| Step | What | Time |
|---|---|---|
| R1a | Extract 128-dim embeddings from ConvNeXt backbone (before FPN/detection head) for all validation set frames | 1h |
| R1b | Implement nearest-neighbor retrieval: for each test image, find the closest training embedding by cosine similarity | 1 day |
| R1c | Compute F1@1 and MAP@R using Paper 3's definition (Figure 4) | 1h |
| R1d | Compare: our ConvNeXt embeddings vs their ResNet-34/ViT-S | — |

**Paper 3 baseline numbers:**

| Backbone | Method | F1@1 | MAP@R(+) |
|---|---|---|---|
| ResNet-34 | SupCon + ISIL (best) | ~55 | ~48 |
| ResNet-34 | SupCon | ~50 | ~40 |
| ResNet-34 | Batch Hard | ~45 | ~35 |
| ResNet-34 | Cross-entropy | ~35 | ~30 |
| ViT-S | SupCon + ISIL | ~32 | ~25 |

**Expected outcome:** Our ConvNeXt-Tiny (random init, no contrastive pretraining) will likely achieve F1@1 ≈ 20-35. Below their ResNet-34 (ImageNet pretrained + contrastive learning), but competitive with ViT-S. This is expected — we're not optimized for embedding retrieval.

**Paper narrative:** "Our ConvNeXt backbone, trained only with detection supervision, achieves F1@1=X on assembly state retrieval — within Y% of specialist contrastive methods trained specifically for this task."

---

## Part B: Action Recognition Top-1/Top-5 (Paper 1 Table 2)

### What Paper 1 Table 2 Reports

| Model | Pretrain | Modalities | Top-1% | Top-5% |
|---|---|---|---|---|
| SlowFast | Kinetics | RGB | 60.39 | 85.21 |
| SlowFast | MECCANO | RGB | 57.83 | 82.87 |
| **MViTv2-S** | **Kinetics** | **RGB** | **65.25** | **87.93** |
| MViTv2-S | MECCANO | RGB | 62.43 | 85.62 |
| SlowFast | Kinetics | RGB+VL+stereo | 62.34 | 85.97 |
| MViTv2-S | Kinetics | RGB+VL+stereo | **66.45** | **88.43** |

### What We Currently Have

| Model | Temporal? | Classes | Metrics | Comparable? |
|---|---|---|---|---|
| Our per-frame MLP | ❌ No | 69 verb-grouped | macro-F1=0.110 | ❌ NOT comparable |
| Our temporal (T2) | ✅ TCN+ViT | 69 verb-grouped | TBD | ⚠️ After T2+T3+T4 |

### What's Needed (Track C — already planned)

| ID | Step | Makes What Possible | Time |
|---|---|---|---|
| **T2** | Fresh run with ACTIVITY_HEAD_SIMPLE=False | Enables TCN+ViT temporal processing | **3-4 days** |
| **T3** | MViTv2 remap 75→69 classes | Establishes honest baseline under our protocol | **1 day** |
| **T4** | Add act_top1 to Val: line | Enables Top-1 reporting (most cited metric) | **1h** |
| **A3** | Single-task activity run | Multi-task interference quantification | **2 days** |

### The Remaining Gaps After T2+T3+T4

| Gap | Can We Close It? | How |
|---|---|---|
| **Kinetics pretraining** | Yes — add ImageNet pretrain to ConvNeXt | 1 config change, costs ~0.02 mAP |
| **Temporal context** | Yes — TCN+ViT (T2) | Already built, just needs fresh run |
| **75 vs 69 classes** | Yes — remap (T3) | 1 day, already planned |
| **Multi-modal (RGB+VL+stereo)** | No — hardware limitation | Disclose as limitation |
| **Top-1 metric** | Yes — T4 (act_top1) | 1h, already planned |

### Expected Comparison Table (after all experiments)

| Method | Temporal? | Top-1% | macro-F1 | Classes | Pretrain | GPU Cost |
|---|---|---|---|---|---|---|
| MViTv2 (Paper 1 SOTA) | ✅ 16 clips | **65.25** | — | 75 | Kinetics | $2,500+ |
| MViTv2 remapped (T3) | ✅ 16 clips | **~25** | **~0.20** | 69 | Kinetics | $2,500+ |
| **Ours temporal (T2)** | **✅ TCN+ViT** | **~15** | **~0.15** | **69** | **None** | **$429** |
| Ours per-frame (current) | ❌ | — | 0.110 | 69 | None | $429 |

**Paper narrative for activity:** *"Our temporal activity head (TCN+ViT) achieves macro-F1 0.15 under verb-grouped 69-class protocol — reaching 75% of MViTv2 remapped to the same protocol (0.20), without Kinetics pretraining and at 1/6th the GPU cost."*

### Additional Experiment: Kinetics-Pretrain Our Activity Head (optional, 1 day)

If we want to close the remaining gap, we can initialize ConvNeXt-Tiny with ImageNet-1k weights (already available in torchvision) and rerun the temporal activity experiment. Expected gain: +0.02-0.05 macro-F1. This would make the comparison even stronger.

---

## Updated Master Plan

### New experiments added to the queue:

| ID | Experiment | Time | Makes What Comparable | Paper |
|---|---|---|---|---|
| **R1** | Embedding extraction + retrieval eval | **2-3 days** | ASD Rep Learning F1@1/MAP@R | Paper 3 |
| **T2** | Temporal activity fresh run | **3-4 days** | AR Top-1/Top-5 | Paper 1 Table 2 |
| **T3** | MViTv2 remap 75→69 | **1 day** | AR baseline protocol | Paper 1 Table 2 |
| **T4** | Add act_top1 to Val: line | **1h** | Top-1 metric | Paper 1 Table 2 |

### Resource allocation:

| GPU | Priority | Experiment | Duration |
|---|---|---|---|
| **3060 (after D1/D3/D4)** | **P1** | **R1: Embedding retrieval** | **2-3 days** |
| **3060 (after R1)** | **P1** | **T2: Temporal activity fresh run** | **3-4 days** |
| **3060 (during T2)** | **P1** | T3: MViTv2 remap (cpu-only) | 1 day |
| **5060 Ti (after main)** | P2 | Ablation A + efficiency | 5 days |

---

## Bottom Line

| Paper | Metric | Comparable After | Time |
|---|---|---|---|
| **Paper 3 (ASD Rep Learning)** | **F1@1 / MAP@R** | **R1: embedding extraction** | **2-3 days** |
| **Paper 1 Table 2 (AR)** | **Top-1 / Top-5** | **T2: temporal head + T3: remap** | **5 days** |

Both are achievable. Add ~1 week to the timeline.
