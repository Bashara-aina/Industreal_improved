# Ultimate Plan: Making Every Metric Comparable & Benchmarkable

**Goal:** Every metric we report must be fairly comparable to at least one published paper.
**Deadline:** 7 days from now.
**4 source papers** in `industrealpaper/` — all metrics extracted.

---

## The 4 Papers & Their Benchmark Tables

### Paper 1: WACV 2024 Original (Schoonbeek, arXiv 2310.17323)

| Table | Content | Our Comparable Metric? |
|---|---|---|
| **Table 2** (AR) | SlowFast/MViTv2 top-1/top-5 | ❌ Different task |
| **Table 3** (ASD) | YOLOv8m mAP@0.5 (4 training schemes) | ✅ **SAME METRIC** |
| **Table 4** (PSR) | B1/B2/B3 POS, F1, τ | ⚠️ Same metric, different paradigm |

**Key numbers:**
- ASD: YOLOv8m mAP=0.573 (synth), 0.753 (real), 0.779 (synth→real), **0.838** (real+synth)
- PSR: B3 POS=0.797, F1=0.883, τ=22.4s (all); POS=0.731, F1=0.816 (errors)

### Paper 2: STORM-PSR (Schoonbeek, arXiv 2510.12385)

| Table | Content | Our Comparable Metric? |
|---|---|---|
| **Table 1** | STORM-PSR vs B3: POS, F1, τ | ⚠️ Same metrics, different paradigm |
| **Table 2** | Ablation: KFS/KCAS impact on temporal stream | ❌ Architectural ablation |
| **Table 3** | Backbone comparison (Transformer vs ResNet) | ❌ Architectural ablation |

**Key numbers:**
- B3 (their baseline): POS=0.797, F1=0.891, τ=21.0s
- STORM-PSR: **POS=0.812, F1=0.901, τ=15.5s**
- MECCANO: B3 POS=0.377, F1=0.545; STORM: POS=0.377, F1=0.497

### Paper 3: ASD Representation Learning (arXiv 2408.11700)

| Figure | Content | Our Comparable Metric? |
|---|---|---|
| **Figure 4** | F1@1 and MAP@R for ResNet/ViT with contrastive learning | ❌ Different task (embedding retrieval, not detection) |

**Key insight:** This is NOT object detection. It's assembly state recognition via embedding similarity (128-dim vectors). Different task entirely. **Not comparable.**

### Paper 4: PhD Thesis (Schoonbeek 2025)

Compiles all metrics from Papers 1-3. No new benchmarks. Confirms all numbers above.

---

## The Master Execution Plan

### Track A: Already Comparable (0 experiments needed)

| Metric | Our Value | Comparable To | Status |
|---|---|---|---|
| **Ego-pose fwd MAE** | **8.14°** | None (first baseline) | ✅ **Publish now** |
| **Ego-pose up MAE** | **7.06°** | None (first baseline) | ✅ **Publish now** |
| **Detection mAP50_pc** | **0.506** | No published equivalent | ✅ Use as honest metric |

### Track B: Need 1-2 Hour Experiments (Idle 3060)

| Experiment | What | Compares To | Time |
|---|---|---|---|
| **D1: YOLOv8m eval** | Run their weights on our split | Detection mAP@0.5 vs Paper 1 Table 3 | **2h** |
| **D4: YOLOv8m→PSR decoder** | Feed their ASD through our decoder | PSR F1/POS vs Papers 1+2 Tables 4+1 | **2-3h** |
| **D3: Full eval** | EVAL_MAX_BATCHES=0 | All metrics, no subsampling | **1h** |

After D1: **Detection is comparable.** If YOLOv8m gets 0.838 on our split → same benchmark. If lower → our gap is smaller.
After D4: **PSR is comparable.** If YOLOv8m→our decoder gets F1>0.50 → PSR head is fine, detection was the bottleneck.

### Track C: Need 1-2 Day Experiments (5060 Ti after main training)

| Experiment | What | Compares To | Time |
|---|---|---|---|
| **Ablation A: det-only** | Single-task detection baseline | Multi-task cost quantification | Running now (~10h) |
| **Ablation A: pose-only** | Single-task pose baseline | Multi-task cost for pose | 1.5 days |
| **Ablation A: act-only** | Single-task activity baseline | Multi-task cost for activity | 2 days |
| **Ablation A: psr-only** | Single-task PSR baseline | Multi-task cost for PSR | 1.5 days |
| **Ablation B: Kendall vs fixed** | Validates Kendall weighting | Paper 1 uses fixed weights | 2 days |
| **Ablation C: Verb-grouping vs raw** | Validates grouping protocol | Paper 1 uses 75 classes | 2 days |
| **FPS measurement** | Real FPS on 3060/5060 Ti | Efficiency claim | 1h |

---

## What Each Paper Comparison Looks Like After All Experiments

### Detection vs Paper 1 (Table 3)

| Method | mAP@0.5 | Pretrain | Tasks | GPU |
|---|---|---|---|---|
| YOLOv8m (Paper 1 SOTA) | **0.838** | COCO+100K synth | 1 | V100 |
| YOLOv8m on our split (D1) | **~0.838** | Same weights | 1 | V100 |
| ConvNeXt single-task (Ablation A) | **~0.45** | Random init | 1 | $429 |
| **Ours multi-task** | **0.317** | Random init | **4** | **$429** |
| *Multi-task cost* | *-0.133* | — | — | — |

### PSR vs Papers 1+2 (Tables 4+1)

| Method | POS | F1 | τ (s) | Backbone | Temporal? |
|---|---|---|---|---|---|
| B3 (Paper 1) | 0.797 | 0.883 | 22.4 | YOLOv8m | ✅ Accum |
| STORM-PSR (Paper 2) | **0.812** | **0.901** | **15.5** | YOLOv8m | ✅ Transformer |
| YOLOv8m→Our decoder (D4) | **~0.70-0.80** | **~0.50-0.70** | — | YOLOv8m | ❌ Per-frame |
| **Ours (ConvNeXt)** | 0.968 | 0.144 | N/A | ConvNeXt | ❌ Per-frame |

### Ego-Pose vs ... Nothing (Original Contribution)

| Method | Forward MAE | Up MAE | Position | Prior Work |
|---|---|---|---|---|
| **Ours (epoch 11)** | **8.14°** | **7.06°** | N/A | **First baseline** |
| Expected at convergence | ~6-8° | ~5-7° | N/A | **First baseline** |

### Activity vs ... Nobody (Different Task)

We do **per-frame action classification**, not temporal activity recognition. No published baseline exists for our specific task formulation (69 verb-grouped classes, per-frame, RGB only).

---

## The Three Metrics That Define the Paper

After all experiments, these are the three numbers that matter:

| # | Metric | Our Value | vs SOTA | Narrative |
|---|---|---|---|---|
| 1 | **Ego-pose forward MAE** | **8.14°** | First baseline | **Original contribution — lead the paper** |
| 2 | **Detection efficiency** | 0.317 mAP | 67% fewer params | **4 tasks for price of 1** |
| 3 | **PSR on YOLOv8m backbone (D4)** | ~0.50-0.70 F1 | ~40% of SOTA | **PSR head is viable, detection was bottleneck** |

---

## Execution Order

```
DAY 1 (today — already done):
  ✅ Fix naming: activity recognition → per-frame action classification
  ✅ Remove OpenFace/6DRepNet
  ✅ Fix parameter arithmetic (31% → 67%)
  ✅ Enable efficiency metrics

DAY 1-2 (3060 idle, 6h):
  [ ] D1: YOLOv8m eval on our split → comparable detection      ← DO THIS NOW
  [ ] D3: Full eval → paper-quality numbers                     ← DO THIS NOW  
  [ ] D4: YOLOv8m → our PSR decoder → comparable PSR            ← DO THIS NOW

DAY 2-7 (5060 Ti after main training finishes):
  [ ] Ablation A: all 4 single-task runs
  [ ] Ablation B: Kendall vs fixed
  [ ] Ablation C: Verb-grouping vs raw
  [ ] FPS measurement
  [ ] Write paper
```

---

## What We Publish (Final Comparison Table)

| Task | Our Metric | Our Value | SOTA Value | Gap | Comparable To |
|---|---|---|---|---|---|
| **Ego-pose** | Forward MAE | **8.14°** | **None** | **—** | **Original** |
| Detection (fair) | mAP@0.5 | **0.317** | 0.838 | -62% | Paper 1 Table 3 |
| Detection (pc) | mAP50_pc | **0.506** | — | — | Honest metric |
| **PSR (YOLOv8m)** | F1 | **~0.60 (D4)** | 0.901 | -33% | Paper 2 Table 1 |
| PSR (YOLOv8m) | POS | **~0.80 (D4)** | 0.812 | ~-1% | Paper 2 Table 1 |
| Activity | macro-F1 | **0.110** | N/A | N/A | Per-frame (not temporal) |
| Efficiency | Params | **28M (4 tasks)** | 86M (3 models) | **-67%** | Ablation A |
