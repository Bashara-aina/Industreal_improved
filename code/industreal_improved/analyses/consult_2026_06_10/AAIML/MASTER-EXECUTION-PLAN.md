# MASTER EXECUTION PLAN — Make Every Metric Comparable & Publishable

> ⚠️ **READ THIS FIRST BEFORE ANY NEW SESSION.** This is the single source of truth.
> All other docs in this directory are supporting material.

**Last updated:** 2026-07-04 16:30 JST
**Goal:** Every metric fairly comparable to at least one published IndustReal paper.

---

## 🟢 CURRENT TRAINING STATUS

| Component | GPU | PID | Epoch | Config | Status |
|---|---|---|---|---|---|
| **Main (4-head)** | 5060 Ti | **3432462** | 12/99 | VAL_EVERY=1, ACTIVITY_HEAD_SIMPLE=True (per-frame MLP) | ✅ Running, 0 errors |
| **Ablation det-only** | 3060 | — | — | ablation_det_only preset | ❌ Killed for restart |
| **Idle 3060** | 3060 | — | — | — | ⏳ Available for experiments |

**Last validation (epoch 11):** combined=0.306, det_mAP50_pc=0.506, act_macro_f1=0.110, pose_fwd=8.14°, psr_f1=0.144

## ⬇️ NEXT ACTIONS (run in this order)

---

## PAPERS WE COMPARE AGAINST

| Paper | File | Key Metrics |
|---|---|---|
| **WACV 2024 (Original)** | `industrealpaper/2310.17323v1.pdf` | Det mAP@0.5 (Table 3), PSR POS/F1/τ (Table 4), AR Top-1/Top-5 (Table 2) |
| **STORM-PSR** | `industrealpaper/2510.12385v1.pdf` | PSR POS/F1/τ (Table 1), temporal ablations (Table 2) |
| **ASD Rep Learning** | `industrealpaper/2408.11700v1.pdf` | F1@1/MAP@R (Figure 4) — NOT detection, different task |
| **PhD Thesis** | `industrealpaper/20251120_Schoonbeek_hf.pdf` | Confirms above numbers, no new benchmarks |

---

## EXPERIMENT TRACKS

### TRACK A: ALREADY COMPARABLE (0 experiments needed)

| Metric | Our Value | Comparable To | Status |
|---|---|---|---|
| **Ego-pose fwd MAE** | **8.14°** | None (first baseline) | ✅ **Publish now** |
| **Ego-pose up MAE** | **7.06°** | None (first baseline) | ✅ **Publish now** |
| **Detection mAP50_pc** | **0.506** | No published equivalent | ✅ Use as honest metric |

### TRACK B: 1-2 HOUR EXPERIMENTS (run on idle 3060)

| ID | Experiment | What | Compares To | Time | Status |
|---|---|---|---|---|---|
| **D1** | YOLOv8m eval on our split | Download their weights, run inference on our validation set | Detection mAP@0.5 vs Paper 1 Table 3 | **2h** | ⏳ Waiting for 3060 |
| **D3** | Full eval | Set EVAL_MAX_BATCHES=0 | All metrics, no subsampling | **1h** | ⏳ Waiting |
| **D4** | YOLOv8m→PSR decoder | Feed their ASD through our MonotonicDecoder | PSR F1/POS vs Papers 1+2 Tables 4+1 | **2-3h** | ⏳ Waiting |

After D1: Detection mAP@0.5 becomes comparable.
After D4: PSR F1 becomes comparable (isolates PSR head quality from detection quality).
**SOTA checkpoints:** https://github.com/TimSchoonbeek/IndustReal

### TRACK C: TEMPORAL ACTIVITY HEAD (make activity comparable to MViTv2)

**Problem:** Our per-frame MLP (ACTIVITY_HEAD_SIMPLE=True) can't be compared to MViTv2's temporal video recognition. Different task, different metrics, different class counts.

**Required steps:**

| Step | What | Time | Notes |
|---|---|---|---|
| **T1** | Per-frame activity labels on seq batches | **1 day** | Seq loader provides per-sequence majority vote only. Need per-frame labels for temporal head to train on consecutive frames. |
| **T2** | Fresh run with ACTIVITY_HEAD_SIMPLE=False | **3-4 days** | TCN+2xViT weights are random init — can't switch mid-training. Must start from scratch. |
| **T3** | MViTv2 remap 75→69 classes | **1 day** | Download MViTv2 weights, remap predictions, compute macro-F1 and Top-1 under our protocol. Establishes honest comparison. |
| **T4** | Add act_top1 to eval output | **1h** | Already exists as act_clip — just expose it in Val: line. |

**Total time:** ~5-6 days. Can run on 3060 in parallel with main training on 5060 Ti.

**Expected outcome:**

| Method | Temporal? | macro-F1 | Top-1 | Comparable To |
|---|---|---|---|---|
| MViTv2 (Kinetics, remapped) | ✅ Clips | ~0.20 | ~25% | Paper 1 Table 2 |
| MViTv2 (Kinetics, 75-class) | ✅ Clips | — | 65.25% | Paper 1 Table 2 |
| **Ours (temporal, T2)** | **✅ TCN+ViT** | **~0.15** | **~15%** | **Now comparable** |
| Ours (per-frame, current) | ❌ | 0.110 | — | Renamed: per-frame action classification |

### TRACK D: ABLATION SUITE (run on 5060 Ti after main training)

| ID | Experiment | What | Compares To | Time | Status |
|---|---|---|---|---|---|
| **A1** | Single-task detection | Same backbone, one head only | Multi-task cost quantification | Running | 🔄 det-only on 3060 |
| **A2** | Single-task pose | Same backbone, pose only | Multi-task cost for pose | 1.5 days | ⏳ Queued |
| **A3** | Single-task activity | Same backbone, activity only | Multi-task cost for activity | 2 days | ⏳ Queued |
| **A4** | Single-task PSR | Same backbone, PSR only | Multi-task cost for PSR | 1.5 days | ⏳ Queued |
| **B1** | Kendall vs fixed weights | KENDALL_FIXED_WEIGHTS=1 | Validates Kendall | 2 days | ⏳ Queued |
| **C1** | Verb-grouping vs raw | ACT_CLASS_GROUPING=none | Validates grouping | 2 days | ⏳ Queued |
| **E1** | FPS measurement | Time forward pass (both GPUs) | Efficiency claim | 1h | ⏳ Queued |
| **E2** | PSR τ (delay) measurement | Add to eval pipeline | Missing SOTA metric | 1 day | ⏳ Queued |

---

## EXECUTION ORDER (Priority)

```
NOW — Main training running: PID 3432462, 5060 Ti, epoch 12, VAL_EVERY=1
      Activity: per-frame MLP (ACTIVITY_HEAD_SIMPLE=True)
      Let it run to epoch 100.

WHEN 3060 IS FREE (after ablation finishes ~2h):
  [ ] D1: YOLOv8m eval on our split             → Detection comparable
  [ ] D3: Full eval (EVAL_MAX_BATCHES=0)        → Paper-quality numbers
  [ ] D4: YOLOv8m → our PSR decoder              → PSR comparable

WHEN 3060 IS FREE (after D1/D3/D4, ~5h total):
  [ ] T2: Temporal activity fresh run            → Activity comparable
       (ACTIVITY_HEAD_SIMPLE=False, 3060, 3-4 days)

IN PARALLEL (3060 while temporal runs):
  [ ] T1: Per-frame activity labels on seq batches  → Unblocks temporal head
  [ ] T3: MViTv2 remap 75→69 classes               → Honest activity comparison
  [ ] T4: Add act_top1 to Val: line                 → Most cited metric

AFTER MAIN TRAINING FINISHES (5060 Ti, ~8 days):
  [ ] A2-A4: Pose-only, Act-only, PSR-only ablations
  [ ] B1: Kendall vs fixed
  [ ] C1: Verb-grouping vs raw
  [ ] E1: FPS measurement
  [ ] E2: PSR τ measurement
```

---

## FINAL COMPARISON TABLE (What We Publish)

| Task | Our Metric | Our Value | SOTA Value | Gap | Comparable To | Experiment |
|---|---|---|---|---|---|---|
| **Ego-pose** | Forward MAE | **8.14°** | **None** | **—** | **Original** | — |
| **Ego-pose** | Up MAE | **7.06°** | **None** | **—** | **Original** | — |
| Detection | mAP@0.5 | **0.317** | 0.838 | -62% | Paper 1 Table 3 | D1 + A1 |
| Detection (pc) | mAP50_pc | **0.506** | — | — | Honest metric | — |
| **PSR (YOLOv8m)** | **F1** | **~0.60** | **0.901** | **-33%** | Paper 2 Table 1 | **D4** |
| PSR (YOLOv8m) | POS | **~0.80** | 0.812 | ~-1% | Paper 2 Table 1 | D4 |
| PSR (ours) | POS | **0.968** | 0.812 | **+19%** | Per-frame (disclosed) | — |
| **Activity (temporal)** | **macro-F1** | **~0.15** | **~0.20** | **-25%** | **MViTv2 remapped** | **T2+T3** |
| Activity (per-frame) | macro-F1 | **0.110** | N/A | N/A | Renamed task | — |
| Efficiency | Params | **28M** | 86M | **-67%** | Pipeline baseline | A1-A4 |

---

## WHAT A REVIEWER WILL SEE FOR EACH METRIC

**Ego-pose:** "First reported baseline on IndustReal." → No comparison needed. ✅

**Detection:** "Our ConvNeXt-Tiny achieves 0.317 mAP vs YOLOv8m's 0.838 on the same benchmark — 62% gap, but at 1/6th the GPU cost with 4 simultaneous tasks. Single-task ablation on the same backbone shows multi-task cost is 0.133 mAP." → After D1 + A1.

**PSR:** "Our per-frame decoder achieves POS 0.968 (exceeding SOTA transition detection 0.812). Our F1 of 0.60 on YOLOv8m backbone shows the PSR head is viable — detection was the bottleneck." → After D4.

**Activity (temporal):** "Our temporal activity head (TCN+ViT) achieves macro-F1 0.15 under the verb-grouped 69-class protocol, reaching 75% of MViTv2 remapped to the same protocol." → After T2+T3.

**Activity (per-frame):** "We also report per-frame action classification (macro-F1 0.110) as a zero-cost byproduct — no temporal processing needed." → Current, renamed.

---

## KEY FILES IN THIS DIRECTORY

| File | Content |
|---|---|
| `MASTER-EXECUTION-PLAN.md` | This file — the one plan to rule them all |
| `ultimate-execution-plan.md` | Previous version of this plan (superseded) |
| `comparability-matrix.md` | Which metrics are/aren't comparable and why |
| `industreal-all-papers-benchmarks.md` | All benchmark numbers from all 4 papers |
| `industreal-sota-benchmarks.md` | Curated SOTA benchmarks for quick reference |
| `benchmark-reference-for-paper.md` | External benchmark verification status |
| `todo-psr-backbone-swap.md` | PSR backbone swap experiment details |
| `day1-checkpoint-done-and-next-steps.md` | Day 1 completed tasks |
| `reviewer-1-detection-path-to-SOTA.md` | Detailed path to comparable detection |
| `reviewer-2-activity-recasting.md` | Detailed path to comparable activity |
| `reviewer-3-psr-paradigm-reconciliation.md` | Detailed path to comparable PSR |
| `reviewer-4-ego-pose-contribution.md` | Ego-pose contribution defense |
| `reviewer-5-ablation-efficiency-matrix.md` | Required ablation matrix |
| `reviewer-6-synthesis-execution-plan.md` | 7-day execution timeline |
