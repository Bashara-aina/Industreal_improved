# Reviewer 1: Detection (ASD) — Path to SOTA-Comparable Metrics

## Identity: IEEE/CVF Reviewer — Object Detection & Multi-Task Learning
**Focus:** Fair comparison protocols, ablation rigor, detection SOTA landscape.
**Bias:** Will desk-reject papers that compare apples to oranges. Demands matched evaluation.

---

## 1. Current Status

| Metric | Our Value (Epoch 11) | SOTA (YOLOv8m, WACV 2024) | Gap |
|---|---|---|---|
| mAP@0.5 (annotated frames) | 0.317 | 0.838 | -62% |
| mAP@0.5 (entire videos) | — | 0.641 | — |
| mAP50_pc (present-class) | 0.506 | — | No SOTA equivalent |
| Backbone | ConvNeXt-Tiny (28M) | YOLOv8-m (25M) | Comparable params |
| Pretraining | None (random init) | COCO + 100K synth | Massive advantage |
| Tasks | 4 simultaneous | 1 (detection only) | Our advantage |
| GPU cost | $429 (5060 Ti) | $2,500+ (V100) | 6× cheaper |

---

## 2. The Comparability Problem

**A reviewer will say:** *"Your mAP of 0.317 vs their 0.838 is a 62% gap. Why should I care?"*

**There are 4 confounders separating us from a fair comparison:**

| Confounder | Effect on mAP | Can We Control It? |
|---|---|---|
| **No COCO pretrain** | COCO pretrain typically adds +0.05-0.10 mAP | ✅ Yes — can add ImageNet pretrain |
| **No synthetic data** | 100K synth images adds +0.085 mAP (0.753→0.838) | ❌ Hard — need Unity pipeline |
| **Multi-task interference** | 4 tasks compete for backbone, estimated -0.05 to -0.15 | ✅ Yes — Ablation A (single-task run) |
| **Different eval protocol** | Our val vs their test, 250 batches vs full | ✅ Yes — run full eval |
| **No temporal context** | Single-frame vs temporal | ❌ Architectural |

**Honest baseline if all confounders were removed:** ~0.45-0.55. Still below 0.838, but the efficiency narrative holds.

---

## 3. Required Experiments (Ordered by Impact ÷ Time)

### Experiment D1: YOLOv8m Eval on Our Split (2 hours, idle 3060)

**What:** Download their YOLOv8m weights from the IndustReal repo, run inference on our validation split, compute mAP with our evaluation script.

**Why:** This is the single highest-leverage experiment. It answers: *"What mAP does a SOTA detector get on our data split?"* If YOLOv8m gets 0.838 on our split too → our 0.317 is measured against the same benchmark. If YOLOv8m gets lower (different val split) → our gap is smaller than advertised.

**Risks:** None. Offline, 2 hours, no training needed.

### Experiment D2: ImageNet-Pretrain Our Backbone (2 days, 5060 Ti after main training)

**What:** Initialize ConvNeXt-Tiny with ImageNet-1k weights, re-run RF4. Compare mAP.

**Why:** Removes the "no pretrain" confounder. Expected gain: +0.05-0.10 mAP.

### Experiment D3: Full Eval (1 hour, change config)

**What:** Set `EVAL_MAX_BATCHES=0` (evaluate on full validation set, not 250-batch subset). The `det_n_present=15/24` issue (9 classes missing) is likely a sampling artifact per Doc 109 Q7.

**Why:** Our validation metrics may be underestimated due to subsampling. Full eval gives paper-quality numbers.

### Experiment D4: Ablation A — Single-Task Detection (already running on 3060, ~12h remaining)

**What:** `ablation_det_only` preset. Same architecture, same data, only detection head.

**Why:** Isolates multi-task interference. If single-task achieves mAP=0.45 and multi-task achieves 0.317 → the gap (0.45 - 0.317 = 0.133) IS the true multi-task cost, and we can openly claim it.

---

## 4. Expected Outcomes & Paper Framing

| Scenario | Detection mAP | Paper Narrative |
|---|---|---|
| YOLOv8m on our split: 0.838 → We're at 0.317 | 62% gap | "62% below YOLOv8m at 1/10th cost, with 3 extra tasks free" |
| YOLOv8m on our split: 0.650 → We're at 0.317 | 51% gap | "51% below at 1/10th cost" — even better |
| Single-task: 0.45, Multi-task: 0.317 | -0.13 gap | "Multi-task cost: 0.133 mAP (29% relative)" |
| Single-task: 0.35, Multi-task: 0.317 | -0.03 gap | "Multi-task almost free for detection" |

### The Key Table the Paper Must Have

| | Method | mAP@0.5 | mAP50_pc | FPS | Params | Tasks | GPU Cost |
|---|---|---|---|---|---|---|---|
| SOTA | YOLOv8m (WACV 2024) | **0.838** | — | 178 | 25M | 1 | $2,500+ |
| Baseline | ConvNeXt-Tiny single-task | **~0.45** | ~0.60 | ~50 | ~28M | 1 | $429 |
| **Ours** | **POPW multi-task** | **0.317** | **0.506** | **~50** | **~28M** | **4** | **$429** |
| | *Multi-task cost* | *-0.133* | *-0.094* | — | — | — | — |
| | *GPU cost savings* | — | — | — | — | — | *6× cheaper* |

---

## 5. Timeline & Recommendation

| Priority | Experiment | Time | Why |
|---|---|---|---|
| **P0** | D1: YOLOv8m eval on our split | 2h | Establishes honest benchmark — do TODAY |
| **P0** | D3: Full eval (EVAL_MAX_BATCHES=0) | 1h | Paper-quality numbers — do today |
| **P1** | D4: Ablation A (single-task) | Already running | Multi-task interference quantification |
| **P2** | D2: ImageNet pretrain | 2 days | Narrows gap, but may not change narrative |

**Bottom line from Reviewers:** A 62% gap to YOLOv8m is NOT a fatal problem. The paper's thesis is *multi-task efficiency*, not *detection SOTA*. But you must: (a) quantify exactly HOW MUCH of the gap is multi-task vs pretraining vs architecture, and (b) never claim "competitive with YOLOv8m." Claim "4-task single-pass efficiency at 1/6th the GPU cost."
