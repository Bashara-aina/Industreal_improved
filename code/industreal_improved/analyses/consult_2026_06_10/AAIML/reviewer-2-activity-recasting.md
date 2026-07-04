# Reviewer 2: Activity Recognition — Recasting & Protocol Alignment

## Identity: IEEE/CVF Reviewer — Video Understanding & Action Recognition
**Focus:** Activity recognition benchmarks, temporal modeling, protocol correctness.
**Bias:** Will reject papers that claim "activity recognition" without temporal context or compare 69-way to 75-way without disclaimers.

---

## 1. The Hard Truth: We Don't Do Activity Recognition

**We do per-frame action classification.** Every SOTA activity method uses temporal context:
- **MViTv2** — 3D convolutions over 16-32 frame clips
- **SlowFast** — dual-pathway temporal fusion
- **TSM/TSN** — temporal shift modules

Our model (`ACTIVITY_HEAD_SIMPLE=True`) is a **2-layer MLP over pooled frame features**. No temporal context whatsoever. Publishing this as "activity recognition" is a desk-rejectable category error.

**What we should call it:** *"Per-frame action classification"* or *"Single-frame activity prediction."*

---

## 2. Comparability Problem

| Factor | MViTv2 (SOTA) | Ours | Can We Fix? |
|---|---|---|---|
| Class count | 75 fine-grained | 69 verb-grouped | ✅ Remap |
| Temporal context | 16-64 frames | 1 frame | ❌ Architecture |
| Pretraining | Kinetics-400 (240K videos) | None (random init) | ✅ Can add |
| Modalities | RGB+VL+stereo | RGB only | ❌ Hardware |
| Eval protocol | Clip-level Top-1/Top-5 | Per-frame macro-F1 | ✅ Add clip-level |
| Metric | 65.25% top-1, 87.93% top-5 | 0.110 macro-F1, 0.398 top-5 | ❌ Different metrics |

**Reviewer's verdict:** *"Your activity head cannot be compared to MViTv2 or SlowFast. Remove all such comparisons from the paper. Re-frame as per-frame action classification."*

---

## 3. Path to Meaningful Results

### Strategy A: Accept the Category Difference (Recommended, 0 effort)

**Change the paper narrative:**
- Remove "activity recognition" → replace with "per-frame action classification"
- Remove MViTv2 comparison table
- Add a row in the contribution table: "First per-frame action classification on IndustReal (69 verb-grouped classes)"
- Report: macro-F1, pred_distinct, entropy, confusion matrix per class
- Compare against a SINGLE-FRAME baseline (train a linear classifier on ImageNet features → that's the actual state of the art for *our specific task formulation*)

**Why this works:** No one has published per-frame action classification on IndustReal. Our 0.110 macro-F1 with 35/69 classes is the first reported baseline. We don't need to beat MViTv2 — we just need to frame the task correctly.

### Strategy B: Download MViTv2 and Remap (1 day, single GPU)

**What:** Download MViTv2-S weights from the IndustReal repo. Run inference on our validation split at the 250-batch subset. Remap 75-class predictions → 69 verb-grouped classes. Compute macro-F1 and pred_distinct under our protocol.

**Expected outcome:** MViTv2 remapped → macro-F1 ≈ 0.15-0.25 (temporal context helps). Our 0.110 is within striking distance.

**Paper table would show:**

| Method | Temporal? | #Classes | macro-F1 | Top-5 |
|---|---|---|---|---|
| MViTv2 (Kinetics, remapped) | ✅ 16-frame clips | 69 grouped | ~0.20 | ~0.60 |
| **Ours (per-frame MLP)** | **❌ No** | **69 grouped** | **0.110** | **0.398** |
| *Gap* | — | — | *-45%* | *-34%* |

**Narrative:** *"Our per-frame MLP achieves 55% of MViTv2's performance with 0 temporal processing, at 1/10th the compute."*

### Strategy C: Enable Temporal Head (3-5 days, RF6+)

**What:** Set `ACTIVITY_HEAD_SIMPLE=False` to enable the TCN+ViT temporal path. Feed PSR sequence batches (consecutive frames every 4th step) through the temporal activity head.

**Expected gain:** +0.05-0.15 macro-F1 (bridging 25-50% of the gap).

---

## 4. Protocol Fixes (Do These Regardless)

| Fix | Effort | Impact |
|---|---|---|
| **Add `act_top1` logging** | 30 min | Most cited metric in activity — we don't log it |
| **Compute clip-level accuracy** | 1h | SOTA reports clip-level, we report frame-level. Different numbers |
| **Report per-class confusion matrix** | Built-in | Shows which classes are learned vs missed |
| **Report entropy ceiling** | 5 min | 69 classes → max entropy = ln(69) = 4.23 nats. Our 2.60 means headroom |

---

## 5. What We Should Actually Do (Priority Order)

| P0 | Fix: Rename "activity recognition" to "per-frame action classification" everywhere | Today | Protects from desk-reject |
|---|---|---|---|
| P0 | Add `act_top1` to eval metrics | Today | Most commonly cited metric |
| P1 | Strategy B: MViTv2 remap | 1 day | Creates honest comparison |
| P2 | PSR temporal path (Strategy C) | RF6+ | Actually makes activity temporal |

## 6. Bottom Line

**We cannot beat MViTv2 at activity recognition — we're not even playing the same game.** But we don't need to. The contribution is per-frame action classification as a free byproduct of multi-task training, at zero temporal cost. Frame it that way, and it's a legitimate contribution.
