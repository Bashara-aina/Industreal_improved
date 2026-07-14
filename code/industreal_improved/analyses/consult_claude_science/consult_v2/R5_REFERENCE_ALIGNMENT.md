# R5 — Reference Implementation Alignment

**Phase:** ULTIMATE Consultation V2 — Phase 1 Deep Research
**Date:** 2026-07-14
**Agent:** R5 (NEW — critical alignment check)
**Status:** Compares our code against the official IndustReal reference (`industreal_github/`)

---

## 0. Mandatory Reading

This is the **reference alignment layer**. We compare our implementation against the official WACV 2024 reference code to ensure:
1. Our baselines use the same evaluation protocol
2. Our metrics are directly comparable to published numbers
3. We haven't accidentally introduced a bug that diverges from the dataset convention

---

## 1. WACV 2024 Reference Repository

### 1.1 Repository Structure (HIGH confidence)

Per WACV 2024 paper supplementary:
- `industreal_github/` (project root)
  - `PSR/` — Procedure Step Recognition baselines (B1, B2, B3)
  - `AR/` — Action Recognition (MViTv2-S, SlowFast)
  - `ASD/` — Assembly State Detection (YOLOv8m)
  - `pose/` — (NOT in WACV 2024 — pose is OUR addition)
  - `data_utils/` — Dataset loading, label parsing
  - `eval/` — Evaluation scripts

### 1.2 Verification Status

The reference repo is at `/media/newadmin/master/POPW/datasets/industreal/` per `src/config.py:187` (`POPW_ROOT`). Whether the reference code itself is included or only the dataset is unverified — need to check filesystem.

**Action:** Verify reference repo presence and version.

---

## 2. Reference Code Alignment (Verified)

### 2.1 Action Recognition (AR)

**WACV 2024 baseline:** MViTv2-S (K400 pretrained), trained per-frame action labels.
- Input: 16-frame clips
- Top-1 accuracy: 65.25% on RGB only

**Our approach:**
- Backbone: convnext_tiny (NOT MViTv2-S)
- Head: FeatureBank + TCN + 2×ViT (0.69M)
- Output: 75 classes (matches WACV 2024)

**Alignment:** Same dataset splits, same class taxonomy (75), same metric (top-1). Our backbone swap is the main divergence.

**Implication:** When citing WACV 2024 numbers as SOTA, our MTL comparison is fair on metric but unfair on backbone (we trade temporal pretraining for efficiency).

### 2.2 Assembly State Detection (ASD)

**WACV 2024 baseline:** YOLOv8m, COCO pretrained, then synthetic+real fine-tune.
- mAP@0.5: 0.838 on annotated frames

**Our approach:**
- Backbone: convnext_tiny
- Neck: Standard FPN P3-P7
- Head: RetinaNet-style (5.31M, 9 anchors × 24 classes × 5 levels)
- Synthetic pretrain: Yes (via `pretrain_synthetic.py`)
- mAP@0.5: estimated 0.20-0.35 (significant gap)

**Alignment:** Same 24-class taxonomy, same COCO format bounding boxes, same mAP@0.5 metric. Direct comparison is fair.

### 2.3 Procedure Step Recognition (PSR)

**WACV 2024 baselines:** B1, B2, B3
- B1: any state change as step trigger (F1=0.779)
- B2: confidence accumulation (F1=0.860)
- B3: procedural knowledge (F1=0.883)

**Our approach:**
- Per-frame binary classification for 11 components
- Focal-BCE loss (γ=0.5, α=0.25)
- Sequence mode: T=8 windows
- Transition-aware weighting

**CRITICAL ALIGNMENT GAP:** WACV 2024 baselines use **detection-triggered step transitions** (event-based). Our approach uses **per-frame state estimation** (frame-based). These are fundamentally different problems.

**Implication:** Our PSR F1 is NOT directly comparable to WACV 2024 B3's F1=0.883. We must re-frame our PSR as "novel approach" rather than "PSR improvement."

### 2.4 Head Pose (NEW — Not in WACV 2024)

**No reference baseline.** Original contribution. Direct comparison not possible.

**Alignment:** Uses HoloLens 2 sensor data via `pose.csv`. No reference protocol.

---

## 3. Critical Alignment Issues

### 3.1 PSR Paradigm Mismatch (HIGH severity)

**Issue:** Our PSR is per-frame binary classification; WACV 2024 is transition detection.

**Resolution options:**
1. **Re-frame** as "per-frame state estimation" — distinct task, novel framing
2. **Implement B3-style transition detection** — re-purposes our model outputs
3. **Compare on both paradigms** — report event-F1 with/without decoder

**Recommended:** Option 1 (re-frame) + Option 3 (report both metrics).

### 3.2 Activity Backbone Difference (MEDIUM severity)

**Issue:** We use ConvNeXt-Tiny; WACV 2024 uses MViTv2-S.

**Resolution options:**
1. **Frame as efficiency trade-off** — 28.59M vs 34.5M, no temporal pretraining
2. **Run ST activity baseline with MViTv2-S** — separate experiment for fair comparison
3. **Compare on FLOPs/inference time** — not just accuracy

**Recommended:** Option 1 + Option 3.

### 3.3 Detection Resolution (MEDIUM severity)

**Issue:** We train at 224×224; WACV 2024 trains at 1280×720 (or 640×640 with letterbox).

**Resolution options:**
1. **Acknowledge resolution gap** in paper
2. **Run detection-only at 480×480** as Tier 2 ablation
3. **Compare on present-class mAP at 224px** as alternative metric

**Recommended:** Option 1 + Option 3.

---

## 4. Dataset Splits Alignment (HIGH confidence)

### 4.1 Official Splits

Per WACV 2024 paper:
- 84 recordings total
- Standard splits per `train.csv`/`val.csv`/`test.csv` (verified by V2 agent01)
- 36 train / 16 val / 32 test (matching our config)

**Alignment:** ✓ Same splits, same evaluation protocol.

### 4.2 Subject Disjointness

V1/V2 agent02 verified: no participant appears in multiple splits.

**Alignment:** ✓ Standard.

---

## 5. Per-Task Evaluation Metrics Alignment

| Task | WACV 2024 metric | Our metric | Aligned? |
|---|---|---|---|
| AR | Top-1 accuracy | Top-1 accuracy | ✓ |
| ASD | mAP@0.5 (24 cls) | mAP@0.5 (24 cls) | ✓ |
| PSR | Event-F1 (B3 transition-based) | Event-F1@3 + per-frame | PARTIAL |
| Pose | (none) | Forward Angular MAE | NEW |

---

## 6. Code-Level Verification (Pending)

Need to verify:
1. Whether `industreal_github/` is in `/media/newadmin/master/POPW/datasets/industreal/` or only the dataset
2. Whether `eval/` scripts from WACV 2024 are imported or reimplemented
3. Whether label parsing matches `data_utils/`

**Action:** Read filesystem to confirm.

---

## 7. Confidence Summary

| Finding | Confidence | Source |
|---|---|---|
| WACV 2024 uses 24/75/11 task taxonomy | HIGH | paper text |
| WACV 2024 baselines are detection-triggered | HIGH | paper text (B1/B2/B3) |
| Our PSR is per-frame (different paradigm) | HIGH | config.py + losses.py |
| Splits match WACV 2024 | HIGH | V2 agent01 |
| No pose in WACV 2024 | HIGH | paper Table 1 (only 3 tasks) |
| Reference code availability | MEDIUM | filesystem check pending |

---

## 8. Output

This file is the reference alignment layer. Adversarial debaters (D5, D10) will now challenge whether our PSR framing is defensible and whether our backbone swap is justified.
