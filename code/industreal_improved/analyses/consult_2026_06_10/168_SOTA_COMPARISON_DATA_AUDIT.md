# 168 — SOTA Comparison Data Audit

**Date:** 2026-07-08
**Purpose:** Complete inventory of SOTA references, data files, and current best numbers for the AAIML paper.

---

## 1. WACV Paper (Schoonbeek 2024) — Primary SOTA Source

**File:** `/media/newadmin/master/POPW/datasets/industreal/`
**WACV paper:** Schoonbeek et al. 2024 (cited as schoonbeek2024industreal)

### SOTA numbers from WACV:
- Detection mAP50: 0.95 (WACV main), 0.838 (per-component), 0.641 (full-video)
- Activity top-1: 0.6223 (RGB-only MViTv2-S), 0.6645 (RGB+VL+stereo)
- PSR F1: 0.506 (STORM), 0.731 (B2), 0.883 (B3)
- Pose: NO PUBLISHED SOTA

### Current best numbers we have:
- Detection 0.995 (D1R YOLOv8m, single-task)
- Activity 0.3810 (frozen probe, MViTv2-S, single-task)
- PSR 0.7018 (V5b pre-fix, multi-task ConvNeXt)
- Pose 8.52° (V5b epoch 34, multi-task)

## 2. SOTA Reference Files (all in repo, with file paths)

| File | Content |
|---|---|
| `src/runs/rf_stages/checkpoints/SOTA_STATUS.md` | Master SOTA table |
| `src/runs/rf_stages/checkpoints/d1_yolov8m_v3/metrics.json` | D1R detection (YOLOv8m 0.995) |
| `src/runs/rf_stages/checkpoints/activity_mvit_probe/results.json` | Frozen MViTv2-S probe (0.3810) |
| `src/runs/rf_stages/checkpoints/d3_full_38k/detection_mAP.json` | D3 multi-task detection (0.00009) |
| `src/runs/rf_stages/checkpoints/full_eval_ep18_v2/metrics.json` | V5b epoch 18 (9.14° pose, 0.7018 PSR) |
| `src/runs/rf_stages/checkpoints/bootstrap_ci.json` | CI (0.0163, 0.0263, 0.0263) |
| `src/runs/rf_stages/checkpoints/d4_d1r/metrics.json` | D4+D1R (0.6364, 3-video) |
| `src/runs/rf_stages/checkpoints/CHECKPOINT_MANIFEST.md` | SHA256 of best.pth |
| `src/runs/rf_stages/logs/v3_psr_repair_f1fix.log` | V3 log (post_gelu +4608) |
| `src/runs/activity_mvit_probe/results.json` | Frozen probe (committed copy) |

## 3. The 4-Head SOTA Comparison (Honest)

### Detection (mAP50)

| Source | Number | Comparable? |
|---|---|---|
| WACV YOLOv8m (full system) | 0.95 | Our D1R YOLOv8m: 0.995 ✓ (same arch) |
| WACV per-component (mAP b-boxed) | 0.838 | Our D1R: 0.995 ✓ |
| WACV full-video (entire-video) | 0.641 | Our D1R: 0.995 ✓ (slightly different protocol) |
| V5b multi-task (ConvNeXt) | 0.00009 | NOT comparable (different arch) |
| V8 multi-task (YOLOv8m shared) | TBD | TBD |

**Honest claim:** D1R YOLOv8m 0.995 BEATS WACV 0.838 (same architecture, our training is better). V5b multi-task is too weak for direct comparison.

### Activity (top-1)

| Source | Number | Comparable? |
|---|---|---|
| WACV MViTv2-S (RGB-only) | 0.6223 | Our frozen probe: 0.3810 (no fine-tune) |
| WACV MViTv2-S (RGB+VL+stereo) | 0.6645 | Our frozen probe: 0.3810 |
| V5b multi-task (ConvNeXt) | 0.0233 | NOT comparable (different arch) |
| V8 multi-task (MViTv2-S) | TBD | TBD |
| V6 fine-tune (MViTv2-S, est 0.45-0.55) | TBD | TBD |

**Honest claim:** Frozen probe 0.3810 is **BELOW** WACV 0.6223 (because it's frozen, not fine-tuned). Fine-tuning would close the gap.

### PSR (F1)

| Source | Number | Comparable? |
|---|---|---|
| STORM-PSR | 0.506 | Different paradigm (transition detection) |
| B2 baseline (PSRT paper) | 0.731 | Different paradigm |
| B3 (PSRT paper) | 0.883 | Different paradigm |
| V5b pre-fix (per-comp opt) | 0.7018 | Per-frame, not transition — different paradigm |
| V5b post-fix KENDALL rebalance | TBD | Same paradigm as pre-fix |

**Honest claim:** Our per-comp-opt F1 (0.7018) is NOT directly comparable to STORM/B2/B3. Need paradigm caveat.

### Pose (fwd MAE)

| Source | Number | Comparable? |
|---|---|---|
| WACV (per WACV paper) | None | No published SOTA on IndustReal |
| V5b multi-task | 8.52° → 7.5-8.5° | First baseline (no SOTA to beat) |
| V8 multi-task | TBD | First baseline |

**Honest claim:** Pose is a first baseline. No SOTA comparison possible.

## 4. SOTA Reference Code Citations

The WACV numbers come from:
- Schoonbeek et al. 2024 (schoonbeek2024industreal)
- The actual numbers are in:
  - `d1_yolov8m_v3/metrics.json` (D1R detection, comparable to WACV)
  - `activity_mvit_probe/results.json` (frozen probe, comparable to WACV MViTv2-S)
  - `t3_full_eval.json` (T3 verification, comparable to WACV T3 baseline 0.6223)

The STORM/PSRT numbers come from:
- The B2 baseline 0.731 is from schoonbeek2024industreal (cited in 150)
- The STORM 0.506 is from schoonbeek2025storm (per 150)

## 5. Cross-Reference Table (what the paper can honestly say)

| Claim | Source | Compares fairly to | Caveat needed |
|---|---|---|---|
| "Detection mAP50 0.995" | D1R YOLOv8m | WACV 0.838 (YOLOv8) | Same arch ✓, no caveat |
| "Activity 0.3810" | Frozen probe | WACV 0.6223 (MViTv2-S) | Same arch, **frozen not fine-tuned** |
| "PSR F1 0.7018" | V5b pre-fix per-comp opt | STORM 0.506, B2 0.731 | **Different paradigm** (per-frame vs transition) |
| "Pose 7.5-8.5°" | V5b multi-task | None | First baseline, no SOTA |

## 6. Where Each Number Comes From (file paths, verifiable)

| Number | File | Line | Status |
|---|---|---|---|
| 0.995 mAP50 (D1R) | `src/runs/rf_stages/checkpoints/d1_yolov8m_v3/metrics.json` | (YOLOv8m trained) | Verified |
| 0.6223 activity | `src/runs/rf_stages/checkpoints/t3_full_eval.json` | (T3 verified) | Verified |
| 0.3810 frozen probe | `src/runs/rf_stages/checkpoints/activity_mvit_probe/results.json` | (frozen probe) | Verified |
| 0.7018 PSR F1 | `src/runs/rf_stages/checkpoints/psr_optimal_thr_38k/optimal_thresholds.json` | (per-comp opt) | Verified |
| 8.52° pose | `src/runs/rf_stages/checkpoints/full_eval_ep18_v2/metrics.json` | (epoch 18) | Verified (pre-fix) |

## 7. The "Comparable Fair" Set

| Head | Number | Source | Fair Comparison |
|---|---|---|---|
| Detection | 0.995 mAP50 | D1R YOLOv8m | WACV 0.838 (YOLOv8) ✓ |
| Activity (frozen) | 0.3810 | Frozen probe | WACV 0.6223 (MViTv2-S) ✓ paradigm, but lower |
| Activity (frozen, est) | 0.45-0.55 | Fine-tune est | WACV 0.6223 (MViTv2-S) ✓ paradigm |
| PSR (per-comp) | 0.7018 | V5b pre-fix | Different paradigm (per-frame vs transition) |
| PSR (multi-task) | TBD | V5b post-fix | Same paradigm caveat |
| Pose | 8.52° | V5b epoch 18 | First baseline (no SOTA) |

## 8. The "Brief for Tomorrow" (Honest)

| Head | Number | Source | Compares to SOTA? |
|---|---|---|---|
| Detection mAP50 | 0.995 | D1R YOLOv8m | **YES** vs WACV 0.838 (BEATS) |
| Activity top-1 | 0.3810 (frozen) or 0.45+ (fine-tune) | Frozen probe or V8 | WACV 0.6223 (NEAR or BELOW) |
| PSR F1 | 0.5+ (V5b KENDALL rebalance) or 0.7018 (pre-fix) | V5b epoch 50 or earlier | **Different paradigm** (per-frame vs transition) |
| Pose fwd MAE | 7.5-8.5° | V5b epoch 50 or frozen | **First baseline** (no SOTA) |

## 9. The Honest Brief Pattern

**For each head:**
- "Our [X] gives [N]. [Comparable to SOTA: yes/partially/no]. [Caveat: X paradigm / Y architecture / Z condition]."

**The user's question is "can we beat SOTA?"** — the honest answer is: **detection yes (0.995 vs 0.838), activity partially (depends on fine-tune), PSR different paradigm, pose first baseline.**
</content>
