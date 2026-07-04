# DAY 1 CHECKPOINT: Done & Next Steps

**Date:** 2026-07-04
**Time:** 6 files committed at `9a01920`

---

## ✅ COMPLETED: Day 1 Fixes

| # | Task | Status | Evidence |
|---|---|---|---|
| 1 | Rename "activity recognition" → "per-frame action classification" in all AAIML docs | ✅ Done | sed across all AAIML/*.md |
| 2 | Remove OpenFace/6DRepNet comparisons from all docs | ✅ Done | Removed from 99-aaiml-viability-benchmarking.md (already gone from 98/103/104/106) |
| 3 | Fix parameter arithmetic: "31% fewer params" → "67% fewer params" | ✅ Done | Fixed in AAIML/industreal-all-papers-benchmarks.md |
| 4 | Enable efficiency metrics: SKIP_EFFICIENCY_METRICS=False | ✅ Done | src/config.py line 1208 |
| 5 | Position units: confirmed unreliable | ✅ Documented | evaluate.py:1918-1926 — "DO NOT USE FOR REPORTING" |
| 6 | act_top1 (act_clip) already in eval output | ✅ Confirmed | Already logged as act_clip= in Val: lines |

---

## 🟡 IN PROGRESS: Training Runs

| Run | GPU | PID | Epoch | Duration | Status |
|---|---|---|---|---|---|
| **Main (4-head)** | 5060 Ti | 79699 | 12 (71%) | 20h | ✅ 0 errors |
| **Ablation det-only** | RTX 3060 | 80287 | 16 (~est) | 20h | ✅ 0 errors |

---

## 📋 REMAINING: Experiments to Run

### Priority 0 — On idle 3060 after ablation finishes (~4h remaining)

| Experiment | Est. Time | File Reference | Why |
|---|---|---|---|
| **D1: YOLOv8m eval on our split** | 2h | `reviewer-1-detection-path-to-SOTA.md` | Honest detection benchmark — is YOLOv8m 0.838 on our split too? |
| **D3: Full eval (EVAL_MAX_BATCHES=0)** | 1h | `reviewer-1-detection-path-to-SOTA.md` | Remove subsampling — get paper-quality numbers |
| **D4: YOLOv8m → our PSR decoder** | 2-3h | `reviewer-3-psr-paradigm-reconciliation.md` + `todo-psr-backbone-swap.md` | Isolate PSR head quality from detection quality |

### Priority 1 — On 5060 Ti after main training finishes

| Experiment | Est. Time | File Reference |
|---|---|---|
| **Ablation A: Activity-only** | ~2 days | `run_ablation_suite.sh act` |
| **Ablation A: Pose-only** | ~1.5 days | `run_ablation_suite.sh pose` |
| **Ablation A: PSR-only** | ~1.5 days | `run_ablation_suite.sh psr` |
| **Ablation B: Kendall vs fixed** | ~2 days | `run_ablation_suite.sh kendall-fixed` |
| **Ablation C: Verb-grouping vs raw** | ~2 days | `run_ablation_suite.sh grouping-none` |

### Priority 2 — Before Paper Submission

| Task | Est. Time | Why |
|---|---|---|
| **Measure FPS on 3060 and 5060 Ti** | 1h | Need real numbers for efficiency claim |
| **Verify position source units** | 1h | Contact authors or check HoloLens SDK — or drop position claims entirely |
| **Pull IndustReal ASD weights** | 30min | From https://github.com/TimSchoonbeek/IndustReal — needed for D4 |

---

## ⚠️ Non-Negotiable Before Any Submission

1. **Drop position (mm/cm) claims** — code explicitly says "DO NOT USE FOR REPORTING"
2. **Forward/up MAE (8.14°, 7.06°) is publishable alone** — no position needed
3. **Ablation A must be complete** — paper is a demo without it
4. **Never compare to OpenFace/6DRepNet** — category error
5. **Never claim "activity recognition"** — always "per-frame action classification"
