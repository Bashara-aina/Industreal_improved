# 108 — Master Overview: Complete Consultation Package (Files 89-107)

**Date:** 2026-07-03
**Purpose:** Navigation guide for all 19 files in the RF4 consultation package.

---

## How the Files are Organized

The 19 files form four layers plus supporting material:

```
Layer 1: ORIENTATION (89, 108)
  → Start here. Read 89 first (2 min), then dive into what you need.
  
Layer 2: PRE-CONSULTATION STATE (90-95)
  → The situation BEFORE Opus answered. Read these for context on what was wrong.
  - 90: Training history & loss tables
  - 91: Architecture docs (with known errors corrected)
  - 92: Loss analysis & critical gaps
  - 93: Validation metrics & gate criteria
  - 94: Fix history & AAIML strategy
  - 95: 50 original questions for Opus

Layer 3: OPUS ANSWERS (96, 102)
  → THE CORE DOCUMENTS. These contain all fixes and answers.
  - 96: Rounds 1-2 answer: F1-F16, critical seq-batch wipe fix, eval audit, AAIML strategy
  - 102: Round 5 answer: F17-F21, GPU crisis playbook, lv_pose fossil, combined unit bug

Layer 4: POST-FIX STATUS (97-101, 103-107)
  → Current state after implementing all fixes.
  - 97: Status after F1-F16 (epochs 2-4, 8h44m run)
  - 98: Per-head analysis (post-F1-F16)
  - 99: AAIML viability (pre-epoch 5 val)
  - 100: 20 questions (post-F1-F16, pre-epoch 5 answer)
  - 101: Overview of files 89-101
  - 103: Status AFTER epoch 5 val (combined=0.241, RF4 gate passed)
  - 104: Per-head with epoch 5 metrics
  - 105: PSR deep dive (binary acc 0.554, MonotonicDecoder eval bug)
  - 106: AAIML viability UPDATE (65-80%, revised from 40-60%)
  - 107: 20 new questions after epoch 5 val
```

---

## Reading Order

### If you're NEW to the project (start here):
```
89 (2 min) → 96 (15 min) → 108 (this file, 2 min) → 103 (5 min) → 104 (5 min)
→ 107 (10 min) → 102 (10 min, optional)
```

### If you're making a decision about RF gates:
```
103 (metrics + probability) → 104 (per-head) → 106 (AAIML viability) → 107 (questions)
```

### If you're debugging PSR:
```
105 (PSR deep dive) → 107 Q1-Q4 (PSR questions)
```

### If you're writing the paper:
```
106 (AAIML strategy) → 94 (fix history + ablation plan) → 96 (strategy section)
```

---

## Key Current Facts (epoch 5 val, July 3)

| Metric | Value | Status |
|---|---|---|
| RF4 Gate | PASSED | ✅ combined=0.241 |
| det_mAP50_pc | 0.339 | ✅ Near RF10 floor |
| act_macro_f1 | 0.097 | ✅ Recovered from collapse |
| pred_distinct | 48/69 | ✅ Was 5 at epoch 2 |
| pose fwd MAE | 8.92° | ✅ SOTA-competitive |
| pose position | 16.6mm | ✅ Excellent |
| psr comp acc | 0.554 | ⚠️ Learning (above chance) |
| psr transition F1 | 0.0 (eval bug) | ❌ MonotonicDecoder crash |
| Combined | 0.241 | ✅ Improved 32% from epoch 2 |

## All 26 Fixes

**F1-F16** (from doc 96): F1 (seq-batch grad wipe), F2 (KENDALL logging), F3/F3b (PSR loss fixes), F4/F4b (LR peak), F5 (grad centralization off), F6 (BF16 support), F7 (PSR seq 4), F8 (FOCAL_ALPHA 0.50), F9 (ACT_RAMP 3), F10 (clip 5.0), F11 (GATE 250), F12 (cosine probe), F13 (probe parity), F14/F14b (Kendall wd, pose reset), F15 (env override), F16 (ablation presets)

**F17-F21** (from doc 102): F17 (data __init__), F18 (double-ramp fix), F19 (eff pose logging), F20 (combined_v2), F21 (auto LR)

**Stability:** Heartbeat race fix, VAL_EVERY_N_STEPS=0, EVAL_MAX_BATCHES=250, VAL_BATCH_SIZE=8→4, CUDNN_BENCHMARK=False

## Current Training

| PID | Epoch | Uptime | Errors | GPU |
|---|---|---|---|---|
| 2988577 | 6 | 3h12m | 0 | RTX 5060 Ti |
