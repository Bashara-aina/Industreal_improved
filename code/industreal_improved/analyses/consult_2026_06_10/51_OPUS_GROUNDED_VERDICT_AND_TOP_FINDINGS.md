# 51 — OPUS GROUNDED VERDICT: What the Code & Evidence Actually Say

> **Generated:** 2026-06-22 (Opus, claude-opus-4-8) — branch `claude/zen-mccarthy-oahons`
> **Method:** Direct verification against running source + evidence files. Every claim below carries a `file:line` or evidence-file citation. This document **supersedes** the framing in 48/50 where they conflict — and they conflict on the single most important point.
> **Companion files:** `52` (detection diagnosis), `53` (paper strategy), `54` (execution plan), `55` (ablations + code verification + risks).

---

## TL;DR — The one thing that changes everything

**You are not stuck below the gate. You already passed it.**

- The **rf2 → rf3 gate** is `det_mAP50_pc ≥ 0.28` **AND** `forward_angular_MAE_deg ≤ 60` (`src/training/stage_manager.py:140-142`).
- Your **best checkpoint** is `det_mAP50_pc = 0.3036`, `MAE = 9.13°` (`evidence/rf_stage_state.json → best_metrics`).
- **0.3036 > 0.28 ✓ and 9.13 < 60 ✓ → both gate conditions are met.**

The `gate_passed: false` in the state file is **not** because 0.30 < 0.28. The recorded reason is verbatim:

```json
"det_mAP50_pc": { "status": "UNKNOWN", "reason": "metric not found in validation" }
```

The gate evaluator reads the **latest validation line in the recent log window**, and **Run 3 restarted from the epoch-17 checkpoint and has not finished a validation epoch yet**, so the parser sees no fresh `Val:` line. The moment epoch-17 validation completes under Run 3, the gate reads 0.30 / 9.13 and **auto-advances to rf3**.

The "0.40 gate / stuck at 0.207" story in docs 48 and 50 is reading the **obsolete diluted threshold** (old `det_mAP50 ≥ 0.40`) against the **diluted metric** (`det_mAP50 = 0.207`). The honest gate (GUIDE 1) was already lowered to 0.28 in code, and the honest metric already clears it. **Detection is not the blocker. It never was, since the honest gate landed.**

**Immediate consequence:** stop waiting on detection. Advance to rf3 and start training activity — the next real milestone is `act_top1 ≥ 0.22` (the rf3 → rf4 gate), not any detection number. See `54` for the exact command.

---

## The 9 grounded findings

| # | Finding | Evidence | Why it matters |
|---|---------|----------|----------------|
| **F1** | **rf2 gate already passed** (0.3036 ≥ 0.28, 9.13° ≤ 60°). The `false` is a stale-log artifact of the Run-3 restart. | `stage_manager.py:140-142`; `rf_stage_state.json` gate reason = "metric not found in validation" | Detection is *not* gating activity. Advance now. |
| **F2** | **"Detection" = fine-grained assembly-STATE recognition**, not object detection. 24 classes = `background` + 22 **11-bit binary state strings** + `error_state`. Adjacent classes differ by **one bit**. | `config.py:180-205` (`DET_CLASS_NAMES`) | Explains the ceiling, the perfect localization + poor classification, and the binary per-class AP. Reframes the paper. |
| **F3** | **The "Focal suppresses positives" hypothesis is already mitigated in config.** `DET_GAMMA_POS = 0.0` → positives get **zero** focal down-weighting. | `config.py:536-538` (`DET_ASYMMETRIC_GAMMA=True`, `DET_GAMMA_POS=0.0`, `DET_GAMMA_NEG=1.5`) | The doc-50 "OHEM+Focal positive suppression" primary hypothesis is half-wrong. An OHEM ablation tests *negative* mining only — lower expected value than assumed. |
| **F4** | **PSR has never produced a real signal.** The honest metric `psr_f1_at_t = 0.0` in **both** eval snapshots. `psr_overall_f1` (0.09–0.53) is inflated by trivially-constant components (`comp0 = 1.0`). | `evidence/eval_metrics.json`; `runs/full_multi_task_.../metrics.jsonl` epoch-0 | Confirms the `1.546e-08` floor. PSR needs a go/no-go overfit (`54`) or gets dropped from scope. |
| **F5** | **The evaluation harness is complete for all 6 task families**, with named protocols. Numbers appear the instant a head trains. | `evaluate.py` keys: `act_*` (clip majority vote), `psr_*` (symmetric bidirectional greedy + edit + POS), 9-DoF head pose, det (COCO + present-class + all-frames), `as_*`, `ev_*` | De-risks the paper massively. The bottleneck is *training* heads, never *measuring* them. |
| **F6** | **Efficiency is measurable right now, no training.** `thop.profile` for GFLOPs + batched FPS + **streaming FPS** are already coded; all currently logged as `0.0` (never run). | `evaluate.py:2881, 2951-2953, 3877-3879` | The entire efficiency table (paper §5, Tables 5/8) can be filled today. |
| **F7** | **Activity never trained**; at init it collapses to one majority class (`take_short_brace`/`put_tooth_washer`), Top-5 ≈ chance (0.024–0.06 on 75 classes). | `metrics.jsonl` epoch-0 (`act_confusion_matrix` all → one column; `act_top5_accuracy=0.0247`) | rf3 is the *first* real activity training. Expect 10–30% Top-1; even 22% clears the rf3 gate. |
| **F8** | **Phase B (embedding cache) has never run end-to-end.** No cache dir, no smoke test — only an `__main__`. Ablation-A-via-cache is **unvalidated**. | `embedding_cache.py:518` (`__main__` only); no `src/runs/cache*` exists | Ablation A's *cheap* path is a risk. The robust path is the `recovery_det_only` single-task baseline (already a working preset). |
| **F9** | **Anchors are absolute-pixel (96–512) tuned for large assembly-board boxes**, consistent with state-detection. GT p10 height ≈156px; smallest anchor 96px. `TOP_K=9` + `IoU_FLOOR=0.2` force-matching is a band-aid. | `config.py:306-326` | The anchor story is *not* "small parts vs big anchors." It's single-bit class confusion (F2). Anchor recalibration is low-ROI. |

---

## What this reorders (the strategic delta vs docs 48/50)

Docs 48/50 spend ~70% of their energy on the detection ceiling and treat activity/PSR as gated behind it. The grounded reality inverts the priority:

1. **Detection is done "enough."** It clears the honest gate. Squeezing 0.30 → 0.40 changes the paper's acceptance odds by ~nothing (still well below YOLOv8m's 0.838 either way). **Freeze detection effort.**
2. **The thesis lives in the ablations, and they have never run.** "No catastrophic interference" (Ablation A) and "FiLM helps" (Ablation B) are the paper's entire scientific contribution, and you currently have **zero** evidence for either. This is the real risk, not detection.
3. **Activity is unblocked today.** Advance to rf3 (gate passed) and train it. It's the next number the paper needs and the next gate (`act_top1 ≥ 0.22`).
4. **PSR is probably dead.** Run the 1-hour go/no-go. If it can't overfit 50 sequences, drop it — a clean 4-head paper beats a 5-head paper with a fabricated head.
5. **Efficiency is free.** Fill the entire compute table today with the existing `thop` path.

The honest paper is a **multi-task efficiency + interference study on commodity hardware**, where weak detection becomes a *data point* ("the hardest, most interference-prone task is single-bit assembly-state discrimination") instead of a headline failure. See `53`.

---

## Corrections to the record (where 48/49/50 are inaccurate)

| Claim in 48/49/50 | Grounded reality | Source |
|-------------------|------------------|--------|
| "rf3 requires mAP50 ≥ 0.40 gate. We're at 0.207." (48 Q8) | rf3-entry gate is `det_mAP50_pc ≥ 0.28`; current 0.3036 **passes**. 0.207 is the *diluted* metric, not the gate metric. | `stage_manager.py:141`; `rf_stage_state.json` |
| Doc 49 §1.2 says rf2 gate = "det_mAP50_pc ≥ 0.40"; §2.1 row 1.08 says "≥ 0.28" | **Internally contradictory.** Code is authoritative: **0.28**. | `stage_manager.py:141` |
| "OHEM + FocalLoss gradient suppression [suppresses] easy positives" (primary hypothesis, 48/50) | Positives already have `γ_pos = 0.0` (no focal suppression). Only negatives are mined. | `config.py:536-538` |
| "Detection" framed as object detection of assembly parts | It is 11-bit **assembly-state** classification on a localized board. | `config.py:180-205` |
| PSR "supposed to be training… still showed 1.546e-08" — cause uncertain | The honest `psr_f1_at_t = 0.0`; `overall_f1` is a constant-component artifact. Behaviour is fully explained, not mysterious. | `eval_metrics.json`, `metrics.jsonl` |
| Params "53M" stated, GFLOPs/FPS "UNKNOWN, needs scripting" (50 §8) | GFLOPs/FPS/streaming-FPS scripting **already exists** (`thop`); just never executed. | `evaluate.py:2881` |

---

*Read `52` next for the detection diagnosis, `54` for the day-by-day plan you can start executing immediately.*
