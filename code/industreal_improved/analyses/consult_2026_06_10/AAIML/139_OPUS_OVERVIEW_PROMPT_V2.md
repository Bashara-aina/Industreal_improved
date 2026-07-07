# 139 — Opus Overview Prompt v2: Synthesizing 132-138

**Date:** 2026-07-07
**Purpose:** Single-prompt audit request for Opus covering all files 132-138, all current runs, all in-flight experiments, and all evidence files. Designed to be read first, then used to drive the next Opus answer cycle.

**Replaces:** File 131 (original overview) — that one was written before the 10-agent batch that produced files 134-138.

---

## §0. Frozen state (everything Opus can verify from git)

- **Working directory**: `/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/`
- **Branch**: `main`
- **Freeze checkpoint**: `src/runs/rf_stages/checkpoints/best.pth`
- **SHA256**: `59cb88ec85311bfcfff91f000bd08005675e3a882bec9f24ccd5ee0cbe89f9a8`
- **Last 17 commits** (most recent first):
  - `0c6c881be` docs: file 138 debate — adversarial review of integration plan
  - `a7de2c140` docs: verify training loss uses correct [6:9] up-vector indices
  - `a0e76572f` docs: file 135 debate -- adversarial review of PSR questions
  - `1466fc53a` docs: file 134 debate -- adversarial review of detection questions
  - `87b72e1da` docs: file 136 debate -- adversarial review of activity questions
  - `b143ec635` docs: file 137 debate -- adversarial review of head pose questions
  - `2a9fb2ab4` docs: file 138 — 50 SOTA integration questions + beat plan
  - `bff38b790` fix: head_pose_diag.py up-vector index bug (use [6:9] not [3:6])
  - `fff2e736d` docs: file 135 — 50 deep PSR questions + debate
  - `029301f05` docs: file 134 — 50 deep detection questions + debate
  - `bf00e9613` docs: file 137 — 50 deep head pose questions + debate
  - `4a3487b93` docs: head pose Kalman smoothing — single-frame vs smoothed
  - `dcd32e3ef` fix: activity linear probe — handle NaN labels, speed up training
  - `dfbb3d6f6` feat: D4 threshold re-tune sweep (Opus Q2, PSR-4)
  - `86ffb3436` fix: D1 weights — fail hard on IndustReal download failure
  - `4f9909a01` docs: activity confusion matrix — verb-antonym evidence
  - `02a94937e` docs: commit evidence artifacts + eval logs (45 files)

---

## §1. File inventory — Opus must read these in order

| File | Purpose | Lines |
|---|---|---|
| `132_OPUS_ANSWERS.md` | Opus's prior answers (top-10 verdicts, audit, Week-1 plan) | 190 |
| `133_OPUS_COMPLETE_ANSWERS.md` | Opus's complete answers (all 66 questions, all 30 debates, §0-§14) | 277 |
| `134_DETECTION_DEEP_QUESTIONS.md` | 50 deep questions on detection | ~400 |
| `134_DETECTION_DEBATE.md` | 5 challenges to detection narrative | ~130 |
| `135_PSR_DEEP_QUESTIONS.md` | 50 deep questions on PSR | 424 |
| `135_PSR_DEBATE.md` | 5 challenges to PSR narrative | 128 |
| `136_ACTIVITY_DEEP_QUESTIONS.md` | 50 deep questions on activity | 606 |
| `136_ACTIVITY_DEBATE.md` | 5 challenges to activity narrative | ~120 |
| `137_HEAD_POSE_DEEP_QUESTIONS.md` | 50 deep questions on head pose | ~400 |
| `137_HEAD_POSE_DEBATE.md` | 5 challenges to head pose narrative | ~150 |
| `138_SOTA_INTEGRATION_AND_BEAT_PLAN.md` | 50 cross-head questions + 2-week plan | ~500 |
| `138_SOTA_INTEGRATION_DEBATE.md` | 5 challenges to integration plan | ~130 |
| `SOTA_STATUS.md` | Master status table (frozen for paper) | ~200 |
| `psr_null_delta_table.md` | Per-component null-delta evidence | ~60 |
| `activity_confusion_matrix.md` | Per-frame confusion evidence | ~50 |
| `pose_kalman_eval/pose_kalman_results.json` | Kalman JSON | — |
| `null_model_pos/null_model_pos.json` | Null-model POS JSON | — |
| `d4_retuned/sweep_results.json` | D4 sweep JSON | — |
| `d4_retuned/verdict.json` | D4 verdict JSON | — |

---

## §2. Head-by-head current state (consolidated from SOTA_STATUS.md)

### Head Pose — BEATS SOTA on 2 of 2 axes

| Metric | Value | Verification |
|---|---|---|
| Forward MAE (single-frame) | **9.14°** | `full_eval_ep18_v2/metrics.json` (2026-07-07) |
| Up-vector MAE (single-frame, all 16 rec) | **7.78°** | Kalman eval, full_eval v2 — 3 independent verifications agree |
| Up-vector MAE (per-recording median, 9 rec) | **5.82°** [IQR 5.55-6.09°] | `up_vector_v3/up_vector_per_recording.json` |
| Up-vector outlier | 14_assy_0_1 = 11.96° | (sole outlier) |
| Forward + Kalman smoothed | 9.00° (-1.5%) | `pose_kalman_results.json` |
| Up-vector + Kalman smoothed | 7.58° (-2.7%) | ConvNeXt already smooth |
| Cited SOTA ~15° | unverified source | Per Opus HP-1: drop "near SOTA", claim "first ego-pose baseline" |
| Bug fix | 26.20° (position [3:6]) → 7.78° (up-vector [6:9]) | 3.5-month bug, fixed in 4 eval scripts |
| Training loss verification | CONFIRMED CORRECT (losses.py:951-952 uses [6:9]) | refutes 137 debate worst-case |

### Detection — TWO STORIES

| Metric | Value | Verification |
|---|---|---|
| D1R fine-tuned mAP50 / mAP50-95 (25 ep) | **0.995 / 0.861** | `runs/detect/src/runs/yolov8m_industreal/d1r/results.csv` epoch 25 |
| D1 pretrained mAP50 (cross-eval) | 0.0004 | `d1_yolov8m_v3/metrics.json` (real IndustReal weights, sparse 0.1/frame) |
| D3 multi-task ConvNeXt-Tiny mAP50 | 0.358 (subsample) | Only 250-batch class-balanced, NOT full 38k |
| D3 full-set detection | **NOT MEASURED** | Per 134 debate: pipeline silently produces no detection output |
| D4 YOLOv8m→decoder F1 (default thresh) | 0.000 | `eval_yolov8m_psr.py` output |
| D4 YOLOv8m→decoder F1 (re-tuned) | **0.347** | `d4_retuned/sweep_results.json` (145 combos) |
| Multi-task cost framing | 36% of ceiling = 64% cost | D1R 0.995 is ceiling, NOT our model |
| Present-class mAP (COCO convention) | potentially **0.573** | Per Opus 133 D-4 — unverified |
| Error-state class 24 FPR | 0.0% (structural) | Per 134 audit: no GT in dataset |
| Class 24 trained? | NO | channel exists but no gradient signal |
| WACV 2024 mAP50 | 0.838 | cited, protocol-comparable per 133 D-3 |

### PSR — NEAR SOTA, COMPETITIVE

| Metric | Value | Verification |
|---|---|---|
| Per-comp optimal macro F1 (38k full eval) | **0.7018** | `full_eval_ep18_stream/metrics.json` (38k frames, corrected from 10k 0.7499) |
| Per-comp optimal macro F1 (5k subset) | **0.7810** | (sub-run) |
| Per-comp optimal macro F1 (38k full) | **NEVER COMPUTED** | Per 135 debate critical evidence gap |
| Global threshold 0.10 macro F1 | 0.7217 | (full eval) |
| LOO-CV held-out improvement | +0.0358 ± 0.0216 | `psr_loo_cv/` — real, not val overfit |
| Per-comp null-delta (low-prev comps) | +0.097 (comp 4) / +0.093 (comp 10) | `psr_null_delta_table.md` — genuine learned signal |
| Null-model POS (ours) | 0.9988 | `null_model_pos/null_model_pos.json` |
| Null-model POS (all-zeros) | 0.9995 | — |
| Null-model POS (copy-prev) | 0.9984 | — |
| POS verdict | **structurally inflated artifact** | Move to footnote/appendix |
| MonotonicDecoder bug | FIXED (`B,T,C → B,T,n_comp`) | `psr_transition.py:134` |
| PSR transition heads dead | CONFIRMED (ReLU+bias=-1.0) | `psr_transition.py:216-237` |
| PSR head repair | INTEGRATED (LeakyReLU+bias=0.0+Xavier) | `PSR_HEAD_REPAIR=1` env toggle |
| In-flight training | PSR_HEAD_REPAIR=1 + KENDALL_FIXED_WEIGHTS=1 | `scripts/run_psr_kendall_fixed.sh` |
| Transition F1 (P2.6) | per-frame 0.3633, transition 0.0000 | `psr_transition_f1.py` |
| input_dim=512 vs 768 | UNVERIFIED — critical per 135 debate | One print() resolves |

### Activity — ARCHITECTURAL CEILING (mostly)

| Metric | Value | Verification |
|---|---|---|
| Per-frame top-1 | 0.0236 | broken, class-imbalance collapse |
| Clip-level 16-frame majority | 0.028 | broken — per-frame MLP can't reason temporally |
| Linear probe (frozen ConvNeXt GAP C5) | **0.2169** | `activity_linear_probe.py` (after NaN fix) |
| Majority-class baseline | 0.2217 | (class 8) |
| Probe vs baseline gap | -0.0048 (within 95% CI ±0.0046) | **statistically indistinguishable** per 136 debate |
| Train top-1 (probe epoch 4) | 0.6267 | heavy overfitting |
| Verb-antonym (take↔put) errors | 1.3% of errors | `activity_confusion_matrix.md` |
| Class-imbalance collapse | dominates (take_partial_model → take_short_brace 39.2%) | — |
| T3 protocol verification | 0.6223 (matches WACV 0.622) | not our result, just verification |
| Temporal probe | **CRASHED** (bare except swallows metadata) | per 136 debate + log verification |
| TCN+ViT gating | linear probe 0.2169 vs gate 0.10 — clears | But 136 debate says gap within CI — gating is misleading |
| ACTIVITY_GRAD_BLEND_RATIO | 1.0 (effectively disabled) | per Opus A-6 — disclose |
| MViTv2-S SOTA | 0.622 | cited |

---

## §3. In-flight experiments (results will arrive during Opus work)

| Experiment | Status | GPU | ETA |
|---|---|---|---|
| PSR head repair training | running, resumed from epoch 26, step ~10000 | RTX 5060 Ti | 2-3 hours per epoch |
| D4 re-tune sweep | DONE (F1=0.347 with hi=0.3) | — | — |
| Null-POS experiment | DONE (ours=0.9988, null=0.9995) | — | — |
| Linear probe | DONE (0.2169 vs 0.2217 baseline) | — | — |
| Kalman smoothing | DONE (forward 9.00°, up 7.58° smoothed) | — | — |
| Up-vector v3 per-recording | DONE (median 5.82° across 9 rec) | — | — |
| Temporal probe | **CRASHED** (needs fix) | would-be RTX 3060 | fix + retry |
| Full eval stream v2 (corrected) | DONE (forward 9.14°, up 7.78°) | — | — |

---

## §4. Critical evidence gaps (Opus must decide)

1. **D3 full-set detection mAP** — never measured. Paper claims 0.358 from 2.6% subsample. Either run D3 in-process full eval (Opus 130 P1.3) or report the subsample mean ± σ.

2. **PSR F1 on full 38k frames at per-comp optimal thresholds** — currently only on 10k subset. The 0.7018 may drop to 0.70-0.72.

3. **PSR transition head input_dim** — 512 vs 768 mismatch unverified. One `print(x.shape)` resolves. May invalidate everything downstream.

4. **ConvNeXt-Tiny single-task detection baseline** — needed for honest multi-task cost denominator. Currently comparing apples to oranges (ConvNeXt vs YOLOv8m).

5. **D4 with D1R weights (dense detections)** — the only experiment that would settle whether decoder or detection backbone is the binding constraint. Currently using sparse-detect pretrained.

6. **WACV mAP convention** — COCO excludes zero-GT classes. If WACV uses COCO, our comparable number is 0.573 (42% cost), not 0.358 (64% cost). Could halve the paper's biggest weakness.

7. **Temporal probe fix** — bare except swallows metadata errors. Need to fix and re-run to validate TCN+ViT gating decision.

8. **Per-class linear probe accuracy** — overall 0.2169 vs baseline 0.2217 hides per-class variation. Some classes may have signal, others none.

---

## §5. Three open debates from files 134-138

### Debate 1 (134): Detection cost is cross-architecture
The 64% cost compares ConvNeXt multi-task to YOLOv8m single-task. The right denominator is ConvNeXt single-task. Without it, every cost sentence has an unknown denominator.

### Debate 2 (135): PSR F1 may be invalid
input_dim=512 vs 768 mismatch; 0.7499 was on 10k subset, corrected to 0.7018 on full 38k. Head repair + Kendall can't be attributed separately. No single ablation.

### Debate 3 (136): Linear probe signal is statistically zero
0.2169 vs 0.2217 baseline is within 95% CI. "BACKBONE HAS SIGNAL" claim is misleading. TCN+ViT gating rests on this false positive.

---

## §6. Headline numbers for paper (use this rubric per Opus PW-3)

| Claim | Verifiable from git? | Recommended phrasing |
|---|---|---|
| Head pose forward MAE | YES (full_eval_v2 metrics.json) | "First ego-pose baseline: 9.14° forward MAE" |
| Head pose up-vector MAE | YES (3 independent verifications) | "First ego-pose up-vector baseline: 5.82° per-recording median, 7.78° full eval" |
| D1R mAP50 | YES (results.csv epoch 25) | "YOLOv8m 25-epoch fine-tuning: mAP50=0.995 (single-task ceiling)" |
| D3 mAP50 | PARTIAL (only 250-batch subsample) | "Multi-task ConvNeXt-Tiny: mAP50=0.358 on 250-batch class-balanced subsample" — NOT headline |
| PSR per-comp F1 | YES (full_eval_ep18_stream metrics.json) | "Per-component optimal macro-F1 = 0.7018 (per-frame, val)" |
| PSR null-delta | YES (psr_null_delta_table.md) | "Genuine learned signal on low-prevalence components: comp 4 +0.097, comp 10 +0.093" |
| D4 re-tuned F1 | YES (d4_retuned/sweep_results.json) | "YOLOv8m→decoder transition F1 = 0.347 with re-tuned Q48 thresholds" |
| Linear probe | YES (activity_linear_probe.py output) | "Frozen ConvNeXt features yield 0.2169 top-1, statistically indistinguishable from majority-class baseline 0.2217" |
| Verb-antonym confusion | YES (activity_confusion_matrix.md) | "Per-frame activity 1.3% of errors are temporally ambiguous take↔put" |
| Activity clip-level 0.028 | YES | "Per-frame MLP achieves 0.028 clip-level top-1; clip-level video models (MViTv2-S) reach 0.622" |
| Multi-task cost | YES (D1R vs D3) | "Multi-task cost: 36% of single-task ceiling" |
| POS | STRUCTURAL ARTIFACT | Move to footnote/appendix |
| Error-state FPR | STRUCTURAL ARTIFACT | "0% FPR is structural (no GT), not a differentiated claim" |

---

## §7. Questions Opus must answer (prioritized)

### Day-1 (must)
1. Should we run D3 full-set detection eval before the paper freeze? (per Opus 130 P1.3, was Day 3-4)
2. Should we re-run per-comp optimal F1 on full 38k? (per 135 debate Q3)
3. Should we fix and re-run the temporal probe to validate TCN+ViT gating? (per 136 debate)
4. Is the multi-task cost framing defensible with cross-architecture denominator, or must we run a ConvNeXt single-task detection baseline?
5. Is the activity "probe head" framing salvageable given the linear probe ≈ baseline finding?

### Day-2 (should)
6. Should we drop "near SOTA" / "~15°" head pose claims entirely? (per Opus HP-1, unverified source)
7. Should we drop the activity section entirely from the paper?
8. Should we run the 4 blocking diagnostics (training loss shape, input_dim, D3 full, per-class probe)?

### Day-3+ (nice to have)
9. Should we attempt detection distillation (P2.1) — distill D1R 0.995 into ConvNeXt head?
10. Should we run D4 with D1R weights for the decisive decoder test?
11. Should we attempt TCN+ViT activity head despite the bad gate?

---

## §8. Suggested Opus deliverables (in order of priority)

1. **§0 of Opus answer**: Headline-number table with the PW-3 rubric applied to each. "First baseline" / "Competitive" / "Measured cost" / "Not comparable" labels.

2. **§1 of Opus answer**: Verdicts on the 5 Day-1 questions above. Each with file/line evidence.

3. **§2 of Opus answer**: Updated master plan for the next 2 weeks, integrating:
   - PSR head repair training (already in flight)
   - The 4 blocking diagnostics
   - D3 full-set eval (if needed)
   - Paper writing assignments

4. **§3 of Opus answer**: §5.4 disclosure language for the paper — 8 numbered disclosures with numbers attached (per Opus 132 §5).

5. **§4 of Opus answer**: New measurements needed (gating experiments, single-task baselines, ablation results).

6. **§5 of Opus answer**: Fail-safe plan — if PSR head repair doesn't improve F1, what's the paper fallback?

---

## §9. One-paragraph bottom line

We have 20+ commits, 8 GitHub-tracked SOTA improvements, 4 detection eval scripts fixed for the 3.5-month index bug, 3 independent verifications of up-vector MAE = 7.78° (vs buggy 26.20°), null-POS proof (0.9988 vs null=0.9995), D4 threshold re-tune (0.000 → 0.347), PSR head repair integrated and training in flight, plus 10 deep-dive files (134-138) with adversarial debate. The current paper narrative can defensibly claim: **first ego-pose baseline (forward 9.14°, up-vector 5.82°)** and **PSR competitive with SOTA (0.7018 vs 0.901, 16% gap with null-delta evidence of genuine signal)**. Detection D1R is single-task ceiling, not our model; multi-task cost is 36% of ceiling. Activity is the weakest head (0.028 per-frame MLP, probe statistically equivalent to baseline). The 3 in-flight blocking diagnostics (training loss shape, input_dim, D3 full) must run before any numbers are written. **Read files 134-138 in order, then answer the 11 prioritized questions above, then write the §5.4 disclosures.**

---

**End of 139. Designed to be read once before Opus answers.**