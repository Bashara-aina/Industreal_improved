# 177 — Paper Tables A/B/C (per 175 §8)

**Date:** 2026-07-08
**Status:** Tables filled with currently-measured numbers; gaps explicitly noted.
**Source:** fvcore `efficiency_measured/metrics.json`, V8 hash-fix test, PSR transition-F1 eval, activity 75-class eval, split config.

---

## Table A — Accuracy vs SOTA (test split, matched protocol)

This is the headline-comparison table. **Every number below traces to a measured artifact.** Where evidence is missing, the entry is `TBD (test-split eval pending)` and the gap is named.

| Head | Metric | Ours (current state) | SOTA anchor | Protocol | Verdict |
|---|---|---|---|---|---|
| **Detection** | mAP@0.5 (annotated frames) | TBD (test-split eval pending) | WACV **0.838** | COCO-eval on GT-frames subset | parity target |
| **Detection** | mAP@0.5 (entire videos) | TBD (test-split eval pending) | WACV **0.641** | COCO-eval on full sequence | parity target |
| **Detection** (multi-task, val) | mAP50_pc | **0.468** (full-38k val, ep62) | — | present-class average | first public MT number |
| **Detection** (single-task D1R, val) | mAP50 native | **0.00043** (native harness) | — | sparse detection, harness-level | **NOT the 0.995 number; that is Ultralytics-native on a different protocol** |
| **Activity** (multi-task, 75-class clip, val) | top-1 / top-5 | **0.384 / 0.709** (frozen probe, 1984 clips) | WACV MViTv2-S **65.25 / 87.93** (75-cls test) | 16-frame clips, 75-class | **NOT comparable** — frozen probe, val split, end-to-end fine-tune + test eval pending |
| **Activity** (multi-task, 69-group per-frame, val) | top-1 | **0.381** (frozen probe prior) | — | per-frame 69-group | taxonomy mismatch with WACV |
| **PSR** | event_f1@±3 + τ | **0.0000** (current checkpoints; τ=NaN) | STORM **0.901 / 15.5 s**; B3 **0.883 / 22.4 s** | transition-event, greedy match, ±3 frames | **needs post-LeakyReLU training** |
| **PSR** (legacy per-frame opt) | psr_macro_f1 | 0.7018 (CI 0.6436–0.7321) | — | per-frame state F1, post-hoc per-comp thr | **different paradigm; appendix only** |
| **Head pose** | fwd MAE (95% CI) | **9.14° (7.74–10.87°)** (full-38k val bootstrap) | none published | degrees(arccos(cos)) | **first public baseline** |
| **Head pose** | up MAE (95% CI) | **7.78° (6.89–8.81°)** (full-38k val bootstrap) | none published | degrees(arccos(cos)) | **first public baseline** |

**Key protocol verifications:**
- Our pose MAE uses `degrees(arccos(cos(pred_unit, gt_unit)))` (continuous 6D, Zhou et al.); see `gt_pose_variance.py:40`.
- Our PSR's `event_f1@±3` is greedy match within ±3 frames per `decoder_oracle_bound.py:252`, default B3/STORM protocol.
- Our activity 75-class top-1 is computed on 1984 valid clips from `recordings/05_*,14_*,20_*,24_*,26_*` (val split), via `scripts/eval_activity_75class.py` feature-probe mode.

**Important caveats:**
- We compare *ours (val)* to *SOTA (test)*. This is unfair to us — the gap may shrink when re-evaluated on test. **Until test-split evals exist for our numbers, every "beats SOTA" sentence needs a brightness caveat.**
- The 0.995 detection figure cited in some documents is **single-task D1R Ultralytics-native**, on a different protocol from WACV's COCO-eval. It is not in the matrix above.
- The 0.00043 D1R number is the harness's native eval of Microsoft-cached weights on the same val split.

---

## Table B — MTL vs Single-Task (the hypothesis matrix)

The honest version of this table is mostly empty until we run the controlled matrix in 175 §6. **We are not yet in a position to claim MTL helps** — but we can list exactly what we need.

**Currently measured:**

| Head | ST baseline | MTL-All | Δ (95% CI) | Transfer? |
|---|---|---|---|---|
| **Detection** | TBD — ST-Det YOLOv8m full-set mAP eval pending | 0.468 mAP50_pc (val, ep62) | TBD | hypothesis: ≈ (pose-aware backbone helps detection) |
| **Activity** | TBD — ST-Act MViTv2-S fine-tune pending | 0.000 (current; activity collapse) | TBD | hypothesis: ↑ if hash bug fixed + backbone adequate |
| **PSR** | TBD — ST-PSR per-component classifier pending | 0.0000 event_f1@±3; 0.7018 per-frame | TBD | hypothesis: ↑ (detection as auxiliary, see 175 §1) |
| **Pose** | TBD — ST-Pose MLP regression pending | 9.14° fwd / 7.78° up (full-38k val) | TBD | hypothesis: ≈ (pose is the easiest head to MTL) |

**Four single-task baselines needed (per 175 §6):**
1. **ST-Det** — ConvNeXt-Tiny + YOLOv8m detection head, full-set mAP, **val → test re-eval required**.
2. **ST-Act** — MViTv2-S fine-tune end-to-end on 75-class clip-level top-1.
3. **ST-PSR** — Per-component classifier on 11 transitions, run on val then re-eval on test.
4. **ST-Pose** — Single-task MLP regression on the 6D pose target, full-38k bootstrap CI.

Plus ablations (175 §6): MTL-All+PCGrad, LOO-noDet, LOO-noPose, MTL-frozenBB, Backbone-swap (ConvNeXt vs Hiera).

**Run count:** 8–10 runs. Per-agent estimate from `agent-7 PCGrad` (5× backward passes per step for 4 tasks) and `agent-2 Hiera-B` (~58M params) suggests each run is feasible on the available hardware (RTX 5060 Ti + RTX 3060).

---

## Table C — Efficiency (measured, not fabricated)

This table replaces the fabricated 4× / 600M / 6.7× numbers in documents 167 and 170. Source: `scripts/measure_efficiency.py` + fvcore on identical hardware (RTX 5060 Ti, batch=1 unless noted).

| Metric | Σ 4× single-task (estimated) | V5 (MTL) | V8 (MTL) | Tier F (MTL, Hiera-B) | Saving |
|---|---|---|---|---|---|
| **Params (M)** | ~100.0 | **46.47** | **53.80** | **58.26** (Agent 2 spec; Hiera pretrained weights not downloaded) | V5: ~2.15×, V8: ~1.86× |
| **FLOPs (G) per forward** | architecture-dependent | **245.73** | **67.11** | TBD (depends on detection-mode resolution) | V8 is ~3.7× lower FLOPs (lower input resolution, not MTL) |
| **FPS (batch=1)** | 4× sequential ≈ 1/4 MTL | 13.5 | **17.7** | TBD | V8 ≈ V5+30% |
| **Latency (ms, batch=1)** | — | 74.3 | **56.6** | TBD | V8 ≈ V5–24% |
| **Peak VRAM (GB, batch=1)** | — | 0.48 | 0.51 | TBD | comparable |
| **Storage (FP32, MB)** | ~400 | **177.3** | **205.2** | ~222 (estimate) | V5 ≈ V8 (~2× over single backbone) |
| **Inference forward-passes** | 4 sequential | 1 | 1 | 1 (temporal mode) / 2 (with detection mode) | **4×** for single-pass vs sequential |

**Honest comparison:**
- **Where the savings are real:** parameter sharing (~2×), single forward pass for all 4 heads (real 4× speedup vs sequential single-task at inference), storage (real ~2×).
- **Where the savings are inflated:** "4× compute" is for storage/passes, NOT for FLOPs or wall-clock training. The 4× in 167/170 conflated these and overstated savings.
- **FLOPs caveat:** the 67 GFLOPs for V8 reflects 224² input; the architectural-design commitment to a single-shared-backbone that runs both temporal and detection modes will give a higher total if both modes are invoked (precisely what 175 §3.1 prescribes for Tier F).
- **Limitation:** all numbers above are computed on architectures without pretrained weights (random init). Pretrained weights do not affect FLOPs or param count; they do affect final accuracy.

**Net honest claim for the paper:**

> "The proposed shared-backbone multi-task model achieves approximately a 2× reduction in parameter count and a 4× reduction in inference forward-passes relative to running four separate single-task models, while operating in a single forward pass at first-frame latency."

This is defensible. The 4× / 600M / 6.7× claims are not.

---

## Provenance (full citation list)

| Number | Source |
|---|---|
| 0.468 mAP50_pc | `src/runs/full_multi_task_tma_tbank_benchmark/logs/metrics.jsonl` ep62 |
| 0.00043 D1R native eval | `src/runs/rf_stages/checkpoints/d1_yolov8m_v3/metrics.json` |
| 0.995 D1R (single-task, NOT MT) | `src/runs/rf_stages/checkpoints/d4_d1r/metrics.json:16` |
| 9.14° / 7.78° pose + CI | `src/runs/rf_stages/checkpoints/bootstrap_ci.json` |
| 0.7018 PSR per-frame | `src/runs/rf_stages/checkpoints/bootstrap_ci.json` (`psr_f1.headline_optimal_macro_f1`) |
| 0.0000 PSR event_f1@±3 (post-LeakyReLU checkpoints TBD) | `scripts/eval_psr_transition_f1.py` output → `psr_event_f1_run/metrics.json` |
| 0.384 / 0.709 activity 75-class | `scripts/eval_activity_75class.py` → `activity_75class_eval/metrics.json` |
| 0.381 activity 69-group frozen | `src/runs/rf_stages/checkpoints/activity_mvit_probe/results.json` |
| 46.47M V5 params / 245.73 GFLOPs / 13.5 FPS | `scripts/measure_efficiency.py` → `efficiency_measured/metrics.json` |
| 53.80M V8 params / 67.11 GFLOPs / 17.7 FPS | same file |
| 58.26M Tier F params (no pretrained) | `src/models/tier_f_model.py` (Hiera-B via timm, fallback random init) |
| Test split 12 / 5 / 10 | `config/splits/industreal_split.json` |
| STORM 0.901 / 15.5 s | arXiv:2510.12385 (CVIU 2025) Table 1 |
| WACV 0.838 / 0.641 / 65.25 | `reviewer-1-detection-path-to-SOTA.md`, `174_SOTA_PROTOCOLS_AND_EVAL_DEFINITIONS.md` §2 |

---

*These tables close §6 of the paper outline (175 §9). Tables A and C are submission-ready with the limitations noted; Table B is intentionally incomplete until the controlled matrix runs.*
