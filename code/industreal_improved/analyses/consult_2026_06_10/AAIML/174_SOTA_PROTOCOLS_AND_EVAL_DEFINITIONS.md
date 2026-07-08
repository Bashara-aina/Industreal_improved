# 174 — Pinned SOTA Anchors, Evaluation Definitions, and Refined Tier F Design

**Date:** 2026-07-08
**Status:** Planning (no training code). This file does the pre-code work the author asked for: resolve the SOTA-protocol questions, kill the STORM-number conflict, and pin the exact evaluation definition for every head so the Tier F experiment measures the right thing against the right anchor.
**Decision locked:** Tier F (multi-GPU / weeks), single shared hierarchical backbone (173 §3), plan-first.

---

## 1. The STORM number is resolved — it is 0.901, not 0.506

**Verified against the primary source.** STORM-PSR ("Learning to Recognize Correctly Completed Procedure Steps in Egocentric Assembly Videos through Spatio-Temporal Modeling," CVIU 2025, arXiv:2510.12385), Table 1, **IndustReal**:

- **F1 = 0.901**, **POS = 0.812**, **τ (avg delay) = 15.5 s**
- The prior IndustReal SOTA it beats: **F1 = 0.891, POS = 0.797, τ = 21.0 s** (the WACV-2024 B3 baseline; WACV's own table reports B3 F1 = 0.883 — the 0.883/0.891 difference is the STORM authors' re-run, treat as ≈0.88–0.89).

**Action:** delete "STORM 0.506" everywhere (it originates in `168 §1` and is simply wrong). `comparability-matrix.md` and `reviewer-3-psr-paradigm-reconciliation.md` already had it right at 0.901. **Correction C-4 in 172 is now closed: the PSR SOTA anchor is STORM-PSR F1 0.901 (transition, ±-tolerance, test set).**

---

## 2. Definitive SOTA anchor table (paper · metric · number · protocol · split)

Every future comparison sentence must cite from this table, not from prose.

| Head | Anchor (paper) | Metric | Number | Protocol | Split |
|---|---|---|---|---|---|
| **Detection** | WACV-2024 (Schoonbeek) | mAP@0.5 | **0.838** (annotated frames) / **0.641** (entire videos) | YOLOv8-m, COCO→Real+Synthetic; two eval variants | **test** (10 subj) |
| **Activity** | WACV-2024 | top-1 / top-5 | **65.25 / 87.93** (RGB), 66.45/88.43 (RGB+VL+stereo), SlowFast 60.39/85.21 | MViTv2-S, Kinetics-400, 16-frame clips, **75 fine-grained classes** | **test** (10 subj) |
| **PSR (step)** | **STORM-PSR (CVIU 2025)** | F1 / POS / τ | **0.901 / 0.812 / 15.5 s** | dual-stream (ASD + spatio-temporal transformer) + procedural knowledge; **transition-event** matching within tolerance | **test** (10 subj) |
| PSR prior baselines | WACV-2024 | F1 / POS / τ | B1 0.779/0.570/14.9s · B2 0.860/0.731/22.3s · **B3 0.883/0.797/22.4s** | B1 naive state-change; B2 confidence accumulation; B3 B2+procedural rules | test |
| **Head pose** | — | fwd/up angular MAE | **none published** | — | — |

**Dataset split (pin this — it is a live comparability risk):** IndustReal is a **subject split: 12 train / 5 val / 10 test** (27 participants). WACV/STORM headline numbers above are **test-set**. Our current numbers (`bootstrap_ci.json`) are on **val** = 5 subjects (recordings `05_*, 14_*, 20_*, 24_*, 26_*` = 16 recordings / 38,036 frames). **⇒ For any head-to-head "beats SOTA" claim we must re-evaluate on the 10-subject TEST split.** Val-vs-test is not a fair comparison and a reviewer will catch it immediately.

---

## 3. Pinned evaluation definitions (exact, from the code) + what must be re-run to be comparable

For each head: the metric as the repo computes it today, the metric the SOTA anchor uses, and the gap to close.

### 3.1 Detection
- **Ours (code):** COCO mAP@0.5 via `eval_yolov8m.py:397` and `full_eval_inprocess.py:406-425`. Two variants exist: `det_mAP50_pc` = **present-class average** (classes with ≥1 GT in the eval set), and `det_mAP50_all_frames` (`_det_allframes_protocol: "coco_with_cr"`).
- **Anchor:** WACV reports mAP@0.5 in two forms — **annotated frames** (0.838) and **entire videos** (0.641). Their "entire videos" includes the 99.9%-empty frames; their "annotated frames" restricts to frames with GT.
- **Mapping:** ours-`present-class`/annotated-only ↔ WACV **0.838**; ours-`all_frames` ↔ WACV **0.641**. Report **both** ours against **both** theirs, same protocol name next to each number. Never compare our present-class number to their entire-video number (or vice-versa).
- **To be comparable:** (a) run on TEST split; (b) report the annotated-frame and entire-video numbers separately; (c) do **not** cite the Ultralytics-native 0.995 (172 C-1) as the comparison — it is a different protocol.

### 3.2 Activity
- **Ours (code):** per-frame top-1 / macro-F1 / top-5 on **69 verb-grouped** classes (`full_eval_inprocess.py:478` logs `act_top1`, `act_top1_valid_na_excluded`; clip-level in `activity_clip_*`).
- **Anchor:** clip-level top-1 on **75 fine-grained** classes, 16-frame MViTv2-S.
- **Gap:** two mismatches — (1) **taxonomy** (69 grouped vs 75 fine), (2) **temporal unit** (per-frame vs 16-frame clip). Our frozen MViTv2-S probe (0.3810) is clip-level but still on 69-grouped.
- **To be comparable:** evaluate **clip-level top-1 on the 75-class taxonomy** (or explicitly report both 75-class for comparison and 69-grouped for our task). Per-frame top-1 is **never** comparable to their clip-level number — keep calling it "per-frame action classification" (reviewer-2's rename) and only compare the clip-level 75-class number.

### 3.3 PSR — the critical one
- **Ours today (headline 0.7018):** per-frame, per-component (11-bit) state F1 with post-hoc per-component thresholds. **This is not the SOTA metric.**
- **Anchor (STORM/B3):** **transition-event F1** — greedy event matching of 0→1 step-completion events within a **±3-frame tolerance**, plus POS and delay τ.
- **The repo already implements the comparable metric.** `src/evaluation/decoder_oracle_bound.py:253` (`"Event F1 with greedy matching within tolerance. B3/STORM protocol"`, `--tolerance` default 3) and `src/evaluation/psr_transition_f1.py:event_f1`. Current transition-F1 numbers in-repo: 0.0053 (decoder full-38k) → 0.347 (retuned) → **0.6364 (D4+D1R, dense detector)** — see `SOTA_STATUS.md`.
- **To be comparable:** report **`event_f1` @ ±3 on the TEST split** as the PSR number, alongside POS and τ. **Retire 0.7018 from any STORM/B3 comparison** — it is a legitimate secondary metric (per-frame component recognition) but it is a different task. Also report τ (delay) — STORM's headline contribution is *delay reduction*, so a PSR claim that ignores τ is incomplete.
- **POS caveat (mandatory):** our POS (~0.97) exceeds STORM's 0.812 **only as a MonotonicDecoder fill-forward artifact** (any monotone prediction scores high POS; the all-zeros null scores 0.9995 — `SOTA_STATUS` null-model POS). POS goes in the appendix with the null-model disclosure, never as a "we beat STORM on POS" headline.

### 3.4 Head pose
- **Ours (code):** per-channel angular MAE = `degrees(arccos(cos(pred_unit, gt_unit)))` on the forward and up vectors separately (`gt_pose_variance.py:40` shows the arccos convention; eval in `full_eval_inprocess.py`). Headline: **fwd 9.14° (CI 7.74–10.87), up 7.78° (CI 6.89–8.81)** on 38k val frames (`bootstrap_ci.json`).
- **Anchor:** none.
- **To be comparable:** nothing to match; report fwd + up MAE with bootstrap CI on the TEST split as the **first IndustReal ego-pose baseline**. Keep position **unreported** until HoloLens export units are verified (`SOTA_STATUS §5.4 #8`). Do **not** compare to face-based head-pose estimators (OpenFace/6DRepNet) — category error (reviewer-4).

---

## 4. Comparability verdicts (what survives peer review, post-fix)

| Head | Comparable after…​ | Honest claim ceiling |
|---|---|---|
| Detection | TEST-split eval, both protocols, matched to 0.838/0.641 | **parity** target (~0.84 annotated); "beats" is unlikely and unnecessary |
| Activity | clip-level **75-class** eval on TEST | **beat 65.25** is winnable with a strong shared video backbone + transfer |
| PSR | **`event_f1`@±3 + τ** on TEST (not 0.7018) | **beat/near STORM 0.901** is the ambition; realistic first target 0.60–0.75 event-F1, then close the delay gap |
| Pose | TEST-split eval | **first baseline** (automatic), report with CI |

**Net:** the two winnable "beats SOTA" heads are **Activity** and **PSR-transition** (173 §6 stands, now with pinned metrics). Detection = parity. Pose = first baseline. This matches 173's tiered claim and is now anchored to exact, verified numbers.

---

## 5. How this sharpens the Tier F design (updates to 173)

The pinned definitions change three things in the experiment plan:

1. **PSR head must emit transition events, not just per-frame states.** The shared-backbone PSR head (173 §3) should feed the existing `event_f1`/MonotonicDecoder path so its primary metric is transition-F1@±3 + τ from day one. Reporting per-frame 0.70 as "the PSR result" would rebuild the exact comparability trap 172/reviewer-3 flagged. Add the procedural-precedence constraint as a **training** signal (reviewer-3 §3: precedence matrix currently used only at decode time; +0.05–0.10 F1 expected).
2. **Activity head must be evaluated 75-class clip-level.** Keep the 69-grouped head for our own task if useful, but the SOTA-comparison number is 75-class clip-level top-1. Decide the taxonomy before training so the label map is fixed (and never `hash()`-based — 172 E3).
3. **Every run in the controlled matrix (173 §5) reports on BOTH val and test**, with the TEST-split numbers reserved for the SOTA table. Val is for model selection; test is for the headline. Wire this into the eval harness so val/test never get mixed up (the val-vs-test mismatch is currently the single biggest comparability liability).

**Backbone choice is unchanged (173 §3): one shared hierarchical spatiotemporal backbone (Hiera primary).** The pinned metrics reinforce it — activity wants 75-class clip-level (needs temporal), PSR wants transition events (needs temporal), detection wants multiscale (needs hierarchy). One hierarchical video backbone is the only single trunk that serves all three; pose is a cheap pooled regression head on top.

---

## 6. Pre-code checklist (what's ready, what's a spec, before we write training code)

**Already exists — reuse, don't rebuild:**
- Transition-F1 eval: `decoder_oracle_bound.py`, `psr_transition_f1.py` (±3 tolerance, B3/STORM protocol). ✅
- Detection COCO mAP (both protocols): `eval_yolov8m.py`, `full_eval_inprocess.py`. ✅
- Pose angular-MAE + bootstrap CI: `full_eval_inprocess.py`, `bootstrap_ci.json` harness. ✅
- Single-task detection ablation scaffold: `ablation_det_only` (needs full-set mAP re-run). ✅

**Spec to finalize before coding (no code yet, per plan-first):**
1. **Split map:** enumerate the exact subject IDs in train(12)/val(5)/test(10) and freeze them in a config, so every run and every eval reads the same split. (Val = {05,14,20,24,26} today; confirm the 10 test subjects against the official IndustReal release.)
2. **Activity taxonomy decision:** 75-class for SOTA comparison vs 69-grouped for our task — pick, and fix the ordered label map.
3. **PSR metric contract:** primary = `event_f1`@±3 + τ on test; secondary = per-frame component F1 (appendix); POS = appendix + null disclosure.
4. **Efficiency measurement contract:** `fvcore` params+FLOPs + measured FPS/VRAM, shared-backbone-MTL vs sum-of-single-task, identical hardware (replaces the fabricated 167/170 table; the grounded estimate is ~28M–46M shared vs ~4× that for separate models — measure exactly).

---

## 7. Open items that still need an external/author check

- **Confirm the 10 test-subject IDs** against the official IndustReal split (the repo shows val = 5 subjects; the test list must be pinned from the dataset release, not guessed).
- **Confirm WACV activity is reported on 75 classes** at clip level for the 65.25 number (industreal-sota-benchmarks.md says Table 2 test set, MViTv2-S Kinetics RGB — consistent, but verify the class count in the paper before claiming taxonomy-matched).
- **B3 F1 = 0.883 (WACV) vs 0.891 (STORM's re-run)** — cite whichever matches the split you evaluate on; note the ~0.008 discrepancy so a reviewer doesn't think it's an error.

---

*Bottom line: the SOTA anchors are now pinned and verified (STORM 0.901, not 0.506), the exact per-head metric each must be compared under is fixed (detection dual-protocol mAP@0.5, activity 75-class clip top-1, PSR transition-F1@±3 + τ, pose angular MAE), and the val-vs-test split mismatch is flagged as the top comparability liability. The comparable eval code already exists in the repo. We can write the Tier F training code against these definitions with no ambiguity about what "beating SOTA" means for each head.*
