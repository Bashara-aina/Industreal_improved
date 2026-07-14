# RISK REGISTER — AAIML 2027 Submission

**Date:** 2026-07-14 · **Review cadence:** every gate day (Day 8, 21, 33) + weekly in Phase 4
**Scoring:** Probability × Impact (blocker / major / minor). Every risk has a concrete trigger condition and a pre-decided fallback — no mid-crisis improvisation.

---

## R1 — Detection mAP ≈ 0 on the main baseline (CRIT-1, Q3/Item 57)
- **Probability:** 15% · **Impact:** BLOCKER (detection is 1 of 4 headline tasks)
- **Trigger:** mAP50-pc < 0.10 at epoch-30 partial eval of the Day-1 baseline run (checked Day 3).
- **Mitigation (pre-trigger):** present-class protocol already adopted (avoids dilution artifact); OHEM active; threshold calibration script ready (`calibrate_det_threshold.py`).
- **Fallback chain (in order, cheapest first):** (1) confidence-threshold + NMS calibration on existing checkpoint — 1 GPU-h; (2) `DET_LR_MULTIPLIER` 1.0→5.0 restart (supervisor precedent exists at 10.0) — 25 GPU-h; (3) TSBN build (Q7) — 6 h + 25 GPU-h; (4) TAL wiring (Q21) — last resort. Paper floor: if mAP50-pc lands 0.10–0.33, report honestly with the 224px-ceiling analysis; detection becomes a characterized-limitation, not a hidden failure.

## R2 — PSR event-F1 < 0.05 despite transition fix (CRIT-3, Q10/Item 59)
- **Probability:** 35% (V1 said 60%, but USE_PSR_TRANSITION=True + σ=3.0 + LeakyReLU fix are now verified active — the 0.7018 recovery evidence exists)
- **Impact:** MAJOR (PSR is the dataset's signature task; STORM anchor is 0.506)
- **Trigger:** PSR F1@±3 < 0.30 on Day-5 baseline eval.
- **Mitigation:** constant-prediction floor diagnostic (Day 5) separates "model broken" from "metric floor"; per-component rates (Q36) identify never-positive components to exclude/report.
- **Fallback:** ASL wiring already done Day 2 (flag off) → slot #3 ablation (25 GPU-h); if still <0.30, paper reports PSR with the <0.5%-positive-rate analysis (Item 76: no published solution exists at this rate — our characterization IS the contribution) + POS order metric (target 0.75+) as the constructive number.

## R3 — Activity clip top-1 < 20% (CRIT-2, Q9/Item 58)
- **Probability:** 25% · **Impact:** MAJOR
- **Trigger:** clip top-1 < 0.20 on Day-5 eval (target floor 0.35).
- **Mitigation:** LDAM-DRW active; frozen-probe evidence (0.2169 with frozen ConvNeXt, Item 5) proves the head works — a trained backbone should clear it.
- **Fallback chain:** (1) cRT retrain (Q8) — 4 h + 5 GPU-h, cheap and fast; (2) MViTv2-S run (Q39) — 50 GPU-h, pipeline exists, VRAM-check first; (3) report with confusion-matrix analysis showing tail-class collapse (F4) and verb-level granularity as secondary metric (config supports 'verb' grouping ≈10-way).

## R4 — ST baselines also perform poorly (HIGH-2, Q12/Item 62)
- **Probability:** 35% · **Impact:** MAJOR → reframes, does not kill
- **Trigger:** ≥2 ST heads below their target floors at Day 8.
- **Mitigation:** none needed in advance — this is an interpretation risk.
- **Fallback:** switch to pre-written Framing B (PAPER_OUTLINE §5): "IndustReal is hard in both regimes"; contribution pivots to pathology characterization + remedies. Both framings drafted by Day 20, so the pivot costs days, not weeks.

## R5 — Seed variance swamps MTL-vs-ST deltas (MED-3, Q13/Item 64)
- **Probability:** 30% · **Impact:** MINOR-MAJOR (weakens claims, doesn't block)
- **Trigger:** cross-seed std > 0.5 × |MTL−ST| on any headline task at Day 30.
- **Fallback:** report bootstrap-over-test-samples CIs per G9 protocol; state parity-within-noise explicitly; lean on the per-task win/loss *pattern* and efficiency story (params/FPS are deterministic).

## R6 — MediaPipe beats our pose MAE (Q4)
- **Probability:** 30% · **Impact:** MINOR (framing risk only)
- **Trigger:** MediaPipe MAE < ours on the raw all-frames protocol (Day 2).
- **Fallback:** report coverage-stratified comparison (MediaPipe returns no estimate on occluded/rear views — coverage % is the honest headline); our claim was never "beats MediaPipe," it's "first baseline on IndustReal with a defensible reference point." G3 literature supports the occlusion argument.

## R7 — GPU budget breach (both MViT AND distillation gates fire)
- **Probability:** 15% · **Impact:** MAJOR (ledger breaks: +105 GPU-h)
- **Trigger:** Day-8 activity gate AND Day-30 MTL<ST-on-≥2-tasks gate both fire.
- **Fallback:** pre-decided rule (COMPUTE_SCHEDULE): MViT wins the budget, distillation → future work (wiring documented in appendix). No debate at trigger time.

## R8 — GPU 1 hardware failure / thermal throttling mid-Phase-3
- **Probability:** 10% · **Impact:** MAJOR (76% of budget lives on GPU 1)
- **Trigger:** any unexplained run crash or >20% slowdown vs the ~0.5 GPU-h/epoch baseline rate.
- **Mitigation:** checkpoints every epoch (resume supported); runs sequenced so no more than one 50 h run is ever un-checkpointed; 28-day calendar buffer after compute ends Day 60.
- **Fallback:** shift final seeds to GPU 0's reserve (slower: ~40 h remaining ≈ 1 seed at 3060 speed) and/or drop seed 7, reporting 2 seeds with per-sample bootstrap CIs + explicit limitation note.

## R9 — Test-split gap embarrassment (HIGH-5, Q14/Item 63)
- **Probability:** 25% · **Impact:** MINOR (if reported), BLOCKER (if hidden and caught)
- **Trigger:** test metric < val metric − 10% relative, on the single Day-56–60 eval.
- **Fallback:** report both numbers + per-subject variance analysis (splits are subject-disjoint; some gap is expected and citable). Never re-tune post-test — the eval-once rule (standing rule, 30_DAY_EXECUTION_PLAN) is the mitigation.

## R10 — Scooped: new IndustReal MTL preprint before submission (Q46)
- **Probability:** 10% · **Impact:** MAJOR
- **Trigger:** G1 refresh (Day 80) finds a ≥2-task IndustReal paper.
- **Fallback:** differentiate on the 4-task scope + head-pose first + pathology analysis (unlikely to be duplicated in combination); adjust "first" claims to precise "first to jointly…" language. Do not delay submission to add experiments.

## R11 — Wiring regression breaks the frozen config late (any Q5/Q6/Q24 code)
- **Probability:** 15% · **Impact:** MAJOR (invalidates seeds already run)
- **Trigger:** any post-freeze diff to `src/` (guarded by `git tag aaiml-freeze` at Day 21 + gitnexus_detect_changes before each commit per CLAUDE.md).
- **Mitigation:** all new features land behind flags default-off; candidate config smoke-tested Day 4 before any funded run; 6D round-trip unit test guards the pose path.
- **Fallback:** seeds are cheap to re-run individually (50 GPU-h from the ~90 h reserve) if a config-affecting bug is found before Day 45; after Day 45, the bug is documented, not fixed, unless it invalidates a headline claim.

## R12 — AAIML format surprise (page limit/blind policy) discovered late (G8)
- **Probability:** 20% · **Impact:** MINOR
- **Trigger:** G8 answers arrive wk 6 and conflict with draft structure.
- **Fallback:** ablation tables → supplementary; 28-day writing buffer absorbs restructure.

---

## Open risks accepted without funded mitigation (documented as future work)
Per the NO-GO decisions in AAIML_SUBMISSION_CHECKLIST: direction-space gradient surgery unablated (Q29/Q30/Q32/Q45), per-task augmentation absent (Q20), anchor-free detection unwired (Q31), mosaic main-loop unused with cause (Q27), SWA window default (Q26), ConsMTL out of scope (Q33). None blocks submission; each is one future-work sentence.

## Risk-review checklist (run each gate day)
1. Re-read trigger conditions above against the day's written-down metrics.
2. Update probability column; any risk crossing 50% gets its fallback *scheduled*, not just noted.
3. Reconcile GPU ledger vs COMPUTE_SCHEDULE (reserve ≥ 40 h GPU 1 until Day 33 is a hard floor).
4. Confirm no post-freeze src/ diffs (R11).

**End of RISK_REGISTER.md**
