# 141 — Opus Complete Answers v2: Every Question ID in Files 134-138, Every Debate Item

**Date:** 2026-07-07
**Companion to:** `140_OPUS_ANSWERS_V2.md` (extended treatment of the 11 prioritized questions, the §-1 wiring discovery, the headline table, the 2-week plan, the 8 disclosures, and the fail-safe plan). This file completes coverage in the style of 133: **every question ID gets a verdict**, every built-in debate a ruling, every adversarial-file item a disposition.
**Coverage:** 134 (Q1-Q50, §6 Ch1-5, §7 items 1-10) · 134-debate (5 challenges, 5 gaps, 5 alternatives, 5 new questions) · 135 (Q1-Q50, §6 Debates 1-5, §7 D1-D10) · 135-debate (5+5+5+5) · 136 (57 questions, §7 Decisions 1-7) · 136-debate (5+5+5+5) · 137 (Q1-Q50, §6 A-1..A-7, §7 D-1..D-7) · 137-debate (5+5+5+5) · 138 (Q1.01-Q5.10, §6 Attacks 1-10, §8 Decisions 1-10) · 138-debate (5+5+5+5, §E) · SOTA_STATUS.md, psr_null_delta_table.md, activity_confusion_matrix.md dispositions.
**Standing corrections that propagate through every answer below** (argued in 140 §-1, all repo-verified at HEAD `7001107de`):
- **(K)** `PSR_HEAD_REPAIR=1` is a no-op; `PSRTransitionPredictor` (psr_transition.py:188) is dead code; the trained PSR head is `PSRHead` (model.py:1539, GELU heads, first-layer bias +0.1); the in-flight run is a **KENDALL_FIXED_WEIGHTS-only** ablation.
- **(P)** Head-pose training loss slices are correct (`losses.py:951-952`); the 26.20° era was eval-only; 7.78° is genuine.
- **(M)** From committed `pose_kalman_results.json` (16 recordings): up median-of-per-recording-means **7.58°**, forward **8.94°**; excluding outlier 14_assy_0_1: up 7.39°, fwd 8.46° weighted.

Verdict vocabulary: **ANSWERED** (adjudicated here, no work needed) · **RUN** (experiment/analysis scheduled, with cost) · **DESK** (document/read check) · **MOOT** (dissolved by K/P or by a prior ruling) · **SKIP** (not worth pre-freeze time) · **DEFER** (post-freeze/future work).

---

## §1. File 134 — Detection: Q1-Q50

**Q1 (why only class-22 fires from the Microsoft checkpoint).** ANSWERED: distribution/protocol mismatch between Microsoft's training regime and our recording-aware val; class 22 is high-prevalence and visually terminal. The 10-min class-histogram remains the confirmation, but per 133 D-1 the paper drops official-weights cross-eval entirely, so cap total further spend at that histogram.
**Q2 (did it learn only the terminal state).** DESK: cross-reference the log's 0-indexed `[13]` against `DET_CLASS_NAMES` once before any text names "class 22". Otherwise closed.
**Q3 (cached .pt corrupted).** ANSWERED: low probability — loads, infers, passes range checks; fail-hard fix already applied (`86ffb3436`). Closed.
**Q4 (why class 22 specifically).** SKIP beyond Q1's histogram; D1 leaves the paper.
**Q5 (static-vs-ego domain gap).** SKIP: `/shared/nl011006` path confirms institution-specific setup; unresolvable from here and unneeded.
**Q6 (v2 0.0000 vs v3 0.0004).** DESK (5 min): diff the two per-class-AP jsons; stochastic-noise hypothesis confirmed if v2 is all-zeros. Not paper content.
**Q7 (mapping check too weak).** ANSWERED: correct — it only catches the COCO-80 fallback. Adopt the per-class detection-count check next time either eval script is touched (bundle with EP-3's `--save-every`).
**Q8 (re-run at conf=0.01).** SKIP: no paper consequence once cross-eval is dropped; the calibration-vs-no-learning distinction is a curiosity.
**Q9 (GT prevalence vs detection rate).** RUN (10 min, Day 1): count non-zero `per_class_gt` entries — this is the 6-vs-9 zero-GT reconciliation (C-7) that blocks the 0.573 derivation. The correlation question itself is already answered (no simple correlation).
**Q10 (what D1 failure says about splits).** ANSWERED: nothing quotable — weights provenance, preprocessing, and split are confounded. Draw split conclusions from D1R only.
**Q11 (is 0.995 genuine).** ANSWERED provisionally yes (recording-aware split; 0.861 mAP50-95 corroborates), with 134-debate Gap 1's condition: DESK-verify D1R's dataset construction is code-identical to D3's, and commit `results.csv`. Until then "identical split" is an assertion.
**Q12 (0.861 meaning).** ANSWERED as written: well-localized boxes; 0.134 gap normal; mAP50-95 secondary metric throughout.
**Q13 (D1R overfitting).** ANSWERED: recording-aware split already ensures held-out recordings; no extra cross-recording run needed unless Q11's verification fails.
**Q14 (is 0.995 an upper bound for D3).** ANSWERED: it is a *cross-architecture* ceiling — the decision-relevant control is the single-task ConvNeXt-Tiny run (adopted as Week 2's mandatory training run, per 134-debate NQ-1). D3 exceeding YOLOv8m is not a live possibility.
**Q15 (50-epoch extension).** SKIP: 0.995→0.997 changes nothing.
**Q16 (D1R per-class AP).** RUN (30 min when the 3060 is idle): needed so ceiling and D3 share the same zero-GT convention.
**Q17 (task difficulty).** ANSWERED: detection on IndustReal is easy for modern detectors — state as the reason WACV 0.838 is a soft baseline; it's a benchmark finding, not a boast.
**Q18 (D1R as oracle input to other heads).** DEFER, except the D4+D1R eval (promoted — see Q36/Q38). Feature-level fusion is future work.
**Q19 (same val set for 0.0004 and 0.995).** DESK (10 min): confirm both scripts build `IndustRealDataset(split="val")` identically; then the gap attribution stands.
**Q20 (distillation).** DEFER to Week 2 back-half, 3-day timebox, only after the single-task baseline is running; no claim gated on it (133 D-6 stands).
**Q21 (no detection metrics in d3_full_eval).** RUN — BLOCKING (140 Q1): in-process full eval + root-cause the silent absence (suppression flag vs swallowed NaN; 134-debate NQ-2).
**Q22 ("class-balanced" meaning/inflation).** RUN: answered empirically by the full-set eval; the inflation factor (134-debate Gap 2) = full-set mAP vs 0.358. Treat 0.358 as unverified until then.
**Q23 (0.573 changes cost narrative).** ANSWERED conditionally: yes, materially — but blocked on (i) WACV convention DESK check, (ii) Q9's zero-GT count, (iii) Q21's full-set number. No narrative rewrite before all three.
**Q24 (D3 per-class mAP).** RUN: add per-class output to the Q21 full-set run (same run).
**Q25 (detection head dead like PSR).** RUN (folds into Day-1 activation diagnostics): unlikely a priori (focal/sigmoid head, different structure), but post-(K) the lesson is to verify the head that *actually runs*. One forward pass, dead-unit fraction.
**Q26 (D3 detection rate).** RUN (10 min, Day 1): detections/frame at conf ∈ {0.01, 0.05, 0.25} — distinguishes dense-but-wrong from sparse firing (134-debate NQ-3).
**Q27 (gradient-conflict fraction).** SKIP pre-freeze: PCGrad-style diagnostics don't change any decision in 2 weeks; the single-task baseline answers the actionable version.
**Q28 (class-balanced eval optimistic).** ANSWERED presumptively yes; quantified by Q22. Report both if they differ.
**Q29 (D3 mAP50-95).** RUN: same full-set run; the gap vs D1R's 0.134 separates box-quality pathology from classification pathology (134-debate Gap 3 endorsed).
**Q30 (frozen-backbone detection-head training).** SKIP: strictly dominated by the full single-task run for the cost story.
**Q31 (D4 detection pattern same as D1).** ANSWERED: same checkpoint, same sparsity; DESK-confirm both scripts load the same path (5 min).
**Q32 (mechanism of 0.000→0.347).** ANSWERED as written (thresholds tuned to ConvNeXt statistics starve sparse YOLOv8m outputs); the logit-distribution plot folds into the D4+D1R pass.
**Q33 (flat F1 plateau).** ANSWERED: plateau across 8 configs ≈ robust within the sweep; no 5-seed re-run needed — but note the whole sweep is post-hoc on reporting data (134-debate Challenge 4), so it stays a diagnostic, not a result.
**Q34 (per-component D4 breakdown).** RUN inside the D4+D1R pass.
**Q35 (is 0.347 useful).** ANSWERED: no operating point — it is negative-evidence/diagnostic; wording finalized after D4+D1R.
**Q36 (D4 with D1R weights).** RUN — PROMOTED to Week 1 (140 Q10). This reverses the file's own Q38 dismissal; with a 0.0004→0.995 backbone gap it is the decisive test of "detection density is the binding constraint."
**Q37 (detections/frame at multiple confs).** RUN: one-line statistics added to the same pass.
**Q38 (is D4 closed).** ANSWERED: NO — reversed, per Q36. 134-debate Challenge 4 is right and the file was wrong here.
**Q39 (per-recording D4 F1).** RUN: same pass (also feeds 135-debate DQ-3's sequential-order fraction).
**Q40 (why per-comp thresholds < global).** ANSWERED: both readings recorded — noise-overfit (file) vs shared correlated signal (134-debate Alt 5); the D4+D1R per-component output will separate them; one sentence in the D4 text either way.
**Q41 (WACV split).** DESK: assume random/frame-level until read from the paper; never place cross-split numbers in one column (133 D-3). Optional D1R-on-random-split re-eval only if a direct 0.838 comparison is retained (recommend it isn't).
**Q42 (WACV model on our split).** SKIP: checkpoint not available; protocol-disclosure table instead.
**Q43 (is ~0.95 substantiated).** ANSWERED: no — uncited; remove from SOTA_STATUS (Day-1 text task); WACV 0.838 is the only citable baseline.
**Q44 (0.573 → "competitive"?).** ANSWERED: narrative improves but comparability does NOT follow — the split confound survives the convention fix (134-debate Challenge 5). Pin every number to its (model, split, frame-set, convention) cell.
**Q45 (mAP50 vs 50-95 primary).** ANSWERED: both; 50 primary for comparability, 50-95 secondary.
**Q46 (annotated-frames vs entire-video).** DESK (10 min): confirm our eval is entire-video (38,036 frames says yes); then WACV 0.641 — not 0.838 — is the like-for-like row. Adopt; this is the cheapest narrative improvement in the detection section.
**Q47 (reimplement WACV protocol).** ANSWERED: no — 2-3 unbudgeted days (134-debate Gap 4 scoping); protocol-disclosure table suffices.
**Q48 (MViTv2-S detection).** ANSWERED: it has no detection head; never mix per-task baselines. Closed.
**Q49 ("BEATS SOTA" defensible).** ANSWERED: no; adopt the replacement sentence in the file's own "current answer" verbatim; purge SOTA_STATUS line 11.
**Q50 (YOLOv8m vs Faster R-CNN comparability).** ANSWERED: confounded three ways; no Faster R-CNN training; one acknowledging sentence, since no head-to-head is claimed.

**§6 built-in challenges.** Ch1 (0.358 embarrassing): survivable only in cost framing with the three provisos of Q23 resolved; if the full-set number is lower, print the lower number. Ch2 (error-state FPR): keep as one §5.4 sentence + dataset note; never a differentiated claim. Ch3 (D4 noise): keep in §5.4 as disclosure/diagnostic; wording gated on D4+D1R. Ch4 (cherry-picking): framing survives iff split identity verified + cross-architecture caveat + same-architecture baseline lands. Ch5 (36% first impression): print the convention-matched number as primary and standardize on the single "64% cost" phrasing (138 Q4.10) — recomputed if the convention check moves it.

**§7 open decisions 1-10.** 1: WACV-matched convention primary, other in footnote. 2: no "BEATS SOTA" anywhere. 3: protocol = recording-aware split, entire-video frames, mAP50 primary + 50-95 secondary, rows for D3 and D1R-ceiling explicitly labeled. 4: D4 in main §5.4, one paragraph. 5: error-state = one sentence + dataset note. 6: RUN Day 1 (Q9). 7: RUN with full-set eval (Q24). 8: RUN — blocking (Q21). 9: no PCGrad/GradVac pre-freeze. 10: distillation deferred/timeboxed (Q20).

**134-DEBATE dispositions.** Challenges: 1 UPHELD (single-task ConvNeXt scheduled Week 2 — every cost sentence caveated until it lands); 2 UPHELD (zero-GT count is Day-1); 3 UPHELD (full-set eval blocking); 4 UPHELD (D4+D1R promoted, reversing 134 Q38); 5 UPHELD (convention fix ≠ comparability; protocol cells). Gaps: 1 DESK-verify D1R split config; 2 inflation factor from full-set run; 3 mAP50-95 in same run; 4 adopt scoping (no reimplementation); 5 per-recording D4 in the D1R pass. Alternatives: 1 ADOPT the four-cell convention table (ours × both conventions; WACV's read from their paper); 2 REJECT bit-order bug as primary (0-index verified; +1-shift scored worse; histogram covers residual risk); 3 KEEP OPEN (config artifact vs interference — single-task run decides); 4 PLAUSIBLE (0.641 entire-video comparison is the cheap test, Q46); 5 NOTED as the alternative reading of Q40. New questions: 1 ADOPTED (Week-2 mandatory run); 2 ADOPTED (root-cause Day 2-3); 3 ADOPTED (Day-1 rate probe); 4 DESK — read the configured input resolution (one line); if we run below 640², the resolution caveat enters the cost text and the ablation is future work; 5 ADOPTED (per-class AP at multiple confs inside the full-set run).

---

## §2. File 135 — PSR: Q1-Q50

§1 questions re-keyed by **(K)**: the module they interrogate never ran.

**Q1 (was ReLU the sole cause).** MOOT as posed — the trained head has no ReLU. Re-keyed: "is GELU saturation in `PSRHead.output_heads` the cause?" — answered by the Day-1 activation diagnostic (`_debug_log_head0` machinery already exists, model.py:1635). The factorial ablation of the bundled repair is dissolved: the repair never ran.
**Q2 (bias=−1 gradient math).** MOOT: live head has +0.1 first-layer bias, default final bias. (The analysis — dynamic range, not gradient death — was correct about the dead module.)
**Q3 (symmetric repair vs prevalence-aware init).** DEFER into real-repair design: when wiring the actual `PSRHead` repair, consider prevalence-proportional final-bias init; check per-comp recall post-repair exactly as specified.
**Q4 (is the transformer dead).** RUN — now the central live question (Day 1, `encoded.std()`): the `[AUDIT]` +0.1-bias comment (model.py:1606-1608) names "transformer output has near-zero variance" as an already-suspected condition. If variance is collapsed, no head-level fix suffices.
**Q5 (return_states threshold paths).** MOOT in its two-path form: the PSRTransitionPredictor path is dead. Remaining check (DESK, 10 min): confirm `psr_transition_f1.py`'s thresholding matches whatever the paper reports.
**Q6 (Gaussian target coverage on sparse comps).** LIVE (losses.py:1436-1454): RUN per-component loss-contribution logging during the current run (one log line); adjust sigma/focal only post-repair.
**Q7 (monotonicity-reg conflict).** LIVE: log `reg_loss/transition_loss` once; act only if >50%.
**Q8 (input_dim 512 vs 768).** MOOT — CLOSED by this audit: dead code; live `PSRHead` dims consistent (768→512→256, model.py:1569-1577).
**Q9 (25% batch coverage).** ANSWERED (live): endorse `PSR_SEQ_EVERY_N_BATCHES=1` fine-tune **only after** a real repair shows learning.
**Q10 (transformer over-parameterized).** RUN passively: monitor train-vs-val PSR F1 in the current run; act on divergence only.
**Q11 (10k vs 5k thresholds same checkpoint).** ANSWERED: identical arrays ⇒ same sweep; 0.7810 is frame-selection luck. Confirmed or refuted by the 38k run (Day 1).
**Q12 (LOO per-recording spread).** RUN (Day-1 analysis add-on): extract per-recording improvements from `psr_loo_cv` output.
**Q13 (thresholds after repair).** ANSWERED: yes — any new checkpoint requires re-sweep + re-LOO (0.5 day, budgeted post-training).
**Q14 (sweep granularity at extremes).** RUN (10 min with the 38k pass): extend edges for comp0/7/8.
**Q15 (improvement concentrated in comp4/10).** RUN (10 min): the leave-two-at-global counterfactual; if concentrated, the text says "targeted fix for low-prevalence components" (135-debate Interp 5 adopted).
**Q16 (comp1 vs comp2 thresholds).** ANSWERED: noise; don't chase.
**Q17 (comp9 at-null).** ANSWERED: comp9 is the critical post-repair test; if still at-null after a *real* repair, treat as feature-bound (135-debate Interp 4) and report per-component.
**Q18 (macro masks degradation).** RUN: per-component LOO columns, same pass as Q12.
**Q19 (optimal global threshold).** RUN (10 min with 38k pass): sweep global 0.05-0.50 so "improvement over global" has a fair baseline.
**Q20 (LOO train/val membership).** RUN — BLOCKING for LOO interpretation (Day 1, DESK-read the split definition): report improvement separately per membership.
**Q21 (null-zeros POS 0.9995).** ANSWERED: yes — structural; the file's own analysis is correct.
**Q22 (copy-prev ≈ zeros).** ANSWERED: correct; include the closed form POS(constant) = 1 − N/(T−1) in §5.2.1.
**Q23 (POS paradox affects D4).** ANSWERED: yes; report POS at re-tuned D4 thresholds to expose the POS-F1 tradeoff (one number, cached).
**Q24 (3 recordings representative).** RUN (scheduled): extend null-POS to all 16 recordings (~2 hr with Q29's Edit extension).
**Q25 (rate-matched random null).** RUN if time (1 hr): the fairest POS null; nice-to-have, not blocking.
**Q26 (POS@tolerance expectations).** RUN (2 hr) only if the PSR section keeps any ordering claim; otherwise skip and let raw POS die in the appendix.
**Q27 (train-prevalence null).** RUN (10 min, Day 1): recompute F1_null with train prevalence; closes the DS-8 inflation concern.
**Q28 (numerical precision).** ANSWERED: computation correct, artifact real. Closed.
**Q29 (Edit score also inflated).** RUN: extend the null experiment to Edit (same script); print the 2×3 table.
**Q30 (null results on new checkpoint).** ANSWERED: re-run at freeze (protocol covers it); conclusion is metric-structural and will not change.
**Q31 (threshold relaxation vs statistics-matching).** ANSWERED as written; logit-distribution stats fold into the D4+D1R pass.
**Q32 (theoretical max D4 F1).** RUN (cheap, same pass): oracle-recall bound — fraction of GT transitions with a detection within ±3 frames.
**Q33 (per-comp < global).** ANSWERED: ordering interactions make per-component thresholds non-independent — accepted; visualize only if D4 stays prominent.
**Q34 (24→11 mapping verification).** RUN — REQUIRED before D4 text finalizes: locate and verify the mapping code (30 min, part of the D4+D1R pass). A wrong map would be D4's version of the variable-shadow bug.
**Q35 (per-video nonzero F1).** RUN: same pass.
**Q36 (joint 27-combination grid).** RUN (1 hr, cached): fold into the pass; if it beats 0.347, the current "best" wasn't.
**Q37 (min=1).** RUN: fold into the same sweep.
**Q38 (ConvNeXt→decoder, same hysteresis).** RUN (2 hr, Day 2-3): the missing 2×2 cell twin of P2.6 — without it the D4 gap can't be attributed between backbone and decoder-strictness.
**Q39 (any transitions at default).** RUN: one counter during the pass ("never fires" vs "fires but wrong").
**Q40 (YOLOv8m through repaired heads).** MOOT until a real repair exists; DEFER.
**Q41 (protocol comparability with STORM/B3).** ANSWERED: metric comparable (±3-frame event F1), paradigm not; run P2.6 and label rows per 138 Q2.10. DESK-verify each paper's exact metric before the table is typed (the doc set is internally inconsistent on whether B3 is per-frame or event-level).
**Q42 (compute-gap decomposition).** DESK only: literature check; no experiment.
**Q43 (linear procedure_order harmful).** RUN (30 min analysis): count GT order violations across recordings; if present, the hardcoded chain is actively wrong and the decoder text must say so.
**Q44 (Gaussian loss vs transition-focused).** DEFER: loss-variant comparison only after a repaired head learns; keep sigma=3 (tolerance-matched) as default.
**Q45 (per-component transition F1).** RUN (30 min): add per-component output to `psr_transition_f1.py` during P2.6; enables the STORM per-component comparison.
**Q46 (decoder oracle bound).** RUN (2 hr, with Q32): oracle logits → decoder ceiling; decides "decoder isn't the bottleneck" honestly.
**Q47 (STORM/B3 null-delta).** DESK (1 hr): compute their null-deltas from published per-component tables — the fairest available comparison; ADOPT.
**Q48 (attribution of bundled intervention).** MOOT — RESOLVED by (K): the run is single-factor (Kendall). The real head repair becomes the second single-factor arm. D10's 60-90 hr factorial is superseded by construction.
**Q49 (10k vs 38k).** RUN — Day 1 (140 Q2). The honest primary is whatever the 38k pass returns.
**Q50 (combined improvement vs SOTA).** ANSWERED: do not project; recompute the full chain (thresholds → transition F1 → null-delta → LOO) once at freeze against the winning checkpoint.

**§6 Debates 1-5.** D1 (calibration vs signal): thresholds endorsed but re-keyed to the *real* repair run; if training is inconclusive, the MI analysis (135-debate DQ-4) is the desk-level decider. D2 (LOO confound): prosecution partially upheld — Q20 membership split before the number is quoted. D3 (repair could hurt): applies to the future real-repair run; abort criterion set (val F1 global-0.10 < 0.65 twice → stop/restore). D4 (Gaussian smearing wrong): live design concern; sigma-vs-hard-BCE comparison post-repair only; the tolerance-matching defense keeps sigma=3 as default. D5 (POS trivial): both sides right — one algebra sentence + the small null table; DESK-check whether STORM/B3 report POS to size the paragraph (if they do, the disclosure earns its space).

**§7 D1-D10.** D1 (timeline): RESET per (K) — epoch-30 tests Kendall only; expected delta +0.01-0.03; do not stop early for flatness, and do not read flatness as repair failure. D2 (POS@tolerance): conditional RUN (Q26). D3 (38k): Day 1. D4 (per-video D4): merged into D4+D1R. D5 (ConvNeXt→decoder): Day 2-3 (Q38). D6 (train/val prevalence): Day 1 (Q27). D7 (encoded.std print): Day 1 — **in the live `PSRHead` path (model.py), not psr_transition.py**. D8 (oracle bound): with the D4 pass (Q46). D9 (seq-every-1 fine-tune): gated on real-repair success. D10 (factorial): SUPERSEDED — single-factor by construction.

**135-DEBATE dispositions.** Challenges: 1 (input_dim) RESOLVED-MOOT by this audit; 2 (POS trivial) ADOPT the framing — contribution is the salvage proposal + null-model template, table as support; 3 (LOO unreliable) UPHELD — per-recording, per-component, per-membership breakdown before quoting; 4 (0.7499 upper bound) UPHELD — 38k run + the one-row global/per-comp/LOO/38k table; 5 (bundled attribution) DISSOLVED by (K). Gaps: 1 CLOSED (this audit); 2 verify `psr_transition_f1.py` data structure supports per-component extraction during P2.6; 3 Day-1 38k run; 4 Day-1 prevalence check; 5 Day-1 membership check. Interpretations: 1 (repair may reduce F1) — abort criterion + cold-copied epoch-18; 2 (Kendall main driver) — being tested cleanly right now, by accident; 3 (Gaussian ramps vs decoder steps) — live; note in §5.2 as a design-tension sentence; test post-repair; 4 (comp9 feature-bound) — adopt per-component reporting; 5 (two components drive threshold gains) — Q15 counterfactual decides. New questions: 1 (rollback plan) ADOPTED — cold-copy `best.pth`, abort criterion, moderate-bias init as the fallback repair variant; 2 (hierarchical bootstrap CI) ADOPT the cheap version (bootstrap over recordings) whenever the LOO number is printed; 3 (order violations suppress D4) — folded into the D4 pass (sequential-order fraction per video); 4 (MI for comp4/10) — desk-level decider if training is inconclusive; 5 (crossing epoch) — track it, reinterpreted per (K): it now diagnoses Kendall-fix dynamics.

---

## §3. File 136 — Activity: 57 Questions

**ACT-MLP-1 (0.0236 vs random 0.0145).** RUN (5 min): binomial test; expected p<0.001 vs 1/69 — but the decision-relevant comparison is vs 0.2217, and that one fails. Report both.
**ACT-MLP-2 (why take_short_brace).** ANSWERED (prevalence shortcut); prerequisite class-frequency table is Day 1 (ACT-CM-1).
**ACT-MLP-3 (temperature scaling).** SKIP — the question rests on a misconception: temperature scaling divides logits by a positive scalar, which is argmax-invariant, so top-1 cannot change. The experiment that *can* change argmax is **logit adjustment** (subtract τ·log class-prior): RUN that instead (10 min, cached logits) — if it lifts top-1 materially, the collapse is prior-dominance rather than feature absence.
**ACT-MLP-4 (top-5/top-10).** RUN (10-line script, with the per-class pass): if top-5 is 0.30+, the model is confused-not-random (feeds 136-debate Alt 5).
**ACT-MLP-5 (object-right/verb-wrong decomposition).** RUN (30 min): the direct test of "backbone encodes objects, not actions"; high value.
**ACT-MLP-6 (balanced training didn't help).** ANSWERED: correct inference — feature quality, not sampling; add balanced-accuracy to the metric suite.
**ACT-MLP-7 (three numbers, three pipelines).** ANSWERED: adopt the unified-eval requirement at freeze (`results_frozen.json` discipline, AC-6); until then label each number with its pipeline.
**ACT-MLP-8 (2-layer probe).** RUN (30 s on cached features, in the probe battery).
**ACT-MLP-9 (-1 labels at boundaries).** RUN: folded into the transition-distance histogram (ACT-CM-6/7).
**ACT-MLP-10 (single-task MLP).** RUN (1 day) **only if** the interference claim is kept (140 Q5); otherwise the claim is deleted and this is DEFERred.
**ACT-LP-1 (probe ≈ baseline).** ANSWERED: statistically indistinguishable; "BACKBONE HAS SIGNAL" retracted (Day-1 SOTA_STATUS edit).
**ACT-LP-2 (prior-fitting vs visual signal).** RUN (15 min): label-permutation test — cheap and decisive.
**ACT-LP-3 (overfitting fixable).** RUN (30 min, cached): L2×10 + dropout + early stop; low expectation, cheap to close.
**ACT-LP-4 (k-NN probe).** RUN (10-line, Day-1 battery): parameter-free cross-check of feature quality.
**ACT-LP-5 (C3/C4/multi-scale probes).** RUN (~1 hr, battery): tests whether the null result is a C5-GAP artifact (feeds 136-debate Alt 1).
**ACT-LP-6 (-1 filtering bias).** RUN (10 min): recording-level distribution of −1 labels.
**ACT-LP-7 (L2 normalization).** RUN (5 min, battery).
**ACT-LP-8 (per-class probe accuracy).** RUN — Day 1 (already scheduled in 140).
**ACT-LP-9 (stricter gate).** ANSWERED: adopted — gate = baseline + 0.05, with significance (140 Q3).
**ACT-LP-10 (spatial probe, no GAP).** RUN (30 min, battery): conv1×1 on 7×7×768 — the standard-classifier-head control.
**ACT-CM-1 (class distribution).** RUN — Day 1 prerequisite (5-line pandas on AR_labels.csv).
**ACT-CM-2 (partial_model vs short_brace visual similarity).** DESK (10-min eyeball); footnote-level.
**ACT-CM-3 (verb-only confusion).** RUN (30 min remap): tests "verb right, object wrong" — informative either way.
**ACT-CM-4 (antonym errors boundary-localized?).** RUN: one transition-distance histogram covers CM-4, CM-6, CM-7 and MLP-9 (~1 hr total). **The temporal-ambiguity defense may not be written until this exists.**
**ACT-CM-5 (inter-annotator agreement).** DESK: check the WACV paper/supplement; if absent, the paper acknowledges the missing ceiling; no in-house study pre-freeze.
**ACT-CM-6 (transition density).** RUN: same histogram pass.
**ACT-CM-7 (accuracy vs distance-from-transition).** RUN: same pass — the direct test; if accuracy is flat in distance, the ambiguity framing collapses and comes out of the paper.
**ACT-CM-8 (classes with <10 examples).** ANSWERED by the ACT-CM-1 table.
**ACT-CM-9 (confusion symmetry).** RUN (5 min from cached matrix).
**ACT-CM-10 (collapse vs genuine confusion decomposition).** RUN (15 min): determines whether the diagnosis paragraph says "imbalance" or "features".
**ACT-ARCH-1 (probe crash root cause).** ANSWERED/CLOSED: fixed script committed (`7001107de`); run overnight; the bare-except pattern goes to the limitations paragraph (136-debate Ch4).
**ACT-ARCH-2 (mean-pool amplification).** RUN — the deciding experiment, gated per 140 Q3.
**ACT-ARCH-3 (attention pooling).** CONDITIONAL: only if mean-pool lands in the 0.22-0.27 gray zone.
**ACT-ARCH-4 (minimal TCN first).** CONDITIONAL: same gray zone — 2-4 hr Conv1D version before any 2-3 day run. Endorsed as the sequencing principle.
**ACT-ARCH-5 (frozen+temporal vs MViTv2).** ANSWERED: 0.30-0.40 would still be "first-baseline," never "competitive" (PW-3 needs ≥0.56); set expectations accordingly.
**ACT-ARCH-6 (document TCN+ViT config).** DESK (30 min) before any temporal training.
**ACT-ARCH-7 (0.53 s window too short).** CONDITIONAL: clip-length {16,32} ablation only if TCN+ViT runs; else one limitation sentence.
**ACT-ARCH-8 (single-task temporal).** DEFER post-freeze.
**ACT-ARCH-9 (VideoMAE).** CONDITIONAL, timeboxed 0.5 day: a VideoMAE-feature linear probe runs only if (a) the temporal probe fails AND (b) an activity story is still wanted; integration beyond a frozen-feature probe is future work. (136-debate wanted it first; the budget disagrees — the temporal probe is already running and costs nothing more.)
**ACT-ARCH-10 (TSN/TRN/TSM first).** PARTIAL ADOPT: TSN-style segment consensus is nearly free on cached features — fold into the temporal-probe script as a variant; full TSM/TRN out of scope.
**ACT-SOTA-1 (what the T3 match proves).** ANSWERED: clip-pipeline/protocol consistency only; add the explicit sentence; never a capability claim.
**ACT-SOTA-2 (Meccano 0.18/0.04).** QUARANTINE: do not cite anywhere until the 100-clip subset and checkpoint provenance are explained; the paper does not need this file.
**ACT-SOTA-3 (gap decomposition).** DESK: Kinetics-400 pretraining presumed dominant; no MViTv2 probe pre-freeze.
**ACT-SOTA-4 (SlowFast/I3D).** DEFER: future-work sentence.
**ACT-SOTA-5 (75→69 identity artifact).** RUN (10 min): count eval clips in the 6 merged classes; if zero, remove "grouping is benign" as a citation and state the eval set doesn't exercise the merge.
**ACT-SOTA-6 (per-clip agreement with MViTv2).** SKIP: no decision rides on it.
**ACT-SOTA-7 (clip label purity).** RUN if idle (30 min): ceiling context; nice-to-have.
**ACT-SOTA-8 (is 0.30 "competitive").** ANSWERED: no — first-baseline is activity's permanent PW-3 label.
**ACT-SOTA-9 (T3 fine-tuned or off-the-shelf).** DESK — REQUIRED before 0.6223 is cited even as verification: resolve provenance.
**ACT-SOTA-10 (report activity at all).** ANSWERED: yes, as probe/null-result subsection (140 Q7).
**ACT-ADV-1 (why care about 0.0236).** ANSWERED: the defense = baselines table + transition-distance evidence (if CM-7 supports) + latency; if CM-7 contradicts, the ambiguity clause is dropped and the defense is baselines + first-baseline framing only.
**ACT-ADV-2 (interference claim).** ANSWERED: gated on ACT-MLP-10; without it, the claim is deleted (matches 138-debate Ch4).
**ACT-ADV-3 (prove ambiguity with annotation data).** ANSWERED: soften to hypothesis unless CM-5/CM-7 provide evidence.
**ACT-ADV-4 (tricycle vs Ferrari).** ANSWERED: no MViTv2 row in results (133 SOTA-6); the drafted protocol paragraph is adopted.
**ACT-ADV-5 (Kinetics pretraining ablation).** ANSWERED: honest answer is ACT-ARCH-9's conditional probe + a limitations sentence.
**ACT-ADV-6 (crash maturity).** ANSWERED: fixed + disclosed; the systemic bare-except pattern gets its own limitations paragraph.
**ACT-ADV-7 (majority vote barely helps).** ANSWERED: confirms consistent-wrong (collapse), not random noise — cite as evidence in the CM-10 decomposition.

**§7 Decisions 1-7.** 1: activity in-paper as probe/null; thresholds per the corrected gate (140 Q3). 2: probe fixed and running tonight. 3: first-baseline framing (option 1); cost-pairing reserved for detection. 4: interference finding contingent on the single-task control; otherwise the section claims backbone-feature quality only. 5: adopt (b) — statistical condition plus explicit margin. 6: variable-length clips only if TCN+ViT trains. 7: confusion-matrix figure yes; per-class probe bar chart yes once ACT-LP-8 runs.

**136-DEBATE dispositions.** Challenges 1-5: ALL UPHELD — (1) retraction executed; (2) interference language gated on the single-task control; (3) circular gate replaced (the original pass was a false positive, and the plan should say so); (4) bare-except pattern → limitations + fix committed; (5) the "37/66 classes at zero" fact is now inside Disclosure 3 (140 §4). Gaps 1-5: 1 conditional-RUN; 2 Day-1 RUN; 3 Day-1 RUN; 4 DESK; 5 Day-2 RUN. Alternatives: 1 tested via the probe battery (spatial/C4/C3); 2 ADOPT as a framing sentence (majority prediction is the rational Bayesian act) — context, not rescue; 3 testable only via the single-task control; blend-ratio history disclosed regardless (A-6); 4 ADOPT — "first per-frame baseline" is definitionally ours pending the literature search; 5 ADOPT top-5/entropy in the metric pass. New questions: 1 RUN Day 1 (per-recording majority baseline, 10 lines); 2 DEFER (human study post-freeze); 3 RUN Week 2 (latency numbers with the FPS re-measure); 4 OPTIONAL (t-SNE as a figure only if it shows structure); 5 RUN (per-recording accuracy distribution, 30 min with the per-class pass).

---

## §4. File 137 — Head Pose: Q1-Q50

**Q1 (what review step was missing).** ANSWERED: a golden-sample unit test; adopt Q9's suite (Week 2, ~2 hr).
**Q2 (list of artifacts quoting 26.20°).** RUN (Day-1 text task): grep 26.20/26.2/13.52/13.5 across repo + .tex; produce the definitive list for the C-5 pass.
**Q3 (did the bug affect training).** RESOLVED — NO: `losses.py:951-952` slices correctly (standing correction P). Closed.
**Q4 (temporal profile of the buggy metric).** SKIP: post-mortem with no remaining decision value once Q3 resolved.
**Q5 (most trustworthy script).** ANSWERED: adopt `full_eval_stream.py` as the reference implementation; others as cross-checks.
**Q6 (other unfixed scripts).** RUN (10 min): repo-wide slice grep (`3:6`, `3:5`, `(3, 6)`) for fossils beyond head_pose_diag.py.
**Q7 (why 26.20° survived plausibility).** ANSWERED: no per-task sanity bounds existed; adopt the bounds checklist as a freeze-protocol line.
**Q8 (schema enforcement).** DEFER: `HeadPose9Dof` dataclass is a post-freeze refactor; note in fixes catalog.
**Q9 (minimum test suite).** RUN Week 2 (~2 hr): the three tests as listed, before freeze.
**Q10 (stale numbers in .tex).** RUN: part of the C-5 reconciliation pass (Q2's grep list drives it).
**Q11 (why up < forward).** RUN (10 min): GT variance/range analysis first (137-debate Ch1); do not publish the ergonomics story without it — if up's angular range is proportionally smaller, report the error/range ratio alongside raw MAE.
**Q12 (headline: mean vs median-of-medians).** ANSWERED with data (M): primary = 7.78° weighted mean; secondary = 7.58° median of per-recording means (all 16, from committed data); 5.82° is dropped or carries the "easier 9/16 subset, median-of-medians" caveat every time.
**Q13 (frame-level error distribution).** RUN (30 min): histogram + P90/P95; required for any robustness sentence (137-debate Gap 3).
**Q14 (GT noise floor).** DESK+RUN (30 min): check pose.csv for a tracking-confidence field; if present, report high-confidence-subset MAE as secondary. Never deconvolve into a claimed number.
**Q15 (fwd/up error correlation).** RUN (20-line script, Day-2 batch).
**Q16 (pose linear probe).** RUN (~1 GPU-hr, Week 1): the highest-value cheap pose experiment — bounds head-vs-backbone headroom.
**Q17 (error vs absolute orientation).** RUN: bins added to the Q13 pass.
**Q18 (systematic bias).** RUN: same pass; report the bias vector; recommend report-but-don't-subtract for the headline (subtraction admissible only if labeled calibration).
**Q19 (per-recording forward table).** DONE in this audit (M): median 8.94°, min 6.07° (24_assy_2_4), max 17.05° (14_assy_0_1); commit as a small analysis file with the Day-2 batch.
**Q20 (position reporting).** ANSWERED: option 3 (orientation-only) for this paper; option 2 (relative-only) as future work; plus 137-debate Interp 4's DESK check that config.py:853 isn't stale (10 min) — if units turn out verified, still defer position claims to the next paper rather than destabilize this one.
**Q21 (outlier hypotheses).** RUN (30 min, Day 2): 10-frame eyeball + pose.csv checks (with Q29).
**Q22 (hard case vs data quality).** ANSWERED: decided by Q21/Q29 evidence; until then the outlier stays in all aggregates.
**Q23 (Winsorize/remove/report).** ANSWERED: report both, labeled — with-outlier 7.78°/9.14°, without 7.39°/8.46° (computed, M); exclusion from the headline only if Q29 documents GT artifacts.
**Q24 (what best recordings share).** CONDITIONAL: angular-velocity analysis only if the ergonomics claim is kept after Q11.
**Q25 (between vs within variance).** RUN (30 min, Day-2 batch): the decomposition determines whether headroom advice is "diversify data" or "temporal smoothing".
**Q26 (up-advantage not universal).** ANSWERED: correct — publish the per-recording table so the mixed pattern is visible; ergonomics claim stays a hypothesis.
**Q27 (9-recording subset bias).** RESOLVED with data (M): all-16 median of means = 7.58°, so the 5.82° subset statistic is confirmed optimistic; recompute medians-of-frames over 16 from cached npz only if a median-of-medians is still wanted.
**Q28 (MAE vs recording length).** RUN: one Pearson line in the Day-2 batch.
**Q29 (tracking artifacts in outlier pose.csv).** RUN (30 min, Day 2): zeros/jumps/timestamp gaps; >5% artifacts ⇒ footnote the outlier as GT-quality.
**Q30 (fwd/up rank correlation across recordings).** RUN: one line, same batch.
**Q31 (why smoothing gains are small).** ANSWERED: accept "already temporally smooth" as the reported reason; PSD/autocorr verification only if a reviewer challenges. No more smoothing work.
**Q32 (is 2.7% significant).** ANSWERED: statistically yes, practically marginal; report the per-recording range (0.02-0.80°) in the appendix, one sentence in main.
**Q33 (SO(3) smoother).** DEFER: future-work note.
**Q34 (per-vector Q/R).** SKIP (smoothing is one sentence; not worth optimizing a supplementary number).
**Q35 (per-recording Kalman gain).** SKIP.
**Q36 (residual autocorrelation).** SKIP unless smoothing claims expand (they won't).
**Q37 (smoothing hurts 3 recordings).** ANSWERED: supports single-frame-primary; appendix table shows the negatives.
**Q38 (model vs post-processing headroom).** ANSWERED: Q16's probe is the bound; until it runs, no headroom sentence.
**Q39 (learned temporal filter).** CONDITIONAL (2 hr CPU): only if the pose section wants a headroom claim; otherwise future work.
**Q40 (quaternion smoothing).** DEFER: future work, as the file itself recommends.
**Q41 (first-baseline defensible).** ANSWERED: yes with (a) documented search protocol in supplementary, (b) scope pinned to "IndustReal protocol, head orientation". Adopt.
**Q42 (closest existing work).** DESK (Week-2 related-work day): check Jiang ECCV'22, HoloAssist NeurIPS'23, Tome CVPR'23 under comparable protocols; cite-and-compare if any reports ego-head angular MAE.
**Q43 (IMU redundancy).** ANSWERED: one positioning paragraph (visual-only value + auxiliary-signal value); the IMU-dropout simulation is unnecessary.
**Q44 (assembly-domain ego-pose prior work).** DESK: same literature pass; if empty, "first in the application domain" is claimable with the search documented.
**Q45 (recover the ~15° source).** ANSWERED: don't chase; at most a footnote-with-caveat; removed from every table (Day-1 SOTA_STATUS edit).
**Q46 (is 7.78° actually better than prior art).** ANSWERED: unanswerable until Q42/Q44 complete; no comparative language until then.
**Q47 (OpenFace/6DRepNet).** ANSWERED: category error; taxonomy paragraph; zero face-pose numbers (HP-4 stands).
**Q48 (backbone too weak).** ANSWERED: Q16's probe quantifies it; larger backbone is future work either way.
**Q49 (single-frame vs smoothed reporting).** ANSWERED: both, single-frame primary; the drafted sentence is adopted verbatim.
**Q50 (does pose validate the multi-task story).** ANSWERED: pose leads the results section (see D-4) — but with multi-task *attribution* language removed unless the single-task pose ablation runs (137-debate Ch2).

**§6 A-1..A-7.** A-1 (GT noise): report protocol-standard; add the confidence-subset secondary if pose.csv has the field. A-2 (two vectors, one head): keep both with the joint-orientation explanation as drafted. A-3 (what's the claim): the three-part contribution (first baseline + protocol + smoothness finding) is right. A-4 (single-task pose baseline): acknowledge as a gap AND remove multi-task-benefit language now; a pose-only run is optional Week-2 filler (budget ~1 day, not the debate's optimistic 2-3 hr) — only if a GPU idles. A-5 (9-DoF vs 6): the one-sentence §5.4 fix (138 Q1.05). A-6 (how many other bugs): the honest answer is the freeze protocol + Q9's test suite + this audit's finding (K) — state all three; the pattern is the pathology paper's thesis. A-7 (median vs mean): weighted mean primary, median secondary — as drafted, now with the all-16 median computed (7.58°).

**§7 D-1..D-7.** D-1: 7.78° leads; 7.58° all-16 median secondary; 5.82° dropped or heavily caveated. D-2: outlier included in the headline; excluded variant reported alongside; exclusion promoted only on documented GT artifacts (Q29). D-3: orientation-only (silence on position). D-4: head pose LEADS the results section. D-5: smoothing → supplementary + one main-text sentence. D-6: fix `head_pose_diag.py` now or stamp a WARNING banner — 15 min; it is the last live copy of the bug; do not leave it silent. D-7: ergonomics hypothesis included only with the Q11 variance analysis attached; otherwise present both hypotheses as open.

**137-DEBATE dispositions.** Challenges: 1 UPHELD (range-normalized ratio before any "up is better" sentence); 2 UPHELD (attribution language removed; optional single-task run); 3 ADOPT the honest-severity framing — and note the audit finding (K) is now the third exhibit of the same QA failure class; 4 PARTIAL (confidence-field check; the "same GT for everyone" defense is admissible because no beats-SOTA claim remains); 5 SOFTENED BY DATA (M): forward median 8.94° vs weighted mean 9.14° — the mean is only mildly outlier-driven (Δ0.2°), so the "model memorizes recordings" reading overshoots; still, publish the per-recording table. Gaps: 1 RESOLVED (P); 2 RESOLVED (computed; commit with Day-2 batch); 3 RUN (histogram); 4 RUN (correlation); 5 RUN (probe). Interpretations: 1 (GT up-axis bias) — noted, no action; 2 (outlier = annotation difference) — folded into Q21/Q29; 3 (forward as spare-capacity channel) — untestable pre-freeze, SKIP; 4 (position units may be fine) — DESK check config comment provenance; orientation-only stands regardless; 5 (Kalman as outlier suppressor) — no action; smoothing is one sentence. New questions: 1 DONE (per-recording forward table, computed here); 2 DEFER (body-proxy pose estimation — future work); 3 CONFIRMED (subset bias, handled in Q27); 4 RUN (train-set pose MAE from cached features, 30 min, Day-2 batch — cheap overfitting bound); 5 RUN if idle (error conditioned on detection success, 1 hr — also feeds 138 Q4.04).

---

## §5. File 138 — Integration: Q1.01-Q5.10

**Q1.01 (can 0.995 appear).** YES as labeled ceiling/denominator only; split-identity verification (134 Gap 1) is its admission ticket.
**Q1.02 (pose SOTA comparison).** First-baseline with documented search; §5.4 disclosure; footnote-only for the dead ~15°.
**Q1.03 (does all-first-baselines lack novelty).** Preempt with the "what this paper is about" paragraph (adopt the draft); the pathology set + cost measurement is the contribution; venue = applied/systems (Decision 1).
**Q1.04 (FiLM negative changes pose claim?).** RUN γ/β Day 1; if pass-through: drop FiLM novelty, keep the *stronger simplicity* reading ("a linear head on shared features achieves 9.14°"). Either outcome is publishable; the unmeasured state is not.
**Q1.05 (9-DoF vs 6-DoF).** Adopt the one-sentence §5.4 fix verbatim.
**Q1.06 (fwd/up asymmetry story).** RUN the 10-min GT-variance check; hypothesis (b) (narrower distribution) is testable first; claim (a) (gravity cues) only if (b) fails to explain.
**Q1.07 (abstract).** Draft both; choose the methodology/discovery version (PW-5); with 138-debate Ch3's amendment: "two proven pathologies and one theoretical analysis" until the Kendall run reports.
**Q1.08 (cost without other ceilings).** Adopt (b) — the explicit disclosure "detection is the one fully measured ceiling" — plus the Week-2 detection single-task run; pose single-task optional filler.
**Q1.09 (params comparison).** Adopt the restructure as drafted (shared-backbone savings primary; no per-head SOTA-competitive claims; deployment FPS as the practical point).
**Q1.10 (variance of headline numbers).** RUN bootstrap-over-recordings on the cached 38k eval for pose MAE and PSR F1 (cheap); mean ± σ printed for both headline claims.
**Q2.01 (expected repair gain).** REWRITTEN by (K): the repair never ran; the in-flight run tests Kendall alone (+0.01-0.03 expected); the activation diagnostic precedes the real repair's design.
**Q2.02 (transition F1 before or after repair).** BEFORE — P2.6 on epoch-18, Day 1-2; the pre-repair number is the baseline regardless of magnitude.
**Q2.03 (0.83 framing).** MOOT until a real repair runs; if reached: option 3 ("comparable under paradigm difference") with the Q2.10 table.
**Q2.04 (STORM vs B3 as primary).** B3 primary, STORM as with-procedural-knowledge bound — after a DESK verification of each paper's exact metric/paradigm (the doc set contradicts itself on B3; do not type the table until resolved).
**Q2.05 (null table helps or hurts).** KEEP, ≤1 paragraph; it preempts a worse question than it raises.
**Q2.06 (threshold protocol).** One row: global / per-comp-38k / LOO ± σ (+ the 10k number retired to a footnote once 38k exists).
**Q2.07 (D4: disclosure or salvage).** AMENDED: D4+D1R runs regardless (cheap, decisive — 140 Q10); the *expensive* salvage (fine-tuning YOLOv8m to fire densely) stays cut.
**Q2.08 (null-delta as honest metric).** ADOPT: null-delta columns lead the per-component table.
**Q2.09 (report LOO even if it shrinks the gap).** YES — commit in advance, either direction.
**Q2.10 (paradigm table).** ADOPT; build after P2.6 + the Q2.04 verification.
**Q3.01 (does 0.2169 justify TCN+ViT).** NO as it stands: probe ≈ baseline; per-class non-majority accuracy (Day 1) + temporal probe decide, under the corrected gate.
**Q3.02 (expected TCN+ViT lift).** Mechanism (a) is confirmed trivial (0.028); (b)/(c) unproven — interpret through the temporal probe rather than a separate optical-flow study.
**Q3.03 (MViTv2-S).** CUT permanently; compute reallocated.
**Q3.04 (drop activity?).** KEEP as probe head with 140 Q5's corrected framing (interference language gated).
**Q3.05 (review of 3-good-1-broken).** The anchor claim is the cost measurement + pathology set; the venue-threshold table (140 §5) is the reality check the file lacked.
**Q3.06 (verb-antonym proves what?).** Demote to supporting figure; 1.3% cannot carry the limitation claim; linear probe + baselines table are the justification.
**Q3.07 (report clip-level?).** Purge from headline; T3-bridge only.
**Q3.08 (surviving activity claim).** First-baseline + measured null result; the interference claim requires (i) single-task control and (ii) evidence the head trained at all — until both exist it is out.
**Q3.09 (minimal temporal ablation).** The fixed temporal probe IS option (b) — running tonight; it gates everything temporal.
**Q3.10 (activity metrics).** Adopt the suite: per-class F1 top-10 + macro + majority-only and minority-only accuracy (+ top-5 from ACT-MLP-4).
**Q4.01 (0.573 vs 0.358).** Day-1 convention check; report both, WACV-matched primary; the finding is the measurement, not the flattering variant.
**Q4.02 (need the 4-model grid?).** No grid (10-25 GPU-days is the whole budget); scope the claim to "detection cost under multi-task training" + explicit disclosure for unmeasured heads; the detection single-task run is the one mandatory control.
**Q4.03 (FPS).** RUN Week 2 (30 min): YOLOv8m on the RTX 3060; all FPS numbers get hardware + batch labels.
**Q4.04 (detection→pose cascade).** RUN if idle (1 hr): pose MAE conditioned on detection IoU>0.5 — informative for the dependency story (pairs with 137-debate NQ-5).
**Q4.05 (distillation needed for cost story?).** No — separate mitigation subsection if it runs; cost measurement stays pre-distillation; timebox after the baseline.
**Q4.06 (per-task vs system cost).** Detection cost is the empirical core; system overhead (params/FPS) supporting paragraph.
**Q4.07 (params/GFLOPs discrepancy).** RUN Week 2 (1 hr): re-measure on the freeze checkpoint with a committed fvcore/ptflops script; resolves C-6.
**Q4.08 (sub-$450 claim).** Keep in deployment section/footnote only.
**Q4.09 (survives a detection expert?).** Only with the same-backbone control: DESK-verify what the .tex's equal-gradient-update ablation actually was (Day 1); if same-backbone single-task, report it as the architecture-controlled measurement pending the full run.
**Q4.10 (phrasing).** Standardize "64% multi-task cost" (or its convention-corrected value) everywhere; kill "36% of ceiling" phrasings.
**Q5.01 (dead head publishable?).** YES as a detection-method + analysis finding, with the mechanism REWRITTEN per (K): GELU saturation in `PSRHead.output_heads` under low-variance transformer output; the wrong-module episode itself becomes part of the finding (monitoring blind spot: code that exists but does not execute).
**Q5.02 (generalizability).** Keep the three-condition falsifiable form, rewritten for the real mechanism: (a) small-init MLP sub-heads behind a saturating activation, (b) loss active on a minority of steps (sequence-mode batching), (c) aggregate loss curves that hide per-head gradient death.
**Q5.03 (pathology sequencing).** ADOPT + 138-debate amendment: "two proven + one theoretical" in all pre-submission text; the running Kendall-only ablation is Pathology 2's test; upgrade to three only on its evidence.
**Q5.04 (NaN checkpoint publishable?).** YES: one-paragraph cautionary tale + generalizable NaN-guard rule.
**Q5.05 (where pathologies live).** Three locations as drafted (§4 pathologies, §5.4 disclosures, §6 infrastructure).
**Q5.06 (more pathologies?).** Keep three named; A-6 (blend-ratio abandonment) folds into Pathology 2's practice paragraph; A-4/A-5/TI-1 supplementary. The (K) wiring failure joins §6/infrastructure as exhibit three of the monitoring blind spot.
**Q5.07 (class-24).** Both places (dataset section + §5.4), as drafted.
**Q5.08 (sequence-mode as 4th pathology?).** No — efficiency note; three stays the number.
**Q5.09 (venue).** AAIML main or MLSys-workshop; no NeurIPS/CVPR. Plus 138-debate NQ-1: **the actual AAIML deadline is stated nowhere in the doc set** — the lead must confirm it before the Jul-20 freeze is committed; if the deadline is <4 weeks out, compress per the venue-threshold table.
**Q5.10 (contribution sentence).** APPROVE the amended form (140 §7 item 5): two proven pathologies + one theoretical analysis + measured detection cost (caveated/controlled) + three first baselines + eight disclosures. Revise once the Kendall run reports.

**§6 Attacks 1-10 (fix rulings).** 1: fix as drafted + split verification. 2: fix = null-delta table (adopted). 3: fix = documented search protocol (adopted). 4: fix = report both conventions (adopted). 5: fix = freeze protocol **+ the (K) correction disclosed** — the repair-that-never-ran must appear in the pathology/integrity text, or an artifact reviewer will find `PSR_HEAD_REPAIR` unconsumed and conclude worse. 6: fix = probe-section placement (adopted). 7: fix = confident tone + 138-debate Interp-4 partial adoption: each disclosure also appears adjacent to the result it qualifies, with §5.4 as the index. 8: fix = supplementary + one sentence (adopted). 9: fix = paradigm clarification sentence (adopted, pending Q2.04's metric verification). 10: fix = artifact appendix + committed eval stack; the four missing evidence dirs (140 §0) are the first gap to close.

**§8 Decisions 1-10.** 1: AAIML/MLSys-ws; confirm deadline (Q5.09). 2: both, WACV-matched primary. 3: cut P5.1. 4: keep probe head. 5: amended statement approved. 6: freeze Jul 20; mid-training checkpoint accepted. 7: price in deployment section. 8: zero disclosures in the abstract (count named, content not). 9: COMMIT — including the four dirs this audit found missing. 10: fallback paper is viable at systems venues; arXiv-first only if the full-set eval bug proves unfixable.

**138-DEBATE dispositions.** Challenges: 1 ADOPT — the §0-style table keeps an explicit epistemic-status column (done in 140 §0); 2 UPHELD — actions scheduled (single-task run + caveats + Q4.09 verification); 3 PARTIAL ADOPT — "two proven + one theoretical" language, plus state the cascade hypothesis explicitly; the Kendall-only run is its falsification test (if fixing weights alone changes nothing, the Kendall-root-cause cascade weakens); 4 UPHELD — probe framing gated (Q3.08); 5 ADOPT — venue-threshold table added (140 §5). Gaps: 1 scheduled (Week-2 run); 2 RESOLVED (P — loss verified, both by commit `a7de2c140` and independently); 3 accept scope reduction — no 5-run grid; explicit "other task costs unmeasured" disclosure; 4 scheduled (FPS); 5 ADOPT the CUDA-crash disclosure paragraph with crash-frequency numbers (pull from train logs). Interpretations: 1 PARTIAL — the methodology framing carries the paper; "first baselines" alone does not; 2 OPEN — the single-task run decides; 3 see Challenge 3; 4 MITIGATED — inline pairing + confident tone; the count stays; 5 REJECTED as default — governed by the venue-threshold table, not despair. New questions: 1 CONFIRM DEADLINE (unknown in docs — blocking for schedule realism); 2 ADOPT lightweight MPU review at the Week-1 gate: pose-only and PSR-only are each viable short papers if integration fails; 3 ADOPTED (cold-copy + abort criterion, 140 §7); 4 PARTIAL — keep the 4-head system but with scoped claims; full case-study restructure is the worst-case shape (Decision 10); 5 ADOPTED (disclosure paragraph with frequency + sensitivity statement).

**§E cross-head instructions (the four "must adopt" items).** 1 (D3 full-set before §4 writing): ADOPTED — blocking. 2 (PSR input-shape print): SUPERSEDED — resolved statically by this audit (dead code; live dims consistent); replaced by the workstation no-op confirmation. 3 (per-class probe before temporal discussion): ADOPTED — Day 1. 4 (pose training-loss verification before pose writing): DONE — verified correct.

---

## §6. Evidence-File Dispositions (SOTA_STATUS.md, psr_null_delta_table.md, activity_confusion_matrix.md)

**SOTA_STATUS.md.** Day-1 edit list: remove "BEATS SOTA" (detection row), all four "~15°"/"near SOTA" pose cells, and the "BACKBONE HAS SIGNAL" verdict (replace with the CI-overlap sentence); retitle the D1R row "single-task ceiling (cross-architecture)"; add the epistemic-status column per 138-debate Challenge 1; keep the null-POS and D4 sections as-is (they are the good parts). It is not "frozen for paper" until these edits land — freezing the current language would freeze five claims files 132-141 have overturned.
**psr_null_delta_table.md.** ENDORSED as the strongest committed PSR evidence; extend with (i) train-prevalence null column (135 Q27), (ii) per-component transition-F1 column when P2.6 lands, and (iii) STORM/B3 null-deltas from their published tables (135 Q47). Lead the paper's PSR results with it.
**activity_confusion_matrix.md.** ENDORSED as evidence of collapse; extend with the class-frequency table (ACT-CM-1), the collapse-vs-confusion decomposition (ACT-CM-10), verb-only remap (ACT-CM-3), symmetry (ACT-CM-9), and the transition-distance histogram (ACT-CM-4/6/7). The verb-antonym figure survives as supporting material only — its 1.3% cannot anchor the limitation claim.

---

## §7. Consolidated Run-List Delta (items this file adds beyond 140 §5-§6)

Day-1/2 additions, all cheap: logit-adjustment test (not temperature scaling — argmax-invariant, ACT-MLP-3); label-permutation probe (ACT-LP-2); k-NN + 2-layer + L2-norm + C3/C4/spatial probe battery (ACT-LP-4/5/7/8/10 — one script, ~2 hr total); class-frequency + per-recording-majority tables (ACT-CM-1, 136-debate NQ-1); transition-distance histogram (ACT-CM-4/6/7 + MLP-9, ~1 hr); collapse-vs-confusion + symmetry + verb-only decompositions (ACT-CM-3/9/10, ~1 hr); merged-class clip count (ACT-SOTA-5, 10 min); T3 provenance check (ACT-SOTA-9, desk); D4 24→11 mapping verification (135 Q34, inside the D4+D1R pass); D4 joint 27-grid + min=1 + oracle bounds (135 Q32/36/37/46); ConvNeXt→decoder hysteresis eval (135 Q38, 2 hr); GT order-violation count (135 Q43, 30 min); global-threshold sweep + edge-granularity + leave-two-out counterfactual (135 Q19/14/15, with the 38k pass); LOO membership + train-prevalence + per-recording/per-component LOO breakdowns (135 Q20/27/12/18); pose Day-2 batch (histogram, bias, correlations, between/within variance, train-set MAE, outlier forensics — 137 Q13/15/17/18/21/25/28/29/30, NQ-4; ~half a day total on cached outputs); pose linear probe (137 Q16, 1 GPU-hr); grep passes (137 Q2/Q6); head_pose_diag.py fix (137 D-6, 15 min); STORM/B3 null-delta desk hour (135 Q47); AAIML deadline confirmation (138-debate NQ-1 — blocking for the schedule). Everything else in 134-138 is ANSWERED, SKIP, MOOT, or DEFER as marked above.

---

## §8. Bottom Line

Every question ID in 134-138 now has a verdict, every built-in debate a ruling, and every adversarial-file item a disposition — 250 questions, 25 built-in debate items, 27 open decisions, 17 attacks/adversarial responses, and 80 debate-file items. The overlay that reshapes them is unchanged from 140: the PSR repair never ran (the in-flight training is a clean Kendall-only ablation), the pose numbers are training-verified and now carry complete per-recording statistics computed from committed data, and the cheap-deciders-first discipline of 132 §4 survives intact — roughly two-thirds of the 250 questions resolve by desk work, cached-data analysis, or the rulings above, and fewer than a dozen require GPU time, of which exactly one (the single-task ConvNeXt-Tiny detection baseline) is a new training run. The three questions the documents could not answer for themselves and that no experiment here resolves: the actual AAIML deadline, the workstation's working-tree state (does the running process consume PSR_HEAD_REPAIR?), and WACV's mAP convention — all three are minutes of checking for someone with access, and all three gate paper text. Check them first.

---

**End of 141. Read after 140. Files 142+ should carry Day-1 results against the run-list in §7 and 140 §5.**
