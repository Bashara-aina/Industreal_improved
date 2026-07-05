# 126 — Opus Answers to the Complete 120–125 Document Set

**Generated:** 2026-07-05
**Scope:** This single document answers every open item, incident, recommendation, and question raised across the second consultation package:

| Doc | Title | What this document answers from it |
|-----|-------|-----------------------------------|
| 120 | Current State Ultimate | Dispositions for the four broken results (PSR F1=0, D3 NaN, TTA regression, D1 dead-end), gate status updates, T0 queue closure |
| 121 | Training Logs Deep | Verdicts on all 7 recommendations (§23.3), the NaN inventory, sampler distortion, watchdog kills, pose-norm data issue |
| 122 | Metrics Deep | Ratification/correction of the honest-disclosure rulings (§6) and the 38-row comparability matrix (§8) |
| 123 | Plan to Compare Papers | Revised priority queue and fallback tiers given the D1 dead-end and the PSR collapse |
| 124 | Architecture Deep | Rulings on the epoch=-1 fix, the 10 NaN guards, the 5-bug pattern, and the assertion discipline |
| 125 | 50 Truly Deep Questions | All 50 new questions answered with verdict, expected value, and sequencing |

**Relationship to 118:** Document 118 answered the 111–117 package. This document supersedes 118 wherever the epoch-17 results or the four incident reports change a verdict, and says so explicitly in each case. Where 118's ruling stands, it is not re-argued.

---

# Section 0: Executive Verdict and the Ten Decisions That Matter Now

## 0.1 Overall state assessment

Epoch 17 is a real breakthrough — combined 0.363 → 0.414, detection 0.317 → 0.358 (pc 0.573), activity macro-F1 0.110 → 0.205, forward MAE 7.83°, and the Anomaly-2 fix verified in production. G4 passed in the strongest possible form: the canonical blind baseline scored 0.0 against model POS 0.968, meaning the POS claim is 100% visual evidence — better than any hypothesized outcome, and the flagship claim is now essentially review-proof with the disclosure paragraph.

But the T0 execution day also surfaced four failures that change the plan more than the successes do:

1. **PSR F1=0 on full validation is a genuine model collapse** (87% all-ones predictions, six components never transitioning, all transitions at frame 0) — not an eval bug. The subsample F1=0.144 was flattering. The PSR narrative must be rebuilt around this.
2. **D3 detection is NaN** (subprocess epoch-gating; the epoch=-1 default fix at evaluate.py:3342 addresses it) — the full-set detection number still does not exist.
3. **TTA regressed 25%** (0.238 vs 0.317) — broken run, not a TTA verdict.
4. **D1 is a dead end as designed** — COCO YOLOv8m scores 0.0 on ASD and the IndustReal-trained weights no longer exist publicly. This blocks D4, 117-Q34-style distillation, and 117-Q38 pseudo-labels *as specified*.

And two idle-capacity facts dominate everything: **both GPUs are idle** while the main run sits at epoch 18 of a 100-epoch schedule (watchdog-killed), and **the config dump records SUBSET_RATIO=0.02** ("2pct mode"). If the epoch-17 numbers really come from 2% of the training data, a full-data run is the single largest untapped lever in the entire project — larger than any of the 100 questions in 117+125 combined.

## 0.2 The ten decisions

1. **Verify the SUBSET_RATIO=0.02 question before anything else (30 minutes).** 120 Appendix D says SUBSET_RATIO=0.02 and "2pct mode (36→4 training recs)"; 111-era docs said SUBSET_RATIO=1.0 with 26,322 training frames; 121 §16 mentions "10/11 activity classes" in places, vs 69 elsewhere. These cannot all describe the same run. Read `resolved_config.json` for the epoch-14+ lineage and count the actual training frames in the log. If the headline run trained on 2% of data: launch the full-data run **today** on the idle 5060 Ti — every metric in the paper improves, and most of the 125 questions become premature optimizations. If it trained on full data, correct 120 and proceed.
2. **Restart the main training run regardless.** It died at epoch 18 of 100 via watchdog kill with a healthy trajectory (every metric still climbing, LR schedule barely past peak). Resume from `crash_recovery.pth` (combined=0.4140) with the watchdog timeout extended or progress-keepalive added (121 §23.3 item 4). An idle 5060 Ti while the schedule is 18% complete is the most expensive waste in the current plan.
3. **Re-run D3 with the epoch=-1 fix, and persist per-frame predictions.** This produces the missing full-set detection number (estimate 0.30–0.34 mAP50), verifies the fix chain end-to-end, and creates the artifact that Q17/Q18/Q48 and the PSR-collapse diagnosis all consume. Run with a 4h+ timeout or batched-resume so it completes 13,161/13,161 batches this time.
4. **Rebuild the PSR story in three stages, in this order:** (a) *inference-only rescue* — 117-Q18 per-component thresholds + 125-Q48 hysteresis on the D3 artifact (thresholds cannot fix components with flat logits, but they can rescue the five components whose logits do vary); (b) *training-side fix* — 125-Q14 order-regularization + 117-Q36 inverse-prevalence weighting as a resumed probe; (c) *paradigm fix* — 125-Q46 transition-detection head, promoted from journal-tier to **T1** by the collapse. New gate G6 below decides how far down this ladder the AAIML paper goes.
5. **Solve the YOLOv8m problem by retraining it yourself (new experiment "D1-R").** IndustReal ships GT boxes; ultralytics YOLOv8m trains on this scale of data in roughly a GPU-day on the 3060. One retrain unblocks four things at once: the D1 split comparison, D4 (decoder on strong detections), Q34-125 distillation, and 117-Q38 pseudo-labels. If you choose not to spend the day, the fallback is the honest published-number comparison with the weights-unavailability disclosed — acceptable, but D1-R is cheap relative to what it unlocks.
6. **Re-run TTA correctly and decomposed:** same eval entry point as training validation (now unified via epoch=-1), fresh checkpoint fingerprint logged, and in three arms — {flip only}, {flip+scales, standard NMS}, {flip+scales, Soft-NMS} — so the Soft-NMS cumulative-decay hypothesis (120 §7) is actually tested rather than guessed. Do not publish any TTA number until the no-TTA baseline reproduces 0.358 on the same path.
7. **Fix the pose-vector normalization at the data loader** (121 §23.2 item 4: forward norms ~0.02 instead of ~1.0 on 12+ recordings). Normalize to unit vectors on load. This is a data-integrity fix for the paper's anchor contribution; it can only improve the 7.83°, and the fix must be in place before the multi-seed runs so the error bars describe the corrected pipeline.
8. **Adopt the three inference-only pose wins from 125 immediately after:** Q42 Kalman smoothing (no training, expected −0.3 to −0.8°) now, Q41 6D-rotation + geodesic and Q13 uncertainty-weighting in the week-2 pose ablation run (replacing 117-Q11/Q12 as the pose-run design, which they strictly subsume).
9. **Keep the ablation suite (A1-redo, A2–A4) as the 3060's primary occupation** after D3-redo and D1-R. The efficiency thesis now has measured FPS (11.05, E1 effectively done via D3) but still zero valid multi-task-cost numbers. Unchanged from 118 Decision 4.
10. **Freeze the paper's claim set to what survived:** ego-pose 7.83° (pending pose-norm fix + seeds), POS 0.968/0.999 with the blind-baseline row (G4-passed), mAP 0.358/pc 0.573 subsample + full-set number after D3-redo, per-frame activity 0.205/top1 0.311 renamed task, efficiency 46.5M/245 GFLOPs/11.05 FPS measured. PSR F1 is now reported as the honest negative finding with the collapse mechanism analyzed (122 §6.3 ruling stands, strengthened) — which, written well, is a *contribution* (failure analysis of per-frame PSR paradigms), not a hole.

---

# Section 1: Answers to Document 120 — The Four Incidents and the Frozen Snapshot

## 1.1 PSR F1=0 (120 §6): disposition and the corrected narrative

**Ruling: real collapse, correctly diagnosed; the paper narrative changes from "detection-limited F1" to "decoder-degeneracy under threshold miscalibration plus six dead components."**

The mechanism decomposes into two distinct failure modes that need different fixes:

- **Mode A — threshold miscalibration (fixable at inference):** 98.4% of logits exceed 0.3, so every varying component fires at frame 0. For components whose logits *do* vary over time (h1, h2, h5, h6, and partially h3), raising thresholds per-component (117-Q18) or adding hysteresis (125-Q48) can recover real timing. Expected full-val F1 after inference-only rescue: 0.08–0.20 — from zero, that is the difference between "broken" and "weak but analyzable."
- **Mode B — six flat components (h3, h4, h7, h8, h9, h10; only fixable in training):** constant logits mean no threshold recovers them. This matches the liveness record exactly — 112/121 showed h4/h7–h10 gradient RMS <0.005 for many epochs. The training-side fixes that specifically target this: inverse-prevalence weighting (117-Q36), order regularization (125-Q14), sequence contrastive loss (125-Q12), and ultimately the transition head (125-Q46).

**Consistency note the paper must state:** POS=0.999/edit=0.992 on full val are artifacts of degenerate all-ones predictions (120's own analysis is correct). Therefore the paper's POS claim should be quoted from the *subsample epoch-17 value (0.969)* where predictions were non-degenerate, with the full-val 0.999 explicitly flagged as inflated by the collapse — do NOT use 0.999 as the headline even though it is the bigger number. Claiming the artifact-inflated number after documenting the artifact would be exactly the inconsistency a reviewer catches. The G4 blind-baseline result (0.0) still anchors the subsample claim.

## 1.2 D3 NaN (120 §8): disposition

**Ruling: root cause correctly identified (epoch-gating default in the subprocess path); the epoch=-1 fix (124 §20) is the right shape. Three additions before the re-run:**

1. Add the assertion 118 asked for: `(n_present == 0) == (mAP50_pc is NaN or 0)` — the same invariant has now been violated twice through two different code paths (train.py `_s()` and subprocess gating). It will be violated a third way unless it is asserted.
2. NaN must never propagate into a published-metrics JSON. The 10 NaN-guard pattern (124 §19) covers training; extend the same discipline to the metrics writer: refuse to serialize NaN, write `null` + an `errors` field instead, so a failed metric is unmistakable rather than a number-shaped hole.
3. Fix the 2h timeout properly: either 5h budget, or checkpoint the accumulator every 2,000 batches and resume. A 72%-complete eval is not a valid eval — per-class AP on rare channels is exactly what the missing 28% changes.

## 1.3 TTA regression (120 §4, §7): disposition

**Ruling: invalid run — do not analyze the 0.238 as if it measured TTA.** The three candidate causes (Soft-NMS cumulative decay, divergent code path, checkpoint staleness) are confounded in one run; Decision 6's three-arm protocol separates them. One additional prior: on the ASD taxonomy, **horizontal flip is not label-safe if any state code has left/right-asymmetric visual evidence** — verify that flip does not systematically move probability mass between mirror-confusable states before including flip in the augmentation set at all. If the flip arm alone regresses, that is the answer and no Soft-NMS theory is needed. Expected honest outcome once fixed: +0.01 to +0.04 over 0.358, more modest than 117-Q50's original +0.03–0.07 because the base model is now stronger.

## 1.4 D1 dead-end (120 §6 SOTA table): disposition

**Ruling: the 0.0 result is itself a publishable observation, and D1-R (retrain YOLOv8m on IndustReal) is the correct unblock — see Decision 5.** The observation "COCO-pretrained YOLOv8m transfers at 0.0 mAP to the ASD taxonomy" is a genuine dataset finding: ASD states are not COCO objects, and it quantifies how specialized this benchmark is. Frame it as such (one sentence, R10 in the risk register already says this). For the comparison table, until D1-R runs the detection row reads: "0.358 (ours, 4-task, our split) vs 0.838 (YOLOv8m, published, original protocol; trained weights unavailable for split-matched evaluation)" — honest and complete. If D1-R runs and lands near 0.80–0.84, the split-matched gap becomes quotable; if it lands lower, your split is harder and the gap shrinks — both outcomes help.

## 1.5 Gate status (120 §10) — updated

| Gate | 118 definition | Status now | Change |
|------|----------------|------------|--------|
| G1 (T2 launch) | T3 remap ≤0.20 | PENDING — T3 still not run; now week-2 | Unchanged; note activity at 0.205 per-frame already matches the old T2 *target* (~0.15–0.20), which weakens the case for T2 further. If per-frame 0.205 holds on full val after retraining, consider skipping T2 outright regardless of T3. |
| G2 (OHEM ablation) | epoch-30 mAP50_pc <0.55 or cls_mean <−9.5 | NOT TRIGGERED — pc=0.573 at epoch 17, ahead of the gate threshold 13 epochs early | Likely never triggers; Q5/Q2 probably retire un-run. Keep the gate armed until epoch 30. |
| G3 (PSR narrative) | D4 F1 ≥0.45 + tau within ±3 | BLOCKED as defined (D4 needs D1-R) | **Replaced by G6 below.** |
| G4 (POS claim) | blind ≤0.90 | **STRONG PASS** (blind=0.0) | Closed. Quote subsample POS per §1.1. |
| G5 (abstract) | D1/D3 in hand by Jul 13 | READY | Submit with epoch-17 subsample numbers + disclosures. |
| **G6 (new: PSR paper story)** | After D3-redo + Q18/Q48: if full-val F1 ≥0.10, publish "per-frame PSR with calibrated decoding" + collapse analysis; if <0.10, publish the collapse analysis as the finding and promote Q46 transition head to the pre-submission critical path | OPEN | New |
| **G7 (new: YOLOv8m path)** | Decide by Jul 8: spend ~1 GPU-day on D1-R, or take the disclosure fallback | OPEN | New |

## 1.6 T0 queue closure (120 §10)

Done and accepted: D1 (as a negative finding), Q43 (G4 pass), T4/act_top1, Anomaly-2 fix, body-pose freeze. Done but invalid, must re-run: D3 (NaN), TTA (broken). Still pending, unchanged priority: Q17 tau, Q18 thresholds, T3 remap. Newly added to T0 by this document: SUBSET_RATIO verification (Decision 1), main-run restart (Decision 2), pose-norm fix (Decision 7), Q48 hysteresis, Q42 Kalman.

---

# Section 2: Answers to Document 121 — The Log Analysis Recommendations

Verdicts on 121 §23.3's seven recommendations, plus the section-level findings:

1. **Pose data normalization — ACCEPT, T0 (Decision 7).** With one protocol requirement: after normalizing, re-evaluate the epoch-17 checkpoint *before* retraining, to separate "the model was compensating for bad norms" from "the model will learn better with good norms." If eval-only already improves MAE, part of the 7.83° was label noise and the paper should say so.
2. **Activity-specific sampling (decouple from DET_GT reweighting) — ACCEPT in principle, DEFER in form.** The 3.6–7.4× sampler distortion (121 §16) is real, but changing the sampler mid-lineage confounds everything. Correct vehicle: the full-data run (Decision 1) or the A3 activity-only ablation, where a class-balanced sampler is clean. Interim: 117-Q9's blend-ratio probe covers the same symptom more safely.
3. **PSR sub-head protection — ACCEPT, escalated.** This is Mode B of the collapse (§1.1). Inverse-prevalence weighting (117-Q36) + order regularization (125-Q14) in a resumed 5-epoch probe on the 3060, week 2.
4. **Watchdog timeout — ACCEPT, T0.** The watchdog has now killed at least one healthy run at epoch 18. Add a progress-based keepalive (heartbeat advances = alive) rather than a longer fixed timeout — the 1000-step heartbeat file already exists; the watchdog should read it.
5. **Mixed precision — DEFER, unchanged from 118.** BF16 (F6) remains an untested pure-upside option; test as a 2-epoch smoke run on the 3060 only if throughput becomes the binding constraint. Never flip it on the resumed main run.
6. **Resume from latest checkpoint to 30+ epochs — ACCEPT, strengthened to Decision 2.** The 121 analyst is right and understates it: the schedule says 100 epochs, not 30.
7. **PSR F1 volatility diagnosis — SUPERSEDED.** The full-val collapse *is* the diagnosis: the subsample F1 (0.033 → 0.144 → 0.128) was measuring a decoder operating near degeneracy, where small logit shifts flip many transitions between frame-0 and never. The volatility is the collapse seen through a 2.6% keyhole.

**On the NaN inventory (121 §15):** all ten NaN metrics are now explained — eff_* were fixed by the D3 pipeline (now measured: 11.05 FPS, 245 GFLOPs, 46.47M), psr_tau/psr_pos_blind/psr_f1_calibrated are the divide-by-zero consequences of the same collapse + missing Q17/Q18 machinery. After D3-redo + Q17 + Q18, this table should be empty; any survivor is a bug.

**On "eval head all zeros" (121 §23.2 item 5):** per 122 §6.4 this is the F22-adjacent gating issue (as_/ev_/act_seg_ metrics), not a model failure — expected to populate under epoch=-1 post-hoc eval. Verify in the D3-redo output; if still zero there, it graduates to a real bug.

---

# Section 3: Answers to Document 122 — Disclosure Rulings and Comparability Matrix

## 3.1 The §6 honest-disclosure rulings — ratified with four amendments

The §6 classification (genuine / misnamed / inflated / bug-zeroed) is correct and should be pasted nearly verbatim into the paper's Comparability Notes. Amendments:

1. **§6.1 item 2 (up MAE):** 122 quotes 5.82°, 120 quotes 7.06° (epoch 11) and 8.28° (full val). Reconcile before publication — one of these is from a different checkpoint or a stale read. Every headline number in the paper needs a single source-of-truth cell in one table, given the docs now disagree.
2. **§6.2 item 7 (POS):** the "expected 0.85–0.93 blind baseline" clause is now obsolete — Q43 measured 0.0. Update the disclosure row to the measured value; it makes the claim *stronger* than the planned disclosure did. And per §1.1, quote subsample POS, not the artifact-inflated full-val 0.999.
3. **§6.3 item 9 (PSR F1):** strengthen from "must not be compared directly" to "report the full-val collapse as a finding" (G6). The D4 escape hatch ("F1=X on YOLOv8m input") is blocked until D1-R.
4. **§6.3 item 12 (combined metric):** ratified — never in the paper. One addition: 120/122/123 quote *three different formulas/weights* for it (0.3/0.35/0.15/0.2 vs "0.25 each" vs the pc-fallback logic). It is a model-selection heuristic, but its definition still needs to be single-sourced because best.pth selection depended on it.

## 3.2 The §8 comparability matrix — ratified with these cell updates

- Row 1 (det_mAP50 vs P1): "After D1" → "After D1-R, else published-number comparison with weights-unavailability disclosure."
- Rows 22–23 (psr F1): "After D4" → "After D1-R + G6; currently report as negative finding."
- Row 27 (psr_tau): E2 machinery is Q17; on the collapsed decoder tau is degenerate (everything at frame 0) — compute tau only after Q18/Q48 calibration, else it measures the artifact.
- Row 28 (psr_pos_blind): "RUN Q43" → DONE, value 0.0.
- Row 36 (eff_fps): "After E1" → DONE, 11.05 measured (D3 pipeline). Both TTA and non-TTA FPS must appear if any TTA number is published.
- Rows 30–33 (as_/ev_): keep "AFTER F22 FIX" but the actual unlock is the epoch=-1 path — verify in D3-redo.

---

# Section 4: Answers to Document 123 — The Revised Plan

## 4.1 What changes in the priority queue (123 §15)

123's T0/T1/T2/SKIP structure is ratified as the baseline; these are the deltas forced by the incidents:

**Into T0 (this week):** SUBSET_RATIO verification → possible full-data launch; main-run resume with watchdog keepalive; D3-redo (epoch=-1, full 13,161 batches, persisted predictions); pose-norm fix + eval-only recheck; Q48 hysteresis + Q18 thresholds on the D3 artifact; Q42 Kalman smoothing; TTA three-arm redo; G7 decision on D1-R.

**Into T1 (weeks 2–3):** D1-R YOLOv8m retrain (if G7 says go) then D4 + distillation/pseudo-label options; PSR training-side probe (Q36-117 + Q14-125 + Q19-117 in one resumed run); pose ablation redesigned as 6D-rotation + geodesic + no-position + uncertainty (125 Q41/Q13 subsuming 117 Q11/Q12); T3 remap (G1); A1-redo + A2–A4; B1; Q15 multi-seed (after pose-norm fix); Q26-117 discriminative-LR (only if the full-data question is settled — a full-data run changes its baseline).

**Demoted/retired:** Q5/Q2 OHEM arm (G2 almost certainly never fires — detection broke 0.55 pc at epoch 17); E1 as a standalone (already measured); the 123 §19.4 contingency table's "current epoch 11" row (superseded by epoch 17).

## 4.2 The fallback plan (123 §19) — updated floor

The zero-experiment floor is now *higher* than 123 recorded: ego-pose 7.83°, POS 0.969 with measured blind baseline, pc-mAP 0.573, per-frame activity 0.205/top1 0.311, measured 11.05 FPS and 46.5M params, plus the PSR collapse analysis and the COCO-transfer-0.0 finding as honest negative results. That is already an "acceptable conference" paper by 123 §19.4's own scale, before any T1 work. The critical path to strong-accept shrinks to: D3-redo + ablations + error bars + one PSR rescue tier + (optionally) D1-R.

## 4.3 On the efficiency numbers (123 §12)

The parameter bookkeeping across 120/123/124 varies (46.47M vs "~28M current" vs 66.4M/80M/86M pipeline estimates, 30% vs 65% vs 67% savings). Ruling: publish **one** parameter table — 46.5M total / 28.6M backbone, body-pose 1.6M disclosed as frozen — and **one** pipeline comparison with the component estimates itemized and labeled as estimates (the 66.4M itemization in 120 §3 is the most defensible). Pick the resulting savings number (≈30% total-vs-total, ≈57–67% backbone-vs-pipeline depending on the pipeline figure) and use it consistently; a reviewer who finds three different savings percentages across the paper and supplement will distrust all three.

---

# Section 5: Answers to Document 124 — Architecture and Fix-Chain Rulings

1. **epoch=-1 fix (§20): correct and ratified**, with the §1.2 additions (assertion + NaN-refusing serializer + timeout). The deeper lesson: there are now *two* eval code paths (training-val and post-hoc) that have diverged twice. Add a parity test — run both paths on the same 50 batches and assert metric equality — to CI or at least to the pre-submission checklist. This single test would have caught the D3 NaN, the TTA discrepancy, and the act_top5=0.0 bug before they burned three GPU runs.
2. **The 5-bug history (§18): pattern confirmed.** All five are interface mismatches between the subprocess harness and functions written for the training loop. Same root cause as the two-paths problem above; same fix (parity test).
3. **The 10 NaN guards (§19): necessary but double-edged.** They stop NaN cascades in training, but replacing NaN with 1e-4 *silently* also hides dying losses — this is the "silent failure" pattern 118 flagged, now implemented as policy. Add a counter: every time a NaN guard fires, increment a logged counter per location. Guards firing at a nonzero steady rate is a bug signal that is currently invisible.
4. **Fix chronicle (§17): complete and consistent with 113/118.** No new untested-critical items beyond what §1–2 already schedule. F16 (`_s()` int fix) verified in production at epoch 17 closes the last open correctness item from the 118 triage except F22/F22b-on-GPU, which D3-redo closes.

---

# Section 6: Answers to Document 125 — All 50 Questions

Verdict key (as in 118): **T0** = now, inference/config-only · **T1** = before AAIML submission · **T2** = gated/conditional · **SKIP** = not for this paper (journal queue). A recurring theme: many 125 questions propose architecture or paradigm changes that reset the comparability suite — those are individually promising but collectively incompatible with a submission-cycle timeline, and are triaged accordingly. And every training-question verdict below carries one global caveat: **if Decision 1 reveals 2%-data training, run the full-data baseline first — it re-prices every hypothesis in this file.**

## Category 1 — Architecture Changes (Q1–Q5)

**Q1 ConvNeXt-V2-Tiny FCMAE pretrain — T1, the best backbone bet; merge with 117-Q26 into a single pretrain experiment.** FCMAE weights + discriminative LR is strictly the stronger version of 117-Q26's ImageNet proposal (self-supervised pretrain closer to Q48-117's MAE idea, at zero pretraining cost since the weights are published). One 25-epoch run on the 3060, week 2–3. If it delivers even half the hypothesized +0.06–0.12, it becomes the base config for every fresh run. This is the one architecture change cheap enough (drop-in weights, same architecture family) to survive the comparability-reset objection.

**Q2 DyHead — SKIP for this paper.** Net-parameter argument is neat, but it replaces all four heads at once — the maximal comparability reset — for a hypothesized +0.02–0.04 combined. Journal queue, behind Q1/Q31/Q32.

**Q3 Cross-Scale Transformer neck — SKIP for this paper.** Same reasoning as 117-Q3 (BiFPN): real expected value (+0.04–0.09 pc), wrong point in the cycle for neck surgery. Journal queue; test against BiFPN and NAS-FPN in one neck-sweep there.

**Q4 SimOTA decoupled head + dynamic assignment — T2, the one head change worth considering, gated on the detection trajectory.** Unlike Q2/Q3/Q5 this targets a *diagnosed* failure (rare-channel anchor starvation, ch16/19/22) with the standard modern fix. But detection is currently the fastest-improving metric (0.317→0.358 in 6 epochs) — do not operate on a recovering patient. Gate: if rare channels are still ~0 AP at main-run completion, SimOTA is the first architecture intervention, before any neck work.

**Q5 NAS-FPN learnable routing — SKIP.** Highest-cost neck option (search + training), speculative mechanism ("learned P3–P5 shortcut for channel 22"). Journal queue, last among the neck options.

## Category 2 — Training Recipe (Q6–Q10)

**Q6 Easy→hard state curriculum — SKIP.** The premise (backbone needs easy states first) is already served by the natural data distribution — easy states dominate frequency. The 100-epoch curriculum also cannot be tested without a full fresh run. The cheaper route to the same target (rare-channel AP) is data-side: Q16/Q19-125 sampling, which can be probed in 5 epochs.

**Q7 SAM optimizer — SKIP for this paper.** 2× backward cost on an already FP32-slow pipeline, optimizer swap invalidates the tuned LR schedule, and the flat-minima benefit is what EMA/SWA already approximate for free. Journal queue.

**Q8 SWA epochs 75–100 — T1-lite, ratified (identical to 117-Q34).** Offline checkpoint averaging after the resumed main run completes; one hour; publish whichever of EMA/SWA wins.

**Q9 Layer-wise LR decay (LLRD) — T2, fold into the Q1 pretrain run.** LLRD only makes sense with pretrained weights (its mechanism is protecting early-layer generic features). As a standalone on the current random-ish lineage it answers little; inside the Q1/Q26 experiment it is a free second factor: run the FCMAE arm with LLRD 0.9, compare against uniform 0.1× backbone.

**Q10 Two-cycle cosine — T2, and it has a natural free test.** The resumed main run *is* effectively entering a second schedule segment post-crash; and if a full-data run launches (Decision 1), schedule it as OneCycle and keep this question for the journal. Do not add schedule complexity speculatively.

## Category 3 — Loss Redesign (Q11–Q15)

**Q11 Transition-Aware Focal Loss (Hamming-cost matrix) — T1-lite as a *reweighting*, SKIP as a full redesign.** The core insight — 1–2-bit ASD confusions should cost more than random confusion — is the single best loss idea in the set, and it is testable cheaply: add a Hamming-distance-weighted term to the existing focal loss (a cost-matrix multiply, ~20 lines) in a 5-epoch resumed probe. The full "assembly_violation_penalty" machinery is journal material. Also flag it as paper-headline "novel method" material per 125's own selection strategy — but only if the probe shows signal.

**Q12 Sequence contrastive loss for PSR — T2, second-line PSR training fix.** Plausible mechanism, but for the diagnosed collapse (flat logits on six components) the more direct fixes are prevalence weighting and order regularization (Q14). Run Q12 only if the Q36+Q14 probe under-delivers. The hypothesized 0.25–0.40 F1 assumes the decoder is alive, which it currently is not on full val.

**Q13 Uncertainty-aware geodesic pose loss — T1, folded into the pose run.** Subsumes 117-Q11: geodesic + learned per-sample variance + no-position (117-Q12) + 6D representation (Q41-125) = one designed pose experiment, 25 epochs, 3060, week 2. The uncertainty output doubles as paper material (calibrated confidence for the anchor contribution).

**Q14 Order-regularization for PSR — T1, first-line training-side PSR fix (with 117-Q36).** Directly targets the collapse: it penalizes the degenerate all-at-frame-0 solution (whose empirical order diverges maximally from canonical order weighted by prevalence). 5-epoch resumed probe. Accept the hypothesized POS dip to 0.94–0.96 — still far above 0.812, and a decoder that actually times transitions is worth it.

**Q15 Multi-task NT-Xent — SKIP for this paper.** Adds a fifth loss to a four-loss balance that took 22 fix-rounds to stabilize; the gradient-conflict premise (cos −0.3 to −0.5) is still unmeasured (F12 probe has never fired — run it, per 118 Q23). Journal queue, contingent on the measured cosine actually being negative.

## Category 4 — Data Strategy (Q16–Q20)

**Q16 Transition-biased tubelet sampling — T1, the best data-side idea for both weak spots.** Oversampling temporal neighborhoods of state *changes* simultaneously feeds the rare detection channels (which are transitional states) and the PSR transition signal (whose absence is the collapse). Implementable as a sampler weight change; probe in 5 epochs resumed. Prefer this over Q6's curriculum and over 117-Q49's blunt GT-fraction knob.

**Q17 FixMatch semi-supervised detection — SKIP for this cycle.** The strongest hypothesized detection gain in the file (+0.08–0.16), but FixMatch-for-detection is a project, not a probe (pseudo-label geometry, threshold schedules, strong-aug box semantics). Journal queue, #2 behind MAE/FCMAE pretraining. If D1-R produces a strong YOLOv8m, simple pseudo-labeling (117-Q38) captures much of the same value for a tenth of the effort.

**Q18 Counterfactual inpainting augmentation — SKIP.** Generating 50K state-manipulated images requires an inpainting pipeline whose artifacts the detector will learn to shortcut on. High effort, high risk of learning "inpainting fingerprints → rare class." Journal queue with careful leakage controls.

**Q19 Learning-progress adaptive sampling — T2.** Lighter-weight than Q16 and complementary; but two adaptive samplers at once is uninterpretable. Sequence: Q16 first; add Q19 only if rare-channel AP is still flat.

**Q20 Auxiliary optical flow — SKIP.** Requires offline flow generation for 188K frames plus a new head and loss; the activity head just improved 86% in 6 epochs without it. Journal queue as the temporal-signal alternative to VideoMAE (Q40).

## Category 5 — Multi-Task Balancing (Q21–Q25)

All five propose replacing or augmenting Kendall. Global ruling, unchanged from 118 (Q22/Q24 there): **one balancing framework per paper.** The Kendall stack is instrumented, understood, and currently *working* (activity — the head all five questions aim to rescue — improved 0.110→0.205 under it while its Kendall precision was actively rising per 112 App D). The premise "activity is gradient-starved at 14.8%" is weakening in the live data.

- **Q21 CAGrad — T2**, the single method worth one 25-epoch comparison run on the 3060 *after* the ablation suite, chosen because 125's own analysis ranks it best and it has the cleanest worst-case-improvement story. Publishable as "we compared learned uncertainty weighting against a modern gradient method" — the B1 reviewer-defense, upgraded.
- **Q22 GradNorm-adaptive — SKIP** (118's Q22 ruling stands; adaptive alpha adds a hyperparameter to a rejected method).
- **Q23 IMTL-G — SKIP** (same family, no differentiated case over CAGrad).
- **Q24 DWA — SKIP** (loss-ratio weighting is the weakest signal of the four; per-task temperatures = four new hyperparameters).
- **Q25 GradVac — T2-diagnostic only:** fire the F12 cosine probe first (free, still never run). If det-pose cosine is genuinely < −0.3, GradVac vs CAGrad becomes a real question — for the journal. If cosine is ≥ −0.1, the entire conflict premise of Q15/Q21–Q25 deflates and Kendall stands unchallenged.

## Category 6 — Cross-Dataset Transfer (Q26–Q30)

**Q26 IKEA ASM pretrain → IndustReal finetune — T1 if the IKEA ASM plan (120 §13) proceeds; else T2.** The cross-dataset chapter is already planned for the paper; pretraining is the cheapest way to make it *load-bearing* (transfer benefit) rather than decorative (parallel results). Sequence after Q1: FCMAE-vs-IKEA-vs-none is one three-arm pretrain comparison, which is genuinely novel for assembly-state detection.

**Q27 Joint multi-dataset training — T2, behind Q26.** Heavier (dataset-specific heads, loader unification) and confounds the IndustReal comparability story mid-cycle. Journal-grade; Q26's sequential transfer answers the headline question cheaper.

**Q28 Domain-adversarial viewpoint invariance — SKIP.** IndustReal is egocentric with continuous viewpoint variation — there is no clean discrete "domain" for the GRL classifier; the proposed camera-index label doesn't exist in this dataset's structure. The hypothesized rare-channel gain is better pursued via Q16.

**Q29 Pose-branch pretrain on IKEA ASM head poses — T2, verify the premise first.** IKEA ASM is *third-person* (120 §13's own table says so) — whether it contains ego head-pose labels compatible with HoloLens-style forward/up vectors needs a 30-minute check before this question is even well-posed. If the labels don't exist, retire it.

**Q30 Component-bit metric learning — SKIP for this paper.** The most conceptually interesting question in the set (universal assembly representations, zero-shot states) and the clearest *next-paper* seed. It is a research program, not a pre-submission experiment.

## Category 7 — Detection-Specific (Q31–Q35)

**Q31 Objects365 pretrain — T1-alternative arm inside the Q1 pretrain comparison.** Detection-specific pretraining is the strongest a-priori candidate for a detection-bottlenecked system; adding it as a third arm (FCMAE / Objects365 / none) costs one more 25-epoch run and produces the paper's cleanest pretraining ablation. If only one arm fits the schedule, prefer Objects365 for detection impact, FCMAE for all-task balance.

**Q32 DINOv2-S frozen backbone — T2, cheap and strategically interesting.** Head-only training is *fast* (frozen backbone ≈ fraction of the compute), so this fits idle-GPU gaps. It also directly supports the efficiency thesis variant "frozen foundation features + light heads." But it is a different architecture (ViT) — results live in an ablation table, not the headline row.

**Q33 ConvNeXt-Nano — SKIP for this cycle, note the framing value.** "95% of performance at 46% fewer backbone params" would *strengthen* the efficiency thesis (contra 117-Q28's ConvNeXt-S, which cut against it) — right instinct, wrong deadline. Journal queue, paired with the scale sweep.

**Q34 YOLOv8m distillation — T2, unblocked only by D1-R (G7).** If D1-R runs, distillation vs pseudo-labeling (117-Q38) should be *one* comparison probe (soft vs hard targets from the same teacher, 10 epochs). 125's hypothesis that soft targets beat hard ones is plausible but second-order; either captures the main effect.

**Q35 Multi-scale training — T1, the best training-recipe detection lever available without a teacher.** Standard, safe, targets exactly the small-component channels, needs only anchor-scale plumbing. Fold into the full-data run's recipe if Decision 1 triggers one, else a 25-epoch probe. One caution: the FPS claim is resolution-pinned (11.05 @ 720×1280) — multi-scale *training* is fine, but publish inference at the fixed native resolution.

## Category 8 — Activity-Specific (Q36–Q40)

**Q36 Hierarchical dilated TCN — T2, gated on G1 exactly as 117-Q7 was.** If T2-the-temporal-run happens at all, run it with this architecture (receptive field ≥ action length) — ruling unchanged from 118. The stronger per-frame baseline (0.205) raises the bar the temporal head must clear, making G1 harder to pass.

**Q37 Verb-noun hierarchical head — T1-lite, the best activity idea in the set.** Cheap (two MLP branches + bilinear combine), exploits the taxonomy's actual compositional structure, directly attacks the failure mode 120 App G documents (verb-correct/noun-wrong confusions like check↔browse_instruction). 5-epoch resumed probe alongside the 117 activity bundle (Q9/Q35/Q47). The auxiliary verb supervision also fights the 32-classes-at-zero long-tail problem.

**Q38 Detection-logit augmentation of activity input — T1-lite, same probe.** ~24 extra input dims, trivially implementable, mechanistically sensible ("which component is being manipulated"). Combined with Q37: one resumed activity probe carrying {blend 2.0, smoothing 0.05, FeatureBank, verb-noun head, det-logit input} as sequential arms — a week of 3060 time answering five questions.

**Q39 SMOTE feature-space oversampling — SKIP.** Feature-space interpolation between backbone embeddings of a *changing* backbone is chasing a moving target; loss-side class weighting and Q16's sampling are cleaner attacks on the same tail. Journal footnote at best.

**Q40 VideoMAE dual-backbone — SKIP for this paper.** A second pretrained backbone abandons the single-backbone efficiency thesis for the activity task, doubling inference cost precisely where the paper claims parsimony. It is the right *journal* answer to "what would competitive temporal activity actually take," and should be cited as such in future work.

## Category 9 — Ego-Pose-Specific (Q41–Q45)

**Q41 6D rotation representation + geodesic — T1, the pose-run centerpiece.** The 6D continuity argument (Zhou et al.) is the established best practice this head skipped; combined with Q13's uncertainty weighting and 117-Q12's position removal, this is the designed pose experiment (Decision 8). One structural note: forward+up as two unit vectors *is* nearly a 6D representation already — the delta is Gram-Schmidt orthogonalization and joint rotation-matrix loss, which costs little and guarantees consistency between the two vectors (currently unconstrained; check how often predicted forward⊥up is violated — that diagnostic is free and would itself justify the change).

**Q42 Kalman smoothing — T0, run this week on existing predictions.** Inference-only, zero risk, −0.3 to −0.8° expected, and the EM-fitted noise parameters are paper-reportable. Do it after the pose-norm fix (Decision 7) so the filter is fitted to clean vectors. Disclose as a smoothing post-process with latency implications (a non-causal smoother is offline-only; use the causal filter variant for the real-time claim).

**Q43 Coarse-to-fine multi-scale pose head — T2.** Plausible but third in line behind representation (Q41) and data (pose-norm, Q44); pose is already the strongest head — allocate marginal GPU-days to the weak heads.

**Q44 IKEA ASM pose-trajectory augmentation — SKIP pending the Q29 label check.** Same premise risk as Q29 (third-person dataset); also the augmentation as described (replacing GT poses with other-dataset poses for the same frame) breaks the image-label correspondence — as written it would train the model to predict poses uncorrelated with the input image. Retire unless reformulated.

**Q45 MC Dropout — SKIP for the headline, T2 for the uncertainty appendix.** 20× inference cost destroys the FPS claim for a −0.1 to −0.6° gain that Q42 gets for free; Q13's learned variance provides calibrated uncertainty at 1× cost. Only value: an OOD-detection appendix if the factory-pilot story wants it.

## Category 10 — PSR-Specific (Q46–Q50)

**Q46 Transition-detection head — T1, promoted by the collapse (Decision 4c, gate G6).** In 118's world (F1=0.144, detection-limited) this was a journal-tier paradigm swap. In the post-collapse world it is the credible path to a competitive F1: the fill-forward per-frame paradigm has now demonstrably degenerated at scale, and the hypothesized trade (F1 0.35–0.55, POS 0.88–0.92 — still >SOTA 0.812) is exactly the paper the PSR section wants. 25–50-epoch run, 3060, weeks 2–3, *after* the inference-rescue tier establishes the per-frame paradigm's honest ceiling.

**Q47 Temporal cross-attention PSR — T2, behind Q46.** Keeps fill-forward (the constraint that just failed) while adding attention; if Q46 runs, Q47's marginal question is "attention context vs TCN context," a refinement not a rescue. Journal companion to Q46.

**Q48 Hysteresis thresholding — T0, this week, on the D3-redo artifact.** Zero training, directly attacks Mode A of the collapse, and unlike single thresholds (117-Q18) it is robust to the jitter that flat-ish logits produce. Run Q18 and Q48 as one grid (per-component {single threshold, hysteresis pair}), tuned on a held-out recording fold exactly as 118 required for Q18. Expected: full-val F1 from 0.0 to 0.08–0.20; components h3/h4/h7–h10 will stay dead (Mode B) — report per-component so the paper's decomposition is explicit.

**Q49 Detection-quality-adaptive two-stage PSR — SKIP.** A learned meta-model on top of a collapsed base model optimizes the wrong layer; revisit only after Q46 lands, at which point its premise (per-frame reliability gating) may be moot. Journal queue.

**Q50 Multi-decoder ensemble — SKIP for this paper.** Ensembling requires its members to work first; today one member is degenerate and another (transition head) doesn't exist yet. It is the natural journal follow-up *after* Q46: "fill-forward for order + transition detection for timing, gated" is a genuinely nice story — for later.

---

# Section 7: Consolidated Priority Queue (supersedes 118 §8 and 123 §15)

## T0 — This week (Jul 5–8; both GPUs currently idle)

| # | Item | Source | Cost | Why now |
|---|------|--------|------|---------|
| 1 | Verify SUBSET_RATIO / training-data question; launch full-data run if confirmed 2% | 120 App D vs 111 | 30 min + (run) | Re-prices everything (Decision 1) |
| 2 | Resume main training on 5060 Ti with watchdog keepalive | 121 §23.3 | mins to launch | 82 epochs of budget idle (Decision 2) |
| 3 | D3-redo: epoch=-1, full 13,161 batches, persisted per-frame predictions, NaN-refusing serializer + parity assertion | 120 §8, 124 §20 | 3–5h, 3060 | The missing full-set numbers; F22-on-GPU closure; feeds #5–7 |
| 4 | Pose-norm fix at loader + eval-only recheck of epoch-17 ckpt | 121 §23.2 | hrs | Data integrity for the anchor contribution (Decision 7) |
| 5 | Q18-117 + Q48-125 threshold/hysteresis grid on D3 artifact (held-out fold tuning) | collapse Mode A | 1 day | PSR F1 0 → 0.08–0.20; feeds G6 |
| 6 | Q17-117 tau distribution (post-calibration) | 117 | hrs | Honest tau; E2 closure |
| 7 | Q42-125 Kalman smoothing on pose predictions | 125 | hrs | −0.3–0.8° free |
| 8 | TTA three-arm redo on unified eval path (incl. flip-safety check) | 120 §7 | 3h | Replaces the invalid 0.238 |
| 9 | G7 decision: D1-R YOLOv8m retrain (recommended: GO, ~1 GPU-day) | 120 §6 | decision + 1d | Unblocks D4/Q34/pseudo-labels |
| 10 | Fire the F12 gradient-cosine probe once | 118 Q23, 125 Q25 | free | Settles the Cat-5 conflict premise |

## T1 — Weeks 2–3 (before AAIML submission work freezes)

PSR training probe (117-Q36 + 125-Q14 [+ Q19-117 smoothing], 5-epoch resumed) → **Q46 transition head if G6 demands it** → pretrain comparison (125-Q1 FCMAE / 125-Q31 Objects365 [/ Q26 IKEA arm], with Q9-125 LLRD as a factor) → A1-redo + A2–A4 + B1 → pose run (Q41+Q13+no-position) → activity probe bundle (117 Q9/Q35/Q47 + 125 Q37/Q38) → Q16-125 tubelet sampling probe → Q35-125 multi-scale (or fold into full-data run) → T3 remap (G1) → Q15-117 multi-seed (post pose-norm fix) → Q8-125/117-Q34 SWA offline → D4 + distillation-vs-pseudo-label probe (if D1-R ran).

## T2 — Gated

Q4 SimOTA (if rare channels still dead at run end) · Q21 CAGrad comparison (post-ablations) · Q32 DINOv2 frozen (idle-gap filler) · Q36-125 hierarchical TCN (inside T2-temporal, gate G1) · Q12-125 contrastive PSR (if Q14/Q36 under-deliver) · Q10 two-cycle, Q19-125 adaptive sampling, Q29 (after label check) · Q45 MC-dropout appendix.

## SKIP for this paper (journal queue, ordered)

Q17-125 FixMatch → Q30 component-bit metric learning → Q50 multi-decoder ensemble → Q40 VideoMAE dual-backbone → Q27 joint training → Q2 DyHead → Q3 CST / Q5 NAS-FPN neck sweep → Q11-125 full TAFL machinery → Q15-125 NT-Xent → Q33 Nano scale sweep → Q18-125 counterfactual aug → Q7 SAM → Q6 curriculum → Q20 optical flow → Q22/Q23/Q24 balancing variants → Q39 SMOTE → Q28 domain-adversarial → Q44 pose-trajectory aug (as posed) → Q49 quality-adaptive PSR.

## The one-sentence summary

Restart the idle training, settle the 2%-data question before optimizing anything, re-run the two broken evaluations on the now-unified code path, rescue PSR in three explicit tiers (calibrate → re-weight → re-architect) with a gate deciding how far the paper goes, retrain YOLOv8m yourself to unblock every teacher-dependent experiment, and freeze the paper around the claims that survived — ego-pose, blind-baseline-anchored POS, honest detection, renamed activity, measured efficiency, and two well-analyzed negative findings that are contributions in their own right.

---

*Cross-references: verdicts cite source doc/section inline; 117-Qn and 125-Qn disambiguate the two 50-question sets. Evidence base: 120 (full), 121 (§15–16, §23, structure), 122 (§6, §8, structure), 123 (§12, §15, §19, structure), 124 (§17–20, structure), 125 (all 50 questions + hypotheses + impact summary + appendices). Live-state facts (both GPUs idle, epoch-17 best=0.4140, D3 v3/TTA/D1/Q43 outcomes) are as of the 2026-07-05 snapshot in 120. Where this document contradicts 118, this document governs.*
