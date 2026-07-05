# 126 — Opus Answers to the Complete 120–125 Document Set

**Generated:** 2026-07-05 (expanded v2 — full section-by-section coverage)
**Scope:** This single document answers **every** section, appendix, anomaly, recommendation, risk, claim, and question raised across the second consultation package:

| Doc | Title | Coverage in this document |
|-----|-------|---------------------------|
| 120 | Current State Ultimate | §1 — all 18 sections + Appendices A–M, including the four incidents, the R1–R10 risk register, the factory pilot, and the 10-decision status audit |
| 121 | Training Logs Deep | §2 — all 23 sections, including all 7 recommendations, NaN inventory, sampler distortion, crash database, DET_PROBE, Kendall/liveness records |
| 122 | Metrics Deep | §3 — all 10 sections + extended sections, including the 70-metric inventory, cross-head evidence, disclosure rulings, and the 38-row comparability matrix |
| 123 | Plan to Compare Papers | §4 — all 20 sections across Parts A/B/C: paper dives, per-metric gap closures, combined-metric path, disclosure strategy, queue, calendar, risks, venue, fallback |
| 124 | Architecture Deep | §5 — all 22 sections: per-component architecture, fix chronicle, 5-bug history, NaN guards, epoch=-1 fix, line index, inference path |
| 125 | 50 Truly Deep Questions | §6 — all 50 questions individually + the impact summary, selection strategy, and Appendices A–J including the 7 SOTA claims |

**Relationship to 118:** Document 118 answered the 111–117 package. This document supersedes 118 wherever the epoch-17 results or the four incident reports change a verdict, and says so explicitly. Where 118's ruling stands, it is referenced, not re-argued. On 125's closing claim that it "replaces 117": **partially rejected** — 125 supersedes 117 only where questions overlap (noted per-question below); 117 verdicts on non-overlapping questions (e.g., 117-Q18 thresholds, 117-Q36 PSR weighting, 117-Q9 blend ratio) remain in force and appear in the consolidated queue.

---

# Section 0: Executive Verdict and the Ten Decisions That Matter Now

## 0.1 Overall state assessment

Epoch 17 is a real breakthrough — combined 0.363 → 0.414, detection 0.317 → 0.358 (pc 0.573), activity macro-F1 0.110 → 0.205, forward MAE 7.83°, and the Anomaly-2 fix verified in production. G4 passed in the strongest possible form: the canonical blind baseline scored 0.0 against model POS 0.968, meaning the POS gain is 100% visual evidence — better than any hypothesized outcome, and the flagship claim is now essentially review-proof with the disclosure paragraph.

But the T0 execution day also surfaced four failures that change the plan more than the successes do:

1. **PSR F1=0 on full validation is a genuine model collapse** (87% all-ones predictions, six components never transitioning, all transitions at frame 0) — not an eval bug. The subsample F1=0.144 was flattering. The PSR narrative must be rebuilt around this.
2. **D3 detection is NaN** (subprocess epoch-gating; the epoch=-1 default fix at evaluate.py:3342 addresses it) — the full-set detection number still does not exist.
3. **TTA regressed 25%** (0.238 vs 0.317) — broken run, not a TTA verdict.
4. **D1 is a dead end as designed** — COCO YOLOv8m scores 0.0 on ASD and the IndustReal-trained weights no longer exist publicly. This blocks D4, distillation, and pseudo-labels *as specified*.

And two idle-capacity facts dominate everything: **both GPUs are idle** while the main run sits at epoch 18 of a 100-epoch schedule (watchdog-killed), and **the config dump records SUBSET_RATIO=0.02** ("2pct mode"). If the epoch-17 numbers really come from 2% of the training data, a full-data run is the single largest untapped lever in the entire project — larger than any of the 100 questions in 117+125 combined.

## 0.2 The ten decisions

1. **Verify the SUBSET_RATIO=0.02 question before anything else (30 minutes).** 120 Appendix D says SUBSET_RATIO=0.02 and "2pct mode (36→4 training recs)"; 111-era docs said SUBSET_RATIO=1.0 with 26,322 training frames; 121 §16 mentions "10/11 activity classes" in places, vs 69 elsewhere; 120 §14 says both "egocentric" (§1) and "fixed RGB camera (not egocentric)" (§14), and both "26K train" and "~170K train." These cannot all describe the same run. Read `resolved_config.json` for the epoch-14+ lineage and count actual training frames in the log. If the headline run trained on 2% of data: launch the full-data run **today** on the idle 5060 Ti — every metric in the paper improves, and most of the 125 questions become premature optimizations. If it trained on full data, correct 120 and proceed.
2. **Restart the main training run regardless.** It died at epoch 18 of 100 via watchdog kill with a healthy trajectory (every metric still climbing, LR schedule barely past peak). Resume from `crash_recovery.pth` (combined=0.4140) with the watchdog given a progress-based keepalive (121 §23.3 item 4). An idle 5060 Ti while the schedule is 18% complete is the most expensive waste in the current plan.
3. **Re-run D3 with the epoch=-1 fix, and persist per-frame predictions.** Produces the missing full-set detection number (estimate 0.30–0.34 mAP50), verifies the fix chain end-to-end, and creates the artifact that Q17/Q18/Q48 and the PSR-collapse diagnosis all consume. Run with a 4h+ timeout or batched-resume so it completes 13,161/13,161 batches.
4. **Rebuild the PSR story in three stages, in this order:** (a) *inference-only rescue* — 117-Q18 per-component thresholds + 125-Q48 hysteresis on the D3 artifact; (b) *training-side fix* — 125-Q14 order-regularization + 117-Q36 inverse-prevalence weighting as a resumed probe; (c) *paradigm fix* — 125-Q46 transition-detection head, promoted from journal-tier to **T1** by the collapse. New gate G6 decides how far down this ladder the AAIML paper goes.
5. **Solve the YOLOv8m problem by retraining it yourself (new experiment "D1-R").** IndustReal ships GT boxes; ultralytics YOLOv8m trains on this scale in roughly a GPU-day on the 3060. One retrain unblocks four things at once: the D1 split comparison, D4, 125-Q34 distillation, and 117-Q38 pseudo-labels. Fallback: published-number comparison with the weights-unavailability disclosed.
6. **Re-run TTA correctly and decomposed:** same eval entry point as training validation (now unified via epoch=-1), fresh checkpoint fingerprint logged, three arms — {flip only}, {flip+scales, standard NMS}, {flip+scales, Soft-NMS} — so the Soft-NMS cumulative-decay hypothesis is actually tested. Do not publish any TTA number until the no-TTA baseline reproduces 0.358 on the same path.
7. **Fix the pose-vector normalization at the data loader** (121 §23.2: forward norms ~0.02 instead of ~1.0 on 12+ recordings). Normalize to unit vectors on load, re-evaluate the epoch-17 checkpoint eval-only first, then retrain. Must land before the multi-seed runs.
8. **Adopt the three inference-only pose wins from 125:** Q42 Kalman smoothing now; Q41 6D-rotation + geodesic and Q13 uncertainty-weighting in the week-2 pose ablation (subsuming 117-Q11/Q12).
9. **Keep the ablation suite (A1-redo, A2–A4) as the 3060's primary occupation** after D3-redo and D1-R. Measured FPS exists (11.05); multi-task-cost numbers still do not. Unchanged from 118 Decision 4.
10. **Freeze the paper's claim set to what survived:** ego-pose 7.83° (pending pose-norm fix + seeds), POS 0.968 subsample with the blind-baseline row, mAP 0.358/pc 0.573 subsample + full-set after D3-redo, per-frame activity 0.205/top1 0.311 renamed task, efficiency 46.5M/245 GFLOPs/11.05 FPS measured. PSR F1 becomes the honest negative finding with the collapse mechanism analyzed — written well, that is a *contribution* (failure analysis of per-frame PSR paradigms), not a hole.

---

# Section 1: Answers to Document 120 — Every Section and Appendix

## 1.1 §1 Live jobs and system resources — answered by Decisions 1–2

Both GPUs idle + epoch 18/100 + 43 GiB free RAM = restart everything (Decision 2). The PID-continuity table's epoch-12 crash lesson (5 parallel agents consumed ~10 GB and OOM-killed training) yields a standing rule: **cap concurrent agent/tooling RAM whenever training is live** — run heavy analysis agents only against the 3060 box context or nice/cgroup them below the trainer. The `CUDA_VISIBLE_DEVICES` nohup-chain bug is fixed by always launching via `env CUDA_VISIBLE_DEVICES=1 nohup …` (already learned the hard way); encode it in the launch script, not in operator memory.

## 1.2 §2 Metric trajectory (epochs 0–17) — ratified, three notes

The table is accepted as the canonical trajectory. Notes: (a) the epoch-17 PSR F1 dip (0.144→0.128) is within the volatility band that §1.5 below re-explains as decoder-near-degeneracy — do not interpret it as regression; (b) act_top1=0.311 at epoch 17 is the first trustworthy Top-1 (T4 landed) — use it, never the act_clip 0.0625 (122 §6.3 ruling); (c) the "epoch (approx)" labels for early rows reflect the split-log history — for the paper's learning-curve figure, use only epochs 7/11/17 where provenance is unambiguous, plus per-epoch curves from the resumed run onward (VAL_EVERY=1).

**Kendall dynamics at step 2501 (§2 end):** healthy and ratified — det 1.38 precision, pose capped at 1.0, activity clamped and recovering, PSR 1.49. The activity clamp change to [−2,2] with init s_act=−1 (Pathology 2 fix) is confirmed working by the 0.205 macro-F1. No action.

## 1.3 §3 D3 full validation — dispositions

- **PSR block:** collapse, real — full analysis in §1.6 below.
- **Activity full-set (macro-F1 0.057, top1 0.129 vs subsample 0.205/0.311):** the 2.6% subsample substantially overestimates activity. Rule: **the paper's activity numbers must come from the full-set eval of the final checkpoint**, with the subsample numbers used only for trajectory. Also fix `act_top5_accuracy=0.0` in the D3 output — a metric bug (should be ~0.54 per subsample); same class as the NaN family, fix in the D3-redo.
- **Pose full-set (9.94°/8.28° vs 7.83°/subsample):** same rule — full-set numbers are the honest ones; the paper quotes full-set MAE unless the pose-norm fix + final checkpoint changes them. The ~2° subsample-vs-full gap must be disclosed if any subsample number survives into the paper.
- **Efficiency block:** accepted as measured — 46.47M params, 245.3 GFLOPs, 11.05 FPS batch-1 @720×1280, 11.04 streaming. E1 is hereby closed. The sequential-pipeline comparison table (30% fewer params, ~80% fewer FLOPs, ~2.75× faster) is usable **after** the component estimates are labeled as estimates and the params bookkeeping is single-sourced (see §4.9).
- **Detection NaN:** §1.7 below.
- **position_MAE 25.84mm:** remains DO-NOT-REPORT (118 §7.10 ruling stands).

## 1.4 §4 TTA + Soft-NMS results — disposition

Invalid run; three-arm redo per Decision 6. Two additional rulings from the per-class table: (a) the observation that 18 classes are present on full val (vs 15 subsample) is *useful* — carry `det_n_present=18` as the expected value for the D3-redo assertion; (b) the per-class pattern (ch22 = 378 GT, 0.03 AP; ch13 = 57 GT, 0.0 AP) confirms the class-confusion diagnosis (118 §7.14) on the full set — whatever else the broken run got wrong, the *relative* per-class shape matches the subsample, which strengthens the "transitional-state confusion" narrative. And one prior to test in the redo: **verify horizontal flip is label-safe for ASD codes** — if any state's visual evidence is left/right-asymmetric, flip belongs out of the TTA set entirely.

## 1.5 §5 Combined-metric trajectory — ratified with two corrections

(a) The detection-breakout narrative (0.208 plateau → 0.317 → 0.358) and its coincidence with PSR F1 activation is accepted **as a training observation**, but see §3.5 for the required softening of the "cross-head gradient signal" causal claim post-collapse. (b) The all-runs comparison table correctly quarantines Phase A/B/C (118 Anomaly-5 ruling stands). The combined metric itself remains selection-only, never published (122 §6.3 item 12) — and its formula must be single-sourced, since 120 quotes 0.3/0.35/0.15/0.2 weights while 122 mentions "0.25 each" (one of these describes an older code state; resolve from train.py directly).

## 1.6 §6 PSR F1=0 root cause — disposition and corrected narrative

**Ruling: real collapse, correctly diagnosed; the paper narrative changes from "detection-limited F1" to "decoder degeneracy under threshold miscalibration plus six dead components."** The mechanism decomposes into two failure modes needing different fixes:

- **Mode A — threshold miscalibration (inference-fixable):** 98.4% of logits exceed 0.3, so every varying component fires at frame 0. For components whose logits *do* vary (h1, h2, h5, h6, partially h3), per-component thresholds (117-Q18) or hysteresis (125-Q48) can recover real timing. Expected full-val F1 after rescue: 0.08–0.20.
- **Mode B — six flat components (h3, h4, h7–h10; training-only fix):** constant logits mean no threshold recovers them. Matches the liveness record (gradient RMS <0.005 for these sub-heads across many epochs). Fixes, in order: inverse-prevalence weighting (117-Q36), order regularization (125-Q14), sequence contrastive (125-Q12), transition head (125-Q46).

**The per-component table (§6) is ratified** as the paper's Table-N candidate — per-component FAIL/MAYBE/OK status is exactly the honest decomposition reviewers reward. **Consistency requirement:** POS=0.999/edit=0.992 on full val are artifacts of the degenerate predictions (120's own analysis). Therefore quote the **subsample epoch-17 POS (0.969)** where predictions were non-degenerate, flag full-val 0.999 as collapse-inflated, and let the G4 blind baseline (0.0) anchor the claim. Claiming the artifact-inflated 0.999 after documenting the artifact is the exact inconsistency a reviewer catches — and the §10 press-release sentence currently does this (fixed in §1.10).

**The mitigation table (§6) is re-ordered** by the Mode A/B split: Q18+Q48 first (Mode A), Q36+Q14 second (Mode B), Q46 third (paradigm), T2 last (and probably never — see G1).

**SOTA comparison tables (§6):** the PSR row is ratified with the paradigm column; the detection row's key finding (COCO YOLOv8m = 0.0, gap is pretraining/data not architecture) is ratified as a publishable observation and drives G7/D1-R; the activity row is ratified with the protocol caveats already listed.

## 1.7 §8 det_mAP50 NaN root cause — disposition

Root cause correctly identified (epoch-gating default in the subprocess path); the epoch=-1 fix (124 §20) is the right shape. Three additions before the re-run: (1) the invariant assertion `(n_present == 0) == (mAP50_pc is NaN/0)` — this invariant has now been violated through two different code paths (train.py `_s()` and subprocess gating) and will be violated a third way unless asserted; (2) the metrics writer must refuse to serialize NaN — write `null` + an `errors` field so a failed metric is unmistakable; (3) fix the 2h timeout properly — 5h budget or accumulator checkpointing every 2,000 batches. A 72%-complete eval is not a valid eval; rare-channel AP is exactly what the missing 28% changes. **The 5-bug sequence table is answered in §5.2.** The "estimated D3 detection" numbers (0.30–0.34 / 0.48–0.54) are accepted as planning placeholders only — never quotable.

## 1.8 §9 The 10 investigator verdicts — status audit

| 118 Decision | Status | Ruling now |
|---|---|---|
| 1. D1→D3→D4 | D1 done (dead-end finding), D3 invalid (NaN), D4 blocked | Re-run D3; D4 waits on D1-R (G7) |
| 2. Four zero-training experiments | Q43 done (STRONG PASS), TTA broken, Q18 pending, Q17 pending | Redo TTA; run Q18/Q17 on D3-redo artifact |
| 3. Gate T2 on T3 | T3 still not run | Stands; G1 note — per-frame 0.205 further weakens the T2 case |
| 4. A2–A4 before T2 | Not started | Stands, week 2 |
| 5. OHEM gate at epoch 30 | pc=0.573 at 17 — gate won't trigger | Keep armed; expect retirement |
| 6. Freeze body-pose | DONE (config flag) | Closed |
| 7. Fix two bookkeeping bugs | Anomaly-2 DONE + verified; ckpt-dir fix status unconfirmed | Verify the ablation ckpt-dir fix before A1-redo launches |
| 8. Verify F22/F22b on GPU | Blocked by D3 NaN | Closes with D3-redo |
| 9. Dual-track venues | On track (G5 READY) | Stands |
| 10. 6-DoF only, both mAP numbers | Adopted in draft | Stands |

## 1.9 §10 Consensus, contributions, gates, T0 status — dispositions

- **Contribution ranking:** ratified with two changes. C5 (PSR POS) stays rank 3 but its evidence line updates to the measured blind baseline; **add C8: "failure analysis of per-frame PSR decoding at scale"** (the collapse mechanism + per-component decomposition) as a genuine contribution slot, replacing the likely-dropped C7 (temporal activity). C6 (embeddings/R1) remains optional garnish.
- **Gate table:** updated in §1.12 below (G4 closed-PASS; G3 replaced by G6; new G7).
- **T0 queue status table:** ratified as accurate; closure audit in §1.13.
- **Press-release sentence:** **corrected** — replace "POS 0.999 vs 0.812" with "POS 0.969 vs 0.812 (blind baseline 0.0)" per §1.6, and replace "first ego-pose baseline (7.8°)" with the full-set number once D3-redo/pose-norm settle it. The X% detection retention placeholder fills from D1-R or the disclosed published-number comparison.

## 1.10 §11 Fix impact ranking — ratified, one addition

The ranking (F1 wipe > F18 double-ramp > F22/F22b > F13 probes > FeatureBank > F16/Anomaly-2 > OneCycleLR stepping) is accepted. Addition: the OneCycleLR per-epoch-stepping fix deserves elevation in the paper's pathology section — a scheduler that never entered decay for a whole historical run is a first-class silent failure, same family as F1/F13. The "silent failure needs a runtime assertion" recommendation (118 §3) is repeated with force: of the 10 top-ranked fixes, eight were silent. §5.4 adds the NaN-guard counter as the concrete mechanism.

## 1.11 §12 117-question status table — ratified with corrections

The status tallies are accepted with these cell updates: Q40/D3 = "DONE with NaN bug" → **REDO REQUIRED**; Q50/Q1 TTA+Soft-NMS = "RUN (broken)" → **REDO REQUIRED (three-arm)**; Q41/D1 = DONE as negative finding, superseded by D1-R decision; Q16/D4 = BLOCKED pending G7, not merely "pending"; E1 = **DONE** via D3 pipeline (11.05 FPS) — the table lists it under pending T1; act_top1/T4 = DONE and verified at epoch 17. Everything else stands as tabulated.

## 1.12 §13 Factory pilot — disposition (and the section-numbering note)

(120 has two "§13"s — factory pilot and IKEA ASM plan; both answered here.)

**Factory pilot:** Usable in the AAIML paper only with its limits stated exactly as 120 already does: N=20, 80% power only for d≥0.7, observed d=0.51 nominal — so report NASA-TLX as "consistent with a medium effect, underpowered for inference" and never as a significant finding. SUS 72.3 and the 0% opt-out are the safe headline numbers. Two strategic rulings: (a) the x402 blockchain-micropayment component is **scope risk** for an ML venue — one subsection maximum, framed as deployment context, or reviewers will ask why a payments protocol is in a vision paper; consider splitting the pilot into its own HCI submission where SUS/TLX/trust instruments are the point; (b) the thematic analysis (transparency→trust, habituation, digital-literacy onboarding) is good HCI material — again, stronger in the ICHCIIS orbit than in AAIML.

**IKEA ASM plan:** GO, as the cross-dataset chapter, start ~week 3 as scheduled. Two pre-flight checks: (1) IKEA ASM is third-person — the "ego-pose analog" row in the task mapping is the weakest mapping; verify what pose labels actually exist before promising that column (this also gates 125-Q29/Q44); (2) run it with the body-pose branch deleted and the pose-norm fix in — fresh runs carry no legacy baggage (118 §7.1). If 125-Q26 (IKEA pretrain) is adopted, sequence: IKEA-train first, then it doubles as the pretrain arm.

## 1.13 §14 Dataset statistics — flagged contradictions (verification item, T0)

This section contains internal contradictions that must be resolved before any dataset table goes in the paper: (a) "Sensor: Fixed RGB camera (not egocentric)" contradicts §1.1/111/the entire ego-pose premise (IndustReal is egocentric HoloLens 2 — the §14 line is almost certainly wrong; fix it); (b) training frames "~170K" here vs 26,322 in 111 vs the 2pct question (Decision 1); (c) activity "69 verb-grouped" here vs "2/11 classes" language in §2/121 (the 11-class references appear to be PSR components leaking into activity prose — audit and correct); (d) "assembly type: IKEA furniture (Kallax)" — IndustReal is a *toy assembly* (STEM construction set), Kallax is IKEA ASM's furniture; the two datasets are cross-contaminated in this section. **Ruling: §14 is the least reliable section of 120 — rewrite it from the dataset paper + loader logs, not from memory, before any of it reaches the manuscript.**

## 1.14 §15 Architecture reference + §16 config dump — ratified as reference, discrepancy list

Both sections are accepted as the working reference with these to reconcile: activity head 672,267 (120 §15/§16) vs 687,173 (111/116) — likely the 69-vs-75-class output layer difference, confirm which is live; total params 46,454,004 (§15) vs 46,468,910 (112) vs 46.47M (D3 measured) — pick the measured one; EFFECTIVE_BATCH 32 (4×8, §16) vs 16 (4×4, 111-era) — schedule-relevant, confirm from resolved_config; PSR component prevalences differ between §16 (val subsample: comp4=0.094) and 111 (train: comp4=0.191) — both fine, label which split each describes. The per-head latency budget (§15: 90.7ms total = 11.02 FPS) now matches the measured FPS — good; publish that table.

## 1.15 §17 Comparability status — answered in §3.6 (matrix updates)

## 1.16 §18 Risk register R1–R10 — each answered

| Risk | Ruling | Action |
|------|--------|--------|
| R1 D3 NaN | CONFIRMED-OPEN → fix shipped (epoch=-1); closes with D3-redo + assertion | T0 #3 |
| R2 TTA broken | CONFIRMED-OPEN | T0 #8 three-arm redo |
| R3 single-seed | CERTAIN | Q15 seeds week 3, **after** pose-norm fix so error bars describe the corrected pipeline |
| R4 disk exhaustion | MONITORED | Ship the free-space check in the checkpoint hook now (5-line change); 738 MB × 2 writers while both-GPU experiments resume raises exposure |
| R5 ICHCIIS deadline | ON TRACK | Submit with epoch-17 subsample + disclosures; do not wait for D3-redo |
| R6 EMA/eval mismatch | UNVERIFIED | One log-line check, this week — it gates *every* published number; highest urgency-to-cost ratio in the register |
| R7 A1 triple-confounded | CONFIRMED | A1-redo protocol already specified (118 Anomaly-4); verify ckpt-dir fix first (§1.8 item 7) |
| R8 T3 not run | OPEN | Week 2; G1 pricing signal |
| R9 no pose error bars | OPEN | Same as R3; additionally the up-MAE source discrepancy (§3.4 item 1) must be resolved before seeds are averaged |
| R10 COCO YOLOv8m no-transfer | DOCUMENTED | Ratified as an honest dataset finding; one sentence in the paper |

## 1.17 §12(paper-draft) claims tables — dispositions

Claims locked in: ratified except "PSR POS 0.999 (full val)" — replace with 0.969 subsample per §1.6. Claims pending: TTA (redo), D3 detection (redo), D4 (G7), A1–A4 (T1), T2 (G1) — all consistent with the queue. Honest disclosures list: add two — the subsample-vs-full activity/pose gaps (§1.3), and the PSR collapse itself. The three-pathologies table is ratified as the paper's Training Pathologies section skeleton; add the OneCycleLR stepping fix to Pathology 1's fix list per §1.10.

## 1.18 Appendices A–M — dispositions

- **A (file index):** accepted; add 126 (this doc) as the successor decision record.
- **B (process tree):** confirms idle state; superseded by Decision 2 once training resumes.
- **C (metric formulas):** accepted as reference; the combined-metric formula conflict noted in §1.5 applies here too.
- **D (resolved config groups):** the SUBSET_RATIO=0.02 line is Decision 1's trigger; PSR_THRESHOLD=0.3 is Mode A's smoking gun (a single global 0.3 threshold against logits that exceed 0.3 on 98.4% of frames); PSR_PER_COMPONENT_WEIGHTING=False confirms Q36 is genuinely untried.
- **E (loss trajectory):** ratified; the Anomaly-1/-3 Kendall-weighted-loss caveats (118 §2) are correctly carried; the epoch-0 "psr_head backprop DEAD at init" note is expected (seq-batch cadence) and needs no action.
- **F (DET_PROBE):** ratified — the epoch-0 LOCALIZING-before-classifying pattern and the epoch-17 score_max=0.997 confirm the class-confusion (not localization) diagnosis; feeds §1.4's flip-safety and Soft-NMS work.
- **G (activity per-class, full val):** ratified as the paper's activity appendix. The ~32 never-correct classes and the majority-class confusion pattern (check↔browse_instruction) directly motivate 125-Q37 (verb-noun hierarchy) — this appendix is that question's evidence base. Note the appendix's own "Top 10" table has duplicated/misaligned class names (e.g., three rows labeled take_objects with identical 0.233) — regenerate it from the JSON before publication.
- **H (pose per-component):** ratified; the per-axis MAE table is supplementary material; position row stays unpublished.
- **I (forward strategy weeks 1–3):** superseded by §7 of this document (incidents re-ordered it); the ICHCIIS/AAIML framing paragraphs stand.
- **J (glossary):** accepted; fix the IndustReal entry per §1.13(d) (it repeats the "IKEA furniture" error).
- **K (anomaly table A1–A7):** ratified — it faithfully carries the 118 dispositions and correctly marks A2 FIXED-verified and A6 verified-by-recovery. A4's "A1-redo PENDING" stands.
- **L (hour-by-hour timeline):** accepted as the execution record; two lessons extracted — the agent-RAM rule (§1.1) and the env-var launch rule (§1.1). The 19:30 "agents failing with API 401/429" entry is why T0 #5–8 remain pending; those items now run as plain scripts, not agents.
- **M (version history):** accepted; this document (126) extends the package and, per its own convention, supersedes 118 as the active decision record.

---

# Section 2: Answers to Document 121 — All 23 Sections

**§1 (log inventory) + §2 (campaign architecture):** accepted as the historical record; no decisions required. The 84-file inventory justifies one hygiene action: archive pre-RF4 logs to a `historical/` subtree so no future analysis mixes quarantined Phase A/B/C data (Anomaly-5 ruling) into RF4 tables by glob accident.

**§3 (primary run deep-dive) + §4 (epoch-level training metrics) + §5 (all validation metrics ever):** ratified as the source tables behind 120 §2; the §1.2 provenance rule (only unambiguous epochs in the paper figure) applies.

**§6 (per-class activity across epochs):** ratified; combined with 120 App G it shows the never-predicted set *shrinking* over epochs (pred_distinct rising) — quote that trend in the activity subsection as evidence the head is learning the tail slowly rather than ignoring it statically.

**§7 (DET_PROBE verdicts):** ratified — consistent LOCALIZING with rising confidence; feeds the class-confusion narrative. No action beyond §1.4's flip-safety check.

**§8 (Kendall log-var evolution):** ratified; the epoch-12+ dynamics (act precision rising) are the counter-evidence to 125's "activity is gradient-starved at 14.8%" premise — cited in §6 Category 5's global ruling.

**§9 (liveness probes):** ratified; the h4/h7–h10 near-zero gradient record is Mode B's training-side evidence (§1.6). The "PSR NO_GRAD at resume step 1" note is expected (seq cadence); no action.

**§10 (crash database):** accepted as complete. One synthesis the document doesn't draw: **every remaining crash class now has a specific mitigation** — RAM OOM → agent-RAM rule + NUM_WORKERS=0; GPU misassignment → env-launch rule; watchdog false-kill → keepalive (Decision 2); cuDNN/cuSOLVER → already pinned. If a *new* crash class appears after these, it deserves a fresh root-cause, not a restart-and-hope.

**§11 (LR schedule):** ratified; confirms the per-epoch stepping fix; the resumed run must verify the scheduler resumes at the correct step index (F4b history says this exact bug existed once — check the first post-resume LR log line against expectation).

**§12 (PSR metric evolution) + §13 (head-pose metrics):** ratified; §12's volatility is re-explained by the collapse (§2 item 7 of the recommendations below); §13's "position error collapses to 2.2mm late" is interesting but position stays unpublished regardless (units unverified).

**§14 (config comparison across runs):** accepted; it is the evidence base for the A1-confound ruling (batch/peak-factor differences) — attach it to the A1-redo protocol.

**§15 (NaN inventory):** all ten explained and scheduled — eff_* closed by D3 (measured), psr_tau/pos_blind/f1_calibrated close with Q17/Q18/Q43-machinery on the D3-redo artifact. Post-redo rule: this table must be empty; any survivor is a bug (assertion per §1.7).

**§16 (sampler distortion):** confirmed real (3.6–7.4×); deferred to clean runs / A3 / full-data run per §2-recommendation-2 below; interim covered by 117-Q9 probe. Note §16's "67 detection classes / 10 activity classes" phrasing is part of the §1.13(c) taxonomy confusion — audit.

**§17 (checkpoint evaluation runs) + §18 (timeline) + §19 (Fable6 deep analysis) + §21 (misc logs) + §22 (efficiency):** accepted as record; §22's numbers are superseded by the D3 measured efficiency block; no decisions.

**§23 (conclusions & recommendations):** the seven recommendations, verdicts:
1. Pose normalization — **ACCEPT, T0** (Decision 7), with eval-first protocol.
2. Activity-specific sampling — **ACCEPT-DEFERRED** to full-data run / A3; interim = 117-Q9.
3. PSR sub-head protection — **ACCEPT, escalated** (= Mode B fix, Q36+Q14 probe, week 2).
4. Watchdog keepalive — **ACCEPT, T0** (Decision 2); progress-based, reading the existing heartbeat file.
5. Mixed precision — **DEFER** (118 ruling stands; BF16 smoke-test only if throughput becomes binding).
6. Resume to 30+ epochs — **ACCEPT, strengthened**: the schedule says 100, not 30 (Decision 2).
7. PSR F1 volatility diagnosis — **SUPERSEDED**: the full-val collapse *is* the diagnosis; the subsample volatility was the collapse seen through a 2.6% keyhole.

**§23.4 final-state table:** ratified as the epoch-17/18 snapshot; "eval_metrics 0.0000 not yet functional" is the F22-gating item that D3-redo settles (§2 note on 122 §6.4 metrics).

---

# Section 3: Answers to Document 122 — All Sections

## 3.1 §1 The 70-metric inventory (+ extended 1.2–1.6)

Ratified as the definitive metric dictionary; adopt it as the paper's supplementary metric-definitions source (merge with 125 App H, which duplicates it — keep one). Three rulings: (a) the DET_PROBE metrics (§1.2) are diagnostics, never published — agreed; (b) the Paper-8/Paper-9 metric families (as_*, ev_*) stay in the matrix as AFTER-F22-FIX and get one verification pass in D3-redo — if they populate, they add a free supplementary table; if not, cut them from the matrix rather than carrying dead rows; (c) the MonotonicDecoder detail (§1.6) is the reference for implementing Q48 hysteresis — implement hysteresis *inside* the decoder, not as post-hoc filtering, so the fill-forward constraint sees the hysteresis-gated transitions.

## 3.2 §2 Per-epoch progression tables (+ extended 2.1–2.3)

Ratified; same provenance rule as §1.2. The §2.3 "key transitions" analysis (epochs 5/8/11) is the right skeleton for the paper's training-dynamics figure caption.

## 3.3 §3 Subsample vs full validation (+ extended 3.1–3.3)

The direction of every §3 prediction was right (full set harder; more classes populate) but the *magnitude* was underestimated for activity (predicted "slightly lower," actual −52% relative) and PSR (predicted stable, actual collapse). Ruling: **§3's stability analysis is superseded by the measured D3 deltas** — going forward, no subsample number is quotable without its measured full-set counterpart, and best-model selection on the subsample is acceptable only because it is cheap, with the final claim always re-measured full-set. The §3.4 recommendations are thereby strengthened, not merely accepted.

## 3.4 §4 TTA comparison — superseded by the incident

§4's expected +0.03–0.07 stands as the prior; the measured 0.238 is invalid (§1.4); three-arm redo decides. No further action from §4.

## 3.5 §5 Cross-head signal evidence — ratified with one required softening

§5.2 (activity collapse at LR peak), §5.3 (pose-activity anti-correlation), §5.4 (POS saturation independence) are ratified as written. **§5.1 (PSR F1 tracks detection mAP) must be softened post-collapse:** the 6.4× "elasticity" was computed on subsample F1 values now known to sit on a degenerate decoder. The honest version of the cross-head claim rests on (a) psr_comp_acc=0.567 on full val (per-frame state recognition genuinely works and plausibly tracks detection), (b) the gradient-liveness record (PSR sub-heads went from DEAD to ALIVE as detection matured), and (c) the architecture dependency (s2 features). The F1-trajectory version of the claim should not appear in the paper — a reviewer holding the full-val F1=0 will use it against the whole section.

## 3.6 §6 Honest-disclosure rulings — ratified with four amendments

1. **§6.1 item 2 (up MAE):** 122 quotes 5.82°, 120 quotes 7.06° (epoch 11) and 8.28° (full val). Reconcile before publication; every headline number needs one source-of-truth cell.
2. **§6.2 item 7 (POS):** the "expected 0.85–0.93 blind baseline" clause is obsolete — Q43 measured 0.0; update the disclosure row (it makes the claim stronger); quote subsample POS per §1.6.
3. **§6.3 item 9 (PSR F1):** strengthen from "must not be compared directly" to "report the full-val collapse as a finding" (G6). The D4 escape hatch is blocked until D1-R.
4. **§6.3 item 12 (combined):** ratified — never in the paper; formula must be single-sourced (§1.5).
§6.4 (bug-zeroed metrics): verification lands in D3-redo; see §3.1(b).

## 3.7 §7 Per-paper comparison + §9 per-task breakdown + §10 source index

§7 ratified as the related-work fact base (consistent with 123 Part A). §9's "what is really being measured" prose is ratified and should seed the paper's metric-definition paragraphs — especially the PSR paragraph, which after §1.6 gains the order-vs-timing decomposition. §10 (source index) accepted; regenerate after the epoch=-1 and serializer changes move line numbers.

## 3.8 §8 The 38-row comparability matrix — ratified with these cell updates

Row 1 (det_mAP50 vs P1): "After D1" → "After D1-R, else published-number comparison with weights-unavailability disclosed." Rows 22–23 (PSR F1): "After D4" → "After D1-R + G6; currently a negative finding." Row 27 (psr_tau): compute only after Q18/Q48 calibration — on the collapsed decoder, tau measures the artifact. Row 28 (psr_pos_blind): DONE, value 0.0. Row 36 (eff_fps): DONE, 11.05 measured; publish both TTA/non-TTA FPS if any TTA number appears. Rows 30–33 (as_/ev_): keep AFTER-F22 pending the D3-redo check. Row 7 (det_n_present): BUGGED → **FIXED-verified** (epoch 17). All other rows stand.

---

# Section 4: Answers to Document 123 — All 20 Sections

## 4.1 Part A, §§1–4 (the four paper dives)

Ratified as the related-work fact base — consistent with 114/116/122 §7, no contradictions found. Use Part A verbatim for the "Prior Art on IndustReal" table (the three-table architecture from 118 §4.2 stands). One instruction: every published number cited must carry its table/figure provenance exactly as Part A records it, because the D1 dead-end means several comparisons now rest on published numbers alone — provenance is the substitute for reproduction.

## 4.2 §5 Detection gap closure — rulings on each lever

The section's levers, with verdicts: D1 protocol → rewritten as D1-R (Decision 5; the §5 download URL is the one that 404s — the retrain replaces steps 1–3); Q26 discriminative LR → merged into the pretrain comparison (125-Q1/Q31, §6 Cat 1/7); A1-redo protocol → ratified exactly as written (same init, batch 4, clean dir); Soft-NMS insertion point → ratified, now exists as `soft_nms.py`, tested in the TTA three-arm; TTA protocol → ratified with the flip-safety check added; "report FPS both modes" → ratified (matches §3.8 row 36).

## 4.3 §6 Activity closure — rulings

Per-frame label plumbing (T1) → done per earlier docs; remap protocol (sum probabilities, never average/max; bit-identical sanity check) → **ratified verbatim** — this is the T3 spec (118 §7.18 concurs); act_top1 exposure → DONE; the per-frame MLP description → matches 124 §6. The 0.110→0.205 improvement narrows the closure gap the section was written against; recompute its targets after the full-set number lands.

## 4.4 §7 Ego-pose baseline — ruling

Ratified; the section predates the pose-norm discovery — add the data-quality fix (Decision 7) as step 0 of the pose plan, then the Q41/Q13 run, then Q42 smoothing, then Q15 seeds. Target restated: full-set forward MAE < 8° with error bars, stretch < 7°.

## 4.5 §8 PSR POS — ruling

Ratified and strengthened by the measured blind baseline: the §8 uniform-interval blind protocol is exactly what Q43 ran, and it returned 0.0 — the disclosure table row becomes "POS 0.969 (ours) vs 0.0 (blind canonical) vs 0.812 (STORM)". Subsample-vs-full quoting rule per §1.6.

## 4.6 §9 PSR F1 closure — superseded

The section's "0.144 → 0.60 via YOLOv8m backbone" path is superseded by the collapse: the honest path is now the three-tier ladder (§0.2 Decision 4) with G6 deciding the paper's stopping point, and D4 (after D1-R) becomes the *decoder-isolation* experiment rather than the headline fix. The §9 held-out threshold protocol is ratified and reused for Q18/Q48.

## 4.7 §10 ASD embeddings (R1) — ruling

Unchanged from 118: optional, P2, positive-sum if F1@1 ≥ ~20, quietly droppable otherwise. The §10 protocol (gallery=train, per-frame queries, cosine) is ratified for whenever it runs. Do not let it displace ablations.

## 4.8 §11 IKEA ASM — answered in §1.12 (GO with the two pre-flight checks).

## 4.9 §12 Efficiency validation — rulings

FPS: measured, done (11.05 / 11.04 streaming); the §12 "expected 15–30 FPS" estimates are superseded by measurement — delete estimates wherever a measurement exists. Parameter bookkeeping: §12 contains a third inconsistent count ("~28M current total" with a component breakdown that doesn't match 124's) — **the single-source ruling (§1.14) governs**: 46.5M total / 28.6M backbone / body-pose 1.6M frozen-and-disclosed, pipeline comparison from 120 §3's itemization labeled as estimates, one savings number used everywhere. FLOPs-by-resolution table: keep as supplementary, labeled estimated-until-profiled except the measured 245.3 total.

## 4.10 §13 Combined-metric optimization path — rejected as a paper device, retained as planning

A "combined 0.306→0.50+" storyline must not structure the paper (the combined metric is unpublishable per 122 §6.3-12). Internally it remains a fine progress scalar. The section's per-task decomposition of where gains come from is subsumed by this document's queue.

## 4.11 §14 Honest disclosure strategy — ratified

Consistent with 122 §6 + the four amendments (§3.6) + the two additions (§1.17). The compact single-block format (118 §7.6) stands.

## 4.12 Part C: §15 queue / §16 calendar / §17 risks / §18 venue / §19 fallback / §20 references

- **§15:** superseded by §7 of this document (incidents re-ordered T0; PSR ladder added; D1-R added; E1 removed as done).
- **§16 calendar:** structure retained; shift all 3060 slots right by the D3-redo + D1-R days and insert the main-run restart on the 5060 Ti immediately (it was calendared as busy-until-Jul-16, which is no longer true — it is idle *now*).
- **§17 risks:** merged with 120 §18 — answered jointly in §1.16 (no §17 risk exists that R1–R10 + the two additions don't cover).
- **§18 venue:** ratified — dual-track stands (G5 READY); the factory-pilot scope caution (§1.12) is the one addition.
- **§19 fallback:** ratified with the floor *raised* by epoch 17 (see §4.13). The §19.3 time-cut table stands with one edit: "skip Q1+Q50" is no longer a valid cut — Soft-NMS/TTA are near-free and one of them is already coded.
- **§20 references/evidence chain:** accepted; append 126.

## 4.13 §19.4 contingency metrics table — recomputed floor

The "current" row now reads: det 0.358/pc 0.573, act 0.205 (subsample; 0.057 full — quote full), pose 7.83° (subsample; 9.94 full), PSR F1 0 full-val with rescue pending, POS 0.969 blind-anchored, FPS 11.05 measured. That is already 123's "acceptable conference" tier *before* T1; T0-completions (D3-redo, Q18/Q48, TTA-redo, Kalman) plausibly reach the old "+T0" tier; the strong-accept tier still requires ablations + error bars + one PSR rescue tier + (optionally) D1-R — unchanged, ~12–14 days.

---

# Section 5: Answers to Document 124 — All 22 Sections

## 5.1 §§1–16 (system, backbone, FPN, four heads, body-pose, Kendall, HP_PREC_CAP, combined weights, train loop, eval pipeline, subprocess eval, TTA script, eval_post_reinit)

Ratified as the authoritative architecture reference, with the §1.14 discrepancy list (activity-head params, total params, effective batch) to reconcile against it. Section-specific rulings: **§7 (PSR head):** the 3-layer causal transformer + threshold-0.3 decoder description is the collapse's architecture context — when Q48 hysteresis is implemented, it belongs at §7's decoder stage (per §3.1c). **§10 (HP_PREC_CAP):** unchanged, working, keep (118 Q24 ruling stands; 125 has no cap-removal question). **§11 (combined weights):** the formula here is one of the three variants — this section is the likeliest single source of truth since it cites train.py lines; verify and propagate. **§13–§16 (the four eval entry points):** four entry points is one too many — after the epoch=-1 unification, add the 50-batch parity test (§5.3) across train-val vs subprocess vs TTA paths; eval_post_reinit inherits it for free.

## 5.2 §18 The 5-bug history — pattern ruling

All five are interface mismatches between the subprocess harness and functions written for the training loop — the same root cause as the D3 NaN and the TTA divergence. The fix that prevents the *class* (not just the five instances) is the parity test below. The "5 bugs in 3 minutes" fix velocity is impressive and also the tell: these were shallow bugs guarding a deep structural issue (two divergent eval worlds).

## 5.3 §20 epoch=-1 fix — ratified, plus the structural fix

Correct in shape; §1.7's three additions apply (assertion, NaN-refusing serializer, timeout). **Structural:** add a two-path parity test — run training-val and subprocess eval on the same 50 batches, assert metric equality to tolerance — to the pre-submission checklist. This single test would have caught the D3 NaN, the TTA discrepancy, and the act_top5=0.0 bug before they burned three GPU runs. Note §20's own upstream-impact list shows subprocess_eval.py still passes `epoch=int(overrides.get('epoch', 0))` — **that default should now be -1 too**, or the fix is only half-deployed.

## 5.4 §19 The 10 NaN guards — ratified with the counter requirement

Necessary but double-edged: replacing NaN with 1e-4 silently also hides dying losses — the "silent failure" pattern implemented as policy. Requirement: every guard increments a per-location counter logged at epoch end. Guards firing at a nonzero steady rate is a bug signal that is currently invisible. (The train.py batch-skip guard is fine as-is — it already logs.)

## 5.5 §17 Fix chronicle — ratified

Complete and consistent with 113/118/120 §11. F16 verified in production closes the last open correctness item from the 118 triage except F22/F22b-on-GPU, which D3-redo closes. The F1-consult table's F6 (bf16) and F12 (cosine probe) remain the two never-run items — both scheduled (§2 rec-5 smoke-test optional; F12 probe in T0 #10).

## 5.6 §21 line-reference index + §22 inference code path

Accepted as reference. §22's documented path (including the pseudo-keypoint generation when train_pose=False and the double-FiLM modulation) is the citable description for the paper's architecture section; regenerate §21's line numbers after the current fix wave.

---

# Section 6: Answers to Document 125 — All 50 Questions and All Appendices

Verdict key (as in 118): **T0** = now, inference/config-only · **T1** = before AAIML submission · **T2** = gated/conditional · **SKIP** = not for this paper (journal queue). Recurring theme: many 125 questions propose architecture or paradigm changes that reset the comparability suite — individually promising, collectively incompatible with a submission cycle; triaged accordingly. Global caveat on every training question: **if Decision 1 reveals 2%-data training, run the full-data baseline first — it re-prices every hypothesis in this file.**

## Category 1 — Architecture Changes (Q1–Q5)

**Q1 ConvNeXt-V2-Tiny FCMAE pretrain — T1, the best backbone bet; merge with 117-Q26 into one pretrain experiment.** FCMAE + discriminative LR is strictly the stronger version of 117-Q26 (self-supervised pretrain at zero pretraining cost, same architecture family — the one backbone change cheap enough to survive the comparability-reset objection). 25 epochs, 3060, week 2–3. If it delivers even half the hypothesized +0.06–0.12, it becomes the base config for every fresh run.

**Q2 DyHead — SKIP.** Replaces all four heads at once — maximal comparability reset for a hypothesized +0.02–0.04 combined. Journal queue behind Q1/Q31/Q32.

**Q3 Cross-Scale Transformer neck — SKIP.** Same ruling as 117-Q3 (BiFPN): real expected value, wrong point in the cycle for neck surgery. Journal neck-sweep (CST vs BiFPN vs NAS-FPN).

**Q4 SimOTA decoupled head — T2, the one head change worth holding, gated on trajectory.** Targets a *diagnosed* failure (rare-channel anchor starvation) with the standard modern fix — but detection is the fastest-improving metric right now; don't operate on a recovering patient. Gate: rare channels still ~0 AP at main-run completion → SimOTA is the first architecture intervention.

**Q5 NAS-FPN routing — SKIP.** Highest-cost neck option, speculative per-channel mechanism. Journal queue, last among necks.

## Category 2 — Training Recipe (Q6–Q10)

**Q6 Easy→hard state curriculum — SKIP.** The natural frequency distribution already delivers easy-states-first; untestable without a fresh 100-epoch run; the same target (rare-channel AP) is reachable via Q16 sampling in a 5-epoch probe.

**Q7 SAM — SKIP.** 2× backward on an FP32-slow pipeline; optimizer swap invalidates the tuned schedule; EMA/SWA already approximate the flat-minima benefit free. Journal.

**Q8 SWA (75–100 averaging) — T1-lite, ratified (identical to 117-Q34).** Offline, one hour, after the resumed run completes; publish whichever of EMA/SWA wins.

**Q9 LLRD — T2, fold into Q1.** LLRD's mechanism (protecting early pretrained layers) only exists with pretrained weights; run it as a factor inside the FCMAE arm, not standalone.

**Q10 Two-cycle cosine — T2.** The resumed run is already a de-facto second segment; a full-data run stays OneCycle. Journal for the controlled version.

## Category 3 — Loss Redesign (Q11–Q15)

**Q11 Transition-Aware Focal Loss — T1-lite as a Hamming-cost reweighting (~20 lines, 5-epoch probe), SKIP as the full machinery.** The core insight — 1–2-bit ASD confusions should cost more — is the best loss idea in the set and directly targets the diagnosed confusion clusters. Headline-novelty candidate *if* the probe shows signal (see App-I Claim 7).

**Q12 Sequence contrastive PSR loss — T2, second-line.** Run only if the Q36+Q14 probe under-delivers; its 0.25–0.40 F1 hypothesis assumes a live decoder, which full-val currently lacks.

**Q13 Uncertainty-aware geodesic — T1, folded into the pose run** (with Q41 + no-position). Subsumes 117-Q11; the learned variance doubles as paper material (calibrated confidence for the anchor contribution).

**Q14 Order-regularization for PSR — T1, first-line training-side PSR fix (with 117-Q36).** Directly penalizes the degenerate all-at-frame-0 solution; 5-epoch resumed probe; accept the hypothesized POS dip to 0.94–0.96 (still ≫0.812).

**Q15 Multi-task NT-Xent — SKIP.** A fifth loss on a four-loss balance that took 22 fix-rounds to stabilize, premised on an unmeasured gradient conflict (F12 probe never fired — run it first, T0 #10). Journal, contingent on the measured cosine.

## Category 4 — Data Strategy (Q16–Q20)

**Q16 Transition-biased tubelet sampling — T1, the best data-side idea for both weak spots.** Oversampling state-change neighborhoods feeds rare detection channels *and* PSR transition signal simultaneously; sampler-weight change; 5-epoch probe. Prefer over Q6 and over 117-Q49's blunt GT-fraction knob.

**Q17 FixMatch semi-supervised detection — SKIP this cycle.** Strongest hypothesized detection gain in the file, but FixMatch-for-detection is a project, not a probe. Journal #2 behind MAE/FCMAE. If D1-R lands, plain pseudo-labeling (117-Q38) captures most of the value at a tenth of the effort.

**Q18 Counterfactual inpainting augmentation — SKIP.** High effort, high shortcut-learning risk (detector learns inpainting fingerprints → rare class). Journal with leakage controls.

**Q19 Learning-progress adaptive sampling — T2.** Complementary to Q16 but two adaptive samplers at once is uninterpretable; Q16 first, add Q19 only if rare-channel AP stays flat.

**Q20 Auxiliary optical flow — SKIP.** Offline flow for 188K frames + new head/loss, while activity just improved 86% without it. Journal, as the alternative to Q40.

## Category 5 — Multi-Task Balancing (Q21–Q25)

Global ruling unchanged from 118: **one balancing framework per paper.** Kendall is instrumented, understood, and demonstrably working — activity (the head all five questions target) improved 0.110→0.205 under it while its Kendall precision was rising (121 §8); the "14.8% gradient-starved" premise is weakening in the live data.

**Q21 CAGrad — T2**, the single method worth one 25-epoch comparison post-ablations (best-argued of the five; upgrades B1 into "learned uncertainty vs modern gradient method"). **Q22 GradNorm-adaptive — SKIP** (adds a hyperparameter to a 118-rejected method). **Q23 IMTL-G — SKIP** (no differentiated case over CAGrad). **Q24 DWA — SKIP** (weakest signal; four new temperatures). **Q25 GradVac — T2-diagnostic only:** fire the F12 cosine probe first (free, never run). If det-pose cosine ≥ −0.1, the conflict premise of Q15/Q21–Q25 deflates wholesale and Kendall stands unchallenged.

## Category 6 — Cross-Dataset Transfer (Q26–Q30)

**Q26 IKEA ASM pretrain → finetune — T1 if the IKEA chapter proceeds (it does, §1.12); sequence so the IKEA training run doubles as the pretrain arm.** Makes the cross-dataset chapter load-bearing (transfer benefit) rather than decorative; a three-arm pretrain comparison (FCMAE / Objects365 / IKEA) is genuinely novel for assembly-state detection.

**Q27 Joint multi-dataset training — T2, behind Q26.** Heavier and confounds IndustReal comparability mid-cycle; Q26's sequential transfer answers the headline cheaper. Journal-grade.

**Q28 Domain-adversarial viewpoint invariance — SKIP.** Egocentric continuous viewpoint = no clean discrete domain for the GRL classifier; the proposed camera-index label doesn't exist in this dataset. Rare-channel gains better pursued via Q16.

**Q29 Pose-branch pretrain on IKEA head poses — T2 pending a 30-minute label check.** IKEA ASM is third-person; whether ego-compatible head-pose labels exist at all is unverified. If not, retire.

**Q30 Component-bit metric learning — SKIP for this paper; the clearest next-paper seed.** Universal assembly representations with zero-shot states is a research program, not a pre-submission experiment.

## Category 7 — Detection-Specific (Q31–Q35)

**Q31 Objects365 pretrain — T1-arm inside the pretrain comparison.** Detection-specific pretraining is the strongest a-priori candidate for a detection-bottlenecked system; third arm alongside FCMAE (and IKEA). If only one arm fits: Objects365 for detection impact, FCMAE for all-task balance.

**Q32 DINOv2-S frozen — T2, cheap idle-gap filler.** Head-only training is fast; supports the "frozen foundation + light heads" efficiency variant; ViT architecture keeps it in the ablation table, not the headline row.

**Q33 ConvNeXt-Nano — SKIP this cycle; note the framing value.** "95% at 46% fewer backbone params" would *strengthen* the thesis (contra 117-Q28's ConvNeXt-S which cut against it) — right instinct, wrong deadline. Journal scale-sweep.

**Q34 YOLOv8m distillation — T2, unblocked only by D1-R (G7).** If D1-R runs, distillation-vs-pseudo-labels (117-Q38) becomes one 10-epoch soft-vs-hard-targets probe; either captures the main effect.

**Q35 Multi-scale training — T1, the best teacher-free training-recipe detection lever.** Standard, safe, targets small-component channels. Fold into the full-data run if Decision 1 triggers one, else a 25-epoch probe. Caution: the FPS claim is resolution-pinned — multi-scale *training* is fine; publish inference at fixed native resolution.

## Category 8 — Activity-Specific (Q36–Q40)

**Q36 Hierarchical dilated TCN — T2, gated on G1 exactly as 117-Q7.** If the temporal run happens at all, build it with RF ≥ action length from the start. The stronger per-frame baseline (0.205) raises the bar T2 must clear, making G1 harder to pass.

**Q37 Verb-noun hierarchical head — T1-lite, the best activity idea in the set.** Cheap (two branches + bilinear), exploits the taxonomy's real compositional structure, and 120 App G is literally its evidence base (verb-correct/noun-wrong confusions dominate). Into the activity probe bundle.

**Q38 Detection-logit input augmentation — T1-lite, same probe.** ~24 extra dims, trivial, mechanistically sensible. Bundle: one resumed activity probe carrying {117-Q9 blend, 117-Q35 smoothing, 117-Q47 FeatureBank, 125-Q37, 125-Q38} as sequential arms — a week of 3060 answering five questions.

**Q39 SMOTE feature-space oversampling — SKIP.** Interpolating embeddings of a changing backbone chases a moving target; loss weighting + Q16 sampling attack the same tail cleaner.

**Q40 VideoMAE dual-backbone — SKIP for this paper.** A second pretrained backbone abandons single-backbone parsimony exactly where the paper claims it, doubling inference cost. Cite as future work — it *is* the honest answer to "what would competitive temporal activity take."

## Category 9 — Ego-Pose-Specific (Q41–Q45)

**Q41 6D rotation + geodesic — T1, the pose-run centerpiece.** Established best practice this head skipped; with Q13 and no-position it forms the designed pose experiment. Free diagnostic first: measure how often predicted forward⊥up is violated today — forward+up is nearly a 6D representation already, and the orthogonality-violation rate itself justifies (or not) the Gram-Schmidt change.

**Q42 Kalman smoothing — T0, this week, after the pose-norm fix.** Inference-only, −0.3 to −0.8° expected, EM-fitted noise parameters reportable. Use the causal-filter variant for any real-time claim; disclose the smoother as offline post-processing otherwise.

**Q43 Coarse-to-fine multi-scale pose head — T2.** Third in line behind representation (Q41) and data (pose-norm, Q29-check); pose is already the strongest head — marginal GPU-days go to the weak heads.

**Q44 IKEA pose-trajectory augmentation — RETIRE as posed.** Beyond the Q29 label-existence issue, replacing a frame's GT pose with another dataset's pose breaks the image–label correspondence — as written it trains the model to predict poses uncorrelated with the input image. Only a reformulation (image+pose pairs imported together) would be coherent.

**Q45 MC Dropout — SKIP for the headline; T2 for an uncertainty/OOD appendix.** 20× inference cost kills the FPS claim for a gain Q42 gets free; Q13's learned variance gives calibrated uncertainty at 1×.

## Category 10 — PSR-Specific (Q46–Q50)

**Q46 Transition-detection head — T1, promoted by the collapse (gate G6).** In 118's world this was a journal paradigm swap; post-collapse it is the credible path to competitive F1: the hypothesized trade (F1 0.35–0.55, POS 0.88–0.92, still >0.812) is exactly the paper the PSR section wants. 25–50 epochs, 3060, weeks 2–3, *after* the inference-rescue tier establishes the per-frame paradigm's honest ceiling.

**Q47 Temporal cross-attention PSR — T2, behind Q46.** Keeps the constraint that just failed while adding attention; once Q46 exists, Q47's marginal question is "attention vs TCN context" — a refinement. Journal companion.

**Q48 Hysteresis thresholding — T0, this week, on the D3-redo artifact.** Zero training, attacks Mode A, robust to jitter. Run as one grid with 117-Q18 (per-component {single threshold, hysteresis pair}), tuned on a held-out recording fold. Expected full-val F1 0 → 0.08–0.20; Mode-B components stay dead — report per-component so the decomposition is explicit.

**Q49 Detection-quality-adaptive two-stage PSR — SKIP.** A learned meta-model atop a collapsed base optimizes the wrong layer; revisit after Q46, when its premise may be moot.

**Q50 Multi-decoder ensemble — SKIP for this paper.** Ensembles need working members; one is degenerate and one doesn't exist yet. The natural journal follow-up after Q46 ("fill-forward for order + transition detection for timing, gated") — for later.

## 6.11 The 125 impact-summary table and selection strategy — rulings

The per-category "cumulative potential" figures are **not additive** and must not be treated as a roadmap (stacking +0.15–0.30 mAP from three categories double-counts the same headroom). The selection strategy's three tiers are individually endorsed with edits: the "immediate inference-only" trio (Q42/Q48/SWA) matches this document's T0/T1-lite exactly — ratified; the "highest long-term" list is accepted minus Q50 (needs Q46 first) and with Q32 demoted per above; the "most novel" list is where the caution lands — novelty candidates (Q11, Q15, Q30, Q50) are publishable *only with measured wins*, and three of the four are SKIP-tier this cycle; Q11's cheap probe is the one novelty lottery ticket worth buying. The 14-3060-day budget sketch is superseded by §7's queue (which front-loads the incident repairs the sketch predates).

## 6.12 125 Appendices A–J — dispositions

- **A (cross-reference matrix):** accepted as planning input; its synergy/antagonism calls were used in the bundling decisions above (e.g., Q1×Q9, Q37×Q38, Q46 vs Q47).
- **B (complexity & GPU budget):** accepted with the non-additivity caveat (§6.11).
- **C (sequencing phases 0–3):** structure endorsed; superseded in content by §7 (Phase 0 must now include the incident repairs; Phase 3's "integrated best-config 100-epoch run" is exactly the full-data/resumed run — agreed in principle).
- **D (per-question validation protocols):** ratified as the standard — every probe launched from §7 must carry its App-D success criterion so results are decidable, not vibes.
- **E (negative-result template):** **strongly endorsed** — adopt it project-wide, retroactively filing D1, TTA-v1, and the PSR collapse as its first three entries. Negative results written to this template are paper supplement material.
- **F (long-tail tracking):** accepted as bookkeeping.
- **G (GPU-time budget):** recompute after Decision 1/2 (both GPUs' availability just changed); the constraint logic stands.
- **H (metric definitions):** merge with 122 §1 into one supplementary definitions document (§3.1).
- **I (the 7 claims):** answered claim-by-claim in §6.13.
- **J (math formulations):** accepted as reference for the paper's loss appendix.

## 6.13 125 Appendix I — the 7 SOTA claims, adjudicated

1. **"Detection beats YOLOv8m on mAP50_pc" — REJECT as a target; REFRAME.** 125's own numbers concede beating 0.838 is "very unlikely." Chasing it invites the metric-shopping critique (mAP50_pc vs their mAP50 is not the same metric). The right claim family: "closes the gap to X at 1/6 cost with 3 extra tasks," anchored by D1-R and the pretraining decomposition. The 70–80% "gap < 0.25" scenario is a fine internal target, not a paper claim.
2. **"Activity beats MViTv2 on verb-grouped metric" — REJECT as framed.** Protocol-remapped comparison with a different modality/pretrain stack cannot support a "beats" verb (122 §8 rows 8–12: NEVER for direct task comparison). Keep the honest ladder: per-frame baseline (0.205) → T3-remapped reference → optional temporal. The Q37+Q38 path is worth running for its own sake, not for this claim.
3. **"Ego-pose toward HL2 floor" — ACCEPT, best-supported claim.** The Q41+Q42+Q13 path (50–70% chance of <7.0°) is credible and cheap; with the pose-norm fix it may exceed the estimate. This plus error bars is the anchor contribution's growth story.
4. **"PSR POS beats SOTA with disclosure" — ACCEPT, already banked** (95%+ estimate is right; G4 passed with blind=0.0; quote subsample per §1.6).
5. **"PSR F1 beats SOTA through temporal modeling" — REJECT the "beats SOTA" bar; ACCEPT the direction.** Post-collapse, the realistic pre-submission ceiling is Q48/Q18 rescue (0.08–0.20) plus possibly Q46 (0.35–0.55) — 125's own 15–25% odds of >0.50 and "STORM 0.901 requires additional breakthroughs" concede the point. Paper bar: *credible* F1 with the collapse analyzed, not SOTA F1.
6. **"Single-GPU efficiency thesis" — ACCEPT** (60–80% estimate reasonable); the blockers are exactly the corrected ablations (A1-redo, A2–A4) + the single-sourced params table (§4.9). Primarily narrative + measured FPS, as 125 says.
7. **"Novel method contributions" — PARTIALLY ACCEPT.** Realistic pre-submission novelty inventory: the honest-metric design (mAP50_pc + blind-POS-baseline methodology), the collapse failure-analysis (C8, §1.9), Q37 verb-noun hierarchy if the probe wins, Q11-lite if the probe wins, Q46 if G6 forces it. That is 2–3 plausible method-flavored contributions — consistent with 125's 40–60% estimate, achieved through cheaper items than its list assumed.

**Overall SOTA assessment (125's closing):** the "most efficient path" (inference wins → 3060 ablations of winners → integrated 5060 Ti run) is endorsed as the shape of §7; the "beat SOTA on 4 of 5 metrics" aspiration is **rejected as the paper's success criterion** — the paper wins on first-baseline + efficiency + one banked SOTA beat + honest analysis, per 116's contribution structure, and does not need four SOTA beats.

---

# Section 7: Consolidated Priority Queue (supersedes 118 §8, 123 §15, and 125 App C)

## T0 — This week (Jul 5–8; both GPUs currently idle)

| # | Item | Source | Cost | Why now |
|---|------|--------|------|---------|
| 1 | Verify SUBSET_RATIO / training-data / dataset-stats contradictions (§1.13); launch full-data run if 2% confirmed | 120 App D, §14 | 30 min + (run) | Re-prices everything (Decision 1) |
| 2 | Resume main training on 5060 Ti with watchdog keepalive; verify scheduler resume index | 121 §23, §11 | mins | 82 epochs idle (Decision 2) |
| 3 | D3-redo: epoch=-1 (incl. subprocess default → -1, §5.3), full 13,161 batches, persisted predictions, NaN-refusing serializer, n_present assertion (expect 18), act_top5 fix | 120 §8, 124 §20 | 3–5h, 3060 | Full-set numbers; F22-on-GPU; as_/ev_ check; feeds #5–7 |
| 4 | Pose-norm fix at loader + eval-only recheck of epoch-17 ckpt | 121 §23.2 | hrs | Data integrity for anchor contribution (Decision 7) |
| 5 | Q18-117 + Q48-125 threshold/hysteresis grid on D3 artifact (held-out fold; per-component report) | collapse Mode A | 1 day | PSR F1 0 → 0.08–0.20; feeds G6 |
| 6 | Q17-117 tau distribution (post-calibration only) | 117 | hrs | Honest tau; E2 closure |
| 7 | Q42-125 Kalman smoothing (after #4) | 125 | hrs | −0.3–0.8° free |
| 8 | TTA three-arm redo on unified path (+ flip-safety check) | 120 §7 | 3h | Replaces invalid 0.238 |
| 9 | G7 decision: D1-R YOLOv8m retrain (recommended GO, ~1 GPU-day) | 120 §6 | 1d | Unblocks D4/Q34/pseudo-labels |
| 10 | Free housekeeping batch: F12 cosine probe (once) · EMA/eval-path check (R6) · disk-space check in ckpt hook (R4) · NaN-guard counters (§5.4) · forward⊥up violation rate (§6 Q41) · Q13-117 FiLM histogram · IKEA pose-label check (Q29/Q44 gate) | various | <1 day total | Each is minutes-to-hours; several gate later verdicts |

## T1 — Weeks 2–3 (before submission work freezes)

PSR training probe (117-Q36 + 125-Q14 [+117-Q19 smoothing], 5-epoch resumed) → **125-Q46 transition head if G6 demands it** → pretrain comparison (125-Q1 FCMAE / 125-Q31 Objects365 / [Q26 IKEA arm], with 125-Q9 LLRD as a factor) → A1-redo + A2–A4 (after ckpt-dir verification) → B1 (upgradeable to Q21 CAGrad comparison) → pose run (125-Q41 + 125-Q13 + no-position) → activity probe bundle (117-Q9/Q35/Q47 + 125-Q37/Q38, sequential arms) → 125-Q16 tubelet-sampling probe → 125-Q35 multi-scale (or fold into full-data run) → 125-Q11-lite Hamming-cost probe → T3 remap (G1; protocol per §4.3) → 117-Q15 multi-seed (after #4) → SWA offline (125-Q8/117-Q34) → D4 + distillation-vs-pseudo-label probe (if D1-R ran) → IKEA ASM run (doubling as Q26 pretrain arm).

## T2 — Gated

125-Q4 SimOTA (rare channels still dead at run end) · 125-Q21 CAGrad (post-ablations) · 125-Q32 DINOv2 frozen (idle gaps) · 125-Q36 hierarchical TCN (inside T2-temporal, gate G1) · 125-Q12 contrastive PSR (if Q36/Q14 under-deliver) · 125-Q10 two-cycle · 125-Q19 adaptive sampling (after Q16) · 125-Q29 (after label check) · 125-Q43 coarse-to-fine pose · 125-Q45 MC-dropout appendix · 125-Q25 GradVac (only if F12 cosine < −0.3) · 117-Q49/Q10 GT-fraction probe · BF16 smoke test.

## SKIP for this paper (journal queue, ordered by forward priority)

117-Q48 MAE-pretrain-scratch (superseded by 125-Q1 FCMAE for this cycle; scratch-MAE remains journal #1) → 125-Q17 FixMatch → 125-Q30 component-bit metric learning → 125-Q50 multi-decoder ensemble → 125-Q40 VideoMAE dual-backbone → 125-Q27 joint training → 125-Q2 DyHead → neck sweep (125-Q3/Q5 + 117-Q3 BiFPN) → 125-Q11 full TAFL → 125-Q15 NT-Xent → 125-Q33 Nano scale sweep → 125-Q18 counterfactual aug → 125-Q7 SAM → 125-Q6 curriculum → 125-Q20 optical flow → 125-Q22/Q23/Q24 balancing variants → 125-Q39 SMOTE → 125-Q28 domain-adversarial → 125-Q44 pose-trajectory aug (retired as posed) → 125-Q49 quality-adaptive PSR → 117 skips carried forward (Q37 Unity, Q27 Swin-T, Q28 ConvNeXt-S, Q4 head depth, Q6 75-class, Q22 GradNorm, Q25 init, Q29 EfficientNet, Q33 mixup-main, Q39 active learning).

## The one-sentence summary

Restart the idle training, settle the 2%-data question before optimizing anything, re-run the two broken evaluations on the now-unified code path, rescue PSR in three explicit tiers (calibrate → re-weight → re-architect) with gate G6 deciding how far the paper goes, retrain YOLOv8m yourself to unblock every teacher-dependent experiment, and freeze the paper around the claims that survived — ego-pose, blind-baseline-anchored POS, honest detection, renamed activity, measured efficiency, and two well-analyzed negative findings that are contributions in their own right.

---

*Coverage audit: 120 §§1–18 + App A–M (§1 of this doc), 121 §§1–23 (§2), 122 §§1–10 + extended (§3), 123 §§1–20 (§4), 124 §§1–22 (§5), 125 Q1–Q50 + impact summary + selection strategy + App A–J incl. the 7 claims (§6). Notation: 117-Qn / 125-Qn disambiguates the two question sets; 125 supersedes 117 only where noted. Live-state facts as of the 2026-07-05 snapshot in 120. Where this document contradicts 118, this document governs.*
