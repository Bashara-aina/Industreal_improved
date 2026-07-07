# 140 — Opus Answers v2: Responses to the 139 Overview (Files 132-138 Synthesized)

**Date:** 2026-07-07
**Responds to:** `139_OPUS_OVERVIEW_PROMPT_V2.md` — all 11 prioritized questions (§7), all 6 deliverables (§8), all 3 open debates (§5), all 8 evidence gaps (§4).
**Read in full before answering:** 132, 133, 134 + debate, 135 + debate, 136 + debate, 137 + debate, 138 + debate, SOTA_STATUS.md, psr_null_delta_table.md, null_model_pos.json, pose_kalman_results.json, plus fresh code verification of `model.py`, `psr_transition.py`, `losses.py`, `config.py`, and `scripts/run_psr_kendall_fixed.sh`.
**Companion:** 132 (top-10 depth), 133 (all 66 verdicts). This file supersedes both wherever they conflict — one such conflict is material and is §-1 below.

---

## §-1. STOP-PRESS: The PSR head repair is a no-op. The in-flight training is a Kendall-only ablation.

This audit re-verified the PSR code path from the committed tree (HEAD = `7001107de`) instead of trusting the 132/135 citations, and found a wiring failure that changes several answers below. Chain of evidence, all repo-verifiable:

1. **`PSR_HEAD_REPAIR` is defined but never consumed.** `config.py:105` sets `PSR_HEAD_REPAIR` from the env var. A whole-repo grep finds no other reference to `config.PSR_HEAD_REPAIR` — the only code containing `use_repaired_head` is `PSRTransitionPredictor.__init__` (`psr_transition.py:206-209`), whose default is `False`, and nothing passes the config flag into it.
2. **`PSRTransitionPredictor` is dead code.** No file in `src/` instantiates it. The only imports from `psr_transition.py` anywhere in the pipeline are `MonotonicDecoder` (evaluate.py:417, psr_transition_f1.py:65, eval_yolov8m_psr.py:236, d4_threshold_retune.py:54) and `build_transition_targets` (losses.py:1439). The class that carries the ReLU+bias=−1.0 heads — and their LeakyReLU repair — is never in the forward path.
3. **The PSR head that actually trains is `PSRHead` (`model.py:1539`).** Its structure: fused P3+P4+P5 GAP (768-d) → `Linear(768,512)→LayerNorm→GELU→Dropout→Linear(512,256)→LayerNorm` per-frame MLP → 3-layer causal transformer (d_model=256, GELU, pre-norm) → 11 output heads of `Linear(256,64)→GELU→Dropout→Linear(64,1)` with first-layer bias init +0.1 (`model.py:1609-1611`, an `[AUDIT]` comment that already names GELU zero-collapse as the failure it guards against). **There is no ReLU and no bias=−1.0 anywhere in the trained PSR head.**
4. **Therefore the in-flight run (`scripts/run_psr_kendall_fixed.sh`) applies exactly one live intervention: `KENDALL_FIXED_WEIGHTS=1`** (consumed at `losses.py:1666`; also echoed in the train.py config dump at train.py:3381 — note the dump prints KENDALL_FIXED_WEIGHTS but not PSR_HEAD_REPAIR, which is itself a tell).

**Caveat:** this is the state of the *committed* tree. If the workstation's working tree carries an uncommitted edit wiring the repair into `PSRHead`, the conclusion changes. Verifying that is a 2-minute `git status` + `grep -rn PSR_HEAD_REPAIR src/` on the workstation and is the new Blocking Diagnostic #1.

**Consequences, in order of importance:**

- **(a) Expectation reset.** The "expected F1 0.83+ from head repair" projections (138 §7 Week-1 table, 135 Q50) are void. The live intervention is the one 135 Q48 itself estimated at **+0.01-0.03**. If F1 at epoch ~30 is flat, that is not evidence the repair failed — the repair never ran.
- **(b) The attribution problem dissolves.** 135 Q48 / 135-debate Challenge 5 worried the bundled intervention would be unattributable. It is now a clean single-factor experiment — accidentally the exact `KENDALL_FIXED_WEIGHTS=1` ablation arm that 133 PSR-3 prescribed. Do not kill the run; rename it. It is Pathology 2's empirical leg, running right now.
- **(c) Pathology 1's mechanism text is wrong and must be rewritten.** The gradient-starvation *evidence* is real (zero per-component RMS gradients; TI-3's "GELU fully saturated to zero after linear64" — note that "linear64" matches `PSRHead.output_heads`' `Linear(256,64)→GELU`, not anything in PSRTransitionPredictor). But the *mechanism* published in 132 Q1 ("ReLU gating + bias −1.0, verified at psr_transition.py:216-237") describes a module that never executes. The correct statement: **GELU saturation in `PSRHead.output_heads` under near-zero-variance transformer output** — precisely what the existing +0.1 bias `[AUDIT]` init was an earlier attempt to patch. The 132 verification confirmed the cited code *exists*; nobody confirmed it *runs*. This is the same class of failure as the up-vector index bug, and §5.4's credibility depends on stating it plainly.
- **(d) The real head repair has not been tested.** A genuine repair now means editing `PSRHead.output_heads` (e.g., LeakyReLU in place of GELU, or re-init, or a variance-restoring fix upstream in the transformer) and warm-starting — a *new* run, gated on the activation diagnostic below.
- **(e) 135 Q8's input_dim 512-vs-768 blocker is moot.** It was a property of the dead class. The live head's dimensions are consistent: 3×256=768 fused GAP → `Linear(768, 512)` (`model.py:1569-1571`). One of the three "blocking diagnostics" in 139 §9 closes with zero GPU time.

Two more code verifications this audit performed, both reassuring:

- **Head-pose training loss indices are correct** — `losses.py:951-952` slices `fwd=[0:3], pos=[3:6], up=[6:9]` for both pred and target. The 26.20° era was eval-only; the 7.78° is a genuine measurement of a correctly-trained output. 137-debate Gap 1's worst case is refuted (independently of commit `a7de2c140`, which reached the same conclusion).
- **The Gaussian-smeared transition objective is live** — `losses.py:1436-1454` builds transition targets via `build_transition_targets(sigma=3)` on sequence batches and skips PSR loss on per-frame batches. So 135's loss-design questions (Q6, Q7, Q44; debate Interpretation 3) apply to the real pipeline unchanged.

---

## §0. Headline-Number Table (PW-3 Rubric Applied)

Labels per 133 PW-3: **beats-SOTA** / **competitive** (≤10% relative, identical protocol) / **first-baseline** / **measured-cost** / **not-comparable**. "Committed?" = auditable from this repo today.

| # | Claim | Number | PW-3 label | Committed? | Recommended phrasing |
|---|---|---|---|---|---|
| 1 | Head pose forward MAE | 9.14° (weighted mean, 38,036 fr); 8.94° median of per-recording means; 8.46° excl. outlier | **first-baseline** | pose_kalman_eval ✔; full_eval_ep18_v2 ✘ **commit it** | "First ego-pose forward baseline on IndustReal: 9.14° single-frame MAE (per-recording median 8.94°)" |
| 2 | Head pose up-vector MAE | 7.78° (weighted); 7.58° median of per-recording means; 7.39° excl. outlier | **first-baseline** | pose_kalman_eval ✔ | "First ego-pose up-vector baseline: 7.78° single-frame MAE"; training-loss indices verified correct |
| 3 | Up-vector 5.82° (9-rec median of medians) | 5.82° [IQR 5.55-6.09] | do **not** headline | up_vector_v3 ✘ | Covers the easier 9/16 recordings (137-debate NQ-3). If a median is wanted, use the all-16 median of per-recording means = **7.58°**, computable today from committed pose_kalman JSON |
| 4 | Kalman smoothing | 9.00°/7.58° (−1.5%/−2.7%) | supporting only | ✔ | One sentence + appendix (138 Attack 8) |
| 5 | D1R YOLOv8m | 0.995 / 0.861 (ep 25) | **measured-cost denominator** (cross-architecture — say so) | results.csv workstation ✘ **commit** | "Single-task YOLOv8m ceiling on our split: mAP50 0.995" — never "our detection", never "beats SOTA" (the ~0.95 'SOTA' in SOTA_STATUS is uncited; remove) |
| 6 | D3 multi-task detection | 0.358 (250-batch class-balanced subsample) | **not headline until full-set eval exists** | metrics only, no det fields in d3_full_eval ✔(gap visible) | "0.358 on a class-balanced 2.6% subsample; full-set evaluation in progress" |
| 7 | Present-class mAP 0.573 | derived 0.358×24/15 | **unverified derivation** | ✘ | Blocked on (i) WACV convention check, (ii) 6-vs-9 zero-GT count reconciliation (134 §7 item 6). Do not print until both resolve |
| 8 | PSR per-comp optimal F1 | 0.7499 (10k frames) | **first-baseline** (per-frame PSR); *not* "competitive" (16.8% rel. gap to STORM, different paradigm → also **not-comparable**) | psr_optimal_thr ✔ | "Per-component-calibrated macro-F1 0.7499 (10k val subset, val-selected); global-threshold 0.7217; full-38k figure [run D3-135]" |
| 9 | PSR global 0.10 F1 | 0.7217 (10k) / 0.6773 (38k stream) | honest primary until LOO+38k reconcile | ✔ | Note the 10k-vs-38k gap explicitly — it is the same order as every claimed improvement |
| 10 | PSR null-delta | +0.097 (c4), +0.093 (c10), −0.000 (c9) | **measured** — strongest PSR evidence | ✔ | Lead the PSR table with these columns (138 Q2.08) |
| 11 | LOO-CV | +0.0358 ± 0.0216 | **measured**, caveated | psr_loo_cv ✔ | Report with train/val-membership caveat (135 Q20, unresolved) and per-component breakdown (135 Q18) |
| 12 | D4 re-tuned | 0.000 → 0.347 (hi=0.3, lo=0.1, min=2; 145 combos) | **diagnostic**, not a result (post-hoc sweep, no held-out) | d4_retuned ✘ **commit** | "Decoder transfer requires threshold recalibration; bounded by detector density (<1% frames)" — final wording gated on D4+D1R run (§Q10) |
| 13 | POS | 0.9988 vs null 0.9995/0.9984 | **structural artifact** | ✔ | Footnote/appendix only; the null table is the proof |
| 14 | Activity per-frame / clip | 0.0236 / 0.028 vs baseline 0.2217 | **floor-baseline / null result** | ✔ | Always printed beside the majority baseline; clip 0.028 only in the T3 bridge |
| 15 | Linear probe | 0.2169 vs 0.2217 (95% CI ±0.0046) | **null result** — retract "BACKBONE HAS SIGNAL" | ✔ | "Statistically indistinguishable from the majority-class baseline" |
| 16 | T3 verification | 0.6223 = WACV 0.622 | protocol verification only | ✔ | Methods section, one line |
| 17 | Multi-task cost | 64% (or 42% under COCO convention) | **measured-cost, provisional ×3** | — | Provisional on: convention (Q4.01), full-set D3 eval, same-architecture ceiling (134-debate Ch.1). Until all three land, every cost sentence must carry the cross-architecture caveat |
| 18 | Error-state FPR 0% | structural (0 GT anywhere) | **not a claim** | ✔ | One §5.4 sentence + dataset-section note (138 Q5.07) |

**Repo hygiene flag:** four of the evidence files 139 §1 lists are *not in the repo*: `d4_retuned/{sweep_results,verdict}.json`, `full_eval_ep18_v2/metrics.json`, `up_vector_v3/up_vector_per_recording.json`. 132 §7 already made committing evidence non-negotiable; these four escaped. Commit them Day 1 — two of the paper's headline numbers (9.14°, 0.347) are currently unauditable.

---

## §1. Verdicts on the Five Day-1 Questions (139 §7)

**Q1 — Run D3 full-set detection eval before freeze? YES — it blocks §4 of the paper.**
Evidence: `d3_full_eval/metrics.json` contains act/pose/PSR but no detection fields (134 Q21); the 0.358 headline rests on a 2.6% class-balanced subsample (134-debate Challenge 3). Run the in-process eval (`EVAL_MAX_BATCHES=0`) on the RTX 3060; if it NaNs, run the 10-seed subsample and print 0.358 ± σ. Additionally answer 134-debate New-Q2 while there: determine whether detection metrics were *silently suppressed* (config flag) or *crashed* (NaN swallowed) — the former means detection was never measured on the full set, the latter means a live metric-accumulation bug that also threatens the numbers that did appear. Do not write any detection-cost sentence until this lands. Also do the two 10-minute companions: zero-GT count from `per_class_gt` (resolves 6-vs-9, C-7) and detection rate per frame at conf ∈ {0.01, 0.05, 0.25} (134-debate New-Q3 — distinguishes "fires with wrong labels" from "barely fires").

**Q2 — Re-run per-comp optimal F1 on full 38k? YES — 30 minutes, cached logits, zero conflict with training (135 D3).**
The 0.7499 is a 10k-frame val-selected upper bound (135 Q49; 135-debate Challenge 4). Three numbers must appear in one table row: global-0.10, per-comp-optimal-38k, LOO-CV mean ± σ (138 Q2.06). If the 38k figure lands at 0.70-0.72, that is the honest primary and the STORM gap widens — report it anyway; committing in advance to publish whichever way it moves is the paper's strongest integrity signal (138 Q2.09). Reconcile in the same pass why the 38k stream at global 0.10 gave 0.6773 while the 10k gave 0.7217.

**Q3 — Fix and re-run the temporal probe? YES — but fix the gate first, then let the probe decide.**
The fixed script is committed (`7001107de`); run it overnight on the 3060. But the ACT-1 gate ("probe > 0.10 → TCN+ViT justified") is invalidated: 0.2169 is *below* the 0.2217 baseline and inside its 95% CI (136 ACT-LP-1; 136-debate Challenges 1+3 — temporal integration of features carrying no frame-level signal cannot conjure signal). New gate, decided before results arrive: **temporal-pooled probe ≥ baseline + 0.05 (≥ 0.27) → TCN+ViT (2-3 days); 0.22-0.27 → only if ≥2 weeks remain; ≤ baseline → cut TCN+ViT, write activity as a diagnosed null result.** Run the two 30-minute co-diagnostics with it: per-class probe accuracy excluding the majority class (136 Gap 2 / 138 Q3.01 — this is what "backbone has signal" should have meant) and the per-recording majority baseline (136-debate NQ-1 — if recordings are action-homogeneous, even 0.2217 flatters the model).

**Q4 — Is the cross-architecture cost denominator defensible? NO as "multi-task cost." Two-part fix.**
(i) *Text now:* every cost sentence gets the caveat "relative to a single-task YOLOv8m ceiling (cross-architecture)", and the paper drops "multi-task cost" as a general claim in favor of "detection degradation under multi-task training" (138 Q4.02; 138-debate Challenge 2). Also verify whether the .tex's equal-gradient-update ablation is same-backbone single-task or not (138 Q4.09) — if it is, it is the architecture-controlled measurement and must be reported as such. (ii) *Experiment:* schedule the **single-task ConvNeXt-Tiny detection run (2-3 GPU-days)** as Week 2's one new training run — it is the only way the denominator becomes clean, and it outranks distillation (distillation moves the numerator; this fixes the denominator). Resolve the mAP convention (30 min) *before* it starts so the run is evaluated under the convention the paper will use.

**Q5 — Is the activity "probe head" framing salvageable? Half of it.**
Keep: the *backbone-feature-quality probe* (linear probe = measured null result; per-class + temporal probes complete the diagnosis). Retract: (a) the "BACKBONE HAS SIGNAL" verdict in SOTA_STATUS.md — replace with "statistically indistinguishable from the majority baseline (0.2169 vs 0.2217, CI ±0.0046)"; (b) the *multi-task interference probe* claim — no interference was measured, and with ACTIVITY_GRAD_BLEND_RATIO starting at 0.05 the head may never have effectively trained (136-debate Challenge 2; 138 Q3.08: "if the activity head never trains, there's no interference to study"). The 10× probe-vs-MLP gap may be interference, a broken schedule, or hyperparameters — claiming interference without the single-task MLP control is storytelling. Either run the single-task MLP control (1 day, gate it on wanting the claim) or delete the interference language. Activity's surviving claims: first per-frame baseline (with documented literature search) + diagnosed null result + class-imbalance analysis. Also purge the verb-antonym line as a *primary* justification — 1.3% of errors cannot carry "per-frame is inherently limited" (138 Q3.06); it survives as one supporting figure.

---

## §2. Answers to Day-2 and Day-3 Questions (139 §7 items 6-11)

**Q6 — Drop "near SOTA"/"~15°" head-pose claims? YES, everywhere, immediately.**
The source is unverifiable (HP-1, 133 §4); an uncitable number is a nonexistent number. SOTA_STATUS.md *still* carries "~15°"/"near SOTA" in four rows and "BEATS SOTA" for detection — edit it in the same pass (Day-1 text task). Claim "first ego-pose orientation baseline on the IndustReal protocol," scoped exactly that narrowly, backed by the documented literature search (137 Q41-44: check Jiang ECCV'22, HoloAssist NeurIPS'23, Tome CVPR'23 under comparable protocols before "first" is typed). Report against HoloLens GT with the noise-floor acknowledged, not deconvolved (137-debate Challenge 4: check pose.csv for a tracking-confidence field; if present, report high-confidence-subset MAE as secondary).

**Q7 — Drop the activity section? NO.**
Keep as a probe/null-result subsection titled "Per-Frame Action Classification Probe" (138 Decision 4) with the corrected framing from Q5. Removing it mid-audit reads as hiding a bad result, and the diagnosis (class-imbalance collapse + no frame-level linear signal + temporal-probe outcome) is genuinely informative. Cut MViTv2-S (P5.1) permanently — it cannot share the backbone, so it breaks the one-model story at 5+ GPU-days cost (138 Q3.03).

**Q8 — Run the 4 blocking diagnostics? Status: two are already closed by code reading; run the other two.**
(a) *Training-loss pose indices* — **CLOSED, correct** (`losses.py:951-952`). (b) *PSR input_dim 512-vs-768* — **CLOSED, moot**: the mismatched class is dead code; the live `PSRHead` dims are consistent (§-1e). Replaced by a new blocker: **confirm on the workstation that PSR_HEAD_REPAIR is a no-op in the running process** (§-1 caveat). (c) *D3 full-set eval* — RUN (Q1). (d) *Per-class linear probe* — RUN (Q3). Add one more from §-1: the **1-hour PSRHead activation diagnostic** (forward one sequence batch; print `encoded.std()` and per-head post-GELU stats — the `_debug_log_head0` machinery at model.py:1635 already exists for exactly this). It decides whether the *real* repair targets the output heads or the transformer's collapsed variance.

**Q9 — Detection distillation (P2.1)? DEFER to Week 2 back-half, strictly timeboxed at 3 days, and only after the single-task ConvNeXt baseline is running.**
Rationale ordering: the cost story needs a clean denominator (single-task ConvNeXt) more than a better numerator (distilled multi-task head). If both cannot fit before freeze, distillation is future work — 133 D-6's "do not gate any paper claim on it" stands.

**Q10 — D4 with D1R weights? YES — promote to Week 1.**
This is eval-only on an existing checkpoint (~0.5 day, RTX 3060), and 134-debate Challenge 4 is right that it is *the* decisive test, not an "incremental confirmation" (134 Q38 got this wrong): with a 0.0004→0.995 mAP gap between the two backbones, D4+D1R settles whether decoder or detection density binds. Expected outcomes: F1 ≥ 0.6 → "decoder transfers given adequate detection density" (strong §5.4 sentence); F1 < 0.4 → the decoder is *also* a bottleneck and the current D4 disclosure text is wrong. Add the per-video breakdown (135 D4) and the frame-sequential-order fraction (135-debate DQ-3) to the same run.

**Q11 — TCN+ViT despite the bad gate? NO — gated exactly as in Q3.**
The corrected gate must clear before 2-3 GPU-days are spent. If the temporal probe lands ≤ baseline, the honest deliverable is the backbone-bottleneck finding, not a 0.10 accuracy (132 §4 already set this fallback).

---

## §3. Rulings on the Three Open Debates (139 §5)

**Debate 1 (134 — cross-architecture cost): UPHELD.** The fix is Q4's two parts (caveat now, ConvNeXt single-task run in Week 2). Until the run lands, the paper's central empirical claim is "detection degradation vs a YOLOv8m ceiling," not "multi-task cost."

**Debate 2 (135 — PSR F1 validity): PARTIALLY RESOLVED, one part by this audit.** input_dim → moot (dead code, §-1e). 10k-vs-38k → run Day 1 (Q2). Attribution → dissolved: the in-flight run is single-factor Kendall-only (§-1b); the head repair, once actually wired, becomes the second single-factor run. What remains open is only whether the 0.7499 survives the 38k eval.

**Debate 3 (136 — probe signal statistically zero): UPHELD in full.** "BACKBONE HAS SIGNAL" is a false positive of a mis-set gate; retract it, adopt the corrected gate (Q3), and let the temporal + per-class probes decide activity's fate. The TCN+ViT gating claim in SOTA_STATUS/139 §2 ("clears") is wrong and must not reach the paper.

---

## §4. §5.4 Disclosure Language — Eight Numbered Disclosures (numbers attached)

Drop-in text, updating 132 §5 with current figures. Bracketed items are the pending experiments that finalize each sentence.

1. **Backbone-swap transfer (D4).** Feeding YOLOv8m detections into our MonotonicDecoder yields transition F1 = 0.000 at thresholds tuned for ConvNeXt statistics and 0.347 after a 145-combination re-tune (hi=0.3, lo=0.1, sustain=2); the detector fires on <1% of frames, bounding any decoder. [Finalize after D4+D1R: "with a dense fine-tuned detector, F1 = X — decoder transfer {is|is not} detection-density-bound."]
2. **POS is structurally inflated** under monotonic fill-forward decoding: an all-zeros predictor scores POS = 0.9995 and copy-previous-frame 0.9984 vs our 0.9988 (3 recordings, 5,000 frames). POS appears only in the appendix; per-frame F1 and transition F1 are the PSR metrics. [Optional: POS@±3 tolerance as the salvageable variant.]
3. **Per-frame action classification is a floor baseline**: top-1 = 0.0236 (28,665 labeled frames), 16-frame majority vote 0.028, vs a majority-class prior of 0.2217; a linear probe on frozen backbone features reaches 0.2169, within the prior's 95% CI (±0.0046); 37 of 66 evaluated classes have zero accuracy. The backbone shows no statistically detectable frame-level action signal. [Temporal probe result: X.]
4. **Multi-task detection** reaches mAP50 = 0.358 on a 250-batch class-balanced subsample — 36% of a single-task YOLOv8m ceiling (0.995) trained on the identical split. The ceiling is cross-architecture; [same-backbone single-task ConvNeXt-Tiny reaches Y, giving a same-architecture cost of Z]. Under COCO convention (15/24 classes with GT) the present-class figure is [0.573, pending convention verification]. [Full-set eval: X.]
5. **PSR per-component gradient starvation**: the per-component output heads (Linear(256,64)→GELU→Linear(64,1)) showed zero RMS gradient over extended training spans, consistent with GELU saturation under low-variance transformer output; reported F1 therefore partly reflects prevalence calibration. Per-component null-deltas over an always-positive prior: +0.097 (comp 4, p=0.14), +0.093 (comp 10, p=0.18), −0.000 (comp 9) — genuine learned signal on the lowest-prevalence components, none on comp 9. *Our earlier internal attribution of this failure to a ReLU/bias=−1.0 head described a module not in the execution path; we disclose the correction.*
6. **PSR thresholds are validation-selected**: per-component-optimal macro-F1 = 0.7499 on a 10k-frame subset vs 0.7217 at a global 0.10 threshold; leave-one-recording-out CV bounds the selection benefit at +0.0358 ± 0.0216 across 16 recordings. [Full-38k per-comp figure: X; LOO caveat: recordings span the model's train/val membership.]
7. **A 3.5-month evaluation-index bug** read position channels [3:6] as the up-vector, reporting 26.20°; the corrected slice [6:9] yields 7.78°, cross-checked by three independent scripts. The training loss always used the correct indices (verified), so only reporting — not learning — was affected. One legacy diagnostic script remains unfixed and is marked deprecated.
8. **Position is unreported**: the head predicts 9-DoF but position units are unverified against the HoloLens export; we evaluate orientation only (6 of 9 DoF) and make no position claims.

Integrity notes that live in §4/§6, *not* §5.4 (138 Q5.05): Pathology 2 is presented as theoretical analysis whose empirical leg is the now-running Kendall-only ablation ("two proven pathologies + one theoretical" until it lands — 138-debate Challenge 3); the NaN-checkpoint selection failure and freeze protocol; the CUDA-crash disclosure paragraph with crash frequency (138-debate NQ-5).

---

## §5. Updated Master Plan — Next Two Weeks

Amendments to 138 §7; unchanged items not repeated. Freeze date: **Jul 20** (138 Decision 6, confirmed).

**Day 1 (Jul 7) — blocking diagnostics and hygiene (RTX 3060 + CPU; training untouched):**
1. Workstation check: is PSR_HEAD_REPAIR consumed by the running process? (2 min; §-1 caveat). Rename the run "Kendall-fixed ablation" in the tracker either way.
2. Commit the four missing evidence dirs (d4_retuned, full_eval_ep18_v2, up_vector_v3, D1R results.csv). Fix SOTA_STATUS.md language ("BEATS SOTA", "near SOTA ~15°", "BACKBONE HAS SIGNAL").
3. WACV mAP convention check (30 min) + zero-GT count from per_class_gt (10 min).
4. Full-38k per-comp PSR F1 on cached logits (30 min) + start P2.6 transition F1 (1 day, cached).
5. Per-class + per-recording linear-probe breakdown (30 min) + launch temporal probe overnight.
6. PSRHead activation diagnostic (1 hr): `encoded.std()`, post-GELU stats via `_debug_log_head0`.
7. FiLM γ/β one-pass stats (1 hr); GT variance fwd-vs-up (10 min); D3 detection rate probe (10 min).

**Day 2-3:** D3 in-process full eval with detection enabled (+10-seed fallback); **D4 with D1R weights** + per-video breakdown; null-POS extended to all 16 recordings + null-Edit (135 Q29); head_pose_diag.py fix-or-deprecate; per-recording forward-MAE table (already computable — median 8.94°, this file §0).

**Day 4-7:** Monitor the Kendall-only run — track the baseline-crossing epoch (135-debate DQ-5); **abort criterion: val PSR F1 (global 0.10) < 0.65 on two consecutive evals**; verify epoch-18 `best.pth` (sha256 `59cb88ec…`) is cold-copied before anything overwrites run dirs. Design + launch the *real* head repair on `PSRHead` (guided by the Day-1 activation diagnostic) as a separate single-factor run. Start writing: §5.1 pathology (rewritten mechanism per §-1c), §5.2.1 null tables + paradigm table (138 Q2.10), §5.4 eight disclosures, §1 intro (discovery hook).

**Week 2:** Single-task ConvNeXt-Tiny detection baseline (2-3 days — the denominator fix, this fortnight's only new mandatory training run). LOO-CV re-run on whichever PSR checkpoint wins. Distillation only if a GPU is idle after the above (3-day timebox). GFLOPs/params re-measure (C-6); YOLOv8m FPS on the RTX 3060 (138 Gap 4); .tex reconciliation (C-3/C-4/C-5); **results freeze Jul 20**: hash the reporting checkpoint, re-run every eval once, emit `results_frozen.json` that the .tex tables consume.

**Venue thresholds (138-debate §A5's missing table):** PSR ≥ 0.78 (38k, defensible thresholding) *and* full-set detection number *and* clean denominator → AAIML main track. PSR 0.72-0.78 with the pathology set intact → AAIML short/MLSys-workshop framing ("What Four Tasks Cost One Backbone", 138 Decision 10). Full-set detection unreportable → arXiv first, venue after the eval bug is fixed. No NeurIPS/CVPR under any branch (138 Decision 1).

---

## §6. New Measurements Needed (consolidated, deduplicated)

| Measurement | Cost | Gates |
|---|---|---|
| Workstation PSR_HEAD_REPAIR no-op confirmation | 2 min | Everything PSR |
| D3 full-set detection eval (+ suppression-vs-crash root cause) | 1 day | §4 of the paper |
| WACV convention + zero-GT count | 40 min | Every cost sentence |
| Full-38k per-comp PSR F1 | 30 min | PSR headline |
| P2.6 transition F1 (epoch 18) | 1 day | PSR narrative |
| D4 + D1R weights (+ per-video) | 0.5-1 day | Disclosure 1 wording |
| Temporal probe + per-class probe + per-recording baseline | overnight + 1 hr | TCN+ViT go/no-go |
| PSRHead activation diagnostic | 1 hr | Real head-repair design |
| Single-task ConvNeXt-Tiny detection | 2-3 GPU-days | Cost denominator |
| Single-task activity MLP (only if interference claim kept) | 1 day | Interference language |
| FiLM γ/β stats | 1 hr | FiLM novelty claim |
| GFLOPs/params re-measure; YOLOv8m FPS on 3060 | 2 hr | Efficiency section |
| Pose frame-level error histogram + fwd/up error correlation | 30 min | §5.3 robustness text |
| Null-POS ×16 recordings + null-Edit | 2 hr | Disclosure 2 completeness |

Explicitly **cut**: MViTv2-S head (P5.1), DDP, CUDA-crash bisection, shuffled-frame control (vacuous per EP-5), 60-90 hr factorial repair ablation (superseded — both factors now run as single-factor arms by construction).

---

## §7. Fail-Safe Plan — If PSR Does Not Improve

Reframed by §-1: the in-flight run tests *Kendall only*; the head repair is untested. Branches:

1. **Kendall-only lifts F1 ≥ +0.03:** Pathology 2 gains its empirical leg; report as "bounded/fixed weighting recovers X"; still run the real head repair afterwards as the second single-factor arm.
2. **Kendall-only is flat (±0.01, the expected case):** publishable negative — "task-weighting was not the binding constraint" — which *strengthens* the head-starvation story. Proceed to the real PSRHead repair run.
3. **Real head repair flat or worse, or clock runs out:** fall back to epoch 18 (`best.pth`, sha256 `59cb88ec…`, cold-copied): PSR primary = global-threshold F1 (0.7217, or the full-38k per-comp figure if it survives), plus the null-delta table, transition F1, and the paradigm-comparison table. Claim set: "first per-frame PSR baseline + prevalence-decomposed calibration analysis + measured gap decomposition (head starvation / prevalence prior / paradigm)." Never "near SOTA."
4. **Any run drives val F1 (global 0.10) below 0.65 twice consecutively:** stop, restore, diagnose — do not train through it.
5. **Paper-level worst case** (PSR flat, temporal probe ≤ baseline, full-set detection eval unfixable): the fallback paper of 138 Decision 10 — "What Four Tasks Cost One Backbone" — with the contribution statement amended per 138-debate: *two proven pathologies (dead/starved PSR heads with corrected mechanism; NaN checkpoint selection), one theoretical analysis (bounded Kendall) with its ablation reported, one measured detection degradation (denominator caveated or fixed), three first baselines, eight numbered disclosures.* Target: MLSys workshop or arXiv-first. The wiring failure disclosed in §-1 belongs in that paper too — it is the third instance (after the NaN checkpoint and the index bug) of the same systemic finding: *code that exists but does not execute is invisible to loss curves*, and per-path runtime verification is the missing monitoring layer. That sentence is the pathology paper's thesis, stated with three in-house exhibits.

---

## §8. One-Paragraph Bottom Line

The single most important fact discovered while answering 139: **the PSR head repair was never wired in — `PSRTransitionPredictor` (with its ReLU/bias=−1.0 heads and their LeakyReLU repair) is dead code, the trained head is `PSRHead` in model.py (GELU heads, +0.1 bias), and the in-flight run is therefore a clean KENDALL_FIXED_WEIGHTS-only ablation with an expected lift of +0.01-0.03, not +0.05-0.10.** Reset expectations accordingly, verify the no-op on the workstation, and design the real repair from the 1-hour PSRHead activation diagnostic. Everything else from 132-138 stands with sharpened priorities: run the cheap deciders first (full-38k PSR F1, D3 full-set eval, temporal + per-class probes, D4+D1R, convention check — under 4 GPU-days combined), schedule the single-task ConvNeXt detection baseline as the one new mandatory training run so the cost claim gets a clean denominator, retract "BACKBONE HAS SIGNAL" and every "near SOTA"/"BEATS SOTA" cell in SOTA_STATUS.md, commit the four missing evidence directories, and hold the Jul 20 freeze. The two headline claims that survive everything are unchanged — first ego-pose baseline (9.14° forward / 7.78° up, training-loss-verified) and first per-frame PSR baseline with null-delta-proven signal — and the paper that carries them is the measurement-and-pathology paper, now with a third exhibit for its thesis.

---

**End of 140. Next: execute §5 Day-1; file 141 should carry the Day-1 diagnostic results (workstation no-op check, 38k PSR F1, D3 full eval, temporal probe).**
