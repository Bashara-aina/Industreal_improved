# 138 — SOTA Integration: 50 Deep Questions + Cross-Head Beat Plan

**Date:** 2026-07-06
**Status:** Parallel agents creating 134 (detection), 135 (PSR), 136 (head pose), 137 (activity) are in flight; their per-head deep dives will populate the per-head cells here on completion. This file integrates across all four heads plus the methodological contributions (multi-task cost, three training pathologies, eight disclosures).
**Prerequisite reading:** 130 (master plan), 132 (Opus top-10 answers + verification audit), 133 (all 66 answers + 30 debate rulings), SOTA_STATUS.md.

---

## §0. Head-by-Head Current Status (Consolidated)

This table merges SOTA_STATUS.md (epoch_18, best.pth) with the corrections and rulings from files 132-133. Where a number is disputed across files, the most recent/adjudicated value is used with a footnote.

| Head | Metric | Our Value | SOTA Target | Gap | Status (132/133 verdict) |
|---|---|---|---|---|---|
| **Detection D1R** | mAP50 (YOLOv8m, 25ep) | **0.995** | 0.838 (WACV) | +0.157 | **CEILING MEASUREMENT** — single-task, not our model. Do not claim as "our detection." Report as the cost denominator. |
| **Detection D1** | mAP50 (official IndustReal weights) | **0.0004** | 0.838 (WACV) | -0.838 | **SILENT COCO FALLBACK SUSPECTED (C-2)** — eval may have hit COCO weights. Resolution: fail-hard fix applied; rerun with verified weights. Current number uninterpretable. |
| **Detection D3 (multi-task ConvNeXt)** | mAP50 | **0.358** | 0.838 | -0.480 | **NaN FULL EVAL** — subsample only. May be 0.573 under COCO convention (excl. zero-GT classes: 15/24 present). Must resolve convention before paper. |
| **PSR (per-frame)** | macro F1 (per-comp optimal) | **0.7499** | 0.901 (STORM) | -0.151 | **NEAR SOTA** — head repair (in flight) expected to lift to 0.83-0.87. Transition F1 (P2.6) not yet computed. |
| **PSR (global thresh 0.10)** | macro F1 | **0.7217** | 0.901 | -0.179 | **HONEST PRIMARY** — per-comp thresholds are val-selected; LOO-CV mandatory before calibrated number is headline. |
| **PSR null-delta** | low-prev comps | **+0.097/+0.093** | n/a | n/a | **GENUINE LEARNED SIGNAL** — proves head learned something despite dead gradients. |
| **PSR (D4: YOLOv8m→decoder)** | transition F1 | **0.000 (default) / 0.347 (retuned)** | 0.883 (B3) | -0.883 | **INPUT STARVATION** — YOLOv8m fires on <1% of frames; decoder works when it has signal. Threshold retune lifts to 0.347. |
| **Activity (per-frame)** | top1 accuracy | **0.0236** | 0.2217 (majority base) | -0.198 | **FLOOR BASELINE** — statistically indistinguishable from majority prior. Per-frame MLP cannot do temporal reasoning. |
| **Activity (clip-level, 16-fr maj)** | top1 | **0.028** | 0.622 (MViTv2-S) | -0.594 | **ARCHITECTURAL GAP** — but the comparison is misleading (different paradigms). Paper's paradigm-difference claim requires the baselines table. |
| **Activity linear probe** | top1 (frozen ConvNeXt) | **0.2169** | 0.2217 (majority) | -0.005 | **BACKBONE HAS SIGNAL** — probe >> 0.05 threshold; backbone encodes weak but detectable action signal. Temporal modeling required to extract it. |
| **Activity T3 baseline** | top1_69 | **0.6223** | 0.622 | +0.0003 | **PROTOCOL VERIFICATION** — matches SOTA baseline; not a competitive claim. |
| **Head Pose forward** | angular MAE (single-frame) | **9.14°** | ~15° (UNCITABLE) | n/a | **FIRST BASELINE** — the ~15° SOTA reference cannot be sourced (HP-1). Claim "first ego-pose baseline on IndustReal." No comparison to cite. |
| **Head Pose up** | angular MAE (single-frame) | **7.78°** | ~15° (UNCITABLE) | n/a | **FIRST BASELINE** — up-vector index [6:9] bug fixed (was 26.20° with wrong indices). 7.78° correct. |
| **Head Pose (Kalman smoothed)** | forward / up | **9.00° / 7.58°** | n/a | n/a | **MINIMAL IMPROVEMENT** — +0.14° forward, +0.21° up. ConvNeXt features are already temporally smooth. |
| **Multi-task cost** | % of ceiling | **36% (64% cost)** | — | — | **PRIMARY CONTRIBUTION** — 0.358/0.995. May be 58% under COCO convention (42% cost). Resolution pending D-4. |

**Note on files 134-137:** These per-head deep-dive files are being created in parallel by specialist agents. When committed, their per-head results tables should be considered authoritative for each individual head. This file integrates across heads and adds the cross-cutting analysis they cannot provide alone.

---

## §1. Where We BEAT SOTA — and How to Keep It (10 Questions)

The two "beats SOTA" claims (detection 0.995, head pose 9.14°/7.78°) both dissolve under scrutiny: 0.995 is a ceiling measurement from a different model, and ~15° is uncitable. The honest framing is "first baselines" and "measured cost." These 10 questions probe what survives, what doesn't, and what must be done to make the defensible claims review-proof.

### Q1.01 — Can detection 0.995 be in the paper at all, and if so, how?

**Status:** File 132 §2 Q5 and 133 C-1 establish that 0.995 belongs to a separately trained single-task YOLOv8m (repo-verified: `d1r/results.csv` epoch 25). It is a ceiling measurement, not a model claim.

**Deep question:** Can 0.995 appear in the paper, and under what conditions? The spectrum runs from "remove entirely" to "full subsection on the single-task ceiling." The 132/133 resolution is the cost-ratio framing: report 0.995 as the denominator, never as a headline. But is even that admissible? A reviewer could argue: "You trained a separate model for this number — it has no place in a paper about multi-task learning." The counterargument: the ceiling establishes what is possible on this data, and the cost ratio is the contribution. Which argument wins?

**Required action before freeze:** Verify the D1R training was on the identical dataset split. If the evaluation protocols differ (class-balanced subsample vs full eval, recording-aware split vs random), the ratio is comparing apples to oranges. Document the split match in supplementary.

### Q1.02 — What is the correct SOTA comparison for head pose when the cited ~15° cannot be sourced?

**Status:** HP-1 in 133 §4 establishes that the "~15° SOTA" originates from unverifiable search snippets; the Ohkawa paper was not found on CVF despite repeated attempts. The only defensible position is "first ego-pose baseline on IndustReal."

**Deep question:** Is "first baseline" a claim that survives an ML systems review? It depends on the thoroughness of the literature search. What is the search strategy — which databases (IEEE Xplore, CVF, PubMed for HoloLens), which query terms, how many results reviewed? The paper must document this in §5.4 as a numbered disclosure, or a reviewer will find a citable ego-pose paper (e.g., a HoloLens IMU-based head-pose paper from a robotics venue) and the "first" claim collapses.

**Required action:** Systematic literature search (file 132 §6 targets this as debate 4.3). If any citable ego-pose work exists, the claim becomes "competitive with [X]" or remains a first-baseline on this specific dataset. The difference determines whether §5.3 exists or is cut.

### Q1.03 — If "first baseline" is the claim for all four heads, does the paper lack novelty?

**Status:** PW-3 in 133 §9 establishes a claim-strength rubric: "beats SOTA," "competitive," "first baseline," "measured cost," "not comparable." Under this rubric, nothing qualifies as "beats SOTA" or "competitive" after the corrections.

**Deep question:** A reviewer assigned a systems/AIML track reads: "First baseline for head pose. First baseline for per-frame PSR. First baseline for per-frame action classification. Measured multi-task cost." Their reaction may be: "Four baselines and a cost measurement is a technical report, not a research paper." How do we preempt this? The answer from 132 is: the three training pathologies (§5) and the eight honest disclosures (§5.4) together constitute a methodology contribution — "here is what happens when you try to train four heads on one backbone, with evidence." But is that enough for a first-tier venue? The paper needs to argue this explicitly in §1.

**Required action:** Write the "what is this paper about" paragraph that preempts the "just baselines" objection. Draft candidate: "This paper reports what it costs to do four tasks on one backbone — measured not as a single number but as three reproducible training pathologies that any practitioner will hit, eight honest disclosures that bound every result, and four first baselines that establish the evaluation protocol for this domain."

### Q1.04 — Does the FiLM contribution to head pose (currently unmeasured) change the pose claim if it's negative?

**Status:** HP-5/A-2 establish that FiLM's contribution to head pose has never been ablated. The pseudo-keypoints that condition FiLM are detached (no gradient flows back to detection), and γ/β values are unknown. The FiLM ablation (P3.2, 2 days) is deferrable.

**Deep question:** If FiLM turns out to be a pass-through (γ≈1, β≈0), does the 9.14° become less interesting (it's just a linear head on ConvNeXt features) or more interesting (linear head on ConvNeXt features achieves 9.14° — which argues the backbone does all the work)? The asymmetry matters: a 3-layer MLP achieving 9.14° with a pass-through conditioning is a clean, simple, deployable result. But the paper currently claims FiLM as a novelty (04_BEST_PAPER_FORMULA scores it 90/100). If γ≈1, the novelty claim must be dropped.

**Required action:** Compute γ/β stats from one forward pass (1 hour). This is the cheapest decisive experiment in the entire plan and should precede any writing about FiLM.

### Q1.05 — Should the $0.005 M (6 DoF orientation only) claim replace the "9-DoF head pose" claim?

**Status:** HP-3 in 133 §4 establishes that position units are unverified (config.py:853 "DO NOT REPORT mm/cm"). Only 6 of 9 predicted DoF are publishable.

**Deep question:** Does "6-DoF orientation baseline" weaken or strengthen the paper? It weakens the scope (predicting 9, reporting 6) but strengthens the honesty signal. Debate 9.2-3 resolves to "orientation baseline, not 9-DoF baseline." But the .tex architecture description still says "9-DoF head pose head." The question: should the paper describe the architecture as 9-DoF (accurate: it outputs 9 values) or as "orientation-only" (accurate: it evaluates only 6)? An architecture described as 9-DoF with evaluation on only 6 DoF is a mismatch a reviewer will flag. The fix is one sentence in §5.4: "The head predicts 9-DoF (position+orientation), but position units are unconfirmed; we report only orientation (6-DoF) here."

**Required action:** One sentence added to §5.4 before the freeze. No experiment needed.

### Q1.06 — What is the correct forward vs up-vector asymmetry story, given both beat uncitable ~15°?

**Status:** Forward 9.14°, up 7.78°. Up is better than forward, which is unexpected — typically up-vector is harder (head tilt is more variable than head turn).

**Deep question:** Why is up-vector MAE (7.78°) lower than forward (9.14°)? Three hypotheses: (a) the up-vector is partially determined by gravity, which the egocentric camera sees via the scene's vertical lines, making it easier to regress; (b) the up-vector distribution is narrower (less variance in the dataset); (c) the index bug was masking that up was always easier. If hypothesis (a) is correct, the paper can make a claim: "Egocentric head pose estimation benefits from visual gravity cues for the up-vector component." If (b), simply report the distribution and move on. If (c), it's a trivial artifact. The experiment: compute per-component variance of forward vs up GT. Cost: 10 minutes.

**Required action:** Compute GT variance for forward vs up across the dataset. If forward has >2x variance, the asymmetry is explained.

### Q1.07 — If everything above loses the "beats SOTA" claims, does the abstract need to start differently?

**Status:** The abstract in the .tex draft currently leads with "multi-task ConvNeXt-Tiny, four heads, competitive across detection, pose, PSR, activity." Files 132-133 systematically dismantle each "competitive" label.

**Deep question:** What is the one-sentence abstract that survives the review? Let me try: "We present a multi-task egocentric assembly monitoring model combining detection, orientation, state, and action classification on a shared ConvNeXt-Tiny backbone, measuring a 64% multi-task cost on detection and documenting three reproducible training pathologies — dead PSR heads from ReLU saturation, bounded Kendall uncertainty weighting under extreme label sparsity, and NaN-inflated checkpoint selection — that account for the gap to single-task performance." This is specific, honest, and methodology-focused. Is it compelling? A reviewer would read "training pathologies" and either lean in ("finally, an honest paper") or lean out ("sounds like they couldn't make it work"). The answer determines the framing of every subsequent section.

**Required action:** Draft the abstract two ways (SOTA-light vs methodology-focused). Opus decides.

### Q1.08 — Can the detection cost ratio survive without the single-task baselines for the other three heads?

**Status:** D-5 in 133 §1 establishes that the efficiency claim requires single-task same-backbone baselines (Ablation A from supplementary Q8). Currently only detection has a ceiling measurement.

**Deep question:** The paper says "multi-task costs 64% of detection ceiling." A reviewer asks: "What about pose? PSR? Activity? Maybe those cost 0% — maybe they're free riders on the shared backbone, and your 'cost' is really a detection-specific penalty." Without the other three ceilings, the cost story is incomplete. The counterargument: training single-task baselines for all four heads costs 12+ GPU-days (3 heads x 3-5 days each). The practical question: is the cost contribution credible with one measured ceiling and three assumed from literature? Or does it require the full grid? The Opus audit (133 A-1) recommends reporting what exists and stating the gap honestly.

**Required action:** Either (a) allocate GPU days for single-task pose and PSR baselines, or (b) add a disclosure: "Single-task baselines for pose, PSR, and activity remain future work; the detection cost is the one fully measured ceiling."

### Q1.09 — What is the correct framing for "our model has 46.5M params vs 4-model pipeline 66.4M" when the params per head are imbalanced?

**Status:** D-5/133 §1 establishes the efficiency thesis survives only at system level. But the per-parameter comparison is misleading: PSR's 3.1M params are demonstrably underutilized (dead head), and activity's share is minimal (single MLP vs MViTv2-S's 38M).

**Deep question:** A reviewer will look at the architecture table and ask: "You're comparing 46.5M (your model) to 66.4M (four separate models). But activity alone should be 38M (MViTv2-S). Your 46.5M includes pose+PSR+detection which total 8.5M. So your activity head is 38M *smaller* than the SOTA activity model. The comparison is not 'four heads on one backbone beats four separate models' — it's 'three small heads piggyback on a ConvNeXt-Tiny for detection, and activity is a toy.'" How does the paper answer this?

**Required action:** Restructure the efficiency argument as: (a) detection backbone is shared across all heads — this is the primary savings; (b) per-task head capacities are documented but the paper does not claim each head is SOTA-competitive; (c) the deployment advantage (single forward pass, 11 FPS on RTX 3060) is the practical contribution.

### Q1.10 — What is the expected variance of the headline numbers, and are any differences statistically significant?

**Status:** AC-6/133 §10 establishes that all numbers will change after the freeze. C-5 documents head pose numbers spanning three eras. No head has a variance estimate.

**Deep question:** If we rerun the full eval 10 times with different seeds, what is the standard deviation of:
- Detection mAP50: ±0.01? ±0.10?
- Pose forward MAE: ±0.5°? ±2.0°?
- PSR macro F1: ±0.02? ±0.08?
- Activity top1: ±0.005? ±0.02?

Without variance bounds, every cross-model comparison is provisional. A one-sigma overlap between our PSR 0.75 and STORM 0.90 changes the story from "16% gap" to "gap may be 8-24%." The 10-seed subsample from debate 1.2 is the minimum. But doing it for all four heads costs 4 GPU-days. The prioritization question: at what point is reporting a point estimate without error bars a confidence-overclaim versus a acceptable practice for a systems paper?

**Required action:** At minimum, compute variance for the two headline claims (pose MAE, PSR F1) using bootstrapped subsampling of the 38k-frame eval set. Report as mean ± 1σ throughout the paper.

---

## §2. Where We Are NEAR SOTA — Closing the Gap (10 Questions)

PSR macro F1 0.7499 vs STORM 0.901. Gap 16%. This is the head with the most actionable path to improvement, but the diagnosis has changed between file 130 and files 132-133: not Kendall suppression, but architectural death of the per-component heads.

### Q2.01 — What is the expected F1 after PSR head repair, and is the head-repair approach validated by any evidence?

**Status:** PSR-3/132 §2 Q1 establishes that the per-component heads are architecturally dead at initialization: `Linear → ReLU(inplace=True) → Linear(bias=-1.0)`. If ReLU inputs are ≤0, the head gates all gradient. Bias=-1.0 parks sigmoid at 0.27 where focal-loss gradient is small. The fix: re-init output bias to 0.0, replace ReLU with LeakyReLU/GELU.

**Deep question:** What is the evidence that this fixes the 16% gap? The argument is architectural plausibility: dead forward path → no learning → head learns nothing. But the null-delta analysis (§0: +0.097/+0.093 on low-prevalence comps) proves the head DID learn something despite the dead path — the learning came through the shared trunk (backbone features leaking through), not the per-component heads themselves. So the head repair may improve F1 by enabling the per-component heads to actually specialize. But by how much? If the backbone features already carry the discriminative signal (null-delta proves this), the per-component heads may only add marginal value (2-5%). If the backbone features need per-component tuning, the lift could be larger (10-15%).

**Required action:** The 1-hour diagnostic (forward one batch, print pre-ReLU activations) is the necessary precondition. Without it, we're guessing. Run it before any retraining.

### Q2.02 — Should transition F1 be computed before or after the head repair?

**Status:** P2.6 (transition F1 on same predictions as 0.7499) is promoted to Week 1 in 132 §4. It costs 1 day on cached logits.

**Deep question:** The decision tree:
- If pre-repair transition F1 is already ~0.7: the per-frame head is doing well on transitions despite being dead. Head repair may push it to STORM parity. Narrative: "even with a dead head, our model captures transitions."
- If pre-repair transition F1 is ~0.3-0.5: the dead head kills transition detection specifically. Head repair is the critical intervention. Narrative: "per-component gradient starvation primarily affects transition events."
- If pre-repair transition F1 is <0.2: the head is barely doing transitions at all. The 0.7499 is prevalence-calibration noise. Narrative becomes salvage (is the paradigm defensible?).

The question: which narrative are we hoping for, and does running P2.6 before head repair commit us to a weaker story? Counterpoint: running it after head repair contaminates the "same predictions" comparison. The honest chain is: compute on epoch_18 (pre-repair) → publish or not based on the number → compute on head-repair checkpoint → compare. The first number (0.75 per-frame F1, X transition F1) is what we have; the pre-repair transition F1 should be reported as the baseline regardless of magnitude.

**Required action:** Run P2.6 now (1 day, cached logits on RTX 3060). Do not wait for head repair.

### Q2.03 — If PSR reaches 0.83 after head repair, does the paper claim "competitive with STORM" or "within striking distance"?

**Status:** 132 §4's amended success metrics condition PSR ≥0.83 on head repair, not fixed weights alone. STORM is 0.901.

**Deep question:** A 0.83 vs 0.901 gap is 8% relative. Three interpretations:
1. "Competitive" — within 10% relative under different evaluation protocols. This requires the paradigm comparison (per-frame F1 vs transition F1) to be resolved and explained. If STORM's 0.901 is on a different metric (transition events with procedural knowledge), our 0.83 on per-frame states may actually be better on the metric that matters for deployment.
2. "Approaching SOTA" — acknowledges the gap but claims trajectory. Only defensible if the head repair was a controlled intervention with a clear mechanism.
3. "Comparable under paradigm difference" — the most defensible but also the most jargon-heavy. Requires the paper to explain two paradigms, show both numbers, and let the reader decide.

The question is strategic: do we want a headline claim ("competitive with STORM") or a measured contribution ("our analysis reveals three factors accounting for the gap: dead head, prevalence prior, paradigm difference")?

**Required action:** Opus decides the framing before §5.2 is written.

### Q2.04 — Is STORM 0.901 the right comparison, or is B3 0.883 the more appropriate SOTA?

**Status:** SOTA_STATUS.md and 128-sota Debate 2 establish that STORM (0.901) adds procedural knowledge (expected transition masks). B3 (0.883) uses semantic features without procedural knowledge.

**Deep question:** If our contribution is "no procedural knowledge, no hand-crafted features, just a ConvNeXt backbone + per-frame MLP," then B3 0.883 is the fair comparison (same inputs: semantic features only). STORM adds procedural knowledge, which is the next axis of comparison. The gap to B3: 0.7499 vs 0.883 = 0.133 (15% relative). After head repair (0.83) the gap would be 0.053 (6%). That's within a standard deviation — potentially "comparable" (not "competitive," but within noise). This changes the narrative from "behind STORM" to "matching B3 after a controlled fix."

**Required action:** Update all paper comparisons to B3 as the primary baseline, STORM as the "with procedural knowledge" upper bound.

### Q2.05 — Does the POS-null-model table (§5.2.1) help or hurt the PSR narrative?

**Status:** 132 §2 Q3 and 133 PSR-1 establish that POS must leave the headline. D4 (POS=0.999 with F1=0) is the accidental null-model. An explicit null-model experiment (all-zeros + copy-prev, ~30 min) is proposed for §5.2.1.

**Deep question:** The explicit null table proves the paper is honest. But it also proves that POS is worthless — and the .tex currently leads with POS=0.9693 in the abstract (C-4). Removing it from the abstract and replacing it with per-frame F1 is a text-level improvement. But is the explicit null table necessary? The argument against: it dedicates a subsection to proving a metric is bad, which the paper then doesn't use. A reviewer could say "you spent half a page telling me a metric is meaningless — why did you report it at all?" The argument for: it preempts the reviewer who says "why is your POS only 0.969 when the other method achieves 0.999?" The cost is low (30 min), and the honesty signal is high.

**Required action:** Run the null-model POS experiment (§5.2.1 table). Keep it to ≤1 paragraph. The reviewer question it preempts is worse than the text it costs.

### Q2.06 — What is the correct per-component threshold selection protocol: global, per-comp optimal, or LOO-CV?

**Status:** PSR-5 in 133 §2 establishes a clear hierarchy: 0.7217 (global threshold 0.10) is the honest primary. 0.7499 (per-comp optimal on val) requires LOO-CV for verification. LOO-CV is P2.5, 2 days.

**Deep question:** The threshold selection problem is not just statistical — it's narrative. The paper can present:
- Best case: 0.7810 (5k subset, per-comp optimal) → looks great but overfits
- Val case: 0.7499 (full eval, per-comp optimal) → honest but val-selected
- Conservative: 0.7217 (global threshold) → robust but lower
- With LOO-CV: 0.74 ± 0.03 → the most defensible

Which one is the headline? The Opus resolution (132 §2 Q9) says 0.7217 until LOO-CV. But after LOO-CV, if the mean is 0.74 ± 0.02, is that the headline? The risk is that LOO-CV degrades the number (if thresholds overfitted to val), but it could also improve it (if the val was a particularly hard fold). The paper needs to commit to reporting LOO-CV as primary regardless of outcome — otherwise it looks like cherry-picking.

**Required action:** Report all three (global, per-comp, LOO-CV) in a single table row. Let the reader see the gap between them. This is the strongest honesty signal the PSR section can send.

### Q2.07 — Is the D4 result (F1=0 at default, 0.347 after retuning) a disclosure or a salvageable pipeline?

**Status:** 132 §2 Q2 resolves that the 0.347 after retuning proves the decoder is not structurally redundant. The binding constraint is YOLOv8m's detection density (<1% of frames).

**Deep question:** Should we invest in making D4 work (e.g., lowering YOLOv8m detection confidence threshold further, or fine-tuning YOLOv8m to fire more densely)? Or should we report the retuned 0.347 and move on? The cost of making D4 work properly is 2-3 days (fine-tuning YOLOv8m to fire at >10% of frames with confidence). The potential return is a demonstration of backbone-agnostic PSR decoding, which would be a strong methodology claim. But it's a stretch goal that preempts higher-impact work (head repair, transition F1, linear probe). The heuristic: invest in D4 only if head repair + transition F1 leave F1 < 0.80. At 0.83+, D4 becomes a curiosity, not a priority.

**Required action:** Gate D4 investment on P2.6 (transition F1) and P1.1 (head repair) results. If PSR F1 is ≥0.80, allocate D4 effort to paper writing instead.

### Q2.08 — Should we compute and report the null-delta (achieved F1 minus prevalence-prior F1) as the honest metric?

**Status:** 133 PSR-6 recommends adding two columns to the per-component table: always-positive F1 = 2p/(1+p) per prevalence p, and achieved-minus-null delta. Comp 4: null 0.249 vs achieved 0.346 (delta +0.097). Comp 10: null 0.310 vs 0.402 (delta +0.092).

**Deep question:** This is the strongest honesty signal in the PSR section. It distinguishes between "the model learned the prior" (prevalence fitting) and "the model learned signal beyond the prior." For components with prevalence >0.5, the null prior F1 is already >0.67 — so our achieved F1 of 0.75-1.0 on those components may be mostly prior. For low-prevalence components (comp 4 at 14%, comp 10 at 18%), the null delta proves genuine learning. The paper should lead with the null-delta table and let the per-component breakdown show the spectrum from "mostly prior" to "learned signal." This would inoculate against the adversarial reviewer (AC-3/133 §10) who reads the 0.75 aggregate as prevalence-calibration noise.

**Required action:** Add null-delta columns to the per-component table (1 hour, analysis only). Lead PSR results with the aggregate null-delta (weighted by component count) rather than raw F1.

### Q2.09 — Should we report N-1 LOO-CV for per-component thresholds even if it shrinks the F1 gap to B3/STORM?

**Status:** PSR-5 clarifies that LOO-CV is mandatory before per-comp-optimal thresholds (0.7499) can be primary. If LOO-CV shrinks to 0.73 ± 0.01, the calibrated threshold advantage (0.7499 - 0.7217 = 0.028) largely disappears.

**Deep question:** The commitment to report LOO-CV regardless of outcome is the paper's strongest integrity signal. If the number degrades, the paper reports it as a finding ("per-component threshold calibration does not generalize across recordings on this dataset"). That's a publishable negative result — it tells practitioners "don't bother with per-comp thresholds on small-N recordings." If the number holds, it validates the approach. Either outcome is informative, which is the mark of a well-designed experiment. The risk is narrative: if LOO-CV shrinks the gap, the PSR story weakens. But the alternative (reporting only val-optimal) is reviewer-bait. The correct answer: run LOO-CV, report the outcome, and let the narrative adjust.

**Required action:** Run LOO-CV (P2.5, 2 days on RTX 3060) regardless of which direction the number moves.

### Q2.10 — What is the correct per-frame F1 vs transition F1 vs STORM comparison table?

**Status:** The paper needs one table showing: (1) our per-frame F1, (2) our transition F1 (P2.6), (3) B3 per-frame or transition, (4) STORM per-frame or transition, (5) the paradigm mapping. Currently no such table exists.

**Deep question:** The paradigm comparison table is the single most important figure in the PSR section. A reviewer needs to see, in one place, that our 0.75 per-frame F1 on state detection is a different quantity from STORM's 0.901 transition-event F1 with procedural knowledge. The table must explicitly label which metric each number uses. Recommended columns: Method | Paradigm (state/transition) | Procedural knowledge? | Metric | F1. Our row: ConvNeXt-Tiny + MLP | Per-frame component state | No | Macro F1 | 0.75 (or post-repair X). B3 row: Semantic features + CRF | Per-frame state | No | Macro F1 | 0.883. STORM row: Transformer + procedural mask | Transition event | Yes | Event F1 | 0.901. This table exposes the comparison as apples-to-oranges and makes the paper's honesty transparent rather than defensive.

**Required action:** Create this table when P2.6 completes. It determines the PSR narrative for the entire paper.

---

## §3. Where We Have an Architectural Gap — TCN+ViT Plan (10 Questions)

Activity per-frame top1 = 0.0236, clip-level 16-frame = 0.028. MViTv2-S achieves 0.622. The gap (0.594) is the largest of any head. The linear probe (0.2169) shows the backbone has signal, but frame-level features are not linearly separable. Temporal aggregation is required.

### Q3.01 — Does the linear probe result (0.2169) justify TCN+ViT spend, or is it too close to majority baseline (0.2217)?

**Status:** The linear probe cleared the 0.05 threshold (file 129/133 ACT-1) but is within 0.005 of the majority-class baseline 0.2217. File 132 Q4 interprets this as "backbone has weak but statistically significant signal."

**Deep question:** A 0.2169 linear probe vs 0.2217 majority baseline is not distinguishably different. The 0.2169 could be noise: the majority class (take_short_brace) accounts for ~22% of frames, so any model that predicts that class every time achieves ~0.22. The linear probe predicted other classes but at near-random accuracy. The comparison that matters: what is the linear probe's top-1 on non-majority classes? If it's ~0.01, the backbone has NO action-discriminative signal for rare classes — only for the majority class. If it's ~0.05-0.10 on minority classes, there's real signal worth extracting with temporal modeling. This question decides whether TCN+ViT gets 2-3 GPU-days or 0.

**Required action:** Compute per-class accuracy of the linear probe, excluding the majority class (take_short_brace). Report mean accuracy on non-majority classes. Gate TCN+ViT on that number exceeding 0.05.

### Q3.02 — What is the expected TCN+ViT lift from 0.028, given the linear probe reveals weak frame-level signal?

**Status:** No TCN+ViT result exists yet. The config has the architecture (TCN + 2-layer ViT with T=16 frames) but it's never been trained.

**Deep question:** TCN+ViT aggregates frame-level features over a 16-frame window. If frame-level features have 0.2169 separable signal (linear probe), temporal aggregation could amplify this by smoothing classification noise across frames. Expected mechanisms: (a) temporal smoothing reduces frame-level noise (majority vote over 16 frames → 0.028 from 0.0236, confirming this); (b) TCN captures motion features (temporal differences) that are discriminative for actions; (c) ViT attends to key frames (transition boundaries) where the action is most discriminative. Mechanism (a) alone gives 0.028. Mechanism (b) could give 0.10-0.20. Mechanism (c) could approach 0.30+. The question: is mechanism (b) or (c) realizable with frozen ConvNeXt features, or do they require trainable backbone features that capture motion?

**Required action:** If TCN+ViT is gated on (expected lift ≥0.15), we need a motion feature analysis first: compute optical flow or frame-difference magnitude at action boundaries and compare to within-action frames. If motion cues are weak in this dataset (assembly actions are subtle — a hand turning a screw vs holding a screw), temporal aggregation alone may not suffice.

### Q3.03 — Is MViTv2-S (P5.1, 5+ GPU-days) a realistic target for this paper, or should it be cut?

**Status:** P5.1 is explicitly listed as a stretch goal in 130. 133 §3 ACT-2 recommends cutting it if the linear probe shows the backbone is the bottleneck — which it does (0.2169 ≈ majority baseline).

**Deep question:** MViTv2-S is a video backbone (38M params, pretrained on Kinetics). It would replace ConvNeXt-Tiny for activity — but then it's no longer a shared backbone. The multi-task architecture would need two backbones: ConvNeXt-Tiny for detection/pose/PSR and MViTv2-S for activity. At that point, the paper is no longer "one model, four heads" — it's "two models, one for three tasks, one for the fourth." The multi-task cost story breaks. Given that MViTv2-S cannot share the backbone, and training it costs 5+ GPU-days, the honest recommendation: cut P5.1 entirely. Activity becomes a disclosure ("per-frame MLP cannot capture temporal information; a video backbone is required"), not a competitive head.

**Required action:** Remove P5.1 from the plan. Reallocate its compute budget to PSR head repair (P1.1) and detection distillation (P2.1).

### Q3.04 — Should the paper report activity at all, or should it be removed from the multi-task claim?

**Status:** 133 §3 establishes the paper is "four tasks + a per-frame probe head." Activity is effectively a probe for studying multi-task interference.

**Deep question:** Removing activity from the contribution list has pros and cons:
- Pro: No "broken head" in the architecture. Three strong heads (detection, pose, PSR) + one probe head is a cleaner story.
- Pro: Saves 2-3 pages of results, baselines, and disclosure text.
- Con: The multi-task cost story loses one dimension. With activity, we have 4 tasks and detect 64% cost on 1. Without it, we have 3 tasks and 64% cost on 1 — the cost argument is thinner.
- Con: The "probe head as interference measurement" argument (AC-2/133 §10) requires activity to exist. Without it, there's no interference dimension.

The resolution in 133 §10 AC-2: keep activity but frame it as a probe, not a competitive head. The probe claim requires the linear probe result. Since the probe cleared at 0.2169, the probe claim survives. The framing: "Activity required temporal modeling beyond a per-frame MLP, revealing an architectural constraint on shared-backbone multi-task learning." This turns a weakness into a methodological finding.

**Required action:** Keep activity in the paper. Frame as probe + disclosure. Remove any competitive language.

### Q3.05 — What would a review say about a paper with three functioning heads and one clearly activity-broken head?

**Status:** No direct evidence, but 133 §3 (ACT-7) and §10 (AC-2) address reviewer expectations.

**Deep question:** A systems reviewer (the target venue for "measurement and pathology" framing) would likely accept a broken activity head IF:
1. The paper explicitly says why it's broken (architectural ceiling, not training failure).
2. The paper provides evidence (linear probe, confusion matrix, baselines table).
3. The paper frames it as a finding about shared-backbone limitations.
4. The other three heads are strong enough that activity is clearly the outlier.

Points 1-3 are satisfied by the planned analysis. Point 4 is the risk: if detection is "ceiling measurement, not our model" and PSR is "dead head, partially fixed" and pose is "first baseline, no comparison," then ALL four heads are weak. A reviewer seeing four weak heads will reject regardless of framing. The paper needs at least ONE unconditionally strong claim to anchor the review. After the corrections, that claim is... the multi-task cost measurement? The training pathologies? The eight disclosures? Those are methodology contributions, not results. If the paper leads with methodology, it lives or dies on methodology strength. If it leads with results, it needs one result that's unambiguously good.

**Required action:** Identify the paper's single strongest claim (proposed: the cost measurement + pathology analysis constitutes a novel methodology contribution). Test this framing on a colleague. If it doesn't work, consider whether the paper has a publishable core.

### Q3.06 — Does the verb-antonym confusion analysis prove per-frame activity is inherently limited, or does it prove the particular MLP is insufficient?

**Status:** 133 ACT-4 recommends computing the confusion matrix on cached predictions. SOTA_STATUS.md §5.4 reports 1.3% of errors are verb-antonym same-object confusions — "temporally ambiguous by construction."

**Deep question:** 1.3% is tiny. If 98.7% of errors are NOT verb-antonym confusions, then the per-frame MLP is wrong on 98.7% of its errors for OTHER reasons — not temporal ambiguity. This actually WEAKENS the temporal-ambiguity argument: only 1.3% of errors are unavoidable. The remaining 98.7% could be reduced with a better per-frame model. The confusion matrix analysis doesn't support the "per-frame is inherently limited" narrative; it supports the "this particular MLP is undertrained/underparameterized" narrative. The paper should NOT cite verb-antonym confusions as the primary justification for temporal modeling. Instead, the justification is: the linear probe shows weak per-frame signal, and temporal aggregation is the standard approach to amplify it.

**Required action:** Remove or reframe the verb-antonym argument. Replace with linear probe + baselines table as the primary justification for temporal modeling.

### Q3.07 — Should clip-level (16-frame majority vote) be reported at all, given the permutation-invariance critique?

**Status:** 133 EP-5 establishes that 16-frame majority vote is permutation-invariant (it measures per-frame accuracy smoothed by mode, not temporal understanding). The shuffled-frame control proposed in 127 is vacuous — it cannot fail.

**Deep question:** Reporting clip-level 0.028 alongside per-frame 0.0236 is at best uninformative (both are near majority baseline) and at worst misleading (it implies temporal understanding). The clip-level metric should be reported ONLY as the bridge to T3's protocol-verification (T3 achieves 0.6223 on clip-level, confirming the T3 baseline matches the published number). In all other contexts, use per-frame metrics. The paper should lead with per-frame top-1 and per-frame macro-F1, never clip-level.

**Required action:** Purge clip-level 0.028 from the abstract, introduction, and results. Keep only in the T3 protocol-verification paragraph.

### Q3.08 — Is there an activity-related claim that survives peer review unchanged?

**Status:** After the corrections in files 132-133, surviving activity claims:
- "First per-frame action classification baseline on IndustReal" (ACT-7, conditional on literature search)
- "Per-frame MLP achieves 0.0236, statistically indistinguishable from majority baseline 0.02217" (factual)
- "Linear probe on frozen ConvNeXt achieves 0.2169, confirming weak but detectable signal" (factual)
- "Temporal modeling is required for competitive performance" (inference from evidence)
- "Activity head reveals a constraint on shared-backbone multi-task learning" (methodological finding)

**Deep question:** Which of these is the paper's activity claim? If it's "first baseline," the literature search must be documented. If it's "methodological finding," the paper needs cross-task interference evidence (does adding activity hurt detection or pose?). The probe-head interference measurement (AC-2) requires showing that activity head training affected other heads — but with the head essentially untrained (dead channels), there's no interference to measure. This is the hole 133 AC-2 identifies: the probe claim only works if the linear probe partially succeeds AND the activity head trains at all. If the activity head never trains, there's no interference to study.

**Required action:** Verify whether the activity head's shared backbone gradients affected detection/pose/PSR. If yes, the interference claim survives. If no (activity head was essentially frozen by gradient starvation or dead channels), the interference story collapses.

### Q3.09 — What is the minimal temporal ablation that provides an informative baseline for activity?

**Status:** The choices are (a) TCN+ViT (P1.4, 2-3 days), (b) temporal averaging of features over T=16 before MLP (1 day), (c) simple frame-difference features (0.5 day), (d) no ablation (current).

**Deep question:** The paper needs at least one temporal ablation to support the claim that temporal modeling is required. Without it, the statement is "we think temporal modeling would help" — which is speculation, not evidence. The minimal informative experiment: temporal averaging of per-frame features over a sliding window of T=16 (option b). This convolves frame-level features with a uniform kernel before the MLP head, essentially asking: "does smoothing the frame-level feature trajectory help?" If it does (0.028 → 0.05+), the paper has evidence that temporal context helps even without learned temporal modeling. If it doesn't, the claim "temporal modeling is required" is weaker but still defensible — the linear probe shows the backbone features are not linearly separable, so a learned temporal model (TCN+ViT) may still extract signal that averaging cannot.

**Required action:** Run temporal averaging ablation (1 day on RTX 3060). Results decide whether TCN+ViT (2-3 days) is worth the additional compute.

### Q3.10 — What activity metric should the paper report: top-1, macro-F1, or a custom metric?

**Status:** The .tex reports macro-F1 (0.205/0.129) and top-1 (class indices differ). The eval stack reports top-1. 133 §3 ACT-5 recommends per-frame macro-F1 as primary, top-1 as secondary.

**Deep question:** Top-1 accuracy on 69 classes with 22% majority class gives 0.0236 expected chance. Macro-F1 (unweighted mean per class) is even more punishing for rare classes. The paper's activity numbers will always look bad on either metric. A 69-way per-frame classifier with <1% of frames belonging to rare classes has no path to good macro-F1. The correct metric for this setting is: (a) top-1 on the majority class (to show the model isn't just guessing), or (b) top-1 excluding the majority class (to show minority-class separability), or (c) per-class F1 on the top-K most frequent classes. The paper should report multiple metrics and explain why each is relevant, not hide behind a single flattering or damning number.

**Required action:** Report per-class F1 for the top-10 most frequent classes plus aggregate macro-F1. The majority-class-only accuracy and minority-class-only accuracy. This gives a complete picture.

---

## §4. Multi-Task Cost — The Contribution (10 Questions)

Detection: 0.358 vs 0.995 = 36% of ceiling = 64% cost. This is the paper's primary empirical contribution — a clean measurement of what multi-task learning costs on this architecture and dataset.

### Q4.01 — Should we report COCO-convention mAP (0.573) or ASD-convention mAP (0.358)?

**Status:** D-4 in 133 §1 establishes that COCO convention excludes zero-GT classes from the mean. With 15/24 classes present, 0.358 = 0.573 × 15/24. The protocol-correct number may be 0.573, not 0.358.

**Deep question:** This is the single highest-impact-per-effort experiment in the plan (30 min to check WACV's convention). If WACV followed COCO convention, our comparable number is 0.573 (58% of ceiling, 42% cost), shrinking the paper's biggest weakness by nearly half. The 64% cost becomes 42% cost. This changes everything: the gap to SOTA shrinks from daunting (-0.48) to achievable (-0.27). The multi-task cost claim shifts from "severe degradation" to "moderate degradation." The question: do we WANT the 64% or 42% number for the narrative? A 64% cost is a stronger "here is a problem we identified" story. A 42% cost is a weaker problem but a better result. If the contribution is the measurement, the correct number is the one that matches WACV's convention, regardless of which is larger.

**Required action:** Check WACV's eval code or paper for zero-GT class handling. Report the convention-matched number as primary and present-class-mAP as secondary. Let the reader see both.

### Q4.02 — Does the cost measurement require a multi-model pipeline baseline (4 single-task models)?

**Status:** D-5 in 133 §1 and supplementary Q8 establish that the equal-gradient-update ablation partially covers this but is not identical. A full baseline requires training ConvNeXt-Tiny separately for each task.

**Deep question:** Without single-task baselines for pose, PSR, and activity, the cost story is detection-only. A reviewer asks: "You measured detection cost at 64%. What about pose? PSR? Activity? Maybe those cost only 10% each. Or 200% (actually benefited from multi-task). Without the grid, you haven't measured multi-task cost — you've measured detection cost under multi-task training." This critique is fair. The response: multi-task cost is defined as the detection head's degradation relative to its single-task ceiling, because detection is the primary task (industrial assembly monitoring). Pose and PSR are auxiliary tasks that ride on the shared features. This is a defensible framing IF the paper explicitly adopts it. But if the paper's title is "Multi-Task Cost Analysis of Egocentric Assembly Monitoring" and only one task has a measured cost, the title overclaims.

**Required action:** Either (a) frame the paper as "Detection Cost Under Multi-Task Training" (narrower contribution, more defensible), or (b) commit to single-task baselines for at least one more head (pose is the cheapest: 2-3 GPU-days). Opus decides.

### Q4.03 — How does FPS interact with the cost story, and should it be reported alongside mAP?

**Status:** D-5 in 133 §1 flags that FPS comparison (ours 11.02 on RTX 3060 vs YOLOv8m 178 on V100) is hardware-disparate and potentially misleading.

**Deep question:** FPS is the deployment argument. A model that runs at 11 FPS on a sub-$500 GPU is usable for real-time assembly monitoring (assembly actions take 2-30 seconds). The latency contribution (ACT-7/133 §3) is that per-frame predictions at zero marginal latency beat clip-window approaches that require 178-1149ms latency. But the FPS comparison to YOLOv8m is meaningless without hardware normalization. Correct comparison: normalize to same GPU (RTX 3060 or equivalent). If YOLOv8m runs at 30 FPS on RTX 3060 (not 178 on V100), our 11 FPS is a 2.7x overhead for the full 4-head pipeline. That's an honest, informative number. If YOLOv8m runs at 15 FPS on RTX 3060 (unlikely — V100 is ~5x faster than RTX 3060 for inference), our 11 FPS for 4 heads is competitive.

**Required action:** Measure YOLOv8m inference FPS on RTX 3060. Compare to our 11 FPS. Report all FPS numbers with GPU model and batch size.

### Q4.04 — Are we double-counting cost by ignoring the detection-to-pose dependency?

**Status:** Detection boxes are non-differentiably fed to pose head (argmax → pseudo-keypoints). If detection fails, pose has no input.

**Deep question:** The multi-task cost on detection (64%) cascades into pose (if detection fails, pose has no input — argmax on bad detection boxes yields bad pseudo-keypoints). The paper measures pose MAE but doesn't control for detection quality. If detection degrades by 64%, and pose degrades by only 15% relative to a detection-oracle baseline, then the cascade effect is small (detection failures don't propagate to pose as severely as expected). But if pose degrades by 50%, much of the pose gap is actually the detection gap. This matters because it changes the intervention target: fixing detection would automatically improve pose. The cost story should account for task dependencies.

**Required action:** Compute pose MAE conditioned on detection success (IoU > 0.5) vs detection failure. Report both. If the gap is large, the cascade effect is a significant finding.

### Q4.05 — Is distillation (P2.1, 3 days) needed for the cost story to be complete, or is it a separate contribution?

**Status:** D-6 in 133 §1 endorses distillation as the one forward-looking detection experiment, with a 3-day timebox. The expected honest outcome is +0.1-0.2 mAP.

**Deep question:** Distillation changes the cost story from "64% cost, unrecoverable" to "64% cost, partially recoverable via distillation." This is a stronger paper: it measures the cost AND shows a mitigation strategy. But it also moves the goalposts: if distillation gets us to 0.55 (55% of ceiling, 45% cost), do we report the post-distillation number or the pre-distillation number? The answer: both. The cost measurement is pre-distillation (what is the natural multi-task degradation). The distillation result is a separate subsection: "Mitigating Multi-Task Cost via Knowledge Distillation." This preserves the cleanliness of the cost measurement while adding a positive result. Budget: 3 days, timeboxed. If it works, great. If it fails, report the negative result.

**Required action:** Run distillation (P2.1) after PSR head repair completes. Timebox at 3 GPU-days. Report result regardless.

### Q4.06 — Should the cost be reported per-task or as a system total?

**Status:** No current cost breakdown.

**Deep question:** A per-task cost breakdown would show:
- Detection: 64% cost (0.358 vs 0.995)
- Detection (COCO convention): 42% cost (0.573 vs 0.995)
- Pose: unknown (no single-task baseline)
- PSR: unknown (no single-task baseline)
- Activity: essentially 100% cost (0.028 vs 0.622 — but paradigm difference)

A system total would be: "4 tasks at 11 FPS, 46.5M params, 93 GFLOPs" vs "4 single-task models at 4× latency, 66.4M params, unknown GFLOPs." The system total is weaker because it compares apples to speculative oranges. The detection-specific cost is stronger because it has a measured denominator. Recommendation: report detection cost as the primary number. Report system-level overhead (params, FPS) as secondary.

**Required action:** Structure §5 (or §4) with "Detection Cost Under Multi-Task Training" as the empirical core. System-level overhead as a supporting paragraph.

### Q4.07 — How does the params/GFLOPs discrepancy (46.47M/245.3 in .tex vs ~53M/~93 in 129) affect the cost story?

**Status:** C-6 in 133 §0 documents a 2.6× GFLOPs discrepancy in the paper's own numbers. No one knows which is correct.

**Deep question:** A reviewer will catch a 2.6× discrepancy instantly. If the real number is 93 GFLOPs, the paper claims 245.3 — a 2.6x inflation that looks dishonest even if it was a measurement error. If the real number is 245.3, the model is less efficient than claimed. The fix: re-measure on the freeze checkpoint with a single, committed measurement script. Report the number with hardware, batch size, and framework (PyTorch eager vs compiled?). This is a 1-hour fix that prevents the paper's most easily falsifiable claim.

**Required action:** Re-measure GFLOPs and params once on the freeze checkpoint. Use `fvcore` or `ptflops` library. Commit the measurement script. This must happen before the paper is submitted.

### Q4.08 — Is the "sub-$450 GPU" claim accurate after corrections?

**Status:** The contribution audit corrected "$299" to "$429" (RTX 3060 street price, not MSRP).

**Deep question:** The "sub-$450 GPU" claim is defensible (RTX 3060 at $429, RTX 5060 Ti at ~$350-400 street). But the question is: does the paper need the GPU price argument? It's a deployment argument for a systems paper. If the paper's contribution is the cost measurement + pathology analysis, not the deployment-ready system, the GPU price is a minor point. If the paper leads with "deployable on sub-$450 hardware," it places weight on the FPS argument, which is weaker after the corrections (no hardware-normalized FPS, no D1R comparison).

**Required action:** Keep the price mention but don't lead with it. Move to a footnote or deployment section, not the abstract.

### Q4.09 — Does the cost story survive a review by a single-task learning expert?

**Status:** No.

**Deep question:** A detection specialist reads: "Our multi-task model achieves 36% of a single-task YOLOv8m's mAP." Their first question: "Why use a ConvNeXt-Tiny for detection? YOLOv8m was designed for detection. Your architecture comparison isn't 'multi-task vs single-task' — it's 'detection head on vision backbone' vs 'purpose-built detection architecture.' The 64% cost may be 90% architecture choice and 10% multi-task interference." The paper does not currently address this. The counter: the 4-model pipeline comparison (the .tex uses 66.4M params for 4 purpose-built models) partly covers this, but doesn't isolate the backbone effect. The correct experiment: train ConvNeXt-Tiny + detection head WITHOUT the other three heads (single-task, same backbone). Compare to the 4-head version. This isolates multi-task interference from architecture choice. This IS the equal-gradient-update ablation in the .tex. If it already exists, report it explicitly.

**Required action:** Verify the equal-gradient-update ablation exists and was computed on the same backbone. If yes, present it as the architecture-controlled cost measurement. If no, this is a gap that a detection expert will flag.

### Q4.10 — What is the correct "64% cost" or "36% of ceiling" phrasing?

**Status:** C-1/133 §0 identifies an internal contradiction: 131 says "64-68% of ceiling" and "-64% cost" interchangeably. The correct decimal: 0.358/0.995 = 0.3598 = 36% of ceiling = 64% cost.

**Deep question:** The two phrasings say the same thing but have opposite tonal valence. "We retain 36% of the single-task ceiling" sounds like a lament. "The multi-task cost is 64%" sounds like a finding. If the paper's narrative is "measurement and disclosure," the cost framing is correct: "We measure a 64% multi-task cost on detection." If the paper's narrative is "we benchmark our system," the efficiency framing is correct: "Our multi-task model retains 36% of the detection ceiling." The choice depends on §1's narrative hook. The paper should NOT use both interchangeably as 131 does — pick one and use it consistently. The Opus resolution (132 §2 Q5) endorses the cost measurement framing.

**Required action:** Standardize on "64% multi-task cost" throughout the paper. Remove all "36% of ceiling" phrasings. Consistency is a cheap fix that eliminates C-1 confusion.

---

## §5. Three Training Pathologies — Methodology Contribution (10 Questions)

Files 132-133 establish that the three training pathologies (dead PSR head, bounded Kendall, NaN checkpoint) collectively constitute the paper's novel methodology contribution. These 10 questions probe whether that contribution is publishable and how to present it.

### Q5.01 — Is the "dead PSR head" pathology publishable as a standalone finding?

**Status:** PSR-3/132 §2 Q1 provides the mechanism: ReLU(Linear(x)) with bias=-1.0 on output → ReLU inputs are pre-activations from a Linear layer with std=0.01 init, easily negative → ReLU gates gradient to zero → head dies. The per-component head is architecturally non-functional at initialization.

**Deep question:** Is this publishing a bug fix or publishing a finding? If the fix is "use LeakyReLU or GELU instead of ReLU," that's a well-known trick — not a publishable contribution. The publishable angle is: "Multi-task training with extreme label sparsity (PSR: 11 components, 0.1-100% prevalence) creates a specific failure mode where ReLU saturation kills per-component heads. Standard monitoring (loss curves, gradient norms) does not detect this because the shared trunk gradients hide the per-component gradient starvation." The contribution is the detection method (per-component gradient monitoring) and the architectural analysis (which components die and why). If the paper focuses on the detection method and the generalizable insight, it's a methodology contribution. If it focuses on the bug fix, it's a technical note.

**Required action:** Frame the dead-head pathology as a detection method + analysis, not a bug fix. The fix is the control/confirmation, not the contribution.

### Q5.02 — How many other papers have undiagnosed dead heads due to ReLU+bias initialization?

**Status:** Speculative. No literature survey exists.

**Deep question:** This is the generalizability argument. If the paper can argue that this pathology is common (Linear → ReLU → Linear is a standard MLP block; bias=-1.0 for focal loss is recommended; std=0.01 initialization is default for linear layers), then the finding applies to any multi-task system with imbalanced per-task gradients. The paper should estimate prevalence: "Any multi-task system where a sub-task head (a) has a ReLU after a low-variance linear layer, (b) uses focal loss with a nonzero baseline bias (e.g., -1.0), and (c) receives gradient updates only on a fraction of training steps (e.g., sequence-mode batching)" is at risk. This is a falsifiable, generalizable claim that a systems venue would find interesting.

**Required action:** Add a "generalizability" sentence to §5.1: three conditions that produce the dead-head pathology. This elevates the finding from a bug report to a methodology contribution.

### Q5.03 — Does Kendall uncertainty weighting with manual overrides invalidate the "automatic" claim?

**Status:** A-1 in 133 §5 confirms: with HP_PREC_CAP, FIXED_LAMBDA, per-task caps, and a full env-var bypass, the deployed Kendall is not automatic. The paper must say so.

**Deep question:** The paper's Pathology 2 is "Kendall uncertainty weighting under extreme label sparsity requires bounding." The theoretical analysis (fixed-point math for convergence under cap) is the contribution. The empirical leg is the fixed-weight ablation being planned. But here's the tension: if the PSR head was dead (Pathology 1), the Kendall analysis may be completely irrelevant — there was nothing for Kendall to weight because the PSR head had no gradient regardless of task weight. The PSR log_var=-0.04 (giving 4-8% down-weight) did not kill the head; the architectural death did. This means Pathology 2 may be a misdiagnosis: the paper thought Kendall was the problem, but it was actually a dead head. If the head repair succeeds, does the Kendall problem re-emerge? Only then is Pathology 2 confirmed.

**Required action:** Sequence the pathologies: Pathology 1 (dead head, proven) → Pathology 2 (Kendall bounding, conditional on head repair) → Pathology 3 (NaN checkpoint, proven). If Pathology 2 is unconfirmed after head repair, present it as theoretical analysis with the fixed-weight ablation as confirmatory evidence. Do NOT present it as an empirically observed pathology unless the data supports it.

### Q5.04 — Is the NaN checkpoint pathology (best-checkpoint promoted via broken metric) publishable?

**Status:** AC-1/133 §10: epoch 11 was promoted because the combined metric included a NaN component that inflated the aggregate. This is an infrastructure failure, not a methodology finding.

**Deep question:** "Our combined evaluation metric had a bug where NaN components were treated as zero, causing a checkpoint with failed branches to appear superior to a balanced one." This is publishable as a cautionary tale: "We document a bug in multi-task checkpoint selection where NaN penetration of the aggregate metric selects the worst checkpoint." The generalizable insight: any multi-task system using a weighted sum of per-task metrics without NaN guards will select checkpoints that favor missing/empty evaluations. The fix (epoch 18 promotion) and the audit procedure (metrics.json audit across all checkpoints) are the contribution. This is a short, actionable finding that reviewers would appreciate.

**Required action:** Write Pathology 3 as a 1-paragraph cautionary tale with the fix procedure. Make it generalizable (not just "our bug").

### Q5.05 — Should the paper present the pathologies as a separate section or integrated into the disclosure section (§5.4)?

**Status:** 132 §5 and 133 §9 PW-7 recommend Pathology 2 as theoretical analysis (not §5.4), AC-1 checkpoint invalidation as §4/§6 (not §5.4), and the 8 disclosures as §5.4.

**Deep question:** The document set is inconsistent on where the pathologies live. 130 P4.3 lumps them all into §5.4. 132 §5 separates them: Pathology 1 (dead head) and Pathology 2 (Kendall theoretical) go in §4 (architecture/methodology), Pathology 3 (NaN checkpoint) goes in §6 (infrastructure/integrity). The disclosures (§5.4) are the 8 numbered items from 132 §5. This is the correct structure: §4 for methodology contributions (pathologies 1-2), §5.4 for result-boundary disclosures (D4 F1=0, activity 0.028, etc.), §6 for infrastructure (NaN checkpoint). This prevents §5.4 from becoming a dumping ground.

**Required action:** Organize the paper with three separate integrity locations. §4: training pathologies (2 subsections: dead head, Kendall bounds). §5.4: honest disclosures (8 items, each with numbers). §6: infrastructure notes (NaN checkpoint, eval pipeline fixes).

### Q5.06 — Are there more undiagnosed pathologies in the training pipeline beyond these three?

**Status:** 133 documents several additional anomalies: A-6 (ACTIVITY_GRAD_BLEND_RATIO 0.05→1.0 — gradient scaling was progressively abandoned), A-4 (3-layer transformer on T=1 for 75% of batches — dead compute), A-5 (FeatureBank bypassed — dead code), TI-1 (CUDA crashes during training).

**Deep question:** Should any of these be elevated to "pathology" status? The gradient-blend abandonment (A-6) is the strongest candidate: the paper's own training regime was progressively simplified as mechanisms failed. This is actually a meta-finding: "Our multi-task training required progressively disabling advanced features (gradient blending, FeatureBank, Kendall automatic weighting) as they proved unstable under extreme label sparsity." This is a publishable arc: we started with a complex system and stripped it down to what works. The activity-grad-blend history is one paragraph of honest documentation.

**Required action:** Decide which anomalies are "pathologies" vs "infrastructure notes." Recommendation: keep the three named pathologies tight. Move A-6 (gradient blend abandonment) into §4 as part of Pathology 2's story (Kendall bounding is the theory, gradient blend removal is the practice). Keep A-4, A-5, TI-1 in supplementary.

### Q5.07 — Does the class-24 error-state pathology belong in §5.4 or is it a separate methodology finding?

**Status:** SOTA_STATUS.md §5.4 reports 0 GT instances of error_state in the entire dataset. WACV reports error-state FPR=65%. Our model's FPR=0% because it was never trained on error-state examples.

**Deep question:** This is both a disclosure (our model cannot detect error states because none exist in training) and a dataset finding (the IndustReal COCO dataset does not contain error-state annotations despite defining class 24 for that purpose). As a disclosure, it belongs in §5.4 with the D4 and activity disclosures. As a dataset finding, it could go in the dataset description section. The paper should include it in both places: dataset section for completeness, §5.4 for the specific comparison to WACV's 65% FPR. The disclosure text should be: "The 24-class ASD taxonomy defines error_state (class 24), but no frames in any split were annotated for this class. Our model's frame-level FPR is 0.0% because it was never exposed to the concept. WACV reports 65% error-state FPR on a model trained with actual error instances."

**Required action:** Add class-24 disclosure to §5.4 and a note to the dataset section. This is a 10-minute text addition.

### Q5.08 — Should the sequence-mode overhead be reported as a fourth pathology?

**Status:** A-3/133 §5: sequence-mode adds 25% training overhead, and the PSR head's gradient was dead for the per-frame batches (75% of steps). The overhead was partially or entirely wasted.

**Deep question:** Four pathologies dilute the "three pathologies" framing. Keep it at three unless the sequence-mode overhead is independently shown to harm training (not just waste compute). If it harms training (e.g., by destabilizing batch statistics), it qualifies as a pathology. If it just wastes compute, it's an efficiency note. The distinction: a pathology is a failure mode that degrades the model. Waste is a failure mode that degrades the training budget. Both matter for practitioners, but they have different solutions. For the paper's narrative, three is a cleaner number. Move sequence-mode overhead to the ablation table as a cost item.

**Required action:** Keep three pathologies. Move sequence-mode overhead to a supplementary paragraph about training efficiency.

### Q5.09 — What paper venue values pathology analysis over SOTA claims?

**Status:** Not yet determined.

**Deep question:** A pathology-focused paper is not a NeurIPS or CVPR submission (those venues expect SOTA results). The targets are:
- **AAIML (Applied AI for Manufacturing and Logistics)** — if it exists as a standalone venue. The .tex targets AAIML 2027. If AAIML values applied systems work, the pathology framing fits.
- **MLSys** — systems-focused. Pathology analysis as a "systems failure mode" is in scope.
- **ICLR Workshop on Practical ML** — workshop venues welcome honest accounting.
- **IEEE Access / Sensors** — applied journals that accept deployment-focused studies.
- **arXiv preprint + blog post** — if no venue fits, the honest accounting may find an audience outside peer review.

The paper should identify its target venue BEFORE writing §1, because the narrative hook changes. A systems venue gets the pathology framing. An applied manufacturing venue gets the practical deployment framing. A general ML venue needs at least one competitive result.

**Required action:** Identify target venue. Write §1 to match. The venue determines whether the paper is accepted or desk-rejected.

### Q5.10 — What is the one-sentence summary of the methodology contribution?

**Status:** No canonical sentence exists.

**Deep question:** Every paper needs a one-sentence contribution statement. After all the corrections, what is it? Proposed: "We document three training pathologies in multi-task egocentric assembly monitoring — dead sub-task heads from ReLU saturation under label sparsity, the need for bounded uncertainty weighting, and NaN-corrupted checkpoint selection — that account for the measured 64% multi-task cost on detection and establish evaluation baselines for per-frame action classification, PSR, and egocentric head pose on the IndustReal dataset." This is specific, honest, and comprehensive. It also names all four heads and the cost measurement. Does it fit within 2-3 sentences? With editing, yes.

**Required action:** Opus approves or revises this contribution statement. Every other section of the paper must support it.

---

## §6. Adversarial Review — Kill the Paper, Fix It

This section collects the strongest possible reviewer attacks and the fix for each. Each attack is phrased as a reviewer would write it; the fix describes what must be in the paper to survive it.

### Attack 1: "Your detection 0.995 is a single-task YOLOv8m, not your model. Why should I care about a cost ratio to a model you didn't build?"

**Fix:** Preempt in §1: "Our single-task detection baseline uses a standard YOLOv8m trained on the identical split. We measure cost as degradation from this ceiling because (a) it is the strongest achievable detection performance on this data, and (b) it isolates multi-task interference from architecture choice. The cost measurement is the contribution; the ceiling model is a yardstick, not a claim."

### Attack 2: "Your PSR 0.7499 is prevalence calibration, not learning. Components with >50% prevalence score 0.75-1.0 regardless of model quality."

**Fix:** Include the null-delta table (Q2.08). Show that low-prevalence components (comp 4 at 14%, comp 10 at 18%) achieve delta >+0.09 over the prevalence-prior null. The aggregate 0.7499 includes prevalence calibration AND genuine signal. The decomposition makes this transparent.

### Attack 3: "You claim 'first baseline' for head pose, but I can find 5 ego-pose papers on HoloLens. Your literature search was insufficient."

**Fix:** Document the systematic search in supplementary: databases searched, query terms, date range, results count. If any ego-pose paper on IndustReal exists, the claim changes to "competitive." If no IndustReal-specific work exists but ego-pose on other HoloLens datasets exists, the claim changes to "first on this dataset" (weaker but defensible).

### Attack 4: "The 64% cost is an artifact of your metric choice. Under COCO convention (excluding zero-GT classes), your cost is 42%. Which is it?"

**Fix:** Report both. Primary: WACV-matched convention. Secondary: present-class mAP. Explain the difference in one sentence. Do not hide either number.

### Attack 5: "You documented that your PSR head was dead. You fixed it. But you're reporting pre-fix numbers. Why should I believe the post-fix numbers when the pre-fix ones were wrong?"

**Fix:** The freeze protocol (132 §2 Q10) answers this: re-run every eval once against the freeze checkpoint. The paper reports post-fix numbers. The pre-fix numbers appear only in the pathology analysis as evidence of the dead head. The freeze checkpoint SHA256 is committed alongside the eval logs.

### Attack 6: "Your activity head at 0.028 is a null result. You don't benchmark activity. You show that your model can't do activity. Why is this in a results section?"

**Fix:** Activity is in the probe head section (§5.3), not a results section. It is explicitly labeled as a diagnostic head for studying multi-task interference. The results section reports only the three functional heads (detection cost, PSR per-frame F1, head pose MAE). The activity probe has its own subsection with baselines, linear probe, and temporal ablation.

### Attack 7: "Your paper has 8 disclosures in a dedicated honesty section. This reads as defensive. Why not fix the problems instead of disclosing them?"

**Fix:** The paper's thesis is that measurement + disclosure IS the contribution. The pathologies (dead head, bounded Kendall, NaN checkpoint) are the methodology findings. The eight disclosures are honesty-bound: they document what the reader cannot infer from the results alone. The tone must be confident, not apologetic. "We disclose these eight factors so that the reader can calibrate their assessment of each result" — not "we acknowledge these limitations."

### Attack 8: "Your Kalman smoothing improved MAE by 0.14-0.21° (1.5-2.7%). This is noise. Why is this in the paper?"

**Fix:** Move to supplementary. In the main paper, one sentence: "RTS Kalman smoothing over per-frame predictions yields <3% improvement in angular MAE, confirming that ConvNeXt-Tiny features are already temporally consistent (see Appendix X)." Do not dedicate a subsection.

### Attack 9: "You claim 'first per-frame PSR baseline' but STORM also does per-frame prediction. What's novel?"

**Fix:** Clarify the paradigm: STORM predicts transitions (event-based), not component states per frame. Our per-frame component-state prediction is a different paradigm. The paper must state: "We predict the state of each PSR component (engaged/disengaged) at every frame — a finer-grained prediction than STORM's transition events. This paradigm enables per-frame monitoring of assembly state changes." The novelty is the paradigm + the per-frame metrics, not the problem.

### Attack 10: "Your paper has no code release, no dataset release, and no reproducible training procedure. How is this useful to the community?"

**Fix:** Commit the best-checkpoint freeze, the eval scripts, and the metric computation code. Even if the full training setup is not reproducible (requires specific GPU setup), the evaluation protocol is. The paper should include an artifact-evaluation appendix with checkout + run instructions. This is table stakes for any systems venue.

---

## §7. Updated Master Plan (Day-by-Day, Next 2 Weeks)

This plan integrates:
- PSR head repair (in flight on RTX 5060 Ti)
- TCN+ViT activity (gated on temporal probe result)
- Detection distillation (P2.1, timeboxed)
- Paper writing (disclosures, pathology sections, paradigm comparison tables)
- The cheap decisive experiments identified in files 132-133

GPU allocation: RTX 5060 Ti (16 GB) for training, RTX 3060 (12 GB) for eval + analysis.

### Week 1 Day 1-2 (Jul 7-8): Cheap Decisive Experiments

| Task | GPU | Cost | What it decides | 
|------|-----|------|-----------------|
| PSR head activation diagnostic (1-hour forward pass, print per-component pre-ReLU activations) | RTX 3060 | 1 hr | Whether head repair is needed (expected: yes, ReLU inputs ≤0) |
| Null-model POS baselines (all-zeros + copy-prev, cached logits) | RTX 3060 | 1 hr | §5.2.1 null table content |
| P2.6 Transition F1 on epoch_18 predictions (same pipeline, transition metric) | RTX 3060 | 1 day | PSR narrative (if ~0.7: head is doing transitions; if <0.3: head is dead on transitions) |
| D-4 mAP convention check (email/code-read WACV eval, 30 min) | none | 30 min | Whether detection cost is 64% or 42% |
| C-2 D1 weights identity check (read logs, verify download) | RTX 3060 | 2 hr | Whether D1 mAP=0.0004 is interpretable |
| γ/β FiLM stats (one forward pass) | RTX 3060 | 1 hr | Whether FiLM is a pass-through (γ≈1, β≈0) |
| GT variance analysis (forward vs up-vector distribution) | none | 10 min | Explains forward vs up asymmetry |
| Per-class linear probe accuracy (non-majority classes) | RTX 3060 | 1 hr | Gates TCN+ViT spend |

### Week 1 Day 3-5 (Jul 9-11): Training Interventions

| Task | GPU | Cost | Notes |
|------|-----|------|-------|
| PSR head repair (bias 0.0, LeakyReLU, warm-start from epoch_18) + KENDALL_FIXED_WEIGHTS=1 | RTX 5060 Ti | 2-3 days | **CRITICAL PATH** — expected F1 0.83+ |
| Temporal averaging ablation (T=16 sliding window, uniform kernel) | RTX 3060 | 1 day | Gates TCN+ViT: if temporal averaging <0.05, TCN+ViT unlikely to exceed 0.10 |
| P1.3 In-process full eval (EVAL_MAX_BATCHES=0) | RTX 3060 | 1 day | Full eval mAP for detection (unblocks cost ratio) |
| 10-seed subsample variance (fallback if in-process fails) | RTX 3060 | 0.5 day | Variance bound for detection mAP |

### Week 1 Day 6-7 (Jul 12-13): Paper Writing + Analysis

| Task | Who | Cost |
|------|-----|------|
| §5.1 Dead-head pathology (with diagnostic results) | Paper author | 0.5 day |
| §5.2.1 POS null table + PSR paradigm comparison table | Paper author | 0.5 day |
| §5.4 Eight disclosures (with numbers from experiments) | Paper author | 0.5 day |
| §1 Introduction (pathology-first narrative hook) | Paper author | 1 day |
| LOO-CV for PSR thresholds (P2.5) — start on RTX 3060 | RTX 3060 | 2 days (runs in background) |

### Week 2 Day 1-3 (Jul 14-16): Conditional Execution

Gate decisions based on Week 1 results:

| Condition | Action | GPU | Cost |
|-----------|--------|-----|------|
| Head repair F1 ≥0.83 | Run LOO-CV (P2.5 started, completes) | RTX 3060 | completing |
| Head repair F1 ≥0.83 | Run D4 threshold re-tune on head-repair logits | RTX 3060 | 0.5 day |
| Head repair F1 ≥0.83 | Write §5.2 PSR results | paper | 1 day |
| Head repair F1 <0.80 | Diagnose: is head still dead despite LeakyReLU? | RTX 3060 | 1 day |
| Temporal averaging ≥0.05 | Train TCN+ViT (P1.4) | RTX 5060 Ti | 2-3 days |
| Temporal averaging <0.05 | Cut P1.4. Write activity as probe-only (§5.3) | paper | 0.5 day |
| In-process eval succeeds | Report full-set detection mAP | paper | — |
| In-process eval fails | Report 10-seed subsample ±σ | paper | — |
| D-4 confirms WACV uses COCO convention | Update all cost numbers to 42% cost | paper | 0.5 day |

### Week 2 Day 4-7 (Jul 17-20): Detection Distillation + Paper Completion

| Task | GPU | Cost | Notes |
|------|-----|------|-------|
| P2.1 Knowledge distillation (if compute available) | RTX 5060 Ti | 3 days | Timeboxed. Report result either way. |
| Per-recording head pose breakdown (P2.4) | RTX 3060 | 0.5 day | Median + IQR up-vector |
| Error-state FPR (D-7) | RTX 3060 | 1 hr | Counting on existing predictions |
| §2 Related work (paradigm comparison, literature search) | paper | 1 day | Include systematic search for ego-pose |
| §5.3 Head pose (first baseline framing, FiLM disclosure) | paper | 0.5 day | |
| §6 Infrastructure (NaN checkpoint, eval fixes) | paper | 0.5 day | |
| Abstract + Conclusion (consistent with contribution statement) | paper | 1 day | |
| Results freeze (hash + commit all eval outputs) | paper | 1 day | All numbers must trace to freeze checkpoint |
| GFLOPs/params re-measurement | RTX 3060 | 1 hr | Resolve C-6 discrepancy |
| .tex consistency pass (taxonomy 47→69, POS removal, cost phrasing) | paper | 1 day | Fix C-3, C-4, C-5 |

### Fallback plan if Week 1 experiments reveal fatal problems

**If PSR head repair F1 <0.80:** The head repair didn't work. Fall back to reporting global-threshold 0.7217 as primary. The PSR contribution becomes "paradigm comparison + null-delta analysis" rather than "competitive F1." Do not claim "near SOTA." Write §5.2 as a disclosure-heavy analysis.

**If temporal averaging <0.05:** Cut TCN+ViT entirely. Activity §5.3 becomes: "per-frame action classification is not feasible with a shared ConvNeXt-Tiny backbone and per-frame MLP. Temporal modeling is required but was not attempted due to compute constraints." This is a null result with a clear diagnosis.

**If in-process eval fails AND 10-seed subsample still NaNs:** Detection mAP becomes unreportable. The paper loses its strongest empirical contribution (the cost measurement). Fall back to: "multi-task detection cost measurement was prevented by metric accumulation instability; we report the 250-batch subsample of 0.358 as an estimate." This is a weak position. Consider whether the paper is publishable without the cost measurement.

**If head pose literature search reveals a citable ~11° ego-pose paper:** The "first baseline" claim weakens. The comparison becomes: "our 9.14° is competitive with [citation] 11° on a different dataset." Still defensible but weaker. The 2° gap is a real achievement if the dataset is harder.

---

## §8. Open Decisions for Opus

These are decisions that the evidence cannot resolve — they require judgment calls from Opus (or the lead author).

### Decision 1: What is the paper's target venue?

The venue determines the narrative hook, the contribution statement, and the minimum acceptable results. Options:
- **AAIML (Applied AI for Manufacturing):** Pathology framing + deployment argument. Minimum: three functional heads, cost measurement, honest disclosures. Activity as probe head is acceptable.
- **ICLR Workshop / MLSys:** Methodology framing. Minimum: three pathologies documented, cost measurement, first baselines. SOTA claims not required.
- **NeurIPS / CVPR:** Not viable after the corrections. Zero quantifiable "beats SOTA" claims survive.

**Recommendation:** AAIML or MLSys workshop. Do not submit to NeurIPS/CVPR — the paper would be desk-rejected for "insufficient novelty."

### Decision 2: Should the paper report detection cost at 64% or 42% (or both)?

This depends on D-4 (WACV's mAP convention). If WACV uses COCO convention (excludes zero-GT classes):
- Our number: 0.573 present-class mAP
- Cost: 42% (not 64%)
- The gap to WACV 0.838 shrinks from -0.480 to -0.265
- The entire detection narrative changes from "severe degradation" to "moderate degradation with a clear path"

Which story serves the paper better? A 42% cost is a better result metric but a weaker "problem to solve" narrative. If the paper's hook is "we discovered three pathologies," a 64% cost is a more dramatic pathology. If the hook is "here is a functional system," a 42% cost makes the system look better.

**Recommendation:** Report both. Primary = WACV-matched. Secondary = present-class-mAP. The paper's honesty is demonstrated by the transparency, not by which number is larger.

### Decision 3: Do we run MViTv2-S activity (P5.1) or cut it entirely?

**Data:** Linear probe 0.2169 ≈ majority baseline 0.2217. Backbone has weak frame-level signal. MViTv2-S would require a separate backbone (not shared), 5+ GPU-days, and breaks the "one model" architecture story.

**Recommendation:** Cut P5.1. The compute is better spent on PSR head repair + detection distillation + paper writing. Activity becomes a per-frame probe head with the temporal averaging ablation as the only temporal experiment.

### Decision 4: Activity: keep as probe head or remove from results entirely?

**Argument for keeping:** (a) Completes the 4-head story. (b) The linear probe provides actionable diagnostic information. (c) Removing it mid-audit looks like hiding a bad result.

**Argument for removing:** (a) A head performing at majority baseline is not a functional head. (b) The "interference probe" claim collapses if the head was untrained. (c) Three strong heads (detection cost, PSR per-frame, pose orientation) + one null head is an unbalanced paper.

**Recommendation:** Keep as probe head with explicit framing in §5.3. Title the section "Per-Frame Action Classification Probe" not "Activity Recognition." The linear probe analysis is genuinely informative about backbone feature quality.

### Decision 5: What is the paper's contribution statement (one sentence for the abstract)?

**Proposed:** We document three training pathologies in multi-task egocentric assembly monitoring — dead sub-task heads from ReLU saturation under label sparsity, bounded Kendall uncertainty weighting requirements, and NaN-corrupted checkpoint selection — that collectively account for a measured 64% multi-task cost on detection, while establishing first baselines for per-frame PSR (0.75 macro F1), egocentric head pose (9.14 degrees forward MAE), and per-frame action classification on the IndustReal dataset.

**Alternate (systems framing):** We present a multi-task ConvNeXt-Tiny for egocentric assembly monitoring with four heads, measuring a 64% detection cost under multi-task training and documenting three training pathologies with eight transparent disclosures, achieving competitive PSR (0.75 F1) and head pose (9.14 degrees) on the IndustReal benchmark.

**Alternate (discovery framing):** Multi-task training of a shared backbone for egocentric assembly monitoring reveals three reproducible pathologies — dead sub-task heads from ReLU saturation, automatic uncertainty weighting bounds under extreme sparsity, and NaN-corrupted checkpoint selection — together accounting for a 64% multi-task cost that single-task evaluations miss.

Opus chooses.

### Decision 6: What is the results freeze date?

**Proposed:** End of Week 2 (Jul 20, 2026). Whatever checkpoint is best-per-head on that date becomes the reporting checkpoint. Every eval re-run once against it. All numbers in .tex trace to that run. No new numbers after Jul 20.

**Risk:** If PSR head repair is still training on Jul 20 and hasn't converged, the freeze captures a mid-training checkpoint. Mitigation: freeze at end of Week 2 regardless of convergence — the paper reports what the model achieved in 2 weeks of focused training, not what it could achieve in infinite time.

### Decision 7: Should the paper include the "sub-$450 GPU" claim?

**Data:** The $429 RTX 3060 runs 4-head inference at 11 FPS. The cost claim is factually correct but the deployment framing may not match the methodology contribution.

**Recommendation:** Keep in the deployment section only. Do not put in the abstract or contribution list. If the paper's venue is a systems venue, this is a selling point. If it's a methodology venue, it's a footnote.

### Decision 8: How many of the 8 disclosures end up in the abstract?

**Data:** 132/133 recommend the abstract leads with the methodology contribution, not the disclosures. Disclosure text in the abstract reads as defensive.

**Recommendation:** Zero disclosures in the abstract. The abstract says "we document three pathologies with eight transparent disclosures" — naming neither the count nor the content. The disclosures are enumerated and discussed in §5.4.

### Decision 9: Do we commit the eval artifacts (SOTA_STATUS.md, metrics.json files) or leave them as workstation-only?

**Status:** 132 §7 recommends committing. Currently they are workstation-only.

**Recommendation:** Commit. They are KB-scale JSON/markdown files. The freeze protocol requires them. An artifact-evaluation reviewer needs them. The SHA256 of the freeze checkpoint must be committed alongside them. This is non-negotiable for a reproducible paper.

### Decision 10: What is the fallback paper if the 2-week plan fails to improve any metric?

**Worst-case scenario:** PSR head repair F1 < 0.80. Temporal averaging < 0.05. In-process eval fails. Detection cost remains ~36% (or 58% under COCO convention but with unreportable variance). Pose remains 9.14°/7.78° (first baseline, no comparison).

In that case, the paper has:
- A measured detection cost (64% or 42%, depending on convention)
- A documented dead-head pathology with diagnostic procedure
- A bounded-Kendall theoretical analysis
- A NaN-checkpoint cautionary tale
- Eight honest disclosures
- First baselines for three heads on IndustReal

**Is this publishable?** At a systems or applied venue (AAIML, MLSys workshop, IEEE Access): yes. At a general ML venue (NeurIPS, ICML, ICLR): no, because there is no competitive result to anchor the review.

**Fallback paper title:** "What Four Tasks Cost One Backbone: Training Pathologies in Multi-Task Egocentric Assembly Monitoring"

**Fallback contribution statement:** "We measure the cost of multi-task training on a shared ConvNeXt-Tiny backbone for egocentric assembly monitoring and document three training pathologies — dead sub-task heads from ReLU saturation, the need for bounded uncertainty weighting, and NaN-corrupted checkpoint selection — establishing first baselines for per-frame PSR state classification, egocentric head pose orientation, and per-frame action classification on IndustReal."

This is a honest, defensible paper. The question is: is it a strong enough paper for the target venue? Opus decides.

---

**End of 138. This file integrates all available evidence from files 130-133 and SOTA_STATUS.md. Files 134-137 (per-head deep dives) are being created in parallel and should be merged into §0's status table when they commit.**
