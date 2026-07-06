# 138 — Adversarial Review: Integration Plan & 50 Cross-Head Questions

**Review date:** 2026-07-07
**Source file:** `138_SOTA_INTEGRATION_AND_BEAT_PLAN.md` (50 integrated questions across 8 sections, 746 lines)
**Prerequisite reading:** 134 (detection debate), 135 (PSR debate), 136 (activity debate), 137 (head pose debate), 130 (master plan), 132-133 (Opus answers + 30 debate rulings)
**Reviewer role:** Adversarial integration agent — challenges the cross-head narrative, identifies gaps the per-head debates could not see, proposes alternative framings, and tests the master plan's robustness to failure.

---

## §A. Five Strongest Challenges to the Integration Narrative

### Challenge 1: The paper claims "our model spans four heads" but three of four have uncorrectable pathologies, and the integration narrative papers over this.

The §0 status table shows: detection D3 at 0.358 (64% cost, NaN-subsampled), PSR at 0.7499 (dead head, unrepaired), activity at 0.0236/0.028 (majority-baseline, no temporal model), pose at 9.14/7.78 (first baseline, no comparison). The file claims this is a "four-head multi-task ConvNeXt" but it is more honestly described as "one trained head (detection, degraded), one partially trained head (PSR, dead MLPs), one untrained head (activity, gradient-starved), and one linear probe (pose, no ablation)." A reviewer will count the functional heads and arrive at "one-and-a-half." The integration narrative depends on the heads being equivalently characterized, but the evidence basis is radically asymmetric: detection has a ceiling measurement from a different architecture, PSR has a null-delta analysis that proves the head learned something despite the dead path, pose has no single-task comparison, activity has a linear probe that equals the majority baseline. The §0 table's uniformity of formatting (rows with numbers in every cell) hides this asymmetry.

**Counter (from file's own defense):** The file explicitly distinguishes the heads in the status column (SILENT COCO FALLOFF, FLOOR BASELINE, FIRST BASELINE, GENUINE LEARNED SIGNAL). The asymmetry is documented in the footnotes, not hidden. The integration across heads is precisely about measuring and documenting these asymmetries, not about claiming uniform strength. The distinction between the §0 table (status documentation) and the §5.4 disclosures is the integrity mechanism.

**My assessment:** The footnotes are present but the §0 table's uniform-number format visually implies comparability. The visual presentation should mirror the epistemic asymmetry: use color coding or shading to indicate which numbers are ceiling measurements (detection D1R), which are honest primaries (PSR 0.7217), which are first baselines (pose), and which are probe heads (activity). A table with 14 rows of numbers that reads "look at all our results" but has only 2-3 trustworthy numbers is a reviewer trap, even with footnotes.

### Challenge 2: The 64% multi-task cost is detection-only on a cross-architecture denominator, yet it anchors the entire paper's contribution claim.

§4 opens with "Detection: 0.358 vs 0.995 = 36% of ceiling = 64% cost. This is the paper's primary empirical contribution." But Q4.02 and Q4.09 establish that: (a) no single-task ConvNeXt-Tiny detection ceiling exists, (b) the 0.995 is from YOLOv8m (a different architecture), and (c) no single-task baselines exist for PSR, activity, or pose. The paper's primary empirical contribution is "detection cost under multi-task training, measured against a cross-architecture ceiling, with three other heads unmeasured." This is a single-head efficiency measurement against a confounded ceiling — not a multi-task cost measurement. The 134 debate Challenge 1 makes this explicit: "The ceiling is cross-architecture, not same-architecture." The 138 file acknowledges the gap (Q4.02, Q4.09) but does not adopt the consequences: if the contribution is detection-only, the paper's title, abstract, and contribution statement must all change.

**Counter:** The file proposes two defenses: (a) the cost-ratio framing is honest because the ceiling measurement is explicitly labeled as such and used only as a denominator, and (b) the equal-gradient-update ablation (§4.09) partially controls for architecture. But the equal-gradient-update ablation's existence is unverified. Q4.09 says "If it already exists, report it explicitly. If no, this is a gap." This is not a defense — it is an unresolved question that determines the validity of the paper's central empirical claim.

**My assessment:** The cost measurement is the paper's best number, and the authors know it. But the cross-architecture denominator and detection-only scope mean the number says less than the paper needs it to say. The file's §4.02 recognizes this but punted the decision to Opus ("Opus decides"). The paper cannot proceed to writing without resolving whether the cost claim is detection-on-YOLOv8m-ceiling or multi-task-on-ConvNeXt-ceiling. These are different findings with different publishability.

### Challenge 3: The three training pathologies may be one pathology (Kendall weighting broken → cascading failures), and the paper treats them as independent findings.

§5 presents three pathologies: (1) dead PSR head from ReLU saturation, (2) bounded Kendall uncertainty weighting, (3) NaN checkpoint selection. The file deserves credit for questioning their independence in Q5.03: "if the PSR head was dead (Pathology 1), the Kendall analysis may be completely irrelevant — there was nothing for Kendall to weight because the PSR head had no gradient regardless of task weight." This is the correct insight, but the file does not follow through to the structural conclusion: if Pathology 1 and Pathology 2 are causally linked (dead head means Kendall never had to weight the PSR head, so the "bounded Kendall" finding is a theoretical analysis with no empirical leg), then the three pathologies reduce to two (dead head + NaN checkpoint) with a theoretical appendix on Kendall. The "three pathologies" framing is a hook — it sounds better than "two pathologies" — but it is partially speculative. A reviewer who reads Pathology 2's theoretical analysis and notices the missing empirical confirmation (because the fixed-weight ablation is unscheduled until head repair completes) will ask: "where is the experiment that proves Kendall needed bounding?"

**Counter:** The file explicitly sequences the pathologies in Q5.03: "Sequence the pathologies: Pathology 1 (dead head, proven) → Pathology 2 (Kendall bounding, conditional on head repair) → Pathology 3 (NaN checkpoint, proven). If Pathology 2 is unconfirmed after head repair, present it as theoretical analysis." This is the correct scientific approach — present proven findings first, conditional ones second. The risk is narrative: the abstract and introduction may lead with "three training pathologies" when empirically only two are confirmed.

**My assessment:** The file has the right idea (sequence the pathologies by confidence) but the paper's outline likely does not. The abstract (§1.07 Decision 5) and contribution statement (§5.10) both say "three training pathologies" as if all three are equally established. This is a writing risk that the file's own analysis has identified but not corrected. The fix: change all pre-submission references to "two proven training pathologies (dead head, NaN checkpoint) and one theoretical analysis (bounded Kendall)." If head repair confirms Kendall plays a role, upgrade to three.

### Challenge 4: The activity head at 0.028 has no architectural fix in flight and no plausible path to competitiveness, yet the integration plan keeps it as a "probe head" without quantifying what it probes.

§3 spends 10 questions on activity, concluding it should be "a probe head, not a competitive head." But Q3.08 asks the killer question: "if the activity head never trains, there's no interference to study." The interference claim (that adding an activity head hurts other heads through gradient competition) requires the activity head to have trained at all. If the activity head was gradient-starved for 75% of training (gradient blend ratio starting at 0.05, A-6), then there was no interference signal — the head was barely connected. The "probe head" framing is a salvage attempt that papers over the absence of any cross-task interference evidence. A reviewer will ask: "You say this head probes multi-task interference. Show me the interference. Without a single-task ablation and without evidence that the activity head's gradients affected other heads, your probe is a dead weight, not an instrument."

**Counter:** The linear probe (0.2169) provides the diagnostic value — it isolates backbone feature quality from MLP training artifacts. The per-frame MLP's 0.0236 is the baseline that proves per-frame classification cannot work without temporal modeling. These are separate findings from the interference story. The interference story (that adding activity hurts other tasks) is weaker and should be dropped until a single-task ablation confirms it.

**My assessment:** The file's Q3.08 and Q3.04 explicitly identify this problem. But the Summary (§5.10 contribution statement) and Decision 4 both keep the interference framing active. The integration plan should explicitly state: "Activity is a probe of backbone feature quality (via linear probe), not a probe of multi-task interference (because no interference was measured)." This is a small text change that prevents a large reviewer attack.

### Challenge 5: The 2-week plan's conditional execution tree has a symmetry problem — success conditions are well-defined, failure conditions are under-analyzed.

The Week 2 conditional table (lines 611-620) defines success: "Head repair F1 >= 0.83 → run LOO-CV, run D4, write §5.2." The failure branch: "Head repair F1 < 0.80 → Diagnose: is head still dead despite LeakyReLU?" This diagnosis (1 day) has no conditional actions after it — what happens after the diagnosis? There is no "if the head repair fails, do X" with the same specificity as "if F1 >= 0.83, do Y." The same applies to temporal averaging: if < 0.05, cut TCN+ViT, but then what? The activity section becomes a 2-page placeholder with no experiments in flight. The implicit assumption is that failure leads to fallback writing (§7 fallback plan, lines 639-647), but the fallback plan's viability is untested. "Report global-threshold 0.7217 as primary" assumes 0.7217 is stable. "Detection mAP becomes unreportable — consider whether the paper is publishable" admits the fallback may fail. The plan's failure branches do not loop back to success conditions — they terminate in "weaker position" or "weak position." The plan has no second-order recovery: if the fallback also fails, what? The only answer is §8 Decision 10's fallback paper, which the file itself describes as "a honest, defensible paper" that may not be competitive for any venue.

**Counter:** A research plan that defines recovery from every failure mode at two levels of detail is a research plan that never finishes. The conditional tree captures the highest-probability paths. The fallback plan captures the "everything goes wrong" scenario. The second-level details (what if the fallback fails?) are unknowable without running the first-level experiments. The plan is appropriately scoped for 2 weeks.

**My assessment:** This is reasonable for a 2-week timeline, but the plan's failure analysis should include concrete venue thresholds. "If PSR is 0.78 (not 0.83) and detection cost is 42% (not 64%) and activity remains at 0.0236, submit to [venue X]. If PSR is 0.72 and detection cost is unknown, do not submit — publish as arXiv preprint." The file's Decision 1 asks "what is the target venue" but there is no minimum-threshold table mapping numeric outcomes to submission decisions. Without it, the 2-week plan may produce a paper that gets submitted to the wrong venue because the authors don't realize the numbers are below the minimum.

---

## §B. Five Evidence Gaps

### Gap 1: No single-task ConvNeXt-Tiny detection ceiling exists, and the plan does not allocate compute for it.

Q4.09 identifies this gap correctly: "The correct experiment: train ConvNeXt-Tiny + detection head WITHOUT the other three heads." The file estimates 2-3 GPU-days for this experiment and defers to Opus (Q4.02). But the cost measurement (§4) is the paper's primary empirical contribution. A contribution with an unresolved denominator is not a contribution — it is a question. The plan should allocate compute for a single-task ConvNeXt-Tiny detection run, timeboxed at 2 GPU-days, scheduled for Week 1 or early Week 2. If the result shows 0.85 mAP for single-task ConvNeXt-Tiny, the multi-task cost is 42% (or 67% present-class), changing every narrative sentence in the paper. Not running this experiment means writing the paper's central claim with a confounded denominator.

### Gap 2: The training loss index verification for head pose (evaluate.py line 1280) is not scheduled, yet it determines whether the 7.78 deg up-vector is genuine.

137 Debate §B Gap 1 identifies this: "If the training loss used [3:6] (position data) as the up-vector target, then the model was trained to regress position coordinates into the up-vector output channels. The corrected eval would then show 7.78 deg not because the model is good, but because the eval now reads from the correct channels — channels that were trained on position data. This would mean the 7.78 deg number is meaningless." The 138 file mentions the up-vector fix (lines 28, 75-85) but does not reference or schedule the training loss verification. The Week 1 task table (lines 576-587) lists "Gamma/beta FiLM stats" and "GT variance analysis" but does not list "verify training loss head pose slicing." This is a 1-hour check (read evaluate.py line 1280, trace the loss computation, verify which indices the training loss references for up-vector supervision). If the training loss used position indices for the up-vector target, the 7.78 deg number is a measurement artifact and must be removed from the paper. This is a higher priority than FiLM gamma/beta computation (which only affects a novelty claim), yet it is unscheduled.

### Gap 3: The paper has no cross-head interference analysis — whether adding each head affects the others' performance.

The integration narrative claims the paper measures "multi-task cost" but the only measurement is detection vs its ceiling. There is no data on: does adding pose degrade detection? Does adding activity degrade PSR? Does adding PSR degrade pose? The equal-gradient-update ablation (if it exists) partially addresses this, but it is not verified to exist (Q4.09). The 138 file's own probes (Q3.08 on activity interference, Q4.04 on detection-to-pose cascade) identify the need but schedule no experiment. A true multi-task cost analysis requires at least 5 data points: single-task detection, single-task pose, single-task PSR, 2-task (det+pose), 3-task (det+pose+PSR), 4-task. This is 5 training runs at 2-5 days each = 10-25 GPU-days, or approximately the full project compute budget for 2 weeks. The plan does not make this tradeoff explicit. Either the paper accepts the single-cost-measurement scope and drops "multi-task" from the title, or it allocates compute for a proper ablation. Currently it does neither.

### Gap 4: The inference latency on RTX 3060 for the full 4-head system is not measured, yet the efficiency claim depends on it.

Q4.03 identifies this: "FPS comparison (ours 11.02 on RTX 3060 vs YOLOv8m 178 on V100) is hardware-disparate and potentially misleading." The file recommends measuring YOLOv8m on RTX 3060 but does not schedule it. The efficiency argument (46.5M params, 11 FPS on sub-$450 GPU) appears in the paper's contribution list. If YOLOv8m runs at 30 FPS on RTX 3060 (reasonable for a lightweight detection model), the 11 FPS for 4 heads looks like a 2.7x overhead per frame, which the paper must explain (the overhead of running 4 heads instead of 1). If YOLOv8m runs at 15 FPS, the 11 FPS for 4 heads is a success (close to single-task speed with 4x the outputs). The correct comparison determines whether the efficiency argument is a strength or a weakness. The experiment costs 30 minutes. Not scheduling it means writing the efficiency section with a guestimate.

### Gap 5: The training loss crash issues (TI-1 in 133, CUDA crashes during training) are not addressed in the plan, and the paper's narrative does not account for them.

Q5.06 mentions A-4, A-5, TI-1 (CUDA crashes) as anomalies that could be supplementary or elevated to pathology status. The file recommends keeping them in supplementary. But CUDA crashes during training are not a minor infrastructure issue — they mean the training pipeline produced partial checkpoints, potentially with different numbers of epochs per head, different data subsets, or different random seeds for restarted runs. If the training crashed and restarted mid-epoch, the checkpoint that produced every number in the paper may have been trained on a different data distribution than intended. The 138 file's §6 (Adversarial Attack 5 on freeze protocol) partially addresses this by establishing a post-freeze eval protocol. But if the CUDA crashes mean the model weights are non-deterministic (different runs produce different weight configurations), the freeze protocol's SHA256 commitment provides traceability but not reproducibility. The file should address TI-1 explicitly: "Were any checkpoints affected by CUDA crashes? If yes, which ones, and how was the effect quantified?"

---

## §C. Five Alternative Interpretations

### Interpretation 1: The paper should be reframed as "first baselines on IndustReal" rather than "SOTA-beating multi-task system."

The 138 file's §1.03 asks this explicitly: "If 'first baseline' is the claim for all four heads, does the paper lack novelty?" The file's answer is "the three training pathologies and eight disclosures constitute a methodology contribution." But this is an assertion, not an argument. The alternative interpretation is: a paper with four first baselines, three training pathologies (that are partially linked), and eight disclosures (that are mostly confessions of error) is a technical report, not a research contribution. The 138 file's Decision 10 (fallback paper) implicitly agrees: the fallback paper title is "What Four Tasks Cost One Backbone: Training Pathologies in Multi-Task Egocentric Assembly Monitoring" — which is a methodology framing, not a results framing. The question the file does not answer: is the methodology contribution sufficient for a first-tier venue? Q5.09 asks about venue but gives options ranging from AAIML to arXiv. The answer determines everything about the paper's narrative and nothing in the plan addresses it with evidence.

### Interpretation 2: The detection cost may be 90% architecture choice and 10% multi-task interference, not the other way around.

Q4.09 from a detection specialist: "Your architecture comparison isn't 'multi-task vs single-task' — it's 'detection head on vision backbone' vs 'purpose-built detection architecture.'" The file's counter (equal-gradient-update ablation) is unverified. If the ablation does not exist, the cost measurement is architecture-confounded and the paper's central claim is misleading. The equal-gradient-update ablation's existence should be verified before any further narrative development, because if it does not exist, the paper cannot claim any multi-task cost measurement at all.

### Interpretation 3: The three pathologies are one pathology: Kendall weighting broke, causing PSR dead head → NaN checkpoint → activity gradient starvation.

Challenge 3 addressed this partially. But the cascade goes further: if Kendall weighting was broken from the start (incorrect log_var initialization, gradient suppression on low-prevalence tasks), then:
- PSR head never received enough gradient to overcome ReLU saturation (Pathology 1 = symptom of Pathology 2)
- The NaN checkpoint was promoted because the broken weighted metric favored checkpoints where failed branches (from Pathology 1) had zero contribution to the aggregate metric (Pathology 3 = symptom of Pathology 1)
- Activity gradient blend was increased because the head never trained (A-6 = symptom of Pathology 2)

If this cascade is correct, the paper has ONE pathology (Kendall weighting under extreme sparsity) that cascaded into the other problems. The "three pathologies" framing overclaims the number of independent findings. The cascade interpretation is actually a stronger paper — "one root cause, three manifestations" — but the file does not consider it.

### Interpretation 4: The 8 disclosures weaken the paper by signaling defect density, not transparency.

Attack 7 in §6 addresses this: "Your paper has 8 disclosures in a dedicated honesty section. This reads as defensive." The file's fix (confident tone) is about presentation, not substance. A reviewer reading a disclosure list that includes "the detection mAP is from a 2.6% subsample," "the PSR head was dead," "the activity head is at majority baseline," and "the head pose comparison is uncitable" will conclude the project had pervasive quality issues rather than the authors were transparent. The alternative interpretation: the disclosures should be interleaved with the results they qualify, not collected in a single "confessions" section. A dedicated §5.4 says "here is everything we did wrong" — a confident §5.4 that treats each disclosure as a finding, not an excuse, may still fail because the reader stops at "8 things wrong with this paper."

### Interpretation 5: The paper is unpublishable at any venue that requires a positive result, and the team should prepare an arXiv preprint.

This is the honest consequence of the evidence in 130-138. The file's §8 Decision 10 asks "what is the fallback paper" and proposes a methodology-focused alternative. But the file does not assess: would this fallback paper be accepted at AAIML? At MLSys workshop? At any venue? The evidence from Q5.09 is: "A pathology-focused paper is not a NeurIPS or CVPR submission." The targets are lower-tier. But the lower-tier venues still require some positive result — a workshop paper showing "here is a problem we identified and a partial fix" requires the fix to work. If the fix (head repair) improves PSR from 0.7499 to 0.80, is that enough? If detection cost is 42% (not 64%), is that enough? The file does not answer these questions with venue-specific thresholds. The team should identify the minimum publishable result for their target venue before Week 1 experiments begin, not after.

---

## §D. Five New Questions the Integration Plan Should Answer

### New Question 1: What is the AAIML submission deadline, and does the 2-week plan align with it?

The file references AAIML as the target venue (Q5.09, Decision 1) but never states the submission deadline. If AAIML's deadline is August 1, 2026, the 2-week plan (Jul 7-20) + 1 week writing + 1 week buffer aligns with a mid-August deadline. If the deadline is July 15, 2026, the 2-week plan must be compressed into 1 week and the scope must be reduced. If the deadline is not until October 2026, the team should invest in stronger experiments (single-task baselines, MViTv2-S, full factorial ablations) rather than the current 2-week sprint. The deadline determines the entire plan's feasibility. The file should state the deadline and show the calendar alignment.

### New Question 2: What is the minimum publishable unit (MPU) — the smallest subset of heads and analyses that constitutes a publishable paper?

If the 2-week plan fails to improve any metric, the team needs to know whether any subset is publishable. The candidate MPUs are:
- **Detection-only:** 0.358 mAP on ConvNeXt-Tiny multi-task + 0.995 ceiling. Is "multi-task cost on a first baseline" publishable alone? ~5 pages, 1 table.
- **PSR-only:** 0.7499 F1 (first baseline) + null-delta analysis + paradigm comparison. ~7 pages, 3 tables.
- **Pose-only:** 9.14 deg forward / 7.78 deg up (first baseline on IndustReal). ~4 pages, 1 table.
- **Pathology-only:** Three training pathologies (methodology contribution). ~6 pages, no result tables.

The 138 file should evaluate each MPU for: page count, independent publishability, error-bar completeness, and venue match. This would inform a go/no-go decision at the end of Week 1 — if the integrated paper is not viable, which single-head paper can be produced in the remaining week?

### New Question 3: What is the recovery plan if the PSR head repair training actively degrades F1 below the epoch 18 baseline?

The in-flight training on RTX 5060 Ti applies both PSR head repair (ReLU → LeakyReLU, bias 0.0) and Kendall fixed weights. 135 Debate Challenge 5 (bundled intervention) and 135 DQ-1 (rollback plan) raise the risk: the repair could reduce F1 from 0.7499 to 0.65 or lower. The 138 file's fallback plan (lines 640-642) says: "If PSR head repair F1 < 0.80: report global-threshold 0.7217 as primary." This assumes 0.7217 remains available as a fallback — but if the head repair training overwrites the epoch 18 checkpoint and produces worse weights, the fallback must use the cached epoch 18 checkpoint, which may not be saved. The file does not specify: which checkpoint is the fallback, is it preserved on disk, and what is the recovery action (restore + abort repair, or continue with degraded performance)? The plan needs a "dead checkpoint recovery" step: save epoch 18 weights to cold storage before starting head repair, and define the abort threshold (F1 drops below 0.72 at any epoch → stop, restore, regroup).

### New Question 4: Should the paper drop the "multi-task" framing entirely and restructure as a pathology case study?

The evidence that this is a genuine multi-task system is thin: the four heads share a backbone but three of four are non-functional or weakly trained. The paper could be restructured as "Case Study: Training Pathologies in Multi-Head Egocentric Models" with a single-head ConvNeXt-Tiny detection baseline as the control condition. The four heads become the experimental conditions, each revealing a different failure mode. The "multi-task cost" becomes the detection head's degradation under multi-head training, which is one data point in the case study. This restructuring would: (a) eliminate the "insufficient novelty" attack (Q1.03), (b) convert the activity head from a weakness to evidence, (c) make the paper a methodology contribution with clear generalizability claims (Q5.02), and (d) match the fallback paper's framing (Decision 10) without requiring fallback conditions to trigger. The cost: the paper loses the positive result framing entirely, and some venues (applied ML, manufacturing) may reject a case study format. The benefit: the paper becomes reviewer-proof against "your results are weak" because the results were never the claim — the methodology findings were.

### New Question 5: How does the paper handle the CUDA training crashes in the text — mention, explain, or hide?

The 138 file's Q5.06 delegates CUDA crashes (TI-1) to supplementary. But the training crashes affect the fundamental claim of the paper. If the training crashed and restarted at a different epoch, the checkpoint that produced the paper's numbers may have been trained on a different number of epochs than reported, with different random seeds, and potentially on a different data subset. The paper should disclose this. The disclosure text should state: "The training pipeline experienced CUDA crashes during the multi-head training run. Checkpoints from crashed runs were discarded; the reported checkpoint (epoch 18) completed training without interruption. The crash frequency was [X] crashes per [Y] training hours, and [Z]% of training attempts produced usable checkpoints." If the crashes were isolated, the disclosure is one sentence. If the crashes were frequent, the disclosure must include a sensitivity analysis (do checkpoints from crashed-and-restarted runs produce significantly different results?).

---

## §E. Cross-Head Synthesis: What Each Per-Head Debate Reveals That the Integration Plan Misses

### From 134 (Detection Debate)

The detection debate's strongest challenge — "cross-architecture ceiling is not the right ceiling" — directly undermines the 138 file's primary contribution claim (§4). The 138 file addresses this in Q4.09 but does not schedule the fix. The 134 debate also reveals that D3's full-set eval produces no detection metrics, meaning the 0.358 number is from a 2.6% subsample. The 138 file's §7 schedule includes "P1.3 In-process full eval" but this is a Week 1 Day 3-5 task, meaning the paper's main detection number will be written before it is verified. The writing schedule should be reordered: verify the detection number (Day 1-2) before writing §4 (Day 6-7).

### From 135 (PSR Debate)

The PSR debate's most actionable finding is the input_dim mismatch (512 vs 768) — a blocking diagnostic that costs 5 minutes to run. The 138 file does not schedule it. The PSR debate also reveals that the 0.7499 number is from a 10k subset, not the full 38k eval. The 138 file's status table (§0) uses 0.7499 as the headline PSR F1 without caveating that it is a subset number. The 0.7217 global-threshold F1 (which the file identifies as "honest primary") should be the §0 headline, with 0.7499 as the optimal upper bound.

### From 136 (Activity Debate)

The activity debate's core challenge — "linear probe is statistically indistinguishable from majority baseline" — means the temporal architecture decision tree (TCN+ViT) is built on a false positive. The 138 file's §3.01 asks for per-class linear probe analysis to resolve this but does not schedule it. The 136 debate also identifies the structural pattern of silent error handling (bare except Exception in temporal probe, NaN detection eval, PSR dead head). The 138 file's §5.05 and §5.06 mention some of these but do not connect them into a pattern. The integration plan should include a "codebase health" paragraph in the limitations section that addresses the systematic error-handling fragility.

### From 137 (Head Pose Debate)

The head pose debate's blocking issue — training loss index verification for the up-vector — is entirely absent from the 138 file. The 138 file references the index bug fix (line 28) but does not verify that the training loss used the correct indices. If the training loss trained on position data for what the eval reads as up-vector, the 7.78 deg number is the most dangerous number in the paper — it looks right but measures the wrong thing. This verification should be a Day 1 task, scheduled before any pose writing.

### Four instructions from the per-head debates that the integration plan must adopt

1. **From 134:** Verify D3 full-set eval produces detection metrics before writing §4 detection cost.
2. **From 135:** Run `print(x.shape)` on PSR head input to confirm 512 vs 768 dims before any head repair analysis.
3. **From 136:** Run per-class linear probe accuracy (cached features, 30 minutes) before any temporal architecture discussion.
4. **From 137:** Verify training loss head pose index slicing before writing any pose numbers.

---

## §F. Summary

### Top 3 Challenges (Most Threatening to the Integration Plan)

1. **The detection cost is detection-only on a cross-architecture denominator.** The paper's primary contribution is "0.358 vs 0.995 = 64% cost for one head." This is not a multi-task cost measurement — it is a single-head efficiency number against a ceiling from a different architecture. Without a single-task ConvNeXt-Tiny detection baseline, every cost sentence in the paper is confounded. The file acknowledges this gap but does not schedule the fix. This is the most dangerous challenge because it undermines the paper's strongest claimed contribution.

2. **The activity head at 0.028 has no path to relevance and the "probe head" framing lacks evidence of interference.** The paper claims activity is a "probe of multi-task interference" but the head was essentially untrained (gradient blend ratio 0.05 for most of training). No measurement of cross-head interference exists. If activity never trained, it could not interfere. The probe framing is a salvage attempt that papers over an empty result. The honest position is: activity is a null result on per-frame action classification using ConvNeXt-Tiny features, with no conclusions about multi-task interference.

3. **The three training pathologies may be one cascade, and the "three" hook overclaims.** The file's own Q5.03 identifies the Kendall-dead head link but does not restructure the narrative. If one root cause (Kendall weighting under extreme sparsity) produced the dead head, the NaN checkpoint, and the activity starvation, the paper has one finding with three manifestations — not three independent findings. A one-finding paper at a systems venue is a short paper or a workshop paper, not a full conference submission.

### What the Integration Plan Does Well

The 138 file is the strongest document in the 130-138 series. It systematically identifies every causal link between the per-head findings. It probes its own assumptions (Q5.03 on pathology independence, Q3.08 on interference evidence, Q4.09 on architecture confound). It structures a 2-week plan with conditional gates that reflect the uncertainty in each deadline. The Fallback Plan (lines 639-647) is honest about worst-case outcomes. The Open Decisions (§8) correctly identify judgment calls that evidence cannot resolve. The Adversarial Review (§6) is the most candid self-critique in the entire document set.

### What Must Change Before Opus Sees This

1. **Schedule the four blocking diagnostics** (from §E above) as Day 1 Week 1 tasks. Do not write any section that depends on an unverified number.

2. **Restructure the contribution statement** to reflect the actual evidence: "We measure detection cost under cross-architecture comparison at 64% (or 42% under COCO convention), document two proven and one theoretical training pathology, and establish first per-frame baselines for PSR (0.72 F1 global threshold), head pose (9.14/7.78 deg), and action classification (0.0236 top-1) on IndustReal."

3. **Add a minimum-threshold table** mapping numeric outcomes to venue decisions, so Opus can see whether the 2-week plan's output is sufficient for the target venue.

4. **Remove or downgrade the "multi-task" framing** unless a single-task ConvNeXt-Tiny detection ceiling is produced. The paper should explicitly say "per-task cost for detection, with other task costs unmeasured."

5. **Acknowledge the CUDA crash training infrastructure issues** in the limitations section with a concrete disclosure paragraph.

---

**End of 138_SOTA_INTEGRATION_DEBATE.md. Prepared to make the integration plan ironclad before Opus sees it.**
