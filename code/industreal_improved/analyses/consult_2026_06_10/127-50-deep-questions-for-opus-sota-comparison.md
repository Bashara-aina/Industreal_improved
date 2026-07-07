> **SUPERSEDED** by `AAIML/132_OPUS_ANSWERS.md` and `AAIML/133_OPUS_COMPLETE_ANSWERS.md`. This supplementary file contains a stale attribution (mAP=0.995 attributed to ConvNeXt head instead of separately-trained YOLOv8m) and head-pose numbers from a pre-fix era. See C-1 and C-5 in 133 §0.

# 127 — 50 Deep Questions for Opus: SOTA Comparison Section

**Generated:** 2026-07-06
**Purpose:** 8 questions about SOTA comparison framing — which comparisons survive peer review, which are paradigm artifacts, and how to honestly position our results against STORM-PSR, B3, T3 MViTv2-S, and YOLOv8m.
**Evidence base:** SOTA benchmarks (`AAIML/industreal-sota-benchmarks.md`), all-papers benchmarks (`AAIML/industreal-all-papers-benchmarks.md`), comparability matrix (`AAIML/comparability-matrix.md`), contribution audit (`AAIML/contribution-audit-reviewer-factcheck.md`), competitor analysis (`AAIML/08_COMPETITOR_ANALYSIS.md`), current SOTA status (`src/runs/rf_stages/checkpoints/SOTA_STATUS.md`).

---

# Section: SOTA Comparison — What We Can Honestly Claim

Our four tasks land in fundamentally different comparison regimes, from "clearly beats SOTA" (detection at epoch 18) to "not comparable, different paradigm" (activity). These 8 questions probe the boundaries of each claim so the paper's SOTA positioning is bulletproof under review.

---

## Q1. Detection SOTA Claim — The D1 Audit Question

**Context:** At epoch 18, our detection head achieves mAP50=0.995 and mAP50-95=0.861, surpassing YOLOv8m's published 0.838 on the same IndustReal dataset (`SOTA_STATUS.md:9`). However, the D1 full evaluation (YOLOv8m eval on our split with our class mapping) produced mAP50=0.0004 — a number so low it indicates a broken class mapping or evaluation protocol mismatch, not a genuine comparison (`SOTA_STATUS.md:10`). The comparability matrix flags this as needing "YOLOv8m eval on our split (Experiment D1)" before any SOTA claim (`AAIML/comparability-matrix.md:21`). The detection head uses ConvNeXt-Tiny (random init) with RetinaNet-style subnets, operating at 3x fewer tasks than the original single-task YOLOv8m. The original YOLOv8m used COCO pretrain + 100K synthetic Unity frames + 26.9K real frames.

**Question:** The epoch 18 detection numbers (0.995 mAP50, 0.861 mAP50-95) are internally consistent and clearly beat YOLOv8m's published 0.838 — but the D1 cross-evaluation result (0.0004) raises a red flag. Which of the following explanations is most likely, and what one experiment definitively resolves the question?

(a) The class mapping between our 24-class ASD codes and the original YOLOv8m evaluation protocol is wrong — we are comparing apples to oranges. The D1 needs a careful audit of which class IDs/names correspond.

(b) The YOLOv8m weights from the original repo are not directly compatible with our validation split (different frame selection, frame index mapping, or temporal sampling). The 0.838 was reported on the original paper's dev/test split, not our internal split.

(c) Our epoch 18 evaluation uses a different mAP computation (e.g., different IoU thresholds, different NMS parameters, different confidence thresholds) than the published 0.838. The actual gap is smaller or reversed under identical protocol.

(d) The 0.995 is real — our ConvNeXt-Tiny has genuinely surpassed YOLOv8m on this task through longer training (epoch 18 vs presumably fewer epochs for YOLOv8m) and the multi-task regularization benefit.

**Why this matters:** If the detection SOTA claim is wrong, the paper's strongest quantitative claim collapses. The D1 audit (`AAIML/comparability-matrix.md:21`, `SOTA_STATUS.md:52`) directly determines whether "beats SOTA on detection" or "mAP=0.0004 indicates broken mapping" is the headline in the detection section.

**Evidence:**
- `SOTA_STATUS.md:9-11` — epoch 18: mAP50=0.995, mAP50-95=0.861; D1 full eval: mAP50=0.0004
- `AAIML/industreal-sota-benchmarks.md:44-49` — YOLOv8m published: 0.838 (COCO pretrain, Real+Synth)
- `AAIML/comparability-matrix.md:19-22` — Detection partially comparable, needs D1 + Ablation A
- `AAIML/industreal-all-papers-benchmarks.md:109-111` — Our mAP50=0.317 at epoch 11 (pre-epoch-18 improvement)
- `AAIML/industreal-sota-benchmarks.md:55-56` — YOLOv8m FPS=178 on V100

**Evidence missing:** D1 class mapping audit not yet run. YOLOv8m weights not yet run on our exact validation split with identical mAP computation. Without this, the 0.995 vs 0.838 comparison is not cross-validated.

---

## Q2. STORM-PSR Paradigm Gap — The Transition Detection Question

**Context:** STORM-PSR achieves F1=0.901, POS=0.812, tau=15.5s on IndustReal PSR (`AAIML/industreal-all-papers-benchmarks.md:51-56`). Our PSR head at epoch 18 achieves F1=0.7217 (global threshold 0.10) or 0.7018 (per-comp optimal thresholds) with the MonotonicDecoder (`SOTA_STATUS.md:16-17`). But this is a paradigm difference: STORM-PSR detects **transitions** (step completions) using temporal processing + procedural knowledge + ASD stream from a YOLOv8m backbone at mAP=0.838. Our PSR head performs **per-frame 11-component binary state recognition** as a byproduct of multi-task training from a ConvNeXt-Tiny backbone at mAP=0.317 (`AAIML/industreal-sota-benchmarks.md:99-106`). The comparability matrix rates this as "NOT COMPARABLE (Paradigm Difference)" (`AAIML/comparability-matrix.md:40-54`). The D4 experiment (YOLOv8m -> MonotonicDecoder) gave F1=0.000 with POS=0.999 — the decoder fails utterly even with SOTA detection because YOLOv8m detects objects on <1% of frames while the ground truth expects continuous state vectors (`SOTA_STATUS.md:18`).

**Question:** Does the 0.7018 F1 represent genuine progress toward STORM-PSR's 0.901, or is it a fundamentally different quantity measured in different units? Specifically, the D4 result (YOLOv8m -> MonotonicDecoder: F1=0.000) suggests our MonotonicDecoder is structurally incapable of producing the transition-based evaluations that STORM-PSR's F1 measures, even when fed perfect detection. If this is the case, does "approaching STORM-PSR F1" become a misleading frame even if the raw numbers appear close?

**Why this matters:** A hostile reviewer will point out that STORM-PSR's F1 measures transition detection accuracy (did you find the exact frame where step 4 completed?) while our F1 measures per-component state match (is component 7 correctly identified as ON in this frame?). These are described by the same symbol (F1) but measure entirely different competencies. Publishing "F1=0.7018 vs STORM-PSR F1=0.901" without paradigm disclosure would be flagged as deceptive.

**Evidence:**
- `AAIML/industreal-all-papers-benchmarks.md:51-56` — STORM-PSR: F1=0.901, POS=0.812, tau=15.5s
- `AAIML/industreal-sota-benchmarks.md:99-106` — paradigm disclosure: per-frame component recognition vs transition detection
- `AAIML/comparability-matrix.md:40-54` — NOT COMPARABLE, different paradigm
- `SOTA_STATUS.md:18` — D4: YOLOv8m -> MonotonicDecoder yields F1=0.000
- `SOTA_STATUS.md:16-17` — Our PSR: 0.7217 (global) / 0.7018 (per-comp optimal)
- `AAIML/industreal-all-papers-benchmarks.md:118` — paradigm difference note

**Evidence missing:** A transition-based F1 evaluation on our model (requires computing transitions from per-frame state predictions, then matching against GT transition timestamps). The current F1@plusminus3 is frame-based, not transition-based. Need to clarify exactly what our eval computes vs what STORM-PSR publishes.

---

## Q3. B3 Comparison — The Procedural Knowledge Confound

**Context:** B3 (WACV 2024) achieves PSR F1=0.883, POS=0.797, tau=22.4s using detection outputs from YOLOv8m (mAP=0.838) plus procedural knowledge (the B3 baseline restricts candidate steps to those expected by the assembly procedure) (`AAIML/industreal-all-papers-benchmarks.md:35-42`). B3 is a heuristic baseline, not a learned decoder: it accumulates detection confidence over time and applies procedural rules to determine step completions. Our MonotonicDecoder learns per-frame binary state from raw features, then applies fill-forward ordering constraints. B3 has two advantages we lack: (1) a 4x stronger detection backbone (0.838 vs 0.317 mAP at epoch 11, though our detection now exceeds 0.838 at epoch 18), and (2) explicit procedural knowledge that prevents impossible step sequences (`AAIML/comparability-matrix.md:42-51`). The comparability matrix says this is partially comparable after backbone swap (Experiment D4), but D4 gave F1=0.000.

**Question:** Now that our detection head at epoch 18 achieves mAP50=0.995 (exceeding YOLOv8m's 0.838), does the backbone-gap argument for incomparability to B3 still hold? If we freeze our epoch 18 detection weights and retrain only the MonotonicDecoder, would the PSR F1 approach B3's 0.883? Or is the D4 result (YOLOv8m -> our decoder: F1=0.000) evidence that our decoder architecture itself — not the detection quality — is the limiting factor, regardless of backbone quality?

**Why this matters:** If our detection is now BETTER than YOLOv8m, the "weaker backbone" excuse no longer applies. The PSR gap would be wholly attributable to paradigm difference (per-frame vs transition) and decoder architecture. This changes the paper's framing from "our PSR is competitive despite weaker backbone" to "our PSR uses a fundamentally different approach where per-component state recognition achieves 0.75 F1, which is not comparable to transition detection."

**Evidence:**
- `AAIML/industreal-all-papers-benchmarks.md:35-42` — B3: F1=0.883, POS=0.797, YOLOv8m backbone
- `AAIML/comparability-matrix.md:40-54` — NOT COMPARABLE, paradigm difference
- `SOTA_STATUS.md:9` — Our detection now mAP50=0.995 (exceeds YOLOv8m)
- `SOTA_STATUS.md:18` — D4: YOLOv8m -> MonotonicDecoder yields F1=0.000
- `AAIML/industreal-sota-benchmarks.md:83-97` — B3 performance breakdown (all recordings, error recordings)
- `AAIML/contribution-audit-reviewer-factcheck.md:15-21` — Claim 6: PSR evaluation integrity confirmed

**Evidence missing:** A MonotonicDecoder retrain using frozen epoch-18 detection weights (i.e., upstream detection quality equivalent to YOLOv8m or better). This would isolate the decoder contribution from the detection contribution. Also missing: adding procedural knowledge constraints as a training loss (not just decoding constraint) — the PSR reviewer identified this as a open path (`AAIML/reviewer-3-psr-paradigm-reconciliation.md:64-76`).

---

## Q4. T3 MViTv2-S Baseline — The Remapped Metric Question

**Context:** The T3 baseline evaluation matches MViTv2-S performance on verb-grouped 69-class activity recognition, achieving top1_69=0.6223 (`SOTA_STATUS.md:13`). This is presented as matching MViTv2-S's published top-1 accuracy (65.25%) when the class taxonomy is appropriately remapped from 75 fine-grained classes to 69 verb-grouped classes (`AAIML/industreal-sota-benchmarks.md:18-24`). However, the comparison has multiple confounds: MViTv2-S uses 16-frame clip-level spatiotemporal processing with Kinetics-400 pretraining, while our model uses a per-frame MLP with random initialization. Our "clip-level" evaluation for this comparison computes a 16-frame majority vote over per-frame MLP outputs — not genuine clip-level processing (`SOTA_STATUS.md:11-13`). Our actual clip-level top-1 without majority voting is 0.028, confirming the per-frame MLP cannot do temporal reasoning (`SOTA_STATUS.md:12`). The comparability matrix rates activity as "NOT COMPARABLE (Category Error)" and the task has been renamed to "per-frame action classification" everywhere in the docs (`AAIML/comparability-matrix.md:25-37`).

**Question:** Is the T3 baseline match (top1_69=0.622) a legitimate sanity check showing the verb-grouping protocol reproduces published numbers under a remapped evaluation, or is it misleading to present it as "matching SOTA" when the underlying architecture (per-frame MLP) and evaluation (16-frame majority vote) differ from MViTv2-S's genuine clip-level spatiotemporal processing? Specifically, would a reviewer accept "Our T3 baseline matches MViTv2-S under verb-grouped remapping" as sufficient for stating our activity head is competitive with published activity SOTA on IndustReal?

**Why this matters:** The contribution audit downgraded verb-grouping as the smallest confirmed contribution (`AAIML/contribution-audit-reviewer-factcheck.md:19` — Claim 9). Presenting the T3 baseline match as activity SOTA comparison reopens the comparability problem the audit conclusively closed. The paper's honesty hinges on whether "verifying our evaluation protocol reproduces published numbers" is presented as exactly that — a protocol verification — not as an activity recognition SOTA claim.

**Evidence:**
- `SOTA_STATUS.md:11-13` — clip-level top1=0.028, T3 baseline top1_69=0.6223, per-frame MLP cannot do temporal
- `AAIML/comparability-matrix.md:25-37` — NOT COMPARABLE, category error, renamed to per-frame classification
- `AAIML/industreal-sota-benchmarks.md:18-24` — MViTv2-S: 65.25% top-1 on 75-class, Kinetics pretrain
- `AAIML/contribution-audit-reviewer-factcheck.md:19` — Claim 9: verb-grouping is smallest contribution, breaks comparability
- `AAIML/industreal-sota-benchmarks.md:153` — activity comparison: "Different class taxonomy (75 vs 69 verb-grouped)"

**Evidence missing:** An actual MViTv2-S evaluation under verb-grouped 69-class protocol on our exact split. Without this, the "match" claim relies on the assumption that remapping preserves relative performance, which is unverified.

---

## Q5. POS Paradox — The MonotonicDecoder Artifact Question

**Context:** Our PSR achieves POS=0.968, exceeding STORM-PSR's 0.812 and B3's 0.797 (`AAIML/industreal-all-papers-benchmarks.md:112-113`). The comparability matrix explicitly flags this as a metric artifact: "Our POS is higher but it's a metric artifact from the MonotonicDecoder fill-forward constraint" (`AAIML/comparability-matrix.md:52`). The contribution audit specifies "fill-forward gaming, psr_pos weakness" confirmed as a verified phenomenon (`AAIML/contribution-audit-reviewer-factcheck.md:16` — Claim 6). The D4 experiment confirms the paradox structurally: YOLOv8m -> MonotonicDecoder yields F1=0.000 with POS=0.999, because a sparse-detection decoder trivially matches an "almost always empty" ground truth under the fill-forward constraint (`SOTA_STATUS.md:46-47`). The reviewer reconciliation doc devotes an entire section to explaining the POS paradox (`AAIML/reviewer-3-psr-paradigm-reconciliation.md:22-24`).

**Question:** Should POS be reported at all in the paper, given that every available piece of evidence confirms it is inflated by the MonotonicDecoder's architectural properties rather than genuine procedural understanding? If we must report it, what disclosure text is sufficient to prevent a reviewer from concluding we are deliberately gaming the metric? And is there any version of POS — perhaps POS@tolerance where transitions must occur within a temporal window — that would be a meaningful and honest measure of procedural ordering capability?

**Why this matters:** A single number (POS=0.968, "beating SOTA by 19%") is the most eye-catching claim in the PSR section. If presented without disclosure, it will be the first thing flagged by any reviewer familiar with the B3/STORM-PSR baselines. The paper's credibility depends on how transparently this is handled.

**Evidence:**
- `AAIML/comparability-matrix.md:52` — POS artifact from MonotonicDecoder fill-forward
- `AAIML/contribution-audit-reviewer-factcheck.md:16` — Claim 6: fill-forward gaming confirmed
- `SOTA_STATUS.md:46-47` — POS paradox structural: sparse detection yields POS=0.999 with F1=0
- `AAIML/reviewer-3-psr-paradigm-reconciliation.md:22-24` — POS paradox section
- `AAIML/industreal-all-papers-benchmarks.md:113` — Our POS=0.968 vs STORM-PSR=0.812 (+19%)

**Evidence missing:** A tolerance-based POS metric where the ordering score is computed only within windows around ground truth transitions. This would strip the fill-forward artifact and measure genuine ordering capability. This is the experiment that would determine whether POS has any informational content worth reporting.

---

## Q6. Per-Frame vs Clip-Level Activity — The Temporal Reasoning Question

**Context:** Our activity head achieves top-5=0.398 and macro-F1=0.110 as a per-frame MLP with zero temporal context, random initialization, and verb-grouped 69 classes (`AAIML/comparability-matrix.md:27-36`). MViTv2-S achieves 65.25% top-1 with 16-frame clip-level spatiotemporal processing, Kinetics-400 pretraining, and 75 fine-grained classes (`AAIML/industreal-sota-benchmarks.md:19`). The comparability matrix states these are "Fundamentally different tasks" and the community has already renamed our task to "per-frame action classification" in all documentation (`AAIML/comparability-matrix.md:34-37`). At epoch 18, our per-frame MLP hits a ceiling: the SOTA_STATUS explicitly states "ConvNeXt-Tiny + per-frame MLP hits ceiling on activity (needs video-level architecture)" (`SOTA_STATUS.md:45`). The activity T3 baseline (top1_69=0.622) was achieved via a 16-frame majority vote over per-frame outputs — not genuine temporal processing.

**Question:** Is there any defensible way to present the activity head's performance alongside published activity SOTA numbers, given that every dimension of comparison (temporal processing, class count, pretraining, metric) differs? Or should the paper make a clean break: rename to "per-frame action classification," report metrics in isolation, and explicitly state that no comparison to MViTv2-S is intended or possible without an architectural change to video-level processing?

**Why this matters:** The temptation will be to include a comparison table showing "Activity: 65.25% (MViTv2-S) vs 0.622 (ours, verb-grouped)" because the numbers are close enough to look good in a table. But the contribution audit and comparability matrix both conclude this comparison is indefensible. Including it — even with footnotes — invites the reviewer to question the entire paper's benchmarking rigor. This question asks whether the cost of including the comparison (reputation damage from hostile review) outweighs any benefit.

**Evidence:**
- `AAIML/comparability-matrix.md:25-37` — category error: per-frame vs clip-level, different classes, metrics, pretraining
- `SOTA_STATUS.md:45` — "hits ceiling on activity, needs video-level architecture"
- `SOTA_STATUS.md:11-13` — genuine clip-level top1=0.028, T3 baseline (majority vote)=0.622
- `AAIML/industreal-sota-benchmarks.md:18-24` — MViTv2-S: 65.25% top-1, 75 classes, Kinetics
- `AAIML/industreal-sota-benchmarks.md:153` — comparison row: "Different class taxonomy"

**Evidence missing:** A temporal architecture comparison (e.g., adding 16-frame video-level head to ConvNeXt-Tiny backbone). The SOTA_STATUS says this is needed but hasn't been done. Without it, the comparison gap is architectural, not incremental.

---

## Q7. Ego-Pose Contribution — The OpenFace/6DRepNet Distinction Question

**Context:** Our head pose head achieves forward MAE=8.14-9.14 degrees and up MAE=7.06-7.48 degrees, establishing the first reported ego-pose baseline on IndustReal (`AAIML/comparability-matrix.md:58-67`, `AAIML/industreal-sota-benchmarks.md:120-129`, `SOTA_STATUS.md:14-15`). The contribution audit explicitly states: "pose.csv is the WEARER's head pose: this is egocentric ego-pose regression, NOT face head-pose estimation. OpenFace/6DRepNet comparisons (docs 98/106) are category errors — remove" (`AAIML/contribution-audit-reviewer-factcheck.md:11`). The benchmark reference further clarifies: "MediaPipe is a dedicated face tracker. We predict 9-DoF ego-pose (wearer's head orientation from HoloLens) from a single RGB frame — this is NOT comparable to face-based head pose estimators" (`AAIML/benchmark-reference-for-paper.md:36-37`). Despite this, earlier documents referenced SOTA head pose numbers of ~15 degrees as a comparison target. The SOTA_STATUS shows "near SOTA" language with "~15 degrees" as the implicit target (`SOTA_STATUS.md:14`).

**Question:** Given that the ~15 degree "SOTA" reference for head pose was found to be from unverifiable search snippets (`AAIML/benchmark-reference-for-paper.md:5-9,96-98`) — the Ohkawa "IndustReal head pose" paper appears not to exist on CVF — should the paper use "~15 degrees" as any kind of comparison anchor at all? Or does the absence of any verifiable prior head pose work on IndustReal mean the paper's cleanest position is "first reported baseline, no prior comparison," absolutely zero implied SOTA numbers?

**Why this matters:** Saying "near SOTA" or "approaching ~15 degrees" when the ~15 degree figure comes from unverifiable search snippets is an evidentiary risk. If the source paper doesn't exist, the number is fabricated-by-search-engine. The safer and more defensible position is pure "first baseline" without any implied comparison, which is also what the contribution audit recommends.

**Evidence:**
- `AAIML/benchmark-reference-for-paper.md:5-9` — Ohkawa paper NOT FOUND on CVF
- `AAIML/benchmark-reference-for-paper.md:96-98` — all head pose numbers from unverifiable snippets
- `AAIML/contribution-audit-reviewer-factcheck.md:11` — Claim 2: RECATEGORIZED as egocentric ego-pose, NOT face head-pose
- `SOTA_STATUS.md:14-15` — forward MAE=9.14, up MAE=26.20 (eval)/13.5 (300-subset)
- `AAIML/comparability-matrix.md:58-67` — FIRST BASELINE, no prior
- `AAIML/industreal-sota-benchmarks.md:120-129` — first ego-pose baseline on IndustReal

**Evidence missing:** The position values remain unreliable due to scale ambiguity (`HEAD_POSE_POS_SCALE=100` heuristic). Without resolving this, position-based claims are not reportable. The up MAE discrepancy (26.20 vs 13.5 on 300-subset) suggests evaluation-level issues that need resolution.

---

## Q8. The Efficiency/Accuracy Tradeoff — What the Paper Should Lead With

**Context:** The competitor analysis positions our system as "a unified, affordable alternative to the multi-model approach, not a replacement for specialist models" (`AAIML/08_COMPETITOR_ANALYSIS.md:168-169`). Multi-task efficiency claims include ~67% fewer params (28M vs estimated 112M for 4 single-task models) and ~75% cheaper hardware ($429 vs ~$1,716) (`AAIML/comparability-matrix.md:71-81`). However, the contribution audit states the "$299 GPU" claim in prior docs is false: "5060 Ti 16GB ($429 MSRP; $299 = 3060)" (`AAIML/contribution-audit-reviewer-factcheck.md:13`). The comparability matrix further clarifies: "Parameter savings are real (~67%, not 31% as previously claimed). But need Ablation A numbers for the single-task baseline" (`AAIML/comparability-matrix.md:82-83`). The detection head now beats SOTA (mAP50=0.995 at epoch 18), PSR is competitive (F1=0.7018), and ego-pose is first baseline — but activity is not comparable.

**Question:** Given that (1) detection now clearly beats YOLOv8m, (2) PSR is competitive under the correct paradigm frame, (3) ego-pose is first baseline, (4) activity is not comparable, and (5) efficiency savings need Ablation A to verify — what is the single honest headline for the paper's SOTA positioning? Specifically, should the paper lead with "we beat SOTA on detection and establish the first ego-pose baseline, while PSR is competitive under per-frame component recognition" or "our multi-task system matches the cost of a single specialist model while performing 4 tasks with competitive accuracy on 3 of them"?

**Why this matters:** The headline determines what the paper is "about." The first version positions POPW as a SOTA-contender on specific tasks; the second positions it as a systems-efficiency contribution. These imply different review criteria (accuracy benchmarks vs systems novelty) and different reviewer expertise. The wrong choice invites a reviewer mismatch.

**Evidence:**
- `SOTA_STATUS.md:9` — detection mAP50=0.995 beats 0.838 (YOLOv8m)
- `SOTA_STATUS.md:16-17` — PSR F1=0.7217/0.7018 near STORM-PSR 0.901 (different paradigm)
- `SOTA_STATUS.md:14` — ego-pose forward MAE=9.14 (first baseline)
- `AAIML/comparability-matrix.md:71-81` — efficiency: ~67% fewer params, ~75% cheaper
- `AAIML/contribution-audit-reviewer-factcheck.md:13` — Claim 3: "$299 GPU" corrected to $429
- `AAIML/08_COMPETITOR_ANALYSIS.md:168-169` — competitive positioning: unified affordable alternative
- `AAIML/comparability-matrix.md:82-83` — need Ablation A before efficiency claims

**Evidence missing:** Ablation A (single-task baselines on same backbone) has not been run. Without it, efficiency savings (parameter count, inference cost, GPU cost) are estimates, not measurements. The paper cannot claim "67% fewer params" with a verified number until this experiment completes.
