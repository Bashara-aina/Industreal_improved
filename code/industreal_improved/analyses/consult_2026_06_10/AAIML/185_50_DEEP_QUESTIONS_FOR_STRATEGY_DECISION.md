# 185 — 50 Deep Questions for Strategy Decision (Opus Consultation Round 2)

**Date:** 2026-07-09
**Goal:** Reach ≥80% of SOTA on each head (det ≥0.67, act ≥0.52, PSR ≥0.72, pose ≤12°).
**Companion to:** 182 (strategy), 183 (architecture), 184 (training & data).
**Format:** Each question follows the `(a) question / (b) why it matters / (c) evidence we have / (d) evidence missing / (e) what the answer changes` pattern used in Opus 181 / file 178.

---

## SECTION A — Strategy & Direction (A-1 to A-8)

### A-1 — Is MTL itself the goal, or is "MTL as good as single-task" the goal?
**(a)** Are we optimizing for one MTL model that reaches near-SOTA, or a comparison "MTL vs 4 ST" where MTL can be 70% of ST?  
**(b)** The framing determines how much engineering we invest in making MTL work vs accepting it underperforms.  
**(c)** The original docs say "MTL provides positive transfer" and "MTL hypothesis." Opus 181 reframed this as L2 (positive transfer on ≥1 task) + L3 (efficiency).  
**(d)** No explicit user statement of which framing.  
**(e)** Determines whether Strat-1 (incremental, ~60% SOTA) is acceptable or only Strat-4 (sequential, ~80%+ SOTA) is acceptable.

### A-2 — What is the actual AAIML submission deadline?
**(a)** When is the paper due?  
**(b)** Determines how much compute we can spend on risky strategies (Strat-4 takes 7-10 days; Strat-2 takes 5-10).  
**(c)** Not stated in any AAIML doc I can see.  
**(d)** Submission deadline, conference date, camera-ready date.  
**(e)** If deadline is < 2 weeks, Strat-1 is the only realistic option. If 4-6 weeks, all strategies feasible.

### A-3 — Is the user willing to abandon MTL and report single-task results?
**(a)** If all MTL strategies fail, do we report 4 ST models as "the system" and frame MTL as a finding?  
**(b)** This is the "fallback" path (Strat-5).  
**(c)** User said "MTL even we can get more efficient model, faster training, with more accurate results across all heads" — MTL is the goal.  
**(d)** No explicit statement on abandonment threshold.  
**(e)** Determines whether to invest in Strat-5's contingencies (4 ST scripts as fallback).

### A-4 — Is the "MTL cost" measurement (64% of ceiling) sufficient for AAIML?
**(a)** Does the AAIML community accept "MTL reaches 64% of single-task ceiling with 3× fewer params" as a contribution?  
**(b)** AAIML favors efficiency + practical applicability. The 64% number is honest.  
**(c)** No community survey data.  
**(d)** Acceptance criteria of AAIML reviewers.  
**(e)** If yes, we can ship with Strat-1 results even if absolute metrics are below 80% SOTA. If no, we must hit absolute 80%.

### A-5 — Should we add a 5th task (e.g., anticipation, skill assessment)?
**(a)** Does adding a 5th task make MTL more compelling (more "multi-task") or just spread features thinner?  
**(b)** More tasks may dilute per-task performance. Adding anticipation (predicting next state) could be useful.  
**(c)** Existing model has 4 heads; adding a 5th requires new data labels, new head, new loss.  
**(d)** Availability of labels for additional tasks.  
**(e)** If a 5th task helps, we get a stronger MTL story. If it dilutes, we should stick with 4.

### A-6 — Is the pose target (≤12° MAE) reasonable for AAIML?
**(a)** Pose SOTA is ~15° (uncited, per Opus 132 HP-1). 80% of "15°" could mean 12° (better) or 18° (worse) — which does the user want?  
**(b)** Pose is the only working head. If we already meet 12°, the "80% of SOTA" interpretation matters less.  
**(c)** Current pose MAE is 10° (ep5 eval).  
**(d)** Definitive SOTA pose number.  
**(e)** Probably no impact — pose is fine.

### A-7 — Should the paper lead with the MTL model or the MTL pathology?
**(a)** Lead option: "MTL via foundation adapters" (positive story). Alt: "Diagnosing MTL optimization pathologies" (Opus 181's methodological contribution).  
**(b)** The methodology angle is more reviewer-proof but less novel.  
**(c)** Both stories are viable; we have data for both.  
**(d)** Which story AAIML prefers.  
**(e)** Determines paper structure (positive transfer section vs pathology section).

### A-8 — How important is the "single GPU" constraint?
**(a)** Can we run 4 single-task baselines in parallel on 2 GPUs, or must they be sequential?  
**(b)** Affects the wall-clock for Strat-4 (parallel: 5 days; sequential: 10 days).  
**(c)** User said "2 GPUs, 1 primary." Implies we can use the 2nd for parallel work.  
**(d)** Actual utilization of the 2nd GPU.  
**(e)** If parallel works, Strat-4 fits in 5 days. If sequential, 10+ days.

---

## SECTION B — Backbone Choice (B-1 to B-8)

### B-1 — Is InternVideo2-L's video pretraining strictly better than EVA-02-L's image pretraining for assembly activity?
**(a)** Assembly is fine-grained spatial discrimination (which component is at state X) plus temporal ordering (transitioning from X to Y). Which backbone wins on each axis?  
**(b)** Determines Strat-2's backbone choice.  
**(c)** Both are at ~300M params. Pretraining data is video (InternVideo2) vs image (EVA-02).  
**(d)** No empirical data on IndustReal.  
**(e)** If InternVideo2 wins, use it. If EVA-02 wins, use that. If they're tied, pick by license (EVA-02 MIT is safer than InternVideo2 Apache for industrial applications).

### B-2 — Is DINOv2-L competitive with EVA-02 / InternVideo2 for activity classification?
**(a)** DINOv2 has the strongest general features on dense tasks (depth, segmentation). What about classification?  
**(b)** DINOv2 is Apache 2.0 (very permissive). If competitive, it's the safest choice.  
**(c)** DINOv2 benchmarks: 86.5% linear-probe ImageNet. EVA-02: 89.5%. InternVideo2: depends on benchmark.  
**(d)** Activity classification numbers on Kinetics-400 or similar.  
**(e)** If DINOv2 is competitive, license-friendliness makes it the default.

### B-3 — Does scaling up MViTv2 (L or H) close the gap to a foundation model?
**(a)** MViTv2-L (53M) or MViTv2-H (99M) might give us the representational capacity without licensing concerns.  
**(b)** If yes, we avoid the licensing risk entirely.  
**(c)** MViTv2-S is 34M. Doubling to 68M via L is well-trodden.  
**(d)** No Kinetics-400 numbers for MViTv2-L for assembly tasks.  
**(e)** If MViTv2-L gets us within 10% of InternVideo2, we save licensing risk.

### B-4 — Should we use a video-language model like VideoCLIP or LanguageBind?
**(a)** These have both visual and language features; assembly labels could be embedded in language space.  
**(b)** Activity labels are explicit text (e.g., "assembling 10000000000"); a video-language model could align visual features to label text directly.  
**(c)** VideoCLIP and LanguageBind exist; both have video + text alignment.  
**(d)** No benchmark on assembly-style fine-grained actions.  
**(e)** Could be a wild card; potentially very strong for activity.

### B-5 — What if we just use YOLOv8m as the detection backbone (no MViTv2)?
**(a)** YOLOv8m pretrained on COCO has strong object features. Transfer to assembly might be easier than MViTv2 → assembly.  
**(b)** Detection alone might dominate; the other tasks use a separate backbone.  
**(c)** Opus 132 reports single-task YOLOv8m reaches mAP 0.995 (separate backbone).  
**(d)** Whether other tasks (activity, PSR, pose) can share a backbone with YOLOv8m.  
**(e)** If yes, this is the easiest path for detection. If no, we'd need a separate backbone per task, defeating MTL.

### B-6 — Is the depth-of-features argument for PSR (block-3 vs conv_proj) empirically validated?
**(a)** Opus 181 §2.3 says PSR fails because features are from block-1. Does this generalize?  
**(b)** If true, switching to block-3 features is a 1-line fix that should give significant improvement.  
**(c)** No empirical data on block-3 PSR features yet.  
**(d)** A 1-epoch ablation: train PSR with block-1, block-2, block-3 features. Measure F1.  
**(e)** If block-3 works, that's our lowest-hanging PSR fix.

### B-7 — Can we use a frozen image backbone (DINOv2, EVA-02) with a learned temporal transformer?
**(a)** Image backbones are stronger per-frame than video backbones (which sacrifice image quality for temporal modeling). A learned temporal transformer on top might be the best of both worlds.  
**(b)** This is Strat-2 with an image backbone + temporal head.  
**(c)** Standard recipe in video understanding (e.g., TimeSformer uses ViT + temporal attention).  
**(d)** How well temporal attention learned on top of frozen features works vs end-to-end video pretraining.  
**(e)** If competitive with InternVideo2, this is cheaper to train (smaller temporal module).

### B-8 — Is there a domain-specific pretrained model for industrial assembly?
**(a)** IndustReal / similar industrial datasets may have published pretraining checkpoints.  
**(b)** Would be the strongest transfer signal.  
**(c)** IndustReal itself is the dataset; no public pretraining I know of.  
**(d)** Whether external assembly-pretrained models exist.  
**(e)** Unlikely to exist; if it does, very high value.

---

## SECTION C — Per-Task Head Capacity (C-1 to C-8)

### C-1 — Is a 2-layer MLP head on class-token features *ever* sufficient for 75-class video activity?
**(a)** Or do we need a temporal transformer regardless of backbone?  
**(b)** Determines activity head complexity.  
**(c)** Current 1-layer head fails (0.008 top-1).  
**(d)** Empirical data with 2-layer MLP on stronger backbone features.  
**(e)** If 2-layer is sufficient, Strat-1/2 work. If not, we need transformer head or temporal attention.

### C-2 — Does ArcFace (additive angular margin) help for 75-class long-tail?
**(a)** ArcFace is known to help face recognition (10K+ classes). Does it transfer to assembly activity?  
**(b)** Activity has 75 classes with long tail. ArcFace's angular margin might be perfect.  
**(c)** No empirical data on assembly + ArcFace.  
**(d)** A 1-day ablation: train activity with CE vs CE+ArcFace on current MViTv2-S.  
**(e)** If ArcFace gives +5-10% top-1, adopt it. Otherwise, use CE.

### C-3 — For PSR, does block-3 + T=32 input + 4-layer transformer reach 0.72 F1?
**(a)** These are the three PSR upgrades combined. Do they get us to 80% SOTA?  
**(b)** PSR is the largest gap. If these three upgrades don't close it, we need a different approach (e.g., InternVideo2 features).  
**(c)** No empirical data on this combination.  
**(d)** A 1-day ablation: train PSR with current config vs upgraded config. Compare F1.  
**(e)** Determines PSR strategy. If upgrades don't reach 0.72, need foundation-model features.

### C-4 — Does YOLOv8's TAL+DFL+CIoU+Focal loss converge on assembly detection in <30 epochs?
**(a)** If yes, adopt YOLOv8 head. If no, the assembly detection problem has unique characteristics that need a custom head.  
**(b)** Determines detection strategy.  
**(c)** TAL is widely used; should converge in standard cases.  
**(d)** No empirical data on our 24-class assembly detection with TAL.  
**(e)** If yes, simple adopt. If no, we need to tune TAL hyperparameters or design a custom assignment.

### C-5 — Is 24-class detection fundamentally harder than COCO 80-class?
**(a)** COCO has more classes but with more data per class (118K images). Our 24-class is fewer but smaller dataset (~6K labeled frames).  
**(b)** Affects how much data augmentation we need.  
**(c)** No head-to-head comparison on similar-size datasets.  
**(d)** Empirical: how does YOLOv8 perform on a 24-class subset of COCO with our dataset size?  
**(e)** If similar, our setup is reasonable. If much harder, we need more aggressive regularization.

### C-6 — For pose, is geodesic loss on rotation matrices meaningfully better than cosine on 6D?
**(a)** Cosine on 6D vectors works (current 10° MAE). Geodesic is theoretically more correct.  
**(b)** Cosine is simpler; geodesic requires matrix operations.  
**(c)** Pose is the working head; not worth breaking.  
**(d)** A 1-day ablation: train pose with cosine vs geodesic.  
**(e)** Probably no change. But if geodesic gives 1-2° improvement, easy win.

### C-7 — Does adding a 5th head (e.g., depth estimation, optical flow) help the other 4?
**(a)** Auxiliary tasks can serve as regularization.  
**(b)** If yes, MTL becomes more compelling.  
**(c)** Standard in multi-task learning literature.  
**(d)** Whether we have depth/flow labels.  
**(e)** Unlikely we have the labels; skip this path.

### C-8 — Should each head have its own adapter, or share an adapter across all heads?
**(a)** Per-task adapters (4 separate) vs shared adapter (1, larger).  
**(b)** Per-task gives specialization; shared gives parameter efficiency.  
**(c)** LoRA standard is per-task.  
**(d)** Empirical: per-task vs shared ablation.  
**(e)** Probably per-task; cheap to implement.

---

## SECTION D — MTL Topology (D-1 to D-8)

### D-1 — Does MMoE (multi-gate mixture-of-experts) actually help on this codebase?
**(a)** MMoE adds experts + per-task gates; standard MTL improvement.  
**(b)** Might give 5-10% improvement on weakest tasks.  
**(c)** Standard recipe; well-tested on NLP/vision.  
**(d)** Empirical: current shared vs MMoE (8 experts, 4 gates).  
**(e)** If helps, ~2 days engineering. If doesn't, try PLE or cross-task attention.

### D-2 — Does cross-task attention (task tokens query backbone) help when each head already has its own adapter?
**(a)** Cross-task attention adds dynamic task-feature routing.  
**(b)** Might be redundant with per-task adapters.  
**(c)** Not standard; novel.  
**(d)** Empirical.  
**(e)** Higher risk; could help a lot or nothing.

### D-3 — Is model soup (averaging 4 task-specific backbones) a good initialization for MTL?
**(a)** Soup'd backbone is in the "average" basin; might be good for MTL.  
**(b)** Standard 2023 finding.  
**(c)** Wortsman et al. 2022, Ilharco et al. 2022 "task arithmetic."  
**(d)** Empirical: soup init vs random init for MTL.  
**(e)** If soup init helps, it's basically free (just average weights).

### D-4 — Should the MTL finetune use a lower LR than single-task pretraining?
**(a)** Lower LR preserves the soup'd initialization.  
**(b)** Standard recipe for finetuning.  
**(c)** Yes, 5-10× lower LR is typical.  
**(d)** What LR works for our setup.  
**(e)** Standard; ~5e-5 backbone, 5e-4 heads.

### D-5 — Is the cross-task attention topology stable to train?
**(a)** Task tokens can collapse to similar values.  
**(b)** Determines if cross-task attention is worth the engineering.  
**(c)** Known issue in cross-attention literature.  
**(d)** Need to monitor task token diversity during training.  
**(e)** If unstable, skip cross-task attention.

### D-6 — For Strat-2 (frozen backbone + adapters), do we need the MTL topology at all?
**(a)** If backbone is frozen, each head sees fixed features. Topology is just head wiring.  
**(b)** MMoE / cross-task attention matter less when backbone doesn't update.  
**(c)** Standard practice: frozen backbone + simple heads.  
**(d)** Empirical.  
**(e)** Likely just simple heads; topology is overhead.

### D-7 — Should we share the BN/LayerNorm parameters across tasks in MTL?
**(a)** BN/LN are cheap; sharing is parameter-efficient.  
**(b)** Standard in MTL.  
**(c)** Current model shares backbone BN/LN.  
**(d)** Whether per-task BN helps.  
**(e)** Sharing is fine; per-task BN rarely helps.

### D-8 — Is PCGrad still needed after Path-D fixes?
**(a)** PCGrad resolves per-step gradient conflicts. After Path-D (EMA + caps), is it still doing useful work?  
**(b)** PCGrad adds 30% per-step overhead. If unnecessary, drop it.  
**(c)** Opus 181 §6 K-5/P-3/P-4: "PCGrad is not the thing starving it (Kendall is). Keep PCGrad for now; E8 tells you if it's a no-op."  
**(d)** Empirical: PCGrad on vs off after Path-D.  
**(e)** If PCGrad is no-op, drop it for ~30% speedup.

---

## SECTION E — Training Recipe (E-1 to E-10)

### E-1 — Does MixUp help long-tail 75-class activity?
**(a)** MixUp interpolates samples and labels. For long-tail, this might hurt rare classes (their labels get diluted).  
**(b)** Determines augmentation strategy.  
**(c)** Standard augmentation; usually helps but not always.  
**(d)** Empirical on assembly activity.  
**(e)** If MixUp hurts, use class-aware MixUp (only mix within same class).

### E-2 — Is RandAugment appropriate for video data?
**(a)** RandAugment was designed for images. Video has temporal structure.  
**(b)** Standard but worth verifying.  
**(c)** Often used for video with caution.  
**(d)** Empirical.  
**(e)** Use lighter RandAugment (N=1, M=5) for video, or video-specific augmentation.

### E-3 — Should we use EMA model weights (Polyak averaging)?
**(a)** Maintains a moving average of model weights; often improves final performance.  
**(b)** Standard technique.  
**(c)** Currently not used.  
**(d)** Empirical: with vs without EMA weights.  
**(e)** Cheap to add; if helps, +1-2% across the board.

### E-4 — Is Lion optimizer better than AdamW for this codebase?
**(a)** Lion is sign-based; faster convergence on transformers.  
**(b)** Might save 20-30% training time.  
**(c)** Recently published (2023); not yet standard.  
**(d)** Empirical.  
**(e)** Cheap to test; if helps, adopt.

### E-5 — Should we use a longer warmup for Strat-2 (frozen backbone)?
**(a)** Frozen backbone means large initial gradients on adapters + heads.  
**(b)** Standard recipe for transfer learning: 5-10 epoch warmup.  
**(c)** Standard practice.  
**(d)** Empirical.  
**(e)** Use 10-epoch warmup for Strat-2.

### E-6 — Does gradient clipping at norm=1.0 hurt for our setup?
**(a)** Default is 1.0; standard practice uses 5.0 for transformers.  
**(b)** Might be too aggressive; cutting useful gradient magnitude.  
**(c)** Current setting is 1.0.  
**(d)** Empirical: norm=1.0 vs norm=5.0.  
**(e)** If higher norm helps, change default.

### E-7 — Is the 4000-batch cap per epoch the right data coverage?
**(a)** Cap is 10% of full epoch. Less coverage → fewer effective updates per epoch.  
**(b)** Affects convergence rate.  
**(c)** Cap is set for time budget.  
**(d)** Empirical: 4000 vs 8000 vs full (39195) batches/epoch.  
**(e)** If we can afford 8000, do it. If not, 4000 stays.

### E-8 — Does cosine warm restart help compared to single cosine?
**(a)** Warm restart at epoch 50 might help escape local minima.  
**(b)** Standard technique.  
**(c)** Currently single cosine.  
**(d)** Empirical.  
**(e)** Cheap to add; if helps, adopt.

### E-9 — Is bf16 AMP still working correctly with our changes?
**(a)** Path-D changes affect backward pass. Need to verify bf16 still works.  
**(b)** Already smoke-tested on CPU. Need GPU verification.  
**(c)** Smoke test passed on CPU.  
**(d)** GPU run is in progress (PID 2248854).  
**(e)** Check first epoch for any bf16 NaN/Inf issues.

### E-10 — Should we add label smoothing to PSR (currently only activity has it)?
**(a)** PSR has 11 components; labels are 0/1. Label smoothing = 0.05 might help.  
**(b)** Standard for classification; less standard for per-frame BCE.  
**(c)** Currently no label smoothing for PSR.  
**(d)** Empirical.  
**(e)** If helps, easy add.

---

## SECTION F — Compute & Timeline (F-1 to F-6)

### F-1 — Can we run 2 single-task pretrainings in parallel on the 2nd GPU?
**(a)** User has 2 GPUs (1 primary, 1 secondary). Parallel work would halve the wall-clock for Strat-3/4.  
**(b)** Determines if we can finish in 1-2 weeks.  
**(c)** Stated in docs.  
**(d)** Actual VRAM and compute capacity of the 2nd GPU.  
**(e)** If yes, Strat-4 fits in ~5 days wall-clock.

### F-2 — What is the maximum batch size for InternVideo2-L on a 24GB GPU?
**(a)** Affects training throughput.  
**(b)** Bigger batch = faster training but more VRAM.  
**(c)** Unknown for InternVideo2-L on our setup.  
**(d)** Empirical measurement needed.  
**(e)** If batch=4 fits, great. If only batch=1, need more accumulation.

### F-3 — How long does the current Path-D run need to reach ep30 for a meaningful projection?
**(a)** Opus 181's projections are ep30. Current rate is 22 min/epoch → ~11 hours.  
**(b)** Determines when we have data to decide.  
**(c)** Run started 17:09:55 today.  
**(d)** No empirical confirmation yet.  
**(e)** Wait ~12 hours for ep30 projection.

### F-4 — Is 8000-batch-per epoch feasible without increasing wall-clock?
**(a)** Currently 4000 batches = 22 min. 8000 = 44 min/epoch = ~73 hours for 100 epochs.  
**(b)** Doubles training time.  
**(c)** Compute budget unclear.  
**(d)** Wall-clock constraints.  
**(e)** If time allows, do 8000. If not, stick with 4000.

### F-5 — How much does each Strat-2 experiment cost (single-task baseline)?
**(a)** Single-task activity on InternVideo2-L: 1-2 GPU-days?  
**(b)** Determines the cost of the "try foundation models" decision.  
**(c)** Unknown without testing.  
**(d)** Empirical.  
**(e)** If 1 GPU-day, try it. If 5 GPU-days, deprioritize.

### F-6 — Should we run multiple Strat-2 candidates in parallel (InternVideo2 vs EVA-02 vs DINOv2)?
**(a)** Parallel backbones on 2 GPUs would give us data on which is best within 1-2 days.  
**(b)** Higher cost but more informed decision.  
**(c)** Unknown.  
**(d)** Empirical.  
**(e)** Yes, if compute allows.

---

## SECTION G — Risk & Robustness (G-1 to G-6)

### G-1 — What's the failure mode if Strat-2's foundation-model adapter approach underperforms?
**(a)** If activity only reaches 0.30 (vs target 0.52), do we still have a paper?  
**(b)** Determines whether to invest in fallbacks.  
**(c)** Unknown.  
**(d)** Empirical.  
**(e)** If yes, we need Strat-4 as a parallel hedge.

### G-2 — Is the 80% SOTA bar realistic for PSR given the data scarcity?
**(a)** PSR has sparse positive labels (~5% of frames have transitions). 80% F1 may be intrinsically hard.  
**(b)** Determines if PSR is the rate-limiting task.  
**(c)** PSR F1=0.0 currently.  
**(d)** Empirical with better features.  
**(e)** If PSR is intrinsically hard, paper narrative focuses on the other 3 tasks.

### G-3 — What if the user's tolerance for "MTL not beating single-task" is actually 80%, not 100%?
**(a)** The user said "at least 80% MINIMUM." If MTL reaches 80% of single-task ceiling, is that the goal?  
**(b)** This is the L2/L3 framing from Opus 181.  
**(c)** Ambiguous.  
**(d)** User clarification.  
**(e)** If 80% of ST is acceptable, we have more flexibility in Strat-1 results.

### G-4 — Does InternVideo2-L have licensing restrictions for industrial use?
**(a)** Apache 2.0 is permissive; should be OK.  
**(b)** Determines if InternVideo2 is viable for an industry-relevant paper.  
**(c)** InternVideo2 is Apache 2.0; commercial use allowed.  
**(d)** Verify with Apache 2.0 license text.  
**(e)** If confirmed, proceed.

### G-5 — What's the downside if our path-D run fails to improve over current ep6 metrics by ep30?
**(a)** If activity is still at 0.05 by ep30, the optimization fixes aren't enough.  
**(b)** Determines whether to switch strategies mid-run.  
**(c)** Run in progress.  
**(d)** Empirical.  
**(e)** If ep30 doesn't improve, switch to Strat-1 (incremental architecture upgrades).

### G-6 — Are there any known issues with training MViTv2-S on bf16 that might cause divergence?
**(a)** bf16 has 7 bits of mantissa; some gradients might underflow.  
**(b)** Path-D run is on bf16; need to watch for divergence.  
**(c)** Smoke test passed.  
**(d)** Empirical.  
**(e)** Watch for any NaN/Inf during training.

---

## SECTION H — Validation & Metrics (H-1 to H-6)

### H-1 — Is the validation eval protocol consistent with the SOTA numbers (0.838, 0.652, 0.901)?
**(a)** If our eval differs from SOTA's eval, comparisons are invalid.  
**(b)** Critical for paper credibility.  
**(c)** Our eval uses NMS at IoU=0.5, score_thresh=0.001. Need to verify SOTA's protocol.  
**(d)** SOTA's exact eval protocol (file 180 may have this).  
**(e)** If different, we need to align.

### H-2 — Is the test split truly held out, or are there leakages from train?
**(a)** Test contamination → inflated metrics → not SOTA-comparable.  
**(b)** Standard concern.  
**(c)** Doc 175 §7.1 mentions split discipline.  
**(d)** Verify no recording-id overlap between train/val/test.  
**(e)** If contamination exists, must redo.

### H-3 — Does "activity top-1" measure what we want?
**(a)** Top-1 on 75 classes is a strict metric. Is it consistent with SOTA's measurement?  
**(b)** Affects comparison validity.  
**(c)** File 180 reports top-1.  
**(d)** SOTA's exact metric.  
**(e)** If SOTA uses top-5 or per-frame top-1, ours differs.

### H-4 — Are 80% SOTA bars adjusted correctly for pose (lower is better)?
**(a)** SOTA 15° MAE. 80% could be MAE ≤ 12° (we already achieve this) or MAE ≤ 18.75° (much easier).  
**(b)** Pose is the only "lower is better" metric.  
**(c)** File 180 says SOTA ~15°.  
**(d)** Definitive interpretation.  
**(e)** Pose is fine either way.

### H-5 — Should we report SOTA ratios (MTL/SOTA) or absolute deltas?
**(a)** Ratios are more interpretable; deltas are more concrete.  
**(b)** Standard in MTL papers.  
**(c)** Both used in literature.  
**(d)** No clear preference.  
**(e)** Use ratios for the main table, deltas for supplementary.

### H-6 — How do we measure "MTL provides positive transfer"?
**(a)** Need a quantitative metric for transfer.  
**(b)** Standard: compare MTL performance to single-task on same backbone.  
**(c)** Need single-task baselines.  
**(d)** Empirical (run ST baselines).  
**(e)** Standard.

---

## SECTION I — Alternative Paths & Wildcards (I-1 to Hmm, J — let me re-number)

(Replacing I-1 to J-2 to keep 50 total)

### I-1 — Could a video-language model (VideoCLIP, LanguageBind) bypass the activity head entirely?
**(a)** These models have a "classify by text similarity" capability. Could classify 75 assembly activities by name.  
**(b)** Wild card for activity.  
**(c)** VideoCLIP / LanguageBind exist; assembly classes could be defined as text.  
**(d)** No empirical test on assembly.  
**(e)** If works, huge simplification (no head needed).

### I-2 — Could we use a tracking-by-detection pipeline for activity?
**(a)** Detect components per frame, then track and infer assembly state from the sequence.  
**(b)** Modular approach; might work if detection is reliable.  
**(c)** Detection is the failing head; this would not work without fixing detection first.  
**(d)** Empirical.  
**(e)** Depends on detection working.

### I-3 — Should we use weakly supervised pretraining on unlabeled assembly videos?
**(a)** If we have unlabeled video, MAE / DINO-style pretraining could help.  
**(b)** Industrial video is abundant; unlabeled pretraining is cheap.  
**(c)** Unknown if unlabeled video is available.  
**(d)** Check dataset.  
**(e)** If yes, easy pretraining boost.

### I-4 — Should the paper focus on "MTL as a method contribution" rather than "MTL beats single-task"?
**(a)** Reframe MTL as a methodological advance (pathology + fix), not a competition.  
**(b)** Opus 181's Option 3.  
**(c)** Path-D's contribution is real and citable.  
**(d)** Which framing AAIML prefers.  
**(e)** Safer narrative; less risky.

### I-5 — Could we use zero-shot foundation models (CLIP, ALIGN) for activity directly?
**(a)** CLIP-style models can classify images by text. Could we use them as the activity head?  
**(b)** Direct zero-shot without training.  
**(c)** CLIP zero-shot on ImageNet is competitive.  
**(d)** Activity class names need to be defined as text. Assembly classes are state codes (10000000000); might not be CLIP-friendly.  
**(e)** Probably need to encode activity labels as text descriptions first.

### I-6 — Could we use a learned optimizer (e.g., L2L) instead of AdamW?
**(a)** L2L (Learning to Learn) learns the optimizer; might converge faster.  
**(b)** Niche; not standard.  
**(c)** Not standard practice.  
**(d)** Empirical.  
**(e)** Not worth the engineering.

### I-7 — What if we just use a much bigger batch size (32 or 64)?
**(a)** Larger batch often helps convergence for transformers.  
**(b)** Affects training stability.  
**(c)** Currently batch=2, accum=2 (effective 4).  
**(d)** Empirical: batch=4, 8, 16.  
**(e)** Standard ablations.

### I-8 — Should we use checkpoint averaging (SWA — Stochastic Weight Averaging)?
**(a)** Average weights from the last 5-10 epochs.  
**(b)** Standard; cheap improvement.  
**(c)** Not currently used.  
**(d)** Empirical.  
**(e)** Easy add if we save last 5 checkpoints.

### I-9 — Can we use the 4 single-task teachers as the basis for a final ensemble?
**(a)** Ensemble 4 single-task models (each at 0.85+) instead of one MTL model.  
**(b)** Different paper framing (ensemble vs MTL).  
**(c)** Standard practice.  
**(d)** Empirical: 4-ST ensemble vs 1-MTL.  
**(e)** If ST ensemble >> MTL, paper compares them.

### I-10 — Should we publish the dataset and code, or just the MTL approach?
**(a)** Open-sourcing the implementation helps the community.  
**(b)** Standard for AAIML.  
**(c)** Unclear policy.  
**(d)** User decision.  
**(e)** Likely yes.

---

## SECTION J — Paper Narrative (J-1 to J-4)

### J-1 — What is the most defensible paper title?
**(a)** Many options: "MTL for IndustReal", "Diagnosing MTL Optimization", "Foundation-Model Adapters for Video MTL", etc.  
**(b)** Sets reader expectations.  
**(c)** Depends on which Strat wins.  
**(d)** User preference.  
**(e)** TBD after experiments.

### J-2 — What's the headline figure for the paper?
**(a)** The 1 image that summarizes the contribution.  
**(b)** Most-cited part of the paper.  
**(c)** Could be: MTL architecture diagram, MTL vs ST bar chart, gradient conflict heatmap, loss trajectory.  
**(d)** Empirical.  
**(e)** TBD.

### J-3 — Which of the 4 heads is the paper's "hero" result?
**(a)** Which head's improvement is most compelling?  
**(b)** Affects abstract and introduction framing.  
**(c)** TBD after experiments.  
**(d)** User preference.  
**(e)** Pose is already good. PSR is most-improved if upgrades work. Activity is most-needed. Detection is most-defensible (YOLOv8 has track record).

### J-4 — Should the paper include a "negative result" section (Kendall paradox)?
**(a)** Documenting the pathology + fix is a methodological contribution.  
**(b)** Opus 181 §0: "the methodological finding is genuinely novel and is your most reviewer-proof angle."  
**(c)** Yes, include it.  
**(d)** Always.  
**(e)** Definitely include.

---

## FINAL — The 4 Strategic Questions for Opus

These are the 4 most decision-changing questions; the rest are supporting:

1. **Q-Backbone** (B-1, B-2, B-3): Which backbone is most likely to close the gap to 80% SOTA across all 4 heads? InternVideo2-L, EVA-02-L, DINOv2-L, or scale-up MViTv2-S?

2. **Q-Architecture** (C-1, C-3, D-3): Is the bottleneck per-head head capacity, MTL topology, or backbone features? Which single change has the highest leverage?

3. **Q-Strategy** (A-1, A-4, G-3): Is "MTL reaches 80% of SOTA" the right bar, or is "MTL reaches 80% of single-task ceiling" acceptable? This determines whether Strat-1 (incremental) is sufficient.

4. **Q-Compute** (F-1, F-4): Given 1 primary GPU, can we realistically run Strat-4 (7-10 GPU-days) in 2 weeks, or do we need Strat-1 only?

---

*50 questions across 9 sections. All questions are grounded in Opus 181 + files 176-180. The "evidence missing" column is where the actual experiments live. See file 182 for the strategic decision tree that consumes the answers.*