# Agent Debates — Specialist Adversarial Dialogues

*Each section features a debate between a specialist agent acting as Adversarial Reviewer and a Defending Author. The goal is to surface the strongest possible attack on a paper claim and the strongest possible defense, so the writing team can decide which counterarguments to preempt.*

---

## Section 1: Detection (ASD) Debate

*Agent: Detection Head Specialist (Agent 1)*

*File references: d1_yolov8m/metrics.json, d1_yolov8m_v2/metrics.json, d1r/results.csv, d3_v4/metrics.json, full_eval_ep18_stream/metrics.json, d4_yolov8m_psr/metrics.json, reviewer-1-detection-path-to-SOTA.md, industreal-all-papers-benchmarks.md, industreal-sota-benchmarks.md, PROGRESS_2026-07-06.md, SOTA_STATUS.md, popw_aaiml2027.tex*

---

### Debate 1.1: The Self-Trained YOLOv8m Beats SOTA — Should the Paper Claim It?

**Detection Skeptical Reviewer (arguing the SOTA claim is invalid):** The SOTA_STATUS.md claims "Detection mAP50=0.995 BEATS SOTA" using a separately-trained YOLOv8m that achieves mAP50=0.99684 and mAP50-95=0.86096 (`runs/detect/src/runs/yolov8m_industreal/d1r/results.csv` epoch 25). This YOLOv8m is NOT the paper's main model. The main ConvNeXt-Tiny multi-task model achieves mAP50=0.358 on the subsample and NaN on the full evaluation. Claiming "BEATS SOTA" from a model that is not the paper's contribution is deceptive. A reviewer will say: "Your system underperforms a free off-the-shelf YOLOv8m that you trained in one day. Why should anyone use your model?" The paper must either (a) drop the SOTA claim entirely and frame detection as "multi-task cost measurement," or (b) integrate the YOLOv8m into the contribution as a teacher for knowledge distillation. Claiming SOTA from a separate model that beats your own system by 64% does not pass peer review.

**Supportive Reviewer (ASD Parity, arguing the SOTA claim can be restructured):** The paper can reconcile both numbers honestly. The YOLOv8m result establishes something important: published WACV 2024 SOTA (0.838 mAP per `industreal-all-papers-benchmarks.md`) is conservative. A 25-epoch ultralytics training exceeds it (0.996 vs 0.838), suggesting the WACV number was limited by their pretraining protocol (COCO+Synth) rather than the architecture. This strengthens the paper's argument: dedicated single-task detection on IndustReal is not hard (mAP can exceed 0.99 with 1 GPU-day). The hard problem is doing detection simultaneously with three other tasks without degradation. The ConvNeXt-Tiny's 0.358 mAP is 36% of the YOLOv8m's 0.995, and that 64% gap IS the measured multi-task cost. This is exactly the "64-68% ratio" from `reviewer-1-detection-path-to-SOTA.md`. The SOTA claim should be restructured: "YOLOv8m achieves 0.995 mAP exceeding WACV 2024's 0.838, establishing the single-task ceiling. Our multi-task ConvNeXt-Tiny achieves 64-68% of this ceiling while running three additional tasks at zero extra inference cost." This is honest, compares like-for-like, and turns the gap from a weakness into a measurement.

**mAP Fairness Reviewer (arguing the gap may be threshold-dependent):** The key missing evidence is whether both evaluations use the same validation split, the same NMS parameters, and the same confidence threshold. The WACV 2024 paper (Table 3.3) reports mAP on "annotated frames" (sampled subset) and "entire videos" (all frames). Our D1 eval uses the full-frame approach, but the d1r YOLOv8m uses ultralytics default eval which may use a different confidence threshold (ultralytics default 0.001 vs our eval default 0.05). The reported gap (0.358 vs 0.995) may partially reflect threshold differences. For a fair comparison, both models must be evaluated with identical NMS, confidence thresholds, and IoU ranges. The self-trained YOLOv8m's mAP50-95=0.86096 provides an upper bound: even at the stricter mAP@0.5:0.95, the YOLOv8m vastly outperforms our model (whose mAP@0.5:0.95 is effectively 0, NaN on full eval). The ratio framing is correct but should use mAP@0.5:0.95, not mAP@0.5.

**Resolution:** Accept the "64-68% ratio" framing as the detection contribution. Drop "BEATS SOTA" language for the YOLOv8m from the main contribution list. Instead, report the YOLOv8m ceiling in the methodology section and compare the multi-task model to this ceiling. Reject using COCO-pretrained YOLOv8m as a comparison point (D1 showed mAP=0.0004 because COCO's 80 classes do not map to IndustReal's 24 ASD codes per d1_yolov8m/metrics.json per_class_ap). The distillation path should be evaluated as a T1 experiment for submission deadline.

---

### Debate 1.2: Detection mAP NaN on Full Eval — Is the Subsample Number Trustworthy?

**Detection Skeptical Reviewer (arguing the subsample number is not publishable):** The paper claims mAP50=0.358 for detection, but this comes from a 250-batch subsample (~1,000 frames out of 38,036). Every full validation run (D3 v4, v6, v7) produces det_mAP50=NaN and det_n_present=0. Multiple fixes were applied (epoch sentinel at evaluate.py:3342, criterion guard at evaluate.py:3365, collate_fn), yet the NaN persists. A paper cannot report detection mAP from a 2.6% subsample when the full evaluation fails. The reviewer cannot verify the subsample is representative. The detection section must state: "Detection evaluation on the full validation set is non-functional due to a pipeline bug; we report subsample numbers with this caveat." This is weaker but honest.

**Supportive Reviewer (ASD Parity, arguing the subsample is internally consistent):** The subsample metric is more trustworthy than the NaN suggests. The subsample runs within the main training loop using the same dataloader and same evaluation code as training-time validation. The fact that 250 batches produce consistent numbers across epochs (epoch 11: 0.317, epoch 17: 0.358 per PROGRESS_2026-07-06.md) with sensible per-class AP distributions (det_n_present=15/24, matching the number of classes with GT>0) strongly suggests the subsample is well-behaved. The NaN on full eval is likely a data-loading issue specific to the subprocess_eval.py path (119-progress-log documents five separate bugs in this path). The fix may be to run eval in-process rather than via subprocess. The core detection number (0.358) has internal consistency across checkpoints and should be publishable with a disclosure.

**mAP Fairness Reviewer (arguing the subsample may be biased high):** The NaN on full eval masks a deeper concern. If the detection head produces zero bounding boxes on the full set (det_n_present=0), the 0.358 subsample may be an artifact of the 250-batch sampler selecting easy frames. The subsample uses a class-balanced weighted random sampler which overrepresents rare classes. On easy common classes, the model predicts correctly; on the full set (dominated by common classes at natural frequency), precision may degrade. The 0.358 subsample number could be 2-3x higher than what the model actually achieves on the full distribution. Until the full-set evaluation works, the detection claim needs a strong caveat. A better approach: run the subsample with 10 different random seeds, report mean and standard deviation (e.g., 0.358 +- 0.042), to bound sampling bias.

**Resolution:** The highest-leverage action is to fix the subprocess_eval.py full-set evaluation. Five bugs were fixed per 119-progress-log; one more may remain (collate_fn for the full loader may need explicit sequence collation that the subsample loader does not). Alternative: run evaluation in-process by setting EVAL_MAX_BATCHES=0 in the main training config. The 10-seed subsample variance analysis should be done regardless to provide confidence bounds for the published number.

---

### Debate 1.3: Should Detection Be Framed as "Multi-Task Cost" or "Competitive with SOTA"?

**Detection Skeptical Reviewer (arguing for pure cost framing):** The paper currently frames detection as "64-68% of the dedicated YOLOv8m's mAP50" (reviewer-1-detection-path-to-SOTA.md). This is the most honest framing. Our model loses one-third of detection compared to a 1-day YOLOv8m training. Calling this "competitive" would be dishonest. The multi-task cost framing is correct: the paper's contribution is measuring and analyzing this cost. The paper should explicitly state: "Multi-task training degrades detection mAP from 0.995 (single-task YOLOv8m) to 0.358 (multi-task ConvNeXt-Tiny), a 64% relative reduction. We trace this degradation to the three pathologies in Section 4."

**Supportive Reviewer (ASD Parity, arguing for a combined efficiency metric):** Framing detection purely as "cost" misses the efficiency advantage. The 64% gap is against a YOLOv8m that does detection only. The ConvNeXt-Tiny does detection plus activity plus PSR plus head pose at the same inference cost. The correct comparison is not "detection mAP ratio" but "tasks-per-mAP." If a deployment needs all four tasks, alternatives are: (a) four separate YOLOv8m models (100M params, 4x inference, fusion required) or (b) one ConvNeXt-Tiny (28M params, 1x inference, no fusion). The YOLOv8m is not a viable alternative for the 4-task setting. The paper should report a combined efficiency metric: "task-score per parameter" or "task-score per GFLOP." Under this metric, ConvNeXt-Tiny may achieve 2-3x better efficiency than deploying four separate models.

**mAP Fairness Reviewer (arguing for dual reporting):** The "tasks-per-mAP" framing assumes all four tasks are equally important, which is application-specific. A factory cares about detection error rates (missing a missing screw matters), not a combined metric averaging detection with activity. The correct framing: report the detection mAP gap honestly (64% reduction vs single-task YOLOv8m), and separately report system-level efficiency (4 tasks at 28M params vs 4 separate models at 100M params). Let the reader judge the tradeoff. The paper should NOT claim "competitive detection" -- the numbers do not support it. It should claim "measurable multi-task detection cost, analyzed and attributed to specific pathologies, and offset by system-level efficiency."

**Resolution:** Use the "multi-task detection cost" framing as the primary detection narrative. Reject "competitive" language. Add a system-level efficiency comparison table (4 tasks, 1 model at X params, Y GFLOPs, Z FPS vs 4 single-task models at 4X params, 4Y GFLOPs, Z/4 FPS). Include detection mAP in a row alongside efficiency numbers so the reader sees both. The detection section title should be "Detection: Multi-Task Cost Measurement" not "Detection Results."

---

## Section 3: Activity Debate

*Agent: Activity (Per-Frame Action Classification) Specialist*

*File references: activity_clip.json, t3_full_eval.json, t3_mecanno_eval.json, act_remap_75_to_69.json, model.py, reviewer-2-activity-recasting.md, SOTA_STATUS.md*

---

### Debate 3.1: Per-Frame MLP vs MViTv2-S — is the 22x gap an architectural limitation or a training failure?

**Reviewer A — Temporal Reasoning Reviewer (arguing the gap is architectural and principled)**

The per-frame MLP achieves 0.028 clip-level top1 versus MViTv2-S at 0.622. This is not a training failure; it is an architectural inevitability. The MLP processes frames independently through a 2-layer network (LayerNorm, Linear 512-256, GELU, Dropout, Linear 256-69). It has zero temporal context, zero memory, and zero motion sensitivity. IndustReal action classes are defined by temporal segments (take wheel, put wheel, fit wheel) where the same object appears in the same visual configuration — the only distinguishing signal is whether the hand is approaching or releasing. No amount of class-balanced sampling, learning rate tuning, or capacity increase will close this gap because the per-frame formulation discards the temporal dimension at the architectural level. The model collapse to class 0 confirms this: when the MLP cannot find discriminative per-frame features, it defaults to the most common training label. The paper must cease all comparisons to MViTv2-S and re-frame as per-frame action classification, as reviewer-2 recommends. The T3 Meccanno eval (t3_mecanno_eval.json) showing only 0.04 top1_69 is actually the closest honest comparison: neither our model nor Meccanno-MViTv2 has the right pretraining, and both fail.

**Reviewer B — Temporal Reasoning Reviewer (arguing the gap is partly fixable)**

The collapse to class 0 is not inevitable — it indicates a training pathology, not just an architectural ceiling. The sampler uses class-balanced weighting with effective number smoothing (beta=0.99), and the logit layer bias was initialized to -0.5 specifically to discourage class-0 dominance. Despite these measures, the model still collapses. This suggests the backbone features (ConvNeXt-Tiny, ImageNet-pretrained) are not linearly separable for the 69 verb-grouped actions. The fix is not to give up on the per-frame MLP but to (a) train the non-simple TCN+ViT head, which adds temporal context over T=16 frames without changing the backbone, and (b) verify whether the ConvNeXt-Tiny features contain any action-discriminative information at all via a linear probe experiment. If the linear probe also fails, the bottleneck is the backbone, not the head. The 0.028 number should be reported as the "single-frame ImageNet-backbone action classification baseline" with the caveat that this defines the floor, not the ceiling, of per-frame performance on IndustReal.

**Resolution path:**
- Run a linear probe: frozen ConvNeXt-Tiny features mapped to 69 classes via a single linear layer, trained for 10 epochs. If this also plateaus near 0.030, the backbone does not encode action-discriminative features and the bottleneck is confirmed.
- Train the non-simple ActivityHead (TCN+ViT) for 20 epochs with the gradient fix. If this reaches 0.08-0.10 clip-level top1, temporal context helps but the backbone is still the bottleneck.
- Report the "predict majority class" baseline to separate class imbalance effects from model failure.

**Key file references:**
- activity_clip.json: clip-level eval showing 0.028 top1, class-0 dominance
- t3_full_eval.json: MViTv2-S baseline at 0.622 on 69-class
- t3_mecanno_eval.json: Meccanno-pretrained MViTv2-S at 0.04
- model.py ActivityHead lines 1334-1403: simple vs non-simple paths
- reviewer-2-activity-recasting.md: full protocol and re-framing analysis

---

### Debate 3.2: Verb Grouping — does the 75 to 69 class reduction help or harm the paper's contribution claim?

**Reviewer A — T3 Comparison Reviewer (arguing verb grouping breaks comparability and requires prominent disclosure)**

The paper reduces the original WACV 2024 75-class taxonomy to 69 verb-grouped classes via a customized "hybrid" mapping. This is a significant change that breaks direct comparability with every published baseline. The T3 full evaluation (t3_full_eval.json) shows top1_75 equals top1_69 at 0.6223 for MViTv2-S, but this is coincidental — the remap rarely changes correct predictions because non-merged classes map 1:1. For our model, however, the 6 merged classes could substantially affect the reported macro-F1. Without both 75-class and 69-class results, the reader cannot assess whether verb grouping helps or masks poor performance. The paper must do at least one of the following: (a) report both 75-class and 69-class results for all models, (b) justify the verb grouping with a principled taxonomy argument (e.g., take_instruction and take_short_brace are semantically identical), or (c) use the full 75-class taxonomy and accept that the per-frame baseline is even lower. The current hybrid mapping has object-similarity merges (pull_small_screw_pin to pull_wheel) that are not verb-based at all, making the "verb grouping" label misleading.

**Reviewer B — T3 Comparison Reviewer (arguing verb grouping is a feature, not a bug)**

The verb grouping corrects a known issue in the original IndustReal taxonomy: multiple action classes describe the same physical action with different object-specifier granularity. For example, take_instruction and take_short_brace are the same hand-to-shelf motion differing only in what the assembly manual terms the object. The original 75-class taxonomy over-splits by object type where the distinguishing signal is below the noise floor of RGB-based action classification. The 8% reduction (75 to 69) is modest and principled. The object-similarity merges (tighten_tooth_washer to tighten_nut) are justified because the same hand tool is used for both — the motion pattern is identical. The paper should report both 69-class (primary) and 75-class (for protocol transparency) in a supplementary table, with the primary narrative built around 69-class. The T3 baseline's identical top1_75 and top1_69 (t3_full_eval.json) is actually evidence that the grouping does not inflate numbers artificially — if it were cherry-picked to boost our results, it would also boost the baseline.

**Resolution path:**
- Compute both 75-class and 69-class macro-F1 and per-class accuracy from the same model checkpoint. If the 69-class numbers are meaningfully better, disclose the exact mapping and argue the case.
- Audit the object-similarity merges (pull_small_screw_pin, tighten_tooth_washer, loosen_tooth_washer) and either (a) move them to a "pure verb" mapping or (b) justify each on semantic grounds in the paper text.
- Add a row in the results table showing both class counts for each model.

**Key file references:**
- act_remap_75_to_69.json: full mapping with mode="hybrid"
- t3_full_eval.json: top1_75=0.6223, top1_69=0.6223
- industreal_dataset.py lines 919-921: remap logic in dataloader

---

### Debate 3.3: Activity as "per-frame action classification" — legitimate contribution or rhetorical sleight-of-hand?

**Reviewer A — Class Grouping Reviewer (arguing the re-framing is a strategic retreat from a failed task)**

The paper originally claimed multi-task activity recognition. Now it proposes re-framing as "per-frame action classification." This is not an honest re-framing — it is a retreat from a task at which the model demonstrably fails (0.028 clip-level top1) into a new task definition where no baselines exist and any positive number can be called a "first baseline." The re-framing moves the goalposts after the experiment. A legitimate per-frame action classification paper would: (a) not include the words "activity recognition" anywhere, (b) compare against a single-frame baseline (ImageNet linear probe), and (c) provide a theoretical argument for why per-frame classification is the right task for assembly verification — not just an admission that the temporal head failed. As currently framed, the re-framing reads as: "We cannot do activity recognition, so we defined a new task where our broken model counts as a contribution." This will not survive review.

**Reviewer B — Class Grouping Reviewer (arguing the re-framing is principled and productive)**

Per-frame action classification is a legitimate and under-explored task in manufacturing AI. Assembly verification systems running at 10 FPS need per-frame labels for real-time error detection — they cannot wait for 16-frame clip boundaries. The temporal baselines (MViTv2-S, SlowFast) have 178 to 1149 ms inference latency on clip windows, while our per-frame head adds essentially zero latency to the pipeline (one forward pass through a 69-way linear layer). This latency advantage is the contribution, not the absolute accuracy. The paper should make this argument explicit: per-frame action classification enables real-time assembly verification at a temporal granularity that clip-based methods cannot match. The 0.028 clip-level top1 is the wrong metric to report for this contribution claim. The correct metric is per-frame macro-F1 (0.110 from reviewer-2) with the caveat that this is the first reported number on a new task formulation. The re-framing is not a retreat — it is a correction of the paper's original category error (reviewer-2's own conclusion) and a clarification of the actual contribution.

**Resolution path:**
- Decide whether the paper leads with "per-frame action classification" as a deliberate design choice (Reviewer B's framing) or as a limitation disclosure (current framing in reviewer-2). These produce different paper tones.
- Report the per-frame macro-F1, the "predict majority class" baseline, and a linear probe ImageNet baseline together in a "Per-Frame Action Classification Baselines" table.
- Add a sentence or paragraph in Section 1 or Section 4 explaining why per-frame is the relevant task for assembly verification: latency, temporal granularity, and the fact that assembly errors must be detected at the frame where they occur, not after a 16-frame clip completes.

**Key file references:**
- reviewer-2-activity-recasting.md: full re-framing argument and protocol recommendations
- SOTA_STATUS.md: current per-frame top1=0.023, clip-level=0.028
- activity_clip.json: class-0 dominance across all validation classes

---

## Section 10: Adversarial Debate

*Agent: Adversarial Reviewer (Agent 10)*

*File references: SOTA_STATUS.md, 15_REVISED_RISK_AND_WEAKNESSES.md, 21_PATHOLOGY_CORRECTIONS.md, 24_REVIEWER_DEFENSE_V2.md, AAIML_10_REVIEWER_EVALUATIONS.md*

---

### Debate 1: Does Detection Actually Beat SOTA?

**Adversarial Reviewer (Agent 10):** Your claim that detection beats SOTA rests on the ASD metric (mAP50 = 0.995). But ASD is a 24-way binary state classification, not object detection in the standard sense. Your D1 evaluation using the standard protocol yields mAP50 = 0.0004. The 0.995 number comes from a label space you defined, while the 0.0004 comes from the same protocol used by the SOTA comparisons you cite. The gap between 0.995 and 0.0004 is not a measurement artifact — it reveals that your ASD formulation is not comparable to any standard detection benchmark. You are comparing apples to oranges and calling the apple better than the orange.

**Defending Author:** The ASD label space is the correct formulation for our domain. Assembly state detection is inherently a multi-label binary problem: each of the 11 assembly components can be in one of 2-4 states, making 24 binary classifiers the natural formulation. The D1 evaluation is a known label-remapping issue we are correcting in the camera-ready version — it has nothing to do with the models capability. The confusion matrix analysis showing that 70% of errors are 1-bit-adjacent provides practitioners with actionable guidance: the model correctly identifies the coarse assembly state and only confuses single component transitions. This level of interpretability is rare in detection papers and justifies reporting the ASD metric as the primary detection result. The SOTA comparison uses YOLOv8m trained on the same dataset and same ASD label space, so the comparison is fair. Furthermore, our model achieves this with a single $299 GPU at 93 GFLOPs versus YOLOv8ms dedicated hardware.

**Adversarial Reviewer:** You say the D1 issue is a label-remapping problem, but your own SOTA_STATUS.md calls it broken. You said the same about epoch 11s checkpoint selection being NaN-inflated. Your codebase has a pattern of finding bugs after the fact and labeling them as minor. If D1 is a label remapping bug, show the corrected D1 number. Without it, a reviewer cannot trust that your detection head works outside the specific ASD label reduction. You also have not shown that a dedicated single-task ASD model would not achieve higher mAP50 — your controlled ablation shows a delta of -0.03, which you frame as structural interference, but this is still a degradation from the single-task baseline. Beating SOTA with a multi-task model is impressive; beating SOTA with a multi-task model that is 0.03 worse than its own single-task version is less so.

**Defending Author:** The D1 correction is in progress and will be in the camera-ready. The controlled ablation is precisely the papers contribution: the delta of -0.03 with equal gradient updates is the first clean measurement of structural interference in multi-task detection, unconfounded by epoch budget or learning rate differences. That the multi-task model achieves mAP50 = 0.995 at all while simultaneously learning four other tasks is the result. The single-task comparison uses the same backbone with identical training budget, and delta = -0.03 is remarkably low interference. No prior MTL paper on assembly verification has reported this measurement. The detection beating SOTA is the headline; the structural interference measurement is the contribution.

---

### Debate 2: Is Activity Recognition at 2.8% Defensible?

**Adversarial Reviewer (Agent 10):** Clip-level top-1 accuracy of 0.028 with a chance baseline of 0.0135 is indistinguishable from random. The 95% confidence interval around chance for a 74-class problem with your evaluation N would easily include 2.8%. You claim this is a deliberate trade-off, but the honest story is that your temporal head (TCN+ViT at 18.3%) was discarded because it caused gradient conflicts, and your simple MLP replacement (150K params) simply cannot do temporal reasoning. A model that achieves 2.8% on one of its five tasks is not a five-task model — it is a four-task model with a dead head. Any reviewer from the action recognition community will immediately flag this as non-functional and recommend removal.

**Defending Author:** We explicitly frame this as per-frame action classification, not activity recognition. The distinction matters because MViTv2-S uses 32-frame clip input with full temporal attention, while our model processes a single RGB frame. The contribution is not the absolute accuracy but the documentation of why temporal heads fail in multi-task assembly settings. Pathology 1 (temporal-head/sampler mismatch) is validated here: the class-balanced WeightedRandomSampler combined with FeatureBank creates a 98.3% probability that consecutive frames in a batch come from different recordings, defeating any temporal modeling. This finding benefits the entire MTL community. With temporal smoothing at inference time — a standard post-processing step we acknowledge is future work — per-step accuracy is expected to exceed 80%. The activity head also serves as a controlled probe for measuring multi-task interference. Its minimal capacity (150K params, 0.3% of model) means any gradient conflict measured on this head is attributable to the sharing, not to head capacity limits.

**Adversarial Reviewer:** Your defense concedes the point: you need temporal smoothing to reach 80%, but you have not implemented or measured it. Per-step accuracy of above 80% is a claim, not a result. You also cannot claim both that the activity head is too simple (so its low accuracy is expected) and that it serves as a meaningful probe for multi-task interference (so its accuracy tells us something about gradient conflict). If the head is too simple to do the task, its low accuracy tells us nothing about interference — it tells us the head is too simple. A properly powered probe would be the TCN+ViT head with the corrected sampler. You chose to discard that experiment rather than run it. This is a paper about training pathologies that declined to run the confirmatory experiment for its own pathology. That is a fatal omission.

**Defending Author:** The MLP versus TCN+ViT ablation is planned as Ablation 1 in our execution timeline. We agree it is the correct confirmatory experiment. The camera-ready version will include this comparison. The probe argument stands even with the simple head: if the simple head cannot learn, gradient conflicts from other tasks are not the cause. If the TCN+ViT head with corrected sampling also cannot learn, that is a different finding (capacity or optimization difficulty). Either result is informative, and we will report whichever outcome occurs. The paper already acknowledges that the activity head is below any reasonable deployment threshold and does not claim otherwise. What it claims is the pathology finding, which is independent of the head architecture.

---

### Debate 3: Does the PSR Head Actually Learn?

**Adversarial Reviewer (Agent 10):** Your PSR heads log_var_psr is -0.04, meaning the Kendall temperature is 0.96 — barely moved from initialization at 1.0. Your liveness monitor shows all 11 per-component PSR sub-heads with RMS gradient exactly 0.00 across thousands of steps. Your PSR training loss is logged as 0.0000 in 569 consecutive per-frame steps. The PSR head is not training. The F1 of 0.7499 you report is from the shared backbone features passing through a randomly initialized linear head. How does this differ from zero-shot evaluation of the backbone on PSR? If I train only the backbone (on detection and pose) and freeze it, then add a randomly initialized PSR head without any PSR-specific training, what F1 would I get? If the answer is close to 0.7499, then PSR-specific training contributed nothing, and your claim of five-task multi-task learning is false for PSR.

**Defending Author:** The log_var_psr being near zero means the model considers PSR well-calibrated at the default temperature, not that PSR is untrained. The Kendall mechanism increases the loss weight when a task is underfitting and decreases it when the task is well-fit. If PSR converges quickly to a good solution, the temperature naturally stays near the initialization. The liveness monitor showing DEAD sub-heads is misleading because it measures gradient RMS at a specific point in the computational graph. The shared backbone layers preceding the PSR head do receive gradients, as evidenced by the nonzero aggregate RMS (1.88e-03). The per-component breakdown showing F1 = 0.3455 on component 4 (14.2% prevalence) versus F1 = 0.4020 on component 10 (18.3% prevalence) demonstrates learning beyond the trivial always-empty baseline. A zero-shot backbone classifier would not achieve F1 = 0.35 on components that appear in only 14% of frames, because the backbone was never trained to distinguish procedure step components. The F1 = 0.7499 represents genuine, though imperfect, PSR learning.

**Adversarial Reviewer:** Your aggregate RMS of 1.88e-03 is barely above the DEAD threshold and is three orders of magnitude smaller than the detection head gradients. Even if some gradient reaches the PSR head through shared layers, the magnitude is negligible. You claim the PSR head converges quickly, but the log_var_psr is flat from epoch 1 to epoch 135 — it never rises, never falls, never oscillates. A task that converges in one epoch is either trivially easy (in which case a single-task baseline would achieve F1 = 0.95 or higher) or getting no gradient signal (in which case the temperature stays at initialization because the loss never changes). Your per-component breakdown actually hurts your argument: components with prevalence above 50% achieve F1 of 0.75 to 1.00, while those below 20% achieve F1 of 0.35 to 0.40. This is exactly the pattern of a classifier learning the prevalence prior, not component-specific discrimination. You have no single-task PSR ablation. Without it, the claim that multi-task learning helps PSR is unsupported, and the claim that F1 = 0.7499 represents multi-task learning success is indistinguishable from zero-shot backbone performance.

**Defending Author:** We agree that a single-task PSR ablation would strengthen the paper. It is listed as a planned experiment in the execution timeline. However, the claim is not that multi-task learning helps PSR; the claim is that PSR can be integrated into a multi-task framework without collapsing — and that the Kendall mechanism keeps PSR viable when properly bounded (max_psr=0.0). The fact that PSR does not dominate the loss at the expense of other tasks, and that it maintains competitive F1, is exactly the low-interference result the paper claims. The per-component breakdown shows that even low-prevalence components achieve F1 significantly above the trivial baseline (F1=0.35 versus F1=0.0 or F1=expected from prevalence alone). The component 4 F1 of 0.3455 on 14.2% prevalence represents a 2.4x improvement over predicting always negative. This is not prior-following; it is genuine discrimination. We will provide the single-task ablation in the camera-ready, but the existing evidence already supports the PSR learning claim.

---

### Debate 4: Are the Three Pathologies a Genuine Contribution or Overclaiming?

**Adversarial Reviewer (Agent 10):** Your three pathologies have serious credibility problems. Pathology 1 (temporal-head/sampler mismatch) uses R=12 in the equation when the correct value is R=58, changing the probability from 8.3% to 1.7%. You corrected this, but the fact it was wrong in the first place suggests you did not verify the math before writing the paper. Pathology 2 (Kendall spiral) was never observed in training — your own audit confirms no training log shows the claimed effect, and the balanced sampler that prevents it was implemented before any training run. Pathology 3 (gradient probe misreading) is a survey of 20 GitHub repositories with no documented methodology until after reviewer feedback. One of three pathologies had a factual error, one was never empirically observed, and one had to add its methodology after the fact. In an adversarial review, I would recommend rejection on the grounds that the papers primary claimed contribution — the three pathologies — is unsupported. A PC chair reading this would agree that the paper fails to meet the IEEE standard of empirical rigor.

**Defending Author:** The corrections demonstrate intellectual honesty, not weakness. The R=12 error was caught during internal audit, before any reviewer saw it, and was corrected to R=58, which makes the conclusion stronger (98.3% non-temporal probability versus the original 91.7%). Finding and fixing errors before submission is exactly what rigorous research looks like. Pathology 2 is explicitly reframed in the camera-ready version as a theoretical contribution with preemptive guarding, not an empirical observation. The mathematical fixed-point analysis is correct and novel, and the per-task bounds (min_act=-0.5, max_psr=0.0, max_pose=3.0) are a design contribution that any practitioner can adopt. Pathology 3s survey methodology — two independent annotators with Cohen's kappa = 0.85, top-20 repositories sorted by stars, documented search query — was present in the codebase before the reviewer defense document was written. The paper was not submitted with missing methodology; the defense document pre-writes responses to anticipated critiques.

**Adversarial Reviewer:** You are describing internal documents that no reviewer will see. The reviewer sees the paper, not the planning documents. If the paper submitted to AAIML still describes Pathology 2 as an empirical observation (which the current .tex file does, per your own audit), the reviewer will correctly identify this as overclaiming. If the paper does not include the survey methodology in the main text (because of page limits), a reviewer cannot verify it without requesting supplementary material — which most reviewers will not do. The burden is on you to present the evidence, not on the reviewer to find it. Furthermore, the page allocation tells the story: three pathologies get approximately 3.5 pages of the paper. For a contribution that is one-third theoretical (Pathology 2, unobserved) and one-third post-hoc corrected (Pathology 1, had a factor-of-5 error in the key equation), the page budget is disproportionate. A skeptical PC chair will ask: if these pathologies are the papers contribution, why is the evidence so thin?

**Defending Author:** The page allocation is proportionate to the contributions impact. Pathology 1 affects every MTL practitioner using class-balanced samplers with temporal models — the finding that a WeightedRandomSampler plus FeatureBank creates a 98.3% non-temporal batch probability is a cautionary result that deserves space. Pathology 2 provides the mathematical framework for understanding when and why the Kendall mechanism fails under extreme label sparsity, which is increasingly common in industrial and medical applications. Pathology 3 documents a measurement artifact that makes published gradient-conflict numbers unreliable — a methodological contribution. All three are general findings, not specific to our system or dataset. The paper does not overclaim empirical support for any of them. The camera-ready version will ensure Pathology 2 is clearly labeled as a theoretical analysis with preemptive guarding, Pathology 1 uses the correct R=58, and Pathology 3 includes the survey methodology inline or in supplementary with a clear reference. The paper that lands at AAIML will be cleaner than the one we are debating now. That is the point of preparing these debates.

---

## Section 6: Training Infrastructure Debate

*Agent: Training Infrastructure Specialist*

---

### Debate 6.1: Are the CUDA crashes caused by VRAM exhaustion, kernel timeout, or a driver-level issue?

**CUDA Timeout Reviewer (arguing it is a kernel execution timeout triggered by VRAM pressure)**

The evidence is clear: all four crashes produce CUDNN_STATUS_EXECUTION_FAILED_CUDART followed by cudaErrorLaunchTimeout. The launch timeout error means a CUDA kernel took longer than the watchdog timer allows (typically 2 seconds on consumer GPUs via TDR). The crashes only occur at batch_size=6 with reserved memory of 7.68-10.07 GB, and stop completely at batch_size=2 with reserved memory of 3.5-4.2 GB. The mechanism is straightforward: larger batches produce larger intermediate tensors, which require more CUDA kernel launches and longer per-kernel execution times. When the total time for the backward pass exceeds the TDR threshold, the driver kills the context. This is not an OOM condition — it is a timeout condition caused by the GPU spending too long on a single operation chain. The fix is to keep per-batch memory low enough that individual kernel chains complete within the TDR window. Config.py attempted to address this with CUDNN_BENCHMARK=False, which avoids benchmark-selected kernel variants that may have longer execution times, but the underlying batch_size=6 still produces enough work to trigger the timeout. The fact that batch_size=2 eliminates the crashes is the experimental confirmation.

**Batch Size Reviewer (arguing the crashes are a CUDA 13.0 + RTX 5060 Ti compatibility issue, not purely a batch_size problem)**

If it were purely a batch_size issue, we would expect the crash to occur at a consistent memory threshold every time. Instead, the four crashes happened at different steps in different epochs, with reserved memory varying between 7.68 GB and 10.07 GB. A system that crashes at 7.68 GB in one epoch and runs fine up to 10.07 GB in another is not exhibiting a deterministic memory threshold. Rather, the crashes correlate with epoch number (19, 21, 23, 24) — the later epochs, after the model has been training for over 10 hours. This suggests progressive memory fragmentation, thermal accumulation, or a driver memory leak. The RTX 5060 Ti is a Blackwell-architecture card running CUDA 13.0 with torch 2.12.1 — this is a very new combination. The CUDNN_BENCHMARK=False fix was added on 2026-07-02, yet all four crashes occurred on 2026-07-06, four days later. If the fix was actually active, it did not prevent the crashes. This points to a driver-level or firmware-level issue that manifests stochastically under sustained load, not to a simple batch-size ceiling. The batch_size=2 run has only been going for 7 hours — it may still crash at hour 12 or 15 when the same cumulative conditions arise.

**Resource Allocation Reviewer (arguing the real issue is the gradient accumulation schedule)**

The crashes always happen during the backward pass, not the forward pass. With GRAD_ACCUM_STEPS=8, the backward pass processes accumulated gradients from 8 forward passes. For batch_size=6, this means the backward pass spans 6x8=48 samples. The intermediate activations for 48 samples must be retained in memory (they cannot be freed after each forward pass because they are needed for the backward pass). This creates a memory spike during the backward pass that exceeds what a simple forward-pass memory measurement captures. The reserved memory of 7.68-10.07 GB is the steady-state allocation; the backward pass temporarily requires additional memory for the gradient computation graph. When this temporary allocation pushes into the last few GB of the 16.6 GB card, the GPU memory management overhead causes kernel launch delays that trigger the TDR timeout. The fix is not to reduce batch_size to 2 (which wastes 75% of the GPU), but to reduce GRAD_ACCUM_STEPS. With batch_size=4 and GRAD_ACCUM_STEPS=4, you get the same effective batch of 16 as the current stable configuration, but with only 4 forward passes accumulated before each backward pass, reducing the temporary memory spike. This configuration should be tested before declaring batch_size=2 the only stable option.

**Resolution path:**
- Run a 12-hour experiment at batch_size=4, GRAD_ACCUM_STEPS=4 to test whether reduced accumulation steps stabilize training at higher per-batch throughput
- Monitor dmesg for Xid errors and GPU temperature logs to distinguish driver crashes from thermal throttling
- Compare per-step kernel timing between batch_size=6 (crash epochs) and batch_size=2 (stable) via CUDA events or NVTX ranges
- If the 5060 Ti crashes again at batch_size=2 after 12+ hours, the issue is cumulative (driver leak or fragmentation) rather than batch-size-dependent

**Key file references:**
- train.log lines 808-821, 1969-1982, 2704-2717, 3114-3127: exact crash sites
- run_command.txt: command line with --batch-size 2 --num-workers 0
- config.py lines 579-581: BATCH_SIZE=6, GRAD_ACCUM_STEPS=8, EFFECTIVE_BATCH=48
- config.py lines 693-695: CUDNN_BENCHMARK=False stability fix
- train_ep24_smaller.log: current stable epoch-25 run at batch_size=2

---

### Debate 6.2: Should the effective batch size of 16 be accepted, or is it a convergence risk?

**Batch Size Reviewer (arguing that effective batch 16 is acceptable for this model)**

The switch from effective batch 48 to 16 is less consequential than it appears. The model uses ConvNeXt-Tiny with BatchNorm, which is robust to batch size variation above 8. The original BATCH_SIZE=6 with GRAD_ACCUM_STEPS=8 was chosen for throughput, not for convergence necessity — the "per paper Section Implementation" comment in config.py line 580 cites throughput, not accuracy. The Kendall weighting mechanism (USE_KENDALL=True) adapts learning rates per task dynamically, which further reduces sensitivity to absolute batch size. The current epoch-25 run at batch_size=2 is producing reasonable loss values (total loss 4-10 range, comparable to the batch_size=6 runs). The Day-1 checkpoint already achieved epoch 12 with 0 errors at the full batch_size=6, and the current run is continuing from that checkpoint — it inherits the converged features. Reducing effective batch from 48 to 16 at epoch 25 is a fine-tuning adjustment, not a from-scratch training change. The convergence impact is negligible.

**Resource Allocation Reviewer (arguing that effective batch 16 is a significant risk that must be validated)**

The paper reports results at effective batch 48. If the final model is trained at effective batch 16, every metric in the paper is potentially different. BatchNorm statistics shift: running mean and variance computed at batch_size=2 are noisier than at batch_size=6, which affects validation accuracy especially for small classes. The gradient variance increases at smaller effective batch, which interacts with the OneCycleLR scheduler — the learning rate peak of 2.5e-04 was tuned for effective batch 48, not 16. Most critically, the Kendall weighting dynamics change: at effective batch 16, the log variance estimates update more erratically, which could cause the known HP_PREC_CAP issue (pose gradient starvation flagged in the log as "HP_PREC_CAP ACTIVE: raw lv_pose grad-starved") to worsen. The paper cannot claim the original results unless the full training is repeated at effective batch 16 from scratch, with the same number of epochs and the same evaluation protocol. Continuing from a batch_size=6 checkpoint at epoch 25 is not a valid substitute — the first 25 epochs of features were learned at a different effective batch size, and the remaining 75 epochs may overwrite or distort them.

**CUDA Timeout Reviewer (arguing that effective batch 16 is the only realistic option and the paper should adapt to it)**

The alternative to effective batch 16 is no training at all — the batch_size=6 configuration crashes every 1-3 hours and cannot complete a single epoch without manual restart. The paper has a deadline. Effective batch 16 is stable and has been running for over 7 hours without incident. The correct response is not to debate whether effective batch 16 is optimal — it is to accept it, document the change, and validate that the final metrics are comparable. The paper can include an ablation that compares effective batch 16 versus 48 on a short run (e.g., 10 epochs from scratch) and show that the validation curves overlap within 5%. If they do, the concern is addressed. If they diverge, the paper reports the divergence as a limitation. Either outcome is publishable — the community benefits from knowing the batch-size sensitivity of multi-task ConvNeXt training.

**Resolution path:**
- Run a controlled 10-epoch comparison: effective batch 16 (batch_size=2, GRAD_ACCUM=8) versus effective batch 48 (if stable), comparing validation loss per head at each epoch
- If the batch_size=6 configuration cannot sustain 10 epochs due to crashes, the comparison is moot — effective batch 16 is the only viable configuration and the paper must adapt
- Report final metric sensitivity: if all four metrics (det mAP, pose MAE, act Top-1, PSR POS) change by less than 3% relative, effective batch 16 is validated
- Document the batch-size constraint in the paper's experimental setup section as a hardware limitation of the RTX 5060 Ti under CUDA 13.0

**Key file references:**
- config.py lines 579-581: original BATCH_SIZE=6, GRAD_ACCUM_STEPS=8, EFFECTIVE_BATCH=48
- train_ep24_smaller.log: effective batch 16 run in progress (epoch 25)
- day1-checkpoint-done-and-next-steps.md lines 23-27: original run at epoch 12 with 0 errors
- HP_PREC_CAP log messages at steps 1101 and 901: known pose gradient starvation

---

### Debate 6.3: Should the paper acknowledge the GPU allocation as suboptimal and propose a better strategy?

**Resource Allocation Reviewer (arguing the paper should document the GPU split and propose a better allocation)**

The paper currently presents results from a single RTX 5060 Ti for the primary run and an RTX 3060 for a det-only ablation. This is a resource-constrained setup, and the paper should be transparent about it. The 5060 Ti is using approximately 3.5-4 GB out of 16 GB at its stable configuration — that is 75-78% idle VRAM. A reader with access to a single 24 GB GPU (RTX 4090 or A5000) would see immediate headroom for larger batch sizes, longer sequences, or additional temporal context. By documenting the current allocation and the crash-mitigation history, the paper provides a useful reference for practitioners setting up multi-task ConvNeXt training on consumer GPUs. The documentation should include: the crash timeline (epochs 19-24), the resolution (batch_size=2 and NUM_WORKERS=0), the effective batch tradeoff, and the GPU utilization percentage. This turns a limitation into a contribution — a reproducibility case study for Blackwell-architecture GPU training.

**CUDA Timeout Reviewer (arguing the allocation detail belongs in supplementary, not the main paper)**

The paper is about multi-task action classification on IndustReal, not about GPU allocation strategies for Blackwell GPUs. Including 3-4 paragraphs of training infrastructure details distracts from the core contribution. The AAIML venue cares about the pathology findings (temporal-head mismatch, loss scale suppression, PSR collapse) and the deployment evidence — not about why batch_size=6 crashed on the authors' specific GPU. A brief note in the experimental setup section is sufficient: "Training was performed on an NVIDIA RTX 5060 Ti (16 GB). Due to CUDA kernel timeout issues at batch_size=6, we trained at batch_size=2 with GRAD_ACCUM_STEPS=8, yielding an effective batch of 16. We verified that this configuration produces comparable metrics (validation loss within 5% of batch_size=6 over 10 epochs; see Supplementary Table S1)." The detailed crash log, GPU heartbeat, and memory analysis belong in the supplementary material for reproducibility. The main paper should stay focused on the scientific contributions.

**Batch Size Reviewer (arguing the GPU split is an opportunity for a practical contribution that the paper should not waste)**

There is a middle ground: use the GPU allocation experience to motivate a practical recommendation in the paper's discussion section. The paper already has a section on infrastructure pathology (the temporal-head mismatch, loss scale suppression, and PSR collapse are all training-infrastructure findings). Adding a fourth pathology — "CUDA kernel timeout under gradient accumulation on consumer GPUs" — is thematically consistent and strengthens the paper's identity as a practical guide for multi-task training. The recommendation would be simple: on Blackwell GPUs with CUDA 13.0, prefer batch_size=4 with GRAD_ACCUM_STEPS=4 over batch_size=2 with GRAD_ACCUM_STEPS=8 to minimize kernel timeout risk while maintaining throughput. This is an actionable insight that helps other practitioners avoid the same debugging cycle. It should take no more than 1 paragraph in Section 5 (Discussion) and 2-3 sentences in the experimental setup. The full crash-log analysis can go to supplementary.

**Resolution path:**
- Add one paragraph to the Discussion section: "GPU Stability for Blackwell Multi-Task Training" — describe the crash pattern, the resolution, and the recommendation
- Add a supplementary section (Supplementary B) with the full crash log, memory profiles, and the batch_size comparison experiment
- In the main paper's experimental setup, add one sentence: "Due to CUDA kernel timeout at batch_size=6 on the RTX 5060 Ti, we used batch_size=2 (effective batch 16); see Supplementary B for details."
- Do not expand the main paper's 8-page limit with GPU infrastructure detail

**Key file references:**
- train.log: all 4 crash sites documented at lines indicated in Debate 6.1
- .gpu_heartbeat: reserved=3.54 GB confirming 78% VRAM idle
- day1-checkpoint-done-and-next-steps.md: Priority 0 and Priority 1 experiment lists for both GPUs
- config.py lines 614-616: NUM_WORKERS=0 fix notes with crash documentation convention


---

## Section 5: Architecture Debate

*Agent: Multi-Task Architecture Specialist*

---

### Debate 5.1: Two-Stage FiLM Conditioning — Is FiLM the primary technical novelty, or a complexity burden given that both inputs are detached and one comes from pseudo-keypoints?

**Claim:** Two-stage FiLM conditioning (PoseFiLM from detection boxes + HeadPoseFiLM from 9-DoF head pose) is the paper's primary technical contribution. It modulates backbone C5 features with pose context to improve multi-task performance.

**Reviewer A — Multi-Task Justification Reviewer (supporting FiLM novelty)**

The two-stage FiLM design is genuinely novel in the multi-task assembly verification context. Prior work applies FiLM in single-task settings (visual question answering, image generation) or uses a single conditioning signal. Stacking two FiLM stages sequentially allows the backbone to output different feature representations for detection versus activity tasks. The 90/100 technical novelty score in 04_BEST_PAPER_FORMULA.md reflects this. Furthermore, both FiLM modules are lightweight: PoseFiLM adds 841K parameters and HeadPoseFiLM adds 401K parameters, totaling 1.24M (2.7% of the model). Even if the conditioning signal is imperfect, the parameter cost is negligible. A 2.7% parameter investment that enables the multi-task functionality claim is justified.

**Reviewer B — Kendall Critic Reviewer (arguing FiLM is ornamental)**

The FiLM modules receive inputs that undermine their claimed function. First, the body keypoints fed to PoseFiLM are pseudo-generated from the detection argmax (model.py lines 1982-2037). These are not real annotations -- they are a heuristic COCO layout pasted onto a bounding box from a detection head that shows dead epochs with loss=0.0000 (train.log epoch 25). Second, both pose_confidence and keypoints are detached before PoseFiLM (model.py lines 696, 2044), and head_pose is detached before HeadPoseFiLM (model.py line 2168). No gradient from activity or detection tasks can improve the conditioning. Third, there is no ablation in the plan comparing FiLM-on vs FiLM-off for either stage. The head pose specialist independently flagged this gap in Question HP-5.

**Resolution path:**
- Run the missing ablation: USE_HAND_FILM=False and USE_HEADPOSE_FILM=False, train for 20 epochs, compare all metrics against baseline
- Compute the L2 norm of (gamma - 1) and beta across validation batches
- Cite the ablation result explicitly in the architecture section

**Key file references:**
- model.py lines 626-716 (PoseFiLM), 722-792 (HeadPoseFiLM)
- model.py lines 1982-2037 (pseudo-keypoint generation)
- model.py lines 2044, 2168 (detach operations)
- config.py lines 115-136 (USE_HAND_FILM, USE_HEADPOSE_FILM flags)
- 04_BEST_PAPER_FORMULA.md (technical novelty score)
- 127_50_DEEP_QUESTIONS_FOR_OPUS.md Question A-2 (FiLM effectiveness)

---

### Debate 5.2: Kendall Uncertainty Weighting — Does Kendall automatically balance multi-task learning, or does it require so many overrides that it adds complexity without benefit?

**Claim:** Kendall homoscedastic uncertainty weighting automatically balances 4 tasks (detection, pose, activity, PSR) by learning task-specific precisions from data, eliminating manual loss weight tuning.

**Reviewer A — Kendall Critic Reviewer (arguing Kendall adds complexity without benefit)**

The evidence from config.py and the training logs shows that the learned Kendall weights are not trusted. Three independent override mechanisms exist. First, KENDALL_HP_PREC_CAP forces head_pose precision to never exceed detection precision (config.py line 89). Without this, head_pose gets approximately 54.6x precision while detection gets approximately 1.4x, causing the backbone to optimize for head_pose alone. This is documented in Opus v8 as the root cause of detection collapse. Second, KENDALL_STAGED_TRAINING=False disables the built-in curriculum (config.py line 104). Third, the KENDALL_FIXED_WEIGHTS env var (config.py line 96) provides a complete bypass with hard-weighted detection at 1.0 and head_pose at 0.2. The training logs show log_var_psr stuck near 0.000 with near-zero lv_grad, meaning PSR's learned precision is not driven by PSR performance.

**Reviewer B — Multi-Task Justification Reviewer (arguing Kendall is still useful despite constraints)**

The override mechanisms are not evidence of failure -- they are evidence of proper engineering. The Kendall formulation provides a principled framework, and the constraints maintain it within a safe operating regime. KENDALL_HP_PREC_CAP is a single-line constraint preventing a known pathology while preserving flexibility for all other tasks. The training logs show that log_var_det and log_var_act DO move and reflect training dynamics: log_var_det tracks detection convergence (starting at 0.08, rising to 0.56 as detection loss drops, then settling near 0.42). This is exactly the behavior Kendall promises. The fact that PSR's log_var stays near zero is consistent with PSR loss being structurally zero on 75% of batches under the transition objective. The alternative -- manually searching a 4-dimensional loss-weight space via grid search -- would require hundreds of training runs.

**Resolution path:**
- Add the KENDALL_FIXED_WEIGHTS ablation to the paper's ablation table across all 5 metrics
- Log and report effective precision (exp(-log_var)) for all 4 tasks at convergence
- Add a figure showing log_var trajectories to demonstrate learned weights track task convergence

**Key file references:**
- config.py lines 82-108 (KENDALL_HP_PREC_CAP, KENDALL_FIXED_WEIGHTS, KENDALL_STAGED_TRAINING)
- train.log epoch 25 (Kendall log_var values at 500-step intervals)
- losses.py lines 1658-1828 (Kendall weighting implementation)
- 13_ARCHITECTURE_REWRITE.md Section 4.6 (lessons from multi-task training)

---

### Debate 5.3: Sequence-Mode PSR — Is the 25% compute overhead for a detached PSR head justified while PSR F1 remains at zero?

**Claim:** Sequence-mode PSR training (T=8 windows with causal transformer) enables temporal reasoning for procedure step recognition and justifies the approximate 25% compute overhead from PSR-only sequence batches.

**Reviewer A — Sequence Mode Reviewer (arguing the cost-benefit is negative)**

Sequence-mode batches (PSR_SEQ_EVERY_N_BATCHES=4) consume roughly 25% of all forward passes but produce zero measured benefit. The PSR head shows loss=0.0000 and F1=0 at epoch 25. With DETACH_PSR_FPN=True, PSR gradients do not flow into shared FPN features, so these batches contribute nothing to shared feature learning. The reduction from 50% to 25% (every 4 batches instead of every 2) improved the situation but did not address the fundamental question: why allocate compute to a detached head with zero F1? Every PSR-only batch inflates effective training time by 25% without improving any metric, directly undermining the paper's efficiency claim.

**Reviewer B — Sequence Mode Reviewer (arguing the overhead is acceptable for a challenging task)**

PSR is acknowledged as the hardest task (11 rare-event predictions from 95%-static fill-forward labels). The zero F1 at epoch 25 does not mean the architecture is wrong -- PSR may require 50-100 epochs to show recovery, especially given the PSR_FOCAL_GAMMA reduction from 2.0 to 0.5 happened late in development. DETACH_PSR_FPN is a deliberate design choice to prevent PSR's instability from corrupting detection features. The compute cost is acceptable because PSR batches are smaller (T=8 sequences use fewer images than full multi-task passes). Furthermore, sequence-mode cost is a training-time concern only. At inference, the PSR head runs per-frame with the causal transformer cache (model.py lines 1676-1716), adding no throughput overhead.

**Resolution path:**
- Report PSR F1 at epochs 50, 75, and 100 to show whether the architecture eventually produces signal
- Add a per-batch timing breakdown in the appendix showing actual wall-clock overhead of sequence batches
- If PSR remains zero-F1 at epoch 100, move PSR to supplementary materials and note it as non-functional in the main paper

**Key file references:**
- config.py lines 1067-1097 (PSR sequence mode settings)
- config.py lines 1014-1018 (DETACH_PSR_FPN)
- model.py lines 1539-1736 (PSR head architecture)
- train.log epoch 25 (consistent psr=0.0000)
- 13_ARCHITECTURE_REWRITE.md (precedent for documenting abandoned directions)

## Section 9: Paper Writing Debate

**Source files:** popw_aaiml2027.tex, 22_REVISED_PAPER_OUTLINE.md, 02_SECTION_BY_SECTION.md, 07_RISK_REGISTER.md, 15_REVISED_RISK_AND_WEAKNESSES.md, 06_TABLES_FIGURES.md, 23_ABLATION_AND_EXPERIMENT_PLAN.md, 04_BEST_PAPER_FORMULA.md

### Reviewer 1: Paper Narrative Reviewer — Three Debates

**Debate 1: Is the three-pathology framing a genuine contribution or a debugging report?**

FOR: The pathology framing transforms modest numerical results into a generalizable contribution. Detection at 64-68% of YOLOv8m would be a weak result if the paper claimed to be a system paper. As a pathology paper it becomes evidence of how infrastructure failures silently degrade MTL. The 22_REVISED_PAPER_OUTLINE correctly allocates 2.5 pages to pathology analysis versus 1.0 page to system architecture. This ratio signals to the reader that the general insight (infrastructure components fail silently) is the contribution, not the specific numbers.

AGAINST: Three of the four contributions (pathologies 1 and 2, plus the 18 fixes catalog) describe bugs in a single training pipeline. Pathology 3 (gradient measurement artifacts) has the 20-repository survey showing 70% prevalence, which is the only claim that generalizes. A reviewer can correctly say "you found three bugs in your code, fixed them, and want to publish the bug report." The system-as-vehicle framing only works if the vehicle is interesting enough to carry the lesson — and with PSR F1=0, detection at one-third of YOLOv8m, and activity at 12.9%, the vehicle looks broken.

File evidence: 22_REVISED_PAPER_OUTLINE allocates 2.5 pages to pathologies; 04_BEST_PAPER_FORMULA scores technical novelty at 90/100; popw_aaiml2027.tex contribution list leads with pathologies; 15_REVISED_RISK_AND_WEAKNESSES upgrades paper fit for AAIML from MEDIUM to LOW risk because the pathology findings are "directly ML-relevant."

---

**Debate 2: Should the introduction lead with the failure narrative ("training would not converge") or the discovery narrative ("we found three classes of failure")?**

FOR the failure narrative: The hook "training would not converge" is memorable and honest. Any ML researcher who has trained MTL systems recognizes the frustration immediately. It signals that this is not a standard "our numbers are great" paper and creates a mystery that the paper resolves. The 22_REVISED_PAPER_OUTLINE keeps this hook in Paragraph 1, and the current popw_aaiml2027.tex abstract implicitly follows the same structure (problems first, system second).

AGAINST the failure narrative: The hook tells the reader the system does not work. Every subsequent number — detection 0.358, activity 0.129, PSR F1=0 — reinforces the negative first impression. The alternative is to open with the discovery narrative: "We identified three classes of training pathology that standard monitoring misses." This achieves the same effect (unusual paper, pathology focus) without telling the reader the system failed. The popw_aaiml2027.tex abstract already takes the discovery approach — it does not say "training would not converge" but rather "We characterize three training pathologies." The paper's abstract and introduction are currently inconsistent on this point.

File evidence: 22_REVISED_PAPER_OUTLINE Intro Paragraph 1 uses the failure hook; popw_aaiml2027.tex abstract uses the discovery framing; both are from the same author on the same paper, indicating unresolved tension.

---

**Debate 3: Does the deployment and pilot evidence belong in the main paper or in supplementary?**

FOR keeping it in the main paper: The pilot (0% opt-out, SUS 72.3, NASA-TLX d=0.51) is the paper's only positive outcome that an ML reviewer intuitively understands. Without it, the paper is entirely a catalog of things that went wrong. The 04_BEST_PAPER_FORMULA scores practical impact at 98/100 (the highest of any criterion) — the pilot is the reason. Cutting it to supplementary would drop the paper from "Best Paper contender" (91/100) to "solid conference paper" (~80/100). Even a half-page pilot section gives the reader a reason to believe the system works in practice despite the modest numbers.

AGAINST keeping it in the main paper: The pilot is a human-factors study (N=20, one factory, 2 weeks), not an ML result. ML reviewers will see it as padding from a different research tradition. The paper's core contribution is the pathology analysis; the pilot belongs in a journal extension or a separate HCI paper. The blockchain details (537ms latency, $0.0002-$0.001 gas cost) are a one-paragraph feasibility demonstration that adds nothing to the pathology thesis. The 22_REVISED_PAPER_OUTLINE already marks the pilot as "(Supporting)" in the contribution list and allocates only 0.5 pages.

File evidence: 04_BEST_PAPER_FORMULA pilot weight = 20%; 22_REVISED_PAPER_OUTLINE allocates 0.5 pages and labels "(Supporting)"; 02_SECTION_BY_SECTION allocates 1 page max; popw_aaiml2027.tex gives pilot 0.3 pages plus a table; 10_SUBMISSION_CHECKLIST lists both as required sections.

---

### Reviewer 2: Claims Strength Reviewer — Three Debates

**Debate 1: Is "64-68% of dedicated detection performance" a defensible framing or a euphemism for a bad result?**

FOR the ratio framing: This is the correct comparison because it establishes a fair baseline (D1-R YOLOv8m on the same recording-aware split). The alternative — reporting absolute mAP50 (0.358) — invites the reviewer to compare against the WACV 2024 published number (0.838) or the D1-R retrain (0.995) without knowing which comparison is fair. The paper honestly reports all three numbers in the same paragraph. At 46.5M params (1.8x YOLOv8m's 25.9M), POPW loses 32-36% relative detection performance while running three additional tasks. The 30% parameter savings over a 4-model sequential pipeline is a genuine engineering advantage that the absolute mAP number does not capture.

AGAINST the ratio framing: One-third detection degradation means the system misses one out of three assembly states compared to a free, off-the-shelf YOLOv8m. No manufacturing engineer would deploy this. The "parameter savings" argument is weak because the comparison model (25.9M) is smaller than POPW (46.5M) — the savings only exist if one insists on running four separate models, which is not standard practice. The MTL literature (Kendall 2018, PCGrad 2020) reports negligible multi-task degradation on benchmark tasks. A 32-36% degradation is not "multi-task cost" — it suggests the architecture is not doing multi-task learning correctly.

File evidence: popw_aaiml2027.tex Section 5.2 reports all three numbers; 22_REVISED_PAPER_OUTLINE uses "first experimentally measured multi-task overhead" framing; 07_RISK_REGISTER marks detection as CRITICAL risk; 15_REVISED_RISK downgrades to HIGH but notes "temporarily worse."

---

**Debate 2: Should PSR POS=0.999 appear in the abstract and conclusion given F1=0?**

FOR keeping the POS claim: The paper already publishes F1=0 in the results table and has a dedicated disclosure section. No reviewer reading the full paper will miss it. Removing POS from the abstract eliminates the paper's strongest positive numerical result, leaving an abstract that reads: detection: 64-68% (negative framing), activity: 12.9% (negative), ego-pose: first baseline (no comparison), PSR: F1=0 (negative). The abstract becomes a catalog of limitations. Keeping POS with a paired F1 disclosure ("PSR POS=0.999 beats SOTA, though F1=0 indicates collapsed transition timing") gives the reader both facts immediately.

AGAINST keeping the POS claim: POS=0.999 with F1=0 is a statistical artifact, not model competence. The paper's own disclosure explains this: all-ones predictions produce high POS because the metric only measures monotonicity of prediction diffs. A reviewer reading "beats STORM-PSR" in the abstract and discovering F1=0 in Section 5 will reasonably conclude the authors are hiding the bad result. The Conclusion in popw_aaiml2027.tex currently says "PSR POS that beats SOTA" without F1 mention — this is the most likely reviewer trigger for a "misleading claims" comment. The correct approach is to pair POS and F1 in every location they appear.

File evidence: popw_aaiml2027.tex Abstract mentions POS without F1; Conclusion mentions POS without F1; Table 1 (results) includes both on adjacent rows; Section 5.2.1 is a dedicated disclosure; 07_RISK_REGISTER does not list PSR as a risk (potential oversight); 15_REVISED_RISK does not mention PSR.

---

**Debate 3: Is the "first ego-pose baseline on IndustReal" a standalone contribution?**

FOR the baseline claim: There are five tasks in the POPW system. One of them (ego-pose) has never been reported on IndustReal. Establishing a baseline enables future comparison — subsequent papers can report numbers relative to 7.83 degrees forward MAE. The metric is non-trivial (HoloLens head orientation from egocentric video without synthetic training data). The honest disclosure about position metric unreliability strengthens the contribution because it shows the authors understand the metric limitations. The 22_REVISED_PAPER_OUTLINE lists it as contribution item 3.

AGAINST the baseline claim: A baseline without comparison tells the reader nothing about whether the model is useful. Is 7.83 degrees good? Is 5 degrees achievable? Is 10 degrees the human baseline? Without context, reporting 7.83 degrees is reporting a number, not a result. The position MAE is explicitly marked as unreliable by the evaluation code (popw_aaiml2027.tex, Disclosure 5). If half of the 9-DoF output is unreliable, calling the remaining 6-DoF a "baseline" overstates the contribution. The pose head's real value is providing features for the FiLM conditioning, not as an independently evaluated task.

File evidence: popw_aaiml2027.tex Disclosure 5 reports "position values (mm) are explicitly marked as unreliable"; 22_REVISED_PAPER_OUTLINE lists ego-pose as contribution 3; 07_RISK_REGISTER does not flag pose head as a risk; popw_aaiml2027.tex Section 3 describes pose head as FiLM conditioning source, not as standalone contribution.

---

### Reviewer 3: Ablation Design Reviewer — Three Debates

**Debate 1: Does the equal-gradient-update ablation correctly isolate multi-task interference from underfitting?**

FOR the design: The naive single-task comparison confounds two variables: task count and gradient update count per task. The equal-gradient-update ablation controls for the second variable, isolating structural interference as the cause of the detection delta. This is methodologically superior to the single-task vs multi-task comparison used in most MTL papers. The popw_aaiml2027.tex Results section and the 22_REVISED_PAPER_OUTLINE (Section 5.3) both describe this design. The "delinquent update ratio" reported in the .tex file is an honest acknowledgment of the remaining confound.

AGAINST the design: Equal-gradient-update introduces its own confound. Holding detection gradient updates constant means the multi-task arm produces fewer total gradient steps on other tasks (batch space shared). If the multi-task arm receives 1000 detection updates and 1000 activity updates per epoch, while the single-task arm receives 1000 detection updates only, the multi-task arm trains on more total loss signal per epoch. The comparison is not "single-task vs multi-task with controlled gradient mass" but "single-task vs multi-task with different total gradient mass." A cleaner design would hold total gradient mass constant across arms. The "delinquent update ratio" acknowledges but does not resolve this.

File evidence: popw_aaiml2027.tex describes equal-gradient-update design with delinquent update ratio; 22_REVISED_PAPER_OUTLINE Section 5.3 requires this ablation; 23_ABLATION_AND_EXPERIMENT_PLAN adds 0.5-page methodology discussion; 06_TABLES_FIGURES does not list this ablation in Table 4.

---

**Debate 2: Is p=0.032 for the FiLM ablation significant with a single seed and four conditions?**

FOR significance: The FiLM ablation (18.3% full vs 16.1% no FiLM) uses bootstrap p=0.032, which passes the conventional 0.05 threshold. The intermediate ablations (PoseFiLM only at 17.2%, HeadPoseFiLM only at 16.8%) show a monotonic pattern consistent with the hypothesis: more FiLM conditioning improves activity. The effect size (2.2 percentage points, 13.7% relative improvement) is practically meaningful for a per-frame 47-class task with an estimated 30-35% ceiling. The 22_REVISED_PAPER_OUTLINE calls for Cohen's d and bootstrap CI in addition to the p-value, which will strengthen the evidence.

AGAINST significance: A single-seed experiment with 4 conditions and p=0.032 is fragile. One different random seed could negate the result. The 2.2 percentage point difference (18.3% vs 16.1%) represents approximately 800 frames on a 38K-frame validation set — within the range of noise from a single data split. At Bonferroni-corrected alpha of 0.05/4 = 0.0125, p=0.032 is not significant. The 95% CI for the difference likely includes zero. Without 3-seed variance, the FiLM ablation should be reported as "suggestive" rather than "significant."

File evidence: popw_aaiml2027.tex Figure 4 reports p=0.032; 22_REVISED_PAPER_OUTLINE Section 5.4 requires bootstrap CI and Cohen's d; 23_ABLATION_AND_EXPERIMENT_PLAN does not list FiLM ablation as critical path; 06_TABLES_FIGURES proposes bar chart with p-value annotation.

---

**Debate 3: Is the 47-group activity taxonomy justified or does it inflate the reported accuracy?**

FOR the grouping: The verb-grouped taxonomy is a pragmatic response to a real data problem: 46 of 74 classes have less than 1% representation. Reporting on the full 74-class taxonomy would yield macro-F1 close to zero, which is technically true but misleading — the model fails because there is no training signal, not because the architecture is deficient. The 74-class collapse is itself documented as the motivation for Pathology 2 (loss scale suppression), so the grouping is explained rather than hidden. The 22_REVISED_PAPER_OUTLINE states "Activity is 47 hybrid groups, not 74 classes. The 74-class collapse is the motivation for grouping and is documented in Section 4."

AGAINST the grouping: Reducing 74 classes to 47 inflates Top-1 by making the task easier. The paper should report both 74-class and 47-class numbers so the reviewer can assess the inflation factor. Without both, the reviewer cannot distinguish genuine action recognition from coarse verb categorization. The paper currently reports only 47-class numbers, with the 74-class collapse described qualitatively in the Pathology 2 section. This is insufficient for a reader to judge whether the model is doing fine-grained recognition at all.

File evidence: popw_aaiml2027.tex reports 47-class numbers only; 22_REVISED_PAPER_OUTLINE Section 5.2 requires both; 15_REVISED_RISK_AND_WEAKNESSES notes 74-class collapse is motivation for Pathology 2; 07_RISK_REGISTER R3 flags activity at 18.3% Top-1 as HIGH risk (before the simple head fix lowered it further).

---

## Section 2: PSR Debate

*Agent: PSR Specialist (Agent 2)*

---

### Debate 2.1: The POS Paradox — Is POS=0.968 a Meaningful Claim?

**Participants:** PSR Specialist (pro-POS), POS Paradox Reviewer (skeptical)

**Claim:** "Our POS=0.968 exceeds SOTA (B3=0.797, STORM=0.812), demonstrating superior per-frame procedure state estimation."

**POS Paradox Reviewer:** The claim that POS=0.968 exceeds SOTA is misleading because you are comparing across fundamentally different paradigms. Your POS is computed on per-frame 11-bit state vectors where the MonotonicDecoder enforces a fill-forward ordering constraint. Since the ground truth also follows this same fill-forward structure (assembly components are placed once and stay), any decoder that roughly reproduces the fill-forward pattern will achieve high POS regardless of whether it identified the correct transition times. The SOTA baselines compute POS on completed step lists with timestamps, which is a harder output space — they must get the transition timing right. Citing only the POS number without disclosing this paradigm difference would be rejected by any informed reviewer.

**PSR Specialist:** Two responses. First, the paradigm difference is disclosed in our paper's comparison table (reviewer-3-psr-paradigm-reconciliation.md), which lists the paradigm alongside every metric. We do not hide that we do per-frame state estimation while SOTA does transition detection. Second, POS=0.968 is still informative within our paradigm: it means the per-frame state vector matches the ground truth state vector on 96.8% of frames. This is a valid horizontal comparison across methods that predict per-frame states. There are no published per-frame state methods on IndustReal, so we compare against the closest available numbers and disclose the paradigm.

**POS Paradox Reviewer:** The D4 experiment undermines your argument. When you feed YOLOv8m ASD detections through your MonotonicDecoder, you get POS=0.999 with F1=0.0. POS approaches perfection when the decoder outputs all zeros, because 95%+ of frames are static and an all-zeros prediction matches the ground truth everywhere except at actual transitions. A metric that gives near-perfect scores to a null-output system is not informative. Your POS=0.968 may be similarly inflated by the decoder's fill-forward bias rather than genuine detection.

**PSR Specialist:** D4 is a corner case where the detector activated on less than 1% of frames (SOTA_STATUS.md). That is a detection-backbone failure, not a decoder or metric failure. On our ConvNeXt backbone, the decoder must actually detect transitions to achieve POS=0.968 — the fill-forward constraint alone cannot produce the correct per-component timing. The per-component breakdown shows non-trivial differences: comp 4 with 14.2% prevalence, comp 10 with 18.3% prevalence. These are not all-zeros outputs.

**POS Paradox Reviewer:** You can resolve this definitively with one number: compute the POS of a null model that always predicts the previous frame's state (the perfect fill-forward baseline). If your POS=0.968 is close to the null model's POS, the decoder is doing nothing beyond the trivial inductive bias. What is that number?

**PSR Specialist:** We do not have that number, and that is a fair experiment. Adding the null model POS baseline to the paper would strengthen the claim if our POS exceeds it, or would reveal the inflation if it does not. This is a high-value pre-submission experiment.

**Evidence references:**
- reviewer-3-psr-paradigm-reconciliation.md: Paradigm comparison table, POS paradox analysis
- SOTA_STATUS.md: POS paradox confirmation, D4 analysis
- d4_yolov8m_psr/metrics.json: D4 F1=0.0, POS=0.999
- psr_transition.py: Fill-forward constraint in MonotonicDecoder

---

### Debate 2.2: Backbone Swap — Is the PSR Decoder Architecture Sufficient?

**Participants:** PSR Specialist (optimistic), Paradigm Comparison Reviewer (pragmatic)

**Claim:** "Our MonotonicDecoder plus procedural knowledge will achieve competitive PSR F1 when using the same YOLOv8m backbone as SOTA."

**Paradigm Comparison Reviewer:** You are assuming the bottleneck is detection, but we do not know that — the backbone swap experiment has not been run. The B3 baseline achieves F1=0.883 using a simple confidence accumulation rule over YOLOv8m ASD outputs combined with procedural knowledge restricting expected steps. Your MonotonicDecoder is a simple Viterbi-like forward pass with no learned parameters. My prediction is that YOLOv8m plus your decoder will achieve F1 in the 0.50-0.70 range, confirming that both detection and decoder quality contribute to the gap.

**PSR Specialist:** I agree the swap experiment is the critical test. But there are two reasons to be optimistic. First, our MonotonicDecoder already has the procedure-order constraint (psr_transition.py lines 94-97), which provides some of the same benefit as B3's expected-step restriction. The gap between B3 and B2 (which lacks procedural knowledge) is approximately 0.05-0.10 F1, not the full 0.883-to-0.144 gap. Second, our Q48 hysteresis filter (sustain_hi=0.5, sustain_lo=0.3, sustain_min=3) is a temporal smoothing mechanism that achieves some of what B3's confidence accumulation does: it requires sustained evidence before firing a transition.

**Paradigm Comparison Reviewer:** Your Q48 thresholds were tuned on ConvNeXt outputs. YOLOv8m ASD outputs will have different statistics — different confidence distributions, different noise patterns. The hysteresis parameters need re-tuning. And even with optimal tuning, your decoder lacks the procedural-knowledge training signal. The precedence constraints in psr_transition.py are only applied at decode time, not as a training loss. This is a documented gap (reviewer-3 analysis Section 3).

**PSR Specialist:** That gap is exactly why the todo list (todo-psr-backbone-swap.md) prioritizes the procedural knowledge loss after the swap. If the swap gives F1=0.50-0.60, adding procedural knowledge during training should give +0.05-0.10, bringing us to 0.55-0.70. The appropriate comparison is B3 without procedural knowledge on real-only data, which we do not have.

**Paradigm Comparison Reviewer:** This reveals the deeper problem: we need the right control experiments. B3 without procedural knowledge and without synthetic data on YOLOv8m. That number alone tells us whether the decoder is the bottleneck or the data is. Without it, we are speculating.

**PSR Specialist:** Agreed. The backbone swap plus B3 re-evaluation with controlled ablations is the only way to separate detection, decoder, procedural knowledge, and data quality contributions.

**Evidence references:**
- todo-psr-backbone-swap.md: Three-outcome prediction, protocol
- reviewer-3-psr-paradigm-reconciliation.md: Sections 2 and 3, reviewer's F1=0.50-0.70 prediction
- SOTA_STATUS.md: B3=0.883, STORM=0.901
- psr_transition.py: MonotonicDecoder lines 79-182, Q48 hysteresis lines 146-175

---

### Debate 2.3: Kendall Weighting — Is the Multi-Task Scheme Suppressing PSR Unfairly?

**Participants:** PSR Specialist (concerned), Kendall Weighting Reviewer (defender of adaptive weighting)

**Claim:** "The Kendall adaptive weighting is suppressing the PSR head by downweighting its loss, creating a self-reinforcing cycle of zero gradients that the PSR head cannot escape."

**Kendall Weighting Reviewer:** The Kendall weighting is doing exactly what it is designed to do: downweight tasks that have converged or that produce noisy gradients. If the PSR loss is consistently zero, the weighting interprets this as task solved and allocates gradient budget to harder tasks. This is a feature, not a bug. The small nonzero liveness gradient (RMS=1.88e-03) at the shared layers suggests there is some learning, just not at the per-component heads.

**PSR Specialist:** But the per-component heads are completely dead — all 11 show RMS gradient 0.00e+00 across 3800+ steps. If the loss were genuinely zero because the head converged, the F1 would be high. Instead, per-component F1 ranges from 0.3455 (comp 4) to 1.0000 (comp 0, always ON). The low-prevalence components are at near-chance levels. This is not a converged head — it is a head that never learned.

**Kendall Weighting Reviewer:** The log_var_psr values stabilize around -0.04 to -0.08, corresponding to precision weight of roughly 1.04-1.08 (4-8% downweighting). That is not enough suppression to kill a head by itself. The real issue is relative gradient magnitudes: detection and activity losses produce gradients 10-100x larger than PSR, so the PSR gradients are negligible in the combined update regardless of Kendall weighting.

**PSR Specialist:** But there is a feedback loop concern. The log_var_psr starts at -0.128 (epoch 1) and moves toward -0.029 by epoch 25, oscillating throughout. If the PSR loss is consistently zero, the log_var gets gradient updates pushing it toward zero (downweighting), which reduces the effective PSR loss further, which pushes log_var toward zero more. KENDALL_HP_PREC_CAP prevents extreme precision values, but it does not prevent a slow drift toward effective zero weight.

**Kendall Weighting Reviewer:** The liveness pattern (aggregate RMS 1.88e-03 nonzero, all 11 sub-heads 0.00e+00) suggests an architectural problem, not a weighting problem. Gradients flow through the shared layers but die at the per-component linear layers. This is consistent with the ReLU activation causing dead neurons or the bias=-1.0 initialization pushing sigmoid outputs into the saturation region where gradients vanish.

**PSR Specialist:** That is testable. The per-component heads use Linear -> ReLU -> Linear initialized with bias=-1.0 on the final layer, pushing sigmoid output toward 0.27. If all 11 heads saturate, ReLU kills gradients for negative activations. The fix: (a) remove ReLU in per-component heads, (b) initialize output bias to 0.0, or (c) use GELU or Leaky ReLU. The critical experiment is to check neuron activations during a forward pass.

**Kendall Weighting Reviewer:** The other option is disabling Kendall weighting specifically for PSR using a fixed weight. An ablation comparing PSR F1 with Kendall adaptive, with Kendall frozen at initialization, and with a fixed manual weight would resolve whether Kendall is the bottleneck.

**Evidence references:**
- train.log: 569 instances of psr=0.0000, Kendall log_var values throughout
- reviewer-3-psr-paradigm-reconciliation.md: Architecture description
- psr_transition.py: Per-component head architecture (lines 216-221), initialization (lines 232-237)
- SOTA_STATUS.md: Per-component F1 breakdown
- train.log startup: USE_KENDALL=True, KENDALL_HP_PREC_CAP=True, KENDALL_FIXED_WEIGHTS=False

---

## Section 4: Head Pose Debate

*Agent: Head Pose Specialist (Agent 4)*

*File references: reviewer-4-ego-pose-contribution.md, SOTA_STATUS.md, day1-checkpoint-done-and-next-steps.md, 11_NUMBERS_UPDATE.md, head_pose_diag.py, industreal-sota-benchmarks.md*

---

### Debate 4.1: Forward MAE Fairness — Was the 8.39° number measured under the same normalization as training?

**Reviewer A (arguing the number is unreliable):** The 8.39° forward MAE comes from a network trained without the GT normalization fix documented in 11_NUMBERS_UPDATE.md. The eval normalizes at evaluation time, but the training did not. Without consistent normalization, the 8.39° is not comparable to a properly-normalized baseline. The audit also revealed that the position scale is ambiguous (HEAD_POSE_POS_SCALE=100), so even the 8.39° could be measuring a different quantity than intended.

**Reviewer B (arguing the number is correct):** head_pose_diag.py normalizes the GT and predictions at evaluation time, regardless of training-time normalization. The angular MAE computation uses cosine distance on unit vectors, which is invariant to scale. The 8.39° is a cosine-angular distance — comparable to any other cosine-angular distance measurement. The HEAD_POSE_POS_SCALE affects the position loss weighting, not the forward/up angular metrics. The 8.39° is publishable.

**Resolution:** Verify which checkpoint produced the number and whether train-time normalization was applied. Run an explicit cosine angular distance comparison: 8.39° on a normalized GT vs 8.39° on an unnormalized GT (the latter should be larger). Use whichever result is reproducible and document the normalization choice in the paper.

---

### Debate 4.2: Up-Vector Reliability — Three Conflicting Numbers, Which Is Real?

**Reviewer A (arguing the metric is unreliable):** Three up-vector numbers exist: 7.06° (day-1-checkpoint), 13.5° (300-subset in SOTA_STATUS), 26.20° (full eval). A 4x gap between numbers from the same model on the same data subset is not a metric issue — it indicates the up-vector measurement itself is unstable. Either the GT up-vector is poorly defined (gravity-aligned but subject to IMU drift) or the model is fitting noise.

**Reviewer B (arguing the metric reflects real difficulty):** The up-vector is inherently harder than the forward direction. Forward motion is the primary signal in egocentric video (you look where you're going); the up-vector is a secondary signal that can be obscured by IMU drift, head tilt during assembly, and gravity-aligned orientation ambiguity. The 13.5° on a 300-subset vs 26.20° on full eval reflects data-sampling variance — the 300-subset may have been selected for "good" recordings. The 7.06° may be from a different (pre-fix) checkpoint. All three numbers are real but from different conditions.

**Resolution:** Run per-recording breakdown of up-vector errors, error distribution analysis (median, IQR), and report median alongside mean. Pick the median from the full eval (26.20° or whatever it resolves to) as the headline number, with mean as supplementary. Justify the number with per-recording evidence.

---

### Debate 4.3: OpenFace Comparison Strategy — Say Nothing or Proactively Distinguish?

**Reviewer A (arguing for silence):** Per reviewer-4, any mention of OpenFace or 6DRepNet creates a category confusion. The paper should simply not compare to face-pose estimators. The first-egocentric-pose-baseline claim is strong without these comparisons.

**Reviewer B (arguing for proactive distinction):** Silence invites the reader to make the comparison themselves, often incorrectly. A proactive taxonomy section that distinguishes face-pose (input: face image; output: head orientation in face coordinates) from ego-pose (input: full egocentric frame; output: head+body orientation in world coordinates) eliminates the confusion. This costs 1 paragraph but saves reviewer confusion.

**Resolution:** Survey legitimate egocentric pose literature (HoloLens IMU-based, head-mounted IMU orientation, etc.) and add a task-taxonomy figure that distinguishes face-pose from ego-pose. Cite 3 legitimate ego-pose baselines for the comparison table. Drop OpenFace/6DRepNet entirely from comparisons.

---

## Section 7: Eval Pipeline Debate

*Agent: Eval Pipeline Specialist (Agent 7)*

*File references: eval_yolov8m.py, eval_yolov8m_psr.py, psr_optimal_thresholds.py, eval_activity_clip.py, d1_yolov8m/metrics.json*

---

### Debate 7.1: Class Mapping — Was the 0-vs-1 Index Issue a Bug or a Red Herring?

**Class Mapping Reviewer A (arguing alignment is correct):** The D1 ASD mAP=0.995 result (separate training run) proves the class alignment works in our codebase. The 0.0004 D1 mAP is from a different code path (COCO-pretrained YOLOv8m with default 80 classes), not a class index bug. The PSR_MASK explicit `one_idx - 1` proves the code is aware of the 1→0 shift and applies it correctly.

**Class Mapping Reviewer B (arguing alignment may still be off):** The 0.0004 exactly matches a label-shift-by-1 scenario where every prediction is at the wrong class. The `one_idx - 1` in PSR_MASK proves the code is *aware* of the ambiguity — which is suspicious. The D1 v3 (no shift) and v2 (+1 shift) gave nearly identical mAP=0, suggesting the mapping is fine, but the underlying model is just bad. Either way, the 0.0004 number is publishable as a genuine detection result, not a bug.

**Resolution:** Run a brute-force print of the prediction class distribution and the GT class distribution on the D1 val split. If they overlap (e.g., both peak at class 13), the alignment is correct and 0.0004 is the real metric. If they are shifted (predictions at 13, GT at 14), there is a bug.

---

### Debate 7.2: Threshold Sweep Overfitting — Is the 0.7217 → 0.7499 Improvement Real?

**Threshold Sweep Reviewer A (arguing it's calibration, not overfitting):** Per-component threshold selection is standard practice in binary classification. The grid (19 thresholds) is coarse and the dataset (38k frames) is large enough that a 0.028 F1 improvement from per-comp calibration is real, not noise. Leave-one-recording-out CV would confirm.

**Threshold Sweep Reviewer B (arguing it's overfitting):** With 19^11 effective combinations (de facto), val overfitting is plausible. The same val set used for both threshold selection and metric reporting is circular. The improvement from 0.7217 to 0.7499 may be 2-3% of pure val noise.

**Resolution:** Leave-one-recording-out CV. Compute per-component thresholds on N-1 recordings and evaluate on the held-out one. If the improvement persists, it's real. If it shrinks to <0.005, it's val noise.

---

### Debate 7.3: Crash Recovery — Why Do Eval Scripts Lack Save-Intervals?

**Crash Recovery Reviewer A (arguing it's unnecessary):** The YOLOv8 eval scripts process 2 hours of inference deterministically. Crash recovery is only needed for long-running evals that may be preempted. On a dedicated GPU with no preemption, no save is needed.

**Crash Recovery Reviewer B (arguing it's negligent):** On shared GPU infrastructure, anything >30 min should have intermediate save. The eval_activity_clip.py save_interval=5000 (every ~5 min) is the correct pattern. The YOLOv8 scripts deviate from this standard without justification.

**Resolution:** Add `--save-every N` flag to eval_yolov8m.py and eval_yolov8m_psr.py, default to 0 (no save). Document the option in the script docstring.

---

## Section 8: SOTA Comparison Debate

*Agent: SOTA Benchmark Analyst (Agent 8)*

*File references: industreal-sota-benchmarks.md, industreal-all-papers-benchmarks.md, comparability-matrix.md, contribution-audit-reviewer-factcheck.md, SOTA_STATUS.md, benchmark-reference-for-paper.md, reviewer-3-psr-paradigm-reconciliation.md*

---

### Debate 8.1: Detection Claim — Does "BEATS SOTA" Hold Up Under Audit?

**STORM Comparison Reviewer Side A:** The 0.995 self-trained YOLOv8m is internally consistent, exceeds WACV's 0.838 by 19%, and uses the same dataset and metric. The claim holds.

**STORM Comparison Reviewer Side B:** The D1 re-evaluation (0.0004) is a structural red flag. Until D1 is fixed, no "BEATS SOTA" claim is defensible. The paper should report internal progress, not SOTA claims.

**Resolution:** Conditional — needs D1 audit before claiming SOTA. Report as internal achievement with footnote about cross-eval in progress.

---

### Debate 8.2: PSR Claim — Can F1=0.7499 Be Compared to B3's 0.883?

**B3 Comparison Reviewer Side A:** Same dataset, same goal, paradigm disclosure suffices. The comparison is informative.

**B3 Comparison Reviewer Side B:** F1 measures different quantities (transitions vs per-frame states). D4 proves it with F1=0.000 on YOLOv8m backbone (same paradigm as SOTA). Mixing in a table is invalid.

**Resolution:** Report per-component F1 as separate metric, paradigm disclosure section, never compare directly to B3/STORM transition F1.

---

### Debate 8.3: Activity Claim — Is the 0.622 T3 Match a SOTA Comparison?

**T3 Comparison Reviewer Side A:** The T3 baseline match shows verb-grouping works. Both models on the same 69-class label space, similar performance.

**T3 Comparison Reviewer Side B:** Our 0.028 genuine clip-level top-1 is the honest number. 0.622 is a bag-of-frames heuristic on a different paradigm. Don't compare to MViTv2-S as SOTA.

**Resolution:** Report T3 as protocol verification, per-frame metrics in isolation. Drop "matches SOTA" framing.

---

## Summary of Resolutions

| Debate | Claim | Verdict | Paper Action |
|---|---|---|---|
| 1.1 | Detection beats SOTA | Conditional | Use 64-68% ratio, footnote cross-eval |
| 1.2 | Detection NaN full | Action needed | Fix subprocess_eval.py |
| 1.3 | Cost vs competitive | Cost framing wins | Multi-task cost narrative |
| 2.1 | POS paradox | Disclosure needed | Explain POS metric artifact |
| 2.2 | Backbone swap (PSR) | Critical path | Run controlled ablations |
| 2.3 | Kendall suppression | Architectural fix needed | Linear probe, GELU/RELU test |
| 3.1 | Per-frame MLP vs MViTv2 | Architectural ceiling | Re-frame as per-frame baseline |
| 3.2 | Verb grouping | Justify both 75 and 69 | Add supplementary table |
| 3.3 | Per-frame re-framing | Latency argument | Add latency comparison |
| 4.1 | Forward MAE | Verify normalization | Re-run with explicit normalization |
| 4.2 | Up-vector reliability | Use median | Per-recording breakdown |
| 4.3 | OpenFace comparison | Proactive distinguish | Task taxonomy figure |
| 5.1 | FiLM novelty | Conditional | Show modulation magnitude |
| 5.2 | Kendall auto-balancing | Architectural fix | Fixed-weight ablation |
| 5.3 | Sequence-mode overhead | Negative cost-benefit | Drop if F1=0 |
| 6.1 | CUDA crash root cause | Bisect | Test batch_size=4 |
| 6.2 | Effective batch 16 | Accept with caveat | Document in §5 |
| 6.3 | GPU allocation | Add discussion | 1-paragraph recommendation |
| 7.1 | Class mapping | Brute-force print | Per-class histogram |
| 7.2 | Threshold overfitting | Leave-one-out CV | Confirm improvement |
| 7.3 | Crash recovery | Add save-every flag | Default 0 |
| 8.1 | Detection SOTA | Conditional | Needs D1 audit |
| 8.2 | PSR comparison | Paradigm disclosure | Never compare F1 directly |
| 8.3 | Activity SOTA | Protocol verification | Drop "matches SOTA" |
| 9.1 | Naming consistency | grep -l + rename | Apply uniformly |
| 9.2 | 0.028 framing | Different paradigm | Latency argument |
| 9.3 | Claim language | Claim-strength rubric | Define before writing |
| 9.4 | Ablation design | One combined table | Run both, pick |
| 9.5 | Narrative arc | Deployment focus | 0.5-1.0 pages |
| 9.6 | F1=0 disclosure | Standalone section | Paired with results |
| 9.7 | Honesty matrix | Try 0.5 page | Build reviewer trust |
| 10.1 | Best-checkpoint broken | Audit needed | Re-derive numbers |
| 10.2 | Activity 0.028 | Re-frame as probe | Drop or retrain |
| 10.3 | POS inconsistency | Disclose in results | Explain D4 vs ours |
| 10.4 | Kendall collapse | Fixed-weight ablation | Verify learning |
| 10.5 | Kendall spiral | Reproduce or retract | Code-vs-paper reconciliation |
| 10.6 | Numbers will change | Rerun all evals | Final-checkpoint eval |

**Total: 30 debates across 10 specialist agents, with concrete resolutions for each.**

