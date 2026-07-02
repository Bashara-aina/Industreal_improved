# 50 Deep Technical Questions for Opus

> Status: Generated 2026-07-02
> Purpose: Diagnose open architectural, loss-balancing, and training-infrastructure questions in the POPW multi-task model.
> Each question is self-contained with specific code references and measured values.

---

## Section 1: Architecture & Multi-Task Dynamics (Q1-Q10)

### Q1: The TMA Cell Parameter Paradox

The TMA (Temporal Masked Attention) Cell is described as a GRU-based module (config.py line 146: `USE_TMA_CELL = True`). A standard GRU has three gate matrices (update, reset, new) each with dimension `(input_size + hidden_size) * hidden_size`. For the declared shapes in model.py (512-dim input, 256 hidden), this should produce ~590K parameters. However, the param count table in the architecture docs claims "0 params" for the TMA Cell. Is this cell actually functional, or is it a structural placeholder that was implemented as a no-op identity transformation? If it has 0 parameters, it cannot gate temporal information. If it does have parameters, the 0-param claim is a documentation error that needs correction before paper submission.

### Q2: Backbone Gradient Composition by Task

The combined gradient norm for the 5-head model was measured at ~7.3 (before GRAD_CLIP_NORM was raised to 5.0). What fraction of this gradient comes from each head? Specifically: is detection contributing the majority of backbone gradient (as its loss magnitude 1.2-2.6 would suggest), or is head pose (loss 0.1-1.5 with Kendall precision up to 54.6x) actually dominating the backbone update direction despite its small raw loss? The KENDALL_HP_PREC_CAP (losses.py line 1683) prevents head pose precision from exceeding detection precision, but with detection log_var starting at 0.0 and head pose log_var starting at -1.0, the head pose precision starts at exp(1) = 2.7x vs detection at exp(0) = 1.0x. Without actual gradient cosine similarity or per-task gradient norm logs, the backbone update direction is a black box.

### Q3: PSR Gradient Isolation Undermines Multi-Task Thesis

`detach_psr_fpn=True` (model.py line 2082-2088) means PSR gradients never flow into the shared FPN (p3, p4, p5 tensors are detached for PSR branches). Combined with PSR_SEQ_EVERY_N_BATCHES=2 (every other batch is PSR-only), on seq batches the backbone receives zero PSR gradient. On non-seq batches, PSR loss is 0.0 (as observed in training logs). This means PSR effectively never shapes backbone features. If the paper's central claim is that multi-task learning benefits all tasks through shared representations, and PSR never contributes to those shared representations, doesn't this undermine the multi-task thesis for the PSR task specifically? The paper would need to show PSR benefits from shared features WITHOUT contributing gradient — a one-way transfer that contradicts the multi-task motivation.

### Q4: 10x LR Differential After Convergence

The backbone LR is 5e-5 (BASE_LR * 0.1, optimizer.py line 37) and the head LR is 5e-4 (BASE_LR, optimizer.py line 38). This 10x differential was justified during early training when the ImageNet-pretrained backbone needed gentle fine-tuning while heads needed rapid convergence from random init. Now at epoch 2 with pose converged (loss 0.1-1.5), detection stable (c-loss 1.2-2.6), and activity warming up, should this differential be reduced to 2-3x? The concern is that heads may have plateaued at suboptimal performance because the backbone cannot adapt quickly enough at 5e-5 to provide discriminative features for all 4 tasks. Evidence: activity top-1 remains near zero at epoch 2, and if the backbone features are still primarily ImageNet-generic, the activity head has nothing discriminative to classify.

### Q5: Feature Bank Gradient Re-Enabling

`FEATURE_BANK_DETACH = True` and `FEATURE_BANK_DETACH_GRAD_ENTRIES_ONLY = True` (config.py lines 1050-1051). The current configuration detaches all stored bank entries but keeps gradient on the current frame. The bank_i tensor is constructed as `torch.cat([bank_i[:-1].detach(), feat_i.unsqueeze(0)], dim=0)` (model.py line 1244), meaning exactly 1 of 16 time steps has gradient. The ViT+TCN then processes this single gradient position among 15 zero-gradient positions. What would it take to re-enable full gradient flow through the bank? The comment says `FEATURE_BANK_DETACH=False` causes a "backward through graph a second time" crash (config.py line 1046-1047). Is this a fundamental autograd limitation (same feature tensor used in multiple time steps), or could it be fixed with `torch.autograd.grad(materialize_grads=True)` or by cloning features before bank insertion?

### Q6: Three Temporal Mechanisms — Complementary or Conflicting?

The model has three independent temporal mechanisms:
1. **TMA Cell** (GRU, model.py USE_TMA_CELL=True) — operates on backbone C5 features
2. **Feature Bank** (ring buffer of 16 feature vectors, model.py USE_TEMPORAL_BANK=True) — provides temporal context to activity head
3. **PSR Sequence Mode** (8-frame sequences, config.py PSR_SEQUENCE_LENGTH=8, PSR_SEQ_EVERY_N_BATCHES=2) — provides temporal context to PSR head

Are these three mechanisms providing complementary temporal information at different scales, or are they conflicting? Specifically: the Feature Bank operates on every batch with shuffled frames (no temporal coherence), while PSR Sequence Mode operates every 2nd batch with true consecutive frames. The TMA Cell operates on whichever frame arrives. A principled analysis of what temporal signal each mechanism actually receives under the current sampler (WeightedRandomSampler with class balance, frame-level shuffling) would clarify whether any of the three mechanisms can learn from temporally coherent signals at all.

### Q7: Selective PSR Backbone Gradient Enablement

Currently, PSR backbone gradients are completely disabled (detach_psr_fpn=True + PSR_SEQ_EVERY_N_BATCHES=2 means PSR never updates backbone features). An alternative: enable PSR backbone gradients ONLY on seq batches (every 2nd batch) with a reduced LR, say 10% of the normal backbone LR (5e-6). This would give PSR SOME influence on backbone features while limiting disruption to detection. The PSR_WEIGHT=10.0 (config.py line 808) already amplifies PSR loss before Kendall. If backbone gradients were enabled, would the effective PSR backbone gradient be (10.0 * precision_weight * PSR_loss) at 5e-6 LR — approximately what range of backbone update? Is this worth implementing, or would the gradient disruption risk outweigh any potential PSR F1 gain above the current ~0.0?

### Q8: Verb-Grouping Information Loss

The verb-grouping collapses 75 raw action classes into ~41 outputs using a hybrid scheme (config.py ACT_CLASS_GROUPING='hybrid', ACT_HYBRID_THRESHOLD=100). Classes with >=100 frames retain their identity; classes below 100 frames are grouped by their verb prefix (first underscore token). How many of the 34 merged classes share a verb group with a >=100-frame "standalone" class? For example, if "take_short_brace" (63 frames, below threshold) is grouped with "take_screw" (which may or may not be standalone), then the resulting group has ambiguous class boundaries. What fraction of the dataset's frames fall into shared-verb groups where the verb-group label maps to multiple distinct raw actions? If this fraction exceeds 30%, the target of 55-63% Top-1 accuracy may be structurally unreachable because the task itself has label ambiguity.

### Q9: PSR Sequence Length Mismatch

The Feature Bank has a window of 16 frames (FEATURE_BANK_WINDOW=16, config.py line 148) representing 1.6 seconds at 30 FPS. The PSR sequence uses 8 frames (PSR_SEQUENCE_LENGTH=8, config.py line 1014). With stride=1 at 30 FPS, 8 frames = 0.27 seconds, which is much shorter than the median action duration. Should the PSR sequence length be increased from 8 to 16 with stride=3 (covering 16*3/30 = 1.6 seconds, matching the Feature Bank window)? The PSR architecture (d_model=256 Transformer, model.py) can handle 16 tokens with O(n^2) attention cost of 256 vs 64 — a 4x increase in attention computation but only on seq batches (every 2nd batch). Is attention cost or Transformer capacity the binding constraint?

### Q10: Gradient Starvation in FiLM Modules

`HeadPoseFiLM` and `PoseFiLM` (model.py lines 2037, 2162) modulate C5 features based on head pose and keypoint inputs. The head pose tensor is detached before FiLM (`head_pose.detach()`, model.py line 2162) and keypoint confidence was detached earlier (`conf_flat = confidence.detach()`, model.py line 696). This means the FiLM modules receive upstream features with full gradient (C5_mod) but conditioning inputs with zero gradient (detached pose/confidence). Are the FiLM scaling and shifting parameters (gamma, beta) receiving useful gradient through the modulated output? The gradient path is: FiLM_output -> activity/detection loss -> FiLM gamma/beta. If the activity head gradient was measured at 0.012 (30x below detection before the fix), the FiLM gradient at 0.012 * blend_ratio may be too weak to learn meaningful conditioning. Is the pose conditioning actually shaping features, or are the FiLM parameters stuck at their initialization?

---

## Section 2: Loss Balancing & Kendall Analysis (Q11-Q20)

### Q11: Missing Kendall log_var Diagnostics

Despite 4 log_var parameters being central to multi-task balancing (log_var_det, log_var_pose, log_var_act, log_var_psr, losses.py lines 1027-1035), these values are NEVER logged in the standard training output. The loss logging shows "det=2.57, pose=8.38, act=1.02, psr=0.78" but these are raw losses before Kendall weighting. Without knowing the actual log_var values, we cannot tell:
- Is PSR being suppressed (log_var_psr pushed toward +inf)? 
- Is activity precision being clamped by KENDALL_LOG_VAR_MIN_ACT=-0.5?
- Is the KENDALL_HP_PREC_CAP actively capping head pose precision?

This is a critical diagnostic gap. The training loop (train.py) should log log_var values at every LIVENESS_EVERY=500 intervals. Can this be added in the next training restart?

### Q12: PSR log_var Upper Bound Active?

`KENDALL_LOG_VAR_MAX_PSR = 0.0` (config.py line 929) means PSR precision is clamped at exp(0) = 1.0 — PSR can never be suppressed below equal weighting. But is PSR actually hitting this bound? If log_var_psr is being pushed toward +inf (which would suppress PSR to zero weight), the clamp at 0.0 is active. If log_var_psr is oscillating around 0.0 or negative (precision > 1), the clamp is never triggered. Without log_var_psr being logged (Q11), we cannot tell. The clamp's effectiveness depends entirely on whether PSR's gradient is pushing log_var toward suppression. Given that PSR loss is zero on non-seq batches and the log_var receives gradient from the Kendall regularization term (0.5 * log_var for each task), is log_var_psr being driven toward -inf (infinite precision) on non-seq batches because the task loss is zero?

### Q13: Fixed PSR Weight Defeats Kendall

`PSR_WEIGHT = 10.0` (config.py line 808) is a FIXED multiplier applied before Kendall weighting (losses.py line 1785: `psr_contribution = 1.0 / (2.0 * sigma_psr**2) * total_loss_psr + torch.log(sigma_psr)` — but PSR_WEIGHT is applied to total_loss_psr before the Kendall formula). This hybrid approach (fixed 10x + learned log_var) means: even if Kendall tries to suppress PSR (log_var_psr -> +inf), the fixed 10x multiplier counteracts it. Conversely, if PSR is performing well and Kendall wants to boost it (log_var_psr -> -inf, precision -> +inf), the fixed 10x may overshoot. Is this hybrid approach intentional or a design accident? If PSR_WEIGHT were removed and the Kendall system left to balance PSR naturally with `PSR_WEIGHT=1.0`, what would the expected PSR weight be given the observed loss magnitudes?

### Q14: PSR Batch Weight Discrepancy

PSR receives 20x amplification on seq batches (PSR_WEIGHT=10.0 * PSR_SEQ_LOSS_SCALE=1.5 * 1/0.75 for alternating batch schedule) but 0x on non-seq batches. The effective per-batch weight averaged over 2 batches is approximately 10x detection's per-batch gradient signal. However, this averaging is not how optimization works — the optimizer takes separate steps on seq and non-seq batches. On seq batches, PSR dominates with 20x amplification while other heads get zero gradient. On non-seq batches, PSR gets zero gradient while detection/activity/pose train normally. Is this on/off cycle causing optimizer instability? The AdamW state (first and second moments) for shared backbone parameters is updated with completely different gradient distributions on alternating steps. Should the schedule be changed to PSR_SEQ_EVERY_N_BATCHES=3 (2 non-seq, 1 seq) to give detection more consecutive update steps?

### Q15: PSR log_var Divergence on Non-Seq Batches

On non-seq batches, PSR loss is 0.0 (no PSR computation). The Kendall total loss includes `0.5 * log_var_psr` regularization (the log(sigma) term). With task_loss_psr ≈ 0, the gradient on log_var_psr comes entirely from the regularization term: d/d(log_var) [0.5 * log_var + task_loss / (2*exp(log_var))] = 0.5 + 0 = 0.5. This pushes log_var_psr toward -inf (infinite precision). On seq batches, the task_loss_psr term (non-zero) counteracts this. The equilibrium depends on the ratio of seq to non-seq batches. At PSR_SEQ_EVERY_N_BATCHES=2, this is 1:1. Is the log_var_psr oscillating between two values (low on seq, diverging on non-seq)? If so, the clamp at KENDALL_LOG_VAR_MAX_PSR=0.0 is actively holding log_var_psr at the boundary. Should PSR log_var be frozen on non-seq batches (detach in the loss computation when train_psr=False)?

### Q16: Activity Gradient Starvation Relative to PSR

Activity gets ACTIVITY_LOSS_WEIGHT = 0.8 (config.py line 889) while PSR gets PSR_WEIGHT = 10.0 (before Kendall). Activity also has ACTIVITY_HEAD_GRAD_CLIP = 1.0 (config.py line 879). The activity gradient BEFORE the gradient-flow fix was measured at ~0.012 (30x below detection). If the gradient fix restored activity gradient to ~0.48 (comparable to detection), then the effective gradient ratio is:
- Activity: 0.48 * 0.8 (loss weight) / 1.0 (grad clip cap) = 0.384 effective
- Detection: ~0.48 * 1.0 (loss weight) / 5.0 (global clip) = 0.096 effective

Activity would have 4x MORE effective gradient than detection despite having 1/20th the PSR weight multiplier. Is this correct? Should ACTIVITY_LOSS_WEIGHT be increased from 0.8 to 2.0-5.0 to compensate for the higher detection gradient potential?

### Q17: KENDALL_LOG_VAR_MIN_ACT Constraint

`KENDALL_LOG_VAR_MIN_ACT = -0.5` (config.py line 928) allows activity precision to reach exp(0.5) = 1.65x but no higher. Given that activity has the highest target weight in the combined metric (0.35), is this -0.5 bound too restrictive? If activity log_var were allowed to reach -2.0 (precision = 7.4x), activity would contribute 7.4x more to the total weighted loss, potentially accelerating its convergence. The concern is that activity (still near random at epoch 2) would dominate and suppress other tasks. But with the gradient fix (activity gradient restored to ~0.48) and ACTIVITY_HEAD_GRAD_CLIP=1.0, is runaway suppression actually possible? What is the theoretical precision bound before activity gradients cause other task losses to diverge?

### Q18: Pose Weight in Combined Metric is Negligible

The combined metric (train.py lines 2365-2367) gives pose a weight of `_W_POSE / total_active_w = 0.15 / sum(active)`. When all 4 tasks are active, pose contributes 15% of the combined metric. However, the combined metric is used for checkpoint selection and gate decisions. With pose contributing only 15%, a model with excellent pose but poor detection/activity/PSR could still have a low combined metric. Conversely, improving pose is only 15% effective at increasing the combined metric. Given that head pose has converged (training loss 0.1-1.5, validation MAE ~14 deg), is the 0.15 weight essentially wasted — it adds noise to the combined metric without distinguishing between good and bad checkpoints? Should the combined metric weights be revised to 0.35/0.35/0.05/0.25 (det/act/pose/psr) to concentrate the metric on the heads that need improvement?

### Q19: Active-Weight Renormalization Invalidates Cross-Stage Comparisons

The combined metric renormalizes by `total_active_w` (train.py line 2345-2351). In RF3 (det+pose+act: sum = 0.30+0.15+0.35 = 0.80), the combined metric is divided by 0.80. In RF4 (all four: sum = 1.0), the combined metric is divided by 1.0. This means a combined metric of 0.20 in RF3 is NOT comparable to 0.20 in RF4 — the RF3 value is inflated by 1/0.80 = 1.25x. The gate criteria (RF3 requires combined >= 0.80 * RF2_best) are therefore systematically incomparable across stages. Is this renormalization intentional? A principled fix would either (a) always divide by 1.0 (the maximum possible weight sum) so cross-stage comparisons are valid, or (b) never renormalize and document that combined metric ranges differ by stage.

### Q20: Alpha=0.25 with Gamma_Pos=0.0 — Positive Gradient Suppression

At `FOCAL_ALPHA = 0.25` (config.py line 667) and `DET_GAMMA_POS = 0.0`, the focal loss for a positive sample is:
```
FL_pos = -alpha * (1-p)^0 * log(p) = -0.25 * log(p)
```
For a negative sample:
```
FL_neg = -(1-alpha) * p^1.5 * log(1-p) = -0.75 * p^1.5 * log(1-p)
```

At the bias init (p ≈ 0.033 from sigmoid(-3.48)):
- Positive gradient: 0.25 * (1-0.033)^0 = 0.25 (dFL/dx ≈ alpha * (1-p)^gamma)
- Negative gradient: 0.75 * 0.033^1.5 = 0.75 * 0.006 = 0.0045
- Ratio: 55x positive vs negative

At p = 0.5 (ambiguous prediction):
- Positive gradient: 0.25 * (0.5)^0 = 0.25
- Negative gradient: 0.75 * (0.5)^1.5 = 0.75 * 0.354 = 0.265
- Ratio: 0.94x positive vs negative (negatives ALMOST dominate)

At p = 0.9 (confident positive):
- Positive gradient: 0.25 * (0.1)^0 = 0.25
- Negative gradient: 0.75 * (0.9)^1.5 = 0.75 * 0.854 = 0.640
- Ratio: 0.39x — **positives are 2.6x weaker than negatives**

With alpha=0.25 (standard for RetinaNet on COCO with 80 classes and gamma=2), the alpha was designed to compensate for the foreground-background imbalance. With gamma_pos=0 (no positive suppression), the positive gradient at high confidence is always 0.25. But negatives at high confidence (p near 1) have gradient 0.75 * p^1.5 = 0.64. This is a 2.6x NEGATIVE DOMINATION. Should alpha be raised to 0.50 or 0.75 to compensate? At alpha=0.75: positive = 0.75, negative = 0.25 * 0.9^1.5 = 0.21, ratio = 3.5x positive. This would give positives a consistent advantage across all confidence levels.

---

## Section 3: Detection Head Deep Analysis (Q21-Q30)

### Q21: Classification Score Floor at Bias Init

The detection classification head's score_p50 remains at 0.036 (measured value) after 5000+ batches of training. The bias init is -3.48 (from pi=0.03, model.py DetectionHead._init_weights() line 542-544: `pi = 0.03; bias = -log((1 - pi) / pi) = -3.48`). Sigmoid(-3.48) = 0.033, and the observed score_p50 = 0.036 is essentially identical. This means 50% of predictions are at or below the bias init value — the classification head has NOT separated scores from the initialization for any class. Is FocalLoss(gamma=2) the cause? With gamma=2, a positive sample at p=0.5 gets modulating factor (1-0.5)^2 = 0.25, reducing its gradient by 4x. From the gradient analysis in Q20, well-classified positives get 2.6x less gradient than negatives at gamma_pos=0. Is the classification head's gradient so weak that it CANNOT escape the bias init, even with correct localization (bestIoU_max=0.923 from detection probe)?

### Q22: Gamma Annealing for Detection

If FocalLoss(gamma=2) is suppressing the classification gradient so severely that scores cannot separate from the bias init, would gamma=0 (standard cross-entropy) or gamma=1 for the first 2-3 epochs help the classification head escape the p≈0.033 floor? The concern with gamma=0 is that the 173K:1 positive-negative anchor ratio would overwhelm the classifier with negatives. But at init, most negatives are easy (confidence near 0), and focal loss's main benefit is reducing the contribution of easy negatives. For the first few epochs, easy negative suppression may not be critical — the model needs to learn ANY discriminative features first. After scores separate (p > 0.1), gamma=2 can be enabled to focus on hard examples. Is a gamma scheduler (gamma = min(2, epoch * 0.67)) feasible in the current codebase, or would it require modifying the FocalLoss implementation in losses.py?

### Q23: DET_EVAL_SCORE_THRESH Revision History

The evaluation score threshold has been through 7 revisions (config.py lines 627-646): 0.5 -> 0.0 -> 0.05 -> 0.03 -> 0.1 -> 0.02 -> 0.001. Each change was driven by a specific failure mode (zero predictions at 0.5, false-positive flood at 0.0, etc.). The current value of 0.001 matches YOLOv8's reporting threshold and allows comparability. However, at threshold 0.001, the mAP computation evaluates 0.1% of 1.3M+ anchors (about 1300 per image). With 300 max detections per image (DET_EVAL_MAX_PER_IMAGE=300, config.py line 647), the NMS operates on 300 candidates. Is 0.001 the right final value, or is it a band-aid that masks the underlying classification score problem (Q21)? At which epoch does the optimal threshold change? A model at epoch 3 with score_max=0.076 should use a different threshold than a converged model at epoch 30 with score_max > 0.5. Should DET_EVAL_SCORE_THRESH be scheduled as a function of epoch or score distribution?

### Q24: Asymmetric Gamma Theoretical Motivation

`DET_ASYMMETRIC_GAMMA = True` with `gamma_pos=0.0, gamma_neg=1.5` (config.py lines 732-734). The comment explains this empirically: "gradient at p=0.074, gamma=1.0 gives 0.074 effective weight per negative (13.5x increase)... gamma=1.5 gives p^0.5=0.27 -> 3.5x moderate increase." However, there is no citation or theoretical justification. In RetinaNet (Lin et al., ICCV 2017), gamma=2 was justified by the need to suppress easy negatives in a 1:1000 positive-negative ratio. With gamma_pos=0 (no positive suppression), the positive gradient retains the full CE magnitude. But at gamma_neg=1.5, hard negatives (p near 0.5) get 0.5^1.5 = 0.354 modulation — still significantly suppressed. What is the theoretical justification for gamma_pos=0 vs. the paper's gamma=2 on the positive side? Is there a source (Rota et al., 2022 or similar) that demonstrates asymmetric gamma for dense object detection, or is this a heuristic derived from empirical observation of this specific dataset?

### Q25: pi Discrepancy — Init vs Reinit

`DetectionHead._init_weights()` (model.py line 542) hardcodes `pi = 0.03`, giving bias = `-log((1-0.03)/0.03) = -3.48`. However, the reinit mechanism uses `REINIT_PI = 0.01` (config.py line 787), giving bias = `-log((1-0.01)/0.01) = -4.60`. The sigmoid values differ by ~1.2% (0.033 vs 0.010). This 3x difference in pi means a reinit'd detection head starts with a 3x lower positive prior. Is this discrepancy intentional? The comment says "REINIT_PI = 0.01 # cls_score bias prior for reinit (RF1 uses 0.05)" suggesting different values for different stages. But in practice, the bias determines the initial score floor that the model must push predictions above. At -4.60, the initial score is 0.010 — essentially all predictions start at 1% confidence. Combined with FocalLoss(gamma=2) that suppresses positive gradient, a 1% starting confidence may be too low to ever escape. Should REINIT_PI be unified with _init_weights() to pi=0.03?

### Q26: Detection Head as Free Rider

The detection head's per-batch gradient norm is approximately 0.001 of the backbone gradient norm (ratio estimated from liveness probes). This means the detection head is extracting features from the backbone but contributing negligibly to backbone feature learning. Is detection a "free rider" on features shaped by other heads (pose, activity)? The concern: if other heads' tasks don't require object-discriminative features, the backbone will optimize for pose estimation (smooth features) and activity recognition (semantic features) while object detection (LOCALIZATION features) receives no backbone pressure. The detection head can only use whatever features happen to be there. This would explain the persistent low mAP — the detection head is doing its best with features that weren't shaped for its task. Would enabling detection backbone gradients with a higher DET_LR_MULTIPLIER (say 2.0) shift the backbone toward more localization-discriminative features?

### Q27: OHEM Effectiveness

`DET_OHEM_ENABLED = True` (config.py line 719) with `DET_OHEM_RATIO = 2.0` and `DET_OHEM_MIN_NEG = 32` (config.py lines 720, 723). OHEM (Online Hard Example Mining) keeps the top-2:1 hardest negative examples per positive. With ~12 positive anchors per GT (from DET_POS_IOU_TOP_K=9 at IoU > 0.2 floor), and maybe 1-3 GT boxes per image, there are ~12-36 positive anchors. OHEM would keep 24-72 hardest negatives. With MIN_NEG=32, the floor is 32. How many samples are actually kept per image? The OHEM mask in losses.py selects the top-k hardest negatives by focal loss. With 173K total anchors per image, keeping only 24-72 negatives means preserving only 0.04% of negatives. Is the OHEM selection so aggressive that it creates a sparse gradient signal? At 72 negatives vs 173K, the gradient variance would be extremely high, possibly causing the classification head to oscillate rather than converge. Should MIN_NEG be increased to 256 or 512?

### Q28: Positive Anchor Matching Parameters

`DET_POS_IOU_THRESH = 0.4`, `DET_POS_IOU_TOP_K = 9`, `DET_POS_IOU_IOU_FLOOR = 0.2` (config.py lines 483-489). These parameters control how many positive anchors each GT gets. The combination means: (a) any anchor with IoU > 0.4 is positive, (b) the top-9 anchors by IoU (if > 0.2) are force-matched as positive, (c) the single best anchor is always positive regardless of IoU floor. For typical IndustReal objects (h≈156px at 720p, anchor sizes 96-512), how many anchors clear each threshold? At IoU >= 0.4 with 5 FPN levels and 9 anchors per location, the effective positive count per GT is approximately (anchor_locations_with_IoU>0.4) + (top_9_from_floor). Are we getting 12-36 positives per image (as in Q27) or fewer? Can we instrument `POS_ANCHOR_PROBE_EVERY=1000` to log the actual distribution?

### Q29: Alpha-Per-Class Mechanism

The config mentions `ALPHA_PER_CLASS` (from the FocalLoss implementation in losses.py) but no classes have custom alpha values. The global `FOCAL_ALPHA = 0.25` applies uniformly. In the IndustReal dataset, class frequencies for the 63 detection classes follow a long-tail distribution (some classes appear in hundreds of frames, others in single digits). Should tail classes get a higher alpha (e.g., alpha=0.5 for classes with <100 instances) to compensate for the extreme positive-negative imbalance on rare objects? The per-class alpha mechanism exists in the loss code — it just needs to be populated from the training set class frequencies. Would calibrated per-class alpha improve detection recall on rare assembly parts (e.g., specific fastener types)?

### Q30: Epoch-Dependent Score Threshold

The appropriate DET_EVAL_SCORE_THRESH changes as training progresses. At epoch 3, score_max ≈ 0.076 and score_p50 ≈ 0.036. At epoch 10 if training succeeds, score_max might be 0.5+ and score_p50 0.1+. At epoch 30 (convergence), these would be higher still. The current fixed threshold of 0.001 is appropriate for reporting (matching YOLOv8 convention) but produces different mAP values at different epochs because the number of predictions above threshold changes. For the paper, should we report mAP at multiple thresholds (0.001, 0.01, 0.05, 0.1) to show the score distribution is separating? For the gate decision (RF4→RF5 transition), should we use a higher threshold (0.05) that reflects practical deployment conditions where 0.001-confidence detections are filtered out?

---

## Section 4: Activity Head Deep Analysis (Q31-Q40)

### Q31: Verb-Grouping Dataset Coverage

The verb-grouping (ACT_CLASS_GROUPING='hybrid', ACT_HYBRID_THRESHOLD=100) keeps classes with >=100 frames as standalone outputs and merges tail classes by verb prefix. The output is ~41 classes. What fraction of the training set frames have labels that fall into shared-verb groups? If a frame's raw action is "take_short_brace" (63 frames, <100 threshold, grouped as "take" verb) and the verb-group output is also "take" (shared with all other "take_*" actions), then the activity task is to predict "take" without distinguishing which specific "take" action. If 35% of frames map to ambiguous verb groups (where the verb group contains multiple distinct raw actions), then 35% of training data has label ambiguity. The 55-63% Top-1 target may be unreachable because the task definition has inherent ceiling — you can't achieve >65% Top-1 if 35% of labels are ambiguous.

### Q32: Re-Enabling the Temporal Path

`ACTIVITY_HEAD_SIMPLE = True` (config.py line 901) bypasses the TCN+2xViT temporal stack (8.2M params) for a 150K MLP classifier. The reason: with non-consecutive frames from the class-balanced sampler, the temporal stack learned noise, not signal. Once training has converged (epochs 10+), and the model is producing non-random predictions, should the full temporal path be re-enabled to capture temporal structure? The sequence-mode data path (PSR_SEQUENCE_MODE with true consecutive frames) could be shared with the activity head, providing temporally coherent batches for the TCN+ViT. This would require modifying the data loading to emit both PSR sequences and activity labels from the same consecutive frames. Is this architectural change worth implementing for the paper, or should the paper's activity results use the SIMPLE head with a caveat about temporal modeling being future work?

### Q33: Sampler Floor Memorization

`ACT_SAMPLER_MODE = 'balanced'` with `ACT_SAMPLER_COUNT_FLOOR = 15.0` (config.py lines 754-755). Classes with fewer than 15 frames are scaled by their frame count (not balanced to 15). How many classes fall at exactly the floor of 15? If there are classes with exactly 15 frames and the sampler oversamples them to reach the 15-frame target, these 15 frames are repeated multiple times per epoch. With 4387 batches per epoch and EFFECTIVE_BATCH=48, approximately 4387 * 48 / 15 ≈ 14,000 exposures per epoch per frame. These frames would be heavily memorized rather than generalized. Is this memorization causing the activity head to overfit to specific frames while underfitting on well-represented classes? Should the floor be raised to 50 or 100 to reduce the memorization pressure?

### Q34: Activity Head Gradient Clip Mismatch

`ACTIVITY_HEAD_GRAD_CLIP = 1.0` (config.py line 879) is 5x tighter than the global `GRAD_CLIP_NORM = 5.0` (config.py line 577). The activity head's gradient is clipped to 1.0 BEFORE the global clip is applied. This means even if the activity head produces a large gradient (e.g., after the gradient fix restoring it to ~0.48), it will be capped at 1.0. The other heads (detection, pose, PSR) can have gradients up to 5.0. Is this 5x tighter clip actively suppressing activity learning? The comment says "gradient norm at 0.012 is well below even 0.3" (justifying the raise from 0.3 to 1.0), but this was measured BEFORE the gradient fix. Now that the gradient path is restored, what is the actual activity head gradient norm? If it exceeds 1.0, the clip is active and limiting learning.

### Q35: Frozen Activity in Staged Training

During staged training (RF3), the activity head was frozen for 15 epochs while detection, pose, and PSR trained. When activity is activated at epoch 16+ (STAGE3_WARMUP_EPOCHS), the backbone features have been shaped by 15 epochs of non-activity training. The backbone may have specialized for detection (localization features) and pose (smooth features) at the expense of activity-discriminative features. The ACT_RAMP_EPOCHS=5 ramp-up is designed to gradually introduce activity gradient to prevent backbone shock. However, if the backbone features at epoch 16 are fundamentally not activity-discriminative, the ramp-up just gradually applies pressure in the wrong direction. Is there evidence that the backbone retains any activity-discriminative features after 15 epochs of non-activity training? A simple diagnostic: compute the t-SNE or PCA of C5 features for activity classes at epoch 15 vs epoch 0 to see if activity-relevant structure persists.

### Q36: Residual FPN Gradient Competition

`detach_reg_fpn=False` (Fix #13 in the fix table) means detection regression gradients now flow into the shared FPN features (p3-p7). The activity head uses `pyramid['p4'].detach()` (model.py line 2174) for its GAP input, meaning activity does NOT backprop into FPN p4. However, the activity projection (`proj_feat`) comes from C5_mod (not FPN), so activity gradient flows into C5_mod -> C5 -> backbone, bypassing FPN. The regression gradient flows into FPN -> backbone. Both meet at the backbone. Is there residual gradient competition between detection regression and activity at the backbone level? Specifically, regression wants spatially precise features (high-frequency detail) while activity wants semantically discriminative features (category-level abstraction). These are conflicting gradient directions on the same parameters. With 100 epochs of training, does this conflict cause the backbone to converge to a compromise that serves neither task well?

### Q37: Verb-Group Per-Class Frame Distribution

After verb-grouping (ACT_CLASS_GROUPING='hybrid', threshold=100), the output space is ~41 groups. The ACTIVITY_LOSS_WEIGHT is 0.8 but there is no per-group weight. What is the per-group frame count distribution? If the largest group has 10,000 frames and the smallest has 15 (the sampler floor), the imbalance is 667:1. Even with the balanced sampler (which equalizes sampling probability), the LDAM loss (which uses class margins) or the label smoothing (0.1 for all classes) may not compensate. Is there still a long tail in the grouped output space? If N groups have <100 frames, they may never receive enough gradient to learn discriminative features, regardless of the balanced sampler.

### Q38: ViT Attention Dropout Fix Impact

The ViT attention dropout was fixed from 0.3 to 0.1 (Fix #11). The original 0.3 value (3x the paper spec of 0.1) was identified as a cause of activity over-regularization. With ACTIVITY_HEAD_SIMPLE=True, the ViT is not used at all. So the fix is correct but currently irrelevant — the ViT dropout was causing problems when the ViT was active, but now the ViT is bypassed. If/when the temporal path is re-enabled (Q32), will the dropout=0.1 be sufficient? ViT with 0.1 dropout and only 8.2M params on a 3.7K-frame effective training set (after class balancing sampling) may still overfit. Should the re-enabled temporal path use dropout=0.2 to be safe, or was 0.3 the actual problem (not the sampling)?

### Q39: Single-Task Activity Baseline

There is no single-task activity baseline on this ConvNeXt backbone. Without it, we cannot distinguish between (a) multi-task interference causing poor activity performance, and (b) the activity head architecture or data difficulty being inherently insufficient for 41-class classification. This is Ablation A from the required ablations. A single-task activity run (one head, no detection/pose/PSR, same backbone) would immediately answer: at the same epoch count with the same learning rate, does the activity head achieve 40-55% top-1? If yes, multi-task interference is the problem. If no (stays at 10-20%), the activity head or data is the problem. This is a 2-day experiment. Is there any reason NOT to run it?

### Q40: Activity Entropy Trajectory

Activity entropy went from 0 nats (epoch 0, all predictions in 1 class) to 1.036 nats (epoch 2, predictions in 3 classes). With 41 classes, maximum entropy is log2(41) = 5.36 nats (binary) or ln(41) = 3.71 nats (natural). The trajectory shows entropy increasing at ~0.5 nats per epoch. At this rate, healthy entropy (predictions spread across ~10 classes) would be ln(10) = 2.3 nats, reached by epoch ~4-5. Full diversity (all 41 classes used) would be reached by epoch ~7-8. However, the target is NOT maximum entropy — we want the entropy to decrease AFTER convergence, concentrating probability on the correct class. What is the healthy trajectory? A plausible schedule: epoch 3: entropy = 1.5-2.0 (increasing diversity), epoch 6: entropy = 2.5-3.0 (peak diversity), epoch 10: entropy = 1.5-2.0 (concentrating on correct classes), epoch 20: entropy = 0.5-1.0 (sharp predictions). Is this the expected pattern? What would entropy > 2.5 at epoch 10 indicate (overly diffuse predictions, nearing random)?

---

## Section 5: Training Infrastructure & Validation (Q41-Q50)

### Q41: CUDA Kernel Zombie After ThreadPool Timeout

The ThreadPoolExecutor evaluation mechanism (train.py line 4712-4717) uses a timeout to handle CUDA kernel hangs. However, the code comment explicitly states: "ThreadPoolExecutor with a timeout raises TimeoutError without actually stopping the running thread (CUDA kernel stays alive)." After timeout, the code creates a fresh executor for retry. The zombie thread retains the CUDA kernel in the GPU context. On a single-GPU system (RTX 5060 Ti), this means the zombie kernel occupies GPU resources (CUDA context, potentially VRAM) while the new executor tries to launch new kernels. Can concurrent CUDA kernels from the zombie thread corrupt the new evaluation? Does CUDA serialize kernels from the same process, or can the zombie kernel's memory writes (from a partially-executed forward pass) interfere with the new executor's computation? What is the observed behavior: does the retry succeed, or does it also hang because the GPU is still processing the zombie kernel?

### Q42: Subprocess Eval on Idle GPU

The system has two GPUs: RTX 3060 12GB (GPU 0, used for display + VSCode, ~400MB used) and RTX 5060 Ti 16GB (GPU 1, used for training). Subprocess eval was disabled (Fix #19) due to single-GPU contention. However, GPU 0 (RTX 3060) is idle during training (CUDA_VISIBLE_DEVICES presumably selects GPU 1). If subprocess eval were re-enabled on GPU 0, could evaluation run without interfering with training on GPU 1? The subprocess would use CUDA_VISIBLE_DEVICES=0 for eval only. Since the RTX 3060 has 12GB and only ~400MB is used by the display, there's ~11GB free — more than enough for VAL_BATCH_SIZE=8 inference. This would make eval SIGKILL-safe (subprocess can be killed without affecting training) and eliminate the ThreadPoolExecutor zombie kernel problem (Q41). What are the implementation barriers? The eval code would need to load the model on GPU 0, which requires either (a) CPU transfer from GPU 1 to GPU 0, or (b) a separate checkpoint file read directly to GPU 0.

### Q43: Learning Rate Scaling for EFFECTIVE_BATCH=48

The paper specifies EFFECTIVE_BATCH=32 (BATCH_SIZE=2 * GRAD_ACCUM_STEPS=16). The current config has EFFECTIVE_BATCH=48 (BATCH_SIZE=6 * GRAD_ACCUM_STEPS=8). The linear scaling rule (from Goyal et al., "Accurate, Large Minibatch SGD," 2017) states that when batch size increases by k, the learning rate should increase by k (for the same number of epochs). Our batch is 48/32 = 1.5x larger than the paper's. Should the learning rate be 5e-4 * 1.5 = 7.5e-4? Alternatively, should GRAD_ACCUM_STEPS be reduced to 5 (BATCH_SIZE=6 * 5 = 30, close to paper's 32) rather than adjusting the LR? The concern is that the current effective batch of 48 with LR 5e-4 may be undertrained — the model sees fewer parameter updates per epoch (4387 batches vs 6580 at batch=32) while the update magnitude is the same. What is the principled correction?

### Q44: OneCycleLR pct_start Discrepancy

The optimizer code (src/training/optimizer.py line 58) has `pct_start=0.3`, which means the LR peaks at epoch = WARMUP_EPOCHS + 0.3 * (total_epochs - WARMUP_EPOCHS) = 2 + 0.3 * 98 = epoch 31.4. However, the documentation (analyses/91-architecture-deep-dive.md line 118) says `pct_start=0.1` (peak at epoch 2 + 0.1 * 98 = epoch 11.8). The v1/v2/train.py files (lines 2092-2097) also specify `pct_start=0.1` with the log message "OneCycleLR (pct_start=0.1)." Which value is currently active? If 0.3 is active, the LR does not reach its peak until epoch 31 — meaning the first 30 epochs are in the LR-increasing phase. For a 100-epoch schedule, the model spends 30% of training at sub-peak LR and 70% at decaying LR. With pct_start=0.1, the model would spend 10% ramping up and 90% decaying. Given that pose converges in 0.5 epochs and detection is stable by epoch 2, is pct_start=0.3 appropriate? The long ramp-up means heads that converge early (pose at epoch 1) spend 29 more epochs at increasing LR, which may cause overfitting or parameter oscillation.

### Q45: BF16 Mixed Precision Feasibility

`MIXED_PRECISION = False` (config.py line 597) because FP16's limited dynamic range causes PSR sequence loss spikes to corrupt the GradScaler. However, the RTX 5060 Ti (Ada Lovelace architecture) supports BF16, which has the same 8-bit exponent range as FP32 (vs. FP16's 5-bit exponent). BF16 can represent values up to 3.4e38 (same as FP32) vs FP16's 65504 max. PSR loss spikes reaching 45-60 (config.py line 1018 comment mentions "spikes reached 45-60") would overflow FP16 (max 65504 in the accumulation, but the loss magnitude itself is fine). The real issue is the GradScaler — PSR loss spikes inflate the gradient, causing the scaler to downscale, starving other heads. With BF16, no GradScaler is needed because the dynamic range matches FP32. Can MIXED_PRECISION be re-enabled with `torch.amp.autocast(dtype=torch.bfloat16)`? The 2x speedup (from FP32 to BF16) would halve the 6-day training time to 3 days. Is there any code change required beyond setting `MIXED_PRECISION = True` and ensuring the autocast context manager uses `bfloat16`?

### Q46: PSR Warmup Duration

`PSR_WARMUP_STEPS = 500` (config.py line 936) at 2x multiplier ramps the PSR precision multiplier from 2.0 to 1.0 over 500 steps. At 4387 batches per epoch, 500 steps = 0.11 epochs = ~20 minutes. Is 20 minutes sufficient for the PSR head to establish meaningful predictions before Kendall full weighting kicks in? The comment says "500 steps" was set in the agent audit, reduced from a previous state. The PSR head has d_model=256 Transformer with 8-frame sequences — it may need more time to learn temporal structure. If PSR_WARMUP_STEPS were increased to 2000-5000 (0.5-1.1 epochs), the PSR head would have more time to escape the random init before competing with detection/activity for backbone resources. What is the risk of longer warmup? The PSR head trained alone without competition may converge to a local optimum that collapses when Kendall weighting is applied.

### Q47: Minimum Viable Eval Batches

At VAL_BATCH_SIZE=8 and GATE_EVAL_MAX_BATCHES=200, the gated eval processes 1600 frames. The validation set has 1933 segments (per segment/metrics documentation), each segment being a continuous action. These 200 batches sample approximately 1600/1933 = 83% of segments (not exact because segments vary in length). For activity macro-F1, which averages per-class F1, the reliability depends on whether tail classes appear in the 1600-frame sample. With ~41 verb groups and some groups having <50 frames in the full validation set, the probability that a tail group appears in 1600 randomly-sampled frames is approximately 1 - (1 - p)^1600 where p is the group's frequency. For a group with p=0.005 (8 frames in 1933), the probability is 1 - (0.995)^1600 ≈ 0.9997 — essentially certain. But for p=0.001 (2 frames), the probability is 1 - (0.999)^1600 ≈ 0.798 — a 20% chance of missing the group entirely. Should GATE_EVAL_MAX_BATCHES be increased to 300 or 400 for the activity macro-F1 to be reliable, especially during RF4 gate decisions where a 20% variance could trigger a false fail?

### Q48: Tail Class Coverage in Gated Eval

Extending Q47: at VAL_BATCH_SIZE=8 and GATE_EVAL_MAX_BATCHES=200, the tail class coverage is uncertain. The evaluation uses a WeightedRandomSampler (or sequential sampler depending on config), not a class-balanced sampler. This means rare classes are proportionally represented — if a class has 0.1% frequency, it appears in ~2 frames per 1600. For macro-F1, if a class has 0 correct predictions (because no test sample appeared), its per-class F1 is 0.0, dragging down the macro average. For 41 classes, 2 missing classes out of 41 would reduce macro-F1 by 5% even if the other 39 classes have perfect F1=1.0 (actual macro-F1 = 39/41 = 0.95). Is the gated eval's macro-F1 being systematically underestimated due to missing tail classes? Should the evaluation use a class-balanced sampler for activity to ensure all output groups are represented?

### Q49: Gradient Cosine Similarity Between Heads

The backbone update direction is the sum of gradients from all 5 heads (det class, det reg, pose, activity, PSR). If these gradients point in similar directions (cosine similarity > 0.7), the multi-task framework is providing complementary regularization. If they point in conflicting directions (cosine similarity < 0.1 or negative), the heads are competing for different feature representations. We have NOT computed per-head gradient cosine similarity on the shared backbone parameters. This is a critical gap: without knowing whether detection wants to move the backbone in the same direction as pose, we can't determine if multi-task learning is beneficial or harmful. What is the minimum viable implementation? Register a hook on the backbone's final layer (C5) that captures gradients from each head's backward pass, then compute pairwise cosine similarity every LIVENESS_EVERY steps. This adds ~5 lines of code and negligible compute.

### Q50: The Single Most Diagnostic Experiment

Given all the open questions above, what is the SINGLE most diagnostic experiment to run in the next 48 hours? Candidates:
1. **Gradient cosine similarity** (Q49) — tells us if multi-task learning is working
2. **Single-task activity baseline** (Q39) — tells us if activity difficulty is architectural or data-limited
3. **Log Kendall log_vars** (Q11) — tells us the actual loss balancing dynamics
4. **Full eval at epoch 3** (currently scheduled) — gives the first real metrics
5. **Alpha sweep** (Q20) — tests if alpha=0.5 is better for asymmetric gamma

My recommendation: **(1) Gradient cosine similarity** because it feeds into every other decision. If all heads agree on backbone direction, multi-task is working and we should invest in architecture improvements. If they disagree, we need to rebalance loss weights (alpha, gamma, Kendall bounds) before making architectural changes. Additionally, logging the 4 Kendall log_vars is a trivial code change (<10 lines in train.py) and should be done as a prerequisite for the epoch 3 eval.

The epoch 3 eval is already scheduled (VAL_EVERY=3, epoch 3 is next). Combined with log_var logging and gradient cosine similarity, the epoch 3 results will provide a comprehensive diagnostic picture.
