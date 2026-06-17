# 19 — Pre-Training Readiness Audit: 100-Item Deep Checklist

**Purpose.** A go/no-go audit to run *before* you press "train," so that the run
produces **benchmarkable, non-zero, non-NaN, non-NA, protocol-correct metrics for
every task** and you get the unified multi-task model you're after. Every item is
a *verifiable gate*: it states **what** to check, **why** it matters (the failure
mode if it's wrong), **how** to verify, and the **pass** criterion.

**Honesty, once, up front (then we focus on readiness):** passing all 100 makes
your numbers *real, defensible, and as competitive as the architecture allows* —
and lets you **win the winnable cells** (PSR vs STORM-PSR 0.506 / B2 0.731; head
pose, uncontested) and report detection/activity honestly as "competitive at a
fraction of deployed compute." It does **not** make a shared-backbone multi-task
model beat YOLOv8m (83.80) or MViTv2 (65.25) on their headline metrics — that is
a design ceiling no checklist removes. The defensible goal is *one pipeline, all
tasks, no zeros, every number protocol-correct, efficiency as the headline.*

Grounding: file:line refs are to `code/` in this folder. Run the **6 master
gates** (below) last; if any fails, **do not start a multi-hour run.**

---

## THE 6 MASTER GO/NO-GO GATES (all must pass)

- ⛔ **G1 — No silent failure.** `ASSERT_AND_CRASH=True` during bring-up; a 200-step
  smoke runs with **zero** `1e-4` sentinels, zero `[PSR_NAN]`, zero `[COMBINED_NAN]`.
- ⛔ **G2 — Every head ALIVE.** Liveness probe: per head loss > 10× floor,
  grad-norm(first & last layer) > 1e-6, output std > 1e-3.
- ⛔ **G3 — Optimizer commits.** `[RC-29] committed>0, skipped=0` in FP32 over the smoke.
- ⛔ **G4 — Labels mean what the metric claims.** Activity clip-level + NA-excluded;
  detection b-boxed on annotated frames; PSR transitions; subset class-stratified.
- ⛔ **G5 — Eval emits finite numbers for evaluated heads only.** No cosmetic NaN/NA;
  no head reports exactly 0.0000 from a stub.
- ⛔ **G6 — Reproducible.** Seeds fixed, deterministic cuDNN, config snapshotted with the run.

---

## SECTION A — Architecture integrity & forward-pass sanity (1–9)

**1. All 5 head outputs present with correct shapes.** *Why:* a missing/mis-shaped
key crashes eval or silently broadcasts wrong (the activity broadcast crashes you
already hit). *Verify:* `out=model(x[B=2])`; assert keys & shapes — det `[B,N,24]`+
`[B,N,4]`, act `[B,75]`, psr `[B,11]` or `[B,T,11]`, head_pose `[B,9]`. *Pass:* all present.

**2. Forward is deterministic at eval.** *Why:* dropout/nondeterminism → metrics
not reproducible across eval calls. *Verify:* `model.eval()`; two forwards on same
input. *Pass:* max|Δ| < 1e-5.

**3. No NaN/Inf in any head output at init.** *Why:* a single NaN at init poisons
every loss through Kendall. *Verify:* liveness probe at step 0. *Pass:* all finite.

**4. Output logit std > 1e-3 per head at init.** *Why:* constant output = degenerate
start; focal/CE can't break symmetry. *Verify:* probe per-head std. *Pass:* >1e-3.

**5. Param count matches paper.** *Why:* silent architecture drift (a head wired off,
VideoMAE on/off). *Verify:* `count_parameters` (model.py:1996). *Pass:* 53.42M
trainable / 76.16M total (53.3M w/o VideoMAE) within 1%.

**6. `requires_grad` exactly as intended.** *Why:* a frozen head can't learn; an
accidentally-frozen backbone caps everything. *Verify:* print the 3 optimizer param
groups (optimizer.py:19-28) sizes. *Pass:* backbone/heads trainable, VideoMAE frozen.

**7. BN behavior safe at batch=1.** *Why:* BatchNorm on physical batch 1 produces
garbage running stats. *Verify:* confirm backbone uses LayerNorm (ConvNeXt) and head
subnets use **GroupNorm** (model.py:503); `_freeze_bn` honored (model.py:1728).
*Pass:* no BN updating at bs=1.

**8. Input normalization = ImageNet mean/std.** *Why:* wrong normalization makes
pretrained features meaningless → every head caps low for no obvious reason.
*Verify:* dataset transform mean `[0.485,0.456,0.406]`, std `[0.229,0.224,0.225]`.
*Pass:* exact.

**9. Fits in 12 GB FP32 at chosen batch/accum.** *Why:* OOM mid-epoch wastes the run.
*Verify:* 200-step smoke peak mem. *Pass:* < 11.5 GB, no OOM.

## SECTION B — Backbone / FPN / anchors (10–16)

**10. ConvNeXt-Tiny ImageNet weights actually loaded.** *Why:* a random backbone
silently destroys transfer → ~all tasks lose double digits. *Verify:* first-conv
weight stats vs a known-good load; log "loaded pretrained". *Pass:* not random-init.

**11. FPN P3–P7 spatial dims correct at 1280×720.** *Why:* wrong strides misalign
anchors and features. *Verify:* assert P3 `90×160`, P4 `45×80`, P5 `23×40`, P6 `12×20`,
P7 `6×10`. *Pass:* exact.

**12. Anchor count == detection head output N.** *Why:* a mismatch makes anchor↔pred
matching silently wrong. *Verify:* `assert len(anchors)==cls_preds.shape[1]`. *Pass:* equal.

**13. ⛔ Anchors calibrated to GT (k-means).** *Why:* original `(24…384)` vs GT
146–594px → only ~1.6% of anchors can reach IoU≥0.5 → hard recall ceiling. *Verify:*
`calibrate_anchors.py`; set `ANCHOR_SIZES` to clusters (config.py:247). *Pass:* anchor
scales span 146–594.

**14. Anchor↔GT coverage ≥ 90%.** *Why:* unmatchable GT = recall you can never get.
*Verify:* offline: fraction of GT with some anchor IoU≥0.5. *Pass:* ≥90%.

**15. Feature magnitudes not exploding (RC-25).** *Why:* huge FPN features overwhelm
fresh heads → flat scores. *Verify:* `diag_feature_magnitude.py`; median |z|. *Pass:* ≈3, not ≫8.

**16. Small-object path decision documented.** *Why:* C2 (stride-4) is skipped; tiny
GT below P3 RF are unrecoverable. *Verify:* GT size histogram vs P3 RF. *Pass:* smallest
GT ≥ P3 RF, or P2 added.

## SECTION C — Cross-task conditioning (17–22)

**17. det_conf = `sigmoid(cls.max).detach()`.** *Why:* raw logits dominate activity
input (RC-19, L2=243). *Verify:* model.py:1865; activity-input L2. *Pass:* bounded[0,1]+detached, L2 O(1).

**18. PoseFiLM γ = `1+tanh(·)` ∈ (0,2).** *Why:* γ<0 inverts features. *Verify:* assert
γ range over a batch. *Pass:* in (0,2).

**19. HeadPoseFiLM input stop-grad.** *Why:* feedback loop couples head-pose error into
backbone. *Verify:* `.detach()` before FiLM. *Pass:* detached.

**20. FiLM gated by real confidence (or documented).** *Why:* IndustReal pseudo-keypoints
have conf=1 → FiLM injects noise as signal. *Verify:* conf source. *Pass:* gated or noted.

**21. ⛔ `ZERO_DET_CONF_FOR_RECOVERY=False`.** *Why:* zeroing starves activity (the RC-28
deadlock leg). *Verify:* config. *Pass:* False.

**22. Stop-grads honored (no unintended coupling).** *Why:* leaked gradients collapse
heads together. *Verify:* grad-flow test — perturb head_pose output, assert backbone grad
unchanged. *Pass:* isolated.

## SECTION D — Data pipeline & label correctness (23–35)

**23. ⛔ Det GT rescaled with image resize + IMG_SIZE assert.** *Why:* boxes stay in
1280-space if IMG_SIZE shrinks → IoU=0 → det silently zeroes (dataset:891). *Verify:*
decode(encode(gt))==gt at current IMG_SIZE. *Pass:* IoU>0.99.

**24. Boxes pixel xyxy, same W/H normalization as anchors.** *Why:* coordinate-space
mismatch → all IoU=0. *Verify:* assert box max ≤ IMG_W/H. *Pass:* consistent.

**25. category remap 1–24→0–23 with range guard.** *Why:* off-by-one mislabels every box
(dataset:1012). *Verify:* assert labels∈[0,23]. *Pass:* ok.

**26. ⛔ Activity segment index built (1 per AR span).** *Why:* clip-level protocol needs
segments, not frames. *Verify:* segment count vs AR rows. *Pass:* matches.

**27. ⛔ Activity clip sampler = 16 uniform frames/segment.** *Why:* MViTv2 protocol; per-
frame is a different (worse, incomparable) metric. *Verify:* clip shape `[16,3,H,W]`. *Pass:* ok.

**28. ⛔ `activity_mask` in both collates; NA never a metric label.** *Why:* NA-collapse —
"predict NA" wins, Top-1 measures the label imbalance not the model. *Verify:* assert mask
present; metric labels exclude NA. *Pass:* ok.

**29. NA fraction measured & logged.** *Why:* silent dominant class. *Verify:* histogram.
*Pass:* known number.

**30. ⛔ Subset selection class-stratified.** *Why:* first-N-alphabetical (dataset:1104)
can exclude whole classes → Top-1 capped, train/val class sets differ. *Verify:* class
histogram of subset. *Pass:* ≥K classes in BOTH train and val.

**31. ⛔ PSR transition targets built (σ=3).** *Why:* per-frame BCE on 95%-static labels
teaches constant output. *Verify:* `build_transition_targets` peaks at flips
(psr_transition.py:34). *Pass:* peaks present.

**32. PSR `-1` fraction measured; transient vs persistent decided.** *Why:* persistent
`-1` over-counts ignores (dataset:488). *Verify:* live diagnostic (losses.py:763). *Pass:* documented.

**33. PSR %static per component measured.** *Why:* choose objective from data, not assumption.
*Verify:* printed per component. *Pass:* known.

**34. Collate clip_rgb consistent (both paths) or VideoMAE-off documented.** *Why:* eval
feeds zeros where train feeds clips (RC-17), re-arms on VideoMAE enable. *Verify:* train vs
val batch key-sets equal. *Pass:* consistent.

**35. Sampler balances the right axis; det not starved.** *Why:* `WeightedRandomSampler`
balances activity classes, but detection needs GT-bearing frames. *Verify:* GT-frame count per
optimizer window. *Pass:* ≥1 GT frame per update window.

## SECTION E — Loss functions & numerical hygiene (36–46)

**36. ⛔ No `1e-4` sentinel fires.** *Why:* it masks dead/NaN losses (losses.py:1041/1230/1258);
"non-zero" is a lie if a sentinel set it. *Verify:* `ASSERT_AND_CRASH=True`, smoke. *Pass:* no sentinel.

**37. Smooth-caps disabled during bring-up.** *Why:* caps hide exploding losses and gradient
death above cap. *Verify:* set all `*_LOSS_CAP=1e9`. *Pass:* raw losses observed.

**38. Detection focal: empty frames skipped, normalize by `n_img_with_gt`.** *Why:* RC-28 —
empty-frame negative mass dominated the gradient. *Verify:* losses.py:228,295. *Pass:* present.

**39. Focal numerics safe.** *Why:* `p_t→0` → `log(0)` NaN. *Verify:* sigmoid clamp 1e-7
(losses.py:249), logit clamp ±8 (losses.py:719). *Pass:* finite on extreme logits.

**40. GIoU guarded (no negative→Kendall divergence).** *Why:* GIoU∈[-1,1]; negative × big
precision diverges. *Verify:* zero-floor (losses.py:1009). *Pass:* loss_det ≥ 0.

**41. ⛔ PSR loss is the transition objective, O(0.1–0.3), finite.** *Why:* per-frame focal +
sensitivity penalty self-nullifies. *Verify:* `[PSR_DIAG]` raw value. *Pass:* 0.1–0.3, no NaN.

**42. PSR sensitivity penalty clamped or removed.** *Why:* `-log(std)` → ±inf at constant/extreme
logits (losses.py:1188). *Verify:* clamp `[0,5]` or `PSR_SENSITIVITY_WEIGHT=0`. *Pass:* finite.

**43. PSR `-1` masking correct in loss.** *Why:* `-1` makes `p_t` negative → unstable. *Verify:*
ignore-mask path (losses.py:729-748). *Pass:* masked entries contribute 0.

**44. Activity loss = CE + label-smoothing (LDAM off).** *Why:* LDAM s=30 + CB-sampling + LS
→ 1-class collapse. *Verify:* `USE_LDAM_DRW=False` (config.py:424). *Pass:* CE path active.

**45. Head-pose loss geometry-aware (6D + geodesic).** *Why:* raw-9 MSE → 60–70° MAE; acos>1
→ NaN. *Verify:* `USE_GEO_HEAD_POSE=True`; acos clamped. *Pass:* finite, sane scale.

**46. Per-task loss scales balanced before Kendall.** *Why:* a 1000× scale gap makes one head
own the backbone. *Verify:* log raw per-head losses; ratios within ~10×. *Pass:* comparable.

## SECTION F — Multi-task balancing (Kendall & gradients) (47–52)

**47. Kendall init neutral (`s=0`, precision=1).** *Why:* skewed init silently down-weights a
head. *Verify:* log_var init (losses.py:879-886). *Pass:* s_det/act/psr/pose ≈ 0.

**48. Kendall clamp `[-4,2]` applied BEFORE use, not after backward.** *Why:* the original bug —
clamp after backward sees unclamped grads. *Verify:* losses.py:1273-1276 ordering. *Pass:* clamp before precision.

**49. Stage-aware Kendall doesn't permanently zero a head's log_var.** *Why:* zeroing destroys
learned uncertainty (the Stage-3 reset bug). *Verify:* with `STAGED_TRAINING=False`, no zeroing.
*Pass:* all log_vars learn.

**50. Each enabled head contributes `prec·loss + log_var` to total.** *Why:* a missing term =
that head trains unweighted or not at all. *Verify:* losses.py:1331-1358. *Pass:* all enabled heads in total.

**51. Per-head gradient-norm balance logged.** *Why:* one head's gradient drowning others is the
collapse mechanism. *Verify:* log grad-norm per head per N steps. *Pass:* no head >100× another sustained.

**52. No ramps during bring-up.** *Why:* activity/PSR ramps multiply gradient suppression and
confound attribution. *Verify:* `STAGED_TRAINING=False`, ramps=1. *Pass:* full gradient from step 0.

## SECTION G — Optimization & training loop (53–61)

**53. Differential LR groups correct.** *Why:* backbone overfits/forgets if LR too high; heads
starve if too low. *Verify:* optimizer.py:30-37 — backbone 0.1×, head 1×, bias 0.3×. *Pass:* as specified.

**54. Grad-accum loss scaled by `accum_steps`.** *Why:* unscaled accumulation = effective LR ×
accum → divergence. *Verify:* `loss/float(accum_steps)` (train.py:1196). *Pass:* present.

**55. Optimizer steps every `accum_steps` (and at loader end).** *Why:* missed final partial
window wastes data; double-step diverges. *Verify:* train.py:1060. *Pass:* correct cadence.

**56. Grad clipping active.** *Why:* a spike explodes weights → NaN cascade. *Verify:*
`clip_grad_norm_` (train.py:1080). *Pass:* present, sane max-norm.

**57. `zero_grad` after every step, not mid-accum.** *Why:* premature zero loses accumulated grad.
*Verify:* zero only at step sites (train.py:1078/1085/...). *Pass:* correct.

**58. ⛔ FP32 enforced; scaler inert; committed>0.** *Why:* RC-29 — fp16 scaler silently skips
steps → frozen model. *Verify:* `MIXED_PRECISION=False`; `[RC-29] committed/skipped`. *Pass:* skipped=0.

**59. Scheduler warmup→cosine wired to epochs.** *Why:* no warmup → early divergence; wrong T →
LR never decays. *Verify:* optimizer.py:42-53; log LR curve. *Pass:* warms then anneals.

**60. PSR sequence-batch interleave correct.** *Why:* the Causal Transformer needs temporal
batches; wrong cadence → PSR never sees sequences. *Verify:* `is_seq_batch` every `PSR_SEQ_EVERY_N_BATCHES`
(train.py:967), seq path computes PSR-only. *Pass:* fires every N, no shape crash.

**61. EMA off during recovery; on only at R3 after monotonic improvement.** *Why:* EMA blends
collapse into best.pth. *Verify:* `USE_EMA=False` now (config.py:586). *Pass:* off until R3.

## SECTION H — Per-head liveness & gradient flow (62–67)

**62. ⛔ Detection ALIVE.** *Why:* spine of the model; feeds activity/PSR. *Verify:* probe — cls
loss 3–30 on GT frames, grad-norm>1e-6, score std>0.05 after a few hundred steps. *Pass:* ALIVE.

**63. ⛔ Head-pose ALIVE.** *Why:* the free win; also a stable 2nd backbone signal. *Verify:*
loss 0.2–0.5, finite MAE. *Pass:* ALIVE.

**64. ⛔ PSR ALIVE (gradient reaches PSR head).** *Why:* the historically-dead head. *Verify:*
loss 0.1–0.3, PSR-head grad-norm>1e-6, ≥3 unique patterns. *Pass:* ALIVE.

**65. ⛔ Activity ALIVE (predicts ≥4 classes).** *Why:* 1-class collapse is the classic failure.
*Verify:* `pred_seen≥4`, loss 3–4.3, grad-norm>1e-6. *Pass:* ALIVE.

**66. FeatureBank / temporal path carries gradient (if used).** *Why:* detached bank (model.py:1188)
+ slot-−1 overwrite (model.py:1340) = cosmetic temporal layers. *Verify:* bank grad-norm>1e-6 OR
clips routed through ViT directly. *Pass:* temporal positions learnable.

**67. Bring-up adds heads one at a time.** *Why:* enabling all at once hides which head breaks.
*Verify:* det → +hpose → +psr → +act, liveness after each. *Pass:* each ALIVE before next.

## SECTION I — Detection readiness (68–75)

**68. b-boxed eval on annotated frames only.** *Why:* the 83.80 comparison is annotated-frames;
mixing empty frames changes the metric. *Verify:* eval frame count == annotated count. *Pass:* matches.

**69. all-frames eval separate.** *Why:* the 64.10 comparison dilutes with empty frames. *Verify:*
`compute_ap_per_class_all_frames` (evaluate.py:1167). *Pass:* second number reported.

**70. ⛔ Eval conf threshold = 0.001 (not 0.5/0.02).** *Why:* truncating PR at high conf crushes
recall and breaks YOLOv8 comparability. *Verify:* `DET_EVAL_SCORE_THRESH=0.001` (config.py:344). *Pass:* 0.001.

**71. NMS active and per-class.** *Why:* duplicate boxes inflate FP, deflate precision. *Verify:*
evaluate.py:3020. *Pass:* dedup.

**72. mAP@0.5 and mAP@[0.5:0.95] both computed.** *Why:* paper reports both; [0.5:0.95] is COCO
standard. *Verify:* `compute_ap_multi_thresh` 10 thresholds. *Pass:* both present.

**73. Synthetic pretrain pipeline runnable.** *Why:* the gap to 0.7+ comes from synthetic
(`PRETRAIN_DET_ON_SYNTH=True`, config.py:386, unused). *Verify:* synth loader + pretrain script run.
*Pass:* produces a det-pretrained checkpoint.

**74. DET_PROBE shows localization on GT frames.** *Why:* confirms decode/match before trusting mAP.
*Verify:* bestIoU>0.5 on GT batches. *Pass:* yes (you already see 0.94).

**75. Score distribution non-degenerate.** *Why:* std<0.01 = constant output (flat-score collapse).
*Verify:* eval score std (evaluate.py:3361 warns). *Pass:* std>0.05.

## SECTION J — Activity readiness (76–81)

**76. ⛔ Clip-level eval per SEGMENT, not per recording.** *Why:* recording-level aggregation
(evaluate.py:628) degenerates to NA on NA-dominated recordings. *Verify:* 1 prediction per segment.
*Pass:* segment-level.

**77. ≥15 distinct classes predicted at subset 1.0.** *Why:* low diversity = partial collapse.
*Verify:* `pred_seen`. *Pass:* ≥15 (toward 74).

**78. VideoMAE/K400 stream enabled as primary (or documented w/o).** *Why:* per-frame CNN can't
model motion; pick/place/insert look identical in 1 frame. *Verify:* `USE_VIDEOMAE/K400=True`, real
checkpoint loaded (not 3D-conv fallback). *Pass:* real encoder.

**79. Clip tubelets reach the temporal model (not GAP'd per-frame).** *Why:* GAP destroys spatial
+ temporal structure. *Verify:* token shapes into ViT. *Pass:* temporal tokens, not 1 replicated frame.

**80. Class imbalance handled by sampling/CE-LS, not LDAM-s30.** *Why:* over-correction → collapse.
*Verify:* sampler + CE+LS. *Pass:* balanced, no LDAM.

**81. Frame-level Top-1 reported as secondary only.** *Why:* it's not the MViTv2 protocol; lead
with clip-level. *Verify:* table footnote. *Pass:* labeled secondary.

## SECTION K — PSR readiness (82–87)

**82. ⛔ Monotonic decoder enforces fill-forward.** *Why:* without it, predicted state oscillates,
killing edit/POS. *Verify:* `MonotonicDecoder` (psr_transition.py:79). *Pass:* monotone states.

**83. Procedure-order prior applied.** *Why:* B2 (0.731) beats fancy nets *because* of order
constraints; you need the same. *Verify:* order graph in decode. *Pass:* invalid orders penalized.

**84. F1 uses ±3/±5-frame bi-directional greedy matching of transition EVENTS.** *Why:* per-frame
F1 isn't the PSRT metric. *Verify:* matching on events. *Pass:* PSRT-protocol F1.

**85. POS = correctly-ordered adjacent pairs (PSRT def).** *Why:* STORM's code uses a different
(DL-distance) POS; mixing definitions is a review flag. *Verify:* runs-based pair ordering. *Pass:* PSRT def.

**86. Evaluated on full test set, not a skewed slice.** *Why:* a constant pattern scores fake-high
on a comp0=1 slice (the earlier "0.73 artifact"). *Verify:* full split. *Pass:* full test set.

**87. ≥3 unique predicted patterns; transitions track GT.** *Why:* constant output is the failure.
*Verify:* unique-pattern count + transition F1>0. *Pass:* non-constant.

## SECTION L — Head-pose / assembly state / error verification (88–91)

**88. Head-pose MAE finite & non-degenerate under FP32.** *Why:* old NaN was AMP/RC-29; degenerate
constant output gives a fake-good MAE. *Verify:* MAE finite, output std>1e-3. *Pass:* <35°, varied.

**89. Assembly F1@1 derived from det confidences, single-state frames.** *Why:* wrong frame subset
= incomparable to SupCon. *Verify:* protocol matches Schoonbeek '24. *Pass:* single-state frames only.

**90. Error-Verif AP = 1−conf(expected), non-empty error set.** *Why:* AP over empty set → NaN.
*Verify:* Lehman '24 formula; `max(n_err,1)` guard. *Pass:* finite AP.

**91. Vectors L2-normalized before angular MAE; acos clamped.** *Why:* `acos(1.0000001)`→NaN.
*Verify:* clamp `[-1+1e-6,1-1e-6]`. *Pass:* finite.

## SECTION M — Evaluation & metric correctness (92–97)

**92. ⛔ Val-line prints metrics only for evaluated heads; uses `.get(k,nan)`.** *Why:* stub-key
mismatch prints cosmetic NaN (your current psr/act NaN). *Verify:* formatter guards. *Pass:* no false NaN.

**93. No metric is exactly 0.0000 from a stub.** *Why:* 0 from "head skipped" vs "no preds above
thresh" are different bugs. *Verify:* distinguish in logs. *Pass:* every reported 0 explained.

**94. mAP uses COCO all-point interpolation.** *Why:* VOC-11 vs COCO-101 differ by points.
*Verify:* `_coco_ap` (evaluate.py:1153). *Pass:* COCO.

**95. Combined metric components all finite before weighting.** *Why:* one NaN → combined NaN →
no checkpoint saved. *Verify:* clamp (train.py:1700), weights 0.30/0.35/0.15/0.20. *Pass:* finite.

**96. EVAL cadence set (gate metric cheap, full eval periodic).** *Why:* 87-min det eval every epoch
burns the budget. *Verify:* capped `EVAL_MAX_BATCHES` for gate + full every N. *Pass:* < ~15 min/epoch gate.

**97. Each metric's protocol matches its baseline's protocol.** *Why:* apples-to-apples or the gap
must be named (RGB-only vs RGB+VL+stereo for MViTv2). *Verify:* protocol table (paper §eval). *Pass:* documented per row.

## SECTION N — Reproducibility, logging, checkpointing, paper (98–100)

**98. ⛔ Seeds fixed + deterministic cuDNN + config snapshot saved with run.** *Why:* irreproducible
numbers are unpublishable; you can't explain a regression. *Verify:* seed 42 everywhere, cuDNN
deterministic, config copied to run dir. *Pass:* reproducible.

**99. Checkpoint selection saves RAW best (not EMA), and resumes cleanly.** *Why:* best.pth=EMA blend
was the silent metric-killer. *Verify:* best from raw model (config.py:586); resume restores
optimizer+Kendall+scheduler. *Pass:* clean resume.

**100. Efficiency numbers measured; paper framing honest.** *Why:* the efficiency story is the
headline; w/ VideoMAE (75.3M) ≈ sum of baselines (75.4M), so lead with "one forward pass / one
pipeline," not param count. *Verify:* `efficiency_report.py` (params/GFLOPs/FPS, both modes);
report 53.3M w/o-VideoMAE as the smaller variant. *Pass:* table filled, framing matches numbers.

---

## How to use this before the real run

1. Set the **bring-up profile** (G1: ASSERT_AND_CRASH, caps off, sensitivity 0, ramps off).
2. Run a **200-step smoke** with the **liveness probe** → clear Sections A–H + G1/G2/G3.
3. Run a **1-epoch data audit pass** → clear Section D + I/J/K/L label items (G4).
4. Run a **1-epoch eval dry-run** on a capped val subset → clear Section M (G5).
5. Confirm **G6** (repro) and Section N.
6. Only then launch the **R-ladder** (guide §Part 4): R1 det → R1.5 anchors+synth →
   R2 activity → R2.5 PSR → R3 full-data → R4 head-pose/assembly → R5 multi-seed.

**If all 100 pass**, training will produce a non-zero, non-NaN, non-NA,
protocol-correct number for every task; PSR and head-pose are positioned to
match/beat their baselines; detection and activity are honest and competitive at
a fraction of deployed compute; and the efficiency/unification story — one
pipeline predicting all tasks in a single forward pass — is the defensible
headline of the paper. That is the unified multi-task model you set out to build.
