I'll dig into this package thoroughly. Let me start by mapping the repo and reading the entry documents.

The package is all here (note: `industreal_dataset.py` is at `code/` root, not `code/data/`). Reading the entry documents first.

Now reading the numbered analysis docs.

Now the code — the critical part. Checking sizes, then reading everything.

Reading the detection eval pipeline in evaluate.py first (Q1 territory).

Now the box decode/AP functions and `_prepare_images` in evaluate.py, plus `compute_activity_metrics`.

EMA weights are swapped in for validation at stage≥3 (train.py:3178-3181). Now checking what gets saved as `best.pth`, and the seq-mode PSR target handling in the train loop.

I have the full code picture now. Before writing the final analysis, let me ground the recovery-ceiling question (Q12) in the actual IndustReal literature.

I've now read the complete package — every numbered MD, all 17 code files end-to-end (model.py, losses.py, train.py, evaluate.py, config.py, the dataset, eval entrypoint, diagnostics), both evidence runs, the retrain and eval logs line-by-line, the contracts, and the supporting docs. The findings below change the diagnosis substantially: **the headline problem is not in the box decoder or the loss — it's that you never evaluated the model you trained.**

---

# Executive summary

**The single most important finding (new root cause, call it RC-13):** `best.pth` contains **EMA weights, not the trained weights** (train.py:3433-3437), and the EMA shadow for the re-initialized heads was never actually reset — the "reset" at train.py:2636 is `ema.shadow[_n] = ema.shadow[_n].clone().detach()`, a **no-op that copies the old shadow into itself**. Then, three lines later, train.py:2650-2655 restores the checkpoint's epoch-43 `ema_shadow` — the **collapsed weights** — over the shadow anyway. With 3,112 train samples at bs=2 ≈ 1,556 EMA updates at decay 0.999, the saved checkpoint is ≈ `0.999^1556 ≈ 21%` collapsed-old weights + an exponentially-weighted average of a 1-epoch trajectory that itself starts at random init.

The evidence trail is conclusive:

- `logs/retrain_5pct_fp32_bs2/train.log:441` — the official val swapped to EMA weights; `train.log:495-497` shows that EMA model predicting **class 28 at 66.6% of frames — the exact pre-patch collapse signature** — while the raw (actually trained) model at `train.log:606-608` predicts class 8.
- `train.log:649` — `EMA vs Raw delta — psr_f1=-0.0909`: the raw model **had** psr_f1 = 0.0909; the EMA blend destroyed it. The EMA version is what got saved as best.
- `logs/eval_post_retrain_fp32_20260610_194311/eval.log:22-31` — `score_p50 ≈ 0.0 / 1e-39` on every batch: the **median** detection logit in the evaluated checkpoint is ≈ −89, which is the collapsed head's signature (your own diag found "logits mean=-83"), not what a freshly re-initialized (bias −2.94) + 1-epoch-trained head produces. The `RuntimeWarning: overflow encountered in exp` at eval.log:20-21 says the same thing.

Until this is fixed, **every post-retrain metric in `03_CURRENT_RECOVERY.md` is a measurement of a corrupted blend**, and neither "det is still broken" nor "the 8 patches worked" can actually be concluded from it.

**Second correction to your narrative:** the PSR "8× recovery" is an artifact. `eval.log:73` says PSR still produces **1 unique binary pattern across all 200 frames**. The constant pattern happens to be `[1,0,0,...,0]`, and the eval slice (the **first 200 frames of val, sequential**, since `eval_post_reinit.py` builds the loader with `shuffle=False` and `MAX_BATCHES=50`×4) has comp0=1 nearly everywhere (comp0 train prevalence = 1.0, train.log). Constant prediction ∩ skewed slice = edit_score 0.73 and comp0 F1 = 1.0. Same story for `act_top5 = 0.06`: constant logits give a *fixed* top-5 set {8, 73, 74, 71, 70}; 12 of 200 GT labels fall in that set. It is not "above random" or "below random" — the right reference is the GT marginal mass of that fixed set, and OQ-1's "systematic exclusion" inference is unfounded.

**Third correction:** pose was **never re-initialized**. H3.1 is refuted by reading the code, not by experiment: `_reinit_dead_heads` (train.py:1579-1740) touches only `detection_head.cls_score/reg_pred`, activity (`proj_features`, `cls_token`, `vit`, `activity_classifier`, `tcn`), and PSR (`per_frame_mlp`, `output_heads`). No tensor whose name contains `pose` is touched. The log line "169 head tensors" (train.log:58) is misleading — the actual count is 9 submodules (train.log:56), and the EMA "reset" it refers to did nothing.

---

# The 3 blocking questions (required format)

```
Q1: NEW HYPOTHESIS — H1.1–H1.5 all rejected as primary cause. Three stacked causes:
    (a) evaluated checkpoint is EMA-contaminated with collapsed weights [RC-13],
    (b) the detection trunk was never re-initialized: the reinit looks for
        cls_tower/reg_tower which don't exist — the model names them
        cls_subnet/reg_subnet [RC-14],
    (c) the reg head received almost no gradient: GIoU loss g=0.0000000 on the
        large majority of steps (most frames have no GT; 4-recording subset).
     Evidence: code/train.py:2636 (no-op EMA reset), code/train.py:2650-2655
        (collapsed shadow restored AFTER reinit), code/train.py:3433-3437 (best.pth
        = EMA), code/train.py:1623 (cls_tower/reg_tower) vs code/model.py:508-509
        (cls_subnet/reg_subnet)
     Evidence: logs/eval_post_retrain_fp32_20260610_194311/eval.log:22-31
        (score_p50≈1e-39 = collapsed-logit signature), logs/retrain_5pct_fp32_bs2/
        train.log:649 (EMA strictly worse than raw), train.log step lines
        ("g=0.0000000" on most batches)
     Fix: make the EMA reset copy param.data (or set USE_EMA=False for the recovery
        run), save/eval raw weights, and add cls_subnet/reg_subnet to the reinit list.
     Risk: LOW — config/one-line changes; blast radius is the recovery run only.
```

Why the decoder/post-processing hypotheses are dead: the DET_PROBE itself decodes **raw boxes, pre-NMS, at threshold 0.01**, using a decode (`evaluate.py:66-76`) that is byte-for-byte equivalent to the training decode (`losses.py:169-191`) and the model decode (`model.py:1800-1822`). NMS (H1.3) and score thresholds never touch the `bestIoU` numbers. GT is pixel-space xyxy (dataset `industreal_dataset.py:1003-1027`), anchors are pixel-space xyxy (model.py:451-482), and the train-time matcher's [0,1] normalization (losses.py:124-135) is IoU-invariant (uniform axis-wise scaling of all boxes). Both model and targets are anchor-based RetinaNet-style — H1.4/H1.5 rejected. The one *real* residual anchor concern: `ANCHOR_SIZES=(24,48,96,192,384)` (config.py:249) versus the k-means GT statistics in the comment directly above it (config.py:243-247: w=146–594px, centers 164–404px) — only P6/P7 (2,700 of ~173k anchors per image, 1.6%) can reach IoU≥0.5 with typical GT. That caps how fast detection can learn and should be checked with the anchor-coverage diagnostic below, but it is not what makes mAP exactly 0.

```
Q2: NEW HYPOTHESIS — composite, none of H2.1–H2.5 is the primary driver:
    (a) same EMA contamination as Q1 [RC-13];
    (b) Mixup/CutMix mix the OUTPUT LOGITS, not the inputs, and LDAM argmaxes the
        soft label — so a large fraction of activity batches train on the WRONG
        frame's label; CutMix (active during the retrain: epoch 43 is odd) has no
        lam-gating at all [RC-15];
    (c) the standalone eval zero-fills the VideoMAE half of the classifier input
        (collate_fn_sequences has no clip_rgb) while training fed real features
        [RC-17];
    (d) det_conf (raw, unbounded max logits from the still-collapsed det trunk)
        dominates the activity-head input, making it near-constant across frames
        — your own diag measured L2 243.39 ± 0.001 [RC-19].
     Evidence: code/train.py:407,469-470 (logit mixing; images_mixed at 464-465 is
        never fed to the model), code/losses.py:491-495 (argmax of soft labels),
        code/eval_post_reinit.py:63 + code/industreal_dataset.py:1299-1383 (no
        clip_rgb in collate_fn_sequences) + code/model.py:1347-1348 (zeros_like
        fallback), code/model.py:1944-1945,1968-1972 (raw det_conf into proj)
     Evidence: logs/retrain_5pct_fp32_bs2/train.log:495-497 vs 606-608 (EMA model
        = old class-28 collapse; raw model = class-8 collapse), eval.log:72
        (class 8, 100% of frames)
     Fix: disable Mixup/CutMix (USE_MIXUP=False, CUTMIX_ALPHA=0) until they mix
        inputs; eval with the clip-aware collate; sigmoid-bound det_conf; evaluate
        raw weights.
     Risk: LOW for the disables/eval fix; MED for the det_conf sigmoid change
        (changes the activity input distribution — do it only together with the
        head reinit, which you're doing anyway).
```

```
Q3: H3.1 REFUTED, H3.2 (backbone shift) + RC-13 (EMA blend) are the cause;
    H3.3 (noise) is a contributing factor that hasn't been excluded.
     Evidence: code/train.py:1579-1740 — _reinit_dead_heads contains no pose
        tensors; pose_head/head_pose_head never matched (so "the reinit list
        included pose" is false)
     Evidence: logs/retrain_5pct_fp32_bs2/train.log step lines — activity loss
        oscillating 5→70 through the END of the epoch (LDAM s=30 under cutmix
        label corruption) hammered the shared backbone for 1,556 steps; best.pth
        backbone = EMA blend of old + drifting backbone, while head_pose_head
        stayed matched to the OLD backbone features
     Fix: no pose-specific fix needed — evaluate latest.pth (raw weights) and
        re-eval with MAX_BATCHES≥200; if raw pose ≈ baseline pose, the regression
        is entirely an artifact of the saved blend.
     Risk: LOW — it's an evaluation, not a change.
```

**Priority-order validation (your meta-question):** No — detection is not step 1. Step 1 is **fixing the measurement** (RC-13 + RC-17), because right now you cannot evaluate anything, including whether the 8 patches worked. It costs zero GPU-hours: `latest.pth` already contains the raw weights from the same run.

---

# Full numbered root-cause list (new findings, RC-13 onward)

Continuing your numbering from the 12 graded hypotheses in `02_COLLAPSE_CRISIS.md` §3:

**RC-13 — EMA shadow never reset + restored from collapsed checkpoint + best.pth saves EMA.** (train.py:2636, 2650-2655, 3433-3437.) Mechanism and impact above. This invalidates the entire post-retrain evidence set. Also note the in-training "best" decision itself was made on EMA metrics (train.py:3178-3181), so even checkpoint *selection* was driven by the contaminated model. **Impact: total — it's why "nothing moved."**

**RC-14 — Detection reinit misses the trunk.** train.py:1623 iterates `('cls_tower','reg_tower')`; the model's modules are `cls_subnet`/`reg_subnet` (model.py:508-509). Your own forensics (train.py:1583-1584 docstring) concluded "conv weights inside the head also collapsed" — yet only the two final 3×3 convs (`cls_score`, `reg_pred`) were reset. A fresh 0.01-std final conv on top of a collapsed 8-conv trunk explains the bimodal score distribution at eval (median 1e-39, max 0.97): the trunk emits huge-magnitude features. **Impact: high for det.**

**RC-15 — Mixup/CutMix corrupt activity labels.** `mixup_activity`/`cutmix_activity` (train.py:377-486) blend `act_logits` *after* the forward pass (train.py:407, 470) — that is not mixup; the model never sees mixed inputs. CutMix builds `images_mixed` (train.py:464-465) and **never runs it through the model**. The mixed soft target is then argmax'd by LDAM (losses.py:491-495), so whenever `lam < 0.5` the loss supervises frame i's logits with frame j's label. CutMix has no `0.3 ≤ lam ≤ 0.7` gate (unlike mixup, train.py:399-400), uses `Beta(1,1)=U(0,1)`, and was active for the whole retrain (gate at train.py:1136-1139: epoch 43 ≥ 5, stage 3, odd epoch → cutmix). With bs=2, this is a coin-flip label swap on a huge share of activity batches. A fresh head trained one epoch under ~50% label noise on 4 recordings *will* collapse to a constant class. **Impact: high for act.**

**RC-16 — Inverted attention scaling in the activity ViT.** `ViTTemporalBlock` (model.py:1097-1098): `scale = self.head_dim ** -0.5; attn = torch.matmul(q, kᵀ) / scale` — dividing by `d^-0.5` multiplies the attention logits by `√d = 8` instead of dividing, i.e. logits are **64× larger** than standard. After any training, softmax saturates to near-one-hot and gradients through attention vanish (the H2.3 "attention collapse" mechanism, but with a concrete cause). This affects **only** the activity head — PSR uses `nn.TransformerEncoder` (correct internal scaling, model.py:1453-1462), pose heads are MLPs — which matches the observed per-head pattern. Mitigated today only by the fact that RC-18 makes all tokens identical anyway. **Impact: medium now, high once the feature bank works.**

**RC-17 — Train/eval input mismatch on the VideoMAE half of the activity classifier.** `USE_VIDEOMAE=True` (config.py:72) → classifier input is 1024-d (model.py:1279-1288). Training passes real clips (train.py:1125-1128; `collate_fn` includes `clip_rgb` at industreal_dataset.py:1292-1294). The standalone eval uses `collate_fn_sequences` (eval_post_reinit.py:63, because `USE_PSR_SEQUENCE_MODE=True` — even though the dataset is *not* in sequence mode), which omits `clip_rgb` entirely → `model.forward` gets `clip_rgb=None` → `feat = cat([feat, zeros_like(feat)])` (model.py:1347-1348). Half the classifier input is zeroed at eval but real in training. Both your "baseline" and "post-retrain" evals share this flaw. **Impact: high for evaluated act metrics; also explains part of the EMA-val (class 28, clips present) vs standalone-eval (class 8, clips zeroed) discrepancy.**

**RC-18 — The FeatureBank is dead in both training and eval.** Every call site invokes `model(images, clip_rgb=...)` with `video_ids=None` (train.py:975, 1128; evaluate.py:2876), so `FeatureBank.forward` takes the fallback at model.py:1148-1150 and returns the **current frame replicated 16×**. The TCN+2×ViT therefore process 17 near-identical tokens; the activity head is effectively a per-frame MLP and the "T=16 temporal context" in the architecture description does not exist at runtime. (Only `compute_efficiency_metrics`, evaluate.py:2616+, ever passes `video_ids`.) **Impact: caps activity ceiling; explains why the elaborate temporal machinery contributes nothing.**

**RC-19 — `det_conf` is raw, unbounded logits and couples det collapse into the activity head.** model.py:1944-1945 takes `cls_preds.max(dim=1)[0]` — max raw logits, not probabilities — and concatenates it with GAP features (~O(0.1–1)) at model.py:1968-1972. With the collapsed det trunk the 24 `det_conf` dims have enormous, frame-invariant magnitude; your own diag (quoted in train.py:1585-1588) found the activity input constant at L2 243.39 ± 0.001. Activity collapse is *downstream of detection*. A `torch.sigmoid()` here bounds it to [0,1]. **Impact: high for act, and it makes "fix det first" genuinely load-bearing for act.**

**RC-20 — The "combined" best-model metric is pose-only in practice (solves OQ-6 exactly).** `combined = 0.30·mAP50 + 0.35·act_f1 + 0.15·(1/(1+head_pose_MAE)) + 0.20·psr_f1` (train.py:123-126, 1774-1789). With the first, second and fourth terms at 0: `0.15/(1+0.344) = 0.1116`. Your two observed values are identical because head-pose raw MAE is the only living term and it barely moves. Consequence: checkpoint selection cannot reward det/act/psr recovery until they exceed 0; with PATIENCE=10 the scheduler is effectively "save best pose." **Impact: medium — wrong selection pressure.**

**RC-21 — `MATCH_PROBE` can never fire.** `probe_anchor_matching` is called with `_state=None` each time (losses.py:230), so `_state` is recreated per call, `n` is always 1, and `1 % 200 != 0` → the positive-anchor-count diagnostic you built to answer exactly this kind of question has never logged a single line (confirmed: 0 matches in train.log). **Impact: low directly, but it blinded you to the reg-gradient starvation.**

**RC-22 — Anchor/GT scale mismatch (the surviving sliver of H1.1).** config.py:249 vs the k-means stats at config.py:243-247, detailed under Q1. Also the neg-IoU config value is dead: `FocalLoss` is constructed without `neg_iou_thresh` (losses.py:899), so the default 0.2 is used, not `DET_NEG_IOU_THRESH=0.25` (config.py:251). **Impact: medium for det learning speed; the dead config value is cosmetic.**

**RC-23 — The eval slice is unrepresentative.** `eval_post_reinit.py` builds the full 35,084-frame val set (it never passes `subset_ratio`/`max_recordings`) but evaluates the **first 200 frames sequentially** (`shuffle=False`, 50 batches × bs 4). That slice contains 42 GT boxes, all of one class (`background`, eval.log:93-117 — hence `det_n_present_classes=1`), mostly-constant PSR state, and a narrow activity label set. The 12 TOTAL-COLLAPSE / 38 NO-GT split (your OQ-2 said 26; the actual log count is 38) is a property of the slice, not the model. **Impact: medium — inflates PSR, deflates everything else, and makes run-to-run comparisons fragile.**

**RC-24 — The training subset cannot support the task.** `--subset-ratio 0.05` → **4 recordings, 3,112 frames** (train.log header), with ~12 of 75 activity classes present (train.log:496 `gt_seen=12/75`) and very few GT-box frames (the `g=0.0000000` lines). Even a bug-free run cannot learn a 75-way classifier or a 24-class detector from this. The class-balanced `WeightedRandomSampler` (train.py:227, industreal_dataset.py:1172-1193 — H2.4 is real but secondary) reshuffles a 12-class pool. **Impact: hard ceiling on any retrain at 5%.**

Honest note on FIX-1…FIX-8: I read `apply_popw_fixes.py` targets and the patched code paths; the patches themselves look correctly applied (e.g. the dim-5 sequence gate at model.py:1857-1862, per-position PSR predictions at model.py:2004-2017, the temporal-smooth sign fix at losses.py:1233-1243). But because of RC-13 the post-retrain eval neither confirms nor refutes their efficacy.

---

# The 8 open questions

- **OQ-1 (top5 = 0.06 "below random"):** solved — fixed top-5 set from constant logits; the comparison to 1/15 random is meaningless. See executive summary.
- **OQ-2 (12/50 vs NO-GT):** solved — it's 12 GT-bearing batches (b0–b9, b16, b17) / 38 NO-GT, all GT in one class, a property of the sequential first-200-frame slice (RC-23). The "12 OK-ish batches" you hoped to study don't exist — all 12 GT batches are TOTAL COLLAPSE.
- **OQ-3 (LDAM 75 vs 74):** root-caused and benign. `eval_post_reinit.py:101` constructs `MultiTaskLoss()` with the default `num_classes_act=74` (losses.py:843) instead of `C.NUM_CLASSES_ACT=75`; train.py:2402 passes 75 correctly. At forward time `_fit_to_width` (losses.py:470-484, used at 524-525, 536-539) sizes margins/weights to the actual logits width (75), and the 75-entry counts match — no misalignment, no bias. One-line fix for hygiene.
- **OQ-4 (efficiency NaN):** intentional. `SKIP_EFFICIENCY_METRICS=True` (config.py:532) + the epoch gate at evaluate.py:3408-3420 writes `float('nan')` placeholders; `_print_single_run_results` then crashes on the absent `eff_trainable_params_m` (eval.log:198-199). Not a profiler crash. Set the flag False (or pass an epoch divisible by `LOG_EFFICIENCY_EVERY`) when you want the numbers.
- **OQ-5 (det_precision/recall missing):** they are **never computed anywhere**. They exist only as initialized-to-0.0 attributes of the unused `EvaluationMetrics` class (evaluate.py:376-377); the CSV schema has the columns but they're empty. Nothing was sanitized out. If you want a recall signal, the DET_PROBE's `bestIoU>0.5` count is the closest existing proxy.
- **OQ-6 (combined = 0.1116):** solved exactly — RC-20.
- **OQ-7 (which class dominates):** class **8** (100% of frames) in the evaluated checkpoint (eval.log:72); the raw in-training val also shows class 8 (train.log:606-608); the EMA val shows class 28 at 66.6% (train.log:495-497). The full confusion matrix *is* saved — `act_confusion_matrix` and `act_per_class_report` are in `evidence/*/metrics.json` (compute_activity_metrics, evaluate.py:840-853) plus `confusion_matrix.png`; the per-class console report just isn't printed for non-GT classes. Note the GT top-5 in the val slice is {6, 0, 7, 30, 1} and class 8 is not among them — class 8 is *not* the prior's argmax, consistent with "constant garbage from near-constant inputs" (RC-19) rather than "learned the prior" (H2.1's prediction).
- **OQ-8 (pose: reinit vs shift vs noise):** reinit is excluded by code (Q3). Remaining split between backbone-drift+EMA-blend and noise is decided by the free experiment: eval `latest.pth` (raw) on ≥800 frames.

---

# Remaining questions (13–18)

**Q13/Q14 (pose reinit heuristic):** answered under Q3 — the heuristic does **not** match pose tensors; the docs' claim of an over-inclusive list is wrong, as is the "169 tensors" count (9 submodules; the 169 figure comes from the no-op EMA loop's counter at train.py:2637-2640, which counted shadow keys it *didn't actually reset*).

**Q15 (LDAM warning):** benign — OQ-3 above.

**Q16 (data format / H1.4–H1.5):** confirmed consistent — dataset GT is pixel xyxy (industreal_dataset.py:1009 converts COCO xywh→xyxy; category −1 at 1012-1017), anchors pixel xyxy, encode/decode are matched inverses (losses.py:150-191), model and targets both anchor-based. The contracts' claims here match the code. The genuine format-adjacent issue is anchor *sizes* (RC-22), not format.

**Q17 (PSR 1-of-11):** same constant-output collapse family as activity, **not** a separate trunk/specialization issue, and not a recovery. The "trunk learned comp0" theory in 03_CURRENT_RECOVERY.md §2 is contradicted by eval.log:73 (one unique pattern — there is no input-dependence to specialize). Aggravators: fill-forward PSR labels are near-constant within recordings (industreal_dataset.py:450-492), so on 4 recordings a constant output is genuinely near-optimal for the focal loss; the temporal-smooth loss can no longer fire in T=1 mode (it requires `psr_logits.dim()==3`, losses.py:1220, and the T=1 path emits dim-2); and the sensitivity penalty is capped at a 0.05 contribution (losses.py:1216-1218). PSR mostly needs data/epochs and an uncontaminated checkpoint.

**Q18 (contracts vs code):** the 7 contracts in `docs/contracts/` describe a **much older system** — ResNet-50 backbone, an ActivityHead that is "GAP→FC, 74 classes," PSR as "C5 GAP → FC(11) with BCE," HeadPoseHead "C5 GAP→FC(9), L1." The live code is ConvNeXt-Tiny + FiLM×2 + TCN/ViT/CLS activity head (75 classes) + causal-transformer PSR + binary focal. Treat the contracts as historical scaffolding, not as a spec to verify against; specifically `contract-03-model.md`'s `num_classes=74` and `contract-04`'s BCE-for-PSR are both superseded by the code (config.py:151-165's analysis of the 74/75 hazard is the current source of truth). Nothing in the contracts identifies an additional live bug.

---

# Surgical patches (proposed, in order)

| # | File:line | Change | Why | Risk |
|---|---|---|---|---|
| P1 | `train.py:2636` | `ema.shadow[_n] = dict(model.named_parameters())[_n].data.clone()` (and move this **after** the line-2650 shadow restore, or skip restoring shadow for reinit'd prefixes) | RC-13 | LOW |
| P2 | `config.py:294` | `USE_EMA = False` for the recovery run (simplest belt-and-suspenders; re-enable for long runs once heads are alive) | RC-13 | LOW |
| P3 | `train.py:1623` | `for tower_attr in ('cls_subnet', 'reg_subnet'):` | RC-14 | LOW |
| P4 | `config.py:297` + `config.py:498` | `USE_MIXUP = False`, `CUTMIX_ALPHA = 0.0` (the correct long-term fix — mixing `images` *before* the forward — is a bigger change; don't do it now) | RC-15 | LOW |
| P5 | `model.py:1098` | `attn = torch.matmul(q, k.transpose(-2, -1)) * scale` | RC-16 | LOW (heads are being reinit anyway) |
| P6 | `eval_post_reinit.py:63` | `collate_fn = _ds_module.collate_fn` (the val dataset is never in sequence mode here) | RC-17 | LOW |
| P7 | `model.py:1945` | `det_conf = torch.sigmoid(cls_preds).max(dim=1)[0]` | RC-19 | MED — changes activity input distribution; safe only bundled with head reinit (which this run does) |
| P8 | `eval_post_reinit.py:101` | `MultiTaskLoss(num_classes_act=C.NUM_CLASSES_ACT, num_psr_components=C.NUM_PSR_COMPONENTS)` | OQ-3 hygiene | LOW |
| P9 | `losses.py:230` | give `probe_anchor_matching` a module-level `_STATE = {}` default (or pass `_state=_PROBE_STATE`) so it actually logs every 200th call | RC-21 | LOW |
| P10 (optional) | `losses.py:899` | pass `neg_iou_thresh=C.DET_NEG_IOU_THRESH` | RC-22 cosmetic | LOW |
| P11 (defer) | `train.py:1128` / `evaluate.py:2876` | pass `video_ids=[m['recording_id'] for m in targets['metadata']]` to engage the FeatureBank | RC-18 | MED-HIGH — bank stores detached train-mode features across steps; engage only after the heads are alive, and validate train/eval symmetry first |

Anchor sizes (RC-22): don't change blind. Run D2 below first; if it shows the best-achievable anchor IoU is the binding constraint, change `ANCHOR_SIZES` to the k-means centers `(64, 128, 192, 288, 416)`-ish — but that invalidates the det head's learned scale priors, which the reinit makes acceptable.

---

# The zero-GPU-cost experiment to run FIRST

Before any retrain: **re-evaluate `latest.pth`** (raw end-of-epoch weights from the same run — it's already on disk, train.py:3465-3481) with P6/P8 applied and `MAX_BATCHES=200`:

```bash
EVAL_CKPT=.../checkpoints/latest.pth EVAL_SKIP_REINIT=1 EVAL_SPLIT=val \
EVAL_BS=4 MAX_BATCHES=200 RUN_NAME=eval_raw_latest \
python eval_post_reinit.py
```

This tells you, today, how much the 1-epoch retrain actually learned, whether pose "regression" survives on raw weights and a bigger slice, and gives the true baseline for the next retrain.

# Retrain configuration

After P1–P8: re-run `--reinit-heads` from `crash_recovery.pth` (epoch 43). My recommendation, with the reasoning made explicit so you can trade off:

- **Subset 0.25, 3 epochs** (≈ 7.5 h/epoch at bs=2 on the 3060 → ~22 h) rather than 0.05 × more epochs. RC-24 means epochs at 5% mostly re-fit 12 classes and a handful of GT boxes; data breadth is the binding constraint, not step count. If 22 h is unacceptable, 0.10 × 4 epochs is the floor I'd accept for a meaningful det/act readout.
- `--no-amp` (keep), `--batch-size 2` (keep), `--seed 42`.
- **Do NOT reset optimizer state** — but know the R9 spike will recur; your kill-criteria stand. (Resetting Adam state for the reinit'd params only would be cleaner, but it's a code change with its own risk; skip.)
- `USE_EMA=False` for this run (P2). With EMA off, in-training val and best.pth both measure the real model, and RC-13 can't recur.
- Leave LDAM/LR/Kendall alone for this run — per your own hard rule, and because with RC-15 removed there's no evidence yet that LDAM is misbehaving. (If activity is still flat *after* this clean run, the next suspects in order are: LDAM `s=30` with label smoothing, then the sampler H2.4 — both now properly testable.)

```bash
python training/train.py \
  --resume <ckpt_dir>/crash_recovery.pth \
  --reinit-heads --no-amp \
  --subset-ratio 0.25 --batch-size 2 \
  --max-epochs 47 --seed 42
```

Then eval **both** `best.pth` and `latest.pth` with the fixed collate and `MAX_BATCHES=200`.

# Diagnostic scripts (one per unresolved hypothesis)

- **D1 `diag_ema_contamination.py`** — Input: `best.pth`, `latest.pth`, `crash_recovery.pth`. For each reinit'd tensor (e.g. `detection_head.cls_score.weight`), print cosine similarity of best-vs-latest and best-vs-crash. Verdict: best≈crash (cos > 0.5) on head tensors ⇒ RC-13 confirmed quantitatively. (Confirms/denies the residual "maybe best.pth is fine" doubt.)
- **D2 `diag_det_anchor_coverage.py`** — Input: all val GT boxes + the AnchorGenerator output. For every GT, compute max IoU over all anchors (identity regression upper bound). Output: histogram + p50/p90. Verdict: p50 best-anchor IoU < 0.5 ⇒ anchors are a binding constraint (act on RC-22); p50 > 0.6 ⇒ anchors fine, the problem is purely training/contamination.
- **D3 `diag_det_level_scores.py`** — Input: 10 GT-bearing val batches through `latest.pth`. Per FPN level: count of scores>0.5, best decoded IoU. Verdict: confident preds concentrated on P3/P4 (small anchors inside big GT, IoU ceiling ≈ 0.25 — which would exactly explain `bestIoU_max≈0.24-0.27`) ⇒ the cls head is firing on the wrong levels; GT-matched levels P6/P7 silent ⇒ trunk/level assignment issue.
- **D4 `diag_act_input_variance.py`** — Input: 200 val frames through `latest.pth`; record `activity_proj` (model.py:1968-1972) std-across-frames per dim, with `det_conf` as-is vs zeroed vs sigmoided. Verdict: std/mean < 1% with raw det_conf and ≥10× higher with sigmoid ⇒ RC-19 confirmed.
- **D5 `diag_videomae_zero.py`** — Eval the same checkpoint twice on the same 200 frames: clip_rgb real vs forced-zeros. Verdict: any act metric delta ⇒ RC-17 magnitude measured.
- **D6 `diag_attention_saturation.py`** — Forward 50 frames; dump max attention weight per head per ViT block before and after P5. Verdict: pre-fix max-weight ≈ 1.0 (one-hot) and post-fix < 0.5 ⇒ RC-16 was biting (only meaningful after P11/feature-bank engagement, since identical tokens give uniform attention regardless).
- **D7 pose A/B** — covered by the zero-cost `latest.pth` eval at `MAX_BATCHES=200` (decides H3.2 vs H3.3 directly).

# Recovery ceiling (Q12)

Grounded in the IndustReal paper's own baselines ([WACV 2024 paper page](https://timschoonbeek.github.io/industreal.html), [arXiv:2310.17323](https://arxiv.org/abs/2310.17323); numbers below retrieved from the [HTML render](https://ar5iv.labs.arxiv.org/html/2310.17323) via a summarizer — spot-check Tables 2–4 before quoting them anywhere):

- **Action recognition:** SlowFast ≈ 58–60 top-1, MViTv2 ≈ 62–65 top-1, multi-modal ensemble 66.45 — with full training data and clip-level protocol.
- **Assembly state detection:** YOLOv8-m mAP ≈ 0.573 (synthetic only) → 0.753 (real fine-tune) → 0.838 (both).
- **PSR:** baseline B3 F1 ≈ 0.883, POS ≈ 0.797.

For *your* architecture and protocol my estimates (clearly labeled as estimates):

- **Full training data, all patches applied, heads alive:** `act_top1` ≈ **0.30–0.50** frame-level (a single-RGB ConvNeXt-Tiny multi-task model with a per-frame head — RC-18 — should land meaningfully below a dedicated MViTv2 clip model; engaging the feature bank/VideoMAE properly is what would push toward the upper end). `det_mAP50` ≈ **0.35–0.60** (RetinaNet-style with ConvNeXt-Tiny is competitive with YOLOv8-m in principle, but anchor calibration and the tiny-GT-density regime will cost you; 0.84 is not realistic without the synthetic pretrain stage). `psr_overall_f1` ≈ 0.5–0.8 given PSR states are slowly-varying.
- **On the planned 0.25-subset, 3-epoch recovery run:** treat `act_top1 ≥ 0.10`, `det_mAP50 ≥ 0.05–0.10`, `>10` unique PSR patterns, and `pred_seen ≥ 15/75` classes as the success bar — i.e., the goal of that run is *“heads demonstrably alive and metrics responsive,”* not paper numbers.
- **On the current 4-recording 5% subset: don't bother measuring ceilings** — 12/75 classes present is a structural cap (RC-24).

---

## What I'd correct in your assumptions, compactly

1. "Pose was reinit" — false (code-verified).
2. "PSR recovered 8×" — false; still a 1-pattern collapse flattered by the eval slice.
3. "act_top5=0.06 means systematic bias" — artifact of constant logits + slice marginal.
4. "Box decoder/post-processing is the prime suspect" — exonerated; the probe already bypasses post-processing, and decode paths are identical everywhere.
5. "169 tensors reinit, EMA shadow reset" — 9 submodules, detection trunk excluded, EMA reset a no-op then overwritten.
6. "The 8 patches fixed it / didn't fix it" — undetermined: the post-retrain eval measured an EMA blend with half the activity input zeroed, on an unrepresentative 200-frame slice. Fix the measurement first; it's free.

Here are the points I'd actually bet on, ordered by leverage. They come in three tiers: **make the machine trustworthy → redesign the parts that can't win → add the things that beat the baselines.** The targets to beat are concrete: YOLOv8m 83.8 ASD mAP (b-boxed), MViTv2 65.25 activity Top-1 (RGB-only, clip-level), B2 PSR F1 0.731 / POS 0.816, and head pose (no baseline — free win).

---

## Tier 1 — Trust (without this, nothing else is measurable)

**1. Fix the measurement chain before touching architecture, and never let it regress.**
You already have the list: EMA no-op reset + collapsed shadow restore (train.py:2636, 2650-2655), best.pth = EMA blend, eval collate dropping `clip_rgb`, the detection trunk excluded from reinit, mixup/cutmix mixing *logits* instead of images. Add to that a permanent invariant test suite that runs before every training launch: (a) decode(encode(gt)) == gt, (b) train-collate and eval-collate produce identical target keys, (c) the checkpoint you evaluate is byte-identical to the weights that produced the val metric, (d) one overfit-a-single-batch test **per head** (each head must reach ~0 loss on 8 samples in <200 steps — this catches dead gradients in minutes, not days). Your last month was lost to exactly these classes of bug; the test suite is the highest-ROI "architecture" change available.

**2. Delete the defensive machinery and simplify the loss to a boring, proven recipe.**
The codebase has ~15 layers of NaN guards, smooth caps, sensitivity penalties, staged Kendall zeroing, and sentinel flags. These don't fix bugs — they *hide* them and distort gradients (several of your collapse modes were caused or masked by them). Replace with: fixed per-task weights (tune once: det 1.0, act 1.0, psr 1.0, pose 0.1 — uncertainty weighting only after everything trains stably), plain CE + label smoothing 0.1 for activity (drop LDAM s=30 — it's a 30× logit amplifier on top of class-balanced sampling you already have; one imbalance mechanism, not three), AdamW + cosine, grad-clip 1.0, and **assert-and-crash** instead of guard-and-continue. If a loss goes NaN, you want to know that step, not at epoch 43.

**3. Per-task model selection, not the broken `combined`.**
`combined` is mathematically pose-only today (0.15/(1+MAE)=0.1116). Save `best_det.pth`, `best_act.pth`, `best_psr.pth` plus a combined with calibrated weights. Otherwise your training never keeps the checkpoint your paper tables need.

---

## Tier 2 — Redesign the parts that structurally can't win

**4. Two-stage training via a frozen-backbone embedding cache — this is the single biggest unlock on a 12GB RTX 3060.**
Your fundamental constraint is 1.5h/epoch at *5%* data. You will never train temporal heads on full data by pushing every frame through ConvNeXt every epoch. Instead: (Stage A) train backbone + spatial heads (det, head pose) on the frames that matter; (Stage B) run the frozen backbone **once** over the full dataset, cache per-frame embeddings (512-d × ~2M frames ≈ 4GB on disk), then train the activity and PSR temporal heads over **long real sequences (T=64–256)** from the cache at hundreds of epochs per hour. This is exactly how feature-bank papers (LFB, MeMViT-style pipelines) actually train. It turns your dead FeatureBank into the centerpiece, fixes RC-18 by construction (sequences come from real video order), and makes the streaming-FPS efficiency story in §\ref{sec:efficiency} of your paper real: at inference, one backbone pass per frame, temporal heads run on cached embeddings at O(1).

**5. Detection: stop fighting RetinaNet's regime — go ROI-centric for assembly *state*.**
ASD is not COCO. It's 0–3 **large** objects (146–594px) per frame where the hard part is the *fine-grained state* (bolt tight vs loose), not localization. Dense 24-class anchors at 173k locations is the wrong shape for this. Restructure: (a) a class-agnostic localizer — a single-class anchor-free head (FCOS/CenterNet-style, P5–P7 only) that finds "the assembly object" (easy, large object), then (b) ROI-Align a high-resolution crop (from P3, or even the raw image at 224²) into a **state classification head** (24-way). This converts the impossible problem (dense fine-grained detection) into two easy ones, and the state classifier output is *exactly* what PSR needs (see point 7). If you keep RetinaNet instead, then at minimum: anchor sizes from your own k-means (≈64/128/192/288/416, not 24–384), ATSS assignment instead of fixed IoU thresholds, and P4–P7 only.

**6. Use the synthetic data — the baseline you're chasing did.**
YOLOv8m's 83.8 is **COCO + synthetic + real**. IndustReal ships ~260k synthetic ASD images, and your config already has `PRETRAIN_DET_ON_SYNTH=True` wired but unused. Pretrain the localizer + state classifier (+backbone) on synthetic, fine-tune on the real annotated frames (which are a small subset — det training is cheap if you sample only b-boxed frames + synth, not every video frame). Without this you are bringing 5% of the data to a fight the baseline won with 100% + synthetic. This is the difference between det mAP 0.3 and det mAP 0.7+.

**7. PSR: predict *transitions* with a monotonic state accumulator, not per-frame binaries.**
Read what wins: B2 (F1=0.731) is ASD-confidence **accumulation + procedure-order constraints** — barely a neural model. STORM-PSR (a fancy dual-stream transformer) gets only 0.506. The metric is F1 on *state-change events* within ±3 frames, and the state space is monotone fill-forward (components get placed and stay placed). So: feed the ROI state-classifier outputs (point 5) into a small causal transformer over cached embeddings (point 4) that predicts **per-component transition events** $p(\text{comp}_i \text{ flips at } t)$, decoded with a monotonic constraint + procedure-order prior (a component can only flip 0→1 once; order follows the procedure graph). Train with event-detection loss (Gaussian-smeared transition targets), not per-frame BCE — per-frame BCE on 95%-static labels is *why* your head learns the constant pattern. This design is a learned strict-superset of B2 and is your most realistic "beat the paper" claim: target F1 > 0.75, POS > 0.82.

**8. Activity: make a K400-pretrained video stream the primary path and align training to the clip protocol.**
The benchmark is **clip-level Top-1 over 16 uniform frames per action segment**. You currently train per-frame on NA-dominated labels and evaluate frame-level — protocol mismatch costs you double-digit points before architecture even matters. Restructure: sample 16-frame clips *from labeled action segments* (NA excluded or its own class, matching the paper), train a Kinetics-pretrained video encoder — your cached VideoMAE-S is fine to start; VideoMAE-v2 or MViTv2-S if VRAM allows after point 4 frees memory — **fine-tuned, not frozen**, with the CNN stream (det_conf, FiLM-modulated GAP) as auxiliary conditioning, and one classifier on the fused feature. Per-frame ConvNeXt + 2 tiny ViT blocks over a fake bank will not reach 65%; a fine-tuned K400 video transformer + your cross-task conditioning genuinely can (the baseline is RGB-only 65.25 — your multi-task signals are the edge that gets you past it).

---

## Tier 3 — The extra points that close the gap

**9. Knowledge distillation from the dedicated baselines into your unified model.**
This is the cleanest path to the paper's thesis ("one model matches N specialists"). Run YOLOv8m (fine-tuned on IndustReal, weights are reproducible from the paper recipe) and your best activity teacher over the training set once; distill: soft detection targets (logit + box distillation on the localizer/state head) and soft activity logits (KL at T=2–4) alongside the hard labels. Distillation routinely recovers 2–5 points that multi-task interference costs you, and it's nearly free at your scale since teacher inference is offline and cached.

**10. Fix the cross-task conditioning so it's information, not noise.**
Keep the FiLM idea (it's your paper's identity) but repair the inputs: `det_conf` must be `sigmoid()`-bounded (raw logits made the activity input constant — that was a root cause of your collapse); body-keypoint FiLM on IndustReal currently consumes *hand-joint pseudo-keypoints with confidence=1* — either wire actual hands.csv data with honest confidences or gate PoseFiLM off for IndustReal; HeadPoseFiLM keeps stop-grad. Then *prove* each FiLM stage in the ablation table with the cache pipeline (cheap now) — if a stage doesn't help, cut it; a clean negative ablation is publishable, a noise injector is not.

**11. Head pose: switch to a geometry-aware parameterization — this is your uncontested table row.**
9 raw numbers with MSE produces 60–70° angular MAE (barely better than chance for directions). Predict rotation as a **6D continuous representation** (Zhou et al., CVPR 2019) or unit-normalized forward/up with cosine/geodesic loss, position as a separate normalized 3-vector. Expect angular MAE to drop to 10–25° with no baseline to beat — instant headline row, and better FiLM conditioning input for free.

**12. Task-aware sampling instead of "every frame, every head."**
Det trains on annotated/synthetic frames only; activity trains on segment-sampled clips; PSR trains on transition-dense windows (oversample ±N frames around state changes — they're the only frames carrying gradient for the event loss); pose trains on every kth frame. One unified model does not require one unified dataloader — a round-robin multi-loader with per-task batches removes the NA/no-GT gradient starvation that flattened your retrain (`g=0.0000000` on most steps) and cuts epoch time by ~3–5×.

---

## Honest target assessment

| Target | Verdict |
|---|---|
| PSR F1 0.731 / POS 0.816 (B2) | **Most beatable** — points 4+5+7; B2 is a heuristic your model can learn to dominate |
| Head pose (no baseline) | **Free** — point 11 |
| Activity 65.25 Top-1 RGB-only | **Achievable** with point 8 (fine-tuned video stream + protocol alignment + your conditioning); the frozen-VideoMAE per-frame design will not get there |
| ASD mAP 83.8 (b-boxed) | **Hardest** — requires point 6 (synthetic pretrain) + point 5 (ROI design); realistic outcome 0.70–0.85; "within ~3 points of YOLOv8m at 1/3 the deployed compute" is still a winning paper claim |
| Efficiency (params/GFLOPs/FPS) | **Wins by construction** with point 4's streaming design — quantify it, it's your safest contribution |

**Execution order:** Tier 1 this week (it's mostly deletions + the test suite). Then point 4 (cache pipeline) because every later experiment becomes 10× cheaper. Then 5+6 (detection), then 7 (PSR rides on detection), then 8 (activity), with 9–12 interleaved as the ablation table fills in. If you want, I can turn any of these into concrete code on a branch — the cache pipeline (point 4) and the invariant test suite (point 1) are the two I'd start with.


Sources: [IndustReal project page](https://timschoonbeek.github.io/industreal.html), [arXiv:2310.17323](https://arxiv.org/abs/2310.17323), [ar5iv HTML render](https://ar5iv.labs.arxiv.org/html/2310.17323).