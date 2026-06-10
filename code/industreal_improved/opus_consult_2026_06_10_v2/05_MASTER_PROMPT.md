# 05 — Master Prompt for Opus (Round 2)

> **Purpose:** This is the **single entry point** for the opus. It assumes
> the opus has read the prior 4 MDs (`00_JOURNEY.md` through
> `04_HYPOTHESES_FOR_OPUS.md`) and has access to the live source code
> under `code/`, the evidence under `evidence/`, and the logs under
> `logs/`.
>
> If you (the opus) have NOT read the prior MDs, STOP and read them in
> order. They contain the evidence that grounds every claim below.

---

## Mission

You are consulting on a multi-task ConvNeXt-Tiny model trained on the
IndustReal dataset for procedural IKEA assembly. The model has **3 dead
heads** (detection, activity, PSR) and **2 working heads** (body pose,
head pose) that **regressed** after a reinit retrain. The first opus
analysis identified 8 surgical patches; all were applied; a 1-epoch
FP32 retrain with `--reinit-heads` ran successfully. The result was
**partial recovery**: loss dropped 68%, PSR moved 8×, but `det_mAP50`
is still 0.0 and `act_top1` is still 0.0, while pose regressed by
11–73%.

Your job is to answer **3 questions** in priority order, propose **1
new patch + 1 retrain run**, and identify any **structural problems
we missed**.

---

## Question 1 (Priority 1): Why is `det_mAP50` = 0.0?

### Symptom

- `det_mAP50 = 0.0000` on val (50 batches, 200 samples)
- `det_mAP_50_95 = 0.0000`
- 12 of 50 batches in the DET_PROBE show "TOTAL COLLAPSE" verdict
- The model produces **4036 confident predictions per batch** (score
  range 0.01–0.97)
- `bestIoU_max` (over all 4036 preds × 42 GT) **never exceeds 0.27**
- `bestIoU_mean` is 0.05–0.08 (some overlap with GT, but never > 0.5)

### What this is NOT

- ❌ "Head didn't learn" — the model is CONFIDENT (scores 0.97)
- ❌ "Backbone is dead" — confirmed alive by `diag_features_alive.py`
  (variance 0.032–0.036 in DET logits after fresh head)
- ❌ "Post-retrain data corruption" — same val split used, same eval
  code, just `EVAL_SKIP_REINIT=1` to keep the trained weights

### What we need you to find

The bug is in **box decoding or post-processing**. The model produces
~4000 confident boxes per batch; they overlap with GT but are wrong by
a factor of 2–3 in either size or position. The four most likely
causes (with evidence locations):

1. **Box decoder stride mismatch** (H1.1 in `04_HYPOTHESES_FOR_OPUS.md`).
   Look at `code/model.py:DetectionHead.forward()` and the
   `decode_boxes` function. The model uses FPN levels P2/P3/P4 with
   strides 4/8/16/32. If the decoder picks the wrong level for a given
   anchor, the boxes are scaled wrong.
2. **Box coordinates in wrong frame** (H1.2). The DET_PROBE shows
   `bestIoU_mean = 0.06` which is consistent with a 2-3× scale error.
   Check `code/evaluation/evaluate.py` detection-eval section.
3. **NMS too aggressive** (H1.3). The eval uses `nms_iou_thresh=0.5`;
   if the only good box has IoU 0.6 with another pred, it's dropped.
   Test with `nms_iou_thresh=0.9` and see if `det_mAP50` jumps.
4. **Anchor format mismatch** (H1.4, H1.5). The dataset provides GT
   in `(x, y, w, h)` but the model is trained to predict
   `(x1, y1, x2, y2)`, or vice versa. Or the model is anchor-free
   (FCOS-style) but the dataset provides anchor-based targets.

### The experiment we want you to write

```python
# In code/, write: diag_det_box_decoder.py
# 1. Save 1 val batch (4 images, GT boxes in image-pixel coords)
# 2. Run model.forward() in eval mode
# 3. For each FPN level, dump the raw reg output + decoded boxes
# 4. Plot decoded boxes vs GT on the same image
# 5. Print stride + FPN level for each decoded box
# 6. Compute IoU per level; print which level is "right"
```

If the "right" level is the wrong one for what the model was trained
on, the fix is in the level-assignment logic. If ALL levels produce
bad boxes, the fix is in `decode_boxes` itself (or the anchor format).

---

## Question 2 (Priority 2): Why is `act_top1` = 0.0?

### Symptom

- `act_top1_accuracy = 0.0000`
- `act_top5_accuracy = 0.0600` (barely above random 0.05 for 1/20)
- `act_macro_f1 = 0.0000`
- `act_clip_accuracy = 0.0000`
- `act_frame_accuracy = 0.0000` (both with and without NA)
- `act_macro_recall = 0.0000`
- `act_mean_per_class_acc = 0.0000`

The fresh activity head (with `proj_features` reinit + 3-layer
TransformerEncoder reinit + `cls_token=0` reinit + final classifier
reinit) is collapsing to 1 class with 100% of frames. **The dominant
class is different from the pre-patch collapse** (was class 28 with
66.6%, now class 8 or similar).

### What this is NOT

- ❌ "Backbone features are not informative" — pose head works on the
  same features
- ❌ "Loss is zero" — total loss dropped 68%; activity loss component
  is non-zero
- ❌ "Reinit didn't run" — `_reinit_dead_heads` was called, 169
  tensors reinit, EMA shadow reset, optimizer state preserved

### What we need you to find

The bug is in **either the loss function or the activity-head
initialization**. The four most likely causes (with evidence
locations):

1. **LDAM-DRW margin underflow** (H2.1). For 75 classes with the
   LDAM-DRW class-balanced loss, the margin for low-prevalence classes
   is `m_i = max(0, margin_const * (1/n_i)^(1/4))`. For n_i=5,
   m_i=0.5. This is a large margin that the model can't satisfy for
   rare classes; the gradient is then dominated by the margin term
   only for common classes, and the model is forced to predict common
   classes for ALL frames. **Look at `code/training/losses.py` for
   the `LDAMLoss` class.**
2. **cls_token reinit to 0 kills gradient flow** (H2.2). With
   cls_token=0, the ViT's first attention layer has 0 input
   contribution from the cls slot. The gradient w.r.t. cls_token is
   the dot product of the gradient w.r.t. the ViT's output, which is
   small because backbone features dominate. **Look at
   `code/train.py` `_reinit_dead_heads` and the `cls_token.zero_()`
   call.**
3. **ViT attention collapse** (H2.3). With a fresh 3-layer
   TransformerEncoder + cls_token=0, all 3 attention layers collapse
   to uniform attention. The output is then a constant regardless of
   input. After 1 epoch, the attention weights haven't moved much
   because the gradient is dominated by the LDAM-DRW margin.
4. **Class-imbalanced sampler oversamples rare classes** (H2.4). The
   training sampler uses class-balanced sampling that oversamples rare
   classes so much that 80% of training samples are from the bottom
   20% of classes. The model only learns the hard negatives, can't
   escape the trivial solution.

### The experiment we want you to write

```python
# In code/, write: diag_activity_init.py
# 1. Load the post-retrain best.pth
# 2. Compute LDAM margin m_i for all 75 classes, print distribution
# 3. Check if any m_i > 0.5 (degenerate)
# 4. Save attention weights from the activity ViT at eval time
# 5. Plot the attention distribution for each head
# 6. If uniform → confirmed ViT collapse → fix is to warm-start
#    proj_features + add per-class bias init
# 7. If non-uniform but cls_token=0 → fix is to init cls_token to
#    torch.randn_like * 0.02 instead of zero
# 8. Run 1 forward+backward pass and dump the gradient w.r.t.
#    cls_token and proj_features.weight
```

---

## Question 3 (Priority 3): Why did pose regression get worse?

### Symptom

| Pose metric                  | Pre-retrain | Post-retrain | Δ       |
|------------------------------|-------------|--------------|---------|
| `position_MAE_mm`            | 739.5       | 823.5        | **+11%** |
| `head_pose_angular_MAE_deg`  | 61.04       | 71.50        | **+17%** |
| `forward_x_MAE`              | 0.1051      | 0.1821       | **+73%** |
| `forward_y_MAE`              | 0.0278      | 0.1280       | **+361%** |
| `forward_z_MAE`              | 0.9195      | 1.0134       | +10%    |
| `forward_angular_MAE_deg`    | 64.28       | 68.65        | +7%     |
| `up_angular_MAE_deg`         | 57.80       | 74.34        | **+29%** |

The pose head was working (MAE 0.35–0.42, the only non-zero metrics
on the broken model) and is now worse. **The MAE was already high
(0.35–0.42) and is now even higher.**

### What we need you to find

The most likely cause (H3.1) is that `_reinit_dead_heads` reinit'd
the pose MLP. The "dead heads" heuristic was supposed to match
det/act/psr only, but it may have matched pose tensors (e.g., the
module name `head_pose_branch` or any tensor with `pose` in its
name).

### The experiment we want you to write

```python
# In code/, write: diag_pose_reinit.py
# 1. Open the post-retrain best.pth
# 2. Print the names of all pose-related tensors
#    (filter: 'pose' in name OR 'head_pose_branch' in module path)
# 3. Compare their values to a fresh Kaiming-uniform init
# 4. If they MATCH fresh init → pose was reinit → fix is to
#    exclude pose from the reinit list
# 5. If they DON'T match → pose was NOT reinit → regression is
#    from backbone feature shift after 1 epoch
```

### Decision tree

```
Did pose tensors get reinit?
├── YES → exclude pose from reinit list, retrain 3-5 epochs
│        (don't touch pose, only retrain the OTHER reinit'd heads)
│
└── NO  → regression is from backbone shift → retrain 5-10 epochs
          with NO reinit (just let the heads adapt to the new
          backbone features)
```

---

## The patch we need from you (1 patch + 1 retrain)

### FIX-9 (Priority 1: det)

The minimum-change fix for the detection bug. Most likely candidates:

- A) Fix the FPN level assignment in `code/model.py:DetectionHead`
- B) Fix `decode_boxes` to use the right stride
- C) Loosen NMS to `iou_thresh=0.9` in `code/evaluation/evaluate.py`
- D) Add a coordinate-frame assertion at the end of `decode_boxes`

**Choose the smallest change that fixes `det_mAP50 > 0.05`.**

### FIX-10 (Priority 2: act, optional but recommended)

The minimum-change fix for the activity collapse. Most likely
candidates:

- A) Init `cls_token` to `torch.randn_like * 0.02` instead of 0
  (in `code/train.py:_reinit_dead_heads`)
- B) Lower the LDAM `margin_const` from 0.5 to 0.1 (in
  `code/training/losses.py:LDAMLoss`)
- C) Add a 1-epoch warmup with frozen backbone for the activity head
  only

**Choose the smallest change that gets `act_top1 > 0.10` (currently
0).**

### The retrain

After applying FIX-9 (and optionally FIX-10), run a 3-5 epoch retrain
with:

- `--reinit-heads` (re-init det/act/psr, but EXCLUDE pose)
- `--no-amp` (FP32 only; AMP fp16 is known broken in backbone)
- `--batch-size 2` (proven stable)
- `--subset-ratio 0.05` (proven stable; full data crashes PSR seq-mode)
- `--max-epochs 47` (resume from epoch 43 → 47)
- `--seed 42`

Use the existing `scripts/run_reinit_fp32_bs2.sh` as a template; just
change `--max-epochs` and the reinit list (or add a
`--reinit-heads-skip pose` flag).

---

## What we need from you (in order of priority)

1. **Validate the priority order.** Is det really #1? Or should we fix
   activity first because it's "easier"?

2. **Refine the hypotheses.** Add the hypotheses we missed in
   `04_HYPOTHESES_FOR_OPUS.md`. Reject the ones that are clearly not
   the issue. We've already done our homework — see R1–R9 in §"Priority
   0" of that file.

3. **Propose FIX-9 (det) and optionally FIX-10 (act).** The patches
   should be the minimum-change fix. We prefer 1 patch that solves 1
   problem over 1 patch that tries to solve 3.

4. **Estimate the recovery ceiling.** With all 4 problems fixed (det +
   act + psr + pose), what's the realistic val metric range? Baseline
   expectations for a ConvNeXt-Tiny + 75-class activity on IndustReal
   val: act_top1 ~ 0.3–0.5? det_mAP50 ~ 0.2–0.4?

5. **Identify any structural problems we missed.** Are there other bugs
   in `code/model.py` or `code/losses.py` that we haven't surfaced
   yet? The first opus already graded 12 hypotheses (see
   `02_COLLAPSE_CRISIS.md` §3); is there a 13th?

6. **Verify the pose reinit hypothesis.** Read `code/train.py`
   `_reinit_dead_heads` and confirm whether the heuristic matches
   pose tensors.

---

## Hard rules for your response

- **Cite code with file_path:line_number.** When you say "this is the
  bug", paste the offending line and the line you'd change.
- **Cite evidence files with relative paths.** When you say "this is
  confirmed", point to the diagnostic output (in `evidence/` or
  `logs/`).
- **Do NOT propose a 2-week refactor.** We need 1 patch + 1 retrain.
  Keep the change set small.
- **Do NOT re-investigate R1–R9 from `04_HYPOTHESES_FOR_OPUS.md`.**
  These are dead.
- **Do NOT recommend changing the data, the loss weights, the
  optimizer, or the learning rate** unless you have a specific reason
  grounded in evidence.
- **If you're unsure, say so.** "I don't know without running X" is a
  valid answer.

---

## File map for your reference

| Path                                            | What it is                                    |
|-------------------------------------------------|------------------------------------------------|
| `00_JOURNEY.md`                                 | Timeline + what changed opus 1 → opus 2        |
| `01_WHAT_WE_BUILT.md`                           | The 8 patches + scripts + env + auto-discovery |
| `02_COLLAPSE_CRISIS.md`                         | The 3-dead-head collapse, post-retrain picture |
| `03_CURRENT_RECOVERY.md`                        | Side-by-side metric table, per-head verdict   |
| `04_HYPOTHESES_FOR_OPUS.md`                     | Prioritized hypothesis list (Priority 0–4)     |
| `05_MASTER_PROMPT.md`                           | This file                                     |
| `code/train.py`                                 | Training loop (3733 lines)                     |
| `code/model.py`                                 | Multi-task model (2167 lines)                  |
| `code/losses.py`                                | Multi-task loss (1505 lines)                   |
| `code/evaluate.py`                              | Evaluation (4004 lines)                        |
| `code/config.py`                                | All hyperparameters                            |
| `code/eval_post_reinit.py`                      | Post-retrain eval entrypoint (130 lines)      |
| `code/apply_popw_fixes.py`                      | The 8-patch applier                            |
| `code/diag_*.py`                                | Diagnostic scripts (8 files)                   |
| `code/detection_collapse_probe.py`              | The DET_PROBE generator                        |
| `evidence/baseline_eval_post_reinit_v1/`        | Pre-retrain eval (EVAL_SKIP_REINIT=0)          |
| `evidence/post_retrain_fp32_20260610_194311/`   | Post-retrain eval (EVAL_SKIP_REINIT=1)         |
| `logs/retrain_5pct_fp32_bs2/train.log`           | Full 1.5-hour retrain log                      |
| `logs/eval_post_retrain_fp32_20260610_194311/`  | Eval log (28 KB)                               |
| `scripts/run_reinit_fp32_bs2.sh`                | The retrain script                             |
| `scripts/run_eval_post_retrain_fp32.sh`         | The eval script                                |
| `docs/`                                         | All project MDs + reports + contracts          |

---

## Quick-start if you have 5 minutes

1. Read `00_JOURNEY.md` (5 min)
2. Read `03_CURRENT_RECOVERY.md` §1 (the side-by-side metric table)
3. Read `04_HYPOTHESES_FOR_OPUS.md` §"Priority 0" (the 9 ruled-out
   hypotheses)
4. Skim `04_HYPOTHESES_FOR_OPUS.md` §"Priority 1" (the det
   hypotheses)
5. Decide: is H1.1 (box decoder stride) the most likely cause of
   det_mAP50=0? If yes, write the diagnostic and the patch.

That's it. The rest can wait.
