# Master Prompt — Second Opus Consultation (2026-06-11)

## Context: What Changed Since First Consultation

### First Opus Answer (`06_OPUS_ANSWER.md`)
You (Opus) identified 12 new root causes (RC-13 through RC-24) and proposed 11
surgical patches (P1-P11) plus an 8-step zero-GPU experiment plan.
The first answer is included in full at `06_OPUS_ANSWER.md`.

### What We Implemented
All 11 surgical patches were applied to the live codebase. 6 diagnostic scripts
(D1-D6) were written and executed against `latest.pth` (epoch 43). See
`07_DIAGNOSTIC_VALIDATION.md` for the summary table.

### Diagnostic Findings
- **RC-19 CONFIRMED** (D4): det_conf raw logits are effectively constant across
  frames (per-dim std/|mean| = 0.349%). This means the activity head receives a
  near-identical 24-D conditioning vector on every frame, regardless of what the
  detection head sees.
- **RC-17 DORMANT** (D5): Zeroing clip_rgb has zero effect on any activity metric.
  The VideoMAE features are already dead/identical across frames.
- **RC-16 HEALTHY** (D6): Attention is differentiated (off-diag mass=0.925,
  entropy=1.992 nats). P5 fix is intact.
- **RC-13 DENIED** (D1): All 3 checkpoints identical (cos=1.0). EMA was disabled.
- **RC-22 DENIED** (D2): Anchors are adequate for GT box sizes.

### Retrain Experiment
A 1-epoch recovery retrain was run at epoch 43 with --reinit-heads, all patches
applied, FP32, 5% subset, batch size 2. Results: **total collapse persists.**
See `08_RETRAIN_RESULTS.md` for full details.

- det_mAP50 = 0.0000 (predictions fire on background, never match GT)
- act_macro_f1 = 0.0000 (predicts 1/75 classes, class 8, 100% of frames)
- psr_overall_f1 = 0.0909 (1 binary pattern across all frames)
- Best combined = 0.1116 (pose-only — det/act/psr contribute zero)

### Model State After Retrain
The model checkpoint is at:
`src/runs/full_multi_task_tma_tbank_benchmark/checkpoints/latest.pth`

And the eval results at:
`src/runs/eval_post_retrain_fp32_20260610_194311/`

## The Core Question for Round 2

After applying all 11 surgical patches and running a 1-epoch retrain, the model
remains fully collapsed on all 3 functional heads (det/act/psr). The backbone
and pose head are alive. The detection head fires confident predictions on
background, the activity head collapses to 1 class, and PSR produces 1 pattern.

**Why do the patches fail to revive the heads?**

### Specific Questions

**Q1: Detection Collapse Mechanism**
The detection head fires ~15K predictions per batch at confidence >0.5 (score_max
~0.96), but zero match any GT box at IoU>0.5. The best IoU across all predictions
is ~0.27. The cls_loss is in the 10^7 range at epoch start.

Given that:
- Anchors are adequate (D2)
- cls_subnet/reg_subnet were re-initialized (P3)
- EMA shadow was re-anchored to fresh reinit weights (P1)

What mechanism causes the detection head to produce confident-but-wrong predictions
on a 5% subset after reinit? Is this:
(a) Insufficient training time (1 epoch)?
(b) A learning rate / optimizer state mismatch for reinit'd heads vs frozen backbone?
(c) A deeper architectural issue in the RetinaNet-style head itself?

**Q2: Activity Collapse After RC-19 Fix**
P7 applies sigmoid bounding to det_conf, which should prevent raw-logit-scale
features from dominating the activity_proj input. Yet the activity head still
collapses to 1 class. D4 confirmed det_conf variance is 0.35% of mean even AFTER
sigmoid — the sigmoid output is (0,1) bounded but the scores are near-constant.

If det_conf is near-constant even after sigmoid, the activity head still sees
identical conditioning regardless of frame content. Is the fix sufficient, or
should det_conf be:
(a) Zeroed entirely during recovery (activity head learns without detection signal)?
(b) Replaced with a different detection summary (e.g., max score + which class)?
(c) Gated by whether any detection exceeds a threshold?

**Q3: FeatureBank / RC-18 Interaction**
RC-18 was listed as "FeatureBank dead (video_ids=None always)." D6 showed healthy
attention (off-diag mass=0.925), which means tokens ARE differentiated — but D5
showed VideoMAE features are dormant (zero delta from zeroing clip_rgb).

If VideoMAE features are dead and FeatureBank returns the current frame 16×,
what is the source of token differentiation that D6 sees? Is it the GAP features
(c5_mod + p4) and det_conf that differ across frames, while the VideoMAE stream
is uniformly dead?

**Q4: 5% Subset Ceiling (RC-24)**
The opus answer identified RC-24: "5% subset is structurally capped." The 5%
subset has only 4 recordings (37→4), 3,112 frames. With 75 activity classes and
24 ASD classes, is it even theoretically possible for 3,112 frames (with ~4 GT
boxes per frame in early batches) to provide enough signal for:
(a) Detection: learning to localize 24 part classes
(b) Activity: discriminating 75 action classes
(c) PSR: learning 11 binary component states

Is RC-24 the dominant factor, or are the architectural issues (RC-14, RC-17,
RC-18, RC-19) the primary blockers even at larger subsets?

**Q5: Training Dynamics After Reinit**
The training log shows det cls_loss in the 10^7 range at epoch start, dropping
slowly. Adam m/v buffers were reset for reinit'd head params (P1 extension).
The Kendall log_vars were also reset to neutral.

With:
- Backbone frozen at collapsed-adjacent weights (43 epochs of collapse)
- Detection head freshly initialized with prior pi=0.05
- Adam with zeroed m/v on head params
- det cls_loss starting at 10^7

Is this training setup viable for recovery, or does the extreme initial loss
indicate a fundamental incompatibility between the frozen backbone features
and the reinit'd detection head?

**Q6: Priority for Second Recovery Attempt**
The opus answer Tier 1 plan was:
1. Fix measurement chain (P1-P8) ✓ DONE
2. Apply patches (P1-P11) ✓ DONE
3. Re-run at 25% subset, 3 epochs minimum ← NOT DONE

Before we commit GPU time to a 25% subset run, are there additional zero-GPU
fixes or diagnostics that should be done first? Specifically:
- Should det_conf be zeroed (not sigmoid-bounded) for the activity head input?
- Should the detection head architecture be verified against a known-working
  RetinaNet implementation?
- Should we verify the FPN features are actually non-degenerate (not just the
  backbone being "alive" in terms of non-NaN weights)?

## Files Provided

| File | Content |
|------|---------|
| `00_JOURNEY.md` | Complete development timeline |
| `01_WHAT_WE_BUILT.md` | Architecture and training details |
| `02_COLLAPSE_CRISIS.md` | How the collapse was discovered |
| `03_CURRENT_RECOVERY.md` | Recovery attempts before first Opus |
| `04_HYPOTHESES_FOR_OPUS.md` | Pre-Opus hypotheses |
| `05_MASTER_PROMPT.md` | First-round master prompt |
| `06_OPUS_ANSWER.md` | First-round Opus answer (325 lines) |
| `07_DIAGNOSTIC_VALIDATION.md` | Diagnostic script results |
| `08_RETRAIN_RESULTS.md` | Post-patch retrain results |
| `09_MASTER_PROMPT_v2.md` | This file |
| `code/` | Diagnostic scripts + reference source |
| `logs/train.log` | Retrain log (full) |
| `evidence/` | Eval results, metrics CSV, confusion matrix |

## Request

Please produce `10_OPUS_ANSWER_v2.md` with:
1. Analysis of why the 1-epoch retrain failed despite all patches
2. Answers to Q1-Q6 above
3. A revised recovery plan (zero-GPU experiments + retrain config)
4. Any additional root causes discovered from the new evidence
5. Priority-ranked next actions
