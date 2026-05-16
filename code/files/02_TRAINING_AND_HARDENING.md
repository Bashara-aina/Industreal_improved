# 02 — Training Schedule, Monitoring, and Architectural Hardening

Doc 01 listed the per-target interventions. This document covers the cross-cutting work that keeps those interventions actually working: a training schedule that respects the staged Kendall fix, monitoring hooks that catch silent failures (the kind that hid Bug #9 for so long), and a small list of remaining architectural decisions where the current design is correct but fragile.

The MASTER_BUG_REPORT identified one critical lesson from Bug #9: **a model can train for thousands of steps with no NaN, no error, no warning, and produce nothing of value.** This document's purpose is to make that class of failure impossible to hide.

---

## A. The complete training schedule

The training run that beats the benchmarks takes about 5–7 days end to end on an RTX 3060. Here is the full sequence with decision points.

### A.1 — Pretraining phase (overnight, day 0)

**Goal:** strong detection-only initialization.

```
python pretrain_synthetic.py
```

This runs detection-only training for `PRETRAIN_DET_EPOCHS=20` epochs (raise to 30 if val mAP@0.5 < 75% by epoch 15). Save best by val mAP@0.5. Output goes to `runs/pretrain_synthetic/checkpoints/best.pth`.

**What to watch for:**
- Train loss should drop from ~3.0 to ~0.5 by epoch 5, then plateau
- Val mAP@0.5 should rise monotonically — if it oscillates wildly, the LR is too high
- After epoch 20, expect val mAP@0.5 in the 75–82% range
- If it's below 70%, something is wrong with the data pipeline (check `OD_labels.json` paths)

**Decision point at epoch 20:**
- mAP@0.5 ≥ 75%: proceed to A.2.
- mAP@0.5 < 75%: extend to 30 epochs, drop LR to 1e-4 for the last 10 epochs.
- mAP@0.5 < 65% even after 30 epochs: data pipeline bug. Stop and investigate.

### A.2 — Smoke test (30 min, day 1 morning)

Before launching the multi-day main run, verify the staged Kendall logic actually works.

```
python train.py --debug --max-epochs 1 --resume runs/pretrain_synthetic/checkpoints/best.pth
```

The `--debug` flag (already in your code) limits to ~20 recordings and runs in 5–10 min. Watch the log carefully for these specific signals:

| Log line | What it means | Action if missing |
|---|---|---|
| `Backbone: convnext_tiny` | ConvNeXt active | Check `BACKBONE` config |
| `Optimizer: Lion` | Lion installed and active | `pip install lion-pytorch` |
| `[stage=1]` in pbar | Staged training engaged | Check `STAGED_TRAINING=True` |
| `pose=0.000, act=0.000, psr=0.000` in stage 1 logs | Bug #2 fix working | Bug #2 not fixed correctly |
| `log_var_det grad: <nonzero>` after step 1 | Bug #9 fix working | Bug #9 not fixed correctly |
| `log_var_pose grad: <nonzero>` after step 1 | Bug #9 fix working | Bug #9 not fixed correctly |
| Resumed checkpoint with `strict=False` warning about missing heads | Pretrain checkpoint loaded correctly | Resume path wrong |

The `log_var_*` gradient check is the explicit watchdog for Bug #9. Your training script should log gradient norms of the Kendall log_var parameters every N steps. If you don't have this, **add it** — it is the only signal that distinguishes a working Kendall from a broken one (the loss values look identical either way).

**Decision point after smoke test:**
- All checks pass: launch A.3.
- Any check fails: do not start the main run. Debug.

### A.3 — Main training (3–5 days, day 1 night onward)

```
python train.py \
  --resume runs/pretrain_synthetic/checkpoints/best.pth \
  --max-epochs 100 \
  --seed 42
```

The 100-epoch budget breaks down as:
- Stage 1 (epochs 1–5): detection-only, layer1–3 frozen
- Stage 2 (epochs 6–15): + pose + head pose, activity/PSR frozen
- Stage 3 (epochs 16–100): all four task groups active, EMA tracks all params
- DRW activates at `LDAM_DRW_EPOCH=35` (recommended change from default 60, see Doc 01 §B.2)

Expected wall-clock: 60–80 hours on RTX 3060 with `BATCH_SIZE=2`, `GRAD_ACCUM_STEPS=16`, `NUM_WORKERS=0`. With early stopping (`PATIENCE=10`), realistic 35–55 epochs.

**Per-epoch monitoring (do this in `train.py`'s end-of-epoch hook):**

Every 5 epochs, log:
- Each task's val metric (`det_mAP50`, `act_top1`, `act_top5`, `psr_f1_at3`, `head_pose_angular_MAE_deg`)
- Each `log_var_t` parameter value (Kendall uncertainty per task)
- Each task's loss contribution to the Kendall total (precision-weighted)
- EMA model val metrics in Stage 3 only
- `nan_skips` count from the training step (should be < 1% of steps; >5% means LR or loss scale problem)

**Decision points during main run:**

| Epoch | Watch for | If wrong, do |
|---|---|---|
| 5 (end of Stage 1) | `det_mAP50` ≥ pretrain start value | Stage 1 corrupted detection — check Bug #2 fix |
| 15 (end of Stage 2) | `det_mAP50` not regressed; head pose MAE dropping | Multi-task interference — lower head pose loss scale below 0.001 |
| 25 (Stage 3 + 10) | `act_top1` rising past random (>5%); PSR F1 rising | Activity warmup bug — check `act_ramp` math |
| 35 (DRW activation) | `act_top1` macro-F1 jumps within 2 epochs | DRW not engaging — check `LDAM_DRW_EPOCH` |
| 50 | All targets within 5pp of final goal | If not, will not recover — investigate |
| 70 | EMA val ≥ raw val on at least 3 of 4 tasks | EMA decaying wrong; lower decay |

### A.4 — SWA stabilization (optional, day 5)

If your main run plateaus before epoch 100 and you want a small additional boost:

```python
# In config.py
USE_SWA = True
SWA_EPOCHS = 8
SWA_LR = 1e-5
```

This runs 8 extra epochs after the main run with constant low LR, averaging weights via PyTorch's `torch.optim.swa_utils.AveragedModel`. Expected gain: +0.3 to +0.5% on most metrics, almost free.

### A.5 — Evaluation passes (day 5–6)

```
# Without TTA, 30 min
python evaluate.py --checkpoint runs/.../best.pth --split test

# With flip TTA, 1 hour
python evaluate.py --checkpoint runs/.../best.pth --split test --flip-tta

# With flip + 5-crop TTA, 2 hours
python evaluate.py --checkpoint runs/.../best.pth --split test --flip-tta --crop-tta
```

Report all three configurations in your paper. The "honest" headline is no-TTA; the "best possible" headline is with both. Reviewers respect papers that show both.

### A.6 — Multi-seed for statistical defensibility (week 2)

```
python run_multi_seed.py --seeds 42,123,7 --epochs 100
```

Three full runs at different seeds, evaluated identically, mean ± std reported. **Do not skip this.** A single-seed result is not publishable. If compute is tight, 2 seeds is acceptable with a noted caveat.

---

## B. Monitoring hooks that prevent silent failures

Bug #9 went undetected because nothing in the logs distinguished a working run from a broken one. The fix is to add explicit signals for each "load-bearing" mechanism in the architecture.

Add these hooks to `train.py`'s training step or end-of-epoch logger.

### B.1 — Kendall gradient sentinels

Every N steps (N=100 is fine), compute and log the gradient norm of each Kendall log_var parameter:

```
log_var_det.grad.norm() , log_var_pose.grad.norm(), log_var_act.grad.norm(), log_var_psr.grad.norm()
```

These should all be nonzero in stages where the corresponding task is active. If `log_var_act.grad.norm() == 0` during Stage 3, the Kendall computation graph is broken (Bug #9 reincarnated).

### B.2 — Stage transition assertion

At the start of each epoch:

```
print(f"[Epoch {epoch}] stage={get_stage(epoch)}, "
      f"trainable_params: backbone={n_backbone}, head_pose={n_hp}, act={n_act}, psr={n_psr}")
```

You expect:
- Stage 1: `backbone` partial (layer4 only), `head_pose=0`, `act=0`, `psr=0`
- Stage 2: `backbone` partial, `head_pose=full`, `act=0`, `psr=0`
- Stage 3: all `>0`

If a head shows `>0` trainable params during a stage where it should be frozen, the `_set_stage_requires_grad()` call is broken.

### B.3 — Loss component breakdown

Every step, the Kendall total should decompose into:

```
total = exp(-s_det) * L_det + s_det
      + exp(-s_pose) * L_pose * train_pose + s_pose
      + exp(-s_act) * L_act * act_ramp + s_act
      + exp(-s_psr) * L_psr + s_psr
```

Log each weighted component separately. In stage 1 you expect three of the four components to be ~`s_t` (just the precision penalty, no contribution from `L_t` because of `prec_t = 0`). In stage 3 all four should be in the same order of magnitude (~0.5 to 5.0). If any component dominates by 10×, the Kendall balance is broken — manual loss scaling needed.

### B.4 — Per-class activity sanity

Every 10 epochs of Stage 3, log the top-5 hardest activity classes by per-class F1. Expected pattern: rare classes (small samples) have low F1, frequent classes have high F1. If a frequent class has low F1, there's a class-name-mismatch bug somewhere (frame label points to the wrong class index).

### B.5 — PSR component prevalence sanity

For PSR, each of the 11 components has a different "appears as 1" prevalence. Log the prediction prevalence per component versus the ground truth prevalence. They should match within ±5%. If component 10 (wheels) has predicted prevalence 90% but GT prevalence 25%, your model has collapsed to predicting 1 everywhere — focal loss or reweighting is broken.

### B.6 — EMA tracking check

Once per epoch in Stage 3, compute val metrics on both the raw and EMA model. The EMA should track within ±2% of raw and pull ahead by epoch 30. If the EMA val is consistently worse than raw, the EMA decay is too fast for the parameter movement (lower from 0.999 to 0.998).

### B.7 — Validation stride alignment

The known issue from MASTER_BUG_REPORT §9 (validation loss vs eval discrepancy). Your in-loop validation uses a different augmentation pipeline than `evaluate.py`. **Use `evaluate.py` only for the final headline numbers.** In-loop val is fine for tracking progress and triggering early stopping; do not report it in the paper.

---

## C. Architectural hardening — what's correct but fragile

These items are not bugs. They are working as designed. But each has a fragility that could quietly cost performance under specific conditions. Address before the multi-seed runs.

### C.1 — Activity ramp interaction with Kendall in Stage 2

From MASTER_BUG_REPORT "Potential Remaining Bugs" §1: in Stage 2, `act_ramp = min(1, epoch/5) = 1.0`, so `loss_act` is multiplied by 1.0. Then `prec_act = 0` (Bug #2 fix), so the contribution is `0 * loss_act + lv_act = lv_act`. This trains `log_var_act` even though activity is "frozen" in Stage 2.

This is **correct Kendall behavior** — the precision goes to 0 (variance to infinity) which is what you want for a task with no useful signal yet. But it means `log_var_act` drifts during Stage 2, and when Stage 3 starts, the precision is at whatever value the optimizer drifted it to, not at the initial `s_act=0`.

**Hardening:** in `train.py`, at the start of Stage 3, reset `log_var_act` to a sensible value (probably `0.0` for fresh activity precision):

```
if epoch == STAGE1_EPOCHS + STAGE2_EPOCHS + 1:  # first epoch of stage 3
    criterion.log_var_act.data.fill_(0.0)
    criterion.log_var_psr.data.fill_(0.0)
```

This is a 4-line change. Without it you may see slightly suboptimal activity precision in Stage 3.

### C.2 — EMA decay during stage transitions

From MASTER_BUG_REPORT "Potential Remaining Bugs" §2: EMA tracks all parameters from epoch 0, including those that don't receive gradients in Stages 1–2. With `decay=0.999`, those frozen parameters' EMA values stay at ImageNet-pretrained values for ~700 steps before the EMA catches up to actual training-time values when Stage 3 begins.

For activity and PSR heads, "ImageNet-pretrained values" means random initialization (these heads are not pretrained). The EMA model in Stage 2 has random activity/PSR head weights even though the raw model has the same random weights. By Stage 3 epoch 25, the EMA should match — but Stage 3 evaluations before that are unreliable for activity/PSR.

**Hardening:** delay EMA initialization to the start of Stage 3, or reset EMA buffers at the Stage 2→3 transition:

```
if epoch == STAGE1_EPOCHS + STAGE2_EPOCHS + 1:
    ema = EMA(model, decay=C.EMA_DECAY)  # fresh EMA from current model state
```

### C.3 — Wing Loss with `TRAIN_HEAD_POSE = False`

From MASTER_BUG_REPORT Bug #8: when `TRAIN_HEAD_POSE = False` (the IndustReal default), the Kendall computation uses `_loss_pose_staged = loss_head_pose`, and the Wing Loss path is dormant. This is correct.

**Hardening:** the `WingLoss` instance is still created and `pose_loss_fn` is still on the GPU. This wastes ~1 MB of VRAM but more importantly, if any code path (in `evaluate.py` or visualizers) tries to call the pose head with hand keypoint data, it will produce garbage because the head was never trained. Either:
- Set `pose_head` to `None` in the model when `TRAIN_HEAD_POSE = False`, and gate all calls
- Or accept that pose head outputs on IndustReal are random, document this, and don't report any pose metric

The cleaner choice is to gate. ~10 lines in `model.py` to handle the `None` case in `forward()`.

### C.4 — PSR cache contamination across recordings

The PSR cache is keyed by `(video_id, camera_view)`. During eval, frames arrive in order per recording, so the cache fills correctly. But if your test split iterates over multiple recordings in a single batch, the cache for recording A may still hold partial data when recording B starts. This is correct in principle (different keys), but **only if `video_id` is unique per recording**, which depends on how `industreal_dataset.py` produces metadata.

**Hardening:** at the start of evaluation on each new recording, call `model.psr_head.reset_sequence(video_id, camera_view)`. Add this to your `evaluate.py` evaluation loop. Without it, very long evals can hit `_MAX_CACHE_LEN=32` boundaries that misalign with recording boundaries, producing wrong sequence-level metrics (POS, F1@T) for the first few frames of recordings 2+.

### C.5 — Mixed precision interaction with Kendall

From MASTER_BUG_REPORT "Potential Remaining Bugs" §6: `MIXED_PRECISION = False` is the current setting, but if you turn it on for speed, the Kendall `exp(-s_t)` computation can overflow in FP16 when `s_t` drifts negative (precision becomes huge). PyTorch's GradScaler handles loss scaling but not parameter overflow.

**Hardening:** clamp `s_t` more aggressively in mixed precision, e.g., `s_t = clamp(log_var, -2, 2)` instead of `(-4, 2)`. Or keep mixed precision off, which is the simpler option for a final run.

### C.6 — Anchor sizes vs synthetic-pretrain anchor sizes

If you ran `pretrain_synthetic.py` before `calibrate_anchors.py`, the pretrain checkpoint was built with the default anchors `(32, 64, 128, 256, 512)`, not the calibrated `(24, 48, 96, 192, 384)`. The detection head's bounding-box regression is then trying to encode boxes against anchors that don't match what the calibrated config now uses.

**Check:** verify `pretrain_synthetic.py` reads `C.ANCHOR_SIZES` at runtime, not at import time. If it imports the calibrated value, you're fine. If it hardcodes `(32, 64, 128, 256, 512)`, your pretrain checkpoint is mis-calibrated.

If mis-calibrated: re-run `calibrate_anchors.py` first, ensure config has the new sizes, then re-run pretrain.

---

## D. Architecture sanity vs the LaTeX papers

`popw_paper.tex` describes the architecture in §3. I cross-checked against current `model.py`. The implementation matches the paper for:

- ConvNeXt-Tiny backbone, channels 96/192/384/768
- FPN lateral 1×1 (192/384/768 → 256), P3–P7
- Detection: 24-class RetinaNet on P3–P7
- PoseFiLM: 51 → 512 → 768 (γ and β)
- HeadPoseFiLM: 9 → 256 → 768 (γ_hp and β_hp)
- Activity: f_det[24] || GAP(C5_mod2)[768] || GAP(P4)[256] = 1048-D, projected to 512-D
- Feature Bank T=16, TCN, 2× ViT (8 heads, d_k=64), CLS readout
- HeadPoseHead: GAP(C4)+GAP(C5) = 1152 → 512 → 256 → 9
- PSRHead: multi-scale FPN GAPs → 768 → 256 → 128 → causal Transformer (3 layers, 4 heads, d=128) → 11 per-component MLPs

The architecture as paper-described and as code-implemented are aligned. **No revisions needed**.

The one architecture-level question worth deciding before the multi-seed runs: should HeadPoseFiLM's input be the raw `head_pose` prediction or detached? The paper says `[stop_grad]` and the code applies it correctly via `head_pose.detach()` in the model forward. Confirmed aligned.

---

## E. What this document doesn't fix

These are limitations you have to either accept or address separately:

1. **No IKEA-ASM dataset module.** The paper claims dual evaluation but only IndustReal is implemented. Doc 03 covers paper-side handling.

2. **PSR per-frame training.** Doc 01 §D covers the proper fix (sequence-mode training). Without it, the PSR temporal Transformer is dormant during training.

3. **Hand keypoint loss.** With `TRAIN_HEAD_POSE = False`, no body/hand keypoint loss is trained on IndustReal (correctly — the dataset has no COCO keypoints). The Wing Loss code is dormant. If you ever want to evaluate hand pose on IndustReal, you'd need to use the published HoloLens hand joint annotations and add a separate hand pose head — out of scope for this round.

4. **VideoMAE V2 weight licensing.** The HuggingFace checkpoint `MCG-NJU/videomae-small-finetuned-kinetics` is Apache 2.0. Your paper should mention you used it and cite the VideoMAE V2 paper. No license issue.

5. **Synthetic data for pretraining.** The IndustReal authors publish synthetic training frames. Using their published synthetic split is what the YOLOv8m baseline does, so you should too for fair comparison. If you don't use it, your detection result is on a slightly different protocol than the 83.80% you're trying to beat.

The next document (Doc 03) addresses paper-level framing: how to present the results, what to claim and what to caveat, and how to make the paper publishable even if PSR sequence-mode training doesn't land in time.
