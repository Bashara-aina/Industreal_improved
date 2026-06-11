# POPW Architecture — What We Built

> Read this file before opening `code/model.py`. It is a guided tour.

## At a Glance

POPW is a **single ConvNeXt-Tiny + FPN backbone** that branches into **5 task
heads**, with **FiLM conditioning** carrying pose information into the
activity branch. The total parameter count is **76.16M** (53.4M trainable,
the rest are frozen BN stats and an auxiliary VideoMAE-Small stream).

```
                          Input frame (3, 720, 1280)
                                │
                ┌───────────────┴───────────────┐
                │  ConvNeXt-Tiny backbone       │
                │  (28.6M, ImageNet-pretrained)  │
                └───────┬───────┬───────┬───────┘
                        │       │       │       │
                       C3      C4      C5  (also a 2nd VideoMAE stream → 384-D)
                        │       │       │       │
                        └───┬───┘       │       │
                            │           │       │
                         FPN neck        │       │
                       (P3, P4, P5,     │       │
                        P6, P7)         │       │
                            │   │   │   │       │
              ┌─────┬───────┘   │   │   │       │
              │     │           │   │   │       │
         Detection  Pose     PSR  Head-Pose   Activity
          (RetinaNet-style) (T=4)  (9-DoF)    (CLS+TCN+2×ViT)
              5.30M    1.64M  3.73M  1.16M      8.44M
```

## The Backbone — `ConvNeXtBackbone` (model.py:164)

- `convnext_tiny(weights=IMAGENET1K_V1)`, 4 stages at strides 4/8/16/32.
- Outputs `C2, C3, C4, C5` with channel counts 96, 192, 384, 768.
- `use_checkpoint=True` for VRAM. Stage groups allow differential freezing.

## The FPN — `FPN` (model.py:378)

- Top-down pathway: `C5→P5`, `C5→P4 + C4→P4`, etc. 256 channels per level.
- Extra `P6`, `P7` via stride-2 conv on `C5`.
- Anchor sizes: `(24, 48, 96, 192, 384)`. 9 anchors per location (3 ratios × 3 scales).

## Detection Head — `DetectionHead` (model.py:488)

RetinaNet-style. Two subnets (cls + reg), 4× Conv3×3+ReLU + final Conv.
- Focal Loss (α=0.25, γ=2) for classification.
- GIoU Loss for box regression.
- **Current state:** collapsed. 64,950 predictions across 42 GT boxes in val
  (ratio 1546×) — the cls_score was saturated to ~1.0 everywhere.

## Body Pose Head — `PoseHead` (model.py:555)

- Input: `P3` (stride 8, 256ch).
- `ConvTranspose2d(4, 2, 1) + GN + ReLU` → stride 4.
- `Conv1×1 → heatmaps [B, 17, H/4, W/4]`.
- `Soft-Argmax(T=0.1) → keypoints [B, 34] + confidence [B, 17]`.
- Wing Loss (ω=0.05, ε=0.005) × 0.001.
- **Current state:** partly working. `position_MAE_mm = 739` is non-zero
  but huge (the model is at body scale, not mm scale — needs scale calibration).

## Head Pose Head — `HeadPoseHead` (model.py:1358)

- `GAP(C4) ‖ GAP(C5) → [B, 1152] → MLP(1152→512→256→9)`.
- LayerNorm + GELU + Dropout. MSE × 0.001.
- 9-DoF: 3 for forward direction + 3 for up direction + 3 for position
  (the position is later scaled ×100 to mm).
- **Current state:** partly working. Angular MAE 61° is large but non-zero.

## PoseFiLM — `PoseFiLMModule` (model.py:607)

The cross-task conditioner. Takes `(keypoints[34] ‖ confidence[17]) = [51]`.
- `γ-net: 51→512→768, output = 1 + tanh(·)` — range (0, 2).
- `β-net: 51→512→768`, unbounded.
- `C5_mod = γ · C5 + β` where `C5` bypasses the FPN.
- **Important:** γ/β nets do not get gradients from downstream — FiLM
  breaks the cycle.

## HeadPoseFiLM — `HeadPoseFiLMModule` (model.py:703)

- Same as PoseFiLM but condition is `head_pose[9]` with `stop_grad`.
- `γ_hp: 9→256→768`, `β_hp: 9→256→768`.
- `C5_mod2 = γ_hp · C5_mod + β_hp`.

## Activity Head — `ActivityHead` (model.py:1192)

The most complex head. Sequence model on top of per-frame features.
- **Per-frame input:**
  `f_joint = [det_conf(24) ‖ GAP(C5_mod2)(768) ‖ GAP(P4)(256)] = [1048]`
  where `det_conf = MaxPool(cls_preds)` is **stop-gradient** (paper says so).
- `W_proj: 1048→512` → `f̃_t [B, 512]`.
- **Feature Bank T=16** rolling buffer of `f̃_{t-T+1..t}`.
- `TCN`: 1D Depthwise Conv(k=5, dilation=1) — true depthwise per paper.
- **2× ViT blocks** with CLS token, learnable pos embed, MHSA(8 heads, d_k=64),
  FFN(512→2048→512), DropPath 0.10/0.15, pre-norm, attn_dropout=0.1.
- `CLS readout → Dropout(0.1) → Linear(512→74)` over 74 activity classes.
- LDAM-DRW loss with class-balanced reweighting (deferred to epoch 0 in our config).
- **Current state:** collapsed. All 74 outputs converge to the most common
  class. Bias init was 0 → symmetric → majority class.

## PSR Head — `PSRHead` (model.py:1399)

Multi-label binary classifier over 11 assembly components.
- **Multi-scale GAP** of P3, P4, P5 → concat → `MLP(768→256)`.
- **Causal Transformer:** 3 layers, 4 heads, d_model=256.
- 11 small MLPs (256→64→1) produce per-component logits.
- Binary Focal (α=0.25, γ=1.0) + temporal smoothness (w=0.05).
- **Sequence mode** (`PSR_SEQUENCE_LENGTH=4`): the head sees a window of 4
  frames in one forward, so the causal transformer can do useful work.
  Every 10th batch is a sequence batch.
- **Current state:** collapsed. The 11 binary outputs all stuck at 0.27
  sigmoid (bias=-1.0 in the dead version). Edit score 0.09 is just
  numerical noise (the GT has 0 transitions in val).

## VideoMAE Stream (model.py:23-31 of `__init__.py`)

Auxiliary frozen 384-D feature stream. We added it for the activity head
on the hypothesis that the activity benefits from motion features. It is
**frozen** and only consumed at eval-time in our current setup. We are
unsure whether to keep it; it adds 22M frozen params and some VRAM
pressure.

## Loss Aggregation — `MultiTaskLoss` (losses.py:816)

Kendall homoscedastic uncertainty:
```
L = Σ_t exp(-s_t) · L_t · ramp_t + s_t
```
- `s_det=0, s_pose=-1, s_act=0, s_psr=0` (init).
- Hard-clamped to `[-4, 2]` (we added this; without it, s_t drifts
  exponentially and the activity term is silently killed).
- Activity ramp `min(1, epoch/5)`.
- Plus loss caps: `ACTIVITY_LOSS_CAP=80, POSE_LOSS_CAP=30, PSR_LOSS_CAP=20`.

## Training Loop — `train_one_epoch` (train.py:788)

Per batch:
1. Forward through backbone+FPN+heads.
2. Compute per-head loss.
3. `MultiTaskLoss.forward()` returns the Kendall-weighted sum.
4. Backward.
5. EMA update (decay=0.999).
6. `NaNDetector` checks for inf/nan in any head — if found, save
   `crash_recovery.pth` and abort epoch.
7. Periodic val using `evaluate.py` (every 1 epoch on 5% subset).
8. Per-class activity sanity check (catches majority-class collapse).

## What Works Right Now

After the 39-epoch crash + the reinit fixes (see `03_CURRENT_RECOVERY.md`):
- Backbone features are alive (per-image variance > 0.001).
- Body pose head is in the right loss regime (loss cap 30, Wing Loss).
- Head pose head produces non-zero angular MAE (61° is bad but real).
- Detach + FP32 broke the seq-mode autograd poisoning.

## What Doesn't Work

- Detection, activity, PSR **all collapse to trivial solutions**.
- Combined 4-head metric is in the 5–11% range (vs the 80% target).
- The model reaches a 4-head checkpoint (39 epochs) and then dies.

## Targets vs Current (per `popw_paper_improved.tex`)

| Task | Paper baseline | Target | Our current |
|---|---|---|---|
| Detection bbox mAP@0.5 | YOLOv8m: 83.80% | close to 80% | 0.00% |
| Activity Top-1 | MViTv2 RGB-only: 65.25% | close to 80% (Top-5) | 0.00% |
| PSR F1 | PSRT-B2: 0.731 / STORM: 0.506 | close to 0.80 | 0.0000 |
| PSR POS | PSRT-B2: 0.816 | close to 0.80 | 0.0000 |
| Body pose (PCK@0.5) | not reported in paper | non-zero | 739 mm MAE |

(The "80%" target is the user's language for "competitive" — for some
metrics, "competitive" is the 0.7–0.8 range; for body pose, it is a
non-zero MAE in cm-scale. We have translated the user's intent
case-by-case in `04_HYPOTHESES_FOR_OPUS.md`.)

## File-to-Symbol Map (for Opus's grep convenience)

| What | Where |
|---|---|
| Backbone class | `code/model.py` — class `ConvNeXtBackbone` |
| FPN class | `code/model.py` — class `FPN` |
| Detection head | `code/model.py` — class `DetectionHead` |
| Pose head | `code/model.py` — class `PoseHead` |
| PoseFiLM | `code/model.py` — class `PoseFiLMModule` |
| HeadPoseFiLM | `code/model.py` — class `HeadPoseFiLMModule` |
| Head pose head | `code/model.py` — class `HeadPoseHead` |
| Activity head | `code/model.py` — class `ActivityHead` |
| PSR head | `code/model.py` — class `PSRHead` |
| Kendall-weighted loss | `code/losses.py` — class `MultiTaskLoss` |
| Focal / GIoU / Wing | `code/losses.py` — classes `FocalLoss`, `GIoULoss`, `WingLoss` |
| LDAM-DRW | `code/losses.py` — classes `LDAMLoss`, `ClassBalancedFocalLoss` |
| PSR focal + temporal | `code/losses.py` — class `PSRFocalLoss` |
| Train epoch | `code/train.py` — function `train_one_epoch` |
| Reinit dead heads | `code/train.py` — function `_reinit_dead_heads` |
| Eval entry | `code/evaluate.py` — top-level `main` |
| Re-eval | `code/eval_post_reinit.py` |
