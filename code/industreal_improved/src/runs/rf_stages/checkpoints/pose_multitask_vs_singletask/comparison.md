# Pose Multi-Task vs Single-Task Comparison

**Date:** 2026-07-07
**Source checkpoint:** `best.pth` (epoch 18)
**Evaluator:** Agent 65 — Pose Single-Task vs Multi-Task Comparison Specialist
**Motivation:** Opus 141 Q50: "remove multi-task-benefit language unless single-task pose ablation runs"

---

## Multi-Task Pose Numbers (38,036 frames, 16 recordings)

| Metric | Forward (deg) | Up (deg) | Source |
|--------|:------------:|:--------:|--------|
| Weighted mean MAE | **9.14** | **7.78** | `full_eval_ep18_v2/metrics.json` |
| Per-recording mean MAE | 9.29 | 7.73 | `pose_kalman_eval/pose_kalman_results.json` |
| Per-recording median MAE | 8.94 | 7.58 | same |
| Kalman-smoothed weighted | 9.00 | 7.58 | same |
| 95% CI (bootstrap) | [7.74, 10.87] | [6.89, 8.81] | `bootstrap_ci.json` |
| Per-recording std | 2.64 | 1.74 | `pose_kalman_results.json` |
| Frame-level std | 7.85 | 6.84 | `pose_error_stats.json` |

---

## Single-Task Pose Baseline: NOT AVAILABLE

**No single-task pose ablation has been run.** The config preset `ablation_pose_only`
exists (`config.py:1711-1741`) with architecture identical to stage_rf4, but no
checkpoint or evaluation results exist for it.

Preset details:
- Backbone: ConvNeXt-Tiny with TMA cell + temporal bank + hand FiLM
- `train_det=False`, `train_act=False`, `train_psr=False`, `train_head_pose=True`
- Batch size 6, grad accum 4, EMA on, mixed precision off
- Uses `use_geo_head_pose=True` (same as multi-task)

The `ablation_single_task` preset (`config.py:1572-1603`) includes detection +
head_pose (no activity/PSR). It too has no results directory.

---

## Single-Task Estimate

### Gradient allocation in multi-task

From training log epoch 24 (`train.log` line ~54569):
- USE_KENDALL = True with HP_PREC_CAP active
- Effective head_pose precision: **0.54** (capped to match detection precision)
- Raw head_pose precision (without cap): 2.70
- head_pose contributes roughly **19-25%** of total weighted loss signal
- Body pose (Wing loss) is non-zero but sentinel — no keypoint annotations in IndustReal

Head pose head RMS gradient: 4.05e-02 (ALIVE)
Backbone total RMS gradient: 5.43e-03

### Estimated single-task improvement

**Forward: 5-7 degrees** (vs 9.14 multi-task, 25-45% better)
**Up: 5-6.5 degrees** (vs 7.78 multi-task, 16-35% better)

Rationale:
1. Pose is a simple 9-DoF regression from strong ConvNeXt features — the backbone
   already extracts good visual representations. Additional gradient share may not
   transform the feature quality as much as for harder tasks.
2. The best-performing recordings already hit ~6 degrees (24_assy_2_4: 6.07 fwd,
   26_assy_1_5: 6.08 fwd) — suggesting a noise/annotation-quality floor around 5-6 deg.
3. Single-task could overfit: 188k training frames for a small 9-DoF regression head
   on frozen+ features, with no task diversity to regularize.
4. Body pose (keypoint Wing loss) contributes zero real signal — the pose head's
   shared log_var is partly wasted on a null task.
5. The HP_PREC_CAP artificially limits pose gradient. Without it, raw Kendall would
   give pose ~2.70 precision, dominating at 50%+ of gradient. If the cap were removed
   in a controlled single-task comparison, pose would do even better. **But this would
   be a Kendall-tuning comparison, not a single-task architecture comparison.**

### Recommendation

Run `ablation_pose_only` preset for 20-25 epochs to get ground truth. Until then,
state multi-task pose as the **first ego-pose baseline** without multi-task-benefit
language, as Opus 141 Q50 requires.

---

## Per-Recording Variance

### Forward MAE by recording (single-frame)

| Recording | Frames | Forward MAE (deg) | Up MAE (deg) |
|-----------|:-----:|:-----------------:|:------------:|
| 24_assy_2_4 | 2952 | **6.07** | 5.90 |
| 26_assy_1_5 | 4587 | **6.08** | 6.02 |
| 05_assy_0_1 | 2918 | 6.26 | 7.53 |
| 24_main_0_1 | 1371 | 6.80 | 6.09 |
| 26_main_0_1 | 1594 | 8.83 | 9.01 |
| 24_assy_0_1 | 2158 | 8.57 | 8.35 |
| 20_main_0_1 | 2066 | 8.08 | 6.33 |
| 20_assy_0_1 | 2854 | 8.52 | 7.07 |
| 05_main_0_1 | 1380 | 10.17 | 7.76 |
| 05_assy_2_2 | 2323 | 9.37 | 10.28 |
| 26_assy_0_1 | 3093 | 9.05 | 9.20 |
| 14_main_0_1 | 1685 | 10.47 | 5.71 |
| 14_main_2_2 | 1404 | 10.92 | 7.62 |
| 14_main_2_3 | 1679 | 10.97 | 6.56 |
| 20_assy_3_6 | 2967 | 11.49 | 7.99 |
| **14_assy_0_1** | 3005 | **17.05** | **12.32** |

### Forward variance: std = 2.64 deg, range 6.07-17.05 (11.0 deg spread)
### Up variance: std = 1.74 deg, range 5.71-12.32 (6.6 deg spread)

### Key observations

1. **14_assy_0_1 is a strong outlier** — 17.05 fwd, 12.32 up. The next-worst
   recording is more than 5 degrees better for forward (11.49 vs 17.05). This
   recording likely contains extreme head motion or unusual viewpoints not well
   represented in the training distribution.

2. **Forward is harder than up** — forward error is both larger in mean (9.14 vs 7.78)
   and more variable (std 2.64 vs 1.74). This is consistent with forward being a
   direction on the unit sphere while up is partially constrained by gravity.

3. **Best recordings cluster at ~6 deg** — 24_assy_2_4 (6.07 fwd), 26_assy_1_5 (6.08 fwd),
   05_assy_0_1 (6.26 fwd). This suggests a performance floor around 5-6 deg, likely
   set by HoloLens ground-truth noise and the inherent ambiguity of sparse-rig
   head pose estimation.

4. **fwd and up are strongly correlated** (Pearson r=0.67, Spearman rho=0.52) per
   `pose_error_stats.json`. Recordings with high fwd error tend to have high up error.

5. **Kalman smoothing helps most where errors are highest** — the largest improvements
   are on 05_assy_2_2 (fwd +0.38, up +0.80) and 20_assy_3_6 (fwd +0.41, up +0.42),
   both among the higher-error recordings. Low-error recordings like 24_assy_2_4 show
   negligible smoothing benefit.

---

## Implication for Paper

1. **Pose is not a confirmed multi-task win.** Without single-task ablation data,
   we cannot claim multi-task benefits for head pose. The 9.14/7.78 numbers are a
   first ego-pose baseline, not an existence proof of positive transfer.

2. **Best case for multi-task benefit** is weak: if single-task would achieve ~6 deg
   (matching the best recordings), the multi-task 9.14 deg would actually be *worse*,
   indicating negative interference. If single-task would be ~12 deg (matching the
   worst recordings), multi-task's 9.14 shows clear benefit. The truth likely lies
   between, in the 7-8 deg range — making multi-task benefit marginal at best.

3. **Per-recording median (8.94 fwd, 7.58 up) is a better headline** than mean (9.14,
   7.78) because it is robust to the 14_assy_0_1 outlier. The paper should report
   median and IQR, not mean.

4. **Paper language per Opus 141 Q50:** Present pose as a first baseline ("we
   establish ego-pose orientation accuracy at X degrees on 16 recordings of
   IndustReal") without multi-task-benefit framing. Reserve multi-task claims for
   tasks where single-task ablation exists.

---

## Files

| File | Description |
|------|-------------|
| `full_eval_ep18_v2/metrics.json` | Multi-task full eval (38k, corrected indices) |
| `pose_kalman_eval/pose_kalman_results.json` | Per-recording breakdown + Kalman results |
| `pose_error_stats.json` | Frame-level error distribution |
| `bootstrap_ci.json` | 95% CI bootstrap (1000 samples) |
| `config.py:1711-1741` | `ablation_pose_only` preset (unrun) |
| `config.py:1572-1603` | `ablation_single_task` preset (unrun) |
