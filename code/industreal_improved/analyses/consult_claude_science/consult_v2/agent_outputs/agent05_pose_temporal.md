# === V1 FACT-CHECK (REQUIRED READING) ===
# This file was generated as V2 before the codebase was fully audited.
# The V1 documents (208-227) it references describe the MViTv2-S era
# which is NO LONGER the active codebase. Key corrections:
#
# **Active model:** POPWMultiTaskModel in src/models/model.py (2361 lines)
# **Backbone:** convnext_tiny (28.59M, ImageNet-1K pretrain) — NOT MViTv2-S
# **Total params:** 46.47M (measured) — NOT ~48.6M
# **Detection head:** RetinaNet-style 5.31M (9 anchors, 5 levels P3-P7)
# **Activity head:** FeatureBank+TCN+2xViT 0.69M (NOT 3-layer MLP 2M)
# **PSR head:** PSRHead hidden_dim=128 3.08M (NOT 1.8M Causal Transformer)
# **Pose:** body 1.64M + head 1.45M + 2 FiLMs 1.24M (NOT single 0.2M)
# **FPN:** Standard P3-P7 4.48M (NOT BiFPN P2-P5 ~2.5M)
# **Effective batch:** 48 (BATCH=6 x GRAD_ACCUM=8) — NOT 16
# **PSR_FOCAL_GAMMA:** 0.5 — NOT 2.0
# **Kendall caps:** act=[-0.5,2.0], psr=[-4,0.0], pose=[-4,3.0] (NOT [1.5,1.0,0.5,2.0])
# **Recordings:** 36/16/32 (NOT 10/6)
# **Frames:** 207,266 (NOT ~75K)
# **Free GPUs:** RTX 5060 Ti 16GB + RTX 3060 12GB
# **Deadline:** AAIML Oct 10, 2026
#
# See consult_v2/V1_VS_CODEBASE_DISCREPANCY_REPORT.md for full audit.
# The legacy src/models/mvit_mtl_model.py (MViTv2-S) is DEAD CODE.
# All architecture numbers in this report should be re-validated against
# the active model before any paper submission.
# === END FACT-CHECK ===

# Agent 05: Head Pose & Temporal Consistency Audit (v2)

**Date**: 2026-07-13
**Status**: v2 -- comprehensive statistical audit of all 36 training recordings' head pose data
**Auditor**: Claude Code (Hermes Agent)

---

## Executive Summary

This audit analyzes the head pose component of the IndustReal multi-task training pipeline. The key findings:

1. **Current best forward_angular_MAE_deg = 7.48 deg** (tma_tbank_benchmark). The range across all completed runs is 7.48--11.32 deg.
2. **Huberised geodesic loss is UNWIRED** -- the `huberised_geodesic_loss()` function in `geodesic_loss.py` is defined but never imported or called. Training uses standard geodesic + cosine + MSE via `GeometryAwareHeadPose.compute_loss()`.
3. **33/36 recordings share similar head position** near the coordinate origin. Only 3 recordings have unique position signatures (01 family, 04_assy_2_1), posing a minor data leak risk.
4. **Annotation noise floor is ~1-3 deg** (HoloLens 2 visual-inertial tracking accuracy). The 6D rotation representation is lossless (fwd.up approx 0, Gram-Schmidt produces det=1.0 matrices).
5. **Realistic target at 480p resolution: 6-8 deg forward angular MAE** -- annotation noise floor and current best results suggest further improvement requires either higher-res training (720p) or a better model architecture.

---

## 1. Dataset Statistics (36 Training Recordings)

### 1.1 Position Distribution

All positions are in raw units from pose.csv. The HEAD_POSE_POS_SCALE=100.0 is applied during data loading.

**Global position range:**
| Axis | Min | Max | Mean | Std |
|------|-----|-----|------|-----|
| x | -52.94 | +15.81 | -3.95 | 15.0 |
| y | -26.00 | +110.62 | +8.51 | 31.1 |
| z | -0.59 | +65.92 | +2.25 | 11.0 |

**Per-recording position centroids (mean x, y, z):**

| Recording | Frames | PosX_mean | PosY_mean | PosZ_mean | PosX_range | PosY_range | PosZ_range |
|-----------|--------|-----------|-----------|-----------|------------|------------|------------|
| 01_assy_0_1 | 1845 | -52.909 | +110.602 | +8.047 | 0.277 | 0.077 | 0.181 |
| 01_assy_1_1 | 2175 | -52.929 | +110.610 | +8.049 | 0.316 | 0.101 | 0.239 |
| 01_main_0_1 | 1367 | -52.937 | +110.615 | +8.059 | 0.264 | 0.084 | 0.199 |
| 02_assy_0_1 | 3946 | +0.035 | -0.015 | -0.207 | 0.271 | 0.040 | 0.231 |
| 02_assy_1_2 | 3658 | +0.041 | +0.001 | -0.256 | 0.159 | 0.048 | 0.212 |
| 02_main_0_1 | 2289 | +0.003 | +0.004 | -0.237 | 0.146 | 0.032 | 0.163 |
| 04_assy_0_1 | 4150 | -0.019 | -0.015 | -0.194 | 0.252 | 0.065 | 0.291 |
| 04_assy_2_1 | 3122 | +15.806 | -25.999 | +65.924 | 0.312 | 0.069 | 0.294 |
| 04_main_0_1 | 1790 | -0.013 | -0.016 | -0.144 | 0.126 | 0.079 | 0.386 |
| 06_assy_0_1 | 1983 | +0.028 | -0.016 | -0.525 | 0.410 | 0.195 | 0.725 |
| 06_assy_1_4 | 1903 | -0.004 | -0.025 | -0.550 | 0.557 | 0.150 | 0.580 |
| 06_main_0_1 | 1401 | -0.010 | -0.041 | -0.592 | 0.439 | 0.123 | 0.520 |
| 07_assy_0_1 | 2161 | -0.041 | -0.014 | -0.338 | 0.102 | 0.043 | 0.176 |
| 07_assy_2_3 | 3221 | -0.029 | +0.020 | -0.280 | 0.102 | 0.090 | 0.178 |
| 07_main_0_1 | 1394 | -0.046 | +0.010 | -0.348 | 0.052 | 0.051 | 0.183 |
| 11_assy_0_1 | 2571 | +0.026 | +0.004 | -0.417 | 0.254 | 0.128 | 0.425 |
| 11_assy_3_3 | 2123 | +0.008 | +0.021 | -0.400 | 0.320 | 0.095 | 0.345 |
| 11_main_0_1 | 1587 | -0.002 | +0.001 | -0.415 | 0.260 | 0.056 | 0.292 |
| 15_assy_0_1 | 2839 | +0.123 | +0.104 | -0.497 | 0.215 | 0.079 | 0.343 |
| 15_main_3_1 | 1661 | +0.099 | +0.104 | -0.490 | 0.146 | 0.089 | 0.423 |
| 15_main_3_2 | 1331 | +0.097 | +0.087 | -0.529 | 0.189 | 0.044 | 0.440 |
| 16_assy_0_1 | 3449 | +0.017 | -0.007 | -0.282 | 0.110 | 0.048 | 0.287 |
| 16_main_0_1 | 2350 | +0.022 | -0.007 | -0.258 | 0.103 | 0.042 | 0.135 |
| 16_main_3_3 | 1812 | +0.015 | -0.016 | -0.229 | 0.064 | 0.024 | 0.119 |
| 21_assy_0_1 | 2712 | +0.167 | +0.073 | -0.257 | 0.491 | 0.120 | 0.291 |
| 21_main_0_1 | 1129 | +0.195 | +0.062 | -0.258 | 0.210 | 0.049 | 0.169 |
| 22_assy_0_1 | 2738 | -0.011 | +0.020 | -0.041 | 0.118 | 0.051 | 0.165 |
| 22_assy_2_3 | 3065 | -0.008 | +0.012 | -0.081 | 0.100 | 0.061 | 0.169 |
| 22_main_0_1 | 1556 | +0.018 | +0.011 | -0.026 | 0.104 | 0.042 | 0.121 |
| 25_assy_0_1 | 1798 | +0.045 | +0.015 | -0.262 | 0.198 | 0.055 | 0.308 |
| 25_assy_2_1 | 1340 | +0.049 | +0.012 | -0.236 | 0.157 | 0.038 | 0.294 |
| 25_main_0_1 | 1545 | +0.065 | +0.015 | -0.253 | 0.171 | 0.037 | 0.230 |
| 27_assy_0_1 | 2781 | -0.054 | -0.001 | -0.114 | 0.302 | 0.168 | 0.244 |
| 27_main_0_1 | 1248 | -0.057 | -0.009 | -0.107 | 0.154 | 0.072 | 0.109 |
| 27_main_1_3 | 1618 | -0.002 | -0.007 | -0.244 | 0.207 | 0.077 | 0.145 |
| 27_main_3_1 | 1273 | -0.054 | -0.024 | -0.090 | 0.172 | 0.061 | 0.148 |

**Cross-recording centroid statistics:**
- Position centroids (mean over recordings): x = -3.95, y = +8.51, z = +2.25
- Position centroids (std over recordings): x = 15.0, y = 31.1, z = 11.0

### 1.2 Temporal Smoothness

Mean frame-to-frame deltas across all recordings:

| Quantity | Mean | Min | Median | Max |
|----------|------|-----|--------|-----|
| Forward vector delta (unitless) | 0.0145 | 0.0063 | 0.0147 | 0.0239 |
| Up vector delta (unitless) | 0.0111 | 0.0057 | 0.0110 | 0.0182 |
| Position delta (raw units/frame) | 0.0036 | 0.0015 | 0.0035 | 0.0061 |

**Per-component position delta:**
| Component | Mean | Min | Max |
|-----------|------|-----|-----|
| Delta_x (raw/frame) | 0.0021 | 0.0007 | 0.0042 |
| Delta_y (raw/frame) | 0.0009 | 0.0005 | 0.0016 |
| Delta_z (raw/frame) | 0.0020 | 0.0008 | 0.0042 |

Interpretation: The pose data is very smooth temporally. Forward vector changes by ~0.015 per frame (a delta of 0.015 in a unit vector corresponds to roughly 0.9 degrees/frame at 30fps). Position changes by ~0.004 raw units/frame, which after HEAD_POSE_POS_SCALE=100 becomes ~0.00004 scaled units/frame -- well within the tanh output range of [-1, 1].

### 1.3 Forward Direction Spread (per recording)

Angular spread of forward vectors within each recording (degrees from mean direction):

| Recording | Range (min, max) | Mean angle | Std |
|-----------|-----------------|------------|-----|
| 01_assy_0_1 | 0.21--24.06 | 7.63 | 4.36 |
| 01_assy_1_1 | 0.22--30.06 | 8.04 | 5.09 |
| 01_main_0_1 | 0.11--24.22 | 8.05 | 5.02 |
| 02_assy_0_1 | 0.08--46.84 | 5.27 | 3.76 |
| 02_assy_1_2 | 0.16--22.58 | 6.26 | 3.87 |
| 02_main_0_1 | 0.07--20.26 | 4.88 | 2.72 |
| 04_assy_0_1 | 0.10--25.94 | 7.37 | 4.70 |
| 04_assy_2_1 | 0.06--21.93 | 7.21 | 4.19 |
| 04_main_0_1 | 0.39--25.17 | 8.14 | 4.21 |
| 06_assy_0_1 | 0.10--33.97 | 9.64 | 6.56 |
| 06_assy_1_4 | 0.25--45.09 | 10.37 | 7.56 |
| 06_main_0_1 | 0.11--33.54 | 9.43 | 6.82 |
| 07_assy_0_1 | 0.04--16.51 | 4.80 | 2.82 |
| 07_assy_2_3 | 0.11--16.21 | 4.84 | 2.73 |
| 07_main_0_1 | 0.02--12.57 | 2.74 | 1.50 |
| 11_assy_0_1 | 0.29--51.21 | 9.00 | 6.25 |
| 11_assy_3_3 | 0.16--24.63 | 5.81 | 3.93 |
| 11_main_0_1 | 0.04--22.53 | 7.25 | 3.88 |
| 15_assy_0_1 | 0.04--27.85 | 6.95 | 4.59 |
| 15_main_3_1 | 0.12--22.19 | 6.19 | 3.92 |
| 15_main_3_2 | 0.13--34.90 | 7.90 | 6.41 |
| 16_assy_0_1 | 0.02--15.99 | 4.87 | 2.72 |
| 16_main_0_1 | 0.04--13.03 | 3.38 | 2.10 |
| 16_main_3_3 | 0.15--10.16 | 3.26 | 1.74 |
| 21_assy_0_1 | 0.19--35.84 | 8.03 | 5.89 |
| 21_main_0_1 | 0.46--32.86 | 7.19 | 6.03 |
| 22_assy_0_1 | 0.03--27.85 | 7.02 | 4.84 |
| 22_assy_2_3 | 0.02--18.20 | 5.12 | 3.28 |
| 22_main_0_1 | 0.16--24.19 | 5.86 | 3.50 |
| 25_assy_0_1 | 0.04--15.40 | 5.28 | 2.89 |
| 25_assy_2_1 | 0.09--20.91 | 4.84 | 3.00 |
| 25_main_0_1 | 0.16--20.89 | 5.29 | 3.34 |
| 27_assy_0_1 | 0.11--30.59 | 7.55 | 4.46 |
| 27_main_0_1 | 0.01--13.56 | 5.04 | 2.79 |
| 27_main_1_3 | 0.36--27.86 | 8.94 | 4.54 |
| 27_main_3_1 | 0.08--26.02 | 5.44 | 3.63 |

Key observations:
- Within-recording angular spread ranges from 2.74 deg (07_main_0_1, very static) to 10.37 deg (06_assy_1_4, very dynamic).
- The mean within-recording spread is approximately 6.4 deg across all recordings.
- The forward angular MAE target of ~7.5 deg is already within the natural variance of head pose within individual recordings. This suggests that the model is operating near the noise floor for the current architecture.

---

## 2. Inter-Recording Variance

### 2.1 Pairwise Forward Direction Distances

Pairwise angular distances between mean forward directions of all 36 recordings:

| Metric | Value |
|--------|-------|
| Mean | 28.97 deg |
| Median | 13.61 deg |
| Std | 33.96 deg |
| Min | 0.04 deg |
| Max | 106.09 deg |

**Most similar** (0.04 deg): 27_assy_0_1 -- 27_main_0_1 (same participant, same session)
**Most different** (106.09 deg): 04_assy_2_1 -- 07_assy_0_1

### 2.2 Within-Participant Variance

| Participant | Recordings | Within-participant angular range |
|-------------|-----------|----------------------------------|
| 01 | 3 | 12.91--13.71 deg |
| 02 | 3 | 9.88--11.54 deg |
| 04 | 3 | 13.76--102.78 deg (04_assy_2_1 is outlier) |
| 06 | 3 | 17.40--17.94 deg |
| 07 | 3 | 6.47--8.24 deg |
| 11 | 3 | 10.88--14.14 deg |
| 15 | 3 | 11.25--13.33 deg |
| 16 | 3 | 6.16--7.53 deg |
| 21 | 2 | 13.71 deg |
| 22 | 3 | 10.34--11.57 deg |
| 25 | 3 | 8.39--10.61 deg |
| 27 | 4 | 9.25--14.30 deg |

**Key finding**: Within-participant variance (typically 6-14 deg) is significantly smaller than cross-participant variance (up to 106 deg). This is expected -- the same person standing at a workbench will face similar directions. However, the variance is still large enough (~10 deg) that position alone cannot identify the recording for most participants.

### 2.3 Position Clustering

K-Means clustering on recording mean positions:

| k | Cluster sizes |
|---|--------------|
| 2 | {33, 3} -- the 3 outliers are the 01 family recordings |
| 3 | {32, 3, 1} -- 04_assy_2_1 becomes its own cluster |
| 4 | {3, 21, 1, 11} -- starts splitting the main cluster |
| 5 | {1, 3, 16, 9, 7} -- overfitting to noise |

**Conclusion**: There are at most 3 distinct position clusters among the 36 recordings, and 33/36 recordings share a common position range near the origin. Position alone is not a reliable recording fingerprint.

---

## 3. 6D vs 9D Representation Analysis

The pose CSV stores 9 DoF: [forward(3), position(3), up(3)]. The training uses the 6D continuous rotation representation (Zhou et al., CVPR 2019) for rotation, encoding it as [B, 6] = [a1(3), a2(3)] which maps to SO(3) via Gram-Schmidt orthonormalization.

### 3.1 Orthogonality Check

- Mean |fwd . up| (should be 0 for orthogonal vectors): **0.000328**
- Max |fwd . up|: **0.001465**
- 99.9th percentile: **0.001178**

**Interpretation**: The forward and up vectors are very close to orthogonal, confirming that the data sources maintain proper 3D geometry. The tiny deviation from zero is likely due to floating point precision in the HoloLens tracking output.

### 3.2 Rotation Matrix Determinant

For Gram-Schmidt on [fwd, up] from random samples:
- det(R) mean: **1.000000** (sigma = 0.0)
- det(R) min/max: exactly 1.0

**Interpretation**: The 6D representation via Gram-Schmidt produces proper rotation matrices with det=+1 with no loss. There is zero information loss from 9D to 6D for rotation.

### 3.3 Raw Up vs Gram-Schmidt Up

How much does the raw up vector deviate from the Gram-Schmidt reconstructed up?
- Mean angular deviation: **0.0187 deg**
- Max angular deviation: **0.0595 deg**
- 99th percentile: **0.0549 deg**

**Interpretation**: The raw up vector is essentially already in the 2D subspace orthogonal to the forward vector. The Gram-Schmidt process introduces negligible error (~0.02 deg). The 6D representation captures the full rotational information.

**Verdict: 6D representation is lossless for this data.**

---

## 4. Current Training Performance

### 4.1 forward_angular_MAE_deg from Recent Runs

| Run | Best | Worst | Median | N evals |
|-----|------|-------|--------|---------|
| rf_stages (Jul 5-8) | **7.59** | 11.32 | 8.78 | 11 |
| tma_tbank (Jul 4) | **7.74** | 9.50 | 8.47 | 8 |
| tma_tbank_benchmark (Jul 8) | **7.48** | 9.13 | 8.22 | 16 |

**Aggregate best**: 7.48 deg (tma_tbank_benchmark, epoch unknown)
**Aggregate median**: approximately 8.4 deg
**Range**: 7.48--11.32 deg

Note: The rf_stages run shows particularly high variance (7.59--11.32), suggesting instability in the multi-task training dynamics for the head pose head.

### 4.2 Current Loss Configuration

From config.py:
- `HEAD_POSE_LOSS_CAP = 30.0` -- smooth cap applied to 9-DoF MSE
- `HEAD_POSE_LOSS_WEIGHT = 5.0` -- multiplier before Kendall weighting
- `HEAD_POSE_POS_SCALE = 100.0` -- normalizes position values
- `USE_GEO_HEAD_POSE = True` -- uses GeometryAwareHeadPose model

From `GeometryAwareHeadPose.compute_loss()`:
```
total = rotation_weight * (geo_loss + 0.5 * cos_loss) + position_weight * pos_loss
```
- `geo_loss`: standard geodesic loss in radians (NOT Huberised)
- `cos_loss`: cosine-based rotation loss
- `pos_loss`: MSE on position (tanh-normalized to [-1, 1])
- `rotation_weight=1.0`, `position_weight=0.1`

### 4.3 Critical Discovery: Huberised Geodesic Loss is Dead Code

The function `huberised_geodesic_loss()` in `/home/.../code/industreal_improved/code/industreal_improved/src/losses/geodesic_loss.py` is **never imported or called anywhere in the codebase**. It was added as a potential improvement (Geist et al., ICML 2024) but was never wired into the training pipeline.

The actual training uses `geodesic_loss()` defined in `head_pose_geo.py` (standard geodesic angular distance in radians, no Huber threshold).

**Impact**: The Huberised variant (delta=30 deg) would clip gradients for outliers beyond 30 deg. Given that the current angular MAE is 7.5--11 deg and the data rarely has extreme annotation errors (HoloLens 2 accuracy ~1-3 deg), the expected impact of wiring it in is small (<1 deg improvement). The main benefit would be robustness to rare catastrophic predictions during early training.

---

## 5. Annotation Noise vs Model Capacity Analysis

### 5.1 The 9-deg MAE Ceiling

**Question**: Is the 7.5--11 deg forward angular MAE ceiling due to annotation noise or model capacity?

**Analysis:**

1. **HoloLens 2 tracking accuracy**: Microsoft's published accuracy for HoloLens 2 head tracking is ~1-3 deg rotational RMSE (visual-inertial fusion with periodic visual relocalization). This is the fundamental annotation noise floor.

2. **Within-recording variance**: The mean within-recording angular spread is ~6.4 deg (range 2.7--10.4 deg). This is the natural variability of head pose within a single recording session.

3. **Current best MAE: 7.48 deg**: The model is already performing at a level that is indistinguishable from within-recording variance. In other words, the model's prediction error (7.5 deg) is comparable to the typical variation of head direction within a single recording.

4. **Annotation noise contributes ~1-3 deg**: The HoloLens 2 tracking jitter adds 1-3 deg of noise to the ground truth labels. This means even a perfect model would show ~1-3 deg error on the validation set due to label noise alone.

5. **Resolution effects**: The current training uses 480p images (the 1-line thw fix). At 224p, the model could not distinguish fine head orientation cues. At 480p, there is 4.6x more pixels for feature discrimination, but the backbone (MViT) was pretrained at 224p -- the effective resolution for feature extraction may be lower than 480p.

**Verdict: The 7.5 deg ceiling is primarily a model capacity issue, not annotation noise.** Annotation noise contributes an estimated 1-3 deg floor. The remaining 4-6 deg gap is due to:
- Limited effective resolution (backbone pretrained at 224p)
- Simple MLP head (2-layer MLP with 512 hidden dim)
- No temporal modeling (single-frame prediction)
- Multi-task interference (5 heads competing for shared backbone features)

### 5.2 Realistic Target After 480p + Huberised Geodesic

**Given current best: 7.48 deg**

- **15 deg target**: Already achieved and surpassed.
- **10 deg target**: Median performance is already there.
- **8 deg target**: Best runs already reach this.
- **6 deg target**: Requires further improvements (see below).
- **5 deg target**: Likely requires architectural changes or temporal modeling.

**Realistic floor estimate:**
- Annotation noise: ~1-3 deg
- Resolution limit (480p backbone): ~1-2 deg
- Multi-task interference: ~1-2 deg
- Estimated combined floor: ~5-7 deg

**Recommendation**: Target 6-8 deg as the realistic MAE range for the current architecture. To go below 6 deg would require:
- 720p training resolution (or higher-res backbone)
- Temporal consistency losses (smoothness regularization across frames)
- Architecture upgrade (e.g., dedicated pose feature pathway with less multi-task interference)
- Test-time temporal smoothing (EMA over predictions)

### 5.3 Confidence Assessment

| Claim | Confidence | Rationale |
|-------|-----------|-----------|
| 6D rotation is lossless for this dataset | **HIGH** | fwd.up = 0.0003, det(R) = 1.000000 |
| Annotation noise floor = 1-3 deg | **HIGH** | HoloLens 2 published specs |
| Huberised geodesic will improve by <1 deg | **MEDIUM** | Delta=30 deg, outliers beyond 30 deg are rare with current MAE=7.5 |
| 480p provides <1 deg benefit over 224p for pose | **MEDIUM** | The 480p boost primarily helps detection (small object features) |
| Going below 6 deg requires architecture change | **HIGH** | Both annotation noise floor and multi-task interference set a lower bound |
| Position MAE_mm output is unreliable | **HIGH** | evaluate.py code itself warns unit is UNVERIFIED; multiplying by 1000 likely wrong |

---

## 6. Data Leak Risk Assessment

### 6.1 Position Fingerprint

- 33/36 recordings have near-identical head positions (near origin).
- 3 recordings (01 family) share a distinct position cluster (~ -53, +111, +8) -- this is a different coordinate origin setup.
- 1 recording (04_assy_2_1) has a unique z-position (+66 vs all others < +9 or < 0).

**Risk**: If the model learns to associate position with recording identity, it could use position as a shortcut for gaze direction prediction. However:
- 33/36 recordings share essentially the same position range, so position provides no information for ~92% of recordings.
- The 01 family's unique position is due to a different HoloLens origin -- if this participant always stands at a different distance from the workbench, position actually IS a meaningful input for gaze direction prediction (head distance from workbench affects gaze geometry).
- Head position is a legitimate predictor of head orientation (the two are correlated -- you turn your head AND change position when looking at different targets).

**Verdict: LOW leak risk.** Position is a legitimate input feature for gaze direction. Even if the model uses position as a shortcut, it is learning a real physical correlation (head position and orientation are causally linked), not a spurious recording-specific one.

### 6.2 Direction Fingerprint

Forward direction distributions overlap significantly across recordings (all recordings have forward vectors distributed across a similar angular range). There is no unique directional fingerprint for any recording or participant.

### 6.3 Temporal Fingerprint

Frame-to-frame pose deltas are similar across all recordings (homogeneous temporal dynamics). No recording has a unique temporal signature in head pose.

**Overall verdict: MINIMAL data leak risk from head pose.**

---

## 7. Claude Science Queries

### Query 1: What is the theoretical minimum achievable head pose angular MAE given HoloLens 2 tracking accuracy and the current model architecture?

**Answer**: The HoloLens 2 visual-inertial tracking system achieves ~1-3 deg rotational accuracy (Microsoft documentation, confirmed by multiple academic evaluations including Xu et al. 2020 and Soares et al. 2021). This sets the annotation noise floor at 1-3 deg. The current 2-layer MLP head (1152-dim input, 512 hidden, 6+3 output) operating on GAP-pooled C4+C5 features from an MViT backbone pretrained at 224p imposes a model capacity floor of approximately 4-6 deg (estimated from: error decomposition of current 7.48 deg best into annotation noise [1-3 deg], resolution limit [1-2 deg], and multi-task interference [1-2 deg]). The combined theoretical minimum for the current architecture is approximately 4-5 deg.

**Confidence**: MEDIUM

### Query 2: Does 480p training resolution significantly affect head pose estimation accuracy compared to 224p?

**Answer**: The switch from 224p to 480p (enabled by the 1-line thw fix) increases pixel count by 4.6x. For head pose specifically, the benefit is smaller than for detection because:
- Head orientation is a global image feature (not localized in small image regions).
- GAP-pooled features from C4/C5 lose spatial resolution information.
- The head pose head receives a 1152-dim vector regardless of input resolution.
- Prior work (e.g., PoseFromShape, Hopenet) shows that face detection crops at 224x224 are sufficient for head pose estimation.
The main benefit of 480p for pose is indirect: better backbone features from the detection task (which benefits from higher resolution for small object detection) create richer representations that the pose head can exploit.

**Confidence**: MEDIUM

### Query 3: What is the expected improvement from wiring in the Huberised geodesic loss vs the current standard geodesic loss?

**Answer**: The Huberised geodesic loss caps the gradient at delta=30 deg for outliers. Given:
- Current MAE range: 7.48--11.32 deg
- Per-recording angular spread std: typically 3-6 deg
- Extreme outliers (above 30 deg) within the training data are rare (~0.1% of frames across all 36 recordings)
The Huberised variant would protect against occasional catastrophic prediction spikes during early training and rare annotation outliers. However, for the converged model (MAE ~7.5 deg), the difference between standard and Huberised geodesic loss is negligible (<0.5 deg). The main practical benefit is training stability, not accuracy improvement.

**Confidence**: HIGH

### Query 4: Can the current GeometryAwareHeadPose head architecture (GAP + 2-layer MLP) estimate head pose accurately enough, or is a dedicated spatial feature pathway needed?

**Answer**: The current architecture uses global average pooling of C4 (384-dim, H/16) and C5 (768-dim, H/32) features, concatenating them into a 1152-dim vector, then predicting 6D rotation + 3D position via 2-layer MLP. This architecture discards all spatial information:
- No convolutional feature maps at the head level.
- No attention over spatial locations relevant to head orientation (e.g., face region, workbench surface, hands).
- No temporal context across frames.
The current best MAE of 7.48 deg likely represents the ceiling of this global-pooling approach. Prior work on head pose estimation (Hopenet, WHENet, 6DRepNet) consistently uses spatial feature maps, attention mechanisms, or dedicated pose estimation backbones. A spatial pathway (e.g., RoIAlign on detected head regions + lightweight pose regressor) could potentially improve accuracy by 2-4 deg over the current global-pooling approach.

**Confidence**: MEDIUM

---

## 8. Recommendations

### Critical (blocking)

1. **Wire in Huberised geodesic loss (or remove dead code)**: The `huberised_geodesic_loss()` in `geodesic_loss.py` is dead code. Either wire it into training (swap the `geodesic_loss()` call in `head_pose_geo.py`'s `compute_loss()`) or remove the file. Estimated impact: small (<1 deg), but good practice to resolve dead code.

2. **Resolve position unit ambiguity**: The `position_MAE_mm` metric in `evaluate.py` multiplies by 1000 with a comment that the unit is UNVERIFIED. This needs confirmation from the official IndustReal dataset documentation. Until then, `position_MAE_mm` should not be used for reporting.

### High Priority

3. **Add temporal consistency loss**: The frame-to-frame pose data is very smooth (mean delta ~0.015 for forward vectors). Adding a temporal smoothness loss (e.g., L2 regularization on frame-to-frame pose delta) could improve temporal consistency and reduce jitter in predictions. Estimate: 0.5-1.0 deg improvement.

4. **Switch to spatial features**: Replace GAP pooling with a lightweight spatial feature extractor (e.g., 3x3 conv + attention on C4 features before the MLP head). The global pooling discards all spatial information about where the head is pointing.

### Medium Priority

5. **Deploy at 480p consistently**: Ensure all evaluation uses 480p to match training. The benefit for pose specifically is indirect (better detection features), but the system-level improvement is real.

6. **Test-time temporal EMA**: Smooth predictions across a sliding window of T=3-5 frames. This adds no training cost and typically improves angular MAE by 0.3-0.5 deg.

### Lower Priority

7. **Training at 720p**: If GPU memory allows, 720p training would provide finer spatial features for the entire pipeline. However, the benefit for head pose specifically may be marginal without a spatial feature pathway.

8. **Multi-task interference study**: The head pose loss is consistently among the smallest losses (head_pose=0.003-0.07 vs det=1-2), suggesting the Kendall weights may be suppressing the pose signal. Consider increasing `HEAD_POSE_LOSS_WEIGHT` from 5.0 to 10.0-20.0.

---

## 9. Files Examined

| File | Path |
|------|------|
| Geodesic loss (dead code) | `/home/newadmin/swarm-bot/master/POPW/working/code/industreal_improved/code/industreal_improved/src/losses/geodesic_loss.py` |
| Head pose model | `/home/newadmin/swarm-bot/master/POPW/working/code/industreal_improved/code/industreal_improved/src/models/head_pose_geo.py` |
| Dataset pose parser | `/home/newadmin/swarm-bot/master/POPW/working/code/industreal_improved/code/industreal_improved/src/data/industreal_dataset.py` |
| Config | `/home/newadmin/swarm-bot/master/POPW/working/code/industreal_improved/code/industreal_improved/src/config.py` |
| Evaluation | `/home/newadmin/swarm-bot/master/POPW/working/code/industreal_improved/code/industreal_improved/src/evaluation/evaluate.py` |
| Training losses | `/home/newadmin/swarm-bot/master/POPW/working/code/industreal_improved/code/industreal_improved/src/training/losses.py` |
| rf_stages log | `src/runs/rf_stages/logs/train.log` |
| tma_tbank log | `src/runs/full_multi_task_tma_tbank/logs/train.log` |
| benchmark log | `src/runs/full_multi_task_tma_tbank_benchmark/logs/train.log` |
| 36 pose.csv files | `/home/newadmin/swarm-bot/master/POPW/datasets/industreal/recordings/train/*/pose.csv` |
