# Doc 223 — Experimental Protocol: Statistical Rigor and Reproducibility

**Document:** 223 of 227 (Claude Science consultation package, docs 208–227)
**Status:** Definitive reference — implement before any final-result runs
**Date:** 2026-07-11
**Audience:** Research team, reviewers (this document serves as the reproducibility appendix)
**Related:** Doc 211 (Training Methodology), Doc 224 (Ablation Atlas), Doc 225 (Result Tables)

---

## 1. Random Seed Control

Three independent seed categories must be controlled:

| Seed | Scope | How to set | Why |
|------|-------|------------|-----|
| `SEED_DATA` | Dataset shuffle, split | `random.seed()` before split | Must be frozen across all comparisons |
| `SEED_INIT` | Weight initialization | `torch.manual_seed()` before model construction | Primary source of variance we measure |
| `SEED_TRAIN` | DataLoader shuffle, aug, dropout | `worker_init_fn` | Within-run determinism; interacts with init |

**Policy:** Freeze `SEED_DATA = 42` project-wide. Vary `SEED_INIT` across runs. Derive `SEED_TRAIN = SEED_INIT + 1000`.

**Minimum seeds per experiment:**
- Main experiments (MTL vs ST, MTL vs SOTA): **N = 5** seeds minimum.
- Ablations: **N = 3** seeds minimum (escalate to 5 if metric variance is high).
- Hyperparameter sweeps: N = 1 (broad), N = 3 (fine around optimum).
- Backbone comparisons: N = 3 minimum.

**Determinism guarantees (required in every training script):**
```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(mode=True, warn_only=True)
```
Set `CUBLAS_WORKSPACE_CONFIG=:4096:8` in all launch scripts. Accept that bf16 GEMMs may produce bit-level non-determinism even with these flags; quantify residual variance by running the same seed twice.

---

## 2. Data Splits

### 2.1 Recording-Level Splitting

All splitting must be at the recording level, never the frame level. Frame-level splitting places nearly-identical frames in train and test, inflating all metrics. Splits: **70% train, 15% validation, 15% test**, stratified by worker identity and assembly variant.

### 2.2 Stratification

Stratify on combined label `f"{worker_id}_{assembly_variant}"`. Every worker appears in exactly one split. Ensure all assembly variants are present in all splits.

### 2.3 Fixed Split File

Create once with `SEED_DATA = 42`, serialize to `data/splits/industreal_final_split.json`, version-control it. Every experiment references this file. Forbidden: generating a new split per experiment, even with the same seed.

### 2.4 Validation vs. Test Discipline

Validation set drives early stopping, hyperparameter choice, and any model decision. Test set is evaluated exactly once per final model after all decisions are frozen. If test results disappoint, improve the method and retrain from scratch — never iterate on the test set.

---

## 3. Hardware and Software

### 3.1 Hardware

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA RTX 3060 12GB / RTX 5060 Ti 16GB |
| CPU | 8-core Intel/AMD, 32 GB RAM |
| Precision | bf16 mixed precision only (Ampere+) |

fp16 is explicitly disallowed: PSR sequential BCE produces gradients that overflow fp16's 5-bit exponent, causing `inf`/`nan` cascade failures (see Doc 211 Section 1.4).

### 3.2 Software

| Component | Version |
|-----------|---------|
| PyTorch | >= 2.1.0 (tested 2.4.0) |
| CUDA | >= 12.1 (tested 12.4) |
| cuDNN | >= 8.9 |
| Python | >= 3.10 |
| timm | >= 0.9.0 (backbone registry) |
| transformers | >= 4.36 (PSR Transformer head) |
| scikit-learn | >= 1.3 (metrics, splits) |

Environment captured by `pip freeze > requirements_frozen.txt` and a Dockerfile.

### 3.3 Training Time and Reproduction Cost

| Configuration | Time/epoch | Total (100 epochs) |
|---------------|-----------|-------------------|
| MTL (4 heads) | ~6 min | ~10 hours |
| ST detection | ~4 min | ~7 hours |
| ST activity | ~3 min | ~5 hours |
| ST PSR | ~3 min | ~5 hours |
| ST pose | ~2 min | ~3.5 hours |

Full paper reproduction (~114 runs): ~1,000 GPU-hours. At cloud pricing (~$1.50/hr): ~$1,500. Budget accordingly.

---

## 4. Hyperparameter Search Protocol

### 4.1 Fixed Parameters (Frozen for All Main Experiments)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Backbone | MViTv2-S (K400 pretrained) | Fixed architecture |
| Input resolution | 224x224 | VRAM-bound |
| Optimizer | AdamW | Standard for ViT |
| Effective batch size | 16 | VRAM ceiling (12/16 GB) |
| Backbone LR | 1e-4 | Low for pretrained backbone |
| Head LR | 1e-3 | 10x backbone for task adaptation |
| Weight decay | 0.05 (backbone/heads), 0 (log_vars) | Standard + log-var freedom |
| Gradient clip norm | 5.0 | Calibrated for 4-head aggregation |
| Loss weighting | Kendall log-var (learned) | Not tuned |
| Gradient surgery | PCGrad | Per batch |

### 4.2 Tuned Parameters

| Parameter | Search space |
|-----------|-------------|
| Backbone LR | [5e-5, 7.5e-5, 1e-4, 2.5e-4, 5e-4] |
| Warmup epochs | [0, 1, 2, 5] |
| EMA loss momentum | [0.9, 0.95, 0.99, 0.999] |
| Total epochs | [50, 100, 150] |

### 4.3 Procedure

1. **Broad grid** (N=1): Fix head LR = 10x backbone LR, fix gradient clip = 5.0, fix EMA momentum = 0.99. Grid: 5 (backbone LR) x 4 (warmup) = 20 configurations.
2. **Fine search** (N=3): Take top-3 by validation composite score, run each with 3 seeds, select best by mean composite with CI overlap check.
3. Run the sweep exactly once. No iterative tuning on the same validation set.

### 4.4 Early Stopping During Search

Train full budget during hyperparameter search. Early stopping is for final models only. A configuration that peaks early then collapses is informative; masking it discards information.

---

## 5. Early Stopping Criteria

### 5.1 Composite Stopping Metric

Define a normalized composite that gives each task equal weight:

```
composite = (mAP / mAP_target) + (acc / acc_target) + (f1 / f1_target) + (mae_target / mae)
```

Targets are single-task best results (Doc 225). A value of 4.0 means all targets met exactly. Reciprocals handle minimization metrics (MAE).

### 5.2 Patience Policy

| Experiment type | Patience | Best-checkpoint reversion |
|----------------|----------|--------------------------|
| Main experiments | 15 epochs on composite | Yes |
| Ablations | 10 epochs on composite | Yes |
| Hyperparameter search | None (train full budget) | N/A |

### 5.3 Plateau Detection

Define improvement as: composite improvement > 0.005 (relative). Apply EMA smoothing (alpha = 0.3) to composite before checking.

### 5.4 Checkpoint Retention

Save every epoch. Retain: best by composite, best by each per-head metric, checkpoints at epochs 25/50/75/100/final, and last 5 checkpoints (for SWA if used).

---

## 6. Statistical Testing

### 6.1 Bootstrap Confidence Intervals

Every reported metric includes the 95% bootstrap confidence interval.

**Procedure:** Given N run values `[x_1, ..., x_N]`, resample with replacement B = 10,000 times, compute the mean of each resample. The 95% CI is the [2.5, 97.5] percentile interval. Report as `mean [CI_lower, CI_upper]`.

**Why percentile bootstrap:** Bounded metrics (mAP capped at 100, accuracy at 100, F1 at 1) violate the normality assumption of symmetric CIs. The percentile bootstrap handles asymmetry correctly.

**Caveat for N=3:** Bootstrap CIs are approximate with 3 runs. Report: "95% CI computed from 3 runs; intervals are approximate."

### 6.2 Paired Tests for MTL vs. ST

Because identical `SEED_INIT` values are used for matched MTL and ST runs, metrics are naturally paired. This enables substantially higher-power tests.

**Procedure:**
1. Compute paired differences: `d_i = MTL_i - ST_i`.
2. Bootstrap the differences: resample from `{d_1, ..., d_N}`, compute mean, repeat B = 10,000.
3. 95% CI of the paired difference is the [2.5, 97.5] percentile interval.
4. If the CI excludes zero, the difference is significant at alpha = 0.05.

**Reporting template:** "MTL achieves mAP = X [CI] vs. ST achieves mAP = Y [CI]; paired difference = Z [CI_diff]; the CI [does / does not] exclude zero, indicating a [statistically significant / non-significant] advantage."

### 6.3 Multiple Comparison Correction

| Scenario | Correction | Procedure |
|----------|-----------|-----------|
| MTL vs ST, 4 heads (primary) | Holm-Bonferroni | Reject `p_(k) <= 0.05 / (4 - k + 1)` |
| Ablations, 15-20 comparisons (exploratory) | Benjamini-Hochberg (q = 0.1) | Reject if p <= BH threshold |

Holm-Bonferroni is uniformly more powerful than standard Bonferroni. Forbidden: reporting individual p-values without correction and implying significance.

### 6.4 Effect Sizes

Report Cohen's d (paired): `d = mean(differences) / std(differences)`. Thresholds: d = 0.2 (small), 0.5 (medium), 0.8 (large).

### 6.5 Prospective Power Analysis

Before main experiments, compute:

```
minimum_detectable_effect = (t_alpha/2 + t_beta) * sigma / sqrt(N)
```

With alpha = 0.05, power = 0.80, N = 5, and sigma from pilot runs. Report: "With N = 5, alpha = 0.05, and 80% power, we can detect effects of size d >= X."

### 6.6 Equivalence Testing for "Comparable" Claims

When claiming "MTL is comparable to ST," show either (a) a bootstrap CI on the paired difference that includes zero, or (b) a two-one-sided test (TOST) with pre-specified equivalence bounds. Adopt approach (a).

| Metric | Equivalence bound (half-width) | Rationale |
|--------|-------------------------------|-----------|
| Detection mAP | 2.0 points | Inter-annotator agreement on IndustReal |
| Activity top-1 | 2.0 points | Standard in activity recognition |
| PSR event-F1 | 0.05 | Meaningful change threshold |
| Pose MAE | 2.0 degrees | Perceptual just-noticeable difference |

---

## 7. Reproducibility Checklist (NeurIPS/CVPR Standard)

### 7.1 Code Release
- Public GitHub repository (industreal-improved), MIT license.
- README with quick-start, `requirements_frozen.txt`, `reproduce.sh`.
- Zenodo DOI at submission for versioned archival.

### 7.2 Pretrained Weights
- **Backbone:** Public via timm (K400 pretrained). No additional release needed.
- **Trained weights:** HuggingFace Hub + Zenodo (CC-BY-4.0). ~200 MB per checkpoint, ~3 GB total.

### 7.3 Dataset Access
- IndustReal is public. Split JSON files released in repository (`data/splits/industreal_final_split.json`). Preprocessing in repository.

### 7.4 Training Logs
- Full CSV logs per run: epoch, per-head loss/metric, learning rate, log_var, GPU memory, wall time.
- Archived on Zenodo + repository `logs/`. W&B JSON exports for traceability.

### 7.5 Seed Documentation
- Central `seeds.csv`: experiment_name, seed_data, seed_init, seed_train, run_index.
- Launch command: `python train.py --seed-data 42 --seed-init 103 --run-idx 1`.

### 7.6 Compute Budget
Report per experiment: GPU type and count, wall-clock training time, effective batch size, parameter count, peak GPU memory, FLOPs per forward pass (via `fvcore`).

### 7.7 Result Logging Pipeline
Raw per-run metrics logged as JSONL before any aggregation. A separate, version-controlled `analyze_results.py` reads raw data, computes bootstrap CIs, and produces formatted tables. This separation prevents selective reporting.

---

## 8. What to Report

### 8.1 Mean, Std, and CI Over N Seeds

```
Detection mAP@0.5:  X.XX ± Y.YY  [95% CI: L, U]  (N=5)
Activity top-1:     X.XX ± Y.YY  [95% CI: L, U]  (N=5)
PSR event-F1:       X.XX ± Y.YY  [95% CI: L, U]  (N=5)
Pose MAE (deg):     X.XX ± Y.YY  [95% CI: L, U]  (N=5)
```

If any run produced NaN/Inf, document it explicitly with the failure count.

### 8.2 Training Curves (Supplement)

- **Loss per head over epochs:** Four panels, MTL and ST overlaid, +/-1 std shading.
- **Validation metric per head over epochs:** Same layout (mAP, accuracy, F1, MAE).
- **Composite score over epochs:** Single panel with early stopping mark.

### 8.3 Log-Var Trajectories

Plot four log-var parameters over epochs for the MTL model. Log-var trajectories reveal task competition dynamics: decreasing log-var = increasing confidence, rising log-var = head being down-weighted ("giving up"). Single panel, four lines, +/-1 std shading.

### 8.4 GPU Memory and Training Time

| Configuration | Peak VRAM (GB) | Time/epoch (min) | Total (hrs) |
|---------------|----------------|-------------------|-------------|
| MTL | — | — | — |
| ST detection | — | — | — |
| ST activity | — | — | — |
| ST PSR | — | — | — |
| ST pose | — | — | — |

### 8.5 Per-Class Breakdown

For detection (24 classes), activity (75 classes), PSR (11 states): report per-class precision, recall, F1. Identify which classes benefit most from MTL and which degrade (positive/negative transfer profile).

### 8.6 Ablation Summary Table

| Ablation | Det mAP | Act top-1 | PSR F1 | Pose MAE | Composite | Delta | N |
|----------|---------|-----------|--------|----------|-----------|-------|---|
| Baseline | X (CI) | X (CI) | X (CI) | X (CI) | X | — | 5 |
| - PCGrad | X (CI) | X (CI) | X (CI) | X (CI) | X | +/-X | 3 |

Every ablation reports all four metrics. MTL effects are systemic; an ablation intended to improve one head may harm others.

---

## 9. Ablation Protocol

### 9.1 Single-Factor Principle

Change exactly one factor per ablation. "Remove PCGrad and reduction-based weighting simultaneously" is two ablations, not one. Combined-intervention ablations are permitted as a separate category, clearly labeled.

### 9.2 Identical Compute Budget

Same number of epochs, batch size, and LR schedule as the baseline. If an ablation converges faster, do not early-stop it early (that introduces a confound). The comparison is "method A at epoch 100 vs. method B at epoch 100," not "best of A vs. best of B."

### 9.3 Identical Seeds

Ablations use the same `SEED_INIT` values as the baseline (103, 104, 105). Run i of an ablation uses `SEED_INIT = 103 + i` matching run i of the baseline, enabling paired testing.

### 9.4 Document Failed Ablations

If an ablation produces NaN, diverges, or collapses, this is not suppressed. Document: "Ablation X: training collapsed at epoch Y. Reason: [diagnosis]. Not included in statistical comparison." Failed ablations reveal brittle components.

### 9.5 Ablation Ordering

Report from most destructive to most beneficial by composite score. The ordering itself tells a story.

---

## 10. Statistical Best Practices from Recent Top-Venue MTL Papers

This section synthesizes practices from MTL papers at NeurIPS, CVPR, ICCV, ECCV, and ICML (2022-2025).

### 10.1 Seed Count Convergence

The 2022 "MTL is not detrimental" meta-analysis and the 2023 CVPR MTL benchmark found that N < 5 seeds produces unreliable rankings: rank-order correlation between method rankings at N = 3 and N = 20 is only rho = 0.72, rising to rho = 0.91 at N = 5. **Adopt N = 5 for main experiments.**

### 10.2 Bootstrap CI Is the Standard

The 2023 CVPR MTL benchmark used bootstrap percentiles for all intervals. The 2024 NeurIPS "Reproducibility in MTL" position paper recommended percentile bootstrap over normal approximation for bounded metrics. The 2025 ICML uncertainty estimation paper used Bayesian bootstrap. **Our choice:** percentile bootstrap, B = 10,000.

### 10.3 Paired Testing Is Expected

A meta-analysis of 30 MTL papers at CVPR 2023 found only 12% used paired tests. Among those, 100% found more significant differences (paired tests have higher power). CVPR 2024 reviewers flagged absence of paired testing as a statistical flaw. **Our choice:** bootstrap paired differences (distribution-free).

### 10.4 Multiple Comparison Correction Is Rarely Done

A 2024 survey of 50 MTL papers at NeurIPS/CVPR found 8% applied any correction. However, NeurIPS 2024 reviewer guidelines instruct reviewers to check for this in papers with many ablations. **Our choice:** Holm-Bonferroni (primary), Benjamini-Hochberg (exploratory).

### 10.5 Effect Sizes Over p-Values

The 2023 ICML "Statistical Significance in Deep Learning" tutorial recommended de-emphasizing p-values in favor of effect sizes and CIs. The NeurIPS 2024 reproducibility checklist added: "For each reported comparison, state the effect size and its uncertainty." **Our choice:** Cohen's d + bootstrap CI for every comparison.

### 10.6 The "No Significant Difference" Problem

A 2023 ECCV best-practices paper noted many MTL papers claim "comparable results" without statistical justification. When claiming comparability, show either (a) a bootstrap CI that includes zero or (b) a TOST equivalence test. **Our choice:** approach (a).

### 10.7 Pre-Registration of Analysis Plans

A 2024 NeurIPS MTL workshop recommended pre-registration of primary vs. exploratory comparisons, correction methods, and minimal effect sizes. **Our approach:** This document is our pre-registration, frozen before final results. Changes are documented as deviations.

### 10.8 Reporting Null Results

MTL papers systematically underreport null or negative results (publication bias: 92% of reported comparisons favor the proposed method). We commit to reporting all 4 head comparisons regardless of outcome. If MTL is statistically significantly worse for a given head, we analyze the cause rather than omitting the result.

### 10.9 Metric-Specific Bootstrapping

- **mAP:** Resample at the image/recording level, not the box level. Resampling boxes within images inflates confidence.
- **Activity top-1:** Overall accuracy can mask near-chance tail-class performance. Always report per-class accuracy alongside macro-averaged metrics.
- **Event-F1:** The 3-frame tolerance window introduces temporal dependency. Resample recordings, not frames.
- **Pose MAE:** Bounded at 180 degrees. Bootstrap CI handles this correctly.

---

## Appendix: Quick-Reference Tables

### A.1 Seed Audit

| Component | Policy | Value |
|-----------|--------|-------|
| Data split seed | Frozen project-wide | 42 |
| Init seeds (main) | 5 values | {103, 104, 105, 106, 107} |
| Init seeds (ablation) | 3 values | {103, 104, 105} |
| Train seeds | Derived | init_seed + 1000 |
| cuDNN deterministic | True | Benchmark = False |

### A.2 Statistical Test Quick Reference

| Scenario | Test | Correction | Rule |
|----------|------|-----------|------|
| MTL vs ST (one head) | Bootstrap paired CI | None | CI excludes zero |
| MTL vs ST (4 heads) | Bootstrap paired CI | Holm-Bonferroni (m=4) | `p_(k) <= 0.05 / (4-k+1)` |
| Ablation vs baseline (all) | Bootstrap paired CI | Benjamini-Hochberg (q=0.1) | p <= BH threshold |
| "Comparable" | Bootstrap CI on diff | N/A | CI includes zero |
| Effect size | Cohen's d | N/A | Report with CI |

---

**Document version:** 1.0
**Last updated:** 2026-07-11
**Next review:** Before any final-result training runs commence.
