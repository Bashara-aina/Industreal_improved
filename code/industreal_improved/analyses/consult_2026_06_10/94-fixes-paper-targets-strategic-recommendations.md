# 05 — Fixes, Paper Targets & Strategic Recommendations

## Complete Fix History (10+ critical bugs fixed)

### Fix Set 1: Pre-Training (Commits 2e69b1e, ba8c4d2, 75a2fe2)

| # | Bug | Impact | Fix |
|---|---|---|---|
| 1 | **Scheduler steps_per_epoch wrong** | OneCycleLR received ~100 total steps instead of ~80,000 — stayed in warmup forever | Set steps_per_epoch=1, called once per epoch, matching actual usage |
| 2 | **Weight decay 5e-2** | Combined with CLIP_GRAD=5.0, weight decay dominated gradients for any param with norm >4 | Reduced to 1e-3 (standard AdamW value) |
| 3 | **Gradient clip 1.0** | 80-90% of gradients were being clipped in 5-head multi-task, stalling backbone convergence | Increased to 5.0 |
| 4 | **NEG_SLOPE not applied** | Designed to prevent Kendall divergence but code used `with torch.no_grad()` → gradient never flowed | Added proper gradient path |
| 5 | **NUM_WORKERS=4** | CUDA + multiprocessing deadlocks on Python 3.13 + PyTorch 2.12 | Set to 0 (single-process, ~25% slower but stable) |
| 6 | **PSR warmup missing** | PSR had no warmup when STAGED_TRAINING=False — activated immediately at random init | Added PSR_WARMUP_EPOCHS=3 |

### Fix Set 2: GPU Stability (Commits b16cf70, e5ba3db, b1f2cc1)

| # | Bug | Impact | Fix |
|---|---|---|---|
| 7 | **CUDNN_DETERMINISTIC=True** | cuBLAS kernel timeout on full resolution | Set to False (max speed) |
| 8 | **Watchdog kills during eval** | Epoch 1 killed despite being healthy — eval took >1200s, watchdog saw stale heartbeat | IN_EVALUATION_PHASE flag skips watchdog kill; heartbeat refreshed after eval |
| 9 | **Batch size 1** | VideoMAE-era constraint, only 2GB/16GB utilized | Increased to 6 (3x throughput, 7.5GB/16GB) |

### Fix Set 3: Architecture (Commits cb18506, c27476f, 5a03bd6, f8bebc9)

| # | Bug | Impact | Fix |
|---|---|---|---|
| 10 | **PSR d_model=128 (half paper)** | PSR transformer capacity halved, expected -0.05-0.10 PSR F1 | Increased gru_hidden to 256 |
| 11 | **ViT attn_dropout=0.3 (3x paper)** | Activity over-regularized | Set to 0.1 |
| 12 | **TCN not depthwise** | 100x more params than paper spec | Conv1d with groups=embed_dim |
| 13 | **detach_reg_fpn=True** | Regression gradient severed from FPN, features non-discriminative | Set to False for all non-reinit stages |
| 14 | **EMA disabled** | Paper specifies EMA=0.999 | Enabled with decay=0.995 |
| 15 | **Verb grouping (Route A)** | Activity had 69 raw outputs with severe class imbalance | Hybrid: standalone for >=100-frame classes, verb-grouping for tail |

### Fix Set 4: Eval Stability (This Session)

| # | Bug | Impact | Fix |
|---|---|---|---|
| 16 | **SIGALRM in thread kills eval** | segment metrics runs without timeout protection in thread | Gated behind `_seg_have_alarm` |
| 17 | **No fresh heartbeat after eval** | Watchdog kills immediately when eval ends | Write heartbeat after `IN_EVALUATION_PHASE=False` |
| 18 | **Epoch 0 val wastes time** | Activity shows 1-class collapse, all metrics zero | VAL_EVERY=3 |
| 19 | **Subprocess eval deadlock** | Single GPU contention between main + subprocess | Disabled, main-thread eval with all fixes |

### Fix Set 5: Activity Collapse & Gradient Fixes (2026-06-30 through 2026-07-02)

| # | Bug | Impact | Fix |
|---|---|---|---|
| 20 | **FeatureBank gradient severed by in-place assignments** | config.py lines 883-885: in-place tensor assignments in FeatureBank and ActivityHead (model.py lines 1240-1241 and 1384) prevented gradient from flowing through proj_feat. Activity gradient was ~0.012 (30x below detection). Activity collapsed to 1/75 classes. | Root cause identified and fixed. Gradient path re-established. ACTIVITY_LR_MULTIPLIER reset to 1.0 (was 20x during debugging). Activity gradient now estimated at ~0.48 (comparable to detection). |
| 21 | **ACTIVITY_HEAD_SIMPLE bypass** | ACTIVITY_HEAD_SIMPLE=True (config.py line 901) bypasses the 8.2M-param TCN+2xViT temporal stack. With non-consecutive frame batches (balanced sampler shuffles frames across videos), the temporal stack was learning noise. The simple 150K MLP provides a strong, short gradient path. | ACTIVITY_HEAD_SIMPLE=True retained. Re-enable temporal stack only when training on true sequence batches. The verb-grouping questions (69->41) are partially moot — the collapse was largely a gradient flow problem. |
| 22 | **ACTIVITY_GRAD_BLEND_RATIO raised to 1.0** | At 0.30, activity gradient was 0.012. Successive increases (0.30 -> 0.50 -> 0.70 -> 1.00) each helped. At 1.0, full gradient flows through c5_mod_blend. | ACTIVITY_GRAD_BLEND_RATIO = 1.00 (config.py line 922) |

## Paper Targets & AAIML Strategy

### Conference: AAIML (2025/2026)
- **Track:** Likely full paper
- **Differentiator:** Efficiency — 31% fewer params, single-pass inference, comparable accuracy
- **Key claim:** Multi-task POPW achieves within 5-15% of single-task SOTA at 31% fewer parameters

### Required Benchmark Comparisons

To be competitive at AAIML, we need comparisons against:

| Baseline | What to Compare | Expected Result |
|---|---|---|
| YOLOv8m (detection SOTA) | ASD mAP@0.5 | -5 to -14% (tradeoff) |
| MViTv2 (activity SOTA) | Activity Top-1 | -2 to -10% (without VideoMAE) |
| B2 Heuristic (PSR SOTA) | PSR F1@+-3 | -0.08 to -0.23 (learned vs heuristic) |
| Single-task versions of each head | Per-task accuracy | Comparable or slightly below |
| 3-model pipeline (separate det/act/psr) | Total params, FPS, accuracy | 31% fewer params, 1 pass vs 3 |

### What We Need for Paper Submission

1. **Validation metrics at convergence** (epochs 20-30) — we have zero real metrics
2. **Full detection mAP** at DET_METRICS_EVERY_N=3 cycle (epochs 3, 6, 9, 12, 15...)
3. **Comparisons against baselines** — need to run YOLOv8m, MViTv2, or cite literature values
4. **Ablation studies** — EMA on/off, Kendall vs fixed, verb-grouping vs raw 69 classes
5. **Inference benchmarks** — FPS, params, FLOPs on target hardware
6. **Qualitative results** — detection visualizations, PSR transition graphs

## CORRECTIONS AND ADDITIONS (2026-07-02)

### 1. Activity Collapse Root Cause Already Fixed

The activity collapse was diagnosed as a gradient flow problem, not a verb-grouping problem. Key facts:

- **Root cause:** in-place tensor assignments in FeatureBank (model.py lines 1240-1241) and ActivityHead (model.py line 1384) severed the gradient path through proj_feat. Activity gradient was measured at ~0.012, which is 30x below detection's gradient.
- **Fix applied:** gradient path re-established via DETACH_GRAD_ENTRIES_ONLY=True (config.py line 1051) and ACTIVITY_GRAD_BLEND_RATIO=1.00 (config.py line 922).
- **ACTIVITY_HEAD_SIMPLE=True** (config.py line 901): With non-consecutive frame batches from the balanced sampler, the 8.2M-param TCN+2xViT temporal stack was learning from noise. The simple 150K MLP gives a strong, short gradient path.
- **Implication:** The questions about whether verb-grouping (69->41) is causing confusion are PARTIALLY MOOT. The collapse was primarily a gradient flow issue. Verb-grouping may still have an effect, but the dominant factor was the severed gradient.

### 2. Asymmetric Gamma Already Active

The question "should we use asymmetric gamma?" is answered: **we already do.**

- `DET_ASYMMETRIC_GAMMA = True` (config.py line 732)
- `DET_GAMMA_POS = 0.0` (config.py line 733) — no gamma suppression for positives
- `DET_GAMMA_NEG = 1.5` (config.py line 734) — moderate negative suppression
- Implemented in losses.py lines 344-347

The real question is whether `alpha=0.25` (FocalLoss default, config.py line 667) is appropriate when `gamma_pos=0.0`. At alpha=0.25 with gamma_pos=0:
- Positive gradient weight = alpha * (1-p)^gamma_pos = 0.25 * 1.0 = 0.25
- Negative gradient weight = (1-alpha) * p^gamma_neg = 0.75 * p^1.5
- For a well-classified positive (p=0.9): positive weight = 0.25, negative weight = 0.75 * 0.9^1.5 = 0.64
- **Positive gradient is 2.6x SMALLER than negative** even for well-classified positives
- At the bias init (p≈0.033): positive = 0.25, negative = 0.75 * 0.033^1.5 = 0.0045
- The net effect is that positives get ~55x MORE gradient than negatives at init, which is desirable, but well-classified positives are still suppressed

### 3. AAIML Strategy Critique

**Ablation A (single-task vs multi-task on same backbone) is MANDATORY for acceptance.**
- Currently zero-done. Without it, we cannot attribute any degradation to multi-task interference vs. architectural limitations.
- This is the single most important ablation for the paper's core claim.

**Detection gap is larger than stated.**
- Per the competitor analysis, detection is 59% below YOLOv8m (0.344 vs 0.838 mAP@0.5), not the 5-14% range optimistically cited in the paper targets.
- The 5-14% gap assumes convergence at 0.70-0.80 mAP. Current trajectory suggests 0.30-0.45 is more realistic.
- This changes the paper narrative from "comparable accuracy with 31% fewer params" to "competitive detection with significant efficiency advantage."

**Head pose is the uncontested contribution.**
- No prior baseline exists for multi-task head pose in assembly POPW.
- The head pose results are publishable regardless of other head performance.
- This is the strongest unique selling point for the paper.

**"Efficiency as differentiator" only works with Ablation A.**
- Without a single-task baseline on the same backbone, "31% fewer params" vs what? A 3-model pipeline is the comparison, but reviewers will ask: "How much of the efficiency gain is from the shared backbone vs. the architecture itself?"
- Ablation A answers this definitively.

### 4. Contingency Plans

| Tier | Trigger | Action |
|---|---|---|
| **Tier 1** | Fix FeatureBank RC-18, det_conf RC-19, ViT attention RC-16 | Continue current architecture with targeted fixes. Expected: detection 0.35-0.50 mAP, activity 35-55% top-1, PSR 0.35-0.55 F1 |
| **Tier 2** | Tier 1 fixes insufficient by epoch 10 | Drop PSR from multi-task (publish 4-task paper: det+act+pose+head_pose). Drop "assembly state recognition" framing. Publish as "Efficient Multi-Task Assembly Monitoring" |
| **Tier 3** | Activity/detection still below publishable threshold by epoch 20 | Full two-stage training with embedding cache. Train backbone+detection+pose in stage 1. Freeze backbone, train activity+PSR with cached embeddings in stage 2. |

### 5. Missing Baselines: Assessment

| Baseline | Verdict | Action |
|---|---|---|
| **YOLOv8m** | **Acceptable via citation.** YOLOv8 is a standardized benchmark. The POPW detection task (small industrial parts) is harder than COCO but the relative comparison is valid. | Cite YOLOv8m results on POPW from literature or run a single eval. |
| **MViTv2** | **Needs re-evaluation under grouped protocol.** MViTv2 was trained on raw 69-class activity. Our grouped protocol (41 classes) changes the task. Re-run MViTv2 on grouped labels or note the task difference. | Comparative eval with grouped protocol or caveat in paper. |
| **B2 Heuristic** | **Acceptable with exact protocol matching.** The B2 heuristic (state persistence + 1.5s timeout) is well-defined. Ensure evaluation uses the same +/-3 frame tolerance. | Run B2 on our test split with the same evaluation script. |

### 6. Required Ablations (for Paper)

| Ablation | Priority | Description |
|---|---|---|
| **Ablation A** | MANDATORY | Single-task vs multi-task on same ConvNeXt backbone. Train each head individually (no shared backbone training). Compare accuracy. Without this, the efficiency argument is hollow. |
| **Ablation B** | HIGH | FiLM on/off. Does head-pose-conditioned feature modulation (HeadPoseFiLM + PoseFiLM) help detection and activity? |
| **EMA on/off** | HIGH | EMA decay=0.995 vs no EMA. How much does EMA smooth the final metrics? |
| **Kendall vs Fixed** | MEDIUM | Kendall learned weights vs. fixed lambda weights (0.30/0.35/0.15/0.20). How much does Kendall help balance? |
| **Verb-grouping vs Raw** | MEDIUM | Hybrid verb-grouping (41 outputs) vs. raw 69 classes. Does grouping help or hurt? |

### 7. Epoch 3 Action Thresholds

At epoch 3 (first real eval with VAL_EVERY=3), evaluate the following:

| Metric | Healthy | Needs Intervention | Critical |
|---|---|---|---|
| Detection mAP@0.5 | >0.15 | 0.05-0.15 | <0.05 |
| Detection score_p50 | >0.10 | 0.036-0.10 | ~0.036 (bias init) |
| Activity top-1 | >0.10 | 0.05-0.10 | <0.05 (random: ~0.013) |
| Activity macro-F1 | >0.08 | 0.03-0.08 | <0.03 |
| Activity entropy | >1.5 nats | 1.0-1.5 nats | <1.0 nats (collapse) |
| PSR F1@+-3 | >0.10 | 0.02-0.10 | <0.02 |
| Head pose MAE | <15 deg | 15-25 deg | >25 deg |
| Combined metric | >0.12 | 0.05-0.12 | <0.05 |

**If detection is healthy (>0.15 mAP):** Continue RF4 schedule. The detection gap (59% below YOLOv8m) will close as training progresses.

**If activity is critical (<1.0 nats entropy):** Revert verb-grouping, check gradient flow with liveness probe, consider increasing ACTIVITY_LOSS_WEIGHT from 0.8 to 2.0.

**If PSR is critical (<0.02 F1):** Increase PSR_SENSITIVITY_WEIGHT beyond 0.50, extend PSR_WARMUP_STEPS beyond 500.

## Strategic Concerns for Opus

### High Priority
1. **Activity collapse** — 1/69 classes predicted at epoch 0 val, improving to 3/69 at epoch 2. Is verb-grouping (41 outputs) causing confusion? Should we revert to raw 69 classes?
2. **PSR zero on non-seq batches** — Expected behavior or sign that PSR is being suppressed by Kendall?
3. **Detection low confidence** — score_p50=0.036 means 96.4% of predictions are near the bias init. The bias init for cls is -3.4, which gives sigmoid ~0.03. If scores never separate, mAP will be near zero despite good localization. Is the classification head not learning?

### Medium Priority
4. **Backbone LR=5e-5 vs head LR=5e-4** — 10x difference. Is the backbone adapting fast enough?
5. **MIXED_PRECISION=False** — 2x speedup possible but PSR seq loss spikes corrupt GradScaler. Can we fix this?
6. **VideoMAE not used** — The paper uses VideoMAE for +5-7% activity. Is it worth adding back?

### Low Priority
7. **EMA decay 0.995 vs paper's 0.999** — Slightly faster decay means less smoothing
8. **BATCH_SIZE=6 with GRAD_ACCUM=8 -> effective 48** — Paper uses effective 32. Higher effective batch may hurt convergence
9. **Segmented (1915 segments) vs per-frame metrics** — Need to clarify in paper

## Configuration Files Changed This Session

- `config.py`: BATCH_SIZE=6, VAL_EVERY=3, DET_METRICS_EVERY_N=3, WATCHDOG_TIMEOUT=1800, ACTIVITY_GRAD_BLEND_RATIO=1.0, FEATURE_BANK_DETACH_GRAD_ENTRIES_ONLY=True, ACTIVITY_LR_MULTIPLIER=1.0
- `train.py`: Fresh heartbeat after eval ends (lines 4805-4818)
- `evaluate.py`: Segment eval guard for thread-safety (line 3601)
- `scripts/training/train_monitor.sh`: Deleted (was stale)
- System: vm.overcommit_memory set from 2 to 1

## What We Need From Opus

1. **Review activity head** — Is verb-grouping (41 outputs) fundamentally correct? Could the projection from 69->41 be losing information? Should we add an auxiliary loss on the raw 69?

2. **Diagnose detection class head** — Low confidence (score_p50=0.036) suggests the classification head may have a gradient issue. Is FocalLoss with alpha=0.25, gamma=2 appropriate for 63-class detection? Should we use asymmetric focal (gamma_pos < gamma_neg)?

3. **PSR architecture review** — With d_model=256 and 8-frame sequences, can the PSR transformer learn meaningful temporal dependencies? Should we increase sequence length?

4. **Loss balancing** — Is the combined metric formula (0.30/0.35/0.15/0.20) optimal for multi-task convergence? Should PSR get higher weight given it receives gradient only every 2nd batch?

5. **Gate RF4 strategy** — Should we consider modifying the architecture (e.g., adding auxiliary losses, changing backbone, adding VideoMAE) before completing RF4, or let it converge first?

6. **Paper positioning** — Is AAIML the right venue? Is "efficiency with comparable accuracy" a strong enough narrative?
