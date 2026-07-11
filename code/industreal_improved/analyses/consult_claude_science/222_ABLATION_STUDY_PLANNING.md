# 222 — Ablation Study Planning

**Document:** 222 of 227 (Claude Science consultation package, docs 208–227)
**Status: Draft for Claude Science review**
**Date:** 2026-07-11
**Audience:** Claude Science
**Goal:** Complete, publishable ablation study design for a winning AAIML paper.

---

## Table of Contents

1. Philosophy: Every Ablation Must Be Independently Informative
2. Ablation 1: MTL vs. ST Baselines (Primary Table)
3. Ablation 2: Optimization Ablations
4. Ablation 3: Architecture Ablations
5. Ablation 4: Training Ablations
6. Ablation 5: Loss Ablations
7. Ablation 6: Efficiency Ablations
8. Ablation 7: Data Ablations
9. Ablation 8: What Claude Science Should Find
10. Compute Budget Summary
11. Analysis Protocol

---

## 1. Philosophy: Every Ablation Must Be Independently Informative

An ablation study is not a checkbox exercise. Each row must teach the reader something they did not know before, and each row must be designed so that its outcome is not predictable from any other row. The principle: **if you already know what the row will show, do not run it — cite existing literature instead.**

For every ablation row in this document we specify:

- **Hypothesis:** The precise claim under test, stated as a falsifiable prediction.
- **What it tests:** The code/config change from the full model. Exact flags, config keys, or code changes.
- **Expected outcome:** Our best guess, with justification. "No change" is a valid finding that saves future researchers from running this ablation.
- **Compute cost:** GPU-hours relative to the full training run (50 epochs, ~8K batches/epoch, ~2 days on RTX 3060/5060 Ti). Expressed as multiple of one full run (1x = ~48 GPU-hours = ~2 days on one GPU).

The full model (our "max" configuration) is:

| Lever | Setting | CLI Flag |
|-------|---------|----------|
| Backbone | MViTv2-S, Kinetics-400 pretrained | (default) |
| FPN | BiFPN, 256ch, P2-P5 | (default) |
| Detection head | Decoupled cls/reg, TOOD-TAL, CIoU+DFL+QFL | (default) |
| Activity head | 3-layer MLP (768→2048→1024→75) | (default) |
| PSR head | Causal Transformer (d=256, 2-layer, nhead=4) | (default) |
| Pose head | MLP (768→256→6) + Gram-Schmidt | (default) |
| Multi-task weighting | Kendall uncertainty + EMA normalization + per-task caps | (default) |
| Gradient surgery | PCGrad | `--pcgrad` (default True) |
| PSR loss | Focal-BCE (gamma=2.0, alpha=0.25) + transition boost (3x) | `--psr-focal` (default True) |
| Detection aug | Flip + color + crop | `--det-aug` (default True) |
| Activity logit-adjust | Menon et al. 2020, in loss | (internal, default on) |
| Activity class weights | Inverse effective sample size | (internal, default on) |
| Activity decoupled phase | Two-phase: frozen sharing at epoch 25 | `--act-decoupled` (default off) |
| SWA | Last 5 checkpoints averaged | `--swa-checkpoints 5` |
| Warm-start | Head weights from ST best checkpoints | `--warm-start-dir` |
| Distillation | KL-div from ST teachers (alpha=0.1, T=4.0) | `--distill-teacher-dir` |
| Epochs | 50 | `--epochs 50` |
| Effective batch | 16 (4 micro-batch x 4 grad accum) | `--batch-size 4 --grad-accum-steps 4` |
| Optimizer | AdamW, 3-group LR (1e-4 backbone, 1e-3 heads, 1e-3 log-var) | (default) |
| Scheduler | CosineAnnealingLR, T_max=50 | (default) |
| Precision | bf16 mixed | (default) |
| Gradient clip | global norm 5.0 | (default) |

---

## 2. Ablation 1: MTL vs. ST Baselines (Primary Table)

This is Table 1 of the paper. It establishes the fundamental claim: does MTL help or hurt relative to four independently-trained single-task models? The table must be the most rigorously executed experiment in the paper.

### 2.1 Full MTL (all levers active)

| Row | Model | det mAP@0.5 | act top-1 | psr event-F1 | pose fwd-MAE |
|-----|-------|-------------|-----------|-------------|-------------|
| A | **Full MTL** | ? | ? | ? | ? |
| B | ST-pose (MViTv2-S) | — | — | — | ? |
| C | ST-activity (MViTv2-S) | — | ? | — | — |
| D | ST-detection (MViTv2-S) | ? | — | — | — |
| E | ST-PSR (MViTv2-S) | — | — | ? | — |
| F | MTL/ST ratio (A/B..E) | ? | ? | ? | ? |

**Hypothesis:** MTL will match or exceed ST for pose (MTL/ST >= 0.95), degrade modestly for PSR (MTL/ST >= 0.60), degrade significantly for detection (MTL/ST >= 0.50), and degrade severely for activity (MTL/ST >= 0.30). The paper's narrative depends on these numbers: a 0.95 ratio for pose is a "positive transfer" headline; a 0.30 ratio for activity is a "future work" disclosure.

**What it tests:** Same backbone (MViTv2-S), same input resolution (224px), same data splits. ST models use identical head architectures trained on single-task datasets. ST models train for 50 epochs with `--max-batches-per-epoch 8000` (matching MTL's per-epoch batch count per task). ST models use task-specific loss functions (ST-pose uses geodesic; ST-det uses CIoU+DFL+QFL; ST-activity uses CE+logit-adjust; ST-PSR uses focal-BCE+transition-boost).

**Expected outcome:** The MTL/ST ratios will follow the gradient starvation ordering: pose > PSR > detection > activity. This directly mirrors the per-head gradient norm measurements from Doc 210 (pose: 0.44, PSR: 3.18, detection: 0.48, activity: 0.010). The inverse correlation between gradient norm and MTL/ST ratio is the paper's core evidence that gradient starvation causes the MTL gap.

**Compute cost:** 5x (four ST runs + one MTL run). ST runs are ~0.5x each (single task, same epoch budget, but lower VRAM per batch). MTL is 1x. Total: ~3x.

**Critical protocol:**
- All ST runs use the same RNG seed sequence as the MTL run for the first 8K batches/epoch. This controls for data ordering noise.
- ST detection uses its own backbone forward pass without FPN overhead. This gives ST-detection a slight parameter advantage (no FPN params in backbone). This is *conservative* — it makes the MTL/ST ratio harder to beat.
- Metrics must be computed by the same eval script from the same checkpoint format. No special eval paths for ST vs MTL.
- Report 3-seed mean + std for a representative subset (pose + detection, because they bracket the expected range). This establishes that MTL/ST ratios are statistically significant.

### 2.2 Ablation: MTL with minimal levers (no warm-start, no distillation, no SWA)

**Hypothesis:** Without the three supporting levers (warm-start, distillation, SWA), MTL/ST ratios drop by 5-20% depending on the head. Distillation is the most powerful lever for activity and detection; warm-start matters most for activity; SWA is a flat 0.5-2% benefit for all heads.

**What it tests:** `train_mtl_mvit.py` with `--no-warm-start-dir --no-distill-teacher-dir --swa-checkpoints 0`. Compares to Row A to isolate the contribution of each lever.

**Expected outcome:** Activity drops most (distillation provides KL-guidance for long-tail classes). Detection drops moderately (warm-start stabilizes TAL assigner). PSR and pose are nearly flat (their losses are low enough that extra support is redundant).

**Compute cost:** 1x

---

## 3. Ablation 2: Optimization Ablations

This ablation block tests every decision in the multi-task optimization loop. These ablations directly support the paper's claimed contribution: **characterization and mitigation of Kendall collapse**.

### 3.1 Kendall capped vs. Kendall uncapped

**Hypothesis:** Uncapped Kendall (all log-var caps at 4.0) causes "Kendall collapse" — the task with the highest raw loss (activity, ~12.3) gets its precision weight driven to near-zero, effectively removing it from the multi-task objective. Capped Kendall (det<=1.5, act<=1.0, psr<=0.5, pose<=2.0) forces minimum precision floors that prevent any task from being completely starved.

**What it tests:** `--kendall-uncapped` flag (code line 1087). This sets `LV_CLAMP_MAX = {"det": 4.0, "act": 4.0, "psr": 4.0, "pose": 4.0}`. The key comparison: log-var values throughout training, per-task loss trajectories, and final metrics.

**Expected outcome:** Uncapped: activity precision weight collapses to floor (exp(-4) ≈ 0.018), activity accuracy drifts toward random (1/75 ≈ 1.3%), other tasks degrade modestly because they lose the auxiliary activity signal. Capped: activity weight floor is exp(-1) ≈ 0.37, activity stays at least 5-10x better than random.

**This is the paper's Figure 2.** A convincing figure shows log-var trajectories: uncapped curves diverge (act log_var → +4, others hover near 0), capped curves converge to stable equilibrium values.

**Compute cost:** 1x (single run with flag change)

### 3.2 Kendall + PCGrad vs. Kendall-only

**Hypothesis:** PCGrad removes conflicting gradient components between tasks, preventing one task's gradient from canceling another's. Without PCGrad, PSR and detection gradients (which often point in opposite directions — PSR wants "assembly state change" features, detection wants "stable object presence" features) interfere on the shared backbone, degrading both.

**What it tests:** `--no-pcgrad` flag (code line 1724). Compares full model (Kendall + PCGrad, all caps) vs Kendall-only (no gradient surgery). Evaluates whether PCGrad's ~2x backward-pass overhead is justified.

**Expected outcome:** Without PCGrad: detection mAP drops (gradient conflict with PSR), PSR event-F1 may improve slightly (its gradient is 7x stronger and no longer projected away from detection). Activity degrades modestly (loses the regularization benefit of conflict resolution). The net effect: one task wins, another loses — the definition of gradient conflict that PCGrad is designed to resolve.

**Key insight for paper:** This ablation tests whether gradient conflict actually exists. If detection and PSR both degrade without PCGrad, the conflict is confirmed. If one improves and the other degrades, PCGrad is doing its job. If nothing changes, PCGrad is unnecessary compute overhead — a publishable negative result.

**Compute cost:** 1x

### 3.3 Kendall vs. GradNorm vs. DWA vs. equal weights

**Hypothesis:** Kendall uncertainty weighting (learned per-task log variances) outperforms fixed-weight alternatives because it dynamically adapts to training stage. GradNorm (Chen et al. 2018) applies gradient normalization but introduces a hyperparameter (alpha) that needs tuning. DWA (Liu et al. 2019) uses temporal loss ratios but ignores gradient magnitude. Equal weights are naive and known to fail when tasks have different loss scales.

**What it tests:** Four separate runs with different multi-task weighting strategies:
- **Kendall (our method):** `--kendall-uncapped` (use learned log-var but WITH caps as our default)
- **GradNorm:** Replace Kendall weighting with GradNorm (gradient normalization that balances gradient magnitudes across tasks). Requires implementing GradNorm loss term in `train_step()`. See Chen et al. 2018.
- **DWA:** Replace Kendall weighting with Dynamic Weight Averaging (exponential decay of loss ratio over time). See Liu et al. 2019.
- **Equal weights:** Set all per-task weights to 1.0, disable log-var learning. Use `--no-pcgrad` to isolate weighting effect? No — keep PCGrad for fair comparison. Set log_vars fixed to 0.0 (weight=1.0).

**Expected outcome:** GradNorm and Kendall will be close, with GradNorm possibly better for detection (it directly balances gradient magnitudes). DWA will underperform because it only looks at loss ratios, not gradient magnitudes. Equal weights will be worst because activity loss (~12.3) dominates the unweighted sum.

**Compute cost:** 3x (GradNorm, DWA, equal weights). GradNorm requires implementation (~2 days engineering). Kendall is already run as the 1x default.

### 3.4 EMA loss normalization on/off

**Hypothesis:** EMA normalization (dividing each task's loss by its running EMA before Kendall weighting) ensures Kendall balances losses at comparable O(1) scales. Without EMA, Kendall sees raw losses at very different magnitudes (pose ~0.01, activity ~12.3), which forces extreme log-var values that saturate the caps.

**What it tests:** Disable EMA normalization in `train_step()` (set `ema_losses=None` or skip the normalization block at line 1065). Compare log-var trajectories and final metrics.

**Expected outcome:** Without EMA: log-vars drift to cap boundaries (pose hits +4, activity hits -4), making the caps load-bearing rather than safety nets. Metrics degrade 1-5% across tasks, with activity most affected (its raw loss is largest, so it gets the smallest Kendall weight).

**Compute cost:** 1x

### 3.5 Logit-adjust in loss vs. in forward

**Hypothesis:** Logit-adjustment (Menon et al. 2020) should be applied in the loss function only (additive correction before softmax during training), not in the model's forward pass (which would change inference behavior). Applying in both places double-counts the correction.

**What it tests:** Move the `logits += tau * log(freq)` from `activity_loss()` (line 384) into `ActivityHead.forward()` (line 321). Compare test-set accuracy on tail classes (samples <10 instances) and head classes (samples >100 instances).

**Expected outcome:** Double-applying logit-adjust over-corrects: tail classes get too much boost, head classes degrade. Tail accuracy may increase 1-2% but at the cost of 3-5% head accuracy, lowering overall top-1. Single application (in loss only, current design) is correct.

**Compute cost:** 1x (flag change or code revert)

---

## 4. Ablation 3: Architecture Ablations

These ablations test every architectural decision. The goal is to show that our chosen architecture is Pareto-optimal — no simpler architecture matches it, no more complex architecture justifies its extra cost.

### 4.1 Shared backbone vs. task-conditional adapters

**Hypothesis:** Task-conditional adapters (small bottleneck MLPs inserted into each transformer block, conditioned on a task embedding) improve per-task performance by allowing the backbone to specialize features for each task. However, they break the single-forward-pass efficiency claim — adapters must be applied per-task, requiring either multiple forward passes or batched task-specific processing that increases VRAM.

**What it tests:** Add LoRA-style adapters to MViTv2-S query/value projections (Hu et al. 2021), with rank r=8, conditioned by learned task embeddings (4 tasks, each 64-dim). Insert after each transformer block's self-attention. The model now has two modes: (a) shared forward (no adapters, baseline) and (b) task-conditional (each task's data goes through backbone + that task's adapters, requiring 4x forward passes).

**Expected outcome:** Task-conditional mode improves detection (+1-2 mAP) and activity (+2-3% top-1) by giving each task dedicated feature pathways. Pose and PSR show <1% change. But inference cost increases ~4x (from single-forward to 4-forward), destroying the paper's efficiency advantage. Parameter count increases by ~2M (500K per task × 4 tasks).

**Critical insight:** This ablation MUST be run to defend against the reviewer question: "Why not use task-specific adapters?" The answer: "We tried them; they add 4x inference cost for <2% improvement on two of four tasks. Not worth it."

**Compute cost:** 1x (adapter training) + 0.5x (adapter inference eval). Implementation ~3 days.

### 4.2 BiFPN vs. simple FPN vs. no FPN (direct from P3/P4/P5)

**Hypothesis:** BiFPN's weighted top-down + bottom-up fusion provides marginal benefit over a simple FPN (one top-down pass, no bottom-up) for our use case because only one head (detection) uses the FPN, and detection is already capacity-limited by gradient starvation, not feature quality.

**What it tests:** Three variants:
- **BiFPN (current):** `LightweightFPN` with top-down + bottom-up + learnable fusion weights. ~2.5M params.
- **Simple FPN:** Top-down only, no bottom-up pass. Replace `LightweightFPN` with a simple `nn.ModuleDict` of lateral 1x1 convs + 3x3 smooth convs, no weighted fusion, no bottom-up. ~1.5M params.
- **No FPN:** Each detection head level reads directly from the corresponding backbone feature (P3 from blocks[4], P4 from blocks[8], P5 from blocks[12]), projected to 256ch via 1x1 conv. No multi-scale fusion. Tri-linear interpolation for up/down within the detection head. ~0.3M params.

**Expected outcome:** BiFPN = simple FPN ≈ no FPN for detection metrics. The detection bottleneck is gradient starvation (gradient norm 0.48 vs PSR at 3.18), not feature representation quality. If this holds, the ~2.2M extra FPN params can be removed without hurting performance — a direct efficiency gain.

**Risky prediction:** If BiFPN significantly outperforms (≥1 mAP), then detection's bottleneck is actually feature quality, and we should invest in better features (not larger gradients). Either finding is publishable.

**Compute cost:** 2x (simple FPN, no FPN). Implementation ~1 day each.

### 4.3 Detection head: decoupled vs. coupled

**Hypothesis:** Decoupled heads (separate conv paths for classification and regression) outperform coupled heads (shared conv path with two output heads) for detection because classification and regression have conflicting optimal features — classification needs translation-invariant features (what object is present), regression needs translation-covariant features (where is it precisely).

**What it tests:** Replace the decoupled `cls_head` + `reg_head` in `DetectionHead` (lines 239-250) with a single conv path that branches only at the final 1x1 conv: `conv -> GN -> ReLU -> conv_cls (out=num_classes) + conv_reg (out=4*reg_max)`. Keep all else equal.

**Expected outcome:** Coupled head degrades detection 1-3 mAP. The decoupled design is standard in modern detectors (YOLOX, TOOD, RT-DETR) for good reason. A negative result (no change) would be surprising and worth reporting.

**Compute cost:** 1x

### 4.4 PSR head: causal vs. bidirectional attention

**Hypothesis:** Causal attention (masked so each frame can only attend to past frames) is necessary for the PSR task because PSR is a streaming prediction problem — the model should predict "picking" at frame t based only on frames up to t. Bidirectional attention leaks future information, making the task easier during training but impossible to deploy in a streaming setting.

**What it tests:** Remove the causal mask in `PSRHead.forward()` (line 432-435): `mask = None`. Train and evaluate on the non-causal variant. Compare event-F1 on the streaming eval split (where future frames are not available).

**Expected outcome:** Bidirectional attention improves training event-F1 by 0.05-0.10 (future context helps disambiguate transitions) but the gap narrows at eval. On the streaming eval split, bidirectional may actually perform worse because it has learned to rely on future context that is not available.

**This is a critical reviewer defense.** The question "Why causal attention for PSR?" must be answered with this ablation.

**Compute cost:** 1x

### 4.5 Activity head: MLP vs. temporal pool vs. transformer

**Hypothesis:** The current 3-layer MLP on the CLS token is not optimal for activity recognition from video — it ignores temporal structure entirely. A temporal pooling or lightweight transformer head could aggregate information across the T=16 frames for better classification.

**What it tests:** Three activity head variants:
- **MLP (current):** 768→2048→1024→75 on CLS token (frame 0, the video-level CLS). ~2M params.
- **Temporal pool:** Take CLS tokens from all T=16 frames, mean-pool across time, then 768→1024→75 MLP. ~1.3M params.
- **Lightweight transformer:** CLS tokens from T=16 frames → 2-layer Transformer (d=256, nhead=4) → mean pool → Linear(256→75). ~1.5M params.

**Expected outcome:** Temporal pooling and transformer variants improve activity by 1-3% because they can aggregate evidence across multiple frames. However, the MViTv2-S already performs spatiotemporal pooling internally, so the CLS token already contains temporal information — the marginal benefit of additional temporal aggregation is small.

**Compute cost:** 2x (temporal pool + transformer variants). Implementation ~1 day.

---

## 5. Ablation 4: Training Ablations

These ablations quantify the contribution of each "lever" in our training recipe. The goal is to show which levers are essential and which are nice-to-haves.

### 5.1 With/without detection augmentation

**Hypothesis:** Detection augmentation (flip + color jitter + random crop) adds +3-5 mAP at zero model cost. This is the single highest-ROI lever in the system.

**What it tests:** `--no-det-aug` flag (code line 1731). Disables the `DetectionAugment` transform in `train_step()` (line 963-966).

**Expected outcome:** Without det-aug, detection mAP drops from ~0.207 epoch-50 to ~0.10-0.15. This establishes that detection augmentation is mandatory, not optional. The boost magnitude directly supports the paper's argument that data diversity, not architecture complexity, is the primary lever for detection improvement.

**Compute cost:** 1x

### 5.2 With/without SWA checkpoint averaging

**Hypothesis:** SWA provides a 0.5-2% improvement across all tasks at zero training cost (it's applied post-training). The benefit is largest for detection (where weights oscillate between modes) and smallest for pose (which converges to a narrow minimum).

**What it tests:** `--swa-checkpoints 0` (code line 2361). Compare the best single checkpoint (from val eval) against the SWA-averaged checkpoint on test metrics.

**Expected outcome:** SWA provides consistent small improvements. This ablation is primarily to quantify "how much does SWA help" for the paper. If SWA helps >2% on any task, the paper should highlight it; if <0.5%, SWA is optional.

**Compute cost:** 0x (already computed during the run; just need to save both metrics).

### 5.3 With/without class-balanced activity sampling

**Hypothesis:** Class-balanced sampling (over-sampling rare activity classes during training) improves tail-class accuracy without hurting head-class accuracy. The dataset's activity labels follow a power-law distribution (head classes have >1000 samples, tail classes have 1-5 samples), making class-balanced sampling essential.

**What it tests:** Disable class-balanced sampling in the dataset (set `IndustRealMultiTaskDataset` to uniform frame sampling instead of class-balanced sampling). This is not currently a CLI flag — requires code change in the data loader or a config switch.

**Expected outcome:** Without class-balanced sampling, tail-class accuracy drops to near-zero (model never sees tail examples often enough to learn them). Overall top-1 accuracy may drop 1-3% if tail-class errors accumulate. This ablation establishes that the data loader's class-balancing is essential, not optional.

**Compute cost:** 1x

### 5.4 With/without distillation from ST teachers

**Hypothesis:** Knowledge distillation from ST teachers provides the largest single improvement for the two classification heads (detection and activity). The ST teachers provide "soft targets" that guide the MTL model toward correct predictions even when gradient starvation would otherwise prevent learning.

**What it tests:** `--no-distill-teacher-dir` (omit the argument). Compare against Row A of the primary table.

**Expected outcome:** Without distillation: activity accuracy drops 3-5%, detection mAP drops 1-3%, PSR event-F1 drops <1%, pose MAE increases <0.5 deg. Distillation's effect is strongest on the task with the highest gradient starvation (activity), confirming that distillation compensates for insufficient gradient signal.

**Compute cost:** 1x (already run as full MTL without distillation flag)

### 5.5 Batch size and gradient accumulation impacts

**Hypothesis:** Effective batch size of 16 is optimal for our setup. Smaller batches (effective 4 or 8) increase gradient noise and destabilize TAL assignment. Larger batches (effective 32) exceed VRAM and require more accumulation steps, which slow training without improving metrics.

**What it tests:** Three variants:
- Small batch: `--batch-size 2 --grad-accum-steps 2` (effective 4)
- Current batch: `--batch-size 4 --grad-accum-steps 4` (effective 16)
- Large batch: `--batch-size 4 --grad-accum-steps 8` (effective 32, slower training)

**Expected outcome:** Effective 4: detection suffers most (TAL assigner needs sufficient positive samples per batch). Effective 32: marginal improvement (if any) for activity (more classes per batch), but 2x slower training. Current 16 is Pareto-optimal.

**Compute cost:** 2x (small and large batch variants)

---

## 6. Ablation 5: Loss Ablations

### 6.1 CIoU vs. GIoU vs. EIoU for detection

**Hypothesis:** CIoU (Complete IoU, our current loss) outperforms GIoU (Generalized IoU) because CIoU considers center distance and aspect ratio in addition to overlap area. CIoU and EIoU (Efficient IoU, which normalizes center distance by the enclosing box diagonal) are expected to be similar.

**What it tests:** Replace `ciou_loss()` (line 75) with:
- **GIoU:** `from torchvision.ops import generalized_box_iou_loss`
- **EIoU:** Same as CIoU but without the aspect ratio consistency term (remove v and alpha from line 118-121)

**Expected outcome:** CIoU = EIoU > GIoU by 0.5-1.0 mAP. CIoU's aspect ratio term helps for elongated assembly objects (bolts, washers, plates). EIoU's simpler formulation may converge faster. GIoU's slower convergence may hurt on our small dataset.

**Compute cost:** 2x (GIoU and EIoU variants)

### 6.2 Focal-BCE vs. standard BCE for PSR

**Hypothesis:** Focal-BCE (our current default, gamma=2.0, alpha=0.25) reduces the contribution from easy negative frames (where PSR=0 for all components) and focuses learning on the rare positive frames (where PSR transitions occur). Standard BCE weights all frames equally, so the 99% negative frames dominate the loss.

**What it tests:** `--no-psr-focal` flag (code line 1725). Sets `use_focal=False` in `psr_loss()`, falling back to standard BCE.

**Expected outcome:** Without focal loss, PSR event-F1 drops from ~0.25 to ~0.05-0.10. The model predicts all negatives (loss is low because negatives dominate, but event-F1 is near-zero). This is the "focal-collapse-to-negatives" failure mode that focal loss is designed to prevent.

**This is the paper's Figure 3 (or supplementary).** A convincing visualization: loss curves are nearly identical (BCE and focal-BCE converge to similar loss values), but event-F1 diverges sharply (focal-BCE > 0.20, BCE < 0.05). This proves that BCE loss value alone is misleading for rare-event tasks.

**Compute cost:** 1x

### 6.3 Transition-aware weighting for PSR

**Hypothesis:** Transition-aware weighting (our `transition_boost=3.0` in `psr_loss()`, line 402) further improves event-F1 by up-weighting frames immediately before and after state transitions. This directly targets the failure mode where the model predicts transitions one frame too early or too late (a common error that standard focal-BCE does not penalize).

**What it tests:** Set `transition_boost=1.0` (no transition weighting) in `psr_loss()` at line 1005 (reads from `C.PSR_TRANSITION_BOOST`). Compare event-F1 at 3-frame tolerance vs. 1-frame tolerance.

**Expected outcome:** Without transition weighting: event-F1 at 3-frame tolerance drops modestly (0.02-0.05), but event-F1 at 1-frame tolerance drops sharply (0.05-0.10). The weighting mechanism primarily helps temporal precision, not detection recall.

**Compute cost:** 1x

### 6.4 Geodesic vs. cosine for pose

**Hypothesis:** The combined geodesic + cosine loss (current, line 470) outperforms either component alone. Geodesic loss on SO(3) is the principled rotation metric, but its gradient behavior near singularities can be unstable. Cosine loss on individual vectors provides a smoother gradient signal.

**What it tests:** Three variants:
- **Geodesic only:** Remove cosine term from `pose_loss()`.
- **Cosine only:** Remove geodesic term from `pose_loss()`.
- **Combined (current):** `0.5 * cosine + 0.5 * geodesic`.

**Expected outcome:** Combined > geodesic-only > cosine-only for final angular MAE. Geodesic-only may show training instability in early epochs. Cosine-only converges smoothly but saturates 1-2 deg above geodesic MAE. This ablation supports the paper's design choice but is unlikely to produce a large headline difference.

**Compute cost:** 2x (geodesic-only and cosine-only variants)

---

## 7. Ablation 6: Efficiency Ablations

Efficiency is the paper's co-primary claim alongside accuracy. These ablations quantify exactly where the parameters and FLOPs are spent.

### 7.1 Parameter count per component

**Hypothesis:** The shared backbone dominates total parameters (34.5M of ~45M total = 77%), with the FPN being the largest non-backbone component (~2.5M, 5.6%) despite being used by only one head.

**Method:** Run `torchsummary` or manual parameter counting for each sub-module:

| Component | Params | % of Total |
|-----------|--------|------------|
| MViTv2-S backbone | 34.5M | ? |
| BiFPN (lateral + TD + BU convs) | 2.5M | ? |
| Detection head (cls + reg) | 0.8M | ? |
| Activity head (3-layer MLP) | 2.0M | ? |
| PSR head (input_proj + Transformer + proj) | 1.8M | ? |
| Pose head (MLP: 768→256→6) | 0.2M | ? |
| **Total** | **~41.8M** | **100%** |

**Expected outcome:** The parameter distribution confirms that MTL efficiency is real: 41.8M vs ~100M for four separate ST models (each with its own 34.5M backbone + small head). The ratio is ~2.4x parameter efficiency, which is the paper's headline efficiency number.

**Compute cost:** 0x (static analysis)

### 7.2 FLOPs per component

**Hypothesis:** The backbone accounts for >90% of total FLOPs because MViTv2-S's multi-head self-attention at 56×56×96 resolution (P2) dominates compute. Head FLOPs are negligible in comparison.

**Method:** Use `torchprofile` or `fvcore.nn.FlopCountAnalysis` to measure FLOPs per forward pass for each sub-module. Report as GMACs (Giga multiply-accumulate operations).

**Expected outcome:** Total ~7.2 GMACs per frame at 224px. Backbone ~6.5 GMACs (90%), FPN ~0.5 GMACs (7%), heads combined ~0.2 GMACs (3%). Four ST models would total ~28 GMACs (4 × 7 GMACs for four backbone passes). The MTL model at ~7.2 GMACs is ~4x more compute-efficient at inference.

**Compute cost:** 0x (static analysis using profiling tools)

### 7.3 Latency (batch=1) per component

**Hypothesis:** Single-forward-pass latency on RTX 3060 (12GB) is ~40ms per frame (25 fps) for the full MTL model. Deploying four ST models sequentially would take ~160ms (6.25 fps). The MTL model achieves real-time (≥24 fps) while ST models do not.

**Method:** Time each sub-module's forward pass over 100 warmup + 100 measured iterations with `torch.cuda.Event`. Report mean ± std in milliseconds.

| Component | Latency (ms, batch=1) | % of Total |
|-----------|----------------------|------------|
| MViTv2-S backbone | ? | ? |
| BiFPN | ? | ? |
| Detection head (all 4 levels) | ? | ? |
| Activity head | ? | ? |
| PSR head | ? | ? |
| Pose head | ? | ? |
| **Total** | **?** | **100%** |

**Expected outcome:** Backbone ~35ms (87%), FPN ~3ms (8%), heads combined ~2ms (5%). Total ~40ms. The small head latency confirms the efficiency claim: adding three extra tasks adds only ~5% latency overhead.

**Compute cost:** 0x (profiling, no training)

---

## 8. Ablation 7: Data Ablations

### 8.1 Training dataset size impact (25%, 50%, 75%, 100%)

**Hypothesis:** MTL provides larger relative benefits at smaller dataset sizes because shared representations help when individual tasks have insufficient data. The gap between MTL and ST should widen as dataset size decreases.

**What it tests:** Sub-sample the training dataset at 25%, 50%, and 75% of original size (stratified by recording session, not random — to maintain temporal coherence). Train full MTL and ST-pose baselines at each size. Compare MTL/ST ratios.

| Dataset % | det mAP | act top-1 | psr F1 | pose MAE | MTL/ST pose |
|-----------|---------|-----------|--------|----------|-------------|
| 25% | ? | ? | ? | ? | ? |
| 50% | ? | ? | ? | ? | ? |
| 75% | ? | ? | ? | ? | ? |
| 100% | ? | ? | ? | ? | ? |

**Expected outcome:** The MTL/ST ratio for pose (the head with strongest positive transfer signal) increases from ~0.95 at 100% data to ~1.05-1.10 at 25% data — meaning MTL *beats* ST at small data sizes. This is the paper's strongest evidence for positive transfer. For detection and activity, the ratio also improves but may not cross 1.0.

**This is the paper's Figure 4.** Cross-task transfer map as a function of data availability. If the trend is monotonic (smaller data = larger MTL benefit), it directly supports the paper's central claim: MTL helps most when data is scarce, which is the realistic deployment scenario for industrial assembly verification.

**Compute cost:** 6x (4 dataset sizes × MTL + 2 dataset sizes × ST-pose for comparison). Can be reduced to 4x by running ST-pose only at 25% and 100%.

### 8.2 Without pose labels: does pose help other tasks?

**Hypothesis:** Removing pose supervision tests whether the head-pose estimation task provides positive transfer to detection and activity (via better spatial understanding of the worker's viewpoint) or negative transfer (noise from pose-specific features corrupting shared representations).

**What it tests:** Train full MTL model with `NUM_POSE_COMPONENTS = 0` or equivalent flag to disable pose supervision. The pose head still exists but receives zero loss — it neither contributes to the shared backbone nor benefits from it.

**Expected outcome:** Detection and activity metrics degrade 0-3% without pose supervision, suggesting pose provides mild positive transfer. The effect is small because pose's gradient signal (0.44 norm) is already the second-weakest. If detection degrades significantly (>3%), pose provides meaningful spatial context.

**Compute cost:** 1x

### 8.3 Without PSR labels: does PSR help other tasks?

**Hypothesis:** PSR has the strongest gradient (3.18 norm, 312x vs activity). Removing PSR supervision will significantly affect other tasks — either negatively (PSR was providing useful temporal structure to the shared backbone) or positively (PSR was dominating the shared backbone at the expense of other tasks).

**What it tests:** Disable PSR supervision (set PSR weight to zero, or skip PSR loss computation). Keep the PSR head in the model (it still runs forward but contributes no gradient).

**Expected outcome:** Without PSR, detection and activity *improve* 1-5% because PSR's dominant gradient (3.18 norm) was putting the backbone into a regime that primarily serves PSR needs. This is the most important negative result: the gradient starvation is structural, and removing the dominant task helps the starved tasks.

**This is the paper's Table 3.** Per-task transfer map measured by leave-one-task-out ablation. Matrix form: column = removed task, row = affected task metric change. This directly measures inter-task interference (positive or negative).

**Compute cost:** 1x

---

## 9. Ablation 8: What Claude Science Should Find

These are not experiments we run — they are literature searches Claude Science should perform to validate or challenge our ablation design.

### 9.1 Ablation study designs from top MTL papers

Claude Science should find and report how the following papers structure their ablations. We need to know if our design is missing any standard category:

- **"Multi-Task Learning with Gradient Surgery" (Yu et al. 2020, PCGrad paper):** How did they structure their ablation? Did they test PCGrad vs. random gradient ordering? Did they ablate per-task weighting separately from gradient surgery?

- **"Uncertainty in Multi-Task Learning" (Kendall et al. 2018, the original Kendall paper):** Did they present per-task MTL/ST ratios? Did they ablate the log-var caps?

- **"GradNorm: Gradient Normalization for Multi-Task Learning" (Chen et al. 2018):** What ablation comparisons did they run? How did they handle tasks with different loss scales (which is our central problem)?

- **"Taskonomy: Disentangling Task Transfer Learning" (Zamir et al. 2018):** Did they measure per-task transfer matrices? Does their taxonomy predict which tasks transfer positively in our setting (egocentric video, industrial assembly)?

- **"Nash-MTL: Solving Multi-Task Learning as a Nash Bargaining Problem" (Navon et al. 2022):** What ablations did they present? Did they compare to Kendall + PCGrad? What was the compute cost of their method vs. alternatives?

- **"Just Pick a Sign: Optimizing Deep Multitask Models with Gradient Sign Dropout" (Chen et al. 2020, GradDrop):** How did they structure their comparison? Did they include equal-weight baselines?

- **"Polytope: Multi-Task Learning with Conflict Resolution" (Dimitriadis et al. 2024):** Any ablation designs that test conflict resolution vs. loss weighting as separate mechanisms (which is our exact question)?

- **"Multi-Task Self-Supervised Learning for Human Activity Analysis" (any egocentric video MTL paper):** How did they handle activity recognition MTL? What was their MTL/ST ratio?

### 9.2 Missing ablation categories

Claude Science should also identify any ablation category we have not considered that appears in the top papers:

- **Random seed robustness:** Do top papers report multi-seed stats? How many seeds?
- **Sensitivity analysis:** Do top papers sweep hyperparameters (LR, batch size, weight decay) per ablation? Or fix them across all ablations?
- **Statistical significance:** Do any MTL papers report confidence intervals for MTL/ST ratios?
- **Component removal order:** Is there a standard order for ablating components (remove biggest first? remove most expensive first? random order?)
- **Interaction effects:** Do any papers test 2-way or 3-way interactions between ablation factors (e.g., does Kendall + PCGrad interact positively, negatively, or independently)?

### 9.3 Compute budget benchmarking

Claude Science should find published compute budgets for ablation studies in top venues (AAIML, CVPR, ICCV, ECCV, NeurIPS, ICLR):

- Typical number of ablation rows in an MTL paper accepted at AAIML
- Typical total compute hours reported in the ablation section
- Do papers report compute cost per ablation row?
- What is the precedent for "zero-cost ablations" (profiling, static analysis) vs. "full training ablations"?
- Do any papers trade completeness for compute (e.g., running ablations on a subset of epochs and verifying that the subset ranking matches full-training ranking)?

---

## 10. Compute Budget Summary

| Ablation Group | Runs | Compute (x of full) | Priority |
|----------------|------|---------------------|----------|
| 2.1 MTL vs ST baseline | 5 | 3.0x | **MANDATORY** |
| 2.2 MTL minimal levers | 1 | 1.0x | Nice-to-have |
| 3.1 Kendall capped vs uncapped | 1 | 1.0x | **MANDATORY** |
| 3.2 Kendall + PCGrad vs Kendall-only | 1 | 1.0x | **MANDATORY** |
| 3.3 Kendall vs GradNorm vs DWA vs equal | 3 | 3.0x | **MANDATORY** (at least GradNorm) |
| 3.4 EMA on/off | 1 | 1.0x | **MANDATORY** |
| 3.5 Logit-adjust placement | 1 | 1.0x | Nice-to-have |
| 4.1 Task-conditional adapters | 1+0.5 | 1.5x | **MANDATORY** (reviewer defense) |
| 4.2 BiFPN vs simple FPN vs no FPN | 2 | 2.0x | Nice-to-have |
| 4.3 Decoupled vs coupled detection | 1 | 1.0x | Nice-to-have |
| 4.4 Causal vs bidirectional PSR | 1 | 1.0x | **MANDATORY** (reviewer defense) |
| 4.5 Activity head variants | 2 | 2.0x | Nice-to-have |
| 5.1 Detection aug on/off | 1 | 1.0x | **MANDATORY** |
| 5.2 SWA on/off | 0 | 0.0x | Good for supplementary |
| 5.3 Class-balanced sampling on/off | 1 | 1.0x | Nice-to-have |
| 5.4 Distillation on/off | 1 | 1.0x | **MANDATORY** |
| 5.5 Batch size sweep | 2 | 2.0x | Nice-to-have |
| 6.1 CIoU vs GIoU vs EIoU | 2 | 2.0x | Nice-to-have |
| 6.2 Focal-BCE vs BCE | 1 | 1.0x | **MANDATORY** |
| 6.3 Transition weighting on/off | 1 | 1.0x | Nice-to-have |
| 6.4 Geodesic vs cosine | 2 | 2.0x | Supplementary |
| 7.1 Parameter count | 0 | 0.0x | **MANDATORY** (free) |
| 7.2 FLOPs per component | 0 | 0.0x | **MANDATORY** (free) |
| 7.3 Latency per component | 0 | 0.0x | **MANDATORY** (free) |
| 8.1 Dataset size impact | 4-6 | 4.0x | **MANDATORY** |
| 8.2 Without pose labels | 1 | 1.0x | Nice-to-have |
| 8.3 Without PSR labels | 1 | 1.0x | **MANDATORY** |
| **Total mandatory** | **18-20 runs** | **~20x GPU-days** | **~40 days on 1 GPU** |
| **Total if all are run** | **~35 runs** | **~35x GPU-days** | **~70 days on 1 GPU** |

**Practical recommendation:** Run the mandatory ablations (~20x) on two GPUs in parallel (RTX 3060 + RTX 5060 Ti), achieving ~20 calendar days. Reserve nice-to-have ablations for supplementary material or for the rebuttal period.

**Budget management strategies:**
- **Progressive disclosure:** Run the 3.3 optimization weighting sweep (GradNorm, DWA, equal weights) at 25 epochs instead of 50, if a validation run shows ranking stabilizes by epoch 25.
- **Proxy metrics:** For 4.2 (FPN ablation), eval after epoch 10 instead of 50; FPN ablation ranking is unlikely to change in later epochs.
- **Reuse checkpoints:** The "without PSR labels" ablation (8.3) shares 90% of the training loop with the full MTL baseline; code paths diverge only in the PSR loss computation. A single run with an on/off switch after epoch 25 could test both conditions.
- **Zero-cost cross-validation:** Ablations 7.1-7.3 (efficiency) are free — profile once and done.

---

## 11. Analysis Protocol

Every ablation must follow this analysis protocol to ensure fair comparison:

### 11.1 Statistical rigor
- **Minimum 1 seed per ablation:** Full model runs with 3 seeds (to establish noise floor). Key ablations (3.1, 3.2, 3.3, 5.1, 6.2, 8.1, 8.3) run with 2 seeds. Minor ablations run with 1 seed.
- **Report mean and range** for 2+ seed runs. The noise floor (standard deviation of full model across 3 seeds) defines the "significant difference" threshold.
- **Bootstrap confidence intervals** for MTL/ST ratios if the noise floor is high.

### 11.2 Metric recording
For each ablation run, record:
- **Train metrics:** Per-task loss (det, act, psr, pose), log-var values, EMA values, learning rate, gradient norms per head (backbone only). Logged every 100 batches.
- **Val metrics:** mAP@0.5 (det), top-1 accuracy (act), event-F1@3 (psr), angular MAE (pose). Logged every 5 epochs.
- **Test metrics:** Same as val, computed at end of training (best checkpoint + SWA checkpoint).
- **Diagnostic dumps:** Per-head gradient norms at epoch 1, 10, 25, 50 (to track starvation dynamics).

### 11.3 Failure mode documentation
If any ablation produces unexpected results (MTL/ST ratio > 1 for a starved head, or negative transfer from a dominant head), document:
- The direction of surprise (better or worse than expected)
- The diagnostic outputs that explain it (gradient norms, log-var trajectories, per-class accuracies)
- The actionable insight (should we change our default configuration?)

### 11.4 Paper placement
| Section | Ablations |
|---------|-----------|
| Table 1 (primary) | 2.1 MTL vs ST |
| Figure 1 (main) | 3.1 Kendall capped vs uncapped (log-var trajectories) |
| Figure 2 (main) | 8.1 Dataset size impact (MTL/ST ratio as function of data) |
| Figure 3 (main) | 6.2 Focal-BCE vs BCE for PSR (loss vs event-F1 divergence) |
| Table 2 (main) | 5.1 Det aug on/off, 5.4 Distillation on/off, 6.3 Transition weighting |
| Table 3 (main) | 8.2-8.3 Leave-one-task-out transfer matrix |
| Table 4 (supp) | 4.4 Causal vs bidirectional PSR, 4.1 Task-conditional adapters |
| Table 5 (supp) | 3.2 Kendall+PCGrad vs Kendall-only, 4.2 FPN ablations |
| Supplementary | 3.3 GradNorm/DWA/equal, 3.4 EMA on/off, 3.5 Logit-adjust, 5.2 SWA, 4.3, 4.5, 5.3, 5.5, 6.1, 6.4 |

---

*End of Doc 222. Ablation study design reviewed by Claude Opus (Docs 150-207) and presented to Claude Science for validation and literature-grounded improvement suggestions.*
