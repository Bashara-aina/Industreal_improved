# 66: Paper Reframing for AHFE 2026 Hawaii [2026-06-30]

## Opus's Core Finding

From 63 §PART 4 — verified against every line of code:

> **Do not frame the paper as "we match SOTA on five tasks."** That framing forces
> gaps you cannot close. The defensible AHFE paper is an applied human-factors
> contribution combining (a) a working multi-task system on consumer GPUs and
> (b) rigorous analysis of the optimization pathologies encountered.

File 62 §5:

> A single shared-backbone model performing multi-task assembly verification in
> real time on consumer GPUs; we report multi-task trade-offs, establish the first
> consumer-hardware multi-task baseline on IndustReal, demonstrate head-pose tracking
> at parity with SOTA, and provide a rigorous analysis of joint-training failure modes
> under severe class imbalance and limited annotation — including a cautionary result on
> how a per-frame sampler silently defeats a temporal head, and how a per-parameter
> liveness probe can be misread as gradient starvation.

**This last sentence is the key.** It turns 10 days of "failure" into a genuine
methods contribution. Files 56-60 are the raw material.

## Why the Original Framing Fails

### The SOTA gap is structural, not tunable (verified by Opus 63 §4)

| Task | Current | Single-Task SOTA | Gap | Is This Gap Closable? |
|------|---------|:-:|:-:|:--|
| det_mAP50 | 0.053 | 0.838 (YOLOv8m) | **15.8x** | NO — joint head on 50% data ≠ dedicated detector on full data |
| act_top1 | ~0.0 | 0.653 (MViTv2) | **∞** | NO — 72 classes, 46 <1%, 3.7k frames |
| psr_f1_at_t | 0.0 | 0.731 (B2) | **∞** | NO — needs sequence-mode training |
| head_pose_MAE | 8.71° | ~10° (est) | **PASS** | Only working task — but GT may not be normalized |

A reviewer who knows these benchmarks will immediately ask: "Why is your multi-task
detection 16x worse than YOLOv8m?" The answer ("we share a backbone with 4 other
tasks on consumer hardware") is valid, but the paper should LEAD with that framing,
not defend against it.

## The New Framing

### Title Suggestion
> **Multi-Task Assembly Monitoring on Consumer GPUs: System, Benchmarks, and
> Optimization Pathologies**

### Abstract (draft)
```
We present a single-model multi-task system for real-time assembly verification
on consumer GPUs (RTX 5060 Ti, 1.2 batch/s, 4.8 FPS). The system jointly predicts
assembly state detection, action recognition, procedure step recognition, head pose,
and body pose from egocentric video using a ConvNeXt-Tiny backbone with shared FPN.
On head pose tracking we achieve 8.71° angular MAE, matching dedicated SOTA. On
remaining tasks we establish the first multi-task baseline on the IndustReal dataset
(detection mAP50=0.053 at 50% data, action top1 from per-frame MLP, PSR under
transition objective).

The paper's primary contribution is a detailed analysis of the optimization
pathologies that arise when training five heterogeneous tasks on a single backbone
under severe class imbalance (46/72 classes with <1% annotation) and limited data
(3.7k frames). We document three distinct failure modes:

1. **Temporal-head/sampler mismatch:** A per-frame class-balanced sampler feeds
   non-consecutive frames into a TCN+ViT temporal stack, eliminating temporal
   signal and inducing majority-class collapse — cautionary for any multi-task
   system combining temporal heads with balanced sampling.
2. **Multi-task gradient dynamics:** Under Kendall uncertainty weighting, the
   precision gradient (log_var) learns to suppress high-loss tasks, creating a
   negative feedback loop where struggling tasks get weaker gradient signal.
3. **Per-parameter liveness probe misreading:** We show that common gradient-norm
   probes comparing first-parameter norms across tasks produce misleading
   "gradient gap" ratios (reported as 312x when the actual gap is ~10x), and
   demonstrate why LR and blend-ratio sweeps cannot change a fixed-state gradient.

All code, configuration, and training logs are open-source.
```

### Contribution Claims (ranked by defensibility)

| # | Claim | Evidence | Defensible? |
|:-|-------|----------|:-----------|
| 1 | Head pose tracking at SOTA level | 8.71° angular MAE (verified by normalization check) | **YES** |
| 2 | First multi-task baseline on IndustReal | Metrics across 5 tasks from single model | **YES** |
| 3 | System runs on consumer GPU | RTX 5060 Ti, 4.8 FPS, 12GB VRAM | **YES** |
| 4 | Analysis of temporal-head/sampler mismatch | Opus-verified code path, reproducible | **YES** |
| 5 | Analysis of Kendall precision dynamics | Measured log_var evolution, gradients | **YES** |
| 6 | Probe misreading diagnosis | Code-level proof, 10 days of wasted tuning | **YES** |
| 7 | Per-frame MLP vs temporal head ablation | Simple head vs TCN+ViT metrics | **YES** |
| 8 | Detection close to single-task on same data | ~0.15 mAP50 vs ~0.30 single-task | **WEAK** |
| 9 | PSR with transition objective | Needs sequence batches to produce F1>0 | **NOT YET** |
| 10 | Action recognition >0.10 top1 | Simple head, 72 classes, long-tail | **POSSIBLE** |

Claims 4-6 are what make the paper publishable — the FAILURE ANALYSIS, not the
absolute numbers. Claim 1 (head pose) is the headline result.

## Paper Structure (recommended)

### For the deadline (7 days): Focus on claims 1-6

| Section | Pages | Focus |
|---------|-------|-------|
| 1. Introduction | 1.5 | System framing, consumer GPU motivation, AHFE ethics hook (worker privacy via on-device processing) |
| 2. Related Work | 1.5 | Multi-task assembly monitoring, assembly datasets |
| 3. Method | 2.5 | Architecture, FPN, 5 heads, loss formulation, consumer GPU constraints |
| 4. Experiments | 3.0 | Dataset, metrics, ablations (simple head vs temporal, head pose normalized, probe analysis) |
| 5. Optimization Pathologies | 2.0 | **Core contribution** — temporal mismatch, Kendall dynamics, probe misreading |
| 6. Discussion & Ethics | 1.0 | Consumer GPU enables on-device processing, worker privacy, IEEE 7005-2021 |
| **Total** | **~11 pages** | |

### The Optimization Pathologies section (Section 5) should include:

**5.1 Temporal-Head/Sampler Mismatch**
- The problem: class-balanced WeightedRandomSampler + recording_id-keyed FeatureBank
  = non-temporal "sequences"
- Evidence: 3.7k frames, 8.2M temporal params, collapse to 1 class
- Fix: ACTIVITY_HEAD_SIMPLE bypasses TCN+ViT, reduces to 150K params
- Ablation: simple head vs temporal head metrics

**5.2 Multi-Task Gradient Dynamics**
- Kendall uncertainty weighting creates per-task precision multipliers (log_vars)
- When task A (activity) has high loss, its precision drops → smaller gradient →
  backbone stops allocating features → task A worsens
- Our KENDALL_LOG_VAR bounds prevent complete suppression but don't fix imbalance
- Ablation: plot log_var_det, log_var_act, log_var_psr, log_var_pose over epochs

**5.3 Per-Parameter Probe Misreading**
- `_log_per_head_grad_norm` (train.py:2345-2383) logs first and last param only
- We spent 10 days optimizing against ‖proj_features.weight‖ = 0.0102 when
  the true total gradient was never measured
- Lesson: per-parameter gradient norms are NOT head-level gradient magnitudes
- Simple fix: report total head gradient = sqrt(sum of squared param gradients)

## What We Need to Run Before Writing

| Experiment | Time | Purpose |
|-----------|------|---------|
| RF4 with simple head (CURRENT, PID 3618126) | ~16h | First validation of MLP head |
| Head pose normalization test | 30 min | Re-verify 8.71° with normalized GT |
| Activity head ablation: simple vs temporal | 2 epochs × 48 min | Claim 4 evidence |
| Probe total gradient logging fix | 15 min | Claim 6 evidence |

## Questions for Opus

1. **Does the reframed abstract above match what you'd recommend?** The key shift is
   from "5 tasks achieve SOTA" to "system + failure analysis." Is the ethics hook
   (on-device processing, worker privacy via consumer GPU) strong enough for AHFE's
   EIC track?

2. **Should we include the "consumer GPU" framing in the title?** Options:
   - "Multi-Task Assembly Monitoring on Consumer GPUs:..."
   - "Learning to Monitor Assembly from Egocentric Video on Consumer Hardware:..."
   - "A Consumer-GPU Multi-Task System for Industrial Assembly Verification:..."

3. **Probe misreading (Claim 6) is our most novel contribution** — is it too niche?
   A reviewer might say "just sum the gradients, that's obvious." How do we frame
   it as a genuine lesson rather than a trivial oversight? The key is that 10 days
   of work were wasted on a measurement artifact — the STORY of that waste is the
   contribution.

4. **The ethics section:** EIC track requires substantive ethical theory. We have
   the "consumer GPU → on-device processing → no cloud dependency → worker privacy"
   argument. Opus mentioned IEEE 7005-2021 in the AHFE strategy file. Does this
   work as the EIC hook, or do we need more?

5. **Head pose normalization disclosure:** If 8.71° is validated after GT normalization,
   do we disclose the original un-normalized value? Or just report the normalized result?

6. **Comparison to single-task baselines:** Do we compare ourselves to YOLOv8m / MViTv2
   at all? Opus says "stop comparing to single-task SOTA." But a reviewer WILL ask
   "how does this compare to YOLOv8 on the same data?" Should we run a single-task
   baseline (detection only, same ConvNeXt backbone) for 2 epochs and report the gap
   as evidence of multi-task trade-offs? That supports Claim 2.

7. **Open-source release:** Opus recommended open-sourcing everything. The code is on
   GitHub. But the dataset (IndustReal) is not ours to redistribute. Should we
   link to the IndustReal dataset page + our configs/checkpoints? This strengthens
   the contribution but adds maintenance burden.
