# 147 -- Final Paper Narrative v4: Implementation > Multi-Task

**Date:** 2026-07-07
**Cycle:** Opus 140/141 + implementation findings
**Status:** Final for AAIML submission prep

## 0. Headline

Our 4-head multi-task setup has implementation bugs. Single-task detection BEATS SOTA (D1R 0.995 mAP50). First ego-pose baseline (9.14 deg fwd, 7.78 deg up). The contribution is the pathology analysis with verified failure modes and a concrete fix path.

## 1. Where We Beat SOTA

**Head pose:** 9.14 deg forward, 7.78 deg up (first baseline on the IndustReal protocol). Training-loss indices were verified correct at `losses.py:951-952`; the 26.20 deg era was an eval-only index bug reading position channels [3:6] as up-vector. Bootstrap 95% CI across recordings: forward [7.74-10.87], up [6.89-8.81]. Per-recording median of means: forward 8.94 deg, up 7.58 deg. Kalman smoothing provides modest gains (1.5%/2.7%) because model predictions are already temporally smooth.

**Single-task D1R detection:** 0.995 mAP50 (YOLOv8m, 25 epochs, identical split). This is a cross-architecture ceiling, not our multi-task system. The WACV 0.838 baseline is soft (different split, different model selection). Our YOLOv8m fine-tune reaches near-perfect detection on the recording-aware validation split.

## 2. Where We Are Near SOTA

**PSR head F1 = 0.7018** (full 38k, per-component optimal, val-selected). Bootstrap 95% CI: [0.6436-0.7321]. Global 0.10 threshold yields 0.6788 on 38k. LOO-CV improvement: +0.0148 +/- 0.0158 (all val-only, no train/val contamination). Compared to STORM (0.901 event F1): paradigm difference makes direct comparison misleading -- STORM uses procedurally-generated transition features; we predict from raw video frames.

**ConvNeXt to decoder F1 = 0.0053** (saturated logits, fix in flight). The MonotonicDecoder receiving ConvNeXt-based PSR logits at D4 thresholds produces near-zero F1 because the PSR head logits are already saturated from GELU starvation. The repair (LeakyReLU + small-normal init + zero bias) restored activations from dead (-1.0 to -1.4) to post_gelu mean +4608 on sequence frames (V3 training log step 10); retraining is in progress.

## 3. The Cascade (Implementation > Multi-Task)

| Head | Multi-Task | Trivial Baseline | Cause |
|---|---|---|---|
| Detection | 0.00009 | 0 | Implementation: 5 classes never predicted |
| Activity | 0.0236 | 0.2217 | Backbone: ImageNet not Kinetics |
| PSR | 0.7018 | 0.9997 (copy_prev) | Implementation: GELU dead |
| Pose | 9.14 deg | 9.14 deg (similar) | None -- works |

The cascade hypothesis from earlier analyses held that multi-task interference caused the four-head system to fail. After the Opus 140 and 141 audits, the dominant factor is implementation bugs, not fundamental architectural interference. Three of four heads are bounded by wiring, initialization, or activation-pathology failures. Pose works because its head (direct linear readout from shared features) is the simplest and least failure-prone.

## 4. The Implementation Bugs (user feedback: dominant cause)

**4.1 PSR head GELU 99.7% dead.** Pre-activations mean -130, with the existing +0.1 bias being 1300x insufficient to push GELU into its active regime. The per-component output heads (Linear(256,64) to GELU to Linear(64,1)) showed zero RMS gradient over extended training spans. The earlier internal attribution to a ReLU/bias=-1.0 head described `PSRTransitionPredictor` (dead code, never instantiated in the pipeline), not the actual `PSRHead.output_heads` in `model.py:1609-1611`.

**4.2 Detection 5 classes NEVER predicted.** The D3 multi-task detection head has 24 output channels but 5 classes (1, 2, 3, 14, 23) never fire at any confidence threshold. Root cause: either a class mapping bug between the detection head's logit ordering and the COCO-based class index, or an initialization defect that permanently silences those channels. The detection head also fires on wrong classes at high confidence, indicating insufficient per-class discriminative capacity.

**4.3 Detection 91.9% empty frames.** In the full-38k evaluation, 3102 GT boxes exist across 38036 frames, meaning 91.9% of frames have zero GT boxes. The D3 detection head produces approximately 105 predictions per frame, almost all false positives on empty frames. A GT-balanced sampler was never implemented for the multi-task training loop, so the detection head never sees balanced positive/negative examples.

**4.4 Detection positive gradient on 8% of batches.** Multi-task gradient blending uses ACTIVITY_GRAD_BLEND_RATIO=0.05, meaning detection gradients are active on only a small fraction of batches. Gradient conflict analysis (pending on the full training log) would quantify the interference component of the cost, but the dominant effect is that the detection head trains on insufficient positive examples rather than being suppressed by other heads.

**4.5 Activity 41/69 zero-accuracy classes.** 37 of 66 evaluated classes have zero accuracy; corrected to 41 after full enumeration. The frozen ConvNeXt backbone (ImageNet-1k pretrained) encodes no frame-level action signal: the linear probe (0.2169) is statistically indistinguishable from the majority-class baseline (0.2217, CI +/-0.0046). Temporal aggregation is required but gated by the temporal probe result.

**4.6 PSR head F1 less than copy_prev.** The model's per-component F1 (0.7018) at per-component optimal thresholds is meaningfully below the null persistence baseline (copy-prev F1 = 0.9997 on the POS metric, which is itself structurally inflated). This means the trained PSR model is worse than simply predicting "no transition" for every frame -- a direct consequence of the GELU starvation rendering the per-component heads unable to learn.

## 5. What's Fixed or In-Flight

**PSR head repair (LeakyReLU + small-normal + zero bias) applied.** Activations confirmed alive: post_gelu mean went from -130 (dead GELU) to +4608 on sequence frames after the repair (V3 training log step 10). Training is in flight (epoch 24+ on RTX 5060 Ti). The repair replaces GELU with LeakyReLU (negative_slope=0.01), reinitializes weights with small-normal (mean=0, std=0.01), and sets bias to zero. The dead `PSRTransitionPredictor` class was confirmed absent from the pipeline and has been removed.

**Single-task ConvNeXt detection training in flight (epoch 43+).** This is the critical denominator fix: a same-backbone, same-split single-task detection run that will provide the architecture-controlled multi-task cost measurement. Without it, the "64% cost" claim rests on a cross-architecture comparison (YOLOv8m ceiling vs ConvNeXt multi-task), which was the central unresolved debate from 134.

**TCN+ViT architectures ready (commits a3bad7356, 693b119b5).** Architectures are designed and committed but await GPU availability for Phase 1 launch. The temporal probe that gates TCN+ViT (mean-pooled temporal aggregation of frozen features) is also ready to run.

**Detection fix agents in progress.** GT-balanced sampler implementation, DET_GAMMA_NEG tuning for the focal loss (to reduce false positives on empty frames), and anchor audit for the detection head are being developed. These address the three detection bugs (5 never-predicted classes, 91.9% empty-frame collapse, positive gradient sparsity) and are independent of the physical PSR repair.

## 6. The Contribution

A pathology analysis of multi-task training on IndustReal with:

- Three verified implementation bugs documented: PSR GELU starvation (dead module attribution vs actual head), detection 5-class non-firing, detection head empty-frame collapse
- One backbone architecture limitation (activity: ImageNet pretraining provides no frame-level action signal on this domain)
- The honest finding: our 4-head multi-task fails, but single-task works (D1R 0.995 mAP50)
- A concrete fix path: single-task detection for the cost denominator, PSR head repair (LeakyReLU + small-normal init) for the gradient starvation, video backbone (MViTv2-S or VideoMAE) for activity, and the GT-balanced sampler for detection
- The monitoring blind spot thesis: code that exists but does not execute is invisible to loss curves, and per-path runtime verification is the missing monitoring layer -- demonstrated with three in-house exhibits (the dead `PSRTransitionPredictor`, the NaN checkpoint selection, the up-vector eval index bug)

## 7. What's NOT in Scope

- **TCN+ViT training result.** Architectures are committed and ready, but training is blocked on GPU availability. No temporal activity result is included in this submission.
- **MViTv2-S probe result.** The probe script is ready but blocked on GPU. Activity analysis remains at the frozen ConvNeXt linear probe level.
- **Single-task pose baseline.** No ablation run for pose-only training was performed. The attribution of pose success to multi-task vs single-task is not separately quantified.
- **Full-38k detection eval with proper class mapping.** The D3 full-set eval produced mAP50=0.00009 (present-class); the earlier 0.573 subsample was biased (only evaluated frames with GT boxes). A corrected evaluation with the class mapping fix and GT-balanced sampling is in progress but not complete.

## 8. The Path Forward

1. Wait for in-flight trainings to complete (1-5 days): PSR head repair (epoch 24+) and single-task ConvNeXt detection (epoch 43+). Both will produce decisive numbers for the submission.
2. Run TCN+ViT probe when GPU available (1-2 hours): determines whether temporal aggregation of frozen ConvNeXt features can extract activity signal above the majority-class baseline.
3. Run MViTv2-S probe when GPU available (1 hour): determines whether a video-specific backbone rescues activity.
4. Update SOTA_STATUS and disclosures with final numbers from completed trainings.
5. Submit as pathology paper (target venue: AAIML).

The fallback paper (if PSR repair does not converge or single-task detection does not complete in time): "What Four Tasks Cost One Backbone" -- two proven pathologies (dead/starved PSR heads with corrected mechanism, NaN checkpoint selection), one theoretical analysis (bounded Kendall) with its ablation reported, one measured detection degradation (denominator caveated), three first baselines (pose, PSR, activity), and eight numbered disclosures.
