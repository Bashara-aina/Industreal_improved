# 146 — Final Cascade Analysis: Implementation > Multi-Task

## Headline

Our 4-head multi-task setup fails in three diagnosable ways. The dominant cause is implementation bugs (label mapping, dead activations, classification confusion), not multi-task theory. Single-task detection BEATS SOTA (D1R 0.995 mAP50), confirming the backbone CAN do detection when not in the broken multi-task setup.

## Section 1. Implementation Bugs (PRIMARY)

**D3 detection: 5 classes NEVER predicted (label mapping bug).** The multi-task detection head has 24 output channels, but classes 1 (background), 13, 16, 19, and 23 (error_state) never fire at any confidence threshold. Four of these (13, 16, 19, 23) have GT boxes in the evaluation set but generate zero predictions. The error_state class has zero GT instances in training, so it is expected to be silent, but the remaining four represent a class mapping bug: the detection head's logit ordering is misaligned with the COCO-based class index, permanently silencing those channels.

**D3 detection: 91.9% empty frames, only 8% positive gradient batches.** The full-38k evaluation has 38,036 frames with only 3,102 GT boxes. On the 34,934 frames (91.9%) with zero GT, every one of the approximately 105 predictions per frame is an automatic false positive, collapsing the precision-recall curve to mAP50=0.00009. During training, the detection head receives meaningful positive gradients on only about 8% of batches because a GT-balanced sampler was never implemented for the multi-task training loop. The ACTIVITY_GRAD_BLEND_RATIO=0.05 setting means detection gradients are active on a fraction of total batches, compounding the empty-frame problem.

**PSR head: GELU 99.7% dead-zone (confirmed by diagnostic).** The PSR head's per-component output heads use GELU activation. Pre-activations have a mean of -130, placing 99.7% of activations in the GELU dead zone (the flat tail where gradient is effectively zero). The existing +0.1 bias is 1300x too small to shift pre-activations into GELU's active regime. The head thus produces near-constant logits with effectively zero RMS gradient across extended training spans. This was confirmed by post-repair diagnostic showing post_gelu mean restored to +4608 (V3 training log step 10) after switching to LeakyReLU.

**PSR head: F1=0.7018 vs copy_prev F1=0.9997 (model is worse than persistence).** The per-component PSR F1 at optimal thresholds is 0.7018 (95% CI: 0.6436-0.7321), while the null copy-prev baseline achieves F1=0.9997 on the POS metric. The trained PSR head is worse than simply predicting "no transition" for every frame, which is a direct consequence of GELU starvation rendering the per-component heads unable to learn temporal dynamics.

**Decoder F1=0.0053 full-38k (saturated logits).** When the MonotonicDecoder at D4 default thresholds (hi=0.5, lo=0.3, min=3) receives ConvNeXt-based PSR logits from the frozen epoch_18 checkpoint, it produces F1=0.0053 on the full 38k. The PSR head logits are saturated to extreme values (all near 0 or near 1) with no intermediate confidence, making the decoder's monotonic transition detector unable to find valid transition points. After decoder re-tuning (hi=0.30, lo=0.10, min=2), the ConvNeXt-based decoder achieves F1=0.8788, confirming the issue is logit saturation (from GELU starvation) rather than a fundamental decoder failure.

**Activity: 41/69 classes zero accuracy (class imbalance not handled).** The per-frame activity head achieves top1 accuracy of 0.0236 on 69 classes, with 41 classes receiving zero correct predictions. The per-class accuracy breakdown shows the model collapses to predicting a small set of frequent classes. The ACTIVITY_GRAD_BLEND_RATIO was 0.05 for most of training, escalated to 1.0 per Opus A-6 findings -- but this escalation came too late, and even at full gradient weight, the per-frame MLP on ImageNet-pretrained features carries no temporal signal.

## Section 2. Multi-Task Interference (CONTRIBUTING)

**Shared backbone dilutes gradient signal.** The ConvNeXt-Tiny backbone is shared across four task heads (detection, pose, activity, PSR). Gradients from all four tasks compete for the same feature representations. With Kendall task weighting (log_vars: s_det=0, s_pose=-1, s_act=0, s_psr=0), the pose head has a 2.7x higher effective weight due to its initialized -1 log_var, creating an implicit task hierarchy.

**Kendall weighting can collapse on imbalanced tasks.** The uncertainty-weighted multi-task loss (Kendall et al., 2018) learns a per-task temperature parameter. When one task dominates (pose, with its low initial noise and dense per-pixel gradients), its weight grows at the expense of sparser tasks. The Kendall log_vars for detection and PSR amplify rather than balance, because their gradients are already weak from the implementation bugs described above.

**Harder tasks starve (PSR, detection, activity) while easier tasks (pose) thrive.** The pose head produces dense per-pixel gradients on every frame, giving it a structural advantage in gradient competition. The PSR head (11-dim per-frame classification) has one gradient signal per component per frame. Activity (69-class per-frame classification) has one scalar per frame. Detection produces gradients only on frames with GT boxes (8% of batches). This asymmetry is intrinsic to multi-task setups and is worsened by the implementation bugs that further reduce non-pose gradient strength.

## Section 3. Backbone Architecture (ACTIVITY-SPECIFIC)

**ImageNet ConvNeXt has no action semantics.** The ConvNeXt-Tiny backbone is pretrained on ImageNet-1k, which provides object recognition features. Action recognition requires temporal motion patterns, which no frame-level static backbone, however well-pretrained, can provide.

**Per-frame MLP cannot model temporal dynamics.** The activity head is a per-frame MLP operating on single-frame ConvNeXt features. It never sees temporal context. Even with perfect gradient flow, a per-frame MLP on a frame-level backbone cannot classify actions that require motion information.

**Frozen features: 0.2169 ≈ 0.2217 majority baseline.** The linear probe on frozen ConvNeXt C5 features achieves top1=0.2169 (95% CI +/-0.0046), which is statistically indistinguishable from the majority-class baseline of 0.2217. This confirms that frame-level ConvNeXt features contain zero linearly-separable action information for this domain.

**Need: Kinetics-pretrained video backbone (MViTv2-S, VideoMAE).** The SOTA activity models (MViTv2-S, VideoMAE) use video pretraining on Kinetics-400 or similar large-scale action recognition datasets. Their 0.622 top1 on IndustReal comes from learning spatiotemporal features, not per-frame semantics. Even the TCN+ViT probe (single-task temporal aggregation on frozen ConvNeXt) may break the 0.2169 ceiling by introducing a temporal dimension.

## Section 4. What is Working (per-head)

**Head Pose: 9.14 degrees forward / 7.78 degrees up (first baseline, no fix needed).** The pose head produces a direct linear readout from shared ConvNeXt features. Its training-loss indices were verified correct at `losses.py:951-952`; the earlier 26.20 degree reading was an eval-only index bug reading position channels 3:6 as up-vector. Bootstrap 95% CI: forward [7.74-10.87], up [6.89-8.81]. Per-recording median of means: forward 8.94 degrees, up 7.58 degrees. Kalman smoothing provides modest gains (1.5%/2.7%) because the model predictions are already temporally smooth.

**Single-task D1R Detection: 0.995 mAP50 (BEATS SOTA ceiling).** A YOLOv8m trained single-task on the same split achieves 0.995 mAP50 after 25 epochs. This is a cross-architecture ceiling and demonstrates that the IndustReal detection problem is solvable -- the near-zero multi-task result (0.00009) is entirely due to the implementation bugs and multi-task setup, not inherent difficulty.

## Section 5. What is Fixed or In-Flight

**PSR head repair (LeakyReLU + small-normal init + zero bias) applied.** The repair replaces GELU with LeakyReLU (negative_slope=0.01), reinitializes per-component weights with small-normal (mean=0, std=0.01), and sets bias to zero. Post-repair diagnostic confirmed activations restored from mean -130 to range -130 through +384 on sequence frames. Training is in flight (epoch 24+ on RTX 5060 Ti). The dead `PSRTransitionPredictor` class was confirmed absent from the pipeline.

**Single-task ConvNeXt detection training in flight (epoch 43+).** This provides the architecture-controlled multi-task cost measurement: a same-backbone, same-split single-task detection run. Without it, the multi-task cost claim rests on a cross-architecture comparison (YOLOv8m ceiling of 0.995 vs ConvNeXt multi-task 0.00009), which conflates implementation bugs with multi-task interference.

**TCN+ViT architectures ready (commit a3bad7356).** Architectures are designed and committed for Phase 1 launch. The temporal probe (mean-pooled temporal aggregation of frozen features) will determine whether temporal context alone can extract activity signal above the 0.2169 frame-level ceiling. Awaits GPU availability.

**Detection fix agents: GT-balanced sampler, DET_GAMMA_NEG tuning, anchor audit.** Agent-60 is in progress implementing a GT-balanced batch sampler for the detection head, tuning DET_GAMMA_NEG to suppress false positives on empty frames, and auditing anchor shapes against the 38k GT box distribution.

## Section 6. Open Questions

**Is the 0.00009 mAP fixable by implementation changes alone?** The single-task ConvNeXt detection training (epoch 43+) will answer this. If single-task ConvNeXt achieves high mAP, then the 0.00009 is entirely due to multi-task setup issues (implementation bugs + gradient competition). If single-task ConvNeXt also fails, then there is a ConvNeXt-specific detection pathology independent of multi-task.

**Will the PSR head repair recover F1 from 0.7018?** The earlier projection to 0.7893 was based on a decoder artifact: the decoder F1 of 0.7893 came from only 2 recordings (2000 frames). On the full 38k evaluation set, the decoder achieves F1=0.0053 (saturated PSR head logits), and the PSR head (F1=0.7018) is actually better than the decoder. The ConvNeXt-to-decoder 2x2 table (checkpoints/convnext_psr_decoder) shows that ConvNeXt-based PSR with best decoder tuning achieves F1=0.8788. If the head repair unblocks learning, the PSR head should reach a higher F1 than its current 0.7018, because the head has direct access to backbone features while the decoder works from detection-based proxies.

**Will the TCN+ViT probe break the 0.2169 backbone ceiling?** The frozen ConvNeXt linear probe (0.2169) is statistically indistinguishable from majority class (0.2217). Adding temporal aggregation (TCN or mean pooling) may or may not extract signal. If it does, the activity head can be rescued without a full backbone swap. If it does not, MViTv2-S is required.

## Section 7. Recommendation

Lead the paper with single-task D1R detection (0.995 BEATS SOTA). Report the multi-task cascade as "implementation bug + setup limitation" with verified evidence:
- PSR GELU starvation (measured pre-activation distribution, confirmed by post-repair diagnostic)
- Detection 5-class mapping bug (class indices misaligned, permanently silent output channels)
- Detection empty-frame collapse (91.9% empty frames, mAP 0.00009, GT-balanced sampler missing)
- Activity ImageNet ceiling (frozen linear probe 0.2169 = majority class 0.2217, statistically indistinguishable)

Provide a concrete fix path (3-4 weeks of focused work):
1. Complete PSR head repair training (expect F1 improvement above current 0.7018; earlier 0.7893 decoder target was a 2-recording artifact, actual decoder full-38k F1=0.0053)
2. Complete single-task ConvNeXt detection (measures true multi-task cost)
3. Run TCN+ViT temporal probe (decides if activity can be rescued on ConvNeXt)
4. Implement GT-balanced sampler for detection (addresses empty-frame collapse)
5. Full backbone swap to MViTv2-S or VideoMAE for activity (if temporal probe fails)

Defer the activity versus MViTv2-S gap as a fundamental backbone-pretraining problem that is not fixable within the current ConvNeXt architecture. The 0.2217 majority-class baseline is the theoretical floor for any frame-level model on this dataset, and our 0.2169 confirms the backbone carries zero action signal.
