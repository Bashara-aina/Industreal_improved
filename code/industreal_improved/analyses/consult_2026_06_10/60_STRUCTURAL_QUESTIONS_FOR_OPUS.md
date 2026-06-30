# 60: 31 Structural Questions for Opus [2026-06-30]

## Questions About the Activity Head Collapse (1-12)

### Q1: Is the ViT+TCN in activity_head fundamentally broken for single-frame input?
The activity head contains a 2-layer ViT (with pos_embed for 1024 positions) and a TCN
(depthwise conv with kernel_size=5). In non-staged mode (RF4-RF10), sequence_length=1.
Does the ViT produce meaningful attention when only 1 token is present with full pos_embed?
Does the TCN's depthwise_conv produce non-zero gradients for a 1-element input?

### Q2: Does feature_bank detach gradients from proj_feat?
The gradient path is: proj_feat → feature_bank(video_ids) → bank_output → activity_head.
If feature_bank.detach_grad_entries_only=True (config line 1267), does this mean
the gradient is severed before reaching proj_features and thus c5_mod_blend?

### Q3: Is the NaN guard at model.py:2117 killing gradients silently?
```python
if not torch.isfinite(proj_feat).all():
    proj_feat = torch.zeros_like(proj_feat)
```
If proj_feat contains NaN on ANY frame, ALL frames get zeroed. With 512 components,
the probability of at least one being NaN is non-negligible at initialization.
A zero feature then passes through ViT+TCN and the gradient of "zero in → zero out"
with LayerNorm is pathologically small.

### Q4: Why is the gradient norm 0.010 across ALL config variations?
This is the most concerning observation. We changed LR (0.5x to 20x), blend ratio
(0.05 to 1.00), clip (0.3 to 1.0 to disabled), and even reinitialized the classifier.
The LIVENESS_GRAD at step=0 reports activity_head=0.010 in EVERY case.
Is this the expected gradient norm of a CE loss with constant predictions?

### Q5: Would removing ViT+TCN and using a simple Linear(512→75) classifier fix the collapse?
Proposal:
```python
activity_logits = self.classifier(self.proj_features(activity_proj))
```
This reduces activity params from 8.2M to ~0.5M, eliminates the 7-layer gradient chain,
and the gradient would flow directly from CE to classifier to proj_features to c5_mod.

### Q6: Is the long-tail data distribution (46/72 classes with <1% frames) fundamentally incompatible with standard CE + label smoothing?
With CB_BETA=0.99 and CB_LABEL_SMOOTHING=0.1, the class weights are computed as:
```python
_eff_num = (1.0 - 0.99^counts) / 0.01
_weights = 1.0 / _eff_num
_weights[0] = 0.0  # NA class
_weights /= _weights.sum() * len(_weights)
```
For a class with 1 frame: _eff_num = (1 - 0.99^1) / 0.01 = (1 - 0.99) / 0.01 = 1.0
_weights = 1.0. For a class with 404 frames: _eff_num = (1 - 0.99^404) / 0.01 ≈ 98.2
_weights = 0.010. The weight ratio is 100:1 between rarest and most common classes.
Does this extreme weight ratio create gradient instability?

### Q7: Should we replace CE + label smoothing with focal loss for activity?
Focal loss naturally handles class imbalance by down-weighting easy examples.
```python
FocalLoss(alpha=0.25, gamma=2.0)
```
This is what we use for detection and PSR. Activity is the only head using CE.

### Q8: Would training the activity head in ISOLATION (freeze all other heads) tell us if the architecture can learn at all?
If we freeze backbone, detection, pose, PSR and train only activity_head for 2 epochs,
we could determine whether the collapse is from multi-task interference or a fundamental
architecture issue. If even isolated training shows collapse, the architecture is the problem.

### Q9: Is ACTIVITY_LOSS_CAP=80.0 protecting NaN but also killing gradient?
```python
loss_act = torch.clamp(loss_act, max=C.ACTIVITY_LOSS_CAP)
```
When act_macro_f1=0.0 (random), what is the actual CE loss value?
If it's ~4.3 (ln 75), the cap of 80 is irrelevant. But if CB weights push rare-class
losses to 80+, clamping creates zero gradient for those classes.

### Q10: Could the 'class 12' (put_instruction) dominance be a data label error?
put_instruction has 100% frame accuracy in EVERY run's EVAL COLLAPSE report.
If the validation data has mislabeled frames where the correct activity is always
assigned class 12, the model would correctly predict class 12 for everything.
How many val frames are labeled as put_instruction?

### Q11: Does ACTIVITY_GRAD_BLEND_RATIO=1.0 actually flow gradient to the backbone?
The formula: `c5_mod_blend = blend * c5_mod + (1-blend) * c5_mod.detach()`
With blend=1.0: `c5_mod_blend = 1.0 * c5_mod + 0.0 * c5_mod.detach() = c5_mod`
In theory this is full gradient flow. But in practice, does PyTorch optimize away the
0.0 * detach() term? Does the computation graph keep the gradient path intact?

### Q12: Should we set ACTIVITY_HEAD_GRAD_CLIP=0 (disabled) instead of 1.0?
The gradient norm is 0.010. A clip threshold of 1.0 is 100x larger than the actual norm.
This means clip_grad_norm_ does nothing — it's a no-op that still iterates all 8.2M
params to compute norms. Would disabling it entirely (check `if _act_gc > 0`) improve
the gradient path by removing the iteration overhead?

## Questions About Multi-Task Architecture (13-19)

### Q13: Should the PSR_SEQ_LOSS_SCALE be increased to stabilize PSR oscillation?
PSR cycles between ALIVE and DEAD. The per-component heads stay alive. If we increase
the sequence loss scale, would the total PSR gradient stabilize?

### Q14: Is DETACH_PSR_FPN=True (config line 1282) hurting PSR?
```python
'detach_psr_fpn': True  # RF4 config
```
If the PSR gradient is detached from FPN, it flows through a separate path
(GAP of P3/P4 features). This may cause the PSR gradient to be inconsistent
with the detection/pose gradient, creating the oscillation.

### Q15: Should DETACH_REG_FPN be True for RF4?
It's True for all non-reinit stages. The reg-loss warmup (REINIT_REG_WARMUP_STEPS)
was designed for RF1's detection-only stage. In RF4 with all heads active,
does detaching reg gradient from FPN help or hurt?

### Q16: Is the FPN gradient path correct for all 5 heads?
The gradient flows from each head through its specific path to the FPN features:
- Detection: regression subnet → classification subnet → P3-P7 FPN levels
- Pose: regressed keypoints → P3-P7 FPN levels  
- Head pose: FiLM modulation of c5_mod
- Activity: c5_mod_blend → proj_features → ViT → TCN → classifier
- PSR: det_psr_fpn → transformer → per-component classifiers

Are any of these paths double-counting (heads sharing the same FPN level) or
zero-counting (head's gradient detached from all FPN levels)?

### Q17: Can we use gradient surgery (PCGrad, CAGrad) to resolve gradient conflicts?
With 5 heads producing gradients of vastly different magnitudes (312:1 ratio),
standard gradient clipping and Kendall weighting cannot resolve the imbalance.
Gradient surgery methods explicitly resolve conflicting gradient directions
between tasks.

### Q18: Should we implement Nash-MTL or similar bargaining approach?
Nash-MTL treats multi-task optimization as bargaining between tasks. It finds
a gradient direction that benefits all tasks simultaneously. This directly
addresses the gradient starvation problem better than Kendall weights.

### Q19: Is the backbone capacity (ConvNeXt-Tiny, 28.6M params) sufficient for 5 tasks?
YOLOv8m uses 25M params for detection only. We have 5 tasks sharing 28.6M backbone
params. Is the bottleneck too small for the representational demands of 5 diverse tasks?

## Questions About Infrastructure (20-25)

### Q20: Should we implement DistributedDataParallel for dual GPU training?
GPU 0 (RTX 3060) is completely idle while GPU 1 (RTX 5060 Ti) trains at 100%.
DDP with 2 GPUs would give ~2.0 batch/s throughput. Is this worth the engineering
effort given the activity collapse problem?

### Q21: Can evaluate_all be rewritten to avoid CUDA kernel hangs?
The ThreadPoolExecutor approach cannot interrupt CUDA kernels. Options:
(a) Process each validation batch with `torch.cuda.synchronize()` between batches
    so a hang is detected at batch granularity, not at full-eval granularity.
(b) Use CUDA events to check if kernel has completed after timeout.
(c) Fork a subprocess for validation and kill it if it hangs.

### Q22: Should we move IndustReal dataset to SSD?
Current HDD gives 1.2 batch/s with NUM_WORKERS=0. SSD would likely give 2-3 batch/s
and eliminate the Dataloader bottleneck. But the dataset is only 3,667 frames ×
480×360×3 ≈ 1.8GB — small enough to fit in RAM (we have 32GB available).

### Q23: Should we load the entire dataset into RAM?
With RAM_CACHE_MAX_IMAGES=5000 (~1.8GB), the frame cache can hold all 3,667 frames.
Is the cache fully utilized? The log shows "CPU RAM: avail=20.9GB, buffers=563GB,
cached=18384GB". The kernel is aggressively caching file data. But the Python-level
FRAME_CACHE dict may not be filling because the preloader threads are disabled
with NUM_WORKERS=0.

### Q24: Should crash_recovery.pth be loaded automatically on restart?
Currently, resume always loads best.pth (or latest.pth). crash_recovery.pth is only
loaded manually with --resume crash_recovery.pth. If we auto-detect crash_recovery.pth
with a newer timestamp than best.pth and load it, we save 100 steps of re-training per crash.

### Q25: Should we validate on every epoch or every N epochs?
Current: VAL_EVERY=1 (every epoch, 200 batches gate-only). Full eval every epoch.
Risk: validation CUDA hang kills training progress.
If VAL_EVERY=3: 3x fewer validation attempts → 3x lower crash probability.
But we lose the per-epoch metric signal needed to detect collapse.

## Questions About Paper Targets (26-31)

### Q26: Can we meet the AHFE paper targets with the current architecture?
Paper target act_top1=0.375. Current act_top1=0.0. After 10 days of effort.
Is this gap bridgeable? What would it take?

### Q27: Are the paper targets realistic given our compute budget?
We have 1 GPU (effectively, since GPU 0 is idle), HDD storage, and 2 weeks until
AHFE deadline. At 1.2 batch/s × ~48 min/epoch × 68 epochs for RF4-RF10 = ~55 hours
of training. But we've never completed 3 consecutive epochs without a crash.
Is this schedule achievable?

### Q28: Should we reduce the scope of the paper?
Current claim: "first consumer-GPU multi-task assembly verification system."
If we can only achieve detection + pose, the paper scope narrows significantly.
Would AHFE still accept a paper with (detection + head pose) only, without
activity recognition and PSR?

### Q29: Is the issue that we're comparing against single-task SOTA?
YOLOv8m for det: 0.838 mAP50 on 24 classes + time. We're at 0.053 with 5 tasks.
MViTv2 for act: 0.653 top1. We're at 0.0.
Is multi-task learning on a consumer GPU inherently 5-10x worse than single-task SOTA,
and the paper should SET THIS BASELINE rather than trying to match single-task numbers?

### Q30: Should we switch to a simpler architecture for the paper deadline?
Instead of the current ConvNeXt-Tiny + FPN + 5 specialized heads (53M params),
could we use a single ViT-B backbone with task-specific heads (e.g., linear probes)?
This would reduce engineering complexity but potentially hurt performance. Is the
tradeoff worth it for deadline feasibility?

### Q31: What is the minimum viable system for the AHFE paper?
If we can achieve:
- Detection: 0.30 mAP50 (localizes assembly states)
- Head pose: 10° angular MAE (tracks operator's gaze region)
- Activity: Fails completely
- PSR: Fails completely

Does the paper still make a contribution? The claim would be "consumer GPU real-time
multi-task monitoring" but with only 2/5 tasks working. Is that acceptable for AHFE?
