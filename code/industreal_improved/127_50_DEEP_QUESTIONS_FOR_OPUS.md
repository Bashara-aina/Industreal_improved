# Deep Questions for Opus — Eval Pipeline

## Q1: Class index alignment — is there really a 0-vs-1 index mismatch, and can mAP=0.0004 be explained by a single -1 bug?

**The claim:** eval_yolov8m.py line 341 asserts "Both YOLOv8 and the dataset's gt_classes are 0-indexed (dataset converts COCO 1-indexed -> 0-indexed via -1). No shift needed." Yet D1 full eval mAP50 is 0.0004, which the SOTA_STATUS flags as "broken — class mapping needs verification."

**Why this matters:** If the ground truth labels are 1-indexed (1..24) and predictions are 0-indexed (0..23), then no predicted box will ever match a ground truth of the same class, producing mAP ~0. If the dataset converter is applying -1 correctly, the bug is elsewhere and time spent on class mapping is wasted.

**Evidence:**
- DET_CLASS_NAMES at config.py:202 uses 1-indexed keys (1='background' ... 24='error_state') — human-readable but ambiguous about model output indexing.
- eval_yolov8m.py:340-341 comment claims 0-indexed alignment, no shift needed.
- PSR_MASK builder in eval_yolov8m_psr.py:77 does `zero_idx = one_idx - 1` — implying the model builder believes one_idx is the native format.
- eval_yolov8m.py:191 range check expects classes in [0, NUM_DET_CLASSES) = [0, 24).
- D1 mAP=0.0004 vs ASD mAP50=0.995 — the 0.995 on the ASD benchmark suggests YOLOv8m works on that task, making a global class index error unlikely.

**Evidence missing:** The exact line in the dataset __getitem__ where COCO label IDs are decremented is not in the evaluation files — it lives in `src/data/industreal_dataset.py`. Whether this -1 is applied to all label paths (detection, psr, activity) or only detection is unclear. Also missing: a direct histogram comparison of predicted class IDs vs GT class IDs on the D1 split to confirm whether the distributions overlap at all.

## Q2: RGB vs BGR channel order — does the eval pipeline match the training pipeline?

**The claim:** The dataset loads images via PIL `Image.open().convert('RGB')` (confirmed in industreal_dataset.py lines 885, 1005, 1066, 1081, 1137), producing RGB-ordered tensors stored as `images['rgb']`. The eval code converts these to BGR via `[:,:,::-1]` before passing to YOLOv8 (eval_yolov8m.py lines 168, 330).

**Why this matters:** If the IndustReal YOLOv8m model was fine-tuned on RGB inputs (no channel inversion), the eval code is silently feeding BGR data to an RGB-trained model. This would systematically degrade detection confidence across all classes — potentially explaining D1 mAP=0.0004 while ASD mAP=0.995 (ASD may use a different data pipeline). If the model expects BGR (consistent with YOLOv8 COCO pretraining), there is no bug.

**Evidence:**
- IndustRealDataset stores images as `images['rgb']` from PIL (RGB convention).
- eval_yolov8m.py:168 does `[:,:,::-1]` with comment "BGR for YOLOv8".
- eval_yolov8m_psr.py:395 same RGB->BGR conversion.
- The training pipeline in the training directory is not read here — we don't know whether it also does `[:,:,::-1]`.

**Evidence missing:** The training pipeline's data loading path (likely `src/training/train.py` or a Lightning datamodule). We need to confirm whether the training augmentations include an RGB-to-BGR conversion before feeding the model. Also missing: a simple unit test that passes a known RGB image through the full model pipeline and checks the output channels.

## Q3: No save-interval for crash recovery in eval_yolov8m.py and eval_yolov8m_psr.py — why was this omitted when eval_activity_clip.py has it?

**The claim:** Both YOLOv8 eval scripts process the full validation set (1000+ batches, ~2 hours estimated) without any intermediate state persistence. A crash at batch 990/1000 loses all progress. eval_activity_clip.py has a hardcoded save_interval=5000 that dumps pickle checkpoints. This is an inconsistency in the evaluation infrastructure.

**Why this matters:** On shared GPU infrastructure (the task references RTX 5060 Ti), eval jobs can be preempted, run out of memory, or encounter transient hardware errors. Without incremental save, a 2-hour run is a single point of failure. The cost of crash recovery is high in both wall time and developer attention.

**Evidence:**
- eval_yolov8m.py:321-366 — main inference loop, no persistence beyond the final json.dump at line 476.
- eval_yolov8m_psr.py:385-437 — same pattern, no intermediate saves.
- eval_activity_clip.py:58 `save_interval = 5000`, with pickle checkpoint at lines 92-99 every 5000 frames.
- sweep_psr_threshold.py:166-168 — caches data to `psr_data_cache.pt` after collection (different pattern but still crash-resilient).

**Evidence missing:** No performance benchmark of pickle serialization overhead at different intervals. The optimal save frequency (tradeoff between I/O cost and crash recovery cost) is unknown. Also missing: whether the eval scripts were designed for a specific runtime environment (e.g., SLURM with job arrays) where individual batch jobs are already atomic.

## Q4: Per-component threshold selection on the validation set — how much F1 improvement is real vs overfitting to val noise?

**The claim:** psr_optimal_thresholds.py sweeps 19 thresholds (0.05 to 0.95 step 0.05) per component on the validation set and reports the resulting macro F1 as the "optimal" score. The improvement from global 0.10 (0.7217) to per-comp optimal (0.7499) is attributed to better per-component calibration. A similar sweep in sweep_psr_threshold.py uses an even finer grid.

**Why this matters:** Choosing per-component thresholds that maximize F1 on the same data used for metric reporting is a form of validation set overfitting. With 11 components and 19 candidate thresholds each, the search space is large enough to find thresholds that happen to perform well on val noise. The true generalization gap can only be measured on a held-out test set.

**Evidence:**
- psr_optimal_thresholds.py:85-104 — per-component sweep on all val frames.
- psr_optimal_thresholds.py:108 — reports the aggregated macro F1 using optimally chosen thresholds.
- psr_optimal_thresholds.py:113-126 — also reports global thresh=0.10 for comparison (0.7217 vs 0.7499).
- SOTA_STATUS.md:17 shows per-comp F1 values ranging from 0.3455 (comp 4) to 1.0000 (comp 0) — components with extreme class imbalance (comp 4: 0.142 gt_pos_frac, comp 0: 1.000 gt_pos_frac) get extreme thresholds that maximize F1 on val but may not generalize.
- sweep_psr_threshold.py:213-220 — global sweep, best thresh selected on same val data.

**Evidence missing:** A held-out threshold-tuning set (e.g., 20% of val frames) or a proper test set with fixed thresholds. Cross-validation of threshold F1 across recordings (recording-level fold) would measure generalization. The 0.7810 figure for the "5k subset" (SOTA_STATUS.md:17) hints at subsampling but it's unclear whether frames or recordings were subsampled.

## Q5: Clip-level activity evaluation — does majority-vote over per-frame predictions measure anything meaningful?

**The claim:** eval_activity_clip.py builds 16-frame clips with stride 8, assigns each clip the majority-vote label and prediction, and reports clip-level top-1 accuracy of 0.028. The SOTA_STATUS acknowledges this is "broken — per-frame MLP can't do temporal reasoning" and compares against MViTv2-S at 0.622.

**Why this matters:** The comparison to 0.622 (MViTv2-S) is misleading because the underlying model architectures are fundamentally different — ConvNeXt-Tiny + per-frame MLP vs a video transformer that processes spatiotemporal features. The 0.028 clip accuracy is essentially the per-frame accuracy with majority smoothing, which may actually be _lower_ than per-frame accuracy if clips aggregate conflicting frame-level predictions. If clip-level eval is meant to measure temporal reasoning, it requires a temporally-aware model first.

**Evidence:**
- eval_activity_clip.py:119-137 — sliding window clips with majority voting.
- eval_activity_clip.py:74 uses model outputs `act_logits` — the per-frame head, no temporal layer.
- model._seq_len is set to clip_length (line 44), but this only controls how many frames are stacked, not how they're fused — ConvNeXt backbone processes each frame independently if no temporal fusion is added.
- SOTA_STATUS.md:12 — explicitly states the model lacks temporal reasoning.

**Evidence missing:** The per-frame accuracy (without clip aggregation) is not reported, so we can't tell whether clip-level aggregation improves or degrades accuracy. Also missing: comparison against simple baselines (always-predict-majority-class, random). The verb-group remapping at lines 115-117 applies to labels but not predictions (the model outputs raw class IDs, not remapped ones), which may systematically misalign the comparison.
