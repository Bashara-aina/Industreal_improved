# Section 7: Eval Pipeline Debate

## 7.1 Class Mapping: Are YOLOv8m outputs 0-indexed or 1-indexed?

**Reviewer: Class Mapping Reviewer**

**Reference files:**
- eval_yolov8m.py (lines 191-194, 340-341)
- eval_yolov8m_psr.py (lines 76-88, PSR_MASK builder)
- config.py (lines 200-230, DET_CLASS_NAMES)
- SOTA_STATUS.md (line 10, D1 mAP=0.0004)

**Side A (the claim is correct; 0-indexed alignment is consistent):**

The YOLOv8m model at the ultralytics API level always outputs 0-indexed class IDs regardless of whether the training data annotations were 1-indexed. The Ultralytics YOLO trainer internally decrements COCO-format labels during dataset loading. The Config's DET_CLASS_NAMES dictionary uses 1-indexed keys solely for human readability (the binary strings match the ASD paper) and is never fed into the model. The PSR_MASK builder's `one_idx - 1` conversion (line 77) correctly translates this lookup. The D1 mAP of 0.0004 has a different root cause — likely the RGB-BGR channel issue or a weight loading problem for the D1 split — because the ASD eval within the same script produces 0.995 mAP50. A class index shift would affect both D1 and ASD equally since they share the same 24-class taxonomy.

**Side B (the claim is wrong; a 1-indexed mismatch explains the 0.0004 mAP):**

The D1 detection mAP of 0.0004 is not a statistical anomaly — it is exactly what you would expect from a complete class-label mismatch. With 24 classes, if every predicted class is off by 1, the expected IoU-based mAP at 0.5 for random label assignments is near zero. The ASD benchmark produces 0.995 because it may use a different annotation format or because its evaluation handles the 0-1 offset differently. The evidence from the PSR_MASK builder is telling: the explicit `one_idx - 1` conversion was added exactly because the config stores 1-indexed keys, and if someone forgot to apply this conversion consistently, class IDs 1-24 would be compared against YOLOv8's 0-23 output, yielding ID 24 out of range and everything else shifted by 1. The comment at eval_yolov8m.py line 340 asserting "no shift needed" is dangerously trusting. The safest fix would be to add a brute-force verification: swap each predicted class by +1 and re-run mAP to see if 0.0004 jumps to ~0.995.

**Resolution quality:** The most useful test would be to print the first 200 predicted labels alongside the matching GT labels side-by-side for one full batch. If the histogram of predicted labels is a right-shifted-by-1 version of the GT histogram, the bug is confirmed. If they overlap, look elsewhere.

---

## 7.2 Per-Component Threshold: Oracle bound or reportable metric?

**Reviewer: Threshold Sweep Reviewer**

**Reference files:**
- psr_optimal_thresholds.py (lines 85-108, threshold sweep)
- sweep_psr_threshold.py (lines 225-245, per-component sweep)
- SOTA_STATUS.md (lines 16-17, 20-35 per-comp breakdown)

**Side A (reportable but should be qualified as "selected on val"):**

Per-component threshold tuning on the validation set is standard practice in multi-label classification. The search grid is coarse (19 values spaced 0.05 apart) for 11 independent components, and F1 at each threshold is computed over thousands of frames — the law of large numbers means noise is small. The improvement from 0.7217 (global 0.10) to 0.7499 (per-comp optimal) is consistent across components: components with extreme prevalence (comp 0 at 100% positive, comp 4 at 14% positive) get extreme thresholds (0.05 and 0.80 respectively), which is not overfitting but correct calibration. The 5k-subset result of 0.7810 is actually higher than the full 0.7499, which is inconsistent with an overfitting story (overfitting to the full set would give higher F1 on full than subset, not lower). The per-component thresholds are fair to report as "optimal per-component thresholds" as long as the methodology is disclosed.

**Side B (this is validation set overfitting and should be called an oracle bound):**

Any parameter tuned to maximize F1 on the data used to report F1 is overfit by definition. With 11 components, each sweeping 19 thresholds, the effective degrees of freedom are larger than they appear because threshold selections interact — a threshold shift on component 3 changes which frames are counted as "fully assembled" and affects later metrics like transition F1. The inconsistency in the 5k-subset result (0.7810 > 0.7499) actually _supports_ the overfitting hypothesis: the 5k subset thresholds overfit to a noisier sample, producing an inflated estimate that then falls when evaluated on the full set. The correct methodology is either to use cross-validation (train thresholds on 4/5 of recordings, test on 1/5) or to hold out a separate threshold-tuning split. The current macro F1 should be labeled as "oracle per-component F1" — an upper bound on what real performance could be — not as a reportable metric.

**Resolution quality:** The most informative test is recording-level cross-validation. Leave one recording out, compute optimal thresholds on the remaining recordings, and evaluate on the held-out recording. If the cross-validated F1 is within 0.01 of 0.7499, the result is robust. If it drops below 0.73, the overfitting concern is real.

---

## 7.3 Crash Recovery: Is save-interval mandatory for batch eval scripts?

**Reviewer: Crash Recovery Reviewer**

**Reference files:**
- eval_yolov8m.py (lines 321-366, no save logic)
- eval_yolov8m_psr.py (lines 385-437, no save logic)
- eval_activity_clip.py (lines 58, 92-99, save_interval=5000)

**Side A (save-interval is unnecessary overhead for fast deterministic evals):**

A YOLOv8m eval on 1000 batches at batch_size=16 with a modern GPU completes in well under two hours. The inference loop is deterministic (shuffle=False on the DataLoader), so a crash merely means you restart from batch 0 — no data is lost or corrupted because you haven't accumulated anything beyond in-memory lists. Adding pickle serialization every N batches introduces I/O latency on every save interval, adds disk pressure (duplicating the full accumulated state each time), and creates a maintenance burden (checkpoint cleanup, stale file detection). The eval scripts are research tools, not production pipelines — the developer is expected to be present during the ~2-hour run or to use a job scheduler with preemption handling (e.g., SLURM's --signal for SIGUSR1-based checkpointing). The eval_activity_clip.py pattern with hardcoded save_interval=5000 is not clearly superior — its serialization format (pickle) is fragile across Python version changes and its I/O cost was never benchmarked.

**Side B (no persistence in a multi-hour eval is negligent for shared infrastructure):**

The absence of any intermediate state saving in a multi-hour evaluation loop is indefensible for code running on shared or preemptible infrastructure. A single OOM kill at batch 990 wastes ~2 hours of GPU time, which is costly and disruptive. The argument that "the developer will be present" contradicts the research reality where evals are launched and checked hours later. The I/O cost of writing a few hundred KB of accumulated boxes/scores/labels every 100 batches is negligible compared to the inference cost of 100 batches. The eval_activity_clip.py pattern is exactly right — it saved someone hours when they added it. Even a minimal approach — appending results to a JSON-lines file after each batch, or using a simple SQLite database — would provide crash recovery at near-zero I/O cost. The argument about pickle fragility is a straw man: deterministic eval outputs can be stored as JSON or compressed numpy arrays, both of which are version-independent.

**Resolution quality:** A pragmatic middle ground: add a `--save-every N` flag to both eval scripts, defaulting to 0 (no save) for backwards compatibility, but letting users opt into periodic checkpointing. The eval_activity_clip.py pattern should be replicated with JSON-serializable intermediate files rather than pickle.
