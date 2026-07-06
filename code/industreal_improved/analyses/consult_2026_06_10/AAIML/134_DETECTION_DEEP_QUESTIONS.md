# 134 — Detection: 50 Deep Questions for Opus

**Date:** 2026-07-06
**Purpose:** In-depth detection analysis to drive SOTA-beating strategy. All questions reference committed evidence files and current run logs.

---

## §0. Evidence Inventory — file paths and current numbers

### Reference files and checkpoints

| File | Path | Size | SHA256 | Key metrics |
|---|---|---|---|---|
| Freeze checkpoint (best.pth) | `src/runs/rf_stages/checkpoints/best.pth` | 738 MB | `59cb88ec85311bfcfff91f000bd08005675e3a882bec9f24ccd5ee0cbe89f9a8` | Epoch 18 multi-task ConvNeXt-Tiny (promoted epoch_18) |
| IndustReal YOLOv8m weights | `src/runs/rf_stages/checkpoints/yolov8m_industreal.pt` | 311 MB | `ed1a3000fd1cc72c86af95551afa3a6479c8dcb84e4cb47f5e6e011c14287c18` | Microsoft-published 24-class ASD weights |
| SOTA status | `src/runs/rf_stages/checkpoints/SOTA_STATUS.md` | — | — | Main reconciliation document |
| D1 v1 eval | `src/runs/rf_stages/checkpoints/d1_yolov8m/metrics.json` | — | — | mAP50=0.0004, mAP50-95=0.0004, only class 22 (11101111111) fires |
| D1 v3 eval | `src/runs/rf_stages/checkpoints/d1_yolov8m_v3/metrics.json` | — | — | mAP50=0.0004, identical class distribution |
| D3 full eval | `src/runs/rf_stages/checkpoints/d3_full_eval/metrics.json` | — | — | NO detection metrics — act_frame_acc=0.129, psr_f1=0.0, head_pose MAE=9.1 deg |
| D4 retuned sweep | `src/runs/rf_stages/checkpoints/d4_retuned/sweep_results.json` | — | — | Best F1=0.347 (hi=0.3, lo=0.1, min=2) |
| D4 verdict | `src/runs/rf_stages/checkpoints/d4_retuned/verdict.json` | — | — | "threshold-partial: decoder shows marginal benefit" |
| D1 v2 log | `src/runs/rf_stages/checkpoints/logs/d1_v2.log` | — | — | "Using cached IndustReal weights", mAP@0.5=0.0000 |
| D1 v3 log | `src/runs/rf_stages/checkpoints/logs/d1_v3.log` | — | — | "Using cached IndustReal weights", mAP@0.5=0.0004 |
| PSR eval log | `src/runs/rf_stages/checkpoints/logs/eval_yolov8m_psr.log` | — | — | YOLOv8m->PSR detections on <1% of frames |
| D1R train log | `runs/detect/src/runs/yolov8m_industreal/d1r/results.csv` (workstation) | — | — | mAP50=0.995 at epoch 25 |
| Opus complete answers | `analyses/consult_2026_06_10/AAIML/133_OPUS_COMPLETE_ANSWERS.md` | — | — | D-1 through D-7 verdicts, C-1 through C-7 contradictions |
| Eval script | `src/evaluation/eval_yolov8m.py` | — | — | Has FAIL HARD fix for IndustReal download |
| PSR eval script | `src/evaluation/eval_yolov8m_psr.py` | — | — | YOLOv8m->s2->MonotonicDecoder pipeline |

### Critical number reconciliation

| Number | Source | Context |
|---|---|---|
| mAP50=0.995 / mAP50-95=0.861 | D1R results.csv epoch 25 | **Separately-trained YOLOv8m** from COCO init, 25 epochs on recording-aware split. Ceiling, not our result. |
| mAP50=0.0004 | D1 v1/v3 metrics.json | Microsoft IndustReal checkpoint evaluated on our val set. Sparse detections ~0.1/frame at conf>=0.25. Only class 22 fires. |
| mAP50=0.358 | D3 subsample eval (workstation) | Multi-task ConvNeXt-Tiny detection head. **Class-balanced subsample, 250 batches.** Headline: 0.358/0.995 = 36% of ceiling = 64% cost. |
| Present-class mAP50=0.573 | Derived from 0.358 x 24/15 | COCO convention excludes zero-GT classes. With 15 present classes of 24, the corrected number is 0.573 = 58% of ceiling. |
| WACV 2024 mAP50=0.838 | Published baseline | 24-class ASD, soft baseline (1 GPU-day beats it). Comparison complicated by split sensitivity (0.2 gap). |
| D4 F1=0.000 (default) / 0.347 (retuned) | sweep_results.json | YOLOv8m -> MonotonicDecoder. Threshold-sensitive, backbone-limited. |
| Error-state FPR=0.0% | SOTA_STATUS.md §5.4 | Class 24 has 0 GT instances in entire dataset. WACV publishes 65% FPR on trained model. |
| D3 full eval | d3_full_eval/metrics.json | Contains act (0.129), pose (9.1 deg), PSR (0.0) — **no detection metrics at all** |
| Zero-GT classes | D1 metrics.json per_class_gt | Classes 1,2,3,14,15,23 have gt=0. Six classes; Opus D-4 counts nine. Discrepancy needs resolution. |

---

## §1. D1 v1-v3 audit (the mAP=0.0004 question)

### Q1. Why does the Microsoft IndustReal checkpoint (yolov8m_industreal.pt) produce only class-22 detections (binary string "11101111111") on our validation set?

**Evidence:** D1 v2 log (line 10) shows class mapping verification on 10 frames finds only class ID 13 (1-indexed = class 22, 0-indexed = class 22). D1 metrics.json (class 22 channel) shows AP=0.0102, the only class with any AP.

**Current answer (best guess):** The model has learned to detect only one assembly state — the fully-assembled product (class 22/23 in the 24-class taxonomy). Our validation set contains many frames of the complete assembly, so class 22 detections come through. Other classes representing intermediate states don't fire because the model was trained on a different data distribution.

**What we need to verify:** Load the checkpoint and run inference on a dense-sampled subset of frames. Count per-class detection frequencies at multiple confidence thresholds. Compare class-id histograms between our val set and the Microsoft-published test set (if available).

---

### Q2. Does the checkpoint binary string "11101111111" (class 22) mean the model learned the terminal assembly state, and that's all it learned?

**Evidence:** D1 v2 log (line 10): "Unique class IDs detected: [13]" (0-indexed 13 = 1-indexed 14 = "11110111111"? Wait — need to verify mapping). D1 per_class_gt shows class 22 (11101111111) has 378 GT instances, one of the highest counts in the dataset.

**Current answer (best guess):** Yes, the model has collapsed to the most frequent class. With 378 GT instances and AP=0.0102, it barely detects even this class. The signal is almost entirely absent, which is why mAP50=0.0004.

**What we need to verify:** Cross-reference the 0-indexed class IDs in the log with the 1-indexed DET_CLASS_NAMES mapping. The log says class [13] but DET_CLASS_NAMES are 1-indexed. Confirm which class name "13" maps to.

---

### Q3. The D1 log says "Using cached IndustReal weights" — is there any chance the cached .pt file got corrupted between download and evaluation?

**Evidence:** D1 v2 log (line 2) and D1 v3 log (line 2): "Using cached IndustReal weights: src/runs/rf_stages/checkpoints/yolov8m_industreal.pt". The file is 311 MB (verified). SHA256 is `ed1a3000...`, which differs from the specified best.pth SHA (`59cb88ec...`).

**Current answer (best guess):** Low probability. The file loads successfully, class mapping verification passes (classes within [0,24)), and inference produces valid detection outputs. A corrupted file would likely fail to load or produce garbage coordinates.

**What we need to verify:** Compare the cached SHA against the Microsoft-published SHA (if available). Re-download from source and re-run D1. The Microsoft URL currently returns 404 (SOTA_STATUS.md line 153), so we'd need the file from another source.

---

### Q4. Only class 22 has AP>0 (AP=0.0102) — why class 22 specifically? What distinguishes it from the other 23 ASD classes?

**Evidence:** D1 metrics.json per_class_ap (class 22 column): AP=0.010252846658376613. All others: 0.0. D1 metrics.json per_class_gt shows class 22 has 378 GT instances (one of the highest counts). The class binary string is "11101111111" (11 bits).

**Current answer (best guess):** Class 22 ("11101111111") represents the near-terminal assembly state where all components except comp 4 (in the 0-indexed 11-bit scheme) are assembled. The fully assembled state would be class 23 ("11111111111" or similar). Our validation set contains many frames in this state. The model learned to fire for a high-prevalence class but learned nothing about earlier assembly stages.

**What we need to verify:** Check per-class prevalence in the training split. If class 22 dominates the training set, the model may have converged to a majority-class solution.

---

### Q5. Is the issue that the Microsoft checkpoint was trained on industrial static images while our validation set is egocentric video frames? What's the domain gap?

**Evidence:** SOTA_STATUS.md (D1 integrity verdict, lines 157-158): "Root cause hypothesis: The model binary strings ... match our DET_CLASS_NAMES in config.py, but the model was trained on a different dataset split or with different preprocessing (/shared/nl011006/... path in overrides)."

**Current answer (best guess):** The IndustReal dataset likely contains both static and assembly images. The published checkpoint was trained on a specific data split that differs from our validation set. The path `/shared/nl011006/...` references a cluster mount point, suggesting the training data layout is institution-specific.

**What we need to verify:** Check the original IndustReal paper for training data description. Compare their training/validation split protocol against ours. Run a small fine-tuning experiment from the checkpoint on our data.

---

### Q6. D1 v2 produced mAP@0.5=0.0000 while D1 v3 produced mAP@0.5=0.0004 — what changed between runs to produce this difference?

**Evidence:** D1 v2 log (line 252): "mAP@0.5: 0.0000  mAP@[0.5:0.95]: 0.0000". D1 v3 log (line 252): "mAP@0.5: 0.0004  mAP@[0.5:0.95]: 0.0004". Both use the same weights file. Both use batch_size=16.

**Current answer (best guess):** The v2 run got 0.0000 because of floating-point noise — no detections met the confidence threshold for any class, resulting in zero AP everywhere. The v3 run produced a tiny AP (0.0004) because class 22 (11101111111) had one or two detections that happened to align with GT boxes at some IoU threshold. The difference is stochastic (GPU nondeterminism, batch ordering).

**What we need to verify:** Compare the metrics.json files for per-class AP. If v2 has all zeros and v3 has class 22 at 0.0102, the stochastic hypothesis is confirmed.

---

### Q7. How does the class mapping verification pass (classes in [0,24)) when the model only predicts 1 class (class 13 = class 22)? Doesn't this suggest the model loaded correctly but is failing at inference?

**Evidence:** D1 v2 log (line 11): "Class mapping OK: detected classes in [13, 14) within [0, 24)". The verification loop runs on 10 frames and finds only class ID 13 (0-indexed).

**Current answer (best guess):** The class mapping check is too weak — it only verifies that class IDs are within [0, 24), not that the model can detect all 24 classes. The check is designed to catch the COCO-fallback bug (80 classes), but it cannot catch the "model detects only one class" problem.

**What we need to verify:** Add a per-class detection count to the verification function. A healthy model should detect at least 8-10 distinct classes in 10 frames.

---

### Q8. What would the mAP be if we re-ran D1 with the confidence threshold lowered from 0.25 (default) to 0.01?

**Evidence:** The default YOLOv8 inference uses `conf=0.25`. The D4 PSR eval uses `detection_thresh=0.1` and still gets detections on <1% of frames (eval_yolov8m_psr.log pattern: mostly "0 total detections" with occasional "16 total detections" in bursts).

**Current answer (best guess):** Lowering the threshold to 0.01 would produce many low-confidence detections, but the mAP would remain near-zero because the model's class predictions are essentially random for non-dominant classes. The model produces sparse detections because its learned confidence estimates are low across all classes (the model is uncertain about everything on our domain).

**What we need to verify:** Re-run with conf=0.01, note the detection count and per-class AP distribution. If mAP rises significantly, the issue is confidence calibration, not absent learning.

---

### Q9. How many GT instances exist for each of the 24 classes in the validation set, and does the detection rate correlate with GT prevalence?

**Evidence:** D1 metrics.json per_class_gt shows wide variance: class 12 (11110111101) has 430 GT, class 4 (10010110000) has 324, class 0 (background) has 331, class 22 (11101111111) has 378. But also: class 1 (10000000000) has 0, class 2 (10010010000) has 0, class 3 (10010100000) has 0, class 14 (11110101111) has 0, class 15 (11110011111) has 0, class 23 (error_state) has 0.

**Current answer (best guess):** The detection rate does NOT correlate simply with GT prevalence. Class 12 has 430 GT but AP=0.0. The model detects only class 22 (378 GT). The difference is likely the visual distinctiveness of the terminal assembly state vs earlier states.

**What we need to verify:** Compute per-class GT counts for both train and val splits. Check if class 22 also dominates the training distribution.

---

### Q10. If the Microsoft IndustReal checkpoint is unusable for our val set, what does this say about our recording-aware split vs the original IndustReal split?

**Evidence:** SOTA_STATUS.md (D1 integrity verdict, line 157): "the model was trained on a different dataset split or with different preprocessing." The D1R fine-tuned model (COCO init, 25 epochs, our split) achieves mAP50=0.995.

**Current answer (best guess):** The recording-aware split (grouping frames by recording before splitting) creates a harder generalization problem than the original random split. Microsoft's checkpoint, trained on random-split data, fails to generalize to unseen recordings. The 0.995 from D1R shows that YOLOv8m can learn our split from scratch — the checkpoint's failure is a generalization gap, not a model capacity issue.

**What we need to verify:** Compare the WACV 0.838 numbers (their split, their eval) with the D1R 0.995 (our split, our eval). The WACV number on random-split is 0.838; our random-split D1R would likely exceed this.

---

## §2. D1R fine-tuned (the mAP=0.995 ceiling)

### Q11. Is mAP50=0.995 at epoch 25 a genuine convergence or a data contamination artifact from the recording-aware split?

**Evidence:** D1R results.csv (workstation only, not in repo): epoch 25 mAP50=0.995. The D1R training run initializes from COCO-pretrained YOLOv8m and trains on our recording-aware split for 25 epochs.

**Current answer (best guess):** 0.995 is genuine and reflects the ceiling that a dedicated single-task detector can reach on this dataset. The recording-aware split means train and val have different recordings, so data contamination is unlikely. The 0.995 suggests the detection task on this dataset is relatively easy for a modern detector.

**What we need to verify:** Commit the D1R results.csv and train log to the repo. Confirm epoch-25 convergence curve shows saturation. Check mAP50-95=0.861 as a harder sanity check.

---

### Q12. What is the mAP50-95 at epoch 25 (0.861) telling us about the quality of detections vs the mAP50 (0.995)?

**Evidence:** SOTA_STATUS.md line 11: "0.995 / 0.861" for mAP50 / mAP50-95. D1R results.csv at epoch 25.

**Current answer (best guess):** The 0.861 mAP50-95 is very high, indicating that not only are detections present at IoU=0.5, they are well-localized (good IoU overlap up to 0.95). This confirms the detection is high-quality, not just barely-there. The gap 0.995-0.861=0.134 is the IoU sensitivity budget — approximately 13% of mAP is lost when tightening IoU from 0.5 to 0.95.

**What we need to verify:** Compare the mAP50-95 gap with WACV's published mAP50-95 (if available). Is 0.861 near SOTA for mAP50-95 as well?

---

### Q13. Does the D1R model (YOLOv8m from COCO init, our split) actually generalize to held-out recordings, or is it overfitting to split-specific visual patterns?

**Evidence:** Recording-aware split (frames grouped by recording before splitting) ensures no recording appears in both train and val. D1R achieves 0.995 on val.

**Current answer (best guess):** The 0.995 is unlikely to be pure overfitting on a recording-aware split. The detection task (locating assembly parts) is visually simple — parts have distinctive shapes and colors. A YOLOv8m with adequate training should generalize across recordings of the same assembly process.

**What we need to verify:** Run a cross-recording evaluation: train on recordings 1-13, evaluate on 14-16. If mAP90 holds, overfitting is low.

---

### Q14. Is the D1R mAP50=0.995 an upper bound for the multi-task ConvNeXt detection head (D3), or could D3 theoretically exceed it?

**Evidence:** D1R is YOLOv8m (25.9M params, specialized detection head). D3 is ConvNeXt-Tiny (28.6M backbone + shared head). YOLOv8m has a task-specific architecture designed for detection; ConvNeXt has a general-purpose backbone.

**Current answer (best guess):** 0.995 is likely near the dataset ceiling, not a model-specific ceiling. Any reasonable detector should reach 0.95+ on this dataset. The multi-task model's 0.358 (or 0.573 present-class) is far below this ceiling, indicating the bottleneck is the multi-task training regime (competing gradients, shared backbone), not the backbone capacity.

**What we need to verify:** Train ConvNeXt-Tiny as a single-task detector (no activity/PSR/pose heads). If it reaches 0.90+ mAP, the bottleneck is confirmed as multi-task gradient conflict, not backbone capacity.

---

### Q15. What happens if we train D1R for 50 epochs instead of 25? Does mAP improve or start to overfit?

**Evidence:** D1R was trained for 25 epochs (from results.csv). YOLOv8m typically converges around 100-300 epochs on COCO.

**Current answer (best guess):** On a dataset of this size, 25 epochs may be insufficient for full convergence. Additional training would likely push mAP50 to 0.997+ and mAP50-95 to 0.90+. However, the improvement over 0.995 is marginal for practical purposes — the ceiling is already near-saturated.

**What we need to verify:** Check the convergence curve in results.csv. If epoch 25 has not plateaued, extend to 50 epochs.

---

### Q16. What is the per-class mAP breakdown for D1R? Are there any classes with significantly lower AP that would indicate a class imbalance issue?

**Evidence:** Not available in committed files — D1R results.csv only contains overall mAP50/mAP50-95.

**Current answer (best guess):** Classes with low GT counts (classes 1,2,3,14,15,23 with 0 GT in val) would show 0 AP regardless of model performance. Among classes with GT, the lowest AP would likely be on rare visual patterns (e.g., partially assembled states that resemble other states).

**What we need to verify:** Run D1R eval with per-class AP output. Compare against the zero-GT class list.

---

### Q17. If D1R achieves 0.995 from COCO init, what does this imply about the difficulty of the IndustReal detection task?

**Evidence:** 25 epochs, COCO init, recording-aware split, mAP50=0.995. Training was on a YOLOv8m, not a carefully tuned architecture.

**Current answer (best guess):** The IndustReal detection task is comparatively easy for modern detectors. The objects are large, well-lit, and visually distinct (color-coded assembly parts). The difficulty lies in the multi-task integration and the activity/PSR/pose tasks, not detection itself.

**What we need to verify:** Confirm the WACV 2024 authors also found detection easy (0.838 mAP was likely a single-run number, not the result of extensive tuning).

---

### Q18. Could the D1R YOLOv8m serve as a "detection oracle" for training the other heads? Specifically, use its features as additional input to the activity/PSR heads?

**Evidence:** D4 experiment shows that YOLOv8m->PSR decoder achieves F1=0.347 (retuned), severely limited by sparse detection rate.

**Current answer (best guess):** Using D1R's detection logits as additional input to the multi-task model's other heads would likely improve PSR (the decoder is proven effective with ConvNeXt logits at F1=0.75). However, YOLOv8m's feature space differs from ConvNeXt's, requiring a separate encoder or aligner network.

**What we need to verify:** Check if D1R produces denser detection outputs than the IndustReal checkpoint. The D4 experiment used the IndustReal checkpoint (<1% detection rate); D1R likely produces detections on 80%+ of frames.

---

### Q19. Is the D1R YOLOv8m's 0.995 mAP50 reported on the same validation set that produced 0.0004 for the IndustReal checkpoint? If so, the entire gap is training regime, not data.

**Evidence:** Both evaluations use the same IndustRealDataset(split="val") with IMG_WIDTH/IMG_HEIGHT from config.py. D1R was trained on the corresponding training split; the IndustReal checkpoint was trained on Microsoft's split.

**Current answer (best guess):** Yes, same val set. The entire mAP gap (0.0004 vs 0.995) is the difference between a model trained on a different data distribution and a model trained on our split. This confirms the OOD generalization gap is the root cause, not a model bug or data corruption.

**What we need to verify:** Confirm the same IndustRealDataset val constructor is used in both eval scripts. Verify the split definitions match.

---

### Q20. Can we distill the D1R YOLOv8m detector into the ConvNeXt multi-task head (Distill D-6 recommendation)?

**Evidence:** 133_OPUS_COMPLETE_ANSWERS.md D-6: "Endorse as the one forward-looking detection experiment (P2.1), with two constraints: time-box it (3 days)... Expected honest outcome: +0.1-0.2 mAP."

**Current answer (best guess):** Soft-distillation using D1R logits as teacher targets could improve D3 detection by +0.1-0.2 mAP, turning the cost framing from "36% of ceiling" to "40-45% of ceiling." The improvement would be capped by architecture mismatch (YOLOv8m detection head vs ConvNeXt RetinaNet-style head).

**What we need to verify:** Check if the ConvNeXt detection head produces logits compatible with soft-label distillation. The RetinaNet focal loss formulation accepts soft targets natively.

---

## §3. D3 multi-task (the 0.358 = 36% of ceiling)

### Q21. Why does the D3 full eval (d3_full_eval/metrics.json) contain NO detection metrics at all?

**Evidence:** d3_full_eval/metrics.json has act_frame_accuracy=0.129, head_pose_MAE=9.1 deg, psr_f1=0.0, but NO det_mAP50, det_mAP_50_95, or per-class AP fields. The SOTA_STATUS.md reports D3 mAP50=0.358 from a separate "subsample, 250-batch, class-balanced" evaluation.

**Current answer (best guess):** The D3 full eval pipeline either crashed during detection metric computation or was configured without detection eval enabled. The 0.358 comes from a subsample evaluation that may not represent full-set performance. This means we have two different D3 detection numbers from different protocols, and they may disagree.

**What we need to verify:** Rerun D3 evaluation with detection metrics explicitly enabled. Check the D3 evaluation config for detection head output flag. The full eval file (at d3_full_eval/metrics.json) is missing the key number we need to characterize the system.

---

### Q22. The SOTA_STATUS says D3 mAP50=0.358 from "subsample, 250-batch, class-balanced" — what does "class-balanced" mean in this context, and does it inflate mAP?

**Evidence:** SOTA_STATUS.md line 15 (context): "D3 multi-task ConvNeXt-Tiny mAP50=0.358 (subsample, 250-batch, class-balanced)". Opus 133 D-4 confirms: "the headline 0.358 is the present-class mAP diluted by zero-GT classes counted as 0."

**Current answer (best guess):** "Class-balanced subsample" likely means the eval ran on a subset of validation batches with class-balanced sampling (undersampling majority classes, oversampling minority). This could inflate mAP above full-set performance if applied incorrectly. Combined with the zero-GT dilution issue, the true mAP is uncertain.

**What we need to verify:** Reproduce the 0.358 number. Run D3 evaluation on full validation set. Compare full-set mAP against subsample mAP. The discrepancy (if any) tells us how representative the 0.358 is.

---

### Q23. If present-class mAP50=0.573 (correcting for zero-GT dilution), and the ceiling is 0.995, the multi-task model achieves 58% of the detection ceiling. How does this change the cost narrative?

**Evidence:** Opus 133 D-4: "COCO convention excludes classes with no GT from the mean — so if WACV followed COCO convention, the protocol-comparable number for our model is 0.573, not 0.358, and the multi-task result is 58% of ceiling (42% cost), not 36%."

**Current answer (best guess):** The cost narrative shifts from "64% cost" to "42% cost" for detection. This makes the multi-task approach look substantially better — the detection head is doing real work, just not at single-task levels. The paper should use whichever convention WACV used.

**What we need to verify:** Check WACV 2024's evaluation code to determine whether they include or exclude zero-GT classes from mAP averaging. This single fact changes the paper's central claim.

---

### Q24. What is the per-class mAP50 for the 15 present classes in D3? Which classes drag down the average?

**Evidence:** Not available — D3 full eval metrics.json does not contain detection per-class AP. Opus 133 C-7 confirms "15 present classes" from det_n_present_classes=15 in D1 metrics.

**Current answer (best guess):** Classes with few GT examples (e.g., class 5 with 18 GT, class 8 with 20 GT) likely have near-zero AP. The average 0.573 is pulled down by these rare classes. A per-class AP breakdown would reveal whether the model handles common assembly states well.

**What we need to verify:** Run D3 eval with per-class AP output. Check whether the 15 present classes form a long tail (few high-prevalence classes carry the average) or a more uniform distribution.

---

### Q25. Does the D3 multi-task model's detection head suffer from the same issue as the PSR head (dead ReLU units, saturated bias)?

**Evidence:** PSR-3 analysis (133) found dead per-component PSR heads from ReLU gating + bias -1.0 saturation. The detection head uses a RetinaNet-style Focal Loss head with separate classification and regression branches.

**Current answer (best guess):** Unlikely. The detection head's architecture is different (Focal Loss with sigmoid outputs, anchor-based regression). The PSR head's issue was specific to the monotonic decoder constraint and per-component binary classification with negative bias.

**What we need to verify:** Check the detection head activation statistics (mean, dead unit fraction) from a D3 run. If the head has >20% dead units, the PSR problem may be architecture-wide.

---

### Q26. How does the multi-task ConvNeXt detection head compare to YOLOv8m in terms of detection rate (detections per frame at conf>=0.25)?

**Evidence:** D4 eval log shows YOLOv8m_industreal.pt produces detections on <1% of frames. D1R produces detections on ~80% of frames. D3 multi-task detection rate is unknown.

**Current answer (best guess):** The D3 detection head likely produces dense detections (>1 per frame average) because it was trained on our data distribution. The 0.358 mAP (even if diluted) implies non-trivial detection output, not the near-zero density of the OOD checkpoint.

**What we need to verify:** Add detection rate statistics to the D3 eval output. Compare against YOLOv8m density.

---

### Q27. In a multi-task gradient conflict analysis, what fraction of updates improve detection vs harm it? Is detection the victim of conflicting gradients?

**Evidence:** SOTA_STATUS.md reports act 0.129 (near majority baseline) and pose 9.1 deg (near SOTA). Detection at 0.358 is the weakest head relative to its ceiling.

**Current answer (best guess):** Detection is likely the victim of competing gradients. The activity and pose heads have different loss landscapes, and the shared backbone must serve all four tasks. Gradient surgery (PCGrad, GradVac) or uncertainty-weighted loss weighting could improve detection at the expense of other tasks.

**What we need to verify:** Run a gradient conflict diagnostic on the multi-task model. Compare per-task gradient cosine similarity across a batch. If detection gradients consistently oppose activity gradients, the conflict hypothesis is confirmed.

---

### Q28. Does class-balanced sampling during D3 evaluation represent real-world performance or an optimistic bound?

**Evidence:** "subsample, 250-batch, class-balanced" — this implies the evaluation set was sampled to balance class frequencies, which does not match the natural class distribution.

**Current answer (best guess):** Class-balanced eval gives an optimistic estimate because it upweights rare classes. Real-world performance would be lower under natural class distribution (where majority classes dominate). However, for a research paper, class-balanced metrics are standard for rare-class tasks.

**What we need to verify:** Compare class-balanced mAP against natural-distribution mAP for D3. The gap between them quantifies the class imbalance penalty.

---

### Q29. What is the D3 mAP50-95? Is the gap between mAP50 and mAP50-95 consistent with D1R (gap=0.134)?

**Evidence:** Not available — D3 metrics missing from committed files.

**Current answer (best guess):** The mAP50-mAP50-95 gap for D3 is likely wider than D1R's (0.134) because the multi-task model produces lower-quality boxes. A gap of 0.2-0.3 would mean the model detects objects at approximate locations but with poor bounding box fit.

**What we need to verify:** Add mAP50-95 computation to D3 eval. Compare gap with D1R's gap to assess box quality degradation.

---

### Q30. If we trained only the detection head (freeze backbone), what mAP would we get vs the full multi-task setup?

**Evidence:** Current D3 trains all heads simultaneously. Activity linear probe (frozen backbone) achieved 0.2169, proving the backbone has weak but present signal.

**Current answer (best guess):** A backbone-frozen detection head would likely achieve lower mAP than the full setup because the backbone features are not optimized for detection. However, if the multi-task setup is dominated by non-detection gradients, freezing the backbone might focus the detection head on its task. The experiment would isolate backbone adaptation effects.

**What we need to verify:** Run a head-only training experiment (freeze backbone, train detection head with cross-entropy/Focal Loss). Compare mAP against full multi-task.

---

## §4. D4 backbone swap (F1=0.000 -> 0.347 with re-tuning)

### Q31. Why does the YOLOv8m->PSR decoder have the same detection pattern as D1 (detections on <1% of frames)?

**Evidence:** eval_yolov8m_psr.log shows detection pattern: most batches produce "0 total detections" (16 images, 0 detections). Occasional bursts of "16 total detections" correspond to frames with terminal assembly state (class 22 activation). The same checkpoint is used as D1.

**Current answer (best guess):** The YOLOv8m_industreal.pt checkpoint is the same file used in D1. Its sparse detection pattern (<1% of frames) is consistent across both eval scripts because they use the same model. The D4 decoder receives near-constant -3.0 default logits (sigmoid(-3.0) ~ 0.047) on 99% of frames, providing no transition signal.

**What we need to verify:** Check both eval scripts load `yolov8m_industreal.pt` from the same path. Confirm the detection pattern is identical.

---

### Q32. How does the threshold retuning from default Q48 (hi=0.5, lo=0.3, min=3) to best (hi=0.3, lo=0.1, min=2) improve F1 from 0.000 to 0.347? What's the mechanism?

**Evidence:** D4 retuned sweep results (sweep_results.json): default (Q48) gives F1=0.000, best (hi=0.3, lo=0.1, min=2) gives F1=0.347. The sweep log shows 145 combinations tested.

**Current answer (best guess):** The default Q48 thresholds (hi=0.5, lo=0.3) were tuned for ConvNeXt-derived PSR logits, which span a wider range. YOLOv8m-derived logits are near-constant at -3.0 (no-detection baseline) with rare spikes. Lowering the hysteresis thresholds captures these rare spikes as state transitions. The hi=0.3 threshold catches any logit above -3.0 (actual detection), while min=2 prevents single-frame noise from triggering transitions.

**What we need to verify:** Plot the YOLOv8m-derived logit distribution for each component. Confirm the -3.0 default dominates and the non-default tail is what the retuned thresholds capture.

---

### Q33. The sweep_results.json shows multiple threshold combinations producing identical F1=0.3470 — is the F1 surface flat, and how robust is the best configuration?

**Evidence:** sweep_results.json shows 8 entries with F1=0.347 (hi=0.3, various lo and min). Many entries show F1=0.329 or 0.335. The best is 0.347.

**Current answer (best guess):** The F1 surface has a plateau around hi=0.3-0.4, lo=0.1-0.25, min=2-3. The 0.347 is within this plateau. This suggests the decoder is relatively insensitive to exact thresholds as long as they capture the same set of rare detection events. The plateau width (~0.02 F1 difference across 8 configs) indicates the result is not a fragile peak.

**What we need to verify:** Compute the standard deviation across 5 runs at the best configuration. If SD < 0.01, the result is robust.

---

### Q34. What is the per-component breakdown of the D4 F1=0.347? Which components benefit from YOLOv8m detections?

**Evidence:** sweep_results.json has per-component threshold arrays (sustain_hi/sustain_lo/sustain_min per component 0-10). But comp_stats show near-identical mean=-2.7, std~1.0 across components.

**Current answer (best guess):** Components that have high prevalence in the PSR labels (comp 0: 100%, comp 1: 91%) likely benefit most because they coincide with frames where class 22 detections fire. Low-prevalence components (comp 4: 14%, comp 10: 25%) likely see no benefit because the detections don't correlate with their state changes.

**What we need to verify:** Compute per-component F1 for D4. Compare against PSR per-component prevalence table (SOTA_STATUS.md lines 30-43).

---

### Q35. The verdict.json says "threshold-partial: decoder shows marginal benefit — thresholds partially helpful." What does "marginal" mean in context — is 0.347 useful for anything?

**Evidence:** verdict.json line 5-6: "f1_at_t_original": 0.0, "f1_at_t_best_global": 0.347, "f1_at_t_retuned": 0.261, "verdict": "threshold-partial..."

**Current answer (best guess):** In the PSR evaluation protocol, F1=0.347 is below any useful operating point. With 11 PSR components, random guessing achieves ~0.09 F1 (one correct out of 11). The 0.347 is better than random but far below the ConvNeXt-based PSR F1=0.75. The decoder transfer is confirmed to work (proven by F1>0), but it needs a better detection backbone to reach competitive levels.

**What we need to verify:** Compare D4 F1=0.347 with the ConvNeXt-based PSR F1=0.75. The gap = 0.403 is the headroom from a better backbone.

---

### Q36. If the YOLOv8m->PSR experiment used D1R (fine-tuned, 0.995 mAP) instead of the IndustReal checkpoint (0.0004 mAP), would the D4 F1 approach the ConvNeXt-based PSR F1 (0.75)?

**Evidence:** D4 uses yolov8m_industreal.pt (same file as D1, mAP=0.0004). D1R is a separate fine-tuned checkpoint achieving mAP=0.995.

**Current answer (best guess):** Yes, significantly. D1R produces dense detections (80%+ of frames) with correct class labels. Feeding these into the PSR decoder would likely achieve close to the ConvNeXt-based PSR F1 (0.75), possibly higher. This would be the strongest evidence that "backbone detection density is the binding constraint" (SOTA_STATUS.md).

**What we need to verify:** Run D4 eval using D1R weights. Expected outcome: F1 should jump from 0.347 to ~0.7+.

---

### Q37. How many detections per frame does the YOLOv8m checkpoint actually produce (at conf>=0.1, conf>=0.25, conf>=0.5)?

**Evidence:** D4 log shows aggregate "total detections" per batch, not per-frame. The pattern: most batches (16 frames) have 0 detections; occasional bursts of 16.

**Current answer (best guess):** At conf>=0.1, approximately 0.05-0.1 detections per frame (5-10% of frames have 1 detection; 90-95% have 0). At conf>=0.25, even fewer. This is consistent with the class-22-only detection pattern.

**What we need to verify:** Add per-frame detection count statistics to the eval script. Compute mean, median, P95 detection counts.

---

### Q38. The D4 verdict calls this "threshold-partial" — should we run the same experiment with D1R weights as the definitive test, or does the limited YOLOv8m->PSR framing already serve its purpose as an ablation?

**Evidence:** verdict.json calls the result "threshold-partial: decoder shows marginal benefit." SOTA_STATUS.md states: "backbone detection density is the binding constraint."

**Current answer (best guess):** The D4 experiment already serves its purpose as an ablation — it proves the MonotonicDecoder is not architecturally broken (it achieves F1=0.347 with threshold retuning) and it identifies the binding constraint (backbone detection density, not decoder capability). Running D1R->PSR would be an incremental confirmation, not a new finding. The marginal time is better spent elsewhere.

**What we need to verify:** Check whether any upstream user (Opus 133) requested D1R->PSR as a follow-up. If yes, prioritize; if not, treat D4 as closed.

---

### Q39. The D4 sweep tests 145 combinations — is this exhaustive enough, or could fine-tuning thresholds per-recording improve F1 further?

**Evidence:** sweep_results.json: 145 combinations tested, best F1=0.347. The per-component optimal threshold set gives lower F1=0.261 (verdict.json).

**Current answer (best guess):** The sweep appears comprehensive across the useful range (hi=0.3-0.7, lo=0.1-0.3, min=2-6). Per-recording calibration could potentially improve F1 by adapting to recording-specific detection statistics, but with 16 recordings and each having <1% detection frames, the sample size per recording is too small for reliable calibration.

**What we need to verify:** Compute per-recording D4 F1. Check variance across recordings.

---

### Q40. Why do per-component optimal thresholds (F1=0.261) perform worse than global thresholds (F1=0.347)? What does this tell us about the signal structure?

**Evidence:** verdict.json: global best F1=0.347, per-comp optimal F1=0.261. Sweep per-component thresholds (sustain_hi: all 0.85, sustain_lo: 0.085-0.112, sustain_min: 2-4).

**Current answer (best guess):** Per-component thresholds overfit to component-specific noise patterns because the YOLOv8m detection signal is extremely sparse and uncorrelated across components. A global threshold acts as a noise filter — only detections above the global threshold are trusted, smoothing the signal across components. Per-component tuning catches component-specific noise spikes as false transitions.

**What we need to verify:** Compare the detection pattern across components for a batch of frames. If per-component noise is decorrelated, the global-threshold advantage is explained.

---

## §5. SOTA comparison and fair-comparison framing

### Q41. WACV 2024 publishes mAP50=0.838 on 24-class ASD — does this number come from a random-split eval or a recording-aware split like ours?

**Evidence:** SOTA_STATUS.md line 11 claims D1R "BEATS SOTA" with 0.995 vs ~0.95. Opus 133 D-3 confirms: "WACV's own annotated-frames (0.838) vs entire-video (0.641) numbers show a 0.2 split sensitivity."

**Current answer (best guess):** WACV likely uses a random split where frames from the same recording can appear in both train and val. Our recording-aware split is harder, making the 0.838 not directly comparable to our 0.995. The real comparison should be WACV on recording-aware split vs our model on recording-aware split.

**What we need to verify:** Re-evaluate D1R on a random split (WACV-compatible protocol). If D1R achieves 0.995 even on random split, the model architecture comparison is fair. If D1R drops to 0.95 on random split, the 0.995 advantage is entirely split choice.

---

### Q42. What mAP does WACV's model achieve on our recording-aware split (cross-evaluation)?

**Evidence:** WACV 2024 model is based on a two-stage detector with FPN, trained on the IndustReal random split. We could download their checkpoint and evaluate on our split.

**Current answer (best guess):** WACV's model likely achieves 0.75-0.80 on our recording-aware split (a ~0.04-0.09 drop from their 0.838 on random split). The recording-aware split generalization gap affects both models, but YOLOv8m (D1R, 0.995) handles it better due to stronger data augmentation.

**What we need to verify:** Either download WACV's model or re-implement their protocol. Run eval on our split. Report the cross-split mAP.

---

### Q43. Is "~0.95" (the SOTA_STATUS claimed SOTA for detection) actually substantiated? Where does this number come from?

**Evidence:** SOTA_STATUS.md line 11: "~0.95" as the SOTA comparison for mAP50. No citation is given. Opus 133 C-1 flags: "The supplementary files carry a stale/incorrect attribution."

**Current answer (best guess):** The 0.95 is likely an informal estimate from the WACV paper (their annotated-frames-only 0.838 and their claim about MViTv2-S). No published detection benchmark exists on this dataset with a verified 0.95 number. The ~0.95 is a placeholder, not a verified SOTA.

**What we need to verify:** Search published literature for IndustReal detection benchmarks. Check the WACV 2024 supplementary materials for any 0.95+ number. If none exists, remove the ~0.95 claim and replace with WACV 0.838 as the only published baseline.

---

### Q44. The paper structure currently frames the multi-task detection result (0.358) against WACV (0.838) as a gap. If we correct to present-class mAP (0.573), the gap shrinks from 0.480 to 0.265. Does this change the paper's narrative from "significant gap" to "competitive"?

**Evidence:** Opus 133 D-4 analyzes the present-class correction. 0.573 vs 0.838 gap = 0.265. 0.358 vs 0.838 gap = 0.480.

**Current answer (best guess):** Yes, the narrative changes. A 0.480 gap is "poor performance" — a reviewer would say the multi-task model is failing at detection. A 0.265 gap is "competitive with headroom" — the multi-task model is doing useful detection, just not at single-task levels. The corrected framing makes the entire paper stronger.

**What we need to verify:** Determine WACV's mAP convention (COCO standard excludes zero-GT classes). If WACV uses COCO convention, the corrected 0.573 is the correct comparison number. If WACV includes zero-GT classes as 0, the uncorrected 0.358 is correct.

---

### Q45. Is mAP50 the right metric for this paper, or should mAP50-95 be the primary detection metric?

**Evidence:** D1R mAP50=0.995, mAP50-95=0.861 (gap=0.134). WACV reports mAP50 but not mAP50-95.

**Current answer (best guess):** Both should be reported. mAP50 is the legacy metric and directly comparable to WACV. mAP50-95 is the current COCO standard and harder. For the multi-task model, mAP50-95 would be more informative because mAP50 saturates easily (any box containing the object counts). The paper should lead with mAP50 for comparison and provide mAP50-95 in the supplementary.

**What we need to verify:** Compute D3 mAP50-95. Compare gap with D1R's gap to assess box quality.

---

### Q46. The WACV 2024 paper reports both "annotated-frames" (0.838) and "entire-video" (0.641) mAP — which protocol does our eval match?

**Evidence:** Our eval script runs on the entire validation set (38,036 frames) via `IndustRealDataset(split="val")`. WACV's 0.838 uses only frames that overlap with COCO annotations; their 0.641 uses the full video.

**Current answer (best guess):** Our eval matches the "entire-video" protocol (all frames in the val split). The direct comparison is therefore our mAP vs WACV's 0.641, not their 0.838. If D3 achieves 0.573 (present-class corrected), it's already close to WACV's entire-video 0.641 — a much stronger result.

**What we need to verify:** Check whether WACV's "annotated-frames" subset matches our labelled frames or a different selection. Confirm our eval is entire-video by checking the val dataset size (38,036 frames).

---

### Q47. Should the paper include a WACV protocol reimplementation as a baseline, or cite their number with a protocol disclosure?

**Evidence:** WACV does not provide open-source evaluation code. Their protocol description is from the paper text.

**Current answer (best guess):** Reimplementing WACV's protocol would be the gold standard but is high effort. Instead: cite their published numbers, explicitly list the protocol differences (split type, frame selection, mAP convention), and report our numbers under both our protocol and an approximated WACV protocol. A single table with protocol columns is sufficient.

**What we need to verify:** Identify the exact eval protocol from WACV's paper. List protocol dimensions where we differ.

---

### Q48. MViTv2-S is cited as SOTA for activity (0.622) — does it also produce detection outputs, and how does our detection compare to MViTv2-S's detection?

**Evidence:** MViTv2-S is a video transformer architecture for action recognition, not detection. The "BEATS SOTA" claim mixes detection and activity baselines.

**Current answer (best guess):** MViTv2-S does not have a detection head. Its activity SOTA (0.622 top1) is not comparable to detection mAP. The paper must keep detection and activity comparisons separate and cite different baselines for each.

**What we need to verify:** Confirm MViTv2-S does not perform object detection. If needed, find a video detection baseline (e.g., Video RetinaNet, TubeR).

---

### Q49. Is "BEATS SOTA" a defensible claim for detection given the protocol differences (recording-aware split, YOLOv8m vs two-stage detector)?

**Evidence:** SOTA_STATUS.md line 11 claims "BEATS SOTA" based on D1R mAP50=0.995 > ~0.95. Opus 133 C-1 says this attribution is stale.

**Current answer (best guess):** "BEATS SOTA" is not defensible as a current paper claim because: (1) the 0.995 comes from a separately-trained YOLOv8m, not the multi-task model; (2) the 0.95 SOTA number is unverifiable; (3) protocol differences make cross-study comparison unreliable. The defensible claim is: "Our single-task YOLOv8m upper bound (0.995 mAP50) substantially exceeds the published WACV baseline (0.838), confirming the detection task can be solved with adequate training."

**What we need to verify:** Replace all "BEATS SOTA" claims in the paper with the defensible framing. The multi-task model comparison should be against its own single-task ceiling (0.995), not against external SOTA.

---

### Q50. Is the mAP50=0.995 from YOLOv8m (a one-stage anchor-free detector) meaningfully comparable to WACV's mAP=0.838 from a two-stage Faster R-CNN with FPN?

**Evidence:** YOLOv8m uses a CSPDarknet backbone with TaskAlignedAssigner. WACV uses ResNet-50 FPN with Faster R-CNN (two-stage). D1R mAP50=0.995, WACV mAP50=0.838.

**Current answer (best guess):** Different architectures, different training recipes, different data splits. The 0.157 gap (0.995-0.838) is a combination of architecture advantage, training recipe advantage, and split advantage. A controlled experiment (YOLOv8m vs Faster R-CNN on the same split) would isolate the architecture contribution. Until then, the comparison is suggestive but not conclusive.

**What we need to verify:** Train a Faster R-CNN (or use ultralytics' implementation) on our split. If it achieves 0.95+, the gap is mostly training recipe, not architecture.

---

## §6. Adversarial review (built-in debate)

### Challenge 1: "The 0.358 mAP from D3 is embarrassingly low. A model that can't even detect the objects it needs to analyze cannot claim to perform multi-task assembly analysis."

**Evidence supporting challenge:** D3 mAP50=0.358 is 64% below the single-task ceiling (0.995). Activity (0.129) and PSR (0.75) are both weak for different reasons. The entire system's bottleneck is detection — if objects aren't found, activity, pose, and PSR are all degraded.

**Counter-evidence:** The D3 number is diluted by zero-GT classes (present-class corrected mAP=0.573). The evaluation uses WACV's entire-video protocol (not their annotated-frames 0.838). The multi-task model achieves competitive PSR (0.75) and near-SOTA head pose (9.1 deg) simultaneously — the detection cost is the price of four-task integration. A single-task detector (YOLOv8m, 0.995) cannot produce activity or PSR outputs.

**Ruling needed from Opus:** Is 0.358 (or 0.573 present-class) defensible as a multi-task detection result, or does it require improvement before the paper is submitted?

---

### Challenge 2: "Disclosure 5.4 about the error-state class (24) is misleading. Claiming FPR=0% is trivial when the model was never trained on error frames and error states don't exist in the evaluation set."

**Evidence supporting challenge:** SOTA_STATUS.md lines 71-76: "no frames in any split were annotated for it." The FPR=0 comparison with WACV's 65% is not informative — WACV trained on actual error instances.

**Counter-evidence:** The FPR=0 is reported precisely to document this limitation. The comparison with WACV's 65% is paired with the disclosure that our model "was never exposed to the concept during training." This is a null-finding disclosure, not a claim of superiority.

**Ruling needed from Opus:** Should the error-state FPR comparison remain in the paper with its disclosure, or be removed as potentially misleading?

---

### Challenge 3: "The D4 experiment is irrelevant. A threshold sweep that barely lifts F1 from 0.000 to 0.347 is not a finding — it's noise. Of course lowering thresholds catches more events; this proves nothing."

**Evidence supporting challenge:** F1=0.347 is below any useful PSR operating point. 16 recordings, each with <1% detection frames, produce statistically unreliable results. The "binding constraint" conclusion is an overinterpretation.

**Counter-evidence:** The D4 experiment is explicitly framed as a disclosure experiment (SOTA_STATUS.md §5.4). Its contribution is negative evidence: it proves the decoder is not the bottleneck (it reaches F1=0.347 with retuning) and isolates the backbone detection density as the constraint. Negative findings are valid scientific contributions. The 145-combination sweep confirms the F1 plateau, ruling out a missed optimum.

**Ruling needed from Opus:** Should D4 appear in the paper as a §5.4 negative finding, or be moved to supplementary/appendix?

---

### Challenge 4: "The entire detection narrative is built on a YOLOv8m we didn't train (D1R), evaluated on a split we designed, compared against a baseline we didn't reproduce. This is not science — it's cherry-picking comparison points."

**Evidence supporting challenge:** D1R is a separate training run, not the multi-task model. WACV's numbers are cited without protocol reimplementation. The recording-aware split comparison is apples-to-oranges.

**Counter-evidence:** D1R is explicitly framed as an upper bound (ceiling measurement), not a system result. The paper's detection contribution is the multi-task model's performance, with D1R as reference. WACV numbers are cited with full protocol disclosure. The paper does not claim a direct head-to-head comparison with WACV — it claims the multi-task model achieves X% of its single-task ceiling, which is an internally controlled experiment.

**Ruling needed from Opus:** Is the current detection framing defensible, or does it require rewording to avoid cherry-picking accusations?

---

### Challenge 5: "The '36% of ceiling' framing is a liability. If the reviewer reads it as 'our model achieves 36% of the best possible performance,' they will reject immediately. This number makes the paper look weak, not cost-efficient."

**Evidence supporting challenge:** 36% of ceiling sounds terrible. A reviewer might not read the "multiple tasks share one backbone" justification before concluding the model is broken.

**Counter-evidence:** The cost framing is precisely the paper's contribution: demonstrating that a single ConvNeXt-Tiny can perform four assembly-analysis tasks at once, with detection accepting a 64% cost relative to its ceiling. This cost is the price of multi-task efficiency. The alternative — four independent models — would be 4x the parameters and 4x the inference cost with zero cross-task synergy. The 36% number is honest and the paper's argument defends it.

**Ruling needed from Opus:** Should the paper lead with the "36% of ceiling" number (current plan) or reframe around the corrected 58% of ceiling (present-class mAP) to reduce negative first impression?

---

## §7. Open decisions for Opus

The following decisions require Opus input to resolve contradictions and set the detection narrative:

1. **Which mAP convention to use for headline comparisons?** COCO standard (exclude zero-GT classes, giving 0.573) or our default (include as 0, giving 0.358)? This depends on what WACV used. Highest impact-to-effort ratio of any detection decision.

2. **Should the paper claim "BEATS SOTA" for detection anywhere?** The claim is currently unsupported (SOTA status uses unverifiable ~0.95 number). Options: remove entirely, replace with "WACV baseline comparison with protocol disclosure," or run WACV reimplementation and substantiate.

3. **What is the protocol for the detection evaluation table?** Need to decide: which split (recording-aware vs random), which frames (entire-video vs annotated-only), which mAP variant (50 vs 50-95), and which models (multi-task only, or include D1R ceiling).

4. **Does D4 go in the main paper or supplementary?** The negative finding (decoder transfer requires backbone density) is scientifically valid but may distract from the positive PSR story. Opus decides its placement.

5. **Error-state class (24) presentation.** FPR=0% is trivial without training data. Options: single sentence in §5.4, expanded discussion of error-detection as future work, or removed entirely.

6. **Zero-GT class count discrepancy.** Our data shows 6 classes with gt=0 in val; Opus 133 D-4 counts 9. Resolve this discrepancy before any per-class metrics are published.

7. **Per-class D3 mAP breakdown.** This data is missing from committed files. Priority: compute and commit before any paper writing that references per-class performance.

8. **D3 full-set detection metrics.** The d3_full_eval/metrics.json has act, pose, PSR but no detection. This gap must be filled before submission.

9. **Gradient conflict diagnostic priority.** Is running PCGrad/GradVac for detection improvement worth the time before submission? Expected benefit: +0.05-0.10 mAP.

10. **Distillation experiment (D-6) priority.** Opus 133 endorses it as P2.1 with 3-day timebox. Decision needed on whether to attempt before submission.

---

*End of file 134 — 50 deep detection questions.*
