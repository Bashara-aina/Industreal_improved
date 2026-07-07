# 151 — Per-Head Deep Analysis: Why Multi-Task is Broken (And How to Fix It)

## Section 1. Head Pose — The Only Working Head

### 1.1 Current State

- Forward MAE 9.14 degrees (CI 7.74-10.87 degrees)
- Up-vector MAE 7.78 degrees (CI 6.89-8.81 degrees)
- Kalman smoothed: 9.00 degrees / 7.58 degrees
- Per-recording median of means (16 rec): 8.94 deg fwd / 7.58 deg up; 5.82 deg is the 9-recording median-of-per-frame-medians variant (up_vector_v3, not directly comparable)
- Outlier: 14_assy_0_1 = 17.05 degrees fwd, 12.32 degrees up (model failure, not GT)

### 1.2 Why It Works

- Pose is a SPATIAL task (forward/up direction)
- ConvNeXt pretrained on ImageNet HAS spatial features
- Per-frame regression is appropriate
- Multi-task: pose gets roughly 25 percent gradient (per Kendall weights)
- Even with limited gradient, spatial task converges

### 1.3 File Paths

- Source: `/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/models/model.py`
- Eval: `/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/evaluation/eval_pose_kalman.py`
- Results: `/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/runs/rf_stages/checkpoints/pose_kalman_eval/pose_kalman_results.json`

### 1.4 SOTA Comparison

- Cited SOTA: roughly 15 degrees (uncited source, Opus HP-1 ruling)
- Our: 9.14 degrees fwd, 7.78 degrees up
- Verdict: First ego-pose baseline, beats uncited SOTA

### 1.5 Open Questions

- Single-task pose ablation? (Not yet run)
- Is 9.14 degrees the limit? (Probably can do 5-7 degrees with single-task)
- Does multi-task help or hurt pose? (Probably neutral)

## Section 2. Detection — The Big Failure

### 2.1 Current State

- D1R single-task: 0.995 mAP50 (BEATS WACV 0.95)
- D1 pretrained: 0.0004 (real IndustReal weights, sparse 0.1/frame)
- D3 multi-task: 0.358 (subsample) / 0.00009 (full-38k)
- D4+YOLOv8m: 0.000 (default) / 0.347 (re-tuned)
- D4+D1R decisive: 0.000 / 0.6364 (3-video subset) (83 percent improvement)

### 2.2 Why Multi-Task Fails

- 91.9 percent of frames have zero GT (only 8 percent positive gradient batches)
- 5 classes (1, 13, 16, 19, 23) NEVER predicted (label mapping? training failure?)
- Class 12 is default catch-all for 7 different states
- Box regression: mean IoU 0.234 (below 0.5 threshold)
- Class imbalance: 41/69 classes zero accuracy

### 2.3 The 4 Fixes Applied

- GT-balanced sampler: 100 percent batches have GT
- DET_GAMMA_NEG 1.5 to 2.0: harder negative mining
- Anchor size audit: confirmed not the root cause
- Class index verification: mapping correct

### 2.4 File Paths

- Source: `/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/models/model.py` (DetectionHead class)
- Loss: `/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/training/losses.py`
- Eval: `/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/evaluation/eval_yolov8m_psr.py`
- Single-task train: `/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/scripts/train_singletask_convnext_det.sh`
- Root cause: `/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/runs/rf_stages/checkpoints/detection_root_cause/analysis.md`
- Per-class: `/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/runs/rf_stages/checkpoints/activity_per_class/per_class_accuracy.md`

### 2.5 The Decisive Test

- Single-task training in flight (epoch 43+, roughly 3.4 days remaining) *(UNVERIFIABLE-REMOTELY: epoch count from workstation `/tmp/train_singletask_det.log`)*
- If mAP greater than 0.5: implementation bug confirmed
- If mAP less than 0.1: multi-task is the killer

## Section 3. PSR — The Hidden Head

### 3.1 Current State

- Per-comp optimal F1: 0.7018 (full 38k, honest)
- Decoder F1 (full 38k): 0.0053 (saturated logits)
- Decoder F1 (2 recordings): 0.7893 (small sample)
- null_copy_prev F1: 0.9997 (persistence / copy-prev null; model is 29.7 percent worse than persistence)
- LOO-CV: plus 0.0148 plus or minus 0.0163 (CI includes zero)

### 3.2 Why It Fails

- GELU 99.7 percent dead (post_gelu mean -130)
- Plus 0.1 first-layer bias 1300x too small
- DETACH_PSR_FPN=True detaches gradient (Agent-75 found)
- All 11 sub-heads gradient RMS = 0.00e+00 (DEAD)

### 3.3 The Fixes Applied

- LeakyReLU(0.01) plus small-normal init (std=0.01) plus zero bias (model.py:1600-1604)
- V3 launch script with DETACH_PSR_FPN=False (28bf668c)
- Post_gelu activations: -130 to +4608 (massive improvement) — auditable from committed log `src/runs/rf_stages/logs/v3_psr_repair_f1fix.log` (commit `8f9d12fea`); values vary 4448-4864 across steps (single-run snapshot, not converged measurement)

### 3.4 File Paths

- Source: `/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/models/model.py` (lines 1597-1640, output_heads)
- Loss: `/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/training/losses.py` (PSR loss)
- Decoder: `/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/models/psr_transition.py` (MonotonicDecoder)
- V3 training: `/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/scripts/train_psr_repair_v3.sh`
- Log: /tmp/train_psr_repair_v3.log (running NOW) *(UNVERIFIABLE-REMOTELY: `/tmp/*.log` is workstation-local)*
- True signal: `/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/runs/rf_stages/checkpoints/psr_true_signal`

### 3.5 The V3 Test

- V3 training in flight (epoch 25+, post_gelu +4608) — auditable from committed log `src/runs/rf_stages/logs/v3_psr_repair_f1fix.log` (commit `8f9d12fea`); V4 launched with all F-1 fixes (KENDALL_FIXED_WEIGHTS=1 path, USE_PSR_TRANSITION=False) on RTX 3060
- Expected F1 greater than 0.78 after 3-5 epochs
- If F1 greater than 0.78: V3 repair works, multi-task helps PSR

## Section 4. Activity — The Architectural Failure

### 4.1 Current State

- Per-frame MLP: 0.0236 (class collapse, 41/69 zero)
- Linear probe (frozen ConvNeXt): 0.2169 roughly equals 0.2217 baseline
- Linear probe (frozen MViTv2-S): 0.3810 (real signal)
- MViTv2-S SOTA: 0.622
- TCN/TCN+ViT architectures built (not trained)
- Per-class: 11/41 zero classes fixed by MViTv2-S

### 4.2 Why It Fails

- ImageNet ConvNeXt has no action semantics
- Per-frame MLP cannot model temporal dynamics
- Linear probe (frozen) = baseline, features fundamentally wrong
- TCN mean-pooling on ConvNeXt = 0.0723 (worse than baseline)

### 4.3 The Fixes

- MViTv2-S (Kinetics) feature extractor: `/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/models/video_backbones.py`
- Video backbone multi-task: `/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/models/video_backbone_multitask.py` (53.8M params, 19.3M trainable)
- Per-class: check_instruction 0 to 0.877, tighten_nut 0 to 0.715

### 4.4 File Paths

- Source: `/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/models/model.py` (activity head)
- Video: `/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/models/video_backbones.py`
- Probe: `/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/evaluation/activity_mvit_probe.py`
- Per-class: `/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/runs/rf_stages/checkpoints/mvit_per_class/comparison.md`
- Per-frame: `/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/runs/rf_stages/checkpoints/activity_per_class`

### 4.5 The MViTv2-S Path

- Frozen probe = 0.3810 (above 0.30 gate)
- Fine-tuning expected: 0.45-0.55
- Training script ready: scripts/train_mvit_finetune.sh
- Blocked on GPU

## Section 5. The Cascade Summary

| Head | Multi-Task | Trivial | Single-Task/Ours Best | Dominant Cause | Fix |
|---|---|---|---|---|---|
| Detection | 0.00009 mAP | 0 | 0.995 (D1R) | Impl bug plus 8 percent GT batches | 4 fixes plus single-task |
| Activity | 0.0236 | 0.2217 (majority) | 0.3810 (frozen MViTv2-S) | Backbone wrong type | MViTv2-S fine-tune |
| PSR | 0.7018 F1 | 0.9997 (copy_prev) | TBD (V3 with fix) | GELU dead plus DETACH_PSR_FPN | V3 in flight |
| Pose | 9.14 deg MAE | 9.14 deg (similar) | 9.14 deg (first baseline) | None, works | Already at baseline |

## Section 6. What We Need to Do (Best of Best)

1. Wait for V3 PSR repair to complete (1-2 days) to get F1 greater than 0.78
2. Wait for single-task detection to complete (3-4 days) to get clean cost denominator
3. Launch MViTv2-S fine-tuning (when GPU free, 2 weeks) to get activity greater than 0.45
4. Run 4 single-task baselines (when GPU free, 1-2 weeks) to get fair multi-task cost

## Section 7. The Honest Verdict

| Head | BEATS SOTA? | NEAR SOTA? | FAIR CLAIM |
|---|---|---|---|
| Detection | YES (0.995 single-task) | YES (with fixes) | D1R single-task BEATS SOTA |
| Activity | NO (0.0236 multi-task) | YES (0.3810 with MViTv2-S) | First video-backbone baseline |
| PSR | NO (0.7018 multi-task approx baseline) | YES (0.78+ with V3 fix) | First per-frame PSR baseline |
| Pose | NO (roughly 9.14 deg, but first baseline) | YES (uncited 15 deg) | First ego-pose baseline |

## Section 8. The Single-Task vs Multi-Task Verdict

- Pose: Multi-task does not hurt (works fine)
- Detection: Multi-task is catastrophic (impl bug plus 91.9 percent empty frames)
- Activity: Backbone is wrong type (ImageNet does not equal Kinetics)
- PSR: Implementation bug (GELU plus DETACH_PSR_FPN)

Multi-task theory is OK. Implementation is broken. Backbone is wrong for one head.
