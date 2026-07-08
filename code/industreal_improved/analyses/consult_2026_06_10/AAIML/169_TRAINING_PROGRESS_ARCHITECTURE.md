# 169 — Training Progress and Architecture Plan

**Date:** 2026-07-08
**Purpose:** All training runs, V5b/V8 status, architecture plan, log file paths.

---

## 1. All Training Runs (Log Files)

| Run | Started | Log File | Status |
|---|---|---|---|
| V5b multi-task (KENDALL rebalance) | 2026-07-08 09:02 | `/tmp/train_v5b.log` | RUNNING, epoch 35 82% |
| V8 multi-task (MViTv2-S + YOLOv8m) | 2026-07-08 09:24 | `/tmp/train_v8.log` | RUNNING, epoch 0 step 700+ |
| V5b prior (KENDALL_FIXED=1, died from watchdog) | 2026-07-08 03:23 | `/tmp/train_v5b.log` (old) | KILLED |
| V4 LIVENESS probe (KENDALL_FIXED=1) | 2026-07-07 | `/tmp/train_psr_v4.log` | DONE (gradient paths confirmed) |
| D1R YOLOv8m detection | pre-2026-07-08 | `src/runs/rf_stages/checkpoints/d1_yolov8m_v3/metrics.json` | DONE (0.995 mAP50) |
| Frozen MViTv2-S probe | 2026-07-07 | `src/runs/rf_stages/checkpoints/activity_mvit_probe/results.json` | DONE (0.3810) |

## 2. V5b Current Status

- **PID:** 758477
- **GPU:** 0 (RTX 5060 Ti, CUDA 0 via CUDA_VISIBLE_DEVICES=0)
- **Started:** 2026-07-08 09:02
- **Current epoch:** 35 step 10770-13161 (82% through)
- **Speed:** 2.3 batch/s, ~95 min/epoch, 1:19 elapsed in epoch
- **ETA epoch 50:** ~18h (end-to-end through 50 epochs)
- **Key change:** KENDALL_FIXED_WEIGHTS=0 (let Kendall rebalance)
- **Last val (epoch 33/34):** pose 8.82°→8.52° (improving), det NaN (broken), PSR 0.0 (collapsed), activity 0.0 (collapsed)

## 3. V8 Current Status

- **PID:** 843794
- **GPU:** 1 (RTX 3060, CUDA 1)
- **Started:** 2026-07-08 09:24
- **Current epoch:** 0 step 700+
- **Speed:** ~3.5s/step
- **Loss trajectory:** act 4.0→0.001, pose 0.23→0.03, psr 0.7→0.001
- **Risk:** Classification heads collapsing to zero
- **Architecture:** V8 (YOLOv8m det + MViTv2-S activity + shared pose/PSR)

## 4. V8 Architecture Code

**File:** `/home/newadmin/swarm-bot/master/POPW/working/code/industreal_improved/code/industreal_improved/scripts/train_v8_multitask.py`

Key components:
- `V8Model` class: MViTv2-S backbone (frozen) + 3 heads (activity 69-class, pose 6-dim, PSR 11-comp)
- `V8Dataset` class: reads train.csv (3667 clips), loads 16-frame video clips, per-frame targets
- Kendall-weighted loss with learnable log_var per head
- KENDALL_FIXED_WEIGHTS=0 (let Kendall rebalance)

**Data sources (all verified):**
- `train.csv` (3667 clips, format: recording, offset, action, start_frame, end_frame)
- `recordings/{recording}/videos/{frame:06d}.mp4` (16-frame video clips)
- `recordings/{recording}/AR_labels.csv` (per-frame activity class)
- `recordings/{recording}/pose.csv` (per-frame fwd/up)
- `recordings/{recording}/PSR_labels.csv` (per-frame binary transitions)

## 5. Architecture Plan (Future)

### V8 (Current, 2 backbone multi-task)
- MViTv2-S (frozen, Kinetics-400) for activity/pose/PSR
- YOLOv8m (frozen, D1R weights) for detection
- 4 heads, Kendall-weighted loss
- ETA: 5+ epochs by tomorrow morning

### V9 (Single unified backbone, Future)
- Backbone: e.g., Hiera or unified transformer
- 1 backbone, all 4 heads
- Less memory, simpler training
- ETA: multi-day project

### V6 (MViTv2-S single-backbone, planned but not built)
- MViTv2-S for ALL 4 heads
- Detection would still be weak
- ETA: 1-week project

## 6. File Paths for Opus to Check

### Training Logs
- `/tmp/train_v5b.log` — V5b current (epoch 35 82%)
- `/tmp/train_v8.log` — V8 current (epoch 0 step 700+)
- `/tmp/train_v5b_fresh3.log` — V5b launch log (with launch errors)
- `/tmp/train_v5b_fresh4.log` — V5b launch log (final successful launch)
- `/tmp/train_v5b_fresh2.log` — V5b failed launch (bash syntax error)

### Python Files
- `/home/newadmin/swarm-bot/master/POPW/working/code/industreal_improved/code/industreal_improved/scripts/train_v8_multitask.py` — V8 full pipeline (committed)
- `/home/newadmin/swarm-bot/master/POPW/working/code/industreal_improved/code/industreal_improved/scripts/train_v5b_fresh_kendall_rebalanced.sh` — V5b launch (committed)
- `/home/newadmin/swarm-bot/master/POPW/working/code/industreal_improved/code/industreal_improved/scripts/train_v5_multitask.sh` — V5 launch
- `/home/newadmin/swarm-bot/master/POPW/working/code/industreal_improved/code/industreal_improved/scripts/train_psr_repair_wrapper.py` — wrapper

### Checkpoint / Eval Files
- `/home/newadmin/swarm-bot/master/POPW/working/code/industreal_improved/code/industreal_improved/src/runs/rf_stages/checkpoints/best.pth` (V5b pre-fix)
- `/home/newadmin/swarm-bot/master/POPW/working/code/industreal_improved/code/industreal_improved/src/runs/rf_stages/checkpoints/crash_recovery.pth` (V5b resume)
- `/home/newadmin/swarm-bot/master/POPW/working/code/industreal_improved/code/industral_improved/src/runs/rf_stages/checkpoints/full_eval_ep18_v2/metrics.json`
- `/home/newadmin/swarm-bot/master/POPW/working/code/industreal_improved/code/industreal_improved/src/runs/rf_stages/checkpoints/d1_yolov8m_v3/metrics.json`
- `/home/newadmin/swarm-bot/master/POPW/working/code/industreal_improved/code/industral_improved/src/runs/rf_stages/checkpoints/activity_mvit_probe/results.json`
- `/home/newadmin/swarm-bot/master/POPW/working/code/industreal_improved/code/industral_improved/src/runs/rf_stages/checkpoints/bootstrap_ci.json`

### Code Files (Architecture)
- `/home/newadmin/swarm-bot/master/POPW/working/code/industreal_improved/code/industreal_improved/src/models/model.py` — POPWMultiTaskModel (V5 model)
- `/home/newadmin/swarm-bot/master/POPW/working/code/industreal_improved/code/industral_improved/src/models/video_backbones.py` — MViTv2-S wrapper
- `/home/newadmin/swarm-bot/master/POPW/working/code/industral_improved/code/industral_improved/src/models/video_backbone_multitask.py` — V8 model class (V6)

## 7. Architecture Plan Diagram

```
User's "all 4 heads SOTA-comparable" goal:

            V5b (current, KENDALL rebalance)              V8 (running, MViTv2-S+ConvNeXt replacement)
            ↓                                                       ↓
            Single architecture: ConvNeXt shared              Two architectures: YOLOv8m+MViTv2-S shared
            ↓                                                       ↓
            Output: pose (first baseline), others collapsed  Output: all 4 heads functional
            ↓                                                       ↓
            ETA: ~18h, partial multi-task                  ETA: ~5h, full multi-task

            Best case: 4 heads all SOTA-comparable (single multi-task run)
            Realistic: detection (D1R 0.995), activity (frozen 0.3810 or V8 fine-tune),
                      pose (V5b 7-8°), PSR (V5b 0.5+)
```

## 8. Status Today vs Status Yesterday

| Aspect | Yesterday (2026-07-07 23:00) | Today (2026-07-08 09:30) |
|---|---|---|
| V5b (multi-task) | Running, KENDALL_FIXED=1 (collapsed) | Restarted KENDALL=0, at epoch 35 82% |
| V8 (multi-task + MViTv2) | Not running | Running on GPU 1, epoch 0 |
| D1R detection | 0.995 in repo | 0.995 in repo (no change) |
| Frozen probe | 0.3810 in repo | 0.3810 in repo (no change) |
| Best multi-task result | None (V5b was collapsing) | V5b epoch 33-34 (pose improving, others collapsed) |

**Net progress today:** V5b restart with KENDALL rebalance, V8 architecture committed and started.
</content>
