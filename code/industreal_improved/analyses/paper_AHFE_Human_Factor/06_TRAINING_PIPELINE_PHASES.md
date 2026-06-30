# Plan 6: Training Pipeline — Exact Phases, Duration Estimates, and Error Recovery

> **Backbone:** ConvNeXt-Tiny (confirmed in config.py line 103)
> **Dataset:** IndustReal — 3667 train, 1928 val, 3678 test images (104,751 total)
> **PyTorch:** 2.12.1+cu130, CUDA 13.0, Python 3.13.13
> **Environment:** Both GPUs confirmed working

---

## 1. Current State (June 27, 15:04)

```
GPU 0 (RTX 3060 12GB): IDLE — 70 MiB used
GPU 1 (RTX 5060 Ti 16GB): IDLE — 411 MiB used
Checkpoint: crash_recovery.pth (epoch 0, step 167 — SIGTERM)
Prior best metrics: det_mAP50_pc = 0.304, forward_angular_MAE_deg = 9.13
Training presets: 
  - RF2: batch_size=4, grad_accum=8, effective_batch=32, ConvNeXt-Tiny
  - RF3: batch_size=4, grad_accum=8, same backbone
  - recovery_det_only: single-task detection baseline
```

### Speed Estimates (from prior logs)

| GPU | Batch/s | Epochs/day | 15 epochs | 30 epochs |
|-----|---------|-----------|-----------|-----------|
| RTX 3060 (observed) | 1.6 batch/s | ~42 epochs/day | ~9 hours | ~18 hours |
| RTX 5060 Ti (estimated) | 2.5-3.0 batch/s | ~65-78 epochs/day | ~5-6 hours | ~10-12 hours |

---

## 2. Training Phases — Exact Detail

### Phase 0: Pre-Flight Checks (30 min)

Before any training, verify:

```bash
# 0.1: Verify dataset integrity
cd /media/newadmin/master/POPW/datasets/industreal
echo "Train: $(wc -l < train.csv) images"
echo "Val: $(wc -l < val.csv) images"
echo "Test: $(wc -l < test.csv) images"

# 0.2: Verify checkpoint loads correctly
cd /media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src
python3 -c "
import torch
ckpt = torch.load('runs/full_multi_task_tma_tbank/checkpoints/crash_recovery.pth', 
                  map_location='cpu', weights_only=True)
print('Epoch:', ckpt.get('epoch', 'unknown'))
print('Model keys:', len(ckpt.get('model', {})))
"

# 0.3: Verify both GPUs accessible
python3 -c "import torch; print(f'GPU0: {torch.cuda.get_device_name(0)}, GPU1: {torch.cuda.get_device_name(1)}')"

# 0.4: Run a single validation step to confirm evaluation pipeline works
CUDA_VISIBLE_DEVICES=1 python3 src/evaluation/evaluate.py \
  --ckpt runs/full_multi_task_tma_tbank/checkpoints/crash_recovery.pth \
  --profile-efficiency-only
```

### Phase A: RF2 Training (5060 Ti, CUDA GPU 0) — ~6 hours

**Purpose:** Train detection + body pose + head pose. Resuming from crash_recovery.pth.

**Config:**
- Preset: stage_rf2 (train_det=True, train_act=False, train_psr=False, train_head_pose=True)
- Batch: 4, Grad accum: 8, Effective batch: 32
- Epochs: 15 (defined in preset)
- Data: full dataset (SUBSET_RATIO=1.0 per resolved_config)

```bash
# Launch on 5060 Ti (CUDA GPU 0)
CUDA_VISIBLE_DEVICES=0 nohup python3 -u src/training/train.py \
  --preset stage_rf2 \
  --resume src/runs/full_multi_task_tma_tbank/checkpoints/crash_recovery.pth \
  --seed 42 --num-workers 4 \
  > src/runs/phase_A_5060ti/logs/train.log 2>&1 &

echo "Phase A PID: $!"
```

**Expected outputs:**
- `checkpoints/best.pth` — best checkpoint by combined metric
- `checkpoints/latest.pth` — latest epoch
- `logs/metrics.jsonl` — per-epoch metrics
- `logs/train.log` — full training log

**Target metrics:**
- det_mAP50_pc ≥ 0.30 (prior run achieved 0.304 at epoch 17)
- forward_angular_MAE_deg ≤ 10 deg (prior run achieved 9.13)
- Both detection and head pose heads ALIVE (LIVENESS check)

**Monitoring:**
```bash
# Check progress
tail -20 src/runs/phase_A_5060ti/logs/train.log

# Check LIVENESS
grep "LIVENESS" src/runs/phase_A_5060ti/logs/train.log | tail -3

# Check metrics
python3 -c "
import json
with open('src/runs/phase_A_5060ti/logs/metrics.jsonl') as f:
    lines = f.readlines()
    if len(lines) > 0:
        last = json.loads(lines[-1])
        print('det_mAP50:', last.get('det_mAP50', 'N/A'))
        print('det_mAP50_pc:', last.get('det_mAP50_pc', 'N/A'))
        print('head_pose MAE:', last.get('forward_angular_MAE_deg', 'N/A'))
"
```

**Failure recovery:**
| Symptom | Cause | Fix |
|---------|-------|-----|
| loss > 50 or NaN | Exploding gradients | Lower BASE_LR, check detach_reg_fpn |
| det_mAP50_pc < 0.15 after 5 epochs | Not learning | Check DET_OHEM_ENABLED, try False |
| head_pose MAE > 30 deg | Not converging | Check USE_GEO_HEAD_POSE, HEAD_POSE_LOSS_WEIGHT |
| CUDA OOM | Memory exhaustion | Reduce batch_size to 2, grad_accum to 4 |
| SIGTERM (seen before) | External kill | Resume from latest checkpoint |

### Phase B: RF3 Activity Training (5060 Ti, CUDA GPU 0) — ~6 hours

**Purpose:** Add activity recognition head to the trained model. Train all three heads.

```bash
# Launch after Phase A completes
CUDA_VISIBLE_DEVICES=0 nohup python3 -u src/training/train.py \
  --preset stage_rf3 \
  --resume src/runs/phase_A_5060ti/checkpoints/best.pth \
  --seed 42 --num-workers 4 \
  > src/runs/phase_B_5060ti/logs/train.log 2>&1 &

echo "Phase B PID: $!"
```

**Config:** stage_rf3 (train_det=True, train_act=True, train_psr=False, train_head_pose=True)
**Epochs:** 15
**Data:** Full dataset

**Target metrics:**
- Activity Top-1 ≥ 10% (vs 1.3% chance baseline for 74 classes)
- Activity Top-5 ≥ 30% (if available)
- Detection should not regress below RF2 levels

**Known Issue from Prior Analysis:** LDAM-DRW schedule switches at epoch ~60 but RF3 is only 15 epochs. DRW never activates. **Mitigation:** Config has `use_ldam_drw: False` — this is the intended setting. Activity uses standard CE loss with label smoothing.

**Collapse detection:** In first 3 epochs, check if activity predictions spread across classes or collapse to one class:
```bash
grep "act_confusion\|activity" src/runs/phase_B_5060ti/logs/train.log | head -5
```
If collapsed to single class: lower ACT_LR_MULTIPLIER or check act_ramp parameter.

### Phase C: Ablation A — Single-Task Detection (3060 (CUDA GPU 1), parallel) — ~9 hours

**Purpose:** Run detection + head pose as single-task baseline to compare with multi-task RF2.

```bash
# Launch in parallel with Phase A/B on 5060 Ti (uses 3060 on CUDA GPU 1)
CUDA_VISIBLE_DEVICES=1 nohup python3 -u src/training/train.py \
  --preset recovery_det_only \
  --seed 123 --num-workers 4 \
  > src/runs/phase_C_3060/logs/train.log 2>&1 &

echo "Phase C PID: $!"
```

**Config:** recovery_det_only (train_det=True, train_head_pose=True, train_act=False, train_psr=False)
**Note:** This starts from scratch (no pretrained checkpoint for multi-task), so it may take longer to converge.

**Expected:** ~9 hours on 3060 for 15 epochs.

### Phase D: PSR Go/No-Go (3060 (CUDA GPU 1)) — ~1 hour

**Purpose:** Determine if PSR head can learn anything at all.

```bash
# Run after Phase C completes or during idle time
CUDA_VISIBLE_DEVICES=1 python3 -u src/training/train.py \
  --preset stage_rf2 --train_psr True \
  --resume src/runs/phase_A_5060ti/checkpoints/best.pth \
  --epochs 2 --subset 0.35
```

**Decision rule:**
- If `psr_f1_at_t > 0.3` on train set after 200 steps → PSR is viable. Schedule RF4.
- If `psr_f1_at_t = 0.0` or `comp0=1.0` only → PSR structurally stuck. **Drop from paper.** One paragraph negative result.

### Phase E: Final Full Evaluation (Both GPUs) — ~2 hours

```bash
# On 5060 Ti (CUDA GPU 0): Evaluate Phase B results
CUDA_VISIBLE_DEVICES=0 python3 src/evaluation/evaluate.py \
  --ckpt src/runs/phase_B_5060ti/checkpoints/best.pth --split test \
  > src/runs/eval_final_5060ti.log 2>&1

# On 3060 (CUDA GPU 1): Evaluate Ablation A results
CUDA_VISIBLE_DEVICES=1 python3 src/evaluation/evaluate.py \
  --ckpt src/runs/phase_C_3060/checkpoints/best.pth --split test \
  > src/runs/eval_final_3060.log 2>&1

# Per-class diagnostic (from best overall checkpoint)
CUDA_VISIBLE_DEVICES=0 python3 src/diag_per_class_truth.py \
  --run src/runs/phase_B_5060ti
```

---

## 3. Data Requirements — Complete Inventory

| Component | Path | Size | Status |
|-----------|------|------|--------|
| Train images | datasets/industreal/images/ | ~105K files | ✅ Present |
| Train CSV | datasets/industreal/train.csv | 3667 entries | ✅ Present |
| Val CSV | datasets/industreal/val.csv | 1928 entries | ✅ Present |
| Test CSV | datasets/industreal/test.csv | 3678 entries | ✅ Present |
| Labels COCO | datasets/industreal/labels_coco.json | 48 MB | ✅ Present |
| Synthetic data | datasets/industreal/ASD_SyntheticOnly_test/ | ~260K synthetic | ⚠️ Not used (intentional) |
| Model weights (ASD) | datasets/industreal/assembly_state_detection_model_weights/ | Pretrained YOLO | ⚠️ For baseline comparison only |
| Model weights (AR) | datasets/industreal/action_recognition_model_weights/ | Pretrained MViTv2 | ⚠️ For baseline comparison only |

**Note:** We train on real data only (manual_only dataset_mode). The synthetic data exists but is intentionally excluded to demonstrate feasibility under realistic data constraints.

---

## 4. Environment Verification

| Component | Required | Current | Status |
|-----------|----------|---------|--------|
| Python | 3.10+ | 3.13.13 | ✅ |
| PyTorch | 2.0+ | 2.12.1 | ✅ |
| CUDA | 12+ | 13.0 | ✅ |
| GPU 0 | Any NVIDIA | RTX 3060 12GB | ✅ |
| GPU 1 | Any NVIDIA | RTX 5060 Ti 16GB | ✅ |
| Disk space | 200 GB+ | Check with `df -h` | Needs verification |
| RAM | 32 GB+ | 64 GB | ✅ (from earlier logs) |

**Check disk space:**
```bash
df -h /media/newadmin/master/POPW/
```

---

## 5. Training Schedule Summary

| Phase | GPU | Duration | When | Depends On |
|-------|-----|----------|------|------------|
| 0: Pre-flight | Both | 30 min | Jun 27 immediate | Nothing |
| A: RF2 | 5060 Ti | ~6 hrs | Jun 27-28 | Phase 0 |
| B: RF3 | 5060 Ti | ~6 hrs | Jun 29-30 | Phase A complete |
| C: Ablation A | 3060 | ~9 hrs | Jun 28-Jul 1 (parallel) | Phase 0 |
| D: PSR go/no-go | 3060 | ~1 hr | Jun 28 (after Phase C start) | Phase A checkpoint |
| E: Final eval | Both | ~2 hrs | Jul 1-4 (as results land) | Phases A, B, C |

**Total GPU time:** ~22 hrs on 5060 Ti, ~12 hrs on 3060 = ~34 GPU-hours
**Wall-clock time with parallel execution:** ~24-30 hours over 3-4 days
**Buffer remaining:** 23 days (Jun 27 to Jul 20)
