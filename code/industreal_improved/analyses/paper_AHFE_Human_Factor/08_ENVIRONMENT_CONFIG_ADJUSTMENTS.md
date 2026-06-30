# Plan 8: Environment, Configuration, and Code Adjustments

> **Target framework:** PyTorch 2.12.1 + CUDA 13.0, Python 3.13.13
> **Codebase:** industreal_improved/src (config.py, training/train.py, evaluation/evaluate.py)
> **Key adjustments verified and needed before training**

---

## 1. Current Environment — Verified

| Component | Value | Status |
|-----------|-------|--------|
| OS | Ubuntu 22.04 (from prior docs) | ✅ |
| Python | 3.13.13 | ✅ |
| PyTorch | 2.12.1+cu130 | ✅ |
| CUDA runtime | 13.0 (nvidia-smi 595.71.05) | ✅ |
| GPU 0 | RTX 3060 12GB (Turing) | ✅ |
| GPU 1 | RTX 5060 Ti 16GB (Blackwell) | ✅ |
| RAM | 64 GB (from prior logs) | ✅ |
| Disk | /media/newadmin/master/POPW/ (external) | ⚠️ Needs `df -h` |
| Conda env | miniconda3 (from prior logs) | ✅ |

### Required Verification Commands

```bash
# 1. Check disk space
df -h /media/newadmin/master/POPW/

# 2. Verify CUDA compatibility between GPUs
python3 -c "
import torch
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f'GPU {i}: {props.name}, {props.total_memory/1e9:.1f}GB, CC {props.major}.{props.minor}')
"

# 3. Verify datasets accessible
ls /media/newadmin/master/POPW/datasets/industreal/images/ | wc -l

# 4. Verify source code integrity
cd /media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src
python3 -c "import config; print('Config OK')"
python3 -c "from models.model import POPW; print('Model OK')"
python3 -c "from training.train import main; print('Train OK')"

# 5. Check available Python packages
pip list 2>/dev/null | grep -E "torch|numpy|matplotlib|scipy|pillow|tqdm|tensorboard"
```

---

## 2. Critical Configuration Adjustments

### 2.1 Activity Training Fix (RF3)

**Known issue from prior analysis:** LDAM-DRW schedule switches at epoch ~60 but RF3 is only 15 epochs. DRW never activates.

**Current config** (verified from resolved_config.json):
```python
use_ldam_drw = False  # Correct — standard CE loss with label smoothing
```

**Action:** Ensure `use_ldam_drw: False` is set in the stage_rf3 preset. This is already confirmed.

**Additional check needed:** Verify the activity loss ramp (`act_ramp`) doesn't suppress signal:
```python
# In config.py, search for ACT_RAMP_EPOCHS
# Current value: ACT_RAMP_EPOCHS = 5 (from resolved_config)
# This means activity loss is linearly ramped over 5 epochs
# This is correct — gives detection time to stabilize before activity kicks in
```

### 2.2 Detection Head Configuration

**Current** (from resolved_config):
```python
DET_OHEM_ENABLED = True     # Online Hard Example Mining
DET_ASYMMETRIC_GAMMA = True # Asymmetric Focal Loss
DET_LR_MULTIPLIER = 1.0     # No separate LR for detection
DET_BIAS_LR_FACTOR = 1.0    # No separate bias LR
```

**Note from prior analysis:** OHEM+FocalLoss gradient suppression is suspected as the primary bottleneck causing the 0.207 mAP50 ceiling. If RF2 performance plateaus below 0.25 mAP50, run OHEM ablation:

```bash
# OHEM OFF experiment (if needed)
CUDA_VISIBLE_DEVICES=0 python3 src/training/train.py \
  --preset stage_rf2 --ohem_enabled False \
  --resume src/runs/phase_A_5060ti/checkpoints/best.pth \
  --epochs 5 --subset 0.35
```

### 2.3 Gradient Flow (Already Fixed in Config)

From config.py comments:
- `detach_reg_fpn = False` — regression gradients flow into FPN (previously detached, now fixed)
- `detach_psr_fpn = True` — PSR gradients detached (PSR not trained in RF2/RF3 anyway)

**This is the correct setting** as documented in the config.py comment at line 1134: "RF2 is NOT a reinit stage — the regression head already has decent GIoU signal that SHOULD flow into FPN."

---

## 3. Checkpoint Strategy

### 3.1 Current Checkpoints

| File | Size | Epoch | Status |
|------|------|-------|--------|
| crash_recovery.pth | 216 MB | 0 (step 167) | Valid but early |
| crash_recovery.pth.tmp | 0 B | — | Stale (delete) |

### 3.2 Checkpoint Management

```bash
# Delete stale .tmp file
rm src/runs/full_multi_task_tma_tbank/checkpoints/crash_recovery.pth.tmp

# Training saves automatically to:
# - runs/phase_A_5060ti/checkpoints/best.pth (best by metric)
# - runs/phase_A_5060ti/checkpoints/latest.pth (latest epoch)
# - runs/phase_A_5060ti/checkpoints/epoch_<N>.pth (every epoch)

# All training metadata in:
# - runs/phase_A_5060ti/logs/train.log
# - runs/phase_A_5060ti/logs/metrics.jsonl
# - runs/phase_A_5060ti/logs/resolved_config.json
```

### 3.3 Archive Old Runs (Free Up Space)

```bash
# Check size of old runs
du -sh src/runs/*/

# If needed, archive benchmark run:
tar czf src/runs/archive_benchmark.tar.gz src/runs/full_multi_task_tma_tbank_benchmark/
rm -rf src/runs/full_multi_task_tma_tbank_benchmark/
```

---

## 4. GPU Configuration

### 4.1 CUDA_VISIBLE_DEVICES Mapping

| GPU | Physical | Training Role |
|-----|----------|---------------|
| GPU 0 | RTX 3060 | Ablation A, diagnostics, efficiency |
| GPU 1 | RTX 5060 Ti | Main training pipeline (RF2, RF3) |

**Important:** The 5060 Ti is GPU 1 (bus 04:00.0). Always use `CUDA_VISIBLE_DEVICES=1` for primary training.

### 4.2 Memory Configuration

From resolved_config.json:
```python
# Current settings (verified working):
BATCH_SIZE = 4 on RTX 3060 with ConvNeXt-Tiny
GRAD_ACCUM_STEPS = 8
Mixed precision = False (FP32 only)
GPU memory: ~2 GB allocated during training (from logs)
```

**On 5060 Ti (16 GB):** Could increase batch_size to 8 for faster training, but batch_size=4 with grad_accum=8 (effective 32) is safer and has proven stability.

### 4.3 CUDA Configuration

```python
# From training logs:
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
# These are for reproducibility. If speed is priority:
# torch.backends.cudnn.benchmark = True  # ~10-20% faster
# torch.backends.cudnn.deterministic = False  # Allow non-deterministic algorithms
```

---

## 5. Code Freeze

After launching training, **do not modify config.py or train.py** unless a critical bug is found. The current config has been validated across multiple runs.

### Files to NOT Modify:
- `config.py` — Contains all training configuration
- `training/train.py` — Main training loop
- `models/model.py` — Model architecture
- `losses.py` — Loss functions

### Files That CAN Be Modified (for paper writing):
- `evaluation/evaluate.py` — If new metrics are needed for the paper
- `diag_per_class_truth.py` — For per-class analysis
- Any new analysis scripts in a separate directory

---

## 6. Package Dependencies

If any package is missing:

```bash
# Core dependencies
pip install torch==2.12.1 torchvision==2.12.1 --index-url https://download.pytorch.org/whl/cu130

# Paper/plotting dependencies
pip install matplotlib seaborn pandas numpy scipy

# Evaluation/utility
pip install tqdm tensorboard thop
```

The codebase uses `thop` for FLOPs counting (referenced in evaluate.py):
```python
# from thop import profile
# This is called during --profile-efficiency-only evaluation
```

---

## 7. Blockchain Environment Setup

For the x402 implementation (Plan 3):

```bash
# Solana CLI
sh -c "$(curl -sSfL https://release.anza.xyz/v2.1.0/install)"
export PATH="$HOME/.local/share/solana/install/active_release/bin:$PATH"

# Verify
solana --version
solana config set --url devnet

# Node.js for x402 template (if using the full template)
node --version  # Needs 18+
npm --version

# Python deps for bridge
pip install aiohttp solders base58
```
