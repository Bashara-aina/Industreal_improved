# MViTv2-S Fine-Tuning Launch Status

**Date:** 2026-07-07 16:59 UTC
**Agent:** Opus 144 (MVITV2-S Fine-Tuning Launch Status Specialist)

## GPU Status

| GPU | Model | Memory Used | Memory Total | Utilization | Status |
|-----|-------|-------------|--------------|-------------|--------|
| 0   | RTX 3060 | 3357 MiB | 12288 MiB | 89% | **BUSY** — ConvNeXt detection training (PID 1574104) |
| 1   | RTX 5060 Ti | 3188 MiB | 16311 MiB | 80% | **BUSY** — PSR repair training (PID 1901740) |

**Both GPUs are occupied.** Fine-tuning cannot launch.

## Script Verification

| Item | Path | Status |
|------|------|--------|
| Launch script | `scripts/train_mvit_finetune.sh` | EXISTS — validated, runnable |
| Training module | `src/training/train_video_finetune.py` | EXISTS — validated, parsable |
| Checkpoints dir | `src/runs/rf_stages/checkpoints/activity_mvit_probe/` | EXISTS — probe weights present |

## Configuration

- **Backbone:** MViTv2-S (Kinetics-400 pretrained, 34.5M params)
- **Batch size:** 2
- **Clip length:** 16 frames
- **Epochs:** 20 (Stage 1: frozen 3 epochs, Stage 2: full fine-tune 17 epochs)
- **Learning rates:** head 5e-5, backbone 1e-5
- **Memory budget:** ~3.5 GB (gradient checkpointing + FP16)
- **Target GPU:** CUDA_VISIBLE_DEVICES=0 (RTX 3060)
- **OOM safeguard:** Built into script header — "BOTH GPUs busy — BUILD ONLY, do NOT launch"

## Frozen Probe Baseline

| Metric | Value | Threshold | Pass? |
|--------|-------|-----------|-------|
| Frozen probe accuracy | 0.3810 | 0.30 | YES |
| Fine-tune target | 0.45-0.55 | — | PENDING |

## Blocked Reason

Both GPUs (0 and 1) are actively running training jobs.

- GPU 0: ConvNeXt detection training (PID 1574104, running since ~14:54, 255 min elapsed)
- GPU 1: PSR repair training (PID 1901740, running since ~16:50, 15 min elapsed)

The script's OOM safeguard explicitly blocks launch when both GPUs are busy. Launch requires at least one GPU free.

## Recommendation

Re-attempt after either ConvNeXt detection (GPU 0) or PSR repair (GPU 1) completes. The MViTv2-S fine-tuning has a ~3.5 GB memory footprint and will fit comfortably on RTX 3060 once freed.

Launch command once GPU is free:
```bash
cd /media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved
bash scripts/train_mvit_finetune.sh
```
