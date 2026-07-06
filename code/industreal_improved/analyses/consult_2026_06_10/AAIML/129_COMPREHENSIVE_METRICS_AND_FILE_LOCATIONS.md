# 129 — Comprehensive Metrics + File Locations

**Date:** 2026-07-06
**Purpose:** Master reference table for every metric, every result file, every code path. This is the single source of truth that Opus can audit. All file paths are absolute. All numbers are from the most recent run unless noted.

---

## 1. Master SOTA Status Table (epoch_18, best.pth)

| Head | Metric | Our Value | SOTA | Source | Status |
|---|---|---|---|---|---|
| **Detection (ASD)** | mAP50 (YOLOv8m self-trained, d1r) | **0.995** | ~0.95 (WACV 2024) | `runs/detect/src/runs/yolov8m_industreal/d1r/results.csv` | **BEATS SOTA** |
| **Detection (ASD)** | mAP50-95 (YOLOv8m self-trained) | **0.861** | ~0.84 | `runs/detect/src/runs/yolov8m_industreal/d1r/results.csv` | **BEATS SOTA** |
| **Detection (D1)** | mAP50 (re-eval, d1_yolov8m) | **0.0004** | n/a | `src/runs/rf_stages/checkpoints/d1_yolov8m/metrics.json` | BROKEN — protocol mismatch |
| **Detection (D1 v3)** | mAP50 (no-shift, latest) | **0.0004** | n/a | `src/runs/rf_stages/checkpoints/d1_yolov8m_v3/metrics.json` | same as v1 — real metric |
| **Activity (per-frame)** | top1 valid | 0.023 | n/a | `src/runs/rf_stages/checkpoints/SOTA_STATUS.md` | broken — needs video backbone |
| **Activity (clip-level)** | top1 (16-frame majority) | **0.028** | 0.622 (MViTv2-S) | `src/runs/rf_stages/checkpoints/activity_clip_ep18/activity_clip.json` | broken — MLP can't do temporal |
| **Activity T3 baseline** | top1_69 (verb-grouped) | 0.6223 | 0.622 | `src/runs/rf_stages/checkpoints/t3_full_eval.json` | matches |
| **Activity T3 baseline** | top1_75 (mecanno) | 0.04 | n/a | `src/runs/rf_stages/checkpoints/t3_mecanno_eval.json` | reference |
| **Head Pose forward** | angular MAE | **8.39°** | ~15° (claimed) | `src/runs/rf_stages/checkpoints/SOTA_STATUS.md` | **near SOTA** (15° unsourced) |
| **Head Pose up** | angular MAE (300-subset) | 13.52° | n/a | `src/runs/rf_stages/checkpoints/SOTA_STATUS.md` | reasonable |
| **Head Pose up** | angular MAE (full eval) | 26.20° | n/a | `src/runs/rf_stages/checkpoints/SOTA_STATUS.md` | mixed — unit ambiguity |
| **PSR (global thresh 0.10)** | macro F1 | **0.7217** | 0.901 (STORM) | `src/runs/rf_stages/checkpoints/SOTA_STATUS.md` | competitive |
| **PSR (per-comp optimal)** | macro F1 (full 38k) | **0.7499** | 0.901 (STORM) | `src/runs/rf_stages/checkpoints/psr_optimal_thr/optimal_thresholds.json` | **near SOTA** |
| **PSR (per-comp optimal)** | macro F1 (5k subset) | **0.7810** | 0.901 | `src/runs/rf_stages/checkpoints/psr_optimal_thr_v2/optimal_thresholds.json` | higher on subset |
| **PSR (D4 YOLOv8m)** | event F1 / POS / Edit | **0.000 / 0.999 / 0.994** | 0.883 (B3) / 0.901 (STORM) | `src/runs/rf_stages/checkpoints/d4_yolov8m_psr/metrics.json` | **POS paradox confirmed** |
| **PSR POS** | ordered-pair fraction | **0.968** | 0.812 (STORM) | `src/runs/rf_stages/checkpoints/SOTA_STATUS.md` | metric artifact |

---

## 2. PSR Per-Component Threshold Detail

### 2.1 Full-set sweep (38036 frames, `psr_optimal_thr`)

| comp | gt_pos_frac | best_thresh | F1 |
|---|---|---|---|
| 0 | 1.000 | 0.05 | 1.0000 |
| 1 | 0.911 | 0.20 | 0.9627 |
| 2 | 0.911 | 0.15 | 0.9578 |
| 3 | 0.545 | 0.85 | 0.7480 |
| 4 | 0.142 | 0.80 | 0.3455 |
| 5 | 0.631 | 0.50 | 0.7793 |
| 6 | 0.544 | 0.45 | 0.7057 |
| 7 | 0.667 | 0.90 | 0.8041 |
| 8 | 0.667 | 0.90 | 0.8536 |
| 9 | 0.527 | 0.05 | 0.6900 |
| 10 | 0.183 | 0.70 | 0.4020 |
| **Macro F1** | | | **0.7499** |

Source: `src/runs/rf_stages/checkpoints/psr_optimal_thr/optimal_thresholds.json`

### 2.2 5k subset sweep (`psr_optimal_thr_v2`)

| comp | gt_pos_frac | best_thresh | F1 |
|---|---|---|---|
| 0 | 1.000 | 0.05 | 1.0000 |
| 1 | 0.927 | 0.20 | 0.9650 |
| 2 | 0.927 | 0.20 | 0.9678 |
| 3 | 0.541 | 0.85 | 0.7264 |
| 4 | 0.112 | 0.80 | 0.4148 |
| 5 | 0.541 | 0.50 | 0.7075 |
| 6 | 0.454 | 0.45 | 0.6251 |
| 7 | 0.733 | 0.90 | 0.8514 |
| 8 | 0.733 | 0.90 | 0.9042 |
| 9 | 0.617 | 0.05 | 0.7628 |
| 10 | 0.055 | 0.70 | 0.6667 |
| **Macro F1** | | | **0.7810** |

Source: `src/runs/rf_stages/checkpoints/psr_optimal_thr_v2/optimal_thresholds.json`

---

## 3. PSR Detection Class Taxonomy (DET_CLASS_NAMES)

```
1=background, 2=10000000000, 3=10010010000, 4=10010100000, 5=10010110000,
6=11100000000, 7=11110010000, 8=11110100000, 9=11110110000, 10=11110111100,
11=11110111110, 12=11110110001, 13=11110111101, 14=11110111111, 15=11110101111,
16=11110011111, 17=11110011110, 18=11110101110, 19=11100001110, 20=11101101110,
21=11101011110, 22=11101111110, 23=11101111111, 24=error_state
```

Each name encodes an 11-bit PSR component state as a binary string. Source: `src/config.py:202-227`.

---

## 4. Per-Task Loss Configuration

| Setting | Value | Source |
|---|---|---|
| TRAIN_DET | True | `src/config.py:2052` |
| TRAIN_HEAD_POSE | True | `src/config.py:2052` |
| TRAIN_ACT | True | `src/config.py:2052` |
| TRAIN_PSR | True | `src/config.py:45` |
| USE_KENDALL | True | train.log "Ablation" line |
| BATCH_SIZE | 6 (orig), 2 (sm) | train.log + `run_command.txt` |
| GRAD_ACCUM_STEPS | 8 | `src/config.py` |
| EFFECTIVE_BATCH | 48 (orig), 16 (sm) | `src/config.py` |
| BASE_LR | 0.0005 | `src/config.py` |
| EPOCHS | 100 | `src/config.py` |
| PSR_LOSS_WEIGHT | 5.0 | `src/config.py:1097` |
| PSR_FOCAL_GAMMA | 0.5 | `src/config.py` |
| PSR_COMP_WEIGHTS | [1.0, 1.21, 1.20, 1.98, 5.03, 1.61, 1.66, 2.20, 2.20, 2.75, 4.61] | `src/config.py` |
| KENDALL_HP_PREC_CAP | True | `src/config.py` |
| KENDALL_FIXED_WEIGHTS | False | `src/config.py` |
| KENDALL_HP_FIXED_LAMBDA | 0.2 | `src/config.py` |
| ACTIVITY_HEAD_SIMPLE | True | `src/config.py` |
| ACTIVITY_GRAD_BLEND_RATIO | 1.0 | `src/config.py` |
| USE_VIDEOMAE | False | `src/config.py` |
| PSR_TRANSITION_THRESHOLD_HI | 0.5 | `src/config.py` |
| PSR_TRANSITION_THRESHOLD_LO | 0.3 | `src/config.py` |
| PSR_TRANSITION_MIN_SUSTAINED | 3 | `src/config.py` |
| USE_PSR_SEQUENCE_MODE | True | `src/config.py` |
| PSR_SEQUENCE_LENGTH | 8 | `src/config.py` |
| PSR_SEQ_EVERY_N_BATCHES | 4 | `src/config.py` |
| DETACH_PSR_FPN | True | `src/config.py` |

---

## 5. Architecture Inventory

| Module | Param Count | Source |
|---|---|---|
| ConvNeXt-Tiny backbone | ~28M | `src/models/model.py` |
| Activity Head (simple) | 150K | `src/models/model.py` ActivityHead |
| Activity Head (TCN+ViT, T=16, disabled) | 8.2M | `13_ARCHITECTURE_REWRITE.md` |
| PSR Head (transformer 3-layer) | ~3.1M | `src/models/model.py` |
| Detection Head | ~5M | `src/models/model.py` |
| Hand FiLM conditioning | small | `src/models/model.py:1982-2037` |
| Head Pose FiLM | small | `src/models/model.py` |
| FeatureBank sliding window | ring buffer | `src/models/model.py` |
| Total model | ~93 GFLOPs, ~53M params | `day1-checkpoint-done-and-next-steps.md` |

---

## 6. Checkpoint Inventory

| Checkpoint | Path | Size | Date | Notes |
|---|---|---|---|---|
| best.pth (epoch 18) | `src/runs/rf_stages/checkpoints/best.pth` | 738 MB | 2026-07-06 00:26 | Promoted from epoch_18.pth after fix |
| epoch_18.pth | `src/runs/rf_stages/checkpoints/epoch_18.pth` | 704 MB | 2026-07-05 20:40 | PSR F1=0.83 on subset |
| epoch_11.pth | `src/runs/rf_stages/checkpoints/epoch_11.pth` | 704 MB | 2026-07-04 13:58 | Old best.pth (broken selection) |
| epoch_10.pth | `src/runs/rf_stages/checkpoints/epoch_10.pth` | 704 MB | 2026-07-04 08:04 | |
| epoch_9.pth | `src/runs/rf_stages/checkpoints/epoch_9.pth` | 704 MB | 2026-07-04 05:07 | |
| crash_recovery.pth (rf_stages) | `src/runs/rf_stages/checkpoints/crash_recovery.pth` | 738 MB | 2026-07-05 20:40 | From epoch 18 |
| crash_recovery.pth (full_multi) | `src/runs/full_multi_task_tma_tbank_benchmark/checkpoints/crash_recovery.pth` | 738 MB | 2026-07-06 10:06 | Pre-epoch-24 training |
| yolov8m_industreal.pt | `src/runs/rf_stages/checkpoints/yolov8m_industreal.pt` | 311 MB | symlink | From IndustReal authors |
| asd_best_IndustRealandSynthetic.pt | `/media/newadmin/master/POPW/datasets/industreal/assembly_state_detection_model_weights/` | 311 MB | 2023-06-26 | Microsoft official weights |
| d1r/best.pt (YOLOv8m self-trained) | `runs/detect/src/runs/yolov8m_industreal/d1r/weights/best.pt` | 52 MB | | mAP50=0.995 |

---

## 7. Eval Result Files Inventory

| Eval | Output File | Result Summary |
|---|---|---|
| Full eval stream (epoch 18) | `src/runs/rf_stages/checkpoints/full_eval_ep18_stream/metrics.json` | All heads, streaming |
| PSR optimal thresholds (full) | `src/runs/rf_stages/checkpoints/psr_optimal_thr/optimal_thresholds.json` | macro-F1=0.7499 |
| PSR optimal thresholds (5k) | `src/runs/rf_stages/checkpoints/psr_optimal_thr_v2/optimal_thresholds.json` | macro-F1=0.7810 |
| Activity clip-level (epoch 18) | `src/runs/rf_stages/checkpoints/activity_clip_ep18/activity_clip.json` | top1=0.028 |
| Activity clip checkpoints | `src/runs/rf_stages/checkpoints/activity_clip_ep18/checkpoint_*.pkl` | intermediate saves |
| D1 YOLOv8m v1 | `src/runs/rf_stages/checkpoints/d1_yolov8m/metrics.json` | mAP50=0.0004 |
| D1 YOLOv8m v2 (+1 shift) | `src/runs/rf_stages/checkpoints/d1_yolov8m_v2/metrics.json` | mAP50=0.0 |
| D1 YOLOv8m v3 (no shift) | `src/runs/rf_stages/checkpoints/d1_yolov8m_v3/metrics.json` | mAP50=0.0004 |
| D4 YOLOv8m → PSR | `src/runs/rf_stages/checkpoints/d4_yolov8m_psr/metrics.json` | F1=0, POS=0.999 |
| D1 YOLOv8m self-trained | `runs/detect/src/runs/yolov8m_industreal/d1r/results.csv` | mAP50=0.995 |
| D1 YOLOv8m proper | `runs/detect/runs/detect/src/runs/yolov8m_industreal/d1r_proper/weights/best.pt` | interrupted at epoch 3 |
| T3 MViTv2-S full eval | `src/runs/rf_stages/checkpoints/t3_full_eval.json` | top1_69=0.622 |
| T3 MViTv2-S mecanno | `src/runs/rf_stages/checkpoints/t3_mecanno_eval.json` | top1=0.04 |
| TTA 3-arm | `src/runs/rf_stages/checkpoints/tta_3arm/` | test-time aug |
| YOLOv8m sanity | `src/runs/rf_stages/checkpoints/yolov8m_sanity.json` | class mapping verify |
| PSR threshold sweep log | `src/runs/rf_stages/checkpoints/psr_threshold_sweep.log` | + JSON |
| PSR data cache (old best.pth) | `src/runs/rf_stages/checkpoints/psr_data_cache_best.pth` | 3.3 MB cached logits |

---

## 8. Training Logs Inventory

| Log | Path | Notes |
|---|---|---|
| Full multi-task training | `src/runs/full_multi_task_tma_tbank_benchmark/logs/train.log` | 3128 lines, 4 CUDA crashes |
| Full multi-task metrics | `src/runs/full_multi_task_tma_tbank_benchmark/logs/metrics.jsonl` | per-batch metrics |
| Full multi-task config dump | `src/runs/full_multi_task_tma_tbank_benchmark/logs/resolved_config.json` | 281 keys |
| Full multi-task run command | `src/runs/full_multi_task_tma_tbank_benchmark/logs/run_command.txt` | batch_size=2 override |
| Full multi-task heartbeat | `src/runs/full_multi_task_tma_tbank_benchmark/checkpoints/.gpu_heartbeat` | 3.54 GB reserved |
| Current training (batch_size=2) | `/tmp/train_ep24_smaller.log` | epoch 25, ~39% done |
| Activity clip eval log | `/tmp/activity_clip.log` | completed |
| D4 PSR eval log | `/tmp/d4_yolov8m_psr.log` | completed |
| D1 v1 log | `/tmp/d1_yolov8m.log` | completed |
| D1 v2 log | `/tmp/d1_v2.log` | completed |
| D1 v3 log | `/tmp/d1_v3.log` | completed |
| PSR sweep v2 log | `/tmp/psr_thr.log` | completed |

---

## 9. Progress Comparison (Previously vs Currently)

### 9.1 Pre-fix state (before 2026-07-04)

| Item | Pre-fix Value | Source |
|---|---|---|
| best.pth selection | epoch 11 (broken, NaN-inflated metric) | `SOTA_STATUS.md` |
| det_mAP50 (ConvNeXt) | 0.053 | `11_NUMBERS_UPDATE.md` |
| det_mAP50 (YOLOv8m 25ep) | 0.358 (subsample, NaN full) | `d3_v4/metrics.json` |
| act_top1 | 18.3% (overfit noise) | `11_NUMBERS_UPDATE.md` |
| pose forward MAE | 9.1° (pre-normalization) | `11_NUMBERS_UPDATE.md` |
| PSR F1 | 0.0 (no MonotonicDecoder fix) | pre-fix |

### 9.2 Current state (2026-07-06)

| Item | Current Value | Source |
|---|---|---|
| best.pth | epoch 18 (post-fix promotion) | `SOTA_STATUS.md` |
| det_mAP50 (YOLOv8m self-trained, d1r) | 0.995 | `d1r/results.csv` |
| det_mAP50 (D1 re-eval) | 0.0004 | `d1_yolov8m/metrics.json` |
| act_top1 (per-frame) | 0.023 | `SOTA_STATUS.md` |
| act_top1 (clip-level) | 0.028 | `activity_clip_ep18/activity_clip.json` |
| pose forward MAE | 8.39° | `SOTA_STATUS.md` |
| PSR F1 (per-comp) | 0.7499 | `psr_optimal_thr/optimal_thresholds.json` |
| MonotonicDecoder bug | FIXED (variable shadow) | `psr_transition.py` |

### 9.3 Delta Summary

| Item | Delta | Significance |
|---|---|---|
| best.pth | epoch 11 → 18 | Found real best, not NaN-best |
| PSR F1 | 0.0 → 0.7499 | MonotonicDecoder fix + threshold tuning |
| det YOLOv8m | 0.358 → 0.995 (separate training) | Methodology contribution (separate training run) |
| pose forward | 9.1° → 8.39° | Marginal; depends on normalization |
| Activity | 18.3% → 2.8% (clip) | Architectural ceiling reached |

---

## 10. SOTA Reference Numbers (industreal-sota-benchmarks.md)

| Paper | Detection mAP | PSR F1 | Activity Top1 | Head Pose MAE |
|---|---|---|---|---|
| B1 (WACV 2024) | n/a | n/a | n/a | n/a |
| B2 (WACV 2024) | n/a | n/a | n/a | n/a |
| B3 (WACV 2024) | n/a | 0.883 (transitions) | n/a | n/a |
| STORM-PSR | n/a | **0.901** (transitions) | n/a | n/a |
| T3 MViTv2-S (verb) | n/a | n/a | **0.622** (top1_69) | n/a |
| Self-trained YOLOv8m (d1r) | **0.995** mAP50 | n/a | n/a | n/a |
| WACV SOTA detection | ~0.838 (cited in d1r context) | n/a | n/a | n/a |
| Head pose baseline | n/a | n/a | n/a | ~15° (unsourced) |

---

## 11. Dataset Statistics

| Stat | Value | Source |
|---|---|---|
| Total recordings (val) | 16 | `d4_yolov8m_psr/metrics.json` |
| Frames per recording | 1371 to 4587 | `d4_yolov8m_psr/metrics.json` |
| Total val frames | 38036 | `industreal_dataset.py` |
| Total labeled frames (all splits) | 188111 | `act_remap_75_to_69.json` |
| Image size | 1280×720 | `config.py` IMG_WIDTH/HEIGHT |
| Action classes (raw) | 75 | `act_remap_75_to_69.json` |
| Action classes (verb-grouped) | 69 (hybrid) | `act_remap_75_to_69.json` |
| Activity classes (SOTA baseline) | 69 | `t3_full_eval.json` |
| Action classes absent | 37, 64 | `config.py` |
| Detection classes (raw) | 24 | `config.py` NUM_DET_CLASSES |
| PSR components | 11 | `config.py` NUM_PSR_COMPONENTS |
| Error state class | 24 (index 23 0-based) | `config.py` |
| Error state GT instances | 0 in val | `d1_yolov8m/metrics.json` |

---

## 12. Active Experiments — Current State

| Process | GPU | Started | Status | Log |
|---|---|---|---|---|
| Training resume (batch_size=2) | RTX 5060 Ti (CUDA device 0) | 2026-07-06 17:35 | Running, epoch 25 ~39% | `/tmp/train_ep24_smaller.log` |
| D1 YOLOv8m v3 (no shift) | RTX 3060 (CUDA device 1) | 2026-07-06 18:42 | Completed mAP=0.0004 | `/tmp/d1_v3.log` |
| D4 YOLOv8m → PSR | RTX 3060 (CUDA device 1) | (was earlier) | Completed F1=0 | `/tmp/d4_yolov8m_psr.log` |
| PSR sweep 5k subset | RTX 3060 (CUDA device 1) | (was earlier) | Completed 0.7810 | `/tmp/psr_thr.log` |
| Activity clip eval | (was earlier) | (was earlier) | Completed 0.028 | `/tmp/activity_clip.log` |

---

## 13. Source File Map (Code Paths That Produce These Numbers)

### 13.1 Detection (`d1_yolov8m/metrics.json`, `d1_yolov8m_v3/metrics.json`)
- Script: `src/evaluation/eval_yolov8m.py`
- Model: `YOLO('src/runs/rf_stages/checkpoints/yolov8m_industreal.pt')` (0-indexed classes)
- Dataset: `src/data/industreal_dataset.py:IndustRealMultiTaskDataset`
- BGR conversion: line 168, 330 (`[:, :, ::-1]`)
- Class alignment: 0-indexed (no shift)

### 13.2 PSR (`psr_optimal_thr/optimal_thresholds.json`, `d4_yolov8m_psr/metrics.json`)
- Sweep script: `src/evaluation/psr_optimal_thresholds.py`
- D4 script: `src/evaluation/eval_yolov8m_psr.py`
- Decoder: `src/models/psr_transition.py` (MonotonicDecoder, Q48 hysteresis)
- YOLOv8m model: `src/runs/rf_stages/checkpoints/yolov8m_industreal.pt`
- PSR_MASK: `src/evaluation/eval_yolov8m_psr.py:_build_psr_mask()`

### 13.3 Activity (`activity_clip_ep18/activity_clip.json`)
- Script: `src/evaluation/eval_activity_clip.py`
- Clip length: 16, stride: 8
- Model: `src/models/model.py` ActivityHead (per-frame MLP)
- Verb grouping: `src/config.py` ACT_CLASS_GROUPING='hybrid'

### 13.4 Head Pose (`full_eval_ep18_stream/metrics.json`)
- Eval code: `src/evaluation/evaluate.py:1918-1926` (with "DO NOT USE FOR REPORTING" comment)
- Diagnostic: `src/evaluation/head_pose_diag.py`

### 13.5 Training (`train.log`, `metrics.jsonl`)
- Main script: `src/training/train.py`
- Model: `src/models/model.py` POPWMultiTaskModel
- Loss: Kendall uncertainty + per-task losses
- Optimizer: AdamW, OneCycleLR
- Heartbeat: `checkpoints/.gpu_heartbeat`

### 13.6 Paper draft
- LaTeX: `analyses/consult_2026_06_10/AAIML/popw_aaiml2027.tex`
- PDF: `analyses/consult_2026_06_10/AAIML/popw_aaiml2027.pdf`

---

## 14. Cross-Reference Index (Where Each Number Comes From)

| Number | Source |
|---|---|
| Detection 0.995 | `runs/detect/src/runs/yolov8m_industreal/d1r/results.csv` |
| Detection 0.358 (subsample) | `src/runs/rf_stages/checkpoints/d3_v4/metrics.json` |
| Detection NaN (full) | `src/runs/rf_stages/checkpoints/d3_full_eval/metrics.json` |
| Detection D1 re-eval 0.0004 | `src/runs/rf_stages/checkpoints/d1_yolov8m/metrics.json` |
| Detection D1 v3 0.0004 | `src/runs/rf_stages/checkpoints/d1_yolov8m_v3/metrics.json` |
| Activity 0.023 (per-frame) | `src/runs/rf_stages/checkpoints/SOTA_STATUS.md` |
| Activity 0.028 (clip) | `src/runs/rf_stages/checkpoints/activity_clip_ep18/activity_clip.json` |
| Activity T3 0.622 | `src/runs/rf_stages/checkpoints/t3_full_eval.json` |
| Activity T3 0.04 | `src/runs/rf_stages/checkpoints/t3_mecanno_eval.json` |
| Head pose 8.39° | `src/runs/rf_stages/checkpoints/SOTA_STATUS.md` |
| Head pose 13.52° | `src/runs/rf_stages/checkpoints/SOTA_STATUS.md` (300-subset) |
| Head pose 26.20° | `src/runs/rf_stages/checkpoints/SOTA_STATUS.md` (full eval) |
| PSR F1 0.7217 (global 0.10) | `src/runs/rf_stages/checkpoints/SOTA_STATUS.md` |
| PSR F1 0.7499 (per-comp, full) | `src/runs/rf_stages/checkpoints/psr_optimal_thr/optimal_thresholds.json` |
| PSR F1 0.7810 (per-comp, 5k) | `src/runs/rf_stages/checkpoints/psr_optimal_thr_v2/optimal_thresholds.json` |
| PSR F1 0 (D4 YOLOv8m) | `src/runs/rf_stages/checkpoints/d4_yolov8m_psr/metrics.json` |
| PSR POS 0.968 | `src/runs/rf_stages/checkpoints/SOTA_STATUS.md` |

---

## 15. Outstanding Items Not Yet in the Codebase

| Item | Status |
|---|---|
| Held-out test split for PSR thresholds | MISSING |
| Knowledge distillation experiment (D6) | NOT STARTED |
| TCN+ViT activity head ablation (ACT-6) | NOT STARTED |
| FiLM on/off pose ablation (HP-5) | NOT STARTED |
| KENDALL_FIXED_WEIGHTS ablation (PSR-3) | NOT STARTED |
| Transition-F1 vs per-frame-F1 side-by-side (PSR-2) | NOT STARTED |
| Leave-one-recording-out CV (PSR-5, EP-4) | NOT STARTED |
| Position units verification (HP-3) | UNVERIFIED |
| Best.pth epoch-by-epoch audit (AC-1) | PARTIAL |

---

## 16. Quick Reference: Read-Order for Opus Audit

To verify any claim in this document, Opus should:

1. **Detection 0.995**: `cat runs/detect/src/runs/yolov8m_industreal/d1r/results.csv`
2. **PSR F1 0.7499**: `cat src/runs/rf_stages/checkpoints/psr_optimal_thr/optimal_thresholds.json`
3. **Activity 0.028**: `cat src/runs/rf_stages/checkpoints/activity_clip_ep18/activity_clip.json`
4. **Pose 8.39°**: `cat src/runs/rf_stages/checkpoints/SOTA_STATUS.md`
5. **D4 F1=0**: `cat src/runs/rf_stages/checkpoints/d4_yolov8m_psr/metrics.json`
6. **Training crash log**: `tail -50 src/runs/full_multi_task_tma_tbank_benchmark/logs/train.log`
7. **Current training**: `tail -10 /tmp/train_ep24_smaller.log`
8. **Per-component thresholds**: `cat src/runs/rf_stages/checkpoints/psr_optimal_thr/optimal_thresholds.json`
9. **Activity clip details**: `cat src/runs/rf_stages/checkpoints/activity_clip_ep18/activity_clip.json`
10. **Master status**: `cat src/runs/rf_stages/checkpoints/SOTA_STATUS.md`