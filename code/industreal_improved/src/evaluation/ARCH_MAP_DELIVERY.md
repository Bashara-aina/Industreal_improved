# POPW Metric Code Architecture Map
**Author**: metric-scout (deliverable to metric-investigator)
**Date**: 2026-06-04
**Scope**: complete map of `src/evaluation/` metric path, postprocess funcs, metric compute funcs, and config thresholds

---

## 1. File Layout

| File | Lines | Role |
|------|------:|------|
| `src/evaluation/evaluate.py` | 3886 | All metric logic (single entry: `evaluate_all()` at L2637) |
| `src/evaluation/metrics.py` | 202 | Lightweight per-batch dispatcher (calls evaluate.py's funcs) |
| `src/evaluation/__init__.py` | 0 | empty |
| `src/config.py` | ~700 | Thresholds and constants referenced by evaluate.py |

`metrics.py` is **NOT** the live metric path. It is a single-batch dispatcher used only by smoke/unit tests. Production `evaluate_all()` lives in `evaluate.py`.

---

## 2. Main Entry: `evaluate_all()`

`src/evaluation/evaluate.py:2637-3366`

Signature:
```python
@torch.no_grad()
def evaluate_all(
    model, criterion, loader, device,
    max_batches: int = 2500,
    save_dir: Optional[str] = None,
    use_flip_tta: bool = False,
    use_crop_tta: bool = False,
    epoch: int = 0,
) -> Dict[str, Any]
```

Flow (line by line):
- L2665 `model.eval()`
- L2667 `C._CURRENT_EPOCH = epoch` (gates efficiency metric)
- L2672 `_get_dl_osa_numba()` pre-warms numba JIT
- L2678-2726 crash-safe CUDA health check + 5-second timeout save
- L2742-2755 init accumulators
- L2756-2998 **per-batch loop**:
  - L2766 `_prepare_images()` (uint8->float, ImageNet norm)
  - L2771-2783 PSR cache reset on recording boundary
  - L2785-2799 move targets to device
  - L2812 `outputs_raw = model(images, clip_rgb=...)`
  - L2815-2820 optional flip-TTA
  - L2823-2842 optional 5-crop TTA (224x224)
  - L2844-2850 cast all outputs to float32
  - L2852 compute loss
  - L2858-2890 collect act_logits, act_pred (argmax), rec_id, frame_num
  - L2892-2903 collect head_pose
  - L2905-2907 collect psr_logits
  - L2910-2987 **detection postprocess** (per-image, in-batch):
    - sigmoid cls_preds
    - score_thresh = `C.DET_EVAL_SCORE_THRESH` (0.03)
    - top-`C.DET_EVAL_MAX_PER_IMAGE` (300)
    - `decode_boxes()` on kept anchors
    - clip to IMG_WIDTH x IMG_HEIGHT
    - per-class `nms_numpy()` with `C.DET_EVAL_NMS_IOU_THRESH` (0.5)
  - L2917 `probe_detection_batch()` diagnostic (first 5 batches)
  - L2992-2997 crash checkpoint every 5 batches
- L3059 `results = {'loss': total_loss/lc}`
- L3074-3125 **Activity metrics** -> `compute_activity_metrics()`
- L3130-3147 **Head pose metrics** -> `compute_head_pose_metrics()`
- L3152-3193 **PSR metrics** -> `compute_psr_metrics(tolerance_frames=3)` (also computes t=5)
- L3199-3228 **Assembly State metrics** -> `compute_assembly_state_metrics()` (Paper 8)
- L3233-3241 **Error Verification metrics** -> `compute_error_verification_metrics()` (Paper 9)
- L3249-3295 **Detection metrics** (skipped if `C.SKIP_DET_METRICS_EVAL=True`):
  - `compute_det_metrics_extended()` -> mAP@0.5 + mAP@[0.5:0.95]
  - `compute_det_metrics_all_frames()` -> full-video mAP
- L3303-3330 **Efficiency metrics** (skipped unless `epoch % C.LOG_EFFICIENCY_EVERY == 0`)
- L3341 `model.train()` (restore)
- L3343-3349 metric aliases
- L3351-3359 **NaN guard**: replace any remaining NaN/Inf with 0.0
- L3362-3364 save JSON + CSV if save_dir
- L3366 return results

Crash-safety: returns safe fallback dict if all batches are empty (L3040-3051).

---

## 3. Detection Postprocess Chain (in evaluate.py)

Per-image (L2925-2986):
1. `cls_sigmoid = sigmoid(outputs['cls_preds'])` -> `[B, N, 24]` on GPU
2. `max_scores = cls_sigmoid[i].max(dim=1).values` -> `[N]`
3. `keep_mask = max_scores > C.DET_EVAL_SCORE_THRESH` (0.03)
4. Optional `topk` to `C.DET_EVAL_MAX_PER_IMAGE` (300)
5. `decode_boxes(anchors[keep], reg_preds[i][keep])` -> `[K, 4]`
6. Clip to `(0, 0, IMG_WIDTH, IMG_HEIGHT)` = (0,0,1280,720)
7. Per-class NMS at IoU 0.5

Helpers (lines 1044-1088):
- `compute_iou_matrix(a, b)` -- pairwise IoU
- `decode_boxes(anchors, deltas)` -- anchor + delta -> xyxy, clips deltas to +/-4
- `nms_numpy(boxes, scores, iou_thresh=0.5)` -- pure numpy NMS

Inputs:
- `outputs['cls_preds']` -- `[B, N, 24]` raw logits
- `outputs['reg_preds']` -- `[B, N, 4]` (dx, dy, dw, dh)
- `outputs['anchors']` -- `[N, 4]` xyxy, populated from `AnchorGenerator` (model.py:435)
- `targets['detection'][i]['boxes']`, `['labels']` -- per-image GT

---

## 4. Activity Postprocess Chain

Per-batch (L2858-2890):
1. `act_logits = outputs['act_logits'].cpu().numpy()` -> `[B, 75]`
2. `act_pred = argmax(axis=1)` -> `[B]` class indices
3. GT `act_labels = targets['activity'].cpu().numpy()` (raw IDs are 1:1 with class indices 0=NA, 1-74)
4. Per-frame: collect `recording_id` (for clip-level metric) and `frame_num` (for 16-uniform-frame protocol)

No score threshold; pure argmax. Class 0 = NA (prepended by config; not predicted explicitly during training but accepted at metric time).

---

## 5. Head Pose Postprocess Chain

Per-batch (L2892-2903):
1. `outputs['head_pose']` is `[B, 9]` = `[forward(3), pos(3), up(3)]`
2. None fallback: zeros `[B, 9]`
3. Collect raw pred + GT arrays; concat across batches -> `[N, 9]`

Unit-vector detection (L1505-1510) decides metric branch:
- Compute mean norm of pred/GT forward (cols 0-2) and up (cols 6-8)
- If both pred and GT have mean norm > 0.5 -> angular MAE in degrees
- Otherwise -> raw MAE with `_deg` keys set to NaN + `*_raw_MAE` keys populated

Position (cols 3-5): L1 error in metres, multiplied by 1000 -> mm. Comment in code says "may produce meaningless values" -- units unverified.

---

## 6. PSR Postprocess Chain

Per-batch (L2905-2907):
1. `outputs['psr_logits']` is `[B, 11]` raw logits
2. Collect raw logits + GT labels (`-1` = unknown/error) across batches -> `[N, 11]`

`compute_psr_metrics()` at L2090-2217:
- Sigmoid -> probability, threshold at 0.5 -> binary `[N, 11]`
- For each of 11 components:
  - Per-component F1 (TP/FP/FN)
  - Macro-averaged overall F1
- F1@T: GPU-fused adjacency matrix, computes BOTH t=3 AND t=5 in single pass (L2161-2190)
- Edit Score: DL-OSA on int8 binary sequences per component (L2196), numba JIT for |seq|>=5000
- POS (Percentage of Ordering Success): vectorized run-based adjacent-pair check (L2199)

GT handling: `-1` in any component -> masked via `valid_mask = gt_labels != -1`, replaced with 0 for matching.

---

## 7. Assembly State Postprocess Chain (Paper 8)

`compute_assembly_state_metrics()` at L2270-2377:
1. Build state vocabulary from GT 11-D patterns (`_build_state_vocabulary`, L2234-2245)
2. Convert GT patterns to state IDs via `_psr_to_state_id` (L2224-2231); `-1` patterns treated as 0
3. Convert pred logits to state IDs via `_psr_logits_to_state_ids` (L2248-2267) -- sigmoid->binary->vocab lookup, unknown patterns -> state_id=K
4. Frame-level top-1 accuracy over valid GT frames
5. Macro F1 over all K states (`sklearn.f1_score`)
6. MAP@R(+): for each GT transition, check predicted state in window [t-R, t+R] intersect [start, next_transition)

Tolerances:
- `tolerance_frames=3` (default arg, not configurable from `C`)

---

## 8. Error Verification Postprocess Chain (Paper 9)

`compute_error_verification_metrics()` at L2384-2486:
1. `error_score = 1 - sigmoid(psr_logits).max(axis=1)` per frame (low confidence -> high error)
2. `gt_error = (gt_labels < 0).any(axis=1)` (any component = -1 -> error frame)
3. AP via sorted recall-level interpolation (L2461-2468)
4. F1/P/R at threshold 0.5 (L2470-2479)

Critical: only frames with `valid_mask = (gt_labels >= 0).any(axis=1)` are counted. Frames where ALL components are -1 are excluded.

---

## 9. Efficiency Postprocess Chain

`compute_efficiency_metrics()` at L2502-2629:
- `total_params` / `trainable_params` via `sum(p.numel() for p in model.parameters())`
- GFLOPs via `thop.profile()` (optional, `_THOP_AVAILABLE` flag, L2496-2499)
- Batched FPS: 5 warmup + 30 timed `model(...)` calls with random dummy
- Streaming FPS: 1 cold frame + (timed_runs-1) warm frames with same `video_id` (FeatureBank hit)
- Multi-model pipeline (YOLOv8m + MViTv2 + STORM-PSR): static hardcoded estimates L2612-2615

Gated in `evaluate_all()` by `C.SKIP_EFFICIENCY_METRICS` and `C.LOG_EFFICIENCY_EVERY` (L3303-3306).

---

## 10. Metric Compute Functions (all in evaluate.py)

| Function | Line | Role |
|---|---:|---|
| `_compute_clip_level_accuracy` | 627 | 16-uniform-frame majority-vote accuracy |
| `compute_activity_metrics` | 725 | frame_acc, no-NA, macro-F1, weighted-F1, top-5, per-class, confusion matrix |
| `report_per_class_accuracy` | 857 | log worst/best K classes |
| `_save_confusion_matrix` | 888 | PNG via seaborn |
| `_save_per_class_f1_csv` | 916 | per-class CSV sorted by F1 |
| `_plot_topk_bottomk_classes` | 963 | bar chart best/worst |
| `compute_iou_matrix` | 1044 | pairwise IoU |
| `decode_boxes` | 1055 | anchor+delta -> xyxy |
| `nms_numpy` | 1069 | pure-numpy NMS |
| `compute_ap_per_class` | 1091 | per-class AP, COCO or VOC interp |
| `_coco_ap` | 1150 | 101-point COCO AP |
| `compute_ap_per_class_all_frames` | 1164 | per-class AP with empty-frame handling |
| `compute_ap_multi_thresh` | 1237 | vectorized single-pass multi-threshold AP (~9x speedup) |
| `compute_det_metrics_extended` | 1364 | mAP@0.5 + mAP@[0.5:0.95] |
| `compute_det_metrics_all_frames` | 1402 | full-video mAP |
| `compute_head_pose_metrics` | 1433 | per-DoF MAE, angular MAE deg, position MAE mm |
| `_damerau_levenshtein` (string) | 1547 | DL-OSA (string version, unused in PSR path) |
| `_symmetric_prf_at_t_cuda` | 1580 | GPU P/R/F1 at +/-T |
| `_symmetric_prf_at_t` (dict) | 1631 | CPU P/R/F1 at +/-T (dict-based) |
| `_get_dl_osa_numba` | 1714 | lazy-load numba JIT DL-OSA |
| `_levenshtein_on_intarrays` | 1756 | O(min(m,n))-space Wagner-Fischer on int arrays |
| `_damerau_levenshtein_on_intarrays_osa` | 1789 | dispatches to numba if |seq|>=5000 |
| `_compute_psr_edit_score_vectorized` | 1810 | per-component Edit Score (DL-OSA) |
| `_symmetric_prf_at_t_numpy` | 1862 | CPU P/R/F1 (numpy-vectorized) |
| `_compute_psr_f1_at_t_fused_cuda` | 1908 | fused GPU t=3+t=5 in single pass (~12x speedup) |
| `_compute_psr_f1_at_t_vectorized` | 1993 | CPU fallback for PSR F1@T |
| `_compute_psr_pos_vectorized` | 2028 | POS vectorized run-pair ordering |
| `_symmetric_f1_at_t` | 2085 | backward-compat alias |
| `compute_psr_metrics` | 2090 | PSR master (calls all helpers) |
| `_psr_to_state_id` | 2224 | 11-D vec -> vocab state id |
| `_build_state_vocabulary` | 2234 | 11-D vec -> vocab dict |
| `_psr_logits_to_state_ids` | 2248 | logits -> binary -> vocab state ids |
| `compute_assembly_state_metrics` | 2270 | Paper 8: Top-1, F1@1, MAP@R(+) |
| `compute_error_verification_metrics` | 2384 | Paper 9: AP, F1, P, R |
| `compute_efficiency_metrics` | 2502 | params, GFLOPs, FPS |
| `evaluate_all` | 2637 | main entry |
| `_serialize_for_json` | 3374 | numpy/torch -> JSON |
| `_save_results_json` | 3389 | JSON + metrics.jsonl append |
| `_save_results_csv` | 3412 | row-append to eval_results.csv |
| `_print_multi_seed_summary` | 3461 | multi-seed formatted table |
| `_print_single_run_results` | 3535 | single-run formatted table |

---

## 11. Config Thresholds (from `src/config.py`)

### Detection
| Constant | Value | Line | Used by |
|---|---|---:|---|
| `NUM_DET_CLASSES` | 24 | 118 | det head + mAP loop bound |
| `ANCHOR_SIZES` | (24, 48, 96, 192, 384) | 246 | `AnchorGenerator` (model.py:435) |
| `DET_POS_IOU_THRESH` | 0.3 | 247 | training anchor matching |
| `DET_NEG_IOU_THRESH` | 0.25 | 248 | training background anchor |
| `IMG_WIDTH` | 1280 | 249 | box clip + dummy image size |
| `IMG_HEIGHT` | 720 | 250 | box clip + dummy image size |
| `DET_EVAL_SCORE_THRESH` | 0.03 | 315 | keep mask |
| `DET_EVAL_MAX_PER_IMAGE` | 300 | 316 | top-k cap |
| `DET_EVAL_NMS_IOU_THRESH` | 0.5 | 317 | NMS |
| `SKIP_DET_METRICS_EVAL` | False | 459 | skip det mAP gate |

### mAP IoU Sweep
| Source | Value |
|---|---|
| `compute_ap_multi_thresh()` arg `iou_thresholds` | `np.arange(0.5, 1.0, 0.05)` -> [0.5, 0.55, ..., 0.95] (11 values) |
| Single-threshold call | `iou_thresh=0.5` |

### Activity
| Constant | Value | Line | Used by |
|---|---|---:|---|
| `NUM_ACT_RAW_IDS` | 74 | (computed) | 1:1 with raw action IDs |
| `NUM_CLASSES_ACT` | 75 | 162 | NA(0) + IDs 1..74 |

### PSR
| Constant | Value | Line | Used by |
|---|---|---:|---|
| `NUM_PSR_STEPS` | 36 | 233 | (not used in metric -- model output is 11-dim only) |
| `NUM_PSR_COMPONENTS` | 11 | 234 | PSR metric loop bound |
| `PSR_TEMPORAL_SMOOTH_WEIGHT` | 0.05 | 348 | training loss |
| `PSR_FOCAL_ALPHA` | 0.25 | 408 | training loss |
| `PSR_FOCAL_GAMMA` | 1.0 | 409 | training loss |
| `PSR_LOSS_CAP` | 20.0 | 375 | training cap |
| `PSR_SEQUENCE_LENGTH` | 4 | 420 | training sequence mode |

### PSR Constants (in evaluate.py)
- `tolerance_frames=3` default for `compute_psr_metrics()`; t=5 also computed in fused pass
- Sigmoid threshold = 0.5
- DL-OSA JIT trigger = `max(m,n) >= 5000`

### Assembly State
- `tolerance_frames=3` (default arg of `compute_assembly_state_metrics`, not in `C`)

### Error Verification
- Sigmoid threshold = 0.5
- AP uses sorted recall-level interpolation

### Efficiency
| Constant | Value | Line | Used by |
|---|---|---:|---|
| `LOG_EFFICIENCY_EVERY` | 10 | 456 | compute every N epochs |
| `SKIP_EFFICIENCY_METRICS` | True | 462 | only compute when `epoch % EVERY == 0` |
| `eff warmup_runs` | 5 | (arg) | dummy warmup |
| `eff timed_runs` | 30 | (arg) | FPS sample count |

### Validation
| Constant | Value | Line | Used by |
|---|---|---:|---|
| `VAL_BATCH_SIZE` | 16 | 268 | val DataLoader |
| `VAL_NUM_WORKERS` | 1 | 269 | val workers |
| `VAL_EVERY` | 1 | 281 | run every N epochs |
| `EVAL_MAX_BATCHES` | -1 | 282 | -1 = all batches |

---

## 12. Multi-Seed / Multi-Run Orchestration

- `run_multi_seed_evaluation()` at L434-534 -- outer loop over seeds, calls `evaluate_all()` per seed
- `print_ablation_table()` at L537-596 -- prints baseline->full ablation table
- `_ablate_component()` at L599-606 -- stub (returns full_results unchanged)

---

## 13. Model Output Keys Consumed by `evaluate_all()`

From `model.py:1941-1956`:
```python
return {
    'cls_preds': cls_preds,        # [B, N, 24] raw logits
    'reg_preds': reg_preds,        # [B, N, 4] dx,dy,dw,dh
    'anchors': anchors,            # [N, 4] xyxy
    'heatmaps': heatmaps,          # [B, 24, H, W] (not used in metric path)
    'keypoints': keypoints,        # (not used in metric)
    'pose_confidence': pose_confidence,  # (not used in metric)
    'head_pose': head_pose,        # [B, 9] = [forward3, pos3, up3]
    'c5_mod': c5_mod,              # (not used in metric)
    'det_conf': det_conf,          # (not used in metric)
    'act_logits': act_logits,      # [B, 75]
    'psr_logits': psr_logits[..., :11],  # [B, 11] for PSR/AS/EV
    'psr_confidence': psr_confidence if not self.training else None,
    'temporal_features': bank_output,
    'c5_raw': c5,
}
```

Note: PSR confidence, heatmaps, keypoints, pose_confidence, c5_mod, det_conf, temporal_features, c5_raw are returned by the model but **not consumed by `evaluate_all()`** -- kept for potential future use.

---

## 14. Robustness / Crash Safety

1. L2678-2726 CUDA health check; `_save_eval_crash_recovery()` daemon thread with 5s timeout.
2. L2728-2740 GPU + CPU memory snapshot at start.
3. L2761-2764 per-10-batch memory log.
4. L2917 `probe_detection_batch()` for first 5 batches (verdict: TOTAL COLLAPSE / NEAR / LOCALIZING).
5. L2982-2986 per-image GPU cache release after detection postprocess.
6. L2988-2989 `gc.collect()` after each batch.
7. L2992-2997 crash checkpoint every 5 batches.
8. L3013-3051 empty-Dataloader guard -- returns safe fallback dict.
9. L3303-3330 efficiency skip gate -- respects `LOG_EFFICIENCY_EVERY`.
10. L3351-3359 final NaN/Inf -> 0.0 before return.
11. L3341 `model.train()` restoration at end.
12. L3362-3364 results saved to JSON + CSV if save_dir.

---

## 15. Critical Files for Investigator

- `/media/newadmin/master/POPW/working/code/industreal_improved_to_archive/src/evaluation/evaluate.py` -- main entry (3886 lines)
- `/media/newadmin/master/POPW/working/code/industreal_improved_to_archive/src/evaluation/metrics.py` -- single-batch dispatcher (unused in production)
- `/media/newadmin/master/POPW/working/code/industreal_improved_to_archive/src/config.py` -- thresholds (lines 118, 162, 246-251, 315-317, 459-462, 268-282)
- `/media/newadmin/master/POPW/working/code/industreal_improved_to_archive/src/models/model.py:1941-1956` -- output dict shape
- `/media/newadmin/master/POPW/working/code/industreal_improved_to_archive/src/POPW_EVAL_LOOP_MASTER_PROMPT.md` -- historical metric-loop bug context
- `/media/newadmin/master/POPW/working/code/industreal_improved_to_archive/src/quick_eval.py` -- minimal runner (loads crash_recovery.pth, runs evaluate_all with max_batches=4)

---

## 16. Metric Output Keys (for train.py / combined-score)

Returned by `evaluate_all()`:
```
loss, det_mAP50, det_mAP_50_95, det_per_class_ap,
det_mAP50_all_frames, det_per_class_ap_all_frames,
act_accuracy, act_frame_accuracy, act_accuracy_no_na,
act_macro_f1, act_macro_f1_present, act_weighted_f1,
act_macro_recall, act_mean_per_class_acc, act_top5_accuracy,
act_per_class_acc, act_per_class_report, act_confusion_matrix,
act_clip_accuracy, _ar_baseline_protocol,
head_pose_MAE, head_pose_MAE_std, head_pose_angular_MAE_deg,
forward_angular_MAE_deg, up_angular_MAE_deg, head_pose_status,
forward_raw_MAE, up_raw_MAE,  # only when non-unit vectors
position_MAE_mm,
psr_overall_f1, psr_f1_at_t, psr_precision_at_t, psr_recall_at_t,
psr_f1_at_t5, psr_precision_at_t5, psr_recall_at_t5,
psr_edit_score, psr_pos, psr_per_component_f1,
psr_num_valid_components, psr_num_samples,
_psr_edit_protocol, _psr_f1_at_t_protocol, _psr_pos_protocol,
psr_macro_f1, psr_overall_f1_at5,  # aliases for train.py compat
as_f1, as_top1_accuracy, as_map_at_r, as_num_states, as_num_transitions,
ev_ap, ev_f1, ev_precision, ev_recall,
assembly_state_f1, error_detection_f1,  # aliases
eff_params_m, eff_trainable_params_m, eff_gflops,
eff_fps, eff_fps_streaming, eff_batch_size, eff_resolution,
pipeline_params_m, pipeline_gflops, pipeline_fps
```

All NaN/Inf values are replaced with 0.0 before return (L3351-3359).
