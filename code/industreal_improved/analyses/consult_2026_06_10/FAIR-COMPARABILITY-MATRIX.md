# FAIR COMPARABILITY MATRIX — Every POPW Metric vs. Every SOTA Paper

**Date:** 2026-07-05
**Principle:** Every metric must use the SAME computation protocol as the SOTA it compares against. No decoder-smoothed inflatables. No paradigm mismatches.

---

## ✅ FAIR NOW — same protocol, same metric

| Our Metric | SOTA Metric | SOTA Value (Source) | Protocol Match |
|---|---|---|---|
| `det_mAP50` | mAP@0.5 (WACV 2024 Tab 3) | 0.838 (YOLOv8m) | ✅ Both COCO 101-point, 24-class ASD |
| `det_mAP_50_95` | mAP@[.5:.95] (WACV 2024) | not reported | ✅ Both COCO standard |
| `psr_pos_raw_t05` (NEW) | POS (STORM-PSR Tab 1) | 0.812 | ✅ Raw binary at 0.5, no decoder |
| `psr_f1_raw_t05` (NEW) | F1@±3 (STORM-PSR Tab 1) | 0.901 | ✅ Raw transitions, ±3 frame tolerance |
| `psr_edit_raw_t05` (NEW) | Edit (STORM-PSR) | not reported | ✅ Raw binary, normalized DL |

## ⚠️ FAIR AFTER CODE — same protocol, already coded

| Our Metric | SOTA Metric | SOTA Value | Protocol matched by |
|---|---|---|---|
| act_macro_f1 | Macro F1 (MViTv2) | 0.452 (75-class) | T3 remap 75→69 (1-day protocol change) |
| act_top1_raw | Top-1 (MViTv2) | 65.25% (75-class, 16-frame) | T3 remap + temporal clip vote |

## ❌ DIFFERENT PARADIGM — never directly comparable

| Our Metric | SOTA Metric | Why never comparable |
|---|---|---|
| `act_frame_accuracy` | MViTv2 Top-1 65.25% | Per-frame vs 16-frame clip. 69-class vs 75-class. Different pretraining. |
| `psr_tau` (per-frame delay) | STORM τ 15.5s (transition delay) | Per-frame state change vs transition event detection. |
| `forward_angular_MAE_deg` | No SOTA exists | First ego-pose baseline for IndustReal |

## ✅ NEW — no SOTA exists, original contribution

| Our Metric | Contribution |
|---|---|
| `det_mAP50_pc` | Present-class companion metric (honest, no SOTA equivalent) |
| `act_top1` (per-frame, 69-class) | First per-frame baseline for verb-grouped protocol |
| `psr_pos_blind` | Canonical-order POS baseline (Q43, G4 PASS) |
| `psr_comp_acc` | Per-frame component accuracy (no published equivalent) |
| `eff_fps` / `eff_params_m` / `eff_gflops` | Single-GPU efficiency measurement |

---

## The 5 honest disclosures for every comparison

1. **PSR**: POS/F1/Edit reported from raw per-frame binary at threshold=0.5 (no MonotonicDecoder). The fill-forward decoder inflates POS artificially; the raw numbers are the SOTA-comparable ones.
2. **Detection**: COCO mAP same protocol. The YOLOv8m 0.838 uses IndustReal-trained weights (unavailable). D1-R retrains YOLOv8m on our split; D1 COCO-pretrained gave mAP=0 (class taxonomy mismatch).
3. **Activity**: Per-frame 69-class vs MViTv2 16-frame 75-class. The closest fair comparison requires T3 remap of MViTv2 to 69-class protocol.
4. **Ego-pose**: First IndustReal baseline. No SOTA comparison exists. Forward and up vectors now unit-normalized.
5. **Efficiency**: 46.47M params, 245.3 GFLOPs, 11.02 FPS at batch=1 720×1280. 67% smaller than 4 dedicated models.

---

## The 2 code changes that made metrics fair

### 1. PSR raw metrics (evaluate.py, 2026-07-05)
```python
pred_binary = (sigmoid(logits) > 0.5).astype(int)  # NO MonotonicDecoder
psr_pos_raw = _compute_psr_pos_vectorized(pred_binary, gt, valid)  # honest POS
psr_f1_raw = _compute_psr_f1_at_t_vectorized(pred_binary, gt, valid, 3)  # honest F1
```

### 2. PSR raw at threshold=0.3 (captures the 1.6% of frames below 0.3)
```python
pred_binary_t03 = (sigmoid(logits) > 0.3).astype(int)
psr_f1_raw_t03 = _compute_psr_f1_at_t_vectorized(pred_binary_t03, gt, valid, 3)
```

---

## Remaining 2 protocol alignments needed (separate training runs)

| Fix | Time | Status |
|---|---|---|
| D1-R YOLOv8m retrain on our split | 21h | Running (GPU 0) |
| T3 MViTv2 remap 75→69 + T2 temporal head | 1 day + 3-4 days | Queued (Week 2) |

---

## The paper's honest comparison table (what reviewers will see)

| Task | Metric | POPW (raw) | SOTA | Δ | SOTA Source |
|---|---|---|---|---|---|
| Detection | mAP@0.5 | 0.358 (subsample) / TBD (full) | 0.838 (YOLOv8m) | TBD | WACV 2024 Tab 3 |
| Activity | Macro-F1 | 0.205 (subsample, 69-class per-frame) | TBD (T3 remap, same protocol) | TBD | WACV 2024 Tab 2 (remapped) |
| Ego-pose | fwd MAE (deg) | 7.83° (subsample) | — | — | FIRST BASELINE |
| PSR | POS | TBD (raw_t05) | 0.812 (STORM) | TBD | STORM-PSR Tab 1 |
| PSR | F1@±3 | TBD (raw_t05) | 0.901 (STORM) | TBD | STORM-PSR Tab 1 |
| PSR | Edit | TBD (raw_t05) | not reported | — | — |
| Efficiency | FPS | 11.02 (measured) | ~15 (4 separate models) | — | Measured vs estimated |

**The PSR and Ego-pose rows are the paper's strongest contributions.** Detection and Activity are "in-progress on protocol alignment" disclosures.
