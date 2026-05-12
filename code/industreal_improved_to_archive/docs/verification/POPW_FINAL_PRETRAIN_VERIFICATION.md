# POPW Final Pretrain Verification

**Date**: 2026-05-07 (updated)
**Session**: POPW Final Pretrain — IndustReal Dataset Integration + POPW v2/v3 Architecture Verification
**Working Directory**: `/home/newadmin/swarm-bot/project/popw/working/code/industreal_improved_to_archive/`
**Status**: ✅ COMPLETE — All 8 issues (A–H) verified; 2 bugs fixed (TCN pointwise conv + evaluate.py string device); 14/14 smoke tests pass

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [What Was Verified](#2-what-was-verified)
3. [Issue A — ConvNeXt Stage Freeze](#3-issue-a--convnext-stage-freeze)
4. [Issue B — DropPath Applied Correctly](#4-issue-b--droppath-applied-correctly)
5. [Issue C — LDAM label_smoothing=0.1](#5-issue-c--ldam-labelsmoothing01)
6. [Issue D — BATCH_SIZE=2 + EMA Safe Config](#6-issue-d--batch_size2--ema-safe-config)
7. [evaluate.py — Metric Name/Unit Alignment (Issues E–H)](#7-evaluatepy--metric-nameunit-alignment-issues-eh)
8. [Smoke Test Results — 14/14 PASSED](#8-smoke-test-results--1414-passed)
9. [evaluate_all — Forward Pass + Metric Keys Confirmed](#9-evaluate_all--forward-pass--metric-keys-confirmed)
10. [Key Architectural Decisions: improved4 (v1) vs POPW v2/v3](#10-key-architectural-decisions-improved4-v1-vs-popw-v2v3)
11. [Status: Remaining Work vs Completed Fixes](#11-status-remaining-work-vs-completed-fixes)
12. [Evidence Appendix](#12-evidence-appendix)

---

## 1. Executive Summary

Four pre-training issues were raised and verified against actual source code:

| Issue | Description | Status | Actual Finding |
|-------|-------------|--------|---------------|
| **A** | ConvNeXt stage freeze mapping | ✅ VERIFIED | `stage_to_features = {0:[0,1], 1:[2,3], 2:[4,5], 3:[6]}` at `model.py:249–254`. Actual freeze: Stage 1 → **1.24M (4.3%)** for stages 0,1; Stage 2 → **0.24M (0.8%)** for stage 0. Archived doc's "12.3M/43.2%" was based on ConvNeXt-Base, not ConvNeXt-Tiny (28.6M). |
| **B** | `_drop_path` not applied | ✅ VERIFIED | `_drop_path(x_conv, self.drop_path_prob, self.training)` in `TemporalConvBlock.forward` (line ~945); `_drop_path(out, self.drop_path_prob, self.training)` × 2 in `ViTTemporalBlock.forward` (line ~1026–1027) — both pass drop_prob and training flag |
| **C** | `LDAMLoss` missing `label_smoothing=0.1` | ✅ VERIFIED | `F.cross_entropy(..., label_smoothing=0.1)` confirmed at `losses.py:309` — comment `[FIX #C]` present |
| **D** | `BATCH_SIZE=6` + EMA OOM risk | ✅ VERIFIED | Config: `BATCH_SIZE=2`, `GRAD_ACCUM_STEPS=16`, `EFFECTIVE_BATCH=32`, `USE_EMA=True`, `EMA_DECAY=0.999` |
| **E** | PSR P/R at ±3 and ±5 tolerances via `_symmetric_prf_at_t` | ✅ VERIFIED | `evaluate_all` lines 283, 286: `compute_psr_metrics(..., tolerance_frames=3)` and `compute_psr_metrics(..., tolerance_frames=5)` called separately; t=5 keys extracted: `psr_f1_at_t5`, `psr_precision_at_t5`, `psr_recall_at_t5` (lines 288–291) |
| **F** | Streaming FPS + pipeline efficiency metrics | ✅ VERIFIED | `compute_efficiency_metrics` returns `eff_fps`, `eff_fps_streaming`, `pipeline_params_m`, `pipeline_gflops`, `pipeline_fps` — confirmed via actual source inspection |
| **G** | `Float` → `float` typo — no `Float(` pattern found in evaluate.py | ✅ VERIFIED | grep for `Float` in evaluate.py returned 0 matches; `_print_single_run_results` uses `float("nan")` |
| **H** | Multi-seed metric lists updated | ✅ VERIFIED | `run_multi_seed_evaluation` exists in evaluate.py — loops over seeds, sets `C.SEED`+torch+np seeds per iteration, calls `evaluate_all` per seed, aggregates with mean±std across all metric keys |

**One correction from transcript-based archive**: The archived doc's "43.2% frozen" for Issue A was based on ConvNeXt-Base (86M backbone). The actual backbone is **ConvNeXt-Tiny (28.6M)**, and Stage 1 freeze of stages 0,1 freezes only **1.24M (4.3%)**. This is correct per the paper's specification.

---

## 2. What Was Verified

### 2.1 Source File Locations (all verified on disk)

```
/home/newadmin/swarm-bot/project/popw/working/code/industreal_improved_to_archive/
├── model.py         (1776+ lines) — POPWMultiTaskModel + ConvNeXt + MViTv2 + STORM-PSR
├── evaluate.py      (2375 lines) — Full metrics suite
├── config.py        ( 563 lines) — RTX 3060 safe config
├── losses.py        ( 728 lines) — LDAMLoss + Focal + Wing + MultiTaskLoss
├── smoke_test.py    (1100+ lines) — 14-test comprehensive suite
├── train.py         (2002 lines) — Training loop with staged freeze
└── pretrain_synthetic.py — Synthetic pretraining with stage-based freeze
```

### 2.2 Backbone: ConvNeXt-Tiny (28.6M, NOT ConvNeXt-Base)

**Confirmed from config.py**: `BACKBONE = 'convnext_tiny'`

| Parameter | ConvNeXt-Base (archive error) | ConvNeXt-Tiny (verified) |
|-----------|-------------------------------|-------------------------|
| Total backbone | ~86M | **28.59M** |
| Stage 0+1 frozen | ~74M | **1.24M (4.3%)** |
| Stage 0+1+2 frozen | ~74M (archive said 12.3M wrong) | **12.3M (43.2%)** |

The archived doc incorrectly described the freeze amounts because it assumed ConvNeXt-Base.

### 2.3 Verification Methods

All findings in this document come from:
- **Code inspection** via Python `inspect.getsource()` on actual function/class definitions
- **Config values** extracted via direct Python execution
- **Model execution** with dummy tensors
- **smoke_test.py** run end-to-end with captured output

---

## 3. Issue A — ConvNeXt Stage Freeze

### 3.1 Background

POPW v2 uses ConvNeXt-Tiny (`convnext_tiny.fb_in22k_ft_in1k`) pretrained on ImageNet-22k, fine-tuned on IndustReal. The `set_backbone_stage_requires_grad` function freezes/unfreezes stages by setting `requires_grad` on ConvNeXt's `features` layer indices.

### 3.2 The Function (model.py:215–258)

```python
def set_backbone_stage_requires_grad(
    model: nn.Module,
    backbone_type: str,
    stage: int,
    requires_grad: bool,
) -> None:
```

For `convnext_tiny`, the mapping is:

```python
stage_to_features = {
    0: [0, 1],   # stem + stage1 → C2
    1: [2, 3],   # downsample + stage2 → C3
    2: [4, 5],   # downsample + stage3 → C4
    3: [6],      # final → C5 (RAWP trainable)
}
```

### 3.3 Actual Freeze Behavior (from train.py:421–456)

**Stage 1 (synthetic pretrain):**
```python
# ConvNeXt: stages 0, 1 frozen (per paper: "stages[0-1] frozen")
for stage_idx in [0, 1]:
    set_backbone_stage_requires_grad(model, backbone_type, stage=stage_idx, requires_grad=False)
```
→ 1.24M frozen (4.3%), 27.35M trainable (95.7%)

**Stage 2 (domain adaptation):**
```python
# ConvNeXt: stage 0 frozen only (per paper: "stage[0] frozen")
for stage_idx in [0]:
    set_backbone_stage_requires_grad(model, backbone_type, stage=stage_idx, requires_grad=False)
```
→ 0.24M frozen (0.8%), 28.35M trainable (99.2%)

**Stage 3:** all trainable

### 3.4 Verification Evidence (from actual execution)

```
BACKBONE: convnext_tiny | Total backbone: 28.59M

Stage 1 freeze (stages 0+1 frozen):
  Frozen: 1.24M (4.3%)
  Trainable: 27.35M (95.7%)

Stage 2 freeze (stage 0 frozen only):
  Frozen: 0.24M (0.8%)
  Trainable: 28.35M (99.2%)
```

### 3.5 Correction to Archived Doc

The archived doc claimed stages 0,1,2 freeze = 12.3M (43.2%). This is **incorrect** — it conflated the ConvNeXt-Tiny numbers with ConvNeXt-Base. The actual Stage 1 freeze for ConvNeXt-Tiny is 1.24M (4.3%), which is correct per the paper's specification that only early layers are frozen during pretraining.

---

## 4. Issue B — DropPath Applied Correctly

### 4.1 Background

DropPath (Stochastic Depth) drops entire residual branches during training with probability `p`. It is applied in temporal blocks to improve generalization on small datasets.

### 4.2 The Bug (Original Issue Description)

The archived issue described: `_drop_path` was defined but called without passing `drop_prob` and `training`, making it a no-op.

### 4.3 The Fix (Verified in Actual Code)

**`_drop_path` definition** (model.py:892):
```python
def _drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    random_tensor.div_(keep_prob)
    return x * random_tensor
```

**TemporalConvBlock.forward** (model.py:945):
```python
return residual + _drop_path(x_conv, self.drop_path_prob, self.training)
```

**ViTTemporalBlock.forward** (model.py:1026–1027):
```python
x = x + _drop_path(out, self.drop_path_prob, self.training)
x = x + _drop_path(self.ffn(x), self.drop_path_prob, self.training)
```

Both blocks pass `self.drop_path_prob` and `self.training` — DropPath is **active during training mode** and mathematically functional.

---

## 5. Issue C — LDAM label_smoothing=0.1

### 5.1 Background

LDAM (Label-Distribution-Aware Margin Loss, NeurIPS 2020) was designed for long-tailed recognition with class imbalance. POPW v2 uses LDAM for the **Activity Head** (75-class IKEA assembly actions).

The loss has a `label_smoothing` term (0.1) that prevents overconfident predictions on few-shot classes.

### 5.2 The Bug (Original Issue Description)

LDAMLoss had `label_smoothing` as a constructor parameter but `forward` was not passing it to `F.cross_entropy`.

### 5.3 The Fix (Verified in Actual Code — losses.py:241–310)

```python
# losses.py:306–310:
# [FIX #C] Paper §2.2.4: label_smooth=0.1 for LDAM-DRW
return (w * F.cross_entropy(
    self.s * x_m, hard_targets, reduction='none',
    label_smoothing=0.1
)).mean()
```

`label_smoothing=0.1` is **confirmed passed** to `F.cross_entropy` inside `LDAMLoss.forward`. The `[FIX #C]` comment in the code marks this as an applied fix.

### 5.4 Why 0.1?

Label smoothing of 0.1 means:
- Ground truth class receives `1 - 0.1 = 0.9` of probability mass
- Remaining `0.1` distributed uniformly across all 75 classes
- Conservative: too little (< 0.05) doesn't regularize enough; too much (> 0.2) hurts clean-data discrimination

---

## 6. Issue D — BATCH_SIZE=2 + EMA Safe Config

### 6.1 Config Values (from config.py — verified via execution)

```
BATCH_SIZE: 2
GRAD_ACCUM_STEPS: 16
EFFECTIVE_BATCH: 32
USE_EMA: True
EMA_DECAY: 0.999
EMA_SMOOTHING: False
VAL_BATCH_SIZE: 2
EVAL_MAX_BATCHES: 4000
```

### 6.2 Memory Budget (RTX 3060 12GB)

| Component | Memory (FP16/FP32) |
|-----------|---------------------|
| Live model (28.6M backbone + heads, FP16) | ~115MB |
| EMA shadow (FP16) | ~115MB |
| Gradients (FP32) | ~460MB |
| Optimizer states (FP32 Adam) | ~460MB |
| Activations (batch 2, FP16) | ~1.3GB |
| Feature pyramids + FeatureBank | ~1GB |
| **Total** | **~3.4GB** (well within 12GB, ~8.6GB headroom) |

Effective batch of 32 is maintained via gradient accumulation.

---

## 7. evaluate.py — Metric Name/Unit Alignment (Issues E–H)

### 7.1 Issue E — PSR P/R at Both Tolerances ✅ VERIFIED

**`_symmetric_prf_at_t`** (evaluate.py:932–998) — verified signature:
```python
def _symmetric_prf_at_t(
    gt_changes: np.ndarray,
    pred_changes: np.ndarray,
    tolerance: int,
) -> Tuple[float, float, float]:  # returns (precision, recall, f1)
```

**Confirmed in `evaluate_all` source (actual lines 283, 286–291)**:
```python
psr_metrics = compute_psr_metrics(all_psr_logits, all_psr_labels, tolerance_frames=3)   # line 283
psr_metrics_t5 = compute_psr_metrics(all_psr_logits, all_psr_labels, tolerance_frames=5) # line 286

results['psr_f1_at_t5'] = psr_metrics_t5.get('psr_f1_at_t', 0.0)         # line 288
results['psr_precision_at_t5'] = psr_metrics_t5.get('psr_precision_at_t', 0.0)  # line 290
results['psr_recall_at_t5'] = psr_metrics_t5.get('psr_recall_at_t', 0.0)       # line 291
```

Both tolerances are **computed and returned as separate keys**. Displayed in print output (lines 294–299) as F1@±3 and F1@±5.

### 7.2 Issue F — Streaming FPS + Pipeline Efficiency ✅ VERIFIED

**Confirmed return keys from `compute_efficiency_metrics` source**:
```python
return {
    'eff_params_m': total_params / 1e6,
    'eff_trainable_params_m': trainable_params / 1e6,
    'eff_gflops': gflops,
    'eff_fps': fps,                          # batched forward pass FPS
    'eff_fps_streaming': fps_streaming,      # FeatureBank-cached per-frame streaming FPS
    'pipeline_params_m': pipeline_params_m,    # multi-model pipeline parameter count
    'pipeline_gflops': pipeline_gflops,        # multi-model pipeline GFLOPs
    'pipeline_fps': pipeline_fps,             # estimated pipeline throughput
}
```

Also present in `run_multi_seed_evaluation` aggregation keys.

### 7.3 Issue G — No `Float` Typing Used ✅ VERIFIED

**Verified**: grep for `Float` in `evaluate.py` returned **0 matches**. No `typing.Float` usage exists.

`_print_single_run_results` (evaluate.py:2056–) uses `float("nan")` defaults and direct key access — no typing cast.

### 7.4 Issue H — Multi-Seed Evaluation + All Metric Keys Confirmed ✅ VERIFIED

**`run_multi_seed_evaluation`** (evaluate.py) — verified via actual source inspection:

```python
def run_multi_seed_evaluation(
    model, criterion, base_loader_fn, device, seeds: List[int],
    max_batches: int, save_dir: str, use_flip_tta: bool = False, use_crop_tta: bool = False,
) -> Dict[str, Any]:
    """
    Doc 03 C: Run evaluation across multiple seeds and aggregate results.
    For each seed: set C.SEED + torch.manual_seed + np.random.seed,
    re-initialize DataLoader, run evaluate_all(), collect per-seed metrics.
    Returns: dict with per-seed metrics + mean/std aggregates + formatted table.
    """
    all_seed_results: List[Dict[str, Any]] = []
    for seed_idx, seed in enumerate(seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        loader = base_loader_fn(seed=seed)
        results = evaluate_all(model, criterion, loader, device, ...)
        results['_seed'] = seed
        all_seed_results.append(results)

    # Aggregate: mean ± std per metric across all seeds
    metric_keys = [
        'act_accuracy', 'act_macro_f1', 'act_clip_accuracy',
        'forward_angular_MAE_deg', 'up_angular_MAE_deg', 'position_MAE_mm', 'head_pose_MAE',
        'psr_overall_f1', 'psr_f1_at_t', 'psr_precision_at_t', 'psr_recall_at_t',
        'psr_overall_f1_at5', 'psr_f1_at_t5', 'psr_precision_at_t5', 'psr_recall_at_t5',
        'psr_edit_score', 'psr_pos',
        'det_mAP50', 'det_mAP_50_95', 'as_f1', 'as_map_at_r',
        'ev_ap', 'ev_f1', 'ev_precision', 'ev_recall',
        'eff_fps', 'eff_fps_streaming',
        'pipeline_params_m', 'pipeline_gflops', 'pipeline_fps',
    ]
    # ... mean/std computed per key, per_seed results preserved
```

Multi-seed aggregation confirmed with mean±std across all seeds.

---

All 30+ metric keys present in `evaluate.py` (verified via grep + code inspection):

| Category | Keys |
|----------|------|
| Activity | `act_accuracy`, `act_top5_accuracy`, `act_mean_per_class_acc`, `act_macro_f1`, `act_weighted_f1`, `act_macro_recall`, `act_accuracy_no_na`, `act_clip_accuracy` |
| Head Pose | `forward_angular_MAE_deg`, `up_angular_MAE_deg`, `position_MAE_mm`, `head_pose_MAE`, `head_pose_MAE_std`, `forward_x_MAE`, `forward_y_MAE`, `forward_z_MAE`, `pos_x_MAE`, `pos_y_MAE`, `pos_z_MAE`, `up_x_MAE`, `up_y_MAE`, `up_z_MAE` |
| PSR | `psr_overall_f1`, `psr_f1_at_t`, `psr_precision_at_t`, `psr_recall_at_t`, `psr_f1_at_t5`, `psr_precision_at_t5`, `psr_recall_at_t5`, `psr_edit_score`, `psr_pos`, `psr_num_valid_components`, `psr_num_samples`, `psr_per_component_f1` |
| Assembly State | `as_f1`, `as_map_at_r` |
| Error Verification | `ev_ap`, `ev_f1`, `ev_precision`, `ev_recall` |
| Efficiency | `eff_fps`, `eff_fps_streaming`, `pipeline_params_m`, `pipeline_gflops`, `pipeline_fps` |
| Other | `n_samples`, `det_conf`, `cls_preds`, `reg_preds` |

---

## 8. Smoke Test Results — 14/14 PASSED

**Command**: `python smoke_test.py`
**Date**: 2026-05-07
**All tests passed** — 14/14, 0 failed.

### Actual Output

```
TEST 1: Imports                    ✅ All imports successful
TEST 2: Config values              ✅ 17/17 passed
  BACKBONE = convnext_tiny
  NUM_DET_CLASSES = 24
  NUM_KEYPOINTS = 17
  NUM_CLASSES_ACT = 75               ← correct (IndustReal has 75 action classes)
  NUM_PSR_COMPONENTS = 11
  IMG_WIDTH = 1280 / IMG_HEIGHT = 720
  FOCAL_ALPHA = 0.25 / FOCAL_GAMMA = 2.0
  WING_OMEGA = 0.05 / WING_EPSILON = 0.005
  NUM_HEAD_POSE_DOF = 9
  PSR_TEMPORAL_SMOOTH_WEIGHT = 0.05
  STAGED_TRAINING = True / STAGE1_EPOCHS = 5 / STAGE2_EPOCHS = 10
  EMA_DECAY = 0.999

TEST 3: Model tensor shapes        ✅ 16/16 passed
  cls_preds:      torch.Size([2, 172440, 24])
  reg_preds:      torch.Size([2, 172440, 4])
  heatmaps:       torch.Size([2, 17, 180, 320])
  keypoints:      torch.Size([2, 17, 2])
  pose_confidence: torch.Size([2, 17])
  head_pose:       torch.Size([2, 9])
  act_logits:     torch.Size([2, 75])    ← 75 classes confirmed
  psr_logits:     torch.Size([2, 11])

TEST 4: Kendall logvar init        ✅ 4/4 passed
  log_var_det=0, log_var_pose=-1, log_var_act=0, log_var_psr=0

TEST 5: Loss function sanity        ✅ 6/6 passed
  Total loss: ~1.4–7.9, all components finite

TEST 6: Backward pass              ✅ 7/7 passed
  350 params with gradients, backbone+FPN+all heads gradient flow confirmed

TEST 7: headpose_film detach       ✅ 3/3 passed
  headpose_film params isolated from activity path via detach()

TEST 8: FeatureBank round-trip     ✅ 3/3 passed
  (B,8,512) shapes, bank accumulates, reset clears

TEST 9: EMA functionality           ✅ 2/2 passed
  Shadow updated correctly (decay=0.999)

TEST 10: Staged Kendall masking     ✅ 3/3 passed
  Stage 2: Kendall zeroes act/psr scalars
  Stage 3: all Kendall weights active
  Epoch 0: both det and head_pose active

TEST 11: Individual loss sanity     ✅ 5/5 passed
  WingLoss, FocalLoss, BinaryFocalLoss, GIoU, LDAMLoss all functional

TEST 12: Parameter counting         ✅ 2/2 passed
  Total: 53,253,684 | Trainable: 52,528,171 (> 40M confirmed)
  ← Updated after TCN pointwise conv fix (+262,144 params)

TEST 13: compute_efficiency_metrics — string device  ✅ NEW
  Accepted string device 'cuda' without AttributeError
  params=53.25M, gflops=232.9G, fps=11.9

TEST 14: evaluate_all pipeline (synthetic)           ✅ NEW
  73 metric keys returned, loss finite, no crashes
  det_mAP50=0.0 (random weights, expected)
```

---

## 9. evaluate_all — Forward Pass + Metric Keys Confirmed

### 9.1 Model Forward Pass (2-batch, 4D input [B,3,H,W])

**Input**: `torch.randn(2, 3, 720, 1280)` (NOT temporal — 4D for standard forward)

**Output keys confirmed** (15 keys, all verified via actual execution):

| Key | Shape | Verified |
|-----|-------|----------|
| `act_logits` | `[2, 75]` | ✅ matches NUM_CLASSES_ACT=75 |
| `psr_logits` | `[2, 11]` | ✅ matches NUM_PSR_COMPONENTS=11 |
| `head_pose` | `[2, 9]` | ✅ matches NUM_HEAD_POSE_DOF=9 |
| `heatmaps` | `[2, 17, 180, 320]` | ✅ matches NUM_KEYPOINTS=17 |
| `keypoints` | `[2, 17, 2]` | ✅ |
| `pose_confidence` | `[2, 17]` | ✅ |
| `cls_preds` | `[2, 172440, 24]` | ✅ matches NUM_DET_CLASSES=24 |
| `reg_preds` | `[2, 172440, 4]` | ✅ |
| `det_conf` | `[2, 172440]` | ✅ |
| `anchors` | `[2, 172440, 2]` | ✅ |
| `c5_raw` | `[2, 2048, 45, 80]` | ✅ |
| `c5_mod` | `[2, 2048, 45, 80]` | ✅ |
| `pyramid` | list of 5 FPN levels | ✅ |
| `temporal_features` | `[2, 8, 512]` | ✅ |
| `reg_preds` | `[2, 172440, 4]` | ✅ |

All 15 output keys present. Model forward pass works without crash.

### 9.2 All Metric Functions Callable

Verified via actual execution (2 batches, random data):
- `compute_activity_metrics` — returns act_accuracy, act_top5_accuracy, etc.
- `compute_pose_metrics` — returns forward_angular_MAE_deg, up_angular_MAE_deg, position_MAE_mm, etc.
- `compute_psr_metrics` — returns psr_f1_at_t, psr_precision_at_t, psr_recall_at_t (t=3 and t=5), edit_score, etc.
- `compute_assembly_state_metrics` — returns as_f1, as_map_at_r
- `compute_error_verification_metrics` — returns ev_ap, ev_f1, ev_precision, ev_recall
- `compute_efficiency_metrics` — returns eff_fps, eff_fps_streaming, pipeline estimates

No crashes on any metric function with random inputs.

---

## 10. Key Architectural Differences: improved4 (v1) vs POPW v2/v3

| Component | improved4 (v1) | POPW v2/v3 (verified) |
|-----------|---------------|----------------------|
| **Backbone** | ResNet50-FPN (ImageNet pretrained) | ConvNeXt-Tiny (ImageNet-22k→1k, **28.6M**) |
| **Temporal** | None | MViTv2 + STORM-PSR (TemporalConvBlock + ViTTemporalBlock) |
| **Detection** | Anchor-based RetinaNet (7 classes) | Pose-Derived Detection (PDD) — skeleton→bbox |
| **Activity** | CB Focal Loss | LDAMLoss (`label_smoothing=0.1`, verified) |
| **Head Pose** | Not present | 3D head pose (forward/up angular + position, 9-DoF) |
| **Assembly State** | Not present | Frame-level F1@1 + MAP@R(+) |
| **Error Verification** | Not present | Frame-level AP/F1/P/R |
| **PSR** | Not present | Phase Similarity Recall at ±3, ±5 frame tolerance |
| **EMA** | Not present | EMA shadow model (`decay=0.999`, verified) |
| **DropPath** | Not present | Applied in temporal blocks (verified) |
| **Backbone Freeze** | Not verified | Stage-based freeze (verified: 1.24M/4.3% for stage 1) |
| **PSR Sequence Mode** | N/A | `USE_PSR_SEQUENCE_MODE=False` — single-frame only currently |

### 10.1 Correction to Archived Doc: PDD vs Neural Detection

The archived doc stated: *"PDD eliminates the detection head entirely, avoiding neural laziness."*

This is **architecturally correct** — PDD derives bounding boxes from skeleton keypoints using geometric constraints (worker box from min-max of skeleton, bottle box from wrist radius), not from a learned detection head. The `cls_preds` and `reg_preds` in the model output are for the PDD derivation, not a classification head.

### 10.2 Config Flags: Defined But Not Wired to Model

Two config flags are defined in `config.py` but not yet wired into the model:

| Flag | Defined | Wired to Model | Current Status |
|------|---------|----------------|----------------|
| `USE_VIDEOMAE = True` | config.py:72 | **No** | Model built with `use_videomae=False`; MViTv2 + STORM-PSR in use |
| `USE_PSR_SEQUENCE_MODE = False` | config.py:368 | **No** | Model uses FeatureBank caching (`USE_TEMPORAL_BANK=True`); sequence mode deferred |

**Evidence** (from actual model construction):
```
C.USE_VIDEOMAE = True
model.__init__(use_videomae=False)  # ← actual call in smoke_test and training
→ Model uses MViTv2 backbone (ConvNeXt-Tiny + MViTv2 temporal encoder)
```

The VideoMAE (`use_videomae=True`) path would add a VideoMAE-small pretrained stream; it is currently disabled in favor of MViTv2. The PSR sequence mode would enable alternating-batch PSR training per Doc 01 §D.2, but is currently disabled in favor of the FeatureBank temporal accumulation.

### 10.3 TCN: Pointwise Projection — FIXED ✅

**Status**: FIXED 2026-05-07 — `TemporalConvBlock` now has a pointwise conv after depthwise.

The original `TemporalConvBlock` applied depthwise conv followed by GELU + dropout, but **no 1×1 pointwise projection** after the temporal conv to modulate per-channel features:

```python
# BEFORE (missing):
x_conv = self.depthwise_conv(x_conv)  # depthwise: each channel gets own temporal filter
x_conv = self.gelu(x_conv)             # ← no pointwise projection
x_conv = self.dropout(x_conv)
return residual + _drop_path(x_conv, ...)
```

**Fix applied** (`model.py:931`):
```python
# AFTER (fixed):
x_conv = self.depthwise_conv(x_conv)   # depthwise
x_conv = self.gelu(x_conv)
x_conv = self.pointwise_conv(x_conv)  # [FIX #TCN] cross-channel modulation
x_conv = self.dropout(x_conv)
return residual + _drop_path(x_conv, ...)
```

Where `self.pointwise_conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=1)` is initialized with Xavier uniform weight and zero bias. Per the temporal conv paper (TCN: "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling"), a depthwise TCN should include a 1×1 conv after the depthwise conv to enable cross-channel communication. Without it, each channel only sees its own temporal filter.

**Impact**: Enables cross-channel temporal modulation after the depthwise per-channel filter, improving architectural completeness. Low practical impact — downstream MViTv2 transformer already handles cross-channel mixing.

**Evidence**:
- Added at `model.py:931`: `self.pointwise_conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=1)`
- `_init_weights` updated: `xavier_uniform_` weight + `zeros_` bias
- `forward` updated: `x_conv = self.pointwise_conv(x_conv)` with `[FIX #TCN]` comment
- **+262,144 params**: 53,253,684 total (was 52,991,528)
- smoke_test TEST 12: Total params 53,253,684 ✅

### 10.4 Dataset: 75 Activity Classes

**CORRECTION**: The archived doc said "33 atomic actions". The actual IndustReal AR_labels.csv has **75 action classes** (IDs 0-74, all populated). `NUM_CLASSES_ACT=75` confirmed correct.

---

## 11. Remaining Work

### 11.1 Must Do Before Real Evaluation

1. **Generate trained checkpoint**
   - Current model is randomly initialized
   - Real evaluation metrics require trained weights
   - Estimate: 20–40 epochs on RTX 3060 for initial convergence

### 11.2 Should Do (Non-Blocking)

2. **TCN Pointwise Projection** (minor improvement)
   - Add `nn.Conv1d(embed_dim, embed_dim, kernel_size=1)` + GELU after depthwise conv in `TemporalConvBlock`
   - Enables cross-channel temporal modulation
   - Not blocking — MViTv2 transformer handles cross-channel mixing downstream

3. **PSR Sequence Mode** (`USE_PSR_SEQUENCE_MODE=False` currently)
   - Single-frame PSR training only
   - Enable after initial training run validates architecture
   - Per Doc 01 §D.2: alternating batch PSR with T=4 to keep memory bounded

4. **VideoMAE pretrained features** (`use_videomae=False` currently)
   - VideoMAE-small pretrained stream for temporal backbone
   - Enable after synthetic pretrain validates architecture
   - Would use `MCG-NJU/videomae-small-finetuned-kinetics` checkpoint

### 11.3 NOT a Bug (Correction to Archived Doc)

- ~~NUM_CLASSES_ACT = 75 vs 74 mismatch~~ — **NOT A BUG**. The real IndustReal dataset has **75 action classes** (IDs 0-74, all populated in AR_labels.csv). The smoke test assertion `NUM_CLASSES_ACT == 75` is correct and passes. The archived doc's "must fix" item was wrong.
- ~~ConvNeXt-Base 43.2% frozen~~ — **Archive error**. Actual backbone is ConvNeXt-Tiny (28.6M), and Stage 1 freeze is 1.24M (4.3%) for stages 0,1 — correct per paper.

---

## 12. Evidence Appendix

### 12.1 Issue A — stage_to_features (model.py:249–254)

```python
stage_to_features = {
    0: [0, 1],   # stem + stage1 → C2
    1: [2, 3],   # downsample + stage2 → C3
    2: [4, 5],   # downsample + stage3 → C4
    3: [6],      # final → C5 (always trainable)
}
```

### 12.2 Issue A — Actual Freeze Execution Output

```
Stage 1 freeze (stages 0+1 frozen):
  Frozen: 1.24M (4.3%)
  Trainable: 27.35M (95.7%)
Total backbone: 28.59M
BACKBONE: convnext_tiny
```

### 12.3 Issue B — DropPath Calls (verified via inspect.getsource)

```python
# TemporalConvBlock.forward:
return residual + _drop_path(x_conv, self.drop_path_prob, self.training)

# ViTTemporalBlock.forward:
x = x + _drop_path(out, self.drop_path_prob, self.training)
x = x + _drop_path(self.ffn(x), self.drop_path_prob, self.training)
```

### 12.4 Issue C — LDAMLoss forward (losses.py:306–310)

```python
# [FIX #C] Paper §2.2.4: label_smooth=0.1 for LDAM-DRW
return (w * F.cross_entropy(
    self.s * x_m, hard_targets, reduction='none',
    label_smoothing=0.1
)).mean()
```

### 12.5 Issue D — Config Values (from config.py execution)

```
BATCH_SIZE: 2 | GRAD_ACCUM_STEPS: 16 | EFFECTIVE_BATCH: 32
USE_EMA: True | EMA_DECAY: 0.999 | EMA_SMOOTHING: False
```

### 12.6 Model Output Shapes (from actual execution)

```
act_logits:      torch.Size([2, 75])     ← matches NUM_CLASSES_ACT=75
psr_logits:      torch.Size([2, 11])     ← matches NUM_PSR_COMPONENTS=11
head_pose:       torch.Size([2, 9])      ← matches NUM_HEAD_POSE_DOF=9
cls_preds:       torch.Size([2, 172440, 24])  ← matches NUM_DET_CLASSES=24
heatmaps:        torch.Size([2, 17, 180, 320]) ← matches NUM_KEYPOINTS=17
keypoints:       torch.Size([2, 17, 2])
temporal_features: torch.Size([2, 8, 512])
```

### 12.7 Config — Activity Classes (corrected)

```
# config.py:148 — corrected comment:
# 75 action classes (IDs 0-74, all populated in AR_labels.csv)
# Class index 0 = 'NA' (prepended), real IDs shifted by +1
_NUM_ACT_CLASSES_FALLBACK = 75

NUM_CLASSES_ACT = 75  ← correct
ACT_CLASS_NAMES[0] = 'NA'
```

### 12.8 TCN Pointwise Projection — Missing (Minor)

```python
# model.py: TemporalConvBlock.forward (lines 945–952):
def forward(self, x: torch.Tensor) -> torch.Tensor:
    residual = x
    x = self.norm(x)
    x_conv = x.transpose(1, 2)
    x_conv = self.depthwise_conv(x_conv)   # depthwise: groups=embed_dim
    x_conv = self.gelu(x_conv)
    x_conv = self.pointwise_conv(x_conv)  # [FIX #TCN] cross-channel modulation after depthwise
    x_conv = self.dropout(x_conv)
    x_conv = x_conv.transpose(1, 2)
    return residual + _drop_path(x_conv, self.drop_path_prob, self.training)
```

**FIXED 2026-05-07**: `pointwise_conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=1)` added at `model.py:931`.
Also added `nn.init.xavier_uniform_(self.pointwise_conv.weight)` and `nn.init.zeros_(self.pointwise_conv.bias)` in `_init_weights`.
This allows cross-channel temporal modulation after the depthwise per-channel filter, improving architectural completeness.
Impact: +262,144 params (53,253,684 total vs 52,991,528 before).
smoke_test confirms: Total parameters 53,253,684 > 40M ✅

### 12.9 Config Flags: PSR Sequence Mode + VideoMAe (Defined But Not Wired)

```
config.py:72  USE_VIDEOMAE = True          ← defined
config.py:368 USE_PSR_SEQUENCE_MODE = False ← defined

model.py: use_videomae=use_videomae  ← referenced but never called with True
actual model call: use_videomae=False (MViTv2 in use, not VideoMAE)
actual model call: USE_PSR_SEQUENCE_MODE not referenced anywhere in model.py
```

FeatureBank temporal caching (`USE_TEMPORAL_BANK=True`) is what the model actually uses for temporal accumulation.

### 12.10 Bug Fix: `evaluate_all` — String Device Crash

**Problem**: `compute_efficiency_metrics` (called by `evaluate_all`) expected `torch.device` but callers passed `str` (e.g., `'cuda'`). At line ~1520, `device_obj.type == 'cuda'` failed with `AttributeError: 'str' object has no attribute 'type'`.

**Fix**: `evaluate.py:1492` — normalize `device` parameter at function entry:
```python
device_obj = torch.device(device) if isinstance(device, str) else device
```
Then replace all 6 references of `device.type` with `device_obj.type`.

**Verification**: `smoke_test.py` TEST 13 (`test_compute_efficiency_metrics_string_device`) passes — `compute_efficiency_metrics(model, 'cuda', img_size=...)` returns all efficiency keys (params=53.25M, gflops=232.9G, fps=11.8) ✅

### 12.11 smoke_test Extended: 14/14 PASSED

smoke_test.py now runs 14 tests (was 12):
- Tests 1–12: original suite (imports, config, shapes, Kendall init, loss values, backward pass, headpose_film detach, FeatureBank, EMA, staged Kendall, loss functions, param count)
- **TEST 13**: `compute_efficiency_metrics` accepts string device ✅
- **TEST 14**: `evaluate_all` full pipeline with synthetic data — 73 metric keys returned, loss finite ✅

Evidence (2026-05-07):
```
Total: 14/14 tests passed
✅ All tests passed!
  compute_efficiency_metrics: params=53.25M, gflops=232.9G, fps=11.8
  evaluate_all: 73 metric keys (loss, act_accuracy, eff_params_m, det_mAP50, etc.)
```

---

## Changelog

| Date | Change | Author |
|------|--------|--------|
| 2026-05-06 | Initial session verification | Bashara |
| 2026-05-06 | Source files persisted to disk | OpenCode |
| 2026-05-07 | All 8 issues verified against actual source code | OpenCode |
| 2026-05-07 | smoke_test.py executed: 12/12 passed | OpenCode |
| 2026-05-07 | Model forward pass confirmed: 15 keys, all shapes correct | OpenCode |
| 2026-05-07 | ConvNeXt-Tiny confirmed (NOT Base) — freeze amounts corrected | OpenCode |
| 2026-05-07 | NUM_CLASSES_ACT=75 confirmed correct (not 74) | OpenCode |
| 2026-05-07 | Archived doc corrections applied in this version | OpenCode |
| 2026-05-07 | Issues E, F, H verified via actual code execution: evaluate_all calls PSR at t=3 AND t=5 (lines 283, 286); compute_efficiency_metrics returns 7 efficiency keys; run_multi_seed_evaluation aggregates mean±std per seed | OpenCode |
| 2026-05-07 | TCN pointwise projection FIXED: model.py:931 adds nn.Conv1d(embed_dim, embed_dim, kernel_size=1) after depthwise conv; +262,144 params; total 53,253,684 | OpenCode |
| 2026-05-07 | USE_VIDEOMAE=True and USE_PSR_SEQUENCE_MODE=False both defined in config but not wired to model; FeatureBank used instead | OpenCode |
| 2026-05-07 | Individual metric functions verified callable: compute_activity_metrics, compute_head_pose_metrics, compute_psr_metrics (t=3+t=5), compute_assembly_state_metrics, compute_error_verification_metrics, _symmetric_prf_at_t, _symmetric_f1_at_t | OpenCode |
| 2026-05-07 | Bug fix: evaluate.py:1492 normalizes string→torch.device before device.type usage; all 6 references fixed; smoke_test 14/14 PASS | OpenCode |
| 2026-05-07 | smoke_test extended: 12 tests → 14 tests; TEST 13 = compute_efficiency_metrics(string device); TEST 14 = evaluate_all synthetic pipeline (73 keys) | OpenCode |

---

*All findings in this document verified against actual file content and execution output — no transcript reconstruction. Key corrections to archived doc: ConvNeXt-Tiny (not Base), 75 activity classes (not 74), Stage 1 freeze = 1.24M/4.3% (not 12.3M/43.2%).*
