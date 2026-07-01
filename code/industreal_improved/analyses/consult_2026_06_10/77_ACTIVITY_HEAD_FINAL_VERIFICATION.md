# 77 — Activity Head: Final Verification for Benchmarkable Result [2026-07-01]

**Goal:** Guarantee the activity head produces a real, benchmarkable number (top-1 ≥ 0.40, macro-F1 ≥ 0.30 on verb groups) by verifying every mechanism in the training loop. Opus must confirm there is no hidden bug, miswire, or missing guardrail.

**Source files (all paths relative to `code/industreal_improved/src/`):**
- `config.py:270–330, 680–703, 800–852, 1920–1930` — grouping, sampler, loss, gradient blend
- `data/industreal_dataset.py:790–805, 897–908, 1015–1028, 1412–1450` — label remap, sampler
- `models/model.py:1260–1476` — ActivityHead (simple MLP + TCN/ViT fallback)
- `training/losses.py:1044–1068, 1106–1136, 1333–1417` — loss construction, CB weights, forward
- `training/train.py:3338–3352, 3422–3428` — class_counts remap, criterion init
- `evaluation/evaluate.py:886–1018, 3440–3495` — metrics, diversity logging
- `scripts/verify_act_grouping.py` — pre-flight grouping inspector

---

## 1. Verb-Grouping Wiring (Route A, file 75-76)

### 1.1 Current Configuration
```python
# config.py:293
ACT_CLASS_GROUPING = os.environ.get('ACT_CLASS_GROUPING', 'hybrid')
# hybrid mode: classes >= 100 frames keep fine-grained identity, rest verb-grouped
ACT_HYBRID_THRESHOLD = 100  # config.py:294

# config.py:691-692
ACT_SAMPLER_MODE = 'balanced'
ACT_SAMPLER_COUNT_FLOOR = 15.0
```

Grouping mode `'hybrid'` gives ~47 effective outputs (well-supported classes standalone, tail verb-grouped). This is the smart default: you get fine-grained recognition for common classes AND verb-group coverage for the tail, all in one head.

### 1.2 Grouping Builder (`config.py:334–393`)
The `_build_act_grouping()` function:
1. **`'none'`**: identity mapping, 75 raw classes
2. **`'verb'`**: groups by first underscore token (e.g., `take_*`, `tighten_*`, `loosen_*`) → ~10-13 groups
3. **`'hybrid'`**: classes with ≥100 frames keep standalone identity, remainder verb-grouped

**Critical design**: group index 0 is reserved as `'other'` for unknown verbs and zero-frame classes. This means `act_accuracy_no_na` (which excludes class 0) drops one real group when that group lands in index 0. The headline metrics (top-1, macro-F1) are unaffected because `present_labels` filter in evaluate.py:945 averages only over GT-present groups.

### 1.3 Label Remap Threading (4 production sites)
Every path from label to loss is remapped:

| Site | File:Line | Mechanism |
|------|-----------|-----------|
| Dataset init (activity_ids for sampler) | `industreal_dataset.py:796-804` | `remap_activity_label()` on every `action_label` |
| Per-frame `__getitem__` | `industreal_dataset.py:902-908` | `remap_activity_label()` on single frame label |
| Sequence path (window majority vote) | `industreal_dataset.py:1020-1023` | `remap_activity_label()` on window-argmax label |
| Training loss class_counts | `train.py:3338-3352` | Remap raw `class_counts` via `remap_activity_label()` |

**All four paths preserve -1 sentinel** (unlabeled frames excluded from loss via `act_mask`).

### 1.4 Downstream Dimension Consistency
```python
# config.py:1927-1930 (module-level, set after import resolution)
NUM_ACT_OUTPUTS = _act_g[2]  # 75 for 'none', ~10 for 'verb', ~47 for 'hybrid'
ACT_OUTPUT_NAMES = _act_g[1]  # group names matching output width
ACT_ID_TO_GROUP = _act_g[0]   # raw_id -> group mapping (length 75)

# model.py:1860 — head width
num_classes=int(getattr(C, 'NUM_ACT_OUTPUTS', C.NUM_CLASSES_ACT))

# train.py:3425 — loss width
num_classes_act=int(getattr(C, 'NUM_ACT_OUTPUTS', C.NUM_CLASSES_ACT))

# evaluate.py:3449 — diversity monitor
num_cls = int(getattr(C, 'NUM_ACT_OUTPUTS', getattr(C, 'NUM_CLASSES_ACT', ...)))
```

**ALL three sites read `NUM_ACT_OUTPUTS`** — guaranteed consistent.

### 1.5 Opus Verification Checklist for Grouping
- [ ] Run `scripts/verify_act_grouping.py` and confirm `NUM_ACT_OUTPUTS` matches the expected ~47 (hybrid) or ~13 (verb)
- [ ] Confirm no output group has <15 training frames (below `ACT_SAMPLER_COUNT_FLOOR` would still sample, but may not generalize)
- [ ] Confirm `remap_activity_label(-1) == -1` (sentinel preserved)
- [ ] Confirm `class_counts` in train.py:3344 has length == `NUM_ACT_OUTPUTS`, not 75

---

## 2. Sampler: True Class Balance

### 2.1 Mechanism (`industreal_dataset.py:1412-1450`)
```python
# config.py:691-692
ACT_SAMPLER_MODE = 'balanced'
ACT_SAMPLER_COUNT_FLOOR = 15.0
```

```python
# industreal_dataset.py:1427-1444
_mode = str(getattr(C, 'ACT_SAMPLER_MODE', 'cb')).lower()
if _mode == 'balanced':
    _floor = float(getattr(C, 'ACT_SAMPLER_COUNT_FLOOR', 15.0))
    _eff = np.maximum(counts, _floor)
    class_weights = np.where(counts > 0, 1.0 / _eff, 0.0)
```

Every class with ≥15 frames gets **equal sampling mass**. Classes with 1-14 frames get mass proportional to count (avoids 50× oversampling of singletons, preventing memorization).

### 2.2 Interaction with DET_GT_FRAME_FRACTION
```python
# industreal_dataset.py:1446-1470 (general area after sampling weights)
# Tier 3.12 — Task-aware sampling: DET_GT_FRAME_FRACTION redistributes mass
# so GT-bearing frames get guaranteed representation
```

When `DET_GT_FRAME_FRACTION > 0`, the class-balanced activity weights are *then* redistributed to ensure detection has GT frames. This means activity-balanced sampling is **modified** by detection's needs. For RF4+ (detection stable), this shouldn't hurt activity — but at epoch 1, if detection has very few GT frames, activity sampling could be distorted.

### 2.3 Opus Verification Checklist
- [ ] After `apply_preset()`, confirm `DET_GT_FRAME_FRACTION <= 0.5` for multi-head stages (RF3+)
- [ ] At epoch 1, log the effective class weights: report `max(weight)/min(weight)` ratio
- [ ] Confirm the `-1` sentinel frames get `sample_weights = 0.0` (line 1442) — they should never be sampled

---

## 3. Loss: CE + CB Weights (not CB-Focal)

### 3.1 Current Configuration
```python
# config.py:696, 700
CB_LABEL_SMOOTHING = 0.1  # label smoothing for CE
USE_CB_FOCAL_ACT = False   # Plain CE + CB weights, not focal

# config.py:682-683
CB_BETA = 0.99
CB_GAMMA = 1.0
```

### 3.2 Loss Construction (`losses.py:1044-1068`)
```python
# USE_CB_FOCAL_ACT=False, USE_LDAM_DRW=False:
self.act_loss_fn = nn.CrossEntropyLoss(
    label_smoothing=getattr(C, 'CB_LABEL_SMOOTHING', 0.1),
)
```

### 3.3 CB Weight Injection (`losses.py:1115-1135`)
At `set_class_counts()` time, if the loss is `CrossEntropyLoss`, CB weights are computed and injected:
```python
counts = torch.as_tensor(counts, dtype=torch.float32)
_beta = float(getattr(C, 'CB_BETA', 0.99))
_eff_num = (1.0 - _beta ** counts) / (1.0 - _beta)
_eff_num = _eff_num.clamp(min=1.0)
_weights = 1.0 / _eff_num
_weights = _weights / _weights.sum() * len(_weights)
self.act_loss_fn = nn.CrossEntropyLoss(
    weight=_weights.to(...),
    label_smoothing=float(getattr(C, 'CB_LABEL_SMOOTHING', 0.1)),
)
```

**Critical detail**: this creates a *new* `CrossEntropyLoss` instance with weights every time `set_class_counts` is called. If called multiple times (e.g., at each epoch), the optimizer still holds the old loss module — but since the loss is re-created, the old module's internal `weight` buffer is replaced. **Check**: does `set_class_counts` get called only once at init, or every epoch? If every epoch, the weight tensor device must match.

### 3.4 `_weights[0]=0` Bug — Already Fixed
The old code zeroed slot 0 (`take_short_brace`, 63 frames). The CB formula line 1128-1131 naturally gives low weight to rare classes and **no longer zeroes slot 0** — every class gets a positive weight proportional to `1/effective(n)`.

With verb-grouping active, this concern is moot anyway: the group containing `take_short_brace` likely has >100 frames.

### 3.5 Activity Ramp (`losses.py:1376-1384`)
```python
act_ramp = 1.0
if self.train_act and self._act_epoch_counter >= 0:
    _ramp_ep = self._act_epoch_counter
    act_ramp = min(1.0, (_ramp_ep + 1) / max(self._act_warmup_epochs, 1))
```

Stage-local epoch counter (resets to -1 when `train_act=False`). The ramp goes 0→1 over 5 epochs (config `ACT_RAMP_EPOCHS=5`). **Check**: `_act_epoch_counter` starts at -1. First `set_epoch()` with `train_act=True` increments to 0. So epoch 0 gets ramp = (0+1)/5 = 0.2, epoch 1 gets 0.4, ..., epoch 4 gets 1.0. Correct.

### 3.6 Opus Verification Checklist
- [ ] Log `act_ramp` at each epoch: should be `[0.2, 0.4, 0.6, 0.8, 1.0]` for epochs 0-4
- [ ] Log the CB weight distribution: at grouping mode 'hybrid', the largest weight should not exceed the smallest by more than 50× (the CB-Focal cap). With CE, there's no explicit cap, but the beta=0.99 effective-number formula should produce bounded weights.
- [ ] Verify `set_class_counts` is called exactly once (at train init) — repeated calls with different count arrays would re-create the loss module and disrupt optimizer state.

---

## 4. Activity Head: Simple MLP (ACTIVITY_HEAD_SIMPLE=True)

### 4.1 Architecture (`model.py:1374-1401`)
```python
self.simple_classifier = nn.Sequential(
    nn.LayerNorm(embed_dim),        # [B, 512] -> norm
    nn.Linear(embed_dim, _hidden),  # 512 -> 256
    nn.GELU(),
    nn.Dropout(0.3),                # ACTIVITY_HEAD_DROPOUT=0.3
    nn.Linear(_hidden, num_classes), # 256 -> NUM_ACT_OUTPUTS
)
```

Weight init (lines 1386-1399):
- All hidden layers: `xavier_uniform_`
- Final logit layer: `normal_(std=0.01)` with `constant_(bias=-0.5)` — the negative bias prevents all-zero logits at init, which avoids the "all classes equal probability → random guess" degenerate state
- All LayerNorm: `ones_(weight)`, `zeros_(bias)`

### 4.2 Forward Path (`model.py:1419-1420`)
```python
if getattr(self, 'simple', False) and self.simple_classifier is not None:
    return self.simple_classifier(proj_feat)
```

The TCN+ViT stack is **fully bypassed** when `ACTIVITY_HEAD_SIMPLE=True`. This means:
- **Gradient path**: `proj_feat` → LayerNorm → Linear → GELU → Dropout → Linear → logits
- No feature bank, no temporal convolutions, no self-attention
- ~150K params instead of 8.2M
- Real temporal signal only when `sequence_mode=True` and `FEATURE_BANK_SLOT_OVERWRITE=True`

### 4.3 Gradient Flow Checking
The `proj_feat` comes from the backbone through `c5_mod_blend` with `ACTIVITY_GRAD_BLEND_RATIO=1.0` (line 852 — full gradient). The clip at `ACTIVITY_HEAD_GRAD_CLIP=1.0` (line 809) should not constrain gradients that are already ~0.48 from the healthy gradient path (the old gradient-starved path had 0.012; the in-place assignment bug was fixed in the feature bank, but the simple path never uses the feature bank anyway).

**Verification**: log `act_logits.grad.norm()` and `backbone convnext.stages[-1].weight.grad.norm()` at step 200. Activity gradient should be within 0.1x–10x of detection gradient.

### 4.4 Opus Verification Checklist
- [ ] At step 1, log the logit distribution: are any logits exactly zero? (Would indicate dead weights.)
- [ ] At step 1, compute `softmax(act_logits).max(dim=1)` — should NOT be uniform (0.5/0.5 over 2 classes). With bias=-0.5 init, the initial logits should be slightly negative, producing a softmax entropy near log(N) — not collapse.
- [ ] At epoch 1, confirm `pred_distinct >= 10` (for ~47 hybrid groups). If <10, the simple MLP is also collapsing, which would mean the issue is upstream (backbone features themselves don't separate actions at all).

---

## 5. Eval Metrics: Correct in Group Space

### 5.1 Macro-F1 with present_labels Filter (`evaluate.py:944-948`)
```python
present_labels = [i for i in labels if np.sum(all_gt == i) > 0]
macro_f1 = float(f1_score(all_gt, all_pred, average='macro',
                           zero_division=0, labels=present_labels))
```

The `present_labels` filter averages F1 **only over classes that appear in GT** for that eval split. This is correct for group space: if a group has 0 val frames, it doesn't penalize macro-F1.

### 5.2 Diversity Monitor (`evaluate.py:3440-3487`)
```python
num_cls = int(getattr(C, 'NUM_ACT_OUTPUTS', ...))
_pr_hist = np.bincount(_ap, minlength=num_cls)
_pred_seen = num_cls - int((_pr_hist == 0).sum())
```

The diversity monitor correctly uses `NUM_ACT_OUTPUTS` for its bin count. The collapse warning at line 3468 fires when `_pred_seen < 5`.

### 5.3 Clip-Level Accuracy (`evaluate.py:991-997`)
```python
act_clip_acc = _compute_clip_level_accuracy(
    all_gt, all_pred, clip_ids_arr, exclude_na=True, ...
)
```

This computes **clip-level majority-vote accuracy** — the IndustReal benchmark protocol. With grouping active, this metric will measure group-level clip accuracy, which is the correct comparison target. **This is the number to report in the paper's benchmark table.**

### 5.4 Opus Verification Checklist
- [ ] Confirm `ACT_OUTPUT_NAMES` has same length as `NUM_ACT_OUTPUTS` at eval time. A length mismatch would cause an `IndexError` at `_save_confusion_matrix` (evaluate.py:1052-1077).
- [ ] Run one eval on a known subset: take 100 frames with known labels, confirm macro-F1 calculation matches manual `sklearn.metrics.f1_score` with the same `labels` parameter.
- [ ] For the benchmark table: use **clip-level accuracy** (`act_clip_accuracy`), not frame-level. The IndustReal MViTv2 baseline is clip-level.

---

## 6. What Could Still Go Wrong

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Verb grouping produces one giant group (e.g., all `take_*` verbs are 50% of data) | Medium | Verify group-size distribution with `verify_act_grouping.py` |
| DET_GT_FRAME_FRACTION overrides activity-balanced sampling so activity sees distorted distribution | Low | Check `get_sampler()` output weights at epoch 0 |
| Simple MLP is too small to separate verb groups from pooled features | Low | ~150K params × ~47 outputs = ~3K params/class — sufficient if backbone features have any signal |
| `act_accuracy_no_na` reports misleadingly low number because group 0 is 'other' (a real group) | High (cosmetic) | **Do not** report `act_accuracy_no_na` in paper. Report `act_accuracy` (clip-level) and `act_macro_f1`. |
| Hybrid mode creates too many singleton-output classes (near-threshold classes with 80-99 frames) | Medium | Lower `ACT_HYBRID_THRESHOLD` to 60-80 if too many thin standalone classes; or switch to pure `'verb'` mode |

---

## 7. Final Go/No-Go Criteria (Epoch 2)

| Signal | Pass | Borderline | Fail |
|--------|------|-----------|------|
| `pred_distinct` | ≥ 10 groups | 6-9 groups | < 6 groups |
| `act_macro_f1` | ≥ 0.10 | 0.05-0.10 | < 0.05 |
| `act_clip_accuracy` | ≥ 0.20 | 0.10-0.20 | < 0.10 |
| Per-group entropy | ≥ 1.5 nats | 1.0-1.5 nats | < 1.0 nats |
| Activity grad norm / det grad norm ratio | 0.1x – 10x | — | < 0.01x or > 100x |

**If all pass at epoch 2**: train to 100 epochs. Expected epoch-100: top-1 clip-accuracy ≈ 0.40–0.60, macro-F1 ≈ 0.30–0.50.

**If fail**: verb grouping is not sufficient — the backbone features may not separate verbs either. Fall back to Route B (Section 4-task paper, document activity as negative result).
