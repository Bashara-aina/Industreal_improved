# POPW Training Readiness Audit — May 2026

**Verdict: NOT READY → READY TO TRAIN (after applying these 5 fixes)**

Cannot fill paper tables yet — no benchmark run exists.

---

## Bugs Found & Fixed

### BUG 1 — BLOCKER: head_pose loss dropped from Kendall total
**File:** `losses.py` lines 899–904  
**Root cause:** `elif self.train_act` meant when `train_pose=True`, the 9-DoF head pose MSE loss was never added to the total. Head pose head was untrained.  
**Fix:** Replaced `if/elif` with additive combination — both body pose + head pose under shared `prec_hp`. Applied same fix to NaN guard and non-Kendall fallback.  
**Verified:** head_pose_head has 10 gradient params flowing at stage 3.

### BUG 2 — WARN: Weight logging double-divided
**File:** `losses.py` lines 978–981  
**Root cause:** `wd = a_det / ws` already normalizes, then `'w_det': wd / ws` divides again.  
**Fix:** Changed to `'w_det': wd`.  
**Verified:** Weight sum = 1.000000 at all epochs.

### BUG 3 — BLOCKER: Criterion flags corrupted after PSR sequence batch
**File:** `train.py` lines 778–781  
**Root cause:** `criterion.train_det = False` etc. set for PSR-only mode, never restored. Next normal batch would compute only PSR loss.  
**Fix:** Save flags before override, restore after both exit paths.  
**Verified:** Flags round-trip correctly.

### BUG 4 — HIGH: ConvNeXt backbone gets 10× too high learning rate
**File:** `train.py` line 1698, `optimizer.py` line 23  
**Root cause:** Filter used `['layer0'...'layer4']` (ResNet names). ConvNeXt params use `'backbone.*'`. All 28.6M backbone params got head LR (5e-4) instead of backbone LR (5e-5).  
**Fix:** Added `'backbone.'` to filter list.  
**Verified:** 28,589,128 params now correctly at 0.1× LR.

### BUG 5 — WARN: Non-Kendall stage 2 zeroed pose loss
**File:** `losses.py` line 952  
**Root cause:** `if self.train_pose: _loss_pose_staged = zero` in stage 2 dropped head_pose from non-Kendall path.  
**Fix:** Removed the conditional zero. Stage 2 now keeps pose active.  
**Verified:** Non-Kendall stage 2 total > det-only.

### Minor fix: ConvNeXt stage 3 freeze map missing features[7]
**File:** `model.py` line 291  
**Fix:** `3: [6]` → `3: [6, 7]` (adds 14.3M params to freeze coverage).

---

## 20-Category Audit Matrix

| # | Category | Status | Risk |
|---|----------|--------|------|
| 1 | Paper-to-code parity | FIXED | was HIGH |
| 2 | Dataset structure | PASS | LOW |
| 3 | Label semantics | PASS | LOW |
| 4 | Input size / normalization | PASS | LOW |
| 5 | Backbone geometry | PASS | LOW |
| 6 | FPN / neck | PASS | LOW |
| 7 | Task head shapes | PASS | LOW |
| 8 | Loss formulas | FIXED | was HIGH |
| 9 | Loss scaling (×0.001 etc) | PASS | LOW |
| 10 | Kendall weighting | FIXED | was HIGH |
| 11 | Staged training | FIXED | was HIGH |
| 12 | Temporal modeling | PASS | LOW |
| 13 | FiLM conditioning | PASS | LOW |
| 14 | Data loading / collate | PASS | LOW |
| 15 | Optimizer / scheduler | FIXED | was HIGH |
| 16 | Checkpoint save/load | PASS | LOW |
| 17 | Evaluation protocol | PASS | LOW |
| 18 | Logging | FIXED | was MED |
| 19 | Reproducibility / seeds | PASS | LOW |
| 20 | EMA | PASS | LOW |

---

## Files Changed

| File | Changes |
|------|---------|
| `losses.py` | 4 fixes: Kendall head_pose, weight logging, NaN guard, non-Kendall stage 2 |
| `train.py` | 2 fixes: PSR flag restore, backbone LR filter |
| `model.py` | 1 fix: ConvNeXt stage 3 freeze map |
| `optimizer.py` | 1 fix: backbone LR filter |

## How to Apply

Copy the 4 fixed files over the originals:
```
cp losses.py   /path/to/src/training/losses.py
cp train.py    /path/to/src/training/train.py
cp model.py    /path/to/src/models/model.py
cp optimizer.py /path/to/src/training/optimizer.py
```

## What's Next

1. Run full training (`python train.py`) — now safe to train
2. First benchmark run to fill paper table cells
3. Compare against baselines
