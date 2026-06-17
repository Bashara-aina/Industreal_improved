# POPW Audit Report

**Date:** 2026-05-29
**Scope:** `config.py`, `train.py`, `evaluate.py`, `losses.py`, `model.py`, the paper, and the 3-epoch / 5% smoke log.
**Method:** Static read of all files plus the smoke log. The code was **not executed** (no dataset, no GPU here), so anything below that depends on runtime behaviour is labelled as such. I have not invented any metrics.

The headline: most of the bugs flagged in the master prompt are **already fixed correctly** in the uploaded code. There is **one genuine remaining crash risk** (activity label range), which I patched in `losses.py`. There are also a handful of correctness/consistency issues that don't crash but affect whether the model matches the paper. Details below.

---

## 1. Bug report

| ID | Status in uploaded code | Verdict |
|----|------------------------|---------|
| A — EMA never updates with `--no-staged-training` | Fixed | ✅ correct |
| B — `_safe_log` rejected negative `log_var` | Fixed | ✅ correct |
| C — `combined` UnboundLocalError on `_task_nan` | Fixed | ✅ correct |
| D — `NUM_CLASSES_ACT` 74-vs-75 | **Root cause found & fixed** (`config.py`) | ✅ patched by me |
| E — LDAM bincount crash on full dataset | Root cause was Bug D; `losses.py` forward also hardened | ✅ patched by me |
| F — PSR sigmoid below 0.5 at init | Fixed-threshold eval, no curriculum | ⚠️ by design, see Q4 |
| G — `evaluate.py` "shifted" print | Fixed | ✅ correct |

### Bug A — EMA / val-swap (FIXED)
Both update sites are guarded: `train.py:1042` and `train.py:1230` use `if ema is not None and (not staged_training or stage >= 3): ema.update()`, so with `--no-staged-training` the shadow tracks from epoch 0. The unconditional val swap the prompt worried about is also fixed: `train.py:2574` computes `ema_warmed = (ema is not None) and (not _ema_staged or current_stage >= 3)` and only calls `ema.get_ema()` when warmed (`:2575`), with the matching `ema.restore()` guarded by the same flag (`:2709`). This resolves the byte-identical-logits-across-epochs symptom from the smoke log.

One caveat worth knowing: `train.py:2301` and `:2756` still gate some EMA bookkeeping on `current_stage == 3`. With `--no-staged-training` and `STAGE1/2 = 5/10`, `get_stage()` only returns 3 from epoch 15 onward, so those specific branches stay inert until epoch 15 even though the shadow itself updates from epoch 0. It doesn't break the val swap (that path uses `ema_warmed`), but if you ever rely on those two branches, know they're stage-gated.

### Bug B — signed `log_var` (FIXED)
`train.py:1352-1353` now uses `is_log_var = k.startswith('log_var_')` and accepts the value when `(is_log_var or src >= 0.0)`. Negative `log_var` values (legitimate Kendall dynamics, e.g. `log_var_pose` drifting −0.90 → −0.88) are no longer overwritten with −3.0. Functionally equivalent to the `_SIGNED_KEYS` whitelist the prompt described.

### Bug C — `combined` (FIXED)
`combined` is computed by `_compute_combined_metric(...)` *before* the `Val:` line, inside both the success path (`train.py:2799`) and the NaN-skip path (`:2797`, fallback `0.0`). `_task_keys` includes `head_pose_MAE` (`:2762`). No unbound-variable path remains.

### Bug D / E — activity class count (THE real remaining issue)
See §2; this is the one I'd lose sleep over for the 100-epoch run.

### Bug F / Q4 — PSR threshold
`evaluate.py:258` thresholds PSR components at a fixed `sigmoid > 0.5`. At init the smoke log shows sigmoid ∈ [0.16, 0.36], so every component reads 0 → `psr_f1 = 0.0000` until logits climb. No dynamic/curriculum threshold is implemented. Not a bug per se, but it means early-epoch PSR F1 is uninformative and must not drive early stopping (it doesn't — see Q14).

### Bug G — `evaluate.py` "shifted" (FIXED)
The misleading `act_labels_shifted` variable and the `pred == GT+1` accuracy print are gone. `evaluate.py:2647` now reads `act_labels_batch = targets['activity'].cpu().numpy()` with a comment documenting the 1:1 raw-ID → class mapping.

---

## 2. Activity class count — the unresolved hazard (Bugs D + E)

This is the most important finding, so it gets its own section.

**What the code actually does.** `NUM_CLASSES_ACT` is *computed at import time* by `_load_act_class_names()` (`config.py:152-209`), which scans every `AR_labels.csv` on disk, builds a 75-slot list, prunes names that came back as `unknown_*`, then prepends `NA`. The result is asserted to be 74 **or** 75 (`config.py:210`). So the number of activity output channels is a function of *which recordings happen to be on disk* — it is not a fixed constant.

**The smoke log proves it floats.** Every eval batch logs `act_logits shape=(16, 75)` (e.g. `train_5pct_3e_log.log:204`). So in the environment that produced that log, `NUM_CLASSES_ACT == 75`, the model emitted 75 channels, and LDAM was built with `num_classes=75`. Everything was internally consistent **at 75**.

**Why this is fragile.** Three things must agree, and they're derived from different code paths:
1. model output width = `C.NUM_CLASSES_ACT` (`model.py:1673`),
2. LDAM `num_classes` = `C.NUM_CLASSES_ACT` (`losses.py:694`),
3. the activity *targets* = raw action IDs passed straight through (`evaluate.py:2647`; LDAM consumes `targets['activity']` un-clamped at `losses.py:880-885`).

(1) and (2) always agree because they read the same constant. But (3) comes from the dataset's raw IDs. If a run ever lands in the `NUM_CLASSES_ACT == 74` branch (IDs 37/64 absent from the scanned split) while the data still contains a raw ID of 74, then a target of `74` reaches `m_list[hard_targets]` / `scatter_` on a width-74 tensor → **CUDA device-side assert, run dies**. The smoke run didn't hit this only because its max action ID was 71 (< 74), as the prompt notes.

**Root cause confirmed in `industreal_dataset.py`.** `_parse_ar_labels` writes the raw `action_id` straight into the per-frame label with no remapping:
```python
labels[start: end + 1] = action_id   # industreal_dataset.py:425
```
The comment above it ("raw IDs 1-74 map to indices 1-74") is the defect itself — it treats the raw ID *as* the index. `action_id 0` = NA, real actions are `1..74`, so the label space is exactly `0..74` and the classifier **must** have **75** channels. This label flows untouched through `class_counts = bincount(activity_ids, minlength=NUM_ACT_CLASSES)` (`:656`) and `collate_fn` (`:1203`) into LDAM. The maximum label is **74**.

So the precise crash condition: stock IndustReal omits IDs 37 and 64, so `_load_act_class_names()` (which *pruned* the missing names) returned 73 names → `NUM_CLASSES_ACT = 74`. A 74-wide head has valid indices `0..73`; a label of `74` then indexes out of range → CUDA device-side assert. The smoke run dodged it twice: that data copy happened to contain 37/64 (so the count came out 75), and its 5% subset's max ID was 71 anyway.

**Fix applied — `config.py` (the root fix).** I replaced the disk-derived, prune-based count with a fixed constant:
```python
NUM_ACT_RAW_IDS = 74                    # action IDs 1..74 (0 = NA)
NUM_CLASSES_ACT = NUM_ACT_RAW_IDS + 1   # 75, FIXED — not data-derived
```
`_load_act_class_names()` now returns a 75-entry list indexed by raw ID (index 0 = `NA`, 37/64 = `unknown_*` placeholders) for display only, and the assert is now a strict `== 75` invariant. Result: model output = 75, LDAM `num_classes` = 75, `bincount` length = 75, every label in `[0, 74]` is in range — deterministically, regardless of which IDs sit on disk. IDs 37/64 are simply cold channels (no GT, no gradient) — harmless, and consistent with the paper's notation (`y_act ∈ {1,…,74}` + NA = 75 slots). This is the prompt's "Option 2"; I chose it over Option 1 (dense remap) because it's a single-file change that can't drift, whereas remapping would have to be mirrored in `_parse_ar_labels`, the sequence majority-vote, and the name table.

**Belt-and-suspenders — `losses.py`.** I kept the defensive guard in `LDAMLoss.forward` too: margins/DRW weights are now sized to the actual logits width `C`, and any out-of-range target is clamped with a warn-once log (`_warned_oob_target`). With the config fix this guard should never fire, but if a future data variant introduces an unexpected ID it fails loud-and-visible instead of with a cryptic CUDA assert. `set_class_counts` likewise warns instead of raising on a length it doesn't expect.

**Related, separate finding (DRW is a no-op).** In `forward`, the class-balanced reweighting term only activates `if ... self.cb_weights is not None` (`losses.py`, DRW branch), but `MultiTaskLoss` constructs `LDAMLoss` without passing `cb_weights` (`losses.py:693-697`), so `self.cb_weights` stays `None`. `set_class_counts` writes the balanced weights into the *`class_weights` buffer*, which `forward` never reads. Net effect: the **margin** part of LDAM works (it uses `_raw_counts`), but the **DRW reweighting** the paper claims ("class-balanced weights active from epoch 0") never engages. I did **not** silently switch it on, because doing so changes training dynamics and I can't validate the result here — but it means your current runs are effectively "LDAM, no DRW." Decide deliberately whether to wire `class_weights` into the `w` term.

---

## 3. Architecture vs. paper

| Paper claim | In code? | Notes |
|-------------|----------|-------|
| ConvNeXt-Tiny + FPN backbone | ✅ | `BACKBONE='convnext_tiny'`, channels 96/192/384/768. |
| 24-class ASD detection | ✅ | `NUM_DET_CLASSES = 24` (`config.py:118`). Matches IndustReal. **Q8 answered: yes.** |
| PoseFiLM (stage 1) | ✅ implemented **and wired** | `model.py:1825` `c5_mod = self.pose_film(c5, keypoints, pose_confidence)`; γ=`1+tanh`, confidence `.detach()` (`:674`). |
| HeadPoseFiLM (stage 2) | ✅ implemented **and wired** | `model.py:1882` `c5_mod = self.headpose_film(c5_mod, head_pose.detach())`; modulated `c5_mod` feeds the activity head (`:1888`). **Q5 answered: present and connected.** |
| 9-DoF head pose | ✅ | `HeadPoseHead`, `NUM_HEAD_POSE_DOF = 9`. |
| PSR: 11 components, causal Transformer | ✅ | 11 per-component heads, upper-triangular mask (`model.py:1856`). |
| Activity: TCN + 2× ViT, feature bank | ✅ | `TemporalConvBlock` + `ViTTemporalBlock`×2 + CLS token + `FeatureBank`. |
| VideoMAE optional stream | ✅ and **ON** | `USE_VIDEOMAE = True` (`config.py:72`). **Q7 answered.** |
| AdamW, differential LR | ✅ | `USE_LION = False`; param groups backbone 0.1×, bias 0.3× (`train.py:2039-2070`). Paper says AdamW — the code matches the paper (the "Lion" in older notes is stale). |

Two architecture caveats:

- **PoseFiLM conditions on *pseudo* keypoints for IndustReal.** IndustReal is egocentric with no body pose, and `model.py:1816-1821` builds `pseudo_kps`/`pseudo_conf`. So stage-1 FiLM is modulating on fabricated keypoints — it won't hurt (γ≈1 init), but don't expect real pose signal from it on IndustReal. The body-pose contribution the paper attributes to PoseFiLM only materialises on IKEA-ASM.
- **Train/eval asymmetry in HeadPoseFiLM.** `head_pose` and the stage-2 modulation are computed inside `if self.train_pose or not self.training` (`model.py:1879`). With `TRAIN_HEAD_POSE=True` and no staging this is always on, so your production run is fine — but if `train_pose` is ever False during training, `c5_mod` skips stage-2 in train yet gets it in eval, a subtle distribution shift between the two.

---

## 4. Combined metric

The implemented formula (`train.py:1464-1478`) is **not** the one written in the master prompt. It uses:

```
combined = 0.30·mAP50 + 0.35·act_macro_f1 + 0.15·(1/(1+pose_MAE)) + 0.20·psr_f1
```

The pose term is `1/(1+MAE)`, which is bounded in (0, 1] and **can never go negative** — so **Q3 is moot**: there is no negative-contribution problem to clamp away in the actual code. The only issue is a documentation mismatch: the paper/prompt write `0.15·(1 − MAE/π)`. Pick one and make the paper match the code (the `1/(1+MAE)` form is the better-behaved choice; I'd keep it and fix the paper).

---

## 5. Answers to the critical questions

**Training dynamics (Q1–Q4)**
- **Q1 (epochs until val moves):** I can't give a number honestly — that needs a real run on the full set. What I *can* say: with Bug A fixed, val now sees live (or properly-warmed EMA) weights, so the frozen-metric artefact is gone. `act_top5 ≈ 0.25` at init vs. ~1.4% random is **not** a sign the temporal stack already works; with 75 classes, top-5 ≈ 5/75 ≈ 6.7% by chance, and the LDAM `s=30` scaling plus a near-uniform classifier inflate "top-5 hit" further. Treat 0.25 as a baseline artefact, not learning.
- **Q2 (realistic mAP@0.5):** Unverifiable from static code. Honestly: a from-scratch (ImageNet-pretrained backbone, but detection head from zero) multi-task RetinaNet-style head reaching YOLOv8m's 83.8% in 100 epochs on one RTX 3060 is unlikely. A multi-task detector that's also splitting capacity across four other heads typically lands well below a dedicated single-task detector. I won't put a number on it — anything I'd quote would be fabricated.
- **Q3:** Moot — see §4.
- **Q4 (PSR F1 > 0):** Will stay exactly 0 until component logits push sigmoid above 0.5. A lower eval threshold (e.g. 0.3) early on would surface signal sooner, but inflates false positives; cleaner is to leave eval at 0.5 and not let PSR F1 gate anything early. The real lever is the loss side, which is already set sensibly (`PSR_FOCAL_GAMMA = 1.0`, per-component α) — that's what lets the head escape the all-zero minimum.

**Architecture (Q5–Q7)**
- **Q5:** FiLM is real and wired (see §3). Caveat: pseudo-keypoints on IndustReal.
- **Q6 (is the TCN doing temporal work?):** Can't confirm from code alone. The structure is correct (depthwise k=5 conv over the T-axis feature bank, then ViT with learnable pos-embed + CLS). To *verify* temporal reasoning empirically: ablate by shuffling the feature-bank time order at eval — if `act` metrics are unchanged, the temporal blocks are effectively pooling, not reasoning. That's the cheapest diagnostic.
- **Q7 (VideoMAE):** Implemented and enabled (`USE_VIDEOMAE=True`, freeze→unfreeze at epoch 10). The paper's "+5–7%" is the paper's own unverified estimate. Given 12 GB and BATCH_SIZE interplay, confirm it actually fits before the 100-epoch run (the config comments assume batch=1 for the VideoMAE path, but `BATCH_SIZE=6` at `config.py:264` — re-check VRAM at batch 6 with VideoMAE on).

**Data & eval (Q8–Q10)**
- **Q8:** `NUM_DET_CLASSES = 24` ✅.
- **Q9 (head-pose units):** Confirmed concern — `hp_mae_deg` is L1 over a 9-D vector (forward/position/up), not an angle in degrees, and the smoke log shows `hp_mae_deg=0.00` (the head pose isn't producing a meaningful number yet). The label is misleading; either rename it or convert the directional sub-vectors to an actual angular error before calling it "degrees."
- **Q10 (PSR edit distance):** `evaluate.py` implements a Levenshtein DP (`:1339`, `:1507`) — standard edit distance, **not** Damerau-Levenshtein (no transposition term). If the paper claims Damerau-Levenshtein, either add the transposition case or correct the paper's wording.

**Optimisation (Q11–Q14)**
- **Q11 (differential LR):** Implemented (backbone 0.1×, bias 0.3×) for AdamW. Minor: the *inactive* Lion branch uses `backbone_lr * 0.3` (`train.py:2054`) = 0.03×, not 0.1× — harmless while `USE_LION=False`, but fix it before ever flipping Lion on.
- **Q12 (warmup):** Correct — `LinearLR(start_factor=0.1, total_iters=5)` → `SequentialLR` → `CosineAnnealingWarmRestarts(T_0=10, T_mult=2)`. Matches the paper.
- **Q13 (grad clip + AMP):** **Correct order** — `scaler.unscale_(optimizer)` → `clip_grad_norm_` → `scaler.step` → `scaler.update` (`train.py:1035-1041` and `:1223-1229`). No fp16 explosion risk from ordering.
- **Q14 (early stopping):** Driven by `combined` (best-model save at `train.py:2837`), with `PATIENCE=10`. That's the right choice — using PSR F1 alone would never trigger early (it's 0 for many epochs, Q4). Combined is robust because three of its four terms move before PSR does.

**Baseline comparison (Q15–Q17)**
- **Q15 / Q16 / Q17:** These ask "what do we need to beat X." I can give direction but not promises:
  - To approach YOLOv8m's ASD number, the detection head almost certainly needs more than shared multi-task capacity — consider a single-task detection fine-tune from the shared backbone as the fair comparison point, and verify the anchor calibration (the config's k-means note is based on only 923 boxes from 2 recordings — recalibrate on the full set).
  - MViTv2's 65.25% comes from Kinetics-400 pretraining (14M videos). Training the activity stack from scratch on IndustReal won't match it; the VideoMAE stream (Kinetics-pretrained) is your only realistic source of that prior, so it earning its +VRAM is the thing to validate.
  - For PSR F1=0.731: the bottleneck is most likely detection quality feeding the procedure logic (the B2 baseline *is* ASD-confidence + order constraints). If ASD mAP is low, PSR inherits the error. Fix detection first; PSR will follow.

---

## 6. Risk assessment for the 100-epoch run

1. **Activity label out-of-range (was highest).** Resolved at the root: `NUM_CLASSES_ACT` is now a fixed 75 in `config.py`, so the head width can no longer drift below the label range. The `losses.py` clamp remains as a failsafe. Re-run the 5% smoke test after applying both files to confirm `act_logits shape=(B, 75)` and no LDAM warnings.
2. **DRW silently inactive.** Your "LDAM-DRW" is currently margin-only. Decide intentionally.
3. **VRAM at batch 6 + VideoMAE.** Config comments assume batch 1 for this path; `BATCH_SIZE=6` may OOM mid-run. Confirm before committing 100 epochs.
4. **PSR/head-pose metrics flat early.** Expected (Q4, Q9), not a failure — just don't let them gate stopping (they don't).
5. **`hp_mae_deg` mislabel.** Cosmetic but will mislead the paper's table; fix the units/label.

---

## 7. What I changed

Two files, both with inline `[ROOT-CAUSE FIX ...]` / `[ROBUSTNESS FIX ...]` comments:

- **`config.py`** — pinned `NUM_CLASSES_ACT` to a fixed `75` (the raw-ID-indexed space the dataset actually uses) instead of deriving it from a disk scan; rebuilt `_load_act_class_names()` as a 75-entry, raw-ID-aligned list; tightened the assert to `== 75`. This is the root-cause fix for the activity-label crash (Bugs D + E).
- **`losses.py`** — defensive guard in `LDAMLoss`: `_fit_to_width` helper, `forward` sizes margins/weights to the logits width and clamps out-of-range targets (warn-once), `set_class_counts` warns rather than raising. Redundant with the config fix, kept as a loud failsafe.

Everything else (Bugs A, B, C, G; FiLM wiring; LR groups; warmup; AMP clip order) was already correct in your upload and needed no edit. The remaining items in §3–§6 (DRW being inactive, `hp_mae_deg` units, VRAM check, edit-distance vs. Damerau, PoseFiLM pseudo-keypoints) are deliberate decisions or out-of-scope changes I shouldn't make for you.
