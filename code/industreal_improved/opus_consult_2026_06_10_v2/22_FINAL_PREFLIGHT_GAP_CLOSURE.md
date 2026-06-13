# 22 — Final Pre-Flight & Gap Closure: The Last Guide Before the Paper Run

**What this is.** The single document that closes the gap between the 100-item
checklist (`19_PRE_TRAINING_READINESS_AUDIT_100.md`) and the 10-pass audit
(`21_FINAL_10X_AUDIT_COMPLETE.md`). Apply Parts 1–4 (exact patches), pass the
Part 5 pre-flight, then run Part 6. After that, your IndustReal train produces a
**real, non-zero, non-NaN, non-NA, protocol-correct number for every task** and
the results are paper-usable.

**Audited against `main` (HEAD `72aff80`).** Detection, the safety harness
(ASSERT_AND_CRASH, liveness probe, RC-29 telemetry), data labels (activity_mask,
NA exclusion, subset stratification), anchors (96–512), conf 0.001, and the
eval cadence are **genuinely done**. Three gaps remain — and the project's own
`21_FINAL_10X` audit missed them because it checked *existence*, not *wiring*:

| Gap | What the prior audit saw | What is actually true on `main` | Impact |
|---|---|---|---|
| **C** | "paper_run preset enables PSR transition + geo head pose + bank gradient" | `apply_preset()` **never sets** `USE_PSR_TRANSITION / USE_GEO_HEAD_POSE / FEATURE_BANK_DETACH / USE_VIDEOMAE`; the keys aren't in the dict or the `global` list (config.py:711–742). Defaults stand: all OFF. | The "paper run" silently runs with **every fix disabled**. |
| **A** | "psr_transition imports cleanly; dim==3 gate present" | True — but (A1) PSR loss still trains on **static fill-forward labels on 9/10 (per-frame) batches**, and (A2) `MonotonicDecoder` + order-prior are **never called at eval** (`git grep` shows them only inside `psr_transition.py`). | PSR converges to the **constant-pattern artifact**; F1 not benchmarkable. |
| **B** | (not in the 10 passes) | Activity eval is **per-recording** (16 frames across the whole recording), not per-action-segment (evaluate.py:636). | Activity Top-1 **not comparable to MViTv2 65.25**. |

> Honesty (unchanged): closing these makes every cell *real and defensible* and
> makes **PSR + head-pose actually work** (your winnable cells). It does not make
> the architecture beat YOLOv8m/MViTv2 headline — that's a design ceiling. The
> paper's claim is *one pipeline, all tasks, no zeros, efficiency as headline.*

All patches below are surgical, anchored to real lines, marked `[GAP-x]`. They
ship **verified-by-inspection** — run the per-part verification + the Part 5
smoke on the 3060 before trusting them.

---

# PART 1 — Close GAP C: make `paper_run` actually enable the fixes

**File:** `code/industreal_improved/src/config.py`

### 1a. Extend `apply_preset` to set the four flags (currently it can't)

```python
# in apply_preset(), EXTEND the global declaration block (after line ~720):
    global USE_MIXUP, USE_EMA
    global TRAIN_DET, TRAIN_ACT, TRAIN_PSR, TRAIN_HEAD_POSE
    global USE_PSR_TRANSITION, USE_GEO_HEAD_POSE, FEATURE_BANK_DETACH, USE_VIDEOMAE  # [GAP-C]

# ...and ADD these assignments right after the TRAIN_HEAD_POSE line (~line 742):
    TRAIN_HEAD_POSE = preset.get('train_head_pose', TRAIN_HEAD_POSE)
    # [GAP-C] paper-critical fix flags — previously applied by NO preset
    USE_PSR_TRANSITION  = preset.get('use_psr_transition',  USE_PSR_TRANSITION)
    USE_GEO_HEAD_POSE   = preset.get('use_geo_head_pose',   USE_GEO_HEAD_POSE)
    FEATURE_BANK_DETACH = preset.get('feature_bank_detach', FEATURE_BANK_DETACH)
    USE_VIDEOMAE        = preset.get('use_videomae',        USE_VIDEOMAE)
```

### 1b. Add the keys to the `paper_run` dict (config.py ~line 689)

```python
    'paper_run': {
        'description': 'Final paper-run preset — PSR transition, geo head pose, bank gradient.',
        # ... existing keys unchanged ...
        'train_head_pose':    True,
        # [GAP-C] ACTUALLY enable the paper-critical fixes (these were only in the description):
        'use_psr_transition':  True,
        'use_geo_head_pose':   True,
        'feature_bank_detach': False,   # gradient flows through the FeatureBank
        'use_videomae':        False,   # KEEP OFF at 12GB FP32 (OOM risk). Report activity w/o-VideoMAE
                                        # as the paper's separate row; enable only if VRAM headroom proven.
    },
```

### 1c. Guarantee fresh reads at construction time
`USE_PSR_TRANSITION` is read in `MultiTaskLoss.__init__` (losses.py:914) and
`USE_GEO_HEAD_POSE`/`USE_VIDEOMAE` at model build; `FEATURE_BANK_DETACH` is read
per-step via `getattr(C, ...)`. **Confirm in `train.py main()` that
`apply_preset()` runs *before* the model and `MultiTaskLoss` are constructed**
(it does today). If any flag is cached into a `CFG_*` mirror via
`_refresh_runtime_cfg()`, add it there too.

### ✅ Verify GAP C (no GPU needed)
```bash
cd code/industreal_improved && python3 - <<'PY'
import sys; sys.path.insert(0, '.')
from src import config as C
C.apply_preset('paper_run')
assert C.USE_PSR_TRANSITION is True,  "USE_PSR_TRANSITION still OFF"
assert C.USE_GEO_HEAD_POSE is True,   "USE_GEO_HEAD_POSE still OFF"
assert C.FEATURE_BANK_DETACH is False,"bank gradient still OFF"
print("GAP-C CLOSED:", C.USE_PSR_TRANSITION, C.USE_GEO_HEAD_POSE, C.FEATURE_BANK_DETACH, C.USE_VIDEOMAE)
PY
```
Pass = prints `GAP-C CLOSED: True True False False`.

---

# PART 2 — Close GAP A: make PSR learn transitions and decode them

PSR is your **single best winnable cell** (beat STORM-PSR 0.506, approach B2
0.731). Two sub-fixes.

## 2A1 — Stop the static-label gradient from drowning the transition signal

**Problem:** PSR loss is computed on every batch, but the transition target is
only built on `dim==3` sequence batches (1 in `PSR_SEQ_EVERY_N_BATCHES`=10).
The other 9/10 per-frame batches train on static fill-forward labels →
constant-output pressure wins 9:1.

**File:** `code/industreal_improved/src/training/losses.py` (PSR block, ~line 1176)

```python
        # === PSR ===
        preds = None
        if self.train_psr:
            _is_seq = (outputs['psr_logits'].dim() == 3)
            # [GAP-A1] With the transition objective ON, per-frame (dim==2) batches
            # have no time axis — their only target is the static fill-forward label,
            # which teaches constant output. Skip PSR on per-frame batches so 100% of
            # PSR gradient is transition-based.
            if self.use_psr_transition and not _is_seq:
                loss_psr = zero
            else:
                _psr_targets = targets['psr_labels']
                if self.use_psr_transition and _is_seq:
                    from src.models.psr_transition import build_transition_targets
                    _psr_targets = build_transition_targets(
                        targets['psr_labels'].to(outputs['psr_logits'].device),
                        sigma=float(getattr(C, 'PSR_TRANSITION_SIGMA', 3.0)),
                    )
                # ... existing focal/BCE computation on _psr_targets, unchanged ...
```

**File:** `config.py` — give PSR enough sequence batches now that it *only*
trains on them:
```python
PSR_SEQ_EVERY_N_BATCHES = 4   # [GAP-A1] was 10. PSR now trains ONLY on sequence
                              # batches; raise frequency so it gets enough updates.
                              # Keep PSR_SEQUENCE_LENGTH=4 (T=8 OOMs at 12GB).
```
*Tradeoff:* more sequence batches ≈ +VRAM/step and slightly slower epochs;
acceptable at T=4, batch 1.

## 2A2 — Decode transition logits → monotone states at eval (for F1/POS/Edit)

**Problem:** `MonotonicDecoder` + procedure-order prior exist in
`src/models/psr_transition.py` but are **never called** in `evaluate.py`. PSR
F1/POS/Edit are computed from raw per-frame logits with no monotone decode — so
even good transition logits don't become the state sequence the metrics need.

**File:** `code/industreal_improved/src/evaluation/evaluate.py` — add a helper and
call it where PSR metrics are computed (near `compute_psr_accuracy`, ~line 322).

```python
# [GAP-A2] Decode per-recording transition logits into monotone states, then score.
from src.models.psr_transition import MonotonicDecoder
_PSR_DECODER = MonotonicDecoder(num_components=11)  # carries the procedure-order prior

def decode_and_score_psr(psr_logits_by_rec, gt_states_by_rec, tol_frames=3):
    """
    psr_logits_by_rec: {rec_id: Tensor[T, 11]}  raw PSR head logits in FRAME ORDER
    gt_states_by_rec : {rec_id: Tensor[T, 11]}  GT monotone fill-forward states
    Returns dict with psr_f1@tol, psr_pos, psr_edit (all finite, full test set).
    """
    import torch
    f1s, poss, edits = [], [], []
    for rec, logits in psr_logits_by_rec.items():
        gt = gt_states_by_rec[rec]
        events = torch.sigmoid(logits).unsqueeze(0)            # [1,T,11] event probs
        pred_states = _PSR_DECODER(events).squeeze(0)          # [T,11] monotone states
        # transition frames = where state flips 0->1
        pred_tr = (pred_states[1:] - pred_states[:-1]).clamp(min=0)
        gt_tr   = (gt[1:] - gt[:-1]).clamp(min=0)
        f1s.append(_event_f1(pred_tr, gt_tr, tol=tol_frames))  # bi-dir greedy match
        poss.append(_ordered_pair_fraction(pred_states, gt))   # PSRT POS
        edits.append(_edit_score(pred_states, gt))             # DL distance, GT-normalized
    import numpy as np
    return {
        'psr_f1':   float(np.mean(f1s))  if f1s   else float('nan'),
        'psr_pos':  float(np.mean(poss)) if poss  else float('nan'),
        'psr_edit': float(np.mean(edits))if edits else float('nan'),
    }
```
Wire-in: in `evaluate_all`, collect PSR logits per recording in frame order (the
activity path already groups by `recording_id` — reuse that grouping), then
replace the raw-logit PSR metric with `decode_and_score_psr(...)`. Implement
`_event_f1` (±3/±5 bi-directional greedy match of transition frames),
`_ordered_pair_fraction` (PSRT POS), `_edit_score` (Damerau-Levenshtein,
GT-normalized) per the protocol table in `18_ULTIMATE_MASTER_GUIDE` §5.

### ✅ Verify GAP A
- **A1 (no GPU):** unit-test that `MultiTaskLoss.forward` returns `loss_psr==0`
  on a `[B,11]` (dim==2) batch and `loss_psr>0` finite on a `[B,T,11]` batch when
  `USE_PSR_TRANSITION=True`.
- **A2 (no GPU):** feed a synthetic recording where component k flips at frame t;
  assert `decode_and_score_psr` returns `psr_f1≈1.0` (decoder recovers the flip)
  and **not** the constant-pattern artifact.
- **On box (smoke):** `[PSR_DIAG]` raw loss O(0.1–0.3) finite on sequence batches;
  ≥3 unique predicted patterns; transition F1 > 0 at R2.5.

---

# PART 3 — Close GAP B: per-action-segment activity protocol

**Problem:** MViTv2's 65.25 is one prediction per **action segment** from 16
uniform frames. The code does one prediction per **recording** (evaluate.py:636)
→ NA/majority-dominated, not comparable.

**File:** `code/industreal_improved/src/data/industreal_dataset.py` — add a segment
index + clip sampler (from the AR spans already parsed in `_parse_ar_labels`).

```python
# [GAP-B] one entry per action segment (NOT per recording)
def build_activity_segments(self):
    """Returns [(rec_id, start, end, action_id), ...] from AR spans; NA excluded."""
    segs = []
    for rec_id, spans in self._ar_spans_by_rec.items():     # spans: (start, end, action_id)
        for start, end, aid in spans:
            if aid == 0:                                     # 0 == NA -> never a metric clip
                continue
            segs.append((rec_id, int(start), int(end), int(aid)))
    return segs

def sample_segment_clip(self, seg, T=16):
    rec_id, start, end, aid = seg
    idxs = np.linspace(start, end, T).round().astype(int)    # 16 uniform frames in-segment
    frames = torch.stack([self._load_frame(rec_id, i) for i in idxs])  # [T,3,H,W]
    return frames, aid
```

**File:** `evaluate.py` — segment-level metric (replaces/augments
`_compute_clip_level_accuracy`):
```python
# [GAP-B] one prediction per ACTION SEGMENT, NA excluded
def compute_activity_segment_metrics(model, dataset, device, T=16):
    segs = dataset.build_activity_segments()
    top1 = top5 = 0
    for seg in segs:
        clip, label = dataset.sample_segment_clip(seg, T=T)   # [T,3,H,W], int
        logits = model.clip_logits(clip.to(device))           # [75] CLS readout over the clip
        top1 += int(logits.argmax().item() == label)
        top5 += int(label in logits.topk(5).indices.tolist())
    n = max(len(segs), 1)
    return {'act_top1': top1/n, 'act_top5': top5/n, 'n_segments': len(segs)}
```
(`model.clip_logits` = run the activity head over the 16-frame clip and read the
CLS token / mean-pool to one 75-vector. If no such method, add a thin wrapper
around the existing activity forward that accepts `[T,3,H,W]`.)

### ✅ Verify GAP B
- `n_segments` ≈ total AR spans (not #recordings); on box `act_top1` computed
  over segments, **NA never a label**; print the predicted-class histogram → ≥15
  distinct classes (not collapsed to NA/majority).

---

# PART 4 — Secondary closes (the ⚠️ items)

1. **IMG_SIZE box rescale (real fix, not just the assert).** Today boxes aren't
   rescaled on resize; correct only because `IMG_SIZE==1280×720`. Make it robust:
   ```python
   # industreal_dataset._extract_boxes_from_coco (~line 1015)
   sx = self.img_size[0] / native_w; sy = self.img_size[1] / native_h
   boxes.append([x*sx, y*sy, (x+w)*sx, (y+h)*sy])   # [GAP] rescale to IMG_SIZE
   ```
   Keep the `IMG_SIZE==(IMG_WIDTH,IMG_HEIGHT)` assert (config.py:279) as well.
2. **Liveness probe should include grad-norm.** The probe (losses.py:1307) checks
   loss/finite. Add, in `train.py` after `loss.backward()` for the first
   `LIVENESS_EVERY` steps: print per-head first/last-layer `grad.norm()`. A head
   is ALIVE only if grad-norm > 1e-6. (Loss-finite alone can hide a detached head.)
3. **VideoMAE decision.** Keep `use_videomae=False` for the 12GB FP32 paper run
   (OOM risk). Report activity as the **w/o-VideoMAE** row; the paper already has
   both rows. Only flip on if a 200-step smoke shows peak mem < 11.5GB.
4. **FeatureBank slot-−1 overwrite (model.py:1350)** is fine to keep *now that*
   `FEATURE_BANK_DETACH=False`: history slots carry gradient, slot-−1 injects the
   live frame. Just confirm in the liveness probe that bank history grad-norm>1e-6.

---

# PART 5 — The executable pre-flight (the 6 gates, in order)

Set the bring-up profile, then run each gate. **Do not proceed past a failing gate.**

```bash
export ASSERT_AND_CRASH=1          # NaN raises (no silent 1e-4 sentinel)
# bring-up: caps off, sensitivity 0, ramps off (config or env)
```

| Gate | Command | PASS criterion |
|---|---|---|
| **G0 Provenance** | `git rev-parse HEAD` == the commit you train; `git status` clean; pushed | code trained == code audited |
| **G-C** | Part 1 verify snippet | prints `GAP-C CLOSED: True True False False` |
| **G-A/B static** | the A1/A2/B unit checks (no GPU) | all asserts pass |
| **G1 No-silent-fail** | `TRAIN_MAX_STEPS=200 ... --preset paper_run --subset-ratio 0.25` | zero `[PSR_NAN]`, zero `1e-4` sentinel, zero `[COMBINED_NAN]` |
| **G2 Liveness** | same run, read `[LIVENESS]` | det, head-pose, PSR, activity all **ALIVE** (loss>10×floor, grad-norm>1e-6, std>1e-3) |
| **G3 Commit** | same run, read `[RC-29]` | `committed>0, skipped=0` (FP32) |
| **G4 Labels** | 1-epoch data audit | NA-frac printed; `n_segments`≈#spans; PSR transition targets peak at flips; subset ≥K classes in train&val; `decode(encode(gt))==gt` |
| **G5 Eval** | capped val dry-run | every metric finite; **no exactly-0.0000 stub**; det@0.001 b-boxed+all-frames; activity per-segment; PSR via decoder; no cosmetic NaN |

If G1–G5 all pass on the 0.25 subset → proceed to Part 6.

---

# PART 6 — The paper-run training ladder

Run on the authoritative tree, `--preset paper_run` (now real), FP32, eff-batch 8.

| Stage | Subset | Epochs | Gate to advance | Expected |
|---|---|---|---|---|
| **R1** det bootstrap (`recovery_det_only`) | 0.25 | 3 | b-boxed mAP ≥ 0.05 | 0.05–0.20 |
| **R1.5** + calibrated anchors + (synth pretrain if available) | 1.0 | 15–20 | b-boxed ≥ 0.30 | 0.30–0.55 (0.55–0.75 w/ synth) |
| **R2** joint, activity (CE) on | 0.25 | 4 | activity ≥4 classes, det not −30% | act_top1 ≥ 0.10 |
| **R2.5** PSR transition on (G-A closed) | 0.25 | 4 | ≥3 patterns, transition F1 > 0.30 | psr_f1 0.30→ |
| **R3** full joint `paper_run` | **1.0** | 30–50 | all 5 heads non-zero & improving | §7 targets |
| **R4** finalize (geo head-pose already on) | 1.0 | — | head-pose MAE finite < 35° | 10–25° |
| **R5** multi-seed ×3 + ablations + efficiency | 1.0 | — | mean±std, tables generated | paper-ready |

EMA on at R3 (paper_run sets `use_ema=True`) once metrics move monotonically.
Full det mAP every `DET_METRICS_EVERY_N`=5 epochs; gate-only eval otherwise.

---

# PART 7 — Eval → `\popwres` mapping & per-cell defensibility

| `\popwres` cell | Source metric | Protocol gate | Honest target |
|---|---|---|---|
| Det mAP (b-boxed) | `det_mAP50` @0.001, annotated frames | G5 | 0.55–0.75 (vs 83.80, gap named) |
| Det mAP@0.5 (all-frames) | `det_mAP50_all_frames` | G5 | 0.35–0.55 (vs 64.10) |
| Det mAP@[0.5:0.95] | `det_mAP_50_95` | G5 | 0.35–0.55 |
| Activity Top-1/5 | `compute_activity_segment_metrics` | G-B | 0.30–0.50 / 0.65–0.80 (vs 65.25/87.93, RGB-only noted) |
| PSR F1±3/±5, POS | `decode_and_score_psr` | G-A2 | **0.50–0.65 / 0.75–0.82 — beats STORM 0.506** |
| Assembly F1@1 | top-conf state vs GT, single-state frames | G5 | 0.5–0.7 |
| Error-Verif AP | 1−conf(expected), non-empty error set | G5 | 0.4–0.6 (vs ~0.58) |
| Head-pose MAE | geo head (USE_GEO on via G-C) | G2 | **10–25° (uncontested)** |
| Efficiency | `efficiency_report.py` (no training) | — | params/GFLOPs/FPS, both modes |

A cell is paper-usable iff: finite (I1), gradient-earned & non-zero (I2),
non-degenerate & protocol-correct (I3), comparison fair-or-named.

---

# PART 8 — Final sign-off checklist (the last confirmations before you press train)

Tick all 15. Any unticked = do not start the paper run.

1. ☐ `git status` clean; training commit pushed; HEAD recorded in the run dir.
2. ☐ GAP-C verify prints `True True False`.
3. ☐ GAP-A1 unit: `loss_psr==0` on dim==2, `>0` on dim==3.
4. ☐ GAP-A2 unit: synthetic flip → `psr_f1≈1.0` via decoder.
5. ☐ GAP-B: `n_segments`≈#AR spans; NA never a metric label.
6. ☐ IMG_SIZE rescale patch in; `decode(encode(gt))==gt`.
7. ☐ 200-step smoke: zero sentinels / PSR_NAN / COMBINED_NAN.
8. ☐ Liveness: all 4 heads ALIVE incl. grad-norm > 1e-6 (incl. bank history).
9. ☐ `[RC-29] committed>0, skipped=0` (FP32).
10. ☐ Eval dry-run: every metric finite; no exactly-0.0000 stub; no cosmetic NaN.
11. ☐ Activity eval per-segment, NA excluded, ≥15 classes predicted.
12. ☐ PSR eval uses MonotonicDecoder; ≥3 unique patterns; F1 on full test set.
13. ☐ Seeds fixed (42), cuDNN deterministic, config snapshot saved with the run.
14. ☐ Best checkpoint = RAW model (not EMA); resume restores optim+Kendall+sched.
15. ☐ `efficiency_report.py` numbers captured (params/GFLOPs/FPS, both modes).

---

# APPENDIX — Failure-mode quick table (if a gate fails)

| Symptom at a gate | Cause | Fix |
|---|---|---|
| GAP-C verify prints `False …` | preset still not wiring flags | re-check Part 1a global list + 1b dict keys |
| `loss_psr=0.0001000` | 1e-4 sentinel fired | `ASSERT_AND_CRASH=1`; trace `[PSR_DIAG]` |
| PSR F1 high on subset, low on full | constant pattern on skewed slice | confirm A2 decoder + full test set |
| activity Top-1 ~ NA prevalence | per-recording eval / NA not excluded | GAP-B segment eval |
| det mAP 0 but bestIoU 0.9 | conf too high / anchors | conf 0.001 (done); anchors (done) |
| any metric exactly 0.0000 | stub logged, head skipped | wire eval; distinguish from "no preds" |
| 4 val cycles identical | scaler skipping (RC-29) | FP32; check committed/skipped |
| head-pose NaN | acos>1 / AMP | clamp acos; FP32; USE_GEO on |
| OOM after enabling VideoMAE | 12GB FP32 limit | keep `use_videomae=False`; report w/o-VideoMAE row |

---

## Bottom line

After Parts 1–4 are applied and Part 5 (G0–G5) passes on the 0.25 subset, the
three real gaps (C, A, B) are closed: the `paper_run` preset truly enables the
fixes, PSR learns transitions and is decoded to monotone states (benchmarkable
F1), and activity is measured per action segment (comparable to MViTv2). Run
Part 6, fill Part 7, sign off Part 8 — and **that train is the last IndustReal
train, with results you can put directly in the paper.** It will not beat
YOLOv8m/MViTv2 headline (design ceiling), but every cell will be a real,
defensible number, PSR and head-pose will be competitive/uncontested, and the
unified single-pipeline efficiency story — one forward pass, all five tasks —
will be the paper's headline, backed by numbers that survive review.
