# 18 — The Ultimate Master Guide: POPW → Defensible IndustReal Metrics, End to End

**Purpose.** A single, exhaustive, step-by-step plan to make POPW produce a
**real, non-zero, non-NaN, non-NA, protocol-correct number for every task** on
IndustReal, with the train loss and validation metrics computed correctly end to
end — and to push the *winnable* tasks (PSR, head pose) to match/beat their
baselines while reporting detection and activity honestly as
"competitive-at-a-fraction-of-deployed-compute."

**Read this honestly.** This architecture will **not** beat YOLOv8m (83.80) or
MViTv2 (65.25) on their headline metrics — that is a design ceiling, not a bug
(see §1.3). "Acquiring the goal" here means: **every `\popwres` filled with a
number you can defend in review**, PSR/head-pose competitive, the efficiency/
unification story carrying the paper. That is achievable. Chasing 83.80 is not.

Grounding: all file:line references are to `code/` in this folder
(`losses.py`, `model.py`, `config.py`, `evaluate.py`, `industreal_dataset.py`,
`psr_transition.py`, `train.py`). Verified 2026‑06‑13.

---

## TABLE OF CONTENTS

- **Part 0** — Mental model & the three invariants
- **Part 1** — Pre-flight: telemetry, determinism, assert-and-crash
- **Part 2** — Data & label correctness (fix BEFORE training)
- **Part 3** — Per-task recipes (data→model→loss→train→eval→gate)
- **Part 4** — The training ladder R0→R5 (exact commands, gates, expected values)
- **Part 5** — Evaluation correctness (every metric, every protocol, non-NaN rules)
- **Part 6** — Loss assembly & numerical hygiene (guaranteeing non-zero/non-NaN)
- **Part 7** — The 50-item checklist, ANSWERED
- **Part 8** — Efficiency table (free numbers) & honest paper framing
- **Part 9** — Failure-mode playbook (symptom→diagnosis→fix)
- **Part 10** — Definition of Done & GPU budget

---

# PART 0 — Mental model & the three invariants

Every problem in this project is a violation of one of three invariants. The
guide is organized to enforce them in order. **Do not chase metrics until all
three hold for a head.**

| Invariant | Meaning | How it's violated today | Where enforced |
|---|---|---|---|
| **I1 — Non-NaN** | Every loss & metric is finite | `1e-4` NaN-sentinels + smooth-caps hide inf/NaN (losses.py:1041/1230/1258) | Part 1, Part 6 |
| **I2 — Non-zero** | Gradient reaches every head's params; loss > floor | PSR floored to sentinel; activity bank detached (model.py:1188) | Part 3, Part 6 |
| **I3 — Non-degenerate** | Output isn't constant / NA-collapsed; metric measures the model | Activity NA-collapse + recording-level eval; PSR constant pattern | Part 2, Part 5 |

**The cardinal rule:** *measurement correctness before training, liveness before
SOTA, honesty in framing.* A number that is non-NaN but measured under the wrong
protocol (today's activity Top-1) is **worse** than no number — it will be
attacked in review.

---

# PART 1 — Pre-flight (do once, before any multi-hour run)

### 1.1 Determinism & the telemetry line
Already present (RC-29): FP32 enforced, per-epoch `committed/skipped/scaler`
summary. **Rule:** every run prints `[RC-29] optimizer windows: committed=N
skipped=0`. If `skipped>0` in FP32, stop — something is wrong.

### 1.2 Replace silent guards with assert-and-crash (bring-up only)
The codebase has ~15 layers that convert bugs into silent degradation. During
bring-up, flip them to crash so bugs surface in 200 steps, not 8 GPU-hours:

```python
# config.py — bring-up profile (revert for production)
ASSERT_AND_CRASH = True          # NaN → raise, do not replace with 1e-4
DET_LOSS_CAP = 1e9               # disable smooth-cap during bring-up
POSE_LOSS_CAP = 1e9
ACTIVITY_LOSS_CAP = 1e9
PSR_LOSS_CAP = 1e9
HEAD_POSE_LOSS_CAP = 1e9
PSR_SENSITIVITY_WEIGHT = 0.0     # remove -log(std) term while diagnosing PSR
```
And in `losses.py`, gate every `1e-4` fallback (lines 1041, 1230, 1258) behind
`if not C.ASSERT_AND_CRASH: <fallback> else: raise FloatingPointError(name)`.
**Why:** you cannot trust "non-zero" until you've proven no sentinel is firing.

### 1.3 Per-head liveness probe (run at step 0 and after every enable)
Add a one-shot probe that prints, per head: output logit `mean/std/min/max`,
loss value, and **gradient norm of the head's first and last layer**. A head is
*alive* iff: loss > 10× its floor, grad-norm > 1e-6, and output std > 1e-3.

```
[LIVENESS] det   loss=4.21  gnorm(first/last)=3e-2/1e-1  out std=0.42  ALIVE
[LIVENESS] act   loss=4.05  gnorm=2e-2/8e-2              out std=0.31  ALIVE
[LIVENESS] psr   loss=0.18  gnorm=1e-2/4e-2             out std=0.22  ALIVE
[LIVENESS] hpose loss=0.33  gnorm=5e-3/2e-2             out std=0.18  ALIVE
```
**This probe is the single highest-leverage tool in the project.** It directly
tests I1+I2 per head and would have caught the PSR floor and the detached bank.

---

# PART 2 — Data & label correctness (the metrics get their MEANING here)

> Fix all of Part 2 **before** spending GPU-days. A perfectly trained model on
> broken labels produces broken numbers. This is where today's activity metric
> is unmeasurable and subset runs are untrustworthy.

### 2.1 Detection GT ↔ image-size coupling (latent landmine)
**Problem (checklist #6):** `_load_image` resizes every frame to `IMG_SIZE`
(industreal_dataset.py:891) but **never rescales boxes**; anchors+GT are
normalized by `C.IMG_WIDTH/HEIGHT`. Correct *only* because `IMG_SIZE==1280×720`
(config.py:257). Any smaller `IMG_SIZE` silently zeroes detection.

**Fix:** couple resize to box scale, and assert at load:
```python
# industreal_dataset._load_image / _extract_boxes_from_coco
sx, sy = IMG_SIZE[0]/native_w, IMG_SIZE[1]/native_h
boxes[:, [0,2]] *= sx ; boxes[:, [1,3]] *= sy
assert C.IMG_WIDTH == IMG_SIZE[0] and C.IMG_HEIGHT == IMG_SIZE[1], \
    "Anchor normalization space must equal image size"
```
**Verify:** decode(encode(gt))==gt at the *current* IMG_SIZE; DET_PROBE bestIoU
unchanged after setting IMG_SIZE=(960,540). (I1/I3 for detection.)

### 2.2 Activity: segment-level clip sampler + NA masking (THE activity fix)
**Problem (checklist #17–20):** No per-action-segment, 16-uniform-frame sampler
exists. The MViTv2 protocol is *one prediction per action segment from 16 frames*.
The current eval aggregates per **whole recording** (evaluate.py:628) and never
excludes NA, and **no `activity_mask` is ever emitted** (both collates), so NA is
trained and scored as a real class on NA-dominated data → "predict NA" wins.

**Fix — three coordinated changes:**

1. **Build a segment index** from `AR_labels.csv` spans: one entry per
   `(recording, action_id, start, end)`. This is the unit of evaluation.
2. **Add an `ActivityClipDataset`** that, given a segment, samples **16 uniform
   frames** across `[start, end]`, returns `[16,3,H,W]` + the single segment
   label. Train *and* eval activity on these clips.
3. **Emit `activity_mask`** in both collates and **exclude NA** from the
   clip-level metric (NA frames may still be sampled for context but the *label*
   for a metric clip is the action, never NA).

```python
# dataset: segment-level clip sampling (per MViTv2 protocol)
def sample_clip(self, seg, T=16):
    idxs = np.linspace(seg.start, seg.end, T).round().astype(int)
    frames = torch.stack([self._load_frame(seg.rec, i) for i in idxs])  # [T,3,H,W]
    return frames, seg.action_id           # single label per clip, never NA
```
**Eval (replace recording-level aggregation):**
```python
# evaluate.py — clip-level Top-1/5 per ACTION SEGMENT (not per recording)
preds = [model.clip_forward(clip).argmax() for clip in segment_clips]   # 1/clip
top1 = mean(pred == seg.label for pred, seg in zip(preds, segments))
top5 = mean(seg.label in clip_logits.topk(5) for ...)
```
**Verify (I3):** `pred_seen ≥ 15` distinct classes; Top-1 computed over
**segments**, not frames; NA never appears as a metric label. Print the
confusion-matrix diagonal to prove non-NA-collapse.

### 2.3 PSR: verify the data assumptions, then predict transitions
**Problem (checklist #29–32):** Fill-forward is correct (dataset:483–490), but
(a) the "95% static" and "-1 fraction" are *assumptions the authors never
verified* (the live `[PSR_DIAG]` at losses.py:763 exists precisely to find out),
and (b) `-1` is carried forward as a persistent state (dataset:488) — an error
component stays `-1` for all later frames, which may over-count ignores.

**Fix:**
1. **Measure first.** Run a 1-epoch data pass that prints, per component:
   `%static`, `%==1`, `%==-1`, and the transition count. Decide the objective
   from data, not assumption.
2. **Switch the objective to transitions** (`psr_transition.py` is ready):
   `build_transition_targets(psr_labels, σ=3)` → Gaussian-smeared 0→1 events;
   `MonotonicDecoder` enforces monotone fill-forward; add a procedure-order prior.
3. **Make `-1` transient** (ignore only the error frame, not all subsequent), or
   keep persistent if that matches IndustReal semantics — but **decide from the
   format spec**, don't leave it ambiguous.

**Verify (I2/I3):** PSR raw focal loss O(0.1–0.3) and finite (not 1e-4);
≥3 unique predicted patterns; transition F1 > 0 on held-out.

### 2.4 Subset stratification (so subset runs are trustworthy)
**Problem (checklist #22):** `--subset-ratio` takes the **first N recordings
alphabetically** (industreal_dataset.py:1104) — entire action classes can be
excluded.

**Fix:** select recordings to **maximize class coverage**, deterministically:
```python
# greedy coverage: pick recordings until all (or max) AR classes are represented
chosen, covered = [], set()
for rec in sorted(recs, key=lambda r: -len(rec.classes - covered)):
    chosen.append(rec); covered |= rec.classes
    if len(chosen) >= max_recordings and covered >= target_classes: break
```
**Verify:** print class histogram of the chosen subset; assert ≥ K classes
present in **both** train and val subsets.

### 2.5 Collate consistency
**Problem (checklist #19,21):** `collate_fn_sequences` drops `clip_rgb`
(dataset:1324); it's selected for *both* loaders when `USE_PSR_SEQUENCE_MODE=True`
(train.py:106). Dormant only because VideoMAE is off — re-arms on enable. Neither
collate emits `activity_mask`.

**Fix:** make `clip_rgb` and `activity_mask` present in **both** collates; select
the sequence collate **only** for the dedicated PSR-sequence loader, never for the
per-frame train/val loaders.
**Verify:** assert identical key-sets between train and val batches.

---

# PART 3 — Per-task recipes (end to end)

Each recipe: **data → model → loss → train → eval → expected curve → gate.**
Order of bring-up: **Detection → Head pose → PSR → Activity → Assembly/Error.**
(Detection first because `det_conf` feeds activity and PSR consumes assembly state.)

## 3.1 Detection (ASD) — vs YOLOv8m 83.80 (b-boxed) / 64.10 (all-frames)

**Data:** annotated frames only for the b-boxed metric; all frames for the
all-frames metric. Boxes in pixel xyxy at IMG_SIZE (§2.1).

**Model:** RetinaNet head is **correct** (model.py:487, 4-conv subnets+GroupNorm,
focal prior π=0.03 at :524, decode at :1704). Do **not** redesign the head.

**The three levers that actually move detection mAP:**
1. **Anchor calibration (checklist #7).** `ANCHOR_SIZES=(24…384)` vs GT 146–594px
   → only ~1.6% anchors match. Run k-means on GT, set sizes to clusters
   (e.g. ~(96,160,256,384,512)). Biggest single recall win.
2. **Synthetic pretrain (checklist #8).** `PRETRAIN_DET_ON_SYNTH=True`
   (config.py:386) is wired but unused. Pretrain backbone+FPN+det head on
   synthetic, then fine-tune. This is what separates 0.5 from 0.7+.
3. **Eval threshold (checklist #9).** Report at `DET_EVAL_SCORE_THRESH=0.001`
   (YOLOv8 standard) for the comparison; 0.02 understates you.

**Loss:** Focal(α=0.75,γ=2)+GIoU, empty frames skipped, normalize by
`n_img_with_gt` — already correct (losses.py:228,295).

**Train:** `recovery_det_only` → joint. FP32, eff-batch 8.

**Eval:** `compute_det_metrics_extended` (b-boxed) + `_all_frames` variant;
COCO all-point AP (evaluate.py:1153) is correct.

**Expected trajectory (b-boxed mAP@0.5):**

| Stage | Config | Expected |
|---|---|---|
| R1 ep0–3 | det_only, default anchors, no synth | 0.05 → 0.20 |
| R1.5 | + anchor calib | 0.20 → 0.40 |
| R1.5 | + synth pretrain, full data | 0.40 → **0.55–0.75** |
| Ceiling | shared backbone, no more data | ~0.75 |

**Gate:** R1 → 0.05; R1.5 → 0.30; report when ≥0.50 and not regressing under joint.
**Honest target:** 0.55–0.75 b-boxed (vs 83.80), 0.35–0.55 all-frames (vs 64.10),
framed as "competitive at 1/3 deployed compute, no synthetic-scale pretraining."

## 3.2 Head pose (9-DoF) — no baseline = FREE WIN

**Model:** swap raw-9-number MLP for `head_pose_geo.py` (6D continuous rotation
+ normalized position), enable `USE_GEO_HEAD_POSE=True`.
**Loss:** geodesic on rotation + L2 on position (the module provides it).
**Eval:** Forward/Up angular MAE (L2-normalize vectors before dot), Position MAE
(mm). Guarantee non-NaN by clamping `acos` input to [-1+ε, 1-ε].
**Expected:** raw-MSE gives ~60–70°; geometry-aware → **10–25°**.
**Gate:** any finite MAE < 35° is publishable (uncontested row). **Verify (I1):**
no NaN under FP32 (the old NaN was the AMP/RC-29 issue).

## 3.3 PSR — vs B2 0.731 / STORM-PSR 0.506 (YOUR BEST WINNABLE TASK)

**Data:** transition targets via `build_transition_targets(σ=3)` (§2.3).
**Model:** existing multi-scale GAP → Causal Transformer (3L/4H) → 11 MLPs is
fine; add `MonotonicDecoder` (psr_transition.py:79) + procedure-order prior.
**Loss:** event-detection focal on Gaussian-smeared transitions (NOT per-frame
BCE on static labels). Remove/clamp the sensitivity penalty (losses.py:1188).
**Train:** enable at R2.5 with the raw-loss probe (Part 1) on.
**Eval:** F1(±3)/F1(±5) bi-directional greedy matching of *transition events*;
POS = fraction of correctly-ordered adjacent pairs; Edit = Damerau-Levenshtein
on state-change sequences, GT-normalized.
**Expected:** per-frame focal → constant-pattern artifact (do not report).
Transition model → **F1 0.50–0.65** (beats STORM 0.506, approaches B2 0.731),
POS 0.75–0.82.
**Gate:** ≥3 unique patterns, transition F1 > 0.30 at R2.5; report at ≥0.50.
**Verify (I2/I3):** `[PSR_DIAG]` shows raw loss O(0.1–0.3) finite; predicted
state changes track GT changes within tolerance.

## 3.4 Activity — vs MViTv2 65.25/87.93 (RGB)

> This is the **hardest** and the one whose *measurement* is currently broken.
> Do §2.2 first or any number here is meaningless.

**Data:** segment-level 16-frame clips (§2.2). Train and eval on clips.
**Model:** the per-frame CNN→2-ViT→FC path is **structurally too weak** (verified:
bank detached at model.py:1188, slot-−1 overwrite at :1340, VideoMAE off). To
have any chance at 0.40+:
1. **Enable the K400 video stream** (`USE_K400_VIDEO_STREAM`/VideoMAE-v2) as the
   **primary** activity path — real spatiotemporal attention over the clip.
2. **Fix the FeatureBank** *or bypass it*: feed the clip tubelets through the ViT
   directly; if keeping the bank, **remove `.detach()`** (model.py:1188) and the
   slot-−1 overwrite (model.py:1340), and feed **temporally-contiguous** frames.
3. **Deepen** to 4–6 temporal blocks once tokens are genuinely temporal.
**Loss:** **plain CE + label smoothing 0.1**, `USE_LDAM_DRW=False`
(s=30 stacked on CB-sampling+LS causes 1-class collapse). Add LDAM later at
s=10–15 only if long-tail recall is the *sole* remaining gap.
**Eval:** clip-level Top-1/5 per segment (§2.2), NA excluded.
**Expected:** GAP-only per-frame → <0.20 (don't report). + VideoMAE + clips +
CE → **0.30–0.50 Top-1, 0.65–0.80 Top-5**.
**Gate:** R2 → ≥4 classes & Top-1 ≥ 0.10; report at ≥0.30 Top-1.
**Honest target:** 0.30–0.50 Top-1 (vs 65.25), with the RGB-only vs RGB+VL+stereo
modality gap named, framed as efficiency.

## 3.5 Assembly State F1@1 & Error-Verification AP

**Derivation (no new training):** both come from the detection head's per-frame
class confidences.
- **F1@1:** top-confidence detected state vs GT state, single-annotated-state
  frames only. Compute precision/recall over frames, F1.
- **Error-Verif AP:** score = `1 − confidence(expected_state)`; binary AP over
  error vs no-error frames (Lehman 2024 protocol).
**Expected:** F1@1 0.5–0.7 (vs SupCon ~0.83 — a contrastive specialist; don't
expect to beat); Error-Verif AP 0.4–0.6 (vs ~0.58 — competitive).
**Verify (I1):** AP over a non-empty error set; guard divide-by-zero with
`max(denom, 1)`.

---

# PART 4 — The training ladder (chronological execution)

Each stage: command · what's on · gate · expected · what to watch. **Run a
200-step smoke (`TRAIN_MAX_STEPS=200`) with the liveness probe before every
multi-hour stage.** Subset 0.25 for liveness, 1.0 for paper numbers.

### R0 — Freeze-proof smoke (~20 min) — DONE pattern
```bash
TRAIN_MAX_STEPS=200 python3 src/training/train.py --preset recovery_det_only \
  --subset-ratio 0.25 --max-epochs 1 --seed 42
```
Gate: `committed>0, skipped=0`; det cls loss trends down; liveness=ALIVE for det+hpose.

### R1 — Detection bootstrap (running)
```bash
python3 src/training/train.py --preset recovery_det_only \
  --subset-ratio 0.25 --max-epochs 3 --seed 42
```
Gate: **b-boxed mAP@0.5 ≥ 0.05**, DET_PROBE bestIoU>0.5. Expected 0.05–0.20.

### R1.5 — Detection to competitive (anchors + synth + scale)
```bash
python3 scripts/training/calibrate_anchors.py     # set ANCHOR_SIZES
# enable PRETRAIN_DET_ON_SYNTH; pretrain, then:
python3 src/training/train.py --preset recovery_det_only \
  --subset-ratio 1.0 --max-epochs 20 --seed 42 --resume <synth_pretrain.pth>
```
Gate: b-boxed ≥ 0.30 (→0.55–0.75). Watch: precision/recall both rising; score std > 0.05.

### R2 — Add activity (CE, clips), keep PSR off
```bash
# config: USE_LDAM_DRW=False, activity clip dataset on, USE_K400_VIDEO_STREAM=True
python3 src/training/train.py --preset recovery \
  --resume <R1.5_best.pth> --subset-ratio 0.25 --max-epochs 4 --seed 42
```
Gate: activity `pred_seen ≥ 4`, Top-1 ≥ 0.10, det not −30%. Watch: NA not dominating preds.

### R2.5 — Add PSR (transition objective + probe)
```bash
# config: USE_PSR_TRANSITION=True, PSR_SENSITIVITY_WEIGHT clamped, probe ON
python3 src/training/train.py --preset recovery \
  --resume <R2_best.pth> --subset-ratio 0.25 --max-epochs 4 --seed 42
```
Gate: PSR ≥3 patterns, transition F1 > 0.30, raw loss O(0.1–0.3). Watch: `[PSR_DIAG]`.

### R3 — Full-data joint, protocol-correct eval, EMA on
```bash
python3 src/training/train.py --preset recovery \
  --resume <R2.5_best.pth> --subset-ratio 1.0 --max-epochs 50 --seed 42
# USE_EMA=True (decay 0.999) once metrics move monotonically
```
Gate: all 5 heads non-zero & improving; clip-level activity; b-boxed+all-frames det.
Expected: the §3 per-task targets.

### R4 — Geometry head pose + assembly/error derivations + polish
Enable `USE_GEO_HEAD_POSE`; compute F1@1 + Error-Verif AP from det outputs.

### R5 — Multi-seed (±std) + ablations + efficiency table
```bash
python3 scripts/training/run_multi_seed.py --seeds 42 2024 1337
python3 scripts/training/efficiency_report.py ...   # see Part 8
python3 scripts/training/generate_paper_tables.py
```
Gate: every `\popwres` filled with mean±std, protocol-correct, non-degenerate.

---

# PART 5 — Evaluation correctness (every metric, non-NaN/non-NA guaranteed)

For **each** metric: protocol source, exact computation, and the explicit guard
that prevents NaN/NA. The Val-line formatter must use `.get(k, float('nan'))`
and **only print a metric when its head was evaluated** (the current stub-key
mismatch prints cosmetic NaN — checklist #44/Val-line issue).

| Metric | Protocol | Computation | Non-NaN / non-NA guard |
|---|---|---|---|
| **Det mAP (b-boxed)** | Schoonbeek '24 T3, annotated frames | COCO all-point AP@0.5, conf 0.001, NMS | skip class if total_gt=0; mean over classes-with-GT |
| **Det mAP@0.5 (all-frames)** | full videos | same, all frames (empty dilute) | `rec = tc/max(total_gt,1)` |
| **Det mAP@[0.5:0.95]** | COCO | mean over 10 IoU thresh | `compute_ap_multi_thresh` |
| **Activity Top-1/5** | MViTv2 clip-level, 16f/segment | 1 pred/segment, NA excluded | assert ≥1 segment; mask NA labels |
| **PSR F1(±3/±5)** | PSRT paper | bi-dir greedy match of transition events | empty-GT recording → skip, not 0/0 |
| **PSR POS** | PSRT paper | correctly-ordered adjacent pairs | `max(n_pairs,1)` |
| **PSR Edit** | STORM convention | Damerau-Levenshtein, GT-normalized | `1 - dist/max(len(GT),1)` |
| **Assembly F1@1** | Schoonbeek '24 | top-conf state vs GT, single-state frames | `max(prec+rec,1e-9)` |
| **Error-Verif AP** | Lehman '24 | binary AP, score=1-conf(expected) | non-empty error set assert |
| **Head pose MAE** | (none) | angular MAE (clamp acos), pos MAE mm | `acos(clamp(dot,-1+1e-6,1-1e-6))` |
| **Combined** | internal selection | 0.30·mAP+0.35·actF1+0.15·hp+0.20·psrF1 | clamp each component finite (train.py:1700) |

**The non-NA rule for activity (I3):** a metric clip's label is *always* an
action id, never NA. NA frames may provide temporal context but are never the
prediction target for a reported number. Print the per-class diagonal to prove it.

**The non-zero rule for eval:** if a head reports exactly 0.0000, it's almost
always (a) the head was skipped but a 0-stub was logged, or (b) all preds were
thresholded away. Distinguish these — 0 from "no preds above thresh" means lower
the threshold; 0 from "stub" means wire the eval.

---

# PART 6 — Loss assembly & numerical hygiene (guaranteeing I1+I2)

The Kendall pipeline (losses.py:1271–1358) is correct in structure; the danger is
the **sentinels masking dead gradients**. Bring-up procedure:

1. **Disable caps/sentinels** (Part 1.2) so any NaN crashes with the head name.
2. **Prove each head's gradient is non-zero** via the liveness probe before
   adding the next head. Add heads one at a time (det → +hpose → +psr → +act).
3. **Kendall init:** `s_det=0, s_pose=0, s_act=0, s_psr=0` (neutral precision=1).
   The `s_pose=-1` legacy can be zeroed by the clamp; keep at 0 (losses.py:879).
4. **No ramp during bring-up** (`STAGED_TRAINING=False`) — ramps multiply
   gradient suppression and confound attribution.
5. **Re-enable production guards** only after all heads are ALIVE for ≥1 epoch,
   and keep the telemetry that *logs* when a guard fires (a guard that fires
   silently is a bug; a guard that fires loudly is insurance).

**Per-head non-zero loss floors (what "alive" looks like at init):**

| Head | Floor (dead) | Alive at init |
|---|---|---|
| det cls (focal) | constant ≈ bias output | 3–30 (GT frames) |
| head pose | — | 0.2–0.5 |
| PSR (focal, transitions) | **1e-4 (sentinel)** | 0.1–0.3 |
| activity (CE, 75-way) | ln(75)≈4.32 const | 3–4.3 |

If PSR shows 1e-4 → a sentinel fired → fix upstream (Part 2.3 / §3.3), never
accept the floor.

---

# PART 7 — The 50-item checklist, ANSWERED

Format: **#. status — root cause → fix → verify.**
(✅ ok · ❌ broken · ⚠️ latent · ❓ confirm-by-running)

### A. Detection
1. ✅ xywh→xyxy (dataset:1009) → none → decode(encode(gt))==gt.
2. ✅ category remap 1-24→0-23 (dataset:1012) → none → assert labels∈[0,23].
3. ✅ box decode (model:1704) → none → IoU on positives >0.8.
4. ✅ det head 4-conv+GN, π=0.03 (model:524) → none → step-0 median|z|≈3.
5. ✅ empty-frame skip + norm (losses:228,295) → none → GT-only gradient.
6. ⚠️→fix box rescale on resize + assert IMG_SIZE==anchor space (§2.1) → DET_PROBE stable at IMG_SIZE=(960,540).
7. ❌→fix run `calibrate_anchors.py`, set GT-cluster sizes (config:247) → recall↑, anchors cover 146–594px.
8. ❌→fix enable `PRETRAIN_DET_ON_SYNTH` pipeline (config:386) → b-boxed 0.4→0.6+.
9. ⚠️→fix report at `DET_EVAL_SCORE_THRESH=0.001` (config:344) → comparable to YOLOv8.
10. ❓ confirm b-boxed eval uses annotated frames only → log frame count == annotated count.
11. ❓ confirm "mAP (b-boxed)" reported at IoU0.5 matching 83.80 def → protocol note in table caption.

### B. Activity
12. ✅ ViT scale `*scale` (model:1101) → none → attn entropy not one-hot.
13. ✅ det_conf sigmoid+detach (model:1865) → none → activity input L2 O(1).
14. ❌→fix FeatureBank fed contiguous clips or bypassed (§3.4) → bank holds [t-15..t].
15. ❌→fix remove `.detach()` (model:1188) → bank grad-norm > 1e-6.
16. ❌→fix remove slot-−1 overwrite (model:1340) → all T positions learnable.
17. ❌→fix add segment-level 16-frame sampler (§2.2) → 1 clip/segment.
18. ❌→fix clip-level eval per segment not recording (evaluate:628) → Top-1 over segments.
19. ❌→fix emit `activity_mask`, exclude NA (both collates) → NA never a metric label.
20. ⚠️→fix delete stale "-1 excluded" comment; implement real masking (evaluate:2923) → mask is all-True only if truly all-labeled.
21. ❌→fix enable VideoMAE/K400 as primary; verify checkpoint loads (config:73) → not 3D-CNN fallback.
22. ❌→fix stratified subset selection (dataset:1104) → class histogram balanced.
23. ⚠️→fix `USE_LDAM_DRW=False`, CE+LS (config:426) → ≥4 classes predicted.
24. ❓ confirm 74/75 classes present at subset 1.0 → class histogram.

### C. PSR
25. ✅ fill-forward (dataset:483) → none → state monotone within recording.
26. ✅ -1 ignore plumbed (losses:729) → none → masked entries contribute 0.
27. ❌→fix eliminate 1e-4 sentinel path for PSR (losses:1041/1230/1258) → raw loss O(0.1–0.3).
28. ⚠️→fix clamp `-log(std)` sensitivity to [0,5] or remove (losses:1188) → loss finite at T=1.
29. ⚠️→decide -1 transient vs persistent from spec (dataset:488) → ignore-fraction sane.
30. ❓ measure -1 fraction (losses:763 diagnostic) → printed per component.
31. ❓ measure %static per component → printed; choose objective from data.
32. ❌→fix wire `USE_PSR_TRANSITION` + MonotonicDecoder (psr_transition) → transition F1>0.

### D. Head pose / assembly / error
33. ✅ head pose head present → none → MAE term in combined.
34. ⚠️→fix enable `USE_GEO_HEAD_POSE` (6D+geodesic) → MAE 60°→10–25°.
35. ❓ confirm finite under FP32 → no NaN in val.
36. ❓ derive F1@1 from det, verify vs SupCon protocol → single-state frames.
37. ❓ Error-Verif AP = 1-conf(expected), verify vs Lehman → binary AP.
38. ⚠️ ensure head pose number non-degenerate (uncontested but must be real) → MAE < 35°.

### E. Eval & metrics
39. ✅ COCO all-point AP (evaluate:1153) → none → AP∈[0,1].
40. ✅ greedy score-sorted match (evaluate:1128) → none → no double-count.
41. ✅ NMS in active path (evaluate:3020) → none → dedup preds.
42. ✅ b-boxed vs all-frames separated (evaluate:1094/1167) → none → two numbers.
43. ⚠️ minor greedy deviation from COCO (global argmax) → negligible at 0–3 GT/frame; optional best-available match.
44. ❓→fix Val-line formatter `.get(k, nan)`; print metric only if head evaluated → no cosmetic NaN.

### F. Training / plumbing / infra
45. ✅ FP32 + RC-29 telemetry (config:578) → none → committed>0,skipped=0.
46. ✅ EMA off for recovery; best=raw (config:586) → none → best.pth not EMA blend.
47. ✅ mixup off (config:585) → keep off until mixes images → no label corruption.
48. ⚠️→fix `ASSERT_AND_CRASH=True` during bring-up; loud guards in prod (Part 1.2) → NaN surfaces by head.
49. ⚠️→simplify Kendall ramps/caps during bring-up (Part 6) → attributable gradients.
50. ❓ 200-step smoke + liveness probe after every change (Part 1.3) → all heads ALIVE before multi-hour run.

---

# PART 8 — Efficiency table (free numbers) & honest framing

**Fill today, no training** (`count_parameters` model.py:1996; fvcore in
`efficiency_report.py`):
```bash
python3 scripts/training/efficiency_report.py --backbone convnext_tiny --batch_size 1  # streaming
python3 scripts/training/efficiency_report.py --backbone convnext_tiny --batch_size 8  # batched
python3 scripts/training/efficiency_report.py --use_videomae                            # w/ VideoMAE
```
Known: **76.16M total / 53.42M trainable**; paper rows **53.3M (w/o VideoMAE)**,
**75.3M (w/ VideoMAE)**.

**Honesty flag:** w/ VideoMAE (75.3M) ≈ sum of baselines (YOLOv8m 25.9 + MViTv2
34.5 + STORM ~15 = 75.4M). So **do not claim "fewer parameters" with VideoMAE
on.** Lead the efficiency story with **one forward pass / one pipeline / shared
backbone / streaming throughput**, and report the **53.3M w/o-VideoMAE** config
as the genuinely-smaller variant (with its lower activity number stated).

---

# PART 9 — Failure-mode playbook (symptom → diagnosis → fix)

| Symptom | Diagnosis | Fix |
|---|---|---|
| metric = exactly 0.0000 | head skipped (stub) OR all preds thresholded | wire eval / lower conf to 0.001 |
| metric = NaN on Val line | stub-key mismatch in formatter | `.get(k, nan)`; print only evaluated heads |
| loss = 0.0001000 (psr) | 1e-4 NaN-sentinel fired | Part 2.3 / §3.3; `ASSERT_AND_CRASH` |
| 4 val cycles identical to 4 decimals | scaler skipping steps (RC-29) | FP32; check committed/skipped |
| activity predicts 1 class | LDAM s=30 collapse OR NA dominance | `USE_LDAM_DRW=False`; NA mask + clips |
| det mAP 0 but bestIoU 0.9 | cls confidence flat / thresh too high | train cls; conf 0.001; anchor calib |
| PSR F1 looks great (0.73) on subset | constant pattern on skewed slice | transition objective; full test set |
| det loss → 0, no positive matches | IMG_SIZE≠anchor space (box scale) | §2.1 rescale + assert |
| head pose NaN | acos(>1) or AMP | clamp acos; FP32 |
| grad-norm 0 for a head | detached path / frozen / ramp=0 | remove detach; check stage ramp |

---

# PART 10 — Definition of Done & GPU budget

**Done = every `\popwres` filled with a number that is:** finite (I1), non-zero &
gradient-earned (I2), non-degenerate & protocol-correct (I3), and either
competitive or with the gap explicitly named.

**Per-cell acceptance:**

| Cell | Accept when | Target |
|---|---|---|
| Det b-boxed / all-frames | conf 0.001, annotated-frames protocol, not regressing | 0.55–0.75 / 0.35–0.55 |
| Det [0.5:0.95] | COCO multi-thresh | 0.35–0.55 |
| Activity Top-1/5 | segment clip-level, NA-excluded, ≥15 classes | 0.30–0.50 / 0.65–0.80 |
| PSR F1±3/±5, POS | transition events, full test set | 0.50–0.65 / 0.75–0.82 |
| Assembly F1@1 | single-state frames | 0.5–0.7 |
| Error-Verif AP | non-empty error set | 0.4–0.6 |
| Head pose MAE | finite, geometry-aware | 10–25° |
| Efficiency | measured both modes | params/GFLOPs/FPS filled |

**GPU budget (single RTX 3060, with the eval fix from §5/checklist #44):**

| Phase | GPU-days |
|---|---|
| Day-1 free (efficiency, anchor calib, data audits) | ~0.5 |
| R1→R1.5 detection competitive | 3–5 |
| R2→R2.5 activity+PSR alive | 3–4 |
| R3 full-data joint | 7–12 |
| R4 head-pose/assembly/error | 1–2 |
| R5 multi-seed ×3 + ablations | 5–10 |
| **Total** | **~20–35** |

Minimum-viable paper (all cells non-zero & honest, PSR+head-pose competitive,
efficiency): **~10–14 GPU-days.**

**Two accelerators worth taking:** (1) the embedding-cache two-stage path
(`EMBEDDING_CACHE_DIR`) for activity/PSR — "hundreds of epochs/hour" from cache;
(2) a cloud A100 for the final R3 + multi-seed only — compresses the 3–5 week
tail to ~1 week.

---

## Closing

The model works; the architecture is fixed; what remains is **measurement
correctness (Part 2 + Part 5), liveness per head (Part 1 + Part 6), and the two
targeted wins (transition-PSR §3.3, geometry head-pose §3.2)** on top of a free
efficiency table (Part 8). Fix the labels and protocols first, prove every head
ALIVE with the liveness probe, then spend the GPU-days — and every `\popwres`
becomes a number you can defend in review. Beating YOLOv8m/MViTv2 is not the
goal and not on the table; a publishable unified-efficiency paper with honest,
non-degenerate numbers across all five tasks **is**, and this guide is the path
to it.
