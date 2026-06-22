# 52 — Detection: The Real Diagnosis (Grounded)

> **Generated:** 2026-06-22 (Opus). Companion to `51`. Answers docs 48 (Q1–Q4) and 50 (§2) with code-grounded mechanism, not hypothesis.
> **One-line verdict:** Detection is **fine-grained single-bit assembly-state discrimination**. The model already localizes near-perfectly; the ceiling is a *classification* ceiling between states that differ by one component. OHEM is **not** the primary cause. Do the **1-hour per-class diagnostic** before any 5-epoch ablation.

---

## 1. What the task actually is

`NUM_DET_CLASSES = 24` (`config.py:178`) decomposes as `background + 22 assembly states + error_state`. The 22 "states" are **11-bit binary strings** (`config.py:180-205`):

```
ch1  background
ch2  10000000000      ch13 11110111101
ch3  10010010000      ch14 11110111111
ch4  10010100000      ch15 11110101111
ch5  10010110000      ch16 11110011111
ch6  11100000000      ch17 11110011110
ch7  11110010000      ch18 11110101110
ch8  11110100000      ch19 11100001110
ch9  11110110000      ch20 11101101110
ch10 11110111100      ch21 11101011110
ch11 11110111110      ch22 11101111110
ch12 11110110001      ch23 11101111111
ch24 error_state
```

Each bit = one component present(1)/absent(0). **Adjacent classes differ by a single bit.** Examples:
- `11110111110` (ch11) vs `11110111111` (ch14): differ in **bit 11 only**.
- `11110011111` (ch16) vs `11110111111` (ch14): differ in **bit 6 only**.
- `11110111110` (ch11) vs `11110111100` (ch10): differ in **bit 10 only**.

This is why the **DET_PROBE verdict is permanently "LOCALIZING"** (doc 49 §1.6: `bestIoU_max > 0.90`, `bestIoU_mean ≈ 0.05`): the model **always finds the assembly board** (one big box, high IoU) but **rarely predicts the correct 11-bit state**, because distinguishing two states means detecting whether one screw/washer is present. The box is trivial; the class is a fine-grained visual-difference problem.

**Implication:** comparing `mAP50 = 0.207` to YOLOv8m's `0.838` is a comparison on the same benchmark but not the same difficulty regime for *our* training budget — YOLOv8m had COCO pretrain + **260K synthetic images** + real fine-tune + full data. We have ImageNet-only, real-only, 50% subset, shared backbone across 5 tasks. The gap is explained by **pretraining + data + multi-task load**, not architecture.

---

## 2. The detection signal, decoded from evidence

| Symptom | Evidence | Reading |
|---|---|---|
| Localizes, doesn't classify | DET_PROBE `bestIoU_max>0.90`, `bestIoU_mean~0.05` (49 §1.6) | **Recall/classification-limited**, not localization-limited |
| Confidence never saturates | `score_p99 = 0.2–0.8` (49 §1.6) | Under-confident scoring → poor PR curve → mAP capped |
| Config-independent ceiling | Run1 (4× LR/Bias) ≡ Run2 (1×) to ±0.002 (48) | **No suppressed gradient to amplify** → points to assignment/data/label limit, *not* loss-gain |
| LR restart does nothing | both runs (48) | consistent with near-equilibrium on the *classifiable* subset |
| Binary per-class AP (0 or works) | (48 Q4) | a state either gets enough discriminable positives or it doesn't — threshold behaviour |
| `cls_mean ≈ -7.0`, `near_zero=0` | `rf_stage_state.json → det_health` | head healthy, not collapsed; biased to background (expected with 23:1 bg) |

**Conclusion:** the bottleneck is **fine-grained class separability + tail-class data**, surfaced as low recall on hard states. This is a *data/label/task-difficulty* ceiling, not a *loss-gain* ceiling.

---

## 3. Why OHEM is **not** the primary suspect (correcting the doc-50 hypothesis)

Doc 50 §2.1 names "OHEM + FocalLoss gradient suppression [of] easy positives" as the primary hypothesis. The config already refutes the "positives" half:

```python
# config.py:536-538
DET_ASYMMETRIC_GAMMA = True   # per-class gamma (pos vs neg)
DET_GAMMA_POS = 0.0           # positives get NO focal suppression
DET_GAMMA_NEG = 1.5           # negatives down-weighted at γ=1.5
```

With `γ_pos = 0`, the positive focal term is `(1 - p)^0 = 1` — **positive gradients flow at full strength.** Focal only down-weights *negatives* (and mildly, at 1.5 not 2.0). OHEM (`DET_OHEM_ENABLED=True`, `ratio=2.0`, `min_neg=32`, `config.py:523-538`) keeps **all positives** + the top `2×n_pos` hardest negatives. So neither mechanism starves positive gradient.

Therefore an OHEM-off ablation tests **negative mining only**. It is still worth one cheap run *iff* the per-class diagnostic (below) shows present classes with positives that still won't separate — but it is **not** the first thing to run, and its expected upside is small. **Predicted outcome of OHEM-off: ceiling stays ~0.20–0.23** (consistent with config-independence). If you run it, treat a *null* result as confirmation, not failure.

---

## 4. The 12 dead classes — decomposed, not mysterious

`det_n_present_classes = 16` (`rf_stage_state.json`). So of 24 channels: **8 are background/zero-GT** in the 50% subset (mechanically AP=0, *excluded* from `det_mAP50_pc`), and **~4 are present-with-GT but AP=0** — those four are the only real mystery. The binary pattern is exactly what F2 predicts: a state whose single distinguishing bit is rare/occluded never accumulates discriminable positives.

This is **cheaply testable** and the script to do it already exists: `src/diag_per_class_truth.py` (stdlib-only, no GPU). It separates "background/zero-GT (unmeasurable)" from "has GT but AP=0 (genuinely stuck)" and labels by channel/category_id/name. It just needs a `metrics.jsonl` that contains a **completed detection eval with per-class AP** — the current rf2 log snapshot does not yet have one (it returned "No records with per-class detection data found").

---

## 5. The single most informative diagnostic to run RIGHT NOW (≈1 GPU-hour, or 0 if from logs)

**Do this before any multi-epoch detection experiment.** Three artifacts, in priority order:

1. **Per-class GT count × per-class AP, on the *present* classes** — answer "are the ~4 stuck classes data-starved or separability-starved?"
   - From a completed rf2 eval: `python3 src/diag_per_class_truth.py --run src/runs/rf_stages` (once a detection eval has written per-class AP since the v11 patch).
   - Cross-tabulate: stuck class → GT instances → which **single bit** distinguishes it from its nearest present neighbour.
2. **The 24×24 detection confusion matrix** (already implemented, `evaluate.py` `compute_det_confusion_matrix`, GUIDE 7 P5). Are the ~4 stuck states being predicted **as their one-bit neighbour**? If yes → confirmed single-bit confusion (F2), and the honest paper framing writes itself.
3. **Per-class positive-anchor count** (the aggregate 364–783/img hides per-class starvation). If a present-but-stuck class has positives but is still AP=0 → separability, not assignment. If it has ~0 positives → assignment (then, and only then, consider anchor work).

**Decision tree from the diagnostic:**

```
stuck class has many GT + many positives + confused with 1-bit neighbour
        → fine-grained separability ceiling (EXPECTED). Stop. Reframe paper (53). Do NOT chase.
stuck class has GT but ~0 positive anchors
        → assignment gap. Lower DET_POS_IOU_THRESH to 0.3 OR recalibrate anchors. ~1 cheap run.
stuck class has <50 GT instances
        → data scarcity. Only full-data (100%) helps. Note as limitation; do not block on it.
stuck class predicted as a DIFFERENT (non-neighbour) class
        → label noise. Audit 20 GT boxes for that class.
```

---

## 6. What to actually do about detection (ROI-ordered)

| Action | Cost | Expected Δ mAP50_pc | Verdict |
|---|---|---|---|
| Run the per-class diagnostic (§5) | ~1h or free | 0 (diagnostic) | **DO FIRST** — it decides everything below |
| **Freeze detection at ~0.30, advance to rf3** | 0 | 0 | **DO** — gate already passed (`51` F1) |
| Let rf2 finish to epoch 36 (already running) | ~13h (sunk) | +0.00–0.03 | Passive — keep it running, don't depend on it |
| OHEM-off, 3–5 ep | ~5–7h | likely +0.00–0.02 | **SKIP** unless §5 shows mined-away positives |
| `DET_POS_IOU_THRESH` 0.4→0.3, 3 ep | ~5h | +0.00–0.03 | Only if §5 shows assignment gap on present classes |
| Anchor recalibration | ~0.5h + 5 ep | low | **SKIP** — boxes are large boards; anchors already fit (F9) |
| Full data (100% vs 50%) | ~2× epoch time | +0.02–0.05 (tail classes) | Optional, post-ablation, if time allows |
| Synthetic pretrain (260K imgs) | days | potentially large | **OUT OF BUDGET** — note as future work |

**Recommended detection endgame:** run the diagnostic, let rf2 finish passively, report `det_mAP50_pc` (present-class) and `det_mAP50` (diluted) **both**, frame the gap via the confusion matrix as single-bit state confusion, and move all freed compute to ablations + activity. **Do not spend another targeted GPU-day on detection.**

---

## 7. The realistic detection ceiling (for the paper's expectation-setting)

- On the **honest present-class protocol**, expect a final `det_mAP50_pc` in **0.30–0.38** by epoch 36 at 50% subset.
- The **diluted** `det_mAP50` will trail at **0.20–0.25** (8 zero-GT channels + background).
- Full data + more epochs could plausibly reach `det_mAP50_pc ≈ 0.40–0.45`, but **not** YOLOv8m's regime without synthetic pretraining.
- **This is fine.** The paper does not need to win detection (see `53`). It needs detection to be *non-trivial and honestly characterized as fine-grained state recognition* — which 0.30 pc + a clean single-bit confusion matrix delivers.

---

*Next: `53` for how the paper frames all of this honestly, and `54` for the exact run order.*
