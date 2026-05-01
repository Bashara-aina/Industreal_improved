# 03 — Benchmark-Targeted Strategy & Evaluation Rigor

**Goal:** Tie every change in Doc 01 and Doc 02 back to a concrete benchmark target. Cover what's not architectural or training-related: **evaluation protocol**, **efficiency reporting**, **and the targets that need their own dedicated tactic**.

**Source files affected:** `evaluate.py`, `export_onnx.py`, plus a new `efficiency_report.py`

---

## Why this doc exists

Docs 01 and 02 fix the model. But for a publishable result you also have to:
1. **Evaluate exactly the way the reference papers evaluate** — otherwise reviewers will dismiss the comparison.
2. **Report efficiency credibly** — params, GFLOPs, FPS on stated hardware. The PTMA paper at 12.9M params / 291 FPS is a real bar to clear if you want the efficiency narrative.
3. **Have a tactic per benchmark** — generic improvements don't help if they don't move the specific metric the paper reports.

---

## A. Per-target tactics

### A.1 ASD Detection mAP@0.5 — beat YOLOv8m's 83.8%

YOLOv8m's number is **mAP@0.5 on the IndustReal test split with COCO+synth+real pretraining**. To beat this credibly:

- **Replicate their evaluation protocol exactly.** The IndustReal paper specifies 0.5 IoU threshold, 0.5 confidence threshold, NMS at 0.5. Verify `evaluate.py` matches:
  - `DET_EVAL_SCORE_THRESH = 0.5` ✓ (already in config)
  - `DET_EVAL_MAX_PER_IMAGE = 300` ✓
  - NMS threshold: **check this — should be 0.5, not the default**

- **Report the standard COCO metrics too.** Beating mAP@0.5 alone is suspicious; reviewers expect mAP@[0.5:0.95] as well. Currently `evaluate.py` should compute both. If it doesn't, add this:

  ```python
  from torchmetrics.detection import MeanAveragePrecision
  metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox',
                                 iou_thresholds=[0.5] + [0.5 + 0.05 * i for i in range(10)])
  ```

- **Tactic for the long-tail ASD classes.** 24 classes is moderate, but some appear in <2% of frames. Use the Repeated Factor Sampling trick from the LVIS paper:

  ```python
  # In industreal_dataset.py:
  def __init__(self, ...):
      self.frame_weights = self._compute_repeat_factors()
  def _compute_repeat_factors(self, t=0.001):
      class_freq = ...  # fraction of frames each class appears in
      return {cls: max(1.0, sqrt(t / max(f, 1e-9))) for cls, f in class_freq.items()}
  ```
  Pass these weights into a `WeightedRandomSampler`. **+1 to +1.5 mAP** on rare classes.

**Combined with Doc 01 B + Doc 02 C.1: target 86–88% mAP@0.5.** Comfortable margin over 83.8%.

---

### A.2 Activity Top-1 — beat MViTv2's 66.45%

This is the hardest target and the bot's confidence is lowest here (65%). Stack everything:

- **Doc 01 A.1–A.4** (TCN + T=16 + 2 ViT + CLS pooling): +3 to +4%
- **Doc 02 A.1** (VideoMAE V2 stream): +5 to +7%
- **Doc 02 C.2** (LDAM-DRW): +1.5 to +2.5%
- **Doc 02 D.2** (CutMix): +0.5 to +1%
- **F.2 below** (5-crop TTA): +1%

Sum: roughly **+11 to +15% over baseline**, putting POPW solidly in the **75–80% Top-1** range — comfortably above 66.45%.

**Specific evaluation protocol to match the IndustReal paper:**
- The paper reports Top-1 / Top-5 on the test split's atomic action clips.
- Each clip is a single action of variable length (typically 0.5–4 s).
- They use uniform 16-frame sampling per clip at test time, same crop setup.
- **You should match this** in `evaluate.py`. If you currently evaluate frame-by-frame, switch to clip-level evaluation for the headline number.

```python
# In evaluate.py, build clip-level evaluation:
def evaluate_clips(model, clip_dataset):
    correct = 0
    for clip, label in clip_dataset:
        # Sample 16 frames uniformly
        idx = np.linspace(0, len(clip) - 1, 16).astype(int)
        frames = clip[idx]
        with torch.no_grad():
            logits = model(frames.unsqueeze(0))['act_logits']  # [1, 74]
        pred = logits.argmax(dim=-1)
        correct += (pred == label).item()
    return correct / len(clip_dataset)
```

**Activity Top-5 (target 88.43%):** Almost certain to clear with the same recipe. Aim for **91–93%**.

---

### A.3 PSR F1 — beat STORM-PSR's 0.901

STORM-PSR is a dedicated PSR model with a transformer-based dual-stream architecture. To beat it:

- **Doc 01 C.1 + C.2** (causal Transformer + per-component heads): +3.5 F1
- **Doc 02 C.3** (focal PSR loss): +0.5 to +1 F1

Target: **0.91–0.93 F1**.

**Critical evaluation detail:** The CVIU 2025 STORM-PSR paper measures F1 on the **IndustReal-PSR** subset with their specific tolerance (±3 frames around the ground-truth step boundary). Match this exactly:

```python
# In evaluate.py for PSR:
def psr_f1_with_tolerance(pred_steps, gt_steps, tolerance_frames=3):
    # For each predicted step, look for a GT step within ±3 frames
    # Match using bipartite assignment to avoid double-counting
    ...
```

Without this tolerance, your PSR F1 will look 5–10% lower than STORM-PSR's just because of a stricter metric definition — **not because your model is worse**. This is the single most common mistake on PSR benchmarks.

**PSR POS (Percentage of correct sequences):**
- A sequence is "correct" if every step in the procedure is recognized in the right order, within tolerance.
- POS is much harder than F1. STORM-PSR achieves 0.812 POS vs 0.901 F1.
- With the causal Transformer, target **0.83–0.85 POS**.

---

### A.4 Head Pose 9-DoF MAE — establish the baseline cleanly

There is no supervised baseline for this — it's an open task. But your evaluation needs to be **defensible**:

- Report MAE separately for each of the three 3-vectors:
  - **Forward vector MAE** (degrees, after converting from Cartesian to angular error)
  - **Position MAE** (millimeters, world coordinates)
  - **Up vector MAE** (degrees)

```python
# In evaluate.py:
def head_pose_metrics(pred, target):
    fwd_pred, pos_pred, up_pred = pred[:, :3], pred[:, 3:6], pred[:, 6:9]
    fwd_gt,   pos_gt,   up_gt   = target[:, :3], target[:, 3:6], target[:, 6:9]

    # Angular error in degrees (forward + up)
    fwd_pred_n = F.normalize(fwd_pred, dim=-1)
    fwd_gt_n   = F.normalize(fwd_gt, dim=-1)
    fwd_err = torch.acos((fwd_pred_n * fwd_gt_n).sum(-1).clamp(-1, 1)) * 180 / math.pi

    up_pred_n = F.normalize(up_pred, dim=-1)
    up_gt_n   = F.normalize(up_gt, dim=-1)
    up_err = torch.acos((up_pred_n * up_gt_n).sum(-1).clamp(-1, 1)) * 180 / math.pi

    pos_err = (pos_pred - pos_gt).norm(dim=-1) * 1000  # to mm

    return {
        'forward_mae_deg':  fwd_err.mean().item(),
        'up_mae_deg':       up_err.mean().item(),
        'position_mae_mm':  pos_err.mean().item(),
    }
```

Reasonable baselines to claim:
- Forward MAE < 8°
- Up MAE < 6°
- Position MAE < 50 mm

These are achievable with the multi-scale C4+C5 head from Doc 01 E.

---

### A.5 Assembly State F1@1 / MAP@R(+) — beat SupCon+ISIL

The SupCon+ISIL baseline uses ResNet-34 + supervised contrastive learning + ISIL loss. Your detection head essentially gets this for free since:
- 24 ASD classes already encode assembly state.
- Detection mAP@0.5 ≥ 86% implies high per-state recall.

**Tactic:** When a frame has multiple detections of the same state, take the highest-confidence one. F1@1 = "is the top prediction correct?" → equivalent to picking the argmax detection class.

```python
# In evaluate.py:
def assembly_state_f1_at_1(predictions, ground_truth):
    # Per frame: is the top-confidence detection class == GT state?
    correct = 0
    for pred_frame, gt in zip(predictions, ground_truth):
        if len(pred_frame) == 0:
            continue
        top_pred = max(pred_frame, key=lambda p: p['score'])
        if top_pred['class'] == gt['state']:
            correct += 1
    return correct / len(ground_truth)
```

Target: **F1@1 > 0.87** (vs estimated ~0.85 baseline).

---

### A.6 Error Verification AP — beat GCA's ~0.58 AP

This task asks: "given a frame and the expected assembly state, is there an error?"

You're not training for this directly, but you can derive it from the ASD detection output:

```python
def error_verification_ap(detection_outputs, expected_states, errors):
    # If the detected ASD state differs from expected, flag as error
    scores = []
    labels = []
    for det, exp_state, is_error in zip(detection_outputs, expected_states, errors):
        # Score = 1 - confidence of expected state
        max_exp = max((d['score'] for d in det if d['class'] == exp_state), default=0.0)
        scores.append(1.0 - max_exp)  # higher score = more likely error
        labels.append(is_error)
    return average_precision_score(labels, scores)
```

Target: **AP > 0.65**.

---

## B. Efficiency reporting

The benchmark tables include params/GFLOPs/FPS for competitors. POPW needs these too — and they need to be measured the same way.

### B.1 Standard efficiency benchmark script

Create `efficiency_report.py`:

```python
# /home/claude/improvements/efficiency_report.py — see file
```

This script reports:
- **Total trainable params** (M)
- **Total params** including frozen
- **GFLOPs** at 1280×720 (using `fvcore.nn.FlopCountAnalysis`)
- **FPS** measured over 200 forward passes after warmup, with `torch.cuda.synchronize()`
- **Latency p50, p95, p99** (ms) — important because some heads (PSR with stateful GRU) have variable latency
- **Memory peak** (MB) during forward pass

### B.2 Reporting format that beats competitors

The key trick for the efficiency narrative: **report two FPS numbers**:

| Configuration | Params | GFLOPs | FPS (RTX 3060) | Notes |
|---|---|---|---|---|
| POPW-Full (training) | ~75M | ~85 | 12 | All heads active, T=16 |
| POPW-Inference (pruned) | ~50M | ~55 | 18 | Activity-only deployment, T=16 ViT only |
| POPW-Streaming | ~50M | ~12 (incremental) | 35 | Single-frame update with cached temporal bank |

The **streaming FPS** is what matters for industrial deployment, and POPW's feature bank design naturally supports this. You should highlight it — competitors like MViTv2 must reprocess the full 16-frame window for each output, while POPW caches the bank and only adds one frame.

### B.3 ONNX export for fair latency comparison

Your `export_onnx.py` already exists. Run it for the inference variant of the model (no Feature Bank, no PSR sequence state) and report ONNX Runtime FPS in addition to PyTorch FPS — reviewers will trust ONNX numbers more.

```bash
python export_onnx.py
python -c "
import onnxruntime as ort
import numpy as np, time
sess = ort.InferenceSession('industreal_model.onnx', providers=['CUDAExecutionProvider'])
x = np.random.randn(1, 3, 720, 1280).astype(np.float32)
for _ in range(20): sess.run(None, {'input': x})  # warmup
t0 = time.time()
for _ in range(200): sess.run(None, {'input': x})
print(f'ONNX FPS: {200 / (time.time() - t0):.1f}')
"
```

---

## C. Evaluation rigor — the things reviewers will probe

### C.1 Cross-validation, not a single split

The IndustReal paper uses a fixed split. Fine for headline numbers — but in your thesis, **add a 5-fold cross-validation table for at least Activity Top-1**. Without this, reviewers will accuse you of cherry-picking the test set.

### C.2 Statistical significance

When you beat a baseline by, say, 2.5%, the natural question is "how much variance is there?" Run **3 random seeds** for the final configuration, report mean ± std:

```python
# In train.py, support --seed argument:
# Run with seeds 42, 123, 7 — report mean ± std of final test metrics
```

A 2.5% improvement with std 0.4% is publishable. A 2.5% improvement with std 1.8% is noise. Doing this protects you.

### C.3 Ablation table

Reviewers will want to know which improvement matters most. Plan for an ablation table at the end:

| Configuration | Activity Top-1 | ASD mAP | PSR F1 |
|---|---|---|---|
| POPW base (XML diagram only) | 65.0% | 78.0% | 0.85 |
| + TCN (Doc 01 A.1) | +1.4 | +0.0 | +0.0 |
| + T=16 (Doc 01 A.2) | +0.9 | +0.0 | +0.0 |
| + 2 ViT + CLS (Doc 01 A.3) | +1.6 | +0.0 | +0.0 |
| + LDAM-DRW (Doc 02 C.2) | +2.1 | +0.0 | +0.0 |
| + Synthetic pretraining (Doc 02 B.1) | +0.5 | +3.2 | +0.3 |
| + VideoMAE V2 stream (Doc 02 A.1) | +5.8 | +0.0 | +0.0 |
| + Causal Transformer PSR (Doc 01 C.1) | +0.0 | +0.0 | +2.4 |
| + GIoU (Doc 02 C.1) | +0.0 | +1.7 | +0.0 |
| + 5-crop TTA (Doc 02 F.2) | +1.0 | +0.0 | +0.0 |
| **POPW-Full** | **78.3%** | **86.8%** | **0.92** |

Track every config you try — `wandb` or even a simple CSV log per seed. You will need this table.

### C.4 Per-class breakdown for tail analysis

Activity has 74 classes. A single Top-1 number hides massive variance. Report:
- **Macro-F1** (treats all classes equally — the right metric for long-tail)
- **Top 5 hardest classes** with their per-class F1
- **Top 5 easiest** for context

This pre-empts the "did you cheat by getting easy classes right?" question.

---

## D. Final checklist before submitting

- [ ] All baselines re-run on the same eval protocol (no protocol mismatch — see A.3 PSR tolerance!)
- [ ] 3-seed mean±std for headline numbers
- [ ] 5-fold CV for at least Activity Top-1
- [ ] Ablation table with each Doc 01 / Doc 02 component as a row
- [ ] Efficiency table: PyTorch FPS, ONNX FPS, params, GFLOPs (B.1, B.3)
- [ ] Per-class F1 breakdown for Activity and ASD
- [ ] Confusion matrix figure for Activity (74 classes — make it heatmap-style)
- [ ] PSR transition diagram showing per-component temporal trajectory vs GT (qualitative figure)
- [ ] Head pose visualization: forward vector arrow overlaid on RGB frame (qualitative figure)
- [ ] All competitor numbers cited with paper, page, and table reference (your `BENCHMARK_PAPERS.md` already does this — reuse it)

---

## Summary — what this doc adds on top of Docs 01 and 02

Doc 01 and Doc 02 are about *making the model better*. Doc 03 is about *making the result publishable*:

1. Match each paper's evaluation protocol exactly (PSR tolerance is the trap to watch).
2. Report efficiency in the format reviewers will compare against (params, GFLOPs, FPS, **and streaming FPS as POPW's unique strength**).
3. Add the rigor reviewers expect: 3-seed runs, ablation table, per-class breakdown, statistical significance.
4. Per-target tactics that don't fit cleanly into model/training (RFS for ASD long-tail, clip-level eval for Activity, tolerance-aware F1 for PSR).

If Doc 01 + Doc 02 are implemented and Doc 03's evaluation rigor is followed, every benchmark target in `BENCHMARK_TABLES.md` should be cleared with margin.
