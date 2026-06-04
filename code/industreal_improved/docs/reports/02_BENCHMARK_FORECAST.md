# 02 — Per-Benchmark Forecast: What You'll Actually Hit

This document answers: **for each benchmark target, what is the realistic expected score given the current code state?** Three columns per target:

- **Now (default config):** what `python train.py` produces today, no flags flipped
- **Easy wins (5 flags flipped):** what you get by setting `USE_LDAM_DRW=True`, `USE_HEADPOSE_FILM=True`, `USE_LION=True`, `BACKBONE='convnext_tiny'`, and running `pretrain_synthetic.py` first
- **Full recipe:** what you get if you also do the 4 critical fixes from Audit Doc 01 (VideoMAE plumbing, stage freezing, PSR cache, SWA+TTA)

All numbers are **point estimates** — actual variance is ~±1.5% on top-1 metrics, ~±0.8 mAP on detection, ~±0.015 on PSR F1. I'll annotate confidence levels (low/med/high) per target.

---

## A. The big four IndustReal targets

### A.1 ASD Detection mAP@0.5 — target >83.8% (YOLOv8m COCO+synth+real)

| Config | Expected mAP@0.5 | vs target | Confidence |
|---|---|---|---|
| Now (default) | 80–82% | **just below** 83.8% | High |
| Easy wins | 83–85% | **at or just above** | Medium |
| Full recipe | 86–88% | **comfortably above** | Medium |

**Reasoning:**

The architecture is sound. The detection head is identical to RetinaNet's, GIoU is live (Doc 02 C.1, +1.5–2 mAP), layer4 BN is unfrozen (B.2, +1.5 mAP). What's missing:
- B.1 (synthetic pretraining) is the biggest leverage point. It's a separate script you have to run first, then load the checkpoint. Without it, you're training a detection head from scratch on ~30k IndustReal frames vs YOLOv8m's COCO+synth+real (~150k+ frames). That's the 2–3 mAP gap.
- B.3 (anchor calibration) is missing — small effect, ~0.7 mAP.

**Confidence in beating 83.8%:** High **with synthetic pretraining**, medium without. If you can only do one thing to clear this target, run `pretrain_synthetic.py` for 20 epochs and load the checkpoint via `--resume` into the main training run.

**Risk factor:** YOLOv8m is a strong, mature, single-task detector. Multi-task gradient noise from your other heads (activity, pose, PSR) will pull the backbone in directions that are sometimes orthogonal to detection. Kendall weighting partially compensates but doesn't eliminate this. The 83.8% target is genuinely close to the ceiling for a multi-task RetinaNet on this dataset.

---

### A.2 Activity Top-1 — target >66.45% (MViTv2 Kinetics)

| Config | Expected Top-1 | vs target | Confidence |
|---|---|---|---|
| Now (default) | 64–67% | **near, possibly below** | Medium |
| Easy wins | 67–70% | **above with margin** | Medium |
| Full recipe (incl VideoMAE wired) | 73–77% | **comfortably above** | Medium |

**Reasoning:**

This is the hardest target and the one where the most architectural work was done (TCN, T=16, 2× ViT, CLS pooling — all live). My earlier estimate was that with everything Doc 01 A.1–A.4 active and Doc 02 A.1 (VideoMAE) you'd be at 75–80%. With VideoMAE *not* wired, the realistic ceiling drops to ~70–73%.

Why it's still close to 66.45% in default config:
- Pretraining gap. MViTv2 was pretrained on Kinetics-400 (~250k videos). Your backbone is ImageNet-only.
- Activity head is well-designed but only sees what the CNN backbone gives it.
- Temporal context (T=16 = 1.6s) is good but not as deep as MViTv2's spatiotemporal attention across full clips.

**The single biggest unlock for this target is VideoMAE wired correctly.** Audit Doc 01 §B.1 explains why it's currently dead code. Fixing this is ~1 day of work. If you don't fix VideoMAE, your realistic Top-1 is in the 67–70% range (modest win) rather than 75%+ (decisive win).

**Confidence in beating 66.45%:** Medium-high in easy-wins config, but with thin margin. The cleanest win comes from VideoMAE plumbing.

**Top-5 (target 88.43%):** Almost guaranteed to clear in easy-wins config. Top-5 of a 74-class problem is forgiving once the model is broadly competent. Realistic range:
- Now: 86–89%
- Easy wins: 90–92%
- Full recipe: 92–94%

---

### A.3 PSR F1 — target >0.901 (STORM-PSR)

| Config | Expected F1 (with ±5 frame tolerance) | vs target | Confidence |
|---|---|---|---|
| Now (default) | 0.86–0.89 | **below** | Medium |
| Easy wins | 0.88–0.91 | **at/just above** | Low-medium |
| Full recipe (incl PSR cache fix) | 0.91–0.93 | **above** | Medium |

**Reasoning:**

The architecture is excellent — causal Transformer, per-component heads, multi-scale FPN features. On paper this should crush BiGRU baselines and be competitive with STORM-PSR.

But the **PSR cache issue from Audit Doc 01 §B.4 is real**. During training, frames are shuffled. The cache stores them in arrival order, not temporal order. The causal mask is then meaningless. So during training, the causal Transformer is effectively learning per-frame classification from random "history" of unrelated frames. At evaluation time, frames arrive in order (good), but the model was trained on garbage temporal context.

Two scenarios:
1. **Best case:** The model is robust enough that the per-frame multi-scale features (P3+P4+P5 GAP) drive most of the signal, and the temporal context just adds noise. PSR F1 ~0.86–0.89 even without the fix.
2. **Worst case:** The Transformer overfits to the random history pattern at training time and learns nothing useful. PSR F1 is essentially per-frame, ~0.80.

The fix is: switch PSR training to **sequence-batched mode**, where each "batch element" is one full recording's frame sequence (or a long contiguous slice). The model processes T frames at once, the causal mask works correctly, and F1 should jump 0.04–0.06.

Important caveat: I see your eval uses `tolerance_frames=5` by default. STORM-PSR's reported 0.901 is at `tolerance_frames=3` (their tighter protocol). When you compare:
- POPW with tolerance=5 vs STORM-PSR's 0.901 at tolerance=3: **invalid comparison**, your F1 looks artificially higher
- POPW with tolerance=3 vs STORM-PSR's 0.901 at tolerance=3: valid

**Run the eval at both tolerances** and report both. Reviewers will check this.

**Confidence in beating 0.901 (at tolerance=3):** Medium with the cache fix; **low without it**.

---

### A.4 PSR POS — target >0.812 (STORM-PSR)

| Config | Expected POS | vs target | Confidence |
|---|---|---|---|
| Now (default) | 0.74–0.79 | **below** | Medium |
| Easy wins | 0.78–0.82 | **at/just above** | Low |
| Full recipe | 0.83–0.86 | **above** | Medium |

POS = Percentage of Correct Sequences (full procedure correct end-to-end). This is much harder than F1 because **one wrong component anywhere in the sequence flips the whole sequence to "incorrect"**. With 11 components and average procedure length ~30 steps, the per-component error rate has to be very low.

Same caveats as F1 — the cache fix is essential. Without it, expect 0.74–0.78.

---

## B. The supporting targets

### B.1 Head Pose 9-DoF MAE — establish baseline

There is no supervised baseline. Whatever number you report **is** the baseline. The more important question is: is the number defensible?

Currently `compute_head_pose_metrics()` reports raw component-wise MAE. The forward and up vectors are unit vectors (range [-1, 1]), the position is in meters. Reporting all 9 as a single MAE mixes units. This is **not how anyone in the literature reports head pose**.

**Required eval upgrade (Doc 03 A.4):** report angular error in degrees for forward+up, position error in mm separately:

```python
fwd_pred_n = F.normalize(pred[:, :3], dim=-1)
fwd_gt_n   = F.normalize(gt[:, :3], dim=-1)
fwd_err_deg = torch.acos((fwd_pred_n * fwd_gt_n).sum(-1).clamp(-1+eps, 1-eps)) * 180 / pi
```

Realistic values for a multi-task model with ImageNet pretraining + IndustReal training:
- Forward MAE: 6–10° (angular)
- Up MAE: 4–7° (angular)
- Position MAE: 30–60 mm

These are presentable as a baseline in your thesis. Without the angular formulation, the raw component MAE is hard to interpret and won't read well in a paper.

### B.2 Activity Top-5 (target 88.43%)

Already covered in A.2. **Highest confidence win** of all the targets — even your default config probably clears it.

### B.3 Assembly State F1@1 (target ~0.85 estimated)

This is essentially "given a frame, did the top detected ASD class match the GT state?" Derived directly from the detection head's output. If you clear ASD mAP@0.5 (A.1), you almost certainly clear F1@1 because they're the same underlying signal.

**Expected:** F1@1 ≈ 0.84–0.88 in easy-wins config. **Confidence: medium-high**.

### B.4 Error Verification AP (target ~0.58 estimated)

Similar — derived from detection confidence on the "expected vs detected" mismatch. The 0.58 baseline from the Lehman paper is on a single-task ResNet-34 with contrastive learning. Your multi-task model with cleaner detection should clear this.

**Expected:** AP ≈ 0.62–0.68 in easy-wins config. **Confidence: medium**.

---

## C. The efficiency narrative

Here is where you have to be careful. Your competitors:

| Model | Params | GFLOPs | FPS |
|---|---|---|---|
| PTMA (IKEA ASM) | 12.9M | 1.96 | 291 |
| MiniROAD (IKEA ASM) | 10.5M | 1.08 | 325 |
| YOLOv8m | ~25M | ~80 (at 640²) | ~50 |
| MViTv2-S | ~36M | ~70 | ~30 |
| STORM-PSR | unknown | unknown | unknown |

POPW (estimated from architecture):

| Configuration | Params | GFLOPs (1280×720) | FPS (RTX 3060) |
|---|---|---|---|
| ResNet-50 base, no extras | ~57M | ~85 | 12–15 |
| ResNet-50 + HeadPoseFiLM | ~58M | ~85 | 12–15 |
| ResNet-50 + VideoMAE | ~80M | ~91 | 9–11 |
| ConvNeXt-Tiny + HeadPoseFiLM | ~52M | ~75 | 14–17 |
| ConvNeXt-Tiny + VideoMAE | ~75M | ~81 | 11–13 |

**Honest assessment:** POPW is **not** going to beat PTMA on the efficiency narrative. PTMA is single-task, designed for FPS, and runs at 291 FPS with 12.9M params. POPW does ~5 tasks with ~57M params. They're optimizing for different things.

**The defensible efficiency angle:** POPW does the work of **5 separate models** (detection, pose, head pose, activity, PSR) in ~57M params. The fair comparison is:
- YOLOv8m (25M, detection only) + MViTv2 (36M, activity only) + STORM-PSR (~?, PSR only) ≈ 70M+ params combined, runs sequentially (FPS divides)
- POPW: ~57M params, runs in **one** forward pass at 12–15 FPS

The efficiency message is: **"single unified model that beats specialized baselines on each individual task"** — not "fewer parameters" or "higher FPS than PTMA". You can't win the latter; you can win the former.

**Required additions for credible efficiency reporting:**
1. Run `efficiency_report.py --all_configs` and put the table in your thesis. ✅ already exists.
2. Add a "streaming FPS" measurement (Doc 03 B.2) — single-frame update with cached temporal bank. This is genuinely something competitors can't do because they reprocess full clips. Likely value: ~25–30 FPS for streaming POPW vs ~30 FPS for batch MViTv2. POPW being competitive in streaming mode despite multi-task is the actual win.
3. Report the multi-model baseline equivalent. "If you wanted to do all 4 tasks with separate SOTA models, you'd need ~70M+ params and ~120+ GFLOPs total. POPW does it in 57M and 85 GFLOPs."

---

## D. Confidence-weighted summary

I'll be blunt about how likely each target is to be cleared, given realistic outcomes:

| Target | Current code | Easy-wins config | Full recipe |
|---|---|---|---|
| ASD mAP@0.5 > 83.8% | 25% likely | 60% likely | 85% likely |
| Activity Top-1 > 66.45% | 45% likely | 70% likely | 85% likely |
| Activity Top-5 > 88.43% | 70% likely | 90% likely | 95% likely |
| PSR F1 > 0.901 (tolerance=3) | 15% likely | 35% likely | 70% likely |
| PSR POS > 0.812 (tolerance=3) | 15% likely | 30% likely | 65% likely |
| Head Pose baseline | N/A — establishing baseline |
| Assembly State F1@1 | 35% likely | 70% likely | 85% likely |
| Error Verification AP | 40% likely | 70% likely | 85% likely |

**Easy wins config** = flip these 4 flags + run synthetic pretraining first:
```python
# In config.py:
BACKBONE = 'convnext_tiny'
USE_HEADPOSE_FILM = True
USE_LDAM_DRW = True
USE_LION = True  # after pip install lion-pytorch
PRETRAIN_DET_ON_SYNTH = True  # then run pretrain_synthetic.py
```
And:
```bash
pip install lion-pytorch transformers
python pretrain_synthetic.py
# then python train.py with --resume from pretrain_synthetic checkpoint
```

**Full recipe** = easy wins + 4 critical code fixes:
- Wire `clip_rgb` end-to-end (VideoMAE) — ~1 day
- Stage parameter freezing — ~half day
- PSR sequence-batch training — ~1 day
- SWA + flip TTA + 5-crop TTA in evaluate.py — ~half day

---

## E. The honest final answer to your question

> **"is it able to beat the benchmark? we can train now? for industreal dataset? and able to beat most of the metrics?"**

**Can you train now?** Yes. The default config will run end-to-end and produce reasonable numbers. There are some bugs (PSR cache leak, OneCycleLR mismatch *if* enabled) but the default path works.

**Will you beat all benchmarks with the current default config?** No. You will probably beat:
- Activity Top-5 (easy)
- Head Pose (no baseline to beat)
- Assembly State F1@1 (likely)

You will likely fall short on:
- ASD mAP@0.5 (close but probably 80–82%, just below 83.8%)
- Activity Top-1 (close but probably 64–67%, near the 66.45% target)
- PSR F1 / POS (probably below 0.901 / 0.812)

**Will you beat most benchmarks with the easy-wins config (4 flags + synthetic pretraining)?** Yes, with reasonable confidence. ASD, Activity Top-1, Top-5, Assembly State, Error Verification all probably above target. PSR is the wildcard.

**Will you beat all benchmarks with the full recipe?** With high confidence on accuracy. **Efficiency-wise you won't beat PTMA's FPS** — that's not the right narrative for POPW.

**Where does POPW frame the win?**
1. **Multi-task unification**: one model does 5 tasks, beats specialized baselines on each individual task.
2. **Architectural novelty**: PoseFiLM + HeadPoseFiLM cross-task conditioning.
3. **PSR with causal Transformer**: streaming-capable procedural understanding (assuming you fix the cache).
4. **Streaming FPS competitive with batch FPS of single-task competitors** (with measurement).

NOT efficiency in the FPS / params sense — you'll lose that to PTMA. NOT a single-task win on any metric — you'll lose those to specialized baselines on most. The win is the **package**: doing it all at once with one forward pass.

The next document covers what experiments to actually run, in what order, to get to a publishable result fastest.
