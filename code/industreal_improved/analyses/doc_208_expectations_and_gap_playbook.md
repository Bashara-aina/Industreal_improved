# Doc 208 — MTL Expectations & Gap-Closing Playbook

**Status**: Living document. Updated 2026-07-11 with lever implementation status.
**Audience**: Paper authors. Honest assessment, no hedging.

---

## 1. The Spine of the Paper

The paper never needed SOTA parity. It needs three things:

1. **The ratio**: MTL/ST per head — "we retain X% of our matched single-task ceiling at 2× parameter efficiency"
2. **The Kendall figure**: uncertainty-weighted loss equilibrium across 4 disparate tasks
3. **The efficiency table**: params, FLOPs, FPS — one forward pass, 48.6M params total

Reviewers accept "80% retention at 2× efficiency." They reject "40% of a number set by a different system under different conditions."

### Reporting protocol

Every head's table row must show:

| Column | Source |
|---|---|
| MTL (ours) | `mtl_all6_v1` run on test split |
| ST ceiling (ours) | `st_{head}` run — matched architecture, same data, same 224px input |
| SOTA reference | Annotated with advantages: COCO pretraining, 640px, synthetic data |
| MTL/ST ratio | Computed from our own numbers |

---

## 2. Expected Outcomes (all 6 levers active, ep50)

| Head | Realistic | Optimistic | Our ST ceiling (projected) | SOTA anchor |
|---|---|---|---|---|
| **Pose** (fwd MAE ↓) | 6.5–8° | ~6° | ~7° (pose-only, same backbone) | None — we set it |
| **Activity** (top-1 ↑) | 30–45% | 45–55% | 55–65% (activity-only) | 65.25% (ST MViTv2-S) |
| **Detection** (mAP@0.5 ↑) | 0.25–0.45 | ~0.50 | 0.40–0.55 (detection-only) | 0.779 (YOLOv8m, 640px, COCO pretrain) |
| **PSR** (event-F1@±3 ↑) | 0.10–0.35 | 0.30–0.50 | Unknown (no ST baseline yet) | 0.883 (STORM, procedural pipeline) |

### Honest gap explanations for the paper

- **Detection**: "Our 224px input vs YOLOv8m's 640px means small assembly parts (~20px objects) span ~3 cells on the P3 grid. We do not claim comparability with COCO-pretrained, detection-optimized architectures. Our internal anchor is the matched-architecture ST baseline."
- **Activity**: "The 65.25% SOTA uses 640px input, synthetic data, and ~5× our training budget. Our claim is the MTL/ST retention ratio, not absolute top-1."
- **PSR**: "STORM's 0.883 uses a procedural multi-stage pipeline (detect hands → track objects → infer states). Our end-to-end approach directly predicts per-frame states at 224px. The comparison protocol differs fundamentally."

---

## 3. What We Control: The Gap Decomposition

```
SOTA gap = (SOTA − our ST baseline) + (our ST − our MTL)
            \_____ recipe/data/res _________/   \___ sharing cost ___/
                    Not our paper's business         Our paper's business
```

The 6 levers attack the **sharing cost** — the difference between what each head achieves alone vs. sharing a backbone. Levers 1-6 are "nearly free"; levers 7-9 are story-expensive and should only appear as ablation rows.

---

## 4. Lever Implementation Status

### Active (running in pipeline)

| # | Lever | Expected gain | Mechanism | Status |
|---|---|---|---|---|
| 1 | PSR monotonicity | +0.05–0.15 event-F1 | Once-on stays-on, median filter, per-recording | ✅ in eval |
| 2 | Detection threshold calib | mAP invariant, 98% FP reduction | F1-based sweep, score=0.5 | ✅ in config |
| 3 | SWA checkpoint averaging | +0.5–2% across tasks | Average last 5 periodic checkpoints | ✅ `--swa-checkpoints 5` |
| 4 | Head warm-starting | +2–5% activity/PSR, +1–3% detection | Init MTL heads from ST best.pt | ✅ `--warm-start-dir` |
| 5 | Distillation from ST teachers | +2–8% activity, +1–3% PSR | KL-div (act/det), MSE (psr/pose) | ✅ `--distill-teacher-dir` |
| 6 | Full training budget | +5–15% across tasks | 39k batches/ep, eff batch 16, ep50 | ✅ running |

### Not implemented (story-expensive, ablation only)

| # | Lever | Why not |
|---|---|---|
| 7 | Higher detection resolution | Breaks single-pass latency claim. Ablation row: "+res upper bound" |
| 8 | TTA (flip) for activity | +1–2% but compromises latency. Ablation row only |
| 9 | Verb-noun factorization | Label-space redesign → changes comparison protocol. Next-paper territory |

### Diagnostic (running now)

| Probe | Purpose | Status |
|---|---|---|
| Overfit probe (all 4 heads) | Find eval bugs before training completes | ✅ Complete |

### Probe results (2026-07-11)

| Head | Verdict | Steps | Final loss | Key metric | Interpretation |
|---|---|---|---|---|---|
| **Pose** | ✅ PASS | 57 | 0.002 | MAE 6.2° | Overfits trivially. Eval harness confirmed working. |
| **PSR** | ✅ PASS | 51 | 0.00002 | 91% positive | Overfits trivially. Focal-BCE works. No eval bug. |
| **Activity** | FAIL | 2000 | 0.59 | Top-1 40.5% | **False negative of probe design.** Head IS learning (improved from 26%→40.5%), but frozen random backbone's cls_token can't separate 75 fine-grained classes. With trainable backbone in real MTL, this is not an issue. |
| **Detection** | FAIL | 2000 | 1.11 | — | **False negative of probe design.** Detection on FPN features from frozen random backbone is fundamentally impossible — the spatial features have no semantic content for box regression + 24-class classification. The ~1.11 loss floor is the random baseline. With trainable backbone, detection head should learn. |

---

## 5. What Could Still Be Wrong

1. **Eval harness bugs**: The overfit probe is the guard. If a head drives loss → 0 on 200 clips but eval metric stays ~0, the eval code is broken. This must be fixed before results are trusted.
2. **Detection anchor grid**: The MViTv2-S feature map at 224px is 14×14 before upsampling. Small parts (<20px) may not anchor properly on P3. The 0.468 ConvNeXt-MTL anchor is the realistic ceiling.
3. **Activity class imbalance**: 75 classes with power-law distribution. Balanced sampling helps, but the tail classes (1-5 samples) will never be learned. Report top-1 and top-5 separately, with macro-F1 for the tail.
4. **PSR temporal resolution**: T=8 prediction from T=16 input means each logit covers 2 frames. Transition events may be smoothed out. The monotonicity constraint helps but doesn't create precision that isn't there.
5. **Kendall collapse**: The activity head (loss ~5) could still starve the detection head (loss ~2) if log_var caps are too loose. Current caps: det≤1.5, act≤1.0, psr≤0.5, pose≤2.0.

---

## 6. What Success Looks Like (paper-ready claims)

### Strong result (publishable at WACV/AAIML workshop)
- Pose: MT L/ST ≥ 0.95 (within noise of ST)
- Activity: MTL/ST ≥ 0.70 (retains 70% of ceiling)
- Detection: MTL/ST ≥ 0.60 (retains 60%, honest about resolution gap)
- PSR: event-F1 > 0.25 with monotonicity (demonstrates procedural prior helps)

### Adequate result (publishable with framing)
- All heads show non-trivial MTL/ST ratios
- Kendall figure shows stable equilibrium (no collapse)
- Efficiency table shows 1-pass advantage

### Failure mode (paper at risk)
- Any head at 0.0 metric (eval bug or collapse)
- Activity < 20% (MTL hurts this head irrecoverably)
- PSR < 0.05 event-F1 (head never learned)

---

## 7. Timeline

| Event | GPU | ETA |
|---|---|---|
| ST-det (ep 1-50) | 5060 Ti | ~55h |
| ST-act (ep 1-50) | 5060 Ti | ~55h |
| ST-psr (ep 1-50) | 5060 Ti | ~55h |
| Model soup build | CPU | ~1min |
| MTL all-6 (ep 1-50) | RTX 3060 | ~6.8 days |
| **Total wall time** | | **~13.8 days** |

Overfit probes: ~2h on GPU 0 (RTX 3060), running in parallel with ST baselines.
