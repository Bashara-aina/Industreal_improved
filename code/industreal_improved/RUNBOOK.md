# MTL Training Runbook

**Last updated:** 2026-07-09
**Audience:** Anyone running / continuing the MTL training
**See also:** 187 (status), 192 (Opus Tier A plan), 193 (implementation status)

---

## TL;DR

**Current state:** Path-D MTL run is live on GPU 1 (PID 4140887) with all 19 Tier A fixes. Resumed from `best.pt` (ep5), running since 2026-07-10 05:24. Eval-every=10 (first eval at ep10, ~08:30 JST). Activity and PSR are at fresh-init ep0 (per Opus 192's prediction); the epoch 10 eval will be the first real metric signal. Previous run (PID 2925005, run9) crashed with cudaErrorLaunchTimeout at the eval boundary — fixed by increasing eval-every to 10.

**Subject overlap: PASS** (train 36 / val 16 / test 32, 0 overlap — Opus 186 H-2 non-negotiable verified).
**Checkpoint best.pt: VALID** (435/473 tensors loaded, all 4 heads produce valid output).

---

## 1. Training Lifecycle

### Current run: `run10` (PID 4140887)

- **Log:** `/tmp/mtl_mvit_run10.log`
- **Output dir:** `src/runs/rf_stages/checkpoints/mtl_mvit_run/`
- **Resumed from:** `best.pt` (epoch 5)
- **Per epoch:** ~35 min for 8000 batches
- **Eval:** every 10 epochs (ep10, 20, 30, ...)
- **Started:** 2026-07-10 05:24 JST
- **Previous run (run9, PID 2925005):** crashed with cudaErrorLaunchTimeout at eval boundary — fixed by eval-every=10

### What's running

| Fix | What | Status |
|-----|------|--------|
| Path D (181) | 8 fixes: D1, D1b, D1c, D2, D3, D4, §5.1, §5.2 | ✅ LIVE |
| Opus 186 (round 2) | B-6 PSR→P5, B-9 shape filter, B-10 optim skip, E-3 EMA model, E-6 grad-clip 5.0, E-7 batch cap 8000 | ✅ LIVE |
| Opus 192 (Tier A) | FC-2 det P2 skip, FC-4 PSR T=8 | ✅ LIVE |
| Other | Q3 2-layer MLP, auto-soup init, focal-BCE option | ✅ LIVE |

### What's NOT in this run

- ❌ Foundation backbone (inverts efficiency claim — ablation-only per Opus 192 Q4)
- ❌ Cross-task attention (high risk, redundant — 186 Q7)
- ❌ MMoE (marginal upside — 186 D-1)
- ❌ STORM-like decoder (real bug was FC-4; now fixed)
- ❌ ArcFace, temporal attn pool (per-frame tokens not exposed; 192 Q3)

---

## 2. Monitoring

```bash
# ASCII sparkline plot
python scripts/plot_training_curves.py --log /tmp/mtl_mvit_run9.log

# Anomaly detection (NaN, Kendall caps, loss trends)
python scripts/training_monitor.py --log /tmp/mtl_mvit_run9.log --once

# Continuous monitor (every 60s)
python scripts/training_monitor.py --log /tmp/mtl_mvit_run9.log
```

What to watch for:
- ✅ **activity loss** dropping from 4.8 toward 3.0 by ep5 (currently flat — fresh-init)
- ✅ **psr loss** dropping from 1.55 toward 1.0 by ep5 (currently flat — fresh-init)
- ✅ **det** alternating 0.001 / 2.5 per batch (expected; depends on GT presence)
- ✅ **pose** ~0.05 mostly (healthy)
- ⚠ **log_var_act** > 1.0 (caps active; the actual loss weight is `min(exp(-lv), 1.0)` = 0.37)
- ⚠ **log_var_psr** > 0.5 (caps active; actual weight = 0.61)
- ❌ **NaN/Inf** in any loss (training will fail)

---

## 3. The 14 Ops Scripts

| Script | Purpose | When to run |
|--------|---------|-------------|
| `mvp_smoke_suite.py` | 4-probe diagnostic (overfit-200, ST-act 5ep, PSR A/B, TAL vs 3×3) | After ep10 eval if 0.0 persists |
| `mvp_probe3_psr_ab.py` | PSR T=8 vs T=16 F1 comparison | Conditional on Probe 1 |
| `mvp_probe4_tal_vs_3x3.py` | TAL vs 3×3 on overfit-200 | Conditional on Probe 1 |
| `e8_gradient_diagnostic.py` | Per-task cosine heatmap (Figure 1) | After ep10 |
| `e8_gradient_diagnostic_lite.py` | Memory-efficient E8 | OOM-safe version |
| `train_st.py` | 4 single-task baselines (mandatory for paper) | When GPU-2 is free |
| `build_soup.py` | Average backbone weights from 4 ST runs | After ST completes |
| `verify_subject_split.py` | No recording_id overlap (Opus 186 H-2) | ✅ PASS, re-run if data changes |
| `verify_checkpoint.py` | Sanity check loaded checkpoint | Before resuming, after any change |
| `generate_paper_table.py` | Headline table from metrics.json | After all runs complete |
| `training_monitor.py` | Real-time anomaly detection | Continuous or `--once` |
| `plot_training_curves.py` | ASCII sparklines of per-task loss | After any training run |
| `integration_test.py` | 18-check verification of all 17 fixes | Before launch, after major changes |
| `compare_checkpoints.py` | Parameter diff between two checkpoints | Debug training, sanity-check |

---

## 4. The Mandatory Steps (Opus 192 §5)

### Step 1: Let Path-D reach ep30+ (~10 hours from launch)
- Watch eval at ep10 (first real metric signal)
- Watch eval at ep15, ep20, ep25, ep30

### Step 2: MVP smoke suite (1.5 days on GPU-2)
```bash
# Probe 1: Eval-harness sanity via overfit-200 (THE most important probe)
python scripts/mvp_smoke_suite.py --probe 1 --head det --n-steps 500

# Probe 2: ST-activity 5 epochs
python scripts/mvp_smoke_suite.py --probe 2 --head act

# Probe 3: PSR temporal A/B (after Probe 1 passes)
python scripts/mvp_probe3_psr_ab.py --n-steps 300

# Probe 4: TAL vs 3×3 (after Probe 1 passes)
python scripts/mvp_probe4_tal_vs_3x3.py --n-steps 200
```

**Decision rule (Opus 192 §6):** Commit expensive work only to a head whose MVP probe shows the cheap fixes are exhausted. Most likely ≥2 of the four "0.0" / "0.008" numbers dissolve here.

### Step 3: 4 single-task baselines (5-6 GPU-days, mandatory for paper)
```bash
python scripts/train_st.py --task det --epochs 30 --output_dir runs/st_det
python scripts/train_st.py --task act --epochs 30 --output_dir runs/st_act
python scripts/train_st.py --task psr --epochs 30 --output_dir runs/st_psr
python scripts/train_st.py --task pose --epochs 20 --output_dir runs/st_pose
```

### Step 4: Model soup (5 minutes)
```bash
python scripts/build_soup.py \
    --det runs/st_det/best.pt \
    --act runs/st_act/best.pt \
    --psr runs/st_psr/best.pt \
    --pose runs/st_pose/best.pt \
    --output runs/mtl_mvit_run/soup_backbone.pt
```

### Step 5: 1 MTL finetune from soup
```bash
# Re-launch training; auto-soup will pick up soup_backbone.pt
# (soup is skipped if --resume is given, so use --resume None)
python scripts/train_mtl_mvit.py \
    --epochs 30 --lr-backbone 5e-5 --lr-head 5e-4 \
    --output_dir runs/mtl_mvit_run \
    # NO --resume
```

### Step 6: E8 gradient diagnostic (2 hours)
```bash
python scripts/e8_gradient_diagnostic_lite.py --max-batches 100 --output /tmp/e8.json
```

### Step 7: Generate paper table
```bash
python scripts/generate_paper_table.py \
    --mtl-runs runs/mtl_mvit_run \
    --st-runs runs/st_det runs/st_act runs/st_psr runs/st_pose \
    --output paper_table.md
```

### Step 8: Write paper (L2 + L3 + method)
See RUNBOOK §5 for the paper outline.

---

## 5. Paper Outline (L2 + L3 + method)

**Title:** *One Backbone, Four Tasks: Diagnosing and Fixing Uncertainty-Weighting Collapse in Multi-Task Assembly Understanding*

**Abstract:** (1 para) MTL on IndustReal with one shared MViTv2-S backbone across 4 tasks. Demonstrate that Kendall uncertainty weighting silently degenerates to inverse-loss scaling and starves the highest-loss task (activity). Characterize, fix (per-task log_var caps + EMA-normalized losses), and show the result is a positive-transfer win on pose + efficiency on all tasks.

**1. Introduction**
- 1.1 MTL for industrial assembly understanding
- 1.2 Hypothesis: one shared backbone gives positive transfer on ≥1 task
- 1.3 Contributions: (a) Kendall-collapse diagnosis, (b) per-task cap fix, (c) MTL/ST comparison on IndustReal, (d) E8 gradient heatmap

**2. Related Work**
- 2.1 Multi-task learning
- 2.2 Loss balancing (Kendall, GradNorm, PCGrad)
- 2.3 Industrial assembly understanding (Schoonbeek et al. WACV 2024)

**3. Method**
- 3.1 Shared MViTv2-S backbone
- 3.2 Per-task heads (lightweight: 2-layer MLP for activity, STORM-like decoder for PSR, decoupled head for det, 6D MLP for pose)
- 3.3 **The Kendall pathology** (with derivations from Opus 181 §0)
- 3.4 **The fix** (per-task log_var caps + EMA-normalized losses + per-cell DFL targets)
- 3.5 E8 gradient heatmap as evidence

**4. Experiments**
- 4.1 IndustReal dataset (subj overlap verified = 0)
- 4.2 Eval protocol (WACV-aligned: clip-level activity, dual-protocol det, no subject overlap)
- 4.3 MTL vs single-task comparison
- 4.4 Per-head ablations (log_var caps, EMA, per-cell DFL, PSR T=8 fix)

**5. Results**
- 5.1 Headline table (per-head MTL/ST ratios)
- 5.2 Pose: positive transfer (MTL < ST MAE)
- 5.3 Activity: bounded cost (MTL ≈ 0.9-0.95 ST)
- 5.4 PSR: bounded cost, honest miss
- 5.5 Detection: bounded cost

**6. Discussion**
- 6.1 When MTL helps vs when it doesn't
- 6.2 Kendall weighting pitfalls
- 6.3 Limitations (PSR is the pre-registered miss; 4-task on 78K frames is data-limited)

**7. Conclusion**

---

## 6. Quick Decision Tree

```
Q: What's wrong with my training?
├─ NaN in any loss → STOP, check grad-clip / log_var caps
├─ activity loss stuck > 4.0 after ep10 → heads may be under-trained, let it run to ep30
├─ activity loss at 0.008 forever → check if --act-grouping=none is set, re-run smoke probe
├─ psr loss stuck > 1.5 after ep10 → check if backbone P5 features are loading, run smoke probe 3
├─ det mAP = 0 → run Probe 1 (overfit-200) to verify eval-harness
└─ pose MAE > 15° → check if backbone is frozen / broken

Q: Should I commit to expensive work?
├─ Probe 1 (overfit-200) fails → eval bug, fix before any architecture work
├─ Probe 1 passes → bottlenecks are real, consider the head upgrade (YOLOv8 head, STORM decoder, etc.)
└─ MVP suite says "fine" → continue Path-D, just wait for ep30+

Q: Should I add a foundation backbone?
├─ NO for the headline (inverts efficiency claim per Opus 192 Q4)
├─ YES for a one-head headroom ablation (InternVideo2-L or DINOv2-L)
└─ ONLY if license clears (InternVideo2 weights ≠ clean Apache; DINOv2-L is Apache)
```

---

## 7. Resume / Restart

```bash
# Kill current training
ps aux | grep train_mtl_mvit | grep -v grep | awk '{print $2}' | xargs kill

# Restart from best.pt with all fixes
cd /media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved
nohup python3 scripts/train_mtl_mvit.py \
    --epochs 100 --batch-size 2 --grad-accum-steps 2 --num-workers 0 \
    --lr-backbone 1e-4 --lr-head 1e-3 --pcgrad --hp-prec-cap \
    --eval-every 5 --max-batches-per-epoch 8000 --grad-clip-norm 5.0 \
    --resume /path/to/best.pt \
    --output-dir /path/to/output > /tmp/mtl_mvit_run.log 2>&1 &
```

The resume path handles head reshapes (PSR T=8, 2-layer activity MLP) automatically via the shape filter (B-9).

---

## 8. Diagnostic Flowchart

```
Training at ep5+
  │
  ├─ Eval shows mAP > 0.0 → great, let it run
  │    │
  │    └─ Eval shows activity top-1 < 0.10 still → fresh-init, wait
  │
  ├─ Eval still shows mAP = 0.0 → run Probe 1 (overfit-200)
  │    │
  │    ├─ Probe 1 fails (eval metric stuck) → eval-harness bug, fix
  │    └─ Probe 1 passes → eval is fine, check features
  │         │
  │         ├─ Backbone frozen? → unfreeze
  │         └─ P5 features loading? → check feature_pyramid
  │
  └─ Eval shows NaN/Inf → STOP, check log_var caps
       │
       └─ log_var_act > 1.5 → cap is too high, reduce to 1.0
```

---

## 9. Paper-Write Checklist

Before submitting the paper, verify:

- [ ] Subject overlap check PASS (`scripts/verify_subject_split.py`)
- [ ] Checkpoint produces valid output (`scripts/verify_checkpoint.py`)
- [ ] Per-head metrics recorded (eval-every 5)
- [ ] 4 single-task baselines trained (per Opus 192 §5 step 7)
- [ ] Model soup done if useful
- [ ] MTL/ST ratio per head > 0.9 OR honest L2+L3+method story
- [ ] E8 gradient heatmap generated (Figure 1)
- [ ] Log_var caps verified to be active (act ≤ 1.0, psr ≤ 0.5)
- [ ] Eval protocol matches WACV (clip-level activity, dual-protocol det, no subj overlap)
- [ ] Paper story: L2 + L3 + method, NOT "we beat SOTA on all 4 heads"
- [ ] PSR pre-registered as the honest miss
- [ ] Pose positive transfer as the headline win

---

## 10. Quick-Reference Commit Map

| Commit | What |
|--------|------|
| `f2b01cc4a` | Tier A: PSR T=8, det P2, MVP suite, ST scripts, soup |
| `7c0456ff7` | Per-component PSR loss: downsample labels |
| `62f90110a` | E8 diagnostic, TAL (TOOD), auto-soup init |
| `ca043b47f` | File 193: Tier A status doc |
| `7c0456ff7` | (dup) per-component PSR fix |
| `c6d2f5259` | Probe 3, Probe 4, subject-overlap, integration, focal-BCE |
| `28e99f938` | Subject overlap (fast) — PASS |
| `7f6ca8e8a` | Paper table gen, training monitor, divide-by-zero fix |
| `7bfbb91ae` | Training-curve plotter |
| `43e871326` | E8 lite (memory-efficient) |
| `cd83d52b6` | verify_checkpoint |
| `6555e5dde` | compare_checkpoints |

Plus all prior Path-D + Opus 186 commits.

---

*This runbook is the operational counterpart to file 192 (Opus Tier A plan). When in doubt, run the integration_test first to confirm all 17 fixes are still active.*
