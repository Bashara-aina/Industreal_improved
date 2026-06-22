# GUIDE 5 — THE RUNBOOK (exact commands, what to do now)

*The operational guide. Everything here is a command or a decision rule. No analysis.*

> Strategy: GUIDE_2. Metrics: GUIDE_3. Paper: GUIDE_4. All commands run from
> `code/industreal_improved/` on **your** machine (GPU + dataset). I could not run them in
> the cloud container, but every code change referenced is applied and syntax-checked.

---

## 1. Definition of Done (your finish line — print this, tape it up)

You are DONE when, on the **full test split**:
- [ ] Detection: `det_mAP50_pc` reported (target 0.33–0.45) + confusion matrix
- [ ] Activity: clip Top-1/Top-5 reported (target 0.35–0.45 / 0.70+)
- [ ] PSR: F1(±3)/POS reported (target 0.50–0.62 / 0.75+)
- [ ] Head pose: MAE reported (have ~9° — done)
- [ ] Body pose: PCK@0.2 reported
- [ ] Efficiency table (params/FLOPs/FPS)
- [ ] Ablation A (single-task vs multi-task) + Ablation B (FiLM)
- [ ] Every `\todo`/`\popwres` in `popw_paper_improved.tex` filled

**Not on the list:** mAP ≥ 0.40, beating YOLOv8m, an OHEM ablation, a 49th consult doc.

---

## 2. First three actions (today, in order)

```bash
cd code/industreal_improved

# (1) See your HONEST detection number from data you ALREADY have. ~1 min, no GPU.
python src/diag_per_class_truth.py        # prints per-class AP/GT + det_mAP50_pc

# (2) Pull my code changes (honest metric + honest gates) — confirm they're present:
grep -n "HONEST METRIC" src/training/train.py
grep -n "det_mAP50_pc" src/training/stage_manager.py | head

# (3) Decide the path (see §3). Recommended: the decoupled A/B/C plan below.
```

Action (1) alone will likely show you're at ~0.31–0.35 present-class — i.e. *fine*. Let that
sink in before doing anything else.

---

## 3. Pick ONE path (don't run all three)

| Path | Command surface | When to use |
|------|-----------------|-------------|
| **A/B/C Decoupled** *(recommended)* | `train.py --preset recovery_det_only` → `embedding_cache.py` | Robust, ends interference, frees VRAM for activity. Best for "all heads benchmarkable." |
| RF gauntlet (now honest) | `stage_manager.py --launch RF2 … RF10` | If you prefer your existing orchestrator. Gates are now achievable. |
| Single joint run | `train.py --preset paper_run` | One command, all heads on. Simplest, but carries interference tax. Use as Phase-C polish. |

The rest of this runbook assumes **Path A/B/C** (GUIDE_2). It is the surest route to your goal.

---

## 4. The commands, phase by phase

### Phase A — strong backbone via detection
```bash
# Single-task-ish detector bootstrap (det + head_pose; activity/PSR off).
python src/training/train.py --preset recovery_det_only
#   (add your usual resume/output flags — same ones stage_manager passes; see
#    launch_training() in stage_manager.py around line 1647 for the full arg list.)

# OPTIONAL but high-impact IF you have the synthetic data: pretrain detection first.
python src/training/pretrain_synthetic.py --epochs 20
#   then point Phase A's resume at the pretrained checkpoint.

# Watch (GUIDE_2 §2): det_mAP50_pc rising; POS_ANCHOR_PROBE 400–800; per-class easy→hard.
# Exit when det_mAP50_pc plateaus (3 epochs < +0.005). That value IS your detection number.
```

### Phase B — freeze, cache, train temporal heads
First apply the 3 small fixes to `embedding_cache.py` (§6 below — 10 min). Then:
```bash
# Cache frozen-backbone features over the dataset (do train AND val/test splits).
python src/training/embedding_cache.py --cache \
    --ckpt runs/.../best.pth --cache-dir runs/cache_train --split train
python src/training/embedding_cache.py --cache \
    --ckpt runs/.../best.pth --cache-dir runs/cache_test  --split test

# Train activity + PSR heads on the cache (fast: hundreds of epochs/hour).
python src/training/embedding_cache.py --train \
    --ckpt runs/.../best.pth --cache-dir runs/cache_train \
    --output-dir runs/cache_heads --seq-len 64 --epochs 50 --batch-size 128
```
With the backbone frozen you now have VRAM headroom — this is where you can afford
`USE_VIDEOMAE=True` / a deeper temporal model for the activity head (GUIDE_2 §3.1).

Head pose & body pose: already trained in Phase A (they're spatial, per-frame). Just
evaluate them.

### Phase C — optional joint fine-tune (for the "jointly optimized" claim)
```bash
# Load Phase-A backbone + Phase-B heads, short low-LR joint pass.
python src/training/train.py --preset paper_run    # 3–5 epochs, LR 1e-5
# If it destabilizes (interference returns), STOP and keep the decoupled result. It's enough.
```

### Final evaluation (the paper numbers) — FULL TEST SPLIT
```bash
python src/evaluation/evaluate.py --split test --ckpt runs/.../best.pth   # adapt to your eval CLI
python src/diag_per_class_truth.py                                        # detection per-class + confusion
python scripts/training/efficiency_report.py                             # params/FLOPs/FPS
python scripts/training/generate_paper_tables.py                         # → .tex tables
```

---

## 5. Dead-head triage (only if a head is truly DEAD, not just climbing)

| Head DEAD | First check | Fix |
|-----------|-------------|-----|
| Detection | `POS_ANCHOR_PROBE` n_pos > 0? `LIVENESS det` > 0? | if both yes, it's *converging*, not dead — wait. If n_pos=0, check GT-frame sampling. |
| Activity | `pred_seen` stuck at 1–4 classes | lower imbalance pressure: class-balanced sampling, keep `USE_LDAM_DRW=False`, `CB_GAMMA=1.0` |
| PSR | constant output (unique patterns ≤ 1) | confirm `USE_PSR_TRANSITION=True` + sequence mode; per-frame focal on static labels is the trap |
| Head/body pose | grad ≈ 0 | check soft-argmax temp (train τ=1.0, eval τ=0.1) and loss weight |

**A head that is alive and rising is not a problem. Do not "fix" it.**

---

## 6. The `embedding_cache.py` rough edges to fix before Phase B (~10 min)

Your Tier-2.4 code was written but never run. Three spots to fix:
1. **Line ~489** `if max_batches and (batch_idx := 1) > max_batches:` — stray/incorrect.
   Replace with a real counter (`for bi, batch in enumerate(loader): … if max_batches and bi >= max_batches: break`).
2. **`CacheDataset` split (line ~199–204)** uses "first 80% of recordings = train." Replace
   with your **official** train/val/test split so numbers are comparable to baselines.
3. **Output-key names (line ~472–475)**: confirm `activity_proj`/`proj_feat`, `det_conf`,
   `c5_mod`, `pyramid['p4']` match your current `model.py` forward output dict. Print
   `outputs.keys()` once and adjust.

(These are the only blockers; the HDF5 cache + trainer logic is sound.)

---

## 7. Decision rules (if X then Y — no deliberation)

- **If `diag_per_class_truth.py` shows `det_mAP50_pc` ≥ 0.30** → detection is good enough;
  do NOT optimize it further; move to Phase B.
- **If a Phase-B head hits its GUIDE_3 target** → freeze it, stop, next head.
- **If Phase C destabilizes** → drop it; ship the decoupled model.
- **If you catch yourself writing a `.md` analysis** → stop; make it an experiment or skip it.
- **If you want to tune OHEM / gamma / bias-LR / anchors** → don't; GUIDE_1 §1.2 settled it.
- **If detection won't pass even the honest 0.22 gate after Phase A** → it's data/label,
  not training: run the confusion matrix, accept rare-class zeros, report honestly.

---

## 8. What to STOP doing (explicit)

- ❌ Writing consult/analysis/status `.md` files (you have 48; they contradict each other).
- ❌ Run1/Run2 log forensics on the 313k-line train.log.
- ❌ Tuning OHEM, `gamma_neg`, `DET_BIAS_LR_FACTOR`, `DET_LR_MULTIPLIER`, anchors.
- ❌ Reporting/gating on diluted `det_mAP50`.
- ❌ The RF1→RF10 gauntlet *if* it's slowing you — prefer A/B/C.
- ❌ Comparing yourself to YOLOv8m's 0.838 as a pass/fail bar.
- ❌ Waiting for "one more diagnostic" before training/writing.

## 9. What to START doing

- ✅ Judge per-task on honest metrics (GUIDE_3).
- ✅ Run the decoupled A/B/C plan.
- ✅ Evaluate on the full test split, clip-level activity, ±3-frame PSR.
- ✅ Fill the paper tables as numbers arrive (GUIDE_4 §5) — don't wait for all of them.
- ✅ Write the limitations section now (GUIDE_4 §4); it makes weak numbers defensible.

---

## 10. Suggested cadence (so this actually ends)

| Block | Goal | Output |
|-------|------|--------|
| Day 1 | Honest baseline + decide path | `diag_per_class_truth` number; A/B/C chosen |
| Days 2–4 | Phase A backbone to plateau | `best.pth`, detection number locked |
| Day 5 | Fix + run Phase B cache | embeddings cached |
| Days 6–7 | Phase B heads (activity, PSR) | all heads trained |
| Day 8 | Full-test eval + tables | every results row filled |
| Day 9 | Ablations A + B | the idea proven |
| Day 10 | Paper: fill `.tex`, limitations, figures | **draft complete** |

Two weeks of *execution* finishes what months of *analysis* could not. The model is ready.
You are ready. Stop consulting; start finishing.

---

### The code I changed for you (recap)
- `src/training/train.py` — `best.pth`/combined metric on `det_mAP50_pc`; `Val:` line logs
  `det_mAP50_pc` + `det_n_present`. *(honest checkpoint selection + reporting)*
- `src/training/stage_manager.py` — all 10 gates on `det_mAP50_pc` at achievable thresholds.
  *(curriculum can advance → all heads get trained)*

Both files `python -m py_compile` clean. Pull, run, finish.
