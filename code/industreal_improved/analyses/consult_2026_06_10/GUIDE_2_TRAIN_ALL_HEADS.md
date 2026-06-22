# GUIDE 2 — TRAIN ALL HEADS (so the model learns correctly)

*The concrete plan to get all five heads to "alive and competitive" on a single 12 GB GPU,
without the gradient-interference war you've been fighting for months.*

> Prereq mindset: GUIDE_1. Numbers/targets: GUIDE_3. Commands: GUIDE_5.

---

## 1. The core decision: DECOUPLE training

You have been training 5 heads jointly from near-scratch on 12 GB. That forces them to
fight over backbone capacity and forces tiny batches, and it is the source of ~50 config
patches. The fix that ends the war:

```
PHASE A  Train a strong SHARED BACKBONE via detection (the densest spatial signal).
PHASE B  FREEZE the backbone. Cache its features. Train each temporal/pose head on the
         cache — independently, fast, with NO interference, and with the VRAM freed up.
PHASE C  (optional, for the "joint" claim) short joint fine-tune of everything together.
```

**Why this is correct, not a compromise:**
- At inference it is *still one model, one forward pass* — your paper's contribution is
  intact. The backbone is shared; heads read shared features. (Frozen-trunk multi-task is
  a standard, respected design — see GUIDE_4 §3 for the framing.)
- It **eliminates** the interference that every Kendall-cap / log-var-floor / grad-clip /
  detach flag was fighting. You can *delete* most of those band-aids.
- Freezing the backbone in Phase B **frees the VRAM** that forced `USE_VIDEOMAE=False` and
  batch_size=2. With cached 512-D features, you can train the activity head with a real
  temporal model (even the K400 video stream) at batch 128 and hundreds of epochs/hour.
- **The code already exists**: `src/training/embedding_cache.py` implements Phase B
  (`--cache` then `--train`). It's your own Tier-2.4 work, never run.

> **Alternative if you truly want a single joint run** (less robust, but one command):
> the `paper_run` preset already turns on all 5 heads with `staged_training=False`. It
> will work now that metrics/gates are honest — but it carries the interference tax.
> **Recommendation: decouple.** Use `paper_run` only as a final Phase-C polish.

---

## 2. PHASE A — a strong backbone via detection

**Goal:** backbone + FPN + detection head that reach `det_mAP50_pc` ≈ 0.33–0.45 and, more
importantly, produce *object-discriminative features* for every downstream head.

**Use the `recovery_det_only` preset** (det + head_pose only; activity/PSR off). It already
encodes the right choices: `detach_reg_fpn=False` (regression gradient shapes the FPN),
`reinit_pi` warm start, GT-frame oversampling at 0.9. Head-pose stays on because it's a
cheap, dense, stabilizing second signal for the trunk (your own RF1 notes are right about
this).

**What "learning correctly" looks like (watch these, not mAP-per-step):**
- `POS_ANCHOR_PROBE`: 400–800 positive anchors/image (you already confirmed this — anchor
  coverage is fine, do not touch anchors).
- Positive-anchor sigmoid scores (`POS_ANCHOR_PROBE` mean) rising past ~0.3 over epochs.
- `det_mAP50_pc` rising into the 0.30s. Ignore `det_mAP50` (diluted).
- Per-class: easy/frequent states (idx 7,10,12,17) climb first; rare states (idx 6,8,16)
  stay low — **expected**, not a bug.

**Two detection levers that actually matter (in priority order):**
1. **Synthetic pretraining** — *if you have the synthetic data*. `pretrain_synthetic.py`
   exists. This is the difference between ~0.35 and ~0.55 present-class. If you don't have
   the data, that's fine — it becomes a one-line limitation in the paper (GUIDE_4 §4).
2. **Class-balanced sampling for rare states** — oversample frames containing the
   <100-instance classes so they get gradient. This lifts the dead classes more than any
   loss-tuning. (Don't expect miracles on 26-instance classes; report them honestly.)

**Do NOT**: run the OHEM ablation, tune `gamma_neg`/`alpha`/`DET_BIAS_LR_FACTOR`, change
anchors, or reinit heads again. Those are the rabbit holes that cost you a month.

**Phase A exit:** `det_mAP50_pc` plateaus (3 epochs < +0.005). Save `best.pth`. Whatever
value it plateaus at *is your detection number* — accept it and move on.

---

## 3. PHASE B — freeze, cache, train the temporal/pose heads

This is where activity and PSR finally get a fair chance, because they stop competing with
detection for the backbone and stop being starved by tiny batches.

```
Step 1: cache features from the Phase-A checkpoint (frozen backbone over full dataset)
Step 2: train activity + PSR heads on the cache (fast, no interference)
```

Your `embedding_cache.py` does both (`--cache`, `--train`). **It was written but never
run, so it has 2–3 rough edges to fix first** (10-minute job, flagged in GUIDE_5 §6):
- line ~489: `if max_batches and (batch_idx := 1) > max_batches` is a stray bug — remove
  or replace with a real batch counter.
- `CacheDataset` splits train/val by "first 80% of recordings" — replace with your
  *official* train/val split so numbers are comparable to baselines.
- confirm `outputs` keys (`activity_proj`/`proj_feat`, `det_conf`, `c5_mod`, `pyramid.p4`)
  match your current `model.py` forward output names.

### 3.1 Activity head (the hardest "alive" head)
- **Eval clip-level, not per-frame.** Your goals doc already notes per-frame eval costs
  double-digit points. Aggregate the 16-frame clip to one prediction (GUIDE_3 §2.2).
- **Use the freed VRAM**: with a frozen backbone you can afford a proper temporal model.
  Two options, in order of payoff: (a) re-enable the K400 video stream
  (`USE_VIDEOMAE=True`) on cached/streamed clips; (b) a deeper TCN/Transformer over the
  cached 512-D features (your TMA cell + feature bank, now actually engaged because Phase B
  trains on long T=64 sequences).
- **Keep** CB-Focal + label smoothing (`CB_GAMMA=1.0`). **Keep `USE_LDAM_DRW=False`** for
  the first clean run (s=30 collapses). Add class-balanced sampling.
- **Learning correctly:** `pred_seen` grows past ~20/75 classes; Top-1 climbs past 0.10
  early, toward 0.35–0.45 clip-level. If it collapses to 1–4 classes, lower the effective
  class-imbalance pressure (sampling, not margin loss).

### 3.2 PSR head (your most beatable baseline)
- Train it as a **transition predictor**, not per-frame classifier. You already have
  `USE_PSR_TRANSITION=True`, `PSR_TRANSITION_SIGMA`, the `MonotonicDecoder`, and
  `USE_PSR_ORDER_PRIOR`. Per-frame focal on ~95%-static labels makes constant output
  near-optimal — that was a real trap; transition targets fix it.
- PSR is the natural **consumer of detection state**: feed the detection state-classifier
  output (`det_conf`, already cached) into the PSR temporal model. This is exactly the B2
  heuristic's strength (ASD-accumulation + order prior), but *learned* — a strict superset.
- **Learning correctly:** unique predicted patterns > 10 (not constant output); F1(±3)
  climbing past 0.25 toward 0.50–0.62.

### 3.3 Head pose & body pose
- **Head pose is already excellent (MAE ~9°).** Geo 6D rotation is on. Just *report it* —
  it's an uncontested table row. Don't touch it.
- **Body pose** is alive (grad ~0.94). Wing loss + soft-argmax with the train/eval
  temperature split you already fixed (train τ=1.0, eval τ=0.1). Report PCK/MAE.
- These two are spatial, per-frame heads; you can either keep them in Phase A (they share
  the backbone cleanly) or attach them in Phase B. Keeping head-pose in Phase A is what
  `recovery_det_only` already does.

**Phase B exit:** each temporal head hits its GUIDE_3 target or plateaus. Save head
weights. You now have all five heads trained.

---

## 4. PHASE C — optional joint fine-tune (for the "unified, jointly optimized" claim)

Load the Phase-A backbone + Phase-B heads, unfreeze, and run a **short** (3–5 epoch),
**low-LR** (1e-5) joint pass with Kendall weighting and `KENDALL_FIXED_WEIGHTS` as you
have it. Purpose:
- Lets you state "jointly fine-tuned" and report the **single-forward-pass** efficiency
  numbers honestly.
- Provides the data point for the cross-task ablation (GUIDE_4 §2): does joint help, hurt,
  or stay neutral vs the decoupled heads? *Any* of those three is a publishable finding.

If joint fine-tuning destabilizes (it might — that's the interference you escaped), **skip
it**. The decoupled model is a complete result on its own. Do not let Phase C re-open the
war.

---

## 5. Which config band-aids to KEEP vs DROP

Once decoupled, most interference patches are unnecessary. You don't have to delete them
immediately, but understand they're now inert/optional:

| Keep (still useful) | Drop / now irrelevant in decoupled training |
|---|---|
| `detach_reg_fpn=False` (Phase A) | Kendall log-var floors/ceilings (no joint optimization in B) |
| `reinit_pi` warm start | `ACTIVITY_HEAD_GRAD_CLIP`, `ACTIVITY_LOSS_WEIGHT` damping |
| Focal `alpha=0.90`, `gamma_pos=0` | `DETACH_PSR_FPN` (PSR trains on cache, not FPN) |
| GT-frame oversampling (Phase A) | `PSR_WARMUP_*`, `STAGE3_WARMUP_EPOCHS`, staged ramps |
| CB-Focal + label smoothing (activity) | `DET_LR_MULTIPLIER`/`DET_BIAS_LR_FACTOR` tuning |
| Transition PSR + order prior | OHEM ablation experiments (settled — not your problem) |
| Head-pose geo 6D | the entire RF1→RF10 *gauntlet* if you prefer A/B/C phases |

---

## 6. Hardware fit (12 GB, this is no longer a wall)

| Phase | What's on GPU | VRAM | Notes |
|------|----------------|------|-------|
| A | ConvNeXt-T + FPN + det head (+ head pose), batch 4 | ~6–8 GB | FP32 fine; AMP optional later |
| B-cache | frozen backbone, inference only | ~3–4 GB | one pass over dataset, then done |
| B-train | tiny heads over 512-D cache, batch 128 | ~1–2 GB | hundreds of epochs/hour; VideoMAE now affordable |
| C | full model, batch 2, LR 1e-5, 3–5 ep | ~8–10 GB | short; skip if unstable |

The "CUDA OOM wall" from your presentation doc was the *joint 5-head* configuration.
Decoupling removes it.

---

## 7. The "learning correctly" dashboard (one line per head)

Watch these every epoch instead of refreshing mAP:

| Head | Alive signal | Learning signal | Done signal |
|------|-------------|-----------------|-------------|
| Detection | `LIVENESS det` > 0.3 | `det_mAP50_pc` ↑ | plateau in 0.33–0.45 |
| Body pose | grad > 0.1 | MAE ↓ | PCK stable |
| Head pose | grad alive | MAE ≤ 15° | already there (~9°) |
| Activity | `pred_seen` ↑ past 20 | clip Top-1 ↑ | 0.35–0.45 |
| PSR | unique patterns > 10 | F1(±3) ↑ | 0.50–0.62 |

If a head is *alive and rising*, leave it alone. If it's *dead*, GUIDE_5 §5 has the
3-line triage. Do not "fix" a head that is simply still climbing.

➡ **Next:** GUIDE_3 — how to measure all of this honestly and turn it into paper tables.
