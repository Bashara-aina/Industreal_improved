# 205 — Risk Analysis, Execution Timeline, and Decision Framework

**Date:** 2026-07-10
**Documents in this round:** 195 (overview), 196 (architecture run11), 197 (results), 198 (per-head), 199 (paths), 200 (Opus prompt), 201 (Opus answer), 202 (SOTA path), 203 (architecture specs), 204 (training), 205 (this document)

---

## 1. Risk Register: What Can Kill This Paper

### Risk 1: We don't run the experiments (REPEAT OFFENDER)

**Probability:** HIGH (happened in rounds 1, 2, 3)

**Impact:** FATAL. No ST baselines = no MTL/ST ratios = paper cannot answer "does MTL help?"

**Mitigation:** The ST baselines (`train_st.py`) must be launched BEFORE any architecture changes from documents 202-204. They gate everything. GPU 2 is free.

### Risk 2: The PSR diet head doesn't learn (underpowered)

**Probability:** LOW-MEDIUM

**Evidence:** The 1.78M diet head at d=256 is 40× smaller than the 70.9M head at d=768. But both read the same P5 features. The P5 feature source fix was the load-bearing change (proven by 1.56→0.17 loss drop). A 2-layer T at d=256 on T=8 tokens should be sufficient — T=8 is a tiny sequence.

**Mitigation:** If run12 EP10 shows PSR F1 < 0.02, scale to the 5.5M detection-conditioned head (document 203 §2.1). This is the "hedge" Opus 201 described. Do NOT go back to 70.9M.

### Risk 3: Activity logit-adjust + 3-layer MLP still below random

**Probability:** LOW

**Evidence:** Below-random (0.58%) was likely class-weight collapse, not capacity. Logit-adjust directly counteracts this. The 75↔69 remap was investigated and found NOT active (`ACT_CLASS_GROUPING="none"` = identity).

**Mitigation:** If EP10 activity < 5%:
1. Verify eval label space matches training (75 vs 69)
2. Run eval WITHOUT class weights (unweighted argmax)
3. Enable decoupled training (Phase A/B from document 204 §4)
4. Only THEN consider temporal attention pool (document 203 §3.1)

### Risk 4: Detection TAL + P5/P4/P3 still gives 0.0 mAP

**Probability:** LOW-MEDIUM

**Evidence:** The alternating 0.001/4-5 loss pattern proves TAL is providing gradient signal on GT batches. But the overfit probe has never been run — if eval is broken, TAL can't fix it.

**Mitigation:** Run the overfit probe FIRST (`scripts/overfit_probe.py --head det`). If detection overfits to high accuracy on 50 clips but eval still shows 0.0 mAP → eval harness is broken. Fix eval before any architecture changes.

### Risk 5: BiFPN/GFLV2/Nash-MTL changes break existing working code

**Probability:** MEDIUM

**Impact:** HIGH (regression on currently-working heads)

**Mitigation:** Implement changes incrementally, one component at a time, with syntax check + smoke test after each. Commit after each working component. The current run12 architecture is the fallback.

### Risk 6: Adapter training takes too long / is too complex

**Probability:** MEDIUM

**Impact:** MEDIUM (training time, not correctness)

**Mitigation:** Phase 1 can be parallelized across GPUs. Each adapter trains independently in ~85 minutes. If adapter training proves too complex, skip to Phase 2 directly — joint training without adapters still works, just with slightly lower per-task performance.

### Risk 7: 78K windows is too little data for a 60M model

**Probability:** MEDIUM-HIGH

**Impact:** Overfitting → poor generalization → weak paper

**Evidence:** 60M params on 78K windows is 770 params per window. MViTv2-S single-task works at 34.5M on the same data. Adding 25M params (adapters + bigger heads) may overfit.

**Mitigation:**
1. Heavy augmentation (document 204 §3)
2. LoRA adapters have built-in regularization (low rank)
3. Decoupled training reduces overfitting on rare classes
4. Early stopping based on val performance, not train loss
5. If overfitting observed → reduce adapter rank to r=4, reduce head sizes

---

## 2. Execution Timeline: Minimal Path to Paper

### Week 1: Diagnosis + Baseline Launch (0-3 days)

| Day | Action | Time | GPU |
|-----|--------|------|-----|
| Day 1 | Run overfit probe on all 4 heads | 3 hours | GPU 1 |
| Day 1 | Fix any eval bugs found by probe | 1-4 hours | — |
| Day 1 | Launch ST-det baseline | 30 min setup | GPU 2 |
| Day 2 | ST-det running (30 epochs, ~18 hours) | — | GPU 2 |
| Day 2 | Launch ST-act baseline | 30 min | GPU 1 (night) |
| Day 3 | ST-act running | — | GPU 1 |
| Day 3 | Launch ST-psr baseline | 30 min | GPU 2 |
| Day 3 | Launch ST-pose baseline | 30 min | GPU 1 |

**End of Week 1:** 4 ST baselines running + overfit probe results + run12 EP10 eval.

### Week 2: Architecture Upgrades (Days 4-7)

| Day | Action | Time |
|-----|--------|------|
| Day 4 | Implement BiFPN + GFLV2 detection head | 4 hours |
| Day 4 | Implement 6D pose head + geodesic loss | 2 hours |
| Day 5 | Implement PSR detection-conditioned head | 3 hours |
| Day 5 | Implement temporal attention pool for activity | 2 hours |
| Day 6 | Implement LoRA + FiLM adapters | 4 hours |
| Day 6 | Implement Nash-MTL gradient bargaining | 2 hours |
| Day 7 | Integration testing + smoke test | 4 hours |

### Week 3: Training + Evaluation (Days 8-14)

| Day | Action | Time |
|-----|--------|------|
| Day 8 | Phase 1: Independent adapter training (4 tasks) | 6 hours |
| Day 9-11 | Phase 2: Joint MTL fine-tuning (45 epochs) | 26 hours |
| Day 12 | Collect all eval metrics at multiple checkpoints | 2 hours |
| Day 13 | Generate paper table + Kendall ablation results | 3 hours |
| Day 14 | Write paper L2+L3+method | Full day |

### Total: 14 days to paper-ready

**Parallelization option:** If both GPUs are available throughout, Phase 1 adapter training can be parallelized (2 tasks per GPU), reducing Phase 1 from 6 hours to 3 hours. ST baselines can also overlap with architecture implementation.

---

## 3. Decision Framework: What to Build vs What to Skip

### Tier 1: MUST DO (paper cannot exist without these)

| Item | Status | Time |
|------|--------|------|
| ST baselines (4 tasks) | Code exists, not launched | 3 days GPU time |
| Overfit probe per head | Code exists, not run | 3 hours |
| Kendall-collapse ablation | Flag exists (`--kendall-uncapped`) | 0 code, just run |
| PSR head diet (already done) | In run12 | Done |
| Activity logit-adjust (already done) | In run12 | Done |
| BiFPN detection neck | Spec in 203 §1.1 | 4 hours |
| 6D pose head + geodesic loss | Spec in 203 §4.1 | 2 hours |
| GFLV2 quality head | Spec in 203 §1.2 | 2 hours |

### Tier 2: SHOULD DO (measurable gains, moderate effort)

| Item | Status | Time | Expected gain |
|------|--------|------|---------------|
| Per-task LoRA adapters | Spec in 203 §5 | 4 hours | +3-8% per task |
| Nash-MTL gradient bargaining | Spec in 204 §1.2 | 2 hours | +2-8% vs PCGrad |
| Detection conditioning for PSR | Spec in 203 §2.1 | 3 hours | +0.05-0.15 F1 |
| Temporal attention pool for activity | Spec in 203 §3.1 | 2 hours | +5-8% top-1 |
| CopyPaste + Mosaic augmentation | Research done, code exists | 1 hour | +3-5% mAP |
| Decoupled training for activity | Spec in 204 §4 | 1 hour config change | +5-10% top-1 tail |

### Tier 3: NICE TO HAVE (would strengthen paper, higher effort)

| Item | Status | Time | Expected gain |
|------|--------|------|---------------|
| Knowledge distillation from ST teachers | Spec in 204 §2 | 80 hours (ST training) | MTL/ST → 93-97% |
| VideoMAE ViT-B backbone swap | Research done | 8 hours | +5-10% activity, +0.05 det |
| Ego4D domain adaptation | Research done | 12 hours | +3-5% across tasks |
| Model soup from ST backbones | Script exists | 1 hour | +2-5% if it works |
| FiLM modulation layers | Spec in 203 §5.2 | 1 hour | +1-3% |

### Tier 4: SKIP (diminishing returns or high risk)

| Item | Why skip |
|------|---------|
| Foundation backbone (300M+) | Efficiency claim dies |
| ArcFace / margin losses | Unproven for 75-class, fragile |
| YOLOv8-n detection branch | Breaks "shared backbone" story, use only as fallback |
| InternVideo2-1B | License risk, 1B params destroys efficiency |
| Synthetic data generation | Months of work for uncertain transfer |
| VideoMAE V2 stream | +22M params, VRAM tight, marginal over logit-adjust |

---

## 4. What Changed Since File 201 (Opus Round 4)

File 201 said: "Stop building architecture. Run the probe. Run the baselines."

We implemented: PSR diet (70.9M→1.78M), activity logit-adjust, overfit probe script, Kendall ablation flag, ST baseline fixes. run12 launched at 48.6M params with 2.06× efficiency.

**Files 202-205 go further.** They answer: "IF the probe shows the eval harness works, and IF the ST baselines establish the ceiling, WHAT architecture gives us the best chance of getting per-head numbers close enough to SOTA that the paper is compelling, not just novel?"

The answer, synthesized from 22 research agents:

1. **Keep MViTv2-S** — it's the right backbone class for the activity SOTA comparison
2. **Specialize the detection neck** — BiFPN + GFLV2 + mosaic augmentation. The backbone was pretrained for video, not detection. The neck must compensate.
3. **Isolate tasks with adapters** — LoRA + FiLM at 4.2M params eliminates gradient interference while keeping one backbone
4. **Upgrade losses** — 6D geodesic for pose, transition-aware focal for PSR, QFL for detection, decoupled training for activity
5. **Measure, don't assume** — the overfit probe tells us which of these is necessary vs cosmetic

**The paper's spine is unchanged from file 201:** Kendall-collapse characterization + per-task transfer map + genuine ~2× efficiency. The architecture upgrades in these documents make the transfer map stronger and the efficiency ratio more defensible.

---

## 5. Checklist: Is Everything in Place?

| # | Item | Document Reference | Implemented? | Verified? |
|---|------|-------------------|-------------|-----------|
| 1 | PSR head diet (1.78M) | 203 §2.1 | ✅ In run12 | ✅ Smoke test |
| 2 | Activity logit-adjust | 203 §3.2 | ✅ In run12 | ✅ Smoke test |
| 3 | Kendall uncapped ablation | 204 §7.2 | ✅ Flag exists | ❌ Never run |
| 4 | Overfit probe script | 203 (reference) | ✅ Code exists | ❌ Never run |
| 5 | ST baseline script | 204 §1 | ✅ Fixed code | ❌ Never launched |
| 6 | BiFPN neck | 203 §1.1 | ❌ Spec only | — |
| 7 | GFLV2 detection head | 203 §1.2 | ❌ Spec only | — |
| 8 | 6D pose head + geodesic | 203 §4.1 | ❌ Spec only | — |
| 9 | Detection-conditioned PSR | 203 §2.1 | ❌ Spec only | — |
| 10 | Temporal attention pool (activity) | 203 §3.1 | ❌ Spec only | — |
| 11 | LoRA adapters | 203 §5 | ❌ Spec only | — |
| 12 | Nash-MTL | 204 §1.2 | ❌ Spec only | — |
| 13 | Transition-aware focal loss | 203 §2.2 | ❌ Spec only | — |
| 14 | Decoupled activity training | 204 §4 | ❌ Spec only | — |
| 15 | Detection augmentation active | 204 §3.1 | ❌ Flag exists, not active | — |

**Items 5-15 are the implementation plan.** Items 1-4 are what should be verified/running NOW before any implementation starts.

---

## 6. Final Recommendation

**Do NOT start implementing items 5-15 yet.**

1. Wait for run12 EP10 eval (~now, check `/tmp/mtl_mvit_run12.log`)
2. Run the overfit probe on whichever heads show 0.0 at EP10
3. Launch ST baselines on GPU 2 TODAY — they are the long pole
4. THEN implement the architecture upgrades from document 203 in priority order (Tier 1 first, Tier 2 if time permits, skip Tier 3/4 entirely)

The paper is strong with just Tier 1. Tier 2 makes it stronger. Tiers 3-4 are distractions until Tier 1 is done and evaluated.
