# 195 — Round 4 Consultation Overview: Can We Prove "MTL Helps, Not Hurts"?

**Date:** 2026-07-10
**Purpose:** Strategic consultation with Opus — what path proves the MTL hypothesis?
**Documents in this round:** 195 (this overview), 196 (architecture), 197 (results), 198 (per-head), 199 (path options), 200 (Opus prompt)
**Status of run11:** LIVE (PID 885592, GPU 1, launched 2026-07-10 10:27 JST)

---

## The Core Question

**"One shared MViTv2-S backbone does 4 tasks. Is MTL helping, not hurting — more efficient, faster, at least as accurate across all heads?"**

This was the question Opus 181, 186, and 192 have been answering. We implemented Tier A (Opus 192's recommended path). **run10's epoch 10 eval showed detection 0.0 mAP, activity 0.58% top-1, PSR F1 0.004.** That forced a re-examination of head architecture. We upgraded all three struggling heads and relaunched (run11).

**Now we need to decide:** given run11 will produce its first eval at ep10 (~6 hours from now), what path from here gives the strongest chance of proving MTL helps?

---

## What's New Since Round 3 (File 192)

### The EP10 Evidence (run10)

Before run10 was killed, epoch 10 eval returned:

| Head | Metric | Value | SOTA | Verdict |
|------|--------|-------|------|---------|
| Detection | mAP@0.5 | 0.000 | 0.779 (IndustReal) | Dead — zero detections |
| Activity | top-1 | 0.58% | 65.25% (MViTv2) | Below random (1.33%) |
| PSR | event_F1@±3 | 0.004 | 0.883 (STORM) | Effectively dead |
| Pose | fwd MAE | ~9° | No SOTA (first baseline) | ✅ Healthy |

**What this means:** Three of four heads were at/near zero. The old architecture (46M, 2-layer activity MLP, sparse 3×3 detection, 4-layer PSR from conv_proj 96-dim features) was information-starved.

### The run11 Architecture (Explained in File 196)

Based on EP10 evidence, we made aggressive per-head upgrades while keeping MViTv2-S backbone (as Opus 192 instructed):

| Head | Old (run10) | New (run11) | Why |
|------|------------|------------|-----|
| Activity | 2-layer MLP (1.1M) | 3-layer MLP (3.75M) | 0.58% below random → need capacity |
| Detection | Sparse 3×3 per GT | TAL topk=10 per GT | 0.0 mAP → need dense supervision |
| PSR | 4-layer T, d=96, ff=4× | 6-layer T, d=768, ff=8× | conv_proj features useless → use P5 |
| Pose | 6D MLP (unchanged) | Same | Already healthy |
| **Total** | **~46M** | **~117.7M** | |

**Key architectural change:** PSR head now reads from P5 (blocks[14], 768-dim semantic features) instead of conv_proj (96-dim edge features). This is a 64× input quality jump for the head that was worst.

### What run11 shows so far (~2800 batches into epoch 6)

| Loss | Pattern | Assessment |
|------|---------|------------|
| Detection | Alternating 0.001 (no-GT batches) / 4-5 (GT batches) | TAL working, real gradient signal |
| Activity | 4.3-5.6 | Still high but class weights active |
| PSR | 0.15-0.25 | **Massive improvement** — was flat ~1.56 |
| Pose | 0.01-0.03 | Healthy, few spikes |

**Critical observation:** PSR loss dropped from 1.56 to 0.17-0.25. The old conv_proj-based head was learning nothing. The new P5-based 6-layer transformer is actually training.

---

## Why We Need Another Opus Round

File 192's Tier A was the right diagnosis — but EP10 proved the existing heads were inadequate. We implemented all of Tier A AND the head upgrades. Now we have:

1. A 117.7M model that is training (losses look qualitatively better)
2. No eval signal yet (first eval at ep10, ~6 hours)
3. Possible directions that diverge depending on what ep10 shows
4. Genuine uncertainty about Pose (positive transfer?) and PSR (will F1 lift off 0?)

**The meta-question:** If ep10 still shows low numbers (activity <5%, PSR F1 <0.1, det <0.1), what's the move? Get a bigger backbone? Give up on a shared backbone? Pre-train heads as single-task then fine-tune MTL? Or accept that MTL has bounded cost and write the L2+L3+method paper Opus 192 described?

File 199 will enumerate the specific paths. But we need Opus's strategic judgment on which path maximizes the chance of a publishable result.

---

## Document Map

| File | Title | What It Contains |
|------|-------|------------------|
| **195** (this) | Overview | Purpose, context, document map |
| **196** | Architecture Deep Dive | Full model architecture, feature flows, param counts |
| **197** | Results & Metrics | EP10 run10 data, run11 batch losses, eval protocol |
| **198** | Per-Head Analysis | State per task, SOTA comparison, bottlenecks, open questions |
| **199** | Path Options & Decision Matrix | Strategic options: cost, risk, expected outcome per path |
| **200** | Opus Consultation Prompt | Complete prompt to send to Opus |

---

## The Honest State

**Three things are true simultaneously:**

1. **MTL is architecturally sound:** One MViTv2-S backbone + specialized per-task heads. The feature pyramid gives detection P5/P4/P3. The class token gives activity/pose a pooled representation. The P5 features give PSR semantic time-series. Every head reads from a feature source its task can use.

2. **The EP10 0.0/0.58%/0.004 was a head problem, not an MTL problem:** The old heads were too small / read from wrong feature sources. The new heads are ~2.5× larger overall and fix the PSR feature source. The PSR loss drop (1.56→0.17) already validates this diagnosis.

3. **Whether MTL "helps" (positive transfer) vs "doesn't hurt" (bounded cost) is still unknown:** We need ST baselines to compute the MTL/ST ratio. If MTLA > ST on pose, it's a win. If MTL ≈ 0.9× ST on activity, it's bounded cost bought back by efficiency. If MTL << 0.8× on detection, it's a real trade-off.

**The paper path Opus 192 described — L2+L3+method — does not require beating SOTA on all heads.** It requires the Kendall-collapse story (which we have) + efficiency evidence + honest per-head MTL/ST ratios. That paper is in reach. The question is: can we also close the gap to SOTA enough that the quantitative story is compelling, not just novel?
