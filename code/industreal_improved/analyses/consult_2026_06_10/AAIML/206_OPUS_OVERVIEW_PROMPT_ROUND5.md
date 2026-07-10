# 206 — Opus Round 5 Consultation Prompt: Architecture Strategy Verification

**Date:** 2026-07-10
**Documents to read:** 202 (SOTA path), 203 (implementation specs), 204 (training methodology), 205 (risks/timeline)
**Documents for context:** 201 (your Round 4 answer), 195-200 (our Round 4 prompt)
**Status:** run12 LIVE at epoch 10 eval (48.6M diet model, eval in progress)

---

## What We Did Since Round 4

You said: freeze architecture, run the probe, run the baselines, put PSR on a diet.

We did:
1. **PSR diet:** 70.9M → 1.78M (Linear 768→256 proj + 2-layer T d=256 ff=4×). run12 launched at 48.6M total, 2.06× vs ~100M specialists.
2. **Activity logit-adjust:** Enabled Menon et al. 2020 balanced softmax.
3. **Kendall ablation:** `--kendall-uncapped` flag wired for Figure 1.
4. **Overfit probe:** `scripts/overfit_probe.py` rewritten for current MViTv2-S arch.
5. **ST baseline fix:** `scripts/train_st.py` syntax bugs fixed.

We **did not** run the overfit probe or ST baselines yet. EP10 eval is happening now.

## What We Built in 202-205

After that, you asked: "make a really deep consideration of changing the architecture to make all heads closer to SOTA." We dispatched 22 research agents across backbones, detection, PSR, pose, activity, adapters, distillation, pretraining, augmentation, curriculum, gradient management, efficiency, and benchmarks. These 4 documents synthesize their findings.

### Document 202 — Architecture Path to SOTA

**Core argument:** No single backbone can simultaneously be a detection CNN, a video transformer, a state tracker, and a pose regressor. The solution is specialized branching at key points while keeping the efficiency story.

**Three levers:**
1. **Backbone pretraining:** VideoMAE on K400+SSv2 → Ego4D domain adaptation → assembly. Keeps MViTv2-S as fallback.
2. **Detection-specialized branch:** BiFPN (128ch, weighted fusion) + GFLV2 quality head + mosaic augmentation. The backbone was pretrained for video, not detection — the neck must compensate.
3. **Adapter-based task isolation:** Per-task LoRA (r=8 on Q/V) + FiLM modulation at 4.2M params (1.37% of backbone). Eliminates gradient interference while keeping one backbone.

**Parameter budget at full spec:** ~60M (34.5M backbone + 8M BiFPN/det + 15M heads + 4.2M LoRA). Efficiency: 1.67× vs ~100M specialists.

### Document 203 — Exact PyTorch Implementations

**Detection:** BiFPN neck with learned fusion weights (128ch), GFLV2DetectionHead (96ch, QFL+GIoU+DFL+quality), per-level TAL topk (9/12/15 for P3/P4/P5).

**PSR:** DetectionConditionedPSRHead (~5.2M) — P5 features + optional ROI detection features via learned gate, 2-stage hierarchical T (T=8→4), bi-directional attention (NOT causal), transition-aware focal loss.

**Activity:** TemporalAttentionPool (~2.4M) surging per-frame tokens from backbone (not shared cls_token), ImprovedActivityHead (~5M total), decoupled training (Kang et al. 2020).

**Pose:** Pose6DHead with Gram-Schmidt orthonormalization → SO(3), geodesic loss, 3-frame context (~0.5M).

**Adapters:** LoRALayer (r=8, α=16) on Q/V + FiLMLayer (γ,β) after FFN. TaskAdapterStore manages 4 sets. Total 4.2M.

### Document 204 — Training Methodology

**3-phase schedule:**
- Phase 1 (ep 1-5): Independent adapter training. Backbone frozen. Each task reaches near-ST performance. 6 hours total.
- Phase 2 (ep 6-50): Joint MTL. Backbone unfrozen. Nash-MTL gradient bargaining (replaces PCGrad). 26 hours.
- Phase 3 (ep 51-70, optional): Per-head boost for lagging heads.

**Key hyperparameters:** Backbone LR 1e-5, heads 1e-4, cosine to 1e-7. Grad clip 2.0. BF16 autocast. Effective batch 4.

**Losses:** QFL+GIoU+DFL for detection, logit-adjusted CE for activity, transition-aware focal for PSR, geodesic for pose.

**Distillation option:** Train ST teachers at 2× head capacity, distill into MTL. Expected MTL/ST: 93-97%.

**Ablation framework:** 5 configurations for Kendall-collapse Figure 1.

### Document 205 — Risks, Timeline, Decision Framework

**7 risks identified:** Not running experiments (REPEAT OFFENDER), PSR diet underpowered, activity logit-adjust insufficient, detection still zero, BiFPN/Nash-MTL breaking existing code, adapter complexity, overfitting 60M on 78K windows.

**14-day execution plan:** Diagnosis (days 1-3) → Architecture (4-7) → Training (8-11) → Paper (12-14).

**3-tier priority:**
- MUST DO (8 items): ST baselines, overfit probe, Kendall ablation, BiFPN, 6D pose, GFLV2. Paper cannot exist without.
- SHOULD DO (6 items): LoRA adapters, Nash-MTL, detection-conditioned PSR, temporal attention pool, CopyPaste augmentation, decoupled training.
- SKIP (6 items): Distillation, VideoMAE swap, Ego4D, foundation backbone, ArcFace, synthetic data.

**15-item checklist:** 4 verified, 11 pending.

---

## The Specific Questions for Opus

### Q1 — Is the 3-lever strategy correct?

Document 202 proposes: (a) better pretraining, (b) specialized detection neck, (c) adapter-based task isolation. The total parameter budget is ~60M at 1.67× efficiency.

**Is this the right architecture direction, or are we overcomplicating a problem whose real bottleneck is data volume (78K windows) and evaluation bugs we haven't ruled out?**

### Q2 — Is the PSR head diet sufficient?

The current 1.78M head (d=256, 2-layer) matches the 70.9M head on training loss (0.27 vs 0.17). Document 203 proposes a 5.2M detection-conditioned head. The two-stream research agent recommends a completely different PSR approach via ASFormer-style decoder.

**Should we trust the 1.78M head and focus on data/loss, or is the 5.2M detection-conditioned head worth the param cost? Does the research agent's ASFormer recommendation conflict with our transformer approach?**

### Q3 — Is adapter-based MTL the right mechanism?

Per-task LoRA at 4.2M + Nash-MTL. The research shows this can match or beat full fine-tuning while eliminating gradient interference.

**But: does this complicate the paper's story? "Shared backbone + adapters + Nash-MTL" is more complex than "shared backbone + Kendall caps." Is the added complexity justified by the expected 3-8% gain?**

### Q4 — Which of the many architecture options in 203 are load-bearing?

Document 203 specifies: BiFPN, GFLV2, detection-conditioned PSR, temporal attention pool, 6D geodesic pose, LoRA, FiLM, Nash-MTL, transition-aware focal, decoupled training, GFLV2 quality head, per-level TAL topk.

**Which 3-5 of these give 80% of the gain? Which are cosmetic? We cannot implement all 12 in two weeks.**

### Q5 — Does the two-stream agent's SlowFast-Hydra-MTAN recommendation (47.4M) change the architecture direction?

One agent recommended a completely different architecture: SlowFast backbone + MTAN attention masks + FPN + ASFormer decoder at 47.4M. This would require a full rewrite of our MViTv2-S codebase.

**Is this a better foundation than upgrading our existing MViTv2-S, or is it a distraction from the experiments we still haven't run?**

### Q6 — Are we still over-investing in architecture before diagnosis?

You said in Round 4: "Stop building architecture. Run the probe and baselines."

We built architecture documents 202-205 and changed the PSR head. We still haven't run the overfit probe or ST baselines. EP10 eval is happening now.

**Honest verdict: are 202-205 another round of "building architecture instead of running experiments," or is this the right preparatory analysis to make the experiments meaningful?**

### Q7 — What's the one thing we should do tomorrow?

From the 15-item checklist (4 done, 11 pending), the 22-agent research findings, the 5 risk items, and whatever EP10 shows — **what is the single highest-leverage action?**

---

## EP10 Results (to be filled)

```
[PENDING — eval in progress]
Expected ~30-40 minutes from 15:26 JST start
```

Once EP10 numbers are available, append them here before sending to Opus. The decision tree branches on:
- If detection > 0.01 mAP → TAL is working, proceed with BiFPN
- If activity > 5% top-1 → logit-adjust fixed the collapse, proceed with temporal pool
- If PSR F1 > 0.02 → P5 feature fix is real, proceed with detection-conditioning
- If any head still ~0 → run overfit probe FIRST before any architecture change
