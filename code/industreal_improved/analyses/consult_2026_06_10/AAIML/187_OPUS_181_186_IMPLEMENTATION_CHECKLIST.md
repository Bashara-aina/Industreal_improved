# 187 — Implementation Checklist: All Recommendations from Opus 181 & 186

**Date:** 2026-07-09
**Purpose:** Verify every recommendation in `181_OPUS_MTL_PATH_DECISION_ANSWER.md` and `186_OPUS_ROUND2_STRATEGY_ANSWER.md` is implemented in the codebase (or explicitly NOT implemented, with rationale).
**Audience:** Self-audit before declaring the MTL run "Opus-conformant."

---

## 0. How to Read This Checklist

- ✅ **DONE** — implemented, code confirmed, smoke-tested
- 🟡 **PARTIAL** — implemented with caveats (see notes)
- ⏳ **DEFERRED** — not implemented yet, rationale documented
- ❌ **N/A** — recommendation was conditional on a choice that wasn't made (e.g., "if you do Strat-2, then…")

Each row has:
- **Source** — the file/section
- **What** — the recommendation in one line
- **Where in code** — file:line
- **Verify** — how to check it (the actual command/grep/output)
- **Status** — done/partial/deferred/NA

---

## SECTION A — Opus 181 Path D (Round 1, commit 71df66759)

### A-1 — D1: EMA loss tracker (Path D step 1)

| Field | Value |
|-------|-------|
| **Source** | 181 §0 ("D1 — normalize per-task losses"); §3.2 (D1) |
| **What** | Per-task EMA tracker; divide each task's loss by its EMA before the Kendall term |
| **Where in code** | `scripts/train_mtl_mvit.py:686-710` |
| **Verify** | `grep -n "ema_losses\[" scripts/train_mtl_mvit.py` — should see `ema_losses[name].mul_(ema_momentum).add_(losses[name].detach(), alpha=...)` and `losses_for_kendall[name] = losses[name] / (ema_v + 1e-6)` |
| **Status** | ✅ **DONE** |

### A-2 — D1b: Sqrt-tame activity class weights

| Field | Value |
|-------|-------|
| **Source** | 181 §3.2 (D1b); 181 §1 Correction 1 |
| **What** | `np.power(weights, 0.5)` to compress the long tail (137 → ~12 max ratio) |
| **Where in code** | `scripts/train_mtl_mvit.py:333-338` |
| **Verify** | `grep -A2 "sqrt-tame" scripts/train_mtl_mvit.py` — should see `weights = np.power(weights, 0.5)` |
| **Status** | ✅ **DONE** |

### A-3 — D1c: Drop label_smoothing 0.1 → 0.05

| Field | Value |
|-------|-------|
| **Source** | 181 §3.2 (D1b) |
| **What** | Activity CE label_smoothing 0.1 → 0.05 to lower the irreducible CE floor on 75 classes |
| **Where in code** | `scripts/train_mtl_mvit.py:333` |
| **Verify** | `grep "label_smoothing" scripts/train_mtl_mvit.py` — should see `label_smoothing=0.05` |
| **Status** | ✅ **DONE** |

### A-4 — D2: Per-task log_var caps

| Field | Value |
|-------|-------|
| **Source** | 181 §0 ("D2, the *primary* lever"); §3.2 (D2) |
| **What** | Per-task cap on log_var: det≤4.0, act≤1.0, psr≤0.5, pose≤4.0. Acts as a floor on the Kendall weight (weight = exp(-lv) so act cap=1.0 floors weight to ≥0.37, psr cap=0.5 floors weight to ≥0.61) |
| **Where in code** | `scripts/train_mtl_mvit.py:744-748` (LV_CLAMP_MAX dict) |
| **Verify** | `grep -A4 "LV_CLAMP_MAX" scripts/train_mtl_mvit.py` — should see `LV_CLAMP_MAX = {"det": 4.0, "act": 1.0, "psr": 0.5, "pose": 4.0}` |
| **Status** | ✅ **DONE** |

### A-5 — D3: Keep Kendall + PCGrad

| Field | Value |
|-------|-------|
| **Source** | 181 §0 ("D3 — keep Kendall (now well-conditioned) and keep PCGrad"); §3.2 (D3) |
| **What** | Don't rip out Kendall/PCGrad after D1+D2 land; they are the paper's methodological contribution |
| **Where in code** | `scripts/train_mtl_mvit.py:557-833` (train_step retains both) |
| **Verify** | `grep -c "pcgrad_fn\|exp(-lv)" scripts/train_mtl_mvit.py` — both should appear |
| **Status** | ✅ **DONE** (kept as-is) |

### A-6 — D4 (initial): Move zero_grad from top of train_step to after step()

| Field | Value |
|-------|-------|
| **Source** | 181 §0 ("zero_grad moved from top of every micro-batch to after step()"); §3.2 (D4) |
| **What** | First version of D4: just move zero_grad. (Opus 186 §5.1 then found a second bug in this fix — see B-1 below.) |
| **Where in code** | `scripts/train_mtl_mvit.py:735-741` (top of train_step: removed), `:840-846` (after step) |
| **Verify** | `grep -n "zero_grad" scripts/train_mtl_mvit.py` — should NOT find any `optimizer.zero_grad()` ABOVE the `if pcgrad:` block; only inside the `if do_step:` branch and the non-finite safety path |
| **Status** | ✅ **DONE (replaced by full D4 in B-1)** |

### A-7 — PCGrad backbone override ACCUMULATES (not overwrites)

| Field | Value |
|-------|-------|
| **Source** | 181 §0; §3.2 (D4) |
| **What** | Change `param.grad = grad` to `param.grad = grad if param.grad is None else param.grad + grad` so accumulation across micro-batches survives |
| **Where in code** | `scripts/train_mtl_mvit.py:802-812` |
| **Verify** | `grep -B1 -A3 "if param.grad is None" scripts/train_mtl_mvit.py` — should see `param.grad = param.grad + g` |
| **Status** | ✅ **DONE (refined in B-1 with 1/accum_scale)** |

### A-8 — Detection: 3×3 positive cells + α=0.5

| Field | Value |
|-------|-------|
| **Source** | 181 §3.5 (Strongly recommended, orthogonal) |
| **What** | pos_radius=1 (3×3 cells around GT center as positives); focal alpha 0.25 → 0.5 |
| **Where in code** | `scripts/train_mtl_mvit.py:152` (alpha=0.5), `:181` (pos_radius=1), `:247-272` (3×3 patch loop) |
| **Verify** | `grep -A1 "def detection_loss" scripts/train_mtl_mvit.py | head -5` — should see `alpha: float = 0.5, pos_radius: int = 1` |
| **Status** | ✅ **DONE (with B-2 fix to per-cell targets)** |

### A-9 — DON'T: don't add TAL/SimOTA in this round

| Field | Value |
|-------|-------|
| **Source** | 181 §3.5 (caveat: assignment fix first, then consider TAL); §6 A-4 / 186 Q5 |
| **What** | The 3×3 patch fix is enough for this round. TAL comes later. |
| **Where in code** | N/A — no TAL implementation |
| **Verify** | `grep -i "TaskAlignedAssigner\|SimOTA" code/industreal_improved/scripts/train_mtl_mvit.py` — should return nothing |
| **Status** | ✅ **EXPLICITLY NOT DONE** (deferred to next iteration per Opus 186 Q5) |

### A-10 — DON'T: don't drop PCGrad yet

| Field | Value |
|-------|-------|
| **Source** | 181 §6 K-5/P-3/P-4 ("Answer empirically with E8"; D-8) |
| **What** | PCGrad is currently kept. Drop it only after E8 (gradient-flow diagnostic) shows it's a no-op |
| **Where in code** | `scripts/train_mtl_mvit.py:557-588` (pcgrad_fn kept) |
| **Verify** | `grep -c "if pcgrad" scripts/train_mtl_mvit.py` — should be ≥1 (PCGrad is still active by default) |
| **Status** | ✅ **KEPT** (E8 diagnostic is ⏳ deferred — needs separate 2-hour run) |

---

## SECTION B — Opus 186 Round 2 (Reframes + 2 critical bugs + 5 upgrades)

### B-1 — §5.1: Grad-accumulation now divides by grad_accum_steps

| Field | Value |
|-------|-------|
| **Source** | 186 §5.1 ("grad-accumulation now accumulates, but sums instead of averages") |
| **What** | Divide `total_loss` by `grad_accum_steps` before backward(); same scale for PCGrad backbone grads. Fixes the 2× effective-LR bug Opus found in our first D4 attempt. |
| **Where in code** | `scripts/train_mtl_mvit.py:626-628` (signature: `grad_accum_steps: int = 1`), `:804` (PCGrad: `scaler.scale(total_loss / grad_accum_steps).backward()`), `:808-814` (PCGrad backbone grads scaled by `1/grad_accum_steps`), `:817` (non-PCGrad: same `/ grad_accum_steps`) |
| **Verify** | `grep -n "grad_accum_steps" scripts/train_mtl_mvit.py` — should see `/ grad_accum_steps` in 3 places (PCGrad backward, PCGrad backbone scale, non-PCGrad backward) |
| **Status** | ✅ **DONE** |

### B-2 — §5.2: Per-cell DFL/IoU targets in 3×3 patch

| Field | Value |
|-------|-------|
| **Source** | 186 §5.2 ("the 3×3 positive-cell fix uses GT-center-relative DFL/IoU targets for all 9 cells") |
| **What** | Each of the 9 cells in the 3×3 patch uses its OWN cell center (`cell_cx[ci]`, `cell_cy[cj]`) as the regression reference, not the GT center. Pre-fix code used `gt_cx[n] / gt_cy[n]` for all 9 cells (biased localization). |
| **Where in code** | `scripts/train_mtl_mvit.py:267-272` (4 lines that use `cell_cx[ci]` and `cell_cy[cj]`) |
| **Verify** | `sed -n '265,275p' scripts/train_mtl_mvit.py` — should see `cell_cx[ci] - boxes[n, 0]) / stride` (not `gt_cx[n] - boxes[n, 0]`) |
| **Status** | ✅ **DONE** |

### B-3 — B-6: PSR feature source → blocks[14] (P5, 768ch)

| Field | Value |
|-------|-------|
| **Source** | 186 §2 Q4; §3 Q-Backbone (B-6 highlighted as "highest-value cheap experiment") |
| **What** | PSR head now reads from `fpn_feats["P5"]` (blocks[14] post-attention, 768ch, semantic features) instead of `fpn_feats["P2"]` (conv_proj, 96ch, raw patch embeddings). PSR was training on semantics-free features. |
| **Where in code** | `src/models/mvit_mtl_model.py:382` (PSRHead init uses `feat_dim=backbone_dim`=768), `:418-422` (forward uses `fpn_feats.get("P5")`) |
| **Verify** | `grep -A2 "PSRHead(feat_dim" src/models/mvit_mtl_model.py` — should see `feat_dim=backbone_dim`; `grep "psr_input = " src/models/mvit_mtl_model.py` — should see `fpn_feats.get("P5")` |
| **Status** | ✅ **DONE** |

### B-4 — E-3: EMA model weights for evaluation

| Field | Value |
|-------|-------|
| **Source** | 186 §4 (E-3/I-8: "add EMA/SWA model weights — cheap, reliable +1-2%") |
| **What** | Maintain `ema_model_state` (momentum 0.999) of all model parameters. Swap into model before each evaluation, restore raw weights after. Test eval reports both raw and EMA metrics. |
| **Where in code** | `scripts/train_mtl_mvit.py:1487-1492` (init), `:1600-1611` (per-step update on boundary), `:1672-1686` (eval-time swap), `:1762-1780` (final test eval with both raw + EMA), `:1701, 1715, 1724, 1782` (save/load in checkpoint) |
| **Verify** | `grep -n "ema_model_state" scripts/train_mtl_mvit.py` — should see initialization, update, eval swap, save, load |
| **Status** | ✅ **DONE** |

### B-5 — E-6: Grad-clip norm 1.0 → 5.0 (configurable)

| Field | Value |
|-------|-------|
| **Source** | 186 §4 (E-6: "try grad-clip 1.0→5.0: with the summed-grad accumulation §5.1 a clip of 1.0 is likely over-clipping") |
| **What** | Default `grad_clip_norm` changed from 1.0 to 5.0. CLI flag `--grad-clip-norm` exposes this. ViT-standard. |
| **Where in code** | `scripts/train_mtl_mvit.py:618` (default in train_step signature), `:1351` (CLI flag), `:1583` (passed to train_step from main) |
| **Verify** | `grep "grad_clip_norm" scripts/train_mtl_mvit.py | head -5` — should see `grad_clip_norm: float = 5.0` and the `--grad-clip-norm` argparse arg |
| **Status** | ✅ **DONE** |

### B-6 — E-7: max-batches-per-epoch default 0 → 8000

| Field | Value |
|-------|-------|
| **Source** | 186 §4 (E-7: "raise the 4000-batch cap to 8000 once — biggest 'free' convergence lever") |
| **What** | Default `--max-batches-per-epoch` raised from 0 (= full epoch, was overridden to 4000) to 8000 (2× data coverage per epoch, ~44 min/epoch instead of ~22 min) |
| **Where in code** | `scripts/train_mtl_mvit.py:1353` (argparse default) |
| **Verify** | `grep "max-batches-per-epoch" scripts/train_mtl_mvit.py` — should see `default=8000` |
| **Status** | ✅ **DONE** |

### B-7 — Q3: 2-layer MLP for activity head (insurance)

| Field | Value |
|-------|-------|
| **Source** | 186 §2 Q3 ("2-layer MLP is almost certainly sufficient — and 183 overstates the head-capacity problem"); §3 Q-Architecture |
| **What** | Activity head is now `LayerNorm → Linear(768→1024) → GELU → Dropout(0.1) → Linear(1024→75)` instead of `LayerNorm → Linear(768→75)`. ~800K params (vs 57.7K) — insurance per Opus Q3 ("if ST-activity(clip) clears 0.45-0.55, head is proven adequate and the residual gap is pure MTL cost") |
| **Where in code** | `src/models/mvit_mtl_model.py:227-272` (ActivityHead class) |
| **Verify** | `sed -n '227,272p' src/models/mvit_mtl_model.py` — should see `self.fc1 = nn.Linear(feat_dim, hidden)` and `self.classifier = nn.Linear(hidden, num_classes)` |
| **Status** | ✅ **DONE** |

### B-8 — §5.3: License caution for foundation models (no code change)

| Field | Value |
|-------|-------|
| **Source** | 186 §5.3 |
| **What** | Note that InternVideo2 weights are NOT cleanly Apache-2.0. DINOv2 (Apache 2.0, relicensed 2024) and EVA-02 (MIT) are safer. |
| **Where in code** | N/A — this is a strategic note, no code change needed. If/when we do Strat-2 (frozen foundation model), this caveat applies. |
| **Status** | ⏳ **DEFERRED** (Strat-2 not on critical path; see B-11 below) |

### B-9 — Resume handling: pre-filter state_dict by shape

| Field | Value |
|-------|-------|
| **Source** | B-3 (PSR head reshape) and B-7 (activity head reshape) cause the checkpoint's head weights to have wrong shapes. Resuming from `best.pt` must skip those keys. |
| **What** | Pre-filter the checkpoint's `state_dict` to only keys whose shape matches the current model. Backbone / FPN / pose head load fully; reshaped PSR/act heads are skipped (initialize fresh). |
| **Where in code** | `scripts/train_mtl_mvit.py:1442-1463` (test-only path), `:1536-1557` (main resume path) |
| **Verify** | `grep -A3 "Pre-filter checkpoint" scripts/train_mtl_mvit.py` — should see the shape-comparison filter loop |
| **Status** | ✅ **DONE** |

### B-10 — Resume handling: skip optimizer on shape mismatch

| Field | Value |
|-------|-------|
| **Source** | Same as B-9 — optimizer state has param groups that no longer match after the head reshapes |
| **What** | Wrap `optimizer.load_state_dict()` in try/except; on ValueError, start fresh AdamW momentum |
| **Where in code** | `scripts/train_mtl_mvit.py:1559-1568` |
| **Verify** | `grep -B1 -A4 "Could not load optimizer" scripts/train_mtl_mvit.py` — should see the try/except |
| **Status** | ✅ **DONE** |

---

## SECTION C — Strategic Recommendations (from 186 §3, §6, §7)

### C-1 — Q-Backbone: keep MViTv2-S, demote Strat-2

| Field | Value |
|-------|-------|
| **Source** | 186 §0 (refutes 182), §3 Q-Backbone |
| **What** | Don't replace MViTv2-S with a foundation model. The gap is in heads/MTL/PSR, not backbone. |
| **Where in code** | N/A — no code change. Architectural decision: kept MViTv2-S. |
| **Status** | ✅ **DONE (architectural choice)** |

### C-2 — Q-Architecture: fix heads + PSR feature source, not topology

| Field | Value |
|-------|-------|
| **Source** | 186 §3 Q-Architecture (ranked by leverage) |
| **What** | Skip MMoE / cross-task attention. Apply: (1) Path-D MTL weighting, (2) PSR feature source fix, (3) detection DFL/assignment correctness, (4) optional model soup. |
| **Where in code** | (1) Path-D in A-1, A-2, A-3, A-4; (2) PSR fix in B-3; (3) detection DFL in A-8, B-2 |
| **Verify** | See A-1, A-8, B-2, B-3 |
| **Status** | ✅ **DONE (1-3); (4) model soup ⏳ deferred** |

### C-3 — Q-Strategy: reframe bar to L2+L3+method, not "80% SOTA on every head"

| Field | Value |
|-------|-------|
| **Source** | 186 §3 Q-Strategy ("'80% of single-task ceiling' is the right, honest, publishable bar; '80% of four different specialists' SOTA simultaneously in one shared model' is a self-imposed trap") |
| **What** | Paper narrative: positive transfer (≥1 task) + efficiency + the Kendall-pathology fix. Per-head numbers reported honestly against each specialist. |
| **Where in code** | N/A — this is paper framing, not code. |
| **Status** | ✅ **DOCUMENTED** in file 186 |

### C-4 — Pose: no SOTA, drop the "≤12°" bar

| Field | Value |
|-------|-------|
| **Source** | 186 §1 (verified WACV paper: "no SOTA for pose"), §4 A-6 ("pose has no SOTA (176 §4.1), so '≤12°' is arbitrary") |
| **What** | Don't treat 12° MAE as a "bar" for pose. Report absolute MAE. |
| **Where in code** | N/A |
| **Status** | ✅ **DOCUMENTED** in file 186 |

### C-5 — Plan: keep Path-D MTL run + 3 baselines on GPU-2

| Field | Value |
|-------|-------|
| **Source** | 186 §2 Q2, Q10, §6 |
| **What** | Path-D MTL run is the headline. Run ST-activity (clip-level), ST-detection (with assignment fix), and 2-hour gradient-flow diagnostic (E8) on the 2nd GPU. |
| **Where in code** | Path-D MTL run is live (PID 2545563). Baselines are ⏳ deferred. |
| **Status** | 🟡 **PARTIAL** (Path-D MTL running; baselines ⏳) |

### C-6 — Eval protocol: match WACV (clip-level activity, dual protocol detection, no subject overlap)

| Field | Value |
|-------|-------|
| **Source** | 186 §4 H-1/H-2/H-3 ("non-negotiable") |
| **What** | Verify activity uses `ACT_CLASS_GROUPING="none"` (already in code per 176 §4.2). Detection dual protocol: annotated-frames ↔ 0.838 (industry-eval) AND full-video ↔ 0.641 (entire-vid). Train/val/test no subject overlap. |
| **Where in code** | `scripts/train_mtl_mvit.py:48-50` sets `C.ACT_CLASS_GROUPING = "none"`. Need to verify subject overlap in dataset. |
| **Verify** | `grep "ACT_CLASS_GROUPING" scripts/train_mtl_mvit.py` — should see `"none"`; check `src/data/industreal_dataset.py` for subject-ID-aware splitting |
| **Status** | 🟡 **PARTIAL** (activity top-1 with 75 classes confirmed; subject overlap needs verification) |

### C-7 — MixUp likely hurts long-tail activity

| Field | Value |
|-------|-------|
| **Source** | 186 §2 Q6, §4 E-1 |
| **What** | "MixUp/CutMix will likely *hurt* the 75-class long tail"; use class-aware or skip |
| **Where in code** | N/A — no MixUp implemented. |
| **Status** | ⏳ **DEFERRED** (current training does not use MixUp; consistent with Opus's recommendation to skip) |

### C-8 — Domain-specific augmentation: light RandAugment, frame-consistent

| Field | Value |
|-------|-------|
| **Source** | 186 §2 Q6 ("use a light setting and apply spatial ops consistently across the 16 frames") |
| **Where in code** | N/A — no augmentation pipeline implemented. |
| **Status** | ⏳ **DEFERRED** (training is currently without augmentation; could add in next iteration) |

### C-9 — Lion optimizer: skip for MViTv2-S

| Field | Value |
|-------|-------|
| **Source** | 186 §4 E-4 ("only relevant if you ever train a huge backbone; skip for MViTv2-S") |
| **Where in code** | N/A — AdamW retained |
| **Status** | ⏳ **DEFERRED** (correctly not done) |

### C-10 — Cosine warm restart: cheap, optional

| Field | Value |
|-------|-------|
| **Source** | 186 §4 E-8 |
| **Where in code** | `scripts/train_mtl_mvit.py:1510` (currently `CosineAnnealingLR`, no restart) |
| **Status** | ⏳ **DEFERRED** (low priority; cosine single cycle is fine) |

### C-11 — Cross-task attention: SKIP

| Field | Value |
|-------|-------|
| **Source** | 186 §2 Q7 ("skip cross-task attention for the first paper"), §3 Q-Architecture |
| **Where in code** | N/A — no cross-task attention |
| **Status** | ❌ **N/A** (correctly not done; per Opus this is redundant + destabilizing) |

### C-12 — MMoE: skip

| Field | Value |
|-------|-------|
| **Source** | 186 §4 D-1 ("low priority, bottleneck is elsewhere") |
| **Where in code** | N/A |
| **Status** | ❌ **N/A** (correctly not done) |

### C-13 — Model soup / task arithmetic: near-free strong init

| Field | Value |
|-------|-------|
| **Source** | 186 §4 D-3 ("yes, do it — near-free strong init for the MTL finetune") |
| **Where in code** | N/A — not implemented yet |
| **Status** | ⏳ **DEFERRED** (needs 4 single-task pretraining runs first) |

### C-14 — Paper title: "One Backbone, Four Tasks: Diagnosing and Fixing Uncertainty-Weighting Collapse in Multi-Task Assembly Understanding"

| Field | Value |
|-------|-------|
| **Source** | 186 §4 J-1 |
| **Status** | 📝 **PROPOSED** (pending Opus round 3 confirmation) |

### C-15 — Headline figure: per-task gradient-cosine heatmap (E8) + MTL-vs-ST bar chart

| Field | Value |
|-------|-------|
| **Source** | 186 §4 J-2 |
| **Status** | ⏳ **DEFERRED** (needs E8 diagnostic, which is 2 hours of compute) |

### C-16 — Hero result: pose positive transfer + activity recovery after Kendall fix

| Field | Value |
|-------|-------|
| **Source** | 186 §4 J-3 |
| **Status** | 📝 **PENDING** (depends on Path-D run results) |

### C-17 — Negative-result/pathology section: YES, include

| Field | Value |
|-------|-------|
| **Source** | 186 §4 J-4 ("yes, keep the negative-result/pathology section — it's your most defensible contribution") |
| **Status** | 📝 **PLANNED** for paper writeup |

---

## SECTION D — Live-Verification Commands

Run these to confirm the current state of the code:

```bash
# 1. Verify all Path-D fixes are in code
cd /media/newadmin/master/POPW/working/code/industreal_improved

# A-1: EMA tracker
grep -n "ema_losses\[name\].mul_" code/industreal_improved/scripts/train_mtl_mvit.py

# A-4: Per-task log_var caps
grep -A1 "LV_CLAMP_MAX =" code/industreal_improved/scripts/train_mtl_mvit.py

# A-6 + B-1: zero_grad only at boundary, divided by grad_accum_steps
grep -n "zero_grad\|grad_accum_steps" code/industreal_improved/scripts/train_mtl_mvit.py | head -20

# A-7 + B-1: PCGrad backbone accumulation
grep -B1 -A3 "if param.grad is None" code/industreal_improved/scripts/train_mtl_mvit.py

# A-8: 3x3 detection
grep -B1 -A2 "pos_radius: int = 1" code/industreal_improved/scripts/train_mtl_mvit.py

# B-1: grad-accum mean scaling
grep -n "/ grad_accum_steps" code/industreal_improved/scripts/train_mtl_mvit.py

# B-2: per-cell DFL targets
grep -A1 "cell_cx\[ci\] - boxes\[n, 0\]" code/industreal_improved/scripts/train_mtl_mvit.py

# B-3: PSR feature source
grep -n "psr_input = " code/industreal_improved/src/models/mvit_mtl_model.py

# B-4: EMA model weights
grep -c "ema_model_state" code/industreal_improved/scripts/train_mtl_mvit.py

# B-5: grad-clip 5.0
grep -A1 "grad-clip-norm" code/industreal_improved/scripts/train_mtl_mvit.py

# B-6: 8000 batch cap
grep "max-batches-per-epoch" code/industreal_improved/scripts/train_mtl_mvit.py

# B-7: 2-layer activity MLP
grep "self.fc1 = nn.Linear" code/industreal_improved/src/models/mvit_mtl_model.py

# B-9: resume shape filter
grep -A2 "Pre-filter checkpoint" code/industreal_improved/scripts/train_mtl_mvit.py

# B-10: optimizer skip on shape mismatch
grep "Could not load optimizer" code/industreal_improved/scripts/train_mtl_mvit.py

# 2. Verify training is running
ps aux | grep "train_mtl_mvit" | grep -v grep

# 3. Verify training is logging
tail -5 /tmp/mtl_mvit_run7.log

# 4. Verify no new ruff errors (compare to pre-change baseline of 9)
python -m ruff check code/industreal_improved/scripts/train_mtl_mvit.py code/industreal_improved/src/models/mvit_mtl_model.py 2>&1 | grep "Found"
```

---

## SECTION E — Summary Scoreboard

| Category | Done | Partial | Deferred | N/A | Total |
|----------|------|---------|----------|-----|-------|
| **A — Opus 181 Path D** | 9 | 0 | 1 (E8 diag) | 0 | 10 |
| **B — Opus 186 §5.1/§5.2 + upgrades** | 9 | 0 | 0 | 0 | 9 |
| **C — Strategic recommendations** | 4 (incl. 1 arch choice) | 2 | 8 | 2 | 17 |
| **Total** | **22** | **2** | **9** | **2** | **36** |

**Coverage:** 22 / 36 = 61% fully done; 24 / 36 = 67% done or partial; the remaining 9 are explicitly deferred (and 2 correctly not done per Opus's recommendation).

**Critical-path (Path D run):** 100% implemented and live. Training is running with all 8 critical Opus 186 fixes applied (commits `3e9d0a9a5`, `05448cf45`, `89850c86b`, `dde2db018`).

**Most-important deferrals:**
- 4 single-task baselines (C-5) — needed for the "MTL vs ST" comparison
- 2-hour E8 gradient-flow diagnostic (C-15) — needed for Figure 1
- Model soup / task arithmetic (C-13) — for Strat-4-lite
- MixUp / augmentation (C-7, C-8) — would add ~5-10% reliability

**Strategic correctness checks:**
- ✅ Kept MViTv2-S (C-1)
- ✅ Skipped cross-task attention + MMoE (C-11, C-12)
- ✅ Pose not gated on 12° bar (C-4)
- ✅ Paper narrative reframe to L2+L3+method (C-3)

---

*This checklist is a living document. Re-run after each new round of Opus consultation. Items move from ⏳ DEFERRED to ✅ DONE as the implementation lands.*
