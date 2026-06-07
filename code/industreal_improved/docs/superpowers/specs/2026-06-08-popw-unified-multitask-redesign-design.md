# POPW Unified Multitask Redesign — Design Spec

**Date**: 2026-06-08
**Status**: Draft for user review
**Author**: 5-agent audit + 3-question brainstorm round
**Target**: `popw_paper_improved.tex` SOTA, not toy numbers
**Project root**: `/home/newadmin/swarm-bot/project/popw/working/code/industreal_improved/`

---

## 1. Context

### 1.1 Where we are

Current POPW model = ConvNeXt-Tiny + FPN (P3-P7, 256ch) + 5 heads (DET, POSE, ACT, PSR + HeadPoseFiLM) = 53.91M params. Trained with Kendall uncertainty weighting, LDAM-DRW, CB-Focal, 3-stage curriculum.

Quick-eval on `crash_recovery.pth` (v3 of 25% restart, epoch=2 step=397, **CLEAN SIGTERM** not crash) shows a **3-head collapse**:

| Metric | Value | Verdict |
|---|---|---|
| `det_mAP50` | 0.0000 | collapse |
| `act_top1` | 0.0000 | collapse |
| `act_top5` | 0.2031 | useful (3× random) |
| `psr_mAP50_pc` | 0.0001 | expected (PSR frozen in stage 1-2) |
| `psr_edit` | 0.0909 | useful, very early stage |

Only 2/5 metrics are non-zero. Of those, 1 (psr) is the *frozen-head regime*, not a real learning signal. We are not learning on DET or ACT.

### 1.2 Two confirmed training bugs

| # | File:line | Bug |
|---|---|---|
| 1 | `src/training/train.py:1242-1248` | `log_var.clamp(-5, 5)` runs **after** `loss.backward()`. Gradients see unclamped values; clamp only affects the displayed loss, not the parameter update. |
| 2 | `src/training/train.py:2418-2424` | Stage 3 transition resets `log_var` to 0 for all 5 heads. Should let `s_t` accumulate from stage 1 → 2 → 3. |

Both bugs are in the auto-memory (`popw-eval-loop-fix-confirmed.md`, `popw-full-bugfix-audit-20260527.md`) but not yet fixed. Until they are, every loss-curve argument is suspect.

### 1.3 The paper's mechanism

`popw_paper_improved.tex` uses **PoseFiLM** = unidirectional pose→all_heads FiLM modulation. Pose is the *only* context-sharing signal. Detection and activity are not directly coupled.

### 1.4 What this spec changes

We keep the backbone, FPN, PoseFiLM, and PSR/DET/pose heads as-is. We add **bidirectional cross-head attention on the activity head**: activity queries attend to (a) detection proposals and (b) pose keypoints, with gated residual. We do **not** add det↔pose (deferred to future work — see §10).

---

## 2. Thesis Claim (single, defensible)

> **Cross-head context sharing — specifically, allowing the activity head to attend to detection proposals and pose keypoints as a unified evidence stream — improves multi-task performance compared to independent heads.**

This is one sentence, one ablation table, one paper paragraph. It is the entire thesis contribution. Phase 2 (future work) extends to det↔pose.

---

## 3. Goals & Non-Goals

### 3.1 Goals

| Goal | Threshold | Source |
|---|---|---|
| DET mAP@0.5 | > 83.80 | paper SOTA (TBD — verify in popw_paper_improved.tex) |
| Activity Top-1 | > 66.45 | paper SOTA (TBD — verify) |
| Activity Top-5 | > 88.00 | paper SOTA, ≈ (TBD — verify) |
| PSR mAP@0.5 per-class-present | competitive with paper | paper SOTA (TBD — verify exact number; codebase uses mAP+edit, paper may report F1) |
| All val metrics | non-NaN, non-0, improving | internal |
| Training time | < 8h for 100 epochs on RTX 3060 | hardware |
| Model size | < 60M params (current 53.91M + ~6M new) | hardware |

### 3.2 Non-Goals

- Matching the paper on every single metric (some are post-processing artifacts)
- Replacing ConvNeXt with yolov8m (thesis is about cross-head, not backbone)
- Real-time inference optimization
- Online learning / continual updates
- Multi-GPU distributed training (RTX 3060 is single-GPU)

---

## 4. Architecture

### 4.1 Backbone & FPN (unchanged)

- ConvNeXt-Tiny ImageNet pretrained
- FPN P3-P7, 256 channels
- Output: per-frame feature pyramid `{P3, P4, P5, P6, P7}` ∈ ℝ^(B×256×H×W)

### 4.2 Heads — change matrix

| Head | Status | Notes |
|---|---|---|
| Detection (DET) | unchanged | per-frame object + atomic action |
| Pose (POSE) | unchanged | body (17 kp) + head (5 kp) heatmap regression |
| Activity (ACT) | **REWRITTEN** | now consumes cross-head evidence |
| PSR | unchanged | temporal causal transformer over step transitions |
| HeadPoseFiLM | unchanged | pose→all_heads FiLM modulation (paper's mechanism) |

### 4.3 NEW: CrossHeadCrossAttn module (`src/models/cross_head.py`)

Two parallel cross-attention modules, both feeding the activity head. Implemented as a single class with two contexts to keep wiring symmetric.

```
                    ┌──────────────────────────┐
                    │ Detection proposals       │
                    │ (top-k=100/frame,         │
                    │  pooled RoI features)     │
                    └────────────┬─────────────┘
                                 │ LinearProj(256→256)
                                 ▼
   ┌──────────────┐  CrossAttn(act_q, det_kv)   ┌──────────────────┐
   │ Activity     │ ◀─────────────────────────── │ + residual       │
   │ queries      │                              │ + LayerNorm      │
   │ B,T,256      │                              │ + gate σ(W·[·])  │
   │              │ ◀─────────────────────────── ┌──────────────────┐
   │              │  CrossAttn(act_q, pose_kv)   │ + residual       │
   └──────┬───────┘                              │ + LayerNorm      │
          │                                      │ + gate σ(W·[·])  │
          │  act_enhanced = act_q + g·attn(...)  └──────────────────┘
          ▼
   ┌──────────────┐
   │ existing ACT │  (linear → 75 classes)
   │ classifier   │
   └──────────────┘
                                 ▲
                                 │ LinearProj(44→256)  (22 kp × [x,y] = 44, body 17 + head 5)
                                 │
                    ┌────────────┴─────────────┐
                    │ Pose keypoint features   │
                    │ (per-frame, normalized)  │
                    └──────────────────────────┘
```

**Module spec:**
- d_model = 256, n_heads = 4, n_layers = 2
- Pre-norm residual + dropout 0.1
- Gated residual: `g = σ(W_g · [act_q; cross_attn_out])` so the head can learn to ignore the cross-attn early in training
- Output: `act_enhanced = act_q + g · cross_attn(act_q, ctx_kv)`
- ~6M extra params (2 × [linear proj 256² + 2 cross-attn layers + gates])

**Why these numbers:**
- d_model=256 matches FPN channel width → no width bottleneck
- n_heads=4 is the standard for d=256 (head_dim=64)
- n_layers=2 is sufficient because both contexts are *single-frame* — no temporal aggregation needed at the cross-attn level (PSR head still does that)
- top-k=100 detection proposals is the standard RetinaNet convention; pose uses all 17 keypoints (only 17+5=22 features per frame, no top-k needed)

### 4.4 Loss Function

```
L = Σ_t exp(-s_t) · L_t · ramp_t  +  Σ_t s_t  +  λ · L_act_consistency
     ↑ Kendall uncertainty          ↑ L2 reg on s   ↑ NEW: cross-head agreement
```

- `t ∈ {det, pose, act, psr}` (pose loss is heatmap MSE, not class)
- `s_t` = learnable `log_var` per head (clamped to [-5, 5] **before** backward — fixes bug #1)
- `ramp_t` = per-stage warmup (DET pi=0.10, pose smooth-diff, ACT focal α=0.35, PSR warmup-3-epoch on stage 3 transition)
- `L_act_consistency`: hinge loss, `max(0, margin − cos_sim(act_logits, derived_activity))`
  - `derived_activity` = argmax over a **fixed** (det_class × pose_pattern) → activity_class co-occurrence table, computed once from training-set labels (NOT learned end-to-end)
  - `margin = 0.5`
  - `λ` starts at 0, ramps to 0.1 over 5 epochs starting at stage 3 (epoch 16)

### 4.5 Staged training (refined)

| Stage | Epochs | Cross-Head | Consistency λ | `s_t` | DET pi | PSR head |
|---|---|---|---|---|---|---|
| 1 | 0-4 | disabled (g=0) | 0 | s_init | 0.10 | frozen |
| 2 | 5-14 | enabled (g=0.1 init) | 0 | accumulates | 0.10 | frozen |
| 3 | 15-19 | full | 0 → 0.1 | accumulates | 0.10 | warmup (3-ep ramp) |
| 3 | 20-100 | full | 0.1 | accumulates | 0.10 | full |

**Stage 3 transition at epoch 16 is the first real PSR test point.** psr_mAP50_pc ≈ 0 in stages 1-2 is the expected frozen regime, NOT collapse.

---

## 5. Bug Fixes (from 5-agent audit) — must land before validation

| # | Severity | File | Bug | Fix |
|---|---|---|---|---|
| 1 | **High** | `src/training/train.py:1242-1248` | `log_var.clamp` AFTER backward | Move clamp BEFORE backward |
| 2 | **High** | `src/training/train.py:2418-2424` | Stage 3 `log_var` reset to 0 | Don't reset; let `s_t` accumulate |
| 3 | **High** | `src/training/losses.py:452-456` | `LDAM_USE_DRW` flag mismatch with config | Use `config.LDAM_USE_DRW` |
| 4 | Med | `src/config.py:345` | `FOCAL_ALPHA=0.35` (non-standard) | A/B test 0.25 vs 0.35 — keep 0.35 for sparse activity (75 classes) |
| 5 | Med | `src/training/train.py:243` | `pin_memory=True` unconditional | Guard with `torch.cuda.is_available()` |
| 6 | Med | `src/training/losses.py:1130-1197` | `DET_LOSS_CAP=50` clips legit loss | A/B test remove cap; if unstable, raise to 200 |
| 7 | Low | `src/models/model.py:1304` | ActivityHead bias zero-init | Init to `log(class_freq / (1 - class_freq))` from training prior |
| 8 | Low | `src/models/model.py:1782-1837` | Pseudo-keypoint path | Remove (paper uses GT-only) |
| 9 | Low | `src/models/model.py:1480` | PSRHead bias=-1.0 | Init to `log(1/36) ≈ -3.58` (uniform over 36 steps) |
| 10 | Low | `src/models/model.py:2028-2035` | EMA auto-register | Verify EMA tracks all 5 heads, not just backbone |

All 10 fixes are pre-approved (per auto-memory). Bug #1-#3 are **mandatory** before training v4. Bug #4-#10 are quality improvements that can land in v4 or v5.

---

## 6. Eval Pipeline Hardening

Already applied in prior commits (no further action):

- PASCAL VOC 2007 any-GT matching at 3 `evaluate.py` sites — fixes the "best-UNMATCHED-GT" bug
- Per-class-present mAP for sparse val sets (commit `2c3668e`)
- eff_* NaN is the documented "not measured this run" sentinel (`SKIP_EFFICIENCY_METRICS=True`)

**Verification gate:** post-fix eval at v4 epoch 7 is the first measurement that is not contaminated by the eval bug.

---

## 7. Data Pipeline (no change)

- 36 train recordings / 17 val
- 75,457 frames (train) / ~32,000 frames (val, est)
- bs=2 at 720×1280, bs=4 at 640×640
- Frame cache (`FRAME_CACHE.clear()` in `train.py:2398-2406`)
- Async loader, pin_memory only when CUDA available

---

## 8. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Cross-head attention OOMs | Med | High | 6M not 10M params; gradient checkpointing on cross-attn; bs=2 at 720×1280 |
| det→act dominates early, head collapses to "if screwdriver visible → driving screw" | High | Med | Gate `g` init 0; ramp in stage 2; consistency loss added in stage 3 only |
| Kendall `log_var` collapses one head to 0 | Med | High | Bug fix #1: clamp BEFORE backward |
| Activity head bias init collapses to majority class | Med | Med | Bug fix #7: init to class prior |
| det/pose at different scales (boxes vs heatmaps) destabilize cross-attn | Med | Med | Linear projection (256→256, 34→256) before cross-attn |
| Stage 1-2 PSR frozen looks like collapse (false alarm) | High | Low | Already documented in auto-memory; alarm fires only at epoch 16+ |
| Cross-head gradients flow into DET/pose backbones via projection | Low | Med | `cross_head.requires_grad = False` on DET/pose feature extractors during stage 1-2 |
| New checkpoint incompatible with old eval code | Low | Med | `strict=False` in load_state_dict; per-module feature flags |

---

## 9. Success Criteria

### 9.1 Training sanity (post v4 epoch 7)

- [ ] `det_mAP50 > 0` (collapse broken)
- [ ] `act_top1 > 0.10` (was 0.0)
- [ ] `act_top5 > 0.25` (was 0.20)
- [ ] `psr_mAP50_pc ≈ 0` (expected, PSR frozen in stage 1)
- [ ] No NaN in any metric
- [ ] No loss explosion (>10× spike vs running mean)

### 9.2 SOTA parity (post v4 epoch 100)

- [ ] `det_mAP50 > 83.80` (paper)
- [ ] `act_top1 > 66.45` (paper)
- [ ] `act_top5 > 88.00` (paper, est)
- [ ] `psr_mAP50_pc` improves over crash_recovery baseline (0.0001 → > 0.05 by epoch 30)
- [ ] Training completes 100 epochs without manual intervention
- [ ] All 5 heads have non-zero, non-NaN, monotonically improving val metrics

### 9.3 Thesis validation (post v4)

- [ ] Ablation: w/ cross-head > w/o cross-head on activity Top-1 (p < 0.05 over 3 seeds)
- [ ] Ablation: det→act alone > pose→act alone (information gain)
- [ ] Ablation: det→act + pose→act > each alone (joint info gain)
- [ ] Visualizations: cross-attn maps localize to (a) detected object + (b) active hand keypoint

---

## 10. Out of Scope (Future Work)

| Direction | Why deferred | When |
|---|---|---|
| det↔pose bidirectional | det and pose are weakly correlated; feedback loop risks instability in early training; harder to ablate | Phase 2 (after thesis submission) |
| pose→det unidirectional | Marginal gain; pose already informs DET via HeadPoseFiLM | Phase 2 |
| PSR cross-head context | PSR is temporal-causal; injecting per-frame features would change its inductive bias; needs separate design | Phase 2 |
| yolov8m backbone | Rejected per user constraint; thesis is about cross-head, not backbone | TBD |
| Real-time inference | Hardware-dependent; not core thesis | Post-thesis |
| Multi-GPU DDP | RTX 3060 is single-GPU; no immediate need | TBD |

---

## 11. Rollback Plan

The cross-head module is **purely additive**. Three rollback levels:

### 11.1 Per-module disable (no code change)

```python
# src/config.py
USE_CROSS_HEAD_DET_TO_ACT = False   # disable det→act only
USE_CROSS_HEAD_POSE_TO_ACT = False  # disable pose→act only
USE_ACT_CONSISTENCY_LOSS = False    # disable consistency loss
```

### 11.2 Full cross-head disable

```python
USE_CROSS_HEAD = False  # disables all cross-head modules; falls back to baseline 5-head architecture
```

### 11.3 Checkpoint compat

New checkpoints (`cross_head_v1.*`) load into old code with:
```python
state_dict = torch.load(ckpt_path, map_location='cpu')
model.load_state_dict(state_dict, strict=False)  # ignores cross_head.* keys
```

Old checkpoints load into new code with `USE_CROSS_HEAD=False` (cross-head modules init from scratch, gate stays at 0).

### 11.4 Git rollback

```bash
git revert <commit-hash>           # single spec commit
git reset --hard <previous-sha>    # all spec changes
```

The spec itself does not modify model code. Only the implementation plan (§13) does.

---

## 12. File-by-File Impact (estimated, for the implementation plan)

| File | Change | New lines | Modified lines |
|---|---|---|---|
| `src/models/cross_head.py` | NEW: `CrossHeadCrossAttn` module | 120 | 0 |
| `src/models/model.py` | Wire cross-head into ActivityHead | 30 | 5 |
| `src/models/model.py` | Init activity head bias from class prior (bug #7) | 0 | 3 |
| `src/models/model.py` | Remove pseudo-keypoint path (bug #8) | 0 | -55 |
| `src/models/model.py` | Init PSR head bias to -3.58 (bug #9) | 0 | 1 |
| `src/models/model.py` | Verify EMA tracks all 5 heads (bug #10) | 5 | 2 |
| `src/training/losses.py` | Add `ActConsistencyLoss` | 40 | 0 |
| `src/training/losses.py` | Fix LDAM_USE_DRW flag (bug #3) | 0 | 2 |
| `src/training/losses.py` | A/B test DET_LOSS_CAP (bug #6) | 5 | 3 |
| `src/training/train.py` | Move log_var clamp BEFORE backward (bug #1) | 0 | 4 |
| `src/training/train.py` | Don't reset log_var at stage 3 (bug #2) | 0 | -6 |
| `src/training/train.py` | Guard pin_memory (bug #5) | 0 | 2 |
| `src/training/train.py` | Wire consistency λ ramp | 15 | 3 |
| `src/config.py` | New feature flags (USE_CROSS_HEAD_*, USE_ACT_CONSISTENCY_LOSS) | 12 | 0 |
| `src/config.py` | FOCAL_ALPHA=0.35 A/B flag (bug #4) | 2 | 1 |
| **Total** | | **~229 new** | **~35 modified** |

Roughly 260 lines of code change. No new dependencies (PyTorch nn.MultiheadAttention already in use).

---

## 13. Test Plan

| # | Test | Pass criteria |
|---|---|---|
| 1 | `tests/test_cross_head.py` | forward shapes match; gradient flows; gate behavior correct |
| 2 | `tests/test_ckpt_compat.py` | old ckpt → new model with `strict=False`; new ckpt → old model with `strict=False` |
| 3 | `tests/test_losses.py` | ActConsistencyLoss returns scalar; gradient flows; `λ=0` is no-op |
| 4 | Smoke test: bs=2 for 5 steps | no OOM, no NaN, loss decreasing |
| 5 | Stage-1 dry run: 5 epochs | activity head exits zero-collapse (`act_top1 > 0.05`) |
| 6 | Stage-2 dry run: 15 epochs | cross-head gate ramps from 0.1 to 0.5; det/act metrics improving |
| 7 | Stage-3 full run: 100 epochs | target SOTA on all 5 heads; no manual intervention |
| 8 | Ablation suite: w/o cross-head | baseline recovered within 1% of pre-redesign v3 |
| 9 | Ablation suite: det→act only | intermediate activity Top-1 |
| 10 | Ablation suite: pose→act only | intermediate activity Top-1 (lower than det→act) |
| 11 | Cross-attn visualization | attention maps localize to (a) detected object bbox + (b) active hand keypoint |

Tests 1-7 are mandatory before declaring "done". Tests 8-11 are thesis evidence.

---

## 14. Open Questions for User Review

1. **Bias init for activity head** (bug #7): use raw class frequency, or smoothed (Laplace +1)? Recommend +1.
2. **DET_LOSS_CAP** (bug #6): remove entirely, or raise to 200? Recommend 200 (safety net for genuine explosions).
3. **top-k detection proposals** for cross-attn: 50, 100, or 200? Recommend 100 (RetinaNet default; memory budget allows it).
4. **Consistency loss margin** = 0.5 — too tight, too loose, or about right? Recommend 0.5 (matches cosine sim range).
5. **PSR cross-head in spec or future work?** — Currently in future work. User confirmed earlier ("act can benefit from detection and pose, others for future work"). Confirm.
6. **EMA** — should the cross-head modules be inside the EMA shadow? Recommend YES (otherwise EMA-averaged model would diverge from the trained model at inference time).

---

## 15. References

- `popw_paper_improved.tex` — target paper (SOTA thresholds)
- `AUDIT_REPORT.md` — historical bug-fix log
- `POPW_FINAL_REPORT.md` — full audit chain
- auto-memory: `popw-eval-loop-fix-confirmed.md`, `popw-full-bugfix-audit-20260527.md`, `popw-stage-psr-frozen-routing-rule.md`, `popw-eval-audit-deep-20260604.md`
- PoseFiLM: paper §4.3 (pose→all_heads FiLM)
- Kendall uncertainty weighting: paper §5.2

---

**Next step (per brainstorming skill)**: user reviews this spec. If approved, invoke `superpowers:writing-plans` to create the implementation plan.
