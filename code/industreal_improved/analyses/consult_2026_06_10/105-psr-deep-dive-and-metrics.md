# 105 — PSR Deep Dive: Current State, Eval Bug & Path to Benchmarkable Results

**Date:** 2026-07-03 | **Run:** PID 2988577 epoch 6

---

## 1. PSR Architecture Summary

- **11 binary components** — each representing a sub-step of the furniture assembly procedure
- **Transformer**: 3 layers, 4 heads, d_model=256, GRU hidden=256
- **Training frequency**: Every PSR_SEQ_EVERY_N_BATCHES=4 batches (25% of batches)
- **Sequence length**: 8 consecutive frames per sequence batch
- **Backbone isolation**: detach_psr_fpn=True (PSR gradient never reaches FPN/backbone)
- **Loss amplification**: PSR_WEIGHT=10 × PSR_SEQ_LOSS_SCALE=1.5 = 15x effective on seq batches
- **Transition objective**: USE_PSR_TRANSITION=True with temporal smoothness
- **Target**: Component binary accuracy 0.65+, transition F1@±3 0.15+

## 2. Metrics Comparison: Epoch 2 → Epoch 5

| Metric | Epoch 2 | Epoch 5 | Δ | Interpretation |
|---|---|---|---|---|
| Binary accuracy | **0.291** | **0.554** | +90% | Above chance (0.5) first time |
| Unique patterns | 4 | 5 | +25% | Still very low (target 500+) |
| Sigmoid range | [-1.1,1.5] | **[-4.3,3.6]** | 3x wider | Confidence separating |
| First frame logit[0] | ~0.0 | **1.797** | +inf | Component 0 now confidently active |
| psr_f1 | 0.0 | 0.0 | — | **Eval bug** |
| psr_edit | 0.0 | 0.0 | — | **Eval bug** |
| psr_pos | 0.0 | 0.0 | — | **Eval bug** |

## 3. Eval Pipeline Bug: MonotonicDecoder Crash

The validation log shows:
```
[PSR METRICS] Failed: only 0-dimensional arrays can be converted to Python scalars -- using safe defaults
```

This means the MonotonicDecoder receives malformed input and falls through to `except Exception`, producing all-zero metrics. The actual PSR transition performance is UNKNOWN — it could be 0.05 or 0.25, we can't tell.

**Root cause:** The decoder expects (N, 11) binary predictions but receives (N, 1, 11) or (N,) shaped arrays due to a dimension squeeze in the evaluation pipeline. This is a post-processing shape mismatch, not a training failure.

**Without this fix, PSR transition metrics are invisible.** Fix priority: HIGH — it blocks gate decisions on PSR.

## 4. Per-Component Prevalence

PSR components have heavily imbalanced prevalence:
- Component 0: ~1.0 (always active) — now predicted correctly at 0.858 sigmoid
- Components 1, 2, 5, 6: ~0.6-0.8
- Components 3, 7, 8, 9: ~0.3-0.5
- Component 4: ~0.19-0.22 (rare — only active in 19-22% of frames)

The model's sigmoid outputs at epoch 5 first frame:
```
[0.858, 0.602, 0.701, 0.911, 0.971, 0.686, 0.852, 0.965, 0.95, 0.943, 0.95] 
binary: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
```

The model predicts ALL-1s for the first frame, which has 11/11 components active. This matches the fill-forward label pattern (once a component activates, it stays active).

## 5. Unique Binary Patterns Analysis

All 5 detected patterns at epoch 5:
```
[1,0,0,1,1,0,0,1,1,1,1]  — Most common, component 1-2-5-6 inactive
[1,0,0,1,1,1,0,1,1,1,1]  — Component 5 activates
[1,0,0,1,1,1,1,1,1,1,1]  — Component 6 activates  
[1,0,1,1,1,1,1,1,1,1,1]  — Component 2 activates
[1,1,1,1,1,1,1,1,1,1,1]  — ALL active (late assembly)
```

Only 5 patterns out of 2048 possible. Ground truth has 20+ unique states per 1000 frames. The model is under-predicting state diversity — it only captures 5 of ~20 states present in the validation set.

## 6. Training Dynamics

PSR receives gradient on 25% of batches. At epoch 5 with 6580 batches/epoch:
- Total training batches: 6580 × 5 = 32,900
- Actual PSR gradient steps: 32,900 / 4 ≈ 8,225
- Effective PSR epochs: 8,225 / (6580/4) ≈ 5.0

So PSR has had ~5 effective epochs of training. Compare to detection which has trained since RF1 (20+ effective epochs). PSR at 5 epochs is showing reasonable progress.

**PSR warmup completed at epoch 3.** From epoch 3-5 (2 full epochs), binary accuracy rose from ~0.30 to 0.554. At this rate, comp acc should reach 0.60 by epoch 8-10 and 0.65 by epoch 15-20.

## 7. Path to Benchmarkable Results

| Requirement | Current | Target | When | How |
|---|---|---|---|---|
| Binary accuracy >0.65 | 0.554 | 0.65+ | Epoch 12-15 | More training |
| Transition F1 measurable | 0.0 (eval bug) | >0.05 | **Now** | Fix MonotonicDecoder |
| Unique patterns >100 | 5 | 100+ | Epoch 10-15 | More training |
| detach_psr_fpn=False | True | **False needed** | RF6+ | Flip flag for backbone adaptation |
| PSR_SEQ_EVERY_N_BATCHES | 4 | **2 possible** | RF6+ | More frequent PSR training |

**Critical dependency:** Fix MonotonicDecoder → get real transition F1 → decide if PSR is viable for paper.

## 8. PSR vs Published Baselines

| Baseline | PSR F1@±3 | Notes |
|---|---|---|
| B2 heuristic (IndustReal WACV 2024) | 0.731 | Rule-based, dense temporal features |
| STORM-PSR (CVIU 2025) | 0.901 | State of the art, dedicated model |
| **POPW current** | **Unknown (eval bug)** | Multi-task, isolated head |
| POPW projected with fix | 0.05-0.15 | At epoch 5, limited by isolation |
| POPW projected at convergence | 0.15-0.35 | With detach_psr_fpn=False at RF6+ |

POPW will not match B2/STORM PSR accuracy due to multi-task isolation tradeoff. The paper should frame PSR as "preliminary per-frame component recognition" not "transition detection."
