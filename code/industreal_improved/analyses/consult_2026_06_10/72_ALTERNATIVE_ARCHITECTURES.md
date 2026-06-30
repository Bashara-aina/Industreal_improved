# 72: Alternative Activity Head Architectures [2026-07-01]

## Current Architecture

```
Input: proj_feat [B, 512]  (joint projection: det_conf ⊕ GAP(c5) ⊕ GAP(p4))
  → LayerNorm(512)
  → Linear(512 → 256)
  → GELU
  → Dropout(0.2 → now 0.3)
  → Linear(256 → 75)
  → act_logits [B, 75]
Params: ~150K
```

## Alternative 1: Deeper MLP (Q4)

```
proj_feat [B, 512]
  → LayerNorm(512)
  → Linear(512 → 512)     ← wider hidden
  → BatchNorm1d(512)       ← supports gradient flow
  → GELU
  → Dropout(0.3)
  → Linear(512 → 256)      ← second hidden
  → GELU
  → Dropout(0.2)
  → Linear(256 → 75)
Params: ~420K
```

**Rationale:** More capacity. The current 512→256→75 has a 8:1 bottleneck ratio.
A 512→512→256→75 is more gradual. BatchNorm may help stabilize.

**Risk:** 420K params on 3.7k frames = 8.8× params/data ratio. Overfitting risk.

**Implementation cost:** 15 min to modify model.py and test forward pass.

## Alternative 2: Residual MLP (Q4 variant)

```
proj_feat [B, 512]
  → LayerNorm(512)
  → Linear(512 → 512)
  → GELU
  → Dropout(0.3)
  → Linear(512 → 512)       ← residual block
  → + proj_feat (skip connection)
  → LayerNorm(512)
  → GELU
  → Linear(512 → 75)
```

**Rationale:** Skip connection preserves gradient flow directly from output to
input. The residual path can learn class-specific features while the skip
connection maintains the pre-trained feature distribution.

**Risk:** Residual connections on a 150K head may be over-engineered.

**Implementation cost:** 20 min.

## Alternative 3: Wider Single Layer (Q5)

```
proj_feat [B, 512]
  → LayerNorm(512)
  → Linear(512 → 512)
  → GELU
  → Dropout(0.3)
  → Linear(512 → 75)
Params: ~320K
```

**Rationale:** The 512→256 bottleneck may be discarding information. Keeping
512-dim throughout preserves the joint projection's full representation.

**Risk:** Minimal. 320K params is still tiny.

**Implementation cost:** 5 min.

## Alternative 4: Lightweight Temporal MLP (Q6)

```
Input: 3 consecutive frames' proj_feat [B, 3, 512]  (stacked, no feature bank)
  → Conv1d(in=512, out=256, kernel=3, padding=1)
  → GELU
  → AdaptiveAvgPool1d(1) → squeeze to [B, 256]
  → Linear(256 → 75)
Params: ~420K
```

**Key insight:** This does NOT use the feature bank or any recording_id ordering.
It simply stacks 3 consecutive frames from the batch. Since the DataLoader with
WeightedRandomSampler shuffles frames, consecutive batch items are NOT temporally
adjacent — but this approach uses batch dimension as "time," which is meaningless.

**Verdict:** This will NOT work with shuffled batches. Only useful if we implement
a per-recording sequential sampler for activity.

## Alternative 5: Per-Recording Sequential Sampler (Q6 extended)

**Key finding from Opus (file 63):** The feature bank + shuffled sampler = non-temporal
sequences. The fix is NOT architecture — it's the DATA ORDER.

**Design:**
1. Create `ActivitySequentialBatchSampler` that yields batches of consecutive frames
   from the same recording_id, without replacement.
2. Activity head only: feed these sequential batches through a 1D conv with kernel=3.
3. Detection/pose/PSR continue using the shuffled DataLoader as before.

**Challenge:** Multi-task DataLoader serves all 5 heads. One loader cannot be both
shuffled and sequential. Solution: Two loaders — one shuffled (for det/pose/PSR),
one sequential (for activity). Aggregate batches with `zip()`.

**Implementation cost:** ~2 hours. Requires significant DataLoader refactor.

## Alternative 6: Drop Activity (Paper §5 Case Study)

**If all architecture alternates fail**, the fallback is NOT an architecture — it's
a paper decision. Activity becomes a documented failure in §5:

> "The activity recognition head collapsed to predicting 2/74 classes despite
> switching from CE+label-smoothing to class-balanced focal loss (γ=2.0, β=0.999)
> and increasing regularization. We attribute this to 46/72 classes having <1%
> annotation support among 3,667 training frames — a data constraint that no
> tested loss function or architecture could overcome."

This is a valid negative result for AAIML (training pathology documentation).

## Recommendation Priority for Opus

We need Opus to rank these by probability of success:

1. Deeper/wider MLP (Alts 1-3)
2. Higher loss weight + higher γ (config change, no code)
3. Sequential sampler for activity (Alt 5 — highest potential, highest cost)
4. Drop activity (Alt 6 — fallback)
