# PSRHead Repair v2 Design (Opus §-1c/d, §Q8)

## Diagnostic Results (10 batches, seq_len=16, T=1, val split)

### Transformer Output (Upstream Variance)
- **encoded.std() = 35.5 +/- 1.5** (range 33.5-37.0 across 10 batches)
- **Verdict: HEALTHY.** The causal transformer is producing high-variance features.
- The suspected "transformer variance collapse" (Opus §-1c) is **NOT present**.

### Per-Component GELU Activation
| Component | Pre-GELU mean | Pre-GELU std | Post-GELU zero_frac | Status |
|-----------|--------------|-------------|--------------------|--------|
| comp[00]  | -129.9       | 38.5        | 0.9920             | DEAD   |
| comp[01]  | -125.0       | 49.4        | 0.9920             | DEAD   |
| comp[02]  | -130.9       | 46.9        | 0.9886             | DEAD   |
| comp[03]  | -121.0       | 53.1        | 0.9753             | DEAD   |
| comp[04]  | -128.7       | 57.6        | 0.9844             | DEAD   |
| comp[05]  | -137.2       | 49.6        | 0.9997             | DEAD   |
| comp[06]  | -143.2       | 49.4        | 0.9966             | DEAD   |
| comp[07]  | -119.4       | 59.1        | 0.9688             | DEAD   |
| comp[08]  | -122.2       | 53.3        | 0.9844             | DEAD   |
| comp[09]  | -113.8       | 38.0        | 0.9844             | DEAD   |
| comp[10]  | -123.4       | 53.3        | 0.9682             | DEAD   |

### Final Logits (after Linear(64,1))
- Logits std across all batches: **0.0001-0.07** — effectively constant.
- PSR F1 of ~0.75 is achieved despite near-constant logits because the per-component threshold calibration absorbs the bias offset. The logits encode zero frame-to-frame signal.

## Root Cause

The GELU activation in `Linear(256,64) -> GELU -> Dropout -> Linear(64,1)` is **fully saturated** due to massive negative pre-activations:

1. Transformer output has healthy variance (std ~35.5) and mean ~0
2. `Linear(256,64)` weights project this into pre-activations with **mean ≈ -130, std ≈ 50**
3. GELU(x) is asymptotically zero for x < -3. With mean=-130, std=50, ~99.5% of values are in GELU's dead zone
4. The existing `+0.1 bias` fix (line 1611) is **insufficient by ~1300x** against bias shifts of this magnitude
5. Post-GELU ≈ 0 for all 11 heads → final `Linear(64,1)` produces near-constant logits

This is NOT a transformer variance problem. It is a **GELU saturation problem caused by large negative pre-activations** from the default Linear weight init (Kaiming uniform, which produces weights with std ~ sqrt(2/fan_in) ≈ 0.09 for 256->64). When these weights multiply a transformer output with std ~35, the resulting pre-activations have absurdly large magnitude.

## Proposed Repair: LeakyReLU + Zero Bias + Smaller Weight Init

Replace the GELU in each output head with LeakyReLU(0.01) and zero the first-layer bias:

```python
self.output_heads = nn.ModuleList([
    nn.Sequential(
        nn.Linear(gru_hidden, 64),
        nn.LeakyReLU(0.01),        # Was: GELU — GELU saturates for large negative inputs
        nn.Dropout(dropout * 0.3),
        nn.Linear(64, 1),
    ) for _ in range(num_components)
])
```

And change the init to use small-normal weights instead of the +0.1 bias hack:

```python
# Remove: nn.init.constant_(head[0].bias, 0.1)
# Replace with proper variance-controlled init:
for head in self.output_heads:
    if isinstance(head[0], nn.Linear):
        nn.init.normal_(head[0].weight, std=0.01)  # Small init prevents GELU saturation
        nn.init.zeros_(head[0].bias)
```

**Why LeakyReLU works:**
- Gradient is always 1 for x > 0 and 0.01 for x < 0 — never zero
- Even with mean=-130, the gradient flows through at slope 0.01
- Training will shift the weights to center activations near zero over time

**Alternative considered — Keep GELU + LayerNorm:**
Adding `LayerNorm(64)` after `Linear(256,64)` would normalize pre-GELU activations to mean=0, std=1, preventing saturation. However, this adds parameters, introduces a learnable affine transform that the per-component heads don't need, and breaks the "tiny MLP" design intent. LeakyReLU is simpler.

## Implementation Plan

1. In `model.py:1597-1604`: Replace `nn.GELU()` with `nn.LeakyReLU(0.01)` in `output_heads`
2. In `model.py:1609-1611`: Remove the `+0.1 bias` init, replace with `nn.init.normal_(weight, std=0.01)`
3. Warm-start from the current rf_stages checkpoint (the heads were never training anyway, so a reset costs nothing)
4. Run training for 25 epochs on RTX 5060 Ti

Estimated impact: +0.03-0.08 F1 (from actual gradient flow into output heads; previously the only training signal came through the +0.1 bias trick allowing ~0.5% of activations through GELU).
