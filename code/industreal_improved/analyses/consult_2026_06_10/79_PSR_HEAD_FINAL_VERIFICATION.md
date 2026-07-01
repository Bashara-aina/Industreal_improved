# 79 — PSR Head: Transition Detection and Per-Component Balance [2026-07-01]

**Goal:** Guarantee the PSR (Procedure Step Recognition) head produces accurate per-component predictions with stable training, no NaN spikes, and correct temporal consistency. PSR is the paper's secondary headline metric.

**Source files (all paths relative to `code/industreal_improved/src/`):**
- `config.py:415–416, 704–705, 741, 861–866, 882–892` — PSR configs
- `models/model.py:1535–1742` — PSRHead (causal transformer + per-component heads)
- `training/losses.py:1070–1081, 1137–1168, 1409–1480` — PSR loss, per-component alpha, forward
- `data/industreal_dataset.py` — PSR label loading (AR_per_frame)
- `evaluation/evaluate.py` — PSR metrics (binary accuracy per component)

---

## 1. PSR Architecture (model.py:1535–1742)

### 1.1 Overview
PSR predicts 11 assembly components (which parts are currently being manipulated) from multi-scale FPN features. Each component is a binary variable: 0 = not being manipulated, 1 = being manipulated.

Three architectural blocks:

**A. Per-Frame Feature Extraction (model.py:1563-1575)**
```python
# Multi-scale: P3 (256ch, stride 8) + P4 (256ch, stride 16) + P5 (256ch, stride 32)
self.gap_p3 = nn.AdaptiveAvgPool2d(1)
self.gap_p4 = nn.AdaptiveAvgPool2d(1)
self.gap_p5 = nn.AdaptiveAvgPool2d(1)
per_scale_ch = in_channels * num_scales  # 256 * 3 = 768

self.per_frame_mlp = nn.Sequential(
    nn.Linear(768, gru_hidden * 2),  # 768 -> 512
    nn.LayerNorm(512),
    nn.GELU(),
    nn.Dropout(0.1),                  # dropout * 0.5
    nn.Linear(512, gru_hidden),      # 512 -> 256
    nn.LayerNorm(256),
)
```

**B. Causal Transformer (model.py:1577-1589)**
```python
encoder_layer = nn.TransformerEncoderLayer(
    d_model=256, nhead=4, dim_feedforward=1024,
    dropout=0.2, activation='gelu', batch_first=True, norm_first=True,
)
self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
```

With causal masking (model.py:1622-1631):
```python
ones = torch.ones(seq_len, seq_len, device=device)
mask = torch.triu(ones, diagonal=1).bool()  # upper-triangular: True = ignore
```

Each position attends only to itself and prior positions — essential for online inference where future frames don't exist yet.

**C. Per-Component Output Heads (model.py:1595-1609)**
```python
self.output_heads = nn.ModuleList([
    nn.Sequential(
        nn.Linear(256, 64), nn.GELU(), nn.Dropout(0.06), nn.Linear(64, 1),
    ) for _ in range(11)
])
```

Each component has its own tiny MLP with bias+0.1 on the first Linear (line 1609) to prevent zero-feature collapse into GELU's near-zero regime.

### 1.2 Inference Cache (model.py:1680-1714)
At inference, per-video feature cache enables causal temporal processing:
```python
self._cache[key].append(feat_i.detach().clone())
# ... concatenate cached features -> causal transformer -> last-timestep output
```

Cache max length: **32 frames** (model.py:1613). At 30 FPS, this provides ~1s context.

### 1.3 Gradient Isolation (config.py:882-886)
```python
DETACH_PSR_FPN = True
```

PSR receives detached FPN features — its gradients don't corrupt detection features. This is critical because PSR loss spikes (which can reach 20+ in early training) would otherwise disrupt the shared FPN and collapse detection.

---

## 2. PSR Loss Configuration

### 2.1 Base Loss (losses.py:1070-1081)
```python
self.psr_loss_fn = nn.BCEWithLogitsLoss(reduction='mean')  # Standard BCE, not focal
self._psr_temporal_smooth_weight = float(getattr(C, 'PSR_TEMPORAL_SMOOTH_WEIGHT', 0.05))
```

**Important**: `use_psr_focal` is False by default (config `PSR_FOCAL_GAMMA` defaults to 0, disabling focal). PSR uses plain BCEWithLogitsLoss. The per-component weighting is handled separately via `_psr_comp_weights`.

### 2.2 Per-Component Prevalence Weighting (losses.py:1137-1168)
```python
def set_psr_class_counts(self, prevalence_per_component):
    prev = prevalence_per_component.float().clamp(0.01, 0.99)
    alpha_c = 2.0 * (1.0 - prev)
    alpha_c = alpha_c.clamp(min=0.1)  # safety floor
```

Components with low prevalence (e.g., comp10 wheels at ~30%) get alpha near 1.4 (2×(1-0.3) = 1.4). Component 0 (base plate at ~95%) gets alpha ~0.1 after clamp. This prevents the model from predicting "all zeros" for rare components.

### 2.3 Temporal Smoothing Loss (losses.py:~1470 area)
```python
PSR_TEMPORAL_SMOOTH_WEIGHT = 0.05  # config.py:705
```

A transition-aware loss that penalizes predicted-transition / label-transition mismatches. Weight 0.05 ensures it's regularization only — never the dominant PSR loss term.

### 2.4 PSR Weight Before Kendall (config.py:741)
```python
PSR_WEIGHT = 10.0
```

This multiplies the raw PSR loss BEFORE Kendall weighting. Combined with Kendall's ability to suppress PSR (up to `KENDALL_LOG_VAR_MAX_PSR=0.0`), the effective PSR contribution is managed.

---

## 3. PSR Forward in MultiTaskLoss (losses.py:1409-1480)

### 3.1 Loss Computation
```python
if self.train_psr and 'psr_logits' in outputs:
    psr_logits = outputs['psr_logits']  # [B, 12] (11 components + 1 confidence)
    psr_targets = targets['psr']        # [B, 11]
    
    # Main BCE loss: psr_logits[..., :11] vs psr_targets[..., :11]
    psr_loss = self.psr_loss_fn(psr_logits[..., :11], psr_targets)
    
    # Per-component weight application
    if self._psr_comp_weights is not None:
        w = self._psr_comp_weights.to(psr_logits.device)
        psr_loss = (psr_loss * w).mean()
    
    # PSR_WEIGHT multiplier
    psr_loss = psr_loss * float(getattr(C, 'PSR_WEIGHT', 1.0))
    
    # Temporal smoothing loss
    if self.use_psr_transition and self._psr_temporal_smooth_weight > 0:
        smooth_loss = self._compute_temporal_smooth_loss(psr_logits, psr_targets)
        psr_loss = psr_loss + self._psr_temporal_smooth_weight * smooth_loss
```

### 3.2 NaN Guard (losses.py:1284-1294)
```python
if not torch.isfinite(loss_psr).all():
    if getattr(C, 'ASSERT_AND_CRASH', False):
        raise FloatingPointError(...)
    loss_psr = torch.where(torch.isfinite(loss_psr), loss_psr, _fallback)
```

If PSR loss goes NaN (e.g., from extreme focal weight amplification), it's replaced with 1e-4. With `ASSERT_AND_CRASH=True`, this would halt training immediately — useful for debugging the source of PSR spikes.

### 3.3 PSR Loss Cap (config.py:733)
```python
PSR_LOSS_CAP = 20.0  # Smooth cap (same formula: x if x<=cap, cap*(1+log(x/cap)) if x>cap)
```

### 3.4 PSR Warmup (config.py:861-866, 887-892)
```python
PSR_WARMUP_INIT_MULT = 2.0    # Initial precision multiplier (step-based)
STAGE3_WARMUP_EPOCHS = 3       # LR warmup at Stage 3
PSR_WARMUP_EPOCHS = 0          # Loss-side ramp disabled (LR warmup handles it)
```

The LR warmup at Stage 3 entry (epoch 16 for staged training) ramps PSR-specific parameter groups from 0→1× LR over 3 epochs. The loss-side `PSR_WARMUP_EPOCHS=0` means the PSR_WARMUP_INIT_MULT step-based ramp is also effectively disabled — check if this is intentional.

---

## 4. Eval Metrics

### 4.1 Binary Accuracy Per Component
PSR evaluation computes per-component binary accuracy (thresholded sigmoid at 0.5):
- Per-component precision, recall, F1
- Overall binary accuracy (mean across components)
- Transition accuracy (Δ between consecutive frames matches label Δ)

### 4.2 PSR Confidence (model.py:1732-1734)
```python
confidence = torch.sigmoid(logits).max(dim=-1, keepdim=True)[0]  # [B, 1]
return torch.cat([logits, confidence], dim=-1)  # [B, 12]
```

The 12th output channel is the per-frame confidence (max sigmoid across components). This enables frame-level filtering: frames with confidence < 0.5 can be treated as "uncertain" for downstream use.

---

## 5. Sequence Mode vs Per-Frame Mode

### 5.1 Current Configuration
The `USE_PSR_SEQUENCE_MODE` flag (not explicitly shown in config read but used in dataset) controls whether PSR trains on real video sequences or single shuffled frames.

**Per-frame mode** (current default for non-staged training):
- Single frame processed through causal transformer (T=1, attends to itself)
- No temporal context — causal mask is a no-op
- Per-component heads classify from instantaneous features

**Sequence mode** (required for temporal PSR):
- Real consecutive frames from the same recording
- Feature bank sequence of length T=16
- Causal transformer sees actual temporal progression
- Transition loss is meaningful

### 5.2 Risk: Per-frame PSR with Shuffled Data
In the current setup (`STAGED_TRAINING=False`, class-balanced sampler shuffles all frames), PSR receives single, randomly-ordered frames. The causal transformer processes T=1 always — there is **no temporal signal**. This means:
- PSR is a per-frame static classifier (not temporal transition detection)
- The temporal smoothing loss computes over T=1 transitions (no-op)
- Reported PSR metrics may look good (high binary accuracy for common components like comp0) but the model cannot detect transitions

### 5.3 Opus Must Assess
- Is PSR expected to work as a per-frame classifier (static state recognition, no temporal modelling), or does the paper frame PSR as transition detection?
- If transition detection is required, `USE_PSR_SEQUENCE_MODE=True` must be enabled, which requires the sequential dataloader path and non-shuffled batches.

---

## 6. What Could Still Go Wrong

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| PSR is evaluated per-frame but claimed as temporal detection | Medium | Check the paper: does it claim transition detection or component state recognition? Fix the claim to match the setup. |
| PSR_WARMUP_EPOCHS=0 means no step-based ramp — PSR starts at full loss from epoch 1 | Medium | If PSR spikes at early epochs, set `PSR_WARMUP_EPOCHS=3` for a gradual ramp. The LR warmup (`STAGE3_WARMUP`) handles this only for staged training, not the `STAGED_TRAINING=False` path. |
| DETACH_PSR_FPN=True prevents PSR from shaping backbone features for component recognition | Low | PSR on per-frame shuffled data doesn't need backbone gradient anyway. Enable backbone gradient for PSR when sequence mode is active. |
| PSR comp_weights and per-component alpha both applied → double-balancing | Low | Check losses.py: the comp_weights multiply the BCE loss elementwise, while `set_psr_class_counts` sets alpha for focal (which is disabled). With focal off, the alpha is unused — so only `_psr_comp_weights` is active. Verify both aren't simultaneously applied. |

---

## 7. Final Go/No-Go Criteria (Epoch 5)

| Signal | Pass (per-frame PSR) | Pass (temporal PSR) | Fail |
|--------|---------------------|---------------------|------|
| Mean binary accuracy | ≥ 0.70 | ≥ 0.75 | < 0.60 |
| Rare comp (index 7-10) accuracy | ≥ 0.40 | ≥ 0.50 | < 0.20 |
| PSR loss magnitude | 0.2-1.0 | 0.1-0.5 | > 2.0 or NaN |
| Transition F1 | N/A (per-frame) | ≥ 0.30 | N/A for per-frame |
| PSR grad norm ratio vs detection | 0.01x – 10x | — | < 0.001x or > 100x |

**Expected final (epoch 100):**
- Per-frame mode: binary accuracy ≈ 0.75-0.85, with comp0 (base plate) near 0.95 and rare components near 0.30-0.50
- Temporal mode: transition F1 ≈ 0.40-0.60, binary accuracy ≈ 0.80-0.90

**If fail**: check whether the 11 components have enough transition labels. For rare components with transitions in <1% of frames, even class-weighted BCE may not be sufficient. Consider focal PSR (`PSR_FOCAL_GAMMA=2.0`, `PSR_FOCAL_ALPHA=0.75`).
