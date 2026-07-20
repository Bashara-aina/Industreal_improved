# Imp4: UW-SO Multi-Task Loss Balancing

## Summary

Replaced manual fixed-weight loss balancing (`det=1, act=0.5, pose=0.1, psr=0.5`) with learnable uncertainty-weighted loss balancing using the homoscedastic uncertainty formulation from Kendall et al. (2018).

## Implementation

### New File: `src/losses/uw_so.py`

Defines `class UWSOLoss(nn.Module)` with learnable `log_sigma` parameters (one per task):

- **Formula**: `L_total = sum_i (exp(-2 * log_sigma_i) * L_i + log_sigma_i)`
- **Init**: `log_sigma = 0.0` for all tasks, so initial weight = `exp(-2 * 0) = 1.0`
- **Tasks**: `det`, `act`, `pose`, `psr`
- **`sigma` property**: returns `exp(log_sigma.detach())` for monitoring convergence

Also preserves the static `uw_so_loss()` function for baseline comparisons.

### Integration: `train_mtl_v3.py`

- **CLI**: `--use-uw-so` flag (default: False)
- **Loss function**: `multi_task_loss_v3()` accepts optional `uw_so` kwarg; when active, collects raw losses and passes to `UWSOLoss.forward()`; when inactive, uses existing fixed weights
- **Optimizer**: UW-SO params in separate param group with `weight_decay=0.0`
- **Checkpoint**: Saves `uw_so_state_dict` alongside model and optimizer states
- **Logging**: Displays `uw_sig=[det, act, pose, psr]` in both Phase 1 and Phase 2 batch logs

## Usage

```bash
# Train with learnable uncertainty weighting
python train_mtl_v3.py --use-uw-so

# Retain original fixed-weight behavior (default)
python train_mtl_v3.py
```

## Files Changed

| File | Change |
|------|--------|
| `src/losses/uw_so.py` | NEW: `UWSOLoss` class + static `uw_so_loss` |
| `src/losses/__init__.py` | Added `UWSOLoss`, `uw_so_loss` exports |
| `train_mtl_v3.py` | CLI flag, loss integration, optimizer, checkpoint, logging |

## Verification

Smoke test passes:
- Model loads with UW-SO enabled (4 log_sigma params)
- Optimizer correctly segregates UW-SO params in own group
- Forward + backward pass executes without error
- Phase 1 and Phase 2 logging display sigma values
