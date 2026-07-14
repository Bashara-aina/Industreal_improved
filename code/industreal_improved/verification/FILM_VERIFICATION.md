# FiLM Wiring Verification

## Module Definitions

### PoseFiLMModule
- **File**: `src/models/model.py:733-823`
- **Architecture**: keypoints [B,17,2] + confidence [B,17] -> gamma_net / beta_net (51->512->768)
- **Param count**: 0.84M (core, excl. optional 1x1 C5 projection)
- **Config flag**: `USE_HAND_FILM` (`src/config.py:162`, default `True`)

### HeadPoseFiLMModule
- **File**: `src/models/model.py:829-899`
- **Architecture**: 9-DoF head pose -> gamma_net / beta_net (9->256->768, with LayerNorm + GELU)
- **Param count**: 0.40M
- **Config flag**: `USE_HEADPOSE_FILM` (`src/config.py:190`, default `True`)

## Forward Pass Wiring

### PoseFiLM
- **Instantiation**: `model.py:1956-1961` — `PoseFiLMModule` when `use_hand_film=True`
- **Forward call**: `model.py:2175-2179`
  ```
  c5_mod = self.pose_film(c5, keypoints.detach(), pose_confidence)
  ```
- **Keypoint provenance**: From `pose_head` soft-argmax (line 2111), or pseudo-keypoints from detection bboxes when `train_pose=False` (lines 2117-2172)

### HeadPoseFiLM
- **Instantiation**: `model.py:1964-1968` — `HeadPoseFiLMModule` when `use_headpose_film=True`
- **Forward call**: `model.py:2300-2301`
  ```
  c5_mod = self.headpose_film(c5_mod, head_pose.detach())
  ```
- **Head pose provenance**: From `self.head_pose_head(c4, c5)` (line 2293), converted to [B,9] via `to_legacy_9dof` (line 2299)

## Stop-Gradient Verification

| Location | Mechanism | Purpose |
|----------|-----------|---------|
| `model.py:2179` | `keypoints.detach()` | Prevents activity gradients from flowing back through PoseFiLM -> keypoints -> pose_head -> FPN |
| `model.py:803` | `confidence.detach()` | Inside `PoseFiLMModule.forward` — confidence detached at input |
| `model.py:2301` | `head_pose.detach()` | Prevents activity gradients from flowing back through HeadPoseFiLM -> head_pose_head |
| `model.py:2313` | gradient blend | `blend * c5_mod + (1-blend) * c5_mod.detach()` — small gradient leaks into FiLM heads per paper SS5.4 |

## Parameter Counts (empirically verified)

| Module | Expected | Actual | Status |
|--------|----------|--------|--------|
| PoseFiLMModule (core) | 0.84M | 841,216 (0.84M) | CORRECT |
| HeadPoseFiLMModule | 0.40M | 400,896 (0.40M) | CORRECT |

Both match PAPER_OUTLINE SS3.1 efficiency table claims.

## Data Dependencies

- **PoseFiLM keypoints**: Real hand keypoints from `pose_head` soft-argmax when `train_pose=True`; pseudo-keypoints from detection bboxes when `train_pose=False`. Hand keypoints come from hands.csv annotations in the dataset.
- **HeadPoseFiLM head pose**: 9-DoF (forward[3], position[3], up[3]) from `head_pose_head(c4, c5)` — geometry-aware head pose regressor. Ground truth from pose.csv.
- **FiLM conditioning uses real data** in both modules — not dummy/zero inputs.

## Summary

All FiLM wiring confirmed. PoseFiLM operates on real hand keypoints, HeadPoseFiLM on real head pose. Both have proper stop_gradient isolation. Parameter counts match paper specifications. Forward pass ordering: PoseFiLM modulates C5 -> HeadPoseFiLM modulates c5_mod -> blended c5_mod_2 feeds activity head.
