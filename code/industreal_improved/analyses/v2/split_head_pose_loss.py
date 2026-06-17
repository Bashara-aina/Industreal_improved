"""
split_head_pose_loss.py
=======================

Replaces the single 9-DoF MSE head-pose loss in MultiTaskLoss with a two-term
loss so that BOTH the position channels and the direction channels (forward/up
unit vectors) receive comparable gradient.

Current loss (losses.py ~1064):
    loss_head_pose = self.head_pose_loss_fn(outputs['head_pose'],
                                            targets['head_pose']) * 0.001   # MSE over all 9 DoF

Problem: position values are ~110 (NOT metres) while forward/up are unit vectors
(~1). A single MSE weights every channel equally, so the position squared-error
(~110**2) dominates the loss VALUE ~1e4x and the position gradient dominates the
direction gradient ~1e2x. forward/up never learn, never reach unit norm, and the
angular MAE prints 0.00.

THE FIX IS TWO COUPLED CHANGES
------------------------------
(A) Dataset: standardize the position target to O(1). This is also your unit
    fix -- it removes the mm/cm ambiguity entirely. In industreal_dataset.py
    _parse_pose, before `return pose_data`:

        pose_data[:, 3:6] /= C.HEAD_POSE_POS_SCALE   # standardize position to ~O(1)

    Pick HEAD_POSE_POS_SCALE so standardized position is ~O(1): with observed
    |pos|max ~110, use ~100. (If the mm/cm one-liner says cm with |pos|~10, use
    ~10.) The exact value is not critical -- it only needs to bring position into
    the same order of magnitude as the unit direction vectors.

(B) Loss: split into position MSE + direction MSE on L2-normalized vectors,
    weighted comparably. Because position is now O(1) (from A) and directions
    are normalized, plain 1:1 weights give balanced gradient -- no fragile 1/s**2.

Why not just standardize and keep one MSE? Standardizing position alone balances
magnitudes, but the direction term still needs L2-normalization so the head can
output non-unit vectors and still get correct *directional* gradient, and so the
optional norm regulariser can drive outputs to unit norm (which is what makes the
angular MAE metric well-defined instead of NaN->0.00).

Eval: replace the `* 1000` (assume-metres) position reporting with
`* HEAD_POSE_POS_SCALE` to report position error back in the CSV's native unit.

9-DoF layout (industreal_dataset.py:476): [0:3] forward  [3:6] position  [6:9] up

Run:  python split_head_pose_loss.py   # prints the gradient-balance proof
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Drop-in loss  (assumes position target standardized to ~O(1) -- change A)
# ---------------------------------------------------------------------------
def head_pose_loss_split(
    pred: torch.Tensor,          # [B, 9] forward[0:3] + position[3:6] + up[6:9]
    target: torch.Tensor,        # [B, 9] same layout; position standardized to ~O(1)
    pos_weight: float = 1.0,
    dir_weight: float = 1.0,
    norm_reg_weight: float = 0.0,   # optional: pull ||fwd||,||up|| toward 1
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Two-term head-pose loss. With standardized position targets, both terms are
    O(1) and pos_weight/dir_weight=1.0 give comparable gradient to all channels.

      position term : plain MSE on the (standardized) position channels
      direction term: MSE between L2-normalized predicted and target fwd/up
                      vectors -- scale-invariant, geometrically a chord distance
    """
    fwd_p, pos_p, up_p = pred[:, 0:3], pred[:, 3:6], pred[:, 6:9]
    fwd_t, pos_t, up_t = target[:, 0:3], target[:, 3:6], target[:, 6:9]

    # 1) Position term -- position target is already standardized to ~O(1).
    pos_loss = F.mse_loss(pos_p, pos_t)

    # 2) Direction term -- MSE on L2-normalized vectors (scale-invariant).
    fwd_pn = F.normalize(fwd_p, dim=1, eps=eps)
    up_pn = F.normalize(up_p, dim=1, eps=eps)
    fwd_tn = F.normalize(fwd_t, dim=1, eps=eps)
    up_tn = F.normalize(up_t, dim=1, eps=eps)
    dir_loss = F.mse_loss(fwd_pn, fwd_tn) + F.mse_loss(up_pn, up_tn)

    total = pos_weight * pos_loss + dir_weight * dir_loss

    # Optional: encourage raw predicted vectors toward unit norm so the angular
    # MAE metric (which needs unit vectors) is well-defined. Off by default.
    if norm_reg_weight > 0.0:
        fwd_norm = fwd_p.norm(dim=1)
        up_norm = up_p.norm(dim=1)
        norm_reg = ((fwd_norm - 1.0) ** 2 + (up_norm - 1.0) ** 2).mean()
        total = total + norm_reg_weight * norm_reg

    return total


# ---------------------------------------------------------------------------
# Exact replacement for the MultiTaskLoss head-pose block
# ---------------------------------------------------------------------------
# In MultiTaskLoss.__init__ add:
#     self.head_pose_pos_weight = float(getattr(C, 'HEAD_POSE_POS_WEIGHT', 1.0))
#     self.head_pose_dir_weight = float(getattr(C, 'HEAD_POSE_DIR_WEIGHT', 1.0))
#     self.head_pose_norm_reg   = float(getattr(C, 'HEAD_POSE_NORM_REG', 0.0))
#
# REPLACE the head-pose block (losses.py ~1062-1069) with:
#
#     if 'head_pose' in outputs and outputs['head_pose'] is not None:
#         loss_head_pose = head_pose_loss_split(
#             outputs['head_pose'], targets['head_pose'],
#             pos_weight=self.head_pose_pos_weight,
#             dir_weight=self.head_pose_dir_weight,
#             norm_reg_weight=self.head_pose_norm_reg,
#         )
#     else:
#         loss_head_pose = zero
#
# DROP the old `* 0.001`: the split loss is already O(1) and Kendall
# (exp(-log_var_pose)) handles task weighting; keeping 0.001 would re-starve it.
# HEAD_POSE_LOSS_CAP smooth-cap downstream still applies unchanged.
# Requires change (A): position target standardized in _parse_pose.


# ===========================================================================
# Self-test: prove both channel groups get comparable gradient
# ===========================================================================
def _per_group_grad_norms(loss: torch.Tensor, pred: torch.Tensor):
    if pred.grad is not None:
        pred.grad = None
    loss.backward(retain_graph=True)
    g = pred.grad
    return float(g[:, 0:3].norm()), float(g[:, 3:6].norm()), float(g[:, 6:9].norm())


if __name__ == "__main__":
    torch.manual_seed(0)
    B = 16
    POS_RAW_MAG = 110.0     # observed |pos|max ~110 in pose.csv (NOT metres)
    POS_SCALE = 100.0       # HEAD_POSE_POS_SCALE: standardizes raw position -> ~O(1)

    fwd_t = F.normalize(torch.randn(B, 3), dim=1)
    up_t = F.normalize(torch.randn(B, 3), dim=1)

    def fresh_pred():
        return torch.randn(B, 9, requires_grad=True)

    print("=" * 84)
    print("Head-pose loss: per-channel-group gradient balance")
    print("=" * 84)

    # --- OLD: raw position (~110) + single 9-DoF MSE * 0.001 ---
    pos_t_raw = torch.randn(B, 3) * POS_RAW_MAG
    target_raw = torch.cat([fwd_t, pos_t_raw, up_t], dim=1)
    p_old = fresh_pred()
    old_loss = F.mse_loss(p_old, target_raw) * 0.001
    gf, gp, gu = _per_group_grad_norms(old_loss, p_old)
    print("\nOLD  raw position (~110) + single MSE*0.001")
    print(f"  grad-norm  forward={gf:.3e}  position={gp:.3e}  up={gu:.3e}")
    print(f"  position / direction grad ratio = {gp / max((gf + gu) / 2, 1e-12):.1f}x"
          f"   <-- directions starved (and loss VALUE ratio is ~pos_scale^2 ~1e4x)")

    # --- NEW: standardized position (change A) + split loss (change B) ---
    pos_t_std = pos_t_raw / POS_SCALE            # change A: dataset standardization
    target_std = torch.cat([fwd_t, pos_t_std, up_t], dim=1)
    p_new = fresh_pred()
    new_loss = head_pose_loss_split(p_new, target_std, pos_weight=1.0, dir_weight=1.0)
    gf2, gp2, gu2 = _per_group_grad_norms(new_loss, p_new)
    print("\nNEW  standardized position + split loss (pos MSE + normalized-dir MSE)")
    print(f"  grad-norm  forward={gf2:.3e}  position={gp2:.3e}  up={gu2:.3e}")
    print(f"  position / direction grad ratio = {gp2 / max((gf2 + gu2) / 2, 1e-12):.2f}x"
          f"   <-- comparable; all groups learn")

    # --- NEW + optional norm regulariser (makes fwd/up unit-norm) ---
    p_reg = fresh_pred()
    reg_loss = head_pose_loss_split(p_reg, target_std, norm_reg_weight=0.1)
    gf3, gp3, gu3 = _per_group_grad_norms(reg_loss, p_reg)
    print("\nNEW + norm_reg=0.1 (drives predicted fwd/up to unit norm -> angular MAE defined)")
    print(f"  grad-norm  forward={gf3:.3e}  position={gp3:.3e}  up={gu3:.3e}")

    # sanity: tune pos_weight if you want exact parity
    print("\nTakeaway:")
    print(f"  OLD ratio ~{gp / max((gf + gu) / 2, 1e-12):.0f}x (position dominates gradient;")
    print("      loss-value dominance is ~pos_scale^2 ~1e4x).")
    print(f"  NEW ratio ~{gp2 / max((gf2 + gu2) / 2, 1e-12):.1f}x (balanced).")
    print("  If you keep position RAW (no change A), set HEAD_POSE_POS_WEIGHT ~ 1/pos_scale")
    print("  (~0.01) instead -- but standardizing in the dataset is cleaner and also")
    print("  resolves the mm/cm unit ambiguity.")
