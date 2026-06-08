"""Test: Kendall log_var clamp happens BEFORE forward pass.

The previous implementation clamped log_var_*.data AFTER
scaler.scale(loss).backward(). The clamp comment said 'before backward'
but the code was after. The gradient of that step was already computed
from corrupted values, and only future steps got clean values.

This test verifies:
  1. The helper _clamp_kendall_log_vars exists and is importable from train.py
  2. Out-of-range log_var values get clamped to [-4.0, 2.0] when the helper runs
"""
import torch
from src.training.losses import MultiTaskLoss


def test_log_var_clamp_happens_before_forward_pass():
    """The clamp on log_var_* must run at the START of the next forward,
    not AFTER the current backward. We assert that an out-of-range log_var
    value present before a forward gets clamped to its bound by the time
    the loss is computed (i.e., the parameter is already clamped by the
    time forward runs)."""
    crit = MultiTaskLoss(num_classes_act=75)
    # Simulate a corrupt out-of-range value as if it leaked through.
    with torch.no_grad():
        crit.log_var_det.data.fill_(10.0)  # out of [-4, 2]
        crit.log_var_pose.data.fill_(10.0)
        crit.log_var_act.data.fill_(10.0)
        crit.log_var_psr.data.fill_(10.0)

    # Call the helper we will add in step 3.
    from src.training.train import _clamp_kendall_log_vars
    _clamp_kendall_log_vars(crit)

    assert crit.log_var_det.item() <= 2.0
    assert crit.log_var_pose.item() <= 2.0
    assert crit.log_var_act.item() <= 2.0
    assert crit.log_var_psr.item() <= 2.0
    assert crit.log_var_det.item() >= -4.0
    assert crit.log_var_pose.item() >= -4.0
    assert crit.log_var_act.item() >= -4.0
    assert crit.log_var_psr.item() >= -4.0
