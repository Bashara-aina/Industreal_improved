"""Test: Stage 3 transition PRESERVES Kendall log_var values (do NOT reset).

The old implementation (Bug #9 reincarnation prevention) reset log_var_act and
log_var_psr to 0.0 at Stage 3 entry, destroying learned uncertainty from Stage 2.
Now that log_var values are clamped before every forward (_clamp_kendall_log_vars,
see Task 1), the Stage 3 reset is unnecessary and harmful.

This test verifies:
  1. The helper _on_stage_transition exists and is importable from train.py
  2. Log_var values are preserved (not reset) when transitioning to Stage 3
  3. Stage 3 warmup is still activated correctly
"""
import torch
import pytest
from src.training.losses import MultiTaskLoss


def test_stage3_preserves_log_var_values():
    """When entering Stage 3, Kendall log_var_act and log_var_psr must be
    PRESERVED, not reset to 0.0. The old implementation reset them, destroying
    learned uncertainty from Stage 2."""
    crit = MultiTaskLoss(num_classes_act=75)

    # Simulate learned values after Stage 2 (different from init)
    with torch.no_grad():
        crit.log_var_act.data.fill_(-1.5)   # drifted from init 0.0
        crit.log_var_psr.data.fill_(-2.0)   # drifted from init 0.0
        crit.log_var_det.data.fill_(-0.5)   # should be unaffected
        crit.log_var_pose.data.fill_(-0.8)  # should be unaffected

    # Snapshot values before transition
    expected_act = crit.log_var_act.item()
    expected_psr = crit.log_var_psr.item()
    expected_det = crit.log_var_det.item()
    expected_pose = crit.log_var_pose.item()

    # Call the helper — this will fail at import if _on_stage_transition
    # doesn't exist yet (TDD: write failing test first).
    from src.training.train import _on_stage_transition

    warmup_state = {
        'active': False,
        'warmup_epochs': 3,
        'start_epoch': 0,
        'epochs_remaining': 0,
        'param_group_idx': 2,
    }
    _on_stage_transition(
        model=None,
        criterion=crit,
        current_stage=3,
        epoch=16,
        backbone_type='convnext_tiny',
        stage3_warmup_state=warmup_state,
    )

    # Assert ALL log_var values are preserved (not reset)
    assert crit.log_var_act.item() == pytest.approx(expected_act), \
        f"log_var_act should preserve Stage 2 value {expected_act}, got {crit.log_var_act.item()}"
    assert crit.log_var_psr.item() == pytest.approx(expected_psr), \
        f"log_var_psr should preserve Stage 2 value {expected_psr}, got {crit.log_var_psr.item()}"
    assert crit.log_var_det.item() == pytest.approx(expected_det), \
        "log_var_det must not be affected by Stage 3 transition"
    assert crit.log_var_pose.item() == pytest.approx(expected_pose), \
        "log_var_pose must not be affected by Stage 3 transition"

    # Verify warmup was activated
    assert warmup_state['active'] is True, \
        "Stage 3 warmup should be activated by transition helper"
    assert warmup_state['start_epoch'] == 16, \
        "warmup start_epoch should match transition epoch"
    assert warmup_state['epochs_remaining'] == 3, \
        "warmup epochs_remaining should equal warmup_epochs"
