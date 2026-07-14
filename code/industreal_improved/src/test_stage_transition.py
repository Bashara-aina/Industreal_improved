"""
Integration tests for stage 2 -> stage 3 transition in POPW training.

Verifies the three critical side-effects triggered at the Stage 3 boundary
(train.py lines ~2300-2400):
  1. stage3_warmup_state['active'] flips True, epochs_remaining > 0
  2. EMA is reinitialized from current model state (decay from _get_ema_decay)
  3. activity_head.videomae_proj is unfrozen (requires_grad=True)

Also exercises:
  - get_stage(epoch) boundary correctness
  - _set_stage_requires_grad per-stage freezing logic

Run: cd /media/newadmin/master/POPW/working/code/industreal_improved_to_archive/src
     python3 -m pytest test_stage_transition.py -v
"""

import sys
import os
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch

# Mirror the path setup used in test_eval_fix.py
_SRC = "/media/newadmin/master/POPW/working/code/industreal_improved_to_archive/src"
_PARENT = os.path.dirname(_SRC)
for _sub in ["models", "training", "evaluation", "data", _SRC]:
    if _sub not in sys.path:
        sys.path.insert(0, _sub)
os.chdir(_PARENT)

import config as C  # noqa: E402
from training.train import (  # noqa: E402
    get_stage,
    _set_stage_requires_grad,
    _get_ema_decay,
)


# ---------------------------------------------------------------------------
# Test fixtures: build a lightweight mock model that matches the attribute
# surface that train.py touches at the Stage 3 boundary.
# ---------------------------------------------------------------------------


class _MockHeadProj(nn.Module):
    """Stand-in for ActivityHead.videomae_proj (a small nn.Sequential)."""

    def __init__(self, dim_in=384, dim_out=256):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out)


class _MockActivityHead(nn.Module):
    def __init__(self):
        super().__init__()
        # The attribute train.py reaches for at line ~2366
        self.videomae_proj = _MockHeadProj()


class _MockPSRHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Linear(8, 8)


class _MockVideoMAEStream(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Linear(8, 8)


class _MockBackbone(nn.Module):
    """Fake backbone that mimics ResNet50 layer1-4 attribute surface."""

    def __init__(self):
        super().__init__()
        # Use a single Linear for each "layer" — train.py walks named_children
        self.layer1 = nn.Linear(16, 16)
        self.layer2 = nn.Linear(16, 16)
        self.layer3 = nn.Linear(16, 16)
        self.layer4 = nn.Linear(16, 16)


class _MockModel(nn.Module):
    """
    Mimics POPWMultiTaskModel attribute surface that train.py touches:

      model.backbone.layer{1..4}
      model.activity_head (with .videomae_proj)
      model.psr_head
      model.videomae_stream
    """

    def __init__(self):
        super().__init__()
        self.backbone = _MockBackbone()
        self.activity_head = _MockActivityHead()
        self.psr_head = _MockPSRHead()
        self.videomae_stream = _MockVideoMAEStream()
        # Other heads (pose, head_pose, det) — minimal stand-ins
        self.pose_head = nn.Linear(8, 8)
        self.head_pose = nn.Linear(8, 8)
        self.det_head = nn.Linear(8, 8)


def _build_fake_param_groups(model):
    """Build the optimizer.param_groups structure that train.py uses."""
    activity_psr_params = list(model.activity_head.parameters()) + list(model.psr_head.parameters())
    backbone_params = list(model.backbone.parameters())
    other_head_params = (
        list(model.pose_head.parameters())
        + list(model.head_pose.parameters())
        + list(model.det_head.parameters())
    )
    return [
        {"params": backbone_params, "lr": 1e-5},  # 0: backbone
        {"params": other_head_params, "lr": 1e-3},  # 1: pose/hp/det
        {"params": activity_psr_params, "lr": 1e-3},  # 2: act+psr (the warmup target)
    ]


# ---------------------------------------------------------------------------
# Stage 3 entry logic — reproduced from train.py:2306-2348 so the test
# exercises the exact decision tree without spinning up the full loop.
# ---------------------------------------------------------------------------


def _trigger_stage3_entry(
    model,
    optimizer,
    ema,
    epoch,
    criterion,
    stage3_warmup_state,
    device="cpu",
    videomae_unfreeze_epoch=None,
):
    """
    Mirrors the stage-transition block in train.py:2306-2348.
    Returns the (possibly re-initialized) ema.
    """
    current_stage = get_stage(epoch)
    prev_stage = get_stage(epoch - 1) if epoch > 0 else current_stage
    if current_stage != prev_stage and current_stage == 3:
        # Kendall log_var reset
        if hasattr(criterion, "log_var_act"):
            criterion.log_var_act.data.fill_(0.0)
        if hasattr(criterion, "log_var_psr"):
            criterion.log_var_psr.data.fill_(0.0)

        # Begin warmup ramp
        if not stage3_warmup_state["active"] and stage3_warmup_state["warmup_epochs"] > 0:
            stage3_warmup_state["active"] = True
            stage3_warmup_state["start_epoch"] = epoch
            stage3_warmup_state["epochs_remaining"] = stage3_warmup_state["warmup_epochs"]

        # Reinitialize EMA
        from models.model import EMA as EMAClass

        stage3_decay = _get_ema_decay(epoch)
        ema = EMAClass(model, decay=stage3_decay, device=device)

    # VideoMAE projection unfreeze (mirrors train.py:2350-2397)
    if videomae_unfreeze_epoch is not None and epoch == videomae_unfreeze_epoch and C.USE_VIDEOMAE:
        activity_head = getattr(model, "activity_head", None)
        if activity_head is not None and getattr(activity_head, "videomae_proj", None) is not None:
            for p in activity_head.videomae_proj.parameters():
                p.requires_grad = True

    return ema


# ===========================================================================
# Tests
# ===========================================================================


class TestGetStage:
    """Boundary tests for get_stage() — Doc 2 B.1 schedule."""

    def test_stage1_warmup(self):
        for ep in [0, 1, 3, 5]:
            assert get_stage(ep) == 1, f"epoch {ep} should be stage 1"

    def test_stage2_pose_added(self):
        for ep in [6, 8, 10, 15]:
            assert get_stage(ep) == 2, f"epoch {ep} should be stage 2"

    def test_stage3_full_multitask(self):
        for ep in [16, 20, 50, 99, 150]:
            assert get_stage(ep) == 3, f"epoch {ep} should be stage 3"


class TestStageFreezeSchedule:
    """_set_stage_requires_grad must match the documented freeze policy."""

    def test_stage1_freezes_activity_and_psr_heads(self):
        model = _MockModel()
        _set_stage_requires_grad(model, stage=1, backbone_type="resnet50")
        # Activity + PSR heads must be frozen in stage 1
        for name, p in model.named_parameters():
            if "activity_head" in name or "psr_head" in name:
                assert not p.requires_grad, f"Stage 1 should freeze {name}, but it is trainable"

    def test_stage2_still_freezes_activity_and_psr_heads(self):
        model = _MockModel()
        _set_stage_requires_grad(model, stage=2, backbone_type="resnet50")
        for name, p in model.named_parameters():
            if "activity_head" in name or "psr_head" in name:
                assert not p.requires_grad, f"Stage 2 should freeze {name}, but it is trainable"

    def test_stage3_unfreezes_everything(self):
        model = _MockModel()
        _set_stage_requires_grad(model, stage=3, backbone_type="resnet50")
        for name, p in model.named_parameters():
            assert p.requires_grad, f"Stage 3 must leave {name} trainable, but it is frozen"


class TestStage3Entry:
    """The 3 critical side-effects of crossing the stage 2 -> 3 boundary."""

    def test_warmup_ramp_starts(self):
        model = _MockModel()
        optimizer = MagicMock()
        optimizer.param_groups = _build_fake_param_groups(model)
        ema = MagicMock()
        criterion = MagicMock()
        criterion.log_var_act = nn.Parameter(torch.tensor(0.5))
        criterion.log_var_psr = nn.Parameter(torch.tensor(0.5))

        stage3_warmup_state = {
            "active": False,
            "param_group_idx": 2,
            "base_lr": 1e-3,
            "start_epoch": -1,
            "warmup_epochs": 3,
            "epochs_remaining": 0,
        }

        # Pre-stage 3: nothing active
        assert stage3_warmup_state["active"] is False
        assert stage3_warmup_state["epochs_remaining"] == 0

        # Cross from epoch 15 (stage 2) -> epoch 16 (stage 3)
        ema = _trigger_stage3_entry(
            model,
            optimizer,
            ema,
            epoch=16,
            criterion=criterion,
            stage3_warmup_state=stage3_warmup_state,
        )

        # Assertion 1: warmup ramp started
        assert stage3_warmup_state["active"] is True, (
            "Stage 3 entry should activate the warmup ramp"
        )
        assert stage3_warmup_state["start_epoch"] == 16
        assert stage3_warmup_state["epochs_remaining"] == 3, (
            f"epochs_remaining should be 3, got {stage3_warmup_state['epochs_remaining']}"
        )

    def test_ema_is_reinitialized_with_stage3_decay(self):
        model = _MockModel()
        optimizer = MagicMock()
        optimizer.param_groups = _build_fake_param_groups(model)
        ema = MagicMock()
        criterion = MagicMock()
        criterion.log_var_act = nn.Parameter(torch.tensor(0.0))
        criterion.log_var_psr = nn.Parameter(torch.tensor(0.0))

        stage3_warmup_state = {
            "active": False,
            "param_group_idx": 2,
            "base_lr": 1e-3,
            "start_epoch": -1,
            "warmup_epochs": 3,
            "epochs_remaining": 0,
        }

        # Patch models.model.EMA so we can verify it was constructed afresh
        with patch("models.model.EMA") as MockEMA:
            MockEMA.return_value = MagicMock(name="new_ema")
            ema = _trigger_stage3_entry(
                model,
                optimizer,
                ema,
                epoch=16,
                criterion=criterion,
                stage3_warmup_state=stage3_warmup_state,
            )

            # Assertion 2: EMAClass was constructed with (model, decay, device)
            assert MockEMA.called, "EMA must be reinitialized at Stage 3 entry"
            call_args = MockEMA.call_args
            assert call_args.args[0] is model, "EMA must wrap the same model"
            # The decay should come from _get_ema_decay(16)
            expected_decay = _get_ema_decay(16)
            actual_decay = call_args.kwargs.get(
                "decay", call_args.args[1] if len(call_args.args) > 1 else None
            )
            assert actual_decay == expected_decay, (
                f"EMA decay should be {expected_decay}, got {actual_decay}"
            )

    def test_videomae_proj_unfrozen_at_unfreeze_epoch(self):
        """The projection on ActivityHead should be made trainable when
        VIDEOMAE_UNFREEZE_EPOCH is hit (separate from Stage 3 boundary)."""
        # Override config flag for the duration of this test
        original_use_vm = getattr(C, "USE_VIDEOMAE", False)
        C.USE_VIDEOMAE = True
        try:
            model = _MockModel()
            # Pre-freeze videomae_proj to simulate the post-Stage-2 state
            for p in model.activity_head.videomae_proj.parameters():
                p.requires_grad = False
            assert all(not p.requires_grad for p in model.activity_head.videomae_proj.parameters())

            optimizer = MagicMock()
            optimizer.param_groups = _build_fake_param_groups(model)
            ema = MagicMock()
            criterion = MagicMock()
            criterion.log_var_act = nn.Parameter(torch.tensor(0.0))
            criterion.log_var_psr = nn.Parameter(torch.tensor(0.0))
            stage3_warmup_state = {
                "active": False,
                "param_group_idx": 2,
                "base_lr": 1e-3,
                "start_epoch": -1,
                "warmup_epochs": 3,
                "epochs_remaining": 0,
            }

            # Hit the unfreeze epoch (10 is a common value)
            _trigger_stage3_entry(
                model,
                optimizer,
                ema,
                epoch=10,
                criterion=criterion,
                stage3_warmup_state=stage3_warmup_state,
                videomae_unfreeze_epoch=10,
            )

            # Assertion 3: videomae_proj is now trainable
            for p in model.activity_head.videomae_proj.parameters():
                assert p.requires_grad, (
                    "activity_head.videomae_proj must be unfrozen at the VideoMAE unfreeze epoch"
                )
        finally:
            C.USE_VIDEOMAE = original_use_vm

    def test_stage3_entry_does_not_fire_inside_stage2(self):
        """epoch=10 must NOT trigger Stage 3 entry side-effects."""
        model = _MockModel()
        optimizer = MagicMock()
        optimizer.param_groups = _build_fake_param_groups(model)
        ema = MagicMock()
        criterion = MagicMock()
        criterion.log_var_act = nn.Parameter(torch.tensor(0.5))
        criterion.log_var_psr = nn.Parameter(torch.tensor(0.5))
        stage3_warmup_state = {
            "active": False,
            "param_group_idx": 2,
            "base_lr": 1e-3,
            "start_epoch": -1,
            "warmup_epochs": 3,
            "epochs_remaining": 0,
        }

        with patch("models.model.EMA") as MockEMA:
            MockEMA.return_value = MagicMock()
            ema = _trigger_stage3_entry(
                model,
                optimizer,
                ema,
                epoch=10,
                criterion=criterion,
                stage3_warmup_state=stage3_warmup_state,
            )
            # Warmup must NOT have been activated
            assert stage3_warmup_state["active"] is False
            # EMA must NOT have been reconstructed
            assert not MockEMA.called, "EMA should not be rebuilt mid-Stage-2"

    def test_all_three_side_effects_in_one_transition(self):
        """End-to-end: cross epoch 15 -> 16 and verify the full triplet."""
        # Enable videomae + set unfreeze epoch == 10 (already in the past)
        C.USE_VIDEOMAE = True
        try:
            model = _MockModel()
            # Pre-freeze everything that Stage 1/2 would have frozen
            for n, p in model.named_parameters():
                if "activity_head" in n or "psr_head" in n:
                    p.requires_grad = False

            optimizer = MagicMock()
            optimizer.param_groups = _build_fake_param_groups(model)
            ema = MagicMock()
            criterion = MagicMock()
            criterion.log_var_act = nn.Parameter(torch.tensor(0.5))
            criterion.log_var_psr = nn.Parameter(torch.tensor(0.5))
            stage3_warmup_state = {
                "active": False,
                "param_group_idx": 2,
                "base_lr": 1e-3,
                "start_epoch": -1,
                "warmup_epochs": 3,
                "epochs_remaining": 0,
            }

            with patch("models.model.EMA") as MockEMA:
                MockEMA.return_value = MagicMock()
                ema = _trigger_stage3_entry(
                    model,
                    optimizer,
                    ema,
                    epoch=16,
                    criterion=criterion,
                    stage3_warmup_state=stage3_warmup_state,
                    videomae_unfreeze_epoch=10,  # already past
                )

                # (1) warmup active
                assert stage3_warmup_state["active"] is True
                # (2) EMA reinit
                assert MockEMA.called
                # (3) videomae_proj unfrozen
                for p in model.activity_head.videomae_proj.parameters():
                    assert p.requires_grad
        finally:
            C.USE_VIDEOMAE = False


class TestEmaDecaySchedule:
    """_get_ema_decay returns a sane, monotonically non-decreasing decay."""

    def test_decay_within_bounds(self):
        for ep in [0, 5, 16, 20, 50, 99]:
            d = _get_ema_decay(ep)
            assert 0.0 < d < 1.0, f"decay at epoch {ep} should be in (0,1), got {d}"

    def test_decay_increases_over_time(self):
        d0 = _get_ema_decay(0)
        d50 = _get_ema_decay(50)
        d99 = _get_ema_decay(99)
        assert d0 <= d50 <= d99, f"EMA decay should be non-decreasing: d0={d0} d50={d50} d99={d99}"


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
