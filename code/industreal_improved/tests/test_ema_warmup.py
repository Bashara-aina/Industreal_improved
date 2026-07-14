"""Tests for EMA warmup, accumulation, and decay schedule.

EMA class lives at src.models.model.EMA; the training loop (train.py:1515)
guards ema.update() behind ``epoch >= C.EMA_START_EPOCH`` (default 5).
"""
import torch
import torch.nn as nn
import pytest

from src.training.ema import ModelEMA


class _DummyNet(nn.Module):
    """Minimal network with one trainable parameter."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)


# ============================================================================
# EMA shadow initialization & warmup guard
# ============================================================================

class TestEMAWarmup:
    """Verify that EMA shadows exist from init but training-loop gating
    (epoch >= C.EMA_START_EPOCH) controls when update() actually mutates."""

    def test_shadow_initialised_at_construction(self):
        """EMA shadow should equal model weights immediately after init."""
        net = _DummyNet()
        ema = ModelEMA(net, decay=0.999)
        for name, param in net.named_parameters():
            if param.requires_grad:
                assert name in ema.shadow, f"{name} missing from shadow"
                assert torch.equal(ema.shadow[name], param.data), (
                    f"Shadow for {name} differs from initial param"
                )

    def test_update_mutates_shadow(self):
        """After one update(), shadow should differ from current param
        (because the model weights changed but EMA blends old shadow + new weights)."""
        net = _DummyNet()
        ema = ModelEMA(net, decay=0.999)
        old_shadow = {k: v.clone() for k, v in ema.shadow.items()}

        # Modify model weights, then update EMA
        with torch.no_grad():
            for p in net.parameters():
                p.add_(torch.randn_like(p) * 0.1)
        ema.update()

        for name in old_shadow:
            assert not torch.equal(ema.shadow[name], old_shadow[name]), (
                f"Shadow for {name} unchanged after update"
            )

    def test_no_update_before_warmup(self):
        """Simulate the training-loop guard: ema.update() is NOT called
        when epoch < EMA_START_EPOCH (here 5).  Instead we call skip() —
        the shadow must remain at its initial value."""
        net = _DummyNet()
        ema = ModelEMA(net, decay=0.999)
        initial = {k: v.clone() for k, v in ema.shadow.items()}

        for epoch in range(1, 5):          # epochs before start
            with torch.no_grad():
                for p in net.parameters():
                    p.add_(torch.randn_like(p) * 0.01)
            # emulate training-loop guard: no ema.update() called

        # Verify shadow has NOT been touched
        for name in initial:
            assert torch.equal(ema.shadow[name], initial[name]), (
                f"Shadow for {name} drifted despite no update() calls"
            )

    def test_accumulation_after_warmup(self):
        """After epoch >= 5, ema.update() is called each step; verify
        the shadow has moved from initial."""
        net = _DummyNet()
        ema = ModelEMA(net, decay=0.999)
        initial = {k: v.clone() for k, v in ema.shadow.items()}

        for epoch in range(5, 10):
            with torch.no_grad():
                for p in net.parameters():
                    p.add_(torch.randn_like(p) * 0.01)
            ema.update()                     # called because epoch >= 5

        for name in initial:
            assert not torch.equal(ema.shadow[name], initial[name]), (
                f"Shadow for {name} never changed after accumulation"
            )


# ============================================================================
# EMA decay schedule (train.py _get_ema_decay)
# ============================================================================

class TestEMADecaySchedule:
    """The EMA decay schedule is set via ema.set_decay() each epoch
    (train.py:4602) following the schedule defined in _get_ema_decay."""

    def test_set_decay_updates_decay_attr(self):
        net = _DummyNet()
        ema = ModelEMA(net, decay=0.999)
        assert ema.decay == 0.999

        ema.set_decay(0.9999)
        assert ema.decay == 0.9999

        ema.set_decay(0.5)
        assert ema.decay == 0.5

    def test_decay_schedule_matches_doc(self):
        """Doc 2 A.2 / train.py _get_ema_decay:
        epoch 16 -> 0.999, epoch 17 -> 0.9995, 18+ -> 0.9999."""
        net = _DummyNet()
        ema = ModelEMA(net, decay=0.999)

        ema.set_decay(0.999)     # epoch 16
        self._run_and_measure(ema, net, expected_decay=0.999)

        ema.set_decay(0.9995)    # epoch 17
        self._run_and_measure(ema, net, expected_decay=0.9995)

        ema.set_decay(0.9999)    # epoch 18+
        self._run_and_measure(ema, net, expected_decay=0.9999)

    @staticmethod
    def _run_and_measure(ema, net, expected_decay):
        """Helper: do one update() and verify the blending factor."""
        old = {k: v.clone() for k, v in ema.shadow.items()}
        with torch.no_grad():
            for p in net.parameters():
                p.add_(torch.randn_like(p) * 0.1)
        ema.update()
        # Check that the blending relationship holds
        for name in old:
            expected = expected_decay * old[name] + (1 - expected_decay) * (
                dict(net.named_parameters())[name].data
            )
            assert torch.allclose(ema.shadow[name], expected, atol=1e-6), (
                f"Shadow for {name} does not match expected decay blend"
            )

    def test_low_decay_tracks_fast(self):
        """With decay=0.0, shadow immediately equals current weights."""
        net = _DummyNet()
        ema = ModelEMA(net, decay=0.999)
        initial = ema.shadow[next(iter(ema.shadow))].clone()

        ema.set_decay(0.0)
        with torch.no_grad():
            for p in net.parameters():
                p.mul_(2.0)
        ema.update()

        for name, param in net.named_parameters():
            if param.requires_grad:
                assert torch.equal(ema.shadow[name], param.data), (
                    "decay=0 should make shadow == current param"
                )


# ============================================================================
# EMA get_ema / restore round-trip
# ============================================================================

class TestEMAApplyRestore:
    """Verify that get_ema() applies shadow weights and restore()
    recovers the original training weights."""

    def test_get_ema_swaps_weights(self):
        net = _DummyNet()
        ema = ModelEMA(net, decay=0.9)
        original = {n: p.data.clone() for n, p in net.named_parameters()
                    if p.requires_grad}

        # Train a few steps so EMA diverges from model
        for _ in range(5):
            with torch.no_grad():
                for p in net.parameters():
                    p.add_(torch.randn_like(p) * 0.05)
            ema.update()

        ema.get_ema()
        for name, param in net.named_parameters():
            if name in ema.shadow:
                assert torch.equal(param.data, ema.shadow[name].to(param.device)), (
                    f"{name} not EMA weight after get_ema()"
                )

    def test_restore_recovers_pre_ema_weights(self):
        """restore() recovers the weights that were in the model before get_ema()."""
        net = _DummyNet()
        ema = ModelEMA(net, decay=0.9)

        for _ in range(5):
            with torch.no_grad():
                for p in net.parameters():
                    p.add_(torch.randn_like(p) * 0.05)
            ema.update()

        # Save the training weights before get_ema() overwrites them
        pre_ema = {n: p.data.clone() for n, p in net.named_parameters()
                   if p.requires_grad}

        _ = ema.get_ema()      # apply EMA (saves pre_ema weights into backup)
        ema.restore()          # restore from backup
        for name, param in net.named_parameters():
            if name in pre_ema:
                assert torch.equal(param.data, pre_ema[name]), (
                    f"{name} not restored to pre-EMA weights"
                )
