"""Test: LDAM_USE_DRW config flag exists and is used directly (no getattr fallback).

The bug: `losses.py` used `getattr(C, 'LDAM_USE_DRW', True)` which silently
defaulted to True if the config attribute was never defined, masking config errors.
In fact, `LDAM_USE_DRW` was NOT actually defined in config.py — only commented.

This test verifies:
  1. C.LDAM_USE_DRW exists as a defined attribute in config (not via getattr fallback)
  2. LDAMLoss.set_class_counts reads C.LDAM_USE_DRW directly (not via getattr)
"""
import importlib
import src.config as C
from src.training.losses import LDAMLoss


def test_ldam_use_drw_defined_in_config():
    """LDAM_USE_DRW must be a defined attribute in config.py, not silently defaulted."""
    assert hasattr(C, 'LDAM_USE_DRW'), (
        "C.LDAM_USE_DRW is not defined in config.py! "
        "The config flag was only a comment, never an actual assignment. "
        "The old getattr(C, 'LDAM_USE_DRW', True) silently masked this."
    )
    # Must be a bool
    assert isinstance(C.LDAM_USE_DRW, bool)


def test_ldam_loss_uses_direct_attr_not_getattr():
    """LDAMLoss.set_class_counts must read C.LDAM_USE_DRW directly, not via getattr.

    We check by reading the source code — if 'getattr(C,' appears in the
    condition, the fix hasn't been applied yet.
    """
    import inspect
    source = inspect.getsource(LDAMLoss.set_class_counts)
    # The condition should read C.LDAM_USE_DRW directly
    assert 'C.LDAM_USE_DRW' in source, (
        "LDAMLoss.set_class_counts does not reference C.LDAM_USE_DRW directly!"
    )
    # Must NOT use getattr(C, 'LDAM_USE_DRW', ...) which silently defaults
    assert 'getattr(C' not in source or 'LDAM_USE_DRW' not in source.split('getattr(C')[1], (
        "LDAMLoss.set_class_counts still uses getattr(C, 'LDAM_USE_DRW', ...) "
        "which silently defaults — fix it to use C.LDAM_USE_DRW directly."
    )


def test_ldam_drw_toggle():
    """When LDAM_USE_DRW=True, LDAMLoss.set_class_counts should set cb_weights.

    When False, cb_weights should remain None.
    """
    loss = LDAMLoss(num_classes=75)

    # With DRW enabled, set_class_counts should wire cb_weights
    C.LDAM_USE_DRW = True
    import numpy as np
    loss.set_class_counts(np.ones(75, dtype=np.float32) * 10.0)
    assert loss.cb_weights is not None, "cb_weights should be set when LDAM_USE_DRW=True"

    # With DRW disabled, cb_weights should be None
    C.LDAM_USE_DRW = False
    loss.set_class_counts(np.ones(75, dtype=np.float32) * 10.0)
    assert loss.cb_weights is None, "cb_weights should be None when LDAM_USE_DRW=False"

    # Reset
    C.LDAM_USE_DRW = True
