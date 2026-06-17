"""Alias module — the single source of truth is ``src.training.losses``.

[AUDIT FIX 2026-06-17 — duplicate-module split-brain]
This was a full copy of src/training/losses.py. ``import losses``
(quick_eval.py, run_eval_direct.py, audit_eval.py, etc.) executed the file
a second time under the module name ``losses``, creating a separate module
object from ``src.training.losses`` (used by train.py). Runtime overrides
and stateful loss configs mutated one object while training read the other.
This alias makes both import paths resolve to the SAME module object.
"""
import os as _os
import sys as _sys

_ROOT = _os.path.dirname(_os.path.abspath(__file__))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

from src.training import losses as _src_losses

_sys.modules[__name__] = _src_losses
