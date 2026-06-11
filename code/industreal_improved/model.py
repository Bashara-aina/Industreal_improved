"""Alias module — the single source of truth is ``src/models/model.py``.

[AUDIT FIX 2026-06-11 — duplicate-module split-brain]
This was previously a symlink to src/models/model.py. ``import model``
(train.py:90-91, ``from model import EMA``) executed the file a second time
under the module name ``model``, creating a separate module object from
``src.models.model``. This alias makes both import paths resolve to the SAME
module object (single execution, single state).
"""
import os as _os
import sys as _sys

_ROOT = _os.path.dirname(_os.path.abspath(__file__))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

from src.models import model as _src_model

_sys.modules[__name__] = _src_model
