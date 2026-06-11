"""Alias module — the single source of truth is ``src/config.py``.

[AUDIT FIX 2026-06-11 — config split-brain]
This was previously a symlink to src/config.py. A symlink guarantees identical
*code* but NOT identical *state*: ``import config`` executed src/config.py a
second time under the module name ``config``, creating a separate module
object from ``src.config`` (used by model.py / losses.py / train.py via
``from src import config``). ``apply_preset()`` and every runtime override
(``--preset recovery``, ``ZERO_DET_CONF_FOR_RECOVERY``, ``MIXED_PRECISION``,
``BATCH_SIZE``) mutated one object while the model and training loop read the
other — the entire 'recovery' preset was a silent no-op for training. This
alias makes both import paths resolve to the SAME module object.
"""
import os as _os
import sys as _sys

_ROOT = _os.path.dirname(_os.path.abspath(__file__))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

from src import config as _src_config

_sys.modules[__name__] = _src_config
