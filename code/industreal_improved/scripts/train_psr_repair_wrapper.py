#!/usr/bin/env python3
"""Wrapper: Enable bf16 mixed precision + launch PSR head repair training.

Patches src.config to set MIXED_PRECISION=True BEFORE train.py imports it.
This allows stage_rf4 preset's mixed_precision=False to be overridden,
because we also intercept C.apply_preset() to re-apply the override.

Usage:
    CUDA_VISIBLE_DEVICES=1 python3 scripts/train_psr_repair_wrapper.py --preset stage_rf4 ...
"""
import os
import sys

# Set CUDA env vars before any torch import (same order as train.py)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['NVIDIA_TF32_OVERRIDE'] = '0'
os.environ.setdefault('CUDA_MODULE_LOADING', 'LAZY')

# Ensure project root is in sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Patch config module BEFORE train.py runs its import
from src import config as C

# Intercept apply_preset() so MIXED_PRECISION stays True even after
# stage_rf4 preset sets mixed_precision=False.
# [FIX 2026-07-07 File-157 F-1] Also force DETACH_PSR_FPN=False if env var set.
# The hardcoded `DETACH_PSR_FPN=True` in presets was blocking the PSR
# gradient flow even when the V3 launch script set the env var to False.
_orig_apply = C.apply_preset

def _patched_apply(name):
    _orig_apply(name)
    C.MIXED_PRECISION = True
    if os.environ.get('DETACH_PSR_FPN', 'True') == 'False':
        C.DETACH_PSR_FPN = False
        print(f'[wrapper] Post-preset override: DETACH_PSR_FPN=False '
              f'(per env var, PSR gradient flow to backbone ENABLED)',
              file=sys.stderr, flush=True)
    print(f'[wrapper] Post-preset override: MIXED_PRECISION=True (applied after {name})',
          file=sys.stderr, flush=True)

C.apply_preset = _patched_apply

# Also set AMP_DTYPE from env (should be 'bf16')
_amp_dtype = os.environ.get('AMP_DTYPE', 'bf16')
print(f'[wrapper] AMP_DTYPE={_amp_dtype}', file=sys.stderr, flush=True)

# Run train.py with proper __main__ semantics
import runpy
sys.argv = ['train.py'] + sys.argv[1:]
runpy.run_path(os.path.join(_PROJECT_ROOT, 'src', 'training', 'train.py'),
               run_name='__main__')
