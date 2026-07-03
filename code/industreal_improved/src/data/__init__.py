"""Data package for IndustReal multi-task training.

[F17 2026-07-02 Fable RF4 consult] This __init__.py was MISSING from the git
repository (train.py does `import data as _ds_module` and then getattr's the
names below, which only works when the package re-exports them — so a
populated __init__.py must exist untracked on the training machine). Without
this file, a fresh clone cannot run training or evaluation. Reconstructed to
re-export the public API of industreal_dataset.py.
"""
from .industreal_dataset import (  # noqa: F401
    IndustRealMultiTaskDataset,
    collate_fn,
    collate_fn_sequences,
    clear_frame_cache,
)
