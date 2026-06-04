"""
TensorBoard logging utilities for IndustReal training.
"""

import os
from typing import Optional

from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:
    """
    Thin wrapper around torch.utils.tensorboard.SummaryWriter.

    Provides a simple `log_scalar(tag, value, step)` interface.
    """

    def __init__(self, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        self._writer = SummaryWriter(log_dir=log_dir)

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a scalar value to TensorBoard."""
        self._writer.add_scalar(tag, value, step)

    def log_histogram(self, tag: str, values, step: int) -> None:
        """Log a histogram of values to TensorBoard."""
        self._writer.add_histogram(tag, values, step)

    def close(self) -> None:
        """Close the underlying writer."""
        self._writer.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()