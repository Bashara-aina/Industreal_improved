"""
BestModelTracker — tracks best score and saves best checkpoint.
Replaces inline best-model logic from train.py.
"""

import torch
from pathlib import Path


class BestModelTracker:
    """
    Tracks the best score across training epochs and saves best checkpoint.

    Usage:
        tracker = BestModelTracker(mode='max')  # higher is better
        tracker.set_save_dir('/path/to/checkpoints')
        tracker.update(0.75, {'model': model.state_dict(), ...})
        print(tracker.best_score, tracker.best_epoch)
    """

    def __init__(self, mode="max"):
        """
        Args:
            mode: 'max' (higher score = better) or 'min' (lower score = better)
        """
        if mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got {mode}")
        self.mode = mode
        self.best_score = None
        self.best_epoch = None
        self.best_checkpoint = None
        self.save_dir = None

    def set_save_dir(self, save_dir):
        """Set directory for best checkpoint saves."""
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def update(self, score, checkpoint_dict):
        """
        Update best model if score improves.

        Args:
            score: float metric value
            checkpoint_dict: dict with 'model', 'optimizer', 'epoch', etc.

        Returns:
            True if new best, False otherwise
        """
        if self.best_score is None:
            is_best = True
        elif self.mode == "max":
            is_best = score > self.best_score
        else:
            is_best = score < self.best_score

        if is_best:
            self.best_score = score
            self.best_epoch = checkpoint_dict.get("epoch", None)
            self.best_checkpoint = checkpoint_dict.copy() if checkpoint_dict else None

            if self.save_dir is not None:
                torch.save(checkpoint_dict, self.save_dir / "best.pth")
            return True
        return False

    @property
    def improved(self):
        """True if a new best has been set."""
        return self.best_score is not None
