"""Random seed utilities for reproducibility."""
import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seed for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic mode for reproducible training
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False