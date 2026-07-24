"""Balanced PSR loss with LogitAdjust + class-balanced weighting.

Implements multi-label LogitAdjust (Menon et al. ICLR 2021) and
class-balanced effective-number weights (Cui et al. CVPR 2019) to
address the imbalanced positive rates across the 11 PSR components.

Without balancing, rare components (positive rate <25%) are predicted
as 0 (majority class) for high accuracy but 0% recall / F1. LogitAdjust
shifts the decision boundary to account for class prior, restoring
recall for rare components.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_psr_priors(loader, num_components: int = 11, max_batches: int = 200) -> np.ndarray:
    """Compute per-component positive rate from a DataLoader.

    Returns:
        [num_components] float32 array of positive rates in [eps, 1-eps].
    """
    n_pos = np.zeros(num_components, dtype=np.int64)
    n_total = 0
    n_batches = 0
    for batch in loader:
        targets = batch[1] if isinstance(batch, (tuple, list)) and len(batch) == 2 else batch
        if isinstance(targets, dict) and 'psr' in targets:
            psr_list = targets['psr']
            for p in psr_list:
                if p is None:
                    continue
                arr = p.detach().cpu().numpy() if torch.is_tensor(p) else np.asarray(p)
                arr = arr.flatten()[:num_components]
                n_pos += (arr > 0.5).astype(np.int64)
                n_total += 1
        n_batches += 1
        if n_batches >= max_batches:
            break
    rates = n_pos / max(n_total, 1)
    return rates.astype(np.float32)


def logit_adjustment_vector(priors: np.ndarray, tau: float = 1.0, eps: float = 1e-3) -> np.ndarray:
    """Compute LogitAdjust bias for each component.

    For multi-label binary classification with positive rate pi_c, we want to
    boost rare classes (low pi_c) so the model is more willing to predict
    them. The bias added to the logit is:

        bias_c = -tau * log(pi_c / (1 - pi_c)) = tau * log((1 - pi_c) / pi_c)

    Properties:
      - pi_c = 0.5: bias = 0 (no adjustment)
      - pi_c < 0.5: bias > 0 (boost rare positive prediction)
      - pi_c > 0.5: bias < 0 (dampen common positive prediction)

    Args:
        priors: [num_components] positive rates in (0, 1).
        tau: temperature. 1.0 = full adjustment. 0.5 = half.
        eps: clip priors to [eps, 1-eps] to avoid inf.

    Returns:
        [num_components] float32 bias to add to logits.
    """
    priors = np.clip(priors, eps, 1.0 - eps)
    return (tau * np.log((1.0 - priors) / priors)).astype(np.float32)


def effective_number_weights(priors: np.ndarray, beta: float = 0.999) -> np.ndarray:
    """Class-balanced weights from effective number (Cui et al. CVPR 2019).

    Args:
        priors: [num_components] positive rates in (0, 1).
        beta: decay. 0.999 = severe, 0.99 = moderate.

    Returns:
        [num_components] float32 weights, normalized to sum=num_components.
    """
    n_c = priors * 10000.0  # convert rate to pseudo-count
    eff_num = (1.0 - np.power(beta, n_c)) / (1.0 - beta)
    eff_num = np.maximum(eff_num, 1e-8)
    weights = 1.0 / eff_num
    weights = weights / weights.sum() * len(weights)
    return weights.astype(np.float32)


class PSRBalancedLoss(nn.Module):
    """Balanced multi-label BCE with LogitAdjust and per-component weights.

    Args:
        priors: [num_components] positive rates from training data.
        tau: LogitAdjust temperature. 0 = no adjustment, 1 = full.
        beta: Effective-number beta. <0.99 to disable per-component weights.
        num_components: 11.
    """

    def __init__(
        self,
        priors: np.ndarray,
        tau: float = 1.0,
        beta: float = 0.999,
        num_components: int = 11,
    ):
        super().__init__()
        self.tau = tau
        self.num_components = num_components
        # Register bias as buffer (not parameter — not learned)
        # logit_adjustment_vector clips priors internally to avoid inf
        bias = logit_adjustment_vector(priors, tau=tau)
        self.register_buffer("logit_bias", torch.tensor(bias, dtype=torch.float32))
        # Per-component weight for imbalance
        if beta > 0:
            weights = effective_number_weights(priors, beta=beta)
        else:
            weights = np.ones(num_components, dtype=np.float32)
        self.register_buffer("comp_weights", torch.tensor(weights, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute balanced multi-label BCE.

        Args:
            logits: [B, 11] raw logits.
            targets: [B, 11] binary targets in {0, 1}.

        Returns:
            Scalar loss.
        """
        adjusted = logits + self.logit_bias
        # Per-element BCE then weighted sum across components
        bce = F.binary_cross_entropy_with_logits(adjusted, targets, reduction='none')
        weighted = bce * self.comp_weights
        return weighted.mean()


__all__ = [
    "compute_psr_priors",
    "logit_adjustment_vector",
    "effective_number_weights",
    "PSRBalancedLoss",
]
