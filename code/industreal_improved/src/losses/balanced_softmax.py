"""Balanced Softmax (Menon et al. 2020 NeurIPS).

Reference:
    Menon, A. K., Jayasumana, S., Rawat, A. S., Jain, H., Veit, A., & Kumar, S.
    (2020). Long-tail learning via logit adjustment. ICLR 2021.
    https://openreview.net/forum?id=klO8aM6Ycn

The loss shifts logits by log(prior(y)) before softmax, equivalent to
logit-adjustment with tau=1 and the reference distribution set to the
marginal class priors.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BalancedSoftmaxLoss(nn.Module):
    """Balanced Softmax with deferred class-prior initialisation.

    Priors are initialised as uniform and overwritten by ``set_class_counts``
    once per-class frequencies are available from the training dataset.

    Parameters
    ----------
    num_classes : int
        Number of activity classes (typically NUM_CLASSES_ACT).
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.register_buffer('class_priors', torch.ones(num_classes) / num_classes)

    def set_class_counts(self, counts, eps: float = 1e-8):
        """Update class priors from empirical class frequencies.

        Parameters
        ----------
        counts : array-like of shape (num_classes,)
            Per-class frame counts.  May be shorter (74) or longer (75)
            than self.num_classes; lengths are reconciled automatically.
        eps : float, default=1e-8
            Small constant to avoid log(0).
        """
        counts = torch.as_tensor(counts, dtype=torch.float32, device=self.class_priors.device)
        # Reconcile length (see LDAMLoss.set_class_counts for rationale)
        n_in, n_want = counts.shape[0], self.num_classes
        if n_in == n_want - 1:
            # caller omitted the NA slot; prepend a sentinel of 1
            counts = torch.cat([torch.ones(1, device=counts.device), counts])
        elif n_in != n_want:
            # unexpected length — pad or truncate
            if n_in < n_want:
                counts = torch.cat([counts, torch.zeros(n_want - n_in, device=counts.device)])
            else:
                counts = counts[:n_want]
        priors = counts / (counts.sum() + eps)
        self.class_priors.data.copy_(priors)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits_shifted = logits + torch.log(self.class_priors.unsqueeze(0))
        return F.cross_entropy(logits_shifted, targets)
