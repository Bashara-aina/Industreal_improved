"""RLW: Random Loss Weighting (Lin et al., TMLR 2022).

Samples task weights from Normal(0,1)+softmax at each step.
CRITICAL control baseline — if our adaptive method cannot beat
random weighting, the gain is not from the weighting scheme.

Single backward pass, O(1) complexity, stateless.
"""

import torch
import torch.nn.functional as F


class RLWWeighter:
    """RLW weight sampler — one instance per training run.

    Usage:
        rlw = RLWWeighter(num_tasks=4, distribution="normal")
        for step in range(N):
            loss_tensor = torch.stack(list(losses.values()))
            total = (loss_tensor * rlw.get_weights(loss_tensor.device)).sum()
    """

    def __init__(
        self,
        num_tasks: int = 4,
        distribution: str = "normal",
        temperature: float = 1.0,
    ):
        self.num_tasks = num_tasks
        self.distribution = distribution
        self.temperature = temperature

    def get_weights(self, device: torch.device) -> torch.Tensor:
        if self.distribution == "normal":
            z = torch.randn(self.num_tasks, device=device)
            return F.softmax(z / self.temperature, dim=0)
        elif self.distribution == "dirichlet":
            return torch.distributions.Dirichlet(torch.ones(self.num_tasks, device=device)).sample()
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")
