"""MS-TCN Truncated Mean-Squared Error Smoothing Loss.

From Abu Farha & Gall, "MS-TCN: Multi-Stage Temporal Convolutional Network
for Action Segmentation", CVPR 2019 (arXiv:1903.01945), Equations 8-12.

Key result: On 50Salads, adding this smoothing loss to a single-stage TCN
improves F1@10 from 71.3 to 76.3 (+5.0) at constant frame-accuracy.

The loss penalizes frame-to-frame changes in log-probabilities, truncated
at tau to avoid suppressing genuine action boundaries.
"""

import torch
import torch.nn.functional as F


def ms_tcn_smoothing_loss(
    logits: torch.Tensor,
    tau: float = 4.0,
    lambda_smooth: float = 0.15,
) -> torch.Tensor:
    """MS-TCN truncated MSE smoothing loss on log-probabilities.

    L_T-MSE = (1 / T*C) * sum_t sum_c clamp(|log p_{t,c} - log p_{t-1,c}|, max=tau)^2

    Applied per-component independently. Gradients flow only to y_t (not y_{t-1}),
    matching the asymmetric gradient flow in the original paper.

    Args:
        logits: [B, T, C] per-frame per-component logits BEFORE sigmoid.
        tau: truncation threshold (default 4.0 from paper).
        lambda_smooth: weight multiplier (default 0.15 from paper).

    Returns:
        Scalar smoothing loss (already multiplied by lambda_smooth).
    """
    if logits.size(1) < 2:
        return torch.tensor(0.0, device=logits.device)

    # Convert to log-probabilities of the sigmoid output
    # log p = log(sigmoid(x)) = logsigmoid(x) for positive class
    # We penalize changes in log-probabilities as per MS-TCN Eq. 8
    log_p = F.logsigmoid(logits)  # [B, T, C]

    # Diff between adjacent frames: log_p_t - log_p_{t-1}
    diff = log_p[:, 1:, :] - log_p[:, :-1, :]  # [B, T-1, C]

    # Truncated MSE: clamp |delta| at tau, then square
    delta = diff.abs()
    delta_tilde = delta.clamp(max=tau)  # [B, T-1, C], tau=4

    # L_T-MSE = mean over (B, T-1, C) of delta_tilde^2
    loss = delta_tilde.pow(2).mean()

    return lambda_smooth * loss
