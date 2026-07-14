"""MS-TCN Multi-Stage Refinement Head for PSR (Abu Farha & Gall, CVPR 2019).

Stacks dilated temporal convolution stages on top of the causal transformer
output to suppress over-segmentation (spurious short flips) in PSR predictions.

Key result from paper (50Salads):
  Single-stage TCN: F1@10 = 27.0, frame-acc = 78.2
  4-stage MS-TCN:   F1@10 = 76.3, frame-acc = 80.7
  → F1 nearly TRIPLES while frame-acc barely moves (+2.5)

Architecture:
  Each stage: 10 dilated conv layers (filter=3, filters=64, dilation 2^layer)
  Input to each stage: ONLY frame-wise probabilities (no features — Table 5)
  Output: refined per-frame component logits

This module operates HEAD-ONLY — no backbone gradient impact. The refinement
stages are detached from the backbone, so they improve PSR temporal coherence
without affecting inter-task gradient competition.

Args:
    num_components: number of PSR binary classes (11)
    num_stages: number of refinement stages (1-4; paper uses 4)
    num_layers: dilated conv layers per stage (paper uses 10)
    num_filters: conv filters per layer (paper uses 64)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MSRefinementStage(nn.Module):
    """Single MS-TCN refinement stage: 10 dilated conv layers → output."""

    def __init__(
        self,
        num_components: int = 11,
        num_layers: int = 10,
        num_filters: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_layers = num_layers
        layers = []
        for i in range(num_layers):
            dilation = 2 ** i  # 1, 2, 4, 8, ..., 512
            in_ch = num_components if i == 0 else num_filters
            out_ch = num_components if i == num_layers - 1 else num_filters
            padding = dilation  # same-length output
            layers.append(
                nn.Conv1d(in_ch, out_ch, kernel_size=3, dilation=dilation, padding=padding)
            )
            if i < num_layers - 1:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        self.convs = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B*T, C] or [B, C, T] → same shape."""
        if x.dim() == 2:
            # [B*T, C] → [B*T, C, 1] for conv → squeeze back
            x = x.unsqueeze(-1)
            out = self.convs(x)
            return out.squeeze(-1)
        return self.convs(x)


class PSRRefinementHead(nn.Module):
    """Multi-stage refinement on PSR causal transformer output.

    Stacks MSRefinementStages that operate ONLY on probabilities (not features),
    preventing any additional gradient flow into the shared backbone.

    Input: raw logits from causal transformer [B, T, C]
    Output: refined logits [B, T, C] (sum of all stage logits per paper)
    """

    def __init__(
        self,
        num_components: int = 11,
        num_stages: int = 2,
        num_layers: int = 10,
        num_filters: int = 64,
        dropout: float = 0.2,
        smoothing_lambda: float = 0.15,
        smoothing_tau: float = 4.0,
    ):
        super().__init__()
        self.num_stages = num_stages
        self.num_components = num_components
        self.smoothing_lambda = smoothing_lambda
        self.smoothing_tau = smoothing_tau

        self.stages = nn.ModuleList([
            MSRefinementStage(num_components, num_layers, num_filters, dropout)
            for _ in range(num_stages)
        ])

    def forward(self, logits: torch.Tensor, apply_sigmoid: bool = True) -> torch.Tensor:
        """Refine PSR logits through dilated temporal convolutions.

        Args:
            logits: [B, T, C] raw logits from causal transformer.
            apply_sigmoid: if True, convert to probabilities before refinement
                          (matches paper's "probabilities-only" input finding).

        Returns:
            Refined logits [B, T, C] — sum of all stage outputs.
        """
        B, T, C = logits.shape

        # First stage input: probabilities from causal transformer
        if apply_sigmoid:
            stage_input = torch.sigmoid(logits)  # [B, T, C]
        else:
            stage_input = logits

        # Accumulate across stages (paper: sum of all stage outputs)
        total_output = torch.zeros_like(logits)

        for stage in self.stages:
            # Reshape for Conv1d: [B, T, C] → [B, C, T]
            x = stage_input.permute(0, 2, 1)  # [B, C, T]
            out = stage(x)                     # [B, C, T] → refined probs
            out_probs = out.permute(0, 2, 1)  # [B, T, C]
            total_output = total_output + out_probs
            # Next stage input: refined probabilities (detach to prevent
            # gradient flow through earlier stages per paper's finding)
            stage_input = out_probs.detach()

        return total_output

    def smoothing_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """MS-TCN truncated MSE smoothing loss on log-probabilities.

        L_T-MSE applied to the FINAL stage output. Encodes temporal
        smoothness prior without suppressing genuine transitions.

        Args:
            logits: [B, T, C] output logits from forward().

        Returns:
            Scalar loss term.
        """
        if logits.size(1) < 2:
            return torch.tensor(0.0, device=logits.device)

        log_p = F.logsigmoid(logits)  # [B, T, C]
        diff = log_p[:, 1:, :] - log_p[:, :-1, :]  # [B, T-1, C]
        delta_tilde = diff.abs().clamp(max=self.smoothing_tau)
        return self.smoothing_lambda * delta_tilde.pow(2).mean()
