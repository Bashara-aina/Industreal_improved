"""
Tier 2.7 — PSR Transition Predictor with Monotonic State Accumulator
=====================================================================
Predicts per-component transition EVENTS p(cᵢ flips at t), decoded with a
monotonic constraint + procedure-order prior.

Why this beats per-frame BCE:
  - Per-frame BCE on 95%-static labels collapses to a constant pattern
  - The PSR metric is F1 on state-change events within ±3 frames
  - The state space is monotone fill-forward (components are placed once and stay)
  - B2 (F1=0.731) is ASD-confidence accumulation + constraints — barely a neural model

Design:
  Input: ROI state-classifier outputs (Tier 2.5) or cached embeddings (Tier 2.4)
         → Causal Transformer (from PSRHead, shared) → [B, T, hidden]
  Output: per-component transition logits [B, T, 11] → decoding
  Loss: Gaussian-smeared event detection loss (target: smoothed transition indicator)
  Decode: monotonic constraint + procedure-order graph

Target: F1 > 0.75, POS > 0.82
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

# ============================================================================
# Gaussian Transition Target
# ============================================================================
def build_transition_targets(psr_labels: torch.Tensor, sigma: float = 3.0
                             ) -> torch.Tensor:
    """Convert binary-per-frame PSR labels to Gaussian-smeared transition targets.

    Args:
        psr_labels: [B, T, 11] binary labels (monotone fill-forward)
        sigma: Gaussian sigma for smoothing (±3σ frames around transition)

    Returns:
        targets: [B, T, 11] — Gaussian-smoothed transition indicators
                 Each channel peaks at 1.0 at the transition frame, decays
                 as exp(-(t-t_transition)² / (2σ²))
    """
    B, T, C = psr_labels.shape
    device = psr_labels.device

    # Find transition frames: where label flips from 0 to 1
    # For monotone fill-forward, we only care about 0→1 transitions
    padded = F.pad(psr_labels, (0, 0, 1, 0), value=0.0)[:, :-1, :]  # [B, T, C]
    transitions = (psr_labels - padded).clamp(min=0)  # [B, T, C] — 1 at transition frame

    # Build Gaussian kernel
    radius = int(3 * sigma)
    t_range = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    gauss = torch.exp(-0.5 * (t_range / sigma) ** 2)
    gauss = gauss / gauss.max()  # Normalize to peak at 1

    # Convolve with Gaussian
    targets = torch.zeros_like(psr_labels)
    for c in range(C):
        for b in range(B):
            trans_times = transitions[b, :, c].nonzero(as_tuple=True)[0]
            for t in trans_times:
                t = t.item()
                for offset, val in enumerate(gauss):
                    frame = t + offset - radius
                    if 0 <= frame < T:
                        targets[b, frame, c] = max(targets[b, frame, c], val)

    return targets


# ============================================================================
# Monotonic Decoder
# ============================================================================
class MonotonicDecoder(nn.Module):
    """Decode per-frame transition logits into monotonic state predictions.

    Uses a simple Viterbi-like forward pass: at each frame, a component can only
    transition from 0→1 (once) or stay at 1. This enforces the monotone fill-forward
    structure of assembly processes.

    Also incorporates a procedure-order prior: certain components must be placed
    before others (derived from the ground-truth procedure graph).
    """

    def __init__(self, num_components: int = 11,
                 procedure_order: Optional[List[Tuple[int, int]]] = None):
        super().__init__()
        self.num_components = num_components
        # Default procedure order: comp0→comp1→comp2→... (sequential)
        self.procedure_order = procedure_order or [
            (i, i + 1) for i in range(num_components - 1)
        ]
        # Register as buffer so it's on the right device
        self.register_buffer('_order_matrix', self._build_order_matrix())

    def _build_order_matrix(self) -> torch.Tensor:
        """Build [C, C] matrix where M[i,j] = 1 if i must be placed before j."""
        M = torch.zeros(self.num_components, self.num_components)
        for before, after in self.procedure_order:
            if before < self.num_components and after < self.num_components:
                M[before, after] = 1.0
        return M

    def forward(self, transition_logits: torch.Tensor,
                threshold: float = 0.3) -> torch.Tensor:
        """Decode per-frame transition logits into state predictions.

        Args:
            transition_logits: [B, T, C] — sigmoid probabilities of transition at frame t
            threshold: minimum probability to trigger a transition

        Returns:
            states: [B, T, C] — monotone binary state predictions
        """
        B, T, C = transition_logits.shape
        device = transition_logits.device

        # Initialize: all components start at 0
        states = torch.zeros(B, T, C, device=device)
        current_state = torch.zeros(B, C, device=device)  # [B, C] — current per-component state

        for t in range(T):
            # Get transition probabilities at frame t
            trans_prob = transition_logits[:, t, :]  # [B, C]

            # Components that haven't been placed yet
            can_transition = (current_state == 0)

            # Apply procedure-order constraint: a component can only transition
            # if ALL components that must come before it are already placed
            order_constraint = self._order_matrix.to(device)  # [C, C]
            # [GAP-A2 FIX] Check predecessor states (rows), not successor states (cols).
            # current_state.unsqueeze(2) → [B, C, 1] broadcasts with [C, C] → [B, C, C]
            # At [b, i, j]: state[i] >= M[i,j] — predecessor i is placed.
            # .all(dim=1) over i: for successor j, ALL predecessors i are placed.
            predecessors_placed = (current_state.unsqueeze(2) >= order_constraint).all(dim=1)
            can_transition = can_transition & predecessors_placed

            # Decide transitions
            transition = (trans_prob > threshold) & can_transition  # [B, C] bool
            # Once placed, stays placed — use addition+clamp instead of bitwise OR
            # to avoid NotImplementedError: 'bitwise_or' not implemented for Float on CPU
            current_state = (current_state + transition.float()).clamp(max=1.0)

            states[:, t, :] = current_state

        return states


# ============================================================================
# PSR Transition Predictor (full module)
# ============================================================================
class PSRTransitionPredictor(nn.Module):
    """End-to-end PSR transition prediction model.

    Feeds ROI state-classifier outputs through a causal transformer, predicts
    per-component transition events, and decodes with monotonic constraints.

    This can replace the per-frame BCE PSR head (Tier 2.7 point 7 design).
    """

    def __init__(self, input_dim: int = 512, hidden_dim: int = 256,
                 num_components: int = 11, num_heads: int = 4,
                 num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.num_components = num_components
        self.hidden_dim = hidden_dim

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Causal transformer for temporal modeling
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Transition logit heads (one per component — binary transition prediction)
        self.transition_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim // 2, 1),
            ) for _ in range(num_components)
        ])

        # Monotonic decoder
        self.decoder = MonotonicDecoder(num_components=num_components)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.input_proj.weight, std=0.02)
        nn.init.zeros_(self.input_proj.bias)
        for head in self.transition_heads:
            nn.init.normal_(head[0].weight, std=0.01)
            nn.init.zeros_(head[0].bias)
            nn.init.normal_(head[2].weight, std=0.01)
            # Bias toward "no transition" — sigmoid(-1) ≈ 0.27
            nn.init.constant_(head[2].bias, -1.0)

    def forward(self, features: torch.Tensor,
                return_states: bool = False) -> Dict[str, torch.Tensor]:
        """Predict PSR transitions and states from temporal features.

        Args:
            features: [B, T, input_dim] — temporal features (e.g., from embedding cache)
            return_states: if True, also decode states

        Returns:
            dict with:
                'transition_logits': [B, T, C] — per-component transition logits
                'states': [B, T, C] (if return_states) — decoded monotonic states
        """
        B, T, D = features.shape

        # Project to hidden dim
        x = self.input_proj(features)  # [B, T, hidden_dim]

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(T, T, device=features.device), diagonal=1
        ).bool()

        # Causal transformer
        encoded = self.transformer(x, mask=causal_mask)  # [B, T, hidden_dim]

        # Per-component transition logits
        transition_logits = torch.cat([
            head(encoded) for head in self.transition_heads
        ], dim=-1)  # [B, T, C]

        result = {'transition_logits': transition_logits}

        if return_states:
            transition_probs = torch.sigmoid(transition_logits)
            result['states'] = self.decoder(transition_probs)

        return result

    def compute_loss(self, transition_logits: torch.Tensor,
                     psr_labels: torch.Tensor,
                     sigma: float = 3.0) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute event-detection loss with Gaussian-smeared transition targets.

        Args:
            transition_logits: [B, T, C] — predicted logits
            psr_labels: [B, T, C] — binary fill-forward labels
            sigma: Gaussian smoothing sigma

        Returns:
            loss: scalar tensor
            metrics: dict with f1, precision, recall
        """
        # Build smoothed transition targets
        targets = build_transition_targets(psr_labels, sigma=sigma)

        # Binary logistic loss on transition targets
        # We use BCEWithLogitsLoss but weight transitions higher
        masks = targets > 0.01  # Only compute loss near transitions
        total = masks.sum() + 1e-8

        # Focal-like weighting: transitions (targets ~ 1.0) get high weight
        alpha = 0.25 * targets + 0.75 * (1 - targets)
        focal_weight = abs(targets - torch.sigmoid(transition_logits)) ** 2

        bce = F.binary_cross_entropy_with_logits(
            transition_logits, targets, reduction='none'
        )
        loss = (alpha * focal_weight * bce * masks.float()).sum() / total

        # Training regularization: encourage monotonicity
        # Penalize transitions 1→0 (should never happen in monotonic assembly)
        probs = torch.sigmoid(transition_logits)
        reverse_mask = (psr_labels[:, 1:, :] - psr_labels[:, :-1, :]).clamp(max=0).abs() > 0.5
        if reverse_mask.any():
            reg_loss = (probs[:, :-1, :][reverse_mask]).mean() * 0.1
            loss = loss + reg_loss
        else:
            reg_loss = torch.tensor(0.0)

        # Metrics (on raw transitions, not Gaussian-smoothed)
        trans_t = (psr_labels[:, 1:, :] - psr_labels[:, :-1, :]).clamp(min=0)
        pred_t = (torch.sigmoid(transition_logits[:, 1:, :]) > 0.3).float()
        tp = (pred_t * trans_t).sum()
        fp = (pred_t * (1 - trans_t)).sum()
        fn = ((1 - pred_t) * trans_t).sum()
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)

        metrics = {
            'transition_loss': loss.item(),
            'transition_precision': prec.item(),
            'transition_recall': rec.item(),
            'transition_f1': f1.item(),
            'reg_loss': reg_loss.item() if isinstance(reg_loss, torch.Tensor) else reg_loss,
        }

        return loss, metrics
