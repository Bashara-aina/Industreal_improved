"""
psr_loss_diagnostic.py
======================

Goal
----
Find out why PSR training loss sits at exactly `psr=0.0001000` on ~18,615/18,635
steps while NO `PSR_NAN` warning is ever logged. The MASTER_PROMPT assumed
"non-finite loss floored to 1e-4 by `_safe`", but the warning that guards that
path never fires -- so the real mechanism is something else.

This file does three things:

  1. `binary_focal_loss_EXACT`  -- a verbatim copy of the CURRENT function in
     losses.py (the one with the `-1` masking "fix"). No edits, so we measure
     the real behaviour.

  2. `replay_downstream_chain`  -- faithfully reproduces the tail of
     MultiTaskLoss.forward that runs AFTER the PSR loss is computed:
         loss_psr  -> _smooth_cap(loss_psr, PSR_LOSS_CAP)
                   -> PSR_NAN finiteness check  (the warning that never fires)
                   -> _safe(loss_psr)           (the 1e-4 sentinel)
                   -> clamp(min=0)
     so we can see, per scenario, BOTH the displayed `psr=` value AND whether
     the warning would have fired.

  3. A scenario sweep over the conditions seen in the real run, run under both
     fp32 and fp16, to discriminate between the competing hypotheses:
        H1  non-finite loss  -> sentinel  (warning SHOULD fire)
        H2  dilution: `.mean()` divides by ALL elements incl. masked `-1`s, so a
            high `-1` fraction drives a genuinely-finite loss toward ~1e-4
            (warning does NOT fire -- matches the observation)

The instrumented drop-in for the real run is at the BOTTOM of the file
(`binary_focal_loss_instrumented`). It returns the RAW value (no sentinel) and
dumps the offending tensors through BOTH logging and a flushed print, so the
next real run reveals the trigger in stdout no matter how logging is wired.

Run:  python psr_loss_diagnostic.py
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Constants mirrored from config.py (the values the real run uses)
# ---------------------------------------------------------------------------
NUM_PSR_COMPONENTS = 11
PSR_FOCAL_ALPHA = 0.25
PSR_FOCAL_GAMMA = 1.0          # config.PSR_FOCAL_GAMMA (NOT 2.0)
PSR_LOSS_CAP = 20.0            # config.PSR_LOSS_CAP
BATCH = 6                      # config.BATCH_SIZE (T=1 normal batch -> [6, 11])

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("psr_diag")


# ===========================================================================
# 1. EXACT copy of the current binary_focal_loss (verbatim from losses.py)
# ===========================================================================
def binary_focal_loss_EXACT(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    per_component_alpha: torch.Tensor = None,
) -> torch.Tensor:
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)

    if (targets < 0).any():
        ignore_mask = (targets < 0).float()
        targets_safe = targets.clone().masked_fill_(ignore_mask.bool(), 0)
    else:
        ignore_mask = None
        targets_safe = targets

    if per_component_alpha is not None:
        alpha_c = per_component_alpha.to(logits.device).unsqueeze(0)
        alpha_t = alpha_c * targets_safe + (1 - alpha_c) * (1 - targets_safe)
    else:
        alpha_t = alpha * targets_safe + (1 - alpha) * (1 - targets_safe)

    if ignore_mask is not None:
        alpha_t = alpha_t * (1 - ignore_mask)
        p_t = p_t.masked_fill(ignore_mask.bool(), 1.0)
        ce = ce.masked_fill(ignore_mask.bool(), 0.0)

    p_t = p_t.clamp(min=1e-7, max=1.0 - 1e-7)
    return (alpha_t * (1 - p_t) ** gamma * ce).mean()


# ===========================================================================
# 2. Faithful replay of the MultiTaskLoss downstream chain
# ===========================================================================
def _smooth_cap(x: torch.Tensor, cap: float) -> torch.Tensor:
    """Verbatim from MultiTaskLoss.forward."""
    x_safe = x.clamp(min=1e-6, max=1e6)
    return torch.where(x > cap, cap * (1 + torch.log(x_safe / cap)), x.clamp(min=1e-6))


def replay_downstream_chain(loss_psr: torch.Tensor) -> dict:
    """
    Reproduce what happens to loss_psr after binary_focal_loss returns, and
    report the value the progress bar would print plus whether PSR_NAN fires.
    """
    device = loss_psr.device
    zero = torch.tensor(0.0, device=device, dtype=loss_psr.dtype)

    # smooth cap (losses.py ~1073)
    capped = _smooth_cap(loss_psr, PSR_LOSS_CAP)

    # PSR_NAN warning gate (losses.py ~1089) -- the warning that never fires
    warning_fires = not torch.isfinite(capped).all()

    # _safe sentinel (losses.py ~1084 / applied ~1094)
    if not torch.isfinite(capped).all():
        safed = torch.tensor(1e-4, device=device, dtype=capped.dtype)
    else:
        safed = torch.where(capped < 0, zero, capped)

    # final clamp(min=0) inside Kendall total (losses.py ~1158)
    final = safed.clamp(min=0.0)

    return {
        "raw_focal": float(loss_psr),
        "after_smooth_cap": float(capped),
        "PSR_NAN_warning_fires": bool(warning_fires),
        "displayed_psr": float(final),       # what tqdm prints as psr=...
        "is_exactly_1e-4": abs(float(final) - 1e-4) < 5e-9,
    }


# ===========================================================================
# 3. Scenario builder + sweep
# ===========================================================================
def make_psr_batch(
    neg_one_frac: float,
    pos_frac_of_valid: float = 0.10,
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
):
    """
    Build a realistic PSR batch.

    logits: frozen in the OBSERVED band [-1.5, -0.5] (the dead-head signature
            from eval: sigmoid 0.18-0.38, unique_binary_patterns=1).
    targets in {0, 1, -1}:
        - `neg_one_frac` of all entries are -1 (error/aborted/not-reached)
        - of the remaining VALID entries, `pos_frac_of_valid` are 1, rest 0
    """
    g = torch.Generator().manual_seed(seed)
    n = BATCH * NUM_PSR_COMPONENTS

    logits = (torch.rand(n, generator=g) * (-1.5 - (-0.5)) + (-0.5)).reshape(
        BATCH, NUM_PSR_COMPONENTS
    )  # uniform in [-1.5, -0.5]

    targets = torch.zeros(n)
    perm = torch.randperm(n, generator=g)
    n_neg1 = int(round(neg_one_frac * n))
    neg1_idx = perm[:n_neg1]
    valid_idx = perm[n_neg1:]
    n_pos = int(round(pos_frac_of_valid * valid_idx.numel()))
    pos_idx = valid_idx[:n_pos]
    targets[neg1_idx] = -1.0
    targets[pos_idx] = 1.0
    targets = targets.reshape(BATCH, NUM_PSR_COMPONENTS)

    return logits.to(dtype), targets.to(dtype)


def per_component_alpha_from_prevalence(prevalence: torch.Tensor) -> torch.Tensor:
    """set_psr_class_counts: alpha_c = 2*(1 - clamp(prev, 0.01, 0.99))."""
    prev = prevalence.float().clamp(0.01, 0.99)
    return 2.0 * (1.0 - prev)


def run_sweep():
    # A plausible per-component prevalence (component 0 ~95%, tail rare)
    prevalence = torch.tensor(
        [0.95, 0.90, 0.70, 0.55, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15]
    )
    pc_alpha = per_component_alpha_from_prevalence(prevalence)

    neg1_fracs = [0.0, 0.30, 0.60, 0.90, 0.98, 1.0]
    alpha_modes = [("scalar", None), ("per_component", pc_alpha)]
    dtypes = [("fp32", torch.float32), ("fp16", torch.float16)]

    header = (
        f"{'dtype':5} {'alpha':14} {'-1 frac':8} "
        f"{'raw_focal':>12} {'displayed':>12} {'warn?':>6} {'==1e-4?':>8} {'finite?':>8}"
    )
    print(header)
    print("-" * len(header))

    for dt_name, dt in dtypes:
        for a_name, a_tensor in alpha_modes:
            for nf in neg1_fracs:
                logits, targets = make_psr_batch(nf, dtype=dt)
                # mimic AMP: focal math under fp16 if requested
                loss = binary_focal_loss_EXACT(
                    logits,
                    targets,
                    alpha=PSR_FOCAL_ALPHA,
                    gamma=PSR_FOCAL_GAMMA,
                    per_component_alpha=(a_tensor.to(dt) if a_tensor is not None else None),
                )
                finite = bool(torch.isfinite(loss).all())
                chain = replay_downstream_chain(loss.float())
                print(
                    f"{dt_name:5} {a_name:14} {nf:<8.2f} "
                    f"{chain['raw_focal']:>12.7f} {chain['displayed_psr']:>12.7f} "
                    f"{str(chain['PSR_NAN_warning_fires']):>6} "
                    f"{str(chain['is_exactly_1e-4']):>8} {str(finite):>8}"
                )

    print()
    print("Reading the table:")
    print("  * 'raw_focal'  = value binary_focal_loss returns (before any guard)")
    print("  * 'displayed'  = value tqdm prints as psr=...  (after smooth_cap+_safe)")
    print("  * 'warn?'      = whether the PSR_NAN logger.warning would fire")
    print("  * '==1e-4?'    = displayed value rounds to exactly 0.0001000")
    print("  * 'finite?'    = raw focal loss is finite")


# ===========================================================================
# 4. Instrumented drop-in for the REAL run (paste into losses.py)
# ===========================================================================
# Replace the body of binary_focal_loss in losses.py with this. It returns the
# RAW value (no sentinel) and, whenever the result is non-finite OR suspiciously
# small (< 1e-3, the dead-PSR signature), dumps the inputs through BOTH
# logging and a flushed print so the trigger is visible regardless of how the
# root logger is wired. Set PSR_DIAG_EVERY to throttle once you have a few hits.
PSR_DIAG_EVERY = 0          # 0 = log every offending batch; N = every Nth call
_psr_diag_call = 0


def binary_focal_loss_instrumented(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    per_component_alpha: torch.Tensor = None,
) -> torch.Tensor:
    global _psr_diag_call
    _psr_diag_call += 1

    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)

    neg_mask = targets < 0
    n_neg1 = int(neg_mask.sum())
    if neg_mask.any():
        ignore_mask = neg_mask.float()
        targets_safe = targets.clone().masked_fill_(ignore_mask.bool(), 0)
    else:
        ignore_mask = None
        targets_safe = targets

    if per_component_alpha is not None:
        alpha_c = per_component_alpha.to(logits.device).unsqueeze(0)
        alpha_t = alpha_c * targets_safe + (1 - alpha_c) * (1 - targets_safe)
    else:
        alpha_t = alpha * targets_safe + (1 - alpha) * (1 - targets_safe)

    if ignore_mask is not None:
        alpha_t = alpha_t * (1 - ignore_mask)
        p_t = p_t.masked_fill(ignore_mask.bool(), 1.0)
        ce = ce.masked_fill(ignore_mask.bool(), 0.0)

    p_t = p_t.clamp(min=1e-7, max=1.0 - 1e-7)
    per_elem = alpha_t * (1 - p_t) ** gamma * ce
    loss = per_elem.mean()

    # ---- diagnostic ----
    total = targets.numel()
    valid = total - n_neg1
    suspicious = (not torch.isfinite(loss).all()) or (float(loss) < 1e-3)
    throttled = (PSR_DIAG_EVERY == 0) or (_psr_diag_call % max(PSR_DIAG_EVERY, 1) == 0)
    if suspicious and throttled:
        msg = (
            "[PSR_DIAG] "
            f"loss={float(loss):.3e} finite={bool(torch.isfinite(loss).all())} | "
            f"shape={tuple(targets.shape)} total={total} valid={valid} neg1={n_neg1} "
            f"(neg1_frac={n_neg1 / max(total, 1):.3f}) | "
            f"logits[min/max/mean]={float(logits.min()):.3f}/"
            f"{float(logits.max()):.3f}/{float(logits.mean()):.3f} | "
            f"target counts: zeros={int((targets == 0).sum())} "
            f"ones={int((targets == 1).sum())} neg1={n_neg1} | "
            f"per_elem[min/max/sum]={float(per_elem.min()):.3e}/"
            f"{float(per_elem.max()):.3e}/{float(per_elem.sum()):.3e} | "
            f"gamma={gamma} pc_alpha={'yes' if per_component_alpha is not None else 'no'}"
        )
        logger.warning(msg)
        print(msg, flush=True)   # guaranteed-captured second channel

    return loss   # RAW value -- no sentinel, so the real run shows the truth


if __name__ == "__main__":
    print("=" * 88)
    print("PSR binary_focal_loss diagnostic -- exact function + full downstream chain")
    print("=" * 88)
    print(f"gamma={PSR_FOCAL_GAMMA}  alpha={PSR_FOCAL_ALPHA}  batch={BATCH}  "
          f"components={NUM_PSR_COMPONENTS}  PSR_LOSS_CAP={PSR_LOSS_CAP}")
    print()
    run_sweep()
