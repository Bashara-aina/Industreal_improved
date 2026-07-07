#!/usr/bin/env python3
"""
PSRHead Activation Diagnostic v2 (Opus §-1c/d, §Q8)

Measures activation statistics across all PSRHead components to determine
whether GELU saturation is killing gradient flow at the output heads or
upstream transformer variance collapse is the root cause.

Diagnostic verdicts:
  - GELU saturated (zero-fraction > 0.9 for most heads):
    Activation is fine; fix upstream transformer variance-restoring init.
  - GELU alive but low variance:
    Issue is post-GELU; try LeakyReLU(0.01) + bias=0.0.
  - Outputs constant:
    Deeper issue; investigate input_dim / per-frame MLP collapse.
"""
import sys
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
NUM_COMPONENTS = 11
DIAG_BATCHES = 10

# Module-level storage for hook captures
_hook_data = {
    'transformer_out': [],   # list of tensors
    'head0_pre_gelu': [],    # list of lists: [batch][comp]
    'head0_post_gelu': [],
    'head0_logits_pre_sigmoid': [],
}


def _make_hooks(num_components):
    """Create forward hooks that capture intermediate activations per component."""
    # We need separate hooks for each component
    pre_gelu_data = [[] for _ in range(num_components)]
    post_gelu_data = [[] for _ in range(num_components)]
    logits_data = [[] for _ in range(num_components)]
    transformer_out = []

    def make_pre_gelu_hook(comp_idx):
        def hook(module, input, output):
            pre_gelu_data[comp_idx].append(output.detach().cpu())
        return hook

    def make_post_gelu_hook(comp_idx):
        def hook(module, input, output):
            post_gelu_data[comp_idx].append(output.detach().cpu())
        return hook

    def make_logits_hook(comp_idx):
        """Capture final logit before returning from the component head."""
        def hook(module, input, output):
            logits_data[comp_idx].append(output.detach().cpu())
        return hook

    def transformer_hook(module, input, output):
        transformer_out.append(output.detach().cpu())

    return {
        'pre_gelu_data': pre_gelu_data,
        'post_gelu_data': post_gelu_data,
        'logits_data': logits_data,
        'transformer_out': transformer_out,
        'pre_gelu_hooks': [],
        'post_gelu_hooks': [],
        'logits_hooks': [],
        'transformer_hook_handle': None,
        'num_components': num_components,
        'make_pre_gelu_hook': make_pre_gelu_hook,
        'make_post_gelu_hook': make_post_gelu_hook,
        'make_logits_hook': make_logits_hook,
        'transformer_hook': transformer_hook,
    }


def register_hooks(model, hook_state):
    """Register forward hooks on PSRHead components."""
    psr_head = model.psr_head

    # Transformer output hook
    hook_state['transformer_hook_handle'] = psr_head.transformer.register_forward_hook(
        hook_state['transformer_hook']
    )

    # Per-component head hooks
    for i in range(hook_state['num_components']):
        comp = psr_head.output_heads[i]
        # comp[0] = Linear(256, 64)
        # comp[1] = GELU
        # comp[2] = Dropout
        # comp[3] = Linear(64, 1)

        # Pre-GELU: after Linear(256,64), before GELU
        h1 = comp[0].register_forward_hook(hook_state['make_pre_gelu_hook'](i))
        hook_state['pre_gelu_hooks'].append(h1)

        # Post-GELU: after GELU, before dropout
        h2 = comp[1].register_forward_hook(hook_state['make_post_gelu_hook'](i))
        hook_state['post_gelu_hooks'].append(h2)

        # Final logits: after Linear(64,1)
        h3 = comp[3].register_forward_hook(hook_state['make_logits_hook'](i))
        hook_state['logits_hooks'].append(h3)

    return hook_state


def remove_hooks(hook_state):
    """Remove all registered hooks."""
    for h in hook_state['pre_gelu_hooks']:
        h.remove()
    for h in hook_state['post_gelu_hooks']:
        h.remove()
    for h in hook_state['logits_hooks']:
        h.remove()
    if hook_state['transformer_hook_handle'] is not None:
        hook_state['transformer_hook_handle'].remove()


def compute_stats(tensor: torch.Tensor, name: str) -> dict:
    """Compute diagnostic statistics for a tensor."""
    if tensor.numel() == 0:
        return {'name': name, 'mean': float('nan'), 'std': float('nan'),
                'min': float('nan'), 'max': float('nan'), 'frac_zero': float('nan')}

    t = tensor.flatten()
    frac_zero = (t.abs() < 1e-8).float().mean().item()
    return {
        'name': name,
        'mean': t.mean().item(),
        'std': t.std().item(),
        'min': t.min().item(),
        'max': t.max().item(),
        'frac_zero': frac_zero,
    }


def print_stats(stats: dict):
    """Print a single stats dict nicely."""
    print(f"  {stats['name']:40s}  mean={stats['mean']:10.6f}  "
          f"std={stats['std']:10.6f}  min={stats['min']:10.6f}  "
          f"max={stats['max']:10.6f}  zero_frac={stats['frac_zero']:.4f}")


def analyze_batch(hook_state, batch_idx: int):
    """Analyze captured activations for one batch and print results."""
    nc = hook_state['num_components']

    print(f"\n{'='*80}")
    print(f"BATCH {batch_idx}")
    print(f"{'='*80}")

    # --- Transformer output ---
    tx_out = hook_state['transformer_out'][-1] if hook_state['transformer_out'] else None
    if tx_out is not None:
        stats = compute_stats(tx_out, f'encoded (transformer output) [{list(tx_out.shape)}]')
        print_stats(stats)
        encoded_std = stats['std']
    else:
        print("  [WARN] No transformer output captured")
        encoded_std = float('nan')

    # --- Per-component stats ---
    dead_components = []
    alive_components = []

    for i in range(nc):
        post_gelu = hook_state['post_gelu_data'][i][-1] if hook_state['post_gelu_data'][i] else None
        pre_gelu = hook_state['pre_gelu_data'][i][-1] if hook_state['pre_gelu_data'][i] else None
        logits = hook_state['logits_data'][i][-1] if hook_state['logits_data'][i] else None

        if post_gelu is None:
            print(f"  [WARN] Component {i}: no post-GELU data")
            continue

        pg_stats = compute_stats(post_gelu, f'comp[{i:02d}] post-GELU [{list(post_gelu.shape)}]')
        print_stats(pg_stats)

        if pre_gelu is not None:
            pre_stats = compute_stats(pre_gelu, f'comp[{i:02d}] pre-GELU  [{list(pre_gelu.shape)}]')
            # Print pre-GELU on the same line with less detail
            print(f"  {'':40s}  pre-GELU mean={pre_stats['mean']:10.6f} std={pre_stats['std']:10.6f}")

        if logits is not None:
            logit_stats = compute_stats(logits, f'comp[{i:02d}] logits   [{list(logits.shape)}]')
            print(f"  {'':40s}  logits  mean={logit_stats['mean']:10.6f} std={logit_stats['std']:10.6f}")

        if pg_stats['frac_zero'] > 0.9:
            dead_components.append(i)
        else:
            alive_components.append(i)

    print(f"\n  --- Summary batch {batch_idx} ---")
    print(f"  encoded.std() = {encoded_std:.6f}")
    print(f"  Dead components (GELU zero-frac > 0.9): {dead_components}")
    print(f"  Alive components: {alive_components}")

    return {
        'batch': batch_idx,
        'encoded_std': encoded_std,
        'dead_components': dead_components,
        'alive_components': alive_components,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='PSRHead activation diagnostic v2')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint .pth file')
    parser.add_argument('--num-batches', type=int, default=DIAG_BATCHES,
                        help='Number of batches to forward')
    parser.add_argument('--sequence-length', type=int, default=8,
                        help='Sequence length for PSR temporal context')
    parser.add_argument('--batch-size', type=int, default=2,
                        help='Batch size (number of sequences)')
    args = parser.parse_args()

    # Determine checkpoint path
    if args.checkpoint is None:
        # Default: use rf_stages best.pth
        ckpt_path = Path(__file__).resolve().parent.parent.parent / 'src' / 'runs' / 'rf_stages' / 'checkpoints' / 'best.pth'
    else:
        ckpt_path = Path(args.checkpoint)

    if not ckpt_path.exists():
        print(f"FATAL: Checkpoint not found at {ckpt_path}")
        sys.exit(1)

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
    print(f"  Epoch: {ckpt.get('epoch', '?')}, best_metric: {ckpt.get('best_metric', '?')}")

    # Build model
    from src.models.model import POPWMultiTaskModel
    model = POPWMultiTaskModel(
        pretrained=True,
        backbone_type='convnext_tiny',
        use_hand_film=True,
        use_headpose_film=True,
        use_videomae=False,
        train_pose=False,
    )
    state_dict = {k: v for k, v in ckpt['model'].items()
                  if 'total_ops' not in k and 'total_params' not in k}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"  Missing keys: {len(missing)}, Unexpected: {len(unexpected)}")

    # Set sequence length hint so model forward routes through PSR sequence path
    model._seq_len = args.sequence_length
    model = model.cuda().eval()

    # Register hooks on PSRHead
    hook_state = _make_hooks(NUM_COMPONENTS)
    register_hooks(model, hook_state)

    # Load validation dataset
    from src.data.industreal_dataset import IndustRealMultiTaskDataset, collate_fn_sequences
    from torch.utils.data import DataLoader

    # Use sequence mode to get temporal context for PSR transformer
    val_ds = IndustRealMultiTaskDataset(
        split='val',
        sequence_mode=True,
        sequence_length=args.sequence_length * args.batch_size,  # enough frames
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,  # sequences must stay together
        num_workers=0,
        collate_fn=collate_fn_sequences,
        shuffle=True,
    )

    print(f"\nVal dataset has {len(val_ds)} sequences, starting diagnostic...")
    print(f"Sequence length: {args.sequence_length}, Batch size: {args.batch_size}")
    print(f"Will forward {args.num_batches} batches")

    all_results = []
    batch_count = 0
    iterator = iter(val_loader)

    while batch_count < args.num_batches:
        try:
            batch = next(iterator)
        except StopIteration:
            print("  [INFO] Dataset exhausted, restarting iterator")
            iterator = iter(val_loader)
            continue

        # collate_fn_sequences returns (images, targets) or a dict
        # Handle both return formats
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            images, targets = batch
        elif isinstance(batch, dict):
            images = batch['images']['rgb']
            targets = batch
        else:
            print(f"  [WARN] Unexpected batch type: {type(batch)}, skipping")
            continue

        # images shape: typically [B, T, 3, H, W] or [1, T, 3, H, W]
        if images.dim() == 4:
            # Single frame mode — add seq dim
            images = images.unsqueeze(1)  # [B, 1, 3, H, W]
        elif images.dim() == 5:
            # Already [B, T, 3, H, W] — fine
            pass
        else:
            print(f"  [WARN] Unexpected image dims: {images.shape}, skipping")
            continue

        B = images.shape[0]
        T = images.shape[1]

        # If batch is too large, take a slice
        if B > args.batch_size:
            images = images[:args.batch_size]
            B = args.batch_size

        # Collapse B and T into single batch dimension for backbone
        # The model's forward will handle this via _seq_len
        images_flat = images.reshape(B * T, *images.shape[2:])  # [B*T, 3, H, W]

        # Normalize
        images_flat = images_flat.cuda().float()
        if images_flat.max() > 1.0:
            images_flat = images_flat.div_(255.0)
        mean = torch.tensor(_IMAGENET_MEAN, device=images_flat.device).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD, device=images_flat.device).view(1, 3, 1, 1)
        images_flat = (images_flat - mean) / std

        with torch.no_grad():
            # Forward through full model
            # We need the model to route through PSR sequence path
            # This requires setting _seq_len on the model before forward
            model._seq_len = T
            outputs = model(images_flat)

        # Analyze captured activations
        result = analyze_batch(hook_state, batch_count)
        all_results.append(result)
        batch_count += 1

        # Print a quick separator
        if batch_count < args.num_batches:
            print(f"\n  --- Proceeding to batch {batch_count} ---")

    # --- Final aggregate summary ---
    print(f"\n\n{'='*80}")
    print(f"FINAL AGGREGATE REPORT (across {len(all_results)} batches)")
    print(f"{'='*80}")

    encoded_stds = [r['encoded_std'] for r in all_results]
    all_dead = set()
    all_alive = set(range(NUM_COMPONENTS))

    for r in all_results:
        all_dead.update(r['dead_components'])
        all_alive.intersection_update(r['alive_components'])

    print(f"\nTransformer output:")
    print(f"  encoded.std() — mean: {np.mean(encoded_stds):.6f}, "
          f"min: {min(encoded_stds):.6f}, max: {max(encoded_stds):.6f}")

    # Per-component aggregate
    print(f"\nPer-component GELU saturation (zero-frac > 0.9):")
    n_dead_across = sum(1 for i in range(NUM_COMPONENTS) if i in all_dead)

    for i in range(NUM_COMPONENTS):
        # Collect all post-GELU activations for this component across batches
        post_gelu_all = []
        for batch_idx in range(min(len(hook_state['post_gelu_data'][i]), len(all_results))):
            post_gelu_all.append(hook_state['post_gelu_data'][i][batch_idx])

        if post_gelu_all:
            all_t = torch.cat([p.flatten() for p in post_gelu_all])
            frac_zero = (all_t.abs() < 1e-8).float().mean().item()
            mean_val = all_t.mean().item()
            std_val = all_t.std().item()

            status = "DEAD" if frac_zero > 0.9 else "ALIVE"
            print(f"  comp[{i:02d}]: mean={mean_val:10.6f} std={std_val:10.6f} "
                  f"zero_frac={frac_zero:.4f} [{status}]")
        else:
            print(f"  comp[{i:02d}]: NO DATA")

    print(f"\nVerdict:")
    print(f"  Total dead components: {len(all_dead)} / {NUM_COMPONENTS}")

    if len(all_dead) > int(NUM_COMPONENTS * 0.5):
        print(f"  -> GELU is SATURATED (majority dead). Issue is upstream transformer variance.")
        print(f"     Repair: Fix transformer variance-restoring init or increase dropout.")
        print(f"     The +0.1 bias on Linear(256,64) is insufficient.")
    elif encoded_stds and np.mean(encoded_stds) < 0.01:
        print(f"  -> Transformer variance is COLLAPSED (mean std={np.mean(encoded_stds):.6f}).")
        print(f"     Even if GELU is alive, the signal into output heads is too weak.")
        print(f"     Repair: LeakyReLU(0.01) + bias=0.0 in output heads, AND")
        print(f"     fix transformer variance (increase d_model dropout or use LayerNorm after transformer).")
    elif len(all_dead) > 0:
        print(f"  -> Some GELU heads saturated, some alive. Mixed picture.")
        print(f"     Repair: LeakyReLU(0.01) + bias=0.0 on output heads, or individual variance scaling.")
    else:
        print(f"  -> All GELU heads ALIVE. Issue may be in the final Linear(64,1) or elsewhere.")
        print(f"     Check per-head logits variance and gradient flow at training time.")

    # Cleanup
    remove_hooks(hook_state)
    print(f"\nDiagnostic complete.")


if __name__ == '__main__':
    main()
