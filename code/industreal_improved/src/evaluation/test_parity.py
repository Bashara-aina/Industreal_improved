"""Parity test: train-val vs subprocess vs TTA paths produce identical metrics on the same 50 batches.

[Opus 126 §5.3] Would have caught the D3 NaN, TTA broken run, and act_top5=0.0 bug
before they burned three GPU runs. Run pre-submission: 5-10 minutes.

Usage: python3 src/evaluation/test_parity.py [--ckpt path] [--n_batches 50]
"""

import json
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def run_parity_check(
    ckpt_path: str = "src/runs/rf_stages/checkpoints/best.pth", n_batches: int = 50
) -> dict:
    """Run the same 50 batches through all three eval paths and compare.

    Returns dict with {path: metrics, parity: bool, max_diff: float}.
    """
    from torch.utils.data import DataLoader
    from src.data.industreal_dataset import IndustRealMultiTaskDataset, collate_fn

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    model_state = ckpt["model"]

    # Build val dataset (first n_batches only)
    val_ds = IndustRealMultiTaskDataset(split="val")
    val_loader = DataLoader(
        val_ds, batch_size=4, shuffle=False, num_workers=0, collate_fn=collate_fn
    )
    # Truncate to n_batches
    limited_batches = []
    for i, batch in enumerate(val_loader):
        if i >= n_batches:
            break
        limited_batches.append(batch)

    print(f"Running parity check on {len(limited_batches)} batches, ckpt={ckpt_path}")
    print(f"  this may take 1-3 minutes on RTX 3060...")

    # Build a minimal model wrapper that loads weights
    # For now, just test that the eval entry points exist and return consistent keys
    # (full parity requires loading the full POPWMultiTaskModel — out of scope for this patch)

    # TODO: full implementation when POPWMultiTaskModel import is verified
    return {
        "n_batches": len(limited_batches),
        "ckpt": ckpt_path,
        "parity_check_placeholder": True,
        "message": "Parity test stub. Full implementation pending POPWMultiTaskModel wrapper.",
    }


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="src/runs/rf_stages/checkpoints/best.pth")
    p.add_argument("--n_batches", type=int, default=50)
    args = p.parse_args()
    result = run_parity_check(args.ckpt, args.n_batches)
    print(json.dumps(result, indent=2))
