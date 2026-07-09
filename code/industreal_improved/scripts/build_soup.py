#!/usr/bin/env python3
"""Build model soup by averaging backbone weights from Phase 2 single-task specialists.

[OPUS 192 §5 step 8] Near-free init increment for the MTL finetune. If
soup'd backbone loss is lower than cold init, keep it. Otherwise drop.

Wortsman 2022: averaging weights of fine-tuned models often improves accuracy.
For our 4-task MTL: averaging 3-4 single-task backbone weights (det, act, psr,
+pose) produces a backbone that's "good for all tasks" — a strong init for
MTL finetune.

Note: 192 Q5 caveat — Wortsman soups average models fine-tuned on the SAME
task; we average backbones fine-tuned on FOUR different objectives, which
may land between basins. So this is an experiment, not a dependency.

Usage:
    python scripts/build_soup.py \
        --det /path/to/st_det/best.pt \
        --act /path/to/st_act/best.pt \
        --psr /path/to/st_psr/best.pt \
        --pose /path/to/st_pose/best.pt \
        --output /path/to/soup_backbone.pt
"""
import argparse
import sys
from collections import OrderedDict
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description="Build model soup from specialist backbones")
    parser.add_argument("--det", type=str, default=None, help="Path to ST-det best.pt")
    parser.add_argument("--act", type=str, default=None, help="Path to ST-act best.pt")
    parser.add_argument("--psr", type=str, default=None, help="Path to ST-psr best.pt")
    parser.add_argument("--pose", type=str, default=None, help="Path to ST-pose best.pt")
    parser.add_argument("--output", type=str, required=True, help="Output path for soup backbone")
    parser.add_argument("--backbone-key", type=str, default="model_state_dict",
                        help="Top-level key in checkpoint (default: model_state_dict)")
    parser.add_argument("--backbone-prefix", type=str, default="feature_pyramid.backbone",
                        help="Prefix for backbone state dict keys")
    args = parser.parse_args()

    # Collect available specialists
    specs = []
    spec_names = []
    for name, path in [("det", args.det), ("act", args.act),
                       ("psr", args.psr), ("pose", args.pose)]:
        if path is None:
            print(f"  Skipping {name} (not provided)")
            continue
        if not Path(path).exists():
            print(f"  WARNING: {name} checkpoint not found: {path}")
            continue
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        sd = ckpt.get(args.backbone_key, ckpt)
        # Filter to backbone keys
        backbone_sd = {k: v for k, v in sd.items()
                       if k.startswith(args.backbone_prefix)}
        if not backbone_sd:
            print(f"  WARNING: no keys with prefix '{args.backbone_prefix}' in {path}")
            continue
        # Strip the prefix so the saved soup has the right shape for direct load
        backbone_sd = OrderedDict(
            (k[len(args.backbone_prefix) + 1:] if k.startswith(args.backbone_prefix + ".")
             else k, v)
            for k, v in backbone_sd.items()
        )
        specs.append(backbone_sd)
        spec_names.append(name)
        print(f"  Loaded {name} from {path} ({len(backbone_sd)} backbone tensors)")

    if not specs:
        print("ERROR: no specialist checkpoints provided")
        sys.exit(1)

    if len(specs) == 1:
        print("WARNING: only 1 specialist — no averaging to do. Saving as-is.")
        avg_state = specs[0]
    else:
        # Average backbone weights (uniform)
        avg_state = OrderedDict()
        all_keys = set()
        for sd in specs:
            all_keys.update(sd.keys())
        for key in all_keys:
            tensors = []
            dtypes = []
            for sd in specs:
                if key in sd:
                    tensors.append(sd[key].float())
                    dtypes.append(sd[key].dtype)
            if not tensors:
                print(f"  WARNING: key {key} not in any specialist, skipping")
                continue
            stacked = torch.stack(tensors)
            avg = stacked.mean(dim=0)
            # Cast back to original dtype (typically float32)
            avg = avg.to(dtypes[0])
            avg_state[key] = avg

    # Wrap with the backbone_prefix so it loads directly into model.backbone
    wrapped = OrderedDict()
    for k, v in avg_state.items():
        wrapped[f"{args.backbone_prefix}.{k}"] = v

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(wrapped, output_path)
    print(f"\nSoup backbone saved: {output_path}")
    print(f"  Averaged {len(specs)} specialists: {spec_names}")
    print(f"  Total tensors: {len(avg_state)}")


if __name__ == "__main__":
    main()
