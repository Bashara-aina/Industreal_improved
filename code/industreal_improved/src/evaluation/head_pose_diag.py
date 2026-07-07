# DEPRECATED 2026-07-07: use eval_pose_kalman.py for reporting numbers.
# This script is for diagnostic/debugging only — do not use for paper figures
# or result tables. The up-vector index bug (using [3:6] instead of [6:9]) was
# fixed at commit bff38b790; eval_pose_kalman.py uses the correct indices.

"""Quick head pose diagnostic — shows forward vs up vs position breakdown."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="src/runs/rf_stages/checkpoints/best.pth")
    parser.add_argument("--max-batches", type=int, default=500)
    parser.add_argument("--ckpt2", default="src/runs/rf_stages/checkpoints/epoch_11.pth")
    args = parser.parse_args()

    from src.models.model import POPWMultiTaskModel

    for ckpt_path, name in [(args.checkpoint, "checkpoint"), (args.ckpt2, "ckpt2")]:
        print(f"\n=== {name}: {ckpt_path} ===")
        ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model = POPWMultiTaskModel(
            pretrained=True,
            backbone_type='convnext_tiny',
            use_hand_film=True,
            use_headpose_film=True,
            use_videomae=False,
            train_pose=False,
        )
        state_dict = {k: v for k, v in ck["model"].items()
                      if 'total_ops' not in k and 'total_params' not in k}
        model.load_state_dict(state_dict, strict=False)
        model._seq_len = 1
        model = model.cuda().eval()

        from src.data.industreal_dataset import IndustRealMultiTaskDataset, collate_fn
        from torch.utils.data import DataLoader
        val_ds = IndustRealMultiTaskDataset(split="val", sequence_mode=False)
        val_loader = DataLoader(val_ds, batch_size=1, num_workers=0,
                                collate_fn=collate_fn, shuffle=False)

        # Per-channel angular MAE
        ch_mae = [0.0] * 9  # per-dimension MAE
        ch_n = 0
        # Per-vector angular MAE (forward=0-2, up=3-5, position=6-8)
        vec_mae = [0.0] * 3
        vec_n = 0

        for i, batch in enumerate(val_loader):
            if i >= args.max_batches:
                break
            images, targets = batch
            images = images.cuda().float()
            if images.max() > 1.0:
                images = images.div_(255.0)
            mean = torch.tensor(_IMAGENET_MEAN, device=images.device).view(1, 3, 1, 1)
            std = torch.tensor(_IMAGENET_STD, device=images.device).view(1, 3, 1, 1)
            images = (images - mean) / std
            with torch.no_grad():
                outputs = model(images)
            hp_p = outputs.get("head_pose").cpu()
            hp_l = targets.get("head_pose").cpu()
            ch_mae[0] += torch.abs(hp_p[0, 0] - hp_l[0, 0]).item()
            ch_mae[1] += torch.abs(hp_p[0, 1] - hp_l[0, 1]).item()
            ch_mae[2] += torch.abs(hp_p[0, 2] - hp_l[0, 2]).item()
            ch_mae[3] += torch.abs(hp_p[0, 3] - hp_l[0, 3]).item()
            ch_mae[4] += torch.abs(hp_p[0, 4] - hp_l[0, 4]).item()
            ch_mae[5] += torch.abs(hp_p[0, 5] - hp_l[0, 5]).item()
            ch_mae[6] += torch.abs(hp_p[0, 6] - hp_l[0, 6]).item()
            ch_mae[7] += torch.abs(hp_p[0, 7] - hp_l[0, 7]).item()
            ch_mae[8] += torch.abs(hp_p[0, 8] - hp_l[0, 8]).item()
            ch_n += 1

            # Angular for forward, up  (forward[0:3], position[3:6], up[6:9])
            for vi, (s, e) in enumerate([(0, 3), (6, 9)]):
                p = hp_p[0, s:e]
                l = hp_l[0, s:e]
                p_n = p / (p.norm() + 1e-8)
                l_n = l / (l.norm() + 1e-8)
                cos = (p_n * l_n).sum().clamp(-1, 1)
                vec_mae[vi] += torch.rad2deg(torch.acos(cos)).item()
            vec_mae[2] += ((hp_p[0, 3:6] - hp_l[0, 3:6]).norm() * 1000).item()  # mm (position, do not report)
            vec_n += 1

        print(f"  Per-dim MAE (raw units):")
        labels = ['fwd_x', 'fwd_y', 'fwd_z', 'up_x', 'up_y', 'up_z', 'pos_x', 'pos_y', 'pos_z']
        for c, l in enumerate(labels):
            print(f"    {l}: {ch_mae[c]/ch_n:.4f}")
        print(f"  Vector MAE:")
        print(f"    forward: {vec_mae[0]/vec_n:.2f}°")
        print(f"    up: {vec_mae[1]/vec_n:.2f}°")
        print(f"    position: {vec_mae[2]/vec_n:.2f} mm")


if __name__ == "__main__":
    main()