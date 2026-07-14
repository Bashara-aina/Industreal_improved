"""FiLM gamma/beta statistics — forward pass over 100 batches.

Computes L2 norm of (gamma - 1) and beta for each FiLM layer
to determine whether FiLM is a pass-through (gamma ~= 1, beta ~= 0)
or actually modulating.

Usage:
    CUDA_VISIBLE_DEVICES=1 python3 src/evaluation/film_gamma_beta_stats.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


# Monkey-patch FiLM forward methods to save gamma/beta as instance attributes
# so we can collect them after each forward pass.
_orig_posefilm_forward = None
_orig_headposefilm_forward = None


def _patched_posefilm_forward(self, c5, keypoints, confidence):
    from src import config as C

    B = keypoints.shape[0]
    scale = torch.tensor(
        [C.IMG_WIDTH, C.IMG_HEIGHT], device=keypoints.device, dtype=keypoints.dtype
    )
    keypoints_norm = keypoints / scale.view(1, 1, 2)
    kp_flat = keypoints_norm.flatten(1)
    conf_flat = confidence.detach()
    pose_flat = torch.cat([kp_flat, conf_flat], dim=1)
    gamma_raw = self.gamma_net(pose_flat)
    beta_raw = self.beta_net(pose_flat)
    gamma = (1.0 + torch.tanh(gamma_raw)).unsqueeze(-1).unsqueeze(-1)
    beta = beta_raw.unsqueeze(-1).unsqueeze(-1)
    # Store for collection
    self._last_gamma = gamma.detach()
    self._last_beta = beta.detach()
    if self.c5_channels != self.FILM_DIM:
        c5_768 = self._c5_proj(c5)
    else:
        c5_768 = c5
    return gamma * c5_768 + beta


def _patched_headposefilm_forward(self, c5_mod, head_pose):
    hp_flat = head_pose.flatten(1)
    gamma_raw = self.gamma_net(hp_flat)
    beta_raw = self.beta_net(hp_flat)
    gamma = (1.0 + torch.tanh(gamma_raw)).unsqueeze(-1).unsqueeze(-1)
    beta = beta_raw.unsqueeze(-1).unsqueeze(-1)
    self._last_gamma = gamma.detach()
    self._last_beta = beta.detach()
    return gamma * c5_mod + beta


@torch.no_grad()
def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="src/runs/rf_stages/checkpoints/best.pth")
    parser.add_argument("--max-batches", type=int, default=100)
    parser.add_argument("--save-dir", default="src/runs/rf_stages/checkpoints")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    from src.models import model as model_module
    from src.data.industreal_dataset import IndustRealMultiTaskDataset, collate_fn
    from torch.utils.data import DataLoader

    # Apply monkey-patches
    global _orig_posefilm_forward, _orig_headposefilm_forward
    _orig_posefilm_forward = model_module.PoseFiLMModule.forward
    _orig_headposefilm_forward = model_module.HeadPoseFiLMModule.forward
    model_module.PoseFiLMModule.forward = _patched_posefilm_forward
    model_module.HeadPoseFiLMModule.forward = _patched_headposefilm_forward

    print("Loading checkpoint...")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    print("Building model...")
    from src.models.model import POPWMultiTaskModel

    model = POPWMultiTaskModel(
        pretrained=True,
        backbone_type="convnext_tiny",
        use_hand_film=True,
        use_headpose_film=True,
        use_videomae=False,
        train_pose=True,
    )
    state_dict = {
        k: v for k, v in ckpt["model"].items() if "total_ops" not in k and "total_params" not in k
    }
    model.load_state_dict(state_dict, strict=False)
    model._seq_len = 1
    model = model.cuda().eval()

    print("Loading val dataset...")
    val_ds = IndustRealMultiTaskDataset(split="val", sequence_mode=False)
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        num_workers=0,
        collate_fn=collate_fn,
        shuffle=False,
    )
    print(f"Val dataset: {len(val_ds)} frames, {len(val_loader)} batches")

    # Identify FiLM modules
    film_module_names = []
    for mod_name, mod in model.named_modules():
        class_name = mod.__class__.__name__
        if class_name in ("PoseFiLMModule", "HeadPoseFiLMModule"):
            film_module_names.append(mod_name)

    print(f"Found FiLM modules: {film_module_names}")

    # Forward pass and collect stats
    film_gammas = {name: [] for name in film_module_names}
    film_betas = {name: [] for name in film_module_names}

    n_batches = 0
    for batch_idx, (images, targets) in enumerate(val_loader):
        if batch_idx >= args.max_batches:
            break
        if images.shape[0] == 0:
            continue

        images_f = images.cuda().float()
        if images_f.max() > 1.0:
            images_f = images_f.div_(255.0)
        mean = torch.tensor(_IMAGENET_MEAN, device=images_f.device).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD, device=images_f.device).view(1, 3, 1, 1)
        images_n = (images_f - mean) / std

        _ = model(images_n)
        n_batches += 1

        for mod_name in film_module_names:
            mod = dict(model.named_modules())[mod_name]
            gamma = mod._last_gamma.squeeze(-1).squeeze(-1).cpu()  # [B, 768]
            beta = mod._last_beta.squeeze(-1).squeeze(-1).cpu()  # [B, 768]
            film_gammas[mod_name].append(gamma)
            film_betas[mod_name].append(beta)

    # Restore original forward methods
    model_module.PoseFiLMModule.forward = _orig_posefilm_forward
    model_module.HeadPoseFiLMModule.forward = _orig_headposefilm_forward

    print(f"\nProcessed {n_batches} batches")

    # Compute statistics per FiLM module
    results = {}
    for mod_name in film_module_names:
        gamma_all = torch.cat(film_gammas[mod_name], dim=0)  # [N, 768]
        beta_all = torch.cat(film_betas[mod_name], dim=0)  # [N, 768]

        gamma_deviation = gamma_all - 1.0  # deviation from identity
        gamma_l2_per_sample = torch.norm(gamma_deviation, dim=1)  # [N]
        beta_l2_per_sample = torch.norm(beta_all, dim=1)  # [N]

        results[mod_name] = {
            "gamma": {
                "mean": float(gamma_all.mean().item()),
                "std": float(gamma_all.std().item()),
                "min": float(gamma_all.min().item()),
                "max": float(gamma_all.max().item()),
                "deviation_from_1_L2_mean": float(gamma_l2_per_sample.mean().item()),
                "deviation_from_1_L2_std": float(gamma_l2_per_sample.std().item()),
                "deviation_from_1_L2_median": float(gamma_l2_per_sample.median().item()),
            },
            "beta": {
                "mean": float(beta_all.mean().item()),
                "std": float(beta_all.std().item()),
                "min": float(beta_all.min().item()),
                "max": float(beta_all.max().item()),
                "L2_mean": float(beta_l2_per_sample.mean().item()),
                "L2_std": float(beta_l2_per_sample.std().item()),
                "L2_median": float(beta_l2_per_sample.median().item()),
            },
            "n_samples": gamma_all.shape[0],
            "n_channels": gamma_all.shape[1],
        }

    # Print summary
    print("\n" + "=" * 60)
    print("FiLM GAMMA/BETA STATISTICS")
    print("=" * 60)
    for mod_name, stats in results.items():
        print(f"\n--- {mod_name} ---")
        g = stats["gamma"]
        b = stats["beta"]
        print(
            f"  Gamma mean={g['mean']:.4f}  std={g['std']:.4f}  "
            f"range=[{g['min']:.4f}, {g['max']:.4f}]"
        )
        print(
            f"  Gamma |dev-1| L2:  mean={g['deviation_from_1_L2_mean']:.4f}  "
            f"median={g['deviation_from_1_L2_median']:.4f}"
        )
        print(
            f"  Beta   mean={b['mean']:.4f}  std={b['std']:.4f}  "
            f"range=[{b['min']:.4f}, {b['max']:.4f}]"
        )
        print(f"  Beta L2 norm:     mean={b['L2_mean']:.4f}  median={b['L2_median']:.4f}")

    # Overall verdict
    print("\n" + "-" * 60)
    all_gamma_dev_l2 = np.mean([s["gamma"]["deviation_from_1_L2_mean"] for s in results.values()])
    all_beta_l2 = np.mean([s["beta"]["L2_mean"] for s in results.values()])
    print(f"Across all FiLM layers:")
    print(f"  Mean gamma |deviation from 1| L2: {all_gamma_dev_l2:.4f}")
    print(f"  Mean beta L2:                    {all_beta_l2:.4f}")
    print(f"  FILM_DIM = 768 channels")
    print(f"  At 768 channels, L2=27.7 would mean average deviation of ~1.0 per channel.")
    if all_gamma_dev_l2 < 10.0 and all_beta_l2 < 10.0:
        verdict = "PASS-THROUGH: gamma ~= 1, beta ~= 0. FiLM layers are effectively identity."
    elif all_gamma_dev_l2 < 50.0 and all_beta_l2 < 50.0:
        verdict = "WEAK MODULATION: Non-trivial but small relative to 768-dim feature scale."
    else:
        verdict = (
            "ACTIVE MODULATION: Large gamma/beta values. "
            "FiLM layers significantly modulate features."
        )
    print(f"  VERDICT: {verdict}")
    results["verdict"] = verdict

    out_path = save_dir / "film_gamma_beta.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
