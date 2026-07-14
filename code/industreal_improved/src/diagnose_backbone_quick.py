"""
Quick backbone diagnostic: compare real images vs random noise features.
"""

import sys, os, torch
from pathlib import Path

SRC = Path(__file__).parent
os.chdir(SRC.parent)
sys.path.insert(0, str(SRC))

from src import config as cfg

cfg.apply_preset("stage_rf1")
cfg.update_dynamic_paths()

# Load model
from src.models.model import POPWMultiTaskModel

ckpt = torch.load(
    SRC / "runs" / "rf_stages" / "checkpoints" / "crash_recovery.pth",
    map_location="cpu",
    weights_only=False,
)
model = POPWMultiTaskModel(
    pretrained=False,
    backbone_type="convnext_tiny",
    use_headpose_film=False,
    use_hand_film=False,
    use_videomae=False,
    train_pose=False,
    use_backbone_checkpoint=False,
)
model.load_state_dict(ckpt["model"], strict=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()
print(f"Device: {device}")

# Get one real batch
from data.industreal_dataset import IndustRealMultiTaskDataset
from torch.utils.data import DataLoader

ds = IndustRealMultiTaskDataset(split="val", img_size=cfg.IMG_SIZE, augment=False, max_recordings=2)
loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
batch = next(iter(loader))
real_imgs = batch["images"]["rgb"].to(device)
# Normalize: the training pipeline uses ImageNet stats
real_imgs = real_imgs.float() / 255.0
mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
real_imgs = (real_imgs - mean) / std
print(f"Real images: {real_imgs.shape} μ={real_imgs.mean():.3f} σ={real_imgs.std():.3f}")

# Random noise (standard normal)
rand_imgs = torch.randn_like(real_imgs)

with torch.no_grad():
    # Store REAL features
    c2_r, c3_r, c4_r, c5_r = model.backbone(real_imgs)
    # Store RANDOM features
    c2_n, c3_n, c4_n, c5_n = model.backbone(rand_imgs)

    print("\n=== COMPARISON: REAL vs RANDOM ===")
    for name, f_r, f_n in [
        ("c2", c2_r, c2_n),
        ("c3", c3_r, c3_n),
        ("c4", c4_r, c4_n),
        ("c5", c5_r, c5_n),
    ]:
        f_r = f_r.float()
        f_n = f_n.float()

        # Mean/std
        print(f"\n  {name}:")
        print(
            f"    REAL:   μ={f_r.mean():.4f} σ={f_r.std():.4f} "
            f"spatial_std={f_r.std(dim=[2, 3]).mean():.4f}"
        )
        print(
            f"    RANDOM: μ={f_n.mean():.4f} σ={f_n.std():.4f} "
            f"spatial_std={f_n.std(dim=[2, 3]).mean():.4f}"
        )

        # Cosine similarity between REAL and RANDOM outputs
        # If backbone is input-dependent, this should be well below 1.0
        cos_sim = torch.cosine_similarity(f_r.flatten(), f_n.flatten(), dim=0)
        print(f"    REAL vs RANDOM cosine sim: {cos_sim:.4f}")

        # Per-channel distribution shift
        ch_mean_r = f_r.mean(dim=[0, 2, 3])
        ch_mean_n = f_n.mean(dim=[0, 2, 3])
        ch_diff = (ch_mean_r - ch_mean_n).abs().mean()
        print(f"    Per-channel mean difference: {ch_diff:.4f}")

        # Spatial pattern similarity (average per-location corr across channels)
        b, c, h, w = f_r.shape
        corr = torch.stack(
            [
                torch.corrcoef(torch.stack([f_r[0, ch_idx].flatten(), f_n[0, ch_idx].flatten()]))[
                    0, 1
                ]
                for ch_idx in range(min(c, 50))  # sample 50 channels
            ]
        )
        print(f"    Avg spatial correlation (50ch): {corr.mean():.4f} (σ={corr.std():.4f})")

    # Detection head with real vs random
    pyramid_r = model.fpn(
        cfg._sanitize(c3_r) if hasattr(cfg, "_sanitize") else c3_r, c4_r, c5_r
    )  # simplified
    cls_r, _ = model.detection_head(pyramid_r)

    print(f"\n  DETECTION HEAD:")
    print(f"    REAL cls_mean: {cls_r.float().mean():.4f}  std: {cls_r.float().std():.4f}")
    print(f"    REAL bg-fg gap: {(cls_r[..., 0].mean() - cls_r[..., 1:].mean()).item():.4f}")
