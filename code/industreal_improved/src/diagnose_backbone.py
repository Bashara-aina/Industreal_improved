"""
Diagnose ConvNeXt backbone feature health from latest checkpoint.
Checks: feature variance, activation stats, detection head output distribution.
"""

import sys, os, torch
from pathlib import Path

# Setup paths
RUNS_ROOT = Path(__file__).parent / "runs"
RF_STATE = RUNS_ROOT / "rf_stage_state.json"
CKPT_DIR = RUNS_ROOT / "rf_stages" / "checkpoints"
CFG_PATH = Path(__file__).parent / "config.py"

# Load config
os.chdir(Path(__file__).parent.parent)  # project root
import importlib.util

spec = importlib.util.spec_from_file_location("config", CFG_PATH)
cfg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cfg)

# Apply RF1 preset
cfg.apply_preset("stage_rf1")
cfg.update_dynamic_paths()

# Find latest checkpoint
ckpt_files = sorted(CKPT_DIR.glob("*.pth"), key=os.path.getmtime)
print(f"Checkpoints found: {[f.name for f in ckpt_files[-3:]]}")
latest_ckpt = ckpt_files[-1]
print(f"Loading: {latest_ckpt.name} ({latest_ckpt.stat().st_size / 1e6:.0f} MB)")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load checkpoint
ckpt = torch.load(latest_ckpt, map_location="cpu", weights_only=False)
print(f"Checkpoint keys: {list(ckpt.keys())[:10]}...")

# Find model state dict
state_key = (
    "model_state_dict"
    if "model_state_dict" in ckpt
    else "model"
    if "model" in ckpt and isinstance(ckpt["model"], dict)
    else None
)
if state_key:
    state_dict = ckpt[state_key]
    print(f"Using key: '{state_key}' — {len(state_dict)} params")
else:
    state_dict = {
        k: v
        for k, v in ckpt.items()
        if any(k.startswith(p) for p in ["backbone.", "fpn.", "detection_head."])
    }
    if not state_dict:
        # Try using entire checkpoint as state dict
        state_dict = {k: v for k, v in ckpt.items() if isinstance(v, torch.Tensor)}
    print(f"Inferred state dict: {len(state_dict)} tensor keys")

# Build model
from src.models.model import POPWMultiTaskModel

model = POPWMultiTaskModel(
    pretrained=False,
    backbone_type="convnext_tiny",
    use_headpose_film=cfg.USE_HEADPOSE_FILM,
    use_hand_film=True,
    use_videomae=False,
    train_pose=False,
    use_backbone_checkpoint=False,
)

# Load weights
missing, unexpected = model.load_state_dict(state_dict, strict=False)
print(f"Model loaded: {len(missing)} missing, {len(unexpected)} unexpected")
if missing:
    for m in missing[:20]:
        print(f"  MISSING: {m}")
if unexpected:
    for u in unexpected[:20]:
        print(f"  UNEXPECTED: {u}")

model = model.to(device)
model.eval()

# Try to load a real image from the dataset
print("\n--- Loading sample images ---")
try:
    from torch.utils.data import DataLoader

    sys.path.insert(0, str(Path(__file__).parent))
    from data.industreal_dataset import IndustRealMultiTaskDataset as IndustRealDataset

    val_csv = cfg.VAL_CSV
    recordings_root = cfg.RECORDINGS_ROOT

    dataset = IndustRealDataset(
        split="val",
        img_size=(cfg.IMG_WIDTH, cfg.IMG_HEIGHT),
    )
    print(f"Val dataset: {len(dataset)} samples")

    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    images = batch["images"].to(device)
    print(f"Images shape: {images.shape} dtype={images.dtype}")
except Exception as e:
    print(f"Dataset load failed: {e}")
    # Use random data instead
    print("Using random data as fallback")
    images = torch.randn(2, 3, cfg.IMG_HEIGHT, cfg.IMG_WIDTH, device=device)

# Run forward pass
print("\n=== BACKBONE FEATURE DIAGNOSIS ===")
with torch.no_grad():
    c2, c3, c4, c5 = model.backbone(images)

    for name, feat in [("c2", c2), ("c3", c3), ("c4", c4), ("c5", c5)]:
        f = feat.float()
        print(f"\n  {name}: shape={list(f.shape)}")
        print(f"    mean={f.mean().item():.6f}  std={f.std().item():.6f}")
        print(f"    min={f.min().item():.6f}  max={f.max().item():.6f}")

        # Per-channel stats
        ch_mean = f.mean(dim=[0, 2, 3])
        ch_std = f.std(dim=[0, 2, 3])
        print(
            f"    channel mean:  μ={ch_mean.mean().item():.6f}  σ={ch_mean.std().item():.6f}  range=[{ch_mean.min().item():.6f}, {ch_mean.max().item():.6f}]"
        )
        print(
            f"    channel std:   μ={ch_std.mean().item():.6f}  σ={ch_std.std().item():.6f}  range=[{ch_std.min().item():.6f}, {ch_std.max().item():.6f}]"
        )

        # Spatial variance (is the feature map flat?)
        spatial_std = f.std(dim=[2, 3]).mean(dim=0)
        print(
            f"    spatial std per channel: μ={spatial_std.mean().item():.6f}  σ={spatial_std.std().item():.6f}"
        )

        # Dead channel ratio (std < 1e-6)
        dead_ratio = (ch_std < 1e-6).float().mean().item()
        print(f"    dead channels (std<1e-6): {dead_ratio * 100:.1f}%")

        # Near-constant channels (std < 1e-3)
        flat_ratio = (ch_std < 1e-3).float().mean().item()
        print(f"    flat channels (std<1e-3): {flat_ratio * 100:.1f}%")

    # FPN outputs
    def _sanitize(x, bound=100.0):
        if torch.isfinite(x).all():
            return x.clamp(-bound, bound)
        return torch.where(torch.isfinite(x), x.clamp(-bound, bound), torch.zeros_like(x))

    _c3, _c4, _c5 = _sanitize(c3), _sanitize(c4), _sanitize(c5)
    pyramid = model.fpn(_c3, _c4, _c5)

    print("\n=== FPN FEATURE DIAGNOSIS ===")
    for name in ["p3", "p4", "p5", "p6", "p7"]:
        if name in pyramid:
            f = pyramid[name].float()
            print(
                f"  {name}: shape={list(f.shape)}  mean={f.mean().item():.6f}  std={f.std().item():.6f}  "
                f"min={f.min().item():.6f}  max={f.max().item():.6f}"
            )

    # Detection head outputs
    cls_preds, reg_preds = model.detection_head(pyramid)

    print("\n=== DETECTION HEAD DIAGNOSIS ===")
    print(f"  cls_preds: shape={list(cls_preds.shape)}")
    cls = cls_preds.float()
    print(f"    mean={cls.mean().item():.6f}  std={cls.std().item():.6f}")
    print(f"    min={cls.min().item():.6f}  max={cls.max().item():.6f}")

    # Per-anchor max (class confidence)
    cls_max = cls.sigmoid().max(dim=-1)[0]
    print(
        f"    sigmoid max: mean={cls_max.mean().item():.6f}  std={cls_max.std().item():.6f}  "
        f"max={cls_max.max().item():.6f}"
    )

    # Score distribution
    all_scores = cls.sigmoid().flatten()
    if all_scores.numel() > 0:
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        print(f"    score percentiles:")
        for p in percentiles:
            val = torch.quantile(all_scores, p / 100.0).item()
            print(f"      P{p:02d}: {val:.6f}")

    # Flat prediction check
    cls_flat_std = cls.std(dim=-1)
    print(
        f"    per-anchor class std: μ={cls_flat_std.mean().item():.6f}  "
        f"σ={cls_flat_std.std().item():.6f}  min={cls_flat_std.min().item():.6f}"
    )
    flat_pct = (cls_flat_std < 0.01).float().mean().item()
    print(f"    flat predictions (std<0.01): {flat_pct * 100:.1f}%")

    # cls_mean (cls_score average across all classes and anchors)
    cls_mean_val = cls.mean().item()
    print(
        f"    cls_mean: {cls_mean_val:.6f}  (equivalent sigmoid: {torch.sigmoid(cls.mean()).item():.6f})"
    )

    # Check if cls_mean is dominated by background class (index 0)
    bg_mean = cls[..., 0].mean().item()
    fg_mean = cls[..., 1:].mean().item()
    print(f"    bg class (idx 0) mean: {bg_mean:.6f}")
    print(f"    fg classes (idx 1+) mean: {fg_mean:.6f}")
    print(f"    bg-fg gap: {abs(bg_mean - fg_mean):.6f}")

print("\n=== VERDICT ===")
# Check for collapse signatures
c5_std = c5.float().std().item()
cls_std_val = cls.float().std().item()
if c5_std < 0.01:
    print("  [COLLAPSED] Backbone C5 has near-zero variance — model producing constant features")
elif c5_std < 0.1:
    print(
        "  [WARNING] Backbone C5 variance very low ({:.4f}) — features may be too weak".format(
            c5_std
        )
    )
else:
    print("  [OK] Backbone C5 variance: {:.4f}".format(c5_std))

if flat_pct > 0.9:
    print("  [COLLAPSED] Detection head producing near-identical scores for all anchors")
elif flat_pct > 0.5:
    print("  [WARNING] {:.0f}% of anchors have flat predictions".format(flat_pct * 100))

if abs(bg_mean - fg_mean) < 0.1:
    print(
        "  [WARNING] Background/foreground scores nearly identical — classifier not discriminating"
    )

if c5_std >= 0.1 and flat_pct < 0.5:
    print("  [HEALTHY] Backbone features and detection head both active")
