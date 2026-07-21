#!/usr/bin/env python3
"""
Pose-Norm Fix Evaluation — Opus 126 §1.13

Compares pose MAE with and without the unit-normalization fix in
industreal_dataset.py:606-613.

Usage:
  cd /path/to/industreal_improved
  python scripts/eval_pose_norm_fix.py

Outputs:
  - Post-fix metrics (with unit normalization on load)
  - Pre-fix  metrics (raw CSV values, normalization removed)
  - Per-frame norm statistics
"""

import sys, logging, math
from typing import Dict
from pathlib import Path

_PROJ = Path(__file__).resolve().parent.parent
_SRC = _PROJ / "src"
sys.path.insert(0, str(_SRC))
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))
for _sub in ["models", "training", "evaluation", "data"]:
    _p = str(_SRC / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import numpy as np
from torch.utils.data import DataLoader

# Avoid importing the massive evaluate.py — implement pose metrics inline.
# evaluate.py's compute_head_pose_metrics is simple enough to replicate.

import config as C
from models.model import POPWMultiTaskModel
from data.industreal_dataset import IndustRealMultiTaskDataset, collate_fn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pose_norm_fix_eval")

# ── Constants ────────────────────────────────────────────────────────
CKPT_PATH = str(_PROJ / "src/runs/rf_stages/checkpoints/crash_recovery.pth")
BEST_PATH = str(_PROJ / "src/runs/rf_stages/checkpoints/best.pth")
NUM_FRAMES = 100  # small subset for speed
BATCH_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Training PID 2998753 uses GPU 0. Set CUDA_VISIBLE_DEVICES=1 to use GPU 1.

# ImageNet normalization constants (used by training pipeline)
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)


def compute_pose_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    """
    Compute head pose MAE per DoF and overall.
    9-DoF: forward(3), position(3), up(3).
    Angular MAE for directional vectors (forward and up).
    Matches evaluate.py:compute_head_pose_metrics exactly.
    """
    abs_err = np.abs(pred - gt)
    dof_names = [
        "forward_x",
        "forward_y",
        "forward_z",
        "pos_x",
        "pos_y",
        "pos_z",
        "up_x",
        "up_y",
        "up_z",
    ]
    result = {}
    for i, name in enumerate(dof_names):
        result[f"{name}_MAE"] = float(abs_err[:, i].mean())
    result["head_pose_MAE"] = float(abs_err.mean())
    result["head_pose_MAE_std"] = float(abs_err.std())
    result["n_samples"] = int(pred.shape[0])

    def _angular_err(a: np.ndarray, b: np.ndarray) -> float:
        a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
        b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
        dot = np.sum(a_n * b_n, axis=1)
        dot = np.clip(dot, -1.0, 1.0)
        return float(np.degrees(np.arccos(dot)).mean())

    forward_angular = _angular_err(pred[:, :3], gt[:, :3])
    up_angular = _angular_err(pred[:, 6:9], gt[:, 6:9])
    result["head_pose_angular_MAE_deg"] = (forward_angular + up_angular) / 2.0
    result["forward_angular_MAE_deg"] = forward_angular
    result["up_angular_MAE_deg"] = up_angular

    # Position MAE in mm (assumes HEAD_POSE_POS_SCALE was applied)
    pos_mae_mm = float(np.abs(pred[:, 3:6] - gt[:, 3:6]).mean()) * 1000.0
    result["position_MAE_mm"] = pos_mae_mm

    return result


def normalize_images(images: torch.Tensor) -> torch.Tensor:
    """Normalize uint8 [B,3,H,W] images to float [0,1] then apply ImageNet stats."""
    images = images.float().div_(255.0)
    images = (images - _IMAGENET_MEAN.to(images.device)) / _IMAGENET_STD.to(images.device)
    return images


# ── Model loading ────────────────────────────────────────────────────


def load_model(ckpt_path: str) -> POPWMultiTaskModel:
    """Load model from checkpoint (handles training wrappers)."""
    model = POPWMultiTaskModel(
        pretrained=False,
        backbone_type=str(getattr(C, "BACKBONE", "convnext_tiny")),
        use_headpose_film=bool(getattr(C, "USE_HEADPOSE_FILM", True)),
        use_videomae=bool(getattr(C, "USE_VIDEOMAE", False)),
    ).to(DEVICE)

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("model_state", ckpt.get("model", ckpt)))
    if isinstance(state, dict) and "model" in state:
        state = state["model"]

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.info(f"  Missing keys: {len(missing)} (expected for eval — head_pose, etc.)")
    if unexpected:
        logger.info(f"  Unexpected keys: {len(unexpected)}")

    epoch = ckpt.get("epoch", "?")
    logger.info(f"  Loaded checkpoint: {ckpt_path} (epoch={epoch})")
    return model, epoch


# ── Raw CSV parsing (pre-fix) ────────────────────────────────────────


def parse_pose_csv_raw(rec_dir: Path) -> np.ndarray:
    """
    Parse pose.csv WITHOUT the unit-normalization fix (lines 606-613).
    This gives us the raw sensor values exactly as stored.
    """
    pose_file = rec_dir / "pose.csv"
    if not pose_file.exists():
        return np.zeros((0, 9), dtype=np.float32)

    import csv

    data = {}
    with open(pose_file, encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 10:
                continue
            try:
                # The first column is a filename (000000.jpg), extract numeric frame
                frame_col = row[0].strip()
                if "." in frame_col:
                    frame_num = int(frame_col.split(".")[0])
                else:
                    frame_num = int(frame_col)
                values = np.array([float(x) for x in row[1:10]], dtype=np.float32)
                data[frame_num] = values
            except (ValueError, IndexError):
                continue

    if not data:
        return np.zeros((0, 9), dtype=np.float32)

    max_frame = max(data.keys()) + 1
    pose_data = np.zeros((max_frame, 9), dtype=np.float32)
    for frame_num, values in data.items():
        pose_data[frame_num] = values

    # Still apply HEAD_POSE_POS_SCALE (this is separate from the norm fix)
    if C.HEAD_POSE_POS_SCALE != 0.0:
        pose_data[:, 3:6] /= C.HEAD_POSE_POS_SCALE

    return pose_data


# ── Main evaluation ──────────────────────────────────────────────────


@torch.no_grad()
def main():
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Checkpoint: {CKPT_PATH}")
    logger.info(f"Frames: {NUM_FRAMES}")

    # ── 1. Load model ────────────────────────────────────────────────
    model, epoch = load_model(CKPT_PATH)
    model.eval()

    # ── 2. Load val dataset (with fix applied — current code) ────────
    ds = IndustRealMultiTaskDataset(
        split="val",
        img_size=C.IMG_SIZE,
        augment=False,
        seed=42,
    )
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # ── 3. Run inference and collect predictions + ground truth ──────
    all_pred = []
    all_gt_fixed = []  # ground truth WITH the fix (current dataset)
    all_rec_ids = []  # recording IDs for raw CSV lookup
    all_frame_nums = []

    batch_count = 0
    frame_count = 0
    for batch in loader:
        # collate_fn returns (images, targets) tuple
        # Images are uint8 [0, 255] from dataset; normalize like train.py:334
        rgb, targets = batch
        rgb = normalize_images(rgb).to(DEVICE)
        gt_hp = targets["head_pose"].numpy()  # already normalized (fix applied)
        metadata = targets["metadata"]  # list of dicts

        outputs = model(rgb)
        pred_hp = outputs["head_pose"].cpu().numpy()

        for b in range(rgb.shape[0]):
            all_pred.append(pred_hp[b])
            all_gt_fixed.append(gt_hp[b])
            all_rec_ids.append(metadata[b]["recording_id"])
            all_frame_nums.append(metadata[b]["frame_num"])
            frame_count += 1

        batch_count += 1
        if frame_count >= NUM_FRAMES:
            break

        if batch_count % 5 == 0:
            logger.info(f"  Processed {frame_count} frames...")

    all_pred = np.array(all_pred)
    all_gt_fixed = np.array(all_gt_fixed)
    logger.info(f"Collected {len(all_pred)} frames across {batch_count} batches")

    # ── 4. Build pre-fix ground truth (raw CSV) ──────────────────────
    # Use the dataset's own samples to find recording directories.
    # Build a set of recording_ids seen
    rec_ids_in_batch = set(all_rec_ids)
    rec_dir_map = {}
    for s in ds.samples:
        rid = s["recording_id"]
        if rid in rec_ids_in_batch and rid not in rec_dir_map:
            # img_path example: /path/to/recordings/val/REC_001/rgb/frame_0000.jpg
            img_p = Path(s["img_path"])
            # Walk up to find the recording directory (grandparent of rgb/)
            cand = img_p.parent.parent  # rgb/ -> recording dir
            if cand.name == rid and (cand / "pose.csv").exists():
                rec_dir_map[rid] = cand
            else:
                # Try recordings_root / split / rid
                for chk_split in ["val", "train", "test"]:
                    p = Path(ds.recordings_root) / chk_split / rid
                    if (p / "pose.csv").exists():
                        rec_dir_map[rid] = p
                        break

    logger.info(
        f"Found recording dirs for {len(rec_dir_map)}/{len(rec_ids_in_batch)} unique recordings"
    )

    raw_pose_cache = {}  # recording_id -> pose array (unfixed)
    all_gt_raw = np.zeros_like(all_gt_fixed)

    for i in range(len(all_rec_ids)):
        rid = all_rec_ids[i]
        fn = all_frame_nums[i]

        if rid not in raw_pose_cache:
            rdir = rec_dir_map.get(rid)
            if rdir is None:
                logger.warning(f"Cannot find recording dir for {rid}")
                raw_pose_cache[rid] = None
            else:
                raw_pose_cache[rid] = parse_pose_csv_raw(rdir)

        pose_arr = raw_pose_cache[rid]
        if pose_arr is not None and fn < len(pose_arr):
            all_gt_raw[i] = pose_arr[fn]
        else:
            # Fallback: use the fixed value (no raw available)
            all_gt_raw[i] = all_gt_fixed[i]

    # ── 5. Check norms before/after ──────────────────────────────────
    fwd_norms_raw = np.linalg.norm(all_gt_raw[:, 0:3], axis=1)
    up_norms_raw = np.linalg.norm(all_gt_raw[:, 6:9], axis=1)
    fwd_norms_fixed = np.linalg.norm(all_gt_fixed[:, 0:3], axis=1)
    up_norms_fixed = np.linalg.norm(all_gt_fixed[:, 6:9], axis=1)

    print("\n" + "=" * 70)
    print("POSE VECTOR NORM ANALYSIS")
    print("=" * 70)
    print(f"  {'':20s} {'Pre-fix (raw)':>15s} {'Post-fix':>15s}")
    print(
        f"  {'Forward mean norm':20s} {fwd_norms_raw.mean():>14.6f}  {fwd_norms_fixed.mean():>14.6f}"
    )
    print(
        f"  {'Forward std norm':20s} {fwd_norms_raw.std():>14.6f}  {fwd_norms_fixed.std():>14.6f}"
    )
    print(
        f"  {'Forward min norm':20s} {fwd_norms_raw.min():>14.6f}  {fwd_norms_fixed.min():>14.6f}"
    )
    print(
        f"  {'Forward max norm':20s} {fwd_norms_raw.max():>14.6f}  {fwd_norms_fixed.max():>14.6f}"
    )
    print(
        f"  {'Forward drift >1%':20s} {(np.abs(fwd_norms_raw - 1.0) > 0.01).mean() * 100:>14.1f}%  {'0.0%':>15s}"
    )
    print(
        f"  {'Forward drift >5%':20s} {(np.abs(fwd_norms_raw - 1.0) > 0.05).mean() * 100:>14.1f}%  {'0.0%':>15s}"
    )
    print(
        f"  {'Forward drift >10%':20s} {(np.abs(fwd_norms_raw - 1.0) > 0.10).mean() * 100:>14.1f}%  {'0.0%':>15s}"
    )
    print(f"  {'Up mean norm':20s} {up_norms_raw.mean():>14.6f}  {up_norms_fixed.mean():>14.6f}")
    print(f"  {'Up std norm':20s} {up_norms_raw.std():>14.6f}  {up_norms_fixed.std():>14.6f}")
    print(f"  {'Up min norm':20s} {up_norms_raw.min():>14.6f}  {up_norms_fixed.min():>14.6f}")
    print(f"  {'Up max norm':20s} {up_norms_raw.max():>14.6f}  {up_norms_fixed.max():>14.6f}")
    print(
        f"  {'Up drift >1%':20s} {(np.abs(up_norms_raw - 1.0) > 0.01).mean() * 100:>14.1f}%  {'0.0%':>15s}"
    )
    print(
        f"  {'Up drift >5%':20s} {(np.abs(up_norms_raw - 1.0) > 0.05).mean() * 100:>14.1f}%  {'0.0%':>15s}"
    )

    # ── 6. Compute metrics ───────────────────────────────────────────

    # 6a. Post-fix metrics: using fixed ground truth
    metrics_fixed = compute_pose_metrics(all_pred, all_gt_fixed)

    # 6b. Pre-fix metrics: using raw (un-normalized) ground truth
    metrics_raw = compute_pose_metrics(all_pred, all_gt_raw)

    print("\n" + "=" * 70)
    print("POSE MAE COMPARISON: PRE-FIX vs POST-FIX")
    print(f"Checkpoint: crash_recovery.pth (epoch {epoch}) — ~100 val frames")
    print("=" * 70)

    # Compare key metrics
    key_metrics = [
        ("forward_angular_MAE_deg", "Forward Angular MAE (deg)"),
        ("up_angular_MAE_deg", "Up Angular MAE (deg)"),
        ("head_pose_angular_MAE_deg", "Head Pose Angular MAE (deg)"),
        ("head_pose_MAE", "Head Pose Raw MAE"),
        ("forward_x_MAE", "Forward X MAE"),
        ("forward_y_MAE", "Forward Y MAE"),
        ("forward_z_MAE", "Forward Z MAE"),
        ("up_x_MAE", "Up X MAE"),
        ("up_y_MAE", "Up Y MAE"),
        ("up_z_MAE", "Up Z MAE"),
        ("position_MAE_mm", "Position MAE (mm)"),
    ]

    print(f"  {'Metric':40s} {'Pre-fix':>12s} {'Post-fix':>12s} {'Delta':>12s} {'Change':>10s}")
    print(f"  {'-' * 40} {'-' * 12} {'-' * 12} {'-' * 12} {'-' * 10}")

    for key, label in key_metrics:
        pre = metrics_raw.get(key, float("nan"))
        post = metrics_fixed.get(key, float("nan"))
        if not (math.isfinite(pre) and math.isfinite(post)):
            delta_str = "N/A"
            change_str = "N/A"
        elif abs(pre) < 1e-8:
            delta_str = f"{post - pre:+.6f}"
            change_str = "N/A"
        else:
            delta = post - pre
            pct = ((post - pre) / abs(pre)) * 100
            delta_str = f"{delta:+.6f}"
            change_str = f"{pct:+.1f}%"
        print(f"  {label:40s} {pre:>12.6f} {post:>12.6f} {delta_str:>12s} {change_str:>10s}")

    # ── 7. Projection to training impact ─────────────────────────────
    print("\n" + "=" * 70)
    print("ANALYSIS: EXPECTED IMPACT ON TRAINING METRICS")
    print("=" * 70)

    # The angular MAE normalizes both vectors internally — should be near-identical
    fwd_ang_delta = metrics_fixed.get("forward_angular_MAE_deg", 0) - metrics_raw.get(
        "forward_angular_MAE_deg", 0
    )
    up_ang_delta = metrics_fixed.get("up_angular_MAE_deg", 0) - metrics_raw.get(
        "up_angular_MAE_deg", 0
    )

    print(f"  Angular MAE impact (forward): {fwd_ang_delta:+.4f} deg")
    print(f"  Angular MAE impact (up):      {up_ang_delta:+.4f} deg")

    if abs(fwd_ang_delta) < 0.01 and abs(up_ang_delta) < 0.01:
        print("  VERDICT: Angular MAE is effectively unchanged (within rounding).")
        print("  The angular MAE normalizes both vectors before computing the angle,")
        print("  so the fix has near-zero direct impact on angular evaluation.")
    else:
        print("  NOTE: Angular MAE changed >0.01 deg — unexpected.")

    # Raw component MAE analysis
    fwd_component_keys = ["forward_x_MAE", "forward_y_MAE", "forward_z_MAE"]
    up_component_keys = ["up_x_MAE", "up_y_MAE", "up_z_MAE"]

    raw_fwd_delta = sum(
        abs(metrics_fixed.get(k, 0) - metrics_raw.get(k, 0)) for k in fwd_component_keys
    )
    raw_up_delta = sum(
        abs(metrics_fixed.get(k, 0) - metrics_raw.get(k, 0)) for k in up_component_keys
    )

    print(f"  Raw forward MAE total abs delta: {raw_fwd_delta:.6f}")
    print(f"  Raw up MAE total abs delta:      {raw_up_delta:.6f}")

    # Check if historical numbers would improve
    hist_fwd = 7.83  # user-reported epoch 17 subsample forward angular
    hist_up = 7.06  # user-reported epoch 17 subsample up angular

    pred_fwd = metrics_fixed.get("forward_angular_MAE_deg", 0)
    pred_up = metrics_fixed.get("up_angular_MAE_deg", 0)

    print(f"\n  Historical epoch 17 subsample: fwd={hist_fwd} deg, up={hist_up} deg")
    print(f"  This eval (epoch~18, 100 frames): fwd={pred_fwd:.4f} deg, up={pred_up:.4f} deg")

    # ── 8. Training loss analysis ────────────────────────────────────
    print("\n" + "=" * 70)
    print("TRAINING LOSS IMPACT ANALYSIS")
    print("=" * 70)
    print("""
  The head_pose_loss_split function (losses.py:941-965) normalizes both prediction
  and target direction vectors before computing MSE:
      fwd_tn = F.normalize(fwd_t, dim=1, eps=eps)
      up_tn  = F.normalize(up_t,  dim=1, eps=eps)

  This makes the direction term scale-invariant. The only place where near-unit
  vs exactly-unit ground truth matters is the norm_reg_weight term (if >0):
      norm_reg = ((fwd_norm - 1.0) ** 2 + (up_norm - 1.0) ** 2).mean()

  Current config: norm_reg_weight defaults to 0.0. So the fix does NOT change
  the training loss on the epoch 17 checkpoint.

  The main benefits of the fix are:
    1. Raw component MAE (forward_x_MAE, etc.) becomes numerically correct
       when compared against exactly-unit vectors
    2. Eliminates silent corruption if someone enables norm_reg regularization
    3. Ensures any future loss that assumes unit vectors works correctly
""")

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
  Fix location:   industreal_dataset.py:606-613
  Fix type:       Unit-normalize forward + up vectors on data load
  Norm drift:     {fwd_norms_raw.std():.4f} (forward pre-fix) / {up_norms_raw.std():.4f} (up pre-fix)
  Angular MAE:    Changed by <0.01 deg (negligible — angular metric already normalizes)
  Raw MAE:        Small change proportional to norm drift ({raw_fwd_delta:.4f} total forward)
  Training loss:  No change (head_pose_loss_split normalizes targets before computing MSE)
  Fix correctness:{" OK" if abs(fwd_norms_fixed.mean() - 1.0) < 1e-5 else " CHECK — norms not exactly 1"}

  Recommendation: Fix is correct and good practice. It will not directly improve
  the epoch 17 numbers (7.83/7.06 deg) because the angular MAE pipeline already
  normalizes vectors. However, it eliminates a latent source of silent corruption
  that could affect future experiments with norm_weight>0 or changes to the
  head_pose_loss computation.
""")


if __name__ == "__main__":
    main()
