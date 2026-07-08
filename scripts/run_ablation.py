#!/usr/bin/env python3
"""Ablation runner for Tier F experiment matrix (175 ULTIMATE GUIDE §6 rows 7-10).

Five ablation flavors:

  loo-no-det      MTL-All minus detection       proves detection --> PSR mechanism
  loo-no-pose     MTL-All minus pose            is pose useful auxiliary or dead weight?
  mt-frozen-bb    MTL-All with backbone frozen  cost of freezing (explains V5 activity failure)
  backbone-swap-convnext  MTL-All on ConvNeXt   compare with Hiera; interference is representation-mediated
  backbone-swap-hiera     MTL-All on Hiera       reference for the swap comparison

Usage:
  python scripts/run_ablation.py --ablation loo-no-det --epochs 5
  python scripts/run_ablation.py --ablation backbone-swap-convnext --epochs 5
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Path plumbing
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CODE_ROOT = _PROJECT_ROOT / "code" / "industreal_improved"
for _p in [
    str(_CODE_ROOT), str(_CODE_ROOT / "src"), str(_CODE_ROOT / "src" / "models"),
    str(_CODE_ROOT / "src" / "training"), str(_CODE_ROOT / "src" / "evaluation"),
    str(_CODE_ROOT / "src" / "data"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_ablation")

DATA_ROOT = Path("/media/newadmin/master/POPW/datasets/industreal")
TRAIN_CSV = DATA_ROOT / "train.csv"

# ---------------------------------------------------------------------------
# Label path resolution
# ---------------------------------------------------------------------------
_SPLIT_DIRS = ["train", "val", "test"]


def _resolve_rec_dir(recording: str) -> Path:
    """Find the recording directory under any split."""
    base = DATA_ROOT / "recordings"
    for s in _SPLIT_DIRS:
        p = base / s / recording
        if p.exists():
            return p
    if (base / recording).exists():
        return base / recording
    return base / "train" / recording


def _resolve_label_path(recording: str, filename: str) -> Path:
    rd = _resolve_rec_dir(recording)
    for p in [rd / filename, rd / "rgb" / filename]:
        if p.exists():
            return p
    return rd / filename


def _frame_base_dir(recording: str) -> Path:
    """Find frame location: typically recordings/{split}/{recording}/rgb/."""
    rd = _resolve_rec_dir(recording)
    rgb = rd / "rgb"
    if rgb.exists():
        return rgb
    # Sometimes frames are directly in the recording dir
    if any(rd.glob("*.jpg")):
        return rd
    # Fallback with warning
    return rgb


# ---------------------------------------------------------------------------
# Frame loading
# ---------------------------------------------------------------------------
def load_clip_frames(recording: str, start_frame: int, num_frames: int = 16):
    """Load num_frames consecutive frames, return [T, C, H, W] tensor or None."""
    frame_dir = _frame_base_dir(recording)
    frames = []
    for i in range(num_frames):
        frame_idx = start_frame + i
        fp = frame_dir / f"{frame_idx:06d}.jpg"
        if not fp.exists():
            return None
        import cv2
        img = cv2.imread(str(fp))
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        scale = 256.0 / min(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        img = cv2.resize(img, (new_w, new_h))
        top = (new_h - 224) // 2
        left = (new_w - 224) // 2
        img = img[top:top+224, left:left+224].astype(np.float32) / 255.0
        img = (img - 0.45) / 0.225
        frames.append(img.transpose(2, 0, 1))
    clip = np.stack(frames, axis=0)
    return torch.from_numpy(clip).float()


# ---------------------------------------------------------------------------
# Label loading
# ---------------------------------------------------------------------------

# Activity: AR_labels.csv has recording,offset,action,start.jpg,end.jpg per row.
# Each row spans a contiguous frame range all labeled with the same action.
def load_ar_labels(recording: str) -> dict:
    path = _resolve_label_path(recording, "AR_labels.csv")
    labels = {}
    if not path.exists():
        logger.warning("AR_labels.csv missing for %s", recording)
        return labels
    with open(path) as f:
        for row in csv.reader(f):
            if len(row) < 5:
                continue
            action = row[2]
            try:
                sf = int(row[3].replace(".jpg", "").lstrip("0") or "0")
                ef = int(row[4].replace(".jpg", "").lstrip("0") or "0")
            except ValueError:
                continue
            for frm in range(sf, ef + 1):
                labels[frm] = action
    return labels


# Pose: pose.csv has frame.jpg, fwd_x, fwd_y, fwd_z, up_x, up_y, up_z, ...
# Columns 1-3 = forward vector, 4-6 = up vector (9 DoF total but we use 6D).
def load_pose_labels(recording: str) -> dict:
    path = _resolve_label_path(recording, "pose.csv")
    labels = {}
    if not path.exists():
        logger.warning("pose.csv missing for %s", recording)
        return labels
    with open(path) as f:
        for row in csv.reader(f):
            if len(row) < 7:
                continue
            try:
                frame = int(row[0].split(".")[0])
            except ValueError:
                continue
            fwd = np.array([float(row[1]), float(row[2]), float(row[3])])
            up = np.array([float(row[4]), float(row[5]), float(row[6])])
            labels[frame] = (fwd, up)
    return labels


# PSR: PSR_labels.csv has sparse event format: frame.jpg, comp_id, step_name.
# We convert to per-frame binary 11-component vectors.
# comp_id is 0-based or 1-based; 11 components.
def load_psr_labels(recording: str) -> dict:
    path = _resolve_label_path(recording, "PSR_labels.csv")
    labels = {}  # frame -> 11-element binary list
    if not path.exists():
        logger.warning("PSR_labels.csv missing for %s", recording)
        return labels

    # First pass: collect all component changes per frame
    frame_events = defaultdict(set)
    with open(path) as f:
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            try:
                frame = int(row[0].split(".")[0])
                comp = int(row[1])
            except (ValueError, IndexError):
                continue
            if 0 <= comp < 11:
                frame_events[frame].add(comp)

    # Convert to binary vectors per frame (all 0s unless an event was recorded)
    # Note: this is an approximation; the actual PSR pipeline interpolates
    # between keyframes. For plumbing, this provides a valid training signal.
    for frame, comps in frame_events.items():
        vec = [0] * 11
        for c in comps:
            vec[c] = 1
        labels[frame] = vec
    return labels


# Detection: OD_labels.json is COCO format.
def load_det_labels(recording: str) -> dict:
    path = _resolve_label_path(recording, "OD_labels.json")
    labels = {}
    if not path.exists():
        return labels
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return labels

    # Build image_id -> file_name map
    id_to_file = {}
    for img in data.get("images", []):
        fn = img.get("file_name", "")
        try:
            fid = int(fn.split(".")[0])
        except ValueError:
            continue
        id_to_file[img["id"]] = fid

    for ann in data.get("annotations", []):
        image_id = ann.get("image_id", 0)
        frame = id_to_file.get(image_id, image_id)
        bbox = ann.get("bbox", [0, 0, 0, 0])
        labels.setdefault(frame, []).append({
            "cls": ann.get("category_id", 0),
            "bbox": bbox,  # [x, y, w, h]
            "area": ann.get("area", 0),
        })
    return labels


# ---------------------------------------------------------------------------
# Per-recording label cache (load labels once, use many times)
# ---------------------------------------------------------------------------
_LABEL_CACHE: dict = {}


def _get_labels(recording: str):
    """Get or load all label data for a recording."""
    if recording not in _LABEL_CACHE:
        _LABEL_CACHE[recording] = {
            "ar": load_ar_labels(recording),
            "pose": load_pose_labels(recording),
            "psr": load_psr_labels(recording),
            "det": load_det_labels(recording),
        }
    return _LABEL_CACHE[recording]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class AblationDataset(torch.utils.data.Dataset):
    """IndustReal clip dataset with robust per-task label extraction."""

    def __init__(self, csv_path, num_frames: int = 16):
        self.num_frames = num_frames
        self.samples = []
        if not Path(csv_path).exists():
            logger.warning("CSV %s not found; empty dataset", csv_path)
            return
        with open(csv_path) as f:
            for row in csv.reader(f):
                if len(row) < 5:
                    continue
                rec, off, act, start, end = row[0], row[1], row[2], row[3], row[4]
                if not rec:
                    continue
                try:
                    sf = int(start.replace(".jpg", "").lstrip("0") or "0")
                    ef = int(end.replace(".jpg", "").lstrip("0") or "0")
                except ValueError:
                    continue
                self.samples.append({
                    "recording": rec, "start_frame": sf, "end_frame": ef,
                    "class": act,
                })
        unique = sorted({s["class"] for s in self.samples})
        self._cls_map = {n: i for i, n in enumerate(unique)}
        self._class_count = len(unique)
        logger.info("Dataset: %d samples, %d activity classes", len(self.samples), len(unique))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        clip = load_clip_frames(s["recording"], s["start_frame"], self.num_frames)
        if clip is None:
            return None, None
        last_frame = s["end_frame"]
        labels = _get_labels(s["recording"])

        # Activity
        cls_str = labels["ar"].get(last_frame, s["class"])
        act_cls = self._cls_map.get(cls_str, 0)

        # Pose (6D: fwd3 + up3)
        pose = labels["pose"].get(last_frame, (np.zeros(3), np.zeros(3)))
        pose_vec = np.concatenate([pose[0], pose[1]])

        # PSR (11 binary components)
        psr_vec = labels["psr"].get(last_frame, [0] * 11)

        # Detection
        det_anns = labels["det"].get(last_frame, [])

        targets = {
            "activity": act_cls,
            "pose": torch.from_numpy(pose_vec).float(),
            "psr": torch.tensor(psr_vec, dtype=torch.float32),
            "det_labels": det_anns,
            "recording": s["recording"],
            "start_frame": s["start_frame"],
        }
        return clip, targets


def collate_fn(batch):
    clips = [b[0] for b in batch if b[0] is not None]
    targets = [b[1] for b in batch if b[0] is not None]
    if not clips:
        return None, None
    return torch.stack(clips), targets


# ---------------------------------------------------------------------------
# Ablation config registry
# ---------------------------------------------------------------------------
ABLATION_CONFIG = {
    "loo-no-det": {
        "model": "tier_f",
        "backbone": "hiera",
        "freeze_backbone": False,
        "heads": {"detection": False, "activity": True, "psr": True, "pose": True},
        "label": "LOO-noDet",
    },
    "loo-no-pose": {
        "model": "tier_f",
        "backbone": "hiera",
        "freeze_backbone": False,
        "heads": {"detection": True, "activity": True, "psr": True, "pose": False},
        "label": "LOO-noPose",
    },
    "mt-frozen-bb": {
        "model": "tier_f",
        "backbone": "hiera",
        "freeze_backbone": True,
        "heads": {"detection": True, "activity": True, "psr": True, "pose": True},
        "label": "MT-frozenBB",
    },
    "backbone-swap-convnext": {
        "model": "convnext",
        "backbone": "convnext_tiny",
        "freeze_backbone": False,
        "heads": {"detection": True, "activity": True, "psr": True, "pose": True},
        "label": "BS-ConvNeXt",
    },
    "backbone-swap-hiera": {
        "model": "tier_f",
        "backbone": "hiera",
        "freeze_backbone": False,
        "heads": {"detection": True, "activity": True, "psr": True, "pose": True},
        "label": "BS-Hiera",
    },
}


# ---------------------------------------------------------------------------
# Detection loss (surrogate — not the full YOLOv8/RetinaNet loss)
# ---------------------------------------------------------------------------
def det_loss(out_dict, targets_list, device):
    """Minimal detection loss for gradient signal during ablation training."""
    cls_logits = out_dict.get("det_cls_logits")
    if cls_logits is None:
        return None

    # Handle various output formats: list of tensors (TierFModel FPN levels)
    # or single [B, N_anchors, num_classes] (ConvNeXt/RetinaNet)
    if isinstance(cls_logits, (list, tuple)):
        if len(cls_logits) == 0:
            return None
        cls_p3 = cls_logits[0]
    elif isinstance(cls_logits, torch.Tensor):
        cls_p3 = cls_logits
    else:
        return None

    # For RetinaNet-style [B, N, C], flatten to [B*N, C] and supervise
    if cls_p3.dim() == 3:
        B, N, C = cls_p3.shape
        cls_flat = cls_p3.reshape(-1, C)  # [B*N, C]
        # All-negative BCE (background outweighs foreground)
        pos_w = torch.tensor(0.1, device=device)
        loss_c = F.binary_cross_entropy_with_logits(
            cls_flat, torch.zeros_like(cls_flat),
            pos_weight=pos_w,
        )
        return loss_c * 0.01  # heavily downweight

    # TierFModel spatial format: [B, C, H, W]
    elif cls_p3.dim() == 4:
        pos_w = torch.tensor(0.01, device=device)
        return F.binary_cross_entropy_with_logits(
            cls_p3, torch.zeros_like(cls_p3), pos_weight=pos_w,
        ) * 0.1

    return None


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------
def build_tier_f_model(ablation_cfg, device):
    """Build TierFModel (Hiera-B) with head enable/disable mask."""
    from src.models.tier_f_model import TierFModel

    model = TierFModel(pretrained=True).to(device)
    heads_enabled = ablation_cfg["heads"]

    if not heads_enabled["detection"]:
        model.detection_head = nn.Identity()
        model.fpn = nn.Identity()
    if not heads_enabled["activity"]:
        model.activity_head = nn.Identity()
    if not heads_enabled["psr"]:
        model.psr_head = nn.Identity()
    if not heads_enabled["pose"]:
        model.pose_head = nn.Identity()

    if ablation_cfg["freeze_backbone"]:
        for p in model.backbone.parameters():
            p.requires_grad = False
        logger.info("Backbone FROZEN")
    return model


def build_convnext_model(ablation_cfg, device):
    """Build POPWMultiTaskModel (ConvNeXt-Tiny) with head enable/disable and clip wrapper.

    The POPWMultiTaskModel expects single frames [B, C, H, W].  The wrapper
    extracts the last frame from each clip [B, T, C, H, W] for frame-based
    processing.  This means ConvNeXt lacks temporal context -- the Hiera
    model is the correct backbone for temporal tasks like activity and PSR.
    """
    from src.models.model import POPWMultiTaskModel

    base = POPWMultiTaskModel(pretrained=True, backbone_type="convnext_tiny").to(device)
    heads = ablation_cfg["heads"]

    if not heads["detection"]:
        base.detection_head = nn.Identity()
        base.fpn = nn.Identity()
    if not heads["activity"]:
        base.activity_head = nn.Identity()
    if not heads["psr"]:
        base.psr_head = nn.Identity()
    if not heads["pose"]:
        base.pose_head = nn.Identity()
        base.pose_film = nn.Identity()

    if ablation_cfg["freeze_backbone"]:
        for p in base.backbone.parameters():
            p.requires_grad = False
        logger.info("Backbone FROZEN")

    class _ConvNextClipModel(nn.Module):
        """Clip wrapper feeding the last frame through ConvNeXt.

        Output dict uses TierFModel's key naming for compatibility with
        compute_losses: act_logits, pose_6d, psr_logits, det_cls_logits/box.
        """
        def __init__(self, base_model):
            super().__init__()
            self.base = base_model

        def forward(self, clip):
            # clip: [B, T, C, H, W] -> last frame
            frame = clip[:, -1, :, :, :]
            out = self.base(frame, video_ids=None, clip_rgb=None)

            # Remap keys to TierFModel convention
            remapped = {}
            remapped["act_logits"] = out.get("act_logits")
            remapped["pose_6d"] = out.get("head_pose")
            remapped["psr_logits"] = out.get("psr_logits")
            remapped["det_cls_logits"] = out.get("cls_preds")
            remapped["det_box_logits"] = out.get("reg_preds")
            return remapped

    model = _ConvNextClipModel(base)
    return model


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------
def compute_losses(out, targets, heads_enabled, device):
    """Compute per-head losses for active heads.

    Handles both TierFModel (temporal mode) and POPWMultiTaskModel
    (frame-based) output formats.

    Returns (losses_dict, total_loss_scalar).
    """
    losses = {}

    # --- Activity ---
    if heads_enabled.get("activity", True) and "act_logits" in out:
        act = out["act_logits"]
        if isinstance(act, torch.Tensor) and act.ndim >= 1 and act.shape[-1] >= 1:
            act_t = torch.tensor([t["activity"] for t in targets], dtype=torch.long, device=device)
            # Handle class count mismatch between model and data
            num_cls = act.shape[-1]
            act_t = act_t.clamp(0, num_cls - 1)
            losses["act"] = F.cross_entropy(act.view(-1, num_cls), act_t.view(-1))

    # --- Pose ---
    if heads_enabled.get("pose", True) and "pose_6d" in out:
        pose_pred = out["pose_6d"]
        if isinstance(pose_pred, torch.Tensor) and pose_pred.ndim >= 1:
            pose_t = torch.stack([t["pose"] for t in targets]).to(device)
            # Handle 9-DoF vs 6-DoF (use last 6 dims for 9-DoF models)
            if pose_pred.shape[-1] == 9:
                pose_pred = pose_pred[..., :6]  # fwd3 + up3 only, drop position
            if pose_pred.shape[-1] == 6:
                losses["pose"] = F.l1_loss(pose_pred, pose_t)

    # --- PSR ---
    if heads_enabled.get("psr", True) and "psr_logits" in out:
        psr = out["psr_logits"]
        if isinstance(psr, torch.Tensor) and psr.shape[-1] == 11:
            psr_t = torch.stack([t["psr"] for t in targets]).to(device)
            # PSR logits may be [B, T, 11] (TierFModel temporal) or [B, 11] (ConvNeXt frame)
            psr_flat = psr[:, -1, :] if psr.dim() == 3 else psr
            losses["psr"] = F.binary_cross_entropy_with_logits(psr_flat, psr_t)

    # --- Detection ---
    if heads_enabled.get("detection", True):
        det = det_loss(out, targets, device)
        if det is not None:
            losses["det"] = det

    total = sum(losses.values()) if losses else torch.tensor(0.0, device=device)
    return losses, total


# ---------------------------------------------------------------------------
# Validation step
# ---------------------------------------------------------------------------
@torch.no_grad()
def val_step(model, loader, heads_enabled, device):
    model.eval()
    metrics = {}
    count = 0
    for clip, targets in loader:
        if clip is None:
            continue
        clip = clip.to(device)
        try:
            out = model(clip, mode="temporal")
        except (TypeError, ValueError):
            out = model(clip)
        losses, _ = compute_losses(out, targets, heads_enabled, device)
        for k, v in losses.items():
            metrics[k] = metrics.get(k, 0.0) + v.item() * len(targets)
        count += len(targets)
    if count == 0:
        return None
    return {k: v / count for k, v in metrics.items()}


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def run_ablation(ablation_name: str, epochs: int, output_dir: str):
    if ablation_name not in ABLATION_CONFIG:
        logger.error("Unknown ablation '%s'. Valid: %s", ablation_name, list(ABLATION_CONFIG.keys()))
        return 1

    cfg = ABLATION_CONFIG[ablation_name]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Ablation: %s (%s) | device=%s | epochs=%d", cfg["label"], ablation_name, device, epochs)
    logger.info("Heads: %s | freeze_backbone=%s", cfg["heads"], cfg["freeze_backbone"])

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # --- Dataset ---
    ds = AblationDataset(TRAIN_CSV)
    if len(ds) == 0:
        logger.error("Dataset empty")
        return 1

    val_size = max(1, len(ds) // 10)
    rest = len(ds) - val_size
    train_ds, val_ds = torch.utils.data.random_split(ds, [rest, val_size])
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=2, shuffle=True, num_workers=0, collate_fn=collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_fn
    )
    logger.info("Train: %d samples | Val: %d samples", len(train_ds), len(val_ds))

    # --- Model ---
    if cfg["model"] == "tier_f":
        model = build_tier_f_model(cfg, device)
    elif cfg["model"] == "convnext":
        model = build_convnext_model(cfg, device)
    else:
        logger.error("Unknown model type: %s", cfg["model"])
        return 1

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model params: %d total, %d trainable", total_params, trainable)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4, weight_decay=0.05,
    )

    # --- Training loop ---
    metrics_log = {"epochs": [], "val": []}
    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        train_metrics = {}
        n_batches = 0

        for clip, targets in train_loader:
            if clip is None:
                continue
            clip = clip.to(device)
            optimizer.zero_grad()

            if cfg["model"] == "tier_f":
                out = model(clip, mode="temporal")
            else:
                out = model(clip)

            losses, total = compute_losses(out, targets, cfg["heads"], device)
            if total > 0:
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            for k, v in losses.items():
                train_metrics[k] = train_metrics.get(k, 0.0) + v.item() * len(targets)
            n_batches += len(targets)

            if n_batches > 0 and n_batches % 100 == 0:
                logger.info("  [epoch %d] %d batches processed", epoch + 1, n_batches)

        elapsed = time.time() - t0
        avg_train = {k: v / n_batches for k, v in train_metrics.items()} if n_batches > 0 else {}
        logger.info("Epoch %d/%d done in %.1fs — train: %s", epoch + 1, epochs, elapsed,
                     {k: f"{v:.4f}" for k, v in sorted(avg_train.items())})

        # Validation
        avg_val = {}
        val_metrics = val_step(model, val_loader, cfg["heads"], device)
        if val_metrics is not None:
            avg_val = val_metrics
            logger.info("  val: %s", {k: f"{v:.4f}" for k, v in sorted(avg_val.items())})

        mean_val = sum(avg_val.values()) / max(len(avg_val), 1) if avg_val else float("inf")
        if mean_val < best_val_loss:
            best_val_loss = mean_val
            torch.save(model.state_dict(), out_path / "best.pth")

        metrics_log["epochs"].append({
            "epoch": epoch + 1, "train": avg_train, "val": avg_val, "elapsed_s": elapsed,
        })
        metrics_log["val"].append(avg_val)

    # --- Save outputs ---
    torch.save({
        "model_state_dict": model.state_dict(),
        "ablation": ablation_name, "config": cfg,
        "epochs": epochs, "trainable_params": trainable, "total_params": total_params,
    }, out_path / "latest.pth")

    with open(out_path / "metrics.json", "w") as f:
        json.dump(metrics_log, f, indent=2)

    summary = {
        "ablation": ablation_name, "label": cfg["label"],
        "heads": cfg["heads"], "freeze_backbone": cfg["freeze_backbone"],
        "epochs": epochs, "final_val": avg_val,
    }
    with open(out_path / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=== Ablation %s complete: saved to %s ===", cfg["label"], out_path)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Run a Tier F ablation (175 ULTIMATE GUIDE 6 rows 7-10)"
    )
    parser.add_argument("--ablation", required=True, choices=list(ABLATION_CONFIG.keys()),
                        help="Ablation flavor")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--output", default=None,
                        help="Output directory. Defaults to rf_stages/checkpoints/<name>")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    if args.output is None:
        args.output = str(
            _CODE_ROOT / "src" / "runs" / "rf_stages" / "checkpoints" / args.ablation.replace("-", "_")
        )

    return run_ablation(args.ablation, args.epochs, args.output)


if __name__ == "__main__":
    sys.exit(main())
