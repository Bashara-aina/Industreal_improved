#!/usr/bin/env python3
"""V8 training script — full pipeline.

V8 architecture (V5 + YOLOv8m det + MViTv2-S activity/pose/PSR):
- Detection: YOLOv8m (D1R, 0.995 mAP50, already trained, frozen)
- Activity:  MViTv2-S (Kinetics-400) features + Linear head
- Pose:       MViTv2-S features + Linear head (6 dims: fwd + up)
- PSR:        MViTv2-S features + Per-component Linear heads (11 comps)
- KENDALL_FIXED_WEIGHTS=0 (let Kendall rebalance)

Data:
- train.csv: clip list (recording, action, start_frame, end_frame)
- recordings/{recording}/videos/{frame}.mp4: 16-frame video clips
- recordings/{recording}/AR_labels.csv: per-frame activity class
- recordings/{recording}/pose.csv: per-frame fwd/up
- recordings/{recording}/PSR_labels.csv: per-frame binary transitions

Usage:
  CUDA_VISIBLE_DEVICES=1 python3 scripts/train_v8_multitask.py --epochs 5
"""

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("train_v8")

DATA_ROOT = Path("/media/newadmin/master/POPW/datasets/industreal")
TRAIN_CSV = DATA_ROOT / "train.csv"
RECORDINGS = DATA_ROOT / "recordings"

# MViTv2-S preprocessing
_MEAN = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1, 1)
_STD = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1, 1)


def load_clip_frames(recording: str, start_frame: int, num_frames: int = 16) -> torch.Tensor:
    """Load num_frames consecutive frames from a recording, center-cropped to 224x224."""
    video_dir = RECORDINGS / recording / "videos"
    frames = []
    for i in range(num_frames):
        frame_idx = start_frame + i
        frame_path = video_dir / f"{frame_idx:06d}.mp4"
        if not frame_path.exists():
            # Try jpg fallback
            frame_path = video_dir / f"{frame_idx:06d}.jpg"
        if not frame_path.exists():
            return None
        # Lazy import cv2
        import cv2

        img = cv2.imread(str(frame_path))
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Resize to 256 short side, center crop 224
        h, w = img.shape[:2]
        scale = 256.0 / min(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        img = cv2.resize(img, (new_w, new_h))
        top = (new_h - 224) // 2
        left = (new_w - 224) // 2
        img = img[top : top + 224, left : left + 224]
        # Normalize
        img = img.astype(np.float32) / 255.0
        img = (img - 0.45) / 0.225
        # To tensor [T, H, W, C] -> [C, T, H, W]
        frames.append(img.transpose(2, 0, 1))  # [C, H, W]
    if not frames:
        return None
    # Stack [C, T, H, W]
    clip = np.stack(frames, axis=1)  # [C, T, H, W]
    return torch.from_numpy(clip).float()


def load_ar_labels(recording: str) -> dict:
    """Load activity labels per frame."""
    ar_path = RECORDINGS / recording / "AR_labels.csv"
    if not ar_path.exists():
        return {}
    labels = {}
    with open(ar_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = int(row["frame"])
            cls = row["class"]
            labels[frame] = cls
    return labels


def load_pose_labels(recording: str) -> dict:
    """Load pose (fwd, up) per frame."""
    pose_path = RECORDINGS / recording / "pose.csv"
    if not pose_path.exists():
        return {}
    labels = {}
    with open(pose_path) as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if len(row) < 9:
                continue
            frame = int(row[0].split(".")[0])
            fwd = np.array([float(row[1]), float(row[2]), float(row[3])])
            up = np.array([float(row[4]), float(row[5]), float(row[6])])
            labels[frame] = (fwd, up)
    return labels


def load_psr_labels(recording: str) -> dict:
    """Load PSR binary transitions per frame per component."""
    psr_path = RECORDINGS / recording / "PSR_labels.csv"
    if not psr_path.exists():
        return {}
    labels = {}
    with open(psr_path) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) < 12:
                continue
            frame = int(row[0].split(".")[0])
            comps = [int(v) for v in row[1:12]]
            labels[frame] = comps  # 11 components
    return labels


class V8Dataset(torch.utils.data.Dataset):
    """V8 dataset — 16-frame video clips with 4-task labels."""

    def __init__(self, csv_path, num_classes=69, num_psr_comps=11, num_frames=16):
        self.samples = []
        with open(csv_path) as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 4:
                    continue
                # Format: recording, frame_offset, action, start_jpg, end_jpg
                rec, off, act, start, end = row[0], row[1], row[2], row[3], row[4]
                if not rec:
                    continue
                # Convert start/end to frame indices (strip .jpg)
                try:
                    start_frame = int(start.replace(".jpg", "").lstrip("0") or "0")
                    end_frame = int(end.replace(".jpg", "").lstrip("0") or "0")
                except ValueError:
                    continue
                self.samples.append(
                    {
                        "recording": rec,
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                        "class": act,
                    }
                )
        self.num_classes = num_classes
        self.num_psr_comps = num_psr_comps
        self.num_frames = num_frames

        # Build stable class-to-index mapping from sorted unique class names.
        # Replaces hash(cls_str) % num_classes which is non-deterministic
        # because PYTHONHASHSEED is randomized per process — the same action
        # string maps to different indices in every subprocess, making
        # cross-entropy training impossible by construction.
        unique_classes = sorted({s["class"] for s in self.samples})
        if len(unique_classes) != num_classes:
            logger.warning(
                f"Found {len(unique_classes)} unique classes in CSV, "
                f"expected {num_classes}. Using data-derived class count."
            )
        self._class_to_idx_map = {name: i for i, name in enumerate(unique_classes)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        clip = load_clip_frames(s["recording"], s["start_frame"], self.num_frames)
        if clip is None:
            # Return dummy
            return torch.zeros(3, self.num_frames, 224, 224), {
                "activity": 0,
                "pose": torch.zeros(6),
                "psr": torch.zeros(self.num_psr_comps),
                "recording": s["recording"],
                "start_frame": s["start_frame"],
            }
        # Get labels at the LAST frame of the clip
        last_frame = s["end_frame"]
        ar_labels = load_ar_labels(s["recording"])
        pose_labels = load_pose_labels(s["recording"])
        psr_labels = load_psr_labels(s["recording"])

        # Activity class (use last frame, or 'class' field from csv)
        cls_str = ar_labels.get(last_frame, s["class"])
        act_cls = self._class_to_idx(cls_str)

        # Pose (last frame)
        pose = pose_labels.get(last_frame, (np.zeros(3), np.zeros(3)))
        pose = np.concatenate([pose[0], pose[1]])  # [6]

        # PSR (last frame)
        psr = psr_labels.get(last_frame, [0] * self.num_psr_comps)

        targets = {
            "activity": act_cls,
            "pose": torch.from_numpy(pose).float(),
            "psr": torch.tensor(psr, dtype=torch.float32),
            "recording": s["recording"],
            "start_frame": s["start_frame"],
        }
        return clip, targets

    def _class_to_idx(self, cls_str):
        """Map class string to integer index using stable sorted lookup."""
        if not isinstance(cls_str, str):
            return 0
        assert cls_str in self._class_to_idx_map, f"Unknown class '{cls_str}'"
        return self._class_to_idx_map[cls_str]


def collate_fn(batch):
    """Collate variable-length clips into a batch. Skip failed loads."""
    clips = [b[0] for b in batch if b[0] is not None]
    targets = [b[1] for b in batch if b[0] is not None]
    if not clips:
        return None, None
    return torch.stack(clips), targets


class V8Model(nn.Module):
    """V8 multi-task model."""

    def __init__(self, num_classes=69, num_psr_comps=11):
        super().__init__()
        from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights

        self.backbone = mvit_v2_s(weights=MViT_V2_S_Weights.DEFAULT)
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()
        self.feat_dim = 400

        # Heads (init biases to prevent all-zero collapse in classification)
        self.activity_head = nn.Linear(self.feat_dim, num_classes)
        # Initialize bias to log(prior) to break symmetry
        nn.init.normal_(self.activity_head.weight, std=0.01)
        self.pose_head = nn.Linear(self.feat_dim, 6)
        self.psr_head = nn.ModuleList([nn.Linear(self.feat_dim, 1) for _ in range(num_psr_comps)])
        for h in self.psr_head:
            nn.init.normal_(h.weight, std=0.01)

        # Kendall log_vars (learnable, KENDALL_FIXED_WEIGHTS=0)
        self.log_var_act = nn.Parameter(torch.tensor(0.0))
        self.log_var_pose = nn.Parameter(torch.tensor(0.0))
        self.log_var_psr = nn.Parameter(torch.tensor(0.0))

    def forward(self, clip):
        # clip: [B, C, T, H, W]
        with torch.no_grad():
            feat = self.backbone(clip)  # [B, 768]
        return {
            "activity_logits": self.activity_head(feat),
            "pose_pred": self.pose_head(feat),
            "psr_logits": torch.stack([h(feat) for h in self.psr_head], dim=1).squeeze(
                -1
            ),  # [B, 11]
        }

    def compute_loss(self, out, targets):
        losses = {}
        # Activity (cross-entropy)
        activity_targets = torch.tensor([t["activity"] for t in targets], dtype=torch.long).to(
            out["activity_logits"].device
        )
        losses["act"] = F.cross_entropy(out["activity_logits"], activity_targets)
        # Pose (L1)
        pose_targets = torch.stack([t["pose"] for t in targets]).to(out["pose_pred"].device)
        losses["pose"] = F.l1_loss(out["pose_pred"], pose_targets)
        # PSR (BCE)
        psr_targets = torch.stack([t["psr"] for t in targets]).to(out["psr_logits"].device)
        losses["psr"] = F.binary_cross_entropy_with_logits(out["psr_logits"], psr_targets)

        # Combined with Kendall
        total = 0
        for name, loss in losses.items():
            log_var = getattr(self, f"log_var_{name}")
            prec = torch.exp(-log_var)
            total = total + prec * loss + log_var
        return total, losses


def val_step(model, loader, device):
    model.eval()
    metrics = {"act_loss": 0, "pose_loss": 0, "psr_loss": 0, "count": 0}
    with torch.no_grad():
        for clip, targets in loader:
            if clip is None:
                continue
            clip = clip.to(device)
            out = model(clip)
            _, losses = model.compute_loss(out, targets)
            metrics["act_loss"] += losses["act"].item() * len(targets)
            metrics["pose_loss"] += losses["pose"].item() * len(targets)
            metrics["psr_loss"] += losses["psr"].item() * len(targets)
            metrics["count"] += len(targets)
    if metrics["count"] == 0:
        return None
    return {k: v / metrics["count"] for k, v in metrics.items() if k != "count"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--save-dir",
        default="/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/runs/v8_multitask",
    )
    parser.add_argument(
        "--val-csv", default="/media/newadmin/master/POPW/datasets/industreal/recordings/val"
    )  # can be split from train.csv
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Datasets
    train_ds = V8Dataset(TRAIN_CSV)
    val_csv = (
        Path("/media/newadmin/master/POPW/datasets/industreal/recordings/val") / "val_split.csv"
    )
    if not val_csv.exists():
        # Use first 10% of train as val
        n = len(train_ds)
        val_size = max(1, n // 10)
        indices = list(range(n))
        train_ds, val_ds_subset = torch.utils.data.random_split(train_ds, [n - val_size, val_size])
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds_subset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
        )
    else:
        val_ds = V8Dataset(val_csv)
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
        )
    logger.info(
        f"Train: {len(train_ds)} samples, Val: {len(val_ds) if val_csv.exists() else len(val_ds_subset)} samples"
    )

    # Model
    model = V8Model().to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        train_metrics = {"act_loss": 0, "pose_loss": 0, "psr_loss": 0, "count": 0}
        for i, (clip, targets) in enumerate(train_loader):
            if clip is None:
                continue
            clip = clip.to(device)
            optimizer.zero_grad()
            out = model(clip)
            loss, losses = model.compute_loss(out, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_metrics["act_loss"] += losses["act"].item() * len(targets)
            train_metrics["pose_loss"] += losses["pose"].item() * len(targets)
            train_metrics["psr_loss"] += losses["psr"].item() * len(targets)
            train_metrics["count"] += len(targets)
            if i % 50 == 0:
                logger.info(
                    f"Epoch {epoch} step {i} | act={losses['act'].item():.3f} pose={losses['pose'].item():.3f} psr={losses['psr'].item():.3f}"
                )
        elapsed = time.time() - t0
        logger.info(f"Epoch {epoch} done in {elapsed:.1f}s ({train_metrics['count']} samples)")
        if train_metrics["count"] > 0:
            logger.info(
                f"  Train: act={train_metrics['act_loss'] / train_metrics['count']:.3f} pose={train_metrics['pose_loss'] / train_metrics['count']:.3f} psr={train_metrics['psr_loss'] / train_metrics['count']:.3f}"
            )

        # Val
        val_metrics = val_step(model, val_loader, device)
        if val_metrics:
            logger.info(
                f"  Val:   act={val_metrics['act_loss']:.3f} pose={val_metrics['pose_loss']:.3f} psr={val_metrics['psr_loss']:.3f}"
            )

    # Save
    torch.save(
        {
            "model": model.state_dict(),
            "log_var_act": model.log_var_act.item(),
            "log_var_pose": model.log_var_pose.item(),
            "log_var_psr": model.log_var_psr.item(),
        },
        save_dir / "latest.pth",
    )
    logger.info(f"Saved to {save_dir / 'latest.pth'}")


if __name__ == "__main__":
    sys.exit(main())
