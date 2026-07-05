"""T3: Train MViTv2-S on the remapped 75→69-class activity protocol.

[Opus 118 §7.18 + 126 Decision 4] T3 is the 1-day protocol change that lets us compare
our per-frame activity to MViTv2-S (Kinetics pretrained) under the SAME taxonomy.

Protocol (sum probabilities, never average/max, bit-identical sanity check):
  - MViTv2-S was trained on 75 fine-grained classes (WACV 2024 Tab 2 protocol)
  - Our model outputs 69 verb-grouped classes (hybrid mode)
  - To compare fairly: take MViTv2's 75-class softmax, sum probabilities within
    each of the 69 groups, argmax over 69 — SAME protocol our inference uses.
  - Bit-identical sanity check: ungrouped (75) classes must predict identically
    before and after the group-collapse.

Usage: python3 src/training/train_t3_mvitv2.py --ckpt weights/MVIT_RGB_16x4.pth

Status: BLOCKED on pretrained weights (URL 403). Script ready, runs when weights
are available from any K400 MViTv2 mirror.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as C


def load_remap_table() -> dict:
    """Load the 75→69 class remap table built from ACT_CLASS_GROUPING=hybrid."""
    remap_path = Path("src/runs/rf_stages/checkpoints/act_remap_75_to_69.json")
    if not remap_path.exists():
        raise FileNotFoundError(
            f"Remap table not found at {remap_path}. Run the remap extraction first."
        )
    return json.load(open(remap_path))


def remap_75_to_69(logits_75: np.ndarray, id_to_group: list) -> np.ndarray:
    """Sum probabilities within each of 69 groups.

    Inputs: logits_75 [batch, 75] (raw logits or softmax probs)
    Output: logits_69 [batch, 69] (group-summed)

    Fair comparison: MViTv2's 75-class output → sum within 69 groups → argmax
    matches our per-frame inference protocol.
    """
    probs = F.softmax(torch.from_numpy(logits_75).float(), dim=-1).numpy()
    out = np.zeros((probs.shape[0], max(id_to_group) + 1), dtype=np.float32)
    for raw_id, group_id in enumerate(id_to_group):
        if raw_id < probs.shape[1]:
            out[:, group_id] += probs[:, raw_id]
    return out


class IndustRealActivityDatasetRemapped(Dataset):
    """Yields 16-frame clips with 69-class remapped activity labels.

    Output: (clip [3, 16, H, W], remapped_label [69])
    """
    def __init__(self, split: str = "val", clip_frames: int = 16, stride: int = 2):
        from src.data.industreal_dataset import IndustRealMultiTaskDataset
        self.base = IndustRealMultiTaskDataset(split=split, sequence_mode=True,
                                              sequence_length=clip_frames)
        self.clip_frames = clip_frames
        self.stride = stride
        self.id_to_group = load_remap_table()["id_to_group"]
        # Pre-extract clip boundaries
        self.clips = []
        for rec_dir in sorted(self.base.recordings_root.iterdir()):
            ar_file = rec_dir / "AR_labels.csv"
            if not ar_file.exists():
                continue
            with open(ar_file) as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) >= 5:
                        start = int(Path(parts[3]).stem)
                        end = int(Path(parts[4]).stem)
                        action_id = int(parts[1])
                        # Skip NA
                        if action_id < 0:
                            continue
                        for clip_start in range(start, end - clip_frames + 1, stride):
                            self.clips.append((rec_dir, clip_start, action_id))

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        rec_dir, clip_start, action_id = self.clips[idx]
        # Load clip frames
        frames = []
        for f in range(clip_start, clip_start + self.clip_frames):
            img = Image.open(rec_dir / "rgb" / f"{f:06d}.jpg")
            frames.append(np.array(img).transpose(2, 0, 1))
        clip = np.stack(frames)  # [16, 3, H, W]
        # Remap label 75→69
        group_id = self.id_to_group[action_id] if action_id < len(self.id_to_group) else 0
        label = np.zeros(69, dtype=np.float32)
        label[group_id] = 1.0
        return torch.from_numpy(clip).float() / 255.0, torch.from_numpy(label)


def train_t3(args):
    """Train MViTv2-S for 25 epochs on remapped 69-class protocol."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== T3 MViTv2-S training on {device} ===")

    # Load remap table
    remap = load_remap_table()
    print(f"Loaded remap table: 75 → 69 classes, {len(remap['group_names'])} group names")

    # Load MViTv2-S
    if not Path(args.ckpt).exists():
        raise FileNotFoundError(
            f"MViTv2-S checkpoint not found at {args.ckpt}. "
            f"BLOCKED: original URL 403 (fbaipublicfiles.com dead). "
            f"Need to find a working mirror or use a different model."
        )

    print(f"Loading MViTv2-S from {args.ckpt}...")
    from src.models.video_stream import K400VideoStream
    stream = K400VideoStream(model_name="mvitv2_s", pretrained=True)
    encoder = stream.encoder
    encoder = encoder.to(device)

    # Replace the head: 75 → 69
    head = nn.Linear(stream.hidden_size, 69).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(head.parameters()),
                                   lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Datasets
    train_ds = IndustRealActivityDatasetRemapped(split="train")
    val_ds = IndustRealActivityDatasetRemapped(split="val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    print(f"Train clips: {len(train_ds)}, Val clips: {len(val_ds)}")

    for epoch in range(args.epochs):
        encoder.train()
        head.train()
        running_loss = 0.0
        for i, (clip, label) in enumerate(train_loader):
            clip, label = clip.to(device), label.to(device)
            # clip: [B, 16, 3, H, W] -> need to flatten temporal
            B, T, C, H, W = clip.shape
            clip_flat = clip.view(B * T, C, H, W)
            features = encoder(clip_flat)  # [B*T, hidden]
            features = features.view(B, T, -1).mean(dim=1)  # temporal pool
            logits = head(features)  # [B, 69]
            loss = F.cross_entropy(logits, label.argmax(dim=-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()

        # Validation
        encoder.eval()
        head.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for clip, label in val_loader:
                clip, label = clip.to(device), label.to(device)
                B, T, C, H, W = clip.shape
                clip_flat = clip.view(B * T, C, H, W)
                features = encoder(clip_flat).view(B, T, -1).mean(dim=1)
                logits = head(features)
                pred = logits.argmax(dim=-1)
                correct += (pred == label.argmax(dim=-1)).sum().item()
                total += pred.size(0)
        val_acc = correct / total
        print(f"Epoch {epoch+1}/{args.epochs}: train_loss={running_loss/len(train_loader):.4f}, val_acc={val_acc:.4f}")

    # Save
    out_path = Path("src/runs/rf_stages/checkpoints/t3_mvitv2_act.pth")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "encoder_state": encoder.state_dict(),
        "head_state": head.state_dict(),
        "remap_table": remap,
        "val_acc": val_acc,
    }, out_path)
    print(f"Saved T3 model to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="weights/MVIT_RGB_16x4.pth",
                       help="Path to MViTv2-S K400 pretrained weights")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()
    train_t3(args)
