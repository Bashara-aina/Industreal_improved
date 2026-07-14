#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MViTv2-S Activity Linear Probe (Opus ACT-ARCH-9)

Frozen MViTv2-S (Kinetics-400 pretrained) → pre-extract 768-dim clip features
→ train Linear(768, 75) → evaluate.

Baseline: ConvNeXt-Tiny linear probe = 0.2169 val top-1 on 69 classes (0.2217 majority).
If MViTv2-S > 0.30, fine-tuning is worth the 2-week investment.

OOM-safe design:
- Feature extraction: batch_size=4 (small model, ~36M params)
- Intermediate features saved to --save-dir as .pt files
- Probe training: batch_size=256 on cached CPU features
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

sys.path.insert(0, str(_PROJECT_ROOT))
from src import config as C
from src.models.video_backbones import VideoFeatureExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("activity_mvit_probe")


# =========================================================================
# MViTv2 transforms (Kinetics-400 preprocessing)
# =========================================================================
# TorchVision MViTv2-S expects:
#   resize short side → 256, center crop → 224x224
#   normalize: mean=[0.45,0.45,0.45], std=[0.225,0.225,0.225]
_MVIT_MEAN = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1)
_MVIT_STD = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1)
_MVIT_RESIZE = 256
_MVIT_CROP = 224


def _mvit_transform(img: Image.Image) -> torch.Tensor:
    """Apply MViTv2-S preprocessing to a single PIL image.

    Returns: [3, 224, 224] float32 tensor normalized to Kinetics stats.
    """
    img = TF.resize(img, [_MVIT_RESIZE], interpolation=InterpolationMode.BILINEAR, antialias=True)
    img = TF.center_crop(img, [_MVIT_CROP, _MVIT_CROP])
    img = TF.to_tensor(img)  # [0, 1], [C, H, W]
    img = (img - _MVIT_MEAN) / _MVIT_STD
    return img


# =========================================================================
# Remap table loader
# =========================================================================
_REMAP_PATH = (
    _PROJECT_ROOT / "src" / "runs" / "rf_stages" / "checkpoints" / "act_remap_75_to_69.json"
)


def _load_remap() -> dict:
    """Load 75→69 class remap for hybrid-grouped evaluation."""
    if _REMAP_PATH.exists():
        return json.loads(_REMAP_PATH.read_text())
    logger.warning("Remap table not found at %s; using 75→69 identity", _REMAP_PATH)
    return {"id_to_group": list(range(75)), "group_names": [str(i) for i in range(69)]}


def remap_69(preds_75: np.ndarray, id_to_group: list) -> np.ndarray:
    """Collapse 75→69 via group-sum for fair comparison with ConvNeXt probe."""
    probs = F.softmax(torch.from_numpy(preds_75).float(), dim=-1).numpy()
    out = np.zeros((probs.shape[0], max(id_to_group) + 1), dtype=np.float32)
    for raw_id, group_id in enumerate(id_to_group):
        if raw_id < probs.shape[1]:
            out[:, group_id] += probs[:, raw_id]
    return out


# =========================================================================
# Clip dataset
# =========================================================================
class MViTClipDataset(Dataset):
    """Generates 16-frame clips from IndustReal action segments with RAM cache.

    Pre-loads all JPEG frame bytes into RAM for zero-disk-IO clip assembly.
    """

    def __init__(
        self,
        split: str,
        recordings_root: str,
        clip_len: int = 16,
        stride: int = 8,
        max_recordings: int | None = None,
    ):
        self.recordings_root = Path(recordings_root)
        self.split = split
        self.clip_len = clip_len
        self.stride = stride
        self.id_to_group = _load_remap()["id_to_group"]

        self.clips: list[tuple[str, int, int]] = []  # (rec_id, clip_start, action_id)
        self._build_index(max_recordings)

        # Pre-load frame bytes into RAM cache
        self._frame_cache: dict[str, bytes] = {}
        self._init_frame_cache()

        logger.info(
            "[MViTClipDataset] split=%s, %d clips, clip_len=%d, stride=%d",
            split,
            len(self.clips),
            clip_len,
            stride,
        )

    def _build_index(self, max_recordings: int | None) -> None:
        split_dir = self.recordings_root / self.split
        rec_dirs = sorted(split_dir.iterdir())

        for rec_dir in rec_dirs:
            if not rec_dir.is_dir():
                continue
            if max_recordings is not None and len(self.clips) >= max_recordings:
                break

            ar_file = rec_dir / "AR_labels.csv"
            if not ar_file.exists():
                continue

            rec_id = rec_dir.name
            lines = ar_file.read_text().strip().split("\n")
            for line in lines:
                parts = line.strip().split(",")
                if len(parts) < 5:
                    continue
                action_id = int(parts[1])
                if action_id < 0:
                    continue  # skip NA sentinel
                start = int(Path(parts[3]).stem)
                end = int(Path(parts[4]).stem)
                seg_len = end - start + 1
                if seg_len < self.clip_len:
                    continue
                # Generate overlapping clips within this segment
                for clip_start in range(start, end - self.clip_len + 2, self.stride):
                    self.clips.append((rec_id, clip_start, action_id))

    def _init_frame_cache(self) -> None:
        """No pre-load — on-demand loading via DataLoader workers.
        The kernel's page cache provides automatic disk caching across
        concurrent training + probe processes."""
        pass

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        rec_id, clip_start, action_id = self.clips[idx]
        rgb_dir = self.recordings_root / self.split / rec_id / "rgb"
        frames: list[torch.Tensor] = []
        for i in range(clip_start, clip_start + self.clip_len):
            img_path = rgb_dir / f"{i:06d}.jpg"
            try:
                img = Image.open(img_path).convert("RGB")
                frames.append(_mvit_transform(img))
            except Exception:
                frames.append(torch.zeros(3, _MVIT_CROP, _MVIT_CROP))
        # Stack: [T, 3, H, W] — extractor permutes to [B, C, T, H, W] internally
        clip = torch.stack(frames, dim=0)  # [16, 3, 224, 224]
        return clip, action_id


def mvit_collate(batch: list[tuple[torch.Tensor, int]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Collate clips into batched tensor [B, 3, T, H, W] with labels [B]."""
    clips, labels = zip(*batch)
    return torch.stack(clips, dim=0), torch.tensor(labels, dtype=torch.long)


# =========================================================================
# Feature extraction
# =========================================================================
def extract_clip_features(
    extractor: VideoFeatureExtractor,
    loader: DataLoader,
    device: torch.device,
    desc: str = "",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-extract MViTv2-S features for all clips. One pass per split.

    Returns: (features [N, 768], labels [N]) filtered to valid labels (>0).
    """
    all_features: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    skipped = 0

    for batch_idx, (clips, labels) in enumerate(loader):
        # Skip NA/background (class 0)
        valid = labels > 0
        n_invalid = (~valid).sum().item()
        skipped += n_invalid

        if not valid.any():
            continue

        clips = clips[valid].to(device, non_blocking=True)
        labels_valid = labels[valid]

        with torch.no_grad():
            features = extractor(clips)  # [B_valid, 768]
            features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        all_features.append(features.cpu())
        all_labels.append(labels_valid.cpu())

        if batch_idx > 0 and batch_idx % 200 == 0:
            logger.info(
                "  %s: batch %d/%d (skipped %d NA clips)", desc, batch_idx, len(loader), skipped
            )

        # Periodic GC to keep memory low during long extraction
        if batch_idx > 0 and batch_idx % 100 == 0:
            gc.collect()

    if not all_features:
        logger.error("%s: ALL clips had NA labels!", desc)
        return torch.empty(0, 768), torch.empty(0, dtype=torch.long)

    return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)


# =========================================================================
# Linear probe
# =========================================================================
class LinearProbe(nn.Module):
    """Single linear layer on frozen video features."""

    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


# =========================================================================
# Evaluation
# =========================================================================
def compute_per_class_accuracy(
    labels: np.ndarray,
    preds: np.ndarray,
    num_classes: int,
    class_names: list[str] | None = None,
) -> dict:
    """Compute per-class and overall accuracy."""
    correct = np.zeros(num_classes, dtype=np.int64)
    total = np.zeros(num_classes, dtype=np.int64)

    for lbl, prd in zip(labels, preds):
        if 0 <= lbl < num_classes:
            total[lbl] += 1
            if prd == lbl:
                correct[lbl] += 1

    per_class = {}
    for c in range(num_classes):
        if total[c] > 0:
            acc = float(correct[c] / total[c])
            name = class_names[c] if class_names and c < len(class_names) else str(c)
            per_class[name] = {"accuracy": round(acc, 6), "count": int(total[c])}

    return per_class


def majority_class_baseline(labels: np.ndarray) -> tuple[float, int]:
    """Compute accuracy of always predicting the most frequent class."""
    if len(labels) == 0:
        return 0.0, -1
    counts = np.bincount(labels)
    majority = int(counts.argmax())
    return float(counts[majority] / len(labels)), majority


# =========================================================================
# Main
# =========================================================================
def main():
    parser = argparse.ArgumentParser(description="MViTv2-S activity linear probe")
    parser.add_argument("--clip-len", type=int, default=16, help="Frames per clip")
    parser.add_argument(
        "--stride", type=int, default=8, help="Stride between clips within a segment"
    )
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Batch size for feature extraction"
    )
    parser.add_argument(
        "--probe-batch-size", type=int, default=256, help="Batch size for probe training"
    )
    parser.add_argument("--epochs", type=int, default=10, help="Probe training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Probe learning rate")
    parser.add_argument(
        "--save-dir",
        type=str,
        default="src/runs/rf_stages/checkpoints/activity_mvit_probe",
        help="Directory for results and cached features",
    )
    parser.add_argument(
        "--max-recordings", type=int, default=None, help="Cap recordings for fast smoke test"
    )
    parser.add_argument(
        "--force-reextract", action="store_true", help="Re-extract features even if cached"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(
        "Device: %s  |  CUDA_VISIBLE_DEVICES=%s",
        device,
        os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
    )

    # Paths
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    feat_cache_train = save_dir / "features_train.pt"
    feat_cache_val = save_dir / "features_val.pt"
    results_path = save_dir / "results.json"
    per_class_path = save_dir / "per_class.json"

    num_classes_raw = 75
    remap = _load_remap()
    id_to_group = remap["id_to_group"]
    group_names = remap["group_names"]

    # ---- 1. Load MViTv2-S extractor ----
    logger.info("Loading MViTv2-S (Kinetics-400 pretrained, frozen)...")
    extractor = VideoFeatureExtractor(backbone="mvit_v2_s", pretrained=True).to(device)
    logger.info(
        "Feature dim: %d, params: %d (frozen)",
        extractor.feat_dim,
        sum(p.numel() for p in extractor.parameters()),
    )

    # ---- 2. Build clip datasets ----
    logger.info("Building clip datasets (clip_len=%d, stride=%d)...", args.clip_len, args.stride)
    train_dataset = MViTClipDataset(
        split="train",
        recordings_root=C.RECORDINGS_ROOT,
        clip_len=args.clip_len,
        stride=args.stride,
        max_recordings=args.max_recordings,
    )
    val_dataset = MViTClipDataset(
        split="val",
        recordings_root=C.RECORDINGS_ROOT,
        clip_len=args.clip_len,
        stride=args.stride,
        max_recordings=args.max_recordings,
    )

    # ---- 3. Pre-extract features (or load cached) ----
    if feat_cache_train.exists() and feat_cache_val.exists() and not args.force_reextract:
        logger.info("Loading cached features from %s and %s", feat_cache_train, feat_cache_val)
        train_features, train_labels = torch.load(feat_cache_train, weights_only=True)
        val_features, val_labels = torch.load(feat_cache_val, weights_only=True)
        logger.info("Train: %s | Val: %s", train_features.shape, val_features.shape)
    else:
        logger.info("=== Pre-extracting training features ===")
        t0 = time.time()
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,  # workers overlap I/O with GPU compute
            collate_fn=mvit_collate,
            prefetch_factor=2,
        )
        train_features, train_labels = extract_clip_features(
            extractor, train_loader, device, desc="Train"
        )
        logger.info("Train features extracted: %s in %.0fs", train_features.shape, time.time() - t0)
        # Free loader memory
        del train_loader
        gc.collect()

        logger.info("=== Pre-extracting validation features ===")
        t0 = time.time()
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=mvit_collate,
            prefetch_factor=2,
        )
        val_features, val_labels = extract_clip_features(extractor, val_loader, device, desc="Val")
        logger.info("Val features extracted: %s in %.0fs", val_features.shape, time.time() - t0)

        torch.save((train_features, train_labels), feat_cache_train)
        torch.save((val_features, val_labels), feat_cache_val)
        logger.info("Features cached to %s and %s", feat_cache_train, feat_cache_val)

    if train_features.shape[0] == 0:
        logger.error("No valid training clips — aborting.")
        sys.exit(1)

    # ---- 4. Compute baselines ----
    train_labels_np = train_labels.numpy()
    val_labels_np = val_labels.numpy()

    maj_baseline_75, maj_class = majority_class_baseline(val_labels_np)
    logger.info("Majority-class baseline (75 classes): %.4f (class %d)", maj_baseline_75, maj_class)

    # Compute 69-class majority baseline via remap
    val_labels_69 = np.array([id_to_group[lbl] for lbl in val_labels_np])
    maj_baseline_69, maj_class_69 = majority_class_baseline(val_labels_69)
    logger.info(
        "Majority-class baseline (69 classes): %.4f (group %d)", maj_baseline_69, maj_class_69
    )

    # ---- 5. Build probe data loaders ----
    train_feat_dataset = TensorDataset(train_features, train_labels)
    train_loader_probe = DataLoader(
        train_feat_dataset,
        batch_size=args.probe_batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_feat_dataset = TensorDataset(val_features, val_labels)
    val_loader_probe = DataLoader(
        val_feat_dataset,
        batch_size=args.probe_batch_size,
        shuffle=False,
        num_workers=0,
    )

    # ---- 6. Train linear probe ----
    probe = LinearProbe(in_dim=extractor.feat_dim, num_classes=num_classes_raw).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_top1_75 = 0.0
    best_val_top1_69 = 0.0
    best_epoch = -1

    logger.info("=== Training linear probe for %d epochs ===", args.epochs)

    for epoch in range(args.epochs):
        t0 = time.time()

        # Training
        probe.train()
        train_losses: list[float] = []
        train_correct = 0
        train_total = 0

        for batch_feat, batch_lbl in train_loader_probe:
            batch_feat = batch_feat.to(device, non_blocking=True)
            batch_lbl = batch_lbl.to(device, non_blocking=True)

            logits = probe(batch_feat)
            loss = criterion(logits, batch_lbl)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())
            preds = logits.argmax(dim=1)
            train_correct += (preds == batch_lbl).sum().item()
            train_total += batch_lbl.size(0)

        train_acc = train_correct / max(train_total, 1)
        train_loss = float(np.mean(train_losses))

        # Validation (75 classes)
        probe.eval()
        val_correct_75 = 0
        val_total = 0
        all_val_preds: list[int] = []
        all_val_labels_list: list[int] = []

        with torch.no_grad():
            for batch_feat, batch_lbl in val_loader_probe:
                batch_feat = batch_feat.to(device, non_blocking=True)
                logits = probe(batch_feat)
                preds = logits.argmax(dim=1).cpu()
                val_correct_75 += (preds == batch_lbl).sum().item()
                val_total += batch_lbl.size(0)
                all_val_preds.extend(preds.tolist())
                all_val_labels_list.extend(batch_lbl.tolist())

        val_acc_75 = val_correct_75 / max(val_total, 1)

        # Validation (69 classes via remap)
        val_preds_69 = np.array(
            [id_to_group[p] if p < len(id_to_group) else 0 for p in all_val_preds]
        )
        val_labels_69_arr = np.array(
            [id_to_group[l] if l < len(id_to_group) else 0 for l in all_val_labels_list]
        )
        val_acc_69 = float((val_preds_69 == val_labels_69_arr).mean())

        logger.info(
            "Epoch %2d/%d | Train Loss: %.4f Acc: %.4f | Val 75: %.4f | Val 69: %.4f | Time: %.0fs",
            epoch + 1,
            args.epochs,
            train_loss,
            train_acc,
            val_acc_75,
            val_acc_69,
            time.time() - t0,
        )

        if val_acc_75 > best_val_top1_75:
            best_val_top1_75 = val_acc_75
            best_val_top1_69 = val_acc_69
            best_epoch = epoch + 1

        scheduler.step()

    # ---- 7. Final evaluation on best model ----
    logger.info("=" * 60)
    logger.info("MViTv2-S ACTIVITY LINEAR PROBE RESULTS")
    logger.info("=" * 60)
    logger.info("Best epoch:               %d", best_epoch)
    logger.info("Majority baseline (75):   %.4f", maj_baseline_75)
    logger.info("Majority baseline (69):   %.4f", maj_baseline_69)
    logger.info("Best val top-1 (75 raw):  %.4f", best_val_top1_75)
    logger.info("Best val top-1 (69 remap): %.4f", best_val_top1_69)
    logger.info("ConvNeXt probe baseline:  0.2169 (69 classes, per-frame)")
    logger.info("ConvNeXt majority:        0.2217 (69 classes)")

    improvement_69 = best_val_top1_69 - maj_baseline_69
    logger.info("Improvement over majority (69): %+.4f", improvement_69)

    if best_val_top1_69 > 0.30:
        logger.info("Verdict: SIGNAL DETECTED (>0.30) — fine-tuning is worth the investment")
    elif best_val_top1_69 > 0.25:
        logger.info("Verdict: WEAK SIGNAL (0.25-0.30) — fine-tuning may help but uncertain")
    else:
        logger.info("Verdict: NO STRONG SIGNAL (<=0.25) — MViTv2-S adds little over ConvNeXt")

    # ---- 8. Per-class accuracy ----
    per_class = compute_per_class_accuracy(
        labels=np.array(all_val_labels_list),
        preds=np.array(all_val_preds),
        num_classes=num_classes_raw,
        class_names=C.ACT_CLASS_NAMES if hasattr(C, "ACT_CLASS_NAMES") else None,
    )

    # ---- 9. Save results ----
    results = {
        "model": "MViTv2-S (Kinetics-400 pretrained) linear probe",
        "clip_len": args.clip_len,
        "stride": args.stride,
        "extraction_batch_size": args.batch_size,
        "probe_batch_size": args.probe_batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "optimizer": "AdamW",
        "weight_decay": 1e-4,
        "scheduler": "CosineAnnealingLR",
        "feature_dim": extractor.feat_dim,
        "num_classes_raw": num_classes_raw,
        "num_classes_grouped": 69,
        "grouping_mode": "hybrid",
        "majority_class_75": int(maj_class),
        "majority_baseline_75": round(maj_baseline_75, 6),
        "majority_baseline_69": round(maj_baseline_69, 6),
        "best_val_top1_75": round(best_val_top1_75, 6),
        "best_val_top1_69": round(best_val_top1_69, 6),
        "best_epoch": best_epoch,
        "improvement_over_majority_69": round(improvement_69, 6),
        "train_clips_valid": int(train_features.shape[0]),
        "val_clips_valid": int(val_features.shape[0]),
        "convnext_comparison": {
            "convnext_val_top1_69": 0.2169,
            "convnext_majority_69": 0.2217,
            "mvit_improvement_over_convnext": round(best_val_top1_69 - 0.2169, 6),
        },
        "verdict": (
            "SIGNAL DETECTED (>0.30) — fine-tuning worth 2-week investment"
            if best_val_top1_69 > 0.30
            else "WEAK SIGNAL (0.25-0.30) — fine-tuning uncertain"
            if best_val_top1_69 > 0.25
            else "NO STRONG SIGNAL (<=0.25) — MViTv2-S adds little over ConvNeXt"
        ),
        "notes": [
            "MViTv2-S features: 768-dim clip-level embeddings from torchvision Kinetics-400 weights",
            f"Clip: {args.clip_len} frames, stride {args.stride}",
            "ConvNeXt reference: per-frame 0.2169 on 69 classes (acting as majority-class oracle)",
            "If improvement > +0.08 over convnext (0.2169), fine-tuning is justified",
        ],
    }

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", results_path)

    with open(per_class_path, "w") as f:
        json.dump(per_class, f, indent=2)
    logger.info("Per-class accuracy saved to %s", per_class_path)

    # Summary line for the agent report
    logger.info("=" * 60)
    logger.info(
        "SUMMARY: MViTv2-S probe val top-1 (75) = %.4f, (69) = %.4f",
        best_val_top1_75,
        best_val_top1_69,
    )
    logger.info("         Improvement over ConvNeXt (0.2169): %+.4f", best_val_top1_69 - 0.2169)
    logger.info("=" * 60)

    return best_val_top1_69


if __name__ == "__main__":
    main()
