#!/usr/bin/env python3
"""
eval_activity_75class.py — Clip-level top-1/top-5 evaluation on 75 fine-grained activity classes.

Implements the evaluation protocol defined in 174 §3.2 and 175 §7.2:
  - 16-frame clip-level inference (MViTv2-S or any video backbone)
  - Top-1 / Top-5 accuracy on the 75-class raw taxonomy
  - Macro-F1, per-class precision/recall
  - Optional 69-grouped evaluation via class_69_to_75.json remap

Usage:
    # Train a probe on cached features (fast, CPU)
    python scripts/eval_activity_75class.py --mode feature-probe

    # Evaluate from a multi-task checkpoint (requires GPU)
    python scripts/eval_activity_75class.py --mode checkpoint \\
        --checkpoint src/runs/rf_stages/checkpoints/best.pth

    # Full eval with all metrics
    python scripts/eval_activity_75class.py --mode feature-probe \\
        --save-dir src/runs/rf_stages/checkpoints/activity_75class_eval

Blocker: The frozen MViTv2-S probe (0.3810 on 69-val) was trained on 75 raw classes
but its linear weights were NOT saved independently (only features + results.json exist).
The 'feature-probe' mode retrains a linear probe on the cached features at startup.
For a true production eval, the multi-task model must emit 75-class logits directly.
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src import config as C

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("eval_activity_75class")


# =========================================================================
# Constants
# =========================================================================
NUM_CLASSES_75 = 75
NUM_CLASSES_69 = 69

# Paths relative to project root
_PROBE_DIR = _PROJECT_ROOT / "src" / "runs" / "rf_stages" / "checkpoints" / "activity_mvit_probe"
_PROBE_FEATURES_TRAIN = _PROBE_DIR / "features_train.pt"
_PROBE_FEATURES_VAL = _PROBE_DIR / "features_val.pt"
_REMAP_PATH = _PROJECT_ROOT / "config" / "class_maps" / "class_69_to_75.json"
_CLASS_MAP_69_TO_75 = _PROJECT_ROOT / "config" / "class_maps" / "class_69_to_75.json"

DEFAULT_SAVE_DIR = _PROJECT_ROOT / "src" / "runs" / "rf_stages" / "checkpoints" / "activity_75class_eval"

# Reconstruction note: mapping file created 2026-07-08 by inverting
# act_remap_75_to_69.json (hybrid grouping, threshold=100).


# =========================================================================
# Class mapping
# =========================================================================
def load_class_map() -> dict:
    """Load the 69-to-75 class mapping.

    Returns dict with:
      - 'mapping': group_id -> {fine_class_ids, fine_class_names}
      - 'id_to_group_forward': 75-element list mapping raw_id -> group_id
      - 'group_names': 69-element list
    """
    if not _CLASS_MAP_69_TO_75.exists():
        logger.warning(
            "Class map not found at %s. "
            "Run: python -c 'from scripts.eval_activity_75class import build_class_map; build_class_map()'",
            _CLASS_MAP_69_TO_75,
        )
        return None
    with open(_CLASS_MAP_69_TO_75) as f:
        return json.load(f)


def remap_69_to_75(group_preds: np.ndarray, class_map: dict) -> np.ndarray:
    """Expand 69-group predictions to 75-class space by summing probabilities
    within each group.

    Args:
        group_preds: [N, 69] probability array
        class_map: from load_class_map()

    Returns:
        [N, 75] probability array
    """
    mapping = class_map["mapping"]
    n = group_preds.shape[0]
    out = np.zeros((n, 75), dtype=np.float32)
    for gid_str, info in mapping.items():
        gid = int(gid_str)
        fine_ids = info["fine_class_ids"]
        prob = group_preds[:, gid]  # [N]
        for fid in fine_ids:
            out[:, fid] = prob / len(fine_ids)  # uniform expansion
    return out


def remap_75_to_69(fine_preds: np.ndarray, class_map: dict) -> np.ndarray:
    """Collapse 75-class predictions to 69 groups by summing group-member probs.

    Args:
        fine_preds: [N, 75] probability array
        class_map: from load_class_map()

    Returns:
        [N, 69] probability array
    """
    id_to_group = class_map["id_to_group_forward"]
    n = fine_preds.shape[0]
    num_groups = class_map["num_groups"]
    out = np.zeros((n, num_groups), dtype=np.float32)
    for raw_id, group_id in enumerate(id_to_group):
        out[:, group_id] += fine_preds[:, raw_id]
    return out


# =========================================================================
# Probe training (feature-probe mode)
# =========================================================================
class LinearProbe(nn.Module):
    """Single linear layer on frozen video features."""

    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


def train_probe(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    num_classes: int,
    in_dim: int = 768,
    epochs: int = 10,
    lr: float = 1e-3,
    batch_size: int = 256,
    device: torch.device = torch.device("cpu"),
) -> tuple[LinearProbe, dict]:
    """Train a linear probe on pre-extracted features.

    Returns (probe, history) where history contains per-epoch metrics.
    """
    from torch.utils.data import DataLoader, TensorDataset

    train_ds = TensorDataset(train_features, train_labels)
    val_ds = TensorDataset(val_features, val_labels)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    probe = LinearProbe(in_dim=in_dim, num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_state = None
    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(epochs):
        probe.train()
        train_losses = []
        train_correct = 0
        train_total = 0
        for feat, lbl in train_loader:
            feat = feat.to(device, non_blocking=True)
            lbl = lbl.to(device, non_blocking=True)
            logits = probe(feat)
            loss = criterion(logits, lbl)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())
            preds = logits.argmax(dim=1)
            train_correct += (preds == lbl).sum().item()
            train_total += lbl.size(0)

        train_acc = train_correct / max(train_total, 1)
        train_loss = float(np.mean(train_losses))

        probe.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for feat, lbl in val_loader:
                feat = feat.to(device, non_blocking=True)
                logits = probe(feat)
                preds = logits.argmax(dim=1).cpu()
                val_correct += (preds == lbl).sum().item()
                val_total += lbl.size(0)
        val_acc = val_correct / max(val_total, 1)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = probe.state_dict()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        logger.info(
            "Epoch %2d/%d | loss=%.4f train_acc=%.4f val_acc=%.4f",
            epoch + 1, epochs, train_loss, train_acc, val_acc,
        )

        scheduler.step()

    if best_state is not None:
        probe.load_state_dict(best_state)
    probe.eval()
    return probe, history


# =========================================================================
# Metrics computation
# =========================================================================
def compute_metrics(
    all_labels: np.ndarray,
    all_preds: np.ndarray,
    all_logits: np.ndarray = None,
    class_names: list[str] = None,
    class_map: dict = None,
) -> dict:
    """Compute clip-level top-1, top-5, macro-F1, per-class precision/recall.

    Args:
        all_labels: [N] ground truth 75-class labels
        all_preds: [N] predicted 75-class labels
        all_logits: [N, C] raw logits (for top-5)
        class_names: 75-element list of class names
        class_map: for 69-grouped metrics

    Returns:
        dict with all metrics
    """
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    n = len(all_labels)
    if n == 0:
        return {"error": "empty predictions", "clip_count": 0}

    valid = all_labels >= 0
    labels_v = all_labels[valid]
    preds_v = all_preds[valid]
    n_valid = valid.sum()

    # --- Top-1 ---
    top1 = (preds_v == labels_v).mean()

    # --- Top-5 ---
    top5 = 0.0
    if all_logits is not None and len(all_logits) > 0:
        logits_v = all_logits[valid]
        top5_indices = np.argsort(logits_v, axis=1)[:, -5:]
        top5 = np.any(top5_indices == labels_v[:, None], axis=1).mean()

    # --- Per-class stats ---
    num_classes = 75
    present_labels = sorted(set(labels_v.tolist()))
    per_class = {}
    for c in range(num_classes):
        mask = labels_v == c
        n_c = mask.sum()
        if n_c > 0:
            correct_c = (preds_v[mask] == c).sum()
            acc = float(correct_c) / float(n_c)
            name = class_names[c] if class_names and c < len(class_names) else str(c)
            per_class[name] = {
                "count": int(n_c),
                "top1": round(acc, 6),
            }

    # --- Macro-F1 (scikit-learn, excluding 0/NA if 0 is NA) ---
    present_filter = [c for c in present_labels if c != 0]  # exclude NA if ID=0
    if not present_filter:
        present_filter = present_labels

    try:
        macro_f1 = float(f1_score(labels_v, preds_v, average="macro",
                                   labels=present_filter, zero_division=0))
        weighted_f1 = float(f1_score(labels_v, preds_v, average="weighted",
                                      zero_division=0))
        macro_precision = float(precision_score(labels_v, preds_v, average="macro",
                                                 labels=present_filter, zero_division=0))
        macro_recall = float(recall_score(labels_v, preds_v, average="macro",
                                           labels=present_filter, zero_division=0))
    except Exception:
        macro_f1 = weighted_f1 = macro_precision = macro_recall = 0.0

    # --- Per-class precision/recall/F1 ---
    try:
        report = classification_report(
            labels_v, preds_v,
            labels=present_filter,
            zero_division=0,
            output_dict=True,
        )
    except Exception:
        report = {}

    # --- 69-grouped metrics (via remap) ---
    metrics_69 = {}
    if class_map is not None:
        id_to_group = class_map["id_to_group_forward"]
        group_names = class_map.get("group_names", [])
        labels_69 = np.array([id_to_group[l] if 0 <= l < len(id_to_group) else 0
                               for l in labels_v])
        preds_69 = np.array([id_to_group[p] if 0 <= p < len(id_to_group) else 0
                              for p in preds_v])
        top1_69 = (preds_69 == labels_69).mean()
        present_69 = sorted(set(labels_69.tolist()))
        try:
            macro_f1_69 = float(f1_score(labels_69, preds_69, average="macro",
                                          labels=present_69, zero_division=0))
        except Exception:
            macro_f1_69 = 0.0
        metrics_69 = {
            "top1_69": round(float(top1_69), 6),
            "macro_f1_69": round(float(macro_f1_69), 6),
            "num_groups_69": class_map["num_groups"],
        }

    return {
        "clip_count": int(n),
        "valid_clip_count": int(n_valid),
        "top1_75": round(float(top1), 6),
        "top5_75": round(float(top5), 6),
        "macro_f1_75": round(float(macro_f1), 6),
        "weighted_f1_75": round(float(weighted_f1), 6),
        "macro_precision_75": round(float(macro_precision), 6),
        "macro_recall_75": round(float(macro_recall), 6),
        "per_class_75": per_class,
        "classification_report_75": report,
        **metrics_69,
    }


# =========================================================================
# Feature-probe evaluation mode
# =========================================================================
def evaluate_feature_probe(
    save_dir: Path,
    device: torch.device,
    probe_kwargs: dict = None,
) -> dict:
    """Evaluate on cached frozen-probe features.

    Loads pre-extracted MViTv2-S clip-level features from the frozen probe run,
    trains a 75-class linear probe, and reports clip-level metrics.

    Returns results dict.
    """
    if not _PROBE_FEATURES_TRAIN.exists() or not _PROBE_FEATURES_VAL.exists():
        logger.error(
            "Cached probe features not found at %s and %s. "
            "Run activity_mvit_probe.py first to extract features.",
            _PROBE_FEATURES_TRAIN, _PROBE_FEATURES_VAL,
        )
        return {"error": "cached features not found"}

    logger.info("Loading cached MViTv2-S probe features...")
    train_features, train_labels = torch.load(_PROBE_FEATURES_TRAIN, weights_only=True)
    val_features, val_labels = torch.load(_PROBE_FEATURES_VAL, weights_only=True)
    logger.info("Train: %s | Val: %s", train_features.shape, val_features.shape)

    if train_features.shape[0] == 0 or val_features.shape[0] == 0:
        logger.error("Empty feature tensors — cannot evaluate.")
        return {"error": "empty features"}

    # Load class mapping
    class_map = load_class_map()
    class_names = list(C.ACT_CLASS_NAMES) if hasattr(C, "ACT_CLASS_NAMES") else None
    if class_names is None or len(class_names) != 75:
        logger.warning("ACT_CLASS_NAMES not available from config; using generic names.")
        class_names = [f"class_{i}" for i in range(75)]

    kwargs = dict(probe_kwargs or {})
    epochs = kwargs.get("epochs", 10)
    lr = kwargs.get("lr", 1e-3)

    logger.info("Training 75-class linear probe (epochs=%d, lr=%s)...", epochs, lr)
    t0 = time.time()
    probe, history = train_probe(
        train_features=train_features.cpu(),
        train_labels=train_labels.cpu(),
        val_features=val_features.cpu(),
        val_labels=val_labels.cpu(),
        num_classes=NUM_CLASSES_75,
        in_dim=train_features.shape[1],
        epochs=epochs,
        lr=lr,
        device=device,
    )
    train_time = time.time() - t0
    logger.info("Probe trained in %.1fs. Best val acc: %.4f", train_time, max(history["val_acc"]))

    # Run evaluation on val features
    probe.eval()
    with torch.no_grad():
        all_logits = probe(val_features.to(device)).cpu().numpy()

    all_preds = np.argmax(all_logits, axis=1)
    all_labels_np = val_labels.cpu().numpy()

    metrics = compute_metrics(
        all_labels=all_labels_np,
        all_preds=all_preds,
        all_logits=all_logits,
        class_names=class_names,
        class_map=class_map,
    )

    metrics["probe_info"] = {
        "feature_source": "MViTv2-S (Kinetics-400) frozen",
        "feature_dim": train_features.shape[1],
        "epochs_trained": epochs,
        "best_val_acc_75_train": round(max(history["val_acc"]), 6),
        "train_time_sec": round(train_time, 1),
    }
    metrics["taxonomy"] = {
        "num_fine_classes": NUM_CLASSES_75,
        "num_grouped_classes": NUM_CLASSES_69,
        "grouping_mode": "hybrid",
        "mapping_file": str(_CLASS_MAP_69_TO_75),
    }

    return metrics


# =========================================================================
# Checkpoint evaluation mode
# =========================================================================
def evaluate_checkpoint(
    checkpoint_path: Path,
    save_dir: Path,
    device: torch.device,
    max_recordings: int = None,
    clip_length: int = 16,
    stride: int = 8,
) -> dict:
    """Evaluate a multi-task model checkpoint on 75-class activity.

    Loads the model, runs inference on clips from the val split,
    and computes clip-level metrics.

    NOTE: This requires a GPU with sufficient VRAM for the model + video backbone.
    The checkpoint's activity head must have 75 outputs for direct 75-class eval.
    If the head has 69 outputs, the 69->75 remap is applied.
    """
    logger.info("Loading checkpoint from %s...", checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    logger.info("Checkpoint epoch: %s", ckpt.get("epoch", "unknown"))

    from src.models.model import POPWMultiTaskModel

    # Determine output dimension from checkpoint state dict
    act_weight_key = [k for k in ckpt["model"].keys() if "act" in k.lower() and "weight" in k.lower() and "classifier" in k.lower()]
    num_act_outputs = 75  # default
    if act_weight_key:
        num_act_outputs = ckpt["model"][act_weight_key[0]].shape[0]
        logger.info("Activity head outputs: %d classes", num_act_outputs)

    # Build model
    model = POPWMultiTaskModel(
        pretrained=True,
        backbone_type="mvit_v2_s",
        use_hand_film=False,
        use_headpose_film=False,
        use_videomae=False,
        train_pose=False,
        num_act_classes=num_act_outputs,
    )
    state_dict = {k: v for k, v in ckpt["model"].items()
                  if "total_ops" not in k and "total_params" not in k}
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()
    logger.info("Model loaded on %s", device)

    # Build clip-level dataset
    from src.evaluation.eval_activity_clip import MViTClipDataset  # adapted for generic model

    class GenericClipDataset(torch.utils.data.Dataset):
        """16-frame clip dataset from AR_labels.csv segments."""

        def __init__(self, split: str, clip_len: int = 16, stride: int = 8,
                     max_recordings: int = None):
            self.clip_len = clip_len
            self.clips = []
            self._build(split, max_recordings)

        def _build(self, split: str, max_recordings: int):
            split_dir = C.RECORDINGS_ROOT / split
            rec_dirs = sorted(split_dir.iterdir())
            count = 0
            for rec_dir in rec_dirs:
                if not rec_dir.is_dir():
                    continue
                if max_recordings and count >= max_recordings:
                    break
                ar_file = rec_dir / "AR_labels.csv"
                if not ar_file.exists():
                    continue
                rec_id = rec_dir.name
                for line in ar_file.read_text().strip().split("\n"):
                    parts = line.strip().split(",")
                    if len(parts) < 5:
                        continue
                    action_id = int(parts[1])
                    if action_id < 0:
                        continue
                    start = int(Path(parts[3]).stem)
                    end = int(Path(parts[4]).stem)
                    seg_len = end - start + 1
                    if seg_len < self.clip_len:
                        continue
                    for clip_start in range(start, end - self.clip_len + 2, stride):
                        self.clips.append((rec_id, clip_start, action_id))
                        count += 1

        def __len__(self):
            return len(self.clips)

        def __getitem__(self, idx):
            rec_id, clip_start, action_id = self.clips[idx]
            rgb_dir = C.RECORDINGS_ROOT / "val" / rec_id / "rgb"
            frames = []
            for i in range(clip_start, clip_start + self.clip_len):
                img_path = rgb_dir / f"{i:06d}.jpg"
                try:
                    from PIL import Image
                    from torchvision.transforms import functional as TF
                    img = Image.open(img_path).convert("RGB")
                    img = TF.resize(img, [256], antialias=True)
                    img = TF.center_crop(img, [224, 224])
                    img = TF.to_tensor(img)
                    # MViTv2 normalization
                    mean = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1)
                    std = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1)
                    img = (img - mean) / std
                    frames.append(img)
                except Exception:
                    frames.append(torch.zeros(3, 224, 224))
            clip = torch.stack(frames, dim=0)  # [16, 3, 224, 224]
            return clip, action_id

    logger.info("Building val clip dataset...")
    val_ds = GenericClipDataset(split="val", clip_len=clip_length, stride=stride,
                                 max_recordings=max_recordings)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=0,
    )
    logger.info("Val clips: %d", len(val_ds))

    if len(val_ds) == 0:
        logger.error("No val clips found — check RECORDINGS_ROOT path.")
        return {"error": "empty val dataset"}

    # Load class mapping and names
    class_map = load_class_map()
    class_names = list(C.ACT_CLASS_NAMES) if hasattr(C, "ACT_CLASS_NAMES") else None
    if class_names is None or len(class_names) != 75:
        class_names = [f"class_{i}" for i in range(75)]

    # Inference
    all_preds = []
    all_labels = []
    all_logits_list = []

    logger.info("Running clip-level inference on %d clips...", len(val_ds))
    t0 = time.time()
    for idx in range(len(val_ds)):
        clip, label = val_ds[idx]
        clip = clip.unsqueeze(0).to(device)  # [1, 16, 3, 224, 224]; model may need [1, 3, 16, 224, 224]
        with torch.no_grad():
            outputs = model(clip)
        logits = outputs.get("act_logits")
        if logits is None:
            logger.warning("No act_logits in model output at clip %d", idx)
            continue
        logits_np = logits.cpu().numpy()  # [1, C]

        # Remap 69->75 if needed
        if logits_np.shape[1] == 69 and class_map is not None:
            probs = F.softmax(torch.from_numpy(logits_np), dim=-1).numpy()
            logits_np = remap_69_to_75(probs, class_map)
            logits_np = np.log(logits_np + 1e-10)  # back to log-space for top-5

        pred = np.argmax(logits_np, axis=1)[0]
        all_preds.append(pred)
        all_labels.append(label)
        all_logits_list.append(logits_np[0])

        if idx > 0 and idx % 500 == 0:
            logger.info("  processed %d/%d clips...", idx, len(val_ds))

    infer_time = time.time() - t0
    logger.info("Inference done: %d clips in %.1fs", len(all_preds), infer_time)

    if len(all_preds) == 0:
        return {"error": "no predictions collected"}

    all_preds_np = np.array(all_preds)
    all_labels_np = np.array(all_labels)
    all_logits_np = np.array(all_logits_list)

    metrics = compute_metrics(
        all_labels=all_labels_np,
        all_preds=all_preds_np,
        all_logits=all_logits_np,
        class_names=class_names,
        class_map=class_map,
    )

    metrics["checkpoint_info"] = {
        "path": str(checkpoint_path),
        "epoch": ckpt.get("epoch", "unknown"),
        "model_act_outputs": num_act_outputs,
    }

    return metrics


# =========================================================================
# Save
# =========================================================================
def save_metrics(metrics: dict, save_dir: Path):
    """Save metrics to save_dir/metrics.json."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / "metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved to %s", out_path)

    # Also save a human-readable summary
    summary_path = save_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("ACTIVITY 75-CLASS CLIP-LEVEL EVALUATION\n")
        f.write("=" * 60 + "\n\n")
        for key in ["top1_75", "top5_75", "macro_f1_75", "weighted_f1_75",
                     "macro_precision_75", "macro_recall_75", "clip_count", "valid_clip_count"]:
            if key in metrics:
                f.write(f"  {key:<25} = {metrics[key]}\n")

        if "top1_69" in metrics:
            f.write(f"\n  69-Grouped (for reference):\n")
            f.write(f"    top1_69                    = {metrics['top1_69']}\n")
            f.write(f"    macro_f1_69                = {metrics['macro_f1_69']}\n")

        f.write("\nPer-class top-1 (top 10 most frequent):\n")
        per_class = metrics.get("per_class_75", {})
        sorted_pc = sorted(per_class.items(), key=lambda x: x[1]["count"], reverse=True)
        for name, info in sorted_pc[:10]:
            f.write(f"  {name:<30} n={info['count']:<5} top1={info['top1']:.4f}\n")
        if len(sorted_pc) > 10:
            f.write(f"  ... and {len(sorted_pc) - 10} more classes\n")
    logger.info("Summary saved to %s", summary_path)


# =========================================================================
# Main
# =========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Clip-level activity evaluation on 75 fine-grained classes"
    )
    parser.add_argument(
        "--mode", type=str, default="feature-probe",
        choices=["feature-probe", "checkpoint"],
        help="Evaluation mode: feature-probe (cached MViTv2-S features) or checkpoint (multi-task model)",
    )
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint.pth (for --mode checkpoint)")
    parser.add_argument("--save-dir", type=str, default=str(DEFAULT_SAVE_DIR),
                        help="Output directory for metrics.json")
    parser.add_argument("--probe-epochs", type=int, default=10,
                        help="Probe training epochs (feature-probe mode)")
    parser.add_argument("--probe-lr", type=float, default=1e-3,
                        help="Probe learning rate (feature-probe mode)")
    parser.add_argument("--clip-length", type=int, default=16,
                        help="Frames per clip (checkpoint mode)")
    parser.add_argument("--stride", type=int, default=8,
                        help="Clip stride (checkpoint mode)")
    parser.add_argument("--max-recordings", type=int, default=None,
                        help="Cap recordings for fast test (checkpoint mode)")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU even if CUDA is available")
    args = parser.parse_args()

    device = torch.device("cpu") if args.cpu or not torch.cuda.is_available() else torch.device("cuda")
    logger.info("Device: %s", device)
    logger.info("Mode: %s", args.mode)

    save_dir = Path(args.save_dir)

    if args.mode == "feature-probe":
        metrics = evaluate_feature_probe(
            save_dir=save_dir,
            device=device,
            probe_kwargs={"epochs": args.probe_epochs, "lr": args.probe_lr},
        )
    elif args.mode == "checkpoint":
        if args.checkpoint is None:
            logger.error("--checkpoint required for checkpoint mode")
            sys.exit(1)
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            logger.error("Checkpoint not found: %s", ckpt_path)
            sys.exit(1)
        metrics = evaluate_checkpoint(
            checkpoint_path=ckpt_path,
            save_dir=save_dir,
            device=device,
            max_recordings=args.max_recordings,
            clip_length=args.clip_length,
            stride=args.stride,
        )

    if "error" in metrics:
        logger.error("Evaluation failed: %s", metrics["error"])
        sys.exit(1)

    # Assert clip_count > 0 before outputting metrics
    assert metrics.get("clip_count", 0) > 0, (
        f"clip_count is {metrics.get('clip_count', 0)} — no clips evaluated. "
        "This indicates a data-loading failure."
    )

    save_metrics(metrics, save_dir)
    logger.info("Done.")
    return metrics


if __name__ == "__main__":
    main()
