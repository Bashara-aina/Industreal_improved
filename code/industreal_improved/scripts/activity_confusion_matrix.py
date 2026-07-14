#!/usr/bin/env python3
"""
activity_confusion_matrix.py — 75x75 confusion matrix analysis on the test set.

Loads a multi-task checkpoint, runs clip-level inference on the test split,
computes the full 75x75 confusion matrix, identifies the top-10 most confused
off-diagonal class pairs, and outputs:
  - confusion_matrix.json   (full matrix + top pairs + per-class stats)
  - confusion_matrix.png    (seaborn heatmap)

Usage:
    python scripts/activity_confusion_matrix.py \\
        --checkpoint src/runs/rf_stages/checkpoints/best.pth \\
        --save-dir /tmp/confusion_analysis
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src import config as C
from src.models.model import POPWMultiTaskModel

logger = logging.getLogger("activity_confusion_matrix")

NUM_CLASSES = 75

# MViTv2 normalization constants
MEAN = torch.tensor([0.45, 0.45, 0.45])
STD = torch.tensor([0.225, 0.225, 0.225])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class TestClipDataset(torch.utils.data.Dataset):
    """16-frame clip dataset built from AR_labels.csv on the test split."""

    def __init__(self, split: str = "test", clip_len: int = 16, stride: int = 8,
                 max_recordings: int = None, max_clips: int = None):
        self.clip_len = clip_len
        self.clips = []  # (recording_id, clip_start_frame, action_label)
        self._build(split, max_recordings, max_clips)

    def _build(self, split: str, max_recordings: int, max_clips: int):
        split_dir = C.RECORDINGS_ROOT / split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"Split directory not found: {split_dir}\n"
                f"Check RECORDINGS_ROOT in src/config.py"
            )
        rec_dirs = sorted(split_dir.iterdir())
        rec_count = 0
        clip_count = 0
        for rec_dir in rec_dirs:
            if not rec_dir.is_dir():
                continue
            if max_recordings and rec_count >= max_recordings:
                break
            ar_file = rec_dir / "AR_labels.csv"
            if not ar_file.exists():
                continue
            rec_id = rec_dir.name
            lines = ar_file.read_text().strip().split("\n")
            for line in lines:
                if max_clips and clip_count >= max_clips:
                    break
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
                    if max_clips and clip_count >= max_clips:
                        break
                    self.clips.append((rec_id, clip_start, action_id))
                    clip_count += 1
            rec_count += 1

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        rec_id, clip_start, action_id = self.clips[idx]
        rgb_dir = C.RECORDINGS_ROOT / "test" / rec_id / "rgb"
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
                img = (img - MEAN.view(3, 1, 1)) / STD.view(3, 1, 1)
                frames.append(img)
            except Exception:
                frames.append(torch.zeros(3, 224, 224))
        clip = torch.stack(frames, dim=0)  # [16, 3, 224, 224]
        return clip, int(action_id)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_inference(model, dataset, device, batch_size: int = 4):
    """Run clip-level inference.

    Returns (all_preds, all_labels, all_logits) as numpy arrays.
    """
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0,
    )
    all_preds, all_labels, all_logits = [], [], []

    for clip, label in loader:
        clip = clip.to(device, non_blocking=True)   # [B, 16, 3, 224, 224]
        label = label.to(device, non_blocking=True)
        # Model expects [B, 3, 16, H, W] — rearrange
        clip = clip.permute(0, 2, 1, 3, 4).contiguous()
        outputs = model(clip)
        logits = outputs.get("act_logits")
        if logits is None:
            logger.warning("No act_logits in model output — skipping batch")
            continue
        logits_np = logits.cpu().numpy()
        preds = np.argmax(logits_np, axis=1)
        all_preds.append(preds)
        all_labels.append(label.cpu().numpy())
        all_logits.append(logits_np)

    if not all_preds:
        raise RuntimeError("No predictions collected — check model output keys")

    return (
        np.concatenate(all_preds),
        np.concatenate(all_labels),
        np.concatenate(all_logits),
    )


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------
def compute_confusion_matrix(all_labels: np.ndarray, all_preds: np.ndarray,
                             num_classes: int = NUM_CLASSES) -> np.ndarray:
    """Compute the full confusion matrix."""
    from sklearn.metrics import confusion_matrix
    valid = all_labels >= 0
    return confusion_matrix(
        all_labels[valid], all_preds[valid],
        labels=list(range(num_classes)),
    )


def find_top_confusions(cm: np.ndarray, class_names: list[str],
                        top_k: int = 10) -> list[dict]:
    """Find top-K most confused off-diagonal class pairs."""
    n = cm.shape[0]
    pairs = []
    for i in range(n):
        true_total = cm[i, :].sum()
        if true_total == 0:
            continue
        for j in range(n):
            if i == j:
                continue
            count = int(cm[i, j])
            if count > 0:
                pairs.append({
                    "true_class_id": i,
                    "true_class_name": class_names[i] if i < len(class_names) else str(i),
                    "pred_class_id": j,
                    "pred_class_name": class_names[j] if j < len(class_names) else str(j),
                    "count": count,
                    "pct_of_true": round(100.0 * count / true_total, 1),
                })
    pairs.sort(key=lambda x: x["count"], reverse=True)
    return pairs[:top_k]


def compute_per_class_stats(cm: np.ndarray, class_names: list[str]) -> dict:
    """Compute per-class precision, recall, F1, and support."""
    n = cm.shape[0]
    stats = {}
    for i in range(n):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum()) - tp
        fn = int(cm[i, :].sum()) - tp
        support = int(cm[i, :].sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-10)
        name = class_names[i] if i < len(class_names) else str(i)
        stats[name] = {
            "class_id": i,
            "support": support,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
        }
    return stats


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def plot_confusion_matrix(cm: np.ndarray, class_names: list[str],
                          save_path: Path, figsize: tuple = (48, 40)):
    """Render a full 75x75 confusion matrix heatmap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Normalize by row (true class) for display
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sums > 0, cm.astype(np.float32) / row_sums, 0)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm_norm, annot=False, fmt=".2f", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, cbar_kws={"shrink": 0.8},
    )
    ax.set_xlabel("Predicted Class", fontsize=24)
    ax.set_ylabel("True Class", fontsize=24)
    ax.set_title("75-Class Activity Confusion Matrix (row-normalized)", fontsize=28)
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    plt.tight_layout()
    fig.savefig(str(save_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Heatmap saved to %s", save_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="75x75 confusion matrix on test set"
    )
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint .pth. If omitted, searches default locations.")
    parser.add_argument("--save-dir", type=str, default="/tmp/confusion_analysis",
                        help="Output directory for JSON + PNG")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split (test, val)")
    parser.add_argument("--clip-length", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-recordings", type=int, default=None,
                        help="Cap number of recordings for fast debugging")
    parser.add_argument("--max-clips", type=int, default=None,
                        help="Cap total clips for fast debugging")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    logger.info("Device: %s", device)
    logger.info("Args: %s", vars(args))

    # --- Resolve checkpoint path ---
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        candidates = [
            _PROJECT_ROOT / "src" / "runs" / "rf_stages" / "checkpoints" / "best.pth",
            _PROJECT_ROOT / "src" / "runs" / "rf_stages" / "checkpoints" / "latest.pth",
        ]
        for c in candidates:
            if c.exists():
                ckpt_path = str(c)
                logger.info("Auto-selected checkpoint: %s", ckpt_path)
                break
        if ckpt_path is None:
            logger.error("No checkpoint found. Use --checkpoint to specify a path.")
            sys.exit(1)
    else:
        ckpt_path = str(Path(ckpt_path).resolve())
        if not Path(ckpt_path).exists():
            logger.error("Checkpoint not found: %s", ckpt_path)
            sys.exit(1)

    # --- Load checkpoint ---
    logger.info("Loading checkpoint from %s ...", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    logger.info("Checkpoint epoch: %s", ckpt.get("epoch", "unknown"))

    # Determine activity head output dimension
    act_weight_key = [
        k for k in ckpt["model"].keys()
        if "act" in k.lower() and "weight" in k.lower() and "classifier" in k.lower()
    ]
    num_act_outputs = NUM_CLASSES
    if act_weight_key:
        num_act_outputs = ckpt["model"][act_weight_key[0]].shape[0]
        logger.info("Activity head outputs: %d classes", num_act_outputs)

    # --- Build model ---
    logger.info("Building POPWMultiTaskModel ...")
    model = POPWMultiTaskModel(
        pretrained=True,
        backbone_type="mvit_v2_s",
        use_hand_film=False,
        use_headpose_film=False,
        use_videomae=False,
        train_pose=False,
        num_act_classes=num_act_outputs,
    )
    state_dict = {
        k: v for k, v in ckpt["model"].items()
        if "total_ops" not in k and "total_params" not in k
    }
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()
    logger.info("Model loaded on %s", device)

    # --- Build dataset ---
    logger.info("Building %s clip dataset ...", args.split)
    dataset = TestClipDataset(
        split=args.split,
        clip_len=args.clip_length,
        stride=args.stride,
        max_recordings=args.max_recordings,
        max_clips=args.max_clips,
    )
    logger.info("Total clips: %d", len(dataset))

    if len(dataset) == 0:
        logger.error("No clips found. Check RECORDINGS_ROOT and split name.")
        sys.exit(1)

    # --- Get class names ---
    class_names = list(C.ACT_CLASS_NAMES) if hasattr(C, "ACT_CLASS_NAMES") else []
    if len(class_names) != NUM_CLASSES:
        logger.warning(
            "ACT_CLASS_NAMES has %d entries (expected %d); using generic names.",
            len(class_names), NUM_CLASSES,
        )
        class_names = [f"class_{i}" for i in range(NUM_CLASSES)]

    # --- Inference ---
    logger.info("Running inference ...")
    t0 = time.time()
    all_preds, all_labels, all_logits = run_inference(
        model, dataset, device, batch_size=args.batch_size,
    )
    infer_time = time.time() - t0
    logger.info("Inference done: %d clips in %.1fs", len(all_preds), infer_time)

    # --- Remap if head has 69 outputs ---
    if num_act_outputs == 69:
        logger.info("Remapping 69-group predictions to 75-class space ...")
        probs = F.softmax(torch.from_numpy(all_logits), dim=-1).numpy()
        # Load class mapping
        mapping_path = (
            _PROJECT_ROOT / "config" / "class_maps" / "class_69_to_75.json"
        )
        if mapping_path.exists():
            import json as _json
            class_map = _json.loads(mapping_path.read_text())
            mapping = class_map["mapping"]
            n = probs.shape[0]
            remapped = np.zeros((n, NUM_CLASSES), dtype=np.float32)
            for gid_str, info in mapping.items():
                gid = int(gid_str)
                fine_ids = info["fine_class_ids"]
                prob = probs[:, gid]
                for fid in fine_ids:
                    remapped[:, fid] = prob / len(fine_ids)
            all_logits = remapped
            all_preds = np.argmax(all_logits, axis=1)
            num_act_outputs = NUM_CLASSES
            logger.info("Remapped to 75 classes.")
        else:
            logger.warning(
                "69-head checkpoint but no class map at %s; using 69-class output.",
                mapping_path,
            )

    # --- Compute confusion matrix ---
    logger.info("Computing 75x75 confusion matrix ...")
    cm = compute_confusion_matrix(all_labels, all_preds, num_classes=NUM_CLASSES)
    logger.info("Confusion matrix shape: %s", cm.shape)

    top_pairs = find_top_confusions(cm, class_names, top_k=10)
    per_class = compute_per_class_stats(cm, class_names)

    # Overall metrics
    valid = all_labels >= 0
    top1 = float((all_preds[valid] == all_labels[valid]).mean())
    logger.info("Overall top-1 accuracy: %.4f", top1)

    # --- Build output ---
    output = {
        "metadata": {
            "checkpoint": ckpt_path,
            "checkpoint_epoch": ckpt.get("epoch", "unknown"),
            "split": args.split,
            "num_clips": int(len(all_preds)),
            "num_valid": int(valid.sum()),
            "num_classes": int(cm.shape[0]),
            "top1_accuracy": round(top1, 6),
            "inference_time_sec": round(infer_time, 1),
            "model_act_outputs": int(num_act_outputs),
        },
        "confusion_matrix": cm.tolist(),
        "top_10_confused_pairs": top_pairs,
        "per_class_stats": per_class,
    }

    # --- Save JSON ---
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    json_path = save_dir / "confusion_matrix.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("JSON saved to %s", json_path)

    # --- Save PNG heatmap ---
    png_path = save_dir / "confusion_matrix.png"
    plot_confusion_matrix(cm, class_names, png_path)

    # --- Print top confusions ---
    logger.info("=" * 60)
    logger.info("Top-10 Most Confused Class Pairs:")
    logger.info("=" * 60)
    for rank, pair in enumerate(top_pairs, 1):
        logger.info(
            "  %2d. %-30s vs %-30s  count=%5d  (%5.1f%% of true)",
            rank,
            pair["true_class_name"],
            pair["pred_class_name"],
            pair["count"],
            pair["pct_of_true"],
        )

    logger.info("Done. Results in %s", save_dir)


if __name__ == "__main__":
    main()
