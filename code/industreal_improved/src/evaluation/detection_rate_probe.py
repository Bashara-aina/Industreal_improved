"""
Quick detection rate probe (134-debate NQ-3):
  - Forward 200 frames
  - Count detections at conf ∈ {0.01, 0.05, 0.25}
  - Output: detections_per_frame at each threshold, per-class detection counts

Usage: CUDA_VISIBLE_DEVICES=1 python3 src/evaluation/detection_rate_probe.py
"""

from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("detection_rate_probe")

_SRC = Path(__file__).resolve().parent.parent
for _sub in ["models", "training", "evaluation", "data", str(_SRC)]:
    _p = _SRC / _sub if _sub != str(_SRC) else _SRC
    _p = str(_p)
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(_SRC.parent) not in sys.path:
    sys.path.insert(0, str(_SRC.parent))

import src.config as C
from src.data.industreal_dataset import IndustRealMultiTaskDataset, collate_fn
from src.models.model import POPWMultiTaskModel

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

THRESHOLDS = [0.01, 0.05, 0.25]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    base_ckpt_dir = (
        Path(getattr(C, "RUNS", _SRC.parent / "src" / "runs")) / "rf_stages" / "checkpoints"
    )
    ckpt_path = base_ckpt_dir / "best.pth"
    logger.info("Loading checkpoint: %s", ckpt_path)

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = state.get("config", {})

    model = (
        POPWMultiTaskModel(
            pretrained=True,
            backbone_type=cfg.get("BACKBONE_TYPE", "convnext_tiny"),
            use_hand_film=bool(cfg.get("USE_HAND_FILM", True)),
            use_headpose_film=bool(cfg.get("USE_HEADPOSE_FILM", False)),
            use_videomae=bool(cfg.get("USE_VIDEOMAE", False)),
            train_pose=bool(cfg.get("TRAIN_HEAD_POSE", True)),
            use_backbone_checkpoint=bool(cfg.get("USE_BACKBONE_CHECKPOINT", False)),
        )
        .to(device)
        .eval()
    )

    result = model.load_state_dict(state["model"], strict=False)
    if result.missing_keys:
        logger.warning("Missing keys: %d", len(result.missing_keys))
    if result.unexpected_keys:
        logger.warning("Unexpected keys (harmless fluff): %d keys", len(result.unexpected_keys))

    val_ds = IndustRealMultiTaskDataset(split="val", img_size=(C.IMG_HEIGHT, C.IMG_WIDTH))
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_fn,
    )

    # Per-threshold accumulators
    det_counts = {t: defaultdict(int) for t in THRESHOLDS}  # {thr: {class_id: count}}
    det_frames = {t: 0 for t in THRESHOLDS}  # frames with any detection
    total_frames = 0
    total_gt_boxes = 0
    gt_class_counts = defaultdict(int)

    cached_anchors_np = None

    for bi, (images, targets) in enumerate(val_loader):
        if bi >= 200:
            break

        images = images.to(device, non_blocking=True).float()
        if images.max() > 1.0:
            images = images.div_(255.0)
        mean = torch.tensor(_IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD, device=device).view(1, 3, 1, 1)
        images = (images - mean) / std

        with torch.no_grad():
            outputs = model(images)

        if cached_anchors_np is None:
            cached_anchors_np = outputs["anchors"].cpu().numpy()

        cls_sigmoid = torch.sigmoid(outputs["cls_preds"])  # [1, N, 24]

        # GT
        det_targets = targets["detection"]
        for i in range(len(det_targets)):
            gt_labels = det_targets[i]["labels"]
            total_gt_boxes += len(gt_labels)
            for lbl in gt_labels:
                gt_class_counts[int(lbl)] += 1

        # Detection counts at each threshold
        scores_i = cls_sigmoid[0]  # [N, 24]
        max_scores = scores_i.max(dim=1).values  # [N]
        reg_preds = outputs["reg_preds"][0]
        anchors = cached_anchors_np

        for thr in THRESHOLDS:
            keep = max_scores > thr
            if keep.sum().item() == 0:
                det_frames[thr] += 0
            else:
                det_frames[thr] += 1
                kept_idx = keep.nonzero(as_tuple=True)[0].cpu().numpy()
                kept_cls = scores_i[keep].cpu().numpy()
                kept_labels = kept_cls.argmax(axis=1)
                for lbl in kept_labels:
                    det_counts[thr][int(lbl)] += 1

        total_frames += 1
        if bi % 50 == 0 and bi > 0:
            logger.info("  processed %d frames...", bi)

    # Summary
    print("\n" + "=" * 60)
    print("DETECTION RATE PROBE (200 frames)")
    print("=" * 60)
    print(f"Total frames: {total_frames}")
    print(f"Total GT boxes: {total_gt_boxes}")
    print(f"GT classes present: {len(gt_class_counts)}")
    print()

    for thr in THRESHOLDS:
        n_det = sum(det_counts[thr].values())
        det_per_frame = n_det / max(total_frames, 1)
        frames_with_det = det_frames[thr]
        print(f"Threshold {thr:.2f}:")
        print(f"  Total detections: {n_det}")
        print(f"  Detections/frame: {det_per_frame:.2f}")
        print(
            f"  Frames with any detection: {frames_with_det}/{total_frames} ({100 * frames_with_det / max(total_frames, 1):.1f}%)"
        )
        print()

    print("Per-class GT counts (top 20):")
    sorted_gt = sorted(gt_class_counts.items(), key=lambda x: -x[1])
    for cid, cnt in sorted_gt[:20]:
        cname = (
            C.INDUSTREAL_CLASS_NAMES.get(str(cid), cid)
            if hasattr(C, "INDUSTREAL_CLASS_NAMES")
            else cid
        )
        print(f"  Class {cid} ({cname}): {cnt}")

    print()
    for thr in THRESHOLDS:
        print(f"\nPer-class detection counts @ thr={thr:.2f} (top 15):")
        sorted_det = sorted(det_counts[thr].items(), key=lambda x: -x[1])
        for cid, cnt in sorted_det[:15]:
            cname = (
                C.INDUSTREAL_CLASS_NAMES.get(str(cid), cid)
                if hasattr(C, "INDUSTREAL_CLASS_NAMES")
                else cid
            )
            print(f"  Class {cid} ({cname}): {cnt}")

    # Save
    out = {
        "n_frames": total_frames,
        "n_gt_boxes": total_gt_boxes,
        "gt_class_counts": {str(k): int(v) for k, v in gt_class_counts.items()},
        "per_threshold": {},
    }
    for thr in THRESHOLDS:
        out["per_threshold"][str(thr)] = {
            "total_detections": int(sum(det_counts[thr].values())),
            "detections_per_frame": float(sum(det_counts[thr].values()) / max(total_frames, 1)),
            "frames_with_detections": int(det_frames[thr]),
            "class_counts": {str(k): int(v) for k, v in det_counts[thr].items()},
        }

    out_path = Path("src/runs/rf_stages/checkpoints/detection_rate_probe.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info("Saved to %s", out_path)


if __name__ == "__main__":
    main()
