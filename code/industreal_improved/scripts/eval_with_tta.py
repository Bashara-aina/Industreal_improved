"""Standalone evaluation script with horizontal-flip TTA for detection.

Loads a saved checkpoint and runs detection mAP eval with TTA.
Does not require training to be stopped — reads from best.pt.
"""

# DEPRECATED: This script uses the legacy MTLMViTModel. Use POPWMultiTaskModel from src/models/model.py instead.
import argparse
import sys
from pathlib import Path

_CODE_ROOT = Path(__file__).resolve().parent.parent
for _p in [str(_CODE_ROOT), str(_CODE_ROOT / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval_tta")

import torch
import numpy as np
from torch.utils.data import DataLoader

import src.config as C
from src.models.mvit_mtl_model import MTLMViTModel
from src.data.industreal_dataset import IndustRealMultiTaskDataset, collate_fn_sequences
from src.evaluation.det_tta import decode_det_tta
from src.evaluation.evaluate import compute_det_metrics_extended


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=0, help="0 = full")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--score-thresh", type=float, default=0.05)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument(
        "--sequence-length", type=int, default=32, help="T frames per sequence (matching training)"
    )
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Building model...")
    model = MTLMViTModel(num_act_classes=75).to(device)

    logger.info(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    model_sd = model.state_dict()
    filtered = {k: v for k, v in sd.items() if k in model_sd and v.shape == model_sd[k].shape}
    skipped = sum(1 for k in sd if k not in filtered)
    logger.info(f"Loaded {len(filtered)}/{len(sd)} tensors, skipped {skipped}")
    model.load_state_dict(filtered, strict=False)
    model.eval()

    logger.info(f"Building {args.split} dataset (img_size={args.img_size}, sequence_mode=True)...")
    ds = IndustRealMultiTaskDataset(
        split=args.split,
        img_size=(args.img_size, args.img_size),
        sequence_mode=True,
        augment=False,
        sequence_length=args.sequence_length,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn_sequences,
    )

    pred_boxes_all = []
    pred_scores_all = []
    pred_labels_all = []
    gt_boxes_all = []
    gt_labels_all = []
    n_batches = 0

    logger.info(f"Running TTA eval (h-flip) on {len(ds)} samples...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if args.max_batches > 0 and batch_idx >= args.max_batches:
                break
            images, targets = batch
            # MViT expects [B, 3, T=16, H, W]
            # Dataset returns [B, T=16, 3, H, W] -> permute to [B, 3, T, H, W]
            if images.dim() == 5:
                images = images.permute(0, 2, 1, 3, 4).contiguous()
            else:
                raise ValueError(f"Expected 5D sequence tensor, got {images.dim()}D")
            assert images.size(2) >= 1, f"Expected T >= 1, got T={images.size(2)}"
            images = images.float() / 255.0
            mean = torch.tensor([0.45, 0.45, 0.45], device=device).view(1, 3, 1, 1, 1)
            std = torch.tensor([0.225, 0.225, 0.225], device=device).view(1, 3, 1, 1, 1)
            images = (images.to(device) - mean) / std

            tta_results = decode_det_tta(
                model,
                images,
                img_size=args.img_size,
                score_thresh=args.score_thresh,
                nms_iou_thresh=args.nms_iou,
                device=device,
            )

            for b, tta_res in enumerate(tta_results):
                pred_boxes_all.append(tta_res["boxes"])
                pred_scores_all.append(tta_res["scores"])
                pred_labels_all.append(tta_res["labels"])
                det_item = targets.get("detection", [])
                if isinstance(det_item, list):
                    det_item = det_item[b] if b < len(det_item) else {}
                if isinstance(det_item, dict):
                    gt_b = det_item.get("boxes", torch.zeros(0, 4))
                    gt_l = det_item.get("labels", torch.zeros(0, dtype=torch.long))
                    gt_boxes_all.append(gt_b.cpu().numpy() if hasattr(gt_b, "cpu") else gt_b)
                    gt_labels_all.append(gt_l.cpu().numpy() if hasattr(gt_l, "cpu") else gt_l)
                else:
                    gt_boxes_all.append(np.zeros((0, 4), dtype=np.float32))
                    gt_labels_all.append(np.zeros(0, dtype=np.int64))

            n_batches += 1
            if n_batches % 100 == 0:
                logger.info(f"  Eval batch {n_batches} ({len(pred_boxes_all)} samples)")

    logger.info(f"Total samples: {len(pred_boxes_all)}")
    logger.info("Computing mAP with COCO-style interpolation...")
    result = compute_det_metrics_extended(
        pred_boxes_all,
        pred_scores_all,
        pred_labels_all,
        gt_boxes_all,
        gt_labels_all,
        num_classes=C.NUM_DET_CLASSES,
        interpolation_mode="coco",
    )
    logger.info("=" * 60)
    logger.info(f"WITH HORIZONTAL-FLIP TTA")
    logger.info(f"  mAP@0.5:               {result['det_mAP50']:.4f}")
    logger.info(f"  mAP@0.5:0.95:          {result['det_mAP_50_95']:.4f}")
    logger.info(f"  mAP@0.5 (PC):          {result['det_mAP50_pc']:.4f}")
    logger.info(f"  Classes w/ GT in eval: {result['det_n_present_classes']}/{C.NUM_DET_CLASSES}")
    logger.info("=" * 60)
    # Per-class top performers
    per_class = result.get("det_per_class", [])
    sorted_pc = sorted(
        [(c["ap"], c["gt"], c["name"]) for c in per_class if c["gt"] > 0], reverse=True
    )
    logger.info("Top 5 classes by AP@0.5:")
    for ap, gt, name in sorted_pc[:5]:
        logger.info(f"  {name}: AP={ap:.4f} ({int(gt)} GT)")

    if args.output:
        import json

        out = {
            "checkpoint": args.checkpoint,
            "split": args.split,
            "n_samples": len(pred_boxes_all),
            "det_mAP50": result["det_mAP50"],
            "det_mAP_50_95": result["det_mAP_50_95"],
            "det_mAP50_pc": result["det_mAP50_pc"],
            "det_n_present_classes": result["det_n_present_classes"],
            "per_class": [
                {"channel": c["channel"], "name": c["name"], "ap": c["ap"], "gt": c["gt"]}
                for c in per_class
                if c["gt"] > 0
            ],
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2, default=str)
        logger.info(f"Saved metrics to {args.output}")


if __name__ == "__main__":
    main()
