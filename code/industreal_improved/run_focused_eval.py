#!/usr/bin/env python3
"""Focused full eval: detection mAP + PSR transition F1 on full 38K set.

Usage:
    python run_focused_eval.py \
        --ckpt runs/aa_path_b/checkpoints/best.pth \
        --save-dir runs/aa_path_b/focused_eval \
        --device cuda:1
"""
import json, logging, os, sys, time
from pathlib import Path

import numpy as np
import torch

_SRC = Path(__file__).resolve().parent / "src"
for p in [_SRC, _SRC / "evaluation", _SRC / "models", _SRC / "training", _SRC / "data"]:
    sys.path.insert(0, str(p))

import src.config as C
from src.data.industreal_dataset import IndustRealMultiTaskDataset, collate_fn
from src.models.model import POPWMultiTaskModel
from evaluate import evaluate_all

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)
logger = logging.getLogger("focused_eval")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str,
                        default="runs/aa_path_b/checkpoints/best.pth")
    parser.add_argument("--save-dir", type=str, default="runs/aa_path_b/focused_eval")
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--max-batches", type=int, default=0,
                        help="0 = full 38K dataset")
    args = parser.parse_args()

    device = torch.device(args.device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Override config for full eval
    C.SKIP_DET_METRICS_EVAL = False
    C.DET_METRICS_EVERY_N = 1
    C.EVAL_MAX_BATCHES = args.max_batches if args.max_batches > 0 else 99999

    logger.info("Device: %s", device)
    logger.info("Save dir: %s", save_dir)
    logger.info("Checkpoint: %s", args.ckpt)

    # Load model
    model = POPWMultiTaskModel()
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)), strict=False)
    model.to(device)
    model.eval()
    logger.info("Model loaded (epoch %s)", ckpt.get("epoch", "?"))

    # Build val loader
    val_ds = IndustRealMultiTaskDataset(split="val", img_size=(C.IMG_HEIGHT, C.IMG_WIDTH))
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=8, shuffle=False,
        num_workers=2, pin_memory=True,
        collate_fn=collate_fn,
    )
    logger.info("Val dataset: %d samples", len(val_ds))

    # Run full eval
    max_batches = args.max_batches if args.max_batches > 0 else None
    logger.info("max_batches=%s", max_batches or "FULL DATASET")
    t0 = time.time()

    metrics = evaluate_all(
        model=model,
        criterion=None,
        loader=val_loader,
        device=device,
        max_batches=max_batches or 99999,
        save_dir=str(save_dir),
        epoch=-1,
    )

    elapsed = time.time() - t0
    logger.info("Eval complete: %.1f min", elapsed / 60)

    # Save metrics
    out_path = save_dir / "metrics.json"
    # Convert non-serializable
    clean = {}
    for k, v in metrics.items():
        if isinstance(v, (np.floating,)):
            clean[k] = float(v)
        elif isinstance(v, (np.integer,)):
            clean[k] = int(v)
        elif isinstance(v, np.ndarray):
            clean[k] = v.tolist()
        else:
            try:
                json.dumps(v)
                clean[k] = v
            except (TypeError, OverflowError):
                clean[k] = str(v)

    with open(out_path, "w") as f:
        json.dump(clean, f, indent=2, default=str)
    logger.info("Metrics saved to %s", out_path)

    # Print headline metrics
    headlines = [
        ("det_mAP50", "Detection AP@0.50"),
        ("det_mAP_50_95", "Detection AP@[.5:.95]"),
        ("det_mAP50_all_frames", "Detection AP@0.50 (all frames)"),
        ("det_mAP50_pc", "Detection AP@0.50 (present-class)"),
        ("act_macro_f1", "Activity macro F1"),
        ("act_top1", "Activity top-1 acc"),
        ("psr_f1", "PSR transition F1"),
        ("psr_pos", "PSR POS"),
        ("psr_edit", "PSR edit score"),
        ("forward_angular_MAE_deg", "Head pose MAE (fwd deg)"),
    ]
    logger.info("=" * 60)
    logger.info("HEADLINE METRICS")
    logger.info("=" * 60)
    for key, label in headlines:
        val = metrics.get(key, "N/A")
        logger.info("  %-35s = %s", label, val)

if __name__ == "__main__":
    main()
