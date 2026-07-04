"""
Per-Component PSR Tau Distribution — frame delay between predicted and GT state changes.

Computes the per-component distribution of detection delay (tau): for each GT state
transition (0->1 or 1->0) in a PSR component, finds the nearest predicted transition
and records the frame offset. Positive tau = prediction lags GT; negative = prediction
leads GT.

Output JSON schema:
{
  "per_component": {
    "0": {"mean": 3.2, "median": 2.8, "p25": 1.5, "p75": 4.0, "p95": 8.0, "n_transitions": 245},
    ...
  },
  "overall": {"mean": 3.5, "median": 3.0, "p95": 9.0}
}

Usage:
    python3 src/evaluation/eval_psr_tau_dist.py --ckpt path/to/checkpoint.pth
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger('eval_psr_tau_dist')

# Path setup matching evaluate.py pattern
_SRC = Path(__file__).resolve().parent  # src/evaluation/
for _sub in ['..', 'models', 'training', 'data']:
    _p = (_SRC / _sub).resolve()
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
if str(_SRC.parent.parent) not in sys.path:
    sys.path.insert(0, str(_SRC.parent.parent))


def _detect_state_changes(binary_seq: np.ndarray) -> np.ndarray:
    """Return frame indices where binary_seq changes (0->1 or 1->0)."""
    if binary_seq.ndim != 1 or binary_seq.size < 2:
        return np.array([], dtype=np.int64)
    diffs = np.diff(binary_seq.astype(np.int8))
    return np.flatnonzero(diffs != 0) + 1  # +1 because diff shifts left


def compute_tau_distribution(
    pred_logits: np.ndarray,
    gt_labels: np.ndarray,
    num_components: int = 11,
) -> dict:
    """
    Compute per-component tau (frame delay) distribution between predicted
    and GT state changes.

    Args:
        pred_logits: [N, num_components] sigmoid logits from PSR head
        gt_labels:   [N, num_components] binary labels (0/1, -1 = unknown)

    Returns:
        dict with per_component and overall tau statistics
    """
    pred_probs = 1.0 / (1.0 + np.exp(-pred_logits))
    pred_binary = (pred_probs > 0.5).astype(np.int8)

    # Mask unknown labels
    valid_mask = gt_labels != -1
    gt_safe = gt_labels.copy()
    gt_safe[~valid_mask] = 0

    per_component = {}
    all_taus = []

    for c in range(num_components):
        vm = valid_mask[:, c]
        gt_c = gt_safe[:, c]
        pred_c = pred_binary[:, c]

        # Get GT transitions on valid frames
        gt_valid_frames = np.flatnonzero(vm)
        if len(gt_valid_frames) < 2:
            per_component[str(c)] = {
                "mean": None, "median": None, "p25": None, "p75": None,
                "p95": None, "n_transitions": 0,
            }
            continue

        # Restrict to contiguous valid segments to avoid edge effects
        gt_changes = _detect_state_changes(gt_c)
        pred_changes = _detect_state_changes(pred_c)

        if len(gt_changes) == 0:
            per_component[str(c)] = {
                "mean": None, "median": None, "p25": None, "p75": None,
                "p95": None, "n_transitions": 0,
            }
            continue

        if len(pred_changes) == 0:
            per_component[str(c)] = {
                "mean": None, "median": None, "p25": None, "p75": None,
                "p95": None, "n_transitions": len(gt_changes),
            }
            all_taus.append(None)
            continue

        # For each GT transition, find nearest prediction transition
        taus = []
        for gtc in gt_changes:
            delays = np.abs(pred_changes - gtc)
            nearest_idx = np.argmin(delays)
            nearest_delay = int(pred_changes[nearest_idx] - gtc)
            taus.append(nearest_delay)

        taus_arr = np.array(taus, dtype=np.float64)
        all_taus.extend(taus)

        per_component[str(c)] = {
            "mean": float(np.mean(taus_arr)),
            "median": float(np.median(taus_arr)),
            "p25": float(np.percentile(taus_arr, 25)),
            "p75": float(np.percentile(taus_arr, 75)),
            "p95": float(np.percentile(taus_arr, 95)),
            "n_transitions": len(taus),
        }

    # Overall stats (aggregate over all components)
    flat_taus = np.array([t for t in all_taus if t is not None], dtype=np.float64)
    if len(flat_taus) > 0:
        overall = {
            "mean": float(np.mean(flat_taus)),
            "median": float(np.median(flat_taus)),
            "p95": float(np.percentile(flat_taus, 95)),
        }
    else:
        overall = {"mean": None, "median": None, "p95": None}

    return {"per_component": per_component, "overall": overall}


def extract_psr_predictions_from_eval(
    ckpt_path: str,
    max_batches: int = 0,
    device: str = "cuda",
) -> tuple:
    """
    Run evaluation to extract PSR predictions and labels.

    Returns:
        (all_psr_logits, all_psr_labels) numpy arrays of shape [N, 11]
    """
    from src import config as C
    from src.data.industreal_dataset import IndustRealMultiTaskDataset as IndustRealDataset
    from src.models.model import POPWMultiTaskModel

    logger.info(f"Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = state.get("config", {})

    num_classes_list = cfg.get(
        "NUM_CLASSES_LIST",
        [C.NUM_CLASSES_DET, C.NUM_CLASSES_ACT, C.NUM_CLASSES_PSR],
    )
    model = POPWMultiTaskModel(num_classes_list=num_classes_list)
    model.load_state_dict(state["model"], strict=False)
    model.to(device)
    model.eval()
    logger.info("Model loaded and in eval mode.")

    # Build validation dataset
    transform = C.get_transform('val')
    val_ds = IndustRealDataset(
        C.VAL_ANNOTATION_FILE,
        transform=transform,
        is_train=False,
        frame_stride=C.EVAL_FRAME_STRIDE,
        max_videos=C.DEBUG_MAX_VIDEOS if C.DEBUG_MODE else -1,
        multi_crop=C.USE_MULTI_CROP,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=C.VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=C.collate_fn,
        pin_memory=False,
    )
    logger.info(f"Val dataset: {len(val_ds)} samples")

    # Run inference collecting PSR predictions
    psr_preds_logits, psr_labels = [], []

    for bi, (images, targets) in enumerate(val_loader):
        if max_batches > 0 and bi >= max_batches:
            break
        images = images.to(device)
        with torch.no_grad():
            outputs = model(images)
        psr_preds_logits.append(outputs["psr_logits"].cpu().numpy())
        psr_labels.append(targets["psr_labels"].cpu().numpy())
        if bi % 10 == 0:
            logger.info(f"  [batch {bi}] images.shape={images.shape}")

    all_psr_logits = np.concatenate(psr_preds_logits)
    all_psr_labels = np.concatenate(psr_labels)
    logger.info(
        f"Collected PSR predictions: logits={all_psr_logits.shape}, "
        f"labels={all_psr_labels.shape}"
    )
    return all_psr_logits, all_psr_labels


def main():
    parser = argparse.ArgumentParser(
        description="Compute per-component PSR tau (delay) distribution."
    )
    parser.add_argument(
        "--ckpt", type=str, required=True,
        help="Path to checkpoint .pth file"
    )
    parser.add_argument(
        "--out_path", type=str, default=None,
        help="Output JSON path (default: auto-generated next to checkpoint)"
    )
    parser.add_argument(
        "--max_batches", type=int, default=0,
        help="Max batches to eval (0 = all)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device (cuda or cpu)"
    )
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        logger.error(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    logits, labels = extract_psr_predictions_from_eval(
        ckpt_path=str(ckpt_path),
        max_batches=args.max_batches,
        device=args.device,
    )

    result = compute_tau_distribution(logits, labels, num_components=11)

    # Determine output path
    if args.out_path:
        out_path = Path(args.out_path)
    else:
        out_dir = ckpt_path.parent / "eval_psr_tau_dist"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{ckpt_path.stem}_tau_dist.json"

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Saved tau distribution to {out_path}")

    # Summary
    pc = result["per_component"]
    valid = [k for k, v in pc.items() if v["n_transitions"] > 0]
    logger.info(f"Components with transitions: {len(valid)}/{len(pc)}")
    for k in valid:
        v = pc[k]
        logger.info(
            f"  comp {k}: n={v['n_transitions']} "
            f"mean={v['mean']:.1f} median={v['median']:.1f} "
            f"p25={v['p25']:.1f} p75={v['p75']:.1f} p95={v['p95']:.1f}"
        )
    ov = result["overall"]
    if ov["mean"] is not None:
        logger.info(
            f"  Overall: mean={ov['mean']:.1f} median={ov['median']:.1f} p95={ov['p95']:.1f}"
        )


if __name__ == "__main__":
    main()
