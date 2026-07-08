#!/usr/bin/env python3
"""
train_st_det.py — Single-Task Detection Baseline (175 §6 Row 1)

Trains a detection-only model for the experiment matrix.  Wraps the existing
train_singletask_detection.py (which patches config + delegates to train.py)
and then runs eval_detection_dual_protocol.py for both protocols.

Split discipline (§7.1):
  - val split (5 subjects) for model selection.
  - test split (10 subjects) via --test for final SOTA-comparable numbers.

Usage:
    # Model selection on val (default):
    python scripts/train_st_det.py [--max-epochs 100]

    # Final eval on test:
    python scripts/train_st_det.py --test [--resume <best.pth>]

    # Plumbing check (1 epoch, 5 recordings):
    python scripts/train_st_det.py --plumbing

Output:
    src/runs/rf_stages/checkpoints/st_det_run/
    ├── metrics.json           (dual-protocol results)
    ├── per_frame_predictions_{val,test}.json
    └── {best,latest}.pth      (training checkpoints)

Reference: AAIML 175 §6 (ST-Det row), §7.1 (split), §7.2 (detection protocol)
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

# Must set this BEFORE importing config so the train script sees it.
_ST_DET_RUN = Path(__file__).resolve().parent.parent / "src/runs/rf_stages/checkpoints/st_det_run"
os.environ["OUTPUT_ROOT_OVERRIDE"] = str(_ST_DET_RUN)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.split_config import require_split

logger = logging.getLogger("train_st_det")

_METRICS_PATH = _ST_DET_RUN / "metrics.json"


def _call_with_timeout(cmd: list, desc: str, timeout: int = 86400) -> int:
    """Run a subprocess command with streaming log output.

    Args:
        cmd: Command list (e.g. [sys.executable, "script.py", ...])
        desc: Human-readable description for logging
        timeout: Maximum wall-clock seconds (default 24h)

    Returns:
        Exit code from the subprocess.
    """
    logger.info("Starting: %s", desc)
    logger.info("  cmd: %s", " ".join(str(c) for c in cmd))
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            logger.info("  [%s] %s", desc, line)
    proc.wait(timeout=timeout)
    elapsed = time.time() - t0
    logger.info("%s finished (exit=%d, %.1fs)", desc, proc.returncode, elapsed)
    return proc.returncode


def _run_eval_dual_protocol(predictions_path: Path, output_path: Path, split_name: str) -> dict:
    """Run eval_detection_dual_protocol.py subprocess and return parsed metrics."""
    eval_script = Path(__file__).resolve().parent / "eval_detection_dual_protocol.py"
    if not eval_script.exists():
        logger.error("eval script not found: %s", eval_script)
        return None
    if not predictions_path.exists():
        logger.error("predictions not found: %s", predictions_path)
        return None

    rc = _call_with_timeout(
        [sys.executable, str(eval_script),
         "--predictions", str(predictions_path),
         "--out", str(output_path)],
        desc=f"dual-protocol eval ({split_name})",
        timeout=3600,
    )
    if rc != 0:
        return None
    with open(output_path) as f:
        return json.load(f)


def _generate_predictions(ckpt_path: Path, split: str, out_path: Path,
                          max_batches: int = 0) -> bool:
    """Run a detection-only eval to produce per_frame_predictions.json.

    Uses the existing evaluate_all from src.evaluation with the dual-protocol
    detection setup.
    """
    import torch
    from src.data.industreal_dataset import IndustRealMultiTaskDataset
    from src.evaluation.evaluate import evaluate_all
    from src.models.model import POPWMultiTaskModel
    from src.training.losses import MultiTaskLoss

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Generating predictions on %s split (device=%s)", split, device)

    # Patch config for detection-only
    import src.config as C
    C.TRAIN_DET = True
    C.TRAIN_HEAD_POSE = False
    C.TRAIN_ACT = False
    C.TRAIN_PSR = False

    ds = IndustRealMultiTaskDataset(
        split=split, img_size=(640, 640), augment=False, seed=42,
    )
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)

    model = POPWMultiTaskModel(
        pretrained=True, backbone_type="convnext_tiny",
        use_hand_film=False, use_headpose_film=False, use_videomae=False,
        train_pose=False, use_backbone_checkpoint=False,
    ).to(device)
    model.eval()

    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        model_state = state.get("model_state_dict",
                        state.get("model_state",
                        state.get("model", state)))
        model.load_state_dict(model_state, strict=False)
        logger.info("Loaded checkpoint: %s", ckpt_path)
    else:
        logger.warning("No checkpoint at %s — using untrained model", ckpt_path)

    criterion = MultiTaskLoss(
        num_classes_act=75, train_det=True,
        train_pose=False, train_act=False, train_psr=False, use_kendall=False,
    ).to(device)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    evaluate_all(model, criterion, loader, device,
                 max_batches=max_batches or 0, epoch=-1,
                 predictions_path=str(out_path))
    return out_path.exists()


def main():
    parser = argparse.ArgumentParser(
        description="ST-Det: single-task detection baseline (175 §6 row 1)",
    )
    parser.add_argument("--plumbing", action="store_true",
                        help="1 epoch, 5 recordings only")
    parser.add_argument("--test", action="store_true",
                        help="Eval on test split (default: val for model selection)")
    parser.add_argument("--max-epochs", type=int, default=None,
                        help="Override C.EPOCHS")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override C.BATCH_SIZE")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training, only evaluate checkpoint")
    parser.add_argument("--ckpt-path", type=str, default=None,
                        help="Checkpoint for --eval-only (default: best.pth)")
    parser.add_argument("--max-eval-batches", type=int, default=0,
                        help="Cap eval batches (0 = full)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S", force=True,
    )

    eval_split = "test" if args.test else "val"
    logger.info("ST-Det target split: %s", eval_split)

    # ── Output directory ───────────────────────────────────────────────────
    _ST_DET_RUN.mkdir(parents=True, exist_ok=True)

    # ── Split discipline ───────────────────────────────────────────────────
    if args.eval_only:
        require_split(eval_split, allow_test_only=(eval_split == "test"))
        logger.info("Split discipline: eval-only on %s", eval_split)
    else:
        require_split("val", allow_test_only=False)
        logger.info("Split discipline: training (12 train) -> val selection")

    # ── Training ───────────────────────────────────────────────────────────
    train_script = Path(__file__).resolve().parent / "train_singletask_detection.py"
    if not train_script.exists():
        logger.error("train_singletask_detection.py not found at %s", train_script)
        sys.exit(1)

    train_cmd = [sys.executable, str(train_script)]
    if args.max_epochs is not None:
        train_cmd.extend(["--max-epochs", str(args.max_epochs)])
    if args.batch_size is not None:
        train_cmd.extend(["--batch-size", str(args.batch_size)])
    if args.resume is not None:
        train_cmd.extend(["--resume", args.resume])
    if args.plumbing:
        train_cmd.append("--debug")

    if not args.eval_only:
        rc = _call_with_timeout(train_cmd, "detection-only training", timeout=86400)
        if rc != 0:
            logger.error("Training failed with exit code %d", rc)
            sys.exit(1)
    else:
        logger.info("Skipping training (--eval-only)")

    # ── Locate best checkpoint ─────────────────────────────────────────────
    ckpt_path = Path(args.ckpt_path) if args.ckpt_path else None
    if ckpt_path is None or not ckpt_path.exists():
        for candidate in ["best.pth", "latest.pth", "crash_recovery.pth"]:
            p = _ST_DET_RUN / candidate
            if p.exists():
                ckpt_path = p
                break
        if ckpt_path is None:
            # Fall back to default checkpoint dir
            import src.config as C
            for candidate in ["best.pth", "latest.pth"]:
                p = Path(C.CHECKPOINT_DIR) / candidate
                if p.exists():
                    ckpt_path = p
                    break
    if ckpt_path is None or not ckpt_path.exists():
        logger.error("No checkpoint found; cannot run evaluation")
        sys.exit(1)
    logger.info("Using checkpoint: %s", ckpt_path)

    # ── Generate predictions & dual-protocol eval ──────────────────────────
    pred_path = _ST_DET_RUN / f"per_frame_predictions_{eval_split}.json"

    success = _generate_predictions(
        ckpt_path, eval_split, pred_path,
        max_batches=args.max_eval_batches,
    )
    if not success:
        logger.error("Failed to generate predictions")
        sys.exit(1)

    metrics = _run_eval_dual_protocol(pred_path, _METRICS_PATH, eval_split)
    if metrics is None:
        logger.error("Dual-protocol evaluation failed")
        sys.exit(1)

    # ── Summary ────────────────────────────────────────────────────────────
    af = metrics.get("annotated_frames", {})
    ev = metrics.get("entire_video", {})
    sota = metrics.get("sota_anchor", {})
    logger.info("=" * 60)
    logger.info("ST-Det FINAL (%s split)  |  Annotated-frames mAP: %.4f  "
                "Entire-video mAP: %.4f",
                eval_split, af.get("det_mAP50", -1), ev.get("det_mAP50_all_frames", -1))
    logger.info("WACV targets:           |  (0.838)               (0.641)")
    logger.info("Saved: %s", _METRICS_PATH)

    print()
    print(f"ST-Det results ({eval_split} split, dual-protocol mAP@0.5)")
    print(f"  ({af.get('det_mAP50', -1):.4f}, {ev.get('det_mAP50_all_frames', -1):.4f})")
    print(f"  WACV anchors: (0.838, 0.641)")
    print()


if __name__ == "__main__":
    main()
