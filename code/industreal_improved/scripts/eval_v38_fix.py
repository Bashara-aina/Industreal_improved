#!/usr/bin/env python3
"""Eval mtl_v3.8_fix checkpoint (bias-update disabled) vs v3.7 baseline.

This script is configured to evaluate the trained checkpoint and produce
a comparison report against the v3.7 baseline mAP=0.0519.

Usage:
    python scripts/eval_v38_fix.py \
        --checkpoint runs/mtl_v3.8_fix/checkpoints/phase2_e0_bN.pth \
        --output /tmp/v38_eval.json
"""
import argparse, json, logging, subprocess, sys, time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('eval_v38_fix')


def run_eval(checkpoint: str, num_anchors: int = 16, max_frames: int = 2000,
             output: str = '/tmp/v38_eval_results.json') -> dict:
    """Run the fixed mAP eval script on the checkpoint."""
    cmd = [
        'python', 'scripts/eval_mvit_mAP.py',
        '--checkpoint', checkpoint,
        '--num-anchors', str(num_anchors),
        '--max-frames', str(max_frames),
        '--output', output,
    ]
    logger.info(f"Running: {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0
    logger.info(f"Eval finished in {dt:.0f}s")
    if result.returncode != 0:
        logger.error(f"Eval failed:\n{result.stderr[-2000:]}")
        return None
    with open(output) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--num-anchors', type=int, default=16)
    parser.add_argument('--max-frames', type=int, default=2000)
    parser.add_argument('--output', type=str, default='/tmp/v38_eval.json')
    parser.add_argument('--v37-baseline', type=str, default='/tmp/mvit_eval_v3.7_fixed.json')
    args = parser.parse_args()

    # Run eval on v3.8 checkpoint
    v38_result = run_eval(args.checkpoint, args.num_anchors, args.max_frames)

    # Load v3.7 baseline
    v37_result = None
    try:
        with open(args.v37_baseline) as f:
            v37_result = json.load(f)
    except FileNotFoundError:
        logger.warning(f"V3.7 baseline not found: {args.v37_baseline}")

    # Compare
    if v38_result and v37_result:
        v37_mAP = v37_result.get('mAP50_annotated_frames', 0)
        v38_mAP = v38_result.get('mAP50_annotated_frames', 0)
        delta = v38_mAP - v37_mAP
        rel = (v38_mAP / v37_mAP - 1) * 100 if v37_mAP > 0 else 0
        logger.info(f"\n{'='*60}")
        logger.info(f"COMPARISON: v3.7 vs v3.8_fix (bias-update disabled)")
        logger.info(f"  v3.7  mAP@0.5: {v37_mAP:.4f}")
        logger.info(f"  v3.8  mAP@0.5: {v38_mAP:.4f}")
        logger.info(f"  Delta:         {delta:+.4f} ({rel:+.1f}%)")
        logger.info(f"  v3.7 n_preds: {v37_result.get('n_preds', 0):,}")
        logger.info(f"  v3.8 n_preds: {v38_result.get('n_preds', 0):,}")
        logger.info(f"  v3.7 n_gt:    {v37_result.get('n_gt_boxes', 0)}")
        logger.info(f"  v3.8 n_gt:    {v38_result.get('n_gt_boxes', 0)}")

        verdict = "tied"
        if delta > 0.02:
            verdict = "improvement"
            logger.info(f"  ✓ IMPROVEMENT > 2pp — fix is validated")
        elif delta > 0.005:
            verdict = "marginal-improvement"
            logger.info(f"  ~ Marginal improvement")
        elif delta < -0.005:
            verdict = "regression"
            logger.info(f"  ✗ REGRESSION — revert the fix")
        else:
            verdict = "tied"
            logger.info(f"  ~ Tied")
        summary = {
            'checkpoint': args.checkpoint,
            'v37_mAP': v37_mAP,
            'v38_mAP': v38_mAP,
            'delta': delta,
            'relative_change_pct': rel,
            'v37_n_preds': v37_result.get('n_preds'),
            'v38_n_preds': v38_result.get('n_preds'),
            'verdict': verdict,
            'v37_full': v37_result,
            'v38_full': v38_result,
        }
    else:
        summary = {'checkpoint': args.checkpoint, 'v38_result': v38_result, 'v37_result': v37_result}

    with open(args.output, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nSummary saved: {args.output}")


if __name__ == '__main__':
    main()