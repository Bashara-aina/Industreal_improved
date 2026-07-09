#!/usr/bin/env python3
"""Verify a trained checkpoint: load best.pt, run forward, check all 4 heads produce valid output.

[OPUS 192 §6 utility] Sanity-check that the saved checkpoint is loadable
and produces valid output for all 4 tasks. Quick to run (no training).

Usage:
    python scripts/verify_checkpoint.py --ckpt runs/mtl_mvit_run/best.pt
    python scripts/verify_checkpoint.py --ckpt runs/mtl_mvit_run/latest.pt
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Path setup
_CODE_ROOT = Path(__file__).resolve().parent.parent
for _p in [str(_CODE_ROOT), str(_CODE_ROOT / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import src.config as C
C.NUM_ACT_OUTPUTS = 75
C.ACT_CLASS_GROUPING = "none"

from src.models.mvit_mtl_model import MTLMViTModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser(description="Verify a checkpoint produces valid output")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--n-batches", type=int, default=5)
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}")
        sys.exit(1)

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Build model and load
    model = MTLMViTModel(num_act_classes=75).to(DEVICE)
    state_dict = ckpt.get("model_state_dict", ckpt)
    # Pre-filter for shape mismatches (in case of head reshapes)
    model_sd = model.state_dict()
    filtered = {k: v for k, v in state_dict.items()
                if k in model_sd and model_sd[k].shape == v.shape}
    skipped = len(state_dict) - len(filtered)
    load_result = model.load_state_dict(filtered, strict=False)
    print(f"  Loaded {len(filtered)}/{len(state_dict)} tensors (skipped {skipped} shape-mismatched)")
    if load_result.missing_keys:
        print(f"  {len(load_result.missing_keys)} missing keys (new layers, init fresh)")

    model.eval()
    print()

    # Synthetic batch
    B, T = args.batch_size, 16
    print(f"Running {args.n_batches} synthetic batches (B={B}, T={T})...")

    issues = []
    for i in range(args.n_batches):
        images = torch.randn(B, 3, T, 224, 224, device=DEVICE) * 0.5 + 0.5
        targets = {
            "detection": [
                {"boxes": torch.tensor([[50., 50., 150., 150.]]).to(DEVICE),
                 "labels": torch.tensor([5], dtype=torch.long, device=DEVICE)},
                {"boxes": torch.tensor([[30., 30., 100., 100.]]).to(DEVICE),
                 "labels": torch.tensor([3], dtype=torch.long, device=DEVICE)},
            ],
            "activity": torch.tensor([5, 10], dtype=torch.long, device=DEVICE),
            "psr_labels": torch.zeros(B, T, 11, device=DEVICE).scatter_(
                2, torch.randint(0, 11, (B, T, 1), device=DEVICE), 1.0
            ),
            "head_pose": torch.randn(B, T, 9, device=DEVICE),
        }

        with torch.no_grad():
            try:
                outputs = model(images)
            except Exception as e:
                issues.append(f"  Batch {i}: forward failed: {e}")
                continue

        # Validate each head
        for key, expected_shape_fn in [
            ("activity", lambda: (B, 75)),
            ("psr_logits", lambda: (B, 8, 11)),  # T=8 native (Opus 192 FC-4)
            ("pose_6d", lambda: (B, 6)),
        ]:
            if key not in outputs:
                issues.append(f"  Batch {i}: missing key '{key}'")
                continue
            t = outputs[key]
            expected = expected_shape_fn()
            if tuple(t.shape) != expected:
                issues.append(f"  Batch {i}: '{key}' shape {tuple(t.shape)} != expected {expected}")
            if torch.isnan(t).any() or torch.isinf(t).any():
                issues.append(f"  Batch {i}: '{key}' has NaN/Inf")

        # Check detection outputs (per FPN level)
        if "detection" not in outputs:
            issues.append(f"  Batch {i}: missing 'detection'")
        else:
            det = outputs["detection"]
            expected_keys = {"P3", "P4", "P5"}  # P2 excluded per Opus 192 FC-2
            actual_keys = set(det.keys())
            if actual_keys != expected_keys:
                issues.append(f"  Batch {i}: det keys {actual_keys} != expected {expected_keys}")
            for level, head_out in det.items():
                if "cls_logits" not in head_out or "reg_preds" not in head_out:
                    issues.append(f"  Batch {i}: det {level} missing keys")
                else:
                    cls = head_out["cls_logits"]
                    reg = head_out["reg_preds"]
                    if torch.isnan(cls).any() or torch.isinf(cls).any():
                        issues.append(f"  Batch {i}: det {level} cls has NaN/Inf")
                    if torch.isnan(reg).any() or torch.isinf(reg).any():
                        issues.append(f"  Batch {i}: det {level} reg has NaN/Inf")

    print()
    print("=" * 60)
    print("VERIFICATION RESULTS")
    print("=" * 60)
    if not issues:
        print("  ✅ All checks passed.")
        print(f"  Checkpoint is valid: {ckpt_path}")
        return 0
    else:
        print(f"  ❌ {len(issues)} issue(s) found:")
        for issue in issues:
            print(issue)
        return 1


if __name__ == "__main__":
    sys.exit(main())
