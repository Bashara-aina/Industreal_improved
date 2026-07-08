#!/usr/bin/env python3
"""
measure_efficiency.py — FVcore-based efficiency measurement for V5, V8 (and Tier F).

Replaces the fabricated 4x / 600M table from 167/170 with measured numbers.

Per 175 section 7.4:
  - Use fvcore/ptflops for params + FLOPs.
  - Measured FPS and peak VRAM on identical hardware.
  - Report params/storage/one-pass latency as the real efficiency win.
  - Report FLOPs honestly (two frozen-weight modes may exceed single ConvNeXt pass — do not spin it).

Usage:
  python3 scripts/measure_efficiency.py

Output:
  src/runs/rf_stages/checkpoints/efficiency_measured/metrics.json
"""

import json
import logging
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import torch

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)  # parent of scripts/ = repo root
CODE_DIR = os.path.join(REPO_ROOT, "code", "industreal_improved")
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, os.path.join(CODE_DIR, "src"))

from fvcore.nn import FlopCountAnalysis

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("measure_efficiency")


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(
    f"Device: {DEVICE}  ({torch.cuda.device_count()} GPUs)"
    if torch.cuda.is_available()
    else f"Device: {DEVICE} (no CUDA)"
)

N_WARMUP = 10
N_TIMING = 50


# ---------------------------------------------------------------------------
# fvcore wrappers — FlopCountAnalysis expects positional tensor inputs
# ---------------------------------------------------------------------------
class ModelWrapperV5(torch.nn.Module):
    """Wraps POPWMultiTaskModel so forward accepts a single positional tensor."""
    def __init__(self, model):
        super().__init__()
        self._model = model

    def forward(self, images: torch.Tensor):
        return self._model(images=images)


class ModelWrapperV8(torch.nn.Module):
    """Wraps VideoMultiTaskModel so forward accepts a single positional tensor."""
    def __init__(self, model):
        super().__init__()
        self._model = model

    def forward(self, clip: torch.Tensor):
        return self._model(clip=clip)


class ModelWrapperV8Simple(torch.nn.Module):
    """Wraps simple V8Model so forward accepts a single positional tensor."""
    def __init__(self, model):
        super().__init__()
        self._model = model

    def forward(self, clip: torch.Tensor):
        return self._model(clip=clip)


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------
def measure_model(
    model: torch.nn.Module,
    model_name: str,
    input_tensor: torch.Tensor,
    wrapper_cls,
    batch: int = 1,
) -> dict:
    """Measure params, FLOPs, FPS, and peak VRAM for a model."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Measuring: {model_name}  (batch={batch})")
    logger.info(f"{'='*60}")

    model.to(DEVICE)
    model.eval()
    wrapper = wrapper_cls(model).to(DEVICE)
    wrapper.eval()

    device_tensor = input_tensor.to(DEVICE)

    # --- Params ---
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Total params:     {total_params/1e6:.2f}M")
    logger.info(f"  Trainable params: {trainable_params/1e6:.2f}M")

    # --- FLOPs (fvcore) ---
    flops = None
    try:
        with torch.no_grad():
            flops_analysis = FlopCountAnalysis(wrapper, device_tensor)
            flops = flops_analysis.total()
        logger.info(f"  FLOPs:            {flops/1e9:.2f} GFLOPs")
    except Exception as e:
        logger.warning(f"  FLOPs: FAILED — {e}")

    # --- Timing (FPS) ---
    with torch.no_grad():
        for _ in range(N_WARMUP):
            _ = wrapper(device_tensor)
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats() if DEVICE.type == "cuda" else None

    if DEVICE.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        with torch.no_grad():
            for _ in range(N_TIMING):
                _ = wrapper(device_tensor)
                torch.cuda.synchronize()
        end_event.record()
        torch.cuda.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event)
        fps = N_TIMING / (elapsed_ms / 1000.0)
        latency_ms = elapsed_ms / N_TIMING
        logger.info(f"  Timing:           {latency_ms:.1f} ms/forward")
        logger.info(f"  FPS:              {fps:.1f}")
    else:
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(N_TIMING):
                _ = wrapper(device_tensor)
        elapsed = time.perf_counter() - t0
        fps = N_TIMING / elapsed
        latency_ms = elapsed / N_TIMING * 1000
        logger.info(f"  Timing:           {latency_ms:.1f} ms/forward")
        logger.info(f"  FPS:              {fps:.1f}")

    # --- Peak VRAM (CUDA only) ---
    vram_gb = None
    if DEVICE.type == "cuda":
        vram_gb = torch.cuda.max_memory_allocated(DEVICE) / (1024**3)
        logger.info(f"  Peak VRAM:        {vram_gb:.2f} GB")

    # --- Storage estimate (params * 4 bytes for FP32) ---
    storage_mb = total_params * 4 / (1024**2)
    logger.info(f"  Storage (FP32):   {storage_mb:.1f} MB")

    return {
        "params_m": round(total_params / 1e6, 2),
        "trainable_params_m": round(trainable_params / 1e6, 2),
        "flops_g": round(flops / 1e9, 2) if flops is not None else None,
        "fps": round(fps, 1),
        "latency_ms": round(latency_ms, 1),
        "vram_gb": round(vram_gb, 2) if vram_gb is not None else None,
        "storage_mb": round(storage_mb, 1),
        "gpu": torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "cpu",
        "batch": batch,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    results = {}

    # ======================================================================
    # 1. V5 — POPWMultiTaskModel (ConvNeXt-Tiny)
    # ======================================================================
    logger.info("\n\n### Building V5 (POPWMultiTaskModel, ConvNeXt-Tiny)...")
    from src.models.model import POPWMultiTaskModel

    model_v5 = POPWMultiTaskModel(
        pretrained=False,
        backbone_type="convnext_tiny",
        use_headpose_film=True,
        use_hand_film=True,
        use_videomae=False,
        train_pose=True,
    )

    # V5 takes frame input [B, 3, H, W] at native resolution
    results["v5_b1"] = measure_model(
        model_v5, "V5 (ConvNeXt-Tiny, batch=1)",
        torch.randn(1, 3, 720, 1280), ModelWrapperV5, batch=1,
    )

    results["v5_b2"] = measure_model(
        model_v5, "V5 (ConvNeXt-Tiny, batch=2)",
        torch.randn(2, 3, 720, 1280), ModelWrapperV5, batch=2,
    )

    del model_v5
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    # ======================================================================
    # 2. V8 — VideoMultiTaskModel (MViTv2-S, 4 heads)
    # ======================================================================
    logger.info("\n\n### Building V8 (VideoMultiTaskModel, MViTv2-S)...")
    from src.models.video_backbone_multitask import VideoMultiTaskModel

    model_v8 = VideoMultiTaskModel(
        backbone_name="mvit_v2_s",
        pretrained=False,
        freeze_backbone=True,
        use_checkpoint=False,
        train_pose=True,
    )

    # V8 takes clip [B, 3, T, H, W] at 224x224, T=16
    results["v8_b1"] = measure_model(
        model_v8, "V8 (MViTv2-S, batch=1)",
        torch.randn(1, 3, 16, 224, 224), ModelWrapperV8, batch=1,
    )

    results["v8_b2"] = measure_model(
        model_v8, "V8 (MViTv2-S, batch=2)",
        torch.randn(2, 3, 16, 224, 224), ModelWrapperV8, batch=2,
    )

    del model_v8
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    # ======================================================================
    # 3. Simple V8Model (frozen MViTv2-S, linear heads)
    # ======================================================================
    logger.info("\n\n### Building V8Simple (frozen MViTv2-S, linear heads)...")
    from scripts.train_v8_multitask import V8Model

    model_v8s = V8Model(num_classes=69, num_psr_comps=11)

    results["v8_simple_b1"] = measure_model(
        model_v8s, "V8-Simple (frozen MViTv2-S, linear heads, batch=1)",
        torch.randn(1, 3, 16, 224, 224), ModelWrapperV8Simple, batch=1,
    )

    del model_v8s
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    # ======================================================================
    # 4. Tier F — Hiera-B (not yet available)
    # ======================================================================
    results["tier_f"] = {
        "status": "not_available",
        "note": "Hiera-B backbone + 4-head Tier F model not yet implemented "
                "(Agent 2 pending). Once built, measure with clip [B, 3, 16, 224, 224].",
    }

    # ======================================================================
    # Save results
    # ======================================================================
    output_dir = os.path.join(
        CODE_DIR, "src", "runs", "rf_stages", "checkpoints", "efficiency_measured"
    )
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "metrics.json")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info(f"Results saved to: {output_path}")
    logger.info(f"{'='*60}")
    logger.info(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
