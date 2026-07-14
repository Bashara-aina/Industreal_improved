"""
Efficiency Report Generator
============================
Doc 03 B.1: Standard efficiency benchmark for POPW.

Reports:
- Total / trainable params (M)
- GFLOPs at 1280x720 (via fvcore.nn.FlapCountAnalysis)
- FPS over 200 forward passes after warmup
- Latency p50/p95/p99 (ms)
- Peak GPU memory (MB)

Usage:
    python efficiency_report.py
    python efficiency_report.py --backbone convnext_tiny --batch_size 1
    python efficiency_report.py --backbone resnet50 --use_headpose_film --use_videomae

Author: Bashara
Date: April 2026
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "0")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn

import config as C
import model as _model_module

POPWMultiTaskModel = getattr(_model_module, "POPWMultiTaskModel")
count_parameters = getattr(_model_module, "count_parameters")


def count_gflops(model: nn.Module, input_shape=(1, 3, 720, 1280)) -> float:
    """Count GFLOPs using fvcore."""
    try:
        from fvcore.nn import FlopCountAnalysis

        x = torch.randn(*input_shape)
        flops = FlopCountAnalysis(model, (x,))
        total = flops.total()
        return total / 1e9
    except ImportError:
        return -1.0


def measure_streaming_fps(
    model: nn.Module,
    device: torch.device,
    seq_length: int = 300,
    n_warmup: int = 20,
) -> Dict[str, float]:
    """
    Measure streaming FPS — processing a continuous video sequence one frame at a time.

    Doc 03 B.2: Streams T=seq_length frames one-by-one (as they arrive from a camera)
    through the full model including PSR temporal cache. Measures throughput without
    real-time waiting (pure processing FPS). Simulates the inference pipeline where
    each frame is processed as it arrives in a live video stream.

    Args:
        model: POPWMultiTaskModel in eval mode
        device: torch device
        seq_length: number of frames in the simulated video stream
        n_warmup: warmup iterations before timing

    Returns:
        dict with streaming_fps and frame processing times
    """
    model.eval()
    input_shape = (1, 3, 720, 1280)
    x = torch.randn(*input_shape, device=device)

    recording_ids = [f"seq_{i:06d}" for i in range(seq_length)]
    camera_views = ["rgb"] * seq_length

    warmup_x = torch.randn(*input_shape, device=device)
    with torch.no_grad():
        for _ in range(n_warmup):
            model(warmup_x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    with torch.no_grad():
        t0 = time.perf_counter()
        for i in range(seq_length):
            model(x, video_ids=[recording_ids[i]], camera_views=[camera_views[i]])
        if device.type == "cuda":
            torch.cuda.synchronize()

    elapsed = time.perf_counter() - t0
    streaming_fps = seq_length / elapsed

    return {
        "streaming_fps": streaming_fps,
        "total_frames": seq_length,
        "total_time_s": elapsed,
    }


def measure_fps(
    model: nn.Module,
    device: torch.device,
    input_shape: tuple = (1, 3, 720, 1280),
    n_warmup: int = 20,
    n_runs: int = 200,
) -> float:
    """Measure FPS over n_runs forward passes after n_warmup warmup iterations."""
    model.eval()
    x = torch.randn(*input_shape, device=device)

    if device.type == "cuda":
        torch.cuda.synchronize()

    with torch.no_grad():
        for _ in range(n_warmup):
            model(x)

        if device.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(n_runs):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

    elapsed = time.perf_counter() - t0
    return n_runs / elapsed


def measure_latency(
    model: nn.Module,
    device: torch.device,
    input_shape: tuple = (1, 3, 720, 1280),
    n_warmup: int = 20,
    n_runs: int = 200,
) -> Dict[str, float]:
    """Measure latency percentiles in ms."""
    model.eval()
    latencies = []

    x = torch.randn(*input_shape, device=device)

    with torch.no_grad():
        for _ in range(n_warmup):
            model(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    with torch.no_grad():
        for _ in range(n_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000)

    latencies = sorted(latencies)
    n = len(latencies)
    return {
        "p50_ms": latencies[n // 2],
        "p95_ms": latencies[int(n * 0.95)],
        "p99_ms": latencies[int(n * 0.99)],
    }


def measure_memory(
    model: nn.Module,
    device: torch.device,
    input_shape: tuple = (1, 3, 720, 1280),
) -> float:
    """Peak GPU memory in MB during a forward pass."""
    if device.type != "cuda":
        return -1.0

    torch.cuda.reset_peak_memory_stats()
    model.eval()
    x = torch.randn(*input_shape, device=device)
    with torch.no_grad():
        model(x)
    peak_bytes = torch.cuda.max_memory_allocated()
    return peak_bytes / 1024 / 1024


def build_report(
    backbone_type: str = "resnet50",
    use_headpose_film: bool = False,
    use_videomae: bool = False,
    pretrained: bool = False,
    batch_size: int = 1,
    device: Optional[torch.device] = None,
) -> Dict:
    """Build full efficiency report for a POPW configuration."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = POPWMultiTaskModel(
        pretrained=pretrained,
        backbone_type=backbone_type,
        use_headpose_film=use_headpose_film,
        use_videomae=use_videomae,
    ).to(device)
    model.eval()

    params = count_parameters(model)

    input_shape = (batch_size, 3, 720, 1280)
    gflops = count_gflops(model, input_shape)
    fps = measure_fps(model, device, input_shape)
    latency = measure_latency(model, device, input_shape)
    memory_mb = measure_memory(model, device, input_shape)

    # Doc 03 B.2: Streaming FPS (per-frame processing through full pipeline)
    seq_length = 300
    streaming = measure_streaming_fps(model, device, seq_length=seq_length)

    return {
        "config": {
            "backbone": backbone_type,
            "headpose_film": use_headpose_film,
            "videomae": use_videomae,
            "batch_size": batch_size,
            "device": str(device),
        },
        "params": {
            "total_m": params["total_all"] / 1e6,
            "trainable_m": params["total_trainable"] / 1e6,
            "frozen_m": (params["total_all"] - params["total_trainable"]) / 1e6,
        },
        "gflops": gflops,
        "fps": fps,
        "streaming_fps": streaming["streaming_fps"],
        "latency_ms": latency,
        "peak_memory_mb": memory_mb,
    }


def print_report(report: Dict):
    """Pretty-print a single efficiency report."""
    cfg = report["config"]
    prm = report["params"]

    print("=" * 60)
    print("POPW EFFICIENCY REPORT")
    print("=" * 60)
    print(f"  Backbone:        {cfg['backbone']}")
    print(f"  HeadPoseFiLM:    {cfg['headpose_film']}")
    print(f"  VideoMAE:        {cfg['videomae']}")
    print(f"  Batch size:      {cfg['batch_size']}")
    print(f"  Device:          {cfg['device']}")
    print("-" * 60)
    print(f"  Total params:    {prm['total_m']:.2f} M")
    print(f"  Trainable:       {prm['trainable_m']:.2f} M")
    print(f"  Frozen:          {prm['frozen_m']:.2f} M")
    print(f"  GFLOPs:          {report['gflops']:.1f}")
    print(f"  FPS (batched):   {report['fps']:.1f}")
    if "streaming_fps" in report:
        print(f"  Streaming FPS:   {report['streaming_fps']:.1f}")
    lat = report["latency_ms"]
    print(f"  Latency p50:     {lat['p50_ms']:.2f} ms")
    print(f"  Latency p95:     {lat['p95_ms']:.2f} ms")
    print(f"  Latency p99:     {lat['p99_ms']:.2f} ms")
    mem = report["peak_memory_mb"]
    if mem > 0:
        print(f"  Peak GPU memory: {mem:.0f} MB")
    print("=" * 60)


BASELINE_CONFIGS = [
    ("ResNet-50 (RGB)", "resnet50", False, False),
    ("ConvNeXt-Tiny (RGB)", "convnext_tiny", False, False),
    ("ConvNeXt-Tiny + HPFiLM", "convnext_tiny", True, False),
    ("ConvNeXt-Tiny + HPFiLM + VMAE", "convnext_tiny", True, True),
]


def print_comparison_table(reports: list[Dict]) -> None:
    headers = [
        "Model",
        "Params (M)",
        "GFLOPs",
        "FPS",
        "Streaming FPS",
        "p50 (ms)",
        "p95 (ms)",
        "Mem (MB)",
    ]
    col_widths = [30, 10, 8, 7, 15, 9, 9, 9]

    def fmt_row(values: list[str]) -> str:
        parts = []
        for val, w in zip(values, col_widths):
            parts.append(f" {val:<{w - 1}}")
        return "|".join(parts)

    sep = "+" + "+".join("-" * w for w in col_widths) + "+"
    header_row = fmt_row(headers)
    print("\n" + sep)
    print("|" + header_row + "|")
    print(sep.replace("-", "="))

    for report, (label, *_rest) in zip(reports, BASELINE_CONFIGS):
        cfg = report["config"]
        prm = report["params"]
        mem = report["peak_memory_mb"]
        mem_str = f"{mem:.0f}" if mem > 0 else "N/A"
        lat = report["latency_ms"]

        row_vals = [
            label[:29],
            f"{prm['total_m']:.1f}",
            f"{report['gflops']:.1f}" if report["gflops"] > 0 else "N/A",
            f"{report['fps']:.1f}",
            f"{report.get('streaming_fps', 0):.1f}",
            f"{lat['p50_ms']:.2f}",
            f"{lat['p95_ms']:.2f}",
            mem_str,
        ]
        print("|" + fmt_row(row_vals) + "|")
        print(sep)

    print("\n  Note: Streaming FPS = frame-by-frame processing FPS through full pipeline.\n")


def main(args):
    print("POPW Efficiency Report Generator")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    configs = []

    if args.all_configs:
        backbone_types = ["resnet50", "convnext_tiny"]
        flags = [(False, False), (True, False), (True, True)]
        for bt in backbone_types:
            for hpf, vmae in flags:
                configs.append((bt, hpf, vmae))
    else:
        configs.append((args.backbone, args.headpose_film, args.use_videomae))

    for bt, hpf, vmae in configs:
        report = build_report(
            backbone_type=bt,
            use_headpose_film=hpf,
            use_videomae=vmae,
            pretrained=False,
            batch_size=args.batch_size,
            device=device,
        )
        print()
        print_report(report)

    if args.baseline_compare:
        print("\n" + "=" * 70)
        print("MULTI-MODEL BASELINE COMPARISON")
        print("=" * 70)
        all_reports = []
        for label, bt, hpf, vmae in BASELINE_CONFIGS:
            report = build_report(
                backbone_type=bt,
                use_headpose_film=hpf,
                use_videomae=vmae,
                pretrained=False,
                batch_size=args.batch_size,
                device=device,
            )
            all_reports.append(report)
        print_comparison_table(all_reports)

    if args.onnx_export:
        onnx_path = Path(C.OUTPUT_ROOT) / "industreal_model.onnx"
        if onnx_path.exists():
            try:
                import onnxruntime as ort

                sess = ort.InferenceSession(str(onnx_path), providers=["CUDAExecutionProvider"])
                x = np.random.randn(1, 3, 720, 1280).astype(np.float32)
                for _ in range(20):
                    sess.run(None, {"input": x})
                t0 = time.perf_counter()
                for _ in range(200):
                    sess.run(None, {"input": x})
                onnx_fps = 200 / (time.perf_counter() - t0)
                print(f"\nONNX Runtime FPS: {onnx_fps:.1f}")
            except Exception as e:
                print(f"\nONNX Runtime benchmark failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="resnet50")
    parser.add_argument("--use_headpose_film", action="store_true")
    parser.add_argument("--use_videomae", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--all_configs", action="store_true")
    parser.add_argument(
        "--baseline_compare",
        action="store_true",
        help="Run all baseline configs and print comparison table",
    )
    parser.add_argument("--onnx_export", action="store_true")
    args = parser.parse_args()
    main(args)
