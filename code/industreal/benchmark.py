import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

"""
WorkerNet Benchmark: GFLOPs, FPS, GPU Memory
=============================================
Standalone script. Does NOT interfere with training.
Loads model from checkpoint if available, otherwise initializes fresh.

Usage:
    python benchmark.py
    python benchmark.py --checkpoint runs/checkpoints/best.pth

Author: Bashara
Date: April 2026
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch

import config as C
from model import MultiTaskIndustReal

logger = logging.getLogger(__name__)


def measure_gflops(model, device, img_h=C.IMG_HEIGHT, img_w=C.IMG_WIDTH):
    """Measure GFLOPs using fvcore. Returns total GFLOPs or None if unavailable."""
    try:
        from fvcore.nn import FlopCountAnalysis, flop_count_table
    except ImportError:
        logger.warning('fvcore not installed. Run: pip install fvcore')
        return None, None

    dummy = torch.randn(1, 3, img_h, img_w, device=device)
    model.eval()
    with torch.no_grad():
        flops = FlopCountAnalysis(model, dummy)
        table = flop_count_table(flops)
        total_gflops = flops.total() / 1e9
    return total_gflops, table


def measure_fps(model, device, img_h=C.IMG_HEIGHT, img_w=C.IMG_WIDTH,
                warmup_iters=20, benchmark_iters=100):
    """Measure FPS at batch=1."""
    dummy = torch.randn(1, 3, img_h, img_w, device=device)
    model.eval()

    torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(dummy)
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(benchmark_iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(dummy)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)

    times = np.array(times)
    median_latency = np.median(times)
    peak_mem_mb = torch.cuda.max_memory_allocated(device) / 1e6

    return {
        'fps_batch1': float(1.0 / median_latency),
        'latency_ms': float(median_latency * 1000),
        'peak_gpu_mb': float(peak_mem_mb),
        'mean_fps': float(1.0 / np.mean(times)),
        'p95_latency_ms': float(np.percentile(times, 95) * 1000),
    }


def measure_fps_batched(model, device, batch_size=C.BATCH_SIZE,
                        img_h=C.IMG_HEIGHT, img_w=C.IMG_WIDTH,
                        warmup_iters=10, benchmark_iters=50):
    """Measure throughput at training batch size."""
    dummy = torch.randn(batch_size, 3, img_h, img_w, device=device)
    model.eval()

    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(dummy)
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(benchmark_iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(dummy)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)

    times = np.array(times)
    return {
        'fps_batched': float(batch_size / np.median(times)),
        'batch_size': batch_size,
        'batch_latency_ms': float(np.median(times) * 1000),
    }


def main(args):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        logger.error('CUDA not available. Benchmark requires GPU.')
        return

    logger.info(f'GPU: {torch.cuda.get_device_name()}')
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    logger.info(f'VRAM: {vram:.1f} GB')
    logger.info(f'Input size: {C.IMG_HEIGHT}x{C.IMG_WIDTH}')

    if args.checkpoint and Path(args.checkpoint).exists():
        logger.info(f'Loading checkpoint: {args.checkpoint}')
        model = MultiTaskIndustReal(pretrained=False).to(device)
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt['model'], strict=False)
        logger.info(f'  Loaded from epoch {ckpt.get("epoch", "?")}')
    else:
        logger.info('No checkpoint found. Initializing fresh model.')
        model = MultiTaskIndustReal(pretrained=True).to(device)

    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('\n' + '=' * 60)
    print('GFLOPS MEASUREMENT')
    print('=' * 60)
    gflops, table = measure_gflops(model, device)
    if gflops is not None:
        print(table)
        print(f'\nTotal GFLOPs: {gflops:.2f}')
    else:
        print('Skipped (install fvcore: pip install fvcore)')

    print('\n' + '=' * 60)
    print('FPS BENCHMARK (batch=1)')
    print('=' * 60)
    fps_results = measure_fps(model, device)
    print(f'  Median FPS     : {fps_results["fps_batch1"]:.1f}')
    print(f'  Median latency : {fps_results["latency_ms"]:.1f} ms')
    print(f'  P95 latency    : {fps_results["p95_latency_ms"]:.1f} ms')
    print(f'  Peak GPU memory: {fps_results["peak_gpu_mb"]:.0f} MB')

    print('\n' + '=' * 60)
    print(f'FPS BENCHMARK (batch={C.BATCH_SIZE})')
    print('=' * 60)
    try:
        batch_results = measure_fps_batched(model, device)
        print(f'  Throughput     : {batch_results["fps_batched"]:.1f} images/sec')
        print(f'  Batch latency  : {batch_results["batch_latency_ms"]:.1f} ms')
    except torch.cuda.OutOfMemoryError:
        print(f'  OOM at batch={C.BATCH_SIZE}. Skipped.')
        batch_results = {}

    print('\n' + '=' * 60)
    print('SUMMARY (copy to guide)')
    print('=' * 60)
    print(f'  GFLOPs (720x1280, batch=1) : {gflops:.2f}' if gflops else '  GFLOPs: install fvcore')
    print(f'  Parameters (total)         : {total_params:,}')
    print(f'  Parameters (trainable)     : {trainable_params:,}')
    print(f'  FPS (batch=1)              : {fps_results["fps_batch1"]:.1f}')
    print(f'  Latency (batch=1)          : {fps_results["latency_ms"]:.1f} ms')
    print(f'  Peak GPU Memory (inference): {fps_results["peak_gpu_mb"]:.0f} MB')
    if batch_results:
        print(f'  FPS (batch={C.BATCH_SIZE})             : {batch_results["fps_batched"]:.1f}')
    print('=' * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark WorkerNet IndustReal Model')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint path (optional)')
    main(parser.parse_args())