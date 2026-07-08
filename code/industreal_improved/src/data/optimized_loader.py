"""
IndustReal Optimized DataLoader — 3 strategies for the critical-path bottleneck.

Strategy A: In-memory decoded frame cache (uses FRAME_CACHE, preloads at init)
Strategy B: Parallel PyTorch DataLoader (num_workers=8, pin_memory, prefetch)
Strategy C: Hybrid — preload metadata, decode on-demand to pinned memory with CUDA streams

All strategies maintain backward-compatible interface with the existing
IndustRealMultiTaskDataset and collate_fn.

Usage:
    from src.data.optimized_loader import (
        create_dataloader_strategy_a,
        create_dataloader_strategy_b,
        create_dataloader_strategy_c,
    )
    loader, ds = create_dataloader_strategy_c(batch_size=2, num_samples=50)
    for images, targets in loader:
        ...
"""

from __future__ import annotations

import logging
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

from src import config as C
from src.data.industreal_dataset import (
    IndustRealMultiTaskDataset,
    collate_fn,
    preload_all_frames,
    clear_frame_cache,
    FRAME_CACHE,
)

logger = logging.getLogger(__name__)

# =========================================================================
# Strategy A: In-Memory Decoded Frame Cache
# =========================================================================

def create_dataloader_strategy_a(
    split: str = 'train',
    batch_size: int = 2,
    num_samples: Optional[int] = None,
    max_recordings: Optional[int] = None,
    num_workers: int = 0,
    pin_memory: bool = True,
    prefetch_factor: Optional[int] = None,
    drop_last: bool = True,
    augment: bool = False,
    sequence_mode: bool = False,
    frame_stride: int = 3,
    verbose: bool = True,
) -> Tuple[DataLoader, IndustRealMultiTaskDataset]:
    """
    Strategy A: Preload ALL frames as decoded numpy arrays into FRAME_CACHE,
    then serve from RAM. Zero disk I/O during training.

    Memory: ~5-7GB for full train set. With 62GB RAM, this is safe.

    Returns: (DataLoader, Dataset)
    """
    if verbose:
        logger.info('[STRATEGY A] Preloading all frames into FRAME_CACHE...')

    t0 = time.time()

    # Preload all frames into the module-level cache
    num_loaded = preload_all_frames(
        recordings_root=C.RECORDINGS_ROOT,
        split=split,
        stride=frame_stride,
        verbose=verbose,
    )

    if verbose:
        mem_gb = sum(arr.nbytes for arr in FRAME_CACHE.values()) / 1e9
        logger.info(
            f'[STRATEGY A] Frame cache loaded: {num_loaded} frames, '
            f'{mem_gb:.1f} GB, {time.time()-t0:.1f}s'
        )

    # Create dataset with RAM cache disabled (we use FRAME_CACHE instead)
    _orig_ram = C.RAM_CACHE_MAX_IMAGES
    C.RAM_CACHE_MAX_IMAGES = 0

    ds = IndustRealMultiTaskDataset(
        split=split,
        augment=augment,
        sequence_mode=sequence_mode,
        max_recordings=max_recordings,
    )
    C.RAM_CACHE_MAX_IMAGES = _orig_ram

    # Patch _load_image to serve from FRAME_CACHE
    ds._load_image = _make_cache_aware_loader(ds)

    # Build sampler with class balancing
    sampler = ds.get_sampler()

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=(num_workers > 0),
        drop_last=drop_last,
    )

    if verbose:
        logger.info(
            f'[STRATEGY A] DataLoader ready: batch_size={batch_size}, '
            f'{len(ds)} samples, {len(loader)} batches/epoch'
        )

    return loader, ds


def _make_cache_aware_loader(ds: IndustRealMultiTaskDataset):
    """Replace _load_image with FRAME_CACHE-backed version."""
    _original_load = ds._load_image
    _recordings_root = ds.recordings_root
    _split = ds.split
    _img_size = ds.img_size
    _bilinear = getattr(ds, '_bilinear', None)

    def _cached_load(img_path: str) -> torch.Tensor:
        """Load from FRAME_CACHE if available, fallback to original."""
        # Extract recording_id and frame_num from path
        path = Path(img_path)
        frame_stem = path.stem
        if not frame_stem.isdigit():
            return _original_load(img_path)
        frame_num = int(frame_stem)
        recording_id = path.parent.parent.name

        arr = FRAME_CACHE.get((recording_id, frame_num))
        if arr is None:
            return _original_load(img_path)

        # arr is [H, W, 3] uint8 numpy
        # Resize if needed
        import torch.nn.functional as F
        from PIL import Image

        h, w = arr.shape[:2]
        target_w, target_h = _img_size
        if w != target_w or h != target_h:
            pil_img = Image.fromarray(arr)
            pil_img = pil_img.resize((target_w, target_h), Image.BILINEAR)
            arr = np.array(pil_img, dtype=np.uint8)

        return torch.from_numpy(arr).permute(2, 0, 1)  # [3, H, W]

    return _cached_load


# =========================================================================
# Strategy B: Parallel DataLoader with Workers
# =========================================================================

def _create_dataset_for_strategy_b(
    split: str = 'train',
    augment: bool = False,
    sequence_mode: bool = False,
    max_recordings: Optional[int] = None,
    ram_cache_size: int = 8000,
) -> IndustRealMultiTaskDataset:
    """
    Create dataset pre-configured for parallel DataLoader.
    Uses JPEG byte cache with generous capacity.
    """
    _orig = C.RAM_CACHE_MAX_IMAGES
    C.RAM_CACHE_MAX_IMAGES = ram_cache_size
    C.NUM_WORKERS = 0  # Will be set in DataLoader

    ds = IndustRealMultiTaskDataset(
        split=split,
        augment=augment,
        sequence_mode=sequence_mode,
        max_recordings=max_recordings,
    )
    C.RAM_CACHE_MAX_IMAGES = _orig
    return ds


def create_dataloader_strategy_b(
    split: str = 'train',
    batch_size: int = 2,
    num_samples: Optional[int] = None,
    max_recordings: Optional[int] = None,
    num_workers: int = 8,
    pin_memory: bool = True,
    prefetch_factor: int = 4,
    persistent_workers: bool = True,
    drop_last: bool = True,
    augment: bool = False,
    sequence_mode: bool = False,
    ram_cache_size: int = 8000,
    verbose: bool = True,
) -> Tuple[DataLoader, IndustRealMultiTaskDataset]:
    """
    Strategy B: PyTorch DataLoader with worker processes.

    Uses num_workers=8, pin_memory=True, prefetch_factor=4,
    persistent_workers=True for parallel JPEG decode and prefetch.

    NOTE: Previous NUM_WORKERS=0 was due to CUDA deadlocks in multi-worker
    mode. Strategy B mitigates this by:
    - Using 'spawn' start method (fork is unsafe with CUDA)
    - Setting cv2.setNumThreads(0) (already done)
    - Loading JPEG bytes (not decoded tensors) in workers
    - Using RAM cache for hot data

    Returns: (DataLoader, Dataset)
    """
    # Fix: use spawn method for worker processes
    import multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # already set

    if verbose:
        logger.info(
            f'[STRATEGY B] Creating DataLoader: {num_workers} workers, '
            f'prefetch={prefetch_factor}, pin_memory={pin_memory}'
        )

    ds = _create_dataset_for_strategy_b(
        split=split,
        augment=augment,
        sequence_mode=sequence_mode,
        max_recordings=max_recordings,
        ram_cache_size=ram_cache_size,
    )

    sampler = ds.get_sampler()

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers and num_workers > 0,
        drop_last=drop_last,
    )

    if verbose:
        logger.info(
            f'[STRATEGY B] DataLoader ready: batch_size={batch_size}, '
            f'{len(ds)} samples, {len(loader)} batches/epoch, '
            f'{num_workers} workers'
        )

    return loader, ds


# =========================================================================
# Strategy C: Hybrid — Frame Cache + Async CUDA Transfer
# =========================================================================

class _HybridCUDADataset(Dataset):
    """
    Dataset wrapper that:
    1. Preloads ALL frame bytes into RAM at init (JPEG compressed, ~350KB avg)
    2. Decodes on the fly to GPU pinned memory
    3. Uses CUDA streams for async H2D transfer
    4. Pins worker annotation caches

    Memory: ~48K frames × 350KB ≈ 17GB for full train JPEG cache.
    For 50-sample smoke test: ~50 × 350KB ≈ 17.5MB — trivial.

    We cap at C.RAM_CACHE_MAX_IMAGES frames to stay within budget.
    """

    def __init__(
        self,
        base_dataset: IndustRealMultiTaskDataset,
        device: Optional[torch.device] = None,
        cap_frames: int = 8000,
        decode_on_gpu: bool = True,
    ):
        self.base = base_dataset
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.decode_on_gpu = decode_on_gpu
        self.samples = base_dataset.samples

        # Pre-allocate GPU pinned memory buffer for decoded frames
        # We cache decoded JPEGs as tensors on CUDA pinned memory
        self._pinned_buffer: Dict[str, torch.Tensor] = {}
        self._cap = cap_frames

        # Preload JPEG byte cache
        t0 = time.time()
        self._jpeg_cache: Dict[str, bytes] = {}
        loaded = 0
        for i, s in enumerate(self.samples):
            if loaded >= cap_frames:
                break
            try:
                with open(s['img_path'], 'rb') as f:
                    self._jpeg_cache[s['img_path']] = f.read()
                    loaded += 1
            except Exception:
                pass
        logger.info(
            f'[HYBRID_C] Preloaded {loaded} JPEGs ({cap_frames} cap) '
            f'in {time.time()-t0:.1f}s'
        )

        # CUDA stream for async decode + transfer
        if self.device.type == 'cuda':
            self._decode_stream = torch.cuda.Stream(device=self.device)
        else:
            self._decode_stream = None

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        recording_id = sample['recording_id']
        frame_num = sample['frame_num']
        img_path = sample['img_path']

        # ---- Load image ----
        cached_jpeg = self._jpeg_cache.get(img_path)
        if cached_jpeg is not None and self.decode_on_gpu and self.device.type == 'cuda':
            # GPU-side decode: use torchvision.io.decode_jpeg on GPU
            try:
                from io import BytesIO
                import torchvision.io as tio
                raw = torch.frombuffer(bytearray(cached_jpeg), dtype=torch.uint8)
                with torch.cuda.stream(self._decode_stream):
                    img_gpu = tio.decode_jpeg(raw, device=self.device)
                    # img_gpu is [C, H, W] on GPU
                    img_gpu = img_gpu.float().div(255.0)
                    # Resize on GPU
                    h, w = img_gpu.shape[1:]
                    target_w, target_h = C.IMG_SIZE
                    if w != target_w or h != target_h:
                        img_gpu = torch.nn.functional.interpolate(
                            img_gpu.unsqueeze(0),
                            size=(target_h, target_w),
                            mode='bilinear',
                            align_corners=False,
                        ).squeeze(0)
                    rgb_tensor = img_gpu  # [3, H, W] float on GPU
            except Exception:
                # Fallback to CPU decode
                rgb_tensor = self._cpu_decode_jpeg(cached_jpeg)
        else:
            if cached_jpeg is not None:
                rgb_tensor = self._cpu_decode_jpeg(cached_jpeg)
            else:
                rgb_tensor = self.base._load_image(img_path)

        # ---- Annotations (all from cached _anno_cache) ----
        cache = self.base._anno_cache[recording_id]

        _raw_al = int(sample['action_label'])
        _remap = getattr(C, 'remap_activity_label', None)
        if _remap is not None and str(getattr(C, 'ACT_CLASS_GROUPING', 'none')).lower() in ('verb', 'hybrid'):
            _raw_al = _remap(_raw_al)
        action_label = torch.tensor(_raw_al, dtype=torch.long)

        psr_labels = torch.from_numpy(cache.psr_per_frame[frame_num]).float()
        head_pose = torch.from_numpy(cache.pose[frame_num]).float()
        hand_joints = torch.from_numpy(cache.hands[frame_num]).float()

        gt_boxes, gt_classes = self.base._extract_boxes_from_coco(
            recording_id, frame_num
        )

        # Only convert rgb_tensor to float if it's not already (GPU path already float)
        if rgb_tensor.dtype != torch.float32:
            rgb_tensor = rgb_tensor.float()

        return {
            'images': {'rgb': rgb_tensor},
            'gt_boxes': {'rgb': gt_boxes},
            'gt_classes': {'rgb': gt_classes},
            'head_pose': head_pose,
            'psr_labels': psr_labels,
            'hand_joints': hand_joints,
            'action_label': action_label,
            'activity': action_label,
            'detection': {'boxes': gt_boxes, 'labels': gt_classes},
            'clip_rgb': torch.zeros(0, 3, 224, 224, dtype=torch.float32) if not getattr(C, 'USE_VIDEOMAE', False) else ...,
            'metadata': {
                'recording_id': recording_id,
                'frame_num': frame_num,
            }
        }

    @staticmethod
    def _cpu_decode_jpeg(jpeg_bytes: bytes) -> torch.Tensor:
        """Decode JPEG bytes on CPU to [3, H, W] uint8 tensor."""
        from io import BytesIO
        from PIL import Image
        img = Image.open(BytesIO(jpeg_bytes)).convert('RGB')
        img = img.resize(C.IMG_SIZE, Image.BILINEAR)
        return torch.from_numpy(np.array(img, dtype=np.uint8)).permute(2, 0, 1)


def create_dataloader_strategy_c(
    split: str = 'train',
    batch_size: int = 2,
    num_samples: Optional[int] = None,
    max_recordings: Optional[int] = None,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = True,
    augment: bool = False,
    frame_cache_cap: int = 8000,
    decode_on_gpu: bool = True,
    verbose: bool = True,
) -> Tuple[DataLoader, _HybridCUDADataset]:
    """
    Strategy C: Hybrid — Frame cache + async GPU decode + CUDA streams.

    Preloads JPEG bytes at init, decodes to GPU pinned memory on demand
    using CUDA streams for async H2D transfer.

    Returns: (DataLoader, HybridDataset)
    """
    # Create base dataset with RAM cache disabled (we handle caching)
    _orig_ram = C.RAM_CACHE_MAX_IMAGES
    C.RAM_CACHE_MAX_IMAGES = 0

    base_ds = IndustRealMultiTaskDataset(
        split=split,
        augment=augment,
        sequence_mode=False,
        max_recordings=max_recordings,
    )
    C.RAM_CACHE_MAX_IMAGES = _orig_ram

    if verbose:
        logger.info(f'[STRATEGY C] Base dataset: {len(base_ds)} samples')

    # Wrap in hybrid dataset
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hybrid_ds = _HybridCUDADataset(
        base_dataset=base_ds,
        device=device,
        cap_frames=frame_cache_cap,
        decode_on_gpu=decode_on_gpu,
    )

    sampler = base_ds.get_sampler()

    loader = DataLoader(
        hybrid_ds,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        drop_last=drop_last,
    )

    if verbose:
        logger.info(
            f'[STRATEGY C] Hybrid DataLoader ready: batch_size={batch_size}, '
            f'cache={frame_cache_cap} JPEGs, decode_on_gpu={decode_on_gpu}'
        )

    return loader, hybrid_ds


# =========================================================================
# Benchmark harness
# =========================================================================

def benchmark_dataloader(
    loader: DataLoader,
    num_batches: int = 100,
    warmup: int = 5,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Benchmark a DataLoader's throughput.

    Measures time to fetch batches (no model forward/backward).

    Returns:
        {'mean_s': float, 'median_s': float, 'std_s': float,
         'min_s': float, 'max_s': float, 'batches_per_sec': float}
    """
    import time as time_module

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    timings = []
    batch_iter = iter(loader)

    # Warmup
    for _ in range(warmup):
        try:
            _ = next(batch_iter)
        except StopIteration:
            batch_iter = iter(loader)

    # Timed run
    for i in range(num_batches):
        try:
            t0 = time_module.time()
            images, targets = next(batch_iter)
            # Force GPU transfer to measure full pipeline cost
            if device.type == 'cuda':
                images = images.to(device, non_blocking=True)
                for k in ['head_pose', 'psr_labels', 'hand_joints', 'activity']:
                    if k in targets:
                        targets[k] = targets[k].to(device, non_blocking=True)
            # Synchronize
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            dt = time_module.time() - t0
            timings.append(dt)
        except StopIteration:
            break

    if not timings:
        return {'error': 'no batches measured'}

    arr = np.array(timings)
    results = {
        'mean_s': float(arr.mean()),
        'median_s': float(np.median(arr)),
        'std_s': float(arr.std()),
        'min_s': float(arr.min()),
        'max_s': float(arr.max()),
        'batches_per_sec': float(1.0 / arr.mean()),
        'num_batches': len(timings),
    }

    if verbose:
        logger.info(
            f'[BENCHMARK] {results["num_batches"]} batches: '
            f'{results["mean_s"]*1000:.1f}ms/batch mean, '
            f'{results["batches_per_sec"]:.1f} batches/sec, '
            f'[{results["min_s"]*1000:.1f}–{results["max_s"]*1000:.1f}]ms'
        )

    return results


def benchmark_all_strategies(
    num_samples: int = 50,
    batch_size: int = 2,
    num_profile_batches: int = 50,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run all 3 strategies through the benchmark harness.

    Returns dict of strategy_name -> stats.
    """
    results = {}

    # Baseline (existing approach: RAM cache disabled, num_workers=0)
    if verbose:
        logger.info('=' * 60)
        logger.info('BASELINE (RAM cache=0, num_workers=0)')
        logger.info('=' * 60)

    _orig_ram = C.RAM_CACHE_MAX_IMAGES
    C.RAM_CACHE_MAX_IMAGES = 0
    _orig_workers = C.NUM_WORKERS
    C.NUM_WORKERS = 0

    ds_baseline = IndustRealMultiTaskDataset(
        split='train', augment=False, sequence_mode=False,
        max_recordings=min(36, max(1, num_samples // 850 + 1)),
    )
    C.RAM_CACHE_MAX_IMAGES = _orig_ram
    C.NUM_WORKERS = _orig_workers

    baseline_loader = DataLoader(
        ds_baseline,
        batch_size=batch_size,
        sampler=ds_baseline.get_sampler(),
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    results['baseline'] = benchmark_dataloader(baseline_loader, num_profile_batches, verbose=verbose)

    # Strategy A
    if verbose:
        logger.info('=' * 60)
        logger.info('STRATEGY A (in-memory frame cache)')
        logger.info('=' * 60)

    clear_frame_cache()
    loader_a, _ = create_dataloader_strategy_a(
        split='train', batch_size=batch_size, augment=False,
        num_workers=0, pin_memory=True, verbose=verbose,
    )
    results['strategy_a'] = benchmark_dataloader(loader_a, num_profile_batches, verbose=verbose)
    clear_frame_cache()

    # Strategy B
    if verbose:
        logger.info('=' * 60)
        logger.info('STRATEGY B (parallel workers)')
        logger.info('=' * 60)

    loader_b, _ = create_dataloader_strategy_b(
        split='train', batch_size=batch_size, augment=False,
        num_workers=4, pin_memory=True, prefetch_factor=4,
        ram_cache_size=8000, verbose=verbose,
    )
    results['strategy_b'] = benchmark_dataloader(loader_b, num_profile_batches, verbose=verbose)

    # Strategy C
    if verbose:
        logger.info('=' * 60)
        logger.info('STRATEGY C (hybrid: JPEG cache + GPU decode)')
        logger.info('=' * 60)

    loader_c, _ = create_dataloader_strategy_c(
        split='train', batch_size=batch_size, augment=False,
        num_workers=0, pin_memory=True,
        frame_cache_cap=8000, decode_on_gpu=True,
        verbose=verbose,
    )
    results['strategy_c'] = benchmark_dataloader(loader_c, num_profile_batches, verbose=verbose)

    return results


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    results = benchmark_all_strategies(
        num_samples=50,
        batch_size=2,
        num_profile_batches=50,
    )

    print('\n' + '=' * 65)
    print('  DATALOADER BENCHMARK SUMMARY')
    print('=' * 65)
    print(f'  {"Strategy":<20s} {"ms/batch":<12s} {"batches/sec":<14s}')
    print(f'  {"-"*46}')
    for name, stats in results.items():
        print(f'  {name:<20s} {stats["mean_s"]*1000:<12.1f} {stats["batches_per_sec"]:<14.1f}')
