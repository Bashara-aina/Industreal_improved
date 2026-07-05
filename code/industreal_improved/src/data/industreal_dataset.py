"""
 IndustReal Multi-Task Dataset Loader
 =====================================
 Supports:
   - Action Recognition (AR): 74 atomic action classes (per-frame labels via interpolation)
   - Assembly State Detection (ASD): 24 classes (COCO format OD_labels.json)
   - Procedure Step Recognition (PSR): 11 components (per-frame binary state via interpolation)
   - Head Pose: 9-DoF (per-frame from pose.csv)
   - Hand Joints: 52 coordinates (per-frame from hands.csv)
   - Single egocentric RGB camera (1280x720, 10 FPS)

 Recording structure:
     recordings/
     ├── train/
     │   └── {recording_id}/
     │       ├── rgb/000000.jpg ...          (1280x720, 10 FPS)
     │       ├── AR_labels.csv               (sparse per-recording action spans)
     │       ├── OD_labels.json              (COCO format object detections)
     │       ├── PSR_labels_raw.csv          (sparse per-component state changes)
     │       ├── pose.csv                    (dense 9-DoF head pose)
     │       └── hands.csv                   (dense 52-D hand joints)
     ├── val/
     └── test/
 """

from __future__ import annotations

import csv
import gc
import json
import logging
import random
import threading
import time as time_module
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional
from typing import Any, Dict, List, Optional, Tuple

import cv2
cv2.setNumThreads(0)
_cv_set_loglevel = getattr(cv2, 'setLogLevel', None)
_cv_log_level_error = getattr(cv2, 'LOG_LEVEL_ERROR', None)
if callable(_cv_set_loglevel) and _cv_log_level_error is not None:
    _cv_set_loglevel(_cv_log_level_error)

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image

from src import config as C

logger = logging.getLogger(__name__)

# =========================================================================
# Spatial Augmentation
# =========================================================================

COCO_FLIP_PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15), (16, 16)]


def apply_spatial_aug(
    image: torch.Tensor,
    boxes: torch.Tensor,
    keypoints: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply spatial augmentation (horizontal flip + random crop) during training.

    Args:
        image: [3, H, W] tensor
        boxes: [N, 4] xyxy tensor (can be empty)
        keypoints: [17, 2] or [17, 3] tensor (COCO format)

    Returns:
        aug_image: [3, H, W] tensor
        aug_boxes: [M, 4] xyxy tensor (invalid boxes filtered out)
        aug_keypoints: [17, 2] tensor with flipped/swapped keypoints
    """
    if not getattr(C, 'USE_SPATIAL_AUG', False):
        return image, boxes, keypoints

    _, H, W = image.shape

    # --- Horizontal flip (p=0.5) ---
    if random.random() < 0.5:
        image = torch.flip(image, dims=[2])  # [3, H, W]

        if boxes.shape[0] > 0:
            # Flip x coordinates: new_x = W - 1 - old_x
            boxes = boxes.clone()
            boxes[:, [0, 2]] = W - 1 - boxes[:, [2, 0]]

        if keypoints.shape[0] > 0:
            keypoints = keypoints.clone()
            keypoints[:, 0] = W - 1 - keypoints[:, 0]
            # Swap symmetric keypoint pairs
            for left_idx, right_idx in COCO_FLIP_PAIRS:
                keypoints[[left_idx, right_idx]] = keypoints[[right_idx, left_idx]]

    # --- Random crop ---
    scale = random.uniform(0.8, 1.0)
    aspect_ratio = random.uniform(0.9, 1.1)

    crop_h = int(H * scale)
    crop_w = int(W * scale * aspect_ratio)

    x_offset = random.randint(0, max(0, W - crop_w))
    y_offset = random.randint(0, max(0, H - crop_h))

    # Crop image
    image = image[:, y_offset:y_offset + crop_h, x_offset:x_offset + crop_w]
    # Resize back to original size
    image = torch.nn.functional.interpolate(
        image.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False
    ).squeeze(0)

    if boxes.shape[0] > 0:
        boxes = boxes.clone()
        boxes[:, [0, 2]] -= x_offset
        boxes[:, [1, 3]] -= y_offset

        # Clip to image bounds
        boxes[:, 0].clamp_(min=0, max=W - 1)
        boxes[:, 2].clamp_(min=0, max=W - 1)
        boxes[:, 1].clamp_(min=0, max=H - 1)
        boxes[:, 3].clamp_(min=0, max=H - 1)

        # Filter out invalid boxes (negative or zero area)
        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes = boxes[valid]

    if keypoints.shape[0] > 0:
        keypoints = keypoints.clone()
        keypoints[:, 0] -= x_offset
        keypoints[:, 1] -= y_offset
        # Clip to image bounds
        keypoints[:, 0].clamp_(min=0, max=W - 1)
        keypoints[:, 1].clamp_(min=0, max=H - 1)

    return image, boxes, keypoints


_IMAGENET_MEAN = torch.tensor(C.IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
_IMAGENET_STD  = torch.tensor(C.IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
_RESAMPLING = getattr(Image, 'Resampling', None)
_BILINEAR = _RESAMPLING.BILINEAR if _RESAMPLING is not None else Image.BILINEAR

# =========================================================================
# Module-Level Frame Cache (Bashara 2026-05-14)
# =========================================================================
# Pre-loads ALL frames into RAM once, eliminating per-batch HDD reads.
# Access: FRAME_CACHE[(recording_id, frame_num)] → np.ndarray [H, W, 3] uint8
# Memory: ~5-7GB for full train set (48,586 frames × 1280×720×3 bytes ≈ 33GB raw,
# but JPEG compression + numpy overhead → ~5.6GB). With 64GB RAM this is safe.
# =========================================================================

FRAME_CACHE: Dict[str, np.ndarray] = {}  # {(recording_id, frame_num): np.ndarray}
_FRAME_CACHE_LOADED = False
_FRAME_CACHE_LOCK = threading.Lock()
_FRAME_CACHE_METADATA: Dict[str, Dict] = {}  # {recording_id: {num_frames, loaded, corrupt_frames[]}}


def clear_frame_cache() -> None:
    """Free all RAM used by FRAME_CACHE. Idempotent.

    Called at epoch boundaries by train.py to release ~5-7GB between epochs
    so the OS can reclaim memory before the next epoch's pre-load step.
    Also resets the _FRAME_CACHE_LOADED flag so a subsequent _preload_frames()
    call will re-populate the cache.
    """
    global _FRAME_CACHE_LOADED, _FRAME_CACHE_METADATA
    with _FRAME_CACHE_LOCK:
        n_entries = len(FRAME_CACHE)
        n_bytes = sum(arr.nbytes for arr in FRAME_CACHE.values())
        FRAME_CACHE.clear()
        _FRAME_CACHE_METADATA.clear()
        _FRAME_CACHE_LOADED = False
    logger.info(
        f'[FRAME_CACHE] cleared {n_entries} entries '
        f'({n_bytes / (1024 ** 3):.2f} GB freed)'
    )

# Corrupted frames known to fail PIL — substitute with zero arrays
_KNOWN_BAD_FRAMES: set = set()


def preload_all_frames(
    recordings_root: Path,
    split: str = 'train',
    stride: int = 3,
    verbose: bool = True,
    progress_bar=None,
) -> int:
    """
    Pre-load ALL frames into FRAME_CACHE.
    Call this ONCE before training starts.
    Returns total frames loaded.
    """
    global _FRAME_CACHE_LOADED, _FRAME_CACHE_METADATA

    if _FRAME_CACHE_LOADED:
        if verbose:
            logger.info('[FRAME_CACHE] Already loaded, skipping preload.')
        return len(FRAME_CACHE)

    t0 = time_module.time()
    recordings_dir = recordings_root / split

    # Build list of all (recording_id, frame_num) tuples
    all_frames: List[tuple] = []
    rec_frame_counts: Dict[str, int] = {}

    for rec_dir in sorted(recordings_dir.iterdir()):
        if not rec_dir.is_dir():
            continue
        recording_id = rec_dir.name
        rgb_dir = rec_dir / 'rgb'
        if not rgb_dir.exists():
            continue

        frame_nums = []
        for entry in sorted(rgb_dir.iterdir()):
            name = entry.name
            if name.endswith('.jpg') and name[:-4].isdigit():
                frame_nums.append(int(name[:-4]))

        if not frame_nums:
            continue

        # Apply stride (train stride=3, val stride=1)
        if stride > 1:
            frame_nums = [fn for fn in frame_nums if fn % stride == 0]

        rec_frame_counts[recording_id] = len(frame_nums)

        for fn in frame_nums:
            all_frames.append((recording_id, fn))

    if verbose:
        logger.info(
            f'[FRAME_CACHE] Pre-loading {len(all_frames)} frames '
            f'({len(rec_frame_counts)} recordings) into RAM...'
        )

    # Pre-load all frames using thread pool
    import concurrent.futures
    from concurrent.futures import ThreadPoolExecutor

    def _load_one(args):
        recording_id, frame_num = args
        rgb_dir = recordings_dir / recording_id / 'rgb'
        img_path = rgb_dir / f'{frame_num:06d}.jpg'
        try:
            with open(img_path, 'rb') as f:
                data = f.read()
            import io
            img = Image.open(io.BytesIO(data))
            arr = np.array(img, dtype=np.uint8)  # [H, W, 3]
            return (recording_id, frame_num, arr)
        except Exception as e:
            # Known corrupted frames → zero array
            return (recording_id, frame_num, None)

    loaded_count = 0
    failed_count = 0

    # Use ThreadPoolExecutor for I/O parallelism (HDD sequential read ≈ 70 MB/s)
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(_load_one, args): args for args in all_frames}

        for future in concurrent.futures.as_completed(futures):
            try:
                recording_id, frame_num, arr = future.result(timeout=30)
                if arr is not None:
                    FRAME_CACHE[(recording_id, frame_num)] = arr
                    loaded_count += 1
                else:
                    # Corrupt frame → zero array
                    arr = np.zeros((C.IMG_HEIGHT, C.IMG_WIDTH, 3), dtype=np.uint8)
                    FRAME_CACHE[(recording_id, frame_num)] = arr
                    _KNOWN_BAD_FRAMES.add((recording_id, frame_num))
                    failed_count += 1
            except Exception:
                failed_count += 1
                arr = np.zeros((C.IMG_HEIGHT, C.IMG_WIDTH, 3), dtype=np.uint8)
                FRAME_CACHE[(recording_id, frame_num)] = arr

            if progress_bar is not None:
                progress_bar.update(1)

    _FRAME_CACHE_METADATA = {
        'loaded': True,
        'num_frames': loaded_count,
        'num_recordings': len(rec_frame_counts),
        'corrupt_frames': list(_KNOWN_BAD_FRAMES),
        'rec_frame_counts': rec_frame_counts,
    }
    _FRAME_CACHE_LOADED = True

    elapsed = time_module.time() - t0
    mem_bytes = sum(arr.nbytes for arr in FRAME_CACHE.values())
    mem_gb = mem_bytes / 1e9

    if verbose:
        logger.info(
            f'[FRAME_CACHE] Done. Loaded {loaded_count} frames '
            f'({failed_count} corrupt → zero) in {elapsed:.0f}s ≈ {mem_gb:.1f} GB. '
            f'Speed: {loaded_count / elapsed:.0f} frames/s.'
        )

    return loaded_count


def get_cached_frame(recording_id: str, frame_num: int) -> np.ndarray:
    """Access a pre-loaded frame from FRAME_CACHE. Returns zero array if missing."""
    arr = FRAME_CACHE.get((recording_id, frame_num))
    if arr is not None:
        return arr
    # Fallback: return zero array (should not happen after preload)
    return np.zeros((C.IMG_HEIGHT, C.IMG_WIDTH, 3), dtype=np.uint8)


# =========================================================================
# COCO Cache (process-level) — one JSON per recording
# =========================================================================
_PROC_COCO_CACHE: Dict[str, Dict[int, List[Dict[str, Any]]]] = {}
_PROC_COCO_CACHE_MAX = max(1, min(getattr(C, 'COCO_CACHE_SIZE', 30), 30))

# [FIX D5] COCO image dimensions cache — parsed once per recording, reused per frame.
# Maps coco_path -> {frame_num: (width, height)}. Avoids re-parsing the full COCO
# JSON (~40 min/epoch overhead) every time _extract_boxes_from_coco resizes boxes.
_PROC_COCO_DIMS_CACHE: Dict[str, Dict[int, Tuple[int, int]]] = {}


def _parse_coco_file(coco_path: str) -> Dict[int, List[Dict[str, Any]]]:
    """
    Parse COCO JSON and return dict mapping frame_num → annotations list.
    The JSON uses file_name like '000000.jpg' → extract frame number.
    """
    if not Path(coco_path).exists():
        return {}
    try:
        with open(coco_path, 'r', encoding='utf-8', errors='ignore') as f:
            coco = json.load(f)
    except Exception:
        return {}

    # Build image_id → frame_num mapping from file_name
    id_to_frame: Dict[int, int] = {}
    for img_info in coco.get('images', []):
        file_name = img_info.get('file_name', '')
        stem = Path(file_name).stem
        try:
            frame_num = int(stem)
            id_to_frame[img_info['id']] = frame_num
        except (ValueError, KeyError):
            pass

    # Group annotations by frame number
    frame_to_annots: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for ann in coco.get('annotations', []):
        frame_num = id_to_frame.get(ann.get('image_id'))
        if frame_num is not None:
            frame_to_annots[frame_num].append(ann)

    return dict(frame_to_annots)


def _get_coco(coco_path: str) -> Dict[int, List[Dict[str, Any]]]:
    """Return cached COCO data (frame_num -> annotations list)."""
    if coco_path in _PROC_COCO_CACHE:
        return _PROC_COCO_CACHE[coco_path]

    result = _parse_coco_file(coco_path)

    if len(_PROC_COCO_CACHE) >= _PROC_COCO_CACHE_MAX:
        _PROC_COCO_CACHE.pop(next(iter(_PROC_COCO_CACHE)))
    _PROC_COCO_CACHE[coco_path] = result
    return result


def _get_coco_img_dims(coco_path: str, frame_num: int) -> Tuple[int, int]:
    """Return cached COCO image dimensions for a given frame.

    Parses the COCO JSON once per recording and caches all image dimensions
    (width, height) keyed by frame_num (derived from file_name stem).
    Returns (1280, 720) as fallback if lookup fails.
    """
    if coco_path in _PROC_COCO_DIMS_CACHE and frame_num in _PROC_COCO_DIMS_CACHE[coco_path]:
        return _PROC_COCO_DIMS_CACHE[coco_path][frame_num]

    # Parse image dimensions from COCO JSON (first access for this recording)
    try:
        raw_coco = json.loads(Path(coco_path).read_text())
    except Exception:
        _PROC_COCO_DIMS_CACHE.setdefault(coco_path, {})[frame_num] = (1280, 720)
        return (1280, 720)

    dims: Dict[int, Tuple[int, int]] = {}
    for img in raw_coco.get('images', []):
        fname = str(img.get('file_name', ''))
        stem = Path(fname).stem
        try:
            fid = int(stem)
        except (ValueError, TypeError):
            continue
        w = int(img.get('width', 1280))
        h = int(img.get('height', 720))
        dims[fid] = (w, h)

    _PROC_COCO_DIMS_CACHE[coco_path] = dims
    return dims.get(frame_num, (1280, 720))


# =========================================================================
# Per-Recording Annotation Cache
# =========================================================================


class _PerRecordingCache:
    """
    Thread-safe per-recording annotation cache.
    Loads AR, PSR, pose, hands data once per recording and provides
    per-frame lookup via interpolation.
    """

    def __init__(self, rec_dir: Path):
        self.rec_dir = rec_dir
        self.recording_id = rec_dir.name

        # Sparse AR labels → per-frame array
        self._ar_per_frame: Optional[np.ndarray] = None

        # Sparse PSR raw → per-frame dense [num_frames, 11]
        self._psr_per_frame: Optional[np.ndarray] = None

        # Dense pose.csv → [num_frames, 9]
        self._pose: Optional[np.ndarray] = None

        # Dense hands.csv → [num_frames, 52]
        self._hands: Optional[np.ndarray] = None

        self._loaded = False

    def _parse_ar_labels(self) -> np.ndarray:
        """
        Parse AR_labels.csv and interpolate to per-frame labels.
        Format: recording_id,action_class_id,action_description,start_frame.jpg,end_frame.jpg
        Returns: np.ndarray of shape [num_frames] with raw action IDs.
        Frames without any AR coverage get -1 (sentinel for "unlabeled").
        """
        ar_file = self.rec_dir / 'AR_labels.csv'
        if not ar_file.exists():
            return np.full(self._num_frames, -1, dtype=np.int64)

        # Load all action spans
        spans: List[Tuple[int, int, int]] = []  # (start_frame, end_frame, action_id)
        with open(ar_file, encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 5:
                    continue
                try:
                    action_id = int(row[1])
                    start_frame = int(Path(row[3]).stem)
                    end_frame = int(Path(row[4]).stem)
                    spans.append((start_frame, end_frame, action_id))
                except (ValueError, IndexError):
                    continue

        # Sort by start frame
        spans.sort(key=lambda x: x[0])

        # Initialize to -1 = unlabeled (sentinel); frames covered by action spans
        # get their raw action_id (including 0 for "take_short_brace").
        labels = np.full(self._num_frames, -1, dtype=np.int64)
        for start, end, action_id in spans:
            end = min(end, self._num_frames - 1)
            if start < self._num_frames:
                # [FIX 2026-06-14] Clamp action_id to valid range [0, NUM_CLASSES_ACT-1]
                # to prevent ScatterGatherKernel OOB in loss functions during validation.
                if action_id < 0 or action_id >= C.NUM_CLASSES_ACT:
                    logger.warning(
                        f'[industreal_dataset] Clamping OOB action_id={action_id} '
                        f'(valid 0..{C.NUM_CLASSES_ACT - 1}) in {self.rec_dir.name} '
                        f'frames [{start}:{end}]'
                    )
                    action_id = max(0, min(action_id, C.NUM_CLASSES_ACT - 1))
                labels[start: end + 1] = action_id

        return labels

    def _parse_ar_segments(self) -> list:
        """[GAP-B] Return action segments from AR_labels.csv without frame interpolation.
        Each segment = (start_frame, end_frame, action_id). NA (action_id=0) excluded.
        Returns: list of (start, end, action_id) tuples.
        """
        ar_file = self.rec_dir / 'AR_labels.csv'
        if not ar_file.exists():
            return []
        segments = []
        with open(ar_file, encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 5: continue
                try:
                    action_id = int(row[1])
                    if action_id == 0: continue  # NA excluded from metric clips
                    start = int(Path(row[3]).stem)
                    end = int(Path(row[4]).stem)
                    end = min(end, self._num_frames - 1)
                    if start < self._num_frames and end > start:
                        segments.append((start, end, action_id))
                except (ValueError, IndexError): continue
        return segments

    def _parse_psr_raw(self) -> np.ndarray:
        """
        Parse PSR_labels_raw.csv and fill forward to get per-frame dense labels.
        Format: frame.jpg,comp0,comp1,...,comp10  (values: -1=error, 0=not_done, 1=done)
        Sparse: only rows where state changes are recorded.
        Fill forward: once a component becomes 1 (or -1), it stays that way.
        Initial state (before first change): all zeros.
        """
        psr_file = self.rec_dir / 'PSR_labels_raw.csv'
        if not psr_file.exists():
            return np.zeros((self._num_frames, C.NUM_PSR_COMPONENTS), dtype=np.float32)

        # Load sparse rows: (frame_num, [11 component values])
        sparse: List[Tuple[int, np.ndarray]] = []
        with open(psr_file, encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 12:
                    continue
                try:
                    frame_num = int(Path(row[0]).stem)
                    values = np.array([float(v) for v in row[1:12]], dtype=np.float32)
                    sparse.append((frame_num, values))
                except (ValueError, IndexError):
                    continue

        if not sparse:
            return np.zeros((self._num_frames, C.NUM_PSR_COMPONENTS), dtype=np.float32)

        sparse.sort(key=lambda x: x[0])

        # Fill forward: start with zeros, then apply each change
        dense = np.zeros((self._num_frames, C.NUM_PSR_COMPONENTS), dtype=np.float32)
        current = np.zeros(C.NUM_PSR_COMPONENTS, dtype=np.float32)

        # [OPUS v5 AUDIT] -1 transient fix: -1 is an error component — do NOT carry it
        # forward. It over-counts ignores if propagated to all later frames. Instead,
        # keep the last valid value (not -1) when encountering -1 in sparse data.
        _last_valid = np.zeros(C.NUM_PSR_COMPONENTS, dtype=np.int64)  # defaults: all absent
        sparse_idx = 0
        for frame in range(self._num_frames):
            if sparse_idx < len(sparse) and frame == sparse[sparse_idx][0]:
                _new = sparse[sparse_idx][1].copy()
                sparse_idx += 1
                # Only update components that are NOT -1 (error); keep last valid
                _valid_mask = _new >= 0
                _last_valid[_valid_mask] = _new[_valid_mask]
            dense[frame] = _last_valid.copy()

        return dense

    def _parse_pose(self) -> np.ndarray:
        """
        Parse pose.csv — dense 9-DoF head pose per frame.
        Format: frame.jpg,forward_x,forward_y,forward_z,position_x,position_y,position_z,up_x,up_y,up_z
        Returns: np.ndarray of shape [num_frames, 9]
        """
        pose_file = self.rec_dir / 'pose.csv'
        if not pose_file.exists():
            return np.zeros((self._num_frames, C.NUM_HEAD_POSE_DOF), dtype=np.float32)

        pose_data = np.zeros((self._num_frames, C.NUM_HEAD_POSE_DOF), dtype=np.float32)

        with open(pose_file, encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 10:
                    continue
                try:
                    frame_num = int(Path(row[0]).stem)
                    values = [float(v) for v in row[1:10]]
                    pose_data[frame_num] = values
                except (ValueError, IndexError):
                    continue

        # Bug F fix + unit fix: standardize position to O(1) so loss gradient is
        # balanced across forward (unit) and position (~110 raw) channels. Also
        # resolves mm/cm ambiguity: HEAD_POSE_POS_SCALE=100 treats CSV values as
        # ~cm-scale (|pos|max~110/100≈1.1m plausible for head displacement).
        if C.HEAD_POSE_POS_SCALE != 0.0:
            pose_data[:, 3:6] /= C.HEAD_POSE_POS_SCALE

        # [FIX 2026-07-05 Opus 126 Decision 7] Unit-normalize forward and up vectors.
        # HoloLens pose.csv records 9-D vectors that are NEAR-unit but not exact
        # (mean norm observed ~1.0 with 5-10% drift per Opus 121 §23.2). The pose
        # head's loss assumes unit length; non-unit inputs silently corrupt the
        # forward-MAE and pose-POS comparisons. Normalize on load.
        if pose_data.shape[0] > 0:
            fwd_norms = np.linalg.norm(pose_data[:, 0:3], axis=1, keepdims=True)
            up_norms = np.linalg.norm(pose_data[:, 6:9], axis=1, keepdims=True)
            # Avoid div-by-zero: only normalize rows with non-zero norm
            fwd_safe = np.where(fwd_norms > 1e-6, fwd_norms, 1.0)
            up_safe = np.where(up_norms > 1e-6, up_norms, 1.0)
            pose_data[:, 0:3] = pose_data[:, 0:3] / fwd_safe
            pose_data[:, 6:9] = pose_data[:, 6:9] / up_safe

        # Bug F fix — sanity-check pose schema at load time. The eval pipeline
        # multiplies position residuals by 1000 (assumes metres in pose.csv); a
        # change in CSV units would silently corrupt position_MAE_mm.
        if pose_data.shape[0] > 0:
            fwd = pose_data[:, 0:3]
            pos = pose_data[:, 3:6]
            fwd_norm = float(np.linalg.norm(fwd, axis=1).mean()) if np.any(fwd) else 0.0
            pos_max = float(np.abs(pos).max()) if np.any(pos) else 0.0
            if fwd_norm > 0.0 and not (0.5 < fwd_norm < 1.5):
                logger.warning(
                    f'[_parse_pose {self.rec_dir.name}] forward vector mean norm '
                    f'{fwd_norm:.3f} is not ~1; check that pose.csv columns 1-3 are unit vectors.'
                )
            if pos_max > 5.0:
                logger.warning(
                    f'[_parse_pose {self.rec_dir.name}] position max abs {pos_max:.2f} > 5; '
                    f'evaluate.py:position_MAE_mm assumes metres and applies *1000.'
                )

        return pose_data

    def _parse_hands(self) -> np.ndarray:
        """
        Parse hands.csv — dense 52-D hand joints per frame.
        Format: frame.jpg, left_hand_26_joints*2, right_hand_26_joints*2
        Returns: np.ndarray of shape [num_frames, 52]
        """
        hands_file = self.rec_dir / 'hands.csv'
        if not hands_file.exists():
            return np.zeros((self._num_frames, 52), dtype=np.float32)

        hands_data = np.zeros((self._num_frames, 52), dtype=np.float32)

        with open(hands_file, encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 53:
                    continue
                try:
                    frame_num = int(Path(row[0]).stem)
                    # Left hand (26*2=52) + right hand (26*2=52) = 104 coords
                    # We take first 52 (left hand only, as per contract requirement)
                    values = [float(v) for v in row[1:53]]
                    hands_data[frame_num] = values
                except (ValueError, IndexError):
                    continue

        return hands_data

    def load(self, num_frames: int) -> None:
        """Load all annotations for this recording."""
        if self._loaded:
            return
        self._num_frames = num_frames
        self._ar_per_frame = self._parse_ar_labels()
        self._psr_per_frame = self._parse_psr_raw()
        self._pose = self._parse_pose()
        self._hands = self._parse_hands()
        self._loaded = True

    @property
    def ar_per_frame(self) -> np.ndarray:
        assert self._ar_per_frame is not None, 'Call load() first'
        return self._ar_per_frame

    @property
    def psr_per_frame(self) -> np.ndarray:
        assert self._psr_per_frame is not None, 'Call load() first'
        return self._psr_per_frame

    @property
    def pose(self) -> np.ndarray:
        assert self._pose is not None, 'Call load() first'
        return self._pose

    @property
    def hands(self) -> np.ndarray:
        assert self._hands is not None, 'Call load() first'
        return self._hands


# =========================================================================
# Dataset
# =========================================================================


class IndustRealMultiTaskDataset(Dataset):
    """
    Multi-task IndustReal dataset loader.

    Supports:
      - Action Recognition (AR): per-frame action class (0 = NA/background)
      - Assembly State Detection (ASD): per-frame bounding boxes (COCO format)
      - Procedure Step Recognition (PSR): per-frame 11-D binary component state
      - Head Pose: per-frame 9-DoF head pose
      - Hand Joints: per-frame 52-D hand joint coordinates

    Single RGB camera at 1280x720, 10 FPS.
    """

    def __init__(
        self,
        split: str = 'train',
        img_size: Tuple[int, int] = C.IMG_SIZE,
        augment: bool = False,
        seed: int = C.SEED,
        max_recordings: Optional[int] = None,
        subset_ratio: float = 1.0,
        sequence_mode: bool = False,
        sequence_length: int = 32,
    ):
        """
        Args:
            split: 'train', 'val', or 'test'
            img_size: (width, height) to resize images to
            augment: apply data augmentation (color jitter, flip)
            seed: random seed for reproducibility
            max_recordings: cap number of recordings (for debugging)
            subset_ratio: fraction of recordings to use (Item 29) — applied via
                max_recordings when < 1.0; resolved in training script before
                constructing dataset; kept here for compatibility
            sequence_mode: if True, return T-frame windows for PSR sequence training (Doc 01 §D.1)
            sequence_length: number of consecutive frames per sequence sample (default 32)
        """
        self.split = split
        self.img_size = img_size
        self.augment = augment
        self.seed = seed
        self.max_recordings = max_recordings
        self.subset_ratio = subset_ratio
        self.sequence_mode = sequence_mode
        self.sequence_length = sequence_length

        # Paths
        self.recordings_root = C.RECORDINGS_ROOT
        self.split_csv = {
            'train': C.TRAIN_CSV,
            'val': C.VAL_CSV,
            'test': C.TEST_CSV,
        }[split]

        # Per-recording annotation cache
        self._anno_cache: Dict[str, _PerRecordingCache] = {}

        # Per-recording sorted frame paths for VideoMAE clip loading (Doc 2 A.1)
        self._rec_frame_paths: Dict[str, List[int]] = {}

        # Load clip for VideoMAE only when enabled
        self._load_clip = bool(getattr(C, 'USE_VIDEOMAE', False))
        self._clip_num_frames = getattr(C, 'VIDEOMAE_NUM_FRAMES', 16)
        self._clip_stride = getattr(C, 'VIDEOMAE_SAMPLE_STRIDE', 1)

        # Scan recordings and build sample index
        self.samples = self._scan_and_index()

        # [RAM CACHE 2026-06-28 v3] Cache JPEG bytes (compressed, ~350 KB avg per image)
        # instead of decoded tensors (~2.8 MB). Capped at C.RAM_CACHE_MAX_IMAGES (default 5000)
        # with LRU eviction. ~5,000 images × 350 KB ≈ 1.8 GB — fits in available RAM.
        # Validation/test: cached too but with a tighter cap (min(max_cache, 2000)) since
        # val sets are smaller and the bottleneck there is lower.
        self._ram_cache: Dict[str, bytes] = {}
        _max_cache = getattr(C, 'RAM_CACHE_MAX_IMAGES', 5000)
        if _max_cache > 0:
            if self.split != 'train':
                _max_cache = min(_max_cache, 2000)
            n = min(len(self.samples), _max_cache)
            logger.info(f'[RAM_CACHE] Pre-loading {n} {self.split} images as JPEG bytes (cap={_max_cache})...')
            t0 = time_module.time()
            from collections import OrderedDict
            self._ram_cache = OrderedDict()
            for i, s in enumerate(self.samples):
                if len(self._ram_cache) >= _max_cache:
                    logger.info(f'[RAM_CACHE] Cap reached ({_max_cache}) — stopping cache fill')
                    break
                if i % 1000 == 0 and i > 0:
                    elapsed = time_module.time() - t0
                    logger.info(f'[RAM_CACHE] {i}/{n} scanned — {len(self._ram_cache)} cached, {elapsed:.0f}s')
                try:
                    with open(s['img_path'], 'rb') as _f:
                        self._ram_cache[s['img_path']] = _f.read()
                except Exception:
                    pass  # skip unreadable — fallback to disk
            total = time_module.time() - t0
            mb = len(self._ram_cache) * 350 / 1024
            logger.info(f'[RAM_CACHE] Done — {len(self._ram_cache)} images cached in {total:.0f}s (~{mb:.1f}MB estimated)')
        else:
            logger.info(f'[RAM_CACHE] Skipped — cap=0')

        # Activity IDs for class-balanced sampling.
        # [Route A — file 75] When ACT_CLASS_GROUPING='verb', remap raw action_id
        # to its verb-group index so the sampler/counts operate in group space.
        # Identity when grouping is off.
        _raw_ids = np.array(
            [s['action_label'] for s in self.samples], dtype=np.int64
        )
        _remap = getattr(C, 'remap_activity_label', None)
        _mode = str(getattr(C, 'ACT_CLASS_GROUPING', 'none')).lower()
        if _remap is not None and _mode in ('verb', 'hybrid'):
            self.activity_ids = np.array([_remap(int(a)) for a in _raw_ids], dtype=np.int64)
        else:
            self.activity_ids = _raw_ids
        valid_ids = self.activity_ids[self.activity_ids >= 0]  # exclude -1 sentinel
        self.class_counts = np.bincount(
            valid_ids, minlength=int(getattr(C, 'NUM_ACT_OUTPUTS', C.NUM_ACT_CLASSES))
        )

        # Doc 01 §D.1: Build sequence sample index for PSR sequence training
        self._seq_samples: List[Dict[str, Any]] = []
        if self.sequence_mode:
            self._seq_samples = self._build_seq_sample_index()
            logger.info(
                f'[industreal_dataset] Sequence mode: {len(self._seq_samples)} windows '
                f'(T={self.sequence_length}, stride=1) across {len(self._rec_frame_paths)} recordings'
            )

        logger.info(
            f'[industreal_dataset] Loaded {len(self.samples)} frames '
            f'(split={split}, recordings={max_recordings or "all"})'
        )

    def clear_coco_cache(self) -> None:
        _PROC_COCO_CACHE.clear()

    @property
    def psr_prevalence(self) -> np.ndarray:
        """
        Per-component PSR prevalence (fraction of frames where each component = 1).
        Used to compute per-component focal loss alpha: alpha_c = 2 * (1 - prevalence_c).
        Lazily computes from the per-recording annotation caches.
        Returns: [11] array of prevalence values in [0, 1].
        """
        if getattr(self, '_psr_prevalence_cache', None) is not None:
            return self._psr_prevalence_cache

        total_frames = 0
        component_sums = np.zeros(C.NUM_PSR_COMPONENTS, dtype=np.float64)

        for cache in self._anno_cache.values():
            psr = cache.psr_per_frame  # [num_frames_rec, 11]
            component_sums += psr.sum(axis=0)
            total_frames += psr.shape[0]

        self._psr_prevalence_cache = (component_sums / max(total_frames, 1)).astype(np.float32)
        return self._psr_prevalence_cache

    # [GAP-B] Action-segment clip protocol (MViTv2-comparable activity eval)
    def build_activity_segments(self):
        """Returns [(rec_id, start, end, action_id), ...] from AR spans; NA excluded."""
        segs = []
        for rec_dir in sorted((self.recordings_root / self.split).iterdir()):
            if not rec_dir.is_dir(): continue
            rec_id = rec_dir.name
            cache = self._anno_cache.get(rec_id)
            if cache is None:
                cache = _PerRecordingCache(rec_dir); cache.load(99999)
                self._anno_cache[rec_id] = cache
            for start, end, aid in cache._parse_ar_segments():
                segs.append((rec_id, int(start), int(end), int(aid)))
        return segs

    def sample_segment_clip(self, seg, T=16):
        """Sample T uniform frames from a segment [start, end].
        Returns: frames [T,3,H,W], action_id (int, never NA).
        """
        rec_id, start, end, aid = seg
        idxs = np.linspace(start, end, T).round().astype(int)
        frames = []
        for fn in idxs:
            img_path = self.recordings_root / self.split / rec_id / 'rgb' / f'{fn:06d}.jpg'
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize((self.img_size[0], self.img_size[1]), _BILINEAR)
                arr = np.array(img, dtype=np.float32) / 255.0
                arr = (arr - np.array([0.485,0.456,0.406])) / np.array([0.229,0.224,0.225])
                frames.append(torch.from_numpy(arr).permute(2,0,1))
            except Exception:
                frames.append(torch.zeros(3, self.img_size[1], self.img_size[0]))
        if not frames:
            return torch.zeros(T, 3, self.img_size[1], self.img_size[0]), 0
        return torch.stack(frames), aid

    def __len__(self) -> int:
        if self.sequence_mode:
            return len(self._seq_samples)
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.sequence_mode:
            return self._getitem_sequence(idx)

        sample = self.samples[idx]
        recording_id = sample['recording_id']
        frame_num = sample['frame_num']
        img_path = sample['img_path']

        # Load and resize RGB image
        rgb_tensor = self._load_image(img_path)

        # Get annotation cache
        cache = self._anno_cache[recording_id]

        # AR action label. [Route A — file 75] Remap raw action_id to verb-group
        # index when ACT_CLASS_GROUPING='verb' (identity otherwise). Preserves -1.
        _raw_al = int(sample['action_label'])
        _remap = getattr(C, 'remap_activity_label', None)
        if _remap is not None and str(getattr(C, 'ACT_CLASS_GROUPING', 'none')).lower() in ('verb', 'hybrid'):
            _raw_al = _remap(_raw_al)
        action_label = torch.tensor(_raw_al, dtype=torch.long)

        # PSR per-frame labels
        psr_labels = torch.from_numpy(
            cache.psr_per_frame[frame_num]
        ).float()

        # Head pose
        head_pose = torch.from_numpy(
            cache.pose[frame_num]
        ).float()

        # Hand joints (52-D)
        hand_joints = torch.from_numpy(
            cache.hands[frame_num]
        ).float()

        # Detection boxes from COCO
        gt_boxes, gt_classes = self._extract_boxes_from_coco(
            recording_id, frame_num
        )

        # Apply spatial augmentation during training
        if self.augment and getattr(C, 'USE_SPATIAL_AUG', False):
            rgb_tensor, gt_boxes, _ = apply_spatial_aug(
                rgb_tensor,
                gt_boxes,
                torch.zeros((17, 2), dtype=torch.float32),  # No keypoints in IndustReal
            )

        # VideoMAE clip: [T, 3, 224, 224] normalized (Doc 2 A.1)
        # When VideoMAE is disabled, return empty tensor (not None) so default collate works
        clip_rgb: torch.Tensor = torch.zeros(0, 3, 224, 224, dtype=torch.float32)
        if self._load_clip:
            clip_rgb = self._load_clip_frames(recording_id, frame_num)

        return {
            'images': {'rgb': rgb_tensor},
            'gt_boxes': {'rgb': gt_boxes},
            'gt_classes': {'rgb': gt_classes},
            'head_pose': head_pose,
            'psr_labels': psr_labels,
            'hand_joints': hand_joints,
            'action_label': action_label,
            'activity': action_label,  # Alias for IKEA loss compatibility
            'detection': {'boxes': gt_boxes, 'labels': gt_classes},  # Flat detection dict
            'clip_rgb': clip_rgb,
            'metadata': {
                'recording_id': recording_id,
                'frame_num': frame_num,
            }
        }

    def _getitem_sequence(self, idx: int) -> Dict[str, Any]:
        """
        Doc 01 §D.1: Sequence-mode __getitem__ for PSR training.

        Returns a T-frame window from one recording with all per-frame annotations.
        The causal transformer in PSRHead can now see temporal context during training.

        Returns:
            Dict with:
                - images['rgb']: [T, 3, H, W] stacked RGB frames
                - psr_labels: [T, 11] PSR labels for all T frames
                - head_pose: [T, 9] head poses
                - hand_joints: [T, 52] hand joints
                - metadata: {recording_id, frame_nums: list of T frame indices}
                - action_label: [T] action labels (most common in window, for AR task)
                - gt_boxes/gt_classes: only for middle frame (detection task)
        """
        seq_sample = self._seq_samples[idx]
        recording_id = seq_sample['recording_id']
        frame_nums = seq_sample['frames']
        T = len(frame_nums)

        cache = self._anno_cache[recording_id]
        rgb_dir = self.recordings_root / self.split / recording_id / 'rgb'

        # Load all T frames as [T, 3, H, W]
        frames_list: List[torch.Tensor] = []
        for fn in frame_nums:
            img_path = rgb_dir / f'{fn:06d}.jpg'
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize((self.img_size[0], self.img_size[1]), _BILINEAR)
                img = np.array(img, dtype=np.uint8)
            except Exception as e:
                logger.warning(f'Failed to load {img_path}: {e}. Using blank.')
                img = np.zeros((self.img_size[1], self.img_size[0], 3), dtype=np.uint8)
            frames_list.append(torch.from_numpy(img).permute(2, 0, 1))

        rgb_tensor = torch.stack(frames_list, dim=0)  # [T, 3, H, W]

        # PSR labels for all T frames: [T, 11]
        psr_labels = torch.from_numpy(
            cache.psr_per_frame[frame_nums]
        ).float()

        # Head poses for all T frames: [T, 9]
        head_pose = torch.from_numpy(
            cache.pose[frame_nums]
        ).float()

        # Hand joints for all T frames: [T, 52]
        hand_joints = torch.from_numpy(
            cache.hands[frame_nums]
        ).float()

        # AR: majority vote action label in window (most common non-NA)
        action_labels_per_frame = cache.ar_per_frame[frame_nums]
        unique, counts = np.unique(action_labels_per_frame, return_counts=True)
        most_common_action = int(unique[np.argmax(counts)])
        # [Route A — file 75] Remap to verb-group index (identity when off). Preserves -1.
        _remap = getattr(C, 'remap_activity_label', None)
        if _remap is not None and str(getattr(C, 'ACT_CLASS_GROUPING', 'none')).lower() in ('verb', 'hybrid'):
            most_common_action = _remap(most_common_action)

        # Detection: only for middle frame (for ASD task alignment)
        mid_frame = frame_nums[T // 2]
        gt_boxes, gt_classes = self._extract_boxes_from_coco(recording_id, mid_frame)

        return {
            'images': {'rgb': rgb_tensor},
            'gt_boxes': {'rgb': gt_boxes},
            'gt_classes': {'rgb': gt_classes},
            'head_pose': head_pose,
            'psr_labels': psr_labels,
            'hand_joints': hand_joints,
            'action_label': torch.tensor(most_common_action, dtype=torch.long),
            'clip_rgb': None,
            'metadata': {
                'recording_id': recording_id,
                'frame_nums': frame_nums,
                'sequence_length': T,
            }
        }

    # =====================================================================
    # Image Loading
    # =====================================================================

    def _load_image_raw(self, img_path: str) -> torch.Tensor:
        """Load and resize RGB image from disk to [3, H, W]."""
        try:
            img = Image.open(img_path).convert('RGB')
            img = img.resize((self.img_size[0], self.img_size[1]), _BILINEAR)
            img = np.array(img, dtype=np.uint8)
        except Exception as e:
            logger.warning(f'Failed to load {img_path}: {e}. Using blank image.')
            img = np.zeros((self.img_size[1], self.img_size[0], 3), dtype=np.uint8)

        return torch.from_numpy(img).permute(2, 0, 1)  # (3, H, W)

    def _load_image(self, img_path: str) -> torch.Tensor:
        """Load image from RAM cache (JPEG bytes) or disk. Decode bytes ~2ms vs HDD ~50ms."""
        cached = self._ram_cache.get(img_path) if self._ram_cache else None
        if cached is not None:
            try:
                from io import BytesIO
                img = Image.open(BytesIO(cached)).convert('RGB')
                img = img.resize((self.img_size[0], self.img_size[1]), _BILINEAR)
                return torch.from_numpy(np.array(img, dtype=np.uint8)).permute(2, 0, 1)
            except Exception:
                pass  # fall through to disk load
        return self._load_image_raw(img_path)

    def _load_clip_frames(
        self, recording_id: str, target_frame: int
    ) -> torch.Tensor:
        """
        Load a temporal clip of T frames for VideoMAE using random temporal stride.

        Doc 2 D.3: Random temporal stride — sample T frames at a random stride
        (1, 2, or 3) rather than always consecutive. This improves temporal
        robustness and is used in VideoMAE-style self-supervised learning.

        Returns:
            clip: [T, 3, 224, 224] torch tensor (normalized with IMAGENET_MEAN/STD)
        """
        frame_nums = self._rec_frame_paths.get(recording_id, [])
        if not frame_nums:
            return torch.zeros(
                self._clip_num_frames, 3, 224, 224, dtype=torch.float32
            )

        T = self._clip_num_frames

        # Random temporal stride: sample every N frames (Doc 2 D.3)
        if self.augment:
            stride = random.choice([1, 2, 3])
        else:
            stride = 1

        fn_idx = frame_nums.index(target_frame) if target_frame in frame_nums else -1
        if fn_idx == -1:
            fn_idx = 0

        # Compute valid start index for the clip window
        max_start = max(0, fn_idx - stride * (T - 1))
        min_start = min(fn_idx, max_start)
        start_idx = random.randint(min_start, max(fn_idx, min_start)) if self.augment else min_start

        # Build clip indices at the chosen stride
        clip_indices: List[int] = []
        for t in range(T):
            idx = start_idx + t * stride
            idx = max(0, min(idx, len(frame_nums) - 1))
            clip_indices.append(frame_nums[idx])

        # Load and resize frames to 224×224
        rgb_dir = self.recordings_root / self.split / recording_id / 'rgb'
        frames: List[torch.Tensor] = []
        for fn in clip_indices:
            img_path = rgb_dir / f'{fn:06d}.jpg'
            try:
                img = Image.open(str(img_path)).convert('RGB')
                img = img.resize((224, 224), _BILINEAR)
                arr = np.array(img, dtype=np.float32)
            except Exception:
                arr = np.zeros((224, 224, 3), dtype=np.float32)
            frame = torch.from_numpy(arr).permute(2, 0, 1)  # [3, 224, 224]
            frames.append(frame)

        clip = torch.stack(frames, dim=0)  # [T, 3, 224, 224]

        # Normalize
        mean = torch.tensor(C.IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(C.IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
        clip = (clip / 255.0 - mean) / std

        return clip.float()

    # =====================================================================
    # Detection Annotation Loading
    # =====================================================================

    def _get_coco_path(self, recording_id: str) -> str:
        """Construct path to COCO JSON for a recording."""
        return str(self.recordings_root / self.split / recording_id / 'OD_labels.json')

    def _extract_boxes_from_coco(
        self, recording_id: str, frame_num: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract bboxes and classes from COCO JSON for a single recording/frame.
        Returns: (boxes [N, 4] xyxy, classes [N])
        """
        coco_file = self._get_coco_path(recording_id)

        if not Path(coco_file).exists():
            return (
                torch.zeros((0, 4), dtype=torch.float32),
                torch.zeros(0, dtype=torch.long),
            )

        try:
            frame_to_annots = _get_coco(coco_file)
            annots = frame_to_annots.get(frame_num, [])
        except Exception:
            return (
                torch.zeros((0, 4), dtype=torch.float32),
                torch.zeros(0, dtype=torch.long),
            )

        if not annots:
            return (
                torch.zeros((0, 4), dtype=torch.float32),
                torch.zeros(0, dtype=torch.long),
            )

        # [GAP Part 4.1] Rescale boxes to match IMG_SIZE. COCO annotations are in
        # native image coordinates (1280×720); _load_image resizes to self.img_size.
        # Currently self.img_size == native (1280,720), so scale is identity — but
        # this guard ensures correctness if IMG_SIZE changes.
        # [FIX D5] Use cached COCO image dimensions instead of re-parsing the full
        # JSON per frame (~40 min/epoch overhead without cache).
        _img_w, _img_h = _get_coco_img_dims(coco_file, frame_num)
        _sx = self.img_size[0] / max(_img_w, 1)
        _sy = self.img_size[1] / max(_img_h, 1)

        boxes = []
        classes = []
        for ann in annots:
            bbox = ann.get('bbox', [])
            if len(bbox) == 4:
                x, y, w, h = bbox
                # [GAP 4.1] Rescale to IMG_SIZE
                boxes.append([x*_sx, y*_sy, (x+w)*_sx, (y+h)*_sy])
                # [OPUS FIX #5] COCO category_id is 1-24; head outputs 0-23.
                # Subtract 1 to convert to 0-indexed (matches head output).
                raw_cat = ann.get('category_id', 0)
                idx = raw_cat - 1
                # Safety guard: clamp to valid range [0, NUM_DET_CLASSES-1]
                if idx < 0 or idx >=24:
                    idx = 0  # map out-of-range to background on any anomaly
                classes.append(idx)

        if not boxes:
            return (
                torch.zeros((0, 4), dtype=torch.float32),
                torch.zeros(0, dtype=torch.long),
            )

        return (
            torch.from_numpy(np.array(boxes, dtype=np.float32)),
            torch.from_numpy(np.array(classes, dtype=np.int64)),
        )

    # =====================================================================
    # Dataset Indexing
    # =====================================================================

    def _load_split_recordings(self) -> set:
        """Load recording IDs for this split from the CSV file."""
        if not self.split_csv.exists():
            logger.warning(f'Split CSV not found: {self.split_csv}')
            return set()

        recording_ids = set()
        with open(self.split_csv, encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 1:
                    recording_ids.add(row[0].strip())

        return recording_ids

    def _build_seq_sample_index(self) -> List[Dict[str, Any]]:
        """
        Doc 01 §D.1: Build sequence sample index for PSR sequence training.

        Each sequence sample represents a T-frame window from one recording.
        Windows start at every frame (stride=1) within each recording.

        Returns:
            list of sequence sample dicts: {recording_id, start_frame, frames}
            where frames = list of frame indices for this window
        """
        seq_samples: List[Dict[str, Any]] = []
        T = self.sequence_length

        for recording_id, frame_nums in self._rec_frame_paths.items():
            num_frames = len(frame_nums)
            if num_frames < T:
                logger.debug(
                    f'  [_build_seq_seq] Skipping {recording_id}: only {num_frames} frames < T={T}'
                )
                continue

            for start_idx in range(num_frames - T + 1):
                window_frames = frame_nums[start_idx : start_idx + T]
                seq_samples.append({
                    'recording_id': recording_id,
                    'start_frame_idx': start_idx,
                    'frames': window_frames,
                })

        return seq_samples

    def _scan_and_index(self) -> List[Dict[str, Any]]:
        """
        Scan recordings for this split and build sample index.

        For each recording in the split:
          1. List rgb/ frames to determine num_frames
          2. Load per-recording annotations (AR, PSR, pose, hands)
          3. Build per-frame sample records
          4. Apply frame stride

        Returns:
            list of sample dicts: {recording_id, frame_num, img_path, action_label}
        """
        split_rec_ids = self._load_split_recordings()
        if not split_rec_ids:
            logger.warning(f'[industreal_dataset] No recordings found in split: {self.split}')
            return []

        all_samples: List[Dict[str, Any]] = []

        recordings_root = self.recordings_root / self.split

        # Collect candidate recording dirs
        _candidates = []
        for rec_dir in recordings_root.iterdir():
            if not rec_dir.is_dir():
                continue
            _rid = rec_dir.name
            if _rid not in split_rec_ids:
                continue
            _candidates.append((_rid, rec_dir))

        # [OPUS v5 AUDIT] Greedy coverage stratification when subset_ratio < 1.0.
        # Alphabetical subset can exclude entire action classes → confounded activity metrics.
        # Greedily pick recordings that maximize unique AR class coverage.
        _max_recs = self.max_recordings
        if _max_recs and len(_candidates) > _max_recs:
            # Pre-scan: map recording_id → set of AR action classes present
            _rec_ar_classes: dict = {}
            for _rid, _rdir in _candidates:
                _ar_file = _rdir / 'AR_labels.csv'
                _cls_set = set()
                if _ar_file.exists():
                    import csv as _csv
                    with open(_ar_file, encoding='utf-8') as _f:
                        for _row in _csv.reader(_f):
                            if len(_row) >= 2:
                                try:
                                    _cls_set.add(int(_row[1]))
                                except ValueError:
                                    pass
                _rec_ar_classes[_rid] = _cls_set

            # Greedy coverage: iteratively pick recording that adds most new classes
            _chosen = []
            _covered = set()
            _remaining = list(_candidates)
            for _ in range(_max_recs):
                _best, _best_new = None, 0
                for _rid, _rdir in _remaining:
                    _new = len(_rec_ar_classes.get(_rid, set()) - _covered)
                    if _new > _best_new:
                        _best, _best_new = (_rid, _rdir), _new
                if _best is None or _best_new == 0:
                    break
                _chosen.append(_best)
                _covered |= _rec_ar_classes.get(_best[0], set())
                _remaining.remove(_best)
            # Append remaining recordings (in alphabetical order) if we didn't fill
            _chosen += _remaining
            _candidates[:] = _chosen[: _max_recs]

        # Build ordered list: if greedy coverage was applied, use that order; otherwise alphabetical
        _ordered = _candidates if (_max_recs and len(_candidates) <= _max_recs) else _candidates
        rec_count = 0
        for recording_id, rec_dir in _ordered:

            rgb_dir = rec_dir / 'rgb'
            if not rgb_dir.exists():
                continue

            # Get frame numbers from rgb/ directory
            frame_nums: List[int] = []
            for entry in rgb_dir.iterdir():
                name = entry.name
                if name.endswith('.jpg'):
                    stem = name[:-4]
                    if stem.isdigit():
                        frame_nums.append(int(stem))

            if not frame_nums:
                continue

            frame_nums = sorted(frame_nums)
            num_frames = frame_nums[-1] + 1  # frame numbering starts at 0

            # Load per-recording annotations
            cache = _PerRecordingCache(rec_dir)
            cache.load(num_frames)
            self._anno_cache[recording_id] = cache

            # Store sorted frame list for clip loading (Doc 2 A.1)
            self._rec_frame_paths[recording_id] = frame_nums

            # Pre-load COCO annotations for task-aware sampling metadata
            coco_data = _get_coco(self._get_coco_path(recording_id))

            # Build per-frame sample records
            stride = (
                C.TRAIN_FRAME_STRIDE
                if self.split == 'train' and not C.DEBUG_MODE
                else C.EVAL_FRAME_STRIDE
            )

            for fn in frame_nums:
                if fn % stride != 0:
                    continue

                # Handle gap frames: if frame is beyond actual AR labels,
                # use 0 (NA)
                action_label = int(cache.ar_per_frame[fn]) if fn < len(cache.ar_per_frame) else 0

                # Task-aware sampling metadata
                psr_row = cache.psr_per_frame[fn] if fn < len(cache.psr_per_frame) else None
                psr_has_any = bool(psr_row.any()) if psr_row is not None else False
                num_dets = len(coco_data.get(fn, []))

                all_samples.append({
                    'recording_id': recording_id,
                    'frame_num': fn,
                    'img_path': str(rgb_dir / f'{fn:06d}.jpg'),
                    'action_label': action_label,
                    'psr_has_any': psr_has_any,
                    'num_dets': num_dets,
                })

            rec_count += 1

        return all_samples

    # =====================================================================
    # Sampler
    # =====================================================================

    def get_sampler(self) -> WeightedRandomSampler:
        """
        Class-balanced WeightedRandomSampler for AR action classes.
        Uses sqrt(counts) smoothing (effective number of samples) with beta=C.CB_BETA.
        """
        counts = self.class_counts.astype(np.float64)
        # [FIX 2026-07-01 Opus consult] ACT_SAMPLER_MODE — the legacy 'cb' path uses
        # CB effective-number weighting (beta=0.99). Its per-class sampling MASS is
        # proportional to n/effective(n), which for n >> 1/(1-beta) grows ~linearly
        # with n — so the 5 head classes still get ~4-5x the mass of the tail. The
        # 'balanced' mode gives every class with >= COUNT_FLOOR frames EQUAL mass
        # (weight = 1/max(n, floor)) while classes below the floor get mass
        # proportional to their count (so 1-7 frame singletons are NOT repeated
        # ~50x/epoch and memorized). Use 'balanced' for CE runs / when the loss is
        # NOT already class-balanced; with CB-Focal (beta=0.999, 50x cap) the loss
        # already rebalances, so leave this on 'cb' to avoid double-emphasis.
        _mode = str(getattr(C, 'ACT_SAMPLER_MODE', 'cb')).lower()
        if _mode == 'balanced':
            _floor = float(getattr(C, 'ACT_SAMPLER_COUNT_FLOOR', 15.0))
            _eff = np.maximum(counts, _floor)
            class_weights = np.where(counts > 0, 1.0 / _eff, 0.0)
        else:
            beta = C.CB_BETA
            effective = np.where(
                counts > 0,
                (1.0 - np.power(beta, counts)) / (1.0 - beta),
                1.0,
            )
            class_weights = 1.0 / np.maximum(effective, 1e-8)
        class_weights /= class_weights.sum()
        sample_weights = np.array(
            [class_weights[aid] if aid >= 0 else 0.0 for aid in self.activity_ids],
            dtype=np.float64,
        )

        # Tier 3.12 — Task-aware sample weighting. When enabled, upweight frames
        # where the rarer secondary tasks (PSR, detection) have valid labels.
        # The boost is small (≤2x) and applied multiplicatively, so it never
        # overpowers the activity class balance.
        if bool(getattr(C, 'USE_TASK_AWARE_SAMPLING', False)):
            psr_boost = float(getattr(C, 'TASK_AWARE_PSR_BOOST', 1.5))
            det_boost = float(getattr(C, 'TASK_AWARE_DET_BOOST', 1.2))
            for i in range(len(sample_weights)):
                if i >= len(self.samples):
                    continue
                sample = self.samples[i]
                # PSR: boost frames where any component has a positive label
                if sample.get('psr_has_any', False):
                    sample_weights[i] *= psr_boost
                # Detection: boost frames with at least one detection target
                if sample.get('num_dets', 0) > 0:
                    sample_weights[i] *= det_boost
            # Re-normalize so weights still sum to 1
            total_w = sample_weights.sum()
            if total_w > 0:
                sample_weights = sample_weights / total_w

        # [DET GT-FRAME SAMPLING 2026-06-16] Absolute GT-frame fraction targeting.
        # This is the real fix for the detection class-imbalance death spiral.
        # Unlike the constant TASK_AWARE_DET_BOOST above (whose effect scales with
        # the base OD density), this forces the *total* sampling mass on GT-bearing
        # frames to exactly `det_frac`, so in expectation that fraction of every
        # batch carries boxes regardless of how sparse the OD labels are. Activity
        # class-balance is preserved *within* the GT and non-GT sub-populations.
        det_frac = float(getattr(C, 'DET_GT_FRAME_FRACTION', 0.0))
        if det_frac > 0.0:
            gt_mask = np.array(
                [
                    (i < len(self.samples)) and (self.samples[i].get('num_dets', 0) > 0)
                    for i in range(len(sample_weights))
                ],
                dtype=bool,
            )
            n_gt = int(gt_mask.sum())
            n_total = len(gt_mask)
            if n_gt == 0:
                logger.warning(
                    '[get_sampler] DET_GT_FRAME_FRACTION=%.2f requested but this '
                    'subset contains ZERO frames with detection boxes. The detector '
                    'CANNOT learn from it — the death spiral is upstream of the '
                    'sampler. Check OD_labels.json coverage, raise SUBSET_RATIO, or '
                    'select an OD-bearing recording subset (run diag_gt_coverage.py).',
                    det_frac,
                )
            elif 0 < n_gt < n_total:
                w_gt = sample_weights[gt_mask].sum()
                w_bg = sample_weights[~gt_mask].sum()
                if w_gt > 0 and w_bg > 0:
                    sample_weights[gt_mask] = sample_weights[gt_mask] / w_gt * det_frac
                    sample_weights[~gt_mask] = sample_weights[~gt_mask] / w_bg * (1.0 - det_frac)
                # Final renormalize (already sums to 1 by construction, but be safe)
                total_w = sample_weights.sum()
                if total_w > 0:
                    sample_weights = sample_weights / total_w
                logger.info(
                    '[get_sampler] DET_GT_FRAME_FRACTION=%.2f: %d/%d (%.2f%%) frames '
                    'carry GT boxes -> reweighted so ~%.0f%% of every batch is '
                    'GT-bearing (was ~%.2f%% under base sampler).',
                    det_frac, n_gt, n_total, 100.0 * n_gt / max(n_total, 1),
                    100.0 * det_frac, 100.0 * n_gt / max(n_total, 1),
                )

        # [DIAG 2026-07-01 Opus round-4 Q1] Effective per-class sampling rate.
        # DET_GT_FRAME_FRACTION + task-aware boosts distort the activity-balanced
        # weights: a class whose frames rarely co-occur with detection GT is pushed
        # into the (1-det_frac) non-GT pool, while a class that co-occurs with GT
        # draws from both pools. Log the realized per-class mass so the distortion
        # is visible before a 100-epoch run (answers the "confirm or log" ask).
        try:
            _no = int(getattr(C, 'NUM_ACT_OUTPUTS', C.NUM_ACT_CLASSES))
            _mass = np.zeros(_no, dtype=np.float64)
            for _i, _aid in enumerate(self.activity_ids):
                if 0 <= _aid < _no:
                    _mass[_aid] += sample_weights[_i]
            _present = _mass[_mass > 0]
            if _present.size > 0:
                _uniform = 1.0 / _present.size
                _ratio = float(_present.max() / max(_present.min(), 1e-12))
                _topk = np.argsort(_mass)[::-1][:5]
                logger.info(
                    '[get_sampler] effective per-class sampling mass: %d classes present, '
                    'max/min ratio=%.1fx (uniform would be 1.0x), max=%.4f vs uniform=%.4f. '
                    'Top-5 sampled class ids=%s. Ratio >> 1 means DET_GT/task-aware reweighting '
                    'is distorting activity balance.',
                    _present.size, _ratio, float(_present.max()), _uniform, _topk.tolist(),
                )
        except Exception as _diag_exc:  # never let diagnostics break sampler build
            logger.debug('[get_sampler] per-class mass diag skipped: %s', _diag_exc)

        return WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )


# =========================================================================
# Collate Function
# =========================================================================


def collate_fn(batch: List[Dict[str, Any]]) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Collate IndustReal samples into batched tensors.

    Returns:
        images: [B, 3, H, W]
        targets: dict with IKEA-compatible keys:
            - detection: list of {boxes, labels} per sample (for FocalLoss)
            - head_pose: [B, 9]
            - psr_labels: [B, 11]
            - hand_joints: [B, 52]
            - activity: [B]  (renamed from action_label for loss compatibility)
            - box_mask: {'rgb': [B, max_boxes]} bool mask
            - metadata: list of dicts
    """
    # Guard: empty batch returns empty tensors (avoids stack crash)
    if not batch:
        return torch.empty(0, 3, C.IMG_HEIGHT, C.IMG_WIDTH), {}
    images = torch.stack([item['images']['rgb'] for item in batch], dim=0)

    detection_list = []
    head_poses = []
    psr_labels_list = []
    hand_joints_list = []
    activity_labels = []
    box_masks_list = []

    for item in batch:
        boxes = item['gt_boxes']['rgb']        # [N, 4]
        classes = item['gt_classes']['rgb']     # [N]

        # Flatten into per-sample detection dict (IKEA pattern)
        detection_list.append({'boxes': boxes, 'labels': classes})

        head_poses.append(item['head_pose'])
        psr_labels_list.append(item['psr_labels'])
        hand_joints_list.append(item['hand_joints'])
        activity_labels.append(
            item['action_label'] if isinstance(item['action_label'], torch.Tensor)
            else torch.tensor(item['action_label'], dtype=torch.long)
        )

        # Build per-sample box mask
        n = boxes.shape[0]
        box_mask = torch.zeros(max(n, 1), dtype=torch.bool)
        if n > 0:
            box_mask[:n] = True
        box_masks_list.append(box_mask)

    # Determine max_boxes for padding
    max_boxes = max(b.shape[0] for b in [item['gt_boxes']['rgb'] for item in batch]) if batch else 0
    max_boxes = max(max_boxes, 1)

    # Pad boxes/classes across batch
    stacked_boxes = torch.zeros(len(batch), max_boxes, 4, dtype=torch.float32)
    stacked_classes = torch.zeros(len(batch), max_boxes, dtype=torch.long)
    stacked_box_mask = torch.zeros(len(batch), max_boxes, dtype=torch.bool)

    for i, item in enumerate(batch):
        boxes = item['gt_boxes']['rgb']
        classes = item['gt_classes']['rgb']
        n = boxes.shape[0]
        if n > 0:
            stacked_boxes[i, :n] = boxes
            stacked_classes[i, :n] = classes
            stacked_box_mask[i, :n] = True

    # VideoMAE clip_rgb: [B, T, 3, 224, 224] — only present when USE_VIDEOMAE=True (Doc 2 A.1)
    # IndustReal has 52-D hand joints (26 pts × 2 coords), not 17-keypoint COCO pose.
    # hand_joints comes as flat [52] per sample → reshape to [26, 2]
    n_joints_per_hand = 26  # IndustReal format: 26 hand keypoints × 2 coords = 52
    n_select = 17
    step = n_joints_per_hand // n_select  # 1
    selected_indices = list(range(0, n_joints_per_hand, step))[:n_select]  # [0, 1, 2, ..., 16] = 17 joints
    keypoints = torch.stack([
        h.view(n_joints_per_hand, 2)[selected_indices] for h in hand_joints_list
    ], dim=0)  # [B, 17, 2]
    # [FIX 2026-06-18] Normalize hand keypoints from pixel coords to [0, 1].
    # hands.csv stores MediaPipe hand landmarks in 1280×720 pixel space. The Wing
    # loss expects both predictions and targets in the same [0, 1] normalized space.
    # Without this, the raw Wing loss is ~267 vs ~0.2 after normalization — the
    # coordinate scale mismatch makes the loss contribution meaningless.
    keypoints = keypoints / torch.tensor(
        [C.IMG_WIDTH, C.IMG_HEIGHT], dtype=keypoints.dtype
    )
    pose_confidence = torch.ones_like(keypoints[:, :, 0])  # [B, 17]

    activity_stacked = torch.stack(activity_labels, dim=0)
    activity_mask = (activity_stacked >= 0).float()  # 1.0 for labeled, 0.0 for unlabeled (-1 sentinel)

    targets: Dict[str, Any] = {
        'detection': detection_list,
        'box_mask': {'rgb': stacked_box_mask},
        'head_pose': torch.stack(head_poses, dim=0),
        'psr_labels': torch.stack(psr_labels_list, dim=0),
        'hand_joints': torch.stack(hand_joints_list, dim=0),
        'activity': activity_stacked,
        'activity_mask': activity_mask,
        'metadata': [item['metadata'] for item in batch],
        'keypoints': keypoints,
        'pose_confidence': pose_confidence,
    }

    clip_rgb_items = [item.get('clip_rgb') for item in batch]
    if any(c is not None for c in clip_rgb_items):
        targets['clip_rgb'] = torch.stack(clip_rgb_items, dim=0)  # [B, T, 3, 224, 224]

    return images, targets


def collate_fn_sequences(
    batch: List[Dict[str, Any]],
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Doc 01 §D.1: Sequence-mode collate for PSR training.

    Each item in the batch has already been assembled into a T-frame sequence by
    _getitem_sequence(): images['rgb'] = [T, 3, H, W], psr_labels = [T, 11], etc.

    This collate stacks them into [B, T, ...] tensors. Sequence length is uniform
    within a batch (all items come from the same DataLoader with fixed T), so
    no padding is needed.

    Returns:
        images: [B, T, 3, H, W] — stacked RGB sequences
        targets: dict with:
            - detection: list of {boxes, labels} for each sample (middle-frame only)
            - head_pose: [B, T, 9]
            - psr_labels: [B, T, 11]
            - sequence_lengths: [B] — all equal to T (for completeness)
            - hand_joints: [B, T, 52]
            - activity: [B] — majority-vote action label
            - metadata: list of metadata dicts
    """
    if not batch:
        return torch.empty(0), {}

    B = len(batch)

    images = torch.stack([item['images']['rgb'] for item in batch], dim=0)
    psr_labels = torch.stack([item['psr_labels'] for item in batch], dim=0)
    head_pose = torch.stack([item['head_pose'] for item in batch], dim=0)
    hand_joints = torch.stack([item['hand_joints'] for item in batch], dim=0)

    sequence_lengths = torch.tensor(
        [item['metadata'].get('sequence_length', psr_labels.shape[1]) for item in batch],
        dtype=torch.long
    )

    activity_labels = torch.tensor(
        [
            item['action_label'].item() if isinstance(item['action_label'], torch.Tensor)
            else int(item['action_label'])
            for item in batch
        ],
        dtype=torch.long
    )

    metadata_list = [item['metadata'] for item in batch]

    detection_list = []
    max_boxes = 0
    for item in batch:
        boxes = item['gt_boxes']['rgb']
        detection_list.append({'boxes': boxes, 'labels': item['gt_classes']['rgb']})
        if boxes.shape[0] > max_boxes:
            max_boxes = boxes.shape[0]

    if max_boxes == 0:
        max_boxes = 1
    stacked_boxes = torch.zeros(B, max_boxes, 4, dtype=torch.float32)
    stacked_classes = torch.zeros(B, max_boxes, dtype=torch.long)
    stacked_box_mask = torch.zeros(B, max_boxes, dtype=torch.bool)

    for i, item in enumerate(batch):
        boxes = item['gt_boxes']['rgb']
        classes = item['gt_classes']['rgb']
        n = boxes.shape[0]
        if n > 0:
            stacked_boxes[i, :n] = boxes
            stacked_classes[i, :n] = classes
            stacked_box_mask[i, :n] = True

    targets: Dict[str, Any] = {
        'detection': detection_list,
        'box_mask': {'rgb': stacked_box_mask},
        'head_pose': head_pose,
        'psr_labels': psr_labels,
        'sequence_lengths': sequence_lengths,
        'hand_joints': hand_joints,
        'activity': activity_labels,
        'activity_mask': (activity_labels >= 0).float(),  # [OPUS v5] was missing from sequence collate
        'metadata': metadata_list,
    }

    return images, targets
