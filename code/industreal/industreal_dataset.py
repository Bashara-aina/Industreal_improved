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
    │       ├── rgb/000000.jpg ...          (1080x720, 10 FPS)
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
from collections import defaultdict
from pathlib import Path
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

import config as C

logger = logging.getLogger(__name__)

_IMAGENET_MEAN = torch.tensor(C.IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
_IMAGENET_STD  = torch.tensor(C.IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
_RESAMPLING = getattr(Image, 'Resampling', None)
_BILINEAR = _RESAMPLING.BILINEAR if _RESAMPLING is not None else Image.BILINEAR

# =========================================================================
# COCO Cache (process-level) — one JSON per recording
# =========================================================================
_PROC_COCO_CACHE: Dict[str, Dict[int, List[Dict[str, Any]]]] = {}
_PROC_COCO_CACHE_MAX = max(1, min(getattr(C, 'COCO_CACHE_SIZE', 30), 30))


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
    """Return cached COCO data (frame_num → annotations list)."""
    if coco_path in _PROC_COCO_CACHE:
        return _PROC_COCO_CACHE[coco_path]

    result = _parse_coco_file(coco_path)

    if len(_PROC_COCO_CACHE) >= _PROC_COCO_CACHE_MAX:
        _PROC_COCO_CACHE.pop(next(iter(_PROC_COCO_CACHE)))
    _PROC_COCO_CACHE[coco_path] = result
    return result


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
        Returns: np.ndarray of shape [num_frames] with action IDs (shifted by +1 for NA=0).
        """
        ar_file = self.rec_dir / 'AR_labels.csv'
        if not ar_file.exists():
            return np.zeros(self._num_frames, dtype=np.int64)

        # Load all action spans
        spans: List[Tuple[int, int, int]] = []  # (start_frame, end_frame, action_id)
        with open(ar_file, encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 5:
                    continue
                # row[0]=recording_id, row[1]=action_class_id, row[2]=description,
                # row[3]=start_frame.jpg, row[4]=end_frame.jpg
                try:
                    action_id = int(row[1])
                    start_frame = int(Path(row[3]).stem)
                    end_frame = int(Path(row[4]).stem)
                    spans.append((start_frame, end_frame, action_id))
                except (ValueError, IndexError):
                    continue

        # Sort by start frame
        spans.sort(key=lambda x: x[0])

        # Interpolate: each frame gets the action active during that frame.
        # action_id 0 → NA/background → class index 0
        # action_id > 0 → class index = action_id + 1 (shifts IDs to leave room for NA=0)
        labels = np.zeros(self._num_frames, dtype=np.int64)
        for start, end, action_id in spans:
            end = min(end, self._num_frames - 1)
            if start < self._num_frames:
                # Shift: action_id 0 → 0 (NA), action_id > 0 → action_id + 1
                shifted = action_id  # keep original ID 0 as 0 (NA), shift others if needed
                labels[start: end + 1] = shifted

        return labels

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

        sparse_idx = 0
        for frame in range(self._num_frames):
            if sparse_idx < len(sparse) and frame == sparse[sparse_idx][0]:
                current = sparse[sparse_idx][1].copy()
                sparse_idx += 1
            dense[frame] = current

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
    ):
        """
        Args:
            split: 'train', 'val', or 'test'
            img_size: (width, height) to resize images to
            augment: apply data augmentation (color jitter, flip)
            seed: random seed for reproducibility
            max_recordings: cap number of recordings (for debugging)
        """
        self.split = split
        self.img_size = img_size
        self.augment = augment
        self.seed = seed
        self.max_recordings = max_recordings

        # Paths
        self.recordings_root = C.RECORDINGS_ROOT
        self.split_csv = {
            'train': C.TRAIN_CSV,
            'val': C.VAL_CSV,
            'test': C.TEST_CSV,
        }[split]

        # Per-recording annotation cache
        self._anno_cache: Dict[str, _PerRecordingCache] = {}

        # Scan recordings and build sample index
        self.samples = self._scan_and_index()

        # Activity IDs for class-balanced sampling
        self.activity_ids = np.array(
            [s['action_label'] for s in self.samples], dtype=np.int64
        )
        self.class_counts = np.bincount(
            self.activity_ids, minlength=C.NUM_ACT_CLASSES
        )

        logger.info(
            f'[industreal_dataset] Loaded {len(self.samples)} frames '
            f'(split={split}, recordings={max_recordings or "all"})'
        )

    def clear_coco_cache(self) -> None:
        _PROC_COCO_CACHE.clear()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        recording_id = sample['recording_id']
        frame_num = sample['frame_num']
        img_path = sample['img_path']

        # Load and resize RGB image
        rgb_tensor = self._load_image(img_path)

        # Get annotation cache
        cache = self._anno_cache[recording_id]

        # AR action label
        action_label = torch.tensor(
            sample['action_label'], dtype=torch.long
        )

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

        return {
            'images': {'rgb': rgb_tensor},
            'gt_boxes': {'rgb': gt_boxes},
            'gt_classes': {'rgb': gt_classes},
            'head_pose': head_pose,
            'psr_labels': psr_labels,
            'hand_joints': hand_joints,
            'action_label': action_label,
            'metadata': {
                'recording_id': recording_id,
                'frame_num': frame_num,
            }
        }

    # =====================================================================
    # Image Loading
    # =====================================================================

    def _load_image(self, img_path: str) -> torch.Tensor:
        """Load and resize RGB image to [3, H, W]."""
        try:
            img = Image.open(img_path).convert('RGB')
            img = img.resize((self.img_size[0], self.img_size[1]), _BILINEAR)
            img = np.array(img, dtype=np.uint8)
        except Exception as e:
            logger.warning(f'Failed to load {img_path}: {e}. Using blank image.')
            img = np.zeros((self.img_size[1], self.img_size[0], 3), dtype=np.uint8)

        return torch.from_numpy(img).permute(2, 0, 1)  # (3, H, W)

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

        boxes = []
        classes = []
        for ann in annots:
            bbox = ann.get('bbox', [])
            if len(bbox) == 4:
                x, y, w, h = bbox
                boxes.append([x, y, x + w, y + h])  # Convert to xyxy format
                classes.append(ann.get('category_id', 0))

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

        rec_count = 0
        for rec_dir in sorted(recordings_root.iterdir()):
            if not rec_dir.is_dir():
                continue
            recording_id = rec_dir.name

            if recording_id not in split_rec_ids:
                continue

            if self.max_recordings and rec_count >= self.max_recordings:
                break

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

                all_samples.append({
                    'recording_id': recording_id,
                    'frame_num': fn,
                    'img_path': str(rgb_dir / f'{fn:06d}.jpg'),
                    'action_label': action_label,
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
        beta = C.CB_BETA
        counts = self.class_counts.astype(np.float64)
        effective = np.where(
            counts > 0,
            (1.0 - np.power(beta, counts)) / (1.0 - beta),
            1.0,
        )
        class_weights = 1.0 / np.maximum(effective, 1e-8)
        class_weights /= class_weights.sum()
        sample_weights = np.array(
            [class_weights[aid] for aid in self.activity_ids], dtype=np.float64
        )
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

    return images, {
        'detection': detection_list,
        'box_mask': {'rgb': stacked_box_mask},
        'head_pose': torch.stack(head_poses, dim=0),
        'psr_labels': torch.stack(psr_labels_list, dim=0),
        'hand_joints': torch.stack(hand_joints_list, dim=0),
        'activity': torch.stack(activity_labels, dim=0),
        'metadata': [item['metadata'] for item in batch],
    }
