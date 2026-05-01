import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gc
import json
import logging
import random
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

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

C = cast(Any, C)

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

CORRUPTED_VIDEOS = {
    "Kallax_Shelf_Drawer/0040_black_floor_09_04_2019_08_28_13_21",
    "Lack_TV_Bench/0041_white_floor_09_04_2019_08_28_12_51",
    "Lack_Side_Table/0005_white_table_10_04_2019_08_28_14_40",
    "Lack_Side_Table/0033_oak_floor_06_03_2019_08_21_11_06",
}

# =========================================================================
# COCO Cache (process-level)
# =========================================================================
_PROC_COCO_CACHE: Dict[str, Dict[int, List[Dict[str, Any]]]] = {}
_PROC_COCO_CACHE_MAX = max(1, min(int(getattr(C, 'COCO_CACHE_SIZE', 8)), 8))

_S_VIDEO_KEY = 2
_S_FRAME_NUM = 3

def _parse_coco_file(coco_path: str) -> Dict[int, List[Dict[str, Any]]]:
    """Parse COCO JSON and return dict mapping image_id → image_info."""
    if not Path(coco_path).exists():
        return {}
    try:
        with open(coco_path, 'r', encoding='utf-8', errors='ignore') as f:
            coco = json.load(f)
    except Exception:
        return {}

    id_to_frame: Dict[int, int] = {}
    for img_info in coco.get('images', []):
        stem = Path(img_info['file_name']).stem
        try:
            frame_num = int(stem)
            id_to_frame[img_info['id']] = frame_num
        except (ValueError, KeyError):
            pass

    frame_to_annots: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for ann in coco.get('annotations', []):
        frame_num = id_to_frame.get(ann['image_id'])
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
# Temporal Action Localization — GT Segments
# =========================================================================

GT_SEGMENTS_FILE = Path(
    '/media/newadmin/master/POPW/IKEA_RAW/annotations/gt_segments.json'
)

ACTION_LABEL_TO_IDX = {
    'NA': 0,
    'pick up leg': 1,
    'align leg screw with table thread': 2,
    'spin leg': 3,
    'tighten leg': 4,
    'flip table': 5,
    'flip table top': 6,
    'pick up shelf': 7,
    'attach shelf to table': 8,
    'rotate table': 9,
    'lay down leg': 10,
    'lay down shelf': 11,
    'pick up front panel': 12,
    'attach front panel': 13,
    'pick up back panel': 14,
    'attach back panel': 15,
    'pick up side panel': 16,
    'attach side panel': 17,
    'align side panel holes with front panel dowels': 18,
    'pick up bottom panel': 19,
    'slide bottom of drawer': 20,
    'pick up drawer': 21,
    'insert drawer': 22,
    'pick up pin': 23,
    'insert drawer pin': 24,
    'position the drawer right side up': 25,
    'push table': 26,
    'push table top': 27,
    'other': 28,
    'flip shelf': 29,
    'attach drawer back panel': 30,
    'attach drawer side panel': 31,
    'lay down back panel': 32,
    'lay down bottom panel': 33,
    'lay down front panel': 34,
    'lay down table top': 35,
}

_IDX_TO_ACTION_LABEL = {v: k for k, v in ACTION_LABEL_TO_IDX.items()}


def _load_gt_segments() -> Dict[str, Dict]:
    """Load GT temporal action segments from gt_segments.json (ActivityNet format)."""
    if not GT_SEGMENTS_FILE.exists():
        logger.warning(
            f'[ikea_dataset] gt_segments.json not found at {GT_SEGMENTS_FILE}. '
            f'Temporal localization metrics will be unavailable.'
        )
        return {}
    with open(GT_SEGMENTS_FILE, 'r') as f:
        return json.load(f)


class IKEAMultiTaskDataset(Dataset):
    """
    Multi-camera IKEA dataset loader.
    Supports:
      - DETECTION_MODE='all_cameras': triplets (dev1, dev2, dev3) with separate detection heads
      - DETECTION_MODE='dev3_only': all 3 cameras for pose/activity, dev3 only for detection
      - DATASET_MODE='manual_only': only 1% manually annotated frames
      - DATASET_MODE='manual_pseudo': 1% manual + 99% pseudo-GT frames
    """

    def __init__(
        self,
        split: str = 'train',
        img_size: Tuple[int, int] = C.IMG_SIZE,
        augment: bool = False,
        seed: int = C.SEED,
        max_videos: Optional[int] = None,
        split_file: Optional[Path] = None,
    ):
        """
        Args:
            split: 'train', 'val', or 'test'
            max_videos: max videos to load (for debugging)
            split_file: optional Path to a .txt split file (one video_key per line).
                        If provided, overrides the default C.TRAIN_SPLIT_FILE / C.TEST_SPLIT_FILE
                        logic. Enables evaluation on PTMA-style CV/CSV/CS splits stored in
                        splits/cv_test.txt, splits/cs_test.txt, splits/csv_test.txt.
        """
        self.split = split
        self.img_size = img_size
        self.augment = augment
        self.seed = seed
        self.max_videos = max_videos
        self.split_file = split_file

        # Dataset roots
        self.images_root = C.IMAGES_ROOT
        self.anno_root = C.IKEA_RAW_ROOT / 'annotations'
        self.coco_raw_root = C.COCO_RAW_ROOT

        # Load action lookup
        self.action_lookup = self._load_action_lookup()

        # Scan and index
        self._on_disk_videos, all_records = self._scan_and_index()

        # Split
        split_vids = self._load_official_splits(split, seed=self.seed,
                                              split_file=self.split_file)
        self.samples = [r for r in all_records if r[2] in split_vids]

        # Apply frame stride
        self._apply_frame_stride()

        # Cache activity IDs for sampling
        self.activity_ids = np.array(
            [r[7] for r in self.samples], dtype=np.int32
        )
        self.class_counts = np.bincount(
            self.activity_ids, minlength=C.NUM_ACT_CLASSES
        )

        logger.info(
            f'[ikea_dataset] Loaded {len(self.samples)} frames '
            f'(split={split}, detection_mode={C.DETECTION_MODE}, '
            f'dataset_mode={C.DATASET_MODE})'
        )

    def clear_coco_cache(self) -> None:
        _PROC_COCO_CACHE.clear()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return multi-camera sample."""
        sample = self.samples[idx]
        furniture, video_id, video_key, frame_num = (
            sample[0], sample[1], sample[2], sample[3]
        )

        # Load RGB images from all 3 cameras
        images_dict = self._load_images(furniture, video_id, frame_num)

        # Load detection annotations (per camera, or None for dev3_only)
        gt_boxes_dict, gt_classes_dict = self._load_detection_annotations(
            furniture, video_id, frame_num
        )

        # Load pose keypoints (all 3 cameras)
        gt_keypoints_dict = self._load_pose_keypoints(
            furniture, video_id, frame_num
        )

        # Apply spatial augmentation during training
        if self.augment and C.USE_SPATIAL_AUG:
            for camera in C.CAMERAS:
                img = images_dict[camera]
                boxes = gt_boxes_dict.get(camera)
                keypoints = gt_keypoints_dict[camera]

                if boxes is not None and boxes.shape[0] > 0:
                    aug_img, aug_boxes, aug_kpts = apply_spatial_aug(
                        img, boxes, keypoints
                    )
                else:
                    aug_img, aug_boxes, aug_kpts = apply_spatial_aug(
                        img,
                        torch.zeros((0, 4), dtype=torch.float32),
                        keypoints,
                    )
                    # Restore None for dev3_only mode where boxes were None
                    if C.DETECTION_MODE == 'dev3_only' and camera != 'dev3':
                        aug_boxes = None

                images_dict[camera] = aug_img
                gt_boxes_dict[camera] = aug_boxes
                gt_keypoints_dict[camera] = aug_kpts

        # Activity label (shared)
        action_label = torch.tensor(sample[7], dtype=torch.long)

        return {
            'images': images_dict,  # {'dev1': tensor, 'dev2': tensor, 'dev3': tensor}
            'gt_boxes': gt_boxes_dict,  # per-camera or None
            'gt_classes': gt_classes_dict,  # per-camera or None
            'gt_keypoints': gt_keypoints_dict,  # all cameras
            'action_label': action_label,
            'metadata': {
                'furniture': furniture,
                'video_id': video_id,
                'video_key': video_key,
                'frame_num': frame_num,
            }
        }

    # =====================================================================
    # Image Loading
    # =====================================================================

    def _load_images(self, furniture: str, video_id: str, frame_num: int) -> Dict[str, torch.Tensor]:
        """Load RGB images from all 3 cameras."""
        images_dict = {}
        dev3_img_path = self.images_root / furniture / video_id / 'dev3' / 'images' / f'{frame_num:06d}.jpg'

        for camera in C.CAMERAS:
            img_path = self.images_root / furniture / video_id / camera / 'images' / f'{frame_num:06d}.jpg'

            # In fallback mode, if dev1/dev2 frame is missing, reuse dev3 frame.
            # This keeps multiview tensor shapes consistent for pose/activity heads.
            if (
                camera in ('dev1', 'dev2')
                and C.DETECTION_MODE == 'dev3_only'
                and not img_path.exists()
                and dev3_img_path.exists()
            ):
                img_path = dev3_img_path
            if (not img_path.exists()) and C.DETECTION_MODE == 'dev3_only' and camera in ('dev1', 'dev2'):
                img_path = self.images_root / furniture / video_id / 'dev3' / 'images' / f'{frame_num:06d}.jpg'

            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize((self.img_size[0], self.img_size[1]), _BILINEAR)
                img = np.array(img, dtype=np.uint8)  # Keep as uint8 for efficiency
            except Exception as e:
                logger.warning(f'Failed to load {img_path}: {e}. Using blank image.')
                img = np.zeros((self.img_size[1], self.img_size[0], 3), dtype=np.uint8)

            images_dict[camera] = torch.from_numpy(img).permute(2, 0, 1)  # (3, H, W)

        return images_dict

    # =====================================================================
    # Detection Annotation Loading
    # =====================================================================

    def _load_detection_annotations(
        self, furniture: str, video_id: str, frame_num: int
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Load bounding boxes and class labels for each camera.
        Returns None for cameras without annotations (dev3_only mode, or missing data).
        """
        gt_boxes_dict = {}
        gt_classes_dict = {}

        if C.DETECTION_MODE == 'all_cameras':
            cameras = C.CAMERAS  # all 3
        else:  # dev3_only
            cameras = ['dev3']

        for camera in cameras:
            boxes, classes = self._extract_boxes_from_coco(
                furniture, video_id, camera, frame_num
            )
            gt_boxes_dict[camera] = boxes
            gt_classes_dict[camera] = classes

        # For dev3_only, set dev1/dev2 to None explicitly
        if C.DETECTION_MODE == 'dev3_only':
            for camera in ['dev1', 'dev2']:
                gt_boxes_dict[camera] = None
                gt_classes_dict[camera] = None

        return gt_boxes_dict, gt_classes_dict

    def _extract_boxes_from_coco(
        self, furniture: str, video_id: str, camera: str, frame_num: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract bboxes and classes from COCO JSON for a single camera/frame."""
        coco_file = self._get_coco_path(furniture, video_id, camera)

        if not Path(coco_file).exists():
            return torch.zeros((0, 4), dtype=torch.float32), torch.zeros(0, dtype=torch.long)

        try:
            frame_to_annots = _get_coco(coco_file)
            annots = frame_to_annots.get(frame_num, [])
        except Exception:
            return torch.zeros((0, 4), dtype=torch.float32), torch.zeros(0, dtype=torch.long)

        if not annots:
            return torch.zeros((0, 4), dtype=torch.float32), torch.zeros(0, dtype=torch.long)

        boxes = []
        classes = []
        for ann in annots:
            bbox = ann.get('bbox', [])
            if len(bbox) == 4:
                x, y, w, h = bbox
                boxes.append([x, y, x + w, y + h])  # Convert to xyxy format
                classes.append(ann.get('category_id', 0))

        if not boxes:
            return torch.zeros((0, 4), dtype=torch.float32), torch.zeros(0, dtype=torch.long)

        boxes_tensor = torch.from_numpy(np.array(boxes, dtype=np.float32))
        classes_tensor = torch.from_numpy(np.array(classes, dtype=np.int64))

        return boxes_tensor, classes_tensor

    def _get_coco_path(self, furniture: str, video_id: str, camera: str) -> str:
        """Construct path to COCO JSON file for a camera.

        COCO_RAW_ROOT = .../Final_Annotations_Segmentation_Tracking (parent dir).
        anno_split is always appended so train -> .../train/..., test -> .../test/...
        """
        anno_split = 'test' if self.split == 'test' else 'train'

        if self.split == 'test':
            coco_name = 'all_gt_coco_format.json'
        elif C.DATASET_MODE == 'manual_only':
            coco_name = 'manual_coco_format.json'
        else:
            coco_name = 'all_gt_coco_format.json'

        return str(
            self.coco_raw_root / anno_split / furniture / video_id / camera / coco_name
        )

    # =====================================================================
    # Pose Keypoint Loading
    # =====================================================================

    def _load_pose_keypoints(
        self, furniture: str, video_id: str, frame_num: int
    ) -> Dict[str, torch.Tensor]:
        """Load 2D pose keypoints from OpenPose predictions (all 3 cameras)."""
        keypoints_dict = {}

        for camera in C.CAMERAS:
            kpts = self._extract_keypoints(furniture, video_id, camera, frame_num)
            keypoints_dict[camera] = kpts

        return keypoints_dict

    def _extract_keypoints(
        self, furniture: str, video_id: str, camera: str, frame_num: int
    ) -> torch.Tensor:
        """
        Extract keypoints from OpenPose JSON.
        Returns: (17, 3) tensor [x, y, confidence] for each keypoint.
        """
        pose_file = (
            self.anno_root / furniture / video_id / camera
            / 'predictions' / 'pose2d' / 'openpose'
            / f'{frame_num:06d}_keypoints.json'
        )

        default_kpts = torch.zeros((C.NUM_KEYPOINTS, 3), dtype=torch.float32)

        if not pose_file.exists():
            return default_kpts

        try:
            with open(pose_file, 'r') as f:
                pose_data = json.load(f)
        except Exception:
            return default_kpts

        people = pose_data.get('people', [])
        if not people:
            return default_kpts

        # Use first person (closest to center, typically)
        person = people[0]
        keypoints = person.get('pose_keypoints_2d', [])

        if len(keypoints) < C.NUM_KEYPOINTS * 3:
            return default_kpts

        kpts_array = np.array(keypoints, dtype=np.float32).reshape(C.NUM_KEYPOINTS, 3)
        kpts_tensor = torch.from_numpy(kpts_array)

        return kpts_tensor

    # =====================================================================
    # Dataset Indexing
    # =====================================================================

    def _load_action_lookup(self) -> Dict[str, List[int]]:
        """Load action labels from JSON file."""
        if not C.ACTION_LOOKUP_FILE.exists():
            logger.warning(f'Action lookup file not found: {C.ACTION_LOOKUP_FILE}')
            return {}

        try:
            with open(C.ACTION_LOOKUP_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f'Failed to load action lookup: {e}')
            return {}

    def _scan_and_index(self) -> Tuple[set, List[Tuple]]:
        """
        Scan image directories and build sample index.
        Returns:
            on_disk_videos: set of video IDs found on disk
            all_records: list of (furniture, video_id, video_key, frame_num, 
                                  img_dir_dev3, coco_file_dev3, pose_dir_dev3, activity_id)
        """
        all_records = []
        on_disk_videos = set()

        if not self.images_root.exists():
            logger.warning(f'[ikea_dataset] images root not found: {self.images_root}')
            return on_disk_videos, all_records

        for furniture_dir in sorted(self.images_root.iterdir()):
            if not furniture_dir.is_dir():
                continue
            furniture = furniture_dir.name

            vids_for_this_furn = 0

            for video_dir in sorted(furniture_dir.iterdir()):
                if not video_dir.is_dir():
                    continue
                video_id = video_dir.name

                if C.DEBUG_MODE and vids_for_this_furn >= C.DEBUG_MAX_VIDEOS:
                    break

                # Must have dev3 images
                img_dir_dev3 = video_dir / 'dev3' / 'images'
                if not img_dir_dev3.exists():
                    continue

                video_key = f'{furniture}/{video_id}'
                if video_key in CORRUPTED_VIDEOS:
                    continue
                if video_key not in self.action_lookup:
                    continue

                action_labels = self.action_lookup[video_key]

                # Get frame numbers from dev3 images
                frame_nums = sorted(
                    int(entry.name.split('.')[0])
                    for entry in img_dir_dev3.iterdir()
                    if entry.name.endswith('.jpg') and entry.name.split('.')[0].isdigit()
                )
                if not frame_nums:
                    continue

                # Get COCO paths
                coco_file_dev3 = self._get_coco_path(furniture, video_id, 'dev3')

                # Get pose dir for keypoint discovery (we check all cameras)
                pose_dir_dev3 = str(
                    self.anno_root / furniture / video_id / 'dev3'
                    / 'predictions' / 'pose2d' / 'openpose'
                )

                on_disk_videos.add(video_key)
                vids_for_this_furn += 1

                # Create records for each frame
                for fn in frame_nums:
                    act_id = (
                        int(action_labels[fn])
                        if fn < len(action_labels) else 0
                    )
                    act_id = max(0, min(act_id, C.NUM_ACT_CLASSES - 1))

                    all_records.append((
                        furniture, video_id, video_key, fn,
                        str(img_dir_dev3 / f'{fn:06d}.jpg'),
                        coco_file_dev3, pose_dir_dev3, act_id,
                    ))

        return on_disk_videos, all_records

    def _apply_frame_stride(self):
        """Apply frame striding (skip frames) based on split.
        
        Training: random frame stride between TRAIN_FRAME_STRIDE_RANGE per video
        Validation/Evaluation: fixed stride of 1 (uniform sampling via EVAL_FRAME_STRIDE=1)
        """
        if self.split == 'train' and C.TRAIN_FRAME_STRIDE_RANGE:
            # Random stride per video (selected once per video at initialization)
            min_stride, max_stride = C.TRAIN_FRAME_STRIDE_RANGE
            rng = random.Random(self.seed + hash((self.split, 'stride')))

            by_vid: Dict[str, List] = defaultdict(list)
            for r in self.samples:
                by_vid[r[2]].append(r)

            strided = []
            for vk in sorted(by_vid):
                # Random stride for this video (consistent within this dataset init)
                stride = rng.randint(min_stride, max_stride)
                vframes = sorted(by_vid[vk], key=lambda x: x[3])  # Sort by frame_num
                strided.extend(vframes[::stride])

            self.samples = strided
        else:
            # Validation/evaluation: fixed stride
            stride = C.DEBUG_FRAME_STRIDE if C.DEBUG_MODE else C.EVAL_FRAME_STRIDE

            if stride > 1:
                by_vid: Dict[str, List] = defaultdict(list)
                for r in self.samples:
                    by_vid[r[2]].append(r)

                strided = []
                for vk in sorted(by_vid):
                    vframes = sorted(by_vid[vk], key=lambda x: x[3])  # Sort by frame_num
                    strided.extend(vframes[::stride])

                self.samples = strided

    def _load_official_splits(self, split: str, seed: int,
                              split_file: Optional[Path] = None) -> set:
        """Load official train/val/test splits.

        Args:
            split: 'train', 'val', or 'test'
            seed: random seed for train/val split
            split_file: optional override for the split file path.
                        Use for PTMA cv/cs/csv split evaluation.
        """
        if split_file is not None:
            with open(split_file) as f:
                return {ln.strip() for ln in f if ln.strip()} & self._on_disk_videos

        def _read(path: Path) -> set:
            with open(path) as f:
                return {ln.strip() for ln in f if ln.strip()}

        official_train = _read(C.TRAIN_SPLIT_FILE)
        official_test = _read(C.TEST_SPLIT_FILE)

        if split == 'test':
            return official_test & self._on_disk_videos

        # Split official_train into train/val
        rng = random.Random(seed)
        by_furniture: Dict[str, List[str]] = defaultdict(list)
        for vk in official_train:
            by_furniture[vk.split('/')[0]].append(vk)

        train_vids, val_vids = set(), set()
        for furn in sorted(by_furniture.keys()):
            vids = sorted(by_furniture[furn])
            rng.shuffle(vids)
            n_val = max(1, int(len(vids) * C.VAL_RATIO))
            val_vids.update(vids[:n_val])
            train_vids.update(vids[n_val:])

        chosen = {'train': train_vids, 'val': val_vids}[split]
        return chosen & self._on_disk_videos

    def get_sampler(self) -> WeightedRandomSampler:
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


def collate_fn(batch):
    """
    Convert multi-camera sample dicts into legacy training format.

    Strategy:
      - Flatten camera views into batch dimension.
      - Keep per-view detection/pose targets.
      - Repeat activity label per camera view.
    """
    items = list(batch)
    flat_images = []
    detection = []
    keypoints_list = []
    activity_list = []

    for item in items:
        images = item['images']
        gt_boxes = item['gt_boxes']
        gt_classes = item['gt_classes']
        gt_keypoints = item['gt_keypoints']
        action_label = item['action_label']

        for camera in C.CAMERAS:
            flat_images.append(images[camera])

            boxes = gt_boxes.get(camera)
            labels = gt_classes.get(camera)
            if boxes is None:
                boxes = torch.zeros((0, 4), dtype=torch.float32)
            if labels is None:
                labels = torch.zeros((0,), dtype=torch.long)

            detection.append({'boxes': boxes, 'labels': labels})
            keypoints_list.append(gt_keypoints[camera])
            activity_list.append(action_label)

    while True:
        try:
            images_tensor = torch.stack(flat_images, dim=0)
            break
        except RuntimeError as exc:
            msg = str(exc)
            is_cpu_oom = (
                'DefaultCPUAllocator' in msg
                or "can't allocate memory" in msg
                or 'Error code 12' in msg
            )
            if not is_cpu_oom or len(flat_images) <= 1:
                raise
            new_len = max(1, len(flat_images) // 2)
            logger.warning(
                f'[collate_fn] CPU OOM while stacking batch={len(flat_images)}; '
                f'retrying with batch={new_len}'
            )
            gc.collect()
            flat_images = flat_images[:new_len]
            detection = detection[:new_len]
            keypoints_list = keypoints_list[:new_len]
            activity_list = activity_list[:new_len]

    keypoints = torch.stack(keypoints_list, dim=0)
    visibility = (keypoints[:, :, 2] > C.POSE_CONF_THRESHOLD).float()

    return images_tensor, {
        'detection': detection,
        'keypoints': keypoints[:, :, :2],
        'visibility': visibility,
        'kpt_confidence': keypoints[:, :, 2],
        'activity': torch.stack(activity_list, dim=0),
    }


# =============================================================================
# Temporal Sequence Dataset (for video sequence training)
# =============================================================================
class IKEAMultiTaskSequenceDataset(Dataset):
    """
    Temporal sequence dataset for video-based training.

    Loads T consecutive frames per sample to enable temporal action localization
    and temporal ordering tasks. Samples are anchored at the center frame.

    Args:
        base_dataset: IKEAMultiTaskDataset instance to load frames from
        sequence_len: Number of frames in each temporal sequence (default 16)
        stride: Temporal stride between consecutive frames (default 1)
        target_camera: Camera to use as primary for temporal modeling (default 'dev3')
    """

    def __init__(
        self,
        base_dataset: IKEAMultiTaskDataset,
        sequence_len: int = None,
        stride: int = 1,
        target_camera: str = 'dev3',
    ):
        self.base = base_dataset
        self.sequence_len = sequence_len or getattr(C, 'TEMPORAL_SEQUENCE_LEN', 16)
        self.stride = stride
        self.target_camera = target_camera

        self.gt_segments_db = _load_gt_segments().get('database', {})

        self._video_max_frames: Dict[str, int] = {}
        for vk in self.gt_segments_db:
            if vk not in self._video_max_frames:
                max_frame = 0
                for ann in self.gt_segments_db[vk].get('annotation', []):
                    end_frame = int(ann['segment'][1])
                    if end_frame > max_frame:
                        max_frame = end_frame
                self._video_max_frames[vk] = max_frame

        self._build_sequence_index()

    def _extract_gt_temporal_proposals(
        self,
        video_key: str,
        start_frame: int,
        end_frame: int,
    ) -> List[Dict]:
        """
        Extract GT temporal proposals that overlap with [start_frame, end_frame].

        Returns list of dicts with keys:
            - start: float (0-1 normalized)
            - end: float (0-1 normalized)
            - action_class: int

        Only proposals with non-NA labels are included.
        """
        video_annotations = self.gt_segments_db.get(video_key, {}).get('annotation', [])
        if not video_annotations:
            return []

        max_frames = self._video_max_frames.get(video_key, 1)
        if max_frames <= 0:
            max_frames = 1

        proposals = []
        for ann in video_annotations:
            seg = ann['segment']
            seg_start = float(seg[0])
            seg_end = float(seg[1])
            label = ann.get('label', 'NA')

            if label == 'NA' or seg_end < seg_start:
                continue

            overlap_start = max(seg_start, start_frame)
            overlap_end = min(seg_end, end_frame)
            if overlap_end <= overlap_start:
                continue

            label_idx = ACTION_LABEL_TO_IDX.get(label, 0)
            if label_idx == 0:
                continue

            norm_start = overlap_start / max_frames
            norm_end = overlap_end / max_frames
            norm_start = max(0.0, min(1.0, norm_start))
            norm_end = max(0.0, min(1.0, norm_end))

            if norm_end <= norm_start:
                continue

            proposals.append({
                'start': norm_start,
                'end': norm_end,
                'action_class': label_idx,
            })

        return proposals

    def _build_sequence_index(self):

        logger.info(
            f'[ikea_dataset] TemporalSequenceDataset: {len(self.sequences)} sequences '
            f'(T={self.sequence_len}, stride={self.stride})'
        )

    def _build_sequence_index(self):
        """Build index of (video_key, start_frame) tuples for temporal sequences."""
        self.sequences = []
        seen = set()

        for record in self.base.samples:
            furniture, video_id, video_key, frame_num = record[0], record[1], record[2], record[3]

            seq_key = (video_key, frame_num)
            if seq_key in seen:
                continue
            seen.add(seq_key)

            max_start = frame_num
            min_start = max(0, frame_num - (self.sequence_len - 1) * self.stride)

            if min_start > max_start:
                continue

            for start_frame in range(min_start, max_start + 1, self.stride):
                end_frame = start_frame + (self.sequence_len - 1) * self.stride
                seq_key = (video_key, start_frame)
                if seq_key in seen:
                    continue
                seen.add(seq_key)
                self.sequences.append((video_key, start_frame, end_frame))

        self.sequences = sorted(set(self.sequences))

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return temporal sequence of T frames."""
        video_key, start_frame, end_frame = self.sequences[idx]

        parts = video_key.split('/')
        furniture = parts[0]
        video_id = parts[1] if len(parts) > 1 else parts[0]

        frames = []
        target_records = []
        action_labels = []

        for offset in range(self.sequence_len):
            frame_num = start_frame + offset * self.stride

            record_idx = None
            for i, r in enumerate(self.base.samples):
                if r[2] == video_key and r[3] == frame_num:
                    record_idx = i
                    break

            if record_idx is not None:
                sample = self.base.samples[record_idx]
                item = self.base[record_idx]
                frames.append(item['images'][self.target_camera])
                action_labels.append(item['action_label'].item())
            else:
                dummy_img = torch.zeros(3, C.IMG_HEIGHT, C.IMG_WIDTH, dtype=torch.float32)
                frames.append(dummy_img)
                action_labels.append(0)

        images_seq = torch.stack(frames, dim=0)  # [T, 3, H, W]
        action_labels_seq = torch.tensor(action_labels, dtype=torch.long)

        gt_temporal = self._extract_gt_temporal_proposals(video_key, start_frame, end_frame)

        return {
            'images_seq': images_seq,
            'action_labels_seq': action_labels_seq,
            'gt_temporal': gt_temporal,
            'metadata': {
                'video_key': video_key,
                'start_frame': start_frame,
                'end_frame': end_frame,
            },
        }


def temporal_sequence_collate_fn(batch):
    """
    Collate function for temporal sequence batches.

    Stacks sequences into [B, T, 3, H, W] tensor and handles activity labels.
    gt_temporal is collected as a list of lists (one list of proposals per sample).
    """
    images_seqs = torch.stack([item['images_seq'] for item in batch], dim=0)
    action_labels_seqs = torch.stack([item['action_labels_seq'] for item in batch], dim=0)

    gt_temporal = [item.get('gt_temporal', []) for item in batch]
    metadata = [item['metadata'] for item in batch]

    return images_seqs, {
        'action_labels_seq': action_labels_seqs,
        'gt_temporal': gt_temporal,
        'metadata': metadata,
    }
