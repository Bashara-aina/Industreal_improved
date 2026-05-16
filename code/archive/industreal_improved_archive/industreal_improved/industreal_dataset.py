"""
IndustReal dataset loader for POPW multi-task model.
Handles: Detection (ASD), Activity Recognition, Head Pose, PSR.
"""

import json
import os
import torch
import torch.utils.data as data
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image
import cv2

from config import C


class IndustRealRecording:
    """Single recording (video) from IndustReal dataset."""

    def __init__(self, recording_path: str, split: str = "train"):
        self.path = Path(recording_path)
        self.recording_id = self.path.name
        self.split = split

        # Paths
        self.rgb_dir = self.path / "rgb"
        self.rgb_files = sorted(list(self.rgb_dir.glob("*.jpg")))

        # Load annotations
        self.od_labels = self._load_od_labels()
        self.psr_labels = self._load_psr_labels()
        self.pose_labels = self._load_pose_labels()
        self.ar_labels = self._load_ar_labels()

    def _load_od_labels(self) -> Dict:
        """Load object detection labels from OD_labels.json."""
        od_path = self.path / "OD_labels.json"
        if od_path.exists():
            with open(od_path, 'r') as f:
                return json.load(f)
        return {"images": [], "annotations": []}

    def _load_psr_labels(self) -> np.ndarray:
        """Load PSR labels from PSR_labels_raw.csv."""
        psr_path = self.path / "PSR_labels_raw.csv"
        labels = {}
        if psr_path.exists():
            with open(psr_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 12:
                        frame_name = parts[0]
                        states = [int(x) for x in parts[1:12]]
                        labels[frame_name] = states
        return labels

    def _load_pose_labels(self) -> Dict:
        """Load head pose labels from pose.csv."""
        pose_path = self.path / "pose.csv"
        labels = {}
        if pose_path.exists():
            with open(pose_path, 'r') as f:
                # Skip header
                next(f)
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 10:
                        frame_name = parts[0]
                        # forward[3], position[3], up[3]
                        forward = [float(x) for x in parts[1:4]]
                        position = [float(x) for x in parts[4:7]]
                        up = [float(x) for x in parts[7:10]]
                        labels[frame_name] = forward + position + up
        return labels

    def _load_ar_labels(self) -> Dict:
        """Load action recognition labels from AR_labels.csv.
        Format: recording_id, action_id, action_desc, start_frame, end_frame
        Build per-frame index by iterating from start to end for each action.
        """
        ar_path = self.path / "AR_labels.csv"
        frame_to_action = {}
        if ar_path.exists():
            with open(ar_path, 'r') as f:
                next(f)  # Skip header
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 5:
                        action_id = int(parts[1])
                        start_frame = int(parts[3].replace('.jpg', ''))
                        end_frame = int(parts[4].replace('.jpg', ''))
                        # Index only frames in this action's range
                        for frame_num in range(start_frame, end_frame + 1):
                            frame_key = f"{frame_num:06d}"
                            frame_to_action[frame_key] = action_id
        return frame_to_action

    def get_frame_count(self) -> int:
        return len(self.rgb_files)

    def get_rgb_path(self, frame_idx: int) -> str:
        return str(self.rgb_files[frame_idx])

    def get_annotations_for_frame(self, frame_name: str) -> List[Dict]:
        """Get bounding box annotations for a frame."""
        anns = []
        for ann in self.od_labels.get("annotations", []):
            img_info = None
            for img in self.od_labels.get("images", []):
                if img["id"] == ann["image_id"]:
                    img_info = img
                    break
            if img_info and img_info["file_name"] == frame_name:
                anns.append(ann)
        return anns

    def get_psr_for_frame(self, frame_name: str) -> Optional[List[int]]:
        """Get PSR component states for a frame."""
        return self.psr_labels.get(frame_name)

    def get_pose_for_frame(self, frame_name: str) -> Optional[List[float]]:
        """Get head pose for a frame."""
        return self.pose_labels.get(frame_name)

    def get_activity_for_frame(self, frame_name: str) -> int:
        """Get activity label for a frame. Returns action_id or 0 if not found."""
        return self.ar_labels.get(frame_name.rstrip('.jpg'), 0)


class IndustRealDataset(data.Dataset):
    """
    Full IndustReal dataset with per-frame sampling.

    Format:
    - Each sample is one frame with all task labels
    - Supports sequential sampling for PSR temporal modeling
    """

    def __init__(self, split: str = "train", transform=None, sequence_mode: bool = False,
                 sequence_length: int = 32):
        """
        Args:
            split: "train", "val", or "test"
            transform: optional augmentation transforms
            sequence_mode: return T-frame sequences instead of single frames
            sequence_length: number of frames per sequence
        """
        self.split = split
        self.transform = transform
        self.sequence_mode = sequence_mode
        self.sequence_length = sequence_length

        # Load split
        if split == "train":
            split_file = C.TRAIN_SPLIT
        elif split == "val":
            split_file = C.VAL_SPLIT
        else:
            split_file = C.TEST_SPLIT

        self._load_split(split_file)

        # Cache for recordings
        self.recording_cache = {}

        # Build per-recording frame index for sequence mode
        self.rec_slices = []  # list of (rec_id, start_idx, end_idx)
        if self.sequence_mode:
            self._build_recording_frames_index()

        # Class mappings
        self._load_class_mappings()

    def _build_recording_frames_index(self):
        """Build per-recording frame index for sequence mode.
        Returns: list of (rec_id, start_frame_idx, end_frame_idx) for each recording.
        Each recording's frames are contiguous in frame_records.
        """
        rec_slices = []
        current_rec = None
        start = 0
        for i, (rec_id, frame_idx, frame_name) in enumerate(self.frame_records):
            if rec_id != current_rec:
                if current_rec is not None:
                    rec_slices.append((current_rec, start, i))
                current_rec = rec_id
                start = i
        if current_rec is not None:
            rec_slices.append((current_rec, start, len(self.frame_records)))
        self.rec_slices = rec_slices
        print(f"[{self.split}] Built {len(self.rec_slices)} recording slices for sequence mode")

    def _load_split(self, split_file: str):
        """Load recording IDs from split CSV."""
        self.recording_ids = set()
        self.frame_records = []  # (recording_id, frame_idx, frame_name)

        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 2:
                        rec_id = parts[0]
                        self.recording_ids.add(rec_id)

                        # Also load frame info from the recording
                        rec_path = Path(C.RECORDINGS_DIR) / rec_id
                        if rec_path.exists():
                            rgb_dir = rec_path / "rgb"
                            if rgb_dir.exists():
                                rgb_files = sorted(list(rgb_dir.glob("*.jpg")))
                                for idx, rgb_file in enumerate(rgb_files):
                                    frame_name = rgb_file.name
                                    self.frame_records.append((rec_id, idx, frame_name))

        # Deduplicate recording IDs
        self.recording_ids = sorted(list(self.recording_ids))
        print(f"[{self.split}] Loaded {len(self.recording_ids)} recordings, {len(self.frame_records)} frames")

    def _load_class_mappings(self):
        """Load class name to index mappings."""
        # ASD classes (24 assembly states)
        self.asd_classes = [
            "base_plate", "short_brace", "long_brace", "pin_short", "pin_long",
            "tooth_washer", "nut", "wing_nut", "washer", "bushing",
            "axis_pin", "pulley", "spacer", "wheel", "axle",
            "corner_brace", "screw", "bolt", "shaft", "gear",
            "belt", "motor", "battery_pack", "switch"
        ]
        self.asd_class_to_idx = {c: i for i, c in enumerate(self.asd_classes)}

        # Activity classes (74 actions)
        self.activity_classes = self._load_activity_classes()

        # PSR components (11)
        self.psr_component_names = [
            "base_plate", "short_brace_assy", "long_brace_assy",
            "front_chassis", "rear_chassis", "motor_mount",
            "wheel_assy_left", "wheel_assy_right", "belt_gear_assembly",
            "battery_cover", "switch_mount"
        ]

    def _load_activity_classes(self) -> List[str]:
        """Load 74 activity classes from the dataset."""
        # Common assembly actions in IndustReal
        activities = [
            "take_short_brace", "fit_short_brace", "align_objects",
            "take_pin_short", "plug_short_pin", "take_tooth_washer",
            "fit_tooth_washer", "take_nut", "fit_nut", "tighten_nut",
            "check_instruction", "put_partial_model", "take_partial_model",
            "take_long_brace", "fit_long_brace", "take_pin_long",
            "plug_long_pin", "take_washer", "fit_washer", "take_bushing",
            "fit_bushing", "take_axis_pin", "install_axis_pin",
            "take_pulley", "fit_pulley", "take_spacer", "fit_spacer",
            "take_wheel", "install_wheel", "take_axle", "install_axle",
            "take_corner_brace", "fit_corner_brace", "take_screw",
            "fit_screw", "tighten_screw", "take_bolt", "fit_bolt",
            "tighten_bolt", "take_gear", "fit_gear", "take_belt",
            "install_belt", "take_motor", "install_motor",
            "connect_motor", "take_battery", "insert_battery",
            "close_battery_cover", "take_switch", "install_switch",
            "test_assembly", "adjust_position", "inspect_quality",
            "remove_part", "replace_part", "align_clearly",
            "check_connection", "secure_parts", "finalize_assembly",
            "idle", "look_at_instruction", "look_at_parts",
            "position_parts", "prepare_workspace", "clean_workspace"
        ]
        return activities[:74] if len(activities) >= 74 else activities

    def get_recording(self, recording_id: str) -> Optional[IndustRealRecording]:
        """Get or create recording object."""
        if recording_id not in self.recording_cache:
            rec_path = Path(C.RECORDINGS_DIR) / self.split / recording_id
            if rec_path.exists():
                self.recording_cache[recording_id] = IndustRealRecording(
                    str(rec_path), self.split
                )
            else:
                return None
        return self.recording_cache[recording_id]

    def __len__(self) -> int:
        if self.sequence_mode:
            # Number of sequences = number of recording slices
            # Each slice can provide multiple sequences of length sequence_length
            if len(self.rec_slices) == 0:
                return 0
            total_frames = sum(end - start for _, start, end in self.rec_slices)
            return total_frames // self.sequence_length
        return len(self.frame_records)

    def __getitem__(self, idx: int) -> Dict:
        if self.sequence_mode:
            return self._get_sequence(idx)
        return self._get_single_frame(idx)

    def _get_single_frame(self, idx: int) -> Dict:
        """Get single frame data."""
        rec_id, frame_idx, frame_name = self.frame_records[idx]
        rec = self.get_recording(rec_id)

        if rec is None:
            # Return dummy data
            return self._dummy_sample()

        # Load image
        img_path = rec.get_rgb_path(frame_idx)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Get detection labels
        anns = rec.get_annotations_for_frame(frame_name)
        det_labels = self._process_detection_annotations(anns)

        # Get PSR labels
        psr_labels = rec.get_psr_for_frame(frame_name)
        if psr_labels is None:
            psr_labels = [0] * 11

        # Get head pose
        pose = rec.get_pose_for_frame(frame_name)
        if pose is None:
            pose = [0.0] * 9

        # Get activity (from split file)
        act_label = self._get_activity_label(rec_id, frame_name)

        sample = {
            'image': torch.from_numpy(image).permute(2, 0, 1).float() / 255.0,
            'recording_id': rec_id,
            'frame_idx': frame_idx,
            'frame_name': frame_name,
            'det_labels': det_labels,
            'psr_labels': torch.tensor(psr_labels, dtype=torch.float32),
            'head_pose': torch.tensor(pose, dtype=torch.float32),
            'activity_label': act_label,
            'clip_rgb': image,  # For VideoMAE
        }

        return sample

    def _get_sequence(self, idx: int) -> Dict:
        """Get sequence of T frames from ONE recording (PSR training mode).

        Each __getitem__ returns one contiguous T-frame sequence from a single recording.
        Uses rec_slices to pick a recording and stays within its boundaries.
        """
        if len(self.rec_slices) == 0:
            return self._dummy_sequence()

        # Pick recording slice using idx modulo number of slices
        # Each slice may provide multiple sequences; we pick based on idx
        slice_idx = idx % len(self.rec_slices)
        rec_id, slice_start, slice_end = self.rec_slices[slice_idx]

        # Compute how many sequences this recording can provide
        slice_num_frames = slice_end - slice_start
        seqs_per_slice = max(1, slice_num_frames // self.sequence_length)

        # Position within this recording's sequences
        pos_in_slice = (idx // len(self.rec_slices)) % seqs_per_slice

        # Actual start/end within the full frame_records
        rec_start = slice_start + pos_in_slice * self.sequence_length
        rec_end = min(rec_start + self.sequence_length, slice_end)

        # If not enough frames left in this recording, wrap or return dummy
        if rec_end - rec_start < self.sequence_length:
            rec_start = max(slice_start, slice_end - self.sequence_length)
            rec_end = slice_end

        sequence_frames = self.frame_records[rec_start:rec_end]

        images = []
        all_psr_labels = []
        all_pose_labels = []
        frame_names = []

        for rec_id_seq, frame_idx_seq, frame_name in sequence_frames:
            rec = self.get_recording(rec_id_seq)
            if rec is None:
                continue

            img_path = rec.get_rgb_path(frame_idx_seq)
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Apply transforms
            if self.transform:
                image = self.transform(image)

            images.append(image)

            psr = rec.get_psr_for_frame(frame_name)
            all_psr_labels.append(psr if psr else [0] * 11)

            pose = rec.get_pose_for_frame(frame_name)
            all_pose_labels.append(pose if pose else [0.0] * 9)

            frame_names.append(frame_name)

        if len(images) == 0:
            return self._dummy_sequence()

        # Stack images: [T, H, W, C] -> [T, C, H, W]
        images_tensor = torch.stack(
            [torch.from_numpy(img).permute(2, 0, 1).float() / 255.0 for img in images],
            dim=0
        )

        # For detection - per-frame labels but same recording
        det_labels = []
        for rec_id_seq, frame_idx_seq, frame_name in sequence_frames:
            rec = self.get_recording(rec_id_seq)
            if rec:
                anns = rec.get_annotations_for_frame(frame_name)
                det_labels.append(self._process_detection_annotations(anns))
            else:
                det_labels.append({'boxes': torch.zeros(0, 4), 'labels': torch.zeros(0, dtype=torch.long)})

        return {
            'images': images_tensor,  # [T, C, H, W]
            'recording_id': rec_id,   # single recording_id for this sequence
            'frame_names': frame_names,
            'psr_labels': torch.tensor(all_psr_labels, dtype=torch.float32),  # [T, 11]
            'head_pose': torch.tensor(all_pose_labels, dtype=torch.float32),  # [T, 9]
            'clip_rgb': images_tensor,  # [T, C, H, W] for VideoMAE
            'det_labels': det_labels,  # list of per-frame labels
        }

    def _process_detection_annotations(self, anns: List[Dict]) -> Dict:
        """Process detection annotations to training format."""
        boxes = []
        labels = []

        for ann in anns:
            bbox = ann.get("bbox", [])
            if len(bbox) >= 4:
                x, y, w, h = bbox
                # Convert to center format
                cx = x + w / 2
                cy = y + h / 2
                boxes.append([cx, cy, w, h])
                labels.append(ann.get("category_id", 0))

        if len(boxes) == 0:
            return {
                'boxes': torch.zeros(0, 4),
                'labels': torch.zeros(0, dtype=torch.long)
            }

        return {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.long)
        }

    def _get_activity_label(self, rec_id: str, frame_name: str) -> int:
        """Get activity label for a frame from AR_labels.csv."""
        rec = self.get_recording(rec_id)
        if rec is None:
            return 0
        stem = frame_name.rstrip('.jpg')
        return rec.ar_labels.get(stem, 0)

    def _dummy_sample(self) -> Dict:
        """Return dummy sample for missing recordings."""
        return {
            'image': torch.zeros(3, C.IMG_HEIGHT, C.IMG_WIDTH),
            'recording_id': 'dummy',
            'frame_idx': 0,
            'frame_name': '000000.jpg',
            'det_labels': {'boxes': torch.zeros(0, 4), 'labels': torch.zeros(0, dtype=torch.long)},
            'psr_labels': torch.zeros(11),
            'head_pose': torch.zeros(9),
            'activity_label': 0,
            'clip_rgb': torch.zeros(3, C.IMG_HEIGHT, C.IMG_WIDTH),
        }

    def _dummy_sequence(self) -> Dict:
        return {
            'images': torch.zeros(self.sequence_length, 3, C.IMG_HEIGHT, C.IMG_WIDTH),
            'recording_id': 'dummy',
            'frame_names': ['000000.jpg'] * self.sequence_length,
            'psr_labels': torch.zeros(self.sequence_length, 11),
            'head_pose': torch.zeros(self.sequence_length, 9),
            'clip_rgb': torch.zeros(self.sequence_length, 3, C.IMG_HEIGHT, C.IMG_WIDTH),
            'det_labels': [{'boxes': torch.zeros(0, 4), 'labels': torch.zeros(0, dtype=torch.long)}] * self.sequence_length,
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """Collate function for DataLoader - handles both single-frame and sequence modes.

    Single-frame mode: {'image': [B, 3, H, W], ...}
    Sequence mode: {'images': [B, T, 3, H, W], ...}
    """
    if 'images' in batch[0]:  # sequence mode
        images = torch.stack([b['images'] for b in batch], dim=0)  # [B, T, C, H, W]
        result = {
            'images': images,
            'recording_ids': [b['recording_id'] for b in batch],
            'frame_names': [b['frame_names'] for b in batch],
            'psr_labels': torch.stack([b['psr_labels'] for b in batch]),  # [B, T, 11]
            'head_pose': torch.stack([b['head_pose'] for b in batch]),     # [B, T, 9]
            'clip_rgb': images,  # [B, T, C, H, W] for VideoMAE
            'det_labels': [b['det_labels'] for b in batch],  # list of list of per-frame labels
        }
        return result
    else:  # single-frame mode
        images = torch.stack([b['image'] for b in batch])

        result = {
            'images': images,
            'recording_ids': [b['recording_id'] for b in batch],
            'frame_indices': [b['frame_idx'] for b in batch],
            'frame_names': [b['frame_name'] for b in batch],
            'psr_labels': torch.stack([b['psr_labels'] for b in batch]),
            'head_pose': torch.stack([b['head_pose'] for b in batch]),
            'activity_labels': torch.tensor([b['activity_label'] for b in batch], dtype=torch.long),
            'det_labels': [b['det_labels'] for b in batch],
            'clip_rgb': [b.get('clip_rgb', images) for b in batch],
        }

        return result


class Transforms:
    """Data augmentation transforms."""

    def __init__(self, is_train: bool = True):
        self.is_train = is_train

    def __call__(self, image: np.ndarray) -> np.ndarray:
        # Resize
        image = cv2.resize(image, (C.IMG_WIDTH, C.IMG_HEIGHT))

        # Random horizontal flip
        if self.is_train and np.random.rand() < 0.5:
            image = cv2.flip(image, 1)

        # Color jitter (simple version)
        if self.is_train:
            # Brightness
            if np.random.rand() < 0.3:
                factor = 1.0 + np.random.uniform(-0.2, 0.2)
                image = np.clip(image * factor, 0, 255).astype(np.uint8)

        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0

        # ImageNet normalization (will be done in model)
        return image


if __name__ == "__main__":
    # Test dataset loading
    print("Testing IndustReal dataset...")

    # Train
    train_ds = IndustRealDataset(split="train")
    print(f"Train dataset size: {len(train_ds)}")

    # Val
    val_ds = IndustRealDataset(split="val")
    print(f"Val dataset size: {len(val_ds)}")

    # Test
    test_ds = IndustRealDataset(split="test")
    print(f"Test dataset size: {len(test_ds)}")

    # Sample a few items
    if len(train_ds) > 0:
        sample = train_ds[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"Image shape: {sample['image'].shape}")
        print(f"PSR labels: {sample['psr_labels'].shape}")
        print(f"Head pose: {sample['head_pose'].shape}")
        print(f"Activity label: {sample['activity_label']}")